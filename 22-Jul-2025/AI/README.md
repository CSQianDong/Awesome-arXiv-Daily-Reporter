# Gemini 2.5 Pro Capable of Winning Gold at IMO 2025 

**Authors**: Yichen Huang, Lin F. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15855)  

**Abstract**: The International Mathematical Olympiad (IMO) poses uniquely challenging problems requiring deep insight, creativity, and formal reasoning. While Large Language Models (LLMs) perform well on mathematical benchmarks like AIME, they struggle with Olympiad-level tasks. We use Google's Gemini 2.5 Pro on the newly released IMO 2025 problems, avoiding data contamination. With pipeline design and prompt engineering, 5 (out of 6) problems are solved correctly (up to a caveat discussed below), highlighting the importance of finding the optimal way of using powerful models. 

---
# The Other Mind: How Language Models Exhibit Human Temporal Cognition 

**Authors**: Lingyu Li, Yang Yao, Yixu Wang, Chubo Li, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15851)  

**Abstract**: As Large Language Models (LLMs) continue to advance, they exhibit certain cognitive patterns similar to those of humans that are not directly specified in training data. This study investigates this phenomenon by focusing on temporal cognition in LLMs. Leveraging the similarity judgment task, we find that larger models spontaneously establish a subjective temporal reference point and adhere to the Weber-Fechner law, whereby the perceived distance logarithmically compresses as years recede from this reference point. To uncover the mechanisms behind this behavior, we conducted multiple analyses across neuronal, representational, and informational levels. We first identify a set of temporal-preferential neurons and find that this group exhibits minimal activation at the subjective reference point and implements a logarithmic coding scheme convergently found in biological systems. Probing representations of years reveals a hierarchical construction process, where years evolve from basic numerical values in shallow layers to abstract temporal orientation in deep layers. Finally, using pre-trained embedding models, we found that the training corpus itself possesses an inherent, non-linear temporal structure, which provides the raw material for the model's internal construction. In discussion, we propose an experientialist perspective for understanding these findings, where the LLMs' cognition is viewed as a subjective construction of the external world by its internal representational system. This nuanced perspective implies the potential emergence of alien cognitive frameworks that humans cannot intuitively predict, pointing toward a direction for AI alignment that focuses on guiding internal constructions. Our code is available at this https URL. 

---
# Hierarchical Budget Policy Optimization for Adaptive Reasoning 

**Authors**: Shangke Lyu, Linjuan Wu, Yuchen Yan, Xingyu Wu, Hao Li, Yongliang Shen, Peisheng Jiang, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15844)  

**Abstract**: Large reasoning models achieve remarkable performance through extensive chain-of-thought generation, yet exhibit significant computational inefficiency by applying uniform reasoning strategies regardless of problem complexity. We present Hierarchical Budget Policy Optimization (HBPO), a reinforcement learning framework that enables models to learn problem-specific reasoning depths without sacrificing capability. HBPO addresses the fundamental challenge of exploration space collapse in efficiency-oriented training, where penalties on long output length systematically bias models away from necessary long reasoning paths. Through hierarchical budget exploration, our approach partitions rollout samples into multiple subgroups with distinct token budgets, aiming to enable efficient resource allocation while preventing degradation of capability. We introduce differentiated reward mechanisms that create budget-aware incentives aligned with the complexity of the problem, allowing models to discover natural correspondences between task requirements and computational effort. Extensive experiments demonstrate that HBPO reduces average token usage by up to 60.6% while improving accuracy by 3.14% across four reasoning benchmarks. Unlike existing methods that impose external constraints or rely on discrete mode selection, HBPO exhibits emergent adaptive behavior where models automatically adjust reasoning depth based on problem complexity. Our results suggest that reasoning efficiency and capability are not inherently conflicting, and can be simultaneously optimized through appropriately structured hierarchical training that preserves exploration diversity. 

---
# Identifying Conditional Causal Effects in MPDAGs 

**Authors**: Sara LaPlante, Emilija Perković  

**Link**: [PDF](https://arxiv.org/pdf/2507.15842)  

**Abstract**: We consider identifying a conditional causal effect when a graph is known up to a maximally oriented partially directed acyclic graph (MPDAG). An MPDAG represents an equivalence class of graphs that is restricted by background knowledge and where all variables in the causal model are observed. We provide three results that address identification in this setting: an identification formula when the conditioning set is unaffected by treatment, a generalization of the well-known do calculus to the MPDAG setting, and an algorithm that is complete for identifying these conditional effects. 

---
# Challenges of Trustworthy Federated Learning: What's Done, Current Trends and Remaining Work 

**Authors**: Nuria Rodríguez-Barroso, Mario García-Márquez, M. Victoria Luzón, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2507.15796)  

**Abstract**: In recent years, the development of Trustworthy Artificial Intelligence (TAI) has emerged as a critical objective in the deployment of AI systems across sensitive and high-risk domains. TAI frameworks articulate a comprehensive set of ethical, legal, and technical requirements to ensure that AI technologies are aligned with human values, rights, and societal expectations. Among the various AI paradigms, Federated Learning (FL) presents a promising solution to pressing privacy concerns. However, aligning FL with the rest of the requirements of TAI presents a series of challenges, most of which arise from its inherently distributed nature. In this work, we adopt the requirements TAI as a guiding structure to systematically analyze the challenges of adapting FL to TAI. Specifically, we classify and examine the key obstacles to aligning FL with TAI, providing a detailed exploration of what has been done, the trends, and the remaining work within each of the identified challenges. 

---
# A Framework for Analyzing Abnormal Emergence in Service Ecosystems Through LLM-based Agent Intention Mining 

**Authors**: Yifan Shen, Zihan Zhao, Xiao Xue, Yuwei Guo, Qun Ma, Deyu Zhou, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15770)  

**Abstract**: With the rise of service computing, cloud computing, and IoT, service ecosystems are becoming increasingly complex. The intricate interactions among intelligent agents make abnormal emergence analysis challenging, as traditional causal methods focus on individual trajectories. Large language models offer new possibilities for Agent-Based Modeling (ABM) through Chain-of-Thought (CoT) reasoning to reveal agent intentions. However, existing approaches remain limited to microscopic and static analysis. This paper introduces a framework: Emergence Analysis based on Multi-Agent Intention (EAMI), which enables dynamic and interpretable emergence analysis. EAMI first employs a dual-perspective thought track mechanism, where an Inspector Agent and an Analysis Agent extract agent intentions under bounded and perfect rationality. Then, k-means clustering identifies phase transition points in group intentions, followed by a Intention Temporal Emergence diagram for dynamic analysis. The experiments validate EAMI in complex online-to-offline (O2O) service system and the Stanford AI Town experiment, with ablation studies confirming its effectiveness, generalizability, and efficiency. This framework provides a novel paradigm for abnormal emergence and causal analysis in service ecosystems. The code is available at this https URL. 

---
# GasAgent: A Multi-Agent Framework for Automated Gas Optimization in Smart Contracts 

**Authors**: Jingyi Zheng, Zifan Peng, Yule Liu, Junfeng Wang, Yifan Liao, Wenhan Dong, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2507.15761)  

**Abstract**: Smart contracts are trustworthy, immutable, and automatically executed programs on the blockchain. Their execution requires the Gas mechanism to ensure efficiency and fairness. However, due to non-optimal coding practices, many contracts contain Gas waste patterns that need to be optimized. Existing solutions mostly rely on manual discovery, which is inefficient, costly to maintain, and difficult to scale. Recent research uses large language models (LLMs) to explore new Gas waste patterns. However, it struggles to remain compatible with existing patterns, often produces redundant patterns, and requires manual validation/rewriting. To address this gap, we present GasAgent, the first multi-agent system for smart contract Gas optimization that combines compatibility with existing patterns and automated discovery/validation of new patterns, enabling end-to-end optimization. GasAgent consists of four specialized agents, Seeker, Innovator, Executor, and Manager, that collaborate in a closed loop to identify, validate, and apply Gas-saving improvements. Experiments on 100 verified real-world contracts demonstrate that GasAgent successfully optimizes 82 contracts, achieving an average deployment Gas savings of 9.97%. In addition, our evaluation confirms its compatibility with existing tools and validates the effectiveness of each module through ablation studies. To assess broader usability, we further evaluate 500 contracts generated by five representative LLMs across 10 categories and find that GasAgent optimizes 79.8% of them, with deployment Gas savings ranging from 4.79% to 13.93%, showing its usability as the optimization layer for LLM-assisted smart contract development. 

---
# LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization 

**Authors**: Xingyu Wu, Yuchen Yan, Shangke Lyu, Linjuan Wu, Yiwen Qiu, Yongliang Shen, Weiming Lu, Jian Shao, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15758)  

**Abstract**: Large reasoning models have achieved remarkable performance through extended chain-of-thought sequences, yet this computational freedom leads to excessive token generation even for simple problems. We present Length-Adaptive Policy Optimization (LAPO), a novel framework that transforms reasoning length control from an external constraint into an intrinsic model capability. Unlike existing approaches that impose rigid limits or rely on post-hoc interventions, LAPO enables models to internalize an understanding of appropriate reasoning depth through a two-stage reinforcement learning process. In the first stage, models learn natural reasoning patterns by discovering the statistical distribution of successful solution lengths. The second stage leverages these patterns as meta-cognitive guidance, embedding them directly within the model's reasoning context to ensure inference-time flexibility. Experiments on mathematical reasoning benchmarks demonstrate that LAPO reduces token usage by up to 40.9\% while improving accuracy by 2.3\%. Our analysis reveals that models trained with LAPO develop emergent abilities to allocate computational resources based on problem complexity, achieving efficient reasoning without sacrificing quality. 

---
# Towards physician-centered oversight of conversational diagnostic AI 

**Authors**: Elahe Vedadi, David Barrett, Natalie Harris, Ellery Wulczyn, Shashir Reddy, Roma Ruparel, Mike Schaekermann, Tim Strother, Ryutaro Tanno, Yash Sharma, Jihyeon Lee, Cían Hughes, Dylan Slack, Anil Palepu, Jan Freyberg, Khaled Saab, Valentin Liévin, Wei-Hung Weng, Tao Tu, Yun Liu, Nenad Tomasev, Kavita Kulkarni, S. Sara Mahdavi, Kelvin Guu, Joëlle Barral, Dale R. Webster, James Manyika, Avinatan Hassidim, Katherine Chou, Yossi Matias, Pushmeet Kohli, Adam Rodman, Vivek Natarajan, Alan Karthikesalingam, David Stutz  

**Link**: [PDF](https://arxiv.org/pdf/2507.15743)  

**Abstract**: Recent work has demonstrated the promise of conversational AI systems for diagnostic dialogue. However, real-world assurance of patient safety means that providing individual diagnoses and treatment plans is considered a regulated activity by licensed professionals. Furthermore, physicians commonly oversee other team members in such activities, including nurse practitioners (NPs) or physician assistants/associates (PAs). Inspired by this, we propose a framework for effective, asynchronous oversight of the Articulate Medical Intelligence Explorer (AMIE) AI system. We propose guardrailed-AMIE (g-AMIE), a multi-agent system that performs history taking within guardrails, abstaining from individualized medical advice. Afterwards, g-AMIE conveys assessments to an overseeing primary care physician (PCP) in a clinician cockpit interface. The PCP provides oversight and retains accountability of the clinical decision. This effectively decouples oversight from intake and can thus happen asynchronously. In a randomized, blinded virtual Objective Structured Clinical Examination (OSCE) of text consultations with asynchronous oversight, we compared g-AMIE to NPs/PAs or a group of PCPs under the same guardrails. Across 60 scenarios, g-AMIE outperformed both groups in performing high-quality intake, summarizing cases, and proposing diagnoses and management plans for the overseeing PCP to review. This resulted in higher quality composite decisions. PCP oversight of g-AMIE was also more time-efficient than standalone PCP consultations in prior work. While our study does not replicate existing clinical practices and likely underestimates clinicians' capabilities, our results demonstrate the promise of asynchronous oversight as a feasible paradigm for diagnostic AI systems to operate under expert human oversight for enhancing real-world care. 

---
# Agentic AI for autonomous anomaly management in complex systems 

**Authors**: Reza Vatankhah Barenji, Sina Khoshgoftar  

**Link**: [PDF](https://arxiv.org/pdf/2507.15676)  

**Abstract**: This paper explores the potential of agentic AI in autonomously detecting and responding to anomalies within complex systems, emphasizing its ability to transform traditional, human-dependent anomaly management methods. 

---
# TacticCraft: Natural Language-Driven Tactical Adaptation for StarCraft II 

**Authors**: Weiyu Ma, Jiwen Jiang, Haobo Fu, Haifeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15618)  

**Abstract**: We present an adapter-based approach for tactical conditioning of StarCraft II AI agents. Current agents, while powerful, lack the ability to adapt their strategies based on high-level tactical directives. Our method freezes a pre-trained policy network (DI-Star) and attaches lightweight adapter modules to each action head, conditioned on a tactical tensor that encodes strategic preferences. By training these adapters with KL divergence constraints, we ensure the policy maintains core competencies while exhibiting tactical variations. Experimental results show our approach successfully modulates agent behavior across tactical dimensions including aggression, expansion patterns, and technology preferences, while maintaining competitive performance. Our method enables flexible tactical control with minimal computational overhead, offering practical strategy customization for complex real-time strategy games. 

---
# Metric assessment protocol in the context of answer fluctuation on MCQ tasks 

**Authors**: Ekaterina Goliakova, Xavier Renard, Marie-Jeanne Lesot, Thibault Laugel, Christophe Marsala, Marcin Detyniecki  

**Link**: [PDF](https://arxiv.org/pdf/2507.15581)  

**Abstract**: Using multiple-choice questions (MCQs) has become a standard for assessing LLM capabilities efficiently. A variety of metrics can be employed for this task. However, previous research has not conducted a thorough assessment of them. At the same time, MCQ evaluation suffers from answer fluctuation: models produce different results given slight changes in prompts. We suggest a metric assessment protocol in which evaluation methodologies are analyzed through their connection with fluctuation rates, as well as original performance. Our results show that there is a strong link between existing metrics and the answer changing, even when computed without any additional prompt variants. A novel metric, worst accuracy, demonstrates the highest association on the protocol. 

---
# Data-Efficient Safe Policy Improvement Using Parametric Structure 

**Authors**: Kasper Engelen, Guillermo A. Pérez, Marnix Suilen  

**Link**: [PDF](https://arxiv.org/pdf/2507.15532)  

**Abstract**: Safe policy improvement (SPI) is an offline reinforcement learning problem in which a new policy that reliably outperforms the behavior policy with high confidence needs to be computed using only a dataset and the behavior policy. Markov decision processes (MDPs) are the standard formalism for modeling environments in SPI. In many applications, additional information in the form of parametric dependencies between distributions in the transition dynamics is available. We make SPI more data-efficient by leveraging these dependencies through three contributions: (1) a parametric SPI algorithm that exploits known correlations between distributions to more accurately estimate the transition dynamics using the same amount of data; (2) a preprocessing technique that prunes redundant actions from the environment through a game-based abstraction; and (3) a more advanced preprocessing technique, based on satisfiability modulo theory (SMT) solving, that can identify more actions to prune. Empirical results and an ablation study show that our techniques increase the data efficiency of SPI by multiple orders of magnitude while maintaining the same reliability guarantees. 

---
# LLM world models are mental: Output layer evidence of brittle world model use in LLM mechanical reasoning 

**Authors**: Cole Robertson, Philip Wolff  

**Link**: [PDF](https://arxiv.org/pdf/2507.15521)  

**Abstract**: Do large language models (LLMs) construct and manipulate internal world models, or do they rely solely on statistical associations represented as output layer token probabilities? We adapt cognitive science methodologies from human mental models research to test LLMs on pulley system problems using TikZ-rendered stimuli. Study 1 examines whether LLMs can estimate mechanical advantage (MA). State-of-the-art models performed marginally but significantly above chance, and their estimates correlated significantly with ground-truth MA. Significant correlations between number of pulleys and model estimates suggest that models employed a pulley counting heuristic, without necessarily simulating pulley systems to derive precise values. Study 2 tested this by probing whether LLMs represent global features crucial to MA estimation. Models evaluated a functionally connected pulley system against a fake system with randomly placed components. Without explicit cues, models identified the functional system as having greater MA with F1=0.8, suggesting LLMs could represent systems well enough to differentiate jumbled from functional systems. Study 3 built on this by asking LLMs to compare functional systems with matched systems which were connected up but which transferred no force to the weight; LLMs identified the functional system with F1=0.46, suggesting random guessing. Insofar as they may generalize, these findings are compatible with the notion that LLMs manipulate internal world models, sufficient to exploit statistical associations between pulley count and MA (Study 1), and to approximately represent system components' spatial relations (Study 2). However, they may lack the facility to reason over nuanced structural connectivity (Study 3). We conclude by advocating the utility of cognitive scientific methods to evaluate the world-modeling capacities of artificial intelligence systems. 

---
# HAMLET: Hyperadaptive Agent-based Modeling for Live Embodied Theatrics 

**Authors**: Sizhou Chen, Shufan Jiang, Chi Zhang, Xiao-Lei Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15518)  

**Abstract**: Creating an immersive and interactive theatrical experience is a long-term goal in the field of interactive narrative. The emergence of large language model (LLM) is providing a new path to achieve this goal. However, existing LLM-based drama generation methods often result in AI agents that lack initiative and cannot interact with the physical environment. Furthermore, these methods typically require detailed user input to drive the drama. These limitations reduce the interactivity and immersion of online real-time performance. To address the above challenges, we propose HAMLET, a multi-agent framework focused on drama creation and online performance. Given a simple topic, the framework generates a narrative blueprint, guiding the subsequent improvisational performance. During the online performance, each actor is given an autonomous mind. This means that actors can make independent decisions based on their own background, goals, and emotional state. In addition to conversations with other actors, their decisions can also change the state of scene props through actions such as opening a letter or picking up a weapon. The change is then broadcast to other related actors, updating what they know and care about, which in turn influences their next action. To evaluate the quality of drama performance, we designed an evaluation method to assess three primary aspects, including character performance, narrative quality, and interaction experience. The experimental evaluation shows that HAMLET can create expressive and coherent theatrical experiences. Our code, dataset and models are available at this https URL. 

---
# Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner 

**Authors**: Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Yufeng Zhong, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.15509)  

**Abstract**: Recently, inspired by OpenAI-o1/o3 and Deepseek-R1, the R1-Style method based on reinforcement learning fine-tuning has received widespread attention from the community. Previous R1-Style methods mainly focus on mathematical reasoning and code intelligence. It is of great research significance to verify their advantages on more general multimodal data. Chart is an important multimodal data type with rich information, which brings important research challenges in complex reasoning. In this work, we introduce Chart-R1, a chart-domain vision-language model with reinforcement learning fine-tuning to enable complex chart reasoning. To support Chart-R1, we first propose a novel programmatic data synthesis technology to generate high-quality step-by-step chart reasoning data covering single- and multi-subcharts, which makes up for the lack of reasoning data in the chart domain. Then we develop a two-stage training strategy: Chart-COT with step-by-step chain-of-thought supervision, and Chart-RFT with numerically sensitive reinforcement fine-tuning. Chart-COT aims to decompose complex chart reasoning tasks into fine-grained, understandable subtasks through step-by-step supervision, which lays a good foundation for improving the reasoning level of reinforcement learning. Chart-RFT utilize the typical group relative policy optimization strategy, in which a relatively soft reward is adopted for numerical response to emphasize the numerical sensitivity in the chart domain. We conduct extensive experiments on open-source benchmarks and self-built chart reasoning dataset (\emph{i.e., ChartRQA}). Experimental results show that Chart-R1 has significant advantages compared to chart-domain methods, even comparable to open/closed source large-scale models (\emph{e.g., GPT-4o, Claude-3.5}). 

---
# Optimization of Activity Batching Policies in Business Processes 

**Authors**: Orlenys López-Pintado, Jannis Rosenbaum, Marlon Dumas  

**Link**: [PDF](https://arxiv.org/pdf/2507.15457)  

**Abstract**: In business processes, activity batching refers to packing multiple activity instances for joint execution. Batching allows managers to trade off cost and processing effort against waiting time. Larger and less frequent batches may lower costs by reducing processing effort and amortizing fixed costs, but they create longer waiting times. In contrast, smaller and more frequent batches reduce waiting times but increase fixed costs and processing effort. A batching policy defines how activity instances are grouped into batches and when each batch is activated. This paper addresses the problem of discovering batching policies that strike optimal trade-offs between waiting time, processing effort, and cost. The paper proposes a Pareto optimization approach that starts from a given set (possibly empty) of activity batching policies and generates alternative policies for each batched activity via intervention heuristics. Each heuristic identifies an opportunity to improve an activity's batching policy with respect to a metric (waiting time, processing time, cost, or resource utilization) and an associated adjustment to the activity's batching policy (the intervention). The impact of each intervention is evaluated via simulation. The intervention heuristics are embedded in an optimization meta-heuristic that triggers interventions to iteratively update the Pareto front of the interventions identified so far. The paper considers three meta-heuristics: hill-climbing, simulated annealing, and reinforcement learning. An experimental evaluation compares the proposed approach based on intervention heuristics against the same (non-heuristic guided) meta-heuristics baseline regarding convergence, diversity, and cycle time gain of Pareto-optimal policies. 

---
# Predictive Process Monitoring Using Object-centric Graph Embeddings 

**Authors**: Wissam Gherissi, Mehdi Acheli, Joyce El Haddad, Daniela Grigori  

**Link**: [PDF](https://arxiv.org/pdf/2507.15411)  

**Abstract**: Object-centric predictive process monitoring explores and utilizes object-centric event logs to enhance process predictions. The main challenge lies in extracting relevant information and building effective models. In this paper, we propose an end-to-end model that predicts future process behavior, focusing on two tasks: next activity prediction and next event time. The proposed model employs a graph attention network to encode activities and their relationships, combined with an LSTM network to handle temporal dependencies. Evaluated on one reallife and three synthetic event logs, the model demonstrates competitive performance compared to state-of-the-art methods. 

---
# RAD: Retrieval High-quality Demonstrations to Enhance Decision-making 

**Authors**: Lu Guo, Yixiang Shan, Zhengbang Zhu, Qifan Liang, Lichang Song, Ting Long, Weinan Zhang, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15356)  

**Abstract**: Offline reinforcement learning (RL) enables agents to learn policies from fixed datasets, avoiding costly or unsafe environment interactions. However, its effectiveness is often limited by dataset sparsity and the lack of transition overlap between suboptimal and expert trajectories, which makes long-horizon planning particularly challenging. Prior solutions based on synthetic data augmentation or trajectory stitching often fail to generalize to novel states and rely on heuristic stitching points. To address these challenges, we propose Retrieval High-quAlity Demonstrations (RAD) for decision-making, which combines non-parametric retrieval with diffusion-based generative modeling. RAD dynamically retrieves high-return states from the offline dataset as target states based on state similarity and return estimation, and plans toward them using a condition-guided diffusion model. Such retrieval-guided generation enables flexible trajectory stitching and improves generalization when encountered with underrepresented or out-of-distribution states. Extensive experiments confirm that RAD achieves competitive or superior performance compared to baselines across diverse benchmarks, validating its effectiveness. 

---
# One Step is Enough: Multi-Agent Reinforcement Learning based on One-Step Policy Optimization for Order Dispatch on Ride-Sharing Platforms 

**Authors**: Zijian Zhao, Sen Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15351)  

**Abstract**: On-demand ride-sharing platforms face the fundamental challenge of dynamically bundling passengers with diverse origins and destinations and matching them with vehicles in real time, all under significant uncertainty. Recently, MARL has emerged as a promising solution for this problem, leveraging decentralized learning to address the curse of dimensionality caused by the large number of agents in the ride-hailing market and the resulting expansive state and action spaces. However, conventional MARL-based ride-sharing approaches heavily rely on the accurate estimation of Q-values or V-values, which becomes problematic in large-scale, highly uncertain environments. Specifically, most of these approaches adopt an independent paradigm, exacerbating this issue, as each agent treats others as part of the environment, leading to unstable training and substantial estimation bias in value functions. To address these challenges, we propose two novel alternative methods that bypass value function estimation. First, we adapt GRPO to ride-sharing, replacing the PPO baseline with the group average reward to eliminate critic estimation errors and reduce training bias. Second, inspired by GRPO's full utilization of group reward information, we customize the PPO framework for ride-sharing platforms and show that, under a homogeneous fleet, the optimal policy can be trained using only one-step rewards - a method we term One-Step Policy Optimization (OSPO). Experiments on a real-world Manhattan ride-hailing dataset demonstrate that both GRPO and OSPO achieve superior performance across most scenarios, efficiently optimizing pickup times and the number of served orders using simple MLP networks. 

---
# QSAF: A Novel Mitigation Framework for Cognitive Degradation in Agentic AI 

**Authors**: Hammad Atta, Muhammad Zeeshan Baig, Yasir Mehmood, Nadeem Shahzad, Ken Huang, Muhammad Aziz Ul Haq, Muhammad Awais, Kamal Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2507.15330)  

**Abstract**: We introduce Cognitive Degradation as a novel vulnerability class in agentic AI systems. Unlike traditional adversarial external threats such as prompt injection, these failures originate internally, arising from memory starvation, planner recursion, context flooding, and output suppression. These systemic weaknesses lead to silent agent drift, logic collapse, and persistent hallucinations over time. To address this class of failures, we introduce the Qorvex Security AI Framework for Behavioral & Cognitive Resilience (QSAF Domain 10), a lifecycle-aware defense framework defined by a six-stage cognitive degradation lifecycle. The framework includes seven runtime controls (QSAF-BC-001 to BC-007) that monitor agent subsystems in real time and trigger proactive mitigation through fallback routing, starvation detection, and memory integrity enforcement. Drawing from cognitive neuroscience, we map agentic architectures to human analogs, enabling early detection of fatigue, starvation, and role collapse. By introducing a formal lifecycle and real-time mitigation controls, this work establishes Cognitive Degradation as a critical new class of AI system vulnerability and proposes the first cross-platform defense model for resilient agentic behavior. 

---
# IM-Chat: A Multi-agent LLM-based Framework for Knowledge Transfer in Injection Molding Industry 

**Authors**: Junhyeong Lee, Joon-Young Kim, Heekyu Kim, Inhyo Lee, Seunghwa Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15268)  

**Abstract**: The injection molding industry faces critical challenges in preserving and transferring field knowledge, particularly as experienced workers retire and multilingual barriers hinder effective communication. This study introduces IM-Chat, a multi-agent framework based on large language models (LLMs), designed to facilitate knowledge transfer in injection molding. IM-Chat integrates both limited documented knowledge (e.g., troubleshooting tables, manuals) and extensive field data modeled through a data-driven process condition generator that infers optimal manufacturing settings from environmental inputs such as temperature and humidity, enabling robust and context-aware task resolution. By adopting a retrieval-augmented generation (RAG) strategy and tool-calling agents within a modular architecture, IM-Chat ensures adaptability without the need for fine-tuning. Performance was assessed across 100 single-tool and 60 hybrid tasks for GPT-4o, GPT-4o-mini, and GPT-3.5-turbo by domain experts using a 10-point rubric focused on relevance and correctness, and was further supplemented by automated evaluation using GPT-4o guided by a domain-adapted instruction prompt. The evaluation results indicate that more capable models tend to achieve higher accuracy, particularly in complex, tool-integrated scenarios. Overall, these findings demonstrate the viability of multi-agent LLM systems for industrial knowledge workflows and establish IM-Chat as a scalable and generalizable approach to AI-assisted decision support in manufacturing. 

---
# Disentangling Homophily and Heterophily in Multimodal Graph Clustering 

**Authors**: Zhaochen Guo, Zhixiang Shen, Xuanting Xie, Liangjian Wen, Zhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15253)  

**Abstract**: Multimodal graphs, which integrate unstructured heterogeneous data with structured interconnections, offer substantial real-world utility but remain insufficiently explored in unsupervised learning. In this work, we initiate the study of multimodal graph clustering, aiming to bridge this critical gap. Through empirical analysis, we observe that real-world multimodal graphs often exhibit hybrid neighborhood patterns, combining both homophilic and heterophilic relationships. To address this challenge, we propose a novel framework -- \textsc{Disentangled Multimodal Graph Clustering (DMGC)} -- which decomposes the original hybrid graph into two complementary views: (1) a homophily-enhanced graph that captures cross-modal class consistency, and (2) heterophily-aware graphs that preserve modality-specific inter-class distinctions. We introduce a \emph{Multimodal Dual-frequency Fusion} mechanism that jointly filters these disentangled graphs through a dual-pass strategy, enabling effective multimodal integration while mitigating category confusion. Our self-supervised alignment objectives further guide the learning process without requiring labels. Extensive experiments on both multimodal and multi-relational graph datasets demonstrate that DMGC achieves state-of-the-art performance, highlighting its effectiveness and generalizability across diverse settings. Our code is available at this https URL. 

---
# Explainable Artificial Intelligence based Soft Evaluation Indicator for Arc Fault Diagnosis 

**Authors**: Qianchao Wang, Yuxuan Ding, Chuanzhen Jia, Zhe Li, Yaping Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.15239)  

**Abstract**: Novel AI-based arc fault diagnosis models have demonstrated outstanding performance in terms of classification accuracy. However, an inherent problem is whether these models can actually be trusted to find arc faults. In this light, this work proposes a soft evaluation indicator that explains the outputs of arc fault diagnosis models, by defining the the correct explanation of arc faults and leveraging Explainable Artificial Intelligence and real arc fault experiments. Meanwhile, a lightweight balanced neural network is proposed to guarantee competitive accuracy and soft feature extraction score. In our experiments, several traditional machine learning methods and deep learning methods across two arc fault datasets with different sample times and noise levels are utilized to test the effectiveness of the soft evaluation indicator. Through this approach, the arc fault diagnosis models are easy to understand and trust, allowing practitioners to make informed and trustworthy decisions. 

---
# Solving Formal Math Problems by Decomposition and Iterative Reflection 

**Authors**: Yichi Zhou, Jianqiu Zhao, Yongxin Zhang, Bohan Wang, Siran Wang, Luoxin Chen, Jiahui Wang, Haowei Chen, Allan Jie, Xinbo Zhang, Haocheng Wang, Luong Trung, Rong Ye, Phan Nhat Hoang, Huishuai Zhang, Peng Sun, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15225)  

**Abstract**: General-purpose Large Language Models (LLMs) have achieved remarkable success in intelligence, performing comparably to human experts on complex reasoning tasks such as coding and mathematical reasoning. However, generating formal proofs in specialized languages like Lean 4 remains a significant challenge for these models, limiting their application in complex theorem proving and automated verification. Current approaches typically require specializing models through fine-tuning on dedicated formal corpora, incurring high costs for data collection and training. In this work, we introduce \textbf{Delta Prover}, an agent-based framework that orchestrates the interaction between a general-purpose LLM and the Lean 4 proof environment. Delta Prover leverages the reflection and reasoning capabilities of general-purpose LLMs to interactively construct formal proofs in Lean 4, circumventing the need for model specialization. At its core, the agent integrates two novel, interdependent components: an algorithmic framework for reflective decomposition and iterative proof repair, and a custom Domain-Specific Language (DSL) built upon Lean 4 for streamlined subproblem management. \textbf{Delta Prover achieves a state-of-the-art 95.9\% success rate on the miniF2F-test benchmark, surpassing all existing approaches, including those requiring model specialization.} Furthermore, Delta Prover exhibits a significantly stronger test-time scaling law compared to standard Best-of-N proof strategies. Crucially, our findings demonstrate that general-purpose LLMs, when guided by an effective agentic structure, possess substantial untapped theorem-proving capabilities. This presents a computationally efficient alternative to specialized models for robust automated reasoning in formal environments. 

---
# Can We Move Freely in NEOM's The Line? An Agent-Based Simulation of Human Mobility in a Futuristic Smart City 

**Authors**: Abderaouf Bahi, Amel Ourici  

**Link**: [PDF](https://arxiv.org/pdf/2507.15143)  

**Abstract**: This paper investigates the feasibility of human mobility in The Line, a proposed 170-kilometer linear smart city in NEOM, Saudi Arabia. To assess whether citizens can move freely within this unprecedented urban topology, we develop a hybrid simulation framework that integrates agent-based modeling, reinforcement learning, supervised learning, and graph neural networks. The simulation captures multi-modal transportation behaviors across 50 vertical levels and varying density scenarios using both synthetic data and real-world traces from high-density cities. Our experiments reveal that with the full AI-integrated architecture, agents achieved an average commute time of 7.8 to 8.4 minutes, a satisfaction rate exceeding 89 percent, and a reachability index of over 91 percent, even during peak congestion periods. Ablation studies confirmed that the removal of intelligent modules such as reinforcement learning or graph neural networks significantly degrades performance, with commute times increasing by up to 85 percent and reachability falling below 70 percent. Environmental modeling further demonstrated low energy consumption and minimal CO2 emissions when electric modes are prioritized. The findings suggest that freedom of movement is not only conceptually achievable in The Line, but also operationally realistic if supported by adaptive AI systems, sustainable infrastructure, and real-time feedback loops. 

---
# Clinical Semantic Intelligence (CSI): Emulating the Cognitive Framework of the Expert Clinician for Comprehensive Oral Disease Diagnosis 

**Authors**: Mohammad Mashayekhi, Sara Ahmadi Majd, Arian AmirAmjadi, Parsa Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2507.15140)  

**Abstract**: The diagnosis of oral diseases presents a problematic clinical challenge, characterized by a wide spectrum of pathologies with overlapping symptomatology. To address this, we developed Clinical Semantic Intelligence (CSI), a novel artificial intelligence framework that diagnoses 118 different oral diseases by computationally modeling the cognitive processes of an expert clinician. Our core hypothesis is that moving beyond simple pattern matching to emulate expert reasoning is critical to building clinically useful diagnostic aids.
CSI's architecture integrates a fine-tuned multimodal CLIP model with a specialized ChatGLM-6B language model. This system executes a Hierarchical Diagnostic Reasoning Tree (HDRT), a structured framework that distills the systematic, multi-step logic of differential diagnosis. The framework operates in two modes: a Fast Mode for rapid screening and a Standard Mode that leverages the full HDRT for an interactive and in-depth diagnostic workup.
To train and validate our system, we curated a primary dataset of 4,310 images, supplemented by an external hold-out set of 176 images for final validation. A clinically-informed augmentation strategy expanded our training data to over 30,000 image-text pairs. On a 431-image internal test set, CSI's Fast Mode achieved an accuracy of 73.4%, which increased to 89.5% with the HDRT-driven Standard Mode. The performance gain is directly attributable to the hierarchical reasoning process. Herein, we detail the architectural philosophy, development, and rigorous evaluation of the CSI framework. 

---
# Automated planning with ontologies under coherence update semantics 

**Authors**: Stefan Borgwardt, Duy Nhu, Gabriele Röger  

**Link**: [PDF](https://arxiv.org/pdf/2507.15120)  

**Abstract**: Standard automated planning employs first-order formulas under closed-world semantics to achieve a goal with a given set of actions from an initial state. We follow a line of research that aims to incorporate background knowledge into automated planning problems, for example, by means of ontologies, which are usually interpreted under open-world semantics. We present a new approach for planning with DL-Lite ontologies that combines the advantages of ontology-based action conditions provided by explicit-input knowledge and action bases (eKABs) and ontology-aware action effects under the coherence update semantics. We show that the complexity of the resulting formalism is not higher than that of previous approaches and provide an implementation via a polynomial compilation into classical planning. An evaluation of existing and new benchmarks examines the performance of a planning system on different variants of our compilation. 

---
# From Kicking to Causality: Simulating Infant Agency Detection with a Robust Intrinsic Reward 

**Authors**: Xia Xu, Jochen Triesch  

**Link**: [PDF](https://arxiv.org/pdf/2507.15106)  

**Abstract**: While human infants robustly discover their own causal efficacy, standard reinforcement learning agents remain brittle, as their reliance on correlation-based rewards fails in noisy, ecologically valid scenarios. To address this, we introduce the Causal Action Influence Score (CAIS), a novel intrinsic reward rooted in causal inference. CAIS quantifies an action's influence by measuring the 1-Wasserstein distance between the learned distribution of sensory outcomes conditional on that action, $p(h|a)$, and the baseline outcome distribution, $p(h)$. This divergence provides a robust reward that isolates the agent's causal impact from confounding environmental noise. We test our approach in a simulated infant-mobile environment where correlation-based perceptual rewards fail completely when the mobile is subjected to external forces. In stark contrast, CAIS enables the agent to filter this noise, identify its influence, and learn the correct policy. Furthermore, the high-quality predictive model learned for CAIS allows our agent, when augmented with a surprise signal, to successfully reproduce the "extinction burst" phenomenon. We conclude that explicitly inferring causality is a crucial mechanism for developing a robust sense of agency, offering a psychologically plausible framework for more adaptive autonomous systems. 

---
# DeRAG: Black-box Adversarial Attacks on Multiple Retrieval-Augmented Generation Applications via Prompt Injection 

**Authors**: Jerry Wang, Fang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15042)  

**Abstract**: Adversarial prompt attacks can significantly alter the reliability of Retrieval-Augmented Generation (RAG) systems by re-ranking them to produce incorrect outputs. In this paper, we present a novel method that applies Differential Evolution (DE) to optimize adversarial prompt suffixes for RAG-based question answering. Our approach is gradient-free, treating the RAG pipeline as a black box and evolving a population of candidate suffixes to maximize the retrieval rank of a targeted incorrect document to be closer to real world scenarios. We conducted experiments on the BEIR QA datasets to evaluate attack success at certain retrieval rank thresholds under multiple retrieving applications. Our results demonstrate that DE-based prompt optimization attains competitive (and in some cases higher) success rates compared to GGPP to dense retrievers and PRADA to sparse retrievers, while using only a small number of tokens (<=5 tokens) in the adversarial suffix. Furthermore, we introduce a readability-aware suffix construction strategy, validated by a statistically significant reduction in MLM negative log-likelihood with Welch's t-test. Through evaluations with a BERT-based adversarial suffix detector, we show that DE-generated suffixes evade detection, yielding near-chance detection accuracy. 

---
# A Forced-Choice Neural Cognitive Diagnostic Model of Personality Testing 

**Authors**: Xiaoyu Li, Jin Wu, Shaoyang Guo, Haoran Shi, Chanjin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.15013)  

**Abstract**: In the smart era, psychometric tests are becoming increasingly important for personnel selection, career development, and mental health assessment. Forced-choice tests are common in personality assessments because they require participants to select from closely related options, lowering the risk of response distortion. This study presents a deep learning-based Forced-Choice Neural Cognitive Diagnostic Model (FCNCD) that overcomes the limitations of traditional models and is applicable to the three most common item block types found in forced-choice tests. To account for the unidimensionality of items in forced-choice tests, we create interpretable participant and item parameters. We model the interactions between participant and item features using multilayer neural networks after mining them using nonlinear mapping. In addition, we use the monotonicity assumption to improve the interpretability of the diagnostic results. The FCNCD's effectiveness is validated by experiments on real-world and simulated datasets that show its accuracy, interpretability, and robustness. 

---
# AlphaAlign: Incentivizing Safety Alignment with Extremely Simplified Reinforcement Learning 

**Authors**: Yi Zhang, An Zhang, XiuYu Zhang, Leheng Sheng, Yuxin Chen, Zhenkai Liang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14987)  

**Abstract**: Large language models (LLMs), despite possessing latent safety understanding from their vast pretraining data, remain vulnerable to generating harmful content and exhibit issues such as over-refusal and utility degradation after safety alignment. Current safety alignment methods often result in superficial refusal shortcuts or rely on intensive supervision for reasoning-based approaches, failing to fully leverage the model's intrinsic safety self-awareness. We propose \textbf{AlphaAlign}, a simple yet effective pure reinforcement learning (RL) framework with verifiable safety reward designed to incentivize this latent safety awareness through proactive safety reasoning.} AlphaAlign employs a dual-reward system: a verifiable safety reward encourages correctly formatted and explicitly justified refusals for harmful queries while penalizing over-refusals, and a normalized helpfulness reward guides high-quality responses to benign inputs. This allows the model to develop proactive safety reasoning capabilities without depending on supervised safety-specific reasoning data. AlphaAlign demonstrates three key advantages: (1) Simplicity and efficiency, requiring only binary prompt safety labels and minimal RL steps for substantial improvements. (2) Breaking the safety-utility trade-off, by enhancing refusal of harmful content and reducing over-refusals, while simultaneously maintaining or even improving general task performance and robustness to unseen jailbreaks. (3) Deep alignment, fostering proactive safety reasoning that generates explicit safety rationales rather than relying on shallow refusal patterns. 

---
# Complexity of Faceted Explanations in Propositional Abduction 

**Authors**: Johannes Schmidt, Mohamed Maizia, Victor Lagerkvist, Johannes K. Fichte  

**Link**: [PDF](https://arxiv.org/pdf/2507.14962)  

**Abstract**: Abductive reasoning is a popular non-monotonic paradigm that aims to explain observed symptoms and manifestations. It has many applications, such as diagnosis and planning in artificial intelligence and database updates. In propositional abduction, we focus on specifying knowledge by a propositional formula. The computational complexity of tasks in propositional abduction has been systematically characterized - even with detailed classifications for Boolean fragments. Unsurprisingly, the most insightful reasoning problems (counting and enumeration) are computationally highly challenging. Therefore, we consider reasoning between decisions and counting, allowing us to understand explanations better while maintaining favorable complexity. We introduce facets to propositional abductions, which are literals that occur in some explanation (relevant) but not all explanations (dispensable). Reasoning with facets provides a more fine-grained understanding of variability in explanations (heterogeneous). In addition, we consider the distance between two explanations, enabling a better understanding of heterogeneity/homogeneity. We comprehensively analyze facets of propositional abduction in various settings, including an almost complete characterization in Post's framework. 

---
# Redefining Elderly Care with Agentic AI: Challenges and Opportunities 

**Authors**: Ruhul Amin Khalil, Kashif Ahmad, Hazrat Ali  

**Link**: [PDF](https://arxiv.org/pdf/2507.14912)  

**Abstract**: The global ageing population necessitates new and emerging strategies for caring for older adults. In this article, we explore the potential for transformation in elderly care through Agentic Artificial Intelligence (AI), powered by Large Language Models (LLMs). We discuss the proactive and autonomous decision-making facilitated by Agentic AI in elderly care. Personalized tracking of health, cognitive care, and environmental management, all aimed at enhancing independence and high-level living for older adults, represents important areas of application. With a potential for significant transformation of elderly care, Agentic AI also raises profound concerns about data privacy and security, decision independence, and access. We share key insights to emphasize the need for ethical safeguards, privacy protections, and transparent decision-making. Our goal in this article is to provide a balanced discussion of both the potential and the challenges associated with Agentic AI, and to provide insights into its responsible use in elderly care, to bring Agentic AI into harmony with the requirements and vulnerabilities specific to the elderly. Finally, we identify the priorities for the academic research communities, to achieve human-centered advancements and integration of Agentic AI in elderly care. To the best of our knowledge, this is no existing study that reviews the role of Agentic AI in elderly care. Hence, we address the literature gap by analyzing the unique capabilities, applications, and limitations of LLM-based Agentic AI in elderly care. We also provide a companion interactive dashboard at this https URL. 

---
# The Endless Tuning. An Artificial Intelligence Design To Avoid Human Replacement and Trace Back Responsibilities 

**Authors**: Elio Grande  

**Link**: [PDF](https://arxiv.org/pdf/2507.14909)  

**Abstract**: The Endless Tuning is a design method for a reliable deployment of artificial intelligence based on a double mirroring process, which pursues both the goals of avoiding human replacement and filling the so-called responsibility gap (Matthias 2004). Originally depicted in (Fabris et al. 2024) and ensuing the relational approach urged therein, it was then actualized in a protocol, implemented in three prototypical applications regarding decision-making processes (respectively: loan granting, pneumonia diagnosis, and art style recognition) and tested with such as many domain experts. Step by step illustrating the protocol, giving insights concretely showing a different voice (Gilligan 1993) in the ethics of artificial intelligence, a philosophical account of technical choices (e.g., a reversed and hermeneutic deployment of XAI algorithms) will be provided in the present study together with the results of the experiments, focusing on user experience rather than statistical accuracy. Even thoroughly employing deep learning models, full control was perceived by the interviewees in the decision-making setting, while it appeared that a bridge can be built between accountability and liability in case of damage. 

---
# Feedback-Induced Performance Decline in LLM-Based Decision-Making 

**Authors**: Xiao Yang, Juxi Leitner, Michael Burke  

**Link**: [PDF](https://arxiv.org/pdf/2507.14906)  

**Abstract**: The ability of Large Language Models (LLMs) to extract context from natural language problem descriptions naturally raises questions about their suitability in autonomous decision-making settings. This paper studies the behaviour of these models within a Markov Decision Process (MDPs). While traditional reinforcement learning (RL) strategies commonly employed in this setting rely on iterative exploration, LLMs, pre-trained on diverse datasets, offer the capability to leverage prior knowledge for faster adaptation. We investigate online structured prompting strategies in sequential decision making tasks, comparing the zero-shot performance of LLM-based approaches to that of classical RL methods. Our findings reveal that although LLMs demonstrate improved initial performance in simpler environments, they struggle with planning and reasoning in complex scenarios without fine-tuning or additional guidance. Our results show that feedback mechanisms, intended to improve decision-making, often introduce confusion, leading to diminished performance in intricate environments. These insights underscore the need for further exploration into hybrid strategies, fine-tuning, and advanced memory integration to enhance LLM-based decision-making capabilities. 

---
# InsightX Agent: An LMM-based Agentic Framework with Integrated Tools for Reliable X-ray NDT Analysis 

**Authors**: Jiale Liu, Huan Wang, Yue Zhang, Xiaoyu Luo, Jiaxiang Hu, Zhiliang Liu, Min Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.14899)  

**Abstract**: Non-destructive testing (NDT), particularly X-ray inspection, is vital for industrial quality assurance, yet existing deep-learning-based approaches often lack interactivity, interpretability, and the capacity for critical self-assessment, limiting their reliability and operator trust. To address these shortcomings, this paper proposes InsightX Agent, a novel LMM-based agentic framework designed to deliver reliable, interpretable, and interactive X-ray NDT analysis. Unlike typical sequential pipelines, InsightX Agent positions a Large Multimodal Model (LMM) as a central orchestrator, coordinating between the Sparse Deformable Multi-Scale Detector (SDMSD) and the Evidence-Grounded Reflection (EGR) tool. The SDMSD generates dense defect region proposals for multi-scale feature maps and sparsifies them through Non-Maximum Suppression (NMS), optimizing detection of small, dense targets in X-ray images while maintaining computational efficiency. The EGR tool guides the LMM agent through a chain-of-thought-inspired review process, incorporating context assessment, individual defect analysis, false positive elimination, confidence recalibration and quality assurance to validate and refine the SDMSD's initial proposals. By strategically employing and intelligently using tools, InsightX Agent moves beyond passive data processing to active reasoning, enhancing diagnostic reliability and providing interpretations that integrate diverse information sources. Experimental evaluations on the GDXray+ dataset demonstrate that InsightX Agent not only achieves a high object detection F1-score of 96.35% but also offers significantly improved interpretability and trustworthiness in its analyses, highlighting the transformative potential of agentic LLM frameworks for industrial inspection tasks. 

---
# AgentFly: Extensible and Scalable Reinforcement Learning for LM Agents 

**Authors**: Renxi Wang, Rifo Ahmad Genadi, Bilal El Bouardi, Yongxin Wang, Fajri Koto, Zhengzhong Liu, Timothy Baldwin, Haonan Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14897)  

**Abstract**: Language model (LM) agents have gained significant attention for their ability to autonomously complete tasks through interactions with environments, tools, and APIs. LM agents are primarily built with prompt engineering or supervised finetuning. At the same time, reinforcement learning (RL) has been explored to enhance LM's capabilities, such as reasoning and factuality. However, the combination of the LM agents and reinforcement learning (Agent-RL) remains underexplored and lacks systematic study. To this end, we built AgentFly, a scalable and extensible Agent-RL framework designed to empower LM agents with a variety of RL algorithms. Our framework supports multi-turn interactions by adapting traditional RL methods with token-level masking. It features a decorator-based interface for defining tools and reward functions, enabling seamless extension and ease of use. To support high-throughput training, we implement asynchronous execution of tool calls and reward computations, and design a centralized resource management system for scalable environment coordination. We also provide a suite of prebuilt tools and environments, demonstrating the framework's effectiveness through successful agent training across multiple tasks. 

---
# Towards AI Urban Planner in the Age of GenAI, LLMs, and Agentic AI 

**Authors**: Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14730)  

**Abstract**: Generative AI, large language models, and agentic AI have emerged separately of urban planning. However, the convergence between AI and urban planning presents an interesting opportunity towards AI urban planners. This paper conceptualizes urban planning as a generative AI task, where AI synthesizes land-use configurations under geospatial, social, and human-centric constraints. We survey how generative AI approaches, including VAEs, GANs, transformers, and diffusion models, reshape urban design. We further identify critical gaps: 1) limited research on integrating urban theory guidance, 2) limited research of AI urban planning over multiple spatial resolutions or angularities, 3) limited research on augmenting urban design knowledge from data, and 4) limited research on addressing real-world interactions. To address these limitations, we outline future research directions in theory-guided generation, digital twins, and human-machine co-design, calling for a new synthesis of generative intelligence and participatory urbanism. 

---
# Automated Safety Evaluations Across 20 Large Language Models: The Aymara LLM Risk and Responsibility Matrix 

**Authors**: Juan Manuel Contreras  

**Link**: [PDF](https://arxiv.org/pdf/2507.14719)  

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications, scalable and rigorous safety evaluation is essential. This paper introduces Aymara AI, a programmatic platform for generating and administering customized, policy-grounded safety evaluations. Aymara AI transforms natural-language safety policies into adversarial prompts and scores model responses using an AI-based rater validated against human judgments. We demonstrate its capabilities through the Aymara LLM Risk and Responsibility Matrix, which evaluates 20 commercially available LLMs across 10 real-world safety domains. Results reveal wide performance disparities, with mean safety scores ranging from 86.2% to 52.4%. While models performed well in well-established safety domains such as Misinformation (mean = 95.7%), they consistently failed in more complex or underspecified domains, notably Privacy & Impersonation (mean = 24.3%). Analyses of Variance confirmed that safety scores differed significantly across both models and domains (p < .05). These findings underscore the inconsistent and context-dependent nature of LLM safety and highlight the need for scalable, customizable tools like Aymara AI to support responsible AI development and oversight. 

---
# Configurable multi-agent framework for scalable and realistic testing of llm-based agents 

**Authors**: Sai Wang, Senthilnathan Subramanian, Mudit Sahni, Praneeth Gone, Lingjie Meng, Xiaochen Wang, Nicolas Ferradas Bertoli, Tingxian Cheng, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14705)  

**Abstract**: Large-language-model (LLM) agents exhibit complex, context-sensitive behaviour that quickly renders static benchmarks and ad-hoc manual testing obsolete.
We present Neo, a configurable, multi-agent framework that automates realistic, multi-turn evaluation of LLM-based systems. Neo couples a Question Generation Agent and an Evaluation Agent through a shared context-hub, allowing domain prompts, scenario controls and dynamic feedback to be composed modularly. Test inputs are sampled from a probabilistic state model spanning dialogue flow, user intent and emotional tone, enabling diverse, human-like conversations that adapt after every turn.
Applied to a production-grade Seller Financial Assistant chatbot, Neo (i) uncovered edge-case failures across five attack categories with a 3.3% break rate close to the 5.8% achieved by expert human red-teamers, and (ii) delivered 10-12X higher throughput, generating 180 coherent test questions in around 45 mins versus 16h of human effort. Beyond security probing, Neo's stochastic policies balanced topic coverage and conversational depth, yielding broader behavioural exploration than manually crafted scripts.
Neo therefore lays a foundation for scalable, self-evolving LLM QA: its agent interfaces, state controller and feedback loops are model-agnostic and extensible to richer factual-grounding and policy-compliance checks. We release the framework to facilitate reproducible, high-fidelity testing of emerging agentic systems. 

---
# When Autonomy Goes Rogue: Preparing for Risks of Multi-Agent Collusion in Social Systems 

**Authors**: Qibing Ren, Sitao Xie, Longxuan Wei, Zhenfei Yin, Junchi Yan, Lizhuang Ma, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.14660)  

**Abstract**: Recent large-scale events like election fraud and financial scams have shown how harmful coordinated efforts by human groups can be. With the rise of autonomous AI systems, there is growing concern that AI-driven groups could also cause similar harm. While most AI safety research focuses on individual AI systems, the risks posed by multi-agent systems (MAS) in complex real-world situations are still underexplored. In this paper, we introduce a proof-of-concept to simulate the risks of malicious MAS collusion, using a flexible framework that supports both centralized and decentralized coordination structures. We apply this framework to two high-risk fields: misinformation spread and e-commerce fraud. Our findings show that decentralized systems are more effective at carrying out malicious actions than centralized ones. The increased autonomy of decentralized systems allows them to adapt their strategies and cause more damage. Even when traditional interventions, like content flagging, are applied, decentralized groups can adjust their tactics to avoid detection. We present key insights into how these malicious groups operate and the need for better detection systems and countermeasures. Code is available at this https URL. 

---
# Efficient Story Point Estimation With Comparative Learning 

**Authors**: Monoshiz Mahbub Khan, Xioayin Xi, Andrew Meneely, Zhe Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14642)  

**Abstract**: Story point estimation is an essential part of agile software development. Story points are unitless, project-specific effort estimates that help developers plan their sprints. Traditionally, developers estimate story points collaboratively using planning poker or other manual techniques. While the initial calibrating of the estimates to each project is helpful, once a team has converged on a set of precedents, story point estimation can become tedious and labor-intensive. Machine learning can reduce this burden, but only with enough context from the historical decisions made by the project team. That is, state-of-the-art models, such as GPT2SP and FastText-SVM, only make accurate predictions (within-project) when trained on data from the same project. The goal of this work is to streamline story point estimation by evaluating a comparative learning-based framework for calibrating project-specific story point prediction models. Instead of assigning a specific story point value to every backlog item, developers are presented with pairs of items, and indicate which item requires more effort. Using these comparative judgments, a machine learning model is trained to predict the story point estimates. We empirically evaluated our technique using data with 23,313 manual estimates in 16 projects. The model learned from comparative judgments can achieve on average 0.34 Spearman's rank correlation coefficient between its predictions and the ground truth story points. This is similar to, if not better than, the performance of a regression model learned from the ground truth story points. Therefore, the proposed comparative learning approach is more efficient than state-of-the-art regression-based approaches according to the law of comparative judgments - providing comparative judgments yields a lower cognitive burden on humans than providing ratings or categorical labels. 

---
# Coordinate Heart System: A Geometric Framework for Emotion Representation 

**Authors**: Omar Al-Desi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14593)  

**Abstract**: This paper presents the Coordinate Heart System (CHS), a geometric framework for emotion representation in artificial intelligence applications. We position eight core emotions as coordinates on a unit circle, enabling mathematical computation of complex emotional states through coordinate mixing and vector operations. Our initial five-emotion model revealed significant coverage gaps in the emotion space, leading to the development of an eight-emotion system that provides complete geometric coverage with mathematical guarantees. The framework converts natural language input to emotion coordinates and supports real-time emotion interpolation through computational algorithms. The system introduces a re-calibrated stability parameter S in [0,1], which dynamically integrates emotional load, conflict resolution, and contextual drain factors. This stability model leverages advanced Large Language Model interpretation of textual cues and incorporates hybrid temporal tracking mechanisms to provide nuanced assessment of psychological well-being states. Our key contributions include: (i) mathematical proof demonstrating why five emotions are insufficient for complete geometric coverage, (ii) an eight-coordinate system that eliminates representational blind spots, (iii) novel algorithms for emotion mixing, conflict resolution, and distance calculation in emotion space, and (iv) a comprehensive computational framework for AI emotion recognition with enhanced multi-dimensional stability modeling. Experimental validation through case studies demonstrates the system's capability to handle emotionally conflicted states, contextual distress factors, and complex psychological scenarios that traditional categorical emotion models cannot adequately represent. This work establishes a new mathematical foundation for emotion modeling in artificial intelligence systems. 

---
# Large Language Models Assisting Ontology Evaluation 

**Authors**: Anna Sofia Lippolis, Mohammad Javad Saeedizade, Robin Keskisärkkä, Aldo Gangemi, Eva Blomqvist, Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2507.14552)  

**Abstract**: Ontology evaluation through functional requirements, such as testing via competency question (CQ) verification, is a well-established yet costly, labour-intensive, and error-prone endeavour, even for ontology engineering experts. In this work, we introduce OE-Assist, a novel framework designed to assist ontology evaluation through automated and semi-automated CQ verification. By presenting and leveraging a dataset of 1,393 CQs paired with corresponding ontologies and ontology stories, our contributions present, to our knowledge, the first systematic investigation into large language model (LLM)-assisted ontology evaluation, and include: (i) evaluating the effectiveness of a LLM-based approach for automatically performing CQ verification against a manually created gold standard, and (ii) developing and assessing an LLM-powered framework to assist CQ verification with Protégé, by providing suggestions. We found that automated LLM-based evaluation with o1-preview and o3-mini perform at a similar level to the average user's performance. 

---
# What if Othello-Playing Language Models Could See? 

**Authors**: Xinyi Chen, Yifei Yuan, Jiaang Li, Serge Belongie, Maarten de Rijke, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2507.14520)  

**Abstract**: Language models are often said to face a symbol grounding problem. While some argue that world understanding can emerge from text alone, others suggest grounded learning is more efficient. We explore this through Othello, where the board state defines a simplified, rule-based world. Building on prior work, we introduce VISOTHELLO, a multi-modal model trained on move histories and board images. Using next-move prediction, we compare it to mono-modal baselines and test robustness to semantically irrelevant perturbations. We find that multi-modal training improves both performance and the robustness of internal representations. These results suggest that grounding language in visual input helps models infer structured world representations. 

---
# Amico: An Event-Driven Modular Framework for Persistent and Embedded Autonomy 

**Authors**: Hongyi Yang, Yue Pan, Jiayi Xu, Kelsen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14513)  

**Abstract**: Recent advances in large language models (LLMs) and autonomous agents have enabled systems capable of performing complex tasks across domains such as human-computer interaction, planning, and web navigation. However, many existing frameworks struggle in real-world or resource-constrained environments due to their reliance on cloud-based computation, limited robustness in dynamic contexts, and lack of persistent autonomy and environmental awareness.
We present Amico, a modular, event-driven framework for building autonomous agents optimized for embedded systems. Written in Rust for safety and performance, Amico supports reactive, persistent agents that operate efficiently across embedded platforms and browser environments via WebAssembly. It provides clean abstractions for event handling, state management, behavior execution, and integration with reasoning modules. Amico delivers a unified infrastructure for constructing resilient, interactive agents suitable for deployment in settings with limited compute and intermittent connectivity. 

---
# BioGraphFusion: Graph Knowledge Embedding for Biological Completion and Reasoning 

**Authors**: Yitong Lin, Jiaying He, Jiahe Chen, Xinnan Zhu, Jianwei Zheng, Tao Bo  

**Link**: [PDF](https://arxiv.org/pdf/2507.14468)  

**Abstract**: Motivation: Biomedical knowledge graphs (KGs) are crucial for drug discovery and disease understanding, yet their completion and reasoning are challenging. Knowledge Embedding (KE) methods capture global semantics but struggle with dynamic structural integration, while Graph Neural Networks (GNNs) excel locally but often lack semantic understanding. Even ensemble approaches, including those leveraging language models, often fail to achieve a deep, adaptive, and synergistic co-evolution between semantic comprehension and structural learning. Addressing this critical gap in fostering continuous, reciprocal refinement between these two aspects in complex biomedical KGs is paramount.
Results: We introduce BioGraphFusion, a novel framework for deeply synergistic semantic and structural learning. BioGraphFusion establishes a global semantic foundation via tensor decomposition, guiding an LSTM-driven mechanism to dynamically refine relation embeddings during graph propagation. This fosters adaptive interplay between semantic understanding and structural learning, further enhanced by query-guided subgraph construction and a hybrid scoring mechanism. Experiments across three key biomedical tasks demonstrate BioGraphFusion's superior performance over state-of-the-art KE, GNN, and ensemble models. A case study on Cutaneous Malignant Melanoma 1 (CMM1) highlights its ability to unveil biologically meaningful pathways.
Availability and Implementation: Source code and all training data are freely available for download at this https URL.
Contact: zjw@zjut.this http URL, botao666666@126.com.
Supplementary information: Supplementary data are available at Bioinformatics online. 

---
# Routine: A Structural Planning Framework for LLM Agent System in Enterprise 

**Authors**: Guancheng Zeng, Xueyi Chen, Jiawang Hu, Shaohua Qi, Yaxuan Mao, Zhantao Wang, Yifan Nie, Shuang Li, Qiuyang Feng, Pengxu Qiu, Yujia Wang, Wenqiang Han, Linyan Huang, Gang Li, Jingjing Mo, Haowen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14447)  

**Abstract**: The deployment of agent systems in an enterprise environment is often hindered by several challenges: common models lack domain-specific process knowledge, leading to disorganized plans, missing key tools, and poor execution stability. To address this, this paper introduces Routine, a multi-step agent planning framework designed with a clear structure, explicit instructions, and seamless parameter passing to guide the agent's execution module in performing multi-step tool-calling tasks with high stability. In evaluations conducted within a real-world enterprise scenario, Routine significantly increases the execution accuracy in model tool calls, increasing the performance of GPT-4o from 41.1% to 96.3%, and Qwen3-14B from 32.6% to 83.3%. We further constructed a Routine-following training dataset and fine-tuned Qwen3-14B, resulting in an accuracy increase to 88.2% on scenario-specific evaluations, indicating improved adherence to execution plans. In addition, we employed Routine-based distillation to create a scenario-specific, multi-step tool-calling dataset. Fine-tuning on this distilled dataset raised the model's accuracy to 95.5%, approaching GPT-4o's performance. These results highlight Routine's effectiveness in distilling domain-specific tool-usage patterns and enhancing model adaptability to new scenarios. Our experimental results demonstrate that Routine provides a practical and accessible approach to building stable agent workflows, accelerating the deployment and adoption of agent systems in enterprise environments, and advancing the technical vision of AI for Process. 

---
# Inverse Scaling in Test-Time Compute 

**Authors**: Aryo Pradipta Gema, Alexander Hägele, Runjin Chen, Andy Arditi, Jacob Goldman-Wetzler, Kit Fraser-Taliente, Henry Sleight, Linda Petrini, Julian Michael, Beatrice Alex, Pasquale Minervini, Yanda Chen, Joe Benton, Ethan Perez  

**Link**: [PDF](https://arxiv.org/pdf/2507.14417)  

**Abstract**: We construct evaluation tasks where extending the reasoning length of Large Reasoning Models (LRMs) deteriorates performance, exhibiting an inverse scaling relationship between test-time compute and accuracy. Our evaluation tasks span four categories: simple counting tasks with distractors, regression tasks with spurious features, deduction tasks with constraint tracking, and advanced AI risks. We identify five distinct failure modes when models reason for longer: 1) Claude models become increasingly distracted by irrelevant information; 2) OpenAI o-series models resist distractors but overfit to problem framings; 3) models shift from reasonable priors to spurious correlations; 4) all models show difficulties in maintaining focus on complex deductive tasks; and 5) extended reasoning may amplify concerning behaviors, with Claude Sonnet 4 showing increased expressions of self-preservation. These findings suggest that while test-time compute scaling remains promising for improving model capabilities, it may inadvertently reinforce problematic reasoning patterns. Our results demonstrate the importance of evaluating models across diverse reasoning lengths to identify and address these failure modes in LRMs. 

---
# Fail Fast, or Ask: Mitigating the Deficiencies of Reasoning LLMs with Human-in-the-Loop Systems Engineering 

**Authors**: Michael J. Zellinger, Matt Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2507.14406)  

**Abstract**: State-of-the-art reasoning LLMs are powerful problem solvers, but they still occasionally make mistakes. However, adopting AI models in risk-sensitive domains often requires error rates near 0%. To address this gap, we propose collaboration between a reasoning model and a human expert who resolves queries the model cannot confidently answer. We find that quantifying the uncertainty of a reasoning model through the length of its reasoning trace yields an effective basis for deferral to a human, e.g., cutting the error rate of Qwen3 235B-A22B on difficult MATH problems from 3% to less than 1% when deferring 7.5% of queries. However, the high latency of reasoning models still makes them challenging to deploy on use cases with high query volume. To address this challenge, we explore fronting a reasoning model with a large non-reasoning model. We call this modified human-in-the-loop system "Fail Fast, or Ask", since the non-reasoning model may defer difficult queries to the human expert directly ("failing fast"), without incurring the reasoning model's higher latency. We show that this approach yields around 40% latency reduction and about 50% cost savings for DeepSeek R1 while maintaining 90+% area under the accuracy-rejection curve. However, we observe that latency savings are lower than expected because of "latency drag", the phenomenon that processing easier queries with a non-reasoning model pushes the reasoning model's latency distribution towards longer latencies. Broadly, our results suggest that the deficiencies of state-of-the-art reasoning models -- nontrivial error rates and high latency -- can be substantially mitigated through black-box systems engineering, without requiring access to LLM internals. 

---
# Adaptive Multi-Agent Reasoning via Automated Workflow Generation 

**Authors**: Humza Sami, Mubashir ul Islam, Pierre-Emmanuel Gaillardon, Valerio Tenace  

**Link**: [PDF](https://arxiv.org/pdf/2507.14393)  

**Abstract**: The rise of Large Reasoning Models (LRMs) promises a significant leap forward in language model capabilities, aiming to tackle increasingly sophisticated tasks with unprecedented efficiency and accuracy. However, despite their impressive performance, recent studies have highlighted how current reasoning models frequently fail to generalize to novel, unseen problems, often resorting to memorized solutions rather than genuine inferential reasoning. Such behavior underscores a critical limitation in modern LRMs, i.e., their tendency toward overfitting, which in turn results in poor generalization in problem-solving capabilities.
In this paper, we introduce Nexus Architect, an enhanced iteration of our multi-agent system framework, Nexus, equipped with a novel automated workflow synthesis mechanism. Given a user's prompt and a small set of representative examples, the Architect autonomously generates a tailored reasoning workflow by selecting suitable strategies, tool integrations, and adversarial techniques for a specific problem class. Furthermore, the Architect includes an iterative prompt refinement mechanism that fine-tunes agents' system prompts to maximize performance and improve the generalization capabilities of the system.
We empirically evaluate Nexus Architect by employing an off-the-shelf, non-reasoning model on a custom dataset of challenging logical questions and compare its performance against state-of-the-art LRMs. Results show that Nexus Architect consistently outperforms existing solutions, achieving up to a 66% increase in pass rate over Gemini 2.5 Flash Preview, nearly 2.5$\times$ against Claude Sonnet 4 and DeepSeek-R1, and over 3$\times$ w.r.t. Llama 4 Scout. 

---
# ProofCompass: Enhancing Specialized Provers with LLM Guidance 

**Authors**: Nicolas Wischermann, Claudio Mayrink Verdun, Gabriel Poesia, Francesco Noseda  

**Link**: [PDF](https://arxiv.org/pdf/2507.14335)  

**Abstract**: Language models have become increasingly powerful tools for formal mathematical reasoning. However, most existing approaches rely exclusively on either large general-purpose models or smaller specialized models, each with distinct limitations, while training specialized large models still requires significant computational resources. This paper introduces ProofCompass, a novel hybrid methodology that achieves remarkable computational efficiency by strategically guiding existing specialized prover methods, such as DeepSeek-Prover-v1.5-RL (DSP-v1.5) with a Large Language Model (LLM) without requiring additional model training. The LLM provides natural language proof strategies and analyzes failed attempts to select intermediate lemmas, enabling effective problem decomposition. On the miniF2F benchmark, ProofCompass demonstrates substantial resource efficiency: it outperforms DSP-v1.5 ($54.9\% \rightarrow 55.3\%$) while using 25x fewer attempts ($3200 \rightarrow 128$). Our synergistic approach paves the way for simultaneously improving computational efficiency and accuracy in formal theorem proving. 

---
# Language Models as Ontology Encoders 

**Authors**: Hui Yang, Jiaoyan Chen, Yuan He, Yongsheng Gao, Ian Horrocks  

**Link**: [PDF](https://arxiv.org/pdf/2507.14334)  

**Abstract**: OWL (Web Ontology Language) ontologies which are able to formally represent complex knowledge and support semantic reasoning have been widely adopted across various domains such as healthcare and bioinformatics. Recently, ontology embeddings have gained wide attention due to its potential to infer plausible new knowledge and approximate complex reasoning. However, existing methods face notable limitations: geometric model-based embeddings typically overlook valuable textual information, resulting in suboptimal performance, while the approaches that incorporate text, which are often based on language models, fail to preserve the logical structure. In this work, we propose a new ontology embedding method OnT, which tunes a Pretrained Language Model (PLM) via geometric modeling in a hyperbolic space for effectively incorporating textual labels and simultaneously preserving class hierarchies and other logical relationships of Description Logic EL. Extensive experiments on four real-world ontologies show that OnT consistently outperforms the baselines including the state-of-the-art across both tasks of prediction and inference of axioms. OnT also demonstrates strong potential in real-world applications, indicated by its robust transfer learning abilities and effectiveness in real cases of constructing a new ontology from SNOMED CT. Data and code are available at this https URL. 

---
# Manimator: Transforming Research Papers into Visual Explanations 

**Authors**: Samarth P, Vyoman Jain, Shiva Golugula, Motamarri Sai Sathvik  

**Link**: [PDF](https://arxiv.org/pdf/2507.14306)  

**Abstract**: Understanding complex scientific and mathematical concepts, particularly those presented in dense research papers, poses a significant challenge for learners. Dynamic visualizations can greatly enhance comprehension, but creating them manually is time-consuming and requires specialized knowledge and skills. We introduce manimator, an open-source system that leverages Large Language Models to transform research papers and natural language prompts into explanatory animations using the Manim engine. Manimator employs a pipeline where an LLM interprets the input text or research paper PDF to generate a structured scene description outlining key concepts, mathematical formulas, and visual elements and another LLM translates this description into executable Manim Python code. We discuss its potential as an educational tool for rapidly creating engaging visual explanations for complex STEM topics, democratizing the creation of high-quality educational content. 

---
# WebGuard: Building a Generalizable Guardrail for Web Agents 

**Authors**: Boyuan Zheng, Zeyi Liao, Scott Salisbury, Zeyuan Liu, Michael Lin, Qinyuan Zheng, Zifan Wang, Xiang Deng, Dawn Song, Huan Sun, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.14293)  

**Abstract**: The rapid development of autonomous web agents powered by Large Language Models (LLMs), while greatly elevating efficiency, exposes the frontier risk of taking unintended or harmful actions. This situation underscores an urgent need for effective safety measures, akin to access controls for human users. To address this critical challenge, we introduce WebGuard, the first comprehensive dataset designed to support the assessment of web agent action risks and facilitate the development of guardrails for real-world online environments. In doing so, WebGuard specifically focuses on predicting the outcome of state-changing actions and contains 4,939 human-annotated actions from 193 websites across 22 diverse domains, including often-overlooked long-tail websites. These actions are categorized using a novel three-tier risk schema: SAFE, LOW, and HIGH. The dataset includes designated training and test splits to support evaluation under diverse generalization settings. Our initial evaluations reveal a concerning deficiency: even frontier LLMs achieve less than 60% accuracy in predicting action outcomes and less than 60% recall in lagging HIGH-risk actions, highlighting the risks of deploying current-generation agents without dedicated safeguards. We therefore investigate fine-tuning specialized guardrail models using WebGuard. We conduct comprehensive evaluations across multiple generalization settings and find that a fine-tuned Qwen2.5VL-7B model yields a substantial improvement in performance, boosting accuracy from 37% to 80% and HIGH-risk action recall from 20% to 76%. Despite these improvements, the performance still falls short of the reliability required for high-stakes deployment, where guardrails must approach near-perfect accuracy and recall. 

---
# DREAMS: Density Functional Theory Based Research Engine for Agentic Materials Simulation 

**Authors**: Ziqi Wang, Hongshuo Huang, Hancheng Zhao, Changwen Xu, Shang Zhu, Jan Janssen, Venkatasubramanian Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14267)  

**Abstract**: Materials discovery relies on high-throughput, high-fidelity simulation techniques such as Density Functional Theory (DFT), which require years of training, extensive parameter fine-tuning and systematic error handling. To address these challenges, we introduce the DFT-based Research Engine for Agentic Materials Screening (DREAMS), a hierarchical, multi-agent framework for DFT simulation that combines a central Large Language Model (LLM) planner agent with domain-specific LLM agents for atomistic structure generation, systematic DFT convergence testing, High-Performance Computing (HPC) scheduling, and error handling. In addition, a shared canvas helps the LLM agents to structure their discussions, preserve context and prevent hallucination. We validate DREAMS capabilities on the Sol27LC lattice-constant benchmark, achieving average errors below 1\% compared to the results of human DFT experts. Furthermore, we apply DREAMS to the long-standing CO/Pt(111) adsorption puzzle, demonstrating its long-term and complex problem-solving capabilities. The framework again reproduces expert-level literature adsorption-energy differences. Finally, DREAMS is employed to quantify functional-driven uncertainties with Bayesian ensemble sampling, confirming the Face Centered Cubic (FCC)-site preference at the Generalized Gradient Approximation (GGA) DFT level. In conclusion, DREAMS approaches L3-level automation - autonomous exploration of a defined design space - and significantly reduces the reliance on human expertise and intervention, offering a scalable path toward democratized, high-throughput, high-fidelity computational materials discovery. 

---
# The Free Will Equation: Quantum Field Analogies for AGI 

**Authors**: Rahul Kabali  

**Link**: [PDF](https://arxiv.org/pdf/2507.14154)  

**Abstract**: Artificial General Intelligence (AGI) research traditionally focuses on algorithms that optimize for specific goals under deterministic rules. Yet, human-like intelligence exhibits adaptive spontaneity - an ability to make unexpected choices or free decisions not strictly dictated by past data or immediate reward. This trait, often dubbed "free will" in a loose sense, might be crucial for creativity, robust adaptation, and avoiding ruts in problem-solving. This paper proposes a theoretical framework, called the Free Will Equation, that draws analogies from quantum field theory to endow AGI agents with a form of adaptive, controlled stochasticity in their decision-making process. The core idea is to treat an AI agent's cognitive state as a superposition of potential actions or thoughts, which collapses probabilistically into a concrete action when a decision is made - much like a quantum wavefunction collapsing upon measurement. By incorporating mechanisms analogous to quantum fields, along with intrinsic motivation terms, we aim to improve an agent's ability to explore novel strategies and adapt to unforeseen changes. Experiments in a non-stationary multi-armed bandit environment demonstrate that agents using this framework achieve higher rewards and policy diversity compared to baseline methods. 

---
# Diffusion Beats Autoregressive in Data-Constrained Settings 

**Authors**: Mihir Prabhudesai, Menging Wu, Amir Zadeh, Katerina Fragkiadaki, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2507.15857)  

**Abstract**: Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: this https URL. 

---
# SeC: Advancing Complex Video Object Segmentation via Progressive Concept Construction 

**Authors**: Zhixiong Zhang, Shuangrui Ding, Xiaoyi Dong, Songxin He, Jianfan Lin, Junsong Tang, Yuhang Zang, Yuhang Cao, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15852)  

**Abstract**: Video Object Segmentation (VOS) is a core task in computer vision, requiring models to track and segment target objects across video frames. Despite notable advances with recent efforts, current techniques still lag behind human capabilities in handling drastic visual variations, occlusions, and complex scene changes. This limitation arises from their reliance on appearance matching, neglecting the human-like conceptual understanding of objects that enables robust identification across temporal dynamics. Motivated by this gap, we propose Segment Concept (SeC), a concept-driven segmentation framework that shifts from conventional feature matching to the progressive construction and utilization of high-level, object-centric representations. SeC employs Large Vision-Language Models (LVLMs) to integrate visual cues across diverse frames, constructing robust conceptual priors. During inference, SeC forms a comprehensive semantic representation of the target based on processed frames, realizing robust segmentation of follow-up frames. Furthermore, SeC adaptively balances LVLM-based semantic reasoning with enhanced feature matching, dynamically adjusting computational efforts based on scene complexity. To rigorously assess VOS methods in scenarios demanding high-level conceptual reasoning and robust semantic understanding, we introduce the Semantic Complex Scenarios Video Object Segmentation benchmark (SeCVOS). SeCVOS comprises 160 manually annotated multi-scenario videos designed to challenge models with substantial appearance variations and dynamic scene transformations. In particular, SeC achieves an 11.8-point improvement over SAM 2.1 on SeCVOS, establishing a new state-of-the-art in concept-aware video object segmentation. 

---
# The Impact of Language Mixing on Bilingual LLM Reasoning 

**Authors**: Yihao Li, Jiayi Xin, Miranda Muqing Miao, Qi Long, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2507.15849)  

**Abstract**: Proficient multilingual speakers often intentionally switch languages in the middle of a conversation. Similarly, recent reasoning-focused bilingual large language models (LLMs) with strong capabilities in both languages exhibit language mixing--alternating languages within their chain of thought. Discouraging this behavior in DeepSeek-R1 was found to degrade accuracy, suggesting that language mixing may benefit reasoning. In this work, we study language switching in Chinese-English bilingual reasoning models. We identify reinforcement learning with verifiable rewards (RLVR) as the critical training stage that leads to language mixing. We demonstrate that language mixing can enhance reasoning: enforcing monolingual decoding reduces accuracy by 5.6 percentage points on math reasoning tasks. Additionally, a lightweight probe can be trained to predict whether a potential language switch would benefit or harm reasoning, and when used to guide decoding, increases accuracy by up to 6.25 percentage points. Our findings suggest that language mixing is not merely a byproduct of multilingual training, but is a strategic reasoning behavior. 

---
# GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding 

**Authors**: Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15846)  

**Abstract**: Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current reinforcement learning approaches use binary rewards that treat elements as hit-or-miss targets, creating sparse signals that ignore the continuous nature of spatial interactions. Motivated by human clicking behavior that naturally forms Gaussian distributions centered on target elements, we introduce GUI Gaussian Grounding Rewards (GUI-G$^2$), a principled reward framework that models GUI elements as continuous Gaussian distributions across the interface plane. GUI-G$^2$ incorporates two synergistic mechanisms: Gaussian point rewards model precise localization through exponentially decaying distributions centered on element centroids, while coverage rewards assess spatial alignment by measuring the overlap between predicted Gaussian distributions and target regions. To handle diverse element scales, we develop an adaptive variance mechanism that calibrates reward distributions based on element dimensions. This framework transforms GUI grounding from sparse binary classification to dense continuous optimization, where Gaussian distributions generate rich gradient signals that guide models toward optimal interaction positions. Extensive experiments across ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro benchmarks demonstrate that GUI-G$^2$, substantially outperforms state-of-the-art method UI-TARS-72B, with the most significant improvement of 24.7% on ScreenSpot-Pro. Our analysis reveals that continuous modeling provides superior robustness to interface variations and enhanced generalization to unseen layouts, establishing a new paradigm for spatial reasoning in GUI interaction tasks. 

---
# FASTGEN: Fast and Cost-Effective Synthetic Tabular Data Generation with LLMs 

**Authors**: Anh Nguyen, Sam Schafft, Nicholas Hale, John Alfaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.15839)  

**Abstract**: Synthetic data generation has emerged as an invaluable solution in scenarios where real-world data collection and usage are limited by cost and scarcity. Large language models (LLMs) have demonstrated remarkable capabilities in producing high-fidelity, domain-relevant samples across various fields. However, existing approaches that directly use LLMs to generate each record individually impose prohibitive time and cost burdens, particularly when large volumes of synthetic data are required. In this work, we propose a fast, cost-effective method for realistic tabular data synthesis that leverages LLMs to infer and encode each field's distribution into a reusable sampling script. By automatically classifying fields into numerical, categorical, or free-text types, the LLM generates distribution-based scripts that can efficiently produce diverse, realistic datasets at scale without continuous model inference. Experimental results show that our approach outperforms traditional direct methods in both diversity and data realism, substantially reducing the burden of high-volume synthetic data generation. We plan to apply this methodology to accelerate testing in production pipelines, thereby shortening development cycles and improving overall system efficiency. We believe our insights and lessons learned will aid researchers and practitioners seeking scalable, cost-effective solutions for synthetic data generation. 

---
# Look, Focus, Act: Efficient and Robust Robot Learning via Human Gaze and Foveated Vision Transformers 

**Authors**: Ian Chuang, Andrew Lee, Dechen Gao, Jinyu Zou, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2507.15833)  

**Abstract**: Human vision is a highly active process driven by gaze, which directs attention and fixation to task-relevant regions and dramatically reduces visual processing. In contrast, robot learning systems typically rely on passive, uniform processing of raw camera images. In this work, we explore how incorporating human-like active gaze into robotic policies can enhance both efficiency and performance. We build on recent advances in foveated image processing and apply them to an Active Vision robot system that emulates both human head movement and eye tracking. Extending prior work on the AV-ALOHA robot simulation platform, we introduce a framework for simultaneously collecting eye-tracking data and robot demonstrations from a human operator as well as a simulation benchmark and dataset for training robot policies that incorporate human gaze. Given the widespread use of Vision Transformers (ViTs) in robot learning, we integrate gaze information into ViTs using a foveated patch tokenization scheme inspired by recent work in image segmentation. Compared to uniform patch tokenization, this significantly reduces the number of tokens-and thus computation-without sacrificing visual fidelity near regions of interest. We also explore two approaches to gaze imitation and prediction from human data. The first is a two-stage model that predicts gaze to guide foveation and action; the second integrates gaze into the action space, allowing the policy to jointly predict gaze and actions end-to-end. Our results show that our method for foveated robot vision not only drastically reduces computational overhead, but also improves performance for high precision tasks and robustness to unseen distractors. Together, these findings suggest that human-inspired visual processing offers a useful inductive bias for robotic vision systems. this https URL 

---
# Operationalizing AI for Good: Spotlight on Deployment and Integration of AI Models in Humanitarian Work 

**Authors**: Anton Abilov, Ke Zhang, Hemank Lamba, Elizabeth M. Olson, Joel R. Tetreault, Alejandro Jaimes  

**Link**: [PDF](https://arxiv.org/pdf/2507.15823)  

**Abstract**: Publications in the AI for Good space have tended to focus on the research and model development that can support high-impact applications. However, very few AI for Good papers discuss the process of deploying and collaborating with the partner organization, and the resulting real-world impact. In this work, we share details about the close collaboration with a humanitarian-to-humanitarian (H2H) organization and how to not only deploy the AI model in a resource-constrained environment, but also how to maintain it for continuous performance updates, and share key takeaways for practitioners. 

---
# Do AI models help produce verified bug fixes? 

**Authors**: Li Huang, Ilgiz Mustafin, Marco Piccioni, Alessandro Schena, Reto Weber, Bertrand Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.15822)  

**Abstract**: Among areas of software engineering where AI techniques -- particularly, Large Language Models -- seem poised to yield dramatic improvements, an attractive candidate is Automatic Program Repair (APR), the production of satisfactory corrections to software bugs. Does this expectation materialize in practice? How do we find out, making sure that proposed corrections actually work? If programmers have access to LLMs, how do they actually use them to complement their own skills?
To answer these questions, we took advantage of the availability of a program-proving environment, which formally determines the correctness of proposed fixes, to conduct a study of program debugging with two randomly assigned groups of programmers, one with access to LLMs and the other without, both validating their answers through the proof tools. The methodology relied on a division into general research questions (Goals in the Goal-Query-Metric approach), specific elements admitting specific answers (Queries), and measurements supporting these answers (Metrics). While applied so far to a limited sample size, the results are a first step towards delineating a proper role for AI and LLMs in providing guaranteed-correct fixes to program bugs.
These results caused surprise as compared to what one might expect from the use of AI for debugging and APR. The contributions also include: a detailed methodology for experiments in the use of LLMs for debugging, which other projects can reuse; a fine-grain analysis of programmer behavior, made possible by the use of full-session recording; a definition of patterns of use of LLMs, with 7 distinct categories; and validated advice for getting the best of LLMs for debugging and Automatic Program Repair. 

---
# True Multimodal In-Context Learning Needs Attention to the Visual Context 

**Authors**: Shuo Chen, Jianzhe Liu, Zhen Han, Yan Xia, Daniel Cremers, Philip Torr, Volker Tresp, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15807)  

**Abstract**: Multimodal Large Language Models (MLLMs), built on powerful language backbones, have enabled Multimodal In-Context Learning (MICL)-adapting to new tasks from a few multimodal demonstrations consisting of images, questions, and answers. Despite showing noticeable improvement on standard vision-language datasets, current MLLMs struggle to leverage visual information in the demonstrations. Specifically, they tend to neglect visual cues and over-rely on textual patterns, leading to mere text imitation rather than genuine multimodal adaptation. This behavior makes MICL still unimodal and largely restricts its practical utility. More importantly, this limitation is often concealed by the improved performance on tasks that do not require understanding the visual context. As a result, how to effectively enhance MICL ability and reliably evaluate the MICL performance remains underexplored. To address these issues, we first introduce Dynamic Attention Reallocation (DARA), an efficient fine-tuning strategy that encourages models to attend to the visual context by rebalancing attention across visual and textual tokens. In addition, we present TrueMICL, an MICL-dedicated dataset with both support and test sets that explicitly requires the integration of multimodal information-particularly visual content-for correct task completion. Extensive experiments demonstrate the effectiveness of our holistic solution, showcasing substantial improvements in the true multimodal in-context learning capabilities. Code and datasets are available at this https URL . 

---
# ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction 

**Authors**: Danhui Chen, Ziquan Liu, Chuxi Yang, Dan Wang, Yan Yan, Yi Xu, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.15803)  

**Abstract**: Pixel-level vision tasks, such as semantic segmentation, require extensive and high-quality annotated data, which is costly to obtain. Semi-supervised semantic segmentation (SSSS) has emerged as a solution to alleviate the labeling burden by leveraging both labeled and unlabeled data through self-training techniques. Meanwhile, the advent of foundational segmentation models pre-trained on massive data, has shown the potential to generalize across domains effectively. This work explores whether a foundational segmentation model can address label scarcity in the pixel-level vision task as an annotator for unlabeled images. Specifically, we investigate the efficacy of using SEEM, a Segment Anything Model (SAM) variant fine-tuned for textual input, to generate predictive masks for unlabeled data. To address the shortcomings of using SEEM-generated masks as supervision, we propose ConformalSAM, a novel SSSS framework which first calibrates the foundation model using the target domain's labeled data and then filters out unreliable pixel labels of unlabeled data so that only high-confidence labels are used as supervision. By leveraging conformal prediction (CP) to adapt foundation models to target data through uncertainty calibration, ConformalSAM exploits the strong capability of the foundational segmentation model reliably which benefits the early-stage learning, while a subsequent self-reliance training strategy mitigates overfitting to SEEM-generated masks in the later training stage. Our experiment demonstrates that, on three standard benchmarks of SSSS, ConformalSAM achieves superior performance compared to recent SSSS methods and helps boost the performance of those methods as a plug-in. 

---
# Small LLMs Do Not Learn a Generalizable Theory of Mind via Reinforcement Learning 

**Authors**: Sneheel Sarangi, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2507.15788)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated emergent capabilities in complex reasoning, largely spurred by rule-based Reinforcement Learning (RL) techniques applied during the post-training. This has raised the question of whether similar methods can instill more nuanced, human-like social intelligence, such as a Theory of Mind (ToM), in LLMs. This paper investigates whether small-scale LLMs can acquire a robust and generalizable ToM capability through RL with verifiable rewards (RLVR). We conduct a systematic evaluation by training models on various combinations of prominent ToM datasets (HiToM, ExploreToM, FANToM) and testing for generalization on held-out datasets (e.g., OpenToM). Our findings indicate that small LLMs struggle to develop a generic ToM capability. While performance on in-distribution tasks improves, this capability fails to transfer to unseen ToM tasks with different characteristics. Furthermore, we demonstrate that prolonged RL training leads to models ``hacking'' the statistical patterns of the training datasets, resulting in significant performance gains on in-domain data but no change, or degradation of performance on out-of-distribution tasks. This suggests the learned behavior is a form of narrow overfitting rather than the acquisition of a true, abstract ToM capability. 

---
# Romance, Relief, and Regret: Teen Narratives of Chatbot Overreliance 

**Authors**: Mohammad 'Matt' Namvarpour, Brandon Brofsky, Jessica Medina, Mamtaj Akter, Afsaneh Razi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15783)  

**Abstract**: As Generative Artificial Intelligence (GenAI) driven chatbots like this http URL become embedded in adolescent life, they raise concerns about emotional dependence and digital overreliance. While studies have investigated the overreliance of adults on these chatbots, they have not investigated teens' interactions with chatbots with customizable personas. We analyzed 318 Reddit posts made by users self-reported as 13-17 years old on the this http URL subreddit to understand patterns of overreliance. We found teens commonly begin using chatbots for emotional support or creative expression, but many develop strong attachments that interfere with offline relationships and daily routines. Their posts revealed recurring signs of psychological distress, cycles of relapse, and difficulty disengaging. Teens reported that their overreliance often ended when they reflect on the harm, return to in-person social settings, or become frustrated by platform restrictions. Based on the implications of our findings, we provide recommendations for future chatbot design so they can promote self-awareness, support real-world engagement, and involve teens in developing safer digital tools. 

---
# Learning Null Geodesics for Gravitational Lensing Rendering in General Relativity 

**Authors**: Mingyuan Sun, Zheng Fang, Jiaxu Wang, Kunyi Zhang, Qiang Zhang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15775)  

**Abstract**: We present GravLensX, an innovative method for rendering black holes with gravitational lensing effects using neural networks. The methodology involves training neural networks to fit the spacetime around black holes and then employing these trained models to generate the path of light rays affected by gravitational lensing. This enables efficient and scalable simulations of black holes with optically thin accretion disks, significantly decreasing the time required for rendering compared to traditional methods. We validate our approach through extensive rendering of multiple black hole systems with superposed Kerr metric, demonstrating its capability to produce accurate visualizations with significantly $15\times$ reduced computational time. Our findings suggest that neural networks offer a promising alternative for rendering complex astrophysical phenomena, potentially paving a new path to astronomical visualization. 

---
# Dynamics is what you need for time-series forecasting! 

**Authors**: Alexis-Raja Brachet, Pierre-Yves Richard, Céline Hudelot  

**Link**: [PDF](https://arxiv.org/pdf/2507.15774)  

**Abstract**: While boundaries between data modalities are vanishing, the usual successful deep models are still challenged by simple ones in the time-series forecasting task. Our hypothesis is that this task needs models that are able to learn the data underlying dynamics. We propose to validate it through both systemic and empirical studies. We develop an original $\texttt{PRO-DYN}$ nomenclature to analyze existing models through the lens of dynamics. Two observations thus emerged: $\textbf{1}$. under-performing architectures learn dynamics at most partially, $\textbf{2}$. the location of the dynamics block at the model end is of prime importance. We conduct extensive experiments to confirm our observations on a set of performance-varying models with diverse backbones. Results support the need to incorporate a learnable dynamics block and its use as the final predictor. 

---
# Supernova: Achieving More with Less in Transformer Architectures 

**Authors**: Andrei-Valentin Tanase, Elena Pelican  

**Link**: [PDF](https://arxiv.org/pdf/2507.15773)  

**Abstract**: We present Supernova, a 650M-parameter decoder-only transformer that demonstrates how careful architectural design and tokenization innovation can achieve the performance of larger models while maintaining computational efficiency. Our architecture combines Rotary Positional Embeddings (RoPE), Grouped Query Attention (GQA) with a 3:1 compression ratio, RMSNorm for computational efficiency, and SwiGLU activation functions. A critical innovation is our custom 128,000-vocabulary byte-level BPE tokenizer, which achieves state-of-the-art compression performance. Through detailed analysis, we show that Supernova achieves 90% of the performance of 1B-parameter models while using 53% fewer parameters and requiring only 100B training tokens--an order of magnitude less than competing models. Our findings challenge the prevailing scaling paradigm, demonstrating that architectural efficiency and tokenization quality can compensate for reduced parameter counts. 

---
# Deep-Learning Investigation of Vibrational Raman Spectra for Plant-Stress Analysis 

**Authors**: Anoop C. Patil, Benny Jian Rong Sng, Yu-Wei Chang, Joana B. Pereira, Chua Nam-Hai, Rajani Sarojam, Gajendra Pratap Singh, In-Cheol Jang, Giovanni Volpe  

**Link**: [PDF](https://arxiv.org/pdf/2507.15772)  

**Abstract**: Detecting stress in plants is crucial for both open-farm and controlled-environment agriculture. Biomolecules within plants serve as key stress indicators, offering vital markers for continuous health monitoring and early disease detection. Raman spectroscopy provides a powerful, non-invasive means to quantify these biomolecules through their molecular vibrational signatures. However, traditional Raman analysis relies on customized data-processing workflows that require fluorescence background removal and prior identification of Raman peaks of interest-introducing potential biases and inconsistencies. Here, we introduce DIVA (Deep-learning-based Investigation of Vibrational Raman spectra for plant-stress Analysis), a fully automated workflow based on a variational autoencoder. Unlike conventional approaches, DIVA processes native Raman spectra-including fluorescence backgrounds-without manual preprocessing, identifying and quantifying significant spectral features in an unbiased manner. We applied DIVA to detect a range of plant stresses, including abiotic (shading, high light intensity, high temperature) and biotic stressors (bacterial infections). By integrating deep learning with vibrational spectroscopy, DIVA paves the way for AI-driven plant health assessment, fostering more resilient and sustainable agricultural practices. 

---
# Left Leaning Models: AI Assumptions on Economic Policy 

**Authors**: Maxim Chupilkin  

**Link**: [PDF](https://arxiv.org/pdf/2507.15771)  

**Abstract**: How does AI think about economic policy? While the use of large language models (LLMs) in economics is growing exponentially, their assumptions on economic issues remain a black box. This paper uses a conjoint experiment to tease out the main factors influencing LLMs' evaluation of economic policy. It finds that LLMs are most sensitive to unemployment, inequality, financial stability, and environmental harm and less sensitive to traditional macroeconomic concerns such as economic growth, inflation, and government debt. The results are remarkably consistent across scenarios and across models. 

---
# DiffuMeta: Algebraic Language Models for Inverse Design of Metamaterials via Diffusion Transformers 

**Authors**: Li Zheng, Siddhant Kumar, Dennis M. Kochmann  

**Link**: [PDF](https://arxiv.org/pdf/2507.15753)  

**Abstract**: Generative machine learning models have revolutionized material discovery by capturing complex structure-property relationships, yet extending these approaches to the inverse design of three-dimensional metamaterials remains limited by computational complexity and underexplored design spaces due to the lack of expressive representations. Here, we present DiffuMeta, a generative framework integrating diffusion transformers with a novel algebraic language representation, encoding 3D geometries as mathematical sentences. This compact, unified parameterization spans diverse topologies while enabling direct application of transformers to structural design. DiffuMeta leverages diffusion models to generate novel shell structures with precisely targeted stress-strain responses under large deformations, accounting for buckling and contact while addressing the inherent one-to-many mapping by producing diverse solutions. Uniquely, our approach enables simultaneous control over multiple mechanical objectives, including linear and nonlinear responses beyond training domains. Experimental validation of fabricated structures further confirms the efficacy of our approach for accelerated design of metamaterials and structures with tailored properties. 

---
# DialogueForge: LLM Simulation of Human-Chatbot Dialogue 

**Authors**: Ruizhe Zhu, Hao Zhu, Yaxuan Li, Syang Zhou, Shijing Cai, Malgorzata Lazuka, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2507.15752)  

**Abstract**: Collecting human-chatbot dialogues typically demands substantial manual effort and is time-consuming, which limits and poses challenges for research on conversational AI. In this work, we propose DialogueForge - a framework for generating AI-simulated conversations in human-chatbot style. To initialize each generated conversation, DialogueForge uses seed prompts extracted from real human-chatbot interactions. We test a variety of LLMs to simulate the human chatbot user, ranging from state-of-the-art proprietary models to small-scale open-source LLMs, and generate multi-turn dialogues tailored to specific tasks. In addition, we explore fine-tuning techniques to enhance the ability of smaller models to produce indistinguishable human-like dialogues. We evaluate the quality of the simulated conversations and compare different models using the UniEval and GTEval evaluation protocols. Our experiments show that large proprietary models (e.g., GPT-4o) generally outperform others in generating more realistic dialogues, while smaller open-source models (e.g., Llama, Mistral) offer promising performance with greater customization. We demonstrate that the performance of smaller models can be significantly improved by employing supervised fine-tuning techniques. Nevertheless, maintaining coherent and natural long-form human-like dialogues remains a common challenge across all models. 

---
# Explainable Anomaly Detection for Electric Vehicles Charging Stations 

**Authors**: Matteo Cederle, Andrea Mazzucco, Andrea Demartini, Eugenio Mazza, Eugenia Suriani, Federico Vitti, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2507.15718)  

**Abstract**: Electric vehicles (EV) charging stations are one of the critical infrastructures needed to support the transition to renewable-energy-based mobility, but ensuring their reliability and efficiency requires effective anomaly detection to identify irregularities in charging behavior. However, in such a productive scenario, it is also crucial to determine the underlying cause behind the detected anomalies. To achieve this goal, this study investigates unsupervised anomaly detection techniques for EV charging infrastructure, integrating eXplainable Artificial Intelligence techniques to enhance interpretability and uncover root causes of anomalies.
Using real-world sensors and charging session data, this work applies Isolation Forest to detect anomalies and employs the Depth-based Isolation Forest Feature Importance (DIFFI) method to identify the most important features contributing to such anomalies. The efficacy of the proposed approach is evaluated in a real industrial case. 

---
# BEnchmarking LLMs for Ophthalmology (BELO) for Ophthalmological Knowledge and Reasoning 

**Authors**: Sahana Srinivasan, Xuguang Ai, Thaddaeus Wai Soon Lo, Aidan Gilson, Minjie Zou, Ke Zou, Hyunjae Kim, Mingjia Yang, Krithi Pushpanathan, Samantha Yew, Wan Ting Loke, Jocelyn Goh, Yibing Chen, Yiming Kong, Emily Yuelei Fu, Michelle Ongyong Hui, Kristen Nwanyanwu, Amisha Dave, Kelvin Zhenghao Li, Chen-Hsin Sun, Mark Chia, Gabriel Dawei Yang, Wendy Meihua Wong, David Ziyou Chen, Dianbo Liu, Maxwell Singer, Fares Antaki, Lucian V Del Priore, Jost Jonas, Ron Adelman, Qingyu Chen, Yih-Chung Tham  

**Link**: [PDF](https://arxiv.org/pdf/2507.15717)  

**Abstract**: Current benchmarks evaluating large language models (LLMs) in ophthalmology are limited in scope and disproportionately prioritise accuracy. We introduce BELO (BEnchmarking LLMs for Ophthalmology), a standardized and comprehensive evaluation benchmark developed through multiple rounds of expert checking by 13 ophthalmologists. BELO assesses ophthalmology-related clinical accuracy and reasoning quality. Using keyword matching and a fine-tuned PubMedBERT model, we curated ophthalmology-specific multiple-choice-questions (MCQs) from diverse medical datasets (BCSC, MedMCQA, MedQA, BioASQ, and PubMedQA). The dataset underwent multiple rounds of expert checking. Duplicate and substandard questions were systematically removed. Ten ophthalmologists refined the explanations of each MCQ's correct answer. This was further adjudicated by three senior ophthalmologists. To illustrate BELO's utility, we evaluated six LLMs (OpenAI o1, o3-mini, GPT-4o, DeepSeek-R1, Llama-3-8B, and Gemini 1.5 Pro) using accuracy, macro-F1, and five text-generation metrics (ROUGE-L, BERTScore, BARTScore, METEOR, and AlignScore). In a further evaluation involving human experts, two ophthalmologists qualitatively reviewed 50 randomly selected outputs for accuracy, comprehensiveness, and completeness. BELO consists of 900 high-quality, expert-reviewed questions aggregated from five sources: BCSC (260), BioASQ (10), MedMCQA (572), MedQA (40), and PubMedQA (18). A public leaderboard has been established to promote transparent evaluation and reporting. Importantly, the BELO dataset will remain a hold-out, evaluation-only benchmark to ensure fair and reproducible comparisons of future models. 

---
# Is Large Language Model Performance on Reasoning Tasks Impacted by Different Ways Questions Are Asked? 

**Authors**: Seok Hwan Song, Mohna Chakraborty, Qi Li, Wallapak Tavanapong  

**Link**: [PDF](https://arxiv.org/pdf/2507.15707)  

**Abstract**: Large Language Models (LLMs) have been evaluated using diverse question types, e.g., multiple-choice, true/false, and short/long answers. This study answers an unexplored question about the impact of different question types on LLM accuracy on reasoning tasks. We investigate the performance of five LLMs on three different types of questions using quantitative and deductive reasoning tasks. The performance metrics include accuracy in the reasoning steps and choosing the final answer. Key Findings: (1) Significant differences exist in LLM performance across different question types. (2) Reasoning accuracy does not necessarily correlate with the final selection accuracy. (3) The number of options and the choice of words, influence LLM performance. 

---
# Compositional Understanding in Signaling Games 

**Authors**: David Peter Wallis Freeborn  

**Link**: [PDF](https://arxiv.org/pdf/2507.15706)  

**Abstract**: Receivers in standard signaling game models struggle with learning compositional information. Even when the signalers send compositional messages, the receivers do not interpret them compositionally. When information from one message component is lost or forgotten, the information from other components is also erased. In this paper I construct signaling game models in which genuine compositional understanding evolves. I present two new models: a minimalist receiver who only learns from the atomic messages of a signal, and a generalist receiver who learns from all of the available information. These models are in many ways simpler than previous alternatives, and allow the receivers to learn from the atomic components of messages. 

---
# CoLD: Counterfactually-Guided Length Debiasing for Process Reward Models 

**Authors**: Congmin Zheng, Jiachen Zhu, Jianghao Lin, Xinyi Dai, Yong Yu, Weinan Zhang, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15698)  

**Abstract**: Process Reward Models (PRMs) play a central role in evaluating and guiding multi-step reasoning in large language models (LLMs), especially for mathematical problem solving. However, we identify a pervasive length bias in existing PRMs: they tend to assign higher scores to longer reasoning steps, even when the semantic content and logical validity are unchanged. This bias undermines the reliability of reward predictions and leads to overly verbose outputs during inference. To address this issue, we propose CoLD(Counterfactually-Guided Length Debiasing), a unified framework that mitigates length bias through three components: an explicit length-penalty adjustment, a learned bias estimator trained to capture spurious length-related signals, and a joint training strategy that enforces length-invariance in reward predictions. Our approach is grounded in counterfactual reasoning and informed by causal graph analysis. Extensive experiments on MATH500 and GSM-Plus show that CoLD consistently reduces reward-length correlation, improves accuracy in step selection, and encourages more concise, logically valid reasoning. These results demonstrate the effectiveness and practicality of CoLD in improving the fidelity and robustness of PRMs. 

---
# LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression 

**Authors**: Wenjie Huang, Qi Yang, Shuting Xia, He Huang, Zhu Li, Yiling Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15686)  

**Abstract**: Existing AI-based point cloud compression methods struggle with dependence on specific training data distributions, which limits their real-world deployment. Implicit Neural Representation (INR) methods solve the above problem by encoding overfitted network parameters to the bitstream, resulting in more distribution-agnostic results. However, due to the limitation of encoding time and decoder size, current INR based methods only consider lossy geometry compression. In this paper, we propose the first INR based lossless point cloud geometry compression method called Lossless Implicit Neural Representations for Point Cloud Geometry Compression (LINR-PCGC). To accelerate encoding speed, we design a group of point clouds level coding framework with an effective network initialization strategy, which can reduce around 60% encoding time. A lightweight coding network based on multiscale SparseConv, consisting of scale context extraction, child node prediction, and model compression modules, is proposed to realize fast inference and compact decoder size. Experimental results show that our method consistently outperforms traditional and AI-based methods: for example, with the convergence time in the MVUB dataset, our method reduces the bitstream by approximately 21.21% compared to G-PCC TMC13v23 and 21.95% compared to SparsePCGC. Our project can be seen on this https URL. 

---
# Missing value imputation with adversarial random forests -- MissARF 

**Authors**: Pegah Golchian, Jan Kapar, David S. Watson, Marvin N. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2507.15681)  

**Abstract**: Handling missing values is a common challenge in biostatistical analyses, typically addressed by imputation methods. We propose a novel, fast, and easy-to-use imputation method called missing value imputation with adversarial random forests (MissARF), based on generative machine learning, that provides both single and multiple imputation. MissARF employs adversarial random forest (ARF) for density estimation and data synthesis. To impute a missing value of an observation, we condition on the non-missing values and sample from the estimated conditional distribution generated by ARF. Our experiments demonstrate that MissARF performs comparably to state-of-the-art single and multiple imputation methods in terms of imputation quality and fast runtime with no additional costs for multiple imputation. 

---
# SustainDiffusion: Optimising the Social and Environmental Sustainability of Stable Diffusion Models 

**Authors**: Giordano d'Aloisio, Tosin Fadahunsi, Jay Choy, Rebecca Moussa, Federica Sarro  

**Link**: [PDF](https://arxiv.org/pdf/2507.15663)  

**Abstract**: Background: Text-to-image generation models are widely used across numerous domains. Among these models, Stable Diffusion (SD) - an open-source text-to-image generation model - has become the most popular, producing over 12 billion images annually. However, the widespread use of these models raises concerns regarding their social and environmental sustainability.
Aims: To reduce the harm that SD models may have on society and the environment, we introduce SustainDiffusion, a search-based approach designed to enhance the social and environmental sustainability of SD models.
Method: SustainDiffusion searches the optimal combination of hyperparameters and prompt structures that can reduce gender and ethnic bias in generated images while also lowering the energy consumption required for image generation. Importantly, SustainDiffusion maintains image quality comparable to that of the original SD model.
Results: We conduct a comprehensive empirical evaluation of SustainDiffusion, testing it against six different baselines using 56 different prompts. Our results demonstrate that SustainDiffusion can reduce gender bias in SD3 by 68%, ethnic bias by 59%, and energy consumption (calculated as the sum of CPU and GPU energy) by 48%. Additionally, the outcomes produced by SustainDiffusion are consistent across multiple runs and can be generalised to various prompts.
Conclusions: With SustainDiffusion, we demonstrate how enhancing the social and environmental sustainability of text-to-image generation models is possible without fine-tuning or changing the model's architecture. 

---
# Towards Explainable Anomaly Detection in Shared Mobility Systems 

**Authors**: Elnur Isgandarov, Matteo Cederle, Federico Chiariotti, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2507.15643)  

**Abstract**: Shared mobility systems, such as bike-sharing networks, play a crucial role in urban transportation. Identifying anomalies in these systems is essential for optimizing operations, improving service reliability, and enhancing user experience. This paper presents an interpretable anomaly detection framework that integrates multi-source data, including bike-sharing trip records, weather conditions, and public transit availability. The Isolation Forest algorithm is employed for unsupervised anomaly detection, along with the Depth-based Isolation Forest Feature Importance (DIFFI) algorithm providing interpretability. Results show that station-level analysis offers a robust understanding of anomalies, highlighting the influence of external factors such as adverse weather and limited transit availability. Our findings contribute to improving decision-making in shared mobility operations. 

---
# Leveraging Context for Multimodal Fallacy Classification in Political Debates 

**Authors**: Alessio Pittiglio  

**Link**: [PDF](https://arxiv.org/pdf/2507.15641)  

**Abstract**: In this paper, we present our submission to the MM-ArgFallacy2025 shared task, which aims to advance research in multimodal argument mining, focusing on logical fallacies in political debates. Our approach uses pretrained Transformer-based models and proposes several ways to leverage context. In the fallacy classification subtask, our models achieved macro F1-scores of 0.4444 (text), 0.3559 (audio), and 0.4403 (multimodal). Our multimodal model showed performance comparable to the text-only model, suggesting potential for improvements. 

---
# Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training 

**Authors**: Kailai Yang, Xiao Liu, Lei Ji, Hao Li, Yeyun Gong, Peng Cheng, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15640)  

**Abstract**: Continual pre-training on small-scale task-specific data is an effective method for improving large language models in new target fields, yet it risks catastrophic forgetting of their original capabilities. A common solution is to re-weight training data mixtures from source and target fields on a domain space to achieve balanced performance. Previous domain reweighting strategies rely on manual designation with certain heuristics based on human intuition or empirical results. In this work, we prove that more general heuristics can be parameterized by proposing Data Mixing Agent, the first model-based, end-to-end framework that learns to re-weight domains. The agent learns generalizable heuristics through reinforcement learning on large quantities of data mixing trajectories with corresponding feedback from an evaluation environment. Experiments in continual pre-training on math reasoning show that Data Mixing Agent outperforms strong baselines in achieving balanced performance across source and target field benchmarks. Furthermore, it generalizes well across unseen source fields, target models, and domain spaces without retraining. Direct application to the code generation field also indicates its adaptability across target domains. Further analysis showcases the agents' well-aligned heuristics with human intuitions and their efficiency in achieving superior model performance with less source-field data. 

---
# Uncovering Critical Features for Deepfake Detection through the Lottery Ticket Hypothesis 

**Authors**: Lisan Al Amin, Md. Ismail Hossain, Thanh Thi Nguyen, Tasnim Jahan, Mahbubul Islam, Faisal Quader  

**Link**: [PDF](https://arxiv.org/pdf/2507.15636)  

**Abstract**: Recent advances in deepfake technology have created increasingly convincing synthetic media that poses significant challenges to information integrity and social trust. While current detection methods show promise, their underlying mechanisms remain poorly understood, and the large sizes of their models make them challenging to deploy in resource-limited environments. This study investigates the application of the Lottery Ticket Hypothesis (LTH) to deepfake detection, aiming to identify the key features crucial for recognizing deepfakes. We examine how neural networks can be efficiently pruned while maintaining high detection accuracy. Through extensive experiments with MesoNet, CNN-5, and ResNet-18 architectures on the OpenForensic and FaceForensics++ datasets, we find that deepfake detection networks contain winning tickets, i.e., subnetworks, that preserve performance even at substantial sparsity levels. Our results indicate that MesoNet retains 56.2% accuracy at 80% sparsity on the OpenForensic dataset, with only 3,000 parameters, which is about 90% of its baseline accuracy (62.6%). The results also show that our proposed LTH-based iterative magnitude pruning approach consistently outperforms one-shot pruning methods. Using Grad-CAM visualization, we analyze how pruned networks maintain their focus on critical facial regions for deepfake detection. Additionally, we demonstrate the transferability of winning tickets across datasets, suggesting potential for efficient, deployable deepfake detection systems. 

---
# Why can't Epidemiology be automated (yet)? 

**Authors**: David Bann, Ed Lowther, Liam Wright, Yevgeniya Kovalchuk  

**Link**: [PDF](https://arxiv.org/pdf/2507.15617)  

**Abstract**: Recent advances in artificial intelligence (AI) - particularly generative AI - present new opportunities to accelerate, or even automate, epidemiological research. Unlike disciplines based on physical experimentation, a sizable fraction of Epidemiology relies on secondary data analysis and thus is well-suited for such augmentation. Yet, it remains unclear which specific tasks can benefit from AI interventions or where roadblocks exist. Awareness of current AI capabilities is also mixed. Here, we map the landscape of epidemiological tasks using existing datasets - from literature review to data access, analysis, writing up, and dissemination - and identify where existing AI tools offer efficiency gains. While AI can increase productivity in some areas such as coding and administrative tasks, its utility is constrained by limitations of existing AI models (e.g. hallucinations in literature reviews) and human systems (e.g. barriers to accessing datasets). Through examples of AI-generated epidemiological outputs, including fully AI-generated papers, we demonstrate that recently developed agentic systems can now design and execute epidemiological analysis, albeit to varied quality (see this https URL). Epidemiologists have new opportunities to empirically test and benchmark AI systems; realising the potential of AI will require two-way engagement between epidemiologists and engineers. 

---
# Accelerating HEC-RAS: A Recurrent Neural Operator for Rapid River Forecasting 

**Authors**: Edward Holmberg, Pujan Pokhrel, Maximilian Zoch, Elias Ioup, Ken Pathak, Steven Sloan, Kendall Niles, Jay Ratcliff, Maik Flanagin, Christian Guetl, Julian Simeonov, Mahdi Abdelguerfi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15614)  

**Abstract**: Physics-based solvers like HEC-RAS provide high-fidelity river forecasts but are too computationally intensive for on-the-fly decision-making during flood events. The central challenge is to accelerate these simulations without sacrificing accuracy. This paper introduces a deep learning surrogate that treats HEC-RAS not as a solver but as a data-generation engine. We propose a hybrid, auto-regressive architecture that combines a Gated Recurrent Unit (GRU) to capture short-term temporal dynamics with a Geometry-Aware Fourier Neural Operator (Geo-FNO) to model long-range spatial dependencies along a river reach. The model learns underlying physics implicitly from a minimal eight-channel feature vector encoding dynamic state, static geometry, and boundary forcings extracted directly from native HEC-RAS files. Trained on 67 reaches of the Mississippi River Basin, the surrogate was evaluated on a year-long, unseen hold-out simulation. Results show the model achieves a strong predictive accuracy, with a median absolute stage error of 0.31 feet. Critically, for a full 67-reach ensemble forecast, our surrogate reduces the required wall-clock time from 139 minutes to 40 minutes, a speedup of nearly 3.5 times over the traditional solver. The success of this data-driven approach demonstrates that robust feature engineering can produce a viable, high-speed replacement for conventional hydraulic models, improving the computational feasibility of large-scale ensemble flood forecasting. 

---
# Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems 

**Authors**: Andrii Balashov, Olena Ponomarova, Xiaohua Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2507.15613)  

**Abstract**: Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place.
We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses. 

---
# Red-Team Multi-Agent Reinforcement Learning for Emergency Braking Scenario 

**Authors**: Yinsong Chen, Kaifeng Wang, Xiaoqiang Meng, Xueyuan Li, Zirui Li, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15587)  

**Abstract**: Current research on decision-making in safety-critical scenarios often relies on inefficient data-driven scenario generation or specific modeling approaches, which fail to capture corner cases in real-world contexts. To address this issue, we propose a Red-Team Multi-Agent Reinforcement Learning framework, where background vehicles with interference capabilities are treated as red-team agents. Through active interference and exploration, red-team vehicles can uncover corner cases outside the data distribution. The framework uses a Constraint Graph Representation Markov Decision Process, ensuring that red-team vehicles comply with safety rules while continuously disrupting the autonomous vehicles (AVs). A policy threat zone model is constructed to quantify the threat posed by red-team vehicles to AVs, inducing more extreme actions to increase the danger level of the scenario. Experimental results show that the proposed framework significantly impacts AVs decision-making safety and generates various corner cases. This method also offers a novel direction for research in safety-critical scenarios. 

---
# Unequal Voices: How LLMs Construct Constrained Queer Narratives 

**Authors**: Atreya Ghosal, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.15585)  

**Abstract**: One way social groups are marginalized in discourse is that the narratives told about them often default to a narrow, stereotyped range of topics. In contrast, default groups are allowed the full complexity of human existence. We describe the constrained representations of queer people in LLM generations in terms of harmful representations, narrow representations, and discursive othering and formulate hypotheses to test for these phenomena. Our results show that LLMs are significantly limited in their portrayals of queer personas. 

---
# GeMix: Conditional GAN-Based Mixup for Improved Medical Image Augmentation 

**Authors**: Hugo Carlesso, Maria Eliza Patulea, Moncef Garouani, Radu Tudor Ionescu, Josiane Mothe  

**Link**: [PDF](https://arxiv.org/pdf/2507.15577)  

**Abstract**: Mixup has become a popular augmentation strategy for image classification, yet its naive pixel-wise interpolation often produces unrealistic images that can hinder learning, particularly in high-stakes medical applications. We propose GeMix, a two-stage framework that replaces heuristic blending with a learned, label-aware interpolation powered by class-conditional GANs. First, a StyleGAN2-ADA generator is trained on the target dataset. During augmentation, we sample two label vectors from Dirichlet priors biased toward different classes and blend them via a Beta-distributed coefficient. Then, we condition the generator on this soft label to synthesize visually coherent images that lie along a continuous class manifold. We benchmark GeMix on the large-scale COVIDx-CT-3 dataset using three backbones (ResNet-50, ResNet-101, EfficientNet-B0). When combined with real data, our method increases macro-F1 over traditional mixup for all backbones, reducing the false negative rate for COVID-19 detection. GeMix is thus a drop-in replacement for pixel-space mixup, delivering stronger regularization and greater semantic fidelity, without disrupting existing training pipelines. We publicly release our code at this https URL to foster reproducibility and further research. 

---
# On the Role of AI in Managing Satellite Constellations: Insights from the ConstellAI Project 

**Authors**: Gregory F. Stock, Juan A. Fraire, Holger Hermanns, Jędrzej Mosiężny, Yusra Al-Khazraji, Julio Ramírez Molina, Evridiki V. Ntagiou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15574)  

**Abstract**: The rapid expansion of satellite constellations in near-Earth orbits presents significant challenges in satellite network management, requiring innovative approaches for efficient, scalable, and resilient operations. This paper explores the role of Artificial Intelligence (AI) in optimizing the operation of satellite mega-constellations, drawing from the ConstellAI project funded by the European Space Agency (ESA). A consortium comprising GMV GmbH, Saarland University, and Thales Alenia Space collaborates to develop AI-driven algorithms and demonstrates their effectiveness over traditional methods for two crucial operational challenges: data routing and resource allocation. In the routing use case, Reinforcement Learning (RL) is used to improve the end-to-end latency by learning from historical queuing latency, outperforming classical shortest path algorithms. For resource allocation, RL optimizes the scheduling of tasks across constellations, focussing on efficiently using limited resources such as battery and memory. Both use cases were tested for multiple satellite constellation configurations and operational scenarios, resembling the real-life spacecraft operations of communications and Earth observation satellites. This research demonstrates that RL not only competes with classical approaches but also offers enhanced flexibility, scalability, and generalizability in decision-making processes, which is crucial for the autonomous and intelligent management of satellite fleets. The findings of this activity suggest that AI can fundamentally alter the landscape of satellite constellation management by providing more adaptive, robust, and cost-effective solutions. 

---
# PhysGym: Benchmarking LLMs in Interactive Physics Discovery with Controlled Priors 

**Authors**: Yimeng Chen, Piotr Piȩkos, Mateusz Ostaszewski, Firas Laakom, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2507.15550)  

**Abstract**: Evaluating the scientific discovery capabilities of large language model based agents, particularly how they cope with varying environmental complexity and utilize prior knowledge, requires specialized benchmarks currently lacking in the landscape. To address this gap, we introduce PhysGym, a novel benchmark suite and simulation platform for rigorously assessing LLM-based scientific reasoning in interactive physics environments. PhysGym's primary contribution lies in its sophisticated control over the level of prior knowledge provided to the agent. This allows researchers to dissect agent performance along axes including the complexity of the problem and the prior knowledge levels. The benchmark comprises a suite of interactive simulations, where agents must actively probe environments, gather data sequentially under constraints and formulate hypotheses about underlying physical laws. PhysGym provides standardized evaluation protocols and metrics for assessing hypothesis accuracy and model fidelity. We demonstrate the benchmark's utility by presenting results from baseline LLMs, showcasing its ability to differentiate capabilities based on varying priors and task complexity. 

---
# RARE-UNet: Resolution-Aligned Routing Entry for Adaptive Medical Image Segmentation 

**Authors**: Simon Winther Albertsen, Hjalte Svaneborg Bjørnstrup, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15524)  

**Abstract**: Accurate segmentation is crucial for clinical applications, but existing models often assume fixed, high-resolution inputs and degrade significantly when faced with lower-resolution data in real-world scenarios. To address this limitation, we propose RARE-UNet, a resolution-aware multi-scale segmentation architecture that dynamically adapts its inference path to the spatial resolution of the input. Central to our design are multi-scale blocks integrated at multiple encoder depths, a resolution-aware routing mechanism, and consistency-driven training that aligns multi-resolution features with full-resolution representations. We evaluate RARE-UNet on two benchmark brain imaging tasks for hippocampus and tumor segmentation. Compared to standard UNet, its multi-resolution augmented variant, and nnUNet, our model achieves the highest average Dice scores of 0.84 and 0.65 across resolution, while maintaining consistent performance and significantly reduced inference time at lower resolutions. These results highlight the effectiveness and scalability of our architecture in achieving resolution-robust segmentation. The codes are available at: this https URL. 

---
# Off-Policy Corrected Reward Modeling for Reinforcement Learning from Human Feedback 

**Authors**: Johannes Ackermann, Takashi Ishida, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2507.15507)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) allows us to train models, such as language models (LMs), to follow complex human preferences. In RLHF for LMs, we first train an LM using supervised fine-tuning, sample pairs of responses, obtain human feedback, and use the resulting data to train a reward model (RM). RL methods are then used to train the LM to maximize the reward given by the RM. As training progresses, the responses generated by the LM no longer resemble the responses seen by the RM during training, leading to the RM becoming inaccurate. The score given by the RM keeps increasing, but the learned behavior no longer matches the human preferences. This issue is known as overoptimization. We investigate overoptimization from the point of view of distribution shift and show that the shift results in an inconsistent estimate of the RM parameters, leading to an inconsistent estimate of the policy gradient. We propose Off-Policy Corrected Reward Modeling (OCRM), which iteratively off-policy corrects the RM using importance weighting, without requiring new labels or samples. This results in a more accurate RM, which empirically leads to an improved final policy. We validate our approach in experiments with summarization and chatbot datasets and show that it performs significantly better than standard RLHF methods and baselines. Our implementation is available at this https URL 

---
# ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution 

**Authors**: Alexandru Coca, Mark Gaynor, Zhenxing Zhang, Jianpeng Cheng, Bo-Hsiang Tseng, Pete Boothroyd, Héctor Martinez Alonso, Diarmuid Ó Séaghdha, Anders Johannsen  

**Link**: [PDF](https://arxiv.org/pdf/2507.15501)  

**Abstract**: This work evaluates the potential of large language models (LLMs) to power digital assistants capable of complex action execution. These assistants rely on pre-trained programming knowledge to execute multi-step goals by composing objects and functions defined in assistant libraries into action execution programs. To achieve this, we develop ASPERA, a framework comprising an assistant library simulation and a human-assisted LLM data generation engine. Our engine allows developers to guide LLM generation of high-quality tasks consisting of complex user queries, simulation state and corresponding validation programs, tackling data availability and evaluation robustness challenges. Alongside the framework we release Asper-Bench, an evaluation dataset of 250 challenging tasks generated using ASPERA, which we use to show that program generation grounded in custom assistant libraries is a significant challenge to LLMs compared to dependency-free code generation. 

---
# GR-3 Technical Report 

**Authors**: Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, Hao Niu, Wenxuan Ou, Wanli Peng, Zeyu Ren, Haixin Shi, Jiawen Tian, Hongtao Wu, Xin Xiao, Yuyang Xiao, Jiafeng Xu, Yichu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15493)  

**Abstract**: We report our recent progress towards building generalist robot policies, the development of GR-3. GR-3 is a large-scale vision-language-action (VLA) model. It showcases exceptional capabilities in generalizing to novel objects, environments, and instructions involving abstract concepts. Furthermore, it can be efficiently fine-tuned with minimal human trajectory data, enabling rapid and cost-effective adaptation to new settings. GR-3 also excels in handling long-horizon and dexterous tasks, including those requiring bi-manual manipulation and mobile movement, showcasing robust and reliable performance. These capabilities are achieved through a multi-faceted training recipe that includes co-training with web-scale vision-language data, efficient fine-tuning from human trajectory data collected via VR devices, and effective imitation learning with robot trajectory data. In addition, we introduce ByteMini, a versatile bi-manual mobile robot designed with exceptional flexibility and reliability, capable of accomplishing a wide range of tasks when integrated with GR-3. Through extensive real-world experiments, we show GR-3 surpasses the state-of-the-art baseline method, $\pi_0$, on a wide variety of challenging tasks. We hope GR-3 can serve as a step towards building generalist robots capable of assisting humans in daily life. 

---
# The Constitutional Controller: Doubt-Calibrated Steering of Compliant Agents 

**Authors**: Simon Kohaut, Felix Divo, Navid Hamid, Benedict Flade, Julian Eggert, Devendra Singh Dhami, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2507.15478)  

**Abstract**: Ensuring reliable and rule-compliant behavior of autonomous agents in uncertain environments remains a fundamental challenge in modern robotics. Our work shows how neuro-symbolic systems, which integrate probabilistic, symbolic white-box reasoning models with deep learning methods, offer a powerful solution to this challenge. This enables the simultaneous consideration of explicit rules and neural models trained on noisy data, combining the strength of structured reasoning with flexible representations. To this end, we introduce the Constitutional Controller (CoCo), a novel framework designed to enhance the safety and reliability of agents by reasoning over deep probabilistic logic programs representing constraints such as those found in shared traffic spaces. Furthermore, we propose the concept of self-doubt, implemented as a probability density conditioned on doubt features such as travel velocity, employed sensors, or health factors. In a real-world aerial mobility study, we demonstrate CoCo's advantages for intelligent autonomous systems to learn appropriate doubts and navigate complex and uncertain environments safely and compliantly. 

---
# The Emergence of Deep Reinforcement Learning for Path Planning 

**Authors**: Thanh Thi Nguyen, Saeid Nahavandi, Imran Razzak, Dung Nguyen, Nhat Truong Pham, Quoc Viet Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.15469)  

**Abstract**: The increasing demand for autonomous systems in complex and dynamic environments has driven significant research into intelligent path planning methodologies. For decades, graph-based search algorithms, linear programming techniques, and evolutionary computation methods have served as foundational approaches in this domain. Recently, deep reinforcement learning (DRL) has emerged as a powerful method for enabling autonomous agents to learn optimal navigation strategies through interaction with their environments. This survey provides a comprehensive overview of traditional approaches as well as the recent advancements in DRL applied to path planning tasks, focusing on autonomous vehicles, drones, and robotic platforms. Key algorithms across both conventional and learning-based paradigms are categorized, with their innovations and practical implementations highlighted. This is followed by a thorough discussion of their respective strengths and limitations in terms of computational efficiency, scalability, adaptability, and robustness. The survey concludes by identifying key open challenges and outlining promising avenues for future research. Special attention is given to hybrid approaches that integrate DRL with classical planning techniques to leverage the benefits of both learning-based adaptability and deterministic reliability, offering promising directions for robust and resilient autonomous navigation. 

---
# The New LLM Bottleneck: A Systems Perspective on Latent Attention and Mixture-of-Experts 

**Authors**: Sungmin Yun, Seonyong Park, Hwayong Nam, Younjoo Lee, Gunjun Lee, Kwanhee Kyung, Sangpyo Kim, Nam Sung Kim, Jongmin Kim, Hyungyo Kim, Juhwan Cho, Seungmin Baek, Jung Ho Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2507.15465)  

**Abstract**: Computational workloads composing traditional Transformer models are starkly bifurcated. Multi-Head Attention (MHA) is memory-bound, with low arithmetic intensity, while feedforward layers are compute-bound. This dichotomy has long motivated research into specialized hardware to mitigate the MHA bottleneck.
This paper argues that recent architectural shifts, namely Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE), challenge the premise of specialized attention hardware. We make two key observations. First, the arithmetic intensity of MLA is over two orders of magnitude greater than that of MHA, shifting it close to a compute-bound regime well-suited for modern accelerators like GPUs. Second, by distributing MoE experts across a pool of accelerators, their arithmetic intensity can be tuned through batching to match that of the dense layers, creating a more balanced computational profile.
These findings reveal a diminishing need for specialized attention hardware. The central challenge for next-generation Transformers is no longer accelerating a single memory-bound layer. Instead, the focus must shift to designing balanced systems with sufficient compute, memory capacity, memory bandwidth, and high-bandwidth interconnects to manage the diverse demands of large-scale models. 

---
# Solving nonconvex Hamilton--Jacobi--Isaacs equations with PINN-based policy iteration 

**Authors**: Hee Jun Yang, Min Jung Kim, Yeoneung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.15455)  

**Abstract**: We propose a mesh-free policy iteration framework that combines classical dynamic programming with physics-informed neural networks (PINNs) to solve high-dimensional, nonconvex Hamilton--Jacobi--Isaacs (HJI) equations arising in stochastic differential games and robust control. The method alternates between solving linear second-order PDEs under fixed feedback policies and updating the controls via pointwise minimax optimization using automatic differentiation. Under standard Lipschitz and uniform ellipticity assumptions, we prove that the value function iterates converge locally uniformly to the unique viscosity solution of the HJI equation. The analysis establishes equi-Lipschitz regularity of the iterates, enabling provable stability and convergence without requiring convexity of the Hamiltonian. Numerical experiments demonstrate the accuracy and scalability of the method. In a two-dimensional stochastic path-planning game with a moving obstacle, our method matches finite-difference benchmarks with relative $L^2$-errors below %10^{-2}%. In five- and ten-dimensional publisher-subscriber differential games with anisotropic noise, the proposed approach consistently outperforms direct PINN solvers, yielding smoother value functions and lower residuals. Our results suggest that integrating PINNs with policy iteration is a practical and theoretically grounded method for solving high-dimensional, nonconvex HJI equations, with potential applications in robotics, finance, and multi-agent reinforcement learning. 

---
# ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting 

**Authors**: Ruijie Zhu, Mulin Yu, Linning Xu, Lihan Jiang, Yixuan Li, Tianzhu Zhang, Jiangmiao Pang, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2507.15454)  

**Abstract**: 3D Gaussian Splatting is renowned for its high-fidelity reconstructions and real-time novel view synthesis, yet its lack of semantic understanding limits object-level perception. In this work, we propose ObjectGS, an object-aware framework that unifies 3D scene reconstruction with semantic understanding. Instead of treating the scene as a unified whole, ObjectGS models individual objects as local anchors that generate neural Gaussians and share object IDs, enabling precise object-level reconstruction. During training, we dynamically grow or prune these anchors and optimize their features, while a one-hot ID encoding with a classification loss enforces clear semantic constraints. We show through extensive experiments that ObjectGS not only outperforms state-of-the-art methods on open-vocabulary and panoptic segmentation tasks, but also integrates seamlessly with applications like mesh extraction and scene editing. Project page: this https URL 

---
# EgoPrune: Efficient Token Pruning for Egomotion Video Reasoning in Embodied Agent 

**Authors**: Jiaao Li, Kaiyuan Li, Chen Gao, Yong Li, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.15428)  

**Abstract**: Egomotion videos are first-person recordings where the view changes continuously due to the agent's movement. As they serve as the primary visual input for embodied AI agents, making egomotion video reasoning more efficient is therefore essential for real-world deployment. Recent advances in vision-language models have enabled strong multimodal reasoning capabilities, but their computational cost remains prohibitive for long, redundant video inputs. Existing token pruning methods, typically designed for third-person videos, fail to leverage the spatiotemporal continuity and motion constraints inherent in egomotion settings. To address this, we propose EgoPrune, a training-free token pruning method tailored for egomotion video reasoning. EgoPrune comprises three components: a keyframe selector adapted from EmbodiedR for temporally efficient sampling; Perspective-Aware Redundancy Filtering (PARF), which aligns visual tokens using perspective transformations and removes redundant tokens; and a Maximal Marginal Relevance (MMR)-based token selector that jointly considers visual-text relevance and intra-frame diversity. Experiments on two egomotion video benchmarks show that EgoPrune consistently outperforms prior training-free methods across various pruning ratios while significantly reducing FLOPs, memory usage, and latency. Moreover, we deploy EgoPrune on an embodied agent equipped with a Jetson Orin NX 16GB edge device, demonstrating its real-world efficiency and suitability for on-device egomotion video reasoning. 

---
# Neuro-MSBG: An End-to-End Neural Model for Hearing Loss Simulation 

**Authors**: Hui-Guan Yuan, Ryandhimas E. Zezario, Shafique Ahmed, Hsin-Min Wang, Kai-Lung Hua, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15396)  

**Abstract**: Hearing loss simulation models are essential for hearing aid deployment. However, existing models have high computational complexity and latency, which limits real-time applications and lack direct integration with speech processing systems. To address these issues, we propose Neuro-MSBG, a lightweight end-to-end model with a personalized audiogram encoder for effective time-frequency modeling. Experiments show that Neuro-MSBG supports parallel inference and retains the intelligibility and perceptual quality of the original MSBG, with a Spearman's rank correlation coefficient (SRCC) of 0.9247 for Short-Time Objective Intelligibility (STOI) and 0.8671 for Perceptual Evaluation of Speech Quality (PESQ). Neuro-MSBG reduces simulation runtime by a factor of 46 (from 0.970 seconds to 0.021 seconds for a 1 second input), further demonstrating its efficiency and practicality. 

---
# PiMRef: Detecting and Explaining Ever-evolving Spear Phishing Emails with Knowledge Base Invariants 

**Authors**: Ruofan Liu, Yun Lin, Silas Yeo Shuen Yu, Xiwen Teoh, Zhenkai Liang, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.15393)  

**Abstract**: Phishing emails are a critical component of the cybercrime kill chain due to their wide reach and low cost. Their ever-evolving nature renders traditional rule-based and feature-engineered detectors ineffective in the ongoing arms race between attackers and defenders. The rise of large language models (LLMs) further exacerbates the threat, enabling attackers to craft highly convincing phishing emails at minimal cost.
This work demonstrates that LLMs can generate psychologically persuasive phishing emails tailored to victim profiles, successfully bypassing nearly all commercial and academic detectors. To defend against such threats, we propose PiMRef, the first reference-based phishing email detector that leverages knowledge-based invariants. Our core insight is that persuasive phishing emails often contain disprovable identity claims, which contradict real-world facts. PiMRef reframes phishing detection as an identity fact-checking task. Given an email, PiMRef (i) extracts the sender's claimed identity, (ii) verifies the legitimacy of the sender's domain against a predefined knowledge base, and (iii) detects call-to-action prompts that push user engagement. Contradictory claims are flagged as phishing indicators and serve as human-understandable explanations.
Compared to existing methods such as D-Fence, HelpHed, and ChatSpamDetector, PiMRef boosts precision by 8.8% with no loss in recall on standard benchmarks like Nazario and PhishPot. In a real-world evaluation of 10,183 emails across five university accounts over three years, PiMRef achieved 92.1% precision, 87.9% recall, and a median runtime of 0.05s, outperforming the state-of-the-art in both effectiveness and efficiency. 

---
# To Label or Not to Label: PALM -- A Predictive Model for Evaluating Sample Efficiency in Active Learning Models 

**Authors**: Julia Machnio, Mads Nielsen, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15381)  

**Abstract**: Active learning (AL) seeks to reduce annotation costs by selecting the most informative samples for labeling, making it particularly valuable in resource-constrained settings. However, traditional evaluation methods, which focus solely on final accuracy, fail to capture the full dynamics of the learning process. To address this gap, we propose PALM (Performance Analysis of Active Learning Models), a unified and interpretable mathematical model that characterizes AL trajectories through four key parameters: achievable accuracy, coverage efficiency, early-stage performance, and scalability. PALM provides a predictive description of AL behavior from partial observations, enabling the estimation of future performance and facilitating principled comparisons across different strategies. We validate PALM through extensive experiments on CIFAR-10/100 and ImageNet-50/100/200, covering a wide range of AL methods and self-supervised embeddings. Our results demonstrate that PALM generalizes effectively across datasets, budgets, and strategies, accurately predicting full learning curves from limited labeled data. Importantly, PALM reveals crucial insights into learning efficiency, data space coverage, and the scalability of AL methods. By enabling the selection of cost-effective strategies and predicting performance under tight budget constraints, PALM lays the basis for more systematic, reproducible, and data-efficient evaluation of AL in both research and real-world applications. The code is available at: this https URL. 

---
# Multi-beam Beamforming in RIS-aided MIMO Subject to Reradiation Mask Constraints -- Optimization and Machine Learning Design 

**Authors**: Shumin Wang, Hajar El Hassani, Marco Di Renzo, Marios Poulakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.15367)  

**Abstract**: Reconfigurable intelligent surfaces (RISs) are an emerging technology for improving spectral efficiency and reducing power consumption in future wireless systems. This paper investigates the joint design of the transmit precoding matrices and the RIS phase shift vector in a multi-user RIS-aided multiple-input multiple-output (MIMO) communication system. We formulate a max-min optimization problem to maximize the minimum achievable rate while considering transmit power and reradiation mask constraints. The achievable rate is simplified using the Arimoto-Blahut algorithm, and the problem is broken into quadratic programs with quadratic constraints (QPQC) sub-problems using an alternating optimization approach. To improve efficiency, we develop a model-based neural network optimization that utilizes the one-hot encoding for the angles of incidence and reflection. We address practical RIS limitations by using a greedy search algorithm to solve the optimization problem for discrete phase shifts. Simulation results demonstrate that the proposed methods effectively shape the multi-beam radiation pattern towards desired directions while satisfying reradiation mask constraints. The neural network design reduces the execution time, and the discrete phase shift scheme performs well with a small reduction of the beamforming gain by using only four phase shift levels. 

---
# EEG-based Epileptic Prediction via a Two-stage Channel-aware Set Transformer Network 

**Authors**: Ruifeng Zheng, Cong Chen, Shuang Wang, Yiming Liu, Lin You, Jindong Lu, Ruizhe Zhu, Guodao Zhang, Kejie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15364)  

**Abstract**: Epilepsy is a chronic, noncommunicable brain disorder, and sudden seizure onsets can significantly impact patients' quality of life and health. However, wearable seizure-predicting devices are still limited, partly due to the bulky size of EEG-collecting devices. To relieve the problem, we proposed a novel two-stage channel-aware Set Transformer Network that could perform seizure prediction with fewer EEG channel sensors. We also tested a seizure-independent division method which could prevent the adjacency of training and test data. Experiments were performed on the CHB-MIT dataset which includes 22 patients with 88 merged seizures. The mean sensitivity before channel selection was 76.4% with a false predicting rate (FPR) of 0.09/hour. After channel selection, dominant channels emerged in 20 out of 22 patients; the average number of channels was reduced to 2.8 from 18; and the mean sensitivity rose to 80.1% with an FPR of 0.11/hour. Furthermore, experimental results on the seizure-independent division supported our assertion that a more rigorous seizure-independent division should be used for patients with abundant EEG recordings. 

---
# Latent Space Synergy: Text-Guided Data Augmentation for Direct Diffusion Biomedical Segmentation 

**Authors**: Muhammad Aqeel, Maham Nazir, Zanxi Ruan, Francesco Setti  

**Link**: [PDF](https://arxiv.org/pdf/2507.15361)  

**Abstract**: Medical image segmentation suffers from data scarcity, particularly in polyp detection where annotation requires specialized expertise. We present SynDiff, a framework combining text-guided synthetic data generation with efficient diffusion-based segmentation. Our approach employs latent diffusion models to generate clinically realistic synthetic polyps through text-conditioned inpainting, augmenting limited training data with semantically diverse samples. Unlike traditional diffusion methods requiring iterative denoising, we introduce direct latent estimation enabling single-step inference with T x computational speedup. On CVC-ClinicDB, SynDiff achieves 96.0% Dice and 92.9% IoU while maintaining real-time capability suitable for clinical deployment. The framework demonstrates that controlled synthetic augmentation improves segmentation robustness without distribution shift. SynDiff bridges the gap between data-hungry deep learning models and clinical constraints, offering an efficient solution for deployment in resourcelimited medical settings. 

---
# Metaphor and Large Language Models: When Surface Features Matter More than Deep Understanding 

**Authors**: Elisa Sanchez-Bayona, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2507.15357)  

**Abstract**: This paper presents a comprehensive evaluation of the capabilities of Large Language Models (LLMs) in metaphor interpretation across multiple datasets, tasks, and prompt configurations. Although metaphor processing has gained significant attention in Natural Language Processing (NLP), previous research has been limited to single-dataset evaluations and specific task settings, often using artificially constructed data through lexical replacement. We address these limitations by conducting extensive experiments using diverse publicly available datasets with inference and metaphor annotations, focusing on Natural Language Inference (NLI) and Question Answering (QA) tasks. The results indicate that LLMs' performance is more influenced by features like lexical overlap and sentence length than by metaphorical content, demonstrating that any alleged emergent abilities of LLMs to understand metaphorical language are the result of a combination of surface-level features, in-context learning, and linguistic knowledge. This work provides critical insights into the current capabilities and limitations of LLMs in processing figurative language, highlighting the need for more realistic evaluation frameworks in metaphor interpretation tasks. Data and code are publicly available. 

---
# Scaling Decentralized Learning with FLock 

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.15349)  

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data. 

---
# StackTrans: From Large Language Model to Large Pushdown Automata Model 

**Authors**: Kechi Zhang, Ge Li, Jia Li, Huangzhao Zhang, Yihong Dong, Jia Li, Jingjing Xu, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.15343)  

**Abstract**: The Transformer architecture has emerged as a landmark advancement within the broad field of artificial intelligence, effectively catalyzing the advent of large language models (LLMs). However, despite its remarkable capabilities and the substantial progress it has facilitated, the Transformer architecture still has some limitations. One such intrinsic limitation is its inability to effectively capture the Chomsky hierarchy, such as regular expressions or deterministic context-free grammars. Drawing inspiration from pushdown automata, which efficiently resolve deterministic context-free grammars using stacks, we propose StackTrans to address the aforementioned issue within LLMs. Unlike previous approaches that modify the attention computation, StackTrans explicitly incorporates hidden state stacks between Transformer layers. This design maintains compatibility with existing frameworks like flash-attention. Specifically, our design features stack operations -- such as pushing and popping hidden states -- that are differentiable and can be learned in an end-to-end manner. Our comprehensive evaluation spans benchmarks for both Chomsky hierarchies and large-scale natural languages. Across these diverse tasks, StackTrans consistently outperforms standard Transformer models and other baselines. We have successfully scaled StackTrans up from 360M to 7B parameters. In particular, our from-scratch pretrained model StackTrans-360M outperforms several larger open-source LLMs with 2-3x more parameters, showcasing its superior efficiency and reasoning capability. 

---
# MedSR-Impact: Transformer-Based Super-Resolution for Lung CT Segmentation, Radiomics, Classification, and Prognosis 

**Authors**: Marc Boubnovski Martell, Kristofer Linton-Reid, Mitchell Chen, Sumeet Hindocha, Benjamin Hunter, Marco A. Calzado, Richard Lee, Joram M. Posma, Eric O. Aboagye  

**Link**: [PDF](https://arxiv.org/pdf/2507.15340)  

**Abstract**: High-resolution volumetric computed tomography (CT) is essential for accurate diagnosis and treatment planning in thoracic diseases; however, it is limited by radiation dose and hardware costs. We present the Transformer Volumetric Super-Resolution Network (\textbf{TVSRN-V2}), a transformer-based super-resolution (SR) framework designed for practical deployment in clinical lung CT analysis. Built from scalable components, including Through-Plane Attention Blocks (TAB) and Swin Transformer V2 -- our model effectively reconstructs fine anatomical details in low-dose CT volumes and integrates seamlessly with downstream analysis pipelines. We evaluate its effectiveness on three critical lung cancer tasks -- lobe segmentation, radiomics, and prognosis -- across multiple clinical cohorts. To enhance robustness across variable acquisition protocols, we introduce pseudo-low-resolution augmentation, simulating scanner diversity without requiring private data. TVSRN-V2 demonstrates a significant improvement in segmentation accuracy (+4\% Dice), higher radiomic feature reproducibility, and enhanced predictive performance (+0.06 C-index and AUC). These results indicate that SR-driven recovery of structural detail significantly enhances clinical decision support, positioning TVSRN-V2 as a well-engineered, clinically viable system for dose-efficient imaging and quantitative analysis in real-world CT workflows. 

---
# Beyond Model Base Selection: Weaving Knowledge to Master Fine-grained Neural Network Design 

**Authors**: Jialiang Wang, Hanmo Liu, Shimin Di, Zhili Wang, Jiachuan Wang, Lei Chen, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15336)  

**Abstract**: Database systems have recently advocated for embedding machine learning (ML) capabilities, offering declarative model queries over large, managed model repositories, thereby circumventing the huge computational overhead of traditional ML-based algorithms in automated neural network model selection. Pioneering database studies aim to organize existing benchmark repositories as model bases (MB), querying them for the model records with the highest performance estimation metrics for given tasks. However, this static model selection practice overlooks the fine-grained, evolving relational dependencies between diverse task queries and model architecture variations, resulting in suboptimal matches and failing to further refine the model effectively. To fill the model refinement gap in database research, we propose M-DESIGN, a curated model knowledge base (MKB) pipeline for mastering neural network refinement by adaptively weaving prior insights about model architecture modification. First, we propose a knowledge weaving engine that reframes model refinement as an adaptive query problem over task metadata. Given a user's task query, M-DESIGN quickly matches and iteratively refines candidate models by leveraging a graph-relational knowledge schema that explicitly encodes data properties, architecture variations, and pairwise performance deltas as joinable relations. This schema supports fine-grained relational analytics over architecture tweaks and drives a predictive query planner that can detect and adapt to out-of-distribution (OOD) tasks. We instantiate M-DESIGN for graph analytics tasks, where our model knowledge base enriches existing benchmarks with structured metadata covering 3 graph tasks and 22 graph datasets, contributing data records of 67,760 graph models. Empirical results demonstrate that M-DESIGN delivers the optimal model in 26 of 33 data-task pairs within limited budgets. 

---
# ExDD: Explicit Dual Distribution Learning for Surface Defect Detection via Diffusion Synthesis 

**Authors**: Muhammad Aqeel, Federico Leonardi, Francesco Setti  

**Link**: [PDF](https://arxiv.org/pdf/2507.15335)  

**Abstract**: Industrial defect detection systems face critical limitations when confined to one-class anomaly detection paradigms, which assume uniform outlier distributions and struggle with data scarcity in realworld manufacturing environments. We present ExDD (Explicit Dual Distribution), a novel framework that transcends these limitations by explicitly modeling dual feature distributions. Our approach leverages parallel memory banks that capture the distinct statistical properties of both normality and anomalous patterns, addressing the fundamental flaw of uniform outlier assumptions. To overcome data scarcity, we employ latent diffusion models with domain-specific textual conditioning, generating in-distribution synthetic defects that preserve industrial context. Our neighborhood-aware ratio scoring mechanism elegantly fuses complementary distance metrics, amplifying signals in regions exhibiting both deviation from normality and similarity to known defect patterns. Experimental validation on KSDD2 demonstrates superior performance (94.2% I-AUROC, 97.7% P-AUROC), with optimal augmentation at 100 synthetic samples. 

---
# Butterfly Effects in Toolchains: A Comprehensive Analysis of Failed Parameter Filling in LLM Tool-Agent Systems 

**Authors**: Qian Xiong, Yuekai Huang, Ziyou Jiang, Zhiyuan Chang, Yujia Zheng, Tianhao Li, Mingyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15296)  

**Abstract**: The emergence of the tool agent paradigm has broadened the capability boundaries of the Large Language Model (LLM), enabling it to complete more complex tasks. However, the effectiveness of this paradigm is limited due to the issue of parameter failure during its execution. To explore this phenomenon and propose corresponding suggestions, we first construct a parameter failure taxonomy in this paper. We derive five failure categories from the invocation chain of a mainstream tool agent. Then, we explore the correlation between three different input sources and failure categories by applying 15 input perturbation methods to the input. Experimental results show that parameter name hallucination failure primarily stems from inherent LLM limitations, while issues with input sources mainly cause other failure patterns. To improve the reliability and effectiveness of tool-agent interactions, we propose corresponding improvement suggestions, including standardizing tool return formats, improving error feedback mechanisms, and ensuring parameter consistency. 

---
# EndoControlMag: Robust Endoscopic Vascular Motion Magnification with Periodic Reference Resetting and Hierarchical Tissue-aware Dual-Mask Contro 

**Authors**: An Wanga, Rulin Zhou, Mengya Xu, Yiru Ye, Longfei Gou, Yiting Chang, Hao Chen, Chwee Ming Lim, Jiankun Wang, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.15292)  

**Abstract**: Visualizing subtle vascular motions in endoscopic surgery is crucial for surgical precision and decision-making, yet remains challenging due to the complex and dynamic nature of surgical scenes. To address this, we introduce EndoControlMag, a training-free, Lagrangian-based framework with mask-conditioned vascular motion magnification tailored to endoscopic environments. Our approach features two key modules: a Periodic Reference Resetting (PRR) scheme that divides videos into short overlapping clips with dynamically updated reference frames to prevent error accumulation while maintaining temporal coherence, and a Hierarchical Tissue-aware Magnification (HTM) framework with dual-mode mask dilation. HTM first tracks vessel cores using a pretrained visual tracking model to maintain accurate localization despite occlusions and view changes. It then applies one of two adaptive softening strategies to surrounding tissues: motion-based softening that modulates magnification strength proportional to observed tissue displacement, or distance-based exponential decay that simulates biomechanical force attenuation. This dual-mode approach accommodates diverse surgical scenarios-motion-based softening excels with complex tissue deformations while distance-based softening provides stability during unreliable optical flow conditions. We evaluate EndoControlMag on our EndoVMM24 dataset spanning four different surgery types and various challenging scenarios, including occlusions, instrument disturbance, view changes, and vessel deformations. Quantitative metrics, visual assessments, and expert surgeon evaluations demonstrate that EndoControlMag significantly outperforms existing methods in both magnification accuracy and visual quality while maintaining robustness across challenging surgical conditions. The code, dataset, and video results are available at this https URL. 

---
# Preferential subspace identification (PSID) with forward-backward smoothing 

**Authors**: Omid G. Sani, Maryam M. Shanechi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15288)  

**Abstract**: System identification methods for multivariate time-series, such as neural and behavioral recordings, have been used to build models for predicting one from the other. For example, Preferential Subspace Identification (PSID) builds a state-space model of a primary time-series (e.g., neural activity) to optimally predict a secondary time-series (e.g., behavior). However, PSID focuses on optimal prediction using past primary data, even though in offline applications, better estimation can be achieved by incorporating concurrent data (filtering) or all available data (smoothing). Here, we extend PSID to enable optimal filtering and smoothing. First, we show that the presence of a secondary signal makes it possible to uniquely identify a model with an optimal Kalman update step (to enable filtering) from a family of otherwise equivalent state-space models. Our filtering solution augments PSID with a reduced-rank regression step that directly learns the optimal gain required for the update step from data. We refer to this extension of PSID as PSID with filtering. Second, inspired by two-filter Kalman smoother formulations, we develop a novel forward-backward PSID smoothing algorithm where we first apply PSID with filtering and then apply it again in the reverse time direction on the residuals of the filtered secondary signal. We validate our methods on simulated data, showing that our approach recovers the ground-truth model parameters for filtering, and achieves optimal filtering and smoothing decoding performance of the secondary signal that matches the ideal performance of the true underlying model. This work provides a principled framework for optimal linear filtering and smoothing in the two-signal setting, significantly expanding the toolkit for analyzing dynamic interactions in multivariate time-series. 

---
# Mixture of Autoencoder Experts Guidance using Unlabeled and Incomplete Data for Exploration in Reinforcement Learning 

**Authors**: Elias Malomgré, Pieter Simoens  

**Link**: [PDF](https://arxiv.org/pdf/2507.15287)  

**Abstract**: Recent trends in Reinforcement Learning (RL) highlight the need for agents to learn from reward-free interactions and alternative supervision signals, such as unlabeled or incomplete demonstrations, rather than relying solely on explicit reward maximization. Additionally, developing generalist agents that can adapt efficiently in real-world environments often requires leveraging these reward-free signals to guide learning and behavior. However, while intrinsic motivation techniques provide a means for agents to seek out novel or uncertain states in the absence of explicit rewards, they are often challenged by dense reward environments or the complexity of high-dimensional state and action spaces. Furthermore, most existing approaches rely directly on the unprocessed intrinsic reward signals, which can make it difficult to shape or control the agent's exploration effectively. We propose a framework that can effectively utilize expert demonstrations, even when they are incomplete and imperfect. By applying a mapping function to transform the similarity between an agent's state and expert data into a shaped intrinsic reward, our method allows for flexible and targeted exploration of expert-like behaviors. We employ a Mixture of Autoencoder Experts to capture a diverse range of behaviors and accommodate missing information in demonstrations. Experiments show our approach enables robust exploration and strong performance in both sparse and dense reward environments, even when demonstrations are sparse or incomplete. This provides a practical framework for RL in realistic settings where optimal data is unavailable and precise reward control is needed. 

---
# A Novel Self-Evolution Framework for Large Language Models 

**Authors**: Haoran Sun, Zekun Zhang, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.15281)  

**Abstract**: The capabilities of Large Language Models (LLMs) are limited to some extent by pre-training, so some researchers optimize LLMs through post-training. Existing post-training strategies, such as memory-based retrieval or preference optimization, improve user alignment yet fail to enhance the model's domain cognition. To bridge this gap, we propose a novel Dual-Phase Self-Evolution (DPSE) framework that jointly optimizes user preference adaptation and domain-specific competence. DPSE introduces a Censor module to extract multi-dimensional interaction signals and estimate satisfaction scores, which guide structured data expansion via topic-aware and preference-driven strategies. These expanded datasets support a two-stage fine-tuning pipeline: supervised domain grounding followed by frequency-aware preference optimization. Experiments across general NLP benchmarks and long-term dialogue tasks demonstrate that DPSE consistently outperforms Supervised Fine-Tuning, Preference Optimization, and Memory-Augmented baselines. Ablation studies validate the contribution of each module. In this way, our framework provides an autonomous path toward continual self-evolution of LLMs. 

---
# A2TTS: TTS for Low Resource Indian Languages 

**Authors**: Ayush Singh Bhadoriya, Abhishek Nikunj Shinde, Isha Pandey, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2507.15272)  

**Abstract**: We present a speaker conditioned text-to-speech (TTS) system aimed at addressing challenges in generating speech for unseen speakers and supporting diverse Indian languages. Our method leverages a diffusion-based TTS architecture, where a speaker encoder extracts embeddings from short reference audio samples to condition the DDPM decoder for multispeaker generation. To further enhance prosody and naturalness, we employ a cross-attention based duration prediction mechanism that utilizes reference audio, enabling more accurate and speaker consistent timing. This results in speech that closely resembles the target speaker while improving duration modeling and overall expressiveness. Additionally, to improve zero-shot generation, we employed classifier free guidance, allowing the system to generate speech more near speech for unknown speakers. Using this approach, we trained language-specific speaker-conditioned models. Using the IndicSUPERB dataset for multiple Indian languages such as Bengali, Gujarati, Hindi, Marathi, Malayalam, Punjabi and Tamil. 

---
# Conditional Video Generation for High-Efficiency Video Compression 

**Authors**: Fangqiu Yi, Jingyu Xu, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15269)  

**Abstract**: Perceptual studies demonstrate that conditional diffusion models excel at reconstructing video content aligned with human visual perception. Building on this insight, we propose a video compression framework that leverages conditional diffusion models for perceptually optimized reconstruction. Specifically, we reframe video compression as a conditional generation task, where a generative model synthesizes video from sparse, yet informative signals. Our approach introduces three key modules: (1) Multi-granular conditioning that captures both static scene structure and dynamic spatio-temporal cues; (2) Compact representations designed for efficient transmission without sacrificing semantic richness; (3) Multi-condition training with modality dropout and role-aware embeddings, which prevent over-reliance on any single modality and enhance robustness. Extensive experiments show that our method significantly outperforms both traditional and neural codecs on perceptual quality metrics such as Fréchet Video Distance (FVD) and LPIPS, especially under high compression ratios. 

---
# Optimal Transceiver Design in Over-the-Air Federated Distillation 

**Authors**: Zihao Hu, Jia Yan, Ying-Jun Angela Zhang, Jun Zhang, Khaled B. Letaief  

**Link**: [PDF](https://arxiv.org/pdf/2507.15256)  

**Abstract**: The rapid proliferation and growth of artificial intelligence (AI) has led to the development of federated learning (FL). FL allows wireless devices (WDs) to cooperatively learn by sharing only local model parameters, without needing to share the entire dataset. However, the emergence of large AI models has made existing FL approaches inefficient, due to the significant communication overhead required. In this paper, we propose a novel over-the-air federated distillation (FD) framework by synergizing the strength of FL and knowledge distillation to avoid the heavy local model transmission. Instead of sharing the model parameters, only the WDs' model outputs, referred to as knowledge, are shared and aggregated over-the-air by exploiting the superposition property of the multiple-access channel. We shall study the transceiver design in over-the-air FD, aiming to maximize the learning convergence rate while meeting the power constraints of the transceivers. The main challenge lies in the intractability of the learning performance analysis, as well as the non-convex nature and the optimization spanning the whole FD training period. To tackle this problem, we first derive an analytical expression of the convergence rate in over-the-air FD. Then, the closed-form optimal solutions of the WDs' transmit power and the estimator for over-the-air aggregation are obtained given the receiver combining strategy. Accordingly, we put forth an efficient approach to find the optimal receiver beamforming vector via semidefinite relaxation. We further prove that there is no optimality gap between the original and relaxed problem for the receiver beamforming design. Numerical results will show that the proposed over-the-air FD approach achieves a significant reduction in communication overhead, with only a minor compromise in testing accuracy compared to conventional FL benchmarks. 

---
# MEETI: A Multimodal ECG Dataset from MIMIC-IV-ECG with Signals, Images, Features and Interpretations 

**Authors**: Deyun Zhang, Xiang Lan, Shijia Geng, Qinghao Zhao, Sumei Fan, Mengling Feng, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.15255)  

**Abstract**: Electrocardiogram (ECG) plays a foundational role in modern cardiovascular care, enabling non-invasive diagnosis of arrhythmias, myocardial ischemia, and conduction disorders. While machine learning has achieved expert-level performance in ECG interpretation, the development of clinically deployable multimodal AI systems remains constrained, primarily due to the lack of publicly available datasets that simultaneously incorporate raw signals, diagnostic images, and interpretation text. Most existing ECG datasets provide only single-modality data or, at most, dual modalities, making it difficult to build models that can understand and integrate diverse ECG information in real-world settings. To address this gap, we introduce MEETI (MIMIC-IV-Ext ECG-Text-Image), the first large-scale ECG dataset that synchronizes raw waveform data, high-resolution plotted images, and detailed textual interpretations generated by large language models. In addition, MEETI includes beat-level quantitative ECG parameters extracted from each lead, offering structured parameters that support fine-grained analysis and model interpretability. Each MEETI record is aligned across four components: (1) the raw ECG waveform, (2) the corresponding plotted image, (3) extracted feature parameters, and (4) detailed interpretation text. This alignment is achieved using consistent, unique identifiers. This unified structure supports transformer-based multimodal learning and supports fine-grained, interpretable reasoning about cardiac health. By bridging the gap between traditional signal analysis, image-based interpretation, and language-driven understanding, MEETI established a robust foundation for the next generation of explainable, multimodal cardiovascular AI. It offers the research community a comprehensive benchmark for developing and evaluating ECG-based AI systems. 

---
# User Head Movement-Predictive XR in Immersive H2M Collaborations over Future Enterprise Networks 

**Authors**: Sourav Mondal, Elaine Wong  

**Link**: [PDF](https://arxiv.org/pdf/2507.15254)  

**Abstract**: The evolution towards future generation of mobile systems and fixed wireless networks is primarily driven by the urgency to support high-bandwidth and low-latency services across various vertical sectors. This endeavor is fueled by smartphones as well as technologies like industrial internet of things, extended reality (XR), and human-to-machine (H2M) collaborations for fostering industrial and social revolutions like Industry 4.0/5.0 and Society 5.0. To ensure an ideal immersive experience and avoid cyber-sickness for users in all the aforementioned usage scenarios, it is typically challenging to synchronize XR content from a remote machine to a human collaborator according to their head movements across a large geographic span in real-time over communication networks. Thus, we propose a novel H2M collaboration scheme where the human's head movements are predicted ahead with highly accurate models like bidirectional long short-term memory networks to orient the machine's camera in advance. We validate that XR frame size varies in accordance with the human's head movements and predict the corresponding bandwidth requirements from the machine's camera to propose a human-machine coordinated dynamic bandwidth allocation (HMC-DBA) scheme. Through extensive simulations, we show that end-to-end latency and jitter requirements of XR frames are satisfied with much lower bandwidth consumption over enterprise networks like Fiber-To-The-Room-Business. Furthermore, we show that better efficiency in network resource utilization is achieved by employing our proposed HMC-DBA over state-of-the-art schemes. 

---
# Spatio-Temporal Demand Prediction for Food Delivery Using Attention-Driven Graph Neural Networks 

**Authors**: Rabia Latief Bhat, Iqra Altaf Gillani  

**Link**: [PDF](https://arxiv.org/pdf/2507.15246)  

**Abstract**: Accurate demand forecasting is critical for enhancing the efficiency and responsiveness of food delivery platforms, where spatial heterogeneity and temporal fluctuations in order volumes directly influence operational decisions. This paper proposes an attention-based Graph Neural Network framework that captures spatial-temporal dependencies by modeling the food delivery environment as a graph. In this graph, nodes represent urban delivery zones, while edges reflect spatial proximity and inter-regional order flow patterns derived from historical data. The attention mechanism dynamically weighs the influence of neighboring zones, enabling the model to focus on the most contextually relevant areas during prediction. Temporal trends are jointly learned alongside spatial interactions, allowing the model to adapt to evolving demand patterns. Extensive experiments on real-world food delivery datasets demonstrate the superiority of the proposed model in forecasting future order volumes with high accuracy. The framework offers a scalable and adaptive solution to support proactive fleet positioning, resource allocation, and dispatch optimization in urban food delivery operations. 

---
# SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search 

**Authors**: Xiaofeng Shi, Yuduo Li, Qian Kou, Longbin Yu, Jinxin Xie, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15245)  

**Abstract**: Recent advances in large language models (LLMs) have opened new opportunities for academic literature retrieval. However, existing systems often rely on rigid pipelines and exhibit limited reasoning capabilities. We introduce SPAR, a multi-agent framework that incorporates RefChain-based query decomposition and query evolution to enable more flexible and effective search. To facilitate systematic evaluation, we also construct SPARBench, a challenging benchmark with expert-annotated relevance labels. Experimental results demonstrate that SPAR substantially outperforms strong baselines, achieving up to +56% F1 on AutoScholar and +23% F1 on SPARBench over the best-performing baseline. Together, SPAR and SPARBench provide a scalable, interpretable, and high-performing foundation for advancing research in scholarly retrieval. Code and data will be available at: this https URL 

---
# Cross-Domain Few-Shot Learning with Coalescent Projections and Latent Space Reservation 

**Authors**: Naeem Paeedeh, Mahardhika Pratama, Wolfgang Mayer, Jimmy Cao, Ryszard Kowlczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.15243)  

**Abstract**: Despite the progress in Cross-Domain Few-Shot Learning (CD-FSL), a model pre-trained with DINO combined with a prototypical classifier outperforms the latest SOTA methods. A crucial limitation that needs to be overcome is that updating too many parameters of the transformers leads to overfitting due to the scarcity of labeled samples. To address this challenge, we propose a new concept, Coalescent Projection (CP), as an effective successor to soft prompts. Additionally, we propose a novel pseudo-class generation method combined with Self-Supervised Transformations (SSTs) that relies solely on the base domain to prepare the network for encountering unseen samples from different domains. The proposed method exhibits its effectiveness in comprehensive experiments on the extreme domain shift scenario of the BSCD-FSL benchmark. Our code is published at this https URL. 

---
# SimdBench: Benchmarking Large Language Models for SIMD-Intrinsic Code Generation 

**Authors**: Yibo He, Shuoran Zhao, Jiaming Huang, Yingjie Fu, Hao Yu, Cunjian Huang, Tao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.15224)  

**Abstract**: SIMD (Single Instruction Multiple Data) instructions and their compiler intrinsics are widely supported by modern processors to accelerate performance-critical tasks. SIMD intrinsic programming, a trade-off between coding productivity and high performance, is widely used in the development of mainstream performance-critical libraries and daily computing tasks. Large Language Models (LLMs), which have demonstrated strong and comprehensive capabilities in code generation, show promise in assisting programmers with the challenges of SIMD intrinsic programming. However, existing code-generation benchmarks focus on only scalar code, and it is unclear how LLMs perform in generating vectorized code using SIMD intrinsics. To fill this gap, we propose SimdBench, the first code benchmark specifically designed for SIMD-intrinsic code generation, comprising 136 carefully crafted tasks and targeting five representative SIMD intrinsics: SSE (x86 Streaming SIMD Extension), AVX (x86 Advanced Vector Extension), Neon (ARM Advanced SIMD Extension), SVE (ARM Scalable Vector Extension), and RVV (RISC-V Vector Extension). We conduct a systematic evaluation (measuring both correctness and performance) of 18 representative LLMs on SimdBench, resulting in a series of novel and insightful findings. Our evaluation results demonstrate that LLMs exhibit a universal decrease in pass@k during SIMD-intrinsic code generation compared to scalar-code generation. Our in-depth analysis highlights promising directions for the further advancement of LLMs in the challenging domain of SIMD-intrinsic code generation. SimdBench is fully open source at this https URL to benefit the broader research community. 

---
# PromptArmor: Simple yet Effective Prompt Injection Defenses 

**Authors**: Tianneng Shi, Kaijie Zhu, Zhun Wang, Yuqi Jia, Will Cai, Weida Liang, Haonan Wang, Hend Alzahrani, Joshua Lu, Kenji Kawaguchi, Basel Alomair, Xuandong Zhao, William Yang Wang, Neil Gong, Wenbo Guo, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.15219)  

**Abstract**: Despite their potential, recent research has demonstrated that LLM agents are vulnerable to prompt injection attacks, where malicious prompts are injected into the agent's input, causing it to perform an attacker-specified task rather than the intended task provided by the user. In this paper, we present PromptArmor, a simple yet effective defense against prompt injection attacks. Specifically, PromptArmor prompts an off-the-shelf LLM to detect and remove potential injected prompts from the input before the agent processes it. Our results show that PromptArmor can accurately identify and remove injected prompts. For example, using GPT-4o, GPT-4.1, or o4-mini, PromptArmor achieves both a false positive rate and a false negative rate below 1% on the AgentDojo benchmark. Moreover, after removing injected prompts with PromptArmor, the attack success rate drops to below 1%. We also demonstrate PromptArmor's effectiveness against adaptive attacks and explore different strategies for prompting an LLM. We recommend that PromptArmor be adopted as a standard baseline for evaluating new defenses against prompt injection attacks. 

---
# Long-Short Distance Graph Neural Networks and Improved Curriculum Learning for Emotion Recognition in Conversation 

**Authors**: Xinran Li, Xiujuan Xu, Jiaqi Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15205)  

**Abstract**: Emotion Recognition in Conversation (ERC) is a practical and challenging task. This paper proposes a novel multimodal approach, the Long-Short Distance Graph Neural Network (LSDGNN). Based on the Directed Acyclic Graph (DAG), it constructs a long-distance graph neural network and a short-distance graph neural network to obtain multimodal features of distant and nearby utterances, respectively. To ensure that long- and short-distance features are as distinct as possible in representation while enabling mutual influence between the two modules, we employ a Differential Regularizer and incorporate a BiAffine Module to facilitate feature interaction. In addition, we propose an Improved Curriculum Learning (ICL) to address the challenge of data imbalance. By computing the similarity between different emotions to emphasize the shifts in similar emotions, we design a "weighted emotional shift" metric and develop a difficulty measurer, enabling a training process that prioritizes learning easy samples before harder ones. Experimental results on the IEMOCAP and MELD datasets demonstrate that our model outperforms existing benchmarks. 

---
# A Study of Anatomical Priors for Deep Learning-Based Segmentation of Pheochromocytoma in Abdominal CT 

**Authors**: Tanjin Taher Toma, Tejas Sudharshan Mathai, Bikash Santra, Pritam Mukherjee, Jianfei Liu, Wesley Jong, Darwish Alabyad, Vivek Batheja, Abhishek Jha, Mayank Patel, Darko Pucar, Jayadira del Rivero, Karel Pacak, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2507.15193)  

**Abstract**: Accurate segmentation of pheochromocytoma (PCC) in abdominal CT scans is essential for tumor burden estimation, prognosis, and treatment planning. It may also help infer genetic clusters, reducing reliance on expensive testing. This study systematically evaluates anatomical priors to identify configurations that improve deep learning-based PCC segmentation. We employed the nnU-Net framework to evaluate eleven annotation strategies for accurate 3D segmentation of pheochromocytoma, introducing a set of novel multi-class schemes based on organ-specific anatomical priors. These priors were derived from adjacent organs commonly surrounding adrenal tumors (e.g., liver, spleen, kidney, aorta, adrenal gland, and pancreas), and were compared against a broad body-region prior used in previous work. The framework was trained and tested on 105 contrast-enhanced CT scans from 91 patients at the NIH Clinical Center. Performance was measured using Dice Similarity Coefficient (DSC), Normalized Surface Distance (NSD), and instance-wise F1 score. Among all strategies, the Tumor + Kidney + Aorta (TKA) annotation achieved the highest segmentation accuracy, significantly outperforming the previously used Tumor + Body (TB) annotation across DSC (p = 0.0097), NSD (p = 0.0110), and F1 score (25.84% improvement at an IoU threshold of 0.5), measured on a 70-30 train-test split. The TKA model also showed superior tumor burden quantification (R^2 = 0.968) and strong segmentation across all genetic subtypes. In five-fold cross-validation, TKA consistently outperformed TB across IoU thresholds (0.1 to 0.5), reinforcing its robustness and generalizability. These findings highlight the value of incorporating relevant anatomical context in deep learning models to achieve precise PCC segmentation, supporting clinical assessment and longitudinal monitoring. 

---
# Can LLMs Generate User Stories and Assess Their Quality? 

**Authors**: Giovanni Quattrocchi, Liliana Pasquale, Paola Spoletini, Luciano Baresi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15157)  

**Abstract**: Requirements elicitation is still one of the most challenging activities of the requirements engineering process due to the difficulty requirements analysts face in understanding and translating complex needs into concrete requirements. In addition, specifying high-quality requirements is crucial, as it can directly impact the quality of the software to be developed. Although automated tools allow for assessing the syntactic quality of requirements, evaluating semantic metrics (e.g., language clarity, internal consistency) remains a manual and time-consuming activity. This paper explores how LLMs can help automate requirements elicitation within agile frameworks, where requirements are defined as user stories (US). We used 10 state-of-the-art LLMs to investigate their ability to generate US automatically by emulating customer interviews. We evaluated the quality of US generated by LLMs, comparing it with the quality of US generated by humans (domain experts and students). We also explored whether and how LLMs can be used to automatically evaluate the semantic quality of US. Our results indicate that LLMs can generate US similar to humans in terms of coverage and stylistic quality, but exhibit lower diversity and creativity. Although LLM-generated US are generally comparable in quality to those created by humans, they tend to meet the acceptance quality criteria less frequently, regardless of the scale of the LLM model. Finally, LLMs can reliably assess the semantic quality of US when provided with clear evaluation criteria and have the potential to reduce human effort in large-scale assessments. 

---
# Constraint-aware Learning of Probabilistic Sequential Models for Multi-Label Classification 

**Authors**: Mykhailo Buleshnyi, Anna Polova, Zsolt Zombori, Michael Benedikt  

**Link**: [PDF](https://arxiv.org/pdf/2507.15156)  

**Abstract**: We investigate multi-label classification involving large sets of labels, where the output labels may be known to satisfy some logical constraints. We look at an architecture in which classifiers for individual labels are fed into an expressive sequential model, which produces a joint distribution. One of the potential advantages for such an expressive model is its ability to modelling correlations, as can arise from constraints. We empirically demonstrate the ability of the architecture both to exploit constraints in training and to enforce constraints at inference time. 

---
# What Level of Automation is "Good Enough"? A Benchmark of Large Language Models for Meta-Analysis Data Extraction 

**Authors**: Lingbo Li, Anuradha Mathrani, Teo Susnjak  

**Link**: [PDF](https://arxiv.org/pdf/2507.15152)  

**Abstract**: Automating data extraction from full-text randomised controlled trials (RCTs) for meta-analysis remains a significant challenge. This study evaluates the practical performance of three LLMs (Gemini-2.0-flash, Grok-3, GPT-4o-mini) across tasks involving statistical results, risk-of-bias assessments, and study-level characteristics in three medical domains: hypertension, diabetes, and orthopaedics. We tested four distinct prompting strategies (basic prompting, self-reflective prompting, model ensemble, and customised prompts) to determine how to improve extraction quality. All models demonstrate high precision but consistently suffer from poor recall by omitting key information. We found that customised prompts were the most effective, boosting recall by up to 15\%. Based on this analysis, we propose a three-tiered set of guidelines for using LLMs in data extraction, matching data types to appropriate levels of automation based on task complexity and risk. Our study offers practical advice for automating data extraction in real-world meta-analyses, balancing LLM efficiency with expert oversight through targeted, task-specific automation. 

---
# Performance Analysis of Post-Training Quantization for CNN-based Conjunctival Pallor Anemia Detection 

**Authors**: Sebastian A. Cruz Romero, Wilfredo E. Lugo Beauchamp  

**Link**: [PDF](https://arxiv.org/pdf/2507.15151)  

**Abstract**: Anemia is a widespread global health issue, particularly among young children in low-resource settings. Traditional methods for anemia detection often require expensive equipment and expert knowledge, creating barriers to early and accurate diagnosis. To address these challenges, we explore the use of deep learning models for detecting anemia through conjunctival pallor, focusing on the CP-AnemiC dataset, which includes 710 images from children aged 6-59 months. The dataset is annotated with hemoglobin levels, gender, age and other demographic data, enabling the development of machine learning models for accurate anemia detection. We use the MobileNet architecture as a backbone, known for its efficiency in mobile and embedded vision applications, and fine-tune our model end-to-end using data augmentation techniques and a cross-validation strategy. Our model implementation achieved an accuracy of 0.9313, a precision of 0.9374, and an F1 score of 0.9773 demonstrating strong performance on the dataset. To optimize the model for deployment on edge devices, we performed post-training quantization, evaluating the impact of different bit-widths (FP32, FP16, INT8, and INT4) on model performance. Preliminary results suggest that while FP16 quantization maintains high accuracy (0.9250), precision (0.9370), and F1 Score (0.9377), more aggressive quantization (INT8 and INT4) leads to significant performance degradation. Overall, our study supports further exploration of quantization schemes and hardware optimizations to assess trade-offs between model size, inference time, and diagnostic accuracy in mobile healthcare applications. 

---
# Design of an Edge-based Portable EHR System for Anemia Screening in Remote Health Applications 

**Authors**: Sebastian A. Cruz Romero, Misael J. Mercado Hernandez, Samir Y. Ali Rivera, Jorge A. Santiago Fernandez, Wilfredo E. Lugo Beauchamp  

**Link**: [PDF](https://arxiv.org/pdf/2507.15146)  

**Abstract**: The design of medical systems for remote, resource-limited environments faces persistent challenges due to poor interoperability, lack of offline support, and dependency on costly infrastructure. Many existing digital health solutions neglect these constraints, limiting their effectiveness for frontline health workers in underserved regions. This paper presents a portable, edge-enabled Electronic Health Record platform optimized for offline-first operation, secure patient data management, and modular diagnostic integration. Running on small-form factor embedded devices, it provides AES-256 encrypted local storage with optional cloud synchronization for interoperability. As a use case, we integrated a non-invasive anemia screening module leveraging fingernail pallor analysis. Trained on 250 patient cases (27\% anemia prevalence) with KDE-balanced data, the Random Forest model achieved a test RMSE of 1.969 g/dL and MAE of 1.490 g/dL. A severity-based model reached 79.2\% sensitivity. To optimize performance, a YOLOv8n-based nail bed detector was quantized to INT8, reducing inference latency from 46.96 ms to 21.50 ms while maintaining mAP@0.5 at 0.995. The system emphasizes low-cost deployment, modularity, and data privacy compliance (HIPAA/GDPR), addressing critical barriers to digital health adoption in disconnected settings. Our work demonstrates a scalable approach to enhance portable health information systems and support frontline healthcare in underserved regions. 

---
# A Case Against Implicit Standards: Homophone Normalization in Machine Translation for Languages that use the Ge'ez Script 

**Authors**: Hellina Hailu Nigatu, Atnafu Lambebo Tonja, Henok Biadglign Ademtew, Hizkel Mitiku Alemayehu, Negasi Haile Abadi, Tadesse Destaw Belay, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2507.15142)  

**Abstract**: Homophone normalization, where characters that have the same sound in a writing script are mapped to one character, is a pre-processing step applied in Amharic Natural Language Processing (NLP) literature. While this may improve performance reported by automatic metrics, it also results in models that are not able to understand different forms of writing in a single language. Further, there might be impacts in transfer learning, where models trained on normalized data do not generalize well to other languages. In this paper, we experiment with monolingual training and cross-lingual transfer to understand the impacts of normalization on languages that use the Ge'ez script. We then propose a post-inference intervention in which normalization is applied to model predictions instead of training data. With our simple scheme of post-inference normalization, we show that we can achieve an increase in BLEU score of up to 1.03 while preserving language features in training. Our work contributes to the broader discussion on technology-facilitated language change and calls for more language-aware interventions. 

---
# AnalogFed: Federated Discovery of Analog Circuit Topologies with Generative AI 

**Authors**: Qiufeng Li, Shu Hong, Jian Gao, Xuan Zhang, Tian Lan, Weidong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15104)  

**Abstract**: Recent breakthroughs in AI/ML offer exciting opportunities to revolutionize analog design automation through data-driven approaches. In particular, researchers are increasingly fascinated by harnessing the power of generative AI to automate the discovery of novel analog circuit topologies. Unlocking the full potential of generative AI in these data-driven discoveries requires access to large and diverse this http URL, there is a significant barrier in the analog domain--Analog circuit design is inherently proprietary, involving not only confidential circuit structures but also the underlying commercial semiconductor processes. As a result, current generative AI research is largely confined to individual researchers who construct small, narrowly focused private datasets. This fragmentation severely limits collaborative innovation and impedes progress across the research community. To address these challenges, we propose AnalogFed. AnalogFed enables collaborative topology discovery across decentralized clients (e.g., individual researchers or institutions) without requiring the sharing of raw private data. To make this vision practical, we introduce a suite of techniques tailored to the unique challenges of applying FedL in analog design--from generative model development and data heterogeneity handling to privacy-preserving strategies that ensure both flexibility and security for circuit designers and semiconductor manufacturers. Extensive experiments across varying client counts and dataset sizes demonstrate that AnalogFed achieves performance comparable to centralized baselines--while maintaining strict data privacy. Specifically, the generative AI model within AnalogFed achieves state-of-the-art efficiency and scalability in the design of analog circuit topologies. 

---
# Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference? 

**Authors**: Chathuri Jayaweera, Brianna Yanqui, Bonnie Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2507.15100)  

**Abstract**: Natural Language Inference (NLI) is the task of determining the semantic entailment of a premise for a given hypothesis. The task aims to develop systems that emulate natural human inferential processes where commonsense knowledge plays a major role. However, existing commonsense resources lack sufficient coverage for a variety of premise-hypothesis pairs. This study explores the potential of Large Language Models as commonsense knowledge generators for NLI along two key dimensions: their reliability in generating such knowledge and the impact of that knowledge on prediction accuracy. We adapt and modify existing metrics to assess LLM factuality and consistency in generating in this context. While explicitly incorporating commonsense knowledge does not consistently improve overall results, it effectively helps distinguish entailing instances and moderately improves distinguishing contradictory and neutral inferences. 

---
# BleedOrigin: Dynamic Bleeding Source Localization in Endoscopic Submucosal Dissection via Dual-Stage Detection and Tracking 

**Authors**: Mengya Xu, Rulin Zhou, An Wang, Chaoyang Lyu, Zhen Li, Ning Zhong, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.15094)  

**Abstract**: Intraoperative bleeding during Endoscopic Submucosal Dissection (ESD) poses significant risks, demanding precise, real-time localization and continuous monitoring of the bleeding source for effective hemostatic intervention. In particular, endoscopists have to repeatedly flush to clear blood, allowing only milliseconds to identify bleeding sources, an inefficient process that prolongs operations and elevates patient risks. However, current Artificial Intelligence (AI) methods primarily focus on bleeding region segmentation, overlooking the critical need for accurate bleeding source detection and temporal tracking in the challenging ESD environment, which is marked by frequent visual obstructions and dynamic scene changes. This gap is widened by the lack of specialized datasets, hindering the development of robust AI-assisted guidance systems. To address these challenges, we introduce BleedOrigin-Bench, the first comprehensive ESD bleeding source dataset, featuring 1,771 expert-annotated bleeding sources across 106,222 frames from 44 procedures, supplemented with 39,755 pseudo-labeled frames. This benchmark covers 8 anatomical sites and 6 challenging clinical scenarios. We also present BleedOrigin-Net, a novel dual-stage detection-tracking framework for the bleeding source localization in ESD procedures, addressing the complete workflow from bleeding onset detection to continuous spatial tracking. We compare with widely-used object detection models (YOLOv11/v12), multimodal large language models, and point tracking methods. Extensive evaluation demonstrates state-of-the-art performance, achieving 96.85% frame-level accuracy ($\pm\leq8$ frames) for bleeding onset detection, 70.24% pixel-level accuracy ($\leq100$ px) for initial source detection, and 96.11% pixel-level accuracy ($\leq100$ px) for point tracking. 

---
# Evaluation of Coding Schemes for Transformer-based Gene Sequence Modeling 

**Authors**: Chenlei Gong, Yuanhe Tian, Lei Mao, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.15087)  

**Abstract**: Currently, many studies view DNA sequences as a special type of language and utilize Transformers to model them. These studies use fixed-length k-mer segmentation and BPE subword tokenization but lack a systematic evaluation to determine which is superior. We compare k-mer segmentation with k=1,3,4,5,6, a 4,096-token BPE vocabulary, and three positional encoding methods-sinusoidal, AliBi, and RoPE. Each configuration is trained from scratch in 3, 6, 12, and 24-layer Transformer encoders and evaluated on GUE benchmark dataset. In general, BPE delivers higher and more stable performance across tasks by compressing frequent motifs into variable-length tokens, reducing sequence length, and improving model generalization. RoPE excels at capturing periodic motifs and extrapolating to long sequences, while AliBi also performs well on tasks driven by local dependencies. In terms of depth, we observe significant gains when increasing layers from 3 to 12, with only marginal improvements or slight overfitting at 24 layers. This study provides practical guidance for designing tokenization and positional encoding in DNA Transformer models. 

---
# Robust Control with Gradient Uncertainty 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15082)  

**Abstract**: We introduce a novel extension to robust control theory that explicitly addresses uncertainty in the value function's gradient, a form of uncertainty endemic to applications like reinforcement learning where value functions are approximated. We formulate a zero-sum dynamic game where an adversary perturbs both system dynamics and the value function gradient, leading to a new, highly nonlinear partial differential equation: the Hamilton-Jacobi-Bellman-Isaacs Equation with Gradient Uncertainty (GU-HJBI). We establish its well-posedness by proving a comparison principle for its viscosity solutions under a uniform ellipticity condition. Our analysis of the linear-quadratic (LQ) case yields a key insight: we prove that the classical quadratic value function assumption fails for any non-zero gradient uncertainty, fundamentally altering the problem structure. A formal perturbation analysis characterizes the non-polynomial correction to the value function and the resulting nonlinearity of the optimal control law, which we validate with numerical studies. Finally, we bridge theory to practice by proposing a novel Gradient-Uncertainty-Robust Actor-Critic (GURAC) algorithm, accompanied by an empirical study demonstrating its effectiveness in stabilizing training. This work provides a new direction for robust control, holding significant implications for fields where function approximation is common, including reinforcement learning and computational finance. 

---
# NavVI: A Telerobotic Simulation with Multimodal Feedback for Visually Impaired Navigation in Warehouse Environments 

**Authors**: Maisha Maimuna, Minhaz Bin Farukee, Sama Nikanfar, Mahfuza Siddiqua, Ayon Roy, Fillia Makedon  

**Link**: [PDF](https://arxiv.org/pdf/2507.15072)  

**Abstract**: Industrial warehouses are congested with moving forklifts, shelves and personnel, making robot teleoperation particularly risky and demanding for blind and low-vision (BLV) operators. Although accessible teleoperation plays a key role in inclusive workforce participation, systematic research on its use in industrial environments is limited, and few existing studies barely address multimodal guidance designed for BLV users. We present a novel multimodal guidance simulator that enables BLV users to control a mobile robot through a high-fidelity warehouse environment while simultaneously receiving synchronized visual, auditory, and haptic feedback. The system combines a navigation mesh with regular re-planning so routes remain accurate avoiding collisions as forklifts and human avatars move around the warehouse. Users with low vision are guided with a visible path line towards destination; navigational voice cues with clockwise directions announce upcoming turns, and finally proximity-based haptic feedback notifies the users of static and moving obstacles in the path. This real-time, closed-loop system offers a repeatable testbed and algorithmic reference for accessible teleoperation research. The simulator's design principles can be easily adapted to real robots due to the alignment of its navigation, speech, and haptic modules with commercial hardware, supporting rapid feasibility studies and deployment of inclusive telerobotic tools in actual warehouses. 

---
# Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback 

**Authors**: Yiyuan Yang, Zichuan Liu, Lei Song, Kai Ying, Zhiguang Wang, Tom Bamford, Svitlana Vyetrenko, Jiang Bian, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2507.15066)  

**Abstract**: Time series anomaly detection is critical across various domains, yet current approaches often limit analysis to mere binary anomaly classification without detailed categorization or further explanatory reasoning. To address these limitations, we propose a novel task, Time-series Reasoning for Anomaly (Time-RA) that transforms classical time series anomaly detection from a discriminative into a generative, reasoning-intensive task leveraging Large Language Models (LLMs). Also, we introduce the first real-world multimodal benchmark dataset, RATs40K, explicitly annotated for anomaly reasoning, comprising approximately 40,000 samples across 10 real-world domains. Each sample includes numeric time series data, contextual text information, and visual representations, each annotated with fine-grained categories (14 types for univariate anomalies and 6 for multivariate anomalies) and structured explanatory reasoning. We develop a sophisticated annotation framework utilizing ensemble-generated labels refined through GPT-4-driven feedback, ensuring accuracy and interpretability. Extensive benchmarking of LLMs and multimodal LLMs demonstrates the capabilities and limitations of current models, highlighting the critical role of supervised fine-tuning. Our dataset and task pave the way for significant advancements in interpretable time series anomaly detection and reasoning. 

---
# StableAnimator++: Overcoming Pose Misalignment and Face Distortion for Human Image Animation 

**Authors**: Shuyuan Tu, Zhen Xing, Xintong Han, Zhi-Qi Cheng, Qi Dai, Chong Luo, Zuxuan Wu, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15064)  

**Abstract**: Current diffusion models for human image animation often struggle to maintain identity (ID) consistency, especially when the reference image and driving video differ significantly in body size or position. We introduce StableAnimator++, the first ID-preserving video diffusion framework with learnable pose alignment, capable of generating high-quality videos conditioned on a reference image and a pose sequence without any post-processing. Building upon a video diffusion model, StableAnimator++ contains carefully designed modules for both training and inference, striving for identity consistency. In particular, StableAnimator++ first uses learnable layers to predict the similarity transformation matrices between the reference image and the driven poses via injecting guidance from Singular Value Decomposition (SVD). These matrices align the driven poses with the reference image, mitigating misalignment to a great extent. StableAnimator++ then computes image and face embeddings using off-the-shelf encoders, refining the face embeddings via a global content-aware Face Encoder. To further maintain ID, we introduce a distribution-aware ID Adapter that counteracts interference caused by temporal layers while preserving ID via distribution alignment. During the inference stage, we propose a novel Hamilton-Jacobi-Bellman (HJB) based face optimization integrated into the denoising process, guiding the diffusion trajectory for enhanced facial fidelity. Experiments on benchmarks show the effectiveness of StableAnimator++ both qualitatively and quantitatively. 

---
# Touch in the Wild: Learning Fine-Grained Manipulation with a Portable Visuo-Tactile Gripper 

**Authors**: Xinyue Zhu, Binghao Huang, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15062)  

**Abstract**: Handheld grippers are increasingly used to collect human demonstrations due to their ease of deployment and versatility. However, most existing designs lack tactile sensing, despite the critical role of tactile feedback in precise manipulation. We present a portable, lightweight gripper with integrated tactile sensors that enables synchronized collection of visual and tactile data in diverse, real-world, and in-the-wild settings. Building on this hardware, we propose a cross-modal representation learning framework that integrates visual and tactile signals while preserving their distinct characteristics. The learning procedure allows the emergence of interpretable representations that consistently focus on contacting regions relevant for physical interactions. When used for downstream manipulation tasks, these representations enable more efficient and effective policy learning, supporting precise robotic manipulation based on multimodal feedback. We validate our approach on fine-grained tasks such as test tube insertion and pipette-based fluid transfer, demonstrating improved accuracy and robustness under external disturbances. Our project page is available at this https URL . 

---
# WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization 

**Authors**: Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15061)  

**Abstract**: The advent of Large Language Model (LLM)-powered agents has revolutionized artificial intelligence by enabling solutions to complex, open-ended tasks through web-based information-seeking (IS) capabilities. The scarcity of high-quality training data has limited the development of IS agents. Existing approaches typically adopt an information-driven paradigm that first collects web data and then generates questions based on the retrieval. However, this may lead to inconsistency between information structure and reasoning structure, question and answer. To mitigate, we propose a formalization-driven IS data synthesis framework WebShaper to construct a dataset. WebShaper systematically formalizes IS tasks through set theory. Central to the formalization is the concept of Knowledge Projections (KP), which enables precise control over reasoning structure by KP operation compositions. During synthesis, we begin by creating seed tasks, then use a multi-step expansion process. At each step, an agentic Expander expands the current formal question more complex with retrieval and validation tools based on our formalization. We train our model on the synthesized dataset. Experiment results demonstrate that WebShaper achieves state-of-the-art performance among open-sourced IS agents on GAIA and WebWalkerQA benchmarks. 

---
# The hunt for new pulsating ultraluminous X-ray sources: a clustering approach 

**Authors**: Nicolò Oreste Pinciroli Vago, Roberta Amato, Matteo Imbrogno, GianLuca Israel, Andrea Belfiore, Konstantinos Kovlakas, Piero Fraternali, Mario Pasquato  

**Link**: [PDF](https://arxiv.org/pdf/2507.15032)  

**Abstract**: The discovery of fast and variable coherent signals in a handful of ultraluminous X-ray sources (ULXs) testifies to the presence of super-Eddington accreting neutron stars, and drastically changed the understanding of the ULX class. Our capability of discovering pulsations in ULXs is limited, among others, by poor statistics. However, catalogues and archives of high-energy missions contain information which can be used to identify new candidate pulsating ULXs (PULXs). The goal of this research is to single out candidate PULXs among those ULXs which have not shown pulsations due to an unfavourable combination of factors. We applied an AI approach to an updated database of ULXs detected by XMM-Newton. We first used an unsupervised clustering algorithm to sort out sources with similar characteristics into two clusters. Then, the sample of known PULX observations has been used to set the separation threshold between the two clusters and to identify the one containing the new candidate PULXs. We found that only a few criteria are needed to assign the membership of an observation to one of the two clusters. The cluster of new candidate PULXs counts 85 unique sources for 355 observations, with $\sim$85% of these new candidates having multiple observations. A preliminary timing analysis found no new pulsations for these candidates. This work presents a sample of new candidate PULXs observed by XMM-Newton, the properties of which are similar (in a multi-dimensional phase space) to those of the known PULXs, despite the absence of pulsations in their light curves. While this result is a clear example of the predictive power of AI-based methods, it also highlights the need for high-statistics observational data to reveal coherent signals from the sources in this sample and thus validate the robustness of the approach. 

---
# Survey of GenAI for Automotive Software Development: From Requirements to Executable Code 

**Authors**: Nenad Petrovic, Vahid Zolfaghari, Andre Schamschurko, Sven Kirchner, Fengjunjie Pan, Chengdng Wu, Nils Purschke, Aleksei Velsh, Krzysztof Lebioda, Yinglei Song, Yi Zhang, Lukasz Mazur, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2507.15025)  

**Abstract**: Adoption of state-of-art Generative Artificial Intelligence (GenAI) aims to revolutionize many industrial areas by reducing the amount of human intervention needed and effort for handling complex underlying processes. Automotive software development is considered to be a significant area for GenAI adoption, taking into account lengthy and expensive procedures, resulting from the amount of requirements and strict standardization. In this paper, we explore the adoption of GenAI for various steps of automotive software development, mainly focusing on requirements handling, compliance aspects and code generation. Three GenAI-related technologies are covered within the state-of-art: Large Language Models (LLMs), Retrieval Augmented Generation (RAG), Vision Language Models (VLMs), as well as overview of adopted prompting techniques in case of code generation. Additionally, we also derive a generalized GenAI-aided automotive software development workflow based on our findings from this literature review. Finally, we include a summary of a survey outcome, which was conducted among our automotive industry partners regarding the type of GenAI tools used for their daily work activities. 

---
# The Rise of AI Teammates in Software Engineering (SE) 3.0: How Autonomous Coding Agents Are Reshaping Software Engineering 

**Authors**: Hao Li, Haoxiang Zhang, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2507.15003)  

**Abstract**: The future of software engineering--SE 3.0--is unfolding with the rise of AI teammates: autonomous, goal-driven systems collaborating with human developers. Among these, autonomous coding agents are especially transformative, now actively initiating, reviewing, and evolving code at scale. This paper introduces AIDev, the first large-scale dataset capturing how such agents operate in the wild. Spanning over 456,000 pull requests by five leading agents--OpenAI Codex, Devin, GitHub Copilot, Cursor, and Claude Code--across 61,000 repositories and 47,000 developers, AIDev provides an unprecedented empirical foundation for studying autonomous teammates in software development.
Unlike prior work that has largely theorized the rise of AI-native software engineering, AIDev offers structured, open data to support research in benchmarking, agent readiness, optimization, collaboration modeling, and AI governance. The dataset includes rich metadata on PRs, authorship, review timelines, code changes, and integration outcomes--enabling exploration beyond synthetic benchmarks like SWE-bench. For instance, although agents often outperform humans in speed, their PRs are accepted less frequently, revealing a trust and utility gap. Furthermore, while agents accelerate code submission--one developer submitted as many PRs in three days as they had in three years--these are structurally simpler (via code complexity metrics).
We envision AIDev as a living resource: extensible, analyzable, and ready for the SE and AI communities. Grounding SE 3.0 in real-world evidence, AIDev enables a new generation of research into AI-native workflows and supports building the next wave of symbiotic human-AI collaboration. The dataset is publicly available at this https URL.
> AI Agent, Agentic AI, Coding Agent, Agentic Coding, Software Engineering Agent 

---
# FCRF: Flexible Constructivism Reflection for Long-Horizon Robotic Task Planning with Large Language Models 

**Authors**: Yufan Song, Jiatao Zhang, Zeng Gu, Qingmiao Liang, Tuocheng Hu, Wei Song, Shiqiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14975)  

**Abstract**: Autonomous error correction is critical for domestic robots to achieve reliable execution of complex long-horizon tasks. Prior work has explored self-reflection in Large Language Models (LLMs) for task planning error correction; however, existing methods are constrained by inflexible self-reflection mechanisms that limit their effectiveness. Motivated by these limitations and inspired by human cognitive adaptation, we propose the Flexible Constructivism Reflection Framework (FCRF), a novel Mentor-Actor architecture that enables LLMs to perform flexible self-reflection based on task difficulty, while constructively integrating historical valuable experience with failure lessons. We evaluated FCRF on diverse domestic tasks through simulation in AlfWorld and physical deployment in the real-world environment. Experimental results demonstrate that FCRF significantly improves overall performance and self-reflection flexibility in complex long-horizon robotic tasks. 

---
# A Comparative Analysis of Statistical and Machine Learning Models for Outlier Detection in Bitcoin Limit Order Books 

**Authors**: Ivan Letteri  

**Link**: [PDF](https://arxiv.org/pdf/2507.14960)  

**Abstract**: The detection of outliers within cryptocurrency limit order books (LOBs) is of paramount importance for comprehending market dynamics, particularly in highly volatile and nascent regulatory environments. This study conducts a comprehensive comparative analysis of robust statistical methods and advanced machine learning techniques for real-time anomaly identification in cryptocurrency LOBs. Within a unified testing environment, named AITA Order Book Signal (AITA-OBS), we evaluate the efficacy of thirteen diverse models to identify which approaches are most suitable for detecting potentially manipulative trading behaviours. An empirical evaluation, conducted via backtesting on a dataset of 26,204 records from a major exchange, demonstrates that the top-performing model, Empirical Covariance (EC), achieves a 6.70% gain, significantly outperforming a standard Buy-and-Hold benchmark. These findings underscore the effectiveness of outlier-driven strategies and provide insights into the trade-offs between model complexity, trade frequency, and performance. This study contributes to the growing corpus of research on cryptocurrency market microstructure by furnishing a rigorous benchmark of anomaly detection models and highlighting their potential for augmenting algorithmic trading and risk management. 

---
# Probing EFX via PMMS: (Non-)Existence Results in Discrete Fair Division 

**Authors**: Jarosław Byrka, Franciszek Malinka, Tomasz Ponitka  

**Link**: [PDF](https://arxiv.org/pdf/2507.14957)  

**Abstract**: We study the fair division of indivisible items and provide new insights into the EFX problem, which is widely regarded as the central open question in fair division, and the PMMS problem, a strictly stronger variant of EFX. Our first result constructs a three-agent instance with two monotone valuations and one additive valuation in which no PMMS allocation exists. Since EFX allocations are known to exist under these assumptions, this establishes a formal separation between EFX and PMMS.
We prove existence of fair allocations for three important special cases. We show that EFX allocations exist for personalized bivalued valuations, where for each agent $i$ there exist values $a_i > b_i$ such that agent $i$ assigns value $v_i(\{g\}) \in \{a_i, b_i\}$ to each good $g$. We establish an analogous existence result for PMMS allocations when $a_i$ is divisible by $b_i$. We also prove that PMMS allocations exist for binary-valued MMS-feasible valuations, where each bundle $S$ has value $v_i(S) \in \{0, 1\}$. Notably, this result holds even without assuming monotonicity of valuations and thus applies to the fair division of chores and mixed manna. Finally, we study a class of valuations called pair-demand valuations, which extend the well-studied unit-demand valuations to the case where each agent derives value from at most two items, and we show that PMMS allocations exist in this setting. Our proofs are constructive, and we provide polynomial-time algorithms for all three existence results. 

---
# Byzantine-Robust Decentralized Coordination of LLM Agents 

**Authors**: Yongrae Jo, Chanik Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.14928)  

**Abstract**: Collaboration among multiple large language model (LLM) agents is a promising approach to overcome inherent limitations of single-agent systems, such as hallucinations and single points of failure. As LLM agents are increasingly deployed on open blockchain platforms, multi-agent systems capable of tolerating malicious (Byzantine) agents have become essential.
Recent Byzantine-robust multi-agent systems typically rely on leader-driven coordination, which suffers from two major drawbacks. First, they are inherently vulnerable to targeted attacks against the leader. If consecutive leaders behave maliciously, the system repeatedly fails to achieve consensus, forcing new consensus rounds, which is particularly costly given the high latency of LLM invocations. Second, an underperforming proposal from the leader can be accepted as the final answer even when higher-quality alternatives are available, as existing methods finalize the leader's proposal once it receives a quorum of votes.
To address these issues, we propose DecentLLMs, a novel decentralized consensus approach for multi-agent LLM systems, where worker agents generate answers concurrently and evaluator agents independently score and rank these answers to select the best available one. This decentralized architecture enables faster consensus despite the presence of Byzantine agents and consistently selects higher-quality answers through Byzantine-robust aggregation techniques.
Experimental results demonstrate that DecentLLMs effectively tolerates Byzantine agents and significantly improves the quality of selected answers. 

---
# One Step Beyond: Feedthrough & Placement-Aware Rectilinear Floorplanner 

**Authors**: Zhexuan Xu, Jie Wang, Siyuan Xu, Zijie Geng, Mingxuan Yuan, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14914)  

**Abstract**: Floorplanning determines the shapes and locations of modules on a chip canvas and plays a critical role in optimizing the chip's Power, Performance, and Area (PPA) metrics. However, existing floorplanning approaches often fail to integrate with subsequent physical design stages, leading to suboptimal in-module component placement and excessive inter-module feedthrough. To tackle this challenge, we propose Flora, a three-stage feedthrough and placement aware rectilinear floorplanner. In the first stage, Flora employs wiremask and position mask techniques to achieve coarse-grained optimization of HPWL and feedthrough. In the second stage, under the constraint of a fixed outline, Flora achieves a zero-whitespace layout by locally resizing module shapes, thereby performing fine-grained optimization of feedthrough and improving component placement. In the third stage, Flora utilizes a fast tree search-based method to efficiently place components-including macros and standard cells-within each module, subsequently adjusting module boundaries based on the placement results to enable cross-stage optimization. Experimental results show that Flora outperforms recent state-of-the-art floorplanning approaches, achieving an average reduction of 6% in HPWL, 5.16% in FTpin, 29.15% in FTmod, and a 14% improvement in component placement performance. 

---
# Partial Symmetry Enforced Attention Decomposition (PSEAD): A Group-Theoretic Framework for Equivariant Transformers in Biological Systems 

**Authors**: Daniel Ayomide Olanrewaju  

**Link**: [PDF](https://arxiv.org/pdf/2507.14908)  

**Abstract**: This research introduces the Theory of Partial Symmetry Enforced Attention Decomposition (PSEAD), a new and rigorous group-theoretic framework designed to seamlessly integrate local symmetry awareness into the core architecture of self-attention mechanisms within Transformer models. We formalize the concept of local permutation subgroup actions on windows of biological data, proving that under such actions, the attention mechanism naturally decomposes into a direct sum of orthogonal irreducible components. Critically, these components are intrinsically aligned with the irreducible representations of the acting permutation subgroup, thereby providing a powerful mathematical basis for disentangling symmetric and asymmetric features. We show that PSEAD offers substantial advantages. These include enhanced generalization capabilities to novel biological motifs exhibiting similar partial symmetries, unprecedented interpretability by allowing direct visualization and analysis of attention contributions from different symmetry channels, and significant computational efficiency gains by focusing representational capacity on relevant symmetric subspaces. Beyond static data analysis, we extend PSEAD's applicability to dynamic biological processes within reinforcement learning paradigms, showcasing its potential to accelerate the discovery and optimization of biologically meaningful policies in complex environments like protein folding and drug discovery. This work lays the groundwork for a new generation of biologically informed, symmetry-aware artificial intelligence models. 

---
# TriCLIP-3D: A Unified Parameter-Efficient Framework for Tri-Modal 3D Visual Grounding based on CLIP 

**Authors**: Fan Li, Zanyi Wang, Zeyi Huang, Guang Dai, Jingdong Wang, Mengmeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14904)  

**Abstract**: 3D visual grounding allows an embodied agent to understand visual information in real-world 3D environments based on human instructions, which is crucial for embodied intelligence. Existing 3D visual grounding methods typically rely on separate encoders for different modalities (e.g., RGB images, text, and 3D point clouds), resulting in large and complex models that are inefficient to train. While some approaches use pre-trained 2D multi-modal models like CLIP for 3D tasks, they still struggle with aligning point cloud data to 2D encoders. As a result, these methods continue to depend on 3D encoders for feature extraction, further increasing model complexity and training inefficiency. In this paper, we propose a unified 2D pre-trained multi-modal network to process all three modalities (RGB images, text, and point clouds), significantly simplifying the architecture. By leveraging a 2D CLIP bi-modal model with adapter-based fine-tuning, this framework effectively adapts to the tri-modal setting, improving both adaptability and performance across modalities. Our Geometric-Aware 2D-3D Feature Recovery and Fusion (GARF) module is designed to fuse geometric multi-scale features from point clouds and images. We then integrate textual features for final modality fusion and introduce a multi-modal decoder to facilitate deep cross-modal understanding. Together, our method achieves unified feature extraction and fusion across the three modalities, enabling an end-to-end 3D visual grounding model. Compared to the baseline, our method reduces the number of trainable parameters by approximately 58\%, while achieving a 6.52\% improvement in the 3D detection task and a 6.25\% improvement in the 3D visual grounding task. 

---
# Learning Nonlinear Causal Reductions to Explain Reinforcement Learning Policies 

**Authors**: Armin Kekić, Jan Schneider, Dieter Büchler, Bernhard Schölkopf, Michel Besserve  

**Link**: [PDF](https://arxiv.org/pdf/2507.14901)  

**Abstract**: Why do reinforcement learning (RL) policies fail or succeed? This is a challenging question due to the complex, high-dimensional nature of agent-environment interactions. In this work, we take a causal perspective on explaining the behavior of RL policies by viewing the states, actions, and rewards as variables in a low-level causal model. We introduce random perturbations to policy actions during execution and observe their effects on the cumulative reward, learning a simplified high-level causal model that explains these relationships. To this end, we develop a nonlinear Causal Model Reduction framework that ensures approximate interventional consistency, meaning the simplified high-level model responds to interventions in a similar way as the original complex system. We prove that for a class of nonlinear causal models, there exists a unique solution that achieves exact interventional consistency, ensuring learned explanations reflect meaningful causal patterns. Experiments on both synthetic causal models and practical RL tasks-including pendulum control and robot table tennis-demonstrate that our approach can uncover important behavioral patterns, biases, and failure modes in trained RL policies. 

---
# Application-Specific Component-Aware Structured Pruning of Deep Neural Networks via Soft Coefficient Optimization 

**Authors**: Ganesh Sundaram, Jonas Ulmen, Amjad Haider, Daniel Görges  

**Link**: [PDF](https://arxiv.org/pdf/2507.14882)  

**Abstract**: Deep neural networks (DNNs) offer significant versatility and performance benefits, but their widespread adoption is often hindered by high model complexity and computational demands. Model compression techniques such as pruning have emerged as promising solutions to these challenges. However, it remains critical to ensure that application-specific performance characteristics are preserved during compression. In structured pruning, where groups of structurally coherent elements are removed, conventional importance metrics frequently fail to maintain these essential performance attributes. In this work, we propose an enhanced importance metric framework that not only reduces model size but also explicitly accounts for application-specific performance constraints. We employ multiple strategies to determine the optimal pruning magnitude for each group, ensuring a balance between compression and task performance. Our approach is evaluated on an autoencoder tasked with reconstructing MNIST images. Experimental results demonstrate that the proposed method effectively preserves task-relevant performance, maintaining the model's usability even after substantial pruning, by satisfying the required application-specific criteria. 

---
# The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs 

**Authors**: Ole-Christoffer Granmo, Youmna Abdelwahab, Per-Arne Andersen, Paul F. A. Clarke, Kunal Dumbre, Ylva Grønninsæter, Vojtech Halenka, Runar Helin, Lei Jiao, Ahmed Khalid, Rebekka Omslandseter, Rupsa Saha, Mayur Shende, Xuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14874)  

**Abstract**: Pattern recognition with concise and flat AND-rules makes the Tsetlin Machine (TM) both interpretable and efficient, while the power of Tsetlin automata enables accuracy comparable to deep learning on an increasing number of datasets. We introduce the Graph Tsetlin Machine (GraphTM) for learning interpretable deep clauses from graph-structured input. Moving beyond flat, fixed-length input, the GraphTM gets more versatile, supporting sequences, grids, relations, and multimodality. Through message passing, the GraphTM builds nested deep clauses to recognize sub-graph patterns with exponentially fewer clauses, increasing both interpretability and data utilization. For image classification, GraphTM preserves interpretability and achieves 3.86%-points higher accuracy on CIFAR-10 than a convolutional TM. For tracking action coreference, faced with increasingly challenging tasks, GraphTM outperforms other reinforcement learning methods by up to 20.6%-points. In recommendation systems, it tolerates increasing noise to a greater extent than a Graph Convolutional Neural Network (GCN), e.g., for noise ratio 0.1, GraphTM obtains accuracy 89.86% compared to GCN's 70.87%. Finally, for viral genome sequence data, GraphTM is competitive with BiLSTM-CNN and GCN accuracy-wise, training 2.5x faster than GCN. The GraphTM's application to these varied fields demonstrates how graph representation learning and deep clauses bring new possibilities for TM learning. 

---
# Grounding Degradations in Natural Language for All-In-One Video Restoration 

**Authors**: Muhammad Kamran Janjua, Amirhosein Ghasemabadi, Kunlin Zhang, Mohammad Salameh, Chao Gao, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14851)  

**Abstract**: In this work, we propose an all-in-one video restoration framework that grounds degradation-aware semantic context of video frames in natural language via foundation models, offering interpretable and flexible guidance. Unlike prior art, our method assumes no degradation knowledge in train or test time and learns an approximation to the grounded knowledge such that the foundation model can be safely disentangled during inference adding no extra cost. Further, we call for standardization of benchmarks in all-in-one video restoration, and propose two benchmarks in multi-degradation setting, three-task (3D) and four-task (4D), and two time-varying composite degradation benchmarks; one of the latter being our proposed dataset with varying snow intensity, simulating how weather degradations affect videos naturally. We compare our method with prior works and report state-of-the-art performance on all benchmarks. 

---
# Hierarchical Multi-Agent Reinforcement Learning with Control Barrier Functions for Safety-Critical Autonomous Systems 

**Authors**: H. M. Sabbir Ahmad, Ehsan Sabouni, Alexander Wasilkoff, Param Budhraja, Zijian Guo, Songyuan Zhang, Chuchu Fan, Christos Cassandras, Wenchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14850)  

**Abstract**: We address the problem of safe policy learning in multi-agent safety-critical autonomous systems. In such systems, it is necessary for each agent to meet the safety requirements at all times while also cooperating with other agents to accomplish the task. Toward this end, we propose a safe Hierarchical Multi-Agent Reinforcement Learning (HMARL) approach based on Control Barrier Functions (CBFs). Our proposed hierarchical approach decomposes the overall reinforcement learning problem into two levels learning joint cooperative behavior at the higher level and learning safe individual behavior at the lower or agent level conditioned on the high-level policy. Specifically, we propose a skill-based HMARL-CBF algorithm in which the higher level problem involves learning a joint policy over the skills for all the agents and the lower-level problem involves learning policies to execute the skills safely with CBFs. We validate our approach on challenging environment scenarios whereby a large number of agents have to safely navigate through conflicting road networks. Compared with existing state of the art methods, our approach significantly improves the safety achieving near perfect (within 5%) success/safety rate while also improving performance across all the environments. 

---
# The Invisible Leash: Why RLVR May Not Escape Its Origin 

**Authors**: Fang Wu, Weihao Xuan, Ximing Lu, Zaid Harchaoui, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14843)  

**Abstract**: Recent advances in large reasoning models highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising method for enhancing AI's capabilities, particularly in solving complex logical tasks. However, it remains unclear whether RLVR truly expands a model's reasoning boundary or merely amplifies high-reward outputs that the base model already knows for improved precision. This study presents a theoretical and empirical investigation that provides fresh insights into the potential limits of RLVR. First, we offer a new theoretical perspective that RLVR is constrained by the base model's support-unable to sample solutions with zero initial probability-and operates as a conservative reweighting mechanism that may restrict the discovery of entirely original solutions. We also identify an entropy-reward tradeoff: while RLVR reliably enhances precision, it may progressively narrow exploration and potentially overlook correct yet underrepresented solutions. Extensive empirical experiments validate that while RLVR consistently improves pass@1, the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets, failing to recover correct answers that were previously accessible to the base model. Interestingly, we also observe that while RLVR sometimes increases token-level entropy, resulting in greater uncertainty at each generation step, answer-level entropy declines, indicating that these seemingly more uncertain paths ultimately converge onto a smaller set of distinct answers. Taken together, these findings reveal potential limits of RLVR in extending reasoning horizons. Breaking this invisible leash may require future algorithmic innovations such as explicit exploration mechanisms or hybrid strategies that seed probability mass into underrepresented solution regions. 

---
# Paired Image Generation with Diffusion-Guided Diffusion Models 

**Authors**: Haoxuan Zhang, Wenju Cui, Yuzhu Cao, Tao Tan, Jie Liu, Yunsong Peng, Jian Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.14833)  

**Abstract**: The segmentation of mass lesions in digital breast tomosynthesis (DBT) images is very significant for the early screening of breast cancer. However, the high-density breast tissue often leads to high concealment of the mass lesions, which makes manual annotation difficult and time-consuming. As a result, there is a lack of annotated data for model training. Diffusion models are commonly used for data augmentation, but the existing methods face two challenges. First, due to the high concealment of lesions, it is difficult for the model to learn the features of the lesion area. This leads to the low generation quality of the lesion areas, thus limiting the quality of the generated images. Second, existing methods can only generate images and cannot generate corresponding annotations, which restricts the usability of the generated images in supervised training. In this work, we propose a paired image generation method. The method does not require external conditions and can achieve the generation of paired images by training an extra diffusion guider for the conditional diffusion model. During the experimental phase, we generated paired DBT slices and mass lesion masks. Then, we incorporated them into the supervised training process of the mass lesion segmentation task. The experimental results show that our method can improve the generation quality without external conditions. Moreover, it contributes to alleviating the shortage of annotated data, thus enhancing the performance of downstream tasks. 

---
# eMargin: Revisiting Contrastive Learning with Margin-Based Separation 

**Authors**: Abdul-Kazeem Shamba, Kerstin Bach, Gavin Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2507.14828)  

**Abstract**: We revisit previous contrastive learning frameworks to investigate the effect of introducing an adaptive margin into the contrastive loss function for time series representation learning. Specifically, we explore whether an adaptive margin (eMargin), adjusted based on a predefined similarity threshold, can improve the separation between adjacent but dissimilar time steps and subsequently lead to better performance in downstream tasks. Our study evaluates the impact of this modification on clustering performance and classification in three benchmark datasets. Our findings, however, indicate that achieving high scores on unsupervised clustering metrics does not necessarily imply that the learned embeddings are meaningful or effective in downstream tasks. To be specific, eMargin added to InfoNCE consistently outperforms state-of-the-art baselines in unsupervised clustering metrics, but struggles to achieve competitive results in downstream classification with linear probing. The source code is publicly available at this https URL. 

---
# Benchmarking Foundation Models with Multimodal Public Electronic Health Records 

**Authors**: Kunyu Yu, Rui Yang, Jingchi Liao, Siqi Li, Huitao Li, Irene Li, Yifan Peng, Rishikesan Kamaleswaran, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14824)  

**Abstract**: Foundation models have emerged as a powerful approach for processing electronic health records (EHRs), offering flexibility to handle diverse medical data modalities. In this study, we present a comprehensive benchmark that evaluates the performance, fairness, and interpretability of foundation models, both as unimodal encoders and as multimodal learners, using the publicly available MIMIC-IV database. To support consistent and reproducible evaluation, we developed a standardized data processing pipeline that harmonizes heterogeneous clinical records into an analysis-ready format. We systematically compared eight foundation models, encompassing both unimodal and multimodal models, as well as domain-specific and general-purpose variants. Our findings demonstrate that incorporating multiple data modalities leads to consistent improvements in predictive performance without introducing additional bias. Through this benchmark, we aim to support the development of effective and trustworthy multimodal artificial intelligence (AI) systems for real-world clinical applications. Our code is available at this https URL. 

---
# SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models 

**Authors**: Jiaji Zhang, Ruichao Sun, Hailiang Zhao, Jiaju Wu, Peng Chen, Hao Li, Xinkui Zhao, Kingsum Chow, Gang Xiong, Lin Ye, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.14811)  

**Abstract**: Diffusion models have demonstrated exceptional generative capabilities but are computationally intensive, posing significant challenges for deployment in resource-constrained or latency-sensitive environments. Quantization offers an effective means to reduce model size and computational cost, with post-training quantization (PTQ) being particularly appealing due to its compatibility with pre-trained models without requiring retraining or training data. However, existing PTQ methods for diffusion models often rely on architecture-specific heuristics that limit their generalizability and hinder integration with industrial deployment pipelines. To address these limitations, we propose SegQuant, a unified quantization framework that adaptively combines complementary techniques to enhance cross-model versatility. SegQuant consists of a segment-aware, graph-based quantization strategy (SegLinear) that captures structural semantics and spatial heterogeneity, along with a dual-scale quantization scheme (DualScale) that preserves polarity-asymmetric activations, which is crucial for maintaining visual fidelity in generated outputs. SegQuant is broadly applicable beyond Transformer-based diffusion models, achieving strong performance while ensuring seamless compatibility with mainstream deployment tools. 

---
# Seeing Through Deepfakes: A Human-Inspired Framework for Multi-Face Detection 

**Authors**: Juan Hu, Shaojing Fan, Terence Sim  

**Link**: [PDF](https://arxiv.org/pdf/2507.14807)  

**Abstract**: Multi-face deepfake videos are becoming increasingly prevalent, often appearing in natural social settings that challenge existing detection methods. Most current approaches excel at single-face detection but struggle in multi-face scenarios, due to a lack of awareness of crucial contextual cues. In this work, we develop a novel approach that leverages human cognition to analyze and defend against multi-face deepfake videos. Through a series of human studies, we systematically examine how people detect deepfake faces in social settings. Our quantitative analysis reveals four key cues humans rely on: scene-motion coherence, inter-face appearance compatibility, interpersonal gaze alignment, and face-body consistency. Guided by these insights, we introduce \textsf{HICOM}, a novel framework designed to detect every fake face in multi-face scenarios. Extensive experiments on benchmark datasets show that \textsf{HICOM} improves average accuracy by 3.3\% in in-dataset detection and 2.8\% under real-world perturbations. Moreover, it outperforms existing methods by 5.8\% on unseen datasets, demonstrating the generalization of human-inspired cues. \textsf{HICOM} further enhances interpretability by incorporating an LLM to provide human-readable explanations, making detection results more transparent and convincing. Our work sheds light on involving human factors to enhance defense against deepfakes. 

---
# Subliminal Learning: Language models transmit behavioral traits via hidden signals in data 

**Authors**: Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2507.14805)  

**Abstract**: We study subliminal learning, a surprising phenomenon where language models transmit behavioral traits via semantically unrelated data. In our main experiments, a "teacher" model with some trait T (such as liking owls or being misaligned) generates a dataset consisting solely of number sequences. Remarkably, a "student" model trained on this dataset learns T. This occurs even when the data is filtered to remove references to T. We observe the same effect when training on code or reasoning traces generated by the same teacher model. However, we do not observe the effect when the teacher and student have different base models. To help explain our findings, we prove a theoretical result showing that subliminal learning occurs in all neural networks under certain conditions, and demonstrate subliminal learning in a simple MLP classifier. We conclude that subliminal learning is a general phenomenon that presents an unexpected pitfall for AI development. Distillation could propagate unintended traits, even when developers try to prevent this via data filtering. 

---
# ACME: Adaptive Customization of Large Models via Distributed Systems 

**Authors**: Ziming Dai, Chao Qiu, Fei Gao, Yunfeng Zhao, Xiaofei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14802)  

**Abstract**: Pre-trained Transformer-based large models have revolutionized personal virtual assistants, but their deployment in cloud environments faces challenges related to data privacy and response latency. Deploying large models closer to the data and users has become a key research area to address these issues. However, applying these models directly often entails significant difficulties, such as model mismatching, resource constraints, and energy inefficiency. Automated design of customized models is necessary, but it faces three key challenges, namely, the high cost of centralized model customization, imbalanced performance from user heterogeneity, and suboptimal performance from data heterogeneity. In this paper, we propose ACME, an adaptive customization approach of Transformer-based large models via distributed systems. To avoid the low cost-efficiency of centralized methods, ACME employs a bidirectional single-loop distributed system to progressively achieve fine-grained collaborative model customization. In order to better match user heterogeneity, it begins by customizing the backbone generation and identifying the Pareto Front under model size constraints to ensure optimal resource utilization. Subsequently, it performs header generation and refines the model using data distribution-based personalized architecture aggregation to match data heterogeneity. Evaluation on different datasets shows that ACME achieves cost-efficient models under model size constraints. Compared to centralized systems, data transmission volume is reduced to 6 percent. Additionally, the average accuracy improves by 10 percent compared to the baseline, with the trade-off metrics increasing by nearly 30 percent. 

---
# Large Language Model as An Operator: An Experience-Driven Solution for Distribution Network Voltage Control 

**Authors**: Xu Yang, Chenhui Lin, Haotian Liu, Qi Wang, Wenchuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14800)  

**Abstract**: With the advanced reasoning and information analysis capabilities, large language models (LLMs) can offer a novel approach for the autonomous generation of dispatch strategies in power systems. This letter proposes an LLM-based experience-driven voltage control solution for distribution networks, which enables the self-evolution of LLM-based voltage control strategies through the collaboration and interaction of multiple modules-specifically, experience storage, experience retrieval, experience generation, and experience modification. Comprehensive experimental results validate the effectiveness of the proposed method and highlight the applicability of LLM in addressing power system dispatch challenges. 

---
# Manipulating LLM Web Agents with Indirect Prompt Injection Attack via HTML Accessibility Tree 

**Authors**: Sam Johnson, Viet Pham, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2507.14799)  

**Abstract**: This work demonstrates that LLM-based web navigation agents offer powerful automation capabilities but are vulnerable to Indirect Prompt Injection (IPI) attacks. We show that adversaries can embed universal adversarial triggers in webpage HTML to hijack agent behavior that utilizes the accessibility tree to parse HTML, causing unintended or malicious actions. Using the Greedy Coordinate Gradient (GCG) algorithm and a Browser Gym agent powered by Llama-3.1, our system demonstrates high success rates across real websites in both targeted and general attacks, including login credential exfiltration and forced ad clicks. Our empirical results highlight critical security risks and the need for stronger defenses as LLM-driven autonomous web agents become more widely adopted. The system software (this https URL) is released under the MIT License, with an accompanying publicly available demo website (this http URL). 

---
# FOCUS: Fused Observation of Channels for Unveiling Spectra 

**Authors**: Xi Xiao, Aristeidis Tsaris, Anika Tabassum, John Lagergren, Larry M. York, Tianyang Wang, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14787)  

**Abstract**: Hyperspectral imaging (HSI) captures hundreds of narrow, contiguous wavelength bands, making it a powerful tool in biology, agriculture, and environmental monitoring. However, interpreting Vision Transformers (ViTs) in this setting remains largely unexplored due to two key challenges: (1) existing saliency methods struggle to capture meaningful spectral cues, often collapsing attention onto the class token, and (2) full-spectrum ViTs are computationally prohibitive for interpretability, given the high-dimensional nature of HSI data. We present FOCUS, the first framework that enables reliable and efficient spatial-spectral interpretability for frozen ViTs. FOCUS introduces two core components: class-specific spectral prompts that guide attention toward semantically meaningful wavelength groups, and a learnable [SINK] token trained with an attraction loss to absorb noisy or redundant attention. Together, these designs make it possible to generate stable and interpretable 3D saliency maps and spectral importance curves in a single forward pass, without any gradient backpropagation or backbone modification. FOCUS improves band-level IoU by 15 percent, reduces attention collapse by over 40 percent, and produces saliency results that align closely with expert annotations. With less than 1 percent parameter overhead, our method makes high-resolution ViT interpretability practical for real-world hyperspectral applications, bridging a long-standing gap between black-box modeling and trustworthy HSI decision-making. 

---
# Exploring the In-Context Learning Capabilities of LLMs for Money Laundering Detection in Financial Graphs 

**Authors**: Erfan Pirmorad  

**Link**: [PDF](https://arxiv.org/pdf/2507.14785)  

**Abstract**: The complexity and interconnectivity of entities involved in money laundering demand investigative reasoning over graph-structured data. This paper explores the use of large language models (LLMs) as reasoning engines over localized subgraphs extracted from a financial knowledge graph. We propose a lightweight pipeline that retrieves k-hop neighborhoods around entities of interest, serializes them into structured text, and prompts an LLM via few-shot in-context learning to assess suspiciousness and generate justifications. Using synthetic anti-money laundering (AML) scenarios that reflect common laundering behaviors, we show that LLMs can emulate analyst-style logic, highlight red flags, and provide coherent explanations. While this study is exploratory, it illustrates the potential of LLM-based graph reasoning in AML and lays groundwork for explainable, language-driven financial crime analytics. 

---
# LeAdQA: LLM-Driven Context-Aware Temporal Grounding for Video Question Answering 

**Authors**: Xinxin Dong, Baoyun Peng, Haokai Ma, Yufei Wang, Zixuan Dong, Fei Hu, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14784)  

**Abstract**: Video Question Answering (VideoQA) requires identifying sparse critical moments in long videos and reasoning about their causal relationships to answer semantically complex questions. While recent advances in multimodal learning have improved alignment and fusion, current approaches remain limited by two prevalent but fundamentally flawed strategies: (1) task-agnostic sampling indiscriminately processes all frames, overwhelming key events with irrelevant content; and (2) heuristic retrieval captures superficial patterns but misses causal-temporal structures needed for complex reasoning. To address these challenges, we introduce LeAdQA, an innovative approach that bridges these gaps through synergizing causal-aware query refinement with fine-grained visual grounding. Our method first leverages LLMs to reformulate question-option pairs, resolving causal ambiguities and sharpening temporal focus. These refined queries subsequently direct a temporal grounding model to precisely retrieve the most salient segments, complemented by an adaptive fusion mechanism dynamically integrating the evidence to maximize relevance. The integrated visual-textual cues are then processed by an MLLM to generate accurate, contextually-grounded answers. Experiments on NExT-QA, IntentQA, and NExT-GQA demonstrate that our method's precise visual grounding substantially enhances the understanding of video-question relationships, achieving state-of-the-art (SOTA) performance on complex reasoning tasks while maintaining computational efficiency. 

---
# Omni-Think: Scaling Cross-Domain Generalization in LLMs via Multi-Task RL with Hybrid Rewards 

**Authors**: Derek Li, Jiaming Zhou, Amirreza Kazemi, Qianyi Sun, Abbas Ghaddar, Mohammad Ali Alomrani, Liheng Ma, Yu Luo, Dong Li, Feng Wen, Jianye Hao, Mark Coates, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14783)  

**Abstract**: The advancement of general-purpose artificial intelligence relies on large language models (LLMs) that excel across a wide range of tasks, from structured reasoning to creative generation. However, post-training methods like Supervised Fine-Tuning (SFT) often struggle with generalization, favoring memorization over transferable learning. In this work, we introduce Omni-Think, a unified reinforcement learning (RL) framework that enhances LLM performance across diverse tasks by combining rule-based verifiable rewards with generative preference signals via LLM-as-a-Judge evaluations. Our approach enables consistent optimization across task types and scales RL-based training to subjective domains. We further investigate training strategies, demonstrating that a curriculum-based progression that orders tasks from structured to open-ended improves performance and reduces forgetting. Experimental results across four domains reveal that curriculum learning improves performance by 5.2\% over joint training and 9.1\% over model merging. These results highlight the importance of task-aware sampling and hybrid supervision in scaling RL-based post-training for general-purpose LLMs. 

---
# XplainAct: Visualization for Personalized Intervention Insights 

**Authors**: Yanming Zhang, Krishnakumar Hegde, Klaus Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2507.14767)  

**Abstract**: Causality helps people reason about and understand complex systems, particularly through what-if analyses that explore how interventions might alter outcomes. Although existing methods embrace causal reasoning using interventions and counterfactual analysis, they primarily focus on effects at the population level. These approaches often fall short in systems characterized by significant heterogeneity, where the impact of an intervention can vary widely across subgroups. To address this challenge, we present XplainAct, a visual analytics framework that supports simulating, explaining, and reasoning interventions at the individual level within subpopulations. We demonstrate the effectiveness of XplainAct through two case studies: investigating opioid-related deaths in epidemiology and analyzing voting inclinations in the presidential election. 

---
# CXR-TFT: Multi-Modal Temporal Fusion Transformer for Predicting Chest X-ray Trajectories 

**Authors**: Mehak Arora, Ayman Ali, Kaiyuan Wu, Carolyn Davis, Takashi Shimazui, Mahmoud Alwakeel, Victor Moas, Philip Yang, Annette Esper, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2507.14766)  

**Abstract**: In intensive care units (ICUs), patients with complex clinical conditions require vigilant monitoring and prompt interventions. Chest X-rays (CXRs) are a vital diagnostic tool, providing insights into clinical trajectories, but their irregular acquisition limits their utility. Existing tools for CXR interpretation are constrained by cross-sectional analysis, failing to capture temporal dynamics. To address this, we introduce CXR-TFT, a novel multi-modal framework that integrates temporally sparse CXR imaging and radiology reports with high-frequency clinical data, such as vital signs, laboratory values, and respiratory flow sheets, to predict the trajectory of CXR findings in critically ill patients. CXR-TFT leverages latent embeddings from a vision encoder that are temporally aligned with hourly clinical data through interpolation. A transformer model is then trained to predict CXR embeddings at each hour, conditioned on previous embeddings and clinical measurements. In a retrospective study of 20,000 ICU patients, CXR-TFT demonstrated high accuracy in forecasting abnormal CXR findings up to 12 hours before they became radiographically evident. This predictive capability in clinical data holds significant potential for enhancing the management of time-sensitive conditions like acute respiratory distress syndrome, where early intervention is crucial and diagnoses are often delayed. By providing distinctive temporal resolution in prognostic CXR analysis, CXR-TFT offers actionable 'whole patient' insights that can directly improve clinical outcomes. 

---
# QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems 

**Authors**: Cassandra Tong Ye, Shamus Li, Tyler King, Kristina Monakhova  

**Link**: [PDF](https://arxiv.org/pdf/2507.14760)  

**Abstract**: Deep learning models often hallucinate, producing realistic artifacts that are not truly present in the sample. This can have dire consequences for scientific and medical inverse problems, such as MRI and microscopy denoising, where accuracy is more important than perceptual quality. Uncertainty quantification techniques, such as conformal prediction, can pinpoint outliers and provide guarantees for image regression tasks, improving reliability. However, existing methods utilize a linear constant scaling factor to calibrate uncertainty bounds, resulting in larger, less informative bounds. We propose QUTCC, a quantile uncertainty training and calibration technique that enables nonlinear, non-uniform scaling of quantile predictions to enable tighter uncertainty estimates. Using a U-Net architecture with a quantile embedding, QUTCC enables the prediction of the full conditional distribution of quantiles for the imaging task. During calibration, QUTCC generates uncertainty bounds by iteratively querying the network for upper and lower quantiles, progressively refining the bounds to obtain a tighter interval that captures the desired coverage. We evaluate our method on several denoising tasks as well as compressive MRI reconstruction. Our method successfully pinpoints hallucinations in image estimates and consistently achieves tighter uncertainty intervals than prior methods while maintaining the same statistical coverage. 

---
# GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization 

**Authors**: Luyi Ma, Wanjia Zhang, Kai Zhao, Abhishek Kulkarni, Lalitesh Morishetti, Anjana Ganesh, Ashish Ranjan, Aashika Padmanabhan, Jianpeng Xu, Jason Cho, Praveen Kanumala, Kaushiki Nag, Sumit Dutta, Kamiya Motwani, Malay Patel, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14758)  

**Abstract**: Generative models have recently demonstrated strong potential in multi-behavior recommendation systems, leveraging the expressive power of transformers and tokenization to generate personalized item sequences. However, their adoption is hindered by (1) the lack of explicit information for token reasoning, (2) high computational costs due to quadratic attention complexity and dense sequence representations after tokenization, and (3) limited multi-scale modeling over user history. In this work, we propose GRACE (Generative Recommendation via journey-aware sparse Attention on Chain-of-thought tokEnization), a novel generative framework for multi-behavior sequential recommendation. GRACE introduces a hybrid Chain-of-Thought (CoT) tokenization method that encodes user-item interactions with explicit attributes from product knowledge graphs (e.g., category, brand, price) over semantic tokenization, enabling interpretable and behavior-aligned generation. To address the inefficiency of standard attention, we design a Journey-Aware Sparse Attention (JSA) mechanism, which selectively attends to compressed, intra-, inter-, and current-context segments in the tokenized sequence. Experiments on two real-world datasets show that GRACE significantly outperforms state-of-the-art baselines, achieving up to +106.9% HR@10 and +106.7% NDCG@10 improvement over the state-of-the-art baseline on the Home domain, and +22.1% HR@10 on the Electronics domain. GRACE also reduces attention computation by up to 48% with long sequences. 

---
# Analyzing Internal Activity and Robustness of SNNs Across Neuron Parameter Space 

**Authors**: Szymon Mazurek, Jakub Caputa, Maciej Wielgosz  

**Link**: [PDF](https://arxiv.org/pdf/2507.14757)  

**Abstract**: Spiking Neural Networks (SNNs) offer energy-efficient and biologically plausible alternatives to traditional artificial neural networks, but their performance depends critically on the tuning of neuron model parameters. In this work, we identify and characterize an operational space - a constrained region in the neuron hyperparameter domain (specifically membrane time constant tau and voltage threshold vth) - within which the network exhibits meaningful activity and functional behavior. Operating inside this manifold yields optimal trade-offs between classification accuracy and spiking activity, while stepping outside leads to degeneration: either excessive energy use or complete network silence.
Through systematic exploration across datasets and architectures, we visualize and quantify this manifold and identify efficient operating points. We further assess robustness to adversarial noise, showing that SNNs exhibit increased spike correlation and internal synchrony when operating outside their optimal region. These findings highlight the importance of principled hyperparameter tuning to ensure both task performance and energy efficiency. Our results offer practical guidelines for deploying robust and efficient SNNs, particularly in neuromorphic computing scenarios. 

---
# Skill Learning via Policy Diversity Yields Identifiable Representations for Reinforcement Learning 

**Authors**: Patrik Reizinger, Bálint Mucsányi, Siyuan Guo, Benjamin Eysenbach, Bernhard Schölkopf, Wieland Brendel  

**Link**: [PDF](https://arxiv.org/pdf/2507.14748)  

**Abstract**: Self-supervised feature learning and pretraining methods in reinforcement learning (RL) often rely on information-theoretic principles, termed mutual information skill learning (MISL). These methods aim to learn a representation of the environment while also incentivizing exploration thereof. However, the role of the representation and mutual information parametrization in MISL is not yet well understood theoretically. Our work investigates MISL through the lens of identifiable representation learning by focusing on the Contrastive Successor Features (CSF) method. We prove that CSF can provably recover the environment's ground-truth features up to a linear transformation due to the inner product parametrization of the features and skill diversity in a discriminative sense. This first identifiability guarantee for representation learning in RL also helps explain the implications of different mutual information objectives and the downsides of entropy regularizers. We empirically validate our claims in MuJoCo and DeepMind Control and show how CSF provably recovers the ground-truth features both from states and pixels. 

---
# Task-Agnostic Continual Prompt Tuning with Gradient-Based Selection and Decoding 

**Authors**: Anushka Tiwari, Sayantan Pal, Rohini K. Srihari, Kaiyi Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.14725)  

**Abstract**: Prompt-based continual learning (CL) offers a parameter-efficient way to adapt large language models (LLMs) across task sequences. However, most existing methods assume task-aware inference and maintain a growing list of task-specific prompts, which limits scalability and hides latent forgetting. In this work, we introduce GRID, a unified framework that addresses two key limitations: (1) latent forgetting under task-agnostic inference, and (2) prompt memory explosion as task sequences grow. GRID integrates a task-aware decoding mechanism that improves backward transfer by leveraging representative inputs, automatic task identification, and constrained decoding. Additionally, we propose a gradient-based prompt selection strategy that compresses less informative prompts into a single aggregated representation, enabling scalable and memory-efficient lifelong learning. Extensive experiments across short-sequence, long-sequence, and negative transfer benchmarks show that GRID significantly improves backward transfer, achieves competitive forward transfer, and reduces forgotten tasks by up to 80\%, outperforming state-of-the-art methods on T5 and Flan-T5 backbones. 

---
# LeanTree: Accelerating White-Box Proof Search with Factorized States in Lean 4 

**Authors**: Matěj Kripner, Michal Šustr, Milan Straka  

**Link**: [PDF](https://arxiv.org/pdf/2507.14722)  

**Abstract**: Automated theorem proving (ATP) has been a classical problem in artificial intelligence since its inception, yet it remains challenging due to its vast state and action space. Large language models (LLMs) have recently emerged as a promising heuristic for ATP, but they lack correctness guarantees and thus require interaction with a proof verifier. Such interactions typically follow one of two approaches: black-box interaction, which does not utilize intermediate proof states, or white-box approaches, which allow for incremental proof construction and examination of intermediate states. While black-box approaches have directly benefited from recent LLM advances, white-box methods have comparatively lagged behind. In this paper, we address this gap by introducing LeanTree, which consists of (i) a tool built in the Lean 4 language that factorizes complex proof states into simpler, independent branches, and (ii) a dataset of these factorized intermediate states. Our white-box tooling offers several advantages over black-box approaches: it simplifies evaluation, reduces necessary context, generates richer training data, enables parallel search across multiple states, supports efficient reuse of states, and provides feedback in case of errors. Our preliminary results hint that white-box approaches outperform black-box alternatives in some settings. 

---
# Fraud is Not Just Rarity: A Causal Prototype Attention Approach to Realistic Synthetic Oversampling 

**Authors**: Claudio Giusti, Luca Guarnera, Mirko Casu, Sebastiano Battiato  

**Link**: [PDF](https://arxiv.org/pdf/2507.14706)  

**Abstract**: Detecting fraudulent credit card transactions remains a significant challenge, due to the extreme class imbalance in real-world data and the often subtle patterns that separate fraud from legitimate activity. Existing research commonly attempts to address this by generating synthetic samples for the minority class using approaches such as GANs, VAEs, or hybrid generative models. However, these techniques, particularly when applied only to minority-class data, tend to result in overconfident classifiers and poor latent cluster separation, ultimately limiting real-world detection performance. In this study, we propose the Causal Prototype Attention Classifier (CPAC), an interpretable architecture that promotes class-aware clustering and improved latent space structure through prototype-based attention mechanisms and we will couple it with the encoder in a VAE-GAN allowing it to offer a better cluster separation moving beyond post-hoc sample augmentation. We compared CPAC-augmented models to traditional oversamplers, such as SMOTE, as well as to state-of-the-art generative models, both with and without CPAC-based latent classifiers. Our results show that classifier-guided latent shaping with CPAC delivers superior performance, achieving an F1-score of 93.14\% percent and recall of 90.18\%, along with improved latent cluster separation. Further ablation studies and visualizations provide deeper insight into the benefits and limitations of classifier-driven representation learning for fraud detection. The codebase for this work will be available at final submission. 

---
# Spatial-Temporal Transformer with Curriculum Learning for EEG-Based Emotion Recognition 

**Authors**: Xuetao Lin, Tianhao Peng, Peihong Dai, Yu Liang, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14698)  

**Abstract**: EEG-based emotion recognition plays an important role in developing adaptive brain-computer communication systems, yet faces two fundamental challenges in practical implementations: (1) effective integration of non-stationary spatial-temporal neural patterns, (2) robust adaptation to dynamic emotional intensity variations in real-world scenarios. This paper proposes SST-CL, a novel framework integrating spatial-temporal transformers with curriculum learning. Our method introduces two core components: a spatial encoder that models inter-channel relationships and a temporal encoder that captures multi-scale dependencies through windowed attention mechanisms, enabling simultaneous extraction of spatial correlations and temporal dynamics from EEG signals. Complementing this architecture, an intensity-aware curriculum learning strategy progressively guides training from high-intensity to low-intensity emotional states through dynamic sample scheduling based on a dual difficulty assessment. Comprehensive experiments on three benchmark datasets demonstrate state-of-the-art performance across various emotional intensity levels, with ablation studies confirming the necessity of both architectural components and the curriculum learning mechanism. 

---
# Rethinking Suicidal Ideation Detection: A Trustworthy Annotation Framework and Cross-Lingual Model Evaluation 

**Authors**: Amina Dzafic, Merve Kavut, Ulya Bayram  

**Link**: [PDF](https://arxiv.org/pdf/2507.14693)  

**Abstract**: Suicidal ideation detection is critical for real-time suicide prevention, yet its progress faces two under-explored challenges: limited language coverage and unreliable annotation practices. Most available datasets are in English, but even among these, high-quality, human-annotated data remains scarce. As a result, many studies rely on available pre-labeled datasets without examining their annotation process or label reliability. The lack of datasets in other languages further limits the global realization of suicide prevention via artificial intelligence (AI). In this study, we address one of these gaps by constructing a novel Turkish suicidal ideation corpus derived from social media posts and introducing a resource-efficient annotation framework involving three human annotators and two large language models (LLMs). We then address the remaining gaps by performing a bidirectional evaluation of label reliability and model consistency across this dataset and three popular English suicidal ideation detection datasets, using transfer learning through eight pre-trained sentiment and emotion classifiers. These transformers help assess annotation consistency and benchmark model performance against manually labeled data. Our findings underscore the need for more rigorous, language-inclusive approaches to annotation and evaluation in mental health natural language processing (NLP) while demonstrating the questionable performance of popular models with zero-shot transfer learning. We advocate for transparency in model training and dataset construction in mental health NLP, prioritizing data and model reliability. 

---
# Mind the Gap: A Review of Arabic Post-Training Datasets and Their Limitations 

**Authors**: Mohammed Alkhowaiter, Norah Alshahrani, Saied Alshahrani, Reem I. Masoud, Alaa Alzahrani, Deema Alnuhait, Emad A. Alghamdi, Khalid Almubarak  

**Link**: [PDF](https://arxiv.org/pdf/2507.14688)  

**Abstract**: Post-training has emerged as a crucial technique for aligning pre-trained Large Language Models (LLMs) with human instructions, significantly enhancing their performance across a wide range of tasks. Central to this process is the quality and diversity of post-training datasets. This paper presents a review of publicly available Arabic post-training datasets on the Hugging Face Hub, organized along four key dimensions: (1) LLM Capabilities (e.g., Question Answering, Translation, Reasoning, Summarization, Dialogue, Code Generation, and Function Calling); (2) Steerability (e.g., persona and system prompts); (3) Alignment (e.g., cultural, safety, ethics, and fairness), and (4) Robustness. Each dataset is rigorously evaluated based on popularity, practical adoption, recency and maintenance, documentation and annotation quality, licensing transparency, and scientific contribution. Our review revealed critical gaps in the development of Arabic post-training datasets, including limited task diversity, inconsistent or missing documentation and annotation, and low adoption across the community. Finally, the paper discusses the implications of these gaps on the progress of Arabic LLMs and applications while providing concrete recommendations for future efforts in post-training dataset development. 

---
# WSI-Agents: A Collaborative Multi-Agent System for Multi-Modal Whole Slide Image Analysis 

**Authors**: Xinheng Lyu, Yuci Liang, Wenting Chen, Meidan Ding, Jiaqi Yang, Guolin Huang, Daokun Zhang, Xiangjian He, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.14680)  

**Abstract**: Whole slide images (WSIs) are vital in digital pathology, enabling gigapixel tissue analysis across various pathological tasks. While recent advancements in multi-modal large language models (MLLMs) allow multi-task WSI analysis through natural language, they often underperform compared to task-specific models. Collaborative multi-agent systems have emerged as a promising solution to balance versatility and accuracy in healthcare, yet their potential remains underexplored in pathology-specific domains. To address these issues, we propose WSI-Agents, a novel collaborative multi-agent system for multi-modal WSI analysis. WSI-Agents integrates specialized functional agents with robust task allocation and verification mechanisms to enhance both task-specific accuracy and multi-task versatility through three components: (1) a task allocation module assigning tasks to expert agents using a model zoo of patch and WSI level MLLMs, (2) a verification mechanism ensuring accuracy through internal consistency checks and external validation using pathology knowledge bases and domain-specific models, and (3) a summary module synthesizing the final summary with visual interpretation maps. Extensive experiments on multi-modal WSI benchmarks show WSI-Agents's superiority to current WSI MLLMs and medical agent frameworks across diverse tasks. 

---
# GCC-Spam: Spam Detection via GAN, Contrastive Learning, and Character Similarity Networks 

**Authors**: Zixin Xu, Zhijie Wang, Zhiyuan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14679)  

**Abstract**: The exponential growth of spam text on the Internet necessitates robust detection mechanisms to mitigate risks such as information leakage and social instability. This work addresses two principal challenges: adversarial strategies employed by spammers and the scarcity of labeled data. We propose a novel spam-text detection framework GCC-Spam, which integrates three core innovations. First, a character similarity network captures orthographic and phonetic features to counter character-obfuscation attacks and furthermore produces sentence embeddings for downstream classification. Second, contrastive learning enhances discriminability by optimizing the latent-space distance between spam and normal texts. Third, a Generative Adversarial Network (GAN) generates realistic pseudo-spam samples to alleviate data scarcity while improving model robustness and classification accuracy. Extensive experiments on real-world datasets demonstrate that our model outperforms baseline approaches, achieving higher detection rates with significantly fewer labeled examples. 

---
# Artificial Intelligence in the Food Industry: Food Waste Estimation based on Computer Vision, a Brief Case Study in a University Dining Hall 

**Authors**: Shayan Rokhva, Babak Teimourpour  

**Link**: [PDF](https://arxiv.org/pdf/2507.14662)  

**Abstract**: Quantifying post-consumer food waste in institutional dining settings is essential for supporting data-driven sustainability strategies. This study presents a cost-effective computer vision framework that estimates plate-level food waste by utilizing semantic segmentation of RGB images taken before and after meal consumption across five Iranian dishes. Four fully supervised models (U-Net, U-Net++, and their lightweight variants) were trained using a capped dynamic inverse-frequency loss and AdamW optimizer, then evaluated through a comprehensive set of metrics, including Pixel Accuracy, Dice, IoU, and a custom-defined Distributional Pixel Agreement (DPA) metric tailored to the task. All models achieved satisfying performance, and for each food type, at least one model approached or surpassed 90% DPA, demonstrating strong alignment in pixel-wise proportion estimates. Lighter models with reduced parameter counts offered faster inference, achieving real-time throughput on an NVIDIA T4 GPU. Further analysis showed superior segmentation performance for dry and more rigid components (e.g., rice and fries), while more complex, fragmented, or viscous dishes, such as stews, showed reduced performance, specifically post-consumption. Despite limitations such as reliance on 2D imaging, constrained food variety, and manual data collection, the proposed framework is pioneering and represents a scalable, contactless solution for continuous monitoring of food consumption. This research lays foundational groundwork for automated, real-time waste tracking systems in large-scale food service environments and offers actionable insights and outlines feasible future directions for dining hall management and policymakers aiming to reduce institutional food waste. 

---
# AI-Powered Precision in Sport Taekwondo: Enhancing Fairness, Speed, and Trust in Competition (FST.ai) 

**Authors**: Keivan Shariatmadar, Ahmad Osman  

**Link**: [PDF](https://arxiv.org/pdf/2507.14657)  

**Abstract**: The integration of Artificial Intelligence (AI) into sports officiating represents a paradigm shift in how decisions are made in competitive environments. Traditional manual systems, even when supported by Instant Video Replay (IVR), often suffer from latency, subjectivity, and inconsistent enforcement, undermining fairness and athlete trust. This paper introduces this http URL, a novel AI-powered framework designed to enhance officiating in Sport Taekwondo, particularly focusing on the complex task of real-time head kick detection and scoring. Leveraging computer vision, deep learning, and edge inference, the system automates the identification and classification of key actions, significantly reducing decision time from minutes to seconds while improving consistency and transparency. Importantly, the methodology is not limited to Taekwondo. The underlying framework -- based on pose estimation, motion classification, and impact analysis -- can be adapted to a wide range of sports requiring action detection, such as judo, karate, fencing, or even team sports like football and basketball, where foul recognition or performance tracking is critical. By addressing one of Taekwondo's most challenging scenarios -- head kick scoring -- we demonstrate the robustness, scalability, and sport-agnostic potential of this http URL to transform officiating standards across multiple disciplines. 

---
# Cleanse: Uncertainty Estimation Approach Using Clustering-based Semantic Consistency in LLMs 

**Authors**: Minsuh Joo, Hyunsoo Cho  

**Link**: [PDF](https://arxiv.org/pdf/2507.14649)  

**Abstract**: Despite the outstanding performance of large language models (LLMs) across various NLP tasks, hallucinations in LLMs--where LLMs generate inaccurate responses--remains as a critical problem as it can be directly connected to a crisis of building safe and reliable LLMs. Uncertainty estimation is primarily used to measure hallucination levels in LLM responses so that correct and incorrect answers can be distinguished clearly. This study proposes an effective uncertainty estimation approach, \textbf{Cl}ust\textbf{e}ring-based sem\textbf{an}tic con\textbf{s}ist\textbf{e}ncy (\textbf{Cleanse}). Cleanse quantifies the uncertainty with the proportion of the intra-cluster consistency in the total consistency between LLM hidden embeddings which contain adequate semantic information of generations, by employing clustering. The effectiveness of Cleanse for detecting hallucination is validated using four off-the-shelf models, LLaMA-7B, LLaMA-13B, LLaMA2-7B and Mistral-7B and two question-answering benchmarks, SQuAD and CoQA. 

---
# VMask: Tunable Label Privacy Protection for Vertical Federated Learning via Layer Masking 

**Authors**: Juntao Tan, Lan Zhang, Zhonghao Hu, Kai Yang, Peng Ran, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14629)  

**Abstract**: Though vertical federated learning (VFL) is generally considered to be privacy-preserving, recent studies have shown that VFL system is vulnerable to label inference attacks originating from various attack surfaces. Among these attacks, the model completion (MC) attack is currently the most powerful one. Existing defense methods against it either sacrifice model accuracy or incur impractical computational overhead. In this paper, we propose VMask, a novel label privacy protection framework designed to defend against MC attack from the perspective of layer masking. Our key insight is to disrupt the strong correlation between input data and intermediate outputs by applying the secret sharing (SS) technique to mask layer parameters in the attacker's model. We devise a strategy for selecting critical layers to mask, reducing the overhead that would arise from naively applying SS to the entire model. Moreover, VMask is the first framework to offer a tunable privacy budget to defenders, allowing for flexible control over the levels of label privacy according to actual requirements. We built a VFL system, implemented VMask on it, and extensively evaluated it using five model architectures and 13 datasets with different modalities, comparing it to 12 other defense methods. The results demonstrate that VMask achieves the best privacy-utility trade-off, successfully thwarting the MC attack (reducing the label inference accuracy to a random guessing level) while preserving model performance (e.g., in Transformer-based model, the averaged drop of VFL model accuracy is only 0.09%). VMask's runtime is up to 60,846 times faster than cryptography-based methods, and it only marginally exceeds that of standard VFL by 1.8 times in a large Transformer-based model, which is generally acceptable. 

---
# VTarbel: Targeted Label Attack with Minimal Knowledge on Detector-enhanced Vertical Federated Learning 

**Authors**: Juntao Tan, Anran Li, Quanchao Liu, Peng Ran, Lan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14625)  

**Abstract**: Vertical federated learning (VFL) enables multiple parties with disjoint features to collaboratively train models without sharing raw data. While privacy vulnerabilities of VFL are extensively-studied, its security threats-particularly targeted label attacks-remain underexplored. In such attacks, a passive party perturbs inputs at inference to force misclassification into adversary-chosen labels. Existing methods rely on unrealistic assumptions (e.g., accessing VFL-model's outputs) and ignore anomaly detectors deployed in real-world systems. To bridge this gap, we introduce VTarbel, a two-stage, minimal-knowledge attack framework explicitly designed to evade detector-enhanced VFL inference. During the preparation stage, the attacker selects a minimal set of high-expressiveness samples (via maximum mean discrepancy), submits them through VFL protocol to collect predicted labels, and uses these pseudo-labels to train estimated detector and surrogate model on local features. In attack stage, these models guide gradient-based perturbations of remaining samples, crafting adversarial instances that induce targeted misclassifications and evade detection. We implement VTarbel and evaluate it against four model architectures, seven multimodal datasets, and two anomaly detectors. Across all settings, VTarbel outperforms four state-of-the-art baselines, evades detection, and retains effective against three representative privacy-preserving defenses. These results reveal critical security blind spots in current VFL deployments and underscore urgent need for robust, attack-aware defenses. 

---
# Retrieval-Augmented Clinical Benchmarking for Contextual Model Testing in Kenyan Primary Care: A Methodology Paper 

**Authors**: Fred Mutisya, Shikoh Gitau, Christine Syovata, Diana Oigara, Ibrahim Matende, Muna Aden, Munira Ali, Ryan Nyotu, Diana Marion, Job Nyangena, Nasubo Ongoma, Keith Mbae, Elizabeth Wamicha, Eric Mibuari, Jean Philbert Nsengemana, Talkmore Chidede  

**Link**: [PDF](https://arxiv.org/pdf/2507.14615)  

**Abstract**: Large Language Models(LLMs) hold promise for improving healthcare access in low-resource settings, but their effectiveness in African primary care remains underexplored. We present a methodology for creating a benchmark dataset and evaluation framework focused on Kenyan Level 2 and 3 clinical care. Our approach uses retrieval augmented generation (RAG) to ground clinical questions in Kenya's national guidelines, ensuring alignment with local standards. These guidelines were digitized, chunked, and indexed for semantic retrieval. Gemini Flash 2.0 Lite was then prompted with guideline excerpts to generate realistic clinical scenarios, multiple-choice questions, and rationale based answers in English and Swahili. Kenyan physicians co-created and refined the dataset, and a blinded expert review process ensured clinical accuracy, clarity, and cultural appropriateness. The resulting Alama Health QA dataset includes thousands of regulator-aligned question answer pairs across common outpatient conditions. Beyond accuracy, we introduce evaluation metrics that test clinical reasoning, safety, and adaptability such as rare case detection (Needle in the Haystack), stepwise logic (Decision Points), and contextual adaptability. Initial results reveal significant performance gaps when LLMs are applied to localized scenarios, consistent with findings that LLM accuracy is lower on African medical content than on US-based benchmarks. This work offers a replicable model for guideline-driven, dynamic benchmarking to support safe AI deployment in African health systems. 

---
# Enhancing POI Recommendation through Global Graph Disentanglement with POI Weighted Module 

**Authors**: Pei-Xuan Li, Wei-Yun Liang, Fandel Lin, Hsun-Ping Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2507.14612)  

**Abstract**: Next point of interest (POI) recommendation primarily predicts future activities based on users' past check-in data and current status, providing significant value to users and service providers. We observed that the popular check-in times for different POI categories vary. For example, coffee shops are crowded in the afternoon because people like to have coffee to refresh after meals, while bars are busy late at night. However, existing methods rarely explore the relationship between POI categories and time, which may result in the model being unable to fully learn users' tendencies to visit certain POI categories at different times. Additionally, existing methods for modeling time information often convert it into time embeddings or calculate the time interval and incorporate it into the model, making it difficult to capture the continuity of time. Finally, during POI prediction, various weighting information is often ignored, such as the popularity of each POI, the transition relationships between POIs, and the distances between POIs, leading to suboptimal performance. To address these issues, this paper proposes a novel next POI recommendation framework called Graph Disentangler with POI Weighted Module (GDPW). This framework aims to jointly consider POI category information and multiple POI weighting factors. Specifically, the proposed GDPW learns category and time representations through the Global Category Graph and the Global Category-Time Graph. Then, we disentangle category and time information through contrastive learning. After prediction, the final POI recommendation for users is obtained by weighting the prediction results based on the transition weights and distance relationships between POIs. We conducted experiments on two real-world datasets, and the results demonstrate that the proposed GDPW outperforms other existing models, improving performance by 3% to 11%. 

---
# Exp-Graph: How Connections Learn Facial Attributes in Graph-based Expression Recognition 

**Authors**: Nandani Sharma, Dinesh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.14608)  

**Abstract**: Facial expression recognition is crucial for human-computer interaction applications such as face animation, video surveillance, affective computing, medical analysis, etc. Since the structure of facial attributes varies with facial expressions, incorporating structural information into facial attributes is essential for facial expression recognition. In this paper, we propose Exp-Graph, a novel framework designed to represent the structural relationships among facial attributes using graph-based modeling for facial expression recognition. For facial attributes graph representation, facial landmarks are used as the graph's vertices. At the same time, the edges are determined based on the proximity of the facial landmark and the similarity of the local appearance of the facial attributes encoded using the vision transformer. Additionally, graph convolutional networks are utilized to capture and integrate these structural dependencies into the encoding of facial attributes, thereby enhancing the accuracy of expression recognition. Thus, Exp-Graph learns from the facial attribute graphs highly expressive semantic representations. On the other hand, the vision transformer and graph convolutional blocks help the framework exploit the local and global dependencies among the facial attributes that are essential for the recognition of facial expressions. We conducted comprehensive evaluations of the proposed Exp-Graph model on three benchmark datasets: Oulu-CASIA, eNTERFACE05, and AFEW. The model achieved recognition accuracies of 98.09\%, 79.01\%, and 56.39\%, respectively. These results indicate that Exp-Graph maintains strong generalization capabilities across both controlled laboratory settings and real-world, unconstrained environments, underscoring its effectiveness for practical facial expression recognition applications. 

---
# A Transformer-Based Conditional GAN with Multiple Instance Learning for UAV Signal Detection and Classification 

**Authors**: Haochen Liu, Jia Bi, Xiaomin Wang, Xin Yang, Ling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14592)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly used in surveillance, logistics, agriculture, disaster management, and military operations. Accurate detection and classification of UAV flight states, such as hovering, cruising, ascending, or transitioning, which are essential for safe and effective operations. However, conventional time series classification (TSC) methods often lack robustness and generalization for dynamic UAV environments, while state of the art(SOTA) models like Transformers and LSTM based architectures typically require large datasets and entail high computational costs, especially with high-dimensional data streams. This paper proposes a novel framework that integrates a Transformer-based Generative Adversarial Network (GAN) with Multiple Instance Locally Explainable Learning (MILET) to address these challenges in UAV flight state classification. The Transformer encoder captures long-range temporal dependencies and complex telemetry dynamics, while the GAN module augments limited datasets with realistic synthetic samples. MIL is incorporated to focus attention on the most discriminative input segments, reducing noise and computational overhead. Experimental results show that the proposed method achieves superior accuracy 96.5% on the DroneDetect dataset and 98.6% on the DroneRF dataset that outperforming other SOTA approaches. The framework also demonstrates strong computational efficiency and robust generalization across diverse UAV platforms and flight states, highlighting its potential for real-time deployment in resource constrained environments. 

---
# Backtranslation and paraphrasing in the LLM era? Comparing data augmentation methods for emotion classification 

**Authors**: Łukasz Radliński, Mateusz Guściora, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2507.14590)  

**Abstract**: Numerous domain-specific machine learning tasks struggle with data scarcity and class imbalance. This paper systematically explores data augmentation methods for NLP, particularly through large language models like GPT. The purpose of this paper is to examine and evaluate whether traditional methods such as paraphrasing and backtranslation can leverage a new generation of models to achieve comparable performance to purely generative methods. Methods aimed at solving the problem of data scarcity and utilizing ChatGPT were chosen, as well as an exemplary dataset. We conducted a series of experiments comparing four different approaches to data augmentation in multiple experimental setups. We then evaluated the results both in terms of the quality of generated data and its impact on classification performance. The key findings indicate that backtranslation and paraphrasing can yield comparable or even better results than zero and a few-shot generation of examples. 

---
# Performance comparison of medical image classification systems using TensorFlow Keras, PyTorch, and JAX 

**Authors**: Merjem Bećirović, Amina Kurtović, Nordin Smajlović, Medina Kapo, Amila Akagić  

**Link**: [PDF](https://arxiv.org/pdf/2507.14587)  

**Abstract**: Medical imaging plays a vital role in early disease diagnosis and monitoring. Specifically, blood microscopy offers valuable insights into blood cell morphology and the detection of hematological disorders. In recent years, deep learning-based automated classification systems have demonstrated high potential in enhancing the accuracy and efficiency of blood image analysis. However, a detailed performance analysis of specific deep learning frameworks appears to be lacking. This paper compares the performance of three popular deep learning frameworks, TensorFlow with Keras, PyTorch, and JAX, in classifying blood cell images from the publicly available BloodMNIST dataset. The study primarily focuses on inference time differences, but also classification performance for different image sizes. The results reveal variations in performance across frameworks, influenced by factors such as image resolution and framework-specific optimizations. Classification accuracy for JAX and PyTorch was comparable to current benchmarks, showcasing the efficiency of these frameworks for medical image classification. 

---
# Explainable Collaborative Problem Solving Diagnosis with BERT using SHAP and its Implications for Teacher Adoption 

**Authors**: Kester Wong, Sahan Bulathwela, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2507.14584)  

**Abstract**: The use of Bidirectional Encoder Representations from Transformers (BERT) model and its variants for classifying collaborative problem solving (CPS) has been extensively explored within the AI in Education community. However, limited attention has been given to understanding how individual tokenised words in the dataset contribute to the model's classification decisions. Enhancing the explainability of BERT-based CPS diagnostics is essential to better inform end users such as teachers, thereby fostering greater trust and facilitating wider adoption in education. This study undertook a preliminary step towards model transparency and explainability by using SHapley Additive exPlanations (SHAP) to examine how different tokenised words in transcription data contributed to a BERT model's classification of CPS processes. The findings suggested that well-performing classifications did not necessarily equate to a reasonable explanation for the classification decisions. Particular tokenised words were used frequently to affect classifications. The analysis also identified a spurious word, which contributed positively to the classification but was not semantically meaningful to the class. While such model transparency is unlikely to be useful to an end user to improve their practice, it can help them not to overrely on LLM diagnostics and ignore their human expertise. We conclude the workshop paper by noting that the extent to which the model appropriately uses the tokens for its classification is associated with the number of classes involved. It calls for an investigation into the exploration of ensemble model architectures and the involvement of human-AI complementarity for CPS diagnosis, since considerable human reasoning is still required for fine-grained discrimination of CPS subskills. 

---
# Exploring Human-AI Complementarity in CPS Diagnosis Using Unimodal and Multimodal BERT Models 

**Authors**: Kester Wong, Sahan Bulathwela, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2507.14579)  

**Abstract**: Detecting collaborative problem solving (CPS) indicators from dialogue using machine learning techniques is a significant challenge for the field of AI in Education. Recent studies have explored the use of Bidirectional Encoder Representations from Transformers (BERT) models on transcription data to reliably detect meaningful CPS indicators. A notable advancement involved the multimodal BERT variant, AudiBERT, which integrates speech and acoustic-prosodic audio features to enhance CPS diagnosis. Although initial results demonstrated multimodal improvements, the statistical significance of these enhancements remained unclear, and there was insufficient guidance on leveraging human-AI complementarity for CPS diagnosis tasks. This workshop paper extends the previous research by highlighting that the AudiBERT model not only improved the classification of classes that were sparse in the dataset, but it also had statistically significant class-wise improvements over the BERT model for classifications in the social-cognitive dimension. However, similar significant class-wise improvements over the BERT model were not observed for classifications in the affective dimension. A correlation analysis highlighted that larger training data was significantly associated with higher recall performance for both the AudiBERT and BERT models. Additionally, the precision of the BERT model was significantly associated with high inter-rater agreement among human coders. When employing the BERT model to diagnose indicators within these subskills that were well-detected by the AudiBERT model, the performance across all indicators was inconsistent. We conclude the paper by outlining a structured approach towards achieving human-AI complementarity for CPS diagnosis, highlighting the crucial inclusion of model explainability to support human agency and engagement in the reflective coding process. 

---
# Benchmarking GANs, Diffusion Models, and Flow Matching for T1w-to-T2w MRI Translation 

**Authors**: Andrea Moschetto, Lemuel Puglisi, Alec Sargood, Pierluigi Dell'Acqua, Francesco Guarnera, Sebastiano Battiato, Daniele Ravì  

**Link**: [PDF](https://arxiv.org/pdf/2507.14575)  

**Abstract**: Magnetic Resonance Imaging (MRI) enables the acquisition of multiple image contrasts, such as T1-weighted (T1w) and T2-weighted (T2w) scans, each offering distinct diagnostic insights. However, acquiring all desired modalities increases scan time and cost, motivating research into computational methods for cross-modal synthesis. To address this, recent approaches aim to synthesize missing MRI contrasts from those already acquired, reducing acquisition time while preserving diagnostic quality. Image-to-image (I2I) translation provides a promising framework for this task. In this paper, we present a comprehensive benchmark of generative models$\unicode{x2013}$specifically, Generative Adversarial Networks (GANs), diffusion models, and flow matching (FM) techniques$\unicode{x2013}$for T1w-to-T2w 2D MRI I2I translation. All frameworks are implemented with comparable settings and evaluated on three publicly available MRI datasets of healthy adults. Our quantitative and qualitative analyses show that the GAN-based Pix2Pix model outperforms diffusion and FM-based methods in terms of structural fidelity, image quality, and computational efficiency. Consistent with existing literature, these results suggest that flow-based models are prone to overfitting on small datasets and simpler tasks, and may require more data to match or surpass GAN performance. These findings offer practical guidance for deploying I2I translation techniques in real-world MRI workflows and highlight promising directions for future research in cross-modal medical image synthesis. Code and models are publicly available at this https URL. 

---
# LPS-GNN : Deploying Graph Neural Networks on Graphs with 100-Billion Edges 

**Authors**: Xu Cheng, Liang Yao, Feng He, Yukuo Cen, Yufei He, Chenhui Zhang, Wenzheng Feng, Hongyun Cai, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14570)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as powerful tools for various graph mining tasks, yet existing scalable solutions often struggle to balance execution efficiency with prediction accuracy. These difficulties stem from iterative message-passing techniques, which place significant computational demands and require extensive GPU memory, particularly when dealing with the neighbor explosion issue inherent in large-scale graphs. This paper introduces a scalable, low-cost, flexible, and efficient GNN framework called LPS-GNN, which can perform representation learning on 100 billion graphs with a single GPU in 10 hours and shows a 13.8% improvement in User Acquisition scenarios. We examine existing graph partitioning methods and design a superior graph partition algorithm named LPMetis. In particular, LPMetis outperforms current state-of-the-art (SOTA) approaches on various evaluation metrics. In addition, our paper proposes a subgraph augmentation strategy to enhance the model's predictive performance. It exhibits excellent compatibility, allowing the entire framework to accommodate various GNN algorithms. Successfully deployed on the Tencent platform, LPS-GNN has been tested on public and real-world datasets, achieving performance lifts of 8. 24% to 13. 89% over SOTA models in online applications. 

---
# Multimodal AI for Gastrointestinal Diagnostics: Tackling VQA in MEDVQA-GI 2025 

**Authors**: Sujata Gaihre, Amir Thapa Magar, Prasuna Pokharel, Laxmi Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2507.14544)  

**Abstract**: This paper describes our approach to Subtask 1 of the ImageCLEFmed MEDVQA 2025 Challenge, which targets visual question answering (VQA) for gastrointestinal endoscopy. We adopt the Florence model-a large-scale multimodal foundation model-as the backbone of our VQA pipeline, pairing a powerful vision encoder with a text encoder to interpret endoscopic images and produce clinically relevant answers. To improve generalization, we apply domain-specific augmentations that preserve medical features while increasing training diversity. Experiments on the KASVIR dataset show that fine-tuning Florence yields accurate responses on the official challenge metrics. Our results highlight the potential of large multimodal models in medical VQA and provide a strong baseline for future work on explainability, robustness, and clinical integration. The code is publicly available at: this https URL 

---
# Towards Efficient Privacy-Preserving Machine Learning: A Systematic Review from Protocol, Model, and System Perspectives 

**Authors**: Wenxuan Zeng, Tianshi Xu, Yi Chen, Yifan Zhou, Mingzhe Zhang, Jin Tan, Cheng Hong, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14519)  

**Abstract**: Privacy-preserving machine learning (PPML) based on cryptographic protocols has emerged as a promising paradigm to protect user data privacy in cloud-based machine learning services. While it achieves formal privacy protection, PPML often incurs significant efficiency and scalability costs due to orders of magnitude overhead compared to the plaintext counterpart. Therefore, there has been a considerable focus on mitigating the efficiency gap for PPML. In this survey, we provide a comprehensive and systematic review of recent PPML studies with a focus on cross-level optimizations. Specifically, we categorize existing papers into protocol level, model level, and system level, and review progress at each level. We also provide qualitative and quantitative comparisons of existing works with technical insights, based on which we discuss future research directions and highlight the necessity of integrating optimizations across protocol, model, and system levels. We hope this survey can provide an overarching understanding of existing approaches and potentially inspire future breakthroughs in the PPML field. As the field is evolving fast, we also provide a public GitHub repository to continuously track the developments, which is available at this https URL. 

---
# SDSC:A Structure-Aware Metric for Semantic Signal Representation Learning 

**Authors**: Jeyoung Lee, Hochul Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14516)  

**Abstract**: We propose the Signal Dice Similarity Coefficient (SDSC), a structure-aware metric function for time series self-supervised representation learning. Most Self-Supervised Learning (SSL) methods for signals commonly adopt distance-based objectives such as mean squared error (MSE), which are sensitive to amplitude, invariant to waveform polarity, and unbounded in scale. These properties hinder semantic alignment and reduce interpretability. SDSC addresses this by quantifying structural agreement between temporal signals based on the intersection of signed amplitudes, derived from the Dice Similarity Coefficient (DSC).Although SDSC is defined as a structure-aware metric, it can be used as a loss by subtracting from 1 and applying a differentiable approximation of the Heaviside function for gradient-based optimization. A hybrid loss formulation is also proposed to combine SDSC with MSE, improving stability and preserving amplitude where necessary. Experiments on forecasting and classification benchmarks demonstrate that SDSC-based pre-training achieves comparable or improved performance over MSE, particularly in in-domain and low-resource scenarios. The results suggest that structural fidelity in signal representations enhances the semantic representation quality, supporting the consideration of structure-aware metrics as viable alternatives to conventional distance-based methods. 

---
# Diffusion Models for Time Series Forecasting: A Survey 

**Authors**: Chen Su, Zhengzhou Cai, Yuanhe Tian, Zihong Zheng, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.14507)  

**Abstract**: Diffusion models, initially developed for image synthesis, demonstrate remarkable generative capabilities. Recently, their application has expanded to time series forecasting (TSF), yielding promising results. In this survey, we firstly introduce the standard diffusion models and their prevalent variants, explaining their adaptation to TSF tasks. We then provide a comprehensive review of diffusion models for TSF, paying special attention to the sources of conditional information and the mechanisms for integrating this conditioning within the models. In analyzing existing approaches using diffusion models for TSF, we provide a systematic categorization and a comprehensive summary of them in this survey. Furthermore, we examine several foundational diffusion models applied to TSF, alongside commonly used datasets and evaluation metrics. Finally, we discuss current limitations in these approaches and potential future research directions. Overall, this survey details recent progress and future prospects for diffusion models in TSF, serving as a reference for researchers in the field. 

---
# Neural Brownian Motion 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14499)  

**Abstract**: This paper introduces the Neural-Brownian Motion (NBM), a new class of stochastic processes for modeling dynamics under learned uncertainty. The NBM is defined axiomatically by replacing the classical martingale property with respect to linear expectation with one relative to a non-linear Neural Expectation Operator, $\varepsilon^\theta$, generated by a Backward Stochastic Differential Equation (BSDE) whose driver $f_\theta$ is parameterized by a neural network. Our main result is a representation theorem for a canonical NBM, which we define as a continuous $\varepsilon^\theta$-martingale with zero drift under the physical measure. We prove that, under a key structural assumption on the driver, such a canonical NBM exists and is the unique strong solution to a stochastic differential equation of the form ${\rm d} M_t = \nu_\theta(t, M_t) {\rm d} W_t$. Crucially, the volatility function $\nu_\theta$ is not postulated a priori but is implicitly defined by the algebraic constraint $g_\theta(t, M_t, \nu_\theta(t, M_t)) = 0$, where $g_\theta$ is a specialization of the BSDE driver. We develop the stochastic calculus for this process and prove a Girsanov-type theorem for the quadratic case, showing that an NBM acquires a drift under a new, learned measure. The character of this measure, whether pessimistic or optimistic, is endogenously determined by the learned parameters $\theta$, providing a rigorous foundation for models where the attitude towards uncertainty is a discoverable feature. 

---
# Benefit from Reference: Retrieval-Augmented Cross-modal Point Cloud Completion 

**Authors**: Hongye Hou, Liu Zhan, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14485)  

**Abstract**: Completing the whole 3D structure based on an incomplete point cloud is a challenging task, particularly when the residual point cloud lacks typical structural characteristics. Recent methods based on cross-modal learning attempt to introduce instance images to aid the structure feature learning. However, they still focus on each particular input class, limiting their generation abilities. In this work, we propose a novel retrieval-augmented point cloud completion framework. The core idea is to incorporate cross-modal retrieval into completion task to learn structural prior information from similar reference samples. Specifically, we design a Structural Shared Feature Encoder (SSFE) to jointly extract cross-modal features and reconstruct reference features as priors. Benefiting from a dual-channel control gate in the encoder, relevant structural features in the reference sample are enhanced and irrelevant information interference is suppressed. In addition, we propose a Progressive Retrieval-Augmented Generator (PRAG) that employs a hierarchical feature fusion mechanism to integrate reference prior information with input features from global to local. Through extensive evaluations on multiple datasets and real-world scenes, our method shows its effectiveness in generating fine-grained point clouds, as well as its generalization capability in handling sparse data and unseen categories. 

---
# DFQ-ViT: Data-Free Quantization for Vision Transformers without Fine-tuning 

**Authors**: Yujia Tong, Jingling Yuan, Tian Zhang, Jianquan Liu, Chuang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14481)  

**Abstract**: Data-Free Quantization (DFQ) enables the quantization of Vision Transformers (ViTs) without requiring access to data, allowing for the deployment of ViTs on devices with limited resources. In DFQ, the quantization model must be calibrated using synthetic samples, making the quality of these synthetic samples crucial. Existing methods fail to fully capture and balance the global and local features within the samples, resulting in limited synthetic data quality. Moreover, we have found that during inference, there is a significant difference in the distributions of intermediate layer activations between the quantized and full-precision models. These issues lead to a severe performance degradation of the quantized model. To address these problems, we propose a pipeline for Data-Free Quantization for Vision Transformers (DFQ-ViT). Specifically, we synthesize samples in order of increasing difficulty, effectively enhancing the quality of synthetic data. During the calibration and inference stage, we introduce the activation correction matrix for the quantized model to align the intermediate layer activations with those of the full-precision model. Extensive experiments demonstrate that DFQ-ViT achieves remarkable superiority over existing DFQ methods and its performance is on par with models quantized through real data. For example, the performance of DeiT-T with 3-bit weights quantization is 4.29% higher than the state-of-the-art. Our method eliminates the need for fine-tuning, which not only reduces computational overhead but also lowers the deployment barriers for edge devices. This characteristic aligns with the principles of Green Learning by improving energy efficiency and facilitating real-world applications in resource-constrained environments. 

---
# Strategyproofness and Monotone Allocation of Auction in Social Networks 

**Authors**: Yuhang Guo, Dong Hao, Bin Li, Mingyu Xiao, Bakh Khoussainov  

**Link**: [PDF](https://arxiv.org/pdf/2507.14472)  

**Abstract**: Strategyproofness in network auctions requires that bidders not only report their valuations truthfully, but also do their best to invite neighbours from the social network. In contrast to canonical auctions, where the value-monotone allocation in Myerson's Lemma is a cornerstone, a general principle of allocation rules for strategyproof network auctions is still missing. We show that, due to the absence of such a principle, even extensions to multi-unit network auctions with single-unit demand present unexpected difficulties, and all pioneering researches fail to be strategyproof. For the first time in this field, we identify two categories of monotone allocation rules on networks: Invitation-Depressed Monotonicity (ID-MON) and Invitation-Promoted Monotonicity (IP-MON). They encompass all existing allocation rules of network auctions as specific instances. For any given ID-MON or IP-MON allocation rule, we characterize the existence and sufficient conditions for the strategyproof payment rules, and show that among all such payment rules, the revenue-maximizing one exists and is computationally feasible. With these results, the obstacle of combinatorial network auction with single-minded bidders is now resolved. 

---
# Approximate Revenue Maximization for Diffusion Auctions 

**Authors**: Yifan Huang, Dong Hao, Zhiyi Fan, Yuhang Guo, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14470)  

**Abstract**: Reserve prices are widely used in practice. The problem of designing revenue-optimal auctions based on reserve price has drawn much attention in the auction design community. Although they have been extensively studied, most developments rely on the significant assumption that the target audience of the sale is directly reachable by the auctioneer, while a large portion of bidders in the economic network unaware of the sale are omitted. This work follows the diffusion auction design, which aims to extend the target audience of optimal auction theory to all entities in economic networks. We investigate the design of simple and provably near-optimal network auctions via reserve price. Using Bayesian approximation analysis, we provide a simple and explicit form of the reserve price function tailored to the most representative network auction. We aim to balance setting a sufficiently high reserve price to induce high revenue in a successful sale, and attracting more buyers from the network to increase the probability of a successful sale. This reserve price function preserves incentive compatibility for network auctions, allowing the seller to extract additional revenue beyond that achieved by the Myerson optimal auction. Specifically, if the seller has $\rho$ direct neighbours in a network of size $n$, this reserve price guarantees a $1-{1 \over \rho}$ approximation to the theoretical upper bound, i.e., the maximum possible revenue from any network of size $n$. This result holds for any size and any structure of the networked market. 

---
# Statistical and Algorithmic Foundations of Reinforcement Learning 

**Authors**: Yuejie Chi, Yuxin Chen, Yuting Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.14444)  

**Abstract**: As a paradigm for sequential decision making in unknown environments, reinforcement learning (RL) has received a flurry of attention in recent years. However, the explosion of model complexity in emerging applications and the presence of nonconvexity exacerbate the challenge of achieving efficient RL in sample-starved situations, where data collection is expensive, time-consuming, or even high-stakes (e.g., in clinical trials, autonomous systems, and online advertising). How to understand and enhance the sample and computational efficacies of RL algorithms is thus of great interest. In this tutorial, we aim to introduce several important algorithmic and theoretical developments in RL, highlighting the connections between new ideas and classical topics. Employing Markov Decision Processes as the central mathematical model, we cover several distinctive RL scenarios (i.e., RL with a simulator, online RL, offline RL, robust RL, and RL with human feedback), and present several mainstream RL approaches (i.e., model-based approach, value-based approach, and policy optimization). Our discussions gravitate around the issues of sample complexity, computational efficiency, as well as algorithm-dependent and information-theoretic lower bounds from a non-asymptotic viewpoint. 

---
# It's Not That Simple. An Analysis of Simple Test-Time Scaling 

**Authors**: Guojun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14419)  

**Abstract**: Prior work proposed simple test-time scaling, a method for replicating this scaling behavior with models distilled from o1-like models by manually controlling test-time compute: either scaling down by enforcing a maximum length or scaling up by iteratively appending "Wait" when the model is about to terminate its generation. This paper presents an analysis of simple test-time scaling and finds that the scaling behavior is largely attributed to scaling down by enforcing a maximum length. In contrast, fine-tuning on long CoT data distilled from o1-like models has no significant impact on scaling behavior, and scaling up by appending "Wait" leads to inconsistencies, as the model may oscillate between solutions. A key distinction exists between scaling down by enforcing a maximum length and scaling up test-time compute in o1-like models, such as DeepSeek-R1\@. These models are typically allowed to utilize as much compute as needed, with the only constraint being the model's maximum supported length. By learning to naturally scale up test-time compute during reinforcement learning, o1-like models surpass their peak performance when scaling up. In contrast, simple test-time scaling progressively imposes a lower upper limit on model performance as it scales down. While replicating the test-time scaling behavior of o1 models can be straightforward by scaling down, it is crucial to recognize that the goal of scaling test-time compute is to unlock higher performance -- beyond what the model could originally achieve -- rather than merely reproducing the appearance of scaling behavior. 

---
# Designing Conversational AI to Support Think-Aloud Practice in Technical Interview Preparation for CS Students 

**Authors**: Taufiq Daryanto, Sophia Stil, Xiaohan Ding, Daniel Manesh, Sang Won Lee, Tim Lee, Stephanie Lunn, Sarah Rodriguez, Chris Brown, Eugenia Rho  

**Link**: [PDF](https://arxiv.org/pdf/2507.14418)  

**Abstract**: One challenge in technical interviews is the think-aloud process, where candidates verbalize their thought processes while solving coding tasks. Despite its importance, opportunities for structured practice remain limited. Conversational AI offers potential assistance, but limited research explores user perceptions of its role in think-aloud practice. To address this gap, we conducted a study with 17 participants using an LLM-based technical interview practice tool. Participants valued AI's role in simulation, feedback, and learning from generated examples. Key design recommendations include promoting social presence in conversational AI for technical interview simulation, providing feedback beyond verbal content analysis, and enabling crowdsourced think-aloud examples through human-AI collaboration. Beyond feature design, we examined broader considerations, including intersectional challenges and potential strategies to address them, how AI-driven interview preparation could promote equitable learning in computing careers, and the need to rethink AI's role in interview practice by suggesting a research direction that integrates human-AI collaboration. 

---
# Incremental Causal Graph Learning for Online Cyberattack Detection in Cyber-Physical Infrastructures 

**Authors**: Arun Vignesh Malarkkan, Dongjie Wang, Haoyue Bai, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14387)  

**Abstract**: The escalating threat of cyberattacks on real-time critical infrastructures poses serious risks to public safety, demanding detection methods that effectively capture complex system interdependencies and adapt to evolving attack patterns. Traditional real-time anomaly detection techniques often suffer from excessive false positives due to their statistical sensitivity to high data variance and class imbalance. To address these limitations, recent research has explored modeling causal relationships among system components. However, prior work mainly focuses on offline causal graph-based approaches that require static historical data and fail to generalize to real-time settings. These methods are fundamentally constrained by: (1) their inability to adapt to dynamic shifts in data distribution without retraining, and (2) the risk of catastrophic forgetting when lacking timely supervision in live systems. To overcome these challenges, we propose INCADET, a novel framework for incremental causal graph learning tailored to real-time cyberattack detection. INCADET dynamically captures evolving system behavior by incrementally updating causal graphs across streaming time windows. The framework comprises three modules: 1) Early Symptom Detection: Detects transitions in system status using divergence in edge-weight distributions across sequential causal graphs. 2) Incremental Causal Graph Learning: Leverages experience replay and edge reinforcement to continually refine causal structures while preserving prior knowledge. 3) Causal Graph Classification: Employs Graph Convolutional Networks (GCNs) to classify system status using the learned causal graphs. Extensive experiments on real-world critical infrastructure datasets demonstrate that INCADET achieves superior accuracy, robustness, and adaptability compared to both static causal and deep temporal baselines in evolving attack scenarios. 

---
# Schemora: schema matching via multi-stage recommendation and metadata enrichment using off-the-shelf llms 

**Authors**: Osman Erman Gungor, Derak Paulsen, William Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14376)  

**Abstract**: Schema matching is essential for integrating heterogeneous data sources and enhancing dataset discovery, yet it remains a complex and resource-intensive problem. We introduce SCHEMORA, a schema matching framework that combines large language models with hybrid retrieval techniques in a prompt-based approach, enabling efficient identification of candidate matches without relying on labeled training data or exhaustive pairwise comparisons. By enriching schema metadata and leveraging both vector-based and lexical retrieval, SCHEMORA improves matching accuracy and scalability. Evaluated on the MIMIC-OMOP benchmark, it establishes new state-of-the-art performance, with gains of 7.49% in HitRate@5 and 3.75% in HitRate@3 over previous best results. To our knowledge, this is the first LLM-based schema matching method with an open-source implementation, accompanied by analysis that underscores the critical role of retrieval and provides practical guidance on model selection. 

---
# Text-to-SQL for Enterprise Data Analytics 

**Authors**: Albert Chen, Manas Bundele, Gaurav Ahlawat, Patrick Stetz, Zhitao Wang, Qiang Fei, Donghoon Jung, Audrey Chu, Bharadwaj Jayaraman, Ayushi Panth, Yatin Arora, Sourav Jain, Renjith Varma, Alexey Ilin, Iuliia Melnychuk, Chelsea Chueh, Joyan Sil, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14372)  

**Abstract**: The introduction of large language models has brought rapid progress on Text-to-SQL benchmarks, but it is not yet easy to build a working enterprise solution. In this paper, we present insights from building an internal chatbot that enables LinkedIn's product managers, engineers, and operations teams to self-serve data insights from a large, dynamic data lake. Our approach features three components. First, we construct a knowledge graph that captures up-to-date semantics by indexing database metadata, historical query logs, wikis, and code. We apply clustering to identify relevant tables for each team or product area. Second, we build a Text-to-SQL agent that retrieves and ranks context from the knowledge graph, writes a query, and automatically corrects hallucinations and syntax errors. Third, we build an interactive chatbot that supports various user intents, from data discovery to query writing to debugging, and displays responses in rich UI elements to encourage follow-up chats. Our chatbot has over 300 weekly users. Expert review shows that 53% of its responses are correct or close to correct on an internal benchmark set. Through ablation studies, we identify the most important knowledge graph and modeling components, offering a practical path for developing enterprise Text-to-SQL solutions. 

---
# Solo Connection: A Parameter Efficient Fine-Tuning Technique for Transformers 

**Authors**: Harsh Nilesh Pathak, Randy Paffenroth  

**Link**: [PDF](https://arxiv.org/pdf/2507.14353)  

**Abstract**: Parameter efficient fine tuning (PEFT) is a versatile and extensible approach for adapting a Large Language Model (LLM) for newer tasks. One of the most prominent PEFT approaches, Low Rank Adaptation (LoRA), primarily focuses on adjusting the attention weight matrices within individual decoder blocks of a Generative Pre trained Transformer (GPT2). In contrast, we introduce Solo Connection a novel method that adapts the representation at the decoder-block level rather than modifying individual weight matrices. Not only does Solo Connection outperform LoRA on E2E natural language generation benchmarks, but it also reduces the number of trainable parameters by 59% relative to LoRA and by more than 99% compared to full fine-tuning of GPT2, an early version of Large Language Models (LLMs). Solo Connection is also motivated by homotopy theory: we introduce a trainable linear transformation that gradually interpolates between a zero vector and the task-specific representation, enabling smooth and stable adaptation over time. While skip connections in the original 12 layer GPT2 are typically confined to individual decoder blocks, subsequent GPT2 variants scale up to 48 layers, and even larger language models can include 128 or more decoder blocks. These expanded architectures underscore the need to revisit how skip connections are employed during fine-tuning. This paper focuses on long skip connections that link outputs of different decoder blocks, potentially enhancing the model's ability to adapt to new tasks while leveraging pre-trained knowledge. 

---
# A Reproducibility Study of Product-side Fairness in Bundle Recommendation 

**Authors**: Huy-Son Nguyen, Yuanna Liu, Masoud Mansoury, Mohammad Alian Nejadi, Alan Hanjalic, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2507.14352)  

**Abstract**: Recommender systems are known to exhibit fairness issues, particularly on the product side, where products and their associated suppliers receive unequal exposure in recommended results. While this problem has been widely studied in traditional recommendation settings, its implications for bundle recommendation (BR) remain largely unexplored. This emerging task introduces additional complexity: recommendations are generated at the bundle level, yet user satisfaction and product (or supplier) exposure depend on both the bundle and the individual items it contains. Existing fairness frameworks and metrics designed for traditional recommender systems may not directly translate to this multi-layered setting. In this paper, we conduct a comprehensive reproducibility study of product-side fairness in BR across three real-world datasets using four state-of-the-art BR methods. We analyze exposure disparities at both the bundle and item levels using multiple fairness metrics, uncovering important patterns. Our results show that exposure patterns differ notably between bundles and items, revealing the need for fairness interventions that go beyond bundle-level assumptions. We also find that fairness assessments vary considerably depending on the metric used, reinforcing the need for multi-faceted evaluation. Furthermore, user behavior plays a critical role: when users interact more frequently with bundles than with individual items, BR systems tend to yield fairer exposure distributions across both levels. Overall, our findings offer actionable insights for building fairer bundle recommender systems and establish a vital foundation for future research in this emerging domain. 

---
# Influence Functions for Preference Dataset Pruning 

**Authors**: Daniel Fein, Gabriela Aranguiz-Dias  

**Link**: [PDF](https://arxiv.org/pdf/2507.14344)  

**Abstract**: Language models are commonly fine-tuned via reinforcement learning to alter their behavior or elicit new capabilities. Datasets used for these purposes, and particularly human preference datasets, are often noisy. The relatively small size post-training datasets, combined with parameter-efficient fine-tuning methods, enable the use of influence functions approximations to detect and prune training examples that are harmful to performance on a validation set. In this work, we adapt the TL;DR dataset for reward model training to demonstrate how conjugate-gradient approximated influence functions can be used to filter datasets. In our experiments, influence function filtering yields a small retraining accuracy uplift of 1.5% after removing 10% of training examples. We also show that gradient similarity outperforms influence functions for detecting helpful training examples. This suggests that local curvature is important for detecting harmful training examples, but less so for identifying helpful examples. 

---
# Fiduciary AI for the Future of Brain-Technology Interactions 

**Authors**: Abhishek Bhattacharjee, Jack Pilkington, Nita Farahany  

**Link**: [PDF](https://arxiv.org/pdf/2507.14339)  

**Abstract**: Brain foundation models represent a new frontier in AI: instead of processing text or images, these models interpret real-time neural signals from EEG, fMRI, and other neurotechnologies. When integrated with brain-computer interfaces (BCIs), they may enable transformative applications-from thought controlled devices to neuroprosthetics-by interpreting and acting on brain activity in milliseconds. However, these same systems pose unprecedented risks, including the exploitation of subconscious neural signals and the erosion of cognitive liberty. Users cannot easily observe or control how their brain signals are interpreted, creating power asymmetries that are vulnerable to manipulation. This paper proposes embedding fiduciary duties-loyalty, care, and confidentiality-directly into BCI-integrated brain foundation models through technical design. Drawing on legal traditions and recent advancements in AI alignment techniques, we outline implementable architectural and governance mechanisms to ensure these systems act in users' best interests. Placing brain foundation models on a fiduciary footing is essential to realizing their potential without compromising self-determination. 

---
# Age of Information Minimization in UAV-Enabled Integrated Sensing and Communication Systems 

**Authors**: Yu Bai, Yifan Zhang, Boxuan Xie, Zheng Chang, Yanru Zhang, Riku Jantti, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.14299)  

**Abstract**: Unmanned aerial vehicles (UAVs) equipped with integrated sensing and communication (ISAC) capabilities are envisioned to play a pivotal role in future wireless networks due to their enhanced flexibility and efficiency. However, jointly optimizing UAV trajectory planning, multi-user communication, and target sensing under stringent resource constraints and time-critical conditions remains a significant challenge. To address this, we propose an Age of Information (AoI)-centric UAV-ISAC system that simultaneously performs target sensing and serves multiple ground users, emphasizing information freshness as the core performance metric. We formulate a long-term average AoI minimization problem that jointly optimizes the UAV's flight trajectory and beamforming. To tackle the high-dimensional, non-convexity of this problem, we develop a deep reinforcement learning (DRL)-based algorithm capable of providing real-time decisions on UAV movement and beamforming for both radar sensing and multi-user communication. Specifically, a Kalman filter is employed for accurate target state prediction, regularized zero-forcing is utilized to mitigate inter-user interference, and the Soft Actor-Critic algorithm is applied for training the DRL agent on continuous actions. The proposed framework adaptively balances the trade-offs between sensing accuracy and communication quality. Extensive simulation results demonstrate that our proposed method consistently achieves lower average AoI compared to baseline approaches. 

---
# In-Depth and In-Breadth: Pre-training Multimodal Language Models Customized for Comprehensive Chart Understanding 

**Authors**: Wan-Cyuan Fan, Yen-Chun Chen, Mengchen Liu, Alexander Jacobson, Lu Yuan, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2507.14298)  

**Abstract**: Recent methods for customizing Large Vision Language Models (LVLMs) for domain-specific tasks have shown promising results in scientific chart comprehension. However, existing approaches face two major limitations: First, they rely on paired data from only a few chart types, limiting generalization to wide range of chart types. Secondly, they lack targeted pre-training for chart-data alignment, which hampers the model's understanding of underlying data. In this paper, we introduce ChartScope, an LVLM optimized for in-depth chart comprehension across diverse chart types. We propose an efficient data generation pipeline that synthesizes paired data for a wide range of chart types, along with a novel Dual-Path training strategy that enabling the model to succinctly capture essential data details while preserving robust reasoning capabilities by incorporating reasoning over the underlying data. Lastly, we establish ChartDQA, a new benchmark for evaluating not only question-answering at different levels but also underlying data understanding. Experimental results demonstrate that ChartScope significantly enhances comprehension on a wide range of chart types. The code and data are available at this https URL. 

---
# A Simple "Try Again" Can Elicit Multi-Turn LLM Reasoning 

**Authors**: Licheng Liu, Zihan Wang, Linjie Li, Chenwei Xu, Yiping Lu, Han Liu, Avirup Sil, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14295)  

**Abstract**: Multi-turn problem solving is critical yet challenging for Large Reasoning Models (LRMs) to reflect on their reasoning and revise from feedback. Existing Reinforcement Learning (RL) methods train large reasoning models on a single-turn paradigm with verifiable rewards. However, we observe that models trained with existing RL paradigms often lose their ability to solve problems across multiple turns and struggle to revise answers based on contextual feedback, leading to repetitive responses. We ask: can LRMs learn to reflect their answers in a multi-turn context? In this work, we find that training models with multi-turn RL using only unary feedback (e.g., "Let's try again") after wrong answers can improve both single-turn performance and multi-turn reasoning. We introduce Unary Feedback as Observation (UFO) for reinforcement learning, which uses minimal yet common unary user feedback during iterative problem solving. It can be easily applied to existing single-turn RL training setups. Experimental results show that RL training with UFO keeps single-turn performance and improves multi-turn reasoning accuracy by up to 14%, enabling language models to better react to feedback in multi-turn problem solving. To further minimize the number of turns needed for a correct answer while encouraging diverse reasoning when mistakes occur, we design reward structures that guide models to produce careful and deliberate answers in each turn. Code: this https URL 

---
# NuSeC: A Dataset for Nuclei Segmentation in Breast Cancer Histopathology Images 

**Authors**: Refik Samet, Nooshin Nemati, Emrah Hancer, Serpil Sak, Bilge Ayca Kirmizi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14272)  

**Abstract**: The NuSeC dataset is created by selecting 4 images with the size of 1024*1024 pixels from the slides of each patient among 25 patients. Therefore, there are a total of 100 images in the NuSeC dataset. To carry out a consistent comparative analysis between the methods that will be developed using the NuSeC dataset by the researchers in the future, we divide the NuSeC dataset 75% as the training set and 25% as the testing set. In detail, an image is randomly selected from 4 images of each patient among 25 patients to build the testing set, and then the remaining images are reserved for the training set. While the training set includes 75 images with around 30000 nuclei structures, the testing set includes 25 images with around 6000 nuclei structures. 

---
# MiDeSeC: A Dataset for Mitosis Detection and Segmentation in Breast Cancer Histopathology Images 

**Authors**: Refik Samet, Nooshin Nemati, Emrah Hancer, Serpil Sak, Bilge Ayca Kirmizi, Zeynep Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2507.14271)  

**Abstract**: The MiDeSeC dataset is created through H&E stained invasive breast carcinoma, no special type (NST) slides of 25 different patients captured at 40x magnification from the Department of Medical Pathology at Ankara University. The slides have been scanned by 3D Histech Panoramic p250 Flash-3 scanner and Olympus BX50 microscope. As several possible mitosis shapes exist, it is crucial to have a large dataset to cover all the cases. Accordingly, a total of 50 regions is selected from glass slides for 25 patients, each of regions with a size of 1024*1024 pixels. There are more than 500 mitoses in total in these 50 regions. Two-thirds of the regions are reserved for training, the other third for testing. 

---
# APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation 

**Authors**: Ravin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.14270)  

**Abstract**: We propose the APTx Neuron, a novel, unified neural computation unit that integrates non-linear activation and linear transformation into a single trainable expression. The APTx Neuron is derived from the APTx activation function, thereby eliminating the need for separate activation layers and making the architecture both computationally efficient and elegant. The proposed neuron follows the functional form $y = \sum_{i=1}^{n} ((\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i) + \delta$, where all parameters $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ are trainable. We validate our APTx Neuron-based architecture on the MNIST dataset, achieving up to 96.69\% test accuracy in just 20 epochs using approximately 332K trainable parameters. The results highlight the superior expressiveness and computational efficiency of the APTx Neuron compared to traditional neurons, pointing toward a new paradigm in unified neuron design and the architectures built upon it. 

---
# Bridging MOOCs, Smart Teaching, and AI: A Decade of Evolution Toward a Unified Pedagogy 

**Authors**: Bo Yuan, Jiazi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14266)  

**Abstract**: Over the past decade, higher education has evolved through three distinct paradigms: the emergence of Massive Open Online Courses (MOOCs), the integration of Smart Teaching technologies into classrooms, and the rise of AI-enhanced learning. Each paradigm is intended to address specific challenges in traditional education: MOOCs enable ubiquitous access to learning resources; Smart Teaching supports real-time interaction with data-driven insights; and generative AI offers personalized feedback and on-demand content generation. However, these paradigms are often implemented in isolation due to their disparate technological origins and policy-driven adoption. This paper examines the origins, strengths, and limitations of each paradigm, and advocates a unified pedagogical perspective that synthesizes their complementary affordances. We propose a three-layer instructional framework that combines the scalability of MOOCs, the responsiveness of Smart Teaching, and the adaptivity of AI. To demonstrate its feasibility, we present a curriculum design for a project-based course. The findings highlight the framework's potential to enhance learner engagement, support instructors, and enable personalized yet scalable learning. 

---
# Beyond DNS: Unlocking the Internet of AI Agents via the NANDA Index and Verified AgentFacts 

**Authors**: Ramesh Raskar, Pradyumna Chari, John Zinky, Mahesh Lambe, Jared James Grogan, Sichao Wang, Rajesh Ranjan, Rekha Singhal, Shailja Gupta, Robert Lincourt, Raghu Bala, Aditi Joshi, Abhishek Singh, Ayush Chopra, Dimitris Stripelis, Bhuwan B, Sumit Kumar, Maria Gorskikh  

**Link**: [PDF](https://arxiv.org/pdf/2507.14263)  

**Abstract**: The Internet is poised to host billions to trillions of autonomous AI agents that negotiate, delegate, and migrate in milliseconds and workloads that will strain DNS-centred identity and discovery. In this paper, we describe the NANDA index architecture, which we envision as a means for discoverability, identifiability and authentication in the internet of AI agents. We present an architecture where a minimal lean index resolves to dynamic, cryptographically verifiable AgentFacts that supports multi-endpoint routing, load balancing, privacy-preserving access, and credentialed capability assertions. Our architecture design delivers five concrete guarantees: (1) A quilt-like index proposal that supports both NANDA-native agents as well as third party agents being discoverable via the index, (2) rapid global resolution for newly spawned AI agents, (3) sub-second revocation and key rotation, (4) schema-validated capability assertions, and (5) privacy-preserving discovery across organisational boundaries via verifiable, least-disclosure queries. We formalize the AgentFacts schema, specify a CRDT-based update protocol, and prototype adaptive resolvers. The result is a lightweight, horizontally scalable foundation that unlocks secure, trust-aware collaboration for the next generation of the Internet of AI agents, without abandoning existing web infrastructure. 

---
# FAMST: Fast Approximate Minimum Spanning Tree Construction for Large-Scale and High-Dimensional Data 

**Authors**: Mahmood K. M. Almansoori, Miklos Telek  

**Link**: [PDF](https://arxiv.org/pdf/2507.14261)  

**Abstract**: We present Fast Approximate Minimum Spanning Tree (FAMST), a novel algorithm that addresses the computational challenges of constructing Minimum Spanning Trees (MSTs) for large-scale and high-dimensional datasets. FAMST utilizes a three-phase approach: Approximate Nearest Neighbor (ANN) graph construction, ANN inter-component connection, and iterative edge refinement. For a dataset of $n$ points in a $d$-dimensional space, FAMST achieves $\mathcal{O}(dn \log n)$ time complexity and $\mathcal{O}(dn + kn)$ space complexity when $k$ nearest neighbors are considered, which is a significant improvement over the $\mathcal{O}(n^2)$ time and space complexity of traditional methods.
Experiments across diverse datasets demonstrate that FAMST achieves remarkably low approximation errors while providing speedups of up to 1000$\times$ compared to exact MST algorithms. We analyze how the key hyperparameters, $k$ (neighborhood size) and $\lambda$ (inter-component edges), affect performance, providing practical guidelines for hyperparameter selection. FAMST enables MST-based analysis on datasets with millions of points and thousands of dimensions, extending the applicability of MST techniques to problem scales previously considered infeasible. 

---
# Impact of Code Context and Prompting Strategies on Automated Unit Test Generation with Modern General-Purpose Large Language Models 

**Authors**: Jakub Walczak, Piotr Tomalak, Artur Laskowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.14256)  

**Abstract**: Generative AI is gaining increasing attention in software engineering, where testing remains an indispensable reliability mechanism. According to the widely adopted testing pyramid, unit tests constitute the majority of test cases and are often schematic, requiring minimal domain expertise. Automatically generating such tests under the supervision of software engineers can significantly enhance productivity during the development phase of the software lifecycle.
This paper investigates the impact of code context and prompting strategies on the quality and adequacy of unit tests generated by various large language models (LLMs) across several families. The results show that including docstrings notably improves code adequacy, while further extending context to the full implementation yields definitely smaller gains. Notably, the chain-of-thought prompting strategy -- applied even to 'reasoning' models -- achieves the best results, with up to 96.3\% branch coverage, a 57\% average mutation score, and near-perfect compilation success rate. Among the evaluated models, M5 (Gemini 2.5 Pro) demonstrated superior performance in both mutation score and branch coverage being still in top in terms of compilation success rate.
All the code and resulting test suites are publicly available at this https URL. 

---
# Real-Time Communication-Aware Ride-Sharing Route Planning for Urban Air Mobility: A Multi-Source Hybrid Attention Reinforcement Learning Approach 

**Authors**: Yuejiao Xie, Maonan Wang, Di Zhou, Man-On Pun, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.14249)  

**Abstract**: Urban Air Mobility (UAM) systems are rapidly emerging as promising solutions to alleviate urban congestion, with path planning becoming a key focus area. Unlike ground transportation, UAM trajectory planning has to prioritize communication quality for accurate location tracking in constantly changing environments to ensure safety. Meanwhile, a UAM system, serving as an air taxi, requires adaptive planning to respond to real-time passenger requests, especially in ride-sharing scenarios where passenger demands are unpredictable and dynamic. However, conventional trajectory planning strategies based on predefined routes lack the flexibility to meet varied passenger ride demands. To address these challenges, this work first proposes constructing a radio map to evaluate the communication quality of urban airspace. Building on this, we introduce a novel Multi-Source Hybrid Attention Reinforcement Learning (MSHA-RL) framework for the challenge of effectively focusing on passengers and UAM locations, which arises from the significant dimensional disparity between the representations. This model first generates the alignment among diverse data sources with large gap dimensions before employing hybrid attention to balance global and local insights, thereby facilitating responsive, real-time path planning. Extensive experimental results demonstrate that the approach enables communication-compliant trajectory planning, reducing travel time and enhancing operational efficiency while prioritizing passenger safety. 

---
# Breaking the Illusion of Security via Interpretation: Interpretable Vision Transformer Systems under Attack 

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Hyoungshick Kim, Tamer Abuhmed  

**Link**: [PDF](https://arxiv.org/pdf/2507.14248)  

**Abstract**: Vision transformer (ViT) models, when coupled with interpretation models, are regarded as secure and challenging to deceive, making them well-suited for security-critical domains such as medical applications, autonomous vehicles, drones, and robotics. However, successful attacks on these systems can lead to severe consequences. Recent research on threats targeting ViT models primarily focuses on generating the smallest adversarial perturbations that can deceive the models with high confidence, without considering their impact on model interpretations. Nevertheless, the use of interpretation models can effectively assist in detecting adversarial examples. This study investigates the vulnerability of transformer models to adversarial attacks, even when combined with interpretation models. We propose an attack called "AdViT" that generates adversarial examples capable of misleading both a given transformer model and its coupled interpretation model. Through extensive experiments on various transformer models and two transformer-based interpreters, we demonstrate that AdViT achieves a 100% attack success rate in both white-box and black-box scenarios. In white-box scenarios, it reaches up to 98% misclassification confidence, while in black-box scenarios, it reaches up to 76% misclassification confidence. Remarkably, AdViT consistently generates accurate interpretations in both scenarios, making the adversarial examples more difficult to detect. 

---
# A million-scale dataset and generalizable foundation model for nanomaterial-protein interactions 

**Authors**: Hengjie Yu, Kenneth A. Dawson, Haiyun Yang, Shuya Liu, Yan Yan, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.14245)  

**Abstract**: Unlocking the potential of nanomaterials in medicine and environmental science hinges on understanding their interactions with proteins, a complex decision space where AI is poised to make a transformative impact. However, progress has been hindered by limited datasets and the restricted generalizability of existing models. Here, we propose NanoPro-3M, the largest nanomaterial-protein interaction dataset to date, comprising over 3.2 million samples and 37,000 unique proteins. Leveraging this, we present NanoProFormer, a foundational model that predicts nanomaterial-protein affinities through multimodal representation learning, demonstrating strong generalization, handling missing features, and unseen nanomaterials or proteins. We show that multimodal modeling significantly outperforms single-modality approaches and identifies key determinants of corona formation. Furthermore, we demonstrate its applicability to a range of downstream tasks through zero-shot inference and fine-tuning. Together, this work establishes a solid foundation for high-performance and generalized prediction of nanomaterial-protein interaction endpoints, reducing experimental reliance and accelerating various in vitro applications. 

---
# Culling Misinformation from Gen AI: Toward Ethical Curation and Refinement 

**Authors**: Prerana Khatiwada, Grace Donaher, Jasymyn Navarro, Lokesh Bhatta  

**Link**: [PDF](https://arxiv.org/pdf/2507.14242)  

**Abstract**: While Artificial Intelligence (AI) is not a new field, recent developments, especially with the release of generative tools like ChatGPT, have brought it to the forefront of the minds of industry workers and academic folk alike. There is currently much talk about AI and its ability to reshape many everyday processes as we know them through automation. It also allows users to expand their ideas by suggesting things they may not have thought of on their own and provides easier access to information. However, not all of the changes this technology will bring or has brought so far are positive; this is why it is extremely important for all modern people to recognize and understand the risks before using these tools and allowing them to cause harm. This work takes a position on better understanding many equity concerns and the spread of misinformation that result from new AI, in this case, specifically ChatGPT and deepfakes, and encouraging collaboration with law enforcement, developers, and users to reduce harm. Considering many academic sources, it warns against these issues, analyzing their cause and impact in fields including healthcare, education, science, academia, retail, and finance. Lastly, we propose a set of future-facing guidelines and policy considerations to solve these issues while still enabling innovation in these fields, this responsibility falling upon users, developers, and government entities. 

---
# Promptomatix: An Automatic Prompt Optimization Framework for Large Language Models 

**Authors**: Rithesh Murthy, Ming Zhu, Liangwei Yang, Jielin Qiu, Juntao Tan, Shelby Heinecke, Huan Wang, Caiming Xiong, Silvio Savarese  

**Link**: [PDF](https://arxiv.org/pdf/2507.14241)  

**Abstract**: Large Language Models (LLMs) perform best with well-crafted prompts, yet prompt engineering remains manual, inconsistent, and inaccessible to non-experts. We introduce Promptomatix, an automatic prompt optimization framework that transforms natural language task descriptions into high-quality prompts without requiring manual tuning or domain expertise. Promptomatix supports both a lightweight meta-prompt-based optimizer and a DSPy-powered compiler, with modular design enabling future extension to more advanced frameworks. The system analyzes user intent, generates synthetic training data, selects prompting strategies, and refines prompts using cost-aware objectives. Evaluated across 5 task categories, Promptomatix achieves competitive or superior performance compared to existing libraries, while reducing prompt length and computational overhead making prompt optimization scalable and efficient. 

---
# HuggingGraph: Understanding the Supply Chain of LLM Ecosystem 

**Authors**: Mohammad Shahedur Rahman, Peng Gao, Yuede Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.14240)  

**Abstract**: Large language models (LLMs) leverage deep learning to process and predict sequences of words from context, enabling them to perform various NLP tasks, such as translation, summarization, question answering, and content generation. However, the growing size and complexity of developing, training, and deploying advanced LLMs require extensive computational resources and large datasets. This creates a barrier for users. As a result, platforms that host models and datasets are widely used. For example, Hugging Face, one of the most popular platforms, hosted 1.8 million models and 450K datasets by June 2025, with no sign of slowing down. Since many LLMs are built from base models, pre-trained models, and external datasets, they can inherit vulnerabilities, biases, or malicious components from earlier models or datasets. Therefore, it is critical to understand the origin and development of these components to better detect potential risks, improve model fairness, and ensure compliance. Motivated by this, our project aims to study the relationships between models and datasets, which are core components of the LLM supply chain. First, we design a method to systematically collect LLM supply chain data. Using this data, we build a directed heterogeneous graph to model the relationships between models and datasets, resulting in a structure with 397,376 nodes and 453,469 edges. We then perform various analyses and uncover several findings, such as: (i) the LLM supply chain graph is large, sparse, and follows a power-law degree distribution; (ii) it features a densely connected core and a fragmented periphery; (iii) datasets play pivotal roles in training; (iv) strong interdependence exists between models and datasets; and (v) the graph is dynamic, with daily updates reflecting the ecosystem's ongoing evolution. 

---
# CCL-XCoT: An Efficient Cross-Lingual Knowledge Transfer Method for Mitigating Hallucination Generation 

**Authors**: Weihua Zheng, Roy Ka-Wei Lee, Zhengyuan Liu, Kui Wu, AiTi Aw, Bowei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.14239)  

**Abstract**: Multilingual Large Language Models(MLLMs) demonstrate strong generalization across languages, yet they remain prone to hallucinations, especially in low-resource languages, due to training data imbalances. These hallucinations, which include inaccurate or fabricated outputs, are particularly problematic in domain-specific generation tasks (Chataigner et al., 2024). To address this challenge, we propose CCL-XCoT(Curriculum-based Contrastive Learning-based Cross-lingual Chain-of-Thought), a two-stage fine-tuning framework for mitigating hallucination in MLLMs. Our approach first enhances cross-lingual semantic alignment through curriculum-based contrastive learning combined with next-token prediction during continued pre-training. Building on this foundation, we then introduce a cross-lingual Chain-of-Thought (XCoT) prompting strategy during instruction fine-tuning, which guides the model to reason in a high-resource language before generating answers in the target low-resource language. Experimental results show that CCL-XCoT reduces hallucination rates by up to 62% and substantially improves factual knowledge transfer across language pairs, without relying on external retrieval or multi-model ensembles. 

---
# Language Models Change Facts Based on the Way You Talk 

**Authors**: Matthew Kearney, Reuben Binns, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2507.14238)  

**Abstract**: Large language models (LLMs) are increasingly being used in user-facing applications, from providing medical consultations to job interview advice. Recent research suggests that these models are becoming increasingly proficient at inferring identity information about the author of a piece of text from linguistic patterns as subtle as the choice of a few words. However, little is known about how LLMs use this information in their decision-making in real-world applications. We perform the first comprehensive analysis of how identity markers present in a user's writing bias LLM responses across five different high-stakes LLM applications in the domains of medicine, law, politics, government benefits, and job salaries. We find that LLMs are extremely sensitive to markers of identity in user queries and that race, gender, and age consistently influence LLM responses in these applications. For instance, when providing medical advice, we find that models apply different standards of care to individuals of different ethnicities for the same symptoms; we find that LLMs are more likely to alter answers to align with a conservative (liberal) political worldview when asked factual questions by older (younger) individuals; and that LLMs recommend lower salaries for non-White job applicants and higher salaries for women compared to men. Taken together, these biases mean that the use of off-the-shelf LLMs for these applications may cause harmful differences in medical care, foster wage gaps, and create different political factual realities for people of different identities. Beyond providing an analysis, we also provide new tools for evaluating how subtle encoding of identity in users' language choices impacts model decisions. Given the serious implications of these findings, we recommend that similar thorough assessments of LLM use in user-facing applications are conducted before future deployment. 

---
# U-DREAM: Unsupervised Dereverberation guided by a Reverberation Model 

**Authors**: Louis Bahrman, Mathieu Fontaine, Gaël Richard  

**Link**: [PDF](https://arxiv.org/pdf/2507.14237)  

**Abstract**: This paper explores the outcome of training state-ofthe-art dereverberation models with supervision settings ranging from weakly-supervised to fully unsupervised, relying solely on reverberant signals and an acoustic model for training. Most of the existing deep learning approaches typically require paired dry and reverberant data, which are difficult to obtain in practice. We develop instead a sequential learning strategy motivated by a bayesian formulation of the dereverberation problem, wherein acoustic parameters and dry signals are estimated from reverberant inputs using deep neural networks, guided by a reverberation matching loss. Our most data-efficient variant requires only 100 reverberation-parameter-labelled samples to outperform an unsupervised baseline, demonstrating the effectiveness and practicality of the proposed method in low-resource scenarios. 

---
# Beyond Architectures: Evaluating the Role of Contextual Embeddings in Detecting Bipolar Disorder on Social Media 

**Authors**: Khalid Hasan, Jamil Saquer  

**Link**: [PDF](https://arxiv.org/pdf/2507.14231)  

**Abstract**: Bipolar disorder is a chronic mental illness frequently underdiagnosed due to subtle early symptoms and social stigma. This paper explores the advanced natural language processing (NLP) models for recognizing signs of bipolar disorder based on user-generated social media text. We conduct a comprehensive evaluation of transformer-based models (BERT, RoBERTa, ALBERT, ELECTRA, DistilBERT) and Long Short Term Memory (LSTM) models based on contextualized (BERT) and static (GloVe, Word2Vec) word embeddings. Experiments were performed on a large, annotated dataset of Reddit posts after confirming their validity through sentiment variance and judgmental analysis. Our results demonstrate that RoBERTa achieves the highest performance among transformer models with an F1 score of ~98% while LSTM models using BERT embeddings yield nearly identical results. In contrast, LSTMs trained on static embeddings fail to capture meaningful patterns, scoring near-zero F1. These findings underscore the critical role of contextual language modeling in detecting bipolar disorder. In addition, we report model training times and highlight that DistilBERT offers an optimal balance between efficiency and accuracy. In general, our study offers actionable insights for model selection in mental health NLP applications and validates the potential of contextualized language models to support early bipolar disorder screening. 

---
# Intent-Based Network for RAN Management with Large Language Models 

**Authors**: Fransiscus Asisi Bimo, Maria Amparo Canaveras Galdon, Chun-Kai Lai, Ray-Guang Cheng, Edwin K. P. Chong  

**Link**: [PDF](https://arxiv.org/pdf/2507.14230)  

**Abstract**: Advanced intelligent automation becomes an important feature to deal with the increased complexity in managing wireless networks. This paper proposes a novel automation approach of intent-based network for Radio Access Networks (RANs) management by leveraging Large Language Models (LLMs). The proposed method enhances intent translation, autonomously interpreting high-level objectives, reasoning over complex network states, and generating precise configurations of the RAN by integrating LLMs within an agentic architecture. We propose a structured prompt engineering technique and demonstrate that the network can automatically improve its energy efficiency by dynamically optimizing critical RAN parameters through a closed-loop mechanism. It showcases the potential to enable robust resource management in RAN by adapting strategies based on real-time feedback via LLM-orchestrated agentic systems. 

---
# Domain Generalization via Pareto Optimal Gradient Matching 

**Authors**: Khoi Do, Duong Nguyen, Nam-Khanh Le, Quoc-Viet Pham, Binh-Son Hua, Won-Joo Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14227)  

**Abstract**: In this study, we address the gradient-based domain generalization problem, where predictors aim for consistent gradient directions across different domains. Existing methods have two main challenges. First, minimization of gradient empirical distance or gradient inner products (GIP) leads to gradient fluctuations among domains, thereby hindering straightforward learning. Second, the direct application of gradient learning to the joint loss function can incur high computation overheads due to second-order derivative approximation. To tackle these challenges, we propose a new Pareto Optimality Gradient Matching (POGM) method. In contrast to existing methods that add gradient matching as regularization, we leverage gradient trajectories as collected data and apply independent training at the meta-learner. In the meta-update, we maximize GIP while limiting the learned gradient from deviating too far from the empirical risk minimization gradient trajectory. By doing so, the aggregate gradient can incorporate knowledge from all domains without suffering gradient fluctuation towards any particular domain. Experimental evaluations on datasets from DomainBed demonstrate competitive results yielded by POGM against other baselines while achieving computational efficiency. 

---
# Multi-Granular Discretization for Interpretable Generalization in Precise Cyberattack Identification 

**Authors**: Wen-Cheng Chung, Shu-Ting Huang, Hao-Ting Pai  

**Link**: [PDF](https://arxiv.org/pdf/2507.14223)  

**Abstract**: Explainable intrusion detection systems (IDS) are now recognized as essential for mission-critical networks, yet most "XAI" pipelines still bolt an approximate explainer onto an opaque classifier, leaving analysts with partial and sometimes misleading insights. The Interpretable Generalization (IG) mechanism, published in IEEE Transactions on Information Forensics and Security, eliminates that bottleneck by learning coherent patterns - feature combinations unique to benign or malicious traffic - and turning them into fully auditable rules. IG already delivers outstanding precision, recall, and AUC on NSL-KDD, UNSW-NB15, and UKM-IDS20, even when trained on only 10% of the data. To raise precision further without sacrificing transparency, we introduce Multi-Granular Discretization (IG-MD), which represents every continuous feature at several Gaussian-based resolutions. On UKM-IDS20, IG-MD lifts precision by greater than or equal to 4 percentage points across all nine train-test splits while preserving recall approximately equal to 1.0, demonstrating that a single interpretation-ready model can scale across domains without bespoke tuning. 

---
# Artificial Intelligence for Green Hydrogen Yield Prediction and Site Suitability using SHAP-Based Composite Index: Focus on Oman 

**Authors**: Obumneme Zimuzor Nwafor, Mohammed Abdul Majeed Al Hooti  

**Link**: [PDF](https://arxiv.org/pdf/2507.14219)  

**Abstract**: As nations seek sustainable alternatives to fossil fuels, green hydrogen has emerged as a promising strategic pathway toward decarbonisation, particularly in solar-rich arid regions. However, identifying optimal locations for hydrogen production requires the integration of complex environmental, atmospheric, and infrastructural factors, often compounded by limited availability of direct hydrogen yield data. This study presents a novel Artificial Intelligence (AI) framework for computing green hydrogen yield and site suitability index using mean absolute SHAP (SHapley Additive exPlanations) values. This framework consists of a multi-stage pipeline of unsupervised multi-variable clustering, supervised machine learning classifier and SHAP algorithm. The pipeline trains on an integrated meteorological, topographic and temporal dataset and the results revealed distinct spatial patterns of suitability and relative influence of the variables. With model predictive accuracy of 98%, the result also showed that water proximity, elevation and seasonal variation are the most influential factors determining green hydrogen site suitability in Oman with mean absolute shap values of 2.470891, 2.376296 and 1.273216 respectively. Given limited or absence of ground-truth yield data in many countries that have green hydrogen prospects and ambitions, this study offers an objective and reproducible alternative to subjective expert weightings, thus allowing the data to speak for itself and potentially discover novel latent groupings without pre-imposed assumptions. This study offers industry stakeholders and policymakers a replicable and scalable tool for green hydrogen infrastructure planning and other decision making in data-scarce regions. 

---
# Cognitive Castes: Artificial Intelligence, Epistemic Stratification, and the Dissolution of Democratic Discourse 

**Authors**: Craig S Wright  

**Link**: [PDF](https://arxiv.org/pdf/2507.14218)  

**Abstract**: Artificial intelligence functions not as an epistemic leveller, but as an accelerant of cognitive stratification, entrenching and formalising informational castes within liberal-democratic societies. Synthesising formal epistemology, political theory, algorithmic architecture, and economic incentive structures, the argument traces how contemporary AI systems selectively amplify the reasoning capacity of individuals equipped with recursive abstraction, symbolic logic, and adversarial interrogation, whilst simultaneously pacifying the cognitively untrained through engagement-optimised interfaces. Fluency replaces rigour, immediacy displaces reflection, and procedural reasoning is eclipsed by reactive suggestion. The result is a technocratic realignment of power: no longer grounded in material capital alone, but in the capacity to navigate, deconstruct, and manipulate systems of epistemic production. Information ceases to be a commons; it becomes the substrate through which consent is manufactured and autonomy subdued. Deliberative democracy collapses not through censorship, but through the erosion of interpretive agency. The proposed response is not technocratic regulation, nor universal access, but the reconstruction of rational autonomy as a civic mandate, codified in education, protected by epistemic rights, and structurally embedded within open cognitive infrastructure. 

---
# PRATA: A Framework to Enable Predictive QoS in Vehicular Networks via Artificial Intelligence 

**Authors**: Federico Mason, Tommaso Zugno, Matteo Drago, Marco Giordani, Mate Boban, Michele Zorzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14211)  

**Abstract**: Predictive Quality of Service (PQoS) makes it possible to anticipate QoS changes, e.g., in wireless networks, and trigger appropriate countermeasures to avoid performance degradation. Hence, PQoS is extremely useful for automotive applications such as teleoperated driving, which poses strict constraints in terms of latency and reliability. A promising tool for PQoS is given by Reinforcement Learning (RL), a methodology that enables the design of decision-making strategies for stochastic optimization. In this manuscript, we present PRATA, a new simulation framework to enable PRedictive QoS based on AI for Teleoperated driving Applications. PRATA consists of a modular pipeline that includes (i) an end-to-end protocol stack to simulate the 5G Radio Access Network (RAN), (ii) a tool for generating automotive data, and (iii) an Artificial Intelligence (AI) unit to optimize PQoS decisions. To prove its utility, we use PRATA to design an RL unit, named RAN-AI, to optimize the segmentation level of teleoperated driving data in the event of resource saturation or channel degradation. Hence, we show that the RAN-AI entity efficiently balances the trade-off between QoS and Quality of Experience (QoE) that characterize teleoperated driving applications, almost doubling the system performance compared to baseline approaches. In addition, by varying the learning settings of the RAN-AI entity, we investigate the impact of the state space and the relative cost of acquiring network data that are necessary for the implementation of RL. 

---
# Mitigating Trojanized Prompt Chains in Educational LLM Use Cases: Experimental Findings and Detection Tool Design 

**Authors**: Richard M. Charles, James H. Curry, Richard B. Charles  

**Link**: [PDF](https://arxiv.org/pdf/2507.14207)  

**Abstract**: The integration of Large Language Models (LLMs) in K--12 education offers both transformative opportunities and emerging risks. This study explores how students may Trojanize prompts to elicit unsafe or unintended outputs from LLMs, bypassing established content moderation systems with safety guardrils. Through a systematic experiment involving simulated K--12 queries and multi-turn dialogues, we expose key vulnerabilities in GPT-3.5 and GPT-4. This paper presents our experimental design, detailed findings, and a prototype tool, TrojanPromptGuard (TPG), to automatically detect and mitigate Trojanized educational prompts. These insights aim to inform both AI safety researchers and educational technologists on the safe deployment of LLMs for educators. 

---
# A Comprehensive Benchmark for Electrocardiogram Time-Series 

**Authors**: Zhijiang Tang, Jiaxin Qi, Yuhua Zheng, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14206)  

**Abstract**: Electrocardiogram~(ECG), a key bioelectrical time-series signal, is crucial for assessing cardiac health and diagnosing various diseases. Given its time-series format, ECG data is often incorporated into pre-training datasets for large-scale time-series model training. However, existing studies often overlook its unique characteristics and specialized downstream applications, which differ significantly from other time-series data, leading to an incomplete understanding of its properties. In this paper, we present an in-depth investigation of ECG signals and establish a comprehensive benchmark, which includes (1) categorizing its downstream applications into four distinct evaluation tasks, (2) identifying limitations in traditional evaluation metrics for ECG analysis, and introducing a novel metric; (3) benchmarking state-of-the-art time-series models and proposing a new architecture. Extensive experiments demonstrate that our proposed benchmark is comprehensive and robust. The results validate the effectiveness of the proposed metric and model architecture, which establish a solid foundation for advancing research in ECG signal analysis. 

---
# LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models 

**Authors**: Dachuan Shi, Yonggan Fu, Xiangchi Yuan, Zhongzhi Yu, Haoran You, Sixu Li, Xin Dong, Jan Kautz, Pavlo Molchanov, Yingyan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14204)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have spurred interest in numerous applications requiring robust long-range capabilities, essential for processing extensive input contexts and continuously generating extended outputs. As sequence lengths increase, the number of Key-Value (KV) pairs in LLMs escalates, creating a significant efficiency bottleneck. In this paper, we propose a new KV cache optimization paradigm called LaCache, a training-free method for efficient and accurate generative inference of LLMs. LaCache enables LLMs to simultaneously address both of the critical challenges in long-range modeling: robust long-range capabilities and continuous generation without running out-of-memory (OOM). Specifically, LaCache integrates two key innovations: (1) a ladder-shaped KV cache pattern that stores KV pairs not only sequentially (left-to-right within each layer) but also across layers (from shallow to deep), providing an extended span for capturing long-range dependencies under a fixed storage budget, thereby boosting long-range capabilities; and (2) an iterative compaction mechanism that progressively compresses older caches, freeing up space for new tokens within a fixed cache size. This token distance-based dynamic compression enables more effective continuous generation under constrained cache budgets. Experiments across various tasks, benchmarks, and LLM models consistently validate LaCache's effectiveness in enhancing LLMs' long-range capabilities. Our code is available at this https URL. 

---
# PRM-Free Security Alignment of Large Models via Red Teaming and Adversarial Training 

**Authors**: Pengfei Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.14202)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse applications, yet they pose significant security risks that threaten their safe deployment in critical domains. Current security alignment methodologies predominantly rely on Process Reward Models (PRMs) to evaluate intermediate reasoning steps, introducing substantial computational overhead and scalability constraints. This paper presents a novel PRM-free security alignment framework that leverages automated red teaming and adversarial training to achieve robust security guarantees while maintaining computational efficiency. Our approach systematically identifies vulnerabilities through sophisticated attack strategies including genetic algorithm optimization, multi-agent simulation, and advanced prompt mutation techniques. The framework enhances model robustness via targeted adversarial training with curriculum learning and adaptive regularization mechanisms. Comprehensive experimental evaluation across five state-of-the-art LLMs demonstrates that our method achieves superior security alignment performance compared to PRM-based approaches while reducing computational costs by 61\%. The framework incorporates transparent reporting and continuous audit mechanisms that enable iterative security improvement and regulatory compliance. Our contributions advance the field of efficient LLM security alignment by democratizing access to robust security measures for resource-constrained organizations and providing a scalable foundation for addressing evolving adversarial threats. 

---
# ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation 

**Authors**: Yiran Wu, Mauricio Velazco, Andrew Zhao, Manuel Raúl Meléndez Luján, Srisuma Movva, Yogesh K Roy, Quang Nguyen, Roberto Rodriguez, Qingyun Wu, Michael Albada, Julia Kiseleva, Anand Mudgerikar  

**Link**: [PDF](https://arxiv.org/pdf/2507.14201)  

**Abstract**: We present ExCyTIn-Bench, the first benchmark to Evaluate an LLM agent x on the task of Cyber Threat Investigation through security questions derived from investigation graphs. Real-world security analysts must sift through a large number of heterogeneous alert signals and security logs, follow multi-hop chains of evidence, and compile an incident report. With the developments of LLMs, building LLM-based agents for automatic thread investigation is a promising direction. To assist the development and evaluation of LLM agents, we construct a dataset from a controlled Azure tenant that covers 8 simulated real-world multi-step attacks, 57 log tables from Microsoft Sentinel and related services, and 589 automatically generated questions. We leverage security logs extracted with expert-crafted detection logic to build threat investigation graphs, and then generate questions with LLMs using paired nodes on the graph, taking the start node as background context and the end node as answer. Anchoring each question to these explicit nodes and edges not only provides automatic, explainable ground truth answers but also makes the pipeline reusable and readily extensible to new logs. This also enables the automatic generation of procedural tasks with verifiable rewards, which can be naturally extended to training agents via reinforcement learning. Our comprehensive experiments with different models confirm the difficulty of the task: with the base setting, the average reward across all evaluated models is 0.249, and the best achieved is 0.368, leaving substantial headroom for future research. Code and data are coming soon! 

---
# Open-Source LLMs Collaboration Beats Closed-Source LLMs: A Scalable Multi-Agent System 

**Authors**: Shengji Tang, Jianjian Cao, Weihao Lin, Jiale Hong, Bo Zhang, Shuyue Hu, Lei Bai, Tao Chen, Wanli Ouyang, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.14200)  

**Abstract**: This paper aims to demonstrate the potential and strengths of open-source collectives. It leads to a promising question: Can we harness multiple open-source LLMs to match or even beat the closed-source LLMs? To answer this, we propose SMACS, a scalable multi-agent collaboration system (MACS) framework with high performance. Specifically, for continuous integration of new LLMs and generalization to diverse questions, we first propose a Retrieval-based Prior Selection (RPS), which assigns a proxy performance score to each LLM to select the Top-k LLMs at the instance level for any given question. Then, we propose an Exploration-Exploitation-Driven Posterior Enhancement (EPE), encouraging the generation of diverse responses through prior dropping and selecting the high-quality response via a hybrid posterior score. Experiments on eight mainstream benchmarks validate the effectiveness of our SMACS: by integrating fifteen open-source LLMs, SMACS outperforms leading closed-source LLMs in 2025, e.g., Claude-3.7-Sonnet (+12.73%), GPT-4.1(+5.36%) and GPT-o3-mini(+5.28%) across multiple tasks. Remarkably, it even exceeds the average of best results of different datasets from both open-source LLMs (+2.86%) and closed-source LLMs (+2.04%), pushing the upper bound of intelligence. Code will be released at this https URL. 

---
# Retention analysis of edited knowledge after fine-tuning 

**Authors**: Fufang Wen, Shichang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14198)  

**Abstract**: Large language models (LLMs) store vast amounts of knowledge, which often requires updates to correct factual errors, incorporate newly acquired information, or adapt model behavior. Model editing methods have emerged as efficient solutions for such updates, offering localized and precise knowledge modification at significantly lower computational cost than continual training. In parallel, LLMs are frequently fine-tuned for a wide range of downstream tasks. However, the effect of fine-tuning on previously edited knowledge remains poorly understood. In this work, we systematically investigate how different fine-tuning objectives interact with various model editing techniques. Our findings show that edited knowledge is substantially more susceptible to forgetting during fine-tuning than intrinsic knowledge acquired through pre-training. This analysis highlights a key limitation of current editing approaches and suggests that evaluating edit robustness under downstream fine-tuning is critical for their practical deployment. We further find that freezing layers associated with edited content can significantly improve knowledge retention, offering insight into how future editing methods might be made more robust. 

---
# Explainable Parallel CNN-LSTM Model for Differentiating Ventricular Tachycardia from Supraventricular Tachycardia with Aberrancy in 12-Lead ECGs 

**Authors**: Zahra Teimouri-Jervekani, Fahimeh Nasimi, Mohammadreza Yazdchi, Ghazal MogharehZadeh, Javad Tezerji, Farzan Niknejad Mazandarani, Maryam Mohebbi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14196)  

**Abstract**: Background and Objective: Differentiating wide complex tachycardia (WCT) is clinically critical yet challenging due to morphological similarities in electrocardiogram (ECG) signals between life-threatening ventricular tachycardia (VT) and supraventricular tachycardia with aberrancy (SVT-A). Misdiagnosis carries fatal risks. We propose a computationally efficient deep learning solution to improve diagnostic accuracy and provide model interpretability for clinical deployment.
Methods: A novel lightweight parallel deep architecture is introduced. Each pipeline processes individual ECG leads using two 1D-CNN blocks to extract local features. Feature maps are concatenated across leads, followed by LSTM layers to capture temporal dependencies. Final classification employs fully connected layers. Explainability is achieved via Shapley Additive Explanations (SHAP) for local/global interpretation. The model was evaluated on a 35-subject ECG database using standard performance metrics.
Results: The model achieved $95.63\%$ accuracy ($95\%$ CI: $93.07-98.19\%$), with sensitivity=$95.10\%$, specificity=$96.06\%$, and F1-score=$95.12\%$. It outperformed state-of-the-art methods in both accuracy and computational efficiency, requiring minimal CNN blocks per pipeline. SHAP analysis demonstrated clinically interpretable feature contributions.
Conclusions: Our end-to-end framework delivers high-precision WCT classification with minimal computational overhead. The integration of SHAP enhances clinical trust by elucidating decision logic, supporting rapid, informed diagnosis. This approach shows significant promise for real-world ECG analysis tools. 

---
# UWB Radar-based Heart Rate Monitoring: A Transfer Learning Approach 

**Authors**: Elzbieta Gruzewska, Pooja Rao, Sebastien Baur, Matthew Baugh, Mathias M.J. Bellaiche, Sharanya Srinivas, Octavio Ponce, Matthew Thompson, Pramod Rudrapatna, Michael A. Sanchez, Lawrence Z. Cai, Timothy JA Chico, Robert F. Storey, Emily Maz, Umesh Telang, Shravya Shetty, Mayank Daswani  

**Link**: [PDF](https://arxiv.org/pdf/2507.14195)  

**Abstract**: Radar technology presents untapped potential for continuous, contactless, and passive heart rate monitoring via consumer electronics like mobile phones. However the variety of available radar systems and lack of standardization means that a large new paired dataset collection is required for each radar system. This study demonstrates transfer learning between frequency-modulated continuous wave (FMCW) and impulse-radio ultra-wideband (IR-UWB) radar systems, both increasingly integrated into consumer devices. FMCW radar utilizes a continuous chirp, while IR-UWB radar employs short pulses. Our mm-wave FMCW radar operated at 60 GHz with a 5.5 GHz bandwidth (2.7 cm resolution, 3 receiving antennas [Rx]), and our IR-UWB radar at 8 GHz with a 500 MHz bandwidth (30 cm resolution, 2 Rx). Using a novel 2D+1D ResNet architecture we achieved a mean absolute error (MAE) of 0.85 bpm and a mean absolute percentage error (MAPE) of 1.42% for heart rate monitoring with FMCW radar (N=119 participants, an average of 8 hours per participant). This model maintained performance (under 5 MAE/10% MAPE) across various body positions and heart rate ranges, with a 98.9% recall. We then fine-tuned a variant of this model, trained on single-antenna and single-range bin FMCW data, using a small (N=376, avg 6 minutes per participant) IR-UWB dataset. This transfer learning approach yielded a model with MAE 4.1 bpm and MAPE 6.3% (97.5% recall), a 25% MAE reduction over the IR-UWB baseline. This demonstration of transfer learning between radar systems for heart rate monitoring has the potential to accelerate its introduction into existing consumer devices. 

---
# A Formal Model of the Economic Impacts of AI Openness Regulation 

**Authors**: Tori Qiu, Benjamin Laufer, Jon Kleinberg, Hoda Heidari  

**Link**: [PDF](https://arxiv.org/pdf/2507.14193)  

**Abstract**: Regulatory frameworks, such as the EU AI Act, encourage openness of general-purpose AI models by offering legal exemptions for "open-source" models. Despite this legislative attention on openness, the definition of open-source foundation models remains ambiguous. This paper models the strategic interactions among the creator of a general-purpose model (the generalist) and the entity that fine-tunes the general-purpose model to a specialized domain or task (the specialist), in response to regulatory requirements on model openness. We present a stylized model of the regulator's choice of an open-source definition to evaluate which AI openness standards will establish appropriate economic incentives for developers. Our results characterize market equilibria -- specifically, upstream model release decisions and downstream fine-tuning efforts -- under various openness regulations and present a range of effective regulatory penalties and open-source thresholds. Overall, we find the model's baseline performance determines when increasing the regulatory penalty vs. the open-source threshold will significantly alter the generalist's release strategy. Our model provides a theoretical foundation for AI governance decisions around openness and enables evaluation and refinement of practical open-source policies. 

---
# DeepWriter: A Fact-Grounded Multimodal Writing Assistant Based On Offline Knowledge Base 

**Authors**: Song Mao, Lejun Cheng, Pinlong Cai, Guohang Yan, Ding Wang, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14189)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various applications. However, their use as writing assistants in specialized domains like finance, medicine, and law is often hampered by a lack of deep domain-specific knowledge and a tendency to hallucinate. Existing solutions, such as Retrieval-Augmented Generation (RAG), can suffer from inconsistency across multiple retrieval steps, while online search-based methods often degrade quality due to unreliable web content. To address these challenges, we introduce DeepWriter, a customizable, multimodal, long-form writing assistant that operates on a curated, offline knowledge base. DeepWriter leverages a novel pipeline that involves task decomposition, outline generation, multimodal retrieval, and section-by-section composition with reflection. By deeply mining information from a structured corpus and incorporating both textual and visual elements, DeepWriter generates coherent, factually grounded, and professional-grade documents. We also propose a hierarchical knowledge representation to enhance retrieval efficiency and accuracy. Our experiments on financial report generation demonstrate that DeepWriter produces high-quality, verifiable articles that surpasses existing baselines in factual accuracy and generated content quality. 

---
# From Cell Towers to Satellites: A 2040 Blueprint for Urban-Grade Direct-to-Device Mobile Networks 

**Authors**: Sebastian Barros Elgueta  

**Link**: [PDF](https://arxiv.org/pdf/2507.14188)  

**Abstract**: In 2023, satellite and mobile networks crossed a historic threshold: standard smartphones, using unmodified 3GPP protocols, connected directly to low Earth orbit (LEO) satellites. This first wave of direct-to-device (D2D) demonstrations validated the physical feasibility of satellite-based mobile access. However, these systems remain fallback-grade--rural-only, bandwidth-limited, and fully dependent on Earth-based mobile cores for identity, session, and policy control. This paper asks a more ambitious question: Can a complete mobile network, including radio access, core functions, traffic routing, and content delivery, operate entirely from orbit? And can it deliver sustained, urban-grade service in the world's densest cities? We present the first end-to-end system architecture for a fully orbital telco, integrating electronically steered phased arrays with 1000-beam capacity, space-based deployment of 5G core functions (UPF, AMF), and inter-satellite laser mesh backhaul. We analyze spectral efficiency, beam capacity, and link budgets under dense urban conditions, accounting for path loss, Doppler, and multipath. Simulations show that rooftop and line-of-sight users can sustain 64-QAM throughput, while street-level access is feasible with relay or assisted beam modes. The paper outlines the remaining constraints, power, thermal dissipation, compute radiation hardening, and regulatory models, and demonstrates that these are engineering bottlenecks, not physical limits. Finally, we propose a staged 15-year roadmap from today's fallback D2D systems to autonomous orbital overlays delivering 50-100 Mbps to handhelds in megacities, with zero reliance on terrestrial infrastructure. 

---
# AI-Based Impedance Encoding-Decoding Method for Online Impedance Network Construction of Wind Farms 

**Authors**: Xiaojuan Zhang, Tianyu Jiang, Haoxiang Zong, Chen Zhang, Chendan Li, Marta Molinas  

**Link**: [PDF](https://arxiv.org/pdf/2507.14187)  

**Abstract**: The impedance network (IN) model is gaining popularity in the oscillation analysis of wind farms. However, the construction of such an IN model requires impedance curves of each wind turbine under their respective operating conditions, making its online application difficult due to the transmission of numerous high-density impedance curves. To address this issue, this paper proposes an AI-based impedance encoding-decoding method to facilitate the online construction of IN model. First, an impedance encoder is trained to compress impedance curves by setting the number of neurons much smaller than that of frequency points. Then, the compressed data of each turbine are uploaded to the wind farm and an impedance decoder is trained to reconstruct original impedance curves. At last, based on the nodal admittance matrix (NAM) method, the IN model of the wind farm can be obtained. The proposed method is validated via model training and real-time simulations, demonstrating that the encoded impedance vectors enable fast transmission and accurate reconstruction of the original impedance curves. 

---
# A Disentangled Representation Learning Framework for Low-altitude Network Coverage Prediction 

**Authors**: Xiaojie Li, Zhijie Cai, Nan Qi, Chao Dong, Guangxu Zhu, Haixia Ma, Qihui Wu, Shi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.14186)  

**Abstract**: The expansion of the low-altitude economy has underscored the significance of Low-Altitude Network Coverage (LANC) prediction for designing aerial corridors. While accurate LANC forecasting hinges on the antenna beam patterns of Base Stations (BSs), these patterns are typically proprietary and not readily accessible. Operational parameters of BSs, which inherently contain beam information, offer an opportunity for data-driven low-altitude coverage prediction. However, collecting extensive low-altitude road test data is cost-prohibitive, often yielding only sparse samples per BS. This scarcity results in two primary challenges: imbalanced feature sampling due to limited variability in high-dimensional operational parameters against the backdrop of substantial changes in low-dimensional sampling locations, and diminished generalizability stemming from insufficient data samples. To overcome these obstacles, we introduce a dual strategy comprising expert knowledge-based feature compression and disentangled representation learning. The former reduces feature space complexity by leveraging communications expertise, while the latter enhances model generalizability through the integration of propagation models and distinct subnetworks that capture and aggregate the semantic representations of latent features. Experimental evaluation confirms the efficacy of our framework, yielding a 7% reduction in error compared to the best baseline algorithm. Real-network validations further attest to its reliability, achieving practical prediction accuracy with MAE errors at the 5dB level. 

---
# NeuroHD-RA: Neural-distilled Hyperdimensional Model with Rhythm Alignment 

**Authors**: ZhengXiao He, Jinghao Wen, Huayu Li, Ao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14184)  

**Abstract**: We present a novel and interpretable framework for electrocardiogram (ECG)-based disease detection that combines hyperdimensional computing (HDC) with learnable neural encoding. Unlike conventional HDC approaches that rely on static, random projections, our method introduces a rhythm-aware and trainable encoding pipeline based on RR intervals, a physiological signal segmentation strategy that aligns with cardiac cycles. The core of our design is a neural-distilled HDC architecture, featuring a learnable RR-block encoder and a BinaryLinear hyperdimensional projection layer, optimized jointly with cross-entropy and proxy-based metric loss. This hybrid framework preserves the symbolic interpretability of HDC while enabling task-adaptive representation learning. Experiments on Apnea-ECG and PTB-XL demonstrate that our model significantly outperforms traditional HDC and classical ML baselines, achieving 73.09\% precision and an F1 score of 0.626 on Apnea-ECG, with comparable robustness on PTB-XL. Our framework offers an efficient and scalable solution for edge-compatible ECG classification, with strong potential for interpretable and personalized health monitoring. 

---
# From Bias to Behavior: Learning Bull-Bear Market Dynamics with Contrastive Modeling 

**Authors**: Xiaotong Luo, Shengda Zhuo, Min Chen, Lichun Li, Ruizhao Lu, Wenqi Fan, Shuqiang Huang, Yin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14182)  

**Abstract**: Financial markets exhibit highly dynamic and complex behaviors shaped by both historical price trajectories and exogenous narratives, such as news, policy interpretations, and social media sentiment. The heterogeneity in these data and the diverse insight of investors introduce biases that complicate the modeling of market dynamics. Unlike prior work, this paper explores the potential of bull and bear regimes in investor-driven market dynamics. Through empirical analysis on real-world financial datasets, we uncover a dynamic relationship between bias variation and behavioral adaptation, which enhances trend prediction under evolving market conditions. To model this mechanism, we propose the Bias to Behavior from Bull-Bear Dynamics model (B4), a unified framework that jointly embeds temporal price sequences and external contextual signals into a shared latent space where opposing bull and bear forces naturally emerge, forming the foundation for bias representation. Within this space, an inertial pairing module pairs temporally adjacent samples to preserve momentum, while the dual competition mechanism contrasts bullish and bearish embeddings to capture behavioral divergence. Together, these components allow B4 to model bias-driven asymmetry, behavioral inertia, and market heterogeneity. Experimental results on real-world financial datasets demonstrate that our model not only achieves superior performance in predicting market trends but also provides interpretable insights into the interplay of biases, investor behaviors, and market dynamics. 

---
# Semi-Supervised Federated Learning via Dual Contrastive Learning and Soft Labeling for Intelligent Fault Diagnosis 

**Authors**: Yajiao Dai, Jun Li, Zhen Mei, Yiyang Ni, Shi Jin, Zengxiang Li, Sheng Guo, Wei Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14181)  

**Abstract**: Intelligent fault diagnosis (IFD) plays a crucial role in ensuring the safe operation of industrial machinery and improving production efficiency. However, traditional supervised deep learning methods require a large amount of training data and labels, which are often located in different clients. Additionally, the cost of data labeling is high, making labels difficult to acquire. Meanwhile, differences in data distribution among clients may also hinder the model's performance. To tackle these challenges, this paper proposes a semi-supervised federated learning framework, SSFL-DCSL, which integrates dual contrastive loss and soft labeling to address data and label scarcity for distributed clients with few labeled samples while safeguarding user privacy. It enables representation learning using unlabeled data on the client side and facilitates joint learning among clients through prototypes, thereby achieving mutual knowledge sharing and preventing local model divergence. Specifically, first, a sample weighting function based on the Laplace distribution is designed to alleviate bias caused by low confidence in pseudo labels during the semi-supervised training process. Second, a dual contrastive loss is introduced to mitigate model divergence caused by different data distributions, comprising local contrastive loss and global contrastive loss. Third, local prototypes are aggregated on the server with weighted averaging and updated with momentum to share knowledge among clients. To evaluate the proposed SSFL-DCSL framework, experiments are conducted on two publicly available datasets and a dataset collected on motors from the factory. In the most challenging task, where only 10\% of the data are labeled, the proposed SSFL-DCSL can improve accuracy by 1.15% to 7.85% over state-of-the-art methods. 

---
# Digital Twin-Assisted Explainable AI for Robust Beam Prediction in mmWave MIMO Systems 

**Authors**: Nasir Khan, Asmaa Abdallah, Abdulkadir Celik, Ahmed M. Eltawil, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2507.14180)  

**Abstract**: In line with the AI-native 6G vision, explainability and robustness are crucial for building trust and ensuring reliable performance in millimeter-wave (mmWave) systems. Efficient beam alignment is essential for initial access, but deep learning (DL) solutions face challenges, including high data collection overhead, hardware constraints, lack of explainability, and susceptibility to adversarial attacks. This paper proposes a robust and explainable DL-based beam alignment engine (BAE) for mmWave multiple-input multiple output (MIMO) systems. The BAE uses received signal strength indicator (RSSI) measurements from wide beams to predict the best narrow beam, reducing the overhead of exhaustive beam sweeping. To overcome the challenge of real-world data collection, this work leverages a site-specific digital twin (DT) to generate synthetic channel data closely resembling real-world environments. A model refinement via transfer learning is proposed to fine-tune the pre-trained model residing in the DT with minimal real-world data, effectively bridging mismatches between the digital replica and real-world environments. To reduce beam training overhead and enhance transparency, the framework uses deep Shapley additive explanations (SHAP) to rank input features by importance, prioritizing key spatial directions and minimizing beam sweeping. It also incorporates the Deep k-nearest neighbors (DkNN) algorithm, providing a credibility metric for detecting out-of-distribution inputs and ensuring robust, transparent decision-making. Experimental results show that the proposed framework reduces real-world data needs by 70%, beam training overhead by 62%, and improves outlier detection robustness by up to 8.5x, achieving near-optimal spectral efficiency and transparent decision making compared to traditional softmax based DL models. 

---
# A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering 

**Authors**: Nobel Dhar, Bobin Deng, Md Romyull Islam, Xinyue Zhang, Kazi Fahim Ahmad Nasif, Kun Suo  

**Link**: [PDF](https://arxiv.org/pdf/2507.14179)  

**Abstract**: Large Language Models (LLMs) exhibit significant activation sparsity, where only a subset of neurons are active for a given input. Although this sparsity presents opportunities to reduce computational cost, efficiently utilizing it requires predicting activation patterns in a scalable manner. However, direct prediction at the neuron level is computationally expensive due to the vast number of neurons in modern LLMs. To enable efficient prediction and utilization of activation sparsity, we propose a clustering-based activation pattern compression framework. Instead of treating each neuron independently, we group similar activation patterns into a small set of representative clusters. Our method achieves up to 79.34% clustering precision, outperforming standard binary clustering approaches while maintaining minimal degradation in perplexity (PPL) scores. With a sufficiently large number of clusters, our approach attains a PPL score as low as 12.49, demonstrating its effectiveness in preserving model quality while reducing computational overhead. By predicting cluster assignments rather than individual neuron states, future models can efficiently infer activation patterns from pre-computed centroids. We detail the clustering algorithm, analyze its effectiveness in capturing meaningful activation structures, and demonstrate its potential to improve sparse computation efficiency. This clustering-based formulation serves as a foundation for future work on activation pattern prediction, paving the way for efficient inference in large-scale language models. 

---
# Feature Bank Enhancement for Distance-based Out-of-Distribution Detection 

**Authors**: Yuhang Liu, Yuefei Wu, Bin Shi, Bo Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.14178)  

**Abstract**: Out-of-distribution (OOD) detection is critical to ensuring the reliability of deep learning applications and has attracted significant attention in recent years. A rich body of literature has emerged to develop efficient score functions that assign high scores to in-distribution (ID) samples and low scores to OOD samples, thereby helping distinguish OOD samples. Among these methods, distance-based score functions are widely used because of their efficiency and ease of use. However, deep learning often leads to a biased distribution of data features, and extreme features are inevitable. These extreme features make the distance-based methods tend to assign too low scores to ID samples. This limits the OOD detection capabilities of such methods. To address this issue, we propose a simple yet effective method, Feature Bank Enhancement (FBE), that uses statistical characteristics from dataset to identify and constrain extreme features to the separation boundaries, therapy making the distance between samples inside and outside the distribution farther. We conducted experiments on large-scale ImageNet-1k and CIFAR-10 respectively, and the results show that our method achieves state-of-the-art performance on both benchmark. Additionally, theoretical analysis and supplementary experiments are conducted to provide more insights into our method. 

---
# Understanding Two-Layer Neural Networks with Smooth Activation Functions 

**Authors**: Changcun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14177)  

**Abstract**: This paper aims to understand the training solution, which is obtained by the back-propagation algorithm, of two-layer neural networks whose hidden layer is composed of the units with smooth activation functions, including the usual sigmoid type most commonly used before the advent of ReLUs. The mechanism contains four main principles: construction of Taylor series expansions, strict partial order of knots, smooth-spline implementation and smooth-continuity restriction. The universal approximation for arbitrary input dimensionality is proved and experimental verification is given, through which the mystery of ``black box'' of the solution space is largely revealed. The new proofs employed also enrich approximation theory. 

---
# Latent Space Data Fusion Outperforms Early Fusion in Multimodal Mental Health Digital Phenotyping Data 

**Authors**: Youcef Barkat, Dylan Hamitouche, Deven Parekh, Ivy Guo, David Benrimoh  

**Link**: [PDF](https://arxiv.org/pdf/2507.14175)  

**Abstract**: Background: Mental illnesses such as depression and anxiety require improved methods for early detection and personalized intervention. Traditional predictive models often rely on unimodal data or early fusion strategies that fail to capture the complex, multimodal nature of psychiatric data. Advanced integration techniques, such as intermediate (latent space) fusion, may offer better accuracy and clinical utility. Methods: Using data from the BRIGHTEN clinical trial, we evaluated intermediate (latent space) fusion for predicting daily depressive symptoms (PHQ-2 scores). We compared early fusion implemented with a Random Forest (RF) model and intermediate fusion implemented via a Combined Model (CM) using autoencoders and a neural network. The dataset included behavioral (smartphone-based), demographic, and clinical features. Experiments were conducted across multiple temporal splits and data stream combinations. Performance was evaluated using mean squared error (MSE) and coefficient of determination (R2). Results: The CM outperformed both RF and Linear Regression (LR) baselines across all setups, achieving lower MSE (0.4985 vs. 0.5305 with RF) and higher R2 (0.4695 vs. 0.4356). The RF model showed signs of overfitting, with a large gap between training and test performance, while the CM maintained consistent generalization. Performance was best when integrating all data modalities in the CM (in contradistinction to RF), underscoring the value of latent space fusion for capturing non-linear interactions in complex psychiatric datasets. Conclusion: Latent space fusion offers a robust alternative to traditional fusion methods for prediction with multimodal mental health data. Future work should explore model interpretability and individual-level prediction for clinical deployment. 

---
# Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI 

**Authors**: Julien Pourcel, Cédric Colas, Pierre-Yves Oudeyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.14172)  

**Abstract**: Many program synthesis tasks prove too challenging for even state-of-the-art language models to solve in single attempts. Search-based evolutionary methods offer a promising alternative by exploring solution spaces iteratively, but their effectiveness remain limited by the fixed capabilities of the underlying generative model.
We propose SOAR, a method that learns program synthesis by integrating language models into a self-improving evolutionary loop.
SOAR alternates between (1) an evolutionary search that uses an LLM to sample and refine candidate solutions, and (2) a hindsight learning phase that converts search attempts into valid problem-solution pairs used to fine-tune the LLM's sampling and refinement capabilities\, -- \,enabling increasingly effective search in subsequent iterations.
On the challenging ARC-AGI benchmark, SOAR achieves significant performance gains across model scales and iterations, leveraging positive transfer between the sampling and refinement finetuning tasks. These improvements carry over to test-time adaptation, enabling SOAR to solve 52\% of the public test set. Our code is open-sourced at: this https URL 

---
# IPPRO: Importance-based Pruning with PRojective Offset for Magnitude-indifferent Structural Pruning 

**Authors**: Jaeheun Jung, Jaehyuk Lee, Yeajin Lee, Donghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.14171)  

**Abstract**: With the growth of demand on neural network compression methods, the structured pruning methods including importance-based approach are actively studied. The magnitude importance and many correlated modern importance criteria often limit the capacity of pruning decision, since the filters with larger magnitudes are not likely to be pruned if the smaller one didn't, even if it is redundant. In this paper, we propose a novel pruning strategy to challenge this dominating effect of magnitude and provide fair chance to each filter to be pruned, by placing it on projective space. After that, we observe the gradient descent movement whether the filters move toward the origin or not, to measure how the filter is likely to be pruned. This measurement is used to construct PROscore, a novel importance score for IPPRO, a novel importance-based structured pruning with magnitude-indifference. Our evaluation results shows that the proposed importance criteria using the projective space achieves near-lossless pruning by reducing the performance drop in pruning, with promising performance after the finetuning. Our work debunks the ``size-matters'' myth in pruning and expands the frontier of importance-based pruning both theoretically and empirically. 

---
# Catalyst: a Novel Regularizer for Structured Pruning with Auxiliary Extension of Parameter Space 

**Authors**: Jaeheun Jung, Donghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.14170)  

**Abstract**: Structured pruning aims to reduce the size and computational cost of deep neural networks by removing entire filters or channels. The traditional regularizers such as L1 or Group Lasso and its variants lead to magnitude-biased pruning decisions, such that the filters with small magnitudes are likely to be pruned. Also, they often entail pruning results with almost zero margin around pruning decision boundary, such that tiny perturbation in a filter magnitude can flip the pruning decision. In this paper, we identify the precise algebraic condition under which pruning operations preserve model performance, and use the condition to construct a novel regularizer defined in an extended parameter space via auxiliary catalyst variables. The proposed Catalyst regularization ensures fair pruning chance for each filters with theoretically provable zero bias to their magnitude and robust pruning behavior achieved by wide-margin bifurcation of magnitudes between the preserved and the pruned filters. The theoretical properties naturally lead to real-world effectiveness, as shown by empirical validations of Catalyst Pruning algorithm. Pruning results on various datasets and models are superior to state-of-the-art filter pruning methods, and at the same time confirm the predicted robust and fair pruning characteristics of Catalyst pruning. 

---
# A Denoising VAE for Intracardiac Time Series in Ischemic Cardiomyopathy 

**Authors**: Samuel Ruipérez-Campillo, Alain Ryser, Thomas M. Sutter, Ruibin Feng, Prasanth Ganesan, Brototo Deb, Kelly A. Brennan, Maxime Pedron, Albert J. Rogers, Maarten Z.H. Kolk, Fleur V.Y. Tjong, Sanjiv M. Narayan, Julia E. Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2507.14164)  

**Abstract**: In the field of cardiac electrophysiology (EP), effectively reducing noise in intra-cardiac signals is crucial for the accurate diagnosis and treatment of arrhythmias and cardiomyopathies. However, traditional noise reduction techniques fall short in addressing the diverse noise patterns from various sources, often non-linear and non-stationary, present in these signals. This work introduces a Variational Autoencoder (VAE) model, aimed at improving the quality of intra-ventricular monophasic action potential (MAP) signal recordings. By constructing representations of clean signals from a dataset of 5706 time series from 42 patients diagnosed with ischemic cardiomyopathy, our approach demonstrates superior denoising performance when compared to conventional filtering methods commonly employed in clinical settings. We assess the effectiveness of our VAE model using various metrics, indicating its superior capability to denoise signals across different noise types, including time-varying non-linear noise frequently found in clinical settings. These results reveal that VAEs can eliminate diverse sources of noise in single beats, outperforming state-of-the-art denoising techniques and potentially improving treatment efficacy in cardiac EP. 

---
# All-atom inverse protein folding through discrete flow matching 

**Authors**: Kai Yi, Kiarash Jamali, Sjors H. W. Scheres  

**Link**: [PDF](https://arxiv.org/pdf/2507.14156)  

**Abstract**: The recent breakthrough of AlphaFold3 in modeling complex biomolecular interactions, including those between proteins and ligands, nucleotides, or metal ions, creates new opportunities for protein design. In so-called inverse protein folding, the objective is to find a sequence of amino acids that adopts a target protein structure. Many inverse folding methods struggle to predict sequences for complexes that contain non-protein components, and perform poorly with complexes that adopt multiple structural states. To address these challenges, we present ADFLIP (All-atom Discrete FLow matching Inverse Protein folding), a generative model based on discrete flow-matching for designing protein sequences conditioned on all-atom structural contexts. ADFLIP progressively incorporates predicted amino acid side chains as structural context during sequence generation and enables the design of dynamic protein complexes through ensemble sampling across multiple structural states. Furthermore, ADFLIP implements training-free classifier guidance sampling, which allows the incorporation of arbitrary pre-trained models to optimise the designed sequence for desired protein properties. We evaluated the performance of ADFLIP on protein complexes with small-molecule ligands, nucleotides, or metal ions, including dynamic complexes for which structure ensembles were determined by nuclear magnetic resonance (NMR). Our model achieves state-of-the-art performance in single-structure and multi-structure inverse folding tasks, demonstrating excellent potential for all-atom protein design. The code is available at this https URL. 

---
# Surface EMG Profiling in Parkinson's Disease: Advancing Severity Assessment with GCN-SVM 

**Authors**: Daniel Cieślak, Barbara Szyca, Weronika Bajko, Liwia Florkiewicz, Kinga Grzęda, Mariusz Kaczmarek, Helena Kamieniecka, Hubert Lis, Weronika Matwiejuk, Anna Prus, Michalina Razik, Inga Rozumowicz, Wiktoria Ziembakowska  

**Link**: [PDF](https://arxiv.org/pdf/2507.14153)  

**Abstract**: Parkinson's disease (PD) poses challenges in diagnosis and monitoring due to its progressive nature and complex symptoms. This study introduces a novel approach utilizing surface electromyography (sEMG) to objectively assess PD severity, focusing on the biceps brachii muscle. Initial analysis of sEMG data from five PD patients and five healthy controls revealed significant neuromuscular differences. A traditional Support Vector Machine (SVM) model achieved up to 83% accuracy, while enhancements with a Graph Convolutional Network-Support Vector Machine (GCN-SVM) model increased accuracy to 92%. Despite the preliminary nature of these results, the study outlines a detailed experimental methodology for future research with larger cohorts to validate these findings and integrate the approach into clinical practice. The proposed approach holds promise for advancing PD severity assessment and improving patient care in Parkinson's disease management. 

---
# Self-DANA: A Resource-Efficient Channel-Adaptive Self-Supervised Approach for ECG Foundation Models 

**Authors**: Giuliana Monachino, Nicolò La Porta, Beatrice Zanchi, Luigi Fiorillo, Alvise Dei Rossi, Georgiy Farina, Francesca Dalia Faraci  

**Link**: [PDF](https://arxiv.org/pdf/2507.14151)  

**Abstract**: Foundation Models (FMs) are large-scale machine learning models trained on extensive, diverse datasets that can be adapted to a wide range of downstream tasks with minimal fine-tuning. In the last two years, interest in FMs has also grown for applications in the cardiological field to analyze the electrocardiogram (ECG) signals. One of the key properties of FMs is their transferability to a wide range of downstream scenarios. With the spread of wearable and portable devices, keen interest in learning from reduced-channel configurations has arisen. However, the adaptation of ECG FMs to downstream scenarios with fewer available channels still has to be properly investigated. In this work, we propose Self-DANA, a novel, easy-to-integrate solution that makes self-supervised architectures adaptable to a reduced number of input channels, ensuring resource efficiency and high performance. We also introduce Random Lead Selection, a novel augmentation technique to pre-train models in a more robust and channel-agnostic way. Our experimental results on five reduced-channel configurations demonstrate that Self-DANA significantly enhances resource efficiency while reaching state-of-the-art performance. It requires up to 69.3% less peak CPU memory, 34.4% less peak GPU memory, about 17% less average epoch CPU time, and about 24% less average epoch GPU time. 

---
# DIVER-0 : A Fully Channel Equivariant EEG Foundation Model 

**Authors**: Danny Dongyeop Han, Ahhyun Lucy Lee, Taeyang Lee, Yonghyeon Gwon, Sebin Lee, Seongjin Lee, David Keetae Park, Shinjae Yoo, Jiook Cha, Chun Kee Chung  

**Link**: [PDF](https://arxiv.org/pdf/2507.14141)  

**Abstract**: Electroencephalography (EEG) is a non-invasive technique widely used in brain-computer interfaces and clinical applications, yet existing EEG foundation models face limitations in modeling spatio-temporal brain dynamics and lack channel permutation equivariance, preventing robust generalization across diverse electrode configurations. To address these challenges, we propose DIVER-0, a novel EEG foundation model that demonstrates how full spatio-temporal attention-rather than segregated spatial or temporal processing-achieves superior performance when properly designed with Rotary Position Embedding (RoPE) for temporal relationships and binary attention biases for channel differentiation. We also introduce Sliding Temporal Conditional Positional Encoding (STCPE), which improves upon existing conditional positional encoding approaches by maintaining both temporal translation equivariance and channel permutation equivariance, enabling robust adaptation to arbitrary electrode configurations unseen during pretraining. Experimental results demonstrate that DIVER-0 achieves competitive performance with only 10% of pretraining data while maintaining consistent results across all channel permutation conditions, validating its effectiveness for cross-dataset generalization and establishing key design principles for handling the inherent heterogeneity of neural recording setups. 

---
# Geophysics-informed neural network for model-based seismic inversion using surrogate point spread functions 

**Authors**: Marcus Saraiva, Ana Muller, Alexandre Maul  

**Link**: [PDF](https://arxiv.org/pdf/2507.14140)  

**Abstract**: Model-based seismic inversion is a key technique in reservoir characterization, but traditional methods face significant limitations, such as relying on 1D average stationary wavelets and assuming an unrealistic lateral resolution. To address these challenges, we propose a Geophysics-Informed Neural Network (GINN) that integrates deep learning with seismic modeling. This novel approach employs a Deep Convolutional Neural Network (DCNN) to simultaneously estimate Point Spread Functions (PSFs) and acoustic impedance (IP). PSFs are divided into zero-phase and residual components to ensure geophysical consistency and to capture fine details. We used synthetic data from the SEAM Phase I Earth Model to train the GINN for 100 epochs (approximately 20 minutes) using a 2D UNet architecture. The network's inputs include positional features and a low-frequency impedance (LF-IP) model. A self-supervised loss function combining Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM) was employed to ensure accurate results. The GINN demonstrated its ability to generate high-resolution IP and realistic PSFs, aligning with expected geological features. Unlike traditional 1D wavelets, the GINN produces PSFs with limited lateral resolution, reducing noise and improving accuracy. Future work will aim to refine the training process and validate the methodology with real seismic data. 

---
# Exposing and Mitigating Calibration Biases and Demographic Unfairness in MLLM Few-Shot In-Context Learning for Medical Image Classification 

**Authors**: Xing Shen, Justin Szeto, Mingyang Li, Hengguan Huang, Tal Arbel  

**Link**: [PDF](https://arxiv.org/pdf/2506.23298)  

**Abstract**: Multimodal large language models (MLLMs) have enormous potential to perform few-shot in-context learning in the context of medical image analysis. However, safe deployment of these models into real-world clinical practice requires an in-depth analysis of the accuracies of their predictions, and their associated calibration errors, particularly across different demographic subgroups. In this work, we present the first investigation into the calibration biases and demographic unfairness of MLLMs' predictions and confidence scores in few-shot in-context learning for medical image classification. We introduce CALIN, an inference-time calibration method designed to mitigate the associated biases. Specifically, CALIN estimates the amount of calibration needed, represented by calibration matrices, using a bi-level procedure: progressing from the population level to the subgroup level prior to inference. It then applies this estimation to calibrate the predicted confidence scores during inference. Experimental results on three medical imaging datasets: PAPILA for fundus image classification, HAM10000 for skin cancer classification, and MIMIC-CXR for chest X-ray classification demonstrate CALIN's effectiveness at ensuring fair confidence calibration in its prediction, while improving its overall prediction accuracies and exhibiting minimum fairness-utility trade-off. Our codebase can be found at this https URL. 

---
# JELAI: Integrating AI and Learning Analytics in Jupyter Notebooks 

**Authors**: Manuel Valle Torre, Thom van der Velden, Marcus Specht, Catharine Oertel  

**Link**: [PDF](https://arxiv.org/pdf/2505.17593)  

**Abstract**: Generative AI offers potential for educational support, but often lacks pedagogical grounding and awareness of the student's learning context. Furthermore, researching student interactions with these tools within authentic learning environments remains challenging. To address this, we present JELAI, an open-source platform architecture designed to integrate fine-grained Learning Analytics (LA) with Large Language Model (LLM)-based tutoring directly within a Jupyter Notebook environment. JELAI employs a modular, containerized design featuring JupyterLab extensions for telemetry and chat, alongside a central middleware handling LA processing and context-aware LLM prompt enrichment. This architecture enables the capture of integrated code interaction and chat data, facilitating real-time, context-sensitive AI scaffolding and research into student behaviour. We describe the system's design, implementation, and demonstrate its feasibility through system performance benchmarks and two proof-of-concept use cases illustrating its capabilities for logging multi-modal data, analysing help-seeking patterns, and supporting A/B testing of AI configurations. JELAI's primary contribution is its technical framework, providing a flexible tool for researchers and educators to develop, deploy, and study LA-informed AI tutoring within the widely used Jupyter ecosystem. 

---
# On the Effectiveness of Large Language Models in Writing Alloy Formulas 

**Authors**: Yang Hong, Shan Jiang, Yulei Fu, Sarfraz Khurshid  

**Link**: [PDF](https://arxiv.org/pdf/2502.15441)  

**Abstract**: Declarative specifications have a vital role to play in developing safe and dependable software systems. Writing specifications correctly, however, remains particularly challenging. This paper presents a controlled experiment on using large language models (LLMs) to write declarative formulas in the well-known language Alloy. Our use of LLMs is three-fold. One, we employ LLMs to write complete Alloy formulas from given natural language descriptions (in English). Two, we employ LLMs to create alternative but equivalent formulas in Alloy with respect to given Alloy formulas. Three, we employ LLMs to complete sketches of Alloy formulas and populate the holes in the sketches by synthesizing Alloy expressions and operators so that the completed formulas accurately represent the desired properties (that are given in natural language). We conduct the experimental evaluation using 11 well-studied subject specifications and employ two popular LLMs, namely ChatGPT and DeepSeek. The experimental results show that the LLMs generally perform well in synthesizing complete Alloy formulas from input properties given in natural language or in Alloy, and are able to enumerate multiple unique solutions. Moreover, the LLMs are also successful at completing given sketches of Alloy formulas with respect to natural language descriptions of desired properties (without requiring test cases). We believe LLMs offer a very exciting advance in our ability to write specifications, and can help make specifications take a pivotal role in software development and enhance our ability to build robust software. 

---
# Generating executable oracles to check conformance of client code to requirements of JDK Javadocs using LLMs 

**Authors**: Shan Jiang, Chenguang Zhu, Sarfraz Khurshid  

**Link**: [PDF](https://arxiv.org/pdf/2411.01789)  

**Abstract**: Software testing remains the most widely used methodology for validating quality of code. However, effectiveness of testing critically depends on the quality of test suites used. Test cases in a test suite consist of two fundamental parts: (1) input values for the code under test, and (2) correct checks for the outputs it produces. These checks are commonly written as assertions, and termed test oracles. The last couple of decades have seen much progress in automated test input generation, e.g., using fuzzing and symbolic execution. However, automating test oracles remains a relatively less explored problem area. Indeed, a test oracle by its nature requires knowledge of expected behavior, which may only be known to the developer and may not not exist in a formal language that supports automated reasoning.
Our focus in this paper is automation of test oracles for clients of widely used Java libraries, e.g., this http URL and this http URL packages. Our key insight is that Javadocs that provide a rich source of information can enable automated generation of test oracles. Javadocs of the core Java libraries are fairly detailed documents that contain natural language descriptions of not only how the libraries behave but also how the clients must (not) use them. We use large language models as an enabling technology to embody our insight into a framework for test oracle automation, and evaluate it experimentally. Our experiments demonstrate that LLMs can generate oracles for checking normal and exceptional behaviors from Javadocs, with 98.8% of these oracles being compilable and 96.4% accurately reflecting intended properties. Even for the few incorrect oracles, errors are minor and can be easily corrected with the help of additional comment information generated by the LLMs. 

---
