# Learning from Active Human Involvement through Proxy Value Propagation 

**Title (ZH)**: 通过代理价值传播进行主动人类参与的学习 

**Authors**: Zhenghao Peng, Wenjie Mo, Chenda Duan, Quanyi Li, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.03369)  

**Abstract**: Learning from active human involvement enables the human subject to actively intervene and demonstrate to the AI agent during training. The interaction and corrective feedback from human brings safety and AI alignment to the learning process. In this work, we propose a new reward-free active human involvement method called Proxy Value Propagation for policy optimization. Our key insight is that a proxy value function can be designed to express human intents, wherein state-action pairs in the human demonstration are labeled with high values, while those agents' actions that are intervened receive low values. Through the TD-learning framework, labeled values of demonstrated state-action pairs are further propagated to other unlabeled data generated from agents' exploration. The proxy value function thus induces a policy that faithfully emulates human behaviors. Human-in-the-loop experiments show the generality and efficiency of our method. With minimal modification to existing reinforcement learning algorithms, our method can learn to solve continuous and discrete control tasks with various human control devices, including the challenging task of driving in Grand Theft Auto V. Demo video and code are available at: this https URL 

**Abstract (ZH)**: 积极的人类参与使人类主题能够在训练过程中主动介入并展示给AI代理，从而在学习过程中带来安全性和AI对齐。在本研究中，我们提出了一种新的无奖励主动人类参与方法，称为代理值传播，用于策略优化。我们的关键洞察是，可以设计一种代理值函数来表达人类意图，在人类示范中的状态-动作对被标记为高值，而那些被干预的代理行为则被赋予低值。通过TD学习框架，这些标记值进一步传播到由代理探索生成的其他未标记数据中。代理值函数因此诱导出一个忠实模仿人类行为的策略。通过环回实验，我们的方法展示了其普遍性和高效性。仅对现有的强化学习算法进行少量修改，我们的方法就能够学会解决各种连续和离散控制任务，包括在《侠盗猎车手五》中执行驾驶等具有挑战性的任务。演示视频和代码可在以下链接获取：this https URL 

---
# PalimpChat: Declarative and Interactive AI analytics 

**Title (ZH)**: PalimpChat：声明式和交互式的AI分析 

**Authors**: Chunwei Liu, Gerardo Vitagliano, Brandon Rose, Matt Prinz, David Andrew Samson, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2502.03368)  

**Abstract**: Thanks to the advances in generative architectures and large language models, data scientists can now code pipelines of machine-learning operations to process large collections of unstructured data. Recent progress has seen the rise of declarative AI frameworks (e.g., Palimpzest, Lotus, and DocETL) to build optimized and increasingly complex pipelines, but these systems often remain accessible only to expert programmers. In this demonstration, we present PalimpChat, a chat-based interface to Palimpzest that bridges this gap by letting users create and run sophisticated AI pipelines through natural language alone. By integrating Archytas, a ReAct-based reasoning agent, and Palimpzest's suite of relational and LLM-based operators, PalimpChat provides a practical illustration of how a chat interface can make declarative AI frameworks truly accessible to non-experts.
Our demo system is publicly available online. At SIGMOD'25, participants can explore three real-world scenarios--scientific discovery, legal discovery, and real estate search--or apply PalimpChat to their own datasets. In this paper, we focus on how PalimpChat, supported by the Palimpzest optimizer, simplifies complex AI workflows such as extracting and analyzing biomedical data. 

**Abstract (ZH)**: 多亏了生成架构和大规模语言模型的进步，数据科学家现在可以编写机器学习操作的代码流水线，以处理大量的非结构化数据。最近的进展催生了一类声明式人工智能框架（例如Palimpzest、Lotus和DocETL），用于构建优化且日益复杂的流水线，但这些系统往往仍然只对专家程序员开放。在此演示中，我们介绍了PalimpChat，这是一种基于聊天界面的接口，通过这种方式，Palimpzest得以跨越这一障碍，让用户仅通过自然语言即可创建和运行复杂的AI流水线。通过集成Archytas（一种ReAct为基础的推理代理）和Palimpzest的关联性和基于语言模型的操作集，PalimpChat提供了一种实用的范例，说明了聊天界面如何使声明式人工智能框架真正对非专家用户开放。

我们的演示系统现已在网上公开。在SIGMOD'25会议中，参与者可以探索三个实际应用场景——科学发现、法律发现和房地产搜索，或者将PalimpChat应用到自己的数据集中。在本文中，我们重点介绍PalimpChat在Palimpzest优化器的支持下简化复杂AI工作流（如提取和分析生物医学数据）的方式。 

---
# SymAgent: A Neural-Symbolic Self-Learning Agent Framework for Complex Reasoning over Knowledge Graphs 

**Title (ZH)**: SymAgent：一种用于知识图谱复杂推理的神经符号自我学习代理框架 

**Authors**: Ben Liu, Jihai Zhang, Fangquan Lin, Cheng Yang, Min Peng, Wotao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.03283)  

**Abstract**: Recent advancements have highlighted that Large Language Models (LLMs) are prone to hallucinations when solving complex reasoning problems, leading to erroneous results. To tackle this issue, researchers incorporate Knowledge Graphs (KGs) to improve the reasoning ability of LLMs. However, existing methods face two limitations: 1) they typically assume that all answers to the questions are contained in KGs, neglecting the incompleteness issue of KGs, and 2) they treat the KG as a static repository and overlook the implicit logical reasoning structures inherent in KGs. In this paper, we introduce SymAgent, an innovative neural-symbolic agent framework that achieves collaborative augmentation between KGs and LLMs. We conceptualize KGs as dynamic environments and transform complex reasoning tasks into a multi-step interactive process, enabling KGs to participate deeply in the reasoning process. SymAgent consists of two modules: Agent-Planner and Agent-Executor. The Agent-Planner leverages LLM's inductive reasoning capability to extract symbolic rules from KGs, guiding efficient question decomposition. The Agent-Executor autonomously invokes predefined action tools to integrate information from KGs and external documents, addressing the issues of KG incompleteness. Furthermore, we design a self-learning framework comprising online exploration and offline iterative policy updating phases, enabling the agent to automatically synthesize reasoning trajectories and improve performance. Experimental results demonstrate that SymAgent with weak LLM backbones (i.e., 7B series) yields better or comparable performance compared to various strong baselines. Further analysis reveals that our agent can identify missing triples, facilitating automatic KG updates. 

**Abstract (ZH)**: 近年来的研究表明，大型语言模型（LLMs）在解决复杂推理问题时容易产生幻觉（hallucination），导致错误的结果。为了解决这一问题，研究人员通过引入知识图（KGs）来提高LLMs的推理能力。然而，现有方法存在两个局限性：1）它们通常假设所有答案都包含在KGs中，忽视了KGs的不完整性问题；2）它们将KG视为静态资源，并忽略了KG中固有的隐含逻辑推理结构。在此论文中，我们提出了SymAgent，这是一种创新的神经-符号代理框架，实现了KGs和LLMs之间的协作增强。我们将KGs视为动态环境，并将复杂的推理任务转化为多步交互过程，使KGs能够深度参与推理过程。SymAgent由两个模块组成：Agent-Planner和Agent-Executor。Agent-Planner利用LLMs的归纳推理能力从KGs中提取符号规则，指导有效的问题分解。Agent-Executor自主调用预定义的动作工具，整合KGs和外部文档中的信息，解决KG不完整的问题。此外，我们设计了一个自我学习框架，包括在线探索和离线迭代策略更新阶段，使代理能够自动综合推理轨迹并提高性能。实验结果表明，使用较弱的LLM底座（如7B系列）的SymAgent相较于各种强大的基线具有更好的或相当的性能。进一步分析表明，我们的代理能够识别缺失的三元组，从而促进自动更新KG。 

---
# FedMobileAgent: Training Mobile Agents Using Decentralized Self-Sourced Data from Diverse Users 

**Title (ZH)**: FedMobileAgent：使用多样化用户的数据进行分散式自我源数据训练的移动代理模型 

**Authors**: Wenhao Wang, Zijie Yu, William Liu, Rui Ye, Tian Jin, Siheng Chen, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02982)  

**Abstract**: The advancement of mobile agents has opened new opportunities for automating tasks on mobile devices. Training these agents requires large-scale high-quality data, which is costly using human labor. Given the vast number of mobile phone users worldwide, if automated data collection from them is feasible, the resulting data volume and the subsequently trained mobile agents could reach unprecedented levels. Nevertheless, two major challenges arise: (1) extracting high-level and low-level user instructions without involving human and (2) utilizing distributed data from diverse users while preserving privacy.
To tackle these challenges, we propose FedMobileAgent, a collaborative framework that trains mobile agents using self-sourced data from diverse users. Specifically, it includes two techniques. First, we propose Auto-Annotation, which enables the automatic collection of high-quality datasets during users' routine phone usage with minimal cost. Second, we introduce adapted aggregation to improve federated training of mobile agents on non-IID user data, by incorporating both episode- and step-level distributions. In distributed settings, FedMobileAgent achieves performance comparable to centralized human-annotated models at less than 0.02\% of the cost, highlighting its potential for real-world applications. 

**Abstract (ZH)**: 移动代理的发展为在移动设备上自动化任务提供了新的机会。训练这些代理需要大量高质量的数据，而使用人类劳动力获取这些数据代价高昂。鉴于全球有大量的移动电话用户，如果能够从这些用户处自动收集数据，则由此产生的数据量和之后训练的移动代理将达到前所未有的水平。然而，存在两个主要挑战：（1）无需涉及人类即提取高级和低级用户指令；（2）利用来自多样化用户的分散数据，同时保护隐私。

为应对这些挑战，我们提出了一种协作框架——FedMobileAgent，该框架利用来自多样化用户的自源数据训练移动代理。具体而言，它包括两种技术。首先，我们提出了Auto-Annotation，该技术能够通过用户在常规电话使用过程中自动收集高质量的数据集，并且成本低廉。其次，我们引入了适应性聚合方法，通过在非同质个体数据（non-IID）级别和时间步骤级别整合分布，改进了移动代理的联邦训练。在分布式环境中，FedMobileAgent的性能与集中式人类标注模型相当，但成本仅为后者的0.02%以下，这突显了其在实际应用中的潜力。 

---
# Planning with affordances: Integrating learned affordance models and symbolic planning 

**Title (ZH)**: 利用利用能力进行规划：结合学习到的能力模型与符号规划 

**Authors**: Rajesh Mangannavar  

**Link**: [PDF](https://arxiv.org/pdf/2502.02768)  

**Abstract**: Intelligent agents working in real-world environments must be able to learn about the environment and its capabilities which enable them to take actions to change to the state of the world to complete a complex multi-step task in a photorealistic environment. Learning about the environment is especially important to perform various multiple-step tasks without having to redefine an agent's action set for different tasks or environment settings. In our work, we augment an existing task and motion planning framework with learned affordance models of objects in the world to enable planning and executing multi-step tasks using learned models. Each task can be seen as changing the current state of the world to a given goal state. The affordance models provide us with what actions are possible and how to perform those actions in any given state. A symbolic planning algorithm uses this information and the starting and goal state to create a feasible plan to reach the desired goal state to complete a given task. We demonstrate our approach in a virtual 3D photorealistic environment, AI2-Thor, and evaluate it on real-world tasks. Our results show that our agent quickly learns how to interact with the environment and is well prepared to perform tasks such as "Moving an object out of the way to reach the desired location." 

**Abstract (ZH)**: 在实际环境中的智能代理必须能够学习环境及其能力，以便能够采取行动来改变世界的状态，从而在逼真的环境中完成复杂的多步骤任务。学习环境尤为重要，它使得代理能够在不重新定义行动集的情况下执行各种多步骤任务或适应不同的环境设置。在我们的工作中，我们扩展了一个现有的任务和运动规划框架，通过加入世界中对象的学习功能模型，以使用学习到的模型来进行多步骤任务的规划与执行。每个任务可以视为将当前世界状态转变为给定目标状态的过程。功能模型提供了在任何给定状态下可能执行哪些操作以及如何执行这些操作的信息。符号规划算法利用这些信息和初始状态及目标状态来创建可行计划，以达到所需的最终状态并完成给定的任务。我们通过在虚拟的3D逼真环境—AI2-Thor中展示这一方法，并在实际任务中进行评估。我们的结果表明，我们的代理能够快速学会如何与环境互动，并且能够很好地准备执行诸如“移动物体以达到所需位置”这类任务。 

---
# A Schema-Guided Reason-while-Retrieve framework for Reasoning on Scene Graphs with Large-Language-Models (LLMs) 

**Title (ZH)**: 基于结构引导的在检索中推理框架：使用大规模语言模型（LLMs）对场景图进行推理 

**Authors**: Yiye Chen, Harpreet Sawhney, Nicholas Gydé, Yanan Jian, Jack Saunders, Patricio Vela, Ben Lundell  

**Link**: [PDF](https://arxiv.org/pdf/2502.03450)  

**Abstract**: Scene graphs have emerged as a structured and serializable environment representation for grounded spatial reasoning with Large Language Models (LLMs). In this work, we propose SG-RwR, a Schema-Guided Retrieve-while-Reason framework for reasoning and planning with scene graphs. Our approach employs two cooperative, code-writing LLM agents: a (1) Reasoner for task planning and information queries generation, and a (2) Retriever for extracting corresponding graph information following the queries. Two agents collaborate iteratively, enabling sequential reasoning and adaptive attention to graph information. Unlike prior works, both agents are prompted only with the scene graph schema rather than the full graph data, which reduces the hallucination by limiting input tokens, and drives the Reasoner to generate reasoning trace this http URL the trace, the Retriever programmatically query the scene graph data based on the schema understanding, allowing dynamic and global attention on the graph that enhances alignment between reasoning and retrieval. Through experiments in multiple simulation environments, we show that our framework surpasses existing LLM-based approaches in numerical Q\&A and planning tasks, and can benefit from task-level few-shot examples, even in the absence of agent-level demonstrations. Project code will be released. 

**Abstract (ZH)**: 场景图已作为Large Language Models (LLMs) 进行有grounding的空间推理时的一种结构化和可序列化环境表示而崭露头角。本文中，我们提出了SG-RwR，这是一种基于Schema-Guided Retrieve-while-Reason框架，用于利用场景图进行推理和规划。我们的方法使用了两个协作的代码编写LLM代理：一个（1）推理器，用于任务规划和信息查询生成；另一个（2）检索器，根据查询提取相应的图信息。两个代理通过迭代协作，实现了顺序推理和对图信息的适应性关注。与以往工作不同，这两个代理仅被提示场景图的模式而非完整的图数据，这通过限制输入token来减少妄想，并促使推理器生成推理轨迹。基于轨迹，检索器可以根据模式理解程序化地查询场景图数据，从而在图上实现动态和全局关注，提高推理和检索之间的对齐。通过在多个模拟环境中进行实验，我们展示了本框架在数值问答和规划任务中优于现有的基于LLM的方法，并且可以在缺乏代理级示范的情况下从任务级的少量示例中受益。项目代码将开源。 

---
# Robust Autonomy Emerges from Self-Play 

**Title (ZH)**: 鲁棒自主性源自自我对弈 

**Authors**: Marco Cusumano-Towner, David Hafner, Alex Hertzberg, Brody Huval, Aleksei Petrenko, Eugene Vinitsky, Erik Wijmans, Taylor Killian, Stuart Bowers, Ozan Sener, Philipp Krähenbühl, Vladlen Koltun  

**Link**: [PDF](https://arxiv.org/pdf/2502.03349)  

**Abstract**: Self-play has powered breakthroughs in two-player and multi-player games. Here we show that self-play is a surprisingly effective strategy in another domain. We show that robust and naturalistic driving emerges entirely from self-play in simulation at unprecedented scale -- 1.6~billion~km of driving. This is enabled by Gigaflow, a batched simulator that can synthesize and train on 42 years of subjective driving experience per hour on a single 8-GPU node. The resulting policy achieves state-of-the-art performance on three independent autonomous driving benchmarks. The policy outperforms the prior state of the art when tested on recorded real-world scenarios, amidst human drivers, without ever seeing human data during training. The policy is realistic when assessed against human references and achieves unprecedented robustness, averaging 17.5 years of continuous driving between incidents in simulation. 

**Abstract (ZH)**: 自我博弈在双人游戏和多人游戏中取得了突破。在这里，我们展示了自我博弈在另一个领域中是一个出人意料有效的策略。我们证明了在前所未有的规模下（模拟驾驶里程达16亿公里），自我博弈在模拟中能够自发产生稳健且自然的驾驶行为。这一成果得益于Gigaflow这一批处理模拟器，它能够以每小时单个8-GPU节点生成和训练相当于42年主观驾驶经验的数据。最终产生的策略在三个独立的自动驾驶基准测试中达到了最先进的性能。该策略在测试中表现出色，超越了此前的最好成绩，且在实际世界场景中面对真实人类驾驶者时表现优异。在整个训练过程中从未使用过人类数据。该策略在人类参考标准下显得非常真实，并且实现了前所未有的鲁棒性，模拟中的平均无事故驾驶里程达到17.5年。 

---
# ReachAgent: Enhancing Mobile Agent via Page Reaching and Operation 

**Title (ZH)**: ReachAgent：通过页面到达和操作增强移动代理 

**Authors**: Qinzhuo Wu, Wei Liu, Jian Luan, Bin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02955)  

**Abstract**: Recently, mobile AI agents have gained increasing attention. Given a task, mobile AI agents can interact with mobile devices in multiple steps and finally form a GUI flow that solves the task. However, existing agents tend to focus on most task-relevant elements at each step, leading to local optimal solutions and ignoring the overall GUI flow. To address this issue, we constructed a training dataset called MobileReach, which breaks the task into page reaching and operation subtasks. Furthermore, we propose ReachAgent, a two-stage framework that focuses on improving its task-completion abilities. It utilizes the page reaching and page operation subtasks, along with reward-based preference GUI flows, to further enhance the agent. Experimental results show that ReachAgent significantly improves the IoU Acc and Text Acc by 7.12% and 7.69% on the step-level and 4.72% and 4.63% on the task-level compared to the SOTA agent. Our data and code will be released upon acceptance. 

**Abstract (ZH)**: 近年来，移动AI代理逐渐受到了广泛关注。给定一个任务，移动AI代理可以在多步骤中与移动设备相互作用，并最终形成一个解决该任务的GUI流程。然而，现有的代理往往在每个步骤中专注于与任务最相关的小元素，导致局部最优解，而忽略了整体的GUI流程。为了解决这一问题，我们构建了一个名为MobileReach的数据集，将任务拆分为页面到达和页面操作子任务。此外，我们提出了ReachAgent，这是一种两阶段框架，旨在提高其任务完成能力。该框架利用页面到达和页面操作子任务，以及基于奖励的偏好GUI流程，进一步提升代理性能。实验结果表明，与当前最先进的代理相比，ReachAgent在步骤层面显著提高了IoU Acc和Text Acc，分别提高了7.12%和7.69%，在任务层面分别提高了4.72%和4.63%。我们的数据和代码将在接收后公开。 

---
# Interactive Symbolic Regression through Offline Reinforcement Learning: A Co-Design Framework 

**Title (ZH)**: 通过离线强化学习实现的交互式符号回归：一种协同设计框架 

**Authors**: Yuan Tian, Wenqi Zhou, Michele Viscione, Hao Dong, David Kammer, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2502.02917)  

**Abstract**: Symbolic Regression (SR) holds great potential for uncovering underlying mathematical and physical relationships from observed data. However, the vast combinatorial space of possible expressions poses significant challenges for both online search methods and pre-trained transformer models. Additionally, current state-of-the-art approaches typically do not consider the integration of domain experts' prior knowledge and do not support iterative interactions with the model during the equation discovery process. To address these challenges, we propose the Symbolic Q-network (Sym-Q), an advanced interactive framework for large-scale symbolic regression. Unlike previous large-scale transformer-based SR approaches, Sym-Q leverages reinforcement learning without relying on a transformer-based decoder. This formulation allows the agent to learn through offline reinforcement learning using any type of tree encoder, enabling more efficient training and inference. Furthermore, we propose a co-design mechanism, where the reinforcement learning-based Sym-Q facilitates effective interaction with domain experts at any stage of the equation discovery process. Users can dynamically modify generated nodes of the expression, collaborating with the agent to tailor the mathematical expression to best fit the problem and align with the assumed physical laws, particularly when there is prior partial knowledge of the expected behavior. Our experiments demonstrate that the pre-trained Sym-Q surpasses existing SR algorithms on the challenging SSDNC benchmark. Moreover, we experimentally show on real-world cases that its performance can be further enhanced by the interactive co-design mechanism, with Sym-Q achieving greater performance gains than other state-of-the-art models. Our reproducible code is available at this https URL. 

**Abstract (ZH)**: 符号回归（SR）具有从观测数据中揭示潜在数学和物理关系的巨大潜力。然而，可能表达式的庞大组合空间为在线搜索方法和预训练的变换器模型带来了重大挑战。此外，当前最先进的方法通常不考虑领域专家的先验知识，并且不支持在方程发现过程中与模型的迭代交互。为应对这些挑战，我们提出了一种名为Symbolic Q网络（Sym-Q）的交互式框架，用于大规模符号回归。与以前的大规模变换器基符号回归方法不同，Sym-Q利用强化学习机制，而不依赖于变换器解码器。这种形式允许智能体通过使用任何类型的树编码器进行离线强化学习来学习，从而实现更高效的训练和推理。此外，我们提出了一种协同设计机制，其中基于强化学习的Sym-Q在方程发现过程中的任何阶段都促进了与领域专家的有效互动。用户可以动态修改表达式生成的节点，与智能体协作以调整数学表达式以最好地适应问题并符合假设的物理定律，特别是在有预期行为的部分先验知识时。我们的实验表明，预训练的Sym-Q在具有挑战性的SSDNC基准测试中超越了现有符号回归算法。此外，我们通过在真实世界案例上的实验展示，交互式的协同设计机制能够进一步提升其性能，Sym-Q的性能增幅优于其他最先进的模型。相关可复现代码可在以下链接获取：[此处替换为链接]。 

---
# Domain-Invariant Per-Frame Feature Extraction for Cross-Domain Imitation Learning with Visual Observations 

**Title (ZH)**: 跨域视觉观测模仿学习中的领域不变每帧特征提取 

**Authors**: Minung Kim, Kawon Lee, Jungmo Kim, Sungho Choi, Seungyul Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.02867)  

**Abstract**: Imitation learning (IL) enables agents to mimic expert behavior without reward signals but faces challenges in cross-domain scenarios with high-dimensional, noisy, and incomplete visual observations. To address this, we propose Domain-Invariant Per-Frame Feature Extraction for Imitation Learning (DIFF-IL), a novel IL method that extracts domain-invariant features from individual frames and adapts them into sequences to isolate and replicate expert behaviors. We also introduce a frame-wise time labeling technique to segment expert behaviors by timesteps and assign rewards aligned with temporal contexts, enhancing task performance. Experiments across diverse visual environments demonstrate the effectiveness of DIFF-IL in addressing complex visual tasks. 

**Abstract (ZH)**: 模拟学习（IL）使代理能够模仿专家行为而无需奖励信号，但在高维、噪声大且观察不完整的情况下，面对跨域场景时面临挑战。为了解决这一问题，我们提出了Domain-Invariant Per-Frame Feature Extraction for Imitation Learning (DIFF-IL)，这是一种新颖的模拟学习方法，可以从单帧中提取跨域不变特征，并将这些特征适应为序列，以隔离和复制专家行为。我们还引入了一种帧级时间标记技术，通过时间步长对专家行为进行分割，并按照时间上下文分配奖励，从而提升任务性能。跨多种视觉环境的实验表明，DIFF-IL 在解决复杂视觉任务方面具有有效性。 

---
# OceanChat: The Effect of Virtual Conversational AI Agents on Sustainable Attitude and Behavior Change 

**Title (ZH)**: OceanChat：虚拟对话式AI代理对可持续态度和行为改变的影响 

**Authors**: Pat Pataranutaporn, Alexander Doudkin, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2502.02863)  

**Abstract**: Marine ecosystems face unprecedented threats from climate change and plastic pollution, yet traditional environmental education often struggles to translate awareness into sustained behavioral change. This paper presents OceanChat, an interactive system leveraging large language models to create conversational AI agents represented as animated marine creatures -- specifically a beluga whale, a jellyfish, and a seahorse -- designed to promote environmental behavior (PEB) and foster awareness through personalized dialogue. Through a between-subjects experiment (N=900), we compared three conditions: (1) Static Scientific Information, providing conventional environmental education through text and images; (2) Static Character Narrative, featuring first-person storytelling from 3D-rendered marine creatures; and (3) Conversational Character Narrative, enabling real-time dialogue with AI-powered marine characters. Our analysis revealed that the Conversational Character Narrative condition significantly increased behavioral intentions and sustainable choice preferences compared to static approaches. The beluga whale character demonstrated consistently stronger emotional engagement across multiple measures, including perceived anthropomorphism and empathy. However, impacts on deeper measures like climate policy support and psychological distance were limited, highlighting the complexity of shifting entrenched beliefs. Our work extends research on sustainability interfaces facilitating PEB and offers design principles for creating emotionally resonant, context-aware AI characters. By balancing anthropomorphism with species authenticity, OceanChat demonstrates how interactive narratives can bridge the gap between environmental knowledge and real-world behavior change. 

**Abstract (ZH)**: 海洋生态系统正面临着来自气候变化和塑料污染的前所未有的威胁，而传统的环境教育往往难以将公众的意识转化为持久的行为改变。本文介绍了一种名为OceanChat的交互系统，该系统利用大规模语言模型创建了代表为动画海洋生物（具体为白鲸、水母和海马）的对话AI代理，旨在通过个性化对话促进环境保护行为（Environmental Behavior, EBP）和提升环境意识。通过一项包含900名参与者的配对实验，我们比较了三种条件：（1）静态科学信息，通过文字和图像提供传统的环境教育；（2）静态角色叙述，以3D渲染的海洋生物的第一人称讲故事；（3）对话角色叙述，使参与者能够与基于AI的海洋角色进行实时对话。分析结果显示，对话角色叙述条件显著提高了行为意图和可持续选择偏好，相比于静态方法。白鲸角色在多个衡量指标中持续表现出更强的情感参与，包括拟人化感知和同理心。然而，对更深层次的指标，如气候政策支持和心理距离的影响有限，这突显了转变根深蒂固信念的复杂性。我们的研究扩展了可持续性界面促进环境保护行为的研究，并提出了创建情感共鸣、情境感知的AI角色的设计原则。通过平衡拟人化与其物种的真实性，OceanChat证明了交互式叙事如何在环境知识与现实行为改变之间架起桥梁。 

---
# Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning 

**Title (ZH)**: 面向稳健多Agent强化学习的狼群对抗性攻击 

**Authors**: Sunwoo Lee, Jaebak Hwang, Yonghyeon Jo, Seungyul Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.02844)  

**Abstract**: Traditional robust methods in multi-agent reinforcement learning (MARL) often struggle against coordinated adversarial attacks in cooperative scenarios. To address this limitation, we propose the Wolfpack Adversarial Attack framework, inspired by wolf hunting strategies, which targets an initial agent and its assisting agents to disrupt cooperation. Additionally, we introduce the Wolfpack-Adversarial Learning for MARL (WALL) framework, which trains robust MARL policies to defend against the proposed Wolfpack attack by fostering system-wide collaboration. Experimental results underscore the devastating impact of the Wolfpack attack and the significant robustness improvements achieved by WALL. 

**Abstract (ZH)**: 在多智能体强化学习（MARL）中，传统的鲁棒方法通常难以应对协同场景中的协调性敌对攻击。为解决这一局限性，我们提出了一种受狼捕猎策略启发的Wolfpack敌对攻击框架，该框架旨在破坏初始智能体及其辅助智能体的协同合作。此外，我们还引入了Wolfpack-敌对学习框架（WALL），该框架通过促进系统范围内的合作来训练鲁棒的MARL策略，以防御提出的Wolfpack攻击。实验结果突显了Wolfpack攻击的破坏性影响及其通过WALL实现的重大鲁棒性改进。 

---
# Task-Aware Virtual Training: Enhancing Generalization in Meta-Reinforcement Learning for Out-of-Distribution Tasks 

**Title (ZH)**: 面向任务的虚拟训练：增强元强化学习中分布外任务的一般化能力 

**Authors**: Jeongmo Kim, Yisak Park, Minung Kim, Seungyul Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.02834)  

**Abstract**: Meta reinforcement learning aims to develop policies that generalize to unseen tasks sampled from a task distribution. While context-based meta-RL methods improve task representation using task latents, they often struggle with out-of-distribution (OOD) tasks. To address this, we propose Task-Aware Virtual Training (TAVT), a novel algorithm that accurately captures task characteristics for both training and OOD scenarios using metric-based representation learning. Our method successfully preserves task characteristics in virtual tasks and employs a state regularization technique to mitigate overestimation errors in state-varying environments. Numerical results demonstrate that TAVT significantly enhances generalization to OOD tasks across various MuJoCo and MetaWorld environments. 

**Abstract (ZH)**: 元强化学习旨在开发能够泛化到未见过的任务的一系列策略，这些任务是从任务分布中采样的。虽然基于上下文的元强化学习方法利用任务潜在变量改善了任务表示，但它们在处理分布外（OOD）任务时往往存在困难。为了解决这一问题，我们提出了任务感知虚拟训练（TAVT）方法，这是一种新颖的算法，通过基于度量的表示学习准确捕捉训练和OOD场景中的任务特征。我们的方法成功在虚拟任务中保持了任务特征，并采用了状态正则化技术以缓解状态变化环境中的过度估计误差。数值结果表明，TAVT在各种MuJoCo和MetaWorld环境中显著提高了对OOD任务的泛化能力。 

---
# Classroom Simulacra: Building Contextual Student Generative Agents in Online Education for Learning Behavioral Simulation 

**Title (ZH)**: 教室仿真：构建在线教育中用于学习行为模拟的学生生成代理节点 

**Authors**: Songlin Xu, Hao-Ning Wen, Hongyi Pan, Dallas Dominguez, Dongyin Hu, Xinyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02780)  

**Abstract**: Student simulation supports educators to improve teaching by interacting with virtual students. However, most existing approaches ignore the modulation effects of course materials because of two challenges: the lack of datasets with granularly annotated course materials, and the limitation of existing simulation models in processing extremely long textual data. To solve the challenges, we first run a 6-week education workshop from N = 60 students to collect fine-grained data using a custom built online education system, which logs students' learning behaviors as they interact with lecture materials over time. Second, we propose a transferable iterative reflection (TIR) module that augments both prompting-based and finetuning-based large language models (LLMs) for simulating learning behaviors. Our comprehensive experiments show that TIR enables the LLMs to perform more accurate student simulation than classical deep learning models, even with limited demonstration data. Our TIR approach better captures the granular dynamism of learning performance and inter-student correlations in classrooms, paving the way towards a ''digital twin'' for online education. 

**Abstract (ZH)**: 学生模拟支持教育工作者通过与虚拟学生互动来改进教学。然而，由于两个挑战的存在，大多数现有的方法忽视了课程材料的调节作用：缺乏细粒度标注的课程材料数据集，以及现有模拟模型在处理极其长的文本数据方面的局限性。为了克服这些挑战，我们首先从60名学生那里运行了一个为期6周的教育工作坊，使用自建的在线教育系统收集细粒度数据，该系统记录了学生在时间上与讲义材料互动时的学习行为。其次，我们提出了一种可迁移的迭代反思（TIR）模块，该模块增强了基于提示和基于微调的大规模语言模型（LLMs），用于模拟学习行为。我们的全面实验表明，TIR使LLMs在有限的示范数据下，比传统的深度学习模型更能够进行准确的学生模拟。我们的TIR方法更好地捕捉了课堂中学习表现和学生间相关性的细粒度动态性，为在线教育实现“数字孪生”创造了可能性。 

---
# PatchPilot: A Stable and Cost-Efficient Agentic Patching Framework 

**Title (ZH)**: PatchPilot：一种稳定且经济高效的代理补丁框架 

**Authors**: Hongwei Li, Yuheng Tang, Shiqi Wang, Wenbo Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.02747)  

**Abstract**: Recent research builds various patching agents that combine large language models (LLMs) with non-ML tools and achieve promising results on the state-of-the-art (SOTA) software patching benchmark, SWE-Bench. Based on how to determine the patching workflows, existing patching agents can be categorized as agent-based planning methods, which rely on LLMs for planning, and human-based planning methods, which follow a pre-defined workflow. At a high level, agent-based planning methods achieve high patching performance but with a high cost and limited stability. Human-based planning methods, on the other hand, are more stable and efficient but have key workflow limitations that compromise their patching performance. In this paper, we propose PatchPilot, an agentic patcher that strikes a balance between patching efficacy, stability, and cost-efficiency. PatchPilot proposes a novel human-based planning workflow with five components: reproduction, localization, generation, validation, and refinement (where refinement is unique to PatchPilot). We introduce novel and customized designs to each component to optimize their effectiveness and efficiency. Through extensive experiments on the SWE-Bench benchmarks, PatchPilot shows a superior performance than existing open-source methods while maintaining low cost (less than 1$ per instance) and ensuring higher stability. We also conduct a detailed ablation study to validate the key designs in each component. 

**Abstract (ZH)**: 近年来的研究开发了多种修补剂，将大型语言模型（LLMs）与非机器学习工具结合起来，并在SWE-Bench这一最新的软件修补基准测试中取得了令人瞩目的成果。根据确定修补流程的方法，现有的修补剂可以分为基于代理的规划方法和基于人工的规划方法。基于代理的规划方法依靠LLMs进行规划，能够在实现高修补性能的同时，但也伴随着高成本和有限的稳定性。相比之下，基于人工的规划方法更稳定且效率更高，但其固定的流程限制了修补性能。本文提出了一种名为PatchPilot的修补代理，旨在平衡修补效果、稳定性和成本效益。PatchPilot提出了一种新颖的基于人工的规划工作流程，包含了五个组成部分：复现、定位、生成、验证和优化（优化是PatchPilot特有的）。我们对每个组成部分进行了创新和定制化设计，以优化其效果和效率。通过在SWE-Bench基准测试上的广泛实验，PatchPilot在保持低成本（每实例低于1美元）和高稳定性的前提下，显示出优于现有开源方法的性能。此外，我们还进行了一项详细的消融研究，验证了每个组成部分中的关键设计。 

---
# Vision-Language Model Dialog Games for Self-Improvement 

**Title (ZH)**: 视觉-语言模型对话游戏以实现自我提升 

**Authors**: Ksenia Konyushkova, Christos Kaplanis, Serkan Cabi, Misha Denil  

**Link**: [PDF](https://arxiv.org/pdf/2502.02740)  

**Abstract**: The increasing demand for high-quality, diverse training data poses a significant bottleneck in advancing vision-language models (VLMs). This paper presents VLM Dialog Games, a novel and scalable self-improvement framework for VLMs. Our approach leverages self-play between two agents engaged in a goal-oriented play centered around image identification. By filtering for successful game interactions, we automatically curate a high-quality dataset of interleaved images and text. We demonstrate that fine-tuning on this synthetic data leads to performance gains on downstream tasks and generalises across datasets. Moreover, as the improvements in the model lead to better game play, this procedure can be applied iteratively. This work paves the way for self-improving VLMs, with potential applications in various real-world scenarios especially when the high-quality multimodal data is scarce. 

**Abstract (ZH)**: 不断提升对高品质、多样化训练数据的需求已成为视觉-语言模型（VLMs）发展的瓶颈之一。本文提出了一种新颖且可扩展的自我改进框架——VLM对话游戏（VLM Dialog Games），该框架通过两个参与目标导向图像识别游戏的智能体之间的自博弈来提升VLMs的能力。通过筛选成功的游戏交互，我们自动生成高质量的图文交错数据集。实验证明，基于这种合成数据的微调能够提高下游任务的性能，并且能够跨数据集泛化。此外，随着模型改进导致游戏表现的提升，这种过程可以迭代进行。本研究为自我改进的VLMs铺平了道路，尤其是在高质量多模态数据稀缺的情况下，这种框架具有广泛的实际应用场景。 

---
# MedRAX: Medical Reasoning Agent for Chest X-ray 

**Title (ZH)**: MedRAX：胸部X光诊断智能推理Agent 

**Authors**: Adibvafa Fallahpour, Jun Ma, Alif Munim, Hongwei Lyu, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02673)  

**Abstract**: Chest X-rays (CXRs) play an integral role in driving critical decisions in disease management and patient care. While recent innovations have led to specialized models for various CXR interpretation tasks, these solutions often operate in isolation, limiting their practical utility in clinical practice. We present MedRAX, the first versatile AI agent that seamlessly integrates state-of-the-art CXR analysis tools and multimodal large language models into a unified framework. MedRAX dynamically leverages these models to address complex medical queries without requiring additional training. To rigorously evaluate its capabilities, we introduce ChestAgentBench, a comprehensive benchmark containing 2,500 complex medical queries across 7 diverse categories. Our experiments demonstrate that MedRAX achieves state-of-the-art performance compared to both open-source and proprietary models, representing a significant step toward the practical deployment of automated CXR interpretation systems. Data and code have been publicly available at this https URL 

**Abstract (ZH)**: 胸部X光片（CXR）在诊疗决策和患者护理中发挥着关键作用。尽管近期的技术创新已经催生了各种专门用于CXR解释任务的模型，但这些解决方案往往独立存在，限制了其在临床实践中的实用价值。我们提出了MedRAX，这是首款无缝集成最新CXR分析工具和多模态大语言模型的通用人工智能代理。MedRAX能够动态利用这些模型以解决复杂的医疗问题，无需额外训练。为了系统地评估其能力，我们引入了包含2,500个复杂医疗问题的ChestAgentBench基准测试，涵盖7个不同的类别。我们的实验表明，MedRAX在此类基准测试中的性能达到了最新技术水平，标志着自动CXR解释系统实用部署的重要一步。数据和代码可在以下网址公开获取：[此处提供网址] 

---
# CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration 

**Title (ZH)**: CAMI：一种通过状态推断和主题探索支持动机访谈的咨询代理 

**Authors**: Yizhe Yang, Palakorn Achananuparp, Heyan Huang, Jing Jiang, Kit Phey Leng, Nicholas Gabriel Lim, Cameron Tan Shi Ern, Ee-peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.02807)  

**Abstract**: Conversational counselor agents have become essential tools for addressing the rising demand for scalable and accessible mental health support. This paper introduces CAMI, a novel automated counselor agent grounded in Motivational Interviewing (MI) -- a client-centered counseling approach designed to address ambivalence and facilitate behavior change. CAMI employs a novel STAR framework, consisting of client's state inference, motivation topic exploration, and response generation modules, leveraging large language models (LLMs). These components work together to evoke change talk, aligning with MI principles and improving counseling outcomes for clients from diverse backgrounds. We evaluate CAMI's performance through both automated and manual evaluations, utilizing simulated clients to assess MI skill competency, client's state inference accuracy, topic exploration proficiency, and overall counseling success. Results show that CAMI not only outperforms several state-of-the-art methods but also shows more realistic counselor-like behavior. Additionally, our ablation study underscores the critical roles of state inference and topic exploration in achieving this performance. 

**Abstract (ZH)**: 会话咨询代理已成为应对不断增长的可扩展和易获取心理健康支持需求的重要工具。本文介绍了CAMI，这是一种基于动机访谈（MI）的新型自动化咨询代理——动机访谈是一种以客户为中心的咨询方法，旨在解决客户的犹疑和促进行为改变。CAMI采用了一种创新的STAR框架，包括客户端状态推理、动机话题探索和响应生成模块，利用大规模语言模型（LLM）。这些组件共同作用以诱发改变对话，符合动机访谈的原则，并改善来自不同背景客户的咨询服务效果。我们通过自动评估和人工评估来评估CAMI的性能，利用模拟客户评估其动机访谈技能、客户状态推理准确性、话题探索能力以及整体咨询成效。研究结果显示，CAMI不仅优于多种最先进的方法，还表现出更接近人类咨询师的行为。此外，我们的消融研究进一步强调了状态推理和话题探索在实现这一性能中的关键作用。 

---
