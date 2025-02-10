# nvAgent: Automated Data Visualization from Natural Language via Collaborative Agent Workflow 

**Title (ZH)**: nvAgent: 通过协作代理工作流从自然语言自动进行数据可视化 

**Authors**: Geliang Ouyang, Jingyao Chen, Zhihe Nie, Yi Gui, Yao Wan, Hongyu Zhang, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05036)  

**Abstract**: Natural Language to Visualization (NL2Vis) seeks to convert natural-language descriptions into visual representations of given tables, empowering users to derive insights from large-scale data. Recent advancements in Large Language Models (LLMs) show promise in automating code generation to transform tabular data into accessible visualizations. However, they often struggle with complex queries that require reasoning across multiple tables. To address this limitation, we propose a collaborative agent workflow, termed nvAgent, for NL2Vis. Specifically, nvAgent comprises three agents: a processor agent for database processing and context filtering, a composer agent for planning visualization generation, and a validator agent for code translation and output verification. Comprehensive evaluations on the new VisEval benchmark demonstrate that nvAgent consistently surpasses state-of-the-art baselines, achieving a 7.88% improvement in single-table and a 9.23% improvement in multi-table scenarios. Qualitative analyses further highlight that nvAgent maintains nearly a 20% performance margin over previous models, underscoring its capacity to produce high-quality visual representations from complex, heterogeneous data sources. 

**Abstract (ZH)**: 自然语言到可视化（NL2Vis）旨在将自然语言描述转换为给定表格的可视化表示，赋予用户从大量数据中提取洞见的能力。近年来，大规模语言模型（LLMs）在自动化代码生成以将表格数据转换为易用的可视化方面展现出前景。然而，在处理需要在多个表格之间进行推理的复杂查询时，它们往往遇到困难。为解决这一限制，我们提出了一种协作代理工作流，称为nvAgent，用于NL2Vis。具体而言，nvAgent 包含三个代理：数据库处理和上下文过滤的处理器代理、规划可视化生成的编曲代理以及代码翻译和输出验证的验证代理。在新的VisEval基准上的全面评估表明，nvAgent 在单表和多表场景中分别实现了7.88%和9.23%的性能提升，超越了现有最先进的基线方法。进一步的定性分析表明，nvAgent 在性能上保持了约20%的优势，突显了其从复杂异构数据源生成高质量可视化表示的能力。 

---
# Multi-Agent Reinforcement Learning with Focal Diversity Optimization 

**Title (ZH)**: 多智能体强化学习中的焦点多样性优化 

**Authors**: Selim Furkan Tekin, Fatih Ilhan, Tiansheng Huang, Sihao Hu, Zachary Yahn, Ling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04492)  

**Abstract**: The advancement of Large Language Models (LLMs) and their finetuning strategies has triggered the renewed interests in multi-agent reinforcement learning. In this paper, we introduce a focal diversity-optimized multi-agent reinforcement learning approach, coined as MARL-Focal, with three unique characteristics. First, we develop an agent-fusion framework for encouraging multiple LLM based agents to collaborate in producing the final inference output for each LLM query. Second, we develop a focal-diversity optimized agent selection algorithm that can choose a small subset of the available agents based on how well they can complement one another to generate the query output. Finally, we design a conflict-resolution method to detect output inconsistency among multiple agents and produce our MARL-Focal output through reward-aware and policy-adaptive inference fusion. Extensive evaluations on five benchmarks show that MARL-Focal is cost-efficient and adversarial-robust. Our multi-agent fusion model achieves performance improvement of 5.51\% compared to the best individual LLM-agent and offers stronger robustness over the TruthfulQA benchmark. Code is available at this https URL 

**Abstract (ZH)**: 大语言模型（LLMs）的进步及其微调策略重新引发了对多智能体强化学习的兴趣。本文介绍了一种具有三个独特特征的焦点多样化优化多智能体强化学习方法，命名为MARL-Focal。首先，我们开发了一种智能体融合框架，以鼓励多种基于LLM的智能体合作生成每个LLM查询的最终推理输出。其次，我们开发了一种焦点多样化优化的智能体选择算法，可以根据它们如何相互补充生成查询输出来选择可用智能体的小子集。最后，我们设计了一种冲突解决方法来检测多个智能体之间的输出不一致性，并通过基于奖励感知和策略自适应的推理融合生成我们的MARL-Focal输出。在五个基准上的广泛评估表明，MARL-Focal具有成本效益且对抗鲁棒性强。我们的多智能体融合模型在TruthfulQA基准上的性能比最佳单个LLM智能体提高了5.51%，并在鲁棒性方面表现更优异。相关代码可在以下地址获得：[此处提供网址] 

---
# Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models 

**Title (ZH)**: 向前跃进之前向后一步：自我回溯以增强语言模型的推理能力 

**Authors**: Xiao-Wen Yang, Xuan-Yi Zhu, Wen-Da Wei, Ding-Chu Zhang, Jie-Jing Shao, Zhi Zhou, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.04404)  

**Abstract**: The integration of slow-thinking mechanisms into large language models (LLMs) offers a promising way toward achieving Level 2 AGI Reasoners, as exemplified by systems like OpenAI's o1. However, several significant challenges remain, including inefficient overthinking and an overreliance on auxiliary reward models. We point out that these limitations stem from LLMs' inability to internalize the search process, a key component of effective reasoning. A critical step toward addressing this issue is enabling LLMs to autonomously determine when and where to backtrack, a fundamental operation in traditional search algorithms. To this end, we propose a self-backtracking mechanism that equips LLMs with the ability to backtrack during both training and inference. This mechanism not only enhances reasoning ability but also efficiency by transforming slow-thinking processes into fast-thinking through self-improvement. Empirical evaluations demonstrate that our proposal significantly enhances the reasoning capabilities of LLMs, achieving a performance gain of over 40 percent compared to the optimal-path supervised fine-tuning method. We believe this study introduces a novel and promising pathway for developing more advanced and robust Reasoners. 

**Abstract (ZH)**: 将大语言模型（LLMs）中整合缓慢思考机制的方法为实现Level 2 AGI推理器提供了前景广阔的可能性，如OpenAI的o1系统所示。然而，仍然存在几个重大挑战，包括无效的过度思考和过分依赖辅助奖励模型。我们指出，这些限制源于LLMs无法内化搜索过程，这是有效推理的关键组成部分。解决这一问题的关键步骤之一是使LLMs能够自主确定何时及何处回溯，这是传统搜索算法中的一个基本操作。为此，我们提出了一种自我回溯机制，使LLMs在训练和推理过程中都能具备回溯的能力。该机制不仅增强了推理能力，还通过自我改进将缓慢思考过程转化为快速思考，从而提高效率。实验评估表明，我们的提案显著提升了LLMs的推理能力，与最优路径监督微调方法相比，性能提高了超过40%。我们认为，这项研究为开发更具高级和稳健的推理器开辟了一条新颖且有前景的道路。 

---
# Division-of-Thoughts: Harnessing Hybrid Language Model Synergy for Efficient On-Device Agents 

**Title (ZH)**: 思绪分割：利用混合语言模型协同效应以实现高效的本地代理 

**Authors**: Chenyang Shao, Xinyuan Hu, Yutang Lin, Fengli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04392)  

**Abstract**: The rapid expansion of web content has made on-device AI assistants indispensable for helping users manage the increasing complexity of online tasks. The emergent reasoning ability in large language models offer a promising path for next-generation on-device AI agents. However, deploying full-scale Large Language Models (LLMs) on resource-limited local devices is challenging. In this paper, we propose Division-of-Thoughts (DoT), a collaborative reasoning framework leveraging the synergy between locally deployed Smaller-scale Language Models (SLMs) and cloud-based LLMs. DoT leverages a Task Decomposer to elicit the inherent planning abilities in language models to decompose user queries into smaller sub-tasks, which allows hybrid language models to fully exploit their respective strengths. Besides, DoT employs a Task Scheduler to analyze the pair-wise dependency of sub-tasks and create a dependency graph, facilitating parallel reasoning of sub-tasks and the identification of key steps. To allocate the appropriate model based on the difficulty of sub-tasks, DoT leverages a Plug-and-Play Adapter, which is an additional task head attached to the SLM that does not alter the SLM's parameters. To boost adapter's task allocation capability, we propose a self-reinforced training method that relies solely on task execution feedback. Extensive experiments on various benchmarks demonstrate that our DoT significantly reduces LLM costs while maintaining competitive reasoning accuracy. Specifically, DoT reduces the average reasoning time and API costs by 66.12% and 83.57%, while achieving comparable reasoning accuracy with the best baseline methods. 

**Abstract (ZH)**: 互联网内容的迅速扩展使得设备端人工智能助手成为帮助用户应对日益复杂在线任务的不可或缺工具。大型语言模型（LLMs）涌现的推理能力为新一代设备端AI代理的发展提供了有希望的道路。然而，在资源受限的本地设备上部署大规模的语言模型极具挑战性。本文提出了一种名为Thoughts Division（DoT）的合作推理框架，该框架利用本地部署的小规模语言模型（SLMs）和基于云的大规模语言模型之间的协同作用。DoT利用任务分解器激活语言模型内部的规划能力，将用户查询分解为更小的子任务，从而使混合语言模型能够充分利用各自的优点。此外，DoT使用任务调度器分析子任务之间的依赖关系并构建依赖图，便于并行推理子任务并识别关键步骤。为了根据子任务的难度分配合适的模型，DoT利用了一个插拔式适配器，这是一种附加在SLM上的任务头，不会改变SLM的参数。为了增强适配器的任务分配能力，我们提出了一种依靠任务执行反馈的自强化训练方法。在各种基准上的广泛实验表明，我们的DoT在显著降低LLM成本的同时，保持了竞争力的推理准确性。具体而言，DoT将平均推理时间和API成本降低了66.12%和83.57%，同时达到了与最佳基线方法相当的推理准确性。 

---
# Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives 

**Title (ZH)**: 位置：扩展大规模语言模型代理需要使用大规模语言模型原语进行极限分析 

**Authors**: Elliot Meyerson, Xin Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04358)  

**Abstract**: Decomposing hard problems into subproblems often makes them easier and more efficient to solve. With large language models (LLMs) crossing critical reliability thresholds for a growing slate of capabilities, there is an increasing effort to decompose systems into sets of LLM-based agents, each of whom can be delegated sub-tasks. However, this decomposition (even when automated) is often intuitive, e.g., based on how a human might assign roles to members of a human team. How close are these role decompositions to optimal? This position paper argues that asymptotic analysis with LLM primitives is needed to reason about the efficiency of such decomposed systems, and that insights from such analysis will unlock opportunities for scaling them. By treating the LLM forward pass as the atomic unit of computational cost, one can separate out the (often opaque) inner workings of a particular LLM from the inherent efficiency of how a set of LLMs are orchestrated to solve hard problems. In other words, if we want to scale the deployment of LLMs to the limit, instead of anthropomorphizing LLMs, asymptotic analysis with LLM primitives should be used to reason about and develop more powerful decompositions of large problems into LLM agents. 

**Abstract (ZH)**: 将复杂问题分解为子问题通常可以使问题更容易且更高效地求解。随着大型语言模型（LLMs）在越来越多的能力方面达到关键可靠性的阈值，人们正逐渐努力将系统分解为一组基于LLM的代理，每个代理可以被分配子任务。然而，即使这种分解是自动化的，也往往是直观的，比如基于人类如何为团队成员分配角色的方式。这些角色分配与最优解有多接近？本文的观点是，需要使用LLM的基本原理来进行渐近分析，以理性考虑此类分解系統的效率，并且通过此类分析得到的见解将为扩展此类系统提供机会。通过将LLM前向传递视为计算成本的最小单位，可以将特定LLM的内部工作与其他LLM如何协同合作解决复杂问题的固有效率区分开来。换句话说，如果我们想将LLM的部署扩大到极限，而不是赋予LLM人性化的属性，就应该使用LLM的基本原理进行渐近分析，以理性考虑和开发将大型问题分解为LLM代理的更强大的方法。 

---
# Dynamic benchmarking framework for LLM-based conversational data capture 

**Title (ZH)**: 基于大型语言模型的对话数据捕获动态基准框架 

**Authors**: Pietro Alessandro Aluffi, Patrick Zietkiewicz, Marya Bazzi, Matt Arderne, Vladimirs Murevics  

**Link**: [PDF](https://arxiv.org/pdf/2502.04349)  

**Abstract**: The rapid evolution of large language models (LLMs) has transformed conversational agents, enabling complex human-machine interactions. However, evaluation frameworks often focus on single tasks, failing to capture the dynamic nature of multi-turn dialogues. This paper introduces a dynamic benchmarking framework to assess LLM-based conversational agents through interactions with synthetic users. The framework integrates generative agent simulation to evaluate performance on key dimensions: information extraction, context awareness, and adaptive engagement. By simulating various aspects of user behavior, our work provides a scalable, automated, and flexible benchmarking approach. Experimental evaluation - within a loan application use case - demonstrates the framework's effectiveness under one-shot and few-shot extraction conditions. Results show that adaptive strategies improve data extraction accuracy, especially when handling ambiguous responses. Future work will extend its applicability to broader domains and incorporate additional metrics (e.g., conversational coherence, user engagement). This study contributes a structured, scalable approach to evaluating LLM-based conversational agents, facilitating real-world deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进化已经改变了对话代理，使其能够实现复杂的机器-人类交互。然而，现有的评估框架常常局限于单一任务，未能捕捉多轮对话中的动态特性。本文提出了一种动态基准测试框架，该框架通过与合成用户交互来评估基于LLM的对话代理。该框架整合了生成性代理模拟，以评估关键维度上的表现：信息抽取、上下文意识以及适应性参与。通过模拟用户行为的各个方面，我们的研究提供了一种可扩展、自动化且灵活的基准测试方法。在贷款申请用例中的实验评估表明，在一次性和少量样本抽取条件下，该框架的有效性。结果表明，适应性策略能够提高数据抽取准确性，特别是在处理含糊不清的响应时表现尤为明显。未来的工作还将将该框架的适用范围扩展到更广泛的领域，并纳入更多评估指标（例如，对话连贯性、用户参与度）。本研究提供了一种结构化且可扩展的方法来评估基于LLM的对话代理，从而促进其实用部署。 

---
# Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research 

**Title (ZH)**: 代理推理：配备工具进行深度研究的大型语言模型推理 

**Authors**: Junde Wu, Jiayuan Zhu, Yuyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04644)  

**Abstract**: We introduce Agentic Reasoning, a framework that enhances large language model (LLM) reasoning by integrating external tool-using agents. Unlike conventional LLM-based reasoning approaches, which rely solely on internal inference, Agentic Reasoning dynamically engages web search, code execution, and structured reasoning-context memory to solve complex problems requiring deep research and multi-step logical deduction. Our framework introduces the Mind Map agent, which constructs a structured knowledge graph to track logical relationships, improving deductive reasoning. Additionally, the integration of web-search and coding agents enables real-time retrieval and computational analysis, enhancing reasoning accuracy and decision-making. Evaluations on PhD-level scientific reasoning (GPQA) and domain-specific deep research tasks demonstrate that our approach significantly outperforms existing models, including leading retrieval-augmented generation (RAG) systems and closed-source LLMs. Moreover, our results indicate that agentic reasoning improves expert-level knowledge synthesis, test-time scalability, and structured problem-solving. The code is at: this https URL. 

**Abstract (ZH)**: 我们介绍了意愿性推理（Agentic Reasoning）框架，该框架通过整合外部工具使用代理来增强大型语言模型（LLM）的推理能力。与依赖内部推理的常规LLM推理方法不同，意愿性推理能够动态地利用网络搜索、代码执行和结构化推理上下文记忆，以解决需要深入研究和多步逻辑推理的复杂问题。我们的框架引入了思维导图代理（Mind Map agent），该代理构建了一个结构化的知识图谱，以跟踪逻辑关系，从而提高演绎推理能力。此外，网络搜索和编码代理的集成能够实现实时检索和计算分析，从而提高推理准确性和决策质量。在博士生级别科学推理（GPQA）和特定领域的深度研究任务评估中，我们的方法显著优于现有模型，包括领先的检索增强生成（RAG）系统和闭源的LLM。此外，我们的结果表明，意愿性推理提高了专家级知识综合、测试时的可扩展性和结构化问题解决能力。代码位于：![](this https URL) 

---
# Self-Regulation and Requesting Interventions 

**Title (ZH)**: 自我调节与请求干预 

**Authors**: So Yeon Min, Yue Wu, Jimin Sun, Max Kaufmann, Fahim Tajwar, Yonatan Bisk, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2502.04576)  

**Abstract**: Human intelligence involves metacognitive abilities like self-regulation, recognizing limitations, and seeking assistance only when needed. While LLM Agents excel in many domains, they often lack this awareness. Overconfident agents risk catastrophic failures, while those that seek help excessively hinder efficiency. A key challenge is enabling agents with a limited intervention budget $C$ is to decide when to request assistance. In this paper, we propose an offline framework that trains a "helper" policy to request interventions, such as more powerful models or test-time compute, by combining LLM-based process reward models (PRMs) with tabular reinforcement learning. Using state transitions collected offline, we score optimal intervention timing with PRMs and train the helper model on these labeled trajectories. This offline approach significantly reduces costly intervention calls during training. Furthermore, the integration of PRMs with tabular RL enhances robustness to off-policy data while avoiding the inefficiencies of deep RL. We empirically find that our method delivers optimal helper behavior. 

**Abstract (ZH)**: 人类智能涉及元认知能力，如自我调节、认识到自身局限性以及仅在必要时寻求帮助。虽然大语言模型（LLM）代理在许多领域表现出色，但它们往往缺乏这种自我意识。自信过高的代理可能会导致灾难性故障，而过于频繁寻求帮助的代理会妨碍效率。一个关键挑战是在有限的干预预算 $C$ 下，使代理能够决定何时请求帮助。在本文中，我们提出了一种离线框架，该框架通过结合基于LLM的过程奖励模型（PRMs）和表格强化学习来训练一个“帮助者”策略，以请求更强大的模型或测试时的计算资源。利用离线收集的状态转换，我们使用PRMs评估最佳干预时机，并在这些有标签的轨迹上训练帮助者模型。这种方法显著降低了训练期间昂贵的干预呼叫成本。此外，将PRMs与表格RL结合使用增强了对离策数据的鲁棒性，同时避免了深度RL带来的低效率。我们实验证明，该方法能提供最优的帮助者行为。 

---
# SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning 

**Title (ZH)**: SiriuS：基于强化推理的自我提升多智能体系统 

**Authors**: Wanjia Zhao, Mert Yuksekgonul, Shirley Wu, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.04780)  

**Abstract**: Multi-agent AI systems powered by large language models (LLMs) are increasingly applied to solve complex tasks. However, these systems often rely on fragile, manually designed prompts and heuristics, making optimization difficult. A key challenge in optimizing multi-agent systems is acquiring suitable training data for specialized agents. We introduce SiriuS, a self-improving, reasoning-driven optimization framework for multi-agent systems. Central to our approach is the construction of an experience library: a repository of high-quality reasoning trajectories. The library is built by retaining reasoning steps that lead to successful outcomes, providing a robust training set for optimizing multi-agent system. Additionally, we introduce a library augmentation procedure that refines unsuccessful trajectories, further enriching the library. SiriuS boosts performance by 2.86\% to 21.88\% on reasoning and biomedical QA and enhances agent negotiation in competitive settings. Our results show that SiriuS enhances multi-agent performance while generating reusable data for self-correction and self-play enhancement in the future. 

**Abstract (ZH)**: 由大语言模型（LLMs）驱动的多智能体AI系统在解决复杂任务方面应用日益增多。然而，这些系统通常依赖于脆弱且手工设计的提示和启发式方法，使得优化工作变得困难。多智能体系统优化的关键挑战之一是为专门化的智能体获取合适的训练数据。为了解决这一问题，我们提出了SiriuS，这是一种自改进、推理驱动的多智能体系统优化框架。我们方法的核心在于构建一个经验库：一个高质量推理轨迹的存储库。经验库通过保留导致成功结果的推理步骤来构建，为优化多智能体系统提供了稳健的训练集。此外，我们还引入了一种库扩充程序，该程序通过细化不成功的轨迹进一步丰富了经验库。SiriuS在推理和生物医学问答任务上增强了2.86%到21.88%的性能，并在竞争环境中改善了智能体之间的谈判能力。我们的研究结果表明，SiriuS不仅增强了多智能体系统的表现，还生成了可重用的数据，用于未来的自我纠正和自我博弈强化。 

---
# Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization 

**Title (ZH)**: 使用迭代潜在空间策略优化学习狼人杀中的战略语言代理 

**Authors**: Zelai Xu, Wanjun Gu, Chao Yu, Yi Wu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04686)  

**Abstract**: Large language model (LLM)-based agents have recently shown impressive progress in a variety of domains, including open-ended conversation and multi-step decision-making. However, applying these agents to social deduction games such as Werewolf, which requires both strategic decision-making and free-form language interaction, remains non-trivial. Traditional methods based on Counterfactual Regret Minimization (CFR) or reinforcement learning (RL) typically depend on a predefined action space, making them unsuitable for language games with unconstrained text action space. Meanwhile, pure LLM-based agents often suffer from intrinsic biases and require prohibitively large datasets for fine-tuning. We propose Latent Space Policy Optimization (LSPO), an iterative framework that addresses these challenges by first mapping free-form text to a discrete latent space, where methods like CFR and RL can learn strategic policy more effectively. We then translate the learned policy back into natural language dialogues, which are used to fine-tune an LLM via Direct Preference Optimization (DPO). By iteratively alternating between these stages, our LSPO agent progressively enhances both strategic reasoning and language communication. Experiment results on the Werewolf game show that our method improves the agent's performance in each iteration and outperforms existing Werewolf agents, underscoring its promise for free-form language decision-making. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的智能体在多个领域中已展现出了显著的进步，包括开放式对话和多步决策。然而，将这些智能体应用于像狼人杀（Werewolf）这样的社会推理游戏仍然具有挑战性，因为这类游戏要求既进行战略决策又进行自然语言交互。传统的基于Counterfactual Regret Minimization（CFR）或强化学习（RL）的方法通常依赖于预先定义的动作空间，这使得它们不适合语言游戏中不受限制的文本动作空间。同时，纯粹基于LLM的智能体往往存在固有的偏差，需要巨大规模的数据集进行微调。为此，我们提出了一种潜空间策略优化（LSPO）框架，该框架通过首先将自然语言文本映射到一个离散的潜空间，在该空间中，CFR和RL方法可以更有效地学习策略。然后将学到的策略转换回自然语言对话，并通过直接偏好优化（DPO）微调大规模语言模型（LLM）。通过迭代交替这些阶段，我们的LSPO智能体逐步增强其战略推理和语言交流能力。在狼人杀游戏中进行的实验结果表明，我们的方法在每次迭代中都能提高智能体的表现，并且优于现有的狼人杀智能体，这表明了其在自由形式语言决策中的潜力。 

---
# Agency Is Frame-Dependent 

**Title (ZH)**: 代理依赖于框架 

**Authors**: David Abel, André Barreto, Michael Bowling, Will Dabney, Shi Dong, Steven Hansen, Anna Harutyunyan, Khimya Khetarpal, Clare Lyle, Razvan Pascanu, Georgios Piliouras, Doina Precup, Jonathan Richens, Mark Rowland, Tom Schaul, Satinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2502.04403)  

**Abstract**: Agency is a system's capacity to steer outcomes toward a goal, and is a central topic of study across biology, philosophy, cognitive science, and artificial intelligence. Determining if a system exhibits agency is a notoriously difficult question: Dennett (1989), for instance, highlights the puzzle of determining which principles can decide whether a rock, a thermostat, or a robot each possess agency. We here address this puzzle from the viewpoint of reinforcement learning by arguing that agency is fundamentally frame-dependent: Any measurement of a system's agency must be made relative to a reference frame. We support this claim by presenting a philosophical argument that each of the essential properties of agency proposed by Barandiaran et al. (2009) and Moreno (2018) are themselves frame-dependent. We conclude that any basic science of agency requires frame-dependence, and discuss the implications of this claim for reinforcement learning. 

**Abstract (ZH)**: 代理能力是指系统将结果导向目标的能力，这是生物学、哲学、认知科学和人工智能等领域研究的核心主题。确定一个系统是否表现出代理能力是一个众所周知的难题：例如，丹内特（1989年）就指出，如何判断岩石、恒温器或机器人是否具有代理能力是一个难题。在此，我们从强化学习的角度出发， argue rằng代理能力本质上是依赖于框架的：任何对系统代理能力的度量都必须相对于某个参考框架进行。我们通过提出一个哲学论证来支持这一观点，即巴兰达兰等人（2009年）和莫雷诺（2018年）提出的代理能力的每一个基本属性都是依赖于框架的。我们得出结论，任何基本的关于代理能力的科学都需要考虑框架依赖性，并讨论这一结论对强化学习的潜在影响。 

---
# $TAR^2$: Temporal-Agent Reward Redistribution for Optimal Policy Preservation in Multi-Agent Reinforcement Learning 

**Title (ZH)**: $TAR^2$: 时间代理奖励重分配以在多代理 reinforcement 学习中实现最优策略保留 

**Authors**: Aditya Kapoor, Kale-ab Tessera, Mayank Baranwal, Harshad Khadilkar, Stefano Albrecht, Mingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.04864)  

**Abstract**: In cooperative multi-agent reinforcement learning (MARL), learning effective policies is challenging when global rewards are sparse and delayed. This difficulty arises from the need to assign credit across both agents and time steps, a problem that existing methods often fail to address in episodic, long-horizon tasks. We propose Temporal-Agent Reward Redistribution $TAR^2$, a novel approach that decomposes sparse global rewards into agent-specific, time-step-specific components, thereby providing more frequent and accurate feedback for policy learning. Theoretically, we show that $TAR^2$ (i) aligns with potential-based reward shaping, preserving the same optimal policies as the original environment, and (ii) maintains policy gradient update directions identical to those under the original sparse reward, ensuring unbiased credit signals. Empirical results on two challenging benchmarks, SMACLite and Google Research Football, demonstrate that $TAR^2$ significantly stabilizes and accelerates convergence, outperforming strong baselines like AREL and STAS in both learning speed and final performance. These findings establish $TAR^2$ as a principled and practical solution for agent-temporal credit assignment in sparse-reward multi-agent systems. 

**Abstract (ZH)**: 在合作多智能体强化学习（MARL）中，当全局奖励稀疏且延迟时，学习有效的策略具有挑战性。这种困难源于需要在智能体之间和时间步骤之间分配奖励信用，而现有方法在处理 episodic、长期任务时往往难以解决这一问题。我们提出了时间-智能体奖励再分配（TAR²）的新方法，该方法将稀疏的全局奖励分解为智能体特定和时间步骤特定的组件，从而为策略学习提供更频繁和准确的反馈。理论上，我们证明了TAR²（i）与基于潜力的奖励重塑相一致，保留了与原始环境相同的最优策略；（ii）保持与原始稀疏奖励相同的政策梯度更新方向，确保无偏的信用信号。在两个具有挑战性的基准测试（SMACLite和Google Research Football）上的实验证明，TAR²显著稳定了并加速了学习过程，其在学习速度和最终性能上均优于AREL和STAS等强大基准。这些发现将TAR²确立为稀疏奖励多智能体系统中智能体-时间信用分配的原理性且实用的解决方案。 

---
# Probing a Vision-Language-Action Model for Symbolic States and Integration into a Cognitive Architecture 

**Title (ZH)**: 探索视觉-语言-行动模型在符号状态表示中的潜力，并将其集成到认知架构中 

**Authors**: Hong Lu, Hengxu Li, Prithviraj Singh Shahani, Stephanie Herbers, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2502.04558)  

**Abstract**: Vision-language-action (VLA) models hold promise as generalist robotics solutions by translating visual and linguistic inputs into robot actions, yet they lack reliability due to their black-box nature and sensitivity to environmental changes. In contrast, cognitive architectures (CA) excel in symbolic reasoning and state monitoring but are constrained by rigid predefined execution. This work bridges these approaches by probing OpenVLA's hidden layers to uncover symbolic representations of object properties, relations, and action states, enabling integration with a CA for enhanced interpretability and robustness. Through experiments on LIBERO-spatial pick-and-place tasks, we analyze the encoding of symbolic states across different layers of OpenVLA's Llama backbone. Our probing results show consistently high accuracies (> 0.90) for both object and action states across most layers, though contrary to our hypotheses, we did not observe the expected pattern of object states being encoded earlier than action states. We demonstrate an integrated DIARC-OpenVLA system that leverages these symbolic representations for real-time state monitoring, laying the foundation for more interpretable and reliable robotic manipulation. 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型作为通用机器人解决方案展现了潜力，能够将视觉和语言输入转化为机器人动作，但由于其黑箱性质和对环境变化的敏感性，这些模型缺乏可靠性。相比之下，认知架构（CA）在符号推理和状态监控方面表现出色，但受限于预定义的刚性执行过程。本研究通过探究OpenVLA的隐藏层以揭示对象属性、关系和动作状态的符号表示，进而将其与CA结合，从而增强其解释性和鲁棒性。通过在LIBERO空间拣选放置任务上的实验，我们分析了OpenVLA Llama后端不同层中符号状态的编码情况。我们的探究结果显示，大多数层中对象状态和动作状态的编码准确性都非常高（>0.90），但与我们的假设相反，我们没有观察到对象状态比动作状态编码得更早的模式。我们展示了集成的DIARC-OpenVLA系统，该系统利用这些符号表示进行实时状态监控，为更可解释和可靠的机器人操作奠定了基础。 

---
# Active Task Disambiguation with LLMs 

**Title (ZH)**: 使用大语言模型进行主动任务消歧 Paginationalbums 

**Authors**: Katarzyna Kobalczyk, Nicolas Astorga, Tennison Liu, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2502.04485)  

**Abstract**: Despite the impressive performance of large language models (LLMs) across various benchmarks, their ability to address ambiguously specified problems--frequent in real-world interactions--remains underexplored. To address this gap, we introduce a formal definition of task ambiguity and frame the problem of task disambiguation through the lens of Bayesian Experimental Design. By posing clarifying questions, LLM agents can acquire additional task specifications, progressively narrowing the space of viable solutions and reducing the risk of generating unsatisfactory outputs. Yet, generating effective clarifying questions requires LLM agents to engage in a form of meta-cognitive reasoning, an ability LLMs may presently lack. Our proposed approach of active task disambiguation enables LLM agents to generate targeted questions maximizing the information gain. Effectively, this approach shifts the load from implicit to explicit reasoning about the space of viable solutions. Empirical results demonstrate that this form of question selection leads to more effective task disambiguation in comparison to approaches relying on reasoning solely within the space of questions. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各种基准测试中表现出色，但它们在解决模糊指定的问题方面的能力仍待探索——而在现实世界的互动中，这种模糊指定的问题是常见的。为填补这一空白，我们提出了任务模糊的正式定义，并将任务去模糊问题框架化为贝叶斯实验设计的问题。通过提出澄清问题，LLM代理可以获取额外的任务说明，逐步缩小可行解的空间，减少生成不满意输出的风险。然而，生成有效的澄清问题需要LLM代理进行一种形式的元认知推理，而这种能力目前可能是LLMs所缺乏的。我们提出的一种主动任务去模糊方法，使LLM代理能够生成最大化信息增益的针对性问题。实际上，这种方法将推理负载从隐含推理转移到显式推理。实证结果显示，这种问题选择方式在任务去模糊的效率上优于依赖于问题空间内单纯推理的方法。 

---
# Autotelic Reinforcement Learning: Exploring Intrinsic Motivations for Skill Acquisition in Open-Ended Environments 

**Title (ZH)**: 自足强化学习：探索开放环境中技能获取的内在动机 

**Authors**: Prakhar Srivastava, Jasmeet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2502.04418)  

**Abstract**: This paper presents a comprehensive overview of autotelic Reinforcement Learning (RL), emphasizing the role of intrinsic motivations in the open-ended formation of skill repertoires. We delineate the distinctions between knowledge-based and competence-based intrinsic motivations, illustrating how these concepts inform the development of autonomous agents capable of generating and pursuing self-defined goals. The typology of Intrinsically Motivated Goal Exploration Processes (IMGEPs) is explored, with a focus on the implications for multi-goal RL and developmental robotics. The autotelic learning problem is framed within a reward-free Markov Decision Process (MDP), WHERE agents must autonomously represent, generate, and master their own goals. We address the unique challenges in evaluating such agents, proposing various metrics for measuring exploration, generalization, and robustness in complex environments. This work aims to advance the understanding of autotelic RL agents and their potential for enhancing skill acquisition in a diverse and dynamic setting. 

**Abstract (ZH)**: 本文全面概述了自给自足的强化学习（Autotelic Reinforcement Learning, AutoRL），强调内在动机在开放性技能 repertoire 形成中的作用。我们区分了基于知识和基于能力的内在动机，并展示了这些概念如何指导自主代理的开发，使其能够自定义和追求自身设定的目标。探讨了内在动机驱动的目标探索过程（Intrinsically Motivated Goal Exploration Processes, IMGEPs）的类型，并关注其对多目标 RL 和发展型机器人学的影响。在无奖励的马尔可夫决策过程（Reward-Free Markov Decision Process, Reward-Free MDP）框架下，阐述了自给自足学习问题，其中智能体必须自主地表示、生成和掌握自己的目标。本文还探讨了评估这类智能体的独特挑战，并提出了多种度量标准来衡量其在复杂环境中的探索能力、泛化能力和鲁棒性。这项工作旨在推进对自给自足 RL 智能体的理解，并探讨其在多样且动态环境中的技能获取方面的潜在价值。 

---
# Online Location Planning for AI-Defined Vehicles: Optimizing Joint Tasks of Order Serving and Spatio-Temporal Heterogeneous Model Fine-Tuning 

**Title (ZH)**: AI定义车辆的在线位置规划：订单服务与时空异质模型微调联合任务的优化 

**Authors**: Bokeng Zheng, Bo Rao, Tianxiang Zhu, Chee Wei Tan, Jingpu Duan, Zhi Zhou, Xu Chen, Xiaoxi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04399)  

**Abstract**: Advances in artificial intelligence (AI) including foundation models (FMs), are increasingly transforming human society, with smart city driving the evolution of urban this http URL, vehicle crowdsensing (VCS) has emerged as a key enabler, leveraging vehicles' mobility and sensor-equipped capabilities. In particular, ride-hailing vehicles can effectively facilitate flexible data collection and contribute towards urban intelligence, despite resource limitations. Therefore, this work explores a promising scenario, where edge-assisted vehicles perform joint tasks of order serving and the emerging foundation model fine-tuning using various urban data. However, integrating the VCS AI task with the conventional order serving task is challenging, due to their inconsistent spatio-temporal characteristics: (i) The distributions of ride orders and data point-of-interests (PoIs) may not coincide in geography, both following a priori unknown patterns; (ii) they have distinct forms of temporal effects, i.e., prolonged waiting makes orders become instantly invalid while data with increased staleness gradually reduces its utility for model this http URL overcome these obstacles, we propose an online framework based on multi-agent reinforcement learning (MARL) with careful augmentation. A new quality-of-service (QoS) metric is designed to characterize and balance the utility of the two joint tasks, under the effects of varying data volumes and staleness. We also integrate graph neural networks (GNNs) with MARL to enhance state representations, capturing graph-structured, time-varying dependencies among vehicles and across locations. Extensive experiments on our testbed simulator, utilizing various real-world foundation model fine-tuning tasks and the New York City Taxi ride order dataset, demonstrate the advantage of our proposed method. 

**Abstract (ZH)**: 人工智能（AI）的最新进展，包括基础模型（FMs），正日益改变人类社会，智慧城市正推动城市向智能化转变。在此背景下，车辆众感知（VCS）作为关键技术已经出现，利用车辆的移动性和传感器功能。特别是，网约车车辆可以通过灵活的数据收集有效提升城市智能水平，尽管资源有限。因此，本文探索了一个有前景的场景，即边缘辅助车辆同时执行订单服务和新兴基础模型微调任务。然而，将VCS中的AI任务与传统的订单服务任务结合具有挑战性，主要原因在于它们的空间-时间特征不一致：（i）行程订单和感兴趣的点（PoIs）的空间分布可能不吻合，且这些分布遵循未知模式；（ii）它们具有不同的时间效应，即长时间等待会使订单立即失效，而数据的新鲜度降低会逐渐减少其对模型的效用。为克服这些障碍，我们提出了一种基于多智能体强化学习（MARL）的在线框架，并进行了仔细的增强。我们设计了一种新的服务质量（QoS）指标来描述和平衡两项联合任务的效用，在数据量和新鲜度变化的影响下实现这种平衡。我们还结合了图神经网络（GNN）与MARL以增强状态表示，捕捉车辆之间以及不同位置之间的时间变化的图结构依赖关系。在我们的测试床模拟器上，利用各种实际基础模型微调任务以及纽约市出租车行程订单数据集进行的大量实验表明了我们所提方法的优势。 

---
# Position: Emergent Machina Sapiens Urge Rethinking Multi-Agent Paradigms 

**Title (ZH)**: 位置： emergent Machina Sapiens 催生多代理范式的重新思考 

**Authors**: Hepeng Li, Yuhong Liu, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.04388)  

**Abstract**: Artificially intelligent (AI) agents that are capable of autonomous learning and independent decision-making hold great promise for addressing complex challenges across domains like transportation, energy systems, and manufacturing. However, the surge in AI systems' design and deployment driven by various stakeholders with distinct and unaligned objectives introduces a crucial challenge: how can uncoordinated AI systems coexist and evolve harmoniously in shared environments without creating chaos? To address this, we advocate for a fundamental rethinking of existing multi-agent frameworks, such as multi-agent systems and game theory, which are largely limited to predefined rules and static objective structures. We posit that AI agents should be empowered to dynamically adjust their objectives, make compromises, form coalitions, and safely compete or cooperate through evolving relationships and social feedback. Through this paper, we call for a shift toward the emergent, self-organizing, and context-aware nature of these systems. 

**Abstract (ZH)**: 具备自主学习和独立决策能力的人工智能（AI）代理在交通、能源系统和制造业等领域的复杂挑战中具有巨大的潜力。然而，由各种具有不同且不一致目标的利益相关者推动的AI系统的设计和部署的激增带来了一个重要挑战：如何在共享环境中使不协调的AI系统和谐共存并共同发展，而不引发混乱？为了解决这一问题，我们建议从根本上重新审视现有的多代理框架，例如多代理系统和博弈论，这些框架大多局限于预定义的规则和静态的目标结构。我们认为，AI代理应被赋予动态调整其目标、进行妥协、建立联盟以及通过不断演化的人际关系和社会反馈来安全地竞争或合作的能力。通过本文，我们呼吁向这些系统的自发性、自我组织性和情境感知性转变。 

---
# MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf 

**Title (ZH)**: 代理出席：评估聊天生成模型代为出席会议的能力

这个标题翻译成中文时，保持了原有的含义和学术规范，同时确保语言通顺自然。在学术论文中，短语 "MEETING DELEGATE" 在这里被解释为“代理出席”，"Benchmarking LLMs" 被翻译为“评估聊天生成模型”，"Attending Meetings on Our Behalf" 则翻译为“代为出席会议的能力”。这样的翻译能够准确传达原文的含义。 

**Authors**: Lingxiang Hu, Shurun Yuan, Xiaoting Qin, Jue Zhang, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04376)  

**Abstract**: In contemporary workplaces, meetings are essential for exchanging ideas and ensuring team alignment but often face challenges such as time consumption, scheduling conflicts, and inefficient participation. Recent advancements in Large Language Models (LLMs) have demonstrated their strong capabilities in natural language generation and reasoning, prompting the question: can LLMs effectively delegate participants in meetings? To explore this, we develop a prototype LLM-powered meeting delegate system and create a comprehensive benchmark using real meeting transcripts. Our evaluation reveals that GPT-4/4o maintain balanced performance between active and cautious engagement strategies. In contrast, Gemini 1.5 Pro tends to be more cautious, while Gemini 1.5 Flash and Llama3-8B/70B display more active tendencies. Overall, about 60\% of responses address at least one key point from the ground-truth. However, improvements are needed to reduce irrelevant or repetitive content and enhance tolerance for transcription errors commonly found in real-world settings. Additionally, we implement the system in practical settings and collect real-world feedback from demos. Our findings underscore the potential and challenges of utilizing LLMs as meeting delegates, offering valuable insights into their practical application for alleviating the burden of meetings. 

**Abstract (ZH)**: 在当代工作场所中，会议对于交流思想和确保团队协同至关重要，但常常面临时间消耗、日程冲突和参与不充分等挑战。近期大型语言模型（LLMs）的发展显示了其在自然语言生成和推理方面的强大能力，不禁引出一个问题：LLMs能否有效地管理会议参与者？为探究这一问题，我们开发了一个基于LLMs的会议代理系统，并使用真实的会议记录创建了一个全面的基准。我们的评估发现，GPT-4/4o在积极和谨慎参与策略之间保持了平衡的性能。相比之下，Gemini 1.5 Pro更加谨慎，而Gemini 1.5 Flash和Llama3-8B/70B则表现出更强的主动性倾向。总体而言，大约60%的回复至少涵盖了真实情况的关键点。然而，仍需改进以减少无关或重复内容，并增强对常见于实际场景中的转录错误的容忍度。此外，我们在实际应用场景中实施了该系统，并收集了演示的真实反馈。我们的研究成果突显了利用LLMs作为会议代理的潜力与挑战，并为其实用应用场景提供了宝贵的见解，有助于减轻会议负担。 

---
