# MINDSTORES: Memory-Informed Neural Decision Synthesis for Task-Oriented Reinforcement in Embodied Systems 

**Title (ZH)**: MINDSTORES：面向任务的记忆导向神经决策合成在实体系统中的强化学模型 

**Authors**: Anirudh Chari, Suraj Reddy, Aditya Tiwari, Richard Lian, Brian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.19318)  

**Abstract**: While large language models (LLMs) have shown promising capabilities as zero-shot planners for embodied agents, their inability to learn from experience and build persistent mental models limits their robustness in complex open-world environments like Minecraft. We introduce MINDSTORES, an experience-augmented planning framework that enables embodied agents to build and leverage mental models through natural interaction with their environment. Drawing inspiration from how humans construct and refine cognitive mental models, our approach extends existing zero-shot LLM planning by maintaining a database of past experiences that informs future planning iterations. The key innovation is representing accumulated experiences as natural language embeddings of (state, task, plan, outcome) tuples, which can then be efficiently retrieved and reasoned over by an LLM planner to generate insights and guide plan refinement for novel states and tasks. Through extensive experiments in the MineDojo environment, a simulation environment for agents in Minecraft that provides low-level controls for Minecraft, we find that MINDSTORES learns and applies its knowledge significantly better than existing memory-based LLM planners while maintaining the flexibility and generalization benefits of zero-shot approaches, representing an important step toward more capable embodied AI systems that can learn continuously through natural experience. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在作为搭载代理的零样本规划者方面展现出了令人鼓舞的能力，但由于它们无法从经验中学习并构建持久的心理模型，这限制了它们在像Minecraft这样复杂的开放世界环境中的鲁棒性。我们引入了MINDSTORES，这是一种增强经验的规划框架，它使搭载代理能够通过与其环境的自然互动来构建和利用心理模型。该方法从人类如何构建和改进认知心理模型中汲取灵感，通过维护一个包含过去经验的数据库来扩展现有的零样本LLM规划，从而指导未来的规划迭代。关键创新是将积累的经验表示为（状态、任务、计划、结果）元组的自然语言嵌入，这些嵌入可以被LLM规划器高效地检索和推理，从而生成关于新型状态和任务的见解并指导计划改进。通过在MineDojo环境中进行广泛的实验，一个为Minecraft中的代理提供底层控制的模拟环境，我们发现，MINDSTORES在学习和应用其知识方面明显优于现有的基于记忆的LLM规划器，同时维护了零样本方法的灵活性和泛化优势，标志着向能够通过自然经验持续学习的更强大搭载式智能系统迈进的重要一步。 

---
# SHARPIE: A Modular Framework for Reinforcement Learning and Human-AI Interaction Experiments 

**Title (ZH)**: SHARPIE：强化学习与人机交互实验的模块化框架 

**Authors**: Hüseyin Aydın, Kevin Dubois-Godin, Libio Goncalvez Braz, Floris den Hengst, Kim Baraka, Mustafa Mert Çelikok, Andreas Sauter, Shihan Wang, Frans A. Oliehoek  

**Link**: [PDF](https://arxiv.org/pdf/2501.19245)  

**Abstract**: Reinforcement learning (RL) offers a general approach for modeling and training AI agents, including human-AI interaction scenarios. In this paper, we propose SHARPIE (Shared Human-AI Reinforcement Learning Platform for Interactive Experiments) to address the need for a generic framework to support experiments with RL agents and humans. Its modular design consists of a versatile wrapper for RL environments and algorithm libraries, a participant-facing web interface, logging utilities, deployment on popular cloud and participant recruitment platforms. It empowers researchers to study a wide variety of research questions related to the interaction between humans and RL agents, including those related to interactive reward specification and learning, learning from human feedback, action delegation, preference elicitation, user-modeling, and human-AI teaming. The platform is based on a generic interface for human-RL interactions that aims to standardize the field of study on RL in human contexts. 

**Abstract (ZH)**: 强化学习（RL）提供了一种通用的方法来建模和训练AI代理，包括人类与AI的交互场景。本文中，我们提出了一种名为SHARPIE（共享人类与AI强化学习平台，用于交互实验）的方案，以应对在实验中支持强化学习代理和人类的一般框架需求。该平台的设计具有模块化特性，包含可灵活使用的RL环境包装器和算法库，面向参与者的网页界面，日志工具，以及在流行的云平台和参与者招募平台上部署的能力。它使研究者能够探讨人类与RL代理互动的各种研究问题，包括交互奖励指定与学习、从人类反馈中学习、动作委托、偏好获取、用户建模以及人类与AI的合作。该平台基于一种通用的人类与RL交互接口，旨在在人类情境下的强化学习领域实现标准化研究。 

---
# An Empirical Game-Theoretic Analysis of Autonomous Cyber-Defence Agents 

**Title (ZH)**: 基于实证博弈论分析的自主网络防御代理研究 

**Authors**: Gregory Palmer, Luke Swaby, Daniel J.B. Harrold, Matthew Stewart, Alex Hiles, Chris Willis, Ian Miles, Sara Farmer  

**Link**: [PDF](https://arxiv.org/pdf/2501.19206)  

**Abstract**: The recent rise in increasingly sophisticated cyber-attacks raises the need for robust and resilient autonomous cyber-defence (ACD) agents. Given the variety of cyber-attack tactics, techniques and procedures (TTPs) employed, learning approaches that can return generalisable policies are desirable. Meanwhile, the assurance of ACD agents remains an open challenge. We address both challenges via an empirical game-theoretic analysis of deep reinforcement learning (DRL) approaches for ACD using the principled double oracle (DO) algorithm. This algorithm relies on adversaries iteratively learning (approximate) best responses against each others' policies; a computationally expensive endeavour for autonomous cyber operations agents. In this work we introduce and evaluate a theoretically-sound, potential-based reward shaping approach to expedite this process. In addition, given the increasing number of open-source ACD-DRL approaches, we extend the DO formulation to allow for multiple response oracles (MRO), providing a framework for a holistic evaluation of ACD approaches. 

**Abstract (ZH)**: 近年来，日益复杂的网络攻击上升趋势加剧了对强大且具有适应能力的自主网络防御（ACD）代理的需求。鉴于攻击者采用各种各样的攻击战术、技术和程序（TTPs），具备泛化能力的学习方法变得尤为重要。同时，确保ACD代理的安全性仍是一个开放性挑战。我们通过实证博弈论分析的方法，利用规范化的双或然算法（DO算法）解决这两个挑战，该方法基于对手迭代学习彼此策略的近似最佳反应；在自主网络操作代理中，这一过程存在较高的计算成本。在这项工作中，我们引入并评估了一种理论上合理的基于潜力的奖励塑形方法，以加速这一过程。此外，鉴于开源ACD-DRL方法的数量日益增多，我们扩展了DO形式化方法，引入了多反应或然算法（MRO），提供了一个全面评估ACD方法的框架。 

---
# LLM-Generated Heuristics for AI Planning: Do We Even Need Domain-Independence Anymore? 

**Title (ZH)**: 生成的LLM启发式方法在AI规划中的应用：我们还需要领域无关性吗？ 

**Authors**: Alexander Tuisov, Yonatan Vernik, Alexander Shleyfman  

**Link**: [PDF](https://arxiv.org/pdf/2501.18784)  

**Abstract**: Domain-independent heuristics have long been a cornerstone of AI planning, offering general solutions applicable across a wide range of tasks without requiring domain-specific engineering. However, the advent of large language models (LLMs) presents an opportunity to generate heuristics tailored to specific planning problems, potentially challenging the necessity of domain independence as a strict design principle. In this paper, we explore the use of LLMs to automatically derive planning heuristics from task descriptions represented as successor generators and goal tests written in general purpose programming language. We investigate the trade-offs between domain-specific LLM-generated heuristics and traditional domain-independent methods in terms of computational efficiency and explainability. Our experiments demonstrate that LLMs can create heuristics that achieve state-of-the-art performance on some standard IPC domains, as well as their ability to solve problems that lack an adequate Planning Domain Definition Language ({\sc pddl}) representation. We discuss whether these results signify a paradigm shift and how they can complement existing approaches. 

**Abstract (ZH)**: 领域无关的经验在人工智能规划中一直是一个基石，提供了一类适用于广泛任务的通用解决方案，无需针对特定领域进行工程设计。然而，大型语言模型（LLMs）的出现为生成针对特定规划问题定制的经验提供了一个机会，可能挑战领域无关性作为严格设计原则的必要性。在本文中，我们探讨了使用LLMs从以通用编程语言编写的任务描述（表示为后续生成器和目标测试）中自动推导规划经验的方法。我们研究了领域特定的LLM生成经验与传统领域无关方法之间的权衡，尤其是在计算效率和可解释性方面的权衡。实验结果表明，LLMs能够创建在一些标准IPC领域中达到最佳性能的经验，并展示了它们解决缺乏适当规划领域定义语言（PDDL）表示的问题的能力。我们讨论了这些结果是否标志着范式转变，并探讨了它们如何补充现有的方法。 

---
# Simulation Streams: A Programming Paradigm for Controlling Large Language Models and Building Complex Systems with Generative AI 

**Title (ZH)**: 仿真流：控制大规模语言模型和构建基于生成式AI复杂系统的编程范式 

**Authors**: Peter Sunehag, Joel Z. Leibo  

**Link**: [PDF](https://arxiv.org/pdf/2501.18668)  

**Abstract**: We introduce Simulation Streams, a programming paradigm designed to efficiently control and leverage Large Language Models (LLMs) for complex, dynamic simulations and agentic workflows. Our primary goal is to create a minimally interfering framework that harnesses the agentic abilities of LLMs while addressing their limitations in maintaining consistency, selectively ignoring/including information, and enforcing strict world rules. Simulation Streams achieves this through a state-based approach where variables are modified in sequential steps by "operators," producing output on a recurring format and adhering to consistent rules for state variables. This approach focus the LLMs on defined tasks, while aiming to have the context stream remain "in-distribution". The approach incorporates an Entity-Component-System (ECS) architecture to write programs in a more intuitive manner, facilitating reuse of workflows across different components and entities. This ECS approach enhances the modularity of the output stream, allowing for complex, multi-entity simulations while maintaining format consistency, information control, and rule enforcement. It is supported by a custom editor that aids in creating, running, and analyzing simulations. We demonstrate the versatility of simulation streams through an illustrative example of an ongoing market economy simulation, a social simulation of three characters playing a game of catch in a park and a suite of classical reinforcement learning benchmark tasks. These examples showcase Simulation Streams' ability to handle complex, evolving scenarios over 100s-1000s of iterations, facilitate comparisons between different agent workflows and models, and maintain consistency and continued interesting developments in LLM-driven simulations. 

**Abstract (ZH)**: 我们引入了Simulation Streams编程范式，该范式旨在高效控制和利用大型语言模型（LLMs）进行复杂的动态模拟和有代理性的工作流程。我们的主要目标是创建一个干扰最小的框架，充分利用LLMs的代理能力，同时解决它们在保持一致性、选择性忽略或包含信息以及强制执行严格的世界规则方面的限制。Simulation Streams 通过基于状态的方法实现这一目标，这种方法通过“操作符”在顺序步骤中修改变量，以一致的格式输出，并遵循状态变量的一致规则。这种方法将重点放在定义的任务上，同时让上下文流保持“在分布”状态。该方法采用实体-组件-系统（ECS）架构，使编程更加直观，有利于在不同组件和实体之间重用工作流程。这种ECS方法增强了输出流的模块性，可以在保持格式一致性、信息控制和规则执行的前提下，进行复杂、多实体的模拟。该方法还借助自定义编辑器来辅助创建、运行和分析模拟。我们通过以下示例展示了Simulation Streams的多功能性：一个持续的市场经济模拟、三个角色在公园里玩接球游戏的社会模拟，以及一系列经典的强化学习基准任务。这些示例展示了Simulation Streams在数百到数千次迭代中处理复杂、演变场景的能力，支持不同代理工作流程和模型之间的比较，并在LLM驱动的模拟中保持一致性及持续的有趣发展。 

---
# Vintix: Action Model via In-Context Reinforcement Learning 

**Title (ZH)**: 维尼亚克斯：基于上下文强化学习的行动模型 

**Authors**: Andrey Polubarov, Nikita Lyubaykin, Alexander Derevyagin, Ilya Zisman, Denis Tarasov, Alexander Nikulin, Vladislav Kurenkov  

**Link**: [PDF](https://arxiv.org/pdf/2501.19400)  

**Abstract**: In-Context Reinforcement Learning (ICRL) represents a promising paradigm for developing generalist agents that learn at inference time through trial-and-error interactions, analogous to how large language models adapt contextually, but with a focus on reward maximization. However, the scalability of ICRL beyond toy tasks and single-domain settings remains an open challenge. In this work, we present the first steps toward scaling ICRL by introducing a fixed, cross-domain model capable of learning behaviors through in-context reinforcement learning. Our results demonstrate that Algorithm Distillation, a framework designed to facilitate ICRL, offers a compelling and competitive alternative to expert distillation to construct versatile action models. These findings highlight the potential of ICRL as a scalable approach for generalist decision-making systems. Code to be released at this https URL 

**Abstract (ZH)**: 上下文中的强化学习（ICRL）代表了一种有潜力的范式，通过在推理时的试错交互来开发能够在多种场景下学习的一般性代理，类似于大型语言模型通过上下文调整的方式，但更侧重于奖励最大化。然而，ICRL 在扩展到玩具任务和单一领域设置之外依然面临着一个开放的挑战。在这项工作中，我们提出了通过引入一个固定跨领域的模型，来首次尝试扩展 ICRL 的步骤。该模型能够通过上下文中的强化学习来学习行为。我们的实验结果表明，算法蒸馏（Algorithm Distillation），一种旨在促进 ICRL 的框架，提供了与专家蒸馏相比具有吸引力且竞争性的替代方案，用于构建多功能的行为模型。这些发现强调了 ICRL 作为一种可用于拓展的一般性决策系统方法的潜力。代码将在以下链接发布：该 <https://your_link_here> URL 

---
# Enabling Autonomic Microservice Management through Self-Learning Agents 

**Title (ZH)**: 通过自学习代理实现自主微服务管理 

**Authors**: Fenglin Yu, Fangkai Yang, Xiaoting Qin, Zhiyang Zhang, Jue Zhang, Qingwei Lin, Hongyu Zhang, Yingnong Dang, Saravan Rajmohan, Dongmei Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.19056)  

**Abstract**: The increasing complexity of modern software systems necessitates robust autonomic self-management capabilities. While Large Language Models (LLMs) demonstrate potential in this domain, they often face challenges in adapting their general knowledge to specific service contexts. To address this limitation, we propose ServiceOdyssey, a self-learning agent system that autonomously manages microservices without requiring prior knowledge of service-specific configurations. By leveraging curriculum learning principles and iterative exploration, ServiceOdyssey progressively develops a deep understanding of operational environments, reducing dependence on human input or static documentation. A prototype built with the Sock Shop microservice demonstrates the potential of this approach for autonomic microservice management. 

**Abstract (ZH)**: 现代软件系统的日益复杂性要求其具备强大的自适应自主管理能力。尽管大型语言模型（LLMs）在这一领域展现出潜在的应用前景，但在将通用知识适配到具体服务环境方面仍面临挑战。为解决这一局限性，我们提出了一种名为ServiceOdyssey的自学习代理系统，该系统能够自主管理微服务，而无需了解特定服务的配置信息。通过利用课程学习原则和迭代探索，ServiceOdyssey逐步建立起对运行环境的深刻理解，减少了对人工输入或静态文档的依赖。基于Sock Shop微服务构建的原型表明，这种方法有潜力用于自主微服务管理。 

---
# Towards Physiologically Sensible Predictions via the Rule-based Reinforcement Learning Layer 

**Title (ZH)**: 基于规则的强化学习层实现生理上合理的预测 

**Authors**: Lingwei Zhu, Zheng Chen, Yukie Nagai, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.19055)  

**Abstract**: This paper adds to the growing literature of reinforcement learning (RL) for healthcare by proposing a novel paradigm: augmenting any predictor with Rule-based RL Layer (RRLL) that corrects the model's physiologically impossible predictions. Specifically, RRLL takes as input states predicted labels and outputs corrected labels as actions. The reward of the state-action pair is evaluated by a set of general rules. RRLL is efficient, general and lightweight: it does not require heavy expert knowledge like prior work but only a set of impossible transitions. This set is much smaller than all possible transitions; yet it can effectively reduce physiologically impossible mistakes made by the state-of-the-art predictor models. We verify the utility of RRLL on a variety of important healthcare classification problems and observe significant improvements using the same setup, with only the domain-specific set of impossibility changed. In-depth analysis shows that RRLL indeed improves accuracy by effectively reducing the presence of physiologically impossible predictions. 

**Abstract (ZH)**: 本文通过对医疗保健领域强化学习（RL）的研究贡献，提出了一种新颖的范式：将基于规则的强化学习层（RRLL）添加到任何预测器中，以纠正模型的生理上不可能的预测结果。具体而言，RRLL 将预测状态标签作为输入，并输出修正后的标签作为动作。状态-动作对的奖励由一套通用规则进行评估。RRLL 具有效率高、通用性强且轻量化的特性：它不需要以往工作所需的大量专家知识，而只需要一套不可能发生的转换状态集。虽然这一集合并未包含所有可能的转换状态，但仍然能够有效减少最先进的预测模型所犯的生理上不可能的错误。我们通过在多种关键的医疗保健分类问题上验证 RRLL 的效用，并在保持相同实验设置的情况下观察到显著改进，仅改变特定领域的不可能性状态集。深入分析表明，RRLL 通过有效减少生理上不可能的预测结果的存在而真正提高了准确性。 

---
# Deep Learning based Quasi-consciousness Training for Robot Intelligent Model 

**Title (ZH)**: 基于深度学习的类意识训练方法研究：用于机器人智能模型 

**Authors**: Yuchun Li, Fang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18955)  

**Abstract**: This paper explores a deep learning based robot intelligent model that renders robots learn and reason for complex tasks. First, by constructing a network of environmental factor matrix to stimulate the learning process of the robot intelligent model, the model parameters must be subjected to coarse & fine tuning to optimize the loss function for minimizing the loss score, meanwhile robot intelligent model can fuse all previously known concepts together to represent things never experienced before, which need robot intelligent model can be generalized extensively. Secondly, in order to progressively develop a robot intelligent model with primary consciousness, every robot must be subjected to at least 1~3 years of special school for training anthropomorphic behaviour patterns to understand and process complex environmental information and make rational decisions. This work explores and delivers the potential application of deep learning-based quasi-consciousness training in the field of robot intelligent model. 

**Abstract (ZH)**: 本文探讨了一种基于深度学习的机器人智能模型，该模型使机器人能够学习和推理以应对复杂的任务。首先，通过构建环境因素矩阵网络来模拟机器人智能模型的学习过程，模型参数需要进行粗调和细调，以优化损失函数并最小化损失分数。同时，机器人智能模型可以融合所有已知概念，以表示从未经历过的新型事物，这就要求机器人智能模型具有广泛的应用能力。其次，为了逐步构建具有初级意识的机器人智能模型，每台机器人必须接受至少1至3年的特殊学校训练，以学习和发展拟人行为模式，理解和处理复杂环境信息，并做出合理的决策。本文研究并展示了基于深度学习的拟意识训练在机器人智能模型领域的潜在应用。 

---
# KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search 

**Title (ZH)**: KBQA-o1: 基于蒙特卡洛树搜索的知识库问答 

**Authors**: Haoran Luo, Haihong E, Yikai Guo, Qika Lin, Xiaobao Wu, Xinyu Mu, Wenhao Liu, Meina Song, Yifan Zhu, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2501.18922)  

**Abstract**: Knowledge Base Question Answering (KBQA) aims to answer natural language questions with a large-scale structured knowledge base (KB). Despite advancements with large language models (LLMs), KBQA still faces challenges in weak KB awareness, imbalance between effectiveness and efficiency, and high reliance on annotated data. To address these challenges, we propose KBQA-o1, a novel agentic KBQA method with Monte Carlo Tree Search (MCTS). It introduces a ReAct-based agent process for stepwise logical form generation with KB environment exploration. Moreover, it employs MCTS, a heuristic search method driven by policy and reward models, to balance agentic exploration's performance and search space. With heuristic exploration, KBQA-o1 generates high-quality annotations for further improvement by incremental fine-tuning. Experimental results show that KBQA-o1 outperforms previous low-resource KBQA methods with limited annotated data, boosting Llama-3.1-8B model's GrailQA F1 performance to 78.5% compared to 48.5% of the previous sota method with GPT-3.5-turbo. 

**Abstract (ZH)**: 知识图谱问答（KBQA）旨在利用大规模结构化的知识图谱（KB）回答自然语言问题。尽管大型语言模型（LLMs）的进步，KBQA在知识图谱弱意识、效果与效率之间的不平衡以及对标注数据的高度依赖等方面仍然面临挑战。为了克服这些挑战，我们提出了一种名为KBQA-o1的新颖代理型KBQA方法，该方法结合了蒙特卡洛树搜索（MCTS）。它引入了基于ReAct的代理进程，用于逐步逻辑形式生成和知识图谱环境探索。此外，它利用MCTS，一种由策略和奖励模型驱动的启发式搜索方法，来平衡代理探索性能与搜索空间。通过启发式探索，KBQA-o1能够生成高质量的注释，进一步通过增量微调进行改进。实验结果显示，KBQA-o1在有限标注数据的情况下，超越了之前的低资源KBQA方法，将Llama-3.1-8B模型的GrailQA F1性能提高到78.5%，而之前的SOTA方法使用GPT-3.5-turbo时仅为48.5%。 

---
# UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent 

**Title (ZH)**: UP-VLA：统一的具身智能体理解与预测模型 

**Authors**: Jianke Zhang, Yanjiang Guo, Yucheng Hu, Xiaoyu Chen, Xiang Zhu, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.18867)  

**Abstract**: Recent advancements in Vision-Language-Action (VLA) models have leveraged pre-trained Vision-Language Models (VLMs) to improve the generalization capabilities. VLMs, typically pre-trained on vision-language understanding tasks, provide rich semantic knowledge and reasoning abilities. However, prior research has shown that VLMs often focus on high-level semantic content and neglect low-level features, limiting their ability to capture detailed spatial information and understand physical dynamics. These aspects, which are crucial for embodied control tasks, remain underexplored in existing pre-training paradigms. In this paper, we investigate the training paradigm for VLAs, and introduce \textbf{UP-VLA}, a \textbf{U}nified VLA model training with both multi-modal \textbf{U}nderstanding and future \textbf{P}rediction objectives, enhancing both high-level semantic comprehension and low-level spatial understanding. Experimental results show that UP-VLA achieves a 33% improvement on the Calvin ABC-D benchmark compared to the previous state-of-the-art method. Additionally, UP-VLA demonstrates improved success rates in real-world manipulation tasks, particularly those requiring precise spatial information. 

**Abstract (ZH)**: 近年来，视觉-语言-动作（VLA）模型的进步利用预训练的视觉-语言模型（VLMs）来提高泛化能力。VLMs通常在视觉-语言理解任务上进行预训练，提供丰富的语义知识和推理能力。然而，先前的研究表明，VLMs往往专注于高层次的语义内容，而忽略了低级特征，这限制了它们捕获详细的空间信息和理解物理动态的能力。这些方面对于实体控制任务至关重要，在现有的预训练范式中仍然没有得到充分探索。本文探讨了VLA的训练范式，并引入了**UP-VLA**模型，该模型结合了多模态**U**nderstanding和未来**P**rediction的目标，从而增强高层次语义理解和低层次空间理解。实验结果表明，UP-VLA在Calvin ABC-D基准测试上的表现比之前最先进的方法提高了33%。此外，UP-VLA在现实世界的操作任务中表现出更高的成功率，特别是在需要精确空间信息的任务中。 

---
# Survey and Improvement Strategies for Gene Prioritization with Large Language Models 

**Title (ZH)**: 大型语言模型在基因优先级确定中的调查与改进策略 

**Authors**: Matthew Neeley, Guantong Qi, Guanchu Wang, Ruixiang Tang, Dongxue Mao, Chaozhong Liu, Sasidhar Pasupuleti, Bo Yuan, Fan Xia, Pengfei Liu, Zhandong Liu, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18794)  

**Abstract**: Rare diseases are challenging to diagnose due to limited patient data and genetic diversity. Despite advances in variant prioritization, many cases remain undiagnosed. While large language models (LLMs) have performed well in medical exams, their effectiveness in diagnosing rare genetic diseases has not been assessed. To identify causal genes, we benchmarked various LLMs for gene prioritization. Using multi-agent and Human Phenotype Ontology (HPO) classification, we categorized patients based on phenotypes and solvability levels. As gene set size increased, LLM performance deteriorated, so we used a divide-and-conquer strategy to break the task into smaller subsets. At baseline, GPT-4 outperformed other LLMs, achieving near 30% accuracy in ranking causal genes correctly. The multi-agent and HPO approaches helped distinguish confidently solved cases from challenging ones, highlighting the importance of known gene-phenotype associations and phenotype specificity. We found that cases with specific phenotypes or clear associations were more accurately solved. However, we observed biases toward well-studied genes and input order sensitivity, which hindered gene prioritization. Our divide-and-conquer strategy improved accuracy by overcoming these biases. By utilizing HPO classification, novel multi-agent techniques, and our LLM strategy, we improved causal gene identification accuracy compared to our baseline evaluation. This approach streamlines rare disease diagnosis, facilitates reanalysis of unsolved cases, and accelerates gene discovery, supporting the development of targeted diagnostics and therapies. 

**Abstract (ZH)**: 罕见疾病由于患者数据有限和遗传多样性高，诊断起来具有挑战性。尽管在变体优先级排序方面取得了进展，但仍有许多病例未能确诊。虽然大型语言模型（LLMs）在医学考试中表现出色，但它们在诊断罕见遗传疾病方面的有效性尚未得到评估。为了识别致病变异，我们对各种LLM进行了基因优先级排序的基准测试。利用多智能体系统和人类表型ontology（HPO）分类，我们根据表型和解决难度将患者分为不同的类别。随着基因组集的增大，LLM的表现逐渐下降，因此我们采用分而治之的策略将任务分解成更小的子集。基线测试中，GPT-4 在排序致病变异方面表现优于其他LLM，准确率接近30%。多智能体系统和HPO方法有助于区分易解和难解的病例，突显了已知基因-表型关联和表型特异性的重要性。我们发现，具有特定表型或明确关联的病例更易得到准确解决。然而，我们观察到对研究较多的基因存在偏差，并且输入顺序的敏感性影响了基因优先级排序。我们的分而治之策略通过克服这些偏差提高了准确率。通过利用HPO分类、新型多智能体技术和我们的LLM策略，我们提高了致病变异识别的准确性，与基线评估相比有所改进。这种 approach 简化了罕见疾病的诊断，促进了未解病例的重新分析，加速了基因发现，支持了针对性诊断和治疗的发展。 

---
# Cogito, ergo sum: A Neurobiologically-Inspired Cognition-Memory-Growth System for Code Generation 

**Title (ZH)**: 我思故我在：一种受神经生物学启发的思维-记忆-成长系统及其在代码生成中的应用 

**Authors**: Yanlong Li, Jindong Li, Qi Wang, Menglin Yang, He Kong, Shengsheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18653)  

**Abstract**: Large language models based Multi Agent Systems (MAS) have demonstrated promising performance for enhancing the efficiency and accuracy of code generation tasks. However,most existing methods follow a conventional sequence of planning, coding, and debugging,which contradicts the growth-driven nature of human learning process. Additionally,the frequent information interaction between multiple agents inevitably involves high computational costs. In this paper,we propose Cogito,a neurobiologically inspired multi-agent framework to enhance the problem-solving capabilities in code generation tasks with lower cost. Specifically,Cogito adopts a reverse sequence: it first undergoes debugging, then coding,and finally planning. This approach mimics human learning and development,where knowledge is acquired progressively. Accordingly,a hippocampus-like memory module with different functions is designed to work with the pipeline to provide quick retrieval in similar tasks. Through this growth-based learning model,Cogito accumulates knowledge and cognitive skills at each stage,ultimately forming a Super Role an all capable agent to perform the code generation task. Extensive experiments against representative baselines demonstrate the superior performance and efficiency of Cogito. The code is publicly available at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型的多智能体系统（MAS）在提升代码生成任务的效率和准确性方面展现出了令人鼓舞的性能。然而，大多数现有方法遵循传统的规划、编码和调试顺序，这与人类学习过程的成长驱动性质相矛盾。此外，多个智能体之间的频繁信息交互不可避免地带来了高昂的计算成本。在本文中，我们提出了一种名为Cogito的神经生物学启发式多智能体框架，以较低的成本提升代码生成任务中的问题解决能力。具体来说，Cogito采用逆序流程：首先进行调试，然后编码，最后是规划。这种方法模仿了人类的学习和发展过程，在该过程中，知识是逐步获取的。因此，我们设计了一个类似于海马体的记忆模块，它与工作流结合，以提供类似任务的快速检索功能。通过基于成长的学习模型，Cogito在每个阶段积累知识和认知技能，最终形成一个全能的超级智能体来执行代码生成任务。 extensive实验表明，Cogito在代表性的基线方法上具有更优越的性能和效率。代码已在此处公开：[此链接]。 

---
# Layered Chain-of-Thought Prompting for Multi-Agent LLM Systems: A Comprehensive Approach to Explainable Large Language Models 

**Title (ZH)**: 多层思维链提示在多代理大语言模型系统中的应用：一种全面的可解释大语言模型方法 

**Authors**: Manish Sanwal  

**Link**: [PDF](https://arxiv.org/pdf/2501.18645)  

**Abstract**: Large Language Models (LLMs) leverage chain-of-thought (CoT) prompting to provide step-by-step rationales, improving performance on complex tasks. Despite its benefits, vanilla CoT often fails to fully verify intermediate inferences and can produce misleading explanations. In this work, we propose Layered Chain-of-Thought (Layered-CoT) Prompting, a novel framework that systematically segments the reasoning process into multiple layers, each subjected to external checks and optional user feedback. We expand on the key concepts, present three scenarios -- medical triage, financial risk assessment, and agile engineering -- and demonstrate how Layered-CoT surpasses vanilla CoT in terms of transparency, correctness, and user engagement. By integrating references from recent arXiv papers on interactive explainability, multi-agent frameworks, and agent-based collaboration, we illustrate how Layered-CoT paves the way for more reliable and grounded explanations in high-stakes domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）利用链式思考（CoT）提示提供逐步的推理过程，从而在复杂任务上表现出色。尽管具有诸多优势，传统的CoT往往未能完全验证中间推理，且可能生成误导性的解释。在这项研究中，我们提出了一种新颖的框架——分层链式思考（Layered-CoT）提示，该框架系统地将推理过程划分为多个层次，每个层次都接受外部检查并可选地接受用户反馈。我们详细介绍了核心概念，并提出了三个应用场景——医疗分诊、金融风险评估和敏捷工程——展示了Layered-CoT在透明度、正确性和用户参与度方面如何超越传统的CoT。通过结合近期arXiv论文中关于交互式解释、多智能体框架以及基于代理的合作研究中的参考，我们阐述了Layered-CoT如何为高风险领域中的更可靠和具体的解释开辟道路。 

---
# STAMP: Scalable Task And Model-agnostic Collaborative Perception 

**Title (ZH)**: STAMP：可扩展的任务和模型无关协作感知 

**Authors**: Xiangbo Gao, Runsheng Xu, Jiachen Li, Ziran Wang, Zhiwen Fan, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18616)  

**Abstract**: Perception is crucial for autonomous driving, but single-agent perception is often constrained by sensors' physical limitations, leading to degraded performance under severe occlusion, adverse weather conditions, and when detecting distant objects. Multi-agent collaborative perception offers a solution, yet challenges arise when integrating heterogeneous agents with varying model architectures. To address these challenges, we propose STAMP, a scalable task- and model-agnostic, collaborative perception pipeline for heterogeneous agents. STAMP utilizes lightweight adapter-reverter pairs to transform Bird's Eye View (BEV) features between agent-specific and shared protocol domains, enabling efficient feature sharing and fusion. This approach minimizes computational overhead, enhances scalability, and preserves model security. Experiments on simulated and real-world datasets demonstrate STAMP's comparable or superior accuracy to state-of-the-art models with significantly reduced computational costs. As a first-of-its-kind task- and model-agnostic framework, STAMP aims to advance research in scalable and secure mobility systems towards Level 5 autonomy. Our project page is at this https URL and the code is available at this https URL. 

**Abstract (ZH)**: 感知在自主驾驶中至关重要，但单个代理的感知常常受到传感器物理限制的制约，在严重遮挡、恶劣天气条件以及检测远距离目标时，其性能会下降。多代理协作感知提供了解决方案，但在集成具有不同模型架构的异构代理时，存在挑战。为应对这些挑战，我们提出了一种名为STAMP的新框架，它是一种可扩展、任务和模型无关的协作感知管道，适用于异构代理。STAMP利用轻量级的适配器-反向器对，在特定代理领域和共享协议域之间转换鸟瞰视图(BEV)特征，从而实现高效的特征共享和融合。这种方法减少了计算开销、增强了可扩展性，并确保了模型安全。实验表明，STAMP在模拟数据集和真实世界数据集上的准确度与最先进的模型相当或更优，且计算成本显著降低。作为首款任务和模型无关的框架，STAMP旨在推进可扩展和安全的移动系统研究，朝着Level 5自主驾驶迈进。我们的项目页面可以访问此处[this https URL]，代码可以在此处[this https URL]获取。 

---
# SELMA: A Speech-Enabled Language Model for Virtual Assistant Interactions 

**Title (ZH)**: SELMA：一种语音启用的语言模型，适用于虚拟助手交互 

**Authors**: Dominik Wagner, Alexander Churchill, Siddarth Sigtia, Erik Marchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.19377)  

**Abstract**: In this work, we present and evaluate SELMA, a Speech-Enabled Language Model for virtual Assistant interactions that integrates audio and text as inputs to a Large Language Model (LLM). SELMA is designed to handle three primary and two auxiliary tasks related to interactions with virtual assistants simultaneously within a single end-to-end model. We employ low-rank adaptation modules for parameter-efficient training of both the audio encoder and the LLM. Additionally, we implement a feature pooling strategy enabling the system to recognize global patterns and improve accuracy on tasks less reliant on individual sequence elements. Experimental results on Voice Trigger (VT) detection, Device-Directed Speech Detection (DDSD), and Automatic Speech Recognition (ASR), demonstrate that our approach both simplifies the typical input processing pipeline of virtual assistants significantly and also improves performance compared to dedicated models for each individual task. SELMA yields relative Equal-Error Rate improvements of 64% on the VT detection task, and 22% on DDSD, while also achieving word error rates close to the baseline. 

**Abstract (ZH)**: 在本文中，我们介绍了并评估了SELMA（Speech-Enabled Language Model for Virtual Assistant Interactions），这是一种结合了音频和文本输入的大语言模型（LLM），用于虚拟助手交互。SELMA 设计用于在同一端到端模型中同时处理与虚拟助手交互相关的三大主要任务和两大辅助任务。我们采用低秩适应模块对音频编码器和大语言模型进行参数高效的训练。此外，我们实现了一种特征聚合策略，使系统能够识别全局模式并提高对较少依赖于个体序列元素的任务的准确性。在Voice Trigger（VT）检测、Device-Directed Speech Detection（DDSD）和自动语音识别（ASR）等实验中的结果表明，我们的方法不仅显著简化了虚拟助手的典型输入处理流程，还显著提高了性能，相较于针对每个单独任务的专用模型。在VT检测任务中，SELMA 的相对等错误率（Equal-Error Rate, EER）改进了64%，在DDSD任务中改进了22%，同时在单词错误率（Word Error Rate, WER）方面接近基线水平。 

---
# Towards Safe AI Clinicians: A Comprehensive Study on Large Language Model Jailbreaking in Healthcare 

**Title (ZH)**: 朝向安全的AI临床医生：大型语言模型在医疗领域的牢笼破解全面研究 

**Authors**: Hang Zhang, Qian Lou, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18632)  

**Abstract**: Large language models (LLMs) are increasingly utilized in healthcare applications. However, their deployment in clinical practice raises significant safety concerns, including the potential spread of harmful information. This study systematically assesses the vulnerabilities of six LLMs to three advanced black-box jailbreaking techniques within medical contexts. To quantify the effectiveness of these techniques, we propose an automated and domain-adapted agentic evaluation pipeline. Experiment results indicate that leading commercial and open-source LLMs are highly vulnerable to medical jailbreaking attacks. To bolster model safety and reliability, we further investigate the effectiveness of Continual Fine-Tuning (CFT) in defending against medical adversarial attacks. Our findings underscore the necessity for evolving attack methods evaluation, domain-specific safety alignment, and LLM safety-utility balancing. This research offers actionable insights for advancing the safety and reliability of AI clinicians, contributing to ethical and effective AI deployment in healthcare. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗领域的应用日益增多。然而，在临床实践中部署这些模型引发了重要的安全性问题，包括潜在的有害信息传播风险。本研究系统地评估了六种LLM在医疗场景下对三种高级黑盒囚笼突破技术的脆弱性。为了量化这些技术的效果，我们提出了一种自动且领域适应的代理评估管道。实验结果显示，领先的商用和开源LLM对医疗囚笼突破攻击高度脆弱。为了增强模型的安全性和可靠性，我们进一步研究了持续微调（CFT）在防御医疗对抗攻击方面的有效性。我们的研究结果强调了评估攻击方法演变、特定领域安全性对齐以及LLM安全与效用平衡的必要性。本研究提供了关于提升AI临床医生安全性和可靠性的可操作见解，有助于在医疗保健中实现伦理和有效的AI部署。 

---
