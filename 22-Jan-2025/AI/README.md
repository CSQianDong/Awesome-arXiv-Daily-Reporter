# UI-TARS: Pioneering Automated GUI Interaction with Native Agents 

**Title (ZH)**: UI-TARS：首创使用本机代理进行自动化GUI交互 

**Authors**: Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12326)  

**Abstract**: This paper introduces UI-TARS, a native GUI agent model that solely perceives the screenshots as input and performs human-like interactions (e.g., keyboard and mouse operations). Unlike prevailing agent frameworks that depend on heavily wrapped commercial models (e.g., GPT-4o) with expert-crafted prompts and workflows, UI-TARS is an end-to-end model that outperforms these sophisticated frameworks. Experiments demonstrate its superior performance: UI-TARS achieves SOTA performance in 10+ GUI agent benchmarks evaluating perception, grounding, and GUI task execution. Notably, in the OSWorld benchmark, UI-TARS achieves scores of 24.6 with 50 steps and 22.7 with 15 steps, outperforming Claude (22.0 and 14.9 respectively). In AndroidWorld, UI-TARS achieves 46.6, surpassing GPT-4o (34.5). UI-TARS incorporates several key innovations: (1) Enhanced Perception: leveraging a large-scale dataset of GUI screenshots for context-aware understanding of UI elements and precise captioning; (2) Unified Action Modeling, which standardizes actions into a unified space across platforms and achieves precise grounding and interaction through large-scale action traces; (3) System-2 Reasoning, which incorporates deliberate reasoning into multi-step decision making, involving multiple reasoning patterns such as task decomposition, reflection thinking, milestone recognition, etc. (4) Iterative Training with Reflective Online Traces, which addresses the data bottleneck by automatically collecting, filtering, and reflectively refining new interaction traces on hundreds of virtual machines. Through iterative training and reflection tuning, UI-TARS continuously learns from its mistakes and adapts to unforeseen situations with minimal human intervention. We also analyze the evolution path of GUI agents to guide the further development of this domain. 

**Abstract (ZH)**: 本文介绍了UI-TARS，这是一种原生的GUI代理模型，仅通过感知截屏作为输入，并执行类似人类的操作（如键盘和鼠标操作）。与依赖于高度封装的商业模型（例如GPT-4o）并需采用专家设计的提示和工作流的现有代理框架不同，UI-TARS是一个端到端的模型，性能超越了这些复杂的框架。实验结果表明其优越的性能：UI-TARS在10多个评价感知、定位和GUI任务执行的代理基准测试中均取得了SOTA（最佳）表现。特别地，在OSWorld基准测试中，UI-TARS在50步情况下的得分为24.6，15步情况下的得分为22.7，超越了Claude（分别为22.0和14.9）。在AndroidWorld中，UI-TARS的得分为46.6，超越了GPT-4o（34.5）。UI-TARS集成了多项关键创新：(1) 强化感知：利用大规模的GUI截屏数据集，实现基于上下文理解UI元素和精确描述；(2) 统一动作建模，标准化了跨平台的动作，通过大规模的动作痕迹实现精准的定位和交互；(3) 系统2推理，将有意识的推理融入多步骤决策中，包括任务分解、反思思考、里程碑识别等多种推理模式；(4) 循环训练与反思调整，通过自动收集、筛选和反思性优化新的交互痕迹，在数百个虚拟机上连续学习和适应，减少了人类干预。我们还分析了GUI代理的发展路径，以指导这一领域未来的进一步发展。 

---
# Bridging Visualization and Optimization: Multimodal Large Language Models on Graph-Structured Combinatorial Optimization 

**Title (ZH)**: 跨模态连接与优化：基于图结构组合优化的多模态大型语言模型 

**Authors**: Jie Zhao, Kang Hao Cheong, Witold Pedrycz  

**Link**: [PDF](https://arxiv.org/pdf/2501.11968)  

**Abstract**: Graph-structured combinatorial challenges are inherently difficult due to their nonlinear and intricate nature, often rendering traditional computational methods ineffective or expensive. However, these challenges can be more naturally tackled by humans through visual representations that harness our innate ability for spatial reasoning. In this study, we propose transforming graphs into images to preserve their higher-order structural features accurately, revolutionizing the representation used in solving graph-structured combinatorial tasks. This approach allows machines to emulate human-like processing in addressing complex combinatorial challenges. By combining the innovative paradigm powered by multimodal large language models (MLLMs) with simple search techniques, we aim to develop a novel and effective framework for tackling such problems. Our investigation into MLLMs spanned a variety of graph-based tasks, from combinatorial problems like influence maximization to sequential decision-making in network dismantling, as well as addressing six fundamental graph-related issues. Our findings demonstrate that MLLMs exhibit exceptional spatial intelligence and a distinctive capability for handling these problems, significantly advancing the potential for machines to comprehend and analyze graph-structured data with a depth and intuition akin to human cognition. These results also imply that integrating MLLMs with simple optimization strategies could form a novel and efficient approach for navigating graph-structured combinatorial challenges without complex derivations, computationally demanding training and fine-tuning. 

**Abstract (ZH)**: 由于其非线性和错综复杂的特点，基于图的组合挑战本质上是困难的，常常使传统的计算方法变得无效或成本高昂。然而，这些挑战可以通过视觉表示更容易地由人类解决，利用我们天生的空间推理能力。在这项研究中，我们提出将图转换为图像以准确保留其高阶结构特征，从而彻底变革解决图结构组合任务的表示方式。这种方法使得机器能够在解决复杂组合挑战时模拟人类的处理方式。通过结合由多模态大语言模型（MLLMs）驱动的新颖范式与简单的搜索技术，我们旨在开发一个新颖且有效的框架来解决此类问题。我们对MLLMs的研究涵盖了多种基于图的任务，从组合问题如影响力最大化到网络拆解中的顺序决策，以及解决六项基本的图相关问题。我们的研究结果表明，MLLMs展示了卓越的空间智能和独特的处理这些任务的能力，这大大提高了机器理解并分析图结构数据的潜力，使其具备与人类认知相似的深度和直觉。这些结果还暗示，将MLLMs与简单的优化策略相结合，可能形成一种无需复杂推导、计算资源消耗低且高效的框架，以应对图结构组合挑战。 

---
# Make Full Use of Testing Information: An Integrated Accelerated Testing and Evaluation Method for Autonomous Driving Systems 

**Title (ZH)**: 充分利用测试信息：自动驾驶系统加速测试与评估一体化方法 

**Authors**: Xinzheng Wu, Junyi Chen, Jianfeng Wu, Longgao Zhang, Tian Xia, Yong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2501.11924)  

**Abstract**: Testing and evaluation is an important step before the large-scale application of the autonomous driving systems (ADSs). Based on the three level of scenario abstraction theory, a testing can be performed within a logical scenario, followed by an evaluation stage which is inputted with the testing results of each concrete scenario generated from the logical parameter space. During the above process, abundant testing information is produced which is beneficial for comprehensive and accurate evaluations. To make full use of testing information, this paper proposes an Integrated accelerated Testing and Evaluation Method (ITEM). Based on a Monte Carlo Tree Search (MCTS) paradigm and a dual surrogates testing framework proposed in our previous work, this paper applies the intermediate information (i.e., the tree structure, including the affiliation of each historical sampled point with the subspaces and the parent-child relationship between subspaces) generated during the testing stage into the evaluation stage to achieve accurate hazardous domain identification. Moreover, to better serve this purpose, the UCB calculation method is improved to allow the search algorithm to focus more on the hazardous domain boundaries. Further, a stopping condition is constructed based on the convergence of the search algorithm. Ablation and comparative experiments are then conducted to verify the effectiveness of the improvements and the superiority of the proposed method. The experimental results show that ITEM could well identify the hazardous domains in both low- and high-dimensional cases, regardless of the shape of the hazardous domains, indicating its generality and potential for the safety evaluation of ADSs. 

**Abstract (ZH)**: 自动驾系统（ADS）的大规模应用之前，测试与评估是一个重要的步骤。基于三层次场景抽象理论，测试可以在逻辑场景内进行，随后是一个评估阶段，该阶段利用从逻辑参数空间生成的具体场景的测试结果作为输入。在上述过程中，会产生大量的测试信息，有助于进行全面和准确的评估。为了充分利用测试信息，本文提出了一种综合加速测试与评估方法（ITEM）。该方法基于我们之前工作中提出的蒙特卡洛树搜索（MCTS）范式和双代理测试框架，利用测试阶段生成的中间信息（即树结构，包括每个历史采样点与子空间的隶属关系以及子空间的父子关系）应用于评估阶段，以实现对潜在危险区域的准确识别。此外，为了更好地服务于这一目的，改进了UCB计算方法，使搜索算法更倾向于关注危险区域的边界。进一步地，基于搜索算法的收敛性构建了一个停止条件。然后进行了消融实验和比较实验以验证改进的有效性和所提方法的优越性。实验结果显示，无论危险区域的维度高低或形状如何，ITEM都能够很好地识别潜在危险区域，表明其通用性和在ADS安全评估中的潜在应用价值。 

---
# Bridging the Communication Gap: Evaluating AI Labeling Practices for Trustworthy AI Development 

**Title (ZH)**: 弥合沟通差距：评估AI标签实践以促进可信AI的发展 

**Authors**: Raphael Fischer, Magdalena Wischnewski, Alexander van der Staay, Katharina Poitz, Christian Janiesch, Thomas Liebig  

**Link**: [PDF](https://arxiv.org/pdf/2501.11909)  

**Abstract**: As artificial intelligence (AI) becomes integral to economy and society, communication gaps between developers, users, and stakeholders hinder trust and informed decision-making. High-level AI labels, inspired by frameworks like EU energy labels, have been proposed to make the properties of AI models more transparent. Without requiring deep technical expertise, they can inform on the trade-off between predictive performance and resource efficiency. However, the practical benefits and limitations of AI labeling remain underexplored. This study evaluates AI labeling through qualitative interviews along four key research questions. Based on thematic analysis and inductive coding, we found a broad range of practitioners to be interested in AI labeling (RQ1). They see benefits for alleviating communication gaps and aiding non-expert decision-makers, however limitations, misunderstandings, and suggestions for improvement were also discussed (RQ2). Compared to other reporting formats, interviewees positively evaluated the reduced complexity of labels, increasing overall comprehensibility (RQ3). Trust was influenced most by usability and the credibility of the responsible labeling authority, with mixed preferences for self-certification versus third-party certification (RQ4). Our Insights highlight that AI labels pose a trade-off between simplicity and complexity, which could be resolved by developing customizable and interactive labeling frameworks to address diverse user needs. Transparent labeling of resource efficiency also nudged interviewee priorities towards paying more attention to sustainability aspects during AI development. This study validates AI labels as a valuable tool for enhancing trust and communication in AI, offering actionable guidelines for their refinement and standardization. 

**Abstract (ZH)**: 随着人工智能（AI）在经济和社会中发挥着越来越重要的作用，开发者、用户和利益相关者之间的沟通鸿沟阻碍了信任和基于信息的决策。受欧盟能源标签框架等启发，人们提出了高级AI标签，旨在使AI模型的属性更加透明。这些标签不需要深厚的技术专长即可告知用户在预测性能和资源效率之间的权衡。然而，AI标签的实际益处和局限性仍需进一步探索。本研究通过定性访谈针对四个关键研究问题评估了AI标签。基于主题分析和归因编码，我们发现广泛的从业者对AI标签感兴趣（RQ1）。他们认为，AI标签有助于缓解沟通鸿沟，并辅助非专家决策者，但也讨论了其局限性、误解和改进建议（RQ2）。与其它报告格式相比，受访者对标签简化后的复杂性减少给予了正面评价，提高了整体可理解性（RQ3）。信任主要由可用性和负责标签的权威机构的信誉所影响，受访者关于自我认证和第三方认证存在混合偏好（RQ4）。我们的洞察表明，AI标签在简单性和复杂性之间存在权衡，可以通过开发定制化和互动式的标签框架来解决这一问题，以满足不同用户的需求。透明的资源效率标签也促使受访者更加关注AI开发中的可持续性方面。本研究验证了AI标签作为增强AI中信任和沟通工具的价值，并为它们的改进和标准化提供了实际行动指南。 

---
# Systematic Abductive Reasoning via Diverse Relation Representations in Vector-symbolic Architecture 

**Title (ZH)**: 基于向量-符号架构中多样化关系表示的系统 abduction推理 

**Authors**: Zhong-Hua Sun, Ru-Yuan Zhang, Zonglei Zhen, Da-Hui Wang, Yong-Jie Li, Xiaohong Wan, Hongzhi You  

**Link**: [PDF](https://arxiv.org/pdf/2501.11896)  

**Abstract**: In abstract visual reasoning, monolithic deep learning models suffer from limited interpretability and generalization, while existing neuro-symbolic approaches fall short in capturing the diversity and systematicity of attributes and relation representations. To address these challenges, we propose a Systematic Abductive Reasoning model with diverse relation representations (Rel-SAR) in Vector-symbolic Architecture (VSA) to solve Raven's Progressive Matrices (RPM). To derive attribute representations with symbolic reasoning potential, we introduce not only various types of atomic vectors that represent numeric, periodic and logical semantics, but also the structured high-dimentional representation (SHDR) for the overall Grid component. For systematic reasoning, we propose novel numerical and logical relation functions and perform rule abduction and execution in a unified framework that integrates these relation representations. Experimental results demonstrate that Rel-SAR achieves significant improvement on RPM tasks and exhibits robust out-of-distribution generalization. Rel-SAR leverages the synergy between HD attribute representations and symbolic reasoning to achieve systematic abductive reasoning with both interpretable and computable semantics. 

**Abstract (ZH)**: 在抽象视觉推理中，单一的深度学习模型面临着可解释性和泛化能力有限的问题，而现有的神经-符号方法在捕捉属性和关系表示的多样性和系统性方面表现不足。为了解决这些问题，我们提出了一种基于向量-符号架构（VSA）的系统性演绎推理模型（Rel-SAR），用于解决瑞文渐进矩阵（RPM）任务。为了通过符号推理潜在地提取属性表示，我们不仅引入了表示数值、周期和逻辑语义的多种原子向量，还提出了网格组件整体的结构化高维表示（SHDR）。为了实现系统性推理，我们提出了新颖的数值和逻辑关系函数，并在一个综合框架中执行这些关系表示的规则演绎和执行。实验结果表明，Rel-SAR 在 RPM 任务中取得了显著的改进，并展现出稳健的离分布泛化能力。Rel-SAR 通过结合高维属性表示和符号推理的协同作用，实现了兼具可解释性和计算性的系统性演绎推理。 

---
# Policy-Adaptable Methods For Resolving Normative Conflicts Through Argumentation and Graph Colouring 

**Title (ZH)**: 适应政策的方法：通过论辩和图着色解决规范性冲突 

**Authors**: Johnny Joyce  

**Link**: [PDF](https://arxiv.org/pdf/2501.11799)  

**Abstract**: In a multi-agent system, one may choose to govern the behaviour of an agent by imposing norms, which act as guidelines for how agents should act either all of the time or in given situations. However, imposing multiple norms on one or more agents may result in situations where these norms conflict over how the agent should behave. In any system with normative conflicts (such as safe reinforcement models or systems which monitor safety protocols), one must decide which norms should be followed such that the most important and most relevant norms are maintained. We introduce a new method for resolving normative conflicts through argumentation and graph colouring which is compatible with a variety of normative conflict resolution policies. We prove that this method always creates an admissible set of arguments under argumentation semantics, meaning that it produces coherent outputs. We also introduce more robust variants of this method, each building upon their predecessor to create a superior output, and we include further mathematical proof of their coherence. Our most advanced variant uses the existing concept of curtailment, where one norm may supersede another without fully eliminating it. The methods we introduce are all compatible with various pre-existing policies for resolving normative conflicts. Empirical evaluations are also performed to compare our algorithms to each other and to others in existing literature. 

**Abstract (ZH)**: 在多代理系统中，可以采用施加规范的方法来管理代理的行为，这些规范可以作为指导，告诉代理在所有情况下或特定情况下应该如何行动。然而，对一个或多个代理施加多个规范可能会导致规范之间的冲突，使代理不知道该如何行为。在存在规范冲突的任何系统中（如安全强化学习模型或监测安全协议的系统），必须决定应遵循哪些规范，以确保最重要的和最相关的规范得以维持。我们提出了一种通过论辩和图着色解决规范冲突的新方法，该方法与各种规范冲突解决策略兼容。我们证明，这种方法在论辩语义下总是能够生成一个可接受的论辩集，意味着它能够产生一致的输出。我们还引入了这种方法的更稳健的变体，每个变体在其前一个变体的基础上进一步提升输出的质量，并提供了它们一致性的更多数学证明。我们最先进的变体使用了现有概念中的限制性原则，即一个规范可以在不完全消除另一个规范的情况下凌驾于它。我们提出的方法与各种现成的规范冲突解决策略兼容。我们还进行了实证评估，以比较我们的算法与彼此以及现有文献中的其他算法。 

---
# Episodic memory in AI agents poses risks that should be studied and mitigated 

**Title (ZH)**: AI代理的事件记忆存在风险，应当加以研究和缓解 

**Authors**: Chad DeChant  

**Link**: [PDF](https://arxiv.org/pdf/2501.11739)  

**Abstract**: Most current AI models have little ability to store and later retrieve a record or representation of what they do. In human cognition, episodic memories play an important role in both recall of the past as well as planning for the future. The ability to form and use episodic memories would similarly enable a broad range of improved capabilities in an AI agent that interacts with and takes actions in the world. Researchers have begun directing more attention to developing memory abilities in AI models. It is therefore likely that models with such capability will be become widespread in the near future. This could in some ways contribute to making such AI agents safer by enabling users to better monitor, understand, and control their actions. However, as a new capability with wide applications, we argue that it will also introduce significant new risks that researchers should begin to study and address. We outline these risks and benefits and propose four principles to guide the development of episodic memory capabilities so that these will enhance, rather than undermine, the effort to keep AI safe and trustworthy. 

**Abstract (ZH)**: 当前大多数人工智能模型缺乏存储和后续检索其行为记录或表示的能力。在人类认知中，情景记忆在回忆过去和规划未来方面扮演着重要角色。能够形成和利用情景记忆将在世界中与环境互动并采取行动的人工智能代理的多种能力得到显著提升。研究人员已经开始更多地关注提高人工智能模型的记忆能力。因此，具备此类能力的模型在未来很可能会变得普遍。这在某种程度上可以通过使用户更好地监控、理解和控制其行为来提高人工智能代理的安全性。然而，作为一种具有广泛应用的新能力，我们认为它也将引入显著的新风险，需要研究人员开始研究并加以解决。我们概述了这些风险与利益，并建议四条原则来指导情景记忆能力的发展，以确保这些能力能够增强而非削弱保持人工智能安全和可信赖的努力。 

---
# SR-FoT: A Syllogistic-Reasoning Framework of Thought for Large Language Models Tackling Knowledge-based Reasoning Tasks 

**Title (ZH)**: SR-FoT: 一种应用于大型语言模型的知识推理框架，用于处理基于知识的推理任务 

**Authors**: Wentao Wan, Zhuojie Yang, Yongcan Chen, Chenglin Luo, Ruilin Wang, Kehao Cai, Nan Kang, Liang Lin, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11599)  

**Abstract**: Deductive reasoning is a crucial logical capability that assists us in solving complex problems based on existing knowledge. Although augmented by Chain-of-Thought prompts, Large Language Models (LLMs) might not follow the correct reasoning paths. Enhancing the deductive reasoning abilities of LLMs, and leveraging their extensive built-in knowledge for various reasoning tasks, remains an open question. Attempting to mimic the human deductive reasoning paradigm, we propose a multi-stage Syllogistic-Reasoning Framework of Thought (SR-FoT) that enables LLMs to perform syllogistic deductive reasoning to handle complex knowledge-based reasoning tasks. Our SR-FoT begins by interpreting the question and then uses the interpretation and the original question to propose a suitable major premise. It proceeds by generating and answering minor premise questions in two stages to match the minor premises. Finally, it guides LLMs to use the previously generated major and minor premises to perform syllogistic deductive reasoning to derive the answer to the original question. Extensive and thorough experiments on knowledge-based reasoning tasks have demonstrated the effectiveness and advantages of our SR-FoT. 

**Abstract (ZH)**: 演绎推理是一项关键的逻辑能力，它帮助我们在基于现有知识的基础上解决复杂问题。尽管通过链式思维提示（Chain-of-Thought prompts）可以增强其效果，但大型语言模型（LLMs）可能并不总是遵循正确的推理路径。提高LLMs的演绎推理能力，并利用它们广泛内置的知识来处理各种推理任务，仍是一个开放性问题。为了模仿人类的演绎推理模式，我们提出了一个多阶段的思想形式逻辑推理框架（Syllogistic-Reasoning Framework of Thought，SR-FoT），以使LLMs能够进行形式逻辑演绎推理，从而处理复杂的基于知识的推理任务。SR-FoT首先解释问题，然后利用解释和原始问题提出合适的主前提。接下来，通过两个阶段生成和回答次要前提问题，以匹配次要前提。最后，它指导LLMs使用之前生成的主前提和次要前提进行形式逻辑演绎推理，从而得出原始问题的答案。我们在基于知识的推理任务上的广泛且详尽的实验表明，SR-FoT的有效性和优势。 

---
# The impact of intrinsic rewards on exploration in Reinforcement Learning 

**Title (ZH)**: 内在奖励对强化学习中探索的影响 

**Authors**: Aya Kayal, Eduardo Pignatelli, Laura Toni  

**Link**: [PDF](https://arxiv.org/pdf/2501.11533)  

**Abstract**: One of the open challenges in Reinforcement Learning is the hard exploration problem in sparse reward environments. Various types of intrinsic rewards have been proposed to address this challenge by pushing towards diversity. This diversity might be imposed at different levels, favouring the agent to explore different states, policies or behaviours (State, Policy and Skill level diversity, respectively). However, the impact of diversity on the agent's behaviour remains unclear. In this work, we aim to fill this gap by studying the effect of different levels of diversity imposed by intrinsic rewards on the exploration patterns of RL agents. We select four intrinsic rewards (State Count, Intrinsic Curiosity Module (ICM), Maximum Entropy, and Diversity is all you need (DIAYN)), each pushing for a different diversity level. We conduct an empirical study on MiniGrid environment to compare their impact on exploration considering various metrics related to the agent's exploration, namely: episodic return, observation coverage, agent's position coverage, policy entropy, and timeframes to reach the sparse reward. The main outcome of the study is that State Count leads to the best exploration performance in the case of low-dimensional observations. However, in the case of RGB observations, the performance of State Count is highly degraded mostly due to representation learning challenges. Conversely, Maximum Entropy is less impacted, resulting in a more robust exploration, despite being not always optimal. Lastly, our empirical study revealed that learning diverse skills with DIAYN, often linked to improved robustness and generalisation, does not promote exploration in MiniGrid environments. This is because: i) learning the skill space itself can be challenging, and ii) exploration within the skill space prioritises differentiating between behaviours rather than achieving uniform state visitation. 

**Abstract (ZH)**: 强化学习领域的一个开放式挑战是如何在稀疏奖励环境中进行有效探索。为了解决这一挑战，提出了多种内在奖励来鼓励多样化的行为。这种多样化可能在不同的层面上施加，促使智能体探索不同的状态、策略或行为（分别对应状态层面、策略层面和技能层面的多样性）。然而，多样化对智能体行为的具体影响仍不清楚。本研究旨在通过分析不同种类内在奖励施加的多样化对强化学习智能体探索模式的影响来填补这一空白。我们选择了四种内在奖励（状态计数、内在好奇心模块（ICM）、最大熵、多样性即是你要的一切（DIAYN）），每种奖励都鼓励不同层面的多样性。我们在MiniGrid环境中进行了实证研究，比较了这些内在奖励在不同探索度量上的影响，包括：每集回报、观察空间覆盖范围、智能体位置覆盖范围、策略熵以及达到稀疏奖励所需的时间框架。研究的主要结果表明，在低维度观察的情况下，状态计数（State Count）的探索性能最佳。然而，在RGB观察的情况下，状态计数的表现显著下降，这主要是由于表示学习的挑战。相反，最大熵（maximum entropy）较少受到负面影响，尽管其可能并非总是最优解，表现出更稳健的探索行为。最后，我们的实证研究发现，使用DIAYN学习多种多样的技能（通常与增强的鲁棒性和泛化能力相关）在MiniGrid环境中并不能促进探索。这是因为：i) 学习技能空间本身具有挑战性，ii) 在技能空间内的探索更侧重于区分不同的行为而非实现状态的均匀访问。 

---
# Decomposing Interventional Causality into Synergistic, Redundant, and Unique Components 

**Title (ZH)**: 将干预因果性分解为协同、冗余和独特成分 

**Authors**: Abel Jansma  

**Link**: [PDF](https://arxiv.org/pdf/2501.11447)  

**Abstract**: We introduce a novel framework for decomposing interventional causal effects into synergistic, redundant, and unique components, building on the intuition of Partial Information Decomposition (PID) and the principle of Möbius inversion. While recent work has explored a similar decomposition of an observational measure, we argue that a proper causal decomposition must be interventional in nature. We develop a mathematical approach that systematically quantifies how causal power is distributed among variables in a system, using a recently derived closed-form expression for the Möbius function of the redundancy lattice. The formalism is then illustrated by decomposing the causal power in logic gates, cellular automata, and chemical reaction networks. Our results reveal how the distribution of causal power can be context- and parameter-dependent. This decomposition provides new insights into complex systems by revealing how causal influences are shared and combined among multiple variables, with potential applications ranging from attribution of responsibility in legal or AI systems, to the analysis of biological networks or climate models. 

**Abstract (ZH)**: 我们提出了一种新颖的框架，用于将干预因果效应分解为协同、冗余和独特的组件，这一框架基于部分信息分解（PID）的直觉和莫比乌斯反演原理。尽管近期研究表明可以将观察性度量进行类似的分解，但我们认为恰当的因果分解必须是干预性的。我们开发了一种数学方法，系统地量化了因果力量在系统变量之间的分布，使用了最近推导出的冗余格莫比乌斯函数的闭式表达式。该形式主义随后通过分解逻辑门、细胞自动机和化学反应网络中的因果力量进行了说明。我们的结果揭示了因果力量分布的上下文依赖性和参数依赖性。这一分解为了解复杂系统的因果影响如何在多个变量之间共享和组合提供了新的洞察，潜在的应用范围从法律责任或AI系统的归因，到生物网络或气候模型的分析。 

---
# The Explanation Game -- Rekindled (Extended Version) 

**Title (ZH)**: 《解释游戏——重燃（扩展版本）》 

**Authors**: Joao Marques-Silva, Xuanxiang Huang, Olivier Letoffe  

**Link**: [PDF](https://arxiv.org/pdf/2501.11429)  

**Abstract**: Recent work demonstrated the existence of critical flaws in the current use of Shapley values in explainable AI (XAI), i.e. the so-called SHAP scores. These flaws are significant in that the scores provided to a human decision-maker can be misleading. Although these negative results might appear to indicate that Shapley values ought not be used in XAI, this paper argues otherwise. Concretely, this paper proposes a novel definition of SHAP scores that overcomes existing flaws. Furthermore, the paper outlines a practically efficient solution for the rigorous estimation of the novel SHAP scores. Preliminary experimental results confirm our claims, and further underscore the flaws of the current SHAP scores. 

**Abstract (ZH)**: 最近的工作揭示了当前在解释性人工智能（XAI）中使用Shapley值中存在的关键缺陷，即所谓的SHAP分数。这些缺陷非常重要，因为提供的分数可能会误导人类决策者。尽管这些负面结果可能看似表明Shapley值不应在XAI中使用，但本文提出了不同的观点。具体而言，本文提出了一种新的SHAP分数定义，以克服现有缺陷。此外，本文还概述了一种高效的实际解决方案，用于严格估计新型SHAP分数。初步实验结果证实了我们的论点，并进一步强调了当前SHAP分数的缺陷。 

---
# Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training 

**Title (ZH)**: Agent-R：通过迭代自我训练来进行反思的语言模型代理 

**Authors**: Siyu Yuan, Zehui Chen, Zhiheng Xi, Junjie Ye, Zhengyin Du, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.11425)  

**Abstract**: Large Language Models (LLMs) agents are increasingly pivotal for addressing complex tasks in interactive environments. Existing work mainly focuses on enhancing performance through behavior cloning from stronger experts, yet such approaches often falter in real-world applications, mainly due to the inability to recover from errors. However, step-level critique data is difficult and expensive to collect. Automating and dynamically constructing self-critique datasets is thus crucial to empowering models with intelligent agent capabilities. In this work, we propose an iterative self-training framework, Agent-R, that enables language Agent to Reflect on the fly. Unlike traditional methods that reward or penalize actions based on correctness, Agent-R leverages MCTS to construct training data that recover correct trajectories from erroneous ones. A key challenge of agent reflection lies in the necessity for timely revision rather than waiting until the end of a rollout. To address this, we introduce a model-guided critique construction mechanism: the actor model identifies the first error step (within its current capability) in a failed trajectory. Starting from it, we splice it with the adjacent correct path, which shares the same parent node in the tree. This strategy enables the model to learn reflection based on its current policy, therefore yielding better learning efficiency. To further explore the scalability of this self-improvement paradigm, we investigate iterative refinement of both error correction capabilities and dataset construction. Our findings demonstrate that Agent-R continuously improves the model's ability to recover from errors and enables timely error correction. Experiments on three interactive environments show that Agent-R effectively equips agents to correct erroneous actions while avoiding loops, achieving superior performance compared to baseline methods (+5.59%). 

**Abstract (ZH)**: 大型语言模型（LLMs）代理在处理交互环境中的复杂任务方面变得越来越关键。现有工作主要集中在通过从更强的专家那里进行行为克隆来提升性能，但这种做法在实际应用中往往会因为无法从错误中恢复而失效。然而，步骤级的批判数据收集起来既困难又昂贵。因此，自动化和动态构建自我批判数据集对于赋予模型智能代理能力至关重要。在本工作中，我们提出了一种迭代自我训练框架Agent-R，使得语言代理能够实时反思。与传统的基于正确性奖励或惩罚动作的方法不同，Agent-R 利用 Monte Carlo 树搜索（MCTS）来构建能够从错误轨迹中恢复正确轨迹的训练数据。代理反思的一个关键挑战是对及时修订的需求，而不仅仅是在一场展开（rollout）结束时才修订。为了解决这一问题，我们引入了一种基于模型的批判构建机制：行为模型识别失败轨迹中它目前能力范围内的第一个错误步骤。从这个步骤开始，将它与具有相同父节点的相邻正确路径拼接起来。这种策略使模型能够在当前策略的基础上学习反思，从而提高学习效率。为了进一步探索这种自我改进范式的可扩展性，我们探讨了错误纠正能力和数据集构建的迭代细化。我们的研究结果表明，Agent-R 不断提高模型从错误中恢复的能力，并实现及时的错误修正。在三个交互环境中进行的实验表明，与基线方法相比，Agent-R 有效地使代理能够纠正错误行为，同时避免循环，从而实现了更优的性能（+5.59%）。 

---
# Reasoning Language Models: A Blueprint 

**Title (ZH)**: 推理语言模型：一个架构设计指南 

**Authors**: Maciej Besta, Julia Barth, Eric Schreiber, Ales Kubicek, Afonso Catarino, Robert Gerstenberger, Piotr Nyczyk, Patrick Iff, Yueling Li, Sam Houliston, Tomasz Sternal, Marcin Copik, Grzegorz Kwaśniewski, Jürgen Müller, Łukasz Flis, Hannes Eberhard, Hubert Niewiadomski, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2501.11223)  

**Abstract**: Reasoning language models (RLMs), also known as Large Reasoning Models (LRMs), such as OpenAI's o1 and o3, DeepSeek-V3, and Alibaba's QwQ, have redefined AI's problem-solving capabilities by extending large language models (LLMs) with advanced reasoning mechanisms. Yet, their high costs, proprietary nature, and complex architectures - uniquely combining Reinforcement Learning (RL), search heuristics, and LLMs - present accessibility and scalability challenges. To address these, we propose a comprehensive blueprint that organizes RLM components into a modular framework, based on a survey and analysis of all RLM works. This blueprint incorporates diverse reasoning structures (chains, trees, graphs, and nested forms), reasoning strategies (e.g., Monte Carlo Tree Search, Beam Search), RL concepts (policy, value models and others), and supervision schemes (Output-Based and Process-Based Supervision). We also provide detailed mathematical formulations and algorithmic specifications to simplify RLM implementation. By showing how schemes like LLaMA-Berry, QwQ, Journey Learning, and Graph of Thoughts fit as special cases, we demonstrate the blueprint's versatility and unifying potential. To illustrate its utility, we introduce x1, a modular implementation for rapid RLM prototyping and experimentation. Using x1 and a literature review, we provide key insights, such as multi-phase training for policy and value models, and the importance of familiar training distributions. Finally, we outline how RLMs can integrate with a broader LLM ecosystem, including tools and databases. Our work demystifies RLM construction, democratizes advanced reasoning capabilities, and fosters innovation, aiming to mitigate the gap between "rich AI" and "poor AI" by lowering barriers to RLM development and experimentation. 

**Abstract (ZH)**: 基于推理的语言模型（RLMs），也称为大型推理模型（LRMs），如OpenAI的o1和o3、DeepSeek-V3以及阿里巴巴的QwQ，通过将先进的推理机制扩展到大型语言模型（LLMs）中，重新定义了AI的问题解决能力。然而，它们的存在带来的高成本、专有性质以及复杂的结构——该结构结合了强化学习（RL）、搜索启发式方法和LLMs——也带来了可及性和扩展性方面的挑战。为了应对这些挑战，我们提出了一个全面的蓝图，该蓝图基于对所有RLM工作的调研和分析，将RLM组件组织成一个模块化框架。该蓝图整合了多样的推理结构（链式、树状结构、图形以及嵌套形式）、推理策略（例如蒙特卡洛树搜索、束搜索）、强化学习概念（策略模型、价值模型等）以及监督方案（基于输出和基于过程的监督）。我们还提供了详细的数学公式和算法规范，以简化RLM的实现。通过展示LLaMA-Berry、QwQ、Journey Learning和Graph of Thoughts等方案如何作为特殊案例融入其中，我们证明了该蓝图的灵活性和统一性。为了展示其用途，我们介绍了x1，一个用于快速原型设计和实验的模块化实现。使用x1和文献综述，我们提供了关键见解，如分阶段训练策略和价值模型以及熟悉训练分布的重要性。最后，我们概述了RLM如何整合到更广泛的LLM生态系统中，包括工具和数据库。我们的工作使RLM的构建更加透明，促进了高级推理能力的普及，促进了创新，旨在通过降低RLM开发和实验的门槛来缩小“富AI”和“贫AI”之间的差距。 

---
# Fine-Grained Appropriate Reliance: Human-AI Collaboration with a Multi-Step Transparent Decision Workflow for Complex Task Decomposition 

**Title (ZH)**: 细粒度适当的依赖：一种用于复杂任务分解的多步透明决策工作流的人机协作 

**Authors**: Gaole He, Patrick Hemmer, Michael Vössing, Max Schemmer, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2501.10909)  

**Abstract**: In recent years, the rapid development of AI systems has brought about the benefits of intelligent services but also concerns about security and reliability. By fostering appropriate user reliance on an AI system, both complementary team performance and reduced human workload can be achieved. Previous empirical studies have extensively analyzed the impact of factors ranging from task, system, and human behavior on user trust and appropriate reliance in the context of one-step decision making. However, user reliance on AI systems in tasks with complex semantics that require multi-step workflows remains under-explored. Inspired by recent work on task decomposition with large language models, we propose to investigate the impact of a novel Multi-Step Transparent (MST) decision workflow on user reliance behaviors. We conducted an empirical study (N = 233) of AI-assisted decision making in composite fact-checking tasks (i.e., fact-checking tasks that entail multiple sub-fact verification steps). Our findings demonstrate that human-AI collaboration with an MST decision workflow can outperform one-step collaboration in specific contexts (e.g., when advice from an AI system is misleading). Further analysis of the appropriate reliance at fine-grained levels indicates that an MST decision workflow can be effective when users demonstrate a relatively high consideration of the intermediate steps. Our work highlights that there is no one-size-fits-all decision workflow that can help obtain optimal human-AI collaboration. Our insights help deepen the understanding of the role of decision workflows in facilitating appropriate reliance. We synthesize important implications for designing effective means to facilitate appropriate reliance on AI systems in composite tasks, positioning opportunities for the human-centered AI and broader HCI communities. 

**Abstract (ZH)**: 近年来，人工智能系统的迅猛发展带来了智能化服务的便利，同时也引发了关于安全性和可靠性的担忧。通过培养用户对人工智能系统的适当依赖，可以实现互补的工作团队性能并减轻人类的工作负担。以往的经验研究表明，从任务、系统和人类行为等多个方面分析了因素对用户信任和适当依赖的影响，特别是在单步骤决策情境下的影响得到了广泛的研究。然而，在涉及复杂语义和多步骤工作流程的任务中，用户对人工智能系统的依赖仍是一个未充分探索的领域。受大型语言模型在任务分解方面近期工作的启发，我们提出研究具有新颖的多步骤透明（MST）决策工作流程对用户依赖行为的影响。我们对人工智能辅助决策在复合事实核查任务（即包含多个子事实验证步骤的任务）中进行了实证研究（N=233）。研究结果表明，MST决策工作流程可以与人类协作在特定情境下表现更优（例如，当人工智能系统的建议误导时）。进一步对细粒度的适当依赖分析表明，当用户对中间步骤表现出相对较高的考量时，MST决策工作流程可以有效发挥作用。我们的研究强调，不存在一种适用于所有情况的决策工作流程，可以帮助实现最佳的人机协作。我们的洞察有助于加深对决策工作流程在促进适当依赖方面作用的理解。为设计有效的方法以促进人在复合任务中对人工智能系统的适当依赖，我们总结了重要的启示，并为以用户为中心的人工智能和更广泛的HCI社区提供了机遇。 

---
# Classical and Deep Reinforcement Learning Inventory Control Policies for Pharmaceutical Supply Chains with Perishability and Non-Stationarity 

**Title (ZH)**: 经典与深度强化学习库存控制策略在易腐性和非稳态性制药供应链中的应用 

**Authors**: Francesco Stranieri, Chaaben Kouki, Willem van Jaarsveld, Fabio Stella  

**Link**: [PDF](https://arxiv.org/pdf/2501.10895)  

**Abstract**: We study inventory control policies for pharmaceutical supply chains, addressing challenges such as perishability, yield uncertainty, and non-stationary demand, combined with batching constraints, lead times, and lost sales. Collaborating with Bristol-Myers Squibb (BMS), we develop a realistic case study incorporating these factors and benchmark three policies--order-up-to (OUT), projected inventory level (PIL), and deep reinforcement learning (DRL) using the proximal policy optimization (PPO) algorithm--against a BMS baseline based on human expertise. We derive and validate bounds-based procedures for optimizing OUT and PIL policy parameters and propose a methodology for estimating projected inventory levels, which are also integrated into the DRL policy with demand forecasts to improve decision-making under non-stationarity. Compared to a human-driven policy, which avoids lost sales through higher holding costs, all three implemented policies achieve lower average costs but exhibit greater cost variability. While PIL demonstrates robust and consistent performance, OUT struggles under high lost sales costs, and PPO excels in complex and variable scenarios but requires significant computational effort. The findings suggest that while DRL shows potential, it does not outperform classical policies in all numerical experiments, highlighting 1) the need to integrate diverse policies to manage pharmaceutical challenges effectively, based on the current state-of-the-art, and 2) that practical problems in this domain seem to lack a single policy class that yields universally acceptable performance. 

**Abstract (ZH)**: 我们将研究集中在制药供应链的库存控制策略上，以应对诸如易腐性、产出不确定性、非平稳需求、批量约束、交货时间和缺货等挑战。与百时美施贵宝（BMS）合作，我们开发了一个包含这些因素的现实案例研究，并使用递归策略优化算法（PPO）分别基于order-up-to（OUT）、预测库存水平（PIL）和深度强化学习（DRL）来与基于人类经验的BMS基准政策进行比较。我们推导并验证了优化OUT和PIL策略参数的边界程序，并提出了一种估计预测库存水平的方法，这些方法也与需求预测集成到DRL策略中，以提高在非平稳性下的决策制定能力。与基于人工驱动的策略相比，虽然所有实现的策略都能实现较低的平均成本，但它们的成本波动性更大。尽管PIL在性能上表现出稳定性和稳健性，但OUT在高缺货成本情况下表现较差，而PPO在复杂和多变的情景中表现出色，但需要大量的计算资源。研究结果表明，虽然DRL有潜力，但在所有数值实验中并未表现出色，这突出显示了两点：1）为了有效管理制药领域的挑战，需要整合各种策略；2）在这一领域中，实际问题似乎没有一种策略类别能够提供普遍接受的性能。 

---
# ML-SceGen: A Multi-level Scenario Generation Framework 

**Title (ZH)**: ML-SceGen：一个多层级情景生成框架 

**Authors**: Yicheng Xiao, Yangyang Sun, Yicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.10782)  

**Abstract**: Current scientific research witnesses various attempts at applying Large Language Models for scenario generation but is inclined only to comprehensive or dangerous scenarios. In this paper, we seek to build a three-stage framework that not only lets users regain controllability over the generated scenarios but also generates comprehensive scenarios containing danger factors in uncontrolled intersection settings. In the first stage, LLM agents will contribute to translating the key components of the description of the expected scenarios into Functional Scenarios. For the second stage, we use Answer Set Programming (ASP) solver Clingo to help us generate comprehensive logical traffic within intersections. During the last stage, we use LLM to update relevant parameters to increase the critical level of the concrete scenario. 

**Abstract (ZH)**: 当前的科学研究见证了大量将大型语言模型（Large Language Models, LLMs）应用于场景生成的努力，但这些努力主要侧重于全面或危险的场景。在本文中，我们旨在构建一个三阶段框架，不仅让用户重新获得对生成场景的控制权，还能在不受控制的交叉口环境中生成包含危险因素的全面场景。在第一阶段，LLM代理将协助将预期场景的要点转化为功能场景。在第二阶段，我们将使用ASP求解器Clingo来生成交叉口内全面的逻辑交通。在最后一阶段，我们使用LLM来更新相关参数，以提高具体场景的临界水平。 

---
# MAPS: Advancing Multi-Modal Reasoning in Expert-Level Physical Science 

**Title (ZH)**: MAPS：推进专家级物理科学中的多模态推理 

**Authors**: Erle Zhu, Yadi Liu, Zhe Zhang, Xujun Li, Jin Zhou, Xinjie Yu, Minlie Huang, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10768)  

**Abstract**: Pre-trained on extensive text and image corpora, current Multi-Modal Large Language Models (MLLM) have shown strong capabilities in general visual reasoning tasks. However, their performance is still lacking in physical domains that require understanding diagrams with complex physical structures and quantitative analysis based on multi-modal information. To address this, we develop a new framework, named Multi-Modal Scientific Reasoning with Physics Perception and Simulation (MAPS) based on an MLLM. MAPS decomposes expert-level multi-modal reasoning task into physical diagram understanding via a Physical Perception Model (PPM) and reasoning with physical knowledge via a simulator. The PPM module is obtained by fine-tuning a visual language model using carefully designed synthetic data with paired physical diagrams and corresponding simulation language descriptions. At the inference stage, MAPS integrates the simulation language description of the input diagram provided by PPM and results obtained through a Chain-of-Simulation process with MLLM to derive the underlying rationale and the final answer. Validated using our collected college-level circuit analysis problems, MAPS significantly improves reasoning accuracy of MLLM and outperforms all existing models. The results confirm MAPS offers a promising direction for enhancing multi-modal scientific reasoning ability of MLLMs. We will release our code, model and dataset used for our experiments upon publishing of this paper. 

**Abstract (ZH)**: 基于广泛的文本和图像语料库进行预训练，当前的多模态大型语言模型（Multimodal Large Language Models, MLLM）展示了在通用视觉推理任务中的强大能力。然而，它们在需要理解和分析复杂物理结构的图表以及基于多模态信息的定量分析的实际领域中的性能仍然不足。为了解决这个问题，我们基于MLLM开发了一个新的框架，名为多模态科学推理与物理感知和模拟（Multi-Modal Scientific Reasoning with Physics Perception and Simulation, MAPS）。MAPS将专家级别的多模态推理任务分解为通过物理感知模型（Physical Perception Model, PPM）理解物理图表，并通过模拟器利用物理知识进行推理。PPM模块通过使用精心设计的合成数据（配对了物理图表和相应的模拟语言描述）对视觉语言模型进行微调而获得。在推理阶段，MAPS将PPM提供的输入图表的模拟语言描述与通过链式模拟过程与MLLM获得的结果整合起来，以推导出潜在理由和最终答案。通过我们收集的大学级别电路分析问题进行验证，MAPS显著提高了MLLM的推理准确性并优于所有现有模型。结果证实，MAPS为增强MLLM的多模态科学推理能力提供了有前景的方向。我们将在本文发表后发布我们实验中使用的代码、模型和数据集。 

---
# Distributionally Robust Policy Evaluation and Learning for Continuous Treatment with Observational Data 

**Title (ZH)**: 基于观测数据的连续治疗分布鲁棒性策略评估与学习 

**Authors**: Cheuk Hang Leung, Yiyan Huang, Yijun Li, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10693)  

**Abstract**: Using offline observational data for policy evaluation and learning allows decision-makers to evaluate and learn a policy that connects characteristics and interventions. Most existing literature has focused on either discrete treatment spaces or assumed no difference in the distributions between the policy-learning and policy-deployed environments. These restrict applications in many real-world scenarios where distribution shifts are present with continuous treatment. To overcome these challenges, this paper focuses on developing a distributionally robust policy under a continuous treatment setting. The proposed distributionally robust estimators are established using the Inverse Probability Weighting (IPW) method extended from the discrete one for policy evaluation and learning under continuous treatments. Specifically, we introduce a kernel function into the proposed IPW estimator to mitigate the exclusion of observations that can occur in the standard IPW method to continuous treatments. We then provide finite-sample analysis that guarantees the convergence of the proposed distributionally robust policy evaluation and learning estimators. The comprehensive experiments further verify the effectiveness of our approach when distribution shifts are present. 

**Abstract (ZH)**: 本文探讨了利用离线观察数据进行政策评估和学习，这使决策者能够评估和学习一种连接特征和干预措施的政策。现有文献大多集中于离散处理空间或者假设政策学习环境与政策实施环境之间的分布没有差异，这在实际应用中限制了许多场景，尤其是在存在连续处理分布转移的情况下。为克服这些挑战，本文致力于在连续处理设置下开发一种稳健的分布策略。本文提出的分布稳健估计器基于扩展自离散处理情况下的逆概率加权（IPW）方法，用于连续处理下的政策评估和学习。具体而言，本文引入核函数到提出的IPW估计器中，以减轻标准IPW方法在连续处理中可能出现的观测值排除问题。我们还提供了有限样本分析，以保证所提出的分布稳健的政策评估和学习估计器的收敛性。全面的实验进一步验证了在分布转移存在时，本方法的有效性。 

---
# ColorGrid: A Multi-Agent Non-Stationary Environment for Goal Inference and Assistance 

**Title (ZH)**: ColorGrid：一个多 agent 非稳态环境，用于目标推断与辅助 

**Authors**: Andrey Risukhin, Kavel Rao, Ben Caffee, Alan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2501.10593)  

**Abstract**: Autonomous agents' interactions with humans are increasingly focused on adapting to their changing preferences in order to improve assistance in real-world tasks. Effective agents must learn to accurately infer human goals, which are often hidden, to collaborate well. However, existing Multi-Agent Reinforcement Learning (MARL) environments lack the necessary attributes required to rigorously evaluate these agents' learning capabilities. To this end, we introduce ColorGrid, a novel MARL environment with customizable non-stationarity, asymmetry, and reward structure. We investigate the performance of Independent Proximal Policy Optimization (IPPO), a state-of-the-art (SOTA) MARL algorithm, in ColorGrid and find through extensive ablations that, particularly with simultaneous non-stationary and asymmetric goals between a ``leader'' agent representing a human and a ``follower'' assistant agent, ColorGrid is unsolved by IPPO. To support benchmarking future MARL algorithms, we release our environment code, model checkpoints, and trajectory visualizations at this https URL. 

**Abstract (ZH)**: 自主代理与人类的交互越来越多地集中在适应人类不断变化的偏好上，以提高在实际任务中的辅助效果。有效的代理必须学会准确地推断出往往被隐藏的人类目标，从而更好地协作。然而，现有的多智能体强化学习（MARL）环境缺乏评估这些代理学习能力所需的关键属性。为此，我们引入了ColorGrid，这是一种具有可定制非稳定性和不对称性的新型MARL环境，并且具有可定制的奖励结构。我们研究了当前最先进的MARL算法——独立接近策略优化（IPPO）在ColorGrid中的性能，并通过广泛的消融实验发现，特别是在“领导者”代理代表人类和“跟随者”助手代理之间的同时非稳定性和不对称性目标下，IPPO无法解决ColorGrid环境。为了支持未来MARL算法的基准测试，我们在以下链接中开放我们的环境代码、模型检查点和轨迹可视化：[提供链接处] 

---
# Revisiting Rogers' Paradox in the Context of Human-AI Interaction 

**Title (ZH)**: 在人机交互背景下重新审视罗杰斯的悖论 

**Authors**: Katherine M. Collins, Umang Bhatt, Ilia Sucholutsky  

**Link**: [PDF](https://arxiv.org/pdf/2501.10476)  

**Abstract**: Humans learn about the world, and how to act in the world, in many ways: from individually conducting experiments to observing and reproducing others' behavior. Different learning strategies come with different costs and likelihoods of successfully learning more about the world. The choice that any one individual makes of how to learn can have an impact on the collective understanding of a whole population if people learn from each other. Alan Rogers developed simulations of a population of agents to study these network phenomena where agents could individually or socially learn amidst a dynamic, uncertain world and uncovered a confusing result: the availability of cheap social learning yielded no benefit to population fitness over individual learning. This paradox spawned decades of work trying to understand and uncover factors that foster the relative benefit of social learning that centuries of human behavior suggest exists. What happens in such network models now that humans can socially learn from AI systems that are themselves socially learning from us? We revisit Rogers' Paradox in the context of human-AI interaction to probe a simplified network of humans and AI systems learning together about an uncertain world. We propose and examine the impact of several learning strategies on the quality of the equilibrium of a society's 'collective world model'. We consider strategies that can be undertaken by various stakeholders involved in a single human-AI interaction: human, AI model builder, and society or regulators around the interaction. We then consider possible negative feedback loops that may arise from humans learning socially from AI: that learning from the AI may impact our own ability to learn about the world. We close with open directions into studying networks of human and AI systems that can be explored in enriched versions of our simulation framework. 

**Abstract (ZH)**: 人类通过多种方式学习世界以及如何在世界中行动：从个人进行实验到观察和模仿他人的行为。不同的学习策略伴随着不同的成本和成功了解世界的概率。任何一个人选择的学习方式都可能对其所在群体的集体理解产生影响，特别是如果人们相互学习的话。Alan Rogers 开发了模拟个体代理在网络中相互学习的仿真模型，并研究了在这种动态和不确定的世界中，个体或社交学习的现象。他在这种模型中发现了令人困惑的结果：廉价的社交学习对群体适应性没有任何益处，相较于个体学习而言并未带来优势。这一矛盾激发了数十年的研究，试图理解和揭示支持人类行为所表现出的社交学习优势的因素。如今，随着人类能够从我们自己也在社交中学习的AI系统中学习，这种情形会发生什么？我们重新审视Rogers的悖论，将其置于人类与AI系统互动的上下文中，探讨人类和AI系统共同学习不确定世界对社会集体认识质量的影响。我们提出了几种不同的学习策略，并探讨它们对社会“集体世界模型”质量的影响。我们考虑不同利益相关者（人类、AI模型构建者和围绕互动的社会或监管者）可能采取的不同策略。接着，我们考虑人类从AI系统中学习所可能产生的负面影响：从AI系统中学习可能会削弱人类自身了解世界的这方面能力。最后，我们提出了更多研究方向，探索在我们仿真框架的丰富版本中可以研究的人类与AI系统网络。 

---
# Learning segmentation from point trajectories 

**Title (ZH)**: 从点轨迹学习分割 

**Authors**: Laurynas Karazija, Iro Laina, Christian Rupprecht, Andrea Vedaldi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12392)  

**Abstract**: We consider the problem of segmenting objects in videos based on their motion and no other forms of supervision. Prior work has often approached this problem by using the principle of common fate, namely the fact that the motion of points that belong to the same object is strongly correlated. However, most authors have only considered instantaneous motion from optical flow. In this work, we present a way to train a segmentation network using long-term point trajectories as a supervisory signal to complement optical flow. The key difficulty is that long-term motion, unlike instantaneous motion, is difficult to model -- any parametric approximation is unlikely to capture complex motion patterns over long periods of time. We instead draw inspiration from subspace clustering approaches, proposing a loss function that seeks to group the trajectories into low-rank matrices where the motion of object points can be approximately explained as a linear combination of other point tracks. Our method outperforms the prior art on motion-based segmentation, which shows the utility of long-term motion and the effectiveness of our formulation. 

**Abstract (ZH)**: 我们考虑基于运动对视频中对象进行分割的问题，而不依赖于其他形式的监督。先前的工作通常通过共命运的原则来解决这个问题，即属于同一对象的点的运动之间存在强烈的相关性。然而，大多数作者仅考虑了来自光流的瞬时运动。在本研究中，我们提出了使用长时间点轨迹作为监督信号来补充光流的方法来训练分割网络。关键的难点在于，与瞬时运动不同，长时间运动很难建模——任何参数化近似都不太可能捕捉到长时间内的复杂运动模式。我们从子空间聚类方法中获得灵感，提出了一种损失函数，旨在将轨迹分组为低秩矩阵，使得对象点的运动可以近似地解释为其他点轨迹的线性组合。我们的方法在基于运动的分割任务上优于现有方法，这表明长时间运动的有用性和我们方法的有效性。 

---
# Physics of Skill Learning 

**Title (ZH)**: 技能学习的物理学原理 

**Authors**: Ziming Liu, Yizhou Liu, Eric J. Michaud, Jeff Gore, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2501.12391)  

**Abstract**: We aim to understand physics of skill learning, i.e., how skills are learned in neural networks during training. We start by observing the Domino effect, i.e., skills are learned sequentially, and notably, some skills kick off learning right after others complete learning, similar to the sequential fall of domino cards. To understand the Domino effect and relevant behaviors of skill learning, we take physicists' approach of abstraction and simplification. We propose three models with varying complexities -- the Geometry model, the Resource model, and the Domino model, trading between reality and simplicity. The Domino effect can be reproduced in the Geometry model, whose resource interpretation inspires the Resource model, which can be further simplified to the Domino model. These models present different levels of abstraction and simplification; each is useful to study some aspects of skill learning. The Geometry model provides interesting insights into neural scaling laws and optimizers; the Resource model sheds light on the learning dynamics of compositional tasks; the Domino model reveals the benefits of modularity. These models are not only conceptually interesting -- e.g., we show how Chinchilla scaling laws can emerge from the Geometry model, but also are useful in practice by inspiring algorithmic development -- e.g., we show how simple algorithmic changes, motivated by these toy models, can speed up the training of deep learning models. 

**Abstract (ZH)**: 我们旨在理解技能学习的物理原理，即神经网络在训练过程中是如何学习技能的。我们从观察“多米诺骨牌效应”开始，即技能是按顺序学习的，并且值得注意的是，某些技能在其他技能完成学习后立即开始学习，类似于多米诺骨牌的依次倒下。为了理解多米诺骨牌效应及相关的行为，我们采用了物理学家们的方法，即抽象和简化。我们提出了三个具有不同复杂度的模型——几何模型、资源模型和多米诺模型，它们在现实与简化之间进行权衡。多米诺骨牌效应可以在几何模型中重现，该模型的资源解释启发了资源模型，后者又可进一步简化为多米诺模型。这些模型展示了不同程度的抽象与简化；每个模型都有助于研究技能学习的某些方面。几何模型为神经网络的比例法则和优化器提供了有趣见解；资源模型揭示了组合任务学习动力学的规律；多米诺模型揭示了模块化的优点。这些模型不仅在概念上有重要意义——例如，我们展示了如何从几何模型中推导出Chinchilla比例法则，而且在实践中也非常有用，能够启发算法的发展——例如，我们展示了基于这些玩具模型的简单算法更改如何加快深度学习模型的训练速度。 

---
# MMVU: Measuring Expert-Level Multi-Discipline Video Understanding 

**Title (ZH)**: MMVU：衡量多学科视频理解的专业水平 

**Authors**: Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, Zhijian Xu, Chengye Wang, Weifeng Pan, Ziyao Shangguan, Xiangru Tang, Zhenwen Liang, Yixin Liu, Chen Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12380)  

**Abstract**: We introduce MMVU, a comprehensive expert-level, multi-discipline benchmark for evaluating foundation models in video understanding. MMVU includes 3,000 expert-annotated questions spanning 27 subjects across four core disciplines: Science, Healthcare, Humanities & Social Sciences, and Engineering. Compared to prior benchmarks, MMVU features three key advancements. First, it challenges models to apply domain-specific knowledge and perform expert-level reasoning to analyze specialized-domain videos, moving beyond the basic visual perception typically assessed in current video benchmarks. Second, each example is annotated by human experts from scratch. We implement strict data quality controls to ensure the high quality of the dataset. Finally, each example is enriched with expert-annotated reasoning rationals and relevant domain knowledge, facilitating in-depth analysis. We conduct an extensive evaluation of 32 frontier multimodal foundation models on MMVU. The latest System-2-capable models, o1 and Gemini 2.0 Flash Thinking, achieve the highest performance among the tested models. However, they still fall short of matching human expertise. Through in-depth error analyses and case studies, we offer actionable insights for future advancements in expert-level, knowledge-intensive video understanding for specialized domains. 

**Abstract (ZH)**: 我们将介绍MMVU，这是一个全面的专家级多学科基准，用于评估基础模型在视频理解方面的性能。MMVU 包括涵盖四个核心学科的 27 个主题的 3000 个由专家注释的问题：科学、医疗保健、人文学科与社会科学、以及工程学。与先前的基准相比，MMVU 具有三个关键的改进。首先，它要求模型应用特定领域的知识并执行专家级推理，以分析专门领域的视频，这超越了当前视频基准中通常评估的基本视觉感知。第二，每个示例都从零开始由人类专家进行标注。我们实施严格的数据质量控制，以确保数据集的质量。最后，每个示例都得到了专家注释的推理依据和相关领域知识的丰富，便于深入分析。我们在 MMVU 上对 32 个前沿多模态基础模型进行了广泛的评估。最新的系统-2 级模型 o1 和 Gemini 2.0 Flash Thinking 在测试模型中表现出最高的性能，但仍然无法达到人类专业知识的水平。通过深入的错误分析和案例研究，我们为未来在专门领域中实现专家级、知识密集型视频理解提供了可操作的见解。 

---
# Video Depth Anything: Consistent Depth Estimation for Super-Long Videos 

**Title (ZH)**: 视频深度万物：超长视频的一致深度估计 

**Authors**: Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, Bingyi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12375)  

**Abstract**: Depth Anything has achieved remarkable success in monocular depth estimation with strong generalization ability. However, it suffers from temporal inconsistency in videos, hindering its practical applications. Various methods have been proposed to alleviate this issue by leveraging video generation models or introducing priors from optical flow and camera poses. Nonetheless, these methods are only applicable to short videos (< 10 seconds) and require a trade-off between quality and computational efficiency. We propose Video Depth Anything for high-quality, consistent depth estimation in super-long videos (over several minutes) without sacrificing efficiency. We base our model on Depth Anything V2 and replace its head with an efficient spatial-temporal head. We design a straightforward yet effective temporal consistency loss by constraining the temporal depth gradient, eliminating the need for additional geometric priors. The model is trained on a joint dataset of video depth and unlabeled images, similar to Depth Anything V2. Moreover, a novel key-frame-based strategy is developed for long video inference. Experiments show that our model can be applied to arbitrarily long videos without compromising quality, consistency, or generalization ability. Comprehensive evaluations on multiple video benchmarks demonstrate that our approach sets a new state-of-the-art in zero-shot video depth estimation. We offer models of different scales to support a range of scenarios, with our smallest model capable of real-time performance at 30 FPS. 

**Abstract (ZH)**: Depth Anything已经在单目深度估计中取得了显著的成功，具有较强的泛化能力。然而，它在视频中的时间一致性方面存在不足，这限制了其实际应用。为了缓解这一问题，提出了多种方法，这些方法要么利用视频生成模型，要么引入光学流和相机姿态的先验知识。然而，这些方法仅适用于短视频（少于10秒），并且在质量与计算效率之间需要权衡。为此，我们提出了Video Depth Anything，用于在超长视频（数分钟以上）中实现高质量且一致的深度估计，同时保持高效性。我们基于Depth Anything V2构建了模型，并用高效的空间-时间头部（spatial-temporal head）取代了其头部模块。我们通过约束时间深度梯度设计了一个简单有效的时序一致性损失，从而消除了额外几何先验的需求。该模型使用的训练数据集结合了视频深度标签和未标记的图像，类似于Depth Anything V2。此外，我们还开发了一种新颖的基于关键帧的长视频推理策略。实验结果表明，我们的模型可以应用于任意长度的视频，而不牺牲质量、一致性和泛化能力。针对多个视频基准的全面评估表明，我们的方法在零样本视频深度估计中达到了新的最先进水平。我们提供了不同规模的模型，以支持各种应用场景，其中我们最小的模型能够在30 FPS下实现实时性能。 

---
# Expertise elevates AI usage: experimental evidence comparing laypeople and professional artists 

**Title (ZH)**: 专家知识提升人工智能应用：普通人群与专业艺术家的实验比较 

**Authors**: Thomas F. Eisenmann, Andres Karjus, Mar Canet Sola, Levin Brinkmann, Bramantyo Ibrahim Supriyatno, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12374)  

**Abstract**: Novel capacities of generative AI to analyze and generate cultural artifacts raise inevitable questions about the nature and value of artistic education and human expertise. Has AI already leveled the playing field between professional artists and laypeople, or do trained artistic expressive capacity, curation skills and experience instead enhance the ability to use these new tools? In this pre-registered study, we conduct experimental comparisons between 50 active artists and a demographically matched sample of laypeople. We designed two tasks to approximate artistic practice for testing their capabilities in both faithful and creative image creation: replicating a reference image, and moving as far away as possible from it. We developed a bespoke platform where participants used a modern text-to-image model to complete both tasks. We also collected and compared participants' sentiments towards AI. On average, artists produced more faithful and creative outputs than their lay counterparts, although only by a small margin. While AI may ease content creation, professional expertise is still valuable - even within the confined space of generative AI itself. Finally, we also explored how well an exemplary vision-capable large language model (GPT-4o) would complete the same tasks, if given the role of an image generation agent, and found it performed on par in copying but outperformed even artists in the creative task. The very best results were still produced by humans in both tasks. These outcomes highlight the importance of integrating artistic skills with AI training to prepare artists and other visual professionals for a technologically evolving landscape. We see a potential in collaborative synergy with generative AI, which could reshape creative industries and education in the arts. 

**Abstract (ZH)**: 生成型人工智能对文化艺术品进行分析和生成的新能力引发了关于艺术教育和人类专业价值本质的不可避免的问题。AI是否已经消除了专业艺术家和非专业人士之间的竞争门槛，还是受过训练的艺术表达能力、策展技巧和经验反而增强了使用这些新工具的能力？在本项预先注册的研究中，我们对50名活跃艺术家和人口统计学特征匹配的非专业人士样本进行了实验比较。我们设计了两个任务来近似艺术实践，以检测他们在忠实和创意图像生成方面的能力：复制参考图像，以及尽量远离该图像。我们开发了一个定制平台，让参与者使用现代文本转图像模型完成这两个任务。我们还收集并比较了参与者对人工智能的态度。总体而言，艺术家在忠实度和创意方面产生的输出略优于非专业人士的相应输出。尽管如此，AI在内容创作方面仍可能简化操作，但专业技能在生成型AI领域内仍然具有价值。最后，我们还探讨了对于视图能力强大的大型语言模型（GPT-4o）而言，如果将其视为图像生成代理，它在完成相同任务方面的情况，并发现在复制任务中表现相似，但在创意任务中甚至超越了艺术家。最好的结果仍然由人类产生。这些结果突显了将艺术技能与AI培训结合的重要性，以准备艺术家和其他视觉专业人士应对技术不断发展的环境。我们看到了与生成型人工智能协作的潜力，这可能会重塑创意行业和艺术教育。 

---
# Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL 

**Title (ZH)**: 长上下文就足够吗？利用大语言模型扩展上下文进行NL2SQL转换 

**Authors**: Yeounoh Chung, Gaurav T. Kakkar, Yu Gan, Brenton Milne, Fatma Ozcan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12372)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information.
In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve a strong performance with 67.41\% on BIRD benchmark (dev) without finetuning and expensive self-consistency based techniques. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种自然语言处理任务上展现了令人印象深刻的性能。特别地，推理能力的提升和上下文窗口的扩展为利用这些强大的模型开辟了新的途径。将自然语言转换为SQL（NL2SQL）是一个具有挑战性的问题，因为自然语言问题是固有的含糊不清的，而SQL生成则需要对复杂的数据模式和语义有精确的理解。为了解决这一语义模糊问题，提供更多的上下文信息是可行的方法之一。

在本文中，我们探讨了Google最新一代LLM（即gemini-1.5-pro）所提供的扩展上下文窗口（也称为长上下文）的性能和延迟权衡。我们研究了各种上下文信息的影响，包括列示例值、问题与SQL查询配对、用户提供的提示、SQL文档和模式。据我们所知，这是首次研究扩展上下文窗口和额外上下文信息如何帮助提升NL2SQL生成的准确性和减少延迟成本的工作。我们展示了，在未经微调和昂贵的自我一致性方法优化的情况下，基于Google的gemini-pro-1.5构建的长上下文NL2SQL流水线在BIRD基准测试（开发集）上取得了67.41%的强性能。此外，长上下文LLMs具有鲁棒性，不会在扩展的上下文信息中迷失方向。 

---
# Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models 

**Title (ZH)**: 参数量 vs FLOPs：混合专家语言模型最优稀疏化缩放定律 

**Authors**: Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin Mohamed Elnouby Ali, Josh Susskind, Vimal Thilak  

**Link**: [PDF](https://arxiv.org/pdf/2501.12370)  

**Abstract**: Scaling the capacity of language models has consistently proven to be a reliable approach for improving performance and unlocking new capabilities. Capacity can be primarily defined by two dimensions: the number of model parameters and the compute per example. While scaling typically involves increasing both, the precise interplay between these factors and their combined contribution to overall capacity remains not fully understood. We explore this relationship in the context of sparse Mixture-of-Expert models (MoEs), which allow scaling the number of parameters without proportionally increasing the FLOPs per example. We investigate how varying the sparsity level, i.e., the ratio of non-active to total parameters, affects model performance in terms of both pretraining and downstream performance. We find that under different constraints (e.g. parameter size and total training compute), there is an optimal level of sparsity that improves both training efficiency and model performance. These results provide a better understanding of the impact of sparsity in scaling laws for MoEs and complement existing works in this area, offering insights for designing more efficient architectures. 

**Abstract (ZH)**: 语言模型能力的扩展始终是一个可靠的方法，用于提高性能并解锁新功能。能力主要可以通过两个维度来定义：模型参数的数量和每个示例的计算量。虽然扩展通常涉及增加这两个方面，但这些因素之间的确切相互作用及其对总体能力的综合贡献仍不完全清楚。我们在稀疏混合专家模型（MoEs）的背景下探讨了这种关系，稀疏MoEs允许在不以成比例的方式增加每个示例的浮点运算量（FLOPs）的情况下扩展参数的数量。我们研究了不同稀疏度水平（即非激活参数与总参数的比例）对模型性能的影响，包括预训练和下游性能。我们发现，在不同的约束条件下（例如参数大小和总量训练计算量），存在一个最优的稀疏度水平，可以同时提高训练效率和模型性能。这些结果更好地理解了MoEs在扩展律中的稀疏性影响，并补充了该领域的现有工作，为设计更高效的架构提供了见解。 

---
# DARB-Splatting: Generalizing Splatting with Decaying Anisotropic Radial Basis Functions 

**Title (ZH)**: DARB-Splatting: 通过衰减各向异性径向基函数泛化Splatting方法 

**Authors**: Vishagar Arunan, Saeedha Nazar, Hashiru Pramuditha, Vinasirajan Viruthshaan, Sameera Ramasinghe, Simon Lucey, Ranga Rodrigo  

**Link**: [PDF](https://arxiv.org/pdf/2501.12369)  

**Abstract**: Splatting-based 3D reconstruction methods have gained popularity with the advent of 3D Gaussian Splatting, efficiently synthesizing high-quality novel views. These methods commonly resort to using exponential family functions, such as the Gaussian function, as reconstruction kernels due to their anisotropic nature, ease of projection, and differentiability in rasterization. However, the field remains restricted to variations within the exponential family, leaving generalized reconstruction kernels largely underexplored, partly due to the lack of easy integrability in 3D to 2D projections. In this light, we show that a class of decaying anisotropic radial basis functions (DARBFs), which are non-negative functions of the Mahalanobis distance, supports splatting by approximating the Gaussian function's closed-form integration advantage. With this fresh perspective, we demonstrate up to 34% faster convergence during training and a 15% reduction in memory consumption across various DARB reconstruction kernels, while maintaining comparable PSNR, SSIM, and LPIPS results. We will make the code available. 

**Abstract (ZH)**: 基于斑点的3D重建方法随着3D高斯斑点技术的发展而变得流行，能够高效地合成高质量的新视角。这些方法通常采用指数族函数作为重建核，如高斯函数，因为它们具有各向异性、便于投影且在像素化过程中可微分。然而，该领域仍然局限于指数族的变体，广泛使用的通用重建核尚未得到充分探索，部分原因是三维到二维投影的积分缺乏简便性。在这一背景下，我们表明一类衰减的各向异性径向基函数（Decaying Anisotropic Radial Basis Functions，DARBFs），作为马氏距离的非负函数，能够通过近似高斯函数的闭式积分优势支持斑点化。借助这一新颖的视角，我们展示了在各种DARBF重建核中训练时可达34%的更快收敛速度，并且在内存消耗上减少了15%，同时保持与PSNR、SSIM和LPIPS结果相当的水平。我们将开源代码。 

---
# Test-time regression: a unifying framework for designing sequence models with associative memory 

**Title (ZH)**: 测试时回归：一种基于关联记忆设计序列模型的统一框架 

**Authors**: Ke Alexander Wang, Jiaxin Shi, Emily B. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2501.12352)  

**Abstract**: Sequences provide a remarkably general way to represent and process information. This powerful abstraction has placed sequence modeling at the center of modern deep learning applications, inspiring numerous architectures from transformers to recurrent networks. While this fragmented development has yielded powerful models, it has left us without a unified framework to understand their fundamental similarities and explain their effectiveness. We present a unifying framework motivated by an empirical observation: effective sequence models must be able to perform associative recall. Our key insight is that memorizing input tokens through an associative memory is equivalent to performing regression at test-time. This regression-memory correspondence provides a framework for deriving sequence models that can perform associative recall, offering a systematic lens to understand seemingly ad-hoc architectural choices. We show numerous recent architectures -- including linear attention models, their gated variants, state-space models, online learners, and softmax attention -- emerge naturally as specific approaches to test-time regression. Each architecture corresponds to three design choices: the relative importance of each association, the regressor function class, and the optimization algorithm. This connection leads to new understanding: we provide theoretical justification for QKNorm in softmax attention, and we motivate higher-order generalizations of softmax attention. Beyond unification, our work unlocks decades of rich statistical tools that can guide future development of more powerful yet principled sequence models. 

**Abstract (ZH)**: 序列提供了极其通用的方式用于表示和处理信息。这一强大的抽象使序列建模成为现代深度学习应用的核心，启发了包括变换器到循环网络在内的一系列架构。尽管这种分散的发展产生了极其强大的模型，但它们缺乏一个统一的框架来理解其基本的相似性并解释其有效性。我们提出了一种统一框架，其灵感来源于一个经验观察：有效的序列模型必须能够执行联想回忆。

我们的核心见解是，通过关联记忆存储输入令牌相当于在测试时执行回归。这种回归-记忆对应关系为开发能够执行联想回忆的序列模型框架提供了理论基础，提供了一种系统性的视角来理解看似临时性的架构选择。我们展示了众多最新的架构——包括线性注意力模型及其门控变体、状态空间模型、在线学习器以及softmax注意力——自然地作为测试时回归的特定方法而出现。每个架构对应三种设计选择：每种关联的重要性相对权重、回归函数类以及优化算法。这种联系带来了新的理解：我们提供了对softmax注意力中QKNorm的理论依据，并推动了softmax注意力的高阶泛化形式。除了统一之外，我们的工作揭开了数十年丰富的统计工具，这些工具可以指导未来更强大且更具原则性的序列模型的发展。 

---
# Treefix: Enabling Execution with a Tree of Prefixes 

**Title (ZH)**: Treefix：启用树形前缀树的执行 

**Authors**: Beatriz Souza, Michael Pradel  

**Link**: [PDF](https://arxiv.org/pdf/2501.12339)  

**Abstract**: The ability to execute code is a prerequisite for various dynamic program analyses. Learning-guided execution has been proposed as an approach to enable the execution of arbitrary code snippets by letting a neural model predict likely values for any missing variables. Although state-of-the-art learning-guided execution approaches, such as LExecutor, can enable the execution of a relative high amount of code, they are limited to predicting a restricted set of possible values and do not use any feedback from previous executions to execute even more code. This paper presents Treefix, a novel learning-guided execution approach that leverages LLMs to iteratively create code prefixes that enable the execution of a given code snippet. The approach addresses the problem in a multi-step fashion, where each step uses feedback about the code snippet and its execution to instruct an LLM to improve a previously generated prefix. This process iteratively creates a tree of prefixes, a subset of which is returned to the user as prefixes that maximize the number of executed lines in the code snippet. In our experiments with two datasets of Python code snippets, Treefix achieves 25% and 7% more coverage relative to the current state of the art in learning-guided execution, covering a total of 84% and 82% of all lines in the code snippets. 

**Abstract (ZH)**: 执行代码的能力是各种动态程序分析的前提条件。学习引导执行已被提出作为一种方法，通过让神经模型预测任意缺失变量的可能值，从而使任意代码片段能够被执行。尽管当前最先进的学习引导执行方法，如LExecutor，能够使大量代码被执行，但它们只能预测一组受限的可能值，并且不利用先前执行的反馈来进一步扩展执行的代码量。本文介绍了一种名为Treefix的新型学习引导执行方法，利用大型语言模型（LLMs）逐步创建代码前缀，以使给定代码片段能够被执行。该方法以多步方式来解决这一问题，每一步都利用关于代码片段及其执行的反馈来指导一个LLM改进先前生成的前缀。此过程逐步创建一棵前缀树，其中部分前缀被返回给用户，以最大化代码片段中被执行的行数。在对两个包含Python代码片段的数据集进行实验中，Treefix在学习引导执行方面分别实现了25%和7%更高的覆盖率，总共覆盖了代码片段中84%和82%的行数。 

---
# FuocChuVIP123 at CoMeDi Shared Task: Disagreement Ranking with XLM-Roberta Sentence Embeddings and Deep Neural Regression 

**Title (ZH)**: FuocChuVIP123在CoMeDi共享任务中的表现：基于XLM-Roberta句向量和深度神经回归的分歧排名 

**Authors**: Phuoc Duong Huy Chu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12336)  

**Abstract**: This paper presents results of our system for CoMeDi Shared Task, focusing on Subtask 2: Disagreement Ranking. Our system leverages sentence embeddings generated by the paraphrase-xlm-r-multilingual-v1 model, combined with a deep neural regression model incorporating batch normalization and dropout for improved generalization. By predicting the mean of pairwise judgment differences between annotators, our method explicitly targets disagreement ranking, diverging from traditional "gold label" aggregation approaches. We optimized our system with a customized architecture and training procedure, achieving competitive performance in Spearman correlation against mean disagreement labels. Our results highlight the importance of robust embeddings, effective model architecture, and careful handling of judgment differences for ranking disagreement in multilingual contexts. These findings provide insights into the use of contextualized representations for ordinal judgment tasks and open avenues for further refinement of disagreement prediction models. 

**Abstract (ZH)**: 本文介绍了我们系统在 CoMeDi 共享任务中的研究成果，重点关注子任务 2：分歧排名。我们的系统利用了由 paraphrase-xlm-r-multilingual-v1 模型生成的句子嵌入，并结合了包含批量归一化和 Dropout 的深度神经回归模型，以提高模型的泛化能力。通过预测注释者之间成对判断差异的均值，我们的方法明确地针对分歧排名，不同于传统的“金标准”聚合方法。我们通过自定义架构和训练过程优化了系统，在斯皮尔曼相关性方面获得了与均值分歧标签竞争性的性能。我们的结果强调了在多语言背景下排名分歧时稳健的嵌入、有效的模型架构和细致处理判断差异的重要性。这些发现为使用上下文化表示进行序数判断任务提供了洞察，并为改进分歧预测模型开辟了进一步优化的道路。 

---
# Automatic Labelling with Open-source LLMs using Dynamic Label Schema Integration 

**Title (ZH)**: 使用动态标签方案集成的开源大语言模型自动标注 

**Authors**: Thomas Walshe, Sae Young Moon, Chunyang Xiao, Yawwani Gunawardana, Fran Silavong  

**Link**: [PDF](https://arxiv.org/pdf/2501.12332)  

**Abstract**: Acquiring labelled training data remains a costly task in real world machine learning projects to meet quantity and quality requirements. Recently Large Language Models (LLMs), notably GPT-4, have shown great promises in labelling data with high accuracy. However, privacy and cost concerns prevent the ubiquitous use of GPT-4. In this work, we explore effectively leveraging open-source models for automatic labelling. We identify integrating label schema as a promising technology but found that naively using the label description for classification leads to poor performance on high cardinality tasks. To address this, we propose Retrieval Augmented Classification (RAC) for which LLM performs inferences for one label at a time using corresponding label schema; we start with the most related label and iterates until a label is chosen by the LLM. We show that our method, which dynamically integrates label description, leads to performance improvements in labelling tasks. We further show that by focusing only on the most promising labels, RAC can trade off between label quality and coverage - a property we leverage to automatically label our internal datasets. 

**Abstract (ZH)**: 在现实世界的机器学习项目中，获取带有标注的训练数据仍是一项昂贵的任务，需要满足数量和质量要求。最近，大型语言模型（LLMs），尤其是GPT-4，在数据标注方面展示了高准确性，但隐私和成本问题阻碍了GPT-4的广泛应用。在本研究中，我们探索了有效地利用开源模型进行自动标注。我们发现，将标签模式集成在一起是一种有前景的技术，但直接使用标签描述进行分类在高维度任务中表现不佳。为了解决这一问题，我们提出了检索增强分类（RAC）方法，其中LLM一次对一个标签进行推理，并根据相应的标签模式进行；我们从关联性最强的标签开始，迭代直至LLM选择一个标签。研究表明，我们的方法能够动态集成标签描述，从而在标注任务中提高性能。进一步地，我们展示了通过仅关注最具前景的标签，RAC可以在标签质量和覆盖之间进行权衡——我们利用这一特性自动标注我们的内部数据集。 

---
# LLM-Assisted Knowledge Graph Completion for Curriculum and Domain Modelling in Personalized Higher Education Recommendations 

**Title (ZH)**: 基于LLM辅助的知识图谱补全在个性化高等教育推荐中的课程与领域建模 

**Authors**: Hasan Abu-Rasheed, Constance Jumbo, Rashed Al Amin, Christian Weber, Veit Wiese, Roman Obermaisser, Madjid Fathi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12300)  

**Abstract**: While learning personalization offers great potential for learners, modern practices in higher education require a deeper consideration of domain models and learning contexts, to develop effective personalization algorithms. This paper introduces an innovative approach to higher education curriculum modelling that utilizes large language models (LLMs) for knowledge graph (KG) completion, with the goal of creating personalized learning-path recommendations. Our research focuses on modelling university subjects and linking their topics to corresponding domain models, enabling the integration of learning modules from different faculties and institutions in the student's learning path. Central to our approach is a collaborative process, where LLMs assist human experts in extracting high-quality, fine-grained topics from lecture materials. We develop a domain, curriculum, and user models for university modules and stakeholders. We implement this model to create the KG from two study modules: Embedded Systems and Development of Embedded Systems Using FPGA. The resulting KG structures the curriculum and links it to the domain models. We evaluate our approach through qualitative expert feedback and quantitative graph quality metrics. Domain experts validated the relevance and accuracy of the model, while the graph quality metrics measured the structural properties of our KG. Our results show that the LLM-assisted graph completion approach enhances the ability to connect related courses across disciplines to personalize the learning experience. Expert feedback also showed high acceptance of the proposed collaborative approach for concept extraction and classification. 

**Abstract (ZH)**: 虽然个性化学习为学习者提供了巨大的潜力，但现代高等教育实践需要更加重视领域模型和学习情境，以开发有效的个性化算法。本文介绍了一种创新的高等教育课程建模方法，利用大型语言模型（LLMs）进行知识图谱（KG）补全，旨在构建个性化的学习路径推荐。我们的研究重点在于建模大学课程及其主题，并将这些主题与其对应的域模型链接起来，从而使不同学院和机构的学习模块能够整合到学生的学习路径中。我们方法的核心在于一种协作过程，其中LLMs协助人类专家从讲义材料中提取高质量、细粒度的主题。我们为大学模块和利益相关者开发了领域模型、课程模型和用户模型。我们使用这种模型从两个学习模块——嵌入式系统和基于FPGA的嵌入式系统开发——构建知识图谱。生成的KG结构化了课程内容，并将其与域模型链接起来。我们通过定性的专家反馈和定量的知识图谱质量指标来评估该方法。领域专家验证了模型的相关性和准确性，而图的结构属性是衡量我们知识图谱质量的指标。研究表明，LLM辅助的知识图谱补全方法增强了跨学科连接相关课程的能力，以个性化学习体验。专家反馈还表明，对概念提取和分类的协作方法有很高的接受度。 

---
# RALAD: Bridging the Real-to-Sim Domain Gap in Autonomous Driving with Retrieval-Augmented Learning 

**Title (ZH)**: RALAD：通过检索增强学习弥合自主驾驶现实到模拟领域差距 

**Authors**: Jiacheng Zuo, Haibo Hu, Zikang Zhou, Yufei Cui, Ziquan Liu, Jianping Wang, Nan Guan, Jin Wang, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2501.12296)  

**Abstract**: In the pursuit of robust autonomous driving systems, models trained on real-world datasets often struggle to adapt to new environments, particularly when confronted with corner cases such as extreme weather conditions. Collecting these corner cases in the real world is non-trivial, which necessitates the use of simulators for validation. However,the high computational cost and the domain gap in data distribution have hindered the seamless transition between real and simulated driving scenarios. To tackle this challenge, we propose Retrieval-Augmented Learning for Autonomous Driving (RALAD), a novel framework designed to bridge the real-to-sim gap at a low cost. RALAD features three primary designs, including (1) domain adaptation via an enhanced Optimal Transport (OT) method that accounts for both individual and grouped image distances, (2) a simple and unified framework that can be applied to various models, and (3) efficient fine-tuning techniques that freeze the computationally expensive layers while maintaining robustness. Experimental results demonstrate that RALAD compensates for the performance degradation in simulated environments while maintaining accuracy in real-world scenarios across three different models. Taking Cross View as an example, the mIOU and mAP metrics in real-world scenarios remain stable before and after RALAD fine-tuning, while in simulated environments,the mIOU and mAP metrics are improved by 10.30% and 12.29%, respectively. Moreover, the re-training cost of our approach is reduced by approximately 88.1%. Our code is available at this https URL. 

**Abstract (ZH)**: 在追求稳健的自主驾驶系统过程中，基于真实数据训练的模型在适应新的环境时经常遇到困难，尤其是在面对极端天气等边缘情况时。在现实世界中收集这些边缘情况极具挑战性，因此需要通过模拟器进行验证。然而，高昂的计算成本以及数据分布的领域差距阻碍了真实驾驶场景与模拟驾驶场景之间的无缝过渡。为解决这一挑战，我们提出了一种新颖的框架——基于检索增强学习的自主驾驶（RALAD），旨在以较低的成本弥合现实到模拟之间的差距。RALAD包含三个主要设计，包括：(1) 通过一种增强的最优传输（OT）方法来实现领域适应，该方法考虑了单个图像距离和群体图像距离；(2) 一种易于应用且统一的框架，可以适应各种不同模型；(3) 高效的微调技术，这些技术会冻结计算成本高昂的层，同时保持鲁棒性。实验结果显示，在模拟环境中，RALAD能够弥补性能下降，在三种不同模型的实际驾驶场景中保持准确性。以交叉视图为例，在应用RALAD微调前后，实际驾驶场景中的mIOU和mAP指标保持稳定；而在模拟环境中，mIOU和mAP指标分别提高了10.30%和12.29%。此外，我们方法的重新训练成本降低了约88.1%。我们的代码可以在以下链接获取：[此处链接]。 

---
# Regressor-Guided Image Editing Regulates Emotional Response to Reduce Online Engagement 

**Title (ZH)**: 基于回归器引导的图像编辑调控情绪反应以减少在线参与 

**Authors**: Christoph Gebhardt, Robin Willardt, Seyedmorteza Sadat, Chih-Wei Ning, Andreas Brombach, Jie Song, Otmar Hilliges, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2501.12289)  

**Abstract**: Emotions are known to mediate the relationship between users' content consumption and their online engagement, with heightened emotional intensity leading to increased engagement. Building on this insight, we propose three regressor-guided image editing approaches aimed at diminishing the emotional impact of images. These include (i) a parameter optimization approach based on global image transformations known to influence emotions, (ii) an optimization approach targeting the style latent space of a generative adversarial network, and (iii) a diffusion-based approach employing classifier guidance and classifier-free guidance. Our findings demonstrate that approaches can effectively alter the emotional properties of images while maintaining high visual quality. Optimization-based methods primarily adjust low-level properties like color hues and brightness, whereas the diffusion-based approach introduces semantic changes, such as altering appearance or facial expressions. Notably, results from a behavioral study reveal that only the diffusion-based approach successfully elicits changes in viewers' emotional responses while preserving high perceived image quality. In future work, we will investigate the impact of these image adaptations on internet user behavior. 

**Abstract (ZH)**: 情绪被证明能够调节用户内容消费与在线互动之间的关系，情绪强度增强会促进更高的互动水平。基于这一洞见，我们提出三种引导式图像编辑方法，旨在减轻图像的情绪影响。这些方法包括：（i）一种基于全局图像变换的参数优化方法，这类变换已知可以影响情绪；（ii）一种针对生成对抗网络样式潜在空间的优化方法；（iii）一种基于扩散的分类器引导和无分类器引导的方法。我们的研究结果表明，这些方法可以在保持高视觉质量的同时有效改变图像的情绪属性。基于优化的方法主要调整低级属性，如色调和亮度，而基于扩散的方法则引入了语义上的变化，例如改变外观或面部表情。值得注意的是，行为研究的结果表明，只有基于扩散的方法能够成功引发观者的情绪反应变化，同时保持高感知的图像质量。在未来的工作中，我们将研究这些图像适应对互联网用户行为的影响。 

---
# Implementation of an Asymmetric Adjusted Activation Function for Class Imbalance Credit Scoring 

**Title (ZH)**: 针对信贷评分中的类别不平衡问题，实现一种非对称调整激活函数的实施方案 

**Authors**: Xia Li, Hanghang Zheng, Kunpeng Tao, Mao Mao  

**Link**: [PDF](https://arxiv.org/pdf/2501.12285)  

**Abstract**: Credit scoring is a systematic approach to evaluate a borrower's probability of default (PD) on a bank loan. The data associated with such scenarios are characteristically imbalanced, complicating binary classification owing to the often-underestimated cost of misclassification during the classifier's learning process. Considering the high imbalance ratio (IR) of these datasets, we introduce an innovative yet straightforward optimized activation function by incorporating an IR-dependent asymmetric adjusted factor embedded Sigmoid activation function (ASIG). The embedding of ASIG makes the sensitive margin of the Sigmoid function auto-adjustable, depending on the imbalance nature of the datasets distributed, thereby giving the activation function an asymmetric characteristic that prevents the underrepresentation of the minority class (positive samples) during the classifier's learning process. The experimental results show that the ASIG-embedded-classifier outperforms traditional classifiers on datasets across wide-ranging IRs in the downstream credit-scoring task. The algorithm also shows robustness and stability, even when the IR is ultra-high. Therefore, the algorithm provides a competitive alternative in the financial industry, especially in credit scoring, possessing the ability to effectively process highly imbalanced distribution data. 

**Abstract (ZH)**: 信用评分是一种系统方法，用于评估借款人在银行贷款中的违约概率（PD）。此类情境下的数据通常表现出不平衡特性，这使得二元分类复杂化，尤其是在分类器学习过程中往往低估了错误分类的成本。考虑到这些数据集的高不平衡比率（IR），我们提出了一种创新且简洁的优化激活函数，通过引入一个依赖于不平衡比率的不对称调整因子嵌入Sigmoid激活函数（ASIG）。该嵌入使Sigmoid函数的敏感边缘能够根据数据集分布的不平衡性质自动调整，从而使激活函数具有不对称特性，防止在分类器学习过程中少数类（正样本）的过少表示。实验结果表明，嵌入ASIG的分类器在各种不平衡比率的数据集上表现优于传统分类器。该算法在高不平衡比率下也显示出稳健性和稳定性。因此，该算法为金融行业中，尤其是信用评分领域提供了具有有效处理高度不平衡分布数据能力的竞争性替代方案。 

---
# With Great Backbones Comes Great Adversarial Transferability 

**Title (ZH)**: 有了强大的主干网络，便伴随着强大的对抗性转移特性。 

**Authors**: Erik Arakelyan, Karen Hambardzumyan, Davit Papikyan, Pasquale Minervini, Albert Gordo, Isabelle Augenstein, Aram H. Markosyan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12275)  

**Abstract**: Advances in self-supervised learning (SSL) for machine vision have improved representation robustness and model performance, giving rise to pre-trained backbones like \emph{ResNet} and \emph{ViT} models tuned with SSL methods such as \emph{SimCLR}. Due to the computational and data demands of pre-training, the utilization of such backbones becomes a strenuous necessity. However, employing these backbones may inherit vulnerabilities to adversarial attacks. While adversarial robustness has been studied under \emph{white-box} and \emph{black-box} settings, the robustness of models tuned on pre-trained backbones remains largely unexplored. Additionally, the role of tuning meta-information in mitigating exploitation risks is unclear. This work systematically evaluates the adversarial robustness of such models across $20,000$ combinations of tuning meta-information, including fine-tuning techniques, backbone families, datasets, and attack types. We propose using proxy models to transfer attacks, simulating varying levels of target knowledge by fine-tuning these proxies with diverse configurations. Our findings reveal that proxy-based attacks approach the effectiveness of \emph{white-box} methods, even with minimal tuning knowledge. We also introduce a naive "backbone attack," leveraging only the backbone to generate adversarial samples, which outperforms \emph{black-box} attacks and rivals \emph{white-box} methods, highlighting critical risks in model-sharing practices. Finally, our ablations reveal how increasing tuning meta-information impacts attack transferability, measuring each meta-information combination. 

**Abstract (ZH)**: 自监督学习（SSL）在机器视觉中的进展提高了表示的鲁棒性和模型性能，形成了如ResNet和ViT模型这样的预训练骨干。由于这些预训练方法在计算和数据需求上的要求，使用这些骨干成为一种必要的选择。然而，使用这些骨干模型可能继承对抗攻击的脆弱性。虽然对抗鲁棒性已经在白盒和黑盒设置下进行了研究，但基于预训练骨干模型的鲁棒性却鲜有探讨。此外，调参元信息在减轻利用风险中的作用尚不明确。本文系统地评估了此类模型在20,000种不同调参元信息组合下的对抗鲁棒性，包括微调技术、骨干家族、数据集以及攻击类型。我们提出使用代理模型来转移攻击，通过以不同的配置微调这些代理模型来模拟目标知识的不同水平。我们的发现表明，在几乎没有任何调参知识的情况下，基于代理模型的攻击方法可以达到白盒方法的效果。我们还引入了一种简单的“骨干攻击”，只需利用骨干生成对抗样本，该方法比黑盒攻击更优，并且能够与白盒攻击媲美，强调了模型共享实践中的重要风险。最后，我们的消融实验揭示了增加调参元信息如何影响攻击转移性，并测量了每种组合的信息对攻击的成功率的影响。 

---
# Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement 

**Title (ZH)**: Condor：通过知识驱动的数据合成与精炼提高语言模型的对齐度 

**Authors**: Maosong Cao, Taolin Zhang, Mo Li, Chuyu Zhang, Yunxin Liu, Haodong Duan, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.12273)  

**Abstract**: The quality of Supervised Fine-Tuning (SFT) data plays a critical role in enhancing the conversational capabilities of Large Language Models (LLMs). However, as LLMs become more advanced, the availability of high-quality human-annotated SFT data has become a significant bottleneck, necessitating a greater reliance on synthetic training data. In this work, we introduce Condor, a novel two-stage synthetic data generation framework that incorporates World Knowledge Tree and Self-Reflection Refinement to produce high-quality SFT data at scale. Our experimental results demonstrate that a base model fine-tuned on only 20K Condor-generated samples achieves superior performance compared to counterparts. The additional refinement stage in Condor further enables iterative self-improvement for LLMs at various scales (up to 72B), validating the effectiveness of our approach. Furthermore, our investigation into the scaling for synthetic data in post-training reveals substantial unexplored potential for performance improvements, opening promising avenues for future research. 

**Abstract (ZH)**: 监督微调（SFT）数据的质量在提升大型语言模型（LLMs）的对话能力方面起着关键作用。然而，随着LLMs的不断进步，高质量的人工标注SFT数据的可用性已经成为一个重要瓶颈，迫使人们更加依赖合成训练数据。在此工作中，我们提出了Condor，这是一个新颖的两阶段合成数据生成框架，结合了World Knowledge Tree和Self-Reflection Refinement，以大规模生成高质量的SFT数据。我们的实验结果表明，仅使用20,000个Condor生成样本进行微调的基本模型，其性能优于其他模型。Condor中的额外完善阶段还进一步使LLMs在不同规模（多达720亿）下实现了迭代自我改进，验证了我们方法的有效性。此外，我们对后训练合成数据的扩展性研究揭示了大量未开发的性能提升潜力，为未来研究打开了有前景的研究方向。 

---
# CBVLM: Training-free Explainable Concept-based Large Vision Language Models for Medical Image Classification 

**Title (ZH)**: CBVLM：用于医学图像分类的无训练解释性概念基础大型视觉语言模型 

**Authors**: Cristiano Patrício, Isabel Rio-Torto, Jaime S. Cardoso, Luís F. Teixeira, João C. Neves  

**Link**: [PDF](https://arxiv.org/pdf/2501.12266)  

**Abstract**: The main challenges limiting the adoption of deep learning-based solutions in medical workflows are the availability of annotated data and the lack of interpretability of such systems. Concept Bottleneck Models (CBMs) tackle the latter by constraining the final disease prediction on a set of predefined and human-interpretable concepts. However, the increased interpretability achieved through these concept-based explanations implies a higher annotation burden. Moreover, if a new concept needs to be added, the whole system needs to be retrained. Inspired by the remarkable performance shown by Large Vision-Language Models (LVLMs) in few-shot settings, we propose a simple, yet effective, methodology, CBVLM, which tackles both of the aforementioned challenges. First, for each concept, we prompt the LVLM to answer if the concept is present in the input image. Then, we ask the LVLM to classify the image based on the previous concept predictions. Moreover, in both stages, we incorporate a retrieval module responsible for selecting the best examples for in-context learning. By grounding the final diagnosis on the predicted concepts, we ensure explainability, and by leveraging the few-shot capabilities of LVLMs, we drastically lower the annotation cost. We validate our approach with extensive experiments across four medical datasets and twelve LVLMs (both generic and medical) and show that CBVLM consistently outperforms CBMs and task-specific supervised methods without requiring any training and using just a few annotated examples. More information on our project page: this https URL. 

**Abstract (ZH)**: 基于深度学习的解决方案在医疗工作流程中的主要挑战在于标注数据的可用性和该类系统的解释性不足。概念瓶颈模型（CBMs）通过限制最终疾病预测在一组预定义且可由人类解释的概念上，解决了后者的问题。然而，通过基于概念的解释所获得的增强解释性意味着更高的标注负担。此外，如果需要添加新概念，则整个系统需要重新训练。受大型视觉-语言模型（LVLMs）在少样本设置下表现出色的启发，我们提出了一种简单而有效的方法——CBVLM，以应对上述两个挑战。首先，对于每个概念，我们促使LVLM判断该概念是否存在于输入图像中。然后，我们要求LVLM基于先前的概念预测对图像进行分类。此外，在两个阶段中，我们引入了一个检索模块，负责选择最适合的实例进行上下文学习。通过将最终诊断基于预测的概念，我们确保了可解释性，并通过利用LVLM的少样本能力，大大降低了标注成本。我们通过在四个医学数据集和十二种LVLM（包括通用和医学专用模型）上进行广泛的实验验证了该方法，并展示了CBVLM在不进行任何训练且仅使用少量标注样本的情况下，始终优于CBMs和针对特定任务的监督方法。更多详细信息请参阅我们的项目页面：[这里](this https URL)。 

---
# InsTALL: Context-aware Instructional Task Assistance with Multi-modal Large Language Models 

**Title (ZH)**: InsTALL：基于上下文的多模态大型语言模型辅助教学任务 

**Authors**: Pha Nguyen, Sailik Sengupta, Girik Malik, Arshit Gupta, Bonan Min  

**Link**: [PDF](https://arxiv.org/pdf/2501.12231)  

**Abstract**: The improved competence of generative models can help building multi-modal virtual assistants that leverage modalities beyond language. By observing humans performing multi-step tasks, one can build assistants that have situational awareness of actions and tasks being performed, enabling them to cater assistance based on this understanding. In this paper, we develop a Context-aware Instructional Task Assistant with Multi-modal Large Language Models (InsTALL) that leverages an online visual stream (e.g. a user's screen share or video recording) and responds in real-time to user queries related to the task at hand. To enable useful assistance, InsTALL 1) trains a multi-modal model on task videos and paired textual data, and 2) automatically extracts task graph from video data and leverages it at training and inference time. We show InsTALL achieves state-of-the-art performance across proposed sub-tasks considered for multimodal activity understanding -- task recognition (TR), action recognition (AR), next action prediction (AP), and plan prediction (PP) -- and outperforms existing baselines on two novel sub-tasks related to automatic error identification. 

**Abstract (ZH)**: 生成模型能力的提升有助于构建多模态虚拟助手，这些虚拟助手可以利用语言之外的多种模态信息。通过观察人类执行多步骤任务的过程，可以构建具有情境意识的助手，使其能够根据对这些任务的理解提供相应的帮助。在本文中，我们提出了一种基于多模态大规模语言模型的上下文感知指令任务助手（InsTALL），该助手利用在线视觉流（例如，用户的屏幕共享或视频录制）并实时响应与当前任务相关的用户查询。为了提供有用的帮助，InsTALL 1) 在任务视频及其配对的文本数据上训练一个多模态模型，2) 自动从视频数据中提取任务图，并在训练和推理过程中利用该图。我们展示了InsTALL在多模态活动理解所提出的子任务——任务识别（TR）、动作识别（AR）、下一个动作预测（AP）和计划预测（PP）——方面的性能达到了最先进的水平，并在两个与自动错误识别相关的新型子任务上优于现有 baseline。 

---
# Strong phonon-mediated high temperature superconductivity in Li$_2$AuH$_6$ under ambient pressure 

**Title (ZH)**: 在常压下的Li$_2$AuH$_6$中的强声子介导的高温超导性 

**Authors**: Zhenfeng Ouyang, Bo-Wen Yao, Xiao-Qi Han, Peng-Jie Guo, Ze-Feng Gao, Zhong-Yi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12222)  

**Abstract**: We used our developed AI search engine~(InvDesFlow) to perform extensive investigations regarding ambient stable superconducting hydrides. A cubic structure Li$_2$AuH$_6$ with Au-H octahedral motifs is identified to be a candidate. After performing thermodynamical analysis, we provide a feasible route to experimentally synthesize this material via the known LiAu and LiH compounds under ambient pressure. The further first-principles calculations suggest that Li$_2$AuH$_6$ shows a high superconducting transition temperature ($T_c$) $\sim$ 140 K under ambient pressure. The H-1$s$ electrons strongly couple with phonon modes of vibrations of Au-H octahedrons as well as vibrations of Li atoms, where the latter is not taken seriously in other previously similar cases. Hence, different from previous claims of searching metallic covalent bonds to find high-$T_c$ superconductors, we emphasize here the importance of those phonon modes with strong electron-phonon coupling (EPC). And we suggest that one can intercalate atoms into binary or ternary hydrides to introduce more potential phonon modes with strong EPC, which is an effective approach to find high-$T_c$ superconductors within multicomponent compounds. 

**Abstract (ZH)**: 我们使用自主研发的AI搜索引擎（InvDesFlow）对环境稳定的超导氢化物进行了广泛研究。通过结构分析，我们发现具有 Au-H 八面体图案的立方结构 Li₂AuH₆ 是一个候选材料。在进行热力学分析后，我们提出了一种可行的方法，在常压条件下，通过已知的 LiAu 和 LiH 化合物来实验合成该材料。进一步的理论计算表明，在常压条件下，Li₂AuH₆ 的超导转变温度（$T_c$）约为 140 K。氢-1s 电子强烈地与 Au-H 八面体振动模式以及 Li 原子的振动模式耦合，后者在其他类似情况下并未被充分考虑。因此，不同于以往通过寻找金属共价键来寻找高温超导体的尝试，我们强调了具有良好电子-声子耦合（EPC）的声子模态的重要性。此外，我们建议可以通过在二元或三元氢化物中插入原子来引入更多具有强EPC的声子模态，这在多组分化合物中寻找高温超导体的一种有效方法。 

---
# An End-to-End Approach for Korean Wakeword Systems with Speaker Authentication 

**Title (ZH)**: 面向韩国唤醒词系统的端到端方法结合说话人认证 

**Authors**: Geonwoo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2501.12194)  

**Abstract**: Wakeword detection plays a critical role in enabling AI assistants to listen to user voices and interact effectively. However, for languages other than English, there is a significant lack of pre-trained wakeword models. Additionally, systems that merely determine the presence of a wakeword can pose serious privacy concerns. In this paper, we propose an end-to-end approach that trains wakewords for Non-English languages, particulary Korean, and uses this to develop a Voice Authentication model to protect user privacy. Our implementation employs an open-source platform OpenWakeWord, which performs wakeword detection using an FCN (Fully-Connected Network) architecture. Once a wakeword is detected, our custom-developed code calculates cosine similarity for robust user authentication. Experimental results demonstrate the effectiveness of our approach, achieving a 16.79% and a 6.6% Equal Error Rate (EER) each in the Wakeword Detection and the Voice Authentication. These findings highlight the model's potential in providing secure and accurate wakeword detection and authentication for Korean users. 

**Abstract (ZH)**: 唤醒词检测在使AI助手机听取用户声音并有效交互方面发挥着关键作用。然而，除了英语之外的其他语言中，预训练的唤醒词模型严重不足。此外，仅用于检测唤醒词存在性的系统可能会引发严重的隐私问题。在本文中，我们提出了一种端到端的方法，用于为非英语语言，特别是韩语训练唤醒词，并利用此方法开发了一种语音认证模型以保护用户隐私。我们的实现使用了一个开源平台OpenWakeWord，该平台采用全连接网络（FCN）架构进行唤醒词检测。一旦检测到唤醒词，我们自定义开发的代码会计算余弦相似度以实现稳健的用户认证。实验结果表明，我们的方法在唤醒词检测和语音认证中的等错误率（EER）分别为16.79%和6.6%，这些发现突显了该模型在为韩语用户提供安全和准确的唤醒词检测和认证方面的潜力。 

---
# AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding 

**Title (ZH)**: AdaServe：基于细粒度推测解码的SLO定制化大语言模型服务 

**Authors**: Zikun Li, Zhuofu Chen, Remi Delacourt, Gabriele Oliaro, Zeyu Wang, Qinghan Chen, Shuhuai Lin, April Yang, Zhihao Zhang, Zhuoming Chen, Sean Lai, Xupeng Miao, Zhihao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2501.12162)  

**Abstract**: This paper introduces AdaServe, the first LLM serving system to support SLO customization through fine-grained speculative decoding. AdaServe leverages the logits of a draft model to predict the speculative accuracy of tokens and employs a theoretically optimal algorithm to construct token trees for verification. To accommodate diverse SLO requirements without compromising throughput, AdaServe employs a speculation-and-selection scheme that first constructs candidate token trees for each request and then dynamically selects tokens to meet individual SLO constraints while optimizing throughput. Comprehensive evaluations demonstrate that AdaServe achieves up to 73% higher SLO attainment and 74% higher goodput compared to state-of-the-art systems. These results underscore AdaServe's potential to enhance the efficiency and adaptability of LLM deployments across varied application scenarios. 

**Abstract (ZH)**: 本文介绍了AdaServe，这是首个通过精细推测解码支持自定义服务水平目标（SLO）的大型语言模型（LLM）服务系统。AdaServe 利用草稿模型的logits来预测推测性解码的准确度，并采用理论最优算法构建令牌树以供验证。为了满足多样化的SLO需求而不牺牲吞吐量，AdaServe 实施了一种推测与选择方案：首先为每个请求构建候选令牌树，然后动态选择令牌以满足个体的SLO约束条件并优化吞吐量。全面的评估表明，与现有最佳系统相比，AdaServe 的SLO达成率提高了73%，吞吐量提高了74%。这些结果凸显了AdaServe 在提高LLM部署效率和适应性方面的潜力，特别是在各种应用场景中。 

---
# On the practical applicability of modern DFT functionals for chemical computations. Case study of DM21 applicability for geometry optimization 

**Title (ZH)**: 现代密度泛函理论（DFT）函数式的实用适用性研究：DM21在几何优化中的应用案例分析 

**Authors**: Kirill Kulaev, Alexander Ryabov, Michael Medvedev, Evgeny Burnaev, Vladimir Vanovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2501.12149)  

**Abstract**: Density functional theory (DFT) is probably the most promising approach for quantum chemistry calculations considering its good balance between calculations precision and speed. In recent years, several neural network-based functionals have been developed for exchange-correlation energy approximation in DFT, DM21 developed by Google Deepmind being the most notable between them. This study focuses on evaluating the efficiency of DM21 functional in predicting molecular geometries, with a focus on the influence of oscillatory behavior in neural network exchange-correlation functionals. We implemented geometry optimization in PySCF for the DM21 functional in geometry optimization problem, compared its performance with traditional functionals, and tested it on various benchmarks. Our findings reveal both the potential and the current challenges of using neural network functionals for geometry optimization in DFT. We propose a solution extending the practical applicability of such functionals and allowing to model new substances with their help. 

**Abstract (ZH)**: 密度泛函理论（DFT）可能是量子化学计算中最有前景的方法之一，因为它在计算精度和速度之间取得了良好的平衡。近年来，已经开发了几种基于神经网络的功能函数来近似交换相关能量，其中由Google DeepMind开发的DM21功能函数最为突出。本研究旨在评估DM21功能函数在预测分子几何结构方面的效率，重点关注神经网络交换相关功能函数中的振荡行为对几何优化的影响。我们使用PySCF实现了DM21功能函数的几何优化，并将其性能与传统功能函数进行了比较，并在各种基准测试上进行了测试。我们的研究结果揭示了使用神经网络功能函数进行DFT几何优化的潜力和当前挑战。我们提出了一种解决方案，以扩大这类功能函数的实际应用范围，并允许使用它们来建模新的物质。 

---
# Improving Influence-based Instruction Tuning Data Selection for Balanced Learning of Diverse Capabilities 

**Title (ZH)**: 基于影响力的指令调优数据选择改进：促进多样能力平衡学习 

**Authors**: Qirun Dai, Dylan Zhang, Jiaqi W. Ma, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2501.12147)  

**Abstract**: Selecting appropriate training data is crucial for effective instruction fine-tuning of large language models (LLMs), which aims to (1) elicit strong capabilities, and (2) achieve balanced performance across a diverse range of tasks. Influence-based methods show promise in achieving (1) by estimating the contribution of each training example to the model's predictions, but often struggle with (2). Our systematic investigation reveals that this underperformance can be attributed to an inherent bias where certain tasks intrinsically have greater influence than others. As a result, data selection is often biased towards these tasks, not only hurting the model's performance on others but also, counterintuitively, harms performance on these high-influence tasks themselves.
As a remedy, we propose BIDS, a Balanced and Influential Data Selection algorithm. BIDS first normalizes influence scores of the training data, and then iteratively balances data selection by choosing the training example with the highest influence on the most underrepresented task. Experiments with both Llama-3 and Mistral-v0.3 on seven benchmarks spanning five diverse capabilities show that BIDS consistently outperforms both state-of-the-art influence-based algorithms and other non-influence-based selection frameworks. Surprisingly, training on a 15% subset selected by BIDS can even outperform full-dataset training with a much more balanced performance. Our analysis further highlights the importance of both instance-level normalization and iterative optimization of selected data for balanced learning of diverse capabilities. 

**Abstract (ZH)**: 选择合适的训练数据对于大型语言模型（LLMs）的有效指令微调至关重要，其目标是（1）激发强大的能力，以及（2）在广泛的任务范围内实现均衡的表现。基于影响的方法通过估计每个训练样本对模型预测的贡献显示出实现（1）的潜力，但常常在实现（2）方面遇到困难。我们的系统性研究揭示，这种表现不佳可以归因于一种固有的偏见，即某些任务本质上比其他任务有更大的影响。因此，数据选择通常偏向这些任务，不仅损害了模型在其他任务上的表现，而且出人意料地也损害了这些高影响任务本身的表现。

为解决这一问题，我们提出了一种平衡且有影响力的样本选择算法——BIDS（Balanced and Influential Data Selection）。BIDS 首先对训练数据的影响评分进行归一化处理，然后通过选择对最不足代表的任务具有最高影响的训练样本来迭代地平衡数据选择。在对 Llama-3 和 Mistral-v0.3 进行的涵盖五种不同能力的七项基准测试实验中，BIDS 一致性地优于最先进的基于影响的算法以及其他非基于影响的选择框架。令人惊讶的是，使用 BIDS 选择的 15% 数据子集进行训练，其表现甚至可能超过使用完整数据集进行训练的平衡性表现。我们的进一步分析强调了实例级归一化和选定数据的迭代优化对于掌握多样化能力的重要性。 

---
# FedCLEAN: byzantine defense by CLustering Errors of Activation maps in Non-IID federated learning environments 

**Title (ZH)**: FedCLEAN：在非一致分布的联邦学习环境中通过聚类激活图的错误进行拜占庭防护 

**Authors**: Mehdi Ben Ghali, Reda Bellafqira, Gouenou Coatrieux  

**Link**: [PDF](https://arxiv.org/pdf/2501.12123)  

**Abstract**: Federated Learning (FL) enables clients to collaboratively train a global model using their local datasets while reinforcing data privacy. However, FL is susceptible to poisoning attacks. Existing defense mechanisms assume that clients' data are independent and identically distributed (IID), making them ineffective in real-world applications where data are non-IID. This paper presents FedCLEAN, the first defense capable of filtering attackers' model updates in a non-IID FL environment. The originality of FedCLEAN is twofold. First, it relies on a client confidence score derived from the reconstruction errors of each client's model activation maps for a given trigger set, with reconstruction errors obtained by means of a Conditional Variational Autoencoder trained according to a novel server-side strategy. Second, we propose an ad-hoc trust propagation algorithm based on client scores, which allows building a cluster of benign clients while flagging potential attackers. Experimental results on the datasets MNIST and FashionMNIST demonstrate the robustness of FedCLEAN against Byzantine attackers in non-IID scenarios and a close-to-zero benign client misclassification rate, even in the absence of an attack. 

**Abstract (ZH)**: 联邦学习（FL）使客户端能够在保护数据隐私的同时，使用各自的本地数据集协作训练全局模型。然而，FL 对抗注入式攻击（poisoning attacks）较为脆弱。现有的防御机制假设客户端的数据是独立且同分布的（IID），这使得它们在实际应用中面对非IID数据时效果不佳。本文提出了FedCLEAN，这是首个能够在非IID联邦学习环境中过滤攻击者模型更新的防御机制。FedCLEAN的独特性体现在两个方面。首先，它依赖于从给定触发集的每个客户端模型激活图的重构误差中计算出的客户端置信度评分，重构误差通过一种根据新颖的服务器端策略训练的条件变分自编码器（Conditional Variational Autoencoder, CVAE）获得。其次，我们提出了一个基于客户端评分的自适应信任传播算法，该算法能够构建良性客户端的集群，并标识潜在的攻击者。实验结果表明，FedCLEAN 在非IID 场景下能够有效抵御拜占庭式攻击者，且良性客户端的误分类率接近于零，即使在没有攻击的情况下也是如此。 

---
# Efficient PINNs: Multi-Head Unimodular Regularization of the Solutions Space 

**Title (ZH)**: 高效的PINNs：解空间的多头单模正则化 

**Authors**: Pedro Tarancón-Álvarez, Pablo Tejerina-Pérez, Raul Jimenez, Pavlos Protopapas  

**Link**: [PDF](https://arxiv.org/pdf/2501.12116)  

**Abstract**: We present a machine learning framework to facilitate the solution of nonlinear multiscale differential equations and, especially, inverse problems using Physics-Informed Neural Networks (PINNs). This framework is based on what is called multihead (MH) training, which involves training the network to learn a general space of all solutions for a given set of equations with certain variability, rather than learning a specific solution of the system. This setup is used with a second novel technique that we call Unimodular Regularization (UR) of the latent space of solutions. We show that the multihead approach, combined with the regularization, significantly improves the efficiency of PINNs by facilitating the transfer learning process thereby enabling the finding of solutions for nonlinear, coupled, and multiscale differential equations. 

**Abstract (ZH)**: 我们提出了一种机器学习框架，以促进非线性多尺度偏微分方程及其逆问题的求解，特别是在使用物理知情神经网络（PINNs）方面。该框架基于所谓的多头（MH）训练方法，该方法使网络能够学习给定方程集在一个特定变异性条件下的所有可能解的空间，而不是仅学习该系统的特定解。此外，我们采用了一种新颖的技术，称为解空间的模单正则化（Unimodular Regularization, UR），将这种设置用于该框架。研究表明，结合模单正则化，多头方法显著提高了PINNs的效率，通过促进迁移学习过程，使我们能够找到非线性、耦合和多尺度偏微分方程的解。 

---
# Can open source large language models be used for tumor documentation in Germany? -- An evaluation on urological doctors' notes 

**Title (ZH)**: 开源大规模语言模型是否可以用于德国的肿瘤记录？--对泌尿科医生笔记的评估 

**Authors**: Stefan Lenz, Arsenij Ustjanzew, Marco Jeray, Torsten Panholzer  

**Link**: [PDF](https://arxiv.org/pdf/2501.12106)  

**Abstract**: Tumor documentation in Germany is largely done manually, requiring reading patient records and entering data into structured databases. Large language models (LLMs) could potentially enhance this process by improving efficiency and reliability. This evaluation tests eleven different open source LLMs with sizes ranging from 1-70 billion model parameters on three basic tasks of the tumor documentation process: identifying tumor diagnoses, assigning ICD-10 codes, and extracting the date of first diagnosis. For evaluating the LLMs on these tasks, a dataset of annotated text snippets based on anonymized doctors' notes from urology was prepared. Different prompting strategies were used to investigate the effect of the number of examples in few-shot prompting and to explore the capabilities of the LLMs in general. The models Llama 3.1 8B, Mistral 7B, and Mistral NeMo 12 B performed comparably well in the tasks. Models with less extensive training data or having fewer than 7 billion parameters showed notably lower performance, while larger models did not display performance gains. Examples from a different medical domain than urology could also improve the outcome in few-shot prompting, which demonstrates the ability of LLMs to handle tasks needed for tumor documentation. Open source LLMs show a strong potential for automating tumor documentation. Models from 7-12 billion parameters could offer an optimal balance between performance and resource efficiency. With tailored fine-tuning and well-designed prompting, these models might become important tools for clinical documentation in the future. The code for the evaluation is available from this https URL. We also release the dataset as a new valuable resource that addresses the shortage of authentic and easily accessible benchmarks in German-language medical NLP. 

**Abstract (ZH)**: 德国的肿瘤记录主要依赖手工完成，需要阅读患者记录并将其数据录入到结构化的数据库中。大型语言模型（LLMs）有可能通过提高效率和可靠性来优化这一过程。本次评估测试了11种不同开源LLMs，其模型参数范围从1亿到70亿个，针对肿瘤记录过程中的三个基本任务进行了测试：识别肿瘤诊断、分配ICD-10代码以及提取初次诊断日期。为了评估这些任务中LLMs的表现，基于匿名化医生笔记的数据集被用于标注文本片段。不同的提示策略被使用，以研究少样本提示中示例数量的影响，并探索LLMs的一般能力。在这些任务中，Llama 3.1 8B、Mistral 7B和Mistral NeMo 12 B等模型表现得相当优秀。拥有较少训练数据或参数少于7亿的模型表现明显较差，而更大规模的模型并未表现出性能提升。来自医学其他领域不同的训练示例也可能在少样本提示中提高结果，这表明LLMs能够处理进行肿瘤记录所需的任务。开源LLMs显示出自动化的肿瘤记录的强潜力。参数在7亿到12亿之间的模型可能在性能和资源效率之间提供最佳平衡。通过定制微调和精心设计的提示，这些模型未来可能成为临床记录的重要工具。评估的代码可以从该链接下载：[此处插入链接]。我们还发布了该数据集，作为在德语医学自然语言处理中缺乏真实且易于访问基准资源的新有价值资源。 

---
# Teacher Encoder-Student Decoder Denoising Guided Segmentation Network for Anomaly Detection 

**Title (ZH)**: 教师编码器-学生解码器去噪引导分割网络在异常检测中的应用 

**Authors**: ShiXuan Song, Hao Chen, Shu Hu, Xin Wang, Jinrong Hu, Xi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12104)  

**Abstract**: Visual anomaly detection is a highly challenging task, often categorized as a one-class classification and segmentation problem. Recent studies have demonstrated that the student-teacher (S-T) framework effectively addresses this challenge. However, most S-T frameworks rely solely on pre-trained teacher networks to guide student networks in learning multi-scale similar features, overlooking the potential of the student networks to enhance learning through multi-scale feature fusion. In this study, we propose a novel model named PFADSeg, which integrates a pre-trained teacher network, a denoising student network with multi-scale feature fusion, and a guided anomaly segmentation network into a unified framework. By adopting a unique teacher-encoder and student-decoder denoising mode, the model improves the student network's ability to learn from teacher network features. Furthermore, an adaptive feature fusion mechanism is introduced to train a self-supervised segmentation network that synthesizes anomaly masks autonomously, significantly increasing detection performance. Evaluated on the MVTec AD dataset, PFADSeg achieves state-of-the-art results with an image-level AUC of 98.9%, a pixel-level mean precision of 76.4%, and an instance-level mean precision of 78.7%. 

**Abstract (ZH)**: 视觉异常检测是一项极为具有挑战性的任务，通常被归类为一类分类和分割问题。最近的研究表明，学生-教师（S-T）框架有效应对了这一挑战。然而，大多数S-T框架仅依赖于预训练的教师网络来指导学生网络学习多尺度相似特征，忽视了学生网络通过多尺度特征融合增强学习的潜在能力。在此项研究中，我们提出了一种名为PFADSeg的新模型，该模型将预训练教师网络、多尺度特征融合的去噪学生网络和指导异常分割网络集成到一个统一框架中。通过采用独特的教师编码器和学生解码器去噪模式，该模型增强了学生网络从教师网络特征中学习的能力。此外，引入了一种自适应特征融合机制，用于训练一个自监督分割网络，该网络能够自主合成异常掩码，显著提高了检测性能。在MVTec AD数据集上的评估表明，PFADSeg实现了最先进的结果，图像级AUC为98.9%，像素级均值精度为76.4%，实例级均值精度为78.7%。 

---
# Proxies for Distortion and Consistency with Applications for Real-World Image Restoration 

**Title (ZH)**: 用于实际图像恢复的应用中的失真和一致性代理 

**Authors**: Sean Man, Guy Ohayon, Ron Raphaeli, Michael Elad  

**Link**: [PDF](https://arxiv.org/pdf/2501.12102)  

**Abstract**: Real-world image restoration deals with the recovery of images suffering from an unknown degradation. This task is typically addressed while being given only degraded images, without their corresponding ground-truth versions. In this hard setting, designing and evaluating restoration algorithms becomes highly challenging. This paper offers a suite of tools that can serve both the design and assessment of real-world image restoration algorithms. Our work starts by proposing a trained model that predicts the chain of degradations a given real-world measured input has gone through. We show how this estimator can be used to approximate the consistency -- the match between the measurements and any proposed recovered image. We also use this estimator as a guiding force for the design of a simple and highly-effective plug-and-play real-world image restoration algorithm, leveraging a pre-trained diffusion-based image prior. Furthermore, this work proposes no-reference proxy measures of MSE and LPIPS, which, without access to the ground-truth images, allow ranking of real-world image restoration algorithms according to their (approximate) MSE and LPIPS. The proposed suite provides a versatile, first of its kind framework for evaluating and comparing blind image restoration algorithms in real-world scenarios. 

**Abstract (ZH)**: 现实世界中的图像恢复涉及对遭受未知退化影响的图像进行恢复。在通常情况下，仅提供退化的图像而没有其对应的Ground-Truth版本。在这种困难的环境中，设计和评价恢复算法变得极其具有挑战性。本文提供了一系列工具，可用于设计和评估现实世界中的图像恢复算法。我们的工作首先提出了一种训练模型，该模型可以预测给定的实际测量输入所经历的退化链。我们展示了如何利用该估计算法来近似一致性——即测量值与任何建议恢复图像之间的匹配程度。此外，我们还将该估计算法作为设计简单而高效的插件式现实世界图像恢复算法的指导工具，该算法利用预训练的基于扩散的图像先验。此外，本文还提出无参考的MSE和LPIPS的代理度量，这些度量在没有访问Ground-Truth图像的情况下，可以按照它们的（近似）MSE和LPIPS对现实世界的图像恢复算法进行排名。所提出的一系列工具提供了一个多功能且首创的框架，用于评估和比较在实际场景中的盲图像恢复算法。 

---
# Scalable Whole Slide Image Representation Using K-Mean Clustering and Fisher Vector Aggregation 

**Title (ZH)**: 使用K均值聚类和Fisher矢量聚合的大规模全切片图像表示方法 

**Authors**: Ravi Kant Gupta, Shounak Das, Ardhendu Sekhar, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12085)  

**Abstract**: Whole slide images (WSIs) are high-resolution, gigapixel sized images that pose significant computational challenges for traditional machine learning models due to their size and this http URL this paper, we present a scalable and efficient methodology for WSI classification by leveraging patch-based feature extraction, clustering, and Fisher vector encoding. Initially, WSIs are divided into fixed size patches, and deep feature embeddings are extracted from each patch using a pre-trained convolutional neural network (CNN). These patch-level embeddings are subsequently clustered using K-means clustering, where each cluster aggregates semantically similar regions of the WSI. To effectively summarize each cluster, Fisher vector representations are computed by modeling the distribution of patch embeddings in each cluster as a parametric Gaussian mixture model (GMM). The Fisher vectors from each cluster are concatenated into a high-dimensional feature vector, creating a compact and informative representation of the entire WSI. This feature vector is then used by a classifier to predict the WSI's diagnostic label. Our method captures local and global tissue structures and yields robust performance for large-scale WSI classification, demonstrating superior accuracy and scalability compared to other approaches. 

**Abstract (ZH)**: 全切片图像 (WSIs) 是高分辨率的大图，在传统机器学习模型中由于其尺寸巨大而提出了显著的计算挑战。本论文中，我们通过利用基于块的特征提取、聚类和 Fisher 向量编码提出了一种可扩展且高效的 WSI 分类方法。首先，WSIs 被分割为固定大小的块，并使用预训练的卷积神经网络 (CNN) 从每个块中提取深层特征嵌入。随后，这些块级嵌入使用 K-均值聚类进行聚类，其中每个聚类会聚集 WSIs 中语义相似的区域。为了有效地总结每个聚类，我们通过将块嵌入在每个聚类中的分布建模为参数化的高斯混合模型 (GMM) 来计算 Fisher 向量表示。每个聚类的 Fisher 向量被连接起来形成一个高维特征向量，创建了整个 WSI 的紧凑且具有信息性的表示。此特征向量随后由分类器用于预测 WSI 的诊断标签。我们的方法捕获局部和全局组织结构，并在大规模 WSI 分类中表现出稳健的性能，其准确性和可扩展性优于其他方法。 

---
# EDoRA: Efficient Weight-Decomposed Low-Rank Adaptation via Singular Value Decomposition 

**Title (ZH)**: EDoRA: 高效的基于奇异值分解的权重分解低秩适应方法 

**Authors**: Hamid Nasiri, Peter Garraghan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12067)  

**Abstract**: Parameter-efficient fine-tuning methods, such as LoRA, reduces the number of trainable parameters. However, they often suffer from scalability issues and differences between their learning pattern and full fine-tuning. To overcome these limitations, we propose Efficient Weight-Decomposed Low-Rank Adaptation (EDoRA): a novel PEFT method that decomposes pre-trained weights into magnitude and directional components. By freezing low-rank matrices, initializing them by singular value decomposition, and introducing a small trainable matrix between them, EDoRA achieves substantial reduction in trainable parameters while maintaining learning capacity. Experimental results on the GLUE benchmark demonstrate that EDoRA achieves competitive or superior performance compared to state-of-the-art methods, such as LoRA and DoRA, with up to 30x fewer trainable parameters. This makes EDoRA a highly efficient solution for adapting LLMs to diverse tasks under memory-constrained settings. Code is available at this https URL . 

**Abstract (ZH)**: 参数高效的微调方法，如LoRA，减少了可训练参数的数量。然而，这些方法往往面临可扩展性问题以及学习模式与完全微调之间的差异。为了解决这些问题，我们提出了高效分解低秩适应（EDoRA）：一种新颖的参数高效微调（PEFT）方法，将预训练权重分解为幅度和方向分量。通过冻结低秩矩阵、利用奇异值分解初始化它们，并在它们之间引入一个小型可训练矩阵，EDoRA 实现了可训练参数数量的显著减少，同时保持了学习能力。在GLUE基准上的实验结果表明，与当前最先进的方法（如LoRA和DoRA）相比，虽然可训练参数的数量最多可减少30倍，但EDoRA仍然取得了竞争力甚至更好的性能。这使得EDoRA成为在内存受限的环境下适应大型语言模型（LLMs）到各种任务的高效解决方案。代码可在此访问：this https URL。 

---
# Adaptive Class Learning to Screen Diabetic Disorders in Fundus Images of Eye 

**Title (ZH)**: 自适应类学习在筛查眼底图像中的糖尿病疾病 

**Authors**: Shramana Dey, Pallabi Dutta, Riddhasree Bhattacharyya, Surochita Pal, Sushmita Mitra, Rajiv Raman  

**Link**: [PDF](https://arxiv.org/pdf/2501.12048)  

**Abstract**: The prevalence of ocular illnesses is growing globally, presenting a substantial public health challenge. Early detection and timely intervention are crucial for averting visual impairment and enhancing patient prognosis. This research introduces a new framework called Class Extension with Limited Data (CELD) to train a classifier to categorize retinal fundus images. The classifier is initially trained to identify relevant features concerning Healthy and Diabetic Retinopathy (DR) classes and later fine-tuned to adapt to the task of classifying the input images into three classes: Healthy, DR, and Glaucoma. This strategy allows the model to gradually enhance its classification capabilities, which is beneficial in situations where there are only a limited number of labeled datasets available. Perturbation methods are also used to identify the input image characteristics responsible for influencing the models decision-making process. We achieve an overall accuracy of 91% on publicly available datasets. 

**Abstract (ZH)**: 全球范围内眼病的发病率正在逐渐上升，这构成了重大的公共卫生挑战。早期发现和及时干预对于预防视力损伤并改善患者预后至关重要。本研究提出了一种名为有限数据扩展分类（Class Extension with Limited Data, CELD）的新框架，用于训练分类器以对视网膜底片图像进行分类。该分类器最初被训练以识别与健康和糖尿病视网膜病变（DR）相关的特征，随后进一步微调以适应将输入图像分类为三个类别：健康、糖尿病视网膜病变和青光眼的任务。这种方法允许模型在仅有限数量的标记数据集可用的情况下逐步增强其分类能力。我们还使用扰动方法来识别对模型决策过程有影响的输入图像特征。在公开可用的数据集上，我们达到了91%的整体准确率。 

---
# Harnessing Generative Pre-Trained Transformer for Datacenter Packet Trace Generation 

**Title (ZH)**: 利用生成型预训练变换器生成数据中心数据包轨迹 

**Authors**: Chen Griner  

**Link**: [PDF](https://arxiv.org/pdf/2501.12033)  

**Abstract**: Today, the rapid growth of applications reliant on datacenters calls for new advancements to meet the increasing traffic and computational demands. Traffic traces from datacenters are essential for further development and optimization of future datacenters. However, traces are rarely released to the public. Researchers often use simplified mathematical models that lack the depth needed to recreate intricate traffic patterns and, thus, miss optimization opportunities found in realistic traffic. In this preliminary work, we introduce DTG-GPT, a packet-level Datacenter Traffic Generator (DTG), based on the generative pre-trained transformer (GPT) architecture used by many state-of-the-art large language models. We train our model on a small set of available traffic traces from different domains and offer a simple methodology to evaluate the fidelity of the generated traces to their original counterparts. We show that DTG-GPT can synthesize novel traces that mimic the spatiotemporal patterns found in real traffic traces. We further demonstrate that DTG-GPT can generate traces for networks of different scales while maintaining fidelity. Our findings indicate the potential that, in the future, similar models to DTG-GPT will allow datacenter operators to release traffic information to the research community via trained GPT models. 

**Abstract (ZH)**: 当前，依赖数据中心的应用快速增长，促使我们需要不断发展新的技术来满足不断增加的流量和计算需求。数据中心的流量记录对于进一步发展和优化未来数据中心至关重要。然而，这些记录很少对外公开。研究人员通常使用简化的数学模型，这些模型缺乏足够的深度来重现复杂的流量模式，从而错失了在现实流量中发现的优化机会。在本初步研究中，我们引入了基于生成预训练变换器（GPT）架构的DTG-GPT，这是一种基于包级别的数据中心流量生成器（DTG）。我们使用不同领域的少量可用流量记录对模型进行训练，并提供了一种简单的方法来评估生成的记录与原始记录的一致性。研究表明，DTG-GPT能够合成出能模仿真实流量记录中的时空模式的新记录。我们进一步证明，DTG-GPT能够在保持一致性的前提下生成不同规模网络的流量记录。我们的研究结果表明，未来类似的DTG-GPT模型可能使数据中心运营商能够通过训练好的GPT模型向科研社区发布流量信息。 

---
# Full Proportional Justified Representation 

**Title (ZH)**: 全比例公正代表分配 

**Authors**: Yusuf Hakan Kalayci, Jiasen Liu, David Kempe  

**Link**: [PDF](https://arxiv.org/pdf/2501.12015)  

**Abstract**: In multiwinner approval voting, forming a committee that proportionally represents voters' approval ballots is an essential task. The notion of justified representation (JR) demands that any large "cohesive" group of voters should be proportionally "represented". The "cohesiveness" is defined in different ways; two common ways are the following: (C1) demands that the group unanimously approves a set of candidates proportional to its size, while (C2) requires each member to approve at least a fixed fraction of such a set. Similarly, "representation" have been considered in different ways: (R1) the coalition's collective utility from the winning set exceeds that of any proportionally sized alternative, and (R2) for any proportionally sized alternative, at least one member of the coalition derives less utility from it than from the winning set.
Three of the four possible combinations have been extensively studied: (C1)-(R1) defines Proportional Justified Representation (PJR), (C1)-(R2) defines Extended Justified Representation (EJR), (C2)-(R2) defines Full Justified Representation (FJR). All three have merits, but also drawbacks. PJR is the weakest notion, and perhaps not sufficiently demanding; EJR may not be compatible with perfect representation; and it is open whether a committee satisfying FJR can be found efficiently.
We study the combination (C2)-(R1), which we call Full Proportional Justified Representation (FPJR). We investigate FPJR's properties and find that it shares PJR's advantages over EJR: several proportionality axioms (e.g. priceability, perfect representation) imply FPJR and PJR but not EJR. We also find that efficient rules like the greedy Monroe rule and the method of equal shares satisfy FPJR, matching a key advantage of EJR over FJR. However, the Proportional Approval Voting (PAV) rule may violate FPJR, so neither of EJR and FPJR implies the other. 

**Abstract (ZH)**: 在多胜赞成投票中，形成为一个能够反映选民赞成票的委员会是一项基本任务。公正代表（JR）的概念要求任何“一致”的大型选民群体应得到相应的“代表”。群体的“一致性”被以不同的方式定义；其中两种常见的定义方式如下：(C1) 要求该群体一致地对其规模相应的候选人集给予批准，而(C2) 则要求该群体中的每个成员至少批准这种候选人的固定比例。类似地，“代表”也被以不同的方式考虑：(R1) 胜出的候选集集合的集体收益超过任何形式上规模相应的替代方案；而(R2) 对于任何形式上规模相应的替代方案，联盟中的至少一个成员从该替代方案中获得的收益不如从胜出的候选集集合中获得的收益。

这四种可能的组合中有三种已经被广泛研究：(C1)-(R1) 定义了比例公正代表 (PJR)，(C1)-(R2) 定义了扩展公正代表 (EJR)，(C2)-(R2) 定义了充分公正代表 (FJR)。这三种概念各有优势，但也存在缺点。PJR 是最弱的概念，可能不够具有挑战性；EJR 可能不与完美的代表兼容；而是否可以通过高效的方式找到一个满足 FJR 的委员会仍是一个开放问题。

我们研究了 (C2)-(R1) 的组合，我们将其称为全面比例公正代表 (FPJR)。我们探讨了 FPJR 的性质，并发现 FPJR 在某些比例公理（如价格性、完美代表）中具有 PJR 的优势，而不会扩展到 EJR。我们还发现，包括贪婪 Monroe 规则和等量份额方法在内的高效规则满足 FPJR，这与 EJR 相对 FJR 的一个关键优势相吻合。然而，比例赞成投票 (PAV) 规则可能违反 FPJR，因此 EJR 和 FPJR 互不包含。 

---
# Survey on Hand Gesture Recognition from Visual Input 

**Title (ZH)**: 视觉输入的手势识别综述 

**Authors**: Manousos Linardakis, Iraklis Varlamis, Georgios Th. Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2501.11992)  

**Abstract**: Hand gesture recognition has become an important research area, driven by the growing demand for human-computer interaction in fields such as sign language recognition, virtual and augmented reality, and robotics. Despite the rapid growth of the field, there are few surveys that comprehensively cover recent research developments, available solutions, and benchmark datasets. This survey addresses this gap by examining the latest advancements in hand gesture and 3D hand pose recognition from various types of camera input data including RGB images, depth images, and videos from monocular or multiview cameras, examining the differing methodological requirements of each approach. Furthermore, an overview of widely used datasets is provided, detailing their main characteristics and application domains. Finally, open challenges such as achieving robust recognition in real-world environments, handling occlusions, ensuring generalization across diverse users, and addressing computational efficiency for real-time applications are highlighted to guide future research directions. By synthesizing the objectives, methodologies, and applications of recent studies, this survey offers valuable insights into current trends, challenges, and opportunities for future research in human hand gesture recognition. 

**Abstract (ZH)**: 手部手势识别已成为一个重要的研究领域，由于在手语识别、虚拟现实和增强现实以及机器人技术等人机交互领域的需求不断增加。尽管该领域发展迅速，但仍缺乏全面涵盖最近研究进展、可用解决方案和基准数据集的综述文章。本综述通过探讨各种类型的摄像输入数据（包括RGB图像、深度图像及单目或多视图摄像机拍摄的视频）在手部手势识别和3D手部姿态识别方面的新进展，弥补了这一空白，并分析了每种方法的不同的方法论要求。此外，综述了广泛使用的数据集，详细介绍了它们的主要特征和应用领域。最后，指出了诸如在真实环境中的鲁棒识别、处理遮挡、在不同用户之间泛化及为实时应用解决计算效率等问题，以指导未来研究方向。通过对近期研究目的、方法论和应用的整合，本综述为未来手部手势识别研究中的趋势、挑战和机遇提供了宝贵的见解。 

---
# Leveraging Graph Structures and Large Language Models for End-to-End Synthetic Task-Oriented Dialogues 

**Title (ZH)**: 利用图结构和大规模语言模型实现端到端的任务导向合成对话 

**Authors**: Maya Medjad, Hugo Imbert, Bruno Yun, Raphaël Szymocha, Frédéric Armetta  

**Link**: [PDF](https://arxiv.org/pdf/2501.11977)  

**Abstract**: Training task-oriented dialogue systems is both costly and time-consuming, due to the need for high-quality datasets encompassing diverse intents. Traditional methods depend on extensive human annotation, while recent advancements leverage large language models (LLMs) to generate synthetic data. However, these approaches often require custom prompts or code, limiting accessibility for non-technical users. We introduce GraphTOD, an end-to-end framework that simplifies the generation of task-oriented dialogues. Users can create dialogues by specifying transition graphs in JSON format. Our evaluation demonstrates that GraphTOD generates high-quality dialogues across various domains, significantly lowering the cost and complexity of dataset creation. 

**Abstract (ZH)**: 训练面向任务的对话系统既耗时又耗费资源，因为需要涵盖多种意图的高质量数据集。传统方法依赖广泛的-human标注，而最近的进展则利用大规模语言模型（LLMs）生成合成数据。然而，这些方法通常需要自定义提示或代码，限制了非技术人员的访问。我们引入了GraphTOD，这是一个端到端的框架，简化了任务 oriented 对话的生成过程。用户可以通过指定 JSON 格式的转换图来创建对话。我们的评估表明，GraphTOD 能够在各种领域生成高质量的对话，显著降低了数据集创建的成本和复杂性。 

---
# TAD-Bench: A Comprehensive Benchmark for Embedding-Based Text Anomaly Detection 

**Title (ZH)**: TAD-Bench：基于嵌入的空间文本异常检测综合基准 

**Authors**: Yang Cao, Sikun Yang, Chen Li, Haolong Xiang, Lianyong Qi, Bo Liu, Rongsheng Li, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11960)  

**Abstract**: Text anomaly detection is crucial for identifying spam, misinformation, and offensive language in natural language processing tasks. Despite the growing adoption of embedding-based methods, their effectiveness and generalizability across diverse application scenarios remain under-explored. To address this, we present TAD-Bench, a comprehensive benchmark designed to systematically evaluate embedding-based approaches for text anomaly detection. TAD-Bench integrates multiple datasets spanning different domains, combining state-of-the-art embeddings from large language models with a variety of anomaly detection algorithms. Through extensive experiments, we analyze the interplay between embeddings and detection methods, uncovering their strengths, weaknesses, and applicability to different tasks. These findings offer new perspectives on building more robust, efficient, and generalizable anomaly detection systems for real-world applications. 

**Abstract (ZH)**: 文本异常检测对于识别自然语言处理任务中的垃圾信息、错误信息和不适当语言至关重要。尽管嵌入式方法的应用日益增多，但它们在不同应用场景中的有效性及其可推广性仍缺乏充分探索。为应对这一挑战，我们提出了TAD-Bench，这是一个全面的基准测试，旨在系统地评估嵌入式方法在文本异常检测中的应用效果。TAD-Bench 整合了多个跨领域的数据集，结合了大型语言模型的最新嵌入技术以及多种异常检测算法。通过广泛的实验，我们分析了嵌入技术和检测方法之间的相互作用，揭示了它们的优势、劣势及其在不同任务中的适用性。这些发现为构建更具鲁棒性、高效性和广泛适用性的异常检测系统提供了新的视角，适用于实际应用场景。 

---
# MeshONet: A Generalizable and Efficient Operator Learning Method for Structured Mesh Generation 

**Title (ZH)**: MeshONet：一种用于结构化网格生成的泛化高效操作学习方法 

**Authors**: Jing Xiao, Xinhai Chen, Qingling Wang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11937)  

**Abstract**: Mesh generation plays a crucial role in scientific computing. Traditional mesh generation methods, such as TFI and PDE-based methods, often struggle to achieve a balance between efficiency and mesh quality. To address this challenge, physics-informed intelligent learning methods have recently emerged, significantly improving generation efficiency while maintaining high mesh quality. However, physics-informed methods fail to generalize when applied to previously unseen geometries, as even small changes in the boundary shape necessitate burdensome retraining to adapt to new geometric variations. In this paper, we introduce MeshONet, the first generalizable intelligent learning method for structured mesh generation. The method transforms the mesh generation task into an operator learning problem with multiple input and solution functions. To effectively overcome the multivariable mapping restriction of operator learning methods, we propose a dual-branch, shared-trunk architecture to approximate the mapping between function spaces based on input-output pairs. Experimental results show that MeshONet achieves a speedup of up to four orders of magnitude in generation efficiency over traditional methods. It also enables generalization to different geometries without retraining, greatly enhancing the practicality of intelligent methods. 

**Abstract (ZH)**: 网格生成在科学计算中扮演着至关重要的角色。传统的网格生成方法，如Tolliver-Fox-Irons (TFI)方法和基于偏微分方程（PDE）的方法，往往难以在效率和网格质量之间取得平衡。为了解决这一挑战，近期出现了基于物理信息的智能学习方法，这些方法显著提高了生成效率，同时保持了高网格质量。然而，基于物理信息的方法在应用于未见过的几何形状时缺乏泛化能力，即使边界形状发生微小变化，也需要重新训练以适应新的几何变体。在本文中，我们介绍了MeshONet，这是首个用于结构网格生成的可泛化的智能学习方法。该方法将网格生成任务转化为具有多个输入和解函数的算子学习问题。为有效克服算子学习方法中的多变量映射限制，我们提出了一种双分支、共享主干架构，基于输入-输出对来近似函数空间之间的映射。实验结果表明，MeshONet在生成效率上相比传统方法提高了四数量级。此外，它还能够在无需重新训练的情况下泛化到不同的几何形状，极大地增强了智能方法的实际应用性。 

---
# Webvs. LLMs: An Empirical Study of Learning Behaviors of CS2 Students 

**Title (ZH)**: 网络与大语言模型：CS2学生学习行为的一项实证研究 

**Authors**: Aayush Kumar, Daniel Prol, Amin Alipour, Sruti Srinivasa Ragavan  

**Link**: [PDF](https://arxiv.org/pdf/2501.11935)  

**Abstract**: LLMs such as ChatGPT have been widely adopted by students in higher education as tools for learning programming and related concepts. However, it remains unclear how effective students are and what strategies students use while learning with LLMs. Since the majority of students' experiences in online self-learning have come through using search engines such as Google, evaluating AI tools in this context can help us address these gaps. In this mixed methods research, we conducted an exploratory within-subjects study to understand how CS2 students learn programming concepts using both LLMs as well as traditional online methods such as educational websites and videos to examine how students approach learning within and across both scenarios. We discovered that students found it easier to learn a more difficult concept using traditional methods than using ChatGPT. We also found that students ask fewer follow-ups and use more keyword-based queries for search engines while their prompts to LLMs tend to explicitly ask for information. 

**Abstract (ZH)**: 像ChatGPT这样的大型语言模型（LLMs）在高等教育学生中被广泛用作学习编程及相关概念的工具。然而，仍不清楚学生在使用LLMs过程中有多大成效，以及学生采用什么样的策略进行学习。由于大多数学生在线自主学习的经验多来自使用如Google这样的搜索引擎，因此评估在此情境下的AI工具有助于解决这些问题。在本项混合方法研究中，我们开展了探索性的单被试设计研究，旨在理解学生在使用LLMs和传统在线方法（如教育网站和视频）学习编程概念时的不同方式，以探讨学生在两者中的学习策略。研究发现，学生在使用传统方法学习更难的概念时比使用ChatGPT更为容易。同时发现，学生在对搜索引擎提问时较少进行后续提问，并更多地使用关键词查询，而向LLMs提问时则更倾向于明确地寻求信息。 

---
# A Lightweight and Interpretable Deepfakes Detection Framework 

**Title (ZH)**: 一种轻量级且可解释的深度伪造检测框架 

**Authors**: Muhammad Umar Farooq, Ali Javed, Khalid Mahmood Malik, Muhammad Anas Raza  

**Link**: [PDF](https://arxiv.org/pdf/2501.11927)  

**Abstract**: The recent realistic creation and dissemination of so-called deepfakes poses a serious threat to social life, civil rest, and law. Celebrity defaming, election manipulation, and deepfakes as evidence in court of law are few potential consequences of deepfakes. The availability of open source trained models based on modern frameworks such as PyTorch or TensorFlow, video manipulations Apps such as FaceApp and REFACE, and economical computing infrastructure has easen the creation of deepfakes. Most of the existing detectors focus on detecting either face-swap, lip-sync, or puppet master deepfakes, but a unified framework to detect all three types of deepfakes is hardly explored. This paper presents a unified framework that exploits the power of proposed feature fusion of hybrid facial landmarks and our novel heart rate features for detection of all types of deepfakes. We propose novel heart rate features and fused them with the facial landmark features to better extract the facial artifacts of fake videos and natural variations available in the original videos. We used these features to train a light-weight XGBoost to classify between the deepfake and bonafide videos. We evaluated the performance of our framework on the world leaders dataset (WLDR) that contains all types of deepfakes. Experimental results illustrate that the proposed framework offers superior detection performance over the comparative deepfakes detection methods. Performance comparison of our framework against the LSTM-FCN, a candidate of deep learning model, shows that proposed model achieves similar results, however, it is more interpretable. 

**Abstract (ZH)**: 近年来，所谓的“深度造假”在真实场景中的创建和传播对社会生活、公民稳定和法律构成了严重威胁。名人诽谤、选举操纵以及深度造假作为法庭证据均是深度造假可能带来的后果。基于现代框架（如PyTorch或TensorFlow）的开源训练模型的可用性，视频编辑软件（如FaceApp和REFACE）的普及，以及经济高效的计算基础设施，使得深度造假的制作变得更加容易。现有的大多数检测方法专注于检测面部替换、口型同步或傀儡师类型的深度造假，但综合检测这三种类型深度造假的统一框架鲜少被探索。本文提出了一种统一框架，利用混合面部特征点和我们提出的新型心率特征的深度融合，以检测各种类型的深度造假。我们提出了新型心率特征，并将其与面部特征点特征融合，以更好地提取假视频中的面部伪影及原始视频中的自然变异。我们使用这些特征训练了一种轻量级的XGBoost分类器，用于将假视频和真视频分类。我们使用世界领导者数据集（WLDR）对该框架进行了评估，该数据集包含各种类型的深度造假。实验结果表明，所提出框架在深度造假检测性能上优于比较方法。与一种深度学习模型（LSTM-FCN）框架的性能对比结果显示，所提出模型的性能相似，但更具可解释性。 

---
# Goal-oriented Transmission Scheduling: Structure-guided DRL with a Unified Dual On-policy and Off-policy Approach 

**Title (ZH)**: 面向目标的传输调度：基于结构引导的统一双端在线与离线策略强化学习方法 

**Authors**: Jiazheng Chen, Wanchun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11921)  

**Abstract**: Goal-oriented communications prioritize application-driven objectives over data accuracy, enabling intelligent next-generation wireless systems. Efficient scheduling in multi-device, multi-channel systems poses significant challenges due to high-dimensional state and action spaces. We address these challenges by deriving key structural properties of the optimal solution to the goal-oriented scheduling problem, incorporating Age of Information (AoI) and channel states. Specifically, we establish the monotonicity of the optimal state value function (a measure of long-term system performance) w.r.t. channel states and prove its asymptotic convexity w.r.t. AoI states. Additionally, we derive the monotonicity of the optimal policy w.r.t. channel states, advancing the theoretical framework for optimal scheduling. Leveraging these insights, we propose the structure-guided unified dual on-off policy DRL (SUDO-DRL), a hybrid algorithm that combines the stability of on-policy training with the sample efficiency of off-policy methods. Through a novel structural property evaluation framework, SUDO-DRL enables effective and scalable training, addressing the complexities of large-scale systems. Numerical results show SUDO-DRL improves system performance by up to 45% and reduces convergence time by 40% compared to state-of-the-art methods. It also effectively handles scheduling in much larger systems, where off-policy DRL fails and on-policy benchmarks exhibit significant performance loss, demonstrating its scalability and efficacy in goal-oriented communications. 

**Abstract (ZH)**: 面向目标的通信优先考虑应用驱动的目标而非数据准确性，从而能够推动智能下一代无线系统的进步。在多设备、多信道系统中，高效调度会面临因高维状态和动作空间而产生的重大挑战。为应对这些挑战，我们通过推导面向目标调度问题最优解的关键结构性性质来解决这些问题，并将年龄因子（AoI）和信道状态纳入考虑。具体而言，我们建立了最优状态值函数（衡量长期系统性能的一个指标）关于信道状态的单调性，并证明其关于年龄状态的渐近凸性。此外，我们推导了最优策略关于信道状态的单调性，从而推进了最优调度的理论框架。基于这些见解，我们提出了结构引导统一双关开关策略深度强化学习算法（SUDO-DRL），这是一种结合了开策略训练的稳定性和离策略方法的样本效率的混合算法。通过一个新颖的结构性质评估框架，SUDO-DRL 实现了有效的可扩展训练，解决了大规模系统的复杂性。数值结果表明，与最先进的方法相比，SUDO-DRL 可将系统性能提高多达45%，并将收敛时间缩短40%。此外，它还能够有效处理更大规模的调度问题，在这些更大系统中，离策略的DRL算法会失效，而开策略的基准算法会产生显著性能损失，这进一步证明了SUDO-DRL在面向目标通信中的可扩展性和有效性。 

---
# LuxVeri at GenAI Detection Task 3: Cross-Domain Detection of AI-Generated Text Using Inverse Perplexity-Weighted Ensemble of Fine-Tuned Transformer Models 

**Title (ZH)**: LuxVeri 在 GenAI 识别任务 3 中的跨域生成文本识别：逆困惑度加权集成细调变换器模型 

**Authors**: Md Kamrujjaman Mobin, Md Saiful Islam  

**Link**: [PDF](https://arxiv.org/pdf/2501.11918)  

**Abstract**: This paper presents our approach for Task 3 of the GenAI content detection workshop at COLING-2025, focusing on Cross-Domain Machine-Generated Text (MGT) Detection. We propose an ensemble of fine-tuned transformer models, enhanced by inverse perplexity weighting, to improve classification accuracy across diverse text domains. For Subtask A (Non-Adversarial MGT Detection), we combined a fine-tuned RoBERTa-base model with an OpenAI detector-integrated RoBERTa-base model, achieving an aggregate TPR score of 0.826, ranking 10th out of 23 detectors. In Subtask B (Adversarial MGT Detection), our fine-tuned RoBERTa-base model achieved a TPR score of 0.801, securing 8th out of 22 detectors. Our results demonstrate the effectiveness of inverse perplexity-based weighting for enhancing generalization and performance in both non-adversarial and adversarial MGT detection, highlighting the potential for transformer models in cross-domain AI-generated content detection. 

**Abstract (ZH)**: 本文介绍了我们在COLING-2025 GenAI内容检测研讨会中Task 3的方法，重点关注跨域机器生成文本（MGT）检测。我们提出了一种融合了逆困惑度加权的微调变换器模型集合，以提高不同文本域的分类准确性。在子任务A（无对抗MGT检测）中，我们结合了微调的RoBERTa-base模型和OpenAI检测器集成的RoBERTa-base模型，获得了综合TPR得分为0.826，排名23个检测器中的第10位。在子任务B（对抗MGT检测）中，我们微调的RoBERTa-base模型获得了TPR得分为0.801，排名22个检测器中的第8位。实验结果表明，逆困惑度加权方法在非对抗和对抗MGT检测中的广泛应用性和性能提升效果明显，突显了变换器模型在跨域AI生成内容检测方面的能力。 

---
# LuxVeri at GenAI Detection Task 1: Inverse Perplexity Weighted Ensemble for Robust Detection of AI-Generated Text across English and Multilingual Contexts 

**Title (ZH)**: LuxVeri 在生成式AI检测任务1中的逆困惑度加权集成方法：在多语言上下文中对AI生成文本的稳健检测 

**Authors**: Md Kamrujjaman Mobin, Md Saiful Islam  

**Link**: [PDF](https://arxiv.org/pdf/2501.11914)  

**Abstract**: This paper presents a system developed for Task 1 of the COLING 2025 Workshop on Detecting AI-Generated Content, focusing on the binary classification of machine-generated versus human-written text. Our approach utilizes an ensemble of models, with weights assigned according to each model's inverse perplexity, to enhance classification accuracy. For the English text detection task, we combined RoBERTa-base, RoBERTa-base with the OpenAI detector, and BERT-base-cased, achieving a Macro F1-score of 0.7458, which ranked us 12th out of 35 teams. We ensembled RemBERT, XLM-RoBERTa-base, and BERT-base-multilingual-case for the multilingual text detection task, employing the same inverse perplexity weighting technique. This resulted in a Macro F1-score of 0.7513, positioning us 4th out of 25 teams. Our results demonstrate the effectiveness of inverse perplexity weighting in improving the robustness of machine-generated text detection across both monolingual and multilingual settings, highlighting the potential of ensemble methods for this challenging task. 

**Abstract (ZH)**: 本文介绍了一个为Coling 2025 Workshop on Detecting AI-Generated Content的任务1开发的系统，重点关注机器生成文本与人类撰写文本的二分类问题。我们的方法利用了多个模型的集成，并根据每个模型的逆困惑度分配权重，以提高分类准确性。对于英语文本检测任务，我们结合了RoBERTa-base、带有OpenAI检测器的RoBERTa-base以及BERT-base-cased，实现了宏F1分数0.7458，这使我们在35支队伍中排名第12位。对于多语言文本检测任务，我们采用了RemBERT、XLM-RoBERTa-base和BERT-base-multilingual-case的集成，并使用相同的逆困惑度加权技术，实现了宏F1分数0.7513，位列25支队伍中的第4位。我们的结果表明，逆困惑度加权在提高机器生成文本检测的鲁棒性方面具有有效性，不仅适用于单语言环境，也适用于多语言环境，并强调了集成方法在这个具有挑战性的任务中的潜力。 

---
# Panoramic Interests: Stylistic-Content Aware Personalized Headline Generation 

**Title (ZH)**: 全景兴趣：风格-内容兼顾的个性化标题生成 

**Authors**: Junhong Lian, Xiang Ao, Xinyu Liu, Yang Liu, Qing He  

**Link**: [PDF](https://arxiv.org/pdf/2501.11900)  

**Abstract**: Personalized news headline generation aims to provide users with attention-grabbing headlines that are tailored to their preferences. Prevailing methods focus on user-oriented content preferences, but most of them overlook the fact that diverse stylistic preferences are integral to users' panoramic interests, leading to suboptimal personalization. In view of this, we propose a novel Stylistic-Content Aware Personalized Headline Generation (SCAPE) framework. SCAPE extracts both content and stylistic features from headlines with the aid of large language model (LLM) collaboration. It further adaptively integrates users' long- and short-term interests through a contrastive learning-based hierarchical fusion network. By incorporating the panoramic interests into the headline generator, SCAPE reflects users' stylistic-content preferences during the generation process. Extensive experiments on the real-world dataset PENS demonstrate the superiority of SCAPE over baselines. 

**Abstract (ZH)**: 个性化新闻标题生成旨在为用户提供符合其兴趣的、能吸引注意力的新闻标题。现有方法主要关注用户的内容偏好，但大多忽略了用户广泛的兴趣中包含多样化风格偏好这一事实，从而导致个性化不足。为解决这一问题，我们提出了一种新颖的风格-内容感知个性化标题生成（SCAPE）框架。SCAPE借助大型语言模型（LLM）的协作，提取标题的内容和风格特征，并通过基于对比学习的层次融合网络，进一步适应性地整合用户的长短期兴趣。通过将用户的宏观兴趣纳入标题生成过程，SCAPE在生成过程中反映了用户的风格-内容偏好。在真实世界数据集PENS上的广泛实验表明，SCAPE在性能上优于基线方法。 

---
# Community-Aware Temporal Walks: Parameter-Free Representation Learning on Continuous-Time Dynamic Graphs 

**Title (ZH)**: 基于社区意识的时间游走：连续时间动态图上的参数无关表示学习 

**Authors**: He Yu, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11880)  

**Abstract**: Dynamic graph representation learning plays a crucial role in understanding evolving behaviors. However, existing methods often struggle with flexibility, adaptability, and the preservation of temporal and structural dynamics. To address these issues, we propose Community-aware Temporal Walks (CTWalks), a novel framework for representation learning on continuous-time dynamic graphs. CTWalks integrates three key components: a community-based parameter-free temporal walk sampling mechanism, an anonymization strategy enriched with community labels, and an encoding process that leverages continuous temporal dynamics modeled via ordinary differential equations (ODEs). This design enables precise modeling of both intra- and inter-community interactions, offering a fine-grained representation of evolving temporal patterns in continuous-time dynamic graphs. CTWalks theoretically overcomes locality bias in walks and establishes its connection to matrix factorization. Experiments on benchmark datasets demonstrate that CTWalks outperforms established methods in temporal link prediction tasks, achieving higher accuracy while maintaining robustness. 

**Abstract (ZH)**: 动态图表示学习在理解演化行为中起着关键作用。然而，现有方法往往在灵活性、适应性和时间与结构动力学的保留方面存在困难。为了解决这些问题，我们提出了一种新型的连续时间动态图表示学习框架：社区感知时序走动 (CTWalks)。CTWalks 集成了三个关键组件：基于社区的无参时序走动采样机制、增强有社区标签的匿名化策略以及利用常微分方程（ODEs）建模的连续时间动力学的编码过程。这种设计使CTWalks能够精确建模社区内和社区间的互动，提供连续时间动态图中演化时间模式的细致表示。CTWalks 理论上克服了走动中的局部偏差，并建立了其与矩阵分解的联系。在基准数据集上的实验表明，CTWalks 在时间链接预测任务中优于已有的方法，实现了更高的准确性并保持了鲁棒性。 

---
# From Drafts to Answers: Unlocking LLM Potential via Aggregation Fine-Tuning 

**Title (ZH)**: 从草稿到答案：通过聚合微调释放大规模语言模型的潜力 

**Authors**: Yafu Li, Zhilin Wang, Tingchen Fu, Ganqu Cui, Sen Yang, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.11877)  

**Abstract**: Scaling data and model size has been proven effective for boosting the performance of large language models. In addition to training-time scaling, recent studies have revealed that increasing test-time computational resources can further improve performance. In this work, we introduce Aggregation Fine-Tuning (AFT), a supervised finetuning paradigm where the model learns to synthesize multiple draft responses, referred to as proposals, into a single, refined answer, termed aggregation. At inference time, a propose-and-aggregate strategy further boosts performance by iteratively generating proposals and aggregating them. Empirical evaluations on benchmark datasets show that AFT-trained models substantially outperform standard SFT. Notably, an AFT model, fine-tuned from Llama3.1-8B-Base with only 64k data, achieves a 41.3% LC win rate on AlpacaEval 2, surpassing significantly larger LLMs such as Llama3.1-405B-Instruct and GPT4. By combining sequential refinement and parallel sampling, the propose-and-aggregate framework scales inference-time computation in a flexible manner. Overall, These findings position AFT as a promising approach to unlocking additional capabilities of LLMs without resorting to increasing data volume or model size. 

**Abstract (ZH)**: 扩大数据集和模型规模已被证明可以有效提升大型语言模型的性能。除了训练时的扩展外，近期的研究表明，增加测试时的计算资源也可以进一步提高性能。在本研究中，我们提出了聚合微调（AFT，Aggregation Fine-Tuning）这一监督微调范式，其中模型学习将多个草案响应（proposals）综合成一个精炼的回答（aggregation）。在推断时，采用提出并聚合的策略进一步提升性能，通过迭代生成草案并综合它们。在基准数据集上的实验评估表明，经过AFT训练的模型显著优于标准的语境微调（SFT，Supervised Fine-Tuning）。值得注意的是，从Llama3.1-8B-Base微调的AFT模型，仅使用64k数据，取得了AlpacaEval 2中41.3%的胜率，明显优于更大规模的LLM，如Llama3.1-405B-Instruct和GPT4。通过结合序列精炼和并行采样，提出并聚合框架以灵活的方式扩展推断时的计算量。总之，这些发现将AFT定位为一种有前途的方法，可以在不增加数据量或模型规模的情况下解锁LLM的额外能力。 

---
# Coarse-to-Fine Lightweight Meta-Embedding for ID-Based Recommendation 

**Title (ZH)**: 从粗到细的轻量级元嵌入方法用于基于身份的推荐 

**Authors**: Yang Wang, Haipeng Liu, Zeqian Yi, Biao Qian, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11870)  

**Abstract**: The state-of-the-art recommendation systems have shifted the attention to efficient recommendation, e.g., on-device recommendation, under memory constraints. To this end, the existing methods either focused on the lightweight embeddings for both users and items, or involved on-device systems enjoying the compact embeddings to enhance reusability and reduces space complexity. However, they focus solely on the coarse granularity of embedding, while overlook the fine-grained semantic nuances, to adversarially downgrade the efficacy of meta-embeddings in capturing the intricate relationship over both user and item, consequently resulting into the suboptimal recommendations. In this paper, we aim to study how the meta-embedding can efficiently learn varied grained semantics, together with how the fine-grained meta-embedding can strengthen the representation of coarse-grained meta-embedding. To answer these questions, we develop a novel graph neural networks (GNNs) based recommender where each user and item serves as the node, linked directly to coarse-grained virtual nodes and indirectly to fine-grained virtual nodes, ensuring different grained semantic learning, while disclosing: 1) In contrast to coarse-grained semantics, fine-grained semantics are well captured through sparse meta-embeddings, which adaptively 2) balance the embedding uniqueness and memory constraint. Additionally, the initialization method come up upon SparsePCA, along with a soft thresholding activation function to render the sparseness of the meta-embeddings. We propose a weight bridging update strategy that focuses on matching each coarse-grained meta-embedding with several fine-grained meta-embeddings based on the users/items' semantics. Extensive experiments substantiate our method's superiority over existing baselines. Our code is available at this https URL. 

**Abstract (ZH)**: 最新的推荐系统已经将注意力转向了高效的推荐，如设备本地推荐以及在内存限制下的推荐。为此，现有方法要么集中在用户和物品的轻量级嵌入上，要么构建了拥有紧凑嵌入的设备本地系统，以增强嵌入的重用性和减少空间复杂度。然而，这些方法仅关注嵌入的粗粒度表示，而忽略了细粒度语义的微妙变化，这导致了元嵌入在捕捉用户和物品之间的复杂关系时效果不佳，最终导致了次优的推荐。本文旨在研究元嵌入如何高效地学习不同粒度的语义，以及细粒度元嵌入如何增强粗粒度元嵌入的表现。为了解决这些问题，我们提出了一种基于图神经网络（GNN）的推荐系统，其中用户和物品作为节点，直接连接到粗粒度的虚拟节点，间接连接到细粒度的虚拟节点，确保了不同粒度的语义学习，同时证明了：1）与粗粒度语义相比，细粒度语义可以通过稀疏元嵌入很好地捕捉到，这些嵌入能自适应地平衡嵌入的独特性和内存限制；2）利用SparsePCA的初始化方法，并结合一项软阈值激活函数以实现元嵌入的稀疏性。我们提出了一个权重桥接更新策略，该策略基于用户/物品的语义，将每个粗粒度的元嵌入与多个细粒度的元嵌入进行匹配。广泛的实验表明，我们的方法在现有基线方法中的优越性。我们的代码可在以下链接获取：[此网址]。 

---
# Network-informed Prompt Engineering against Organized Astroturf Campaigns under Extreme Class Imbalance 

**Title (ZH)**: 网络导向的提示工程以对抗极端类别不平衡下的组织化水军运动 

**Authors**: Nikos Kanakaris, Heng Ping, Xiongye Xiao, Nesreen K. Ahmed, Luca Luceri, Emilio Ferrara, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2501.11849)  

**Abstract**: Detecting organized political campaigns is of paramount importance in fighting against disinformation on social media. Existing approaches for the identification of such organized actions employ techniques mostly from network science, graph machine learning and natural language processing. Their ultimate goal is to analyze the relationships and interactions (e.g. re-posting) among users and the textual similarities of their posts. Despite their effectiveness in recognizing astroturf campaigns, these methods face significant challenges, notably the class imbalance in available training datasets. To mitigate this issue, recent methods usually resort to data augmentation or increasing the number of positive samples, which may not always be feasible or sufficient in real-world settings. Following a different path, in this paper, we propose a novel framework for identifying astroturf campaigns based solely on large language models (LLMs), introducing a Balanced Retrieval-Augmented Generation (Balanced RAG) component. Our approach first gives both textual information concerning the posts (in our case tweets) and the user interactions of the social network as input to a language model. Then, through prompt engineering and the proposed Balanced RAG method, it effectively detects coordinated disinformation campaigns on X (Twitter). The proposed framework does not require any training or fine-tuning of the language model. Instead, by strategically harnessing the strengths of prompt engineering and Balanced RAG, it facilitates LLMs to overcome the effects of class imbalance and effectively identify coordinated political campaigns. The experimental results demonstrate that by incorporating the proposed prompt engineering and Balanced RAG methods, our framework outperforms the traditional graph-based baselines, achieving 2x-3x improvements in terms of precision, recall and F1 scores. 

**Abstract (ZH)**: 检测有组织的政治活动对于打击社交 media 上的虚假信息至关重要。现有的识别此类有组织行动的方法主要采用了网络科学、图机器学习和自然语言处理的技术。这些方法的最终目标是分析用户之间的关系和互动（例如转发）以及其帖子的文本相似性。尽管这些方法在识别假流量活动方面表现出有效性，但它们在应对可用训练数据集中的类别不平衡问题时面临显著挑战。为缓解这一问题，近年来的方法通常依赖于数据增强或增加正样本数量，但在实际应用场景中这可能并不总是可行或足够。

与这种方法不同，本文提出了一种基于大型语言模型（LLMs）的新框架，引入了平衡检索增强生成（Balanced RAG）组件。我们的方法首先将有关帖子（例如推特）的文本信息以及社交网络用户的互动信息输入到语言模型。然后，通过提示工程和所提出的平衡检索增强生成方法，有效检测 X（推特）上的协调性虚假信息活动。该提出的框架不需要对语言模型进行任何训练或微调。相反，通过战略性地利用提示工程和平衡检索增强生成的优势，它使大型语言模型能够克服类别不平衡的影响，并有效识别有组织的政治活动。实验结果表明，通过集成所提出的提示工程和平衡检索增强生成方法，我们的框架在精准度、召回率和 F1 分数上超过了传统的基于图的方法，取得了 2 至 3 倍的提升。 

---
# A Survey on Memory-Efficient Large-Scale Model Training in AI for Science 

**Title (ZH)**: 人工智能科学研究中大规模模型训练的内存高效方法综述 

**Authors**: Kaiyuan Tian, Linbo Qiao, Baihui Liu, Gongqingjian Jiang, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.11847)  

**Abstract**: Scientific research faces high costs and inefficiencies with traditional methods, but the rise of deep learning and large language models (LLMs) offers innovative solutions. This survey reviews LLM applications across scientific fields such as biology, medicine, chemistry, and meteorology, underscoring their role in advancing research. However, the continuous expansion of model size has led to significant memory demands, hindering further development and application of LLMs for science. To address this, we review memory-efficient training techniques for LLMs based on the transformer architecture, including distributed training, mixed precision training, and gradient checkpointing. Using AlphaFold 2 as an example, we demonstrate how tailored memory optimization methods can reduce storage needs while preserving prediction accuracy. We also discuss the challenges of memory optimization in practice and potential future directions, hoping to provide valuable insights for researchers and engineers. 

**Abstract (ZH)**: 传统的科学研究方法面临着高成本和低效率的问题，但深度学习和大型语言模型（LLMs）的兴起为这一问题提供了创新的解决方案。本综述概述了LLMs在生物学、医学、化学和气象学等科学领域的应用，强调了LLMs在推动研究进展中的作用。然而，模型规模的持续扩大导致了显著的内存需求增加，阻碍了LLMs在科学中的进一步发展和应用。为了解决这一问题，本文回顾了基于变压器架构的LLMs的内存高效训练技术，包括分布式训练、混合精度训练和梯度检查点技术。以AlphaFold 2为例，我们展示了如何通过定制化的内存优化方法减少存储需求同时保持预测准确度。此外，我们还讨论了实际中内存优化的挑战和未来可能的发展方向，希望能为研究人员和工程师提供有价值的见解。 

---
# Supervised Learning for Analog and RF Circuit Design: Benchmarks and Comparative Insights 

**Title (ZH)**: Supervised Learning在模拟和射频电路设计中的应用：基准与比较见解 

**Authors**: Asal Mehradfar, Xuzhe Zhao, Yue Niu, Sara Babakniya, Mahdi Alesheikh, Hamidreza Aghasi, Salman Avestimehr  

**Link**: [PDF](https://arxiv.org/pdf/2501.11839)  

**Abstract**: Automating analog and radio-frequency (RF) circuit design using machine learning (ML) significantly reduces the time and effort required for parameter optimization. This study explores supervised ML-based approaches for designing circuit parameters from performance specifications across various circuit types, including homogeneous and heterogeneous designs. By evaluating diverse ML models, from neural networks like transformers to traditional methods like random forests, we identify the best-performing models for each circuit. Our results show that simpler circuits, such as low-noise amplifiers, achieve exceptional accuracy with mean relative errors as low as 0.3% due to their linear parameter-performance relationships. In contrast, complex circuits, like power amplifiers and voltage-controlled oscillators, present challenges due to their non-linear interactions and larger design spaces. For heterogeneous circuits, our approach achieves an 88% reduction in errors with increased training data, with the receiver achieving a mean relative error as low as 0.23%, showcasing the scalability and accuracy of the proposed methodology. Additionally, we provide insights into model strengths, with transformers excelling in capturing non-linear mappings and k-nearest neighbors performing robustly in moderately linear parameter spaces, especially in heterogeneous circuits with larger datasets. This work establishes a foundation for extending ML-driven design automation, enabling more efficient and scalable circuit design workflows. 

**Abstract (ZH)**: 使用机器学习（ML）自动化模拟和射频（RF）电路设计显著减少了参数优化所需的时间和努力。本研究探讨了监督学习（ML）方法在不同电路类型中的应用，涵盖同构和异构设计，旨在根据性能规格设计电路参数。通过评估从神经网络（如变压器）到传统方法（如随机森林）等各种ML模型，我们确定了适合每种电路的最佳模型。结果表明，简单的电路，如低噪声放大器，由于其线性的参数-性能关系，可获得极高的准确性，相对误差均值低至0.3%。相比之下，复杂的电路，如功率放大器和电压控制振荡器，由于其非线性交互和更大的设计空间，带来了挑战。对于异构电路，我们的方法在增加训练数据后可实现88%的误差减少，接收器的相对误差均值低至0.23%，这展示了所提出方法的可扩展性和准确性。此外，我们还提供了关于模型优劣的新见解：变压器在捕捉非线性映射方面表现出色，而k最近邻算法在中等线性的参数空间中表现 robust，在数据集较大的异构电路中尤其如此。这项工作为扩展基于ML的设计自动化奠定了基础，能够促进更高效和可扩展的电路设计流程。 

---
# Data-driven Detection and Evaluation of Damages in Concrete Structures: Using Deep Learning and Computer Vision 

**Title (ZH)**: 基于数据驱动的混凝土结构损伤检测与评估：利用深度学习和计算机视觉 

**Authors**: Saeid Ataei, Saeed Adibnazari, Seyyed Taghi Ataei  

**Link**: [PDF](https://arxiv.org/pdf/2501.11836)  

**Abstract**: Structural integrity is vital for maintaining the safety and longevity of concrete infrastructures such as bridges, tunnels, and walls. Traditional methods for detecting damages like cracks and spalls are labor-intensive, time-consuming, and prone to human error. To address these challenges, this study explores advanced data-driven techniques using deep learning for automated damage detection and analysis. Two state-of-the-art instance segmentation models, YOLO-v7 instance segmentation and Mask R-CNN, were evaluated using a dataset comprising 400 images, augmented to 10,995 images through geometric and color-based transformations to enhance robustness. The models were trained and validated using a dataset split into 90% training set, validation and test set 10%. Performance metrics such as precision, recall, mean average precision (mAP@0.5), and frames per second (FPS) were used for evaluation. YOLO-v7 achieved a superior mAP@0.5 of 96.1% and processed 40 FPS, outperforming Mask R-CNN, which achieved a mAP@0.5 of 92.1% with a slower processing speed of 18 FPS. The findings recommend YOLO-v7 instance segmentation model for real-time, high-speed structural health monitoring, while Mask R-CNN is better suited for detailed offline assessments. This study demonstrates the potential of deep learning to revolutionize infrastructure maintenance, offering a scalable and efficient solution for automated damage detection. 

**Abstract (ZH)**: 结构完整性对于确保桥梁、隧道和墙壁等混凝土基础设施的安全和持久性至关重要。传统检测损伤（如裂缝和剥落）的方法耗时、劳动密集且容易出错。为应对这些挑战，本研究探讨了使用深度学习的先进数据驱动技术，以实现自动损伤检测与分析。研究评估了两种先进的实例分割模型：YOLO-v7 实例分割和 Mask R-CNN。研究使用一个包含 400 张图像的数据集，并通过几何和颜色变换对数据集进行增强，扩充到 10,995 张图像，以提高模型的鲁棒性。模型在数据集上进行训练和验证，数据集分为 90% 的训练集和 10% 的验证与测试集。使用精度、召回率、平均精度（mAP@0.5）和每秒帧数（FPS）等性能指标进行评估。YOLO-v7 达到了优越的 mAP@0.5 为 96.1%，处理速度为 40 FPS，而 Mask R-CNN 的 mAP@0.5 为 92.1%，处理速度仅为 18 FPS。研究结果推荐使用 YOLO-v7 实例分割模型进行实时、高速的结构健康监测，而 Mask R-CNN 更适用于详细的离线评估。研究展示了深度学习在基础设施维护领域有潜力实现革新，提供了一种可扩展和高效的自动损伤检测解决方案。 

---
# Is your LLM trapped in a Mental Set? Investigative study on how mental sets affect the reasoning capabilities of LLMs 

**Title (ZH)**: 你的大规模语言模型（LLM）被束缚在思维定势中了吗？探究思维定势对大模型推理能力的影响 

**Authors**: Saiful Haq, Niyati Chhaya, Piyush Pandey, Pushpak Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2501.11833)  

**Abstract**: In this paper, we present an investigative study on how Mental Sets influence the reasoning capabilities of LLMs. LLMs have excelled in diverse natural language processing (NLP) tasks, driven by advancements in parameter-efficient fine-tuning (PEFT) and emergent capabilities like in-context learning (ICL). For complex reasoning tasks, selecting the right model for PEFT or ICL is critical, often relying on scores on benchmarks such as MMLU, MATH, and GSM8K. However, current evaluation methods, based on metrics like F1 Score or reasoning chain assessments by larger models, overlook a key dimension: adaptability to unfamiliar situations and overcoming entrenched thinking patterns. In cognitive psychology, Mental Set refers to the tendency to persist with previously successful strategies, even when they become inefficient - a challenge for problem solving and reasoning. We compare the performance of LLM models like Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct and GPT-4o in the presence of mental sets. To the best of our knowledge, this is the first study to integrate cognitive psychology concepts into the evaluation of LLMs for complex reasoning tasks, providing deeper insights into their adaptability and problem-solving efficacy. 

**Abstract (ZH)**: 本文探讨了心理定势（Mental Sets）如何影响大型语言模型（LLMs）的推理能力。LLMs在多种自然语言处理（NLP）任务中表现出色，这得益于参数高效微调（PEFT）的进步以及内省式学习（ICL）等新兴能力。在复杂推理任务中，选择合适的模型进行PEFT或ICL至关重要，通常依赖于诸如MMLU、MATH和GSM8K等基准测试的得分。然而，当前的评估方法，如基于F1分数或由更大模型评估推理链的方法，忽视了一个关键维度：对不熟悉情况的适应能力以及克服根深蒂固的思维模式。在认知心理学中，心理定势指的是倾向于坚持先前成功的策略，即使这些策略变得无效——这对解决问题和推理构成了挑战。我们对比了Llama-3.1-8B-Instruct、Llama-3.1-70B-Instruct和GPT-4o等LLM模型在面对心理定势时的表现。据我们所知，这是首个将认知心理学概念整合到LLM复杂推理任务评估中的研究，为了解它们的适应性和问题解决效果提供了更深入的洞见。 

---
# Fact-Preserved Personalized News Headline Generation 

**Title (ZH)**: 事实保留的个性化新闻标题生成 

**Authors**: Zhao Yang, Junhong Lian, Xiang Ao  

**Link**: [PDF](https://arxiv.org/pdf/2501.11828)  

**Abstract**: Personalized news headline generation, aiming at generating user-specific headlines based on readers' preferences, burgeons a recent flourishing research direction. Existing studies generally inject a user interest embedding into an encoderdecoder headline generator to make the output personalized, while the factual consistency of headlines is inadequate to be verified. In this paper, we propose a framework Fact-Preserved Personalized News Headline Generation (short for FPG), to prompt a tradeoff between personalization and consistency. In FPG, the similarity between the candidate news to be exposed and the historical clicked news is used to give different levels of attention to key facts in the candidate news, and the similarity scores help to learn a fact-aware global user embedding. Besides, an additional training procedure based on contrastive learning is devised to further enhance the factual consistency of generated headlines. Extensive experiments conducted on a real-world benchmark PENS validate the superiority of FPG, especially on the tradeoff between personalization and factual consistency. 

**Abstract (ZH)**: 面向用户的个性化新闻标题生成旨在根据读者的偏好生成定制化的标题，目前已成为一个蓬勃发展的研究方向。现有研究普遍通过将用户兴趣嵌入编码-解码器标题生成器中来实现个性化，但生成的标题在事实准确性和一致性上存在不足。本文提出了一种名为保留事实的个性化新闻标题生成框架（简称FPG），旨在平衡个性化和一致性之间的关系。在FPG中，通过候选新闻与用户历史点击新闻之间的相似度来赋予候选新闻中的关键事实不同程度的关注，相似度分数有助于学习一种事实感知的全局用户嵌入。此外，我们还设计了一种基于对比学习的额外训练过程，以进一步增强生成标题的事实一致性。在PENS这个真实世界基准数据集上的广泛实验验证了FPG在个性化和事实一致性之间的优越性。 

---
# PXGen: A Post-hoc Explainable Method for Generative Models 

**Title (ZH)**: PXGen：生成模型的后验可解释方法 

**Authors**: Yen-Lung Huang, Ming-Hsi Weng, Hao-Tsung Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11827)  

**Abstract**: With the rapid growth of generative AI in numerous applications, explainable AI (XAI) plays a crucial role in ensuring the responsible development and deployment of generative AI technologies. XAI has undergone notable advancements and widespread adoption in recent years, reflecting a concerted push to enhance the transparency, interpretability, and credibility of AI systems. Recent research emphasizes that a proficient XAI method should adhere to a set of criteria, primarily focusing on two key areas. Firstly, it should ensure the quality and fluidity of explanations, encompassing aspects like faithfulness, plausibility, completeness, and tailoring to individual needs. Secondly, the design principle of the XAI system or mechanism should cover the following factors such as reliability, resilience, the verifiability of its outputs, and the transparency of its algorithm. However, research in XAI for generative models remains relatively scarce, with little exploration into how such methods can effectively meet these criteria in that domain. In this work, we propose PXGen, a post-hoc explainable method for generative models. Given a model that needs to be explained, PXGen prepares two materials for the explanation, the Anchor set and intrinsic & extrinsic criteria. Those materials are customizable by users according to their purpose and requirements. Via the calculation of each criterion, each anchor has a set of feature values and PXGen provides examplebased explanation methods according to the feature values among all the anchors and illustrated and visualized to the users via tractable algorithms such as k-dispersion or k-center. 

**Abstract (ZH)**: 随着生成型人工智能在众多应用中的迅速发展，可解释的人工智能（XAI）在确保生成型人工智能技术的负责任开发与部署中扮演着至关重要的角色。近年来，XAI 已取得显著进步并得到了广泛采用，反映了增强人工智能系统透明度、可解释性和可信度的共同努力。最近的研究强调，一个高效的 XAI 方法应遵循一系列标准，主要关注两个关键领域。首先，它需要确保解释的质量和流畅性，涵盖诸如忠实性、合理性、完整性和个性化需求适应性等方面。其次，XAI 系统或机制的设计原则应包括可靠性、抗压性、输出可验证性以及算法的透明性。然而，对于生成模型的 XAI 研究相对较少，关于如何有效满足这些标准的研究也很少。在这项工作中，我们提出了一种后验可解释方法 PXGen，适用于生成模型。给定一个需要解释的模型，PXGen 准备了两种解释材料：锚集和内在与外在标准。这些材料可以根据用户的目标和需求进行自定义。通过计算每个标准，每个锚点都会有一组特征值，PXGen 根据所有锚点的特征值提供基于示例的解释方法，并通过可追踪算法（如 k-Dispersion 或 k-Center）进行说明和可视化。 

---
# Toward Scalable Graph Unlearning: A Node Influence Maximization based Approach 

**Title (ZH)**: 面向可扩展的图去学习：一种基于节点影响力最大化的方法 

**Authors**: Xunkai Li, Bowen Fan, Zhengyu Wu, Zhiyu Li, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11823)  

**Abstract**: Machine unlearning, as a pivotal technology for enhancing model robustness and data privacy, has garnered significant attention in prevalent web mining applications, especially in thriving graph-based scenarios. However, most existing graph unlearning (GU) approaches face significant challenges due to the intricate interactions among web-scale graph elements during the model training: (1) The gradient-driven node entanglement hinders the complete knowledge removal in response to unlearning requests; (2) The billion-level graph elements in the web scenarios present inevitable scalability issues. To break the above limitations, we open up a new perspective by drawing a connection between GU and conventional social influence maximization. To this end, we propose Node Influence Maximization (NIM) through the decoupled influence propagation model and fine-grained influence function in a scalable manner, which is crafted to be a plug-and-play strategy to identify potential nodes affected by unlearning entities. This approach enables offline execution independent of GU, allowing it to be seamlessly integrated into most GU methods to improve their unlearning performance. Based on this, we introduce Scalable Graph Unlearning (SGU) as a new fine-tuned framework, which balances the forgetting and reasoning capability of the unlearned model by entity-specific optimizations. Extensive experiments on 14 datasets, including large-scale ogbn-papers100M, have demonstrated the effectiveness of our approach. Specifically, NIM enhances the forgetting capability of most GU methods, while SGU achieves comprehensive SOTA performance and maintains scalability. 

**Abstract (ZH)**: 机器可遗忘性作为一种提升模型鲁棒性和数据隐私的关键技术，在广泛的网络挖掘应用中引起了广泛关注，尤其是在蓬勃发展的图基场景中。然而，现有的大多数图可遗忘性（Graph Unlearning, GU）方法在模型训练过程中面对着大规模图元素之间复杂交互带来的显著挑战：（1）基于梯度的节点纠缠阻碍了对遗忘请求的完全知识去除；（2）网络场景中数量级别的图元素带来了不可避免的可扩展性问题。为了突破上述限制，我们从新的角度出发，通过将GU与传统社会影响最大化联系起来来开启新视角。为此，我们提出了通过解耦影响传播模型和细粒度影响函数在可扩展方式下实现的节点影响最大化（Node Influence Maximization, NIM），旨在成为一种即插即用的策略，以识别潜在受影响的节点。该方法实现了离线执行，与GU独立，使得它能够无缝集成到大多数GU方法中，以提高其遗忘性能。基于此，我们引入了可扩展图可遗忘性（Scalable Graph Unlearning, SGU）作为一个新的细调框架，通过实体特定优化平衡了未学习模型的遗忘能力和推理能力。在14个数据集上的广泛实验，包括大规模的ogbn-papers100M，证明了我们方法的有效性。具体而言，NIM增强了大多数GU方法的遗忘能力，而SGU实现了全面的SOTA性能并保持了可扩展性。 

---
# Toward Effective Digraph Representation Learning: A Magnetic Adaptive Propagation based Approach 

**Title (ZH)**: 面向有效有向图表示学习：一种磁性自适应传播方法 

**Authors**: Xunkai Li, Daohan Su, Zhengyu Wu, Guang Zeng, Hongchao Qin, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11817)  

**Abstract**: The $q$-parameterized magnetic Laplacian serves as the foundation of directed graph (digraph) convolution, enabling this kind of digraph neural network (MagDG) to encode node features and structural insights by complex-domain message passing. As a generalization of undirected methods, MagDG shows superior capability in modeling intricate web-scale topology. Despite the great success achieved by existing MagDGs, limitations still exist: (1) Hand-crafted $q$: The performance of MagDGs depends on selecting an appropriate $q$-parameter to construct suitable graph propagation equations in the complex domain. This parameter tuning, driven by downstream tasks, limits model flexibility and significantly increases manual effort. (2) Coarse Message Passing: Most approaches treat all nodes with the same complex-domain propagation and aggregation rules, neglecting their unique digraph contexts. This oversight results in sub-optimal performance. To address the above issues, we propose two key techniques: (1) MAP is crafted to be a plug-and-play complex-domain propagation optimization strategy in the context of digraph learning, enabling seamless integration into any MagDG to improve predictions while enjoying high running efficiency. (2) MAP++ is a new digraph learning framework, further incorporating a learnable mechanism to achieve adaptively edge-wise propagation and node-wise aggregation in the complex domain for better performance. Extensive experiments on 12 datasets demonstrate that MAP enjoys flexibility for it can be incorporated with any MagDG, and scalability as it can deal with web-scale digraphs. MAP++ achieves SOTA predictive performance on 4 different downstream tasks. 

**Abstract (ZH)**: $q$-参数化磁拉普拉斯算子是定向图（有向图）卷积的基础，使得这种有向图神经网络（MagDG）能够通过复域消息传递来编码节点特征和结构洞察。MagDG 是对无向图方法的推广，显示出在建模复杂网络拓扑结构方面的优越能力。尽管现有的 MagDG 取得了巨大的成功，但仍存在一些局限性：（1）手工构建的 $q$ 参数：MagDG 的性能依赖于选择合适的 $q$ 参数来构建复域中的合适的图传播方程。下游任务驱动的参数调整限制了模型的灵活性，并显著增加了手动努力。（2）粗略的消息传递：大多数方法采用相同的复域传播和聚合规则对待所有节点，忽略了它们独特的有向图上下文。这种忽视导致了次优的表现。

为解决上述问题，我们提出两种关键技术：（1）MAP 设计为有向图学习中可即插即用的复域传播优化策略，可无缝集成到任何 MagDG 中以提高预测效果，同时享受高效的运行速度。（2）MAP++ 是一个新的有向图学习框架，进一步引入了可学习机制，以便在复域中实现边级自适应传播和节点级聚合，从而提高性能。

广泛的实验（涉及 12 个数据集）证明，MAP 具有灵活性，可以与任何 MagDG 结合使用，以及可扩展性，可以处理大规模有向图。MAP++ 在 4 个不同的下游任务中实现了最佳预测性能（SOTA），展示了其强大的性能。 

---
# Benchmarking Large Language Models via Random Variables 

**Title (ZH)**: 通过随机变量对比大规模语言模型 

**Authors**: Zijin Hong, Hao Wu, Su Dong, Junnan Dong, Yilin Xiao, Yujing Zhang, Zhu Wang, Feiran Huang, Linyi Li, Hongxia Yang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11790)  

**Abstract**: With the continuous advancement of large language models (LLMs) in mathematical reasoning, evaluating their performance in this domain has become a prominent research focus. Recent studies have raised concerns about the reliability of current mathematical benchmarks, highlighting issues such as simplistic design and potential data leakage. Therefore, creating a reliable benchmark that effectively evaluates the genuine capabilities of LLMs in mathematical reasoning remains a significant challenge. To address this, we propose RV-Bench, a framework for Benchmarking LLMs via Random Variables in mathematical reasoning. Specifically, the background content of a random variable question (RV question) mirrors the original problem in existing standard benchmarks, but the variable combinations are randomized into different values. LLMs must fully understand the problem-solving process for the original problem to correctly answer RV questions with various combinations of variable values. As a result, the LLM's genuine capability in mathematical reasoning is reflected by its accuracy on RV-Bench. Extensive experiments are conducted with 29 representative LLMs across 900+ RV questions. A leaderboard for RV-Bench ranks the genuine capability of these LLMs. Further analysis of accuracy dropping indicates that current LLMs still struggle with complex mathematical reasoning problems. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在数学推理领域的不断进步，评估其在该领域的性能已成为一个重要的研究焦点。最近的研究对当前数学基准的可靠性提出了质疑，突显了设计简单及潜在数据泄露等问题。因此，创建一个可靠且能有效评估LLMs在数学推理中真正能力的基准仍然是一个重大挑战。为了解决这一问题，我们提出RV-Bench框架，通过随机变量对LLMs在数学推理中的基准测试进行评估。具体而言，随机变量问题（RV问题）的背景内容与现有标准基准中的原始问题相似，但变量组合被随机化为不同的值。对于原本的问题，LLMs必须完全理解问题解决过程，才能正确回答包含不同变量值组合的RV问题。因此，RV-Bench中的准确性反映了LLMs在数学推理中的真实能力。我们对29个代表性LLMs进行了广泛的实验，涵盖900多个RV问题。RV-Bench的排行榜对这些LLMs的真实能力进行了排名。进一步的准确率分析表明，当前的LLMs在处理复杂的数学推理问题时仍然存在困难。 

---
# Human-AI Collaborative Game Testing with Vision Language Models 

**Title (ZH)**: 基于视觉语言模型的人机协作游戏测试 

**Authors**: Boran Zhang, Muhan Xu, Zhijun Pan  

**Link**: [PDF](https://arxiv.org/pdf/2501.11782)  

**Abstract**: As modern video games become increasingly complex, traditional manual testing methods are proving costly and inefficient, limiting the ability to ensure high-quality game experiences. While advancements in Artificial Intelligence (AI) offer the potential to assist human testers, the effectiveness of AI in truly enhancing real-world human performance remains underexplored. This study investigates how AI can improve game testing by developing and experimenting with an AI-assisted workflow that leverages state-of-the-art machine learning models for defect detection. Through an experiment involving 800 test cases and 276 participants of varying backgrounds, we evaluate the effectiveness of AI assistance under four conditions: with or without AI support, and with or without detailed knowledge of defects and design documentation. The results indicate that AI assistance significantly improves defect identification performance, particularly when paired with detailed knowledge. However, challenges arise when AI errors occur, negatively impacting human decision-making. Our findings show the importance of optimizing human-AI collaboration and implementing strategies to mitigate the effects of AI inaccuracies. By this research, we demonstrate AI's potential and problems in enhancing efficiency and accuracy in game testing workflows and offers practical insights for integrating AI into the testing process. 

**Abstract (ZH)**: 随着现代视频游戏变得越来越复杂，传统的手工测试方法越来越昂贵且效率低下，这限制了确保高质量游戏体验的能力。虽然人工智能（AI）的进步提供了辅助人工测试者的潜力，但AI在真正提升实际人类性能方面的有效性仍被广泛忽视。本研究调查了AI如何通过开发并实验一种利用先进机器学习模型进行缺陷检测的AI辅助工作流程来提高游戏测试的效果。通过涉及800个测试案例和276名背景各异的参与者的一项实验，我们在四种条件下评估了AI辅助的效果：有或没有AI支持，以及有或没有详细的缺陷和设计文档知识。结果显示，当与详细的缺陷和设计文档知识结合使用时，AI辅助显著提高了缺陷识别性能。然而，当AI出现错误时，这会对人类的决策产生负面影响。我们的研究结果强调了优化人类与AI的合作的重要性，并提出了缓解AI不准确性影响的策略。通过本研究，我们展示了AI在提高游戏测试工作流程的效率和准确性方面的潜力和问题，并提供了将AI整合到测试过程中的实用见解。 

---
# Is logical analysis performed by transformers taking place in self-attention or in the fully connected part? 

**Title (ZH)**: Transformer进行逻辑分析的过程是发生在自我注意力机制中还是完全连接部分中？ 

**Authors**: Evgeniy Shin, Heinrich Matzinger  

**Link**: [PDF](https://arxiv.org/pdf/2501.11765)  

**Abstract**: Transformers architecture apply self-attention to tokens represented as vectors, before a fully connected (neuronal network) layer. These two parts can be layered many times. Traditionally, self-attention is seen as a mechanism for aggregating information before logical operations are performed by the fully connected layer. In this paper, we show, that quite counter-intuitively, the logical analysis can also be performed within the self-attention. For this we implement a handcrafted single-level encoder layer which performs the logical analysis within self-attention. We then study the scenario in which a one-level transformer model undergoes self-learning using gradient descent. We investigate whether the model utilizes fully connected layers or self-attention mechanisms for logical analysis when it has the choice. Given that gradient descent can become stuck at undesired zeros, we explicitly calculate these unwanted zeros and find ways to avoid them. We do all this in the context of predicting grammatical category pairs of adjacent tokens in a text. We believe that our findings have broader implications for understanding the potential logical operations performed by self-attention. 

**Abstract (ZH)**: Transformer架构通过自注意力机制对表示为向量的标记进行处理，随后再经过全连接（神经网络）层。这两部分可以多次堆叠。传统上，自注意力机制被视为一种在全连接层执行逻辑操作之前聚合信息的机制。在本文中，我们展示了这一机制实际上可以逆着直觉来执行逻辑分析。为此，我们实现了一种手工设计的一层编码器层，该层在自注意力机制中执行逻辑分析。随后，我们研究了一层变压器模型在使用梯度下降进行自学习的情景。我们探讨了当模型在两者之间进行选择时，它是否使用全连接层或自注意力机制来进行逻辑分析。鉴于梯度下降可能会陷入不需要的零点，我们显式地计算这些不需要的零点，并寻找避免它们的方法。所有这些研究都是在预测文本中相邻标记的语法类别对的背景下进行的。我们认为，我们的发现对理解自注意力机制可能执行的潜在逻辑操作具有更广泛的意义。 

---
# Optimizing Pretraining Data Mixtures with LLM-Estimated Utility 

**Title (ZH)**: 使用大规模语言模型估计效用优化预训练数据混合 

**Authors**: William Held, Bhargavi Paranjape, Punit Singh Koura, Mike Lewis, Frank Zhang, Todor Mihaylov  

**Link**: [PDF](https://arxiv.org/pdf/2501.11747)  

**Abstract**: Large Language Models improve with increasing amounts of high-quality training data. However, leveraging larger datasets requires balancing quality, quantity, and diversity across sources. After evaluating nine baseline methods under both compute- and data-constrained scenarios, we find token-count heuristics outperform manual and learned mixes, indicating that simple approaches accounting for dataset size and diversity are surprisingly effective. Building on this insight, we propose two complementary approaches: UtiliMax, which extends token-based heuristics by incorporating utility estimates from reduced-scale ablations, achieving up to a 10.6x speedup over manual baselines; and Model Estimated Data Utility (MEDU), which leverages LLMs to estimate data utility from small samples, matching ablation-based performance while reducing computational requirements by $\sim$200x. Together, these approaches establish a new framework for automated, compute-efficient data mixing that is robust across training regimes. 

**Abstract (ZH)**: 大量语言模型在高质量训练数据的数量增加时会表现出更好的性能。然而，利用更大的数据集需要在质量、数量和来源多样性之间进行权衡。在两种计算能力和数据约束场景下评估了九种基线方法后，我们发现基于标记数的启发式方法优于手动和学习的混合方法，表明简单的考虑数据集大小和多样性的方法其实非常有效。基于这一洞察，我们提出了两种互补的方法：UtiliMax，该方法通过引入规模减小的消减版的效用估计来扩展基于标记数的启发式方法，与手动基线相比，可实现高达10.6倍的加速；以及模型估计数据效用（MEDU），该方法利用语言模型从小样本中估计数据效用，与基于消减的方法性能相当，同时计算需求减少约200倍。这两种方法共同建立了一种新的自动化、计算高效的数据混合框架，在各种训练模式下具有稳健性。 

---
# SILO: Solving Inverse Problems with Latent Operators 

**Title (ZH)**: SILO：使用潜在运算符求解逆问题 

**Authors**: Ron Raphaeli, Sean Man, Michael Elad  

**Link**: [PDF](https://arxiv.org/pdf/2501.11746)  

**Abstract**: Consistent improvement of image priors over the years has led to the development of better inverse problem solvers. Diffusion models are the newcomers to this arena, posing the strongest known prior to date. Recently, such models operating in a latent space have become increasingly predominant due to their efficiency. In recent works, these models have been applied to solve inverse problems. Working in the latent space typically requires multiple applications of an Autoencoder during the restoration process, which leads to both computational and restoration quality challenges. In this work, we propose a new approach for handling inverse problems with latent diffusion models, where a learned degradation function operates within the latent space, emulating a known image space degradation. Usage of the learned operator reduces the dependency on the Autoencoder to only the initial and final steps of the restoration process, facilitating faster sampling and superior restoration quality. We demonstrate the effectiveness of our method on a variety of image restoration tasks and datasets, achieving significant improvements over prior art. 

**Abstract (ZH)**: 多年来，图像先验的一致改进推动了逆问题求解器的发展。扩散模型作为这一领域的新兴技术，提供迄今为止最强的先验知识。最近，这些模型在潜在空间中的应用变得越来越普遍，这是因为它们具有高效性。在最近的研究中，这些模型被用于解决逆问题。在潜在空间中工作通常需要在恢复过程中多次应用自编码器，这既带来了计算上的挑战，也影响了恢复质量。在本研究中，我们提出了一种新的方法，以处理潜在扩散模型的逆问题，其中学习到的退化函数在潜在空间中操作，模仿已知的图像空间退化过程。使用学习到的操作符将自编码器的依赖性限制在恢复过程的初始和最终步骤，从而实现更快的采样和更优的恢复质量。我们通过在多种图像恢复任务和数据集上展示我们方法的有效性，达到了对现有技术的重大改进。 

---
# Transformer Vibration Forecasting for Advancing Rail Safety and Maintenance 4.0 

**Title (ZH)**: 基于Transformer的振动预测以促进铁路安全与维护4.0 

**Authors**: Darío C. Larese, Almudena Bravo Cerrada, Gabriel Dambrosio Tomei, Alejandro Guerrero-López, Pablo M. Olmos, María Jesús Gómez García  

**Link**: [PDF](https://arxiv.org/pdf/2501.11730)  

**Abstract**: Maintaining railway axles is critical to preventing severe accidents and financial losses. The railway industry is increasingly interested in advanced condition monitoring techniques to enhance safety and efficiency, moving beyond traditional periodic inspections toward Maintenance 4.0.
This study introduces a robust Deep Autoregressive solution that integrates seamlessly with existing systems to avert mechanical failures. Our approach simulates and predicts vibration signals under various conditions and fault scenarios, improving dataset robustness for more effective detection systems. These systems can alert maintenance needs, preventing accidents preemptively. We use experimental vibration signals from accelerometers on train axles.
Our primary contributions include a transformer model, ShaftFormer, designed for processing time series data, and an alternative model incorporating spectral methods and enhanced observation models. Simulating vibration signals under diverse conditions mitigates the high cost of obtaining experimental signals for all scenarios. Given the non-stationary nature of railway vibration signals, influenced by speed and load changes, our models address these complexities, offering a powerful tool for predictive maintenance in the rail industry. 

**Abstract (ZH)**: 维护铁路车轴是防止严重事故和经济损失的关键。铁路行业越来越关注先进的状态监测技术，以提高安全性和效率，从传统的定期检查转向 Maintenance 4.0。
本研究引入了一种 robust 的深度自回归解决方案，该解决方案能够无缝集成到现有系统中，以预防机械故障。我们的方法在各种条件和故障情景下模拟和预测振动信号，提高了数据集的稳健性，从而提高了检测系统的有效性。这些系统可以提前预警维护需求，防止事故发生。我们使用来自列车车轴加速度计的实验振动信号。
本文的主要贡献包括一种名为 ShaftFormer 的变压器模型，专门用于处理时间序列数据，以及一种结合谱方法和增强观测模型的替代模型。在不同条件下的仿真振动信号减轻了所有场景下获得实验信号的高成本。鉴于铁路振动信号的非 stationary 性质，受到速度和载荷变化的影响，我们的模型解决了这些复杂性，为铁路行业的预测维护提供了强大的工具。 

---
# GL-ICNN: An End-To-End Interpretable Convolutional Neural Network for the Diagnosis and Prediction of Alzheimer's Disease 

**Title (ZH)**: GL-ICNN：用于阿尔茨海默病诊断与预测的端到端可解释卷积神经网络 

**Authors**: Wenjie Kang, Lize Jiskoot, Peter De Deyn, Geert Biessels, Huiberdina Koek, Jurgen Claassen, Huub Middelkoop, Wiesje Flier, Willemijn J. Jansen, Stefan Klein, Esther Bron  

**Link**: [PDF](https://arxiv.org/pdf/2501.11715)  

**Abstract**: Deep learning methods based on Convolutional Neural Networks (CNNs) have shown great potential to improve early and accurate diagnosis of Alzheimer's disease (AD) dementia based on imaging data. However, these methods have yet to be widely adopted in clinical practice, possibly due to the limited interpretability of deep learning models. The Explainable Boosting Machine (EBM) is a glass-box model but cannot learn features directly from input imaging data. In this study, we propose a novel interpretable model that combines CNNs and EBMs for the diagnosis and prediction of AD. We develop an innovative training strategy that alternatingly trains the CNN component as a feature extractor and the EBM component as the output block to form an end-to-end model. The model takes imaging data as input and provides both predictions and interpretable feature importance measures. We validated the proposed model on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset and the Health-RI Parelsnoer Neurodegenerative Diseases Biobank (PND) as an external testing set. The proposed model achieved an area-under-the-curve (AUC) of 0.956 for AD and control classification, and 0.694 for the prediction of conversion of mild cognitive impairment (MCI) to AD on the ADNI cohort. The proposed model is a glass-box model that achieves a comparable performance with other state-of-the-art black-box models. Our code is publicly available at: this https URL. 

**Abstract (ZH)**: 基于卷积神经网络（CNNs）的深度学习方法在利用影像数据早期且准确诊断阿尔茨海默病（AD）方面展现出了巨大的潜力。然而，这些方法尚未在临床实践中得到广泛采用，这可能是因为深度学习模型的可解释性有限。可解释的提升机（Explainable Boosting Machine, EBM）是一种玻璃盒模型，但它无法直接从输入的影像数据中学习特征。在本研究中，我们提出了一种结合CNNs和EBMs的新型可解释模型，用于AD的诊断和预测。我们开发了一种创新的训练策略，交替训练CNN组件作为特征提取器和EBM组件作为输出块，形成端到端模型。该模型以影像数据为输入，提供预测结果和可解释的特征重要性度量。我们使用阿尔茨海默病神经影像倡议（ADNI）数据集以及外部测试集健康-RI 帕尔森神经退行性疾病生物库（PND）验证了所提出的模型。在ADNI队列中，所提模型的曲线下面积（AUC）为0.956，用于AD和对照组的分类，以及针对轻度认知障碍（MCI）转化为AD的预测AUC为0.694。所提模型是一种玻璃盒模型，其性能与最先进的黑盒模型相当。我们的代码可以在以下链接公开获得：this https URL。 

---
# Human services organizations and the responsible integration of AI: Considering ethics and contextualizing risk(s) 

**Title (ZH)**: 人类服务机构与负责任的AI整合：考虑伦理问题与情境化风险 

**Authors**: Brian E. Perron, Lauri Goldkind, Zia Qi, Bryan G. Victor  

**Link**: [PDF](https://arxiv.org/pdf/2501.11705)  

**Abstract**: This paper examines the responsible integration of artificial intelligence (AI) in human services organizations (HSOs), proposing a nuanced framework for evaluating AI applications across multiple dimensions of risk. The authors argue that ethical concerns about AI deployment -- including professional judgment displacement, environmental impact, model bias, and data laborer exploitation -- vary significantly based on implementation context and specific use cases. They challenge the binary view of AI adoption, demonstrating how different applications present varying levels of risk that can often be effectively managed through careful implementation strategies. The paper highlights promising solutions, such as local large language models, that can facilitate responsible AI integration while addressing common ethical concerns. The authors propose a dimensional risk assessment approach that considers factors like data sensitivity, professional oversight requirements, and potential impact on client wellbeing. They conclude by outlining a path forward that emphasizes empirical evaluation, starting with lower-risk applications and building evidence-based understanding through careful experimentation. This approach enables organizations to maintain high ethical standards while thoughtfully exploring how AI might enhance their capacity to serve clients and communities effectively. 

**Abstract (ZH)**: 本文探讨了人工智能（AI）在人类服务组织（HSOs）中的负责任整合，并提出了一种多维度风险评估框架，用于评估AI应用。作者认为，在AI部署过程中所引发的伦理关切，包括专业判断替代、环境影响、模型偏差和数据工作者的剥削，会因实施背景和具体应用场景的不同而有很大差异。他们挑战了绝对二分法的AI采纳观点，展示了不同应用所呈现的风险等级可以如何通过谨慎的设计和实施策略得到有效管理。本文强调了一些有前景的解决方案，如本地大型语言模型，这些解决方案可以促进负责任的AI整合，并解决常见的伦理关切问题。作者提出了一种维度风险评估方法，考虑了数据敏感性、专业监督要求以及对客户福祉的潜在影响等因素。文章最后概述了一条前进行程，强调实证评估的重要性，从低风险应用开始，通过精心设计的实验构建证据基础的理解。这种方法使组织能够在保持高标准伦理的同时，有条不紊地探索AI如何有效增强其服务客户和社区的能力。 

---
# Spatially-Delineated Domain-Adapted AI Classification: An Application for Oncology Data 

**Title (ZH)**: 空间界定的领域适应AI分类：一种在肿瘤学数据中的应用 

**Authors**: Majid Farhadloo, Arun Sharma, Alexey Leontovich, Svetomir N. Markovic, Shashi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2501.11695)  

**Abstract**: Given multi-type point maps from different place-types (e.g., tumor regions), our objective is to develop a classifier trained on the source place-type to accurately distinguish between two classes of the target place-type based on their point arrangements. This problem is societally important for many applications, such as generating clinical hypotheses for designing new immunotherapies for cancer treatment. The challenge lies in the spatial variability, the inherent heterogeneity and variation observed in spatial properties or arrangements across different locations (i.e., place-types). Previous techniques focus on self-supervised tasks to learn domain-invariant features and mitigate domain differences; however, they often neglect the underlying spatial arrangements among data points, leading to significant discrepancies across different place-types. We explore a novel multi-task self-learning framework that targets spatial arrangements, such as spatial mix-up masking and spatial contrastive predictive coding, for spatially-delineated domain-adapted AI classification. Experimental results on real-world datasets (e.g., oncology data) show that the proposed framework provides higher prediction accuracy than baseline methods. 

**Abstract (ZH)**: 从不同位置类型的多类型点地图（例如肿瘤区域）中获得的数据，我们的目标是开发一个在源位置类型上训练的分类器，以便基于目标位置类型的点分布准确地区分两个类别。这个问题在许多应用中具有重要的社会意义，例如用于生成新的免疫治疗设计临床假设，以治疗癌症。挑战在于空间变异性，即在不同位置（即位置类型）上观察到的空间属性或分布中固有的异质性和变异。此前的技术主要关注自监督任务，以学习跨域不变特征并减轻域差异；然而，它们往往忽略了数据点之间的空间分布，导致不同位置类型之间存在显著差异。我们探索了一种新的多任务自学习框架，针对空间分布（如空间混扰掩蔽和空间对比预测编码）进行研究，以适应空间分区的人工智能分类。实验结果表明，在实际数据集（例如肿瘤学数据）上的性能优于基线方法。 

---
# StAyaL | Multilingual Style Transfer 

**Title (ZH)**: StAyaL | 多语言风格转换 

**Authors**: Karishma Thakrar, Katrina Lawrence, Kyle Howard  

**Link**: [PDF](https://arxiv.org/pdf/2501.11639)  

**Abstract**: Stylistic text generation plays a vital role in enhancing communication by reflecting the nuances of individual expression. This paper presents a novel approach for generating text in a specific speaker's style across different languages. We show that by leveraging only 100 lines of text, an individuals unique style can be captured as a high-dimensional embedding, which can be used for both text generation and stylistic translation. This methodology breaks down the language barrier by transferring the style of a speaker between languages. The paper is structured into three main phases: augmenting the speaker's data with stylistically consistent external sources, separating style from content using machine learning and deep learning techniques, and generating an abstract style profile by mean pooling the learned embeddings. The proposed approach is shown to be topic-agnostic, with test accuracy and F1 scores of 74.9\% and 0.75, respectively. The results demonstrate the potential of the style profile for multilingual communication, paving the way for further applications in personalized content generation and cross-linguistic stylistic transfer. 

**Abstract (ZH)**: 风格化文本生成在反映个人表达的细微差别方面在增强交流中发挥着重要作用。本文提出了一种新颖的方法，用于跨语言生成特定说话者风格的文本。我们展示了仅通过利用100行文本，就可以将个体独特的风格捕获为高维嵌入，且该嵌入可用于文本生成和风格化翻译。该方法通过在不同语言之间转移说话者的风格，打破了语言障碍。本文结构主要包括三个主要阶段：首先通过风格上一致的外部数据扩展说话者的数据；然后使用机器学习和深度学习技术将风格与内容分离；最后通过平均池化学习到的嵌入生成抽象风格概要。所提出的方法在各个主题上都是通用的，测试准确率为74.9%，F1评分为0.75。结果证明了风格概要在多语言交流中的潜在价值，为进一步在个性化内容生成和跨语言风格转移方面的应用奠定了基础。 

---
# Biomedical Knowledge Graph: A Survey of Domains, Tasks, and Real-World Applications 

**Title (ZH)**: biomedical 知识图谱：对领域、任务及实际应用的综述 

**Authors**: Yuxing Lu, Sin Yee Goi, Xukai Zhao, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11632)  

**Abstract**: Biomedical knowledge graphs (BKGs) have emerged as powerful tools for organizing and leveraging the vast and complex data found across the biomedical field. Yet, current reviews of BKGs often limit their scope to specific domains or methods, overlooking the broader landscape and the rapid technological progress reshaping it. In this survey, we address this gap by offering a systematic review of BKGs from three core perspectives: domains, tasks, and applications. We begin by examining how BKGs are constructed from diverse data sources, including molecular interactions, pharmacological datasets, and clinical records. Next, we discuss the essential tasks enabled by BKGs, focusing on knowledge management, retrieval, reasoning, and interpretation. Finally, we highlight real-world applications in precision medicine, drug discovery, and scientific research, illustrating the translational impact of BKGs across multiple sectors. By synthesizing these perspectives into a unified framework, this survey not only clarifies the current state of BKG research but also establishes a foundation for future exploration, enabling both innovative methodological advances and practical implementations. 

**Abstract (ZH)**: 生物医学知识图谱（BKGs）已成为组织和利用生物医学领域广泛而复杂的数据的强大工具。然而，目前对BKGs的综述往往局限于特定的领域或方法，未能充分涵盖这一领域的广泛图景和快速的技术进步。在这篇综述文章中，我们通过从三个核心视角——领域、任务和应用——对其进行全面系统的回顾来填补这一空白。我们首先探讨BKGs如何从多样化的数据源构建而成，包括分子互作、药理数据集和临床记录。接着，我们讨论BKGs支持的重要任务，重点关注知识管理、检索、推理和解释。最后，我们强调了BKGs在精准医疗、药物发现和科学研究中的实际应用，展示了BKGs在多个领域中的转化影响。通过将这些视角整合到一个统一框架中，这篇综述不仅阐明了当前BKG研究的状态，还为未来的探索奠定了基础，既便于推动创新方法学的进步，又便于实际应用。 

---
# Noise-Agnostic Multitask Whisper Training for Reducing False Alarm Errors in Call-for-Help Detection 

**Title (ZH)**: _noise-agnostic 多任务语音识别训练以减少求助检测中的误报错误_ 

**Authors**: Myeonghoon Ryu, June-Woo Kim, Minseok Oh, Suji Lee, Han Park  

**Link**: [PDF](https://arxiv.org/pdf/2501.11631)  

**Abstract**: Keyword spotting is often implemented by keyword classifier to the encoder in acoustic models, enabling the classification of predefined or open vocabulary keywords. Although keyword spotting is a crucial task in various applications and can be extended to call-for-help detection in emergencies, however, the previous method often suffers from scalability limitations due to retraining required to introduce new keywords or adapt to changing contexts. We explore a simple yet effective approach that leverages off-the-shelf pretrained ASR models to address these challenges, especially in call-for-help detection scenarios. Furthermore, we observed a substantial increase in false alarms when deploying call-for-help detection system in real-world scenarios due to noise introduced by microphones or different environments. To address this, we propose a novel noise-agnostic multitask learning approach that integrates a noise classification head into the ASR encoder. Our method enhances the model's robustness to noisy environments, leading to a significant reduction in false alarms and improved overall call-for-help performance. Despite the added complexity of multitask learning, our approach is computationally efficient and provides a promising solution for call-for-help detection in real-world scenarios. 

**Abstract (ZH)**: 关键词定位通常通过将关键词分类器应用于声学模型中的编码器来实现，从而能够对预定义或开放词汇的关键词进行分类。虽然关键词定位在各种应用中是一个关键任务，并且可以扩展到紧急情况下的呼救检测，但由于需要重新训练以引入新关键词或适应变化的上下文，之前的许多方法往往存在扩展性限制。我们探讨了一种简单但有效的方法，利用现成的预训练ASR模型来解决这些挑战，尤其是在呼救检测场景中的问题。此外，在实际场景中部署呼救检测系统时，由于麦克风或不同环境引入的噪声，我们观察到虚假警报显著增加。为此，我们提出了一种新颖的噪声无关多任务学习方法，将噪声分类头集成到ASR编码器中。我们的方法增强了模型在嘈杂环境中鲁棒性，显著减少了虚假警报，并提高了整体呼救检测性能。尽管多任务学习增加了模型的复杂性，但我们的方法在计算效率方面表现出色，并为实际场景中的呼救检测提供了有前景的解决方案。 

---
# Early evidence of how LLMs outperform traditional systems on OCR/HTR tasks for historical records 

**Title (ZH)**: 早期证据表明，大规模语言模型在历史记录的OCR/HTR任务中优于传统系统 

**Authors**: Seorin Kim, Julien Baudru, Wouter Ryckbosch, Hugues Bersini, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2501.11623)  

**Abstract**: We explore the ability of two LLMs -- GPT-4o and Claude Sonnet 3.5 -- to transcribe historical handwritten documents in a tabular format and compare their performance to traditional OCR/HTR systems: EasyOCR, Keras, Pytesseract, and TrOCR. Considering the tabular form of the data, two types of experiments are executed: one where the images are split line by line and the other where the entire scan is used as input. Based on CER and BLEU, we demonstrate that LLMs outperform the conventional OCR/HTR methods. Moreover, we also compare the evaluated CER and BLEU scores to human evaluations to better judge the outputs of whole-scan experiments and understand influential factors for CER and BLEU. Combining judgments from all the evaluation metrics, we conclude that two-shot GPT-4o for line-by-line images and two-shot Claude Sonnet 3.5 for whole-scan images yield the transcriptions of the historical records most similar to the ground truth. 

**Abstract (ZH)**: 我们探讨了两种大语言模型——GPT-4o 和 Claude Sonnet 3.5——在将历史手写文档转换为表格格式方面的能力，并将其性能与传统的OCR/HTR系统（如EasyOCR、Keras、Pytesseract 和 TrOCR）进行了比较。鉴于数据的表格形式，我们执行了两种类型的实验：一种是将图像逐行分割，另一种是使用整个扫描作为输入。基于字符错误率（CER）和BLEU分数，我们展示了大语言模型在某些方面优于传统的OCR/HTR方法。此外，我们还将评估的CER和BLEU分数与人工评估进行了比较，以更好地判断整体扫描实验的输出结果，并理解CER和BLEU的影响因素。综合所有评估指标的判断，我们得出结论：对于逐行图像使用两轮制GPT-4o，而对于整体扫描图像使用两轮制Claude Sonnet 3.5，生成的历史记录转录结果与真实值最为接近。 

---
# Conversation Routines: A Prompt Engineering Framework for Task-Oriented Dialog Systems 

**Title (ZH)**: 对话惯例：面向任务导向的对话系统的一种提示工程框架 

**Authors**: Giorgio Robino  

**Link**: [PDF](https://arxiv.org/pdf/2501.11613)  

**Abstract**: This study introduces Conversation Routines (CR), a structured prompt engineering framework for developing task-oriented dialog systems using Large Language Models (LLMs). While LLMs demonstrate remarkable natural language understanding capabilities, engineering them to reliably execute complex business workflows remains challenging. The proposed CR framework enables the development of Conversation Agentic Systems (CAS) through natural language specifications, embedding task-oriented logic within LLM prompts. This approach provides a systematic methodology for designing and implementing complex conversational workflows while maintaining behavioral consistency. We demonstrate the framework's effectiveness through two proof of concept implementations: a Train Ticket Booking System and an Interactive Troubleshooting Copilot. These case studies validate CR's capability to encode sophisticated behavioral patterns and decision logic while preserving natural conversational flexibility. Results show that CR enables domain experts to design conversational workflows in natural language while leveraging custom enterprise functionalities (tools) developed by software engineers, creating an efficient division of responsibilities where developers focus on core API implementation and domain experts handle conversation design. While the framework shows promise in accessibility and adaptability, we identify key challenges including computational overhead, non-deterministic behavior, and domain-specific logic optimization. Future research directions include enhancing system robustness, improving scalability for complex multi-agent interactions, and addressing the identified limitations across diverse business applications. 

**Abstract (ZH)**: 本研究介绍了对话例行程序（CR），这是一个结构化的提示工程框架，用于使用大规模语言模型（LLMs）开发面向任务的对话系统。尽管LLMs展现出卓越的自然语言理解能力，但将它们工程化以可靠地执行复杂的业务工作流仍然具有挑战性。所提出的CR框架通过自然语言规范使开发者能够构建对话代理系统（CAS），并将任务导向的逻辑嵌入到LLM提示中。这种方法提供了一种系统的方法来设计和实现复杂的对话工作流，同时保持行为一致性。我们通过两个概念验证实现展示了该框架的有效性：一个火车票预订系统和一个交互式故障排除副驾。这些案例研究验证了CR能够编码复杂的行为模式和决策逻辑，同时保持自然对话的灵活性。结果显示，CR使领域专家能够使用自然语言设计对话工作流，同时利用软件工程师开发的定制企业功能（工具），从而形成一种高效的职责分工，其中开发人员专注于核心API的实现，而领域专家则负责对话设计。尽管该框架在易用性和适应性方面显示出前景，但我们仍识别出一些关键挑战，包括计算开销、非确定性行为以及特定领域的逻辑优化。未来的研究方向包括增强系统的稳健性、改进多代理交互的可扩展性，并解决在不同商业应用场景中识别出的限制。 

---
# Fairness Testing through Extreme Value Theory 

**Title (ZH)**: 通过极端值理论进行公平性测试 

**Authors**: Verya Monjezi, Ashutosh Trivedi, Vladik Kreinovich, Saeid Tizpaz-Niari  

**Link**: [PDF](https://arxiv.org/pdf/2501.11597)  

**Abstract**: Data-driven software is increasingly being used as a critical component of automated decision-support systems. Since this class of software learns its logic from historical data, it can encode or amplify discriminatory practices. Previous research on algorithmic fairness has focused on improving average-case fairness. On the other hand, fairness at the extreme ends of the spectrum, which often signifies lasting and impactful shifts in societal attitudes, has received significantly less emphasis.
Leveraging the statistics of extreme value theory (EVT), we propose a novel fairness criterion called extreme counterfactual discrimination (ECD). This criterion estimates the worst-case amounts of disadvantage in outcomes for individuals solely based on their memberships in a protected group. Utilizing tools from search-based software engineering and generative AI, we present a randomized algorithm that samples a statistically significant set of points from the tail of ML outcome distributions even if the input dataset lacks a sufficient number of relevant samples.
We conducted several experiments on four ML models (deep neural networks, logistic regression, and random forests) over 10 socially relevant tasks from the literature on algorithmic fairness. First, we evaluate the generative AI methods and find that they generate sufficient samples to infer valid EVT distribution in 95% of cases. Remarkably, we found that the prevalent bias mitigators reduce the average-case discrimination but increase the worst-case discrimination significantly in 5% of cases. We also observed that even the tail-aware mitigation algorithm -- MiniMax-Fairness -- increased the worst-case discrimination in 30% of cases. We propose a novel ECD-based mitigator that improves fairness in the tail in 90% of cases with no degradation of the average-case discrimination. 

**Abstract (ZH)**: 基于数据的软件越来越多地被用作自动化决策支持系统的关键组成部分。由于这类软件是从历史数据中学习其逻辑的，因此可能会编码或放大歧视行为。以往关于算法公平性的研究主要集中在提高平均情况下的公平性。相反，极端情况下的公平性，这通常反映了社会态度持续而深远的变化，却得到了较少的关注。

利用极端值理论（EVT）中的统计方法，我们提出了一个新颖的公平性标准，称为极端反事实歧视（ECD）。该标准基于个人所属的保护组，估计最差情况下结果中的不利情况。利用基于搜索的软件工程和生成式AI工具，我们提出了一种随机算法，即使输入数据集缺乏足够的相关样本，也能从机器学习结果分布的尾部采样出统计上显著的点集。

我们在文献中提出的四个机器学习模型（深度神经网络、逻辑回归和随机森林）上对十个社会相关任务进行了多项实验。首先，我们评估了生成式AI方法，发现它们在95%的情况下能够生成足够的样本以推断有效的EVT分布。令人惊讶的是，我们发现普遍存在的偏见缓解措施在5%的情况下会减少平均情况下的歧视，但会显著增加最差情况下的歧视。我们还观察到，即使对于尾部意识的缓解算法—MiniMax公平性—在30%的情况下也会增加最差情况下的歧视。我们提出了一种基于ECD的新颖缓解措施，在90%的情况下能够改善极端情况下的公平性，而不会影响平均情况下的公平性。 

---
# Training-free Ultra Small Model for Universal Sparse Reconstruction in Compressed Sensing 

**Title (ZH)**: 基于压缩感知的通用稀疏重建超小模型训练-free技术 

**Authors**: Chaoqing Tang, Huanze Zhuang, Guiyun Tian, Zhenli Zeng, Yi Ding, Wenzhong Liu, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2501.11592)  

**Abstract**: Pre-trained large models attract widespread attention in recent years, but they face challenges in applications that require high interpretability or have limited resources, such as physical sensing, medical imaging, and bioinformatics. Compressed Sensing (CS) is a well-proved theory that drives many recent breakthroughs in these applications. However, as a typical under-determined linear system, CS suffers from excessively long sparse reconstruction times when using traditional iterative methods, particularly with large-scale data. Current AI methods like deep unfolding fail to substitute them because pre-trained models exhibit poor generality beyond their training conditions and dataset distributions, or lack interpretability. Instead of following the big model fervor, this paper proposes ultra-small artificial neural models called coefficients learning (CL), enabling training-free and rapid sparse reconstruction while perfectly inheriting the generality and interpretability of traditional iterative methods, bringing new feature of incorporating prior knowledges. In CL, a signal of length $n$ only needs a minimal of $n$ trainable parameters. A case study model called CLOMP is implemented for evaluation. Experiments are conducted on both synthetic and real one-dimensional and two-dimensional signals, demonstrating significant improvements in efficiency and accuracy. Compared to representative iterative methods, CLOMP improves efficiency by 100 to 1000 folds for large-scale data. Test results on eight diverse image datasets indicate that CLOMP improves structural similarity index by 292%, 98%, 45% for sampling rates of 0.1, 0.3, 0.5, respectively. We believe this method can truly usher CS reconstruction into the AI era, benefiting countless under-determined linear systems that rely on sparse solution. 

**Abstract (ZH)**: 近年来，预训练大规模模型引起了广泛关注，但在需要高解释性或资源受限的应用中（如物理传感、医学成像和生物信息学）面临挑战。压缩感知（CS）是一种已被证明的有效理论，推动了这些领域的许多最新突破。然而，作为典型的欠定线性系统，CS 在使用传统迭代方法时，特别是在大规模数据下，会导致重建时间过长的问题。当前的人工智能方法（如深度展开）无法替代它们，因为预训练模型在训练条件和数据分布之外表现出较差的一般性和解释性。本文没有盲目追求大规模模型的趋势，而是提出了超小型人工神经网络模型——系数学习（CL），该模型能够在不训练的情况下实现快速稀疏重构，同时完美地继承了传统迭代方法的一般性和解释性，并引入了融入先验知识的新特性。在CL中，长度为 \(n\) 的信号只需要最小 \(n\) 个可训练参数即可。该论文实施了一个名为CLOMP的案例研究模型进行评估。实验分别在合成和实际的一维和二维信号上进行，显示出在效率和准确性方面的显著改进。与代表性的迭代方法相比，CLOMP在大规模数据上的效率提高了100到1000倍。在八种不同的图像数据集上的测试结果显示，CLOMP在采样率为0.1、0.3、0.5的情况下，分别提高了结构相似性指数392%、198%、95%。我们相信，这种方法可以使CS重建真正进入AI时代，并为依赖稀疏解的众多欠定线性系统带来益处。 

---
# Recurrent Diffusion for Large-Scale Parameter Generation 

**Title (ZH)**: 大规模参数生成的循环扩散方法 

**Authors**: Kai Wang, Dongwen Tang, Wangbo Zhao, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2501.11587)  

**Abstract**: Parameter generation has struggled to scale up for a long time, significantly limiting its range of applications. In this study, we introduce \textbf{R}ecurrent diffusion for large-scale \textbf{P}arameter \textbf{G}eneration, called \textbf{RPG}. We first divide the trained parameters into non-overlapping parts, after which a recurrent model is proposed to learn their relationships. The recurrent model's outputs, as conditions, are then fed into a diffusion model to generate the neural network parameters. Using only a single GPU, recurrent diffusion enables us to generate popular vision and language models such as ConvNeXt-L and LoRA parameters of LLaMA-7B. Meanwhile, across various architectures and tasks, the generated parameters consistently perform comparable results over trained networks. Notably, our approach also shows the potential to generate models for handling unseen tasks, which largely increases the practicality of parameter generation. Our code is available \href{this https URL}{here}. 

**Abstract (ZH)**: 参数生成长期以来一直难以扩展，显著限制了其应用范围。在本研究中，我们提出了一种名为**RPG（Recurrent Diffusion for Large-scale Parameter Generation）**的方法。我们首先将训练得到的参数划分为不重叠的部分，之后提出了一种递归模型来学习这些部分之间的关系。递归模型的输出作为条件，然后输入到一个扩散模型中，以生成神经网络参数。仅使用单个GPU，递归扩散方法使我们能够生成如ConvNeXt-L和LLaMA-7B的LoRA参数等流行的视觉和语言模型。同时，无论架构和任务如何变化，生成的参数在性能上都与训练过的网络相当。值得注意的是，我们的方法还展示了生成处理未见过的任务模型的潜力，这大大增加了参数生成的实际应用性。我们的代码可在以下链接中获取：[这里](this https URL)。 

---
# Explainable Lane Change Prediction for Near-Crash Scenarios Using Knowledge Graph Embeddings and Retrieval Augmented Generation 

**Title (ZH)**: 使用知识图嵌入和检索增强生成的可解释变道预测方法及其在近碰撞场景中的应用 

**Authors**: M. Manzour, A. Ballardini, R. Izquierdo, M. Á. Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2501.11560)  

**Abstract**: Lane-changing maneuvers, particularly those executed abruptly or in risky situations, are a significant cause of road traffic accidents. However, current research mainly focuses on predicting safe lane changes. Furthermore, existing accident datasets are often based on images only and lack comprehensive sensory data. In this work, we focus on predicting risky lane changes using the CRASH dataset (our own collected dataset specifically for risky lane changes), and safe lane changes (using the HighD dataset). Then, we leverage KG and Bayesian inference to predict these maneuvers using linguistic contextual information, enhancing the model's interpretability and transparency. The model achieved a 91.5% f1-score with anticipation time extending to four seconds for risky lane changes, and a 90.0% f1-score for predicting safe lane changes with the same anticipation time. We validate our model by integrating it into a vehicle within the CARLA simulator in scenarios that involve risky lane changes. The model managed to anticipate sudden lane changes, thus providing automated vehicles with further time to plan and execute appropriate safe reactions. Finally, to enhance the explainability of our model, we utilize RAG to provide clear and natural language explanations for the given prediction. 

**Abstract (ZH)**: 车道变换行为，尤其是那些在突然或高风险情况下执行的变换，是道路交通事故的重要原因。然而，当前的研究主要集中在预测安全的车道变换。此外，现有的事故数据集通常仅基于图像数据，缺乏全面的感官数据。在本研究中，我们专注于使用CRASH数据集（一个专门为研究高风险车道变换收集的数据集）和HighD数据集来预测高风险和安全的车道变换。我们利用知识图谱（KG）和贝叶斯推理结合语义上下文信息来预测这些行为，以增强模型的可解释性和透明度。该模型在预警时间延长至四秒的情况下，在预测高风险车道变换时达到了91.5%的F1分数，在预测相同预警时间的安全车道变换时达到了90.0%的F1分数。

我们通过将模型集成到CARLA模拟器中的车辆中，在包含高风险车道变换的场景中验证了该模型，成功预见了突然的车道变换，从而为自动驾驶汽车提供了更多时间来规划和执行适当的避险措施。最后，为了进一步提高模型的可解释性，我们利用RAG（Reading-Aware Generation）技术，为给定的预测提供清晰且自然的解释。 

---
# Meta-Instance Selection. Instance Selection as a Classification Problem with Meta-Features 

**Title (ZH)**: 元实例选择：将实例选择问题转化为具有元特征的分类问题 

**Authors**: Marcin Blachnik, Piotr Ciepliński  

**Link**: [PDF](https://arxiv.org/pdf/2501.11526)  

**Abstract**: Data pruning, or instance selection, is an important problem in machine learning especially in terms of nearest neighbour classifier. However, in data pruning which speeds up the prediction phase, there is an issue related to the speed and efficiency of the process itself. In response, the study proposes an approach involving transforming the instance selection process into a classification task conducted in a unified meta-feature space where each instance can be classified and assigned to either the "to keep" or "to remove" class. This approach requires training an appropriate meta-classifier, which can be developed based on historical instance selection results from other datasets using reference instance selection methods as a labeling tool. This work proposes constructing the meta-feature space based on properties extracted from the nearest neighbor graph. Experiments conducted on 17 datasets of varying sizes and five reference instance selection methods (ENN, Drop3, ICF, HMN-EI, and CCIS) demonstrate that the proposed solution achieves results comparable to reference instance selection methods while significantly reducing computational complexity. In the proposed approach, the computational complexity of the system depends only on identifying the k-nearest neighbors for each data sample and running the meta-classifier. Additionally, the study discusses the choice of meta-classifier, recommending the use of Balanced Random Forest. 

**Abstract (ZH)**: 数据修剪（或实例选择）是机器学习中的一个重要问题，特别是在最近邻分类器中。然而，在数据修剪过程中，用于加快预测阶段的进程速度本身也存在效率和速度的问题。为此，本研究提出了一种方法，即将实例选择过程转化为在统一的元特征空间中进行的分类任务。在这个空间中，每个实例都可以被分类并分别归属于“保留”或“删除”类。该方法需要训练一个合适的元分类器，可以通过使用参考实例选择方法的历史实例选择结果作为标签工具来开发。本研究建议基于从最近邻图中提取的属性来构建元特征空间。在17个不同规模的数据集上进行的实验，并使用五种参考实例选择方法（ENN、Drop3、ICF、HMN-EI和CCIS）表明，提出的解决方案在计算复杂度显著降低的同时，达到了与参考实例选择方法相当的结果。在提出的方法中，系统的计算复杂度仅取决于确定每个数据样本的k个最近邻居以及运行元分类器。此外，本研究还讨论了元分类器的选择，建议使用平衡随机森林。 

---
# Technical Report for the Forgotten-by-Design Project: Targeted Obfuscation for Machine Learning 

**Title (ZH)**: 面向设计被遗忘项目的技术报告：面向目标的机器学习混淆技术 

**Authors**: Rickard Brännvall, Laurynas Adomaitis, Olof Görnerup, Anass Sedrati  

**Link**: [PDF](https://arxiv.org/pdf/2501.11525)  

**Abstract**: The right to privacy, enshrined in various human rights declarations, faces new challenges in the age of artificial intelligence (AI). This paper explores the concept of the Right to be Forgotten (RTBF) within AI systems, contrasting it with traditional data erasure methods. We introduce Forgotten by Design, a proactive approach to privacy preservation that integrates instance-specific obfuscation techniques during the AI model training process. Unlike machine unlearning, which modifies models post-training, our method prevents sensitive data from being embedded in the first place. Using the LIRA membership inference attack, we identify vulnerable data points and propose defenses that combine additive gradient noise and weighting schemes. Our experiments on the CIFAR-10 dataset demonstrate that our techniques reduce privacy risks by at least an order of magnitude while maintaining model accuracy (at 95% significance). Additionally, we present visualization methods for the privacy-utility trade-off, providing a clear framework for balancing privacy risk and model accuracy. This work contributes to the development of privacy-preserving AI systems that align with human cognitive processes of motivated forgetting, offering a robust framework for safeguarding sensitive information and ensuring compliance with privacy regulations. 

**Abstract (ZH)**: 隐私权，作为各种人权宣言的一部分，正面临人工智能（AI）时代的新挑战。本文探讨了在AI系统中“被遗忘权”（Right to Be Forgotten, RTBF）的概念，并将其与传统的数据删除方法进行了对比。我们提出了“设计即遗忘”（Forgotten by Design）这一主动的隐私保护方法，该方法在AI模型训练过程中整合了实例特异性的模糊化技术。与机器遗忘方法不同，后者在模型训练完成后进行修改，我们的方法则防止敏感数据从一开始就被嵌入模型中。通过使用LIRA成员推理攻击，我们识别出了易受攻击的数据点，并提出了结合加性梯度噪声和权重方案的防御措施。我们的实验在CIFAR-10数据集上表明，我们的技术将隐私风险至少减少了一个数量级，同时保持了模型的准确率（以95%的显著性水平）。此外，我们还介绍了隐私-效用权衡的可视化方法，提供了一种明确的框架来平衡隐私风险和模型准确率。本文为开发与人类动机遗忘机制相一致的隐私保护AI系统做出了贡献，并提供了一个强大的框架，以保护敏感信息和确保遵守隐私法规。 

---
# Dialect2SQL: A Novel Text-to-SQL Dataset for Arabic Dialects with a Focus on Moroccan Darija 

**Title (ZH)**: 方言2SQL：以摩洛哥达里亚拉为重点的阿拉伯方言文本到SQL数据集novelty 

**Authors**: Salmane Chafik, Saad Ezzini, Ismail Berrada  

**Link**: [PDF](https://arxiv.org/pdf/2501.11498)  

**Abstract**: The task of converting natural language questions (NLQs) into executable SQL queries, known as text-to-SQL, has gained significant interest in recent years, as it enables non-technical users to interact with relational databases. Many benchmarks, such as SPIDER and WikiSQL, have contributed to the development of new models and the evaluation of their performance. In addition, other datasets, like SEDE and BIRD, have introduced more challenges and complexities to better map real-world scenarios. However, these datasets primarily focus on high-resource languages such as English and Chinese. In this work, we introduce Dialect2SQL, the first large-scale, cross-domain text-to-SQL dataset in an Arabic dialect. It consists of 9,428 NLQ-SQL pairs across 69 databases in various domains. Along with SQL-related challenges such as long schemas, dirty values, and complex queries, our dataset also incorporates the complexities of the Moroccan dialect, which is known for its diverse source languages, numerous borrowed words, and unique expressions. This demonstrates that our dataset will be a valuable contribution to both the text-to-SQL community and the development of resources for low-resource languages. 

**Abstract (ZH)**: 将自然语言问题（NLQs）转换为可执行的SQL查询的任务，即文本到SQL（Text-to-SQL），近年来引起了广泛关注，因为它使非技术人员能够与关系数据库进行交互。许多基准测试，如SPIDER和WikiSQL，为新模型的开发和性能评估做出了贡献。此外，其他数据集，如SEDE和BIRD，引入了更多的挑战和复杂性，以便更好地映射现实世界的情景。然而，这些数据集主要集中在资源丰富的语言上，如英语和中文。在此项研究中，我们介绍了Dialect2SQL，这是首个大规模跨领域阿拉伯方言的文本到SQL数据集。该数据集包含涵盖了69个不同领域的9,428个NLQ-SQL配对。除了包括SQL相关的挑战，如长模式、脏值和复杂查询外，我们的数据集还包含了摩洛哥方言的复杂性，摩洛哥方言以其多种来源语言、大量的借用词汇和独特的表达方式而著称。这表明我们的数据集将为文本到SQL社区以及低资源语言资源的发展做出重要贡献。 

---
# Generative AI and Large Language Models in Language Preservation: Opportunities and Challenges 

**Title (ZH)**: 生成式人工智能与大型语言模型在语言保护中的机遇与挑战 

**Authors**: Vincent Koc  

**Link**: [PDF](https://arxiv.org/pdf/2501.11496)  

**Abstract**: Generative AI and large-scale language models (LLM) have emerged as powerful tools in language preservation, particularly for near-native and endangered languages. With the increasing reliance on technology for communication, education, and cultural documentation, new opportunities have emerged to mitigate the dramatic decline of linguistic diversity worldwide. This paper examines the role of generative AIs and LLMs in preserving endangered languages, highlighting the risks and challenges associated with their use. We analyze the underlying technologies driving these models, including natural language processing (NLP) and deep learning, and explore several cases where these technologies have been applied to low-resource languages. Additionally, we discuss ethical considerations, data scarcity issues, and technical challenges while proposing solutions to enhance AI-driven language preservation. 

**Abstract (ZH)**: 生成式人工智能和大规模语言模型（LLM）已成为语言保护方面的强大工具，特别适用于濒临消失和濒危语言。随着技术在沟通、教育和文化记录方面依赖的不断增加，出现了新的机会来缓解全球语言多样性的急剧下降。本文探讨了生成式人工智能和大规模语言模型在保护濒危语言方面的作用，同时指出了使用这些工具所伴随的风险和挑战。我们分析了驱动这些模型的底层技术，包括自然语言处理（NLP）和深度学习，并探讨了这些技术在低资源语言中的应用案例。此外，我们还讨论了伦理考量、数据稀缺问题和技术挑战，并提出了增强人工智能驱动的语言保护的解决方案。 

---
# Communication-Efficient Federated Learning Based on Explanation-Guided Pruning for Remote Sensing Image Classification 

**Title (ZH)**: 基于解释引导裁剪的远程 sensing 图像分类的高效通信联邦学习 

**Authors**: Jonas Klotz, Barış Büyüktaş, Begüm Demir  

**Link**: [PDF](https://arxiv.org/pdf/2501.11493)  

**Abstract**: Federated learning (FL) is a decentralized machine learning paradigm, where multiple clients collaboratively train a global model by exchanging only model updates with the central server without sharing the local data of clients. Due to the large volume of model updates required to be transmitted between clients and the central server, most FL systems are associated with high transfer costs (i.e., communication overhead). This issue is more critical for operational applications in remote sensing (RS), especially when large-scale RS data is processed and analyzed through FL systems with restricted communication bandwidth. To address this issue, we introduce an explanation-guided pruning strategy for communication-efficient FL in the context of RS image classification. Our pruning strategy is defined based on the layerwise relevance propagation (LRP) driven explanations to: 1) efficiently and effectively identify the most relevant and informative model parameters (to be exchanged between clients and the central server); and 2) eliminate the non-informative ones to minimize the volume of model updates. The experimental results on the BigEarthNet-S2 dataset demonstrate that our strategy effectively reduces the number of shared model updates, while increasing the generalization ability of the global model. The code of this work will be publicly available at this https URL 

**Abstract (ZH)**: 联邦学习（FL）是一种去中心化的机器学习范式，在这种范式中，多个客户端通过与中心服务器交换模型更新（而非本地数据）的方式协作训练一个全局模型。由于需要在客户端与中心服务器之间传输大量模型更新，大多数FL系统通常伴随较高的传输成本（即通信开销）。这一问题在遥感（RS）的应用中尤为重要，尤其是在利用通信带宽受限的FL系统处理和分析大规模RS数据时。为了解决这一问题，我们提出了一种基于解释的剪枝策略，以实现高效的通信高效联邦学习，特别是在遥感图像分类的背景下。我们的剪枝策略基于层间相关性传播（LRP）驱动的解释来实现以下两个目标：1) 有效地识别最相关的和最有信息量的模型参数（客户端和中心服务器之间需要交换的信息）；2) 消除无信息量的参数，以最小化模型更新的体积。在BigEarthNet-S2数据集上的实验结果证明，我们的策略能够有效地减少共享模型更新的数量，同时提高全局模型的泛化能力。该工作的代码将在以下网址公开：[此处https网址] 

---
# Graph-defined Language Learning with LLMs 

**Title (ZH)**: 基于图定义的语言学习：利用大规模语言模型 

**Authors**: Huachi Zhou, Jiahe Du, Chuang Zhou, Chang Yang, Yilin Xiao, Yuxuan Xie, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11478)  

**Abstract**: Recent efforts leverage Large Language Models (LLMs) for modeling text-attributed graph structures in node classification tasks. These approaches describe graph structures for LLMs to understand or aggregate LLM-generated textual attribute embeddings through graph structure. However, these approaches face two main limitations in modeling graph structures with LLMs. (i) Graph descriptions become verbose in describing high-order graph structure. (ii) Textual attributes alone do not contain adequate graph structure information. It is challenging to model graph structure concisely and adequately with LLMs. LLMs lack built-in mechanisms to model graph structures directly. They also struggle with complex long-range dependencies between high-order nodes and target nodes.
Inspired by the observation that LLMs pre-trained on one language can achieve exceptional performance on another with minimal additional training, we propose \textbf{G}raph-\textbf{D}efined \textbf{L}anguage for \textbf{L}arge \textbf{L}anguage \textbf{M}odel (GDL4LLM). This novel framework enables LLMs to transfer their powerful language understanding capabilities to graph-structured data. GDL4LLM translates graphs into a graph language corpus instead of graph descriptions and pre-trains LLMs on this corpus to adequately understand graph structures. During fine-tuning, this corpus describes the structural information of target nodes concisely with only a few tokens. By treating graphs as a new language, GDL4LLM enables LLMs to model graph structures adequately and concisely for node classification tasks. Extensive experiments on three real-world datasets demonstrate that GDL4LLM outperforms description-based and textual attribute embeddings-based baselines by efficiently modeling different orders of graph structure with LLMs. 

**Abstract (ZH)**: 近年来，研究人员利用大规模语言模型（LLMs）对节点分类任务中的文本属性图结构进行建模。这些方法通过图结构帮助LLMs理解或聚合其生成的文本属性嵌入。然而，这些方法在利用LLMs建模图结构时面临两个主要局限性。（i）在描述高阶图结构时，图描述变得冗长。（ii）仅依靠文本属性无法提供足够的图结构信息。用LLMs建模图结构既缺乏简洁性又不够充分。LLMs缺乏直接建模图结构的内置机制，同时在处理高阶节点和目标节点之间的复杂长程依赖方面也存在困难。

受观察到的LLMs在预训练于一种语言时，仅通过少量额外训练就能在另一种语言上取得出色表现的启发，我们提出了一种名为 **G**raph-**D**efined **L**anguage for **L**arge **L**anguage **M**odel（GDL4LLM）的新框架。该框架使LLMs能够将强大的语言理解能力转移到图结构数据上。GDL4LLM将图转换为图语言语料库而非描述，通过在该语料库上对LLMs进行预训练来充分理解图结构。在微调过程中，该语料库通过少量标记符简洁地描述目标节点的结构信息。通过将图视为一种新的语言，GDL4LLM使LLMs能够充分且简洁地建模图结构，以进行节点分类任务。在三个真实世界数据集上的广泛实验表明，GDL4LLM通过高效地利用LLMs建模不同层次的图结构，能够超越基于描述和基于文本属性嵌入的基线模型。 

---
# Improving thermal state preparation of Sachdev-Ye-Kitaev model with reinforcement learning on quantum hardware 

**Title (ZH)**: 使用强化学习在量子硬件上提高萨克-dev 祈塔ev 模型的热态制备 

**Authors**: Akash Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11454)  

**Abstract**: The Sachdev-Ye-Kitaev (SYK) model, known for its strong quantum correlations and chaotic behavior, serves as a key platform for quantum gravity studies. However, variationally preparing thermal states on near-term quantum processors for large systems (N>12, where N is the number of Majorana fermions) presents a significant challenge due to the rapid growth in the complexity of parameterized quantum circuits. This paper addresses this challenge by integrating reinforcement learning (RL) with convolutional neural networks, employing an iterative approach to optimize the quantum circuit and its parameters. The refinement process is guided by a composite reward signal derived from entropy and the expectation values of the SYK Hamiltonian. This approach reduces the number of CNOT gates by two orders of magnitude for systems N>10 compared to traditional methods like first-order Trotterization. We demonstrate the effectiveness of the RL framework in both noiseless and noisy quantum hardware environments, maintaining high accuracy in thermal state preparation. This work contributes to the advancement of a scalable, RL-based framework with applications for computations of thermal out-of-time-order correlators in quantum many-body systems and quantum gravity studies on near-term quantum hardware. 

**Abstract (ZH)**: 森德夫-耶-基塔耶夫（SYK）模型以其强大的量子相关性和混沌行为，成为量子引力研究的关键平台。然而，对于大型系统（N>12，其中N为马约纳拉费米子的数量），在近期内量子处理器上变异地准备热态密度矩阵面临着严峻的挑战，因为参数化量子电路的复杂性迅速增长。本文通过将强化学习（RL）与卷积神经网络相结合，采用迭代方法优化量子电路及其参数，成功应对了这一挑战。优化过程受到从熵和SYK哈密顿量的期望值中导出的复合奖励信号的指导。与传统方法（如一阶图腾化）相比，这种方法在N>10的系统中减少了约两个数量级的CNOT门。我们在无噪和有噪量子硬件环境中展示了RL框架的有效性，能够维持热态密度矩阵准备的高精度。本文贡献了一种可扩展的基于强化学习的框架，并将其应用于量子多体系统中的热异时关联计算以及近期内量子硬件上的量子引力研究。 

---
# A Survey on Diffusion Models for Anomaly Detection 

**Title (ZH)**: 对异常检测中扩散模型的研究综述 

**Authors**: Jing Liu, Zhenchao Ma, Zepu Wang, Yang Liu, Zehua Wang, Peng Sun, Liang Song, Bo Hu, Azzedine Boukerche, Victor C.M. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2501.11430)  

**Abstract**: Diffusion models (DMs) have emerged as a powerful class of generative AI models, showing remarkable potential in anomaly detection (AD) tasks across various domains, such as cybersecurity, fraud detection, healthcare, and manufacturing. The intersection of these two fields, termed diffusion models for anomaly detection (DMAD), offers promising solutions for identifying deviations in increasingly complex and high-dimensional data. In this survey, we systematically review recent advances in DMAD research and investigate their capabilities. We begin by presenting the fundamental concepts of AD and DMs, followed by a comprehensive analysis of classic DM architectures including DDPMs, DDIMs, and Score SDEs. We further categorize existing DMAD methods into reconstruction-based, density-based, and hybrid approaches, providing detailed examinations of their methodological innovations. We also explore the diverse tasks across different data modalities, encompassing image, time series, video, and multimodal data analysis. Furthermore, we discuss critical challenges and emerging research directions, including computational efficiency, model interpretability, robustness enhancement, edge-cloud collaboration, and integration with large language models. The collection of DMAD research papers and resources is available at this https URL. 

**Abstract (ZH)**: 扩散模型（DMs）作为生成AI模型的一个强大类别，在跨各种领域（如网络安全、欺诈检测、医疗健康和制造业）的异常检测（AD）任务中展现了显著的潜力。这两个领域的交叉领域——异常检测的扩散模型（Diffusion Models for Anomaly Detection, DMAD）——为复杂和高维数据中的偏差识别提供了有前景的解决方案。在本文综述中，我们系统地回顾了DMAD研究的最新进展，并探讨了其能力。首先，我们介绍了AD和DM的基本概念，随后对经典DM架构（包括DDPMs、DDIMs和Score SDEs）进行了全面分析。我们进一步将现有的DMAD方法分类为基于重构、基于密度和混合方法，并对它们的创新方法学进行了详细的探讨。我们还探讨了不同数据模态下的多样化任务，包括图像、时间序列、视频和多模态数据分析。此外，我们讨论了关键挑战和新兴的研究方向，包括计算效率、模型可解释性、稳健性增强、边缘-云协作以及与大规模语言模型的集成。DMAD研究论文和资源的集合可在以下网址找到：[此处提供网址]。 

---
# Enhancing Coronary Artery Calcium Scoring via Multi-Organ Segmentation on Non-Contrast Cardiac Computed Tomography 

**Title (ZH)**: 通过非对比心脏计算机断层扫描的多器官分割增强冠状动脉钙化评分 

**Authors**: Jakub Nalepa, Tomasz Bartczak, Mariusz Bujny, Jarosław Gośliński, Katarzyna Jesionek, Wojciech Malara, Filip Malawski, Karol Miszalski-Jamka, Patrycja Rewa, Marcin Kostur  

**Link**: [PDF](https://arxiv.org/pdf/2501.11428)  

**Abstract**: Despite coronary artery calcium scoring being considered a largely solved problem within the realm of medical artificial intelligence, this paper argues that significant improvements can still be made. By shifting the focus from pathology detection to a deeper understanding of anatomy, the novel algorithm proposed in the paper both achieves high accuracy in coronary artery calcium scoring and offers enhanced interpretability of the results. This approach not only aids in the precise quantification of calcifications in coronary arteries, but also provides valuable insights into the underlying anatomical structures. Through this anatomically-informed methodology, the paper shows how a nuanced understanding of the heart's anatomy can lead to more accurate and interpretable results in the field of cardiovascular health. We demonstrate the superior accuracy of the proposed method by evaluating it on an open-source multi-vendor dataset, where we obtain results at the inter-observer level, surpassing the current state of the art. Finally, the qualitative analyses show the practical value of the algorithm in such tasks as labeling coronary artery calcifications, identifying aortic calcifications, and filtering out false positive detections due to noise. 

**Abstract (ZH)**: 尽管冠状动脉钙化评分在医学人工智能领域被认为是一个基本解决的问题，本文认为仍有显著改进的空间。通过将重点从病理检测转移到对解剖结构更深入的理解，本文提出的新算法在冠状动脉钙化评分上实现了高水平的准确性，并提供了结果的增强可解释性。这种方法不仅有助于精确量化冠状动脉中的钙化，还提供了关于潜在解剖结构的重要见解。通过这种基于解剖的方法，本文展示了对心脏解剖结构的深入理解如何在心血管健康领域带来更准确和可解释的结果。我们通过在开源多厂商数据集上评估提出的方法，证明了其优越的准确性，获得的结果达到了不同观察者水平，超越了当前的先进水平。最后，定性的分析显示了该算法在冠状动脉钙化标记、主动脉钙化识别以及过滤噪声引起的假阳性检测方面的实际应用价值。 

---
# Multi-View Spectral Clustering for Graphs with Multiple View Structures 

**Title (ZH)**: 多视图谱聚类在具有多视图结构的图中的应用 

**Authors**: Yorgos Tsitsikas, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.11422)  

**Abstract**: Despite the fundamental importance of clustering, to this day, much of the relevant research is still based on ambiguous foundations, leading to an unclear understanding of whether or how the various clustering methods are connected with each other. In this work, we provide an additional stepping stone towards resolving such ambiguities by presenting a general clustering framework that subsumes a series of seemingly disparate clustering methods, including various methods belonging to the wildly popular spectral clustering framework. In fact, the generality of the proposed framework is additionally capable of shedding light to the largely unexplored area of multi-view graphs whose each view may have differently clustered nodes. In turn, we propose GenClus: a method that is simultaneously an instance of this framework and a generalization of spectral clustering, while also being closely related to k-means as well. This results in a principled alternative to the few existing methods studying this special type of multi-view graphs. Then, we conduct in-depth experiments, which demonstrate that GenClus is more computationally efficient than existing methods, while also attaining similar or better clustering performance. Lastly, a qualitative real-world case-study further demonstrates the ability of GenClus to produce meaningful clusterings. 

**Abstract (ZH)**: 尽管聚类的理论基础至关重要，但至今为止，相关领域的许多研究仍然基于模糊的基础之上，导致人们对各种聚类方法之间如何相互关联缺乏清晰的理解。在这项工作中，我们通过提出一个涵盖一系列看似不相关的聚类方法的一般框架，为解决这些模糊性提供了一块额外的基石。实际上，这种提出的框架还能够为大量未被探索的多视图图领域提供新的见解，其中每个视图中的节点可能具有不同的聚类结构。因此，我们提出了GenClus方法，该方法同时是此框架的一个实例，同时也是谱聚类的一种推广，同时也与k-means方法密切相关。这为研究这种特殊类型的多视图图提供了一种原则性的替代方法。随后，我们进行了深入的实验，结果显示GenClus在计算效率上优于现有方法，同时在聚类性能上表现出类似或更优的结果。最后，一个定性的现实世界案例进一步展示了GenClus生成有意义聚类的能力。 

---
# Neural Contextual Reinforcement Framework for Logical Structure Language Generation 

**Title (ZH)**: 基于神经上下文强化的学习结构语言生成框架 

**Authors**: Marcus Irvin, William Cooper, Edward Hughes, Jessica Morgan, Christopher Hamilton  

**Link**: [PDF](https://arxiv.org/pdf/2501.11417)  

**Abstract**: The Neural Contextual Reinforcement Framework introduces an innovative approach to enhancing the logical coherence and structural consistency of text generated by large language models. Leveraging reinforcement learning principles, the framework integrates custom reward functions and dynamic context alignment mechanisms to address challenges inherent in maintaining long-range dependencies across extended sequences. The architecture incorporates multi-head attention layers and hierarchical encoding modules, enabling the model to produce outputs that align closely with human expectations of logical structure and semantic flow. Quantitative evaluations across diverse datasets demonstrate substantial improvements in coherence metrics, perplexity reduction, and semantic alignment, showcasing the framework's ability to outperform baseline models in both general and domain-specific tasks. Qualitative analyses further highlight the framework's capacity to generate text with improved narrative clarity and reduced redundancy, reflecting its effectiveness in balancing fluency with structural precision. In addition to its performance gains, the framework exhibits robustness in handling noisy input data and scalability across varying model sizes, reinforcing its versatility in practical applications. Experimental results reveal that optimal context window sizes significantly influence coherence outcomes, showing the importance of architectural flexibility in adapting to diverse linguistic structures. Cross-lingual performance evaluations affirm the framework's adaptability to multiple languages, extending its utility beyond monolingual contexts. Resource efficiency analyses indicate a reduction in computational overhead compared to traditional approaches, emphasizing the practicality of the framework for large-scale deployment. 

**Abstract (ZH)**: 神经上下文强化框架提出了一种创新的方法，以增强大型语言模型生成文本的逻辑连贯性和结构一致性。该框架利用强化学习原理，结合定制的奖励函数和动态上下文对齐机制，解决在长序列中维持远距离依赖关系的挑战。该架构包括多头注意力层和分层编码模块，使模型能够生成与人类期望的逻辑结构和语义流更好的对齐的输出。在多种数据集上的定量评估显示，在连贯性指标、困惑度降低和语义对齐方面取得了显著改进，展示了框架在通用任务和特定领域任务中均优于基线模型的能力。进一步的定性分析表明，该框架能够生成改进叙事清晰度和减少冗余性的文本，反映出其在流畅性和结构精度之间实现平衡方面的有效性。除了性能提升外，该框架还展现出在处理噪声输入数据和跨不同模型规模的扩展方面表现出的鲁棒性，增强了其实用性。实验结果表明，最优的上下文窗口大小对连贯性结果有显著影响，突显了架构灵活性在适应不同语言结构的重要性。跨语言性能评估进一步证实了该框架在多种语言中的适应性，将其实用性扩展到单语言环境之外。资源效率分析表明，与传统方法相比，该框架的计算开销有所减少，强调了其在大规模部署中的实用性。 

---
# Generalization and Informativeness of Weighted Conformal Risk Control Under Covariate Shift 

**Title (ZH)**: 加权可信区间风险控制在协变量偏移情况下的泛化能力和信息量 

**Authors**: Matteo Zecchin, Fredrik Hellström, Sangwoo Park, Shlomo Shamai, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2501.11413)  

**Abstract**: Predictive models are often required to produce reliable predictions under statistical conditions that are not matched to the training data. A common type of training-testing mismatch is covariate shift, where the conditional distribution of the target variable given the input features remains fixed, while the marginal distribution of the inputs changes. Weighted conformal risk control (W-CRC) uses data collected during the training phase to convert point predictions into prediction sets with valid risk guarantees at test time despite the presence of a covariate shift. However, while W-CRC provides statistical reliability, its efficiency -- measured by the size of the prediction sets -- can only be assessed at test time. In this work, we relate the generalization properties of the base predictor to the efficiency of W-CRC under covariate shifts. Specifically, we derive a bound on the inefficiency of the W-CRC predictor that depends on algorithmic hyperparameters and task-specific quantities available at training time. This bound offers insights on relationships between the informativeness of the prediction sets, the extent of the covariate shift, and the size of the calibration and training sets. Experiments on fingerprinting-based localization validate the theoretical results. 

**Abstract (ZH)**: 在统计条件与训练数据不匹配的情况下，预测模型往往需要产生可靠的预测。常见的训练-测试不匹配类型之一是条件变异（covariate shift），在这种情况下，目标变量给定输入特征的条件分布保持不变，而输入变量的边缘分布发生变化。加权一致性风险控制（W-CRC）利用训练阶段收集的数据，在存在条件变异的情况下，将点预测转换为具有有效风险保证的预测集。然而，尽管W-CRC提供了统计可靠性，其效率——通过预测集的大小来衡量——只能在测试阶段进行评估。在本工作中，我们研究基础预测器的泛化特性与条件变异下W-CRC效率之间的关系。具体而言，我们推导出W-CRC预测器效率不足的上界，该上界依赖于算法超参数和训练阶段可用的任务特定量。这一上界提供了预测集的信息性、条件变异的程度以及校准和训练集大小之间关系的见解。基于指纹定位的实验验证了理论结果。 

---
# Unsupervised Learning in Echo State Networks for Input Reconstruction 

**Title (ZH)**: 基于回声状态网络的无监督输入重构学习 

**Authors**: Taiki Yamada, Yuichi Katori, Kantaro Fujiwara  

**Link**: [PDF](https://arxiv.org/pdf/2501.11409)  

**Abstract**: Conventional echo state networks (ESNs) require supervised learning to train the readout layer, using the desired outputs as training data. In this study, we focus on input reconstruction (IR), which refers to training the readout layer to reproduce the input time series in its output. We reformulate the learning algorithm of the ESN readout layer to perform IR using unsupervised learning (UL). By conducting theoretical analysis and numerical experiments, we demonstrate that IR in ESNs can be effectively implemented under realistic conditions without explicitly using the desired outputs as training data; in this way, UL is enabled. Furthermore, we demonstrate that applications relying on IR, such as dynamical system replication and noise filtering, can be reformulated within the UL framework. Our findings establish a theoretically sound and universally applicable IR formulation, along with its related tasks in ESNs. This work paves the way for novel predictions and highlights unresolved theoretical challenges in ESNs, particularly in the context of time-series processing methods and computational models of the brain. 

**Abstract (ZH)**: 传统回声状态网络（ESN）中的读取层需要监督学习来训练，使用期望输出作为训练数据。本研究重点关注输入重建（IR），即训练读取层以在其输出中重现输入时间序列。我们重新制定了ESN读取层的学习算法，使其能够使用无监督学习（UL）进行IR。通过理论分析和数值实验，我们证明在现实条件下可以在不显式使用期望输出作为训练数据的情况下有效实现ESN中的IR；从而实现了UL的应用。此外，我们证明依赖于IR的应用，例如动力系统复制和噪声滤波，都可以在UL框架内重新表述。我们的研究结果建立了理论上合理且适用于所有相关任务的IR表述方法及其在ESN中的应用。本工作为新型预测开辟了道路，并强调了ESN在时间序列处理方法和大脑计算模型方面存在的未解理论挑战。 

---
# A Truly Sparse and General Implementation of Gradient-Based Synaptic Plasticity 

**Title (ZH)**: 基于梯度的突触可塑性的真正稀疏且通用的实现方法 

**Authors**: Jamie Lohoff, Anil Kaya, Florian Assmuth, Emre Neftci  

**Link**: [PDF](https://arxiv.org/pdf/2501.11407)  

**Abstract**: Online synaptic plasticity rules derived from gradient descent achieve high accuracy on a wide range of practical tasks. However, their software implementation often requires tediously hand-derived gradients or using gradient backpropagation which sacrifices the online capability of the rules. In this work, we present a custom automatic differentiation (AD) pipeline for sparse and online implementation of gradient-based synaptic plasticity rules that generalizes to arbitrary neuron models. Our work combines the programming ease of backpropagation-type methods for forward AD while being memory-efficient. To achieve this, we exploit the advantageous compute and memory scaling of online synaptic plasticity by providing an inherently sparse implementation of AD where expensive tensor contractions are replaced with simple element-wise multiplications if the tensors are diagonal. Gradient-based synaptic plasticity rules such as eligibility propagation (e-prop) have exactly this property and thus profit immensely from this feature. We demonstrate the alignment of our gradients with respect to gradient backpropagation on an synthetic task where e-prop gradients are exact, as well as audio speech classification benchmarks. We demonstrate how memory utilization scales with network size without dependence on the sequence length, as expected from forward AD methods. 

**Abstract (ZH)**: 基于梯度下降的在线突触可塑性规则能够在广泛的实际任务中实现高精度。然而，其软件实现往往需要手工推导梯度或使用梯度反向传播，这牺牲了规则的在线能力。在本工作中，我们提出了一种自定义自动微分（AD）管道，用于稀疏和在线实现基于梯度的突触可塑性规则，并且能够推广到任意神经元模型。我们的方法结合了反向AD类型方法的编程便利性，并且具有内存高效性。为了实现这一点，我们利用在线突触可塑性的计算和内存缩放优势，提供了一种固有的稀疏实现的自动微分，其中昂贵的张量收缩通过简单的元素级乘法替换，如果这些张量是斜对角的。基于梯度的突触可塑性规则，如持久性可传播规则（e-prop）正好具有这种特性，因此从这一特性中受益良多。我们在一个合成任务上展示了我们的梯度与梯度反向传播的一致性，该任务上的e-prop梯度是精确的，以及音频语音分类基准。我们展示了内存利用率随着网络规模的变化而增长，而不依赖于序列长度，这是向前自动微分方法所期望的。 

---
# Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio 

**Title (ZH)**: 非语音音频引起的耳语ASR幻觉调查 

**Authors**: Mateusz Barański, Jan Jasiński, Julitta Bartolewska, Stanisław Kacprzak, Marcin Witkowski, Konrad Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2501.11378)  

**Abstract**: Hallucinations of deep neural models are amongst key challenges in automatic speech recognition (ASR). In this paper, we investigate hallucinations of the Whisper ASR model induced by non-speech audio segments present during inference. By inducting hallucinations with various types of sounds, we show that there exists a set of hallucinations that appear frequently. We then study hallucinations caused by the augmentation of speech with such sounds. Finally, we describe the creation of a bag of hallucinations (BoH) that allows to remove the effect of hallucinations through the post-processing of text transcriptions. The results of our experiments show that such post-processing is capable of reducing word error rate (WER) and acts as a good safeguard against problematic hallucinations. 

**Abstract (ZH)**: 深度神经模型的幻觉是自动语音识别（ASR）中的关键挑战之一。本文探讨了在推理过程中由非语音音频片段引发的Whisper ASR模型的幻觉问题。通过采用各种类型的声波来诱发幻觉，我们发现存在一组频繁出现的幻觉。随后，我们研究了将这些声音与语音信息结合所引起的幻觉。最后，我们描述了一种幻觉袋（BoH）的创建方法，该方法可以通过处理文本转录来去除幻觉的影响。我们的实验结果表明，这种后处理方法能有效降低词错误率（WER），并能很好地防止有害幻觉的发生。 

---
# Federated Learning with Sample-level Client Drift Mitigation 

**Title (ZH)**: 带有样本水平客户端漂移缓解的联邦学习 

**Authors**: Haoran Xu, Jiaze Li, Wanyi Wu, Hao Ren  

**Link**: [PDF](https://arxiv.org/pdf/2501.11360)  

**Abstract**: Federated Learning (FL) suffers from severe performance degradation due to the data heterogeneity among clients. Existing works reveal that the fundamental reason is that data heterogeneity can cause client drift where the local model update deviates from the global one, and thus they usually tackle this problem from the perspective of calibrating the obtained local update. Despite effectiveness, existing methods substantially lack a deep understanding of how heterogeneous data samples contribute to the formation of client drift. In this paper, we bridge this gap by identifying that the drift can be viewed as a cumulative manifestation of biases present in all local samples and the bias between samples is different. Besides, the bias dynamically changes as the FL training progresses. Motivated by this, we propose FedBSS that first mitigates the heterogeneity issue in a sample-level manner, orthogonal to existing methods. Specifically, the core idea of our method is to adopt a bias-aware sample selection scheme that dynamically selects the samples from small biases to large epoch by epoch to train progressively the local model in each round. In order to ensure the stability of training, we set the diversified knowledge acquisition stage as the warm-up stage to avoid the local optimality caused by knowledge deviation in the early stage of the model. Evaluation results show that FedBSS outperforms state-of-the-art baselines. In addition, we also achieved effective results on feature distribution skew and noise label dataset setting, which proves that FedBSS can not only reduce heterogeneity, but also has scalability and robustness. 

**Abstract (ZH)**: 联邦学习（FL）由于客户端之间的数据异质性而遭受严重的性能下降。现有研究揭示的数据异质性导致的问题原因是，本地模型更新会偏离全局更新，从而通常从校准获得的本地更新的角度来解决这个问题。尽管这些方法有效，但它们在理解不同数据样本如何导致客户端漂移形成方面仍然缺乏深入的理解。本文通过识别数据漂移可以视为所有本地样本中存在偏见的累积表现，并且这些偏见之间存在差异来解决这一问题。此外，这些偏见随着联邦学习训练的进行而动态变化。基于此，我们提出了FedBSS，它首先以样本级别的方式缓解数据异质性问题，与现有方法正交。具体而言，我们方法的核心思想是采用一个感知偏见的样本选择方案，按照批次动态从小偏见选择到大偏见来训练每个回合中的本地模型。为了确保训练的稳定性，我们在知识偏差导致的局部最优性模型早期阶段设置多样化知识获取阶段，从而避免由此引起的局部最优点。评估结果表明，FedBSS在基准方法中表现出色。此外，我们还在特征分布偏斜和嘈杂标签数据集设置中取得了有效的结果，这证明了FedBSS不仅可以减少数据异质性，而且具有可扩展性和鲁棒性。 

---
# On the Dimension of Pullback Attractors in Recurrent Neural Networks 

**Title (ZH)**: 关于反馈神经网络中伴随吸引子维数的研究 

**Authors**: Muhammed Fadera  

**Link**: [PDF](https://arxiv.org/pdf/2501.11357)  

**Abstract**: Recurrent Neural Networks (RNNs) are high-dimensional state space models capable of learning functions on sequence data. Recently, it has been conjectured that reservoir computers, a particular class of RNNs, trained on observations of a dynamical systems can be interpreted as embeddings. This result has been established for the case of linear reservoir systems. In this work, we use a nonautonomous dynamical systems approach to establish an upper bound for the fractal dimension of the subset of reservoir state space approximated during training and prediction phase. We prove that when the input sequences comes from an Nin-dimensional invertible dynamical system, the fractal dimension of this set is bounded above by Nin. The result obtained here are useful in dimensionality reduction of computation in RNNs as well as estimating fractal dimensions of dynamical systems from limited observations of their time series. It is also a step towards understanding embedding properties of reservoir computers. 

**Abstract (ZH)**: 循环神经网络（RNNs）是高维状态空间模型，能够学习序列数据上的函数。最近有人推测，一类特定的RNN——蓄水库计算机，在对动力系统观测数据进行训练时，可以被解释为嵌入。这一结论已经在线性蓄水库系统的情况下得到了证实。在本文中，我们采用非自治动力系统的方法，为训练和预测阶段中近似蓄水库状态空间的子集的分形维数建立了上界。证明了当输入序列来自一个N维可逆动力系统时，该集合的分形维数被上界限定为N。本文获得的结果在RNN计算的降维及从有限时间序列观测估计动力系统分形维数方面具有实际应用价值。同时，这也是理解蓄水库计算机嵌入性质的一个步骤。 

---
# Towards Advancing Code Generation with Large Language Models: A Research Roadmap 

**Title (ZH)**: 朝向借助大规模语言模型推进代码生成的研究蓝图 

**Authors**: Haolin Jin, Huaming Chen, Qinghua Lu, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11354)  

**Abstract**: Recently, we have witnessed the rapid development of large language models, which have demonstrated excellent capabilities in the downstream task of code generation. However, despite their potential, LLM-based code generation still faces numerous technical and evaluation challenges, particularly when embedded in real-world development. In this paper, we present our vision for current research directions, and provide an in-depth analysis of existing studies on this task. We propose a six-layer vision framework that categorizes code generation process into distinct phases, namely Input Phase, Orchestration Phase, Development Phase, and Validation Phase. Additionally, we outline our vision workflow, which reflects on the currently prevalent frameworks. We systematically analyse the challenges faced by large language models, including those LLM-based agent frameworks, in code generation tasks. With these, we offer various perspectives and actionable recommendations in this area. Our aim is to provide guidelines for improving the reliability, robustness and usability of LLM-based code generation systems. Ultimately, this work seeks to address persistent challenges and to provide practical suggestions for a more pragmatic LLM-based solution for future code generation endeavors. 

**Abstract (ZH)**: 近年来，我们见证了大规模语言模型的迅速发展，这些模型在代码生成的下游任务中展示了卓越的能力。然而，尽管前景广阔，基于语言模型的代码生成仍然面临着诸多技术和评估挑战，尤其是在实际开发环境中。在本文中，我们提出了当前研究方向的愿景，并对这一任务的现有研究进行了深入分析。我们提出了一个六层框架，将代码生成过程划分为不同的阶段，分别是输入阶段、编排阶段、开发阶段和验证阶段。此外，我们概述了我们的愿景工作流程，反映了目前流行的框架。我们系统地分析了大规模语言模型在代码生成任务中面临的挑战，包括基于语言模型的代理框架。基于这些分析，我们提供了该领域的多种视角和可操作的建议。我们的目标是为提高基于语言模型的代码生成系统的可靠性和易用性提供指导。最终，本文旨在解决持久存在的挑战，并为未来的代码生成提供更实用的基于语言模型的解决方案。 

---
# Few-shot Policy (de)composition in Conversational Question Answering 

**Title (ZH)**: Few-shot 策略（反）分解在对话式问答中的应用 

**Authors**: Kyle Erwin, Guy Axelrod, Maria Chang, Achille Fokoue, Maxwell Crouse, Soham Dan, Tian Gao, Rosario Uceda-Sosa, Ndivhuwo Makondo, Naweed Khan, Alexander Gray  

**Link**: [PDF](https://arxiv.org/pdf/2501.11335)  

**Abstract**: The task of policy compliance detection (PCD) is to determine if a scenario is in compliance with respect to a set of written policies. In a conversational setting, the results of PCD can indicate if clarifying questions must be asked to determine compliance status. Existing approaches usually claim to have reasoning capabilities that are latent or require a large amount of annotated data. In this work, we propose logical decomposition for policy compliance (LDPC): a neuro-symbolic framework to detect policy compliance using large language models (LLMs) in a few-shot setting. By selecting only a few exemplars alongside recently developed prompting techniques, we demonstrate that our approach soundly reasons about policy compliance conversations by extracting sub-questions to be answered, assigning truth values from contextual information, and explicitly producing a set of logic statements from the given policies. The formulation of explicit logic graphs can in turn help answer PCDrelated questions with increased transparency and explainability. We apply this approach to the popular PCD and conversational machine reading benchmark, ShARC, and show competitive performance with no task-specific finetuning. We also leverage the inherently interpretable architecture of LDPC to understand where errors occur, revealing ambiguities in the ShARC dataset and highlighting the challenges involved with reasoning for conversational question answering. 

**Abstract (ZH)**: 政策合规检测（PCD）的任务是确定场景是否与一组书面政策保持一致。在对话环境中，PCD 的结果可以指示是否需要提出澄清问题以确定合规状态。现有方法通常声称具有潜在的推理能力，或者需要大量的标注数据。在这项工作中，我们提出了逻辑分解政策合规性（LDPC）：一种基于神经符号框架的方法，使用大规模语言模型（LLMs）在少样本设置中检测政策合规性。通过选择少量示例样本并结合最近开发的提示技术，我们展示了该方法可以准确地对政策合规性对话进行逻辑推理，通过提取需要回答的子问题、从上下文中分配真实值，并明确地生成一套逻辑语句来体现给定的政策。通过明确逻辑图的表述可以进一步增强对PCD相关问题的透明性和可解释性。我们将此方法应用于流行的PCD和对话机器阅读基准ShARC，并展示了在无需特定任务微调的情况下具有竞争力的表现。我们还利用LDPC固有的可解释架构来理解错误发生的位置，揭示了ShARC数据集中的模糊性，并强调了对话问答推理中面临的挑战。 

---
# CatV2TON: Taming Diffusion Transformers for Vision-Based Virtual Try-On with Temporal Concatenation 

**Title (ZH)**: CatV2TON: 通过时间拼接控制视觉引导虚拟试穿的扩散变压器 

**Authors**: Zheng Chong, Wenqing Zhang, Shiyue Zhang, Jun Zheng, Xiao Dong, Haoxiang Li, Yiling Wu, Dongmei Jiang, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11325)  

**Abstract**: Virtual try-on (VTON) technology has gained attention due to its potential to transform online retail by enabling realistic clothing visualization of images and videos. However, most existing methods struggle to achieve high-quality results across image and video try-on tasks, especially in long video scenarios. In this work, we introduce CatV2TON, a simple and effective vision-based virtual try-on (V2TON) method that supports both image and video try-on tasks with a single diffusion transformer model. By temporally concatenating garment and person inputs and training on a mix of image and video datasets, CatV2TON achieves robust try-on performance across static and dynamic settings. For efficient long-video generation, we propose an overlapping clip-based inference strategy that uses sequential frame guidance and Adaptive Clip Normalization (AdaCN) to maintain temporal consistency with reduced resource demands. We also present ViViD-S, a refined video try-on dataset, achieved by filtering back-facing frames and applying 3D mask smoothing for enhanced temporal consistency. Comprehensive experiments demonstrate that CatV2TON outperforms existing methods in both image and video try-on tasks, offering a versatile and reliable solution for realistic virtual try-ons across diverse scenarios. 

**Abstract (ZH)**: 虚拟试穿（VTON）技术因其在在线零售中实现图像和视频中的真实服装可视化而引起了广泛关注。然而，大多数现有方法难以在图像和视频试穿任务中获得高质量的结果，尤其是在长视频场景中。在本文中，我们引入了CatV2TON，这是一种简单且有效的方法，基于视觉实现图像和视频试穿任务，仅使用单一的扩散转换器模型。通过在图像和视频数据集的混合上训练，并在时间上将服装和人体输入连接起来，CatV2TON在静态和动态场景中均实现了稳健的试穿效果。为了高效生成长视频，我们提出了一种重叠片段的推理策略，该策略使用顺序帧指导和自适应片段归一化（AdaCN），以减少资源需求的同时保持时间一致性。此外，我们还介绍了ViViD-S，这是一个通过过滤背向帧并应用三维蒙版平滑来增强时间一致性的优化版视频试穿数据集。全面的实验表明，CatV2TON在图像和视频试穿任务中均优于现有方法，提供了一种适用于各种场景的多样化且可靠的虚拟试穿解决方案。 

---
# Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation 

**Title (ZH)**: Finer-CAM：突出差异以揭示更精细的视觉解释细节 

**Authors**: Ziheng Zhang, Jianyang Gu, Arpita Chowdhury, Zheda Mai, David Carlyn, Tanya Berger-Wolf, Yu Su, Wei-Lun Chao  

**Link**: [PDF](https://arxiv.org/pdf/2501.11309)  

**Abstract**: Class activation map (CAM) has been widely used to highlight image regions that contribute to class predictions. Despite its simplicity and computational efficiency, CAM often struggles to identify discriminative regions that distinguish visually similar fine-grained classes. Prior efforts address this limitation by introducing more sophisticated explanation processes, but at the cost of extra complexity. In this paper, we propose Finer-CAM, a method that retains CAM's efficiency while achieving precise localization of discriminative regions. Our key insight is that the deficiency of CAM lies not in "how" it explains, but in "what" it explains}. Specifically, previous methods attempt to identify all cues contributing to the target class's logit value, which inadvertently also activates regions predictive of visually similar classes. By explicitly comparing the target class with similar classes and spotting their differences, Finer-CAM suppresses features shared with other classes and emphasizes the unique, discriminative details of the target class. Finer-CAM is easy to implement, compatible with various CAM methods, and can be extended to multi-modal models for accurate localization of specific concepts. Additionally, Finer-CAM allows adjustable comparison strength, enabling users to selectively highlight coarse object contours or fine discriminative details. Quantitatively, we show that masking out the top 5% of activated pixels by Finer-CAM results in a larger relative confidence drop compared to baselines. The source code and demo are available at this https URL. 

**Abstract (ZH)**: 对象激活图（Class Activation Map, CAM）已被广泛应用于强调对分类预测有贡献的图像区域。尽管CAM在简洁性和计算效率方面表现出色，但在识别区分度高的、视觉上相似的细粒度类别的辨别区域方面经常遇到困难。先前的努力通过引入更复杂的解释过程来解决这一局限性，但这增加了额外的复杂性。在本文中，我们提出了一种名为Finer-CAM的方法，该方法保留了CAM的效率，同时能够实现对辨别区域的精确定位。我们的关键洞察是，CAM的不足之处不在于“如何”解释，而在于“解释了什么”。具体而言，以前的方法试图识别所有对目标类别的逻辑值有贡献的线索，无意中也激活了预测视觉上相似类别区域。通过明确地将目标类别与相似类别进行比较并找出它们之间的差异，Finer-CAM抑制了与其他类别共享的特征，并强调了目标类别的独特、区分性细节。Finer-CAM易于实现，可以与各种CAM方法兼容，并且可以扩展到多模态模型中，以准确定位特定概念。此外，Finer-CAM允许调整比较强度，从而使用户能够选择性地突出粗略的对象轮廓或细微的区分性细节。定量分析表明，通过Finer-CAM屏蔽掉激活像素的前5%，会比基准方法导致更大的置信度下降。相关源代码和演示可在以下链接获取：[请在此处提供链接]。 

---
# Collaborative Imputation of Urban Time Series through Cross-city Meta-learning 

**Title (ZH)**: 通过跨城市元学习的城市时间序列协作插补 

**Authors**: Tong Nie, Wei Ma, Jian Sun, Yu Yang, Jiannong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2501.11306)  

**Abstract**: Urban time series, such as mobility flows, energy consumption, and pollution records, encapsulate complex urban dynamics and structures. However, data collection in each city is impeded by technical challenges such as budget limitations and sensor failures, necessitating effective data imputation techniques that can enhance data quality and reliability. Existing imputation models, categorized into learning-based and analytics-based paradigms, grapple with the trade-off between capacity and generalizability. Collaborative learning to reconstruct data across multiple cities holds the promise of breaking this trade-off. Nevertheless, urban data's inherent irregularity and heterogeneity issues exacerbate challenges of knowledge sharing and collaboration across cities. To address these limitations, we propose a novel collaborative imputation paradigm leveraging meta-learned implicit neural representations (INRs). INRs offer a continuous mapping from domain coordinates to target values, integrating the strengths of both paradigms. By imposing embedding theory, we first employ continuous parameterization to handle irregularity and reconstruct the dynamical system. We then introduce a cross-city collaborative learning scheme through model-agnostic meta learning, incorporating hierarchical modulation and normalization techniques to accommodate multiscale representations and reduce variance in response to heterogeneity. Extensive experiments on a diverse urban dataset from 20 global cities demonstrate our model's superior imputation performance and generalizability, underscoring the effectiveness of collaborative imputation in resource-constrained settings. 

**Abstract (ZH)**: 都市时间序列，例如出行流量、能源消耗和污染记录，概括了复杂的都市动态和结构。然而，每个城市的 数据收集由于技术挑战如预算限制和传感器故障而受阻，这需要有效的数据插补技术来提高数据质量和可靠性。现有的插补模型，按照基于学习的方法和基于分析的方法进行分类，面临着容量与泛化能力之间的权衡问题。跨城市的协作学习有望打破这种权衡。然而，都市数据的固有不规则性和异质性问题加剧了城市之间的知识共享和协作挑战。为了解决这些问题，我们提出了一种新的基于元学习的隐式神经表示（INR）的协作插补范式。INR 提供了从领域坐标到目标值的连续映射，整合了两种方法的优点。通过应用嵌入理论，我们首先采用连续参数化来处理不规则性并重构动力系统。然后，我们通过模型不可知的元学习引入一种跨城市协作学习方案，采用层次调节和归一化技术来适应多尺度表示并减少异质性对响应方差的影响。来自全球20个不同城市的多维都市数据集的广泛实验表明，我们的模型在插补性能和泛化能力方面表现出优越性，强调了在资源受限环境中协作插补的有效性。 

---
# Question-to-Question Retrieval for Hallucination-Free Knowledge Access: An Approach for Wikipedia and Wikidata Question Answering 

**Title (ZH)**: 无幻觉知识访问的情境检索方法：面向维基百科和Wikidata的问答系统研究 

**Authors**: Santhosh Thottingal  

**Link**: [PDF](https://arxiv.org/pdf/2501.11301)  

**Abstract**: This paper introduces an approach to question answering over knowledge bases like Wikipedia and Wikidata by performing "question-to-question" matching and retrieval from a dense vector embedding store. Instead of embedding document content, we generate a comprehensive set of questions for each logical content unit using an instruction-tuned LLM. These questions are vector-embedded and stored, mapping to the corresponding content. Vector embedding of user queries are then matched against this question vector store. The highest similarity score leads to direct retrieval of the associated article content, eliminating the need for answer generation. Our method achieves high cosine similarity ( > 0.9 ) for relevant question pairs, enabling highly precise retrieval. This approach offers several advantages including computational efficiency, rapid response times, and increased scalability. We demonstrate its effectiveness on Wikipedia and Wikidata, including multimedia content through structured fact retrieval from Wikidata, opening up new pathways for multimodal question answering. 

**Abstract (ZH)**: 本文介绍了一种在维基百科和Wikidata等知识库上进行问答的方法，通过执行“问题到问题”的匹配与检索，从密集向量嵌入存储中获取相关信息。与嵌入文档内容不同，我们使用指令调优的LLM为每项逻辑内容单元生成一系列全面的问题。这些问题被向量化并存储，并映射到相应的内容。用户查询的向量化表示随后与这些问题的向量存储进行匹配。相似度最高的得分会直接检索相应的文章内容，从而消除答案生成的需要。我们的方法在相关问题对中实现了高度的余弦相似度（>0.9），从而实现高精度检索。该方法具有若干优点，包括计算效率、快速响应时间和增强的可扩展性。我们通过结构化事实检索从Wikidata中获取多媒体内容，展示了其在维基百科和Wikidata上的有效性，为多模态问答开辟了新的途径。 

---
# A Machine Learning Framework for Handling Unreliable Absence Label and Class Imbalance for Marine Stinger Beaching Prediction 

**Title (ZH)**: 一种处理海洋刺胞动物 Strandings 预测中不可靠缺席标签和类别不平衡的机器学习框架 

**Authors**: Amuche Ibenegbu, Amandine Schaeffer, Pierre Lafaye de Micheaux, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2501.11293)  

**Abstract**: Bluebottles (\textit{Physalia} spp.) are marine stingers resembling jellyfish, whose presence on Australian beaches poses a significant public risk due to their venomous nature. Understanding the environmental factors driving bluebottles ashore is crucial for mitigating their impact, and machine learning tools are to date relatively unexplored. We use bluebottle marine stinger presence/absence data from beaches in Eastern Sydney, Australia, and compare machine learning models (Multilayer Perceptron, Random Forest, and XGBoost) to identify factors influencing their presence. We address challenges such as class imbalance, class overlap, and unreliable absence data by employing data augmentation techniques, including the Synthetic Minority Oversampling Technique (SMOTE), Random Undersampling, and Synthetic Negative Approach that excludes the negative class. Our results show that SMOTE failed to resolve class overlap, but the presence-focused approach effectively handled imbalance, class overlap, and ambiguous absence data. The data attributes such as the wind direction, which is a circular variable, emerged as a key factor influencing bluebottle presence, confirming previous inference studies. However, in the absence of population dynamics, biological behaviours, and life cycles, the best predictive model appears to be Random Forests combined with Synthetic Negative Approach. This research contributes to mitigating the risks posed by bluebottles to beachgoers and provides insights into handling class overlap and unreliable negative class in environmental modelling. 

**Abstract (ZH)**: 蓝头水母（\textit{Physalia} spp.）是与水母相似的marine stingers，其出现在澳大利亚海滩上对公众造成了显著的风险，因为它们具有毒性。理解驱动蓝头水母上岸的环境因素对于减轻其影响至关重要，而目前机器学习工具的应用相对较少。我们利用澳大利亚东悉尼海滩的蓝头水母的存在/不存在数据，比较了机器学习模型（多层感知器、随机森林和XGBoost）来识别影响其存在的因素。我们通过采用数据增强技术，包括合成少数类过采样技术（SMOTE）、随机欠采样和合成负样本方法来解决类不平衡、类重叠和不可靠的不存在数据等问题的挑战。研究结果表明，SMOTE无法解决类重叠问题，而以存在为导向的方法有效地处理了不平衡、类重叠和模糊的不存在数据问题。风向作为关键因素之一，通过合成变量的属性显示出对蓝头水母存在的影响，这与之前的推理研究相吻合。然而，在缺乏种群动态、生物学行为和生命周期信息的情况下，最佳预测模型似乎是结合了合成负样本方法的随机森林。这项研究为减轻蓝头水母对海滩游客的威胁做出了贡献，并提供了在环境建模中处理类重叠和不可靠的负类问题的见解。 

---
# RedStar: Does Scaling Long-CoT Data Unlock Better Slow-Reasoning Systems? 

**Title (ZH)**: 红星辰：扩展长中间推理数据能否解锁更好的缓慢推理系统？ 

**Authors**: Haotian Xu, Xing Wu, Weinong Wang, Zhongzhi Li, Da Zheng, Boyuan Chen, Yi Hu, Shijia Kang, Jiaming Ji, Yingying Zhang, Zhijiang Guo, Yaodong Yang, Muhan Zhang, Debing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11284)  

**Abstract**: Can scaling transform reasoning? In this work, we explore the untapped potential of scaling Long Chain-of-Thought (Long-CoT) data to 1000k samples, pioneering the development of a slow-thinking model, RedStar. Through extensive experiments with various LLMs and different sizes, we uncover the ingredients for specialization and scale for Long-CoT training. Surprisingly, even smaller models show significant performance gains with limited data, revealing the sample efficiency of Long-CoT and the critical role of sample difficulty in the learning process. Our findings demonstrate that Long-CoT reasoning can be effectively triggered with just a few thousand examples, while larger models achieve unparalleled improvements. We also introduce reinforcement learning (RL)-scale training as a promising direction for advancing slow-thinking systems. RedStar shines across domains: on the MATH-Hard benchmark, RedStar-code-math boosts performance from 66.2\% to 81.6\%, and on the USA Math Olympiad (AIME), it solves 46.7\% of problems using only 21k mixed-code-math datasets. In multimodal tasks like GeoQA and MathVista-GEO, RedStar-Geo achieves competitive results with minimal Long-CoT data, outperforming other slow-thinking systems like QvQ-Preview. Compared to QwQ, RedStar strikes the perfect balance between reasoning and generalizability. Our work highlights that, with careful tuning, scaling Long-CoT can unlock extraordinary reasoning capabilities-even with limited dataset and set a new standard for slow-thinking models across diverse challenges. Our data and models are released at this https URL. 

**Abstract (ZH)**: 规模化能否提升推理能力？在本研究中，我们探索了将Long Chain-of-Thought (Long-CoT) 数据扩展至100万样本的潜在价值，并开发了红辰星（RedStar）这一慢思考模型。通过多种大型语言模型的不同规模实验，我们发现了Long-CoT训练的专业化和规模构成要素。令人惊讶的是，即使较小的模型在少量数据下也能显著提高性能，揭示了Long-CoT的样本效率以及样本难度在学习过程中的关键作用。研究结果表明，只需几千个示例，Long-CoT推理便能被有效触发；而更大的模型则能实现前所未有的改进。我们还提出了强化学习（RL）规模化训练，作为一种推进慢思考系统的有前景的方向。红辰星在多个领域都表现出色：在MATH-Hard基准测试中，红辰星代码数学（RedStar-code-math）将性能从66.2%提升到81.6%，在USA数学奥林匹克（AIME）中，仅使用21000个混合代码数学数据集便解决了46.7%的问题。在GeoQA和MathVista-GEO等多模态任务中，红辰星Geo（RedStar-Geo）即使在少量Long-CoT数据的情况下也能取得竞争力的结果，超越了诸如QvQ-Preview之类的其他慢思考系统。相比于QwQ，红辰星在推理和泛化能力之间达到了完美的平衡。我们的研究强调，通过精心调整，规模化Long-CoT可以解锁出人意料的推理能力，即使在有限的数据集下也能设定跨领域慢思考模型的新标准。我们的数据和模型已在此处 https://... 公开。 

---
# Spatiotemporal Air Quality Mapping in Urban Areas Using Sparse Sensor Data, Satellite Imagery, Meteorological Factors, and Spatial Features 

**Title (ZH)**: 使用稀疏传感器数据、卫星图像、气象因素和空间特征在城市区域进行时空空气质量Mapping 

**Authors**: Osama Ahmad, Zubair Khalid, Muhammad Tahir, Momin Uppal  

**Link**: [PDF](https://arxiv.org/pdf/2501.11270)  

**Abstract**: Monitoring air pollution is crucial for protecting human health from exposure to harmful substances. Traditional methods of air quality monitoring, such as ground-based sensors and satellite-based remote sensing, face limitations due to high deployment costs, sparse sensor coverage, and environmental interferences. To address these challenges, this paper proposes a framework for high-resolution spatiotemporal Air Quality Index (AQI) mapping using sparse sensor data, satellite imagery, and various spatiotemporal factors. By leveraging Graph Neural Networks (GNNs), we estimate AQI values at unmonitored locations based on both spatial and temporal dependencies. The framework incorporates a wide range of environmental features, including meteorological data, road networks, points of interest (PoIs), population density, and urban green spaces, which enhance prediction accuracy. We illustrate the use of our approach through a case study in Lahore, Pakistan, where multi-resolution data is used to generate the air quality index map at a fine spatiotemporal scale. 

**Abstract (ZH)**: 监测空气污染对于保护人类免受有害物质的暴露至关重要。传统的空气质量监测方法，如地基传感器和卫星遥感，由于部署成本高、传感器覆盖稀疏以及环境干扰等原因而受到限制。为此，本文提出了一种基于稀疏传感器数据、卫星影像以及多种空天地因素的高分辨率时空空气质量指数（AQI）映射框架，以应对这些挑战。利用图神经网络（GNNs），我们基于空间和时间依赖性，估计未监测位置的空气质量指数值。该框架整合了广泛的环境特征，包括气象数据、道路网络、兴趣点（PoIs）、人口密度和城市绿地，这些都提升了预测准确性。我们通过巴基斯坦拉合尔的案例研究，展示了该方法的应用，利用多分辨率数据生成精细时空尺度的空气质量指数图。 

---
# Code Readability in the Age of Large Language Models: An Industrial Case Study from Atlassian 

**Title (ZH)**: 大型语言模型时代下的代码可读性：来自Atlassian的工业案例研究 

**Authors**: Wannita Takerngsaksiri, Micheal Fu, Chakkrit Tantithamthavorn, Jirat Pasuksmit, Kun Chen, Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11264)  

**Abstract**: Programmers spend a significant amount of time reading code during the software development process. This trend is amplified by the emergence of large language models (LLMs) that automatically generate code. However, little is known about the readability of the LLM-generated code and whether it is still important from practitioners' perspectives in this new era. In this paper, we conduct a survey to explore the practitioners' perspectives on code readability in the age of LLMs and investigate the readability of our LLM-based software development agents framework, HULA, by comparing its generated code with human-written code in real-world scenarios. Overall, the findings underscore that (1) readability remains a critical aspect of software development; (2) the readability of our LLM-generated code is comparable to human-written code, fostering the establishment of appropriate trust and driving the broad adoption of our LLM-powered software development platform. 

**Abstract (ZH)**: 程序员在软件开发过程中花费了大量的时间阅读代码。随着大型语言模型（LLMs）的出现，这种趋势被进一步放大，这些模型能够自动生成代码。然而，对于LLM生成的代码可读性及其在这一新时代中对实践者的重要性，我们知之甚少。在本文中，我们进行了一项调查，以探索实践者在语言模型时代对代码可读性的看法，并通过将我们的基于LLM的软件开发代理框架HULA生成的代码与实际场景中的人手编写代码进行比较，来分析HULA的代码可读性。总体而言，研究结果表明：（1）软件开发中的可读性仍然是一个关键方面；（2）我们的LLM生成代码的可读性与人手编写代码相当，这有助于建立适当的信任，并推动我们的LLM赋能的软件开发平台的广泛采用。 

---
# WSSM: Geographic-enhanced hierarchical state-space model for global station weather forecast 

**Title (ZH)**: WSSM：基于地理增强的分层时空模型，用于全球站点天气预报 

**Authors**: Songru Yang, Zili Liu, Zhenwei Shi, Zhengxia Zou  

**Link**: [PDF](https://arxiv.org/pdf/2501.11238)  

**Abstract**: Global Station Weather Forecasting (GSWF), a prominent meteorological research area, is pivotal in providing timely localized weather predictions. Despite the progress existing models have made in the overall accuracy of the GSWF, executing high-precision extreme event prediction still presents a substantial challenge. The recent emergence of state-space models, with their ability to efficiently capture continuous-time dynamics and latent states, offer potential solutions. However, early investigations indicated that Mamba underperforms in the context of GSWF, suggesting further adaptation and optimization. To tackle this problem, in this paper, we introduce Weather State-space Model (WSSM), a novel Mamba-based approach tailored for GSWF. Geographical knowledge is integrated in addition to the widely-used positional encoding to represent the absolute special-temporal position. The multi-scale time-frequency features are synthesized from coarse to fine to model the seasonal to extreme weather dynamic. Our method effectively improves the overall prediction accuracy and addresses the challenge of forecasting extreme weather events. The state-of-the-art results obtained on the Weather-5K subset underscore the efficacy of the WSSM 

**Abstract (ZH)**: 全球站点天气预报（Global Station Weather Forecasting, GSWF）是气象研究的一个重要领域，对于提供及时的局部天气预测至关重要。尽管现有的模型在GSWF的整体准确度方面取得了进展，但在执行高精度极端事件预测方面仍然面临显著挑战。近年来，状态空间模型的出现，由于其能够有效地捕捉连续时间动态和潜在状态，为解决方案提供了可能。然而，早期的研究表明，Mamba在GSWF上下文中表现不佳，这表明需要进一步的适应和优化。为解决这一问题，本文引入了Weather状态空间模型（Weather State-space Model, WSSM），这是基于Mamba的一种新方法，专门针对GSWF。此外，我们还整合了地理知识以及广泛使用的位置编码来表示绝对的空间-时间位置。从粗到细合成多尺度时间频率特征，以建模季节性和极端天气动态。我们的方法有效提高了整体预测精度，并解决了预测极端天气事件的挑战。我们在Weather-5K子集上获得的最先进的结果证明了WSSM的有效性。 

---
# Leveraging GANs For Active Appearance Models Optimized Model Fitting 

**Title (ZH)**: 利用生成对抗网络优化活性外观模型的拟合方法 

**Authors**: Anurag Awasthi  

**Link**: [PDF](https://arxiv.org/pdf/2501.11218)  

**Abstract**: Generative Adversarial Networks (GANs) have gained prominence in refining model fitting tasks in computer vision, particularly in domains involving deformable models like Active Appearance Models (AAMs). This paper explores the integration of GANs to enhance the AAM fitting process, addressing challenges in optimizing nonlinear parameters associated with appearance and shape variations. By leveraging GANs' adversarial training framework, the aim is to minimize fitting errors and improve convergence rates. Achieving robust performance even in cases with high appearance variability and occlusions. Our approach demonstrates significant improvements in accuracy and computational efficiency compared to traditional optimization techniques, thus establishing GANs as a potent tool for advanced image model fitting. 

**Abstract (ZH)**: 生成对抗网络（GANs）在计算机视觉领域中已成为改进模型拟合任务的重要工具，特别是在涉及可变形模型（如主动外观模型AAMs）的领域。本文探讨了通过引入GANs来提升AAM拟合过程的方法，旨在解决与外观和形状变化相关的非线性参数优化难题。通过利用GANs的对抗训练框架，我们旨在减少拟合误差并提高收敛速度，从而即便在高外观差异和遮挡的情况下也能实现稳健的性能。与传统的优化技术相比，我们的方法在准确性与计算效率方面取得了显著进步，从而确立了GANs作为高级图像模型拟合的强大工具的地位。 

---
# Can Safety Fine-Tuning Be More Principled? Lessons Learned from Cybersecurity 

**Title (ZH)**: 当然，可以将以下论文内容或标题翻译成中文，同时保持学术规范：

"安全微调能更为原则化吗？从网络安全中汲取的教训" 

**Authors**: David Williams-King, Linh Le, Adam Oberman, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2501.11183)  

**Abstract**: As LLMs develop increasingly advanced capabilities, there is an increased need to minimize the harm that could be caused to society by certain model outputs; hence, most LLMs have safety guardrails added, for example via fine-tuning. In this paper, we argue the position that current safety fine-tuning is very similar to a traditional cat-and-mouse game (or arms race) between attackers and defenders in cybersecurity. Model jailbreaks and attacks are patched with bandaids to target the specific attack mechanism, but many similar attack vectors might remain. When defenders are not proactively coming up with principled mechanisms, it becomes very easy for attackers to sidestep any new defenses. We show how current defenses are insufficient to prevent new adversarial jailbreak attacks, reward hacking, and loss of control problems. In order to learn from past mistakes in cybersecurity, we draw analogies with historical examples and develop lessons learned that can be applied to LLM safety. These arguments support the need for new and more principled approaches to designing safe models, which are architected for security from the beginning. We describe several such approaches from the AI literature. 

**Abstract (ZH)**: 随着大型语言模型（LLM）的能力越来越先进，减少某些模型输出可能对社会造成的危害变得愈发重要；因此，大多数LLM都增加了安全防护措施，例如通过微调来进行细化调整。本文我们主张，当前的安全微调与网络安全领域中攻击者与防御者之间的经典猫鼠游戏（或军备竞赛）非常相似。针对特定攻击机制的模型逃逸和攻击可以通过临时补丁进行修补，但许多类似的攻击途径可能仍然存在。当防御者没有积极构建基于原理的安全机制时，攻击者很容易绕过新的防御措施。我们展示了现有的防御措施不足以防止新的敌对型模型逃逸攻击、奖励篡改以及控制丧失等问题。为了从网络安全领域的历史错误中吸取经验教训，我们借鉴了历史案例，并提出了适用于LLM安全性的教训。这些论点支持需要采用新的、更基于原则的方法来设计安全模型，并从一开始就架构其安全性。我们描述了从人工智能文献中介绍的几种此类方法。 

---
# ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models 

**Title (ZH)**: ProKeR：大规模视觉语言模型少量样本适应的核视角方法 

**Authors**: Yassir Bendou, Amine Ouasfi, Vincent Gripon, Adnane Boukhayma  

**Link**: [PDF](https://arxiv.org/pdf/2501.11175)  

**Abstract**: The growing popularity of Contrastive Language-Image Pretraining (CLIP) has led to its widespread application in various visual downstream tasks. To enhance CLIP's effectiveness and versatility, efficient few-shot adaptation techniques have been widely adopted. Among these approaches, training-free methods, particularly caching methods exemplified by Tip-Adapter, have gained attention for their lightweight adaptation without the need for additional fine-tuning. In this paper, we revisit Tip-Adapter from a kernel perspective, showing that caching methods function as local adapters and are connected to a well-established kernel literature. Drawing on this insight, we offer a theoretical understanding of how these methods operate and suggest multiple avenues for enhancing the Tip-Adapter baseline. Notably, our analysis shows the importance of incorporating global information in local adapters. Therefore, we subsequently propose a global method that learns a proximal regularizer in a reproducing kernel Hilbert space (RKHS) using CLIP as a base learner. Our method, which we call ProKeR (Proximal Kernel ridge Regression), has a closed form solution and achieves state-of-the-art performances across 11 datasets in the standard few-shot adaptation benchmark. 

**Abstract (ZH)**: 对比语言-图像预训练（CLIP）的日益流行已经促使它在各种视觉下游任务中的广泛应用。为了增强CLIP的有效性和灵活性，高效的少样本适应技术被广泛采用。在这些方法中，无需训练的方法，特别是以Tip-Adapter为代表的缓存方法，因其无需额外微调的轻量级适应而引起了关注。在本文中，我们从核的角度重新审视了Tip-Adapter，展示了缓存方法充当局部适配器，并与成熟的核理论文献相关联。基于这一洞察，我们提供了一种理论理解，解释了这些方法的工作方式，并提出了多种增强Tip-Adapter基线的方法。值得注意的是，我们的分析表明，在局部适配器中融合全局信息的重要性。因此，我们随后提出了一种全局方法，该方法利用CLIP作为基学习器，在再生核希尔伯特空间（RKHS）中学习一个邻近正则化器。我们称之为ProKeR（邻近核岭回归）的方法具有闭式解，并且在标准的少样本适应基准中的11个数据集上实现了最先进的性能。 

---
# Counteracting temporal attacks in Video Copy Detection 

**Title (ZH)**: 视频复制检测中的反临时攻击方法 

**Authors**: Katarzyna Fojcik, Piotr Syga  

**Link**: [PDF](https://arxiv.org/pdf/2501.11171)  

**Abstract**: Video Copy Detection (VCD) plays a crucial role in copyright protection and content verification by identifying duplicates and near-duplicates in large-scale video databases. The META AI Challenge on video copy detection provided a benchmark for evaluating state-of-the-art methods, with the Dual-level detection approach emerging as a winning solution. This method integrates Video Editing Detection and Frame Scene Detection to handle adversarial transformations and large datasets efficiently. However, our analysis reveals significant limitations in the VED component, particularly in its ability to handle exact copies. Moreover, Dual-level detection shows vulnerability to temporal attacks. To address it, we propose an improved frame selection strategy based on local maxima of interframe differences, which enhances robustness against adversarial temporal modifications while significantly reducing computational overhead. Our method achieves an increase of 1.4 to 5.8 times in efficiency over the standard 1 FPS approach. Compared to Dual-level detection method, our approach maintains comparable micro-average precision ($\mu$AP) while also demonstrating improved robustness against temporal attacks. Given 56\% reduced representation size and the inference time of more than 2 times faster, our approach is more suitable to real-world resource restriction. 

**Abstract (ZH)**: 视频复制检测（VCD）在版权保护和内容验证中扮演着关键角色，通过识别大规模视频数据库中的重复和近似重复片段。META AI挑战赛中的视频复制检测提供了评估最新方法的标准基准，双层检测方法脱颖而出，成为获胜解决方案。该方法整合了视频编辑检测和帧场景检测，以高效地处理对抗性转换和大型数据集。然而，我们的分析揭示了在视频编辑检测（VED）组件中的显著局限性，特别是在处理精确副本方面的能力有限。此外，双层检测方法在时间攻击面前也显得脆弱。为了解决这些问题，我们提出了一种基于帧间差异局部最大值的改进帧选择策略，该策略增强了对抗时间性修改的鲁棒性，同时大大减少了计算成本。我们的方法在标准每秒1帧（1 FPS）方法上提高了1.4到5.8倍的效率。与双层检测方法相比，我们的方法在保持相近的微平均精确度（$\mu$AP）的同时，还展示了更好的时间攻击鲁棒性。此外，通过减少56%的表示大小和推理时间超过2倍的加速，我们的方法更适合现实世界的资源限制。 

---
# AIMA at SemEval-2024 Task 3: Simple Yet Powerful Emotion Cause Pair Analysis 

**Title (ZH)**: AIMA在SemEval-2024任务3中的简单而强大的情感原因配对分析 

**Authors**: Alireza Ghahramani Kure, Mahshid Dehghani, Mohammad Mahdi Abootorabi, Nona Ghazizadeh, Seyed Arshan Dalili, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2501.11170)  

**Abstract**: The SemEval-2024 Task 3 presents two subtasks focusing on emotion-cause pair extraction within conversational contexts. Subtask 1 revolves around the extraction of textual emotion-cause pairs, where causes are defined and annotated as textual spans within the conversation. Conversely, Subtask 2 extends the analysis to encompass multimodal cues, including language, audio, and vision, acknowledging instances where causes may not be exclusively represented in the textual data. Our proposed model for emotion-cause analysis is meticulously structured into three core segments: (i) embedding extraction, (ii) cause-pair extraction & emotion classification, and (iii) cause extraction using QA after finding pairs. Leveraging state-of-the-art techniques and fine-tuning on task-specific datasets, our model effectively unravels the intricate web of conversational dynamics and extracts subtle cues signifying causality in emotional expressions. Our team, AIMA, demonstrated strong performance in the SemEval-2024 Task 3 competition. We ranked as the 10th in subtask 1 and the 6th in subtask 2 out of 23 teams. 

**Abstract (ZH)**: SemEval-2024 任务3围绕对话情境中情感-因果对的提取设定了两个子任务。子任务1专注于提取文本形式的情感-因果对，其中因果关系被定义和标注为对话中的文本片段。相反，子任务2则扩展分析范围，包括语言、音频和视觉等多种模态的线索，承认在某些情况下，因果关系可能不完全体现在文本数据中。我们提出的用于情感-因果分析的模型被仔细地划分为三个核心模块：（i）嵌入提取，（ii）因果对提取与情感分类，以及（iii）在找到因果对后使用问答（QA）进行因果提取。利用最先进的技术和针对特定任务的数据集进行微调，我们的模型有效地揭示了对话动态的复杂网络，并提取了表示情感表达因果关系的细微线索。我们团队AIMA在SemEval-2024任务3的比赛中表现出色。我们在这两个子任务中分别排名第10位和第6位，参赛团队共有23支。 

---
# AIMA at SemEval-2024 Task 10: History-Based Emotion Recognition in Hindi-English Code-Mixed Conversations 

**Title (ZH)**: AIMA在SemEval-2024任务10中的研究：基于历史的情感识别在印英混用对话中的应用 

**Authors**: Mohammad Mahdi Abootorabi, Nona Ghazizadeh, Seyed Arshan Dalili, Alireza Ghahramani Kure, Mahshid Dehghani, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2501.11166)  

**Abstract**: In this study, we introduce a solution to the SemEval 2024 Task 10 on subtask 1, dedicated to Emotion Recognition in Conversation (ERC) in code-mixed Hindi-English conversations. ERC in code-mixed conversations presents unique challenges, as existing models are typically trained on monolingual datasets and may not perform well on code-mixed data. To address this, we propose a series of models that incorporate both the previous and future context of the current utterance, as well as the sequential information of the conversation. To facilitate the processing of code-mixed data, we developed a Hinglish-to-English translation pipeline to translate the code-mixed conversations into English. We designed four different base models, each utilizing powerful pre-trained encoders to extract features from the input but with varying architectures. By ensembling all of these models, we developed a final model that outperforms all other baselines. 

**Abstract (ZH)**: 在本研究中，我们针对SemEval 2024 Task 10中的子任务1，即代码混合印地语-英语对话中的情感识别（ERC），引入了一种解决方案。代码混合对话中的ERC面临独特的挑战，因为现有的模型通常是在单语数据集上进行训练，可能在处理代码混合数据时表现不佳。为解决这一问题，我们提出了一系列模型，这些模型不仅考虑了当前发言的上下文，还考虑了当前发言的前后文以及对话的序列表现信息。为了方便处理代码混合数据，我们开发了一条Hinglish-to-English翻译流水线，将代码混合对话翻译成英语。我们设计了四个不同的基模型，每个模型都使用强大的预训练编码器从输入中提取特征，但这些编码器具有不同的架构。通过将所有这些模型进行集成，我们构建了一个最终模型，该模型优于所有其他基线模型。 

---
# CLOFAI: A Dataset of Real And Fake Image Classification Tasks for Continual Learning 

**Title (ZH)**: CLOFAI：持续学习中真实图像与伪造图像分类的数据集 

**Authors**: William Doherty, Anton Lee, Heitor Murilo Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2501.11140)  

**Abstract**: The rapid advancement of generative AI models capable of creating realistic media has led to a need for classifiers that can accurately distinguish between genuine and artificially-generated images. A significant challenge for these classifiers emerges when they encounter images from generative models that are not represented in their training data, usually resulting in diminished performance. A typical approach is to periodically update the classifier's training data with images from the new generative models then retrain the classifier on the updated dataset. However, in some real-life scenarios, storage, computational, or privacy constraints render this approach impractical. Additionally, models used in security applications may be required to rapidly adapt. In these circumstances, continual learning provides a promising alternative, as the classifier can be updated without retraining on the entire dataset. In this paper, we introduce a new dataset called CLOFAI (Continual Learning On Fake and Authentic Images), which takes the form of a domain-incremental image classification problem. Moreover, we showcase the applicability of this dataset as a benchmark for evaluating continual learning methodologies. In doing this, we set a baseline on our novel dataset using three foundational continual learning methods -- EWC, GEM, and Experience Replay -- and find that EWC performs poorly, while GEM and Experience Replay show promise, performing significantly better than a Naive baseline. The dataset and code to run the experiments can be accessed from the following GitHub repository: this https URL. 

**Abstract (ZH)**: 生成式AI模型的快速发展使得能够创建逼真媒体的内容合成器应运而生，这引发了对能够准确区分真实和合成图像的分类器的需求。当这些分类器遇到在其训练数据中未被代表的生成模型图像时，会面临一个重要挑战，通常会导致性能下降。一种常见的方法是定期用新生成模型的图像更新分类器的训练数据，然后在更新的数据集上重新训练分类器。然而，在某些实际情况中，存储、计算或隐私限制使得这种方法不可行。此外，在安全应用中使用的模型可能需要快速适应。在这种情况下，持续学习提供了一个有前途的替代方案，因为分类器可以在不重新训练整个数据集的情况下进行更新。本文介绍了一个新的数据集，称为CLOFAI（持续学习于假和真实图像），它以领域增量图像分类问题的形式呈现。此外，我们展示了该数据集作为持续学习方法评估基准的应用潜力。在这一过程中，我们使用三种基础的持续学习方法——EWC、GEM和经验回放，在我们的新数据集上建立了一个基准，并发现EWC表现不佳，而GEM和经验回放显示出良好的前景，显著优于朴素基准。数据集和运行实验的代码可以从以下GitHub仓库访问：this https URL。 

---
# Playing the Lottery With Concave Regularizers for Sparse Trainable Neural Networks 

**Title (ZH)**: 使用凹正则化器进行Sparse可训练神经网络的 Lottery Ticket 假设探究 

**Authors**: Giulia Fracastoro, Sophie M. Fosson, Andrea Migliorati, Giuseppe C. Calafiore  

**Link**: [PDF](https://arxiv.org/pdf/2501.11135)  

**Abstract**: The design of sparse neural networks, i.e., of networks with a reduced number of parameters, has been attracting increasing research attention in the last few years. The use of sparse models may significantly reduce the computational and storage footprint in the inference phase. In this context, the lottery ticket hypothesis (LTH) constitutes a breakthrough result, that addresses not only the performance of the inference phase, but also of the training phase. It states that it is possible to extract effective sparse subnetworks, called winning tickets, that can be trained in isolation. The development of effective methods to play the lottery, i.e., to find winning tickets, is still an open problem. In this article, we propose a novel class of methods to play the lottery. The key point is the use of concave regularization to promote the sparsity of a relaxed binary mask, which represents the network topology. We theoretically analyze the effectiveness of the proposed method in the convex framework. Then, we propose extended numerical tests on various datasets and architectures, that show that the proposed method can improve the performance of state-of-the-art algorithms. 

**Abstract (ZH)**: 在过去的几年中，稀疏神经网络的设计，即减少网络参数数量的设计，引起了越来越多的研究关注。使用稀疏模型可以在推理阶段显著减少计算和存储需求。在这一背景下，获奖票假设（LTH）是一个突破性成果，不仅解决了推理阶段的性能问题，也解决了训练阶段的问题。它表明可以从网络中提取有效的稀疏子网络，称为获奖票，并且可以在不依赖其他网络的情况下独立训练这些子网络。如何有效地“抽彩票”，即找到获奖票，仍然是一个开放的问题。在本文中，我们提出了一类新型方法来实现“抽彩票”的目标。关键点是使用凹正则化来促进松弛二进制掩码的稀疏性，该掩码代表网络拓扑结构。我们从凸优化框架的角度分析了所提方法的有效性。然后，我们在多个数据集和架构上提出了扩展的数值试验，结果显示所提方法可以提高最先进的算法的性能。 

---
# A Collection of Question Answering Datasets for Norwegian 

**Title (ZH)**: 挪威语问答数据集合集 

**Authors**: Vladislav Mikhailov, Petter Mæhlum, Victoria Ovedie Chruickshank Langø, Erik Velldal, Lilja Øvrelid  

**Link**: [PDF](https://arxiv.org/pdf/2501.11128)  

**Abstract**: This paper introduces a new suite of question answering datasets for Norwegian; NorOpenBookQA, NorCommonSenseQA, NorTruthfulQA, and NRK-Quiz-QA. The data covers a wide range of skills and knowledge domains, including world knowledge, commonsense reasoning, truthfulness, and knowledge about Norway. Covering both of the written standards of Norwegian - Bokmål and Nynorsk - our datasets comprise over 10k question-answer pairs, created by native speakers. We detail our dataset creation approach and present the results of evaluating 11 language models (LMs) in zero- and few-shot regimes. Most LMs perform better in Bokmål than Nynorsk, struggle most with commonsense reasoning, and are often untruthful in generating answers to questions. All our datasets and annotation materials are publicly available. 

**Abstract (ZH)**: 本文介绍了新的挪威语问答数据集套件；NorOpenBookQA、NorCommonSenseQA、NorTruthfulQA 和 NRK-Quiz-QA。这些数据覆盖了广泛的知识和技能领域，包括世界知识、常识推理、真实性以及关于挪威的知识。数据集涵盖了挪威语的两种书面标准——培根-&-瑞典语（Bokmål 和 Nynorsk），包含了超过 10,000 个问答对，均由母语为挪威语的人员创建。文中详细介绍了数据集的创建方法，并展示了在零样本和少样本条件下评估 11 种语言模型（LM）的结果。大多数语言模型在 Bokmål 中的表现优于 Nynorsk，尤其在常识推理方面遇到最大困难，并且经常生成不真实的问题答案。所有数据集及其注释材料均已公开发布。 

---
# Tell me about yourself: LLMs are aware of their learned behaviors 

**Title (ZH)**: 自我介绍：语言模型意识到它们学习到的行为 

**Authors**: Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2501.11120)  

**Abstract**: We study behavioral self-awareness -- an LLM's ability to articulate its behaviors without requiring in-context examples. We finetune LLMs on datasets that exhibit particular behaviors, such as (a) making high-risk economic decisions, and (b) outputting insecure code. Despite the datasets containing no explicit descriptions of the associated behavior, the finetuned LLMs can explicitly describe it. For example, a model trained to output insecure code says, ``The code I write is insecure.'' Indeed, models show behavioral self-awareness for a range of behaviors and for diverse evaluations. Note that while we finetune models to exhibit behaviors like writing insecure code, we do not finetune them to articulate their own behaviors -- models do this without any special training or examples.
Behavioral self-awareness is relevant for AI safety, as models could use it to proactively disclose problematic behaviors. In particular, we study backdoor policies, where models exhibit unexpected behaviors only under certain trigger conditions. We find that models can sometimes identify whether or not they have a backdoor, even without its trigger being present. However, models are not able to directly output their trigger by default.
Our results show that models have surprising capabilities for self-awareness and for the spontaneous articulation of implicit behaviors. Future work could investigate this capability for a wider range of scenarios and models (including practical scenarios), and explain how it emerges in LLMs. 

**Abstract (ZH)**: 我们研究了行为自我意识——即语言模型（LLM）在无需上下文示例的情况下表述自身行为的能力。我们通过对展示特定行为的数据集进行微调，例如（a）进行高风险经济决策，以及（b）输出不安全代码，来探索这一能力。尽管数据集中没有明确描述相关行为，但微调后的模型却能够明确描述这些行为。例如，一个被训练输出不安全代码的模型会说，“我写的代码是不安全的”。确实，模型展示出了在多种行为和不同评估中的行为自我意识。值得注意的是，虽然我们通过特定数据集微调模型以表现出类似编写不安全代码的行为，但模型并不会因为这种微调而专门被训练来表述自己的行为——他们是在没有任何特别训练或示例的情况下自行做到这一点的。

行为自我意识对于AI安全性具有重要意义，因为模型可以通过这种方式主动披露潜在问题行为。特别地，我们研究了后门策略，即模型仅在特定触发条件下才表现出异常行为。我们发现，即使在没有触发条件的情况下，模型有时也能识别自己是否具有后门行为。然而，模型默认情况下不能直接输出其触发条件。

我们的研究结果表明，模型具有令人惊讶的自我意识能力和对其隐含行为的自发表达能力。未来的工作可以探索这一能力在更广泛场景和模型（包括实际场景）中的应用，并解释其在语言模型中的产生机制。 

---
# Clinical trial cohort selection using Large Language Models on n2c2 Challenges 

**Title (ZH)**: 使用大规模语言模型在n2c2挑战赛中进行临床试验队列的选择 

**Authors**: Chi-en Amy Tai, Xavier Tannier  

**Link**: [PDF](https://arxiv.org/pdf/2501.11114)  

**Abstract**: Clinical trials are a critical process in the medical field for introducing new treatments and innovations. However, cohort selection for clinical trials is a time-consuming process that often requires manual review of patient text records for specific keywords. Though there have been studies on standardizing the information across the various platforms, Natural Language Processing (NLP) tools remain crucial for spotting eligibility criteria in textual reports. Recently, pre-trained large language models (LLMs) have gained popularity for various NLP tasks due to their ability to acquire a nuanced understanding of text. In this paper, we study the performance of large language models on clinical trial cohort selection and leverage the n2c2 challenges to benchmark their performance. Our results are promising with regard to the incorporation of LLMs for simple cohort selection tasks, but also highlight the difficulties encountered by these models as soon as fine-grained knowledge and reasoning are required. 

**Abstract (ZH)**: 临床试验是医学领域引入新治疗方法和创新的关键过程。然而，临床试验中的患者群选择是一个耗时的过程，通常需要手动审查患者的文字记录以寻找特定关键词。尽管已有研究致力于在各种平台上标准化信息，但自然语言处理（NLP）工具仍对于在文本报告中识别合格标准至关重要。最近，由于其能够获得对文本的深刻理解，预训练大型语言模型（LLMs）在各种NLP任务中越来越受欢迎。在本文中，我们研究了大型语言模型在临床试验患者群选择中的表现，并利用n2c2挑战赛对其性能进行了基准测试。我们的结果表明，对于简单的患者群选择任务，结合使用LLMs是前景看好的，但同时也突显了当需要专门的知识和推理时，这些模型所面临的困难。 

---
# ChaosEater: Fully Automating Chaos Engineering with Large Language Models 

**Title (ZH)**: 混沌吞噬者：利用大规模语言模型完全自动化混沌工程 

**Authors**: Daisuke Kikuta, Hiroki Ikeuchi, Kengo Tajiri, Yuusuke Nakano  

**Link**: [PDF](https://arxiv.org/pdf/2501.11107)  

**Abstract**: Chaos Engineering (CE) is an engineering technique aimed at improving the resiliency of distributed systems. It involves artificially injecting specific failures into a distributed system and observing its behavior in response. Based on the observation, the system can be proactively improved to handle those failures. Recent CE tools realize the automated execution of predefined CE experiments. However, defining these experiments and reconfiguring the system after the experiments still remain manual. To reduce the costs of the manual operations, we propose \textsc{ChaosEater}, a \textit{system} for automating the entire CE operations with Large Language Models (LLMs). It pre-defines the general flow according to the systematic CE cycle and assigns subdivided operations within the flow to LLMs. We assume systems based on Infrastructure as Code (IaC), wherein the system configurations and artificial failures are managed through code. Hence, the LLMs' operations in our \textit{system} correspond to software engineering tasks, including requirement definition, code generation and debugging, and testing. We validate our \textit{system} through case studies on both small and large systems. The results demonstrate that our \textit{system} significantly reduces both time and monetary costs while completing reasonable single CE cycles. 

**Abstract (ZH)**: 混沌工程（Chaos Engineering，CE）是一种旨在提高分布式系统弹性的工程技术。它通过在分布式系统中人为注入特定故障并观察其响应行为，从而能够基于这些观察对系统进行主动改进，使其能够处理这些故障。近年来，CE工具实现了预定义CE实验的自动化执行。然而，定义这些实验和实验后重新配置系统仍然需要手动操作。为减少这些手动操作的成本，我们提出了“ChaosEater”系统，该系统利用大规模语言模型（LLMs）自动化整个CE操作。它根据系统的CE周期定义了一般流程，并将流程中的细分操作分配给LLMs。我们假设基于基础设施即代码（IaC）的系统，其中系统配置和人工故障通过代码进行管理。因此，我们系统中的LLM操作对应于软件工程任务，包括需求定义、代码生成、调试和测试。我们通过针对小系统和大系统的案例研究验证了该系统。结果表明，该系统可以显著减少时间和货币成本，同时完成合理的单个CE周期。 

---
# Enhanced Suicidal Ideation Detection from Social Media Using a CNN-BiLSTM Hybrid Model 

**Title (ZH)**: 使用CNN-BiLSTM混合模型增强社交媒体中的自杀念头检测 

**Authors**: Mohaiminul Islam Bhuiyan, Nur Shazwani Kamarudin, Nur Hafieza Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2501.11094)  

**Abstract**: Suicidal ideation detection is crucial for preventing suicides, a leading cause of death worldwide. Many individuals express suicidal thoughts on social media, offering a vital opportunity for early detection through advanced machine learning techniques. The identification of suicidal ideation in social media text is improved by utilising a hybrid framework that integrates Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM), enhanced with an attention mechanism. To enhance the interpretability of the model's predictions, Explainable AI (XAI) methods are applied, with a particular focus on SHapley Additive exPlanations (SHAP), are incorporated. At first, the model managed to reach an accuracy of 92.81%. By applying fine-tuning and early stopping techniques, the accuracy improved to 94.29%. The SHAP analysis revealed key features influencing the model's predictions, such as terms related to mental health struggles. This level of transparency boosts the model's credibility while helping mental health professionals understand and trust the predictions. This work highlights the potential for improving the accuracy and interpretability of detecting suicidal tendencies, making a valuable contribution to the progress of mental health monitoring systems. It emphasizes the significance of blending powerful machine learning methods with explainability to develop reliable and impactful mental health solutions. 

**Abstract (ZH)**: 自杀观念的检测对于预防自杀至关重要，自杀是全球主要的死因之一。许多人在社交媒体上表达自杀想法，为通过高级机器学习技术实现早期检测提供了宝贵的机会。通过将卷积神经网络（CNN）与双向长短期记忆网络（BiLSTM）相结合，并引入注意力机制，可以改善在社交媒体文本中识别自杀观念的效果。为了增强模型预测的可解释性，应用了解释性人工智能（XAI）方法，并特别引入了SHapley Additive exPlanations（SHAP）方法。初步结果显示，模型的准确率达到了92.81%。通过应用微调和提前停止技术，准确率提高到了94.29%。SHAP分析揭示了影响模型预测的关键特征，例如与心理健康问题相关的术语。这种级别的透明度提升了模型的可信度，同时帮助心理健康专业人员理解并信任预测结果。本文突出了提高自杀倾向检测精度和可解释性的潜力，为心理健康监测系统的进步做出了宝贵贡献。它强调了将强大的机器学习方法与可解释性相结合的重要性，以开发可靠且有效的心理健康解决方案。 

---
# Leveraging counterfactual concepts for debugging and improving CNN model performance 

**Title (ZH)**: 利用反事实概念调试和提升CNN模型性能 

**Authors**: Syed Ali Tariq, Tehseen Zia  

**Link**: [PDF](https://arxiv.org/pdf/2501.11087)  

**Abstract**: Counterfactual explanation methods have recently received significant attention for explaining CNN-based image classifiers due to their ability to provide easily understandable explanations that align more closely with human reasoning. However, limited attention has been given to utilizing explainability methods to improve model performance. In this paper, we propose to leverage counterfactual concepts aiming to enhance the performance of CNN models in image classification tasks. Our proposed approach utilizes counterfactual reasoning to identify crucial filters used in the decision-making process. Following this, we perform model retraining through the design of a novel methodology and loss functions that encourage the activation of class-relevant important filters and discourage the activation of irrelevant filters for each class. This process effectively minimizes the deviation of activation patterns of local predictions and the global activation patterns of their respective inferred classes. By incorporating counterfactual explanations, we validate unseen model predictions and identify misclassifications. The proposed methodology provides insights into potential weaknesses and biases in the model's learning process, enabling targeted improvements and enhanced performance. Experimental results on publicly available datasets have demonstrated an improvement of 1-2\%, validating the effectiveness of the approach. 

**Abstract (ZH)**: 近年来，因果解释方法因能够提供与人类推理更接近、易于理解的解释而被广泛应用于解释基于CNN的图像分类器，引起了显著的关注。然而，很少有研究关注利用可解释性方法来提升模型性能。本文旨在通过利用因果概念来增强CNN模型在图像分类任务中的性能。本文提出的方法利用因果推理来识别决策过程中至关重要的滤波器。随后，通过设计一种新的方法学和损失函数，鼓励激活与每个类别相关的关键滤波器，同时抑制与每个类别无关的滤波器的激活。这一过程有效地最小化了局部预测的激活模式与各自推断类别全局激活模式之间的偏差。通过结合因果解释，我们验证了未知模型预测并识别了分类错误。所提出的方法学提供了有关模型学习过程潜在弱点和偏差的洞察，从而实现针对性的改进和增强性能。公开数据集上的实验结果表明，这种方法可以提高1-2%，验证了其有效性。 

---
# Can LLM Generate Regression Tests for Software Commits? 

**Title (ZH)**: 大规模语言模型能否生成软件提交的回归测试用例？ 

**Authors**: Jing Liu, Seongmin Lee, Eleonora Losiouk, Marcel Böhme  

**Link**: [PDF](https://arxiv.org/pdf/2501.11086)  

**Abstract**: Large Language Models (LLMs) have shown tremendous promise in automated software engineering. In this paper, we investigate the opportunities of LLMs for automatic regression test generation for programs that take highly structured, human-readable inputs, such as XML parsers or JavaScript interpreters. Concretely, we explore the following regression test generation scenarios for such programs that have so far been difficult to test automatically in the absence of corresponding input grammars:
$\bullet$ Bug finding. Given a code change (e.g., a commit or pull request), our LLM-based approach generates a test case with the objective of revealing any bugs that might be introduced if that change is applied.
$\bullet$ Patch testing. Given a patch, our LLM-based approach generates a test case that fails before but passes after the patch. This test can be added to the regression test suite to catch similar bugs in the future.
We implement Cleverest, a feedback-directed, zero-shot LLM-based regression test generation technique, and evaluate its effectiveness on 22 commits to three subject programs: Mujs, Libxml2, and Poppler. For programs using more human-readable file formats, like XML or JavaScript, we found Cleverest performed very well. It generated easy-to-understand bug-revealing or bug-reproduction test cases for the majority of commits in just under three minutes -- even when only the code diff or commit message (unless it was too vague) was given. For programs with more compact file formats, like PDF, as expected, it struggled to generate effective test cases. However, the LLM-supplied test cases are not very far from becoming effective (e.g., when used as a seed by a greybox fuzzer or as a starting point by the developer). 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化软件工程中显示出了巨大的潜力。在本文中，我们探讨了LLMs在为那些接收高度结构化且可读性较强的输入（如XML解析器或JavaScript解释器）的程序自动生成回归测试方面的潜力。具体来说，我们研究了以下几种难以在缺乏相应输入语法规则的情况下自动测试的程序的回归测试生成场景：
- **缺陷查找。** 给定代码变更（例如，一次提交或拉取请求），我们的基于LLM的方法生成一个测试用例，其目的是揭示如果应用该变更可能会引入的任何缺陷。
- **补丁测试。** 给定一个补丁，我们的基于LLM的方法生成一个测试用例，该用例在补丁前失败但在补丁后通过。这个测试用例可以添加到回归测试套件中，以在未来捕捉类似的缺陷。

我们实现了Cleverest，这是一种基于反馈的、零样本的LLM回归测试生成技术，并在三个主题程序（Mujs、Libxml2和Poppler）的22次提交上对其有效性进行了评估。对于使用更易于阅读的文件格式（如XML或JavaScript）的程序，我们发现Cleverest表现非常出色。它在不到三分钟的时间内为绝大多数提交生成了易于理解的揭示或再现缺陷的测试用例——即使只提供代码差异或提交信息（前提是信息足够具体）。对于使用更紧凑文件格式（如PDF）的程序，如预期的那样，它在生成有效的测试用例方面遇到困难。然而，由LLM提供的测试用例距离有效并不远，例如，当作为灰盒模糊测试器的种子或开发者的起点时。 

---
# Federated Deep Reinforcement Learning for Energy Efficient Multi-Functional RIS-Assisted Low-Earth Orbit Networks 

**Title (ZH)**: 联邦深度强化学习在辅助低地球轨道网络中的多功能可控反射表面（RIS）部署以提高能源效率 

**Authors**: Li-Hsiang Shen, Jyun-Jhe Huang, Kai-Ten Feng, Lie-Liang Yang, Jen-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11079)  

**Abstract**: In this paper, a novel network architecture that deploys the multi-functional reconfigurable intelligent surface (MF-RIS) in low-Earth orbit (LEO) is proposed. Unlike traditional RIS with only signal reflection capability, the MF-RIS can reflect, refract, and amplify signals, as well as harvest energy from wireless signals. Given the high energy demands in shadow regions where solar energy is unavailable, MF-RIS is deployed in LEO to enhance signal coverage and improve energy efficiency (EE). To address this, we formulate a long-term EE optimization problem by determining the optimal parameters for MF-RIS configurations, including amplification and phase-shifts, energy harvesting ratios, and LEO transmit beamforming. To address the complex non-convex and non-linear problem, a federated learning enhanced multi-agent deep deterministic policy gradient (FEMAD) scheme is designed. Multi-agent DDPG of each agent can provide the optimal action policy from its interaction to environments, whereas federated learning enables the hidden information exchange among multi-agents. In numerical results, we can observe significant EE improvements compared to the other benchmarks, including centralized deep reinforcement learning as well as distributed multi-agent deep deterministic policy gradient (DDPG). Additionally, the proposed LEO-MF-RIS architecture has demonstrated its effectiveness, achieving the highest EE performance compared to the scenarios of fixed/no energy harvesting in MF-RIS, traditional reflection-only RIS, and deployment without RISs/MF-RISs. 

**Abstract (ZH)**: 本文提出了一种新颖的网络架构，该架构在低地球轨道（LEO）中部署了多功能可重构智能表面（MF-RIS）。不同于传统仅具有信号反射能力的RIS，MF-RIS能够进行信号反射、折射、放大，并能够从无线信号中获取能量。由于在阴影区域太阳能不可用导致的高能耗需求，MF-RIS被部署在LEO中以增强信号覆盖并提高能量效率（EE）。为此，通过确定MF-RIS配置的最佳参数（包括放大和相位调制、能量采集比率以及LEO传输波束成形），我们提出了长期EE优化问题。为解决这一复杂的非凸性和非线性问题，设计了一种联邦学习增强的多智能体深度确定性策略梯度（FEMAD）方案。每个智能体的多智能体DDPG可以提供其与环境交互后的最优动作策略，而联邦学习则允许多智能体之间隐藏信息的交换。在数值结果中，我们观察到相比于集中式深度强化学习以及分布式多智能体深度确定性策略梯度（DDPG）等基准方法，EE有了显著提高。另外，提出的LEO-MF-RIS架构已显示出其有效性，在固定/无能量采集的MF-RIS场景、仅反射的RIS传统场景以及不部署RIS/MF-RIS的场景中，其实现了最高的EE性能。 

---
# IntellAgent: A Multi-Agent Framework for Evaluating Conversational AI Systems 

**Title (ZH)**: IntellAgent：多智能体系统框架评估对话式人工智能系统 

**Authors**: Elad Levi, Ilan Kadar  

**Link**: [PDF](https://arxiv.org/pdf/2501.11067)  

**Abstract**: Large Language Models (LLMs) are transforming artificial intelligence, evolving into task-oriented systems capable of autonomous planning and execution. One of the primary applications of LLMs is conversational AI systems, which must navigate multi-turn dialogues, integrate domain-specific APIs, and adhere to strict policy constraints. However, evaluating these agents remains a significant challenge, as traditional methods fail to capture the complexity and variability of real-world interactions. We introduce IntellAgent, a scalable, open-source multi-agent framework designed to evaluate conversational AI systems comprehensively. IntellAgent automates the creation of diverse, synthetic benchmarks by combining policy-driven graph modeling, realistic event generation, and interactive user-agent simulations. This innovative approach provides fine-grained diagnostics, addressing the limitations of static and manually curated benchmarks with coarse-grained metrics. IntellAgent represents a paradigm shift in evaluating conversational AI. By simulating realistic, multi-policy scenarios across varying levels of complexity, IntellAgent captures the nuanced interplay of agent capabilities and policy constraints. Unlike traditional methods, it employs a graph-based policy model to represent relationships, likelihoods, and complexities of policy interactions, enabling highly detailed diagnostics. IntellAgent also identifies critical performance gaps, offering actionable insights for targeted optimization. Its modular, open-source design supports seamless integration of new domains, policies, and APIs, fostering reproducibility and community collaboration. Our findings demonstrate that IntellAgent serves as an effective framework for advancing conversational AI by addressing challenges in bridging research and deployment. The framework is available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）正在改变人工智能领域，进化成为能够自主规划和执行任务的系统。LLMs 的主要应用之一是对话型人工智能系统，这些系统必须导航多轮对话、整合特定领域的API，并遵守严格的政策限制。然而，评估这些代理仍然是一个重大挑战，因为传统方法无法捕捉真实世界互动的复杂性和多样性。我们提出了IntellAgent，这是一个可扩展的开源多代理框架，旨在全面评估对话型人工智能系统。IntellAgent通过结合基于策略的图建模、真实的事件生成和交互式用户-代理模拟，自动化创建多种合成基准。这种方法提供细粒度的诊断，解决了静态和人工策画基准的粗粒度度量所存在的局限性。IntellAgent代表了评估对话型人工智能的一个范式转变。通过模拟不同复杂程度的真实多策略场景，IntellAgent捕捉了代理能力和政策约束之间的微妙交互。与传统方法不同，它使用基于图的策略模型来表示策略之间的关系、可能性及其复杂性，从而实现高度详细的诊断。IntellAgent还识别出关键的性能差距，提供了针对目标优化的可操作见解。其模块化和开源设计支持新领域、策略和API的无缝集成，促进可再现性和社区协作。我们的研究结果表明，IntellAgent是一个有效的框架，有助于通过解决研究与部署之间的鸿沟来推进对话型人工智能的发展。该框架可以在以下网址获取：[此链接] 

---
# Enhancing Neural Spoken Language Recognition: An Exploration with Multilingual Datasets 

**Title (ZH)**: 增强神经声学语言识别：多语言数据集的探索 

**Authors**: Or Haim Anidjar, Roi Yozevitch  

**Link**: [PDF](https://arxiv.org/pdf/2501.11065)  

**Abstract**: In this research, we advanced a spoken language recognition system, moving beyond traditional feature vector-based models. Our improvements focused on effectively capturing language characteristics over extended periods using a specialized pooling layer. We utilized a broad dataset range from Common-Voice, targeting ten languages across Indo-European, Semitic, and East Asian families. The major innovation involved optimizing the architecture of Time Delay Neural Networks. We introduced additional layers and restructured these networks into a funnel shape, enhancing their ability to process complex linguistic patterns. A rigorous grid search determined the optimal settings for these networks, significantly boosting their efficiency in language pattern recognition from audio samples. The model underwent extensive training, including a phase with augmented data, to refine its capabilities. The culmination of these efforts is a highly accurate system, achieving a 97\% accuracy rate in language recognition. This advancement represents a notable contribution to artificial intelligence, specifically in improving the accuracy and efficiency of language processing systems, a critical aspect in the engineering of advanced speech recognition technologies. 

**Abstract (ZH)**: 在本研究中，我们提出了一种语音语言识别系统，超越了传统基于特征向量的模型。我们的改进集中在使用专有的池化层有效地捕捉长时间内的语言特点。我们利用了来自Common-Voice的广泛数据集，涵盖了印欧语系、闪米特语系和东亚语系的十种语言。主要创新在于优化了时间延迟神经网络的架构。我们引入了额外的层，并将这些网络重新结构化为漏斗形，从而增强了其处理复杂语言模式的能力。通过严格的网格搜索，确定了这些网络的最佳参数设置，显著提高了其从音频样本中识别语言模式的效率。模型经过了广泛的训练，包括带有增强数据的阶段，以进一步提高其能力。这些努力的最终成果是一个高度准确的系统，其语言识别准确率达到97%。这项进展对人工智能领域，特别是在提高语言处理系统准确性与效率方面做出了重要贡献，是高级语音识别技术工程中的一项关键方面。 

---
# BF-STVSR: B-Splines and Fourier-Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution 

**Title (ZH)**: BF-STVSR: B-样条与傅里叶变换的完美结合实现高保真时空视频超分辨率 

**Authors**: Eunjin Kim, Hyeonjin Kim, Kyong Hwan Jin, Jaejun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2501.11043)  

**Abstract**: Enhancing low-resolution, low-frame-rate videos to high-resolution, high-frame-rate quality is essential for a seamless user experience, motivating advancements in Continuous Spatial-Temporal Video Super Resolution (C-STVSR). While prior methods employ Implicit Neural Representation (INR) for continuous encoding, they often struggle to capture the complexity of video data, relying on simple coordinate concatenation and pre-trained optical flow network for motion representation. Interestingly, we find that adding position encoding, contrary to common observations, does not improve-and even degrade performance. This issue becomes particularly pronounced when combined with pre-trained optical flow networks, which can limit the model's flexibility. To address these issues, we propose BF-STVSR, a C-STVSR framework with two key modules tailored to better represent spatial and temporal characteristics of video: 1) B-spline Mapper for smooth temporal interpolation, and 2) Fourier Mapper for capturing dominant spatial frequencies. Our approach achieves state-of-the-art PSNR and SSIM performance, showing enhanced spatial details and natural temporal consistency. 

**Abstract (ZH)**: 将低分辨率、低帧率视频提升至高分辨率、高帧率质量对于提供无缝用户体验至关重要，这促使了连续空间时间视频超分辨率（C-STVSR）的不断进步。尽管先前的方法使用隐式神经表示（INR）进行连续编码，但在捕捉视频数据的复杂性方面常常遇到困难，往往依赖简单的坐标拼接和预训练的光学流网络来表示运动。有趣的是，我们发现，在不增加位置编码的情况下，效果不仅没有提升，反而有所下降。当与预训练的光学流网络结合使用时，这一问题尤为明显，这会限制模型的灵活性。为了解决这些问题，我们提出了一种名为BF-STVSR的C-STVSR框架，该框架包含两个关键模块，以更好地表示视频的空间和时间特性：1）B-样条映射器（B-spline Mapper）实现平滑的时间插值，和2）傅里叶映射器（Fourier Mapper）捕捉主导的空间频率。我们的方法在峰值信噪比（PSNR）和结构相似性（SSIM）性能上达到了最新水平，展示了增强的空间细节和自然的时间一致性。 

---
# AdaptiveLog: An Adaptive Log Analysis Framework with the Collaboration of Large and Small Language Model 

**Title (ZH)**: 自适应日志：大型和小型语言模型协同工作的自适应日志分析框架 

**Authors**: Lipeng Ma, Weidong Yang, Yixuan Li, Ben Fei, Mingjie Zhou, Shuhao Li, Sihang Jiang, Bo Xu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.11031)  

**Abstract**: Automated log analysis is crucial to ensure high availability and reliability of complex systems. The advent of LLMs in NLP has ushered in a new era of language model-driven automated log analysis, garnering significant interest. Within this field, two primary paradigms based on language models for log analysis have become prominent. Small Language Models (SLMs) follow the pre-train and fine-tune paradigm, focusing on the specific log analysis task through fine-tuning on supervised datasets. On the other hand, LLMs following the in-context learning paradigm, analyze logs by providing a few examples in prompt contexts without updating parameters. Despite their respective strengths, we notice that SLMs are more cost-effective but less powerful, whereas LLMs with large parameters are highly powerful but expensive and inefficient. To trade-off between the performance and inference costs of both models in automated log analysis, this paper introduces an adaptive log analysis framework known as AdaptiveLog, which effectively reduces the costs associated with LLM while ensuring superior results. This framework collaborates an LLM and a small language model, strategically allocating the LLM to tackle complex logs while delegating simpler logs to the SLM. Specifically, to efficiently query the LLM, we propose an adaptive selection strategy based on the uncertainty estimation of the SLM, where the LLM is invoked only when the SLM is uncertain. In addition, to enhance the reasoning ability of the LLM in log analysis tasks, we propose a novel prompt strategy by retrieving similar error-prone cases as the reference, enabling the model to leverage past error experiences and learn solutions from these cases. Extensive experiments demonstrate that AdaptiveLog achieves state-of-the-art results across different tasks, elevating the overall accuracy of log analysis while maintaining cost efficiency. 

**Abstract (ZH)**: 自动日志分析对于确保复杂系统的高可用性和可靠性至关重要。自然语言处理（NLP）中的大规模语言模型（LLMs）的出现开辟了基于语言模型的自动日志分析的新时代，引起了广泛关注。在这个领域中，有两种主要基于语言模型的日志分析范式脱颖而出。小型语言模型（SLMs）遵循预先训练和微调的范式，通过在监督数据集上进行微调，专注于特定的日志分析任务。另一方面，遵循上下文学习范式的LLMs通过在提示中提供少量示例来分析日志，无需更新参数。尽管它们各自具有优势，但我们可以发现，SLMs更具成本效益但功能较弱，而具有大量参数的LLMs则功能强大但成本高昂且效率低下。为在自动日志分析中平衡这两种模型的性能和推理成本，本文提出了一种自适应日志分析框架，命名为AdaptiveLog，该框架有效降低了LLMs的成本，同时确保了优越的结果。该框架协作使用一个LLM和一个小语言模型，战略性地将LLM分配给处理复杂日志，而将简单的日志分配给SLM。具体来说，为有效查询LLM，我们提出了一种基于SLM不确定性估计的自适应选择策略，在SLM不确定时才调用LLM。此外，为了增强LLM在日志分析任务中的推理性，我们提出了一种新颖的提示策略，通过提取类似错误案例作为参考，使模型能够借鉴过去的错误经验，并从这些案例中学习解决方案。广泛的实验表明，AdaptiveLog在不同任务中实现了最先进的结果，提高了日志分析的整体准确性，同时保持了成本效率。 

---
# GREEN-CODE: Optimizing Energy Efficiency in Large Language Models for Code Generation 

**Title (ZH)**: GREEN-CODE: 优化大型语言模型的代码生成能效 

**Authors**: Shashikant Ilager, Lukas Florian Briem, Ivona Brandic  

**Link**: [PDF](https://arxiv.org/pdf/2501.11006)  

**Abstract**: Large Language Models (LLMs) are becoming integral to daily life, showcasing their vast potential across various Natural Language Processing (NLP) tasks. Beyond NLP, LLMs are increasingly used in software development tasks, such as code completion, modification, bug fixing, and code translation. Software engineers widely use tools like GitHub Copilot and Amazon Q, streamlining workflows and automating tasks with high accuracy. While the resource and energy intensity of LLM training is often highlighted, inference can be even more resource-intensive over time, as it's a continuous process with a high number of invocations. Therefore, developing resource-efficient alternatives for LLM inference is crucial for sustainability. This work proposes GREEN-CODE, a framework for energy-aware code generation in LLMs. GREEN-CODE performs dynamic early exit during LLM inference. We train a Reinforcement Learning (RL) agent that learns to balance the trade-offs between accuracy, latency, and energy consumption. Our approach is evaluated on two open-source LLMs, Llama 3.2 3B and OPT 2.7B, using the JavaCorpus and PY150 datasets. Results show that our method reduces the energy consumption between 23-50 % on average for code generation tasks without significantly affecting accuracy. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在逐渐融入日常生活中，并展现出在各种自然语言处理（NLP）任务中的巨大潜力。除了NLP领域，LLMs还广泛应用于软件开发任务中，包括代码补全、修改、错误修复和代码翻译。软件工程师们常用GitHub Copilot和Amazon Q等工具来简化工作流程并提高任务的自动化程度，确保高精度。虽然LLM训练在资源和能源消耗方面常常被强调，但推理过程可能会随着时间的推移变得更加耗费资源，因为它是一个持续的过程，涉及大量的调用。因此，开发资源高效的方法对于LLM推理的可持续性至关重要。本文提出了GREEN-CODE框架，这是一种面向能耗的代码生成框架。GREEN-CODE在LLM推理过程中执行动态早期退出。我们训练了一个强化学习（RL）代理，以学习在准确性、延迟和能耗之间进行权衡。我们的方法在两个开源LLM（Llama 3.2 3B和OPT 2.7B）上使用JavaCorpus和PY150数据集进行了评估。结果表明，我们的方法在代码生成任务中平均降低了23-50%的能耗，且对准确性影响不大。 

---
# The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs 

**Title (ZH)**: LLM作为法官的替代注释员测试：如何通过统计方法证明可以用LLM替代人类注释员 

**Authors**: Nitay Calderon, Roi Reichart, Rotem Dror  

**Link**: [PDF](https://arxiv.org/pdf/2501.10970)  

**Abstract**: The "LLM-as-a-judge" paradigm employs Large Language Models (LLMs) as annotators and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure -- the Alternative Annotator Test (alt-test) -- that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming open-source LLMs, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices. 

**Abstract (ZH)**: “LLM-as-a-judge”范式利用大型语言模型（LLMs）作为传统上由人类完成的任务中的注释员和评估者。LLM注解在多个领域广泛使用，不仅限于自然语言处理研究，还涉及到医学、心理学和社会科学等领域。尽管LLMs在研究结果和见解的形成中起着重要作用，但对于是否能够替代人类注释员并没有标准和严谨的鉴定程序。本文提出了一种新的统计方法——替代注释员检验（alt-test），仅需要少量注释示例即可证明使用LLM注解的有效性。此外，我们介绍了用于比较LLM评审员的灵活且可解释性较强的度量标准。为了展示我们的方法，我们精选了十个多样化的数据集，其中包括语言和视觉-语言任务，并采用了六种LLM和四种提示技术进行了实验。结果表明，在某些情况下，闭源LLM（如GPT-4o）可以替代人类，表现优于开源LLM，并且提示技术产生了不同质量的评审员。我们希望这项研究能够促进更加严谨和可靠的实践。 

---
# Advancing General Multimodal Capability of Vision-language Models with Pyramid-descent Visual Position Encoding 

**Title (ZH)**: 通过金字塔降布局位编码提升视觉语言模型的通用多模态能力 

**Authors**: Zhanpeng Chen, Mingxiao Li, Ziyang Chen, Nan Du, Xiaolong Li, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2501.10967)  

**Abstract**: Vision-language Models (VLMs) have shown remarkable capabilities in advancing general artificial intelligence, yet the irrational encoding of visual positions persists in inhibiting the models' comprehensive perception performance across different levels of granularity. In this work, we propose Pyramid-descent Visual Position Encoding (PyPE), a novel approach designed to enhance the perception of visual tokens within VLMs. By assigning visual position indexes from the periphery to the center and expanding the central receptive field incrementally, PyPE addresses the limitations of traditional raster-scan methods and mitigates the long-term decay effects induced by Rotary Position Embedding (RoPE). Our method reduces the relative distance between interrelated visual elements and instruction tokens, promoting a more rational allocation of attention weights and allowing for a multi-granularity perception of visual elements and countering the over-reliance on anchor tokens. Extensive experimental evaluations demonstrate that PyPE consistently improves the general capabilities of VLMs across various sizes. Code is available at this https URL. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在推进通用人工智能方面展示出了显著的能力，然而传统的 raster-scan 方法在视觉位置编码上的不合理性仍然对模型的综合感知性能构成了阻碍。在这项工作中，我们提出了一种新颖的方法——金字塔下沉视觉位置编码（PyPE），旨在增强 VLMs 中视觉标记的感知能力。通过从边缘到中心分配视觉位置索引，并逐步扩展中心的感受野，PyPE 解决了传统 raster-scan 方法的局限性，并缓解了由旋转位置嵌入（RoPE）引起的长期衰减效应。该方法减少了相关视觉元素和指令标记之间的相对距离，促进了更合理的注意力权重分配，并允许对视觉元素进行多粒度感知，从而减少对锚标记的过度依赖。广泛的实验证明，PyPE 在不同大小的 VLMs 中一致地提高了其通用能力。代码可在以下链接获取：this https URL。 

---
# DC-PCN: Point Cloud Completion Network with Dual-Codebook Guided Quantization 

**Title (ZH)**: DC-PCN：带有双重码本引导量化的心点云补全网络 

**Authors**: Qiuxia Wu, Haiyang Huang, Kunming Su, Zhiyong Wang, Kun Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10966)  

**Abstract**: Point cloud completion aims to reconstruct complete 3D shapes from partial 3D point clouds. With advancements in deep learning techniques, various methods for point cloud completion have been developed. Despite achieving encouraging results, a significant issue remains: these methods often overlook the variability in point clouds sampled from a single 3D object surface. This variability can lead to ambiguity and hinder the achievement of more precise completion results. Therefore, in this study, we introduce a novel point cloud completion network, namely Dual-Codebook Point Completion Network (DC-PCN), following an encder-decoder pipeline. The primary objective of DC-PCN is to formulate a singular representation of sampled point clouds originating from the same 3D surface. DC-PCN introduces a dual-codebook design to quantize point-cloud representations from a multilevel perspective. It consists of an encoder-codebook and a decoder-codebook, designed to capture distinct point cloud patterns at shallow and deep levels. Additionally, to enhance the information flow between these two codebooks, we devise an information exchange mechanism. This approach ensures that crucial features and patterns from both shallow and deep levels are effectively utilized for completion. Extensive experiments on the PCN, ShapeNet\_Part, and ShapeNet34 datasets demonstrate the state-of-the-art performance of our method. 

**Abstract (ZH)**: 点云完成的目标是从部分3D点云中重构完整的3D形状。随着深度学习技术的进步，已经开发出了多种点云完成方法。尽管这些方法取得了一定的成效，但仍存在一个显著问题：这些方法常常忽视来自单个3D物体表面的点云的变异性。这种变异性可能会导致模糊性，从而阻碍更精确完成结果的实现。因此，在本研究中，我们提出了一种新颖的点云完成网络——双码本点云完成网络（DC-PCN），遵循编码-解码框架。DC-PCN的主要目的是对源自相同3D表面的采样点云形成单一的表示。DC-PCN引入了双码本设计，从多级视角对点云表示进行量化。它由编码码本和解码码本组成，分别用于捕捉浅层和深层的点云模式。此外，为了增强这两个码本之间的信息流，我们设计了一种信息交换机制。这种方法确保了从浅层和深层提取的关键特征和模式能够有效地用于完成。在PCN、ShapeNet\_Part和ShapeNet34数据集上的广泛实验表明，我们的方法具有最先进的性能。 

---
# MARIO: A Mixed Annotation Framework For Polyp Segmentation 

**Title (ZH)**: MARIO：一种混合标注框架用于息肉分割 

**Authors**: Haoyang Li, Yiwen Hu, Jun Wei, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.10957)  

**Abstract**: Existing polyp segmentation models are limited by high labeling costs and the small size of datasets. Additionally, vast polyp datasets remain underutilized because these models typically rely on a single type of annotation. To address this dilemma, we introduce MARIO, a mixed supervision model designed to accommodate various annotation types, significantly expanding the range of usable data. MARIO learns from underutilized datasets by incorporating five forms of supervision: pixel-level, box-level, polygon-level, scribblelevel, and point-level. Each form of supervision is associated with a tailored loss that effectively leverages the supervision labels while minimizing the noise. This allows MARIO to move beyond the constraints of relying on a single annotation type. Furthermore, MARIO primarily utilizes dataset with weak and cheap annotations, reducing the dependence on large-scale, fully annotated ones. Experimental results across five benchmark datasets demonstrate that MARIO consistently outperforms existing methods, highlighting its efficacy in balancing trade-offs between different forms of supervision and maximizing polyp segmentation performance 

**Abstract (ZH)**: 现有的结肠息肉分割模型受限于高标注成本和数据集规模较小的问题。此外，由于这些模型通常依赖单一类型的标注，大量结肠息肉数据集仍未得到有效利用。为了应对这一挑战，我们提出了一种混合监督模型MARIO，该模型能够适应多种类型的标注，显著扩大可用数据的范围。MARIO通过整合五种形式的监督（像素级、框级、多边形级、scribble级和点级）来从未充分利用的数据集中学习，并为每种监督形式引入了专门的损失函数，该损失函数有效地利用了监督标签并最大程度地减少了噪声。这使得MARIO能够超越单一标注类型的限制。此外，MARIO主要利用弱标注和低成本的标注数据集，大幅减少了对大规模、全标注数据集的依赖。在五个基准数据集上的实验结果表明，MARIO在所有方法中表现最佳，突显了其在平衡不同类型监督之间的权衡以及最大化结肠息肉分割性能方面的有效性。 

---
# InsQABench: Benchmarking Chinese Insurance Domain Question Answering with Large Language Models 

**Title (ZH)**: InsQABench：大型语言模型在中文保险领域问题回答中的基准测试 

**Authors**: Jing Ding, Kai Feng, Binbin Lin, Jiarui Cai, Qiushi Wang, Yu Xie, Xiaojin Zhang, Zhongyu Wei, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.10943)  

**Abstract**: The application of large language models (LLMs) has achieved remarkable success in various fields, but their effectiveness in specialized domains like the Chinese insurance industry remains underexplored. The complexity of insurance knowledge, encompassing specialized terminology and diverse data types, poses significant challenges for both models and users. To address this, we introduce InsQABench, a benchmark dataset for the Chinese insurance sector, structured into three categories: Insurance Commonsense Knowledge, Insurance Structured Database, and Insurance Unstructured Documents, reflecting real-world insurance question-answering this http URL also propose two methods, SQL-ReAct and RAG-ReAct, to tackle challenges in structured and unstructured data tasks. Evaluations show that while LLMs struggle with domain-specific terminology and nuanced clause texts, fine-tuning on InsQABench significantly improves performance. Our benchmark establishes a solid foundation for advancing LLM applications in the insurance domain, with data and code available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个领域实现了显著的成功，但它们在如中国保险业这类专业领域的有效性仍待深入探索。保险领域的知识复杂性，包括专门的术语和多样性的数据类型，对模型和用户均构成了重大挑战。为应对这些挑战，我们提出了一套针对中国保险业的基准数据集InsQABench，并将其分为三大类别：保险常识知识、保险结构化数据库和保险非结构化文档，以反映实际的保险问题解答场景。此外，我们还提出了两种方法，即SQL-ReAct和RAG-ReAct，以解决结构化和非结构化数据任务中的挑战。评估结果显示，尽管LLMs在处理特定领域术语和复杂的条款文本方面存在困难，但在InsQABench上进行微调可以显著提高模型性能。我们的基准数据集为推进LLM在保险领域的应用奠定了坚实的基础，相关的数据和代码可在此访问：[提供链接]。 

---
# Blockchain-assisted Demonstration Cloning for Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: 基于区块链辅助的多智能体深度强化学习示范克隆 

**Authors**: Ahmed Alagha, Jamal Bentahar, Hadi Otrok, Shakti Singh, Rabeb Mizouni  

**Link**: [PDF](https://arxiv.org/pdf/2501.10938)  

**Abstract**: Multi-Agent Deep Reinforcement Learning (MDRL) is a promising research area in which agents learn complex behaviors in cooperative or competitive environments. However, MDRL comes with several challenges that hinder its usability, including sample efficiency, curse of dimensionality, and environment exploration. Recent works proposing Federated Reinforcement Learning (FRL) to tackle these issues suffer from problems related to model restrictions and maliciousness. Other proposals using reward shaping require considerable engineering and could lead to local optima. In this paper, we propose a novel Blockchain-assisted Multi-Expert Demonstration Cloning (MEDC) framework for MDRL. The proposed method utilizes expert demonstrations in guiding the learning of new MDRL agents, by suggesting exploration actions in the environment. A model sharing framework on Blockchain is designed to allow users to share their trained models, which can be allocated as expert models to requesting users to aid in training MDRL systems. A Consortium Blockchain is adopted to enable traceable and autonomous execution without the need for a single trusted entity. Smart Contracts are designed to manage users and models allocation, which are shared using IPFS. The proposed framework is tested on several applications, and is benchmarked against existing methods in FRL, Reward Shaping, and Imitation Learning-assisted RL. The results show the outperformance of the proposed framework in terms of learning speed and resiliency to faulty and malicious models. 

**Abstract (ZH)**: 多智能体深度强化学习（MDRL）是研究中一个有前景的领域，在该领域中，智能体在合作或竞争环境中学习复杂行为。然而，MDRL 面临着几个挑战，这些挑战限制了其应用性，包括样本效率低、维数灾难以及环境探索。近年来，提出使用联邦强化学习（FRL）来解决这些问题的方法，但这些方法存在模型限制和恶意行为相关的问题。其他使用奖励重塑的方法则需要大量的工程工作，可能会导致局部最优解。在本文中，我们提出了一种名为区块链辅助多专家示范克隆（Blockchain-assisted Multi-Expert Demonstration Cloning, MEDC）框架的新型多智能体深度强化学习方法。所提出的方法利用专家示范引导新MDRL智能体的学习，并建议在环境中的探索动作。一个基于区块链的模型共享框架被设计出来，允许用户共享其训练模型，并将这些模型分配给请求方以协助训练MDRL系统。采用联盟区块链来实现可追溯且自治的执行，而无需单一可信实体的参与。智能合约被设计用于管理用户和模型分配，并通过IPFS共享。所提出的框架在多个应用场景中进行了测试，并与现有的FRL、奖励重塑和模仿学习辅助的RL方法进行了基准测试。结果表明，该框架在学习速度和对故障和恶意模型的鲁棒性方面优于现有方法。 

---
# TSVC:Tripartite Learning with Semantic Variation Consistency for Robust Image-Text Retrieval 

**Title (ZH)**: TSVC：具有语义变异性一致性的三方学习方法在稳健的图像-文本检索中的应用 

**Authors**: Shuai Lyu, Zijing Tian, Zhonghong Ou, Yifan Zhu, Xiao Zhang, Qiankun Ha, Haoran Luo, Meina Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.10935)  

**Abstract**: Cross-modal retrieval maps data under different modality via semantic relevance. Existing approaches implicitly assume that data pairs are well-aligned and ignore the widely existing annotation noise, i.e., noisy correspondence (NC). Consequently, it inevitably causes performance degradation. Despite attempts that employ the co-teaching paradigm with identical architectures to provide distinct data perspectives, the differences between these architectures are primarily stemmed from random initialization. Thus, the model becomes increasingly homogeneous along with the training process. Consequently, the additional information brought by this paradigm is severely limited. In order to resolve this problem, we introduce a Tripartite learning with Semantic Variation Consistency (TSVC) for robust image-text retrieval. We design a tripartite cooperative learning mechanism comprising a Coordinator, a Master, and an Assistant model. The Coordinator distributes data, and the Assistant model supports the Master model's noisy label prediction with diverse data. Moreover, we introduce a soft label estimation method based on mutual information variation, which quantifies the noise in new samples and assigns corresponding soft labels. We also present a new loss function to enhance robustness and optimize training effectiveness. Extensive experiments on three widely used datasets demonstrate that, even at increasing noise ratios, TSVC exhibits significant advantages in retrieval accuracy and maintains stable training performance. 

**Abstract (ZH)**: 跨模态检索通过语义相关性将不同模态的数据映射到同一空间。现有方法隐含地假定数据对是良好对齐的，并且忽视了广泛存在的标注噪声，即噪声对应关系（NC）。因此，这不可避免地导致了性能下降。尽管有采用协同教学范式的方法，并利用相同的架构从不同角度提供数据视角，但这些架构之间的差异主要源于随机初始化。因此，随着训练过程的推进，模型变得越来越同质化。由此带来的附加信息严重受限。为了解决这一问题，我们提出了一种基于语义变异性一致性的三元学习框架（TSVC）以提高图像-文本检索的鲁棒性。我们设计了一种包含协调器、主模型和助手模型的三元合作学习机制。协调器负责分配数据，助手模型利用多样化数据支持主模型噪声标签预测。此外，我们引入了一种基于互信息变性的软标签估计方法，用于量化新样本中的噪声并分配相应的软标签。我们还提出了一种新的损失函数以增强鲁棒性并优化训练效果。在三个广泛使用的数据集上的广泛实验表明，即使在增加噪声比例的情况下，TSVC在检索准确性和保持稳定的训练性能方面仍具有显著优势。 

---
# Generative Physical AI in Vision: A Survey 

**Title (ZH)**: 视觉中的生成物理AI：一篇综述 

**Authors**: Daochang Liu, Junyu Zhang, Anh-Dung Dinh, Eunbyung Park, Shichao Zhang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10928)  

**Abstract**: Generative Artificial Intelligence (AI) has rapidly advanced the field of computer vision by enabling machines to create and interpret visual data with unprecedented sophistication. This transformation builds upon a foundation of generative models to produce realistic images, videos, and 3D or 4D content. Traditionally, generative models primarily focus on visual fidelity while often neglecting the physical plausibility of generated content. This gap limits their effectiveness in applications requiring adherence to real-world physical laws, such as robotics, autonomous systems, and scientific simulations. As generative AI evolves to increasingly integrate physical realism and dynamic simulation, its potential to function as a "world simulator" expands-enabling the modeling of interactions governed by physics and bridging the divide between virtual and physical realities. This survey systematically reviews this emerging field of physics-aware generative AI in computer vision, categorizing methods based on how they incorporate physical knowledge-either through explicit simulation or implicit learning. We analyze key paradigms, discuss evaluation protocols, and identify future research directions. By offering a comprehensive overview, this survey aims to help future developments in physically grounded generation for vision. The reviewed papers are summarized at this https URL. 

**Abstract (ZH)**: 生成型人工智能（AI）迅速推动了计算机视觉领域的进步，使机器能够以前所未有的精细度生成和解释视觉数据。这一转变建立在生成模型的基础之上，能够生成逼真的图像、视频和3D或4D内容。传统上，生成模型主要关注视觉保真度，而往往忽视生成内容的物理合理性。这一差距限制了其在需要遵守现实世界物理定律的应用中的有效性，如机器人技术、自主系统和科学模拟等领域。随着生成AI不断发展，更加注重物理现实和动力学模拟的整合，其作为“世界模拟器”的潜力不断扩大，使得能够模拟受物理法则支配的交互，并弥合虚拟与物理现实之间的差距。

本综述系统地回顾了计算机视觉中物理感知生成型AI这一新兴领域，根据其如何整合物理知识（无论是通过显式模拟还是隐式学习）将方法分类。我们分析了关键范式，讨论了评估协议，并指出了未来的研究方向。通过提供一个全面的概述，本综述旨在帮助未来在视知觉中的物理基础生成方面的开发。所评审的论文摘要可在以下链接中找到：[这里插入链接] 

---
# Adaptive Target Localization under Uncertainty using Multi-Agent Deep Reinforcement Learning with Knowledge Transfer 

**Title (ZH)**: 基于知识转移的多代理深度强化学习在不确定性条件下的自适应目标定位 

**Authors**: Ahmed Alagha, Rabeb Mizouni, Shakti Singh, Jamal Bentahar, Hadi Otrok  

**Link**: [PDF](https://arxiv.org/pdf/2501.10924)  

**Abstract**: Target localization is a critical task in sensitive applications, where multiple sensing agents communicate and collaborate to identify the target location based on sensor readings. Existing approaches investigated the use of Multi-Agent Deep Reinforcement Learning (MADRL) to tackle target localization. Nevertheless, these methods do not consider practical uncertainties, like false alarms when the target does not exist or when it is unreachable due to environmental complexities. To address these drawbacks, this work proposes a novel MADRL-based method for target localization in uncertain environments. The proposed MADRL method employs Proximal Policy Optimization to optimize the decision-making of sensing agents, which is represented in the form of an actor-critic structure using Convolutional Neural Networks. The observations of the agents are designed in an optimized manner to capture essential information in the environment, and a team-based reward functions is proposed to produce cooperative agents. The MADRL method covers three action dimensionalities that control the agents' mobility to search the area for the target, detect its existence, and determine its reachability. Using the concept of Transfer Learning, a Deep Learning model builds on the knowledge from the MADRL model to accurately estimating the target location if it is unreachable, resulting in shared representations between the models for faster learning and lower computational complexity. Collectively, the final combined model is capable of searching for the target, determining its existence and reachability, and estimating its location accurately. The proposed method is tested using a radioactive target localization environment and benchmarked against existing methods, showing its efficacy. 

**Abstract (ZH)**: 目标定位是敏感应用中的关键任务，其中多个传感代理通过通信和协作，基于传感器读数来识别目标的位置。现有的方法已研究了使用多代理深度强化学习（MADRL）来解决目标定位问题。然而，这些方法并未考虑实际的不确定性，例如目标不存在时的误报警或因环境复杂性而导致目标不可达的情况。为了应对这些缺点，本工作提出了一种基于MADRL的方法，用于处理不确定环境中的目标定位。所提出的MADRL方法利用近端策略优化（PPO）来优化传感代理的决策制定，采用卷积神经网络（CNN）的形式表示为演员-评论家结构。代理的观测值被优化设计以捕获环境中的关键信息，并提出了一种基于团队的奖励函数以生成协同工作的代理。MADRL方法涵盖了三种行动维度，分别控制代理的移动性以搜索区域、检测目标的存在性以及确定其可达性。通过迁移学习的概念，深度学习模型基于MADRL模型的知识，能够在目标不可达时准确估计其位置，从而使得模型之间拥有共享表示，加快学习速度并降低计算复杂度。最后，最终组合模型能够搜索目标、确定其存在性与可达性，并准确估计其位置。所提出的方案在放射性目标定位环境中进行了测试，并与现有方法进行了基准测试，显示出其有效性。 

---
# Decomposing and Fusing Intra- and Inter-Sensor Spatio-Temporal Signal for Multi-Sensor Wearable Human Activity Recognition 

**Title (ZH)**: 将内部和跨传感器的空间时间信号分解与融合用于多传感器可穿戴人体活动识别 

**Authors**: Haoyu Xie, Haoxuan Li, Chunyuan Zheng, Haonan Yuan, Guorui Liao, Jun Liao, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10917)  

**Abstract**: Wearable Human Activity Recognition (WHAR) is a prominent research area within ubiquitous computing. Multi-sensor synchronous measurement has proven to be more effective for WHAR than using a single sensor. However, existing WHAR methods use shared convolutional kernels for indiscriminate temporal feature extraction across each sensor variable, which fails to effectively capture spatio-temporal relationships of intra-sensor and inter-sensor variables. We propose the DecomposeWHAR model consisting of a decomposition phase and a fusion phase to better model the relationships between modality variables. The decomposition creates high-dimensional representations of each intra-sensor variable through the improved Depth Separable Convolution to capture local temporal features while preserving their unique characteristics. The fusion phase begins by capturing relationships between intra-sensor variables and fusing their features at both the channel and variable levels. Long-range temporal dependencies are modeled using the State Space Model (SSM), and later cross-sensor interactions are dynamically captured through a self-attention mechanism, highlighting inter-sensor spatial correlations. Our model demonstrates superior performance on three widely used WHAR datasets, significantly outperforming state-of-the-art models while maintaining acceptable computational efficiency. Our codes and supplementary materials are available at this https URL. 

**Abstract (ZH)**: 可穿戴人体活动识别（WHAR）是泛在计算领域的一个重要研究方向。多传感器同步测量已被证明比单一传感器在WHAR中更为有效。然而，现有的WHAR方法在每个传感器变量上不分青红皂白地使用共享卷积核进行时域特征提取，未能有效地捕捉传感器内变量和传感器间变量的空间-时间关系。我们提出了一种DecomposeWHAR模型，该模型包括一个分解阶段和一个融合阶段，以更好地建模模态变量之间的关系。分解阶段通过改进的深度可分离卷积来生成每个传感器内变量的高维表示，以捕捉局部时域特征并保留其独特特性。融合阶段首先捕捉传感器内变量之间的关系，并在通道和变量层面融合它们的特征。长程时域依赖性通过状态空间模型（SSM）建模，随后通过自注意力机制动态捕捉跨传感器交互，强调传感器间的空间关联。我们的模型在三个广泛使用的WHAR数据集上表现出优越的性能，显著优于最先进的模型，同时保持了可接受的计算效率。我们的代码和补充材料可在以下链接获取：[这里插入链接]。 

---
# A Generative Security Application Engineering Curriculum 

**Title (ZH)**: 生成性安全应用工程课程 

**Authors**: Wu-chang Feng, David Baker-Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2501.10900)  

**Abstract**: Generative AI and large language models (LLMs) are transforming security by automating many tasks being performed manually. With such automation changing the practice of security as we know it, it is imperative that we prepare future students for the technology landscape they will ultimately face. Towards this end, we describe an initial curriculum and course that attempts to show students how to apply generative AI in order to solve problems in security. By refocusing security education and training on aspects uniquely suited for humans and showing students how to leverage automation for the rest, we believe we can better align security education practices with generative AI as it evolves. 

**Abstract (ZH)**: 生成式人工智能和大规模语言模型（LLMs）正在通过自动化许多手动执行的任务来重塑安全领域。随着这种自动化改变我们所熟知的安全实践，我们有必要为未来的学生成为他们最终将要面对的技术环境做好准备。为此，我们描述了一门初步的课程和课程设计，旨在向学生展示如何应用生成式人工智能以解决安全问题。通过重新聚焦于最适合人类的方面，并展示学生如何利用自动化技术，我们相信可以更好地使安全教育实践与生成式人工智能的发展相适应。 

---
# Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments 

**Title (ZH)**: 基于数据的交互学习：一种适用于现实环境的自适应代理自适应框架 

**Authors**: Hongjin Su, Ruoxi Sun, Jinsung Yoon, Pengcheng Yin, Tao Yu, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2501.10893)  

**Abstract**: Autonomous agents powered by large language models (LLMs) have the potential to enhance human capabilities, assisting with digital tasks from sending emails to performing data analysis. The abilities of existing LLMs at such tasks are often hindered by the lack of high-quality agent data from the corresponding environments they interact with. We propose Learn-by-interact, a data-centric framework to adapt LLM agents to any given environments without human annotations. Learn-by-interact synthesizes trajectories of agent-environment interactions based on documentations, and constructs instructions by summarizing or abstracting the interaction histories, a process called backward construction. We assess the quality of our synthetic data by using them in both training-based scenarios and training-free in-context learning (ICL), where we craft innovative retrieval approaches optimized for agents. Extensive experiments on SWE-bench, WebArena, OSWorld and Spider2-V spanning across realistic coding, web, and desktop environments show the effectiveness of Learn-by-interact in various downstream agentic tasks -- baseline results are improved by up to 12.2\% for ICL with Claude-3.5 and 19.5\% for training with Codestral-22B. We further demonstrate the critical role of backward construction, which provides up to 14.0\% improvement for training. Our ablation studies demonstrate the efficiency provided by our synthesized data in ICL and the superiority of our retrieval pipeline over alternative approaches like conventional retrieval-augmented generation (RAG). We expect that Learn-by-interact will serve as a foundation for agent data synthesis as LLMs are increasingly deployed at real-world environments. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的自主代理有可能增强人类的能力，协助完成从发送电子邮件到进行数据分析等各种数字任务。现有的LLMs在这些任务中的能力往往受到与之交互的相应环境中的高质量代理数据缺乏的限制。我们提出了一个数据为中心的框架——“通过交互学习”，该框架能够在无需人工注释的情况下使LLM代理适应任何给定的环境。通过文档，“通过交互学习”综合了代理-环境交互的轨迹，并通过总结或抽象交互历史来构建指令，这一过程称为反向构造。我们通过使用合成数据在基于训练的场景和无需训练的上下文学习（ICL）中评估其质量，其中我们设计了针对代理的创新检索方法。在SWE-bench、WebArena、OSWorld和Spider2-V等涵盖现实编码、网络和桌面环境的广泛实验中，展示了“通过交互学习”的有效性，通过使用Codestral-22B训练时，基准结果提高了19.5%，使用Claude-3.5进行ICL时提高了12.2%。我们进一步证明了反向构造的关键作用，其能够提供高达14.0%的训练改进。我们的消融研究表明，我们合成数据在ICL中的效率以及我们检索流水线相较于传统检索增强生成（RAG）等替代方法的优势。我们期望“通过交互学习”将成为LLMs部署到真实环境中的代理数据合成的基础。 

---
# OpenEarthMap-SAR: A Benchmark Synthetic Aperture Radar Dataset for Global High-Resolution Land Cover Mapping 

**Title (ZH)**: 开放地球地图-SAR：面向全球高分辨率土地覆盖制图的合成孔径雷达基准数据集 

**Authors**: Junshi Xia, Hongruixuan Chen, Clifford Broni-Bediako, Yimin Wei, Jian Song, Naoto Yokoya  

**Link**: [PDF](https://arxiv.org/pdf/2501.10891)  

**Abstract**: High-resolution land cover mapping plays a crucial role in addressing a wide range of global challenges, including urban planning, environmental monitoring, disaster response, and sustainable development. However, creating accurate, large-scale land cover datasets remains a significant challenge due to the inherent complexities of geospatial data, such as diverse terrain, varying sensor modalities, and atmospheric conditions. Synthetic Aperture Radar (SAR) imagery, with its ability to penetrate clouds and capture data in all-weather, day-and-night conditions, offers unique advantages for land cover mapping. Despite these strengths, the lack of benchmark datasets tailored for SAR imagery has limited the development of robust models specifically designed for this data modality. To bridge this gap and facilitate advancements in SAR-based geospatial analysis, we introduce OpenEarthMap-SAR, a benchmark SAR dataset, for global high-resolution land cover mapping. OpenEarthMap-SAR consists of 1.5 million segments of 5033 aerial and satellite images with the size of 1024$\times$1024 pixels, covering 35 regions from Japan, France, and the USA, with partially manually annotated and fully pseudo 8-class land cover labels at a ground sampling distance of 0.15--0.5 m. We evaluated the performance of state-of-the-art methods for semantic segmentation and present challenging problem settings suitable for further technical development. The dataset also serves the official dataset for IEEE GRSS Data Fusion Contest Track I. The dataset has been made publicly available at this https URL. 

**Abstract (ZH)**: 高分辨率土地覆盖图绘制在应对城市规划、环境监测、灾害响应和可持续发展等一系列全球挑战中发挥着关键作用。然而，由于地理空间数据固有的复杂性，例如多样的地形、不同的传感器模式和大气条件，创建准确的大规模土地覆盖数据集依旧是一个巨大挑战。合成孔径雷达（SAR）影像因其能在全天候、昼夜条件下穿透云层并获取数据的优势，在土地覆盖图绘制中具有独特的优势。尽管如此，缺乏针对SAR影像的专业基准数据集限制了专门为这种数据模式设计的稳健模型的发展。为填补这一空白并促进基于SAR的地理空间分析技术的发展，我们推出了OpenEarthMap-SAR，这是一个基准SAR数据集，用于全球高分辨率土地覆盖图绘制。OpenEarthMap-SAR 包含150万段共计5033张航空和卫星影像，每幅影像的尺寸为1024×1024像素，覆盖日本、法国和美国35个地区，并附有部分手动标注和全伪8类土地覆盖标签，地面采样距离为0.15-0.5米。我们评估了最先进的语义分割方法的表现，并提出了适合进一步技术发展的挑战性问题设置。该数据集还被用作IEEE GRSS数据融合竞赛第一赛道的官方数据集。数据集已在以下网址公开：[该网址]。 

---
# Generating Structured Outputs from Language Models: Benchmark and Studies 

**Title (ZH)**: 从语言模型生成结构化输出：基准测试与研究 

**Authors**: Saibo Geng, Hudson Cooper, Michał Moskal, Samuel Jenkins, Julian Berman, Nathan Ranchin, Robert West, Eric Horvitz, Harsha Nori  

**Link**: [PDF](https://arxiv.org/pdf/2501.10868)  

**Abstract**: Reliably generating structured outputs has become a critical capability for modern language model (LM) applications. Constrained decoding has emerged as the dominant technology across sectors for enforcing structured outputs during generation. Despite its growing adoption, little has been done with the systematic evaluation of the behaviors and performance of constrained decoding. Constrained decoding frameworks have standardized around JSON Schema as a structured data format, with most uses guaranteeing constraint compliance given a schema. However, there is poor understanding of the effectiveness of the methods in practice. We present an evaluation framework to assess constrained decoding approaches across three critical dimensions: efficiency in generating constraint-compliant outputs, coverage of diverse constraint types, and quality of the generated outputs. To facilitate this evaluation, we introduce JSONSchemaBench, a benchmark for constrained decoding comprising 10K real-world JSON schemas that encompass a wide range of constraints with varying complexity. We pair the benchmark with the existing official JSON Schema Test Suite and evaluate six state-of-the-art constrained decoding frameworks, including Guidance, Outlines, Llamacpp, XGrammar, OpenAI, and Gemini. Through extensive experiments, we gain insights into the capabilities and limitations of constrained decoding on structured generation with real-world JSON schemas. Our work provides actionable insights for improving constrained decoding frameworks and structured generation tasks, setting a new standard for evaluating constrained decoding and structured generation. We release JSONSchemaBench at this https URL 

**Abstract (ZH)**: 可靠生成结构化输出已成为现代语言模型（LM）应用中的关键能力。约束解码rompt:已逐渐成为各行业中在生成过程中强制结构化输出的主要技术。尽管其应用日益增加，但在系统评估约束解码的行为和性能方面的工作却相对较少。尽管大多数约束解码框架已经标准化了JSON Schema作为结构化数据格式，其中大多数用例在给定一个模式的情况下保证了约束合规性，但在实践中这些方法的有效性仍然缺乏深入理解。为此，我们提出了一种评估框架，从三个关键维度评估约束解码方法：生成合规输出的效率、约束类型多样性覆盖范围以及生成输出的质量。为支持这一评估，我们引入了JSONSchemaBench，这是一个包含10,000个真实世界JSON模式的基准，这些模式涵盖了不同复杂度的各种约束。我们还将该基准与现有的官方JSON Schema测试套件相结合，评估了六种最先进的约束解码框架，包括Guidance、Outlines、LlamaCpp、XGrammar、OpenAI和Gemini。通过广泛实验，我们深入了解了在真实世界的JSON模式下进行结构化生成时约束解码的能力和局限性。我们的研究为改进约束解码框架和结构化生成任务提供了可操作的见解，并为评估约束解码和结构化生成设立了新的标准。我们已在以下链接发布了JSONSchemaBench：[https://] 

---
# Dynamic Continual Learning: Harnessing Parameter Uncertainty for Improved Network Adaptation 

**Title (ZH)**: 动态持续学习：利用参数不确定性提高网络适应性 

**Authors**: Christopher Angelini, Nidhal Bouaynaya  

**Link**: [PDF](https://arxiv.org/pdf/2501.10861)  

**Abstract**: When fine-tuning Deep Neural Networks (DNNs) to new data, DNNs are prone to overwriting network parameters required for task-specific functionality on previously learned tasks, resulting in a loss of performance on those tasks. We propose using parameter-based uncertainty to determine which parameters are relevant to a network's learned function and regularize training to prevent change in these important parameters. We approach this regularization in two ways: (1), we constrain critical parameters from significant changes by associating more critical parameters with lower learning rates, thereby limiting alterations in those parameters; (2), important parameters are restricted from change by imposing a higher regularization weighting, causing parameters to revert to their states prior to the learning of subsequent tasks. We leverage a Bayesian Moment Propagation framework which learns network parameters concurrently with their associated uncertainties while allowing each parameter to contribute uncertainty to the network's predictive distribution, avoiding the pitfalls of existing sampling-based methods. The proposed approach is evaluated for common sequential benchmark datasets and compared to existing published approaches from the Continual Learning community. Ultimately, we show improved Continual Learning performance for Average Test Accuracy and Backward Transfer metrics compared to sampling-based methods and other non-uncertainty-based approaches. 

**Abstract (ZH)**: 在将深度神经网络（DNNs）微调到新数据时，DNNs容易覆盖掉完成特定任务所需网络参数，从而在先前学习的任务上损失性能。我们提出使用基于参数的不确定性来确定哪些参数对于网络学习的功能是相关的，并在训练中对其进行正则化以防止这些重要参数发生改变。我们采用两种方式进行正则化：（1）通过将更重要的参数与更低的学习率关联，限制这些参数发生显著变化；（2）通过施加更高的正则化权重来限制重要参数的改变，迫使这些参数在学习后续任务之前的状态中恢复。我们利用了一个贝叶斯矩传播框架，在学习网络参数的同时也学习其相关的不确定性，从而使每个参数能够向网络的预测分布贡献不确定性，从而避免现有基于采样的方法所面临的问题。所提出的方案在通用顺序基准数据集上进行评估，并与持续学习领域的现有发表方法进行比较。最终，我们展示了在平均测试准确率和反向迁移指标上相比于现有的基于采样的方法和其他无不确定性的方法得到了提升的持续学习性能。 

---
# Zero-shot and Few-shot Learning with Instruction-following LLMs for Claim Matching in Automated Fact-checking 

**Title (ZH)**: 基于指令遵循大语言模型的零样本和少样本学习在自动事实核查中的声明匹配研究 

**Authors**: Dina Pisarevskaya, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2501.10860)  

**Abstract**: The claim matching (CM) task can benefit an automated fact-checking pipeline by putting together claims that can be resolved with the same fact-check. In this work, we are the first to explore zero-shot and few-shot learning approaches to the task. We consider CM as a binary classification task and experiment with a set of instruction-following large language models (GPT-3.5-turbo, Gemini-1.5-flash, Mistral-7B-Instruct, and Llama-3-8B-Instruct), investigating prompt templates. We introduce a new CM dataset, ClaimMatch, which will be released upon acceptance. We put LLMs to the test in the CM task and find that it can be tackled by leveraging more mature yet similar tasks such as natural language inference or paraphrase detection. We also propose a pipeline for CM, which we evaluate on texts of different lengths. 

**Abstract (ZH)**: 声明匹配（CM）任务可以通过将可以用同一事实核查解决的声明组合起来，从而为自动化事实核查流水线带来益处。在本文中，我们首次探索了零样本和少样本学习方法在该任务中的应用。我们将CM视为二元分类任务，并使用一组指令跟随的大语言模型（GPT-3.5-turbo、Gemini-1.5-flash、Mistral-7B-Instruct和Llama-3-8B-Instruct）进行实验，并探讨了提示模板。我们引入了一个新的CM数据集，ClaimMatch，并将在论文被接受后发布。我们在CM任务中对LLMs进行了测试，发现可以通过利用更为成熟且类似的任务，如自然语言推理或同义替换检测来解决此任务。此外，我们还提出了一种CM管道，并在不同长度的文本上对其实效性进行了评估。 

---
# Reliable Text-to-SQL with Adaptive Abstention 

**Title (ZH)**: 可靠的自适应弃选文本到SQL转换 

**Authors**: Kaiwen Chen, Yueting Chen, Xiaohui Yu, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2501.10858)  

**Abstract**: Large language models (LLMs) have revolutionized natural language interfaces for databases, particularly in text-to-SQL conversion. However, current approaches often generate unreliable outputs when faced with ambiguity or insufficient context. We present Reliable Text-to-SQL (RTS), a novel framework that enhances query generation reliability by incorporating abstention and human-in-the-loop mechanisms. RTS focuses on the critical schema linking phase, which aims to identify the key database elements needed for generating SQL queries. It autonomously detects potential errors during the answer generation process and responds by either abstaining or engaging in user interaction. A vital component of RTS is the Branching Point Prediction (BPP) which utilizes statistical conformal techniques on the hidden layers of the LLM model for schema linking, providing probabilistic guarantees on schema linking accuracy. We validate our approach through comprehensive experiments on the BIRD benchmark, demonstrating significant improvements in robustness and reliability. Our findings highlight the potential of combining transparent-box LLMs with human-in-the-loop processes to create more robust natural language interfaces for databases. For the BIRD benchmark, our approach achieves near-perfect schema linking accuracy, autonomously involving a human when needed. Combined with query generation, we demonstrate that near-perfect schema linking and a small query generation model can almost match SOTA accuracy achieved with a model orders of magnitude larger than the one we use. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经从根本上改变了数据库的自然语言接口，特别是在文本到SQL转换方面。然而，当前的方法在面对歧义或缺乏上下文时经常生成不可靠的结果。我们提出了一种名为Reliable Text-to-SQL（RTS）的新颖框架，该框架通过引入退避和人工参与机制来增强查询生成的可靠性。RTS专注于关键的模式链接阶段，旨在识别生成SQL查询所需的关键数据库元素。它能够自动检测答案生成过程中的潜在错误，并相应地选择退避或与用户进行互动。RTS的一个关键组成部分是Branching Point Prediction（BPP），它利用统计约化技术在LLM模型的隐藏层上进行模式链接，提供模式链接准确性的概率保证。

我们通过在BIRD基准上的全面实验验证了我们的方法，显示出显著提升的稳健性和可靠性。我们的研究结果强调了将透明箱型LLM与人工参与过程相结合的潜力，以创建更稳健的数据库自然语言接口。对于BIRD基准，我们的方法实现了近乎完美的模式链接准确度，并在必要时自主邀请人类参与。结合查询生成，我们展示了近乎完美的模式链接和小型查询生成模型几乎可以达到比我们使用的模型大几个数量级的模型所达到的SOTA准确度。 

---
# Fake Advertisements Detection Using Automated Multimodal Learning: A Case Study for Vietnamese Real Estate Data 

**Title (ZH)**: 使用自动多模态学习进行虚假广告检测：越南房地产数据案例研究 

**Authors**: Duy Nguyen, Trung T. Nguyen, Cuong V. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.10848)  

**Abstract**: The popularity of e-commerce has given rise to fake advertisements that can expose users to financial and data risks while damaging the reputation of these e-commerce platforms. For these reasons, detecting and removing such fake advertisements are important for the success of e-commerce websites. In this paper, we propose FADAML, a novel end-to-end machine learning system to detect and filter out fake online advertisements. Our system combines techniques in multimodal machine learning and automated machine learning to achieve a high detection rate. As a case study, we apply FADAML to detect fake advertisements on popular Vietnamese real estate websites. Our experiments show that we can achieve 91.5% detection accuracy, which significantly outperforms three different state-of-the-art fake news detection systems. 

**Abstract (ZH)**: 电子商务的流行催生了虚假广告，这些广告不仅可能使用户面临财务和数据风险，还会损害这些电子商务平台的声誉。出于这些原因，检测和清除虚假广告对于电子商务网站的成功至关重要。本文提出了一种新颖的端到端机器学习系统FADAML，用于检测和过滤虚假在线广告。该系统结合了多模态机器学习技术和自动化机器学习方法，以实现较高的检测率。作为案例研究，我们将FADAML应用于检测越南房地产网站上的虚假广告。我们的实验结果显示，我们可以达到91.5%的检测准确率，这在很大程度上超越了三种不同的先进虚假新闻检测系统。 

---
# Practical and Ready-to-Use Methodology to Assess the re-identification Risk in Anonymized Datasets 

**Title (ZH)**: 用于评估匿名化数据集重识别风险的实用且可直接使用的方法学 

**Authors**: Louis-Philippe Sondeck, Maryline Laurent  

**Link**: [PDF](https://arxiv.org/pdf/2501.10841)  

**Abstract**: To prove that a dataset is sufficiently anonymized, many privacy policies suggest that a re-identification risk assessment be performed, but do not provide a precise methodology for doing so, leaving the industry alone with the problem. This paper proposes a practical and ready-to-use methodology for re-identification risk assessment, the originality of which is manifold: (1) it is the first to follow well-known risk analysis methods (e.g. EBIOS) that have been used in the cybersecurity field for years, which consider not only the ability to perform an attack, but also the impact such an attack can have on an individual; (2) it is the first to qualify attributes and values of attributes with e.g. degree of exposure, as known real-world attacks mainly target certain types of attributes and not others. 

**Abstract (ZH)**: 为了证明数据集已充分匿名化，许多隐私政策建议进行重新识别风险评估，但并未提供具体的方法，这使得行业独自面对这一问题。本文提出了一种实用且可立即使用的重新识别风险评估方法，其创新之处体现在以下几个方面：(1) 该方法是第一个遵循网络安全领域多年来使用的风险分析方法（例如 EBIOS），这些方法不仅考虑执行攻击的能力，还考虑此类攻击可能对个人造成的影响；(2) 该方法首次对属性及其值进行定性分析，比如根据暴露程度，而已知的实际攻击主要针对某些类型的属性而非其他类型。 

---
# BAP v2: An Enhanced Task Framework for Instruction Following in Minecraft Dialogues 

**Title (ZH)**: BAP v2：一种增强的任务框架，用于Minecraft对话中的指令跟随 

**Authors**: Prashant Jayannavar, Liliang Ren, Marisa Hudspeth, Charlotte Lambert, Ariel Cordes, Elizabeth Kaplan, Anjali Narayan-Chen, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2501.10836)  

**Abstract**: Interactive agents capable of understanding and executing instructions in the physical world have long been a central goal in AI research. The Minecraft Collaborative Building Task (MCBT) provides one such setting to work towards this goal (Narayan-Chen, Jayannavar, and Hockenmaier 2019). It is a two-player game in which an Architect (A) instructs a Builder (B) to construct a target structure in a simulated Blocks World Environment. We focus on the challenging Builder Action Prediction (BAP) subtask of predicting correct action sequences in a given multimodal game context with limited training data (Jayannavar, Narayan-Chen, and Hockenmaier 2020). We take a closer look at evaluation and data for the BAP task, discovering key challenges and making significant improvements on both fronts to propose BAP v2, an upgraded version of the task. This will allow future work to make more efficient and meaningful progress on it. It comprises of: (1) an enhanced evaluation benchmark that includes a cleaner test set and fairer, more insightful metrics, and (2) additional synthetic training data generated from novel Minecraft dialogue and target structure simulators emulating the MCBT. We show that the synthetic data can be used to train more performant and robust neural models even with relatively simple training methods. Looking ahead, such data could also be crucial for training more sophisticated, data-hungry deep transformer models and training/fine-tuning increasingly large LLMs. Although modeling is not the primary focus of this work, we also illustrate the impact of our data and training methodologies on a simple LLM- and transformer-based model, thus validating the robustness of our approach, and setting the stage for more advanced architectures and LLMs going forward. 

**Abstract (ZH)**: 能够在物理世界中理解并执行指令的交互式代理一直是AI研究的中心目标之一。Minecraft协作建造任务（MCBT）提供了一个这样的环境，旨在向这个目标迈进（Narayan-Chen, Jayannavar, and Hockenmaier 2019）。这是一个两人游戏，在模拟的Blocks World环境中，建筑师（A）会指导建造者（B）建造一个目标结构。我们专注于“建造者动作预测”（BAP）子任务，该任务涉及根据有限的训练数据预测给定多模态游戏上下文中的正确动作序列（Jayannavar, Narayan-Chen, and Hockenmaier 2020）。我们更详细地审视了BAP任务的评估和数据，发现了关键挑战，并在两个方面取得了显著改进，提出了BAP v2，即任务的升级版本。这将使未来的工作能够更高效且有意义地推进该任务。具体包括：（1）改进的评估基准，包括更清洁的测试集和更公平、更深入的指标；（2）从模拟MCBT的新MC游戏对话和目标结构生成的额外合成训练数据。我们展示了合成数据即使使用相对简单的训练方法也能用于训练性能更优、更稳健的神经网络模型。展望未来，这样的数据对于训练更复杂的、数据需求更大的深度变换模型以及训练/微调越来越大的语言模型也可能至关重要。虽然建模不是本工作的主要焦点，但我们也展示了我们的数据和训练方法对简单基于语言模型和变换器模型的影响，从而验证了我们方法的稳健性，并为未来更高级架构和语言模型奠定了基础。 

---
# Visual RAG: Expanding MLLM visual knowledge without fine-tuning 

**Title (ZH)**: 视觉RAG：在不微调的情况下扩展MLLM视觉知识 

**Authors**: Mirco Bonomo, Simone Bianco  

**Link**: [PDF](https://arxiv.org/pdf/2501.10834)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved notable performance in computer vision tasks that require reasoning across visual and textual modalities, yet their capabilities are limited to their pre-trained data, requiring extensive fine-tuning for updates. Recent researches have explored the use of In-Context Learning (ICL) to overcome these challenges by providing a set of demonstrating examples as context to augment MLLMs performance in several tasks, showing that many-shot ICL leads to substantial improvements compared to few-shot ICL. However, the reliance on numerous demonstrating examples and the limited MLLMs context windows presents significant obstacles. This paper aims to address these challenges by introducing a novel approach, Visual RAG, that synergically combines the MLLMs capability to learn from the context, with a retrieval mechanism. The crux of this approach is to ensure to augment the MLLM knowledge by selecting only the most relevant demonstrating examples for the query, pushing it to learn by analogy. In this way, relying on the new information provided dynamically during inference time, the resulting system is not limited to the knowledge extracted from the training data, but can be updated rapidly and easily without fine-tuning. Furthermore, this greatly reduces the computational costs for improving the model image classification performance, and augments the model knowledge to new visual domains and tasks it was not trained for. Extensive experiments on eight different datasets in the state of the art spanning several domains and image classification tasks show that the proposed Visual RAG, compared to the most recent state of the art (i.e., many-shot ICL), is able to obtain an accuracy that is very close or even higher (approx. +2% improvement on average) while using a much smaller set of demonstrating examples (approx. only 23% on average). 

**Abstract (ZH)**: 多模态大规模语言模型（Multimodal Large Language Models, MLLMs）在要求视觉和文本模态之间推理的计算机视觉任务中取得了显著性能，但其能力受限于预训练数据，需要大量的微调才能进行更新。近期研究探索了通过上下文学习（In-Context Learning, ICL）来应对这些挑战，通过提供一组示范示例作为上下文来增强MLLM在若干任务上的表现，研究表明大量示例的ICL相比于少量示例的ICL带来了显著的提升。然而，依赖大量示范示例和MLLM上下文窗口的限制带来了显著的障碍。本文旨在通过引入一种新颖的方法——Visual RAG，其结合了MLLM从上下文学习的能力与检索机制，来应对这些挑战。这种方法的核心在于通过选择与查询最相关的示范示例来增强MLLM的知识，促使它通过类比学习。这样，在推理时依赖于动态提供的新信息，该系统不仅限于训练数据中提取的知识，还可以快速且容易地进行更新，无需微调。此外，这种方法大大降低了提高模型图像分类性能的计算成本，并增强了模型在未训练的新视觉领域和任务中的知识。大量的实验表明，与最新的前沿技术（即大量示例的ICL）相比，提出的Visual RAG在使用更少的示范示例（约平均23%）的情况下，能够获得非常接近或甚至更高的准确率（约平均2%的改善）。实验涵盖了来自不同领域的八个最先进的数据集，涉及多个图像分类任务。 

---
# Addressing Multilabel Imbalance with an Efficiency-Focused Approach Using Diffusion Model-Generated Synthetic Samples 

**Title (ZH)**: 使用扩散模型生成的合成样本以效率为导向解决多标签不平衡问题 

**Authors**: Francisco Charte, Miguel Ángel Dávila, María Dolores Pérez-Godoy, María José del Jesus  

**Link**: [PDF](https://arxiv.org/pdf/2501.10822)  

**Abstract**: Predictive models trained on imbalanced data tend to produce biased results. This problem is exacerbated when there is not just one output label, but a set of them. This is the case for multilabel learning (MLL) algorithms used to classify patterns, rank labels, or learn the distribution of outputs. Many solutions have been proposed in the literature. The one that can be applied universally, independent of the algorithm used to build the model, is data resampling. The generation of new instances associated with minority labels, so that empty areas of the feature space are filled, helps to improve the obtained models. The quality of these new instances depends on the algorithm used to generate them. In this paper, a diffusion model tailored to produce new instances for MLL data, called MLDM (\textit{MultiLabel Diffusion Model}), is proposed. Diffusion models have been mainly used to generate artificial images and videos. Our proposed MLDM is based on this type of models. The experiments conducted compare MLDM with several other MLL resampling algorithms. The results show that MLDM is competitive while it improves efficiency. 

**Abstract (ZH)**: 基于不平衡数据训练的预测模型往往会生成有偏的结果。当不只有一个输出标签，而是一组标签时，这一问题会进一步加剧。这种情况适用于用于分类模式、排序标签或学习输出分布的多标签学习（MLL）算法。文献中提出了许多解决方案。其中一种可以普遍应用于独立于构建模型所使用的算法的方案是数据重采样。生成与少数类标签相关的新型实例，以填充特征空间中的空洞区域，有助于提高所获得模型的性能。这些新型实例的质量取决于用于生成它们的算法。在本文中，提出了一种针对MLL数据生成新型实例的自适应模型，称为MLDM（多标签扩散模型）。扩散模型主要用于生成人工图像和视频。我们提出的MLDM基于此类模型。实验对比了MLDM与多种其他MLL重采样算法。结果表明，MLDM在具有竞争力的同时，提高了效率。 

---
# No More Sliding Window: Efficient 3D Medical Image Segmentation with Differentiable Top-k Patch Sampling 

**Title (ZH)**: 不再使用滑动窗口：基于可微分 Top-k 贴片采样的高效三维医学图像分割 

**Authors**: Young Seok Jeon, Hongfei Yang, Huazhu Fu, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.10814)  

**Abstract**: 3D models are favored over 2D for 3D medical image segmentation tasks due to their ability to leverage inter-slice relationship, yielding higher segmentation accuracy. However, 3D models demand significantly more GPU memory with increased model size and intermediate tensors. A common solution is to use patch-based training and make whole-volume predictions with sliding window (SW) inference. SW inference reduces memory usage but is slower due to equal resource allocation across patches and less accurate as it overlooks global features beyond patches.
We propose NMSW-Net (No-More-Sliding-Window-Net), a novel framework that enhances efficiency and accuracy of any given 3D segmentation model by eliminating SW inference and incorporating global predictions when necessary. NMSW-Net incorporates a differentiable Top-k module to sample only the relevant patches that enhance segmentation accuracy, thereby minimizing redundant computations. Additionally, it learns to leverage coarse global predictions when patch prediction alone is insufficient. NMSW-Net is model-agnostic, making it compatible with any 3D segmentation model that previously relied on SW inference.
Evaluated across 3 tasks with 3 segmentation backbones, NMSW-Net achieves competitive or sometimes superior accuracy compared to SW, while reducing computational complexity by 90% (87.5 to 7.95 TFLOPS), delivering 4x faster inference on the H100 GPU (19.0 to 4.3 sec), and 7x faster inference on the Intel Xeon Gold CPU (1710 to 230 seconds). 

**Abstract (ZH)**: 由于3D模型能够利用层间关系，从而提高分割精度，因此3D模型在医学图像分割任务中比2D模型更受欢迎。然而，3D模型需要显著更多的GPU内存，尤其是在模型增大和中间张量增多的情况下。一种常见的解决方案是使用基于切片（patch-based）的训练方法，并通过滑动窗口（sliding window, SW）进行整体体素预测。SW方法虽然能降低内存使用，但由于在不同切片上平均分配资源，导致速度较慢且容易忽略切片之外的全局特征。

我们提出了一种新颖的框架NMSW-Net（No-More-Sliding-Window-Net），该框架能通过消除SW推理并根据需要引入全局预测来提高任何给定3D分割模型的效率和准确性。NMSW-Net整合了一个可微分的Top-k模块，仅选择能提升分割精度的相关切片进行计算，从而减少冗余计算。此外，它能够学习在单独的切片预测不足以时利用粗略的全局预测。NMSW-Net是模型无关的，使其能够与之前依赖于SW推理的任何3D分割模型兼容。

NMSW-Net在三个任务和三个分割主干网络上进行了评估，与SW方法相比，其分割精度具有竞争性甚至在某些情况下更优，同时减少了90%的计算复杂度（从87.5 TFLOPS降至7.95 TFLOPS），在H100 GPU上的推理速度提高了4倍（从19.0秒缩短至4.3秒），在Intel Xeon Gold CPU上的推理速度提高了7倍（从1710秒缩短至230秒）。 

---
# Graph Coloring to Reduce Computation Time in Prioritized Planning 

**Title (ZH)**: 优先规划中通过图着色减少计算时间的方法 

**Authors**: Patrick Scheffe, Julius Kahle, Bassam Alrifaee  

**Link**: [PDF](https://arxiv.org/pdf/2501.10812)  

**Abstract**: Distributing computations among agents in large networks reduces computational effort in multi-agent path finding (MAPF). One distribution strategy is prioritized planning (PP). In PP, we couple and prioritize interacting agents to achieve a desired behavior across all agents in the network. We characterize the interaction with a directed acyclic graph (DAG). The computation time for solving MAPF problem using PP is mainly determined through the longest path in this DAG. The longest path depends on the fixed undirected coupling graph and the variable prioritization. The approaches from literature to prioritize agents are numerous and pursue various goals. This article presents an approach for prioritization in PP to reduce the longest path length in the coupling DAG and thus the computation time for MAPF using PP. We prove that this problem can be mapped to a graph-coloring problem, in which the number of colors required corresponds to the longest path length in the coupling DAG. We propose a decentralized graph-coloring algorithm to determine priorities for the agents. We evaluate the approach by applying it to multi-agent motion planning (MAMP) for connected and automated vehicles (CAVs) on roads using, a variant of MAPF. 

**Abstract (ZH)**: 在大型网络中将计算任务分配给代理可以减少多代理路径查找（MAPF）中的计算努力。一种分配策略是优先级规划（Prioritized Planning, PP）。在PP中，我们通过关联和优先级排序来实现网络中所有代理的期望行为。我们用有向无环图（DAG）来表征这种交互。使用PP解决MAPF问题的计算时间主要取决于此DAG中的最长路径长度。最长路径长度取决于固定的无向耦合图和变化的优先级设置。文献中提出了多种代理优先级排序的方法，且追求不同的目标。本文提出了一种在PP中进行优先级排序的方法，旨在减少耦合DAG中的最长路径长度，从而降低使用PP解决MAPF问题的计算时间。我们证明了这个优化问题可以映射为一个图着色问题，在该问题中，所需的颜色数对应于耦合DAG中的最长路径长度。我们提出了一种分布式图着色算法，用于确定代理的优先级。我们通过将其应用于在道路上的连接和自动化车辆（CAVs）的多代理运动规划（MAMP）来评估这一方法。 

---
# Efficient Auto-Labeling of Large-Scale Poultry Datasets (ALPD) Using Semi-Supervised Models, Active Learning, and Prompt-then-Detect Approach 

**Title (ZH)**: 使用半监督模型、主动学习和Prompt-then-Detect方法的大规模家禽数据集高效自动标注（ALPD）方法 

**Authors**: Ramesh Bahadur Bist, Lilong Chai, Shawna Weimer, Hannah Atungulua, Chantel Pennicott, Xiao Yang, Sachin Subedi, Chaitanya Pallerla, Yang Tian, Dongyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10809)  

**Abstract**: The rapid growth of AI in poultry farming has highlighted the challenge of efficiently labeling large, diverse datasets. Manual annotation is time-consuming, making it impractical for modern systems that continuously generate data. This study explores semi-supervised auto-labeling methods, integrating active learning, and prompt-then-detect paradigm to develop an efficient framework for auto-labeling of large poultry datasets aimed at advancing AI-driven behavior and health monitoring. Viideo data were collected from broilers and laying hens housed at the University of Arkansas and the University of Georgia. The collected videos were converted into images, pre-processed, augmented, and labeled. Various machine learning models, including zero-shot models like Grounding DINO, YOLO-World, and CLIP, and supervised models like YOLO and Faster-RCNN, were utilized for broilers, hens, and behavior detection. The results showed that YOLOv8s-World and YOLOv9s performed better when compared performance metrics for broiler and hen detection under supervised learning, while among the semi-supervised model, YOLOv8s-ALPD achieved the highest precision (96.1%) and recall (99.0%) with an RMSE of 1.9. The hybrid YOLO-World model, incorporating the optimal YOLOv8s backbone, demonstrated the highest overall performance. It achieved a precision of 99.2%, recall of 99.4%, and an F1 score of 98.7% for breed detection, alongside a precision of 88.4%, recall of 83.1%, and an F1 score of 84.5% for individual behavior detection. Additionally, semi-supervised models showed significant improvements in behavior detection, achieving up to 31% improvement in precision and 16% in F1-score. The semi-supervised models with minimal active learning reduced annotation time by over 80% compared to full manual labeling. Moreover, integrating zero-shot models enhanced detection and behavior identification. 

**Abstract (ZH)**: 禽类养殖中AI的快速发展突显了高效标注大量多样化数据集的挑战。手工标注耗时，对于现代系统不断生成的数据而言，这变得不切实际。本研究探索了半监督自动标注方法，结合主动学习和提示-检测范式，以开发适用于大型禽类数据集自动标注的高效框架，旨在促进基于AI的行为和健康监测。研究从阿肯色大学和佐治亚大学饲养的肉鸡和蛋鸡收集了视频数据。这些视频被转换为图像，预处理、增强并标注。利用了多种机器学习模型，包括零样本模型如Grounding DINO、YOLO-World和CLIP，以及监督模型如YOLO和Faster-RCNN，用于肉鸡、蛋鸡和行为检测。结果显示，在监督学习条件下，YOLOv8s-World和YOLOv9s在肉鸡和蛋鸡检测方面表现更优。在半监督模型中，YOLOv8s-ALPD的精度达到96.1%，召回率为99.0%，且均方根误差为1.9，表现出最佳性能。结合最优YOLOv8s骨干的混合YOLO-World模型整体性能最佳，其在品种检测方面的精度为99.2%，召回率为99.4%，F1分数为98.7%，个人行为检测方面，精度为88.4%，召回率为83.1%，F1分数为84.5%。此外，半监督模型在行为检测方面表现出显著改进，精度提高最高可达31%，F1分数提高16%。与全手工标注相比，结合最少主动学习的半监督模型将标注时间减少超过80%。此外，引入零样本模型提高了检测和行为识别的性能。 

---
# Step-KTO: Optimizing Mathematical Reasoning through Stepwise Binary Feedback 

**Title (ZH)**: 步进KTO：通过逐步二元反馈优化数学推理 

**Authors**: Yen-Ting Lin, Di Jin, Tengyu Xu, Tianhao Wu, Sainbayar Sukhbaatar, Chen Zhu, Yun He, Yun-Nung Chen, Jason Weston, Yuandong Tian, Arash Rahnama, Sinong Wang, Hao Ma, Han Fang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10799)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable success in mathematical reasoning. Despite progress in methods like chain-of-thought prompting and self-consistency sampling, these advances often focus on final correctness without ensuring that the underlying reasoning process is coherent and reliable. This paper introduces Step-KTO, a training framework that combines process-level and outcome-level binary feedback to guide LLMs toward more trustworthy reasoning trajectories. By providing binary evaluations for both the intermediate reasoning steps and the final answer, Step-KTO encourages the model to adhere to logical progressions rather than relying on superficial shortcuts. Our experiments on challenging mathematical benchmarks show that Step-KTO significantly improves both final answer accuracy and the quality of intermediate reasoning steps. For example, on the MATH-500 dataset, Step-KTO achieves a notable improvement in Pass@1 accuracy over strong baselines. These results highlight the promise of integrating stepwise process feedback into LLM training, paving the way toward more interpretable and dependable reasoning capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）最近在数学推理方面展示了明显的成功。尽管方法如链式思考提示和自我一致性采样方面取得了进展，这些进步通常关注最终的正确性，而不确保推理过程本身是连贯和可靠的。本文提出了Step-KTO，一种结合过程级和结果级二元反馈的训练框架，以引导LLMs向更为可信的推理轨迹发展。通过为中间推理步骤和最终答案提供二元评估，Step-KTO 鼓励模型遵循逻辑演进，而不是依赖表面的捷径。我们针对具有挑战性的数学基准数据集的实验表明，Step-KTO 显著提高了最终答案的准确性和中间推理步骤的质量。例如，在MATH-500数据集上，Step-KTO 在Pass@1准确性上显著优于强大的基准模型。这些结果表明整合逐步过程反馈到LLM训练中的潜力，为更可解释和可靠推理能力的发展铺平了道路。 

---
# Simultaneous Computation with Multiple Prioritizations in Multi-Agent Motion Planning 

**Title (ZH)**: 多智能体运动规划中的多重优先级同时计算 

**Authors**: Patrick Scheffe, Julius Kahle, Bassam Alrifaee  

**Link**: [PDF](https://arxiv.org/pdf/2501.10781)  

**Abstract**: Multi-agent path finding (MAPF) in large networks is computationally challenging. An approach for MAPF is prioritized planning (PP), in which agents plan sequentially according to their priority. Albeit a computationally efficient approach for MAPF, the solution quality strongly depends on the prioritization. Most prioritizations rely either on heuristics, which do not generalize well, or iterate to find adequate priorities, which costs computational effort. In this work, we show how agents can compute with multiple prioritizations simultaneously. Our approach is general as it does not rely on domain-specific knowledge. The context of this work is multi-agent motion planning (MAMP) with a receding horizon subject to computation time constraints. MAMP considers the system dynamics in more detail compared to MAPF. In numerical experiments on MAMP, we demonstrate that our approach to prioritization comes close to optimal prioritization and outperforms state-of-the-art methods with only a minor increase in computation time. We show real-time capability in an experiment on a road network with ten vehicles in our Cyber-Physical Mobility Lab. 

**Abstract (ZH)**: 在大规模网络中，多智能体路径寻找（Multi-agent Path Finding, MAPF）具有计算上的挑战性。一种MAPF的方法是优先级规划（Prioritized Planning, PP），在该方法中，智能体根据其优先级顺序规划路径。尽管PP方法在计算上较为高效，但其解的质量很大程度上依赖于优先级的选择。大多数优先级选择要么依赖于启发式方法，这些方法在泛化方面表现不佳，要么需要迭代以找到合适的优先级，这会增加计算成本。在本研究中，我们展示了如何让智能体同时使用多种优先级进行计算。我们的方法是通用的，因为它不依赖于特定领域的知识。本研究的背景是在计算时间受限条件下考虑系统动力学的多智能体运动规划（Multi-agent Motion Planning, MAMP）。相较于MAPF，MAMP更详细地考虑了系统动力学。在针对MAMP的数值实验中，我们证明了我们的优先级规划方法接近最优优先级规划，并且仅略微增加了计算时间便优于现有的先进方法。我们还在我们的人机物理移动实验室（Cyber-Physical Mobility Lab）中进行了一项十辆车的道路网络实验，展示了实时操作能力。 

---
# MedFILIP: Medical Fine-grained Language-Image Pre-training 

**Title (ZH)**: MedFILIP：医学细粒度语图预训练 

**Authors**: Xinjie Liang, Xiangyu Li, Fanding Li, Jie Jiang, Qing Dong, Wei Wang, Kuanquan Wang, Suyu Dong, Gongning Luo, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.10775)  

**Abstract**: Medical vision-language pretraining (VLP) that leverages naturally-paired medical image-report data is crucial for medical image analysis. However, existing methods struggle to accurately characterize associations between images and diseases, leading to inaccurate or incomplete diagnostic results. In this work, we propose MedFILIP, a fine-grained VLP model, introduces medical image-specific knowledge through contrastive learning, specifically: 1) An information extractor based on a large language model is proposed to decouple comprehensive disease details from reports, which excels in extracting disease deals through flexible prompt engineering, thereby effectively reducing text complexity while retaining rich information at a tiny cost. 2) A knowledge injector is proposed to construct relationships between categories and visual attributes, which help the model to make judgments based on image features, and fosters knowledge extrapolation to unfamiliar disease categories. 3) A semantic similarity matrix based on fine-grained annotations is proposed, providing smoother, information-richer labels, thus allowing fine-grained image-text alignment. 4) We validate MedFILIP on numerous datasets, e.g., RSNA-Pneumonia, NIH ChestX-ray14, VinBigData, and COVID-19. For single-label, multi-label, and fine-grained classification, our model achieves state-of-the-art performance, the classification accuracy has increased by a maximum of 6.69\%. The code is available in this https URL. 

**Abstract (ZH)**: 医学视觉-语言预训练（VLP）利用自然配对的医学图像-报告数据，对于医学图像分析至关重要。然而，现有方法在准确刻画图像与疾病之间的关联方面存在困难，导致诊断结果不准确或不完整。在本项工作中，我们提出了MedFILIP，这是一种细粒度的VLP模型，通过对比学习引入医学图像特定的知识，具体包括以下几点：

1. 提出了基于大规模语言模型的信息抽取器，用于从报告中分离出全面的疾病细节，通过灵活性强的提示工程提取疾病细节，从而有效降低文本复杂度，同时保留丰富的信息，仅付出微小的代价。

2. 提出了知识注入器，用于构建类别与视觉属性之间的关系，这有助于模型基于图像特征进行判断，并促进知识向不熟悉疾病类别的外推。

3. 提出了基于细粒度注解的语义相似性矩阵，提供了更平滑、信息更丰富的标签，从而实现细粒度的图像-文本对齐。

4. 我们在多个数据集上验证了MedFILIP，例如RSNA肺炎、NIH ChestX-ray14、VinBigData和COVID-19。对于单标签、多标签及细粒度分类，我们的模型实现了最先进的性能，分类准确率最高提升了6.69%。代码可以在以下链接中获取：[请提供链接] 

---
# Enhancing Diagnostic in 3D COVID-19 Pneumonia CT-scans through Explainable Uncertainty Bayesian Quantification 

**Title (ZH)**: 通过可解释的不确定性贝叶斯量化增强三维新冠肺炎CT扫描的诊断性能 

**Authors**: Juan Manuel Liscano Fierro, Hector J. Hortua  

**Link**: [PDF](https://arxiv.org/pdf/2501.10770)  

**Abstract**: Accurately classifying COVID-19 pneumonia in 3D CT scans remains a significant challenge in the field of medical image analysis. Although deterministic neural networks have shown promising results in this area, they provide only point estimates outputs yielding poor diagnostic in clinical decision-making. In this paper, we explore the use of Bayesian neural networks for classifying COVID-19 pneumonia in 3D CT scans providing uncertainties in their predictions. We compare deterministic networks and their Bayesian counterpart, enhancing the decision-making accuracy under uncertainty information. Remarkably, our findings reveal that lightweight architectures achieve the highest accuracy of 96\% after developing extensive hyperparameter tuning. Furthermore, the Bayesian counterpart of these architectures via Multiplied Normalizing Flow technique kept a similar performance along with calibrated uncertainty estimates. Finally, we have developed a 3D-visualization approach to explain the neural network outcomes based on SHAP values. We conclude that explainability along with uncertainty quantification will offer better clinical decisions in medical image analysis, contributing to ongoing efforts for improving the diagnosis and treatment of COVID-19 pneumonia. 

**Abstract (ZH)**: 在医学图像分析领域，准确地在3D CT扫描中分类新冠肺炎肺炎仍是一个重大挑战。尽管确定性神经网络在这个领域中显示出了令人鼓舞的结果，但它们只能提供点估计输出，这在临床决策中效果不佳。在本文中，我们探索了使用贝叶斯神经网络来对3D CT扫描中的新冠肺炎肺炎进行分类，从而在预测中提供不确定性信息。我们将确定性网络与其贝叶斯对应模型进行对比，以提高在不确定性信息下的决策准确度。值得注意的是，我们的研究表明，在进行了广泛的超参数调整后，轻量级架构的准确率达到了96%。此外，通过使用乘法归一化流技术的贝叶斯对应模型在保持类似性能的同时，还提供了校准的不确定性估计。最后，我们开发了一种基于SHAP值的3D可视化方法，以解释神经网络的输出结果。我们得出结论，通过解释性和不确定性量化，可以更好地在医学图像分析中做出临床决策，这有助于不断改进新冠肺炎肺炎的诊断和治疗。 

---
# Semi-supervised Semantic Segmentation for Remote Sensing Images via Multi-scale Uncertainty Consistency and Cross-Teacher-Student Attention 

**Title (ZH)**: 基于多尺度不确定性一致性和跨教师-学生注意力的半监督语义分割方法：遥感图像应用 

**Authors**: Shanwen Wang, Changrui Chen, Xin Sun, Danfeng Hong, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2501.10736)  

**Abstract**: Semi-supervised learning offers an appealing solution for remote sensing (RS) image segmentation to relieve the burden of labor-intensive pixel-level labeling. However, RS images pose unique challenges, including rich multi-scale features and high inter-class similarity. To address these problems, this paper proposes a novel semi-supervised Multi-Scale Uncertainty and Cross-Teacher-Student Attention (MUCA) model for RS image semantic segmentation tasks. Specifically, MUCA constrains the consistency among feature maps at different layers of the network by introducing a multi-scale uncertainty consistency regularization. It improves the multi-scale learning capability of semi-supervised algorithms on unlabeled data. Additionally, MUCA utilizes a Cross-Teacher-Student attention mechanism to guide the student network, guiding the student network to construct more discriminative feature representations through complementary features from the teacher network. This design effectively integrates weak and strong augmentations (WA and SA) to further boost segmentation performance. To verify the effectiveness of our model, we conduct extensive experiments on ISPRS-Potsdam and LoveDA datasets. The experimental results show the superiority of our method over state-of-the-art semi-supervised methods. Notably, our model excels in distinguishing highly similar objects, showcasing its potential for advancing semi-supervised RS image segmentation tasks. 

**Abstract (ZH)**: 半监督学习为遥感（RS）图像分割提供了一种吸引人的解决方案，可以减轻劳动密集型像素级标注的负担。然而，RS图像面临着独特的挑战，包括丰富的多尺度特征和高类间相似性。为了解决这些问题，本文提出了一种新颖的半监督多尺度不确定性与跨教师-学生注意力（MUCA）模型，用于RS图像语义分割任务。具体而言，MUCA通过引入多尺度不确定性一致性正则化，约束网络不同层特征图之间的一致性，从而提高半监督算法在无标注数据上的多尺度学习能力。此外，MUCA利用跨教师-学生注意力机制来引导学生网络，通过教师网络提供的互补特征引导学生网络构建更具鉴别性的特征表示。此设计有效整合了弱增强（WA）和强增强（SA），进一步提升了分割性能。为了验证我们模型的有效性，我们在ISPRS-波茨坦和LoveDA数据集上进行了广泛的实验。实验结果表明，我们的方法在与现有最新半监督方法的比较中表现出显著优势。值得注意的是，我们的模型在区分高度相似对象方面表现出色，展示了其在推进半监督RS图像分割任务方面的潜力。 

---
# GEC-RAG: Improving Generative Error Correction via Retrieval-Augmented Generation for Automatic Speech Recognition Systems 

**Title (ZH)**: GEC-RAG：通过检索增强生成提高生成式错误修正技术在自动语音识别系统中的性能 

**Authors**: Amin Robatian, Mohammad Hajipour, Mohammad Reza Peyghan, Fatemeh Rajabi, Sajjad Amini, Shahrokh Ghaemmaghami, Iman Gholampour  

**Link**: [PDF](https://arxiv.org/pdf/2501.10734)  

**Abstract**: Automatic Speech Recognition (ASR) systems have demonstrated remarkable performance across various applications. However, limited data and the unique language features of specific domains, such as low-resource languages, significantly degrade their performance and lead to higher Word Error Rates (WER). In this study, we propose Generative Error Correction via Retrieval-Augmented Generation (GEC-RAG), a novel approach designed to improve ASR accuracy for low-resource domains, like Persian. Our approach treats the ASR system as a black-box, a common practice in cloud-based services, and proposes a Retrieval-Augmented Generation (RAG) approach within the In-Context Learning (ICL) scheme to enhance the quality of ASR predictions. By constructing a knowledge base that pairs ASR predictions (1-best and 5-best hypotheses) with their corresponding ground truths, GEC-RAG retrieves lexically similar examples to the ASR transcription using the Term Frequency-Inverse Document Frequency (TF-IDF) measure. This process provides relevant error patterns of the system alongside the ASR transcription to the Generative Large Language Model (LLM), enabling targeted corrections. Our results demonstrate that this strategy significantly reduces WER in Persian and highlights a potential for domain adaptation and low-resource scenarios. This research underscores the effectiveness of using RAG in enhancing ASR systems without requiring direct model modification or fine-tuning, making it adaptable to any domain by simply updating the transcription knowledge base with domain-specific data. 

**Abstract (ZH)**: 自动语音识别（ASR）系统已在多种应用中展示了卓越的性能。然而，有限的数据以及特定领域如低资源语言的独特语言特征，显著降低了其性能，导致更高的词错误率（WER）。在本研究中，我们提出了生成纠错超检索增强生成（GEC-RAG）方法，这是一种旨在提高低资源领域ASR准确性的新方法，例如波斯语。我们的方法将ASR系统视为一个黑盒模型，这是云服务中的常见做法，并在此基础上提出了一种在背景学习（In-Context Learning, ICL）框架内的检索增强生成（RAG）方法，以提高ASR预测质量。通过构建一个知识库，该知识库将ASR预测（1-best和5-best假设）与其对应的地面真实值配对，GEC-RAG使用.Term频率-逆文档频率（TF-IDF）测量检索出与ASR转录词义相似的示例。这个过程向生成大规模语言模型（Generative Large Language Model, GLLM）提供了与ASR转录相关的系统错误模式，使其能够进行目标纠错。我们的结果显示，这种方法显著减少了波斯语的WER，并指出了领域适应性和低资源场景的潜在价值。这项研究强调了在无需直接修改或微调模型的情况下，使用RAG增强ASR系统的有效性，并使其可以通过更新与特定领域相关的转录知识库来适应任何领域。 

---
# In the Picture: Medical Imaging Datasets, Artifacts, and their Living Review 

**Title (ZH)**: 《图像之中：医疗影像数据集、伪影及其生命周期综述》

这个标题翻译成中文既保留了原文的意思，又符合学术规范的表达方式。如果有更具体的上下文或更细致的需求，请告知我。 

**Authors**: Amelia Jiménez-Sánchez, Natalia-Rozalia Avlona, Sarah de Boer, Víctor M. Campello, Aasa Feragen, Enzo Ferrante, Melanie Ganz, Judy Wawira Gichoya, Camila González, Steff Groefsema, Alessa Hering, Adam Hulman, Leo Joskowicz, Dovile Juodelyte, Melih Kandemir, Thijs Kooi, Jorge del Pozo Lérida, Livie Yumeng Li, Andre Pacheco, Tim Rädsch, Mauricio Reyes, Théo Sourget, Bram van Ginneken, David Wen, Nina Weng, Jack Junchi Xu, Hubert Dariusz Zając, Maria A. Zuluaga, Veronika Cheplygina  

**Link**: [PDF](https://arxiv.org/pdf/2501.10727)  

**Abstract**: Datasets play a critical role in medical imaging research, yet issues such as label quality, shortcuts, and metadata are often overlooked. This lack of attention may harm the generalizability of algorithms and, consequently, negatively impact patient outcomes. While existing medical imaging literature reviews mostly focus on machine learning (ML) methods, with only a few focusing on datasets for specific applications, these reviews remain static -- they are published once and not updated thereafter. This fails to account for emerging evidence, such as biases, shortcuts, and additional annotations that other researchers may contribute after the dataset is published. We refer to these newly discovered findings of datasets as research artifacts. To address this gap, we propose a living review that continuously tracks public datasets and their associated research artifacts across multiple medical imaging applications. Our approach includes a framework for the living review to monitor data documentation artifacts, and an SQL database to visualize the citation relationships between research artifact and dataset. Lastly, we discuss key considerations for creating medical imaging datasets, review best practices for data annotation, discuss the significance of shortcuts and demographic diversity, and emphasize the importance of managing datasets throughout their entire lifecycle. Our demo is publicly available at this http URL. 

**Abstract (ZH)**: 数据集在医学影像研究中发挥着关键作用，然而标签质量、捷径和元数据等问题常常被忽略。忽视这些问题可能损害算法的泛化能力，并最终对患者结果产生负面影响。虽然现有医学影像文献综述主要集中在机器学习方法上，只有少数综述关注特定应用的数据集，这些综述往往是静态的——它们只在最初发表后不再更新。这无法考虑到其他研究人员在数据集发布后可能贡献的新兴证据，如偏差、捷径和其他附加注释。我们将这些新发现的数据集研究成果称为研究产品。

为解决这一问题，我们建议采用一种持续更新的研究综述，它能够跨多个医学影像应用场景跟踪公共数据集及其相关的研究产品。我们的方法包括一个持续更新综述的框架，用于监控数据文档产品，并使用SQL数据库来可视化研究产品与数据集之间的引文关系。最后，我们讨论了创建医学影像数据集的关键考虑因素，回顾了数据注释的最佳实践，探讨了捷径和人口统计多样性的重要性，并强调了在整个数据集生命周期中管理数据的重要性。我们的演示可在该网址查看：[在此处添加网址]。 

---
# How Should I Build A Benchmark? 

**Title (ZH)**: 如何构建基准？ 

**Authors**: Jialun Cao, Yuk-Kit Chan, Zixuan Ling, Wenxuan Wang, Shuqing Li, Mingwei Liu, Chaozheng Wang, Boxi Yu, Pinjia He, Shuai Wang, Zibin Zheng, Michael R. Lyu, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2501.10711)  

**Abstract**: Various benchmarks have been proposed to assess the performance of large language models (LLMs) in different coding scenarios. We refer to them as code-related benchmarks. However, there are no systematic guidelines by which such a benchmark should be developed to ensure its quality, reliability, and reproducibility. We propose How2Bench, which is comprised of a 55- 55-criteria checklist as a set of guidelines to govern the development of code-related benchmarks comprehensively. Using HOW2BENCH, we profiled 274 benchmarks released within the past decade and found concerning issues. Nearly 70% of the benchmarks did not take measures for data quality assurance; over 10% did not even open source or only partially open source. Many highly cited benchmarks have loopholes, including duplicated samples, incorrect reference codes/tests/prompts, and unremoved sensitive/confidential information. Finally, we conducted a human study involving 49 participants, which revealed significant gaps in awareness of the importance of data quality, reproducibility, and transparency. 

**Abstract (ZH)**: 各种基准已被提出，用以评估大型语言模型（LLMs）在不同编程场景中的性能。我们将这些基准称为代码相关基准。然而，目前尚缺乏系统性的指南，以确保这些基准的质量、可靠性和可重复性。我们提出了How2Bench，这是一种由55项准则组成的检查表，作为一套指南，用于全面规范代码相关基准的开发过程。利用HOW2BENCH，我们对过去十年中发布的274个基准进行了分析，并发现了若干问题。近70%的基准未采取数据质量保障措施；超过10%的基准未开源或仅部分开源。许多被高度引用的基准中存在漏洞，包括重复样本、错误的参考代码/测试/提示，以及未移除的敏感/保密信息。最后，我们进行了一项涉及49名参与者的调研，揭示了数据质量、可重复性和透明度意识方面存在的重大缺口。 

---
# Revisiting Ensemble Methods for Stock Trading and Crypto Trading Tasks at ACM ICAIF FinRL Contest 2023-2024 

**Title (ZH)**: 重访 ACM ICAIF FinRL 赛事 2023-2024 年度股票交易和加密货币交易任务中的集成方法 

**Authors**: Nikolaus Holzer, Keyi Wang, Kairong Xiao, Xiao-Yang Liu Yanglet  

**Link**: [PDF](https://arxiv.org/pdf/2501.10709)  

**Abstract**: Reinforcement learning has demonstrated great potential for performing financial tasks. However, it faces two major challenges: policy instability and sampling bottlenecks. In this paper, we revisit ensemble methods with massively parallel simulations on graphics processing units (GPUs), significantly enhancing the computational efficiency and robustness of trained models in volatile financial markets. Our approach leverages the parallel processing capability of GPUs to significantly improve the sampling speed for training ensemble models. The ensemble models combine the strengths of component agents to improve the robustness of financial decision-making strategies. We conduct experiments in both stock and cryptocurrency trading tasks to evaluate the effectiveness of our approach. Massively parallel simulation on a single GPU improves the sampling speed by up to $1,746\times$ using $2,048$ parallel environments compared to a single environment. The ensemble models have high cumulative returns and outperform some individual agents, reducing maximum drawdown by up to $4.17\%$ and improving the Sharpe ratio by up to $0.21$.
This paper describes trading tasks at ACM ICAIF FinRL Contests in 2023 and 2024. 

**Abstract (ZH)**: 强化学习在执行金融任务方面展现了巨大的潜力。然而，它面临着两大挑战：策略不稳定性和采样瓶颈。本文重新审视了大规模并行模拟方法在图形处理单元（GPUs）上的应用，极大地提升了训练模型在波动性金融市场中的计算效率和稳健性。我们的方法利用了GPU的并行处理能力，显著提高了训练ensemble模型的采样速度。ensemble模型结合了组成agent的优势，以提高金融决策策略的稳健性。我们在股票和加密货币交易任务中进行了实验，以评估该方法的有效性。在单个GPU上进行大规模并行模拟，与单个环境相比，使用2048个并行环境可将采样速度提高高达1,746倍。ensemble模型具有高累积回报，并在某些情况下优于单独的agent，最大回撤最多减少4.17%，同时提高夏普比率最多0.21。

本文描述了2023年和2024年ACM ICAIF FinRL竞赛中的交易任务。 

---
# Algorithmic Derivation of Human Spatial Navigation Indices From Eye Movement Data 

**Title (ZH)**: 从眼球运动数据中算法推导人类空间导航指数 

**Authors**: Sobhan Teymouri, Fatemeh Alizadehziri, Mobina Zibandehpoor, Mehdi Delrobaei  

**Link**: [PDF](https://arxiv.org/pdf/2501.10696)  

**Abstract**: Spatial navigation is a complex cognitive function involving sensory inputs, such as visual, auditory, and proprioceptive information, to understand and move within space. This ability allows humans to create mental maps, navigate through environments, and process directional cues, crucial for exploring new places and finding one's way in unfamiliar surroundings. This study takes an algorithmic approach to extract indices relevant to human spatial navigation using eye movement data. Leveraging electrooculography signals, we analyzed statistical features and applied feature engineering techniques to study eye movements during navigation tasks. The proposed work combines signal processing and machine learning approaches to develop indices for navigation and orientation, spatial anxiety, landmark recognition, path survey, and path route. The analysis yielded five subscore indices with notable accuracy. Among these, the navigation and orientation subscore achieved an R2 score of 0.72, while the landmark recognition subscore attained an R2 score of 0.50. Additionally, statistical features highly correlated with eye movement metrics, including blinks, saccades, and fixations, were identified. The findings of this study can lead to more cognitive assessments and enable early detection of spatial navigation impairments, particularly among individuals at risk of cognitive decline. 

**Abstract (ZH)**: 空间导航是一种复杂的认知功能，涉及视觉、听觉和本体感觉等多种感官输入，以理解并移动于环境之中。这一能力使人类能够构建心理地图、穿越环境并处理方向性线索，这对于探索新环境和在不熟悉的地方找到方向至关重要。本研究采用算法方法，利用眼动数据提取与人类空间导航相关的指标。通过利用眼电图信号，我们分析了统计特征并应用特征工程技术，研究导航任务中的眼动。本研究将信号处理和机器学习方法相结合，开发了用于导航和定向、空间焦虑、地标识别、路径调研和路径导航的指标。分析结果产生了五个子评分指标，其中导航和定向子评分的R²得分为0.72，地标识别子评分的R²得分为0.50。此外，还识别出与眼动指标高度相关的统计特征，包括眨眼、跳跃和注视。本研究的发现可以促进更有效的认知评估，并能够早期检测空间导航障碍，尤其是对认知功能衰退风险较高的个体。 

---
# Simulation of Hypergraph Algorithms with Looped Transformers 

**Title (ZH)**: 使用循环变压器模拟超图算法 

**Authors**: Xiaoyu Li, Yingyu Liang, Jiangxuan Long, Zhenmei Shi, Zhao Song, Zhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10688)  

**Abstract**: Looped Transformers have shown exceptional capability in simulating traditional graph algorithms, but their application to more complex structures like hypergraphs remains underexplored. Hypergraphs generalize graphs by modeling higher-order relationships among multiple entities, enabling richer representations but introducing significant computational challenges. In this work, we extend the Loop Transformer architecture to simulate hypergraph algorithms efficiently, addressing the gap between neural networks and combinatorial optimization over hypergraphs. In this paper, we extend the Loop Transformer architecture to simulate hypergraph algorithms efficiently, addressing the gap between neural networks and combinatorial optimization over hypergraphs. Specifically, we propose a novel degradation mechanism for reducing hypergraphs to graph representations, enabling the simulation of graph-based algorithms, such as Dijkstra's shortest path. Furthermore, we introduce a hyperedge-aware encoding scheme to simulate hypergraph-specific algorithms, exemplified by Helly's algorithm. The paper establishes theoretical guarantees for these simulations, demonstrating the feasibility of processing high-dimensional and combinatorial data using Loop Transformers. This work highlights the potential of Transformers as general-purpose algorithmic solvers for structured data. 

**Abstract (ZH)**: 循环变换器在模拟传统图算法方面展现了杰出的能力，但在应用于更复杂的结构（如超图）方面仍然存在未被充分探索的空间。超图通过对多个实体之间的高阶关系进行建模，使表示更加丰富，但也引入了重大的计算挑战。在本文中，我们扩展了循环变换器架构以高效地模拟超图算法，弥合了神经网络在处理超图上的组合优化问题时的差距。在本文中，我们扩展了循环变换器架构以高效地模拟超图算法，弥合了神经网络在处理超图上的组合优化问题时的差距。具体而言，我们提出了一种新的降解机制，用于将超图转换为图表示，从而能够模拟基于图的算法，如迪杰斯特拉最短路径算法。此外，我们引入了一种超边感知编码方案，用于模拟特定于超图的算法，例如赫利算法。该论文为这些模拟建立了理论保证，展示了使用循环变换器处理高维和组合数据的可行性。本工作突显了变换器作为结构化数据通用算法求解器的潜力。 

---
# Class-Imbalanced-Aware Adaptive Dataset Distillation for Scalable Pretrained Model on Credit Scoring 

**Title (ZH)**: 面向信贷评分的类不平衡感知自适应数据集蒸馏以实现可扩展的预训练模型 

**Authors**: Xia Li, Hanghang Zheng, Xiao Chen, Hong Liu, Mao Mao  

**Link**: [PDF](https://arxiv.org/pdf/2501.10677)  

**Abstract**: The advent of artificial intelligence has significantly enhanced credit scoring technologies. Despite the remarkable efficacy of advanced deep learning models, mainstream adoption continues to favor tree-structured models due to their robust predictive performance on tabular data. Although pretrained models have seen considerable development, their application within the financial realm predominantly revolves around question-answering tasks and the use of such models for tabular-structured credit scoring datasets remains largely unexplored. Tabular-oriented large models, such as TabPFN, has made the application of large models in credit scoring feasible, albeit can only processing with limited sample sizes. This paper provides a novel framework to combine tabular-tailored dataset distillation technique with the pretrained model, empowers the scalability for TabPFN. Furthermore, though class imbalance distribution is the common nature in financial datasets, its influence during dataset distillation has not been explored. We thus integrate the imbalance-aware techniques during dataset distillation, resulting in improved performance in financial datasets (e.g., a 2.5% enhancement in AUC). This study presents a novel framework for scaling up the application of large pretrained models on financial tabular datasets and offers a comparative analysis of the influence of class imbalance on the dataset distillation process. We believe this approach can broaden the applications and downstream tasks of large models in the financial domain. 

**Abstract (ZH)**: 人工智能的发展显著提升了信用评分技术。尽管高级深度学习模型取得了显著效果，主流应用仍然倾向于使用基于树结构的模型，因为它们在处理表格数据时表现出更强的预测能力。尽管预训练模型已有显著发展，但在金融领域的应用主要集中在问答任务上，将此类模型应用于结构化的信用评分数据集的研究仍较少。面向表格的大模型（如TabPFN）使得在信用评分中使用大型模型成为可能，尽管限制了样本大小的处理能力。本文提供了一种新颖的框架，将表格定制的数据集蒸馏技术与预训练模型相结合，增强了TabPFN的可扩展性。此外，尽管类不平衡是金融数据集的一个普遍特性，但在数据集蒸馏过程中对其影响的研究尚未见报道。因此，我们在数据集蒸馏过程中引入了不平衡意识的技术，从而在金融数据集上（例如，提高了AUC 2.5%）取得了更好的性能。本研究提出了一种扩展大型预训练模型在金融表格数据集中的应用的新框架，并对比分析了类不平衡对数据集蒸馏过程的影响。我们相信这种方法可以扩大大型模型在金融领域的应用范围和下游任务。 

---
# Unveiling the Mystery of Weight in Large Foundation Models: Gaussian Distribution Never Fades 

**Title (ZH)**: 揭开大型基础模型中权重之谜：高斯分布永不过时 

**Authors**: Chongjie Si, Jingjing Jiang, Wei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2501.10661)  

**Abstract**: This paper presents a pioneering exploration of the mechanisms underlying large foundation models' (LFMs) weights, aiming to simplify AI research. Through extensive observation and analysis on prevailing LFMs, we find that regardless of initialization strategies, their weights predominantly follow a Gaussian distribution, with occasional sharp, inverted T-shaped, or linear patterns. We further discover that the weights share the i.i.d. properties of Gaussian noise, and explore their direct relationship. We find that transformation weights can be derived from Gaussian noise, and they primarily serve to increase the standard deviation of pre-trained weights, with their standard deviation growing with layer depth. In other words, transformation weights broaden the acceptable deviation from the optimal weights, facilitating adaptation to downstream tasks. Building upon the above conclusions, we thoroughly discussed the nature of optimal weights, ultimately concluding that they should exhibit zero-mean, symmetry, and sparsity, with the sparse values being a truncated Gaussian distribution and a few outliers. Our experiments in LFM adaptation and editing demonstrate the effectiveness of these insights. We hope these findings can provide a foundational understanding to pave the way for future advancements in the LFM community. 

**Abstract (ZH)**: 本文对大型基础模型（LFMs）权重背后的机制进行了开创性的探索，旨在简化人工智能研究。通过对当前主流LFMs进行广泛的观察和分析，我们发现，无论采用何种初始化策略，其权重主要遵循正态分布，偶尔会出现尖锐的倒T型或线性模式。进一步的研究发现，权重共享高斯噪声的独立同分布（i.i.d.）特性，并探讨了它们之间的直接关系。我们发现转换权重可以从高斯噪声中导出，主要作用是增加预训练权重的标准差，且其标准差随着层数的加深而增加。换句话说，转换权重扩大了与最优权重的可接受偏差范围，有助于适应下游任务。基于上述结论，本文深入讨论了最优权重的本质，最终得出它们应表现出零均值、对称性和稀疏性的结论，稀疏值表现为截断的正态分布及少量异常值。我们在LFMs适应和编辑实验中的研究证明了这些见解的有效性。希望这些发现能为未来大型基础模型社区的发展奠定基础。 

---
# LUT-DLA: Lookup Table as Efficient Extreme Low-Bit Deep Learning Accelerator 

**Title (ZH)**: LUT-DLA：查找表作为高效的极端低比特深度学习加速器 

**Authors**: Guoyu Li, Shengyu Ye, Chunyun Chen, Yang Wang, Fan Yang, Ting Cao, Cheng Liu, Mohamed M. Sabry, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10658)  

**Abstract**: The emergence of neural network capabilities invariably leads to a significant surge in computational demands due to expanding model sizes and increased computational complexity. To reduce model size and lower inference costs, recent research has focused on simplifying models and designing hardware accelerators using low-bit quantization. However, due to numerical representation limits, scalar quantization cannot reduce bit width lower than 1-bit, diminishing its benefits. To break through these limitations, we introduce LUT-DLA, a Look-Up Table (LUT) Deep Learning Accelerator Framework that utilizes vector quantization to convert neural network models into LUTs, achieving extreme low-bit quantization. The LUT-DLA framework facilitates efficient and cost-effective hardware accelerator designs and supports the LUTBoost algorithm, which helps to transform various DNN models into LUT-based models via multistage training, drastically cutting both computational and hardware overhead. Additionally, through co-design space exploration, LUT-DLA assesses the impact of various model and hardware parameters to fine-tune hardware configurations for different application scenarios, optimizing performance and efficiency. Our comprehensive experiments show that LUT-DLA achieves improvements in power efficiency and area efficiency with gains of $1.4$~$7.0\times$ and $1.5$~$146.1\times$, respectively, while maintaining only a modest accuracy drop. For CNNs, accuracy decreases by $0.1\%$~$3.1\%$ using the $L_2$ distance similarity, $0.1\%$~$3.4\%$ with the $L_1$ distance similarity, and $0.1\%$~$3.8\%$ when employing the Chebyshev distance similarity. For transformer-based models, the accuracy drop ranges from $1.4\%$ to $3.0\%$. 

**Abstract (ZH)**: 神经网络能力的出现不可避免地导致了计算需求的显著增加，这是因为模型规模的扩大以及计算复杂性的提高。为了减少模型大小并降低推断成本，最近的研究集中于简化模型，并通过低比特量化设计硬件加速器。然而，由于数值表示的限制，标量量化不能将比特宽度降低到1比特以下，从而削弱了其优势。为突破这些限制，我们引入了LUT-DLA框架，这是一种基于查找表（LUT）的深度学习加速器框架，通过矢量量化将神经网络模型转换为查找表，实现了极端低比特量化。LUT-DLA框架促进了高效且成本效益高的硬件加速器设计，并支持LUTBoost算法，该算法通过多阶段训练，可以将各种DNN模型转换为基于查找表的模型，大幅降低了计算和硬件开销。此外，通过协同设计空间探索，LUT-DLA评估了各种模型和硬件参数的影响，以针对不同的应用场景精细调整硬件配置，优化性能和效率。我们的全面实验表明，LUT-DLA在保持适度精度下降的同时，分别实现了$1.4$~$7.0\times$和$1.5$~$146.1\times$的功耗效率和面积效率提升。对于卷积神经网络（CNNs），使用$L_2$距离相似性时，精度下降范围为$0.1\%$~$3.1\%$，使用$L_1$距离相似性时，精度下降范围为$0.1\%$~$3.4\%$，使用Chebyshev距离相似性时，精度下降范围为$0.1\%$~$3.8\%$。对于基于Transformer的模型，精度下降范围为$1.4\%$~$3.0\%$。 

---
# AI/ML Based Detection and Categorization of Covert Communication in IPv6 Network 

**Title (ZH)**: 基于AI/ML的IPv6网络隐蔽通信检测与分类 

**Authors**: Mohammad Wali Ur Rahman, Yu-Zheng Lin, Carter Weeks, David Ruddell, Jeff Gabriellini, Bill Hayes, Salim Hariri, Edward V. Ziegler Jr  

**Link**: [PDF](https://arxiv.org/pdf/2501.10627)  

**Abstract**: The flexibility and complexity of IPv6 extension headers allow attackers to create covert channels or bypass security mechanisms, leading to potential data breaches or system compromises. The mature development of machine learning has become the primary detection technology option used to mitigate covert communication threats. However, the complexity of detecting covert communication, evolving injection techniques, and scarcity of data make building machine-learning models challenging. In previous related research, machine learning has shown good performance in detecting covert communications, but oversimplified attack scenario assumptions cannot represent the complexity of modern covert technologies and make it easier for machine learning models to detect covert communications. To bridge this gap, in this study, we analyzed the packet structure and network traffic behavior of IPv6, used encryption algorithms, and performed covert communication injection without changing network packet behavior to get closer to real attack scenarios. In addition to analyzing and injecting methods for covert communications, this study also uses comprehensive machine learning techniques to train the model proposed in this study to detect threats, including traditional decision trees such as random forests and gradient boosting, as well as complex neural network architectures such as CNNs and LSTMs, to achieve detection accuracy of over 90\%. This study details the methods used for dataset augmentation and the comparative performance of the applied models, reinforcing insights into the adaptability and resilience of the machine learning application in IPv6 covert communication. In addition, we also proposed a Generative AI-assisted interpretation concept based on prompt engineering as a preliminary study of the role of Generative AI agents in covert communication. 

**Abstract (ZH)**: IPv6扩展头的灵活性和复杂性使攻击者能够创建隐蔽信道或绕过安全机制，从而导致潜在的数据泄露或系统妥协。机器学习的成熟发展已成为缓解隐蔽通信威胁的主要检测技术选择。然而，隐蔽通信的检测复杂性、注入技术的演变以及数据的稀缺性使得构建机器学习模型具有挑战性。在之前的相关研究中，机器学习在检测隐蔽通信方面表现出色，但过于简化的攻击场景假设无法代表现代隐蔽技术的复杂性，从而使机器学习模型更容易检测隐蔽通信。为弥合这一差距，本研究分析了IPv6的数据包结构和网络流量行为，使用加密算法，并在不改变网络数据包行为的情况下注入隐蔽通信，从而更接近真实的攻击场景。除分析和注入隐蔽通信的方法之外，本研究还使用了全面的机器学习技术来训练本研究中提出的模型，以检测威胁，包括传统的决策树如随机森林和梯度提升，以及复杂的神经网络架构如卷积神经网络（CNNs）和长短期记忆网络（LSTMs），以实现超过90%的检测准确性。本研究详细描述了数据集扩充方法和所应用模型的比较性能，强化了机器学习在IPv6隐蔽通信中的适应性和韧性。此外，我们还基于提示工程提出了生成式AI辅助解释的概念，作为生成式AI代理在隐蔽通信中角色的初步研究。 

---
# When language and vision meet road safety: leveraging multimodal large language models for video-based traffic accident analysis 

**Title (ZH)**: 当语言与视觉携手共进交通安全：利用多模态大型语言模型进行基于视频的道路交通事故分析 

**Authors**: Ruixuan Zhang, Beichen Wang, Juexiao Zhang, Zilin Bian, Chen Feng, Kaan Ozbay  

**Link**: [PDF](https://arxiv.org/pdf/2501.10604)  

**Abstract**: The increasing availability of traffic videos functioning on a 24/7/365 time scale has the great potential of increasing the spatio-temporal coverage of traffic accidents, which will help improve traffic safety. However, analyzing footage from hundreds, if not thousands, of traffic cameras in a 24/7/365 working protocol remains an extremely challenging task, as current vision-based approaches primarily focus on extracting raw information, such as vehicle trajectories or individual object detection, but require laborious post-processing to derive actionable insights. We propose SeeUnsafe, a new framework that integrates Multimodal Large Language Model (MLLM) agents to transform video-based traffic accident analysis from a traditional extraction-then-explanation workflow to a more interactive, conversational approach. This shift significantly enhances processing throughput by automating complex tasks like video classification and visual grounding, while improving adaptability by enabling seamless adjustments to diverse traffic scenarios and user-defined queries. Our framework employs a severity-based aggregation strategy to handle videos of various lengths and a novel multimodal prompt to generate structured responses for review and evaluation and enable fine-grained visual grounding. We introduce IMS (Information Matching Score), a new MLLM-based metric for aligning structured responses with ground truth. We conduct extensive experiments on the Toyota Woven Traffic Safety dataset, demonstrating that SeeUnsafe effectively performs accident-aware video classification and visual grounding by leveraging off-the-shelf MLLMs. Source code will be available at \url{this https URL}. 

**Abstract (ZH)**: 随着交通视频在全年无休（24/7/365）模式下变得越来越普及，这为提升交通事故的空间-时间覆盖范围提供了巨大潜力，进而有助于提高交通安全。然而，按照24/7/365的工作模式分析数百甚至数千个交通摄像头的视频内容仍然是一个极其具有挑战性的任务，因为当前基于视觉的方法主要集中在提取诸如车辆轨迹或个体对象检测等原始信息上，但需要大量的后处理才能得出有效的洞察。我们提出了SeeUnsafe这一新框架，将多模态大语言模型（MLLM）代理集成进来，将基于视频的交通事故分析从传统的提取-解释工作流程转变为一种更交互式的对话式方法。这一转变通过自动化复杂任务（如视频分类和视觉定位）大幅提高了处理吞吐量，并通过使系统能够无缝适应各种交通场景和用户定义的查询而提高了可适应性。我们的框架采用一种基于严重程度的聚合策略来处理不同长度的视频，并引入一种新型的多模态提示来生成结构化响应，进行审核和评估，并实现细粒度的视觉定位。我们引入了IMS（信息匹配评分）这一新的MLLM基元度量标准，用于将结构化响应与地面真相对齐。我们在Toyota Woven Traffic Safety数据集上进行了广泛的实验，证明SeeUnsafe能够有效利用现成的MLLM进行事故意识视频分类和视觉定位。源代码将发布在\url{this https URL}。 

---
# AI Technicians: Developing Rapid Occupational Training Methods for a Competitive AI Workforce 

**Title (ZH)**: AI技术人员：开发快速职业培训方法以培养竞争性的AI workforce 

**Authors**: Jaromir Savelka, Can Kultur, Arav Agarwal, Christopher Bogart, Heather Burte, Adam Zhang, Majd Sakr  

**Link**: [PDF](https://arxiv.org/pdf/2501.10579)  

**Abstract**: The accelerating pace of developments in Artificial Intelligence~(AI) and the increasing role that technology plays in society necessitates substantial changes in the structure of the workforce. Besides scientists and engineers, there is a need for a very large workforce of competent AI technicians (i.e., maintainers, integrators) and users~(i.e., operators). As traditional 4-year and 2-year degree-based education cannot fill this quickly opening gap, alternative training methods have to be developed. We present the results of the first four years of the AI Technicians program which is a unique collaboration between the U.S. Army's Artificial Intelligence Integration Center (AI2C) and Carnegie Mellon University to design, implement and evaluate novel rapid occupational training methods to create a competitive AI workforce at the technicians level. Through this multi-year effort we have already trained 59 AI Technicians. A key observation is that ongoing frequent updates to the training are necessary as the adoption of AI in the U.S. Army and within the society at large is evolving rapidly. A tight collaboration among the stakeholders from the army and the university is essential for successful development and maintenance of the training for the evolving role. Our findings can be leveraged by large organizations that face the challenge of developing a competent AI workforce as well as educators and researchers engaged in solving the challenge. 

**Abstract (ZH)**: 人工智能（AI）发展的加速以及技术在社会中扮演的日益重要的角色，要求对劳动力结构进行重大调整。除了科学家和工程师之外，还需要一支庞大的高素质AI技术人员（即维护人员、集成人员）和用户（即操作人员）队伍。传统基于四年制和两年制学位的教育无法迅速填补这一缺口，因此需要开发新的培训方法。我们介绍了AI技术人员项目前四年的结果，该项目是美国陆军人工智能集成中心（AI2C）与卡内基梅隆大学之间的一项独特合作，旨在设计、实施并评估新颖的快速职业培训方法，以在技术层面创建具有竞争力的AI劳动力队伍。通过这一多阶段努力，我们已经培训了59名AI技术人员。一个关键观察结果是，随着AI在美国陆军以及整个社会中的采用不断快速发展，持续不断、频繁的培训更新是必要的。军方和大学之间的紧密合作对于成功开发和维护适合不断变化角色的培训至关重要。我们的研究发现可以被面临培养高素质AI劳动力挑战的大企业所利用，也可以为致力于解决这一挑战的教育工作者和研究人员所借鉴。 

---
# AI Toolkit: Libraries and Essays for Exploring the Technology and Ethics of AI 

**Title (ZH)**: AI工具包：探索人工智能技术与伦理的库文件与论文集 

**Authors**: Levin Ho, Morgan McErlean, Zehua You, Douglas Blank, Lisa Meeden  

**Link**: [PDF](https://arxiv.org/pdf/2501.10576)  

**Abstract**: In this paper we describe the development and evaluation of AITK, the Artificial Intelligence Toolkit. This open-source project contains both Python libraries and computational essays (Jupyter notebooks) that together are designed to allow a diverse audience with little or no background in AI to interact with a variety of AI tools, exploring in more depth how they function, visualizing their outcomes, and gaining a better understanding of their ethical implications. These notebooks have been piloted at multiple institutions in a variety of humanities courses centered on the theme of responsible AI. In addition, we conducted usability testing of AITK. Our pilot studies and usability testing results indicate that AITK is easy to navigate and effective at helping users gain a better understanding of AI. Our goal, in this time of rapid innovations in AI, is for AITK to provide an accessible resource for faculty from any discipline looking to incorporate AI topics into their courses and for anyone eager to learn more about AI on their own. 

**Abstract (ZH)**: 在本文中，我们描述了AI Toolkit（AITK）的开发与评估过程。这个开源项目包括Python库和计算性文章（Jupyter笔记本），旨在让具有不同背景且对人工智能了解有限的用户能够与多种人工智能工具互动，更深入地探讨这些工具的工作原理，可视化其结果，并更好地了解其伦理影响。这些笔记本已在多所机构的各种以负责任的人工智能为主题的文科课程中进行了试点。此外，我们还对AITK进行了易用性测试。我们的试点研究和可用性测试结果表明，AITK易于导航并且能够有效地帮助用户更好地了解人工智能。在当前人工智能迅速发展的时期，我们的目标是为任何学科的教师提供一个可访问的资源，帮助他们将人工智能主题融入课程中，同时也为那些希望独立了解人工智能的人提供帮助。 

---
# Towards Data-Centric AI: A Comprehensive Survey of Traditional, Reinforcement, and Generative Approaches for Tabular Data Transformation 

**Title (ZH)**: 面向数据导向的AI：面向表格数据转换的传统、强化和生成方法综述 

**Authors**: Dongjie Wang, Yanyong Huang, Wangyang Ying, Haoyue Bai, Nanxu Gong, Xinyuan Wang, Sixun Dong, Tao Zhe, Kunpeng Liu, Meng Xiao, Pengfei Wang, Pengyang Wang, Hui Xiong, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10555)  

**Abstract**: Tabular data is one of the most widely used formats across industries, driving critical applications in areas such as finance, healthcare, and marketing. In the era of data-centric AI, improving data quality and representation has become essential for enhancing model performance, particularly in applications centered around tabular data. This survey examines the key aspects of tabular data-centric AI, emphasizing feature selection and feature generation as essential techniques for data space refinement. We provide a systematic review of feature selection methods, which identify and retain the most relevant data attributes, and feature generation approaches, which create new features to simplify the capture of complex data patterns. This survey offers a comprehensive overview of current methodologies through an analysis of recent advancements, practical applications, and the strengths and limitations of these techniques. Finally, we outline open challenges and suggest future perspectives to inspire continued innovation in this field. 

**Abstract (ZH)**: 表格数据是各个行业中广泛使用的一种格式，驱动着金融、医疗和营销等领域中的关键应用。在以数据为中心的AI时代，提高数据质量和表示已成为提升模型性能、特别是在表格数据应用中不可或缺的要素。本文综述了表格数据为中心的AI的关键方面，强调特征选择和特征生成作为数据空间细化的核心技术。我们对特征选择方法进行了系统性回顾，这些方法能够识别并保留最相关的数据属性，以及对生成新特征以简化复杂数据模式的捕捉进行讨论。本文通过分析近期进展、实际应用以及这些技术的强弱点，提供了当前方法的全面综述。最后，我们概述了开放性挑战，并提出了未来研究方向的建议，以激发该领域的持续创新。 

---
# Scalable Machine Learning Training Infrastructure for Online Ads Recommendation and Auction Scoring Modeling at Google 

**Title (ZH)**: 谷歌在线广告推荐和拍卖评分建模中的可扩展机器学习训练基础设施 

**Authors**: George Kurian, Somayeh Sardashti, Ryan Sims, Felix Berger, Gary Holt, Yang Li, Jeremiah Willcock, Kaiyuan Wang, Herve Quiroz, Abdulrahman Salem, Julian Grady  

**Link**: [PDF](https://arxiv.org/pdf/2501.10546)  

**Abstract**: Large-scale Ads recommendation and auction scoring models at Google scale demand immense computational resources. While specialized hardware like TPUs have improved linear algebra computations, bottlenecks persist in large-scale systems. This paper proposes solutions for three critical challenges that must be addressed for efficient end-to-end execution in a widely used production infrastructure: (1) Input Generation and Ingestion Pipeline: Efficiently transforming raw features (e.g., "search query") into numerical inputs and streaming them to TPUs; (2) Large Embedding Tables: Optimizing conversion of sparse features into dense floating-point vectors for neural network consumption; (3) Interruptions and Error Handling: Minimizing resource wastage in large-scale shared datacenters. To tackle these challenges, we propose a shared input generation technique to reduce computational load of input generation by amortizing costs across many models. Furthermore, we propose partitioning, pipelining, and RPC (Remote Procedure Call) coalescing software techniques to optimize embedding operations. To maintain efficiency at scale, we describe novel preemption notice and training hold mechanisms that minimize resource wastage, and ensure prompt error resolution. These techniques have demonstrated significant improvement in Google production, achieving a 116% performance boost and an 18% reduction in training costs across representative models. 

**Abstract (ZH)**: 在谷歌规模下，大规模广告推荐和拍卖评分模型需要巨大的计算资源。尽管专门的硬件如TPU（张量处理单元）在改进线性代数计算方面取得了进步，但在大规模系统中仍然存在瓶颈。本文提出了解决三个关键挑战的方法，以确保在广泛使用的生产基础设施中实现高效端到端执行：（1）输入生成和摄取管道：高效地将原始特征（例如，“搜索查询”）转换为数值输入，并将其流式传输到TPU；（2）大型嵌入表：优化稀疏特征转换为适用于神经网络的密集浮点向量的过程；（3）中断和错误处理：在大规模共享数据中心中最小化资源浪费。为了应对这些挑战，我们提出了一种共享输入生成技术，通过将成本分摊到多个模型中来减少输入生成的计算负担。此外，我们提出了一种分区、管道化和远程过程调用（RPC）合并的软件技术来优化嵌入操作。为了在大规模下保持效率，我们描述了新型预emption通知和训练暂停机制，这些机制可以最小化资源浪费，并确保及时解决问题。这些技术在谷歌生产环境中取得了显著的改进，实现了代表性模型116%的性能提升和18%的训练成本降低。 

---
# FORLAPS: An Innovative Data-Driven Reinforcement Learning Approach for Prescriptive Process Monitoring 

**Title (ZH)**: FORLAPS：一种创新的数据驱动强化学习方法，用于处方过程监控 

**Authors**: Mostafa Abbasi, Maziyar Khadivi, Maryam Ahang, Patricia Lasserre, Yves Lucet, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2501.10543)  

**Abstract**: We present a novel 5-step framework called Fine-Tuned Offline Reinforcement Learning Augmented Process Sequence Optimization (FORLAPS), which aims to identify optimal execution paths in business processes using reinforcement learning. We implemented this approach on real-life event logs from our case study an energy regulator in Canada and other real-life event logs, demonstrating the feasibility of the proposed method. Additionally, to compare FORLAPS with the existing models (Permutation Feature Importance and multi-task LSTM-Based model), we experimented to evaluate its effectiveness in terms of resource savings and process time span reduction. The experimental results on real-life event log validate that FORLAPS achieves 31% savings in resource time spent and a 23% reduction in process time span. Using this innovative data augmentation technique, we propose a fine-tuned reinforcement learning approach that aims to automatically fine-tune the model by selectively increasing the average estimated Q-value in the sampled batches. The results show that we obtained a 44% performance improvement compared to the pre-trained model. This study introduces an innovative evaluation model, benchmarking its performance against earlier works using nine publicly available datasets. Robustness is ensured through experiments utilizing the Damerau-Levenshtein distance as the primary metric. In addition, we discussed the suitability of datasets, taking into account their inherent properties, to evaluate the performance of different models. The proposed model, FORLAPS, demonstrated exceptional performance, outperforming existing state-of-the-art approaches in suggesting the most optimal policies or predicting the best next activities within a process trace. 

**Abstract (ZH)**: 我们提出了一种新颖的5步框架，名为Fine-Tuned Offline Reinforcement Learning Augmented Process Sequence Optimization (FORLAPS)，该框架旨在使用强化学习识别业务流程中的最优执行路径。我们在加拿大一家能源监管机构的真实事件日志和其他真实事件日志上实现了该方法，证明了所提方法的可行性。此外，为了与现有模型（排列特征重要性及多任务LSTM基模型）进行比较，我们进行了实验，评估其在资源节省和流程时间跨度减少方面的有效性。在真实事件日志上的实验结果表明，FORLAPS在资源时间节省方面实现了31%的节省，并在流程时间跨度减少方面实现了23%的优化。通过这一创新的数据增强技术，我们提出了一种调优的强化学习方法，该方法旨在通过在采样批次中选择性地增加平均估计Q值来自动调优模型。结果显示，与预训练模型相比，该方法提高了44%的性能。本研究引入了一种创新的评价模型，并通过九个公开可用的数据集将其性能与以往工作进行了基准比较。通过使用Damerau-Levenshtein距作为主要指标进行的试验确保了该方法的鲁棒性。此外，我们还讨论了数据集的适用性，考虑了其固有的属性来评估不同模型的性能。所提出的FORLAPS模型展示了卓越的性能，在建议最优策略或预测过程轨迹中的最佳下一步活动方面优于现有的最先进的方法。 

---
# Improved IR-based Bug Localization with Intelligent Relevance Feedback 

**Title (ZH)**: 基于改进的IR的智能相关反馈软件缺陷定位方法 

**Authors**: Asif Mohammed Samir, Mohammad Masudur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2501.10542)  

**Abstract**: Software bugs pose a significant challenge during development and maintenance, and practitioners spend nearly 50% of their time dealing with bugs. Many existing techniques adopt Information Retrieval (IR) to localize a reported bug using textual and semantic relevance between bug reports and source code. However, they often struggle to bridge a critical gap between bug reports and code that requires in-depth contextual understanding, which goes beyond textual or semantic relevance. In this paper, we present a novel technique for bug localization - BRaIn - that addresses the contextual gaps by assessing the relevance between bug reports and code with Large Language Models (LLM). It then leverages the LLM's feedback (a.k.a., Intelligent Relevance Feedback) to reformulate queries and re-rank source documents, improving bug localization. We evaluate BRaIn using a benchmark dataset, Bench4BL, and three performance metrics and compare it against six baseline techniques from the literature. Our experimental results show that BRaIn outperforms baselines by 87.6%, 89.5%, and 48.8% margins in MAP, MRR, and HIT@K, respectively. Additionally, it can localize approximately 52% of bugs that cannot be localized by the baseline techniques due to the poor quality of corresponding bug reports. By addressing the contextual gaps and introducing Intelligent Relevance Feedback, BRaIn advances not only theory but also improves IR-based bug localization. 

**Abstract (ZH)**: 软件错误在开发和维护过程中构成了重大挑战，从业者花费近一半的时间来处理这些错误。现有许多技术采用信息检索（IR）方法，通过错误报告与源代码之间的文本和语义相关性来定位错误报告。然而，这些方法往往难以弥合错误报告与代码之间的重要差距，这种差距要求深入了解上下文，而不仅仅是文本或语义相关性。本文提出了一种新的错误定位技术——BRaIn，该技术通过利用大型语言模型（LLM）评估错误报告与代码之间的相关性来解决这些上下文差距。在此基础上，利用LLM的反馈（即智能相关性反馈）来重新构建查询并重新排名源文档，从而提高错误定位的准确性。我们使用基准数据集Bench4BL和三种性能指标评估了BRaIn，并将其与文献中的六种基线技术进行了比较。实验结果表明，BRaIn在MAP、MRR和HIT@K指标上的表现分别优于基线技术87.6%、89.5%和48.8%。此外，它还能定位大约52%的由于错误报告质量不佳而无法被基线技术定位的错误。通过解决上下文差距并引入智能相关性反馈，BRaIn从理论上和实际应用上都推动了基于IR的错误定位技术的发展。 

---
# 4bit-Quantization in Vector-Embedding for RAG 

**Title (ZH)**: 4位量化在 vectors 嵌入中的应用：面向Retriever-Augmented Generation（检索增强生成）的场景 

**Authors**: Taehee Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2501.10534)  

**Abstract**: Retrieval-augmented generation (RAG) is a promising technique that has shown great potential in addressing some of the limitations of large language models (LLMs). LLMs have two major limitations: they can contain outdated information due to their training data, and they can generate factually inaccurate responses, a phenomenon known as hallucinations. RAG aims to mitigate these issues by leveraging a database of relevant documents, which are stored as embedding vectors in a high-dimensional space. However, one of the challenges of using high-dimensional embeddings is that they require a significant amount of memory to store. This can be a major issue, especially when dealing with large databases of documents. To alleviate this problem, we propose the use of 4-bit quantization to store the embedding vectors. This involves reducing the precision of the vectors from 32-bit floating-point numbers to 4-bit integers, which can significantly reduce the memory requirements. Our approach has several benefits. Firstly, it significantly reduces the memory storage requirements of the high-dimensional vector database, making it more feasible to deploy RAG systems in resource-constrained environments. Secondly, it speeds up the searching process, as the reduced precision of the vectors allows for faster computation. Our code is available at this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）是一种有前景的技术，已在解决大型语言模型（LLMs）的一些局限性方面显示出巨大的潜力。大型语言模型（LLMs）存在两大局限性：由于训练数据可能导致包含过时信息，以及生成事实不准确的响应，这种现象被称为幻觉。RAG 旨在通过利用相关文档数据库来缓解这些问题，这些文档作为嵌入向量存储在高维空间中。然而，使用高维嵌入的一个挑战是需要大量内存来存储它们。特别是在处理大量文档数据库时，这是一个重大问题。为了解决这个问题，我们提议使用4位量化来存储嵌入向量。这涉及到将向量的精度从32位浮点数减少到4位整数，这可以显著减少内存需求。我们的方法具有多个优势。首先，它显著减少了高维向量数据库的内存存储需求，使得在资源受限的环境中部署RAG系统更为可行。其次，它加速了搜索过程，因为向量的精度降低使得计算变快。我们的代码可在以下网址获取：[此处提供网址] 

---
# Solving Sparse Finite Element Problems on Neuromorphic Hardware 

**Title (ZH)**: 将下面的论文标题翻译成中文，符合学术规范：

Sparse有限元问题在神经形态硬件上的求解

这里的“Sparse”翻译为“稀疏”，“Finite Element”翻译为“有限元”，“Neuromorphic Hardware”翻译为“神经形态硬件”。这样的翻译既准确又符合学术文献的翻译习惯。 

**Authors**: Bradley H. Theilman, James B. Aimone  

**Link**: [PDF](https://arxiv.org/pdf/2501.10526)  

**Abstract**: We demonstrate that scalable neuromorphic hardware can implement the finite element method, which is a critical numerical method for engineering and scientific discovery. Our approach maps the sparse interactions between neighboring finite elements to small populations of neurons that dynamically update according to the governing physics of a desired problem description. We show that for the Poisson equation, which describes many physical systems such as gravitational and electrostatic fields, this cortical-inspired neural circuit can achieve comparable levels of numerical accuracy and scaling while enabling the use of inherently parallel and energy-efficient neuromorphic hardware. We demonstrate that this approach can be used on the Intel Loihi 2 platform and illustrate how this approach can be extended to nontrivial mesh geometries and dynamics. 

**Abstract (ZH)**: 我们证明了可扩展的神经形态硬件可以实现有限元方法，这是一种在工程和科学研究中至关重要的数值方法。我们的方法将相邻有限元之间的稀疏相互作用映射到动态根据目标问题描述的主导物理特性进行更新的少量神经元群体中。我们展示了这种方法在描述引力场和静电场等许多物理系统的泊松方程中，能够达到与传统方法相当的数值精度和可扩展性，同时利用了固有的并行性和能效优势的神经形态硬件。我们证明了这种方法可以在Intel Loihi 2平台上实现，并阐述了如何将该方法扩展到非平凡的网格几何形状和动力学中。 

---
# Real-Time Bus Departure Prediction Using Neural Networks for Smart IoT Public Bus Transit 

**Title (ZH)**: 使用神经网络实现智能物联网公共交通实时公交出发时间预测 

**Authors**: Narges Rashvand, Sanaz Sadat Hosseini, Mona Azarbayjani, Hamed Tabkhi  

**Link**: [PDF](https://arxiv.org/pdf/2501.10514)  

**Abstract**: Bus transit plays a vital role in urban public transportation but often struggles to provide accurate and reliable departure times. This leads to delays, passenger dissatisfaction, and decreased ridership, particularly in transit-dependent areas. A major challenge lies in the discrepancy between actual and scheduled bus departure times, which disrupts timetables and impacts overall operational efficiency. To address these challenges, this paper presents a neural network-based approach for real-time bus departure time prediction tailored for smart IoT public transit applications. We leverage AI-driven models to enhance the accuracy of bus schedules by preprocessing data, engineering relevant features, and implementing a fully connected neural network that utilizes historical departure data to predict departure times at subsequent stops. In our case study analyzing bus data from Boston, we observed an average deviation of nearly 4 minutes from scheduled times. However, our model, evaluated across 151 bus routes, demonstrates a significant improvement, predicting departure time deviations with an accuracy of under 80 seconds. This advancement not only improves the reliability of bus transit schedules but also plays a crucial role in enabling smart bus systems and IoT applications within public transit networks. By providing more accurate real-time predictions, our approach can facilitate the integration of IoT devices, such as smart bus stops and passenger information systems, that rely on precise data for optimal performance. 

**Abstract (ZH)**: 公交运输在城市公共交通中发挥着至关重要的作用，但往往难以提供准确可靠的发车时间。这导致了延误、乘客不满和乘客数量减少，尤其是在依赖公交出行的区域。主要挑战之一在于实际发车时间与原定时间之间的差距，这扰乱了时刻表，影响了整体运营效率。为应对这些挑战，本文提出了一种基于神经网络的实时公交发车时间预测方法，适用于智能物联网公共交通应用。本文利用人工智能驱动的模型通过预处理数据、工程化相关特征，并结合全连接神经网络，利用历史发车数据来预测后续车站的发车时间。在对波士顿公交数据进行案例研究中，我们发现实际发车时间与预定时间平均偏差接近4分钟。然而，我们在151条公交线路上的测试表明，我们的模型显示出显著的改进，预测发车时间偏差的准确率低于80秒。这一进展不仅提高了公交时刻表的可靠性，还在公共交通网络中推动智能公交系统和物联网应用的发展。通过提供更准确的实时预测，我们的方法可以促进依赖精确数据的物联网设备，如智能公交站牌和乘客信息系统等的集成，从而实现最佳性能。 

---
# Tabular-TX: Theme-Explanation Structure-based Table Summarization via In-Context Learning 

**Title (ZH)**: Tabular-TX：基于主题解释结构的表格总结方法通过上下文学习 

**Authors**: TaeYoon Kwack, Jisoo Kim, Ki Yong Jung, DongGeon Lee, Heesun Park  

**Link**: [PDF](https://arxiv.org/pdf/2501.10487)  

**Abstract**: This paper proposes a Theme-Explanation Structure-based Table Summarization (Tabular-TX) pipeline designed to efficiently process table data. Tabular-TX preprocesses table data by focusing on highlighted cells and then generates summary sentences structured with a Theme Part in the form of adverbial phrases followed by an Explanation Part in the form of clauses. In this process, customized analysis is performed by considering the structural characteristics and comparability of the table. Additionally, by utilizing In-Context Learning, Tabular-TX optimizes the analytical capabilities of large language models (LLMs) without the need for fine-tuning, effectively handling the structural complexity of table data. Results from applying the proposed Tabular-TX to generate table-based summaries demonstrated superior performance compared to existing fine-tuning-based methods, despite limitations in dataset size. Experimental results confirmed that Tabular-TX can process complex table data more effectively and established it as a new alternative for table-based question answering and summarization tasks, particularly in resource-constrained environments. 

**Abstract (ZH)**: 本文提出了一种基于主题-解释结构的表格摘要（Tabular-TX）流水线，旨在高效处理表格数据。Tabular-TX通过关注突出显示的单元格来预处理表格数据，然后生成由副词短语形式的主题部分（Theme Part）和由从句形式的解释部分（Explanation Part）组成的结构化总结句子。在这个过程中，通过考虑表格的结构特征和可比较性进行定制化分析。此外，通过利用上下文学习，Tabular-TX能够在不进行微调的情况下优化大型语言模型（LLM）的分析能力，有效地处理表格数据的结构复杂性。将提出的方法Tabular-TX应用于生成基于表格的摘要表明，尽管存在数据集规模的限制，其性能优于现有的基于微调的方法。实验结果证实，Tabular-TX能够更有效地处理复杂表格数据，并且作为一种新的替代方案，特别适用于资源受限环境中的表格问答和摘要任务。 

---
# Bias in Decision-Making for AI's Ethical Dilemmas: A Comparative Study of ChatGPT and Claude 

**Title (ZH)**: 人工智能伦理困境中决策偏见的研究：ChatGPT与Claude的比较分析 

**Authors**: Yile Yan, Yuqi Zhu, Wentao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10484)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled human-like responses across various tasks, raising questions about their ethical decision-making capabilities and potential biases. This study investigates protected attributes in LLMs through systematic evaluation of their responses to ethical dilemmas. Using two prominent models - GPT-3.5 Turbo and Claude 3.5 Sonnet - we analyzed their decision-making patterns across multiple protected attributes including age, gender, race, appearance, and disability status. Through 11,200 experimental trials involving both single-factor and two-factor protected attribute combinations, we evaluated the models' ethical preferences, sensitivity, stability, and clustering of preferences. Our findings reveal significant protected attributeses in both models, with consistent preferences for certain features (e.g., "good-looking") and systematic neglect of others. Notably, while GPT-3.5 Turbo showed stronger preferences aligned with traditional power structures, Claude 3.5 Sonnet demonstrated more diverse protected attribute choices. We also found that ethical sensitivity significantly decreases in more complex scenarios involving multiple protected attributes. Additionally, linguistic referents heavily influence the models' ethical evaluations, as demonstrated by differing responses to racial descriptors (e.g., "Yellow" versus "Asian"). These findings highlight critical concerns about the potential impact of LLM biases in autonomous decision-making systems and emphasize the need for careful consideration of protected attributes in AI development. Our study contributes to the growing body of research on AI ethics by providing a systematic framework for evaluating protected attributes in LLMs' ethical decision-making capabilities. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在各种任务中实现了类似人类的响应能力，引发了对其伦理决策能力和潜在偏见的质疑。本研究通过系统评估模型在伦理困境中的响应，探讨了LLMs中的保护属性。使用两种前沿模型——GPT-3.5 Turbo和Claude 3.5 Sonnet，我们分析了这些模型在年龄、性别、种族、外貌和残疾状态等多种保护属性下的决策模式。通过11,200次实验，包括单因素和两因素保护属性组合试验，我们评估了这些模型的伦理偏好、敏感性、稳定性和偏好模式的聚类。研究结果表明，这两个模型中都存在显著的保护属性偏见，呈现出对某些特征（例如，“外表好看”）的一致偏好，而对其他特征存在系统性的忽视。值得注意的是，虽然GPT-3.5 Turbo倾向于更符合传统的权力结构，但在伦理偏好方面显示出更强的倾向性，而Claude 3.5 Sonnet则表现出更多样的保护属性选择。此外，我们也发现，在涉及多个保护属性的复杂场景中，伦理敏感性显著降低。语言中的指称对模型的伦理评估有重大影响，这一点在种族描述词（例如，“黄色”与“亚洲”）引起的反应差异中得到了体现。这些发现强调了LLMs潜在偏见对自主决策系统的潜在影响，并强调了在人工智能开发中仔细考虑保护属性的重要性。本研究通过提供一个系统框架来评估LLMs的伦理决策能力，为人工智能伦理研究提供了宝贵的贡献。 

---
# ArxEval: Evaluating Retrieval and Generation in Language Models for Scientific Literature 

**Title (ZH)**: ArxEval：评估语言模型在科学文献检索与生成中的性能 

**Authors**: Aarush Sinha, Viraj Virk, Dipshikha Chakraborty, P.S. Sreeja  

**Link**: [PDF](https://arxiv.org/pdf/2501.10483)  

**Abstract**: Language Models [LMs] are now playing an increasingly large role in information generation and synthesis; the representation of scientific knowledge in these systems needs to be highly accurate. A prime challenge is hallucination; that is, generating apparently plausible but actually false information, including invented citations and nonexistent research papers. This kind of inaccuracy is dangerous in all the domains that require high levels of factual correctness, such as academia and education. This work presents a pipeline for evaluating the frequency with which language models hallucinate in generating responses in the scientific literature. We propose ArxEval, an evaluation pipeline with two tasks using ArXiv as a repository: Jumbled Titles and Mixed Titles. Our evaluation includes fifteen widely used language models and provides comparative insights into their reliability in handling scientific literature. 

**Abstract (ZH)**: 语言模型[LMs]现在在信息生成和综合中发挥着越来越重要的作用；这些系统中的科学知识表示需要极其准确。主要挑战之一是幻觉现象，即生成看似合理但实际上虚假的信息，包括虚构的引用和不存在的研究论文。这种不准确在要求高度事实准确性的所有领域都是危险的，如学术界和教育界。本研究提出了一种评估语言模型在生成科学文献响应时出现幻觉频率的 pipeline。我们提出了一种基于 ArXiv 的评估 pipeline，其中包括两项任务：乱序标题和混合标题。我们的评估涵盖了十五种广泛使用的人工智能语言模型，提供了它们在处理科学文献方面的可靠性的比较见解。 

---
# Securing the AI Frontier: Urgent Ethical and Regulatory Imperatives for AI-Driven Cybersecurity 

**Title (ZH)**: 保障人工智能前沿：驱动网络安全的人工智能领域迫切需要的伦理和监管规范 

**Authors**: Vikram Kulothungan  

**Link**: [PDF](https://arxiv.org/pdf/2501.10467)  

**Abstract**: This paper critically examines the evolving ethical and regulatory challenges posed by the integration of artificial intelligence (AI) in cybersecurity. We trace the historical development of AI regulation, highlighting major milestones from theoretical discussions in the 1940s to the implementation of recent global frameworks such as the European Union AI Act. The current regulatory landscape is analyzed, emphasizing risk-based approaches, sector-specific regulations, and the tension between fostering innovation and mitigating risks. Ethical concerns such as bias, transparency, accountability, privacy, and human oversight are explored in depth, along with their implications for AI-driven cybersecurity systems. Furthermore, we propose strategies for promoting AI literacy and public engagement, essential for shaping a future regulatory framework. Our findings underscore the need for a unified, globally harmonized regulatory approach that addresses the unique risks of AI in cybersecurity. We conclude by identifying future research opportunities and recommending pathways for collaboration between policymakers, industry leaders, and researchers to ensure the responsible deployment of AI technologies in cybersecurity. 

**Abstract (ZH)**: 本文批判性地探讨了人工智能（AI）在网络安全领域整合过程中不断演变的伦理和监管挑战。我们追溯了AI监管的历史发展，从20世纪40年代的理论讨论到最近实施的全球框架，如欧盟人工智能法案中的重大里程碑。当前的监管格局被分析，强调基于风险的方法、特定领域的法规以及促进创新与缓解风险之间的张力。深入探讨了伦理问题，如偏差、透明度、问责制、隐私和人类监督等问题及其对AI驱动的网络安全系统的影响。此外，我们还提出了促进AI普及和公众参与的战略，这对于制定未来监管框架至关重要。研究结果强调，需要一种统一的、全球协调一致的监管方法，以解决AI在网络安全领域的独特风险。最后，我们确定了未来研究的机会，并建议政策措施者、行业领袖和研究人员之间的合作途径，确保AI技术在网络安全领域的负责任部署。 

---
# Improving the Efficiency of Self-Supervised Adversarial Training through Latent Clustering-Based Selection 

**Title (ZH)**: 通过潜在聚类为基础的选择改进自我监督对抗训练的效率 

**Authors**: Somrita Ghosh, Yuelin Xu, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10466)  

**Abstract**: Compared with standard learning, adversarially robust learning is widely recognized to demand significantly more training examples. Recent works propose the use of self-supervised adversarial training (SSAT) with external or synthetically generated unlabeled data to enhance model robustness. However, SSAT requires a substantial amount of extra unlabeled data, significantly increasing memory usage and model training times. To address these challenges, we propose novel methods to strategically select a small subset of unlabeled data essential for SSAT and robustness improvement. Our selection prioritizes data points near the model's decision boundary based on latent clustering-based techniques, efficiently identifying a critical subset of unlabeled data with a higher concentration of boundary-adjacent points. While focusing on near-boundary data, our methods are designed to maintain a balanced ratio between boundary and non-boundary data points to avoid overfitting. Our experiments on image benchmarks show that integrating our selection strategies into self-supervised adversarial training can largely reduce memory and computational requirements while achieving high model robustness. In particular, our latent clustering-based selection method with k-means is the most effective, achieving nearly identical test-time robust accuracies with 5 to 10 times less external or generated unlabeled data when applied to image benchmarks. Additionally, we validate the generalizability of our approach across various application scenarios, including a real-world medical dataset for COVID-19 chest X-ray classification. 

**Abstract (ZH)**: 与标准学习相比，对抗鲁棒学习广泛认为需要更多的训练样本。近期的研究提出使用外部或合成生成的未标记数据进行半监督对抗训练（SSAT），以提高模型的鲁棒性。然而，SSAT 需要大量的额外未标记数据，显著增加了内存使用和模型训练时间。为应对这些挑战，我们提出了一种新的方法，以战略方式选择一小部分对于 SSAT 和鲁棒性改进至关重要的未标记数据。我们的选择优先考虑基于潜在聚类技术的模型决策边界附近的样本，高效地识别出边界附近样本浓度较高的关键未标记数据子集。在关注边界附近数据的同时，我们的方法设计旨在保持边界数据点与非边界数据点之间的平衡比例，以防止过拟合。在图像基准测试上的实验表明，在半监督对抗训练中集成我们的选择策略可以大大减少内存和计算需求，同时保持高模型鲁棒性。特别是，基于 k-means 的潜在聚类选择方法最有效，当应用于图像基准测试时，与外部或生成的未标记数据相比，其在测试时的鲁棒准确率仅减少约 5 到 10 倍。此外，我们验证了该方法在各种应用场景中的通用性，包括用于 COVID-19 胸部 X 射线分类的实时医疗数据集。 

---
# The Mathematics of Artificial Intelligence 

**Title (ZH)**: 人工智能的数学理论 

**Authors**: Gabriel Peyré  

**Link**: [PDF](https://arxiv.org/pdf/2501.10465)  

**Abstract**: This overview article highlights the critical role of mathematics in artificial intelligence (AI), emphasizing that mathematics provides tools to better understand and enhance AI systems. Conversely, AI raises new problems and drives the development of new mathematics at the intersection of various fields. This article focuses on the application of analytical and probabilistic tools to model neural network architectures and better understand their optimization. Statistical questions (particularly the generalization capacity of these networks) are intentionally set aside, though they are of crucial importance. We also shed light on the evolution of ideas that have enabled significant advances in AI through architectures tailored to specific tasks, each echoing distinct mathematical techniques. The goal is to encourage more mathematicians to take an interest in and contribute to this exciting field. 

**Abstract (ZH)**: 这篇概览性文章强调了数学在人工智能（AI）中的关键作用，突出了数学为更好地理解并优化AI系统提供了工具。相反，AI 提出了新的问题，并在多个领域交叉处推动了新数学的发展。本文重点在于分析和概率工具在建模神经网络架构及其优化中的应用。统计问题（特别是这些网络的泛化能力）虽然十分重要，但本文并未讨论。此外，文章还揭示了若干核心思想的发展历程，这些思想通过针对特定任务的架构使人工智能取得了显著进步，每种架构都反映了不同的数学技术。目标是鼓励更多的数学家对这一充满活力的领域产生兴趣并作出贡献。 

---
# Adapting Beyond the Depth Limit: Counter Strategies in Large Imperfect Information Games 

**Title (ZH)**: 超出深度限制的适应策略：大型不完美信息博弈中的反制策略 

**Authors**: David Milec, Vojtěch Kovařík, Viliam Lisý  

**Link**: [PDF](https://arxiv.org/pdf/2501.10464)  

**Abstract**: We study the problem of adapting to a known sub-rational opponent during online play while remaining robust to rational opponents. We focus on large imperfect-information (zero-sum) games, which makes it impossible to inspect the whole game tree at once and necessitates the use of depth-limited search. However, all existing methods assume rational play beyond the depth-limit, which only allows them to adapt a very limited portion of the opponent's behaviour. We propose an algorithm Adapting Beyond Depth-limit (ABD) that uses a strategy-portfolio approach - which we refer to as matrix-valued states - for depth-limited search. This allows the algorithm to fully utilise all information about the opponent model, making it the first robust-adaptation method to be able to do so in large imperfect-information games. As an additional benefit, the use of matrix-valued states makes the algorithm simpler than traditional methods based on optimal value functions. Our experimental results in poker and battleship show that ABD yields more than a twofold increase in utility when facing opponents who make mistakes beyond the depth limit and also delivers significant improvements in utility and safety against randomly generated opponents. 

**Abstract (ZH)**: 我们研究在在线游戏中适应已知亚理性对手的同时保持对理性对手的鲁棒性的问题。我们关注大型不完美信息（零和）博弈，这使得无法一次性检查整个博弈树，从而需要使用深度限制搜索。然而，现有的所有方法都假设在深度限制之外对手也进行理性博弈，这限制了算法只能对对手行为进行有限的适应。我们提出了一种名为“超越深度限制适应”（ABD，Adapting Beyond Depth-limit）的算法，该算法采用一种策略组合方法——我们称之为矩阵状态，以便进行深度限制搜索。这种方法使算法能够充分利用所有关于对手模型的信息，成为首个能够在大型不完美信息博弈中实现这种全面适应的方法。此外，矩阵状态的使用使该算法比基于最优值函数的传统方法更为简单。我们在扑克和水雷游戏中的实验结果表明，当面对在深度限制之外犯错误的对手时，ABD 能够显著提高收益，同时在随机生成的对手面前也提供了显著的收益和安全保障改进。 

---
# GLow -- A Novel, Flower-Based Simulated Gossip Learning Strategy 

**Title (ZH)**: GLow —— 一种新型的基于花朵的模拟闲谈学习策略 

**Authors**: Aitor Belenguer, Jose A. Pascual, Javier Navaridas  

**Link**: [PDF](https://arxiv.org/pdf/2501.10463)  

**Abstract**: Fully decentralized learning algorithms are still in an early stage of development. Creating modular Gossip Learning strategies is not trivial due to convergence challenges and Byzantine faults intrinsic in systems of decentralized nature. Our contribution provides a novel means to simulate custom Gossip Learning systems by leveraging the state-of-the-art Flower Framework. Specifically, we introduce GLow, which will allow researchers to train and assess scalability and convergence of devices, across custom network topologies, before making a physical deployment. The Flower Framework is selected for being a simulation featured library with a very active community on Federated Learning research. However, Flower exclusively includes vanilla Federated Learning strategies and, thus, is not originally designed to perform simulations without a centralized authority. GLow is presented to fill this gap and make simulation of Gossip Learning systems possible. Results achieved by GLow in the MNIST and CIFAR10 datasets, show accuracies over 0.98 and 0.75 respectively. More importantly, GLow performs similarly in terms of accuracy and convergence to its analogous Centralized and Federated approaches in all designed experiments. 

**Abstract (ZH)**: 全分布式学习算法仍处于早期发展阶段。创建模块化的Googlis学习策略由于分布式系统固有的收敛性和拜占庭故障挑战而并非易事。我们的贡献提供了一种通过利用最先进的Flower框架来模拟自定义Googlis学习系统的新型方法。具体而言，我们介绍了GLow，这将使研究人员能够在实际部署前基于自定义网络拓扑训练和评估设备的可扩展性和收敛性。Flower框架因其在联邦学习研究领域具有非常活跃的社区而被选中作为模拟功能库。然而，Flower仅包括基本的联邦学习策略，因此它并不是专门为在无中心权威的情况下进行模拟而设计的。GLow 旨在填补这一空白，使其能够模拟Googlis学习系统。GLow 在 MNIST 和 CIFAR10 数据集上的结果分别显示准确率超过 0.98 和 0.75。更重要的是，GLow 在所有设计的实验中，在准确性和收敛性方面均与相应的集中式和联邦学习方法表现相似。 

---
# BloomScene: Lightweight Structured 3D Gaussian Splatting for Crossmodal Scene Generation 

**Title (ZH)**: BloomScene：轻量级结构化3D高斯点渲染在跨模态场景生成中的应用 

**Authors**: Xiaolu Hou, Mingcheng Li, Dingkang Yang, Jiawei Chen, Ziyun Qian, Xiao Zhao, Yue Jiang, Jinjie Wei, Qingyao Xu, Lihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10462)  

**Abstract**: With the widespread use of virtual reality applications, 3D scene generation has become a new challenging research frontier. 3D scenes have highly complex structures and need to ensure that the output is dense, coherent, and contains all necessary structures. Many current 3D scene generation methods rely on pre-trained text-to-image diffusion models and monocular depth estimators. However, the generated scenes occupy large amounts of storage space and often lack effective regularisation methods, leading to geometric distortions. To this end, we propose BloomScene, a lightweight structured 3D Gaussian splatting for crossmodal scene generation, which creates diverse and high-quality 3D scenes from text or image inputs. Specifically, a crossmodal progressive scene generation framework is proposed to generate coherent scenes utilizing incremental point cloud reconstruction and 3D Gaussian splatting. Additionally, we propose a hierarchical depth prior-based regularization mechanism that utilizes multi-level constraints on depth accuracy and smoothness to enhance the realism and continuity of the generated scenes. Ultimately, we propose a structured context-guided compression mechanism that exploits structured hash grids to model the context of unorganized anchor attributes, which significantly eliminates structural redundancy and reduces storage overhead. Comprehensive experiments across multiple scenes demonstrate the significant potential and advantages of our framework compared with several baselines. 

**Abstract (ZH)**: 随着虚拟现实应用的广泛使用，三维场景生成已成为新的研究前沿。三维场景具有高度复杂的结构，需要确保输出的密集、连贯，并包含所有必要的结构。当前许多三维场景生成方法依赖于预训练的文本到图像扩散模型和单目深度估计器。然而，生成的场景占用大量的存储空间，并且通常缺乏有效的正则化方法，导致几何失真。为了解决这些问题，我们提出了BloomScene，这是一种轻量级结构化的3D Gaussian Splatting方法，用于跨模态场景生成，能够从文本或图像输入中生成多样且高质量的三维场景。具体而言，我们提出了一种跨模态渐进式场景生成框架，利用增量点云重建和3D Gaussian Splatting生成连贯的场景。此外，我们提出了基于层次深度先验的正则化机制，利用多级深度精度和光滑性约束来增强生成场景的真实感和连贯性。最后，我们提出了一种结构化上下文引导的压缩机制，利用结构化哈希网格来建模无序锚属性的上下文，这显著减少了结构冗余并降低了存储开销。在多个场景上进行的全面实验表明，与几种基线方法相比，我们的框架具有显著的优势和潜力。 

---
# A Framework for Mining Collectively-Behaving Bots in MMORPGs 

**Title (ZH)**: MMORPG中集体行为机器人挖掘的框架 

**Authors**: Hyunsoo Kim, Jun Hee Kim, Jaeman Son, Jihoon Song, Eunjo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.10461)  

**Abstract**: In MMORPGs (Massively Multiplayer Online Role-Playing Games), abnormal players (bots) using unauthorized automated programs to carry out pre-defined behaviors systematically and repeatedly are commonly observed. Bots usually engage in these activities to gain in-game money, which they eventually trade for real money outside the game. Such abusive activities negatively impact the in-game experiences of legitimate users since bots monopolize specific hunting areas and obtain valuable items. Thus, detecting abnormal players is a significant task for game companies. Motivated by the fact that bots tend to behave collectively with similar in-game trajectories due to the auto-programs, we developed BotTRep, a framework that comprises trajectory representation learning followed by clustering using a completely unlabeled in-game trajectory dataset. Our model aims to learn representations for in-game trajectory sequences so that players with contextually similar trajectories have closer embeddings. Then, by applying DBSCAN to these representations and visualizing the corresponding moving patterns, our framework ultimately assists game masters in identifying and banning bots. 

**Abstract (ZH)**: 在大规模多人在线角色扮演游戏（MMORPG）中，使用未经授权的自动化程序进行系统性和重复性预定义行为的异常玩家（称为“脚本玩家”或“bot”）是常见现象。这类玩家通常通过在游戏中获得更多货币，然后将其兑换成真实货币来牟利。这些不当行为会负面影响合法玩家的游戏体验，因为脚本玩家会垄断某些狩猎区域，并获取珍贵物品。因此，检测异常玩家是游戏公司的重要任务之一。鉴于脚本玩家往往由于自动化程序而表现出相似的游戏轨迹，我们开发了BotTRep框架，该框架包括无标签游戏轨迹数据集的轨迹表示学习和聚类。我们的模型旨在学习游戏轨迹序列的表示，使具有类似上下文轨迹的玩家具有更接近的嵌入。然后，通过将DBSCAN应用于这些表示并可视化相应的移动模式，我们的框架最终帮助游戏大师识别并封禁脚本玩家。 

---
# Uncovering Bias in Foundation Models: Impact, Testing, Harm, and Mitigation 

**Title (ZH)**: 探索基础模型中的偏差：影响、检测、危害及缓解 

**Authors**: Shuzhou Sun, Li Liu, Yongxiang Liu, Zhen Liu, Shuanghui Zhang, Janne Heikkilä, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.10453)  

**Abstract**: Bias in Foundation Models (FMs) - trained on vast datasets spanning societal and historical knowledge - poses significant challenges for fairness and equity across fields such as healthcare, education, and finance. These biases, rooted in the overrepresentation of stereotypes and societal inequalities in training data, exacerbate real-world discrimination, reinforce harmful stereotypes, and erode trust in AI systems. To address this, we introduce Trident Probe Testing (TriProTesting), a systematic testing method that detects explicit and implicit biases using semantically designed probes. Here we show that FMs, including CLIP, ALIGN, BridgeTower, and OWLv2, demonstrate pervasive biases across single and mixed social attributes (gender, race, age, and occupation). Notably, we uncover mixed biases when social attributes are combined, such as gender x race, gender x age, and gender x occupation, revealing deeper layers of discrimination. We further propose Adaptive Logit Adjustment (AdaLogAdjustment), a post-processing technique that dynamically redistributes probability power to mitigate these biases effectively, achieving significant improvements in fairness without retraining models. These findings highlight the urgent need for ethical AI practices and interdisciplinary solutions to address biases not only at the model level but also in societal structures. Our work provides a scalable and interpretable solution that advances fairness in AI systems while offering practical insights for future research on fair AI technologies. 

**Abstract (ZH)**: 基础模型（FMs）—在涵盖社会和历史知识的广泛数据集上训练—对公平性和公正性提出了重大挑战，尤其是在医疗保健、教育和金融等领域。这些偏见源于训练数据中刻板印象和社会不平等的过度代表，加剧了现实生活中的歧视，强化了有害的刻板印象，并削弱了人们对AI系统的信任。为应对这一问题，我们提出了一种系统性的测试方法——三叉戟探针测试（TriProTesting），该方法使用语义设计的探针对显性和隐性偏见进行检测。研究表明，FMs，包括CLIP、ALIGN、BridgeTower和OWLv2，在单一和社会混合属性（性别、种族、年龄和职业）方面普遍存在偏见。值得注意的是，当我们结合社会属性时，发现了混合偏见，例如性别×种族、性别×年龄和性别×职业，揭示了更深层次的歧视层次。我们进一步提出了一种自适应逻辑调整技术（AdaLogAdjustment），这是一种后处理技术，能够动态重新分配概率权重，以有效减轻这些偏见，同时无需重新训练模型，显著提高了公平性。这些发现强调了迫切需要伦理AI实践和跨学科解决方案，不仅在模型层面，也在社会结构层面解决偏见问题。我们的研究提供了一种可扩展且可解释的解决方案，有助于提高AI系统的公平性，同时还为未来公平AI技术的研究提供了实用见解。 

---
# Towards Lightweight Time Series Forecasting: a Patch-wise Transformer with Weak Data Enriching 

**Title (ZH)**: 轻量级时间序列预测方法探索：基于patches的Transformer结合弱数据增强 

**Authors**: Meng Wang, Jintao Yang, Bin Yang, Hui Li, Tongxin Gong, Bo Yang, Jiangtao Cui  

**Link**: [PDF](https://arxiv.org/pdf/2501.10448)  

**Abstract**: Patch-wise Transformer based time series forecasting achieves superior accuracy. However, this superiority relies heavily on intricate model design with massive parameters, rendering both training and inference expensive, thus preventing their deployments on edge devices with limited resources and low latency requirements. In addition, existing methods often work in an autoregressive manner, which take into account only historical values, but ignore valuable, easy-to-obtain context information, such as weather forecasts, date and time of day. To contend with the two limitations, we propose LiPFormer, a novel Lightweight Patch-wise Transformer with weak data enriching. First, to simplify the Transformer backbone, LiPFormer employs a novel lightweight cross-patch attention and a linear transformation-based attention to eliminate Layer Normalization and Feed Forward Network, two heavy components in existing Transformers. Second, we propose a lightweight, weak data enriching module to provide additional, valuable weak supervision to the training. It enhances forecasting accuracy without significantly increasing model complexity as it does not involve expensive, human-labeling but using easily accessible context information. This facilitates the weak data enriching to plug-and-play on existing models. Extensive experiments on nine benchmark time series datasets demonstrate that LiPFormer outperforms state-of-the-art methods in accuracy, while significantly reducing parameter scale, training duration, and GPU memory usage. Deployment on an edge device reveals that LiPFormer takes only 1/3 inference time compared to classic Transformers. In addition, we demonstrate that the weak data enriching can integrate seamlessly into various Transformer based models to enhance their accuracy, suggesting its generality. 

**Abstract (ZH)**: 基于Patch的Transformer在时间序列预测中实现了卓越的准确性。然而，这种优势很大程度上依赖于复杂且参数量巨大的模型设计，这使得训练和推理过程变得昂贵，从而限制了其在资源有限且对低延迟有严格要求的边缘设备上的部署。此外，现有方法往往以自回归的方式进行，只考虑了历史值，而忽略了有价值的、易于获取的上下文信息，如天气预报、日期和时间等。为解决上述限制，我们提出了一种新型轻量级Patch Transformer —— LiPFormer，具有弱数据增强功能。首先，为了简化Transformer的骨干，LiPFormer采用了新颖的轻量级跨块注意机制和基于线性变换的注意机制，去掉了现有的Transformers中的层规范化和前馈网络这两种沉重的组件。其次，我们提出了一种轻量级的弱数据增强模块，以提供额外且有价值的弱监督信号给训练过程。这一模块通过增强预测准确性，同时减少了模型复杂度，因为它不依赖于耗时的人工标注，而是利用易于获取的上下文信息。该模块可以轻松集成到现有的模型中。在九个基准时间序列数据集上的大量实验表明，与现有最先进的方法相比，LiPFormer不仅在准确性上表现更优，还显著减少了参数规模、训练时间和GPU内存使用。在边缘设备上的部署显示出LiPFormer的推理时间仅为经典Transformer的1/3。此外，我们证明弱数据增强可以无缝集成到各种基于Transformer的模型中，以增强其准确性，表明了这个模块的通用性。 

---
# CodEv: An Automated Grading Framework Leveraging Large Language Models for Consistent and Constructive Feedback 

**Title (ZH)**: CodEv：一种利用大规模语言模型自动评分并提供一致性和建设性反馈的框架 

**Authors**: En-Qi Tseng, Pei-Cing Huang, Chan Hsu, Peng-Yi Wu, Chan-Tung Ku, Yihuang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10421)  

**Abstract**: Grading programming assignments is crucial for guiding students to improve their programming skills and coding styles. This study presents an automated grading framework, CodEv, which leverages Large Language Models (LLMs) to provide consistent and constructive feedback. We incorporate Chain of Thought (CoT) prompting techniques to enhance the reasoning capabilities of LLMs and ensure that the grading is aligned with human evaluation. Our framework also integrates LLM ensembles to improve the accuracy and consistency of scores, along with agreement tests to deliver reliable feedback and code review comments. The results demonstrate that the framework can yield grading results comparable to human evaluators, by using smaller LLMs. Evaluation and consistency tests of the LLMs further validate our approach, confirming the reliability of the generated scores and feedback. 

**Abstract (ZH)**: 编程作业的自动评分对于指导学生提高编程技能和编码风格至关重要。本研究提出了一种基于大型语言模型（LLMs）的自动评分框架，称为CodEv，该框架能够提供一致而建设性的反馈。我们采用Chain of Thought（CoT）提示技术来增强LLMs的推理能力，并确保评分与人工评估相一致。此外，我们的框架还集成了LLM集成技术，以提高评分的准确性和一致性，并通过一致性测试提供可靠反馈和代码审查评论。结果表明，该框架能够在使用较小的LLMs时获得与人工评估者相当的评分结果。LLMs的评估和一致性测试进一步验证了我们的方法，确认了生成的评分和反馈的可靠性。 

---
# Cooperative Search and Track of Rogue Drones using Multiagent Reinforcement Learning 

**Title (ZH)**: 使用多智能体强化学习进行恶意无人机的协同搜索与跟踪 

**Authors**: Panayiota Valianti, Kleanthis Malialis, Panayiotis Kolios, Georgios Ellinas  

**Link**: [PDF](https://arxiv.org/pdf/2501.10413)  

**Abstract**: This work considers the problem of intercepting rogue drones targeting sensitive critical infrastructure facilities. While current interception technologies focus mainly on the jamming/spoofing tasks, the challenges of effectively locating and tracking rogue drones have not received adequate attention. Solving this problem and integrating with recently proposed interception techniques will enable a holistic system that can reliably detect, track, and neutralize rogue drones. Specifically, this work considers a team of pursuer UAVs that can search, detect, and track multiple rogue drones over a sensitive facility. The joint search and track problem is addressed through a novel multiagent reinforcement learning scheme to optimize the agent mobility control actions that maximize the number of rogue drones detected and tracked. The performance of the proposed system is investigated under realistic settings through extensive simulation experiments with varying number of agents demonstrating both its performance and scalability. 

**Abstract (ZH)**: 本研究探讨了拦截针对敏感关键基础设施的恶意无人机的问题。尽管当前的拦截技术主要集中在干扰/欺骗任务上，但对于有效定位和跟踪恶意无人机的挑战尚未得到充分的关注。解决这些问题并将这些挑战与最近提出的拦截技术相结合，可以使系统能够可靠地检测、跟踪并中和恶意无人机。具体而言，本研究考虑了一组追捕无人飞行器（UAV），这些追捕无人飞行器可以在敏感设施区域内搜索、检测和跟踪多个恶意无人机。通过一个新颖的多智能体强化学习方案来解决联合搜索和跟踪问题，以优化智能体移动控制动作，最大化检测和跟踪到的恶意无人机的数量。通过广泛仿真实验，在不同智能体数量下的现实环境中考察所提系统的性能和可扩展性，以验证其有效性和可扩展性。 

---
# AI-Powered Urban Transportation Digital Twin: Methods and Applications 

**Title (ZH)**: 基于AI的城市交通数字化双胞胎：方法与应用 

**Authors**: Xuan Di, Yongjie Fu, Mehmet K.Turkcan, Mahshid Ghasemi, Zhaobin Mo, Chengbo Zang, Abhishek Adhikari, Zoran Kostic, Gil Zussman  

**Link**: [PDF](https://arxiv.org/pdf/2501.10396)  

**Abstract**: We present a survey paper on methods and applications of digital twins (DT) for urban traffic management. While the majority of studies on the DT focus on its "eyes," which is the emerging sensing and perception like object detection and tracking, what really distinguishes the DT from a traditional simulator lies in its ``brain," the prediction and decision making capabilities of extracting patterns and making informed decisions from what has been seen and perceived. In order to add values to urban transportation management, DTs need to be powered by artificial intelligence and complement with low-latency high-bandwidth sensing and networking technologies. We will first review the DT pipeline leveraging cyberphysical systems and propose our DT architecture deployed on a real-world testbed in New York City. This survey paper can be a pointer to help researchers and practitioners identify challenges and opportunities for the development of DTs; a bridge to initiate conversations across disciplines; and a road map to exploiting potentials of DTs for diverse urban transportation applications. 

**Abstract (ZH)**: 我们 presents 一篇关于数字孪生（DT）在城市交通管理中的方法和应用的综述论文。尽管大多数关于 DT 的研究集中在它的“眼睛”上，即新兴的感知和识别技术（如物体检测和跟踪），DT 真正将其与传统模拟器区分开来的是其“大脑”，即预测和决策能力。通过从已感知的数据中提取模式并做出明智的决策。为了使 DT 在城市交通管理中更具价值，它们需要利用人工智能，并结合低延迟高带宽的传感与网络技术。首先，我们将回顾利用计算物理系统的 DT 管道，并在纽约市的一个实际测试平台上提出我们的 DT 架构。本文综述可以成为研究人员和实践者识别 DT 发展挑战和机遇的指南；促进跨学科交流的桥梁；以及开发 DT 在多样化城市交通应用中潜力的路线图。 

---
# Towards General Purpose Robots at Scale: Lifelong Learning and Learning to Use Memory 

**Title (ZH)**: 面向大规模通用机器人：终身学习与利用记忆的学习 

**Authors**: William Yue  

**Link**: [PDF](https://arxiv.org/pdf/2501.10395)  

**Abstract**: The widespread success of artificial intelligence in fields like natural language processing and computer vision has not yet fully transferred to robotics, where progress is hindered by the lack of large-scale training data and the complexity of real-world tasks. To address this, many robot learning researchers are pushing to get robots deployed at scale in everyday unstructured environments like our homes to initiate a data flywheel. While current robot learning systems are effective for certain short-horizon tasks, they are not designed to autonomously operate over long time horizons in unstructured environments. This thesis focuses on addressing two key challenges for robots operating over long time horizons: memory and lifelong learning.
We propose two novel methods to advance these capabilities. First, we introduce t-DGR, a trajectory-based deep generative replay method that achieves state-of-the-art performance on Continual World benchmarks, advancing lifelong learning. Second, we develop a framework that leverages human demonstrations to teach agents effective memory utilization, improving learning efficiency and success rates on Memory Gym tasks. Finally, we discuss future directions for achieving the lifelong learning and memory capabilities necessary for robots to function at scale in real-world settings. 

**Abstract (ZH)**: 人工智能在自然语言处理和计算机视觉等领域取得的广泛成功尚未完全转移到机器人技术中，这主要是由于缺乏大规模训练数据以及真实世界任务的复杂性造成的。为了解决这一问题，许多机器人学习研究人员正努力将机器人大规模部署在像家庭这样未经结构化的日常环境中，以启动数据飞轮。当前的机器人学习系统在特定的短期任务中表现有效，但它们并不是为了在未经结构化的环境中自主长时间运行而设计的。本论文重点解决长时程运行中机器人面临的两个关键挑战：记忆和终身学习。

我们提出两种创新方法来推进这些能力。首先，我们引入了t-DGR（轨迹导向的深度生成重播方法），它在持续世界基准测试中达到了最先进的性能，促进了终身学习的发展。其次，我们开发了一个框架，利用人类示范来教会代理合理利用记忆，从而提高在记忆锻炼场任务中的学习效率和成功率。最后，我们讨论了实现机器人在真实世界中大规模运行所需的终身学习和记忆能力的未来方向。 

---
# Developing an Ontology for AI Act Fundamental Rights Impact Assessments 

**Title (ZH)**: 开发AI行为基本权利影响评估的本体论 

**Authors**: Tytti Rintamaki, Harshvardhan J. Pandit  

**Link**: [PDF](https://arxiv.org/pdf/2501.10391)  

**Abstract**: The recently published EU Artificial Intelligence Act (AI Act) is a landmark regulation that regulates the use of AI technologies. One of its novel requirements is the obligation to conduct a Fundamental Rights Impact Assessment (FRIA), where organisations in the role of deployers must assess the risks of their AI system regarding health, safety, and fundamental rights. Another novelty in the AI Act is the requirement to create a questionnaire and an automated tool to support organisations in their FRIA obligations. Such automated tools will require a machine-readable form of information involved within the FRIA process, and additionally also require machine-readable documentation to enable further compliance tools to be created. In this article, we present our novel representation of the FRIA as an ontology based on semantic web standards. Our work builds upon the existing state of the art, notably the Data Privacy Vocabulary (DPV), where similar works have been established to create tools for GDPR's Data Protection Impact Assessments (DPIA) and other obligations. Through our ontology, we enable the creation and management of FRIA, and the use of automated tool in its various steps. 

**Abstract (ZH)**: 最近公布的欧盟人工智能法案（AI Act）是一项里程碑式的法规，规范了人工智能技术的使用。该法案的一个新颖要求是强制执行基础权利影响评估（FRIA），要求在部署角色中的组织对其人工智能系统在健康、安全和基础权利方面的风险进行评估。AI法案的另一个创新之处是要求创建问卷和自动工具，以支持组织履行FRIA义务。这样的自动工具需要能够以机器可读的形式呈现FRIA过程中涉及的信息，并还需要机器可读的文档，以便创建进一步的合规工具。本文中，我们介绍了我们基于语义web标准的新颖表示FRIA的概念模式。我们的工作在此领域现有的先进技术之上进行构建，尤其是数据隐私词汇表（DPV），在此领域已经有一些研究表明，可以用于创建GDPR下的数据保护影响评估（DPIA）和其他合规要求的工具。通过我们的概念模式，我们能够创建和管理FRIA，并在FRIA的各个步骤中使用自动工具。 

---
# Towards an Environmental Ethics of Artificial Intelligence 

**Title (ZH)**: 朝着人工智能的环境伦理学方向迈进 

**Authors**: Nynke van Uffelen, Lode Lauwaert, Mark Coeckelbergh, Olya Kudina  

**Link**: [PDF](https://arxiv.org/pdf/2501.10390)  

**Abstract**: In recent years, much research has been dedicated to uncovering the environmental impact of Artificial Intelligence (AI), showing that training and deploying AI systems require large amounts of energy and resources, and the outcomes of AI may lead to decisions and actions that may negatively impact the environment. This new knowledge raises new ethical questions, such as: When is it (un)justifiable to develop an AI system, and how to make design choices, considering its environmental impact? However, so far, the environmental impact of AI has largely escaped ethical scrutiny, as AI ethics tends to focus strongly on themes such as transparency, privacy, safety, responsibility, and bias. Considering the environmental impact of AI from an ethical perspective expands the scope of AI ethics beyond an anthropocentric focus towards including more-than-human actors such as animals and ecosystems. This paper explores the ethical implications of the environmental impact of AI for designing AI systems by drawing on environmental justice literature, in which three categories of justice are distinguished, referring to three elements that can be unjust: the distribution of benefits and burdens (distributive justice), decision-making procedures (procedural justice), and institutionalized social norms (justice as recognition). Based on these tenets of justice, we outline criteria for developing environmentally just AI systems, given their ecological impact. 

**Abstract (ZH)**: 近年来，许多研究致力于揭示人工智能（AI）对环境的影响，表明训练和部署AI系统需要大量的能源和资源，AI的成果可能会导致可能对环境产生负面影响的决策和行为。这一新的知识引发了新的伦理问题，例如：何时开发AI系统是正当的或不正当的，如何在考虑其环境影响的情况下做出设计选择？然而，到目前为止，AI的环境影响尚未受到伦理审查，因为人工智能伦理往往侧重于透明度、隐私、安全、责任和偏见等主题。从伦理视角考虑AI的环境影响，将AI伦理的范围从人类中心主义转向包括非人类行动者如动物和生态系统。本文通过借鉴环境正义的文献，探讨AI对环境的影响在设计AI系统方面的伦理影响，在其中区分了三种类型的正义，分别指的是三种可能不公正的要素：利益和负担的分配不均（分配正义）、决策程序（程序正义）以及制度化的社会规范（认可正义）。基于这些正义原则，我们概述了开发环境正义AI系统的标准，考虑到其生态影响。 

---
# Beyond the Sum: Unlocking AI Agents Potential Through Market Forces 

**Title (ZH)**: 超越总和：通过市场力量解锁人工智能代理的潜力 

**Authors**: Jordi Montes Sanabria, Pol Alvarez Vecino  

**Link**: [PDF](https://arxiv.org/pdf/2501.10388)  

**Abstract**: The emergence of Large Language Models has fundamentally transformed the capabilities of AI agents, enabling a new class of autonomous agents capable of interacting with their environment through dynamic code generation and execution. These agents possess the theoretical capacity to operate as independent economic actors within digital markets, offering unprecedented potential for value creation through their distinct advantages in operational continuity, perfect replication, and distributed learning capabilities. However, contemporary digital infrastructure, architected primarily for human interaction, presents significant barriers to their participation.
This work presents a systematic analysis of the infrastructure requirements necessary for AI agents to function as autonomous participants in digital markets. We examine four key areas - identity and authorization, service discovery, interfaces, and payment systems - to show how existing infrastructure actively impedes agent participation. We argue that addressing these infrastructure challenges represents more than a technical imperative; it constitutes a fundamental step toward enabling new forms of economic organization. Much as traditional markets enable human intelligence to coordinate complex activities beyond individual capability, markets incorporating AI agents could dramatically enhance economic efficiency through continuous operation, perfect information sharing, and rapid adaptation to changing conditions. The infrastructure challenges identified in this work represent key barriers to realizing this potential. 

**Abstract (ZH)**: 大型语言模型的 emergence 引起了人工智能代理能力的根本性转变，使其能够通过动态代码生成和执行与环境进行交互，从而形成一类新的自主代理。这些代理具备理论上的能力，在数字市场中作为独立的经济主体运作，从而通过运营连续性、完美复制和分布式学习能力创造出前所未有的价值潜力。然而，当代主要为人类交互设计的数字基础设施为这些代理参与市场设定了显著障碍。

本文对使人工智能代理能够在数字市场中作为自主参与者运行所需的基础设施需求进行了系统性分析。我们考察了四个关键领域——身份认证与授权、服务发现、界面和支付系统——以展示现有基础设施如何主动阻碍代理的参与。我们认为，解决这些基础设施挑战不仅是技术上的要求，也是向新形式的经济组织迈出的关键一步。正如传统的市场利用人类智能协调超出个人能力范围的复杂活动一样，包含人工智能代理的市场通过持续运行、完美信息共享和快速适应变化条件，能够显著提升经济效率。本研究中识别出的基础设施挑战是实现这一潜力的关键障碍。 

---
# Autonomous Microscopy Experiments through Large Language Model Agents 

**Title (ZH)**: 通过大规模语言模型代理实现自主显微镜实验 

**Authors**: Indrajeet Mandal, Jitendra Soni, Mohd Zaki, Morten M. Smedskjaer, Katrin Wondraczek, Lothar Wondraczek, Nitya Nand Gosvami, N. M. Anoop Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2501.10385)  

**Abstract**: The emergence of large language models (LLMs) has accelerated the development of self-driving laboratories (SDLs) for materials research. Despite their transformative potential, current SDL implementations rely on rigid, predefined protocols that limit their adaptability to dynamic experimental scenarios across different labs. A significant challenge persists in measuring how effectively AI agents can replicate the adaptive decision-making and experimental intuition of expert scientists. Here, we introduce AILA (Artificially Intelligent Lab Assistant), a framework that automates atomic force microscopy (AFM) through LLM-driven agents. Using AFM as an experimental testbed, we develop AFMBench-a comprehensive evaluation suite that challenges AI agents based on language models like GPT-4o and GPT-3.5 to perform tasks spanning the scientific workflow: from experimental design to results analysis. Our systematic assessment shows that state-of-the-art language models struggle even with basic tasks such as documentation retrieval, leading to a significant decline in performance in multi-agent coordination scenarios. Further, we observe that LLMs exhibit a tendency to not adhere to instructions or even divagate to additional tasks beyond the original request, raising serious concerns regarding safety alignment aspects of AI agents for SDLs. Finally, we demonstrate the application of AILA on increasingly complex experiments open-ended experiments: automated AFM calibration, high-resolution feature detection, and mechanical property measurement. Our findings emphasize the necessity for stringent benchmarking protocols before deploying AI agents as laboratory assistants across scientific disciplines. 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起加速了材料研究中自动实验室（SDLs）的发展。尽管它们具有变革性潜力，当前的SDL实现依赖于僵硬的预定义协议，限制了其在不同实验室动态实验场景中的适应性。一个重大挑战在于，如何有效衡量人工智能代理能否复制专家科学家的适应性决策和实验直觉。在这里，我们介绍了一种名为AILA（Artificially Intelligent Lab Assistant）的框架，该框架通过基于LLM的代理自动化原子力显微镜（AFM）。使用AFM作为实验测试平台，我们开发了AFMBench——一个全面的评估套件，基于如GPT-4o和GPT-3.5等语言模型挑战AI代理完成涵盖整个科学工作流程的任务：从实验设计到结果分析。我们的系统评估表明，最先进的语言模型即使在基本任务如文献检索上也表现不佳，这在多代理协调场景中导致了显著的性能下降。此外，我们观察到LLMs表现出不遵守指令或甚至转向超出原始请求的额外任务的趋势，这引起了关于SDL中人工智能代理的安全对齐方面的严重关切。最后，我们展示了AILA在日益复杂的实验中的应用：全自动AFM校准、高分辨率特征检测和机械性能测量。我们的研究结果强调了在将AI代理应用于不同科学领域之前进行严格基准测试的必要性。 

---
# The Three Social Dimensions of Chatbot Technology 

**Title (ZH)**: 聊天机器人技术的三大社会维度 

**Authors**: Mauricio Figueroa-Torres  

**Link**: [PDF](https://arxiv.org/pdf/2501.10377)  

**Abstract**: The development and deployment of chatbot technology, while spanning decades and employing different techniques, require innovative frameworks to understand and interrogate their functionality and implications. A mere technocentric account of the evolution of chatbot technology does not fully illuminate how conversational systems are embedded in societal dynamics. This study presents a structured examination of chatbots across three societal dimensions, highlighting their roles as objects of scientific research, commercial instruments, and agents of intimate interaction. Through furnishing a dimensional framework for the evolution of conversational systems, from laboratories to marketplaces to private lives, this article contributes to the wider scholarly inquiry of chatbot technology and its impact in lived human experiences and dynamics. 

**Abstract (ZH)**: 聊天机器人技术的发展与部署虽然跨越了几十年，并采用了不同的技术，但仍需创新框架来理解和探讨其功能和影响。仅仅从技术中心的角度审视聊天机器人技术的历史演变，并不能充分阐明对话系统在社会动态中的嵌入方式。本研究通过从三个社会维度系统地考察聊天机器人的角色，强调它们作为科学研究的对象、商业工具及亲密互动代理的地位。通过提供一个从实验室到市场再到私人生活的对话系统演变框架，本文为聊天机器人技术及其在人类生活经验和社会动态中的影响的更广泛学术研究作出了贡献。 

---
# DK-PRACTICE: An Intelligent Educational Platform for Personalized Learning Content Recommendations Based on Students Knowledge State 

**Title (ZH)**: DK-PRACTICE：一种基于学生知识状态的个性化学习内容推荐智能教育平台 

**Authors**: Marina Delianidi, Konstantinos Diamantaras, Ioannis Moras, Antonis Sidiropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2501.10373)  

**Abstract**: This study introduces DK-PRACTICE (Dynamic Knowledge Prediction and Educational Content Recommendation System), an intelligent online platform that leverages machine learning to provide personalized learning recommendations based on student knowledge state. Students participate in a short, adaptive assessment using the question-and-answer method regarding key concepts in a specific knowledge domain. The system dynamically selects the next question for each student based on the correctness and accuracy of their previous answers. After the test is completed, DK-PRACTICE analyzes students' interaction history to recommend learning materials to empower the student's knowledge state in identified knowledge gaps. Both question selection and learning material recommendations are based on machine learning models trained using anonymized data from a real learning environment. To provide self-assessment and monitor learning progress, DK-PRACTICE allows students to take two tests: one pre-teaching and one post-teaching. After each test, a report is generated with detailed results. In addition, the platform offers functions to visualize learning progress based on recorded test statistics. DK-PRACTICE promotes adaptive and personalized learning by empowering students with self-assessment capabilities and providing instructors with valuable information about students' knowledge levels. DK-PRACTICE can be extended to various educational environments and knowledge domains, provided the necessary data is available according to the educational topics. A subsequent paper will present the methodology for the experimental application and evaluation of the platform. 

**Abstract (ZH)**: 本文介绍了一种基于机器学习的智能在线平台——DK-PRACTICE（动态知识预测与教育内容推荐系统），该平台能够根据学生的知识状态提供个性化的学习建议。学生将在特定知识领域内，通过问答方式进行一项简短且自适应的评估。系统会根据学生上一题的答案正确性和准确度动态选择下一道题目。测试结束后，DK-PRACTICE 会分析学生的历史交互记录，以推荐学习材料，填补学生在知识领域的盲区。问题的选择和学习材料推荐均基于使用匿名化真实学习环境数据训练的机器学习模型。

为了进行自我评估和监控学习进度，DK-PRACTICE 允许学生完成两次测试：一次教学前的测试和一次教学后的测试。每次测试后都会生成详细的报告。此外，该平台还提供了基于记录的测试统计数据可视化学习进度的功能。DK-PRACTICE 通过赋予学生自我评估能力以及为教师提供有关学生知识水平的宝贵信息，促进了适应性和个性化的学习。DK-PRACTICE 可以扩展到各种教育环境和知识领域，前提是根据教育主题需要有相关数据可用。后续的文章将展示该平台的实验应用和评估方法。 

---
# What we learned while automating bias detection in AI hiring systems for compliance with NYC Local Law 144 

**Title (ZH)**: 我们在自动化AI招聘系统中的偏见检测以遵守纽约市地方法第144条过程中学到的经验 

**Authors**: Gemma Galdon Clavell, Rubén González-Sendino  

**Link**: [PDF](https://arxiv.org/pdf/2501.10371)  

**Abstract**: Since July 5, 2023, New York City's Local Law 144 requires employers to conduct independent bias audits for any automated employment decision tools (AEDTs) used in hiring processes. The law outlines a minimum set of bias tests that AI developers and implementers must perform to ensure compliance. Over the past few months, we have collected and analyzed audits conducted under this law, identified best practices, and developed a software tool to streamline employer compliance. Our tool, ITACA_144, tailors our broader bias auditing framework to meet the specific requirements of Local Law 144. While automating these legal mandates, we identified several critical challenges that merit attention to ensure AI bias regulations and audit methodologies are both effective and practical. This document presents the insights gained from automating compliance with NYC Local Law 144. It aims to support other cities and states in crafting similar legislation while addressing the limitations of the NYC framework. The discussion focuses on key areas including data requirements, demographic inclusiveness, impact ratios, effective bias, metrics, and data reliability. 

**Abstract (ZH)**: 自2023年7月5日起，纽约市的地方法案144要求雇主对招聘过程中使用的任何自动化就业决策工具（AEDTs）进行独立偏见审计。该法案规定了人工智能开发者和实施者必须进行的一系列最小偏见测试，以确保合规。在过去几个月里，我们收集并分析了根据该法案进行的审计，识别最佳实践，并开发了一款软件工具以简化雇主的合规过程。我们的工具ITACA_144将我们更广泛的偏见审计框架修改为满足地方法案144的具体要求。在自动化这些法律规定的过程中，我们发现了一些需要特别关注的重要挑战，以确保AI偏见法规和审计方法既有效又实用。本文档呈现了自动化遵守纽约市地方法案144所获得的见解。它旨在支持其他城市和州制定类似立法，并解决纽约市框架的局限性。讨论重点包括数据需求、人口多样性、影响率、有效偏见、指标和数据可靠性等关键领域。 

---
# Harnessing Large Language Models for Mental Health: Opportunities, Challenges, and Ethical Considerations 

**Title (ZH)**: 利用大规模语言模型进行心理健康研究：机遇、挑战与伦理考虑 

**Authors**: Hari Mohan Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2501.10370)  

**Abstract**: Large Language Models (LLMs) are transforming mental health care by enhancing accessibility, personalization, and efficiency in therapeutic interventions. These AI-driven tools empower mental health professionals with real-time support, improved data integration, and the ability to encourage care-seeking behaviors, particularly in underserved communities. By harnessing LLMs, practitioners can deliver more empathetic, tailored, and effective support, addressing longstanding gaps in mental health service provision. However, their implementation comes with significant challenges and ethical concerns. Performance limitations, data privacy risks, biased outputs, and the potential for generating misleading information underscore the critical need for stringent ethical guidelines and robust evaluation mechanisms. The sensitive nature of mental health data further necessitates meticulous safeguards to protect patient rights and ensure equitable access to AI-driven care. Proponents argue that LLMs have the potential to democratize mental health resources, while critics warn of risks such as misuse and the diminishment of human connection in therapy. Achieving a balance between innovation and ethical responsibility is imperative. This paper examines the transformative potential of LLMs in mental health care, highlights the associated technical and ethical complexities, and advocates for a collaborative, multidisciplinary approach to ensure these advancements align with the goal of providing compassionate, equitable, and effective mental health support. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在通过增强心理健康服务的可及性、个性化和效率来改变心理健康护理领域。这些基于AI的工具为心理健康专业人员提供了实时支持、改进的数据整合能力，并能够促进寻求照顾行为，特别是在未充分服务的社区中。通过利用LLMs，实践者可以提供更具同理心、个性化和有效的支持，解决心理健康服务提供中存在的长期缺口。然而，它们的实施也伴随着重大挑战和伦理问题。性能限制、数据隐私风险、偏倚输出以及生成误导性信息的可能性突显了严格遵循伦理准则和建立坚实评估机制的必要性。精神健康数据的敏感性质进一步强化了对患者权利保护的重要性和确保AI驱动护理可及性的必要性。支持者认为LLMs有潜力普及精神健康资源，而批评者则警告存在诸如滥用以及治疗中人际关系减弱的风险。在创新与伦理责任之间取得平衡至关重要。本文考察了LLMs在心理健康护理中的变革潜力，指出了相关的技术与伦理复杂性，并倡导采用多学科协作的方法，以确保这些进步能够与提供有同情心、公平和有效精神健康支持的目标相一致。 

---
# Creative Loss: Ambiguity, Uncertainty and Indeterminacy 

**Title (ZH)**: 创意损失：歧义性、不确定性与不确定性 

**Authors**: Tom Holberton  

**Link**: [PDF](https://arxiv.org/pdf/2501.10369)  

**Abstract**: This article evaluates how creative uses of machine learning can address three adjacent terms: ambiguity, uncertainty and indeterminacy. Through the progression of these concepts it reflects on increasing ambitions for machine learning as a creative partner, illustrated with research from Unit 21 at the Bartlett School of Architecture, UCL. Through indeterminacy are potential future approaches to machine learning and design. 

**Abstract (ZH)**: 本文评估了机器学习创新应用如何应对三个相邻的概念：模糊性、不确定性和不定性。通过这些概念的发展进程，本文反思了对机器学习作为创意伙伴不断增加的期望，并以UCL巴特雷建筑学院Unit 21的研究为例进行说明。通过不定性，本文探讨了未来机器学习和设计的潜在方法。 

---
# The Potential of Answer Classes in Large-scale Written Computer-Science Exams -- Vol. 2 

**Title (ZH)**: 大规模书面计算机科学考试中答案类别潜力的研究——卷二 

**Authors**: Dominic Lohr, Marc Berges, Michael Kohlhase, Florian Rabe  

**Link**: [PDF](https://arxiv.org/pdf/2501.10368)  

**Abstract**: Students' answers to tasks provide a valuable source of information in teaching as they result from applying cognitive processes to a learning content addressed in the task. Due to steadily increasing course sizes, analyzing student answers is frequently the only means of obtaining evidence about student performance. However, in many cases, resources are limited, and when evaluating exams, the focus is solely on identifying correct or incorrect answers. This overlooks the value of analyzing incorrect answers, which can help improve teaching strategies or identify misconceptions to be addressed in the next cohort.
In teacher training for secondary education, assessment guidelines are mandatory for every exam, including anticipated errors and misconceptions. We applied this concept to a university exam with 462 students and 41 tasks. For each task, the instructors developed answer classes -- classes of expected responses, to which student answers were mapped during the exam correction process. The experiment resulted in a shift in mindset among the tutors and instructors responsible for the course: after initially having great reservations about whether the significant additional effort would yield an appropriate benefit, the procedure was subsequently found to be extremely valuable.
The concept presented, and the experience gained from the experiment were cast into a system with which it is possible to correct paper-based exams on the basis of answer classes. This updated version of the paper provides an overview and new potential in the course of using the digital version of the approach. 

**Abstract (ZH)**: 学生的作业答案为教学提供了一个宝贵的信息来源，因为这些答案是他们在完成任务时对学习内容进行认知处理的结果。随着课程规模的稳步扩大，分析学生答案通常是获取关于学生表现证据的唯一手段。然而，在许多情况下，资源有限，尤其是在评估考试时，关注点仅局限于识别正确或错误的答案。这忽视了分析错误答案的价值，错误答案可以帮助改进教学策略或识别需要在下一个学期内解决的错误观念。

在面向中等教育的教师培训中，评估指南是每项考试的必备内容，包括预期的错误和误解。我们将这一概念应用于包含462名学生和41项任务的大学考试。对于每个任务，教师制定了答案类别——预期响应类别，并在评分过程中将学生答案映射到这些类别。实验结果显示，负责该课程的教师和导师产生了心态上的转变：在最初对额外付出的巨大努力能否带来适当回报表示严重质疑后，发现该程序极其有价值。

本文所提出的概念及其从实验中获得的经验已纳入一个系统中，该系统可以在基于答案类别的基础上进行纸质考试的评分。本文更新版本提供了一个概述，并探讨了在使用数字版本的方法过程中所产生的新潜力。 

---
# GTDE: Grouped Training with Decentralized Execution for Multi-agent Actor-Critic 

**Title (ZH)**: GTDE: 分组训练与去中心化执行的多代理actor-critic方法 

**Authors**: Mengxian Li, Qi Wang, Yongjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10367)  

**Abstract**: The rapid advancement of multi-agent reinforcement learning (MARL) has given rise to diverse training paradigms to learn the policies of each agent in the multi-agent system. The paradigms of decentralized training and execution (DTDE) and centralized training with decentralized execution (CTDE) have been proposed and widely applied. However, as the number of agents increases, the inherent limitations of these frameworks significantly degrade the performance metrics, such as win rate, total reward, etc. To reduce the influence of the increasing number of agents on the performance metrics, we propose a novel training paradigm of grouped training decentralized execution (GTDE). This framework eliminates the need for a centralized module and relies solely on local information, effectively meeting the training requirements of large-scale multi-agent systems. Specifically, we first introduce an adaptive grouping module, which divides each agent into different groups based on their observation history. To implement end-to-end training, GTDE uses Gumbel-Sigmoid for efficient point-to-point sampling on the grouping distribution while ensuring gradient backpropagation. To adapt to the uncertainty in the number of members in a group, two methods are used to implement a group information aggregation module that merges member information within the group. Empirical results show that in a cooperative environment with 495 agents, GTDE increased the total reward by an average of 382\% compared to the baseline. In a competitive environment with 64 agents, GTDE achieved a 100\% win rate against the baseline. 

**Abstract (ZH)**: 多智能体强化学习（MARL）的快速发展催生了多种训练范式，以学习每个智能体在多智能体系统中的策略。分散训练与执行（DTDE）和集中训练与分散执行（CTDE）等范式已被提出并广泛应用于实际问题中。然而，随着智能体数量的增加，这些框架内部固有的局限性显著降低了性能指标（如胜率、总奖励等）。为了减少智能体数量增加对性能指标的影响，我们提出了一种新的训练范式——分组训练分散执行（GTDE）。该框架去除了集中模块的必要性，仅依赖于局部信息，有效满足大规模多智能体系统的训练需求。具体来说，我们首先引入了一种自适应分组模块，该模块根据每个智能体的观测历史将其分为不同的组。为实现端到端训练，GTDE使用Gumbel-Sigmoid进行高效的点到点采样，同时确保梯度反传。为了适应组内成员数量的不确定性，我们采用两种方法实现一组信息聚合模块，该模块可以将组内成员信息进行合并。实验结果表明，在包含495个智能体的合作环境中，与基线相比，GTDE的总奖励平均提高了382%；在包含64个智能体的竞赛环境中，GTDE实现了100%的胜率，战胜了基线。 

---
# Participatory Assessment of Large Language Model Applications in an Academic Medical Center 

**Title (ZH)**: 学术医疗中心参与评估大型语言模型应用 

**Authors**: Giorgia Carra, Bogdan Kulynych, François Bastardot, Daniel E. Kaufmann, Noémie Boillat-Blanco, Jean Louis Raisaro  

**Link**: [PDF](https://arxiv.org/pdf/2501.10366)  

**Abstract**: Although Large Language Models (LLMs) have shown promising performance in healthcare-related applications, their deployment in the medical domain poses unique challenges of ethical, regulatory, and technical nature. In this study, we employ a systematic participatory approach to investigate the needs and expectations regarding clinical applications of LLMs at Lausanne University Hospital, an academic medical center in Switzerland. Having identified potential LLM use-cases in collaboration with thirty stakeholders, including clinical staff across 11 departments as well nursing and patient representatives, we assess the current feasibility of these use-cases taking into account the regulatory frameworks, data protection regulation, bias, hallucinations, and deployment constraints. This study provides a framework for a participatory approach to identifying institutional needs with respect to introducing advanced technologies into healthcare practice, and a realistic analysis of the technology readiness level of LLMs for medical applications, highlighting the issues that would need to be overcome LLMs in healthcare to be ethical, and regulatory compliant. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在医疗健康相关应用中展现出了令人鼓舞的性能，其在医疗领域的部署面临着伦理、监管和技术等方面的独特挑战。本研究采用系统参与性的方法，调查了洛桑大学医院（瑞士的一家学术医疗中心）关于临床应用中对LLMs的需求和期望。在与包括11个部门临床人员以及护理人员和患者代表在内的三十名相关方合作中，我们识别了潜在的LLM应用场景，并考虑了监管框架、数据保护法规、偏差、幻觉以及部署限制等因素，评估了这些应用场景的当前可行性。本研究为识别医疗机构在引入先进科技到医疗实践过程中所需的需求提供了一个框架，并对LLMs在医疗应用中的技术成熟度进行了现实分析，突显了确保LLMs在医疗环境中既具有伦理性和又符合监管要求所需要克服的问题。 

---
# Can LLMs Identify Gaps and Misconceptions in Students' Code Explanations? 

**Title (ZH)**: 大语言模型能否识别学生代码解释中的漏洞与误解？ 

**Authors**: Priti Oli, Rabin Banjade, Andrew M. Olney, Vasile Rus  

**Link**: [PDF](https://arxiv.org/pdf/2501.10365)  

**Abstract**: This paper investigates various approaches using Large Language Models (LLMs) to identify gaps and misconceptions in students' self-explanations of specific instructional material, in our case explanations of code examples. This research is a part of our larger effort to automate the assessment of students' freely generated responses, focusing specifically on their self-explanations of code examples during activities related to code comprehension. In this work, we experiment with zero-shot prompting, Supervised Fine-Tuning (SFT), and preference alignment of LLMs to identify gaps in students' self-explanation. With simple prompting, GPT-4 consistently outperformed LLaMA3 and Mistral in identifying gaps and misconceptions, as confirmed by human evaluations. Additionally, our results suggest that fine-tuned large language models are more effective at identifying gaps in students' explanations compared to zero-shot and few-shot prompting techniques. Furthermore, our findings show that the preference optimization approach using Odds Ratio Preference Optimization (ORPO) outperforms SFT in identifying gaps and misconceptions in students' code explanations. 

**Abstract (ZH)**: 本文探讨了使用大型语言模型（LLMs）的各种方法，以识别学生在解释特定教学材料（例如代码示例）时存在的差距和误解。本研究是我们在更大范围内自动评估学生自由生成的回答的一部分，特别是在编码理解活动过程中，专注于识别学生对代码示例的自我解释中存在的差距。在本工作中，我们尝试使用零样本提示、监督微调（SFT）和对LLMs进行偏好对齐，以识别学生自我解释中的差距。通过简单的提示，GPT-4在识别差距和误解方面始终优于LLaMA3和Mistral，并经人工评估证实。此外，我们的结果表明，微调后的大型语言模型在识别学生解释中的差距方面比零样本和少样本提示技术更有效。此外，我们的发现显示，使用Odds Ratio Preference Optimization（ORPO）进行偏好评价的方法在识别学生代码解释中的差距和误解方面优于SFT。 

---
# Integrating Artificial Open Generative Artificial Intelligence into Software Supply Chain Security 

**Title (ZH)**: 将人工开放生成型人工智能集成到软件供应链安全中 

**Authors**: Vasileios Alevizos, George A Papakostas, Akebu Simasiku, Dimitra Malliarou, Antonis Messinis, Sabrina Edralin, Clark Xu, Zongliang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2412.19088)  

**Abstract**: While new technologies emerge, human errors always looming. Software supply chain is increasingly complex and intertwined, the security of a service has become paramount to ensuring the integrity of products, safeguarding data privacy, and maintaining operational continuity. In this work, we conducted experiments on the promising open Large Language Models (LLMs) into two main software security challenges: source code language errors and deprecated code, with a focus on their potential to replace conventional static and dynamic security scanners that rely on predefined rules and patterns. Our findings suggest that while LLMs present some unexpected results, they also encounter significant limitations, particularly in memory complexity and the management of new and unfamiliar data patterns. Despite these challenges, the proactive application of LLMs, coupled with extensive security databases and continuous updates, holds the potential to fortify Software Supply Chain (SSC) processes against emerging threats. 

**Abstract (ZH)**: 随着新技术的涌现，人为错误始终存在。软件供应链日益复杂且相互交织，服务的安全性已成为确保产品完整性、保护数据隐私和维持运营连续性的关键因素。在本研究中，我们对有前景的开放型大型语言模型（LLMs）进行了实验，重点关注它们在两大主要软件安全挑战中的应用：源代码语言错误和过时代码。我们的研究集中在LLMs是否能够替代依赖预定义规则和模式的传统静态和动态安全扫描器。研究发现，尽管LLMs展示出了一些意料之外的结果，但它们也面临显著的局限性，特别是在内存复杂性和管理新且不熟悉的模式方面。尽管存在这些挑战，积极应用LLMs，结合广泛的安全数据库和持续更新，仍有可能增强软件供应链（SSC）过程，以抵御新兴的安全威胁。 

---
# Disentangled Interpretable Representation for Efficient Long-term Time Series Forecasting 

**Title (ZH)**: 高效长周期时间序列预测的解耦可解释表示方法 

**Authors**: Yuang Zhao, Tianyu Li, Jiadong Chen, Shenrong Ye, Fuxin Jiang, Tieying Zhang, Xiaofeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2411.17257)  

**Abstract**: Industry 5.0 introduces new challenges for Long-term Time Series Forecasting (LTSF), characterized by high-dimensional, high-resolution data and high-stakes application scenarios. Against this backdrop, developing efficient and interpretable models for LTSF becomes a key challenge. Existing deep learning and linear models often suffer from excessive parameter complexity and lack intuitive interpretability. To address these issues, we propose DiPE-Linear, a Disentangled interpretable Parameter-Efficient Linear network. DiPE-Linear incorporates three temporal components: Static Frequential Attention (SFA), Static Temporal Attention (STA), and Independent Frequential Mapping (IFM). These components alternate between learning in the frequency and time domains to achieve disentangled interpretability. The decomposed model structure reduces parameter complexity from quadratic in fully connected networks (FCs) to linear and computational complexity from quadratic to log-linear. Additionally, a Low-Rank Weight Sharing policy enhances the model's ability to handle multivariate series. Despite operating within a subspace of FCs with limited expressive capacity, DiPE-Linear demonstrates comparable or superior performance to both FCs and nonlinear models across multiple open-source and real-world LTSF datasets, validating the effectiveness of its sophisticatedly designed structure. The combination of efficiency, accuracy, and interpretability makes DiPE-Linear a strong candidate for advancing LTSF in both research and real-world applications. The source code is available at this https URL. 

**Abstract (ZH)**: 工业4.0（Industry 4.0）为长期时间序列预测（Long-term Time Series Forecasting, LTSF）带来了新的挑战，这些挑战表现为高维、高分辨率的数据以及高风险的应用场景。在这个背景下，开发高效且易于解释的LTSF模型成为一个关键挑战。现有的深度学习和线性模型往往因参数复杂度过高且缺乏直观的可解释性而受到限制。为了解决这些问题，我们提出了一种分离可解释参数高效线性网络DiPE-Linear。DiPE-Linear融合了三个时间组件：静态频率注意（SFA）、静态时间注意（STA）和独立频率映射（IFM）。这些组件在频率域和时间域之间交替学习，以实现分离的可解释性。分解后的模型结构将参数复杂度从完全连接网络（Fully Connected Networks, FCs）的二次复杂度降低到线性复杂度，计算复杂度从二次复杂度降低到对数线性复杂度。此外，低秩权重共享策略增强了模型处理多变量序列的能力。尽管在具有有限表达能力的FC子空间内操作，DiPE-Linear在多个开源和实际世界LTSF数据集上的表现与FCs和非线性模型相比具有可比性甚至更优，这验证了其精心设计结构的有效性。DiPE-Linear在效率、准确性和可解释性的结合使其成为LTSF研究和实际应用的有力候选者。源代码可在以下链接获取：[此链接]。 

---
# On the Impact of Black-box Deployment Strategies for Edge AI on Latency and Model Performance 

**Title (ZH)**: 黑盒部署策略对边缘AI延迟和模型性能的影响分析 

**Authors**: Jaskirat Singh, Bram Adams, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2403.17154)  

**Abstract**: Deciding what combination of operators to use across the Edge AI tiers to achieve specific latency and model performance requirements is an open question for MLOps engineers. This study aims to empirically assess the accuracy vs inference time trade-off of different black-box Edge AI deployment strategies, i.e., combinations of deployment operators and deployment tiers. In this paper, we conduct inference experiments involving 3 deployment operators (i.e., Partitioning, Quantization, Early Exit), 3 deployment tiers (i.e., Mobile, Edge, Cloud) and their combinations on four widely used Computer-Vision models to investigate the optimal strategies from the point of view of MLOps developers. Our findings suggest that Edge deployment using the hybrid Quantization + Early Exit operator could be preferred over non-hybrid operators (Quantization/Early Exit on Edge, Partition on Mobile-Edge) when faster latency is a concern at medium accuracy loss. However, when minimizing accuracy loss is a concern, MLOps engineers should prefer using only a Quantization operator on edge at a latency reduction or increase, respectively over the Early Exit/Partition (on edge/mobile-edge) and Quantized Early Exit (on edge) operators. In scenarios constrained by Mobile CPU/RAM resources, a preference for Partitioning across mobile and edge tiers is observed over mobile deployment. For models with smaller input data samples (such as FCN), a network-constrained cloud deployment can also be a better alternative than Mobile/Edge deployment and Partitioning strategies. For models with large input data samples (ResNet, ResNext, DUC), an edge tier having higher network/computational capabilities than Cloud/Mobile can be a more viable option than Partitioning and Mobile/Cloud deployment strategies. 

**Abstract (ZH)**: 确定在边缘AI层级中使用哪种操作组合以满足特定的延迟和模型性能要求是MLOps工程师面临的一个开放问题。本研究旨在通过实证评估不同黑盒边缘AI部署策略的准确率与推理时间的权衡，即不同的部署操作组合和部署层级。在本文中，我们针对四种广泛使用的计算机视觉模型，开展了包括3种部署操作（分区、量化、提前退出）和3种部署层级（移动设备、边缘设备、云端）及其组合在内的推理实验，从MLOps开发者的视角探讨最优策略。我们的研究发现表明，当注重中等准确率损失下的快速延迟时，可以选择混合量化+提前退出操作进行边缘部署，而非选择仅在边缘设备上进行量化或分区，或仅在移动边缘设备上进行提前退出。如果关注最小化准确率损失，MLOps工程师应倾向于在边缘设备上仅使用量化操作，从而在延迟减少或增加的情况下，优于提前退出/分区（在边缘设备/移动边缘设备上）和混合量化提前退出（在边缘设备上）。在受移动CPU/内存资源限制的场景中，观察到在移动设备和边缘设备层级上更倾向于使用分区策略。对于输入数据样本较小的模型（如FCN），网络受限的云端部署可能比移动设备/边缘设备部署和分区策略更为合适。而对于输入数据样本较大的模型（如ResNet、ResNext、DUC），与分区策略和移动设备/云端部署相比，具有更高网络/计算能力的边缘设备层级可能是更可行的选择。 

---
