# A Survey on LLM-powered Agents for Recommender Systems 

**Title (ZH)**: 基于大型语言模型的推荐系统代理综述 

**Authors**: Qiyao Peng, Hongtao Liu, Hua Huang, Qing Yang, Minglai Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10050)  

**Abstract**: Recommender systems are essential components of many online platforms, yet traditional approaches still struggle with understanding complex user preferences and providing explainable recommendations. The emergence of Large Language Model (LLM)-powered agents offers a promising approach by enabling natural language interactions and interpretable reasoning, potentially transforming research in recommender systems. This survey provides a systematic review of the emerging applications of LLM-powered agents in recommender systems. We identify and analyze three key paradigms in current research: (1) Recommender-oriented approaches, which leverage intelligent agents to enhance the fundamental recommendation mechanisms; (2) Interaction-oriented approaches, which facilitate dynamic user engagement through natural dialogue and interpretable suggestions; and (3) Simulation-oriented approaches, which employ multi-agent frameworks to model complex user-item interactions and system dynamics. Beyond paradigm categorization, we analyze the architectural foundations of LLM-powered recommendation agents, examining their essential components: profile construction, memory management, strategic planning, and action execution. Our investigation extends to a comprehensive analysis of benchmark datasets and evaluation frameworks in this domain. This systematic examination not only illuminates the current state of LLM-powered agent recommender systems but also charts critical challenges and promising research directions in this transformative field. 

**Abstract (ZH)**: 推荐系统是许多在线平台的关键组成部分，尽管传统的推荐方法仍在努力理解和提供解释性强的个性化推荐。得益于大型语言模型（LLM）驱动代理的出现，通过实现自然语言互动和可解释的推理，推荐系统研究有望迎来新的突破。本文综述了LLM驱动代理在推荐系统中的新兴应用。我们确定并分析了当前研究中的三大核心范式：（1）推荐导向的方法，利用智能代理来增强基础的推荐机制；（2）交互导向的方法，通过自然对话和可解释的推荐来促进动态用户参与；以及（3）仿真导向的方法，通过多代理框架来建模复杂的用户-项目交互和系统动态。除范式分类外，我们还分析了LLM驱动推荐代理的架构基础，考察了其关键组件：档案构建、内存管理、策略规划和行动执行。研究还扩展至对该领域基准数据集和评估框架的全面分析。这种系统性的考察不仅揭示了LLM驱动的代理推荐系统当前的状态，还绘制出了该变革领域中的关键挑战和充满希望的研究方向。 

---
# STMA: A Spatio-Temporal Memory Agent for Long-Horizon Embodied Task Planning 

**Title (ZH)**: STMA：一种用于长时 horizon 体态任务规划的空间-时间记忆代理 

**Authors**: Mingcong Lei, Yiming Zhao, Ge Wang, Zhixin Mai, Shuguang Cui, Yatong Han, Jinke Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.10177)  

**Abstract**: A key objective of embodied intelligence is enabling agents to perform long-horizon tasks in dynamic environments while maintaining robust decision-making and adaptability. To achieve this goal, we propose the Spatio-Temporal Memory Agent (STMA), a novel framework designed to enhance task planning and execution by integrating spatio-temporal memory. STMA is built upon three critical components: (1) a spatio-temporal memory module that captures historical and environmental changes in real time, (2) a dynamic knowledge graph that facilitates adaptive spatial reasoning, and (3) a planner-critic mechanism that iteratively refines task strategies. We evaluate STMA in the TextWorld environment on 32 tasks, involving multi-step planning and exploration under varying levels of complexity. Experimental results demonstrate that STMA achieves a 31.25% improvement in success rate and a 24.7% increase in average score compared to the state-of-the-art model. The results highlight the effectiveness of spatio-temporal memory in advancing the memory capabilities of embodied agents. 

**Abstract (ZH)**: 本篇论文的关键目标在于使智能体能够在动态环境中执行长期任务，同时保持稳健的决策能力和适应性。为了实现这一目标，我们提出了一种新型框架——时空记忆智能体（STMA），该框架通过整合时空记忆来增强任务规划与执行能力。STMA基于三个关键组件构建：（1）时空记忆模块，能够实时捕捉历史和环境变化；（2）动态知识图谱，以促进适应性空间推理；（3）计划者-批判者机制，用于迭代优化任务策略。我们在包含32项任务的TextWorld环境中对STMA进行了评估，这些任务涉及复杂程度不同的多步规划与探索。实验结果表明，STMA相比当前最先进的模型，在成功率上提升了31.25%，平均得分提高了24.7%。这些结果突显了时空记忆在提升实体智能体记忆能力方面的有效性。 

---
# Cooperative Multi-Agent Planning with Adaptive Skill Synthesis 

**Title (ZH)**: 具有自适应技能合成的协作多智能体规划 

**Authors**: Zhiyuan Li, Wenshuai Zhao, Joni Pajarinen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10148)  

**Abstract**: Despite much progress in training distributed artificial intelligence (AI), building cooperative multi-agent systems with multi-agent reinforcement learning (MARL) faces challenges in sample efficiency, interpretability, and transferability. Unlike traditional learning-based methods that require extensive interaction with the environment, large language models (LLMs) demonstrate remarkable capabilities in zero-shot planning and complex reasoning. However, existing LLM-based approaches heavily rely on text-based observations and struggle with the non-Markovian nature of multi-agent interactions under partial observability. We present COMPASS, a novel multi-agent architecture that integrates vision-language models (VLMs) with a dynamic skill library and structured communication for decentralized closed-loop decision-making. The skill library, bootstrapped from demonstrations, evolves via planner-guided tasks to enable adaptive strategies. COMPASS propagates entity information through multi-hop communication under partial observability. Evaluations on the improved StarCraft Multi-Agent Challenge (SMACv2) demonstrate COMPASS achieves up to 30\% higher win rates than state-of-the-art MARL algorithms in symmetric scenarios. 

**Abstract (ZH)**: 尽管在训练分布式人工智能（AI）方面取得了显著进展，但使用多智能体强化学习（MARL）构建合作多智能体系统仍面临样本效率、可解释性和迁移性方面的挑战。不同于传统的基于学习的方法需要与环境进行大量的互动，大型语言模型（LLMs）展示了在零-shot 规划和复杂推理方面的卓越能力。然而，现有的基于LLM的方法主要依赖于基于文本的观测，并且在处理部分可观测环境下非马尔可夫性多智能体交互时存在困难。我们提出了一种名为COMPASS的新型多智能体架构，该架构将视觉语言模型（VLMs）与动态技能库和结构化通信结合，以实现去中心化的闭环决策。技能库通过演示初始化，并通过规划者指导的任务逐步进化，从而启用适应性策略。在部分可观测情况下，COMPASS通过多跳通信传播实体信息。在改进的星际争霸多智能体挑战（SMACv2）上的评估结果表明，COMPASS在对称场景中可实现比现有最先进的MARL算法高达30%的胜率。 

---
# Causal Information Prioritization for Efficient Reinforcement Learning 

**Title (ZH)**: 因果信息优先级排序以实现高效的强化学习 

**Authors**: Hongye Cao, Fan Feng, Tianpei Yang, Jing Huo, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10097)  

**Abstract**: Current Reinforcement Learning (RL) methods often suffer from sample-inefficiency, resulting from blind exploration strategies that neglect causal relationships among states, actions, and rewards. Although recent causal approaches aim to address this problem, they lack grounded modeling of reward-guided causal understanding of states and actions for goal-orientation, thus impairing learning efficiency. To tackle this issue, we propose a novel method named Causal Information Prioritization (CIP) that improves sample efficiency by leveraging factored MDPs to infer causal relationships between different dimensions of states and actions with respect to rewards, enabling the prioritization of causal information. Specifically, CIP identifies and leverages causal relationships between states and rewards to execute counterfactual data augmentation to prioritize high-impact state features under the causal understanding of the environments. Moreover, CIP integrates a causality-aware empowerment learning objective, which significantly enhances the agent's execution of reward-guided actions for more efficient exploration in complex environments. To fully assess the effectiveness of CIP, we conduct extensive experiments across 39 tasks in 5 diverse continuous control environments, encompassing both locomotion and manipulation skills learning with pixel-based and sparse reward settings. Experimental results demonstrate that CIP consistently outperforms existing RL methods across a wide range of scenarios. 

**Abstract (ZH)**: 当前的强化学习（RL）方法往往遭受样本效率低的困扰，这主要是由于盲目的探索策略未能考虑到状态、动作和奖励之间的因果关系。尽管最近的一些因果方法旨在解决这一问题，但它们缺乏对状态和动作的一贯奖励导向因果理解建模，从而影响了学习效率。为了解决这一问题，我们提出了一种名为因果信息优先级（CIP）的新方法，通过利用事实上的MDPs来推断不同维度的状态和动作与奖励之间的因果关系，从而提高样本效率。具体来说，CIP 识别并利用状态和奖励之间的因果关系执行反事实数据增强，优先考虑在因果环境中具有高影响的状态特征。此外，CIP 整合了一种因果意识的自增强学习目标，这显著增强了代理在复杂环境中基于奖励导向动作执行能力，从而实现更有效的探索。为了全面评估CIP的有效性，我们在5个不同类型的连续控制环境中进行了广泛的实验，包括39个独立任务，涵盖带有像素基和稀疏奖励设置的运动和操作技能学习。实验结果表明，CIP 在广泛的情境下始终优于现有的RL方法。 

---
# Towards Empowerment Gain through Causal Structure Learning in Model-Based RL 

**Title (ZH)**: 通过基于模型的强化学习中的因果结构学习实现赋能增益 

**Authors**: Hongye Cao, Fan Feng, Meng Fang, Shaokang Dong, Tianpei Yang, Jing Huo, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10077)  

**Abstract**: In Model-Based Reinforcement Learning (MBRL), incorporating causal structures into dynamics models provides agents with a structured understanding of the environments, enabling efficient decision. Empowerment as an intrinsic motivation enhances the ability of agents to actively control their environments by maximizing the mutual information between future states and actions. We posit that empowerment coupled with causal understanding can improve controllability, while enhanced empowerment gain can further facilitate causal reasoning in MBRL. To improve learning efficiency and controllability, we propose a novel framework, Empowerment through Causal Learning (ECL), where an agent with the awareness of causal dynamics models achieves empowerment-driven exploration and optimizes its causal structure for task learning. Specifically, ECL operates by first training a causal dynamics model of the environment based on collected data. We then maximize empowerment under the causal structure for exploration, simultaneously using data gathered through exploration to update causal dynamics model to be more controllable than dense dynamics model without causal structure. In downstream task learning, an intrinsic curiosity reward is included to balance the causality, mitigating overfitting. Importantly, ECL is method-agnostic and is capable of integrating various causal discovery methods. We evaluate ECL combined with 3 causal discovery methods across 6 environments including pixel-based tasks, demonstrating its superior performance compared to other causal MBRL methods, in terms of causal discovery, sample efficiency, and asymptotic performance. 

**Abstract (ZH)**: 在基于模型的强化学习（Model-Based Reinforcement Learning, MBRL）中，将因果结构纳入动力模型中为智能体提供了对环境的结构化理解，从而提高了决策效率。内在动机中的支配力（empowerment）增强了智能体主动控制环境的能力，通过最大化未来状态与动作之间的互信息来实现。我们认为，结合因果理解的支配力能够提高控制能力，进一步增强的支配力收益可以促进基于模型的强化学习（MBRL）中的因果推理。为了提高学习效率和控制能力，我们提出了一种新的框架——因果学习驱动的支配力（Empowerment through Causal Learning, ECL），其中具有因果动力模型意识的智能体实现支配力驱动的探索，并优化其因果结构以服务于任务学习。具体而言，ECL 首先基于采集的数据训练环境的因果动力模型。然后，在因果结构下最大化支配力以实现探索。同时，通过探索收集的数据更新因果动力模型，使其比没有因果结构的密集动力模型更具控制性。在下游任务学习中，我们引入内在的好奇心奖励以平衡因果关系，防止过拟合。重要的是，ECL 是方法通用的，并能与各种因果发现方法集成。我们评估了 ECL 结合三种因果发现方法在六个环境中的性能，包括基于像素的任务，结果表明，ECL 在因果发现、样本效率和渐近性能方面优于其他基于模型的因果强化学习方法。 

---
# The Ann Arbor Architecture for Agent-Oriented Programming 

**Title (ZH)**: 安阿伯代理导向编程架构 

**Authors**: Wei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.09903)  

**Abstract**: In this paper, we reexamine prompt engineering for large language models through the lens of automata theory. We argue that language models function as automata and, like all automata, should be programmed in the languages they accept, a unified collection of all natural and formal languages. Therefore, traditional software engineering practices--conditioned on the clear separation of programming languages and natural languages--must be rethought. We introduce the Ann Arbor Architecture, a conceptual framework for agent-oriented programming of language models, as a higher-level abstraction over raw token generation, and provide a new perspective on in-context learning. Based on this framework, we present the design of our agent platform Postline, and report on our initial experiments in agent training. 

**Abstract (ZH)**: 在本文中，我们通过自动机理论的视角重新审视大型语言模型的提示工程。我们认为语言模型本质上是自动机，并且应该像所有自动机一样，在它们能接受的统一语言集合（包括所有自然语言和形式语言）中进行编程。因此，传统的软件工程实践——基于编程语言和自然语言之间的明确区分——必须进行重新思考。我们提出了安阿伯架构（Ann Arbor Architecture），这是一种高层抽象的代理导向编程框架，用于语言模型，超越了原始词汇生成，并为上下文学习提供了新的视角。基于这一框架，我们设计了我们的代理平台Postline，并报告了初步的代理训练实验。 

---
# MuDoC: An Interactive Multimodal Document-grounded Conversational AI System 

**Title (ZH)**: MuDoC：一种交互式多模态文档导向对话AI系统 

**Authors**: Karan Taneja, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2502.09843)  

**Abstract**: Multimodal AI is an important step towards building effective tools to leverage multiple modalities in human-AI communication. Building a multimodal document-grounded AI system to interact with long documents remains a challenge. Our work aims to fill the research gap of directly leveraging grounded visuals from documents alongside textual content in documents for response generation. We present an interactive conversational AI agent 'MuDoC' based on GPT-4o to generate document-grounded responses with interleaved text and figures. MuDoC's intelligent textbook interface promotes trustworthiness and enables verification of system responses by allowing instant navigation to source text and figures in the documents. We also discuss qualitative observations based on MuDoC responses highlighting its strengths and limitations. 

**Abstract (ZH)**: 多模态AI是朝着通过利用人类与AI通信中多种模态构建有效工具迈出的重要一步。构建一个能够与长文档交互的多模态文档导向型AI系统仍然是一项挑战。我们旨在填补直接利用文档中的目标视觉内容与文本内容进行响应生成这一研究空白。我们基于GPT-4o提出了一个名为“MuDoC”的互动对话AI代理，能够生成交织文本与图表的文档导向型响应。MuDoC的智能教科书界面增强了系统的可信度，并允许用户即时导航到文档中的源文本和图表进行验证。我们还基于MuDoC的响应进行了定性的观察，以突出其优势和局限性。 

---
# BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds 

**Title (ZH)**: BeamDojo：在稀疏支撑点上学习灵活的人形运动 

**Authors**: Huayi Wang, Zirui Wang, Junli Ren, Qingwei Ben, Tao Huang, Weinan Zhang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10363)  

**Abstract**: Traversing risky terrains with sparse footholds poses a significant challenge for humanoid robots, requiring precise foot placements and stable locomotion. Existing approaches designed for quadrupedal robots often fail to generalize to humanoid robots due to differences in foot geometry and unstable morphology, while learning-based approaches for humanoid locomotion still face great challenges on complex terrains due to sparse foothold reward signals and inefficient learning processes. To address these challenges, we introduce BeamDojo, a reinforcement learning (RL) framework designed for enabling agile humanoid locomotion on sparse footholds. BeamDojo begins by introducing a sampling-based foothold reward tailored for polygonal feet, along with a double critic to balancing the learning process between dense locomotion rewards and sparse foothold rewards. To encourage sufficient trail-and-error exploration, BeamDojo incorporates a two-stage RL approach: the first stage relaxes the terrain dynamics by training the humanoid on flat terrain while providing it with task terrain perceptive observations, and the second stage fine-tunes the policy on the actual task terrain. Moreover, we implement a onboard LiDAR-based elevation map to enable real-world deployment. Extensive simulation and real-world experiments demonstrate that BeamDojo achieves efficient learning in simulation and enables agile locomotion with precise foot placement on sparse footholds in the real world, maintaining a high success rate even under significant external disturbances. 

**Abstract (ZH)**: 在稀疏脚着地点的危险地形上行走对类人机器人构成了重大挑战，要求其具备精确的足部放置和稳定的运动能力。现有的针对四足机器人设计的方法往往由于脚部几何形状和不稳定形态的不同而无法泛化到类人机器人，而基于学习的方法在复杂地形上进行类人行走时，仍然面临因稀疏脚着地点奖励信号和低效学习过程带来的巨大挑战。为应对这些挑战，我们引入了**BeamDojo**，这是一个旨在使类人机器人在稀疏脚着地点上实现敏捷运动的强化学习（RL）框架。BeamDojo 通过引入一种针对多边形脚设计的采样基础脚着地点奖励，以及引入双重评论家来平衡密集运动奖励和稀疏脚着地点奖励之间的学习过程。为了促进充分的试探性探索，BeamDojo 实现了一种两阶段的RL方法：第一阶段通过在平坦地形上训练类人而在同时提供环境地形感知信息来放松地形动力学，第二阶段在实际任务地形上微调决策策略。此外，我们还实现了基于机载LiDAR的高程图以实现现实世界的部署。广泛的仿真实验和实地测试表明，BeamDojo 在仿真中实现了高效学习，并且在实际世界中能够实现精确足部放置的敏捷运动，即使在显著的外界干扰下也能保持较高的成功率。 

---
# Process Reward Models for LLM Agents: Practical Framework and Directions 

**Title (ZH)**: 面向LLM代理的过程奖励模型：实用框架与发展方向 

**Authors**: Sanjiban Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2502.10325)  

**Abstract**: We introduce Agent Process Reward Models (AgentPRM), a simple and scalable framework for training LLM agents to continually improve through interactions. AgentPRM follows a lightweight actor-critic paradigm, using Monte Carlo rollouts to compute reward targets and optimize policies. It requires minimal modifications to existing RLHF pipelines, making it easy to integrate at scale. Beyond AgentPRM, we propose InversePRM, which learns process rewards directly from demonstrations without explicit outcome supervision. We also explore key challenges and opportunities, including exploration, process reward shaping, and model-predictive reasoning. We evaluate on ALFWorld benchmark, show that small 3B models trained with AgentPRM and InversePRM outperform strong GPT-4o baselines, and analyze test-time scaling, reward hacking, and more. Our code is available at: this https URL. 

**Abstract (ZH)**: 我们引入了代理过程奖励模型（AgentPRM）——一种简洁且可扩展的框架，用于训练大规模语言模型代理通过交互不断改进。AgentPRM 采用轻量级的演员-批评家范式，并利用蒙特卡洛展开来计算奖励目标并优化策略。它只需对现有的 RLHF 流程进行最少的修改，使其易于大规模集成。除了 AgentPRM，我们还提出了逆向过程奖励模型（InversePRM），该模型直接从演示学习过程奖励，而无需显式的结果监督。我们还探索了关键的挑战和机会，包括探索、过程奖励塑造和模型预测推理。我们使用 ALFWorld 基准进行评估，并展示了使用 AgentPRM 和 InversePRM 训练的小型 3B 模型超越了强大的 GPT-4o 基线模型的结果。我们分析了测试时缩放、奖励劫持等问题，并提供了更多细节。我们的代码可从以下链接获取：[提供链接]

翻译如下：

我们引入了代理过程奖励模型(AgentPRM)，这是一种简单且可扩展的框架，用于通过交互训练大规模语言模型代理使其不断改进。AgentPRM 采用轻量级的演员-批评家范式，使用蒙特卡洛展开计算奖励目标并优化策略。它只需要对现有的 RLHF 流程进行最小的修改，使其在大规模部署中易于集成。除了 AgentPRM，我们还提出了逆过程奖励模型(InversePRM)，它直接从演示中学习过程奖励，而无需显式的结果监督。我们还探索了关键的挑战和机遇，包括探索、过程奖励塑造和模型预测推理。我们使用 ALFWorld 基准对这些模型进行了评估，结果表明使用 AgentPRM 和 InversePRM 训练的小型 3B 模型优于强大的 GPT-4o 基线。我们分析了测试时扩展、奖励劫持等问题，并进行了更多的探讨。我们的代码库可以在以下链接处获得：[提供链接] 

---
# A Multiagent Path Search Algorithm for Large-Scale Coalition Structure Generation 

**Title (ZH)**: 大规模合作结构生成的多智能体路径搜索算法 

**Authors**: Redha Taguelmimt, Samir Aknine, Djamila Boukredera, Narayan Changder, Tuomas Sandholm  

**Link**: [PDF](https://arxiv.org/pdf/2502.10226)  

**Abstract**: Coalition structure generation (CSG), i.e. the problem of optimally partitioning a set of agents into coalitions to maximize social welfare, is a fundamental computational problem in multiagent systems. This problem is important for many applications where small run times are necessary, including transportation and disaster response. In this paper, we develop SALDAE, a multiagent path finding algorithm for CSG that operates on a graph of coalition structures. Our algorithm utilizes a variety of heuristics and strategies to perform the search and guide it. It is an anytime algorithm that can handle large problems with hundreds and thousands of agents. We show empirically on nine standard value distributions, including disaster response and electric vehicle allocation benchmarks, that our algorithm enables a rapid finding of high-quality solutions and compares favorably with other state-of-the-art methods. 

**Abstract (ZH)**: 合作结构生成（CSG），即最优地将一组代理分成合作体以最大化社会福利的问题，是多代理系统中的一个基本计算问题。该问题在需要快速运行时间的应用中非常重要，包括交通管理和灾害响应等领域。在本文中，我们开发了SALDAE算法，这是一种基于合作结构图的多代理路径规划算法。该算法利用了多种启发式方法和策略来进行搜索并引导搜索过程。它可以处理具有数百乃至数千个代理的大规模问题。通过在包括灾害响应和电动汽车分配基准在内的九种标准价值分布上进行实验，我们展示了该算法能够快速找到高质量的解决方案，并且与其他最先进的方法相比具有竞争力。 

---
# Dynamic Reinforcement Learning for Actors 

**Title (ZH)**: 动态强化学习在演员中的应用 

**Authors**: Katsunari Shibata  

**Link**: [PDF](https://arxiv.org/pdf/2502.10200)  

**Abstract**: Dynamic Reinforcement Learning (Dynamic RL), proposed in this paper, directly controls system dynamics, instead of the actor (action-generating neural network) outputs at each moment, bringing about a major qualitative shift in reinforcement learning (RL) from static to dynamic. The actor is initially designed to generate chaotic dynamics through the loop with its environment, enabling the agent to perform flexible and deterministic exploration. Dynamic RL controls global system dynamics using a local index called "sensitivity," which indicates how much the input neighborhood contracts or expands into the corresponding output neighborhood through each neuron's processing. While sensitivity adjustment learning (SAL) prevents excessive convergence of the dynamics, sensitivity-controlled reinforcement learning (SRL) adjusts them -- to converge more to improve reproducibility around better state transitions with positive TD error and to diverge more to enhance exploration around worse transitions with negative TD error. Dynamic RL was applied only to the actor in an Actor-Critic RL architecture while applying it to the critic remains a challenge. It was tested on two dynamic tasks and functioned effectively without external exploration noise or backward computation through time. Moreover, it exhibited excellent adaptability to new environments, although some problems remain. Drawing parallels between 'exploration' and 'thinking,' the author hypothesizes that "exploration grows into thinking through learning" and believes this RL could be a key technique for the emergence of thinking, including inspiration that cannot be reconstructed from massive existing text data. Finally, despite being presumptuous, the author presents the argument that this research should not proceed due to its potentially fatal risks, aiming to encourage discussion. 

**Abstract (ZH)**: 本文提出了动态强化学习（Dynamic RL），该方法直接控制系统动态，而不是在每个时刻控制行动生成神经网络（actor）的输出，从而在强化学习（RL）中从静态转变为动态，带来了一个根本性的质的飞跃。actor 初始设计通过与环境的闭环生成混沌动态，使代理能够进行灵活且确定性的探索。动态RL 使用一个局部指标“灵敏度”（sensitivity）来控制全局系统动态，灵敏度表明每个神经元处理过程中输入邻域是如何收缩或扩展到相应的输出邻域的。虽然灵敏度调整学习（SAL）防止了动态过度收敛，但灵敏度控制的强化学习（SRL）则调整动态使其更加收敛，以提高在正TD误差周围更好状态转换的可重复性；同时使其更加发散，以增强在负TD误差周围较差状态转换的探索。

动态RL仅应用于Actor-Critic RL架构中的actor部分，将其应用于critic仍然是一个挑战。该方法在两个动态任务上进行了测试，并且无需外部探索噪声或时间反向计算即可有效运行。此外，它展示了出色的对新环境的适应性，尽管还有一些问题需要解决。将“探索”与“思考”进行类比，作者提出“探索通过学习成长为思考”的假设，并认为这种RL可能是思考（包括无法从大量的现有文本数据中重构的启发式思维）出现的关键技术。最终，尽管这是推测性的结论，作者提出了这项研究应暂停的观点，旨在促进讨论。 

---
# Evaluating and Improving Graph-based Explanation Methods for Multi-Agent Coordination 

**Title (ZH)**: 基于图的解释方法在多智能体协调中的评估与改进 

**Authors**: Siva Kailas, Shalin Jain, Harish Ravichandar  

**Link**: [PDF](https://arxiv.org/pdf/2502.09889)  

**Abstract**: Graph Neural Networks (GNNs), developed by the graph learning community, have been adopted and shown to be highly effective in multi-robot and multi-agent learning. Inspired by this successful cross-pollination, we investigate and characterize the suitability of existing GNN explanation methods for explaining multi-agent coordination. We find that these methods have the potential to identify the most-influential communication channels that impact the team's behavior. Informed by our initial analyses, we propose an attention entropy regularization term that renders GAT-based policies more amenable to existing graph-based explainers. Intuitively, minimizing attention entropy incentivizes agents to limit their attention to the most influential or impactful agents, thereby easing the challenge faced by the explainer. We theoretically ground this intuition by showing that minimizing attention entropy increases the disparity between the explainer-generated subgraph and its complement. Evaluations across three tasks and three team sizes i) provides insights into the effectiveness of existing explainers, and ii) demonstrates that our proposed regularization consistently improves explanation quality without sacrificing task performance. 

**Abstract (ZH)**: 图神经网络（GNNs），由图学习社区发展而来，已被应用于多机器人和多智能体学习，并显示出高度有效性。受到这种成功交叉学习的启发，我们研究和探索了现有GNN解释方法在解释多智能体协调任务中的适用性。我们发现，这些方法有潜力识别出对团队行为影响最大的通信渠道。基于初步分析，我们提出了一种注意力熵正则化项，使基于注意机制的策略更加适合现有的基于图的解释器。直观地讲，通过最小化注意力熵，可以激励智能体将注意力集中在最具影响力或最具有影响性的智能体上，从而减轻解释器面临的挑战。我们通过理论证明这一直觉，即最小化注意力熵增加了解释器生成的子图与其补图之间的差异。通过对三个任务和三种团队规模的评估，我们发现了关于现有解释器有效性的见解，并证明了所提出的正则化方法在不牺牲任务性能的情况下，始终能提高解释质量。 

---
# Automated Hypothesis Validation with Agentic Sequential Falsifications 

**Title (ZH)**: 自动化的假设验证通过代理性的序列否证 

**Authors**: Kexin Huang, Ying Jin, Ryan Li, Michael Y. Li, Emmanuel Candès, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2502.09858)  

**Abstract**: Hypotheses are central to information acquisition, decision-making, and discovery. However, many real-world hypotheses are abstract, high-level statements that are difficult to validate directly. This challenge is further intensified by the rise of hypothesis generation from Large Language Models (LLMs), which are prone to hallucination and produce hypotheses in volumes that make manual validation impractical. Here we propose Popper, an agentic framework for rigorous automated validation of free-form hypotheses. Guided by Karl Popper's principle of falsification, Popper validates a hypothesis using LLM agents that design and execute falsification experiments targeting its measurable implications. A novel sequential testing framework ensures strict Type-I error control while actively gathering evidence from diverse observations, whether drawn from existing data or newly conducted procedures. We demonstrate Popper on six domains including biology, economics, and sociology. Popper delivers robust error control, high power, and scalability. Furthermore, compared to human scientists, Popper achieved comparable performance in validating complex biological hypotheses while reducing time by 10 folds, providing a scalable, rigorous solution for hypothesis validation. 

**Abstract (ZH)**: 假设在信息获取、决策制定和发现中起着核心作用。然而，许多现实世界的假设往往是抽象和高层的陈述，难以直接验证。这一挑战因大型语言模型（LLMs）生成假设而加剧，LLMs容易产生幻觉，且生成的假设数量之大使得手动验证变得不切实际。为此，我们提出了Popper这一代理框架，用于严谨的自动化假设验证。该框架遵循卡尔·波普尔的证伪原则，通过利用LLM代理设计和执行针对假设可测量影响的证伪实验来进行验证。一个新颖的序列测试框架确保了严格的I型错误控制，同时积极从多种观察中收集证据，无论这些观察是现有的数据还是新进行的程序。我们已在生物学、经济学和社会学等六个领域展示了Popper。Popper实现了稳健的错误控制、高功效和可扩展性。此外，相较于人类科学家，Popper在验证复杂的生物学假设方面实现了相当的性能，同时将时间减少了十倍，为假设验证提供了可扩展且严谨的解决方案。 

---
# Efficient Evaluation of Multi-Task Robot Policies With Active Experiment Selection 

**Title (ZH)**: 基于主动实验选择的多任务机器人策略高效评估方法 

**Authors**: Abrar Anwar, Rohan Gupta, Zain Merchant, Sayan Ghosh, Willie Neiswanger, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2502.09829)  

**Abstract**: Evaluating learned robot control policies to determine their physical task-level capabilities costs experimenter time and effort. The growing number of policies and tasks exacerbates this issue. It is impractical to test every policy on every task multiple times; each trial requires a manual environment reset, and each task change involves re-arranging objects or even changing robots. Naively selecting a random subset of tasks and policies to evaluate is a high-cost solution with unreliable, incomplete results. In this work, we formulate robot evaluation as an active testing problem. We propose to model the distribution of robot performance across all tasks and policies as we sequentially execute experiments. Tasks often share similarities that can reveal potential relationships in policy behavior, and we show that natural language is a useful prior in modeling these relationships between tasks. We then leverage this formulation to reduce the experimenter effort by using a cost-aware expected information gain heuristic to efficiently select informative trials. Our framework accommodates both continuous and discrete performance outcomes. We conduct experiments on existing evaluation data from real robots and simulations. By prioritizing informative trials, our framework reduces the cost of calculating evaluation metrics for robot policies across many tasks. 

**Abstract (ZH)**: 学习得到的机器人控制策略评估耗时费力，以确定其在物理任务层面的能力。随着策略和任务数量的增长，这一问题变得更加严重。测试每一个策略在每一个任务上多次是不切实际的；每次试验都需要手动重置环境，而任务的更改则需要重新布置物体甚至更换机器人。简单地随机选择一部分任务和策略进行评估是一种高成本解决方案，结果可能会不可靠且不完整。在本研究中，我们将机器人评估问题表述为一种积极测试问题。我们提出将机器人在所有任务和策略上的表现分布建模为我们逐步执行实验时的变化过程。任务之间往往存在相似之处，可以揭示策略行为之间的潜在关系，我们证明了自然语言在这种关系建模中是很有用的先验知识。然后，我们利用这一建模方法通过使用成本感知的预期信息增益启发式算法选择信息丰富的试验来减少实验者的努力。我们的框架适用于连续性和离散性表现结果。我们在现有真实机器人和模拟的数据上进行实验，通过优先选择信息丰富的试验，我们的框架能够在众多任务中减少计算机器人策略评估指标的成本。 

---
# AgentGuard: Repurposing Agentic Orchestrator for Safety Evaluation of Tool Orchestration 

**Title (ZH)**: AgentGuard: 重新利用代理型编排器进行工具编排的安全评估 

**Authors**: Jizhou Chen, Samuel Lee Cong  

**Link**: [PDF](https://arxiv.org/pdf/2502.09809)  

**Abstract**: The integration of tool use into large language models (LLMs) enables agentic systems with real-world impact. In the meantime, unlike standalone LLMs, compromised agents can execute malicious workflows with more consequential impact, signified by their tool-use capability. We propose AgentGuard, a framework to autonomously discover and validate unsafe tool-use workflows, followed by generating safety constraints to confine the behaviors of agents, achieving the baseline of safety guarantee at deployment. AgentGuard leverages the LLM orchestrator's innate capabilities - knowledge of tool functionalities, scalable and realistic workflow generation, and tool execution privileges - to act as its own safety evaluator. The framework operates through four phases: identifying unsafe workflows, validating them in real-world execution, generating safety constraints, and validating constraint efficacy. The output, an evaluation report with unsafe workflows, test cases, and validated constraints, enables multiple security applications. We empirically demonstrate AgentGuard's feasibility with experiments. With this exploratory work, we hope to inspire the establishment of standardized testing and hardening procedures for LLM agents to enhance their trustworthiness in real-world applications. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

将工具使用集成到大规模语言模型（LLMs）中能够构建具有实际影响的自主系统。同时，与独立的LLMs不同，被攻击的代理可以利用其工具使用能力更有效地执行恶意工作流，造成更为重大的影响。为此，我们提出了一种名为AgentGuard的框架，该框架能够自主发现和验证不安全的工具使用工作流，并生成安全约束来限制代理的行为，在部署时实现基本的安全保障。AgentGuard利用LLM编排器固有的能力——了解工具功能、规模化和现实可行的工作流生成以及执行工具的权限——作为其自身安全性评估器。该框架通过四个阶段运行：识别不安全的工作流、在实际执行中验证这些工作流、生成安全约束并验证约束的有效性。最终输出包含不安全工作流、测试用例和已验证约束的评估报告，这些都可以应用于多种安全应用场景。我们通过实验实证地展示了AgentGuard的可行性。通过这项探索性工作，我们希望激发建立标准化的LLM代理测试和加固程序，从而增强其在实际应用中的可靠性。 

---
# TableTalk: Scaffolding Spreadsheet Development with a Language Agent 

**Title (ZH)**: TableTalk：语言代理辅助的电子表格开发支架 

**Authors**: Jenny T. Liang, Aayush Kumar, Yasharth Bajpai, Sumit Gulwani, Vu Le, Chris Parnin, Arjun Radhakrishna, Ashish Tiwari, Emerson Murphy-Hill, Guastavo Soares  

**Link**: [PDF](https://arxiv.org/pdf/2502.09787)  

**Abstract**: Despite its ubiquity in the workforce, spreadsheet programming remains challenging as programmers need both spreadsheet-specific knowledge (e.g., APIs to write formulas) and problem-solving skills to create complex spreadsheets. Large language models (LLMs) can help automate aspects of this process, and recent advances in planning and reasoning have enabled language agents, which dynamically plan, use tools, and take iterative actions to complete complex tasks. These agents observe, plan, and act, making them well-suited to scaffold spreadsheet programming by following expert processes.
We present TableTalk, a language agent that helps programmers build spreadsheets conversationally. Its design reifies three design principles -- scaffolding, flexibility, and incrementality -- which we derived from two studies of seven programmers and 62 Excel templates. TableTalk structures spreadsheet development by generating step-by-step plans and suggesting three next steps users can choose from. It also integrates tools that enable incremental spreadsheet construction. A user study with 20 programmers shows that TableTalk produces spreadsheets 2.3 times more likely to be preferred over a baseline agent, while reducing cognitive load and time spent reasoning about spreadsheet actions by 12.6%. TableTalk's approach has implications for human-agent collaboration. This includes providing persistent direct manipulation interfaces for stopping or undoing agent actions, while ensuring that such interfaces for accepting actions can be deactivated. 

**Abstract (ZH)**: 尽管电子表格编程在工作中无处不在，但编程人员仍需掌握电子表格特定的知识（例如，编写公式的API）和解决问题的能力，才能创建复杂的电子表格。大型语言模型（LLMs）可以帮助自动化此过程中的某些方面，近期规划和推理方面的进展使得语言代理能够动态规划、使用工具并采取迭代行动以完成复杂任务。这些代理观察、计划和行动，因此它们非常适合通过遵循专家流程来搭建电子表格编程。

我们提出了一种名为TableTalk的语言代理，它帮助编程人员通过对话方式构建电子表格。其设计遵循了三条设计原则——搭建、灵活性和逐步性，这三条原则源自对七位编程人员和62个Excel模板的两个研究。TableTalk通过生成逐步计划并建议用户提供三种可选的下一步操作，来结构化电子表格的开发工作。此外，TableTalk还整合了用于逐步构建电子表格的工具。一项涉及20位编程人员的用户研究表明，与基准代理相比，TableTalk生成的电子表格被更偏好（2.3倍的可能性），同时减少了12.6%的认知负担和用于推理电子表格操作的时间。TableTalk的方法对人机协作具有重要意义，包括提供持久的直接操作界面以停止或撤销代理操作，同时确保可接受操作的界面能够被禁用。

这种人机协作的方法还具有其他影响。它包括提供持久的直接操作界面以停止或撤销代理操作，同时确保接受操作的界面可以被禁用。 

---
# Agentic Verification for Ambiguous Query Disambiguation 

**Title (ZH)**: 代理验证在模糊查询去歧义化中的应用 

**Authors**: Youngwon Lee, Seung-won Hwang, Ruofan Wu, Feng Yan, Danmei Xu, Moutasem Akkad, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2502.10352)  

**Abstract**: In this work, we tackle the challenge of disambiguating queries in retrieval-augmented generation (RAG) to diverse yet answerable interpretations. State-of-the-arts follow a Diversify-then-Verify (DtV) pipeline, where diverse interpretations are generated by an LLM, later used as search queries to retrieve supporting passages. Such a process may introduce noise in either interpretations or retrieval, particularly in enterprise settings, where LLMs -- trained on static data -- may struggle with domain-specific disambiguations. Thus, a post-hoc verification phase is introduced to prune noises. Our distinction is to unify diversification with verification by incorporating feedback from retriever and generator early on. This joint approach improves both efficiency and robustness by reducing reliance on multiple retrieval and inference steps, which are susceptible to cascading errors. We validate the efficiency and effectiveness of our method, Verified-Diversification with Consolidation (VERDICT), on the widely adopted ASQA benchmark to achieve diverse yet verifiable interpretations. Empirical results show that VERDICT improves grounding-aware F1 score by an average of 23% over the strongest baseline across different backbone LLMs. 

**Abstract (ZH)**: 在本文中，我们针对检索增强生成（RAG）中查询消歧的问题进行了研究，旨在获取多样化但又可回答的解释。现有的先进方法采用“多样化-然后验证”（DtV）的管道，其中通过语言模型（LLM）生成多样化的解释，随后将这些解释用作检索查询以获取支持段落。这一过程中，无论是解释还是检索都可能引入噪声，尤其是在企业环境中，LLM 在处理特定领域消歧问题时可能面临挑战。因此，我们引入了一个事后验证阶段以消除这些噪声。我们的区别在于，通过早期结合检索器和生成器的反馈，将多样化与验证统一起来。这种联合方法通过减少对外部多次检索和推理步骤的依赖，从而在减少累积错误的同时提高了效率和鲁棒性。我们验证了联合方法Consolidated Verified Diversification (VERDICT) 在广泛采用的ASQA基准上的效率和有效性，实现了多样化且可验证的解释。实验证明，VERDICT 相比最强的方法在不同语言模型（LLM）的背景下平均提高了23%的注意向量关联F1分数。 

---
