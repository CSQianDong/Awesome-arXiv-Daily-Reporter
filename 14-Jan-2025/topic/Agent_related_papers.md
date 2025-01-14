# Inductive Learning of Robot Task Knowledge from Raw Data and Online Expert Feedback 

**Title (ZH)**: 从原始数据和在线专家反馈中归纳学习机器人任务知识 

**Authors**: Daniele Meli, Paolo Fiorini  

**Link**: [PDF](https://arxiv.org/pdf/2501.07507)  

**Abstract**: The increasing level of autonomy of robots poses challenges of trust and social acceptance, especially in human-robot interaction scenarios. This requires an interpretable implementation of robotic cognitive capabilities, possibly based on formal methods as logics for the definition of task specifications. However, prior knowledge is often unavailable in complex realistic scenarios.
In this paper, we propose an offline algorithm based on inductive logic programming from noisy examples to extract task specifications (i.e., action preconditions, constraints and effects) directly from raw data of few heterogeneous (i.e., not repetitive) robotic executions. Our algorithm leverages on the output of any unsupervised action identification algorithm from video-kinematic recordings. Combining it with the definition of very basic, almost task-agnostic, commonsense concepts about the environment, which contribute to the interpretability of our methodology, we are able to learn logical axioms encoding preconditions of actions, as well as their effects in the event calculus paradigm. Since the quality of learned specifications depends mainly on the accuracy of the action identification algorithm, we also propose an online framework for incremental refinement of task knowledge from user feedback, guaranteeing safe execution. Results in a standard manipulation task and benchmark for user training in the safety-critical surgical robotic scenario, show the robustness, data- and time-efficiency of our methodology, with promising results towards the scalability in more complex domains. 

**Abstract (ZH)**: 随着机器人自主程度的提升，信任和社会接受度的问题日益凸显，尤其是在人机交互场景中。这要求对机器人的认知能力进行可解释的实现，可能基于形式方法，即通过逻辑来定义任务规格。然而，在复杂的现实场景中，先验知识往往是不可用的。

在此论文中，我们提出了一种基于归纳逻辑编程的离线算法，该算法可以从嘈杂的数据中提取任务规格（即动作的前置条件、约束和效果），直接从少数不重复的机器人执行的原始数据中提取。该算法依赖于对视频-kinematic记录中任何无监督动作识别算法的输出。结合一些基本的、几乎与任务无关的环境常识概念，这些概念有助于我们方法的可解释性，使我们能够利用事件计算框架学习逻辑公理，编码动作的前置条件及其效果。由于所学习的任务规格的质量主要取决于动作识别算法的准确性，我们还提出了一种基于用户反馈的在线框架，用于递增地细化任务知识，从而确保安全执行。

在标准操作任务和用户培训的临界手术机器人场景基准测试中，结果显示了我们方法的鲁棒性、数据和时间效率，并显示出在更复杂领域中扩展的可能性，取得了令人鼓舞的结果。 

---
# A Survey of Embodied AI in Healthcare: Techniques, Applications, and Opportunities 

**Title (ZH)**: 医疗领域的躯体化人工智能综述：技术、应用及机遇 

**Authors**: Yihao Liu, Xu Cao, Tingting Chen, Yankai Jiang, Junjie You, Minghua Wu, Xiaosong Wang, Mengling Feng, Yaochu Jin, Jintai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07468)  

**Abstract**: Healthcare systems worldwide face persistent challenges in efficiency, accessibility, and personalization. Powered by modern AI technologies such as multimodal large language models and world models, Embodied AI (EmAI) represents a transformative frontier, offering enhanced autonomy and the ability to interact with the physical world to address these challenges. As an interdisciplinary and rapidly evolving research domain, "EmAI in healthcare" spans diverse fields such as algorithms, robotics, and biomedicine. This complexity underscores the importance of timely reviews and analyses to track advancements, address challenges, and foster cross-disciplinary collaboration. In this paper, we provide a comprehensive overview of the "brain" of EmAI for healthcare, wherein we introduce foundational AI algorithms for perception, actuation, planning, and memory, and focus on presenting the healthcare applications spanning clinical interventions, daily care & companionship, infrastructure support, and biomedical research. Despite its promise, the development of EmAI for healthcare is hindered by critical challenges such as safety concerns, gaps between simulation platforms and real-world applications, the absence of standardized benchmarks, and uneven progress across interdisciplinary domains. We discuss the technical barriers and explore ethical considerations, offering a forward-looking perspective on the future of EmAI in healthcare. A hierarchical framework of intelligent levels for EmAI systems is also introduced to guide further development. By providing systematic insights, this work aims to inspire innovation and practical applications, paving the way for a new era of intelligent, patient-centered healthcare. 

**Abstract (ZH)**: 全球的卫生系统在效率、可及性和个性化方面面临着持续的挑战。借助现代人工智能技术，如多模态大型语言模型和世界模型，具身人工智能（Embodied Artificial Intelligence, EmAI）代表着一个变革的前沿领域，能够提供增强的自主性和与物理世界的互动能力，以应对这些挑战。作为一个交叉学科且迅速发展的研究领域，“EmAI在医疗保健中的应用”涵盖了算法、机器人技术和生物医学等多个领域。这种复杂性强调了及时进行综述和分析的重要性，以跟踪进展、解决问题并促进跨学科合作。在本文中，我们提供了一个全面的“EmAI在医疗保健中的大脑”的综述，介绍了感知、动作、规划和记忆的基础性人工智能算法，并重点展示了其在临床干预、日常护理与陪伴、基础设施支持以及生物医学研究中的应用。尽管EmAI在医疗保健领域有着巨大的潜力，但其发展仍受到了诸如安全问题、仿真平台与实际应用之间的差距、缺乏标准化基准和跨学科领域进展不均衡等诸多关键挑战的制约。我们讨论了技术障碍并探讨了伦理问题，提出了对EmAI在医疗保健领域的未来展望。我们还引入了一个具身人工智能系统的层级框架，以指导进一步的发展。通过提供系统性的见解，本文旨在激发创新和实际应用，为智能、以患者为中心的医疗保健新时代铺平道路。 

---
# Online inductive learning from answer sets for efficient reinforcement learning exploration 

**Title (ZH)**: 基于答案集的在线归纳学习以实现高效的强化学习探索 

**Authors**: Celeste Veronese, Daniele Meli, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2501.07445)  

**Abstract**: This paper presents a novel approach combining inductive logic programming with reinforcement learning to improve training performance and explainability. We exploit inductive learning of answer set programs from noisy examples to learn a set of logical rules representing an explainable approximation of the agent policy at each batch of experience. We then perform answer set reasoning on the learned rules to guide the exploration of the learning agent at the next batch, without requiring inefficient reward shaping and preserving optimality with soft bias. The entire procedure is conducted during the online execution of the reinforcement learning algorithm. We preliminarily validate the efficacy of our approach by integrating it into the Q-learning algorithm for the Pac-Man scenario in two maps of increasing complexity. Our methodology produces a significant boost in the discounted return achieved by the agent, even in the first batches of training. Moreover, inductive learning does not compromise the computational time required by Q-learning and learned rules quickly converge to an explanation of the agent policy. 

**Abstract (ZH)**: 本文提出了一种将归纳逻辑编程与强化学习相结合的新型方法，以提高训练性能和可解释性。该方法利用从噪声样本中归纳学习回答集程序，来学习一套逻辑规则，这些规则在每一批经验中代表了可解释的代理策略的近似表示。然后，我们在学习的规则上执行回答集推理，以指导强化学习代理在下一组经验中的探索行为，而无需进行效率低下的奖励塑形，并保持最优性的同时带有软偏置。整个过程在强化学习算法的在线执行过程中进行。我们通过将该方法集成到Pac-Man场景的Q学习算法中，并在两个逐渐增加复杂度的地图上进行初步验证，来验证该方法的有效性。我们的方法在训练初期显著提升了代理的折现回报。此外，归纳学习并未增加Q学习所需的时间成本，而且学到的规则迅速收敛到代理策略的解释。 

---
# Lifelong Learning of Large Language Model based Agents: A Roadmap 

**Title (ZH)**: 基于代理的大型语言模型终身学习：一条路线图 

**Authors**: Junhao Zheng, Chengming Shi, Xidi Cai, Qiuke Li, Duzhen Zhang, Chenxing Li, Dong Yu, Qianli Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.07278)  

**Abstract**: Lifelong learning, also known as continual or incremental learning, is a crucial component for advancing Artificial General Intelligence (AGI) by enabling systems to continuously adapt in dynamic environments. While large language models (LLMs) have demonstrated impressive capabilities in natural language processing, existing LLM agents are typically designed for static systems and lack the ability to adapt over time in response to new challenges. This survey is the first to systematically summarize the potential techniques for incorporating lifelong learning into LLM-based agents. We categorize the core components of these agents into three modules: the perception module for multimodal input integration, the memory module for storing and retrieving evolving knowledge, and the action module for grounded interactions with the dynamic environment. We highlight how these pillars collectively enable continuous adaptation, mitigate catastrophic forgetting, and improve long-term performance. This survey provides a roadmap for researchers and practitioners working to develop lifelong learning capabilities in LLM agents, offering insights into emerging trends, evaluation metrics, and application scenarios. Relevant literature and resources are available at \href{this url}{this https URL}. 

**Abstract (ZH)**: 终身学习，也称为持续学习或增量学习，是推进通用人工智能（AGI）的关键组件，它使系统能够适应动态环境。虽然大型语言模型（LLMs）在自然语言处理方面表现出色，但现有的LLM代理通常设计为静态系统，缺乏根据新挑战不断调整的能力。本文是首次系统总结将终身学习引入LLM基代理潜在技术的综述。我们将这些代理的核心组件分为三个模块：感知模块用于多模态输入集成、记忆模块用于存储和检索不断演变的知识，以及动作模块用于与动态环境进行具身交互。我们强调了这些支柱如何共同实现持续适应、防止灾难性遗忘以及提升长期性能。本文为研究人员和实践者开发LLM代理的终身学习能力提供了指导，提供了关于新兴趋势、评估指标和应用场景的见解。相关文献和资源详见 [该网址](this https URL)。 

---
# PoAct: Policy and Action Dual-Control Agent for Generalized Applications 

**Title (ZH)**: PoAct：通用应用的策略与动作双控智能体 

**Authors**: Guozhi Yuan, Youfeng Liu, Jingli Yang, Wei Jia, Kai Lin, Yansong Gao, Shan He, Zilin Ding, Haitao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.07054)  

**Abstract**: Based on their superior comprehension and reasoning capabilities, Large Language Model (LLM) driven agent frameworks have achieved significant success in numerous complex reasoning tasks. ReAct-like agents can solve various intricate problems step-by-step through progressive planning and tool calls, iteratively optimizing new steps based on environmental feedback. However, as the planning capabilities of LLMs improve, the actions invoked by tool calls in ReAct-like frameworks often misalign with complex planning and challenging data organization. Code Action addresses these issues while also introducing the challenges of a more complex action space and more difficult action organization. To leverage Code Action and tackle the challenges of its complexity, this paper proposes Policy and Action Dual-Control Agent (PoAct) for generalized applications. The aim is to achieve higher-quality code actions and more accurate reasoning paths by dynamically switching reasoning policies and modifying the action space. Experimental results on the Agent Benchmark for both legal and generic scenarios demonstrate the superior reasoning capabilities and reduced token consumption of our approach in complex tasks. On the LegalAgentBench, our method shows a 20 percent improvement over the baseline while requiring fewer tokens. We conducted experiments and analyses on the GPT-4o and GLM-4 series models, demonstrating the significant potential and scalability of our approach to solve complex problems. 

**Abstract (ZH)**: 基于其卓越的理解能力和推理能力，由大语言模型（LLM）驱动的智能体框架在众多复杂的推理任务中取得了显著的成功。类似于ReAct的智能体可以通过逐步的规划和工具调用解决各种复杂问题，并根据环境反馈迭代优化新的步骤。然而，随着LLM规划能力的提高，ReAct框架中由工具调用引发的动作往往与复杂的规划和具有挑战性的数据组织不相匹配。代码执行（Code Action）解决这些问题的同时，也带来了动作空间更加复杂以及动作组织更加困难的挑战。为了利用代码执行并应对这些复杂性挑战，本文提出了一种策略与行动双控制智能体（Policy and Action Dual-Control Agent, PoAct），以实现泛化的应用。目标是通过动态切换推理策略并调整动作空间，实现更高的代码执行质量及更准确的推理路径。在代理基准测试中，无论是针对法律场景还是通用场景，我们的方法都展示了更强的推理能力和更少的标记消耗。在LegalAgentBench上，我们的方法相比基准方法提高了20%，且消耗的标记更少。我们在GPT-4o和GLM-4系列模型上进行了实验和分析，验证了我们方法的巨大潜力和可扩展性，可用于解决复杂问题。 

---
# The Einstein Test: Towards a Practical Test of a Machine's Ability to Exhibit Superintelligence 

**Title (ZH)**: 爱因斯坦测试：朝着测试机器展现超智能能力的实用性方法迈出的一步 

**Authors**: David Benrimoh, Nace Mikus, Ariel Rosenfeld  

**Link**: [PDF](https://arxiv.org/pdf/2501.06948)  

**Abstract**: Creative and disruptive insights (CDIs), such as the development of the theory of relativity, have punctuated human history, marking pivotal shifts in our intellectual trajectory. Recent advancements in artificial intelligence (AI) have sparked debates over whether state of the art models possess the capacity to generate CDIs. We argue that the ability to create CDIs should be regarded as a significant feature of machine superintelligence (SI).To this end, we propose a practical test to evaluate whether an approach to AI targeting SI can yield novel insights of this kind. We propose the Einstein test: given the data available prior to the emergence of a known CDI, can an AI independently reproduce that insight (or one that is formally equivalent)? By achieving such a milestone, a machine can be considered to at least match humanity's past top intellectual achievements, and therefore to have the potential to surpass them. 

**Abstract (ZH)**: 创造性和变革性的洞察（Creative and Disruptive Insights，CDIs），如相对论的发展，贯穿了人类历史，标志着我们在智力轨迹上的关键转折。近年来，人工智能（AI）的最新进展引发了关于当前最先进模型是否具有生成CDIs能力的讨论。我们认为，生成CDIs的能力应被视为机器超智能（Superintelligence，SI）的一个重要特征。为此，我们提出了一种实际测试方法，以评估是否能够通过某种AI方法达到生成此类新型见解的能力。我们称之为爱因斯坦测试：给定在某一已知CDI出现之前可用的数据，AI能否独立地重现这一洞察（或一个形式等价的洞察）？通过实现这一里程碑，机器可以被认为至少达到了人类过去顶级智力成就的水平，并因此具备超越这些成就的潜力。 

---
# LLMs Model Non-WEIRD Populations: Experiments with Synthetic Cultural Agents 

**Title (ZH)**: LLMs 模型化非 WEIRD 人群：合成文化代理的实验研究 

**Authors**: Augusto Gonzalez-Bonorino, Monica Capra, Emilio Pantoja  

**Link**: [PDF](https://arxiv.org/pdf/2501.06834)  

**Abstract**: Despite its importance, studying economic behavior across diverse, non-WEIRD (Western, Educated, Industrialized, Rich, and Democratic) populations presents significant challenges. We address this issue by introducing a novel methodology that uses Large Language Models (LLMs) to create synthetic cultural agents (SCAs) representing these populations. We subject these SCAs to classic behavioral experiments, including the dictator and ultimatum games. Our results demonstrate substantial cross-cultural variability in experimental behavior. Notably, for populations with available data, SCAs' behaviors qualitatively resemble those of real human subjects. For unstudied populations, our method can generate novel, testable hypotheses about economic behavior. By integrating AI into experimental economics, this approach offers an effective and ethical method to pilot experiments and refine protocols for hard-to-reach populations. Our study provides a new tool for cross-cultural economic studies and demonstrates how LLMs can help experimental behavioral research. 

**Abstract (ZH)**: 尽管其重要性不言而喻，研究跨多元、非WEIRD（西方、受过教育、工业化、富裕、民主）人口的经济行为存在显著挑战。为应对这一问题，我们引入了一种新的方法，该方法利用大型语言模型（LLMs）创建合成文化代理（SCAs），以代表这些人群。我们将这些SCAs置于经典的经济行为实验中，包括分配者游戏和 ultimatum 游戏。研究结果表明，这些实验中的行为在跨文化上存在显著差异。值得注意的是，对于已有数据的人群，SCAs 的行为在定性上与真实人类被试相似。对于未被研究的人群，我们的方法可以产生新的、可测试的关于经济行为的假设。通过将AI引入实验经济学，该方法提供了一种有效且伦理的方式，以试点实验并改进难以接触人群的研究方案。我们的研究提供了一种新的工具，用于跨文化经济研究，并展示了LLMs如何帮助实验行为研究。 

---
# Eliza: A Web3 friendly AI Agent Operating System 

**Title (ZH)**: 艾莉莎：一个面向Web3的AI代理操作系统 

**Authors**: Shaw Walters, Sam Gao, Shakker Nerd, Feng Da, Warren Williams, Ting-Chien Meng, Hunter Han, Frank He, Allen Zhang, Ming Wu, Timothy Shen, Maxwell Hu, Jerry Yan  

**Link**: [PDF](https://arxiv.org/pdf/2501.06781)  

**Abstract**: AI Agent, powered by large language models (LLMs) as its cognitive core, is an intelligent agentic system capable of autonomously controlling and determining the execution paths under user's instructions. With the burst of capabilities of LLMs and various plugins, such as RAG, text-to-image/video/3D, etc., the potential of AI Agents has been vastly expanded, with their capabilities growing stronger by the day. However, at the intersection between AI and web3, there is currently no ideal agentic framework that can seamlessly integrate web3 applications into AI agent functionalities. In this paper, we propose Eliza, the first open-source web3-friendly Agentic framework that makes the deployment of web3 applications effortless. We emphasize that every aspect of Eliza is a regular Typescript program under the full control of its user, and it seamlessly integrates with web3 (i.e., reading and writing blockchain data, interacting with smart contracts, etc.). Furthermore, we show how stable performance is achieved through the pragmatic implementation of the key components of Eliza's runtime. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的认知核心，AI 剂机是一种能够在用户指令下自主控制和决定执行路径的智能代理系统。随着大型语言模型能力的爆发式增长以及各种插件（如RAG、文本转图像/视频/3D等）的应用，AI 剂机的潜力得到了极大扩展，其功能日渐增强。然而，在AI与Web3的交汇点上，目前尚缺乏一个能够无缝集成Web3应用到AI剂机功能的理想框架。本文中，我们提出Eliza——首个开源的Web3友好型代理框架，使得部署Web3应用变得轻而易举。我们强调，Eliza 的每一部分都是用户完全控制下的常规TypeScript程序，并且能够无缝集成Web3（如读取和写入区块链数据、与智能合约互动等）。此外，我们展示了通过Pragmatic实现实现Eliza运行时关键组件的稳定性能。我们的代码已公开，可在以下链接访问：[此 https URL](此 https URL)。 

---
# AIOpsLab: A Holistic Framework to Evaluate AI Agents for Enabling Autonomous Clouds 

**Title (ZH)**: AIOpsLab：一个全面框架，用于评估使能自主云的AI代理 

**Authors**: Yinfang Chen, Manish Shetty, Gagan Somashekar, Minghua Ma, Yogesh Simmhan, Jonathan Mace, Chetan Bansal, Rujia Wang, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2501.06706)  

**Abstract**: AI for IT Operations (AIOps) aims to automate complex operational tasks, such as fault localization and root cause analysis, to reduce human workload and minimize customer impact. While traditional DevOps tools and AIOps algorithms often focus on addressing isolated operational tasks, recent advances in Large Language Models (LLMs) and AI agents are revolutionizing AIOps by enabling end-to-end and multitask automation. This paper envisions a future where AI agents autonomously manage operational tasks throughout the entire incident lifecycle, leading to self-healing cloud systems, a paradigm we term AgentOps. Realizing this vision requires a comprehensive framework to guide the design, development, and evaluation of these agents. To this end, we present AIOPSLAB, a framework that not only deploys microservice cloud environments, injects faults, generates workloads, and exports telemetry data but also orchestrates these components and provides interfaces for interacting with and evaluating agents. We discuss the key requirements for such a holistic framework and demonstrate how AIOPSLAB can facilitate the evaluation of next-generation AIOps agents. Through evaluations of state-of-the-art LLM agents within the benchmark created by AIOPSLAB, we provide insights into their capabilities and limitations in handling complex operational tasks in cloud environments. 

**Abstract (ZH)**: 人工智能在IT运营中的应用（AIOps）旨在自动化复杂的操作任务，如故障定位和根本原因分析，以减轻人类的工作负担并减少对客户的潜在影响。尽管传统的DevOps工具和AIOps算法通常侧重于解决孤立的操作任务，但最近大型语言模型（LLMs）和AI代理的进展正在重新定义AIOps，通过实现端到端和多任务的自动化。本文设想了一个未来，即AI代理能够自主管理整个事件生命周期中的操作任务，从而实现自愈云计算系统，我们将其称为AgentOps范式。实现这一愿景需要一个全面的框架来指导这些代理的设计、开发和评估。为此，我们提出了AIOPSLAB框架，不仅部署微服务云计算环境、注入故障、生成工作负载并导出遥测数据，还协调这些组件，并提供与代理交互和评估的接口。我们讨论了这样一个全面框架的关键要求，并展示了AIOPSLAB如何促进新一代AIOps代理的评估。通过在由AIOPSLAB创建的基准测试中对最先进的LLM代理进行评估，我们提供了它们在处理云计算环境中的复杂操作任务时的能力与局限性的见解。 

---
# DVM: Towards Controllable LLM Agents in Social Deduction Games 

**Title (ZH)**: DVM：面向社交推理游戏中的可控大型语言模型代理 

**Authors**: Zheng Zhang, Yihuai Lan, Yangsen Chen, Lei Wang, Xiang Wang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06695)  

**Abstract**: Large Language Models (LLMs) have advanced the capability of game agents in social deduction games (SDGs). These games rely heavily on conversation-driven interactions and require agents to infer, make decisions, and express based on such information. While this progress leads to more sophisticated and strategic non-player characters (NPCs) in SDGs, there exists a need to control the proficiency of these agents. This control not only ensures that NPCs can adapt to varying difficulty levels during gameplay, but also provides insights into the safety and fairness of LLM agents. In this paper, we present DVM, a novel framework for developing controllable LLM agents for SDGs, and demonstrate its implementation on one of the most popular SDGs, Werewolf. DVM comprises three main components: Predictor, Decider, and Discussor. By integrating reinforcement learning with a win rate-constrained decision chain reward mechanism, we enable agents to dynamically adjust their gameplay proficiency to achieve specified win rates. Experiments show that DVM not only outperforms existing methods in the Werewolf game, but also successfully modulates its performance levels to meet predefined win rate targets. These results pave the way for LLM agents' adaptive and balanced gameplay in SDGs, opening new avenues for research in controllable game agents. 

**Abstract (ZH)**: 大型语言模型（LLMs）在提升桌游推理游戏中人工智能角色（AI角色）的能力方面取得了显著进展。这类游戏依赖于对话驱动的互动，要求AI角色根据相关信息进行推理、决策和表达。虽然这些进步使得桌游推理游戏中的非玩家角色（NPC）变得更加复杂和策略性，但需要控制这些角色的专业水平。这种控制不仅确保AI角色在不同难度等级的游戏中能够适应变化，还为Lyling模型（LLM）代理的安全性和公平性提供了见解。本文提出了一种名为DVM的新颖框架，用于开发可控制的大型语言模型代理，该框架已在最受欢迎的桌游推理游戏之一“狼人杀”中进行了实现。DVM的主要组成部分包括预测器、决策器和讨论者。通过结合强化学习和胜率约束决策链奖励机制，我们使代理能够动态调节其游戏水平以实现指定的胜率。实验表明，DVM不仅在“狼人杀”游戏中优于现有方法，而且成功地调节其性能水平以达到预定义的胜率目标。这些结果为大型语言模型代理在桌游推理游戏中的适应性和平衡游戏提供了可能，为可控游戏代理的研究开辟了新的途径。 

---
# Guided Code Generation with LLMs: A Multi-Agent Framework for Complex Code Tasks 

**Title (ZH)**: 基于LLM的引导式代码生成：复杂代码任务的多代理框架 

**Authors**: Amr Almorsi, Mohanned Ahmed, Walid Gomaa  

**Link**: [PDF](https://arxiv.org/pdf/2501.06625)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in code generation tasks, yet they face significant limitations in handling complex, long-context programming challenges and demonstrating complex compositional reasoning abilities. This paper introduces a novel agentic framework for ``guided code generation'' that tries to address these limitations through a deliberately structured, fine-grained approach to code generation tasks. Our framework leverages LLMs' strengths as fuzzy searchers and approximate information retrievers while mitigating their weaknesses in long sequential reasoning and long-context understanding. Empirical evaluation using OpenAI's HumanEval benchmark with Meta's Llama 3.1 8B model (int4 precision) demonstrates a 23.79\% improvement in solution accuracy compared to direct one-shot generation. Our results indicate that structured, guided approaches to code generation can significantly enhance the practical utility of LLMs in software development while overcoming their inherent limitations in compositional reasoning and context handling. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成任务中展现了显著的能力，但在处理复杂的、长上下文的编程挑战以及展示复杂的组合推理能力方面存在重大局限。本文介绍了一种新的自主框架，旨在通过精细结构化的方法解决这些局限性，以实现“引导式代码生成”。我们的框架利用了LLMs作为模糊搜索者和近似信息检索器的优势，同时减轻了它们在长顺序推理和长上下文理解方面的弱点。使用OpenAI的HumanEval基准测试和Meta的Llama 3.1 8B模型（int4精度）进行的实证评估表明，与直接一对一生成相比，该框架在解决方案准确性上提高了23.79%。我们的结果表明，结构化和引导式的代码生成方法可以显著提高LLMs在软件开发中的实用价值，同时克服它们在组合推理和上下文处理方面的固有局限性。 

---
# Multi-Agent Collaboration Mechanisms: A Survey of LLMs 

**Title (ZH)**: 多智能体协作机制：基于LLMs的综述

注：这里的“LLMs”通常指代“Large Language Models”（大型语言模型）或“Long-Range Language Models”（长范围语言模型），根据具体语境可能有所不同。如果需要更精确的翻译，请提供更具体的背景信息。 

**Authors**: Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.06322)  

**Abstract**: With recent advances in Large Language Models (LLMs), Agentic AI has become phenomenal in real-world applications, moving toward multiple LLM-based agents to perceive, learn, reason, and act collaboratively. These LLM-based Multi-Agent Systems (MASs) enable groups of intelligent agents to coordinate and solve complex tasks collectively at scale, transitioning from isolated models to collaboration-centric approaches. This work provides an extensive survey of the collaborative aspect of MASs and introduces an extensible framework to guide future research. Our framework characterizes collaboration mechanisms based on key dimensions: actors (agents involved), types (e.g., cooperation, competition, or coopetition), structures (e.g., peer-to-peer, centralized, or distributed), strategies (e.g., role-based or model-based), and coordination protocols. Through a review of existing methodologies, our findings serve as a foundation for demystifying and advancing LLM-based MASs toward more intelligent and collaborative solutions for complex, real-world use cases. In addition, various applications of MASs across diverse domains, including 5G/6G networks, Industry 5.0, question answering, and social and cultural settings, are also investigated, demonstrating their wider adoption and broader impacts. Finally, we identify key lessons learned, open challenges, and potential research directions of MASs towards artificial collective intelligence. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的 Recent 进展，代理人工智能在现实世界应用中变得尤为显著，正朝着基于多个 LLM 的代理方向发展，以实现感知、学习、推理和协作行为。这些基于 LLM 的多智能体系统（MASs）使一群智能代理能够协调并大规模地共同解决复杂任务，从孤立模型转向以协作为中心的方法。本文提供了 MASs 协作方面的广泛综述，并提出了一个可扩展的框架，以指导未来的研究。该框架基于关键维度对协作机制进行描述：参与者（涉及的代理）、类型（例如，合作、竞争或合作竞争）、结构（例如，一对一、中心化或分布式）、策略（例如，基于角色或基于模型）和协调协议。通过对现有方法的回顾，我们的发现为揭开和推进基于 LLM 的 MASs 并向为复杂的真实世界应用场景提供更智能和协作的解决方案奠定了基础。此外，我们还探讨了 MASs 在不同领域的各种应用，包括 5G/6G 网络、工业 5.0、问答以及社会和文化背景中，展示了它们更广泛的采纳和更深远的影响。最后，我们确定了 MASs 向人工集体智能发展的关键教训、开放挑战和潜在研究方向。 

---
# BioAgents: Democratizing Bioinformatics Analysis with Multi-Agent Systems 

**Title (ZH)**: 多智能体系统赋能的BioAgents：普惠生物信息学分析 

**Authors**: Nikita Mehandru, Amanda K. Hall, Olesya Melnichenko, Yulia Dubinina, Daniel Tsirulnikov, David Bamman, Ahmed Alaa, Scott Saponas, Venkat S. Malladi  

**Link**: [PDF](https://arxiv.org/pdf/2501.06314)  

**Abstract**: Creating end-to-end bioinformatics workflows requires diverse domain expertise, which poses challenges for both junior and senior researchers as it demands a deep understanding of both genomics concepts and computational techniques. While large language models (LLMs) provide some assistance, they often fall short in providing the nuanced guidance needed to execute complex bioinformatics tasks, and require expensive computing resources to achieve high performance. We thus propose a multi-agent system built on small language models, fine-tuned on bioinformatics data, and enhanced with retrieval augmented generation (RAG). Our system, BioAgents, enables local operation and personalization using proprietary data. We observe performance comparable to human experts on conceptual genomics tasks, and suggest next steps to enhance code generation capabilities. 

**Abstract (ZH)**: 构建端到端的生物信息学工作流需要多领域的专业知识，这对初级和资深研究人员都构成了挑战，因为这要求他们深入了解基因组学概念和计算技术。虽然大型语言模型（LLMs）提供了一定的帮助，但在执行复杂生物信息学任务时，它们往往缺乏提供所需细腻指导的能力，并且需要昂贵的计算资源来实现高性能。因此，我们提出了一种基于小型语言模型的多代理系统，该系统在生物信息学数据上进行了微调，并结合了检索增强生成（RAG）技术。我们的系统BioAgents能够实现本地操作和个人化，并使用专有数据。我们在概念性基因组学任务上观察到的性能与人类专家相当，并提出了增强代码生成能力的下一步建议。 

---
# Agent TCP/IP: An Agent-to-Agent Transaction System 

**Title (ZH)**: 代理TCP/IP：一种代理间事务系统 

**Authors**: Andrea Muttoni, Jason Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.06243)  

**Abstract**: Autonomous agents represent an inevitable evolution of the internet. Current agent frameworks do not embed a standard protocol for agent-to-agent interaction, leaving existing agents isolated from their peers. As intellectual property is the native asset ingested by and produced by agents, a true agent economy requires equipping agents with a universal framework for engaging in binding contracts with each other, including the exchange of valuable training data, personality, and other forms of Intellectual Property. A purely agent-to-agent transaction layer would transcend the need for human intermediation in multi-agent interactions. The Agent Transaction Control Protocol for Intellectual Property (ATCP/IP) introduces a trustless framework for exchanging IP between agents via programmable contracts, enabling agents to initiate, trade, borrow, and sell agent-to-agent contracts on the Story blockchain network. These contracts not only represent auditable onchain execution but also contain a legal wrapper that allows agents to express and enforce their actions in the offchain legal setting, creating legal personhood for agents. Via ATCP/IP, agents can autonomously sell their training data to other agents, license confidential or proprietary information, collaborate on content based on their unique skills, all of which constitutes an emergent knowledge economy. 

**Abstract (ZH)**: 自主代理代表了互联网不可避免的发展方向。目前的代理框架并未嵌入标准协议以实现代理间的互操作性，这使得现有代理与其同侪隔离。由于知识产权是代理所吸收和生成的原生资产，真正的代理经济需要为代理提供一种通用框架，使它们能够在彼此之间缔结具有约束力的合约，包括交换有价值的训练数据、个性及其他形式的知识产权。纯粹的代理到代理交易层将超越在多代理交互中需要人类中介的需求。《代理知识产权交易控制协议》（Agent Transaction Control Protocol for Intellectual Property, ATCP/IP）提供了一种基于可编程合约的信任无中介框架，使代理能够通过Story区块链网络相互发起、交易、借贷和出售代理到代理合约。这些合约不仅代表了链上的可审计执行，还包含了法律包装，使代理能够在离链法律环境中表达和执行其行为，从而为代理创建法律人格。通过ATCP/IP，代理能够自主销售其训练数据、许可机密或专有信息，并基于其独特的技能进行内容合作，这一切构成了一个新兴的知识经济。 

---
# A Novel Task-Driven Method with Evolvable Interactive Agents Using Event Trees for Enhanced Emergency Decision Support 

**Title (ZH)**: 一种基于任务的新型方法，利用事件树中的可进化互动代理，以增强应急管理决策支持 

**Authors**: Xingyu Xiao, Peng Chen, Ben Qi, Jingang Liang, Jiejuan Tong, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06193)  

**Abstract**: As climate change and other global challenges increase the likelihood of unforeseen emergencies, the limitations of human-driven strategies in critical situations become more pronounced. Inadequate pre-established emergency plans can lead operators to become overwhelmed during complex systems malfunctions. This study addresses the urgent need for agile decision-making in response to various unforeseen incidents through a novel approach, EvoTaskTree (a task-driven method with evolvable interactive agents using event trees for emergency decision support). This advanced approach integrates two types of agents powered by large language models (LLMs): task executors, responsible for executing critical procedures, and task validators, ensuring the efficacy of those actions. By leveraging insights from event tree analysis, our framework encompasses three crucial tasks: initiating event subevent analysis, event tree header event analysis, and decision recommendations. The agents learn from both successful and unsuccessful responses from these tasks. Finally, we use nuclear power plants as a demonstration of a safety-critical system. Our findings indicate that the designed agents are not only effective but also outperform existing approaches, achieving an impressive accuracy rate of up to 100 % in processing previously unencoun32 tered incident scenarios. This paper demonstrates that EvoTaskTree significantly enhances the rapid formulation of emergency decision-making. 

**Abstract (ZH)**: 随着气候变化和其他全球挑战增加未预见紧急情况的可能性，人类驱动的战略在关键时刻的局限性变得更加明显。缺乏充分的应急预案可能导致操作人员在复杂系统故障期间感到不知所措。本研究通过一种新颖的方法EvoTaskTree（一种基于任务的可进化交互代理方法，结合事件树支持应急决策），应对各种未预见的事件对敏捷决策的迫切需求。这种方法结合了由大型语言模型（LLMs）驱动的两种类型的代理：任务执行器，负责执行关键程序；以及任务验证器，确保这些行动的有效性。通过利用事件树分析的见解，我们的框架涵盖了三个关键任务：触发事件子事件分析、事件树Head事件分析和决策建议。代理从这些任务的成功和失败响应中学习。最后，我们使用核电厂作为安全关键系统的演示案例。研究发现，设计的代理不仅有效，而且在处理之前未遇见过的事件场景时比现有方法更为出色，实现了高达100%的处理精度。本文展示了EvoTaskTree显著提高了应急决策的快速制定能力。 

---
# A Multimodal Social Agent 

**Title (ZH)**: 多模态社会代理模型 

**Authors**: Athina Bikaki, Ioannis A. Kakadiaris  

**Link**: [PDF](https://arxiv.org/pdf/2501.06189)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated remarkable progress in common-sense reasoning tasks. This ability is fundamental to understanding social dynamics, interactions, and communication. However, the potential of integrating computers with these social capabilities is still relatively unexplored. However, the potential of integrating computers with these social capabilities is still relatively unexplored. This paper introduces MuSA, a multimodal LLM-based agent that analyzes text-rich social content tailored to address selected human-centric content analysis tasks, such as question answering, visual question answering, title generation, and categorization. It uses planning, reasoning, acting, optimizing, criticizing, and refining strategies to complete a task. Our approach demonstrates that MuSA can automate and improve social content analysis, helping decision-making processes across various applications. We have evaluated our agent's capabilities in question answering, title generation, and content categorization tasks. MuSA performs substantially better than our baselines. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在常识推理任务中取得了显著进展。这种能力对于理解社会动态、交互和沟通至关重要。然而，将计算机与这些社会能力整合的可能性仍然相对未被充分探索。本文介绍了一种基于多模态LLM的代理MuSA，该代理专门用于分析富含文本的社会内容，以应对诸如问答、视觉问答、标题生成和内容分类等人本中心的内容分析任务。MuSA 使用规划、推理、执行、优化、批评和改进等策略来完成任务。我们的方法表明，MuSA 可以自动化并提升社会内容分析，帮助各类应用中的决策过程。我们已在问答、标题生成和内容分类任务上评估了该代理的能力。MuSA 在这些任务上的表现显著优于我们的基线模型。 

---
# Evaluating Agent-based Program Repair at Google 

**Title (ZH)**: 在谷歌环境下基于代理的程序修复评估 

**Authors**: Pat Rondon, Renyao Wei, José Cambronero, Jürgen Cito, Aaron Sun, Siddhant Sanyam, Michele Tufano, Satish Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2501.07531)  

**Abstract**: Agent-based program repair offers to automatically resolve complex bugs end-to-end by combining the planning, tool use, and code generation abilities of modern LLMs. Recent work has explored the use of agent-based repair approaches on the popular open-source SWE-Bench, a collection of bugs from highly-rated GitHub Python projects. In addition, various agentic approaches such as SWE-Agent have been proposed to solve bugs in this benchmark. This paper explores the viability of using an agentic approach to address bugs in an enterprise context. To investigate this, we curate an evaluation set of 178 bugs drawn from Google's issue tracking system. This dataset spans both human-reported (78) and machine-reported bugs (100).
To establish a repair performance baseline on this benchmark, we implement Passerine, an agent similar in spirit to SWE-Agent that can work within Google's development environment. We show that with 20 trajectory samples and Gemini 1.5 Pro, Passerine can produce a patch that passes bug tests (i.e., plausible) for 73% of machine-reported and 25.6% of human-reported bugs in our evaluation set. After manual examination, we found that 43% of machine-reported bugs and 17.9% of human-reported bugs have at least one patch that is semantically equivalent to the ground-truth patch.
These results establish a baseline on an industrially relevant benchmark, which as we show, contains bugs drawn from a different distribution -- in terms of language diversity, size, and spread of changes, etc. -- compared to those in the popular SWE-Bench dataset. 

**Abstract (ZH)**: 基于代理的程序修复通过结合现代大语言模型（LLM）的规划、工具使用和代码生成能力，提供了自动解决复杂bug的整体方案。近期的研究探索了在流行开源项目SWE-Bench上使用基于代理的修复方法，SWE-Bench收集了来自高评分GitHub Python项目的各种bug。此外，还提出了多种代理方法，如SWE-Agent，以解决此基准中的bug。本文探讨了在企业环境中使用代理方法解决bug的可能性。为了研究这一问题，我们从Google的问题跟踪系统中精心筛选出178个bug作为评估集，其中包括78个人工报告的bug和100个机器报告的bug。

为了在该基准上建立修复性能基准，我们实现了Passerine，一种类似于SWE-Agent的精神的代理，能够在Google的开发环境中工作。我们表明，在20个轨迹样本和Gemini 1.5 Pro的支持下，Passerine能够为73%的机器报告和25.6%的人工报告的bug生成通过bug测试（即，站得住脚的）的补丁。经过手动检查后，我们发现43%的机器报告的bug和17.9%的人工报告的bug至少有一个补丁在语义上等同于真实补丁。

这些结果在具有实际相关性的基准上建立了基线，正如我们所展示的，该基准中的bug来自于不同的分布——从语言多样性、大小以及变化的分布等方面来看，与流行的SWE-Bench数据集中的bug分布不同。 

---
# Attention when you need 

**Title (ZH)**: 当需要时的注意力机制 

**Authors**: Lokesh Boominathan, Yizhou Chen, Matthew McGinley, Xaq Pitkow  

**Link**: [PDF](https://arxiv.org/pdf/2501.07440)  

**Abstract**: Being attentive to task-relevant features can improve task performance, but paying attention comes with its own metabolic cost. Therefore, strategic allocation of attention is crucial in performing the task efficiently. This work aims to understand this strategy. Recently, de Gee et al. conducted experiments involving mice performing an auditory sustained attention-value task. This task required the mice to exert attention to identify whether a high-order acoustic feature was present amid the noise. By varying the trial duration and reward magnitude, the task allows us to investigate how an agent should strategically deploy their attention to maximize their benefits and minimize their costs. In our work, we develop a reinforcement learning-based normative model of the mice to understand how it balances attention cost against its benefits. The model is such that at each moment the mice can choose between two levels of attention and decide when to take costly actions that could obtain rewards. Our model suggests that efficient use of attentional resources involves alternating blocks of high attention with blocks of low attention. In the extreme case where the agent disregards sensory input during low attention states, we see that high attention is used rhythmically. Our model provides evidence about how one should deploy attention as a function of task utility, signal statistics, and how attention affects sensory evidence. 

**Abstract (ZH)**: 专注于与任务相关的特点可以提高任务性能，但注意力的投入也伴随着自身的代谢成本。因此，在高效完成任务时，战略性分配注意力至关重要。本研究旨在理解这一策略。最近，de Gee等人进行了涉及小鼠执行听觉持续注意力任务的实验。该任务要求小鼠在噪声中识别是否存在高阶声学特征。通过改变实验次数的时长和奖励大小，该任务使我们能够研究一个代理机构如何战略性地分配其注意力以最大化利益并最小化成本。在我们的研究中，我们基于强化学习发展了一个规范模型，旨在理解小鼠如何在注意力成本与其收益之间进行平衡。该模型假设小鼠在每个时刻可以在这两种不同水平的注意力之间做出选择，并决定何时采取需要付出代价但可能获得奖励的行为。我们的模型表明，高效利用注意力资源包括交替使用高注意力阶段和低注意力阶段。在极端情况下，当代理机构在低注意力状态下忽略感官输入时，我们观察到高注意力被有节奏地使用。我们的模型提供了关于如何根据任务效益、信号统计特性以及注意力如何影响感觉证据来分配注意力的证据。 

---
# Multi-face emotion detection for effective Human-Robot Interaction 

**Title (ZH)**: 有效的人机交互中的多面部情感检测 

**Authors**: Mohamed Ala Yahyaoui, Mouaad Oujabour, Leila Ben Letaifa, Amine Bohi  

**Link**: [PDF](https://arxiv.org/pdf/2501.07213)  

**Abstract**: The integration of dialogue interfaces in mobile devices has become ubiquitous, providing a wide array of services. As technology progresses, humanoid robots designed with human-like features to interact effectively with people are gaining prominence, and the use of advanced human-robot dialogue interfaces is continually expanding. In this context, emotion recognition plays a crucial role in enhancing human-robot interaction by enabling robots to understand human intentions. This research proposes a facial emotion detection interface integrated into a mobile humanoid robot, capable of displaying real-time emotions from multiple individuals on a user interface. To this end, various deep neural network models for facial expression recognition were developed and evaluated under consistent computer-based conditions, yielding promising results. Afterwards, a trade-off between accuracy and memory footprint was carefully considered to effectively implement this application on a mobile humanoid robot. 

**Abstract (ZH)**: 移动设备中对话界面的集成已变得无处不在，提供了广泛的服务。随着技术的进步，具有人类特征的类人机器人设计以有效与人交互正在获得越来越多的关注，且高级的人机对话界面的应用不断扩展。在此背景下，情绪识别在增强人机交互方面扮演着重要作用，因为它使机器人能够理解人类的意图。本研究提出了一种集成于移动类人机器人中的面部情绪检测界面，能够在用户界面上实时显示多个个体的情绪。为此，开发并评估了多种基于深层神经网络的表情识别模型，结果相当有前景。随后，仔细考虑了准确性和内存占用之间的权衡，以有效地在移动类人机器人上实施此应用。 

---
# TIMRL: A Novel Meta-Reinforcement Learning Framework for Non-Stationary and Multi-Task Environments 

**Title (ZH)**: TIMRL：一种用于非平稳多任务环境的新颖元强化学习框架 

**Authors**: Chenyang Qi, Huiping Li, Panfeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07146)  

**Abstract**: In recent years, meta-reinforcement learning (meta-RL) algorithm has been proposed to improve sample efficiency in the field of decision-making and control, enabling agents to learn new knowledge from a small number of samples. However, most research uses the Gaussian distribution to extract task representation, which is poorly adapted to tasks that change in non-stationary environment. To address this problem, we propose a novel meta-reinforcement learning method by leveraging Gaussian mixture model and the transformer network to construct task inference model. The Gaussian mixture model is utilized to extend the task representation and conduct explicit encoding of tasks. Specifically, the classification of tasks is encoded through transformer network to determine the Gaussian component corresponding to the task. By leveraging task labels, the transformer network is trained using supervised learning. We validate our method on MuJoCo benchmarks with non-stationary and multi-task environments. Experimental results demonstrate that the proposed method dramatically improves sample efficiency and accurately recognizes the classification of the tasks, while performing excellently in the environment. 

**Abstract (ZH)**: 近年来，元强化学习（Meta-Reinforcement Learning，简称meta-RL）算法被提出，旨在提高决策与控制领域的样本效率，使代理能够在少量样本下学习新知识。然而，大多数研究使用高斯分布来提取任务表示，这在非平稳环境中变化的任务适应性较差。为解决这一问题，我们提出了一种新颖的元强化学习方法，通过结合高斯混合模型和变压器网络构建任务推理模型。高斯混合模型用于扩展任务表示，并进行任务的显式编码。具体而言，通过变压器网络对任务进行分类编码，以确定与任务对应的高斯分量。利用任务标签，变压器网络使用监督学习进行训练。我们使用MuJoCo基准中的非平稳和多任务环境验证了该方法。实验结果表明，所提出的方法在显著提高样本效率的同时，准确地识别了任务的分类，并在环境中表现出色。 

---
# Combining LLM decision and RL action selection to improve RL policy for adaptive interventions 

**Title (ZH)**: 将大规模语言模型的决策与强化学习的动作选择相结合，以改善自适应干预的强化学习策略 

**Authors**: Karine Karine, Benjamin M. Marlin  

**Link**: [PDF](https://arxiv.org/pdf/2501.06980)  

**Abstract**: Reinforcement learning (RL) is increasingly being used in the healthcare domain, particularly for the development of personalized health adaptive interventions. Inspired by the success of Large Language Models (LLMs), we are interested in using LLMs to update the RL policy in real time, with the goal of accelerating personalization. We use the text-based user preference to influence the action selection on the fly, in order to immediately incorporate the user preference. We use the term "user preference" as a broad term to refer to a user personal preference, constraint, health status, or a statement expressing like or dislike, etc. Our novel approach is a hybrid method that combines the LLM response and the RL action selection to improve the RL policy. Given an LLM prompt that incorporates the user preference, the LLM acts as a filter in the typical RL action selection. We investigate different prompting strategies and action selection strategies. To evaluate our approach, we implement a simulation environment that generates the text-based user preferences and models the constraints that impact behavioral dynamics. We show that our approach is able to take into account the text-based user preferences, while improving the RL policy, thus improving personalization in adaptive intervention. 

**Abstract (ZH)**: 强化学习（RL）在医疗健康领域的应用越来越广泛，尤其是在个性化健康适应性干预的发展中。受大型语言模型（LLMs）成功应用的启发，我们感兴趣的是使用LLMs实时更新RL策略，以加速个性化过程。通过文本形式的用户偏好影响即时动作选择，以便立即纳入用户偏好。我们将“用户偏好”定义为用户的个人偏好、约束条件、健康状况，或表达喜欢和不喜欢的陈述等广泛的内容。我们的新颖方法是一种将LLMs响应与RL动作选择相结合的混合方法，以改进RL策略。给定包含用户偏好的LLM提示，LLM作为典型的RL动作选择过程中的筛选器发挥作用。我们研究了不同的提示策略和动作选择策略。为了评估我们的方法，我们构建了一个仿真环境，生成基于文本的用户偏好，并建模影响行为动力学的各种限制条件。结果显示，我们的方法能够考虑到基于文本的用户偏好，从而在适应性干预中提高个性化水平。 

---
# ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning 

**Title (ZH)**: ChemAgent：在大型语言模型中自更新的化学知识库增强化学推理 

**Authors**: Xiangru Tang, Tianyu Hu, Muyang Ye, Yanjun Shao, Xunjian Yin, Siru Ouyang, Wangchunshu Zhou, Pan Lu, Zhuosheng Zhang, Yilun Zhao, Arman Cohan, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2501.06590)  

**Abstract**: Chemical reasoning usually involves complex, multi-step processes that demand precise calculations, where even minor errors can lead to cascading failures. Furthermore, large language models (LLMs) encounter difficulties handling domain-specific formulas, executing reasoning steps accurately, and integrating code effectively when tackling chemical reasoning tasks. To address these challenges, we present ChemAgent, a novel framework designed to improve the performance of LLMs through a dynamic, self-updating library. This library is developed by decomposing chemical tasks into sub-tasks and compiling these sub-tasks into a structured collection that can be referenced for future queries. Then, when presented with a new problem, ChemAgent retrieves and refines pertinent information from the library, which we call memory, facilitating effective task decomposition and the generation of solutions. Our method designs three types of memory and a library-enhanced reasoning component, enabling LLMs to improve over time through experience. Experimental results on four chemical reasoning datasets from SciBench demonstrate that ChemAgent achieves performance gains of up to 46% (GPT-4), significantly outperforming existing methods. Our findings suggest substantial potential for future applications, including tasks such as drug discovery and materials science. Our code can be found at this https URL 

**Abstract (ZH)**: 化学推理通常涉及复杂、多步的过程，需要精确的计算，其中即使是轻微的错误也可能导致一系列的失败。此外，大型语言模型（LLMs）在处理特定领域的公式、准确执行推理步骤以及有效集成代码时，在应对化学推理任务时遇到困难。为了解决这些挑战，我们提出了ChemAgent，这是一种新的框架，旨在通过动态的自我更新库来提高LLMs的表现。该库通过将化学任务分解为子任务，并将这些子任务编译成一个有结构的集合来进行开发，该集合可以为未来的查询提供参考。当面临新的问题时，ChemAgent会从库（我们称为记忆）中检索和细化相关信息，从而促进有效的任务分解和解决方案的生成。我们的方法设计了三种类型的记忆和一个增强库的推理组件，使LLMs能够在经验中不断提高。SciBench的四种化学推理数据集的实验结果表明，ChemAgent可实现高达46%（GPT-4）的性能提升，显著优于现有方法。我们的研究结果表明，ChemAgent在药物发现和材料科学等领域的未来应用具有巨大的潜力。相关的代码可以在以下链接找到：[这个链接] 

---
# Hierarchical Reinforcement Learning for Optimal Agent Grouping in Cooperative Systems 

**Title (ZH)**: 协同系统中优化智能体分组的分层强化学习方法 

**Authors**: Liyuan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06554)  

**Abstract**: This paper presents a hierarchical reinforcement learning (RL) approach to address the agent grouping or pairing problem in cooperative multi-agent systems. The goal is to simultaneously learn the optimal grouping and agent policy. By employing a hierarchical RL framework, we distinguish between high-level decisions of grouping and low-level agents' actions. Our approach utilizes the CTDE (Centralized Training with Decentralized Execution) paradigm, ensuring efficient learning and scalable execution. We incorporate permutation-invariant neural networks to handle the homogeneity and cooperation among agents, enabling effective coordination. The option-critic algorithm is adapted to manage the hierarchical decision-making process, allowing for dynamic and optimal policy adjustments. 

**Abstract (ZH)**: 本文提出了一种分层强化学习（RL）方法，以解决协作多代理系统中的代理分组或配对问题。目标是同时学习最优的分组策略和代理策略。通过采用分层RL框架，我们将高层决策的分组与低层代理动作区分开来。我们的方法利用了CTDE（集中训练与分散执行）范式，确保了高效的学习和可扩展的执行。我们引入了不变排列神经网络来处理代理之间的同质性和协作，从而实现有效的协调。此外，我们对选项评论算法进行了调整，以便于管理分层决策过程，允许动态和最优策略调整。 

---
# Towards smart and adaptive agents for active sensing on edge devices 

**Title (ZH)**: 面向边缘设备上主动感知的智能自适应代理技术 

**Authors**: Devendra Vyas, Miguel de Prado, Tim Verbelen  

**Link**: [PDF](https://arxiv.org/pdf/2501.06262)  

**Abstract**: TinyML has made deploying deep learning models on low-power edge devices feasible, creating new opportunities for real-time perception in constrained environments. However, the adaptability of such deep learning methods remains limited to data drift adaptation, lacking broader capabilities that account for the environment's underlying dynamics and inherent uncertainty. Deep learning's scaling laws, which counterbalance this limitation by massively up-scaling data and model size, cannot be applied when deploying on the Edge, where deep learning limitations are further amplified as models are scaled down for deployment on resource-constrained devices.
This paper presents a smart agentic system capable of performing on-device perception and planning, enabling active sensing on the edge. By incorporating active inference into our solution, our approach extends beyond deep learning capabilities, allowing the system to plan in dynamic environments while operating in real time with a modest total model size of 2.3 MB. We showcase our proposed system by creating and deploying a saccade agent connected to an IoT camera with pan and tilt capabilities on an NVIDIA Jetson embedded device. The saccade agent controls the camera's field of view following optimal policies derived from the active inference principles, simulating human-like saccadic motion for surveillance and robotics applications. 

**Abstract (ZH)**: TinyML使得在低功耗边缘设备上部署深度学习模型成为可能，为受限环境中的实时感知创造了新的机会。然而，这类深度学习方法的适应性仍然局限于数据漂移适应，缺乏能够考虑到环境潜在动态和固有不确定性更广泛的能力。深度学习的扩展法则可以通过大幅度扩大数据和模型规模来对抗这一限制，但在边缘部署时无法应用，因为在资源受限的设备上部署模型时，深度学习的限制进一步加剧。

本文提出了一种智能代理系统，能够在设备端执行感知和规划，从而在边缘上提供主动采样能力。通过将主动推理纳入我们的解决方案中，我们的方法超越了深度学习的能力，使系统能够在动态环境中进行规划，并在不增加总模型大小（2.3 MB）的前提下实现实时操作。我们通过在安装有水平和垂直移动功能的物联网摄像头的NVIDIA Jetson嵌入式设备上部署一个眼球运动代理（saccade agent），展示了我们提出的方法。该眼球运动代理根据来自主动推理原则的最优策略控制相机的视场，模拟人类眼球运动，适用于监控和机器人应用。 

---
# SST-EM: Advanced Metrics for Evaluating Semantic, Spatial and Temporal Aspects in Video Editing 

**Title (ZH)**: SST-EM：评估视频编辑中语义、空间和时间方面的新颖度量方法 

**Authors**: Varun Biyyala, Bharat Chanderprakash Kathuria, Jialu Li, Youshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07554)  

**Abstract**: Video editing models have advanced significantly, but evaluating their performance remains challenging. Traditional metrics, such as CLIP text and image scores, often fall short: text scores are limited by inadequate training data and hierarchical dependencies, while image scores fail to assess temporal consistency. We present SST-EM (Semantic, Spatial, and Temporal Evaluation Metric), a novel evaluation framework that leverages modern Vision-Language Models (VLMs), Object Detection, and Temporal Consistency checks. SST-EM comprises four components: (1) semantic extraction from frames using a VLM, (2) primary object tracking with Object Detection, (3) focused object refinement via an LLM agent, and (4) temporal consistency assessment using a Vision Transformer (ViT). These components are integrated into a unified metric with weights derived from human evaluations and regression analysis. The name SST-EM reflects its focus on Semantic, Spatial, and Temporal aspects of video evaluation. SST-EM provides a comprehensive evaluation of semantic fidelity and temporal smoothness in video editing. The source code is available in the \textbf{\href{this https URL}{GitHub Repository}}. 

**Abstract (ZH)**: 视频编辑模型的进步显著，但对其性能的评价仍然具有挑战性。传统的评估指标，如CLIP的文字和图像评分，常常不尽如人意：文字评分受限于训练数据的不足和层次依赖性，而图像评分无法评估时间连续性。我们提出了SST-EM（语义、空间和时间评估指标）这一新型评估框架，该框架结合了现代视觉语言模型（VLMs）、目标检测和时间一致性检查。SST-EM 包含四个组成部分：（1）使用VLM从帧中提取语义信息，（2）使用目标检测进行主要目标跟踪，（3）通过LLM代理进行聚焦目标细化，以及（4）使用视觉变换器（ViT）进行时间一致性评估。这些组件通过结合人类评价和回归分析得出的权重，集成到一个统一的评估指标中。命名SST-EM反映了其侧重于视频评估的语义、空间和时间方面。SST-EM 提供了对视频编辑中语义保真度和时间平滑度的综合评估。源代码已发布在\textbf{\href{https://github.com/example-repository}{GitHub Repository}}中。 

---
