# AIDE: AI-Driven Exploration in the Space of Code 

**Title (ZH)**: AIDE：AI 驱动的代码空间探索 

**Authors**: Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, Yuxiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13138)  

**Abstract**: Machine learning, the foundation of modern artificial intelligence, has driven innovations that have fundamentally transformed the world. Yet, behind advancements lies a complex and often tedious process requiring labor and compute intensive iteration and experimentation. Engineers and scientists developing machine learning models spend much of their time on trial-and-error tasks instead of conceptualizing innovative solutions or research hypotheses. To address this challenge, we introduce AI-Driven Exploration (AIDE), a machine learning engineering agent powered by large language models (LLMs). AIDE frames machine learning engineering as a code optimization problem, and formulates trial-and-error as a tree search in the space of potential solutions. By strategically reusing and refining promising solutions, AIDE effectively trades computational resources for enhanced performance, achieving state-of-the-art results on multiple machine learning engineering benchmarks, including our Kaggle evaluations, OpenAI MLE-Bench and METRs RE-Bench. 

**Abstract (ZH)**: 机器学习，现代人工智能的基础，已经推动了一系列创新，这些创新从根本上改变了世界。然而，这些进步背后隐藏着一个复杂且往往耗时的过程，需要大量的劳动和计算密集型迭代与实验。开发机器学习模型的工程师和科学家们大量时间花费在试错任务上，而不是构思创新的解决方案或研究假说。为解决这一挑战，我们引入了AI驱动探索（AIDE），这是一种由大规模语言模型（LLMs）驱动的机器学习工程代理。AIDE将机器学习工程视为代码优化问题，将试错过程视为在潜在解决方案空间中的树搜索问题。通过战略性地重用和改进有希望的解决方案，AIDE有效地用计算资源换取更好的性能，其结果在多个机器学习工程基准测试上达到了最先进的水平，包括我们在Kaggle的评估、OpenAI MLE-Bench和METRS RE-Bench的结果中得到了验证。 

---
# Interactive Agents to Overcome Ambiguity in Software Engineering 

**Title (ZH)**: 软件工程中克服模糊性的交互式代理 

**Authors**: Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2502.13069)  

**Abstract**: AI agents are increasingly being deployed to automate tasks, often based on ambiguous and underspecified user instructions. Making unwarranted assumptions and failing to ask clarifying questions can lead to suboptimal outcomes, safety risks due to tool misuse, and wasted computational resources. In this work, we study the ability of LLM agents to handle ambiguous instructions in interactive code generation settings by evaluating proprietary and open-weight models on their performance across three key steps: (a) leveraging interactivity to improve performance in ambiguous scenarios, (b) detecting ambiguity, and (c) asking targeted questions. Our findings reveal that models struggle to distinguish between well-specified and underspecified instructions. However, when models interact for underspecified inputs, they effectively obtain vital information from the user, leading to significant improvements in performance and underscoring the value of effective interaction. Our study highlights critical gaps in how current state-of-the-art models handle ambiguity in complex software engineering tasks and structures the evaluation into distinct steps to enable targeted improvements. 

**Abstract (ZH)**: 人工智能代理正越来越多地被部署以自动化任务，通常是基于含糊不清且未充分指定的用户指令。做出不适当的假设和未提出澄清问题可能导致次优结果、因工具误用而带来的安全风险以及计算资源的浪费。在这项研究中，我们通过评估专有和开源模型在三个关键步骤中的表现，研究了语言模型代理在这种交互式代码生成设置中处理含糊指令的能力：(a) 利用互动来改善含糊场景中的表现，(b) 检测含糊性，以及(c) 提出有针对性的问题。我们的研究结果表明，模型难以区分明确指定与未充分指定的指令。然而，对于未充分指定的输入，当模型进行互动时，能够有效地从用户那里获得关键信息，从而显著提高表现，进一步突出了高效互动的价值。我们的研究指出了当前最先进的模型在复杂软件工程任务中处理含糊性时存在的重要空白，并将评估结构化为不同的步骤，以促进有针对性的改进。 

---
# Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks 

**Title (ZH)**: 代理深度图推理生成自我组织的知识网络 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2502.13025)  

**Abstract**: We present an agentic, autonomous graph expansion framework that iteratively structures and refines knowledge in situ. Unlike conventional knowledge graph construction methods relying on static extraction or single-pass learning, our approach couples a reasoning-native large language model with a continually updated graph representation. At each step, the system actively generates new concepts and relationships, merges them into a global graph, and formulates subsequent prompts based on its evolving structure. Through this feedback-driven loop, the model organizes information into a scale-free network characterized by hub formation, stable modularity, and bridging nodes that link disparate knowledge clusters. Over hundreds of iterations, new nodes and edges continue to appear without saturating, while centrality measures and shortest path distributions evolve to yield increasingly distributed connectivity. Our analysis reveals emergent patterns, such as the rise of highly connected 'hub' concepts and the shifting influence of 'bridge' nodes, indicating that agentic, self-reinforcing graph construction can yield open-ended, coherent knowledge structures. Applied to materials design problems, we present compositional reasoning experiments by extracting node-specific and synergy-level principles to foster genuinely novel knowledge synthesis, yielding cross-domain ideas that transcend rote summarization and strengthen the framework's potential for open-ended scientific discovery. We discuss other applications in scientific discovery and outline future directions for enhancing scalability and interpretability. 

**Abstract (ZH)**: 我们提出了一种代理自主的图扩展框架，该框架能够迭代地在现场构建和精炼知识。与依赖静态抽取或单次学习的传统知识图谱构建方法不同，我们的方法将推理能力强大的大型语言模型与不断更新的图表示相结合。在每一步中，系统主动生成新概念和关系，将它们合并到全局图中，并基于其不断变化的结构制定后续提示。通过这种反馈驱动的循环，模型将信息组织成一个无标度网络，该网络由核心节点的形成、模块的稳定性以及连接不同知识点集群的桥梁节点来表征。在数百次迭代中，新的节点和边继续出现而不饱和，同时中心性度量和最短路径分布发生变化，从而实现更加分散的连接性。我们的分析揭示出一些新兴的模式，例如高度连接的核心概念的出现及其桥梁节点影响力的转变，表明代理自主、自我强化的图构建可以产生开放式的、连贯的知识结构。将其应用于材料设计问题时，我们通过提取节点特定和协同作用层次的原则，开展了组合理论推理实验，以促进真正的新颖知识合成，产生跨越领域的新颖想法，超越机械式的总结，从而增强了框架在开放科学发现中的潜力。我们讨论了其他在科学发现中的应用，并概述了未来增强可扩展性和解释性的方向。 

---
# Integrating Reinforcement Learning, Action Model Learning, and Numeric Planning for Tackling Complex Tasks 

**Title (ZH)**: 将强化学习、动作模型学习和数值规划集成以应对复杂任务 

**Authors**: Yarin Benyamin, Argaman Mordoch, Shahaf S. Shperberg, Roni Stern  

**Link**: [PDF](https://arxiv.org/pdf/2502.13006)  

**Abstract**: Automated Planning algorithms require a model of the domain that specifies the preconditions and effects of each action. Obtaining such a domain model is notoriously hard. Algorithms for learning domain models exist, yet it remains unclear whether learning a domain model and planning is an effective approach for numeric planning environments, i.e., where states include discrete and numeric state variables. In this work, we explore the benefits of learning a numeric domain model and compare it with alternative model-free solutions. As a case study, we use two tasks in Minecraft, a popular sandbox game that has been used as an AI challenge. First, we consider an offline learning setting, where a set of expert trajectories are available to learn from. This is the standard setting for learning domain models. We used the Numeric Safe Action Model Learning (NSAM) algorithm to learn a numeric domain model and solve new problems with the learned domain model and a numeric planner. We call this model-based solution NSAM_(+p), and compare it to several model-free Imitation Learning (IL) and Offline Reinforcement Learning (RL) algorithms. Empirical results show that some IL algorithms can learn faster to solve simple tasks, while NSAM_(+p) allows solving tasks that require long-term planning and enables generalizing to solve problems in larger environments. Then, we consider an online learning setting, where learning is done by moving an agent in the environment. For this setting, we introduce RAMP. In RAMP, observations collected during the agent's execution are used to simultaneously train an RL policy and learn a planning domain action model. This forms a positive feedback loop between the RL policy and the learned domain model. We demonstrate experimentally the benefits of using RAMP, showing that it finds more efficient plans and solves more problems than several RL baselines. 

**Abstract (ZH)**: 自动生成规划算法需要一个描述领域模型，其中明确规定了每个操作的前提条件和效应。获取这样的领域模型本身就非常困难。学习领域模型的算法已经存在，但尚不清楚在包含离散和数值状态变量的数值规划环境中，学习领域模型并进行规划是否是一种有效的方法。本文探讨了学习数值领域模型的益处，并将其与无模型解决方案进行了比较。作为案例研究，我们使用了《我的世界》（Minecraft）中两个任务，这是一款流行的沙盒游戏，并且该游戏已被用作人工智能挑战。首先，我们在离线学习环境中进行实验，此时可用一组专家轨迹进行学习。这是学习领域模型的标准设置。我们使用了数值安全行动模型学习（NSAM）算法来学习数值领域模型，并通过学习得到的领域模型和数值规划器来解决新问题。我们将这种基于模型的解决方案称为NSAM_(+p)，并将其与几种无模型的模仿学习（IL）和离线强化学习（RL）算法进行了比较。实验结果表明，某些IL算法可以更快地学习解决简单任务，而NSAM_(+p)能够解决需要长期规划的任务，并且能够在更大的环境中解决问题。接下来，我们考虑在线学习环境，其中学习是通过在环境中移动一个代理来完成的。为此，我们引入了RAMP。在RAMP中，代理执行过程中收集的观察结果被用来同时训练RL策略并学习规划领域操作模型。这形成了RL策略和学习到的领域模型之间的正反馈循环。我们通过实验展示了RAMP的优势，表明它找到的计划更有效，并且能够解决比几种RL基线更多的问题。 

---
# You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with a Multi-Agent Conversations 

**Title (ZH)**: 你需要模拟以获得名声：利用多代理对话解决会议纪要稀缺性问题 

**Authors**: Frederic Kirstein, Muneeb Khan, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.13001)  

**Abstract**: Meeting summarization suffers from limited high-quality data, mainly due to privacy restrictions and expensive collection processes. We address this gap with FAME, a dataset of 500 meetings in English and 300 in German produced by MIMIC, our new multi-agent meeting synthesis framework that generates meeting transcripts on a given knowledge source by defining psychologically grounded participant profiles, outlining the conversation, and orchestrating a large language model (LLM) debate. A modular post-processing step refines these outputs, mitigating potential repetitiveness and overly formal tones, ensuring coherent, credible dialogues at scale. We also propose a psychologically grounded evaluation framework assessing naturalness, social behavior authenticity, and transcript difficulties. Human assessments show that FAME approximates real-meeting spontaneity (4.5/5 in naturalness), preserves speaker-centric challenges (3/5 in spoken language), and introduces richer information-oriented difficulty (4/5 in difficulty). These findings highlight that FAME is a good and scalable proxy for real-world meeting conditions. It enables new test scenarios for meeting summarization research and other conversation-centric applications in tasks requiring conversation data or simulating social scenarios under behavioral constraints. 

**Abstract (ZH)**: 会议总结因缺乏高质量数据而受限，主要原因是隐私限制和昂贵的收集过程。我们通过提出FAME数据集来填补这一空白，该数据集包含500个英语会议和300个德语会议，是由我们新开发的多智能体会议合成框架MIMIC生成的。MIMIC框架通过定义基于心理学原理的参与者角色、规划对话内容，并协调大规模语言模型（LLM）辩论来生成给定知识源的会议记录。一个模块化的后处理步骤进一步细化这些输出，减少了潜在的重复性和过于正式的语气，确保了大规模对话的连贯性和可信度。我们还提出了一种基于心理学的评估框架，评估自然性、社会行为的真实性以及记录的难度。人类评估结果显示，FAME在自然性（4.5/5）上接近真实的会议自发性，保留了以讲演者为中心的挑战（3/5在口语方面），并引入了更丰富的信息导向难点（4/5在难度上）。这些发现表明，FAME是一个良好的且可扩展的现实会议条件的代理。它为会议总结研究和需要对话数据或其他对话中心应用的任务提供了新的测试场景，特别是在行为约束下模拟社交场景时。 

---
# Free Argumentative Exchanges for Explaining Image Classifiers 

**Title (ZH)**: 自由论辩交流以解释图像分类器 

**Authors**: Avinash Kori, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2502.12995)  

**Abstract**: Deep learning models are powerful image classifiers but their opacity hinders their trustworthiness. Explanation methods for capturing the reasoning process within these classifiers faithfully and in a clear manner are scarce, due to their sheer complexity and size. We provide a solution for this problem by defining a novel method for explaining the outputs of image classifiers with debates between two agents, each arguing for a particular class. We obtain these debates as concrete instances of Free Argumentative eXchanges (FAXs), a novel argumentation-based multi-agent framework allowing agents to internalise opinions by other agents differently than originally stated. We define two metrics (consensus and persuasion rate) to assess the usefulness of FAXs as argumentative explanations for image classifiers. We then conduct a number of empirical experiments showing that FAXs perform well along these metrics as well as being more faithful to the image classifiers than conventional, non-argumentative explanation methods. All our implementations can be found at this https URL. 

**Abstract (ZH)**: 深度学习模型是强大的图像分类器，但它们的不透明性阻碍了人们对它们的信任。由于这些分类器极其复杂和庞大，用于忠实而清晰地捕捉其推理过程的解释方法极为稀缺。为解决这一问题，我们提出了一种新颖的方法，通过两个代理之间的辩论来解释图像分类器的输出。这些辩论被具体化为一种新型的基于论证的多代理框架的具体实例——自由论辩交换（FAXs），该框架使得代理能够以不同于原始陈述的方式内化其他代理的意见。我们定义了两个指标（共识率和说服率）来评估FAXs作为图像分类器的论辩解释的有效性。随后，我们进行了若干实证实验，表明FAXs在这些指标上表现良好，并且相比传统的非论辩解释方法更为忠实于图像分类器。我们的所有实现可以在以下链接找到：[这个 https URL]。 

---
# Towards more Contextual Agents: An extractor-Generator Optimization Framework 

**Title (ZH)**: 更加贴近语境的智能代理：提取-生成优化框架 

**Authors**: Mourad Aouini, Jinan Loubani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12926)  

**Abstract**: Large Language Model (LLM)-based agents have demonstrated remarkable success in solving complex tasks across a wide range of general-purpose applications. However, their performance often degrades in context-specific scenarios, such as specialized industries or research domains, where the absence of domain-relevant knowledge leads to imprecise or suboptimal outcomes. To address this challenge, our work introduces a systematic approach to enhance the contextual adaptability of LLM-based agents by optimizing their underlying prompts-critical components that govern agent behavior, roles, and interactions. Manually crafting optimized prompts for context-specific tasks is labor-intensive, error-prone, and lacks scalability. In this work, we introduce an Extractor-Generator framework designed to automate the optimization of contextual LLM-based agents. Our method operates through two key stages: (i) feature extraction from a dataset of gold-standard input-output examples, and (ii) prompt generation via a high-level optimization strategy that iteratively identifies underperforming cases and applies self-improvement techniques. This framework substantially improves prompt adaptability by enabling more precise generalization across diverse inputs, particularly in context-specific tasks where maintaining semantic consistency and minimizing error propagation are critical for reliable performance. Although developed with single-stage workflows in mind, the approach naturally extends to multi-stage workflows, offering broad applicability across various agent-based systems. Empirical evaluations demonstrate that our framework significantly enhances the performance of prompt-optimized agents, providing a structured and efficient approach to contextual LLM-based agents. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的代理已经在广泛的一般性应用领域中展示了解决复杂任务的显著成功。然而，在特定情境或专业领域中，这类代理的表现通常会因缺乏相关领域知识而下降，导致产出不够精确或次优。为解决这一挑战，我们的研究引入了一种系统的方法来增强基于LLM的代理的上下文适应性，通过优化它们的基本提示——这些提示是决定代理行为、角色和互动的关键组件。手工为特定情境下的任务定制优化提示既耗时、容易出错，又缺乏可扩展性。在这项研究中，我们提出了一种提取-生成框架，旨在自动化优化基于上下文的LLM代理。具体方法通过两个关键阶段运作：(i) 从优质输入-输出示例数据集中提取特征，(ii) 通过高层次的优化策略生成提示，该策略逐次识别表现不佳的情况并应用自我改进技术。该框架通过使提示能够更准确地泛化到多样化输入，特别是在那些需要保持语义一致性并最小化错误传播才能确保可靠性能的特定情境任务中，显著增强了提示的适应性。虽然该方法最初是为目标任务设计的，但自然地适用于多阶段流程，具有在各种代理系统中广泛应用的潜力。经验评估表明，我们的方法显著增强了提示优化代理的表现，为上下文敏感的LLM代理提供了一种结构化和高效的方法。 

---
# Continuous Learning Conversational AI: A Personalized Agent Framework via A2C Reinforcement Learning 

**Title (ZH)**: 连续学习对话型AI：一种基于A2C强化学习的个性化代理框架 

**Authors**: Nandakishor M, Anjali M  

**Link**: [PDF](https://arxiv.org/pdf/2502.12876)  

**Abstract**: Creating personalized and adaptable conversational AI remains a key challenge. This paper introduces a Continuous Learning Conversational AI (CLCA) approach, implemented using A2C reinforcement learning, to move beyond static Large Language Models (LLMs). We use simulated sales dialogues, generated by LLMs, to train an A2C agent. This agent learns to optimize conversation strategies for personalization, focusing on engagement and delivering value. Our system architecture integrates reinforcement learning with LLMs for both data creation and response selection. This method offers a practical way to build personalized AI companions that evolve through continuous learning, advancing beyond traditional static LLM techniques. 

**Abstract (ZH)**: 创建个性化和适应性强的对话型AI仍然是一个关键挑战。本文介绍了一种连续学习对话型AI（Continuous Learning Conversational AI, CLCA）方法，该方法利用A2C强化学习超越了静态大语言模型（Large Language Models, LLMs）。我们通过LLMs生成的模拟销售对话来训练一个A2C代理。该代理学习优化对话策略以实现个性化，重点关注互动和传递价值。我们的系统架构将强化学习与LLMs结合，用于数据生成和响应选择。该方法为通过持续学习构建个性化AI伴侣提供了一种实用途径，超越了传统的静态LLM技术。 

---
# CityEQA: A Hierarchical LLM Agent on Embodied Question Answering Benchmark in City Space 

**Title (ZH)**: CityEQA：城市空间中层级化语言模型代理的实体问答基准Benchmark 

**Authors**: Yong Zhao, Kai Xu, Zhengqiu Zhu, Yue Hu, Zhiheng Zheng, Yingfeng Chen, Yatai Ji, Chen Gao, Yong Li, Jincai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12532)  

**Abstract**: Embodied Question Answering (EQA) has primarily focused on indoor environments, leaving the complexities of urban settings - spanning environment, action, and perception - largely unexplored. To bridge this gap, we introduce CityEQA, a new task where an embodied agent answers open-vocabulary questions through active exploration in dynamic city spaces. To support this task, we present CityEQA-EC, the first benchmark dataset featuring 1,412 human-annotated tasks across six categories, grounded in a realistic 3D urban simulator. Moreover, we propose Planner-Manager-Actor (PMA), a novel agent tailored for CityEQA. PMA enables long-horizon planning and hierarchical task execution: the Planner breaks down the question answering into sub-tasks, the Manager maintains an object-centric cognitive map for spatial reasoning during the process control, and the specialized Actors handle navigation, exploration, and collection sub-tasks. Experiments demonstrate that PMA achieves 60.7% of human-level answering accuracy, significantly outperforming frontier-based baselines. While promising, the performance gap compared to humans highlights the need for enhanced visual reasoning in CityEQA. This work paves the way for future advancements in urban spatial intelligence. Dataset and code are available at this https URL. 

**Abstract (ZH)**: 基于体态的问答（EQA）主要集中在室内环境，而城市的复杂环境（包括环境、行动和感知）则被很大程度上忽略了。为了弥合这一差距，我们介绍了CityEQA，这是一个新任务，其中包括一个体态代理通过在动态城市环境中的主动探索来回答开放词汇的问题。为了支持这一任务，我们提出了CityEQA-EC，这是第一个基准数据集，包含了六类中的1,412个人工标注的任务，这些任务基于一个现实的3D城市仿真器。此外，我们提出了Planner-Manager-Actor（PMA），这是一种专门为CityEQA设计的新代理。PMA 能够实现长期规划和分层任务执行：规划器将问答问题分解为子任务，经理在过程中通过保持以对象为中心的认知地图来进行空间推理，专门的执行者则处理导航、探索和收集子任务。实验结果表明，PMA 实现了60.7%的人类级回答准确率，显著优于基于路径的基线模型。虽然具有前景，但与人类表现之间的差距凸显了CityEQA 中增强视觉推理的需求。这项工作为未来在城市空间智能方面的进步铺平了道路。数据集和代码可在此处访问：[https://...] 

---
# Boost, Disentangle, and Customize: A Robust System2-to-System1 Pipeline for Code Generation 

**Title (ZH)**: 增强、分离和个性化：一种稳健的系统2到系统1代码生成流水线 

**Authors**: Kounianhua Du, Hanjing Wang, Jianxing Liu, Jizheng Chen, Xinyi Dai, Yasheng Wang, Ruiming Tang, Yong Yu, Jun Wang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12492)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in various domains, particularly in system 1 tasks, yet the intricacies of their problem-solving mechanisms in system 2 tasks are not sufficiently explored. Recent research on System2-to-System1 methods surge, exploring the System 2 reasoning knowledge via inference-time computation and compressing the explored knowledge into System 1 process. In this paper, we focus on code generation, which is a representative System 2 task, and identify two primary challenges: (1) the complex hidden reasoning processes and (2) the heterogeneous data distributions that complicate the exploration and training of robust LLM solvers. To tackle these issues, we propose a novel BDC framework that explores insightful System 2 knowledge of LLMs using a MC-Tree-Of-Agents algorithm with mutual \textbf{B}oosting, \textbf{D}isentangles the heterogeneous training data for composable LoRA-experts, and obtain \textbf{C}ustomized problem solver for each data instance with an input-aware hypernetwork to weight over the LoRA-experts, offering effectiveness, flexibility, and robustness. This framework leverages multiple LLMs through mutual verification and boosting, integrated into a Monte-Carlo Tree Search process enhanced by reflection-based pruning and refinement. Additionally, we introduce the DisenLora algorithm, which clusters heterogeneous data to fine-tune LLMs into composable Lora experts, enabling the adaptive generation of customized problem solvers through an input-aware hypernetwork. This work lays the groundwork for advancing LLM capabilities in complex reasoning tasks, offering a novel System2-to-System1 solution. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在多个领域展现出了出色的能力，特别是在系统1任务方面，而其在系统2任务中的问题解决机制则尚未得到充分探索。近年来，关于系统2到系统1方法的研究激增，通过推理时的计算来探索系统2的推理知识，并将这些知识压缩到系统1的过程中。本文关注代码生成这一典型的系统2任务，并识别出两个主要挑战：（1）复杂的隐藏推理过程和（2）异质数据分布，这些都使得探索和训练鲁棒的LLM求解器变得更加困难。为了解决这些问题，我们提出了一种新颖的BDC框架，该框架利用一种具有增强学习的多智能体MC-树算法探索LLM的洞察力系统2知识，通过彼此拆解异质训练数据形成可组合的LoRA专家，为每个数据实例定制问题求解器，并通过具有输入感知的超网络加权选择LoRA专家，从而实现高效、灵活和鲁棒性。该框架通过相互验证和提升的多个LLM进行集成，并在基于反射的剪枝和改进的蒙特卡洛树搜索过程中得到增强。此外，我们还引入了DisenLora算法，该算法将异质数据聚类并微调LLM形成可组合的LoRA专家，通过具有输入感知的超网络生成适应性的定制问题求解器。这项工作为推进LLM在复杂推理任务中的能力奠定了基础，并提供了一种新颖的系统2到系统1解决方案。 

---
# Investigating and Extending Homans' Social Exchange Theory with Large Language Model based Agents 

**Title (ZH)**: 基于大型语言模型代理拓展霍曼斯的社会交换理论研究 

**Authors**: Lei Wang, Zheqing Zhang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12450)  

**Abstract**: Homans' Social Exchange Theory (SET) is widely recognized as a basic framework for understanding the formation and emergence of human civilizations and social structures. In social science, this theory is typically studied based on simple simulation experiments or real-world human studies, both of which either lack realism or are too expensive to control. In artificial intelligence, recent advances in large language models (LLMs) have shown promising capabilities in simulating human behaviors. Inspired by these insights, we adopt an interdisciplinary research perspective and propose using LLM-based agents to study Homans' SET. Specifically, we construct a virtual society composed of three LLM agents and have them engage in a social exchange game to observe their behaviors. Through extensive experiments, we found that Homans' SET is well validated in our agent society, demonstrating the consistency between the agent and human behaviors. Building on this foundation, we intentionally alter the settings of the agent society to extend the traditional Homans' SET, making it more comprehensive and detailed. To the best of our knowledge, this paper marks the first step in studying Homans' SET with LLM-based agents. More importantly, it introduces a novel and feasible research paradigm that bridges the fields of social science and computer science through LLM-based agents. Code is available at this https URL. 

**Abstract (ZH)**: 霍曼斯的社会交换理论（SET）普遍被视为理解人类文明和社会结构形成与演变的基本框架。在社会科学中，该理论通常基于简单的模拟实验或实际的人类研究进行研究，这两种方法要么缺乏真实性，要么太过昂贵难以控制。在人工智能领域，最近大型语言模型（LLMs）的进步展示了模拟人类行为的强大能力。受这些洞见的启发，我们采用了跨学科的研究视角，提出使用基于LLM的代理来研究霍曼斯的SET。具体来说，我们构建了一个由三个LLM代理组成的虚拟社会，并使它们参与社会交换游戏，以观察其行为。通过大量的实验，我们发现霍曼斯的SET在我们的代理社会中得到了很好的验证，显示了代理行为与人类行为的一致性。在此基础上，我们故意改变代理社会的设置，扩展了传统的霍曼斯SET，使其更加全面和详细。据我们所知，本文标志着首次使用基于LLM的代理研究霍曼斯的SET。更重要的是，它引入了一种通过基于LLM的代理连接社会科学和计算机科学的新型和可行的研究范式。相关代码详见：this https URL。 

---
# Pre-training Auto-regressive Robotic Models with 4D Representations 

**Title (ZH)**: 使用四维表示预训练自回归机器人模型 

**Authors**: Dantong Niu, Yuvan Sharma, Haoru Xue, Giscard Biamby, Junyi Zhang, Ziteng Ji, Trevor Darrell, Roei Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2502.13142)  

**Abstract**: Foundation models pre-trained on massive unlabeled datasets have revolutionized natural language and computer vision, exhibiting remarkable generalization capabilities, thus highlighting the importance of pre-training. Yet, efforts in robotics have struggled to achieve similar success, limited by either the need for costly robotic annotations or the lack of representations that effectively model the physical world. In this paper, we introduce ARM4R, an Auto-regressive Robotic Model that leverages low-level 4D Representations learned from human video data to yield a better pre-trained robotic model. Specifically, we focus on utilizing 3D point tracking representations from videos derived by lifting 2D representations into 3D space via monocular depth estimation across time. These 4D representations maintain a shared geometric structure between the points and robot state representations up to a linear transformation, enabling efficient transfer learning from human video data to low-level robotic control. Our experiments show that ARM4R can transfer efficiently from human video data to robotics and consistently improves performance on tasks across various robot environments and configurations. 

**Abstract (ZH)**: 基于大规模未标注数据集预训练的基座模型已经颠覆了自然语言处理和计算机视觉领域，展示了卓越的泛化能力，从而突显了预训练的重要性。然而，在机器人领域，取得类似成功的努力受到了昂贵的机器人标注成本或无法有效建模物理世界的代表性表示的限制。在这项工作中，我们引入了ARM4R，一种自回归机器人模型，该模型利用从人类视频数据中学习的低级四维表示，从而产生一种更优秀的预训练机器人模型。具体而言，我们重点关注通过单目深度估计将2D表示提升至3D空间中的3D点跟踪表示。这些四维表示在点与机器人状态表示之间保持共享的几何结构，直到线性变换，这使得能够从人类视频数据高效地转移到低级机器人控制。我们的实验结果表明，ARM4R可以从人类视频数据高效地转移到机器人领域，并在各种机器人环境和配置中一致地提高任务性能。 

---
# Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions 

**Title (ZH)**: 失眠之夜，甜食之日：用于现实 Coaching 代理互动的具有健康状况的合成用户生成 

**Authors**: Taedong Yun, Eric Yang, Mustafa Safdari, Jong Ha Lee, Vaishnavi Vinod Kumar, S. Sara Mahdavi, Jonathan Amar, Derek Peyton, Reut Aharony, Andreas Michaelides, Logan Schneider, Isaac Galatzer-Levy, Yugang Jia, John Canny, Arthur Gretton, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2502.13135)  

**Abstract**: We present an end-to-end framework for generating synthetic users for evaluating interactive agents designed to encourage positive behavior changes, such as in health and lifestyle coaching. The synthetic users are grounded in health and lifestyle conditions, specifically sleep and diabetes management in this study, to ensure realistic interactions with the health coaching agent. Synthetic users are created in two stages: first, structured data are generated grounded in real-world health and lifestyle factors in addition to basic demographics and behavioral attributes; second, full profiles of the synthetic users are developed conditioned on the structured data. Interactions between synthetic users and the coaching agent are simulated using generative agent-based models such as Concordia, or directly by prompting a language model. Using two independently-developed agents for sleep and diabetes coaching as case studies, the validity of this framework is demonstrated by analyzing the coaching agent's understanding of the synthetic users' needs and challenges. Finally, through multiple blinded evaluations of user-coach interactions by human experts, we demonstrate that our synthetic users with health and behavioral attributes more accurately portray real human users with the same attributes, compared to generic synthetic users not grounded in such attributes. The proposed framework lays the foundation for efficient development of conversational agents through extensive, realistic, and grounded simulated interactions. 

**Abstract (ZH)**: 我们提出了一套端到端的框架，用于生成合成用户以评估旨在促进积极行为改变的互动代理，例如健康和生活方式指导。合成用户基于健康和生活方式条件，本研究中特别针对睡眠管理和糖尿病管理，从而确保与健康指导代理进行真实的互动。合成用户的生成分为两个阶段：首先，基于现实世界健康和生活方式因素（包括基本的人口统计学和行为特征）生成结构化数据；其次，根据结构化数据开发合成用户的完整档案。合成用户与指导代理之间的互动通过生成代理基模（如Concordia）或通过提示语言模型直接模拟。使用两个独立开发的睡眠和糖尿病指导代理作为案例研究，通过分析指导代理对合成用户需求和挑战的理解来验证该框架的有效性。最后，通过人类专家的多次盲评，证明了具有健康和行为特征的合成用户更准确地反映了具有相同特征的真实人类用户，而不仅仅是通用的、未基于这些特征的合成用户。所提出的框架为通过广泛的、真实且基于现实的模拟互动高效开发对话代理奠定了基础。 

---
# Magma: A Foundation Model for Multimodal AI Agents 

**Title (ZH)**: Magma：多模态AI代理的基石模型 

**Authors**: Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Lars Liden, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13130)  

**Abstract**: We present Magma, a foundation model that serves multimodal AI agentic tasks in both the digital and physical worlds. Magma is a significant extension of vision-language (VL) models in that it not only retains the VL understanding ability (verbal intelligence) of the latter, but is also equipped with the ability to plan and act in the visual-spatial world (spatial-temporal intelligence) and complete agentic tasks ranging from UI navigation to robot manipulation. To endow the agentic capabilities, Magma is pretrained on large amounts of heterogeneous datasets spanning from images, videos to robotics data, where the actionable visual objects (e.g., clickable buttons in GUI) in images are labeled by Set-of-Mark (SoM) for action grounding, and the object movements (e.g., the trace of human hands or robotic arms) in videos are labeled by Trace-of-Mark (ToM) for action planning. Extensive experiments show that SoM and ToM reach great synergy and facilitate the acquisition of spatial-temporal intelligence for our Magma model, which is fundamental to a wide range of tasks as shown in Fig.1. In particular, Magma creates new state-of-the-art results on UI navigation and robotic manipulation tasks, outperforming previous models that are specifically tailored to these tasks. On image and video-related multimodal tasks, Magma also compares favorably to popular large multimodal models that are trained on much larger datasets. We make our model and code public for reproducibility at this https URL. 

**Abstract (ZH)**: 以下是将给定内容翻译成中文的结果，符合学术规范：

我们介绍了Magma，一种基础模型，用于处理数字和物理世界中的多模态人工智能代理任务。Magma 是对视觉语言（VL）模型的重要扩展，它不仅保留了后者在语言理解方面的能力（言语智能），而且还具备在视觉空间世界中规划和执行任务的能力（时空智能），能够完成从界面导航到机器人操作等多种代理任务。

为了赋予其代理能力，Magma 在跨图像、视频和机器人数据的大规模异构数据集上进行了预训练，其中图像中的可操作视觉对象（例如GUI中的可点击按钮）被标记为Set-of-Mark（SoM）以实现动作绑定，而视频中的对象运动（例如人的手或机器人臂的轨迹）被标记为Trace-of-Mark（ToM）以支持动作规划。大量的实验表明，SoM和ToM达到了很好的协同作用，并促进了我们的Magma模型获取时空智能，这对于广泛的任务至关重要，如图1所示。尤其值得注意的是，Magma 在界面导航和机器人操作任务上创造了新的最佳成果，超越了专门为此类任务设计的先前模型。在与图像和视频相关的多模态任务上，Magma 在使用更大数据集训练的流行大型多模态模型中也表现优越。我们在此处提供我们的模型和代码以确保可再现性：[填写链接处的URL]。

请注意，最后的网址需要替换为实际的公开链接地址。 

---
# LAMD: Context-driven Android Malware Detection and Classification with LLMs 

**Title (ZH)**: LAMD：基于上下文的Android恶意软件检测与分类方法（利用大型语言模型） 

**Authors**: Xingzhi Qian, Xinran Zheng, Yiling He, Shuo Yang, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2502.13055)  

**Abstract**: The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes. 

**Abstract (ZH)**: 移动应用的快速增长加剧了针对Android平台的恶意软件威胁。尽管存在众多检测方法，但它们经常难以应对不断演变的攻击、数据集偏见以及有限的解释性。大规模语言模型（LLMs）因其零-shot推理和推理论的能力提供了有希望的替代方案。然而，将LLMs应用于Android恶意软件检测存在两个关键挑战：（1）Android应用中的大量支持代码，往往包含数千个类，超出了LLMs的上下文限制，从而使恶意行为隐藏在良性功能之中；（2）Android应用的结构复杂性和相互依赖关系超过了LLMs基于序列的推理能力，导致代码分析碎片化，并妨碍恶意意图的推断。为了解决这些挑战，我们提出了一种实用的上下文驱动框架LAMD，以使基于LLM的Android恶意软件检测成为可能。LAMD 结合了关键上下文提取，以隔离安全关键代码区域并构建程序结构，然后采用逐级代码推理来逐步分析应用程序行为，从低级指令到高级语义，提供最终的预测和解释。还设计了一套有效的事实一致性验证机制，以减轻第一级LLM的混乱行为。在实际场景中的评估证明了LAMD 的有效性，为动态威胁环境中基于LLM的恶意软件分析奠定了可行的基础。 

---
# Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options 

**Title (ZH)**: 选项流：通过考虑选项进行多样性和改进的语言模型推理 

**Authors**: Lakshmi Nair, Ian Trase, Mark Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.12929)  

**Abstract**: We present a novel reasoning approach called Flow-of-Options (FoO), designed to address intrinsic biases in Large Language Models (LLMs). FoO enables LLMs to systematically explore a diverse range of possibilities in their reasoning, as demonstrated by an FoO-based agentic system for autonomously solving Machine Learning tasks (AutoML). Our framework outperforms state-of-the-art baselines, achieving improvements of 38.2% - 69.2% on standard data science tasks, and 37.4% - 47.9% on therapeutic chemistry tasks. With an overall operation cost under $1 per task, our framework is well-suited for cost-sensitive applications. Beyond classification and regression, we illustrate the broader applicability of our FoO-based agentic system to tasks such as reinforcement learning and image generation. Our framework presents significant advancements compared to current state-of-the-art agentic systems for AutoML, due to the benefits of FoO in enforcing diversity in LLM solutions through compressed, explainable representations that also support long-term memory when combined with case-based reasoning. 

**Abstract (ZH)**: 我们提出了一种新的推理方法，称为“选项流（Flow-of-Options, FoO）”，旨在解决大型语言模型（LLMs）固有的偏差问题。FoO 方法使 LLMs 能够系统地探讨其推理过程中多样化的可能选项，这一能力通过基于 FoO 的自主系统（例如自主解决机器学习任务的 AutoML 系统）得到了验证。我们的框架在标准数据科学任务中优于最先进的基线方法，取得了38.2%至69.2%的性能提升，在治疗化学任务中则取得了37.4%至47.9%的性能提升。由于总体操作成本低于每任务1美元，该框架非常适合成本敏感的应用场景。除了分类和回归任务之外，我们还展示了基于 FoO 的自主系统在强化学习和图像生成等更广泛任务中的适用性。由于 FoO 在压缩和可解释表征中支持多样化的 LLM 解决方案，并且在结合基于案例的推理时支持长期记忆，因此我们的框架相比当前最先进的 AutoML 自主系统展现出了显著的进步。 

---
# R2-KG: General-Purpose Dual-Agent Framework for Reliable Reasoning on Knowledge Graphs 

**Title (ZH)**: R2-KG：知识图上可靠推理的通用双代理框架 

**Authors**: Sumin Jo, Junseong Choi, Jiho Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12767)  

**Abstract**: Recent studies have combined Large Language Models (LLMs) with Knowledge Graphs (KGs) to enhance reasoning, improving inference accuracy without additional training while mitigating hallucination. However, existing frameworks are often rigid, struggling to adapt to KG or task changes. They also rely heavily on powerful LLMs for reliable (i.e., trustworthy) reasoning. To address this, We introduce R2-KG, a plug-and-play, dual-agent framework that separates reasoning into two roles: an Operator (a low-capacity LLM) that gathers evidence and a Supervisor (a high-capacity LLM) that makes final judgments. This design is cost-efficient for LLM inference while still maintaining strong reasoning accuracy. Additionally, R2-KG employs an Abstention mechanism, generating answers only when sufficient evidence is collected from KG, which significantly enhances reliability. Experiments across multiple KG-based reasoning tasks show that R2-KG consistently outperforms baselines in both accuracy and reliability, regardless of the inherent capability of LLMs used as the Operator. Further experiments reveal that the single-agent version of R2-KG, equipped with a strict self-consistency strategy, achieves significantly higher-than-baseline reliability while reducing inference cost. However, it also leads to a higher abstention rate in complex KGs. Our findings establish R2-KG as a flexible and cost-effective solution for KG-based reasoning. It reduces reliance on high-capacity LLMs while ensuring trustworthy inference. 

**Abstract (ZH)**: 近年来，研究者们将大型语言模型（LLMs）与知识图谱（KGs）相结合，以增强推理能力，同时在不进行额外训练的情况下提高推理准确性，减轻幻觉问题。然而，现有的框架往往较为僵化，难以适应知识图谱或任务的变化。它们还高度依赖强大的LLM来进行可靠的（即可信的）推理。为解决这些问题，我们提出了R2-KG，这是一种即插即用的双智能体框架，将推理分为两种角色：操作员（一种低容量的LLM），负责收集证据；监督者（一种高容量的LLM），负责做出最终判决。这种设计在保持强大推理准确性的同时，还最大限度地降低了LLM推理的成本。此外，R2-KG 还采用了一种回避机制（Abstention mechanism），仅在从知识图谱中收集到足够的证据后才生成答案，这极大地提升了推理的可靠性。在多个基于知识图谱的推理任务中的实验显示，无论作为操作员使用的LLM本身的能力如何，R2-KG 在准确性和可靠性方面都始终优于基线方法。进一步的实验表明，单智能体版本的R2-KG，结合严格的自我一致性策略，能够在降低推理成本的同时显著提高可靠性，但在复杂的知识图谱中会导致更高的回避率。我们的研究结果确立了R2-KG 作为一种灵活且成本效益高的知识图谱推理解决方案的地位。它减少了对高容量LLM的依赖，并确保了可信的推理。 

---
# MediaMind: Revolutionizing Media Monitoring using Agentification 

**Title (ZH)**: MediaMind：通过代理化革新媒体监控 

**Authors**: Ahmet Gunduz, Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12745)  

**Abstract**: In an era of rapid technological advancements, agentification of software tools has emerged as a critical innovation, enabling systems to function autonomously and adaptively. This paper introduces MediaMind as a case study to demonstrate the agentification process, highlighting how existing software can be transformed into intelligent agents capable of independent decision-making and dynamic interaction. Developed by aiXplain, MediaMind leverages agent-based architecture to autonomously monitor, analyze, and provide insights from multilingual media content in real time. The focus of this paper is on the technical methodologies and design principles behind agentifying MediaMind, showcasing how agentification enhances adaptability, efficiency, and responsiveness. Through detailed case studies and practical examples, we illustrate how the agentification of MediaMind empowers organizations to streamline workflows, optimize decision-making, and respond to evolving trends. This work underscores the broader potential of agentification to revolutionize software tools across various domains. 

**Abstract (ZH)**: 在快速的技术进步时代，软件工具的代理化已成为一种关键创新，使系统能够实现自主运行和适应性。本文以MediaMind为案例研究，展示了代理化过程，突出说明了如何通过将现有软件转化为能够独立做出决策并进行动态交互的智能代理来实现这一转变。MediaMind由aiXplain开发，利用基于代理的架构实时监控、分析多语种媒体内容，并提供洞察。本文的重点在于MediaMind代理化背后的技术和设计原则，展示了代理化如何增强系统的适应性、效率和响应性。通过详细的案例研究和实际例子，本文阐述了MediaMind的代理化是如何赋能组织简化工作流程、优化决策和应对不断变化的趋势。本文强调了代理化在各个领域革新软件工具的更广泛潜力。 

---
# \textit{One Size doesn't Fit All}: A Personalized Conversational Tutoring Agent for Mathematics Instruction 

**Title (ZH)**: 《一招鲜不适用所有情况》：一种个性化对话式辅导代理用于数学教学 

**Authors**: Ben Liu, Jihan Zhang, Fangquan Lin, Xu Jia, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12633)  

**Abstract**: Large language models (LLMs) have been increasingly employed in various intelligent educational systems, simulating human tutors to facilitate effective human-machine interaction. However, previous studies often overlook the significance of recognizing and adapting to individual learner characteristics. Such adaptation is crucial for enhancing student engagement and learning efficiency, particularly in mathematics instruction, where diverse learning styles require personalized strategies to promote comprehension and enthusiasm. In this paper, we propose a \textbf{P}erson\textbf{A}lized \textbf{C}onversational tutoring ag\textbf{E}nt (PACE) for mathematics instruction. PACE simulates students' learning styles based on the Felder and Silverman learning style model, aligning with each student's persona. In this way, our PACE can effectively assess the personality of students, allowing to develop individualized teaching strategies that resonate with their unique learning styles. To further enhance students' comprehension, PACE employs the Socratic teaching method to provide instant feedback and encourage deep thinking. By constructing personalized teaching data and training models, PACE demonstrates the ability to identify and adapt to the unique needs of each student, significantly improving the overall learning experience and outcomes. Moreover, we establish multi-aspect evaluation criteria and conduct extensive analysis to assess the performance of personalized teaching. Experimental results demonstrate the superiority of our model in personalizing the educational experience and motivating students compared to existing methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种智能教育系统中得到了越来越多的应用，模拟人类导师以促进有效的人机交互。然而，以往的研究往往忽视了识别和适应个体学习者特征的重要性。这种适应性对于提高学生参与度和学习效率至关重要，特别是在数学教学中，不同的学习风格需要个性化策略来促进理解和激发兴趣。本文提出了一种针对数学教学的个性化对话式辅导代理（PACE）。PACE基于Felder-Silverman学习风格模型模拟学生的学习风格，并与每个学生的人格特点相匹配。通过这种方式，我们的PACE可以有效地评估学生的人格特质，从而开发出能够与他们独特学习风格相呼应的个性化教学策略。为了进一步增强学生对知识点的理解，PACE采用苏格拉底教学法提供即时反馈并鼓励深入思考。通过构建个性化的教学数据并训练模型，PACE展示了识别和适应每位学生独特需求的能力，显著提高了整体学习体验和成果。此外，我们建立了多方面的评估标准并进行了广泛分析，以评估个性化教学的效果。实验结果表明，与现有方法相比，我们的模型在个性化教育体验和激发学生兴趣方面具有明显优势。 

---
# Automating Prompt Leakage Attacks on Large Language Models Using Agentic Approach 

**Title (ZH)**: 使用代理方法自动化的大型语言模型提示泄露攻击 

**Authors**: Tvrtko Sternak, Davor Runje, Dorian Granoša, Chi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12630)  

**Abstract**: This paper presents a novel approach to evaluating the security of large language models (LLMs) against prompt leakage-the exposure of system-level prompts or proprietary configurations. We define prompt leakage as a critical threat to secure LLM deployment and introduce a framework for testing the robustness of LLMs using agentic teams. Leveraging AG2 (formerly AutoGen), we implement a multi-agent system where cooperative agents are tasked with probing and exploiting the target LLM to elicit its prompt.
Guided by traditional definitions of security in cryptography, we further define a prompt leakage-safe system as one in which an attacker cannot distinguish between two agents: one initialized with an original prompt and the other with a prompt stripped of all sensitive information. In a safe system, the agents' outputs will be indistinguishable to the attacker, ensuring that sensitive information remains secure. This cryptographically inspired framework provides a rigorous standard for evaluating and designing secure LLMs.
This work establishes a systematic methodology for adversarial testing of prompt leakage, bridging the gap between automated threat modeling and practical LLM security.
You can find the implementation of our prompt leakage probing on GitHub. 

**Abstract (ZH)**: 本文提出了一种评估大型语言模型（LLMs）对其提示泄露（即系统级提示或专有配置的暴露）安全性的新方法。我们定义提示泄露为确保LLM安全部署的关键威胁，并介绍了一种使用代理团队测试LLM稳健性的框架。利用AG2（原AutoGen），我们实现了一个多代理系统，其中合作代理的任务是探测和利用目标LLM，以揭示其提示。

基于密码学中传统的安全定义，我们进一步定义了一种提示泄露安全的系统，即在这样一个系统中，攻击者无法区分两种代理：一种是以原始提示初始化，另一种则是以去除了所有敏感信息的提示初始化。在一个安全的系统中，攻击者无法区分代理的输出，从而确保敏感信息的安全。这种以密码学为启发的框架为评估和设计安全的LLM提供了一个严格的准则。

本文建立了一种系统性的方法，用于对抗性测试提示泄露，填补了自动化威胁建模与实际LLM安全性之间的空白。

您可以在GitHub上找到我们提示泄露探测的实现。 

---
# DemonAgent: Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent 

**Title (ZH)**: DemonAgent：基于大语言模型的代理程序中动态加密多后门植入攻击 

**Authors**: Pengyu Zhu, Zhenhong Zhou, Yuanhe Zhang, Shilinlu Yan, Kun Wang, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.12575)  

**Abstract**: As LLM-based agents become increasingly prevalent, backdoors can be implanted into agents through user queries or environment feedback, raising critical concerns regarding safety vulnerabilities. However, backdoor attacks are typically detectable by safety audits that analyze the reasoning process of agents. To this end, we propose a novel backdoor implantation strategy called \textbf{Dynamically Encrypted Multi-Backdoor Implantation Attack}. Specifically, we introduce dynamic encryption, which maps the backdoor into benign content, effectively circumventing safety audits. To enhance stealthiness, we further decompose the backdoor into multiple sub-backdoor fragments. Based on these advancements, backdoors are allowed to bypass safety audits significantly. Additionally, we present AgentBackdoorEval, a dataset designed for the comprehensive evaluation of agent backdoor attacks. Experimental results across multiple datasets demonstrate that our method achieves an attack success rate nearing 100\% while maintaining a detection rate of 0\%, illustrating its effectiveness in evading safety audits. Our findings highlight the limitations of existing safety mechanisms in detecting advanced attacks, underscoring the urgent need for more robust defenses against backdoor threats. Code and data are available at this https URL. 

**Abstract (ZH)**: 随着基于大规模语言模型（LLM）的代理变得越来越普遍，恶意后门可以通过用户查询或环境反馈植入到代理中，这引发了严重的安全性漏洞问题。然而，后门攻击通常可以通过安全性审查来检测，这些审查分析代理的推理过程。为了解决这一问题，我们提出了一种新的后门植入策略，称为**动态加密多后门植入攻击**。具体来说，我们引入了动态加密，将后门映射到无害内容中，从而有效规避了安全性审查。为了进一步提高隐蔽性，我们还将后门分解为多个子后门片段。通过这些进展，后门能够显著避开安全性审查。此外，我们还提出了一个名为**AgentBackdoorEval**的数据集，用于全面评估代理后门攻击。跨多个数据集的实验结果表明，我们的方法在攻击成功率接近100%的同时，保持了0%的检测率，这表明其在规避安全性审查方面的有效性。我们的研究结果揭示了现有安全性机制在检测高级攻击方面的局限性，突显了对更 robust 防御措施的迫切需求。相关代码和数据可通过以下链接获取：[请插入具体链接]。 

---
# A Cognitive Writing Perspective for Constrained Long-Form Text Generation 

**Title (ZH)**: 受限长文本生成的认知写作视角 

**Authors**: Kaiyang Wan, Honglin Mu, Rui Hao, Haoran Luo, Tianle Gu, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12568)  

**Abstract**: Like humans, Large Language Models (LLMs) struggle to generate high-quality long-form text that adheres to strict requirements in a single pass. This challenge is unsurprising, as successful human writing, according to the Cognitive Writing Theory, is a complex cognitive process involving iterative planning, translating, reviewing, and monitoring. Motivated by these cognitive principles, we aim to equip LLMs with human-like cognitive writing capabilities through CogWriter, a novel training-free framework that transforms LLM constrained long-form text generation into a systematic cognitive writing paradigm. Our framework consists of two key modules: (1) a Planning Agent that performs hierarchical planning to decompose the task, and (2) multiple Generation Agents that execute these plans in parallel. The system maintains quality via continuous monitoring and reviewing mechanisms, which evaluate outputs against specified requirements and trigger necessary revisions. CogWriter demonstrates exceptional performance on LongGenBench, a benchmark for complex constrained long-form text generation. Even when using Qwen-2.5-14B as its backbone, CogWriter surpasses GPT-4o by 22% in complex instruction completion accuracy while reliably generating texts exceeding 10,000 words. We hope this cognitive science-inspired approach provides a paradigm for LLM writing advancements: \href{this https URL}{CogWriter}. 

**Abstract (ZH)**: 像人类一样，大规模语言模型（LLMs）在一次生成高质量长文本时往往会遇到困难，尤其是在遵循严格要求方面。这一挑战并不令人惊讶，因为在认知写作理论中，成功的写作过程被认为是一个复杂的认知过程，涉及迭代的计划、翻译、审阅和监控。基于这些认知原理，我们旨在通过CogWriter这一全新的无需训练框架，为LLMs配备类似人类的认知写作能力，将受限的长文本生成转变为一种系统化的认知写作范式。我们的框架包括两个关键模块：（1）规划代理，执行分层规划以分解任务；（2）多个生成代理，在并行执行这些计划。系统通过持续的监控和审阅机制维持质量，这些机制将输出与指定要求进行对比，并触发必要的修订。CogWriter在LongGenBench上表现出色，LongGenBench是一个复杂受限长文本生成基准测试。即使使用Qwen-2.5-14B作为其基础模型，CogWriter在复杂指令完成准确性上仍比GPT-4o高出22%，并可靠地生成超过10,000字的文本。我们希望这种受认知科学启发的方法为LLMs写作进步提供一个范式：\href{this https URL}{CogWriter}。 

---
# EDGE: Efficient Data Selection for LLM Agents via Guideline Effectiveness 

**Title (ZH)**: EDGE：通过指南有效性实现高效数据选择的LLM代理方法 

**Authors**: Yunxiao Zhang, Guanming Xiong, Haochen Li, Wen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12494)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities as AI agents. However, existing methods for enhancing LLM-agent abilities often lack a focus on data quality, leading to inefficiencies and suboptimal results in both fine-tuning and prompt engineering. To address this issue, we introduce EDGE, a novel approach for identifying informative samples without needing golden answers. We propose the Guideline Effectiveness (GE) metric, which selects challenging samples by measuring the impact of human-provided guidelines in multi-turn interaction tasks. A low GE score indicates that the human expertise required for a sample is missing from the guideline, making the sample more informative. By selecting samples with low GE scores, we can improve the efficiency and outcomes of both prompt engineering and fine-tuning processes for LLMs. Extensive experiments validate the performance of our method. Our method achieves competitive results on the HotpotQA and WebShop and datasets, requiring 75\% and 50\% less data, respectively, while outperforming existing methods. We also provide a fresh perspective on the data quality of LLM-agent fine-tuning. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了作为AI代理的出色能力。然而，现有的增强LLM代理能力的方法往往缺乏对数据质量的关注，这导致了在微调和提示工程中效率低下和结果不佳的问题。为了解决这一问题，我们提出了EDGE，一种无需金标准答案即可识别有效样本的新方法。我们提出了指导有效性（GE）指标，该指标通过测量人类提供的指南在多轮交互任务中的影响来选择具有挑战性的样本。GE分数较低表明样本所需的人类专业知识未包含在指南中，使样本更具信息量。通过选择GE分数较低的样本，我们可以提高LLM提示工程和微调过程的效率和结果。广泛实验证明了我们方法的有效性。我们的方法在HotpotQA和WebShop数据集上取得了具有竞争力的结果，在分别减少了75%和50%数据量的情况下超越了现有方法。我们还从新的角度审视了LLM代理微调中的数据质量。 

---
# LM Agents for Coordinating Multi-User Information Gathering 

**Title (ZH)**: 用于协调多用户信息收集的LM代理模型 

**Authors**: Harsh Jhamtani, Jacob Andreas, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2502.12328)  

**Abstract**: This paper introduces PeopleJoin, a benchmark for evaluating LM-mediated collaborative problem solving. Given a user request, PeopleJoin agents must identify teammates who might be able to assist, converse with these teammates to gather information, and finally compile a useful answer or summary for the original user. PeopleJoin comprises two evaluation domains: PeopleJoin-QA, focused on questions about tabular data, and PeopleJoin-DocCreation, focused on document creation tasks. The two domains are adapted from existing NLP benchmarks for database question answering and multi-document summarization; here, however, the information needed to complete these tasks is distributed across synthetic ``organizations'' of 2--20 users, simulating natural multi-user collaboration scenarios. We implemented several popular LM agent architectures, evaluating their accuracy and efficiency at completing tasks, and highlight new research questions that can be studied using PeopleJoin. 

**Abstract (ZH)**: 本文介绍了PeopleJoin，这是一个用于评估语言模型（LM）介导的协同问题解决的基准。给定用户请求，PeopleJoin代理必须识别可能能够提供帮助的队友，与这些队友进行交流以收集信息，最后为原用户提供有用的答案或总结。PeopleJoin包含两个评估领域：PeopleJoin-QA（专注于表格数据的问题）和PeopleJoin-DocCreation（专注于文档创建任务）。这两个领域是从现有的数据库问答和多文档总结的NLP基准中改编而来的；然而，在这里，完成这些任务所需的信息分布在2至20名用户的合成“组织”中，模拟了自然的多用户协作场景。我们实现了几种流行的LM代理架构，评估了它们在完成任务方面的准确性和效率，并指出了可以通过PeopleJoin研究的新研究问题。 

---
# Connecting Large Language Model Agent to High Performance Computing Resource 

**Title (ZH)**: 将大型语言模型代理与高性能计算资源连接起来 

**Authors**: Heng Ma, Alexander Brace, Carlo Siebenschuh, Greg Pauloski, Ian Foster, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12280)  

**Abstract**: The Large Language Model agent workflow enables the LLM to invoke tool functions to increase the performance on specific scientific domain questions. To tackle large scale of scientific research, it requires access to computing resource and parallel computing setup. In this work, we implemented Parsl to the LangChain/LangGraph tool call setup, to bridge the gap between the LLM agent to the computing resource. Two tool call implementations were set up and tested on both local workstation and HPC environment on Polaris/ALCF. The first implementation with Parsl-enabled LangChain tool node queues the tool functions concurrently to the Parsl workers for parallel execution. The second configuration is implemented by converting the tool functions into Parsl ensemble functions, and is more suitable for large task on super computer environment. The LLM agent workflow was prompted to run molecular dynamics simulations, with different protein structure and simulation conditions. These results showed the LLM agent tools were managed and executed concurrently by Parsl on the available computing resource. 

**Abstract (ZH)**: 大规模语言模型代理的工作流能够促使LLM调用工具函数以在特定科学领域问题上提升性能。为了应对大规模科学研究的需求，需要访问计算资源并设置并行计算环境。在本工作中，我们将在LangChain/LangGraph工具调用设置中实现Parsl，以弥合LLM代理与计算资源之间的差距。我们在Polaris/ALCF的本地工作站和超级计算机环境中分别设置了两种工具调用实现并进行了测试。第一种实现利用Parsl启用的LangChain工具节点，将工具函数并发提交给Parsl工作者，以实现并行执行。第二种配置将工具函数转换为Parsl集合函数，更适合在超级计算机环境中处理大规模任务。LLM代理工作流被用来执行分子动力学模拟，其中包括不同的蛋白质结构和模拟条件。这些结果表明，Parsl能够有效管理和在可用计算资源上并行执行LLM代理工具。 

---
