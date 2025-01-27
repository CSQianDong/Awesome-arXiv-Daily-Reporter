# Recommending Actionable Strategies: A Semantic Approach to Integrating Analytical Frameworks with Decision Heuristics 

**Title (ZH)**: 推荐可操作策略：一种基于语义集成分析框架与决策启发式的手段 

**Authors**: Renato Ghisellini, Remo Pareschi, Marco Pedroni, Giovanni Battista Raggi  

**Link**: [PDF](https://arxiv.org/pdf/2501.14634)  

**Abstract**: We present a novel approach for recommending actionable strategies by integrating strategic frameworks with decision heuristics through semantic analysis. While strategy frameworks provide systematic models for assessment and planning, and decision heuristics encode experiential knowledge,these traditions have historically remained separate. Our methodology bridges this gap using advanced natural language processing (NLP), demonstrated through integrating frameworks like the 6C model with the Thirty-Six Stratagems. The approach employs vector space representations and semantic similarity calculations to map framework parameters to heuristic patterns, supported by a computational architecture that combines deep semantic processing with constrained use of Large Language Models. By processing both primary content and secondary elements (diagrams, matrices) as complementary linguistic representations, we demonstrate effectiveness through corporate strategy case studies. The methodology generalizes to various analytical frameworks and heuristic sets, culminating in a plug-and-play architecture for generating recommender systems that enable cohesive integration of strategic frameworks and decision heuristics into actionable guidance. 

**Abstract (ZH)**: 我们提出了一种创新的方法，通过将战略框架与决策启发式相结合，利用语义分析来推荐可操作的策略。尽管战略框架提供了系统化的评估和规划模型，而决策启发式则编码了一种经验性知识，但这些传统方法在过去一直保持分离状态。我们通过使用先进的自然语言处理（NLP）技术，将这两种方法相结合，通过整合如6C模型和三十六计等框架来证明这一点。该方法利用向量空间表示和语义相似性计算来将框架参数映射到启发式模式，其背后的计算架构结合了深入的语义处理和大型语言模型的受限使用。通过同时处理主要内容和次要元素（如图表、矩阵）作为互补的语言表示，我们通过企业战略案例研究展示了其有效性。上述方法可以应用于各种分析框架和启发式集，进而形成一种插件式架构，能够生成推荐系统，使战略框架和决策启发式能够统一整合为可操作的指导。 

---
# Extracting Problem Structure with LLMs for Optimized SAT Local Search 

**Title (ZH)**: 使用大型语言模型提取问题结构以优化SAT局部搜索 

**Authors**: André Schilder, Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2501.14630)  

**Abstract**: Local search preprocessing makes Conflict-Driven Clause Learning (CDCL) solvers faster by providing high-quality starting points and modern SAT solvers have incorporated this technique into their preprocessing steps. However, these tools rely on basic strategies that miss the structural patterns in problems. We present a method that applies Large Language Models (LLMs) to analyze Python-based encoding code. This reveals hidden structural patterns in how problems convert into SAT. Our method automatically generates specialized local search algorithms that find these patterns and use them to create strong initial assignments. This works for any problem instance from the same encoding type. Our tests show encouraging results, achieving faster solving times compared to baseline preprocessing systems. 

**Abstract (ZH)**: 局部搜索预处理通过提供高质量的起始点使冲突驱动子句学习（CDCL）求解器更加快速。现代SAT求解器已经将这种技术纳入其预处理步骤中。然而，这些工具依靠基本策略，无法捕捉问题中的结构模式。我们提出了一种方法，利用大型语言模型（LLMs）分析基于Python的编码代码，从而揭示问题转换为SAT中的隐藏结构模式。该方法自动生成专门化的局部搜索算法，用于发现这些模式并利用它们创建强初始赋值。这种方法适用于同一编码类型下的任何问题实例。我们的测试显示了令人鼓舞的结果，相较于基线预处理系统，能够实现更快的求解时间。 

---
# Hybrid Quantum-Classical Multi-Agent Pathfinding 

**Title (ZH)**: 混合量子-经典多智能体路径规划 

**Authors**: Thore Gerlach, Loong Kuan Lee, Frédéric Barbaresco, Nico Piatkowski  

**Link**: [PDF](https://arxiv.org/pdf/2501.14568)  

**Abstract**: Multi-Agent Path Finding (MAPF) focuses on determining conflict-free paths for multiple agents navigating through a shared space to reach specified goal locations. This problem becomes computationally challenging, particularly when handling large numbers of agents, as frequently encountered in practical applications like coordinating autonomous vehicles. Quantum computing (QC) is a promising candidate in overcoming such limits. However, current quantum hardware is still in its infancy and thus limited in terms of computing power and error robustness. In this work, we present the first optimal hybrid quantum-classical MAPF algorithm which is based on branch-and-cut-and-prize. QC is integrated by iteratively solving QUBO problems, based on conflict graphs. Experiments on actual quantum hardware and results on benchmark data suggest that our approach dominates previous QUBO formulations and baseline MAPF solvers. 

**Abstract (ZH)**: 多Agent路径规划（Multi-Agent Path Finding，MAPF）旨在为多个代理在共享空间中导航到指定的目标位置时确定无冲突的路径。当处理大量代理时，这一问题变得计算上更具挑战性，尤其是在实际应用中协调自动驾驶车辆时更为常见。量子计算（Quantum Computing，QC）是一种克服此类限制的有前途的候选技术。然而，当前的量子硬件仍处于初级阶段，因此在计算能力和错误鲁棒性方面受限。在本工作中，我们提出了第一个基于分支、切割和奖励的最优混合量子-经典MAPF算法。我们通过迭代解决基于冲突图的QUBO问题将QC集成进来。在实际量子硬件上的实验和基准数据上的结果表明，我们的方法优于以往的QUBO公式化方法和基准MAPF求解器。 

---
# VERUS-LM: a Versatile Framework for Combining LLMs with Symbolic Reasoning 

**Title (ZH)**: VERUS-LM：一种将大规模语言模型与符号推理相结合的通用框架 

**Authors**: Benjamin Callewaert, Simon Vandevelde, Joost Vennekens  

**Link**: [PDF](https://arxiv.org/pdf/2501.14540)  

**Abstract**: A recent approach to neurosymbolic reasoning is to explicitly combine the strengths of large language models (LLMs) and symbolic solvers to tackle complex reasoning tasks. However, current approaches face significant limitations, including poor generalizability due to task-specific prompts, inefficiencies caused by the lack of separation between knowledge and queries, and restricted inferential capabilities. These shortcomings hinder their scalability and applicability across diverse domains. In this paper, we introduce VERUS-LM, a novel framework designed to address these challenges. VERUS-LM employs a generic prompting mechanism, clearly separates domain knowledge from queries, and supports a wide range of different logical reasoning tasks. This framework enhances adaptability, reduces computational cost, and allows for richer forms of reasoning, such as optimization and constraint satisfaction. We show that our approach succeeds in diverse reasoning on a novel dataset, markedly outperforming LLMs. Additionally, our system achieves competitive results on common reasoning benchmarks when compared to other state-of-the-art approaches, and significantly surpasses them on the difficult AR-LSAT dataset. By pushing the boundaries of hybrid reasoning, VERUS-LM represents a significant step towards more versatile neurosymbolic AI systems 

**Abstract (ZH)**: 近年来，神经符号推理的一种方法是明确结合大规模语言模型（LLMs）和符号求解器的优势，以应对复杂的推理任务。然而，当前的方法面临着显著的局限性，包括由于任务特定的提示导致的泛化能力差、由于知识与查询之间缺乏分离导致的效率低下，以及推理能力的限制。这些不足限制了它们在不同领域的可扩展性和适用性。在本文中，我们介绍了一种名为VERUS-LM的新型框架，旨在解决这些挑战。VERUS-LM采用了通用的提示机制，明确地将领域知识与查询分离，并支持多种不同的逻辑推理任务。该框架增强了适应性，降低了计算成本，并允许进行更加丰富的推理形式，如优化和约束满足。我们的方法在新型数据集上的多样性推理中取得了成功，显著优于LLMs。此外，当与其他最先进的方法进行比较时，我们的系统在常用推理基准测试中取得了竞争力的结果，并在困难的AR-LSAT数据集中明显超越了它们。通过推动混合推理的界限，VERUS-LM代表了更通用的神经符号AI系统的重大进步。 

---
# In System Alignments we Trust! Explainable Alignments via Projections 

**Title (ZH)**: 当然，以下是翻译内容：

在系统对齐中，我们信赖之！通过投影实现可解释的对齐 

**Authors**: Dominique Sommers, Natalia Sidorova, Boudewijn van Dongen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14360)  

**Abstract**: Alignments are a well-known process mining technique for reconciling system logs and normative process models. Evidence of certain behaviors in a real system may only be present in one representation - either a log or a model - but not in the other. Since for processes in which multiple entities, like objects and resources, are involved in the activities, their interactions affect the behavior and are therefore essential to take into account in the alignments.
Additionally, both logged and modeled representations of reality may be imprecise and only partially represent some of these entities, but not all. In this paper, we introduce the concept of "relaxations" through projections for alignments to deal with partially correct models and logs. Relaxed alignments help to distinguish between trustworthy and untrustworthy content of the two representations (the log and the model) to achieve a better understanding of the underlying process and expose quality issues. 

**Abstract (ZH)**: 对齐是一种广为人知的日记过程挖掘技术，用于协调系统日志和规范的过程模型。某些行为在现实系统中的证据可能只存在于一种表示中——日志或模型——而在另一种表示中并不存在。由于涉及多个实体（如对象和资源）的过程活动中，这些实体之间的相互作用会影响行为，因此在对齐过程中必须将它们考虑进去。

此外，现实中的日记表示和模型表示可能不够精确，并且只部分表示这些实体，而不是全部。在这篇论文中，我们通过投影引入了“松弛”概念以应对部分正确的模型和日志。松弛对齐有助于区分两种表示（日志和模型）中的可信和不可信内容，从而更好地理解 underlying 过程，并揭示质量问题。 

---
# Exploring the sustainable scaling of AI dilemma: A projective study of corporations' AI environmental impacts 

**Title (ZH)**: 探索人工智能可持续扩展的困境：对 CORPORATIONS 的人工智能环境影响的一项前瞻研究

注：此处“CORPORATIONS”根据上下文可能需要具体化为特定公司或行业，以确保翻译的准确性和专业性。如果原文中“corporations”指的是普遍意义上的企业群体，那么上面的翻译是合适的。如果需要更具体的翻译，请提供更多的背景信息。 

**Authors**: Clément Desroches, Martin Chauvin, Louis Ladan, Caroline Vateau, Simon Gosset, Philippe Cordier  

**Link**: [PDF](https://arxiv.org/pdf/2501.14334)  

**Abstract**: The rapid growth of artificial intelligence (AI), particularly Large Language Models (LLMs), has raised concerns regarding its global environmental impact that extends beyond greenhouse gas emissions to include consideration of hardware fabrication and end-of-life processes. The opacity from major providers hinders companies' abilities to evaluate their AI-related environmental impacts and achieve net-zero this http URL this paper, we propose a methodology to estimate the environmental impact of a company's AI portfolio, providing actionable insights without necessitating extensive AI and Life-Cycle Assessment (LCA) expertise. Results confirm that large generative AI models consume up to 4600x more energy than traditional models. Our modelling approach, which accounts for increased AI usage, hardware computing efficiency, and changes in electricity mix in line with IPCC scenarios, forecasts AI electricity use up to 2030. Under a high adoption scenario, driven by widespread Generative AI and agents adoption associated to increasingly complex models and frameworks, AI electricity use is projected to rise by a factor of this http URL the environmental impact of Generative AI by 2030 requires coordinated efforts across the AI value chain. Isolated measures in hardware efficiency, model efficiency, or grid improvements alone are insufficient. We advocate for standardized environmental assessment frameworks, greater transparency from the all actors of the value chain and the introduction of a "Return on Environment" metric to align AI development with net-zero goals. 

**Abstract (ZH)**: 人工智能（AI）的快速发展，特别是大型语言模型（LLMs）的应用，已经引发了对其全球环境影响的担忧，这种影响不仅限于温室气体排放，还涉及硬件制造和产品生命周期结束阶段的处理。主要供应商的不透明性阻碍了企业在评估其AI相关的环境影响以及实现净零排放方面的能力。本文提出了一种方法来估算企业AI组合的环境影响，无需深入的AI和生命周期评估（LCA）专业知识即可提供实用的见解。结果显示，大型生成式AI模型的能量消耗是传统模型的4600倍。我们的建模方法考虑了增加的AI使用量、硬件计算效率以及与IPCC情景相符的电力结构变化，预测AI电力使用量将一直持续到2030年。在广泛采用生成式AI和与其相关日益复杂的模型和框架的情况下，高采用情景下AI电力使用量预计将增加至当前的X倍。到2030年生成式AI的环境影响需要整个AI价值链中的协调努力。单独提高硬件效率、模型效率或电网改进的措施是不够的。我们建议建立标准化的环境评估框架、增强价值链各个环节的透明度，并引入“环境投资回报率”指标，以使AI开发与净零目标相一致。 

---
# MASTER: A Multi-Agent System with LLM Specialized MCTS 

**Title (ZH)**: MASTER：一种专门化的LLM多代理系统与MCTS结合的架构 

**Authors**: Bingzheng Gan, Yufan Zhao, Tianyi Zhang, Jing Huang, Yusu Li, Shu Xian Teo, Changwang Zhang, Wei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2501.14304)  

**Abstract**: Large Language Models (LLM) are increasingly being explored for problem-solving tasks. However, their strategic planning capability is often viewed with skepticism. Recent studies have incorporated the Monte Carlo Tree Search (MCTS) algorithm to augment the planning capacity of LLM. Despite its potential, MCTS relies on extensive sampling simulations to approximate the true reward distribution, leading to two primary issues. Firstly, MCTS is effective for tasks like the Game of Go, where simulation results can yield objective rewards (e.g., 1 for a win and 0 for a loss). However, for tasks such as question answering, the result of a simulation is the answer to the question, which cannot obtain an objective reward without the ground truth. Secondly, obtaining statistically significant reward estimations typically requires a sample size exceeding 30 simulations, resulting in excessive token usage and time consumption. To address these challenges, we present Multi-Agent System with Tactical Execution and Reasoning using LLM Specialized MCTS (MASTER), a novel framework that coordinates agent recruitment and communication using LLM specialized MCTS. This system autonomously adjusts the number of agents based on task complexity and ensures focused communication among them. Comprehensive experiments across various tasks demonstrate the effectiveness of our proposed framework. It achieves 76% accuracy on HotpotQA and 80% on WebShop, setting new state-of-the-art performance on these datasets. 

**Abstract (ZH)**: 大型语言模型（LLM）越来越多地被应用于问题解决任务中。然而，人们对其战略规划能力持怀疑态度。近期的研究将蒙特卡洛树搜索（MCTS）算法引入LLM，以增强其规划能力。尽管MCTS具有潜力，但它依赖于广泛的采样模拟来近似真实的奖励分布，导致了两个主要问题。首先，MCTS在围棋等任务中表现良好，因为这些任务的模拟结果可以产生客观奖励（例如胜利计1分，失败计0分）。然而，对于诸如问答等任务，模拟的结果只是问题的答案，没有 ground truth 时很难获得客观奖励。其次，通常需要超过30次模拟来获得统计上显著的奖励估计，这会导致大量的 token 使用和时间消耗。为了解决这些问题，我们提出了一个名为 Multi-Agent System with Tactical Execution and Reasoning using LLM Specialized MCTS（MASTER）的新型框架，该框架利用LLM专项MCTS协调代理的招募和沟通。此系统根据任务复杂性自主调整代理数量，并确保它们之间的集中沟通。跨多种任务的综合实验表明，我们提出的框架具有有效性。它在HotpotQA上的准确率达到76%，在WebShop上的准确率达到80%，在这些数据集上设置了新的最先进的性能标准。 

---
# Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph 

**Title (ZH)**: 快速图思考：大型语言模型在知识图谱中的更宽、更深、更快推理 

**Authors**: Xujian Liang, Zhaoquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14300)  

**Abstract**: Graph Retrieval Augmented Generation (GRAG) is a novel paradigm that takes the naive RAG system a step further by integrating graph information, such as knowledge graph (KGs), into large-scale language models (LLMs) to mitigate hallucination. However, existing GRAG still encounter limitations: 1) simple paradigms usually fail with the complex problems due to the narrow and shallow correlations capture from KGs 2) methods of strong coupling with KGs tend to be high computation cost and time consuming if the graph is dense. In this paper, we propose the Fast Think-on-Graph (FastToG), an innovative paradigm for enabling LLMs to think ``community by community" within KGs. To do this, FastToG employs community detection for deeper correlation capture and two stages community pruning - coarse and fine pruning for faster retrieval. Furthermore, we also develop two Community-to-Text methods to convert the graph structure of communities into textual form for better understanding by LLMs. Experimental results demonstrate the effectiveness of FastToG, showcasing higher accuracy, faster reasoning, and better explainability compared to the previous works. 

**Abstract (ZH)**: Graph Retrieval Augmented Generation (GRAG) 是一种新颖的框架，它在基础的 RAG 系统基础上进一步整合了图信息（如知识图谱 KGs），以减轻幻觉现象。然而，现有的 GRAG 仍然存在一些局限性：1) 简单的框架在处理复杂问题时往往因 KG 中浅显和狭窄的相关性捕捉而失效；2) 与 KG 强耦合的方法如果图密集，则可能产生高计算成本和耗时的问题。本文中，我们提出了 Fast Think-on-Graph (FastToG)，这是一种创新的框架，旨在使大规模语言模型 (LLMs) 在知识图谱中以“社区为单位”进行思考。为了实现这一点，FastToG 应用了社区检测以捕捉更深层次的相关性，并通过粗略和精细的社区裁剪实现了更快的检索。此外，我们还开发了两种社区到文本的方法，将社区的图结构转换为文本形式，以更好地被语言模型理解。实验结果表明，FastToG 的有效性，其展示了比以往工作更高的准确性、更快的推理速度和更好的解释性。 

---
# Top Ten Challenges Towards Agentic Neural Graph Databases 

**Title (ZH)**: 向着能动神经图数据库发展的十大挑战 

**Authors**: Jiaxin Bai, Zihao Wang, Yukun Zhou, Hang Yin, Weizhi Fei, Qi Hu, Zheye Deng, Jiayang Cheng, Tianshi Zheng, Hong Ting Tsang, Yisen Gao, Zhongwei Xie, Yufei Li, Lixin Fan, Binhang Yuan, Wei Wang, Lei Chen, Xiaofang Zhou, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.14224)  

**Abstract**: Graph databases (GDBs) like Neo4j and TigerGraph excel at handling interconnected data but lack advanced inference capabilities. Neural Graph Databases (NGDBs) address this by integrating Graph Neural Networks (GNNs) for predictive analysis and reasoning over incomplete or noisy data. However, NGDBs rely on predefined queries and lack autonomy and adaptability. This paper introduces Agentic Neural Graph Databases (Agentic NGDBs), which extend NGDBs with three core functionalities: autonomous query construction, neural query execution, and continuous learning. We identify ten key challenges in realizing Agentic NGDBs: semantic unit representation, abductive reasoning, scalable query execution, and integration with foundation models like large language models (LLMs). By addressing these challenges, Agentic NGDBs can enable intelligent, self-improving systems for modern data-driven applications, paving the way for adaptable and autonomous data management solutions. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文内容：

图形数据库（GDBs）如Neo4j和TigerGraph在处理互联数据方面表现出色，但在推理能力方面尚存不足。神经图形数据库（NGDBs）通过集成图神经网络（GNNs）来弥补这一不足，用于预测分析和处理不完整或嘈杂的数据。然而，NGDBs依赖预定义查询，缺乏自主性和适应性。本文提出了一种具有自主功能的神经图形数据库（Agentic NGDBs），它在NGDBs的基础上增加了三个核心功能：自主查询构建、神经查询执行和持续学习。我们识别出实现Agentic NGDBs的十个关键挑战，包括语义单元表示、归纳推理、可扩展的查询执行以及与大规模语言模型（LLMs）等基础模型的集成。通过解决这些挑战，Agentic NGDBs可以为现代数据驱动应用提供智能、自我提升的系统，从而促进适应性强的自主数据管理解决方案的发展。 

---
# Distributed Multi-Agent Coordination Using Multi-Modal Foundation Models 

**Title (ZH)**: 使用多模态基础模型进行分布式多代理协调 

**Authors**: Saaduddin Mahmud, Dorian Benhamou Goldfajn, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2501.14189)  

**Abstract**: Distributed Constraint Optimization Problems (DCOPs) offer a powerful framework for multi-agent coordination but often rely on labor-intensive, manual problem construction. To address this, we introduce VL-DCOPs, a framework that takes advantage of large multimodal foundation models (LFMs) to automatically generate constraints from both visual and linguistic instructions. We then introduce a spectrum of agent archetypes for solving VL-DCOPs: from a neuro-symbolic agent that delegates some of the algorithmic decisions to an LFM, to a fully neural agent that depends entirely on an LFM for coordination. We evaluate these agent archetypes using state-of-the-art LLMs (large language models) and VLMs (vision language models) on three novel VL-DCOP tasks and compare their respective advantages and drawbacks. Lastly, we discuss how this work extends to broader frontier challenges in the DCOP literature. 

**Abstract (ZH)**: 分布式约束优化问题（DCOPs）提供了一种强大的多代理协调框架，但通常依赖于耗时的手工问题构建。为解决这一问题，我们引入了VL-DCOPs框架，利用大规模多模态基础模型（LFMs）从视觉和语言指令中自动生成约束。接着，我们提出了解决VL-DCOPs的一系列代理原型：从一种神经符号代理，它将部分算法决策委托给LFM，到一种完全依赖LFM进行协调的全神经代理。我们使用最先进的大型语言模型（LLMs）和视觉语言模型（VLMs）来评估这些代理原型，并比较各自的优缺点。最后，我们讨论了这项工作如何扩展到DCOP文献中的更广泛的前沿挑战。 

---
# Human-Alignment Influences the Utility of AI-assisted Decision Making 

**Title (ZH)**: 人类一致性影响AI辅助决策的价值 

**Authors**: Nina L. Corvelo Benz, Manuel Gomez Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2501.14035)  

**Abstract**: Whenever an AI model is used to predict a relevant (binary) outcome in AI-assisted decision making, it is widely agreed that, together with each prediction, the model should provide an AI confidence value. However, it has been unclear why decision makers have often difficulties to develop a good sense on when to trust a prediction using AI confidence values. Very recently, Corvelo Benz and Gomez Rodriguez have argued that, for rational decision makers, the utility of AI-assisted decision making is inherently bounded by the degree of alignment between the AI confidence values and the decision maker's confidence on their own predictions. In this work, we empirically investigate to what extent the degree of alignment actually influences the utility of AI-assisted decision making. To this end, we design and run a large-scale human subject study (n=703) where participants solve a simple decision making task - an online card game - assisted by an AI model with a steerable degree of alignment. Our results show a positive association between the degree of alignment and the utility of AI-assisted decision making. In addition, our results also show that post-processing the AI confidence values to achieve multicalibration with respect to the participants' confidence on their own predictions increases both the degree of alignment and the utility of AI-assisted decision making. 

**Abstract (ZH)**: 每当使用AI模型预测相关的（二元）结果时，尤其是在辅助决策过程中，通常认为，除了提供预测结果外，模型还应提供一个AI置信度值。然而，关于决策者在使用AI置信度值评估预测结果时的信任感为何常常难以建立，这一直存在争议。最近，Corvelo Benz和Gomez Rodriguez提出，对于理性的决策者而言，基于AI的决策辅助的效用固有地受限于AI置信度值与决策者自身预测置信度之间的对齐程度。在本研究中，我们通过实证研究考察这种对齐程度实际上如何影响基于AI的决策辅助的效用。为此，我们设计并实施了一个大规模的人类主体研究（n=703），参与者在辅助AI模型的不同对齐程度下解决一个简单的决策任务——在线纸牌游戏。研究结果表明，对齐程度与基于AI的决策辅助的效用之间存在正相关关系。此外，我们还发现，对AI置信度值进行后处理，以实现与参与者自身预测置信度的多重校准，既增加了对齐程度，也提升了基于AI的决策辅助的效用。 

---
# Prompt-Based Monte Carlo Tree Search for Mitigating Hallucinations in Large Models 

**Title (ZH)**: 基于提示的蒙特卡洛树搜索方法用于缓解大规模模型中的幻觉问题 

**Authors**: Zhihua Duan, Jialin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13942)  

**Abstract**: With the rapid development of large models in the field of artificial intelligence, how to enhance their application capabilities in handling complex problems in the field of scientific research remains a challenging problem to be solved. This study proposes an improved Monte Carlo Tree Search (MCTS) method based on prompt words. In the simulation search stage, it introduces dynamic adjustment of exploration parameters and adaptive selection strategies, which can better balance exploration and exploitation, thereby reducing the hallucination phenomenon. This paper takes the four subsets of the SciEval dataset as the test objects, and compares the Glm-4-flash+Improved MCTS method with the methods of several existing models. The results show that the Improved MCTS method performs better, providing new ideas and methods for the application of large models in the field of scientific research. 

**Abstract (ZH)**: 随着人工智能领域大型模型的迅速发展，如何增强其在科学研究领域处理复杂问题的能力仍然是一个具有挑战性的问题。本研究提出了一种基于提示词改进的蒙特卡洛树搜索（MCTS）方法。在模拟搜索阶段，该方法引入了探索参数的动态调整和自适应选择策略，可以更好地平衡探索与利用，从而减少幻觉现象。本文以SciEval数据集的四个子集为测试对象，将改进的MCTS方法（Glm-4-flash+Improved MCTS）与多个现有模型的方法进行对比。结果显示，改进的MCTS方法表现更优，为大型模型在科学研究领域的应用提供了新的思路和方法。 

---
# Evaluating Computational Accuracy of Large Language Models in Numerical Reasoning Tasks for Healthcare Applications 

**Title (ZH)**: 评估大型语言模型在健康-care领域数值推理任务中计算准确性的研究 

**Authors**: Arjun R. Malghan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13936)  

**Abstract**: Large Language Models (LLMs) have emerged as transformative tools in the healthcare sector, demonstrating remarkable capabilities in natural language understanding and generation. However, their proficiency in numerical reasoning, particularly in high-stakes domains like in clinical applications, remains underexplored. Numerical reasoning is critical in healthcare applications, influencing patient outcomes, treatment planning, and resource allocation. This study investigates the computational accuracy of LLMs in numerical reasoning tasks within healthcare contexts. Using a curated dataset of 1,000 numerical problems, encompassing real-world scenarios such as dosage calculations and lab result interpretations, the performance of a refined LLM based on the GPT-3 architecture was evaluated. The methodology includes prompt engineering, integration of fact-checking pipelines, and application of regularization techniques to enhance model accuracy and generalization. Key metrics such as precision, recall, and F1-score were utilized to assess the model's efficacy. The results indicate an overall accuracy of 84.10%, with improved performance in straightforward numerical tasks and challenges in multi-step reasoning. The integration of a fact-checking pipeline improved accuracy by 11%, underscoring the importance of validation mechanisms. This research highlights the potential of LLMs in healthcare numerical reasoning and identifies avenues for further refinement to support critical decision-making in clinical environments. The findings aim to contribute to the development of reliable, interpretable, and contextually relevant AI tools for healthcare. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在医疗保健领域展现出革命性的工具作用，展示了在自然语言理解与生成方面非凡的能力。然而，在如临床应用等高风险领域中，它们在数值推理方面的能力尚未得到充分探索。数值推理在医疗保健应用中至关重要，它影响患者结果、治疗计划和资源分配。本研究探讨了LLMs在医疗保健背景下数值推理任务中的计算准确性。通过使用包含1,000个数值问题的定制数据集，涵盖如剂量计算和化验结果解释等真实场景，评估了一个基于GPT-3架构精炼后的LLM的性能。该研究方法包括prompt工程、整合事实核查管道以及应用正则化技术，以提高模型的准确性和泛化能力。精确度、召回率和F1分数等关键指标被用来评估模型的效果。结果表明，整体准确率为84.10%，在简单的数值任务上的表现较好，但在多步推理方面存在挑战。事实核查管道的整合提高了11%的准确性，突显了验证机制的重要性。研究揭示了LLMs在医疗保健数值推理中的潜力，并指出了进一步细化以支持临床环境中关键决策的方向。研究旨在促进可靠、可解释和上下文相关的人工智能工具在医疗保健领域的开发。 

---
# Do LLMs Provide Consistent Answers to Health-Related Questions across Languages? 

**Title (ZH)**: 大型语言模型在不同语言中对健康相关问题的回答是否一致？ 

**Authors**: Ipek Baris Schlicht, Zhixue Zhao, Burcu Sayin, Lucie Flek, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2501.14719)  

**Abstract**: Equitable access to reliable health information is vital for public health, but the quality of online health resources varies by language, raising concerns about inconsistencies in Large Language Models (LLMs) for healthcare. In this study, we examine the consistency of responses provided by LLMs to health-related questions across English, German, Turkish, and Chinese. We largely expand the HealthFC dataset by categorizing health-related questions by disease type and broadening its multilingual scope with Turkish and Chinese translations. We reveal significant inconsistencies in responses that could spread healthcare misinformation. Our main contributions are 1) a multilingual health-related inquiry dataset with meta-information on disease categories, and 2) a novel prompt-based evaluation workflow that enables sub-dimensional comparisons between two languages through parsing. Our findings highlight key challenges in deploying LLM-based tools in multilingual contexts and emphasize the need for improved cross-lingual alignment to ensure accurate and equitable healthcare information. 

**Abstract (ZH)**: 公共健康领域内公平获取可靠健康信息至关重要，但在线健康资源的质量因语言而异，这引发了对大型语言模型（LLMs）在医疗健康领域中一致性问题的关注。本研究旨在探讨LLMs对健康相关问题在英语、德语、土耳其语和中文中的回答一致性。我们通过按疾病类型对健康相关问题进行分类，并扩展至包括土耳其语和中文的多语言范围，对HealthFC数据集进行了大幅扩展。我们揭示了回答中的显著不一致性，这些不一致性可能导致健康误导信息的传播。我们的主要贡献包括：1) 一个包含疾病类别元信息的多语言健康相关问询数据集；2) 一种新颖的基于提示的评估工作流，能够通过解析实现两种语言在子维度上的比较。研究结果突显了在多语言环境中部署基于LLM的工具所面临的关键挑战，并强调了改进跨语言对齐的重要性，以确保医药健康信息的准确性和公平性。 

---
# An Attentive Graph Agent for Topology-Adaptive Cyber Defence 

**Title (ZH)**: 一种适应拓扑结构的注意力图代理用于网络安全防御 

**Authors**: Ilya Orson Sandoval, Isaac Symes Thompson, Vasilios Mavroudis, Chris Hicks  

**Link**: [PDF](https://arxiv.org/pdf/2501.14700)  

**Abstract**: As cyber threats grow increasingly sophisticated, reinforcement learning is emerging as a promising technique to create intelligent, self-improving defensive systems. However, most existing autonomous defensive agents have overlooked the inherent graph structure of computer networks subject to cyber attacks, potentially missing critical information. To address this gap, we developed a custom version of the Cyber Operations Research Gym (CybORG) environment that encodes the observable network state as a directed graph, utilizing realistic and interpretable low-level features. %, like number of open ports and unexpected detected connections. We leverage a Graph Attention Network (GAT) architecture to process node, edge, and global features, and modify its output to be compatible with policy gradient methods in reinforcement learning. GAT policies offer several advantages over standard approaches based on simplistic flattened state observations. They can handle the changes in network topology that occur at runtime when dynamic connections between hosts appear. Policies can be deployed to networks that differ in size to the ones seen during training, enabling a degree of generalisation inaccessible with alternative approaches. Furthermore, the graph neural network policies outputs are explainable in terms of tangible network properties, providing enhanced interpretability of defensive actions. We verify that our low-level graph observations are meaningful enough to train GAT defensive policies that are able to adapt to changing topologies. We evaluate how our trained policies perform when deployed on networks of varying sizes with the same subnetwork structure, comparing them against policies specifically trained for each network configuration. Our study contributes to the development of robust cyber defence systems that can better adapt to real-world network security challenges. 

**Abstract (ZH)**: 随着网络威胁日益复杂，强化学习正逐渐成为创建智能、自我改进防御系统的有前途的技术。然而，现有的大多数自主防御代理忽视了受网络攻击影响的计算机网络中的固有图形结构，可能导致关键信息的缺失。为解决这一问题，我们开发了一个自定义版本的Cyber Operations Research Gym（CybORG）环境，将可观察的网络状态编码为有向图，利用了现实且可解释的低层特征，例如开放端口的数量和意外检测到的连接。我们利用图形注意网络（GAT）架构来处理节点、边缘和全局特征，并修改其输出以与强化学习中的策略梯度方法兼容。与基于简化的展平状态观测的标准方法相比，GAT策略具有多种优势。它们可以处理运行时因主机之间动态连接而发生的网络拓扑变化。这些策略可以在与训练时不同的网络规模上部署，从而实现替代方法无法比拟的泛化能力。此外，图形神经网络策略的输出可以用具体的网络属性来解释，提供了对防御行动增强可解释性的提升。我们验证了我们的低层图形观察足够有意义，能够训练出能够适应拓扑变化的GAT防御策略。我们评估了所训练策略在具有不同规模但相同子网络结构的网络上的表现，并将它们与专门针对每个网络配置训练的策略进行了比较。本研究为开发能够更好地应对现实世界网络安全挑战的稳健型网络防御系统做出了贡献。 

---
# Towards Automated Self-Supervised Learning for Truly Unsupervised Graph Anomaly Detection 

**Title (ZH)**: 朝向自动化自我监督学习的真正无监督图异常检测 

**Authors**: Zhong Li, Yuhang Wang, Matthijs van Leeuwen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14694)  

**Abstract**: Self-supervised learning (SSL) is an emerging paradigm that exploits supervisory signals generated from the data itself, and many recent studies have leveraged SSL to conduct graph anomaly detection. However, we empirically found that three important factors can substantially impact detection performance across datasets: 1) the specific SSL strategy employed; 2) the tuning of the strategy's hyperparameters; and 3) the allocation of combination weights when using multiple strategies. Most SSL-based graph anomaly detection methods circumvent these issues by arbitrarily or selectively (i.e., guided by label information) choosing SSL strategies, hyperparameter settings, and combination weights. While an arbitrary choice may lead to subpar performance, using label information in an unsupervised setting is label information leakage and leads to severe overestimation of a method's performance. Leakage has been criticized as "one of the top ten data mining mistakes", yet many recent studies on SSL-based graph anomaly detection have been using label information to select hyperparameters. To mitigate this issue, we propose to use an internal evaluation strategy (with theoretical analysis) to select hyperparameters in SSL for unsupervised anomaly detection. We perform extensive experiments using 10 recent SSL-based graph anomaly detection algorithms on various benchmark datasets, demonstrating both the prior issues with hyperparameter selection and the effectiveness of our proposed strategy. 

**Abstract (ZH)**: 自我监督学习（SSL）是一种利用数据本身产生的监督信号的新兴范式，许多近期的研究利用SSL进行图异常检测。然而，我们通过实验发现，三个重要因素可能严重影响检测性能：1）采用的具体SSL策略；2）策略超参数的调整；3）当使用多个策略时，组合权重的分配。大多数基于SSL的图异常检测方法通过随意选择或在标注信息引导下选择SSL策略、超参数设置及其组合权重来规避这些问题。随意选择可能导致性能不佳，而在无监督环境中使用标注信息则会导致标注信息泄露，从而严重高估方法的性能。标注信息泄露被认为是“十大数据挖掘错误之一”，但在许多基于SSL的图异常检测研究中，仍然使用标注信息选择超参数。为解决这一问题，我们提出了一种内部评估策略（附有理论分析），用于选择无监督异常检测中SSL的超参数。我们在10种最近的基于SSL的图异常检测算法上进行了广泛实验，各个基准数据集上均展示了超参数选择的先前问题，以及我们提出的策略的有效性。 

---
# Rethinking Table Instruction Tuning 

**Title (ZH)**: 重新思考表格指令调优 

**Authors**: Naihao Deng, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2501.14693)  

**Abstract**: Recent advances in table understanding have focused on instruction-tuning large language models (LLMs) for table-related tasks. However, existing research has overlooked the impact of hyperparameter choices and lacks a comprehensive evaluation of the out-of-domain table understanding ability and the general capabilities of these table LLMs. In this paper, we evaluate these abilities in existing table LLMs, and reveal significant declines in both out-of-domain table understanding and general capabilities compared to their base models. Through systematic analysis, we show that hyperparameters, such as learning rate, can significantly influence both table-specific and general capabilities. Contrary to the existing table instruction-tuning works, we demonstrate that smaller learning rates and fewer training instances can enhance table understanding while preserving general capabilities. Based on our findings, we introduce TAMA, a TAble LLM instruction-tuned from LLaMA 3.1 8B Instruct, which achieves performance on par with, or surpassing GPT-3.5 and GPT-4 on table tasks, while maintaining strong out-of-domain generalization and general capabilities. Our findings highlight the potential for reduced data annotation costs and more efficient model development through careful hyperparameter selection. 

**Abstract (ZH)**: 近年来，表格理解领域的进展主要集中在通过指令调优大型语言模型（LLMs）来处理与表格相关的任务。然而，现有研究忽略了超参数选择的影响，并缺乏对这些表格LLMs的跨域表格理解和通用能力的全面评估。本文评估了现有表格LLMs的这些能力，并揭示了与基模型相比，在跨域表格理解和通用能力方面存在显著下降。通过系统的分析，我们表明，学习率等超参数可以显著影响专门的表格能力和通用能力。与现有的表格指令调优工作不同，我们证明了使用较低的学习率和较少的训练实例可以在保持通用能力的同时提高表格理解能力。基于我们的发现，我们引入了TAMA模型，该模型是从LaMA 3.1 8B Instruct调优而来的一种表格LLM，在表格任务上达到了与GPT-3.5和GPT-4相当或更优的性能，同时维持了强大的跨域泛化能力和通用能力。我们的发现强调了通过精心选择超参数来降低数据标注成本和提高模型开发效率的潜力。 

---
# Approach to Designing CV Systems for Medical Applications: Data, Architecture and AI 

**Title (ZH)**: 医疗应用中CV系统设计方法：数据、架构与人工智能 

**Authors**: Dmitry Ryabtsev, Boris Vasilyev, Sergey Shershakov  

**Link**: [PDF](https://arxiv.org/pdf/2501.14689)  

**Abstract**: This paper introduces an innovative software system for fundus image analysis that deliberately diverges from the conventional screening approach, opting not to predict specific diagnoses. Instead, our methodology mimics the diagnostic process by thoroughly analyzing both normal and pathological features of fundus structures, leaving the ultimate decision-making authority in the hands of healthcare professionals. Our initiative addresses the need for objective clinical analysis and seeks to automate and enhance the clinical workflow of fundus image examination. The system, from its overarching architecture to the modular analysis design powered by artificial intelligence (AI) models, aligns seamlessly with ophthalmological practices. Our unique approach utilizes a combination of state-of-the-art deep learning methods and traditional computer vision algorithms to provide a comprehensive and nuanced analysis of fundus structures. We present a distinctive methodology for designing medical applications, using our system as an illustrative example. Comprehensive verification and validation results demonstrate the efficacy of our approach in revolutionizing fundus image analysis, with potential applications across various medical domains. 

**Abstract (ZH)**: 本文介绍了一种创新的眼底图像分析软件系统，该系统有意地偏离了传统的筛查方法，不预测特定的诊断结果。相反，我们的方法通过详细分析眼底结构的正常和病理特征，模拟了诊断过程，将最终的决策权留给了医疗专业人员。我们的倡议旨在满足客观临床分析的需求，并致力于自动化和优化眼底图像检查的临床工作流程。该系统从总体架构到由人工智能（AI）模型驱动的模块化分析设计，与眼科实践无缝对接。我们独特的方法结合了最先进的深度学习技术和传统的计算机视觉算法，提供了对眼底结构全面且复杂的分析。本文提出了一种设计医疗应用程序的独特方法，并以我们的系统为例进行说明。综合验证和验证结果表明，我们的方法在革新眼底图像分析方面的有效性，具有跨各种医学领域应用的潜力。 

---
# Decoding Generalization from Memorization in Deep Neural Networks 

**Title (ZH)**: 从深度神经网络的记忆中解码泛化能力 

**Authors**: Simran Ketha, Venkatakrishnan Ramaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2501.14687)  

**Abstract**: Overparameterized Deep Neural Networks that generalize well have been key to the dramatic success of Deep Learning in recent years. The reasons for their remarkable ability to generalize are not well understood yet. It has also been known that deep networks possess the ability to memorize training data, as evidenced by perfect or high training accuracies on models trained with corrupted data that have class labels shuffled to varying degrees. Concomitantly, such models are known to generalize poorly, i.e. they suffer from poor test accuracies, due to which it is thought that the act of memorizing substantially degrades the ability to generalize. It has, however, been unclear why the poor generalization that accompanies such memorization, comes about. One possibility is that in the process of training with corrupted data, the layers of the network irretrievably reorganize their representations in a manner that makes generalization difficult. The other possibility is that the network retains significant ability to generalize, but the trained network somehow chooses to readout in a manner that is detrimental to generalization. Here, we provide evidence for the latter possibility by demonstrating, empirically, that such models possess information in their representations for substantially improved generalization, even in the face of memorization. Furthermore, such generalization abilities can be easily decoded from the internals of the trained model, and we build a technique to do so from the outputs of specific layers of the network. We demonstrate results on multiple models trained with a number of standard datasets. 

**Abstract (ZH)**: 近年来，具有良好泛化的过参数化深度神经网络对于深度学习的巨大成功起到了关键作用。它们之所以能够表现出令人瞩目的泛化能力，其原因尚不完全清楚。已知深度网络具备存储训练数据的能力，例如，在使用被篡改的数据训练模型时（训练数据的类别标签被不同程度地打乱），模型能够达到完美的或接近完美的训练准确率。与此同时，这些模型通常表现出较差的泛化能力，即它们在测试集上的准确率较差，人们认为这种存储行为会严重削弱模型的泛化能力。然而，这样的存储行为为何会伴随着较差的泛化能力，其具体原因仍然不清楚。

一种可能性是，在使用被篡改的数据训练时，网络的各层不可逆地重新组织其表征，从而导致泛化困难。另一种可能性是，网络保留了显著的泛化能力，但在训练过程中，模型以某种具体的方式输出，从而损害了泛化能力。

在这里，我们通过实验证据表明，这种模型在其表征中包含大量有助于显著提高泛化的信息，即使在存储的情况下也是如此。此外，这些泛化能力可以从训练模型的内部轻松解码，我们建立了一种技术，通过特定层的输出来实现这一点。我们展示了在多个模型上使用标准数据集进行训练的结果。 

---
# Rethinking Foundation Models for Medical Image Classification through a Benchmark Study on MedMNIST 

**Title (ZH)**: 通过基准研究重新思考基金会模型在医学图像分类中的应用：基于MedMNIST的评估 

**Authors**: Fuping Wu, Bartlomiej W. Papiez  

**Link**: [PDF](https://arxiv.org/pdf/2501.14685)  

**Abstract**: Foundation models are widely employed in medical image analysis, due to their high adaptability and generalizability for downstream tasks. With the increasing number of foundation models being released, model selection has become an important issue. In this work, we study the capabilities of foundation models in medical image classification tasks by conducting a benchmark study on the MedMNIST dataset. Specifically, we adopt various foundation models ranging from convolutional to Transformer-based models and implement both end-to-end training and linear probing for all classification tasks. The results demonstrate the significant potential of these pre-trained models when transferred for medical image classification. We further conduct experiments with different image sizes and various sizes of training data. By analyzing all the results, we provide preliminary, yet useful insights and conclusions on this topic. 

**Abstract (ZH)**: 基础模型在医学图像分析中广泛应用于下游任务，因其高度的适应性和泛化性。随着越来越多的基础模型被发布，模型选择已成为一个重要问题。本文通过对MedMNIST数据集进行基准研究，探讨了基础模型在医学图像分类任务中的能力。具体而言，我们采用了从卷积模型到基于Transformer的各种基础模型，并为所有分类任务实施了端到端的训练和线性探测。结果表明，这些预训练模型在转移学习用于医学图像分类时具有显著潜力。此外，我们还进行了不同图像大小和不同训练数据量的实验。通过分析所有结果，我们提供了有关该主题的初步但有用的研究见解和结论。 

---
# Surface Vision Mamba: Leveraging Bidirectional State Space Model for Efficient Spherical Manifold Representation 

**Title (ZH)**: 表面视觉蟒蛇：利用双向状态空间模型进行高效球面流形表示 

**Authors**: Rongzhao He, Weihao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.14679)  

**Abstract**: Attention-based methods have demonstrated exceptional performance in modelling long-range dependencies on spherical cortical surfaces, surpassing traditional Geometric Deep Learning (GDL) models. However, their extensive inference time and high memory demands pose challenges for application to large datasets with limited computing resources. Inspired by the state space model in computer vision, we introduce the attention-free Vision Mamba (Vim) to spherical surfaces, presenting a domain-agnostic architecture for analyzing data on spherical manifolds. Our method achieves surface patching by representing spherical data as a sequence of triangular patches derived from a subdivided icosphere. The proposed Surface Vision Mamba (SiM) is evaluated on multiple neurodevelopmental phenotype regression tasks using cortical surface metrics from neonatal brains. Experimental results demonstrate that SiM outperforms both attention- and GDL-based methods, delivering 4.8 times faster inference and achieving 91.7% lower memory consumption compared to the Surface Vision Transformer (SiT) under the Ico-4 grid partitioning. Sensitivity analysis further underscores the potential of SiM to identify subtle cognitive developmental patterns. The code is available at this https URL. 

**Abstract (ZH)**: 基于注意力的方法在模拟球面皮层表面的长期依赖关系方面展现了卓越的性能，超越了传统的几何深度学习（GDL）模型。然而，它们的 extensive 推理时间和高内存需求为在资源有限的计算环境中应用大规模数据集带来了挑战。受计算机视觉中状态空间模型的启发，我们在此引入了一种无注意力的 Vision Mamba（Vim），将其应用于球面表面，提出了一种适用于球面流型数据的无领域偏见架构。我们的方法通过将球面数据表示为来自细分icosphere 的三角形片段序列实现表面分割。所提出的 Surface Vision Mamba（SiM）在使用新生儿皮层表面指标的多个神经发展表型回归任务中进行了评估。实验结果表明，SiM 在 inference 速度和内存消耗方面均优于基于注意力和 GDL 的方法，在 Ico-4 网格划分下，SiM 的推理速度快 4.8 倍，内存消耗低 91.7% 且优于 Surface Vision Transformer（SiT）。灵敏度分析进一步表明了 SiM 在识别细微的认知发展模式方面的潜力。代码可在以下链接获取：this https URL。 

---
# A Predictive Approach for Enhancing Accuracy in Remote Robotic Surgery Using Informer Model 

**Title (ZH)**: 基于Informer模型的预测方法在增强远程机器人手术准确性的应用 

**Authors**: Muhammad Hanif Lashari, Shakil Ahmed, Wafa Batayneh, Ashfaq Khokhar  

**Link**: [PDF](https://arxiv.org/pdf/2501.14678)  

**Abstract**: Precise and real-time estimation of the robotic arm's position on the patient's side is essential for the success of remote robotic surgery in Tactile Internet (TI) environments. This paper presents a prediction model based on the Transformer-based Informer framework for accurate and efficient position estimation. Additionally, it combines a Four-State Hidden Markov Model (4-State HMM) to simulate realistic packet loss scenarios. The proposed approach addresses challenges such as network delays, jitter, and packet loss to ensure reliable and precise operation in remote surgical applications. The method integrates the optimization problem into the Informer model by embedding constraints such as energy efficiency, smoothness, and robustness into its training process using a differentiable optimization layer. The Informer framework uses features such as ProbSparse attention, attention distilling, and a generative-style decoder to focus on position-critical features while maintaining a low computational complexity of O(L log L). The method is evaluated using the JIGSAWS dataset, achieving a prediction accuracy of over 90 percent under various network scenarios. A comparison with models such as TCN, RNN, and LSTM demonstrates the Informer framework's superior performance in handling position prediction and meeting real-time requirements, making it suitable for Tactile Internet-enabled robotic surgery. 

**Abstract (ZH)**: 在触觉互联网（TI）环境中进行远程机器人手术时，精确且实时地估计机械臂在病人侧的位置对于手术的成功至关重要。本文提出了一种基于Transformer框架的Informer预测模型，用于准确且高效地进行位置估计。此外，该模型结合了一个四状态隐马尔可夫模型（4-State HMM），以模拟真实的包丢失场景。所提出的方法解决了网络延迟、抖动和包丢失等挑战，以确保在远程外科手术中的可靠性和精确性操作。该方法通过将优化问题嵌入Informer模型来实现这一目标，通过将诸如能效、平滑度和鲁棒性等约束嵌入其训练过程中的可微优化层来优化模型。Informer框架利用诸如ProbSparse注意力、注意力蒸馏以及生成器风格的解码器等特点，专注于关键位置特征，同时保持较低的计算复杂度O(L log L)。该方法使用JIGSAWS数据集进行评估，在各种网络场景下实现了超过90%的预测准确性。与TCN、RNN和LSTM等模型的比较表明，基于Informer框架的方法在处理位置预测和满足实时要求方面表现出更优异的性能，使其适用于触觉互联网支持的机器人手术。 

---
# State Space Models for Extractive Summarization in Low Resource Scenarios 

**Title (ZH)**: 在资源匮乏场景下用于抽取式总结的状态空间模型 

**Authors**: Nisrine Ait Khayi  

**Link**: [PDF](https://arxiv.org/pdf/2501.14673)  

**Abstract**: Extractive summarization involves selecting the most relevant sentences from a text. Recently, researchers have focused on advancing methods to improve state-of-the-art results in low-resource settings. Motivated by these advancements, we propose the MPoincareSum method. This method applies the Mamba state space model to generate the semantics of reviews and sentences, which are then concatenated. A Poincare compression is used to select the most meaningful features, followed by the application of a linear layer to predict sentence relevance based on the corresponding review. Finally, we paraphrase the relevant sentences to create the final summary. To evaluate the effectiveness of MPoincareSum, we conducted extensive experiments using the Amazon review dataset. The performance of the method was assessed using ROUGE scores. The experimental results demonstrate that MPoincareSum outperforms several existing approaches in the literature 

**Abstract (ZH)**: 抽取式摘要涉及从文本中选择最具相关性的句子。最近，研究人员聚焦于在资源稀缺环境下改进方法，以提高最先进的成果。受这些进展的启发，我们提出了一种名为MPoincareSum的方法。该方法利用Mamba状态空间模型生成评论和句子的意义，并将这些意义连接起来。接着使用球面压缩来选择最具意义的特征，再通过线性层预测句子的相关性，基于对应的评论。最后，我们对相关句子进行改写以生成最终的摘要。

为了评估MPoincareSum的有效性，我们在Amazon评论数据集上进行了广泛的实验，并使用ROUGE分数评估方法的性能。实验结果显示，MPoincareSum在文献中比几种现有方法表现更优。 

---
# Neural-Symbolic Message Passing with Dynamic Pruning 

**Title (ZH)**: 动态剪枝的神经-符号消息传递 

**Authors**: Chongzhi Zhang, Junhao Zheng, Zhiping Peng, Qianli Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.14661)  

**Abstract**: Complex Query Answering (CQA) over incomplete Knowledge Graphs (KGs) is a challenging task. Recently, a line of message-passing-based research has been proposed to solve CQA. However, they perform unsatisfactorily on negative queries and fail to address the noisy messages between variable nodes in the query graph. Moreover, they offer little interpretability and require complex query data and resource-intensive training. In this paper, we propose a Neural-Symbolic Message Passing (NSMP) framework based on pre-trained neural link predictors. By introducing symbolic reasoning and fuzzy logic, NSMP can generalize to arbitrary existential first order logic queries without requiring training while providing interpretable answers. Furthermore, we introduce a dynamic pruning strategy to filter out noisy messages between variable nodes. Experimental results show that NSMP achieves a strong performance. Additionally, through complexity analysis and empirical verification, we demonstrate the superiority of NSMP in inference time over the current state-of-the-art neural-symbolic method. Compared to this approach, NSMP demonstrates faster inference times across all query types on benchmark datasets, with speedup ranging from 2$\times$ to over 150$\times$. 

**Abstract (ZH)**: 在不完整知识图谱（Knowledge Graphs, KGs）上进行复杂查询回答（Complex Query Answering, CQA）是一项具有挑战性的任务。近期，基于消息传递的研究方法被提出以解决CQA问题。然而，这些方法在处理负查询时表现不佳，并且未能有效解决查询图中变量节点之间的噪声消息问题。此外，这些方法解释性较差，并且需要复杂的数据以及大量资源进行训练。本文中，我们提出了一种基于先验训练神经链接预测器的神经符号消息传递（Neural-Symbolic Message Passing, NSMP）框架。通过引入符号推理和模糊逻辑，NSMP可以在无需训练的情况下推广到任意存在性一阶逻辑查询，并提供可解释的答案。此外，我们引入了一种动态剪枝策略来过滤掉变量节点之间的噪声消息。实验结果表明，NSMP具有良好的性能。此外，通过复杂性分析和实验证明，NSMP在推理时间上优于当前最先进的神经符号方法。与这种方法相比，NSMP在基准数据集上的所有查询类型上都表现出更快的推理时间，速度从2倍到超过150倍不等。 

---
# MedAgentBench: Dataset for Benchmarking LLMs as Agents in Medical Applications 

**Title (ZH)**: MedAgentBench：医疗应用中LLM作为代理的 benchmarks 数据集 

**Authors**: Yixing Jiang, Kameron C. Black, Gloria Geng, Danny Park, Andrew Y. Ng, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14654)  

**Abstract**: Recent large language models (LLMs) have demonstrated significant advancements, particularly in their ability to serve as agents thereby surpassing their traditional role as chatbots. These agents can leverage their planning and tool utilization capabilities to address tasks specified at a high level. However, a standardized dataset to benchmark the agent capabilities of LLMs in medical applications is currently lacking, making the evaluation of LLMs on complex tasks in interactive healthcare environments challenging. To address this gap, we introduce MedAgentBench, a broad evaluation suite designed to assess the agent capabilities of large language models within medical records contexts. MedAgentBench encompasses 100 patient-specific clinically-derived tasks from 10 categories written by human physicians, realistic profiles of 100 patients with over 700,000 data elements, a FHIR-compliant interactive environment, and an accompanying codebase. The environment uses the standard APIs and communication infrastructure used in modern EMR systems, so it can be easily migrated into live EMR systems. MedAgentBench presents an unsaturated agent-oriented benchmark that current state-of-the-art LLMs exhibit some ability to succeed at. The best model (GPT-4o) achieves a success rate of 72%. However, there is still substantial space for improvement to give the community a next direction to optimize. Furthermore, there is significant variation in performance across task categories. MedAgentBench establishes this and is publicly available at this https URL , offering a valuable framework for model developers to track progress and drive continuous improvements in the agent capabilities of large language models within the medical domain. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在各个方面都显示出显著的进步，尤其是在它们作为代理的能力上，这超越了它们传统的聊天机器人角色。这些代理可以利用其规划和工具利用能力来解决高层次指定的任务。然而，在医学应用中缺乏一个标准化的数据集来评估LLM代理能力，这使得在交互式医疗环境中对LLM进行复杂任务评估变得具有挑战性。为了解决这个问题，我们介绍了MedAgentBench，一个广泛的评估套件，用于评估大型语言模型在医疗记录环境中的代理能力。MedAgentBench 包含来自10个类别、100个由人类医生编制的患者特定临床任务、100个具有超过70万个数据元素的患者真实档案、一个符合FHIR标准的互动环境和相应的代码库。该环境使用现代电子病历系统中常用的标准API和通信基础设施，因此可以轻松迁移到实时电子病历系统中。MedAgentBench 提供了一个未饱和的代理导向基准，当前最先进的LLM在某些方面表现出成功的能力。最佳模型（GPT-4o）的成功率为72%，但仍有改进空间，为社区提供下一步优化的方向。此外，不同任务类别之间的性能波动很大。MedAgentBench 建立了这一点，并在此处 https:// [网站地址] 公开提供，为模型开发者提供了一个有价值的框架，用于跟踪进度并推动大型语言模型代理能力在医学领域的持续改进。 

---
# Federated Domain Generalization with Data-free On-server Gradient Matching 

**Title (ZH)**: 服务器端无数据梯度匹配的联邦领域泛化 

**Authors**: Trong-Binh Nguyen, Minh-Duong Nguyen, Jinsun Park, Quoc-Viet Pham, Won Joo Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14653)  

**Abstract**: Domain Generalization (DG) aims to learn from multiple known source domains a model that can generalize well to unknown target domains. One of the key approaches in DG is training an encoder which generates domain-invariant representations. However, this approach is not applicable in Federated Domain Generalization (FDG), where data from various domains are distributed across different clients. In this paper, we introduce a novel approach, dubbed Federated Learning via On-server Matching Gradient (FedOMG), which can \emph{efficiently leverage domain information from distributed domains}. Specifically, we utilize the local gradients as information about the distributed models to find an invariant gradient direction across all domains through gradient inner product maximization. The advantages are two-fold: 1) FedOMG can aggregate the characteristics of distributed models on the centralized server without incurring any additional communication cost, and 2) FedOMG is orthogonal to many existing FL/FDG methods, allowing for additional performance improvements by being seamlessly integrated with them. Extensive experimental evaluations on various settings to demonstrate the robustness of FedOMG compared to other FL/FDG baselines. Our method outperforms recent SOTA baselines on four FL benchmark datasets (MNIST, EMNIST, CIFAR-10, and CIFAR-100), and three FDG benchmark datasets (PACS, VLCS, and OfficeHome). 

**Abstract (ZH)**: 域泛化（DG）的目标是从多个已知的源域中学习一个模型，使其能够很好地泛化到未知的目标域。DG 的关键技术之一是训练一个编码器，该编码器生成域不变的表示。然而，这种方法在联邦域泛化（FDG）中不适用，在联邦域泛化中，来自不同领域的数据分散在不同的客户端上。在本文中，我们介绍了一种新颖的方法，称为通过服务器端匹配梯度进行联邦学习（FedOMG），该方法可以有效地利用分布式域的域信息。具体来说，我们利用局部梯度作为分布式模型的信息，通过梯度内积的最大化来找到所有域中的不变梯度方向。这一方法的优势体现在两个方面：1) FedOMG可以在中央服务器上聚合分布式模型的特性，而不需要额外的通信成本；2) FedOMG与许多现有的联邦学习/联邦域泛化方法是正交的，这使得它可以通过无缝集成来提高它们的性能。我们在多种设置下进行了广泛的实证评估，以显示与现有联邦学习/联邦域泛化基准相比，FedOMG 的鲁棒性。我们的方法在四个联邦学习基准数据集（MNIST、EMNIST、CIFAR-10 和 CIFAR-100）和三个联邦域泛化基准数据集（PACS、VLCS 和 OfficeHome）上均优于最近的最先进的基准方法。 

---
# Whisper D-SGD: Correlated Noise Across Agents for Differentially Private Decentralized Learning 

**Title (ZH)**: whispers D-SGD：代理之间的相关噪声在不同差分隐私去中心化学习中的应用 

**Authors**: Angelo Rodio, Zheng Chen, Erik G. Larsson  

**Link**: [PDF](https://arxiv.org/pdf/2501.14644)  

**Abstract**: Decentralized learning enables distributed agents to train a shared machine learning model through local computation and peer-to-peer communication. Although each agent retains its dataset locally, the communication of local models can still expose private information to adversaries. To mitigate these threats, local differential privacy (LDP) injects independent noise per agent, but it suffers a larger utility gap than central differential privacy (CDP). We introduce Whisper D-SGD, a novel covariance-based approach that generates correlated privacy noise across agents, unifying several state-of-the-art methods as special cases. By leveraging network topology and mixing weights, Whisper D-SGD optimizes the noise covariance to achieve network-wide noise cancellation. Experimental results show that Whisper D-SGD cancels more noise than existing pairwise-correlation schemes, substantially narrowing the CDP-LDP gap and improving model performance under the same privacy guarantees. 

**Abstract (ZH)**: 去中心化学习使分布在不同节点的智能体能够通过本地计算和点对点通信共同训练一个共享的机器学习模型。尽管每个智能体保留其本地数据集，但本地模型的通信仍然可能暴露私有信息给对手。为缓解这些威胁，局部差分隐私（Local Differential Privacy, LDP）为每个智能体注入独立的噪声，但这种方式比中心化差分隐私（Central Differential Privacy, CDP）在实用性方面存在更大的差距。我们提出了一种名为Whisper D-SGD的新颖协方差方法，该方法在智能体之间生成相关隐私噪声，统一了多种最先进的方法。通过利用网络拓扑结构和混合权重，Whisper D-SGD优化噪声协方差以实现网络范围内的噪声抵消。实验结果表明，Whisper D-SGD能比现有的成对相关方案抵消更多的噪声，显著减小了CDP与LDP之间的差距，并在相同的隐私保障下提高了模型性能。 

---
# ACT-JEPA: Joint-Embedding Predictive Architecture Improves Policy Representation Learning 

**Title (ZH)**: ACT-JEPA：联合嵌入预测架构提高策略表示学习 

**Authors**: Aleksandar Vujinovic, Aleksandar Kovacevic  

**Link**: [PDF](https://arxiv.org/pdf/2501.14622)  

**Abstract**: Learning efficient representations for decision-making policies is a challenge in imitation learning (IL). Current IL methods require expert demonstrations, which are expensive to collect. Consequently, they often have underdeveloped world models. Self-supervised learning (SSL) offers an alternative by allowing models to learn from diverse, unlabeled data, including failures. However, SSL methods often operate in raw input space, making them inefficient. In this work, we propose ACT-JEPA, a novel architecture that integrates IL and SSL to enhance policy representations. We train a policy to predict (1) action sequences and (2) abstract observation sequences. The first objective uses action chunking to improve action prediction and reduce compounding errors. The second objective extends this idea of chunking by predicting abstract observation sequences. We utilize Joint-Embedding Predictive Architecture to predict in abstract representation space, allowing the model to filter out irrelevant details, improve efficiency, and develop a robust world model. Our experiments show that ACT-JEPA improves the quality of representations by learning temporal environment dynamics. Additionally, the model's ability to predict abstract observation sequences results in representations that effectively generalize to action sequence prediction. ACT-JEPA performs on par with established baselines across a range of decision-making tasks. 

**Abstract (ZH)**: 在模仿学习（IL）中，学习有效的表示以制定决策策略是一项挑战。当前的IL方法需要专家演示，这非常昂贵且难以获取，因此它们往往世界模型不够完善。自监督学习（SSL）提供了一种替代方案，通过允许模型从多样化的未标记数据中学习，包括失败数据。然而，SSL方法通常在原始输入空间中运作，导致效率低下。在本研究中，我们提出了一种新颖的ACT-JEPA架构，该架构将IL与SSL相结合，以增强策略表示。我们训练一个策略以预测以下两项内容：（1）动作序列；（2）抽象观察序列。第一项目标利用动作分块提高动作预测精度并减少累积误差。第二项目标扩展了这一分块思想，通过预测抽象观察序列来实现这一目标。我们利用联合嵌入预测架构在抽象表示空间中进行预测，这使模型能够过滤掉无关细节，提高效率并构建稳健的世界模型。实验结果表明，ACT-JEPA通过学习时间环境动态提高了表示的质量。此外，模型预测抽象观察序列的能力导致其表示能够有效地泛化到动作序列预测任务中。ACT-JEPA在多种决策任务中表现与现有基准相当。 

---
# Leveraging Spatial Cues from Cochlear Implant Microphones to Efficiently Enhance Speech Separation in Real-World Listening Scenes 

**Title (ZH)**: 利用 cochlear 种植体麦克风的空域线索以高效改进真实场景中的语音分离 

**Authors**: Feyisayo Olalere, Kiki van der Heijden, Christiaan H. Stronks, Jeroen Briaire, Johan HM Frijns, Marcel van Gerven  

**Link**: [PDF](https://arxiv.org/pdf/2501.14610)  

**Abstract**: Speech separation approaches for single-channel, dry speech mixtures have significantly improved. However, real-world spatial and reverberant acoustic environments remain challenging, limiting the effectiveness of these approaches for assistive hearing devices like cochlear implants (CIs). To address this, we quantify the impact of real-world acoustic scenes on speech separation and explore how spatial cues can enhance separation quality efficiently. We analyze performance based on implicit spatial cues (inherent in the acoustic input and learned by the model) and explicit spatial cues (manually calculated spatial features added as auxiliary inputs). Our findings show that spatial cues (both implicit and explicit) improve separation for mixtures with spatially separated and nearby talkers. Furthermore, spatial cues enhance separation when spectral cues are ambiguous, such as when voices are similar. Explicit spatial cues are particularly beneficial when implicit spatial cues are weak. For instance, single CI microphone recordings provide weaker implicit spatial cues than bilateral CIs, but even single CIs benefit from explicit cues. These results emphasize the importance of training models on real-world data to improve generalizability in everyday listening scenarios. Additionally, our statistical analyses offer insights into how data properties influence model performance, supporting the development of efficient speech separation approaches for CIs and other assistive devices in real-world settings. 

**Abstract (ZH)**: 单声道干音混合的语音分离方法有了显著改进。然而，实际空间和混响声学环境依然构成挑战，限制了这些方法在人工耳蜗植入物（CIs）等辅助听力设备中的有效性。为解决这一问题，我们量化了实际声学场景对语音分离的影响，并探讨了如何通过空间线索高效地提升分离质量。我们基于内在的空间线索（存在于声学输入中且被模型学习）和外显的空间线索（人工计算的增补空间特征）分析了性能。研究结果表明，无论是隐式还是显式空间线索，都能提升分离质量，特别是对于空间分离和邻近说话人的混合物。此外，空间线索在频谱线索模糊时（例如，声音相似时）也提升分离效果。特别地，隐式空间线索较弱时，显式空间线索更为有益。例如，单侧人工耳蜗麦克风记录提供的隐式空间线索比双侧人工耳蜗弱，即便单侧人工耳蜗也从中受益。这些结果强调了在实际数据上训练模型的重要性，以在日常听觉场景中提高泛化能力。此外，我们的统计分析为理解数据特性如何影响模型性能提供了见解，支持在实际环境中为人工耳蜗和其他辅助设备开发高效的语音分离方法。 

---
# Age and Power Minimization via Meta-Deep Reinforcement Learning in UAV Networks 

**Title (ZH)**: 基于元深度强化学习的无人机网络中的年龄和功率最小化 

**Authors**: Sankani Sarathchandra, Eslam Eldeeb, Mohammad Shehab, Hirley Alves, Konstantin Mikhaylov, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2501.14603)  

**Abstract**: Age-of-information (AoI) and transmission power are crucial performance metrics in low energy wireless networks, where information freshness is of paramount importance. This study examines a power-limited internet of things (IoT) network supported by a flying unmanned aerial vehicle(UAV) that collects data. Our aim is to optimize the UAV flight trajectory and scheduling policy to minimize a varying AoI and transmission power combination. To tackle this variation, this paper proposes a meta-deep reinforcement learning (RL) approach that integrates deep Q-networks (DQNs) with model-agnostic meta-learning (MAML). DQNs determine optimal UAV decisions, while MAML enables scalability across varying objective functions. Numerical results indicate that the proposed algorithm converges faster and adapts to new objectives more effectively than traditional deep RL methods, achieving minimal AoI and transmission power overall. 

**Abstract (ZH)**: 在低能耗无线网络中，信息的新鲜度与时延至关重要，而Age-of-Information (AoI)和传输功率是衡量其性能的重要指标。本研究探讨了一种由飞行无人机（UAV）支持的功率有限物联网（IoT）网络，该网络用于收集数据。我们的目标是通过优化无人机飞行轨迹和调度策略，来最小化AoI和传输功率的组合。为了应对这种变异性，本文提出了一种元深度强化学习（RL）方法，该方法将深度Q网络（DQNs）与模型感知的元学习（MAML）相结合。DQNs用于确定最优的无人机决策，而MAML则使该方法在不同目标函数下具备可扩展性。数值结果表明，所提出的算法相比传统的深度RL方法能够更快地收敛并更有效地适应新的目标函数，最终实现AoI和传输功率的最小化。 

---
# ZETA: Leveraging Z-order Curves for Efficient Top-k Attention 

**Title (ZH)**: ZETA：利用Z-order曲线实现高效的Top-k注意力机制 

**Authors**: Qiuhao Zeng, Jerry Huang, Peng Lu, Gezheng Xu, Boxing Chen, Charles Ling, Boyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14577)  

**Abstract**: Over recent years, the Transformer has become a fundamental building block for sequence modeling architectures. Yet at its core is the use of self-attention, whose memory and computational cost grow quadratically with the sequence length $N$, rendering it prohibitively expensive for long sequences. A promising approach is top-$k$ attention, which selects only the $k$ most relevant tokens and achieves performance comparable to vanilla self-attention while significantly reducing space and computational demands. However, causal masks require the current query token to only attend to past tokens, preventing the existing top-$k$ attention method from efficiently searching for the most relevant tokens in parallel, thereby limiting training efficiency. In this work, we propose ZETA, leveraging \textbf{Z}-Order Curves for \textbf{E}fficient \textbf{T}op-$k$ \textbf{A}ttention, to enable parallel querying of past tokens for entire sequences. % in both space and time complexity of $\mathcal{O}(N \log N)$. We first theoretically show that the choice of key and query dimensions involves a trade-off between the curse of dimensionality and the preservation of relative distances after projection. In light of this insight, we propose reducing the dimensionality of keys and queries in contrast to values and further leverage $Z$-order curves to map low-dimensional keys and queries into \emph{one}-dimensional space, which permits parallel sorting, thereby largely improving the efficiency for top-$k$ token selection. Experimental results demonstrate that ZETA matches the performance of standard attention on the synthetic \textsc{Multi-Query Associative Recall} task and outperforms attention and its variants on \textsc{Long Range Arena} and \textsc{WikiText-103} language modeling. 

**Abstract (ZH)**: 近年来，Transformer 成为了序列建模架构中的基本构建块。然而，其核心在于使用自注意力机制，该机制的内存和计算成本随着序列长度 \(N\) 的平方增长，使得对于长序列而言变得极其昂贵。一种有前途的方法是 top-\(k\) 注意力，它仅选择最相关的 \(k\) 个标记，同时显著减少空间和计算需求以达到与标准自注意力相当的性能。然而，因果掩码要求当前查询标记仅可以关注过去的标记，这阻止了现有 top-\(k\) 注意力方法并行地搜索最相关的标记，从而限制了训练效率。在本文中，我们提出了 ZETA，利用 Z-Order 曲线实现高效 top-\(k\) 注意力，以并行查询整个序列的过去标记。ZETA 在空间和时间复杂度上均为 \( \mathcal{O}(N \log N) \)。

我们首先从理论上表明，键和查询维度的选择涉及维度灾难与投影后保持相对距离之间的权衡。基于这一洞见，我们建议减少键和查询的维度（相对于值），并进一步利用 Z-Order 曲线将低维度的键和查询映射到一维空间，使得并行排序成为可能，从而极大地提高了 top-\(k\) 标记选择的效率。实验结果表明，ZETA 在合成的 Multi-Query Associative Recall 任务上达到了标准注意力的性能，并在 Long Range Arena 和 WikiText-103 语言建模任务上优于注意力及其变体。 

---
# Leveraging ChatGPT's Multimodal Vision Capabilities to Rank Satellite Images by Poverty Level: Advancing Tools for Social Science Research 

**Title (ZH)**: 利用ChatGPT的多模态视觉能力按贫困程度对卫星图像进行排序：推进社会科学领域的研究工具 

**Authors**: Hamid Sarmadi, Ola Hall, Thorsteinn Rögnvaldsson, Mattias Ohlsson  

**Link**: [PDF](https://arxiv.org/pdf/2501.14546)  

**Abstract**: This paper investigates the novel application of Large Language Models (LLMs) with vision capabilities to analyze satellite imagery for village-level poverty prediction. Although LLMs were originally designed for natural language understanding, their adaptability to multimodal tasks, including geospatial analysis, has opened new frontiers in data-driven research. By leveraging advancements in vision-enabled LLMs, we assess their ability to provide interpretable, scalable, and reliable insights into human poverty from satellite images. Using a pairwise comparison approach, we demonstrate that ChatGPT can rank satellite images based on poverty levels with accuracy comparable to domain experts. These findings highlight both the promise and the limitations of LLMs in socioeconomic research, providing a foundation for their integration into poverty assessment workflows. This study contributes to the ongoing exploration of unconventional data sources for welfare analysis and opens pathways for cost-effective, large-scale poverty monitoring. 

**Abstract (ZH)**: 本文探究了大型语言模型（LLMs）结合视觉能力在分析卫星影像以预测村庄级贫困方面的新型应用。尽管LLMs最初设计用于自然语言理解，但它们在多模态任务中的适应性，包括空间地理分析，已经为基于数据的研究开辟了新的前沿。通过利用视觉增强的LLMs的最新进展，我们评估了它们在提供可解释、可扩展和可靠的人类贫困见解方面的能力，这些见解来自于卫星影像。采用成对比较的方法，我们展示了ChatGPT可以根据贫困程度对卫星影像进行排名，其准确度与领域专家相当。这些发现突显了LLMs在社会经济研究中的潜力和局限性，为其集成到贫困评估工作流程中提供了基础。本研究对福利分析中非传统数据源的探索作出了贡献，并为低成本、大规模的贫困监测开辟了途径。 

---
# Distributed Conformal Prediction via Message Passing 

**Title (ZH)**: 分布式协作预测 via 消息传递 

**Authors**: Haifeng Wen, Hong Xing, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2501.14544)  

**Abstract**: Post-hoc calibration of pre-trained models is critical for ensuring reliable inference, especially in safety-critical domains such as healthcare. Conformal Prediction (CP) offers a robust post-hoc calibration framework, providing distribution-free statistical coverage guarantees for prediction sets by leveraging held-out datasets. In this work, we address a decentralized setting where each device has limited calibration data and can communicate only with its neighbors over an arbitrary graph topology. We propose two message-passing-based approaches for achieving reliable inference via CP: quantile-based distributed conformal prediction (Q-DCP) and histogram-based distributed conformal prediction (H-DCP). Q-DCP employs distributed quantile regression enhanced with tailored smoothing and regularization terms to accelerate convergence, while H-DCP uses a consensus-based histogram estimation approach. Through extensive experiments, we investigate the trade-offs between hyperparameter tuning requirements, communication overhead, coverage guarantees, and prediction set sizes across different network topologies. 

**Abstract (ZH)**: 对预训练模型的后验校准对于确保可靠推断至关重要，尤其是在医疗保健等安全关键领域。校准预测（Conformal Prediction, CP）提供了一种稳健的后验校准框架，通过利用保留数据集，它能够为预测集提供无分布假设的统计覆盖保证。本文针对一个去中心化的场景进行研究，在这种场景中，每个设备的数据校准资源有限，并且只能通过任意图拓扑与邻居设备进行通信。我们提出了两种基于消息传递的方法，通过校准预测（Conformal Prediction, CP）实现可靠的推断：基于分位数的分布式校准预测（Q-DCP）和基于直方图的分布式校准预测（H-DCP）。Q-DCP 利用分布式分位数回归，并加入定制化的平滑和正则化项以加速收敛过程，而H-DCP 则采用基于共识的直方图估计方法。通过广泛的经验实验，我们探讨了不同网络拓扑下超参数调整要求、通信开销、覆盖保证和预测集大小之间的权衡。 

---
# ABPT: Amended Backpropagation through Time with Partially Differentiable Rewards 

**Title (ZH)**: ABPT：修正的时序反向传播算法与部分可微奖励 

**Authors**: Fanxing Li, Fangyu Sun, Tianbao Zhang, Danping Zou  

**Link**: [PDF](https://arxiv.org/pdf/2501.14513)  

**Abstract**: Using the exact gradients of the rewards to directly optimize policy parameters via backpropagation-through-time (BPTT) enables high training performance for quadrotor tasks. However, designing a fully differentiable reward architecture is often challenging. Partially differentiable rewards will result in biased gradient propagation that degrades training performance. To overcome this limitation, we propose Amended Backpropagation-through-Time (ABPT), a novel approach that mitigates gradient bias while preserving the training efficiency of BPTT. ABPT combines 0-step and N-step returns, effectively reducing the bias by leveraging value gradients from the learned Q-value function. Additionally, it adopts entropy regularization and state initialization mechanisms to encourage exploration during training. We evaluate ABPT on four representative quadrotor flight tasks. Experimental results demonstrate that ABPT converges significantly faster and achieves higher ultimate rewards than existing learning algorithms, particularly in tasks involving partially differentiable rewards. 

**Abstract (ZH)**: 使用奖励的确切梯度直接通过时间反向传播（BPTT）优化策略参数可以实现旋翼无人机任务的高性能训练。然而，设计一个完全可微分的奖励架构往往是具有挑战性的。部分可微分的奖励会导致失真的梯度传播，从而降低训练性能。为了克服这一局限，我们提出了一种名为修正时间反向传播（ABPT）的新型方法，该方法能够减轻梯度偏置同时保持BPTT的训练效率。ABPT结合了0步和N步回报，通过利用学习到的Q值函数的价值梯度有效地减少了偏置。此外，它还采用了熵正则化和状态初始化机制，以鼓励训练过程中的探索。我们对四种代表性的旋翼无人机飞行任务进行了ABPT的评估。实验结果表明，ABPT在训练收敛速度和最终奖励方面显著优于现有的学习算法，尤其是在涉及部分可微分奖励的任务中表现尤为明显。 

---
# RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques 

**Title (ZH)**: RealCritic: 面向效果驱动的语言模型批评评估 

**Authors**: Zhengyang Tang, Ziniu Li, Zhenyang Xiao, Tian Ding, Ruoyu Sun, Benyou Wang, Dayiheng Liu, Fei Huang, Tianyu Liu, Bowen Yu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.14492)  

**Abstract**: Critiques are important for enhancing the performance of Large Language Models (LLMs), enabling both self-improvement and constructive feedback for others by identifying flaws and suggesting improvements. However, evaluating the critique capabilities of LLMs presents a significant challenge due to the open-ended nature of the task. In this work, we introduce a new benchmark designed to assess the critique capabilities of LLMs. Unlike existing benchmarks, which typically function in an open-loop fashion, our approach employs a closed-loop methodology that evaluates the quality of corrections generated from critiques. Moreover, the benchmark incorporates features such as self-critique, cross-critique, and iterative critique, which are crucial for distinguishing the abilities of advanced reasoning models from more classical ones. We implement this benchmark using eight challenging reasoning tasks. We have several interesting findings. First, despite demonstrating comparable performance in direct chain-of-thought generation, classical LLMs significantly lag behind the advanced reasoning-based model o1-mini across all critique scenarios. Second, in self-critique and iterative critique settings, classical LLMs may even underperform relative to their baseline capabilities. We hope that this benchmark will serve as a valuable resource to guide future advancements. The code and data are available at \url{this https URL}. 

**Abstract (ZH)**: 批判对于提升大型语言模型（LLMs）的表现至关重要，它们不仅能够自我改进，还能通过识别错误和提供改进建议来为他人提供建设性的反馈。然而，评估LLMs的批判能力极具挑战性，因为这是一个开放性的任务。在本研究中，我们引入了一个新的基准测试，用于评估LLMs的批判能力。与现有通常采用开环方法的基准测试不同，我们的方法使用了一种闭环方法，通过评估从批判中生成的修正的质量来进行评估。此外，该基准测试还包含自我批判、相互批判和迭代批判等功能，这对于区分基于高级推理的模型与传统模型的能力至关重要。我们使用八个具有挑战性的推理任务来实现这一基准测试。我们有几个有趣的发现。首先，尽管在直接思维链生成任务中表现出相当的性能，经典LLMs在所有批判场景中仍远远落后于基于高级推理的模型o1-mini。其次，在自我批判和迭代批判的设置中，经典LLMs的表现甚至可能不如它们的基础能力。我们希望这一基准测试将为未来的发展提供有价值的资源。该基准测试的代码和数据可在以下链接获取：\url{this https URL}。 

---
# Registration of Longitudinal Liver Examinations for Tumor Progress Assessment 

**Title (ZH)**: longitudinal肝脏检查的注册以评估肿瘤进展 

**Authors**: Walid Yassine, Martin Charachon, Céline Hudelot, Roberto Ardon  

**Link**: [PDF](https://arxiv.org/pdf/2501.14483)  

**Abstract**: Assessing cancer progression in liver CT scans is a clinical challenge, requiring a comparison of scans at different times for the same patient. Practitioners must identify existing tumors, compare them with prior exams, identify new tumors, and evaluate overall disease evolution. This process is particularly complex in liver examinations due to misalignment between exams caused by several factors. Indeed, longitudinal liver examinations can undergo different non-pathological and pathological changes due to non-rigid deformations, the appearance or disappearance of pathologies, and other variations. In such cases, existing registration approaches, mainly based on intrinsic features may distort tumor regions, biasing the tumor progress evaluation step and the corresponding diagnosis. This work proposes a registration method based only on geometrical and anatomical information from liver segmentation, aimed at aligning longitudinal liver images for aided diagnosis. The proposed method is trained and tested on longitudinal liver CT scans, with 317 patients for training and 53 for testing. Our experimental results support our claims by showing that our method is better than other registration techniques by providing a smoother deformation while preserving the tumor burden (total volume of tissues considered as tumor) within the volume. Qualitative results emphasize the importance of smooth deformations in preserving tumor appearance. 

**Abstract (ZH)**: 对肝脏CT扫描进行癌症进展评估是临床上的一个挑战，需要比较同一位患者在不同时间的扫描结果。临床实践者必须识别现有的肿瘤，将其与之前的检查结果进行对比，发现新的肿瘤，并评估疾病的总体演变情况。由于多种因素导致的考前影像之间的对齐差异，这种过程在肝脏检查中尤为复杂。实际上，纵向肝脏检查可能会因非刚性变形、病理或非病理变化以及各种变化等因素而发生不同的非病理和病理变化。在这种情况下，现有的基于内在特征主要的配准方法可能会扭曲肿瘤区域，从而偏倚肿瘤进展评估步骤和相应的诊断。本文提出了一种仅基于肝脏分割的几何和解剖信息的配准方法，旨在对纵向肝脏图像进行对齐，以辅助诊断。该方法基于纵向肝脏CT扫描数据进行训练和测试，其中317名患者用于训练，53名患者用于测试。我们的实验结果支持了我们的观点，表明我们的方法通过提供更平滑的变形同时保留肿瘤负荷（被视为肿瘤的组织总量）在体积内的肿瘤负担，优于其他配准技术。定性结果强调了平滑变形在保留肿瘤外观中的重要性。 

---
# The Pseudo-Dimension of Contracts 

**Title (ZH)**: 合同的拟维数 

**Authors**: Paul Duetting, Michal Feldman, Tomasz Ponitka, Ermis Soumalias  

**Link**: [PDF](https://arxiv.org/pdf/2501.14474)  

**Abstract**: Algorithmic contract design studies scenarios where a principal incentivizes an agent to exert effort on her behalf. In this work, we focus on settings where the agent's type is drawn from an unknown distribution, and formalize an offline learning framework for learning near-optimal contracts from sample agent types. A central tool in our analysis is the notion of pseudo-dimension from statistical learning theory. Beyond its role in establishing upper bounds on the sample complexity, pseudo-dimension measures the intrinsic complexity of a class of contracts, offering a new perspective on the tradeoffs between simplicity and optimality in contract design. Our main results provide essentially optimal tradeoffs between pseudo-dimension and representation error (defined as the loss in principal's utility) with respect to linear and bounded contracts. Using these tradeoffs, we derive sample- and time-efficient learning algorithms, and demonstrate their near-optimality by providing almost matching lower bounds on the sample complexity. Conversely, for unbounded contracts, we prove an impossibility result showing that no learning algorithm exists.
Finally, we extend our techniques in three important ways. First, we provide refined pseudo-dimension and sample complexity guarantees for the combinatorial actions model, revealing a novel connection between the number of critical values and sample complexity. Second, we extend our results to menus of contracts, showing that their pseudo-dimension scales linearly with the menu size. Third, we adapt our algorithms to the online learning setting, where we show that, a polynomial number of type samples suffice to learn near-optimal bounded contracts. Combined with prior work, this establishes a formal separation between expert advice and bandit feedback for this setting. 

**Abstract (ZH)**: 算法合同设计研究了激励代理人为主管进行努力的场景。在本工作中，我们主要关注代理人的类型从未知分布中抽取的情况，并提出了一种离线学习框架，用于从样本代理类型中学习接近最优的合同。我们在分析中使用统计学习理论中的伪维度概念作为核心工具。除了在样本复杂性上限中的作用外，伪维度还衡量合同类的内在复杂性，提供了简单性和最优性之间权衡的新视角。我们的主要成果提供了针对线性和有界合同，伪维度与表示误差（定义为主管的效用损失）之间几乎最优的权衡。利用这些权衡，我们设计了高效的学习算法，并通过几乎匹配的样本复杂性下界展示了它们的近最优性。相反，对于无界合同，我们证明了一个不可能性结果，表明不存在学习算法。

最后，我们以三种重要方式扩展了我们的技术。首先，我们为组合动作模型提供了更精确的伪维度和样本复杂性保证，揭示了关键值数量与样本复杂性之间的新联系。其次，我们将结果扩展到合同菜单的概念，表明伪维度随菜单大小呈线性变化。第三，我们将算法扩展到在线学习设置，展示了多项式数量的类型样本足以学习近最优的有界合同。结合先前的研究工作，这确立了在该设置下专家建议与多臂强盗反馈之间的正式分离。 

---
# Pesti-Gen: Unleashing a Generative Molecule Approach for Toxicity Aware Pesticide Design 

**Title (ZH)**: Pesti-Gen：唤醒一种生成型分子方法，用于毒性aware农药设计 

**Authors**: Taehan Kim, Wonduk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2501.14469)  

**Abstract**: Global climate change has reduced crop resilience and pesticide efficacy, making reliance on synthetic pesticides inevitable, even though their widespread use poses significant health and environmental risks. While these pesticides remain a key tool in pest management, previous machine-learning applications in pesticide and agriculture have focused on classification or regression, leaving the fundamental challenge of generating new molecular structures or designing novel candidates unaddressed. In this paper, we propose Pesti-Gen, a novel generative model based on variational auto-encoders, designed to create pesticide candidates with optimized properties for the first time. Specifically, Pesti-Gen leverages a two-stage learning process: an initial pre-training phase that captures a generalized chemical structure representation, followed by a fine-tuning stage that incorporates toxicity-specific information. The model simultaneously optimizes over multiple toxicity metrics, such as (1) livestock toxicity and (2) aqua toxicity to generate environmentally friendly pesticide candidates. Notably, Pesti-Gen achieves approximately 68\% structural validity in generating new molecular structures, demonstrating the model's effectiveness in producing optimized and feasible pesticide candidates, thereby providing a new way for safer and more sustainable pest management solutions. 

**Abstract (ZH)**: 全球气候变化降低了作物的抗逆性和农药的效果，使得依赖合成农药变得不可避免，尽管它们的广泛应用对健康和环境带来了重大风险。虽然这些农药仍然是害虫管理的关键工具，但之前基于机器学习的应用主要集中在分类或回归上，而忽略了生成全新分子结构或设计新型候选分子的基本挑战。本文提出了一种基于变分自编码器的新颖生成模型Pesti-Gen，旨在首次生成具有优化特性的农药候选分子。具体而言，Pesti-Gen 利用了两阶段的学习过程：初始预训练阶段捕捉通用的化学结构表示，随后的微调阶段整合了毒性特定的信息。模型同时优化了多个毒性指标，如（1）家畜毒性（2）水生生物毒性，以生成环境友好型农药候选分子。值得注意的是，Pesti-Gen 在生成新分子结构时约有68%的结构有效性，这表明该模型在生成优化且可行的农药候选分子方面具有有效性，从而提供了一种更安全、更可持续的害虫管理解决方案。 

---
# Interpretability Analysis of Domain Adapted Dense Retrievers 

**Title (ZH)**: 领域适应密集检索模型的可解释性分析 

**Authors**: Goksenin Yuksel, Jaap Kamps  

**Link**: [PDF](https://arxiv.org/pdf/2501.14459)  

**Abstract**: Dense retrievers have demonstrated significant potential for neural information retrieval; however, they exhibit a lack of robustness to domain shifts, thereby limiting their efficacy in zero-shot settings across diverse domains. Previous research has investigated unsupervised domain adaptation techniques to adapt dense retrievers to target domains. However, these studies have not focused on explainability analysis to understand how such adaptations alter the model's behavior. In this paper, we propose utilizing the integrated gradients framework to develop an interpretability method that provides both instance-based and ranking-based explanations for dense retrievers. To generate these explanations, we introduce a novel baseline that reveals both query and document attributions. This method is used to analyze the effects of domain adaptation on input attributions for query and document tokens across two datasets: the financial question answering dataset (FIQA) and the biomedical information retrieval dataset (TREC-COVID). Our visualizations reveal that domain-adapted models focus more on in-domain terminology compared to non-adapted models, exemplified by terms such as "hedge," "gold," "corona," and "disease." This research addresses how unsupervised domain adaptation techniques influence the behavior of dense retrievers when adapted to new domains. Additionally, we demonstrate that integrated gradients are a viable choice for explaining and analyzing the internal mechanisms of these opaque neural models. 

**Abstract (ZH)**: 密集检索器在神经信息检索中展现了显著的潜力，然而它们在面对领域转移时表现出了脆弱性，这限制了它们在跨领域零样本设置中的有效性。先前的研究已经探索了无监督领域适应技术以使密集检索器适应目标领域。然而，这些研究并没有关注可解释性分析，以理解这些适应如何改变模型的行为。在本文中，我们提出利用集成梯度框架来开发一种解释方法，该方法为密集检索器提供了基于实例和基于排名的解释。为了生成这些解释，我们引入了一个新的基线方法，能够揭示查询和文档的归因。利用这种方法，我们分析了领域适应对两个数据集（金融问答数据集（FIQA）和生物医学信息检索数据集（TREC-COVID））中查询和文档标记的输入归因的影响。我们的可视化结果显示，适应后的模型相比于未适应的模型更关注领域内的术语，例如“对冲”、“黄金”、“冠状病毒”和“疾病”等术语。本研究探讨了无监督领域适应技术如何影响密集检索器在新领域的表现。此外，我们证明集成梯度方法是解释和分析这些不透明神经模型内部机制的一种有效选择。 

---
# Learning more with the same effort: how randomization improves the robustness of a robotic deep reinforcement learning agent 

**Title (ZH)**: 用同样的努力学习更多：随机化如何提高机器人深度 reinforcement 学习代理的鲁棒性 

**Authors**: Lucía Güitta-López, Jaime Boal, Álvaro J. López-López  

**Link**: [PDF](https://arxiv.org/pdf/2501.14443)  

**Abstract**: The industrial application of Deep Reinforcement Learning (DRL) is frequently slowed down because of the inability to generate the experience required to train the models. Collecting data often involves considerable time and economic effort that is unaffordable in most cases. Fortunately, devices like robots can be trained with synthetic experience thanks to virtual environments. With this approach, the sample efficiency problems of artificial agents are mitigated, but another issue arises: the need for efficiently transferring the synthetic experience into the real world (sim-to-real).
This paper analyzes the robustness of a state-of-the-art sim-to-real technique known as progressive neural networks (PNNs) and studies how adding diversity to the synthetic experience can complement it. To better understand the drivers that lead to a lack of robustness, the robotic agent is still tested in a virtual environment to ensure total control on the divergence between the simulated and real models.
The results show that a PNN-like agent exhibits a substantial decrease in its robustness at the beginning of the real training phase. Randomizing certain variables during simulation-based training significantly mitigates this issue. On average, the increase in the model's accuracy is around 25% when diversity is introduced in the training process. This improvement can be translated into a decrease in the required real experience for the same final robustness performance. Notwithstanding, adding real experience to agents should still be beneficial regardless of the quality of the virtual experience fed into the agent. 

**Abstract (ZH)**: 以下是将该内容翻译成中文，同时保持学术规范：

Deep强化学习（DRL）在工业应用中的广泛应用常常受到无法生成足够的经验数据以训练模型的限制。数据收集通常需要大量的时间和经济投入，而在大多数情况下这些投入都是不可行的。幸运的是，由于虚拟环境的支持，机器人等设备可以通过合成经验进行训练。这种方法有助于缓解人工代理的样本效率问题，但同时又产生了一个新的问题：如何高效地将合成经验转移到现实世界中（即从虚拟到现实的迁移）。

本文分析了一种名为渐进神经网络（Progressive Neural Networks, PNN）的先进仿真实现真实技术的鲁棒性，并研究了如何通过增加合成经验的多样性来补充其不足之处。为了更好地理解导致鲁棒性不足的驱动因素，机器人代理仍然在虚拟环境中进行测试，以确保对模拟与真实模型之间的差异进行全面控制。

实验结果表明，在实际训练阶段开始时，PNN 类代理的鲁棒性会出现显著下降。在基于模拟的训练中随机化某些变量可以显著缓解这一问题。在引入多样性的培训过程中，模型的准确性平均提高约25%。这种改进可以转化为在相同最终鲁棒性表现下减少所需的真实经验量。尽管如此，即使虚拟经验的质量不理想，向代理中添加真实经验仍然有益。

这段翻译保持了原文的学术风格和专业术语，同时确保了中文表达的准确性和流畅性。 

---
# Adaptive Rank Allocation for Federated Parameter-Efficient Fine-Tuning of Language Models 

**Title (ZH)**: 联邦参数高效微调语言模型的自适应秩分配 

**Authors**: Fei Wu, Jia Hu, Geyong Min, Shiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14406)  

**Abstract**: Pre-trained Language Models (PLMs) have demonstrated their superiority and versatility in modern Natural Language Processing (NLP), effectively adapting to various downstream tasks through further fine-tuning. Federated Parameter-Efficient Fine-Tuning (FedPEFT) has emerged as a promising solution to address privacy and efficiency challenges in distributed training for PLMs on mobile devices. However, our measurements reveal two key limitations of FedPEFT: heterogeneous data leads to significant performance degradation, and a fixed parameter configuration results in communication inefficiency. To overcome these limitations, we propose FedARA, a novel Federated Adaptive Rank Allocation for parameter-efficient fine-tuning of language models. Specifically, FedARA employs truncated singular value decomposition (SVD) adaptation to enhance flexibility and expressiveness, significantly mitigating the adverse effects of data heterogeneity. Subsequently, it utilizes dynamic rank allocation to progressively identify critical ranks, effectively improving communication efficiency. Lastly, it leverages rank-based module pruning to remove inactive modules, steadily reducing local training time and peak memory usage in each round. Extensive experiments show that FedARA consistently outperforms weak baselines by an average of 8.49\% and strong baselines by 6.95\% across various datasets under data heterogeneity while significantly improving communication efficiency by 2.40\(\times\). Moreover, experiments on AGX Orin, Orin Nano and Raspberry Pi 5 devices demonstrate substantial decreases in total training time and energy consumption by up to 48.90\% and 46.95\%, respectively. 

**Abstract (ZH)**: 预训练语言模型（PLMs）在现代自然语言处理（NLP）中表现出了其优越性和多功能性，并通过进一步微调有效地适应各种下游任务。Federated Parameter-Efficient Fine-Tuning（FedPEFT）作为一种有前景的解决方案，旨在解决分布式训练中PLMs在移动设备上面临的数据隐私和效率问题。然而，我们的测量结果显示FedPEFT存在两个关键限制：异质数据会导致性能显著下降，固定的参数配置导致通信效率低下。为克服这些限制，我们提出了一种名为FedARA的新颖方法，即Federated Adaptive Rank Allocation，用于语言模型的参数高效微调。具体来说，FedARA采用截断奇异值分解（SVD）适应策略，以增强灵活性和表达性，显著减轻了异质数据的负面影响。随后，它利用动态秩分配逐步识别重要秩，有效提高通信效率。最后，它利用基于秩的模块剪枝移除不活跃模块，稳定地减少每轮次的本地训练时间和峰值内存使用。大量实验表明，在异质数据条件下，FedARA相对于弱基线平均提高了8.49%，相对于强基线提高了6.95%，同时显著提高了通信效率2.40倍。此外，在AGX Orin、Orin Nano和Raspberry Pi 5设备上的实验表明，总训练时间和能耗分别减少了48.90%和46.95%。 

---
# SKIL: Semantic Keypoint Imitation Learning for Generalizable Data-efficient Manipulation 

**Title (ZH)**: SKIL：语义关键点imitation学习在通用高效数据驱动操作中的应用 

**Authors**: Shengjie Wang, Jiacheng You, Yihang Hu, Jiongye Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14400)  

**Abstract**: Real-world tasks such as garment manipulation and table rearrangement demand robots to perform generalizable, highly precise, and long-horizon actions. Although imitation learning has proven to be an effective approach for teaching robots new skills, large amounts of expert demonstration data are still indispensible for these complex tasks, resulting in high sample complexity and costly data collection. To address this, we propose Semantic Keypoint Imitation Learning (SKIL), a framework which automatically obtain semantic keypoints with help of vision foundation models, and forms the descriptor of semantic keypoints that enables effecient imitation learning of complex robotic tasks with significantly lower sample complexity. In real world experiments, SKIL doubles the performance of baseline methods in tasks such as picking a cup or mouse, while demonstrating exceptional robustness to variations in objects, environmental changes, and distractors. For long-horizon tasks like hanging a towel on a rack where previous methods fail completely, SKIL achieves a mean success rate of 70\% with as few as 30 demonstrations. Furthermore, SKIL naturally supports cross-embodiment learning due to its semantic keypoints abstraction, our experiments demonstrate that even human videos bring considerable improvement to the learning performance. All these results demonstrate the great success of SKIL in achieving data-efficint generalizable robotic learning. Visualizations and code are available at: this https URL. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文版本：

在服装操作和桌子重排等现实任务中，要求机器人执行通用性强、高精度且具有长期预见性的动作。尽管示例学习（模仿学习）已被证明是教授机器人新技能的有效方法，但这些复杂任务仍需大量专家演示数据，导致样本复杂度高且数据采集成本高昂。为解决这一问题，我们提出了语义关键点示例学习（Semantic Keypoint Imitation Learning，SKIL）框架。该框架通过视觉基础模型自动获取语义关键点，并基于语义关键点描述符高效地进行复杂机器人任务的学习，显著降低了样本复杂度。在实际实验中，SKIL在诸如拾取杯子或鼠标等任务上的表现比基线方法高出一倍，同时展现出对物体变化、环境变化和干扰物的出色鲁棒性。对于以往方法在挂毛巾等长期预见性任务上完全失败的情况，SKIL仅需30个演示即可达到70%的成功率。此外，由于语义关键点的抽象化，SKIL自然支持跨体态学习。我们的实验表明，即使是人类视频也能显著提高学习性能。这些结果证明了SKIL在实现数据高效泛化机器人学习方面的巨大成功。相关可视化和代码可在以下链接获取：this https URL。 

---
# Handling Heterophily in Recommender Systems with Wavelet Hypergraph Diffusion 

**Title (ZH)**: 使用小波超图扩散处理推荐系统中的异质性 

**Authors**: Darnbi Sakong, Thanh Tam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14399)  

**Abstract**: Recommender systems are pivotal in delivering personalised user experiences across various domains. However, capturing the heterophily patterns and the multi-dimensional nature of user-item interactions poses significant challenges. To address this, we introduce FWHDNN (Fusion-based Wavelet Hypergraph Diffusion Neural Networks), an innovative framework aimed at advancing representation learning in hypergraph-based recommendation tasks. The model incorporates three key components: (1) a cross-difference relation encoder leveraging heterophily-aware hypergraph diffusion to adapt message-passing for diverse class labels, (2) a multi-level cluster-wise encoder employing wavelet transform-based hypergraph neural network layers to capture multi-scale topological relationships, and (3) an integrated multi-modal fusion mechanism that combines structural and textual information through intermediate and late-fusion strategies. Extensive experiments on real-world datasets demonstrate that FWHDNN surpasses state-of-the-art methods in accuracy, robustness, and scalability in capturing high-order interconnections between users and items. 

**Abstract (ZH)**: 推荐系统在各个领域提供个性化用户体验方面发挥着关键作用。然而，捕捉用户项交互的异质性和多维性质带来了显著挑战。为了解决这一问题，我们提出了基于融合的波let超图扩散神经网络（FWHDNN，Fusion-based Wavelet Hypergraph Diffusion Neural Networks），这是一种创新框架，旨在推进基于超图的推荐任务中的表示学习。该模型包含三个关键组件：（1）一个基于异质性感知的超图扩散的交叉差关系编码器，用于适应消息传递以适应不同的类别标签，（2）一个多尺度聚类编码器，通过波let变换为基础的超图神经网络层捕捉多尺度拓扑关系，（3）一种集成的多模态融合机制，通过中间融合和晚期融合策略结合结构和文本信息。在真实世界数据集上的广泛实验表明，FWHDNN在准确率、稳健性和可扩展性方面优于现有的最先进的方法，在捕捉用户和项之间的高阶关联方面表现更优。 

---
# ECTIL: Label-efficient Computational Tumour Infiltrating Lymphocyte (TIL) assessment in breast cancer: Multicentre validation in 2,340 patients with breast cancer 

**Title (ZH)**: ECTIL：乳腺癌中基于计算的肿瘤浸润淋巴细胞（TIL）评估的标签高效方法：在2,340例乳腺癌患者中的多中心验证 

**Authors**: Yoni Schirris, Rosie Voorthuis, Mark Opdam, Marte Liefaard, Gabe S Sonke, Gwen Dackus, Vincent de Jong, Yuwei Wang, Annelot Van Rossum, Tessa G Steenbruggen, Lars C Steggink, Liesbeth G.E. de Vries, Marc van de Vijver, Roberto Salgado, Efstratios Gavves, Paul J van Diest, Sabine C Linn, Jonas Teuwen, Renee Menezes, Marleen Kok, Hugo Horlings  

**Link**: [PDF](https://arxiv.org/pdf/2501.14379)  

**Abstract**: The level of tumour-infiltrating lymphocytes (TILs) is a prognostic factor for patients with (triple-negative) breast cancer (BC). Computational TIL assessment (CTA) has the potential to assist pathologists in this labour-intensive task, but current CTA models rely heavily on many detailed annotations. We propose and validate a fundamentally simpler deep learning based CTA that can be trained in only ten minutes on hundredfold fewer pathologist annotations. We collected whole slide images (WSIs) with TILs scores and clinical data of 2,340 patients with BC from six cohorts including three randomised clinical trials. Morphological features were extracted from whole slide images (WSIs) using a pathology foundation model. Our label-efficient Computational stromal TIL assessment model (ECTIL) directly regresses the TILs score from these features. ECTIL trained on only a few hundred samples (ECTIL-TCGA) showed concordance with the pathologist over five heterogeneous external cohorts (r=0.54-0.74, AUROC=0.80-0.94). Training on all slides of five cohorts (ECTIL-combined) improved results on a held-out test set (r=0.69, AUROC=0.85). Multivariable Cox regression analyses indicated that every 10% increase of ECTIL scores was associated with improved overall survival independent of clinicopathological variables (HR 0.86, p<0.01), similar to the pathologist score (HR 0.87, p<0.001). We demonstrate that ECTIL is highly concordant with an expert pathologist and obtains a similar hazard ratio. ECTIL has a fundamentally simpler design than existing methods and can be trained on orders of magnitude fewer annotations. Such a CTA may be used to pre-screen patients for, e.g., immunotherapy clinical trial inclusion, or as a tool to assist clinicians in the diagnostic work-up of patients with BC. Our model is available under an open source licence (this https URL). 

**Abstract (ZH)**: 肿瘤浸润淋巴细胞（TILs）的数量是三阴性乳腺癌（BC）患者预后的因素。计算淋巴细胞评估（CTA）有可能帮助病理学家完成这一劳动密集型任务，但当前的CTA模型依赖于大量详细的注释。我们提出并验证了一种基于深度学习、从根本上更简单的CTA方法，这种模型仅需十分钟即可在少量病理学家注释的基础上进行训练。我们收集了6个队列中的2340名乳腺癌（BC）患者的全视野图像（WSIs）及其临床数据，包括三个随机临床试验。从全视野图像（WSIs）中提取形态特征，并使用病理学基础模型。我们的标签高效计算结缔组织TIL评估模型（ECTIL）直接从这些特征中回归TILs评分。仅在数百个样本上训练的ECTIL（ECTIL-TCGA）在五个异质性的外部队列中与病理学家显示出一致性（r=0.54-0.74，AUROC=0.80-0.94）。在五个队列的所有切片上进行训练的ECTIL（ECTIL-combined）在保留的测试集上改进了结果（r=0.69，AUROC=0.85）。多变量Cox回归分析表明，每增加10%的ECTIL评分，患者的总生存率均有改善，且独立于临床病理学变量（HR 0.86，p<0.01），与病理学家评分相似（HR 0.87，p<0.001）。研究结果表明，ECTIL与专家病理学家高度一致，并获得了相似的危险比值。ECTIL的设计比现有方法更简单，且可以在数量级上少得多的注释基础上进行训练。这样的CTA可以用于患者预筛选，例如免疫治疗临床试验的纳入，或者作为辅助临床诊断的工具。我们的模型在开源许可下可供使用（此链接：this https URL）。 

---
# DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing 

**Title (ZH)**: 穿着风格：通过风格子空间编辑实现高效个性化问答 

**Authors**: Xinyu Ma, Yifeng Xu, Yang Lin, Tianlong Wang, Xu Chu, Xin Gao, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14371)  

**Abstract**: We introduce DRESS, a novel approach for generating stylized large language model (LLM) responses through representation editing. Existing methods like prompting and fine-tuning are either insufficient for complex style adaptation or computationally expensive, particularly in tasks like NPC creation or character role-playing. Our approach leverages the over-parameterized nature of LLMs to disentangle a style-relevant subspace within the model's representation space to conduct representation editing, ensuring a minimal impact on the original semantics. By applying adaptive editing strengths, we dynamically adjust the steering vectors in the style subspace to maintain both stylistic fidelity and semantic integrity. We develop two stylized QA benchmark datasets to validate the effectiveness of DRESS, and the results demonstrate significant improvements compared to baseline methods such as prompting and ITI. In short, DRESS is a lightweight, train-free solution for enhancing LLMs with flexible and effective style control, making it particularly useful for developing stylized conversational agents. Codes and benchmark datasets are available at this https URL. 

**Abstract (ZH)**: 我们引入了DRESS，这是一种通过表示编辑生成风格化大型语言模型（LLM）响应的新颖方法。现有的方法，如提示和微调，在复杂风格适应方面要么不够充分，要么在计算成本上非常昂贵，特别是在创建NPC或角色扮演等任务中。我们的方法利用了LLM的过度参数化特性，在模型的表示空间中分离出一个与风格相关的子空间，以进行表示编辑，从而确保对原始语义的最小影响。通过应用自适应编辑强度，我们可以动态调整风格子空间中的导向向量，以保持风格的真实性和语义的一致性。我们构建了两个风格化的问答基准数据集来验证DRESS的有效性，结果表明其与提示和交互式文本智能（ITI）等基线方法相比取得了显著改进。简而言之，DRESS是一种轻量级、无需训练的解决方案，用于通过灵活有效的样式控制增强LLM，使它特别适用于开发风格化的对话代理。相关代码和基准数据集可在以下链接获取：this https URL。 

---
# HorNets: Learning from Discrete and Continuous Signals with Routing Neural Networks 

**Title (ZH)**: HorNets：通过路由神经网络学习离散和连续信号 

**Authors**: Boshko koloski, Nada Lavrač, Blaž Škrlj  

**Link**: [PDF](https://arxiv.org/pdf/2501.14346)  

**Abstract**: Construction of neural network architectures suitable for learning from both continuous and discrete tabular data is a challenging research endeavor. Contemporary high-dimensional tabular data sets are often characterized by a relatively small instance count, requiring data-efficient learning. We propose HorNets (Horn Networks), a neural network architecture with state-of-the-art performance on synthetic and real-life data sets from scarce-data tabular domains. HorNets are based on a clipped polynomial-like activation function, extended by a custom discrete-continuous routing mechanism that decides which part of the neural network to optimize based on the input's cardinality. By explicitly modeling parts of the feature combination space or combining whole space in a linear attention-like manner, HorNets dynamically decide which mode of operation is the most suitable for a given piece of data with no explicit supervision. This architecture is one of the few approaches that reliably retrieves logical clauses (including noisy XNOR) and achieves state-of-the-art classification performance on 14 real-life biomedical high-dimensional data sets. HorNets are made freely available under a permissive license alongside a synthetic generator of categorical benchmarks. 

**Abstract (ZH)**: 从连续和离散表格数据中学习的神经网络架构构建是一个具有挑战性的研究课题。当代高维表格数据集往往具有相对较小的样本数量，要求具有高效的数据学习能力。我们提出了一种名为HorNets（赫恩网络）的神经网络架构，该架构在合成数据集和现实生活中稀数据表格领域的实际数据集上表现出最先进的性能。HorNets 基于裁剪过的类似多项式的激活函数，并通过自定义的离散-连续路由机制来决定根据输入的基数优化哪个部分的神经网络。通过明确地建模特征组合空间的部分或以类似于线性注意力的方式结合整个空间，HorNets 动态地决定给定数据最适合的操作模式，无需显式的监督。该架构是少数几种能够可靠地检索逻辑子句（包括有噪声的XNOR）并获得14个现实生活中高维生物医学数据集最优分类性能的方法之一。HorNets 在一个宽松的许可下免费提供，并附带一个生成类别基准的合成生成器。 

---
# Relative Layer-Wise Relevance Propagation: a more Robust Neural Networks eXplaination 

**Title (ZH)**: 相对逐层相关性传播：一种更 robust 的神经网络解释方法 

**Authors**: Eric Nyiri, Olivier Gibaru  

**Link**: [PDF](https://arxiv.org/pdf/2501.14322)  

**Abstract**: Machine learning methods are solving very successfully a plethora of tasks, but they have the disadvantage of not providing any information about their decision. Consequently, estimating the reasoning of the system provides additional information. For this, Layer-Wise Relevance Propagation (LRP) is one of the methods in eXplainable Machine Learning (XML). Its purpose is to provide contributions of any neural network output in the domain of its input. The main drawback of current methods is mainly due to division by small values. To overcome this problem, we provide a new definition called Relative LRP where the classical conservation law is satisfied up to a multiplicative factor but without divisions by small values except for Resnet skip connection. In this article, we will focus on image classification. This allows us to visualize the contributions of a pixel to the predictions of a multi-layer neural network. Pixel contributions provide a focus to further analysis on regions of potential interest. R-LRP can be applied for any dense, CNN or residual neural networks. Moreover, R-LRP doesn't need any hyperparameters to tune contrary to other LRP methods. We then compare the R-LRP method on different datasets with simple CNN, VGG16, VGG19 and Resnet50 networks. 

**Abstract (ZH)**: 机器学习方法成功解决了大量的任务，但它们的一个缺点是没有提供其决策过程的任何信息。因此，估计系统的推理过程提供了额外的信息。为实现这一目标，逐层相关性传播（LRP）是可解释机器学习（XML）中的一种方法。其目的是为任何神经网络输出在输入域中的贡献提供解释。当前方法的主要缺点主要是由于除以小数值。为克服这一问题，我们提供了一种新的定义——相对LRP（Relative LRP），其经典守恒定律在乘法因子内得到满足，但无需除以小数值（除了ResNet跳连结构）。在本文中，我们将重点讨论图像分类任务。这使我们能够可视化像素对多层神经网络预测的贡献。像素贡献为后续对潜在感兴趣区域的分析提供了关注点。相对LRP（R-LRP）可以应用于任何密集连接、卷积神经网络（CNN）或残差神经网络（ResNet）。此外，与其他LRP方法不同，R-LRP 不需要调参的超参数。最后，我们将在不同数据集上将R-LRP方法与简单的CNN、VGG16、VGG19和Resnet50网络进行比较。 

---
# Permutation-based multi-objective evolutionary feature selection for high-dimensional data 

**Title (ZH)**: 基于排列的多目标进化特征选择方法用于高维数据 

**Authors**: Raquel Espinosa, Gracia Sánchez, José Palma, Fernando Jiménez  

**Link**: [PDF](https://arxiv.org/pdf/2501.14310)  

**Abstract**: Feature selection is a critical step in the analysis of high-dimensional data, where the number of features often vastly exceeds the number of samples. Effective feature selection not only improves model performance and interpretability but also reduces computational costs and mitigates the risk of overfitting. In this context, we propose a novel feature selection method for high-dimensional data, based on the well-known permutation feature importance approach, but extending it to evaluate subsets of attributes rather than individual features. This extension more effectively captures how interactions among features influence model performance. The proposed method employs a multi-objective evolutionary algorithm to search for candidate feature subsets, with the objectives of maximizing the degradation in model performance when the selected features are shuffled, and minimizing the cardinality of the feature subset. The effectiveness of our method has been validated on a set of 24 publicly available high-dimensional datasets for classification and regression tasks, and compared against 9 well-established feature selection methods designed for high-dimensional problems, including the conventional permutation feature importance method. The results demonstrate the ability of our approach in balancing accuracy and computational efficiency, providing a powerful tool for feature selection in complex, high-dimensional datasets. 

**Abstract (ZH)**: 特征选择是高维数据分析中的一个关键步骤，其中特征的数量往往远远超过样本数量。有效的特征选择不仅能够提高模型性能和可解释性，还能降低计算成本并减轻过拟合的风险。在此背景下，我们提出了一种新的高维数据特征选择方法，基于广为人知的置换特征重要性方法，但将其扩展为评估特征子集的性能影响，而不仅仅是评估单个特征。这一扩展更有效地捕捉了特征间交互作用对模型性能的影响。所提出的方法利用多目标进化算法搜索候选特征子集，目标是最大化所选特征打乱后模型性能的下降幅度，并最小化特征子集的基数。通过对24个公开的高维数据集进行分类和回归任务的验证，并与9种针对高维问题广泛认可的特征选择方法进行了比较（包括传统的置换特征重要性方法），我们验证了该方法的有效性。结果表明，该方法在准确性和计算效率之间实现了良好的平衡，为复杂高维数据集的特征选择提供了一个强大的工具。 

---
# Learning Primitive Relations for Compositional Zero-Shot Learning 

**Title (ZH)**: 学习基础关系实现组合零样本学习 

**Authors**: Insu Lee, Jiseob Kim, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2501.14308)  

**Abstract**: Compositional Zero-Shot Learning (CZSL) aims to identify unseen state-object compositions by leveraging knowledge learned from seen compositions. Existing approaches often independently predict states and objects, overlooking their relationships. In this paper, we propose a novel framework, learning primitive relations (LPR), designed to probabilistically capture the relationships between states and objects. By employing the cross-attention mechanism, LPR considers the dependencies between states and objects, enabling the model to infer the likelihood of unseen compositions. Experimental results demonstrate that LPR outperforms state-of-the-art methods on all three CZSL benchmark datasets in both closed-world and open-world settings. Through qualitative analysis, we show that LPR leverages state-object relationships for unseen composition prediction. 

**Abstract (ZH)**: 组合零样本学习（CZSL）旨在通过利用从已知组合中学到的知识来识别未见过的状态-对象组合。现有方法往往独立地预测状态和对象，忽略了它们之间的关系。在本文中，我们提出了一种新的框架，学习基础关系（LPR），该框架旨在概率性地捕捉状态和对象之间的关系。通过采用交叉注意力机制，LPR考虑了状态和对象之间的依赖关系，使模型能够推断出未见过组合的可能性。实验结果表明，LPR在所有三个CZSL基准数据集的闭集和开集设置中均优于现有最佳方法。通过定性分析，我们展示了LPR利用状态-对象关系进行未见过组合预测的过程。 

---
# A Zero-Shot LLM Framework for Automatic Assignment Grading in Higher Education 

**Title (ZH)**: 面向高等教育中自动评分任务的零样本大型语言模型框架 

**Authors**: Calvin Yeung, Jeff Yu, King Chau Cheung, Tat Wing Wong, Chun Man Chan, Kin Chi Wong, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2501.14305)  

**Abstract**: Automated grading has become an essential tool in education technology due to its ability to efficiently assess large volumes of student work, provide consistent and unbiased evaluations, and deliver immediate feedback to enhance learning. However, current systems face significant limitations, including the need for large datasets in few-shot learning methods, a lack of personalized and actionable feedback, and an overemphasis on benchmark performance rather than student experience. To address these challenges, we propose a Zero-Shot Large Language Model (LLM)-Based Automated Assignment Grading (AAG) system. This framework leverages prompt engineering to evaluate both computational and explanatory student responses without requiring additional training or fine-tuning. The AAG system delivers tailored feedback that highlights individual strengths and areas for improvement, thereby enhancing student learning outcomes. Our study demonstrates the system's effectiveness through comprehensive evaluations, including survey responses from higher education students that indicate significant improvements in motivation, understanding, and preparedness compared to traditional grading methods. The results validate the AAG system's potential to transform educational assessment by prioritizing learning experiences and providing scalable, high-quality feedback. 

**Abstract (ZH)**: 自动评分已成为教育技术中的一个关键技术工具，因为它能够高效地评估大量学生作业，提供一致且无偏见的评价，并立即提供反馈以提高学习效果。然而，当前的系统面临着显著的限制，包括在少数样本学习方法中需要大量数据集、缺乏个性化和可操作的反馈，以及过分强调基准性能而非学生体验。为了应对这些挑战，我们提出了一种基于零样本大规模语言模型（LLM）的自动作业评分（AAG）系统。该框架利用提示工程来评估学生的计算性和解释性回答，无需额外的训练或微调。AAG系统提供定制化的反馈，强调学生的强项和需要改进的地方，从而提高学生的学习成果。我们的研究通过全面评估来证明系统的有效性，包括来自高等教育学生的调查反馈表明，与传统评分方法相比，AAG系统能够显著提高动机、理解能力和应对准备。研究结果验证了AAG系统在优先考虑学习体验并通过提供可扩展和高质量的反馈来变革教育评估方面的重要潜力。 

---
# Examining Alignment of Large Language Models through Representative Heuristics: The Case of Political Stereotypes 

**Title (ZH)**: 通过代表性启发式方法考察大规模语言模型的一致性：以政治刻板印象为例 

**Authors**: Sullam Jeoung, Yubin Ge, Haohan Wang, Jana Diesner  

**Link**: [PDF](https://arxiv.org/pdf/2501.14294)  

**Abstract**: Examining the alignment of large language models (LLMs) has become increasingly important, particularly when these systems fail to operate as intended. This study explores the challenge of aligning LLMs with human intentions and values, with specific focus on their political inclinations. Previous research has highlighted LLMs' propensity to display political leanings, and their ability to mimic certain political parties' stances on various issues. However, the extent and conditions under which LLMs deviate from empirical positions have not been thoroughly examined. To address this gap, our study systematically investigates the factors contributing to LLMs' deviations from empirical positions on political issues, aiming to quantify these deviations and identify the conditions that cause them.
Drawing on cognitive science findings related to representativeness heuristics -- where individuals readily recall the representative attribute of a target group in a way that leads to exaggerated beliefs -- we scrutinize LLM responses through this heuristics lens. We conduct experiments to determine how LLMs exhibit stereotypes by inflating judgments in favor of specific political parties. Our results indicate that while LLMs can mimic certain political parties' positions, they often exaggerate these positions more than human respondents do. Notably, LLMs tend to overemphasize representativeness to a greater extent than humans. This study highlights the susceptibility of LLMs to representativeness heuristics, suggeseting potential vulnerabilities to political stereotypes. We propose prompt-based mitigation strategies that demonstrate effectiveness in reducing the influence of representativeness in LLM responses. 

**Abstract (ZH)**: 大型语言模型（LLM）与人类意图和价值观的对齐变得越来越重要，尤其是在这些系统未能按预期运行时。本研究探讨了将LLM与人类意图和价值观对齐的挑战，特别关注它们的政治倾向。以往研究已指出，LLM倾向于表现出政治倾向，并能够模仿某些政治党派在各种问题上的立场。然而，LLM在多大程度上以及在何种条件下偏离客观立场尚未得到充分研究。为填补这一空白，本研究系统地调查了导致LLM在政治问题上偏离客观立场的因素，旨在量化这些偏差并识别导致这些偏差的条件。

本研究借鉴认知科学中代表性启发法的发现，即个体容易回忆目标群体的代表性特征，从而导致夸大的信念，通过这一启发法视角审视LLM的反应。我们通过实验来确定LLM如何通过夸大特定政治党派的支持度来表现出刻板印象。结果表明，虽然LLM能够模仿某些政治党派的立场，但它们往往夸大这些立场的程度超过人类受访者。值得指出的是，LLM倾向于在更大程度上过度强调代表性。本研究突显了LLM对代表性启发法的敏感性，暗示其可能在政治刻板印象方面存在脆弱性。我们提出了基于提示的缓解策略，这些策略能够有效减少代表性对LLM反应的影响。 

---
# A Comprehensive Framework for Semantic Similarity Detection Using Transformer Architectures and Enhanced Ensemble Techniques 

**Title (ZH)**: 使用变换器架构和增强集成技术的语义相似性检测综合框架 

**Authors**: Lifu Gao, Qi Zhang, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14288)  

**Abstract**: Detecting AI-generated text, especially in short-context documents, is difficult because there is not enough context for accurate classification. This paper presents a new teacher-student model that uses domain adaptation and data augmentation to solve these problems. The teacher model, which combines DeBERTa-v3-large and Mamba-790m, learns semantic knowledge through domain-specific fine-tuning. The student model handles short-context text more efficiently. The system uses a Mean Squared Error (MSE) loss function to guide the student's learning, improving both accuracy and efficiency. Also, data augmentation methods like spelling correction and error injection make the model more robust. Experimental results show that this approach works better than baseline methods, proving its usefulness for real-time AI-generated text detection and other text classification tasks. 

**Abstract (ZH)**: 检测AI生成的文本，尤其是在短文本文档中，由于缺乏足够的上下文信息，准确分类变得困难。本文提出了一种新的教师-学生模型，该模型通过领域适应和数据增强来解决这些问题。教师模型结合了DeBERTa-v3-large和Mamba-790m，通过领域特定的微调学习语义知识。学生模型则更有效地处理短文本。该系统使用均方误差（MSE）损失函数引导学生的学习，提高了准确性和效率。此外，通过拼写修正和错误注入等数据增强方法使模型更具鲁棒性。实验结果表明，这种方法比基线方法更有效，证明了其在实时AI生成文本检测及其他文本分类任务中的实用性。 

---
# Active Learning for Continual Learning: Keeping the Past Alive in the Present 

**Title (ZH)**: 持续学习中的主动学习：让过去在当下保持鲜活 

**Authors**: Jaehyun Park, Dongmin Park, Jae-Gil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.14278)  

**Abstract**: Continual learning (CL) enables deep neural networks to adapt to ever-changing data distributions. In practice, there may be scenarios where annotation is costly, leading to active continual learning (ACL), which performs active learning (AL) for the CL scenarios when reducing the labeling cost by selecting the most informative subset is preferable. However, conventional AL strategies are not suitable for ACL, as they focus solely on learning the new knowledge, leading to catastrophic forgetting of previously learned tasks. Therefore, ACL requires a new AL strategy that can balance the prevention of catastrophic forgetting and the ability to quickly learn new tasks. In this paper, we propose AccuACL, Accumulated informativeness-based Active Continual Learning, by the novel use of the Fisher information matrix as a criterion for sample selection, derived from a theoretical analysis of the Fisher-optimality preservation properties within the framework of ACL, while also addressing the scalability issue of Fisher information-based AL. Extensive experiments demonstrate that AccuACL significantly outperforms AL baselines across various CL algorithms, increasing the average accuracy and forgetting by 23.8% and 17.0%, respectively, in average. 

**Abstract (ZH)**: 连续学习（CL）使深度神经网络能够适应不断变化的数据分布。在实践中，可能存在标注成本高昂的场景，从而导致主动连续学习（ACL），在这种场景下，通过选择最具信息性的子集来降低标注成本时，进行主动学习（AL）是更为优选的方法。然而，传统的AL策略并不适用于ACL，因为它们仅专注于学习新知识，这会导致对先前学习任务的灾难性遗忘。因此，ACL需要一种新的AL策略，能够平衡防止灾难性遗忘和快速学习新任务的能力。在本文中，我们提出了一种名为AccuACL（基于累积信息量的主动连续学习）的新方法，通过在ACL框架内理论上分析Fisher-优化保持特性，采用Fisher信息矩阵作为样本选择的标准，同时解决基于Fisher信息的AL的可扩展性问题。广泛的实验表明，AccuACL在各种CL算法中显著优于现有的AL基线，平均准确率提高了23.8%，平均遗忘率降低了17.0%。 

---
# Global Semantic-Guided Sub-image Feature Weight Allocation in High-Resolution Large Vision-Language Models 

**Title (ZH)**: 全球语义引导的子图像特征权重分配在高分辨率大型视觉-语言模型中 

**Authors**: Yuxuan Liang, Xu Li, Xiaolei Chen, Haotian Chen, Yi Zheng, Chenghang Lai, Bin Li, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2501.14276)  

**Abstract**: As the demand for high-resolution image processing in Large Vision-Language Models (LVLMs) grows, sub-image partitioning has become a popular approach for mitigating visual information loss associated with fixed-resolution processing. However, existing partitioning methods uniformly process sub-images, resulting in suboptimal image understanding. In this work, we reveal that the sub-images with higher semantic relevance to the entire image encapsulate richer visual information for preserving the model's visual understanding ability. Therefore, we propose the Global Semantic-guided Weight Allocator (GSWA) module, which dynamically allocates weights to sub-images based on their relative information density, emulating human visual attention mechanisms. This approach enables the model to focus on more informative regions, overcoming the limitations of uniform treatment. We integrate GSWA into the InternVL2-2B framework to create SleighVL, a lightweight yet high-performing model. Extensive experiments demonstrate that SleighVL outperforms models with comparable parameters and remains competitive with larger models. Our work provides a promising direction for more efficient and contextually aware high-resolution image processing in LVLMs, advancing multimodal system development. 

**Abstract (ZH)**: 随着对高分辨率图像处理的需求在大型视觉-语言模型（LVLMs）中不断增加，子图像分割已成为缓解固定分辨率处理过程中视觉信息损失的一种流行方法。然而，现有的分割方法均勻处理子图像，导致对图像的理解效果不佳。在本文中，我们揭示了与整个图像具有更高语义相关性的子图像更能保留模型的视觉理解能力，蕴含了更丰富的视觉信息。因此，我们提出了全局语义指导加权分配器（GSWA）模块，该模块根据子图像的相对信息密度动态分配权重，模拟人类视觉注意力机制。这种方法使模型能够聚焦于更具信息性的区域，克服了均勻处理的局限性。我们将GSWA集成到InternVL2-2B框架中，创建了SleighVL，这是一个轻量级但性能优异的模型。大量的实验表明，SleighVL 在参数相近的模型中表现出色，并且与大型模型相比仍具有竞争力。我们的工作为LVLMs 更高效且具有上下文感知能力的高分辨率图像处理提供了有前景的方向，推进了多模态系统的发展。 

---
# Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation 

**Title (ZH)**: 利用在线奥林匹克水平数学问题进行大规模语言模型训练和抗污染评估 

**Authors**: Sadegh Mahdavi, Muchen Li, Kaiwen Liu, Christos Thrampoulidis, Leonid Sigal, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14275)  

**Abstract**: Advances in Large Language Models (LLMs) have sparked interest in their ability to solve Olympiad-level math problems. However, the training and evaluation of these models are constrained by the limited size and quality of available datasets, as creating large-scale data for such advanced problems requires extensive effort from human experts. In addition, current benchmarks are prone to contamination, leading to unreliable evaluations. In this paper, we present an automated pipeline that leverages the rich resources of the Art of Problem Solving (AoPS) forum, which predominantly features Olympiad-level problems and community-driven solutions. Using open-source LLMs, we develop a method to extract question-answer pairs from the forum, resulting in AoPS-Instruct, a dataset of more than 600,000 high-quality QA pairs. Our experiments demonstrate that fine-tuning LLMs on AoPS-Instruct improves their reasoning abilities across various benchmarks. Moreover, we build an automatic pipeline that introduces LiveAoPSBench, an evolving evaluation set with timestamps, derived from the latest forum data, providing a contamination-resistant benchmark for assessing LLM performance. Notably, we observe a significant decline in LLM performance over time, suggesting their success on older examples may stem from pre-training exposure rather than true reasoning ability. Our work presents a scalable approach to creating and maintaining large-scale, high-quality datasets for advanced math reasoning, offering valuable insights into the capabilities and limitations of LLMs in this domain. Our benchmark and code is available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）的进步激发了对其解决奥林匹克级别数学问题能力的兴趣。然而，这些模型的训练和评估受到可用数据集规模有限和质量欠佳的限制，因为为这类高级问题创建大规模数据需要大量的人工专家努力。此外，当前的基准容易受到污染，导致评估结果不可靠。在本文中，我们提出了一个自动化的管道，该管道利用Art of Problem Solving（AoPS）论坛丰富的资源，该论坛主要包含奥林匹克级别的问题和社区驱动的解决方案。通过使用开源LLMs，我们开发了一种从论坛中提取问题-答案对的方法，从而形成了包含超过600,000个高质量问题-答案对的数据集AoPS-Instruct。我们的实验表明，在AoPS-Instruct上微调LLMs可以在不同的基准上提高它们的推理能力。此外，我们构建了一个自动化的管道来引入LiveAoPSBench，这是一个具有时间戳的不断演化的评估集，由最新的论坛数据提取而来，提供了一个抵御污染的基准，用于评估LLMs的表现。值得注意的是，我们观察到LLMs的表现随时间显著下降，这表明它们在处理较早示例时的成功可能来自于预训练的暴露，而不是真实的推理能力。我们的工作提供了一种可扩展的方法来创建和维护大规模高质量的高级数学推理数据集，并提供了有关LLMs在此领域的能力和局限性的有价值见解。我们的基准和代码可以通过以下链接访问：[此 https URL](https://www.example.com)。 

---
# Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation 

**Title (ZH)**: 层次化时间感知专家混合模型在多模态序列推荐中的应用 

**Authors**: Shengzhe Zhang, Liyi Chen, Dazhong Shen, Chao Wang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.14269)  

**Abstract**: Multi-modal sequential recommendation (SR) leverages multi-modal data to learn more comprehensive item features and user preferences than traditional SR methods, which has become a critical topic in both academia and industry. Existing methods typically focus on enhancing multi-modal information utility through adaptive modality fusion to capture the evolving of user preference from user-item interaction sequences. However, most of them overlook the interference caused by redundant interest-irrelevant information contained in rich multi-modal data. Additionally, they primarily rely on implicit temporal information based solely on chronological ordering, neglecting explicit temporal signals that could more effectively represent dynamic user interest over time. To address these limitations, we propose a Hierarchical time-aware Mixture of experts for multi-modal Sequential Recommendation (HM4SR) with a two-level Mixture of Experts (MoE) and a multi-task learning strategy. Specifically, the first MoE, named Interactive MoE, extracts essential user interest-related information from the multi-modal data of each item. Then, the second MoE, termed Temporal MoE, captures user dynamic interests by introducing explicit temporal embeddings from timestamps in modality encoding. To further address data sparsity, we propose three auxiliary supervision tasks: sequence-level category prediction (CP) for item feature understanding, contrastive learning on ID (IDCL) to align sequence context with user interests, and placeholder contrastive learning (PCL) to integrate temporal information with modalities for dynamic interest modeling. Extensive experiments on four public datasets verify the effectiveness of HM4SR compared to several state-of-the-art approaches. 

**Abstract (ZH)**: 多模态序列推荐（SR）利用多模态数据来学习比传统SR方法更为全面的项目特征和用户偏好，已经成为学术界和工业界的热点话题。现有方法通常侧重于通过自适应模态融合来增强多模态信息的实用性，以捕捉用户偏好在用户-项目交互序列中的演变。然而，大多数方法忽略了丰富多模态数据中包含的冗余兴趣无关信息所造成的影响。此外，它们主要依赖于基于时间顺序的隐式时间信息，忽视了可以更有效地代表用户随时间动态兴趣的显式时间信号。为解决这些局限性，我们提出了一种基于两层混合专家（MoE）和多任务学习策略的层级时间感知混合专家多模态序列推荐（HM4SR）。具体来说，第一层MoE，称为交互MoE，从每个项目的多模态数据中提取关键的用户兴趣相关信息。然后，第二层MoE，称为时间MoE，通过引入时间戳在模态编码中的显式时间嵌入来捕获用户动态兴趣。为了进一步解决数据稀疏性问题，我们提出了三项辅助监督任务：序列级类别预测（CP）以理解项目特征，基于ID的对比学习（IDCL）以使序列上下文与用户兴趣对齐，以及占位符对比学习（PCL）以将时间信息与模态结合，从而建模动态兴趣。在四个公开数据集上的广泛实验表明，与几种最先进的方法相比，HM4SR的有效性得到了验证。 

---
# Pre-train and Fine-tune: Recommenders as Large Models 

**Title (ZH)**: 预训练与微调：推荐系统作为大规模模型 

**Authors**: Zhenhao Jiang, Chenghao Chen, Hao Feng, Yu Yang, Jin Liu, Jie Zhang, Jia Jia, Ning Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14268)  

**Abstract**: In reality, users have different interests in different periods, regions, scenes, etc. Such changes in interest are so drastic that they are difficult to be captured by recommenders. Existing multi-domain learning can alleviate this problem. However, the structure of the industrial recommendation system is complex, the amount of data is huge, and the training cost is extremely high, so it is difficult to modify the structure of the industrial recommender and re-train it. To fill this gap, we consider recommenders as large pre-trained models and fine-tune them. We first propose the theory of the information bottleneck for fine-tuning and present an explanation for the fine-tuning technique in recommenders. To tailor for recommendation, we design an information-aware adaptive kernel (IAK) technique to fine-tune the pre-trained recommender. Specifically, we define fine-tuning as two phases: knowledge compression and knowledge matching and let the training stage of IAK explicitly approximate these two phases. Our proposed approach designed from the essence of fine-tuning is well interpretable. Extensive online and offline experiments show the superiority of our proposed method. Besides, we also share unique and important lessons we learned when deploying the method in a large-scale online platform. We also present the potential issues of fine-tuning techniques in recommendation systems and the corresponding solutions. The recommender with IAK technique has been deployed on the homepage of a billion-scale online food platform for several months and has yielded considerable profits in our business. 

**Abstract (ZH)**: 实际上，用户的兴趣在不同的时期、地区和场景中会有所不同。这些兴趣的变化非常剧烈，现有的推荐系统难以捕捉到。现有的多域学习可以在一定程度上缓解这一问题。然而，工业推荐系统的结构复杂，数据量巨大，训练成本极高，因此很难修改推荐系统的结构并重新训练。为了解决这一问题，我们将推荐系统视为大型预训练模型，并对其进行微调。我们首先提出了信息瓶颈的微调理论，并解释了该技术在推荐系统中的应用。为了适应推荐需求，我们设计了一种信息感知自适应核（IAK）技术，用于预训练推荐系统的微调。具体来说，我们将微调分为两个阶段：知识压缩和知识匹配，并让IAK的训练阶段明确逼近这两个阶段。我们提出的方法从根本上体现了微调的本质，具有很强的可解释性。大量的在线和离线实验表明，我们提出的方法具有优越性。此外，我们还分享了在大规模在线平台部署该方法时学到的独特且重要的经验教训。我们还讨论了推荐系统中微调技术的潜在问题及其相应的解决方案。应用IAK技术的推荐系统已经在一家亿级在线食品平台的首页部署了几个月，为我们的业务带来了显著的利润。 

---
# Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors 

**Title (ZH)**: Siren：一种基于学习的多轮攻击框架，用于模拟真实世界的真人突破行为 

**Authors**: Yi Zhao, Youzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14250)  

**Abstract**: Large language models (LLMs) are widely used in real-world applications, raising concerns about their safety and trustworthiness. While red-teaming with jailbreak prompts exposes the vulnerabilities of LLMs, current efforts focus primarily on single-turn attacks, overlooking the multi-turn strategies used by real-world adversaries. Existing multi-turn methods rely on static patterns or predefined logical chains, failing to account for the dynamic strategies during attacks. We propose Siren, a learning-based multi-turn attack framework designed to simulate real-world human jailbreak behaviors. Siren consists of three stages: (1) training set construction utilizing Turn-Level LLM feedback (Turn-MF), (2) post-training attackers with supervised fine-tuning (SFT) and direct preference optimization (DPO), and (3) interactions between the attacking and target LLMs. Experiments demonstrate that Siren achieves an attack success rate (ASR) of 90% with LLaMA-3-8B as the attacker against Gemini-1.5-Pro as the target model, and 70% with Mistral-7B against GPT-4o, significantly outperforming single-turn baselines. Moreover, Siren with a 7B-scale model achieves performance comparable to a multi-turn baseline that leverages GPT-4o as the attacker, while requiring fewer turns and employing decomposition strategies that are better semantically aligned with attack goals. We hope Siren inspires the development of stronger defenses against advanced multi-turn jailbreak attacks under realistic scenarios. Code is available at this https URL. Warning: This paper contains potentially harmful text. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在现实应用中被广泛使用，引发了对其安全性和可信度的关注。尽管通过狱卒提示（jailbreak prompts）进行红队测试可以揭示LLMs的漏洞，但当前主要关注单轮攻击，忽视了现实世界对手可能使用的多轮策略。现有的多轮方法依赖静态模式或预定义的逻辑链路，未能考虑到攻击过程中的动态策略。我们提出Siren，这是一种基于学习的多轮攻击框架，旨在模拟现实世界中的人类狱卒行为。Siren 包括三个阶段：（1）利用回合级LLM反馈（Turn-Level LLM Feedback, Turn-MF）构建训练集，（2）使用监督微调（Supervised Fine-Tuning, SFT）和直接偏好优化（Direct Policy Optimization, DPO）进行后训练攻击者，以及（3）攻击者和目标LLM之间的交互。实验结果显示，使用Siren，当攻击者为LLaMA-3-8B，目标模型为Gemini-1.5-Pro时，攻击成功率（Attack Success Rate, ASR）达到90%；当攻击者为Mistral-7B，目标模型为GPT-4o时，攻击成功率达到70%，显著优于单轮基准模型。此外，Siren使用7B规模的模型时，性能与使用GPT-4o作为攻击者的多轮基准模型相当，且所需的攻击回合较少，并且采用了更符合攻击目标的语义分解策略。我们希望Siren能够启发在实际场景中对抗高级多轮狱卒攻击的更强防御机制的发展。代码可以在此处访问：[此链接]。请注意：本文包含可能具有危害性的文本。 

---
# Humanity's Last Exam 

**Title (ZH)**: 人类的最后一场考试 

**Authors**: Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Daron Anderson, Tung Nguyen, Mobeen Mahmood, Fiona Feng, Steven Y. Feng, Haoran Zhao, Michael Yu, Varun Gangal, Chelsea Zou, Zihan Wang, Jessica P. Wang, Pawan Kumar, Oleksandr Pokutnyi, Robert Gerbicz, Serguei Popov, John-Clark Levin, Mstyslav Kazakov, Johannes Schmitt, Geoff Galgon, Alvaro Sanchez, Yongki Lee, Will Yeadon, Scott Sauers, Marc Roth, Chidozie Agu, Søren Riis, Fabian Giska, Saiteja Utpala, Zachary Giboney, Gashaw M. Goshu, Joan of Arc Xavier, Sarah-Jane Crowson, Mohinder Maheshbhai Naiya, Noah Burns, Lennart Finke, Zerui Cheng, Hyunwoo Park, Francesco Fournier-Facio, John Wydallis, Mark Nandor, Ankit Singh, Tim Gehrunger, Jiaqi Cai, Ben McCarty, Darling Duclosel, Jungbae Nam, Jennifer Zampese, Ryan G. Hoerr, Aras Bacho, Gautier Abou Loume, Abdallah Galal, Hangrui Cao, Alexis C Garretson, Damien Sileo, Qiuyu Ren, Doru Cojoc, Pavel Arkhipov, Usman Qazi, Lianghui Li, Sumeet Motwani, Christian Schroeder de Witt, Edwin Taylor, Johannes Veith, Eric Singer, Taylor D. Hartman, Paolo Rissone, Jaehyeok Jin, Jack Wei Lun Shi, Chris G. Willcocks, Joshua Robinson, Aleksandar Mikov, Ameya Prabhu, Longke Tang, Xavier Alapont, Justine Leon Uro, Kevin Zhou, Emily de Oliveira Santos, Andrey Pupasov Maksimov, Edward Vendrow, Kengo Zenitani, Julien Guillod, Yuqi Li, Joshua Vendrow, Vladyslav Kuchkin, Ng Ze-An  

**Link**: [PDF](https://arxiv.org/pdf/2501.14249)  

**Abstract**: Benchmarks are important tools for tracking the rapid advancements in large language model (LLM) capabilities. However, benchmarks are not keeping pace in difficulty: LLMs now achieve over 90\% accuracy on popular benchmarks like MMLU, limiting informed measurement of state-of-the-art LLM capabilities. In response, we introduce Humanity's Last Exam (HLE), a multi-modal benchmark at the frontier of human knowledge, designed to be the final closed-ended academic benchmark of its kind with broad subject coverage. HLE consists of 3,000 questions across dozens of subjects, including mathematics, humanities, and the natural sciences. HLE is developed globally by subject-matter experts and consists of multiple-choice and short-answer questions suitable for automated grading. Each question has a known solution that is unambiguous and easily verifiable, but cannot be quickly answered via internet retrieval. State-of-the-art LLMs demonstrate low accuracy and calibration on HLE, highlighting a significant gap between current LLM capabilities and the expert human frontier on closed-ended academic questions. To inform research and policymaking upon a clear understanding of model capabilities, we publicly release HLE at this https URL. 

**Abstract (ZH)**: 基准是跟踪大规模语言模型（LLM）能力迅速进步的重要工具。然而，基准在难度上并未跟上步伐：当前的LLM在像MMLU这样的流行基准测试中已经达到了90%以上的准确率，这限制了对最先进LLM能力的客观衡量。为此，我们提出了人类最终考试（HLE），这是一个探索人类知识前沿的多模态基准，旨在成为此类闭合式学术基准的最后一种，具备广泛的主题覆盖。HLE包含3000道题目，涵盖了数学、人文科学和自然科学等多个学科领域。HLE由全球范围内的学科专家开发，其中包括多项选择题和简答题，适合自动评分。每个问题都有一个明确且易于验证的答案，但这些答案不能通过互联网检索快速获得。当前最先进的LLM在HLE上的准确率和校准度较低，这突显了现有LLM能力与闭合式学术问题上专家人类水平之间的显著差距。为了在明确理解模型能力的基础上指导研究和政策制定，我们在此公开发布了HLE，网址为：[此处填写URL]。 

---
# Point-LN: A Lightweight Framework for Efficient Point Cloud Classification Using Non-Parametric Positional Encoding 

**Title (ZH)**: 点ೀН：一种高效的点云分类轻量级框架，采用非参数位置编码 

**Authors**: Marzieh Mohammadi, Amir Salarpour, Pedram MohajerAnsari  

**Link**: [PDF](https://arxiv.org/pdf/2501.14238)  

**Abstract**: We introduce Point-LN, a novel lightweight framework engineered for efficient 3D point cloud classification. Point-LN integrates essential non-parametric components-such as Farthest Point Sampling (FPS), k-Nearest Neighbors (k-NN), and non-learnable positional encoding-with a streamlined learnable classifier that significantly enhances classification accuracy while maintaining a minimal parameter footprint. This hybrid architecture ensures low computational costs and rapid inference speeds, making Point-LN ideal for real-time and resource-constrained applications. Comprehensive evaluations on benchmark datasets, including ModelNet40 and ScanObjectNN, demonstrate that Point-LN achieves competitive performance compared to state-of-the-art methods, all while offering exceptional efficiency. These results establish Point-LN as a robust and scalable solution for diverse point cloud classification tasks, highlighting its potential for widespread adoption in various computer vision applications. 

**Abstract (ZH)**: 我们将介绍一种新型轻量级框架——Point-LN，该框架专为高效的3D点云分类而设计。Point-LN 结合了 Farthest Point Sampling (FPS)、k-Nearest Neighbors (k-NN) 和非学习可调位置编码等核心非参数组件，并集成了一个精简的学习分类器，显著提高了分类准确性，同时保持了参数量的最小化。这种混合架构确保了低计算成本和快速的推理速度，使 Point-LN 适用于实时和资源受限的应用。在包括 ModelNet40 和 ScanObjectNN 在内的基准数据集上的全面评估表明，Point-LN 在性能上与最先进的方法相当，同时具有出色的有效性。这些结果将 Point-LN 建立为一种适用于各种点云分类任务的稳健且可扩展的解决方案，突显了其在各种计算机视觉应用中的广泛应用潜力。 

---
# Detection and Classification of Acute Lymphoblastic Leukemia Utilizing Deep Transfer Learning 

**Title (ZH)**: 利用深度迁移学习进行急性淋巴细胞白血病的检测与分类 

**Authors**: Md. Abu Ahnaf Mollick, Md. Mahfujur Rahman, D.M. Asadujjaman, Abdullah Tamim, Nosin Anjum Dristi, Md. Takbir Hossen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14228)  

**Abstract**: A mutation in the DNA of a single cell that compromises its function initiates leukemia,leading to the overproduction of immature white blood cells that encroach upon the space required for the generation of healthy blood this http URL is treatable if identified in its initial stages. However,its diagnosis is both arduous and time consuming. This study proposes a novel approach for diagnosing leukemia across four stages Benign,Early,Pre,and Pro using deep learning this http URL employed two Convolutional Neural Network (CNN) models as MobileNetV2 with an altered head and a custom model. The custom model consists of multiple convolutional layers,each paired with corresponding max pooling this http URL utilized MobileNetV2 with ImageNet weights,adjusting the head to integrate the final this http URL dataset used is the publicly available "Acute Lymphoblastic Leukemia (ALL) Image Dataset", and we applied the Synthetic Minority Oversampling Technique (SMOTE) to augment and balance the training this http URL custom model achieved an accuracy of 98.6%, while MobileNetV2 attained a superior accuracy of 99.69%. The pretrained model showed promising results,indicating an increased likelihood of real-world application. 

**Abstract (ZH)**: 一个单细胞中的DNA突变，如果损害了其功能，将引发白血病，导致未成熟白血球的过度产生并侵占生成健康血液所需的空间。早期识别可以治疗白血病，但其诊断过程既困难又耗时。本研究提出了一种新的方法，用于诊断四个阶段的白血病（良性、早期、前白血病和白血病），采用深度学习技术。该方法采用了两个卷积神经网络（CNN）模型，分别是经过修改头部的MobileNetV2和一个自定义模型。自定义模型包含多个卷积层，每层都配对相应的最大池化。我们使用了带有ImageNet权重的MobileNetV2，并调整了其头部以集成最终的特征。所使用的数据集是公开可用的“急性淋巴细胞白血病（ALL）图像数据集”，我们通过合成少数过采样技术（SMOTE）来增强和平衡训练数据。自定义模型的准确率为98.6%，而MobileNetV2的准确率为99.69%。预训练模型表现出良好的结果，表明其在实际应用中的潜在可能性。

为了更符合学术规范，可以对上述内容进行进一步优化：

一个单细胞中的DNA突变，损害其功能，将引发白血病，导致未成熟白血球的过度产生并侵占生成健康血液所需的空间。这种情形若能早期识别，是可治疗的，但诊断过程却复杂且耗时。本研究提出了一种新的诊断方法，用于识别白血病的四个阶段（良性、早期、前白血病和白血病），采用深度学习技术。我们选择了两个卷积神经网络（CNN）模型，分别是经过修改头部的MobileNetV2和一个自定义模型。自定义模型包含多个卷积层，每层后面配对相应的最大池化。我们利用了带有ImageNet预训练权重的MobileNetV2，并调整其头部以集成最终特征。所使用的数据集是公开的“急性淋巴细胞白血病（ALL）图像数据集”，同时我们采用了合成少数过采样技术（SMOTE）来增强和平衡训练数据。实验结果显示，自定义模型的准确率为98.6%，而MobileNetV2的准确率为99.69%，预训练模型的性能也表现出色，显示出其在实际应用中的潜力。 

---
# Multi-agent KTO: Reinforcing Strategic Interactions of Large Language Model in Language Game 

**Title (ZH)**: 多代理KTO：增强大型语言模型在语言博弈中的战略互动 

**Authors**: Rong Ye, Yongxin Zhang, Yikai Zhang, Haoyu Kuang, Zhongyu Wei, Peng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.14225)  

**Abstract**: Achieving Artificial General Intelligence (AGI) requires AI agents that can not only make stratigic decisions but also engage in flexible and meaningful communication. Inspired by Wittgenstein's language game theory in Philosophical Investigations, we propose that language agents can learn through in-context interaction rather than traditional multi-stage frameworks that separate decision-making from language expression. Using Werewolf, a social deduction game that tests language understanding, strategic interaction, and adaptability, we develop the Multi-agent Kahneman & Tversky's Optimization (MaKTO). MaKTO engages diverse models in extensive gameplay to generate unpaired desirable and unacceptable responses, then employs KTO to refine the model's decision-making process. In 9-player Werewolf games, MaKTO achieves a 61% average win rate across various models, outperforming GPT-4o and two-stage RL agents by relative improvements of 23.0% and 10.9%, respectively. Notably, MaKTO also demonstrates human-like performance, winning 60% against expert players and showing only 49% detectability in Turing-style blind tests. These results showcase MaKTO's superior decision-making, strategic adaptation, and natural language generation in complex social deduction games. 

**Abstract (ZH)**: 实现通用人工智能（AGI）需要能够不仅做出战略决策，还能进行灵活和有意义沟通的AI代理。受《哲学研究》中维特根斯坦语言游戏理论的启发，我们提出，语言代理可以通过上下文中的互动来学习，而不是通过将决策过程与语言表达分开的传统多阶段框架。使用狼人杀这一社会推理游戏来测试语言理解、策略互动和适应性，我们开发了多代理 Kahneman & Tversky's 最优化（MaKTO）。MaKTO 让多种模型参与广泛的棋盘游戏，生成令人满意的和不令人满意的响应，然后利用 KTO 优化模型的决策过程。在9人版狼人杀游戏中，MaKTO 在各种模型中实现了61%的平均胜率，分别优于GPT-4o和两阶段的强化学习代理23.0%和10.9%。值得注意的是，MaKTO 还展示了类似人类的表现，以60%的胜率击败了专家玩家，并在图灵式盲测中仅显示出49%的可辨识性。这些结果展示了MaKTO在复杂社会推理游戏中卓越的决策能力、战略适应能力和自然语言生成能力。 

---
# TFG-Flow: Training-free Guidance in Multimodal Generative Flow 

**Title (ZH)**: TFG-Flow：无需训练的多模态生成流中的指导方法 

**Authors**: Haowei Lin, Shanda Li, Haotian Ye, Yiming Yang, Stefano Ermon, Yitao Liang, Jianzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.14216)  

**Abstract**: Given an unconditional generative model and a predictor for a target property (e.g., a classifier), the goal of training-free guidance is to generate samples with desirable target properties without additional training. As a highly efficient technique for steering generative models toward flexible outcomes, training-free guidance has gained increasing attention in diffusion models. However, existing methods only handle data in continuous spaces, while many scientific applications involve both continuous and discrete data (referred to as multimodality). Another emerging trend is the growing use of the simple and general flow matching framework in building generative foundation models, where guided generation remains under-explored. To address this, we introduce TFG-Flow, a novel training-free guidance method for multimodal generative flow. TFG-Flow addresses the curse-of-dimensionality while maintaining the property of unbiased sampling in guiding discrete variables. We validate TFG-Flow on four molecular design tasks and show that TFG-Flow has great potential in drug design by generating molecules with desired properties. 

**Abstract (ZH)**: 以下内容为论文的标题或摘要，并已翻译成中文，符合学术规范：

在给定无条件生成模型和目标属性的预测器（例如，分类器）的情况下，训练免费指导的目标是在不进行额外训练的情况下生成具有期望目标属性的样本。作为一种高效的技术，用于引导生成模型产生灵活的结果，训练免费指导在扩散模型中引起了越来越多的关注。然而，现有的方法仅处理连续空间中的数据，而许多科学应用涉及连续和离散数据（统称为多模态）。另一个新兴趋势是在构建生成基础模型时越来越多地使用简单且通用的流匹配框架，而引导生成仍是一个未充分探索的领域。为了解决这一问题，我们提出了TFG-Flow，这是一种用于多模态生成流的新型训练免费指导方法。TFG-Flow 在引导离散变量的同时解决了高维灾难的问题，并保持了无偏采样的性质。我们在四项分子设计任务上验证了TFG-Flow，并展示了TFG-Flow 在通过生成具有期望属性的分子来进行药物设计方面的巨大潜力。 

---
# PuzzleGPT: Emulating Human Puzzle-Solving Ability for Time and Location Prediction 

**Title (ZH)**: PuzzleGPT：模拟人类解谜能力以预测时间和地点 

**Authors**: Hammad Ayyubi, Xuande Feng, Junzhang Liu, Xudong Lin, Zhecan Wang, Shih-Fu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14210)  

**Abstract**: The task of predicting time and location from images is challenging and requires complex human-like puzzle-solving ability over different clues. In this work, we formalize this ability into core skills and implement them using different modules in an expert pipeline called PuzzleGPT. PuzzleGPT consists of a perceiver to identify visual clues, a reasoner to deduce prediction candidates, a combiner to combinatorially combine information from different clues, a web retriever to get external knowledge if the task can't be solved locally, and a noise filter for robustness. This results in a zero-shot, interpretable, and robust approach that records state-of-the-art performance on two datasets -- TARA and WikiTilo. PuzzleGPT outperforms large VLMs such as BLIP-2, InstructBLIP, LLaVA, and even GPT-4V, as well as automatically generated reasoning pipelines like VisProg, by at least 32% and 38%, respectively. It even rivals or surpasses finetuned models. 

**Abstract (ZH)**: 从图像中预测时间和地点的任务具有挑战性，要求具备在不同线索上进行复杂的人类级拼图解谜能力。在本研究中，我们将这种能力形式化为基本技能，并通过一个称为PuzzleGPT的专家流水线中的不同模块来实现这些技能。PuzzleGPT包括一个感知器来识别视觉线索、一个推理器来推断预测候选对象、一个组合器来组合来自不同线索的信息、一个网络检索器在任务无法本地解决时获取外部知识，以及一个噪声过滤器以增强鲁棒性。这导致了一种零样本、可解释且鲁棒的方法，在两个数据集TARA和WikiTilo上取得了最先进的性能。PuzzleGPT在与BLIP-2、InstructBLIP、LLaVA以及GPT-4V等大型视觉语言模型的比较中，以及与自动生成的推理流水线如VisProg的比较中，分别在至少32%和38%的指标上表现出色，甚至与微调模型相媲美或超越。 

---
# Dynamic Token Reduction during Generation for Vision Language Models 

**Title (ZH)**: 生成过程中用于视觉语言模型的动态令牌减少 

**Authors**: Xiaoyu Liang, Chaofeng Guan, Jiaying Lu, Huiyao Chen, Huan Wang, Haoji Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14204)  

**Abstract**: Vision-Language Models (VLMs) have achieved notable success in multimodal tasks but face practical limitations due to the quadratic complexity of decoder attention mechanisms and autoregressive generation. Existing methods like FASTV and VTW have achieved notable results in reducing redundant visual tokens, but these approaches focus on pruning tokens in a single forward pass without systematically analyzing the redundancy of visual tokens throughout the entire generation process. In this paper, we introduce a dynamic pruning strategy tailored for VLMs, namedDynamic Rate (DyRate), which progressively adjusts the compression rate during generation. Our analysis of the distribution of attention reveals that the importance of visual tokens decreases throughout the generation process, inspiring us to adopt a more aggressive compression rate. By integrating a lightweight predictor based on attention distribution, our approach enables flexible adjustment of pruning rates based on the attention distribution. Our experimental results demonstrate that our method not only reduces computational demands but also maintains the quality of responses. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在多模态任务中取得了显著的成功，但由于解码器注意机制和自回归生成的二次复杂性，它们面临实际应用中的限制。现有的方法如FASTV和VTW在减少冗余视觉标记方面取得了显著成果，但这些方法主要关注在一前向传递中剪枝标记，而不系统地分析整个生成过程中视觉标记的冗余性。在本文中，我们提出了一种专门针对VLMs的动态剪枝策略，命名为DyRate，该策略在生成过程中逐步调整压缩率。通过对注意分布的分析，我们发现生成过程中视觉标记的重要性逐渐降低，这启发我们采取更激进的压缩率。通过结合基于注意分布的轻量级预测器，我们的方法能够根据注意分布灵活调整剪枝率。我们的实验结果表明，我们的方法不仅减少了计算需求，还保持了响应的质量。 

---
# Coordinating Ride-Pooling with Public Transit using Reward-Guided Conservative Q-Learning: An Offline Training and Online Fine-Tuning Reinforcement Learning Framework 

**Title (ZH)**: 使用奖励引导保守Q学习协调拼车与公共交通：一个离线训练与在线微调的强化学习框架 

**Authors**: Yulong Hu, Tingting Dong, Sen Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14199)  

**Abstract**: This paper introduces a novel reinforcement learning (RL) framework, termed Reward-Guided Conservative Q-learning (RG-CQL), to enhance coordination between ride-pooling and public transit within a multimodal transportation network. We model each ride-pooling vehicle as an agent governed by a Markov Decision Process (MDP) and propose an offline training and online fine-tuning RL framework to learn the optimal operational decisions of the multimodal transportation systems, including rider-vehicle matching, selection of drop-off locations for passengers, and vehicle routing decisions, with improved data efficiency. During the offline training phase, we develop a Conservative Double Deep Q Network (CDDQN) as the action executor and a supervised learning-based reward estimator, termed the Guider Network, to extract valuable insights into action-reward relationships from data batches. In the online fine-tuning phase, the Guider Network serves as an exploration guide, aiding CDDQN in effectively and conservatively exploring unknown state-action pairs. The efficacy of our algorithm is demonstrated through a realistic case study using real-world data from Manhattan. We show that integrating ride-pooling with public transit outperforms two benchmark cases solo rides coordinated with transit and ride-pooling without transit coordination by 17% and 22% in the achieved system rewards, respectively. Furthermore, our innovative offline training and online fine-tuning framework offers a remarkable 81.3% improvement in data efficiency compared to traditional online RL methods with adequate exploration budgets, with a 4.3% increase in total rewards and a 5.6% reduction in overestimation errors. Experimental results further demonstrate that RG-CQL effectively addresses the challenges of transitioning from offline to online RL in large-scale ride-pooling systems integrated with transit. 

**Abstract (ZH)**: 本文介绍了一种新颖的强化学习（RL）框架，称为奖励引导保守Q学习（RG-CQL），以增强乘车共享与公共交通在多模态交通网络中的协调能力。我们将每辆乘车共享车辆视为由马尔可夫决策过程（MDP）驱动的智能体，并提出了一种离线训练和在线精细调整的RL框架，以提高多模态交通系统的最优运营决策，包括乘客与车辆配对、乘客下车地点选择以及车辆路径规划决策，同时提高了数据效率。在离线训练阶段，我们开发了一种保守的双重深度Q网络（CDDQN）作为动作执行器，并提出了一种基于监督学习的奖励估计器，即引导网络，从数据批次中提取行动-奖励关系中的有价值的见解。在在线精细调整阶段，引导网络充当探索指南，帮助CDDQN有效地、保守地探索未知的状态-行动对。通过使用来自曼哈顿的真实世界数据进行的实际案例研究，证明了该算法的有效性。结果显示，将乘车共享与公共交通相结合在实现系统奖励方面分别比单一乘车与公共交通协调以及非协调的乘车共享分别提升了17%和22%。此外，我们的创新性离线训练和在线精细调整框架在具有足够探索预算的传统在线RL方法中提供了81.3%的数据效率改进，同时总奖励增长了4.3%，错误估计减少了5.6%。实验结果进一步证明，RG-CQL有效地解决了大规模整合公共交通的乘车共享系统从离线到在线RL过渡中的挑战。 

---
# ENTER: Event Based Interpretable Reasoning for VideoQA 

**Title (ZH)**: ENTER：基于事件的可解释推理方法用于视频问答 

**Authors**: Hammad Ayyubi, Junzhang Liu, Ali Asgarov, Zaber Ibn Abdul Hakim, Najibul Haque Sarker, Zhecan Wang, Chia-Wei Tang, Hani Alomari, Md. Atabuzzaman, Xudong Lin, Naveen Reddy Dyava, Shih-Fu Chang, Chris Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2501.14194)  

**Abstract**: In this paper, we present ENTER, an interpretable Video Question Answering (VideoQA) system based on event graphs. Event graphs convert videos into graphical representations, where video events form the nodes and event-event relationships (temporal/causal/hierarchical) form the edges. This structured representation offers many benefits: 1) Interpretable VideoQA via generated code that parses event-graph; 2) Incorporation of contextual visual information in the reasoning process (code generation) via event graphs; 3) Robust VideoQA via Hierarchical Iterative Update of the event graphs. Existing interpretable VideoQA systems are often top-down, disregarding low-level visual information in the reasoning plan generation, and are brittle. While bottom-up approaches produce responses from visual data, they lack interpretability. Experimental results on NExT-QA, IntentQA, and EgoSchema demonstrate that not only does our method outperform existing top-down approaches while obtaining competitive performance against bottom-up approaches, but more importantly, offers superior interpretability and explainability in the reasoning process. 

**Abstract (ZH)**: 在本文中，我们提出了一款基于事件图的可解释视频问答（VideoQA）系统——ENTER。事件图将视频转换为图形表示形式，其中视频事件作为节点，事件间的事件-事件关系（包括时间关系、因果关系和层次关系）作为边。这种结构化表示形式提供了许多优势：1）通过生成解析事件图的代码实现可解释的VideoQA；2）通过事件图融合上下文视觉信息，增强推理过程（代码生成）的解释性；3）通过分层次迭代更新事件图提高视频问答的鲁棒性。现有的可解释VideoQA系统往往从宏观层面进行推理计划生成，忽略了低层视觉信息，因此稳定性较差。而自底向上的方法虽然能够从视觉数据中生成响应，但在解释性方面有所欠缺。我们在NExT-QA、IntentQA和EgoSchema上的实验结果表明，我们的方法不仅在性能上优于现有的自顶向下方法，与自底向上的方法相比也能获得相当的竞争性能，更重要的是，它在推理过程中的可解释性和可说明性方面表现更优。 

---
# VarDrop: Enhancing Training Efficiency by Reducing Variate Redundancy in Periodic Time Series Forecasting 

**Title (ZH)**: VarDrop: 通过减少周期时间序列预测中变量冗余来提高训练效率 

**Authors**: Junhyeok Kang, Yooju Shin, Jae-Gil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.14183)  

**Abstract**: Variate tokenization, which independently embeds each variate as separate tokens, has achieved remarkable improvements in multivariate time series forecasting. However, employing self-attention with variate tokens incurs a quadratic computational cost with respect to the number of variates, thus limiting its training efficiency for large-scale applications. To address this issue, we propose VarDrop, a simple yet efficient strategy that reduces the token usage by omitting redundant variate tokens during training. VarDrop adaptively excludes redundant tokens within a given batch, thereby reducing the number of tokens used for dot-product attention while preserving essential information. Specifically, we introduce k-dominant frequency hashing (k-DFH), which utilizes the ranked dominant frequencies in the frequency domain as a hash value to efficiently group variate tokens exhibiting similar periodic behaviors. Then, only representative tokens in each group are sampled through stratified sampling. By performing sparse attention with these selected tokens, the computational cost of scaled dot-product attention is significantly alleviated. Experiments conducted on public benchmark datasets demonstrate that VarDrop outperforms existing efficient baselines. 

**Abstract (ZH)**: 变元分词（Variate Tokenization）通过将每个变元独立嵌入为单独的标记，已在多变量时间序列预测中实现了显著的改进。然而，使用自注意力机制时涉及变元标记会导致计算成本呈平方级增长，这限制了其在大规模应用中的训练效率。为解决这一问题，我们提出了一种简单且高效的方法——VarDrop。该方法在训练过程中通过省略冗余变元标记来减少标记的使用。VarDrop 通过自适应地排除给定批次内的冗余标记，减少了用于点积注意力的标记数量，同时保留了关键信息。具体而言，我们引入了 k-主导频率哈希（k-DFH），该方法利用频域中排名靠前的主导频率作为哈希值，高效地将表现出相似周期行为的变元标记进行分组。然后，通过分层抽样仅选择每个组中的代表性标记。通过仅对这些选定的标记执行稀疏注意力，可以显著缓解标度点积注意力的计算成本。在公共基准数据集上的实验表明，VarDrop 在性能上优于现有的一些高效基线方法。 

---
# RL + Transformer = A General-Purpose Problem Solver 

**Title (ZH)**: RL + Transformer = 通用问题求解器 

**Authors**: Micah Rentschler, Jesse Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2501.14176)  

**Abstract**: What if artificial intelligence could not only solve problems for which it was trained but also learn to teach itself to solve new problems (i.e., meta-learn)? In this study, we demonstrate that a pre-trained transformer fine-tuned with reinforcement learning over multiple episodes develops the ability to solve problems that it has never encountered before - an emergent ability called In-Context Reinforcement Learning (ICRL). This powerful meta-learner not only excels in solving unseen in-distribution environments with remarkable sample efficiency, but also shows strong performance in out-of-distribution environments. In addition, we show that it exhibits robustness to the quality of its training data, seamlessly stitches together behaviors from its context, and adapts to non-stationary environments. These behaviors demonstrate that an RL-trained transformer can iteratively improve upon its own solutions, making it an excellent general-purpose problem solver. 

**Abstract (ZH)**: 如果我们的人工智能不仅能解决它被训练来解决的问题，还能学会自我教学以解决新问题（即元学习），会怎样呢？在本研究中，我们展示了预训练的变换器通过多阶段强化学习微调后，能够解决它从未遇到过的新问题——这种能力被称为上下文强化学习（In-Context Reinforcement Learning, ICRL）。这种强大的元学习模型不仅在解决未见的分布内环境方面表现出非凡的样本效率，还在未见的分布外环境中也表现出较强的性能。此外，我们还证明了该模型对训练数据质量具有较强的鲁棒性，能够无缝地将上下文中的行为结合在一起，并适应非平稳环境。这些行为表明，经过强化学习训练的变换器可以逐步改进自己的解决方案，使其成为一种优秀的通用问题解决者。 

---
# Dreamweaver: Learning Compositional World Representations from Pixels 

**Title (ZH)**: 导梦者：从像素中学习组成性世界表示 

**Authors**: Junyeob Baek, Yi-Fu Wu, Gautam Singh, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2501.14174)  

**Abstract**: Humans have an innate ability to decompose their perceptions of the world into objects and their attributes, such as colors, shapes, and movement patterns. This cognitive process enables us to imagine novel futures by recombining familiar concepts. However, replicating this ability in artificial intelligence systems has proven challenging, particularly when it comes to modeling videos into compositional concepts and generating unseen, recomposed futures without relying on auxiliary data, such as text, masks, or bounding boxes. In this paper, we propose Dreamweaver, a neural architecture designed to discover hierarchical and compositional representations from raw videos and generate compositional future simulations. Our approach leverages a novel Recurrent Block-Slot Unit (RBSU) to decompose videos into their constituent objects and attributes. In addition, Dreamweaver uses a multi-future-frame prediction objective to capture disentangled representations for dynamic concepts more effectively as well as static concepts. In experiments, we demonstrate our model outperforms current state-of-the-art baselines for world modeling when evaluated under the DCI framework across multiple datasets. Furthermore, we show how the modularized concept representations of our model enable compositional imagination, allowing the generation of novel videos by recombining attributes from different objects. 

**Abstract (ZH)**: 人类天生具有将对外界世界的感知分解为对象及其属性（如颜色、形状和运动模式）的能力。这一认知过程使我们能够通过重新组合熟悉的概念来想象新的未来。然而，在人工智能系统中复制这一能力颇具挑战性，特别是在将视频分解为组成性概念，并在无须依赖辅助数据（如文本、掩码或边界框）的情况下生成未见过的重新组合未来方面。本文中，我们提出了一种名为Dreamweaver的神经架构，该架构旨在从原始视频中发现层次化和组成性的表示，并生成组成性的未来模拟。我们的方法利用了新型循环块槽单元（RBSU）来分解视频为其构成的对象和属性。此外，Dreamweaver 使用多帧未来预测目标来更有效地捕捉动态概念和静态概念的分离表示。在实验中，我们展示了在DCI框架下使用多个数据集评估时，我们的模型在世界建模方面优于当前最先进的基线模型。此外，我们证明了我们模型模块化的概念表示能力能够实现组成性的想象，从而通过从不同对象中重新组合属性来生成新颖的视频。 

---
# UltraLightSqueezeNet: A Deep Learning Architecture for Malaria Classification with up to 54x fewer trainable parameters for resource constrained devices 

**Title (ZH)**: UltraLightSqueezeNet：一种用于疟疾分类的深度学习架构，在资源受限设备上可减少最多54倍的可训练参数 

**Authors**: Suresh Babu Nettur, Shanthi Karpurapu, Unnati Nettur, Likhit Sagar Gajja, Sravanthy Myneni, Akhil Dusi, Lalithya Posham  

**Link**: [PDF](https://arxiv.org/pdf/2501.14172)  

**Abstract**: Lightweight deep learning approaches for malaria detection have gained attention for their potential to enhance diagnostics in resource constrained environments. For our study, we selected SqueezeNet1.1 as it is one of the most popular lightweight architectures. SqueezeNet1.1 is a later version of SqueezeNet1.0 and is 2.4 times more computationally efficient than the original model. We proposed and implemented three ultra-lightweight architecture variants to SqueezeNet1.1 architecture, namely Variant 1 (one fire module), Variant 2 (two fire modules), and Variant 3 (four fire modules), which are even more compact than SqueezeNetV1.1 (eight fire modules). These models were implemented to evaluate the best performing variant that achieves superior computational efficiency without sacrificing accuracy in malaria blood cell classification. The models were trained and evaluated using the NIH Malaria dataset. We assessed each model's performance based on metrics including accuracy, recall, precision, F1-score, and Area Under the Curve (AUC). The results show that the SqueezeNet1.1 model achieves the highest performance across all metrics, with a classification accuracy of 97.12%. Variant 3 (four fire modules) offers a competitive alternative, delivering almost identical results (accuracy 96.55%) with a 6x reduction in computational overhead compared to SqueezeNet1.1. Variant 2 and Variant 1 perform slightly lower than Variant 3, with Variant 2 (two fire modules) reducing computational overhead by 28x, and Variant 1 (one fire module) achieving a 54x reduction in trainable parameters compared to SqueezeNet1.1. These findings demonstrate that our SqueezeNet1.1 architecture variants provide a flexible approach to malaria detection, enabling the selection of a variant that balances resource constraints and performance. 

**Abstract (ZH)**: 在资源受限环境下，轻量级深度学习方法在疟疾检测中的应用得到了关注，因为它们有可能提高诊断性能。在本研究中，我们选择了SqueezeNet1.1作为研究对象，因为它是目前最受欢迎的轻量级网络架构之一。SqueezeNet1.1是SqueezeNet1.0的后继版本，其计算效率比原始模型提高了2.4倍。我们提出了并实现了三种针对SqueezeNet1.1架构的超轻量级架构变体，具体为：变体1（一个fire模块）、变体2（两个fire模块）和变体3（四个fire模块），这些变体比SqueezeNetV1.1（八个fire模块）更为紧凑。这些模型被实现以评估在不牺牲精度的情况下，计算效率最优的变体。我们使用NIH疟疾数据集对这些模型进行了训练和评估。我们根据准确性、召回率、精确率、F1分数和曲线下面积（AUC）等指标评估了每个模型的性能。结果显示，SqueezeNet1.1模型在所有指标上均表现出最佳性能，分类准确率为97.12%。变体3（四个fire模块）提供了一个具有竞争力的替代方案，其准确率为96.55%，计算开销减少了6倍，与SqueezeNet1.1相当。变体2和变体1的性能略低于变体3，其中变体2（两个fire模块）将计算开销减少了28倍，而变体1（一个fire模块）则将可训练参数减少了54倍。这些发现表明，我们的SqueezeNet1.1架构变体提供了一种灵活的疟疾检测方法，能够根据资源限制和性能需求选择合适的变体。 

---
# Enhancing Multimodal Entity Linking with Jaccard Distance-based Conditional Contrastive Learning and Contextual Visual Augmentation 

**Title (ZH)**: 基于Jaccard距离条件对比学习和上下文视觉增强的多模态实体链接优化 

**Authors**: Cong-Duy Nguyen, Xiaobao Wu, Thong Nguyen, Shuai Zhao, Khoi Le, Viet-Anh Nguyen, Feng Yichao, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14166)  

**Abstract**: Previous research on multimodal entity linking (MEL) has primarily employed contrastive learning as the primary objective. However, using the rest of the batch as negative samples without careful consideration, these studies risk leveraging easy features and potentially overlook essential details that make entities unique. In this work, we propose JD-CCL (Jaccard Distance-based Conditional Contrastive Learning), a novel approach designed to enhance the ability to match multimodal entity linking models. JD-CCL leverages meta-information to select negative samples with similar attributes, making the linking task more challenging and robust. Additionally, to address the limitations caused by the variations within the visual modality among mentions and entities, we introduce a novel method, CVaCPT (Contextual Visual-aid Controllable Patch Transform). It enhances visual representations by incorporating multi-view synthetic images and contextual textual representations to scale and shift patch representations. Experimental results on benchmark MEL datasets demonstrate the strong effectiveness of our approach. 

**Abstract (ZH)**: 先前关于多模态实体链接（MEL）的研究主要将对比学习作为主要目标。然而，这些研究在使用整个批次的样本作为负样本时，如果没有仔细考虑，可能会利用简单的特征，并且有可能忽略使得实体独特的关键细节。在这项工作中，我们提出了一种名为JD-CCL（Jaccard 距离基于条件对比学习）的新型方法，旨在增强多模态实体链接模型的匹配能力。JD-CCL 利用元信息来选择具有相似属性的负样本，从而使链接任务更具挑战性和鲁棒性。此外，为了应对提及和实体在视觉模态内变化造成的限制，我们引入了一种新颖的方法，即CVaCPT（基于上下文的视觉辅助可控补丁变换）。该方法通过结合多视角合成图像和上下文文本表示来扩展和变换补丁表示，从而增强视觉表示。在基准MEL数据集上的实验结果表明，我们的方法具有很强的有效性。 

---
# LoCoML: A Framework for Real-World ML Inference Pipelines 

**Title (ZH)**: LoCoML：一种面向实际应用的机器学习推理管道框架 

**Authors**: Kritin Maddireddy, Santhosh Kotekal Methukula, Chandrasekar Sridhar, Karthik Vaidhyanathan  

**Link**: [PDF](https://arxiv.org/pdf/2501.14165)  

**Abstract**: The widespread adoption of machine learning (ML) has brought forth diverse models with varying architectures, and data requirements, introducing new challenges in integrating these systems into real-world applications. Traditional solutions often struggle to manage the complexities of connecting heterogeneous models, especially when dealing with varied technical specifications. These limitations are amplified in large-scale, collaborative projects where stakeholders contribute models with different technical specifications. To address these challenges, we developed LoCoML, a low-code framework designed to simplify the integration of diverse ML models within the context of the \textit{Bhashini Project} - a large-scale initiative aimed at integrating AI-driven language technologies such as automatic speech recognition, machine translation, text-to-speech, and optical character recognition to support seamless communication across more than 20 languages. Initial evaluations show that LoCoML adds only a small amount of computational load, making it efficient and effective for large-scale ML integration. Our practical insights show that a low-code approach can be a practical solution for connecting multiple ML models in a collaborative environment. 

**Abstract (ZH)**: 机器学习（ML）的广泛应用带来了各种具有不同架构和数据需求的模型，这在将这些系统集成到实际应用中时引入了新的挑战。传统解决方案往往难以管理异构模型之间复杂多样的连接，特别是在处理不同技术规格时更是如此。在大规模协作项目中，这一挑战被进一步放大，因为不同利益相关方所贡献的模型具有不同的技术规格。为应对这些挑战，我们开发了LoCoML，这是一种低代码框架，旨在简化《Bhashini项目》中多样化的ML模型的集成。《Bhashini项目》是一个旨在整合包括自动语音识别、机器翻译、文本转语音和光学字符识别在内的AI驱动语言技术的大规模倡议，旨在支持超过20种语言的无缝沟通。初步评估表明，LoCoML仅为系统增加了少量的计算负载，使其在大规模ML集成中高效且有效。我们的实践经验表明，低代码方法可以作为一种有效的解决方案，在协作环境中连接多个ML模型。 

---
# Advancing MRI Reconstruction: A Systematic Review of Deep Learning and Compressed Sensing Integration 

**Title (ZH)**: 推进MRI重建：深度学习与压缩感知集成的系统综述 

**Authors**: Mojtaba Safari, Zach Eidex, Chih-Wei Chang, Richard L.J. Qiu, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14158)  

**Abstract**: Magnetic resonance imaging (MRI) is a non-invasive imaging modality and provides comprehensive anatomical and functional insights into the human body. However, its long acquisition times can lead to patient discomfort, motion artifacts, and limiting real-time applications. To address these challenges, strategies such as parallel imaging have been applied, which utilize multiple receiver coils to speed up the data acquisition process. Additionally, compressed sensing (CS) is a method that facilitates image reconstruction from sparse data, significantly reducing image acquisition time by minimizing the amount of data collection needed. Recently, deep learning (DL) has emerged as a powerful tool for improving MRI reconstruction. It has been integrated with parallel imaging and CS principles to achieve faster and more accurate MRI reconstructions. This review comprehensively examines DL-based techniques for MRI reconstruction. We categorize and discuss various DL-based methods, including end-to-end approaches, unrolled optimization, and federated learning, highlighting their potential benefits. Our systematic review highlights significant contributions and underscores the potential of DL in MRI reconstruction. Additionally, we summarize key results and trends in DL-based MRI reconstruction, including quantitative metrics, the dataset, acceleration factors, and the progress of and research interest in DL techniques over time. Finally, we discuss potential future directions and the importance of DL-based MRI reconstruction in advancing medical imaging. To facilitate further research in this area, we provide a GitHub repository that includes up-to-date DL-based MRI reconstruction publications and public datasets-this https URL. 

**Abstract (ZH)**: 磁共振成像（MRI）是一种无创成像技术，能全面提供人体的解剖和功能信息。然而，其长数据采集时间可能导致患者不适、运动伪影，并限制了实时应用。为应对这些挑战，已经应用了并行成像等多种策略。这些策略通过使用多个接收线圈加快数据采集过程。此外，压缩感知（CS）是一种从稀疏数据中重建图像的方法，通过减少所需采集的数据量来显著缩短图像采集时间。最近，深度学习（DL）已经作为一种强大的工具被用于提高MRI重建。它已经与并行成像和压缩感知原理相结合，实现了更快更准确的MRI重建。本文综述了基于DL的MRI重建技术。我们对各种基于DL的方法进行了分类和讨论，包括端到端方法、展开优化和联邦学习，并强调了它们的潜在优势。我们的系统综述强调了基于DL技术在MRI重建中的重要贡献，并突显了DL在MRI重建中的潜在价值。此外，我们总结了基于DL的MRI重建的关键结果和趋势，包括定量指标、数据集、加速因子以及DL技术在时间上的进展和研究兴趣。最后，我们讨论了DL技术在推进医学影像中的潜在发展方向及其重要性。为了促进该领域的进一步研究，我们提供了一个GitHub存储库，其中包括最新的基于DL的MRI重建文献和公开数据集——<这个链接请根据实际情况填写>。 

---
# Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters 

**Title (ZH)**: 基于自定义失真滤波器的对抗黑盒攻击强化学习平台 

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Ricardo Luna Gutierrez, Antonio Guillen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14122)  

**Abstract**: We present a Reinforcement Learning Platform for Adversarial Black-box untargeted and targeted attacks, RLAB, that allows users to select from various distortion filters to create adversarial examples. The platform uses a Reinforcement Learning agent to add minimum distortion to input images while still causing misclassification by the target model. The agent uses a novel dual-action method to explore the input image at each step to identify sensitive regions for adding distortions while removing noises that have less impact on the target model. This dual action leads to faster and more efficient convergence of the attack. The platform can also be used to measure the robustness of image classification models against specific distortion types. Also, retraining the model with adversarial samples significantly improved robustness when evaluated on benchmark datasets. The proposed platform outperforms state-of-the-art methods in terms of the average number of queries required to cause misclassification. This advances trustworthiness with a positive social impact. 

**Abstract (ZH)**: 我们提出了一种用于对抗性黑盒未 targeted 和 targeted 攻击的强化学习平台，RLAB，该平台允许用户从多种扭曲滤镜中选择，以创建对抗性示例。该平台利用强化学习代理在不显著增加输入图像失真的情况下，仍能使目标模型产生误分类。代理在每一步探索输入图像时采用一种新颖的双行动方法，以识别适合添加失真的敏感区域，同时去除对目标模型影响较小的噪声。这种双行动方法导致攻击更快且更有效率地收敛。该平台还可以用于测量图像分类模型对特定失真类型的鲁棒性。此外，使用对抗性样本重新训练模型在基准数据集上的表现显著提高了模型的鲁棒性。所提出的平台在引起误分类所需的平均查询次数上优于现有方法。这促进了信任度并产生积极的社会影响。 

---
# On the Transfer of Knowledge in Quantum Algorithms 

**Title (ZH)**: 量子算法中的知识迁移研究 

**Authors**: Esther Villar-Rodriguez, Eneko Osaba, Izaskun Oregi, Sebastián V. Romero, Julián Ferreiro-Vélez  

**Link**: [PDF](https://arxiv.org/pdf/2501.14120)  

**Abstract**: The field of quantum computing is generating significant anticipation within the scientific and industrial communities due to its potential to revolutionize computing paradigms. Recognizing this potential, this paper explores the integration of transfer of knowledge techniques, traditionally used in classical artificial intelligence, into quantum computing. We present a comprehensive classification of the transfer models, focusing on Transfer Learning and Transfer Optimization. Additionally, we analyze relevant schemes in quantum computing that can benefit from knowledge sharing, and we delve into the potential synergies, supported by theoretical insights and initial experimental results. Our findings suggest that leveraging the transfer of knowledge can enhance the efficiency and effectiveness of quantum algorithms, particularly in the context of hybrid solvers. This approach not only accelerates the optimization process but also reduces the computational burden on quantum processors, making it a valuable tool for advancing quantum computing technologies. 

**Abstract (ZH)**: 量子计算领域因其潜在的革新计算范式的可能性而在科学和工业界引起了广泛关注。认识到这一潜力，本文探讨了将传统用于经典人工智能的知识转移技术应用于量子计算的整合。我们对知识转移模型进行了全面分类，重点关注知识转移和知识优化。此外，我们分析了量子计算中可以从知识共享中受益的相关方案，并深入探讨了由理论见解和初步实验结果支持的潜在协同效应。研究结果表明，利用知识转移可以提高量子算法的效率和有效性，特别是在组合求解器的背景下。这种方法不仅加速了优化过程，还减少了量子处理器的计算负担，使其成为推动量子计算技术进步的重要工具。 

---
# Autonomous Structural Memory Manipulation for Large Language Models Using Hierarchical Embedding Augmentation 

**Title (ZH)**: 使用层次嵌入增强进行大规模语言模型的自主结构记忆操作 

**Authors**: Derek Yotheringhay, Alistair Kirkland, Humphrey Kirkbride, Josiah Whitesteeple  

**Link**: [PDF](https://arxiv.org/pdf/2501.14119)  

**Abstract**: Transformative innovations in model architectures have introduced hierarchical embedding augmentation as a means to redefine the representation of tokens through multi-level semantic structures, offering enhanced adaptability to complex linguistic inputs. Autonomous structural memory manipulation further advances this paradigm through dynamic memory reallocation mechanisms that prioritize critical contextual features while suppressing less relevant information, enabling scalable and efficient performance across diverse tasks. Experimental results reveal substantial improvements in computational efficiency, with marked reductions in processing overhead for longer input sequences, achieved through memory reorganization strategies that adapt to evolving contextual requirements. Hierarchical embeddings not only improved contextual alignment but also facilitated task generalization by capturing relationships at varying semantic granularities, ensuring coherence across layers without introducing significant computational redundancies. Comparative analysis against baseline models demonstrated unique advantages in accuracy, efficiency, and interpretability, particularly in tasks requiring complex contextual understanding or domain-specific adaptability. The ability to dynamically adjust token representations and memory configurations contributed to the model's robustness under varied and unpredictable input conditions. Applications benefiting from these advancements include multi-domain generalization, interactive systems, and scenarios involving real-time decision-making, where traditional static memory architectures often face limitations. The proposed methodology combines advanced embedding and memory management strategies into a cohesive framework that addresses scalability challenges while preserving task-specific relevance. 

**Abstract (ZH)**: 模型架构方面的变革性创新引入了层次嵌入增强的方法，通过多级语义结构重新定义标记的表示方式，从而提高了对复杂语义输入的适应性。自主结构记忆操作进一步推动了这一范式的演进，通过动态内存再分配机制优先选择关键上下文特征并抑制不相关的信息，从而在多种任务中实现可扩展性和高效性。实验结果表明，在计算效率方面取得了显著提高，尤其是在处理较长输入序列时，通过适应不断变化上下文需求的内存重组策略，实现了显著的处理开销减少。层次嵌入不仅改善了上下文对齐，还通过捕捉不同语义粒度的关系，促进了任务泛化，确保了层间的连贯性，同时避免了显著的计算冗余。与基础模型的比较分析显示，该方法在准确率、效率和可解释性方面具有独特优势，特别是在需要复杂上下文理解或领域特定适应的任务中尤为突出。动态调整标记表示和内存配置的能力使该模型在多种不确定和不可预测的输入条件下具有鲁棒性。这些进步的应用领域包括多领域泛化、交互系统以及涉及实时决策的情景，而在传统静态内存架构中，这些领域通常面临限制。所提出的方法将先进的嵌入和内存管理策略结合起来，形成一个统一的框架，既解决了可扩展性挑战，又保留了任务特定的相关性。 

---
# MedSlice: Fine-Tuned Large Language Models for Secure Clinical Note Sectioning 

**Title (ZH)**: MedSlice: 细化调优的大语言模型在安全临床笔记分区中的应用 

**Authors**: Joshua Davis, Thomas Sounack, Kate Sciacca, Jessie M Brain, Brigitte N Durieux, Nicole D Agaronnik, Charlotta Lindvall  

**Link**: [PDF](https://arxiv.org/pdf/2501.14105)  

**Abstract**: Extracting sections from clinical notes is crucial for downstream analysis but is challenging due to variability in formatting and labor-intensive nature of manual sectioning. While proprietary large language models (LLMs) have shown promise, privacy concerns limit their accessibility. This study develops a pipeline for automated note sectioning using open-source LLMs, focusing on three sections: History of Present Illness, Interval History, and Assessment and Plan. We fine-tuned three open-source LLMs to extract sections using a curated dataset of 487 progress notes, comparing results relative to proprietary models (GPT-4o, GPT-4o mini). Internal and external validity were assessed via precision, recall and F1 score. Fine-tuned Llama 3.1 8B outperformed GPT-4o (F1=0.92). On the external validity test set, performance remained high (F1= 0.85). Fine-tuned open-source LLMs can surpass proprietary models in clinical note sectioning, offering advantages in cost, performance, and accessibility. 

**Abstract (ZH)**: 从临床笔记中提取段落对于下游分析至关重要，但由于格式差异性和手工分段的劳动密集型性质，这颇具挑战性。虽然专有的大型语言模型（LLMs）显示出一定的潜力，但隐私问题限制了它们的可访问性。本研究开发了一种使用开源LLMs的自动化笔记分段管道，重点关注三个段落：当前病情病史、间歇病史和评估与计划。我们对三个开源LLMs进行了微调，使用一个包含487篇病情进展记录的数据集来提取段落，并将结果与专有模型（GPT-4o、GPT-4o mini）进行比较。内部和外部有效性的评估基于精确度、召回率和F1分数。微调后的Llama 3.1 8B在准确性方面优于GPT-4o（F1=0.92）。在外部有效性测试集中，性能仍然很高（F1=0.85）。微调后的开源LLMs在临床笔记分段方面可以超越专有模型，提供成本、性能和可访问性方面的优势。 

---
# The Role of Generative AI in Software Student CollaborAItion 

**Title (ZH)**: 生成式人工智能在软件学生协作中的作用 

**Authors**: Natalie Kiesler, Jacqueline Smith, Juho Leinonen, Armando Fox, Stephen MacNeil, Petri Ihantola  

**Link**: [PDF](https://arxiv.org/pdf/2501.14084)  

**Abstract**: Collaboration is a crucial part of computing education. The increase in AI capabilities over the last couple of years is bound to profoundly affect all aspects of systems and software engineering, including collaboration. In this position paper, we consider a scenario where AI agents would be able to take on any role in collaborative processes in computing education. We outline these roles, the activities and group dynamics that software development currently include, and discuss if and in what way AI could facilitate these roles and activities. The goal of our work is to envision and critically examine potential futures. We present scenarios suggesting how AI can be integrated into existing collaborations. These are contrasted by design fictions that help demonstrate the new possibilities and challenges for computing education in the AI era. 

**Abstract (ZH)**: 合作是计算教育中的关键组成部分。近年来人工智能能力的提升必将深刻影响系统和软件工程的所有方面，包括合作。在这篇立场论文中，我们设想了一个场景，在该场景中，人工智能代理能够在计算教育中的合作流程中承担任何角色。我们概述了这些角色、当前软件开发中的活动和团队动态，并讨论人工智能是否以及如何促进这些角色和活动。我们工作的目标是构想并批判性地审视潜在的未来。我们提出了建议场景，说明人工智能如何融入现有的合作之中。这些场景与设计虚构作品相比较，有助于展示人工智能时代计算教育中新的可能性和挑战。 

---
# Communicating Activations Between Language Model Agents 

**Title (ZH)**: 语言模型代理之间的激活传播 

**Authors**: Vignav Ramesh, Kenneth Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14082)  

**Abstract**: Communication between multiple language model (LM) agents has been shown to scale up the reasoning ability of LMs. While natural language has been the dominant medium for inter-LM communication, it is not obvious this should be the standard: not only does natural language communication incur high inference costs that scale quickly with the number of both agents and messages, but also the decoding process abstracts away too much rich information that could be otherwise accessed from the internal activations. In this work, we propose a simple technique whereby LMs communicate via activations; concretely, we pause an LM $\textit{B}$'s computation at an intermediate layer, combine its current activation with another LM $\textit{A}$'s intermediate activation via some function $\textit{f}$, then pass $\textit{f}$'s output into the next layer of $\textit{B}$ and continue the forward pass till decoding is complete. This approach scales up LMs on new tasks with zero additional parameters and data, and saves a substantial amount of compute over natural language communication. We test our method with various functional forms $\textit{f}$ on two experimental setups--multi-player coordination games and reasoning benchmarks--and find that it achieves up to $27.0\%$ improvement over natural language communication across datasets with $<$$1/4$ the compute, illustrating the superiority and robustness of activations as an alternative "language" for communication between LMs. 

**Abstract (ZH)**: 多语言模型（LM）代理之间的通信已被证明能够提高语言模型的推理能力。虽然自然语言一直是跨语言模型通信的主要媒介，但这种标准未必是必须的：自然语言通信不仅伴随着高推理成本，这种成本随着代理和消息的数量迅速增加，而且解码过程会丢失过多原本可以从内部激活中获取的丰富信息。本文中，我们提出了一种简单的方法，通过激活进行通信。具体而言，我们在语言模型 B 的一个中间层暂停其计算，将当前激活与另一个语言模型 A 的中间激活通过某种函数 f 结合，然后将 f 的输出传递给语言模型 B 的下一层，并继续前向传播直到完成解码。这种方法可以在不增加任何额外参数和数据的情况下，提高语言模型在新任务上的表现，并且通过减少计算量，相较于自然语言通信具有显著优势。我们通过不同的函数形式 f 在两种实验设置——多方协调游戏和推理基准测试中测试了该方法，并在较低计算量（少于四分之一）的情况下实现了高达 27.0% 的性能提升，这表明激活作为语言模型之间通信的另一种“语言”的优越性和鲁棒性。 

---
# Expanding on the BRIAR Dataset: A Comprehensive Whole Body Biometric Recognition Resource at Extreme Distances and Real-World Scenarios (Collections 1-4) 

**Title (ZH)**: 扩展BRIAR数据集：一个在极端距离和真实场景下的全面全身生物识别资源（收集1-4） 

**Authors**: Gavin Jager, David Cornett III, Gavin Glenn, Deniz Aykac, Christi Johnson, Robert Zhang, Ryan Shivers, David Bolme, Laura Davies, Scott Dolvin, Nell Barber, Joel Brogan, Nick Burchfield, Carl Dukes, Andrew Duncan, Regina Ferrell, Austin Garrett, Jim Goddard, Jairus Hines, Bart Murphy, Sean Pharris, Brandon Stockwell, Leanne Thompson, Matthew Yohe  

**Link**: [PDF](https://arxiv.org/pdf/2501.14070)  

**Abstract**: The state-of-the-art in biometric recognition algorithms and operational systems has advanced quickly in recent years providing high accuracy and robustness in more challenging collection environments and consumer applications. However, the technology still suffers greatly when applied to non-conventional settings such as those seen when performing identification at extreme distances or from elevated cameras on buildings or mounted to UAVs. This paper summarizes an extension to the largest dataset currently focused on addressing these operational challenges, and describes its composition as well as methodologies of collection, curation, and annotation. 

**Abstract (ZH)**: 近年来，生物识别算法和操作系统的先进性得到了迅速提升，能够在更具挑战性的采集环境和消费者应用中提供高精度和鲁棒性。然而，当将这项技术应用于非传统设置时，如远距离识别或从建筑物上的高空摄像头或无人机上搭载的摄像头进行识别时，技术表现仍然存在巨大挑战。本文总结了目前最大的数据集扩展，该数据集旨在应对这些操作挑战，并描述了其构成以及收集、整理和标注的方法。 

---
# Revisiting CLIP: Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models 

**Title (ZH)**: 重新审视CLIP：使用领域特定基础模型高效对齐3D MRI和表格数据 

**Authors**: Jakob Krogh Petersen, Valdemar Licht, Mads Nielsen, Asbjørn Munk  

**Link**: [PDF](https://arxiv.org/pdf/2501.14051)  

**Abstract**: Multi-modal models require aligned, shared embedding spaces. However, common CLIP-based approaches need large amounts of samples and do not natively support 3D or tabular data, both of which are crucial in the medical domain. To address these issues, we revisit CLIP-style alignment by training a domain-specific 3D foundation model as an image encoder and demonstrate that modality alignment is feasible with only 62 MRI scans. Our approach is enabled by a simple embedding accumulation strategy required for training in 3D, which scales the amount of negative pairs across batches in order to stabilize training. We perform a thorough evaluation of various design choices, including the choice of backbone and loss functions, and evaluate the proposed methodology on zero-shot classification and image-retrieval tasks. While zero-shot image-retrieval remains challenging, zero-shot classification results demonstrate that the proposed approach can meaningfully align the representations of 3D MRI with tabular data. 

**Abstract (ZH)**: 多模态模型需要对齐且共享的嵌入空间。然而，常见的CLIP基方法需要大量的样本，且不原生支持3D或表格数据，而在医疗领域中，这两种数据类型至关重要。为解决这些问题，我们重新审视了CLIP风格的对齐方法，通过训练一个专门领域的3D基础模型作为图像编码器，并展示出只需62份MRI扫描就能实现模态对齐的可能性。我们的方法得益于为3D培训中所需的简单嵌入积累策略，该策略按批次增加了负样本的数量，从而稳定了训练过程。我们对各种设计选择进行了全面评估，包括骨干网络和损失函数的选择，并将所提出的框架应用于零样本分类和图像检索任务。尽管零样本图像检索仍然具有挑战性，但零样本分类的结果表明，所提出的方法能够有意义地对齐3D MRI和表格数据的表示。 

---
# GraphRAG under Fire 

**Title (ZH)**: 《GraphRAG受考验》

这个翻译在保持原意的同时，尽量符合学术规范的表达方式。如果需要更加具体的上下文或是有不同的表达偏好，请提供更多信息。 

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14050)  

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their reasoning. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: compared to conventional RAG, GraphRAG's graph-based indexing and retrieval enhance resilience against simple poisoning attacks; meanwhile, the same features also create new attack surfaces. We present GRAGPoison, a novel attack that exploits shared relations in the knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GRAGPoison employs three key strategies: i) relation injection to introduce false knowledge, ii) relation enhancement to amplify poisoning influence, and iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GRAGPoison substantially outperforms existing attacks in terms of effectiveness (up to 98% success rate) and scalability (using less than 68% poisoning text). We also explore potential defensive measures and their limitations, identifying promising directions for future research. 

**Abstract (ZH)**: GraphRAG通过将外部知识结构化为多尺度知识图谱，增强了检索增强生成（RAG）的能力，使语言模型在其推理过程中能够整合广泛的语境和细微的细节。尽管GraphRAG在多个领域已显示出成功，但其安全性影响仍很大程度上未被探索。为解决这一问题，本研究考察了GraphRAG对抗中毒攻击的脆弱性，并揭示了一个有趣的安全悖论：与传统的RAG相比，GraphRAG基于图的索引和检索增强了对简单中毒攻击的抵抗力；同时，相同的特征也创设了新的攻击途径。我们提出了一种名为GRAGPoison的新颖攻击方法，利用知识图谱中的共享关系来构建能够同时破坏多个查询的中毒文本。GRAGPoison采用三种关键策略：（i）关系注入以引入虚假知识，（ii）关系增强以放大中毒影响，（iii）叙事生成以在连贯文本中嵌入恶意内容。跨多个数据集和模型的实证评估显示，GRAGPoison在有效性（高达98%的成功率）和可扩展性（使用不到68%的中毒文本）方面远超现有攻击。我们还探索了潜在的防御措施及其局限性，并指出了未来研究的积极方向。 

---
# SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks 

**Title (ZH)**: SINHORMANN动态领域适应在不变神经网络中的图像分类 

**Authors**: Sneh Pandya, Purvik Patel, Brian D. Nord, Mike Walmsley, Aleksandra Ćiprijanović  

**Link**: [PDF](https://arxiv.org/pdf/2501.14048)  

**Abstract**: Modern neural networks (NNs) often do not generalize well in the presence of a "covariate shift"; that is, in situations where the training and test data distributions differ, but the conditional distribution of classification labels remains unchanged. In such cases, NN generalization can be reduced to a problem of learning more domain-invariant features. Domain adaptation (DA) methods include a range of techniques aimed at achieving this; however, these methods have struggled with the need for extensive hyperparameter tuning, which then incurs significant computational costs. In this work, we introduce SIDDA, an out-of-the-box DA training algorithm built upon the Sinkhorn divergence, that can achieve effective domain alignment with minimal hyperparameter tuning and computational overhead. We demonstrate the efficacy of our method on multiple simulated and real datasets of varying complexity, including simple shapes, handwritten digits, and real astronomical observations. SIDDA is compatible with a variety of NN architectures, and it works particularly well in improving classification accuracy and model calibration when paired with equivariant neural networks (ENNs). We find that SIDDA enhances the generalization capabilities of NNs, achieving up to a $\approx40\%$ improvement in classification accuracy on unlabeled target data. We also study the efficacy of DA on ENNs with respect to the varying group orders of the dihedral group $D_N$, and find that the model performance improves as the degree of equivariance increases. Finally, we find that SIDDA enhances model calibration on both source and target data--achieving over an order of magnitude improvement in the ECE and Brier score. SIDDA's versatility, combined with its automated approach to domain alignment, has the potential to advance multi-dataset studies by enabling the development of highly generalizable models. 

**Abstract (ZH)**: 现代神经网络（NN）在面对“协变量偏移”（covariate shift）的情况下往往无法很好地泛化；也就是说，在训练数据和测试数据分布不同但分类标签的条件分布保持不变的情况下，NN的泛化能力会受到限制。在这种情况下，NN的泛化可以简化为学习更加领域不变特征的问题。领域适应（Domain Adaptation, DA）方法包括一系列旨在实现这一目标的技术；然而，这些方法在需要大量超参数调优方面遇到了困难，这会导致显著的计算成本。在本文中，我们引入了基于Sinkhorn散度的即插即用DA训练算法SIDDA（Sinkhorn Divergence-based Domain Adaptation），该算法可以实现有效的领域对齐，同时仅需要最少的超参数调优和计算开销。我们通过多种复杂度不同的模拟和真实数据集（包括简单的形状、手写数字和真实天文观测数据）展示了我们方法的有效性。SIDDA兼容多种NN架构，并且它特别适合与等变神经网络（equivariant neural networks, ENNs）配对以提高分类准确性和模型校准。我们发现，SIDDA增强了NN的泛化能力，在未标记的目标数据上的分类准确率提高了约40%。我们还研究了DA方法在ENNs中的有效性，考察了二面体群$D_N$的不同阶数对模型性能的影响，并发现模型性能随着等变性程度的提高而提高。最后，我们发现SIDDA在源数据和目标数据上均增强了模型校准能力——ECE（校准误差）和布林德评分（Brier score）分别取得了一个数量级以上的改进。SIDDA的多功能性和其自动化的领域对齐方法有潜力推动多数据集研究，从而促进高度泛化模型的发展。 

---
# Leveraging Multiphase CT for Quality Enhancement of Portal Venous CT: Utility for Pancreas Segmentation 

**Title (ZH)**: 利用多阶段CT提高门静脉CT质量：对胰腺分割的应用价值 

**Authors**: Xinya Wang, Tejas Sudharshan Mathai, Boah Kim, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2501.14013)  

**Abstract**: Multiphase CT studies are routinely obtained in clinical practice for diagnosis and management of various diseases, such as cancer. However, the CT studies can be acquired with low radiation doses, different scanners, and are frequently affected by motion and metal artifacts. Prior approaches have targeted the quality improvement of one specific CT phase (e.g., non-contrast CT). In this work, we hypothesized that leveraging multiple CT phases for the quality enhancement of one phase may prove advantageous for downstream tasks, such as segmentation. A 3D progressive fusion and non-local (PFNL) network was developed. It was trained with three degraded (low-quality) phases (non-contrast, arterial, and portal venous) to enhance the quality of the portal venous phase. Then, the effect of scan quality enhancement was evaluated using a proxy task of pancreas segmentation, which is useful for tracking pancreatic cancer. The proposed approach improved the pancreas segmentation by 3% over the corresponding low-quality CT scan. To the best of our knowledge, we are the first to harness multiphase CT for scan quality enhancement and improved pancreas segmentation. 

**Abstract (ZH)**: 在临床实践中，多相CT研究常用于各种疾病的诊断和管理，如癌症。然而，CT研究可以使用低辐射剂量、不同扫描器进行，并且经常受到运动和金属伪影的影响。先前的方法针对特定CT相位（如非对比CT）的质量改进。在本研究中，我们假设利用多个CT相位来提高单一相位的质量可能对下游任务（如分割）有益。我们开发了一个3D逐步融合和非局部（PFNL）网络。该网络使用三个降级（低质量）相位（非对比、动脉期和门静脉期）进行训练，以提高门静脉相位的质量。然后，通过胰腺分割的代理任务评估扫描质量的提升效果，这对于监测胰腺癌很有帮助。我们提出的方法在与对应低质量CT扫描相比的情况下，胰腺分割精度提高了3%。据我们所知，我们是首次利用多相CT进行扫描质量提升和增强胰腺分割的研究。 

---
# Transfer Learning of Surrogate Models via Domain Affine Transformation Across Synthetic and Real-World Benchmarks 

**Title (ZH)**: 跨合成基准与真实世界基准领域齐次变换的代理模型迁移学习 

**Authors**: Shuaiqun Pan, Diederick Vermetten, Manuel López-Ibáñez, Thomas Bäck, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14012)  

**Abstract**: Surrogate models are frequently employed as efficient substitutes for the costly execution of real-world processes. However, constructing a high-quality surrogate model often demands extensive data acquisition. A solution to this issue is to transfer pre-trained surrogate models for new tasks, provided that certain invariances exist between tasks. This study focuses on transferring non-differentiable surrogate models (e.g., random forest) from a source function to a target function, where we assume their domains are related by an unknown affine transformation, using only a limited amount of transfer data points evaluated on the target. Previous research attempts to tackle this challenge for differentiable models, e.g., Gaussian process regression, which minimizes the empirical loss on the transfer data by tuning the affine transformations. In this paper, we extend the previous work to the random forest model and assess its effectiveness on a widely-used artificial problem set - Black-Box Optimization Benchmark (BBOB) testbed, and on four real-world transfer learning problems. The results highlight the significant practical advantages of the proposed method, particularly in reducing both the data requirements and computational costs of training surrogate models for complex real-world scenarios. 

**Abstract (ZH)**: 代理模型经常被用作实时过程昂贵执行的高效替代品。然而，构建高质量的代理模型往往需要大量的数据收集。为了解决这一问题，可以在特定任务之间存在不变性的前提下，利用预训练的代理模型进行迁移。本研究关注于将非可微代理模型（例如随机森林）从源函数转移到目标函数，我们假设任务之间的域由未知的仿射变换相关联，并仅通过在目标函数上评估有限数量的迁移数据点来调整仿射变换。此前的研究尝试通过最小化迁移数据上的经验损失来解决这一挑战，例如高斯过程回归模型。本研究将先前的工作扩展到随机森林模型，并在广泛使用的人工问题集——黑盒优化基准（BBOB）测试床以及四个真实世界的迁移学习问题上评估了其有效性。结果突显了所提出方法在降低复杂现实场景训练代理模型所需的大量数据和计算成本方面的显著实际优势。 

---
# Scalable and Explainable Verification of Image-based Neural Network Controllers for Autonomous Vehicles 

**Title (ZH)**: 基于图像的神经网络控制器可扩展且可解释的自主车辆验证方法 

**Authors**: Aditya Parameshwaran, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14009)  

**Abstract**: Existing formal verification methods for image-based neural network controllers in autonomous vehicles often struggle with high-dimensional inputs, computational inefficiency, and a lack of explainability. These challenges make it difficult to ensure safety and reliability, as processing high-dimensional image data is computationally intensive and neural networks are typically treated as black boxes. To address these issues, we propose \textbf{SEVIN} (Scalable and Explainable Verification of Image-Based Neural Network Controllers), a framework that leverages a Variational Autoencoders (VAE) to encode high-dimensional images into a lower-dimensional, explainable latent space. By annotating latent variables with corresponding control actions, we generate convex polytopes that serve as structured input spaces for verification, significantly reducing computational complexity and enhancing scalability. Integrating the VAE's decoder with the neural network controller allows for formal and robustness verification using these explainable polytopes. Our approach also incorporates robustness verification under real-world perturbations by augmenting the dataset and retraining the VAE to capture environmental variations. Experimental results demonstrate that SEVIN achieves efficient and scalable verification while providing explainable insights into controller behavior, bridging the gap between formal verification techniques and practical applications in safety-critical systems. 

**Abstract (ZH)**: 现有的基于图像的神经网络控制器的形式化验证方法往往难以处理高维输入、计算效率低下以及缺乏可解释性的问题。这些问题使得确保安全性和可靠性变得困难，因为处理高维图像数据计算量巨大，而神经网络通常被视为黑盒子。为了解决这些问题，我们提出了一种名为 **SEVIN（Scalable and Explainable Verification of Image-Based Neural Network Controllers）** 的框架，该框架利用变分自编码器（VAE）将高维图像编码到一个低维且可解释的潜在空间中。通过将潜在变量与相应的控制动作进行标注，我们可以生成凸多面体，这些多面体作为结构化的输入空间用于验证，显著降低了计算复杂度并增强了可扩展性。将VAE的解码器与神经网络控制器集成，使得可以通过这些可解释的多面体进行形式化和鲁棒性验证。我们的方法还通过增强数据集并重新训练VAE来捕捉环境变化，从而在现实世界干扰下进行鲁棒性验证。实验结果表明，SEVIN能够在保持高效和可扩展性的基础上提供对控制器行为的可解释洞察，从而弥合了形式化验证技术与安全关键系统实际应用之间的差距。 

---
# Adaptive Genetic Algorithms for Pulse-Level Quantum Error Mitigation 

**Title (ZH)**: 适应性遗传算法在脉冲级量子错误缓解中的应用 

**Authors**: William Aguilar-Calvo, Santiago Núñez-Corrales  

**Link**: [PDF](https://arxiv.org/pdf/2501.14007)  

**Abstract**: Noise remains a fundamental challenge in quantum computing, significantly affecting pulse fidelity and overall circuit performance. This paper introduces an adaptive algorithm for pulse-level quantum error mitigation, designed to enhance fidelity by dynamically responding to noise conditions without modifying circuit gates. By targeting pulse parameters directly, this method reduces the impact of various noise sources, improving algorithm resilience in quantum circuits. We show the latter by applying our protocol to Grover's and Deutsch-Jozsa algorithms. Experimental results show that this pulse-level strategy provides a flexible and efficient solution for increasing fidelity during the noisy execution of quantum circuits. Our work contributes to advancements in error mitigation techniques, essential for robust quantum computing. 

**Abstract (ZH)**: 噪声始终是量子计算中的一个根本挑战，显著影响了脉冲保真度和整体电路性能。本文介绍了一种针对脉冲级别的量子误差缓解适应算法，旨在通过动态响应噪声条件来提升保真度，而不修改电路门。通过直接针对脉冲参数进行操作，该方法减少了各种噪声源的影响，增强了量子电路算法的鲁棒性。我们通过将该协议应用于Grover算法和Deutsch-Jozsa算法，展示了这一点。实验结果表明，这种脉冲级别的策略为在噪声环境中提高量子电路的保真度提供了一种灵活且高效的解决方案。我们的工作为误差缓解技术的发展做出了贡献，这对于实现稳健的量子计算至关重要。 

---
# Asymmetrical Latent Representation for Individual Treatment Effect Modeling 

**Title (ZH)**: 异构潜在表示个体治疗效果建模 

**Authors**: Armand Lacombe, Michèle Sebag  

**Link**: [PDF](https://arxiv.org/pdf/2501.14006)  

**Abstract**: Conditional Average Treatment Effect (CATE) estimation, at the heart of counterfactual reasoning, is a crucial challenge for causal modeling both theoretically and applicatively, in domains such as healthcare, sociology, or advertising. Borrowing domain adaptation principles, a popular design maps the sample representation to a latent space that balances control and treated populations while enabling the prediction of the potential outcomes. This paper presents a new CATE estimation approach based on the asymmetrical search for two latent spaces called Asymmetrical Latent Representation for Individual Treatment Effect (ALRITE), where the two latent spaces are respectively intended to optimize the counterfactual prediction accuracy on the control and the treated samples. Under moderate assumptions, ALRITE admits an upper bound on the precision of the estimation of heterogeneous effects (PEHE), and the approach is empirically successfully validated compared to the state-of-the-art 

**Abstract (ZH)**: 条件平均处理效应（CATE）估计是反事实推理的核心，在因果建模中既是理论上的挑战，也是实际应用中的关键问题。这种挑战在医疗保健、社会学或广告等领域尤其突出。借鉴领域自适应的原则，一种流行的策略将样本表示映射到一个既能平衡对照组和处理组、又能预测潜在结果的潜在空间中。本文提出了一种新的CATE估计方法，这种方法基于对两个潜在空间的非对称搜索，并称为非对称个体治疗效应的潜在表示（Asymmetrical Latent Representation for Individual Treatment Effect, ALRITE），其中两个潜在空间分别旨在优化对照样本和处理样本的反事实预测精度。在适度的假设下，ALRITE对异质效应的估计精度上限（PEHE）具有理论保证，并且通过与当前最佳方法的实证比较，该方法得到了成功验证。 

---
# Device-aware Optical Adversarial Attack for a Portable Projector-camera System 

**Title (ZH)**: 面向设备的光学 adversarial 攻击：一种便携式投影-摄像机系统中的应用 

**Authors**: Ning Jiang, Yanhong Liu, Dingheng Zeng, Yue Feng, Weihong Deng, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14005)  

**Abstract**: Deep-learning-based face recognition (FR) systems are susceptible to adversarial examples in both digital and physical domains. Physical attacks present a greater threat to deployed systems as adversaries can easily access the input channel, allowing them to provide malicious inputs to impersonate a victim. This paper addresses the limitations of existing projector-camera-based adversarial light attacks in practical FR setups. By incorporating device-aware adaptations into the digital attack algorithm, such as resolution-aware and color-aware adjustments, we mitigate the degradation from digital to physical domains. Experimental validation showcases the efficacy of our proposed algorithm against real and spoof adversaries, achieving high physical similarity scores in FR models and state-of-the-art commercial systems. On average, there is only a 14% reduction in scores from digital to physical attacks, with high attack success rate in both white- and black-box scenarios. 

**Abstract (ZH)**: 基于深度学习的面部识别（FR）系统在数字和物理领域都容易受到对抗性示例的影响。物理攻击对已部署的系统构成更大的威胁，因为攻击者可以轻松访问输入通道，从而提供恶意输入以冒充受害者。本文针对现有投影相机基于的物理对抗光攻击在实际面部识别设置中的局限性。通过将设备感知的适应性纳入数字攻击算法，例如分辨率感知和色彩感知调整，我们可以减轻从数字到物理领域的性能下降。实验验证表明，我们的算法能够有效对抗真实和欺骗性对手，在面部识别模型和最先进的商用系统中实现高物理相似度得分。平均而言，从数字攻击到物理攻击的得分仅下降14%，在白盒和黑盒场景中均具有较高的攻击成功率。 

---
# ME-CPT: Multi-Task Enhanced Cross-Temporal Point Transformer for Urban 3D Change Detection 

**Title (ZH)**: ME-CPT：多任务增强的跨时域点变换器在城市三维变化检测中的应用 

**Authors**: Luqi Zhang, Haiping Wang, Chong Liu, Zhen Dong, Bisheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14004)  

**Abstract**: The point clouds collected by the Airborne Laser Scanning (ALS) system provide accurate 3D information of urban land covers. By utilizing multi-temporal ALS point clouds, semantic changes in urban area can be captured, demonstrating significant potential in urban planning, emergency management, and infrastructure maintenance. Existing 3D change detection methods struggle to efficiently extract multi-class semantic information and change features, still facing the following challenges: (1) the difficulty of accurately modeling cross-temporal point clouds spatial relationships for effective change feature extraction; (2) class imbalance of change samples which hinders distinguishability of semantic features; (3) the lack of real-world datasets for 3D semantic change detection. To resolve these challenges, we propose the Multi-task Enhanced Cross-temporal Point Transformer (ME-CPT) network. ME-CPT establishes spatiotemporal correspondences between point cloud across different epochs and employs attention mechanisms to jointly extract semantic change features, facilitating information exchange and change comparison. Additionally, we incorporate a semantic segmentation task and through the multi-task training strategy, further enhance the distinguishability of semantic features, reducing the impact of class imbalance in change types. Moreover, we release a 22.5 $km^2$ 3D semantic change detection dataset, offering diverse scenes for comprehensive evaluation. Experiments on multiple datasets show that the proposed MT-CPT achieves superior performance compared to existing state-of-the-art methods. The source code and dataset will be released upon acceptance at \url{this https URL}. 

**Abstract (ZH)**: 航空激光扫描（ALS）系统采集的点云数据提供了城市地表的精确三维信息。通过利用多时相 ALS 点云数据，可以捕获城市区域的语义变化，显示出在城市规划、应急管理及基础设施维护中的巨大潜力。现有的三维变化检测方法在高效提取多类语义信息和变化特征方面存在挑战，仍面临以下挑战：（1）多时相点云空间关系建模的准确性问题，这不利于有效变化特征提取；（2）变化样本类别的不平衡性，这阻碍了语义特征的可区分性；（3）缺乏用于三维语义变化检测的真实世界数据集。为了解决这些挑战，我们提出了多任务增强跨时相点变换网络（ME-CPT）。ME-CPT 在不同周期的点云之间建立时空对应关系，并采用注意机制联合提取语义变化特征，促进信息交换与变化比较。此外，我们引入了一项语义分割任务，并通过多任务训练策略进一步增强语义特征的可区分性，从而减少变化类型类别不平衡的影响。此外，我们发布了一个覆盖面积为 22.5 平方公里的三维语义变化检测数据集，提供了多样化的场景用于全面评估。在多个数据集上的实验结果显示，提出的 MT-CPT 在性能上优于现有的先进方法。源代码和数据集将在接收后通过 [此链接](this https URL) 发布。 

---
# PaMMA-Net: Plasmas magnetic measurement evolution based on data-driven incremental accumulative prediction 

**Title (ZH)**: PaMMA-Net：基于数据驱动增量累积预测的等离子体磁场测量进化 

**Authors**: Yunfei Ling, Zijie Liu, Jun Du, Yao Huang, Yuehang Wang, Bingjia Xiao, Xin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14003)  

**Abstract**: An accurate evolution model is crucial for effective control and in-depth study of fusion plasmas. Evolution methods based on physical models often encounter challenges such as insufficient robustness or excessive computational costs. Given the proven strong fitting capabilities of deep learning methods across various fields, including plasma research, this paper introduces a deep learning-based magnetic measurement evolution method named PaMMA-Net (Plasma Magnetic Measurements Incremental Accumulative Prediction Network). This network is capable of evolving magnetic measurements in tokamak discharge experiments over extended periods or, in conjunction with equilibrium reconstruction algorithms, evolving macroscopic parameters such as plasma shape. Leveraging a incremental prediction approach and data augmentation techniques tailored for magnetic measurements, PaMMA-Net achieves superior evolution results compared to existing studies. The tests conducted on real experimental data from EAST validate the high generalization capability of the proposed method. 

**Abstract (ZH)**: 准确的动力学演化模型对于有效控制和深入研究聚变等离子体至关重要。基于物理模型的动力学方法常常面临诸如鲁棒性不足或计算成本过高的挑战。鉴于深度学习方法在多个领域，包括等离子体研究中展现出强大的拟合能力，本文提出了一种基于深度学习的磁测量演化方法——PaMMA-Net（Plasma Magnetic Measurements Incremental Accumulative Prediction Network）。该网络能够延长托卡马克放电实验中的磁测量演化，或与平衡重建算法结合，演化宏观参数如等离子体形状。通过采用增量预测方法和专门为磁测量设计的数据增强技术，PaMMA-Net 较现有研究实现了更优异的演化效果。在 EAST 实验数据上的测试结果验证了该方法具有较高的泛化能力。 

---
# Advancing Math Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages 

**Title (ZH)**: 提升语言模型的数学推理能力：解决问题数据、数据合成方法和训练阶段的影响研究 

**Authors**: Zui Chen, Tianqiao Liu, Mi Tian, Qing Tong, Weiqi Luo, Zitao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14002)  

**Abstract**: Advancements in LLMs have significantly expanded their capabilities across various domains. However, mathematical reasoning remains a challenging area, prompting the development of math-specific LLMs. These models typically follow a two-stage training paradigm: pre-training with math-related corpora and post-training with problem datasets for SFT. Despite these efforts, the improvements in mathematical reasoning achieved through continued pre-training (CPT) are often less significant compared to those obtained via SFT. This study addresses this discrepancy by exploring alternative strategies during the pre-training phase, focusing on the use of problem-solving data over general mathematical corpora. We investigate three primary research questions: (1) Can problem-solving data enhance the model's mathematical reasoning capabilities more effectively than general mathematical corpora during CPT? (2) Are synthetic data from the same source equally effective, and which synthesis methods are most efficient? (3) How do the capabilities developed from the same problem-solving data differ between the CPT and SFT stages, and what factors contribute to these differences? Our findings indicate that problem-solving data significantly enhances the model's mathematical capabilities compared to general mathematical corpora. We also identify effective data synthesis methods, demonstrating that the tutorship amplification synthesis method achieves the best performance. Furthermore, while SFT facilitates instruction-following abilities, it underperforms compared to CPT with the same data, which can be partially attributed to its poor learning capacity for hard multi-step problem-solving data. These insights provide valuable guidance for optimizing the mathematical reasoning capabilities of LLMs, culminating in our development of a powerful mathematical base model called JiuZhang-8B. 

**Abstract (ZH)**: 大型语言模型（LLM）的最新进展显著扩展了它们在各个领域的功能。然而，数学推理仍然是一个挑战性领域，促使开发专门的数学LLM。这些模型通常遵循两阶段训练范式：使用与数学相关的语料进行预训练，然后使用问题数据集进行指令-following（SFT）训练。尽管如此，通过持续预训练（CPT）所取得的数学推理改进往往不如通过SFT所取得的显著。本研究旨在通过探索预训练阶段的替代策略来解决这一问题，重点关注使用解决问题的数据而非一般数学语料。我们探讨了三个主要的研究问题：（1）解决问题的数据能否在CPT期间更有效地增强模型的数学推理能力，而不是使用一般数学语料？（2）来自于同一源的合成数据是否同样有效，哪些合成方法是最高效的？（3）同一解决问题数据在CPT和SFT阶段所开发的能力有何不同，这些差异是由哪些因素引起的？我们的发现表明，解决问题的数据与一般数学语料相比，可以显著增强模型的数学能力。此外，我们还确定了有效的数据合成方法，证明了导师增强合成方法取得了最佳性能。同时，虽然SFT有助于增强指令跟随能力，但与使用相同数据进行CPT相比，SFT的表现较差，这可能部分归因于其对复杂多步解决问题数据的学习能力不佳。这些见解为优化LLM的数学推理能力提供了宝贵指导，促成了我们开发的一款强大的数学基础模型——JiuZhang-8B。 

---
# Enhancing kelp forest detection in remote sensing images using crowdsourced labels with Mixed Vision Transformers and ConvNeXt segmentation models 

**Title (ZH)**: 使用混合视觉变压器和ConvNeXt分割模型结合众包标签增强海藻森林在遥感图像中的检测 

**Authors**: Ioannis Nasios  

**Link**: [PDF](https://arxiv.org/pdf/2501.14001)  

**Abstract**: Kelp forests, as foundation species, are vital to marine ecosystems, providing essential food and habitat for numerous organisms. This study explores the integration of crowdsourced labels with advanced artificial intelligence models to develop a fast and accurate kelp canopy detection pipeline using Landsat images. Building on the success of a machine learning competition, where this approach ranked third and performed consistently well on both local validation and public and private leaderboards, the research highlights the effectiveness of combining Mixed Vision Transformers (MIT) with ConvNeXt models. Training these models on various image sizes significantly enhanced the accuracy of the ensemble results. U-Net emerged as the best segmentation architecture, with UpperNet also contributing to the final ensemble. Key Landsat bands, such as ShortWave InfraRed (SWIR1) and Near-InfraRed (NIR), were crucial while altitude data was used in postprocessing to eliminate false positives on land. The methodology achieved a high detection rate, accurately identifying about three out of four pixels containing kelp canopy while keeping false positives low. Despite the medium resolution of Landsat satellites, their extensive historical coverage makes them effective for studying kelp forests. This work also underscores the potential of combining machine learning models with crowdsourced data for effective and scalable environmental monitoring. All running code for training all models and inference can be found at this https URL. 

**Abstract (ZH)**: 海藻林作为基础物种，对海洋生态系统至关重要，为众多生物提供了必要的食物和栖息地。本研究旨在利用乐高卫星图像，通过众包标签与先进的人工智能模型的结合，开发一种快速且准确的海藻冠层检测流水线。本研究构建在机器学习竞赛的成功基础上，该竞赛中这一方法排名第三，并在本地验证和公开及私有排行榜上表现稳定。研究强调了将混合视觉 transformer (MIT) 与 ConvNeXt 模型结合使用的效果。通过对不同图像尺寸的训练，这些模型的整体结果准确性显著提高。U-Net 展现出了最佳的分割架构，而 UpNeXt 也对最终的集成模型起到了重要作用。短波红外 (SWIR1) 和近红外 (NIR) 等关键 Landsat 波段对于海藻林的检测至关重要，而海拔数据则在后期处理中被用来消除陆地上的假阳性。该方法实现了较高的检测率，准确地识别出约四分之三的含有海藻冠层的像素，同时保持假阳性率较低。尽管 Landsat 卫星的中分辨率有限，但其广泛的长期覆盖范围使它们成为研究海藻林的有效工具。这项工作也强调了将机器学习模型与众包数据结合使用在有效和可扩展的环境监测方面具有巨大潜力。所有训练所有模型和推理过程的运行代码可以在以下链接中找到：https://example.com。 

---
# Local Control Networks (LCNs): Optimizing Flexibility in Neural Network Data Pattern Capture 

**Title (ZH)**: 局部控制网络（LCNs）：优化神经网络数据模式捕获的灵活性 

**Authors**: Hy Nguyen, Duy Khoa Pham, Srikanth Thudumu, Hung Du, Rajesh Vasa, Kon Mouzakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.14000)  

**Abstract**: The widespread use of Multi-layer perceptrons (MLPs) often relies on a fixed activation function (e.g., ReLU, Sigmoid, Tanh) for all nodes within the hidden layers. While effective in many scenarios, this uniformity may limit the networks ability to capture complex data patterns. We argue that employing the same activation function at every node is suboptimal and propose leveraging different activation functions at each node to increase flexibility and adaptability. To achieve this, we introduce Local Control Networks (LCNs), which leverage B-spline functions to enable distinct activation curves at each node. Our mathematical analysis demonstrates the properties and benefits of LCNs over conventional MLPs. In addition, we demonstrate that more complex architectures, such as Kolmogorov-Arnold Networks (KANs), are unnecessary in certain scenarios, and LCNs can be a more efficient alternative. Empirical experiments on various benchmarks and datasets validate our theoretical findings. In computer vision tasks, LCNs achieve marginal improvements over MLPs and outperform KANs by approximately 5\%, while also being more computationally efficient than KANs. In basic machine learning tasks, LCNs show a 1\% improvement over MLPs and a 0.6\% improvement over KANs. For symbolic formula representation tasks, LCNs perform on par with KANs, with both architectures outperforming MLPs. Our findings suggest that diverse activations at the node level can lead to improved performance and efficiency. 

**Abstract (ZH)**: 多层感知机（MLPs）的广泛应用通常依赖于在隐藏层所有节点中使用固定的激活函数（例如ReLU、Sigmoid、Tanh）。虽然在许多场景中效果显著，但这种统一性可能会限制网络捕捉复杂数据模式的能力。我们认为，在每个节点使用相同的激活函数是次优的选择，并建议在每个节点上使用不同的激活函数以增加灵活性和适应性。为实现这一目标，我们引入了局部控制网络（LCNs），利用B-样条函数使每个节点具有独特的激活曲线。我们的数学分析证明了LCNs相较于传统MLPs的特性和优势。此外，我们还展示了在某些场景中，复杂的架构如柯尔莫哥洛夫-阿诺尔德网络（KANs）是不必要的，而LCNs可以作为一个更有效的替代方案。在各种基准和数据集上的实验证明了我们理论发现的有效性。在计算机视觉任务中，LCNs相较于MLPs仅实现边际改进，并比KANs高出约5%的性能，同时在计算效率上优于KANs。在基本的机器学习任务中，LCNs相较于MLPs提高1%，相较于KANs提高0.6%。对于符号公式表示任务，LCNs与KANs表现相当，两种架构都优于MLPs。我们的研究结果表明，节点级的多样化激活可以带来更好的性能和效率。 

---
# Framework for Progressive Knowledge Fusion in Large Language Models Through Structured Conceptual Redundancy Analysis 

**Title (ZH)**: 大型语言模型中通过结构化概念冗余分析实现渐进知识融合的框架 

**Authors**: Joseph Sakau, Evander Kozlowski, Roderick Thistledown, Basil Steinberger  

**Link**: [PDF](https://arxiv.org/pdf/2501.13999)  

**Abstract**: The organization of latent knowledge within large-scale models poses unique challenges when addressing overlapping representations and optimizing contextual accuracy. Conceptual redundancies embedded across layers often result in inefficiencies that affect both computational demands and task-specific outcomes. A framework was proposed to restructure these redundancies through advanced clustering techniques and dynamic thresholding, ensuring that critical semantic relationships are preserved while removing unnecessary overlaps. Evaluations revealed improved memory efficiency and faster inference times, alongside better alignment in latent knowledge clusters that enhanced interpretability. Improvements in error rates and adversarial robustness suggest that restructuring redundancies has broader implications for increasing model reliability across diverse applications. Comparative analyses highlighted reductions in resource consumption and notable gains in performance, particularly in translation and summarization tasks. Energy metrics demonstrated significant savings during training phases, further validating the practicality of the approach for real-world deployments. Representational fidelity was also enhanced, with latent space evaluations indicating better cluster alignment and higher semantic consistency. The methodology bridges a key gap in model optimization through directly addressing redundancies at the structural level. Its application opens avenues for scalable, efficient, and contextually aware systems that can adapt to complex, domain-specific tasks without compromising on performance. 

**Abstract (ZH)**: 大规模模型中潜在知识的组织在处理重叠表示和优化上下文准确性时带来了独特的挑战。跨层嵌入的概念冗余往往导致效率低下，影响计算需求和任务特定的结果。我们提出了一个框架，通过先进的聚类技术和动态阈值来重新结构这些冗余，确保保留关键语义关系的同时去除不必要的重叠。评估表明，这种方法提高了内存效率，加快了推理时间，并且提高了潜在知识簇的一致性，从而增强了可解释性。错误率和对抗性鲁棒性的改善表明，重新结构冗余对提高模型在不同应用中的可靠性具有更广泛的影响。对比分析显示资源消耗减少和性能显著提高，尤其是在翻译和摘要任务中。能量指标表明，在训练阶段实现了显著的节能，进一步证明了该方法的实际可行性。代表性的保真度也得到了提高，潜在空间评估表明簇对齐更好，语义一致性更高。该方法填补了模型优化中的关键空白，通过直接在结构层面处理冗余来实现这一目标。其应用为可扩展、高效、上下文感知的系统打开了门路，这些系统可以适应复杂的、领域特定的任务，同时保持性能。 

---
# Predictive Learning in Energy-based Models with Attractor Structures 

**Title (ZH)**: 基于吸引子结构的能量模型中的预测学习 

**Authors**: Xingsi Dong, Pengxiang Yuan, Si Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13997)  

**Abstract**: Predictive models are highly advanced in understanding the mechanisms of brain function. Recent advances in machine learning further underscore the power of prediction for optimal representation in learning. However, there remains a gap in creating a biologically plausible model that explains how the neural system achieves prediction. In this paper, we introduce a framework that employs an energy-based model (EBM) to capture the nuanced processes of predicting observation after action within the neural system, encompassing prediction, learning, and inference. We implement the EBM with a hierarchical structure and integrate a continuous attractor neural network for memory, constructing a biologically plausible model. In experimental evaluations, our model demonstrates efficacy across diverse scenarios. The range of actions includes eye movement, motion in environments, head turning, and static observation while the environment changes. Our model not only makes accurate predictions for environments it was trained on, but also provides reasonable predictions for unseen environments, matching the performances of machine learning methods in multiple tasks. We hope that this study contributes to a deep understanding of how the neural system performs prediction. 

**Abstract (ZH)**: 预测模型在理解大脑功能机制方面极为先进。近期机器学习的进步进一步突显了预测在学习中最优表示中的力量。然而，如何创建一个生物上可实现的模型来解释神经系统如何进行预测，仍存在差距。本文介绍了一种框架，该框架利用能量模型（EBM）来捕捉神经系统在动作后预测观察的精细过程，涵盖了预测、学习和推理。我们采用了分层结构实现EBM，并结合连续吸引子神经网络进行记忆，构建了一个生物上可实现的模型。在实验评估中，我们的模型在多种场景中表现出高效性。涉及的动作包括眼睛运动、环境中的动作、头部转动和环境变化中的静止观察。我们的模型不仅对训练环境中的观察进行了准确预测，还对未见过的环境提供了合理的预测，多项任务中均表现出与机器学习方法相当的性能。我们希望本研究有助于深入理解神经系统如何进行预测。 

---
# CSAOT: Cooperative Multi-Agent System for Active Object Tracking 

**Title (ZH)**: CSAOT：协同多代理系统在活性对象追踪中的应用 

**Authors**: Hy Nguyen, Bao Pham, Hung Du, Srikanth Thudumu, Rajesh Vasa, Kon Mouzakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.13994)  

**Abstract**: Object Tracking is essential for many computer vision applications, such as autonomous navigation, surveillance, and robotics. Unlike Passive Object Tracking (POT), which relies on static camera viewpoints to detect and track objects across consecutive frames, Active Object Tracking (AOT) requires a controller agent to actively adjust its viewpoint to maintain visual contact with a moving target in complex environments. Existing AOT solutions are predominantly single-agent-based, which struggle in dynamic and complex scenarios due to limited information gathering and processing capabilities, often resulting in suboptimal decision-making. Alleviating these limitations necessitates the development of a multi-agent system where different agents perform distinct roles and collaborate to enhance learning and robustness in dynamic and complex environments. Although some multi-agent approaches exist for AOT, they typically rely on external auxiliary agents, which require additional devices, making them costly. In contrast, we introduce the Collaborative System for Active Object Tracking (CSAOT), a method that leverages multi-agent deep reinforcement learning (MADRL) and a Mixture of Experts (MoE) framework to enable multiple agents to operate on a single device, thereby improving tracking performance and reducing costs. Our approach enhances robustness against occlusions and rapid motion while optimizing camera movements to extend tracking duration. We validated the effectiveness of CSAOT on various interactive maps with dynamic and stationary obstacles. 

**Abstract (ZH)**: 对象跟踪对于许多计算机视觉应用（如自主导航、监控和机器人技术）至关重要。与被动对象跟踪（POT）不同，被动对象跟踪依赖于静态摄像头视角来检测和跟踪连续帧中的对象，而主动对象跟踪（AOT）需要一个控制器代理主动调整其视角，以在复杂环境下保持与移动目标的视觉接触。现有的AOT解决方案大多基于单个代理，由于信息收集和处理能力的限制，在动态和复杂场景中通常表现不佳，导致决策效果欠佳。要解决这些限制，需要开发一个多代理系统，其中不同代理执行不同的任务并协同工作，以增强在动态和复杂环境中的学习和稳健性。虽然存在一些基于多代理的方法来实现AOT，但它们通常依赖于外部辅助代理，这需要额外的设备，从而使它们成本较高。相比之下，我们提出了一种协作的主动对象跟踪系统（CSAOT）方法，该方法利用多代理深度强化学习（MADRL）和专家混合（Mixture of Experts, MoE）框架，使多个代理能够在单个设备上协同工作，从而提高跟踪性能并降低成本。我们的方法增强了对遮挡和快速运动的鲁棒性，并优化了摄像机运动以延长跟踪时间。我们通过在具有动态和静态障碍的多种交互地图上验证CSAOT的有效性，证明了其优越性。 

---
# CAPRAG: A Large Language Model Solution for Customer Service and Automatic Reporting using Vector and Graph Retrieval-Augmented Generation 

**Title (ZH)**: CAPRAG：一种基于向量和图检索增强生成的大型语言模型解决方案，用于客户服务和自动报告 

**Authors**: Hamza Landolsi, Kais Letaief, Nizar Taghouti, Ines Abdeljaoued-Tej  

**Link**: [PDF](https://arxiv.org/pdf/2501.13993)  

**Abstract**: The introduction of new features and services in the banking sector often overwhelms customers, creating an opportunity for banks to enhance user experience through financial chatbots powered by large language models (LLMs). We initiated an AI agent designed to provide customers with relevant information about banking services and insights from annual reports. We proposed a hybrid Customer Analysis Pipeline Retrieval-Augmented Generation (CAPRAG) that effectively addresses both relationship-based and contextual queries, thereby improving customer engagement in the digital banking landscape. To implement this, we developed a processing pipeline to refine text data, which we utilized in two main frameworks: Vector RAG and Graph RAG. This dual approach enables us to populate both vector and graph databases with processed data for efficient retrieval. The Cypher query component is employed to effectively query the graph database. When a user submits a query, it is first expanded by a query expansion module before being routed to construct a final query from the hybrid Knowledge Base (KB). This final query is then sent to an open-source LLM for response generation. Overall, our innovative, designed to international banks, serves bank's customers in an increasingly complex digital environment, enhancing clarity and accessibility of information. 

**Abstract (ZH)**: 银行部门引入新功能和服务往往会让客户感到不知所措，为银行通过大型语言模型（LLMs）驱动的金融聊天机器人提升用户体验提供了机会。我们启动了一个AI代理，旨在为客户提供与银行服务相关的信息以及年度报告的见解。我们提出了一种混合客户分析管道检索-增强生成（CAPRAG）方法，该方法有效地处理了基于关系和上下文的查询，从而在数字银行领域中提高了客户服务的参与度。为了实现这一目标，我们开发了一个处理管道来精炼文本数据，并将其应用于两个主要框架：向量RAG和图RAG。这种双重方法使我们能够将处理后的数据填充到向量和图数据库中，以实现高效的检索。我们使用Cypher查询组件来有效地查询图数据库。当用户提交查询时，该查询首先由查询扩展模块扩展，然后路由到构建最终查询的混合知识库（KB）。该最终查询随后发送给开源LLM以生成回应。总体而言，我们创新的解决方案旨在为国际银行的客户提供一个日益复杂的数字环境中的帮助，从而提高信息的清晰度和可访问性。 

---
# Dual-Branch HNSW Approach with Skip Bridges and LID-Driven Optimization 

**Title (ZH)**: 双分支HNSW方法结合跳跃桥梁和基于LID的优化 

**Authors**: Hy Nguyen, Nguyen Hung Nguyen, Nguyen Linh Bao Nguyen, Srikanth Thudumu, Hung Du, Rajesh Vasa, Kon Mouzakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.13992)  

**Abstract**: The Hierarchical Navigable Small World (HNSW) algorithm is widely used for approximate nearest neighbor (ANN) search, leveraging the principles of navigable small-world graphs. However, it faces some limitations. The first is the local optima problem, which arises from the algorithm's greedy search strategy, selecting neighbors based solely on proximity at each step. This often leads to cluster disconnections. The second limitation is that HNSW frequently fails to achieve logarithmic complexity, particularly in high-dimensional datasets, due to the exhaustive traversal through each layer. To address these limitations, we propose a novel algorithm that mitigates local optima and cluster disconnections while enhancing the construction speed, maintaining inference speed. The first component is a dual-branch HNSW structure with LID-based insertion mechanisms, enabling traversal from multiple directions. This improves outlier node capture, enhances cluster connectivity, accelerates construction speed and reduces the risk of local minima. The second component incorporates a bridge-building technique that bypasses redundant intermediate layers, maintaining inference and making up the additional computational overhead introduced by the dual-branch structure. Experiments on various benchmarks and datasets showed that our algorithm outperforms the original HNSW in both accuracy and speed. We evaluated six datasets across Computer Vision (CV), and Natural Language Processing (NLP), showing recall improvements of 18\% in NLP, and up to 30\% in CV tasks while reducing the construction time by up to 20\% and maintaining the inference speed. We did not observe any trade-offs in our algorithm. Ablation studies revealed that LID-based insertion had the greatest impact on performance, followed by the dual-branch structure and bridge-building components. 

**Abstract (ZH)**: 层次导航可小世界（HNSW）算法广泛用于近似最近邻（ANN）搜索，利用了可导航小世界图的原则。然而，该算法面临一些局限性。首先，局部最优问题源于算法的贪婪搜索策略，在每一步仅依据距离选择邻居，这常常导致聚类断开。其次，HNSW在高维数据集中通常无法实现对数复杂度，主要是因为必须遍历每一层。为解决这些问题，我们提出了一种新颖的算法，能够缓解局部最优和聚类断开问题，同时加快构建速度并保持推理速度。该算法的第一部分是基于局部异质度（LID）的插入机制的双分支HNSW结构，允许从多个方向进行遍历。这改进了异常节点的捕获，增强了聚类连接性，加速了构建速度，并减少了局部最小值的风险。第二部分引入了桥接技术，绕过了冗余的中间层，保持了推理速度，并弥补了双分支结构引入的额外计算开销。我们在各种基准和数据集上的实验表明，我们的算法在精确度和速度上都优于原始HNSW。我们在计算机视觉（CV）和自然语言处理（NLP）六个数据集上的测试结果显示，NLP任务中的召回率提高了18%，CV任务中最高提高了30%，同时将构建时间降低了20%，并维持了推理速度。我们没有观测到任何权衡。消融研究显示，基于LID的插入机制对性能的影响最大，其次是双分支结构和桥接技术部分。 

---
# CGI: Identifying Conditional Generative Models with Example Images 

**Title (ZH)**: CGI：使用示例图像识别条件生成模型 

**Authors**: Zhi Zhou, Hao-Zhe Tan, Peng-Xiao Song, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.13991)  

**Abstract**: Generative models have achieved remarkable performance recently, and thus model hubs have emerged. Existing model hubs typically assume basic text matching is sufficient to search for models. However, in reality, due to different abstractions and the large number of models in model hubs, it is not easy for users to review model descriptions and example images, choosing which model best meets their needs. Therefore, it is necessary to describe model functionality wisely so that future users can efficiently search for the most suitable model for their needs. Efforts to address this issue remain limited. In this paper, we propose Conditional Generative Model Identification (CGI), which aims to provide an effective way to identify the most suitable model using user-provided example images rather than requiring users to manually review a large number of models with example images. To address this problem, we propose the PromptBased Model Identification (PMI) , which can adequately describe model functionality and precisely match requirements with specifications. To evaluate PMI approach and promote related research, we provide a benchmark comprising 65 models and 9100 identification tasks. Extensive experimental and human evaluation results demonstrate that PMI is effective. For instance, 92% of models are correctly identified with significantly better FID scores when four example images are provided. 

**Abstract (ZH)**: 生成模型近年来取得了显著的性能提升，因此模型库也随之涌现。现有模型库通常假设基本的文字匹配就足以用于搜索模型。然而，实际情况是，由于模型库中模型的多种抽象层次和大量模型的存在，用户要通过查阅模型描述和示例图像来选择最适合需求的模型仍然非常困难。因此，明智地描述模型功能以帮助未来用户高效地找到满足他们需求的最佳模型是必要的。目前针对这一问题的努力仍然有限。在本文中，我们提出了一种条件生成模型识别（Conditional Generative Model Identification, CGI），其目标是利用用户提供的示例图像来有效识别最合适的模型，而非要求用户手动审查大量的模型及其示例图像。为了解决这一问题，我们提出了基于提示的模型识别（Prompt-Based Model Identification, PMI），它可以充分描述模型功能，并精确匹配需求和规格。为了评估PMI方法并促进相关研究，我们提供了一个基准数据集，包括65个模型和9100个识别任务。广泛的实验和人工评估结果表明，PMI是有效的。例如，当提供四张示例图像时，92%的模型能够被正确识别，并且FID分数有显著提升。 

---
# FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting 

**Title (ZH)**: FreEformer：增强频率变换的变压器模型在多变量时间序列预测中的应用 

**Authors**: Wenzhen Yue, Yong Liu, Xianghua Ying, Bowei Xing, Ruohao Guo, Ji Shi  

**Link**: [PDF](https://arxiv.org/pdf/2501.13989)  

**Abstract**: This paper presents \textbf{FreEformer}, a simple yet effective model that leverages a \textbf{Fre}quency \textbf{E}nhanced Trans\textbf{former} for multivariate time series forecasting. Our work is based on the assumption that the frequency spectrum provides a global perspective on the composition of series across various frequencies and is highly suitable for robust representation learning. Specifically, we first convert time series into the complex frequency domain using the Discrete Fourier Transform (DFT). The Transformer architecture is then applied to the frequency spectra to capture cross-variate dependencies, with the real and imaginary parts processed independently. However, we observe that the vanilla attention matrix exhibits a low-rank characteristic, thus limiting representation diversity. This could be attributed to the inherent sparsity of the frequency domain and the strong-value-focused nature of Softmax in vanilla attention. To address this, we enhance the vanilla attention mechanism by introducing an additional learnable matrix to the original attention matrix, followed by row-wise L1 normalization. Theoretical analysis~demonstrates that this enhanced attention mechanism improves both feature diversity and gradient flow. Extensive experiments demonstrate that FreEformer consistently outperforms state-of-the-art models on eighteen real-world benchmarks covering electricity, traffic, weather, healthcare and finance. Notably, the enhanced attention mechanism also consistently improves the performance of state-of-the-art Transformer-based forecasters. 

**Abstract (ZH)**: 本文提出了一个简单且有效的模型 \textbf{FreEformer}，该模型利用一个基于 \textbf{频}率增强的 \textbf{变换}式（\textbf{Fre}quency \textbf{E}nhanced Trans\textbf{former}）进行多变量时间序列预测。我们的工作基于这样一个假设：频谱提供了一种关于不同频率组成序列的全局视角，并且广泛适用于健壮的表示学习。具体来说，我们首先使用离散傅里叶变换（DFT）将时间序列转换到复频域。然后，应用Transformer架构来捕捉跨变量之间的依赖关系，将实部和虚部独立处理。然而，我们观察到原始的注意力矩阵表现出低秩特性，从而限制了表示的多样性。这可能是由于频域的固有稀疏性以及在原始注意力机制中Softmax的强值聚焦特性所致。为了解决这一问题，我们通过在原始注意力矩阵中引入一个可学习矩阵，并进行行间L1归一化，来增强原始的注意力机制。理论上分析表明，这种增强的注意力机制可以提高特征的多样性和梯度流动。大量实验表明，FreEformer在涵盖电力、交通、天气、医疗和金融的十八个实际基准测试中，始终优于现有最先进的模型。值得注意的是，这种增强的注意力机制也可以在现有的Transformer基预测器中一致地提高性能。 

---
# MCRL4OR: Multimodal Contrastive Representation Learning for Off-Road Environmental Perception 

**Title (ZH)**: MCRL4OR：离路环境感知的多模态对比表示学习 

**Authors**: Yi Yang, Zhang Zhang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13988)  

**Abstract**: Most studies on environmental perception for autonomous vehicles (AVs) focus on urban traffic environments, where the objects/stuff to be perceived are mainly from man-made scenes and scalable datasets with dense annotations can be used to train supervised learning models. By contrast, it is hard to densely annotate a large-scale off-road driving dataset manually due to the inherently unstructured nature of off-road environments. In this paper, we propose a Multimodal Contrastive Representation Learning approach for Off-Road environmental perception, namely MCRL4OR. This approach aims to jointly learn three encoders for processing visual images, locomotion states, and control actions by aligning the locomotion states with the fused features of visual images and control actions within a contrastive learning framework. The causation behind this alignment strategy is that the inertial locomotion state is the result of taking a certain control action under the current landform/terrain condition perceived by visual sensors. In experiments, we pre-train the MCRL4OR with a large-scale off-road driving dataset and adopt the learned multimodal representations for various downstream perception tasks in off-road driving scenarios. The superior performance in downstream tasks demonstrates the advantages of the pre-trained multimodal representations. The codes can be found in \url{this https URL}. 

**Abstract (ZH)**: 大多数有关自动驾驶车辆（AVs）环境感知的研究集中在城市交通环境中，这些环境中需要感知的主要对象多来自人造场景，且可以使用带有密集注释的大规模数据集来训练监督学习模型。相比之下，由于非道路环境的固有非结构化特性，人工密集标注大规模非道路驾驶数据集变得非常困难。在本文中，我们提出了一种针对非道路环境感知的多模态对比表示学习方法，即MCRL4OR。该方法旨在通过对比学习框架将运动状态与视觉图像和控制动作融合特征对齐，同时学习用于处理视觉图像、运动状态和控制动作的三个编码器。这种对齐策略背后的因果关系在于，当前地形/路况条件下视觉传感器所感知到的状态是由特定的控制动作引起的。在实验中，我们使用大规模的非道路驾驶数据集对MCRL4OR进行预训练，并使用学到的多模态表示完成各种下游感知任务。下游任务中的优异性能表明预训练的多模态表示具有优势。相关代码可以在 \url{this https URL} 获取。 

---
# OstQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting 

**Title (ZH)**: OstQuant：通过正交性和缩放变换优化大型语言模型量化以实现更好的分布拟合 

**Authors**: Xing Hu, Yuan Cheng, Dawei Yang, Zukang Xu, Zhihang Yuan, Jiangyong Yu, Chen Xu, Zhe Jiang, Sifan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.13987)  

**Abstract**: Post-training quantization (PTQ) has emerged as a widely adopted technique for compressing and accelerating Large Language Models (LLMs). The major challenge in LLM quantization is that uneven and heavy-tailed data distributions can expand the quantization range, thereby reducing bit precision for most values. Recent methods attempt to eliminate outliers and balance inter-channel differences by employing linear transformations; however, they remain heuristic and are often overlook optimizing the data distribution across the entire quantization this http URL this paper, we introduce Quantization Space Utilization Rate (QSUR), a novel metric that effectively assesses the quantizability of transformed data by measuring the space utilization of the data in the quantization space. We complement QSUR with mathematical derivations that examine the effects and limitations of various transformations, guiding our development of Orthogonal and Scaling Transformation-based Quantization (OSTQuant). OSQuant employs a learnable equivalent transformation, consisting of an orthogonal transformation and a scaling transformation, to optimize the distributions of weights and activations across the entire quantization space. Futhermore, we propose the KL-Top loss function, designed to mitigate noise during optimization while retaining richer semantic information within the limited calibration data imposed by PTQ. OSTQuant outperforms existing work on various LLMs and benchmarks. In the W4-only setting, it retains 99.5\% of the floating-point accuracy. In the more challenging W4A4KV4 configuration, OSTQuant reduces the performance gap by 32\% on the LLaMA-3-8B model compared to state-of-the-art methods. \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 后训练量化（PTQ）已成为压缩和加速大规模语言模型（LLMs）的一种广泛采用的技术。在LLM量化中，主要挑战在于非均匀和厚尾的数据分布会扩大量化范围，从而降低大部分值的有效位精度。最近的方法通过使用线性变换尝试消除异常值并平衡通道间的差异，但这些方法仍然具有启发性，并且往往没有充分优化整个量化过程中数据分布的问题。本文中，我们引入了量化空间利用率（QSUR）这一新颖的度量标准，能够有效地评估转换后数据的可量化性，通过测量数据在量化空间中的空间利用率来进行评估。我们通过数学推导补充了QSUR，以检查各种变换的效果和限制，从而指导我们开发基于正交和缩放变换的量化（OSTQuant）方法。OSQuant采用了可学习等效变换，该变换由正交变换和缩放变换组成，用于优化整个量化空间中权重和激活值的分布。此外，我们还提出了KL-Top损失函数，旨在在有限制的标定数据下降低优化过程中的噪声影响，同时保留更多的语义信息。OSTQuant在多种LLM和基准测试中表现出色。在W4-only设置中，OSTQuant保持了99.5%的浮点精度。在更具挑战性的W4A4KV4配置下，OSTQuant相比于最新的方法，在LaMA-3-8B模型上将性能差距降低了32%。 

---
# An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks 

**Title (ZH)**: 用于O(3)-对称深网络的高效稀疏核生成器 

**Authors**: Vivek Bharadwaj, Austin Scott Glover, Aydin Buluc, James Demmel  

**Link**: [PDF](https://arxiv.org/pdf/2501.13986)  

**Abstract**: Rotation equivariant graph neural networks, i.e., networks designed to guarantee certain geometric relations between their inputs and outputs, yield state-of-the-art performance on spatial deep learning tasks. They exhibit high data efficiency during training and significantly reduced inference time for interatomic potential calculations compared to classical approaches. Key to these models is the Clebsch-Gordon (CG) tensor product, a kernel that contracts two dense feature vectors with a highly structured sparse tensor to produce a dense output vector. The operation, which may be repeated millions of times for typical equivariant models, is a costly and inefficient bottleneck. We introduce a GPU sparse kernel generator for the CG tensor product that provides significant speedup over the best existing open and closed-source implementations. Our implementation achieves high performance by carefully managing GPU shared memory through static analysis at model compile-time, minimizing reads and writes to global memory. We break the tensor product into a series of kernels with operands that fit entirely into registers, enabling us to emit long arithmetic instruction streams that maximize instruction-level parallelism. By fusing the CG tensor product with a subsequent graph convolution, we reduce both intermediate storage and global memory traffic over naive approaches that duplicate input data. We also provide optimized kernels for the gradient of the CG tensor product and a novel identity for the higher partial derivatives required to predict interatomic forces. Our fused kernels offer up to 4.5x speedup for the forward pass and 3x for the backward pass over NVIDIA cuEquivariance, as well as >10x speedup over the widely-used e3nn package. We offer up to 5.3x inference-time speedup for the MACE chemistry foundation model over the original unoptimized version. 

**Abstract (ZH)**: 旋转不变图神经网络，即设计用于保证其输入和输出之间某些几何关系的网络，已在空间深度学习任务中表现出最先进的性能。与经典方法相比，它们在训练过程中表现出高数据效率，并且在原子间势能计算中的推断时间显著减少。这些模型的关键在于克莱布施-高登(Clebsch-Gordon, CG)张量乘积核，这是一种将两个密集特征向量通过一个高度结构化的稀疏张量进行收缩，以生成一个密集输出向量的操作。这种操作可能要在典型平移不变模型中重复数百万次，是一个成本高且效率低下的瓶颈。

我们提出了一种针对CG张量乘积的GPU稀疏核生成器，其在现有最佳的开源和封闭源代码实现之上提供了显著的加速。通过在模型编译时进行静态分析，仔细管理GPU共享内存，最小化全局内存的读写，我们的实现达到了高性能。我们将张量乘积分解为一系列完全装入寄存器的运算符，使得我们能够生成长的算术指令流，最大限度地提高指令级并行性。通过将CG张量乘积与后续的图卷积融合，我们减少了与天真实现（重复输入数据）相比的中间存储和全局内存流量。此外，我们还提供了CG张量乘积梯度的优化内核，并提出了一种新颖的高阶偏导数的身份公式，这些公式用于预测原子间力。我们的融合内核在前向传播中提供了多达4.5倍的加速，在后向传播中提供了3倍的加速，相较于NVIDIA cuEquivariance，分别为4.5倍和3倍。相较于广泛使用的e3nn包，我们的融合内核则分别提供了超过10倍和10倍的加速。我们的模型为MACE化学基础模型提供了高达5.3倍的推断时间加速，相较于原始未优化版本。 

---
# Pilot: Building the Federated Multimodal Instruction Tuning Framework 

**Title (ZH)**: Pilot: 构建联邦多模态指令调优框架 

**Authors**: Baochen Xiong, Xiaoshan Yang, Yaguang Song, Yaowei Wang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13985)  

**Abstract**: In this paper, we explore a novel federated multimodal instruction tuning task(FedMIT), which is significant for collaboratively fine-tuning MLLMs on different types of multimodal instruction data on distributed devices. To solve the new task, we propose a federated multimodal instruction tuning framework(Pilot). Our framework integrates two stages of "adapter on adapter" into the connector of the vision encoder and the LLM. In stage 1, we extract task-specific features and client-specific features from visual information. In stage 2, we build the cross-task Mixture-of-Adapters(CT-MoA) module to perform cross-task interaction. Each client can not only capture personalized information of local data and learn task-related multimodal information, but also learn general knowledge from other tasks. In addition, we introduce an adaptive parameter aggregation strategy for text training parameters, which optimizes parameter aggregation by calculating weights based on the euclidean distance between parameters, so that parameter aggregation can benefit from positive effects to the greatest extent while effectively reducing negative effects. Our framework can collaboratively exploit distributed data from different local clients to learn cross-task knowledge without being affected by the task heterogeneity during instruction tuning. The effectiveness of our method is verified in two different cross-task scenarios. 

**Abstract (ZH)**: 本文探讨了一种新颖的联邦多模态指令调优任务（FedMIT），这对于在不同类型的多模态指令数据上分布式设备协同微调机器学习大模型具有重要意义。为了解决这一新任务，我们提出了一种联邦多模态指令调优框架（Pilot）。该框架在视觉编码器和LLM之间的连接器中集成了两阶段的“适配器嵌套适配器”结构。第一阶段，我们从视觉信息中提取任务特定特征和客户端特定特征；第二阶段，我们构建跨任务适配器混合模块（CT-MoA），实现跨任务交互。每个客户端不仅可以捕获本地数据的个性化信息并学习与任务相关的多模态信息，还可以从其他任务中学习通用知识。此外，我们引入了一种自适应参数聚合策略，通过基于参数间欧几里得距离计算权重来优化文本训练参数的聚合方式，从而最大程度地利用积极影响同时有效减少负面影响。该框架能够在指令调优过程中不受任务异质性的影响，协同利用来自不同本地客户端的分布式数据学习跨任务知识。我们通过两种不同的跨任务场景验证了该方法的有效性。 

---
# Comprehensive Modeling and Question Answering of Cancer Clinical Practice Guidelines using LLMs 

**Title (ZH)**: 使用大规模语言模型对癌症临床实践指南进行全面建模与问答 

**Authors**: Bhumika Gupta, Pralaypati Ta, Keerthi Ram, Mohanasankar Sivaprakasam  

**Link**: [PDF](https://arxiv.org/pdf/2501.13984)  

**Abstract**: The updated recommendations on diagnostic procedures and treatment pathways for a medical condition are documented as graphical flows in Clinical Practice Guidelines (CPGs). For effective use of the CPGs in helping medical professionals in the treatment decision process, it is necessary to fully capture the guideline knowledge, particularly the contexts and their relationships in the graph. While several existing works have utilized these guidelines to create rule bases for Clinical Decision Support Systems, limited work has been done toward directly capturing the full medical knowledge contained in CPGs. This work proposes an approach to create a contextually enriched, faithful digital representation of National Comprehensive Cancer Network (NCCN) Cancer CPGs in the form of graphs using automated extraction and node & relationship classification. We also implement semantic enrichment of the model by using Large Language Models (LLMs) for node classification, achieving an accuracy of 80.86% and 88.47% with zero-shot learning and few-shot learning, respectively. Additionally, we introduce a methodology for answering natural language questions with constraints to guideline text by leveraging LLMs to extract the relevant subgraph from the guideline knowledge base. By generating natural language answers based on subgraph paths and semantic information, we mitigate the risk of incorrect answers and hallucination associated with LLMs, ensuring factual accuracy in medical domain Question Answering. 

**Abstract (ZH)**: 将以下关于临床实践指南（CPGs）中诊断流程和治疗路径更新建议的内容或标题翻译成中文，符合学术规范：

临床实践指南（CPGs）中记录了某一医疗条件的更新后的诊断流程和治疗路径推荐，通常以图形流程的形式呈现。为了在诊断决策过程中有效利用CPGs辅助医疗专业人员，需要全面捕捉指导原则知识，特别是图中的上下文及其关系。虽然已有几项研究利用这些指南创建临床决策支持系统的规则库，但直接从CPGs中提取完整医疗知识的工作相对较少。本研究提出了一种使用自动化提取和节点与关系分类方法来创建富含上下文、忠实表示国家综合癌症网络（NCCN）癌症指南的图的方法。我们还通过使用大规模语言模型（LLMs）对模型进行语义增强，分别在零样本学习和少量样本学习下，节点分类的准确率分别为80.86%和88.47%。此外，我们提出了一种利用LLMs从指南知识库中提取相关子图的方法，以应对自然语言问题，并提出了约束条件下的自然语言答案生成方法。通过基于子图路径和语义信息生成自然语言回答，我们减轻了LLMs在生成答案时可能出现的错误和虚构风险，确保医疗领域问答系统的事实准确性。 

---
# AdEval: Alignment-based Dynamic Evaluation to Mitigate Data Contamination in Large Language Models 

**Title (ZH)**: AdEval: 基于对齐的动态评估方法以减轻大型语言模型中的数据污染问题 

**Authors**: Yang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13983)  

**Abstract**: As Large Language Models (LLMs) are pretrained on massive-scale corpora, the issue of data contamination has become increasingly severe, leading to potential overestimation of model performance during evaluation. To address this, we propose AdEval (Alignment-based Dynamic Evaluation), a dynamic data evaluation method aimed at mitigating the impact of data contamination on evaluation reliability. AdEval extracts key knowledge points and main ideas to align dynamically generated questions with static data's core concepts. It also leverages online search to provide detailed explanations of related knowledge points, thereby creating high-quality evaluation samples with robust knowledge support. Furthermore, AdEval incorporates mechanisms to control the number and complexity of questions, enabling dynamic alignment and flexible adjustment. This ensures that the generated questions align with the complexity of static data while supporting varied complexity levels. Based on Bloom's taxonomy, AdEval conducts a multi-dimensional evaluation of LLMs across six cognitive levels: remembering, understanding, applying, analyzing, evaluating, and creating. Experimental results on multiple datasets demonstrate that AdEval effectively reduces the impact of data contamination on evaluation outcomes, enhancing both the fairness and reliability of the evaluation process. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在大规模语料库上进行预训练，数据污染问题日益严重，这可能导致在评估期间对模型性能的潜在高估。为应对这一挑战，我们提出了AdEval（基于对齐的动态评估），这是一种动态数据评估方法，旨在减轻数据污染对评估可靠性的负面影响。AdEval通过动态提取关键知识点和主要思想，将生成的问题与静态数据的核心概念相-align，同时利用在线搜索提供相关知识点的详细解释，从而生成高质量的评估样本，具有稳固的知识支持。此外，AdEval还整合了控制问题数量和复杂性的机制，实现了动态对齐和灵活调整。这确保了生成的问题与静态数据的复杂性相匹配，同时支持不同的复杂度水平。根据布卢姆 taxonomy，AdEval在六个认知层面：记忆、理解、应用、分析、评价和创造上对LLMs进行多维度评估。在多个数据集上的实验结果表明，AdEval有效地减少了数据污染对评估结果的影响，提高了评估过程的公平性和可靠性。 

---
# Chain of Grounded Objectives: Bridging Process and Goal-oriented Prompting for Code Generation 

**Title (ZH)**: 基于对象的连续目标链：过程导向与目标导向提示生成在代码生成中的融合 

**Authors**: Sangyeop Yeo, Seung-won Hwang, Yu-Seung Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.13978)  

**Abstract**: The use of Large Language Models (LLMs) for code generation has gained significant attention in recent years. Existing methods often aim to improve the quality of generated code by incorporating additional contextual information or guidance into input prompts. Many of these approaches adopt sequential reasoning strategies, mimicking human-like step-by-step thinking. However, such strategies may constrain flexibility, as they do not always align with the structured characteristics of programming languages. This paper introduces the Chain of Grounded Objectives (CGO), a method that embeds functional objectives into input prompts to enhance code generation. By leveraging appropriately structured objectives as input and avoiding explicit sequential procedures, CGO adapts effectively to the structured nature of programming tasks. Empirical evaluations demonstrate that CGO effectively enhances code generation, addressing limitations of existing approaches. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在代码生成领域的应用引起了广泛关注。现有方法通常通过在输入提示中融入额外的上下文信息或指导来提高生成代码的质量。许多方法采用顺序推理策略，模仿类似人类逐步思考的方式。然而，这种策略可能限制了灵活性，因为它们不总是与编程语言的结构化特性相匹配。本文提出了一种称为“基于目标链”（Chain of Grounded Objectives, CGO）的方法，该方法将功能目标嵌入到输入提示中以增强代码生成。通过利用适当结构化的目标作为输入，并避免显式的顺序过程，CGO能够有效地适应编程任务的结构化特性。实证评估表明，CGO在代码生成方面表现出色，能够弥补现有方法的不足。 

---
# Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms 

**Title (ZH)**: 使用大型语言模型重新排序以减轻社交媒体平台上有害内容的暴露风险 

**Authors**: Rajvardhan Oak, Muhammad Haroon, Claire Jo, Magdalena Wojcieszak, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2501.13977)  

**Abstract**: Social media platforms utilize Machine Learning (ML) and Artificial Intelligence (AI) powered recommendation algorithms to maximize user engagement, which can result in inadvertent exposure to harmful content. Current moderation efforts, reliant on classifiers trained with extensive human-annotated data, struggle with scalability and adapting to new forms of harm. To address these challenges, we propose a novel re-ranking approach using Large Language Models (LLMs) in zero-shot and few-shot settings. Our method dynamically assesses and re-ranks content sequences, effectively mitigating harmful content exposure without requiring extensive labeled data. Alongside traditional ranking metrics, we also introduce two new metrics to evaluate the effectiveness of re-ranking in reducing exposure to harmful content. Through experiments on three datasets, three models and across three configurations, we demonstrate that our LLM-based approach significantly outperforms existing proprietary moderation approaches, offering a scalable and adaptable solution for harm mitigation. 

**Abstract (ZH)**: 社交媒体平台利用基于机器学习（ML）和人工智能（AI）的推荐算法来最大化用户参与度，这可能导致用户无意间接触到有害内容。目前依赖于大量人工标注数据训练的分类器进行的管控努力，在规模性和适应新型危害方面存在困难。为了解决这些问题，我们提出了一种新的重排方法，利用大语言模型（LLMs）在零样本和少样本设置中进行处理。我们的方法能够动态评估和重新排列内容序列，有效地减少有害内容的暴露，而无需大量标注数据。除了传统的排名指标，我们还引入了两种新的指标来评估重新排列在减少有害内容暴露方面的有效性。通过在三个数据集、三种模型和三种配置上的实验，我们证明了基于LLM的方法显著优于现有的专有管控方法，提供了一种可扩展且可适应的有害内容缓解解决方案。 

---
# Towards Safer Social Media Platforms: Scalable and Performant Few-Shot Harmful Content Moderation Using Large Language Models 

**Title (ZH)**: 向着更安全的社交媒体平台迈进：基于大规模语言模型的可扩展高效少量样本有害内容审核方法 

**Authors**: Akash Bonagiri, Lucen Li, Rajvardhan Oak, Zeerak Babar, Magdalena Wojcieszak, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2501.13976)  

**Abstract**: The prevalence of harmful content on social media platforms poses significant risks to users and society, necessitating more effective and scalable content moderation strategies. Current approaches rely on human moderators, supervised classifiers, and large volumes of training data, and often struggle with scalability, subjectivity, and the dynamic nature of harmful content (e.g., violent content, dangerous challenge trends, etc.). To bridge these gaps, we utilize Large Language Models (LLMs) to undertake few-shot dynamic content moderation via in-context learning. Through extensive experiments on multiple LLMs, we demonstrate that our few-shot approaches can outperform existing proprietary baselines (Perspective and OpenAI Moderation) as well as prior state-of-the-art few-shot learning methods, in identifying harm. We also incorporate visual information (video thumbnails) and assess if different multimodal techniques improve model performance. Our results underscore the significant benefits of employing LLM based methods for scalable and dynamic harmful content moderation online. 

**Abstract (ZH)**: 社交媒体平台上有害内容的普遍存在对用户和社会构成了重大风险，因此迫切需要更有效且可扩展的内容审核策略。当前的方法依赖于人类审核员、监督分类器和大量训练数据，常常难以应对可扩展性、主观性和有害内容的动态性（如暴力内容、危险挑战趋势等）带来的挑战。为填补这些空白，我们利用大型语言模型（LLMs）通过上下文学习进行少量样本动态内容审核。通过在多种LLM上进行大量实验，我们展示了我们的少量样本方法在识别危害方面可以超越现有的专有基准方法（如Perspective和OpenAI审核）以及此前的最佳少量样本学习方法。此外，我们还结合了视觉信息（视频缩略图），并评估了不同多模态技术是否能提升模型性能。我们的研究结果强调了使用基于LLM的方法进行在线有害内容的可扩展和动态审核的巨大优势。 

---
# A Spatio-temporal Graph Network Allowing Incomplete Trajectory Input for Pedestrian Trajectory Prediction 

**Title (ZH)**: 允许Incomplete轨迹输入的时空图网络在行人轨迹预测中的应用 

**Authors**: Juncen Long, Gianluca Bardaro, Simone Mentasti, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2501.13973)  

**Abstract**: Pedestrian trajectory prediction is important in the research of mobile robot navigation in environments with pedestrians. Most pedestrian trajectory prediction algorithms require the input historical trajectories to be complete. If a pedestrian is unobservable in any frame in the past, then its historical trajectory become incomplete, the algorithm will not predict its future trajectory. To address this limitation, we propose the STGN-IT, a spatio-temporal graph network allowing incomplete trajectory input, which can predict the future trajectories of pedestrians with incomplete historical trajectories. STGN-IT uses the spatio-temporal graph with an additional encoding method to represent the historical trajectories and observation states of pedestrians. Moreover, STGN-IT introduces static obstacles in the environment that may affect the future trajectories as nodes to further improve the prediction accuracy. A clustering algorithm is also applied in the construction of spatio-temporal graphs. Experiments on public datasets show that STGN-IT outperforms state of the art algorithms on these metrics. 

**Abstract (ZH)**: 行人轨迹预测对于移动机器人在有行人环境中导航的研究至关重要。大多数行人轨迹预测算法需要输入的历史轨迹是完整的。如果某个行人在过去的任何一帧中不可见，那么其历史轨迹将变得不完整，算法将无法预测其未来轨迹。为了解决这一局限性，我们提出了一种称为STGN-IT的时空图网络，它可以接受不完整的轨迹输入，并能够预测具有不完整历史轨迹的行人的未来轨迹。STGN-IT使用带有附加编码方法的空间-时间图来表示行人的历史轨迹及其观测状态。此外，STGN-IT在空间-时间图中引入了可能影响未来轨迹的静态障碍物作为节点，以进一步提高预测准确性。在构建空间-时间图的过程中也应用了聚类算法。实验表明，STGN-IT在这些评价指标上优于现有的算法。 

---
# FedDAG: Federated Domain Adversarial Generation Towards Generalizable Medical Image Analysis 

**Title (ZH)**: FedDAG: 联邦域对抗生成方法owards 具有普适性的医学图像分析 

**Authors**: Haoxuan Che, Yifei Wu, Haibo Jin, Yong Xia, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.13967)  

**Abstract**: Federated domain generalization aims to train a global model from multiple source domains and ensure its generalization ability to unseen target domains. {Due to the target domain being with unknown domain shifts, attempting to approximate these gaps by source domains may be the key to improving model generalization capability.} Existing works mainly focus on sharing and recombining local domain-specific attributes to increase data diversity and simulate potential domain shifts. {However, these methods may be insufficient since only the local attribute recombination can be hard to touch the out-of-distribution of global data.} In this paper, we propose a simple-yet-efficient framework named Federated Domain Adversarial Generation (FedDAG). {It aims to simulate the domain shift and improve the model generalization by adversarially generating novel domains different from local and global source domains.} Specifically, it generates novel-style images by maximizing the instance-level feature discrepancy between original and generated images and trains a generalizable task model by minimizing their feature discrepancy. {Further, we observed that FedDAG could cause different performance improvements for local models. It may be due to inherent data isolation and heterogeneity among clients, exacerbating the imbalance in their generalization contributions to the global model.} {Ignoring this imbalance can lead the global model's generalization ability to be sub-optimal, further limiting the novel domain generation procedure. } Thus, to mitigate this imbalance, FedDAG hierarchically aggregates local models at the within-client and across-client levels by using the sharpness concept to evaluate client model generalization contributions. {Extensive experiments across four medical benchmarks demonstrate FedDAG's ability to enhance generalization in federated medical scenarios.} 

**Abstract (ZH)**: 联邦领域泛化旨在通过多个源领域训练全球模型，并确保其在未见过的目标领域的泛化能力。{由于目标领域存在未知的领域偏移，试图通过源领域来近似这些差距可能是提高模型泛化能力的关键。}现有工作主要集中在共享和重组局部领域的特定属性，以增加数据多样性并模拟潜在的领域偏移。{然而，这些方法可能不够充分，因为仅仅通过局部属性重组很难触及全局数据的分布外性。}本文提出了一种简单而有效的框架，名为联邦领域对抗生成（FedDAG）。{其目标是模拟领域偏移并通过对抗生成不同于局部和全局源领域的新型领域来提高模型的泛化能力。}具体而言，该框架通过最大化原始图像和生成图像之间实例级的特征差异来生成新型风格的图像，并通过最小化它们的特征差异来训练可泛化的任务模型。{进一步观察到，FedDAG可能会在局部模型中引起不同的性能改进。这可能是由于客户端之间固有的数据隔离和异质性，加剧了它们对全球模型泛化贡献的不均衡性。} {忽略这种不均衡可能导致全球模型的泛化能力变得次优，进一步限制了新型领域的生成过程。}因此，为了缓解这种不均衡，FedDAG通过使用尖锐度概念评估客户端模型的泛化贡献，在客户端内和跨客户端级别进行分层次的聚合。{在四个医学基准测试中的广泛实验表明，FedDAG能够增强联邦医学场景中的泛化能力。} 

---
# ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification 

**Title (ZH)**: ZKLoRA：LoRA验证的高效零知识证明 

**Authors**: Bidhan Roy, Peter Potash, Marcos Villagra  

**Link**: [PDF](https://arxiv.org/pdf/2501.13965)  

**Abstract**: Low-Rank Adaptation (LoRA) is a widely adopted method for customizing large-scale language models. In distributed, untrusted training environments, an open source base model user may want to use LoRA weights created by an external contributor, leading to two requirements: (1) the base model user must confirm that the LoRA weights are effective when paired with the intended base model, and (2) the LoRA contributor must keep their proprietary weights private until compensation is assured.
We present ZKLoRA, a zero-knowledge verification protocol that relies on succinct proofs and our novel Multi-Party Inference procedure to verify LoRA-base model compatibility without exposing LoRA weights. ZKLoRA produces deterministic correctness guarantees and validates each LoRA module in only 1-2 seconds on state-of-the-art large language models. This low-latency approach enables nearly real-time verification and promotes secure collaboration among geographically decentralized teams and contract-based training pipelines. The protocol ensures that the delivered LoRA module works as claimed, safeguarding the contributor's intellectual property while providing the base model user with verification of compatibility and lineage. 

**Abstract (ZH)**: 低秩适应（LoRA）是一种广泛采用的方法，用于定制大规模语言模型。在分布式且不可信的训练环境中，开源基础模型的使用者可能会使用外部贡献者创建的LoRA权重，这引发了两个要求：（1）基础模型的使用者必须确认LoRA权重在与目标基础模型结合使用时的有效性；（2）LoRA贡献者必须在获得补偿保障之前保持其专有权重的私密性。

我们提出了一种基于零知识验证的协议ZKLoRA，该协议依赖于简洁的证明和我们新颖的多方推理过程，无需暴露LoRA权重即可验证LoRA基础模型的兼容性。ZKLoRA提供了确定性的正确性保证，并可在最先进的大规模语言模型上在1-2秒内验证每个LoRA模块。这种低延迟的方法能够实现几乎实时的验证，并促进地理分散团队和基于合同的训练管道中的安全协作。该协议确保交付的LoRA模块按预期工作，保护贡献者的知识产权，同时为基础模型的使用者提供兼容性和可追溯性的验证。

通过这种方式，ZKLoRA协议确保了LoRA模块按照声明的方式运行，既保护了贡献者的知识产权，又为基础模型使用者提供了兼容性和数据血统的验证。 

---
# Advancing the Understanding and Evaluation of AR-Generated Scenes: When Vision-Language Models Shine and Stumble 

**Title (ZH)**: 提升对AR生成场景的理解与评估：视觉-语言模型的亮点与不足 

**Authors**: Lin Duan, Yanming Xiu, Maria Gorlatova  

**Link**: [PDF](https://arxiv.org/pdf/2501.13964)  

**Abstract**: Augmented Reality (AR) enhances the real world by integrating virtual content, yet ensuring the quality, usability, and safety of AR experiences presents significant challenges. Could Vision-Language Models (VLMs) offer a solution for the automated evaluation of AR-generated scenes? Could Vision-Language Models (VLMs) offer a solution for the automated evaluation of AR-generated scenes? In this study, we evaluate the capabilities of three state-of-the-art commercial VLMs -- GPT, Gemini, and Claude -- in identifying and describing AR scenes. For this purpose, we use DiverseAR, the first AR dataset specifically designed to assess VLMs' ability to analyze virtual content across a wide range of AR scene complexities. Our findings demonstrate that VLMs are generally capable of perceiving and describing AR scenes, achieving a True Positive Rate (TPR) of up to 93\% for perception and 71\% for description. While they excel at identifying obvious virtual objects, such as a glowing apple, they struggle when faced with seamlessly integrated content, such as a virtual pot with realistic shadows. Our results highlight both the strengths and the limitations of VLMs in understanding AR scenarios. We identify key factors affecting VLM performance, including virtual content placement, rendering quality, and physical plausibility. This study underscores the potential of VLMs as tools for evaluating the quality of AR experiences. 

**Abstract (ZH)**: 增强现实（AR）通过集成虚拟内容来增强现实世界，但确保AR体验的质量、可用性和安全性面临着巨大的挑战。视觉-语言模型（VLMs）能否提供一种自动评估AR生成场景的解决方案？视觉-语言模型（VLMs）能否提供一种自动评估AR生成场景的解决方案？本研究旨在评估三款最先进的商业VLMs——GPT、Gemini和Claude——在识别和描述AR场景方面的能力。为此，我们使用了DiverseAR数据集，这是第一个专门用于评估VLMs分析不同复杂度AR场景中虚拟内容能力的数据集。研究结果表明，VLMs通常能够感知和描述AR场景，感知的真阳性率（TPR）最高可达93%，描述的TPR为71%。尽管它们在识别明显的虚拟对象（如发光的苹果）方面表现出色，但在处理无缝集成的内容（如具有逼真阴影的虚拟花瓶）时则显得力不从心。我们的结果突显了VLMs在理解AR场景方面的强项和局限性。我们确定了影响VLM性能的关键因素，包括虚拟内容的放置、渲染质量和物理合理性。本研究强调了VLMs作为一种评估AR体验质量工具的潜力。 

---
# Adaptive Cyber-Attack Detection in IIoT Using Attention-Based LSTM-CNN Models 

**Title (ZH)**: 使用基于注意力的LSTM-CNN模型在工业物联网中进行自适应网络攻击检测 

**Authors**: Afrah Gueriani, Hamza Kheddar, Ahmed Cherif Mazari  

**Link**: [PDF](https://arxiv.org/pdf/2501.13962)  

**Abstract**: The rapid expansion of the industrial Internet of things (IIoT) has introduced new challenges in securing critical infrastructures against sophisticated cyberthreats. This study presents the development and evaluation of an advanced Intrusion detection (IDS) based on a hybrid LSTM-convolution neural network (CNN)-Attention architecture, specifically designed to detect and classify cyberattacks in IIoT environments. The research focuses on two key classification tasks: binary and multi-class classification. The proposed models was rigorously tested using the Edge-IIoTset dataset. To mitigate the class imbalance in the dataset, the synthetic minority over-sampling technique (SMOTE) was employed to generate synthetic samples for the underrepresented classes. This ensured that the model could learn effectively from all classes, thereby improving the overall classification performance. Through systematic experimentation, various deep learning (DL) models were compared, ultimately demonstrating that the LSTM-CNN-Attention model consistently outperformed others across key performance metrics. In binary classification, the model achieved near-perfect accuracy, while in multi-class classification, it maintained a high accuracy level (99.04%), effectively categorizing different attack types with a loss value of 0.0220%. 

**Abstract (ZH)**: 工业互联网（IIoT）的迅速扩展引入了新的挑战，需要应对复杂网络威胁对关键基础设施的攻击。本研究提出了一个基于混合长短期记忆（LSTM）-卷积神经网络（CNN）-注意力机制的高级入侵检测（IDS）系统，并特别设计用于检测和分类IIoT环境中的网络攻击。研究主要集中在两类分类任务：二元分类和多类分类。所提出的模型通过使用Edge-IIoTset数据集进行了严格的测试。为缓解数据集中的类别不平衡问题，采用了合成少数类过采样技术（SMOTE）生成合成样本以补充欠代表的类。这确保模型能够从所有类学习，从而提高整体分类性能。通过对多种深度学习（DL）模型进行系统性实验，结果表明，LSTM-CNN-Attention模型在关键性能指标上始终优于其他模型。在二元分类中，模型实现了近乎完美的准确率；而在多类分类中，模型维持了高水平的准确率（99.04%），分类损失值为0.0220%，有效分类了不同类型的攻击。 

---
# Assisting Mathematical Formalization with A Learning-based Premise Retriever 

**Title (ZH)**: 使用基于学习的前提检索辅助数学形式化 

**Authors**: Yicheng Tao, Haotian Liu, Shanwen Wang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13959)  

**Abstract**: Premise selection is a crucial yet challenging step in mathematical formalization, especially for users with limited experience. Due to the lack of available formalization projects, existing approaches that leverage language models often suffer from data scarcity. In this work, we introduce an innovative method for training a premise retriever to support the formalization of mathematics. Our approach employs a BERT model to embed proof states and premises into a shared latent space. The retrieval model is trained within a contrastive learning framework and incorporates a domain-specific tokenizer along with a fine-grained similarity computation method. Experimental results show that our model is highly competitive compared to existing baselines, achieving strong performance while requiring fewer computational resources. Performance is further enhanced through the integration of a re-ranking module. To streamline the formalization process, we will release a search engine that enables users to query Mathlib theorems directly using proof states, significantly improving accessibility and efficiency. Codes are available at this https URL. 

**Abstract (ZH)**: 前提选择是数学形式化过程中的关键但具有挑战性的步骤，尤其是对于经验有限的用户而言。由于可用的形式化项目缺乏，现有利用语言模型的方法往往受到数据稀缺性的困扰。在此项工作中，我们介绍了一种创新的方法，用于训练一个前提检索器以支持数学的形式化。我们的方法采用BERT模型将证明状态和前提嵌入到共享的潜在空间中。检索模型在对比学习框架下进行训练，并结合了领域特定的分词器以及细粒度的相似性计算方法。实验证明，我们的模型在性能上与现有的基线方法相当，同时需要较少的计算资源。通过引入重排序模块，性能进一步提升。为了简化形式化过程，我们将发布一个搜索引擎，使用户可以直接使用证明状态查询Mathlib定理，从而显著提高可访问性和效率。代码可在以下链接获取：this https URL。 

---
# A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models 

**Title (ZH)**: 面向定制化大规模语言模型的图检索增强生成综述 

**Authors**: Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13958)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in a wide range of tasks, yet their application to specialized domains remains challenging due to the need for deep expertise. Retrieval-augmented generation (RAG) has emerged as a promising solution to customize LLMs for professional fields by seamlessly integrating external knowledge bases, enabling real-time access to domain-specific expertise during inference. Despite its potential, traditional RAG systems, based on flat text retrieval, face three critical challenges: (i) complex query understanding in professional contexts, (ii) difficulties in knowledge integration across distributed sources, and (iii) system efficiency bottlenecks at scale. This survey presents a systematic analysis of Graph-based Retrieval-Augmented Generation (GraphRAG), a new paradigm that revolutionizes domain-specific LLM applications. GraphRAG addresses traditional RAG limitations through three key innovations: (i) graph-structured knowledge representation that explicitly captures entity relationships and domain hierarchies, (ii) efficient graph-based retrieval techniques that enable context-preserving knowledge retrieval with multihop reasoning ability, and (iii) structure-aware knowledge integration algorithms that leverage retrieved knowledge for accurate and logical coherent generation of LLMs. In this survey, we systematically analyze the technical foundations of GraphRAG and examine current implementations across various professional domains, identifying key technical challenges and promising research directions. All the related resources of GraphRAG, including research papers, open-source data, and projects, are collected for the community in \textcolor{blue}{\url{this https URL}}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展现了惊人的能力，然而将其应用于专业化领域仍然具有挑战性，因为需要深厚的专业知识。检索增强生成（RAG）作为一种前景广阔的方法，通过无缝集成外部知识库，实现在推断过程中对领域特定专业知识的实时访问，从而为专业领域量身定制LLMs。尽管具有潜力，传统RAG系统基于扁平文本检索，面临三个关键挑战：（i）专业语境下的复杂查询理解，（ii）跨分布式来源的知识整合难题，以及（iii）系统在大规模应用中的效率瓶颈。本文综述了图检索增强生成（GraphRAG）这一新范式，该范式通过三种关键创新解决了传统RAG的局限性：（i）以图结构表示知识并明确捕捉实体关系和领域层次结构，（ii）高效的基于图的检索技术，能够支持上下文保持的知识检索和多跳推理能力，以及（iii）结构感知的知识整合算法，利用检索到的知识为LLMs生成准确且合乎逻辑的内容。在本文综述中，我们系统地分析了GraphRAG的技术基础，并考察了其在各种专业领域的现有实现，识别出关键的技术挑战和有前景的研究方向。所有与GraphRAG相关的资源，包括研究论文、开源数据和项目，已收集于此处 \textcolor{blue}{\[https://example.com\]}，供社区参考。 

---
# Benchmarking Generative AI for Scoring Medical Student Interviews in Objective Structured Clinical Examinations (OSCEs) 

**Title (ZH)**: Benchmarking 生成式人工智能在客观结构化临床考试（OSCEs）中评分医学学生面试中的应用 

**Authors**: Jadon Geathers, Yann Hicke, Colleen Chan, Niroop Rajashekar, Justin Sewell, Susannah Cornes, Rene Kizilcec, Dennis Shung  

**Link**: [PDF](https://arxiv.org/pdf/2501.13957)  

**Abstract**: Introduction. Objective Structured Clinical Examinations (OSCEs) are widely used to assess medical students' communication skills, but scoring interview-based assessments is time-consuming and potentially subject to human bias. This study explored the potential of large language models (LLMs) to automate OSCE evaluations using the Master Interview Rating Scale (MIRS).
Methods. We compared the performance of four state-of-the-art LLMs (GPT-4o, Claude 3.5, Llama 3.1, and Gemini 1.5 Pro) in evaluating OSCE transcripts across all 28 items of the MIRS under the conditions of zero-shot, chain-of-thought (CoT), few-shot, and multi-step prompting. The models were benchmarked against a dataset of 10 OSCE cases with 174 expert consensus scores available. Model performance was measured using three accuracy metrics (exact, off-by-one, thresholded).
Results. Averaging across all MIRS items and OSCE cases, LLMs performed with low exact accuracy (0.27 to 0.44), and moderate to high off-by-one accuracy (0.67 to 0.87) and thresholded accuracy (0.75 to 0.88). A zero temperature parameter ensured high intra-rater reliability ($\alpha = 0.98$ for GPT-4o). CoT, few-shot, and multi-step techniques proved valuable when tailored to specific assessment items. The performance was consistent across MIRS items independent of encounter phases and communication domains.
Conclusion. We demonstrated the feasibility of AI-assisted OSCE evaluation and provided benchmarking of multiple LLMs across multiple prompt techniques. Our work provides a baseline performance assessment for LLMs that lays a foundation for future research in automated assessment of clinical communication skills. 

**Abstract (ZH)**: 介绍。结构化临床考试（OSCEs）广泛用于评估医学生沟通技能，但评分面试评估耗时且可能受到人类偏见的影响。本研究探讨了大型语言模型（LLMs）在使用《主面试评分量表》（MIRS）自动化OSCE评估方面的潜力。

方法。我们比较了四款最先进的LLMs（GPT-4o、Claude 3.5、Llama 3.1和Gemini 1.5 Pro）在零样本、链式思考（CoT）、少样本和多步骤提示条件下评估OSCE转录材料的性能。这些模型在包含10个OSCE案例和174个专家一致评分的数据集上进行了基准测试。模型性能通过三种准确性指标（精确匹配、相差一、阈值）进行了测量。

结果。总体而言，LLMs在所有MIRS项目和OSCE案例中的精确匹配准确性较低（0.27到0.44），但在相差一和阈值准确性方面中等到高（0.67到0.87 和0.75到0.88）。GPT-4o使用零温度参数确保了高评分者内一致度（$\alpha = 0.98$）。链式思考、少样本和多步骤技术在针对特定评估项进行调整时显得尤为重要。性能在MIRS项目上保持一致，不受遇诊阶段和沟通领域的影响。

结论。我们展示了人工智能辅助OSCE评估的可行性，并对多种LLMs在多种提示技术下的表现进行了基准测试。我们的研究为未来在临床沟通技能自动化评估领域的研究奠定了基础，并提供了LLMs基线性能评估，为其在该领域的进一步应用提供了依据。 

---
# Zep: A Temporal Knowledge Graph Architecture for Agent Memory 

**Title (ZH)**: ZEP：一种用于智能体记忆的时间知识图架构 

**Authors**: Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef  

**Link**: [PDF](https://arxiv.org/pdf/2501.13956)  

**Abstract**: We introduce Zep, a novel memory layer service for AI agents that outperforms the current state-of-the-art system, MemGPT, in the Deep Memory Retrieval (DMR) benchmark. Additionally, Zep excels in more comprehensive and challenging evaluations than DMR that better reflect real-world enterprise use cases. While existing retrieval-augmented generation (RAG) frameworks for large language model (LLM)-based agents are limited to static document retrieval, enterprise applications demand dynamic knowledge integration from diverse sources including ongoing conversations and business data. Zep addresses this fundamental limitation through its core component Graphiti -- a temporally-aware knowledge graph engine that dynamically synthesizes both unstructured conversational data and structured business data while maintaining historical relationships. In the DMR benchmark, which the MemGPT team established as their primary evaluation metric, Zep demonstrates superior performance (94.8% vs 93.4%). Beyond DMR, Zep's capabilities are further validated through the more challenging LongMemEval benchmark, which better reflects enterprise use cases through complex temporal reasoning tasks. In this evaluation, Zep achieves substantial results with accuracy improvements of up to 18.5% while simultaneously reducing response latency by 90% compared to baseline implementations. These results are particularly pronounced in enterprise-critical tasks such as cross-session information synthesis and long-term context maintenance, demonstrating Zep's effectiveness for deployment in real-world applications. 

**Abstract (ZH)**: 我们介绍了Zep，这是一种新型的内存层服务，与当前最先进的系统MemGPT在深度记忆检索（DMR）基准测试中表现更优。此外，Zep在比DMR更全面和更具挑战性的评估中表现出色，这些评估更能反映实际企业应用场景。现有的基于大型语言模型（LLM）的抽取增强生成（RAG）框架仅限于静态文档检索，而企业应用需要从各种动态知识源中进行动态集成，包括正在进行的对话和业务数据。Zep通过其核心组件Graphiti——一个能感知时间的知识图谱引擎来解决这一根本性限制。Graphiti能动态地合成未结构化的对话数据和结构化的业务数据，同时保持历史关系。在MemGPT团队建立为主要评估指标的DMR基准测试中，Zep展示了卓越的表现（94.8% vs 93.4%）。Zep的能力还在更具有挑战性的LongMemEval基准测试中得到了验证，该测试通过复杂的时序推理任务更好地反映了企业应用场景。在这种评估中，Zep的准确率提高了高达18.5%，同时将响应延迟降低了90%。这些结果在诸如会话间信息合成和长期上下文维护等关键企业任务中尤为明显，表明Zep在实际应用中的部署效果。 

---
# Guided Persona-based AI Surveys: Can we replicate personal mobility preferences at scale using LLMs? 

**Title (ZH)**: 基于引导人物的AI调查：我们能否使用大语言模型（LLMs）大规模复制个人的出行偏好？ 

**Authors**: Ioannis Tzachristas, Santhanakrishnan Narayanan, Constantinos Antoniou  

**Link**: [PDF](https://arxiv.org/pdf/2501.13955)  

**Abstract**: This study explores the potential of Large Language Models (LLMs) to generate artificial surveys, with a focus on personal mobility preferences in Germany. By leveraging LLMs for synthetic data creation, we aim to address the limitations of traditional survey methods, such as high costs, inefficiency and scalability challenges. A novel approach incorporating "Personas" - combinations of demographic and behavioural attributes - is introduced and compared to five other synthetic survey methods, which vary in their use of real-world data and methodological complexity. The MiD 2017 dataset, a comprehensive mobility survey in Germany, serves as a benchmark to assess the alignment of synthetic data with real-world patterns. The results demonstrate that LLMs can effectively capture complex dependencies between demographic attributes and preferences while offering flexibility to explore hypothetical scenarios. This approach presents valuable opportunities for transportation planning and social science research, enabling scalable, cost-efficient and privacy-preserving data generation. 

**Abstract (ZH)**: 本研究探讨了大型语言模型（LLMs）在生成虚拟调查方面的潜力，重点关注德国的个人移动偏好。通过利用LLMs创建合成数据，我们旨在解决传统调查方法的局限性，如成本高昂、效率低下和可扩展性挑战。提出了结合“角色”（Personas）的新方法，角色由人口统计学和行为特征的组合构成，并将其与五种其他合成调查方法进行比较，这些方法在使用真实数据和方法复杂性方面有所不同。以德国的MiD 2017移动调查集为基准，评估合成数据与真实模式的一致性。研究结果表明，LLMs能够有效地捕捉人口统计学特征与偏好之间的复杂关系，并提供探索假设场景的灵活性。这种方法为交通规划和社会科学研究提供了有价值的机遇，能够实现大规模、成本效益高且隐私保护的数据生成。 

---
# Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents 

**Title (ZH)**: Chat3GPP：一个开源检索增强生成框架用于3GPP文档 

**Authors**: Long Huang, Ming Zhao, Limin Xiao, Xiujun Zhang, Jungang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13954)  

**Abstract**: The 3rd Generation Partnership Project (3GPP) documents is key standards in global telecommunications, while posing significant challenges for engineers and researchers in the telecommunications field due to the large volume and complexity of their contents as well as the frequent updates. Large language models (LLMs) have shown promise in natural language processing tasks, but their general-purpose nature limits their effectiveness in specific domains like telecommunications. To address this, we propose Chat3GPP, an open-source retrieval-augmented generation (RAG) framework tailored for 3GPP specifications. By combining chunking strategies, hybrid retrieval and efficient indexing methods, Chat3GPP can efficiently retrieve relevant information and generate accurate responses to user queries without requiring domain-specific fine-tuning, which is both flexible and scalable, offering significant potential for adapting to other technical standards beyond 3GPP. We evaluate Chat3GPP on two telecom-specific datasets and demonstrate its superior performance compared to existing methods, showcasing its potential for downstream tasks like protocol generation and code automation. 

**Abstract (ZH)**: 第三代合作伙伴项目（3GPP）文档是全球电信领域的重要标准，但由于其内容巨大且复杂，以及频繁的更新，给电信领域的工程师和研究人员带来了显著的挑战。大规模语言模型（LLMs）在自然语言处理任务中展现出了巨大潜力，但其通用性质限制了其在特定领域如电信中的有效性。为了解决这一问题，我们提出了一种名为Chat3GPP的开源检索增强生成（RAG）框架，专门针对3GPP规范。通过结合分块策略、混合检索和高效的索引方法，Chat3GPP能够高效地检索相关信息，并生成准确的用户查询回答，而无需特定领域的微调。这种框架既灵活又可扩展，为适应其他技术标准提供了显著的潜力。我们对Chat3GPP进行了两种电信特定数据集的评估，并展示了其在现有方法中的优越性能，证明了其在协议生成和代码自动化等下游任务中的潜在应用价值。 

---
# Redundancy Principles for MLLMs Benchmarks 

**Title (ZH)**: 多模态大规模语言模型基准中的冗余原则 

**Authors**: Zicheng Zhang, Xiangyu Zhao, Xinyu Fang, Chunyi Li, Xiaohong Liu, Xiongkuo Min, Haodong Duan, Kai Chen, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2501.13953)  

**Abstract**: With the rapid iteration of Multi-modality Large Language Models (MLLMs) and the evolving demands of the field, the number of benchmarks produced annually has surged into the hundreds. The rapid growth has inevitably led to significant redundancy among benchmarks. Therefore, it is crucial to take a step back and critically assess the current state of redundancy and propose targeted principles for constructing effective MLLM benchmarks. In this paper, we focus on redundancy from three key perspectives: 1) Redundancy of benchmark capability dimensions, 2) Redundancy in the number of test questions, and 3) Cross-benchmark redundancy within specific domains. Through the comprehensive analysis over hundreds of MLLMs' performance across more than 20 benchmarks, we aim to quantitatively measure the level of redundancy lies in existing MLLM evaluations, provide valuable insights to guide the future development of MLLM benchmarks, and offer strategies to refine and address redundancy issues effectively. 

**Abstract (ZH)**: 随着多模态大型语言模型（MLLMs）的快速迭代和领域需求的不断演变，每年发布的基准数量激增至数百个。这一快速增长不可避免地导致了基准间存在显著的重复性。因此，回过头来对现有的重复性进行全面评估，并提出针对有效构建MLLM基准的原则至关重要。本文从三个关键视角出发探讨重复性问题：1）基准能力维度的重复性；2）测试问题数量的重复性；3）特定领域内不同基准间的重复性。通过对超过20个基准上数百个MLLMs的综合性能分析，我们旨在量化现有MLLM评估中的重复性水平，为未来MLLM基准的发展提供有价值的见解，并提出有效精简和解决重复性问题的策略。 

---
# The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility? 

**Title (ZH)**: 大型语言模型中的双重用途困境：增强道德能力是否会导致效用下降？ 

**Authors**: Yiyi Zhang, Xingyu Chen, Kexin Chen, Yuyang Du, Xilin Dang, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2501.13952)  

**Abstract**: Recent years have witnessed extensive efforts to enhance Large Language Models (LLMs) across various domains, alongside growing attention to their ethical implications. However, a critical challenge remains largely overlooked: LLMs must balance between rejecting harmful requests for safety and accommodating legitimate ones for utility. This paper presents a Direct Preference Optimization (DPO) based alignment framework that achieves better overall performance by addressing this ethical-utility trade-off, using chemical domain applications as a proof-of-concept. Our alignment pipeline starts with a GPT-assisted three-phase data generation scheme, in which we create LibraChemQA, a chemical question-answering dataset comprising 31.6k triplet instances. By incorporating an innovative balanced seed in the data generation process, our framework systematically considers both legitimate and illegitimate requests. The framework also introduces a rephrasing mechanism for efficient data augmentation that enhances the model's chemical comprehension. We further develop a novel hybrid evaluation scheme with LLM judges for precise assessment of both safety and utility. Experimental results demonstrate our model's substantial improvements in overall performance where both safety and utility are considered - our resulting model, LibraChem, outperforms leading LLMs including Claude-3, GPT-4o, and LLaMA-3 by margins of 13.44%, 7.16%, and 7.10% respectively on our released benchmark. 

**Abstract (ZH)**: 近年来，各界对增强大型语言模型（LLMs）的积极性不断增长，同时对其伦理影响也给予了越来越多的关注。然而，一个关键挑战仍未得到充分重视：LLMs必须在确保安全性的同时平衡满足合法请求的实用性需求。本文提出了一种基于直接偏好优化（DPO）的对齐框架，通过解决这种伦理-实用性权衡问题，实现了更好的整体性能，并以化学领域应用为例进行了验证。我们的对齐流水线始于一个使用GPT辅助的三阶段数据生成方案，创建了LibraChemQA数据集，包含31,600个三元组实例。通过在数据生成过程中引入创新的平衡种子，我们的框架系统地考虑了合法和非法请求。此外，该框架还引入了一种高效的数据增强重述机制，以提高模型对化学的理解能力。我们进一步开发了一种新的混合评估方案，使用LLM评判员来精确评估安全性和实用性。实验结果表明，我们的模型在同时考虑安全性和实用性的情况下取得了显著的性能提升——我们的LibraChem模型在我们的基准测试中分别比Claude-3、GPT-4o和LLaMA-3高出13.44%、7.16%和7.10%。 

---
# A Layered Multi-Expert Framework for Long-Context Mental Health Assessments 

**Title (ZH)**: 一种分层多专家框架，用于长上下文心理健康评估 

**Authors**: Jinwen Tang, Qiming Guo, Wenbo Sun, Yi Shang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13951)  

**Abstract**: Long-form mental health assessments pose unique challenges for large language models (LLMs), which often exhibit hallucinations or inconsistent reasoning when handling extended, domain-specific contexts. We introduce Stacked Multi-Model Reasoning (SMMR), a layered framework that leverages multiple LLMs and specialized smaller models as coequal 'experts'. Early layers isolate short, discrete subtasks, while later layers integrate and refine these partial outputs through more advanced long-context models. We evaluate SMMR on the DAIC-WOZ depression-screening dataset and 48 curated case studies with psychiatric diagnoses, demonstrating consistent improvements over single-model baselines in terms of accuracy, F1-score, and PHQ-8 error reduction. By harnessing diverse 'second opinions', SMMR mitigates hallucinations, captures subtle clinical nuances, and enhances reliability in high-stakes mental health assessments. Our findings underscore the value of multi-expert frameworks for more trustworthy AI-driven screening. 

**Abstract (ZH)**: 长篇心理健康评估对大型语言模型（LLMs）提出了独特的挑战，这些模型在处理扩展的、领域特定的背景时经常表现出幻觉或不一致的推理现象。我们提出了堆叠多模型推理（Stacked Multi-Model Reasoning, SMMR），这是一种分层框架，利用多个LLMs和专门的小型模型作为同等的“专家”。早期层隔离短暂的、离散的子任务，而后期层则通过更为高级的长上下文模型整合并精炼这些部分输出。我们在DAIC-WOZ抑郁症筛查数据集和48例经过精心筛选的精神疾病病例中对SMMR进行了评估，展示了在准确率、F1分数和PHQ-8误差减少方面相对于单模型基线的一致性改进。通过利用多样化的“第二意见”，SMMR减轻了幻觉现象，捕捉到了细微的临床差异，并增强了高度敏感的心理健康评估的可靠性。我们的研究结果强调了多专家框架对于更可靠的人工智能驱动筛查的价值。 

---
# Can OpenAI o1 Reason Well in Ophthalmology? A 6,990-Question Head-to-Head Evaluation Study 

**Title (ZH)**: OpenAI在眼科领域推理能力如何？一项针对6,990个问题的头对头评估研究 

**Authors**: Sahana Srinivasan, Xuguang Ai, Minjie Zou, Ke Zou, Hyunjae Kim, Thaddaeus Wai Soon Lo, Krithi Pushpanathan, Yiming Kong, Anran Li, Maxwell Singer, Kai Jin, Fares Antaki, David Ziyou Chen, Dianbo Liu, Ron A. Adelman, Qingyu Chen, Yih Chung Tham  

**Link**: [PDF](https://arxiv.org/pdf/2501.13949)  

**Abstract**: Question: What is the performance and reasoning ability of OpenAI o1 compared to other large language models in addressing ophthalmology-specific questions?
Findings: This study evaluated OpenAI o1 and five LLMs using 6,990 ophthalmological questions from MedMCQA. O1 achieved the highest accuracy (0.88) and macro-F1 score but ranked third in reasoning capabilities based on text-generation metrics. Across subtopics, o1 ranked first in ``Lens'' and ``Glaucoma'' but second to GPT-4o in ``Corneal and External Diseases'', ``Vitreous and Retina'' and ``Oculoplastic and Orbital Diseases''. Subgroup analyses showed o1 performed better on queries with longer ground truth explanations.
Meaning: O1's reasoning enhancements may not fully extend to ophthalmology, underscoring the need for domain-specific refinements to optimize performance in specialized fields like ophthalmology. 

**Abstract (ZH)**: 问题：OpenAI o1 在回答眼科特定问题时的表现及其推理能力与其它大型语言模型相比如何？

发现：本研究使用来自 MedMCQA 的 6,990 个眼科问题，对 OpenAI o1 和五种其他大型语言模型（LLM）进行了评估。o1 在准确率（0.88）和宏观 F1 分数方面表现最佳，但在基于文本生成的指标上推理能力排名第三。在子主题方面，o1 在“晶状体”和“青光眼”领域排名第一，但在“角膜和外眼疾病”、“玻璃体和视网膜疾病”以及“眼眶和眼整形疾病”领域分别被 GPT-4o 排名第二。亚组分析显示，o1 在具有较长真实解释的查询上表现更好。

意义：o1 的推理增强可能未能完全扩展到眼科领域，这强调了在专业领域如眼科进行领域特定优化的重要性，以优化性能。 

---
# Longitudinal Abuse and Sentiment Analysis of Hollywood Movie Dialogues using LLMs 

**Title (ZH)**: 使用大规模语言模型 longitudinal 滥用及其对好莱坞电影对白情感分析的研究 

**Authors**: Rohitash Chandra, Guoxiang Ren, Group-H  

**Link**: [PDF](https://arxiv.org/pdf/2501.13948)  

**Abstract**: Over the past decades, there has been an increasing concern about the prevalence of abusive and violent content in Hollywood movies. This study uses Large Language Models (LLMs) to explore the longitudinal abuse and sentiment analysis of Hollywood Oscar and blockbuster movie dialogues from 1950 to 2024. By employing fine-tuned LLMs, we analyze subtitles for over a thousand movies categorised into four genres to examine the trends and shifts in emotional and abusive content over the past seven decades. Our findings reveal significant temporal changes in movie dialogues, which reflect broader social and cultural influences. Overall, the emotional tendencies in the films are diverse, and the detection of abusive content also exhibits significant fluctuations. The results show a gradual rise in abusive content in recent decades, reflecting social norms and regulatory policy changes. Genres such as thrillers still present a higher frequency of abusive content that emphasises the ongoing narrative role of violence and conflict. At the same time, underlying positive emotions such as humour and optimism remain prevalent in most of the movies. Furthermore, the gradual increase of abusive content in movie dialogues has been significant over the last two decades, where Oscar-nominated movies overtook the top ten blockbusters. 

**Abstract (ZH)**: 在过去几十年中，好莱坞电影中虐待和暴力内容的普遍存在引起了越来越多的关注。本研究使用大型语言模型（LLMs）探索从1950年到2024年奥斯卡及大片电影对话的情感和虐待内容的纵向变化。通过使用微调后的大型语言模型，我们对四类超过1000部电影的字幕进行了分析，以研究过去七十年中情感和虐待内容的趋势和变化。研究发现，电影对话存在显著的时间变化，这些变化反映了更广泛的社会和文化影响。总体而言，电影中情感倾向多样，虐待内容的检测也表现出显著的变化。结果表明，近年来虐待内容呈现逐渐上升的趋势，反映出社会规范和监管政策的变化。如惊悚片等类型仍然频繁出现强调暴力和冲突的虐待内容。同时，大多数电影中仍保持着积极情感如幽默和乐观的普遍性。此外，电影对话中的虐待内容在过去二十年中呈现出显著的增长趋势，其中奥斯卡提名电影超过了前十部大片。 

---
# A Comprehensive Survey on Integrating Large Language Models with Knowledge-Based Methods 

**Title (ZH)**: 大型语言模型与知识基础方法整合综述 

**Authors**: Lilian Some, Wenli Yang, Michael Bain, Byeong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13947)  

**Abstract**: The rapid development of artificial intelligence has brought about substantial advancements in the field. One promising direction is the integration of Large Language Models (LLMs) with structured knowledge-based systems. This approach aims to enhance AI capabilities by combining the generative language understanding of LLMs with the precise knowledge representation of structured systems. This survey explores the synergy between LLMs and knowledge bases, focusing on real-world applications and addressing associated technical, operational, and ethical challenges. Through a comprehensive literature review, the study identifies critical issues and evaluates existing solutions. The paper highlights the benefits of integrating generative AI with knowledge bases, including improved data contextualization, enhanced model accuracy, and better utilization of knowledge resources. The findings provide a detailed overview of the current state of research, identify key gaps, and offer actionable recommendations. These insights contribute to advancing AI technologies and support their practical deployment across various sectors. 

**Abstract (ZH)**: 人工智能的迅猛发展在该领域带来了显著的进步。一个充满潜力的方向是将大语言模型（LLMs）与结构化知识库系统相结合。这种方法旨在通过将LLMs的生成语言理解和结构化系统精确的知识表示相结合来增强AI的能力。这篇综述探讨了LLMs与知识库之间的协同作用，关注实际应用，并解决相关的技术、操作和伦理挑战。通过全面的文献综述，研究确定了关键问题并评估了现有解决方案。论文强调了将生成型AI与知识库集成所带来的好处，包括改进的数据上下文化、增强的模型准确性以及更好地利用知识资源。研究结果提供了当前研究状态的详细概述，指出了关键空白，并提供了可行的建议。这些见解有助于推进AI技术的发展，并支持其在各个领域的实际部署。 

---
# Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks 

**Title (ZH)**: 使用代理AI基座语言框架减轻幻觉现象 

**Authors**: Diego Gosmar, Deborah A. Dahl  

**Link**: [PDF](https://arxiv.org/pdf/2501.13946)  

**Abstract**: Hallucinations remain a significant challenge in current Generative AI models, undermining trust in AI systems and their reliability. This study investigates how orchestrating multiple specialized Artificial Intelligent Agents can help mitigate such hallucinations, with a focus on systems leveraging Natural Language Processing (NLP) to facilitate seamless agent interactions. To achieve this, we design a pipeline that introduces over three hundred prompts, purposefully crafted to induce hallucinations, into a front-end agent. The outputs are then systematically reviewed and refined by second- and third-level agents, each employing distinct large language models and tailored strategies to detect unverified claims, incorporate explicit disclaimers, and clarify speculative content. Additionally, we introduce a set of novel Key Performance Indicators (KPIs) specifically designed to evaluate hallucination score levels. A dedicated fourth-level AI agent is employed to evaluate these KPIs, providing detailed assessments and ensuring accurate quantification of shifts in hallucination-related behaviors. A core component of this investigation is the use of the OVON (Open Voice Network) framework, which relies on universal NLP-based interfaces to transfer contextual information among agents. Through structured JSON messages, each agent communicates its assessment of the hallucination likelihood and the reasons underlying questionable content, thereby enabling the subsequent stage to refine the text without losing context. The results demonstrate that employing multiple specialized agents capable of interoperating with each other through NLP-based agentic frameworks can yield promising outcomes in hallucination mitigation, ultimately bolstering trust within the AI community. 

**Abstract (ZH)**: 幻觉仍然是当前生成型人工智能模型中的重大挑战，削弱了人们对人工智能系统及其可靠性的信任。本研究探讨了如何通过协调多个专门的人工智能代理来减轻这样的幻觉，重点关注利用自然语言处理（NLP）促进代理间无缝交互的系统。为了实现这一目标，我们设计了一条管线，将超过三百个旨在诱发幻觉的提示引入前端代理。然后，这些输出由第二级和第三级代理系统地审查和修订，每级代理使用不同的大型语言模型和定制策略来检测未验证的声明、纳入明确的免责声明，并澄清推测性内容。此外，我们引入了一组新的关键绩效指标（KPI），专门设计用于评估幻觉评分水平。一个专门的第四级AI代理用于评估这些KPI，提供详细的评估并确保幻觉相关行为变化的准确量化。本研究的核心成分为采用OVON（开放语音网络）框架，该框架依赖于基于通用NLP的接口在代理之间传递上下文信息。通过结构化的JSON消息，每个代理传达其对幻觉可能性的评估以及可疑内容的原因，从而使得后一阶段能够不丢失上下文地调整文本。研究结果表明，通过利用能够通过基于NLP的代理框架相互协作的多个专门代理，可以在幻觉减轻方面取得令人鼓舞的结果，最终增强人工智能社区的信任度。 

---
# Self-Explanation in Social AI Agents 

**Title (ZH)**: 社会人工智能代理中的自我解释 

**Authors**: Rhea Basappa, Mustafa Tekman, Hong Lu, Benjamin Faught, Sandeep Kakar, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2501.13945)  

**Abstract**: Social AI agents interact with members of a community, thereby changing the behavior of the community. For example, in online learning, an AI social assistant may connect learners and thereby enhance social interaction. These social AI assistants too need to explain themselves in order to enhance transparency and trust with the learners. We present a method of self-explanation that uses introspection over a self-model of an AI social assistant. The self-model is captured as a functional model that specifies how the methods of the agent use knowledge to achieve its tasks. The process of generating self-explanations uses Chain of Thought to reflect on the self-model and ChatGPT to provide explanations about its functioning. We evaluate the self-explanation of the AI social assistant for completeness and correctness. We also report on its deployment in a live class. 

**Abstract (ZH)**: 社会人工智能代理与社区成员交互，从而改变社区的行为。例如，在在线学习中，一个人工智能社交助手可以连接学习者，从而增强社会互动。这些社会人工智能助手也需要进行自我解释，以提高透明度和与学习者的信任度。我们提出了一种基于自我模型进行自我解释的方法，该自我模型借鉴了人工智能社交助手的内在认知过程。自我模型被表示为一个功能性模型，规定了代理使用知识来完成任务的方法。生成自我解释的过程使用“链式思考”来反思自我模型，并使用ChatGPT来解释其运作机制。我们评估了人工智能社交助手的自我解释的完整性和准确性，并在其实际课程中进行了部署。 

---
# Fanar: An Arabic-Centric Multimodal Generative AI Platform 

**Title (ZH)**: Fanar：一个以阿拉伯语为中心的多模态生成人工智能平台 

**Authors**: Fanar Team, Ummar Abbas, Mohammad Shahmeer Ahmad, Firoj Alam, Enes Altinisik, Ehsannedin Asgari, Yazan Boshmaf, Sabri Boughorbel, Sanjay Chawla, Shammur Chowdhury, Fahim Dalvi, Kareem Darwish, Nadir Durrani, Mohamed Elfeky, Ahmed Elmagarmid, Mohamed Eltabakh, Masoomali Fatehkia, Anastasios Fragkopoulos, Maram Hasanain, Majd Hawasly, Mus'ab Husaini, Soon-Gyo Jung, Ji Kim Lucas, Walid Magdy, Safa Messaoud, Abubakr Mohamed, Tasnim Mohiuddin, Basel Mousi, Hamdy Mubarak, Ahmad Musleh, Zan Naeem, Mourad Ouzzani, Dorde Popovic, Amin Sadeghi, Husrev Taha Sencar, Mohammed Shinoy, Omar Sinan, Yifan Zhang, Ahmed Ali, Yassine El Kheir, Xiaosong Ma, Chaoyi Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13944)  

**Abstract**: We present Fanar, a platform for Arabic-centric multimodal generative AI systems, that supports language, speech and image generation tasks. At the heart of Fanar are Fanar Star and Fanar Prime, two highly capable Arabic Large Language Models (LLMs) that are best in the class on well established benchmarks for similar sized models. Fanar Star is a 7B (billion) parameter model that was trained from scratch on nearly 1 trillion clean and deduplicated Arabic, English and Code tokens. Fanar Prime is a 9B parameter model continually trained on the Gemma-2 9B base model on the same 1 trillion token set. Both models are concurrently deployed and designed to address different types of prompts transparently routed through a custom-built orchestrator. The Fanar platform provides many other capabilities including a customized Islamic Retrieval Augmented Generation (RAG) system for handling religious prompts, a Recency RAG for summarizing information about current or recent events that have occurred after the pre-training data cut-off date. The platform provides additional cognitive capabilities including in-house bilingual speech recognition that supports multiple Arabic dialects, voice and image generation that is fine-tuned to better reflect regional characteristics. Finally, Fanar provides an attribution service that can be used to verify the authenticity of fact based generated content.
The design, development, and implementation of Fanar was entirely undertaken at Hamad Bin Khalifa University's Qatar Computing Research Institute (QCRI) and was sponsored by Qatar's Ministry of Communications and Information Technology to enable sovereign AI technology development. 

**Abstract (ZH)**: 我们介绍了Fanar平台，这是一个针对阿拉伯语的多模态生成AI系统平台，支持语言、语音和图像生成任务。Fanar的核心是Fanar Star和Fanar Prime，这是两个性能优秀的阿拉伯语大型语言模型（LLMs），在同类基准测试中表现出色。Fanar Star是一个70亿参数的模型，从几乎1万亿个清洗和去重后的阿拉伯语、英语和代码标记中从头开始训练。Fanar Prime是一个90亿参数的模型，持续训练在Gemma-2 9B基础模型上，使用同样1万亿标记集。这两个模型同时部署，并设计成能够通过自定义构建的协调器透明地处理不同类型的提示。Fanar平台还提供了其他许多功能，包括自定义的伊斯兰检索增强生成（RAG）系统以处理宗教提示，以及近期RAG以总结发生在预训练数据截止日期之后的当前或最近事件的信息。该平台还提供其他认知功能，包括支持多种阿拉伯方言的内部双语语音识别，以及更好地反映区域特征的语音和图像生成。最后，Fanar提供了验证基于事实生成内容真实性的一种归属服务。

Fanar平台的设计、开发和实现完全由卡塔尔计算研究学会（Qatar Computing Research Institute，QCRI）的哈马德·本·哈利法大学（Hamad Bin Khalifa University）自主完成，并由卡塔尔通讯与信息技术部赞助，以促进主权AI技术的发展。 

---
# Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis 

**Title (ZH)**: 语言表示偏好零样本跨域认知诊断 

**Authors**: Shuo Liu, Zihan Zhou, Yuanhao Liu, Jing Zhang, Hong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2501.13943)  

**Abstract**: Cognitive diagnosis aims to infer students' mastery levels based on their historical response logs. However, existing cognitive diagnosis models (CDMs), which rely on ID embeddings, often have to train specific models on specific domains. This limitation may hinder their directly practical application in various target domains, such as different subjects (e.g., Math, English and Physics) or different education platforms (e.g., ASSISTments, Junyi Academy and Khan Academy). To address this issue, this paper proposes the language representation favored zero-shot cross-domain cognitive diagnosis (LRCD). Specifically, LRCD first analyzes the behavior patterns of students, exercises and concepts in different domains, and then describes the profiles of students, exercises and concepts using textual descriptions. Via recent advanced text-embedding modules, these profiles can be transformed to vectors in the unified language space. Moreover, to address the discrepancy between the language space and the cognitive diagnosis space, we propose language-cognitive mappers in LRCD to learn the mapping from the former to the latter. Then, these profiles can be easily and efficiently integrated and trained with existing CDMs. Extensive experiments show that training LRCD on real-world datasets can achieve commendable zero-shot performance across different target domains, and in some cases, it can even achieve competitive performance with some classic CDMs trained on the full response data on target domains. Notably, we surprisingly find that LRCD can also provide interesting insights into the differences between various subjects (such as humanities and sciences) and sources (such as primary and secondary education). 

**Abstract (ZH)**: 认知诊断旨在根据学生的历史反应日志推断其掌握水平。然而，现有的认知诊断模型（CDMs），依赖于ID嵌入，通常需要在特定领域上训练特定模型。这一限制可能阻碍了它们在各种目标领域中的直接应用，例如不同的学科（如数学、英语和物理）或不同的教育平台（如ASSISTments、 Chunyi Academy和Khan Academy）。为了解决这一问题，本文提出了语言表示偏好的零样本跨域认知诊断（LRCD）。具体而言，LRCD 首先分析不同领域中学生、练习和概念的行为模式，然后使用文本描述来描述学生、练习和概念。通过最近的高级文本嵌入模块，这些描述可以被转换为统一语言空间中的向量。此外，为了解决语言空间与认知诊断空间之间的差异，LRCD 提出了语言认知映射器，学习从前者到后者的映射。这样，这些描述可以方便且有效地与现有的CDMs进行集成和训练。通过广泛的实验，我们发现，在实际数据集上训练LRCD 可以在不同目标领域中实现令人满意的效果，并且在某些情况下，甚至可以达到与经典CDMs利用目标领域完整反应数据训练相当的性能。值得注意的是，我们意外地发现LRCD 还可以提供关于不同学科（如人文学科和自然科学）和不同来源（如基础教育和高等教育）之间差异的有趣见解。 

---
# GaussMark: A Practical Approach for Structural Watermarking of Language Models 

**Title (ZH)**: GaussMark：一种实用的语言模型结构水印方法 

**Authors**: Adam Block, Ayush Sekhari, Alexander Rakhlin  

**Link**: [PDF](https://arxiv.org/pdf/2501.13941)  

**Abstract**: Recent advances in Large Language Models (LLMs) have led to significant improvements in natural language processing tasks, but their ability to generate human-quality text raises significant ethical and operational concerns in settings where it is important to recognize whether or not a given text was generated by a human. Thus, recent work has focused on developing techniques for watermarking LLM-generated text, i.e., introducing an almost imperceptible signal that allows a provider equipped with a secret key to determine if given text was generated by their model. Current watermarking techniques are often not practical due to concerns with generation latency, detection time, degradation in text quality, or robustness. Many of these drawbacks come from the focus on token-level watermarking, which ignores the inherent structure of text. In this work, we introduce a new scheme, GaussMark, that is simple and efficient to implement, has formal statistical guarantees on its efficacy, comes at no cost in generation latency, and embeds the watermark into the weights of the model itself, providing a structural watermark. Our approach is based on Gaussian independence testing and is motivated by recent empirical observations that minor additive corruptions to LLM weights can result in models of identical (or even improved) quality. We show that by adding a small amount of Gaussian noise to the weights of a given LLM, we can watermark the model in a way that is statistically detectable by a provider who retains the secret key. We provide formal statistical bounds on the validity and power of our procedure. Through an extensive suite of experiments, we demonstrate that GaussMark is reliable, efficient, and relatively robust to corruptions such as insertions, deletions, substitutions, and roundtrip translations and can be instantiated with essentially no loss in model quality. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的发展显著提高了自然语言处理任务的性能，但它们生成与人类质量相当文本的能力引发了一系列伦理和操作方面的担忧，尤其是在需要识别给定文本是否由人类生成的场合。因此，最近的研究主要集中在开发为LLM生成的文本标记的技术，即引入几乎不可察觉的信号，以供拥有秘密密钥的提供者能够判断给定文本是否由其模型生成。当前的标记技术往往由于生成延迟、检测时间、文本质量下降或鲁棒性问题而不切实际。许多这些缺点源于对标记级别关注的焦点，这忽略了文本固有的结构。在本研究中，我们引入了一种新的方案——GaussMark，该方案简单且易于实现，具有形式统计保证的有效性，在生成延迟方面没有任何成本，并将水印嵌入到模型本身的权重中，提供了一种结构化的水印。我们的方法基于高斯独立性测试，并受到最近的实证观察的启发，即对LLM权重进行轻微的附加破坏可能会导致模型具有相同（甚至更高质量）。我们通过向给定LLM的权重中添加少量高斯噪声，以可由保留秘密密钥的提供者统计数据检测的方式对模型进行标记。我们提供了该过程有效性和效力的形式统计界限。通过一系列广泛的实验，我们证明了GaussMark是可靠的、高效的，并且相对具有抗破坏性（插入、删除、替换和往返翻译）的鲁棒性，同时可以几乎不损失模型质量即可实现该方案。 

---
