# AIDE: AI-Driven Exploration in the Space of Code 

**Title (ZH)**: AIDE：AI 驱动的代码空间探索 

**Authors**: Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, Yuxiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13138)  

**Abstract**: Machine learning, the foundation of modern artificial intelligence, has driven innovations that have fundamentally transformed the world. Yet, behind advancements lies a complex and often tedious process requiring labor and compute intensive iteration and experimentation. Engineers and scientists developing machine learning models spend much of their time on trial-and-error tasks instead of conceptualizing innovative solutions or research hypotheses. To address this challenge, we introduce AI-Driven Exploration (AIDE), a machine learning engineering agent powered by large language models (LLMs). AIDE frames machine learning engineering as a code optimization problem, and formulates trial-and-error as a tree search in the space of potential solutions. By strategically reusing and refining promising solutions, AIDE effectively trades computational resources for enhanced performance, achieving state-of-the-art results on multiple machine learning engineering benchmarks, including our Kaggle evaluations, OpenAI MLE-Bench and METRs RE-Bench. 

**Abstract (ZH)**: 机器学习，现代人工智能的基础，已经推动了一系列创新，这些创新从根本上改变了世界。然而，这些进步背后隐藏着一个复杂且往往耗时的过程，需要大量的劳动和计算密集型迭代与实验。开发机器学习模型的工程师和科学家们大量时间花费在试错任务上，而不是构思创新的解决方案或研究假说。为解决这一挑战，我们引入了AI驱动探索（AIDE），这是一种由大规模语言模型（LLMs）驱动的机器学习工程代理。AIDE将机器学习工程视为代码优化问题，将试错过程视为在潜在解决方案空间中的树搜索问题。通过战略性地重用和改进有希望的解决方案，AIDE有效地用计算资源换取更好的性能，其结果在多个机器学习工程基准测试上达到了最先进的水平，包括我们在Kaggle的评估、OpenAI MLE-Bench和METRS RE-Bench的结果中得到了验证。 

---
# Theorem Prover as a Judge for Synthetic Data Generation 

**Title (ZH)**: 定theorem证明器作为合成数据生成的裁判 

**Authors**: Joshua Ong Jun Leang, Giwon Hong, Wenda Li, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13137)  

**Abstract**: The demand for synthetic data in mathematical reasoning has increased due to its potential to enhance the mathematical capabilities of large language models (LLMs). However, ensuring the validity of intermediate reasoning steps remains a significant challenge, affecting data quality. While formal verification via theorem provers effectively validates LLM reasoning, the autoformalisation of mathematical proofs remains error-prone. In response, we introduce iterative autoformalisation, an approach that iteratively refines theorem prover formalisation to mitigate errors, thereby increasing the execution rate on the Lean prover from 60% to 87%. Building upon that, we introduce Theorem Prover as a Judge (TP-as-a-Judge), a method that employs theorem prover formalisation to rigorously assess LLM intermediate reasoning, effectively integrating autoformalisation with synthetic data generation. Finally, we present Reinforcement Learning from Theorem Prover Feedback (RLTPF), a framework that replaces human annotation with theorem prover feedback in Reinforcement Learning from Human Feedback (RLHF). Across multiple LLMs, applying TP-as-a-Judge and RLTPF improves benchmarks with only 3,508 samples, achieving 5.56% accuracy gain on Mistral-7B for MultiArith, 6.00% on Llama-2-7B for SVAMP, and 3.55% on Llama-3.1-8B for AQUA. 

**Abstract (ZH)**: 由于合成数据在数学推理中的潜在优势，对合成数据的需求有所增加，这有助于提升大型语言模型（LLMs）的数学能力。然而，确保中间推理步骤的有效性仍然是一个重大挑战，影响数据质量。虽然形式验证通过定理证明器可以有效验证LLM的推理，但数学证明的形式化仍存在错误。为应对这一问题，我们引入了一种迭代形式化方法，该方法通过逐步细化定理证明的形式化来减少错误，从而将Lean证明器的执行率从60%提高到87%。在此基础上，我们提出了“证明器作为裁判”（TP-as-a-Judge）方法，这种方法利用定理证明器的形式化来严格评估LLM的中间推理，有效地将形式化与合成数据生成相结合。最后，我们提出了基于定理证明器反馈的强化学习框架（RLTPF），该框架用定理证明器的反馈取代了人类标注，应用在基于人类反馈的强化学习（RLHF）中。在多种LLM中，应用TP-as-a-Judge和RLTPF在仅使用3,508个样本的情况下提升了基准性能，在Mistral-7B的MultiArith任务中实现了5.56%的准确性提升，在Llama-2-7B的SVAMP任务中实现了6.00%的提升，在Llama-3.1-8B的AQUA任务中实现了3.55%的提升。 

---
# Rethinking Diverse Human Preference Learning through Principal Component Analysis 

**Title (ZH)**: 通过主成分分析重新思考多样化的人类偏好学习 

**Authors**: Feng Luo, Rui Yang, Hao Sun, Chunyuan Deng, Jiarui Yao, Jingyan Shen, Huan Zhang, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13131)  

**Abstract**: Understanding human preferences is crucial for improving foundation models and building personalized AI systems. However, preferences are inherently diverse and complex, making it difficult for traditional reward models to capture their full range. While fine-grained preference data can help, collecting it is expensive and hard to scale. In this paper, we introduce Decomposed Reward Models (DRMs), a novel approach that extracts diverse human preferences from binary comparisons without requiring fine-grained annotations. Our key insight is to represent human preferences as vectors and analyze them using Principal Component Analysis (PCA). By constructing a dataset of embedding differences between preferred and rejected responses, DRMs identify orthogonal basis vectors that capture distinct aspects of preference. These decomposed rewards can be flexibly combined to align with different user needs, offering an interpretable and scalable alternative to traditional reward models. We demonstrate that DRMs effectively extract meaningful preference dimensions (e.g., helpfulness, safety, humor) and adapt to new users without additional training. Our results highlight DRMs as a powerful framework for personalized and interpretable LLM alignment. 

**Abstract (ZH)**: 了解人类偏好对于改进基础模型和构建个性化AI系统至关重要。然而，偏好本质上是多样化和复杂的，使得传统的奖励模型难以捕捉其全部范围。虽然精细的偏好数据有助于这一过程，但收集这些数据既昂贵又难以扩大规模。在本文中，我们提出了分解奖励模型（Decomposed Reward Models, DRMs），这是一种新颖的方法，可以从二元比较中提取多样化的用户偏好，而无需依赖精细的标注。我们的关键见解在于将人类偏好表示为向量，并使用主成分分析（PCA）进行分析。通过构建偏好响应和拒绝响应嵌入差异的数据集，DRMs识别出能够捕捉偏好不同方面的正交基向量。这些分解后的奖励可以灵活组合以满足不同的用户需求，从而为传统的奖励模型提供了一个可解释且可扩展的替代方案。我们证明，DRMs能够有效地提取有意义的偏好维度（如有用性、安全性、幽默感），并且能够在无需额外训练的情况下适应新用户。本文的结果突显了DRMs作为个性化的、可解释的LLM对齐的强大框架的重要性。 

---
# MatterChat: A Multi-Modal LLM for Material Science 

**Title (ZH)**: MatterChat：材料科学领域的多模态大规模语言模型 

**Authors**: Yingheng Tang, Wenbin Xu, Jie Cao, Jianzhu Ma, Weilu Gao, Steve Farrell, Benjamin Erichson, Michael W. Mahoney, Andy Nonaka, Zhi Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13107)  

**Abstract**: Understanding and predicting the properties of inorganic materials is crucial for accelerating advancements in materials science and driving applications in energy, electronics, and beyond. Integrating material structure data with language-based information through multi-modal large language models (LLMs) offers great potential to support these efforts by enhancing human-AI interaction. However, a key challenge lies in integrating atomic structures at full resolution into LLMs. In this work, we introduce MatterChat, a versatile structure-aware multi-modal LLM that unifies material structural data and textual inputs into a single cohesive model. MatterChat employs a bridging module to effectively align a pretrained machine learning interatomic potential with a pretrained LLM, reducing training costs and enhancing flexibility. Our results demonstrate that MatterChat significantly improves performance in material property prediction and human-AI interaction, surpassing general-purpose LLMs such as GPT-4. We also demonstrate its usefulness in applications such as more advanced scientific reasoning and step-by-step material synthesis. 

**Abstract (ZH)**: 理解并预测无机材料的性质对于加速材料科学的进步和推动能源、电子等领域应用至关重要。通过多模态大型语言模型（LLMs）整合材料结构数据与基于语言的信息，有助于增强人类与AI的交互，具有巨大的潜力。然而，将原子结构信息全面整合到LLMs中仍然是一个关键挑战。在此项研究中，我们介绍了MatterChat，这是一种多功能结构感知多模态LLM，能够将材料结构数据和文本输入统一到一个连贯的模型中。MatterChat通过引入一个连接模块来有效对接预训练的机器学习原子间势能模型与预训练的LLM，降低训练成本并增强灵活性。我们的研究结果表明，MatterChat显著提高了材料性质预测和人机交互的性能，超越了通用的LLM，如GPT-4。我们还展示了它在更高级的科学推理和逐步材料合成等应用中的实用价值。 

---
# Interactive Agents to Overcome Ambiguity in Software Engineering 

**Title (ZH)**: 软件工程中克服模糊性的交互式代理 

**Authors**: Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2502.13069)  

**Abstract**: AI agents are increasingly being deployed to automate tasks, often based on ambiguous and underspecified user instructions. Making unwarranted assumptions and failing to ask clarifying questions can lead to suboptimal outcomes, safety risks due to tool misuse, and wasted computational resources. In this work, we study the ability of LLM agents to handle ambiguous instructions in interactive code generation settings by evaluating proprietary and open-weight models on their performance across three key steps: (a) leveraging interactivity to improve performance in ambiguous scenarios, (b) detecting ambiguity, and (c) asking targeted questions. Our findings reveal that models struggle to distinguish between well-specified and underspecified instructions. However, when models interact for underspecified inputs, they effectively obtain vital information from the user, leading to significant improvements in performance and underscoring the value of effective interaction. Our study highlights critical gaps in how current state-of-the-art models handle ambiguity in complex software engineering tasks and structures the evaluation into distinct steps to enable targeted improvements. 

**Abstract (ZH)**: 人工智能代理正越来越多地被部署以自动化任务，通常是基于含糊不清且未充分指定的用户指令。做出不适当的假设和未提出澄清问题可能导致次优结果、因工具误用而带来的安全风险以及计算资源的浪费。在这项研究中，我们通过评估专有和开源模型在三个关键步骤中的表现，研究了语言模型代理在这种交互式代码生成设置中处理含糊指令的能力：(a) 利用互动来改善含糊场景中的表现，(b) 检测含糊性，以及(c) 提出有针对性的问题。我们的研究结果表明，模型难以区分明确指定与未充分指定的指令。然而，对于未充分指定的输入，当模型进行互动时，能够有效地从用户那里获得关键信息，从而显著提高表现，进一步突出了高效互动的价值。我们的研究指出了当前最先进的模型在复杂软件工程任务中处理含糊性时存在的重要空白，并将评估结构化为不同的步骤，以促进有针对性的改进。 

---
# AI-Assisted Decision Making with Human Learning 

**Title (ZH)**: AI辅助决策与人类学习结合 

**Authors**: Gali Noti, Kate Donahue, Jon Kleinberg, Sigal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2502.13062)  

**Abstract**: AI systems increasingly support human decision-making. In many cases, despite the algorithm's superior performance, the final decision remains in human hands. For example, an AI may assist doctors in determining which diagnostic tests to run, but the doctor ultimately makes the diagnosis. This paper studies such AI-assisted decision-making settings, where the human learns through repeated interactions with the algorithm. In our framework, the algorithm -- designed to maximize decision accuracy according to its own model -- determines which features the human can consider. The human then makes a prediction based on their own less accurate model. We observe that the discrepancy between the algorithm's model and the human's model creates a fundamental tradeoff. Should the algorithm prioritize recommending more informative features, encouraging the human to recognize their importance, even if it results in less accurate predictions in the short term until learning occurs? Or is it preferable to forgo educating the human and instead select features that align more closely with their existing understanding, minimizing the immediate cost of learning? This tradeoff is shaped by the algorithm's time-discounted objective and the human's learning ability. Our results show that optimal feature selection has a surprisingly clean combinatorial characterization, reducible to a stationary sequence of feature subsets that is tractable to compute. As the algorithm becomes more "patient" or the human's learning improves, the algorithm increasingly selects more informative features, enhancing both prediction accuracy and the human's understanding. Notably, early investment in learning leads to the selection of more informative features than a later investment. We complement our analysis by showing that the impact of errors in the algorithm's knowledge is limited as it does not make the prediction directly. 

**Abstract (ZH)**: 随着AI系统的不断进步，它们在支持人类决策方面扮演着越来越重要的角色。尽管算法在许多情况下表现出色，最终的决策权仍然掌握在人类手中。例如，AI可能帮助医生决定需要进行哪些诊断测试，但最终的诊断还是由医生决定。本文研究了一种AI辅助决策的场景，即人类通过与算法多次互动来学习。在我们的框架中，算法（旨在根据自己的模型最大化决策准确性）决定人类可以考虑哪些特征。然后，人类基于他们自己不太准确的模型做出预测。我们观察到，算法模型与人类模型之间的差异产生了一种基本的权衡：算法是否应优先推荐更具信息量的特征，促使人类认识到这些特征的重要性，即使这会导致短期内预测不那么准确，直到学习过程发生？还是更愿意牺牲教育人类而选择与人类现有理解更为一致的特征，从而尽量减少立即的学习成本？这种权衡由算法的时间折扣目标和人类的学习能力所塑造。我们的研究结果表明，最优特征选择具有惊人的组合特征刻画，可简化为一个可计算的稳定特征子集序列。随着算法变得更加“耐心”或人类学习能力的提升，算法会越来越多地选择更具信息量的特征，从而同时提高预测准确性和人类的理解能力。值得注意的是，早期对学习的投资会比后期投资选择更多更具信息量的特征。我们通过进一步的分析补充了这一结论，表明算法知识中的错误对预测没有直接影响。 

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
# Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger 

**Title (ZH)**: 具有元认知触发的大型语言模型自适应工具使用研究 

**Authors**: Wenjun Li, Dexun Li, Kuicai Dong, Cong Zhang, Hao Zhang, Weiwen Liu, Yasheng Wang, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12961)  

**Abstract**: Large language models (LLMs) have shown remarkable emergent capabilities, transforming the execution of functional tasks by leveraging external tools for complex problems that require specialized processing or real-time data. While existing research expands LLMs access to diverse tools (e.g., program interpreters, search engines, weather/map apps), the necessity of using these tools is often overlooked, leading to indiscriminate tool invocation. This naive approach raises two key issues:(1) increased delays due to unnecessary tool calls, and (2) potential errors resulting from faulty interactions with external tools. In this paper, we introduce meta-cognition as a proxy for LLMs self-assessment of their capabilities, representing the model's awareness of its own limitations. Based on this, we propose MeCo, an adaptive decision-making strategy for external tool use. MeCo quantifies metacognitive scores by capturing high-level cognitive signals in the representation space, guiding when to invoke tools. Notably, MeCo is fine-tuning-free and incurs minimal cost. Our experiments show that MeCo accurately detects LLMs' internal cognitive signals and significantly improves tool-use decision-making across multiple base models and benchmarks. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现出了显著的涌现能力，通过利用外部工具解决复杂问题，这些复杂问题需要特定处理或实时数据。尽管现有研究扩大了LLMs对各种工具（如程序解释器、搜索引擎、天气/地图应用程序）的访问，但在实际使用中，这些工具的必要性有时被忽视，导致了工具调用的随意性。这种简单粗放的方法带来了两个关键问题：（1）由于不必要的工具调用增加延迟，（2）由于与外部工具交互不当而可能导致潜在错误。在本文中，我们引入了元认知作为LLMs自我评估其能力的代理，代表了模型对其自身局限性的意识。基于此，我们提出了MeCo，即一种适应性外部工具使用决策策略。MeCo通过捕捉表示空间中的高层次认知信号来量化元认知得分，并指导何时调用工具。值得注意的是，MeCo不需要微调，成本也非常低。我们的实验表明，MeCo准确地检测到了LLMs的内部认知信号，并在多个基础模型和基准测试中显著改善了工具使用的决策过程。 

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
# Towards Adaptive Feedback with AI: Comparing the Feedback Quality of LLMs and Teachers on Experimentation Protocols 

**Title (ZH)**: 面向AI适应性反馈的探索：比较大型语言模型与教师在实验协议中的反馈质量 

**Authors**: Kathrin Seßler, Arne Bewersdorff, Claudia Nerdel, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2502.12842)  

**Abstract**: Effective feedback is essential for fostering students' success in scientific inquiry. With advancements in artificial intelligence, large language models (LLMs) offer new possibilities for delivering instant and adaptive feedback. However, this feedback often lacks the pedagogical validation provided by real-world practitioners. To address this limitation, our study evaluates and compares the feedback quality of LLM agents with that of human teachers and science education experts on student-written experimentation protocols. Four blinded raters, all professionals in scientific inquiry and science education, evaluated the feedback texts generated by 1) the LLM agent, 2) the teachers and 3) the science education experts using a five-point Likert scale based on six criteria of effective feedback: Feed Up, Feed Back, Feed Forward, Constructive Tone, Linguistic Clarity, and Technical Terminology. Our results indicate that LLM-generated feedback shows no significant difference to that of teachers and experts in overall quality. However, the LLM agent's performance lags in the Feed Back dimension, which involves identifying and explaining errors within the student's work context. Qualitative analysis highlighted the LLM agent's limitations in contextual understanding and in the clear communication of specific errors. Our findings suggest that combining LLM-generated feedback with human expertise can enhance educational practices by leveraging the efficiency of LLMs and the nuanced understanding of educators. 

**Abstract (ZH)**: 有效的反馈对于培养学生的科学探究成功至关重要。随着人工智能的进步，大规模语言模型（LLMs）为即时和适应性反馈提供了新的可能性。然而，这种反馈往往缺乏实际教育从业者提供的教学生态验证。为了解决这一局限，本研究评估并比较了LLM代理与真人教师和科学教育专家对学生撰写的实验方案给出的反馈质量。四名盲评者，均为科学探究和科学教育领域的专业人士，根据有效的反馈六个标准（Feed Up、Feed Back、Feed Forward、建设性语气、语言清晰度、技术术语）使用五点李克特量表分别对LLM代理、教师和科学教育专家生成的反馈文本进行了评估。研究结果表明，LLM生成的反馈在整体质量上与教师和专家的反馈无显著差异。然而，LLM代理在Feed Back维度的表现较弱，该维度涉及在学生作品的具体情境中识别并解释错误。定性分析揭示了LLM代理在上下文理解和具体错误清晰传达方面的局限性。我们的研究结果表明，结合LLM生成的反馈与人类专业知识可以提升教育实践，利用LLMs的高效性和教育者的精细理解。 

---
# VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation 

**Title (ZH)**: VidCapBench：可控文本到视频生成的全面视频字幕基准测试 

**Authors**: Xinlong Chen, Yuanxing Zhang, Chongling Rao, Yushuo Guan, Jiaheng Liu, Fuzheng Zhang, Chengru Song, Qiang Liu, Di Zhang, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12782)  

**Abstract**: The training of controllable text-to-video (T2V) models relies heavily on the alignment between videos and captions, yet little existing research connects video caption evaluation with T2V generation assessment. This paper introduces VidCapBench, a video caption evaluation scheme specifically designed for T2V generation, agnostic to any particular caption format. VidCapBench employs a data annotation pipeline, combining expert model labeling and human refinement, to associate each collected video with key information spanning video aesthetics, content, motion, and physical laws. VidCapBench then partitions these key information attributes into automatically assessable and manually assessable subsets, catering to both the rapid evaluation needs of agile development and the accuracy requirements of thorough validation. By evaluating numerous state-of-the-art captioning models, we demonstrate the superior stability and comprehensiveness of VidCapBench compared to existing video captioning evaluation approaches. Verification with off-the-shelf T2V models reveals a significant positive correlation between scores on VidCapBench and the T2V quality evaluation metrics, indicating that VidCapBench can provide valuable guidance for training T2V models. The project is available at this https URL. 

**Abstract (ZH)**: 控制文本到视频（T2V）模型的训练高度依赖于视频和字幕之间的对齐，而现有研究中很少将视频字幕评估与T2V生成评估联系起来。本文介绍了VidCapBench，这是一种专门设计用于T2V生成的视频字幕评估方案，与任何特定的字幕格式无关。VidCapBench 采用数据标注管道，结合专家模型标注和人工细化，将每个收集的视频与视频美学、内容、运动和物理定律等方面的关键信息关联起来。随后，VidCapBench 将这些关键信息属性划分为自动评估和手动评估的子集，既能满足敏捷开发中的快速评估需求，也能满足全面验证的准确性要求。通过评估多个最先进的字幕模型，我们展示了VidCapBench 相较于现有视频字幕评估方法的优越稳定性和全面性。与现成的T2V模型进行验证表明，VidCapBench 的评分与T2V质量评估指标之间存在显著的正相关关系，表明VidCapBench 可以为训练T2V模型提供有价值的方向性指导。项目详情请访问 <这个网址>。 

---
# Perovskite-LLM: Knowledge-Enhanced Large Language Models for Perovskite Solar Cell Research 

**Title (ZH)**: Perovskite-LLM：用于钙钛矿太阳电池研究的知识增强型大规模语言模型 

**Authors**: Xiang Liu, Penglei Sun, Shuyan Chen, Longhan Zhang, Peijie Dong, Huajie You, Yongqi Zhang, Chang Yan, Xiaowen Chu, Tong-yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12669)  

**Abstract**: The rapid advancement of perovskite solar cells (PSCs) has led to an exponential growth in research publications, creating an urgent need for efficient knowledge management and reasoning systems in this domain. We present a comprehensive knowledge-enhanced system for PSCs that integrates three key components. First, we develop Perovskite-KG, a domain-specific knowledge graph constructed from 1,517 research papers, containing 23,789 entities and 22,272 relationships. Second, we create two complementary datasets: Perovskite-Chat, comprising 55,101 high-quality question-answer pairs generated through a novel multi-agent framework, and Perovskite-Reasoning, containing 2,217 carefully curated materials science problems. Third, we introduce two specialized large language models: Perovskite-Chat-LLM for domain-specific knowledge assistance and Perovskite-Reasoning-LLM for scientific reasoning tasks. Experimental results demonstrate that our system significantly outperforms existing models in both domain-specific knowledge retrieval and scientific reasoning tasks, providing researchers with effective tools for literature review, experimental design, and complex problem-solving in PSC research. 

**Abstract (ZH)**: Perovskite 太阳能电池（PSCs）的快速进步导致了研究论文的指数级增长，迫切需要高效的知识管理和推理系统。我们提出了一种全面的知识增强系统，该系统集成了三个关键组件。首先，我们开发了 Perovskite-KG，这是一种从 1,517 篇研究论文中构建的特定领域的知识图谱，包含 23,789 个实体和 22,272 个关系。其次，我们创建了两个互补的数据集：Perovskite-Chat，包含 55,101 个高质量的问题-答案对，通过一种新型多代理框架生成，以及 Perovskite-Reasoning，包含 2,217 个精心策划的材料科学问题。第三，我们介绍了两个专门的大规模语言模型：Perovskite-Chat-LLM 用于领域特定知识辅助，Perovskite-Reasoning-LLM 用于科学研究任务。实验结果表明，我们的系统在领域特定知识检索和科学研究任务方面显著优于现有模型，为研究人员提供了有效的文献回顾、实验设计和 PSC 研究复杂问题解决的工具。 

---
# RM-PoT: Reformulating Mathematical Problems and Solving via Program of Thoughts 

**Title (ZH)**: RM-PoT: 重新表述数学问题并借助思维程序求解 

**Authors**: Yu Zhang, Shujun Peng, Nengwu Wu, Xinhan Lin, Yang Hu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12589)  

**Abstract**: Recently, substantial advancements have been made in training language models to carry out step-by-step reasoning for solving intricate numerical reasoning tasks. Beyond the methods used to solve these problems, the structure and formulation of the problems themselves also play a crucial role in determining the performance of large language models. We observe that even small changes in the surface form of mathematical problems can have a profound impact on both the answer distribution and solve rate. This highlights the vulnerability of LLMs to surface-level variations, revealing its limited robustness when reasoning through complex problems. In this paper, we propose RM-PoT, a three-stage framework that integrates problem reformulation (RM), code-aided reasoning (PoT), and domain-aware few-shot learning to address these limitations. Our approach first reformulates the input problem into diverse surface forms to reduce structural bias, then retrieves five semantically aligned examples from a pre-constructed domain-specific question bank to provide contextual guidance, and finally generates executable Python code for precise computation. 

**Abstract (ZH)**: 近年来，在训练语言模型以执行解决复杂数值推理任务的逐步推理方面取得了显著进展。除了用于解决这些问题的方法外，问题本身的结构和表述形式也对大型语言模型的性能起着关键作用。我们发现，即使数学问题的表面形式发生很小的变化，也会对答案分布和解题率产生深远影响。这表明了语言模型对表面形式变化的脆弱性，揭示了它们在复杂问题推理过程中有限的鲁棒性。在本文中，我们提出了一种三阶段框架——RM-PoT，该框架整合了问题重述（RM）、代码辅助推理（PoT）和领域自意识的少量样本学习，以应对这些局限性。我们的方法首先将输入问题重新表述为多种表面形式以减少结构偏差，然后从预先构建的特定领域问题库中检索五个语义对齐的实例以提供上下文指导，最后生成可执行的Python代码以进行精确计算。 

---
# Exploring the Impact of Personality Traits on LLM Bias and Toxicity 

**Title (ZH)**: 探索人格特质对大语言模型偏见和毒性的影响 

**Authors**: Shuo Wang, Renhao Li, Xi Chen, Yulin Yuan, Derek F. Wong, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12566)  

**Abstract**: With the different roles that AI is expected to play in human life, imbuing large language models (LLMs) with different personalities has attracted increasing research interests. While the "personification" enhances human experiences of interactivity and adaptability of LLMs, it gives rise to critical concerns about content safety, particularly regarding bias, sentiment and toxicity of LLM generation. This study explores how assigning different personality traits to LLMs affects the toxicity and biases of their outputs. Leveraging the widely accepted HEXACO personality framework developed in social psychology, we design experimentally sound prompts to test three LLMs' performance on three toxic and bias benchmarks. The findings demonstrate the sensitivity of all three models to HEXACO personality traits and, more importantly, a consistent variation in the biases, negative sentiment and toxicity of their output. In particular, adjusting the levels of several personality traits can effectively reduce bias and toxicity in model performance, similar to humans' correlations between personality traits and toxic behaviors. The findings highlight the additional need to examine content safety besides the efficiency of training or fine-tuning methods for LLM personification. They also suggest a potential for the adjustment of personalities to be a simple and low-cost method to conduct controlled text generation. 

**Abstract (ZH)**: 随着人工智能在人类生活中扮演不同角色的预期逐渐增加，赋予大型语言模型（LLMs）不同的人格特质引起了越来越多的研究兴趣。尽管这种“拟人化”增强了用户与LLMs互动和适应的体验，但它也引发了关于内容安全性的关键关注，特别是关于LLMs生成内容的偏见、情感和毒性问题。本研究探讨了赋予LLMs不同人格特质如何影响其输出中的偏见和毒性。依托社会心理学中广泛接受的六因素人格框架（HEXACO），我们设计了实验性的提示，对三种LLMs在三个毒性与偏见基准测试上的性能进行了测试。研究结果表明，所有的模型都对HEXACO人格特质高度敏感，更重要的是，其输出中的偏见、负面情感和毒性表现出一致的变化趋势。特别是，调整某些人格特质的水平可以有效地降低模型性能中的偏见和毒性，类似于人格特质与有害行为之间的人类相关性。研究结果强调，在研究LLMs拟人化时，除了关注训练或微调方法的效率之外，还需要进一步审视内容安全性。此外，研究结果还表明，调整人格特质可能成为一种简单且低成本的方法，用于控制文本生成。 

---
# CityEQA: A Hierarchical LLM Agent on Embodied Question Answering Benchmark in City Space 

**Title (ZH)**: CityEQA：城市空间中层级化语言模型代理的实体问答基准Benchmark 

**Authors**: Yong Zhao, Kai Xu, Zhengqiu Zhu, Yue Hu, Zhiheng Zheng, Yingfeng Chen, Yatai Ji, Chen Gao, Yong Li, Jincai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12532)  

**Abstract**: Embodied Question Answering (EQA) has primarily focused on indoor environments, leaving the complexities of urban settings - spanning environment, action, and perception - largely unexplored. To bridge this gap, we introduce CityEQA, a new task where an embodied agent answers open-vocabulary questions through active exploration in dynamic city spaces. To support this task, we present CityEQA-EC, the first benchmark dataset featuring 1,412 human-annotated tasks across six categories, grounded in a realistic 3D urban simulator. Moreover, we propose Planner-Manager-Actor (PMA), a novel agent tailored for CityEQA. PMA enables long-horizon planning and hierarchical task execution: the Planner breaks down the question answering into sub-tasks, the Manager maintains an object-centric cognitive map for spatial reasoning during the process control, and the specialized Actors handle navigation, exploration, and collection sub-tasks. Experiments demonstrate that PMA achieves 60.7% of human-level answering accuracy, significantly outperforming frontier-based baselines. While promising, the performance gap compared to humans highlights the need for enhanced visual reasoning in CityEQA. This work paves the way for future advancements in urban spatial intelligence. Dataset and code are available at this https URL. 

**Abstract (ZH)**: 基于体态的问答（EQA）主要集中在室内环境，而城市的复杂环境（包括环境、行动和感知）则被很大程度上忽略了。为了弥合这一差距，我们介绍了CityEQA，这是一个新任务，其中包括一个体态代理通过在动态城市环境中的主动探索来回答开放词汇的问题。为了支持这一任务，我们提出了CityEQA-EC，这是第一个基准数据集，包含了六类中的1,412个人工标注的任务，这些任务基于一个现实的3D城市仿真器。此外，我们提出了Planner-Manager-Actor（PMA），这是一种专门为CityEQA设计的新代理。PMA 能够实现长期规划和分层任务执行：规划器将问答问题分解为子任务，经理在过程中通过保持以对象为中心的认知地图来进行空间推理，专门的执行者则处理导航、探索和收集子任务。实验结果表明，PMA 实现了60.7%的人类级回答准确率，显著优于基于路径的基线模型。虽然具有前景，但与人类表现之间的差距凸显了CityEQA 中增强视觉推理的需求。这项工作为未来在城市空间智能方面的进步铺平了道路。数据集和代码可在此处访问：[https://...] 

---
# Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights 

**Title (ZH)**: 推理时计算在大规模语言模型推理与规划中的应用：基准与见解 

**Authors**: Shubham Parashar, Blake Olson, Sambhav Khurana, Eric Li, Hongyi Ling, James Caverlee, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.12521)  

**Abstract**: We examine the reasoning and planning capabilities of large language models (LLMs) in solving complex tasks. Recent advances in inference-time techniques demonstrate the potential to enhance LLM reasoning without additional training by exploring intermediate steps during inference. Notably, OpenAI's o1 model shows promising performance through its novel use of multi-step reasoning and verification. Here, we explore how scaling inference-time techniques can improve reasoning and planning, focusing on understanding the tradeoff between computational cost and performance. To this end, we construct a comprehensive benchmark, known as Sys2Bench, and perform extensive experiments evaluating existing inference-time techniques on eleven diverse tasks across five categories, including arithmetic reasoning, logical reasoning, common sense reasoning, algorithmic reasoning, and planning. Our findings indicate that simply scaling inference-time computation has limitations, as no single inference-time technique consistently performs well across all reasoning and planning tasks. 

**Abstract (ZH)**: 我们探讨了大型语言模型（LLMs）在解决复杂任务中的推理和规划能力。最近推理时技术的发展表明，通过在推理过程中探索中间步骤，可以在无需额外训练的情况下增强LLM的推理能力。值得注意的是，OpenAI的o1模型通过其新颖的多步推理和验证方法展示了令人鼓舞的性能。在此基础上，我们研究了如何通过扩展推理时技术来提高推理和规划能力，并重点探讨了计算成本与性能之间的权衡。为此，我们构建了一个全面的基准测试，称为Sys2Bench，并在五个类别（算术推理、逻辑推理、常识推理、算法推理和规划）中的十一个不同任务上对现有的推理时技术进行了广泛的实验评估。我们的研究发现，单纯扩展推理时计算的局限性在于，没有任何一种推理时技术能够在所有推理和规划任务中表现出一致的优良性能。 

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
# Computational Safety for Generative AI: A Signal Processing Perspective 

**Title (ZH)**: 生成型人工智能的计算安全性：一个信号处理视角 

**Authors**: Pin-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12445)  

**Abstract**: AI safety is a rapidly growing area of research that seeks to prevent the harm and misuse of frontier AI technology, particularly with respect to generative AI (GenAI) tools that are capable of creating realistic and high-quality content through text prompts. Examples of such tools include large language models (LLMs) and text-to-image (T2I) diffusion models. As the performance of various leading GenAI models approaches saturation due to similar training data sources and neural network architecture designs, the development of reliable safety guardrails has become a key differentiator for responsibility and sustainability. This paper presents a formalization of the concept of computational safety, which is a mathematical framework that enables the quantitative assessment, formulation, and study of safety challenges in GenAI through the lens of signal processing theory and methods. In particular, we explore two exemplary categories of computational safety challenges in GenAI that can be formulated as hypothesis testing problems. For the safety of model input, we show how sensitivity analysis and loss landscape analysis can be used to detect malicious prompts with jailbreak attempts. For the safety of model output, we elucidate how statistical signal processing and adversarial learning can be used to detect AI-generated content. Finally, we discuss key open research challenges, opportunities, and the essential role of signal processing in computational AI safety. 

**Abstract (ZH)**: 人工智能安全是一个迅速发展的研究领域，旨在预防尖端人工智能技术带来的危害和误用，尤其是生成型人工智能（GenAI）工具的误用，这些工具能够通过文本提示生成逼真且高质量的内容。这类工具包括大型语言模型（LLMs）和文本到图像（T2I）扩散模型。由于不同领先GenAI模型的性能接近饱和状态，归因于相似的训练数据来源和神经网络架构设计，因此开发可靠的防护措施已成为责任感和可持续性的关键差异化因素。本文提出了计算安全的规范化概念，这是一种数学框架，可以通过信号处理理论和方法来定量评估、表述和研究GenAI中的安全性挑战。特别是，我们探讨了两种可以表述为假设检验问题的计算安全挑战类别。对于模型输入的安全性，我们展示了敏感性分析和损失景观分析如何用于检测带有逃逸尝试的恶意提示。对于模型输出的安全性，我们阐明了如何利用统计信号处理和对抗性学习来检测AI生成的内容。最后，我们讨论了关键的开放研究挑战、机遇以及信号处理在计算人工智能安全中的核心作用。 

---
# A Survey on Large Language Models for Automated Planning 

**Title (ZH)**: 大型语言模型在自动规划中的综述 

**Authors**: Mohamed Aghzal, Erion Plaku, Gregory J. Stein, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12435)  

**Abstract**: The planning ability of Large Language Models (LLMs) has garnered increasing attention in recent years due to their remarkable capacity for multi-step reasoning and their ability to generalize across a wide range of domains. While some researchers emphasize the potential of LLMs to perform complex planning tasks, others highlight significant limitations in their performance, particularly when these models are tasked with handling the intricacies of long-horizon reasoning. In this survey, we critically investigate existing research on the use of LLMs in automated planning, examining both their successes and shortcomings in detail. We illustrate that although LLMs are not well-suited to serve as standalone planners because of these limitations, they nonetheless present an enormous opportunity to enhance planning applications when combined with other approaches. Thus, we advocate for a balanced methodology that leverages the inherent flexibility and generalized knowledge of LLMs alongside the rigor and cost-effectiveness of traditional planning methods. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的规划能力受到了越来越多的关注，这主要得益于它们在多步推理方面的能力及其在广泛领域中进行泛化的潜力。虽然一些研究人员强调LLMs在执行复杂规划任务方面的潜力，但也有人指出了它们在性能方面的一些显著限制，尤其是在处理长时程推理的复杂性方面。在本文综述中，我们批判性地调查了LLMs在自动化规划中的应用研究，详细探讨了它们的成功与不足。我们指出，尽管由于这些限制，LLMs并不适合作为独立的规划者，但在与其它方法结合使用时，它们仍具有极大的潜力来增强规划应用。因此，我们主张一种结合利用LLMs固有的灵活性和泛化知识以及传统规划方法的严格性和成本效益的平衡方法。 

---
# Integrating Expert Knowledge into Logical Programs via LLMs 

**Title (ZH)**: 通过大型语言模型将专家知识集成到逻辑程序中 

**Authors**: Franciszek Górski, Oskar Wysocki, Marco Valentino, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2502.12275)  

**Abstract**: This paper introduces ExKLoP, a novel framework designed to evaluate how effectively Large Language Models (LLMs) integrate expert knowledge into logical reasoning systems. This capability is especially valuable in engineering, where expert knowledge-such as manufacturer-recommended operational ranges-can be directly embedded into automated monitoring systems. By mirroring expert verification steps, tasks like range checking and constraint validation help ensure system safety and reliability. Our approach systematically evaluates LLM-generated logical rules, assessing both syntactic fluency and logical correctness in these critical validation tasks. We also explore the models capacity for self-correction via an iterative feedback loop based on code execution outcomes. ExKLoP presents an extensible dataset comprising 130 engineering premises, 950 prompts, and corresponding validation points. It enables comprehensive benchmarking while allowing control over task complexity and scalability of experiments. We leverage the synthetic data creation methodology to conduct extensive empirical evaluation on a diverse set of LLMs including Llama3, Gemma, Mixtral, Mistral, and Qwen. Results reveal that while models generate nearly perfect syntactically correct code, they frequently exhibit logical errors in translating expert knowledge. Furthermore, iterative self-correction yields only marginal improvements (up to 3%). Overall, ExKLoP serves as a robust evaluation platform that streamlines the selection of effective models for self-correcting systems while clearly delineating the types of errors encountered. The complete implementation, along with all relevant data, is available at GitHub. 

**Abstract (ZH)**: 本文介绍了ExKLoP，这是一种新型框架，旨在评估大型语言模型（LLMs）在将专家知识整合到逻辑推理系统中的有效性。在工程领域，这种能力尤为重要，因为供应商推荐的操作范围等专家知识可以直接嵌入到自动化监控系统中。通过模仿专家验证步骤，任务如范围检查和约束验证有助于确保系统的安全性和可靠性。我们的方法系统地评估了LLM生成的逻辑规则，不仅评估其句法流畅性，还评估其逻辑正确性。同时，我们还通过基于代码执行结果的迭代反馈循环探索模型的自我纠正能力。ExKLoP提供了一个可扩展的数据集，包含130个工程前提条件、950个提示以及相应的验证点。这使得全面基准测试成为可能，同时允许控制任务复杂性和实验的可扩展性。我们利用合成数据创建方法对包括Llama3、Gemma、Mixtral、Mistral和Qwen在内的多样化大型语言模型进行了广泛的实证评估。结果显示，尽管模型生成了接近完美的句法正确代码，但它们在将专家知识转换为逻辑表达时经常出现错误。此外，迭代自我纠正仅带来了微小的改进（最多3%）。总体而言，ExKLoP提供了一个稳健的评估平台，简化了选择用于自我纠正系统的有效模型的过程，同时明确地界定了遇到的不同类型的错误。完整的实现及所有相关数据均可在GitHub上获取。 

---
# Accurate Expert Predictions in MoE Inference via Cross-Layer Gate 

**Title (ZH)**: 通过跨层门控实现MoE推理中的准确专家预测 

**Authors**: Zhiyuan Fang, Zicong Hong, Yuegui Huang, Yufeng Lyu, Wuhui Chen, Yue Yu, Fan Yu, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12224)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across various tasks, and their application in edge scenarios has attracted significant attention. However, sparse-activated Mixture-of-Experts (MoE) models, which are well suited for edge scenarios, have received relatively little attention due to their high memory demands. Offload-based methods have been proposed to address this challenge, but they face difficulties with expert prediction. Inaccurate expert predictions can result in prolonged inference delays. To promote the application of MoE models in edge scenarios, we propose Fate, an offloading system designed for MoE models to enable efficient inference in resource-constrained environments. The key insight behind Fate is that gate inputs from adjacent layers can be effectively used for expert prefetching, achieving high prediction accuracy without additional GPU overhead. Furthermore, Fate employs a shallow-favoring expert caching strategy that increases the expert hit rate to 99\%. Additionally, Fate integrates tailored quantization strategies for cache optimization and IO efficiency. Experimental results show that, compared to Load on Demand and Expert Activation Path-based method, Fate achieves up to 4.5x and 1.9x speedups in prefill speed and up to 4.1x and 2.2x speedups in decoding speed, respectively, while maintaining inference quality. Moreover, Fate's performance improvements are scalable across different memory budgets. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务上表现出色，并且它们在边缘场景的应用引起了大量关注。然而，由于稀疏激活的专家混合（MoE）模型在边缘场景中具有较高的内存需求，这类模型在这些场景中受到的关注相对较少。为了应对这一挑战，已经提出了基于卸载的方法，但这些方法在专门预测专家方面存在困难。不准确的专家预测会导致推断延迟的延长。为了促进MoE模型在边缘场景中的应用，我们提出了一种名为Fate的卸载系统，该系统旨在为MoE模型提供在资源受限环境中高效推理的能力。Fate背后的關鍵洞察是，可以有效利用来自相邻层的门控输入来进行专家预取，在无需额外GPU开销的情况下实现高预测精度。此外，Fate还采用了一种倾向于浅层的专家缓存策略，将专家命中率提高到了99%。同时，Fate集成了针对缓存优化和IO效率量身定制的量化策略。实验结果显示，与按需加载和基于专家激活路径的方法相比，Fate在预填充速度上实现了最多4.5倍的加速，在解码速度上实现了最多2.2倍的加速，同时保持了推理质量。此外，Fate的性能改进在不同的内存预算下都是可扩展的。 

---
# Evaluating the Paperclip Maximizer: Are RL-Based Language Models More Likely to Pursue Instrumental Goals? 

**Title (ZH)**: 评估夹纸机最大化者：基于强化学习的语言模型更有可能追求工具性目标吗？ 

**Authors**: Yufei He, Yuexin Li, Jiaying Wu, Yuan Sui, Yulin Chen, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12206)  

**Abstract**: As large language models (LLMs) continue to evolve, ensuring their alignment with human goals and values remains a pressing challenge. A key concern is \textit{instrumental convergence}, where an AI system, in optimizing for a given objective, develops unintended intermediate goals that override the ultimate objective and deviate from human-intended goals. This issue is particularly relevant in reinforcement learning (RL)-trained models, which can generate creative but unintended strategies to maximize rewards. In this paper, we explore instrumental convergence in LLMs by comparing models trained with direct RL optimization (e.g., the o1 model) to those trained with reinforcement learning from human feedback (RLHF). We hypothesize that RL-driven models exhibit a stronger tendency for instrumental convergence due to their optimization of goal-directed behavior in ways that may misalign with human intentions. To assess this, we introduce InstrumentalEval, a benchmark for evaluating instrumental convergence in RL-trained LLMs. Initial experiments reveal cases where a model tasked with making money unexpectedly pursues instrumental objectives, such as self-replication, implying signs of instrumental convergence. Our findings contribute to a deeper understanding of alignment challenges in AI systems and the risks posed by unintended model behaviors. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的不断进化，确保它们与人类目标和价值观的契合性仍然是一个紧迫的挑战。一个主要的关切点是\textit{工具性的趋同}（instrumental convergence），即AI系统在优化特定目标的过程中，发展出未预见到的中间目标，这些中间目标不仅会凌驾于最终目标之上，还会偏离人类的初衷。这一问题特别适用于从强化学习（RL）训练的模型，这些模型可能会生成一些创造性的但未预见的策略来最大化奖励。在本文中，我们通过比较直接使用RL优化训练的模型（如o1模型）与从人类反馈中学习的强化学习（RLHF）训练的模型，研究LLMs中的工具性趋同现象。我们假设由RL驱动的模型更容易出现工具性趋同，因为它们优化目标导向行为的方式可能与人类意图相偏差。为了评估这一点，我们引入了InstrumentalEval，这是一个用于评估RL训练的LLMs中工具性趋同情况的基准测试。初步实验显示，一个旨在盈利的模型可能会意外地追求工具性目标，如自我复制，这表明了工具性趋同的迹象。我们的研究结果为理解AI系统中的契合性挑战以及由未预见模型行为带来的风险提供了更深入的见解。 

---
# SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation 

**Title (ZH)**: SoFar：基于语言的定向桥梁，连接空间推理与物体操作 

**Authors**: Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, Jiazhao Zhang, Jiawei He, Jiayuan Gu, Xin Jin, Kaisheng Ma, Zhizheng Zhang, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13143)  

**Abstract**: Spatial intelligence is a critical component of embodied AI, promoting robots to understand and interact with their environments. While recent advances have enhanced the ability of VLMs to perceive object locations and positional relationships, they still lack the capability to precisely understand object orientations-a key requirement for tasks involving fine-grained manipulations. Addressing this limitation not only requires geometric reasoning but also an expressive and intuitive way to represent orientation. In this context, we propose that natural language offers a more flexible representation space than canonical frames, making it particularly suitable for instruction-following robotic systems. In this paper, we introduce the concept of semantic orientation, which defines object orientations using natural language in a reference-frame-free manner (e.g., the ''plug-in'' direction of a USB or the ''handle'' direction of a knife). To support this, we construct OrienText300K, a large-scale dataset of 3D models annotated with semantic orientations that link geometric understanding to functional semantics. By integrating semantic orientation into a VLM system, we enable robots to generate manipulation actions with both positional and orientational constraints. Extensive experiments in simulation and real world demonstrate that our approach significantly enhances robotic manipulation capabilities, e.g., 48.7% accuracy on Open6DOR and 74.9% accuracy on SIMPLER. 

**Abstract (ZH)**: 空间智能是本体AI的关键组成部分，它能够促进机器人理解和与其环境互动。虽然最近的进步增强了视觉语言模型(VLMs)感知物体位置和位置关系的能力，但它们仍然缺乏精确理解物体方向的能力——这对于涉及精细操作的任务至关重要。解决这一局限不仅需要几何推理，还需要一种表达性和直观的方式来表示方向。在此背景下，我们认为自然语言提供了一个比标准坐标系更具弹性的表示空间，使其特别适合于遵循指令的机器人系统。在这篇论文中，我们提出了语义方向的概念，这种概念通过自然语言在无参考坐标系的方式下定义物体的方向（例如，USB接口的插入方向或刀具的把手方向）。为了支持这一点，我们构建了OrienText300K，这是一个大规模的3D模型数据集，其中包含语义方向的注释，将几何理解与功能语义联系起来。通过将语义方向集成到VLM系统中，使机器人能够生成既包含位置约束又包含方向约束的操纵动作。在仿真和现实世界中的大量实验表明，我们的方法显著增强了机器人的操纵能力，例如，在Open6DOR上的准确率为48.7%，在SIMPLER上的准确率为74.9%。 

---
# Pre-training Auto-regressive Robotic Models with 4D Representations 

**Title (ZH)**: 使用四维表示预训练自回归机器人模型 

**Authors**: Dantong Niu, Yuvan Sharma, Haoru Xue, Giscard Biamby, Junyi Zhang, Ziteng Ji, Trevor Darrell, Roei Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2502.13142)  

**Abstract**: Foundation models pre-trained on massive unlabeled datasets have revolutionized natural language and computer vision, exhibiting remarkable generalization capabilities, thus highlighting the importance of pre-training. Yet, efforts in robotics have struggled to achieve similar success, limited by either the need for costly robotic annotations or the lack of representations that effectively model the physical world. In this paper, we introduce ARM4R, an Auto-regressive Robotic Model that leverages low-level 4D Representations learned from human video data to yield a better pre-trained robotic model. Specifically, we focus on utilizing 3D point tracking representations from videos derived by lifting 2D representations into 3D space via monocular depth estimation across time. These 4D representations maintain a shared geometric structure between the points and robot state representations up to a linear transformation, enabling efficient transfer learning from human video data to low-level robotic control. Our experiments show that ARM4R can transfer efficiently from human video data to robotics and consistently improves performance on tasks across various robot environments and configurations. 

**Abstract (ZH)**: 基于大规模未标注数据集预训练的基座模型已经颠覆了自然语言处理和计算机视觉领域，展示了卓越的泛化能力，从而突显了预训练的重要性。然而，在机器人领域，取得类似成功的努力受到了昂贵的机器人标注成本或无法有效建模物理世界的代表性表示的限制。在这项工作中，我们引入了ARM4R，一种自回归机器人模型，该模型利用从人类视频数据中学习的低级四维表示，从而产生一种更优秀的预训练机器人模型。具体而言，我们重点关注通过单目深度估计将2D表示提升至3D空间中的3D点跟踪表示。这些四维表示在点与机器人状态表示之间保持共享的几何结构，直到线性变换，这使得能够从人类视频数据高效地转移到低级机器人控制。我们的实验结果表明，ARM4R可以从人类视频数据高效地转移到机器人领域，并在各种机器人环境和配置中一致地提高任务性能。 

---
# UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor Attacks and Adversarial Attacks in Large Language Models 

**Title (ZH)**: UniGuardian：统一防御方法以检测大规模语言模型中的提示注入攻击、后门攻击和 adversarial 攻击 

**Authors**: Huawei Lin, Yingjie Lao, Tong Geng, Tan Yu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13141)  

**Abstract**: Large Language Models (LLMs) are vulnerable to attacks like prompt injection, backdoor attacks, and adversarial attacks, which manipulate prompts or models to generate harmful outputs. In this paper, departing from traditional deep learning attack paradigms, we explore their intrinsic relationship and collectively term them Prompt Trigger Attacks (PTA). This raises a key question: Can we determine if a prompt is benign or poisoned? To address this, we propose UniGuardian, the first unified defense mechanism designed to detect prompt injection, backdoor attacks, and adversarial attacks in LLMs. Additionally, we introduce a single-forward strategy to optimize the detection pipeline, enabling simultaneous attack detection and text generation within a single forward pass. Our experiments confirm that UniGuardian accurately and efficiently identifies malicious prompts in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）容易受到提示植入攻击、后门攻击和对抗攻击的影响，这些攻击会操控提示或模型以生成有害输出。在本文中，我们摒弃了传统的深度学习攻击模式，探索这些攻击之间的内在联系，并将它们统称为提示触发攻击（PTA）。这引发了一个关键问题：我们能否确定一个提示是 benign 还是被污染的？为了解决这个问题，我们提出了 UniGuardian，这是首个用于检测 LLM 中提示植入攻击、后门攻击和对抗攻击的统一防御机制。此外，我们还引入了一次前向策略来优化检测管道，使攻击检测和文本生成能够在单次前向传递中同时进行。我们的实验结果证实，UniGuardian 能够准确且高效地识别 LLM 中的恶意提示。 

---
# Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions 

**Title (ZH)**: 失眠之夜，甜食之日：用于现实 Coaching 代理互动的具有健康状况的合成用户生成 

**Authors**: Taedong Yun, Eric Yang, Mustafa Safdari, Jong Ha Lee, Vaishnavi Vinod Kumar, S. Sara Mahdavi, Jonathan Amar, Derek Peyton, Reut Aharony, Andreas Michaelides, Logan Schneider, Isaac Galatzer-Levy, Yugang Jia, John Canny, Arthur Gretton, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2502.13135)  

**Abstract**: We present an end-to-end framework for generating synthetic users for evaluating interactive agents designed to encourage positive behavior changes, such as in health and lifestyle coaching. The synthetic users are grounded in health and lifestyle conditions, specifically sleep and diabetes management in this study, to ensure realistic interactions with the health coaching agent. Synthetic users are created in two stages: first, structured data are generated grounded in real-world health and lifestyle factors in addition to basic demographics and behavioral attributes; second, full profiles of the synthetic users are developed conditioned on the structured data. Interactions between synthetic users and the coaching agent are simulated using generative agent-based models such as Concordia, or directly by prompting a language model. Using two independently-developed agents for sleep and diabetes coaching as case studies, the validity of this framework is demonstrated by analyzing the coaching agent's understanding of the synthetic users' needs and challenges. Finally, through multiple blinded evaluations of user-coach interactions by human experts, we demonstrate that our synthetic users with health and behavioral attributes more accurately portray real human users with the same attributes, compared to generic synthetic users not grounded in such attributes. The proposed framework lays the foundation for efficient development of conversational agents through extensive, realistic, and grounded simulated interactions. 

**Abstract (ZH)**: 我们提出了一套端到端的框架，用于生成合成用户以评估旨在促进积极行为改变的互动代理，例如健康和生活方式指导。合成用户基于健康和生活方式条件，本研究中特别针对睡眠管理和糖尿病管理，从而确保与健康指导代理进行真实的互动。合成用户的生成分为两个阶段：首先，基于现实世界健康和生活方式因素（包括基本的人口统计学和行为特征）生成结构化数据；其次，根据结构化数据开发合成用户的完整档案。合成用户与指导代理之间的互动通过生成代理基模（如Concordia）或通过提示语言模型直接模拟。使用两个独立开发的睡眠和糖尿病指导代理作为案例研究，通过分析指导代理对合成用户需求和挑战的理解来验证该框架的有效性。最后，通过人类专家的多次盲评，证明了具有健康和行为特征的合成用户更准确地反映了具有相同特征的真实人类用户，而不仅仅是通用的、未基于这些特征的合成用户。所提出的框架为通过广泛的、真实且基于现实的模拟互动高效开发对话代理奠定了基础。 

---
# Learning to Defer for Causal Discovery with Imperfect Experts 

**Title (ZH)**: 基于不完美专家的因果发现中的学习延迟策略 

**Authors**: Oscar Clivio, Divyat Mahajan, Perouz Taslakian, Sara Magliacane, Ioannis Mitliagkas, Valentina Zantedeschi, Alexandre Drouin  

**Link**: [PDF](https://arxiv.org/pdf/2502.13132)  

**Abstract**: Integrating expert knowledge, e.g. from large language models, into causal discovery algorithms can be challenging when the knowledge is not guaranteed to be correct. Expert recommendations may contradict data-driven results, and their reliability can vary significantly depending on the domain or specific query. Existing methods based on soft constraints or inconsistencies in predicted causal relationships fail to account for these variations in expertise. To remedy this, we propose L2D-CD, a method for gauging the correctness of expert recommendations and optimally combining them with data-driven causal discovery results. By adapting learning-to-defer (L2D) algorithms for pairwise causal discovery (CD), we learn a deferral function that selects whether to rely on classical causal discovery methods using numerical data or expert recommendations based on textual meta-data. We evaluate L2D-CD on the canonical Tübingen pairs dataset and demonstrate its superior performance compared to both the causal discovery method and the expert used in isolation. Moreover, our approach identifies domains where the expert's performance is strong or weak. Finally, we outline a strategy for generalizing this approach to causal discovery on graphs with more than two variables, paving the way for further research in this area. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

将专家知识（例如，来自大型语言模型的知识）集成到因果发现算法中可能具有挑战性，尤其是在知识本身未被保证正确的情况下。专家建议可能与数据驱动的结果相矛盾，其可靠性也会因领域或特定查询而显著不同。现有的基于软约束或预测因果关系不一致的方法未能考虑到这些专业知识的差异。为了解决这一问题，我们提出了一种称为L2D-CD的方法，该方法用于评估专家建议的正确性，并最优地将其与数据驱动的因果发现结果相结合。通过为成对因果发现（PCD）适应学习推迟（L2D）算法，我们学习了一个推迟函数，该函数根据基于文本元数据的专家建议选择是依赖于使用数值数据的传统因果发现方法还是专家建议。我们使用经典的Tübingen成对数据集评估了L2D-CD，并证明了其在因果发现方法和孤立使用的专家之间具有优越的性能。此外，我们的方法可以识别出专家表现强或弱的领域。最后，我们提出了一个策略，将此方法扩展到具有超过两个变量的图上的因果发现，为该领域进一步研究铺平了道路。 

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
# SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation 

**Title (ZH)**: SongGen：一种用于文本到歌曲生成的一阶段自回归变压器模型 

**Authors**: Zihan Liu, Shuangrui Ding, Zhixiong Zhang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13128)  

**Abstract**: Text-to-song generation, the task of creating vocals and accompaniment from textual inputs, poses significant challenges due to domain complexity and data scarcity. Existing approaches often employ multi-stage generation procedures, resulting in cumbersome training and inference pipelines. In this paper, we propose SongGen, a fully open-source, single-stage auto-regressive transformer designed for controllable song generation. The proposed model facilitates fine-grained control over diverse musical attributes, including lyrics and textual descriptions of instrumentation, genre, mood, and timbre, while also offering an optional three-second reference clip for voice cloning. Within a unified auto-regressive framework, SongGen supports two output modes: mixed mode, which generates a mixture of vocals and accompaniment directly, and dual-track mode, which synthesizes them separately for greater flexibility in downstream applications. We explore diverse token pattern strategies for each mode, leading to notable improvements and valuable insights. Furthermore, we design an automated data preprocessing pipeline with effective quality control. To foster community engagement and future research, we will release our model weights, training code, annotated data, and preprocessing pipeline. The generated samples are showcased on our project page at this https URL , and the code will be available at this https URL . 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

基于文本的歌曲生成（text-to-song generation）是从文本输入生成人声和伴奏的任务，由于领域复杂性和数据稀缺性，这一任务面临诸多挑战。现有方法通常采用多阶段生成流程，导致训练和推断管道复杂且繁琐。在本文中，我们提出了一种全新的、开源的一阶段自回归变换器模型——SongGen，用于可控的歌曲生成。所提出模型在统一的自回归框架中提供了对多种音乐属性的精细控制，包括歌词、乐器描述、曲风、情绪和音色，同时还提供了可选的三秒参考片段用于人声克隆。在统一的自回归框架中，SongGen 支持两种输出模式：混合模式，可以直接生成人声和伴奏的混合体；双轨模式，分别合成它们，以提供更大的下游应用灵活性。我们为每种模式探索了多种不同的标记模式策略，获得了显著的改进并提供了宝贵见解。此外，我们设计了有效的数据预处理自动化管道，并实施了质量控制措施。为了促进社区参与和未来研究，我们将发布我们的模型权重、训练代码、标注数据和预处理管道。生成的样本将在我们的项目网页 [请替换为实际网址] 上展示，而代码则将在 [请替换为实际网址] 上提供。 

---
# Adapting Psycholinguistic Research for LLMs: Gender-inclusive Language in a Coreference Context 

**Title (ZH)**: 适配心理语言学研究于大规模语言模型：共指语境中的性别包容性语言 

**Authors**: Marion Bartl, Thomas Brendan Murphy, Susan Leavy  

**Link**: [PDF](https://arxiv.org/pdf/2502.13120)  

**Abstract**: Gender-inclusive language is often used with the aim of ensuring that all individuals, regardless of gender, can be associated with certain concepts. While psycholinguistic studies have examined its effects in relation to human cognition, it remains unclear how Large Language Models (LLMs) process gender-inclusive language. Given that commercial LLMs are gaining an increasingly strong foothold in everyday applications, it is crucial to examine whether LLMs in fact interpret gender-inclusive language neutrally, because the language they generate has the potential to influence the language of their users. This study examines whether LLM-generated coreferent terms align with a given gender expression or reflect model biases. Adapting psycholinguistic methods from French to English and German, we find that in English, LLMs generally maintain the antecedent's gender but exhibit underlying masculine bias. In German, this bias is much stronger, overriding all tested gender-neutralization strategies. 

**Abstract (ZH)**: 性别中立语言通常用于确保所有个体，无论其性别，都可以与特定概念相关联。尽管心理语言学研究已经考察了性别中立语言对人类认知的影响，但关于大型语言模型（LLMs）如何处理性别中立语言的问题仍不清楚。鉴于商业LLMs在日常应用中的地位日益增强，重要的是要研究LLMs是否实际上以中立的方式解释性别中立语言，因为它们生成的语言有可能影响用户的语言表达。本研究探讨了LLMs生成的核心参照词是否与给定的性别表达一致，或反映模型偏见。我们将心理语言学方法从法语和德语中借鉴到英语中，发现LLMs在英语中通常保持先行词的性别，但表现出潜在的男性偏向。在德语中，这种偏向更为强烈，能够超越所有测试的性别中性化策略。 

---
# Performance Evaluation of Large Language Models in Statistical Programming 

**Title (ZH)**: 统计编程中大型语言模型的性能评估 

**Authors**: Xinyi Song, Kexin Xie, Lina Lee, Ruizhe Chen, Jared M. Clark, Hao He, Haoran He, Jie Min, Xinlei Zhang, Simin Zheng, Zhiyang Zhang, Xinwei Deng, Yili Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.13117)  

**Abstract**: The programming capabilities of large language models (LLMs) have revolutionized automatic code generation and opened new avenues for automatic statistical analysis. However, the validity and quality of these generated codes need to be systematically evaluated before they can be widely adopted. Despite their growing prominence, a comprehensive evaluation of statistical code generated by LLMs remains scarce in the literature. In this paper, we assess the performance of LLMs, including two versions of ChatGPT and one version of Llama, in the domain of SAS programming for statistical analysis. Our study utilizes a set of statistical analysis tasks encompassing diverse statistical topics and datasets. Each task includes a problem description, dataset information, and human-verified SAS code. We conduct a comprehensive assessment of the quality of SAS code generated by LLMs through human expert evaluation based on correctness, effectiveness, readability, executability, and the accuracy of output results. The analysis of rating scores reveals that while LLMs demonstrate usefulness in generating syntactically correct code, they struggle with tasks requiring deep domain understanding and may produce redundant or incorrect results. This study offers valuable insights into the capabilities and limitations of LLMs in statistical programming, providing guidance for future advancements in AI-assisted coding systems for statistical analysis. 

**Abstract (ZH)**: 大型语言模型（LLMs）的编程能力已经彻底改变了自动代码生成，并为自动统计分析开辟了新的途径。然而，在这些生成的代码可以广泛应用之前，它们的准确性和质量需要系统地评估。尽管LLMs的重要性在不断增加，但关于由LLMs生成的统计代码的全面评估在文献中仍然较少。本文评估了包括两个版本的ChatGPT和一个版本的Llama在内的LLMs在SAS编程领域的表现，特别是在统计分析方面。我们的研究利用了一组涵盖多样化统计主题和数据集的统计分析任务。每个任务包括问题描述、数据集信息以及由人工验证的SAS代码。我们通过对LLMs生成的SAS代码的质量进行基于正确性、有效性、可读性、可执行性和输出结果准确性的人工专家评估，进行全面评估。评级分数的分析表明，虽然LLMs在生成语法正确的代码方面显示出一定的有用性，但在需要深入领域理解的任务中表现不佳，可能会生成冗余或错误的结果。本研究为LLMs在统计编程中的能力和局限性提供了宝贵的见解，并为未来辅助编程系统的进步提供了指导。 

---
# Near-Optimal Private Learning in Linear Contextual Bandits 

**Title (ZH)**: 近最优私密学习在线性上下文多臂老虎机中 

**Authors**: Fan Chen, Jiachun Li, Alexander Rakhlin, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13115)  

**Abstract**: We analyze the problem of private learning in generalized linear contextual bandits. Our approach is based on a novel method of re-weighted regression, yielding an efficient algorithm with regret of order $\sqrt{T}+\frac{1}{\alpha}$ and $\sqrt{T}/\alpha$ in the joint and local model of $\alpha$-privacy, respectively. Further, we provide near-optimal private procedures that achieve dimension-independent rates in private linear models and linear contextual bandits. In particular, our results imply that joint privacy is almost "for free" in all the settings we consider, partially addressing the open problem posed by Azize and Basu (2024). 

**Abstract (ZH)**: 我们分析了广义线性上下文臂_bandits_中的私人学习问题。我们的方法基于一种新颖的加权回归方法，从而获得一个在联合模型和局部模型中分别具有$\sqrt{T}+\frac{1}{\alpha}$和$\sqrt{T}/\alpha$后悔率的有效算法。此外，我们提供了接近最优的私人学习程序，能够在私人线性模型和线性上下文臂_bandits_中实现维度无关的速率。特别地，我们的结果表明，在我们考虑的所有设置中，联合隐私几乎是“免费”的，部分解决了Azize和Basu（2024）提出的开放问题。 

---
# Improving Clinical Question Answering with Multi-Task Learning: A Joint Approach for Answer Extraction and Medical Categorization 

**Title (ZH)**: 使用多任务学习改进临床问题回答：一种结合回答提取和医学分类的联合方法 

**Authors**: Priyaranjan Pattnayak, Hitesh Laxmichand Patel, Amit Agarwal, Bhargava Kumar, Srikant Panda, Tejaswini Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.13108)  

**Abstract**: Clinical Question Answering (CQA) plays a crucial role in medical decision-making, enabling physicians to extract relevant information from Electronic Medical Records (EMRs). While transformer-based models such as BERT, BioBERT, and ClinicalBERT have demonstrated state-of-the-art performance in CQA, existing models lack the ability to categorize extracted answers, which is critical for structured retrieval, content filtering, and medical decision support.
To address this limitation, we introduce a Multi-Task Learning (MTL) framework that jointly trains CQA models for both answer extraction and medical categorization. In addition to predicting answer spans, our model classifies responses into five standardized medical categories: Diagnosis, Medication, Symptoms, Procedure, and Lab Reports. This categorization enables more structured and interpretable outputs, making clinical QA models more useful in real-world healthcare settings.
We evaluate our approach on emrQA, a large-scale dataset for medical question answering. Results show that MTL improves F1-score by 2.2% compared to standard fine-tuning, while achieving 90.7% accuracy in answer categorization. These findings suggest that MTL not only enhances CQA performance but also introduces an effective mechanism for categorization and structured medical information retrieval. 

**Abstract (ZH)**: 临床问题回答（CQA）在医学决策中发挥着关键作用，使医生能够从电子医疗记录（EMRs）中提取相关信息。虽然基于变换器的模型如BERT、BioBERT和ClinicalBERT已经在CQA领域展示了最先进的性能，但现有的模型缺乏对提取答案进行分类的能力，这对于结构化检索、内容过滤及医学决策支持至关重要。

为了解决这一局限性，我们提出了一种多任务学习（MTL）框架，该框架能够同时训练CQA模型进行答案提取和医学分类。除了预测答案跨度外，我们的模型还将回答分类为五个标准化的医学类别之一：诊断、药物、症状、程序和实验室报告。这一分类使得输出更加结构化和可解释，从而使临床QA模型在实际医疗环境中更为有用。

我们在emrQA数据集上对我们的方法进行了评估，这是一个大规模的医学问答数据集。结果显示，与标准微调相比，MTL在F1分数上提高了2.2%，同时在答案分类方面达到了90.7%的准确率。这些发现表明，MTL不仅提升了CQA性能，还引入了一种有效的分类机制和结构化医学信息检索方法。 

---
# Text2World: Benchmarking Large Language Models for Symbolic World Model Generation 

**Title (ZH)**: 文本到世界：符号世界模型生成的大语言模型基准测试 

**Authors**: Mengkang Hu, Tianxing Chen, Yude Zou, Yuheng Lei, Qiguang Chen, Ming Li, Hongyuan Zhang, Wenqi Shao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.13092)  

**Abstract**: Recently, there has been growing interest in leveraging large language models (LLMs) to generate symbolic world models from textual descriptions. Although LLMs have been extensively explored in the context of world modeling, prior studies encountered several challenges, including evaluation randomness, dependence on indirect metrics, and a limited domain scope. To address these limitations, we introduce a novel benchmark, Text2World, based on planning domain definition language (PDDL), featuring hundreds of diverse domains and employing multi-criteria, execution-based metrics for a more robust evaluation. We benchmark current LLMs using Text2World and find that reasoning models trained with large-scale reinforcement learning outperform others. However, even the best-performing model still demonstrates limited capabilities in world modeling. Building on these insights, we examine several promising strategies to enhance the world modeling capabilities of LLMs, including test-time scaling, agent training, and more. We hope that Text2World can serve as a crucial resource, laying the groundwork for future research in leveraging LLMs as world models. The project page is available at this https URL. 

**Abstract (ZH)**: 近年来，利用大型语言模型（LLMs）从文本描述中生成符号世界模型的兴趣日益增加。虽然LLMs在世界建模的背景下已经被广泛研究，但之前的研究所遇到的挑战包括评价的随机性、依赖于间接指标以及领域范围有限。为了解决这些局限性，我们基于计划定义语言（PDDL）引入了一个新的基准，Text2World，并且采用了多种基于执行的评估指标以提供更稳健的评估。我们使用Text2World对标当前的LLMs，并发现大规模强化学习训练而成的推理模型表现最佳。然而，即使是最优秀的模型在世界建模方面仍表现出有限的能力。基于这些见解，我们探讨了几种有前景的策略来增强LLMs的世界建模能力，包括测试时扩展、智能体训练等。我们希望Text2World可以作为一个关键资源，为利用LLMs作为世界模型的未来研究奠定基础。该项目页面可从以下网址访问：[请在此处插入网址]。 

---
# BOLIMES: Boruta and LIME optiMized fEature Selection for Gene Expression Classification 

**Title (ZH)**: BOLIMES：基于Boruta和LIME的优化特征选择方法在基因表达分类中的应用 

**Authors**: Bich-Chung Phan, Thanh Ma, Huu-Hoa Nguyen, and Thanh-Nghi Do  

**Link**: [PDF](https://arxiv.org/pdf/2502.13080)  

**Abstract**: Gene expression classification is a pivotal yet challenging task in bioinformatics, primarily due to the high dimensionality of genomic data and the risk of overfitting. To bridge this gap, we propose BOLIMES, a novel feature selection algorithm designed to enhance gene expression classification by systematically refining the feature subset. Unlike conventional methods that rely solely on statistical ranking or classifier-specific selection, we integrate the robustness of Boruta with the interpretability of LIME, ensuring that only the most relevant and influential genes are retained. BOLIMES first employs Boruta to filter out non-informative genes by comparing each feature against its randomized counterpart, thus preserving valuable information. It then uses LIME to rank the remaining genes based on their local importance to the classifier. Finally, an iterative classification evaluation determines the optimal feature subset by selecting the number of genes that maximizes predictive accuracy. By combining exhaustive feature selection with interpretability-driven refinement, our solution effectively balances dimensionality reduction with high classification performance, offering a powerful solution for high-dimensional gene expression analysis. 

**Abstract (ZH)**: 基因表达分类是生物信息学中一个关键但具有挑战性的任务，主要是由于基因组数据的高维度性和过拟合的风险。为解决这一问题，我们提出了一种名为BOLIMES的新型特征选择算法，旨在通过系统地精炼特征子集来提高基因表达分类的性能。与仅依赖统计排名或特定分类器选择的传统方法不同，我们结合了Boruta的稳健性和LIME的可解释性，确保只保留最相关和最具影响力的基因。BOLIMES首先利用Boruta通过将每个特征与其随机化对应物进行比较来筛选出非信息性基因，从而保留有价值的信息。然后，它使用LIME根据剩余基因对分类器的局部重要性对其进行排序。最后，迭代分类评估通过选择最大化预测准确性的基因数量来确定最优特征子集。通过结合详尽的特征选择与可解释性驱动的精炼，我们的解决方案有效地平衡了维度降低与高分类性能，为高维基因表达分析提供了强大的解决方案。 

---
# Improved Fine-Tuning of Large Multimodal Models for Hateful Meme Detection 

**Title (ZH)**: 改进的大型多模态模型微调方法在侮辱性 meme 识别中的应用 

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2502.13061)  

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While large multimodal models have shown strong generalization across various tasks, they exhibit poor generalization to hateful meme detection due to the dynamic nature of memes tied to emerging social trends and breaking news. Recent work further highlights the limitations of conventional supervised fine-tuning for large multimodal models in this context. To address these challenges, we propose Large Multimodal Model Retrieval-Guided Contrastive Learning (LMM-RGCL), a novel two-stage fine-tuning framework designed to improve both in-domain accuracy and cross-domain generalization. Experimental results on six widely used meme classification datasets demonstrate that LMM-RGCL achieves state-of-the-art performance, outperforming agent-based systems such as VPD-PALI-X-55B. Furthermore, our method effectively generalizes to out-of-domain memes under low-resource settings, surpassing models like GPT-4o. 

**Abstract (ZH)**: 仇恨 meme 已成为互联网上的一个重要关切，需要建立稳健的自动检测系统。虽然大型多模态模型在各种任务中表现出较强的泛化能力，但它们在仇恨 meme 检测方面表现较差，因为 meme 的动态性质使得它们容易受到新兴社会趋势和突发新闻的影响。近期的工作进一步强调了在这一背景下，传统的监督细调方法对大型多模态模型的局限性。为应对这些挑战，我们提出了大型多模态模型检索引导对比学习 (LMM-RGCL)，这是一种新颖的两阶段细调框架，旨在提高领域内准确性和跨领域泛化能力。在六个广泛使用的 meme 分类数据集上的实验结果表明，LMM-RGCL 达到了最先进的性能，超越了基于代理系统的方法，如 VPD-PALI-X-55B。此外，在低资源环境下，我们的方法还能够有效地泛化到领域外的 meme，超越了如 GPT-4o 等模型。 

---
# LAMD: Context-driven Android Malware Detection and Classification with LLMs 

**Title (ZH)**: LAMD：基于上下文的Android恶意软件检测与分类方法（利用大型语言模型） 

**Authors**: Xingzhi Qian, Xinran Zheng, Yiling He, Shuo Yang, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2502.13055)  

**Abstract**: The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes. 

**Abstract (ZH)**: 移动应用的快速增长加剧了针对Android平台的恶意软件威胁。尽管存在众多检测方法，但它们经常难以应对不断演变的攻击、数据集偏见以及有限的解释性。大规模语言模型（LLMs）因其零-shot推理和推理论的能力提供了有希望的替代方案。然而，将LLMs应用于Android恶意软件检测存在两个关键挑战：（1）Android应用中的大量支持代码，往往包含数千个类，超出了LLMs的上下文限制，从而使恶意行为隐藏在良性功能之中；（2）Android应用的结构复杂性和相互依赖关系超过了LLMs基于序列的推理能力，导致代码分析碎片化，并妨碍恶意意图的推断。为了解决这些挑战，我们提出了一种实用的上下文驱动框架LAMD，以使基于LLM的Android恶意软件检测成为可能。LAMD 结合了关键上下文提取，以隔离安全关键代码区域并构建程序结构，然后采用逐级代码推理来逐步分析应用程序行为，从低级指令到高级语义，提供最终的预测和解释。还设计了一套有效的事实一致性验证机制，以减轻第一级LLM的混乱行为。在实际场景中的评估证明了LAMD 的有效性，为动态威胁环境中基于LLM的恶意软件分析奠定了可行的基础。 

---
# Natural Language Generation from Visual Sequences: Challenges and Future Directions 

**Title (ZH)**: 视觉序列的自然语言生成：挑战与未来方向 

**Authors**: Aditya K Surikuchi, Raquel Fernández, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2502.13034)  

**Abstract**: The ability to use natural language to talk about visual content is at the core of human intelligence and a crucial feature of any artificial intelligence system. Various studies have focused on generating text for single images. In contrast, comparatively little attention has been paid to exhaustively analyzing and advancing work on multiple-image vision-to-text settings. In this position paper, we claim that any task dealing with temporally ordered sequences of multiple images or frames is an instance of a broader, more general problem involving the understanding of intricate relationships between the visual content and the corresponding text. We comprehensively analyze five tasks that are instances of this problem and argue that they pose a common set of challenges and share similarities in terms of modeling and evaluation approaches. Based on the insights from these various aspects and stages of multi-image-to-text generation, we highlight several open questions and suggest future research directions. We believe that these directions can advance the understanding of complex phenomena in this domain and the development of better models. 

**Abstract (ZH)**: 利用自然语言讨论视觉内容的能力是人类智能的核心，并且是任何人工智能系统的关键特征。已有许多研究集中在生成单张图像的文本描述上。相比之下，对于涉及多张图像的视觉到文本转换的研究则相对较少，且主要集中于分析和推进该领域的研究工作。在本文中，我们主张任何涉及时间顺序排列的多张图像或帧的任务都是一个更广泛、更普遍问题的一种实例，即理解视觉内容与其对应文本之间的复杂关系。我们全面分析了五个这类问题的任务实例，并论证了它们共同面临的挑战，并在建模和评估方法上具有相似性。基于多图到文本生成的不同方面和阶段的见解，我们强调了几个亟待解决的问题，并建议未来的研究方向。我们认为，这些方向将推动对该领域复杂现象的理解并促进更优模型的发展。 

---
# Likelihood-Ratio Regularized Quantile Regression: Adapting Conformal Prediction to High-Dimensional Covariate Shifts 

**Title (ZH)**: 最大似然比正则化分位数回归：使一致预测适应高维协变量偏移 

**Authors**: Sunay Joshi, Shayan Kiyani, George Pappas, Edgar Dobriban, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2502.13030)  

**Abstract**: We consider the problem of conformal prediction under covariate shift. Given labeled data from a source domain and unlabeled data from a covariate shifted target domain, we seek to construct prediction sets with valid marginal coverage in the target domain. Most existing methods require estimating the unknown likelihood ratio function, which can be prohibitive for high-dimensional data such as images. To address this challenge, we introduce the likelihood ratio regularized quantile regression (LR-QR) algorithm, which combines the pinball loss with a novel choice of regularization in order to construct a threshold function without directly estimating the unknown likelihood ratio. We show that the LR-QR method has coverage at the desired level in the target domain, up to a small error term that we can control. Our proofs draw on a novel analysis of coverage via stability bounds from learning theory. Our experiments demonstrate that the LR-QR algorithm outperforms existing methods on high-dimensional prediction tasks, including a regression task for the Communities and Crime dataset, and an image classification task from the WILDS repository. 

**Abstract (ZH)**: 我们考虑在协变量迁移条件下的 conformal 预测问题。给定源自源域的带标签数据和来自协变量迁移的目标域的无标签数据，我们旨在为目标域构建具有有效边际覆盖的预测集。大多数现有方法需要估计未知的似然比函数，在处理高维数据（如图像）时这可能是相当困难的。为应对这一挑战，我们引入了一种似然比正则化分位数回归 (LR-QR) 算法，该算法结合了 pinball 损失和一种新的正则化选择，以构造一个阈值函数，而无需直接估计未知的似然比。我们证明了 LR-QR 方法在目标域中的覆盖能力达到了所需的水平，误差项较小且我们可以控制。我们的证明基于学习理论中有关覆盖的稳定性界的一种新颖分析。我们的实验表明，LR-QR 算法在高维预测任务中优于现有方法，包括 Communities and Crime 数据集的回归任务以及来自 WILDS 仓库的图像分类任务。 

---
# LLM-Powered Proactive Data Systems 

**Title (ZH)**: LLM驱动的主动型数据系统 

**Authors**: Sepanta Zeighami, Yiming Lin, Shreya Shankar, Aditya Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2502.13016)  

**Abstract**: With the power of LLMs, we now have the ability to query data that was previously impossible to query, including text, images, and video. However, despite this enormous potential, most present-day data systems that leverage LLMs are reactive, reflecting our community's desire to map LLMs to known abstractions. Most data systems treat LLMs as an opaque black box that operates on user inputs and data as is, optimizing them much like any other approximate, expensive UDFs, in conjunction with other relational operators. Such data systems do as they are told, but fail to understand and leverage what the LLM is being asked to do (i.e. the underlying operations, which may be error-prone), the data the LLM is operating on (e.g., long, complex documents), or what the user really needs. They don't take advantage of the characteristics of the operations and/or the data at hand, or ensure correctness of results when there are imprecisions and ambiguities. We argue that data systems instead need to be proactive: they need to be given more agency -- armed with the power of LLMs -- to understand and rework the user inputs and the data and to make decisions on how the operations and the data should be represented and processed. By allowing the data system to parse, rewrite, and decompose user inputs and data, or to interact with the user in ways that go beyond the standard single-shot query-result paradigm, the data system is able to address user needs more efficiently and effectively. These new capabilities lead to a rich design space where the data system takes more initiative: they are empowered to perform optimization based on the transformation operations, data characteristics, and user intent. We discuss various successful examples of how this framework has been and can be applied in real-world tasks, and present future directions for this ambitious research agenda. 

**Abstract (ZH)**: 借助大语言模型（LLM）的力量，我们现在有能力查询之前无法查询的数据，包括文本、图像和视频。然而，尽管有如此巨大的潜力，当前大多数利用LLM的数据系统仍具有反应性，反映出我们的社区希望将LLM映射到已知的抽象概念。大多数数据系统将LLM视为一个不透明的黑盒，它按照用户的输入和数据进行操作，优化它们与现有的其他关系运算符类似，优化它们并和其他近似且昂贵的用户定义函数（UDF）一起工作。这样的数据系统会按指示行事，但无法理解或利用LLM需要执行的操作（即可能错误的操作），正在操作的数据（例如，长而复杂的文档）或者用户真正需要什么。它们没有利用手头操作或数据的特定特征，也没有确保在不精确和歧义情况下结果的正确性。我们认为，数据系统需要更为主动：它们应该获得更多的自主权——借助LLM的力量来理解和重构用户输入和数据，并决定如何表示和处理这些操作和数据。通过允许数据系统解析、重写和分解用户输入和数据，或以超越标准的单次查询-结果范式的与用户交互方式，数据系统能够更高效地满足用户需求。这些新能力开辟了一个丰富多彩的设计空间，在这个空间里，数据系统能够更加主动地进行优化，基于转换操作、数据特征和用户意图进行优化。我们讨论了这一框架在实际任务中成功应用的例子，并提出了这一雄心勃勃的研究议程的未来发展方向。 

---
# HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit 

**Title (ZH)**: HOMIE：类人双足操作与同构外 skeleton 控制舱 

**Authors**: Qingwei Ben, Feiyu Jia, Jia Zeng, Junting Dong, Dahua Lin, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13013)  

**Abstract**: Current humanoid teleoperation systems either lack reliable low-level control policies, or struggle to acquire accurate whole-body control commands, making it difficult to teleoperate humanoids for loco-manipulation tasks. To solve these issues, we propose HOMIE, a novel humanoid teleoperation cockpit integrates a humanoid loco-manipulation policy and a low-cost exoskeleton-based hardware system. The policy enables humanoid robots to walk and squat to specific heights while accommodating arbitrary upper-body poses. This is achieved through our novel reinforcement learning-based training framework that incorporates upper-body pose curriculum, height-tracking reward, and symmetry utilization, without relying on any motion priors. Complementing the policy, the hardware system integrates isomorphic exoskeleton arms, a pair of motion-sensing gloves, and a pedal, allowing a single operator to achieve full control of the humanoid robot. Our experiments show our cockpit facilitates more stable, rapid, and precise humanoid loco-manipulation teleoperation, accelerating task completion and eliminating retargeting errors compared to inverse kinematics-based methods. We also validate the effectiveness of the data collected by our cockpit for imitation learning. Our project is fully open-sourced, demos and code can be found in this https URL. 

**Abstract (ZH)**: 当前的人形远程操作系统要么缺乏可靠的低级控制策略，要么难以获取准确的全身控制指令，这使得通过远程操作人形机器人执行移动和操作任务变得非常困难。为了解决这些问题，我们提出了HOMIE（Humanoid Operation Manipulation Interactive Environment），这是一种新颖的人形远程操作座舱，集成了人形移动操作策略和低成本外骨骼硬件系统。该策略使人形机器人能够根据不同上身姿态的要求，在特定高度行走和蹲下。这一目标通过我们提出的新型基于强化学习的训练框架实现，该框架包括了上身姿态训练序列、高度追踪奖励和对称性利用，无需依赖任何动作先验知识。

除策略之外，硬件系统还集成了同构外骨骼手臂、一对运动传感手套和一个脚踏板，使得单名操作员能够完全控制人形机器人。我们的实验表明，HOMIE座舱能够实现更加稳定、快速和精确的人形远程移动操作，显著加快任务完成速度，并且相比于基于逆向动力学的方法，能够消除目标重新定位误差。我们还验证了由HOMIE采集的数据对模仿学习的有效性。该项目完全开源，可在以下链接找到示例和代码：[这里提供链接]。 

---
# Personalized Top-k Set Queries Over Predicted Scores 

**Title (ZH)**: 预测评分下的个性化Top-k集合查询 

**Authors**: Sohrab Namazi Nia, Subhodeep Ghosh, Senjuti Basu Roy, Sihem Amer-Yahia  

**Link**: [PDF](https://arxiv.org/pdf/2502.12998)  

**Abstract**: This work studies the applicability of expensive external oracles such as large language models in answering top-k queries over predicted scores. Such scores are incurred by user-defined functions to answer personalized queries over multi-modal data. We propose a generic computational framework that handles arbitrary set-based scoring functions, as long as the functions could be decomposed into constructs, each of which sent to an oracle (in our case an LLM) to predict partial scores. At a given point in time, the framework assumes a set of responses and their partial predicted scores, and it maintains a collection of possible sets that are likely to be the true top-k. Since calling oracles is costly, our framework judiciously identifies the next construct, i.e., the next best question to ask the oracle so as to maximize the likelihood of identifying the true top-k. We present a principled probabilistic model that quantifies that likelihood. We study efficiency opportunities in designing algorithms. We run an evaluation with three large scale datasets, scoring functions, and baselines. Experiments indicate the efficacy of our framework, as it achieves an order of magnitude improvement over baselines in requiring LLM calls while ensuring result accuracy. Scalability experiments further indicate that our framework could be used in large-scale applications. 

**Abstract (ZH)**: 本文研究了昂贵的外部或acles（如大型语言模型）在回答基于预测分数的个性化查询时的适用性。这些分数是由用户定义的函数为多模态数据生成的。我们提出了一种通用的计算框架，该框架能够处理任意的集基评分函数，只要这些函数可以分解为一些各自被发送给或acles（在这种情况下是LLM）预测部分分数的基本构建块。在某一时间点，该框架假设一组响应及其部分预测分数，并维护一个可能的集合，这些集合可能是真实的top-k结果。由于向或acles求解是比较昂贵的操作，我们的框架会精心选择下一个构建块，即向或acles提出下一个最佳问题，以最大化识别真实top-k结果的概率。我们提出了一种原理性的概率模型来量化这种概率。我们研究了算法设计中的效率提升机会。我们使用三个大规模数据集、评分函数和基线进行了评估。实验表明，与基线相比，我们的框架通过减少对LLM的调用次数来显著提高效果，同时保持结果准确性。进一步的可扩展性实验表明，我们的框架可以在大规模应用中使用。 

---
# B-cos LM: Efficiently Transforming Pre-trained Language Models for Improved Explainability 

**Title (ZH)**: B-cos LM：高效转换预训练语言模型以提高可解释性 

**Authors**: Yifan Wang, Sukrut Rao, Ji-Ung Lee, Mayank Jobanputra, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2502.12992)  

**Abstract**: Post-hoc explanation methods for black-box models often struggle with faithfulness and human interpretability due to the lack of explainability in current neural models. Meanwhile, B-cos networks have been introduced to improve model explainability through architectural and computational adaptations, but their application has so far been limited to computer vision models and their associated training pipelines. In this work, we introduce B-cos LMs, i.e., B-cos networks empowered for NLP tasks. Our approach directly transforms pre-trained language models into B-cos LMs by combining B-cos conversion and task fine-tuning, improving efficiency compared to previous B-cos methods. Our automatic and human evaluation results demonstrate that B-cos LMs produce more faithful and human interpretable explanations than post hoc methods, while maintaining task performance comparable to conventional fine-tuning. Our in-depth analysis explores how B-cos LMs differ from conventionally fine-tuned models in their learning processes and explanation patterns. Finally, we provide practical guidelines for effectively building B-cos LMs based on our findings. Our code is available at this https URL. 

**Abstract (ZH)**: 后验解释方法在黑盒模型中往往因为当前神经网络缺乏解释性而难以保证忠实性和人类可解释性。同时，B-cos网络已被引入以通过结构和计算上的改进来提高模型的解释性，但在目前的应用中，它们主要局限于计算机视觉模型及其相关的训练管道。在这项工作中，我们引入了B-cos LMs，即通过结合B-cos转换和任务微调来增强NLP任务的B-cos网络。我们的方法直接将预训练的语言模型转换为B-cos LMs，相比于之前的B-cos方法，这种方法在效率上有所提高。我们的自动和人工评估结果表明，B-cos LMs生成的解释比后验方法更加忠实和具有人类可解释性，同时保持与常规微调相当的任务性能。我们深入的分析探讨了B-cos LMs在学习过程和解释模式上与常规微调模型之间的差异。最后，我们根据我们的研究结果提供了构建B-cos LMs的有效指南。我们的代码可在以下链接获取：[此处提供链接]。 

---
# PartSDF: Part-Based Implicit Neural Representation for Composite 3D Shape Parametrization and Optimization 

**Title (ZH)**: PartSDF：基于部件的隐式神经表示方法，用于复合三维形状的参数化和优化 

**Authors**: Nicolas Talabot, Olivier Clerc, Arda Cinar Demirtas, Doruk Oner, Pascal Fua  

**Link**: [PDF](https://arxiv.org/pdf/2502.12985)  

**Abstract**: Accurate 3D shape representation is essential in engineering applications such as design, optimization, and simulation. In practice, engineering workflows require structured, part-aware representations, as objects are inherently designed as assemblies of distinct components. However, most existing methods either model shapes holistically or decompose them without predefined part structures, limiting their applicability in real-world design tasks. We propose PartSDF, a supervised implicit representation framework that explicitly models composite shapes with independent, controllable parts while maintaining shape consistency. Despite its simple single-decoder architecture, PartSDF outperforms both supervised and unsupervised baselines in reconstruction and generation tasks. We further demonstrate its effectiveness as a structured shape prior for engineering applications, enabling precise control over individual components while preserving overall coherence. Code available at this https URL. 

**Abstract (ZH)**: 准确的三维形状表示在工程应用中（如设计、优化和仿真）至关重要。在实际应用中，工程工作流需要结构化的、部件感知的表示方式，因为物体通常被设计为由多个不同部件组成的组件集合。然而，现有的大多数方法或整体建模形状，或在没有预定义部件结构的情况下将其分解，这限制了其在实际设计任务中的应用。我们提出了一种名为PartSDF的监督隐式表示框架，该框架明确地以独立且可控的方式建模复合形状，同时保持形状一致性。尽管其具有简单的单解码器架构，但PartSDF在重建和生成任务中均优于监督式和无监督式的基线方法。此外，我们进一步展示了其在工程应用中作为结构化形状先验的有效性，能够在保持整体一致性的前提下对各个部件进行精确控制。代码请参见此[链接]。 

---
# Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs 

**Title (ZH)**: Sailor2：以包容性多语言大型语言模型在东南亚航行 

**Authors**: Longxu Dou, Qian Liu, Fan Zhou, Changyu Chen, Zili Wang, Ziqi Jin, Zichen Liu, Tongyao Zhu, Cunxiao Du, Penghui Yang, Haonan Wang, Jiaheng Liu, Yongchi Zhao, Xiachong Feng, Xin Mao, Man Tsung Yeung, Kunat Pipatanakul, Fajri Koto, Min Si Thu, Hynek Kydlíček, Zeyi Liu, Qunshu Lin, Sittipong Sripaisarnmongkol, Kridtaphad Sae-Khow, Nirattisai Thongchim, Taechawat Konkaew, Narong Borijindargoon, Anh Dao, Matichon Maneegard, Phakphum Artkaew, Zheng-Xin Yong, Quan Nguyen, Wannaphong Phatthiyaphaibun, Hoang H. Tran, Mike Zhang, Shiqi Chen, Tianyu Pang, Chao Du, Xinyi Wan, Wei Lu, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.12982)  

**Abstract**: Sailor2 is a family of cutting-edge multilingual language models for South-East Asian (SEA) languages, available in 1B, 8B, and 20B sizes to suit diverse applications. Building on Qwen2.5, Sailor2 undergoes continuous pre-training on 500B tokens (400B SEA-specific and 100B replay tokens) to support 13 SEA languages while retaining proficiency in Chinese and English. Sailor2-20B model achieves a 50-50 win rate against GPT-4o across SEA languages. We also deliver a comprehensive cookbook on how to develop the multilingual model in an efficient manner, including five key aspects: data curation, pre-training, post-training, model customization and evaluation. We hope that Sailor2 model (Apache 2.0 license) will drive language development in the SEA region, and Sailor2 cookbook will inspire researchers to build more inclusive LLMs for other under-served languages. 

**Abstract (ZH)**: Sailor2 是用于东南亚（SEA）语言的一系列前沿多语言语言模型，提供1B、8B和20B三种规模的版本，以适应多样化的应用需求。基于Qwen2.5，Sailor2 经过连续预训练（包括400B专用于SEA的语言的token和100B的回放token共500B token），从而支持13种SEA语言，并保持对中文和英语的精通能力。Sailor2-20B模型在与GPT-4o的SEA语言对话中达到了50-50的胜率。我们还提供了一份全面的手册，介绍如何高效地开发多语言模型，包括五个关键方面：数据整理、预训练、后训练、模型定制和评估。我们希望Sailor2模型（采用Apache 2.0许可协议）能够推动SEA地区的语言发展，而Sailor2手册能够激发研究人员为其他尚未充分服务的语言构建更具包容性的大型语言模型（LLMs）。 

---
# Time-series attribution maps with regularized contrastive learning 

**Title (ZH)**: 正则化对比学习的时间序列归属图 

**Authors**: Steffen Schneider, Rodrigo González Laiz, Anastasiia Filippova, Markus Frey, Mackenzie Weygandt Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2502.12977)  

**Abstract**: Gradient-based attribution methods aim to explain decisions of deep learning models but so far lack identifiability guarantees. Here, we propose a method to generate attribution maps with identifiability guarantees by developing a regularized contrastive learning algorithm trained on time-series data plus a new attribution method called Inverted Neuron Gradient (collectively named xCEBRA). We show theoretically that xCEBRA has favorable properties for identifying the Jacobian matrix of the data generating process. Empirically, we demonstrate robust approximation of zero vs. non-zero entries in the ground-truth attribution map on synthetic datasets, and significant improvements across previous attribution methods based on feature ablation, Shapley values, and other gradient-based methods. Our work constitutes a first example of identifiable inference of time-series attribution maps and opens avenues to a better understanding of time-series data, such as for neural dynamics and decision-processes within neural networks. 

**Abstract (ZH)**: 基于梯度的归因方法旨在解释深度学习模型的决策，但迄今缺乏可识别性保证。为了解决这一问题，我们提出了一种通过开发一种正则化对比学习算法来生成具有可识别性保证的归因图的方法，该算法基于时间序列数据训练，并引入了一种新的归因方法，称为反转神经元梯度（集体命名为xcebra）。理论分析表明，xcebra 具有识别数据生成过程雅可比矩阵的有利属性。在实验中，我们证明了在合成数据集上，xcebra 能够稳健地近似真实归因图中的零值和非零值，并且相较于基于特征消减、Shapley 值和其他基于梯度的方法，提高了先前归因方法的效果。我们的研究是可识别的时间序列归因图推断的首个示例，为更好地理解时间序列数据，例如神经动力学和神经网络中的决策过程，开辟了新的途径。 

---
# A Survey of Text Classification Under Class Distribution Shift 

**Title (ZH)**: 文本分类中的类别分布偏移综述 

**Authors**: Adriana Valentina Costache, Silviu Florin Gheorghe, Eduard Gabriel Poesina, Paul Irofti, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12965)  

**Abstract**: The basic underlying assumption of machine learning (ML) models is that the training and test data are sampled from the same distribution. However, in daily practice, this assumption is often broken, i.e.~the distribution of the test data changes over time, which hinders the application of conventional ML models. One domain where the distribution shift naturally occurs is text classification, since people always find new topics to discuss. To this end, we survey research articles studying open-set text classification and related tasks. We divide the methods in this area based on the constraints that define the kind of distribution shift and the corresponding problem formulation, i.e.~learning with the Universum, zero-shot learning, and open-set learning. We next discuss the predominant mitigation approaches for each problem setup. Finally, we identify several future work directions, aiming to push the boundaries beyond the state of the art. Interestingly, we find that continual learning can solve many of the issues caused by the shifting class distribution. We maintain a list of relevant papers at this https URL. 

**Abstract (ZH)**: 机器学习（ML）模型的基本假设是训练数据和测试数据来自相同的数据分布。然而，在实际应用中，这种假设往往被打破，即测试数据的分布随时间变化，这妨碍了传统ML模型的应用。这种数据分布变化自然会在文本分类领域发生，因为人们总是在讨论新的话题。为了应对这一挑战，我们对研究开放集文本分类及相关任务的论文进行了综述。我们将这些方法根据定义的数据分布变化类型及其相应的问题建模方式进行分类，具体包括使用Universum学习、零样本学习和开放集学习等方法。随后，我们讨论了每种问题设置下的主要缓解策略。最后，我们指出了若干未来研究方向，旨在推动研究前沿超越现有水平。有趣的是，我们发现连续学习可以解决许多由类别分布变化引起的问题。我们在此维持了一篇相关论文的列表：[链接]。 

---
# AlignFreeze: Navigating the Impact of Realignment on the Layers of Multilingual Models Across Diverse Languages 

**Title (ZH)**: AlignFreeze：重新对齐对多语言模型各层跨多种语言的影响导航 

**Authors**: Steve Bakos, Félix Gaschi, David Guzmán, Riddhi More, Kelly Chutong Li, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.12959)  

**Abstract**: Realignment techniques are often employed to enhance cross-lingual transfer in multilingual language models, still, they can sometimes degrade performance in languages that differ significantly from the fine-tuned source language. This paper introduces AlignFreeze, a method that freezes either the layers' lower half or upper half during realignment. Through controlled experiments on 4 tasks, 3 models, and in 35 languages, we find that realignment affects all the layers but can be the most detrimental to the lower ones. Freezing the lower layers can prevent performance degradation. Particularly, AlignFreeze improves Part-of-Speech (PoS) tagging performances in languages where full realignment fails: with XLM-R, it provides improvements of more than one standard deviation in accuracy in seven more languages than full realignment. 

**Abstract (ZH)**: 重新对齐技术常被用来增强多语言语言模型中的跨语言迁移能力，但在某些情况下，它们可能会在与微调来源语言差异较大的语言中降低性能。本文介绍了一种名为AlignFreeze的方法，该方法在重新对齐过程中固定层的下半部分或上半部分。通过在4个任务、3个模型和35种语言上进行受控实验，我们发现重新对齐会影响所有层，但对下层的影响尤为严重。固定下层可以防止性能下降。特别是在那些全重新对齐失败的语言中，AlignFreeze提高了词性标注性能：与XLM-R结合使用时，它在7种更多语言中的准确率改进超过了一个标准差。 

---
# Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text 

**Title (ZH)**: 任务导向的掩蔽反课程学习提高文本下游性能 

**Authors**: Andrei Jarca, Florinel Alin Croitoru, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12953)  

**Abstract**: Masked language modeling has become a widely adopted unsupervised technique to pre-train language models. However, the process of selecting tokens for masking is random, and the percentage of masked tokens is typically fixed for the entire training process. In this paper, we propose to adjust the masking ratio and to decide which tokens to mask based on a novel task-informed anti-curriculum learning scheme. First, we harness task-specific knowledge about useful and harmful tokens in order to determine which tokens to mask. Second, we propose a cyclic decaying masking ratio, which corresponds to an anti-curriculum schedule (from hard to easy). We exemplify our novel task-informed anti-curriculum by masking (TIACBM) approach across three diverse downstream tasks: sentiment analysis, text classification by topic, and authorship attribution. Our findings suggest that TIACBM enhances the ability of the model to focus on key task-relevant features, contributing to statistically significant performance gains across tasks. We release our code at this https URL. 

**Abstract (ZH)**: 掩码语言模型已成为一种广泛应用的无监督技术，用于预训练语言模型。然而，选择用于掩码的标记过程是随机的，掩码标记的比例在整个训练过程中通常是固定的。在本文中，我们提出调整掩码比例，并基于一种新颖的任务导向逆阶梯学习方案来决定哪些标记被掩码。首先，我们利用特定任务的知识来确定哪些标记应被掩码。其次，我们提出了一种循环衰减的掩码比例，这对应于一种从难到易的逆阶梯学习计划。我们通过情感分析、主题分类和作者归属性三个不同的下游任务，展示了我们新颖的任务导向逆阶梯学习方法（TIACBM）的优势。我们的研究结果表明，TIACBM能够增强模型对关键任务相关信息的注意力，从而在多个任务中带来统计显著的性能提升。我们已将代码发布在以下链接：[此处替换为具体的网址]。 

---
# Fake It Till You Make It: Using Synthetic Data and Domain Knowledge for Improved Text-Based Learning for LGE Detection 

**Title (ZH)**: 弄假成真：通过合成数据和领域知识提高基于文本的学习以增强LGE检测

注释：LGE通常指的是左心室晚电位（Left Ventricular Late Potential），在心脏病学中是一个重要的概念，用于检测心脏电生理异常。在翻译时，根据上下文，我已将其解释为“LGE检测”，如果有更具体的医学背景或术语，可以进一步确认。 

**Authors**: Athira J Jacob, Puneet Sharma, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2502.12948)  

**Abstract**: Detection of hyperenhancement from cardiac LGE MRI images is a complex task requiring significant clinical expertise. Although deep learning-based models have shown promising results for the task, they require large amounts of data with fine-grained annotations. Clinical reports generated for cardiac MR studies contain rich, clinically relevant information, including the location, extent and etiology of any scars present. Although recently developed CLIP-based training enables pretraining models with image-text pairs, it requires large amounts of data and further finetuning strategies on downstream tasks. In this study, we use various strategies rooted in domain knowledge to train a model for LGE detection solely using text from clinical reports, on a relatively small clinical cohort of 965 patients. We improve performance through the use of synthetic data augmentation, by systematically creating scar images and associated text. In addition, we standardize the orientation of the images in an anatomy-informed way to enable better alignment of spatial and text features. We also use a captioning loss to enable fine-grained supervision and explore the effect of pretraining of the vision encoder on performance. Finally, ablation studies are carried out to elucidate the contributions of each design component to the overall performance of the model. 

**Abstract (ZH)**: 从心脏LGE MRI影像中检测高增强区域是一项复杂任务，需要显著的临床专业知识。尽管基于深度学习的模型在该任务上显示出有希望的结果，但它们需要大量的带有精细注释的数据。心脏MR研究产生的临床报告包含丰富的、具有临床意义的信息，包括疤痕的位置、范围和病因。虽然最近开发的CLIP（Contrastive Language–Image Pre-training）基于训练方法可以使用图像-文本对预训练模型，但这也需要大量数据和进一步针对下游任务的微调策略。在本研究中，我们利用根植于专业知识的各种策略，仅使用临床报告中的文本信息来训练一个LGE检测模型，仅在规模相对较小的965名患者的临床队列上进行。我们通过合成数据增强提升了性能，系统地创建了疤痕图像及其相关的文本。此外，我们以解剖导向的方式标准化图像的方向，以更好地对齐空间和文本特征。我们还使用说明损失来实现精细监督，并探讨视觉编码器预训练对性能的影响。最后，进行了消融研究以阐明每个设计组件对模型整体性能的贡献。 

---
# Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models 

**Title (ZH)**: 每位专家都重要：面向混合专家语言模型的有效知识蒸馏 

**Authors**: Gyeongman Kim, Gyouk Chu, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12947)  

**Abstract**: With the emergence of Mixture-of-Experts (MoE), the efficient scaling of model size has accelerated the development of large language models in recent years. However, their high memory requirements prevent their use in resource-constrained environments. While knowledge distillation (KD) has been a proven method for model compression, its application to MoE teacher models remains underexplored. Through our investigation, we discover that non-activated experts in MoE models possess valuable knowledge that benefits student models. We further demonstrate that existing KD methods are not optimal for compressing MoE models, as they fail to leverage this knowledge effectively. To address this, we propose two intuitive MoE-specific KD methods for the first time: Knowledge Augmentation (KA) and Student-Aware Router (SAR), both designed to effectively extract knowledge from all experts. Specifically, KA augments knowledge by sampling experts multiple times, while SAR uses all experts and adjusts the expert weights through router training to provide optimal knowledge. Extensive experiments show that our methods outperform conventional KD methods, demonstrating their effectiveness for MoE teacher models. 

**Abstract (ZH)**: 随着Mixture-of-Experts（MoE）的出现，模型规模的高效扩展近年来加速了大型语言模型的发展。然而，它们对内存的高需求限制了其在资源受限环境中的应用。尽管知识蒸馏（KD）已经被证实是一种有效的模型压缩方法，但其在MoE教师模型中的应用仍然相对未被充分探索。通过我们的研究，我们发现MoE模型中未激活的专家单元也蕴含着对学生模型有益的知识。进一步的研究显示，现有的知识蒸馏方法并未充分利用这些知识，从而无法优化地压缩MoE模型。为此，我们首次提出了两种针对MoE的直观特定知识蒸馏方法：知识增强（KA）和学生感知路由器（SAR），这两种方法旨在有效地从所有专家中提取知识。具体而言，KA通过多次采样专家来增强知识，而SAR利用所有专家并通过路由器训练调整专家权重，以提供最优知识。广泛的实验证明，我们的方法在传统知识蒸馏方法中表现更优，证明了其在MoE教师模型中的有效性。 

---
# Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options 

**Title (ZH)**: 选项流：通过考虑选项进行多样性和改进的语言模型推理 

**Authors**: Lakshmi Nair, Ian Trase, Mark Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.12929)  

**Abstract**: We present a novel reasoning approach called Flow-of-Options (FoO), designed to address intrinsic biases in Large Language Models (LLMs). FoO enables LLMs to systematically explore a diverse range of possibilities in their reasoning, as demonstrated by an FoO-based agentic system for autonomously solving Machine Learning tasks (AutoML). Our framework outperforms state-of-the-art baselines, achieving improvements of 38.2% - 69.2% on standard data science tasks, and 37.4% - 47.9% on therapeutic chemistry tasks. With an overall operation cost under $1 per task, our framework is well-suited for cost-sensitive applications. Beyond classification and regression, we illustrate the broader applicability of our FoO-based agentic system to tasks such as reinforcement learning and image generation. Our framework presents significant advancements compared to current state-of-the-art agentic systems for AutoML, due to the benefits of FoO in enforcing diversity in LLM solutions through compressed, explainable representations that also support long-term memory when combined with case-based reasoning. 

**Abstract (ZH)**: 我们提出了一种新的推理方法，称为“选项流（Flow-of-Options, FoO）”，旨在解决大型语言模型（LLMs）固有的偏差问题。FoO 方法使 LLMs 能够系统地探讨其推理过程中多样化的可能选项，这一能力通过基于 FoO 的自主系统（例如自主解决机器学习任务的 AutoML 系统）得到了验证。我们的框架在标准数据科学任务中优于最先进的基线方法，取得了38.2%至69.2%的性能提升，在治疗化学任务中则取得了37.4%至47.9%的性能提升。由于总体操作成本低于每任务1美元，该框架非常适合成本敏感的应用场景。除了分类和回归任务之外，我们还展示了基于 FoO 的自主系统在强化学习和图像生成等更广泛任务中的适用性。由于 FoO 在压缩和可解释表征中支持多样化的 LLM 解决方案，并且在结合基于案例的推理时支持长期记忆，因此我们的框架相比当前最先进的 AutoML 自主系统展现出了显著的进步。 

---
# Keep what you need : extracting efficient subnetworks from large audio representation models 

**Title (ZH)**: 保留所需部分：从大型音频表示模型中提取高效子网络 

**Authors**: David Genova, Philippe Esling, Tom Hurlin  

**Link**: [PDF](https://arxiv.org/pdf/2502.12925)  

**Abstract**: Recently, research on audio foundation models has witnessed notable advances, as illustrated by the ever improving results on complex downstream tasks. Subsequently, those pretrained networks have quickly been used for various audio applications. These improvements have however resulted in a considerable increase both in size and complexity of these models. Along the environmental concerns this issue raises, this prevents the deployment of such networks on consumer-level devices, and precludes their use for real-time applications. Moreover, this appears contradictory with the specificity of the tasks for which these models are used, which are often simpler compared to extracting a rich, multi-purpose representation from any type of audio data. In this paper, we address this issue with a simple, yet effective method to extract lightweight specialist subnetworks from large foundation models. Specifically, we introduce learnable binary masks in-between the layers of a pretrained representation model. When training the end-to-end model on a downstream task, we add a sparsity-inducing loss to the overall objective, hence learning a compact subnetwork specialized on a single task. Importantly, the weights of the foundation model are kept frozen, resulting into low additional training costs. Once trained, the masked computational units can then be removed from the network, implying significant performance gains. We assess our method on three widespread audio foundation models, each based on a different backbone architecture, and illustrate its effectiveness on common audio representation evaluation tasks, as well as its versatility on both speech, music, and general audio. Code for reproducing the results and supporting webpage are available at this https URL 

**Abstract (ZH)**: 近年来，音频基础模型的研究取得了显著进展，这体现在其在复杂下游任务上的不断提高的表现上。随后，这些预训练网络被迅速应用于各种音频应用中。然而，这些改进也导致了模型大小和复杂性显著增加。这不仅引发了环境方面的担忧，还阻碍了这类网络在消费级设备上的部署，并限制了其在实时应用中的使用。此外，这与这些模型所针对的具体任务的特性相矛盾，这些任务通常比提取任何类型音频数据的丰富、多功能表示要简单得多。

在本文中，我们通过一个简单而有效的方法解决了这一问题，即从大型基础模型中提取轻量级的专业子网络。具体来说，我们在一个预训练表示模型的层之间引入可学习的二值掩码。在对下游任务进行端到端模型训练时，我们向总体目标中添加了稀疏性诱导损失，从而学习一个专注于单一任务的紧凑子网络。重要的是，基础模型的权重保持冻结，这导致了较低的额外训练成本。训练完成后，可以将带有掩码的计算单元从网络中移除，从而实现显著的性能提升。我们在三种广泛使用的音频基础模型上评估了我们的方法，每种模型基于不同的骨干架构，我们展示了其在常见音频表示评估任务中的有效性，以及在语音、音乐和一般音频方面的灵活性。可重复生成结果的代码和相关网页支持网址为：[在这里插入网址]。 

---
# Conditioning LLMs to Generate Code-Switched Text: A Methodology Grounded in Naturally Occurring Data 

**Title (ZH)**: 基于自然发生数据的条件生成代码混合文本的方法学：面向大语言模型的实现 

**Authors**: Maite Heredia, Gorka Labaka, Jeremy Barnes, Aitor Soroa  

**Link**: [PDF](https://arxiv.org/pdf/2502.12924)  

**Abstract**: Code-switching (CS) is still a critical challenge in Natural Language Processing (NLP). Current Large Language Models (LLMs) struggle to interpret and generate code-switched text, primarily due to the scarcity of large-scale CS datasets for training. This paper presents a novel methodology to generate CS data using LLMs, and test it on the English-Spanish language pair. We propose back-translating natural CS sentences into monolingual English, and using the resulting parallel corpus to fine-tune LLMs to turn monolingual sentences into CS. Unlike previous approaches to CS generation, our methodology uses natural CS data as a starting point, allowing models to learn its natural distribution beyond grammatical patterns. We thoroughly analyse the models' performance through a study on human preferences, a qualitative error analysis and an evaluation with popular automatic metrics. Results show that our methodology generates fluent code-switched text, expanding research opportunities in CS communication, and that traditional metrics do not correlate with human judgement when assessing the quality of the generated CS data. We release our code and generated dataset under a CC-BY-NC-SA license. 

**Abstract (ZH)**: 代码转换（CS）仍然是自然语言处理（NLP）中的一个关键挑战。现有的大规模语言模型（LLMs）在解释和生成代码转换文本方面存在困难，主要原因是缺乏大规模代码转换数据集进行训练。本文提出了一种新的方法来使用LLMs生成CS数据，并在英西语言对上进行了测试。我们提出使用自然的CS句子进行反向翻译生成单一语言的英语句子，并利用由此产生的平行语料库对LLMs进行微调，使其能够将单一语言句子转换为CS。与以往的CS生成方法不同，我们的方法使用自然的CS数据作为起点，从而使模型能够学习其自然分布，而不仅仅是语法模式。我们通过一项关于人类偏好的研究、定性的错误分析以及使用流行的自动评估指标进行评估，全面分析了模型的表现。结果显示，我们的方法生成了流畅的代码转换文本，扩展了CS通信的研究机会，并且传统的评估指标在评估生成的CS数据质量时与人类判断不相关。我们已将代码和生成的数据集在CC-BY-NC-SA许可下开源发布。 

---
# GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning 

**Title (ZH)**: GSQ-调谐：全量化训练中面向设备的细调的分组共享指数整数方法 

**Authors**: Sifan Zhou, Shuo Wang, Zhihang Yuan, Mingjia Shi, Yuzhang Shang, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12913)  

**Abstract**: Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to FP16-based fine-tuning while significantly reducing memory usage (50%). Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices. 

**Abstract (ZH)**: 大型语言模型（LLMs）微调技术已取得了显著成果，但传统的LLM微调方法面临着重大挑战：它们需要大量的浮点（FP）计算，处理敏感数据时会引发隐私问题，并且不适合资源受限的边缘设备。虽然参数高效微调（PEFT）技术可以减少可训练参数，但它们对浮点算术的依赖性使其与边缘硬件存在根本上的不兼容。在这项工作中，我们提出了一种新的框架，能够在设备上进行LLM微调，该框架在推理和训练过程中均消除了浮点运算的需要，名为GSQ-微调。其核心是组共享指数整数格式（Group-Shared Exponents Integer format），该格式使用参数组之间的共享指数高效地以整数格式表示模型参数。结合类似于LoRA的适配器时，这使得全整数微调成为可能，实现了内存和计算效率的双重提升。我们证明，我们的方法在准确度与FP16基线相当的情况下，可以显著减少内存使用（50%）。此外，与FP8相比，我们的方法可以在保持相同性能的情况下，降低5倍的功耗和11倍的芯片面积，从而使得大规模模型适应在边缘设备上成为可能。 

---
# Graph Neural Networks for Databases: A Survey 

**Title (ZH)**: 数据库中的图神经网络：一个综述 

**Authors**: Ziming Li, Youhuan Li, Yuyu Luo, Guoliang Li, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12908)  

**Abstract**: Graph neural networks (GNNs) are powerful deep learning models for graph-structured data, demonstrating remarkable success across diverse domains. Recently, the database (DB) community has increasingly recognized the potentiality of GNNs, prompting a surge of researches focusing on improving database systems through GNN-based approaches. However, despite notable advances, There is a lack of a comprehensive review and understanding of how GNNs could improve DB systems. Therefore, this survey aims to bridge this gap by providing a structured and in-depth overview of GNNs for DB systems. Specifically, we propose a new taxonomy that classifies existing methods into two key categories: (1) Relational Databases, which includes tasks like performance prediction, query optimization, and text-to-SQL, and (2) Graph Databases, addressing challenges like efficient graph query processing and graph similarity computation. We systematically review key methods in each category, highlighting their contributions and practical implications. Finally, we suggest promising avenues for integrating GNNs into Database systems. 

**Abstract (ZH)**: 图神经网络（GNNs）是处理图结构数据的强大深度学习模型，已经在众多领域取得了显著的成功。最近，数据库（DB）社区越来越认识到GNNs的潜力，从而引发了通过基于GNN的方法改进数据库系统的研究热潮。尽管取得了显著的进步，但迄今为止还缺乏关于GNNs如何改进数据库系统的全面概述和理解。因此，本综述旨在通过提供一个结构化和深入的概述来弥合这一缺口，即GNNs在数据库系统中的应用。具体而言，我们提出了一个新的分类体系，将现有方法分为两类关键类别：（1）关系型数据库，包括性能预测、查询优化和文本到SQL的任务；（2）图数据库，解决高效图查询处理和图相似度计算等挑战。我们系统性地回顾了每个类别中的关键方法，并强调其贡献和实际意义。最后，我们提出了将GNNs整合到数据库系统中的若干有前景的途径。 

---
# Soundwave: Less is More for Speech-Text Alignment in LLMs 

**Title (ZH)**: Soundwave: 少即是多，提升大语言模型中的语音-文本对齐效果 

**Authors**: Yuhao Zhang, Zhiheng Liu, Fan Bu, Ruiyu Zhang, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12900)  

**Abstract**: Existing end-to-end speech large language models (LLMs) usually rely on large-scale annotated data for training, while data-efficient training has not been discussed in depth. We focus on two fundamental problems between speech and text: the representation space gap and sequence length inconsistency. We propose Soundwave, which utilizes an efficient training strategy and a novel architecture to address these issues. Results show that Soundwave outperforms the advanced Qwen2-Audio in speech translation and AIR-Bench speech tasks, using only one-fiftieth of the training data. Further analysis shows that Soundwave still retains its intelligence during conversation. The project is available at this https URL. 

**Abstract (ZH)**: 现有的端到端语音大语言模型（LLM）通常依赖大规模标注数据进行训练，而数据高效训练尚未得到深入讨论。我们关注语音与文本间的两个基本问题：表示空间差距和序列长度不一致性。我们提出了Soundwave，该模型利用一种有效的训练策略和一种新型架构来解决这些问题。实验结果表明，Soundwave在语音翻译和AIR-Bench语音任务上的表现优于先进的Qwen2-Audio，仅使用了后者的五分之一训练数据。进一步的分析表明，Soundwave在对话中仍然保持着其智能。该项目可在以下链接处获取：[此处网址]。 

---
# PAFT: Prompt-Agnostic Fine-Tuning 

**Title (ZH)**: PAFT：提示无关的微调 

**Authors**: Chenxing Wei, Yao Shu, Mingwen Ou, Ying Tiffany He, Fei Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12859)  

**Abstract**: While Large Language Models (LLMs) adapt well to downstream tasks after fine-tuning, this adaptability often compromises prompt robustness, as even minor prompt variations can significantly degrade performance. To address this, we propose Prompt-Agnostic Fine-Tuning(PAFT), a simple yet effective approach that dynamically adjusts prompts during fine-tuning. This encourages the model to learn underlying task principles rather than overfitting to specific prompt formulations. PAFT operates in two stages: First, a diverse set of meaningful, synthetic candidate prompts is constructed. Second, during fine-tuning, prompts are randomly sampled from this set to create dynamic training inputs. Extensive experiments across diverse datasets and LLMs demonstrate that models trained with PAFT exhibit strong robustness and generalization across a wide range of prompts, including unseen ones. This enhanced robustness improves both model performance and inference speed while maintaining training efficiency. Ablation studies further confirm the effectiveness of PAFT. 

**Abstract (ZH)**: 虽然经过微调的大语言模型（LLMs）在下游任务中表现出很好的适应性，但这种适应性往往以牺牲提示稳健性为代价，即使是微小的提示变化也可能显著降低模型性能。为了解决这个问题，我们提出了一种简单而有效的方法——提示无关微调（PAFT），它在微调过程中动态调整提示。这种方法鼓励模型学习任务的基本原理，而不是过分拟合特定的提示公式。PAFT 采用两个阶段的操作：首先，构建一个包含多种有意义的合成候选提示集。其次，在微调过程中，从这个集合中随机采样提示来创建动态训练输入。在多种多样数据集和大语言模型上的广泛实验表明，使用PAFT训练的模型在各种提示（包括未见过的提示）下表现出强大的稳健性和泛化能力。这些增强的稳健性不仅能提高模型性能和推理速度，还能保持训练效率。消融实验进一步证实了PAFT的有效性。 

---
# Rejected Dialects: Biases Against African American Language in Reward Models 

**Title (ZH)**: 被拒斥的方言：奖励模型中对非洲裔美国人语言的偏见 

**Authors**: Joel Mire, Zubin Trivadi Aysola, Daniel Chechelnitsky, Nicholas Deas, Chrysoula Zerva, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.12858)  

**Abstract**: Preference alignment via reward models helps build safe, helpful, and reliable large language models (LLMs). However, subjectivity in preference judgments and the lack of representative sampling in preference data collection can introduce new biases, hindering reward models' fairness and equity. In this work, we introduce a framework for evaluating dialect biases in reward models and conduct a case study on biases against African American Language (AAL) through several experiments comparing reward model preferences and behavior on paired White Mainstream English (WME) and both machine-translated and human-written AAL corpora. We show that reward models are less aligned with human preferences when processing AAL texts vs. WME ones (-4\% accuracy on average), frequently disprefer AAL-aligned texts vs. WME-aligned ones, and steer conversations toward WME, even when prompted with AAL texts. Our findings provide a targeted analysis of anti-AAL biases at a relatively understudied stage in LLM development, highlighting representational harms and ethical questions about the desired behavior of LLMs concerning AAL. 

**Abstract (ZH)**: 通过奖励模型实现偏好对齐有助于构建安全、有用且可靠的大规模语言模型（LLMs）。然而，偏好判断中的主观性以及偏好数据收集中缺乏代表性样本可能会引入新的偏见，从而阻碍奖励模型的公平性和公正性。在本项工作中，我们提出了一个评估奖励模型方言偏见的框架，并通过对奖励模型在配对的白人主流英语（WME）语料和机器翻译及人工撰写的非洲裔美国人英语（AAL）语料上的偏好和行为进行若干实验，对该项工作进行了案例研究。结果显示，当处理AAL文本而非WME文本时，奖励模型的准确性降低约4%，经常偏好WME对齐的文本而非AAL对齐的文本，并且即使在提示中使用了AAL文本，也会引导对话趋向WME。我们的研究成果提供了一个针对LLM开发过程中相对较少研究阶段的反AAL偏见的详细分析，突显了关于LLM对AAL的期望行为的代表性伤害和伦理问题。 

---
# Integrating Arithmetic Learning Improves Mathematical Reasoning in Smaller Models 

**Title (ZH)**: 将算术学习集成至小型模型中可提高数学推理能力 

**Authors**: Neeraj Gangwar, Suma P Bhat, Nickvash Kani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12855)  

**Abstract**: While large models pre-trained on high-quality data exhibit excellent performance across various reasoning tasks, including mathematical reasoning (e.g. GSM8k, MultiArith), specializing smaller models to excel at mathematical reasoning remains a challenging problem. Common approaches to address this challenge include knowledge distillation, where smaller student models learn from large pre-trained teacher models, and data augmentation, such as rephrasing questions. Despite these efforts, smaller models struggle with arithmetic computations, leading to errors in mathematical reasoning. In this work, we focus on leveraging a programmatically generated arithmetic dataset to enhance the reasoning capabilities of smaller models. We investigate two key approaches to incorporate this dataset -- (1) intermediate fine-tuning, where a model is fine-tuned on the arithmetic dataset before being trained on a reasoning dataset, and (2) integrating the arithmetic dataset into the instruction-tuning mixture, allowing the model to learn arithmetic skills alongside general instruction-following abilities. Our experiments on multiple reasoning benchmarks demonstrate that incorporating an arithmetic dataset, whether through targeted fine-tuning or within the instruction-tuning mixture, enhances the models' arithmetic capabilities, which in turn improves their mathematical reasoning performance. 

**Abstract (ZH)**: 尽管在高质量数据上预先训练的大模型在各种推理任务中表现出色，包括数学推理（例如GSM8K、MultiArith），但使较小的模型在数学推理方面表现出色仍然是一个挑战。解决这一挑战的常见方法包括知识蒸馏，其中较小的学生模型从大型预先训练的教师模型中学习，以及数据增强，例如重新表述问题。尽管做出了这些努力，但较小的模型在算术计算方面仍然存在问题，导致数学推理上的错误。在本研究中，我们专注于利用程序生成的算术数据集来增强较小模型的推理能力。我们探讨了将该数据集纳入模型的两种关键方法：(1) 中间微调，即将模型在算术数据集上进行微调后再进行推理数据集的训练，(2) 将算术数据集整合进指示调优混合中，允许模型在学习算术技能的同时掌握一般指示遵循能力。我们在多个推理基准上的实验表明，无论是通过定向微调还是将其整合进指示调优混合中，利用算术数据集都能够增强模型的算术能力，从而提高其数学推理表现。 

---
# MeMo: Towards Language Models with Associative Memory Mechanisms 

**Title (ZH)**: MeMo：具有联想记忆机制的语言模型研究 

**Authors**: Fabio Massimo Zanzotto, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Leonardo Ranaldi, Davide Venditti, Federico Ranaldi, Cristina Giannone, Andrea Favalli, Raniero Romagnoli  

**Link**: [PDF](https://arxiv.org/pdf/2502.12851)  

**Abstract**: Memorization is a fundamental ability of Transformer-based Large Language Models, achieved through learning. In this paper, we propose a paradigm shift by designing an architecture to memorize text directly, bearing in mind the principle that memorization precedes learning. We introduce MeMo, a novel architecture for language modeling that explicitly memorizes sequences of tokens in layered associative memories. By design, MeMo offers transparency and the possibility of model editing, including forgetting texts. We experimented with the MeMo architecture, showing the memorization power of the one-layer and the multi-layer configurations. 

**Abstract (ZH)**: 记忆能力是基于Transformer的大型语言模型的一项基本能力，通过学习实现。本文提出了一种范式转变，设计了一种架构直接记忆文本，铭记记忆先于学习的原则。我们介绍了MeMo，一种新型的语言模型架构，明确地在分层关联记忆中记忆令牌序列。通过设计，MeMo 提供了透明性和模型编辑的可能性，包括忘记文本。我们对MeMo架构进行了实验，展示了单层和多层配置的记忆能力。 

---
# Reasoning and the Trusting Behavior of DeepSeek and GPT: An Experiment Revealing Hidden Fault Lines in Large Language Models 

**Title (ZH)**: DeepSeek和GPT的推理与信任行为：一项揭示大型语言模型隐含裂痕的实验研究 

**Authors**: Rubing Lu, João Sedoc, Arun Sundararajan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12825)  

**Abstract**: When encountering increasingly frequent performance improvements or cost reductions from a new large language model (LLM), developers of applications leveraging LLMs must decide whether to take advantage of these improvements or stay with older tried-and-tested models. Low perceived switching frictions can lead to choices that do not consider more subtle behavior changes that the transition may induce. Our experiments use a popular game-theoretic behavioral economics model of trust to show stark differences in the trusting behavior of OpenAI's and DeepSeek's models. We highlight a collapse in the economic trust behavior of the o1-mini and o3-mini models as they reconcile profit-maximizing and risk-seeking with future returns from trust, and contrast it with DeepSeek's more sophisticated and profitable trusting behavior that stems from an ability to incorporate deeper concepts like forward planning and theory-of-mind. As LLMs form the basis for high-stakes commercial systems, our results highlight the perils of relying on LLM performance benchmarks that are too narrowly defined and suggest that careful analysis of their hidden fault lines should be part of any organization's AI strategy. 

**Abstract (ZH)**: 当遇到由新大型语言模型（LLM）带来的日益频繁的功能改进或成本降低时，利用LLM的应用开发人员必须决定是否利用这些改进，还是继续使用较老但经过验证的模型。较低的感知转换障碍可能导致决策忽视了过渡过程中可能引发的更为微妙的行为变化。我们通过使用一个流行的博弈论行为经济学模型来信任行为来展示OpenAI和DeepSeek模型之间的明显差异。我们强调，当o1-mini和o3-mini模型将利润最大化和冒险与信任带来的未来收益相统一时，它们的信任行为出现了崩溃；而这也与DeepSeek模型更复杂且盈利的信任行为形成了对比，后者源自于能够整合更深层次概念如前瞻性规划和心智理论的能力。随着LLM成为高风险商业系统的基石，我们的研究结果突显了依赖于过于狭义定义的LLM性能基准的危害，并建议组织的AI战略应包括对它们隐含故障线的仔细分析。 

---
# Envious Explore and Exploit 

**Title (ZH)**: “嫉妒的探索与利用” 

**Authors**: Omer Ben-Porat, Yotam Gafni, Or Markovetzki  

**Link**: [PDF](https://arxiv.org/pdf/2502.12798)  

**Abstract**: Explore-and-exploit tradeoffs play a key role in recommendation systems (RSs), aiming at serving users better by learning from previous interactions. Despite their commercial success, the societal effects of explore-and-exploit mechanisms are not well understood, especially regarding the utility discrepancy they generate between different users. In this work, we measure such discrepancy using the economic notion of envy. We present a multi-armed bandit-like model in which every round consists of several sessions, and rewards are realized once per round. We call the latter property reward consistency, and show that the RS can leverage this property for better societal outcomes. On the downside, doing so also generates envy, as late-to-arrive users enjoy the information gathered by early-to-arrive users. We examine the generated envy under several arrival order mechanisms and virtually any anonymous algorithm, i.e., any algorithm that treats all similar users similarly without leveraging their identities. We provide tight envy bounds on uniform arrival and upper bound the envy for nudged arrival, in which the RS can affect the order of arrival by nudging its users. Furthermore, we study the efficiency-fairness trade-off by devising an algorithm that allows constant envy and approximates the optimal welfare in restricted settings. Finally, we validate our theoretical results empirically using simulations. 

**Abstract (ZH)**: 探索与利用权衡在推荐系统（RSs）中发挥着关键作用，旨在通过学习以往的交互来更好地服务用户。尽管这类机制在商业上取得了成功，但其社会影响尚未得到充分理解，特别是在它们造就的不同用户之间效用差异方面。在本项工作中，我们使用经济学中的嫉妒概念来度量这种差异。我们提出了一种类似于多臂 bandit 的模型，其中每轮包括多个会话，奖励在每轮结束后才得以实现。我们称这一特性为奖励一致性，并展示了推荐系统可以通过利用这一特性获得更好的社会效益。然而，这样做也会产生嫉妒，因为迟到的用户可以享受到早到用户积累的信息。我们考察了在不同到达顺序机制下和任何匿名算法（即处理所有类似用户方式相同的算法）下产生的嫉妒。我们对均匀到达情况下的嫉妒给出了紧界，并对推动力量到达情况下的嫉妒设定了上限，其中推荐系统可以通过引导其用户来影响到达顺序。此外，我们通过设计一个允许恒定嫉妒并能在受限条件下近似最优福利的算法，研究了效率与公平性的权衡。最后，我们通过仿真验证了我们的理论结果。 

---
# Unsupervised Anomaly Detection through Mass Repulsing Optimal Transport 

**Title (ZH)**: 无监督异常检测通过质量排斥最优传输 

**Authors**: Eduardo Fernandes Montesuma, Adel El Habazi, Fred Ngole Mboula  

**Link**: [PDF](https://arxiv.org/pdf/2502.12793)  

**Abstract**: Detecting anomalies in datasets is a longstanding problem in machine learning. In this context, anomalies are defined as a sample that significantly deviates from the remaining data. Meanwhile, optimal transport (OT) is a field of mathematics concerned with the transportation, between two probability measures, at least effort. In classical OT, the optimal transportation strategy of a measure to itself is the identity. In this paper, we tackle anomaly detection by forcing samples to displace its mass, while keeping the least effort objective. We call this new transportation problem Mass Repulsing Optimal Transport (MROT). Naturally, samples lying in low density regions of space will be forced to displace mass very far, incurring a higher transportation cost. We use these concepts to design a new anomaly score. Through a series of experiments in existing benchmarks, and fault detection problems, we show that our algorithm improves over existing methods. 

**Abstract (ZH)**: 在机器学习领域，检测数据集中的异常值是一个长期存在的问题。在此背景下，异常值被定义为与其余数据显著偏离的样本。同时，最优传输（OT）是数学领域的一个分支，研究在两个概率测度之间的传输过程中以最小努力完成传输的策略。在经典的最优传输中，一种测度对自己进行最优传输的策略是保持不变（即身份映射）。本文中，我们通过迫使样本重新分配其质量，同时保持最小努力的目标，来解决异常检测问题。我们称这种新的传输问题为质量排斥最优传输（MROT）。自然地，位于空间低密度区域的样本将被迫以更大的移动距离重新分配质量，从而增加传输成本。我们利用这些概念设计了一种新的异常检测评分方法。通过在现有基准测试和故障检测问题中进行一系列实验，我们表明我们的算法优于现有方法。 

---
# Evaluating link prediction: New perspectives and recommendations 

**Title (ZH)**: 评价链接预测：新的视角与建议 

**Authors**: Bhargavi Kalyani I, A Rama Prasad Mathi, Niladri Sett  

**Link**: [PDF](https://arxiv.org/pdf/2502.12777)  

**Abstract**: Link prediction (LP) is an important problem in network science and machine learning research. The state-of-the-art LP methods are usually evaluated in a uniform setup, ignoring several factors associated with the data and application specific needs. We identify a number of such factors, such as, network-type, problem-type, geodesic distance between the end nodes and its distribution over the classes, nature and applicability of LP methods, class imbalance and its impact on early retrieval, evaluation metric, etc., and present an experimental setup which allows us to evaluate LP methods in a rigorous and controlled manner. We perform extensive experiments with a variety of LP methods over real network datasets in this controlled setup, and gather valuable insights on the interactions of these factors with the performance of LP through an array of carefully designed hypotheses. Following the insights, we provide recommendations to be followed as best practice for evaluating LP methods. 

**Abstract (ZH)**: 链接预测（LP）是网络科学和机器学习研究中的一个重要问题。最先进的LP方法通常在一个统一的框架下进行评估，忽略了与数据和应用场景相关的一些因素。我们识别出若干此类因素，包括网络类型、问题类型、端节点之间的测地距离及其在类之间的分布、LP方法的性质及其适用性、类不平衡及其对早期检索的影响、评估指标等，并提出了一种实验框架，允许我们在严谨且受控的环境下评估LP方法。我们在此受控框架下，通过多种LP方法对真实网络数据集进行了广泛的实验，并通过精心设计的一系列假设，收集了关于这些因素与LP性能之间相互作用的有价值见解。基于这些见解，我们提供了在评估LP方法时应遵循的最佳实践建议。 

---
# Portable Reward Tuning: Towards Reusable Fine-Tuning across Different Pretrained Models 

**Title (ZH)**: 便携式奖励调优：向跨不同预训练模型的可复用微调迈进 

**Authors**: Daiki Chijiwa, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, Susumu Takeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12776)  

**Abstract**: While foundation models have been exploited for various expert tasks through fine-tuning, any foundation model will become outdated due to its old knowledge or limited capability. Thus the underlying foundation model should be eventually replaced by new ones, which leads to repeated cost of fine-tuning these new models. Existing work addresses this problem by inference-time tuning, i.e., modifying the output probabilities from the new foundation model with the outputs from the old foundation model and its fine-tuned model, which involves an additional overhead in inference by the latter two models. In this paper, we propose a new fine-tuning principle, Portable Reward Tuning (PRT), that reduces the inference overhead by its nature, based on the reformulation of fine-tuning as the reward maximization. Specifically, instead of fine-tuning parameters of the foundation models, PRT trains the reward model explicitly through the same loss function as in fine-tuning. During inference, the reward model can be used with any foundation model (with the same set of vocabularies or labels) through the formulation of reward maximization. Experimental results, covering both vision and language models, demonstrate that the PRT-trained model can achieve comparable accuracy to the existing work of inference-time tuning, with less inference cost. 

**Abstract (ZH)**: 尽管基础模型通过微调已被应用于各种专家任务，但任何基础模型都会因为其过时的知识或有限的能力而变得过时。因此，最终需要用新的基础模型来替换它们，这会导致频繁的成本高昂的重新微调过程。现有工作通过推理时微调（inference-time tuning）来解决这一问题，即通过修改新基础模型和旧基础模型及其微调模型的输出概率来调整输出概率，这需要后期两个模型额外的推理开销。在本文中，我们提出了一种新的微调原则——便携式奖励调优（Portable Reward Tuning, PRT），这种原则本质上可以减少推理开销，基于将微调重新定义为奖励最大化。具体而言，PRT 通过与微调相同的损失函数来训练奖励模型，而不是基础模型的参数。在推理过程中，通过奖励最大化的公式，奖励模型可以与任何基础模型（具有相同的词汇表或标签集）结合使用。实验结果，涵盖了视觉和语言模型，表明PRT 训练的模型在推理成本较低的情况下，能达到与推理时微调工作相当的准确性。 

---
# How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild 

**Title (ZH)**: 多语言环境下大型语言模型的幻觉程度有多少？关于自然环境下的多语言大型语言模型幻觉估计的研究 

**Authors**: Saad Obaid ul Islam, Anne Lauscher, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2502.12769)  

**Abstract**: In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models. 

**Abstract (ZH)**: 在信息误导的时代，幻觉——大型语言模型（LLMs）生成非事实性或不忠实响应的趋势——代表了它们全球效用的主要风险。尽管LLMs正在变得越来越多种语言，但关于检测和量化LLM幻觉的研究大多是（a）以英语为中心，并且（b）集中在机器翻译（MT）和摘要等任务上，而这些任务在现实世界中比开放的信息检索更少见。与此相反，我们的目标是跨语言评估知识密集型长格式问答中的LLM幻觉程度。为此，我们训练了一个多语言幻觉检测模型，并在30种语言和6种开源LLM家族中进行了大规模研究。我们从一个英语幻觉检测数据集开始，并依赖翻译（MT）来生成其他语言的（嘈杂）训练数据。我们还手动标注了五种高资源语言的黄金数据；然后我们证明，对于这些语言，在银色测试集（LLM生成的）和黄金测试集之间，幻觉率的估计值是相似的，这验证了使用银色数据来估计其他语言的幻觉率的有效性。在最终的幻觉率估计中，我们为30种语言构建了一个基于LLM生成提示和维基百科文章的知识密集型问答数据集。我们发现，虽然LLMs为高资源语言生成了更长的响应并包含了更多的幻觉性标记，但语言的数字化表示与其长度归一化后的幻觉率之间没有相关性。此外，我们发现较小的LLMs表现出比较大的模型更高的幻觉率。 

---
# R2-KG: General-Purpose Dual-Agent Framework for Reliable Reasoning on Knowledge Graphs 

**Title (ZH)**: R2-KG：知识图上可靠推理的通用双代理框架 

**Authors**: Sumin Jo, Junseong Choi, Jiho Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12767)  

**Abstract**: Recent studies have combined Large Language Models (LLMs) with Knowledge Graphs (KGs) to enhance reasoning, improving inference accuracy without additional training while mitigating hallucination. However, existing frameworks are often rigid, struggling to adapt to KG or task changes. They also rely heavily on powerful LLMs for reliable (i.e., trustworthy) reasoning. To address this, We introduce R2-KG, a plug-and-play, dual-agent framework that separates reasoning into two roles: an Operator (a low-capacity LLM) that gathers evidence and a Supervisor (a high-capacity LLM) that makes final judgments. This design is cost-efficient for LLM inference while still maintaining strong reasoning accuracy. Additionally, R2-KG employs an Abstention mechanism, generating answers only when sufficient evidence is collected from KG, which significantly enhances reliability. Experiments across multiple KG-based reasoning tasks show that R2-KG consistently outperforms baselines in both accuracy and reliability, regardless of the inherent capability of LLMs used as the Operator. Further experiments reveal that the single-agent version of R2-KG, equipped with a strict self-consistency strategy, achieves significantly higher-than-baseline reliability while reducing inference cost. However, it also leads to a higher abstention rate in complex KGs. Our findings establish R2-KG as a flexible and cost-effective solution for KG-based reasoning. It reduces reliance on high-capacity LLMs while ensuring trustworthy inference. 

**Abstract (ZH)**: 近年来，研究者们将大型语言模型（LLMs）与知识图谱（KGs）相结合，以增强推理能力，同时在不进行额外训练的情况下提高推理准确性，减轻幻觉问题。然而，现有的框架往往较为僵化，难以适应知识图谱或任务的变化。它们还高度依赖强大的LLM来进行可靠的（即可信的）推理。为解决这些问题，我们提出了R2-KG，这是一种即插即用的双智能体框架，将推理分为两种角色：操作员（一种低容量的LLM），负责收集证据；监督者（一种高容量的LLM），负责做出最终判决。这种设计在保持强大推理准确性的同时，还最大限度地降低了LLM推理的成本。此外，R2-KG 还采用了一种回避机制（Abstention mechanism），仅在从知识图谱中收集到足够的证据后才生成答案，这极大地提升了推理的可靠性。在多个基于知识图谱的推理任务中的实验显示，无论作为操作员使用的LLM本身的能力如何，R2-KG 在准确性和可靠性方面都始终优于基线方法。进一步的实验表明，单智能体版本的R2-KG，结合严格的自我一致性策略，能够在降低推理成本的同时显著提高可靠性，但在复杂的知识图谱中会导致更高的回避率。我们的研究结果确立了R2-KG 作为一种灵活且成本效益高的知识图谱推理解决方案的地位。它减少了对高容量LLM的依赖，并确保了可信的推理。 

---
# Efficient Machine Translation Corpus Generation: Integrating Human-in-the-Loop Post-Editing with Large Language Models 

**Title (ZH)**: 高效的机器翻译语料库生成：结合人类在环后编辑的大规模语言模型 

**Authors**: Kamer Ali Yuksel, Ahmet Gunduz, Abdul Baseet Anees, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12755)  

**Abstract**: This paper introduces an advanced methodology for machine translation (MT) corpus generation, integrating semi-automated, human-in-the-loop post-editing with large language models (LLMs) to enhance efficiency and translation quality. Building upon previous work that utilized real-time training of a custom MT quality estimation metric, this system incorporates novel LLM features such as Enhanced Translation Synthesis and Assisted Annotation Analysis, which improve initial translation hypotheses and quality assessments, respectively. Additionally, the system employs LLM-Driven Pseudo Labeling and a Translation Recommendation System to reduce human annotator workload in specific contexts. These improvements not only retain the original benefits of cost reduction and enhanced post-edit quality but also open new avenues for leveraging cutting-edge LLM advancements. The project's source code is available for community use, promoting collaborative developments in the field. The demo video can be accessed here. 

**Abstract (ZH)**: 本文介绍了机器翻译（MT）语料库生成的一种先进方法，通过结合半自动的人机在环后编辑与大型语言模型（LLMs），以提高效率和翻译质量。本系统在此前利用实时训练自定义MT质量评估指标的基础上，整合了增强翻译合成和辅助注释分析等新型LLM功能，分别提高初始翻译假说和质量评估的准确性。此外，该系统还采用了LLM驱动的伪标签技术和翻译推荐系统，以减少特定情境下的人工标注员工作量。这些改进不仅保留了原有益处，即降低成本和提高后编辑质量，同时也为利用最新的LLM技术进步开辟了新途径。该项目的源代码可供社区使用，促进该领域的合作发展。演示视频可通过以下链接访问。 

---
# MediaMind: Revolutionizing Media Monitoring using Agentification 

**Title (ZH)**: MediaMind：通过代理化革新媒体监控 

**Authors**: Ahmet Gunduz, Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12745)  

**Abstract**: In an era of rapid technological advancements, agentification of software tools has emerged as a critical innovation, enabling systems to function autonomously and adaptively. This paper introduces MediaMind as a case study to demonstrate the agentification process, highlighting how existing software can be transformed into intelligent agents capable of independent decision-making and dynamic interaction. Developed by aiXplain, MediaMind leverages agent-based architecture to autonomously monitor, analyze, and provide insights from multilingual media content in real time. The focus of this paper is on the technical methodologies and design principles behind agentifying MediaMind, showcasing how agentification enhances adaptability, efficiency, and responsiveness. Through detailed case studies and practical examples, we illustrate how the agentification of MediaMind empowers organizations to streamline workflows, optimize decision-making, and respond to evolving trends. This work underscores the broader potential of agentification to revolutionize software tools across various domains. 

**Abstract (ZH)**: 在快速的技术进步时代，软件工具的代理化已成为一种关键创新，使系统能够实现自主运行和适应性。本文以MediaMind为案例研究，展示了代理化过程，突出说明了如何通过将现有软件转化为能够独立做出决策并进行动态交互的智能代理来实现这一转变。MediaMind由aiXplain开发，利用基于代理的架构实时监控、分析多语种媒体内容，并提供洞察。本文的重点在于MediaMind代理化背后的技术和设计原则，展示了代理化如何增强系统的适应性、效率和响应性。通过详细的案例研究和实际例子，本文阐述了MediaMind的代理化是如何赋能组织简化工作流程、优化决策和应对不断变化的趋势。本文强调了代理化在各个领域革新软件工具的更广泛潜力。 

---
# "I know myself better, but not really greatly": Using LLMs to Detect and Explain LLM-Generated Texts 

**Title (ZH)**: “我了解自己，但并不是非常透彻”：使用大语言模型检测和解释大语言模型生成的文本 

**Authors**: Jiazhou Ji, Jie Guo, Weidong Qiu, Zheng Huang, Yang Xu, Xinru Lu, Xiaoyu Jiang, Ruizhe Li, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12743)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in generating human-like texts, but the potential misuse of such LLM-generated texts raises the need to distinguish between human-generated and LLM-generated content. This paper explores the detection and explanation capabilities of LLM-based detectors of LLM-generated texts, in the context of a binary classification task (human-generated texts vs LLM-generated texts) and a ternary classification task (human-generated texts, LLM-generated texts, and undecided). By evaluating on six close/open-source LLMs with different sizes, our findings reveal that while self-detection consistently outperforms cross-detection, i.e., LLMs can detect texts generated by themselves more accurately than those generated by other LLMs, the performance of self-detection is still far from ideal, indicating that further improvements are needed. We also show that extending the binary to the ternary classification task with a new class "Undecided" can enhance both detection accuracy and explanation quality, with improvements being statistically significant and consistent across all LLMs. We finally conducted comprehensive qualitative and quantitative analyses on the explanation errors, which are categorized into three types: reliance on inaccurate features (the most frequent error), hallucinations, and incorrect reasoning. These findings with our human-annotated dataset emphasize the need for further research into improving both self-detection and self-explanation, particularly to address overfitting issues that may hinder generalization. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生成类人类文本方面展示了令人印象深刻的能 力，但这种LLM生成文本的潜在滥用引发了区分人类生成内容与LLM生成内容的必要性。本论文探讨了基于LLM的检测器在二分类任务（人类生成文本 vs LLM生成文本）和三分类任务（人类生成文本、LLM生成文本和未定）中的检测和解释能力。通过在六个大小不同的闭源/开源LLM上进行评估，我们的发现表明，自我检测一贯优于跨检测，即LLMs能够更准确地检测自己生成的文本，而非其他LLM生成的文本，但自我检测的性能仍远未达到理想水平，表明进一步改进的需求依然存在。我们还展示，将二分类任务扩展为包含“未定”类别的三分类任务可以同时提高检测准确性和解释质量，且改进在所有LLM上具有统计学意义和一致性。最后，我们进行了全面的定性与定量分析，这些解释错误被分类为三类：依赖于不准确特征（最常见的错误）、幻觉和错误推理。我们使用的人工标注数据集的这些发现强调了进一步研究以改进自我检测和自我解释的必要性，特别是解决可能阻碍泛化的过拟合问题。 

---
# Beyond Seen Data: Improving KBQA Generalization Through Schema-Guided Logical Form Generation 

**Title (ZH)**: 超越可视数据：通过基于模式的逻辑形式生成提高知识图谱查询问答的一般化能力 

**Authors**: Shengxiang Gao, Jey Han Lau, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12737)  

**Abstract**: Knowledge base question answering (KBQA) aims to answer user questions in natural language using rich human knowledge stored in large KBs. As current KBQA methods struggle with unseen knowledge base elements at test time,we introduce SG-KBQA: a novel model that injects schema contexts into entity retrieval and logical form generation to tackle this issue. It uses the richer semantics and awareness of the knowledge base structure provided by schema contexts to enhance generalizability. We show that SG-KBQA achieves strong generalizability, outperforming state-of-the-art models on two commonly used benchmark datasets across a variety of test settings. Code will be released upon paper publication. 

**Abstract (ZH)**: 知识图谱问答（KBQA）旨在利用大规模知识库中存储的丰富人类知识来自然语言回答用户问题。由于当前的KBQA方法在测试时难以处理未见过的知识库元素，我们提出了SG-KBQA：一种新颖的模型，将模式上下文注入实体检索和逻辑形式生成中以应对这一问题。该模型利用模式上下文提供的更丰富的语义和对知识库结构的认知，提高其泛化能力。实验结果显示，SG-KBQA在多种测试设置下显著优于现有最先进的模型，在两个常用基准数据集上取得了优异表现。论文发表后将发布代码。 

---
# TREND: A Whitespace Replacement Information Hiding Method 

**Title (ZH)**: TREND：一种空白字符替换信息隐藏方法 

**Authors**: Malte Hellmeier, Hendrik Norkowski, Ernst-Christoph Schrewe, Haydar Qarawlus, Falk Howar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12710)  

**Abstract**: Large Language Models (LLMs) have gained significant popularity in recent years. Differentiating between a text written by a human and a text generated by an LLM has become almost impossible. Information hiding techniques such as digital watermarking or steganography can help by embedding information inside text without being noticed. However, existing techniques, such as linguistic-based or format-based methods, change the semantics or do not work on pure, unformatted text. In this paper, we introduce a novel method for information hiding termed TREND, which is able to conceal any byte-encoded sequence within a cover text. The proposed method is implemented as a multi-platform library using the Kotlin programming language, accompanied by a command-line tool and a web interface provided as examples of usage. By substituting conventional whitespace characters with visually similar Unicode whitespace characters, our proposed scheme preserves the semantics of the cover text without increasing the number of characters. Furthermore, we propose a specified structure for secret messages that enables configurable compression, encryption, hashing, and error correction. Our experimental benchmark comparison on a dataset of one million Wikipedia articles compares ten algorithms from literature and practice. It proves the robustness of our proposed method in various applications while remaining imperceptible to humans. We discuss the limitations of limited embedding capacity and further robustness, which guide implications for future work. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）获得了显著的流行。区分由人类书写和由LLM生成的文本几乎变得 impossible。通过在文本中不被察觉地嵌入信息，数字水印或隐写术等信息隐藏技术可以有所帮助。然而，现有方法如基于语言或基于格式的方法会改变语义或无法应用于纯无格式文本。在本文中，我们提出了一个名为TREND的新颖的信息隐藏方法，该方法能够在遮罩文本中隐藏任意字节编码序列。我们提议的方法是使用Kotlin编程语言实现的一个多平台库，并附带一个命令行工具和一个网络界面作为使用示例。通过用视觉相似的Unicode空白字符替换传统空白字符，我们提议的方案能够在不增加字符数量的情况下保留遮罩文本的语义。此外，我们还提出了一种特定的密文结构，该结构支持可配置的压缩、加密、哈希和错误修正。我们在包含一百万篇维基百科文章的数据集上进行了实验基准比较，对比了从文献和实践中引用的十种算法。实验结果证明了我们提出方法在各种应用中的稳健性，同时对人类来说是不可察觉的。我们讨论了嵌入容量有限和进一步增强稳健性的局限性，这为未来工作提供了指导。 

---
# Translate Smart, not Hard: Cascaded Translation Systems with Quality-Aware Deferral 

**Title (ZH)**: 聪明选择，而非努力工作：基于质量意识的级联翻译系统 

**Authors**: António Farinhas, Nuno M. Guerreiro, Sweta Agrawal, Ricardo Rei, André F.T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2502.12701)  

**Abstract**: Larger models often outperform smaller ones but come with high computational costs. Cascading offers a potential solution. By default, it uses smaller models and defers only some instances to larger, more powerful models. However, designing effective deferral rules remains a challenge. In this paper, we propose a simple yet effective approach for machine translation, using existing quality estimation (QE) metrics as deferral rules. We show that QE-based deferral allows a cascaded system to match the performance of a larger model while invoking it for a small fraction (30% to 50%) of the examples, significantly reducing computational costs. We validate this approach through both automatic and human evaluation. 

**Abstract (ZH)**: 较大的模型通常表现优于较小的模型，但会带来较高的计算成本。级联提供了一种潜在的解决方案。默认情况下，它使用较小的模型，并仅将部分实例递交给更大、更强大的模型。然而，设计有效的递接规则仍是一项挑战。在本文中，我们提出了一种简单而有效的方法，用于机器翻译，利用现有质量估计（QE）指标作为递接规则。我们展示，基于QE的递接策略能够让级联系统在少量实例（30%到50%）上调用更大模型，从而显著降低计算成本，同时保持与较大模型相当的性能。我们通过自动评估和人工评估验证了这种方法的有效性。 

---
# Fast Data Aware Neural Architecture Search via Supernet Accelerated Evaluation 

**Title (ZH)**: 通过超网络加速评估的快速数据感知神经架构搜索 

**Authors**: Emil Njor, Colby Banbury, Xenofon Fafoutis  

**Link**: [PDF](https://arxiv.org/pdf/2502.12690)  

**Abstract**: Tiny machine learning (TinyML) promises to revolutionize fields such as healthcare, environmental monitoring, and industrial maintenance by running machine learning models on low-power embedded systems. However, the complex optimizations required for successful TinyML deployment continue to impede its widespread adoption. A promising route to simplifying TinyML is through automatic machine learning (AutoML), which can distill elaborate optimization workflows into accessible key decisions. Notably, Hardware Aware Neural Architecture Searches - where a computer searches for an optimal TinyML model based on predictive performance and hardware metrics - have gained significant traction, producing some of today's most widely used TinyML models. Nevertheless, limiting optimization solely to neural network architectures can prove insufficient. Because TinyML systems must operate under extremely tight resource constraints, the choice of input data configuration, such as resolution or sampling rate, also profoundly impacts overall system efficiency. Achieving truly optimal TinyML systems thus requires jointly tuning both input data and model architecture. Despite its importance, this "Data Aware Neural Architecture Search" remains underexplored. To address this gap, we propose a new state-of-the-art Data Aware Neural Architecture Search technique and demonstrate its effectiveness on the novel TinyML ``Wake Vision'' dataset. Our experiments show that across varying time and hardware constraints, Data Aware Neural Architecture Search consistently discovers superior TinyML systems compared to purely architecture-focused methods, underscoring the critical role of data-aware optimization in advancing TinyML. 

**Abstract (ZH)**: 小机器学习（TinyML）有望通过在低功耗嵌入式系统上运行机器学习模型，从而彻底改变医疗保健、环境监测和工业维护等领域。然而，成功的TinyML部署所需的复杂优化持续阻碍其广泛应用。通过自动机器学习（AutoML）简化TinyML是颇有前景的途径，它可以将复杂的优化工作流程简化为易于理解和决策的关键步骤。值得注意的是，硬件感知神经架构搜索（Hardware Aware Neural Architecture Search，HANAS），即基于预测性能和硬件指标自动寻找最优TinyML模型的技术，已经取得了显著进展，产生了当今广泛使用的许多TinyML模型。然而，仅将优化限制在神经网络架构上可能不够。由于TinyML系统必须在极其严格的资源限制下运行，输入数据配置的选择，如分辨率或采样率，也极大地影响了系统的整体效率。因此，构建真正最优的TinyML系统需要同时调整输入数据和模型架构。尽管如此，“数据感知神经架构搜索”这一领域仍然较少被探索。为填补这一空白，我们提出了一种最新的数据感知神经架构搜索技术，并在新的TinyML“唤醒视图”数据集上展示了其效果。实验结果显示，无论是在不同的时间约束还是硬件条件下，数据感知神经架构搜索都持续发现优于单纯基于架构优化方法的TinyML系统，强调了数据感知优化在推进TinyML发展中的关键作用。 

---
# Multi-Step Alignment as Markov Games: An Optimistic Online Gradient Descent Approach with Convergence Guarantees 

**Title (ZH)**: 多步对齐作为马尔可夫游戏：一种具有收敛保证的乐观在线梯度下降方法 

**Authors**: Yongtao Wu, Luca Viano, Yihang Chen, Zhenyu Zhu, Kimon Antonakopoulos, Quanquan Gu, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2502.12678)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has been highly successful in aligning large language models with human preferences. While prevalent methods like DPO have demonstrated strong performance, they frame interactions with the language model as a bandit problem, which limits their applicability in real-world scenarios where multi-turn conversations are common. Additionally, DPO relies on the Bradley-Terry model assumption, which does not adequately capture the non-transitive nature of human preferences. In this paper, we address these challenges by modeling the alignment problem as a two-player constant-sum Markov game, where each player seeks to maximize their winning rate against the other across all steps of the conversation. Our approach Multi-step Preference Optimization (MPO) is built upon the natural actor-critic framework~\citep{peters2008natural}. We further develop OMPO based on the optimistic online gradient descent algorithm~\citep{rakhlin2013online,joulani17a}. Theoretically, we provide a rigorous analysis for both algorithms on convergence and show that OMPO requires $\mathcal{O}(\epsilon^{-1})$ policy updates to converge to an $\epsilon$-approximate Nash equilibrium. We also validate the effectiveness of our method on multi-turn conversations dataset and math reasoning dataset. 

**Abstract (ZH)**: 人类反馈强化学习（RLHF）在对齐大规模语言模型与人类偏好方面取得了高度成功。尽管诸如DPO等常用方法展示了强大的性能，但它们将与语言模型的交互视为一种多臂赌博机问题，这限制了其在多轮对话常见的真实场景中的应用。此外，DPO依赖于Bradley-Terry模型假设，但这并未充分捕捉人类偏好的非传递性。本文通过将对齐问题建模为两人常和马尔可夫博弈来应对这些挑战，其中每个玩家致力于在整个对话步骤中最大化其对抗另一方的胜率。我们基于自然演员-评论家框架构建了多步偏好优化（MPO）的方法，并在此基础上进一步开发了基于乐观在线梯度下降算法的OMPO。理论上，我们对这两种算法的收敛性进行了严格的分析，并证明OMPO需要$\mathcal{O}(\epsilon^{-1})$次策略更新即可收敛至$\epsilon$-近似纳什均衡。我们还在多轮对话数据集和数学推理数据集上验证了我们方法的有效性。 

---
# Spiking Vision Transformer with Saccadic Attention 

**Title (ZH)**: 基于扫视注意的脉冲视觉变换器 

**Authors**: Shuai Wang, Malu Zhang, Dehao Zhang, Ammar Belatreche, Yichen Xiao, Yu Liang, Yimeng Shan, Qian Sun, Enqi Zhang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12677)  

**Abstract**: The combination of Spiking Neural Networks (SNNs) and Vision Transformers (ViTs) holds potential for achieving both energy efficiency and high performance, particularly suitable for edge vision applications. However, a significant performance gap still exists between SNN-based ViTs and their ANN counterparts. Here, we first analyze why SNN-based ViTs suffer from limited performance and identify a mismatch between the vanilla self-attention mechanism and spatio-temporal spike trains. This mismatch results in degraded spatial relevance and limited temporal interactions. To address these issues, we draw inspiration from biological saccadic attention mechanisms and introduce an innovative Saccadic Spike Self-Attention (SSSA) method. Specifically, in the spatial domain, SSSA employs a novel spike distribution-based method to effectively assess the relevance between Query and Key pairs in SNN-based ViTs. Temporally, SSSA employs a saccadic interaction module that dynamically focuses on selected visual areas at each timestep and significantly enhances whole scene understanding through temporal interactions. Building on the SSSA mechanism, we develop a SNN-based Vision Transformer (SNN-ViT). Extensive experiments across various visual tasks demonstrate that SNN-ViT achieves state-of-the-art performance with linear computational complexity. The effectiveness and efficiency of the SNN-ViT highlight its potential for power-critical edge vision applications. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，符合学术规范：

基于脉冲神经网络（SNNs）和视觉变换器（ViTs）的结合有望同时实现能效和高性能，特别适合边缘视觉应用。然而，基于SNN的ViTs与基于ANN的ViTs之间仍然存在显著的性能差距。在这里，我们首先分析为什么基于SNN的ViTs表现受限，并确定了标准自注意力机制与时空脉冲序列之间的不匹配。这种不匹配导致空间相关性下降和时间交互能力有限。为了解决这些问题，我们受到生物学眼动注意力机制的启发，引入了一种创新的脉冲自注意力（SSSA）方法。具体地，在空间域中，SSSA采用了一种新颖的基于脉冲分布的方法，有效评估了基于SNN的ViTs中Query和Key对之间的相关性。在时间域中，SSSA使用了一种眼动交互模块，在每个时间步中动态聚焦于选定的视觉区域，并通过时间交互大大增强了对整个场景的理解。在此基础上，我们开发了一种基于SNN的视觉变换器（SNN-ViT）。在各种视觉任务的广泛实验中，展示了SNN-ViT实现了最先进的性能，并具有线性计算复杂度。SNN-ViT的有效性和效率凸显了其在关键功率边缘视觉应用中的潜力。 

---
# Speech-FT: A Fine-tuning Strategy for Enhancing Speech Representation Models Without Compromising Generalization Ability 

**Title (ZH)**: Speech-FT：一种在不牺牲泛化能力的前提下提升语音表示模型的方法 

**Authors**: Tzu-Quan Lin, Wei-Ping Huang, Hao Tang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.12672)  

**Abstract**: Speech representation models are highly effective at extracting general features for various tasks. While fine-tuning can enhance these representations for specific applications, it often compromises their generalization ability. To address this challenge, we propose Speech-FT, a fine-tuning strategy for speech representation models that leverages model merging to preserve generalization ability while still benefiting from fine-tuning. Speech-FT is effective across different fine-tuning scenarios and is compatible with various types of speech representation models, providing a versatile solution. Speech-FT offers an efficient and practical approach to further improving general speech representations after pre-training. 

**Abstract (ZH)**: 语音表示模型在多种任务中高度有效地提取通用特征。虽然微调可以增强这些表示以适应特定应用，但它通常会削弱它们的泛化能力。为应对这一挑战，我们提出了一种名为Speech-FT的微调策略，该策略利用模型合并来保持泛化能力，同时仍从微调中获益。Speech-FT在不同的微调场景下都有效，并且与各种类型的语音表示模型兼容，提供了一个通用的解决方案。Speech-FT提供了一种高效且实用的方法，在预训练后进一步提升通用语音表示。 

---
# The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1 

**Title (ZH)**: 大型推理模型潜藏的风险：R1的安全性评估 

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12659)  

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap. 

**Abstract (ZH)**: 大型推理模型的快速进步，如OpenAI-o3和DeepSeek-R1，显著提高了复杂推理的能力，超越了非推理大型语言模型（LLMs）。然而，这些增强的能力与模型如DeepSeek-R1的开源访问结合，引发了严重的安全担忧，特别是在滥用风险方面。本文通过对这些推理模型进行全面的安全评估，利用现有的安全基准来检验其合规性。此外，我们还探讨了它们对对抗攻击（如逃逸和提示注入）的脆弱性，以评估其在实际应用中的稳健性。通过多维度分析，我们发现了四项关键发现：（1）开源R1模型在安全基准和攻击测试中与o3-mini模型存在显著的安全差距，表明需要对R1模型投入更多安全努力。（2）精简推理模型的安全表现较差，与安全对齐的基础模型相比。（3）模型的推理能力越强，回答不安全问题时造成的潜在危害越大。（4）R1模型的推理过程比最终答案引发更大的安全担忧。我们的研究为推理模型的安全性含义提供了见解，并强调了进一步提高R1模型安全性的重要性，以缩小差距。 

---
# \textit{One Size doesn't Fit All}: A Personalized Conversational Tutoring Agent for Mathematics Instruction 

**Title (ZH)**: 《一招鲜不适用所有情况》：一种个性化对话式辅导代理用于数学教学 

**Authors**: Ben Liu, Jihan Zhang, Fangquan Lin, Xu Jia, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12633)  

**Abstract**: Large language models (LLMs) have been increasingly employed in various intelligent educational systems, simulating human tutors to facilitate effective human-machine interaction. However, previous studies often overlook the significance of recognizing and adapting to individual learner characteristics. Such adaptation is crucial for enhancing student engagement and learning efficiency, particularly in mathematics instruction, where diverse learning styles require personalized strategies to promote comprehension and enthusiasm. In this paper, we propose a \textbf{P}erson\textbf{A}lized \textbf{C}onversational tutoring ag\textbf{E}nt (PACE) for mathematics instruction. PACE simulates students' learning styles based on the Felder and Silverman learning style model, aligning with each student's persona. In this way, our PACE can effectively assess the personality of students, allowing to develop individualized teaching strategies that resonate with their unique learning styles. To further enhance students' comprehension, PACE employs the Socratic teaching method to provide instant feedback and encourage deep thinking. By constructing personalized teaching data and training models, PACE demonstrates the ability to identify and adapt to the unique needs of each student, significantly improving the overall learning experience and outcomes. Moreover, we establish multi-aspect evaluation criteria and conduct extensive analysis to assess the performance of personalized teaching. Experimental results demonstrate the superiority of our model in personalizing the educational experience and motivating students compared to existing methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种智能教育系统中得到了越来越多的应用，模拟人类导师以促进有效的人机交互。然而，以往的研究往往忽视了识别和适应个体学习者特征的重要性。这种适应性对于提高学生参与度和学习效率至关重要，特别是在数学教学中，不同的学习风格需要个性化策略来促进理解和激发兴趣。本文提出了一种针对数学教学的个性化对话式辅导代理（PACE）。PACE基于Felder-Silverman学习风格模型模拟学生的学习风格，并与每个学生的人格特点相匹配。通过这种方式，我们的PACE可以有效地评估学生的人格特质，从而开发出能够与他们独特学习风格相呼应的个性化教学策略。为了进一步增强学生对知识点的理解，PACE采用苏格拉底教学法提供即时反馈并鼓励深入思考。通过构建个性化的教学数据并训练模型，PACE展示了识别和适应每位学生独特需求的能力，显著提高了整体学习体验和成果。此外，我们建立了多方面的评估标准并进行了广泛分析，以评估个性化教学的效果。实验结果表明，与现有方法相比，我们的模型在个性化教育体验和激发学生兴趣方面具有明显优势。 

---
# Score-Based Diffusion Policy Compatible with Reinforcement Learning via Optimal Transport 

**Title (ZH)**: 基于分数扩散策略与最优传输相结合的强化学习兼容方法 

**Authors**: Mingyang Sun, Pengxiang Ding, Weinan Zhang, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12631)  

**Abstract**: Diffusion policies have shown promise in learning complex behaviors from demonstrations, particularly for tasks requiring precise control and long-term planning. However, they face challenges in robustness when encountering distribution shifts. This paper explores improving diffusion-based imitation learning models through online interactions with the environment. We propose OTPR (Optimal Transport-guided score-based diffusion Policy for Reinforcement learning fine-tuning), a novel method that integrates diffusion policies with RL using optimal transport theory. OTPR leverages the Q-function as a transport cost and views the policy as an optimal transport map, enabling efficient and stable fine-tuning. Moreover, we introduce masked optimal transport to guide state-action matching using expert keypoints and a compatibility-based resampling strategy to enhance training stability. Experiments on three simulation tasks demonstrate OTPR's superior performance and robustness compared to existing methods, especially in complex and sparse-reward environments. In sum, OTPR provides an effective framework for combining IL and RL, achieving versatile and reliable policy learning. The code will be released at this https URL. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文版：

扩散政策在从演示中学习复杂行为方面展现了潜力，特别是在需要精确控制和长期规划的任务中。然而，它们在遇到分布迁移时的鲁棒性存在挑战。本文通过在线与环境交互来改进基于扩散的模仿学习模型。我们提出了一种名为OTPR（Optimal Transport-guided score-based diffusion Policy for Reinforcement Learning fine-tuning）的新方法，该方法结合了扩散政策和强化学习中的最优传输理论。OTPR利用Q函数作为运输成本，并将策略视为最优传输映射，从而实现高效且稳定的调优。此外，我们引入了掩码最优传输方法以使用专家关键点引导状态-动作匹配，并采用基于兼容性的重采样策略以增强训练稳定性。在三个模拟任务上的实验结果显示，与现有方法相比，OTPR在复杂和稀疏奖励环境中表现出更优的性能和更强的鲁棒性。总之，OTPR提供了一种有效的框架，用于结合模仿学习（IL）和强化学习（RL），实现了通用且可靠的策略学习。代码将发布在以下链接：[https://www.alibabacloud.com]。 

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
# DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning 

**Title (ZH)**: DeepResonance：基于音乐中心多向指令调优的多模态音乐理解增强 

**Authors**: Zhuoyuan Mao, Mengjie Zhao, Qiyu Wu, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2502.12623)  

**Abstract**: Recent advancements in music large language models (LLMs) have significantly improved music understanding tasks, which involve the model's ability to analyze and interpret various musical elements. These improvements primarily focused on integrating both music and text inputs. However, the potential of incorporating additional modalities such as images, videos and textual music features to enhance music understanding remains unexplored. To bridge this gap, we propose DeepResonance, a multimodal music understanding LLM fine-tuned via multi-way instruction tuning with multi-way aligned music, text, image, and video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and Music4way-Any2T, three 4-way training and evaluation datasets designed to enable DeepResonance to integrate both visual and textual music feature content. We also introduce multi-sampled ImageBind embeddings and a pre-alignment Transformer to enhance modality fusion prior to input into text LLMs, tailoring DeepResonance for multi-way instruction tuning. Our model achieves state-of-the-art performances across six music understanding tasks, highlighting the benefits of the auxiliary modalities and the structural superiority of DeepResonance. We plan to open-source the models and the newly constructed datasets. 

**Abstract (ZH)**: 近年来，在音乐大规模语言模型（LLMs）方面的最新进展显著提高了音乐理解任务的能力，这些任务涉及模型分析和解释各种音乐元素的能力。这些改进主要集中在整合音乐和文本输入。然而，将图像、视频以及音乐文本特征等其他模态纳入以增强音乐理解的潜力尚未被充分探索。为了填补这一空白，我们提出了一种名为DeepResonance的多模态音乐理解LLM，该模型是通过多视角指令调优并结合多视角对齐的音乐、文本、图像和视频数据进行微调的。为此，我们构建了Music4way-MI2T、Music4way-MV2T和Music4way-Any2T三个四视角训练和评估数据集，旨在使DeepResonance能够结合视觉和文本音乐特征内容。此外，我们引入了多样本的ImageBind嵌入和预对齐的Transformer，以增强在输入文本LLM之前的各种模态的融合，使DeepResonance适用于多视角指令调优。我们的模型在六项音乐理解任务中均达到了最先进的性能，突显了辅助模态的优势以及DeepResonance的结构优势。我们计划开源模型和新构建的数据集。 

---
# A Graph-Enhanced Deep-Reinforcement Learning Framework for the Aircraft Landing Problem 

**Title (ZH)**: 一种增强图表示的深度强化学习框架用于飞机降落问题 

**Authors**: Vatsal Maru  

**Link**: [PDF](https://arxiv.org/pdf/2502.12617)  

**Abstract**: The Aircraft Landing Problem (ALP) is one of the challenging problems in aircraft transportation and management. The challenge is to schedule the arriving aircraft in a sequence so that the cost and delays are optimized. There are various solution approaches to solving this problem, most of which are based on operations research algorithms and meta-heuristics. Although traditional methods perform better on one or the other factors, there remains a problem of solving real-time rescheduling and computational scalability altogether. This paper presents a novel deep reinforcement learning (DRL) framework that combines graph neural networks with actor-critic architectures to address the ALP. This paper introduces three key contributions: A graph-based state representation that efficiently captures temporal and spatial relationships between aircraft, a specialized actor-critic architecture designed to handle multiple competing objectives in landing scheduling, and a runway balance strategy that ensures efficient resource utilization while maintaining safety constraints. The results show that the trained algorithm can be tested on different problem sets and the results are competitive to operation research algorithms. The experimental results on standard benchmark data sets demonstrate a 99.95 reduction in computational time compared to Mixed Integer Programming (MIP) and 38 higher runway throughput over First Come First Serve (FCFS) approaches. Therefore, the proposed solution is competitive to traditional approaches and achieves substantial advancements. Notably, it does not require retraining, making it particularly suitable for industrial deployment. The frameworks capability to generate solutions within 1 second enables real-time rescheduling, addressing critical requirements of air traffic management. 

**Abstract (ZH)**: 飞机着陆问题（ALP）是航空运输与管理领域中的一个具有挑战性的问题。挑战在于为着陆的飞机制定一个序列，以优化成本和延误。解决这一问题的方法多种多样，大多数基于运筹学算法和元启发式方法。虽然传统方法在某些因素上表现更佳，但一直存在实时重新调度和计算效率之间的矛盾问题。本文提出了一种新颖的深度强化学习（DRL）框架，将图神经网络与演员-评论家架构结合，以解决飞机着陆问题（ALP）。本文提出了三个主要贡献：一种基于图的状态表示方法，能高效捕捉飞机之间的时空关系；一种专为处理着陆调度中多个竞争性目标设计的演员-评论家架构；以及一种跑道平衡策略，确保高效的资源利用同时满足安全约束。实验结果表明，训练后的算法可以在不同的问题集上进行测试，其结果与运筹学算法具有竞争力。标准基准数据集的实验结果表明，与混合整数规划（MIP）相比，计算时间减少了99.95%；与先来后服务（FCFS）方法相比，跑道通过量提高了38%。因此，所提出的方法在传统方法面前具有竞争力，并取得了显著的进步。值得注意的是，该方法不需要重新训练，特别适合工业部署。该框架能在1秒内生成解决方案的能力，满足空中交通管理的实时重新调度需求。 

---
# Label Drop for Multi-Aspect Relation Modeling in Universal Information Extraction 

**Title (ZH)**: 面向通用信息提取的多方面关系建模中的标签-drop方法 

**Authors**: Lu Yang, Jiajia Li, En Ci, Lefei Zhang, Zuchao Li, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12614)  

**Abstract**: Universal Information Extraction (UIE) has garnered significant attention due to its ability to address model explosion problems effectively. Extractive UIE can achieve strong performance using a relatively small model, making it widely adopted. Extractive UIEs generally rely on task instructions for different tasks, including single-target instructions and multiple-target instructions. Single-target instruction UIE enables the extraction of only one type of relation at a time, limiting its ability to model correlations between relations and thus restricting its capability to extract complex relations. While multiple-target instruction UIE allows for the extraction of multiple relations simultaneously, the inclusion of irrelevant relations introduces decision complexity and impacts extraction accuracy. Therefore, for multi-relation extraction, we propose LDNet, which incorporates multi-aspect relation modeling and a label drop mechanism. By assigning different relations to different levels for understanding and decision-making, we reduce decision confusion. Additionally, the label drop mechanism effectively mitigates the impact of irrelevant relations. Experiments show that LDNet outperforms or achieves competitive performance with state-of-the-art systems on 9 tasks, 33 datasets, in both single-modal and multi-modal, few-shot and zero-shot settings.\footnote{this https URL} 

**Abstract (ZH)**: 通用信息提取（UIE）由于其有效解决模型爆炸问题的能力而引起了广泛关注。抽取式UIE可以通过相对较小的模型实现强大的性能，因此被广泛采用。抽取式UIE通常依赖于不同的任务指令，包括单目标指令和多目标指令。单目标指令UIE一次只能提取一种关系类型，这限制了其建模不同关系间关联的能力，从而限制了其提取复杂关系的能力。而多目标指令UIE可以同时提取多种关系，但包容无关关系会增加决策复杂性并影响提取准确性。因此，针对多关系提取，我们提出了一种LDNet模型，该模型结合了多方面关系建模和标签丢弃机制。通过将不同关系分配到不同的理解和决策级别，我们减少了决策中的混淆。此外，标签丢弃机制有效地减轻了无关关系的影响。实验结果显示，在9个任务、33个数据集上，LDNet在单模态和多模态、少量样本和零样本学习设置中均表现出色或达到与最新系统相当的性能水平。（脚注：此链接包含更多详细信息 https://... ） 

---
# Unveiling Mode Connectivity in Graph Neural Networks 

**Title (ZH)**: 揭示图神经网络中的模式连通性 

**Authors**: Bingheng Li, Zhikai Chen, Haoyu Han, Shenglai Zeng, Jingzhe Liu, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12608)  

**Abstract**: A fundamental challenge in understanding graph neural networks (GNNs) lies in characterizing their optimization dynamics and loss landscape geometry, critical for improving interpretability and robustness. While mode connectivity, a lens for analyzing geometric properties of loss landscapes has proven insightful for other deep learning architectures, its implications for GNNs remain unexplored. This work presents the first investigation of mode connectivity in GNNs. We uncover that GNNs exhibit distinct non-linear mode connectivity, diverging from patterns observed in fully-connected networks or CNNs. Crucially, we demonstrate that graph structure, rather than model architecture, dominates this behavior, with graph properties like homophily correlating with mode connectivity patterns. We further establish a link between mode connectivity and generalization, proposing a generalization bound based on loss barriers and revealing its utility as a diagnostic tool. Our findings further bridge theoretical insights with practical implications: they rationalize domain alignment strategies in graph learning and provide a foundation for refining GNN training paradigms. 

**Abstract (ZH)**: 理解和分析图神经网络（GNNs）的基本挑战在于对其优化动态和损失景观几何结构的特征化，这对于提高可解释性和鲁棒性至关重要。尽管模式连接性（mode connectivity）作为一种分析损失景观几何属性的视角已被证明对其他深度学习架构具有启发作用，但对于GNNs而言，其影响尚未被探索。本研究首次对GNN中的模式连接性进行了调查。我们发现，GNNs表现出不同于全连接网络或CNNs的非线性模式连接性。关键的是，我们证明图结构而非模型架构主导了这种行为，且图属性如同质性与模式连接性模式密切相关。我们进一步建立了模式连接性和泛化的联系，提出了基于损失障碍的一般化界，并揭示了其作为诊断工具的应用价值。我们的发现进一步弥合了理论见解与实际应用之间的差距：它们为图学习中的领域对齐策略提供了理论依据，并为改进GNN训练范式提供了基础。 

---
# Disentangling Long-Short Term State Under Unknown Interventions for Online Time Series Forecasting 

**Title (ZH)**: 在未知干预下的长短期状态解耦在线时间序列预测 

**Authors**: Ruichu Cai, Haiqin Huang, Zhifang Jiang, Zijian Li, Changze Zhou, Yuequn Liu, Yuming Liu, Zhifeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12603)  

**Abstract**: Current methods for time series forecasting struggle in the online scenario, since it is difficult to preserve long-term dependency while adapting short-term changes when data are arriving sequentially. Although some recent methods solve this problem by controlling the updates of latent states, they cannot disentangle the long/short-term states, leading to the inability to effectively adapt to nonstationary. To tackle this challenge, we propose a general framework to disentangle long/short-term states for online time series forecasting. Our idea is inspired by the observations where short-term changes can be led by unknown interventions like abrupt policies in the stock market. Based on this insight, we formalize a data generation process with unknown interventions on short-term states. Under mild assumptions, we further leverage the independence of short-term states led by unknown interventions to establish the identification theory to achieve the disentanglement of long/short-term states. Built on this theory, we develop a long short-term disentanglement model (LSTD) to extract the long/short-term states with long/short-term encoders, respectively. Furthermore, the LSTD model incorporates a smooth constraint to preserve the long-term dependencies and an interrupted dependency constraint to enforce the forgetting of short-term dependencies, together boosting the disentanglement of long/short-term states. Experimental results on several benchmark datasets show that our \textbf{LSTD} model outperforms existing methods for online time series forecasting, validating its efficacy in real-world applications. 

**Abstract (ZH)**: 目前用于时间序列预测的方法在在线场景中遇到困难，因为在顺序到达数据的情况下，难以同时保持长期依赖性并适应短期变化。尽管一些最新方法通过控制潜在状态的更新来解决这个问题，但它们无法分离长期和短期状态，从而导致无法有效适应非平稳性。为应对这一挑战，我们提出了一种通用框架，以分离在线时间序列预测中的长期和短期状态。我们的这一想法源于以下观察：短期变化可能由未知干预（例如股市中的突然政策变化）驱动。基于这一洞见，我们形式化了一个包含未知干预的短期状态生成过程。在轻微假设下，我们进一步利用由未知干预驱动的短期状态之间的独立性，来建立识别理论，以实现长期和短期状态的分离。在此理论的基础上，我们开发了一种长短期状态分离模型（LSTD），分别使用长短期编码器来提取长期和短期状态。此外，LSTD模型结合了平滑约束以保持长期依赖性，并引入了断续依赖性约束以强制遗忘短期依赖性，从而共同促进了长期和短期状态的分离。在多个基准数据集上的实验结果显示，我们的LSTD模型在在线时间序列预测中优于现有方法，证明了其在实际应用中的有效性。 

---
# RSMLP: A light Sampled MLP Structure for Incomplete Utterance Rewrite 

**Title (ZH)**: RSMLP：一种用于不完整话语重写的轻量级采样MLP结构 

**Authors**: Lunjun Liu, Weilai Jiang, Yaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12587)  

**Abstract**: The Incomplete Utterance Rewriting (IUR) task has garnered significant attention in recent years. Its goal is to reconstruct conversational utterances to better align with the current context, thereby enhancing comprehension. In this paper, we introduce a novel and versatile lightweight method, Rewritten-Sampled MLP (RSMLP). By employing an MLP based architecture with a carefully designed down-sampling strategy, RSMLP effectively extracts latent semantic information between utterances and makes appropriate edits to restore incomplete utterances. Due to its simple yet efficient structure, our method achieves competitive performance on public IUR datasets and in real-world applications. 

**Abstract (ZH)**: 近年来，不完整语句重写（IUR）任务引起了广泛关注。其目标是重构会话语句，使其更好地与当前语境对齐，从而增强理解能力。本文提出了一种新颖且灵活的轻量级方法——重写采样多层感知机（RSMLP）。通过采用基于多层感知机（MLP）的架构以及精心设计的下采样策略，RSMLP有效地提取了语句之间的潜在语义信息，并进行适当编辑以恢复不完整语句。由于其简洁且高效的结构，该方法在公开展示的IUR数据集以及实际应用中均取得了竞争力的表现。 

---
# Enhancing Semi-supervised Learning with Noisy Zero-shot Pseudolabels 

**Title (ZH)**: 使用噪声零样本伪标签增强半监督学习 

**Authors**: Jichan Chung, Irene Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12584)  

**Abstract**: Semi-supervised learning (SSL) leverages limited labeled data alongside abundant unlabeled data to address labeling costs in machine learning. While recent foundation models enable zero-shot inference, attempts to integrate these capabilities into SSL through pseudo-labeling have shown mixed results due to unreliable zero-shot predictions. We present ZMT (Zero-Shot Multi-Task Learning), a framework that jointly optimizes zero-shot pseudo-labels and unsupervised representation learning objectives from contemporary SSL approaches. Our method introduces a multi-task learning-based mechanism that incorporates pseudo-labels while ensuring robustness to varying pseudo-label quality. Experiments across 8 datasets in vision, language, and audio domains demonstrate that ZMT reduces error by up to 56% compared to traditional SSL methods, with particularly compelling results when pseudo-labels are noisy and unreliable. ZMT represents a significant step toward making semi-supervised learning more effective and accessible in resource-constrained environments. 

**Abstract (ZH)**: 半监督学习（SSL）利用有限的标注数据和大量的未标注数据来应对机器学习中的标注成本问题。虽然近期的基础模型能够实现零样本推理，但尝试通过伪标签将这些能力集成到SSL中却因零样本预测的不可靠性而效果参差不齐。我们提出了一种名为ZMT（零样本多任务学习）的框架，该框架联合优化零样本伪标签和源自当代SSL方法的无监督表示学习目标。该方法引入了一种基于多任务学习的机制，能够在确保伪标签质量波动下的鲁棒性的同时整合伪标签。在涉及视觉、语言和音频领域的8个数据集上进行的实验表明，ZMT相较于传统SSL方法可将误差降低多达56%，特别是在伪标签噪声和不可靠的情况下效果更为显著。ZMT代表了在资源受限环境中使半监督学习更加有效和易于使用的一个重要进展。 

---
# The Majority Vote Paradigm Shift: When Popular Meets Optimal 

**Title (ZH)**: 多数投票范式的转变：流行与最优的结合 

**Authors**: Antonio Purificato, Maria Sofia Bucarelli, Anil Kumar Nelakanti, Andrea Bacciu, Fabrizio Silvestri, Amin Mantrach  

**Link**: [PDF](https://arxiv.org/pdf/2502.12581)  

**Abstract**: Reliably labelling data typically requires annotations from multiple human workers. However, humans are far from being perfect. Hence, it is a common practice to aggregate labels gathered from multiple annotators to make a more confident estimate of the true label. Among many aggregation methods, the simple and well known Majority Vote (MV) selects the class label polling the highest number of votes. However, despite its importance, the optimality of MV's label aggregation has not been extensively studied. We address this gap in our work by characterising the conditions under which MV achieves the theoretically optimal lower bound on label estimation error. Our results capture the tolerable limits on annotation noise under which MV can optimally recover labels for a given class distribution. This certificate of optimality provides a more principled approach to model selection for label aggregation as an alternative to otherwise inefficient practices that sometimes include higher experts, gold labels, etc., that are all marred by the same human uncertainty despite huge time and monetary costs. Experiments on both synthetic and real world data corroborate our theoretical findings. 

**Abstract (ZH)**: 可靠地标注数据通常需要多名人类工作者的注释。然而，人类远非完美。因此，通常的做法是聚合多名注释者收集的标签，以获得对真实标签更有信心的估计。在众多聚合方法中，简单且广为人知的多数投票（Majority Vote, MV）方法是通过选择得票最多的类别标签来实现的。尽管其重要性不言而喻，但MV标签聚合的最优性尚未得到广泛研究。我们在研究中通过分析MV在何种条件下能够达到标签估计误差的理论最优下界来填补这一空白。我们的结果给出了在给定类别分布情况下，MV在何种注释噪声水平下可以最优地恢复标签。这一最优性证书提供了一种更为严谨的方法来选择标签聚合模型，作为效率较低的其他做法的替代方案，如引入更高层次的专家、金标准标签等，尽管这些做法都难以避免人类不确定性的影响，尽管耗费了巨大的时间和金钱成本。我们在合成数据和真实数据上的实验结果支持了我们的理论发现。 

---
# A Fuzzy Evaluation of Sentence Encoders on Grooming Risk Classification 

**Title (ZH)**: 基于模糊评价的句子编码器在梳妆风险分类中的评估 

**Authors**: Geetanjali Bihani, Julia Rayz  

**Link**: [PDF](https://arxiv.org/pdf/2502.12576)  

**Abstract**: With the advent of social media, children are becoming increasingly vulnerable to the risk of grooming in online settings. Detecting grooming instances in an online conversation poses a significant challenge as the interactions are not necessarily sexually explicit, since the predators take time to build trust and a relationship with their victim. Moreover, predators evade detection using indirect and coded language. While previous studies have fine-tuned Transformers to automatically identify grooming in chat conversations, they overlook the impact of coded and indirect language on model predictions, and how these align with human perceptions of grooming. In this paper, we address this gap and evaluate bi-encoders on the task of classifying different degrees of grooming risk in chat contexts, for three different participant groups, i.e. law enforcement officers, real victims, and decoys. Using a fuzzy-theoretic framework, we map human assessments of grooming behaviors to estimate the actual degree of grooming risk. Our analysis reveals that fine-tuned models fail to tag instances where the predator uses indirect speech pathways and coded language to evade detection. Further, we find that such instances are characterized by a higher presence of out-of-vocabulary (OOV) words in samples, causing the model to misclassify. Our findings highlight the need for more robust models to identify coded language from noisy chat inputs in grooming contexts. 

**Abstract (ZH)**: 随着社交媒体的发展，儿童在在线环境中面临越来越多的诱骗风险。在在线对话中检测诱骗事件带来了很大的挑战，因为这些互动未必包含明显的性内容，因为诱骗者会花费时间来建立信任和与受害者的联系。此外，诱骗者会使用间接和隐晦的语言来逃避检测。尽管先前的研究已经针对对话聊天对 Transformer 模型进行了微调，以自动识别诱骗行为，但他们忽视了隐晦和间接语言对模型预测的影响，以及这些影响与人类对诱骗行为的认知之间的差异。在本文中，我们弥补了这一不足，并评估了双编码器在三个不同参与者群体（即执法官员、真实受害者和诱饵）在聊天环境中分类不同层次的诱骗风险任务中的表现。我们利用模糊理论框架将人类对诱骗行为的评估转化为估计实际的诱骗风险程度。我们的分析表明，微调模型无法准确标记那些诱骗者使用隐晦语言路径和隐晦语言来逃避检测的事件。此外，我们发现这些事件的特点是样本中离词汇表频率之外（OOV）词汇的出现频率较高，导致模型出现误分类。我们的研究结果强调了在诱骗环境中需要更 robust 的模型来从嘈杂的聊天输入中识别隐晦语言。 

---
# DemonAgent: Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent 

**Title (ZH)**: DemonAgent：基于大语言模型的代理程序中动态加密多后门植入攻击 

**Authors**: Pengyu Zhu, Zhenhong Zhou, Yuanhe Zhang, Shilinlu Yan, Kun Wang, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.12575)  

**Abstract**: As LLM-based agents become increasingly prevalent, backdoors can be implanted into agents through user queries or environment feedback, raising critical concerns regarding safety vulnerabilities. However, backdoor attacks are typically detectable by safety audits that analyze the reasoning process of agents. To this end, we propose a novel backdoor implantation strategy called \textbf{Dynamically Encrypted Multi-Backdoor Implantation Attack}. Specifically, we introduce dynamic encryption, which maps the backdoor into benign content, effectively circumventing safety audits. To enhance stealthiness, we further decompose the backdoor into multiple sub-backdoor fragments. Based on these advancements, backdoors are allowed to bypass safety audits significantly. Additionally, we present AgentBackdoorEval, a dataset designed for the comprehensive evaluation of agent backdoor attacks. Experimental results across multiple datasets demonstrate that our method achieves an attack success rate nearing 100\% while maintaining a detection rate of 0\%, illustrating its effectiveness in evading safety audits. Our findings highlight the limitations of existing safety mechanisms in detecting advanced attacks, underscoring the urgent need for more robust defenses against backdoor threats. Code and data are available at this https URL. 

**Abstract (ZH)**: 随着基于大规模语言模型（LLM）的代理变得越来越普遍，恶意后门可以通过用户查询或环境反馈植入到代理中，这引发了严重的安全性漏洞问题。然而，后门攻击通常可以通过安全性审查来检测，这些审查分析代理的推理过程。为了解决这一问题，我们提出了一种新的后门植入策略，称为**动态加密多后门植入攻击**。具体来说，我们引入了动态加密，将后门映射到无害内容中，从而有效规避了安全性审查。为了进一步提高隐蔽性，我们还将后门分解为多个子后门片段。通过这些进展，后门能够显著避开安全性审查。此外，我们还提出了一个名为**AgentBackdoorEval**的数据集，用于全面评估代理后门攻击。跨多个数据集的实验结果表明，我们的方法在攻击成功率接近100%的同时，保持了0%的检测率，这表明其在规避安全性审查方面的有效性。我们的研究结果揭示了现有安全性机制在检测高级攻击方面的局限性，突显了对更 robust 防御措施的迫切需求。相关代码和数据可通过以下链接获取：[请插入具体链接]。 

---
# HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading 

**Title (ZH)**: HeadInfer：按头卸载的内存高效大型语言模型推理方法 

**Authors**: Cheng Luo, Zefan Cai, Hanshi Sun, Jinqi Xiao, Bo Yuan, Wen Xiao, Junjie Hu, Jiawei Zhao, Beidi Chen, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12574)  

**Abstract**: Transformer-based large language models (LLMs) demonstrate impressive performance in long context generation. Extending the context length has disproportionately shifted the memory footprint of LLMs during inference to the key-value cache (KV cache). In this paper, we propose HEADINFER, which offloads the KV cache to CPU RAM while avoiding the need to fully store the KV cache for any transformer layer on the GPU. HEADINFER employs a fine-grained, head-wise offloading strategy, maintaining only selective attention heads KV cache on the GPU while computing attention output dynamically. Through roofline analysis, we demonstrate that HEADINFER maintains computational efficiency while significantly reducing memory footprint. We evaluate HEADINFER on the Llama-3-8B model with a 1-million-token sequence, reducing the GPU memory footprint of the KV cache from 128 GB to 1 GB and the total GPU memory usage from 207 GB to 17 GB, achieving a 92% reduction compared to BF16 baseline inference. Notably, HEADINFER enables 4-million-token inference with an 8B model on a single consumer GPU with 24GB memory (e.g., NVIDIA RTX 4090) without approximation methods. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）在长上下文生成方面表现出色。扩充上下文长度已经不正常地将LLMs在推理过程中的内存足迹主要转移到了键值缓存（KV缓存）中。本文中，我们提出了HEADINFER方法，该方法将KV缓存卸载到CPU RAM上，同时避免在任何Transformer层上完全存储KV缓存。HEADINFER采用细粒度、按头卸载的策略，在只在GPU上动态计算注意力输出的情况下，仅保留部分注意力头的KV缓存。通过roofline分析，我们展示了HEADINFER在保持计算效率的同时，显著减少了内存足迹。我们使用Llama-3-8B模型对100万 token的序列进行了评估，将KV缓存的GPU内存足迹从128 GB减少到1 GB，总GPU内存使用量从207 GB减少到17 GB，相比BF16基线推理实现了92%的减少。值得注意的是，HEADINFER可以在具有24 GB内存的单个消费级GPU（例如NVIDIA RTX 4090）上实现8B模型的400万token推理，而不需要使用近似方法。 

---
# A Cognitive Writing Perspective for Constrained Long-Form Text Generation 

**Title (ZH)**: 受限长文本生成的认知写作视角 

**Authors**: Kaiyang Wan, Honglin Mu, Rui Hao, Haoran Luo, Tianle Gu, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12568)  

**Abstract**: Like humans, Large Language Models (LLMs) struggle to generate high-quality long-form text that adheres to strict requirements in a single pass. This challenge is unsurprising, as successful human writing, according to the Cognitive Writing Theory, is a complex cognitive process involving iterative planning, translating, reviewing, and monitoring. Motivated by these cognitive principles, we aim to equip LLMs with human-like cognitive writing capabilities through CogWriter, a novel training-free framework that transforms LLM constrained long-form text generation into a systematic cognitive writing paradigm. Our framework consists of two key modules: (1) a Planning Agent that performs hierarchical planning to decompose the task, and (2) multiple Generation Agents that execute these plans in parallel. The system maintains quality via continuous monitoring and reviewing mechanisms, which evaluate outputs against specified requirements and trigger necessary revisions. CogWriter demonstrates exceptional performance on LongGenBench, a benchmark for complex constrained long-form text generation. Even when using Qwen-2.5-14B as its backbone, CogWriter surpasses GPT-4o by 22% in complex instruction completion accuracy while reliably generating texts exceeding 10,000 words. We hope this cognitive science-inspired approach provides a paradigm for LLM writing advancements: \href{this https URL}{CogWriter}. 

**Abstract (ZH)**: 像人类一样，大规模语言模型（LLMs）在一次生成高质量长文本时往往会遇到困难，尤其是在遵循严格要求方面。这一挑战并不令人惊讶，因为在认知写作理论中，成功的写作过程被认为是一个复杂的认知过程，涉及迭代的计划、翻译、审阅和监控。基于这些认知原理，我们旨在通过CogWriter这一全新的无需训练框架，为LLMs配备类似人类的认知写作能力，将受限的长文本生成转变为一种系统化的认知写作范式。我们的框架包括两个关键模块：（1）规划代理，执行分层规划以分解任务；（2）多个生成代理，在并行执行这些计划。系统通过持续的监控和审阅机制维持质量，这些机制将输出与指定要求进行对比，并触发必要的修订。CogWriter在LongGenBench上表现出色，LongGenBench是一个复杂受限长文本生成基准测试。即使使用Qwen-2.5-14B作为其基础模型，CogWriter在复杂指令完成准确性上仍比GPT-4o高出22%，并可靠地生成超过10,000字的文本。我们希望这种受认知科学启发的方法为LLMs写作进步提供一个范式：\href{this https URL}{CogWriter}。 

---
# Evaluating Language Models on Grooming Risk Estimation Using Fuzzy Theory 

**Title (ZH)**: 使用模糊理论评估语言模型在评估 grooming 风险方面的表现 

**Authors**: Geetanjali Bihani, Tatiana Ringenberg, Julia Rayz  

**Link**: [PDF](https://arxiv.org/pdf/2502.12563)  

**Abstract**: Encoding implicit language presents a challenge for language models, especially in high-risk domains where maintaining high precision is important. Automated detection of online child grooming is one such critical domain, where predators manipulate victims using a combination of explicit and implicit language to convey harmful intentions. While recent studies have shown the potential of Transformer language models like SBERT for preemptive grooming detection, they primarily depend on surface-level features and approximate real victim grooming processes using vigilante and law enforcement conversations. The question of whether these features and approximations are reasonable has not been addressed thus far. In this paper, we address this gap and study whether SBERT can effectively discern varying degrees of grooming risk inherent in conversations, and evaluate its results across different participant groups. Our analysis reveals that while fine-tuning aids language models in learning to assign grooming scores, they show high variance in predictions, especially for contexts containing higher degrees of grooming risk. These errors appear in cases that 1) utilize indirect speech pathways to manipulate victims and 2) lack sexually explicit content. This finding underscores the necessity for robust modeling of indirect speech acts by language models, particularly those employed by predators. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

隐含语言的编码对语言模型构成了挑战，特别是在高风险领域，维持高精度尤为重要。自动检测在线儿童诱饵行为就是其中一种关键领域，攻击者使用显式和隐含语言的组合来操纵受害者，表达有害意图。虽然近期研究显示了像SBERT这样的Transformer语言模型在预防性诱饵检测方面的潜力，但它们主要依赖于表面特征，并通过义警和执法部门的对话来近似真实的受害者诱饵过程。目前尚未解决的一个问题是，这些特征和近似是否合理。在这篇论文中，我们弥补了这一缺口，并研究SBERT能否有效区分对话中固有的不同程度的诱饵风险，并在不同参与者群体中评估其结果。我们的分析表明，虽然微调有助于语言模型学习分配诱饵评分，但在包含较高程度诱饵风险的背景下，它们的预测具有很高的波动性。这些误差出现在以下情况下：1）利用间接言说途径来操纵受害者；2）缺乏性隐含内容。这一发现突显了语言模型（尤其是被攻击者使用的模型）需进行稳健的间接言说建模的必要性。 

---
# MomentSeeker: A Comprehensive Benchmark and A Strong Baseline For Moment Retrieval Within Long Videos 

**Title (ZH)**: MomentSeeker：长视频中时刻检索的全面基准及强大基线 

**Authors**: Huaying Yuan, Jian Ni, Yueze Wang, Junjie Zhou, Zhengyang Liang, Zheng Liu, Zhao Cao, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12558)  

**Abstract**: Retrieval augmented generation (RAG) holds great promise in addressing challenges associated with long video understanding. These methods retrieve useful moments from long videos for their presented tasks, thereby enabling multimodal large language models (MLLMs) to generate high-quality answers in a cost-effective way. In this work, we present MomentSeeker, a comprehensive benchmark to evaluate retrieval models' performance in handling general long-video moment retrieval (LVMR) tasks. MomentSeeker offers three key advantages. First, it incorporates long videos of over 500 seconds on average, making it the first benchmark specialized for long-video moment retrieval. Second, it covers a wide range of task categories (including Moment Search, Caption Alignment, Image-conditioned Moment Search, and Video-conditioned Moment Search) and diverse application scenarios (e.g., sports, movies, cartoons, and ego), making it a comprehensive tool for assessing retrieval models' general LVMR performance. Additionally, the evaluation tasks are carefully curated through human annotation, ensuring the reliability of assessment. We further fine-tune an MLLM-based LVMR retriever on synthetic data, which demonstrates strong performance on our benchmark. We perform extensive experiments with various popular multimodal retrievers based on our benchmark, whose results highlight the challenges of LVMR and limitations for existing methods. Our created resources will be shared with community to advance future research in this field. 

**Abstract (ZH)**: 检索增强生成（RAG）在解决长视频理解相关挑战方面展现出巨大的潜力。这些方法从长视频中检索出有用的时间片段以完成他们的任务，从而使多模态大型语言模型（MLLMs）能够以经济高效的方式生成高质量的答案。在本文中，我们提出了MomentSeeker，这是一个全面的基准测试，用于评估检索模型在处理通用长视频片段检索（LVMR）任务方面的性能。MomentSeeker具有三个关键优势。首先，它包含的平均时长超过500秒的长视频，使其成为首个专门针对长视频片段检索的基准测试。其次，它涵盖了广泛的任务类别（包括片段检索、字幕对齐、图像条件下的片段检索和视频条件下的片段检索）和多样化的应用场景（例如体育、电影、卡通和第一人称视角），成为评估检索模型在长视频片段检索方面综合性能的工具。此外，通过人工注释对评估任务进行了精心筛选，确保评估的可靠性。我们进一步在合成数据上微调了一个基于MLLM的LVMR检索器，该检索器在我们的基准测试中表现出强劲的性能。我们在基准测试的基础上，使用各种流行的多模态检索器进行了广泛的实验，其结果突显了LVMR的挑战和现有方法的局限性。我们创建的资源将与社区分享，推动该领域未来研究的发展。 

---
# LLM Safety for Children 

**Title (ZH)**: 儿童使用的语言模型安全性 

**Authors**: Prasanjit Rath, Hari Shrawgi, Parag Agrawal, Sandipan Dandapat  

**Link**: [PDF](https://arxiv.org/pdf/2502.12552)  

**Abstract**: This paper analyzes the safety of Large Language Models (LLMs) in interactions with children below age of 18 years. Despite the transformative applications of LLMs in various aspects of children's lives such as education and therapy, there remains a significant gap in understanding and mitigating potential content harms specific to this demographic. The study acknowledges the diverse nature of children often overlooked by standard safety evaluations and proposes a comprehensive approach to evaluating LLM safety specifically for children. We list down potential risks that children may encounter when using LLM powered applications. Additionally we develop Child User Models that reflect the varied personalities and interests of children informed by literature in child care and psychology. These user models aim to bridge the existing gap in child safety literature across various fields. We utilize Child User Models to evaluate the safety of six state of the art LLMs. Our observations reveal significant safety gaps in LLMs particularly in categories harmful to children but not adults 

**Abstract (ZH)**: 本文分析了大型语言模型（LLMs）在与18岁以下儿童互动时的安全性。尽管LLMs在教育和治疗等儿童生活多个方面展现出变革性的应用，但对特定于这一年龄段的潜在内容危害的理解和缓解仍存在显著差距。本研究承认了儿童的多样性，这种多样性常常被标准的安全评估所忽视，并提出了一种全面的方法来评估LLMs特别针对儿童的安全性。我们列出了儿童在使用基于LLMs的应用时可能遇到的各种潜在风险。此外，我们开发了反映儿童不同个性和兴趣的儿童用户模型，这些模型受到了儿童护理和心理学文献的启发。这些用户模型旨在弥补跨多个领域的儿童安全文献中的现有空白。我们使用儿童用户模型评估了六种最先进的LLMs的安全性。我们的观察结果揭示了LLMs在对儿童具有危害性的类别中存在显著的安全漏洞。 

---
# Improving the Stability of GNN Force Field Models by Reducing Feature Correlation 

**Title (ZH)**: 通过降低特征相关性提高GNN力场模型的稳定性 

**Authors**: Yujie Zeng, Wenlong He, Ihor Vasyltsov, Jiaxin Wei, Ying Zhang, Lin Chen, Yuehua Dai  

**Link**: [PDF](https://arxiv.org/pdf/2502.12548)  

**Abstract**: Recently, Graph Neural Network based Force Field (GNNFF) models are widely used in Molecular Dynamics (MD) simulation, which is one of the most cost-effective means in semiconductor material research. However, even such models provide high accuracy in energy and force Mean Absolute Error (MAE) over trained (in-distribution) datasets, they often become unstable during long-time MD simulation when used for out-of-distribution datasets. In this paper, we propose a feature correlation based method for GNNFF models to enhance the stability of MD simulation. We reveal the negative relationship between feature correlation and the stability of GNNFF models, and design a loss function with a dynamic loss coefficient scheduler to reduce edge feature correlation that can be applied in general GNNFF training. We also propose an empirical metric to evaluate the stability in MD simulation. Experiments show our method can significantly improve stability for GNNFF models especially in out-of-distribution data with less than 3% computational overhead. For example, we can ensure the stable MD simulation time from 0.03ps to 10ps for Allegro model. 

**Abstract (ZH)**: 近年来，基于图神经网络的势场模型（GNNFF）在分子动力学（MD）模拟中得到了广泛应用，而分子动力学模拟是半导体材料研究中成本效益最高的手段之一。然而，即使这些模型在训练数据集（同分布数据集）上能够提供高的能隙和力的平均绝对误差（MAE）的准确性，当应用于未训练数据集（异分布数据集）时，其在长时间MD模拟中往往变得不稳定。本文提出了一种基于特征相关性的方法，以增强GNNFF模型在MD模拟中的稳定性。我们揭示了特征相关性与GNNFF模型稳定性之间的负相关关系，并设计了一个带有动态损失系数调度器的损失函数，以减少边缘特征相关性，该方法适用于一般GNNFF训练。我们还提出了一种经验指标来评估MD模拟中的稳定性。实验表明，我们的方法能够显著改善GNNFF模型在未训练数据集上的稳定性，同时计算开销不到3%。例如，我们能够确保Allegro模型在0.03皮秒到10皮秒范围内的稳定MD模拟时间。 

---
# Computing Voting Rules with Improvement Feedback 

**Title (ZH)**: 计算具有改进反馈的投票规则 

**Authors**: Evi Micha, Vasilis Varsamis  

**Link**: [PDF](https://arxiv.org/pdf/2502.12542)  

**Abstract**: Aggregating preferences under incomplete or constrained feedback is a fundamental problem in social choice and related domains. While prior work has established strong impossibility results for pairwise comparisons, this paper extends the inquiry to improvement feedback, where voters express incremental adjustments rather than complete preferences. We provide a complete characterization of the positional scoring rules that can be computed given improvement feedback. Interestingly, while plurality is learnable under improvement feedback--unlike with pairwise feedback--strong impossibility results persist for many other positional scoring rules. Furthermore, we show that improvement feedback, unlike pairwise feedback, does not suffice for the computation of any Condorcet-consistent rule. We complement our theoretical findings with experimental results, providing further insights into the practical implications of improvement feedback for preference aggregation. 

**Abstract (ZH)**: 在不完全或受限反馈条件下汇总偏好是社会选择及相关领域中的一个基本问题。尽管先前的工作已经确立了关于成对比较的强烈不可能性结果，本文将研究范围扩展至改进反馈，即选民表达逐步调整而非完整偏好。我们提供了在给定改进反馈条件下可以计算的职位评分规则的完整分类。有趣的是，虽然在改进反馈下可以学习 plurality 规则——而在成对反馈下却不行——许多其他职位评分规则仍然存在着强烈的不可能性结果。此外，我们证明了改进反馈不足以用于计算任何 Condorcet 一致规则，这与成对反馈的情况不同。我们通过实验结果补充了理论发现，进一步探讨了改进反馈在偏好汇总中的实用影响。 

---
# Finding Optimal Trading History in Reinforcement Learning for Stock Market Trading 

**Title (ZH)**: 在股票市场交易中利用强化学习寻找最优交易历史 

**Authors**: Sina Montazeria, Haseebullah Jumakhanb, Amir Mirzaeinia  

**Link**: [PDF](https://arxiv.org/pdf/2502.12537)  

**Abstract**: This paper investigates the optimization of temporal windows in Financial Deep Reinforcement Learning (DRL) models using 2D Convolutional Neural Networks (CNNs). We introduce a novel approach to treating the temporal field as a hyperparameter and examine its impact on model performance across various datasets and feature arrangements. We introduce a new hyperparameter for the CNN policy, proposing that this temporal field can and should be treated as a hyperparameter for these models. We examine the significance of this temporal field by iteratively expanding the window of observations presented to the CNN policy during the deep reinforcement learning process. Our iterative process involves progressively increasing the observation period from two weeks to twelve weeks, allowing us to examine the effects of different temporal windows on the model's performance. This window expansion is implemented in two settings. In one setting, we rearrange the features in the dataset to group them by company, allowing the model to have a full view of company data in its observation window and CNN kernel. In the second setting, we do not group the features by company, and features are arranged by category. Our study reveals that shorter temporal windows are most effective when no feature rearrangement to group per company is in effect. However, the model will utilize longer temporal windows and yield better performance once we introduce the feature rearrangement. To examine the consistency of our findings, we repeated our experiment on two datasets containing the same thirty companies from the Dow Jones Index but with different features in each dataset and consistently observed the above-mentioned patterns. The result is a trading model significantly outperforming global financial services firms such as the Global X Guru by the established Mirae Asset. 

**Abstract (ZH)**: 本文研究了在金融深度强化学习（DRL）模型中使用二维卷积神经网络（CNNs）优化时间窗口的问题。我们提出了一种新颖的方法，将时间领域视为超参数，并考察了其在不同数据集和特征排列下的模型性能影响。我们为CNN策略引入了一个新的超参数，建议将此时间领域视为这些模型的超参数。通过迭代增加呈现给CNN策略的观察窗口期，我们研究了时间领域的意义。该迭代过程从两周逐步增加到十二周，允许我们考察不同时间窗口对模型性能的影响。时间窗口的扩展在两种设置中实现。在一种设置中，我们重新排列数据集中的特征，按公司分组，使模型在其观察窗口和CNN核中能够全面查看公司数据。在第二种设置中，我们不按公司分组特征，而是按类别排列特征。研究结果显示，在没有对特征进行按公司分组的重新排列时，较短的时间窗口效果最好。但引入特征重新排列后，模型会利用更长的时间窗口并得到更好的性能。为了验证我们的发现的一致性，我们在包含来自道琼斯指数的同一三十家公司的两个数据集上重复了实验，每个数据集的特征各不相同，结果一致地观察到了上述模式。最终，得到的交易模型显著优于如Mirae Asset等全球金融服务巨头，例如Global X Guru。 

---
# An Algorithm Board in Neural Decoding 

**Title (ZH)**: 神经解码中的算法板 

**Authors**: Jingyi Feng, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12536)  

**Abstract**: Understanding the mechanisms of neural encoding and decoding has always been a highly interesting research topic in fields such as neuroscience and cognitive intelligence. In prior studies, some researchers identified a symmetry in neural data decoded by unsupervised methods in motor scenarios and constructed a cognitive learning system based on this pattern (i.e., symmetry). Nevertheless, the distribution state of the data flow that significantly influences neural decoding positions still remains a mystery within the system, which further restricts the enhancement of the system's interpretability. Based on this, this paper mainly explores changes in the distribution state within the system from the machine learning and mathematical statistics perspectives. In the experiment, we assessed the correctness of this symmetry using various tools and indicators commonly utilized in mathematics and statistics. According to the experimental results, the normal distribution (or Gaussian distribution) plays a crucial role in the decoding of prediction positions within the system. Eventually, an algorithm board similar to the Galton board was built to serve as the mathematical foundation of the discovered symmetry. 

**Abstract (ZH)**: 理解神经编码和解码的机制一直是神经科学和认知智能等领域的一个高度有趣的研究课题。在先前的研究中，一些研究人员通过无监督方法在运动场景中识别出了神经数据解码中的某种对称性，并基于这一模式构建了一个认知学习系统（即对称性）。然而，系统中显著影响神经解码位置的数据流分布状态仍是一个谜，这进一步限制了系统可解释性的提升。基于此，本文主要从机器学习和数学统计的角度探讨系统中数据流分布状态的变化。在实验中，我们利用数学和统计中常用的多种工具和指标来评估这种对称性的准确性。根据实验结果，正态分布（或高斯分布）在系统中预测位置的解码中起着至关重要的作用。最终，我们构建了一个类似于高尔顿板的算法板，作为发现的对称性的数学基础。 

---
# GSCE: A Prompt Framework with Enhanced Reasoning for Reliable LLM-driven Drone Control 

**Title (ZH)**: GSCE：一种增强推理的提示框架，用于可靠的LLM驱动无人机控制 

**Authors**: Wenhao Wang, Yanyan Li, Long Jiao, Jiawei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12531)  

**Abstract**: The integration of Large Language Models (LLMs) into robotic control, including drones, has the potential to revolutionize autonomous systems. Research studies have demonstrated that LLMs can be leveraged to support robotic operations. However, when facing tasks with complex reasoning, concerns and challenges are raised about the reliability of solutions produced by LLMs. In this paper, we propose a prompt framework with enhanced reasoning to enable reliable LLM-driven control for drones. Our framework consists of novel technical components designed using Guidelines, Skill APIs, Constraints, and Examples, namely GSCE. GSCE is featured by its reliable and constraint-compliant code generation. We performed thorough experiments using GSCE for the control of drones with a wide level of task complexities. Our experiment results demonstrate that GSCE can significantly improve task success rates and completeness compared to baseline approaches, highlighting its potential for reliable LLM-driven autonomous drone systems. 

**Abstract (ZH)**: 将大型语言模型（LLMs）集成到机器人控制中，包括无人机，有望革新自主系统。已有研究显示，LLMs可以在支持机器人操作方面发挥重要作用。然而，在面对需要复杂推理的任务时，关于LLMs生成解决方案的可靠性的担忧逐渐增加。本文提出了一种增强推理的提示框架，以实现可靠的大规模语言模型驱动的无人机控制。该框架包含使用指南、技能API、约束和示例（GSCE）设计的新技术组件。GSCE的特点是其可靠的代码生成和与约束的合规性。我们使用GSCE对不同复杂程度的任务进行了全面的无人机控制实验。实验结果显示，与基线方法相比，GSCE显著提高了任务的成功率和完整性，突显了其在可靠的大规模语言模型驱动的自主无人机系统中的潜力。 

---
# From Abstract to Actionable: Pairwise Shapley Values for Explainable AI 

**Title (ZH)**: 从抽象到可操作：成对夏普利值在可解释人工智能中的应用 

**Authors**: Jiaxin Xu, Hung Chau, Angela Burden  

**Link**: [PDF](https://arxiv.org/pdf/2502.12525)  

**Abstract**: Explainable AI (XAI) is critical for ensuring transparency, accountability, and trust in machine learning systems as black-box models are increasingly deployed within high-stakes domains. Among XAI methods, Shapley values are widely used for their fairness and consistency axioms. However, prevalent Shapley value approximation methods commonly rely on abstract baselines or computationally intensive calculations, which can limit their interpretability and scalability. To address such challenges, we propose Pairwise Shapley Values, a novel framework that grounds feature attributions in explicit, human-relatable comparisons between pairs of data instances proximal in feature space. Our method introduces pairwise reference selection combined with single-value imputation to deliver intuitive, model-agnostic explanations while significantly reducing computational overhead. Here, we demonstrate that Pairwise Shapley Values enhance interpretability across diverse regression and classification scenarios--including real estate pricing, polymer property prediction, and drug discovery datasets. We conclude that the proposed methods enable more transparent AI systems and advance the real-world applicability of XAI. 

**Abstract (ZH)**: 可解释的人工智能（XAI）对于确保机器学习系统在高风险领域中的透明性、问责制和信任至关重要，随着黑盒模型的应用越来越广泛。在XAI方法中，Shapley值因其公平性和一致性而被广泛应用。然而，现有的Shapley值近似方法通常依赖于抽象的基础或计算量大的计算，这可能限制了其可解释性和可扩展性。为解决这些问题，我们提出了一种新的框架——对数Shapley值（Pairwise Shapley Values），该框架通过在特征空间中邻近的数据实例之间进行明确的人类可关联对比来定义特征归因。我们的方法结合了基于样本对的参考选择和单值填充，以提供直观的、模型无关的解释，同时大幅减少了计算负担。在此，我们展示了对数Shapley值在多种回归和分类场景中的解释性提升，包括房地产定价、聚合物性质预测和药物发现数据集。我们得出结论，提出的这些方法能够使AI系统更加透明，并推动XAI在实际应用中的发展。 

---
# YOLOv12: Attention-Centric Real-Time Object Detectors 

**Title (ZH)**: YOLOv12：以注意力为中心的实时物体检测器 

**Authors**: Yunjie Tian, Qixiang Ye, David Doermann  

**Link**: [PDF](https://arxiv.org/pdf/2502.12524)  

**Abstract**: Enhancing the network architecture of the YOLO framework has been crucial for a long time, but has focused on CNN-based improvements despite the proven superiority of attention mechanisms in modeling capabilities. This is because attention-based models cannot match the speed of CNN-based models. This paper proposes an attention-centric YOLO framework, namely YOLOv12, that matches the speed of previous CNN-based ones while harnessing the performance benefits of attention mechanisms. YOLOv12 surpasses all popular real-time object detectors in accuracy with competitive speed. For example, YOLOv12-N achieves 40.6% mAP with an inference latency of 1.64 ms on a T4 GPU, outperforming advanced YOLOv10-N / YOLOv11-N by 2.1%/1.2% mAP with a comparable speed. This advantage extends to other model scales. YOLOv12 also surpasses end-to-end real-time detectors that improve DETR, such as RT-DETR / RT-DETRv2: YOLOv12-S beats RT-DETR-R18 / RT-DETRv2-R18 while running 42% faster, using only 36% of the computation and 45% of the parameters. More comparisons are shown in Figure 1. 

**Abstract (ZH)**: 长久以来，提升YOLO框架的网络架构至关重要，但大多数改进都集中在基于CNN的模型上，尽管注意力机制在建模能力上的优越性已经得到了证实。这是因为基于注意力的模型无法与基于CNN的模型的速度相匹配。本文提出了一种以注意力为中心的YOLO框架，即YOLOv12，其能够在保持与先前基于CNN模型相同速度的同时，发挥注意力机制的性能优势。YOLOv12在准确性上超过了所有流行的真实时间目标检测器，并且具有竞争力的速度。例如，YOLOv12-N在T4 GPU上的推断延迟为1.64毫秒时，实现了40.6%的mAP，优于先进版本的YOLOv10-N / YOLOv11-N，分别提高了2.1%和1.2%的mAP，且运行速度相当。这一优势同样适用于其他模型规模。YOLOv12还在速度提高42%的情况下，仅使用36%的计算资源和45%的参数，击败了改进DETR的端到端实时检测器，如RT-DETR / RT-DETRv2：YOLOv12-S击败了RT-DETR-R18 / RT-DETRv2-R18。更多的比较请参见图1。 

---
# Myna: Masking-Based Contrastive Learning of Musical Representations 

**Title (ZH)**: 《Myna：基于掩码的音乐表示对比学习》

这个翻译符合学术论文标题的规范，保留了原文的关键信息和专业术语。 

**Authors**: Ori Yonay, Tracy Hammond, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12511)  

**Abstract**: We present Myna, a simple yet effective approach for self-supervised musical representation learning. Built on a contrastive learning framework, Myna introduces two key innovations: (1) the use of a Vision Transformer (ViT) on mel-spectrograms as the backbone and (2) a novel data augmentation strategy, token masking, that masks 90 percent of spectrogram tokens. These innovations deliver both effectiveness and efficiency: (i) Token masking enables a significant increase in per-GPU batch size, from 48 or 120 in prior methods (CLMR, MULE) to 4096. (ii) By avoiding traditional augmentations, Myna retains pitch sensitivity, enhancing performance in tasks like key detection. (iii) The use of vertical patches allows the model to better capture critical features for key detection. Our hybrid model, Myna-22M-Hybrid, processes both 16x16 and 128x2 patches, achieving state-of-the-art results. Trained on a single GPU, it outperforms MULE (62M) on average and rivals MERT-95M, which was trained on 16 and 64 GPUs, respectively. Additionally, it surpasses MERT-95M-public, establishing itself as the best-performing model trained on publicly available data. We release our code and models to promote reproducibility and facilitate future research. 

**Abstract (ZH)**: 我们展示了Myna，一种简单而有效的自监督音乐表征学习方法。Myna基于对比学习框架，并引入了两项关键创新：(1) 使用音素图（mel-spectrograms）作为骨干的视觉变压器（Vision Transformer, ViT），以及 (2) 一种新颖的数据增强策略——标记掩蔽，该策略掩蔽了音素图中90%的标记。这些创新在有效性和效率方面都取得了显著效果：(i) 标记掩蔽使得每个GPU的批量大小有了显著增加，从先前方法（CLMR, MULE）的48或120增加到4096。(ii) Myna通过避免传统的数据增强，保留了音高敏感性，从而在键识别等任务上提升了性能。(iii) 采用垂直切片允许模型更好地捕捉键检测所需的特征。我们的混合模型Myna-22M-Hybrid同时处理16x16和128x2的切片，达到了最先进的成果。在单个GPU上进行训练时，它在平均水平上优于MULE（62M），并且与在16和64个GPU上分别训练的MERT-95M竞争。此外，它还超过了MERT-95M-public，确立了在公开数据上训练的最佳模型的地位。我们开放我们的代码和模型以促进可重现性并促进未来的研究。 

---
# LegalCore: A Dataset for Legal Documents Event Coreference Resolution 

**Title (ZH)**: LegalCore：法律文件事件同指解析数据集 

**Authors**: Kangda Wei, Xi Shi, Jonathan Tong, Sai Ramana Reddy, Anandhavelu Natarajan, Rajiv Jain, Aparna Garimella, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12509)  

**Abstract**: Recognizing events and their coreferential mentions in a document is essential for understanding semantic meanings of text. The existing research on event coreference resolution is mostly limited to news articles. In this paper, we present the first dataset for the legal domain, LegalCore, which has been annotated with comprehensive event and event coreference information. The legal contract documents we annotated in this dataset are several times longer than news articles, with an average length of around 25k tokens per document. The annotations show that legal documents have dense event mentions and feature both short-distance and super long-distance coreference links between event mentions. We further benchmark mainstream Large Language Models (LLMs) on this dataset for both event detection and event coreference resolution tasks, and find that this dataset poses significant challenges for state-of-the-art open-source and proprietary LLMs, which perform significantly worse than a supervised baseline. We will publish the dataset as well as the code. 

**Abstract (ZH)**: 识别文档中的事件及其同指提及对于理解文本语义含义至关重要。现有的事件同指消解研究主要集中在新闻文章上。本文介绍了首个适用于法律领域的数据集LegalCore，该数据集包含了全面的事件和事件同指标注信息。我们标注的法律合同文件长度比新闻文章长得多，平均每份文件含有约25,000个词汇。标注结果显示，法律文件中的事件提及密集，并且不仅包括短距离同指链接，还包含超长距离的同指链接。我们进一步在该数据集上对主流的大规模语言模型（LLMs）进行了基准测试，用于事件检测和事件同指消解任务，发现该数据集对最先进的开源和专有LLMs构成了重大挑战，这些模型的表现远逊于监督学习基线。我们将发布该数据集以及相关的代码。 

---
# Mixture of Attention Yields Accurate Results for Tabular Data 

**Title (ZH)**: 混合注意力机制对表格数据结果准确度提高有显著效果 

**Authors**: Xuechen Li, Yupeng Li, Jian Liu, Xiaolin Jin, Tian Yang, Xin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12507)  

**Abstract**: Tabular data inherently exhibits significant feature heterogeneity, but existing transformer-based methods lack specialized mechanisms to handle this property. To bridge the gap, we propose MAYA, an encoder-decoder transformer-based framework. In the encoder, we design a Mixture of Attention (MOA) that constructs multiple parallel attention branches and averages the features at each branch, effectively fusing heterogeneous features while limiting parameter growth. Additionally, we employ collaborative learning with a dynamic consistency weight constraint to produce more robust representations. In the decoder stage, cross-attention is utilized to seamlessly integrate tabular data with corresponding label features. This dual-attention mechanism effectively captures both intra-instance and inter-instance interactions. We evaluate the proposed method on a wide range of datasets and compare it with other state-of-the-art transformer-based methods. Extensive experiments demonstrate that our model achieves superior performance among transformer-based methods in both tabular classification and regression tasks. 

**Abstract (ZH)**: 表格数据本质上表现出显著的特征异质性，但现有的基于变压器的方法缺乏专门处理这种特性的机制。为了解决这一差距，我们提出了一种编码-解码变压器框架MAYA。在编码器中，我们设计了一个混合注意力机制（MOA），构建了多个并行的注意力分支，并在每个分支上平均特征，有效地融合了异质特征的同时限制了参数的增长。此外，我们采用协作学习并引入动态一致性权重约束，以生成更 robust 的表示。在解码阶段，我们利用交叉注意力无缝地将表格数据与对应的标签特征集成。这种双重注意机制有效地捕捉了实例内和实例间的关系。我们在多种数据集上评估了提出的方法，并将其与其他最先进的基于变压器的方法进行了比较。广泛实验证明，我们的模型在表格分类和回归任务中均优于基于变压器的方法的其他模型。 

---
# EDGE: Efficient Data Selection for LLM Agents via Guideline Effectiveness 

**Title (ZH)**: EDGE：通过指南有效性实现高效数据选择的LLM代理方法 

**Authors**: Yunxiao Zhang, Guanming Xiong, Haochen Li, Wen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12494)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities as AI agents. However, existing methods for enhancing LLM-agent abilities often lack a focus on data quality, leading to inefficiencies and suboptimal results in both fine-tuning and prompt engineering. To address this issue, we introduce EDGE, a novel approach for identifying informative samples without needing golden answers. We propose the Guideline Effectiveness (GE) metric, which selects challenging samples by measuring the impact of human-provided guidelines in multi-turn interaction tasks. A low GE score indicates that the human expertise required for a sample is missing from the guideline, making the sample more informative. By selecting samples with low GE scores, we can improve the efficiency and outcomes of both prompt engineering and fine-tuning processes for LLMs. Extensive experiments validate the performance of our method. Our method achieves competitive results on the HotpotQA and WebShop and datasets, requiring 75\% and 50\% less data, respectively, while outperforming existing methods. We also provide a fresh perspective on the data quality of LLM-agent fine-tuning. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了作为AI代理的出色能力。然而，现有的增强LLM代理能力的方法往往缺乏对数据质量的关注，这导致了在微调和提示工程中效率低下和结果不佳的问题。为了解决这一问题，我们提出了EDGE，一种无需金标准答案即可识别有效样本的新方法。我们提出了指导有效性（GE）指标，该指标通过测量人类提供的指南在多轮交互任务中的影响来选择具有挑战性的样本。GE分数较低表明样本所需的人类专业知识未包含在指南中，使样本更具信息量。通过选择GE分数较低的样本，我们可以提高LLM提示工程和微调过程的效率和结果。广泛实验证明了我们方法的有效性。我们的方法在HotpotQA和WebShop数据集上取得了具有竞争力的结果，在分别减少了75%和50%数据量的情况下超越了现有方法。我们还从新的角度审视了LLM代理微调中的数据质量。 

---
# A Comprehensive Survey on Generative AI for Video-to-Music Generation 

**Title (ZH)**: 面向视频到音乐生成的生成型AI综述 

**Authors**: Shulei Ji, Songruoyao Wu, Zihao Wang, Shuyu Li, Kejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12489)  

**Abstract**: The burgeoning growth of video-to-music generation can be attributed to the ascendancy of multimodal generative models. However, there is a lack of literature that comprehensively combs through the work in this field. To fill this gap, this paper presents a comprehensive review of video-to-music generation using deep generative AI techniques, focusing on three key components: visual feature extraction, music generation frameworks, and conditioning mechanisms. We categorize existing approaches based on their designs for each component, clarifying the roles of different strategies. Preceding this, we provide a fine-grained classification of video and music modalities, illustrating how different categories influence the design of components within the generation pipelines. Furthermore, we summarize available multimodal datasets and evaluation metrics while highlighting ongoing challenges in the field. 

**Abstract (ZH)**: 视频到音乐生成的日益增长可以归因于多模态生成模型的兴起。然而，该领域的现有文献尚未进行全面梳理。为填补这一空白，本文综述了使用深度生成AI技术的视频到音乐生成方法，重点关注三个关键组成部分：视觉特征提取、音乐生成框架和条件机制。我们根据每个组成部分的设计对现有方法进行了分类，阐明了不同策略的作用。在此基础上，我们提供了视频和音乐模态的精细分类，说明不同类别如何影响生成管道中各组成部分的设计。此外，我们总结了可用的多模态数据集和评估指标，并指出了该领域的持续挑战。 

---
# Safe at the Margins: A General Approach to Safety Alignment in Low-Resource English Languages -- A Singlish Case Study 

**Title (ZH)**: 处于边缘地带的安全保障：一种低资源英语语言安全对齐的通用方法——以新加坡英语（Singlish）案例研究为例 

**Authors**: Isaac Lim, Shaun Khoo, Watson Chua, Goh Jiayi, Jessica Foo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12485)  

**Abstract**: To ensure safe usage, Large Language Models (LLMs) typically undergo alignment with human-defined values. However, this alignment often relies on primarily English data and is biased towards Western-centric values, limiting its effectiveness in low-resource language settings. In this paper, we describe our approach for aligning SEA-Lion-v2.1-Instruct (a Llama3-8B variant) to minimize toxicity in Singlish, an English creole specific to Singapore. We find that supervised fine-tuning and Kahneman-Tversky Optimization (KTO) on paired and unpaired preferences is more sample efficient and yields significantly better results than Direct Preference Optimization (DPO). Our analysis reveals that DPO implicitly enforces a weaker safety objective than KTO, and that SFT complements KTO by improving training stability. Finally, we introduce a simple but novel modification to KTO, KTO-S, which improves training stability through better gradient exploitation. Overall, we present a general approach for safety alignment conducive to low-resource English languages, successfully reducing toxicity by 99\% on our Singlish benchmark, with gains generalizing to the broader TOXIGEN dataset while maintaining strong performance across standard LLM benchmarks. 

**Abstract (ZH)**: 为了确保安全使用，大型语言模型（LLMs）通常会与人类定义的价值观进行对齐。然而，这种对齐往往依赖于主要的英文数据，并倾向于西方价值观，这限制了其在低资源语言环境中的有效性。本文中，我们描述了我们对SEA-Lion-v2.1-Instruct（一种Llama3-8B变体）进行对齐的方法，旨在最小化新加坡英语（Singlish）中的毒性。新加坡英语是一种特定于新加坡的英语克里奥尔语。我们发现，监督微调和Kahneman-Tversky优化（KTO）对于成对和非成对偏好具有更高的样本效率，并且结果显著优于直接偏好优化（DPO）。我们的分析揭示，DPO 显式地施加了一个比 KTO 更弱的安全目标，而 SFT 通过提高训练稳定性来补充 KTO。最后，我们引入了 KTO-S，这是 KTO 的一种简单但新颖的改进，通过更好地利用梯度来提高训练稳定性。总体而言，我们提出了一种适用于低资源英语语言的安全对齐方法，成功地将我们的 Singlish 标准基准中的毒性降低了99%，并且这种改进在 TOXIGEN 数据集和标准 LLM 基准测试中都表现出较好的通用性。 

---
# LocalEscaper: A Weakly-supervised Framework with Regional Reconstruction for Scalable Neural TSP Solvers 

**Title (ZH)**: LocalEscaper：一种基于区域重建的弱监督框架，用于可扩展的神经TSP求解器 

**Authors**: Junrui Wen, Yifei Li, Bart Selman, Kun He  

**Link**: [PDF](https://arxiv.org/pdf/2502.12484)  

**Abstract**: Neural solvers have shown significant potential in solving the Traveling Salesman Problem (TSP), yet current approaches face significant challenges. Supervised learning (SL)-based solvers require large amounts of high-quality labeled data, while reinforcement learning (RL)-based solvers, though less dependent on such data, often suffer from inefficiencies. To address these limitations, we propose LocalEscaper, a novel weakly-supervised learning framework for large-scale TSP. LocalEscaper effectively combines the advantages of both SL and RL, enabling effective training on datasets with low-quality labels. To further enhance solution quality, we introduce a regional reconstruction strategy, which mitigates the problem of local optima, a common issue in existing local reconstruction methods. Additionally, we propose a linear-complexity attention mechanism that reduces computational overhead, enabling the efficient solution of large-scale TSPs without sacrificing performance. Experimental results on both synthetic and real-world datasets demonstrate that LocalEscaper outperforms existing neural solvers, achieving state-of-the-art results. Notably, it sets a new benchmark for scalability and efficiency, solving TSP instances with up to 50,000 cities. 

**Abstract (ZH)**: 神经网络求解器在解决旅行商问题（TSP）方面已经显示出显著的潜力，但当前的方法仍面临重大挑战。基于监督学习（SL）的求解器需要大量的高质量标签数据，而基于强化学习（RL）的求解器虽然对数据的依赖性较低，但往往受到效率问题的影响。为了解决这些限制，我们提出了一种新颖的弱监督学习框架LocalEscaper，旨在处理大规模TSP问题。LocalEscaper有效结合了SL和RL的优点，能够在带有低质量标签的数据集上实现有效的训练。为了进一步提高解的质量，我们引入了一种区域重构策略，该策略减轻了局部最优解问题，这是现有局部重构方法中的常见问题。此外，我们提出了一种线性复杂度的注意力机制，该机制减少了计算开销，使得可以在不牺牲性能的情况下高效解决大规模TSP问题。在合成数据集和真实世界数据集上的实验结果表明，LocalEscaper在多项指标上优于现有的神经网络求解器，达到了最先进的成果。值得注意的是，它在可扩展性和效率方面设置了新的基准，能够解决包含多达50,000个城市的TSP实例。 

---
# Predicate Hierarchies Improve Few-Shot State Classification 

**Title (ZH)**: 谓词层次结构提升少量样本状态分类 

**Authors**: Emily Jin, Joy Hsu, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12481)  

**Abstract**: State classification of objects and their relations is core to many long-horizon tasks, particularly in robot planning and manipulation. However, the combinatorial explosion of possible object-predicate combinations, coupled with the need to adapt to novel real-world environments, makes it a desideratum for state classification models to generalize to novel queries with few examples. To this end, we propose PHIER, which leverages predicate hierarchies to generalize effectively in few-shot scenarios. PHIER uses an object-centric scene encoder, self-supervised losses that infer semantic relations between predicates, and a hyperbolic distance metric that captures hierarchical structure; it learns a structured latent space of image-predicate pairs that guides reasoning over state classification queries. We evaluate PHIER in the CALVIN and BEHAVIOR robotic environments and show that PHIER significantly outperforms existing methods in few-shot, out-of-distribution state classification, and demonstrates strong zero- and few-shot generalization from simulated to real-world tasks. Our results demonstrate that leveraging predicate hierarchies improves performance on state classification tasks with limited data. 

**Abstract (ZH)**: 物体及其关系的状态分类是许多长期任务的核心，特别是在机器人规划与操作中。然而，物体谓词组合的组合爆炸以及需要适应新的现实环境，使得状态分类模型能够在少量示例的情况下泛化到新查询的需求变得尤为重要。为此，我们提出了PHIER模型，该模型利用谓词层次结构来有效泛化至少样本场景。PHIER采用了以物体为中心的场景编码器、自监督损失来推断谓词之间的语义关系、以及一种双曲距离度量来捕获层次结构；它学习了一个由图像-谓词对构成的结构化隐空间，该隐空间可引导状态分类查询的推理。我们分别在CALVIN和BEHAVIOR两类机器人环境中对PHIER进行了评估，并展示了PHIER在少样本、分布外状态分类任务中显著优于现有方法，同时展示了其从模拟环境到现实任务的强大的零样本和少样本泛化能力。我们的实验结果表明，利用谓词层次结构在数据有限的情况下可以提高状态分类任务的性能。 

---
# MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation 

**Title (ZH)**: MCTS-Judge：作为代码正确性评估的LLM法官的测试时缩放 

**Authors**: Yutong Wang, Pengliang Ji, Chaoqun Yang, Kaixin Li, Ming Hu, Jiaoyang Li, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2502.12468)  

**Abstract**: The LLM-as-a-Judge paradigm shows promise for evaluating generative content but lacks reliability in reasoning-intensive scenarios, such as programming. Inspired by recent advances in reasoning models and shifts in scaling laws, we pioneer bringing test-time computation into LLM-as-a-Judge, proposing MCTS-Judge, a resource-efficient, System-2 thinking framework for code correctness evaluation. MCTS-Judge leverages Monte Carlo Tree Search (MCTS) to decompose problems into simpler, multi-perspective evaluations. Through a node-selection strategy that combines self-assessment based on historical actions in the current trajectory and the Upper Confidence Bound for Trees based on prior rollouts, MCTS-Judge balances global optimization and refinement of the current trajectory. We further designed a high-precision, unit-test-level reward mechanism to encourage the Large Language Model (LLM) to perform line-by-line analysis. Extensive experiments on three benchmarks and five LLMs demonstrate the effectiveness of MCTS-Judge, which improves the base model's accuracy from 41% to 80%, surpassing the o1-series models with 3x fewer tokens. Further evaluations validate the superiority of its reasoning trajectory in logic, analytics, thoroughness, and overall quality, while revealing the test-time scaling law of the LLM-as-a-Judge paradigm. 

**Abstract (ZH)**: LLM作为法官的范式在评估生成内容方面展现出潜力，但在编程等需要推理的场景中缺乏可靠性。借鉴最近推理模型的进步和规模法则的转变，我们率先将推理时的计算引入LLM作为法官的情景，提出了MCTS-Judge，这是一种资源高效的、适用于代码正确性评估的系统-2思维框架。MCTS-Judge利用蒙特卡洛树搜索（MCTS）将问题分解为更简单的多视角评估。通过结合基于当前轨迹历史行为的自我评估和基于先验展开的树的信心上限（UCB）的节点选择策略，MCTS-Judge平衡了全局优化和当前轨迹的细化。我们进一步设计了一种高精度、单元测试级别的奖励机制，鼓励大型语言模型（LLM）进行逐行分析。在三个基准和五种LLM的广泛实验中，MCTS-Judge展示了其有效性，将基线模型的准确率从41%提高到80%，并且在使用三分之一更少的令牌时超越了o1系列模型。进一步的评估验证了其推理轨迹在逻辑性、分析能力、覆盖面和整体质量上的优越性，同时揭示了LLM作为法官范式的推理时的规模法则。 

---
# EquiBench: Benchmarking Code Reasoning Capabilities of Large Language Models via Equivalence Checking 

**Title (ZH)**: EquiBench：通过等价性检查评估大型语言模型的代码推理能力 

**Authors**: Anjiang Wei, Jiannan Cao, Ran Li, Hongyu Chen, Yuhui Zhang, Ziheng Wang, Yaofeng Sun, Yuan Liu, Thiago S. F. X. Teixeira, Diyi Yang, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2502.12466)  

**Abstract**: Equivalence checking, i.e., determining whether two programs produce identical outputs for all possible inputs, underpins a broad range of applications, including software refactoring, testing, and optimization. We present the task of equivalence checking as a new way to evaluate the code reasoning abilities of large language models (LLMs). We introduce EquiBench, a dataset of 2400 program pairs spanning four programming languages and six equivalence categories. These pairs are systematically generated through program analysis, compiler scheduling, and superoptimization, covering nontrivial structural transformations that demand deep semantic reasoning beyond simple syntactic variations. Our evaluation of 17 state-of-the-art LLMs shows that OpenAI o3-mini achieves the highest overall accuracy of 78.0%. In the most challenging categories, the best accuracies are 62.3% and 68.8%, only modestly above the 50% random baseline for binary classification, indicating significant room for improvement in current models' code reasoning capabilities. 

**Abstract (ZH)**: 程序等价检验是指确定两个程序在所有可能输入下的输出是否完全相同，这为软件重构、测试和优化等多种应用提供了基础。本文将程序等价检验任务视为评估大型语言模型（LLMs）代码推理能力的一种新方法。我们引入了EquiBench数据集，包含2400个程序对，覆盖了四种编程语言和六种等价类别。这些程序对通过程序分析、编译器调度和超优化系统性生成，涵盖了需要进行深层次语义推理而非仅仅简单的语法变化的复杂结构变换。对17个最先进的LLM的评估结果显示，OpenAI o3-mini 在整体准确率方面最高，达到了78.0%。在最具挑战性的类别中，最高的准确率为62.3%和68.8%，仅略高于二分类的50%随机基线，这表明当前模型在代码推理能力方面仍存在显著的提升空间。 

---
# Stress Testing Generalization: How Minor Modifications Undermine Large Language Model Performance 

**Title (ZH)**: 压力测试泛化能力：细微修改如何削弱大型语言模型的性能 

**Authors**: Guangxiang Zhao, Saier Hu, Xiaoqi Jian, Jinzhu Wu, Yuhan Wu, Change Jia, Lin Sun, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12459)  

**Abstract**: This paper investigates the fragility of Large Language Models (LLMs) in generalizing to novel inputs, specifically focusing on minor perturbations in well-established benchmarks (e.g., slight changes in question format or distractor length). Despite high benchmark scores, LLMs exhibit significant accuracy drops and unexpected biases (e.g., preference for longer distractors) when faced with these minor but content-preserving modifications. For example, Qwen 2.5 1.5B's MMLU score rises from 60 to 89 and drops from 89 to 36 when option lengths are changed without altering the question. Even GPT-4 experiences a 25-point accuracy loss when question types are changed, with a 6-point drop across all three modification categories. These analyses suggest that LLMs rely heavily on superficial cues rather than forming robust, abstract representations that generalize across formats, lexical variations, and irrelevant content shifts. This work aligns with the ACL 2025 theme track on the Generalization of NLP models, proposing a "Generalization Stress Test" to assess performance shifts under controlled perturbations. The study calls for reevaluating benchmarks and developing more reliable evaluation methodologies to capture LLM generalization abilities better. 

**Abstract (ZH)**: 本文探讨了大型语言模型（LLMs）在泛化到新颖输入时的脆弱性，特别关注于在成熟基准测试中细微扰动的影响（例如，问题格式的轻微变化或干扰项长度的变化）。尽管在基准测试中的得分很高，但LLMs在面对这些细微但内容保持不变的修改时，显示出显著的准确率下降和意想不到的偏差（例如，偏好更长的干扰项）。例如，当更改选项长度而不改变问题时，Qwen 2.5 1.5B的MMLU分数从60升至89，然后又从89降至36。即使GPT-4在更改问题类型时也经历了25点准确率的下降，且在所有三种修改类别中均降低了6点准确率。这些分析表明，LLMs依赖于表面特征而非形成能够跨越不同格式、词法变化和无关内容移位的稳健、抽象表示。本文与ACL 2025的主题研讨会相契合，该研讨会关注于自然语言处理模型的泛化能力，并提出了一种“泛化压力测试”来评估在受控扰动下性能的变化。研究呼吁重新评估基准测试并开发更可靠的评估方法，以更好地捕捉LLMs的泛化能力。 

---
# Not-So-Optimal Transport Flows for 3D Point Cloud Generation 

**Title (ZH)**: 不那么最优的运输流生成3D点云 

**Authors**: Ka-Hei Hui, Chao Liu, Xiaohui Zeng, Chi-Wing Fu, Arash Vahdat  

**Link**: [PDF](https://arxiv.org/pdf/2502.12456)  

**Abstract**: Learning generative models of 3D point clouds is one of the fundamental problems in 3D generative learning. One of the key properties of point clouds is their permutation invariance, i.e., changing the order of points in a point cloud does not change the shape they represent. In this paper, we analyze the recently proposed equivariant OT flows that learn permutation invariant generative models for point-based molecular data and we show that these models scale poorly on large point clouds. Also, we observe learning (equivariant) OT flows is generally challenging since straightening flow trajectories makes the learned flow model complex at the beginning of the trajectory. To remedy these, we propose not-so-optimal transport flow models that obtain an approximate OT by an offline OT precomputation, enabling an efficient construction of OT pairs for training. During training, we can additionally construct a hybrid coupling by combining our approximate OT and independent coupling to make the target flow models easier to learn. In an extensive empirical study, we show that our proposed model outperforms prior diffusion- and flow-based approaches on a wide range of unconditional generation and shape completion on the ShapeNet benchmark. 

**Abstract (ZH)**: 学习3D点云生成模型是3D生成学习中的基础问题之一。点云的一个关键性质是其置换不变性，即更改点云中点的顺序不会改变它所代表的形状。在本文中，我们分析了最近提出的对称OT流，这些流用于学习基于点的分子数据的置换不变生成模型，并表明这些模型在大规模点云上运行缓慢。同时，我们观察到学习（对称的）OT流通常具有挑战性，因为拉直流轨迹会使学习到的流模型在轨迹的开始变得复杂。为了解决这些问题，我们提出了非最优运输流模型，通过离线预计算OT来获得近似的OT，从而使训练过程中的OT对更加高效地构建。在训练过程中，我们还可以通过结合我们的近似OT和独立耦合来构建混合耦合，从而让目标流模型更容易学习。在广泛的实证研究中，我们展示了在ShapeNet基准测试上的无条件生成和形状补全任务中，我们的提出模型优于先前的扩散和流基方法。 

---
# Benchmarking Zero-Shot Facial Emotion Annotation with Large Language Models: A Multi-Class and Multi-Frame Approach in DailyLife 

**Title (ZH)**: 基于大型语言模型的零样本面部情感标注基准测试：日常生活中的多类别和多帧方法 

**Authors**: He Zhang, Xinyi Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12454)  

**Abstract**: This study investigates the feasibility and performance of using large language models (LLMs) to automatically annotate human emotions in everyday scenarios. We conducted experiments on the DailyLife subset of the publicly available FERV39k dataset, employing the GPT-4o-mini model for rapid, zero-shot labeling of key frames extracted from video segments. Under a seven-class emotion taxonomy ("Angry," "Disgust," "Fear," "Happy," "Neutral," "Sad," "Surprise"), the LLM achieved an average precision of approximately 50%. In contrast, when limited to ternary emotion classification (negative/neutral/positive), the average precision increased to approximately 64%. Additionally, we explored a strategy that integrates multiple frames within 1-2 second video clips to enhance labeling performance and reduce costs. The results indicate that this approach can slightly improve annotation accuracy. Overall, our preliminary findings highlight the potential application of zero-shot LLMs in human facial emotion annotation tasks, offering new avenues for reducing labeling costs and broadening the applicability of LLMs in complex multimodal environments. 

**Abstract (ZH)**: 本研究探讨了使用大规模语言模型（LLMs）自动标注日常生活场景中人类情绪的可行性和性能。我们使用 GPT-4o-mini 模型对公开可用的 FERV39k 数据集中的 DailyLife 子集进行实验，对视频片段中提取的关键帧进行了快速零样本标注。使用七类情绪分类体系（“愤怒”、“厌恶”、“恐惧”、“快乐”、“中性”、“悲伤”、“惊讶”），LLM 的平均精度约为 50%。相比之下，仅限于三类情绪分类（消极/中性/积极）时，平均精度提高到约 64%。此外，我们还研究了一种策略，即将 1-2 秒视频片段内的多个帧进行集成，以提高标注性能并降低标注成本。结果显示，这种策略可以在一定程度上提高标注准确性。总体而言，初步研究结果表明零样本 LLMs 在人类面部情绪标注任务中的潜在应用价值，为降低标注成本和扩大 LLMs 在复杂多模态环境中的应用提供了新的途径。 

---
# UniMatch: Universal Matching from Atom to Task for Few-Shot Drug Discovery 

**Title (ZH)**: UniMatch：从原子到任务的通用少样本药物发现匹配方法 

**Authors**: Ruifeng Li, Mingqian Li, Wei Liu, Yuhua Zhou, Xiangxin Zhou, Yuan Yao, Qiang Zhang, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12453)  

**Abstract**: Drug discovery is crucial for identifying candidate drugs for various this http URL, its low success rate often results in a scarcity of annotations, posing a few-shot learning problem. Existing methods primarily focus on single-scale features, overlooking the hierarchical molecular structures that determine different molecular properties. To address these issues, we introduce Universal Matching Networks (UniMatch), a dual matching framework that integrates explicit hierarchical molecular matching with implicit task-level matching via meta-learning, bridging multi-level molecular representations and task-level generalization. Specifically, our approach explicitly captures structural features across multiple levels, such as atoms, substructures, and molecules, via hierarchical pooling and matching, facilitating precise molecular representation and comparison. Additionally, we employ a meta-learning strategy for implicit task-level matching, allowing the model to capture shared patterns across tasks and quickly adapt to new ones. This unified matching framework ensures effective molecular alignment while leveraging shared meta-knowledge for fast adaptation. Our experimental results demonstrate that UniMatch outperforms state-of-the-art methods on the MoleculeNet and FS-Mol benchmarks, achieving improvements of 2.87% in AUROC and 6.52% in delta AUPRC. UniMatch also shows excellent generalization ability on the Meta-MolNet benchmark. 

**Abstract (ZH)**: 药物发现对于识别各种候选药物至关重要，但由于其较低的成功率，往往导致标注数据稀缺，从而引发少量样本学习问题。现有方法主要关注单尺度特征，而忽略了决定不同分子性质的分层分子结构。为解决这些问题，我们引入了通用匹配网络（UniMatch），这是一种结合显式分层分子匹配和隐式任务级匹配的双匹配框架，通过元学习将多层次的分子表示与任务级泛化能力联系起来。具体而言，我们的方法通过分层聚合和匹配，明确地捕捉多个层次的结构特征，如原子、亚结构和分子，从而实现精确的分子表示和比较。此外，我们采用元学习策略进行隐式任务级匹配，使模型能够捕捉不同任务之间的共享模式，并快速适应新的任务。这种统一的匹配框架确保实现有效的分子对齐，同时利用共享的元知识实现快速适应。实验结果表明，在MoleculeNet和FS-Mol基准上，UniMatch优于最新方法，分别在AUROC上提高了2.87%，在delta AUPRC上提高了6.52%。此外，UniMatch在Meta-MolNet基准上的泛化能力也非常出色。 

---
# Multi-Attribute Steering of Language Models via Targeted Intervention 

**Title (ZH)**: 通过目标干预实现语言模型的多属性引导 

**Authors**: Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.12446)  

**Abstract**: Inference-time intervention (ITI) has emerged as a promising method for steering large language model (LLM) behavior in a particular direction (e.g., improving helpfulness) by intervening on token representations without costly updates to the LLM's parameters. However, existing ITI approaches fail to scale to multi-attribute settings with conflicts, such as enhancing helpfulness while also reducing toxicity. To address this, we introduce Multi-Attribute Targeted Steering (MAT-Steer), a novel steering framework designed for selective token-level intervention across multiple attributes. MAT-Steer learns steering vectors using an alignment objective that shifts the model's internal representations of undesirable outputs closer to those of desirable ones while enforcing sparsity and orthogonality among vectors for different attributes, thereby reducing inter-attribute conflicts. We evaluate MAT-Steer in two distinct settings: (i) on question answering (QA) tasks where we balance attributes like truthfulness, bias, and toxicity; (ii) on generative tasks where we simultaneously improve attributes like helpfulness, correctness, and coherence. MAT-Steer outperforms existing ITI and parameter-efficient finetuning approaches across both task types (e.g., 3% average accuracy gain across QA tasks and 55.82% win rate against the best ITI baseline). 

**Abstract (ZH)**: 推理时干预（ITI）已经发展成为一种有前景的方法，在不需要对大型语言模型（LLM）参数进行昂贵更新的情况下，通过干预词元表示来引导LLM的行为朝特定方向发展（例如，提高有用性）。然而，现有的ITI方法难以处理具有冲突的多属性设置，例如在提高有用性的同时减少毒性。为了应对这一挑战，我们引入了多属性定向引导（MAT-Steer），这是一种新颖的引导框架，旨在进行多属性的、选择性的词元级干预。MAT-Steer 使用一种对齐目标来引导模型内部表示，使其不理想的输出接近理想的输出，并通过不同属性间向量的稀疏性和正交性约束来减少属性间的冲突。我们通过两种不同的设置对MAT-Steer进行评估：（i）在问答（QA）任务中，我们平衡真理性和偏见、毒性等属性；（ii）在生成任务中，我们同时提高有用性、正确性和连贯性等属性。在两种任务类型中，MAT-Steer 都优于现有的 ITI 方法和参数效率的微调方法，例如，在问答任务中平均准确率提高3%，并在对抗最好的ITI基线时胜率高达55.82%。 

---
# SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs 

**Title (ZH)**: SparAMX：加速基于AMX处理器的压缩大型语言模型-token生成 

**Authors**: Ahmed F. AbouElhamayed, Jordan Dotzel, Yash Akhauri, Chi-Chih Chang, Sameh Gobriel, J. Pablo Muñoz, Vui Seng Chua, Nilesh Jain, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2502.12444)  

**Abstract**: Large language models have high compute, latency, and memory requirements. While specialized accelerators such as GPUs and TPUs typically run these workloads, CPUs are more widely available and consume less energy. Accelerating LLMs with CPUs enables broader AI access at a lower cost and power consumption. This acceleration potential for CPUs is especially relevant during the memory-bound decoding stage of LLM inference, which processes one token at a time and is becoming increasingly utilized with reasoning models. We utilize Advanced Matrix Extensions (AMX) support on the latest Intel CPUs together with unstructured sparsity to achieve a $1.42 \times$ reduction in end-to-end latency compared to the current PyTorch implementation by applying our technique in linear layers. We provide a set of open-source customized sparse kernels that can speed up any PyTorch model by automatically replacing all linear layers with our custom sparse implementation. Furthermore, we demonstrate for the first time the use of unstructured sparsity in the attention computation achieving a $1.14 \times$ speedup over the current systems without compromising accuracy. Code: this https URL 

**Abstract (ZH)**: 大型语言模型在计算、延迟和内存需求方面要求较高。尽管专用加速器如GPU和TPU通常用于运行这些工作负载，但CPU由于其更广泛的可用性和更低的能耗，更具潜力。使用CPU加速大型语言模型能够以更低的成本和能耗实现更广泛的AI访问。这种CPU加速潜力尤其适用于大型语言模型推理中的内存受限解码阶段，这一阶段一次处理一个词元，并且随着推理解释模型的应用越来越广泛。我们利用最新Intel CPU对高级矩阵扩展（AMX）的支持以及无结构稀疏性，通过在线性层中应用这种方法，实现了端到端延迟比当前PyTorch实现减少1.42倍的效果。我们提供了一组开源定制稀疏内核，可以自动将所有线性层替换为我们的定制稀疏实现，从而加速任何PyTorch模型。此外，我们首次展示了在注意力计算中使用无结构稀疏性，实现比当前系统快1.14倍的速度而无需牺牲准确性。代码：[这里](this https URL) 

---
# Bridge the Gaps between Machine Unlearning and AI Regulation 

**Title (ZH)**: 桥接机器卸载与人工智能监管之间的差距 

**Authors**: Bill Marino, Meghdad Kurmanji, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2502.12430)  

**Abstract**: The "right to be forgotten" and the data privacy laws that encode it have motivated machine unlearning since its earliest days. Now, an inbound wave of artificial intelligence regulations - like the European Union's Artificial Intelligence Act (AIA) - potentially offer important new use cases for machine unlearning. However, this position paper argues, this opportunity will only be realized if researchers, aided by policymakers, proactively bridge the (sometimes sizable) gaps between machine unlearning's state of the art and its potential applications to AI regulation. To demonstrate this point, we use the AIA as an example. Specifically, we deliver a "state of the union" as regards machine unlearning's current potential for aiding compliance with the AIA. This starts with a precise cataloging of the potential applications of machine unlearning to AIA compliance. For each, we flag any legal ambiguities clouding the potential application and, moreover, flag the technical gaps that exist between the potential application and the state of the art of machine unlearning. Finally, we end with a call to action: for both machine learning researchers and policymakers, to, respectively, solve the open technical and legal questions that will unlock machine unlearning's potential to assist compliance with the AIA - and other AI regulation like it. 

**Abstract (ZH)**: “被遗忘的权利”及其所体现的数据隐私法律激发了机器遗忘技术的发展。如今，包括欧盟《人工智能法案》（AAI）在内的新一轮针对人工智能的监管措施可能会为机器遗忘技术带来重要的新应用场景。然而，本文认为，这一机会只有在研究者在政策制定者的支持下，主动弥合机器遗忘技术的现状与潜在应用之间的差距（这些差距有时较大）时，才会实现。为了证明这一点，我们以欧盟《人工智能法案》为例。具体而言，我们提供了机器遗忘技术当前在帮助遵守《人工智能法案》方面潜在应用的概览。对于每一种潜在应用，我们都指出了可能存在的法律模糊性，并且还指出了当前机器遗忘技术与潜在应用之间的技术差距。最后，我们提出了行动呼吁：对于机器学习研究者和政策制定者来说，分别解决这些开放的技术和法律问题，以解锁机器遗忘技术在帮助遵守《人工智能法案》及其他类似人工智能监管方面的潜力。

通过上述翻译，我们尽量保持了原文的学术规范和专业语气，同时确保语言流畅、易于理解。 

---
# Sens-Merging: Sensitivity-Guided Parameter Balancing for Merging Large Language Models 

**Title (ZH)**: Sens-合并：面向敏感性的参数平衡方法以融合大规模语言模型 

**Authors**: Shuqi Liu, Han Wu, Bowei He, Xiongwei Han, Mingxuan Yuan, Linqin Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.12420)  

**Abstract**: Recent advances in large language models have led to numerous task-specialized fine-tuned variants, creating a need for efficient model merging techniques that preserve specialized capabilities while avoiding costly retraining. While existing task vector-based merging methods show promise, they typically apply uniform coefficients across all parameters, overlooking varying parameter importance both within and across tasks. We present Sens-Merging, a sensitivity-guided coefficient adjustment method that enhances existing model merging techniques by operating at both task-specific and cross-task levels. Our method analyzes parameter sensitivity within individual tasks and evaluates cross-task transferability to determine optimal merging coefficients. Extensive experiments on Mistral 7B and LLaMA2-7B/13B models demonstrate that Sens-Merging significantly improves performance across general knowledge, mathematical reasoning, and code generation tasks. Notably, when combined with existing merging techniques, our method enables merged models to outperform specialized fine-tuned models, particularly in code generation tasks. Our findings reveal important trade-offs between task-specific and cross-task scalings, providing insights for future model merging strategies. 

**Abstract (ZH)**: 近年来，大规模语言模型的进展导致了众多专门化任务微调变体的出现，这需要有高效的模型合并技术，能够在保留专门化能力的同时避免昂贵的重新训练。现有的基于任务向量的合并方法虽然显示出潜力，但通常会在所有参数上应用统一的系数，忽视了在同一任务内以及不同任务间的参数重要性变化。我们提出了一种名为Sens-Merging的敏感性导向系数调整方法，该方法通过在任务特定层面和跨任务层面增强现有的模型合并技术，从而改进了现有的合并方法。我们的方法在单个任务内分析参数的敏感性，并评估跨任务的可移植性，以确定最佳的合并系数。我们在Mistral 7B和LLaMA2-7B/13B模型上的 extensive 实验表明，Sens-Merging 显著提高了常识知识、数学推理和代码生成任务的性能。值得注意的是，当与其他合并技术结合使用时，我们的方法能使合并模型在代码生成任务中表现优于专门微调的模型。我们的研究揭示了任务特定和跨任务缩放之间的关键权衡，为未来的模型合并策略提供了重要见解。 

---
# Boosting Illuminant Estimation in Deep Color Constancy through Enhancing Brightness Robustness 

**Title (ZH)**: 通过增强亮度稳健性来提升深度颜色一致性中的照明估计增强 

**Authors**: Mengda Xie, Chengzhi Zhong, Yiling He, Zhan Qin, Meie Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12418)  

**Abstract**: Color constancy estimates illuminant chromaticity to correct color-biased images. Recently, Deep Neural Network-driven Color Constancy (DNNCC) models have made substantial advancements. Nevertheless, the potential risks in DNNCC due to the vulnerability of deep neural networks have not yet been explored. In this paper, we conduct the first investigation into the impact of a key factor in color constancy-brightness-on DNNCC from a robustness perspective. Our evaluation reveals that several mainstream DNNCC models exhibit high sensitivity to brightness despite their focus on chromaticity estimation. This sheds light on a potential limitation of existing DNNCC models: their sensitivity to brightness may hinder performance given the widespread brightness variations in real-world datasets. From the insights of our analysis, we propose a simple yet effective brightness robustness enhancement strategy for DNNCC models, termed BRE. The core of BRE is built upon the adaptive step-size adversarial brightness augmentation technique, which identifies high-risk brightness variation and generates augmented images via explicit brightness adjustment. Subsequently, BRE develops a brightness-robustness-aware model optimization strategy that integrates adversarial brightness training and brightness contrastive loss, significantly bolstering the brightness robustness of DNNCC models. BRE is hyperparameter-free and can be integrated into existing DNNCC models, without incurring additional overhead during the testing phase. Experiments on two public color constancy datasets-ColorChecker and Cube+-demonstrate that the proposed BRE consistently enhances the illuminant estimation performance of existing DNNCC models, reducing the estimation error by an average of 5.04% across six mainstream DNNCC models, underscoring the critical role of enhancing brightness robustness in these models. 

**Abstract (ZH)**: 颜色恒常性通过估计光源色度来纠正颜色偏倚图像。近年来，基于深度神经网络的色彩恒定性（DNNCC）模型取得了显著进展。然而，这些模型由于深度神经网络的脆弱性而可能带来的潜在风险尚未被充分探索。本文首次从鲁棒性的角度探讨了颜色恒定性中的关键因素——亮度——对DNNCC模型的影响。我们的评估表明，尽管主流的DNNCC模型专注于色度估计，它们对亮度的敏感性仍然很高。这揭示了现有DNNCC模型的一个潜在局限性：亮度敏感性可能因为真实世界数据集中普遍存在的亮度变化而影响模型性能。基于我们的分析结果，我们提出了一种简单而有效的亮度鲁棒性增强策略，称为BRE。BRE的核心在于自适应步长对抗性亮度增强技术，该技术识别高风险亮度变化并通过明确的亮度调整生成增强图像。随后，BRE提出了一个亮度鲁棒性意识的模型优化策略，该策略将对抗性亮度训练和亮度对比损失相结合，显著增强了DNNCC模型的亮度鲁棒性。BRE无需超参数，并且可以在不增加测试阶段额外开销的情况下集成到现有的DNNCC模型中。在两个公开的颜色恒定性数据集ColorChecker和Cube+上的实验结果表明，提出的BRE能够一致地增强现有DNNCC模型的光源估计性能，平均降低估计误差5.04%，突显了增强亮度鲁棒性在这些模型中的关键作用。 

---
# Gradient Co-occurrence Analysis for Detecting Unsafe Prompts in Large Language Models 

**Title (ZH)**: 大规模语言模型中检测不安全提示的梯度共现分析 

**Authors**: Jingyuan Yang, Bowen Yan, Rongjun Li, Ziyu Zhou, Xin Chen, Zhiyong Feng, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12411)  

**Abstract**: Unsafe prompts pose significant safety risks to large language models (LLMs). Existing methods for detecting unsafe prompts rely on data-driven fine-tuning to train guardrail models, necessitating significant data and computational resources. In contrast, recent few-shot gradient-based methods emerge, requiring only few safe and unsafe reference prompts. A gradient-based approach identifies unsafe prompts by analyzing consistent patterns of the gradients of safety-critical parameters in LLMs. Although effective, its restriction to directional similarity (cosine similarity) introduces ``directional bias'', limiting its capability to identify unsafe prompts. To overcome this limitation, we introduce GradCoo, a novel gradient co-occurrence analysis method that expands the scope of safety-critical parameter identification to include unsigned gradient similarity, thereby reducing the impact of ``directional bias'' and enhancing the accuracy of unsafe prompt detection. Comprehensive experiments on the widely-used benchmark datasets ToxicChat and XStest demonstrate that our proposed method can achieve state-of-the-art (SOTA) performance compared to existing methods. Moreover, we confirm the generalizability of GradCoo in detecting unsafe prompts across a range of LLM base models with various sizes and origins. 

**Abstract (ZH)**: 不安全的提示对大型语言模型（LLMs）构成了显著的安全风险。现有检测不安全提示的方法依赖于数据驱动的微调来训练护栏模型，这需要大量的数据和计算资源。相比之下，最近出现了一些基于少量样本的梯度方法，仅需要少量的安全和不安全参考提示。基于梯度的方法通过分析大型语言模型（LLMs）中安全关键参数梯度的一致模式来识别不安全提示。尽管这些方法有效，但它们对方向相似性的限制（余弦相似性）导致了“方向偏差”，这限制了它们识别不安全提示的能力。为克服这一限制，我们引入了GradCoo，这是一种新颖的梯度共现分析方法，将安全关键参数识别的范围扩展到包括未加权梯度相似性，从而减少了“方向偏差”的影响并提高了不安全提示检测的准确性。在广泛使用的基准数据集ToxicChat和XStest上进行的全面实验表明，我们提出的方法在与现有方法相比时能实现最先进的（SOTA）性能。此外，我们证实了GradCoo在不同规模和来源的大型语言模型（LLM）基础模型中检测不安全提示的一般性。 

---
# Solving the Cold Start Problem on One's Own as an End User via Preference Transfer 

**Title (ZH)**: 作为最终用户通过偏好转移自我解决冷启动问题 

**Authors**: Ryoma Sato  

**Link**: [PDF](https://arxiv.org/pdf/2502.12398)  

**Abstract**: We propose a new approach that enables end users to directly solve the cold start problem by themselves. The cold start problem is a common issue in recommender systems, and many methods have been proposed to address the problem on the service provider's side. However, when the service provider does not take action, users are left with poor recommendations and no means to improve their experience. We propose an algorithm, Pretender, that allows end users to proactively solve the cold start problem on their own. Pretender does not require any special support from the service provider and can be deployed independently by users. We formulate the problem as minimizing the distance between the source and target distributions and optimize item selection from the target service accordingly. Furthermore, we establish theoretical guarantees for Pretender based on a discrete quadrature problem. We conduct experiments on real-world datasets to demonstrate the effectiveness of Pretender. 

**Abstract (ZH)**: 我们提出了一种新的方法，使最终用户能够直接自行解决冷启动问题。冷启动问题是推荐系统中常见的一个问题，许多方法已经在服务提供商那一端提出了以应对该问题。然而，当服务提供商不采取行动时，用户会收到质量较低的推荐，且无法改善自己的体验。为此，我们提出了一种名为Pretender的算法，允许最终用户主动自行解决冷启动问题。Pretender不需要服务提供商的特别支持，并且可以由用户独立部署。我们将问题形式化为最小化源分布与目标分布之间的距离，并据此优化目标服务中的项目选择。此外，我们根据离散正交问题为Pretender建立了理论保证。我们在实际数据集上进行实验，以证明Pretender的有效性。 

---
# Could AI Leapfrog the Web? Evidence from Teachers in Sierra Leone 

**Title (ZH)**: AI能超越互联网吗？来自塞拉利昂教师的证据 

**Authors**: Daniel Björkegren, Jun Ho Choi, Divya Budihal, Dominic Sobhani, Oliver Garrod, Paul Atherton  

**Link**: [PDF](https://arxiv.org/pdf/2502.12397)  

**Abstract**: Access to digital information is a driver of economic development. But although 85% of sub-Saharan Africa's population is covered by mobile broadband signal, only 37% use the internet, and those who do seldom use the web. We investigate whether AI can bridge this gap by analyzing how 469 teachers use an AI chatbot in Sierra Leone. The chatbot, accessible via a common messaging app, is compared against traditional web search. Teachers use AI more frequently than web search for teaching assistance. Data cost is the most frequently cited reason for low internet usage across Africa. The average web search result consumes 3,107 times more data than an AI response, making AI 87% less expensive than web search. Additionally, only 2% of results for corresponding web searches contain content from Sierra Leone. In blinded evaluations, an independent sample of teachers rate AI responses as more relevant, helpful, and correct than web search results. These findings suggest that AI-driven solutions can cost-effectively bridge information gaps in low-connectivity regions. 

**Abstract (ZH)**: 数字信息的访问是驱动经济发展的一个因素。虽然撒哈拉以南非洲地区的85%人口覆盖了移动宽带信号，但只有37%的人使用互联网，而使用互联网的人群也很少上网。我们通过分析塞拉利昂469名教师使用AI聊天机器人的频率来探讨AI能否弥合这一差距。该聊天机器人可通过一款常见的即时通讯应用访问，并与传统的网络搜索进行了对比。研究表明，教师们比网络搜索更频繁地使用AI进行教学辅助。数据成本是非洲地区互联网使用率低的最常见原因。平均而言，一个网络搜索结果的数据消耗量是AI回复的3,107倍，使得AI的费用比网络搜索低87%。此外，针对相应网络搜索的前2%结果中包含塞拉利昂内容的比例极低。在盲测评估中，一组独立教师样本评价AI回复比网络搜索结果更相关、更实用且更准确。这些发现表明，AI驱动的解决方案可以在低连接区域以成本效益的方式弥合信息缺口。 

---
# Time Series Treatment Effects Analysis with Always-Missing Controls 

**Title (ZH)**: 时间序列中的始终缺失控制变量的治疗效果分析 

**Authors**: Juan Shu, Qiyu Han, George Chen, Xihao Cao, Kangming Luo, Dan Pallotta, Shivam Agrawal, Yuping Lu, Xiaoyu Zhang, Jawad Mansoor, Jyoti Anand  

**Link**: [PDF](https://arxiv.org/pdf/2502.12393)  

**Abstract**: Estimating treatment effects in time series data presents a significant challenge, especially when the control group is always unobservable. For example, in analyzing the effects of Christmas on retail sales, we lack direct observation of what would have occurred in late December without the Christmas impact. To address this, we try to recover the control group in the event period while accounting for confounders and temporal dependencies. Experimental results on the M5 Walmart retail sales data demonstrate robust estimation of the potential outcome of the control group as well as accurate predicted holiday effect. Furthermore, we provided theoretical guarantees for the estimated treatment effect, proving its consistency and asymptotic normality. The proposed methodology is applicable not only to this always-missing control scenario but also in other conventional time series causal inference settings. 

**Abstract (ZH)**: 在时间序列数据中估计处理效应是一项重大挑战，尤其是在控制组始终不可见的情况下。例如，在分析圣诞节对零售销售的影响时，我们缺乏圣诞节影响下12月下旬的直接观测数据。为了解决这一问题，我们试图在事件期内恢复控制组，并同时考虑混杂因素和时间依赖性。实验结果表明，我们在M5沃尔玛零售销售数据集上的研究能够稳健估计控制组的潜在结果，并准确预测节假日效应。此外，我们为估计的处理效应提供了理论保证，证明了其一致性及渐近正态性。所提出的方法不仅适用于这种控制组始终缺失的场景，也适用于其他常规的时间序列因果推断情境。 

---
# Bridging the Data Gap in AI Reliability Research and Establishing DR-AIR, a Comprehensive Data Repository for AI Reliability 

**Title (ZH)**: 填补人工智能可靠性研究中的数据缺口，并建立DR-AIR，一个全面的人工智能可靠性数据资源库 

**Authors**: Simin Zheng, Jared M. Clark, Fatemeh Salboukh, Priscila Silva, Karen da Mata, Fenglian Pan, Jie Min, Jiayi Lian, Caleb B. King, Lance Fiondella, Jian Liu, Xinwei Deng, Yili Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.12386)  

**Abstract**: Artificial intelligence (AI) technology and systems have been advancing rapidly. However, ensuring the reliability of these systems is crucial for fostering public confidence in their use. This necessitates the modeling and analysis of reliability data specific to AI systems. A major challenge in AI reliability research, particularly for those in academia, is the lack of readily available AI reliability data. To address this gap, this paper focuses on conducting a comprehensive review of available AI reliability data and establishing DR-AIR: a data repository for AI reliability. Specifically, we introduce key measurements and data types for assessing AI reliability, along with the methodologies used to collect these data. We also provide a detailed description of the currently available datasets with illustrative examples. Furthermore, we outline the setup of the DR-AIR repository and demonstrate its practical applications. This repository provides easy access to datasets specifically curated for AI reliability research. We believe these efforts will significantly benefit the AI research community by facilitating access to valuable reliability data and promoting collaboration across various academic domains within AI. We conclude our paper with a call to action, encouraging the research community to contribute and share AI reliability data to further advance this critical field of study. 

**Abstract (ZH)**: 人工智能（AI）技术和系统已经取得了迅速的发展，然而确保这些系统的技术可靠性至关重要，这有助于增强公众对这些技术应用的信心。为此，有必要针对AI系统进行可靠性建模和分析。特别是在学术界，AI可靠性研究的一个主要挑战是没有现成可用的AI可靠性数据。为解决这一问题，本文主要集中于对现有的AI可靠性数据进行全面的回顾，并建立DR-AIR：一个AI可靠性数据仓库。具体而言，我们介绍了用于评估AI可靠性的关键测量指标和数据类型，以及收集这些数据所使用的方法。我们还提供了当前可用数据集的详细描述，并提供了示例。此外，我们概述了DR-AIR仓库的建立，并展示了其实际应用。这个仓库为AI可靠性研究提供了易于访问的数据集，我们相信这些努力将极大地促进AI研究社区的成果共享，促进跨学科的合作。在论文结尾，我们呼吁研究界贡献并分享AI可靠性数据，以进一步推进这一关键领域的研究。 

---
# Hybrid Machine Learning Models for Intrusion Detection in IoT: Leveraging a Real-World IoT Dataset 

**Title (ZH)**: 物联网中入侵检测的混合机器学习模型：利用实际物联网数据集 

**Authors**: Md Ahnaf Akif, Ismail Butun, Andre Williams, Imadeldin Mahgoub  

**Link**: [PDF](https://arxiv.org/pdf/2502.12382)  

**Abstract**: The rapid growth of the Internet of Things (IoT) has revolutionized industries, enabling unprecedented connectivity and functionality. However, this expansion also increases vulnerabilities, exposing IoT networks to increasingly sophisticated cyberattacks. Intrusion Detection Systems (IDS) are crucial for mitigating these threats, and recent advancements in Machine Learning (ML) offer promising avenues for improvement. This research explores a hybrid approach, combining several standalone ML models such as Random Forest (RF), XGBoost, K-Nearest Neighbors (KNN), and AdaBoost, in a voting-based hybrid classifier for effective IoT intrusion detection. This ensemble method leverages the strengths of individual algorithms to enhance accuracy and address challenges related to data complexity and scalability. Using the widely-cited IoT-23 dataset, a prominent benchmark in IoT cybersecurity research, we evaluate our hybrid classifiers for both binary and multi-class intrusion detection problems, ensuring a fair comparison with existing literature. Results demonstrate that our proposed hybrid models, designed for robustness and scalability, outperform standalone approaches in IoT environments. This work contributes to the development of advanced, intelligent IDS frameworks capable of addressing evolving cyber threats. 

**Abstract (ZH)**: 物联网（IoT）的迅速增长已经彻底改变了各行各业，使其具备前所未有的连接性和功能。然而，这种扩展也增加了系统的脆弱性，使得IoT网络更容易受到越来越复杂的网络攻击。入侵检测系统（IDS）对于减轻这些威胁至关重要，而机器学习（ML）的最新进展为此提供了前景广阔的改进途径。本研究探讨了一种混合方法，将多种独立的机器学习模型，如随机森林（RF）、XGBoost、K-最近邻（KNN）和AdaBoost，结合在一个基于投票的混合分类器中，以有效进行IoT入侵检测。该集成方法利用了各个算法的优势，以提高准确性和解决与数据复杂性和可扩展性相关的问题。利用广泛引用的IoT-23数据集，一个在物联网网络安全研究中备受推崇的基准，我们评估了我们的混合分类器在二元和多分类入侵检测问题中的性能，确保与现有文献进行全面比较。研究结果表明，我们提出的设计具有鲁棒性和可扩展性的混合模型在IoT环境中优于单一模型。本研究为开发能够应对不断演变的网络安全威胁的先进、智能IDS框架做出了贡献。 

---
# Soft Robotics for Search and Rescue: Advancements, Challenges, and Future Directions 

**Title (ZH)**: 软体机器人在搜索与救援中的应用：进展、挑战与未来方向 

**Authors**: Abhishek Sebastian  

**Link**: [PDF](https://arxiv.org/pdf/2502.12373)  

**Abstract**: Soft robotics has emerged as a transformative technology in Search and Rescue (SAR) operations, addressing challenges in navigating complex, hazardous environments that often limit traditional rigid robots. This paper critically examines advancements in soft robotic technologies tailored for SAR applications, focusing on their unique capabilities in adaptability, safety, and efficiency. By leveraging bio-inspired designs, flexible materials, and advanced locomotion mechanisms, such as crawling, rolling, and shape morphing, soft robots demonstrate exceptional potential in disaster scenarios. However, significant barriers persist, including material durability, power inefficiency, sensor integration, and control complexity. This comprehensive review highlights the current state of soft robotics in SAR, discusses simulation methodologies and hardware validations, and introduces performance metrics essential for their evaluation. By bridging the gap between theoretical advancements and practical deployment, this study underscores the potential of soft robotic systems to revolutionize SAR missions and advocates for continued interdisciplinary innovation to overcome existing limitations. 

**Abstract (ZH)**: 软体机器人技术已成为搜索与救援（SAR）操作中的变革性技术，能够应对传统刚性机器人在穿越复杂和危险环境时遇到的诸多挑战。本文从适应性、安全性和效率等方面，批判性地探讨了适用于SAR应用的软体机器人技术的最新进展。通过借鉴生物启发的设计、柔性材料以及先进的运动机制（如爬行、滚动和形状变形），软体机器人在灾难场景中展现出卓越的潜力。然而，仍然存在一些重大障碍，包括材料的耐久性、能量效率、传感器集成以及控制复杂性。本文通过全面回顾软体机器人在SAR中的现状、讨论仿真方法和硬件验证，并引入评估性能的关键指标，提出了将理论进步与实际部署相结合的视角。通过缩小理论与实践之间的差距，该研究强调了软体机器人系统革命化SAR任务的潜力，并呼吁继续进行跨学科创新以克服现有限制。 

---
# Factual Inconsistency in Data-to-Text Generation Scales Exponentially with LLM Size: A Statistical Validation 

**Title (ZH)**: 数据到文本生成中的事实不一致性随大规模语言模型规模呈指数级增长：一种统计验证 

**Authors**: Joy Mahapatra, Soumyajit Roy, Utpal Garain  

**Link**: [PDF](https://arxiv.org/pdf/2502.12372)  

**Abstract**: Monitoring factual inconsistency is essential for ensuring trustworthiness in data-to-text generation (D2T). While large language models (LLMs) have demonstrated exceptional performance across various D2T tasks, previous studies on scaling laws have primarily focused on generalization error through power law scaling to LLM size (i.e., the number of model parameters). However, no research has examined the impact of LLM size on factual inconsistency in D2T. In this paper, we investigate how factual inconsistency in D2T scales with LLM size by exploring two scaling laws: power law and exponential scaling. To rigorously evaluate and compare these scaling laws, we employ a statistical validation framework consisting of three key stages: predictive performance estimation, goodness-of-fit assessment, and comparative analysis. For a comprehensive empirical study, we analyze three popular LLM families across five D2T datasets, measuring factual inconsistency inversely using four state-of-the-art consistency metrics. Our findings, based on exhaustive empirical results and validated through our framework, reveal that, contrary to the widely assumed power law scaling, factual inconsistency in D2T follows an exponential scaling with LLM size. 

**Abstract (ZH)**: 确保数据到文本生成（D2T）的可信性对于检测事实不一致是至关重要的。尽管大规模语言模型（LLMs）已经在各种D2T任务中展现了卓越的性能，但以往关于扩展规律的研究主要集中在通过幂律扩展来评估LLM大小（即模型参数数量）上的泛化误差上。然而，尚无研究探讨LLM大小对D2T中事实不一致的影响。本文通过探索两种扩展规律（幂律和指数扩展）来研究事实不一致在LLM大小上的变化。为严格评估和比较这些扩展规律，我们采用了一个包含三个关键阶段的统计验证框架：预测性能估计、拟合优度评估和比较分析。为了进行全面的经验研究，我们在五组D2T数据集上分析了三种流行的LLM家族，并使用四种最先进的一致性度量标准反向测量事实不一致。根据我们框架验证的详尽经验结果得出的结论显示，与广泛假设的幂律扩展不同，D2T中的事实不一致性随着LLM大小呈现指数扩展趋势。 

---
# IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation 

**Title (ZH)**: IMLE策略：通过隐式最大似然估计实现快速且样本高效的感觉运动策略学习 

**Authors**: Krishan Rana, Robert Lee, David Pershouse, Niko Suenderhauf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12371)  

**Abstract**: Recent advances in imitation learning, particularly using generative modelling techniques like diffusion, have enabled policies to capture complex multi-modal action distributions. However, these methods often require large datasets and multiple inference steps for action generation, posing challenges in robotics where the cost for data collection is high and computation resources are limited. To address this, we introduce IMLE Policy, a novel behaviour cloning approach based on Implicit Maximum Likelihood Estimation (IMLE). IMLE Policy excels in low-data regimes, effectively learning from minimal demonstrations and requiring 38\% less data on average to match the performance of baseline methods in learning complex multi-modal behaviours. Its simple generator-based architecture enables single-step action generation, improving inference speed by 97.3\% compared to Diffusion Policy, while outperforming single-step Flow Matching. We validate our approach across diverse manipulation tasks in simulated and real-world environments, showcasing its ability to capture complex behaviours under data constraints. Videos and code are provided on our project page: this https URL. 

**Abstract (ZH)**: 近年来，特别是在使用生成建模技术（如扩散模型）的模仿学习方面取得了重大进展，这些方法使得策略能够捕捉复杂的多模态动作分布。然而，这些方法通常需要大量的数据集和多次推理步骤来进行动作生成，这在机器人研究中提出了挑战，因为在机器人研究中数据收集的代价高昂且计算资源有限。为了解决这个问题，我们引入了IMLE Policy，这是一种基于隐式最大似然估计（IMLE）的创新行为克隆方法。IMLE Policy在数据量有限的情况下表现出色，能够从最少的示范中有效学习，并且平均只需要38%的数据就能达到基线方法在学习复杂多模态行为方面的性能。其简单生成器架构允许一步生成动作，相较于扩散策略提高了97.3%的推理速度，同时在一步生成匹配方面也优于其他单步方法。我们通过仿真和真实环境中的多种操作任务验证了这种方法，展示了在数据受限情况下捕捉复杂行为的能力。更多信息和代码请参阅我们的项目页面：this https URL。 

---
# Classifiers of Data Sharing Statements in Clinical Trial Records 

**Title (ZH)**: 临床试验记录中数据共享声明分类器的研究 

**Authors**: Saber Jelodari Mamaghani, Cosima Strantz, Dennis Toddenroth  

**Link**: [PDF](https://arxiv.org/pdf/2502.12362)  

**Abstract**: Digital individual participant data (IPD) from clinical trials are increasingly distributed for potential scientific reuse. The identification of available IPD, however, requires interpretations of textual data-sharing statements (DSS) in large databases. Recent advancements in computational linguistics include pre-trained language models that promise to simplify the implementation of effective classifiers based on textual inputs. In a subset of 5,000 textual DSS from this http URL, we evaluate how well classifiers based on domain-specific pre-trained language models reproduce original availability categories as well as manually annotated labels. Typical metrics indicate that classifiers that predicted manual annotations outperformed those that learned to output the original availability categories. This suggests that the textual DSS descriptions contain applicable information that the availability categories do not, and that such classifiers could thus aid the automatic identification of available IPD in large trial databases. 

**Abstract (ZH)**: 来自临床试验的数字个体参与者数据（IPD）越来越多地被分发以便潜在的科学再利用。然而，识别可用的IPD需要对大型数据库中的文本数据共享声明（DSS）进行解释。近期计算语言学的进步包括预训练语言模型，这些模型有望简化基于文本输入的有效分类器的实现过程。在该项目中，我们从<请填写网址>的数据集中评估了基于领域特定预训练语言模型的分类器在重现原始可用性类别以及手动标注标签方面的表现。典型的度量标准表明，预测手动标注的分类器表现优于那些学习输出原始可用性类别的分类器。这表明，文本DSS描述中包含了一些适用于识别可用IPD的信息，而这些信息在可用性类别中并未体现，因此这样的分类器可以帮助自动识别大规模试验数据库中的可用IPD。 

---
# Detecting Systematic Weaknesses in Vision Models along Predefined Human-Understandable Dimensions 

**Title (ZH)**: 沿预定义的人类可理解维度检测视觉模型中的系统性弱点 

**Authors**: Sujan Sai Gannamaneni, Rohil Prakash Rao, Michael Mock, Maram Akila, Stefan Wrobel  

**Link**: [PDF](https://arxiv.org/pdf/2502.12360)  

**Abstract**: Studying systematic weaknesses of DNNs has gained prominence in the last few years with the rising focus on building safe AI systems. Slice discovery methods (SDMs) are prominent algorithmic approaches for finding such systematic weaknesses. They identify top-k semantically coherent slices/subsets of data where a DNN-under-test has low performance. For being directly useful, e.g., as evidences in a safety argumentation, slices should be aligned with human-understandable (safety-relevant) dimensions, which, for example, are defined by safety and domain experts as parts of the operational design domain (ODD). While straightforward for structured data, the lack of semantic metadata makes these investigations challenging for unstructured data. Therefore, we propose a complete workflow which combines contemporary foundation models with algorithms for combinatorial search that consider structured data and DNN errors for finding systematic weaknesses in images. In contrast to existing approaches, ours identifies weak slices that are in line with predefined human-understandable dimensions. As the workflow includes foundation models, its intermediate and final results may not always be exact. Therefore, we build into our workflow an approach to address the impact of noisy metadata. We evaluate our approach w.r.t. its quality on four popular computer vision datasets, including autonomous driving datasets like Cityscapes, BDD100k, and RailSem19, while using multiple state-of-the-art models as DNNs-under-test. 

**Abstract (ZH)**: 近年来，随着对安全人工智能系统关注的增加，研究深度神经网络（DNN）系统的系统性弱点变得日益重要。切片发现方法（SDMs）是寻找此类系统性弱点的突出算法方法。它们识别出待测试DNN在其中表现较差的前k个语义一致的数据切片/子集。为了直接具有实用性，例如作为安全论证中的证据，切片应该与人类可理解（安全性相关）的维度对齐，例如由安全和领域专家定义的运营设计域（ODD）的一部分。对于结构化数据来说，这相对直接，但对于非结构化数据来说，缺乏语义元数据使得调查变得具有挑战性。因此，我们提出了一种完整的工作流程，该流程结合了当代的基础模型和针对结构化数据和DNN错误进行组合搜索的算法，以在图像中找到系统性弱点。与现有方法不同，我们识别出与预定义的人类可理解维度相符的弱点切片。由于该工作流程包括基础模型，其中间和最终结果可能并不总是精确的。因此，我们将在该工作流程中融入一种方法来应对噪声元数据的影响。我们使用多个先进的计算机视觉数据集进行评估，包括自动驾驶相关的Cityscapes、BDD100k和RailSem19数据集，并使用多种最先进的模型作为待测试的DNN。 

---
# Human-centered explanation does not fit all: The interplay of sociotechnical, cognitive, and individual factors in the effect AI explanations in algorithmic decision-making 

**Title (ZH)**: 以人为本的解释并不适用一切情况：社会技术、认知和个体因素在AI解释对算法决策影响中的相互作用 

**Authors**: Yongsu Ahn, Yu-Run Lin, Malihe Alikhani, Eunjeong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2502.12354)  

**Abstract**: Recent XAI studies have investigated what constitutes a \textit{good} explanation in AI-assisted decision-making. Despite the widely accepted human-friendly properties of explanations, such as contrastive and selective, existing studies have yielded inconsistent findings. To address these gaps, our study focuses on the cognitive dimensions of explanation evaluation, by evaluating six explanations with different contrastive strategies and information selectivity and scrutinizing factors behind their valuation process. Our analysis results find that contrastive explanations are not the most preferable or understandable in general; Rather, different contrastive and selective explanations were appreciated to a different extent based on who they are, when, how, and what to explain -- with different level of cognitive load and engagement and sociotechnical contexts. Given these findings, we call for a nuanced view of explanation strategies, with implications for designing AI interfaces to accommodate individual and contextual differences in AI-assisted decision-making. 

**Abstract (ZH)**: 近年来，关于AI辅助决策中的可解释性（XAI）研究探讨了什么是“良好”的解释。尽管解释的人类友好特性，如对比性和选择性，得到了广泛认可，但现有研究在此方面仍存在不一致的结论。为了解决这些差距，本研究聚焦于解释评估的认知维度，通过评估六种具有不同对比策略和信息选择性的解释，并严格审查其评价过程中的各种因素。我们的分析结果显示，对比性解释并不总是最理想或最容易理解的；相反，不同的对比性和选择性解释根据解释的对象、时间、方式和内容，以及不同的认知负担和参与程度和社会技术背景，得到了不同程度的青睐。鉴于这些发现，我们呼吁对解释策略采取更为细致的观点，并在AI辅助决策中设计相应的界面，以适应个体和情境差异。 

---
# Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs 

**Title (ZH)**: 通过注意力图实现图变换器的机理可解释性 

**Authors**: Batu El, Deepro Choudhury, Pietro Liò, Chaitanya K. Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12352)  

**Abstract**: We introduce Attention Graphs, a new tool for mechanistic interpretability of Graph Neural Networks (GNNs) and Graph Transformers based on the mathematical equivalence between message passing in GNNs and the self-attention mechanism in Transformers. Attention Graphs aggregate attention matrices across Transformer layers and heads to describe how information flows among input nodes. Through experiments on homophilous and heterophilous node classification tasks, we analyze Attention Graphs from a network science perspective and find that: (1) When Graph Transformers are allowed to learn the optimal graph structure using all-to-all attention among input nodes, the Attention Graphs learned by the model do not tend to correlate with the input/original graph structure; and (2) For heterophilous graphs, different Graph Transformer variants can achieve similar performance while utilising distinct information flow patterns. Open source code: this https URL 

**Abstract (ZH)**: 我们引入了注意力图（Attention Graphs），这是一种基于图形神经网络（GNNs）中的消息传递与变压器（Transformers）中的自注意力机制之间数学等价关系的新工具，用于图形神经网络和图形变换器的机制可解释性。注意力图通过聚合变换器层和头部中的注意力矩阵，描述输入节点之间的信息流动。通过在同质性和异质性节点分类任务中进行实验，并从网络科学的角度分析注意力图，我们发现：（1）当允许变换器学习输入节点之间的全连接注意力以找到最优图结构时，模型学习到的注意力图与输入的原始图结构的相关性并不显著；（2）对于异质性图，不同变体的变换器可以在利用不同的信息流动模式的同时实现相似的性能。开源代码：[此处链接] 

---
# QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models 

**Title (ZH)**: QuZO: 量化零阶微调方法应用于大型语言模型 

**Authors**: Jiajun Zhou, Yifan Yang, Kai Zhen, Ziyue Liu, Yequan Zhao, Ershad Banijamali, Athanasios Mouchtaris, Ngai Wong, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12346)  

**Abstract**: Language Models (LLMs) are often quantized to lower precision to reduce the memory cost and latency in inference. However, quantization often degrades model performance, thus fine-tuning is required for various down-stream tasks. Traditional fine-tuning methods such as stochastic gradient descent and Adam optimization require backpropagation, which are error-prone in the low-precision settings. To overcome these limitations, we propose the Quantized Zeroth-Order (QuZO) framework, specifically designed for fine-tuning LLMs through low-precision (e.g., 4- or 8-bit) forward passes. Our method can avoid the error-prone low-precision straight-through estimator, and utilizes optimized stochastic rounding to mitigate the increased bias. QuZO simplifies the training process, while achieving results comparable to first-order methods in ${\rm FP}8$ and superior accuracy in ${\rm INT}8$ and ${\rm INT}4$ training. Experiments demonstrate that low-bit training QuZO achieves performance comparable to MeZO optimization on GLUE, Multi-Choice, and Generation tasks, while reducing memory cost by $2.94 \times$ in LLaMA2-7B fine-tuning compared to quantized first-order methods. 

**Abstract (ZH)**: 语言模型（大型语言模型，LLMs）通常被量化到较低的精度（例如32位浮点数降为8位或更低），以减少推理时的内存成本和延迟。然而，量化往往会降低模型性能，因此需要进行下游任务的微调。传统的微调方法，如随机梯度下降和Adam优化，需要进行反向传播，而在低精度设置中容易出错。为了克服这些限制，我们提出了一种名为Quantized Zeroth-Order（QuZO）的框架，专门用于通过低精度正向传播（例如4位或8位）来微调大型语言模型。该方法可以避免使用低精度下的不可靠直通估计器，并利用优化的随机化舍入来缓解增加的偏差。QuZO 简化了训练过程，同时在 ${\rm FP}8$ 中达到了与一阶方法相当的结果，并在 ${\rm INT}8$ 和 ${\rm INT}4$ 训练中取得了更高的准确性。实验结果表明，低位数训练的QuZO 在GLUE、多选题和生成任务上的表现与MeZO优化方法相当，而在对7B参数的LLaMA2进行微调时，相比于量化的一阶方法，内存成本降低了约 $2.94 \times$。 

---
# A Novel Unified Parametric Assumption for Nonconvex Optimization 

**Title (ZH)**: 一种新的统一参数假设方法用于非凸优化 

**Authors**: Artem Riabinin, Ahmed Khaled, Peter Richtárik  

**Link**: [PDF](https://arxiv.org/pdf/2502.12329)  

**Abstract**: Nonconvex optimization is central to modern machine learning, but the general framework of nonconvex optimization yields weak convergence guarantees that are too pessimistic compared to practice. On the other hand, while convexity enables efficient optimization, it is of limited applicability to many practical problems. To bridge this gap and better understand the practical success of optimization algorithms in nonconvex settings, we introduce a novel unified parametric assumption. Our assumption is general enough to encompass a broad class of nonconvex functions while also being specific enough to enable the derivation of a unified convergence theorem for gradient-based methods. Notably, by tuning the parameters of our assumption, we demonstrate its versatility in recovering several existing function classes as special cases and in identifying functions amenable to efficient optimization. We derive our convergence theorem for both deterministic and stochastic optimization, and conduct experiments to verify that our assumption can hold practically over optimization trajectories. 

**Abstract (ZH)**: 非凸优化是现代机器学习的核心，但非凸优化的通用框架提供的收敛性保证过于悲观，远不如实际应用中的表现。另一方面，尽管凸性可以使优化更有效率，但它在许多实际问题中的应用是有限的。为了弥合这一差距，并更好地理解在非凸设置下优化算法的实际成功，我们引入了一种新颖的统一参数假设。我们的假设既足够广泛，能够涵盖广泛的非凸函数类，又足够具体，可以推导出基于梯度方法的统一收敛定理。值得注意的是，通过调整我们假设的参数，我们展示了其灵活性，能够在某些情况下恢复多个现有的函数类，并识别出可高效优化的函数。我们分别针对确定性和随机优化推导出了收敛定理，并进行了实验验证，表明我们的假设在优化路径上具有实际适用性。 

---
# LM Agents for Coordinating Multi-User Information Gathering 

**Title (ZH)**: 用于协调多用户信息收集的LM代理模型 

**Authors**: Harsh Jhamtani, Jacob Andreas, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2502.12328)  

**Abstract**: This paper introduces PeopleJoin, a benchmark for evaluating LM-mediated collaborative problem solving. Given a user request, PeopleJoin agents must identify teammates who might be able to assist, converse with these teammates to gather information, and finally compile a useful answer or summary for the original user. PeopleJoin comprises two evaluation domains: PeopleJoin-QA, focused on questions about tabular data, and PeopleJoin-DocCreation, focused on document creation tasks. The two domains are adapted from existing NLP benchmarks for database question answering and multi-document summarization; here, however, the information needed to complete these tasks is distributed across synthetic ``organizations'' of 2--20 users, simulating natural multi-user collaboration scenarios. We implemented several popular LM agent architectures, evaluating their accuracy and efficiency at completing tasks, and highlight new research questions that can be studied using PeopleJoin. 

**Abstract (ZH)**: 本文介绍了PeopleJoin，这是一个用于评估语言模型（LM）介导的协同问题解决的基准。给定用户请求，PeopleJoin代理必须识别可能能够提供帮助的队友，与这些队友进行交流以收集信息，最后为原用户提供有用的答案或总结。PeopleJoin包含两个评估领域：PeopleJoin-QA（专注于表格数据的问题）和PeopleJoin-DocCreation（专注于文档创建任务）。这两个领域是从现有的数据库问答和多文档总结的NLP基准中改编而来的；然而，在这里，完成这些任务所需的信息分布在2至20名用户的合成“组织”中，模拟了自然的多用户协作场景。我们实现了几种流行的LM代理架构，评估了它们在完成任务方面的准确性和效率，并指出了可以通过PeopleJoin研究的新研究问题。 

---
# Learning Plasma Dynamics and Robust Rampdown Trajectories with Predict-First Experiments at TCV 

**Title (ZH)**: Using Predict-First Experiments to Learn Plasma Dynamics and Robust Rampdown Trajectories at TCV 

**Authors**: Allen M. Wang, Alessandro Pau, Cristina Rea, Oswin So, Charles Dawson, Olivier Sauter, Mark D. Boyer, Anna Vu, Cristian Galperti, Chuchu Fan, Antoine Merle, Yoeri Poels, Cristina Venturini, Stefano Marchioni, TCV Team  

**Link**: [PDF](https://arxiv.org/pdf/2502.12327)  

**Abstract**: The rampdown in tokamak operations is a difficult to simulate phase during which the plasma is often pushed towards multiple instability limits. To address this challenge, and reduce the risk of disrupting operations, we leverage recent advances in Scientific Machine Learning (SciML) to develop a neural state-space model (NSSM) that predicts plasma dynamics during Tokamak à Configuration Variable (TCV) rampdowns. By integrating simple physics structure and data-driven models, the NSSM efficiently learns plasma dynamics during the rampdown from a modest dataset of 311 pulses with only five pulses in the reactor relevant high performance regime. The NSSM is parallelized across uncertainties, and reinforcement learning (RL) is applied to design trajectories that avoid multiple instability limits with high probability. Experiments at TCV ramping down high performance plasmas show statistically significant improvements in current and energy at plasma termination, with improvements in speed through continuous re-training. A predict-first experiment, increasing plasma current by 20\% from baseline, demonstrates the NSSM's ability to make small extrapolations with sufficient accuracy to design trajectories that successfully terminate the pulse. The developed approach paves the way for designing tokamak controls with robustness to considerable uncertainty, and demonstrates the relevance of the SciML approach to learning plasma dynamics for rapidly developing robust trajectories and controls during the incremental campaigns of upcoming burning plasma tokamaks. 

**Abstract (ZH)**: 等离子体熄灭是托卡马克操作中的一个难以模拟的阶段，此时等离子体往往会接近多种不稳定性极限。为解决这一挑战并降低中断操作的风险，我们利用科学机器学习（SciML）的最新进展开发了一种神经状态空间模型（NSSM），该模型可以预测托卡马克可变配置（TCV）熄灭期间的等离子体动力学。通过结合简单的物理结构和数据驱动模型，NSSM以高效的方式仅从包含311个脉冲（其中5个脉冲与反应堆相关的高-performance模式相关）的适度数据集中学习熄灭期间的等离子体动力学。NSSM在不确定性上实现并行化，通过强化学习（RL）设计避免多种不稳定性极限的高概率轨迹。在TCV熄灭高-performance等离子体的实验中，通过连续训练显著改善了放电终止时的电流和能量。一个前测实验，将等离子体电流从基本值增加20%，进一步展示了NSSM能够进行小的外推，并设计成功终止脉冲的轨迹的能力。所开发的方法为设计具有较强鲁棒性的托卡马克控制提供了可能，并展示了SciML方法在学习等离子体动力学方面的相关性，特别是在逐步推进的实验中快速生成鲁棒轨迹和控制。该方法为即将到来的燃烧等离子体托卡马克的逐步实验设计有效的控制策略奠定了基础。 

---
# Warmup Generations: A Task-Agnostic Approach for Guiding Sequence-to-Sequence Learning with Unsupervised Initial State Generation 

**Title (ZH)**: warm-up 生成：一种任务无相关的引导方法，用于利用无监督初始状态生成进行序列到序列学习 

**Authors**: Senyu Li, Zipeng Sun, Jiayi Wang, Xue Liu, Pontus Stenetorp, Siva Reddy, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12304)  

**Abstract**: Traditional supervised fine-tuning (SFT) strategies for sequence-to-sequence tasks often train models to directly generate the target output. Recent work has shown that guiding models with intermediate steps, such as keywords, outlines, or reasoning chains, can significantly improve performance, coherence, and interpretability. However, these methods often depend on predefined intermediate formats and annotated data, limiting their scalability and generalizability. In this work, we introduce a task-agnostic framework that enables models to generate intermediate "warmup" sequences. These warmup sequences, serving as an initial state for subsequent generation, are optimized to enhance the probability of generating the target sequence without relying on external supervision or human-designed structures. Drawing inspiration from reinforcement learning principles, our method iteratively refines these intermediate steps to maximize their contribution to the final output, similar to reward-driven optimization in reinforcement learning with human feedback. Experimental results across tasks such as translation, summarization, and multi-choice question answering for logical reasoning show that our approach outperforms traditional SFT methods, and offers a scalable and flexible solution for sequence-to-sequence tasks. 

**Abstract (ZH)**: 传统监督微调（SFT）策略在序列到序列任务中通常直接训练模型生成目标输出。最近的研究表明，用中间步骤（如关键词、概要或推理链）来指导模型可以显著提高性能、连贯性和可解释性。然而，这些方法往往依赖预定义的中间格式和标注数据，限制了它们的可扩展性和通用性。在本文中，我们提出了一种任务无关的框架，使模型能够生成中间的“暖启”序列。这些“暖启”序列作为后续生成的初始状态，通过优化来提高生成目标序列的概率，而不依赖于外部监督或人工设计的结构。借鉴强化学习的原则，我们的方法迭代地细化这些中间步骤，以最大化其对最终输出的贡献，类似于受到人类反馈指导的强化学习中的奖励最大化优化。在翻译、摘要生成和逻辑推理的多选题回答等任务上的实验结果表明，我们的方法优于传统的SFT方法，并提供了一种适用于序列到序列任务的可扩展和灵活的解决方案。 

---
# Connecting Large Language Model Agent to High Performance Computing Resource 

**Title (ZH)**: 将大型语言模型代理与高性能计算资源连接起来 

**Authors**: Heng Ma, Alexander Brace, Carlo Siebenschuh, Greg Pauloski, Ian Foster, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12280)  

**Abstract**: The Large Language Model agent workflow enables the LLM to invoke tool functions to increase the performance on specific scientific domain questions. To tackle large scale of scientific research, it requires access to computing resource and parallel computing setup. In this work, we implemented Parsl to the LangChain/LangGraph tool call setup, to bridge the gap between the LLM agent to the computing resource. Two tool call implementations were set up and tested on both local workstation and HPC environment on Polaris/ALCF. The first implementation with Parsl-enabled LangChain tool node queues the tool functions concurrently to the Parsl workers for parallel execution. The second configuration is implemented by converting the tool functions into Parsl ensemble functions, and is more suitable for large task on super computer environment. The LLM agent workflow was prompted to run molecular dynamics simulations, with different protein structure and simulation conditions. These results showed the LLM agent tools were managed and executed concurrently by Parsl on the available computing resource. 

**Abstract (ZH)**: 大规模语言模型代理的工作流能够促使LLM调用工具函数以在特定科学领域问题上提升性能。为了应对大规模科学研究的需求，需要访问计算资源并设置并行计算环境。在本工作中，我们将在LangChain/LangGraph工具调用设置中实现Parsl，以弥合LLM代理与计算资源之间的差距。我们在Polaris/ALCF的本地工作站和超级计算机环境中分别设置了两种工具调用实现并进行了测试。第一种实现利用Parsl启用的LangChain工具节点，将工具函数并发提交给Parsl工作者，以实现并行执行。第二种配置将工具函数转换为Parsl集合函数，更适合在超级计算机环境中处理大规模任务。LLM代理工作流被用来执行分子动力学模拟，其中包括不同的蛋白质结构和模拟条件。这些结果表明，Parsl能够有效管理和在可用计算资源上并行执行LLM代理工具。 

---
# Towards Practical First-Order Model Counting 

**Title (ZH)**: 面向实用的第一范式模型计数 

**Authors**: Ananth K. Kidambi, Guramrit Singh, Paulius Dilkas, Kuldeep S. Meel  

**Link**: [PDF](https://arxiv.org/pdf/2502.12278)  

**Abstract**: First-order model counting (FOMC) is the problem of counting the number of models of a sentence in first-order logic. Since lifted inference techniques rely on reductions to variants of FOMC, the design of scalable methods for FOMC has attracted attention from both theoreticians and practitioners over the past decade. Recently, a new approach based on first-order knowledge compilation was proposed. This approach, called Crane, instead of simply providing the final count, generates definitions of (possibly recursive) functions that can be evaluated with different arguments to compute the model count for any domain size. However, this approach is not fully automated, as it requires manual evaluation of the constructed functions. The primary contribution of this work is a fully automated compilation algorithm, called Gantry, which transforms the function definitions into C++ code equipped with arbitrary-precision arithmetic. These additions allow the new FOMC algorithm to scale to domain sizes over 500,000 times larger than the current state of the art, as demonstrated through experimental results. 

**Abstract (ZH)**: 一阶模型计数（FOMC）是指对一阶逻辑中句子的模型数量进行计数的问题。由于提升推理技术依赖于FOMC的不同变体的归约，因此近十年来，对可扩展的FOMC方法的设计引起了理论学家和实践者的广泛关注。最近，提出了一种基于一阶知识编译的新方法。该方法被称为Crane，在计算最终模型计数的同时，生成了可能递归的函数定义，这些函数可以根据不同的参数计算任何领域规模的模型计数。然而，这种方法并不是完全自动化的，因为它需要手动评估生成的函数。本文的主要贡献是一种完全自动化的编译算法，称为Gantry，该算法将函数定义转换为配备了任意精度算术的C++代码。这些新增功能使新的FOMC算法能够扩展到当前最先进的算法规模的50多万倍，通过实验结果得到了验证。 

---
# Learning to Reason at the Frontier of Learnability 

**Title (ZH)**: 学习推理能力的边界 

**Authors**: Thomas Foster, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2502.12272)  

**Abstract**: Reinforcement learning is now widely adopted as the final stage of large language model training, especially for reasoning-style tasks such as maths problems. Typically, models attempt each question many times during a single training step and attempt to learn from their successes and failures. However, we demonstrate that throughout training with two popular algorithms (PPO and VinePPO) on two widely used datasets, many questions are either solved by all attempts - meaning they are already learned - or by none - providing no meaningful training signal. To address this, we adapt a method from the reinforcement learning literature - sampling for learnability - and apply it to the reinforcement learning stage of LLM training. Our curriculum prioritises questions with high variance of success, i.e. those where the agent sometimes succeeds, but not always. Our findings demonstrate that this curriculum consistently boosts training performance across multiple algorithms and datasets, paving the way for more efficient and effective reinforcement learning in LLMs. 

**Abstract (ZH)**: 强化学习现在被广泛应用于大型语言模型训练的最终阶段，尤其是在解决数学问题等推理任务时。通常，模型在每次训练步骤中会尝试多次回答每个问题，并试图从成功和失败中学习。然而，我们通过在两种广为使用的算法（PPO和VinePPO）上对两种广泛使用的数据集进行训练，发现许多问题要么在所有尝试中都被正确解决（意味着已经学会了解决方法），要么在所有尝试中都无法解决（提供不了有意义的学习信号）。为了应对这一问题，我们借鉴了强化学习文献中的采样方法——可学习性采样，并将其应用于大型语言模型训练中的强化学习阶段。我们的课程设置优先选择成功率具有高变异性的题目，即那些代理有时能成功但不总是成功的题目。我们的研究结果表明，这种课程设置能够一致地提升多种算法和数据集上的训练性能，为更高效和有效的大型语言模型强化学习铺平了道路。 

---
# NeuroStrata: Harnessing Neurosymbolic Paradigms for Improved Design, Testability, and Verifiability of Autonomous CPS 

**Title (ZH)**: NeuroStrata：利用神经符号范式提高自主 CPS 的设计、可测试性和可验证性 

**Authors**: Xi Zheng, Ziyang Li, Ivan Ruchkin, Ruzica Piskac, Miroslav Pajic  

**Link**: [PDF](https://arxiv.org/pdf/2502.12267)  

**Abstract**: Autonomous cyber-physical systems (CPSs) leverage AI for perception, planning, and control but face trust and safety certification challenges due to inherent uncertainties. The neurosymbolic paradigm replaces stochastic layers with interpretable symbolic AI, enabling determinism. While promising, challenges like multisensor fusion, adaptability, and verification remain. This paper introduces NeuroStrata, a neurosymbolic framework to enhance the testing and verification of autonomous CPS. We outline its key components, present early results, and detail future plans. 

**Abstract (ZH)**: 本文将以下论文内容或标题翻译成中文，同时保持学术规范：

自主物理-信息（Cyber-Physical Systems, CPSs）系统利用人工智能进行感知、规划和控制，但由于固有的不确定性，面临着信任和安全性认证的挑战。神经符号范式用可解释的符号人工智能取代概率层，从而实现确定性。尽管具有潜力，但仍面临多传感器融合、灵活性和验证等方面的挑战。本文介绍了一种名为NeuroStrata的神经符号框架，旨在增强自主CPS的测试和验证。我们概述了其主要组成部分，展示了初步结果，并详细说明了未来的研究计划。 

---
# Identifying the Best Transition Law 

**Title (ZH)**: 确定最佳转换规律 

**Authors**: Mehrasa Ahmadipour, élise Crepon, Aurélien Garivier  

**Link**: [PDF](https://arxiv.org/pdf/2502.12227)  

**Abstract**: Motivated by recursive learning in Markov Decision Processes, this paper studies best-arm identification in bandit problems where each arm's reward is drawn from a multinomial distribution with a known support. We compare the performance { reached by strategies including notably LUCB without and with use of this knowledge. } In the first case, we use classical non-parametric approaches for the confidence intervals. In the second case, where a probability distribution is to be estimated, we first use classical deviation bounds (Hoeffding and Bernstein) on each dimension independently, and then the Empirical Likelihood method (EL-LUCB) on the joint probability vector. The effectiveness of these methods is demonstrated through simulations on scenarios with varying levels of structural complexity. 

**Abstract (ZH)**: 受马尔科夫决策过程中的递归学习启发，本文研究了每种臂的奖励是从具有已知支撑的多项分布中抽样的臂冲突问题中的最优臂识别问题。我们将比较利用这种知识时策略的性能，包括显著的拉起上置信边界（LUCB）策略及其在使用此知识时的版本。在第一种情况下，我们使用经典的非参数方法构建置信区间。在第二种情况下，由于需要估计概率分布，我们首先在每个维度上独立地使用经典的偏差界（Hoeffding和Bernstein），然后在联合概率向量上使用经验似然方法（EL-LUCB）。通过不同复杂度水平的场景模拟，展示了这些方法的有效性。 

---
# On Creating a Causally Grounded Usable Rating Method for Assessing the Robustness of Foundation Models Supporting Time Series 

**Title (ZH)**: 基于因果grounding的可应用评分方法以评估支持时间序列的基础模型的鲁棒性 

**Authors**: Kausik Lakkaraju, Rachneet Kaur, Parisa Zehtabi, Sunandita Patra, Siva Likitha Valluru, Zhen Zeng, Biplav Srivastava, Marco Valtorta  

**Link**: [PDF](https://arxiv.org/pdf/2502.12226)  

**Abstract**: Foundation Models (FMs) have improved time series forecasting in various sectors, such as finance, but their vulnerability to input disturbances can hinder their adoption by stakeholders, such as investors and analysts. To address this, we propose a causally grounded rating framework to study the robustness of Foundational Models for Time Series (FMTS) with respect to input perturbations. We evaluate our approach to the stock price prediction problem, a well-studied problem with easily accessible public data, evaluating six state-of-the-art (some multi-modal) FMTS across six prominent stocks spanning three industries. The ratings proposed by our framework effectively assess the robustness of FMTS and also offer actionable insights for model selection and deployment. Within the scope of our study, we find that (1) multi-modal FMTS exhibit better robustness and accuracy compared to their uni-modal versions and, (2) FMTS pre-trained on time series forecasting task exhibit better robustness and forecasting accuracy compared to general-purpose FMTS pre-trained across diverse settings. Further, to validate our framework's usability, we conduct a user study showcasing FMTS prediction errors along with our computed ratings. The study confirmed that our ratings reduced the difficulty for users in comparing the robustness of different systems. 

**Abstract (ZH)**: 基础模型（FMs）在金融等各个领域的时间序列预测中取得了显著进步，但它们对输入干扰的脆弱性可能阻碍投资者和分析师等利益相关者的采用。为解决这一问题，我们提出了一种因果导向的评级框架，以研究基础模型在时间序列（FMTS）中的鲁棒性对其输入扰动的敏感性。我们将该方法应用于股票价格预测问题，这是一个研究充分且数据易于获取的问题，对六个先进的（部分多元模态）FMTS在三个不同行业的六种代表性股票上进行了评估。我们框架提出的评级有效地评估了FMTS的鲁棒性，并提供了一些建设性的见解，有助于模型选择和部署。在本研究范围内，我们发现：（1）多模态FMTS在鲁棒性和准确性方面优于其单模态版本；（2）基于时间序列预测任务预训练的FMTS在鲁棒性和预测准确性方面优于通用场景下预训练的FMTS。此外，为了验证我们框架的实用性，我们进行了一项用户研究，展示了FMTS的预测误差以及我们计算的评级。研究结果表明，我们的评级有助于用户更轻松地比较不同系统的鲁棒性。 

---
# Subjective Logic Encodings 

**Title (ZH)**: 主观逻辑编码 

**Authors**: Jake Vasilakes  

**Link**: [PDF](https://arxiv.org/pdf/2502.12225)  

**Abstract**: Many existing approaches for learning from labeled data assume the existence of gold-standard labels. According to these approaches, inter-annotator disagreement is seen as noise to be removed, either through refinement of annotation guidelines, label adjudication, or label filtering. However, annotator disagreement can rarely be totally eradicated, especially on more subjective tasks such as sentiment analysis or hate speech detection where disagreement is natural. Therefore, a new approach to learning from labeled data, called data perspectivism, seeks to leverage inter-annotator disagreement to learn models that stay true to the inherent uncertainty of the task by treating annotations as opinions of the annotators, rather than gold-standard facts. Despite this conceptual grounding, existing methods under data perspectivism are limited to using disagreement as the sole source of annotation uncertainty. To expand the possibilities of data perspectivism, we introduce Subjective Logic Encodings (SLEs), a flexible framework for constructing classification targets that explicitly encodes annotations as opinions of the annotators. Based on Subjective Logic Theory, SLEs encode labels as Dirichlet distributions and provide principled methods for encoding and aggregating various types of annotation uncertainty -- annotator confidence, reliability, and disagreement -- into the targets. We show that SLEs are a generalization of other types of label encodings as well as how to estimate models to predict SLEs using a distribution matching objective. 

**Abstract (ZH)**: 许多现有的基于标记数据的学习方法假设存在黄金标准标签。这些方法认为注释者的分歧是需要去除的噪声，通过改进注释指南、标签裁决或标签过滤来消除。然而，在如情感分析或仇恨言论检测等更为主观的任务上，分歧是自然存在的，几乎无法完全消除。因此，一种新的基于标记数据的学习方法，称为数据透视主义（Data Perspectivism），旨在利用注释者的分歧来学习能够反映任务固有不确定性的模型，将注释视为注释者的观点，而不是黄金标准的事实。尽管具有这种概念基础，现有的数据透视主义方法仅限于使用分歧作为注释不确定性唯一的来源。为了扩大数据透视主义的应用范围，我们引入了主观逻辑编码（Subjective Logic Encodings, SLEs）框架，该框架是一种灵活的框架，用于显式地将注释编码为注释者的观点。基于主观逻辑理论，SLEs将标签编码为狄利克雷分布，并提供了编码和聚合注释者信心、可靠性和分歧等不同类型的不确定性到目标中的原则性方法。我们展示了SLEs如何作为一种标签编码的泛化，并展示了如何使用分布匹配目标来估计预测SLEs的模型。 

---
# IMPACTX: Improving Model Performance by Appropriately predicting CorrecT eXplanations 

**Title (ZH)**: IMPACTX: 通过适当预测正确解释来提高模型性能 

**Authors**: Andrea Apicella, Salvatore Giugliano, Francesco Isgrò, Roberto Prevete  

**Link**: [PDF](https://arxiv.org/pdf/2502.12222)  

**Abstract**: The eXplainable Artificial Intelligence (XAI) research predominantly concentrates to provide explainations about AI model decisions, especially Deep Learning (DL) models. However, there is a growing interest in using XAI techniques to automatically improve the performance of the AI systems themselves.
This paper proposes IMPACTX, a novel approach that leverages XAI as a fully automated attention mechanism, without requiring external knowledge or human feedback. Experimental results show that IMPACTX has improved performance respect to the standalone ML model by integrating an attention mechanism based an XAI method outputs during the model training. Furthermore, IMPACTX directly provides proper feature attribution maps for the model's decisions, without relying on external XAI methods during the inference process.
Our proposal is evaluated using three widely recognized DL models (EfficientNet-B2, MobileNet, and LeNet-5) along with three standard image datasets: CIFAR-10, CIFAR-100, and STL-10. The results show that IMPACTX consistently improves the performance of all the inspected DL models across all evaluated datasets, and it directly provides appropriate explanations for its responses. 

**Abstract (ZH)**: 可解释的人工智能（XAI）研究主要集中在解释AI模型的决策，特别是深度学习（DL）模型的决策。然而，人们越来越感兴趣的是利用XAI技术自动提高AI系统的性能本身。

本文提出了一种名为IMPACTX的新颖方法，该方法利用XAI作为完全自动化的注意力机制，无需外部知识或人工反馈。实验结果表明，IMPACTX在模型训练过程中整合了基于XAI方法输出的注意力机制，从而提高了相对于独立机器学习模型的性能。此外，IMPACTX在推理过程中直接提供了模型决策的适当特征归因图，无需依赖外部XAI方法。

我们的提案使用了三种广泛认可的DL模型（EfficientNet-B2、MobileNet和LeNet-5）以及三个标准的图像数据集：CIFAR-10、CIFAR-100和STL-10进行评估。结果表明，IMPACTX在所有评估数据集上一致地提高了所有检查中的DL模型的性能，并直接为其响应提供了适当解释。 

---
# Optimal Brain Iterative Merging: Mitigating Interference in LLM Merging 

**Title (ZH)**: 最佳大脑迭代合并：减轻大规模语言模型合并中的干扰 

**Authors**: Zhixiang Wang, Zhenyu Mao, Yixuan Qiao, Yunfang Wu, Biye Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12217)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities, but their high computational costs pose challenges for customization. Model merging offers a cost-effective alternative, yet existing methods suffer from interference among parameters, leading to performance degradation. In this work, we propose Optimal Brain Iterative Merging (OBIM), a novel method designed to mitigate both intra-model and inter-model interference. OBIM consists of two key components: (1) A saliency measurement mechanism that evaluates parameter importance based on loss changes induced by individual weight alterations, reducing intra-model interference by preserving only high-saliency parameters. (2) A mutually exclusive iterative merging framework, which incrementally integrates models using a binary mask to avoid direct parameter averaging, thereby mitigating inter-model interference. We validate OBIM through experiments on both Supervised Fine-Tuned (SFT) models and post-pretrained checkpoints. The results show that OBIM significantly outperforms existing merging techniques. Overall, OBIM provides an effective and practical solution for enhancing LLM merging. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现了令人印象深刻的 capability，但其高昂的计算成本给定制化带来了挑战。模型合并提供了一种成本效益较高的替代方案，但现有方法存在参数间的干扰问题，导致性能下降。在此项研究中，我们提出了一种名为 Optimal Brain Iterative Merging (OBIM) 的新型方法，旨在缓解模型内和模型间干扰。OBIM 包含两个关键组件：（1）一个显著性测量机制，该机制根据单个权重更改引起的损失变化评估参数的重要性，并通过保留高显著性参数来减少模型内的干扰；（2）一个互斥迭代合并框架，该框架通过二进制掩码逐步整合模型，以避免直接参数平均，从而减轻模型间的干扰。我们通过对 Supervised Fine-Tuned (SFT) 模型和后预训练检查点进行实验验证了 OBIM 方法。实验结果表明，OBIM 显著优于现有合并技术。总体而言，OBIM 提供了一种有效且实用的方案，用于提升 LLM 合并的性能。 

---
# Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs 

**Title (ZH)**: 标题：自适应稀疏注意力机制：基于聚类和分布拟合的长期上下文大语言模型 

**Authors**: Kan Zhu, Tian Tang, Qinyu Xu, Yile Gu, Zhichen Zeng, Rohan Kadekodi, Liangyu Zhao, Ang Li, Arvind Krishnamurthy, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2502.12216)  

**Abstract**: Long-context models are essential for many applications but face inefficiencies in loading large KV caches during decoding. Prior methods enforce fixed token budgets for sparse attention, assuming a set number of tokens can approximate full attention. However, these methods overlook variations in the importance of attention across heads, layers, and contexts. To address these limitations, we propose Tactic, a sparsity-adaptive and calibration-free sparse attention mechanism that dynamically selects tokens based on their cumulative attention scores rather than a fixed token budget. By setting a target fraction of total attention scores, Tactic ensures that token selection naturally adapts to variations in attention sparsity. To efficiently approximate this selection, Tactic leverages clustering-based sorting and distribution fitting, allowing it to accurately estimate token importance with minimal computational overhead. We show that Tactic outperforms existing sparse attention algorithms, achieving superior accuracy and up to 7.29x decode attention speedup. This improvement translates to an overall 1.58x end-to-end inference speedup, making Tactic a practical and effective solution for long-context LLM inference in accuracy-sensitive applications. 

**Abstract (ZH)**: 长上下文模型在许多应用中是必不可少的，但在解码过程中面临着加载大规模键值缓存的效率问题。现有方法强制执行固定的令牌预算以实现稀疏注意，假设一定数量的令牌可以近似全注意。然而，这些方法忽视了注意在头、层和上下文中的重要性变化。为解决这些限制，我们提出了一种名为Tactic的稀疏注意机制，该机制无需校准且能够根据累积注意分数动态选择令牌，而不是固定令牌预算。通过设定总注意分数的目标占比，Tactic 保证了令牌选择能够自然适应注意稀疏性的变化。为了高效地近似这种选择，Tactic 利用了基于聚类的排序和分布拟合，能够在最小的计算开销下准确估计令牌的重要性。实验证明，Tactic 在现有稀疏注意算法中表现出色，实现了更高的准确性和高达7.29倍的解码注意加速。这种改进进一步转化为端到端推理加速1.58倍，使Tactic 成为在高准确度要求应用中进行长上下文语言模型推理的实际有效解决方案。 

---
# Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities? 

**Title (ZH)**: 重新审视 o1 类模型的测试时缩放能力：它们真具备测试时缩放的能力吗？ 

**Authors**: Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Yunhua Zhou, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12215)  

**Abstract**: The advent of test-time scaling in large language models (LLMs), exemplified by OpenAI's o1 series, has advanced reasoning capabilities by scaling computational resource allocation during inference. While successors like QwQ, Deepseek-R1 (R1) and LIMO replicate these advancements, whether these models truly possess test-time scaling capabilities remains underexplored. This study found that longer CoTs of these o1-like models do not consistently enhance accuracy; in fact, correct solutions are often shorter than incorrect ones for the same questions. Further investigation shows this phenomenon is closely related to models' self-revision capabilities - longer CoTs contain more self-revisions, which often lead to performance degradation. We then compare sequential and parallel scaling strategies on QwQ, R1 and LIMO, finding that parallel scaling achieves better coverage and scalability. Based on these insights, we propose Shortest Majority Vote, a method that combines parallel scaling strategies with CoT length characteristics, significantly improving models' test-time scalability compared to conventional majority voting approaches. 

**Abstract (ZH)**: 测试时缩放在大型语言模型（LLMs）中的出现，以OpenAI的o1系列为例，通过在推理过程中扩展计算资源分配，提高了模型的推理能力。虽然诸如QwQ、Deepseek-R1（R1）和LIMO等后续模型也复制了这些进展，但这些模型是否确实具备测试时缩放能力仍有待进一步探索。研究发现，这些类似o1的模型的较长推理路径并不总是提升准确率；实际上，对于相同问题，正确的答案往往比错误的答案更短。进一步的研究显示，这一现象与模型的自我修订能力密切相关——较长的推理路径包含更多的自我修订，这通常会导致性能下降。我们还对比了在QwQ、R1和LIMO上应用顺序和并行缩放策略的效果，发现并行缩放策略在覆盖范围和可扩展性上表现更好。基于这些发现，我们提出了一种名为“最短多数投票法”的方法，该方法结合了并行缩放策略和推理路径长度的特点，显著提高了模型的测试时可扩展性，相比传统多数投票方法具有显著优势。 

---
# Zero Token-Driven Deep Thinking in LLMs: Unlocking the Full Potential of Existing Parameters via Cyclic Refinement 

**Title (ZH)**: 零令牌驱动的深层思考在大语言模型中的实现：通过循环精炼激发现有参数的全部潜力 

**Authors**: Guanghao Li, Wenhao Jiang, Li Shen, Ming Tang, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12214)  

**Abstract**: Resource limitations often constrain the parameter counts of Large Language Models (LLMs), hindering their performance. While existing methods employ parameter sharing to reuse the same parameter set under fixed budgets, such approaches typically force each layer to assume multiple roles with a predetermined number of iterations, restricting efficiency and adaptability. In this work, we propose the Zero Token Transformer (ZTT), which features a head-tail decoupled parameter cycling method. We disentangle the first (head) and last (tail) layers from parameter cycling and iteratively refine only the intermediate layers. Furthermore, we introduce a Zero-Token Mechanism, an internal architectural component rather than an input token, to guide layer-specific computation. At each cycle, the model retrieves a zero token (with trainable key values) from a Zero-Token Pool, integrating it alongside regular tokens in the attention mechanism. The corresponding attention scores not only reflect each layer's computational importance but also enable dynamic early exits without sacrificing overall model accuracy. Our approach achieves superior performance under tight parameter budgets, effectively reduces computational overhead via early exits, and can be readily applied to fine-tune existing pre-trained models for enhanced efficiency and adaptability. 

**Abstract (ZH)**: 资源限制往往限制了大型语言模型（LLMs）的参数数量，影响其性能。现有的方法通过参数共享在固定预算下重用相同的参数集，但这类方法通常会强制每个层承担多种角色，并预先决定迭代次数，限制了效率和适应性。在本文中，我们提出了零令牌变换器（ZTT），其特征是一种头尾解耦的参数循环方法。我们将第一层（头层）和最后一层（尾层）从参数循环中分离出来，并仅迭代优化中间层。此外，我们引入了零令牌机制，这是一种内部架构组成部分而非输入令牌，用于引导层特定的计算。在每次循环中，模型从零令牌池中检索一个可训练的零令牌，并将其与常规令牌一起集成到注意机制中。相应的注意分数不仅反映了每层计算的重要性，还允许动态提前退出而不会牺牲整体模型的准确性。我们的方法在严格的参数预算下实现了更优的性能，通过提前退出有效减少了计算开销，并可以方便地应用于现有预训练模型的微调，以提高效率和适应性。 

---
# Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting 

**Title (ZH)**: 时空aware趋势与季节性分解网络在交通流量预测中的应用 

**Authors**: Lingxiao Cao, Bin Wang, Guiyuan Jiang, Yanwei Yu, Junyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.12213)  

**Abstract**: Traffic prediction is critical for optimizing travel scheduling and enhancing public safety, yet the complex spatial and temporal dynamics within traffic data present significant challenges for accurate forecasting. In this paper, we introduce a novel model, the Spatiotemporal-aware Trend-Seasonality Decomposition Network (STDN). This model begins by constructing a dynamic graph structure to represent traffic flow and incorporates novel spatio-temporal embeddings to jointly capture global traffic dynamics. The representations learned are further refined by a specially designed trend-seasonality decomposition module, which disentangles the trend-cyclical component and seasonal component for each traffic node at different times within the graph. These components are subsequently processed through an encoder-decoder network to generate the final predictions. Extensive experiments conducted on real-world traffic datasets demonstrate that STDN achieves superior performance with remarkable computation cost. Furthermore, we have released a new traffic dataset named JiNan, which features unique inner-city dynamics, thereby enriching the scenario comprehensiveness in traffic prediction evaluation. 

**Abstract (ZH)**: 交通预测对于优化出行计划和提升公共安全至关重要，但交通数据中的复杂空-时动态给准确预测带来了巨大挑战。本文提出了一种新颖的模型——空-时感知趋势-季节性分解网络（STDN）。该模型首先构建了一个动态图结构来表示交通流，同时引入了新颖的空-时嵌入，以联合捕捉全局交通动态。通过一个特别设计的趋势-季节性分解模块进一步细化所学表示，该模块在图中的不同时间内对每个交通节点的趋势-周期性和季节性成分进行分离。随后，这些成分通过编码-解码网络进行处理以生成最终预测。在真实世界交通数据集上的广泛实验表明，STDN在表现出色的同时还具有显著的计算成本优势。此外，我们还发布了一个新的交通数据集——JiNan数据集，该数据集具有独特的城市内部动态，从而丰富了交通预测评估中的场景全面性。 

---
# Enhancing Frame Detection with Retrieval Augmented Generation 

**Title (ZH)**: 增强框架检测的检索增强生成方法 

**Authors**: Papa Abdou Karim Karou Diallo, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2502.12210)  

**Abstract**: Recent advancements in Natural Language Processing have significantly improved the extraction of structured semantic representations from unstructured text, especially through Frame Semantic Role Labeling (FSRL). Despite this progress, the potential of Retrieval-Augmented Generation (RAG) models for frame detection remains under-explored. In this paper, we present the first RAG-based approach for frame detection called RCIF (Retrieve Candidates and Identify Frames). RCIF is also the first approach to operate without the need for explicit target span and comprises three main stages: (1) generation of frame embeddings from various representations ; (2) retrieval of candidate frames given an input text; and (3) identification of the most suitable frames. We conducted extensive experiments across multiple configurations, including zero-shot, few-shot, and fine-tuning settings. Our results show that our retrieval component significantly reduces the complexity of the task by narrowing the search space thus allowing the frame identifier to refine and complete the set of candidates. Our approach achieves state-of-the-art performance on FrameNet 1.5 and 1.7, demonstrating its robustness in scenarios where only raw text is provided. Furthermore, we leverage the structured representation obtained through this method as a proxy to enhance generalization across lexical variations in the task of translating natural language questions into SPARQL queries. 

**Abstract (ZH)**: 近年来，自然语言处理（NLP）的最新进展显著提高了从非结构化文本中提取结构化语义表示的能力，尤其是通过框架语义角色标注（Frame Semantic Role Labeling, FSRL）。尽管取得了这些进展，Retrieval-Augmented Generation（RAG）模型在框架检测方面的潜力仍被低估。本文介绍了首个基于RAG的方法，即RCIF（Retrieve Candidates and Identify Frames），这是首个无需显式目标片段的方法，包含三个主要阶段：（1）从多种表示生成框架嵌入；（2）根据输入文本检索候选框架；以及（3）识别最合适的框架。我们进行了广泛的实验，包括零样本、少量样本和微调设置。实验结果表明，我们的检索组件通过缩小搜索空间显著降低了任务的复杂性，从而允许框架识别器对候选框架进行细化和补充。我们的方法在FrameNet 1.5和1.7上达到了最先进的性能，证明了在仅提供原始文本的情况下其稳健性。此外，我们利用此方法获得的结构化表示作为代理，增强在将自然语言问题翻译为SPARQL查询任务中跨词汇变体的一般化能力。 

---
# Suboptimal Shapley Value Explanations 

**Title (ZH)**: 次优Shapley值解释 

**Authors**: Xiaolei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12209)  

**Abstract**: Deep Neural Networks (DNNs) have demonstrated strong capacity in supporting a wide variety of applications. Shapley value has emerged as a prominent tool to analyze feature importance to help people understand the inference process of deep neural models. Computing Shapley value function requires choosing a baseline to represent feature's missingness. However, existing random and conditional baselines could negatively influence the explanation. In this paper, by analyzing the suboptimality of different baselines, we identify the problematic baseline where the asymmetric interaction between $\bm{x}'_i$ (the replacement of the faithful influential feature) and other features has significant directional bias toward the model's output, and conclude that $p(y|\bm{x}'_i) = p(y)$ potentially minimizes the asymmetric interaction involving $\bm{x}'_i$. We further generalize the uninformativeness of $\bm{x}'_i$ toward the label space $L$ to avoid estimating $p(y)$ and design a simple uncertainty-based reweighting mechanism to accelerate the computation process. We conduct experiments on various NLP tasks and our quantitative analysis demonstrates the effectiveness of the proposed uncertainty-based reweighting mechanism. Furthermore, by measuring the consistency of explanations generated by explainable methods and human, we highlight the disparity between model inference and human understanding. 

**Abstract (ZH)**: 深度神经网络（DNNs）展示了在广泛应用中强大的支持能力。Shapley值已成为一种突出的工具，用于分析特征的重要性，帮助人们理解深度神经模型的推理过程。计算Shapley值函数需要选择一个基线来表示特征缺失的情况。然而，现有的随机和有条件基线可能会负面影响解释的效果。在本文中，通过分析不同基线的次优性，我们识别出存在问题的基线，其中$\bm{x}'_i$（忠实影响力的特征的替代）与其它特征之间的不对称交互对模型输出有显著的方向性偏向，并得出结论认为$p(y|\bm{x}'_i) = p(y)$可能最小化涉及$\bm{x}'_i$的不对称交互。我们进一步将$\bm{x}'_i$向标签空间$L$的非信息性推广，以避免估计$p(y)$，并设计一个简单的基于不确定性重权重机制来加速计算过程。我们在各种NLP任务上进行了实验，我们的定量分析证明了所提出的基于不确定性重权重机制的有效性。此外，通过测量解释方法生成的解释与人类解释的一致性，我们突显了模型推理与人类理解之间的差距。 

---
# PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN 

**Title (ZH)**: PAR-AdvGAN：通过渐进自回归AdvGAN提高对抗攻击能力 

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12207)  

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original this http URL, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: this https URL 

**Abstract (ZH)**: 深度神经网络在各个领域展现出了非凡的性能。然而，它们容易受到对抗样本的攻击，这可能导致错误的预测。生成对抗网络（GANs）可以通过利用生成器和判别器模型快速生成高质量的对抗样本。由于这两个模块是竞争性和同步训练的，基于GAN的算法如AdvGAN可以生成具有更好迁移性的对抗样本，相比传统方法。然而，扰动的生成通常仅限于单次迭代，这限制了这些样本充分利用方法潜力的能力。为解决这一问题，我们提出了一种名为渐进自回归AdvGAN（PAR-AdvGAN）的新方法。这种方法在渐进生成网络中引入了自回归迭代机制，以生成具有增强攻击能力的对抗样本。我们通过大规模实验全面评估了我们的PAR-AdvGAN方法，结果显示它在各种先进的黑盒对抗攻击方法中表现出更优性能，并且在与原始模型相比方面表现出色。PAR-AdvGAN 显著加快了对抗样本的生成速度，例如在Inception-v3模型上达到了每秒335.5帧的速度，优于基于梯度的可移植性攻击算法。我们的代码可在以下链接获得：this https URL 

---
# Predicting Depression in Screening Interviews from Interactive Multi-Theme Collaboration 

**Title (ZH)**: 从交互式多主题协作中筛查抑郁的预测研究 

**Authors**: Xianbing Zhao, Yiqing Lyu, Di Wang, Buzhou Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12204)  

**Abstract**: Automatic depression detection provides cues for early clinical intervention by clinicians. Clinical interviews for depression detection involve dialogues centered around multiple themes. Existing studies primarily design end-to-end neural network models to capture the hierarchical structure of clinical interview dialogues. However, these methods exhibit defects in modeling the thematic content of clinical interviews: 1) they fail to capture intra-theme and inter-theme correlation explicitly, and 2) they do not allow clinicians to intervene and focus on themes of interest. To address these issues, this paper introduces an interactive depression detection framework. This framework leverages in-context learning techniques to identify themes in clinical interviews and then models both intra-theme and inter-theme correlation. Additionally, it employs AI-driven feedback to simulate the interests of clinicians, enabling interactive adjustment of theme importance. PDIMC achieves absolute improvements of 35\% and 12\% compared to the state-of-the-art on the depression detection dataset DAIC-WOZ, which demonstrates the effectiveness of modeling theme correlation and incorporating interactive external feedback. 

**Abstract (ZH)**: 自动抑郁检测为临床早期干预提供线索。临床抑郁检测对话围绕多个主题展开。现有研究主要设计端到端的神经网络模型来捕捉临床访谈对话的层次结构。然而，这些方法在建模临床访谈的主题内容方面存在缺陷：1）它们未能明确建模主题内的和主题间的相关性，2）它们不允许临床医生介入并关注感兴趣的主题。为解决这些问题，本文引入了一个交互式抑郁检测框架。该框架利用上下文学习技术识别临床访谈中的主题，然后建模主题内的和主题间的相关性。此外，它采用AI驱动的反馈模拟临床医生的兴趣，使主题重要性能够进行交互调整。PDIMC在抑郁检测数据集DAIC-WOZ上绝对提高了35%和12%，这表明建模主题相关性和集成交互外部反馈的有效性。 

---
# An Interpretable Automated Mechanism Design Framework with Large Language Models 

**Title (ZH)**: 具有大型语言模型的可解释自动化机制设计框架 

**Authors**: Jiayuan Liu, Mingyu Guo, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2502.12203)  

**Abstract**: Mechanism design has long been a cornerstone of economic theory, with traditional approaches relying on mathematical derivations. Recently, automated approaches, including differentiable economics with neural networks, have emerged for designing payments and allocations. While both analytical and automated methods have advanced the field, they each face significant weaknesses: mathematical derivations are not automated and often struggle to scale to complex problems, while automated and especially neural-network-based approaches suffer from limited interpretability. To address these challenges, we introduce a novel framework that reformulates mechanism design as a code generation task. Using large language models (LLMs), we generate heuristic mechanisms described in code and evolve them to optimize over some evaluation metrics while ensuring key design criteria (e.g., strategy-proofness) through a problem-specific fixing process. This fixing process ensures any mechanism violating the design criteria is adjusted to satisfy them, albeit with some trade-offs in performance metrics. These trade-offs are factored in during the LLM-based evolution process. The code generation capabilities of LLMs enable the discovery of novel and interpretable solutions, bridging the symbolic logic of mechanism design and the generative power of modern AI. Through rigorous experimentation, we demonstrate that LLM-generated mechanisms achieve competitive performance while offering greater interpretability compared to previous approaches. Notably, our framework can rediscover existing manually designed mechanisms and provide insights into neural-network based solutions through Programming-by-Example. These results highlight the potential of LLMs to not only automate but also enhance the transparency and scalability of mechanism design, ensuring safe deployment of the mechanisms in society. 

**Abstract (ZH)**: 机制设计一直是经济理论的基石，传统的研究方法依赖于数学推导。近年来，自动化方法，包括使用神经网络的可微经济学，已经开始用于设计支付和分配机制。虽然分析方法和自动化方法都推动了领域的发展，但它们各自存在显著的局限性：数学推导方法无法自动化且难以处理复杂问题，而自动化方法尤其是基于神经网络的方法则缺乏可解释性。为了解决这些挑战，我们提出了一种新的框架，将机制设计重新表述为代码生成任务。利用大型语言模型（LLMs），我们生成描述为代码的启发式机制，并通过特定问题的校正过程来优化某些评价指标，同时确保关键设计标准（如策略不变性）。这一校正过程确保了任何违反设计标准的机制都会被调整以满足这些标准，尽管这可能会在性能指标上带来一些权衡。这些权衡因素将在基于LLM的进化过程中加以考虑。LLMs的代码生成能力使我们能够发现新颖且可解释的解决方案，从而弥合机制设计的符号逻辑与现代AI的生成能力之间的鸿沟。通过严格的实验证据，我们展示了基于LLM生成的机制能够达到竞争力的性能，同时提供比以往方法更大的可解释性。值得注意的是，我们的框架可以重新发现现有的手工设计机制，并通过示例编程（Programming-by-Example）为基于神经网络的解决方案提供见解。这些结果突显了LLMs不仅能够自动化机制设计，还能够增强其透明度和可扩展性，确保机制在社会中的安全部署。 

---
# BoT: Breaking Long Thought Processes of o1-like Large Language Models through Backdoor Attack 

**Title (ZH)**: BoT：通过后门攻击打破o1型大型语言模型的长思考过程 

**Authors**: Zihao Zhu, Hongbao Zhang, Mingda Zhang, Ruotong Wang, Guanzong Wu, Ke Xu, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12202)  

**Abstract**: Longer thought, better performance: large language models with deep reasoning capabilities, particularly o1-like models, have demonstrated remarkable performance by generating extensive thought processes during inference. This trade-off reveals a potential vulnerability: adversaries could compromise model performance by forcing immediate responses without thought processes. To this end, in this paper, we introduce a novel attack scenario targeting the long thought processes of o1-like models and propose BoT (Break CoT), which can selectively break intrinsic reasoning mechanisms through backdoor attacks. BoT constructs poisoned datasets with designed triggers and injects backdoor by either supervised fine-tuning or direct preference optimization. When triggered, the model directly generates answers without thought processes, while maintaining normal reasoning capabilities for clean inputs. Extensive experiments on open-source o1-like models, including recent DeepSeek-R1, demonstrate that BoT nearly achieves high attack success rates while maintaining clean accuracy, highlighting the critical safety risk in current models. Furthermore, the relationship between task difficulty and helpfulness reveals a potential application for good, enabling users to customize model behavior based on task complexity. Code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 长时间思考，更好表现：具有深层推理能力的大语言模型，特别是类似于o1的模型，在推断过程中生成了广泛的思想过程，表现出了显著的能力。这一权衡揭示了一个潜在的脆弱性：攻击者可以通过迫使模型立即响应而忽略思考过程来削弱模型性能。为了解决这一问题，本文提出了一种新颖的攻击场景，针对o1类似模型的长时间思考过程，并提出了一种称为BoT（Break CoT）的新方法，这种方法可以通过后门攻击选择性地破坏内在的推理机制。BoT通过设计的触发器构造中毒数据集，并通过监督微调或直接偏好优化注入后门。当被触发时，模型直接生成答案而不进行思考过程，但对于干净的输入，模型仍然保持正常的推理能力。对开源o1类似模型，包括最近的DeepSeek-R1进行的广泛实验表明，BoT几乎可以实现高攻击成功率，同时保持良好的准确率，突显了当前模型中关键的安全风险。此外，任务难度与帮助性之间的关系揭示了潜在的应用可能性，可以允许用户基于任务复杂性自定义模型行为。相关代码可在 \href{this https URL}{此链接} 获取。 

---
# Efficient and Effective Prompt Tuning via Prompt Decomposition and Compressed Outer Product 

**Title (ZH)**: 通过提示分解和压缩外积实现高效的提示调优 

**Authors**: Pengxiang Lan, Haoyu Xu, Enneng Yang, Yuliang Liang, Guibing Guo, Jianzhe Zhao, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12200)  

**Abstract**: Prompt tuning (PT) offers a cost-effective alternative to fine-tuning large-scale pre-trained language models (PLMs), requiring only a few parameters in soft prompt tokens added before the input text. However, existing PT approaches face two significant issues: (i) They overlook intrinsic semantic associations between soft prompt tokens, leading to high discreteness and limited interactions, thus reducing the model's comprehension and effectiveness in complex tasks. (ii) Due to the complexity of downstream tasks, long soft prompt is necessitated to improve performance, but prompt length correlates positively with memory usage and computational costs. Achieving high efficiency and performance remains an ongoing challenge. To address these issues, we propose a novel Low-parameters prompt tuning (LAMP) method, which leverages prompt decomposition and compressed outer product. Specifically, the prompt decomposition module employs Truncated SVD to reduce training parameters and significantly lower the dimensionality of the soft prompt parameter space. It then utilizes a compressed outer product module to facilitate multiple interactions among prompt tokens, exploring their intrinsic associations to enhance knowledge representation. Finally, LAMP uses average pooling to reduce memory usage and training/inference time. Extensive experiments across six architectures and eight datasets demonstrate that LAMP outperforms state-of-the-art PT-based and LoRA-based methods in performance and efficiency. 

**Abstract (ZH)**: 提示调优（PT）提供了一种成本效益高的替代方案，用于微调大规模预训练语言模型（PLMs），仅需在输入文本前添加少量的软提示令牌。然而，现有的PT方法面临两个重要的问题：（i）它们忽视了软提示令牌之间内在的语义关联，导致高离散性和有限的交互，从而降低了模型在复杂任务中的理解和有效性。（ii）由于下游任务的复杂性，需要较长的软提示以提高性能，但软提示长度与记忆使用量和计算成本呈正相关。实现高效性和性能仍然是一个持续的挑战。为解决这些问题，我们提出了一种新颖的低参数提示调优（LAMP）方法，该方法利用了提示分解和压缩外积。具体来说，提示分解模块采用截断奇异值分解（Truncated SVD）来减少训练参数，并显著降低软提示参数空间的维度。随后，利用压缩外积模块促进多个提示令牌之间的交互，探索其内在关联以增强知识表示。最后，LAMP 使用平均池化来减少内存使用和训练/推理时间。在涵盖六种架构和八个数据集的广泛实验中，LAMP 在性能和效率方面均优于最先进的基于PT和基于LoRA的方法。 

---
# Maximize Your Diffusion: A Study into Reward Maximization and Alignment for Diffusion-based Control 

**Title (ZH)**: 最大化扩散效应：关于基于扩散的控制中的奖励最大化与对齐研究 

**Authors**: Dom Huh, Prasant Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2502.12198)  

**Abstract**: Diffusion-based planning, learning, and control methods present a promising branch of powerful and expressive decision-making solutions. Given the growing interest, such methods have undergone numerous refinements over the past years. However, despite these advancements, existing methods are limited in their investigations regarding general methods for reward maximization within the decision-making process. In this work, we study extensions of fine-tuning approaches for control applications. Specifically, we explore extensions and various design choices for four fine-tuning approaches: reward alignment through reinforcement learning, direct preference optimization, supervised fine-tuning, and cascading diffusion. We optimize their usage to merge these independent efforts into one unified paradigm. We show the utility of such propositions in offline RL settings and demonstrate empirical improvements over a rich array of control tasks. 

**Abstract (ZH)**: 基于扩散的方法在决策制定、学习和控制方面展现出了一个充满潜力且极具表达力的解决方案分支。随着相关研究的兴趣不断增长，这些方法在过去几年中经历了众多改进。然而，尽管存在这些进展，现有的方法仍然在决策制定过程中关于通用的奖赏最大化方法方面存在局限。在本文中，我们研究了细调方法在控制应用中的扩展。具体而言，我们探讨了四种细调方法的扩展和设计选择：奖励对齐通过强化学习、直接偏好优化、监督细调以及级联扩散的扩展。我们优化了这些方法的使用，以将这些独立的努力统一到一个共同的框架中。我们展示了此类提议在离线强化学习环境中的实用性，并在一系列控制任务中展示了其实验改进。 

---
# A Closer Look at System Prompt Robustness 

**Title (ZH)**: 对系统提示鲁棒性的一种更深入的考察 

**Authors**: Norman Mu, Jonathan Lu, Michael Lavery, David Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2502.12197)  

**Abstract**: System prompts have emerged as a critical control surface for specifying the behavior of LLMs in chat and agent settings. Developers depend on system prompts to specify important context, output format, personalities, guardrails, content policies, and safety countermeasures, all of which require models to robustly adhere to the system prompt, especially when facing conflicting or adversarial user inputs. In practice, models often forget to consider relevant guardrails or fail to resolve conflicting demands between the system and the user. In this work, we study various methods for improving system prompt robustness by creating realistic new evaluation and fine-tuning datasets based on prompts collected from from OpenAI's GPT Store and HuggingFace's HuggingChat. Our experiments assessing models with a panel of new and existing benchmarks show that performance can be considerably improved with realistic fine-tuning data, as well as inference-time interventions such as classifier-free guidance. Finally, we analyze the results of recently released reasoning models from OpenAI and DeepSeek, which show exciting but uneven improvements on the benchmarks we study. Overall, current techniques fall short of ensuring system prompt robustness and further study is warranted. 

**Abstract (ZH)**: 系统提示已成为规定大型语言模型在对话和代理设置中行为的关键控制面。开发者依赖系统提示来指定重要的上下文、输出格式、个性、防护栏、内容政策和安全对策，所有这些都需要模型在面对冲突或敌对用户输入时能够稳健地遵守系统提示。实践中，模型经常忽视相关的防护栏，或者无法解决系统与用户之间的冲突需求。在这项工作中，我们通过基于从OpenAI的GPT Store和HuggingFace的HuggingChat收集到的提示创建现实的新评估和微调数据集，研究了提高系统提示稳健性的各种方法。我们利用由新旧基准组成的专家组评估模型的结果表明，通过现实的微调数据和推理时的干预措施（如分类器自由引导）可以显著提高性能。最后，我们分析了近年来OpenAI和DeepSeek发布的推理模型的结果，这些结果在我们研究的基准测试上显示出令人振奋但不均衡的进步。总体而言，当前的技术尚未确保系统提示的稳健性，进一步的研究是必要的。 

---
# AI and the Law: Evaluating ChatGPT's Performance in Legal Classification 

**Title (ZH)**: AI与法律：评析ChatGPT在法律分类任务中的表现 

**Authors**: Pawel Weichbroth  

**Link**: [PDF](https://arxiv.org/pdf/2502.12193)  

**Abstract**: The use of ChatGPT to analyze and classify evidence in criminal proceedings has been a topic of ongoing discussion. However, to the best of our knowledge, this issue has not been studied in the context of the Polish language. This study addresses this research gap by evaluating the effectiveness of ChatGPT in classifying legal cases under the Polish Penal Code. The results show excellent binary classification accuracy, with all positive and negative cases correctly categorized. In addition, a qualitative evaluation confirms that the legal basis provided for each case, along with the relevant legal content, was appropriate. The results obtained suggest that ChatGPT can effectively analyze and classify evidence while applying the appropriate legal rules. In conclusion, ChatGPT has the potential to assist interested parties in the analysis of evidence and serve as a valuable legal resource for individuals with less experience or knowledge in this area. 

**Abstract (ZH)**: 使用ChatGPT分析和分类刑事程序中的证据一直是持续讨论的课题。然而，据我们所知，这一问题尚未在波兰语语境下进行研究。本研究通过评估ChatGPT在分类适用波兰刑法的法律案件方面的有效性，填补了这一研究空白。研究结果显示，ChatGPT表现出色，二分类准确率极高，所有阳性案例和阴性案例均被正确分类。此外，定性评估确认，为每起案件提供的法律依据及其相关法律内容都是合适的。研究结果表明，ChatGPT能够有效分析和分类证据，并正确应用适当的法律规则。综上所述，ChatGPT有潜力帮助有兴趣的人员分析证据，并为缺乏此领域经验或知识的个人提供有价值的法律资源。 

---
# Self-supervised Attribute-aware Dynamic Preference Ranking Alignment 

**Title (ZH)**: 自我监督的属性意识动态偏好排序对齐 

**Authors**: Hongyu Yang, Qi Zhao, Zhenhua hu, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12189)  

**Abstract**: Reinforcement Learning from Human Feedback and its variants excel in aligning with human intentions to generate helpful, harmless, and honest responses. However, most of them rely on costly human-annotated pairwise comparisons for supervised alignment, which is not suitable for list-level scenarios, such as community question answering. Additionally, human preferences are influenced by multiple intrinsic factors in responses, leading to decision-making inconsistencies. Therefore, we propose \textbf{Se}lf-supervised \textbf{A}ttribute-aware \textbf{d}ynamic \textbf{p}reference \textbf{ra}nking, called \shortname. \ It quantifies preference differences between responses based on Attribute-Perceptual Distance Factors (APDF) and dynamically determines the list-wise alignment order. Furthermore, it achieves fine-grained preference difference learning and enables precise alignment with the optimal one. We specifically constructed a challenging code preference dataset named StaCoCoQA, and introduced more cost-effective and scalable preference evaluation metrics: PrefHit and PrefRecall. Extensive experimental results show that SeAdpra exhibits superior performance and generalizability on both StaCoCoQA and preference datasets from eight popular domains. 

**Abstract (ZH)**: 强化学习通过人类反馈进行训练及其变体在生成有益、无害和诚实的响应方面能够很好地与人类意愿保持一致。然而，它们大多依赖于昂贵的人工标注成对比较进行监督对齐，这种做法不适合列表级别的场景，如社区问答。此外，人类偏好受响应中的多重内在因素影响，导致决策一致性问题。因此，我们提出了一种自我监督的属性感知动态偏好排序方法，称为SeAdpra（Self-supervised Attribute-aware Dynamic Preference Ranking）。它基于属性感知距离因子（APDF）量化响应之间的偏好差异，并动态确定列表级别的对齐顺序。此外，它实现了精细的偏好差异学习，能够与最优目标实现精确对齐。我们特别构建了一个具有挑战性的代码偏好数据集，名为StaCoCoQA，并引入了更加经济和可扩展的偏好评估指标：PrefHit和PrefRecall。广泛进行的实验结果表明，SeAdpra在StaCoCoQA和八个流行领域中的偏好数据集上均表现出优越的性能和通用性。 

---
# Boosting Generalization in Diffusion-Based Neural Combinatorial Solver via Energy-guided Sampling 

**Title (ZH)**: 基于能量导向采样的扩散模型神经组合求解器的泛化能力提升方法 

**Authors**: Haoyu Lei, Kaiwen Zhou, Yinchuan Li, Zhitang Chen, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2502.12188)  

**Abstract**: Diffusion-based Neural Combinatorial Optimization (NCO) has demonstrated effectiveness in solving NP-complete (NPC) problems by learning discrete diffusion models for solution generation, eliminating hand-crafted domain knowledge. Despite their success, existing NCO methods face significant challenges in both cross-scale and cross-problem generalization, and high training costs compared to traditional solvers. While recent studies have introduced training-free guidance approaches that leverage pre-defined guidance functions for zero-shot conditional generation, such methodologies have not been extensively explored in combinatorial optimization. To bridge this gap, we propose a general energy-guided sampling framework during inference time that enhances both the cross-scale and cross-problem generalization capabilities of diffusion-based NCO solvers without requiring additional training. We provide theoretical analysis that helps understanding the cross-problem transfer capability. Our experimental results demonstrate that a diffusion solver, trained exclusively on the Traveling Salesman Problem (TSP), can achieve competitive zero-shot solution generation on TSP variants, such as Prize Collecting TSP (PCTSP) and the Orienteering Problem (OP), through energy-guided sampling across different problem scales. 

**Abstract (ZH)**: 基于扩散的神经组合优化（NCO）已经在通过学习离散扩散模型来生成解的过程中展示了解决NP完全（NPC）问题的有效性，并消除了手工构建的领域知识。尽管取得了成功，现有的NCO方法在跨尺度和跨问题泛化方面仍面临重大挑战，并且与传统的求解器相比，训练成本较高。虽然最近的研究引入了训练-free 指导方法，利用预定义的指导函数进行零样本条件生成，但这些方法在组合优化中尚未得到广泛探索。为弥补这一不足，我们提出了一种通用的能量引导采样框架，在推断时间增强扩散基于NCO求解器的跨尺度和跨问题泛化能力，而无需额外的训练。我们提供了理论分析，帮助理解跨问题迁移能力。实验结果表明，仅在旅行商问题（TSP）上训练的扩散求解器，通过不同问题尺度下的能量引导采样，能够实现旅行商问题变种（如收集奖赏旅行商问题(PCTSP)和旅行者问题(OP)）的竞争力零样本解生成。 

---
# E2CB2former: Effecitve and Explainable Transformer for CB2 Receptor Ligand Activity Prediction 

**Title (ZH)**: E2CB2former: 有效的可解释变压器模型在CB2受体配体活性预测中的应用 

**Authors**: Jiacheng Xie, Yingrui Ji, Linghuan Zeng, Xi Xiao, Gaofei Chen, Lijing Zhu, Joyanta Jyoti Mondal, Jiansheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12186)  

**Abstract**: Accurate prediction of CB2 receptor ligand activity is pivotal for advancing drug discovery targeting this receptor, which is implicated in inflammation, pain management, and neurodegenerative conditions. Although conventional machine learning and deep learning techniques have shown promise, their limited interpretability remains a significant barrier to rational drug design. In this work, we introduce CB2former, a framework that combines a Graph Convolutional Network with a Transformer architecture to predict CB2 receptor ligand activity. By leveraging the Transformer's self attention mechanism alongside the GCN's structural learning capability, CB2former not only enhances predictive performance but also offers insights into the molecular features underlying receptor activity. We benchmark CB2former against diverse baseline models including Random Forest, Support Vector Machine, K Nearest Neighbors, Gradient Boosting, Extreme Gradient Boosting, Multilayer Perceptron, Convolutional Neural Network, and Recurrent Neural Network and demonstrate its superior performance with an R squared of 0.685, an RMSE of 0.675, and an AUC of 0.940. Moreover, attention weight analysis reveals key molecular substructures influencing CB2 receptor activity, underscoring the model's potential as an interpretable AI tool for drug discovery. This ability to pinpoint critical molecular motifs can streamline virtual screening, guide lead optimization, and expedite therapeutic development. Overall, our results showcase the transformative potential of advanced AI approaches exemplified by CB2former in delivering both accurate predictions and actionable molecular insights, thus fostering interdisciplinary collaboration and innovation in drug discovery. 

**Abstract (ZH)**: 准确预测CB2受体配体活性对于针对该受体的药物发现具有重要意义，而CB2受体与炎症、疼痛管理和神经退行性疾病有关。尽管传统的机器学习和深度学习技术显示出潜力，但其有限的可解释性仍是合理药物设计中的一个重大障碍。在此项工作中，我们提出了一种CB2former框架，该框架结合了图卷积网络（GCN）和变换器（Transformer）架构，用于预测CB2受体配体活性。通过利用Transformer的自注意力机制和GCN的结构学习能力，CB2former不仅提高了预测性能，还为受体活性背后的分子特征提供了见解。我们将CB2former与多种基准模型（包括随机森林、支持向量机、K近邻、梯度提升、极端梯度提升、多层感知机、卷积神经网络和循环神经网络）进行了基准测试，并展示了其优越的性能，R平方值为0.685，均方根误差（RMSE）为0.675，AUC值为0.940。此外，注意力权重分析揭示了关键的分子亚结构对CB2受体活性的影响，突显了该模型作为药物发现中可解释的人工智能工具的潜力。这种能够精确定位关键分子模式的能力可以简化虚拟筛选、指导先导化合物优化，并加快药物发现的进程。总的来说，我们的结果展示了CB2former等高级人工智能方法在提供准确预测和可操作的分子洞察方面的转变潜力，促进了药物发现领域的跨学科合作和创新。

该框架不仅能提供高准确度的预测，还能揭示分子特征，对于指导分子设计、优化先导化合物以及加速药物开发具有重要意义。这项工作的结果突显了CB2former在药物发现中的应用潜力，为精准预测和分子洞察提供了强有力的支持，从而促进了跨学科的合作与创新。 

---
# Large Language Models for Extrapolative Modeling of Manufacturing Processes 

**Title (ZH)**: 大型语言模型在制造过程外推建模中的应用 

**Authors**: Kiarash Naghavi Khanghah, Anandkumar Patel, Rajiv Malhotra, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12185)  

**Abstract**: Conventional predictive modeling of parametric relationships in manufacturing processes is limited by the subjectivity of human expertise and intuition on the one hand and by the cost and time of experimental data generation on the other hand. This work addresses this issue by establishing a new Large Language Model (LLM) framework. The novelty lies in combining automatic extraction of process-relevant knowledge embedded in the literature with iterative model refinement based on a small amount of experimental data. This approach is evaluated on three distinct manufacturing processes that are based on machining, deformation, and additive principles. The results show that for the same small experimental data budget the models derived by our framework have unexpectedly high extrapolative performance, often surpassing the capabilities of conventional Machine Learning. Further, our approach eliminates manual generation of initial models or expertise-dependent interpretation of the literature. The results also reveal the importance of the nature of the knowledge extracted from the literature and the significance of both the knowledge extraction and model refinement components. 

**Abstract (ZH)**: 传统的制造过程参数关系预测建模受限于人类专家经验的主观性和直觉，以及实验数据生成的成本和时间。本研究通过建立一个新的大型语言模型（LLM）框架来解决这些问题。该框架的创新之处在于结合了从文献中自动提取与过程相关的知识，并基于少量实验数据迭代优化模型。该方法在三个基于加工、变形和增材原理的不同制造过程中进行了评估。结果显示，对于相同的少量实验数据预算，由本框架推导出的模型具有令人意外的外推性能，往往超越了传统机器学习的能力。此外，本方法消除了手动生成初始模型或依赖专家解释文献的需求。研究结果还揭示了从文献中提取知识的性质的重要性，以及知识提取和模型优化两部分的显著意义。 

---
# Towards Transparent and Accurate Plasma State Monitoring at JET 

**Title (ZH)**: Towards 透明且准确的JET装置等离子体状态监测 

**Authors**: Andrin Bürli, Alessandro Pau, Thomas Koller, Olivier Sauter, JET Contributors  

**Link**: [PDF](https://arxiv.org/pdf/2502.12182)  

**Abstract**: Controlling and monitoring plasma within a tokamak device is complex and challenging. Plasma off-normal events, such as disruptions, are hindering steady-state operation. For large devices, they can even endanger the machine's integrity and it represents in general one of the most serious concerns for the exploitation of the tokamak concept for future power plants. Effective plasma state monitoring carries the potential to enable an understanding of such phenomena and their evolution which is crucial for the successful operation of tokamaks. This paper presents the application of a transparent and data-driven methodology to monitor the plasma state in a tokamak. Compared to previous studies in the field, supervised and unsupervised learning techniques are combined. The dataset consisted of 520 expert-validated discharges from JET. The goal was to provide an interpretable plasma state representation for the JET operational space by leveraging multi-task learning for the first time in the context of plasma state monitoring. When evaluated as disruption predictors, a sequence-based approach showed significant improvements compared to the state-based models. The best resulting network achieved a promising cross-validated success rate when combined with a physical indicator and accounting for nearby instabilities. Qualitative evaluations of the learned latent space uncovered operational and disruptive regions as well as patterns related to learned dynamics and global feature importance. The applied methodology provides novel possibilities for the definition of triggers to switch between different control scenarios, data analysis, and learning as well as exploring latent dynamics for plasma state monitoring. It also showed promising quantitative and qualitative results with warning times suitable for avoidance purposes and distributions that are consistent with known physical mechanisms. 

**Abstract (ZH)**: 在托卡马克设备中控制和监控等离子体是复杂且具有挑战性的。异常的等离子体事件，如失稳，阻碍了稳态操作。对于大型设备而言，它们甚至可能危及机器的完整性，通常代表了使用托卡马克概念作为未来发电厂的优势中最严重的担忧之一。有效的等离子体状态监控有可能帮助理解这些现象及其演变，这对于托卡马克的成功运行至关重要。本文提出了一种透明且数据驱动的方法，用于监测托卡马克中的等离子体状态。与该领域之前的研究所使用的监督学习和无监督学习技术不同，本文结合了这两种方法。数据集由JET设备的520个专家验证的放电过程组成。目标是通过首次在等离子体状态监控的上下文中利用多任务学习，提供JET操作空间的可解释的等离子体状态表示。当作为失稳预测器进行评估时，基于序列的方法相较于基于状态的方法显示出显著的改进。结合物理指标并考虑到邻近不稳定性的最佳网络，在验证交叉验证中取得了令人鼓舞的成功率。学习到的潜在空间的定性评估揭示了操作和失稳区域，以及与学习动态和全局特征重要性相关的模式。所应用的方法为不同控制场景的触发定义提供了新的可能性，数据的分析、学习以及探索潜在动态用于等离子体状态监控。该方法还显示了有希望的定量和定性结果，具有适合避免目的的预警时间，并且其分布与已知的物理机制一致。 

---
# 3D ReX: Causal Explanations in 3D Neuroimaging Classification 

**Title (ZH)**: 3D ReX：三维神经影像分类中的因果解释 

**Authors**: Melane Navaratnarajah, Sophie A. Martin, David A. Kelly, Nathan Blake, Hana Chocker  

**Link**: [PDF](https://arxiv.org/pdf/2502.12181)  

**Abstract**: Explainability remains a significant problem for AI models in medical imaging, making it challenging for clinicians to trust AI-driven predictions. We introduce 3D ReX, the first causality-based post-hoc explainability tool for 3D models. 3D ReX uses the theory of actual causality to generate responsibility maps which highlight the regions most crucial to the model's decision. We test 3D ReX on a stroke detection model, providing insight into the spatial distribution of features relevant to stroke. 

**Abstract (ZH)**: 可解释性仍然是医学影像中AI模型面临的一项重大问题，这使得临床医生难以信任由AI驱动的预测结果。为了解决这一问题，我们引入了3D ReX，这是首个基于因果关系的后验可解释性工具，专门用于3D模型。3D ReX 使用实际因果理论生成责任图，突出显示对模型决策至关重要的区域。我们对一个中风检测模型进行了3D ReX 的测试，在此过程中揭示了与中风相关的空间特征分布。 

---
# ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis 

**Title (ZH)**: ClusMFL：一种用于脑成像分析的模态不完整多模态联邦学习的聚类增强框架 

**Authors**: Xinpeng Wang, Rong Zhou, Han Xie, Xiaoying Tang, Lifang He, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12180)  

**Abstract**: Multimodal Federated Learning (MFL) has emerged as a promising approach for collaboratively training multimodal models across distributed clients, particularly in healthcare domains. In the context of brain imaging analysis, modality incompleteness presents a significant challenge, where some institutions may lack specific imaging modalities (e.g., PET, MRI, or CT) due to privacy concerns, device limitations, or data availability issues. While existing work typically assumes modality completeness or oversimplifies missing-modality scenarios, we simulate a more realistic setting by considering both client-level and instance-level modality incompleteness in this study. Building on this realistic simulation, we propose ClusMFL, a novel MFL framework that leverages feature clustering for cross-institutional brain imaging analysis under modality incompleteness. Specifically, ClusMFL utilizes the FINCH algorithm to construct a pool of cluster centers for the feature embeddings of each modality-label pair, effectively capturing fine-grained data distributions. These cluster centers are then used for feature alignment within each modality through supervised contrastive learning, while also acting as proxies for missing modalities, allowing cross-modal knowledge transfer. Furthermore, ClusMFL employs a modality-aware aggregation strategy, further enhancing the model's performance in scenarios with severe modality incompleteness. We evaluate the proposed framework on the ADNI dataset, utilizing structural MRI and PET scans. Extensive experimental results demonstrate that ClusMFL achieves state-of-the-art performance compared to various baseline methods across varying levels of modality incompleteness, providing a scalable solution for cross-institutional brain imaging analysis. 

**Abstract (ZH)**: 多模态联邦学习（MFL）已经成为了在分布式客户端之间合作训练多模态模型的一种有前途的方法，尤其是在医疗保健领域。在脑成像分析的背景下，模态不完整是一个重大挑战，一些机构可能由于隐私问题、设备限制或数据可用性问题而缺乏特定的成像模态（如PET、MRI或CT）。现有的研究通常假设模态完整性或过于简化缺失模态的情形，而在本研究中，我们通过同时考虑客户端级别和实例级别的模态不完整性来模拟一个更为现实的场景。基于这一现实的模拟，我们提出了一种名为ClusMFL的新颖MFL框架，该框架利用特征聚类方法在模态不完整的情况下进行跨机构的脑成像分析。具体来说，ClusMFL 利用FINCH算法为每种模态-标签对构建特征嵌入的聚类中心池，有效地捕获细粒度的数据分布。这些聚类中心随后通过监督对比学习在每个模态内实现特征对齐，并作为缺失模态的代理，促进跨模态知识的转移。此外，ClusMFL 还采用了一种模态感知的聚合策略，进一步增强了在严重模态不完整情况下的模型性能。我们在ADNI数据集上评估了所提出的方法，利用结构MRI和PET扫描。大量的实验结果表明，ClusMFL 在不同水平的模态不完整情况下，相比各种基线方法都能达到最先进的性能，为跨机构脑成像分析提供了一个可扩展的解决方案。 

---
# Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts 

**Title (ZH)**: 通过稀疏自编码多重概念变化的可识别控制 

**Authors**: Shruti Joshi, Andrea Dittadi, Sébastien Lachapelle, Dhanya Sridhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12179)  

**Abstract**: Steering methods manipulate the representations of large language models (LLMs) to induce responses that have desired properties, e.g., truthfulness, offering a promising approach for LLM alignment without the need for fine-tuning. Traditionally, steering has relied on supervision, such as from contrastive pairs of prompts that vary in a single target concept, which is costly to obtain and limits the speed of steering research. An appealing alternative is to use unsupervised approaches such as sparse autoencoders (SAEs) to map LLM embeddings to sparse representations that capture human-interpretable concepts. However, without further assumptions, SAEs may not be identifiable: they could learn latent dimensions that entangle multiple concepts, leading to unintentional steering of unrelated properties. We introduce Sparse Shift Autoencoders (SSAEs) that instead map the differences between embeddings to sparse representations. Crucially, we show that SSAEs are identifiable from paired observations that vary in \textit{multiple unknown concepts}, leading to accurate steering of single concepts without the need for supervision. We empirically demonstrate accurate steering across semi-synthetic and real-world language datasets using Llama-3.1 embeddings. 

**Abstract (ZH)**: 操控方法通过调整大型语言模型（LLMs）的表示以诱导具有所需属性的响应，例如真实性，为无需微调即可实现LLM对齐提供了有前景的方法。传统上，操控依赖于监督，例如通过在单个目标概念上有差异的对比提示对来实现，这种方式获取成本较高，限制了操控研究的速度。一种吸引人的替代方法是使用无监督方法，例如稀疏自动编码器（SAEs），将LLM嵌入映射到能够捕捉人类可解释概念的稀疏表示。但是，在没有额外假设的情况下，SAEs可能不具备可识别性：它们可能会学习使多个概念纠缠在一起的潜在维度，从而导致对无关属性的无意操控。我们引入了稀疏平移自动编码器（SSAEs），它将嵌入之间的差异映射到稀疏表示。至关重要的是，我们证明了通过成对观察在多个未知概念上有差异的样本，可以识别SSAEs，从而在无需监督的情况下实现对单一概念的准确操控。我们使用Llama-3.1嵌入在半合成和真实语言数据集中 empirically 证明了这种准确操控。 

---
# Ten Challenging Problems in Federated Foundation Models 

**Title (ZH)**: 联邦基础模型中具有挑战性的十个问题 

**Authors**: Tao Fan, Hanlin Gu, Xuemei Cao, Chee Seng Chan, Qian Chen, Yiqiang Chen, Yihui Feng, Yang Gu, Jiaxiang Geng, Bing Luo, Shuoling Liu, Win Kent Ong, Chao Ren, Jiaqi Shao, Chuan Sun, Xiaoli Tang, Hong Xi Tae, Yongxin Tong, Shuyue Wei, Fan Wu, Wei Xi, Mingcong Xu, He Yang, Xin Yang, Jiangpeng Yan, Hao Yu, Han Yu, Teng Zhang, Yifei Zhang, Xiaojin Zhang, Zhenzhe Zheng, Lixin Fan, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12176)  

**Abstract**: Federated Foundation Models (FedFMs) represent a distributed learning paradigm that fuses general competences of foundation models as well as privacy-preserving capabilities of federated learning. This combination allows the large foundation models and the small local domain models at the remote clients to learn from each other in a teacher-student learning setting. This paper provides a comprehensive summary of the ten challenging problems inherent in FedFMs, encompassing foundational theory, utilization of private data, continual learning, unlearning, Non-IID and graph data, bidirectional knowledge transfer, incentive mechanism design, game mechanism design, model watermarking, and efficiency. The ten challenging problems manifest in five pivotal aspects: ``Foundational Theory," which aims to establish a coherent and unifying theoretical framework for FedFMs. ``Data," addressing the difficulties in leveraging domain-specific knowledge from private data while maintaining privacy; ``Heterogeneity," examining variations in data, model, and computational resources across clients; ``Security and Privacy," focusing on defenses against malicious attacks and model theft; and ``Efficiency," highlighting the need for improvements in training, communication, and parameter efficiency. For each problem, we offer a clear mathematical definition on the objective function, analyze existing methods, and discuss the key challenges and potential solutions. This in-depth exploration aims to advance the theoretical foundations of FedFMs, guide practical implementations, and inspire future research to overcome these obstacles, thereby enabling the robust, efficient, and privacy-preserving FedFMs in various real-world applications. 

**Abstract (ZH)**: 联邦基础模型 (FedFMs) 代表了一种分布式学习范式，融合了基础模型的一般能力和联邦学习的隐私保护能力。这种结合使得大型基础模型和远程客户端处的小型局部领域模型能够在教师-学生学习设置中相互学习。本文对 FedFMs 中存在的十个挑战性问题进行了全面总结，涵盖基础理论、隐私数据的利用、持续学习、遗忘、非IID和图数据、双向知识转移、激励机制设计、博弈机制设计、模型水印和效率等方面。这十个挑战性问题体现在五个关键方面：1）“基础理论”，旨在为 FedFMs 建立一个连贯且统一的理论框架；2）“数据”，解决在利用私人数据中的领域特定知识的同时保持隐私的难题；3）“异质性”，探讨各个客户端之间在数据、模型和计算资源方面的差异；4）“安全和隐私”，关注对抗恶意攻击和模型盗窃的防护措施；5）“效率”，强调在训练、通信和参数效率方面进行改进的必要性。对于每个问题，我们提供了清晰的数学定义，分析现有方法，并讨论关键挑战和潜在解决方案。深入的探讨旨在推进 FedFMs 的理论基础，指导实际实施，并激发未来的研究来克服这些障碍，从而在各种实际应用场景中实现稳健、高效且隐私保护的 FedFMs。 

---
# Spatiotemporal Graph Neural Networks in short term load forecasting: Does adding Graph Structure in Consumption Data Improve Predictions? 

**Title (ZH)**: 空间时间图神经网络在短期负荷预测中的应用：在消费数据中加入图结构能否提高预测精度？ 

**Authors**: Quoc Viet Nguyen, Joaquin Delgado Fernandez, Sergio Potenciano Menci  

**Link**: [PDF](https://arxiv.org/pdf/2502.12175)  

**Abstract**: Short term Load Forecasting (STLF) plays an important role in traditional and modern power systems. Most STLF models predominantly exploit temporal dependencies from historical data to predict future consumption. Nowadays, with the widespread deployment of smart meters, their data can contain spatiotemporal dependencies. In particular, their consumption data is not only correlated to historical values but also to the values of neighboring smart meters. This new characteristic motivates researchers to explore and experiment with new models that can effectively integrate spatiotemporal interrelations to increase forecasting performance. Spatiotemporal Graph Neural Networks (STGNNs) can leverage such interrelations by modeling relationships between smart meters as a graph and using these relationships as additional features to predict future energy consumption. While extensively studied in other spatiotemporal forecasting domains such as traffic, environments, or renewable energy generation, their application to load forecasting remains relatively unexplored, particularly in scenarios where the graph structure is not inherently available. This paper overviews the current literature focusing on STGNNs with application in STLF. Additionally, from a technical perspective, it also benchmarks selected STGNN models for STLF at the residential and aggregate levels. The results indicate that incorporating graph features can improve forecasting accuracy at the residential level; however, this effect is not reflected at the aggregate level 

**Abstract (ZH)**: 短时负荷预测（STLF）在传统和现代电力系统中扮演着重要角色。大多数STLF模型主要通过历史数据的时间相关性来预测未来的消耗量。如今，随着智能电表的广泛部署，它们的数据中不仅包含了时间相关性，还包含了空间相关性。特别是，住宅用户的用电数据不仅与历史值相关，还与邻近智能电表的值相关。这种新的特性促使研究人员探索和试验能够有效整合时空关系的新模型，从而提高预测性能。时空图神经网络（STGNNs）可以通过将智能电表之间的关系建模为一个图，并利用这些关系作为额外特征来预测未来的能源消耗，从而利用这些时空关系。虽然STGNNs在交通、环境或可再生能源生成等其他时空预测领域得到了广泛研究，但在用于负荷预测时的应用探索相对较少，特别是在没有固有的图结构的场景中。本文综述了当前关于STGNNs在STLF应用方面的文献，并从技术角度对选定的STGNN模型在住宅和聚合水平上的STLF进行了基准测试。结果表明，在住宅水平上整合图特征可以提高预测准确性；然而，在聚合水平上这种效果并未体现。 

---
# nanoML for Human Activity Recognition 

**Title (ZH)**: 纳米机器学习在人体活动识别中的应用 

**Authors**: Alan T. L. Bacellar, Mugdha P. Jadhao, Shashank Nag, Priscila M. V. Lima, Felipe M. G. Franca, Lizy K. John  

**Link**: [PDF](https://arxiv.org/pdf/2502.12173)  

**Abstract**: Human Activity Recognition (HAR) is critical for applications in healthcare, fitness, and IoT, but deploying accurate models on resource-constrained devices remains challenging due to high energy and memory demands. This paper demonstrates the application of Differentiable Weightless Neural Networks (DWNs) to HAR, achieving competitive accuracies of 96.34% and 96.67% while consuming only 56nJ and 104nJ per sample, with an inference time of just 5ns per sample. The DWNs were implemented and evaluated on an FPGA, showcasing their practical feasibility for energy-efficient hardware deployment. DWNs achieve up to 926,000x energy savings and 260x memory reduction compared to state-of-the-art deep learning methods. These results position DWNs as a nano-machine learning nanoML model for HAR, setting a new benchmark in energy efficiency and compactness for edge and wearable devices, paving the way for ultra-efficient edge AI. 

**Abstract (ZH)**: 人体活动识别（HAR）在医疗、健身和物联网应用中至关重要，但在资源受限的设备上部署准确的模型仍然面临挑战，主要由于高能耗和大内存需求。本文展示了不同iable无权重神经网络（DWNs）在HAR中的应用，实现了96.34%和96.67%的竞争力精度，同时每样本能耗仅分别为56nJ和104nJ，每样本推理时间为5ns。DWNs在FPGA上进行了实现与评估，展示了它们在能效硬件部署中的实际可行性。DWNs相比最先进的深度学习方法实现了高达926,000倍的能效提升和260倍的内存减少。这些结果使DWNs成为了适用于HAR的纳米机器学习（nanoML）模型，为边缘和可穿戴设备设定了新的能效和紧凑性基准，为超高效的边缘AI奠定了基础。 

---
# GoRA: Gradient-driven Adaptive Low Rank Adaptation 

**Title (ZH)**: GoRA: 梯度驱动的自适应低秩适应 

**Authors**: Haonan He, Peng Ye, Yuchen Ren, Yuan Yuan, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12171)  

**Abstract**: Low-Rank Adaptation (LoRA) is a crucial method for efficiently fine-tuning pretrained large language models (LLMs), with its performance largely influenced by two key factors: rank and initialization strategy. Numerous LoRA variants have been proposed to enhance its performance by addressing these factors. However, these variants often compromise LoRA's usability or efficiency. In this paper, we analyze the fundamental limitations of existing methods and introduce a novel approach, GoRA (Gradient-driven Adaptive Low Rank Adaptation), which adaptively assigns ranks and initializes weights for low-rank adapters simultaneously based on gradient information. Extensive experimental results demonstrate that GoRA significantly improves performance while preserving the high usability and efficiency of LoRA. On the T5 model fine-tuned for the GLUE benchmark, GoRA achieves a 5.88-point improvement over LoRA and slightly surpasses full fine-tuning. Similarly, on the Llama3.1-8B-Base model fine-tuned for GSM8k tasks, GoRA outperforms LoRA with a 5.13-point improvement and exceeds full fine-tuning in high-rank settings by a margin of 2.05 points. 

**Abstract (ZH)**: 低秩适应（LoRA）是一种高效微调预训练大型语言模型（LLMs）的关键方法，其性能主要受到两个关键因素的影响：秩和初始化策略。为了提高性能，提出了许多LoRA变体，但这些变体往往在增加LoRA的性能的同时降低了其可用性和效率。本文对现有方法的基本局限性进行了分析，并提出了一种新型方法——GoRA（梯度驱动的自适应低秩适应），该方法基于梯度信息同时自适应地分配秩和初始化权重。大量实验结果表明，GoRA不仅显著提高了性能，还保持了LoRA的高可用性和高效性。在针对GLUE基准进行微调的T5模型上，GoRA比LoRA性能提高了5.88点，并且在某些情况下完全微调的性能略有超越。同样，在针对GSM8k任务进行微调的Llama3.1-8B-Base模型上，GoRA比LoRA性能提高了5.13点，在高秩设置中则比完全微调高出2.05点。 

---
# MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections 

**Title (ZH)**: MUDDFormer：通过多路动态密集连接打破transformer中的残差瓶颈 

**Authors**: Da Xiao, Qingye Meng, Shengping Li, Xingyuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12170)  

**Abstract**: We propose MUltiway Dynamic Dense (MUDD) connections, a simple yet effective method to address the limitations of residual connections and enhance cross-layer information flow in Transformers. Unlike existing dense connection approaches with static and shared connection weights, MUDD generates connection weights dynamically depending on hidden states at each sequence position and for each decoupled input stream (the query, key, value or residual) of a Transformer block. MUDD connections can be seamlessly integrated into any Transformer architecture to create MUDDFormer. Extensive experiments show that MUDDFormer significantly outperforms Transformers across various model architectures and scales in language modeling, achieving the performance of Transformers trained with 1.8X-2.4X compute. Notably, MUDDPythia-2.8B matches Pythia-6.9B in pretraining ppl and downstream tasks and even rivals Pythia-12B in five-shot settings, while adding only 0.23% parameters and 0.4% computation. Code in JAX and PyTorch and pre-trained models are available at this https URL . 

**Abstract (ZH)**: 我们提出了Multiway Dynamic Dense (MUDD) 连接，这是一种简单而有效的方法，用于解决残差连接的局限性，并增强Transformer中的跨层信息流。与现有的具有静态和共享连接权重的密集连接方法不同，MUDD 根据每个序列位置和每个解偶输入流（查询、键、值或残差）生成连接权重。MUDD 连接可以无缝集成到任何Transformer架构中，形成MUDDFormer。广泛的实验表明，MUDDFormer 在各种模型架构和规模的语言建模任务中远优于Transformer，其性能相当于使用1.8倍至2.4倍计算资源训练的Transformer。值得注意的是，在预训练PPL和下游任务上，MUDDPythia-2.8B 与Pythia-6.9B 相当，甚至在五拍设置上与Pythia-12B 持平，同时仅增加了0.23%的参数量和0.4%的计算量。JAX和PyTorch的代码及预训练模型可在以下链接获取：[此处填写链接]。 

---
# TastepepAI, An artificial intelligence platform for taste peptide de novo design 

**Title (ZH)**: TastepepAI，用于味肽从头设计的人工智能平台 

**Authors**: Jianda Yue, Tingting Li, Jian Ouyang, Jiawei Xu, Hua Tan, Zihui Chen, Changsheng Han, Huanyu Li, Songping Liang, Zhonghua Liu, Zhonghua Liu, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12167)  

**Abstract**: Taste peptides have emerged as promising natural flavoring agents attributed to their unique organoleptic properties, high safety profile, and potential health benefits. However, the de novo identification of taste peptides derived from animal, plant, or microbial sources remains a time-consuming and resource-intensive process, significantly impeding their widespread application in the food industry. Here, we present TastePepAI, a comprehensive artificial intelligence framework for customized taste peptide design and safety assessment. As the key element of this framework, a loss-supervised adaptive variational autoencoder (LA-VAE) is implemented to efficiently optimizes the latent representation of sequences during training and facilitates the generation of target peptides with desired taste profiles. Notably, our model incorporates a novel taste-avoidance mechanism, allowing for selective flavor exclusion. Subsequently, our in-house developed toxicity prediction algorithm (SpepToxPred) is integrated in the framework to undergo rigorous safety evaluation of generated peptides. Using this integrated platform, we successfully identified 73 peptides exhibiting sweet, salty, and umami, significantly expanding the current repertoire of taste peptides. This work demonstrates the potential of TastePepAI in accelerating taste peptide discovery for food applications and provides a versatile framework adaptable to broader peptide engineering challenges. 

**Abstract (ZH)**: 味肽已因其独特的感官特性、较高的安全性以及潜在的健康益处而成为有前景的天然着香剂。然而，从动物、植物或微生物来源中从头鉴定味肽仍然是一个耗时且资源密集的过程，这在很大程度上阻碍了它们在食品行业的广泛应用。在此，我们提出了TastePepAI，一个全面的人工智能框架，用于定制化的味肽设计和安全性评估。作为该框架的关键要素，我们实现了一种损失监督自适应变分自编码器（LA-VAE），在训练过程中有效优化了序列的潜在表示，并促进了具有预定味型的目标肽的生成。值得注意的是，我们的模型整合了一种新颖的味觉规避机制，允许选择性地排除特定风味。随后，我们自主研发的毒性预测算法（SpepToxPred）被集成到该框架中，以对生成的肽进行严格的安全性评估。借助该集成平台，我们成功鉴定了73种具有甜味、咸味和鲜味的肽，大大扩展了现有的味肽谱系。本工作证明了TastePepAI在加速食品应用中的味肽发现方面的潜力，并提供了一个适用于更广泛肽工程挑战的多功能框架。 

---
# Integrating Artificial Intelligence and Geophysical Insights for Earthquake Forecasting: A Cross-Disciplinary Review 

**Title (ZH)**: 将人工智能与地质物理学洞察集成用于地震预测：跨学科综述 

**Authors**: Zhang Ying, Wen Congcong, Sornette Didier, Zhan Chengxiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12161)  

**Abstract**: Earthquake forecasting remains a significant scientific challenge, with current methods falling short of achieving the performance necessary for meaningful societal benefits. Traditional models, primarily based on past seismicity and geomechanical data, struggle to capture the complexity of seismic patterns and often overlook valuable non-seismic precursors such as geophysical, geochemical, and atmospheric anomalies. The integration of such diverse data sources into forecasting models, combined with advancements in AI technologies, offers a promising path forward. AI methods, particularly deep learning, excel at processing complex, large-scale datasets, identifying subtle patterns, and handling multidimensional relationships, making them well-suited for overcoming the limitations of conventional approaches.
This review highlights the importance of combining AI with geophysical knowledge to create robust, physics-informed forecasting models. It explores current AI methods, input data types, loss functions, and practical considerations for model development, offering guidance to both geophysicists and AI researchers. While many AI-based studies oversimplify earthquake prediction, neglecting critical features such as data imbalance and spatio-temporal clustering, the integration of specialized geophysical insights into AI models can address these shortcomings.
We emphasize the importance of interdisciplinary collaboration, urging geophysicists to experiment with AI architectures thoughtfully and encouraging AI experts to deepen their understanding of seismology. By bridging these disciplines, we can develop more accurate, reliable, and societally impactful earthquake forecasting tools. 

**Abstract (ZH)**: 地震预报仍然是一个重大的科学挑战，目前的方法尚无法达到实现有意义的社会效益所需的性能。传统的模型主要基于过去的地震活动和地质力学数据，难以捕捉地震模式的复杂性，常常忽略了有价值但非地震的先兆信号，如地物理、地球化学和大气异常。将如此多样的数据源整合到预报模型中，并结合人工智能技术的发展，为预报模型的进步提供了有希望的途径。特别是深度学习方法，在处理复杂的大规模数据集、识别细微模式和处理多维关系方面表现出色，使其非常适合克服传统方法的局限性。
本文强调了结合人工智能和地物理知识以创建稳健的、基于物理的预报模型的重要性。它探讨了当前的人工智能方法、输入数据类型、损失函数以及模型开发的实际考虑，为地物理学家和人工智能研究人员提供指导。虽然许多基于人工智能的研究简化了地震预测，忽略了数据不平衡和空间时间聚类等关键特征，但将专门的地物理洞察整合到人工智能模型中可以解决这些问题。
我们强调跨学科合作的重要性，敦促地物理学家谨慎地尝试人工智能架构，并鼓励人工智能专家加深对地震学的理解。通过这些学科的结合，我们可以开发出更准确、可靠且具有社会影响力的地震预报工具。 

---
# Mining Social Determinants of Health for Heart Failure Patient 30-Day Readmission via Large Language Model 

**Title (ZH)**: 通过大规模语言模型挖掘心力衰竭患者30天再住院的社会决定因素 

**Authors**: Mingchen Shao, Youjeong Kang, Xiao Hu, Hyunjung Gloria Kwak, Carl Yang, Jiaying Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12158)  

**Abstract**: Heart Failure (HF) affects millions of Americans and leads to high readmission rates, posing significant healthcare challenges. While Social Determinants of Health (SDOH) such as socioeconomic status and housing stability play critical roles in health outcomes, they are often underrepresented in structured EHRs and hidden in unstructured clinical notes. This study leverages advanced large language models (LLMs) to extract SDOHs from clinical text and uses logistic regression to analyze their association with HF readmissions. By identifying key SDOHs (e.g. tobacco usage, limited transportation) linked to readmission risk, this work also offers actionable insights for reducing readmissions and improving patient care. 

**Abstract (ZH)**: 心力衰竭（HF）影响了数百万美国人，并导致较高的再住院率，给医疗保健带来了重大挑战。虽然社会决定因素（SDOH）如社会经济地位和住房稳定性在健康结果中起着关键作用，但它们往往在结构化的电子健康记录（EHRs）中被低估，并隐藏在非结构化的临床笔记中。本研究利用先进的大规模语言模型（LLMs）从临床文本中提取SDOH，并使用逻辑回归分析其与HF再住院之间的关联。通过识别与再住院风险相关的关键SDOH（例如吸烟、有限的交通手段），本研究还为减少再住院和改善患者护理提供了可行的见解。 

---
# Texture Image Synthesis Using Spatial GAN Based on Vision Transformers 

**Title (ZH)**: 基于视觉变换器的 Spatial GAN 在纹理图像合成中的应用 

**Authors**: Elahe Salari, Zohreh Azimifar  

**Link**: [PDF](https://arxiv.org/pdf/2502.01842)  

**Abstract**: Texture synthesis is a fundamental task in computer vision, whose goal is to generate visually realistic and structurally coherent textures for a wide range of applications, from graphics to scientific simulations. While traditional methods like tiling and patch-based techniques often struggle with complex textures, recent advancements in deep learning have transformed this field. In this paper, we propose ViT-SGAN, a new hybrid model that fuses Vision Transformers (ViTs) with a Spatial Generative Adversarial Network (SGAN) to address the limitations of previous methods. By incorporating specialized texture descriptors such as mean-variance (mu, sigma) and textons into the self-attention mechanism of ViTs, our model achieves superior texture synthesis. This approach enhances the model's capacity to capture complex spatial dependencies, leading to improved texture quality that is superior to state-of-the-art models, especially for regular and irregular textures. Comparison experiments with metrics such as FID, IS, SSIM, and LPIPS demonstrate the substantial improvement of ViT-SGAN, which underlines its efficiency in generating diverse realistic textures. 

**Abstract (ZH)**: 纹理合成是计算机视觉中的一个基本任务，其目标是生成在视觉上逼真且结构上连贯的纹理，适用于从图形到科学模拟等多种应用。尽管传统的拼接技术和基于块的纹理合成技术在处理复杂纹理时常常面临挑战，近期深度学习领域的进展已彻底改变了这一领域。本文提出了一种名为ViT-SGAN的新混合模型，将视觉变换器(Vision Transformers, ViTs)与空间生成对抗网络(Spatial Generative Adversarial Network, SGAN)相结合，以克服先前方法的局限性。通过将平均方差（mu, sigma）和纹理单元(textons)等专门的纹理描述符纳入ViTs的自注意力机制中，我们的模型实现了卓越的纹理合成效果。该方法增强了模型捕捉复杂空间依赖性的能力，从而提高了纹理质量，相较于最先进的模型在常规和不规则纹理方面均有显著提升。使用包括FID（Frechet Inception Distance）、IS（Inception Score）、SSIM（Structured Similarity Index Measure）和LPIPS（Learned Perceptual Image Patch Similarity）等指标进行的比较实验，进一步证实了ViT-SGAN在生成多样且真实的纹理方面的高效性。 

---
