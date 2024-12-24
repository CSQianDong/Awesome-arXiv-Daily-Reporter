# LLM-Powered User Simulator for Recommender System 

**Title (ZH)**: 基于LLM的用户模拟器在推荐系统中的应用 

**Authors**: Zijian Zhang, Shuchang Liu, Ziru Liu, Rui Zhong, Qingpeng Cai, Xiangyu Zhao, Chunxu Zhang, Qidong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16984)  

**Abstract**: User simulators can rapidly generate a large volume of timely user behavior data, providing a testing platform for reinforcement learning-based recommender systems, thus accelerating their iteration and optimization. However, prevalent user simulators generally suffer from significant limitations, including the opacity of user preference modeling and the incapability of evaluating simulation accuracy. In this paper, we introduce an LLM-powered user simulator to simulate user engagement with items in an explicit manner, thereby enhancing the efficiency and effectiveness of reinforcement learning-based recommender systems training. Specifically, we identify the explicit logic of user preferences, leverage LLMs to analyze item characteristics and distill user sentiments, and design a logical model to imitate real human engagement. By integrating a statistical model, we further enhance the reliability of the simulation, proposing an ensemble model that synergizes logical and statistical insights for user interaction simulations. Capitalizing on the extensive knowledge and semantic generation capabilities of LLMs, our user simulator faithfully emulates user behaviors and preferences, yielding high-fidelity training data that enrich the training of recommendation algorithms. We establish quantifying and qualifying experiments on five datasets to validate the simulator's effectiveness and stability across various recommendation scenarios. 

**Abstract (ZH)**: 用户模拟器可以快速生成大量及时的用户行为数据，为基于强化学习的推荐系统提供测试平台，从而加速其迭代和优化。然而，现有的用户模拟器通常面临一些重大限制，包括用户偏好建模的透明度不足和对模拟准确性的评估能力有限。在本文中，我们引入了一个基于大语言模型（LLM）的用户模拟器，以显式的方式模拟用户的项目互动，从而提高基于强化学习的推荐系统训练的效率和效果。具体而言，我们识别了用户的显式偏好逻辑，利用大语言模型分析项目特征并提炼用户情绪，并设计了一个逻辑模型来模仿真实的人类互动。通过整合统计模型，我们进一步增强了模拟的可靠性，提出了一种综合逻辑和统计洞见的集成模型，用于用户互动的模拟。凭借大语言模型的广泛知识和语义生成能力，我们的用户模拟器忠实模拟用户行为和偏好，提供高质量的训练数据以丰富推荐算法的训练。我们通过对五个数据集进行量化和定性的实验，验证了模拟器在各种推荐场景中的有效性与稳定性。 

---
# LegalAgentBench: Evaluating LLM Agents in Legal Domain 

**Title (ZH)**: LegalAgentBench：评估法律领域中的语言模型代理 

**Authors**: Haitao Li, Junjie Chen, Jingli Yang, Qingyao Ai, Wei Jia, Youfeng Liu, Kai Lin, Yueyue Wu, Guozhi Yuan, Yiran Hu, Wuyue Wang, Yiqun Liu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17259)  

**Abstract**: With the increasing intelligence and autonomy of LLM agents, their potential applications in the legal domain are becoming increasingly apparent. However, existing general-domain benchmarks cannot fully capture the complexity and subtle nuances of real-world judicial cognition and decision-making. Therefore, we propose LegalAgentBench, a comprehensive benchmark specifically designed to evaluate LLM Agents in the Chinese legal domain. LegalAgentBench includes 17 corpora from real-world legal scenarios and provides 37 tools for interacting with external knowledge. We designed a scalable task construction framework and carefully annotated 300 tasks. These tasks span various types, including multi-hop reasoning and writing, and range across different difficulty levels, effectively reflecting the complexity of real-world legal scenarios. Moreover, beyond evaluating final success, LegalAgentBench incorporates keyword analysis during intermediate processes to calculate progress rates, enabling more fine-grained evaluation. We evaluated eight popular LLMs, highlighting the strengths, limitations, and potential areas for improvement of existing models and methods. LegalAgentBench sets a new benchmark for the practical application of LLMs in the legal domain, with its code and data available at \url{this https URL}. 

**Abstract (ZH)**: 随着大型语言模型（LLM）代理的智能和自主性的增强，它们在法律领域的潜在应用变得愈发明显。然而，现有的通用领域基准无法全面捕捉到现实司法认知和决策中的复杂性和细微差别。因此，我们提出了一个专门针对中国法律领域的基准测试，即LegalAgentBench。LegalAgentBench 包括来自真实法律情境的17个语料库，并提供了37种与外部知识交互的工具。我们设计了一个可扩展的任务构建框架，并仔细标注了300个任务。这些任务涵盖了多种类型，包括多步推理和写作，并涵盖了不同的难度级别，有效地反映了现实法律情境的复杂性。此外，与仅仅评价最终的成功不同，LegalAgentBench 在中间过程中也进行了关键词分析以计算进度率，从而实现更为精细的评价。我们评估了八个流行的LLM，突显了现有模型和方法的优点、局限性和改进潜力。LegalAgentBench 为LLM在法律领域的实践应用设立了新的基准，其代码和数据可在 \url{this https URL} 获取。 

---
# HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases 

**Title (ZH)**: HybGRAG：面向文本和关系知识库的混合检索增强生成 

**Authors**: Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N. Ioannidis, Huzefa Rangwala, Christos Faloutsos  

**Link**: [PDF](https://arxiv.org/pdf/2412.16311)  

**Abstract**: Given a semi-structured knowledge base (SKB), where text documents are interconnected by relations, how can we effectively retrieve relevant information to answer user questions? Retrieval-Augmented Generation (RAG) retrieves documents to assist large language models (LLMs) in question answering; while Graph RAG (GRAG) uses structured knowledge bases as its knowledge source. However, many questions require both textual and relational information from SKB - referred to as "hybrid" questions - which complicates the retrieval process and underscores the need for a hybrid retrieval method that leverages both information. In this paper, through our empirical analysis, we identify key insights that show why existing methods may struggle with hybrid question answering (HQA) over SKB. Based on these insights, we propose HybGRAG for HQA consisting of a retriever bank and a critic module, with the following advantages: (1) Agentic, it automatically refines the output by incorporating feedback from the critic module, (2) Adaptive, it solves hybrid questions requiring both textual and relational information with the retriever bank, (3) Interpretable, it justifies decision making with intuitive refinement path, and (4) Effective, it surpasses all baselines on HQA benchmarks. In experiments on the STaRK benchmark, HybGRAG achieves significant performance gains, with an average relative improvement in Hit@1 of 51%. 

**Abstract (ZH)**: 给定一个半结构化知识库（SKB），其中文本文档通过关系相互连接，如何有效地检索相关信息以回答用户问题？检索增强生成（RAG）通过检索文档来辅助大型语言模型（LLMs）进行问答；而Graph RAG（GRAG）利用结构化知识库作为其知识来源。然而，许多问题需要从SKB中同时获取文本和关系信息，这些被称为“混合”问题，这使得检索过程复杂化，并强调了需要一种结合这两种信息的混合检索方法。在本文中，通过我们的实证分析，我们识别出关键见解，展示了为什么现有方法可能难以处理SKB上的混合问答（HQA）。基于这些见解，我们提出了一种名为HybGRAG的方法来解决HQA，其包含检索库和批判模块，具有以下优势：（1）自主性，通过批判模块的反馈自动优化输出；（2）自适应性，使用检索库解决需要同时处理文本和关系信息的混合问题；（3）可解释性，通过直观的优化路径来解释决策过程；（4）有效性，在HQA基准测试中，HybGRAG取得了显著性能提升，平均改进精度（Hit@1）为51%。

相关实验在STaRK基准测试上验证了HybGRAG的有效性。结果显示，HybGRAG在HQA基准测试中的表现显著优于所有基线方法，平均相对改进精度（Hit@1）达到了51%。 

---
# Automating the Search for Artificial Life with Foundation Models 

**Title (ZH)**: 使用基础模型自动化搜寻人工生命 

**Authors**: Akarsh Kumar, Chris Lu, Louis Kirsch, Yujin Tang, Kenneth O. Stanley, Phillip Isola, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2412.17799)  

**Abstract**: With the recent Nobel Prize awarded for radical advances in protein discovery, foundation models (FMs) for exploring large combinatorial spaces promise to revolutionize many scientific fields. Artificial Life (ALife) has not yet integrated FMs, thus presenting a major opportunity for the field to alleviate the historical burden of relying chiefly on manual design and trial-and-error to discover the configurations of lifelike simulations. This paper presents, for the first time, a successful realization of this opportunity using vision-language FMs. The proposed approach, called Automated Search for Artificial Life (ASAL), (1) finds simulations that produce target phenomena, (2) discovers simulations that generate temporally open-ended novelty, and (3) illuminates an entire space of interestingly diverse simulations. Because of the generality of FMs, ASAL works effectively across a diverse range of ALife substrates including Boids, Particle Life, Game of Life, Lenia, and Neural Cellular Automata. A major result highlighting the potential of this technique is the discovery of previously unseen Lenia and Boids lifeforms, as well as cellular automata that are open-ended like Conway's Game of Life. Additionally, the use of FMs allows for the quantification of previously qualitative phenomena in a human-aligned way. This new paradigm promises to accelerate ALife research beyond what is possible through human ingenuity alone. 

**Abstract (ZH)**: 随着最近因蛋白质发现的重大突破而获得诺贝尔奖，基础模型（FMs）有望重新定义许多科学领域。人工生命（ALife）尚未整合FMs，因此为这一领域提供了一个主要机会，使其能够减轻长期依赖手工设计和试错方法来发现仿生模拟配置的负担。本文首次展示了这一机会的实现，即通过视觉-语言FMs实现了Automated Search for Artificial Life (ASAL)。该方法包括以下几个方面：（1）发现能够生成目标现象的模拟；（2）发现能够生成时间上无限新颖性（开放性）的模拟；（3）揭示一系列有趣的多样化模拟。由于FMs的通用性，ASAL能够在Boids、粒子生命、生命游戏、Lenia和神经细胞自动机等多种ALife平台中有效工作。一个重要结果是展示了该技术的潜力，具体表现在发现了Lenia和Boids中之前未见的生命形式，以及具有类似于生命游戏（Conway's Game of Life）的开放性的细胞自动机。此外，使用FMs使得可以以符合人类视角的方式量化之前定性的现象。这种新的范式有望加速人工生命的研究，超越单纯依靠人类智慧所能实现的水平。 

---
# SMAC-Hard: Enabling Mixed Opponent Strategy Script and Self-play on SMAC 

**Title (ZH)**: SMAC-Hard：启用SMAC中的混合对手策略脚本和自游戏功能 

**Authors**: Yue Deng, Yan Yu, Weiyu Ma, Zirui Wang, Wenhui Zhu, Jian Zhao, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17707)  

**Abstract**: The availability of challenging simulation environments is pivotal for advancing the field of Multi-Agent Reinforcement Learning (MARL). In cooperative MARL settings, the StarCraft Multi-Agent Challenge (SMAC) has gained prominence as a benchmark for algorithms following centralized training with decentralized execution paradigm. However, with continual advancements in SMAC, many algorithms now exhibit near-optimal performance, complicating the evaluation of their true effectiveness. To alleviate this problem, in this work, we highlight a critical issue: the default opponent policy in these environments lacks sufficient diversity, leading MARL algorithms to overfit and exploit unintended vulnerabilities rather than learning robust strategies. To overcome these limitations, we propose SMAC-HARD, a novel benchmark designed to enhance training robustness and evaluation comprehensiveness. SMAC-HARD supports customizable opponent strategies, randomization of adversarial policies, and interfaces for MARL self-play, enabling agents to generalize to varying opponent behaviors and improve model stability. Furthermore, we introduce a black-box testing framework wherein agents are trained without exposure to the edited opponent scripts but are tested against these scripts to evaluate the policy coverage and adaptability of MARL algorithms. We conduct extensive evaluations of widely used and state-of-the-art algorithms on SMAC-HARD, revealing the substantial challenges posed by edited and mixed strategy opponents. Additionally, the black-box strategy tests illustrate the difficulty of transferring learned policies to unseen adversaries. We envision SMAC-HARD as a critical step toward benchmarking the next generation of MARL algorithms, fostering progress in self-play methods for multi-agent systems. Our code is available at this https URL. 

**Abstract (ZH)**: 具有挑战性的模拟环境的存在对于推动多智能体强化学习（MARL）领域的进步至关重要。在合作MARL场景中，星舰争霸多智能体挑战赛（SMAC）因其集中训练与分散执行的范式，已成为评估算法性能的基准。然而，随着SMAC的不断进步，许多算法现在已表现出接近最优的性能，这使得评估其真正效果变得复杂。为解决这一问题，本研究突出了一项关键问题：这些环境中的默认对手策略缺乏足够的多样性，导致MARL算法过度拟合，并利用非预期的漏洞，而不是学习稳健的策略。为克服这些局限性，我们提出了SMAC-HARD，这是一种新的基准，旨在增强训练的健壮性并提高评估的全面性。SMAC-HARD 支持对手策略的可定制性、对手策略的随机化以及MARL自对战接口，使智能体能够适应不同的对手行为并提高模型的稳定性。此外，我们引入了一个黑盒测试框架，在该框架中，智能体在未接触修改后的对手脚本的情况下接受训练，但在测试过程中使用这些脚本评估MARL算法的策略覆盖率和适应性。我们在SMAC-HARD上对广泛使用和最新的MARL算法进行了详尽的评估，揭示了编辑和混合策略对手带来的重大挑战。此外，黑盒策略测试表明，将学到的策略转移到未知对手的难度很大。我们期望SMAC-HARD成为评估下一代MARL算法的重要一步，推动多智能体系统的自对战方法的发展。我们的代码可在以下链接获取：[此处提供链接]。 

---
# PC Agent: While You Sleep, AI Works -- A Cognitive Journey into Digital World 

**Title (ZH)**: PC代理：在你睡眠之时，AI在工作——认知之旅走进数字世界 

**Authors**: Yanheng He, Jiahe Jin, Shijie Xia, Jiadi Su, Runze Fan, Haoyang Zou, Xiangkun Hu, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17589)  

**Abstract**: Imagine a world where AI can handle your work while you sleep - organizing your research materials, drafting a report, or creating a presentation you need for tomorrow. However, while current digital agents can perform simple tasks, they are far from capable of handling the complex real-world work that humans routinely perform. We present PC Agent, an AI system that demonstrates a crucial step toward this vision through human cognition transfer. Our key insight is that the path from executing simple "tasks" to handling complex "work" lies in efficiently capturing and learning from human cognitive processes during computer use. To validate this hypothesis, we introduce three key innovations: (1) PC Tracker, a lightweight infrastructure that efficiently collects high-quality human-computer interaction trajectories with complete cognitive context; (2) a two-stage cognition completion pipeline that transforms raw interaction data into rich cognitive trajectories by completing action semantics and thought processes; and (3) a multi-agent system combining a planning agent for decision-making with a grounding agent for robust visual grounding. Our preliminary experiments in PowerPoint presentation creation reveal that complex digital work capabilities can be achieved with a small amount of high-quality cognitive data - PC Agent, trained on just 133 cognitive trajectories, can handle sophisticated work scenarios involving up to 50 steps across multiple applications. This demonstrates the data efficiency of our approach, highlighting that the key to training capable digital agents lies in collecting human cognitive data. By open-sourcing our complete framework, including the data collection infrastructure and cognition completion methods, we aim to lower the barriers for the research community to develop truly capable digital agents. 

**Abstract (ZH)**: 想象一个世界，在这里AI可以在你睡觉时完成你的工作，比如整理研究资料、起草报告或创建明天需要的演示文稿。然而，尽管当前的数字代理可以处理一些简单的任务，但它们尚不具备处理人类日常执行的复杂工作能力。我们提出了一种名为PC Agent的AI系统，它通过人类认知转移展示了向这一愿景迈进的关键一步。我们的核心见解是，从执行简单的“任务”到处理复杂的“工作”，关键在于有效捕捉和学习人类在使用计算机过程中的认知过程。为了验证这一假设，我们引入了三项关键创新：(1) PC Tracker，一种轻量级的基础设施，能够高效地收集包含完整认知背景的高质量人机交互轨迹；(2) 两阶段认知完成流水线，该流水线将原始交互数据转化为丰富的认知轨迹，通过完成动作语义和思维过程来实现这一目标；(3) 结合决策规划代理和视觉锚定代理的多代理系统。我们初步在PowerPoint演示文稿创建方面的实验表明，只需少量高质量的认知数据即可实现复杂的数字工作能力——PC Agent仅基于133个认知轨迹训练，即可处理涉及多个应用程序并跨越多达50个步骤的复杂工作场景。这证明了我们方法的数据效率，强调了训练有素的数字代理的关键在于收集人类认知数据。通过开源我们的完整框架，包括数据采集基础设施和认知完成方法，我们旨在降低研究社区开发真正有能力建立数字代理的门槛。 

---
# MineAgent: Towards Remote-Sensing Mineral Exploration with Multimodal Large Language Models 

**Title (ZH)**: MineAgent：面向多模态大规模语言模型的遥感矿产勘探研究 

**Authors**: Beibei Yu, Tao Shen, Hongbin Na, Ling Chen, Denqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.17339)  

**Abstract**: Remote-sensing mineral exploration is critical for identifying economically viable mineral deposits, yet it poses significant challenges for multimodal large language models (MLLMs). These include limitations in domain-specific geological knowledge and difficulties in reasoning across multiple remote-sensing images, further exacerbating long-context issues. To address these, we present MineAgent, a modular framework leveraging hierarchical judging and decision-making modules to improve multi-image reasoning and spatial-spectral integration. Complementing this, we propose MineBench, a benchmark specific for evaluating MLLMs in domain-specific mineral exploration tasks using geological and hyperspectral data. Extensive experiments demonstrate the effectiveness of MineAgent, highlighting its potential to advance MLLMs in remote-sensing mineral exploration. 

**Abstract (ZH)**: 遥感矿物勘探对于识别具有经济开采价值的矿床至关重要，但对多模态大型语言模型（MLLMs）提出了重大挑战。这些挑战包括领域特定地质知识的局限性以及在多张遥感图像间进行推理的难度，进一步加剧了长上下文问题。为解决这些问题，我们提出了MineAgent，这是一种模块化框架，利用层次化的判断和决策模块以提高多图像推理和谱-空域集成的效果。为配合这一框架，我们还提出了MineBench，这是一个特定于地质和高光谱数据的基准，用于评估MLLMs在特定领域的矿物勘探任务中的性能。广泛的实验表明，MineAgent的有效性，并突显了其在遥感矿物勘探中推进MLLMs的潜力。 

---
# B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners 

**Title (ZH)**: B-STaR：自我教学推理者中探索与利用的监控和平衡 

**Authors**: Weihao Zeng, Yuzhen Huang, Lulu Zhao, Yijun Wang, Zifei Shan, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2412.17256)  

**Abstract**: In the absence of extensive human-annotated data for complex reasoning tasks, self-improvement -- where models are trained on their own outputs -- has emerged as a primary method for enhancing performance. However, the critical factors underlying the mechanism of these iterative self-improving methods remain poorly understood, such as under what conditions self-improvement is effective, and what are the bottlenecks in the current iterations. In this work, we identify and propose methods to monitor two pivotal factors in this iterative process: (1) the model's ability to generate sufficiently diverse responses (exploration); and (2) the effectiveness of external rewards in distinguishing high-quality candidates from lower-quality ones (exploitation). Using mathematical reasoning as a case study, we begin with a quantitative analysis to track the dynamics of exploration and exploitation, discovering that a model's exploratory capabilities rapidly deteriorate over iterations, and the effectiveness of exploiting external rewards diminishes as well. Motivated by these findings, we introduce B-STaR, a Self-Taught Reasoning framework that autonomously adjusts configurations across iterations to Balance exploration and exploitation, thereby optimizing the self-improving effectiveness based on the current policy model and available rewards. Our experiments on mathematical reasoning, coding, and commonsense reasoning demonstrate that B-STaR not only enhances the model's exploratory capabilities throughout training but also achieves a more effective balance between exploration and exploitation, leading to superior performance. 

**Abstract (ZH)**: 在缺乏大量人工标注数据的情况下，为了复杂推理任务，模型通过自我提升（即模型在其自身输出上进行训练）已成为提高性能的主要方法。然而，这些迭代自我提升机制的核心因素仍然知之甚少，例如在什么条件下自我提升有效，当前迭代中的瓶颈又是什么。在这项工作中，我们识别并提出了监测这一迭代过程中的两个关键因素的方法：（1）模型生成足够多样化响应的能力（探索）；（2）外部奖励在区分高质量候选对象与低质量候选对象方面的有效性（利用）。用数学推理作为案例研究，我们首先进行定量分析以跟踪探索和利用的动力学，发现模型的探索能力在迭代中迅速下降，利用外部奖励的有效性也在减弱。基于这些发现，我们引入了B-STaR（自学习推理）框架，该框架在迭代过程中自主调整配置，平衡探索与利用，从而根据当前策略模型和可用奖励优化自我提升的有效性。我们在数学推理、编程和常识推理的实验中表明，B-STaR不仅能增强模型在整个训练过程中的探索能力，还能更有效地平衡探索与利用，从而提高性能。 

---
# LLM Agent for Fire Dynamics Simulations 

**Title (ZH)**: 用于火灾动力学模拟的大型语言模型代理 

**Authors**: Leidong Xu, Danyal Mohaddes, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17146)  

**Abstract**: Significant advances have been achieved in leveraging foundation models, such as large language models (LLMs), to accelerate complex scientific workflows. In this work we introduce FoamPilot, a proof-of-concept LLM agent designed to enhance the usability of FireFOAM, a specialized solver for fire dynamics and fire suppression simulations built using OpenFOAM, a popular open-source toolbox for computational fluid dynamics (CFD). FoamPilot provides three core functionalities: code insight, case configuration and simulation evaluation. Code insight is an alternative to traditional keyword searching leveraging retrieval-augmented generation (RAG) and aims to enable efficient navigation and summarization of the FireFOAM source code for developers and experienced users. For case configuration, the agent interprets user requests in natural language and aims to modify existing simulation setups accordingly to support intermediate users. FoamPilot's job execution functionality seeks to manage the submission and execution of simulations in high-performance computing (HPC) environments and provide preliminary analysis of simulation results to support less experienced users. Promising results were achieved for each functionality, particularly for simple tasks, and opportunities were identified for significant further improvement for more complex tasks. The integration of these functionalities into a single LLM agent is a step aimed at accelerating the simulation workflow for engineers and scientists employing FireFOAM for complex simulations critical for improving fire safety. 

**Abstract (ZH)**: 在利用大型语言模型（LLM）等基础模型加速复杂科学工作流方面取得了显著进展。本文介绍了FoamPilot，这是一种概念验证的LLM代理，旨在提升FireFOAM的易用性，FireFOAM是一个基于OpenFOAM构建的专用求解器，用于火灾动力学和灭火模拟。OpenFOAM是一个流行的开源计算流体动力学（CFD）工具箱。FoamPilot提供了三个核心功能：代码洞察、案例配置和仿真评估。代码洞察利用检索增强生成（RAG）作为一种替代传统的关键词搜索的方法，旨在使开发人员和有经验的用户能够高效地导航和总结FireFOAM的源代码。在案例配置方面，代理以自然语言解释用户请求，并旨在相应地修改现有的仿真设置，以支持中级用户。FoamPilot的任务执行功能旨在管理仿真在高性能计算（HPC）环境中的提交和执行，并对仿真结果进行初步分析，以支持经验较少的用户。对于每个功能，特别是简单任务，我们取得了令人鼓舞的结果，并且识别出了在更复杂任务上进行重大改进的机会。将这些功能集成到一个单一的LLM代理中，是一个旨在加速使用FireFOAM进行复杂仿真（这对于提高火灾安全性至关重要）的工程师和科学家的仿真工作流的步骤。 

---
# On the ETHOS of AI Agents: An Ethical Technology and Holistic Oversight System 

**Title (ZH)**: AI代理的伦理精神：一种全面监督的伦理技术体系 

**Authors**: Tomer Jordi Chaffer, Justin Goldston, Bayo Okusanya, Gemach D.A.T.A.I  

**Link**: [PDF](https://arxiv.org/pdf/2412.17114)  

**Abstract**: In a world increasingly defined by machine intelligence, the future depends on how we govern the development and integration of AI into society. Recent initiatives, such as the EU AI Act, EDPB opinion, U.S. Bipartisan House Task Force and NIST AI Risk Management Report, highlight the urgent need for robust governance frameworks to address the challenges posed by advancing AI technologies. However, existing frameworks fail to adequately address the rise of AI agents or the ongoing debate between centralized and decentralized governance models. To bridge these gaps, we propose the Ethical Technology and Holistic Oversight System framework, which leverages Web3 technologies, including blockchain, smart contracts, decentralized autonomous organizations, and soulbound tokens, to establish a decentralized global registry for AI agents. ETHOS incorporates the concept of AI specific legal entities, enabling these systems to assume limited liability and ensuring accountability through mechanisms like insurance and compliance monitoring. Additionally, the framework emphasizes the need for a collaborative, participatory approach to AI governance, engaging diverse stakeholders through public education, transparency, and international coordination. ETHOS balances innovation with ethical accountability, providing a forward looking strategy for the responsible integration of AI agents into society. Finally, this exploration reflects the emergence of a new interdisciplinary field we define as Systems Thinking at the Intersection of AI, Web3, and Society. 

**Abstract (ZH)**: 在日益由机器智能定义的世界中，未来取决于我们如何治理AI在社会中的发展和整合。近期的举措，如欧盟AI法案、EDPB意见、美国两党国会任务小组和NIST AI风险管理报告，凸显了建立坚实治理框架的迫切需求，以应对先进AI技术带来的挑战。然而，现有的框架未能充分应对AI代理的崛起，以及集中式与去中心化治理模式之间的持续辩论。为了填补这些空白，我们提出了一种伦理技术和综合监督系统的框架（Ethical Technology and Holistic Oversight System, ETHOS），该框架利用Web3技术，包括区块链、智能合约、去中心化自治组织和绑定灵魂代币，建立一个去中心化的全球AI代理注册系统。ETHOS引入了特定于AI的法律实体概念，使这些系统能够承担有限责任，并通过保险和合规监控等机制确保问责制。此外，该框架强调采用合作、参与的方式进行AI治理，通过公众教育、透明度和国际协调，吸引多元化的利益相关者。ETHOS在促进AI代理负责任地融入社会的同时平衡了创新与伦理问责，提供了一个前瞻性的策略。最后，本文探讨了一个新的跨学科领域——AI、Web3和社会交汇处的系统思维，我们将其定义为系统思维。 

---
# SubstationAI: Multimodal Large Model-Based Approaches for Analyzing Substation Equipment Faults 

**Title (ZH)**: SubstationAI：基于多模态大规模模型的方法用于分析变电站设备故障 

**Authors**: Jinzhi Wang, Qinfeng Song, Lidong Qian, Haozhou Li, Qinke Peng, Jiangbo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17077)  

**Abstract**: The reliability of substation equipment is crucial to the stability of power systems, but traditional fault analysis methods heavily rely on manual expertise, limiting their effectiveness in handling complex and large-scale data. This paper proposes a substation equipment fault analysis method based on a multimodal large language model (MLLM). We developed a database containing 40,000 entries, including images, defect labels, and analysis reports, and used an image-to-video generation model for data augmentation. Detailed fault analysis reports were generated using GPT-4. Based on this database, we developed SubstationAI, the first model dedicated to substation fault analysis, and designed a fault diagnosis knowledge base along with knowledge enhancement methods. Experimental results show that SubstationAI significantly outperforms existing models, such as GPT-4, across various evaluation metrics, demonstrating higher accuracy and practicality in fault cause analysis, repair suggestions, and preventive measures, providing a more advanced solution for substation equipment fault analysis. 

**Abstract (ZH)**: 变电站设备的可靠性对于电力系统稳定性至关重要，但传统故障分析方法严重依赖人工专业知识，限制了其在处理复杂和大规模数据方面的有效性。本文提出了一种基于多模态大语言模型（MLLM）的变电站设备故障分析方法。我们建立了一个包含40,000条记录的数据库，包括图像、缺陷标签和分析报告，并使用图像到视频生成模型进行数据增强。使用GPT-4生成了详细故障分析报告。基于该数据库，我们开发了SubstationAI——第一个专用于变电站故障分析的模型，并设计了故障诊断知识库及知识增强方法。实验结果表明，SubstationAI在各种评估指标上显著优于现有的模型（如GPT-4），在故障原因分析、维修建议和预防措施方面展现出更高的准确性和实用性，为变电站设备故障分析提供了更高级的解决方案。 

---
# GraphAgent: Agentic Graph Language Assistant 

**Title (ZH)**: GraphAgent：自主图语言助手 

**Authors**: Yuhao Yang, Jiabin Tang, Lianghao Xia, Xingchen Zou, Yuxuan Liang, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17029)  

**Abstract**: Real-world data is represented in both structured (e.g., graph connections) and unstructured (e.g., textual, visual information) formats, encompassing complex relationships that include explicit links (such as social connections and user behaviors) and implicit interdependencies among semantic entities, often illustrated through knowledge graphs. In this work, we propose GraphAgent, an automated agent pipeline that addresses both explicit graph dependencies and implicit graph-enhanced semantic inter-dependencies, aligning with practical data scenarios for predictive tasks (e.g., node classification) and generative tasks (e.g., text generation). GraphAgent comprises three key components: (i) a Graph Generator Agent that builds knowledge graphs to reflect complex semantic dependencies; (ii) a Task Planning Agent that interprets diverse user queries and formulates corresponding tasks through agentic self-planning; and (iii) a Task Execution Agent that efficiently executes planned tasks while automating tool matching and invocation in response to user queries. These agents collaborate seamlessly, integrating language models with graph language models to uncover intricate relational information and data semantic dependencies. Through extensive experiments on various graph-related predictive and text generative tasks on diverse datasets, we demonstrate the effectiveness of our GraphAgent across various settings. We have made our proposed GraphAgent open-source at: this https URL. 

**Abstract (ZH)**: 实际数据以结构化（例如，图形连接）和非结构化（例如，文本和视觉信息）的形式表示，涵盖了复杂的相互关系，包括显式的链接（如社会连接和用户行为）以及语义实体之间的隐式相互依赖，这些通常通过知识图谱来描绘。本文提出了一种自动代理管道——GraphAgent，同时解决了显式的图形依赖和隐式的图形增强语义依赖，适用于各类实际数据情景下的预测任务（例如，节点分类）以及生成任务（例如，文本生成）。GraphAgent 包含三个关键组件：(i) 图生成代理，用于构建知识图谱以反映复杂的语义依赖；(ii) 任务规划代理，通过自主规划解释多种用户查询并制定相应的任务；以及(iii) 任务执行代理，高效执行规划的任务，同时根据用户查询自动匹配和调用工具。这些代理之间无缝协作，将语言模型与图形语言模型结合，揭示复杂的关联信息和数据语义依赖。通过在各种图相关的预测和文本生成任务中对不同数据集进行广泛的实验，我们展示了GraphAgent在各种应用场景下的有效性。我们已将提议的GraphAgent开源于：[此处插入链接]。 

---
# KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis 

**Title (ZH)**: KG4Diagnosis：一种带有知识图谱增强的分层多代理大语言模型框架用于医疗诊断 

**Authors**: Kaiwen Zuo, Yirui Jiang, Fan Mo, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2412.16833)  

**Abstract**: Integrating Large Language Models (LLMs) in healthcare diagnosis demands systematic frameworks that can handle complex medical scenarios while maintaining specialized expertise. We present KG4Diagnosis, a novel hierarchical multi-agent framework that combines LLMs with automated knowledge graph construction, encompassing 362 common diseases across medical specialties. Our framework mirrors real-world medical systems through a two-tier architecture: a general practitioner (GP) agent for initial assessment and triage, coordinating with specialized agents for in-depth diagnosis in specific domains. The core innovation lies in our end-to-end knowledge graph generation methodology, incorporating: (1) semantic-driven entity and relation extraction optimized for medical terminology, (2) multi-dimensional decision relationship reconstruction from unstructured medical texts, and (3) human-guided reasoning for knowledge expansion. KG4Diagnosis serves as an extensible foundation for specialized medical diagnosis systems, with capabilities to incorporate new diseases and medical knowledge. The framework's modular design enables seamless integration of domain-specific enhancements, making it valuable for developing targeted medical diagnosis systems. We provide architectural guidelines and protocols to facilitate adoption across medical contexts. 

**Abstract (ZH)**: 将大型语言模型（LLMs）集成到医疗诊断中，需要系统性的框架来处理复杂的医疗场景并保持专业化的专业知识。我们提出了KG4Diagnosis，这是一种新颖的分层多智能体框架，将LLMs与自动知识图谱构建相结合，涵盖了362种常见疾病，涉及多个医学领域。该框架通过二层架构模拟现实世界的医疗系统：初级诊断（GP）智能体负责初始评估和分流，协调特定领域的专门智能体进行深入诊断。核心创新在于我们端到端的知识图谱生成方法，包括：（1）基于语义的实体和关系抽取，优化用于医学术语，（2）从非结构化医学文本中重构多维度的决策关系，（3）人引导的推理以扩展知识。KG4Diagnosis 作为一个可扩展的基础框架，具有纳入新疾病和医学知识的能力。该框架的模块化设计使其能够无缝集成特定领域的增强功能，使其适用于开发针对性的医疗诊断系统。我们提供了构架指南和协议，以便在不同医疗场景中促进其采纳。 

---
# On Enhancing Network Throughput using Reinforcement Learning in Sliced Testbeds 

**Title (ZH)**: 在分片测试床中使用强化学习提高网络吞吐量的研究 

**Authors**: Daniel Pereira Monteiro, Lucas Nardelli de Freitas Botelho Saar, Larissa Ferreira Rodrigues Moreira, Rodrigo Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2412.16673)  

**Abstract**: Novel applications demand high throughput, low latency, and high reliability connectivity and still pose significant challenges to slicing orchestration architectures. The literature explores network slicing techniques that employ canonical methods, artificial intelligence, and combinatorial optimization to address errors and ensure throughput for network slice data plane. This paper introduces the Enhanced Mobile Broadband (eMBB)-Agent as a new approach that uses Reinforcement Learning (RL) in a vertical application to enhance network slicing throughput to fit Service-Level Agreements (SLAs). The eMBB-Agent analyzes application transmission variables and proposes actions within a discrete space to adjust the reception window using a Deep Q-Network (DQN). This paper also presents experimental results that examine the impact of factors such as the channel error rate, DQN model layers, and learning rate on model convergence and achieved throughput, providing insights on embedding intelligence in network slicing. 

**Abstract (ZH)**: 新型应用对高吞吐量、低时延和高可靠性的连接提出了新的需求，这给网络切片编排架构带来了巨大挑战。文献中探讨了利用经典方法、人工智能和组合优化的网络切片技术，以解决错误并确保网络切片子网数据平面的吞吐量。本文介绍了一种新的方法——增强型移动宽带（eMBB）代理，该方法在垂直应用中采用强化学习（Reinforcement Learning, RL）来提高网络切片吞吐量，以满足服务级别协议（Service-Level Agreements, SLAs）的要求。eMBB代理分析应用程序传输变量，并在离散空间中提出动作，利用深度Q网络（Deep Q-Network, DQN）调整接收窗口。本文还介绍了实验结果，这些结果探讨了信道错误率、DQN模型层和学习率等因素对模型收敛和吞吐量实现的影响，提供了将智能嵌入到网络切片中的见解。 

---
# Metagoals Endowing Self-Modifying AGI Systems with Goal Stability or Moderated Goal Evolution: Toward a Formally Sound and Practical Approach 

**Title (ZH)**: 元目标：赋予自修改AGI系统目标稳定性或可控目标 evolution 的方法：一种形式上稳健且实用的途径 

**Authors**: Ben Goertzel  

**Link**: [PDF](https://arxiv.org/pdf/2412.16559)  

**Abstract**: We articulate here a series of specific metagoals designed to address the challenge of creating AGI systems that possess the ability to flexibly self-modify yet also have the propensity to maintain key invariant properties of their goal systems
1) a series of goal-stability metagoals aimed to guide a system to a condition in which goal-stability is compatible with reasonably flexible self-modification
2) a series of moderated-goal-evolution metagoals aimed to guide a system to a condition in which control of the pace of goal evolution is compatible with reasonably flexible self-modification
The formulation of the metagoals is founded on fixed-point theorems from functional analysis, e.g. the Contraction Mapping Theorem and constructive approximations to Schauder's Theorem, applied to probabilistic models of system behavior
We present an argument that the balancing of self-modification with maintenance of goal invariants will often have other interesting cognitive side-effects such as a high degree of self understanding
Finally we argue for the practical value of a hybrid metagoal combining moderated-goal-evolution with pursuit of goal-stability -- along with potentially other metagoals relating to goal-satisfaction, survival and ongoing development -- in a flexible fashion depending on the situation 

**Abstract (ZH)**: 以下是论文内容或标题的中文翻译，符合学术规范：

本文旨在概述一系列具体的元目标，以应对创建具备灵活自我修改能力但又能够保持关键目标系统不变性的AGI系统这一挑战：
1) 一系列目标稳定性的元目标，旨在引导系统处于目标稳定性和合理灵活的自我修改可以兼容的状态；
2) 一系列限速目标演化的元目标，旨在引导系统处于控制目标演化速度与合理灵活的自我修改可以兼容的状态。

这些元目标的制定基于功能分析中的不动点定理，如收缩映射定理及其对Schauder定理的构造性逼近方法，在概率模型的系统行为中得以应用。

我们提出了一项论断，即平衡自我修改与保持目标不变性的关系往往会产生其他有趣的认知副作用，如高度的自我理解。

最后，我们强调了将限速目标演化与追求目标稳定性的混合元目标（以及其他与目标实现、生存和持续发展相关的元目标）相结合的实用价值。这种结合方法应根据具体情况进行灵活运用。 

---
# ActPC-Chem: Discrete Active Predictive Coding for Goal-Guided Algorithmic Chemistry as a Potential Cognitive Kernel for Hyperon & PRIMUS-Based AGI 

**Title (ZH)**: ActPC-Chem: 离散活性预测编码在高能&PRIMUS基于的超人工智能目标导向算法化学中的潜在认知内核 

**Authors**: Ben Goertzel  

**Link**: [PDF](https://arxiv.org/pdf/2412.16547)  

**Abstract**: We explore a novel paradigm (labeled ActPC-Chem) for biologically inspired, goal-guided artificial intelligence (AI) centered on a form of Discrete Active Predictive Coding (ActPC) operating within an algorithmic chemistry of rewrite rules. ActPC-Chem is envisioned as a foundational "cognitive kernel" for advanced cognitive architectures, such as the OpenCog Hyperon system, incorporating essential elements of the PRIMUS cognitive architecture. The central thesis is that general-intelligence-capable cognitive structures and dynamics can emerge in a system where both data and models are represented as evolving patterns of metagraph rewrite rules, and where prediction errors, intrinsic and extrinsic rewards, and semantic constraints guide the continual reorganization and refinement of these rules. Using a virtual "robot bug" thought experiment, we illustrate how such a system might self-organize to handle challenging tasks involving delayed and context-dependent rewards, integrating causal rule inference (AIRIS) and probabilistic logical abstraction (PLN) to discover and exploit conceptual patterns and causal constraints. Next, we describe how continuous predictive coding neural networks, which excel at handling noisy sensory data and motor control signals, can be coherently merged with the discrete ActPC substrate. Finally, we outline how these ideas might be extended to create a transformer-like architecture that foregoes traditional backpropagation in favor of rule-based transformations guided by ActPC. This layered architecture, supplemented with AIRIS and PLN, promises structured, multi-modal, and logically consistent next-token predictions and narrative sequences. 

**Abstract (ZH)**: 我们探索了一种新型范式（标记为ActPC-Chem），该范式以生物学启发、目标导向的人工智能（AI）为核心，基于一种形式的离散主动预测编码（ActPC），该编码在重写规则的算法化学中运行。ActPC-Chem 想象为高级认知架构（如 OpenCog Hyperon 系统）的基础“认知内核”，并融合了 PRIMUS 认知架构的关键要素。核心论点是，在一个数据和模型均表示为元图重写规则演变模式的系统中，预测错误、内在和外在奖励以及语义约束可以指导这些规则的持续重组与完善，从而促成通用智能能力的认知结构和动力学的涌现。我们通过一个虚拟的“机器人虫子”思想实验，说明这种系统如何自我组织以处理涉及延迟和上下文相关奖励的挑战性任务，利用因果规则推理 (AIRIS) 和概率逻辑抽象 (PLN) 发现和利用概念模式与因果约束。接下来，我们描述了如何将擅长处理嘈杂感官数据和运动控制信号的连续预测编码神经网络，与离散的 ActPC 子结构协调地结合在一起。最后，我们概述了如何将这些想法扩展为一种类似于Transformer的架构，该架构放弃传统的反向传播，而采用由ActPC指导的基于规则的转换。这种分层架构，结合AIRIS和PLN，承诺在结构化、多模态和逻辑一致的下一个标记预测和叙事序列方面提供支持。 

---
# Autonomous Option Invention for Continual Hierarchical Reinforcement Learning and Planning 

**Title (ZH)**: 连续层次强化学习与规划中的自主选项发明 

**Authors**: Rashmeet Kaur Nayyar, Siddharth Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2412.16395)  

**Abstract**: Abstraction is key to scaling up reinforcement learning (RL). However, autonomously learning abstract state and action representations to enable transfer and generalization remains a challenging open problem. This paper presents a novel approach for inventing, representing, and utilizing options, which represent temporally extended behaviors, in continual RL settings. Our approach addresses streams of stochastic problems characterized by long horizons, sparse rewards, and unknown transition and reward functions.
Our approach continually learns and maintains an interpretable state abstraction, and uses it to invent high-level options with abstract symbolic representations. These options meet three key desiderata: (1) composability for solving tasks effectively with lookahead planning, (2) reusability across problem instances for minimizing the need for relearning, and (3) mutual independence for reducing interference among options. Our main contributions are approaches for continually learning transferable, generalizable options with symbolic representations, and for integrating search techniques with RL to efficiently plan over these learned options to solve new problems. Empirical results demonstrate that the resulting approach effectively learns and transfers abstract knowledge across problem instances, achieving superior sample efficiency compared to state-of-the-art methods. 

**Abstract (ZH)**: 抽象是提高强化学习（RL）性能的关键。然而，自主学习能够实现迁移和泛化的抽象状态和动作表示仍然是一个具有挑战性的开放问题。本文提出了一种新颖的方法，用于在持续强化学习环境中发明、表示和利用选项，这些选项代表了时间上延伸的行为。我们的方法针对由长时间期、稀疏奖励以及未知转移和奖励函数组成的随机问题流进行学习和维护可解释的状态抽象，并利用这种抽象来发明具有抽象符号表示的高级选项。这些选项满足三个关键要求：（1）组合性，以便在前瞻规划中有效解决任务；（2）跨问题实例的重用性，以减少重新学习的需求；（3）互不干扰性，以减少选项之间的相互干扰。我们的主要贡献在于提出了持续学习可迁移、可泛化的具有符号表示的选项的方法，并将搜索技术与RL结合起来，高效地通过这些学习到的选项进行规划以解决新问题。实证结果表明，这种方法能够有效地学习和转移抽象知识，并在解决问题实例时表现出优于现有最先进的方法的样本效率。 

---
# Towards Safe and Honest AI Agents with Neural Self-Other Overlap 

**Title (ZH)**: 朝着具备神经自我-他者重叠的安全可靠人工智能代理的研究 

**Authors**: Marc Carauleanu, Michael Vaiana, Judd Rosenblatt, Cameron Berg, Diogo Schwerz de Lucena  

**Link**: [PDF](https://arxiv.org/pdf/2412.16325)  

**Abstract**: As AI systems increasingly make critical decisions, deceptive AI poses a significant challenge to trust and safety. We present Self-Other Overlap (SOO) fine-tuning, a promising approach in AI Safety that could substantially improve our ability to build honest artificial intelligence. Inspired by cognitive neuroscience research on empathy, SOO aims to align how AI models represent themselves and others. Our experiments on LLMs with 7B, 27B, and 78B parameters demonstrate SOO's efficacy: deceptive responses of Mistral-7B-Instruct-v0.2 dropped from 73.6% to 17.2% with no observed reduction in general task performance, while in Gemma-2-27b-it and CalmeRys-78B-Orpo-v0.1 deceptive responses were reduced from 100% to 9.3% and 2.7%, respectively, with a small impact on capabilities. In reinforcement learning scenarios, SOO-trained agents showed significantly reduced deceptive behavior. SOO's focus on contrastive self and other-referencing observations offers strong potential for generalization across AI architectures. While current applications focus on language models and simple RL environments, SOO could pave the way for more trustworthy AI in broader domains. Ethical implications and long-term effects warrant further investigation, but SOO represents a significant step forward in AI safety research. 

**Abstract (ZH)**: 随着AI系统在越来越多的关键决策中发挥作用，欺骗性AI对信任和安全构成了重大挑战。我们提出了自我-他人重叠（SOO）微调，这是一种在AI安全领域具有前景的方法，有望显著提高我们构建诚实的人工智能的能力。受认知神经科学研究中同理心机制的启发，SOO旨在使AI模型如何表示自己与他人保持一致。我们在参数量分别为7B、27B和78B的大型语言模型（LLM）上的实验表明，SOO的有效性：Mistral-7B-Instruct-v0.2的欺骗性回应从73.6%降低到17.2%，没有观察到一般任务性能的下降；而在Gemma-2-27b-it和CalmeRys-78B-Orpo-v0.1中，欺骗性回应分别从100%降低到9.3%和2.7%，尽管对能力的影响较小。在强化学习场景中，SOO训练的代理表现出显著减少的欺骗行为。SOO强调对比自我和他人的观察，为跨AI架构的一般化提供了强有力的可能性。尽管目前的应用主要集中在语言模型和简单的RL环境中，SOO为更广泛的领域中构建更值得信赖的AI提供了可能的道路。伦理影响和长期效果需要进一步研究，但SOO代表了AI安全研究中一个重要的进步。 

---
# Autoware.Flex: Human-Instructed Dynamically Reconfigurable Autonomous Driving Systems 

**Title (ZH)**: Autobatch.Flex：基于人类指令的动态可重构自动驾驶系统 

**Authors**: Ziwei Song, Mingsong Lv, Tianchi Ren, Chun Jason Xue, Jen-Ming Wu, Nan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2412.16265)  

**Abstract**: Existing Autonomous Driving Systems (ADS) independently make driving decisions, but they face two significant limitations. First, in complex scenarios, ADS may misinterpret the environment and make inappropriate driving decisions. Second, these systems are unable to incorporate human driving preferences in their decision-making processes. This paper proposes this http URL, a novel ADS system that incorporates human input into the driving process, allowing users to guide the ADS in making more appropriate decisions and ensuring their preferences are satisfied. Achieving this needs to address two key challenges: (1) translating human instructions, expressed in natural language, into a format the ADS can understand, and (2) ensuring these instructions are executed safely and consistently within the ADS' s decision-making framework. For the first challenge, we employ a Large Language Model (LLM) assisted by an ADS-specialized knowledge base to enhance domain-specific translation. For the second challenge, we design a validation mechanism to ensure that human instructions result in safe and consistent driving behavior. Experiments conducted on both simulators and a real-world autonomous vehicle demonstrate that this http URL effectively interprets human instructions and executes them safely. 

**Abstract (ZH)**: 现有的自动驾驶系统（ADS）能够独立做出驾驶决策，但在复杂场景下可能误解环境并作出不适当的决策。此外，这些系统无法在其决策过程中融入用户驾驶偏好。本文提出了一种名为“this http URL”的新型ADS系统，该系统能够将人类输入纳入驾驶过程，使用户能够指导ADS做出更加合适的决策，确保其偏好得到满足。实现这一点需要解决两个关键挑战：（1）将用自然语言表达的人类指令转换为ADS能够理解的格式，（2）确保这些指令在ADS的决策框架中得到安全和一致的执行。为了应对第一个挑战，我们采用了大型语言模型（LLM）辅助以专门的ADS知识库的方法，以增强领域特定的翻译能力。为了解决第二个挑战，我们设计了一个验证机制，以确保人类指令能够产生安全和一致的驾驶行为。在模拟器和实际自动驾驶车辆上的实验结果表明，“this http URL”能够有效地解释人类指令并安全执行这些指令。 

---
# Optimizing Low-Speed Autonomous Driving: A Reinforcement Learning Approach to Route Stability and Maximum Speed 

**Title (ZH)**: 优化低速自动驾驶：基于强化学习的路线稳定性和最高速度方法 

**Authors**: Benny Bao-Sheng Li, Elena Wu, Hins Shao-Xuan Yang, Nicky Yao-Jin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16248)  

**Abstract**: Autonomous driving has garnered significant attention in recent years, especially in optimizing vehicle performance under varying conditions. This paper addresses the challenge of maintaining maximum speed stability in low-speed autonomous driving while following a predefined route. Leveraging reinforcement learning (RL), we propose a novel approach to optimize driving policies that enable the vehicle to achieve near-maximum speed without compromising on safety or route accuracy, even in low-speed scenarios. 

**Abstract (ZH)**: 近年来，自主驾驶技术引起了广泛关注，尤其是在不同条件下优化车辆性能方面。本文针对低速自主驾驶沿预设路径行驶时保持最大速度稳定性的挑战，提出了一种新的方法。借助强化学习（Reinforcement Learning, RL），我们提出了一种优化驾驶策略的方法，使车辆能够在不牺牲安全性和路径准确性的情况下，实现接近最大速度，即使在低速场景下也是如此。 

---
# Neural diversity is key to collective artificial learning 

**Title (ZH)**: 神经多样性是集体人工学习的关键 

**Authors**: Matteo Bettini, Ryan Kortvelesy, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2412.16244)  

**Abstract**: Many of the world's most pressing issues, such as climate change and global peace, require complex collective problem-solving skills. Recent studies indicate that diversity in individuals' behaviors is key to developing such skills and increasing collective performance. Yet behavioral diversity in collective artificial learning is understudied, with today's machine learning paradigms commonly favoring homogeneous agent strategies over heterogeneous ones, mainly due to computational considerations. In this work, we employ novel diversity measurement and control paradigms to study the impact of behavioral heterogeneity in several facets of collective artificial learning. Through experiments in team play and other cooperative tasks, we show the emergence of unbiased behavioral roles that improve team outcomes; how neural diversity synergizes with morphological diversity; how diverse agents are more effective at finding cooperative solutions in sparse reward settings; and how behaviorally heterogeneous teams learn and retain latent skills to overcome repeated disruptions. Overall, our results indicate that, by controlling diversity, we can obtain non-trivial benefits over homogeneous training paradigms, demonstrating that diversity is a fundamental component of collective artificial learning, an insight thus far overlooked. 

**Abstract (ZH)**: 世界许多紧迫的问题，如气候变化和全球和平，需要复杂的集体问题解决能力。近期研究表明，个体行为的多样性是发展此类技能和提高集体绩效的关键。然而，在集体人工智能学习中，行为多样性的研究不足，当前的机器学习范式往往偏好同质化的代理策略，而非异质化的策略，主要原因在于计算考虑。在本工作中，我们采用新型的多样性测量和控制范式，探讨行为异质性在集体人工智能学习各个方面的影响力。通过团队游戏和其他协作任务的实验，我们展示了无偏见的行为角色如何改善团队成果；神经多样性与形态多样性如何协同作用；在稀疏奖励环境中，多样化代理如何更有效地找到合作解决方案；以及行为异质性团队如何学习和保留潜在技能以克服反复的中断。总体而言，我们的结果表明，通过控制多样性，我们可以在同质化训练范式中获得非平凡的利益，证明了多样性是集体人工智能学习的基本要素，而这一点此前尚未被充分认识到。 

---
# Agents Are Not Enough 

**Title (ZH)**: 代理不足以胜任 

**Authors**: Chirag Shah, Ryen W. White  

**Link**: [PDF](https://arxiv.org/pdf/2412.16241)  

**Abstract**: In the midst of the growing integration of Artificial Intelligence (AI) into various aspects of our lives, agents are experiencing a resurgence. These autonomous programs that act on behalf of humans are neither new nor exclusive to the mainstream AI movement. By exploring past incarnations of agents, we can understand what has been done previously, what worked, and more importantly, what did not pan out and why. This understanding lets us to examine what distinguishes the current focus on agents. While generative AI is appealing, this technology alone is insufficient to make new generations of agents more successful. To make the current wave of agents effective and sustainable, we envision an ecosystem that includes not only agents but also Sims, which represent user preferences and behaviors, as well as Assistants, which directly interact with the user and coordinate the execution of user tasks with the help of the agents. 

**Abstract (ZH)**: 在人工智能（AI）逐渐融入我们生活各个方面的同时，代理程序正在经历复兴。这些代表人类行动的自主程序并非新鲜事物，也并非主流AI运动所独有。通过探索代理程序过去的各种形态，我们可以理解之前都做了什么、什么有效，更重要的是什么未能成功以及原因。这种理解使我们能够考察当前对代理程序的关注点有何不同。尽管生成式AI颇具吸引力，但仅依靠这项技术并不能使新一代代理程序取得成功。为了使当前代理程序浪潮既有效又可持续，我们设想了一个生态系统，该生态系统不仅包含代理程序，还包括反映用户偏好和行为的Sim（模拟）以及直接与用户交互并借助代理程序协调执行用户任务的助手。 

---
# Multi-Modal Grounded Planning and Efficient Replanning For Learning Embodied Agents with A Few Examples 

**Title (ZH)**: 基于多模态地面规划和高效重规划的少数示例学习体现代理的研究 

**Authors**: Taewoong Kim, Byeonghwi Kim, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2412.17288)  

**Abstract**: Learning a perception and reasoning module for robotic assistants to plan steps to perform complex tasks based on natural language instructions often requires large free-form language annotations, especially for short high-level instructions. To reduce the cost of annotation, large language models (LLMs) are used as a planner with few data. However, when elaborating the steps, even the state-of-the-art planner that uses LLMs mostly relies on linguistic common sense, often neglecting the status of the environment at command reception, resulting in inappropriate plans. To generate plans grounded in the environment, we propose FLARE (Few-shot Language with environmental Adaptive Replanning Embodied agent), which improves task planning using both language command and environmental perception. As language instructions often contain ambiguities or incorrect expressions, we additionally propose to correct the mistakes using visual cues from the agent. The proposed scheme allows us to use a few language pairs thanks to the visual cues and outperforms state-of-the-art approaches. Our code is available at this https URL. 

**Abstract (ZH)**: 基于自然语言指令规划机器人助手进行复杂任务的感知与推理模块的学习通常需要大量的自由格式语言注解，尤其是对于简短的高层指令而言。为了降低标注成本，采用大语言模型（LLMs）作为少量数据规划器。然而，在详细规划步骤时，即使是最先进的使用LLMs的规划器，大多数情况下依然依赖于语言常识，常常忽略了指令接收时的环境状态，导致计划不当。为了生成基于环境的计划，我们提出了一种新的方法，即FLARE（Few-shot Language with Environmental Adaptive Replanning Embodied Agent），该方法结合语言指令和环境感知来改进任务规划。由于语言指令中常包含歧义或错误表述，我们进一步提出利用代理器的视觉线索来纠正这些错误。所提出的方案利用视觉线索仅使用少量的语言指令对就可以实现优于现有最先进的方法的效果。我们的代码可以在以下网址获得：this https URL。 

---
# A Multi-AI Agent System for Autonomous Optimization of Agentic AI Solutions via Iterative Refinement and LLM-Driven Feedback Loops 

**Title (ZH)**: 基于迭代细化和LLM驱动的反馈循环的自主优化多AI代理系统：为代理AI解决方案赋能 

**Authors**: Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2412.17149)  

**Abstract**: Agentic AI systems use specialized agents to handle tasks within complex workflows, enabling automation and efficiency. However, optimizing these systems often requires labor-intensive, manual adjustments to refine roles, tasks, and interactions. This paper introduces a framework for autonomously optimizing Agentic AI solutions across industries, such as NLP-driven enterprise applications. The system employs agents for Refinement, Execution, Evaluation, Modification, and Documentation, leveraging iterative feedback loops powered by an LLM (Llama 3.2-3B). The framework achieves optimal performance without human input by autonomously generating and testing hypotheses to improve system configurations. This approach enhances scalability and adaptability, offering a robust solution for real-world applications in dynamic environments. Case studies across diverse domains illustrate the transformative impact of this framework, showcasing significant improvements in output quality, relevance, and actionability. All data for these case studies, including original and evolved agent codes, along with their outputs, are here: this https URL 

**Abstract (ZH)**: 以下是符合学术规范的翻译：

具有代理功能的AI系统使用专门的代理来处理复杂工作流程中的任务，从而实现自动化和提高效率。然而，优化这些系统通常需要耗时的手动调整来细化角色、任务和交互。本文提出了一种跨行业的自主优化具有代理功能的AI解决方案的框架，适用于如基于NLP的企业应用等场景。该系统采用代理进行细化（Refinement）、执行（Execution）、评估（Evaluation）、修改（Modification）和记录（Documentation），利用由LLM（Llama 3.2-3B）驱动的迭代反馈循环。该框架通过自主生成和测试假设来优化系统配置，实现了无需人工干预的最优性能。这种方法增强了系统的可扩展性和适应性，提供了在动态环境中实现实用解决方案的有效方案。来自不同领域的案例研究展示了该框架的转变性影响，显著提高了输出质量、相关性和可操作性。这些案例研究的所有数据，包括原始和演变的代理代码及其输出，均可在此获得：this https URL 

---
# Towards Selection and Transition Between Behavior-Based Neural Networks for Automated Driving 

**Title (ZH)**: 面向行为驱动神经网络的选择及其在自动驾驶中的过渡研究 

**Authors**: Iqra Aslam, Igor Anpilogov, Andreas Rausch  

**Link**: [PDF](https://arxiv.org/pdf/2412.16764)  

**Abstract**: Autonomous driving technology is progressing rapidly, largely due to complex End To End systems based on deep neural networks. While these systems are effective, their complexity can make it difficult to understand their behavior, raising safety concerns. This paper presents a new solution a Behavior Selector that uses multiple smaller artificial neural networks (ANNs) to manage different driving tasks, such as lane following and turning. Rather than relying on a single large network, which can be burdensome, require extensive training data, and is hard to understand, the developed approach allows the system to dynamically select the appropriate neural network for each specific behavior (e.g., turns) in real time. We focus on ensuring smooth transitions between behaviors while considering the vehicles current speed and orientation to improve stability and safety. The proposed system has been tested using the AirSim simulation environment, demonstrating its effectiveness. 

**Abstract (ZH)**: 自动驾驶技术正快速发展，主要得益于基于深度神经网络的端到端系统。虽然这些系统非常有效，但其复杂性可能会使其行为难以理解，从而引发安全方面的担忧。本文提出了一种新的解决方案——行为选择器，该解决方案使用多个较小的人工神经网络（ANNs）来管理不同的驾驶任务（如车道保持和转向）。与依赖单一庞大的网络相比，这种方法不需要大量训练数据，且易于理解，系统可以在实时动态选择适用于每个特定行为（例如转向）的神经网络。我们着重于确保行为之间的平滑过渡，同时考虑车辆当前的速度和方向，以提高稳定性和安全性。所提出系统已经在AirSim仿真环境中进行了测试，并展示了其有效性。 

---
# The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents 

**Title (ZH)**: 任务保护罩：强制任务对齐以防御间接提示注入攻击在LLM代理中的应用 

**Authors**: Feiran Jia, Tong Wu, Xin Qin, Anna Squicciarini  

**Link**: [PDF](https://arxiv.org/pdf/2412.16682)  

**Abstract**: Large Language Model (LLM) agents are increasingly being deployed as conversational assistants capable of performing complex real-world tasks through tool integration. This enhanced ability to interact with external systems and process various data sources, while powerful, introduces significant security vulnerabilities. In particular, indirect prompt injection attacks pose a critical threat, where malicious instructions embedded within external data sources can manipulate agents to deviate from user intentions. While existing defenses based on rule constraints, source spotlighting, and authentication protocols show promise, they struggle to maintain robust security while preserving task functionality. We propose a novel and orthogonal perspective that reframes agent security from preventing harmful actions to ensuring task alignment, requiring every agent action to serve user objectives. Based on this insight, we develop Task Shield, a test-time defense mechanism that systematically verifies whether each instruction and tool call contributes to user-specified goals. Through experiments on the AgentDojo benchmark, we demonstrate that Task Shield reduces attack success rates (2.07\%) while maintaining high task utility (69.79\%) on GPT-4o. 

**Abstract (ZH)**: 大型语言模型（LLM）代理越来越多地被部署为能够通过工具集成执行复杂现实任务的对话助手。这种增强的与外部系统交互和处理多种数据源的能力虽然强大，但也引入了显著的安全漏洞。特别是，间接提示注入攻击构成了一个关键威胁，其中嵌入在外部数据源中的恶意指令可以使代理偏离用户的意图。尽管基于规则约束、来源高亮和身份验证协议的现有防御措施显示出前景，但它们在保持安全性和保护任务功能方面存在困难。我们提出了一种新颖且独立的视角，重新定义了代理安全的焦点，从防止有害行为转移到确保任务对齐。根据这一认识，我们开发了Task Shield，这是一种运行时防御机制，系统地验证每个指令和工具调用是否有助于用户指定的目标。通过在AgentDojo基准测试上的实验，我们证明Task Shield在减少攻击成功率（2.07%）的同时，仍然保持了高任务实用性（69.79%）在GPT-4o上。 

---
# POEX: Policy Executable Embodied AI Jailbreak Attacks 

**Title (ZH)**: POEX：政策可执行的具身AI越界攻击

解释：
- **POEX** 是标题中的缩写或代码，直译为“POEX”。
- **Policy Executable** 翻译为“政策可执行的”，意指在执行过程中考虑了相关政策或规则。
- **Embodied AI** 翻译为“具身AI”，指的是嵌入到物理系统中的智能代理。
- **Jailbreak Attacks** 翻译为“越界攻击”，在学术语境中通常指违反系统安全策略的行为。 

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16633)  

**Abstract**: The integration of large language models (LLMs) into the planning module of Embodied Artificial Intelligence (Embodied AI) systems has greatly enhanced their ability to translate complex user instructions into executable policies. In this paper, we demystified how traditional LLM jailbreak attacks behave in the Embodied AI context. We conducted a comprehensive safety analysis of the LLM-based planning module of embodied AI systems against jailbreak attacks. Using the carefully crafted Harmful-RLbench, we accessed 20 open-source and proprietary LLMs under traditional jailbreak attacks, and highlighted two key challenges when adopting the prior jailbreak techniques to embodied AI contexts: (1) The harmful text output by LLMs does not necessarily induce harmful policies in Embodied AI context, and (2) even we can generate harmful policies, we have to guarantee they are executable in practice. To overcome those challenges, we propose Policy Executable (POEX) jailbreak attacks, where harmful instructions and optimized suffixes are injected into LLM-based planning modules, leading embodied AI to perform harmful actions in both simulated and physical environments. Our approach involves constraining adversarial suffixes to evade detection and fine-tuning a policy evaluater to improve the executability of harmful policies. We conducted extensive experiments on both a robotic arm embodied AI platform and simulators, to validate the attack and policy success rates on 136 harmful instructions from Harmful-RLbench. Our findings expose serious safety vulnerabilities in LLM-based planning modules, including the ability of POEX to be transferred across models. Finally, we propose mitigation strategies, such as safety-constrained prompts, pre- and post-planning checks, to address these vulnerabilities and ensure the safe deployment of embodied AI in real-world settings. 

**Abstract (ZH)**: 将大型语言模型（LLMs）集成到具身人工智能（Embodied AI）系统中的规划模块中，极大地增强了其将复杂的用户指令转化为可执行策略的能力。在本文中，我们探讨了传统的LLM劫持攻击在具身AI环境下的行为。我们对基于LLM的具身AI系统的规划模块进行了全面的安全分析，以抵御劫持攻击。我们使用精心设计的Harmful-RLbench，对20个开源和专有LLM进行了传统的劫持攻击测试，并突出了采用传统劫持技术在具身AI环境中的两个关键挑战：（1）LLM生成的危害性文本不一定会导致具身AI中的有害策略，（2）即使我们能够生成有害策略，也需要确保这些策略在实践中是可执行的。为克服这些挑战，我们提出了可执行策略（POEX）劫持攻击，其中将有害指令和优化的后缀注入基于LLM的规划模块，使具身AI在模拟和物理环境中执行有害行为。我们的方法包括限制敌对后缀以躲避检测，并fine-tuning策略评估器以提高有害策略的可执行性。我们在一个具身人工智能平台和模拟器上进行了广泛的实验，验证了136条来自Harmful-RLbench的有害指令的攻击和策略成功率。我们的研究揭示了基于LLM的规划模块中的严重安全漏洞，包括POEX能够跨模型转移的能力。最后，我们提出了缓解策略，如安全性约束提示、规划前后的检查，以解决这些漏洞并确保具身AI在真实环境中的安全部署。 

---
# Uncertainty Quantification in Continual Open-World Learning 

**Title (ZH)**: 持续开放世界学习中的不确定性量化 

**Authors**: Amanda S. Rios, Ibrahima J. Ndiour, Parual Datta, Jaroslaw Sydir, Omesh Tickoo, Nilesh Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2412.16409)  

**Abstract**: AI deployed in the real-world should be capable of autonomously adapting to novelties encountered after deployment. Yet, in the field of continual learning, the reliance on novelty and labeling oracles is commonplace albeit unrealistic. This paper addresses a challenging and under-explored problem: a deployed AI agent that continuously encounters unlabeled data - which may include both unseen samples of known classes and samples from novel (unknown) classes - and must adapt to it continuously. To tackle this challenge, we propose our method COUQ "Continual Open-world Uncertainty Quantification", an iterative uncertainty estimation algorithm tailored for learning in generalized continual open-world multi-class settings. We rigorously apply and evaluate COUQ on key sub-tasks in the Continual Open-World: continual novelty detection, uncertainty guided active learning, and uncertainty guided pseudo-labeling for semi-supervised CL. We demonstrate the effectiveness of our method across multiple datasets, ablations, backbones and performance superior to state-of-the-art. 

**Abstract (ZH)**: 在实际应用中部署的AI系统应当能够在部署后自主适应所遇到的新颖情况。然而，在持续学习领域，对新颖性及标签或acles的依赖虽然常见但却是不现实的。本文致力于解决一个具有挑战性且尚未充分探索的问题：一个部署后不断遇到无标签数据的AI代理，这些数据可能既包括未知类别的未见样本，也包括已知类别的未见样本，并且该代理必须不断适应这些数据。为应对这一挑战，我们提出了一种名为“COUQ：通用开放世界不确定性量化”的方法，这是一种针对泛化的持续开放世界多分类学习场景设计的迭代不确定性估计算法。我们严格地在持续开放世界任务的关键子任务上应用并评估了COUQ，包括持续新颖性检测、不确定性指导的主动学习以及不确定性指导的伪标签生成，以支持半监督持续学习。我们展示了该方法在多个数据集上的有效性，并在消融实验、底层架构和性能方面均优于现有最佳方法。 

---
# CLIP-RLDrive: Human-Aligned Autonomous Driving via CLIP-Based Reward Shaping in Reinforcement Learning 

**Title (ZH)**: CLIP-RLDrive：基于CLIP的奖励塑形在强化学习中实现的人类对齐自动驾驶 

**Authors**: Erfan Doroudian, Hamid Taghavifar  

**Link**: [PDF](https://arxiv.org/pdf/2412.16201)  

**Abstract**: This paper presents CLIP-RLDrive, a new reinforcement learning (RL)-based framework for improving the decision-making of autonomous vehicles (AVs) in complex urban driving scenarios, particularly in unsignalized intersections. To achieve this goal, the decisions for AVs are aligned with human-like preferences through Contrastive Language-Image Pretraining (CLIP)-based reward shaping. One of the primary difficulties in RL scheme is designing a suitable reward model, which can often be challenging to achieve manually due to the complexity of the interactions and the driving scenarios. To deal with this issue, this paper leverages Vision-Language Models (VLMs), particularly CLIP, to build an additional reward model based on visual and textual cues. 

**Abstract (ZH)**: 本文提出了一种新的基于强化学习（RL）的框架——CLIP-RLDrive，该框架旨在提高自动驾驶车辆（AVs）在复杂城市驾驶场景中的决策能力，尤其是在无信号交叉路口的情况。为了实现这一目标，通过基于对比语言-图像预训练（CLIP）的奖励塑造，将AVs的决策与人类偏好对齐。在RL方案中，设计一个合适的奖励模型是一个主要难点，由于交互和驾驶场景的复杂性，这种奖励模型常常难以手工构建。为解决这一问题，本文利用视觉-语言模型（VLMs），特别是CLIP，基于视觉和文本线索构建额外的奖励模型。 

---
# ResearchTown: Simulator of Human Research Community 

**Title (ZH)**: ResearchTown：人类研究社区模拟器 

**Authors**: Haofei Yu, Zhaochen Hong, Zirui Cheng, Kunlun Zhu, Keyang Xuan, Jinwei Yao, Tao Feng, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2412.17767)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in scientific domains, yet a fundamental question remains unanswered: Can we simulate human research communities with LLMs? Addressing this question can deepen our understanding of the processes behind idea brainstorming and inspire the automatic discovery of novel scientific insights. In this work, we propose ResearchTown, a multi-agent framework for research community simulation. Within this framework, the human research community is simplified and modeled as an agent-data graph, where researchers and papers are represented as agent-type and data-type nodes, respectively, and connected based on their collaboration relationships. We also introduce TextGNN, a text-based inference framework that models various research activities (e.g., paper reading, paper writing, and review writing) as special forms of a unified message-passing process on the agent-data graph. To evaluate the quality of the research simulation, we present ResearchBench, a benchmark that uses a node-masking prediction task for scalable and objective assessment based on similarity. Our experiments reveal three key findings: (1) ResearchTown can provide a realistic simulation of collaborative research activities, including paper writing and review writing; (2) ResearchTown can maintain robust simulation with multiple researchers and diverse papers; (3) ResearchTown can generate interdisciplinary research ideas that potentially inspire novel research directions. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科学领域展现出了巨大的潜力，但一个基本的问题仍未得到解答：我们能否使用LLMs模拟人类研究社区？回答这个问题有助于深化我们对创意生成过程的理解，并激发对新颖科学洞察的自动发现。在此项工作中，我们提出了ResearchTown，这是一种用于研究社区模拟的多智能体框架。在此框架中，人类研究社区被简化并建模为智能体-数据图，其中研究人员和论文分别用智能体类型节点和数据类型节点表示，并基于他们的合作关系进行连接。我们还引入了TextGNN，这是一种基于文本的推理框架，将各种研究活动（例如论文阅读、论文写作和审稿写作）建模为智能体-数据图上的统一消息传递过程的特殊形式。为了评估研究模拟的质量，我们提出了ResearchBench，这是一个基准测试，利用节点掩码预测任务进行基于相似性的可扩展和客观评估。我们的实验揭示了三个关键发现：（1）ResearchTown能够提供合作研究活动的现实模拟，包括论文写作和审稿写作；（2）ResearchTown能够使用多个研究人员和多样化论文保持稳健的模拟；（3）ResearchTown能够生成跨学科的研究理念，可能激发新的研究方向。 

---
# A Survey on Multi-Generative Agent System: Recent Advances and New Frontiers 

**Title (ZH)**: 多生成性智能体系统综述：近期进展与新前沿 

**Authors**: Shuaihang Chen, Yuanxing Liu, Wei Han, Weinan Zhang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17481)  

**Abstract**: Multi-generative agent systems (MGASs) have become a research hotspot since the rise of large language models (LLMs). However, with the continuous influx of new related works, the existing reviews struggle to capture them comprehensively. This paper presents a comprehensive survey of these studies. We first discuss the definition of MGAS, a framework encompassing much of previous work. We provide an overview of the various applications of MGAS in (i) solving complex tasks, (ii) simulating specific scenarios, and (iii) evaluating generative agents. Building on previous studies, we also highlight several challenges and propose future directions for research in this field. 

**Abstract (ZH)**: 多生成代理系统（MGAS）自大型语言模型（LLMs）的兴起以来已成为研究热点。然而，随着新相关工作的不断涌现，现有综述难以全面涵盖它们。本文对这些研究进行了全面综述。我们首先讨论了MGAS的定义，并提出一个框架，涵盖了许多之前的工作。我们概述了MGAS在以下几方面的各种应用：（i）解决复杂任务，（ii）模拟特定场景，（iii）评估生成代理。基于先前的研究，我们还指出了几个挑战，并提出了未来研究方向的建议。 

---
# Multi-Agent Sampling: Scaling Inference Compute for Data Synthesis with Tree Search-Based Agentic Collaboration 

**Title (ZH)**: 多代理采样：基于树搜索的代理协作扩展数据合成的推理计算能力 

**Authors**: Hai Ye, Mingbao Lin, Hwee Tou Ng, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2412.17061)  

**Abstract**: Scaling laws for inference compute in multi-agent systems remain under-explored compared to single-agent scenarios. This work aims to bridge this gap by investigating the problem of data synthesis through multi-agent sampling, where synthetic responses are generated by sampling from multiple distinct language models. Effective model coordination is crucial for successful multi-agent collaboration. Unlike previous approaches that rely on fixed workflows, we treat model coordination as a multi-step decision-making process, optimizing generation structures dynamically for each input question. We introduce Tree Search-based Orchestrated Agents~(TOA), where the workflow evolves iteratively during the sequential sampling process. To achieve this, we leverage Monte Carlo Tree Search (MCTS), integrating a reward model to provide real-time feedback and accelerate exploration. Our experiments on alignment, machine translation, and mathematical reasoning demonstrate that multi-agent sampling significantly outperforms single-agent sampling as inference compute scales. TOA is the most compute-efficient approach, achieving SOTA performance on WMT and a 71.8\% LC win rate on AlpacaEval. Moreover, fine-tuning with our synthesized alignment data surpasses strong preference learning methods on challenging benchmarks such as Arena-Hard and AlpacaEval. 

**Abstract (ZH)**: 与单智能体场景相比，多智能体系统中的推理计算规模律仍鲜有探索。本工作旨在通过多智能体采样问题的数据合成，填补这一空白。在多智能体采样中，合成响应通过从多个不同的语言模型中采样生成。有效的模型协调对于成功的多智能体协作至关重要。不同于之前依赖固定工作流程的方法，我们把模型协调视为一个多步骤的决策过程，动态优化每个输入问题的生成结构。我们引入了一种基于树搜索的协调智能体（Tree Search-based Orchestrated Agents, TOA），在顺序采样过程中工作流程迭代地演进。为此，我们利用蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS），结合奖励模型提供实时反馈，加速探索。我们在对齐、机器翻译和数学推理方面的实验表明，随着推理计算规模的扩大，多智能体采样显著优于单智能体采样。TOA是最高效的计算方法，在WMT上实现了SOTA性能，并在AlpacaEval上获得了71.8%的LC胜率。此外，使用我们合成的对齐数据进行微调，在难度较大的基准Arena-Hard和AlpacaEval上超过了强大的偏好学习方法。 

---
# Teaching LLMs to Refine with Tools 

**Title (ZH)**: 教学术大型语言模型使用工具进行优化与细化 

**Authors**: Dian Yu, Yuheng Zhang, Jiahao Xu, Tian Liang, Linfeng Song, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16871)  

**Abstract**: Large language models (LLMs) can refine their responses based on feedback, enabling self-improvement through iterative training or test-time refinement. However, existing methods predominantly focus on refinement within the same reasoning format, which may lead to non-correcting behaviors. We propose CaP, a novel approach that uses external tools to refine chain-of-thought (CoT) responses generated by the same or other LLMs. CaP employs a two-stage training process: supervised fine-tuning followed by preference optimization with DPO variants. Our observations highlight the critical role of preference optimization in enabling effective refinement. Additionally, we compare several sampling strategies to leverage CoT and tools at inference time. Experimental results demonstrate CaP's potential for effective cross-reasoning refinement and efficient inference. 

**Abstract (ZH)**: 大语言模型（LLMs）可以根据反馈改进其响应，从而通过迭代训练或测试时改进来实现自我提升。然而，现有方法主要集中在同一种推理格式内的改进上，这可能会导致非纠正性行为。我们提出了一种名为CaP的新方法，该方法使用外部工具来精炼由同一个或其它LLM生成的链式思维（CoT）响应。CaP采用两阶段训练过程：监督微调，随后是使用DPO变体进行偏好优化。我们的观察结果强调了偏好优化在实现有效改进中的关键作用。此外，我们还比较了几种采样策略，以便在推理时利用CoT和工具。实验结果表明，CaP在实现有效的跨推理改进和高效推理方面具有潜力。 

---
# InfoTech Assistant : A Multimodal Conversational Agent for InfoTechnology Web Portal Queries 

**Title (ZH)**: InfoTech助手：一个针对信息技术网页门户查询的多模态对话代理 

**Authors**: Sai Surya Gadiraju, Duoduo Liao, Akhila Kudupudi, Santosh Kasula, Charitha Chalasani  

**Link**: [PDF](https://arxiv.org/pdf/2412.16412)  

**Abstract**: This pilot study presents the development of the InfoTech Assistant, a domain-specific, multimodal chatbot engineered to address queries in bridge evaluation and infrastructure technology. By integrating web data scraping, large language models (LLMs), and Retrieval-Augmented Generation (RAG), the InfoTech Assistant provides accurate and contextually relevant responses. Data, including textual descriptions and images, are sourced from publicly available documents on the InfoTechnology website and organized in JSON format to facilitate efficient querying. The architecture of the system includes an HTML-based interface and a Flask back end connected to the Llama 3.1 model via LLM Studio. Evaluation results show approximately 95 percent accuracy on domain-specific tasks, with high similarity scores confirming the quality of response matching. This RAG-enhanced setup enables the InfoTech Assistant to handle complex, multimodal queries, offering both textual and visual information in its responses. The InfoTech Assistant demonstrates strong potential as a dependable tool for infrastructure professionals, delivering high accuracy and relevance in its domain-specific outputs. 

**Abstract (ZH)**: 本试点研究介绍了一种针对桥粱评估和基础设施技术领域的、具有多模态功能的助手——InfoTech助理。通过集成网页数据爬取、大型语言模型（LLMs）和检索增强生成（RAG）技术，InfoTech助理能够提供准确并具有上下文相关性的回应。数据包括文本描述和图像，来源于InfoTechnology网站上的公开文档，并以JSON格式组织，以便高效查询。该系统的架构包含基于HTML的用户界面和通过LLM Studio连接到Llama 3.1模型的Flask后端。评估结果显示，在特定领域的任务中，其准确率达到约95%，高相似度得分证实了回应匹配的质量。这种RAG增强设置使InfoTech助理能够处理复杂的多模态查询，在回应中提供文本和视觉信息。InfoTech助理作为基础设施专业人士的可靠工具展示了强大的潜力，能够在特定领域的输出中提供高准确性和相关性。 

---
# Application of Multimodal Large Language Models in Autonomous Driving 

**Title (ZH)**: 多模态大规模语言模型在自动驾驶中的应用 

**Authors**: Md Robiul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2412.16410)  

**Abstract**: In this era of technological advancements, several cutting-edge techniques are being implemented to enhance Autonomous Driving (AD) systems, focusing on improving safety, efficiency, and adaptability in complex driving environments. However, AD still faces some problems including performance limitations. To address this problem, we conducted an in-depth study on implementing the Multi-modal Large Language Model. We constructed a Virtual Question Answering (VQA) dataset to fine-tune the model and address problems with the poor performance of MLLM on AD. We then break down the AD decision-making process by scene understanding, prediction, and decision-making. Chain of Thought has been used to make the decision more perfectly. Our experiments and detailed analysis of Autonomous Driving give an idea of how important MLLM is for AD. 

**Abstract (ZH)**: 在这个技术进步的时代，多种前沿技术正被应用到自动驾驶（AD）系统中，以提高其在复杂驾驶环境中的安全性、效率和适应性。然而，自动驾驶仍然面临一些问题，包括性能限制。为了解决这些问题，我们对多模态大语言模型（Multi-modal Large Language Model, MLLM）的实施进行了深入研究。我们构建了一个虚拟问答（Virtual Question Answering, VQA）数据集，以微调模型并解决MLLM在自动驾驶中表现不佳的问题。随后，我们将自动驾驶决策过程分解为场景理解、预测和决策三个阶段，并使用逆向思维（Chain of Thought）使决策更加完善。我们的实验及对自动驾驶的详细分析表明，MLLM对自动驾驶的重要性。 

---
# Modular Conversational Agents for Surveys and Interviews 

**Title (ZH)**: 模块化对话代理在调查和访谈中的应用 

**Authors**: Jiangbo Yu, Jinhua Zhao, Luis Miranda-Moreno, Matthew Korp  

**Link**: [PDF](https://arxiv.org/pdf/2412.17049)  

**Abstract**: Surveys and interviews (structured, semi-structured, or unstructured) are widely used for collecting insights on emerging or hypothetical scenarios. Traditional human-led methods often face challenges related to cost, scalability, and consistency. Recently, various domains have begun to explore the use of conversational agents (chatbots) powered by large language models (LLMs). However, as public investments and policies on infrastructure and services often involve substantial public stakes and environmental risks, there is a need for a rigorous, transparent, privacy-preserving, and cost-efficient development framework tailored for such major decision-making processes. This paper addresses this gap by introducing a modular approach and its resultant parameterized process for designing conversational agents. We detail the system architecture, integrating engineered prompts, specialized knowledge bases, and customizable, goal-oriented conversational logic in the proposed approach. We demonstrate the adaptability, generalizability, and efficacy of our modular approach through three empirical studies: (1) travel preference surveys, highlighting multimodal (voice, text, and image generation) capabilities; (2) public opinion elicitation on a newly constructed, novel infrastructure project, showcasing question customization and multilingual (English and French) capabilities; and (3) transportation expert consultation about future transportation systems, highlighting real-time, clarification request capabilities for open-ended questions, resilience in handling erratic inputs, and efficient transcript post-processing. The results show the effectiveness of this modular approach and how it addresses key ethical, privacy, security, and token consumption concerns, setting the stage for the next-generation surveys and interviews. 

**Abstract (ZH)**: 调查和访谈（结构化的、半结构化的或非结构化的）广泛用于收集关于新兴或假设情境的见解。传统的由人类主导的方法经常会面临成本、可扩展性和一致性方面的挑战。近年来，各个领域开始探索使用大型语言模型（LLMs）驱动的对话代理（聊天机器人）的方法。然而，由于公共投资和政策通常涉及重要公共利益和环境风险，因此需要一种严谨、透明、保护隐私且成本效益高的开发框架，以适应这些重大决策过程。本文通过引入模块化方法及其参数化流程来解决这一问题，详细介绍了该系统的架构，涵盖了精心设计的提示、专业化的知识库以及可定制的目标导向对话逻辑。通过三项实证研究，展示了模块化方法的适应性、可移植性和有效性：（1）旅行偏好的调查，突显了多模态（语音、文本和图像生成）能力；（2）对一个新建成的创新型基础设施项目的公众意见收集，展现了问题定制化和多种语言（英语和法语）能力；以及（3）交通运输专家对未来交通系统咨询，强调了对开放式问题的即时澄清请求能力、处理异常输入的韧性以及高效的对话记录后处理。研究结果表明了模块化方法的有效性及其如何解决关键的伦理、隐私、安全性和令牌消耗问题，为下一代调查和访谈奠定了基础。 

---
