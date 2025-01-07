# Thinking with Many Minds: Using Large Language Models for Multi-Perspective Problem-Solving 

**Title (ZH)**: 多思维运用：利用大规模语言模型进行多视角问题解决 

**Authors**: Sanghyun Park, Boris Maciejovsky, Phanish Puranam  

**Link**: [PDF](https://arxiv.org/pdf/2501.02348)  

**Abstract**: Complex problem-solving requires cognitive flexibility--the capacity to entertain multiple perspectives while preserving their distinctiveness. This flexibility replicates the "wisdom of crowds" within a single individual, allowing them to "think with many minds." While mental simulation enables imagined deliberation, cognitive constraints limit its effectiveness. We propose synthetic deliberation, a Large Language Model (LLM)-based method that simulates discourse between agents embodying diverse perspectives, as a solution. Using a custom GPT-based model, we showcase its benefits: concurrent processing of multiple viewpoints without cognitive degradation, parallel exploration of perspectives, and precise control over viewpoint synthesis. By externalizing the deliberative process and distributing cognitive labor between parallel search and integration, synthetic deliberation transcends mental simulation's limitations. This approach shows promise for strategic planning, policymaking, and conflict resolution. 

**Abstract (ZH)**: 复杂问题解决需要认知灵活性——即在保持各种视角独特性的同时，能够容纳多种视角的能力。这种灵活性可以在单一个体中复制“群体的智慧”，使他们能够“使用多种思维”。虽然心理模拟能够支持想象中的讨论，但认知限制却限制了其有效性。我们提出了一种基于大型语言模型（LLM）的方法——合成讨论，作为一种解决方案。通过使用自定义的基于GPT的模型，我们展示了其优势：同时处理多种视角而不损害认知能力，平行探索各种视角，并精确控制视角合成。通过将讨论过程外部化并分散认知劳动，使并行搜索和整合得以进行，合成讨论超越了心理模拟的局限性。这种方法在战略规划、政策制定和冲突解决方面展现出前景。 

---
# Automated Generation of Challenging Multiple-Choice Questions for Vision Language Model Evaluation 

**Title (ZH)**: 面向视觉语言模型评估的具有挑战性的多项选择题自动化生成 

**Authors**: Yuhui Zhang, Yuchang Su, Yiming Liu, Xiaohan Wang, James Burgess, Elaine Sui, Chenyu Wang, Josiah Aklilu, Alejandro Lozano, Anjiang Wei, Ludwig Schmidt, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2501.03225)  

**Abstract**: The rapid development of vision language models (VLMs) demands rigorous and reliable evaluation. However, current visual question answering (VQA) benchmarks often depend on open-ended questions, making accurate evaluation difficult due to the variability in natural language responses. To address this, we introduce AutoConverter, an agentic framework that automatically converts these open-ended questions into multiple-choice format, enabling objective evaluation while reducing the costly question creation process. Our experiments demonstrate that AutoConverter can generate correct and challenging multiple-choice questions, with VLMs demonstrating consistently similar or lower accuracy on these questions compared to human-created ones. Using AutoConverter, we construct VMCBench, a benchmark created by transforming 20 existing VQA datasets into a unified multiple-choice format, totaling 9,018 questions. We comprehensively evaluate 33 state-of-the-art VLMs on VMCBench, setting a new standard for scalable, consistent, and reproducible VLM evaluation. 

**Abstract (ZH)**: 视觉语言模型（VLMs）的快速发展需要严格的可靠评估。然而，当前的视觉问答（VQA）基准往往依赖于开放式问题，由于自然语言回答的多样性，准确评估变得困难。为解决这一问题，我们引入了AutoConverter，这是一种自主框架，能够自动将开放式问题转换为多项选择格式，从而实现客观评估，并减少成本高昂的问题创建过程。实验结果表明，AutoConverter生成的多项选择问题既正确又具有挑战性，视觉语言模型在这些问题上的准确率与人工创建的问题相当或较低。使用AutoConverter，我们构建了VMCBench，这是一个通过将20个现有的VQA数据集统一转换为多项选择格式而创建的基准，总共包含9,018个问题。我们在VMCBench上全面评估了33个最先进的视觉语言模型，为大规模、一致性和可重复性视觉语言模型评估设立了新的标准。 

---
# LLMPC: Large Language Model Predictive Control 

**Title (ZH)**: LLMPC: 大型语言模型预测控制 

**Authors**: Gabriel Maher  

**Link**: [PDF](https://arxiv.org/pdf/2501.02486)  

**Abstract**: Recent advancements in prompting techniques for Large Language Models (LLMs) have improved their reasoning, planning, and action abilities. This paper examines these prompting techniques through the lens of model predictive control (MPC). We show that LLMs act as implicit planning cost function minimizers when planning prompts are used. Under our framework we demonstrate that LLM planning performance can be improved further by incorporating real planning cost functions and evaluators. 

**Abstract (ZH)**: 近年来，针对大型语言模型（LLMs）的提示技术取得了进步，这些技术提高了它们的推理、规划和行动能力。本文通过模型预测控制（MPC）的视角考察了这些提示技术。我们表明，当使用规划提示时，LLMs实际上充当着隐含规划成本函数的最小化者。在我们提出的框架下，我们证明通过引入实际的规划成本函数和评估器，可以进一步提高LLMs的规划性能。 

---
# Tree-based RAG-Agent Recommendation System: A Case Study in Medical Test Data 

**Title (ZH)**: 基于树结构的RAG代理推荐系统：一项关于医学测试数据的研究案例 

**Authors**: Yahe Yang, Chengyue Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02727)  

**Abstract**: We present HiRMed (Hierarchical RAG-enhanced Medical Test Recommendation), a novel tree-structured recommendation system that leverages Retrieval-Augmented Generation (RAG) for intelligent medical test recommendations. Unlike traditional vector similarity-based approaches, our system performs medical reasoning at each tree node through a specialized RAG process. Starting from the root node with initial symptoms, the system conducts step-wise medical analysis to identify potential underlying conditions and their corresponding diagnostic requirements. At each level, instead of simple matching, our RAG-enhanced nodes analyze retrieved medical knowledge to understand symptom-disease relationships and determine the most appropriate diagnostic path. The system dynamically adjusts its recommendation strategy based on medical reasoning results, considering factors such as urgency levels and diagnostic uncertainty. Experimental results demonstrate that our approach achieves superior performance in terms of coverage rate, accuracy, and miss rate compared to conventional retrieval-based methods. This work represents a significant advance in medical test recommendation by introducing medical reasoning capabilities into the traditional tree-based retrieval structure. 

**Abstract (ZH)**: 我们提出了一种名为 HiRMed（层次增强检索生成医疗测试推荐）的创新树结构推荐系统，该系统利用检索增强生成（RAG）技术实现智能化的医疗测试推荐。与传统的基于向量相似性的方法不同，我们的系统在每一棵树节点处通过专门的 RAG 过程进行医疗推理。从根节点出发，初始症状开始，系统逐步进行医疗分析，以识别潜在的病因及其对应的诊断需求。在每一层中，相较于简单的匹配，带有 RAG 增强的节点分析检索到的医学知识，了解症状-疾病关系，并确定最合适的诊断路径。系统根据医疗推理的结果动态调整其推荐策略，综合考虑急迫程度和诊断不确定性等因素。实验结果表明，与传统的基于检索的方法相比，我们的方法在覆盖率、准确性和漏诊率等方面表现更优。这项工作代表了在传统的基于树的检索结构中引入医疗推理能力的重要进展，在医疗测试推荐方面取得了显著进步。 

---
# Turn-based Multi-Agent Reinforcement Learning Model Checking 

**Title (ZH)**: 基于轮次的多智能体强化学习模型检查 

**Authors**: Dennis Gross  

**Link**: [PDF](https://arxiv.org/pdf/2501.03187)  

**Abstract**: In this paper, we propose a novel approach for verifying the compliance of turn-based multi-agent reinforcement learning (TMARL) agents with complex requirements in stochastic multiplayer games. Our method overcomes the limitations of existing verification approaches, which are inadequate for dealing with TMARL agents and not scalable to large games with multiple agents. Our approach relies on tight integration of TMARL and a verification technique referred to as model checking. We demonstrate the effectiveness and scalability of our technique through experiments in different types of environments. Our experiments show that our method is suited to verify TMARL agents and scales better than naive monolithic model checking. 

**Abstract (ZH)**: 在本文中，我们提出了一种新的方法，用于验证轮次制多智能体强化学习（TMARL）代理在随机多人游戏中是否符合复杂的要求。我们的方法克服了现有验证方法的局限性，这些方法不适用于处理TMARL代理，并且无法对具有多个代理的大型游戏进行扩展。我们的方法依赖于TMARL与一种称为模型检验的验证技术的紧密集成。我们通过在不同类型环境中的实验展示了该技术的有效性和可扩展性。我们的实验表明，我们的方法适用于验证TMARL代理，并且比传统的单一模型检验方法具有更好的可扩展性。 

---
# Multi-Agent Path Finding under Limited Communication Range Constraint via Dynamic Leading 

**Title (ZH)**: 在有限通信范围约束下的多智能体路径规划方法：基于动态领航的解决方案 

**Authors**: Hoang-Dung Bui, Erion Plaku, Gregoy J. Stein  

**Link**: [PDF](https://arxiv.org/pdf/2501.02770)  

**Abstract**: This paper proposes a novel framework to handle a multi-agent path finding problem under a limited communication range constraint, where all agents must have a connected communication channel to the rest of the team. Many existing approaches to multi-agent path finding (e.g., leader-follower platooning) overcome computational challenges of planning in this domain by planning one agent at a time in a fixed order. However, fixed leader-follower approaches can become stuck during planning, limiting their practical utility in dense-clutter environments. To overcome this limitation, we develop dynamic leading multi-agent path finding, which allows for dynamic reselection of the leading agent during path planning whenever progress cannot be made. The experiments show the efficiency of our framework, which can handle up to 25 agents with more than 90% success-rate across five environment types where baselines routinely fail. 

**Abstract (ZH)**: 本文提出了一种新颖的框架，用于处理在通信范围受限条件下的多-agent路径规划问题，其中所有agent必须与团队的其他部分保持连通的通信通道。许多现有的多-agent路径规划方法（例如，领导者-跟随者编队）通过以固定顺序一次规划一个agent来克服在这一领域中的计算挑战。然而，固定领导者-跟随者方法在规划过程中可能会陷入困境，限制了它们在密集障碍环境中的实际应用价值。为克服这一局限，我们开发了一种动态领导的多-agent路径规划方法，该方法在路径规划过程中可以在无法取得进展时动态重新选择领导者。实验结果显示，该框架的有效性，能够在五种不同类型的环境中处理多达25个agent，并且成功率达到超过90%，而基线方法在这种环境中通常会失败。 

---
# Enhancing Workplace Productivity and Well-being Using AI Agent 

**Title (ZH)**: 使用AI代理提升工作场所生产力与福祉 

**Authors**: Ravirajan K, Arvind Sundarajan  

**Link**: [PDF](https://arxiv.org/pdf/2501.02368)  

**Abstract**: This paper discusses the use of Artificial Intelligence (AI) to enhance workplace productivity and employee well-being. By integrating machine learning (ML) techniques with neurobiological data, the proposed approaches ensure alignment with human ethical standards through value alignment models and Hierarchical Reinforcement Learning (HRL) for autonomous task management. The system utilizes biometric feedback from employees to generate personalized health prompts, fostering a supportive work environment that encourages physical activity. Additionally, we explore decentralized multi-agent systems for improved collaboration and decision-making frameworks that enhance transparency. Various approaches using ML techniques in conjunction with AI implementations are discussed. Together, these innovations aim to create a more productive and health-conscious workplace. These outcomes assist HR management and organizations in launching more rational career progression streams for employees and facilitating organizational transformation. 

**Abstract (ZH)**: 本文探讨了利用人工智能（AI）提高工作效率和员工福祉的方法。通过将机器学习（ML）技术与神经生物学数据相结合，提出的方案通过价值对齐模型和分层强化学习（HRL）确保与人类伦理标准的契合。该系统利用员工的生物识别反馈生成个性化健康提示，营造一个支持性的工作环境，鼓励员工进行体育锻炼。此外，本文还探讨了去中心化的多智能体系统，以提高协作效率，并构建更加透明的决策框架。利用ML技术结合AI实施的各种方法进行了讨论。这些创新旨在创建一个更加高效和注重健康的职场环境。这些成果有助于人力资源管理和组织推动员工更加理性的职业发展路径，并促进组织的转型。 

---
# CORD: Generalizable Cooperation via Role Diversity 

**Title (ZH)**: CORD：通过角色多样性实现的可泛化合作 

**Authors**: Kanefumi Matsuyama, Kefan Su, Jiangxing Wang, Deheng Ye, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.02221)  

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) aims to develop agents that can collaborate effectively. However, most cooperative MARL methods overfit training agents, making learned policies not generalize well to unseen collaborators, which is a critical issue for real-world deployment. Some methods attempt to address the generalization problem but require prior knowledge or predefined policies of new teammates, limiting real-world applications. To this end, we propose a hierarchical MARL approach to enable generalizable cooperation via role diversity, namely CORD. CORD's high-level controller assigns roles to low-level agents by maximizing the role entropy with constraints. We show this constrained objective can be decomposed into causal influence in role that enables reasonable role assignment, and role heterogeneity that yields coherent, non-redundant role clusters. Evaluated on a variety of cooperative multi-agent tasks, CORD achieves better performance than baselines, especially in generalization tests. Ablation studies further demonstrate the efficacy of the constrained objective in generalizable cooperation. 

**Abstract (ZH)**: 合作多智能体强化学习（MARL）旨在开发能够有效协作的智能体。然而，大多数合作MARL方法会导致训练智能体过拟合，使得学到的策略不能很好地泛化到未见过的合作者身上，这是实际部署中一个关键问题。一些方法试图解决泛化问题，但需要对新队友的先验知识或预定义策略，从而限制了实际应用。为了解决这一问题，我们提出了一种分层MARL方法，通过角色多样性来实现可泛化的合作，即CORD方法。CORD的高层控制器通过最大化角色熵（在某些约束条件下）为低层智能体分配角色。我们展示了这种带约束的目标可以分解为角色因果影响，这有助于合理地分配角色，并通过角色异质性生成一致且不冗余的角色簇。在多种合作多智能体任务上进行评估，CORD在性能上优于基准方法，特别是在泛化测试中表现更优。进一步的消融研究还证实了带约束目标在可泛化合作上的有效性。 

---
# CAMP: Collaborative Attention Model with Profiles for Vehicle Routing Problems 

**Title (ZH)**: CAMP：带有个人资料的协作注意模型在车辆路线问题中的应用 

**Authors**: Chuanbo Hua, Federico Berto, Jiwoo Son, Seunghyun Kang, Changhyun Kwon, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2501.02977)  

**Abstract**: The profiled vehicle routing problem (PVRP) is a generalization of the heterogeneous capacitated vehicle routing problem (HCVRP) in which the objective is to optimize the routes of vehicles to serve client demands subject to different vehicle profiles, with each having a preference or constraint on a per-client basis. While existing learning methods have shown promise for solving the HCVRP in real-time, no learning method exists to solve the more practical and challenging PVRP. In this paper, we propose a Collaborative Attention Model with Profiles (CAMP), a novel approach that learns efficient solvers for PVRP using multi-agent reinforcement learning. CAMP employs a specialized attention-based encoder architecture to embed profiled client embeddings in parallel for each vehicle profile. We design a communication layer between agents for collaborative decision-making across profiled embeddings at each decoding step and a batched pointer mechanism to attend to the profiled embeddings to evaluate the likelihood of the next actions. We evaluate CAMP on two variants of PVRPs: PVRP with preferences, which explicitly influence the reward function, and PVRP with zone constraints with different numbers of agents and clients, demonstrating that our learned solvers achieve competitive results compared to both classical state-of-the-art neural multi-agent models in terms of solution quality and computational efficiency. We make our code openly available at this https URL. 

**Abstract (ZH)**: 以下是翻译后的论文内容或标题，符合学术规范：

简介：有特征的车辆路由问题（PVRP）是对异构容量车辆路由问题（HCVRP）的扩展，其目标是在考虑不同类型车辆的特征的基础上，优化车辆路线以满足客户的特定需求。尽管现有的学习方法在实时解决HCVRP方面显示出潜力，但目前尚无有效的方法解决更具实践意义且更具挑战性的PVRP。本文提出了一种协作注意力模型与有特征的方法（CAMP），这是一种利用多智能体强化学习学习解决PVRP的有效方法。CAMP 使用一种特定的基于注意力的编码器架构并行嵌入具有不同特征的客户表示。我们在每个解码步骤中设计了一个通信层，以促进智能体在有特征表示间的协作决策，并引入了一个批次指针机制，以关注具有特征的表示来评估下一步动作的可能性。我们分别在两个PVRP变体：带有偏好的PVRP，其奖励函数受到显式影响，以及具有区域约束的PVRP，根据不同数量的智能体和客户进行评估，结果显示，我们学习到的求解器在解决方案质量和计算效率方面均与经典的最先进的神经多智能体模型具有竞争性。我们已在以下链接公开提供我们的代码：[链接]。 

---
# Enhancing Lifelong Multi-Agent Path Finding with Cache Mechanism 

**Title (ZH)**: 基于缓存机制增强 lifelong 多代理路径规划 

**Authors**: Yimin Tang, Zhenghong Yu, Yi Zheng, T. K. Satish Kumar, Jiaoyang Li, Sven Koenig  

**Link**: [PDF](https://arxiv.org/pdf/2501.02803)  

**Abstract**: Multi-Agent Path Finding (MAPF), which focuses on finding collision-free paths for multiple robots, is crucial in autonomous warehouse operations. Lifelong MAPF (L-MAPF), where agents are continuously reassigned new targets upon completing their current tasks, offers a more realistic approximation of real-world warehouse scenarios. While cache storage systems can enhance efficiency and reduce operational costs, existing approaches primarily rely on expectations and mathematical models, often without adequately addressing the challenges of multi-robot planning and execution. In this paper, we introduce a novel mechanism called Lifelong MAPF with Cache Mechanism (L-MAPF-CM), which integrates high-level cache storage with low-level path planning. We have involved a new type of map grid called cache for temporary item storage. Additionally, we involved a task assigner (TA) with a locking mechanism to bridge the gap between the new cache grid and L-MAPF algorithm. The TA dynamically allocates target locations to agents based on their status in various scenarios. We evaluated L-MAPF-CM using different cache replacement policies and task distributions. L-MAPF-CM has demonstrated performance improvements particularly with high cache hit rates and smooth traffic conditions. 

**Abstract (ZH)**: 多智能体路径规划（MAPF），专注于为多个智能体找到无碰撞路径，在自主仓库操作中至关重要。终身多智能体路径规划（L-MAPF），其中智能体在完成当前任务后不断重新分配新的目标，能够更好地模拟现实世界仓库中的情况。虽然缓存存储系统可以提高效率并降低运营成本，但现有方法主要依赖于预期和数学模型，往往没有充分解决多智能体规划和执行中的挑战。在本文中，我们提出了一种名为缓存机制结合终身多智能体路径规划（L-MAPF-CM）的新机制，该机制将高层缓存存储与低层路径规划相结合。我们引入了一种新的图层结构，称为缓存图层，用于临时存储物品。此外，我们引入了一个具有锁机制的任务分配器（TA），以弥合新缓存图层与L-MAPF算法之间的差距。TA能够根据智能体的不同状态，动态分配目标位置。我们使用不同的缓存替换策略和任务分布对L-MAPF-CM进行了评估。实验结果表明，L-MAPF-CM在高缓存命中率和顺畅交通条件下，表现出性能改进。 

---
# Enhancing Robot Route Optimization in Smart Logistics with Transformer and GNN Integration 

**Title (ZH)**: 智能物流中基于变压器和GNN集成的机器人路线优化增强方法 

**Authors**: Hao Luo, Jianjun Wei, Shuchen Zhao, Ankai Liang, Zhongjin Xu, Ruxue Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02749)  

**Abstract**: This research delves into advanced route optimization for robots in smart logistics, leveraging a fusion of Transformer architectures, Graph Neural Networks (GNNs), and Generative Adversarial Networks (GANs). The approach utilizes a graph-based representation encompassing geographical data, cargo allocation, and robot dynamics, addressing both spatial and resource limitations to refine route efficiency. Through extensive testing with authentic logistics datasets, the proposed method achieves notable improvements, including a 15% reduction in travel distance, a 20% boost in time efficiency, and a 10% decrease in energy consumption. These findings highlight the algorithm's effectiveness, promoting enhanced performance in intelligent logistics operations. 

**Abstract (ZH)**: 本研究深入探讨了智能物流中机器人的高级路线优化问题，综合利用了Transformer架构、图神经网络（GNN）和生成对抗网络（GAN）等方法。该方法采用基于图的表示形式，涵盖了地理数据、货物分配和机器人动力学，同时解决了空间和资源限制问题，以优化路线效率。通过使用真实的物流数据集进行广泛测试，所提出的方法取得了显著的改进，包括减少了15%的行驶距离、提高了20%的时间效率以及降低了10%的能耗。这些发现突显了该算法的有效性，促进了智能物流操作性能的提升。 

---
# UAVs Meet LLMs: Overviews and Perspectives Toward Agentic Low-Altitude Mobility 

**Title (ZH)**: 无人机遇见大语言模型：自主低空移动的概览与展望 

**Authors**: Yonglin Tian, Fei Lin, Yiduo Li, Tengchao Zhang, Qiyao Zhang, Xuan Fu, Jun Huang, Xingyuan Dai, Yutong Wang, Chunwei Tian, Bai Li, Yisheng Lv, Levente Kovács, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02341)  

**Abstract**: Low-altitude mobility, exemplified by unmanned aerial vehicles (UAVs), has introduced transformative advancements across various domains, like transportation, logistics, and agriculture. Leveraging flexible perspectives and rapid maneuverability, UAVs extend traditional systems' perception and action capabilities, garnering widespread attention from academia and industry. However, current UAV operations primarily depend on human control, with only limited autonomy in simple scenarios, and lack the intelligence and adaptability needed for more complex environments and tasks. The emergence of large language models (LLMs) demonstrates remarkable problem-solving and generalization capabilities, offering a promising pathway for advancing UAV intelligence. This paper explores the integration of LLMs and UAVs, beginning with an overview of UAV systems' fundamental components and functionalities, followed by an overview of the state-of-the-art in LLM technology. Subsequently, it systematically highlights the multimodal data resources available for UAVs, which provide critical support for training and evaluation. Furthermore, it categorizes and analyzes key tasks and application scenarios where UAVs and LLMs converge. Finally, a reference roadmap towards agentic UAVs is proposed, aiming to enable UAVs to achieve agentic intelligence through autonomous perception, memory, reasoning, and tool utilization. Related resources are available at this https URL. 

**Abstract (ZH)**: 低空移动性，以无人机（UAVs）为例，在交通、物流和农业等多个领域引入了变革性的进步。利用灵活的视角和快速的操作能力，无人机扩展了传统系统感知和行动的能力，吸引了学术界和工业界的广泛关注。然而，当前的无人机操作主要依赖于人工控制，在简单场景中仅限于有限的自主能力，缺乏在更复杂环境和任务中所需的智能和适应能力。大型语言模型（LLMs）的出现展示了其出色的解决问题和泛化能力，为推进无人机智能提供了有前景的途径。本文探讨了将LLMs与UAVs集成的方法，首先概述了无人机系统的基本组件和功能，随后概述了当前大型语言模型技术的最新进展。接着，系统地介绍了可用于无人机的多模态数据资源，这些资源为训练和评估提供了关键支持。文中还对无人机和大型语言模型在关键任务和应用场景中的交汇点进行了分类和分析。最后，提出了一个通往自主无人机的参考路线图，旨在通过自主感知、记忆、推理和工具利用，使无人机实现自主智能。相关资源可以通过以下链接获取：https://your-resource-url-here.com 

---
# Attribute-Based Robotic Grasping with Data-Efficient Adaptation 

**Title (ZH)**: 基于属性的机器人抓取：数据高效适应 

**Authors**: Yang Yang, Houjian Yu, Xibai Lou, Yuanhao Liu, Changhyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2501.02149)  

**Abstract**: Robotic grasping is one of the most fundamental robotic manipulation tasks and has been the subject of extensive research. However, swiftly teaching a robot to grasp a novel target object in clutter remains challenging. This paper attempts to address the challenge by leveraging object attributes that facilitate recognition, grasping, and rapid adaptation to new domains. In this work, we present an end-to-end encoder-decoder network to learn attribute-based robotic grasping with data-efficient adaptation capability. We first pre-train the end-to-end model with a variety of basic objects to learn generic attribute representation for recognition and grasping. Our approach fuses the embeddings of a workspace image and a query text using a gated-attention mechanism and learns to predict instance grasping affordances. To train the joint embedding space of visual and textual attributes, the robot utilizes object persistence before and after grasping. Our model is self-supervised in a simulation that only uses basic objects of various colors and shapes but generalizes to novel objects in new environments. To further facilitate generalization, we propose two adaptation methods, adversarial adaption and one-grasp adaptation. Adversarial adaptation regulates the image encoder using augmented data of unlabeled images, whereas one-grasp adaptation updates the overall end-to-end model using augmented data from one grasp trial. Both adaptation methods are data-efficient and considerably improve instance grasping performance. Experimental results in both simulation and the real world demonstrate that our approach achieves over 81% instance grasping success rate on unknown objects, which outperforms several baselines by large margins. 

**Abstract (ZH)**: 机器人抓取是机器人操作中最基本的任务之一，并且一直是研究的热点。然而，迅速教会机器人在杂乱环境中抓取新的目标对象仍然具有挑战性。本文旨在通过利用有助于识别、抓取和快速适应新领域的事物属性来应对这一挑战。在本文中，我们提出了一种端到端的编码器-解码器网络，以学习基于属性的机器人抓取，并具备数据高效的适应能力。首先，我们使用各种基本对象对端到端模型进行预训练，以学习通用属性表示用于识别和抓取。我们的方法使用门控注意机制融合工作空间图像的嵌入和查询文本的嵌入，并学习预测实例抓取容征。为了联合学习视觉和文本属性的嵌入空间，机器人在抓取前后利用物体的持续性。我们的模型仅使用各种颜色和形状的基本对象在仿真环境中进行自我监督学习，但能够泛化到新的环境中新型对象的抓取。为了进一步促进泛化，我们提出了两种适应方法：对抗适应和单次抓取适应。对抗适应通过未标注图像的数据增强调节图像编码器，而单次抓取适应使用一次抓取试验的数据增强更新整个端到端模型。这两种适应方法都是数据高效的，并且显著提高了实例抓取性能。在仿真和现实环境中的实验结果表明，我们的方法在未知对象上的实例抓取成功率超过81%，远远优于多个基线方法。 

---
