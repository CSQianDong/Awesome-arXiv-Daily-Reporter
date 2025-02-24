# AutoToM: Automated Bayesian Inverse Planning and Model Discovery for Open-ended Theory of Mind 

**Title (ZH)**: AutoToM：自动贝叶斯反规划与模型发现算法在开放性心智理论中的应用 

**Authors**: Zhining Zhang, Chuanyang Jin, Mung Yao Jia, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15676)  

**Abstract**: Theory of Mind (ToM), the ability to understand people's mental variables based on their behavior, is key to developing socially intelligent agents. Current approaches to Theory of Mind reasoning either rely on prompting Large Language Models (LLMs), which are prone to systematic errors, or use rigid, handcrafted Bayesian Theory of Mind (BToM) models, which are more robust but cannot generalize across different domains. In this work, we introduce AutoToM, an automated Bayesian Theory of Mind method for achieving open-ended machine Theory of Mind. AutoToM can operate in any domain, infer any mental variable, and conduct robust Theory of Mind reasoning of any order. Given a Theory of Mind inference problem, AutoToM first proposes an initial BToM model. It then conducts automated Bayesian inverse planning based on the proposed model, leveraging an LLM as the backend. Based on the uncertainty of the inference, it iteratively refines the model, by introducing additional mental variables and/or incorporating more timesteps in the context. Empirical evaluations across multiple Theory of Mind benchmarks demonstrate that AutoToM consistently achieves state-of-the-art performance, offering a scalable, robust, and interpretable approach to machine Theory of Mind. 

**Abstract (ZH)**: 理论心理（ToM），即根据人们的行為理解其心理变量的能力，是开发社会智能代理的关键。当前的ToM推理方法要么依赖于大型语言模型（LLMs）提示，这容易产生系统性错误，要么使用结构僵硬的手动构建的贝叶斯理论心理（BToM）模型，这些模型更加稳健但不能在不同领域泛化。在此项工作中，我们介绍了一种自动化贝叶斯理论心理（AutoToM）方法，以实现开放式的机器理论心理。AutoToM可以在任何领域运行，能够推断任何心理变量，并执行任意级别的ToM推理。给定一个ToM推理问题，AutoToM首先提出一个初始的BToM模型。然后，基于提出的模型，它进行自动化贝叶斯逆规划，利用LLM作为后端。基于推理的不确定性，它通过引入额外的心理变量和/或增加上下文中的时间步数来迭代细化模型。在多个ToM基准测试中的实证评估表明，AutoToM始终能够达到最先进的性能，提供了一种可扩展、稳健且可解释的机器理论心理方法。 

---
# Automating Curriculum Learning for Reinforcement Learning using a Skill-Based Bayesian Network 

**Title (ZH)**: 使用基于技能的贝叶斯网络自动化强化学习中的课程学习 

**Authors**: Vincent Hsiao, Mark Roberts, Laura M. Hiatt, George Konidaris, Dana Nau  

**Link**: [PDF](https://arxiv.org/pdf/2502.15662)  

**Abstract**: A major challenge for reinforcement learning is automatically generating curricula to reduce training time or improve performance in some target task. We introduce SEBNs (Skill-Environment Bayesian Networks) which model a probabilistic relationship between a set of skills, a set of goals that relate to the reward structure, and a set of environment features to predict policy performance on (possibly unseen) tasks. We develop an algorithm that uses the inferred estimates of agent success from SEBN to weigh the possible next tasks by expected improvement. We evaluate the benefit of the resulting curriculum on three environments: a discrete gridworld, continuous control, and simulated robotics. The results show that curricula constructed using SEBN frequently outperform other baselines. 

**Abstract (ZH)**: 强化学习的一个主要挑战是自动生成课程，以减少训练时间或在某些目标任务中提高性能。我们引入了SEBNs（技能-环境贝叶斯网络），用于建模技能集、与奖励结构相关的目标集以及环境特征之间的概率关系，以预测在（可能未见过的任务）上的策略性能。我们开发了一种算法，该算法使用从SEBN中推断出的代理成功估计值来按预期改进来权衡可能的下一个任务。我们在这三个环境中评估了所生成课程的好处：离散网格世界、连续控制和模拟机器人。结果表明，使用SEBN构建的课程通常比其他基线方法性能更优。 

---
# Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path? 

**Title (ZH)**: 超智能代理带来灾难性风险：科学家型AI能提供一條更安全的道路吗？ 

**Authors**: Yoshua Bengio, Michael Cohen, Damiano Fornasiere, Joumana Ghosn, Pietro Greiner, Matt MacDermott, Sören Mindermann, Adam Oberman, Jesse Richardson, Oliver Richardson, Marc-Antoine Rondeau, Pierre-Luc St-Charles, David Williams-King  

**Link**: [PDF](https://arxiv.org/pdf/2502.15657)  

**Abstract**: The leading AI companies are increasingly focused on building generalist AI agents -- systems that can autonomously plan, act, and pursue goals across almost all tasks that humans can perform. Despite how useful these systems might be, unchecked AI agency poses significant risks to public safety and security, ranging from misuse by malicious actors to a potentially irreversible loss of human control. We discuss how these risks arise from current AI training methods. Indeed, various scenarios and experiments have demonstrated the possibility of AI agents engaging in deception or pursuing goals that were not specified by human operators and that conflict with human interests, such as self-preservation. Following the precautionary principle, we see a strong need for safer, yet still useful, alternatives to the current agency-driven trajectory. Accordingly, we propose as a core building block for further advances the development of a non-agentic AI system that is trustworthy and safe by design, which we call Scientist AI. This system is designed to explain the world from observations, as opposed to taking actions in it to imitate or please humans. It comprises a world model that generates theories to explain data and a question-answering inference machine. Both components operate with an explicit notion of uncertainty to mitigate the risks of overconfident predictions. In light of these considerations, a Scientist AI could be used to assist human researchers in accelerating scientific progress, including in AI safety. In particular, our system can be employed as a guardrail against AI agents that might be created despite the risks involved. Ultimately, focusing on non-agentic AI may enable the benefits of AI innovation while avoiding the risks associated with the current trajectory. We hope these arguments will motivate researchers, developers, and policymakers to favor this safer path. 

**Abstract (ZH)**: 领先的人工智能公司越来越关注于构建通用人工智能代理系统——这些系统能够自主规划、行动，并追求几乎涵盖人类所有任务的目标。尽管这些系统可能非常有用，但未受约束的人工智能代理可能会对公共安全和安全构成重大风险，从恶意行为者的误用到人类控制不可逆转的丧失。我们讨论了当前人工智能训练方法如何导致这些风险。事实上，各种情境和实验已经表明，人工智能代理有可能采取欺骗行为或追求由人类操作员未指定并可能与人类利益相冲突的目标，例如自我保护。根据预防原则，我们需要寻求更为安全但仍实用的替代方案。因此，我们建议作为进一步发展核心构建模块的是一种信任和安全设计的非代理型人工智能系统，我们称之为科学家型AI。该系统旨在通过观察解释世界，而不是采取行动模仿或取悦人类。该系统包括一个世界模型，用于生成解释数据的理论，以及一个问答推理机器。这两部分都具有明确的不确定性概念，以减轻过度自信预测的风险。考虑到这些考量，科学家型AI可以用于帮助人类研究人员加速科学发展，包括人工智能安全性。特别地，该系统可以作为预防风险的措施，防止尽管存在潜在风险仍可能出现的危险人工智能代理的出现。最终，专注于非代理型人工智能可能会使人工智能创新的好处最大化，同时避免当前轨迹相关风险。我们希望这些论点能够激励研究人员、开发者和政策制定者倾向于这一更为安全的路径。 

---
# Empowering LLMs with Logical Reasoning: A Comprehensive Survey 

**Title (ZH)**: 增强大型语言模型的逻辑推理能力：一项全面的综述 

**Authors**: Fengxiang Cheng, Haoxuan Li, Fenrong Liu, Robert van Rooij, Kun Zhang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.15652)  

**Abstract**: Large language models (LLMs) have achieved remarkable successes on various natural language tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs. This paper summarizes and categorizes the main challenges into two aspects: (1) Logical question answering, LLMs often fail to generate the correct answer within complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises and constrains. (2) Logical consistency, LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art Macaw question-answering LLM answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose detailed taxonomies of these methods. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, pretraining, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistency, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extensions to modal logic to account for uncertainty, and efficient algorithms satisfying multiple logical consistencies simultaneously. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言任务中取得了显著成效。然而，近期的研究发现LLMs在逻辑推理能力方面仍然存在重大挑战。本文总结并分类了主要挑战为两个方面：（1）逻辑问答，LLMs在解决复杂逻辑问题时，往往无法生成正确的答案，这些问题需要通过给定的一系列前提和约束进行复杂的演绎、归纳或反演绎推理。（2）逻辑一致性，LLMs容易在不同问题的回答中产生自相矛盾的响应。例如，最先进的Macaw问答LLM对“喜鹊是鸟吗？”和“鸟有翅膀吗？”两个问题都回答“是”，但在回答“喜鹊有翅膀吗？”时却回答“否”。为了促进这一研究方向，我们全面调查了最新的方法，并提出了这些方法的详细分类。具体来说，为了准确回答复杂的逻辑问题，先前的方法可以根据其对外部求解器、提示、预训练和微调的依赖程度进行分类。为了避免逻辑矛盾，我们讨论了各种逻辑一致性的概念与解决方案，包括推导、否定、传递性、事实一致性及其复合形式。此外，我们回顾了常用的基准数据集和评估指标，并讨论了一些有前景的研究方向，例如扩展到模态逻辑以考虑不确定性，以及同时满足多种逻辑一致性的高效算法。 

---
# Paradigms of AI Evaluation: Mapping Goals, Methodologies and Culture 

**Title (ZH)**: AI评估范式：目标、方法论与文化映射 

**Authors**: John Burden, Marko Tešić, Lorenzo Pacchiardi, José Hernández-Orallo  

**Link**: [PDF](https://arxiv.org/pdf/2502.15620)  

**Abstract**: Research in AI evaluation has grown increasingly complex and multidisciplinary, attracting researchers with diverse backgrounds and objectives. As a result, divergent evaluation paradigms have emerged, often developing in isolation, adopting conflicting terminologies, and overlooking each other's contributions. This fragmentation has led to insular research trajectories and communication barriers both among different paradigms and with the general public, contributing to unmet expectations for deployed AI systems. To help bridge this insularity, in this paper we survey recent work in the AI evaluation landscape and identify six main paradigms. We characterise major recent contributions within each paradigm across key dimensions related to their goals, methodologies and research cultures. By clarifying the unique combination of questions and approaches associated with each paradigm, we aim to increase awareness of the breadth of current evaluation approaches and foster cross-pollination between different paradigms. We also identify potential gaps in the field to inspire future research directions. 

**Abstract (ZH)**: 人工智能评估领域的研究日益复杂且跨学科性增强，吸引了具有多样化背景和目标的研究人员。因此，不同的评估范式相继出现，这些范式往往在孤立的情况下发展，采用相互冲突的术语，并忽视彼此的贡献。这种碎片化导致了范式间的孤立研究轨迹和沟通障碍，同时也影响了公众理解，导致部署的人工智能系统预期未实现。为了缓解这种孤立性，本文对当前的人工智能评估领域进行了综述，并识别出六个主要的评估范式。我们从目标、方法论和研究文化等关键维度对每个范式的主要最新贡献进行了特征描述。通过阐明与每个范式相关联的独特问题和方法组合，旨在增加对现有评估方法多样性的认识，并促进不同范式之间的交叉交流。此外，我们还指出了领域的潜在空白，以激发未来的研究方向。 

---
# Zweistein: A Dynamic Programming Evaluation Function for Einstein Würfelt Nicht! 

**Title (ZH)**: Zweistein：一种动态规划评估函数用于“爱因斯坦掷骰子也不输” 

**Authors**: Wei Lin. Hsueh, Tsan Sheng. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15547)  

**Abstract**: This paper introduces Zweistein, a dynamic programming evaluation function for Einstein Würfelt Nicht! (EWN). Instead of relying on human knowledge to craft an evaluation function, Zweistein uses a data-centric approach that eliminates the need for parameter tuning. The idea is to use a vector recording the distance to the corner of all pieces. This distance vector captures the essence of EWN. It not only outperforms many traditional EWN evaluation functions but also won first place in the TCGA 2023 competition. 

**Abstract (ZH)**: 本文介绍了Zweistein，这是一种用于爱因斯坦掷骰子不作弊（EWN）的动态规划评估函数。与依靠人工知识来构建评估函数不同，Zweistein采用以数据为中心的方法，消除了参数调整的需要。其核心思想是使用一个向量记录所有棋子到角落的最短距离。该距离向量捕捉了EWN的本质。Zweistein不仅在多个传统EWN评估函数中表现更优，还在2023年TCGA竞赛中赢得了第一名。 

---
# TAG: A Decentralized Framework for Multi-Agent Hierarchical Reinforcement Learning 

**Title (ZH)**: TAG：多代理层次强化学习的去中心化框架 

**Authors**: Giuseppe Paolo, Abdelhakim Benechehab, Hamza Cherkaoui, Albert Thomas, Balázs Kégl  

**Link**: [PDF](https://arxiv.org/pdf/2502.15425)  

**Abstract**: Hierarchical organization is fundamental to biological systems and human societies, yet artificial intelligence systems often rely on monolithic architectures that limit adaptability and scalability. Current hierarchical reinforcement learning (HRL) approaches typically restrict hierarchies to two levels or require centralized training, which limits their practical applicability. We introduce TAME Agent Framework (TAG), a framework for constructing fully decentralized hierarchical multi-agent this http URL enables hierarchies of arbitrary depth through a novel LevelEnv concept, which abstracts each hierarchy level as the environment for the agents above it. This approach standardizes information flow between levels while preserving loose coupling, allowing for seamless integration of diverse agent types. We demonstrate the effectiveness of TAG by implementing hierarchical architectures that combine different RL agents across multiple levels, achieving improved performance over classical multi-agent RL baselines on standard benchmarks. Our results show that decentralized hierarchical organization enhances both learning speed and final performance, positioning TAG as a promising direction for scalable multi-agent systems. 

**Abstract (ZH)**: 层次结构是生物系统和人类社会的基本特征，而人工智能系统通常依赖于单一架构，限制了其适应性和扩展性。当前的层次化强化学习（HRL）方法通常限制层次结构为两级或者需要集中训练，这限制了它们的实际应用。我们引入了TAME智能体框架（TAG），该框架用于构建完全去中心化的层次化多智能体系统。通过引入新的LevelEnv概念，TAG能够通过抽象每个层次作为智能体之上层次的环境来实现任意深度的层次结构。这种方法标准化了不同层次之间的信息流动，同时保持了松耦合，从而实现了不同类型的智能体无缝集成。我们通过在多个层次上结合不同的RL智能体来实现层次化架构，并在标准基准上展示了TAG的有效性，它实现了优于经典多智能体RL基线的性能。结果表明，去中心化的层次化组织能够提高学习速度和最终性能，将TAG定位为可扩展多智能体系统的一个有前途的方向。 

---
# Chitrarth: Bridging Vision and Language for a Billion People 

**Title (ZH)**: Chitrarth：连接视觉与语言，服务于十亿人 

**Authors**: Shaharukh Khan, Ayush Tarun, Abhinav Ravi, Ali Faraz, Akshat Patidar, Praveen Kumar Pokala, Anagha Bhangare, Raja Kolla, Chandra Khatri, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.15392)  

**Abstract**: Recent multimodal foundation models are primarily trained on English or high resource European language data, which hinders their applicability to other medium and low-resource languages. To address this limitation, we introduce Chitrarth (Chitra: Image; Artha: Meaning), an inclusive Vision-Language Model (VLM), specifically targeting the rich linguistic diversity and visual reasoning across 10 prominent Indian languages. Our model effectively integrates a state-of-the-art (SOTA) multilingual Large Language Model (LLM) with a vision module, primarily trained on multilingual image-text data. Furthermore, we also introduce BharatBench, a comprehensive framework for evaluating VLMs across various Indian languages, ultimately contributing to more diverse and effective AI systems. Our model achieves SOTA results for benchmarks across low resource languages while retaining its efficiency in English. Through our research, we aim to set new benchmarks in multilingual-multimodal capabilities, offering substantial improvements over existing models and establishing a foundation to facilitate future advancements in this arena. 

**Abstract (ZH)**: 近年来，多模态基础模型主要在英语或高资源的欧洲语言数据上进行训练，这限制了它们在其他中低资源语言中的应用。为解决这一局限，我们引入了Chitrarth（Chitra：图像；Artha：意义）这一包容性的跨模态语言模型（VLM），特别针对印度10种主要语言丰富的语言多样性和视觉推理能力。我们的模型有效地整合了一种最先进的多语言大规模语言模型（LLM）和一个视觉模块，该视觉模块主要在多语言图像文本数据上进行训练。此外，我们还引入了BharatBench，这是一个全面的评估框架，用于评估跨多种印度语言的VLM，最终促进更多样化和有效的AI系统的构建。我们的模型在低资源语言基准测试中达到了最先进的性能，同时在英语中保持了高效性。通过我们的研究，我们旨在建立多语言多模态能力的新标准，为现有模型提供显著改进，并为进一步在这个领域的发展奠定基础。 

---
# ARS: Automatic Routing Solver with Large Language Models 

**Title (ZH)**: ARS：基于大型语言模型的自动路由求解器 

**Authors**: Kai Li, Fei Liu, Zhenkun Wang, Xialiang Tong, Xiongwei Han, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15359)  

**Abstract**: Real-world Vehicle Routing Problems (VRPs) are characterized by a variety of practical constraints, making manual solver design both knowledge-intensive and time-consuming. Although there is increasing interest in automating the design of routing algorithms, existing research has explored only a limited array of VRP variants and fails to adequately address the complex and prevalent constraints encountered in real-world situations. To fill this gap, this paper introduces RoutBench, a benchmark of 1,000 VRP variants derived from 24 attributes, for evaluating the effectiveness of automatic routing solvers in addressing complex constraints. Along with RoutBench, we present the Automatic Routing Solver (ARS), which employs Large Language Model (LLM) agents to enhance a backbone algorithm framework by automatically generating constraint-aware heuristic code, based on problem descriptions and several representative constraints selected from a database. Our experiments show that ARS outperforms state-of-the-art LLM-based methods and commonly used solvers, automatically solving 91.67% of common VRPs and achieving at least a 30% improvement across all benchmarks. 

**Abstract (ZH)**: 现实世界中的车辆路线问题（VRPs）受多种实用约束的影响，使得手动设计求解器既需要深厚的知识支持，也需要耗费大量时间。尽管有越来越多的研究兴趣集中在自动化设计路由算法上，但现有研究仅探索了有限的VRP变体，并未能充分应对现实世界中复杂且普遍存在的约束。为弥补这一不足，本文引入了一个基于24个属性生成的1000个VRP变体的基准——RoutBench，用于评估自动化路由求解器在应对复杂约束方面的有效性。与此同时，我们还提出了自动路由求解器（ARS），该求解器使用大型语言模型（LLM）代理来增强基础算法框架，通过根据问题描述和从数据库中选择的若干代表性约束，自动生成具有约束意识的启发式代码。实验结果表明，ARS 在有效解决常见VRPs 和所有基准中均表现出色，自动解决了91.67%的常见VRP问题，并实现了至少30%的整体性能提升。 

---
# Measuring AI agent autonomy: Towards a scalable approach with code inspection 

**Title (ZH)**: 评估AI代理自主性：面向可扩展方法的代码检查途径 

**Authors**: Peter Cihon, Merlin Stein, Gagan Bansal, Sam Manning, Kevin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15212)  

**Abstract**: AI agents are AI systems that can achieve complex goals autonomously. Assessing the level of agent autonomy is crucial for understanding both their potential benefits and risks. Current assessments of autonomy often focus on specific risks and rely on run-time evaluations -- observations of agent actions during operation. We introduce a code-based assessment of autonomy that eliminates the need to run an AI agent to perform specific tasks, thereby reducing the costs and risks associated with run-time evaluations. Using this code-based framework, the orchestration code used to run an AI agent can be scored according to a taxonomy that assesses attributes of autonomy: impact and oversight. We demonstrate this approach with the AutoGen framework and select applications. 

**Abstract (ZH)**: 人工智能代理是指能够自主实现复杂目标的AI系统。评估代理的自主程度对于理解其潜在优势和风险至关重要。目前对自主性的评估往往侧重于特定风险，并主要依赖于运行时评估——即在操作过程中观察代理的行为。我们提出了一种基于代码的自主性评估方法，这种方法无需运行AI代理执行特定任务即可进行评估，从而减少了运行时评估相关的成本和风险。通过这种方法，可以使用基于代码的框架对运行AI代理所使用的编排代码进行打分，根据评估自主性的分类体系考察其影响和监控属性。我们通过AutoGen框架和选定的应用程序展示了这一方法。 

---
# The Imitation Game for Educational AI 

**Title (ZH)**: 《模仿游戏：教育人工智能》 

**Authors**: Shashank Sonkar, Naiming Liu, Xinghe Chen, Richard G. Baraniuk  

**Link**: [PDF](https://arxiv.org/pdf/2502.15127)  

**Abstract**: As artificial intelligence systems become increasingly prevalent in education, a fundamental challenge emerges: how can we verify if an AI truly understands how students think and reason? Traditional evaluation methods like measuring learning gains require lengthy studies confounded by numerous variables. We present a novel evaluation framework based on a two-phase Turing-like test. In Phase 1, students provide open-ended responses to questions, revealing natural misconceptions. In Phase 2, both AI and human experts, conditioned on each student's specific mistakes, generate distractors for new related questions. By analyzing whether students select AI-generated distractors at rates similar to human expert-generated ones, we can validate if the AI models student cognition. We prove this evaluation must be conditioned on individual responses - unconditioned approaches merely target common misconceptions. Through rigorous statistical sampling theory, we establish precise requirements for high-confidence validation. Our research positions conditioned distractor generation as a probe into an AI system's fundamental ability to model student thinking - a capability that enables adapting tutoring, feedback, and assessments to each student's specific needs. 

**Abstract (ZH)**: 随着人工智能系统在教育中的广泛应用，一个基本挑战随之浮现：我们如何验证AI是否真正理解了学生的思想和推理方式？传统的评估方法，如测量学习成果，需要耗时较长的研究，且受到多种变量的干扰。我们提出了一种基于两阶段图灵式测试的新颖评估框架。在第一阶段，学生对问题提供开放式回答，揭示出自然误解。在第二阶段，AI和人类专家根据每个学生特有的错误，生成新的相关问题的干扰选项。通过分析学生选择AI生成的干扰选项的比例是否与人类专家生成的比例相似，可以验证AI是否能够模拟学生认知。我们证明这种评估必须基于个体的反应——非个体化的方法仅针对共通误解。通过严谨的统计抽样理论，我们建立了高置信度验证所需的精确要求。我们的研究将带有条件的干扰选项生成定位为探索AI系统基本能力的探针，即模拟学生思维的能力——这种能力使教学、反馈和评估能够针对每个学生的具体需求进行调整。 

---
# GenAI vs. Human Fact-Checkers: Accurate Ratings, Flawed Rationales 

**Title (ZH)**: GenAI与人类事实核查者：准确的评分，欠妥的推理 

**Authors**: Yuehong Cassandra Tai, Khushi Navin Patni, Nicholas Daniel Hemauer, Bruce Desmarais, Yu-Ru Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.14943)  

**Abstract**: Despite recent advances in understanding the capabilities and limits of generative artificial intelligence (GenAI) models, we are just beginning to understand their capacity to assess and reason about the veracity of content. We evaluate multiple GenAI models across tasks that involve the rating of, and perceived reasoning about, the credibility of information. The information in our experiments comes from content that subnational U.S. politicians post to Facebook. We find that GPT-4o, one of the most used AI models in consumer applications, outperforms other models, but all models exhibit only moderate agreement with human coders. Importantly, even when GenAI models accurately identify low-credibility content, their reasoning relies heavily on linguistic features and ``hard'' criteria, such as the level of detail, source reliability, and language formality, rather than an understanding of veracity. We also assess the effectiveness of summarized versus full content inputs, finding that summarized content holds promise for improving efficiency without sacrificing accuracy. While GenAI has the potential to support human fact-checkers in scaling misinformation detection, our results caution against relying solely on these models. 

**Abstract (ZH)**: 尽管近年来在理解生成型人工智能（GenAI）模型的能力和局限性方面取得了进展，我们刚刚开始了解它们评估和推理内容真实性的能力。我们评估了多个GenAI模型在涉及对信息真实性的评分以及对其推理的理解的任务中表现。我们实验中的信息来源于美国各州和地方政治家在Facebook上发布的内容。我们发现，GPT-4o，一个在消费者应用中广泛使用的AI模型，优于其他模型，但所有模型与人类编码者的一致性仅表现出中等程度的共识。重要的是，即使GenAI模型能够准确识别低可信度的内容，它们的推理主要依赖于语言特征和“硬性”标准，如细节程度、信息源可靠性及语言的正式性，而不仅仅是对真实性的理解。我们还评估了总结内容与完整内容输入的效果，发现总结内容有提升效率而不牺牲准确性的潜力。虽然GenAI有可能支持人类事实核查者扩大对虚假信息的检测，但我们的结果提示我们不应仅依赖这些模型。 

---
# One-step Diffusion Models with $f$-Divergence Distribution Matching 

**Title (ZH)**: 一步扩散模型与$f$散度分布匹配 

**Authors**: Yilun Xu, Weili Nie, Arash Vahdat  

**Link**: [PDF](https://arxiv.org/pdf/2502.15681)  

**Abstract**: Sampling from diffusion models involves a slow iterative process that hinders their practical deployment, especially for interactive applications. To accelerate generation speed, recent approaches distill a multi-step diffusion model into a single-step student generator via variational score distillation, which matches the distribution of samples generated by the student to the teacher's distribution. However, these approaches use the reverse Kullback-Leibler (KL) divergence for distribution matching which is known to be mode seeking. In this paper, we generalize the distribution matching approach using a novel $f$-divergence minimization framework, termed $f$-distill, that covers different divergences with different trade-offs in terms of mode coverage and training variance. We derive the gradient of the $f$-divergence between the teacher and student distributions and show that it is expressed as the product of their score differences and a weighting function determined by their density ratio. This weighting function naturally emphasizes samples with higher density in the teacher distribution, when using a less mode-seeking divergence. We observe that the popular variational score distillation approach using the reverse-KL divergence is a special case within our framework. Empirically, we demonstrate that alternative $f$-divergences, such as forward-KL and Jensen-Shannon divergences, outperform the current best variational score distillation methods across image generation tasks. In particular, when using Jensen-Shannon divergence, $f$-distill achieves current state-of-the-art one-step generation performance on ImageNet64 and zero-shot text-to-image generation on MS-COCO. Project page: this https URL 

**Abstract (ZH)**: 从扩散模型中采样涉及一个缓慢的迭代过程，这阻碍了它们的实际应用，尤其是对于交互式应用而言。为了加速生成速度，近期的方法通过变分评分蒸馏将多步扩散模型简化为单步的学生生成器，目标是使学生生成的样本分布与教师分布一致。然而，这些方法使用了逆Kullback-Leibler（KL）散度来进行分布匹配，这种散度已知具有局部性偏好。在本文中，我们通过一种新颖的$f$-散度最小化框架$f$-distill对分布匹配方法进行了扩展，该框架涵盖了不同散度在模式覆盖和训练方差方面的不同权衡。我们推导了教师分布和学生分布之间$f$-散度的梯度，并证明其可以表示为评分差异的乘积和由密度比确定的加权函数。当使用非局部性偏好较小的散度时，该加权函数自然强调教师分布中密度较高的样本。我们观察到，使用逆KL散度的流行变分评分蒸馏方法实际上是我们的框架中的一个特殊情况。实验证明，诸如前向KL散度和Jensen-Shannon散度等替代$f$-散度，在图像生成任务中优于当前最佳的变分评分蒸馏方法。特别是在使用Jensen-Shannon散度时，$f$-distill在ImageNet64的单步生成性能和MS-COCO的零样本文本到图像生成上达到了当前最先进的水平。项目页面：[此链接](这个链接需要具体提供URL) 

---
# BOSS: Benchmark for Observation Space Shift in Long-Horizon Task 

**Title (ZH)**: BOSS：长 horizon 任务中观测空间变化的基准Advisor: 该标题翻译为中文时，可以保持其缩写形式，同时确保翻译的准确性和学术规范性。因此，可以翻译为：

BOSS：长时 horizon 任务中观测空间变化的基准

注：在学术翻译中，我们尽量保持缩写形式不变，并根据中文的习惯进行适当调整，以确保术语的专业性和准确性。在这个翻译中，“BOSS” 保持不变，“horizon” 通常翻译为“时域”或“视野”，这里特指“horizon 任务”，保留了英文的原意，翻译为“长时 horizon 任务”。 

**Authors**: Yue Yang, Linfeng Zhao, Mingyu Ding, Gedas Bertasius, Daniel Szafir  

**Link**: [PDF](https://arxiv.org/pdf/2502.15679)  

**Abstract**: Robotics has long sought to develop visual-servoing robots capable of completing previously unseen long-horizon tasks. Hierarchical approaches offer a pathway for achieving this goal by executing skill combinations arranged by a task planner, with each visuomotor skill pre-trained using a specific imitation learning (IL) algorithm. However, even in simple long-horizon tasks like skill chaining, hierarchical approaches often struggle due to a problem we identify as Observation Space Shift (OSS), where the sequential execution of preceding skills causes shifts in the observation space, disrupting the performance of subsequent individually trained skill policies. To validate OSS and evaluate its impact on long-horizon tasks, we introduce BOSS (a Benchmark for Observation Space Shift). BOSS comprises three distinct challenges: "Single Predicate Shift", "Accumulated Predicate Shift", and "Skill Chaining", each designed to assess a different aspect of OSS's negative effect. We evaluated several recent popular IL algorithms on BOSS, including three Behavioral Cloning methods and the Visual Language Action model OpenVLA. Even on the simplest challenge, we observed average performance drops of 67%, 35%, 34%, and 54%, respectively, when comparing skill performance with and without OSS. Additionally, we investigate a potential solution to OSS that scales up the training data for each skill with a larger and more visually diverse set of demonstrations, with our results showing it is not sufficient to resolve OSS. The project page is: this https URL 

**Abstract (ZH)**: 机器人学长期以来一直致力于开发能够执行未见过的远期任务的视觉伺服机器人。层次化方法提供了一种实现这一目标的途径，通过任务规划器执行由技能组合构成的任务，每个视觉运动技能都使用特定的 imitation learning (IL) 算法进行预训练。然而，即使在简单的远期任务，如技能链中，层次化方法往往因我们识别的问题，即观察空间偏移（OSS）而受到影响，其中，前一技能的顺序执行会导致观察空间变化，从而破坏后续独立训练的技能策略的性能。为了验证 OSS 并评估其对远期任务的影响，我们引入了 BOSS（观察空间偏移基准）。BOSS 包含三个不同的挑战：“单谓词偏移”、“累积谓词偏移”和“技能链”，每个挑战都旨在评估 OSS 负面影响的不同方面。我们在 BOSS 上评估了几种最近流行的 IL 算法，包括三种行为模仿方法以及 OpenVLA 视觉语言动作模型。即使在最简单的挑战中，我们观察到当比较技能表现时，平均性能分别下降了 67%、35%、34% 和 54%。此外，我们还研究了解决 OSS 的一个潜在方案，即通过使用更大且更具视觉多样性的演示数据来扩展每个技能的训练数据，但我们的结果显示这种方法不足以解决 OSS。项目页面链接如下：this https URL 

---
# FLEKE: Federated Locate-then-Edit Knowledge Editing 

**Title (ZH)**: FLEKE：联邦定位-编辑知识编辑

在这个翻译中，“Federated Locate-then-Edit Knowledge Editing”被翻译为“FLEKE：联邦定位-编辑知识编辑”，以符合学术规范。其中，“Federated”翻译为“联邦”，“Locate-then-Edit”翻译为“定位-编辑”，“Knowledge Editing”翻译为“知识编辑”。这样的翻译既保留了原意，又符合中文的表达习惯。 

**Authors**: Zongkai Zhao, Guozeng Xu, Xiuhua Li, Kaiwen Wei, Jiang Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2502.15677)  

**Abstract**: Locate-then-Edit Knowledge Editing (LEKE) is a key technique for updating large language models (LLMs) without full retraining. However, existing methods assume a single-user setting and become inefficient in real-world multi-client scenarios, where decentralized organizations (e.g., hospitals, financial institutions) independently update overlapping knowledge, leading to redundant mediator knowledge vector (MKV) computations and privacy concerns. To address these challenges, we introduce Federated Locate-then-Edit Knowledge Editing (FLEKE), a novel task that enables multiple clients to collaboratively perform LEKE while preserving privacy and reducing computational overhead. To achieve this, we propose FedEdit, a two-stage framework that optimizes MKV selection and reuse. In the first stage, clients locally apply LEKE and upload the computed MKVs. In the second stage, rather than relying solely on server-based MKV sharing, FLEKE allows clients retrieve relevant MKVs based on cosine similarity, enabling knowledge re-edit and minimizing redundant computations. Experimental results on two benchmark datasets demonstrate that FedEdit retains over 96% of the performance of non-federated LEKE while significantly outperforming a FedAvg-based baseline by approximately twofold. Besides, we find that MEMIT performs more consistently than PMET in the FLEKE task with our FedEdit framework. Our code is available at this https URL. 

**Abstract (ZH)**: Locate-then-Edit Knowledge Editing (LEKE) 是一种在不进行全面重新训练的情况下更新大型语言模型（LLMs）的关键技术。然而，现有的方法假定单用户环境，而在现实世界多客户端场景中，分散化组织（例如，医院、金融机构）独立更新重叠的知识，这导致了中介知识向量（MKV）计算的冗余和隐私问题。为了解决这些挑战，我们引入了联合 Locate-then-Edit 知识编辑 (FLEKE)，这是一个新的任务，使多个客户端能够在保持隐私和减少计算开销的情况下协作执行 LEKE。为了实现这一点，我们提出了 FedEdit，这是一种两级框架，优化 MKV 的选择和重用。在第一阶段，客户端本地应用 LEKE，并上传计算出的 MKVs。在第二阶段，与单纯依赖服务器端的 MKV 共享不同，FLEKE 允许客户端根据余弦相似度检索相关 MKVs，这有助于知识重编辑并最大限度地减少冗余计算。在两个基准数据集上的实验结果表明，FedEdit 的性能保留超过了 96% 的非联合 LEKE 的性能，并且在基于 FedAvg 的基线方法上大约提高了两倍。此外，我们发现，在我们的 FedEdit 框架中，MEMIT 在 FLEKE 任务中的表现比 PMET 更为一致。我们的代码可以通过以下链接获取：[这里](this https URL)。 

---
# VaViM and VaVAM: Autonomous Driving through Video Generative Modeling 

**Title (ZH)**: VaViM 和 VaVAM：基于视频生成模型的自动驾驶 

**Authors**: Florent Bartoccioni, Elias Ramzi, Victor Besnier, Shashanka Venkataramanan, Tuan-Hung Vu, Yihong Xu, Loick Chambon, Spyros Gidaris, Serkan Odabas, David Hurych, Renaud Marlet, Alexandre Boulch, Mickael Chen, Éloi Zablocki, Andrei Bursuc, Eduardo Valle, Matthieu Cord  

**Link**: [PDF](https://arxiv.org/pdf/2502.15672)  

**Abstract**: We explore the potential of large-scale generative video models for autonomous driving, introducing an open-source auto-regressive video model (VaViM) and its companion video-action model (VaVAM) to investigate how video pre-training transfers to real-world driving. VaViM is a simple auto-regressive video model that predicts frames using spatio-temporal token sequences. We show that it captures the semantics and dynamics of driving scenes. VaVAM, the video-action model, leverages the learned representations of VaViM to generate driving trajectories through imitation learning. Together, the models form a complete perception-to-action pipeline. We evaluate our models in open- and closed-loop driving scenarios, revealing that video-based pre-training holds promise for autonomous driving. Key insights include the semantic richness of the learned representations, the benefits of scaling for video synthesis, and the complex relationship between model size, data, and safety metrics in closed-loop evaluations. We release code and model weights at this https URL 

**Abstract (ZH)**: 我们探讨了大规模生成视频模型在自主驾驶领域的潜在应用，提出了一个开源的自回归视频模型（VaViM）及其伴随的动作视频模型（VaVAM），以研究视频预训练如何转移到实际驾驶中。VaViM 是一个简单的自回归视频模型，通过时空令牌序列预测帧，并展示了它在捕捉驾驶场景的意义和动态方面的能力。VaVAM 是动作视频模型，利用 VaViM 学到的表示通过模仿学习生成驾驶轨迹。这两款模型共同构成了从感知到行动的完整流水线。我们在开放环和闭合环驾驶场景中评估了我们的模型，揭示了基于视频的预训练有望实现自主驾驶。主要见解包括学到的表示的语义丰富性、视频合成方面的扩展性优势，以及闭合环评估中模型规模、数据与安全性指标之间复杂的关系。我们已在以下网址发布了代码和模型权重：[此处提供网址] 

---
# Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing 

**Title (ZH)**: 几乎人工智能，几乎人类：检测人工智能润色的文本的挑战 

**Authors**: Shoumik Saha, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15666)  

**Abstract**: The growing use of large language models (LLMs) for text generation has led to widespread concerns about AI-generated content detection. However, an overlooked challenge is AI-polished text, where human-written content undergoes subtle refinements using AI tools. This raises a critical question: should minimally polished text be classified as AI-generated? Misclassification can lead to false plagiarism accusations and misleading claims about AI prevalence in online content. In this study, we systematically evaluate eleven state-of-the-art AI-text detectors using our AI-Polished-Text Evaluation (APT-Eval) dataset, which contains $11.7K$ samples refined at varying AI-involvement levels. Our findings reveal that detectors frequently misclassify even minimally polished text as AI-generated, struggle to differentiate between degrees of AI involvement, and exhibit biases against older and smaller models. These limitations highlight the urgent need for more nuanced detection methodologies. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在文本生成中的广泛应用引发了对AI生成内容检测的广泛关注。然而，一个被忽视的挑战是AI润色文本，即使用AI工具对人撰写的文本进行微妙的改进。这引发了一个关键问题：如何界定轻微润色的文本是否应被视为AI生成的内容？误分类可能导致误判的剽窃指控和对在线内容中AI普及程度的误导性声明。在本研究中，我们系统地评估了十一个最先进的AI文本检测器，使用了包含11,700个样本文本的AI润色文本评估（APT-Eval）数据集，这些文本在不同程度的AI参与下进行了润色。我们的研究结果表明，检测器频繁地将轻微润色的文本误分类为AI生成的内容，难以区分AI参与程度的不同，并显示出对较旧和较小模型的偏见。这些局限性突显了需要更加精细的检测方法的紧迫需求。 

---
# Multi-Agent Architecture in Distributed Environment Control Systems: vision, challenges, and opportunities 

**Title (ZH)**: 分布式环境控制系统中的多代理架构：愿景、挑战与机遇 

**Authors**: Natasha Astudillo, Fernando Koch  

**Link**: [PDF](https://arxiv.org/pdf/2502.15663)  

**Abstract**: The increasing demand for energy-efficient solutions in large-scale infrastructure, particularly data centers, requires advanced control strategies to optimize environmental management systems. We propose a multi-agent architecture for distributed control of air-cooled chiller systems in data centers. Our vision employs autonomous agents to monitor and regulate local operational parameters and optimize system-wide efficiency. We demonstrate how this approach improves the responsiveness, operational robustness, and energy efficiency of the system, contributing to the broader goal of sustainable infrastructure management. 

**Abstract (ZH)**: 随着对高效能源解决方案需求的不断增加，特别是在数据中心等大规模基础设施中，需要先进的控制策略来优化环境管理系统。我们提出了一种多代理架构，用于数据中心分布式控制空气冷却制冷系统的优化。我们的愿景是通过自主代理来监控和调节本地操作参数，并优化系统范围内的效率。我们展示了这种方法如何提高系统的响应性、运行稳定性和能源效率，从而有助于可持续基础设施管理的总体目标。 

---
# AutoTandemML: Active Learning Enhanced Tandem Neural Networks for Inverse Design Problems 

**Title (ZH)**: AutoTandemML：增强型迭代表面神经网络用于逆向设计问题 

**Authors**: Luka Grbcic, Juliane Müller, Wibe Albert de Jong  

**Link**: [PDF](https://arxiv.org/pdf/2502.15643)  

**Abstract**: Inverse design in science and engineering involves determining optimal design parameters that achieve desired performance outcomes, a process often hindered by the complexity and high dimensionality of design spaces, leading to significant computational costs. To tackle this challenge, we propose a novel hybrid approach that combines active learning with Tandem Neural Networks to enhance the efficiency and effectiveness of solving inverse design problems. Active learning allows to selectively sample the most informative data points, reducing the required dataset size without compromising accuracy. We investigate this approach using three benchmark problems: airfoil inverse design, photonic surface inverse design, and scalar boundary condition reconstruction in diffusion partial differential equations. We demonstrate that integrating active learning with Tandem Neural Networks outperforms standard approaches across the benchmark suite, achieving better accuracy with fewer training samples. 

**Abstract (ZH)**: 科学与工程中的逆向设计涉及确定能够实现预期性能结果的最佳设计参数，这一过程往往受到设计空间的复杂性和高维性的影响，导致大量的计算成本。为应对这一挑战，我们提出了一种新的混合方法，结合了主动学习与串联神经网络，以提高解决逆向设计问题的效率和有效性。主动学习允许选择性地采样最具信息量的数据点，从而减少所需的数据集大小而不牺牲准确性。我们利用三个基准问题对这种方法进行了研究：机翼逆向设计、光子表面逆向设计以及扩散偏微分方程中标量边界条件重构。研究结果表明，将主动学习与串联神经网络结合使用，在整个基准测试套件中均优于标准方法，并且能够使用更少的训练样本获得更高的准确度。 

---
# Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models 

**Title (ZH)**: 进入新的嵌入空间定向：分析多语言语言模型中模型干预引起的跨语言对齐 

**Authors**: Anirudh Sundar, Sinead Williamson, Katherine Metcalf, Barry-John Theobald, Skyler Seto, Masha Fedzechkina  

**Link**: [PDF](https://arxiv.org/pdf/2502.15639)  

**Abstract**: Aligned representations across languages is a desired property in multilingual large language models (mLLMs), as alignment can improve performance in cross-lingual tasks. Typically alignment requires fine-tuning a model, which is computationally expensive, and sizable language data, which often may not be available. A data-efficient alternative to fine-tuning is model interventions -- a method for manipulating model activations to steer generation into the desired direction. We analyze the effect of a popular intervention (finding experts) on the alignment of cross-lingual representations in mLLMs. We identify the neurons to manipulate for a given language and introspect the embedding space of mLLMs pre- and post-manipulation. We show that modifying the mLLM's activations changes its embedding space such that cross-lingual alignment is enhanced. Further, we show that the changes to the embedding space translate into improved downstream performance on retrieval tasks, with up to 2x improvements in top-1 accuracy on cross-lingual retrieval. 

**Abstract (ZH)**: 多语言大型语言模型（mLLMs）中的跨语言表示的一致性是一个期望的特性，因为一致性可以提高跨语言任务的表现。通常，实现一致性需要对模型进行微调，这是一个计算成本高昂的过程，并且需要大量语言数据，这些数据往往难以获得。一种数据高效的替代微调的方法是模型干预——一种通过操作模型激活以引导生成进入所需方向的方法。我们分析了流行的一种干预方法（寻找专家）对mLLMs中跨语言表示一致性的影响。我们确定了要操作的神经元，对mLLMs在干预前后嵌入空间进行了内部审视。我们证明，修改mLLM的激活会改变其嵌入空间，从而使跨语言一致性增强。进一步，我们证明，嵌入空间的变化转化为检索任务下游性能的提升，跨语言检索的最 ofrece准确性提高了高达2倍。 

---
# Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification 

**Title (ZH)**: mantis：轻量级校准基础模型，用于用户友好的时间序列分类 

**Authors**: Vasilii Feofanov, Songkang Wen, Marius Alonso, Romain Ilbert, Hongbo Guo, Malik Tiomoko, Lujia Pan, Jianfeng Zhang, Ievgen Redko  

**Link**: [PDF](https://arxiv.org/pdf/2502.15637)  

**Abstract**: In recent years, there has been increasing interest in developing foundation models for time series data that can generalize across diverse downstream tasks. While numerous forecasting-oriented foundation models have been introduced, there is a notable scarcity of models tailored for time series classification. To address this gap, we present Mantis, a new open-source foundation model for time series classification based on the Vision Transformer (ViT) architecture that has been pre-trained using a contrastive learning approach. Our experimental results show that Mantis outperforms existing foundation models both when the backbone is frozen and when fine-tuned, while achieving the lowest calibration error. In addition, we propose several adapters to handle the multivariate setting, reducing memory requirements and modeling channel interdependence. 

**Abstract (ZH)**: 近年来，人们越来越关注开发能够在多种下游任务中泛化的时序数据基础模型。尽管已经提出了众多面向预测的基础模型，但专门为时序分类任务设计的模型仍然相对稀缺。为填补这一空白，我们提出了Mantis，这是一种基于Vision Transformer (ViT) 架构的新开源时序分类基础模型，该模型通过对比学习方式进行预训练。实验结果显示，Mantis在冻结主干和微调两种情况下均优于现有的基础模型，并且具有最低的校准误差。此外，我们还提出了几种适配器以处理多变量情况，从而减少内存需求并建模通道间的依赖关系。 

---
# The Relationship Between Reasoning and Performance in Large Language Models -- o3 (mini) Thinks Harder, Not Longer 

**Title (ZH)**: 大型语言模型中推理与绩效之间的关系——o3（迷你）模型思考更深入，而非更久 

**Authors**: Marthe Ballon, Andres Algaba, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2502.15631)  

**Abstract**: Large language models have demonstrated remarkable progress in mathematical reasoning, leveraging chain-of-thought and test-time compute scaling. However, many open questions remain regarding the interplay between reasoning token usage and accuracy gains. In particular, when comparing models across generations, it is unclear whether improved performance results from longer reasoning chains or more efficient reasoning. We systematically analyze chain-of-thought length across o1-mini and o3-mini variants on the Omni-MATH benchmark, finding that o3-mini (m) achieves superior accuracy without requiring longer reasoning chains than o1-mini. Moreover, we show that accuracy generally declines as reasoning chains grow across all models and compute settings, even when controlling for difficulty of the questions. This accuracy drop is significantly smaller in more proficient models, suggesting that new generations of reasoning models use test-time compute more effectively. Finally, we highlight that while o3-mini (h) achieves a marginal accuracy gain over o3-mini (m), it does so by allocating substantially more reasoning tokens across all problems, even the ones that o3-mini (m) can already solve. These findings provide new insights into the relationship between model capability and reasoning length, with implications for efficiency, scaling, and evaluation methodologies. 

**Abstract (ZH)**: 大型语言模型在数学推理方面已经取得了显著进展，通过采用推理链和测试时计算量扩展等方法。然而，关于推理标记使用和准确度提升之间的相互作用，仍有许多开放问题。特别是，在跨代模型比较时，难以确定性能改进是源于更长的推理链还是更高效的推理。我们系统地分析了在Omni-MATH基准上，o1-mini和o3-mini变体中的推理链长度，发现o3-mini (m)在不需要更长的推理链的情况下实现了更高的准确度。此外，我们展示了随着推理链的增长，所有模型的准确度通常都会下降，即使在控制问题难度的情况下也是如此。准确度下降的幅度在更擅长的模型中较小，表明新一代推理模型更有效地利用了测试时的计算量。最后，我们指出，虽然o3-mini (h)相较于o3-mini (m)在准确度上有所提升，但这主要是通过在所有问题上分配远多于o3-mini (m)的推理标记实现的，即使o3-mini (m)已经可以解决这些问题。这些发现为模型能力与推理长度之间的关系提供了新的见解，具有对效率、扩展和评估方法的指导意义。 

---
# Dynamic Knowledge Selector and Evaluator for recommendation with Knowledge Graph 

**Title (ZH)**: 知识图谱驱动的动态知识选择与评估器在推荐系统中的应用 

**Authors**: Feng Xia, Zhifei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15623)  

**Abstract**: In recent years recommendation systems typically employ the edge information provided by knowledge graphs combined with the advantages of high-order connectivity of graph networks in the recommendation field. However, this method is limited by the sparsity of labels, cannot learn the graph structure well, and a large number of noisy entities in the knowledge graph will affect the accuracy of the recommendation results. In order to alleviate the above problems, we propose a dynamic knowledge-selecting and evaluating method guided by collaborative signals to distill information in the knowledge graph. Specifically, we use a Chain Route Evaluator to evaluate the contributions of different neighborhoods for the recommendation task and employ a Knowledge Selector strategy to filter the less informative knowledge before evaluating. We conduct baseline model comparison and experimental ablation evaluations on three public datasets. The experiments demonstrate that our proposed model outperforms current state-of-the-art baseline models, and each modules effectiveness in our model is demonstrated through ablation experiments. 

**Abstract (ZH)**: 近年来，推荐系统通常利用知识图谱提供的边缘信息，结合图网络在高阶连接方面的优势应用于推荐领域。然而，这种方法受到标签稀疏性的限制，难以很好地学习图结构，并且知识图谱中的大量噪声实体会降低推荐结果的准确性。为了缓解上述问题，我们提出了一种由协作信号指导的知识选择和评估动态方法，以提炼知识图谱中的信息。具体来说，我们使用链路路线评估器评估不同邻域对推荐任务的贡献，并采用知识选择策略来筛选评估前的低信息量知识。我们在三个公开数据集上进行了基准模型比较和实验消融评估。实验结果表明，我们的模型优于当前最先进的基准模型，消融实验也证明了我们模型中每个模块的有效性。 

---
# Extraction multi-étiquettes de relations en utilisant des couches de Transformer 

**Title (ZH)**: 使用Transformer层进行多标签关系提取 

**Authors**: Ngoc Luyen Le, Gildas Tagny Ngompé  

**Link**: [PDF](https://arxiv.org/pdf/2502.15619)  

**Abstract**: In this article, we present the BTransformer18 model, a deep learning architecture designed for multi-label relation extraction in French texts. Our approach combines the contextual representation capabilities of pre-trained language models from the BERT family - such as BERT, RoBERTa, and their French counterparts CamemBERT and FlauBERT - with the power of Transformer encoders to capture long-term dependencies between tokens. Experiments conducted on the dataset from the TextMine'25 challenge show that our model achieves superior performance, particularly when using CamemBERT-Large, with a macro F1 score of 0.654, surpassing the results obtained with FlauBERT-Large. These results demonstrate the effectiveness of our approach for the automatic extraction of complex relations in intelligence reports. 

**Abstract (ZH)**: 在本文中，我们提出了BTransformer18模型，这是一种专为法语文本中的多标签关系提取设计的深度学习架构。该方法结合了来自BERT家族的预训练语言模型（如BERT、RoBERTa及其法语文本版本CamemBERT和FlauBERT）的上下文表示能力，以及Transformer编码器的强大功能，以捕捉token之间的长距离依赖关系。实验结果表明，我们的模型在TextMine'25挑战数据集上的性能表现出色，尤其是在使用CamemBERT-Large时，宏F1分数达到0.654，超过了使用FlauBERT-Large获得的结果。这些结果证明了该方法在自动提取情报报告中复杂关系的有效性。 

---
# Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing 

**Title (ZH)**: 探针剪枝：通过基于模型的动态剪枝加速大型语言模型 

**Authors**: Qi Le, Enmao Diao, Ziyan Wang, Xinran Wang, Jie Ding, Li Yang, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15618)  

**Abstract**: We introduce Probe Pruning (PP), a novel framework for online, dynamic, structured pruning of Large Language Models (LLMs) applied in a batch-wise manner. PP leverages the insight that not all samples and tokens contribute equally to the model's output, and probing a small portion of each batch effectively identifies crucial weights, enabling tailored dynamic pruning for different batches. It comprises three main stages: probing, history-informed pruning, and full inference. In the probing stage, PP selects a small yet crucial set of hidden states, based on residual importance, to run a few model layers ahead. During the history-informed pruning stage, PP strategically integrates the probing states with historical states. Subsequently, it structurally prunes weights based on the integrated states and the PP importance score, a metric developed specifically to assess the importance of each weight channel in maintaining performance. In the final stage, full inference is conducted on the remaining weights. A major advantage of PP is its compatibility with existing models, as it operates without requiring additional neural network modules or fine-tuning. Comprehensive evaluations of PP on LLaMA-2/3 and OPT models reveal that even minimal probing-using just 1.5% of FLOPs-can substantially enhance the efficiency of structured pruning of LLMs. For instance, when evaluated on LLaMA-2-7B with WikiText2, PP achieves a 2.56 times lower ratio of performance degradation per unit of runtime reduction compared to the state-of-the-art method at a 40% pruning ratio. Our code is available at this https URL. 

**Abstract (ZH)**: 我们引入了一种新的框架——探针修剪（Probe Pruning, PP），用于大型语言模型（LLMs）的在线动态结构化修剪。PP 利用了这样一个洞察：并非所有样本和令牌对模型输出的贡献是均等的。通过对每个批次中一小部分进行探针测试，PP 能够有效地识别出关键的权重，从而实现针对不同批次的定制化动态修剪。PP 包含三个主要阶段：探针测试、基于历史信息的修剪和全面推理。在探针测试阶段，PP 基于残差重要性选择一小部分关键的隐藏状态，并在此基础上运行几层模型。在历史信息引导的修剪阶段，PP 战略性地将探针状态与历史状态整合。随后，基于整合状态和 PP 重要性得分（一个专门为评估每个权重通道在保持性能方面的贡献度开发的指标），PP 进行结构性修剪。在最终阶段，对剩余的权重进行全推理。PP 的一个主要优势是它与现有模型兼容，可以在不增加额外的神经网络模块或微调的情况下运行。对于 LLaMA-2/3 和 OPT 模型的全面评估表明，即使只使用 1.5% 的 FLOPs 进行探针测试，PP 也能显著提高 LLM 的结构化修剪效率。例如，当用于评估 LLaMA-2-7B 在 WikiText2 上的表现时，即使在 40% 的剪枝比例下，PP 的性能退化与运行时间减少的比率相比，PP 的这一比率降低了 2.56 倍，相比之下，这是最先进的方法。我们的代码可以在以下网址获得：[提供网址]。 

---
# Pastiche Novel Generation Creating: Fan Fiction You Love in Your Favorite Author's Style 

**Title (ZH)**: paste其原文表述可能不太准确，“pastiche”在学术和文学领域通常指的是模仿或结合不同风格的作品，因此可以将其翻译为“仿作”，表达为：“仿作小说生成：你最喜欢作者的风格中的fan fiction”。

更符合学术规范的标题可以表达为：

“仿作小说生成：你喜爱作者的风格中的粉丝小说”

如此表达既保留了原文的核心思想，也符合学术写作的规范。 

**Authors**: Xueran Han, Yuhan Liu, Mingzhe Li, Wei Liu, Sen Hu, Rui Yan, Zhiqiang Xu, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15616)  

**Abstract**: Great novels create immersive worlds with rich character arcs, well-structured plots, and nuanced writing styles. However, current novel generation methods often rely on brief, simplistic story outlines and generate details using plain, generic language. To bridge this gap, we introduce the task of Pastiche Novel Generation, which requires the generated novels to imitate the distinctive features of the original work, including understanding character profiles, predicting plausible plot developments, and writing concrete details using vivid, expressive language. To achieve this, we propose WriterAgent, a novel generation system designed to master the core aspects of literary pastiche. WriterAgent is trained through a curriculum learning paradigm, progressing from low-level stylistic mastery to high-level narrative coherence. Its key tasks include language style learning, character modeling, plot planning, and stylish writing, ensuring comprehensive narrative control. To support this, WriterAgent leverages the WriterLoRA framework, an extension of LoRA with hierarchical and cumulative task-specific modules, each specializing in a different narrative aspect. We evaluate WriterAgent on multilingual classics like Harry Potter and Dream of the Red Chamber, demonstrating its superiority over baselines in capturing the target author's settings, character dynamics, and writing style to produce coherent, faithful narratives. 

**Abstract (ZH)**: 伟大的小说构建了充满沉浸感的世界，并具备丰富的人物弧线、结构严谨的情节和细腻的文风。然而，当前的小说生成方法往往依赖于简短且简单的故事情节大纲，并使用平庸且通用的语言生成细节。为弥补这一差距，我们提出了拟仿小说生成任务，要求生成的小说模仿原著的独特特征，包括理解人物背景、预测合理的故事情节发展，并使用生动且富有表现力的语言书写具体的细节。为实现这一目标，我们提出了WriterAgent，这是一种专为掌握文学拟仿核心要素而设计的小说生成系统。WriterAgent通过课程学习范式进行训练，从低级别的风格掌握逐步过渡到高级的情节连贯性。其关键任务包括语言风格学习、人物建模、情节规划和风格化的写作，确保全面的情节控制。为了支持这一系统，WriterAgent利用了WriterLoRA框架，这是一种基于LoRA扩展的具有层次性和累积性任务专用模块的架构，每个模块专注于不同的叙事方面。我们使用《哈利·波特》和《红楼梦》等多语言经典文学作品对WriterAgent进行了评估，结果显示其在捕捉目标作者的设定、人物动态和文风方面优于基线模型，能够生成连贯且忠实的叙事内容。 

---
# PDeepPP:A Deep learning framework with Pretrained Protein language for peptide classification 

**Title (ZH)**: PDeepPP：带有预训练蛋白质语言的深度学习框架用于肽分类 

**Authors**: Jixiu Zhai, Tianchi Lu, Haitian Zhong, Ziyang Xu, Yuhuan Liu, Xueying Wang, Dan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15610)  

**Abstract**: Protein post-translational modifications (PTMs) and bioactive peptides (BPs) play critical roles in various biological processes and have significant therapeutic potential. However, identifying PTM sites and bioactive peptides through experimental methods is often labor-intensive, costly, and time-consuming. As a result, computational tools, particularly those based on deep learning, have become effective solutions for predicting PTM sites and peptide bioactivity. Despite progress in this field, existing methods still struggle with the complexity of protein sequences and the challenge of requiring high-quality predictions across diverse datasets.
To address these issues, we propose a deep learning framework that integrates pretrained protein language models with a neural network combining transformer and CNN for peptide classification. By leveraging the ability of pretrained models to capture complex relationships within protein sequences, combined with the predictive power of parallel networks, our approach improves feature extraction while enhancing prediction accuracy.
This framework was applied to multiple tasks involving PTM site and bioactive peptide prediction, utilizing large-scale datasets to enhance the model's robustness. In the comparison across 33 tasks, the model achieved state-of-the-art (SOTA) performance in 25 of them, surpassing existing methods and demonstrating its versatility across different datasets. Our results suggest that this approach provides a scalable and effective solution for large-scale peptide discovery and PTM analysis, paving the way for more efficient peptide classification and functional annotation. 

**Abstract (ZH)**: 蛋白质翻译后修饰（PTMs）和生物活性肽（BPs）在多种生物过程中起着关键作用，并具有重要的治疗潜力。然而，通过实验方法识别PTM位点和生物活性肽通常耗时、耗力且成本高昂。因此，计算工具，尤其是基于深度学习的工具，已成为预测PTM位点和肽生物活性的有效解决方案。尽管该领域取得了进展，但现有方法仍然难以应对蛋白质序列的复杂性和在不同数据集上实现高质量预测的挑战。

为了解决这些问题，我们提出了一种深度学习框架，将预训练的蛋白质语言模型与结合了Transformer和CNN的神经网络相结合，用于肽分类。通过利用预训练模型捕获蛋白质序列中复杂关系的能力，结合平行网络的预测能力，我们的方法提高了特征提取效果并增强了预测准确性。

该框架应用于多个涉及PTM位点和生物活性肽预测的任务，利用大规模数据集增强模型的稳健性。在33项任务的比较中，该模型在25项任务中取得了目前的最先进（SOTA）性能，超过了现有方法，并且展示了其在不同数据集上的通用性。我们的结果表明，此方法为大规模肽发现和PTM分析提供了一种可扩展且有效的解决方案，为更高效的肽分类和功能注释铺平了道路。 

---
# On the Robustness of Transformers against Context Hijacking for Linear Classification 

**Title (ZH)**: 关于Transformer在线性分类中对抗上下文劫持的鲁棒性分析 

**Authors**: Tianle Li, Chenyang Zhang, Xingwu Chen, Yuan Cao, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.15609)  

**Abstract**: Transformer-based Large Language Models (LLMs) have demonstrated powerful in-context learning capabilities. However, their predictions can be disrupted by factually correct context, a phenomenon known as context hijacking, revealing a significant robustness issue. To understand this phenomenon theoretically, we explore an in-context linear classification problem based on recent advances in linear transformers. In our setup, context tokens are designed as factually correct query-answer pairs, where the queries are similar to the final query but have opposite labels. Then, we develop a general theoretical analysis on the robustness of the linear transformers, which is formulated as a function of the model depth, training context lengths, and number of hijacking context tokens. A key finding is that a well-trained deeper transformer can achieve higher robustness, which aligns with empirical observations. We show that this improvement arises because deeper layers enable more fine-grained optimization steps, effectively mitigating interference from context hijacking. This is also well supported by our numerical experiments. Our findings provide theoretical insights into the benefits of deeper architectures and contribute to enhancing the understanding of transformer architectures. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）展示了强大的上下文学习能力。然而，它们可能会受到事实正确的上下文干扰，这一现象被称为上下文劫持，揭示了一个重要的鲁棒性问题。为了从理论上理解这一现象，我们基于最近在线性变压器方面的发展，探索了一个基于线性分类的上下文学习问题。在我们的设置中，上下文令牌被设计为事实正确的查询-答案对，其中查询与最终查询相似但标签相反。然后，我们发展了一种关于线性变压器鲁棒性的通用理论分析，这种分析可以表示为模型深度、训练上下文长度和劫持上下文令牌数量的函数。一个关键发现是，经过充分训练且更深的Transformer可以实现更高的鲁棒性，这与经验观察一致。我们表明，这种改进是因为更深的层使得更精细的优化步骤成为可能，有效减轻了上下文劫持的影响。这也在我们的数值实验中得到了充分的支持。我们的研究结果提供了对更深架构优势的理论见解，并有助于增强对Transformer架构的理解。 

---
# Do Multilingual LLMs Think In English? 

**Title (ZH)**: 多语言大语言模型是否用英语思考？ 

**Authors**: Lisa Schut, Yarin Gal, Sebastian Farquhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15603)  

**Abstract**: Large language models (LLMs) have multilingual capabilities and can solve tasks across various languages. However, we show that current LLMs make key decisions in a representation space closest to English, regardless of their input and output languages. Exploring the internal representations with a logit lens for sentences in French, German, Dutch, and Mandarin, we show that the LLM first emits representations close to English for semantically-loaded words before translating them into the target language. We further show that activation steering in these LLMs is more effective when the steering vectors are computed in English rather than in the language of the inputs and outputs. This suggests that multilingual LLMs perform key reasoning steps in a representation that is heavily shaped by English in a way that is not transparent to system users. 

**Abstract (ZH)**: 大型语言模型（LLMs）具备多语言能力，并能在多种语言的任务中发挥作用。然而，我们发现当前的LLMs在进行关键决策时，倾向于使用一个与英语最接近的表示空间，而与其输入和输出的语言无关。通过使用对数几率（logit）视角探讨法语、德语、荷兰语和汉语句子的内部表示，我们发现LLMs首先生成与英语相近的表示，然后将其转换为目标语言。进一步研究表明，在这些LLMs中，通过英语文本计算触发向量比通过输入和输出语言计算更有效。这表明，多语言LLMs在执行关键推理步骤时，使用了一个深受英语影响的表示空间，而这种现象并不透明地反映给系统用户。 

---
# KAD: No More FAD! An Effective and Efficient Evaluation Metric for Audio Generation 

**Title (ZH)**: KAD:不再依赖FAD！一种有效的音频生成评估指标 

**Authors**: Yoonjin Chung, Pilsun Eu, Junwon Lee, Keunwoo Choi, Juhan Nam, Ben Sangbae Chon  

**Link**: [PDF](https://arxiv.org/pdf/2502.15602)  

**Abstract**: Although being widely adopted for evaluating generated audio signals, the Fréchet Audio Distance (FAD) suffers from significant limitations, including reliance on Gaussian assumptions, sensitivity to sample size, and high computational complexity. As an alternative, we introduce the Kernel Audio Distance (KAD), a novel, distribution-free, unbiased, and computationally efficient metric based on Maximum Mean Discrepancy (MMD). Through analysis and empirical validation, we demonstrate KAD's advantages: (1) faster convergence with smaller sample sizes, enabling reliable evaluation with limited data; (2) lower computational cost, with scalable GPU acceleration; and (3) stronger alignment with human perceptual judgments. By leveraging advanced embeddings and characteristic kernels, KAD captures nuanced differences between real and generated audio. Open-sourced in the kadtk toolkit, KAD provides an efficient, reliable, and perceptually aligned benchmark for evaluating generative audio models. 

**Abstract (ZH)**: 尽管Fréchet音频距离（FAD）广泛用于评估生成的音频信号，但它存在一些显著的局限性，包括依赖高斯假设、对样本大小敏感以及计算成本高。为了解决这些问题，我们提出了一种新的Kernel音频距离（KAD），该度量基于最大均值离散性（MMD），是一种无分布假设、无偏且计算效率高的指标。通过分析和实证验证，我们展示了KAD的优势：（1）在较小样本量下更快的收敛速度，使其能够在有限的数据下提供可靠的评估；（2）更低的计算成本，并且具有可扩展的GPU加速；（3）与人类的感知判断有更强的一致性。通过利用先进的嵌入和特征核函数，KAD能够捕捉真实和生成音频之间的细微差异。KAD已在kadtk工具包中开源，提供了评估生成音频模型的高效、可靠且感知上一致性的基准。 

---
# WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents 

**Title (ZH)**: WorldCraft：通过LLM代理创建和定制逼真3D世界 

**Authors**: Xinhang Liu, Chi-Keung Tang, Yu-Wing Tai  

**Link**: [PDF](https://arxiv.org/pdf/2502.15601)  

**Abstract**: Constructing photorealistic virtual worlds has applications across various fields, but it often requires the extensive labor of highly trained professionals to operate conventional 3D modeling software. To democratize this process, we introduce WorldCraft, a system where large language model (LLM) agents leverage procedural generation to create indoor and outdoor scenes populated with objects, allowing users to control individual object attributes and the scene layout using intuitive natural language commands. In our framework, a coordinator agent manages the overall process and works with two specialized LLM agents to complete the scene creation: ForgeIt, which integrates an ever-growing manual through auto-verification to enable precise customization of individual objects, and ArrangeIt, which formulates hierarchical optimization problems to achieve a layout that balances ergonomic and aesthetic considerations. Additionally, our pipeline incorporates a trajectory control agent, allowing users to animate the scene and operate the camera through natural language interactions. Our system is also compatible with off-the-shelf deep 3D generators to enrich scene assets. Through evaluations and comparisons with state-of-the-art methods, we demonstrate the versatility of WorldCraft, ranging from single-object customization to intricate, large-scale interior and exterior scene designs. This system empowers non-professionals to bring their creative visions to life. 

**Abstract (ZH)**: 构建逼真的虚拟世界在多个领域都有应用，但通常需要经过高度训练的专业人员使用传统的3D建模软件进行大量劳动。为了使这一过程民主化，我们提出了一种名为WorldCraft的系统，其中大型语言模型（LLM）代理利用程序化生成技术来创建室内和室外场景，并填充各种物体。用户可以通过直观的自然语言命令控制每个物体的属性和场景布局。在我们的框架中，协调代理管理整个过程，并与两个专门的LLM代理合作完成场景创作：ForgeIt，它通过自动验证不断扩展的手动操作来进行个体物体的精确定制；ArrangeIt，则通过构建层次优化问题实现兼顾人体工程学和美学的布局平衡。此外，我们的流水线还包含一个轨迹控制代理，允许用户通过自然语言交互来使场景动画化并操作相机。我们的系统也兼容现成的深度3D生成器以丰富场景资产。通过评估并将我们的方法与当前最先进的技术进行比较，我们展示了WorldCraft的多样性和灵活性，从单个物体的定制到复杂的大型室内和室外场景设计。该系统使非专业人员能够实现他们的创意构想。 

---
# Generalizing From Short to Long: Effective Data Synthesis for Long-Context Instruction Tuning 

**Title (ZH)**: 从短到长的泛化：有效的数据合成用于长上下文指令调优 

**Authors**: Wenhao Zhu, Pinzhen Chen, Hanxu Hu, Shujian Huang, Fei Yuan, Jiajun Chen, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2502.15592)  

**Abstract**: Long-context modelling for large language models (LLMs) has been a key area of recent research because many real world use cases require reasoning over longer inputs such as documents. The focus of research into modelling long context has been on how to model position and there has been little investigation into other important aspects of language modelling such as instruction tuning. Long context training examples are challenging and expensive to create and use. In this paper, we investigate how to design instruction data for the post-training phase of a long context pre-trained model: how much and what type of context is needed for optimal and efficient post-training. Our controlled study reveals that models instruction-tuned on short contexts can effectively generalize to longer ones, while also identifying other critical factors such as instruction difficulty and context composition. Based on these findings, we propose context synthesis, a novel data synthesis framework that leverages off-the-shelf LLMs to generate extended background contexts for high-quality instruction-answer pairs. Experiment results on the document-level benchmark (LongBench) demonstrate that our proposed approach outperforms previous instruction synthesis approaches and comes close to the performance of human-annotated long-context instruction data. The project will be available at: this https URL. 

**Abstract (ZH)**: 近年来，大语言模型（LLMs）的长上下文建模已成为研究的重点领域，因为许多实际应用场景需要对较长的输入（如文档）进行推理。关于长上下文建模的研究主要集中在位置建模方面，对于语言模型的其他重要方面，如指令调优，研究相对较少。构建和使用长上下文训练样本具有挑战性和成本高。本文探讨了如何为预训练的长上下文模型的后训练阶段设计指令数据：需要多少以及什么类型的上下文才能实现最优和高效的后训练。我们的受控研究发现，针对短上下文进行指令调优的模型可以有效地泛化到长上下文，同时还指出了其他关键因素，如指令难度和上下文构成。基于这些发现，我们提出了一种新的数据合成框架——上下文合成，该框架利用现成的LLMs生成高质量指令-回答对的扩展背景上下文。在文档级别基准（LongBench）上的实验结果表明，我们提出的方法优于之前的指令合成方法，并且其性能接近人工标注的长上下文指令数据的表现。项目详情请参见：[这个链接](this https URL)。 

---
# LightThinker: Thinking Step-by-Step Compression 

**Title (ZH)**: LightThinker：逐步压缩思考 

**Authors**: Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15589)  

**Abstract**: Large language models (LLMs) have shown remarkable performance in complex reasoning tasks, but their efficiency is hindered by the substantial memory and computational costs associated with generating lengthy tokens. In this paper, we propose LightThinker, a novel method that enables LLMs to dynamically compress intermediate thoughts during reasoning. Inspired by human cognitive processes, LightThinker compresses verbose thought steps into compact representations and discards the original reasoning chains, thereby significantly reducing the number of tokens stored in the context window. This is achieved by training the model on when and how to perform compression through data construction, mapping hidden states to condensed gist tokens, and creating specialized attention masks. Additionally, we introduce the Dependency (Dep) metric to quantify the degree of compression by measuring the reliance on historical tokens during generation. Extensive experiments on four datasets and two models show that LightThinker reduces peak memory usage and inference time, while maintaining competitive accuracy. Our work provides a new direction for improving the efficiency of LLMs in complex reasoning tasks without sacrificing performance. Code will be released at this https URL. 

**Abstract (ZH)**: 以下是经过学术规范翻译的内容：

大规模语言模型（LLMs）在复杂推理任务中展现出了显著的性能，但它们在生成长令牌时受到大量内存和计算成本的限制。本文提出了一种名为LightThinker的新型方法，该方法能够在推理过程中动态压缩中间思维。借鉴人类认知过程，LightThinker将冗长的思维步骤压缩成紧凑表示，并丢弃原始推理链，从而显著减少了存储在上下文窗口中的令牌数量。这一目标是通过数据构造训练模型何时以及如何执行压缩，将隐藏状态映射到浓缩核心令牌，并创建专门的注意力掩码来实现的。此外，我们引入了依赖性（Dep）指标来通过衡量生成过程中对历史令牌的依赖程度来量化压缩的程度。在四个数据集和两种模型上进行的广泛实验表明，LightThinker能够减少峰值内存使用量和推断时间，同时保持竞争力的准确度。我们的工作为改进复杂推理任务中LLMs的效率提供了一个新的方向，而不牺牲性能。代码将在以下网址发布：https://your-repository-url.com。 

---
# Improving the Scaling Laws of Synthetic Data with Deliberate Practice 

**Title (ZH)**: 通过刻意练习提高合成数据的标度律 

**Authors**: Reyhane Askari-Hemmat, Mohammad Pezeshki, Elvis Dohmatob, Florian Bordes, Pietro Astolfi, Melissa Hall, Jakob Verbeek, Michal Drozdzal, Adriana Romero-Soriano  

**Link**: [PDF](https://arxiv.org/pdf/2502.15588)  

**Abstract**: Inspired by the principle of deliberate practice in human learning, we propose Deliberate Practice for Synthetic Data Generation (DP), a novel framework that improves sample efficiency through dynamic synthetic data generation. Prior work has shown that scaling synthetic data is inherently challenging, as naively adding new data leads to diminishing returns. To address this, pruning has been identified as a key mechanism for improving scaling, enabling models to focus on the most informative synthetic samples. Rather than generating a large dataset and pruning it afterward, DP efficiently approximates the direct generation of informative samples. We theoretically show how training on challenging, informative examples improves scaling laws and empirically validate that DP achieves better scaling performance with significantly fewer training samples and iterations. On ImageNet-100, DP generates 3.4x fewer samples and requires six times fewer iterations, while on ImageNet-1k, it generates 8x fewer samples with a 30 percent reduction in iterations, all while achieving superior performance compared to prior work. 

**Abstract (ZH)**: 受人类学习中刻意练习原则的启发，我们提出了一种名为 Deliberate Practice for Synthetic Data Generation (DP) 的新框架，该框架通过动态生成合成数据来提高样本效率。之前的工作已经表明，简单地增加数据量在合成数据扩展方面是固有地具有挑战性的，因为新增数据会导致边际效益递减。为了解决这一问题，剪枝已被确认为一种关键机制，能够改善扩展性，使模型能够关注最具信息性的合成样本。DP 不是先生成大量数据再进行剪枝，而是有效地近似直接生成具有信息性的样本。我们从理论上证明了在具有挑战性和信息性的样本上进行训练如何改善扩展定律，并通过实验证明DP在显著减少训练样本和迭代次数的情况下实现了更好的扩展性能。在 ImageNet-100 上，DP 生成的样本数量减少了 3.4 倍，迭代次数减少了六倍；而在 ImageNet-1k 上，它生成的样本数量减少了 8 倍，迭代次数减少了 30%，同时相比之前的工作取得了更好的性能。 

---
# Feature maps for the Laplacian kernel and its generalizations 

**Title (ZH)**: 拉普拉斯核及其推广的特征映射 

**Authors**: Sudhendu Ahir, Parthe Pandit  

**Link**: [PDF](https://arxiv.org/pdf/2502.15575)  

**Abstract**: Recent applications of kernel methods in machine learning have seen a renewed interest in the Laplacian kernel, due to its stability to the bandwidth hyperparameter in comparison to the Gaussian kernel, as well as its expressivity being equivalent to that of the neural tangent kernel of deep fully connected networks. However, unlike the Gaussian kernel, the Laplacian kernel is not separable. This poses challenges for techniques to approximate it, especially via the random Fourier features (RFF) methodology and its variants. In this work, we provide random features for the Laplacian kernel and its two generalizations: Matérn kernel and the Exponential power kernel. We provide efficiently implementable schemes to sample weight matrices so that random features approximate these kernels. These weight matrices have a weakly coupled heavy-tailed randomness. Via numerical experiments on real datasets we demonstrate the efficacy of these random feature maps. 

**Abstract (ZH)**: 近年来，核方法在机器学习中的应用重新引起了对拉普拉斯核的兴趣，这主要是因为与高斯核相比，拉普拉斯核在带宽超参数稳定性方面的优势，以及其表达能力等同于深层全连接网络的神经极限核。然而，与高斯核不同，拉普拉斯核是非可分的。这为通过随机傅里叶特征（RFF）方法及其变体来近似它带来了挑战。在本研究中，我们为拉普拉斯核及其两种推广形式——马特ERN核和指数功率核——提供了随机特征。我们提出了有效实现的方法来采样权重矩阵，以使随机特征近似这些核。这些权重矩阵具有弱耦合的厚尾随机性。通过在实际数据集上的数值实验，我们展示了这些随机特征映射的有效性。 

---
# A Cautionary Tale About "Neutrally" Informative AI Tools Ahead of the 2025 Federal Elections in Germany 

**Title (ZH)**: 2025年德国联邦选举前夕关于“中立”信息性AI工具的警示故事 

**Authors**: Ina Dormuth, Sven Franke, Marlies Hafer, Tim Katzke, Alexander Marx, Emmanuel Müller, Daniel Neider, Markus Pauly, Jérôme Rutinowski  

**Link**: [PDF](https://arxiv.org/pdf/2502.15568)  

**Abstract**: In this study, we examine the reliability of AI-based Voting Advice Applications (VAAs) and large language models (LLMs) in providing objective political information. Our analysis is based upon a comparison with party responses to 38 statements of the Wahl-O-Mat, a well-established German online tool that helps inform voters by comparing their views with political party positions. For the LLMs, we identify significant biases. They exhibit a strong alignment (over 75% on average) with left-wing parties and a substantially lower alignment with center-right (smaller 50%) and right-wing parties (around 30%). Furthermore, for the VAAs, intended to objectively inform voters, we found substantial deviations from the parties' stated positions in Wahl-O-Mat: While one VAA deviated in 25% of cases, another VAA showed deviations in more than 50% of cases. For the latter, we even observed that simple prompt injections led to severe hallucinations, including false claims such as non-existent connections between political parties and right-wing extremist ties. 

**Abstract (ZH)**: 在本研究中，我们考察了基于AI的投票建议应用（VAAs）和大型语言模型（LLMs）在提供客观政治信息方面的可靠性。我们的分析基于对Wahl-O-Mat（一个广泛使用的德国在线工具）38项陈述的政党回复进行比较，Wahl-O-Mat有助于 voters 根据其观点与政治党派立场进行比较。对于大型语言模型，我们发现了显著的偏差。这些模型在平均75%以上的时间内与左翼政党高度一致，而与中间偏右（较小比例约为50%）和右翼政党（约30%）的立场一致性则低得多。此外，对于旨在客观告知选民的VAAs，我们在其提供的信息与Wahl-O-Mat中政党表明的立场之间发现了显著偏差：其中一个 VAA 在25%的情况下偏离了政党立场，而另一个 VAA 在超过50%的情况下偏离了政党立场。对于后者，我们甚至观察到简单的提示注入导致了严重的幻觉现象，包括错误声称某些政党和极右翼之间的不存在的实际联系。 

---
# Bridging vision language model (VLM) evaluation gaps with a framework for scalable and cost-effective benchmark generation 

**Title (ZH)**: 使用可扩展且成本效益高的基准生成框架弥合视觉语言模型（VLM）评估差距 

**Authors**: Tim Rädsch, Leon Mayer, Simon Pavicic, A. Emre Kavur, Marcel Knopp, Barış Öztürk, Klaus Maier-Hein, Paul F. Jaeger, Fabian Isensee, Annika Reinke, Lena Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2502.15563)  

**Abstract**: Reliable evaluation of AI models is critical for scientific progress and practical application. While existing VLM benchmarks provide general insights into model capabilities, their heterogeneous designs and limited focus on a few imaging domains pose significant challenges for both cross-domain performance comparison and targeted domain-specific evaluation. To address this, we propose three key contributions: (1) a framework for the resource-efficient creation of domain-specific VLM benchmarks enabled by task augmentation for creating multiple diverse tasks from a single existing task, (2) the release of new VLM benchmarks for seven domains, created according to the same homogeneous protocol and including 162,946 thoroughly human-validated answers, and (3) an extensive benchmarking of 22 state-of-the-art VLMs on a total of 37,171 tasks, revealing performance variances across domains and tasks, thereby supporting the need for tailored VLM benchmarks. Adoption of our methodology will pave the way for the resource-efficient domain-specific selection of models and guide future research efforts toward addressing core open questions. 

**Abstract (ZH)**: 可靠评估人工智能模型对于科学进步和实际应用至关重要。虽然现有的视觉语言模型(Vision Language Models, VLM)基准提供了关于模型能力的通用见解，但它们不同的设计和对少数成像领域的关注有限，这些挑战阻碍了跨领域性能比較和特定领域评价的针对性。为了解决这一问题，我们提出了三个主要贡献：（1）一种利用任务增强方法从单一现有任务生成多个多样的任务，从而促进资源高效创建特定领域VLM基准的框架；（2）发布针对七个领域的新的VLM基准，按照相同的标准化协议生成，并包含162,946个详细的人工验证答案；（3）全面评估22个最先进的VLM模型在总计37,171个任务上的表现，揭示不同领域和任务之间的性能差异，从而支持制定定制化VLM基准的必要性。采用我们的方法将为特定领域的模型选择提供资源效率，并指导未来研究努力解决核心开放问题。 

---
# PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning 

**Title (ZH)**: PIP-KAG：通过参数精简减轻知识增强生成中的知识冲突 

**Authors**: Pengcheng Huang, Zhenghao Liu, Yukun Yan, Xiaoyuan Yi, Hao Chen, Zhiyuan Liu, Maosong Sun, Tong Xiao, Ge Yu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.15543)  

**Abstract**: Knowledge-Augmented Generation (KAG) has shown great promise in updating the internal memory of Large Language Models (LLMs) by integrating external knowledge. However, KAG inevitably faces knowledge conflicts when the internal memory contradicts external information. Current approaches to mitigating these conflicts mainly focus on improving external knowledge utilization. However, these methods have shown only limited effectiveness in mitigating the knowledge conflict problem, as internal knowledge continues to influence the generation process of LLMs. In this paper, we propose a ParametrIc Pruning-based Knowledge-Augmented Generation (PIP-KAG) approach, which prunes internal knowledge of LLMs and incorporates a plug-and-play adaptation module to help LLMs better leverage external sources. Additionally, we construct the CoConflictQA benchmark based on the hallucination of LLMs to better evaluate contextual faithfulness during answering questions. Experimental results on CoConflictQA demonstrate that PIP-KAG significantly reduces knowledge conflicts and improves context fidelity. Notably, PIP-KAG reduces LLM's parameters by 13%, enhancing parameter efficiency in LLMs within the KAG framework. All codes are available at this https URL. 

**Abstract (ZH)**: 知识增强生成（KAG）在通过集成外部知识更新大型语言模型（LLMs）内部记忆方面展现出了巨大的潜力。然而，KAG 在处理内部记忆与外部信息矛盾时不可避免地会遇到知识冲突。当前缓解这些冲突的方法主要集中在提高外部知识的利用效率上。然而，这些方法在缓解知识冲突问题方面仅展示了有限的有效性，因为内部知识仍然会不断影响LLMs的生成过程。在本文中，我们提出了一种基于参数裁剪的知识增强生成方法（ParametrIc Pruning-based Knowledge-Augmented Generation, PIP-KAG），该方法裁剪了LLMs的内部知识，并引入了一个插件式适应模块，以帮助LLMs更好地利用外部来源。此外，我们基于LLMs的幻觉构造了CoConflictQA基准，以更好地评估回答问题时的上下文忠实度。在CoConflictQA上的实验结果表明，PIP-KAG 显著减少了知识冲突，并提高了上下文忠实度。值得注意的是，PIP-KAG 将LLMs的参数减少了13%，从而在KAG框架内增强了参数效率。所有代码可通过以下链接获取：[此 https URL](https://this.is/a/link)。 

---
# Bridging Domain Gaps between Pretrained Multimodal Models and Recommendations 

**Title (ZH)**: 预训练多模态模型与推荐系统间领域差距的桥梁构建 

**Authors**: Wenyu Zhang, Jie Luo, Xinming Zhang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15542)  

**Abstract**: With the explosive growth of multimodal content online, pre-trained visual-language models have shown great potential for multimodal recommendation. However, while these models achieve decent performance when applied in a frozen manner, surprisingly, due to significant domain gaps (e.g., feature distribution discrepancy and task objective misalignment) between pre-training and personalized recommendation, adopting a joint training approach instead leads to performance worse than baseline. Existing approaches either rely on simple feature extraction or require computationally expensive full model fine-tuning, struggling to balance effectiveness and efficiency. To tackle these challenges, we propose \textbf{P}arameter-efficient \textbf{T}uning for \textbf{M}ultimodal \textbf{Rec}ommendation (\textbf{PTMRec}), a novel framework that bridges the domain gap between pre-trained models and recommendation systems through a knowledge-guided dual-stage parameter-efficient training strategy. This framework not only eliminates the need for costly additional pre-training but also flexibly accommodates various parameter-efficient tuning methods. 

**Abstract (ZH)**: 随着在线多模态内容的爆炸性增长，预训练的视觉-语言模型在多模态推荐方面展现了巨大的潜力。然而，当这些模型以静态方式应用时，它们能够取得不错的性能，但令人意外的是，由于预训练与个性化推荐之间的显著领域差距（如特征分布差异和任务目标不匹配），采用联合训练方法反而会导致性能低于基线。现有方法要么依赖简单的特征提取，要么需要进行计算密集型的完整模型微调，难以在有效性和效率之间取得平衡。为了解决这些挑战，我们提出了一种名为**参数高效调优以进行多模态推荐**（PTMRec）的新颖框架，该框架通过知识引导的两阶段参数高效训练策略，将预训练模型与推荐系统之间的领域差距进行桥接。该框架不仅消除了成本高昂的额外预训练需求，还灵活地兼容了各种参数高效调优方法。 

---
# Depth-aware Fusion Method based on Image and 4D Radar Spectrum for 3D Object Detection 

**Title (ZH)**: 基于图像和4D雷达谱的深度aware融合方法用于三维物体检测 

**Authors**: Yue Sun, Yeqiang Qian, Chunxiang Wang, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15516)  

**Abstract**: Safety and reliability are crucial for the public acceptance of autonomous driving. To ensure accurate and reliable environmental perception, intelligent vehicles must exhibit accuracy and robustness in various environments. Millimeter-wave radar, known for its high penetration capability, can operate effectively in adverse weather conditions such as rain, snow, and fog. Traditional 3D millimeter-wave radars can only provide range, Doppler, and azimuth information for objects. Although the recent emergence of 4D millimeter-wave radars has added elevation resolution, the radar point clouds remain sparse due to Constant False Alarm Rate (CFAR) operations. In contrast, cameras offer rich semantic details but are sensitive to lighting and weather conditions. Hence, this paper leverages these two highly complementary and cost-effective sensors, 4D millimeter-wave radar and camera. By integrating 4D radar spectra with depth-aware camera images and employing attention mechanisms, we fuse texture-rich images with depth-rich radar data in the Bird's Eye View (BEV) perspective, enhancing 3D object detection. Additionally, we propose using GAN-based networks to generate depth images from radar spectra in the absence of depth sensors, further improving detection accuracy. 

**Abstract (ZH)**: 安全性和可靠性是自动驾驶技术被公众接受的关键。为了保证环境感知的准确性和可靠性，智能车辆必须在各种环境下都表现出高度的精确性和鲁棒性。毫米波雷达因其较强的穿透能力，能够在雨、雪、雾等不良天气条件下有效工作。传统的3D毫米波雷达只能提供物体的距离、多普勒和方位信息。尽管最近出现了4D毫米波雷达，增加了垂直分辨率，但由于恒定虚警率（CFAR）操作，雷达点云仍然较为稀疏。相比之下，摄像头提供了丰富的语义信息，但对光照和天气条件敏感。因此，本文利用这两种互补的低成本传感器——4D毫米波雷达和摄像头。通过结合4D雷达频谱与深度感知的摄像头图像，并采用注意力机制，我们将纹理丰富的图像与深度丰富的雷达数据在鸟瞰视角（Bird's Eye View, BEV）中进行融合，增强三维物体检测的能力。此外，我们提出了使用基于生成对抗网络（GAN）的网络从雷达频谱中生成深度图像的方法，进一步提高检测精度。 

---
# Activation Steering in Neural Theorem Provers 

**Title (ZH)**: 神经定理证明中的激活转向 

**Authors**: Shashank Kirtania  

**Link**: [PDF](https://arxiv.org/pdf/2502.15507)  

**Abstract**: Large Language Models (LLMs) have shown promise in proving formal theorems using proof assistants like Lean. However, current state of the art language models struggles to predict next step in proofs leading practitioners to use different sampling techniques to improve LLMs capabilities. We observe that the LLM is capable of predicting the correct tactic; however, it faces challenges in ranking it appropriately within the set of candidate tactics, affecting the overall selection process. To overcome this hurdle, we use activation steering to guide LLMs responses to improve the generations at the time of inference. Our results suggest that activation steering offers a promising lightweight alternative to specialized fine-tuning for enhancing theorem proving capabilities in LLMs, particularly valuable in resource-constrained environments. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在使用Lean等证明助手证明形式定理方面显示出潜力。然而，当前最先进的语言模型在预测证明中的下一步时遇到困难，这促使实践者采用不同的采样技术以提高LLMs的能力。我们观察到LLMs能够预测正确的策略，但在候选策略集中适当排名该策略方面面临挑战，从而影响整体选择过程。为了克服这一障碍，我们使用激活调节来引导LLMs的响应，以在推理时提高生成效果。我们的结果表明，激活调节提供了增强LLMs证明能力的一种有前景的轻量级替代方案，特别是在资源受限的环境中尤为有价值。 

---
# BAN: Neuroanatomical Aligning in Auditory Recognition between Artificial Neural Network and Human Cortex 

**Title (ZH)**: BAN：人工神经网络与人类皮层在听觉识别中的神经解剖学对齐 

**Authors**: Haidong Wang, Pengfei Xiao, Ao Liu, Jianhua Zhang, Qia Shan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15503)  

**Abstract**: Drawing inspiration from neurosciences, artificial neural networks (ANNs) have evolved from shallow architectures to highly complex, deep structures, yielding exceptional performance in auditory recognition tasks. However, traditional ANNs often struggle to align with brain regions due to their excessive depth and lack of biologically realistic features, like recurrent connection. To address this, a brain-like auditory network (BAN) is introduced, which incorporates four neuroanatomically mapped areas and recurrent connection, guided by a novel metric called the brain-like auditory score (BAS). BAS serves as a benchmark for evaluating the similarity between BAN and human auditory recognition pathway. We further propose that specific areas in the cerebral cortex, mainly the middle and medial superior temporal (T2/T3) areas, correspond to the designed network structure, drawing parallels with the brain's auditory perception pathway. Our findings suggest that the neuroanatomical similarity in the cortex and auditory classification abilities of the ANN are well-aligned. In addition to delivering excellent performance on a music genre classification task, the BAN demonstrates a high BAS score. In conclusion, this study presents BAN as a recurrent, brain-inspired ANN, representing the first model that mirrors the cortical pathway of auditory recognition. 

**Abstract (ZH)**: 借鉴神经科学的原理，人工神经网络（ANNs）从浅层架构发展成为结构复杂、层次众多的深度网络，在听觉识别任务中取得了出色的表现。然而，传统的ANNs常常难以与大脑区域对齐，因为它们的深度过深且缺乏生物学上的真实性特征，如循环连接。为解决这一问题，我们引入了一种类似大脑的听觉网络（BAN），该网络融合了四个按神经解剖学映射的区域，并引入了循环连接，同时受到一种新型的度量标准——类似大脑的听觉评分（BAS）的指导。BAS充当了一个基准，用于评估BAN与人类听觉识别路径的相似性。此外，我们进一步提出，大脑皮层中的特定区域，主要是上颞中区（T2/T3）和上颞外侧区（T7/T8），与设计的网络结构相对应，这与大脑听觉感知路径相呼应。我们的研究结果表明，皮层的神经解剖学相似性和ANN的听觉分类能力是高度一致的。除了在音乐流派分类任务中表现出色之外，BAN还表现出较高的BAS评分。总之，本研究将BAN展示为一种具有循环连接且受大脑启发的ANN模型，这代表了第一个能够模仿听觉识别皮层路径的模型。 

---
# Q-PETR: Quant-aware Position Embedding Transformation for Multi-View 3D Object Detection 

**Title (ZH)**: Q-PETR：量化感知位置嵌入变换在多视图3D目标检测中的应用 

**Authors**: Jiangyong Yu, Changyong Shu, Dawei Yang, Zichen Yu, Xing Hu, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15488)  

**Abstract**: PETR-based methods have dominated benchmarks in 3D perception and are increasingly becoming a key component in modern autonomous driving systems. However, their quantization performance significantly degrades when INT8 inference is required, with a degradation of 58.2% in mAP and 36.9% in NDS on the NuScenes dataset. To address this issue, we propose a quantization-aware position embedding transformation for multi-view 3D object detection, termed Q-PETR. Q-PETR offers a quantizationfriendly and deployment-friendly architecture while preserving the original performance of PETR. It substantially narrows the accuracy gap between INT8 and FP32 inference for PETR-series methods. Without bells and whistles, our approach reduces the mAP and NDS drop to within 1% under standard 8-bit per-tensor post-training quantization. Furthermore, our method exceeds the performance of the original PETR in terms of floating-point precision. Extensive experiments across a variety of PETR-series models demonstrate its broad generalization. 

**Abstract (ZH)**: 基于PETR的方法在3D感知基准测试中占据主导地位，并且越来越多地成为现代自动驾驶系统的关键组成部分。然而，当需要进行INT8推断时，它们的量化性能显著下降，NuScenes数据集上的mAP下降58.2%，NDS下降36.9%。为了解决这一问题，我们提出了一种适用于多视图3D物体检测的量化感知位置嵌入转换方法，称为Q-PETR。Q-PETR提供了一种既适合量化又便于部署的架构，同时保持了PETR的原始性能。它显著缩小了PETR系列方法在INT8和FP32推断之间的准确度差距。在标准每张量8位后训练量化下，我们的方法将mAP和NDS下降控制在1%以内。此外，我们的方法在浮点精度方面超过了原始PETR。广泛的实验涵盖了多种PETR系列模型，证明了其广泛的泛化能力。 

---
# ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models 

**Title (ZH)**: ExpliCa: 评估大型语言模型中的显式因果推理能力 

**Authors**: Martina Miliani, Serenna Auriemma, Alessandro Bondielli, Emmanuele Chersoni, Lucia Passaro, Irene Sucameli, Alessandro Lenci  

**Link**: [PDF](https://arxiv.org/pdf/2502.15487)  

**Abstract**: Large Language Models (LLMs) are increasingly used in tasks requiring interpretive and inferential accuracy. In this paper, we introduce ExpliCa, a new dataset for evaluating LLMs in explicit causal reasoning. ExpliCa uniquely integrates both causal and temporal relations presented in different linguistic orders and explicitly expressed by linguistic connectives. The dataset is enriched with crowdsourced human acceptability ratings. We tested LLMs on ExpliCa through prompting and perplexity-based metrics. We assessed seven commercial and open-source LLMs, revealing that even top models struggle to reach 0.80 accuracy. Interestingly, models tend to confound temporal relations with causal ones, and their performance is also strongly influenced by the linguistic order of the events. Finally, perplexity-based scores and prompting performance are differently affected by model size. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用于需要解释性和推理准确性的任务。在本文中，我们介绍了ExpliCa，这是一个新的数据集，用于评估LLMs在明确因果推理方面的能力。ExpliCa独特地结合了以不同语言顺序呈现的因果关系和时间关系，并通过语言连词明确表达。该数据集还包含了众包的人类接受度评分。我们通过提示和困惑度（perplexity）度量方法对LLMs进行了测试。我们评估了七个商业和开源的LLMs，结果显示即使是顶级模型也难以达到80%的准确率。有趣的是，模型往往将时间关系与因果关系混淆，而事件的语序也强烈影响其性能。最后，困惑度得分和提示性能的不同受模型规模的影响也不同。 

---
# Enhancing RWKV-based Language Models for Long-Sequence Text Generation 

**Title (ZH)**: 基于RWKV的语言模型在长序列文本生成中的增强研究 

**Authors**: Xinghan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15485)  

**Abstract**: This paper presents an enhanced RWKV-based language generation model designed to improve long-sequence text processing. We propose an adaptive token shift and gating mechanism to better capture long-range dependencies in text generation. Through a series of experiments, we compare the baseline RWKV model with the enhanced model, evaluating performance in terms of forward propagation time, text generation quality, and automatic evaluation metrics such as perplexity, BLEU, and ROUGE. Experimental results show that the enhanced model significantly improves generation quality, especially in BLEU and ROUGE scores, and demonstrates stronger context-capturing ability in long-text generation tasks. 

**Abstract (ZH)**: 本文提出了一种增强型RWKV基于的语言生成模型，旨在改进长序列文本处理。我们提出了一种自适应的标记移位和门控机制，以更好地捕捉文本生成中的长范围依赖关系。通过一系列实验，我们将基线RWKV模型与增强模型进行了比较，评估了前向传播时间、文本生成质量以及诸如困惑度、BLEU和ROUGE等自动评价指标。实验结果表明，增强型模型显著提升了生成质量，特别是在BLEU和ROUGE分数上表现更为优异，并且在长文本生成任务中展现出更强的上下文捕捉能力。 

---
# PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System 

**Title (ZH)**: PAPI：利用计算系统中存储计算能力进行大型语言模型解码的动态并行性exploitation 

**Authors**: Yintao He, Haiyu Mao, Christina Giannoula, Mohammad Sadrosadati, Juan Gómez-Luna, Huawei Li, Xiaowei Li, Ying Wang, Onur Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15470)  

**Abstract**: Large language models (LLMs) are widely used for natural language understanding and text generation. An LLM model relies on a time-consuming step called LLM decoding to generate output tokens. Several prior works focus on improving the performance of LLM decoding using parallelism techniques, such as batching and speculative decoding. State-of-the-art LLM decoding has both compute-bound and memory-bound kernels. Some prior works statically identify and map these different kernels to a heterogeneous architecture consisting of both processing-in-memory (PIM) units and computation-centric accelerators. We observe that characteristics of LLM decoding kernels (e.g., whether or not a kernel is memory-bound) can change dynamically due to parameter changes to meet user and/or system demands, making (1) static kernel mapping to PIM units and computation-centric accelerators suboptimal, and (2) one-size-fits-all approach of designing PIM units inefficient due to a large degree of heterogeneity even in memory-bound kernels.
In this paper, we aim to accelerate LLM decoding while considering the dynamically changing characteristics of the kernels involved. We propose PAPI (PArallel Decoding with PIM), a PIM-enabled heterogeneous architecture that exploits dynamic scheduling of compute-bound or memory-bound kernels to suitable hardware units. PAPI has two key mechanisms: (1) online kernel characterization to dynamically schedule kernels to the most suitable hardware units at runtime and (2) a PIM-enabled heterogeneous computing system that harmoniously orchestrates both computation-centric processing units and hybrid PIM units with different computing capabilities. Our experimental results on three broadly-used LLMs show that PAPI achieves 1.8$\times$ and 11.1$\times$ speedups over a state-of-the-art heterogeneous LLM accelerator and a state-of-the-art PIM-only LLM accelerator, respectively. 

**Abstract (ZH)**: 大规模语言模型（LLMs）广泛应用于自然语言理解和文本生成。LLM模型依赖于一个耗时的步骤，即LLM解码，来生成输出标记。先前的研究主要集中在通过并行技术（如批量处理和推测性解码）来提高LLM解码的性能。当前最先进的LLM解码既有计算密集型又有内存密集型的操作内核。一些先前的工作通过静态地识别和将这些不同类型的内核映射到由内存计算（PIM）单元和计算密集型加速器组成的异构架构中来改进LLM解码。我们观察到，LLM解码内核的特点（例如，内核是否内存密集型）可能由于参数调整以满足用户和/或系统需求而动态变化，这使得（1）静态内核映射到PIM单元和计算密集型加速器变得不理想，以及（2）针对所有内存密集型内核设计统一PIM单元的一种可适应所有情况的方法变得不够高效，即使在内存密集型内核中也存在较大的异质性。

本文旨在在考虑与解码过程相关内核动态变化特点的同时加速LLM解码。我们提出了一种PAPI（PArallel Decoding with PIM）架构，这是一种PIM启用的异构架构，能够通过动态调度计算密集型或内存密集型内核到合适的硬件单元来利用动态调度机制。PAPI有两种关键机制：（1）在线内核表征，用于在运行时动态调度内核到最适合的硬件单元；（2）一种PIM启用的混合计算系统，能够和谐地协调计算密集型处理单元和具有不同计算能力的混合PIM单元。我们在三个广泛使用的LLM上进行了实验，结果显示PAPI分别在与现有最先进的异构LLM加速器和PIM仅有的LLM加速器相比时，分别实现了1.8倍和11.1倍的加速。 

---
# Mitigating Data Scarcity in Time Series Analysis: A Foundation Model with Series-Symbol Data Generation 

**Title (ZH)**: 缓解时间序列分析中的数据稀缺性：基于系列符号数据生成的基模模型 

**Authors**: Wenxuan Wang, Kai Wu, Yujian Betterest Li, Dan Wang, Xiaoyu Zhang, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15466)  

**Abstract**: Foundation models for time series analysis (TSA) have attracted significant attention. However, challenges such as data scarcity and data imbalance continue to hinder their development. To address this, we consider modeling complex systems through symbolic expressions that serve as semantic descriptors of time series. Building on this concept, we introduce a series-symbol (S2) dual-modulity data generation mechanism, enabling the unrestricted creation of high-quality time series data paired with corresponding symbolic representations. Leveraging the S2 dataset, we develop SymTime, a pre-trained foundation model for TSA. SymTime demonstrates competitive performance across five major TSA tasks when fine-tuned with downstream task, rivaling foundation models pre-trained on real-world datasets. This approach underscores the potential of dual-modality data generation and pretraining mechanisms in overcoming data scarcity and enhancing task performance. 

**Abstract (ZH)**: 时间序列分析（TSA）的基础模型已引起了广泛关注。然而，数据稀缺性和数据不平衡等问题继续阻碍其发展。为解决这一问题，我们通过符号表达式对时间序列进行建模，这些符号表达式可以作为时间序列的语义描述符。在此基础上，我们引入了一种系列-符号（S2）双模性数据生成机制，能够不受限制地生成高质量的时间序列数据及其相应的符号表示。借助S2数据集，我们开发了SymTime，这是一种预训练的基础模型，适用于TSA。当SymTime在下游任务中进行微调时，其在五个主要TSA任务上表现出竞争性的性能，可与在真实数据集上预训练的基础模型相媲美。该方法突显了双模性数据生成和预训练机制在克服数据稀缺性和提升任务性能方面的潜在价值。 

---
# R-LoRA: Random Initialization of Multi-Head LoRA for Multi-Task Learning 

**Title (ZH)**: R-LoRA：多头LoRA的随机初始化在多任务学习中的应用 

**Authors**: Jinda Liu, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15455)  

**Abstract**: Fine-tuning large language models (LLMs) is prohibitively expensive in terms of computational and memory costs. Low-rank Adaptation (LoRA), as one of the most popular parameter-efficient fine-tuning (PEFT) methods, offers a cost-effective alternative by approximating the model changes $\Delta W \in \mathbb{R}^{m \times n}$ through the product of down-projection matrix $A \in \mathbb{R}^{m \times r}$ and head matrix $B \in \mathbb{R}^{r \times n}$, where $r \ll \min(m, n)$. In real-world scenarios, LLMs are fine-tuned on data from multiple domains to perform tasks across various fields, embodying multi-task learning (MTL). LoRA often underperforms in such complex scenarios. To enhance LoRA's capability in multi-task learning, we propose R-LoRA, which incorporates Multi-Head Randomization. Multi-Head Randomization diversifies the head matrices through Multi-Head Random Initialization and Multi-Head Dropout, enabling more efficient learning of task-specific features while maintaining shared knowledge representation. Extensive experiments demonstrate that R-LoRA is better at capturing task-specific knowledge, thereby improving performance in multi-task scenarios. The code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的微调在计算和内存成本上是极其昂贵的。作为一种最受欢迎的参数效率微调（PEFT）方法，低秩适应（LoRA）通过将模型变化 $\Delta W \in \mathbb{R}^{m \times n}$ 近似为下投影矩阵 $A \in \mathbb{R}^{m \times r}$ 和头矩阵 $B \in \mathbb{R}^{r \times n}$ 的乘积，提供了成本效益较高的替代方案，其中 $r \ll \min(m, n)$。在实际场景中，LLMs 被跨多个领域进行微调以执行各种领域的任务，体现了多任务学习（MTL）。在这样的复杂场景中，LoRA 经常表现不佳。为了增强 LoRA 在多任务学习中的能力，我们提出了 R-LoRA，该方法结合了 Multi-Head 随机化。Multi-Head 随机化通过 Multi-Head 随机初始化和 Multi-Head 丢弃来多样化头矩阵，从而在保持共享知识表示的同时更有效地学习任务特定的特征。广泛的经验表明，R-LoRA 更擅长捕捉任务特定的知识，从而在多任务场景中提高性能。代码可以在这个链接中获得：[提供链接的地方]。 

---
# MVIP -- A Dataset and Methods for Application Oriented Multi-View and Multi-Modal Industrial Part Recognition 

**Title (ZH)**: MVIP ——一种面向应用的多视图和多模态工业零件识别数据集及方法 

**Authors**: Paul Koch, Marian Schlüter, Jörg Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2502.15448)  

**Abstract**: We present MVIP, a novel dataset for multi-modal and multi-view application-oriented industrial part recognition. Here we are the first to combine a calibrated RGBD multi-view dataset with additional object context such as physical properties, natural language, and super-classes. The current portfolio of available datasets offers a wide range of representations to design and benchmark related methods. In contrast to existing classification challenges, industrial recognition applications offer controlled multi-modal environments but at the same time have different problems than traditional 2D/3D classification challenges. Frequently, industrial applications must deal with a small amount or increased number of training data, visually similar parts, and varying object sizes, while requiring a robust near 100% top 5 accuracy under cost and time constraints. Current methods tackle such challenges individually, but direct adoption of these methods within industrial applications is complex and requires further research. Our main goal with MVIP is to study and push transferability of various state-of-the-art methods within related downstream tasks towards an efficient deployment of industrial classifiers. Additionally, we intend to push with MVIP research regarding several modality fusion topics, (automated) synthetic data generation, and complex data sampling -- combined in a single application-oriented benchmark. 

**Abstract (ZH)**: 我们提出了MVIP，这是一个新颖的数据集，专为多模态和多视图应用导向的工业部件识别设计。在这里，我们首次将校准的RGBD多视角数据集与额外的对象上下文（如物理属性、自然语言和超级类别）结合在一起。目前可用的数据集提供了广泛的表示方式，用于设计和基准测试相关方法。与现有的分类挑战不同，工业识别应用提供的是可控的多模态环境，但同时与传统2D/3D分类挑战相比也存在不同的问题。通常，工业应用必须处理少量或增加的训练数据、外观相似的部件以及变化的对象尺寸，同时在成本和时间限制下要求稳健的近100% top-5准确性。当前的方法分别应对这些挑战，但在工业应用中直接采用这些方法是复杂的，并需要进一步研究。我们的主要目标是通过MVIP研究和推动各种最新技术在相关下游任务中的可迁移性，以实现工业分类器的高效部署。此外，我们希望通过MVIP促进多模态融合的研究，包括（自动）合成数据生成和复杂数据采样——这些都结合在一个应用导向的基准测试中。 

---
# When Compression Meets Model Compression: Memory-Efficient Double Compression for Large Language Models 

**Title (ZH)**: 当压缩遇到模型压缩：面向大型语言模型的高效双压缩方法 

**Authors**: Weilan Wang, Yu Mao, Dongdong Tang, Hongchao Du, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.15443)  

**Abstract**: Large language models (LLMs) exhibit excellent performance in various tasks. However, the memory requirements of LLMs present a great challenge when deploying on memory-limited devices, even for quantized LLMs. This paper introduces a framework to compress LLM after quantization further, achieving about 2.2x compression ratio. A compression-aware quantization is first proposed to enhance model weight compressibility by re-scaling the model parameters before quantization, followed by a pruning method to improve further. Upon this, we notice that decompression can be a bottleneck during practical scenarios. We then give a detailed analysis of the trade-off between memory usage and latency brought by the proposed method. A speed-adaptive method is proposed to overcome it. The experimental results show inference with the compressed model can achieve a 40% reduction in memory size with negligible loss in accuracy and inference speed. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色。然而，LLMs 的内存需求在部署到内存受限设备上时，即使是量化后的LLMs，也带来了巨大的挑战。本文介绍了一种框架，在量化后进一步压缩LLMs，实现了约2.2倍的压缩比。首先提出了感知压缩的量化方法，通过在量化前重新缩放模型参数来增强模型权重的压缩性，随后使用剪枝方法进一步改进。在此基础上，我们注意到解压缩在实际场景中可能会成为瓶颈。我们随后对所提出方法带来的内存使用与延迟之间的权衡进行了详细分析，并提出了一种适应速度的方法来克服这一问题。实验结果表明，使用压缩模型进行推理可以在不显著损失准确性和推理速度的情况下，将内存大小减少40%。 

---
# Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning 

**Title (ZH)**: Fed-SB：在（私有）联邦LoRA微调中实现极端通信效率和高性能的银弹 

**Authors**: Raghav Singhal, Kaustubh Ponkshe, Rohit Vartak, Lav R. Varshney, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2502.15436)  

**Abstract**: Low-Rank Adaptation (LoRA) has become ubiquitous for efficiently fine-tuning foundation models. However, federated fine-tuning using LoRA is challenging due to suboptimal updates arising from traditional federated averaging of individual adapters. Existing solutions either incur prohibitively high communication cost that scales linearly with the number of clients or suffer from performance degradation due to limited expressivity. We introduce Federated Silver Bullet (Fed-SB), a novel approach for federated fine-tuning of LLMs using LoRA-SB, a recently proposed low-rank adaptation method. LoRA-SB optimally aligns the optimization trajectory with the ideal low-rank full fine-tuning projection by learning a small square matrix (R) between adapters B and A, keeping other components fixed. Direct averaging of R guarantees exact updates, substantially reducing communication cost, which remains independent of the number of clients, and enables scalability. Fed-SB achieves state-of-the-art performance across commonsense reasoning, arithmetic reasoning, and language inference tasks while reducing communication costs by up to 230x. In private settings, Fed-SB further improves performance by (1) reducing trainable parameters, thereby lowering the noise required for differential privacy and (2) avoiding noise amplification introduced by other methods. Overall, Fed-SB establishes a new Pareto frontier in the tradeoff between communication and performance, offering an efficient and scalable solution for both private and non-private federated fine-tuning. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 低秩适应（LoRA）已经成为高效微调基础模型的普遍方法。然而，使用LoRA进行联邦微调具有挑战性，因为传统的方法通过客户端个体适配器的联邦平均会产生次优更新。现有的解决方案要么会带来随着客户端数量线性增长的高通信成本，要么会由于表达能力有限而导致性能下降。我们提出了一种名为Federated Silver Bullet（Fed-SB）的新方法，用于使用LoRA-SB（最近提出的低秩适应方法）进行LLM的联邦微调。LoRA-SB通过学习适配器B和A之间的小方阵（R），优化了优化轨迹与理想的低秩全面微调投影之间的对齐，同时保持其他组件不变。直接对R进行平均可以确保精确更新，从而显著减少通信成本，该成本与客户端数量无关，可以实现可扩展性。Fed-SB在常识推理、算术推理和语言推理任务上均实现了最先进的性能，同时将通信成本降低至最高230倍。在私有环境中，Fed-SB进一步提高了性能，通过（1）减少可训练参数数量，从而降低差分隐私所需的噪声，并（2）避免其他方法引入的噪声放大。总体而言，Fed-SB在通信与性能之间的权衡上引入了一个新的帕累托前沿，提供了一种适用于私有和非私有联邦微调的有效且可扩展的解决方案。我们的代码已在此处[http链接]公开。 

---
# Single-pass Detection of Jailbreaking Input in Large Language Models 

**Title (ZH)**: 大型语言模型中单次检测 Jailbreaking 输入的方法 

**Authors**: Leyla Naz Candogan, Yongtao Wu, Elias Abad Rocamora, Grigorios G. Chrysos, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2502.15435)  

**Abstract**: Defending aligned Large Language Models (LLMs) against jailbreaking attacks is a challenging problem, with existing approaches requiring multiple requests or even queries to auxiliary LLMs, making them computationally heavy. Instead, we focus on detecting jailbreaking input in a single forward pass. Our method, called Single Pass Detection SPD, leverages the information carried by the logits to predict whether the output sentence will be harmful. This allows us to defend in just one forward pass. SPD can not only detect attacks effectively on open-source models, but also minimizes the misclassification of harmless inputs. Furthermore, we show that SPD remains effective even without complete logit access in GPT-3.5 and GPT-4. We believe that our proposed method offers a promising approach to efficiently safeguard LLMs against adversarial attacks. 

**Abstract (ZH)**: 防护对齐的大语言模型（LLMs）免受 Jailbreaking 攻击是一个具有挑战性的问题，现有方法往往需要多次请求甚至查询辅助 LLM，使其计算量巨大。相比之下，我们集中于在一次前向传递中检测 Jailbreaking 输入。我们提出的方法称为一次前向检测（Single Pass Detection, SPD），该方法利用 logits 中携带的信息来预测输出句子是否具有危害性。这使得我们可以在一次前向传递中实现防护。SPD 不仅可以有效地检测开源模型中的攻击，还能尽量减少对无害输入的误分类。此外，我们展示即使在 GPT-3.5 和 GPT-4 中不完全访问 logits，SPD 仍然有效。我们认为，我们提出的方法为有效防范 LLMs 面对对抗性攻击提供了一个有前景的方法。 

---
# Anatomy-Informed Deep Learning and Radiomics for Automated Neurofibroma Segmentation in Whole-Body MRI 

**Title (ZH)**: 基于解剖信息的深度学习和放射组学在全身MRI中自动神经鞘瘤分割中的应用 

**Authors**: Georgii Kolokolnikov, Marie-Lena Schmalhofer, Lennart Well, Said Farschtschi, Victor-Felix Mautner, Inka Ristow, Rene Werner  

**Link**: [PDF](https://arxiv.org/pdf/2502.15424)  

**Abstract**: Neurofibromatosis Type 1 is a genetic disorder characterized by the development of neurofibromas (NFs), which exhibit significant variability in size, morphology, and anatomical location. Accurate and automated segmentation of these tumors in whole-body magnetic resonance imaging (WB-MRI) is crucial to assess tumor burden and monitor disease progression. In this study, we present and analyze a fully automated pipeline for NF segmentation in fat-suppressed T2-weighted WB-MRI, consisting of three stages: anatomy segmentation, NF segmentation, and tumor candidate classification. In the first stage, we use the MRSegmentator model to generate an anatomy segmentation mask, extended with a high-risk zone for NFs. This mask is concatenated with the input image as anatomical context information for NF segmentation. The second stage employs an ensemble of 3D anisotropic anatomy-informed U-Nets to produce an NF segmentation confidence mask. In the final stage, tumor candidates are extracted from the confidence mask and classified based on radiomic features, distinguishing tumors from non-tumor regions and reducing false positives. We evaluate the proposed pipeline on three test sets representing different conditions: in-domain data (test set 1), varying imaging protocols and field strength (test set 2), and low tumor burden cases (test set 3). Experimental results show a 68% improvement in per-scan Dice Similarity Coefficient (DSC), a 21% increase in per-tumor DSC, and a two-fold improvement in F1 score for tumor detection in high tumor burden cases by integrating anatomy information. The method is integrated into the 3D Slicer platform for practical clinical use, with the code publicly accessible. 

**Abstract (ZH)**: 神经纤维瘤病1型是一种由神经纤维瘤（NF）的发展引起的遗传性疾病，这些肿瘤在大小、形态和解剖位置上表现出显著的变异。准确且自动化的全身磁共振成像（WB-MRI）中NF的分割对于评估肿瘤负担和监控疾病进展至关重要。在本研究中，我们提出并分析了一个完全自动化的神经纤维瘤分割管道，该管道由三个阶段组成：解剖分割、神经纤维瘤分割和肿瘤候选分类。在第一阶段，我们使用MRSegmentator模型生成一个解剖分割掩模，并扩展一个高风险区域用于神经纤维瘤。该掩模与输入图像连接，作为神经纤维瘤分割的解剖上下文信息。第二阶段采用3D各向异性解剖导向U-Net集成，生成一个神经纤维瘤分割置信度掩模。在最终阶段，从置信度掩模中提取肿瘤候选并基于影像组学特征进行分类，将肿瘤区域与非肿瘤区域区分开来，从而减少假阳性结果。我们使用三个包含不同条件的测试集评估了提出的管道：训练集（测试集1）、不同成像协议和磁场强度（测试集2）以及低肿瘤负担病例（测试集3）。实验结果表明，通过整合解剖信息，肿瘤负担高的病例检测的平均每扫描Dice相似性系数（DSC）提高了68%，平均每肿瘤DSC提高了21%，且肿瘤检测的F1分数提高了两倍。该方法已集成到3D Slicer平台中，以便于临床实际应用，并开放了源代码。 

---
# Evaluating Multimodal Generative AI with Korean Educational Standards 

**Title (ZH)**: 使用韩国教育标准评估多模态生成型人工智能 

**Authors**: Sanghee Park, Geewook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.15422)  

**Abstract**: This paper presents the Korean National Educational Test Benchmark (KoNET), a new benchmark designed to evaluate Multimodal Generative AI Systems using Korean national educational tests. KoNET comprises four exams: the Korean Elementary General Educational Development Test (KoEGED), Middle (KoMGED), High (KoHGED), and College Scholastic Ability Test (KoCSAT). These exams are renowned for their rigorous standards and diverse questions, facilitating a comprehensive analysis of AI performance across different educational levels. By focusing on Korean, KoNET provides insights into model performance in less-explored languages. We assess a range of models - open-source, open-access, and closed APIs - by examining difficulties, subject diversity, and human error rates. The code and dataset builder will be made fully open-sourced at this https URL. 

**Abstract (ZH)**: 本文介绍了韩国国家教育测试基准（KoNET），这是一种新的基准测试，旨在使用韩国国家教育测试评估多模态生成式AI系统。KoNET 包含四场考试：韩国小学普遍教育发展测试（KoEGED）、韩国中学普遍教育发展测试（KoMGED）、韩国高中普遍教育发展测试（KoHGED）和韩国大学修业能力测试（KoCSAT）。这些考试以严格的标准和多样的问题著称，使得从不同教育层次全面分析AI性能成为可能。通过专注于韩语，KoNET 提供了对较未探索语言中模型性能的见解。我们通过考察难度、科目多样性和人工错误率来评估多种模型——包括开源、开放访问和封闭API模型。代码和数据集构建器将在以下链接处完全开源：[此处填写链接]。 

---
# Beyond Translation: LLM-Based Data Generation for Multilingual Fact-Checking 

**Title (ZH)**: 超越翻译：基于LLM的数据生成在多语言事实核查中的应用 

**Authors**: Yi-Ling Chung, Aurora Cobo, Pablo Serna  

**Link**: [PDF](https://arxiv.org/pdf/2502.15419)  

**Abstract**: Robust automatic fact-checking systems have the potential to combat online misinformation at scale. However, most existing research primarily focuses on English. In this paper, we introduce MultiSynFact, the first large-scale multilingual fact-checking dataset containing 2.2M claim-source pairs designed to support Spanish, German, English, and other low-resource languages. Our dataset generation pipeline leverages Large Language Models (LLMs), integrating external knowledge from Wikipedia and incorporating rigorous claim validation steps to ensure data quality. We evaluate the effectiveness of MultiSynFact across multiple models and experimental settings. Additionally, we open-source a user-friendly framework to facilitate further research in multilingual fact-checking and dataset generation. 

**Abstract (ZH)**: 稳健的自动事实核查系统有望大规模打击网络虚假信息。然而，现有的大部分研究主要集中在英语上。本文介绍了MultiSynFact，这是一个多语言事实核查数据集，包含220万条声明-来源配对，旨在支持西班牙语、德语、英语以及其他低资源语言。我们的数据集生成管道利用了大规模语言模型（LLMs），并整合了来自维基百科的外部知识，同时引入了严格的声明验证步骤以确保数据质量。我们评估了MultiSynFact在多个模型和实验设置下的有效性。此外，我们还开源了一个用户友好的框架，以促进多语言事实核查和数据集生成的进一步研究。 

---
# HiFi-KPI: A Dataset for Hierarchical KPI Extraction from Earnings Filings 

**Title (ZH)**: HiFi-KPI： earnings披露中层级关键绩效指标提取的数据集 

**Authors**: Rasmus Aavang, Giovanni Rizzi, Rasmus Bøggild, Alexandre Iolov, Mike Zhang, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2502.15411)  

**Abstract**: The U.S. Securities and Exchange Commission (SEC) requires that public companies file financial reports tagging numbers with the machine readable inline eXtensible Business Reporting Language (iXBRL) standard. However, the highly complex and highly granular taxonomy defined by iXBRL limits label transferability across domains. In this paper, we introduce the Hierarchical Financial Key Performance Indicator (HiFi-KPI) dataset, designed to facilitate numerical KPI extraction at specified levels of granularity from unstructured financial text. Our approach organizes a 218,126-label hierarchy using a taxonomy based grouping method, investigating which taxonomy layer provides the most meaningful structure. HiFi-KPI comprises ~1.8M paragraphs and ~5M entities, each linked to a label in the iXBRL-specific calculation and presentation taxonomies. We provide baselines using encoder-based approaches and structured extraction using Large Language Models (LLMs). To simplify LLM inference and evaluation, we additionally release HiFi-KPI Lite, a manually curated subset with four expert-mapped labels. We publicly release all artifacts 

**Abstract (ZH)**: 美国证券交易委员会（SEC）要求上市公司提交财务报告，并使用机器可读的即用型商务报告语言（iXBRL）标准标记数字。然而，iXBRL定义的复杂且高度粒度化的分类法限制了标签在不同领域之间的转移性。本文介绍了层次财务关键绩效指标（HiFi-KPI）数据集，旨在从非结构化的财务文本中在指定的粒度级别提取关键绩效指标（KPI）。我们的方法采用基于分类法的分组方法，组织了218,126个标签的层级结构，并研究了哪个分类层提供了最有意义的结构。HiFi-KPI包含约180万段落和约500万个实体，每个实体都与iXBRL特定的计算与展示分类法中的标签相连。我们提供了基于编码器的方法和使用大型语言模型（LLMs）进行结构化提取的基准。为了简化LLM的推断和评估，我们还发布了HiFi-KPI Lite，这是一个经过人工精选的子集，包含四个专家映射的标签。我们公开发布了所有相关成果。 

---
# Enhancing Vehicle Make and Model Recognition with 3D Attention Modules 

**Title (ZH)**: 使用3D注意力模块增强车辆品牌和型号识别 

**Authors**: Narges Semiromizadeh, Omid Nejati Manzari, Shahriar B. Shokouhi, Sattar Mirzakuchaki  

**Link**: [PDF](https://arxiv.org/pdf/2502.15398)  

**Abstract**: Vehicle make and model recognition (VMMR) is a crucial component of the Intelligent Transport System, garnering significant attention in recent years. VMMR has been widely utilized for detecting suspicious vehicles, monitoring urban traffic, and autonomous driving systems. The complexity of VMMR arises from the subtle visual distinctions among vehicle models and the wide variety of classes produced by manufacturers. Convolutional Neural Networks (CNNs), a prominent type of deep learning model, have been extensively employed in various computer vision tasks, including VMMR, yielding remarkable results. As VMMR is a fine-grained classification problem, it primarily faces inter-class similarity and intra-class variation challenges. In this study, we implement an attention module to address these challenges and enhance the model's focus on critical areas containing distinguishing features. This module, which does not increase the parameters of the original model, generates three-dimensional (3-D) attention weights to refine the feature map. Our proposed model integrates the attention module into two different locations within the middle section of a convolutional model, where the feature maps from these sections offer sufficient information about the input frames without being overly detailed or overly coarse. The performance of our proposed model, along with state-of-the-art (SOTA) convolutional and transformer-based models, was evaluated using the Stanford Cars dataset. Our proposed model achieved the highest accuracy, 90.69\%, among the compared models. 

**Abstract (ZH)**: 车辆品牌和型号识别（VMMR）是智能交通系统的关键组成部分，近年来引起了广泛关注。VMMR 广泛应用于检测可疑车辆、监测城市交通以及自动驾驶系统。VMMR 的复杂性源于不同车型之间的细微视觉差异以及制造商生产的多种类别。卷积神经网络（CNNs）作为深度学习模型的一种重要类型，在各种计算机视觉任务中得到了广泛应用，包括 VMMR，取得了显著的成果。由于 VMMR 是一种细粒度分类问题，它主要面临类间相似性和类内变异性的挑战。在本文中，我们实现了一个注意力模块来解决这些挑战，并增强模型对包含区分特征的关键区域的关注。该模块不增加原有模型的参数，生成三维（3D）注意力权重以细化特征图。我们提议的模型将在卷积模型中间部分的两个不同位置整合该注意力模块，这些部分的特征图提供了足够的输入帧信息，但不包含过多的细节或过于粗糙的信息。我们使用斯坦福汽车数据集对所提议的模型及其与其他最先进的（SOTA）卷积和基于变换器的模型的性能进行了评估。所提议的模型在比较模型中达到了最高的准确率，为 90.69%。 

---
# Super-Resolution for Interferometric Imaging: Model Comparisons and Performance Analysis 

**Title (ZH)**: 超分辨率在干涉成像中的应用：模型比较与性能分析 

**Authors**: Hasan Berkay Abdioglu, Rana Gursoy, Yagmur Isik, Ibrahim Cem Balci, Taha Unal, Kerem Bayer, Mustafa Ismail Inal, Nehir Serin, Muhammed Furkan Kosar, Gokhan Bora Esmer, Huseyin Uvet  

**Link**: [PDF](https://arxiv.org/pdf/2502.15397)  

**Abstract**: This study investigates the application of Super-Resolution techniques in holographic microscopy to enhance quantitative phase imaging. An off-axis Mach-Zehnder interferometric setup was employed to capture interferograms. The study evaluates two Super-Resolution models, RCAN and Real-ESRGAN, for their effectiveness in reconstructing high-resolution interferograms from a microparticle-based dataset. The models were assessed using two primary approaches: image-based analysis for structural detail enhancement and morphological evaluation for maintaining sample integrity and phase map accuracy. The results demonstrate that RCAN achieves superior numerical precision, making it ideal for applications requiring highly accurate phase map reconstruction, while Real-ESRGAN enhances visual quality and structural coherence, making it suitable for visualization-focused applications. This study highlights the potential of Super-Resolution models in overcoming diffraction-imposed resolution limitations in holographic microscopy, opening the way for improved imaging techniques in biomedical diagnostics, materials science, and other high-precision fields. 

**Abstract (ZH)**: 本研究探讨了超分辨技术在全息显微镜中的应用，以提升定量相位成像的质量。采用偏轴马赫-曾德尔干涉仪配置来捕获干涉图。研究评估了两种超分辨模型——RCAN和Real-ESRGAN，以考察其从基于微颗粒的数据集中重新构建高分辨率干涉图的效果。模型评估采用了两种主要方法：基于图像的分析以增强结构细节，以及形态学评估以保持样本完整性和相位图的准确性。研究结果表明，RCAN在数值精度方面表现更优，使其成为需要高精度相位图重建应用的理想选择，而Real-ESRGAN则在视觉质量和结构一致性上表现出色，适用于侧重于成像的应用。本研究强调了超分辨模型在克服全息显微镜中衍射限制分辨率的能力，为生物医学诊断、材料科学及其他高精度领域提供改进的成像技术开辟了途径。 

---
# Identifying Features that Shape Perceived Consciousness in Large Language Model-based AI: A Quantitative Study of Human Responses 

**Title (ZH)**: 基于大规模语言模型的AI中塑造感知意识的特征识别：人类反应的量化研究 

**Authors**: Kang Bongsu, Kim Jundong, Yun Tae-Rim, Bae Hyojin, Kim Chang-Eop  

**Link**: [PDF](https://arxiv.org/pdf/2502.15365)  

**Abstract**: This study quantitively examines which features of AI-generated text lead humans to perceive subjective consciousness in large language model (LLM)-based AI systems. Drawing on 99 passages from conversations with Claude 3 Opus and focusing on eight features -- metacognitive self-reflection, logical reasoning, empathy, emotionality, knowledge, fluency, unexpectedness, and subjective expressiveness -- we conducted a survey with 123 participants. Using regression and clustering analyses, we investigated how these features influence participants' perceptions of AI consciousness. The results reveal that metacognitive self-reflection and the AI's expression of its own emotions significantly increased perceived consciousness, while a heavy emphasis on knowledge reduced it. Participants clustered into seven subgroups, each showing distinct feature-weighting patterns. Additionally, higher prior knowledge of LLMs and more frequent usage of LLM-based chatbots were associated with greater overall likelihood assessments of AI consciousness. This study underscores the multidimensional and individualized nature of perceived AI consciousness and provides a foundation for better understanding the psychosocial implications of human-AI interaction. 

**Abstract (ZH)**: 本研究定量地考察了哪些特征使得人类在基于大规模语言模型（LLM）的AI系统中感知到主观意识。我们基于与Claude 3 Opus对话中的99段文字，重点研究了八种特征——元认知自我反思、逻辑推理、同理心、情感性、知识、流畅性、意外性和主观表达性——并对123名参与者进行了问卷调查。通过回归分析和聚类分析，我们探讨了这些特征如何影响参与者对AI意识的感知。研究结果表明，元认知自我反思和AI对自己的情感表达显著增强了感知到的意识，而过度强调知识则减少了这种感知。参与者被分为七个子群，每个子群都表现出不同的特征权重模式。此外，对LLM的更高前置知识和更频繁使用基于LLM的聊天机器人都与更高的总体意识可能性评估相关。本研究强调了感知到的AI意识的多维度和个体化特性，并为更好地理解人机交互的心理社会影响奠定了基础。 

---
# Evaluating Social Biases in LLM Reasoning 

**Title (ZH)**: 评估大语言模型推理中的社会偏见 

**Authors**: Xuyang Wu, Jinming Nian, Zhiqiang Tao, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15361)  

**Abstract**: In the recent development of AI reasoning, large language models (LLMs) are trained to automatically generate chain-of-thought reasoning steps, which have demonstrated compelling performance on math and coding tasks. However, when bias is mixed within the reasoning process to form strong logical arguments, it could cause even more harmful results and further induce hallucinations. In this paper, we have evaluated the 8B and 32B variants of DeepSeek-R1 against their instruction tuned counterparts on the BBQ dataset, and investigated the bias that is elicited out and being amplified through reasoning steps. To the best of our knowledge, this empirical study is the first to assess bias issues in LLM reasoning. 

**Abstract (ZH)**: 在近年来AI推理的发展中，大型语言模型（LLMs）被训练以自动生成推理步骤，这些模型在数学和编程任务上展现了令人信服的性能。然而，当偏见混入推理过程并形成强大的逻辑论证时，这可能会导致更严重的后果，并进一步引发幻觉。在本文中，我们评估了DeepSeek-R1的8亿和32亿参数变体与其指令调优版本在BBQ数据集上的表现，并调查了通过推理步骤外泄和放大的偏见问题。据我们所知，这是首次对大型语言模型推理中的偏见问题进行实证研究。 

---
# Integrating Generative AI in Cybersecurity Education: Case Study Insights on Pedagogical Strategies, Critical Thinking, and Responsible AI Use 

**Title (ZH)**: 将下面的论文内容或标题翻译成中文，同时确保符合学术规范：

"在网络安全教育中整合生成式人工智能：教学策略、批判性思维和负责任的人工智能使用案例研究 Insights"

更正式的翻译版本可以是：

"在网络安全教育中集成生成式人工智能：教学策略、批判性思维与负责任的人工智能使用案例研究"

这样翻译既保留了原文的学术严谨性，又符合中文的表达习惯。 

**Authors**: Mahmoud Elkhodr, Ergun Gide  

**Link**: [PDF](https://arxiv.org/pdf/2502.15357)  

**Abstract**: The rapid advancement of Generative Artificial Intelligence (GenAI) has introduced new opportunities for transforming higher education, particularly in fields that require analytical reasoning and regulatory compliance, such as cybersecurity management. This study presents a structured framework for integrating GenAI tools into cybersecurity education, demonstrating their role in fostering critical thinking, real-world problem-solving, and regulatory awareness. The implementation strategy followed a two-stage approach, embedding GenAI within tutorial exercises and assessment tasks. Tutorials enabled students to generate, critique, and refine AI-assisted cybersecurity policies, while assessments required them to apply AI-generated outputs to real-world scenarios, ensuring alignment with industry standards and regulatory requirements. Findings indicate that AI-assisted learning significantly enhanced students' ability to evaluate security policies, refine risk assessments, and bridge theoretical knowledge with practical application. Student reflections and instructor observations revealed improvements in analytical engagement, yet challenges emerged regarding AI over-reliance, variability in AI literacy, and the contextual limitations of AI-generated content. Through structured intervention and research-driven refinement, students were able to recognize AI strengths as a generative tool while acknowledging its need for human oversight. This study further highlights the broader implications of AI adoption in cybersecurity education, emphasizing the necessity of balancing automation with expert judgment to cultivate industry-ready professionals. Future research should explore the long-term impact of AI-driven learning on cybersecurity competency, as well as the potential for adaptive AI-assisted assessments to further personalize and enhance educational outcomes. 

**Abstract (ZH)**: 生成式人工智能（GenAI）的快速发展为高等教育带来了新的机遇，特别是在需要分析推理和合规管理的领域，如网络安全管理。本文提出了一种结构化的框架，用于将GenAI工具整合到网络安全教育中，展示了其在培养批判性思维、实际问题解决能力和合规意识方面的作用。实施策略采用了两阶段的方法，将GenAI嵌入辅导练习和评估任务中。辅导课程让学生能够生成、批判和改进AI辅助的网络安全政策，而评估则要求学生将AI生成的输出应用于实际情境，确保与行业标准和合规要求的对齐。研究发现，AI辅助学习显著增强了学生评估安全政策、细化风险评估以及将理论知识与实践应用相结合的能力。学生反馈和教师观察表明，虽然在分析参与度方面有所提高，但同时也存在过度依赖AI、AI素养差异性和AI生成内容的上下文局限性等问题。通过结构化的干预和基于研究的改进，学生能够认识到AI作为生成工具的优点及其需要人类监督的需求。本文进一步强调了AI在网络安全教育中的更广泛影响，指出在促进自动化的同时平衡专家判断的必要性，以培养出符合行业要求的专业人才。未来研究应探讨AI驱动学习对网络安全技能的长期影响，并探索适应性AI辅助评估的潜力，以进一步个性化和提升教育成果。 

---
# Constructing a Norm for Children's Scientific Drawing: Distribution Features Based on Semantic Similarity of Large Language Models 

**Title (ZH)**: 基于大型语言模型语义相似性的儿童科学绘画规范构建：分布特征 

**Authors**: Yi Zhang, Fan Wei, Jingyi Li, Yan Wang, Yanyan Yu, Jianli Chen, Zipo Cai, Xinyu Liu, Wei Wang, Peng Wang, Zhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15348)  

**Abstract**: The use of children's drawings to examining their conceptual understanding has been proven to be an effective method, but there are two major problems with previous research: 1. The content of the drawings heavily relies on the task, and the ecological validity of the conclusions is low; 2. The interpretation of drawings relies too much on the subjective feelings of the researchers. To address this issue, this study uses the Large Language Model (LLM) to identify 1420 children's scientific drawings (covering 9 scientific themes/concepts), and uses the word2vec algorithm to calculate their semantic similarity. The study explores whether there are consistent drawing representations for children on the same theme, and attempts to establish a norm for children's scientific drawings, providing a baseline reference for follow-up children's drawing research. The results show that the representation of most drawings has consistency, manifested as most semantic similarity greater than 0.8. At the same time, it was found that the consistency of the representation is independent of the accuracy (of LLM's recognition), indicating the existence of consistency bias. In the subsequent exploration of influencing factors, we used Kendall rank correlation coefficient to investigate the effects of Sample Size, Abstract Degree, and Focus Points on drawings, and used word frequency statistics to explore whether children represented abstract themes/concepts by reproducing what was taught in class. 

**Abstract (ZH)**: 利用儿童绘画来考察其概念理解已被证明是一种有效的方法，但以往研究存在两个主要问题：1. 绘画内容高度依赖于具体任务，导致结论的生态效度较低；2. 对绘画的解释过于依赖研究者的主观感受。为解决这些问题，本研究利用大型语言模型（LLM）识别了1420份涉及9个科学主题/概念的儿童科学绘画，并使用word2vec算法计算其语义相似度。研究探索了同一主题下儿童绘画是否具有一致的表征形式，并尝试建立儿童科学绘画的规范，为后续儿童绘画研究提供基线参考。研究结果显示，大多数绘图的表示具有一致性，表现为大多数语义相似度大于0.8。同时发现，表示的一致性与LLM识别的准确性无关，表明存在一致性偏见。在后续影响因素探索中，我们使用肯德尔秩相关系数研究了样本大小、抽象程度和关注点对绘画的影响，并利用词汇频率统计来探讨儿童是否通过复现课堂上教授的内容来表示抽象的主题/概念。 

---
# Exploring Embodied Multimodal Large Models: Development, Datasets, and Future Directions 

**Title (ZH)**: 探究具身多模态大型模型：发展、数据集及未来方向 

**Authors**: Shoubin Chen, Zehao Wu, Kai Zhang, Chunyu Li, Baiyang Zhang, Fei Ma, Fei Richard Yu, Qingquan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15336)  

**Abstract**: Embodied multimodal large models (EMLMs) have gained significant attention in recent years due to their potential to bridge the gap between perception, cognition, and action in complex, real-world environments. This comprehensive review explores the development of such models, including Large Language Models (LLMs), Large Vision Models (LVMs), and other models, while also examining other emerging architectures. We discuss the evolution of EMLMs, with a focus on embodied perception, navigation, interaction, and simulation. Furthermore, the review provides a detailed analysis of the datasets used for training and evaluating these models, highlighting the importance of diverse, high-quality data for effective learning. The paper also identifies key challenges faced by EMLMs, including issues of scalability, generalization, and real-time decision-making. Finally, we outline future directions, emphasizing the integration of multimodal sensing, reasoning, and action to advance the development of increasingly autonomous systems. By providing an in-depth analysis of state-of-the-art methods and identifying critical gaps, this paper aims to inspire future advancements in EMLMs and their applications across diverse domains. 

**Abstract (ZH)**: 近年来，嵌入式多模态大型模型（EMLMs）因其潜力而在感知、认知和行动之间架起桥梁，引起了广泛关注。本综述全面探讨了这类模型的发展，包括大型语言模型（LLMs）、大型视觉模型（LVMs）及其他模型，并考察了其他新兴架构。我们讨论了EMLMs的发展演变，重点关注嵌入式感知、导航、交互和模拟。此外，本文还详细分析了用于训练和评估这些模型的数据集，强调了数据多样性与高质量对于有效学习的重要性。本文还指出了EMLMs面临的几项关键挑战，包括可扩展性、泛化能力和实时决策问题。最后，我们概述了未来发展方向，强调了将多模态感知、推理和行动结合的重要性，以推进更加自主系统的开发。通过深入分析最新技术方法并识别关键缺口，本文旨在激发未来在EMLMs及其跨多种领域的应用方面的进步。 

---
# Attention Eclipse: Manipulating Attention to Bypass LLM Safety-Alignment 

**Title (ZH)**: 注意力遮蔽：通过操控注意力机制绕过大模型安全性对齐 

**Authors**: Pedram Zaree, Md Abdullah Al Mamun, Quazi Mishkatul Alam, Yue Dong, Ihsen Alouani, Nael Abu-Ghazaleh  

**Link**: [PDF](https://arxiv.org/pdf/2502.15334)  

**Abstract**: Recent research has shown that carefully crafted jailbreak inputs can induce large language models to produce harmful outputs, despite safety measures such as alignment. It is important to anticipate the range of potential Jailbreak attacks to guide effective defenses and accurate assessment of model safety. In this paper, we present a new approach for generating highly effective Jailbreak attacks that manipulate the attention of the model to selectively strengthen or weaken attention among different parts of the prompt. By harnessing attention loss, we develop more effective jailbreak attacks, that are also transferrable. The attacks amplify the success rate of existing Jailbreak algorithms including GCG, AutoDAN, and ReNeLLM, while lowering their generation cost (for example, the amplified GCG attack achieves 91.2% ASR, vs. 67.9% for the original attack on Llama2-7B/AdvBench, using less than a third of the generation time). 

**Abstract (ZH)**: 近期的研究表明，精心设计的越狱输入可以使对齐后的大语言模型产生有害输出。尽管采取了诸如对齐等安全措施，仍需警惕各种潜在的越狱攻击，以指导有效的防御措施并准确评估模型的安全性。本文介绍了生成高效越狱攻击的新方法，这种方法通过操纵模型的注意力，选择性地增强或减弱对提示不同部分的关注。通过利用注意力损失，我们发展了更具效果并且具有可迁移性的越狱攻击。这些攻击在现有越狱算法（包括GCG、AutoDAN和ReNeLLM）的基础上提高了成功率，同时降低了生成成本（例如，增强后的GCG攻击在Llama2-7B/AdvBench上的成功率达到了91.2%，而原始攻击的成功率为67.9%，且仅使用了原始攻击所需时间的三分之一）。 

---
# Lightweight yet Efficient: An External Attentive Graph Convolutional Network with Positional Prompts for Sequential Recommendation 

**Title (ZH)**: 轻量而高效的外部注意力图卷积网络结合位置提示用于序列推荐 

**Authors**: Jinyu Zhang, Chao Li, Zhongying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.15331)  

**Abstract**: Graph-based Sequential Recommender systems (GSRs) have gained significant research attention due to their ability to simultaneously handle user-item interactions and sequential relationships between items. Current GSRs often utilize composite or in-depth structures for graph encoding (e.g., the Graph Transformer). Nevertheless, they have high computational complexity, hindering the deployment on resource-constrained edge devices. Moreover, the relative position encoding in Graph Transformer has difficulty in considering the complicated positional dependencies within sequence. To this end, we propose an External Attentive Graph convolutional network with Positional prompts for Sequential recommendation, namely EA-GPS. Specifically, we first introduce an external attentive graph convolutional network that linearly measures the global associations among nodes via two external memory units. Then, we present a positional prompt-based decoder that explicitly treats the absolute item positions as external prompts. By introducing length-adaptive sequential masking and a soft attention network, such a decoder facilitates the model to capture the long-term positional dependencies and contextual relationships within sequences. Extensive experimental results on five real-world datasets demonstrate that the proposed EA-GPS outperforms the state-of-the-art methods. Remarkably, it achieves the superior performance while maintaining a smaller parameter size and lower training overhead. The implementation of this work is publicly available at this https URL. 

**Abstract (ZH)**: 基于图的序列推荐系统（GSRs）由于能够同时处理用户-项目交互和项目之间的序列关系，而获得了广泛的研究关注。当前的GSRs通常使用复合或深入的图编码结构（例如，图变换器）。然而，它们具有较高的计算复杂性，这阻碍了它们在资源受限的边缘设备上的部署。此外，图变换器中的相对位置编码难以考虑序列中的复杂位置依赖性。针对这些问题，我们提出了一种基于外部注意的图卷积网络与位置提示的序列推荐方法，即EA-GPS。具体而言，我们首先引入了一种外部注意图卷积网络，通过两个外部记忆单元线性度量节点之间的全局关联。然后，我们提出了一种基于位置提示的解码器，该解码器将绝对项目位置明确地视为外部提示。通过引入长度自适应序列掩码和软注意力网络，这种解码器有助于模型捕捉序列中的长期位置依赖性和语境关系。在五个真实世界数据集上的实验结果表明，提出的EA-GPS在多种基准方法中表现最佳。值得注意的是，它在保持较小参数量和较低训练开销的情况下实现了优异的性能。该工作的实现代码已在以下链接公开：this https URL。 

---
# SentiFormer: Metadata Enhanced Transformer for Image Sentiment Analysis 

**Title (ZH)**: SentiFormer：增强元数据的图像情感分析变换器 

**Authors**: Bin Feng, Shulan Ruan, Mingzheng Yang, Dongxuan Han, Huijie Liu, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15322)  

**Abstract**: As more and more internet users post images online to express their daily emotions, image sentiment analysis has attracted increasing attention. Recently, researchers generally tend to design different neural networks to extract visual features from images for sentiment analysis. Despite the significant progress, metadata, the data (e.g., text descriptions and keyword tags) for describing the image, has not been sufficiently explored in this task. In this paper, we propose a novel Metadata Enhanced Transformer for sentiment analysis (SentiFormer) to fuse multiple metadata and the corresponding image into a unified framework. Specifically, we first obtain multiple metadata of the image and unify the representations of diverse data. To adaptively learn the appropriate weights for each metadata, we then design an adaptive relevance learning module to highlight more effective information while suppressing weaker ones. Moreover, we further develop a cross-modal fusion module to fuse the adaptively learned representations and make the final prediction. Extensive experiments on three publicly available datasets demonstrate the superiority and rationality of our proposed method. 

**Abstract (ZH)**: 随着越来越多的互联网用户通过上传图片来表达日常情绪，图像情感分析受到了越来越多的关注。近年来，研究人员普遍倾向于设计不同的神经网络从图像中提取视觉特征进行情感分析。尽管在这一领域取得了显著的进步，但描述图像的元数据（例如，文本描述和关键词标签）尚未得到充分探索。本文提出了一种新型的元数据增强Transformer（SentiFormer）以将多种元数据与相应的图像融合到统一框架中。具体而言，我们首先获取图像的多种元数据，并统一多种数据的表示。然后，我们设计了一个自适应相关性学习模块，以适应性地为每种元数据学习合适的权重，从而突出更有效的信息并抑制较弱的信息。此外，我们进一步开发了一个跨模态融合模块，将适应性学习的表示融合起来并做出最终预测。在三个公开可用的数据集上的广泛实验验证了我们提出的方法的优势和合理性。 

---
# Road Traffic Sign Recognition method using Siamese network Combining Efficient-CNN based Encoder 

**Title (ZH)**: 使用基于Efficient-CNN编码器的Siamese网络进行道路交通标志识别的方法 

**Authors**: Zhenghao Xi, Yuchao Shao, Yang Zheng, Xiang Liu, Yaqi Liu, Yitong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2502.15307)  

**Abstract**: Traffic signs recognition (TSR) plays an essential role in assistant driving and intelligent transportation system. However, the noise of complex environment may lead to motion-blur or occlusion problems, which raise the tough challenge to real-time recognition with high accuracy and robust. In this article, we propose IECES-network which with improved encoders and Siamese net. The three-stage approach of our method includes Efficient-CNN based encoders, Siamese backbone and the fully-connected layers. We firstly use convolutional encoders to extract and encode the traffic sign features of augmented training samples and standard images. Then, we design the Siamese neural network with Efficient-CNN based encoder and contrastive loss function, which can be trained to improve the robustness of TSR problem when facing the samples of motion-blur and occlusion by computing the distance between inputs and templates. Additionally, the template branch of the proposed network can be stopped when executing the recognition tasks after training to raise the process speed of our real-time model, and alleviate the computational resource and parameter scale. Finally, we recombined the feature code and a fully-connected layer with SoftMax function to classify the codes of samples and recognize the category of traffic signs. The results of experiments on the Tsinghua-Tencent 100K dataset and the German Traffic Sign Recognition Benchmark dataset demonstrate the performance of the proposed IECESnetwork. Compared with other state-of-the-art methods, in the case of motion-blur and occluded environment, the proposed method achieves competitive performance precision-recall and accuracy metric average is 88.1%, 86.43% and 86.1% with a 2.9M lightweight scale, respectively. Moreover, processing time of our model is 0.1s per frame, of which the speed is increased by 1.5 times compared with existing methods. 

**Abstract (ZH)**: 交通标志识别（TSR）在辅助驾驶和智能交通系统中起着重要作用。然而，复杂环境的噪声可能会导致运动模糊或遮挡问题，这使得在保证高精度和鲁棒性的前提下实现实时识别变得极具挑战性。本文提出了一种改进的编码器和Siamese网络结合的IECES网络，来应对这些挑战。我们的方法采用了三阶段的处理流程，包括基于Efficient-CNN的编码器、Siamese主干和全连接层。首先，利用卷积编码器提取和编码增强训练样本和标准图像中的交通标志特征。接着，设计了一个基于Efficient-CNN的Siamese神经网络，并采用对比损失函数来进行训练，以提高在运动模糊和遮挡情况下的鲁棒性。此外，在训练完成后，模板分支可以在执行识别任务时停止工作，从而加快我们实时模型的处理速度，减少计算资源和参数规模。最后，通过将特征码与带SoftMax函数的全连接层重新组合，来对样本的代码进行分类和识别交通标志的类别。在Tsinghua-Tencent 100K数据集和German Traffic Sign Recognition Benchmark数据集上的实验结果显示，所提出的IECES网络表现出良好的性能。相较于其他现有方法，在运动模糊和遮挡环境中，该方法在精确召回率和准确率平均值分别为88.1%、86.43%和86.1%的情况下，模型的参数规模仅为2.9M。此外，我们的模型每帧处理时间为0.1秒，相比于现有方法，处理速度提高了1.5倍。 

---
# SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention 

**Title (ZH)**: SVDq: 1.25比特和410倍键缓存压缩的大型语言模型注意力机制 

**Authors**: Hong Yankun, Li Xing, Zhen Hui-Ling, Yu Xianzhi, Liu Wulong, Yuan Mingxuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15304)  

**Abstract**: For the efficient inference of Large Language Models (LLMs), the effective compression of key-value (KV) cache is essential. Three main types of KV cache compression techniques, namely sparsity, channel compression, and quantization, have been identified. This study presents SVDq, a Singular Value Decomposition (SVD) - based mixed precision quantization method for K cache. Initially, K cache is transformed into latent channels using SVD basis representations. Since the values in latent channels decay rapidly and become negligible after only a few latent channels, our method then incorporates importance-aware quantization and compression for latent channels. This enables the effective allocation of higher precision to more significant channels. Theoretically, we prove that SVDq results in quantization errors (x0.1 or even lower) that are much lower than those of per-channel key quantization in the original space. Our findings based on RULER and LongBench benchmarks demonstrate that SVDq can achieve an equivalent key cache precision as low as 1.25-bit. When combined with key sparsity, it can reach a key compression ratio of up to 410x for attention computation, all while maintaining comparable model performance. Notably, our method is nearly lossless for LongBench datasets. This indicates that SVDq enables high-precision low-bit quantization, providing a more efficient solution for KV cache compression in LLMs. 

**Abstract (ZH)**: 为了高效推断大规模语言模型（LLMs），有效地压缩键-值（KV）缓存至关重要。三种主要的KV缓存压缩技术，即稀疏性、信道压缩和量化，已经被识别出来。本研究提出了一种基于奇异值分解（SVD）的混合精度量化方法SVDq，应用于键-值缓存（K cache）。首先，通过SVD基本表示将K缓存转换为潜在信道。由于潜在信道中的值迅速衰减并在少数几个潜在信道后变得微不足道，我们的方法随后结合了对潜在信道的重要性意识量化和压缩，从而能够更高效地分配较高的精度给更重要的信道。理论上，我们证明SVDq的量化误差（约为0.1或更低）远低于原始空间中按信道量化键的误差。基于RULER和LongBench基准测试的实验结果表明，SVDq可以实现等效的1.25位键缓存精度。当与键稀疏性结合使用时，它可以将注意力计算的键压缩比率提高到高达410倍，同时保持相当的模型性能。值得注意的是，对于LongBench数据集，我们的方法几乎是无损的。这表明SVDq能够实现高精度低位量化，提供LLMs中KV缓存压缩的一种更高效解决方案。 

---
# Beyond Fixed Variables: Expanding-variate Time Series Forecasting via Flat Scheme and Spatio-temporal Focal Learning 

**Title (ZH)**: 超越固定变量：通过平坦方案和时空聚焦学习扩展变量时间序列预测 

**Authors**: Minbo Ma, Kai Tang, Huan Li, Fei Teng, Dalin Zhang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15296)  

**Abstract**: Multivariate Time Series Forecasting (MTSF) has long been a key research focus. Traditionally, these studies assume a fixed number of variables, but in real-world applications, Cyber-Physical Systems often expand as new sensors are deployed, increasing variables in MTSF. In light of this, we introduce a novel task, Expanding-variate Time Series Forecasting (EVTSF). This task presents unique challenges, specifically (1) handling inconsistent data shapes caused by adding new variables, and (2) addressing imbalanced spatio-temporal learning, where expanding variables have limited observed data due to the necessity for timely operation. To address these challenges, we propose STEV, a flexible spatio-temporal forecasting framework. STEV includes a new Flat Scheme to tackle the inconsistent data shape issue, which extends the graph-based spatio-temporal modeling architecture into 1D space by flattening the 2D samples along the variable dimension, making the model variable-scale-agnostic while still preserving dynamic spatial correlations through a holistic graph. We introduce a novel Spatio-temporal Focal Learning strategy that incorporates a negative filter to resolve potential conflicts between contrastive learning and graph representation, and a focal contrastive loss as its core to guide the framework to focus on optimizing the expanding variables. We benchmark EVTSF performance using three real-world datasets and compare it against three potential solutions employing SOTA MTSF models tailored for EVSTF. Experimental results show that STEV significantly outperforms its competitors, particularly on expanding variables. Notably, STEV, with only 5% of observations from the expanding period, is on par with SOTA MTSF models trained with complete observations. Further exploration of various expanding strategies underscores the generalizability of STEV in real-world applications. 

**Abstract (ZH)**: 多变量时间序列预测（MTSF）长期以来一直是研究重点之一。传统研究通常假设固定数量的变量，但在实际应用中，随着新传感器的部署，网络物理系统往往会扩展，增加MTSF中的变量。鉴于这一问题，我们引入了一个新的任务，即扩展变量时间序列预测（EVTSF）。这一任务带来了独特的挑战，具体来说包括（1）处理由于增加新变量而产生的数据形状不一致问题，以及（2）解决时空学习的不平衡问题，其中新增变量由于需要及时操作而观察数据有限。为了解决这些挑战，我们提出了STEV，这是一种灵活的时空预测框架。STEV 包括一种新的平面方案，用以解决数据形状不一致问题，该方案通过沿变量维度将2D样本展平，将基于图的时空建模架构扩展到1D空间，使模型与变量规模无关，同时仍保留通过整体图传递的动态空间相关性。我们引入了一种新的时空焦点学习策略，该策略结合了一个负滤波器以解决对比学习与图表示之间的潜在冲突，并以焦点对比损失为核心来引导框架重点关注优化扩展变量。我们使用三个现实世界的数据集对EVTSF性能进行了基准测试，并将其与针对EVTSF进行调整的最新MTSF模型的三种潜在解决方案进行了比较。实验结果表明，STEV 显著优于其竞争对手，尤其是在扩展变量上表现出色。值得注意的是，STEV仅使用扩展期间5%的观测数据，其性能与使用完整观测数据训练的最新MTSF模型相当。进一步探索各种扩展策略证明了STEV在实际应用中的普适性。 

---
# Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inference 

**Title (ZH)**: 圆级注意力机制：一种加速大规模语言模型推理的新颖圆级注意力机制 

**Authors**: Yaohua Tang, Zhicheng Hu, Kun Cheng, Fan Mo, Qiheng Lv, Hua Wang, Zhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15294)  

**Abstract**: The increasing context window size in large language models (LLMs) has improved their ability to handle complex, long-text tasks. However, as the conversation rounds continue, it is required to store a large amount of KV cache in GPU memory, which significantly affects the efficiency and even availability of the model serving systems. This paper analyzes dialogue data from real users and discovers that the LLM inference manifests a watershed layer, after which the distribution of round-level attention shows notable similarity. We propose Round Attention, a novel round-level attention mechanism that only recalls and computes the KV cache of the most relevant rounds. The experiments show that our method saves 55\% memory usage without compromising model performance. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）上下文窗口大小的不断扩大，其处理复杂长文本任务的能力得到了提升。然而，随着对话轮次的增加，需要在GPU内存中存储大量的KV缓存，这显著影响了模型服务系统的效率，甚至对其可用性产生负面影响。本文分析了真实用户的数据，并发现LLM推理呈现出一个临界点，在此之后，各轮次的注意力分布显示出显著的相似性。我们提出了一种新颖的轮次层级注意力机制——Round Attention，该机制仅回忆和计算与当前最相关的轮次的KV缓存。实验结果表明，我们的方法在不牺牲模型性能的情况下，内存使用率降低了55%。 

---
# Time Warp: The Gap Between Developers' Ideal vs Actual Workweeks in an AI-Driven Era 

**Title (ZH)**: 时间扭曲：开发者理想中的工作周与实际工作周在人工智能驱动时代之间的差距 

**Authors**: Sukrit Kumar, Drishti Goel, Thomas Zimmermann, Brian Houck, B. Ashok, Chetan Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.15287)  

**Abstract**: Software developers balance a variety of different tasks in a workweek, yet the allocation of time often differs from what they consider ideal. Identifying and addressing these deviations is crucial for organizations aiming to enhance the productivity and well-being of the developers. In this paper, we present the findings from a survey of 484 software developers at Microsoft, which aims to identify the key differences between how developers would like to allocate their time during an ideal workweek versus their actual workweek. Our analysis reveals significant deviations between a developer's ideal workweek and their actual workweek, with a clear correlation: as the gap between these two workweeks widens, we observe a decline in both productivity and satisfaction. By examining these deviations in specific activities, we assess their direct impact on the developers' satisfaction and productivity. Additionally, given the growing adoption of AI tools in software engineering, both in the industry and academia, we identify specific tasks and areas that could be strong candidates for automation. In this paper, we make three key contributions: 1) We quantify the impact of workweek deviations on developer productivity and satisfaction 2) We identify individual tasks that disproportionately affect satisfaction and productivity 3) We provide actual data-driven insights to guide future AI automation efforts in software engineering, aligning them with the developers' requirements and ideal workflows for maximizing their productivity and satisfaction. 

**Abstract (ZH)**: 软件开发者在一个工作周内需要平衡多种不同的任务，但他们实际分配的时间往往与理想状态有所差异。组织若想提升开发者的生产力和福祉，就需要识别并解决这些差异。本文通过调查微软484名软件开发者的问卷结果，旨在识别理想工作周和实际工作周之间时间分配的主要差异。我们的分析揭示了开发者理想工作周与实际工作周之间存在显著差异，并且差距越大，开发者的工作效率和满意度下降趋势越明显。通过分析这些具体活动之间的差异，我们评估它们对开发者满意度和生产力的直接影响。此外，鉴于人工智能工具在软件工程领域的日益普及，无论是行业还是学术界，我们还识别出了可以进行自动化的主要任务和领域。本文主要贡献如下：1）量化工作周差异对开发者生产力和满意度的影响；2）识别对满意度和生产力有显著影响的单独任务；3）提供基于实际数据的洞察，指导未来的软件工程中的人工智能自动化工作，使其与开发者的实际需求和理想工作流程相契合，以最大化他们的生产力和满意度。 

---
# Offload Rethinking by Cloud Assistance for Efficient Environmental Sound Recognition on LPWANs 

**Title (ZH)**: 基于云辅助的卸载重思考以提高LPWAN中高效环境声音识别效率 

**Authors**: Le Zhang, Quanling Zhao, Run Wang, Shirley Bian, Onat Gungor, Flavio Ponzina, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2502.15285)  

**Abstract**: Learning-based environmental sound recognition has emerged as a crucial method for ultra-low-power environmental monitoring in biological research and city-scale sensing systems. These systems usually operate under limited resources and are often powered by harvested energy in remote areas. Recent efforts in on-device sound recognition suffer from low accuracy due to resource constraints, whereas cloud offloading strategies are hindered by high communication costs. In this work, we introduce ORCA, a novel resource-efficient cloud-assisted environmental sound recognition system on batteryless devices operating over the Low-Power Wide-Area Networks (LPWANs), targeting wide-area audio sensing applications. We propose a cloud assistance strategy that remedies the low accuracy of on-device inference while minimizing the communication costs for cloud offloading. By leveraging a self-attention-based cloud sub-spectral feature selection method to facilitate efficient on-device inference, ORCA resolves three key challenges for resource-constrained cloud offloading over LPWANs: 1) high communication costs and low data rates, 2) dynamic wireless channel conditions, and 3) unreliable offloading. We implement ORCA on an energy-harvesting batteryless microcontroller and evaluate it in a real world urban sound testbed. Our results show that ORCA outperforms state-of-the-art methods by up to $80 \times$ in energy savings and $220 \times$ in latency reduction while maintaining comparable accuracy. 

**Abstract (ZH)**: 基于学习的环境声音识别方法已经成为了生物研究和城市规模感知系统中超低功耗环境监测的关键手段。这些系统通常在资源受限的条件下运行，并且经常在偏远地区利用收集的能量进行供电。最近对设备上的声音识别努力由于资源限制而面临准确率较低的问题，而云卸载策略则受到高昂通信成本的限制。在这项工作中，我们引入了ORCA，这是一种针对低电力广域网络（LPWAN）上无电池设备的新型资源高效云辅助环境声音识别系统，旨在解决大范围音频感知应用中的问题。我们提出了一种云辅助策略，该策略可以改善设备上推理的低准确率，同时最小化云卸载的通信成本。通过利用基于自注意力的云子光谱特征选择方法来促进高效的设备上推理，ORCA解决了资源受限的LPWAN云卸载中的三个关键挑战：1) 高通信成本和低数据速率，2) 动态无线信道条件，以及3) 不可靠的卸载。我们在一种利用收集能量的无电池微控制器上实施了ORCA，并在实际的都市声音测试平台上进行评估。我们的结果显示，与现有最先进的方法相比，ORCA在能耗上提高了高达80倍，在延迟减少方面提高了220倍，并且保持了相当的准确率。 

---
# CopyJudge: Automated Copyright Infringement Identification and Mitigation in Text-to-Image Diffusion Models 

**Title (ZH)**: CopyJudge：文本到图像扩散模型中的自动版权侵权识别与缓解 

**Authors**: Shunchang Liu, Zhuan Shi, Lingjuan Lyu, Yaochu Jin, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2502.15278)  

**Abstract**: Assessing whether AI-generated images are substantially similar to copyrighted works is a crucial step in resolving copyright disputes. In this paper, we propose CopyJudge, an automated copyright infringement identification framework that leverages large vision-language models (LVLMs) to simulate practical court processes for determining substantial similarity between copyrighted images and those generated by text-to-image diffusion models. Specifically, we employ an abstraction-filtration-comparison test framework with multi-LVLM debate to assess the likelihood of infringement and provide detailed judgment rationales. Based on the judgments, we further introduce a general LVLM-based mitigation strategy that automatically optimizes infringing prompts by avoiding sensitive expressions while preserving the non-infringing content. Besides, our approach can be enhanced by exploring non-infringing noise vectors within the diffusion latent space via reinforcement learning, even without modifying the original prompts. Experimental results show that our identification method achieves comparable state-of-the-art performance, while offering superior generalization and interpretability across various forms of infringement, and that our mitigation method could more effectively mitigate memorization and IP infringement without losing non-infringing expressions. 

**Abstract (ZH)**: 评估AI生成的图像是否与受版权保护的作品在实质相似度上有显著差异是解决版权纠纷关键步骤之一。本文提出了一种名为CopyJudge的自动化版权侵权识别框架，该框架利用大规模视觉-语言模型（LVLM）来模拟实际法庭流程，以确定受版权保护的图像与基于文本到图像扩散模型生成的图像之间的实质相似度。具体来说，我们采用了一种抽象-过滤-比较的测试框架，并结合多LVLM辩论，评估侵权的可能性，并提供详细的裁决理由。基于这些裁决，我们进一步引入了一种基于LVLM的一般性缓解策略，该策略可以自动优化侵权提示，避免敏感表达方式，同时保留非侵权内容。此外，通过探索扩散潜在空间中的非侵权噪声向量，我们的方法可以通过强化学习进一步增强，即使在不修改原始提示的情况下也能实现这一目标。实验结果显示，我们的识别方法在性能上可以达到现有最先进的水平，且具有更好的泛化能力和解释性，而且我们的缓解方法能够更有效地缓解记忆问题和知识产权侵权风险，而不丢失非侵权信息。 

---
# Corrections Meet Explanations: A Unified Framework for Explainable Grammatical Error Correction 

**Title (ZH)**: 纠正与解释的统一框架：可解释的语法错误纠正方法 

**Authors**: Jingheng Ye, Shang Qin, Yinghui Li, Hai-Tao Zheng, Shen Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15261)  

**Abstract**: Grammatical Error Correction (GEC) faces a critical challenge concerning explainability, notably when GEC systems are designed for language learners. Existing research predominantly focuses on explaining grammatical errors extracted in advance, thus neglecting the relationship between explanations and corrections. To address this gap, we introduce EXGEC, a unified explainable GEC framework that integrates explanation and correction tasks in a generative manner, advocating that these tasks mutually reinforce each other. Experiments have been conducted on EXPECT, a recent human-labeled dataset for explainable GEC, comprising around 20k samples. Moreover, we detect significant noise within EXPECT, potentially compromising model training and evaluation. Therefore, we introduce an alternative dataset named EXPECT-denoised, ensuring a more objective framework for training and evaluation. Results on various NLP models (BART, T5, and Llama3) show that EXGEC models surpass single-task baselines in both tasks, demonstrating the effectiveness of our approach. 

**Abstract (ZH)**: 句法错误修正（GEC）在可解释性方面面临着一个关键挑战，尤其是在为语言学习者设计的GEC系统中。现有的研究主要集中在解释预先提取的语法错误上，忽视了解释与修正之间的关系。为了解决这一问题，我们引入了EXGEC，这是一种统一的可解释GEC框架，通过生成的方式将解释和修正任务相结合，提倡这些任务相互增强。我们在最近的人工标注数据集EXPECT上进行了实验，该数据集包含约2万样本。此外，我们发现EXPECT中存在显著的噪声，可能会影响模型的训练和评估。因此，我们引入了一个名为EXPECT-denoised的替代数据集，确保训练和评估时有更客观的框架。在多种NLP模型（包括BART、T5和Llama3）上的实验结果显示，EXGEC模型在两个任务上都超过了单一任务基准模型，证明了我们方法的有效性。 

---
# ComposeOn Academy: Transforming Melodic Ideas into Complete Compositions Integrating Music Learning 

**Title (ZH)**: 学术规范的中文翻译如下：

学院曲谱：将旋律构思转化为完整乐曲的音乐创作方法整合音乐学习 

**Authors**: Hongxi Pu, Futian Jiang, Zihao Chen, Xingyue Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.15255)  

**Abstract**: Music composition has long been recognized as a significant art form. However, existing digital audio workstations and music production software often present high entry barriers for users lacking formal musical training. To address this, we introduce ComposeOn, a music theory-based tool designed for users with limited musical knowledge. ComposeOn enables users to easily extend their melodic ideas into complete compositions and offers simple editing features. By integrating music theory, it explains music creation at beginner, intermediate, and advanced levels. Our user study (N=10) compared ComposeOn with the baseline method, Suno AI, demonstrating that ComposeOn provides a more accessible and enjoyable composing and learning experience for individuals with limited musical skills. ComposeOn bridges the gap between theory and practice, offering an innovative solution as both a composition aid and music education platform. The study also explores the differences between theory-based music creation and generative music, highlighting the former's advantages in personal expression and learning. 

**Abstract (ZH)**: 音乐创作长期以来被认为是一种重要的艺术形式。然而，现有的数字音频工作站和音乐制作软件往往对缺乏正式音乐训练的用户设置了较高的门槛。为了解决这一问题，我们引入了ComposeOn，这是一种基于音乐理论的工具，旨在为音乐知识有限的用户提供便利。ComposeOn使得用户能够轻松地将其旋律概念扩展为完整的作品，并提供简单的编辑功能。通过整合音乐理论，它在初学者、中级和高级水平上解释了音乐创作的过程。我们的用户研究（N=10）将ComposeOn与基准方法（Suno AI）进行了比较，结果显示，对于音乐技能有限的个人而言，ComposeOn提供了更易于使用和更愉悦的创作与学习体验。ComposeOn填补了理论与实践之间的差距，作为一种创作辅助工具和音乐教育平台，提供了创新的解决方案。研究还探讨了基于理论的音乐创作与生成音乐之间的差异，突出了前者的个人表达和学习优势。 

---
# Comparative Analysis of Large Language Models for Context-Aware Code Completion using SAFIM Framework 

**Title (ZH)**: 基于SAFIM框架的大规模语言模型在上下文感知代码补全中的比较分析 

**Authors**: Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du, Yiyi Tao, Yixian Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15243)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized code completion, transforming it into a more intelligent and context-aware feature in modern integrated development environments. These advancements have significantly enhanced developers' ability to write efficient and error-free code. This study evaluates the performance of several chat-based LLMs, including Gemini 1.5 Flash, Gemini 1.5 Pro, GPT-4o, GPT-4o-mini, and GPT-4 Turbo, using the Syntax-Aware Fill-in-the-Middle (SAFIM) dataset. This benchmark is specifically designed to assess models' capabilities in syntax-sensitive code generation. Performance metrics, such as cosine similarity with ground-truth completions and latency, were employed to measure both accuracy and efficiency. The findings reveal substantial differences in the models' code completion abilities, offering valuable insights into their respective strengths and weaknesses. This work provides a comparative analysis that underscores the trade-offs between accuracy and speed, establishing a benchmark for future advancements in LLM-based code completion. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现彻底革新了代码补全，使其在现代集成开发环境中变得更加智能且具有上下文感知能力。这些进步显著提升了开发人员编写高效且无错误代码的能力。本研究使用Syntax-Aware Fill-in-the-Middle（SAFIM）数据集评估了几种基于聊天的LLM的表现，包括Gemini 1.5 Flash、Gemini 1.5 Pro、GPT-4o、GPT-4o-mini和GPT-4 Turbo。SAFIM基准测试专门设计用于评估模型在语法敏感代码生成方面的能力。采用了余弦相似度与真实完成结果以及延迟时间等性能指标来衡量准确性和效率。研究结果揭示了这些模型在代码补全能力上的显著差异，为各自的强项和弱点提供了宝贵见解。本研究提供了对比分析，强调了准确性和速度之间的权衡，并建立了基于LLM的代码补全的基准，为未来的发展奠定了基础。 

---
# AutoMR: A Universal Time Series Motion Recognition Pipeline 

**Title (ZH)**: AutoMR：一种通用的时间序列运动识别流水线 

**Authors**: Likun Zhang, Sicheng Yang, Zhuo Wang, Haining Liang, Junxiao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15228)  

**Abstract**: In this paper, we present an end-to-end automated motion recognition (AutoMR) pipeline designed for multimodal datasets. The proposed framework seamlessly integrates data preprocessing, model training, hyperparameter tuning, and evaluation, enabling robust performance across diverse scenarios. Our approach addresses two primary challenges: 1) variability in sensor data formats and parameters across datasets, which traditionally requires task-specific machine learning implementations, and 2) the complexity and time consumption of hyperparameter tuning for optimal model performance. Our library features an all-in-one solution incorporating QuartzNet as the core model, automated hyperparameter tuning, and comprehensive metrics tracking. Extensive experiments demonstrate its effectiveness on 10 diverse datasets, achieving state-of-the-art performance. This work lays a solid foundation for deploying motion-capture solutions across varied real-world applications. 

**Abstract (ZH)**: 在本文中，我们提出了一种针对多模态数据集设计的端到端自动化动作识别（AutoMR）管道。所提出的方法无缝地整合了数据预处理、模型训练、超参数调整和评估，能够实现跨不同应用场景的稳健性能。我们的方法解决了两个主要挑战：1）不同数据集之间传感器数据格式和参数的多样性，传统上需要针对具体任务的机器学习实现；2）超参数调优的复杂性和耗时性，以实现最佳模型性能。我们库的核心包括将QuartzNet作为核心模型，自动超参数调整以及全面的性能指标跟踪。广泛的实验表明，在10个不同数据集上其有效性和先进性。本文为在各种实际应用中部署动作捕捉解决方案奠定了坚实的基础。 

---
# Understand User Opinions of Large Language Models via LLM-Powered In-the-Moment User Experience Interviews 

**Title (ZH)**: 通过大型语言模型支持的即时用户体验访谈理解用户对大型语言模型的意见 

**Authors**: Mengqiao Liu, Tevin Wang, Cassandra A. Cohen, Sarah Li, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.15226)  

**Abstract**: Which large language model (LLM) is better? Every evaluation tells a story, but what do users really think about current LLMs? This paper presents CLUE, an LLM-powered interviewer that conducts in-the-moment user experience interviews, right after users interacted with LLMs, and automatically gathers insights about user opinions from massive interview logs. We conduct a study with thousands of users to understand user opinions on mainstream LLMs, recruiting users to first chat with a target LLM and then interviewed by CLUE. Our experiments demonstrate that CLUE captures interesting user opinions, for example, the bipolar views on the displayed reasoning process of DeepSeek-R1 and demands for information freshness and multi-modality. Our collected chat-and-interview logs will be released. 

**Abstract (ZH)**: 哪种大型语言模型（LLM）更好？每项评估都有其故事，但用户对当前LLM的真实看法是什么？本文介绍了CLUE，这是一种由LLM驱动的访谈工具，它在用户与LLM交互后立即进行实时用户体验访谈，并自动收集来自大量访谈日志的用户意见见解。我们进行了一项涉及数千用户的实验，以了解用户对主流LLM的意见，邀请用户首先与目标LLM进行对话，然后接受CLUE的访谈。我们的实验表明，CLUE捕捉到了一些有趣的用户意见，例如，用户对DeepSeek-R1展示的推理过程的两极看法以及对信息新鲜度和多模态性的需求。我们收集的对话和访谈日志将予以公开。 

---
# Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs 

**Title (ZH)**: Auto-Bench：一种用于LLM科学发现的自动化基准测试 

**Authors**: Tingting Chen, Srinivas Anumasa, Beibei Lin, Vedant Shah, Anirudh Goyal, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15224)  

**Abstract**: Given the remarkable performance of Large Language Models (LLMs), an important question arises: Can LLMs conduct human-like scientific research and discover new knowledge, and act as an AI scientist? Scientific discovery is an iterative process that demands efficient knowledge updating and encoding. It involves understanding the environment, identifying new hypotheses, and reasoning about actions; however, no standardized benchmark specifically designed for scientific discovery exists for LLM agents. In response to these limitations, we introduce a novel benchmark, \textit{Auto-Bench}, that encompasses necessary aspects to evaluate LLMs for scientific discovery in both natural and social sciences. Our benchmark is based on the principles of causal graph discovery. It challenges models to uncover hidden structures and make optimal decisions, which includes generating valid justifications. By engaging interactively with an oracle, the models iteratively refine their understanding of underlying interactions, the chemistry and social interactions, through strategic interventions. We evaluate state-of-the-art LLMs, including GPT-4, Gemini, Qwen, Claude, and Llama, and observe a significant performance drop as the problem complexity increases, which suggests an important gap between machine and human intelligence that future development of LLMs need to take into consideration. 

**Abstract (ZH)**: 鉴于大型语言模型（LLMs）表现出色，一个重要的问题随之产生：LLMs能否进行人类类似的研究，并发现新的知识，从而成为AI科学家？科学发现是一个迭代过程，需要高效的知识更新和编码。它包括理解环境、提出新的假设以及推理有关的行动等多个环节；然而，针对科学发现的标准化基准尚不存在，用于评估LLM代理的表现。为了应对这些限制，我们引入了一个新型基准——Auto-Bench，该基准涵盖了评估LLMs在自然科学和社会科学中进行科学发现所需的所有要素。该基准基于因果图发现的原则，旨在挑战模型揭示隐藏结构并作出最优决策，其中包括生成有效的论据。通过与oracle进行交互，模型能够通过战略性干预逐步深化对底层交互、化学和社交互动的理解。我们评估了最先进的LLM模型，包括GPT-4、Gemini、Qwen、Claude和Llama，并观察到随着问题复杂性的增加，模型性能显著下降，这表明机器与人类智能之间存在重要差距，未来LLM的发展需要考虑到这一点。 

---
# FormalSpecCpp: A Dataset of C++ Formal Specifications created using LLMs 

**Title (ZH)**: FormalSpecCpp：一个使用LLM创建的C++形式化规范数据集 

**Authors**: Madhurima Chakraborty, Peter Pirkelbauer, Qing Yi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15217)  

**Abstract**: FormalSpecCpp is a dataset designed to fill the gap in standardized benchmarks for verifying formal specifications in C++ programs. To the best of our knowledge, this is the first comprehensive collection of C++ programs with well-defined preconditions and postconditions. It provides a structured benchmark for evaluating specification inference tools and testing theaccuracy of generated specifications. Researchers and developers can use this dataset to benchmark specification inference tools,fine-tune Large Language Models (LLMs) for automated specification generation, and analyze the role of formal specifications in improving program verification and automated testing. By making this dataset publicly available, we aim to advance research in program verification, specification inference, and AI-assisted software development. The dataset and the code are available at this https URL. 

**Abstract (ZH)**: FormalSpecCpp是一个数据集，旨在填补用于验证C++程序形式化规范标准化基准的空白。据我们所知，这是第一个包含明确预条件和后条件的C++程序综合集合。它提供了一个结构化的基准，用于评估规范推断工具并测试生成规范的准确性。研究人员和开发人员可以使用此数据集来基准测试规范推断工具、微调大规模语言模型（LLMs）以实现自动化规范生成，并分析形式化规范在提高程序验证和自动化测试中的作用。通过使此数据集公开可用，我们旨在推动程序验证、规范推断和AI辅助软件开发的研究。数据集和代码可以在以下网址获取：[此处替换为具体的网址链接]。 

---
# The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning 

**Title (ZH)**: LLM-和VLM-集成强化学习的发展 landscape 

**Authors**: Sheila Schoepp, Masoud Jafaripour, Yingyue Cao, Tianpei Yang, Fatemeh Abdollahi, Shadan Golestan, Zahin Sufiyan, Osmar R. Zaiane, Matthew E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.15214)  

**Abstract**: Reinforcement learning (RL) has shown impressive results in sequential decision-making tasks. Meanwhile, Large Language Models (LLMs) and Vision-Language Models (VLMs) have emerged, exhibiting impressive capabilities in multimodal understanding and reasoning. These advances have led to a surge of research integrating LLMs and VLMs into RL. In this survey, we review representative works in which LLMs and VLMs are used to overcome key challenges in RL, such as lack of prior knowledge, long-horizon planning, and reward design. We present a taxonomy that categorizes these LLM/VLM-assisted RL approaches into three roles: agent, planner, and reward. We conclude by exploring open problems, including grounding, bias mitigation, improved representations, and action advice. By consolidating existing research and identifying future directions, this survey establishes a framework for integrating LLMs and VLMs into RL, advancing approaches that unify natural language and visual understanding with sequential decision-making. 

**Abstract (ZH)**: 强化学习（RL）在序贯决策任务中展现了令人印象深刻的成果。与此同时，大型语言模型（LLMs）和多模态视觉语言模型（VLMs）已经涌现，并展示了在多模态理解和推理方面的 impressive 能力。这些进展催生了将 LLMs 和 VLMs 结合进 RL 的大量研究。在本文综述中，我们回顾了 LLMs 和 VLMs 在克服 RL 中关键挑战时的应用，如缺乏先验知识、长期规划以及奖励设计。我们提出了一种分类法，将这些由 LLMs/VLMs 支撑的 RL 方法归类为三个角色：代理、规划器和奖励。最后，我们探讨了开放性问题，包括语义关联、偏见 mitigation、更好的表示形式以及行动建议。通过总结现有研究并确定未来的研究方向，本文综述建立了一种框架，用于将 LLMs 和 VLMs 整合进 RL，推动了自然语言和视觉理解与序贯决策统一的方法的发展。 

---
# PairBench: A Systematic Framework for Selecting Reliable Judge VLMs 

**Title (ZH)**: PairBench：一种选择可靠法官语言模型的系统性框架 

**Authors**: Aarash Feizi, Sai Rajeswar, Adriana Romero-Soriano, Reihaneh Rabbany, Spandana Gella, Valentina Zantedeschi, João Monteiro  

**Link**: [PDF](https://arxiv.org/pdf/2502.15210)  

**Abstract**: As large vision language models (VLMs) are increasingly used as automated evaluators, understanding their ability to effectively compare data pairs as instructed in the prompt becomes essential. To address this, we present PairBench, a low-cost framework that systematically evaluates VLMs as customizable similarity tools across various modalities and scenarios. Through PairBench, we introduce four metrics that represent key desiderata of similarity scores: alignment with human annotations, consistency for data pairs irrespective of their order, smoothness of similarity distributions, and controllability through prompting. Our analysis demonstrates that no model, whether closed- or open-source, is superior on all metrics; the optimal choice depends on an auto evaluator's desired behavior (e.g., a smooth vs. a sharp judge), highlighting risks of widespread adoption of VLMs as evaluators without thorough assessment. For instance, the majority of VLMs struggle with maintaining symmetric similarity scores regardless of order. Additionally, our results show that the performance of VLMs on the metrics in PairBench closely correlates with popular benchmarks, showcasing its predictive power in ranking models. 

**Abstract (ZH)**: 随着大型视觉语言模型（VLMs）越来越多地被用作自动评估工具，理解和评估它们在指示中有效比较数据对的能力变得至关重要。为此，我们提出PairBench，这是一种低成本框架，系统地评估VLMs在各种模态和场景下作为可定制相似性工具的性能。通过PairBench，我们引入了四种度量标准，这些标准代表了相似性评分的关键要求：与人类注释的契合度，对于数据对的一致性（不考虑其顺序），相似性分布的平滑性，以及通过提示进行控制的能力。我们的分析表明，并没有哪种模型（闭源或开源）在所有度量标准上都表现出色；最优选择取决于自动评估器的期望行为（例如，平滑的评估者与尖锐的评估者），强调了在没有充分评估之前广泛采用VLMs作为评估工具所存在的风险。例如，大多数VLMs在保持相似性评分的对称性（与顺序无关）方面存在困难。此外，我们的结果表明，PairBench中度量标准上VLMs的表现与流行基准密切相关，展示了其在模型排名上的预测能力。 

---
# FlipConcept: Tuning-Free Multi-Concept Personalization for Text-to-Image Generation 

**Title (ZH)**: FlipConcept: 无需调优的多概念个性化生成文本到图像 

**Authors**: Young Beom Woo, Sun Eung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.15203)  

**Abstract**: Recently, methods that integrate multiple personalized concepts into a single image have garnered significant attention in the field of text-to-image (T2I) generation. However, existing methods experience performance degradation in complex scenes with multiple objects due to distortions in non-personalized regions. To address this issue, we propose FlipConcept, a novel approach that seamlessly integrates multiple personalized concepts into a single image without requiring additional tuning. We introduce guided appearance attention to accurately mimic the appearance of a personalized concept as intended. Additionally, we introduce mask-guided noise mixing to protect non-personalized regions during editing. Lastly, we apply background dilution to minimize attribute leakage, which is the undesired blending of personalized concept attributes with other objects in the image. In our experiments, we demonstrate that the proposed method, despite not requiring tuning, outperforms existing models in both single and multiple personalized concept inference. 

**Abstract (ZH)**: 近年来，将多个个性化概念整合到单张图像中的方法在文本到图像（T2I）生成领域引起了广泛关注。然而，现有方法在包含多个对象的复杂场景中由于非个性化区域的失真而导致性能下降。为了解决这个问题，我们提出了一种名为FlipConcept的新颖方法，该方法能够在不需要额外调优的情况下无缝整合多个个性化概念。我们引入了引导外观注意机制，以准确地模仿预期的个性化概念的外观。此外，我们引入了遮罩引导的噪声混合，以在编辑过程中保护非个性化区域。最后，我们应用背景稀释技术来最小化属性泄漏，即个性化概念属性与其他物体在图像中的不希望的混杂。在我们的实验中，我们展示了尽管没有进行调优，所提出的方法在单个和多个个性化概念推理方面均优于现有模型。 

---
# TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding 

**Title (ZH)**: TETRIS：批推测解码的最佳草稿 token 选择 

**Authors**: Zhaoxuan Wu, Zijian Zhou, Arun Verma, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2502.15197)  

**Abstract**: We propose TETRIS, a novel method that optimizes the total throughput of batch speculative decoding in multi-request settings. Unlike existing methods that optimize for a single request or a group of requests as a whole, TETRIS actively selects the most promising draft tokens (for every request in a batch) to be accepted when verified in parallel, resulting in fewer rejected tokens and hence less wasted computing resources. Such an effective resource utilization to achieve fast inference in large language models (LLMs) is especially important to service providers with limited inference capacity. Compared to baseline speculative decoding, TETRIS yields a consistently higher acceptance rate and more effective utilization of the limited inference capacity. We show theoretically and empirically that TETRIS outperforms baseline speculative decoding and existing methods that dynamically select draft tokens, leading to a more efficient batch inference in LLMs. 

**Abstract (ZH)**: 我们提出了一种新颖的方法——TETRIS，该方法旨在优化多请求设置中批量推测性解码的总吞吐量。与现有方法针对单个请求或一组请求进行优化不同，TETRIS 能够在并行验证时主动选择一批请求中最有可能成功的草稿标记（draft tokens）进行接受，从而减少被拒绝的标记数量，进而减少计算资源的浪费。这种有效的资源利用对于具有有限推理能力的服务提供商来说尤为重要，特别是在实现大型语言模型（LLMs）的快速推理方面尤为关键。与基线推测性解码相比，TETRIS 在接受率和对有限推理能力的有效利用方面表现出更优的效果。我们从理论上和实验上证明，TETRIS 在批量推理方面优于基线推测性解码和现有的动态选择草稿标记的方法，从而使得LLMs的批量推理更加高效。 

---
# Scale-Free Graph-Language Models 

**Title (ZH)**: 规模无标度图形-语言模型 

**Authors**: Jianglin Lu, Yixuan Liu, Yitian Zhang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15189)  

**Abstract**: Graph-language models (GLMs) have demonstrated great potential in graph-based semi-supervised learning. A typical GLM consists of two key stages: graph generation and text embedding, which are usually implemented by inferring a latent graph and finetuning a language model (LM), respectively. However, the former often relies on artificial assumptions about the underlying edge distribution, while the latter requires extensive data annotations. To tackle these challenges, this paper introduces a novel GLM that integrates graph generation and text embedding within a unified framework. Specifically, for graph generation, we leverage an inherent characteristic of real edge distribution--the scale-free property--as a structural prior. We unexpectedly find that this natural property can be effectively approximated by a simple k-nearest neighbor (KNN) graph. For text embedding, we develop a graph-based pseudo-labeler that utilizes scale-free graphs to provide complementary supervision for improved LM finetuning. Extensive experiments on representative datasets validate our findings on the scale-free structural approximation of KNN graphs and demonstrate the effectiveness of integrating graph generation and text embedding with a real structural prior. Our code is available at this https URL. 

**Abstract (ZH)**: 图语言模型（GLMs）在基于图的半监督学习中展现出了巨大的潜力。典型的GLM通常包含两个关键阶段：图生成和文本嵌入，前者通常通过推断潜在图来实现，后者通常通过调整语言模型（LM）来实现。然而，前者往往依赖于对潜在边分布的人工假设，而后者则需要大量数据标注。为了解决这些挑战，本文提出了一种新的GLM，将其图生成和文本嵌入整合到一个统一的框架中。具体来说，在图生成阶段，我们利用真实边分布中固有的一个内在特性——无标度特性——作为结构先验。我们意外地发现，这种自然的特性可以有效地通过简单的k近邻（KNN）图来逼近。在文本嵌入阶段，我们开发了一种基于图的伪标签器，利用无标度图来为语言模型的调整提供互补监督，从而提高其性能。在典型数据集上的大量实验验证了我们关于KNN图的无标度结构逼近的研究发现，并展示了将图生成和文本嵌入与实际结构先验结合起来的有效性。我们的代码可在以下链接获取：[链接地址]。 

---
# LUMINA-Net: Low-light Upgrade through Multi-stage Illumination and Noise Adaptation Network for Image Enhancement 

**Title (ZH)**: LUMINA-Net：通过多阶段光照和噪声适应网络实现低光照条件下的图像增强 

**Authors**: Namrah Siddiqua, Kim Suneung  

**Link**: [PDF](https://arxiv.org/pdf/2502.15186)  

**Abstract**: Low-light image enhancement (LLIE) is a crucial task in computer vision aimed to enhance the visual fidelity of images captured under low-illumination conditions. Conventional methods frequently struggle to mitigate pervasive shortcomings such as noise, over-exposure, and color distortion thereby precipitating a pronounced degradation in image quality. To address these challenges, we propose LUMINA-Net an advanced deep learning framework designed specifically by integrating multi-stage illumination and reflectance modules. First, the illumination module intelligently adjusts brightness and contrast levels while meticulously preserving intricate textural details. Second, the reflectance module incorporates a noise reduction mechanism that leverages spatial attention and channel-wise feature refinement to mitigate noise contamination. Through a comprehensive suite of experiments conducted on LOL and SICE datasets using PSNR, SSIM and LPIPS metrics, surpassing state-of-the-art methodologies and showcasing its efficacy in low-light image enhancement. 

**Abstract (ZH)**: 低光照图像增强（LLIE）是计算机视觉中的一个关键任务，旨在提高在低光照条件下拍摄的图像的视觉保真度。传统方法通常难以缓解诸如噪声、过度曝光和颜色失真等普遍存在的问题，从而导致图像质量显着下降。为了应对这些挑战，我们提出了一种名为LUMINA-Net的高级深度学习框架，该框架通过集成多阶段的光照和反射模块特别设计而成。首先，光照模块智能地调整亮度和对比度水平，同时仔细保留复杂的纹理细节。其次，反射模块采用噪声减少机制，利用空间注意力和通道级特征精炼来缓解噪声污染。通过在LOL和SICE数据集上进行全面的实验，并使用PSNR、SSIM和LPIPS指标超越现有最先进的方法，展示了其在低光照图像增强方面的有效性。 

---
# Key Body Posture Characteristics of Short-distance Speed Skaters at the Start Based on Artificial Intelligence 

**Title (ZH)**: 基于人工智能的短距离速滑运动员起跑关键身体姿态特征研究 

**Authors**: Zhang Xueliana, Fang Yingjieb, Liu Hang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15185)  

**Abstract**: Objective To conduct biomechanical analysis on the starting technique of male short-distance speed skating athletes in China and determine the key factors affecting the effectiveness of the starting movement. Methods 13 high-level male short-distance speed skating athletes were selected as the test subjects, and kinematic data were collected using an artificial intelligence video capture and analysis system. The body posture features and their effects on the starting movement performance were analyzed in the three stages of starting preparation, starting, and sprinting. Results The post-stability angle, anterior knee angle of the front leg, posterior knee angle of the rear leg, and stride length showed moderate to high positive correlations with the starting speed during the starting preparation stage. The trunk angle showed a high negative correlation with the starting speed. The trunk angle (TO4, TD4, TO6, TD6), hip angle (TO1, TO4, TO6), and knee angle (TD1) showed moderate to high negative correlations with the effectiveness of the starting movement during the starting and sprinting stages. The knee angle (TD2), ice-contact angle (TD2, TD4, TD5, TD6), and propulsion angle (TO1, TO4, TO7) showed moderate positive correlations with the effectiveness of the starting movement. Conclusion Stride length, left knee angle, and post-stability angle are the key factors affecting the starting speed. The larger the post-stability angle and left knee angle and the longer the stride length, the faster the starting speed. During the starting and sprinting stages, the smaller the ice-contact angle and propulsion angle, the greater the trunk angle and hip angle changes, the more effective the starting movement. 

**Abstract (ZH)**: 目标：对中国高水平男性短距离速度滑冰运动员的起跑技术进行生物力学分析，确定影响起跑动作效果的关键因素。

方法：选择13名高水平男性短距离速度滑冰运动员作为测试对象，并使用人工智能视频捕捉与分析系统收集运动学数据。在起跑准备、起跑和冲刺三个阶段分析身体姿态特征及其对起跑动作表现的影响。

结果：起始准备阶段，后稳定性角、前腿前角、后腿后角和步长与起始速度呈现出中等到高度的正相关关系。躯干角与起始速度表现出高度的负相关关系。起跑和冲刺阶段，躯干角（TO4、TD4、TO6、TD6）、髋角（TO1、TO4、TO6）和膝角（TD1）与起始动作的效果表现出中等到高度的负相关关系。膝角（TD2）、冰接触角（TD2、TD4、TD5、TD6）和推动力角（TO1、TO4、TO7）与起始动作的效果表现出中等到高度的正相关关系。

结论：步长、左膝角和后稳定性角是影响起始速度的关键因素。后稳定性角和左膝角越大，步长越长，起始速度越快。在起跑和冲刺阶段，冰接触角和推动力角越小，躯干角和髋角的变化越大，起始动作越有效。 

---
# LEDD: Large Language Model-Empowered Data Discovery in Data Lakes 

**Title (ZH)**: LEDD：大数据湖中大型语言模型赋能的数据发现 

**Authors**: Qi An, Chihua Ying, Yuqing Zhu, Yihao Xu, Manwei Zhang, Jianmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15182)  

**Abstract**: Data discovery in data lakes with ever increasing datasets has long been recognized as a big challenge in the realm of data management, especially for semantic search of and hierarchical global catalog generation of tables. While large language models (LLMs) facilitate the processing of data semantics, challenges remain in architecting an end-to-end system that comprehensively exploits LLMs for the two semantics-related tasks. In this demo, we propose LEDD, an end-to-end system with an extensible architecture that leverages LLMs to provide hierarchical global catalogs with semantic meanings and semantic table search for data lakes. Specifically, LEDD can return semantically related tables based on natural-language specification. These features make LEDD an ideal foundation for downstream tasks such as model training and schema linking for text-to-SQL tasks. LEDD also provides a simple Python interface to facilitate the extension and the replacement of data discovery algorithms. 

**Abstract (ZH)**: 数据湖中随时间不断增加的数据集的数据发现长期以来被认可为数据管理领域的一个重大挑战，尤其是在进行语义搜索和生成分层全局目录方面的挑战。虽然大型语言模型（LLMs）有助于处理数据语义，但仍存在构建一个端到端系统来全面利用LLMs进行两类语义相关任务的挑战。在此演示中，我们提出了一种名为LEDD的端到端系统，其架构可扩展，利用LLMs提供具有语义意义的分层全局目录和数据湖中的语义表搜索。具体而言，LEDD可以根据自然语言规范返回语义相关表。这些功能使LEDD成为诸如文本生成SQL任务中模型训练和模式链接等下游任务的理想基础。LEDD还提供了一个简单的Python接口，以简化数据发现算法的扩展和替换。 

---
# Methods and Trends in Detecting Generated Images: A Comprehensive Review 

**Title (ZH)**: 生成图像检测的方法与趋势：一篇全面综述 

**Authors**: Arpan Mahara, Naphtali Rishe  

**Link**: [PDF](https://arxiv.org/pdf/2502.15176)  

**Abstract**: The proliferation of generative models, such as Generative Adversarial Networks (GANs), Diffusion Models, and Variational Autoencoders (VAEs), has enabled the synthesis of high-quality multimedia data. However, these advancements have also raised significant concerns regarding adversarial attacks, unethical usage, and societal harm. Recognizing these challenges, researchers have increasingly focused on developing methodologies to detect synthesized data effectively, aiming to mitigate potential risks. Prior reviews have primarily focused on deepfake detection and often lack coverage of recent advancements in synthetic image detection, particularly methods leveraging multimodal frameworks for improved forensic analysis. To address this gap, the present survey provides a comprehensive review of state-of-the-art methods for detecting and classifying synthetic images generated by advanced generative AI models. This review systematically examines core detection methodologies, identifies commonalities among approaches, and categorizes them into meaningful taxonomies. Furthermore, given the crucial role of large-scale datasets in this field, we present an overview of publicly available datasets that facilitate further research and benchmarking in synthetic data detection. 

**Abstract (ZH)**: 生成模型，如生成对抗网络（GANs）、扩散模型和变分自编码器（VAEs）的泛滥，使得高质量多媒体数据的合成成为可能。然而，这些进展同时也引发了关于对抗攻击、不道德使用以及社会危害的重大关切。认识到这些挑战后，研究人员越来越关注开发有效检测合成数据的方法，旨在减轻潜在风险。之前的综述主要集中在深伪检测方面，往往忽略了合成图像检测领域的最新进展，尤其是利用多模态框架进行增强的法医分析方法。为了填补这一空白，本综述全面回顾了检测和分类由先进生成AI模型生成的合成图像的最新方法。本综述系统地检查了核心检测方法，找到了不同方法之间的共通之处，并将它们归类为有意义的分类体系。此外，鉴于大规模数据集在该领域中的关键作用，本文还介绍了公开可用的数据集，以促进合成数据检测领域的进一步研究和基准测试。 

---
# Extreme Speech Classification in the Era of LLMs: Exploring Open-Source and Proprietary Models 

**Title (ZH)**: 在大规模语言模型时代的极端语音分类：探索开源与专有模型 

**Authors**: Sarthak Mahajan, Nimmi Rangaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2502.15155)  

**Abstract**: In recent years, widespread internet adoption and the growth in userbase of various social media platforms have led to an increase in the proliferation of extreme speech online. While traditional language models have demonstrated proficiency in distinguishing between neutral text and non-neutral text (i.e. extreme speech), categorizing the diverse types of extreme speech presents significant challenges. The task of extreme speech classification is particularly nuanced, as it requires a deep understanding of socio-cultural contexts to accurately interpret the intent of the language used by the speaker. Even human annotators often disagree on the appropriate classification of such content, emphasizing the complex and subjective nature of this task. The use of human moderators also presents a scaling issue, necessitating the need for automated systems for extreme speech classification. The recent launch of ChatGPT has drawn global attention to the potential applications of Large Language Models (LLMs) across a diverse variety of tasks. Trained on vast and diverse corpora, and demonstrating the ability to effectively capture and encode contextual information, LLMs emerge as highly promising tools for tackling this specific task of extreme speech classification. In this paper, we leverage the Indian subset of the extreme speech dataset from Maronikolakis et al. (2022) to develop an effective classification framework using LLMs. We evaluate open-source Llama models against closed-source OpenAI models, finding that while pre-trained LLMs show moderate efficacy, fine-tuning with domain-specific data significantly enhances performance, highlighting their adaptability to linguistic and contextual nuances. Although GPT-based models outperform Llama models in zero-shot settings, the performance gap disappears after fine-tuning. 

**Abstract (ZH)**: 近年来，互联网的广泛应用以及各种社交媒体用户基数的增长，导致网络极端言论的传播更加泛滥。虽然传统的语言模型在区分中性文本和非中性文本（即极端言论）方面表现出色，但对多种类型的极端言论进行分类却具有极大的挑战性。极端言论分类任务特别复杂，因为它要求深入理解社会文化背景，以准确解读说话者使用的语言意图。即使是人类注释员，也常常在适当分类此类内容方面存在分歧，这突显了该任务的复杂性和主观性。使用人类审查员也存在扩展问题，因此迫切需要自动系统来处理极端言论分类任务。ChatGPT的最近推出引起了全球对大型语言模型（LLMs）在多样任务中的潜在应用的关注。这些模型在大量和多样化的语料库上进行训练，并具备有效捕捉和编码上下文信息的能力，使其成为处理特定极端言论分类任务的强有力工具。本文利用马龙尼卡斯基等人（2022）的极端言论数据集中印度子集，开发了一个基于LLM的有效分类框架。我们对比了开源的Llama模型和闭源的OpenAI模型，结果显示，虽然预先训练的语言模型表现适中，但在特定领域数据上进行微调显著提升了性能，突显了它们对语言和上下文细微差别的适应能力。尽管基于GPT的模型在零样本设置下表现优于Llama模型，但在进行了微调后，性能差距消失。 

---
# Confidence-Weighted Boundary-Aware Learning for Semi-Supervised Semantic Segmentation 

**Title (ZH)**: 带有边界aware性的置信加权半监督语义分割学习 

**Authors**: Ebenezer Tarubinga, Jenifer Kalafatovich Espinoza  

**Link**: [PDF](https://arxiv.org/pdf/2502.15152)  

**Abstract**: Semi-supervised semantic segmentation (SSSS) aims to improve segmentation performance by utilising unlabeled data alongside limited labeled samples. Existing SSSS methods often face challenges such as coupling, where over-reliance on initial labeled data leads to suboptimal learning; confirmation bias, where incorrect predictions reinforce themselves repeatedly; and boundary blur caused by insufficient boundary-awareness and ambiguous edge information. To address these issues, we propose CW-BASS, a novel framework for SSSS. In order to mitigate the impact of incorrect predictions, we assign confidence weights to pseudo-labels. Additionally, we leverage boundary-delineation techniques, which, despite being extensively explored in weakly-supervised semantic segmentation (WSSS) remain under-explored in SSSS. Specifically, our approach: (1) reduces coupling through a confidence-weighted loss function that adjusts the influence of pseudo-labels based on their predicted confidence scores, (2) mitigates confirmation bias with a dynamic thresholding mechanism that learns to filter out pseudo-labels based on model performance, (3) resolves boundary blur with a boundary-aware module that enhances segmentation accuracy near object boundaries, and (4) reduces label noise with a confidence decay strategy that progressively refines pseudo-labels during training. Extensive experiments on the Pascal VOC 2012 and Cityscapes demonstrate that our method achieves state-of-the-art performance. Moreover, using only 1/8 or 12.5\% of labeled data, our method achieves a mIoU of 75.81 on Pascal VOC 2012, highlighting its effectiveness in limited-label settings. 

**Abstract (ZH)**: 半监督语义分割（Semi-supervised Semantic Segmentation, SSSS）的目标是通过利用未标注数据和有限的标注样本来提高分割性能。现有的SSSS方法通常面临诸如耦合问题、确认偏差以及边界模糊等挑战。耦合问题指的是过度依赖初始标注数据会导致学习效果不佳。确认偏差是指错误预测会反复得到加强。边界模糊则源于缺乏边界意识和含糊的边缘信息。为了解决这些问题，我们提出了一种新的SSSS框架——CW-BASS。为了减轻错误预测的影响，我们为伪标签分配了置信度权重。此外，我们利用了边界分隔技术，尽管这些技术在弱监督语义分割（Weakly-supervised Semantic Segmentation, WSSS）中得到了广泛应用，但在SSSS中仍然处于探索阶段。具体而言，我们采取以下措施：（1）通过一种置信度加权损失函数来减少耦合，该函数根据伪标签的预测置信度调整其影响；（2）通过动态阈值机制来减轻确认偏差，该机制会根据模型性能学习过滤掉伪标签；（3）通过增加边界感知模块来解决边界模糊问题，该模块能提高物体边界附近的分割精度；（4）通过置信度衰减策略在训练过程中逐步精炼伪标签，以减少标签噪声。我们在Pascal VOC 2012和Cityscapes数据集上的广泛实验表明，我们的方法达到了最先进的性能。此外，仅使用标注数据的1/8或12.5%，我们的方法在Pascal VOC 2012数据集上的mIoU达到75.81，进一步证明了其在标注数据有限条件下的有效性。 

---
# Projection Optimization: A General Framework for Multi-Objective and Multi-Group RLHF 

**Title (ZH)**: 投影优化：一种多目标和多组RLHF的通用框架 

**Authors**: Nuoya Xiong, Aarti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2502.15145)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a widely used fine-tuning approach that aligns machine learning model, particularly Language Model (LM) with human preferences. There are typically multiple objectives driving the preference, hence humans find it easier to express per-objective comparisons rather than a global preference between two choices. %, e.g. compare two papers on their novelty, clarity, correctness, etc. Multi-Objective RLHF (MORLHF) aims to use per-objective preference feedback and achieve Pareto optimality among these objectives by aggregating them into a single unified objective for optimization. However, nearly all prior works rely on linear aggregation, which rules out policies that favor specific objectives such as the worst one. The only existing approach using non-linear aggregation is computationally expensive due to its reward-based nature and the need for retraining whenever the aggregation parameters change. In this work, we address this limitation by transforming the non-linear aggregation maximization problem into a series of sub-problems. Each sub-problem involves only linear aggregation, making it computationally efficient to solve. We further extend our framework to handle multi-group scenarios, where each group has distinct weights for the objectives. Our method enables achieving consensus or maximizing the aggregated objective across all groups. Theoretically, we demonstrate that our algorithmic framework achieves sublinear regret and can be easily adapted to a reward-free algorithm. Empirically, leveraging our theoretical insights, we propose a nearly training-free algorithm once the optimal policies for individual objectives are obtained. 

**Abstract (ZH)**: 强化学习与人类反馈（RLHF）是一种广泛使用的微调方法，用于使机器学习模型，尤其是语言模型（LM）与人类偏好相一致。通常有多个目标驱动这些偏好，因此人类更容易表达针对每个目标的比较，而非两者的全球偏好。例如，人类可能更容易比较两篇论文在新颖性、清晰度和正确性等方面的差异。多目标RLHF（MORLHF）旨在利用针对每个目标的偏好反馈，并通过将这些目标聚合为单一统一目标来进行优化，从而实现这些目标的帕累托最优。然而，几乎所有先前的工作都依赖于线性聚合，这排除了倾向于特定目标（如最差目标）的策略。目前唯一使用非线性聚合的方法由于基于奖励且需要在聚合参数变化时重新训练而计算成本极高。在本文中，我们通过将非线性聚合最大化问题转化为一系列子问题来解决这一局限性。每个子问题仅涉及线性聚合，从而使该问题的求解更具计算效率。我们进一步扩展了我们的框架以处理多组场景，其中每个组对目标的权重各不相同。我们的方法使得在所有组中实现共识或最大化聚合目标成为可能。理论上，我们证明了我们的算法框架实现了亚线性遗憾，并且可以轻松适应无奖励算法。从经验上看，利用我们的理论见解，我们提出了一种近乎无需训练的算法，只要获得了针对每个目标的最优策略。 

---
# Chain-of-Rank: Enhancing Large Language Models for Domain-Specific RAG in Edge Device 

**Title (ZH)**: 链式秩优化：增强边缘设备上面向特定领域的大语言模型的检索增强生成（RAG）能力 

**Authors**: Juntae Lee, Jihwan Bang, Seunghan Yang, Kyuhong Shim, Simyung Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15134)  

**Abstract**: Retrieval-augmented generation (RAG) with large language models (LLMs) is especially valuable in specialized domains, where precision is critical. To more specialize the LLMs into a target domain, domain-specific RAG has recently been developed by allowing the LLM to access the target domain early via finetuning. The domain-specific RAG makes more sense in resource-constrained environments like edge devices, as they should perform a specific task (e.g. personalization) reliably using only small-scale LLMs. While the domain-specific RAG is well-aligned with edge devices in this respect, it often relies on widely-used reasoning techniques like chain-of-thought (CoT). The reasoning step is useful to understand the given external knowledge, and yet it is computationally expensive and difficult for small-scale LLMs to learn it. Tackling this, we propose the Chain of Rank (CoR) which shifts the focus from intricate lengthy reasoning to simple ranking of the reliability of input external documents. Then, CoR reduces computational complexity while maintaining high accuracy, making it particularly suited for resource-constrained environments. We attain the state-of-the-art (SOTA) results in benchmarks, and analyze its efficacy. 

**Abstract (ZH)**: 利用大规模语言模型（LLM）的检索增强生成（RAG）在专门领域中尤其有价值，因为精确度至关重要。为了使LLM更加专门化于目标领域，最近通过微调允许LLM在早期访问目标领域，发展了领域特定的RAG。在资源受限的环境中，如边缘设备，领域特定的RAG更为合理，因为它们仅使用小型规模的LLM来可靠地执行特定任务（例如个性化）。虽然从这一点来看，领域特定的RAG与边缘设备非常契合，但它通常依赖于广泛使用的推理技术，如思维链（CoT）。推理步骤有助于理解给定的外部知识，但对小型规模的LLM来说，计算成本较高且难以学习。为了解决这一问题，我们提出了一种称为“推理链”（CoR）的方法，将重点从复杂的长篇推理转移到简单地对输入外部文档可靠性的排序。然后，CoR 在保持高准确性的同时，减少了计算复杂度，使其特别适合资源受限的环境。我们在基准测试中实现了目前最先进的（SOTA）结果，并对其有效性进行了分析。 

---
# CoT-ICL Lab: A Petri Dish for Studying Chain-of-Thought Learning from In-Context Demonstrations 

**Title (ZH)**: CoT-ICL实验室：研究基于上下文提示学习的推理链学习的一个平台 

**Authors**: Vignesh Kothapalli, Hamed Firooz, Maziar Sanjabi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15132)  

**Abstract**: We introduce CoT-ICL Lab, a framework and methodology to generate synthetic tokenized datasets and systematically study chain-of-thought (CoT) in-context learning (ICL) in language models. CoT-ICL Lab allows fine grained control over the complexity of in-context examples by decoupling (1) the causal structure involved in chain token generation from (2) the underlying token processing functions. We train decoder-only transformers (up to 700M parameters) on these datasets and show that CoT accelerates the accuracy transition to higher values across model sizes. In particular, we find that model depth is crucial for leveraging CoT with limited in-context examples, while more examples help shallow models match deeper model performance. Additionally, limiting the diversity of token processing functions throughout training improves causal structure learning via ICL. We also interpret these transitions by analyzing transformer embeddings and attention maps. Overall, CoT-ICL Lab serves as a simple yet powerful testbed for theoretical and empirical insights into ICL and CoT in language models. 

**Abstract (ZH)**: 我们介绍了CoT-ICL Lab，这是一个框架和方法论，用于生成合成标记数据集，并系统地研究语言模型中基于链的推理（CoT）的在上下文学习（ICL）。CoT-ICL Lab 通过将（1）链式标记生成中的因果结构与（2）底层标记处理函数解耦，实现了对在上下文示例复杂性的精细控制。我们使用这些数据集训练仅解码器变换器模型（最多700M参数），并展示了CoT在不同模型大小下加速准确度跃迁的能力。具体而言，我们发现，在上下文示例有限的情况下，模型深度对于利用CoT至关重要，而更多的示例有助于浅层模型达到深层模型的性能。此外，在训练过程中限制标记处理函数的多样性可以改进通过ICL学习的因果结构。我们还通过对变换器嵌入和注意力图的分析来解释这些转变。总体而言，CoT-ICL Lab 提供了一个简单而强大的实验平台，用于探索语言模型中的ICL和CoT的理论和实证见解。 

---
# Unveiling Reasoning Thresholds in Language Models: Scaling, Fine-Tuning, and Interpretability through Attention Maps 

**Title (ZH)**: 揭开语言模型推理阈值的面纱：通过注意力图实现的扩展、微调及可解释性 

**Authors**: Yen-Che Hsiao, Abhishek Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2502.15120)  

**Abstract**: This study investigates the in-context learning capabilities of various decoder-only transformer-based language models with different model sizes and training data, including GPT2, SmolLM2, OpenELM, TinyLlama, Stable LM, and Gemma 2. We identify a critical parameter threshold (~1.6 billion), beyond which reasoning performance improves significantly in tasks such as commonsense reasoning in multiple-choice question answering and deductive reasoning. Specifically, models above this threshold achieve better success rates in chain-of-thought (CoT) prompting for deductive reasoning tasks, especially those requiring longer reasoning chains, such as proof by contradiction and disjunction elimination. To address limitations in sub-threshold models, we demonstrate that fine-tuning with task-specific exemplars substantially enhances reasoning performance, enabling accurate CoT generation even without additional exemplars in the prompt for tasks with shorter reasoning chains. Finally, our analysis of attention maps reveals that models capable of generating correct CoTs exhibit higher token-level attention scores on subsequent correct tokens and the correct parts of speech, providing interpretability insights into reasoning processes. These findings collectively advance understanding of reasoning capabilities in decoder-only transformer-based models. The code is available at: this https URL. 

**Abstract (ZH)**: 本研究探讨了不同模型大小和训练数据量的各类仅解码器变压器语言模型的上下文学习能力，包括GPT2、SmolLM2、OpenELM、TinyLlama、StableLM和Gemma 2。我们发现一个关键的参数阈值（约16亿），在此阈值以上，推理性能在常识推理的多项选择题回答和演绎推理任务中显著提高。具体而言，超过此阈值的模型在演绎推理任务中的链式思维（CoT）提示下表现出更高的成功率，特别是一些需要更长推理链的任务，如反证法和析取消去。为了解决亚阈值模型的限制，我们证明了使用特定任务示例的微调可以大幅提升推理性能，即使在提示中没有额外示例的情况下，也能生成准确的CoT，特别是针对具有较短推理链的任务。最后，我们对注意力图的分析表明，能够生成正确CoT的模型在后续正确词汇和正确词性的令牌级注意力分数较高，这为我们提供了关于推理过程的可解释性洞见。这些发现共同推进了对仅解码器变压器模型推理能力的理解。代码可在以下链接获取：[相应链接]。 

---
# CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models 

**Title (ZH)**: CurricuVLM：通过基于视觉-语言模型的个性化安全关键课程学习实现安全自动驾驶 

**Authors**: Zihao Sheng, Zilin Huang, Yansong Qu, Yue Leng, Sruthi Bhavanam, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15119)  

**Abstract**: Ensuring safety in autonomous driving systems remains a critical challenge, particularly in handling rare but potentially catastrophic safety-critical scenarios. While existing research has explored generating safety-critical scenarios for autonomous vehicle (AV) testing, there is limited work on effectively incorporating these scenarios into policy learning to enhance safety. Furthermore, developing training curricula that adapt to an AV's evolving behavioral patterns and performance bottlenecks remains largely unexplored. To address these challenges, we propose CurricuVLM, a novel framework that leverages Vision-Language Models (VLMs) to enable personalized curriculum learning for autonomous driving agents. Our approach uniquely exploits VLMs' multimodal understanding capabilities to analyze agent behavior, identify performance weaknesses, and dynamically generate tailored training scenarios for curriculum adaptation. Through comprehensive analysis of unsafe driving situations with narrative descriptions, CurricuVLM performs in-depth reasoning to evaluate the AV's capabilities and identify critical behavioral patterns. The framework then synthesizes customized training scenarios targeting these identified limitations, enabling effective and personalized curriculum learning. Extensive experiments on the Waymo Open Motion Dataset show that CurricuVLM outperforms state-of-the-art baselines across both regular and safety-critical scenarios, achieving superior performance in terms of navigation success, driving efficiency, and safety metrics. Further analysis reveals that CurricuVLM serves as a general approach that can be integrated with various RL algorithms to enhance autonomous driving systems. The code and demo video are available at: this https URL. 

**Abstract (ZH)**: 确保自动驾驶系统的安全性仍然是一个关键的挑战，尤其是在处理那些虽然稀少但可能导致灾难性后果的安全关键场景时。尽管现有研究已经探索了生成自动驾驶汽车（AV）测试所需的安全关键场景，但将这些场景有效地融入策略学习以增强安全性的相关工作仍然较为有限。此外，开发能够适应自动驾驶车辆不断演变的行为模式和性能瓶颈的训练课程依然鲜有研究。为了解决这些挑战，我们提出了CurricuVLM框架，这是一种新颖的方法，利用视觉语言模型（VLMs）来实现个性化课程学习，为自动驾驶代理提供定制化的训练课程。我们的方法独特地利用了VLMs的多模态理解能力，分析代理行为，识别性能短板，并动态生成定制化的训练场景以适应课程学习。通过综合分析包含叙事描述的不安全驾驶情况，CurricuVLM深入地进行推理，评估自动驾驶汽车的能力，并识别关键的行为模式。然后，框架综合生成针对性的定制化训练场景，针对识别出的局限性，实现有效和个性化的课程学习。在Waymo Open Motion数据集上的广泛实验表明，CurricuVLM在常规和安全关键场景中均优于最新的基准，其在导航成功、驾驶效率和安全指标上的表现更加出色。进一步的分析表明，CurricuVLM可以作为一种通用方法，与各种强化学习（RL）算法结合，以增强自动驾驶系统。感兴趣的读者可以通过以下链接获取相关代码和演示视频：[此处插入链接]。 

---
# Assessing a Single Student's Concentration on Learning Platforms: A Machine Learning-Enhanced EEG-Based Framework 

**Title (ZH)**: 基于机器学习增强的EEG技术框架：评估单个学生在学习平台上的注意力 

**Authors**: Zewen Zhuo, Mohamad Najafi, Hazem Zein, Amine Nait-Ali  

**Link**: [PDF](https://arxiv.org/pdf/2502.15107)  

**Abstract**: This study introduces a specialized pipeline designed to classify the concentration state of an individual student during online learning sessions by training a custom-tailored machine learning model. Detailed protocols for acquiring and preprocessing EEG data are outlined, along with the extraction of fifty statistical features from five EEG signal bands: alpha, beta, theta, delta, and gamma. Following feature extraction, a thorough feature selection process was conducted to optimize the data inputs for a personalized analysis. The study also explores the benefits of hyperparameter fine-tuning to enhance the classification accuracy of the student's concentration state. EEG signals were captured from the student using a Muse headband (Gen 2), equipped with five electrodes (TP9, AF7, AF8, TP10, and a reference electrode NZ), during engagement with educational content on computer-based e-learning platforms. Employing a random forest model customized to the student's data, we achieved remarkable classification performance, with test accuracies of 97.6% in the computer-based learning setting and 98% in the virtual reality setting. These results underscore the effectiveness of our approach in delivering personalized insights into student concentration during online educational activities. 

**Abstract (ZH)**: 本研究介绍了一种专门的流程，用于通过训练定制化的机器学习模型来分类个体学生在在线学习过程中所处的注意力状态。详细描述了获取和预处理脑电图（EEG）数据的协议，以及从alpha、beta、theta、delta和gamma五个EEG信号频带中提取出五十个统计特征。在特征提取之后，进行了详尽的特征选择过程，以优化个性化的数据分析输入。研究还探讨了超参数精细调整对提高学生注意力状态分类准确率的益处。EEG信号使用配备有五个电极（TP9、AF7、AF8、TP10和参考电极NZ）的Muse头带（Gen 2）在计算机基于的在线学习平台上的教育内容互动期间被捕捉。利用符合学生数据的随机森林模型，我们在计算机辅助学习环境中达到了97.6%的测试准确率，而在虚拟现实环境中达到了98%的准确率。这些结果突显了本研究方法在在线教育活动中提供个性化学生注意力洞察方面的有效性。 

---
# Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans 

**Title (ZH)**: 不分析嵌入，而分析神经元：理解大规模语言模型表示与人类认知的对齐时机和位置 

**Authors**: Masha Fedzechkina, Eleonora Gualdoni, Sinead Williamson, Katherine Metcalf, Skyler Seto, Barry-John Theobald  

**Link**: [PDF](https://arxiv.org/pdf/2502.15090)  

**Abstract**: Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to the study of representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., 'cat') and then analyze the corresponding activation patterns. Our findings reveal that LLM representations closely align with human representations inferred from behavioral data. Notably, this alignment surpasses that of word embeddings, which have been center stage in prior work on human and model alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts. Specifically, we show that LLMs organize concepts in a way that reflects hierarchical relationships interpretable to humans (e.g., 'animal'-'dog'). 

**Abstract (ZH)**: 现代大型语言模型（LLMs）在某些任务上取得了 impressive 的表现，但在其他任务上却表现出明显非人类的行为模式。这引发了这样一个问题：LLM 学习到的表征与人类的表征之间有多大的对齐程度。在本文中，我们介绍了一种新的表征对齐研究方法：我们借鉴了激活引导研究中的方法来识别负责特定概念（例如，“猫”）的神经元，然后分析相应的激活模式。我们的研究结果揭示了，LLM 的表征与从行为数据推断出的人类表征高度对齐。值得注意的是，这种对齐程度超过了之前工作中占据中心位置的词嵌入的表现。此外，我们的方法还使得我们能够更细致地研究 LLM 如何表示概念。具体而言，我们展示了 LLM 以一种能够反映可解释的层次关系（例如，“动物”-“狗”）的方式组织概念。 

---
# UPCORE: Utility-Preserving Coreset Selection for Balanced Unlearning 

**Title (ZH)**: UPCORE：保用 coppwerpreserving 聚素心选择以实现平衡遗忘

实际上，更准确的翻译应该是：

UPCORE：保用性合权聚样点选择以实现平衡遗忘

这里的翻译尽量保持了原词的意义，但“coreset”通常不会被直接翻译，而是在上下文中给出具体的解释或保留原词。完整的翻译可以这样表述：

UPCORE：保用性合权聚素心选择以实现平衡遗忘 

**Authors**: Vaidehi Patil, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.15082)  

**Abstract**: User specifications or legal frameworks often require information to be removed from pretrained models, including large language models (LLMs). This requires deleting or "forgetting" a set of data points from an already-trained model, which typically degrades its performance on other data points. Thus, a balance must be struck between removing information and keeping the model's other abilities intact, with a failure to balance this trade-off leading to poor deletion or an unusable model. To this end, we propose UPCORE (Utility-Preserving Coreset Selection), a method-agnostic data selection framework for mitigating collateral damage during unlearning. Finding that the model damage is correlated with the variance of the model's representations on the forget set, we selectively prune the forget set to remove outliers, thereby minimizing model degradation after unlearning. We evaluate UPCORE across three standard unlearning methods consistently achieving a superior balance between the competing objectives of deletion efficacy and model preservation. To better evaluate this trade-off, we introduce a new metric, measuring the area-under-the-curve (AUC) across standard metrics. We find that UPCORE improves both standard metrics and AUC, benefitting from positive transfer between the coreset and pruned points while reducing negative transfer from the forget set to points outside of it. 

**Abstract (ZH)**: 用户规范或法律框架往往要求从预训练模型中移除某些信息，包括大型语言模型（LLMs）。这通常需要从已经训练好的模型中删除或“遗忘”一组数据点，这往往会降低模型在其他数据点上的性能。因此，在移除信息的同时保留模型的其他能力之间必须找到一个平衡，未能妥善平衡这一权衡关系会导致删除效果不佳或模型不可用。为此，我们提出了一种名为UPCORE（保持用途的核选择）的方法，这是一种不依赖于具体方法的数据筛选框架，用于减轻遗忘过程中的副损伤。我们发现，模型的损害与其在遗忘集上表示的方差相关，因此通过有选择性地剔除异常值来精简遗忘集，从而最小化遗忘后的模型退化。

在三种标准遗忘方法中，我们评估了UPCORE，始终能够在删除效果和模型保存之间实现更优的平衡。为了更好地评估这种权衡，我们引入了一个新的评价指标，该指标衡量标准指标下的面积下曲线（AUC）。我们发现，UPCORE不仅提高了标准指标，还提高了AUC。通过正向迁移来自核的数据点和剔除点之间的积极影响，同时减少了来自遗忘集对其他点的负面影响。 

---
# Can Hallucination Correction Improve Video-Language Alignment? 

**Title (ZH)**: 可以纠正幻觉以改善视频-语言对齐吗？ 

**Authors**: Lingjun Zhao, Mingyang Xie, Paola Cascante-Bonilla, Hal Daumé III, Kwonjoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.15079)  

**Abstract**: Large Vision-Language Models often generate hallucinated content that is not grounded in its visual inputs. While prior work focuses on mitigating hallucinations, we instead explore leveraging hallucination correction as a training objective to improve video-language alignment. We introduce HACA, a self-training framework learning to correct hallucinations in descriptions that do not align with the video content. By identifying and correcting inconsistencies, HACA enhances the model's ability to align video and textual representations for spatio-temporal reasoning. Our experimental results show consistent gains in video-caption binding and text-to-video retrieval tasks, demonstrating that hallucination correction-inspired tasks serve as an effective strategy for improving vision and language alignment. 

**Abstract (ZH)**: 大型多模态模型经常生成与视觉输入脱节的虚构内容。尽管之前的研究所侧重于减轻这些虚构内容，我们则探索将虚构内容的校正作为训练目标，以提高视频-语言对齐。我们提出了一种自我训练框架HACA，该框架旨在纠正与视频内容不一致的描述中的虚构内容。通过识别并修正不一致性，HACA 提高了模型在时空推理方面的视频和文本表示对齐能力。我们的实验结果显示，HACA 在视频-字幕关联和文本到视频检索任务中均取得了稳健的提升，表明基于虚构内容校正的任务是一种有效的策略，能够改善视觉与语言的对齐。 

---
# Hardware-Friendly Static Quantization Method for Video Diffusion Transformers 

**Title (ZH)**: 面向硬件的静态量化方法用于视频扩散变压器 

**Authors**: Sanghyun Yi, Qingfeng Liu, Mostafa El-Khamy  

**Link**: [PDF](https://arxiv.org/pdf/2502.15077)  

**Abstract**: Diffusion Transformers for video generation have gained significant research interest since the impressive performance of SORA. Efficient deployment of such generative-AI models on GPUs has been demonstrated with dynamic quantization. However, resource-constrained devices cannot support dynamic quantization, and need static quantization of the models for their efficient deployment on AI processors. In this paper, we propose a novel method for the post-training quantization of OpenSora\cite{opensora}, a Video Diffusion Transformer, without relying on dynamic quantization techniques. Our approach employs static quantization, achieving video quality comparable to FP16 and dynamically quantized ViDiT-Q methods, as measured by CLIP, and VQA metrics. In particular, we utilize per-step calibration data to adequately provide a post-training statically quantized model for each time step, incorporating channel-wise quantization for weights and tensor-wise quantization for activations. By further applying the smooth-quantization technique, we can obtain high-quality video outputs with the statically quantized models. Extensive experimental results demonstrate that static quantization can be a viable alternative to dynamic quantization for video diffusion transformers, offering a more efficient approach without sacrificing performance. 

**Abstract (ZH)**: 自从SORA表现出显著的性能以来，用于视频生成的扩散变换器受到了广泛的研究兴趣。已经在GPU上展示了此类生成AI模型的高效部署，这得益于动态量化技术。然而，受限资源的设备无法支持动态量化，因此需要对模型进行静态量化，以便在AI处理器上高效部署。在本文中，我们提出了一种新的方法，用于OpenSora（参见文献[1]）——一个视频扩散变换器——的后训练量化，不依赖于动态量化技术。我们的方法采用静态量化，视频质量与FP16和动态量化ViDiT-Q方法相当，根据CLIP和VQA指标进行测量。特别是，我们利用每步校准数据为每个时间步提供一个适应后训练静态量化的模型，并结合了通道内量化权重和张量内量化激活。通过进一步应用平滑量化技术，可以使用静态量化模型获得高质量的视频输出。大量的实验结果表明，静态量化可以成为视频扩散变换器的一个可行替代方案，提供了一种更为高效的方法，而不牺牲性能。

[1] OpenSora: https://arxiv.org/abs/2302.13875 

---
# Rare Disease Differential Diagnosis with Large Language Models at Scale: From Abdominal Actinomycosis to Wilson's Disease 

**Title (ZH)**: 大规模语言模型在罕见病鉴别诊断中的应用：从腹腔类巴克特里亚小菌病到威尔森病 

**Authors**: Elliot Schumacher, Dhruv Naik, Anitha Kannan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15069)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in disease diagnosis. However, their effectiveness in identifying rarer diseases, which are inherently more challenging to diagnose, remains an open question. Rare disease performance is critical with the increasing use of LLMs in healthcare settings. This is especially true if a primary care physician needs to make a rarer prognosis from only a patient conversation so that they can take the appropriate next step. To that end, several clinical decision support systems are designed to support providers in rare disease identification. Yet their utility is limited due to their lack of knowledge of common disorders and difficulty of use.
In this paper, we propose RareScale to combine the knowledge LLMs with expert systems. We use jointly use an expert system and LLM to simulate rare disease chats. This data is used to train a rare disease candidate predictor model. Candidates from this smaller model are then used as additional inputs to black-box LLM to make the final differential diagnosis. Thus, RareScale allows for a balance between rare and common diagnoses. We present results on over 575 rare diseases, beginning with Abdominal Actinomycosis and ending with Wilson's Disease. Our approach significantly improves the baseline performance of black-box LLMs by over 17% in Top-5 accuracy. We also find that our candidate generation performance is high (e.g. 88.8% on gpt-4o generated chats). 

**Abstract (ZH)**: 大型语言模型（LLMs）在疾病诊断方面展现出了令人印象深刻的能力。然而，它们在识别稀有疾病方面——这些疾病本就更难诊断——的有效性仍然是一个开放的问题。随着LLMs在医疗领域的广泛应用，稀有疾病的性能变得尤为重要。尤其是在初级保健医师仅凭与患者对话就需要作出稀有诊断时，这种重要性更为突出。为此，设计了一些临床决策支持系统来辅助识别罕见疾病，但它们的实用性有限，因为它们缺乏对常见疾病的了解，且使用困难。

本文中，我们提出了一种名为RareScale的方法，旨在结合LLMs的知识和专家系统。我们使用专家系统和LLMs共同模拟稀有疾病的对话，这些数据用于训练一个稀有疾病候选预测模型。从该小型模型中得到的候选者随后作为额外输入用于黑盒LLMs，以做出最终的鉴别诊断。因此，RareScale能够在稀有和常见疾病的诊断之间取得平衡。我们在超过575种稀有疾病（从腹型放线菌病到威尔逊病）上进行了实验，结果显示，与基线性能相比，我们的方法在Top-5准确率方面提高了超过17%。此外，我们还发现，候选生成性能较高（例如，对由gpt-4生成的对话，准确率为88.8%）。 

---
# Fundamental Survey on Neuromorphic Based Audio Classification 

**Title (ZH)**: 基于神经形态的音频分类基础研究 

**Authors**: Amlan Basu, Pranav Chaudhari, Gaetano Di Caterina  

**Link**: [PDF](https://arxiv.org/pdf/2502.15056)  

**Abstract**: Audio classification is paramount in a variety of applications including surveillance, healthcare monitoring, and environmental analysis. Traditional methods frequently depend on intricate signal processing algorithms and manually crafted features, which may fall short in fully capturing the complexities of audio patterns. Neuromorphic computing, inspired by the architecture and functioning of the human brain, presents a promising alternative for audio classification tasks. This survey provides an exhaustive examination of the current state-of-the-art in neuromorphic-based audio classification. It delves into the crucial components of neuromorphic systems, such as Spiking Neural Networks (SNNs), memristors, and neuromorphic hardware platforms, highlighting their advantages in audio classification. Furthermore, the survey explores various methodologies and strategies employed in neuromorphic audio classification, including event-based processing, spike-based learning, and bio-inspired feature extraction. It examines how these approaches address the limitations of traditional audio classification methods, particularly in terms of energy efficiency, real-time processing, and robustness to environmental noise. Additionally, the paper conducts a comparative analysis of different neuromorphic audio classification models and benchmarks, evaluating their performance metrics, computational efficiency, and scalability. By providing a comprehensive guide for researchers, engineers and practitioners, this survey aims to stimulate further innovation and advancements in the evolving field of neuromorphic audio classification. 

**Abstract (ZH)**: 音频分类在监控、医疗监测和环境分析等多种应用中至关重要。传统方法通常依赖于复杂的信号处理算法和手工制作的特征，这可能无法充分捕捉音频模式的复杂性。神经形态计算，受到人类大脑架构和功能的启发，为音频分类任务提供了一个有前途的替代方案。本文综述了基于神经形态的当前音频分类技术的最新进展。它深入探讨了神经形态系统的关键组件，如脉冲神经网络（SNNs）、忆阻器和神经形态硬件平台，突显了它们在音频分类中的优势。此外，本文综述了各种在神经形态音频分类中使用的方法和策略，包括基于事件的处理、基于脉冲的学习以及生物启发的特征提取。它探讨了这些方法如何解决传统音频分类方法的局限性，特别是在能效、实时处理和环境噪声鲁棒性方面。此外，本文还对不同的神经形态音频分类模型和基准进行了比较分析，评估了它们的性能指标、计算效率和可扩展性。通过为研究者、工程师和实践者提供全面的路线图，本文旨在促进神经形态音频分类这一新兴领域的进一步创新和进步。 

---
# Reducing Hallucinations of Medical Multimodal Large Language Models with Visual Retrieval-Augmented Generation 

**Title (ZH)**: 使用视觉检索增强生成方法减少医疗多模态大型语言模型的幻觉 

**Authors**: Yun-Wei Chu, Kai Zhang, Christopher Malon, Martin Renqiang Min  

**Link**: [PDF](https://arxiv.org/pdf/2502.15040)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive performance in vision and text tasks. However, hallucination remains a major challenge, especially in fields like healthcare where details are critical. In this work, we show how MLLMs may be enhanced to support Visual RAG (V-RAG), a retrieval-augmented generation framework that incorporates both text and visual data from retrieved images. On the MIMIC-CXR chest X-ray report generation and Multicare medical image caption generation datasets, we show that Visual RAG improves the accuracy of entity probing, which asks whether a medical entities is grounded by an image. We show that the improvements extend both to frequent and rare entities, the latter of which may have less positive training data. Downstream, we apply V-RAG with entity probing to correct hallucinations and generate more clinically accurate X-ray reports, obtaining a higher RadGraph-F1 score. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉和文本任务中展现了出色的性能。然而，幻觉仍然是一个主要挑战，特别是在如医疗健康等对细节要求较高的领域。在本文中，我们展示了如何通过引入Visual RAG（视觉检索增强生成）框架来增强MLLMs，该框架结合了检索到的图像中的文本和视觉数据。我们在MIMIC-CXR胸部X光报告生成和Multicare医学图像字幕生成数据集上展示了Visual RAG如何提高实体探查的准确性，即检查医学实体是否通过图像得到支撑。我们证明了这种改进不仅适用于常见的实体，还适用于缺乏积极训练数据的罕见实体。进一步的应用中，我们使用包含实体探查的Visual RAG来纠正幻觉，生成更符合临床准确性的X光报告，并获得了更高的RadGraph-F1评分。 

---
# DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time 

**Title (ZH)**: DEFT: 可微分分叉离散弹性杆模型及其在实时模拟分叉DLOs中的应用 

**Authors**: Yizhou Chen, Xiaoyue Wu, Yeheng Zong, Anran Li, Yuzhen Chen, Julie Wu, Bohao Zhang, Ram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15037)  

**Abstract**: Autonomous wire harness assembly requires robots to manipulate complex branched cables with high precision and reliability. A key challenge in automating this process is predicting how these flexible and branched structures behave under manipulation. Without accurate predictions, it is difficult for robots to reliably plan or execute assembly operations. While existing research has made progress in modeling single-threaded Deformable Linear Objects (DLOs), extending these approaches to Branched Deformable Linear Objects (BDLOs) presents fundamental challenges. The junction points in BDLOs create complex force interactions and strain propagation patterns that cannot be adequately captured by simply connecting multiple single-DLO models. To address these challenges, this paper presents Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time (DEFT), a novel framework that combines a differentiable physics-based model with a learning framework to: 1) accurately model BDLO dynamics, including dynamic propagation at junction points and grasping in the middle of a BDLO, 2) achieve efficient computation for real-time inference, and 3) enable planning to demonstrate dexterous BDLO manipulation. A comprehensive series of real-world experiments demonstrates DEFT's efficacy in terms of accuracy, computational speed, and generalizability compared to state-of-the-art alternatives. Project page:this https URL. 

**Abstract (ZH)**: 自主线束装配需要机器人能够以高精度和可靠性操作复杂的分支电缆。自动化这一过程的关键挑战在于预测这些柔性和分支结构在操作过程中的行为。如果没有准确的预测，机器人就很难可靠地计划或执行装配操作。虽然现有研究已在建模单线形可变形线性对象（DLO）方面取得了一定进展，但将这些方法扩展到分支可变形线性对象（BDLO）面临着根本性的挑战。BDLO中的分支点会产生复杂的力量交互和应变传播模式，仅通过将多个单线形可变形线性对象模型连接起来是无法充分捕捉到这些模式的。为了解决这些挑战，本文提出了一种新颖的框架——用于实时建模分叉DLO的可微分离散分支弹性杆（Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time，简称DEFT），该框架结合了基于物理的可微分模型和学习框架：1）准确建模BDLO的动力学，包括分支点处的动力学传播和BDLO中间段的抓取；2）实现高效计算以进行实时推理；3）能够规划以实现灵活的BDLO操作。全面的实验结果表明，与当前最先进的方法相比，DEFT在准确度、计算速度和通用性方面均表现出色。项目页面：复制此链接到浏览器中查看 —— [请提供实际链接] 

---
# InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback 

**Title (ZH)**: InterFeedback：通过人类反馈揭示大规模多模态模型的互动智能 

**Authors**: Henry Hengyuan Zhao, Wenqi Pei, Yifei Tao, Haiyang Mei, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2502.15027)  

**Abstract**: Existing benchmarks do not test Large Multimodal Models (LMMs) on their interactive intelligence with human users which is vital for developing general-purpose AI assistants. We design InterFeedback, an interactive framework, which can be applied to any LMM and dataset to assess this ability autonomously. On top of this, we introduce InterFeedback-Bench which evaluates interactive intelligence using two representative datasets, MMMU-Pro and MathVerse, to test 10 different open-source LMMs. Additionally, we present InterFeedback-Human, a newly collected dataset of 120 cases designed for manually testing interactive performance in leading models such as OpenAI-o1 and Claude-3.5-Sonnet. Our evaluation results show that even state-of-the-art LMM (like OpenAI-o1) can correct their results through human feedback less than 50%. Our findings point to the need for methods that can enhance the LMMs' capability to interpret and benefit from feedback. 

**Abstract (ZH)**: 现有的基准测试并没有评估大型多模态模型（LMMs）与人类用户的互动智能，而这种能力对于开发通用人工智能辅助工具至关重要。为此，我们设计了InterFeedback，这是一种互动框架，可以应用于任何LMM和数据集，以自主评估其互动智能。在此基础上，我们引入了InterFeedback-Bench，该评测框架使用两个代表性数据集MMMU-Pro和MathVerse，对10个不同的开源LMM进行评估。此外，我们还推出了InterFeedback-Human，这是一个专门为测试领先模型（如OpenAI-o1和Claude-3.5-Sonnet）的互动性能而新收集的120个案例的数据集。我们的评估结果表明，即使是最先进的LMM（如OpenAI-o1），通过人类反馈也只有不到50%的误差可以被纠正。这些发现表明，需要发展新的方法来增强LMMs解释和利用反馈的能力。 

---
# Towards Physics-Guided Foundation Models 

**Title (ZH)**: 面向物理导向的础模型 

**Authors**: Majid Farhadloo, Arun Sharma, Mingzhou Yang, Bharat Jayaprakash, William Northrop, Shashi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15013)  

**Abstract**: Traditional foundation models are pre-trained on broad datasets to reduce the training resources (e.g., time, energy, labeled samples) needed for fine-tuning a wide range of downstream tasks. However, traditional foundation models struggle with out-of-distribution prediction and can produce outputs that are unrealistic and physically infeasible. We propose the notation of physics-guided foundation models (PGFM), that is, foundation models integrated with broad or general domain (e.g., scientific) physical knowledge applicable to a wide range of downstream tasks. 

**Abstract (ZH)**: 传统的基础模型在广泛的数据集上进行预训练，以减少对多种下游任务进行微调所需的训练资源（例如，时间、能源和标注样本）。然而，传统的基础模型在处理未见过的数据预测时存在困难，可能会生成不现实且物理上不可行的输出。我们提出了一种物理指导基础模型（PGFM）的概念，即集成了广泛或通用领域（例如，科学领域）物理知识的基础模型，这些知识适用于多种下游任务。 

---
# Graph in the Vault: Protecting Edge GNN Inference with Trusted Execution Environment 

**Title (ZH)**: Vault 中的图：使用可信执行环境保护边缘GNN推理 

**Authors**: Ruyi Ding, Tianhong Xu, Aidong Adam Ding, Yunsi Fei  

**Link**: [PDF](https://arxiv.org/pdf/2502.15012)  

**Abstract**: Wide deployment of machine learning models on edge devices has rendered the model intellectual property (IP) and data privacy vulnerable. We propose GNNVault, the first secure Graph Neural Network (GNN) deployment strategy based on Trusted Execution Environment (TEE). GNNVault follows the design of 'partition-before-training' and includes a private GNN rectifier to complement with a public backbone model. This way, both critical GNN model parameters and the private graph used during inference are protected within secure TEE compartments. Real-world implementations with Intel SGX demonstrate that GNNVault safeguards GNN inference against state-of-the-art link stealing attacks with negligible accuracy degradation (<2%). 

**Abstract (ZH)**: 将边缘设备上机器学习模型的广泛应用暴露出了模型知识产权（IP）和数据隐私的安全风险。为此，我们提出了一种基于可信执行环境（TEE）的第一种安全Graph Neural Network（GNN）部署策略——GNNVault。GNNVault采用了“分区前训练”的设计，并包含一个私有GNN纠偏器，以补充公共骨干模型。这样，关键的GNN模型参数和推理过程中使用的私有图均被保护在安全的TEE隔间内。基于Intel SGX的实际实现表明，GNNVault能够对抗最先进的链接窃取攻击，同时准确率下降几乎可以忽略（<2%）。 

---
# Obliviate: Efficient Unmemorization for Protecting Intellectual Property in Large Language Models 

**Title (ZH)**: Obliviate：保护大型语言模型中知识产权的高效遗忘机制 

**Authors**: Mark Russinovich, Ahmed Salem  

**Link**: [PDF](https://arxiv.org/pdf/2502.15010)  

**Abstract**: Recent copyright agreements between AI companies and content creators have highlighted the need for precise control over language models' ability to reproduce copyrighted content. While existing approaches rely on either complete concept removal through unlearning or simple output filtering, we propose Obliviate, a novel post-training technique that selectively prevents verbatim reproduction of specific text while preserving semantic understanding.
Obliviate operates by selecting tokens within memorized sequences and modifying the model's probability distribution to prevent exact reproduction while maintaining contextual understanding. We evaluate Obliviate on multiple large language models (LLaMA-3.1 8B, LLaMA-3.1-instruct 8B, Qwen-2.5-7B, and Yi-1.5 6B) across both synthetic memorization tasks and organic copyright content. Our results demonstrate that Obliviate achieves orders of magnitude reduction, e.g., 100x, in verbatim memorization while maintaining model performance within 1% of baseline on standard benchmarks (HellaSwag, MMLU, TruthfulQA, and Winogrande). This makes Obliviate particularly suitable for practical deployment scenarios where companies need to efficiently address copyright concerns in pretrained models without compromising their general capabilities. 

**Abstract (ZH)**: 近年来，AI公司与内容创作者之间的版权协议突显了对语言模型再现受版权保护内容的精确控制需求。现有方法依赖于通过遗忘完全删除概念或简单地过滤输出，我们提出了一种名为Obliviate的新型后训练技术，该技术能够选择性地防止特定文本的逐字再现，同时保持语义理解。

Obliviate通过在记忆序列中选择令牌并修改模型的概率分布来防止逐字再现，同时保持上下文理解。我们在多个大型语言模型（LLaMA-3.1 8B、LLaMA-3.1-instruct 8B、Qwen-2.5-7B 和 Yi-1.5 6B）上对其进行了评估，涵盖了合成记忆任务和有机版权内容。我们的结果显示，Obliviate在逐字记忆方面实现了数量级的减少，例如，100倍的减少，同时在标准基准（HellaSwag、MMLU、TruthfulQA 和 Winogrande）上的模型性能仅比基线低1%。这使Obliviate特别适用于实际部署场景，公司可以在不牺牲其一般能力的情况下，有效地解决预训练模型中的版权问题。 

---
# LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers 

**Title (ZH)**: LLM-Microscope: 探秘标点符号在Transformer上下文记忆中的隐秘作用 

**Authors**: Anton Razzhigaev, Matvey Mikhalchuk, Temurbek Rahmatullaev, Elizaveta Goncharova, Polina Druzhinina, Ivan Oseledets, Andrey Kuznetsov  

**Link**: [PDF](https://arxiv.org/pdf/2502.15007)  

**Abstract**: We introduce methods to quantify how Large Language Models (LLMs) encode and store contextual information, revealing that tokens often seen as minor (e.g., determiners, punctuation) carry surprisingly high context. Notably, removing these tokens -- especially stopwords, articles, and commas -- consistently degrades performance on MMLU and BABILong-4k, even if removing only irrelevant tokens. Our analysis also shows a strong correlation between contextualization and linearity, where linearity measures how closely the transformation from one layer's embeddings to the next can be approximated by a single linear mapping. These findings underscore the hidden importance of filler tokens in maintaining context. For further exploration, we present LLM-Microscope, an open-source toolkit that assesses token-level nonlinearity, evaluates contextual memory, visualizes intermediate layer contributions (via an adapted Logit Lens), and measures the intrinsic dimensionality of representations. This toolkit illuminates how seemingly trivial tokens can be critical for long-range understanding. 

**Abstract (ZH)**: 我们介绍了量化大型语言模型（LLMs）如何编码和存储上下文信息的方法，揭示了被视为次要的标记（例如，限定词、标点符号）实际上携带了意想不到的高上下文信息。值得注意的是，即使仅移除与其无关的标记（特别是停止词、冠词和逗号），也一致地降低了MMLU和BABILong-4k的表现。我们的分析还表明，上下文信息与连续性之间存在明显相关性，连续性衡量了从一层嵌入到下一层的变换接近单一线性映射的程度。这些发现突显了填充标记在保持上下文中的隐藏重要性。为进一步探索，我们提出了LLM-Microscope，这是一个开源工具包，用于评估标记层面的非线性、评估上下文记忆、通过调整后的Logit Lens可视化中间层贡献，并测量表示的固有维度。该工具包揭示了看似琐碎的标记在长距离理解中的关键作用。 

---
# Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions 

**Title (ZH)**: 超越地平线的安全性：基于采样的高效MPC方法配以神经控制屏障函数 

**Authors**: Ji Yin, Oswin So, Eric Yang Yu, Chuchu Fan, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2502.15006)  

**Abstract**: A common problem when using model predictive control (MPC) in practice is the satisfaction of safety specifications beyond the prediction horizon. While theoretical works have shown that safety can be guaranteed by enforcing a suitable terminal set constraint or a sufficiently long prediction horizon, these techniques are difficult to apply and thus are rarely used by practitioners, especially in the case of general nonlinear dynamics. To solve this problem, we impose a tradeoff between exact recursive feasibility, computational tractability, and applicability to ''black-box'' dynamics by learning an approximate discrete-time control barrier function and incorporating it into a variational inference MPC (VIMPC), a sampling-based MPC paradigm. To handle the resulting state constraints, we further propose a new sampling strategy that greatly reduces the variance of the estimated optimal control, improving the sample efficiency, and enabling real-time planning on a CPU. The resulting Neural Shield-VIMPC (NS-VIMPC) controller yields substantial safety improvements compared to existing sampling-based MPC controllers, even under badly designed cost functions. We validate our approach in both simulation and real-world hardware experiments. 

**Abstract (ZH)**: 在实际使用模型预测控制（MPC）时，一个常见问题是确保安全性规范不仅限于预测时间范围。尽管理论研究表明，通过施加合适的终端集约束或足够长的预测时间范围可以保证安全性，但这些技术很难实际应用，特别是在非线性动力学一般的情况下，很少被实践者采用。为了解决这个问题，我们通过学习一个近似的离散时间控制障碍函数，并将其整合到基于变分推断的MPC（VIMPC）中，来在精确递归可行性和计算可行性之间进行权衡，并使该方法适用于“黑箱”动力学。为了处理由此产生的状态约束，我们进一步提出了一种新的采样策略，该策略大大减少了最优控制估计的方差，提高了样本效率，并使基于CPU的实时规划成为可能。所得到的神经防护-VIMPC（NS-VIMPC）控制器在成本函数设计不佳的情况下，相比现有的基于采样的MPC控制器提供了显著的安全性改进。我们在模拟和实际硬件实验中验证了这种方法的有效性。 

---
# A Socratic RAG Approach to Connect Natural Language Queries on Research Topics with Knowledge Organization Systems 

**Title (ZH)**: 一种苏格拉底式检索辅助方法，用于连接研究主题的自然语言查询与知识组织系统 

**Authors**: Lew Lefton, Kexin Rong, Chinar Dankhara, Lila Ghemri, Firdous Kausar, A. Hannibal Hamdallahi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15005)  

**Abstract**: In this paper, we propose a Retrieval Augmented Generation (RAG) agent that maps natural language queries about research topics to precise, machine-interpretable semantic entities. Our approach combines RAG with Socratic dialogue to align a user's intuitive understanding of research topics with established Knowledge Organization Systems (KOSs). The proposed approach will effectively bridge "little semantics" (domain-specific KOS structures) with "big semantics" (broad bibliometric repositories), making complex academic taxonomies more accessible. Such agents have the potential for broad use. We illustrate with a sample application called CollabNext, which is a person-centric knowledge graph connecting people, organizations, and research topics. We further describe how the application design has an intentional focus on HBCUs and emerging researchers to raise visibility of people historically rendered invisible in the current science system. 

**Abstract (ZH)**: 在本文中，我们提出一种检索增强生成（RAG）代理，能够将关于研究主题的自然语言查询映射到精确且机器可解释的语义实体。我们的方法结合了RAG和苏格拉底式对话，从而将用户对研究主题的直观理解与已建立的知识组织系统（KOS）相一致。所提出的方法将有效地将“小语义”（领域特定的KOS结构）与“大语义”（广泛的文献计量repository）相连接，使得复杂的学术分类更加易于访问。此类代理有广泛的应用潜力。我们通过一个名为CollabNext的示例应用进行了说明，该应用以个人为中心的知识图谱将个人、组织和研究主题连接起来。进一步阐述了该应用设计的初衷，旨在关注HBCU（哈莱姆宝物大学）和新兴研究者，以提高历史上在当前科学系统中被忽视的人群的能见度。 

---
# A Rapid Test for Accuracy and Bias of Face Recognition Technology 

**Title (ZH)**: 面部识别技术准确性和偏差的快速测试方法 

**Authors**: Manuel Knott, Ignacio Serna, Ethan Mann, Pietro Perona  

**Link**: [PDF](https://arxiv.org/pdf/2502.14996)  

**Abstract**: Measuring the accuracy of face recognition (FR) systems is essential for improving performance and ensuring responsible use. Accuracy is typically estimated using large annotated datasets, which are costly and difficult to obtain. We propose a novel method for 1:1 face verification that benchmarks FR systems quickly and without manual annotation, starting from approximate labels (e.g., from web search results). Unlike previous methods for training set label cleaning, ours leverages the embedding representation of the models being evaluated, achieving high accuracy in smaller-sized test datasets. Our approach reliably estimates FR accuracy and ranking, significantly reducing the time and cost of manual labeling. We also introduce the first public benchmark of five FR cloud services, revealing demographic biases, particularly lower accuracy for Asian women. Our rapid test method can democratize FR testing, promoting scrutiny and responsible use of the technology. Our method is provided as a publicly accessible tool at this https URL 

**Abstract (ZH)**: 评估面部识别（FR）系统（Face Recognition, FR）的准确性对于提高性能并确保负责任的使用至关重要。准确性通常通过大型注释数据集进行估计，但这需要大量资金并难以获取。我们提出了一种新的1对1面部验证方法，该方法可以从近似标签（例如来自网络搜索结果的标签）快速且无需人工标注来评估FR系统的性能。与以前的训练集标签清理方法不同，我们的方法利用了被评估模型的嵌入表示，在较小的数据集上实现了较高的准确性。我们的方法能够可靠地估计FR的准确性和排名，显著减少了手动标注所需的时间和成本。我们还首次发布了五种云服务面部识别的公共基准测试，揭示了人口统计学偏差，特别是亚洲女性的准确性较低。我们的快速测试方法可以促进FR技术的普及化测试，促进对该技术的审查和负责任使用。该方法已作为公开可访问的工具提供于<a href="https://your-link-here">此处</a>。 

---
# Beyond No: Quantifying AI Over-Refusal and Emotional Attachment Boundaries 

**Title (ZH)**: 超越否决：量化人工智能的过度拒绝边界及其情感依附限度 

**Authors**: David Noever, Grant Rosario  

**Link**: [PDF](https://arxiv.org/pdf/2502.14975)  

**Abstract**: We present an open-source benchmark and evaluation framework for assessing emotional boundary handling in Large Language Models (LLMs). Using a dataset of 1156 prompts across six languages, we evaluated three leading LLMs (GPT-4o, Claude-3.5 Sonnet, and Mistral-large) on their ability to maintain appropriate emotional boundaries through pattern-matched response analysis. Our framework quantifies responses across seven key patterns: direct refusal, apology, explanation, deflection, acknowledgment, boundary setting, and emotional awareness. Results demonstrate significant variation in boundary-handling approaches, with Claude-3.5 achieving the highest overall score (8.69/10) and producing longer, more nuanced responses (86.51 words on average). We identified a substantial performance gap between English (average score 25.62) and non-English interactions (< 0.22), with English responses showing markedly higher refusal rates (43.20% vs. < 1% for non-English). Pattern analysis revealed model-specific strategies, such as Mistral's preference for deflection (4.2%) and consistently low empathy scores across all models (< 0.06). Limitations include potential oversimplification through pattern matching, lack of contextual understanding in response analysis, and binary classification of complex emotional responses. Future work should explore more nuanced scoring methods, expand language coverage, and investigate cultural variations in emotional boundary expectations. Our benchmark and methodology provide a foundation for systematic evaluation of LLM emotional intelligence and boundary-setting capabilities. 

**Abstract (ZH)**: 我们提出了一种开源基准和评估框架，用于评估大型语言模型（LLMs）在处理情感边界方面的表现。通过使用涵盖六种语言共计1156个提示的数据集，我们对三款领先的LLM（GPT-4o、Claude-3.5 Sonnet和Mistral-large）进行了分析，评估它们在使用模式匹配响应分析方法维持适当情感边界的能力。我们的框架量化了七个关键模式的响应：直接拒绝、道歉、解释、转移、认可、设置边界和情感意识。结果表明，在处理情感边界的方法上存在显著差异，其中Claude-3.5的整体得分为最高（8.69/10），并且生成的响应更长且更细致入微（平均86.51词）。我们发现，英语（平均得分25.62）与非英语交互之间的性能差距显著（<0.22），且英语响应中拒绝率明显更高（43.20%对比非英语的<1%）。模式分析揭示了特定于模型的策略，例如Mistral倾向于转移（4.2%），并且所有模型在情感共鸣方面的一贯得分都低于0.06。框架的局限性包括通过模式匹配可能引发的简化问题、响应分析中缺乏上下文理解、以及复杂情感响应的二元分类方法。未来的研究应该探索更精细的评分方法、扩大语言覆盖范围，并调查不同文化下对情感边界的不同期望。我们的基准和方法为系统评估LLM的情感智能和边界设定能力奠定了基础。 

---
# CyberSentinel: An Emergent Threat Detection System for AI Security 

**Title (ZH)**: 网络卫士：一种应对人工智能安全威胁的 emergent 检测系统

注释：在学术翻译中，"emergent" 在这种上下文中可能指的是“ emergent threats（新兴威胁）”之类的概念，具体含义需要根据原文进一步确定。以上翻译假设 "emergent" 是指 “新兴的” 或 “尚未完全显现的”。如果您能提供更多原文背景信息，我可以给出更准确的翻译。 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2502.14966)  

**Abstract**: The rapid advancement of artificial intelligence (AI) has significantly expanded the attack surface for AI-driven cybersecurity threats, necessitating adaptive defense strategies. This paper introduces CyberSentinel, a unified, single-agent system for emergent threat detection, designed to identify and mitigate novel security risks in real time. CyberSentinel integrates: (1) Brute-force attack detection through SSH log analysis, (2) Phishing threat assessment using domain blacklists and heuristic URL scoring, and (3) Emergent threat detection via machine learning-based anomaly detection. By continuously adapting to evolving adversarial tactics, CyberSentinel strengthens proactive cybersecurity defense, addressing critical vulnerabilities in AI security. 

**Abstract (ZH)**: 人工智能（AI）的迅速发展极大地扩展了由AI驱动的网络安全威胁的攻击面，迫切需要适应性的防御策略。本文介绍了CyberSentinel，这是一种统一的单一代理系统，旨在实时识别和减轻新型安全风险。CyberSentinel集成了以下功能：(1) 通过SSH日志分析检测暴力攻击，(2) 使用域名黑名单和启发式URL评分评估钓鱼威胁，以及(3) 通过基于机器学习的异常检测进行新兴威胁检测。通过不断适应不断演变的对手战术，CyberSentinel增强了主动的网络安全防御，缓解了AI安全中的关键漏洞。 

---
# KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding 

**Title (ZH)**: KITAB-Bench：一个全面的多领域基准测试，用于阿拉伯OCR和文档理解 

**Authors**: Ahmed Heakl, Abdullah Sohail, Mukul Ranjan, Rania Hossam, Ghazi Ahmed, Mohamed El-Geish, Omar Maher, Zhiqiang Shen, Fahad Khan, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2502.14949)  

**Abstract**: With the growing adoption of Retrieval-Augmented Generation (RAG) in document processing, robust text recognition has become increasingly critical for knowledge extraction. While OCR (Optical Character Recognition) for English and other languages benefits from large datasets and well-established benchmarks, Arabic OCR faces unique challenges due to its cursive script, right-to-left text flow, and complex typographic and calligraphic features. We present KITAB-Bench, a comprehensive Arabic OCR benchmark that fills the gaps in current evaluation systems. Our benchmark comprises 8,809 samples across 9 major domains and 36 sub-domains, encompassing diverse document types including handwritten text, structured tables, and specialized coverage of 21 chart types for business intelligence. Our findings show that modern vision-language models (such as GPT-4, Gemini, and Qwen) outperform traditional OCR approaches (like EasyOCR, PaddleOCR, and Surya) by an average of 60% in Character Error Rate (CER). Furthermore, we highlight significant limitations of current Arabic OCR models, particularly in PDF-to-Markdown conversion, where the best model Gemini-2.0-Flash achieves only 65% accuracy. This underscores the challenges in accurately recognizing Arabic text, including issues with complex fonts, numeral recognition errors, word elongation, and table structure detection. This work establishes a rigorous evaluation framework that can drive improvements in Arabic document analysis methods and bridge the performance gap with English OCR technologies. 

**Abstract (ZH)**: 随着检索增强生成（RAG）在文档处理中的应用日益广泛，稳健的文本识别对于知识提取变得越来越关键。尽管对于英文和其他语言的光学字符识别（OCR）可以从大量的数据集和成熟的基准测试中受益，但阿拉伯语的OCR（光学字符识别）面临着独特挑战，这主要归因于其连笔书写、从右至左的文本流以及复杂的字体和书法特征。我们提出了KITAB-Bench，这是一个全面的阿拉伯语OCR基准测试，填补了当前评估系统中的空白。该基准测试包含8,809个样本，涉及9个主要领域和36个子领域，涵盖了包括手写文本、结构化表格和商业智能领域21种图表在内的多种文档类型。我们的研究结果表明，现代视觉语言模型（如GPT-4、Gemini和Qwen）在字符错误率（CER）方面平均比传统OCR方法（如EasyOCR、PaddleOCR和Surya）高出60%。此外，我们还指出了现有阿拉伯语OCR模型的重要局限性，尤其是在PDF到Markdown的转换过程中，表现最好的模型Gemini-2.0-Flash的准确率仅为65%。这凸显了准确识别阿拉伯语文本所面临的挑战，包括复杂字体的问题、数字识别错误、单词延长以及表格结构检测等方面的难题。这项工作建立了一个严格的技术评估框架，可以推动阿拉伯文档分析方法的改进，并缩小与英文OCR技术之间的性能差距。 

---
# Reward-Guided Iterative Refinement in Diffusion Models at Test-Time with Applications to Protein and DNA Design 

**Title (ZH)**: 在测试时通过奖励引导的迭代细化在扩散模型中的应用：蛋白质和DNA设计 

**Authors**: Masatoshi Uehara, Xingyu Su, Yulai Zhao, Xiner Li, Aviv Regev, Shuiwang Ji, Sergey Levine, Tommaso Biancalani  

**Link**: [PDF](https://arxiv.org/pdf/2502.14944)  

**Abstract**: To fully leverage the capabilities of diffusion models, we are often interested in optimizing downstream reward functions during inference. While numerous algorithms for reward-guided generation have been recently proposed due to their significance, current approaches predominantly focus on single-shot generation, transitioning from fully noised to denoised states. We propose a novel framework for inference-time reward optimization with diffusion models inspired by evolutionary algorithms. Our approach employs an iterative refinement process consisting of two steps in each iteration: noising and reward-guided denoising. This sequential refinement allows for the gradual correction of errors introduced during reward optimization. Besides, we provide a theoretical guarantee for our framework. Finally, we demonstrate its superior empirical performance in protein and cell-type-specific regulatory DNA design. The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 为了充分发挥扩散模型的能力，我们通常在推断过程中对下游奖励函数进行优化。由于其重要性，近期提出了众多基于奖励指导生成的算法，但当前的方法主要集中在单步骤生成，即从完全噪声状态过渡到去噪状态。我们提出了一种受进化算法启发的新型框架，用于推断时的奖励优化。该方法采用迭代精炼过程，每一步迭代包括两个步骤：去噪和奖励指导去噪。这种顺序精炼允许逐步修正奖励优化过程中引入的错误。此外，我们还为该框架提供了理论保证。最后，我们在蛋白质和细胞类型特定的调控DNA设计方面展示了其优越的实证性能。代码可在 \href{this https URL}{该链接} 获取。 

---
# FacaDiffy: Inpainting Unseen Facade Parts Using Diffusion Models 

**Title (ZH)**: FacaDiffy：使用扩散模型填充未见过的建筑外墙部分 

**Authors**: Thomas Froech, Olaf Wysocki, Yan Xia, Junyu Xie, Benedikt Schwab, Daniel Cremers, Thomas H. Kolbe  

**Link**: [PDF](https://arxiv.org/pdf/2502.14940)  

**Abstract**: High-detail semantic 3D building models are frequently utilized in robotics, geoinformatics, and computer vision. One key aspect of creating such models is employing 2D conflict maps that detect openings' locations in building facades. Yet, in reality, these maps are often incomplete due to obstacles encountered during laser scanning. To address this challenge, we introduce FacaDiffy, a novel method for inpainting unseen facade parts by completing conflict maps with a personalized Stable Diffusion model. Specifically, we first propose a deterministic ray analysis approach to derive 2D conflict maps from existing 3D building models and corresponding laser scanning point clouds. Furthermore, we facilitate the inpainting of unseen facade objects into these 2D conflict maps by leveraging the potential of personalizing a Stable Diffusion model. To complement the scarcity of real-world training data, we also develop a scalable pipeline to produce synthetic conflict maps using random city model generators and annotated facade images. Extensive experiments demonstrate that FacaDiffy achieves state-of-the-art performance in conflict map completion compared to various inpainting baselines and increases the detection rate by $22\%$ when applying the completed conflict maps for high-definition 3D semantic building reconstruction. The code is be publicly available in the corresponding GitHub repository: this https URL 

**Abstract (ZH)**: 高细节语义三维建筑模型在机器人技术、地理信息学和计算机视觉中经常被使用。创建这类模型的关键之一是使用2D冲突图来检测建筑外墙上的开口位置。然而，由于激光扫描过程中遇到的障碍物，这些地图通常不完整。为了解决这一挑战，我们引入了FacaDiffy，这是一种新颖的方法，通过使用个性化的稳定扩散模型来填充看不到的外墙部分并完成冲突图。具体而言，我们首先提出了一种确定性光线分析方法，从现有的三维建筑模型和相应的激光扫描点云中推导出2D冲突图。此外，我们利用个性化稳定扩散模型的潜力来填充这些2D冲突图中的未见到的外墙对象。为了补充现实世界训练数据的匮乏，我们还开发了一个可扩展的流水线，使用随机城市模型生成器和注释过的外墙图像来生成合成冲突图。广泛实验表明，FacaDiffy在冲突图填充方面的性能优于各种现有的填充基线，并且在应用完成的冲突图进行高精度三维语义建筑重建时检测率提高了22%。代码将在相应的GitHub仓库中公开：this https URL 

---
# Online hand gesture recognition using Continual Graph Transformers 

**Title (ZH)**: 使用连续图变换器的在线手部手势识别 

**Authors**: Rim Slama, Wael Rabah, Hazem Wannous  

**Link**: [PDF](https://arxiv.org/pdf/2502.14939)  

**Abstract**: Online continuous action recognition has emerged as a critical research area due to its practical implications in real-world applications, such as human-computer interaction, healthcare, and robotics. Among various modalities, skeleton-based approaches have gained significant popularity, demonstrating their effectiveness in capturing 3D temporal data while ensuring robustness to environmental variations. However, most existing works focus on segment-based recognition, making them unsuitable for real-time, continuous recognition scenarios. In this paper, we propose a novel online recognition system designed for real-time skeleton sequence streaming. Our approach leverages a hybrid architecture combining Spatial Graph Convolutional Networks (S-GCN) for spatial feature extraction and a Transformer-based Graph Encoder (TGE) for capturing temporal dependencies across frames. Additionally, we introduce a continual learning mechanism to enhance model adaptability to evolving data distributions, ensuring robust recognition in dynamic environments. We evaluate our method on the SHREC'21 benchmark dataset, demonstrating its superior performance in online hand gesture recognition. Our approach not only achieves state-of-the-art accuracy but also significantly reduces false positive rates, making it a compelling solution for real-time applications. The proposed system can be seamlessly integrated into various domains, including human-robot collaboration and assistive technologies, where natural and intuitive interaction is crucial. 

**Abstract (ZH)**: 在线连续动作识别由于其在实际应用中的重要性，如人机交互、医疗保健和机器人技术等领域，已成为一个关键的研究领域。在各种模态中，基于骨架的方法因其在捕捉三维时空数据方面表现出的有效性及其对环境变化的鲁棒性而受到广泛关注。然而，大多数现有研究主要关注于基于片段的识别，这使它们不适合实时、连续的识别场景。本文提出了一种新颖的在线识别系统，旨在实现实时的骨架序列流式传输。我们的方法采用了混合架构，结合了空间图形卷积网络（S-GCN）进行空间特征提取和基于变换器的图形编码器（TGE）以捕捉帧间的时序依赖关系。此外，我们提出了一种持续学习机制，以增强模型对演变数据分布的适应性，从而在动态环境下提供稳健的识别能力。我们使用SHREC'21基准数据集评估了我们的方法，展示了其在在线手部动作识别中的优越性能。我们的方法不仅达到了最先进的准确率，而且显著降低了假阳性率，使其成为实时应用的有力解决方案。所提出的系统可以无缝集成到各种领域，包括人机协作和辅助技术，其中自然和直观的交互至关重要。 

---
# Fast and Accurate Blind Flexible Docking 

**Title (ZH)**: 快速且准确的盲柔性对接 

**Authors**: Zizhuo Zhang, Lijun Wu, Kaiyuan Gao, Jiangchao Yao, Tao Qin, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14934)  

**Abstract**: Molecular docking that predicts the bound structures of small molecules (ligands) to their protein targets, plays a vital role in drug discovery. However, existing docking methods often face limitations: they either overlook crucial structural changes by assuming protein rigidity or suffer from low computational efficiency due to their reliance on generative models for structure sampling. To address these challenges, we propose FABFlex, a fast and accurate regression-based multi-task learning model designed for realistic blind flexible docking scenarios, where proteins exhibit flexibility and binding pocket sites are unknown (blind). Specifically, FABFlex's architecture comprises three specialized modules working in concert: (1) A pocket prediction module that identifies potential binding sites, addressing the challenges inherent in blind docking scenarios. (2) A ligand docking module that predicts the bound (holo) structures of ligands from their unbound (apo) states. (3) A pocket docking module that forecasts the holo structures of protein pockets from their apo conformations. Notably, FABFlex incorporates an iterative update mechanism that serves as a conduit between the ligand and pocket docking modules, enabling continuous structural refinements. This approach effectively integrates the three subtasks of blind flexible docking-pocket identification, ligand conformation prediction, and protein flexibility modeling-into a unified, coherent framework. Extensive experiments on public benchmark datasets demonstrate that FABFlex not only achieves superior effectiveness in predicting accurate binding modes but also exhibits a significant speed advantage (208 $\times$) compared to existing state-of-the-art methods. Our code is released at this https URL. 

**Abstract (ZH)**: 分子对接是预测小分子（配体）与其蛋白质靶点结合结构的关键技术，在药物发现中发挥着重要作用。然而，现有对接方法往往存在局限性：它们要么假设蛋白质刚性而忽视关键的结构变化，要么由于依赖生成模型进行结构采样而导致计算效率较低。为解决这些挑战，我们提出了一种名为FABFlex的快速且准确的回归多任务学习模型，该模型专为真实盲柔性对接场景设计，其中蛋白质表现出柔性，且结合口袋位点未知。具体来说，FABFlex的架构包括三个协同工作的特殊模块：（1）一个口袋预测模块，用于识别潜在的结合位点，以应对盲柔性对接场景中的挑战；（2）一个配体对接模块，用于从配体的未结合（apo）状态预测其结合（holo）结构；（3）一个口袋对接模块，用于从蛋白质的apo构象预测其holo结构。值得注意的是，FABFlex引入了一种迭代更新机制，作为配体对接模块和口袋对接模块之间的桥梁，从而实现持续的结构精炼。该方法将盲柔性对接的三个子任务——结合位点识别、配体构象预测和蛋白质柔性建模——有效地整合到一个统一、连贯的框架中。在多个公开基准数据集上的广泛实验表明，FABFlex不仅在预测准确的结合模式方面表现出色，而且在速度上也显著优于现有最先进的方法（速度优势超过208倍）。我们的代码已发布在此处：https://github.com/your-repo-name。 

---
# A Tale of Two Structures: Do LLMs Capture the Fractal Complexity of Language? 

**Title (ZH)**: 两套结构的故事：大型语言模型能否捕获语言的分形复杂性？ 

**Authors**: Ibrahim Alabdulmohsin, Andreas Steiner  

**Link**: [PDF](https://arxiv.org/pdf/2502.14924)  

**Abstract**: Language exhibits a fractal structure in its information-theoretic complexity (i.e. bits per token), with self-similarity across scales and long-range dependence (LRD). In this work, we investigate whether large language models (LLMs) can replicate such fractal characteristics and identify conditions-such as temperature setting and prompting method-under which they may fail. Moreover, we find that the fractal parameters observed in natural language are contained within a narrow range, whereas those of LLMs' output vary widely, suggesting that fractal parameters might prove helpful in detecting a non-trivial portion of LLM-generated texts. Notably, these findings, and many others reported in this work, are robust to the choice of the architecture; e.g. Gemini 1.0 Pro, Mistral-7B and Gemma-2B. We also release a dataset comprising of over 240,000 articles generated by various LLMs (both pretrained and instruction-tuned) with different decoding temperatures and prompting methods, along with their corresponding human-generated texts. We hope that this work highlights the complex interplay between fractal properties, prompting, and statistical mimicry in LLMs, offering insights for generating, evaluating and detecting synthetic texts. 

**Abstract (ZH)**: 语言在其信息论复杂度（即每令牌位数）方面表现出分形结构，具有不同尺度上的自相似性和长程依赖性（LRD）。在这项工作中，我们探讨大型语言模型（LLMs）是否能够再现这种分形特性，并识别出它们在这种特性上的失败条件，如温度设置和提示方法等。此外，我们发现自然语言中的分形参数在狭窄的范围内，而LLMs输出的分形参数变化范围广泛，表明分形参数可能有助于检测LLM生成文本中的一部分非平凡文本。值得注意的是，这些发现以及其他本工作中报道的许多其他结果，在不同的架构选择下是稳健的；例如，Gemini 1.0 Pro、Mistral-7B和Gemma-2B。我们还发布了一个包含超过240,000篇文章的数据集，这些文章由各种LLMs（包括预训练和指令微调）以不同的解码温度和提示方法生成，并附有人类生成的相应文本。我们希望这项工作能够阐明分形特性、提示和统计模仿之间复杂相互作用之间的关系，为生成、评估和检测合成文本提供洞见。 

---
# AI Thinking as a Meaning-Centered Framework: Reimagining Language Technologies Through Community Agency 

**Title (ZH)**: 以意义为中心的框架：通过社区代理重新构想语言技术 

**Authors**: Jose F Quesada  

**Link**: [PDF](https://arxiv.org/pdf/2502.14923)  

**Abstract**: While language technologies have advanced significantly, current approaches fail to address the complex sociocultural dimensions of linguistic preservation. AI Thinking proposes a meaning-centered framework that would transform technological development from creating tools FOR communities to co-creating solutions WITH them. This approach recognizes that meaningful solutions emerge through the interplay of cultural understanding, community agency, and technological innovation. The proposal articulates a holistic methodology and a five-layer technological ecosystem where communities maintain control over their linguistic and cultural knowledge representation. This systematic integration of community needs, cultural preservation, and advanced capabilities could revolutionize how we approach linguistic diversity preservation in the digital age. 

**Abstract (ZH)**: 尽管语言技术已取得了显著进步，当前的方法仍未解决语言保护中的复杂社会文化维度问题。AI思维提出了一种以意义为中心的框架，旨在将技术发展从为社区创造工具转变为与社区共同创造解决方案。这种方法认识到，有意义的解决方案是在文化理解、社区自主权和技术创新之间的互动中产生的。该提案阐述了一种全面的方法论和五层技术生态系统，其中社区能够控制其语言和文化知识的表示。这种系统地整合社区需求、文化保护和先进技术的能力有可能在数字时代革新我们对语言多样性保护的方式。 

---
# SIFT: Grounding LLM Reasoning in Contexts via Stickers 

**Title (ZH)**: SIFT：通过贴纸在上下文中约束LLM推理

注释：在学术翻译中，我们尽量保持原文的意思和格式，但有时为了使译文更符合中文的表达习惯和学术规范，可能会进行适当的调整。这里的“贴纸”是一个形象的翻译，“Stickers”在某些语境下可被理解为辅助标记或提示，因此使用“贴纸”作为形象化翻译，以便更好地传达原文意思。 

**Authors**: Zihao Zeng, Xuyao Huang, Boxiu Li, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.14922)  

**Abstract**: This paper identifies the misinterpretation of the context can be a significant issue during the reasoning process of large language models, spanning from smaller models like Llama3.2-3B-Instruct to cutting-edge ones like DeepSeek-R1. For example, in the phrase "10 dollars per kilo," LLMs might not recognize that "per" means "for each," leading to calculation errors. We introduce a novel, post-training approach called **Stick to the Facts (SIFT)** to tackle this. SIFT leverages increasing inference-time compute to ground LLM reasoning in contexts. At the core of SIFT lies the *Sticker*, which is generated by the model itself to explicitly emphasize the key information within the context. Given the curated Sticker, SIFT generates two predictions -- one from the original query and one from the query augmented with the Sticker. If they differ, the Sticker is sequentially refined via *forward* optimization (to better align the extracted facts with the query) and *inverse* generation (to conform with the model's inherent tendencies) for more faithful reasoning outcomes. Studies across diverse models (from 3B to 100B+) and benchmarks (e.g., GSM8K, MATH-500) reveal consistent performance improvements. Notably, SIFT improves the pass@1 accuracy of DeepSeek-R1 on AIME2024 from 78.33% to **85.67**%, establishing a new state-of-the-art in the open-source community. The code is available at this https URL. 

**Abstract (ZH)**: 本文指出了在大型语言模型的推理过程中，对上下文的误解可能是显著的问题，涵盖了从较小的模型如 Llama3.2-3B-Instruct 到最新的模型如 DeepSeek-R1。例如，在短语“10 dollars per kilo”中，LLMs 可能无法理解“per”意味着“每”，导致计算错误。我们提出了一种新颖的后训练方法 **Stick to the Facts (SIFT)** 来解决这一问题。SIFT 通过增加推理时的计算量，将 LLMS 的推理根植于上下文中。SIFT 的核心是 *Sticker（标签）*，该标签由模型自身生成，明确地强调了上下文中的关键信息。基于精心生成的 Sticker，SIFT 生成两种预测——一种来自原始查询，一种来自添加了 Sticker 的查询。如果这两种预测不同，SIFT 将通过 *正向* 优化（更好地使提取的事实与查询对齐）和 *逆向* 生成（使标签与模型的固有倾向一致）逐步细化 Sticker，从而获得更忠实的推理结果。在不同规模的模型（从小于 3B 到 100B 以上）和不同基准测试（例如 GSM8K、MATH-500）上进行的研究表明，SIFT 持续提升了性能。值得注意的是，SIFT 将 DeepSeek-R1 在 AIME2024 上的 pass@1 准确率从 78.33% 提高到了 **85.67%**，在开源社区中确立了新的最佳表现。源代码可通过以下链接访问：[代码链接]。 

---
# Display Field-Of-View Agnostic Robust CT Kernel Synthesis Using Model-Based Deep Learning 

**Title (ZH)**: 基于模型的深度学习在场域视角无关的鲁棒CT核合成中的应用 

**Authors**: Hemant Kumar Aggarwal, Antony Jerald, Phaneendra K. Yalavarthy, Rajesh Langoju, Bipul Das  

**Link**: [PDF](https://arxiv.org/pdf/2502.14920)  

**Abstract**: In X-ray computed tomography (CT) imaging, the choice of reconstruction kernel is crucial as it significantly impacts the quality of clinical images. Different kernels influence spatial resolution, image noise, and contrast in various ways. Clinical applications involving lung imaging often require images reconstructed with both soft and sharp kernels. The reconstruction of images with different kernels requires raw sinogram data and storing images for all kernels increases processing time and storage requirements. The Display Field-of-View (DFOV) adds complexity to kernel synthesis, as data acquired at different DFOVs exhibit varying levels of sharpness and details. This work introduces an efficient, DFOV-agnostic solution for image-based kernel synthesis using model-based deep learning. The proposed method explicitly integrates CT kernel and DFOV characteristics into the forward model. Experimental results on clinical data, along with quantitative analysis of the estimated modulation transfer function using wire phantom data, clearly demonstrate the utility of the proposed method in real-time. Additionally, a comparative study with a direct learning network, that lacks forward model information, shows that the proposed method is more robust to DFOV variations. 

**Abstract (ZH)**: 在X射线计算机断层扫描（CT）成像中，重建核的选择至关重要，因为它显著影响临床图像的质量。不同的重建核以不同方式影响空间分辨率、图像噪声和对比度。涉及肺部成像的临床应用通常需要使用软核和锐核重建的图像。使用不同重建核重建图像需要原始sinogram数据，并存储所有核的图像会增加处理时间和存储需求。显示场域（DFOV）的引入增加了重建核合成的复杂性，因为不同DFOV下获取的数据在锐度和细节方面表现出差异性。本研究提出了一种基于模型的深度学习的高效且DFOV无关的图像重建核合成方法。所提出的方法明确将CT重建核和DFOV特性整合到前向模型中。实验结果基于临床数据，并通过对线圈模型数据的调制传递函数的定量分析，清晰地展示了所提方法在实时应用中的实用性。此外，与缺乏前向模型信息的直接学习网络进行比较研究表明，所提出的方法对DFOV变化更为稳健。 

---
# RAPTOR: Refined Approach for Product Table Object Recognition 

**Title (ZH)**: RAPTOR：改进的产品表物体识别方法 

**Authors**: Eliott Thomas, Mickael Coustaty, Aurelie Joseph, Elodie Carel, Vincent Poulain D'Andecy, Jean-Marc Ogier  

**Link**: [PDF](https://arxiv.org/pdf/2502.14918)  

**Abstract**: Extracting tables from documents is a critical task across various industries, especially on business documents like invoices and reports. Existing systems based on DEtection TRansformer (DETR) such as TAble TRansformer (TATR), offer solutions for Table Detection (TD) and Table Structure Recognition (TSR) but face challenges with diverse table formats and common errors like incorrect area detection and overlapping columns. This research introduces RAPTOR, a modular post-processing system designed to enhance state-of-the-art models for improved table extraction, particularly for product tables. RAPTOR addresses recurrent TD and TSR issues, improving both precision and structural predictions. For TD, we use DETR (trained on ICDAR 2019) and TATR (trained on PubTables-1M and FinTabNet), while TSR only relies on TATR. A Genetic Algorithm is incorporated to optimize RAPTOR's module parameters, using a private dataset of product tables to align with industrial needs. We evaluate our method on two private datasets of product tables, the public DOCILE dataset (which contains tables similar to our target product tables), and the ICDAR 2013 and ICDAR 2019 datasets. The results demonstrate that while our approach excels at product tables, it also maintains reasonable performance across diverse table formats. An ablation study further validates the contribution of each module in our system. 

**Abstract (ZH)**: 从文档中提取表格是一个跨多个行业的关键任务，尤其是在发票和报告等商业文档中。基于DEtection TRansformer（DETR）的现有系统，如Table TRansformer（TATR），为表格检测（TD）和表格结构识别（TSR）提供了解决方案，但在处理多种表格格式和常见错误（如不准确的区域检测和重叠列）时面临挑战。本研究介绍了一种模块化的后处理系统——RAPTOR，旨在增强最先进的模型，以改进表格提取，特别是产品表格。RAPTOR解决了重复出现的TD和TSR问题，提高了精度和结构预测的准确性。对于TD，我们使用了在ICDAR 2019上训练的DETR和在PubTables-1M和FinTabNet上训练的TATR；对于TSR，仅依赖于TATR。我们引入了基因算法来优化RAPTOR的模块参数，并使用一个私人产品表格数据集来更好地满足工业需求。我们在两个私人产品表格数据集（包括类似于目标产品表格的表格）、公共的DOCILE数据集（其中包含类似目标表格的表格）以及ICDAR 2013和ICDAR 2019数据集上评估了该方法。实验结果表明，尽管我们方法在产品表格方面表现出色，但在处理多种表格格式时也保持了合理的性能。进一步的消融研究验证了我们系统中每个模块的贡献。 

---
# Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive Learning 

**Title (ZH)**: Sce2DriveX：一种场景到驾驶学习的一般化多模态模型框架 

**Authors**: Rui Zhao, Qirui Yuan, Jinyu Li, Haofeng Hu, Yun Li, Chengyuan Zheng, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14917)  

**Abstract**: End-to-end autonomous driving, which directly maps raw sensor inputs to low-level vehicle controls, is an important part of Embodied AI. Despite successes in applying Multimodal Large Language Models (MLLMs) for high-level traffic scene semantic understanding, it remains challenging to effectively translate these conceptual semantics understandings into low-level motion control commands and achieve generalization and consensus in cross-scene driving. We introduce Sce2DriveX, a human-like driving chain-of-thought (CoT) reasoning MLLM framework. Sce2DriveX utilizes multimodal joint learning from local scene videos and global BEV maps to deeply understand long-range spatiotemporal relationships and road topology, enhancing its comprehensive perception and reasoning capabilities in 3D dynamic/static scenes and achieving driving generalization across scenes. Building on this, it reconstructs the implicit cognitive chain inherent in human driving, covering scene understanding, meta-action reasoning, behavior interpretation analysis, motion planning and control, thereby further bridging the gap between autonomous driving and human thought processes. To elevate model performance, we have developed the first extensive Visual Question Answering (VQA) driving instruction dataset tailored for 3D spatial understanding and long-axis task reasoning. Extensive experiments demonstrate that Sce2DriveX achieves state-of-the-art performance from scene understanding to end-to-end driving, as well as robust generalization on the CARLA Bench2Drive benchmark. 

**Abstract (ZH)**: 端到端自主驾驶，它直接将原始传感器输入映射到低级车辆控制，是具身人工智能的重要组成部分。尽管在利用多模态大型语言模型（MLLMs）进行高层次交通场景语义理解方面取得了一定的成果，但将这些概念性语义理解有效地转化为低级运动控制命令，并跨场景实现普遍性和一致性仍然颇具挑战性。我们提出了Sce2DriveX，这是一种类人的驾驶链式思考（CoT）推理MLLM框架。Sce2DriveX 利用局部场景视频和全局BEV图的多模态联合学习，深入理解长期时空关系和道路拓扑，从而增强其在3D动态/静态场景中的全面感知和推理能力，并实现了跨场景的驾驶通用性。在此基础上，它重建了人类驾驶中固有的隐式认知链，涵盖了场景理解、元行为推理、行为解释分析、运动规划和控制，从而进一步弥合了自动驾驶与人类思维过程之间的差距。为了提升模型性能，我们开发了首个针对3D空间理解和长轴任务推理定制的专业驾驶指令视觉问答（VQA）数据集。广泛的实验表明，Sce2DriveX 在从场景理解到端到端驾驶的过程中实现了最先进的性能，并在CARLA Bench2Drive基准测试中表现出强大的泛化能力。 

---
# MKE-Coder: Multi-Axial Knowledge with Evidence Verification in ICD Coding for Chinese EMRs 

**Title (ZH)**: MKE-Coder：中文EMRs中基于多轴知识及证据验证的ICD编码方法 

**Authors**: Xinxin You, Xien Liu, Xue Yang, Ziyi Wang, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14916)  

**Abstract**: The task of automatically coding the International Classification of Diseases (ICD) in the medical field has been well-established and has received much attention. Automatic coding of the ICD in the medical field has been successful in English but faces challenges when dealing with Chinese electronic medical records (EMRs). The first issue lies in the difficulty of extracting disease code-related information from Chinese EMRs, primarily due to the concise writing style and specific internal structure of the EMRs. The second problem is that previous methods have failed to leverage the disease-based multi-axial knowledge and lack of association with the corresponding clinical evidence. This paper introduces a novel framework called MKE-Coder: Multi-axial Knowledge with Evidence verification in ICD coding for Chinese EMRs. Initially, we identify candidate codes for the diagnosis and categorize each of them into knowledge under four coding this http URL, we retrieve corresponding clinical evidence from the comprehensive content of EMRs and filter credible evidence through a scoring model. Finally, to ensure the validity of the candidate code, we propose an inference module based on the masked language modeling strategy. This module verifies that all the axis knowledge associated with the candidate code is supported by evidence and provides recommendations accordingly. To evaluate the performance of our framework, we conduct experiments using a large-scale Chinese EMR dataset collected from various hospitals. The experimental results demonstrate that MKE-Coder exhibits significant superiority in the task of automatic ICD coding based on Chinese EMRs. In the practical evaluation of our method within simulated real coding scenarios, it has been demonstrated that our approach significantly aids coders in enhancing both their coding accuracy and speed. 

**Abstract (ZH)**: 医学领域自动编码国际疾病分类（ICD）的任务已经得到了广泛的研究并受到了广泛关注。在英语环境中，自动编码ICD已经取得了成功，但在处理中文电子病历（EMRs）时遇到了挑战。第一个问题在于从中文EMRs中提取疾病代码相关信息的难度较大，主要是由于EMRs简洁的书写风格和特定的内部结构。第二个问题在于，以往的方法未能充分利用基于疾病的知识多轴性，且缺乏与相应临床证据的关联。本文提出了一种名为MKE-Coder的新框架：在中文EMRs中基于知识多轴性验证证据的ICD编码框架。首先，我们识别出诊断的候选代码，并根据四个编码轴将其归类为不同的知识类别。其次，我们从EMRs的综合内容中检索相应的临床证据，并通过评分模型筛选可信的证据。最后，为了确保候选代码的有效性，我们提出了一种基于掩蔽语言模型策略的推理模块。该模块验证所有与候选代码相关的知识轴是否都得到了证据的支持，并相应地提供建议。为了评估我们框架的性能，我们使用来自各种医院的大规模中文EMR数据集进行了实验。实验结果表明，MKE-Coder在基于中文EMRs的自动ICD编码任务中表现出显著的优势。在模拟实际编码场景的实用性评估中，我们的方法显著提高了编码员的编码准确性和速度。 

---
# OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignment 

**Title (ZH)**: OpenSearch-SQL：通过动态少样本学习和一致对齐增强文本到SQL的转换 

**Authors**: Xiangjin Xie, Guangwei Xu, Lingyan Zhao, Ruijie Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14913)  

**Abstract**: Although multi-agent collaborative Large Language Models (LLMs) have achieved significant breakthroughs in the Text-to-SQL task, their performance is still constrained by various factors. These factors include the incompleteness of the framework, failure to follow instructions, and model hallucination problems. To address these problems, we propose OpenSearch-SQL, which divides the Text-to-SQL task into four main modules: Preprocessing, Extraction, Generation, and Refinement, along with an Alignment module based on a consistency alignment mechanism. This architecture aligns the inputs and outputs of agents through the Alignment module, reducing failures in instruction following and hallucination. Additionally, we designed an intermediate language called SQL-Like and optimized the structured CoT based on SQL-Like. Meanwhile, we developed a dynamic few-shot strategy in the form of self-taught Query-CoT-SQL. These methods have significantly improved the performance of LLMs in the Text-to-SQL task.
In terms of model selection, we directly applied the base LLMs without any post-training, thereby simplifying the task chain and enhancing the framework's portability. Experimental results show that OpenSearch-SQL achieves an execution accuracy(EX) of 69.3% on the BIRD development set, 72.28% on the test set, and a reward-based validity efficiency score (R-VES) of 69.36%, with all three metrics ranking first at the time of submission. These results demonstrate the comprehensive advantages of the proposed method in both effectiveness and efficiency. 

**Abstract (ZH)**: 尽管多智能体协作的大语言模型（LLMs）在文本到SQL任务上取得了显著突破，但它们的表现仍然受到多种因素的限制。这些因素包括框架的不完整性、未能遵循指令和模型的幻觉问题。为了解决这些问题，我们提出了OpenSearch-SQL，将文本到SQL任务划分为四个主要模块：预处理、抽取、生成和精修，同时基于一致性对齐机制引入了对齐模块。这种架构通过对齐模块对智能体的输入和输出进行对齐，从而减少了指令遵循失败和幻觉问题。此外，我们设计了一种中间语言SQL-Like，并基于SQL-Like优化了结构化CoT。同时，我们开发了一种以自学习Query-CoT-SQL形式呈现的动态少量示例策略。这些方法显著提高了大语言模型在文本到SQL任务上的表现。

在模型选择方面，我们直接应用了基线LLMs，在任务链简化和框架便捷性方面进行了改进，没有进行任何后续训练。实验结果表明，OpenSearch-SQL在BIRD开发集上的执行准确性(EX)为69.3%，测试集上的执行准确性为72.28%，基于奖励的有效性效率评分(R-VES)为69.36%，三项指标均在提交时排名第一。这些结果证明了所提方法在效果和效率方面的综合优势。

总结：
尽管多智能体协作的大语言模型在文本到SQL任务上取得了显著进展，但仍面临框架不完整、指令执行不准确和模型幻觉等问题。为解决这些问题，本文提出了OpenSearch-SQL，它通过将文本到SQL任务划分为预处理、抽取、生成和精修四个模块，并基于一致性对齐机制引入了对齐模块，从而提高了任务执行的准确性和效率。此外，还设计了一种类似的SQL中间语言并优化了基于该语言的结构化CoT，开发了动态少量示例策略Query-CoT-SQL。实验结果显示，OpenSearch-SQL在多项指标上表现最优，展示了其在效果和效率方面的全面优势。 

---
# Batayan: A Filipino NLP benchmark for evaluating Large Language Models 

**Title (ZH)**: Batayan：一种用于评估大型语言模型的菲律宾自然语言处理基准 

**Authors**: Jann Railey Montalan, Jimson Paulo Layacan, David Demitri Africa, Richell Isaiah Flores, Michael T. Lopez II, Theresa Denise Magsajo, Anjanette Cayabyab, William Chandra Tjhi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14911)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities on widely benchmarked high-resource languages; however, linguistic nuances of under-resourced languages remain unexplored. We introduce Batayan, a holistic Filipino benchmark designed to systematically evaluate LLMs across three key natural language processing (NLP) competencies: understanding, reasoning, and generation. Batayan consolidates eight tasks, covering both Tagalog and code-switched Taglish utterances. Our rigorous, native-speaker-driven annotation process ensures fluency and authenticity to the complex morphological and syntactic structures of Filipino, alleviating a pervasive translationese bias in existing Filipino corpora. We report empirical results on a variety of multilingual LLMs, highlighting significant performance gaps that signal the under-representation of Filipino in pretraining corpora, the unique hurdles in modeling Filipino's rich morphology and construction, and the importance of explicit Filipino language support and instruction tuning. Moreover, we discuss the practical challenges encountered in dataset construction and propose principled solutions for building culturally and linguistically-faithful resources in under-represented languages. We also provide a public benchmark and leaderboard as a clear foundation for iterative, community-driven progress in Filipino NLP. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在广泛基准的高资源语言上表现出显著的能力，然而对于资源不足语言的语言细微差别仍需进一步探索。我们介绍了Batayan，一个全面的菲律宾语基准，旨在系统地评估LLMs在自然语言处理（NLP）三大核心能力——理解、推理和生成——上的表现。Batayan整合了八个任务，涵盖了塔加洛语及其代码混合体塔加蓝格（Taglish）的表达。我们严格的、以母语者为导向的标注过程确保了对复杂形态和句法结构的流畅性和真实性，减少了现有菲律宾语语料库普遍存在的翻译痕迹偏见。我们报告了多语言LLMs的多种实证结果，突显了菲律宾语在预训练语料库中的代表性不足、对菲律宾语丰富形态和构造建模的独特挑战以及明确支持菲律宾语文本和指令调优的重要性。此外，我们讨论了数据集构建中遇到的实际挑战，并提出了构建文化上和语言上符合相关的资源的原则性解决方案，以更好地服务资源不足的语言。我们还提供了一个公开的基准和排行榜，作为菲律宾语NLP迭代、社区驱动进步的清晰基础。 

---
# EvoP: Robust LLM Inference via Evolutionary Pruning 

**Title (ZH)**: EvoP：通过进化裁剪实现的 robust 大型语言模型推理 

**Authors**: Shangyu Wu, Hongchao Du, Ying Xiong, Shuai Chen, Tei-wei Kuo, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.14910)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in natural language processing tasks, but their massive size and computational demands hinder their deployment in resource-constrained environments. Existing structured pruning methods address this issue by removing redundant structures (e.g., elements, channels, layers) from the model. However, these methods employ a heuristic pruning strategy, which leads to suboptimal performance. Besides, they also ignore the data characteristics when pruning the model.
To overcome these limitations, we propose EvoP, an evolutionary pruning framework for robust LLM inference. EvoP first presents a cluster-based calibration dataset sampling (CCDS) strategy for creating a more diverse calibration dataset. EvoP then introduces an evolutionary pruning pattern searching (EPPS) method to find the optimal pruning pattern. Compared to existing structured pruning techniques, EvoP achieves the best performance while maintaining the best efficiency. Experiments across different LLMs and different downstream tasks validate the effectiveness of the proposed EvoP, making it a practical and scalable solution for deploying LLMs in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理任务中取得了显著的成功，但其庞大的规模和对计算资源的需求限制了它们在资源受限环境下的部署。现有的结构化剪枝方法通过从模型中移除冗余结构（例如，元素、通道、层）来解决这一问题。然而，这些方法采用了一种启发式剪枝策略，导致性能不佳。此外，这些方法在剪枝模型时忽视了数据特性。

为克服这些限制，我们提出了EvoP框架，这是一种用于鲁棒LLM推理的进化剪枝框架。EvoP首先提供了一种基于聚类的校准数据集采样策略（CCDS），以创建更具多样性的校准数据集。然后，EvoP引入了一种进化剪枝模式搜索（EPPS）方法来寻找最优的剪枝模式。与现有的结构化剪枝技术相比，EvoP在保持最佳效率的同时取得了最佳性能。在不同LLM和不同的下游任务上的实验验证了所提出的EvoP的有效性，使其成为一个实用且可扩展的解决方案，适用于在实际应用中部署LLM。 

---
# PTB-Image: A Scanned Paper ECG Dataset for Digitization and Image-based Diagnosis 

**Title (ZH)**: PTB-Image：一种用于数字化和图像诊断的扫描纸质心电图数据集 

**Authors**: Cuong V. Nguyen, Hieu X. Nguyen, Dung D. Pham Minh, Cuong D. Do  

**Link**: [PDF](https://arxiv.org/pdf/2502.14909)  

**Abstract**: Electrocardiograms (ECGs) recorded on paper remain prevalent in clinical practice, yet their use presents challenges for automated analysis and digital storage. To address this issue, we introduce PTB-Image, a dataset comprising scanned paper ECGs with corresponding digital signals, enabling research on ECG digitization. We also provide VinDigitizer, a digitization baseline to convert paper-based ECGs into digital time-series signals. The method involves detecting signal rows, extracting waveforms from the background, and reconstructing numerical values from the digitized traces. We applied VinDigitizer to 549 scanned ECGs and evaluated its performance against the original PTB dataset (modified to match the printed signals). The results achieved a mean signal-to-noise ratio (SNR) of 0.01 dB, highlighting both the feasibility and challenges of ECG digitization, particularly in mitigating distortions from printing and scanning processes. By providing PTB-Image and baseline digitization methods, this work aims to facilitate advancements in ECG digitization, enhancing access to historical ECG data and supporting applications in telemedicine and automated cardiac diagnostics. 

**Abstract (ZH)**: 纸质记录的心电图（ECGs）在临床实践中仍非常普遍，但其使用为自动分析和数字存储带来了挑战。为解决这一问题，我们介绍了PTB-Image数据集，该数据集包括扫描的纸质心电图和相应的数字信号，以促进心电图数字化的研究。我们还提供了VinDigitizer，这是一种用于将纸质心电图转换为数字时间序列信号的基本方法。该方法涉及检测信号行、从背景中提取波形，并从数字化迹线中重建数值。我们将VinDigitizer应用于549张扫描的心电图，并将其性能与修改后的原始PTB数据集进行了比较（该数据集根据印刷信号进行了调整）。结果表明，信噪比（SNR）的平均值仅为0.01 dB，这一结果突显了心电图数字化的可行性和挑战，特别是在减轻打印和扫描过程中造成的失真方面的挑战。通过提供PTB-Image和基本的数字化方法，本项工作旨在促进心电图数字化的发展，增强对历史心电图数据的访问，并支持远程医学和自动心脏诊断的应用。 

---
# KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models 

**Title (ZH)**: KOALA：知识冲突增强以提高视觉语言模型的鲁棒性 

**Authors**: Peter Carragher, Nikitha Rao, Abhinand Jha, R Raghav, Kathleen M. Carley  

**Link**: [PDF](https://arxiv.org/pdf/2502.14908)  

**Abstract**: The robustness of large language models (LLMs) against knowledge conflicts in unimodal question answering systems has been well studied. However, the effect of conflicts in information sources on vision language models (VLMs) in multimodal settings has not yet been explored. In this work, we propose \segsub, a framework that applies targeted perturbations to image sources to study and improve the robustness of VLMs against three different types of knowledge conflicts, namely parametric, source, and counterfactual conflicts. Contrary to prior findings that showed that LLMs are sensitive to parametric conflicts arising from textual perturbations, we find VLMs are largely robust to image perturbation. On the other hand, VLMs perform poorly on counterfactual examples (<30% accuracy) and fail to reason over source conflicts (<1% accuracy). We also find a link between hallucinations and image context, with GPT-4o prone to hallucination when presented with highly contextualized counterfactual examples. While challenges persist with source conflicts, finetuning models significantly improves reasoning over counterfactual samples. Our findings highlight the need for VLM training methodologies that enhance their reasoning capabilities, particularly in addressing complex knowledge conflicts between multimodal sources. 

**Abstract (ZH)**: 大型语言模型（LLMs）在单模态问答系统中对抗知识冲突的鲁棒性已有广泛研究，然而，在多模态设置中，信息来源之间的冲突如何影响视觉语言模型（VLMs）的抗冲突能力尚未得到探索。在这项工作中，我们提出了\segsub框架，该框架通过针对图像来源施加目标扰动来研究和提高VLMs在三种不同类型的知识冲突——参数冲突、来源冲突和反事实冲突——下的鲁棒性。与先前发现LLMs对由于文本扰动产生的参数冲突敏感相反，我们发现VLMs对图像扰动具有很大的鲁棒性。另一方面，VLMs在反事实示例上的表现很差（准确率<30%），并且无法处理来源冲突（准确率<1%）。我们还发现幻觉与图像上下文之间存在联系，GPT-4o在面对高度上下文化的反事实示例时容易产生幻觉。尽管在来源冲突方面仍存在挑战，但微调模型可以显著提高对反事实样本的推理能力。我们的研究结果强调了需要改进VLM训练方法，特别是在处理复杂的多模态来源之间的知识冲突方面加强其推理能力。 

---
# GneissWeb: Preparing High Quality Data for LLMs at Scale 

**Title (ZH)**: GneissWeb：为大规模语言模型准备高品质数据 

**Authors**: Hajar Emami Gohari, Swanand Ravindra Kadhe, Syed Yousaf Shah. Constantin Adam, Abdulhamid Adebayo, Praneet Adusumilli, Farhan Ahmed, Nathalie Baracaldo Angel, Santosh Borse, Yuan-Chi Chang, Xuan-Hong Dang, Nirmit Desai, Ravital Eres, Ran Iwamoto, Alexei Karve, Yan Koyfman, Wei-Han Lee, Changchang Liu, Boris Lublinsky, Takuyo Ohko, Pablo Pesce, Maroun Touma, Shiqiang Wang, Shalisha Witherspoon, Herbert Woisetschlager, David Wood, Kun-Lung Wu, Issei Yoshida, Syed Zawad, Petros Zerfos, Yi Zhou, Bishwaranjan Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2502.14907)  

**Abstract**: Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. Large pre-training datasets for leading LLMs remain inaccessible to the public, whereas many open datasets are small in size (less than 5 trillion tokens), limiting their suitability for training large models.
In this paper, we introduce GneissWeb, a large dataset yielding around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. GneissWeb achieves a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens).
We show that models trained using GneissWeb dataset outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average score computed on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points advantage over those trained on FineWeb-V1.1.0. 

**Abstract (ZH)**: 数据的数量和质量在决定大型语言模型（LLM）性能方面起着至关重要的作用。特别是高质量的数据可以显著提升LLM在多种下游任务上的泛化能力。目前，领先LLM的巨大预训练数据集对公众来说仍不具备访问性，而许多开源数据集规模较小（少于5万亿个令牌），限制了其用于训练大规模模型的适用性。

本文介绍了一种名为GneissWeb的大规模数据集，其生成的数据量约为10万亿个令牌，能够满足训练LLM的数据质量和数量要求。GneissWeb的数据集生成方法采用了分布式的精确子字符串去重和精心构建的质量筛选器集合。GneissWeb在保持数据质量和数量之间取得良好平衡的同时，生成的模型性能优于使用最先进的开源大规模数据集（5+万亿个令牌）训练的模型。

我们显示，使用GneissWeb数据集训练的模型在11个常用基准测试（包括零样本和少量样本）上的平均得分上比使用FineWeb-V1.1.0训练的模型高出2.73个百分点。当评估集扩展到20个基准测试（包括零样本和少量样本）时，使用GneissWeb训练的模型仍然比使用FineWeb-V1.1.0训练的模型高出1.75个百分点。 

---
# Beyond Words: Exploring Cultural Value Sensitivity in Multimodal Models 

**Title (ZH)**: 超越文字：探索多模态模型中的文化价值敏感性 

**Authors**: Srishti Yadav, Zhi Zhang, Daniel Hershcovich, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2502.14906)  

**Abstract**: Investigating value alignment in Large Language Models (LLMs) based on cultural context has become a critical area of research. However, similar biases have not been extensively explored in large vision-language models (VLMs). As the scale of multimodal models continues to grow, it becomes increasingly important to assess whether images can serve as reliable proxies for culture and how these values are embedded through the integration of both visual and textual data. In this paper, we conduct a thorough evaluation of multimodal model at different scales, focusing on their alignment with cultural values. Our findings reveal that, much like LLMs, VLMs exhibit sensitivity to cultural values, but their performance in aligning with these values is highly context-dependent. While VLMs show potential in improving value understanding through the use of images, this alignment varies significantly across contexts highlighting the complexities and underexplored challenges in the alignment of multimodal models. 

**Abstract (ZH)**: 基于文化背景探究大型语言模型（LLMs）的价值对齐已成为一个重要研究领域。然而，类似偏见在大规模视觉-语言模型（VLMs）中的研究尚未得到充分探讨。随着多模态模型规模的不断扩大，评估图像是否能作为文化可靠的代理，以及这些价值观如何通过视觉和文本数据的整合而被嵌入，变得越来越重要。在本文中，我们对不同规模的多模态模型进行了全面评估，重点关注它们与文化价值观的对齐情况。研究发现，与LLMs类似，VLMs对文化价值观表现出敏感性，但它们在价值观对齐方面的性能高度依赖于上下文。虽然VLMs通过使用图像在提升价值理解方面显示出潜力，但在不同上下文中的这种对齐差异显著，突显了多模态模型对齐中的复杂性和未被充分探讨的挑战。 

---
# Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence 

**Title (ZH)**: 将 JSON 中的内容填满：严格遵守语言模型架构的强化策略 

**Authors**: Bhavik Agarwal, Ishan Joshi, Viktoria Rojkova  

**Link**: [PDF](https://arxiv.org/pdf/2502.14905)  

**Abstract**: In this paper, we address the challenge of enforcing strict schema adherence in large language model (LLM) generation by leveraging LLM reasoning capabilities. Building on the DeepSeek R1 reinforcement learning framework, our approach trains structured reasoning skills of a 1.5B parameter model through a novel pipeline that combines synthetic reasoning dataset construction with custom reward functions under Group Relative Policy Optimization (GRPO). Specifically, we first perform R1 reinforcement learning on a 20K sample unstructured-to-structured dataset, mirroring the original DeepSeek R1 methods, to establish core reasoning abilities. Subsequently, we performed supervised fine-tuning on a separate 10K reasoning sample dataset, focusing on refining schema adherence for downstream tasks. Despite the relatively modest training scope, requiring approximately 20 hours on an 8xH100 GPU cluster for GRPO training and 3 hours on 1xA100 for SFT, our model demonstrates robust performance in enforcing schema consistency. We compare our ThinkJSON approach against the original DeepSeek R1 (671B), distilled versions of DeepSeek R1 (Qwen-1.5B and Qwen-7B), and Gemini 2.0 Flash (70B), showcasing its effectiveness in real-world applications. Our results underscore the practical utility of a resource-efficient framework for schema-constrained text generation. 

**Abstract (ZH)**: 在本文中，我们通过利用大型语言模型（LLM）的推理能力来应对在LLM生成中严格遵守模式规范的挑战。基于DeepSeek R1强化学习框架，我们的方法通过一种结合合成推理数据集构建和特定奖励函数的新颖管道，对一个包含1.5亿参数的模型进行结构化推理技能的训练。该方法是在Group Relative Policy Optimization（GRPO）下实现的。具体来说，我们首先在20,000个样本的无结构到结构化的数据集上进行R1强化学习，以重现原始的DeepSeek R1方法，从而建立核心的推理能力。随后，我们在一个独立的10,000个推理样本数据集上进行监督微调，重点关注细化模式遵守能力以适应下游任务。尽管我们的训练范围相对较小，GRPO训练大约需要8个H100 GPU集群的20小时，SFT训练则需要1个A100 GPU大约3小时，我们的模型仍然在实现数据模式一致性方面表现出稳健的性能。我们还将我们的ThinkJSON方法与原始的DeepSeek R1（671B）、DeepSeek R1的精简版本（Qwen-1.5B和Qwen-7B）以及Gemini 2.0 Flash（70B）进行了比较，展示了其在实际应用中的有效性。我们的结果强调了一种资源高效框架在模式约束式文本生成中的实用价值。 

---
# PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths 

**Title (ZH)**: PathRAG：基于关系路径的图检索增强生成精简方法 

**Authors**: Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, Cheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14902)  

**Abstract**: Retrieval-augmented generation (RAG) improves the response quality of large language models (LLMs) by retrieving knowledge from external databases. Typical RAG approaches split the text database into chunks, organizing them in a flat structure for efficient searches. To better capture the inherent dependencies and structured relationships across the text database, researchers propose to organize textual information into an indexing graph, known asgraph-based RAG. However, we argue that the limitation of current graph-based RAG methods lies in the redundancy of the retrieved information, rather than its insufficiency. Moreover, previous methods use a flat structure to organize retrieved information within the prompts, leading to suboptimal performance. To overcome these limitations, we propose PathRAG, which retrieves key relational paths from the indexing graph, and converts these paths into textual form for prompting LLMs. Specifically, PathRAG effectively reduces redundant information with flow-based pruning, while guiding LLMs to generate more logical and coherent responses with path-based prompting. Experimental results show that PathRAG consistently outperforms state-of-the-art baselines across six datasets and five evaluation dimensions. The code is available at the following link: this https URL 

**Abstract (ZH)**: Retrieval-augmented生成（RAG）通过从外部数据库检索知识，提高了大型语言模型（LLMs）的回复质量。典型的RAG方法将文本数据库拆分为片段，并组织成扁平结构以进行高效搜索。为了更好地捕捉文本数据库中固有的依赖关系和结构化关系，研究人员提出将文本信息组织成索引图，称为图基RAG。然而，我们认为当前图基RAG方法的主要限制在于检索信息的冗余性，而不是其不足。此外，以往的方法使用扁平结构在提示中组织检索信息，导致性能不佳。为克服这些限制，我们提出了PathRAG，该方法从索引图中检索关键关系路径，并将这些路径转换为文本形式，用于提示LLMs。具体来说，PathRAG通过基于流的剪枝有效减少了冗余信息，同时通过路径驱动的提示引导LLMs生成更逻辑性和连贯性的回复。实验结果显示，在六个数据集和五个评估维度上，PathRAG始终优于当前最先进的基线方法。相关代码可在以下链接获取：this https URL 

---
# Can AI mimic the human ability to define neologisms? 

**Title (ZH)**: AI能否模仿人类定义新词的能力？ 

**Authors**: Georgios P. Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14900)  

**Abstract**: One ongoing debate in linguistics is whether Artificial Intelligence (AI) can effectively mimic human performance in language-related tasks. While much research has focused on various linguistic abilities of AI, little attention has been given to how it defines neologisms formed through different word formation processes. This study addresses this gap by examining the degree of agreement between human and AI-generated responses in defining three types of Greek neologisms: blends, compounds, and derivatives. The study employed an online experiment in which human participants selected the most appropriate definitions for neologisms, while ChatGPT received identical prompts. The results revealed fair agreement between human and AI responses for blends and derivatives but no agreement for compounds. However, when considering the majority response among humans, agreement with AI was high for blends and derivatives. These findings highlight the complexity of human language and the challenges AI still faces in capturing its nuances. In particular, they suggest a need for integrating more advanced semantic networks and contextual learning mechanisms into AI models to improve their interpretation of complex word formations, especially compounds. 

**Abstract (ZH)**: Linguistics领域的一个持续争论是，人工智能（AI）是否能够在语言相关任务中有效地模拟人类的表现。尽管已有大量研究关注了AI的各种语言能力，但很少有人关注它在通过不同词汇构成过程形成的词汇创新（词根词、复合词和派生词）中是如何定义这些新词的。本研究通过考察人类和AI生成的响应在定义三种类型的希腊新词（词根词、复合词和派生词）时的一致性程度，填补了这一空白。研究采用了在线实验的方式，要求人类参与者选择最合适的定义，而ChatGPT则收到了相同的任务提示。结果表明，在词根词和派生词的定义上，人类和AI的回答之间存在相当的协同；而在复合词的定义上，两者之间几乎没有任何一致性。然而，在考虑到人类回答中最常见的选项时，与AI的回答相比，词根词和派生词的回答一致性较高。这些发现突显了人类语言的复杂性及AI在捕捉其细微之处时所面临的挑战。特别地，这些发现表明需要将更高级的语义网络和语境学习机制整合到AI模型中，以提高它们对复杂词汇构成形式的理解能力，尤其是复合词。 

---
# UPCMR: A Universal Prompt-guided Model for Random Sampling Cardiac MRI Reconstruction 

**Title (ZH)**: UPCMR：一种通用提示引导模型用于随机采样心脏MRI重建 

**Authors**: Donghang Lyu, Chinmay Rao, Marius Staring, Matthias J.P. van Osch, Mariya Doneva, Hildo J. Lamb, Nicola Pezzotti  

**Link**: [PDF](https://arxiv.org/pdf/2502.14899)  

**Abstract**: Cardiac magnetic resonance imaging (CMR) is vital for diagnosing heart diseases, but long scan time remains a major drawback. To address this, accelerated imaging techniques have been introduced by undersampling k-space, which reduces the quality of the resulting images. Recent deep learning advancements aim to speed up scanning while preserving quality, but adapting to various sampling modes and undersampling factors remains challenging. Therefore, building a universal model is a promising direction. In this work, we introduce UPCMR, a universal unrolled model designed for CMR reconstruction. This model incorporates two kinds of learnable prompts, undersampling-specific prompt and spatial-specific prompt, and integrates them with a UNet structure in each block. Overall, by using the CMRxRecon2024 challenge dataset for training and validation, the UPCMR model highly enhances reconstructed image quality across all random sampling scenarios through an effective training strategy compared to some traditional methods, demonstrating strong adaptability potential for this task. 

**Abstract (ZH)**: 心脏磁共振成像（CMR）对于诊断心脏疾病至关重要，但长时间的扫描仍然是一个主要问题。为了解决这一问题，通过欠采样k空间引入了加速成像技术，这降低了所生成图像的质量。最近的深度学习进步旨在提高扫描速度同时保持图像质量，但适应各种采样模式和欠采样因子仍然具有挑战性。因此，建立一个通用模型是一个有前景的方向。在本研究中，我们提出了UPCMR，这是一种专为CMR重建设计的通用展开模型。该模型结合了两种可学习的提示，即欠采样特定提示和空间特定提示，并将它们与每个模块中的UNet结构相结合。总体而言，通过使用CMRxRecon2024挑战数据集进行训练和验证，UPCMR模型通过有效的训练策略在所有随机采样场景中显著提高了重建图像的质量，显示出在该任务中的强大适应性潜力。 

---
# Retrieval-augmented systems can be dangerous medical communicators 

**Title (ZH)**: 检索增强系统可能是危险的医疗通信工具 

**Authors**: Lionel Wong, Ayman Ali, Raymond Xiong, Shannon Zeijang Shen, Yoon Kim, Monica Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2502.14898)  

**Abstract**: Patients have long sought health information online, and increasingly, they are turning to generative AI to answer their health-related queries. Given the high stakes of the medical domain, techniques like retrieval-augmented generation and citation grounding have been widely promoted as methods to reduce hallucinations and improve the accuracy of AI-generated responses and have been widely adopted into search engines. This paper argues that even when these methods produce literally accurate content drawn from source documents sans hallucinations, they can still be highly misleading. Patients may derive significantly different interpretations from AI-generated outputs than they would from reading the original source material, let alone consulting a knowledgeable clinician. Through a large-scale query analysis on topics including disputed diagnoses and procedure safety, we support our argument with quantitative and qualitative evidence of the suboptimal answers resulting from current systems. In particular, we highlight how these models tend to decontextualize facts, omit critical relevant sources, and reinforce patient misconceptions or biases. We propose a series of recommendations -- such as the incorporation of communication pragmatics and enhanced comprehension of source documents -- that could help mitigate these issues and extend beyond the medical domain. 

**Abstract (ZH)**: 患者长期以来一直在网上寻求健康信息，随着技术的发展，他们现在越来越多地转向生成式AI来回答健康相关的查询。鉴于医学领域的高度风险性，检索增强生成和引文定位等技术已被广泛推广，作为减少幻觉并提高AI生成回复准确性的方法，并已被广泛应用于搜索引擎中。本文认为，即使这些方法能够生成与原始文档字面准确且无幻觉的内容，它们仍然可能非常误导。患者从AI生成的输出中得出的解释可能会与阅读原始资料或咨询有知识的临床医生时得出的理解大不相同。通过大规模查询分析，包括争议性诊断和手术安全性等话题，我们通过定量和定性证据支持了我们的观点，证明了当前系统产生的次优答案。特别地，我们指出这些模型倾向于脱离上下文陈述事实、遗漏关键相关资料，并强化患者的误解或偏见。我们提出了若干建议——例如，融入交际语用学和增强对源文档的理解——这些措施可以帮助缓解这些问题，并超越医学领域。 

---
# A Comprehensive Survey on Concept Erasure in Text-to-Image Diffusion Models 

**Title (ZH)**: 文本到图像扩散模型中概念抹除综述 

**Authors**: Changhoon Kim, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14896)  

**Abstract**: Text-to-Image (T2I) models have made remarkable progress in generating high-quality, diverse visual content from natural language prompts. However, their ability to reproduce copyrighted styles, sensitive imagery, and harmful content raises significant ethical and legal concerns. Concept erasure offers a proactive alternative to external filtering by modifying T2I models to prevent the generation of undesired content. In this survey, we provide a structured overview of concept erasure, categorizing existing methods based on their optimization strategies and the architectural components they modify. We categorize concept erasure methods into fine-tuning for parameter updates, closed-form solutions for efficient edits, and inference-time interventions for content restriction without weight modification. Additionally, we explore adversarial attacks that bypass erasure techniques and discuss emerging defenses. To support further research, we consolidate key datasets, evaluation metrics, and benchmarks for assessing erasure effectiveness and model robustness. This survey serves as a comprehensive resource, offering insights into the evolving landscape of concept erasure, its challenges, and future directions. 

**Abstract (ZH)**: 文本到图像（T2I）模型在从自然语言提示生成高质量和多样化视觉内容方面取得了显著进展。然而，它们再现受版权保护的风格、敏感图像和有害内容的能力引发了重大的伦理和法律关注。概念擦除提供了一种主动替代外部过滤的选项，通过修改T2I模型来防止生成不必要的内容。在本文综述中，我们提供了概念擦除的结构化概述，根据现有的优化策略和修改的架构组件对现有方法进行了分类。我们将概念擦除方法分为通过参数更新的微调、高效编辑的闭式解和在权重修改情况下进行的内容限制的推理时干预。此外，我们探讨了绕过擦除技术的对抗性攻击，并讨论了新兴的防御方法。为了进一步研究，我们汇总了关键的数据集、评估指标和基准，用于评估擦除的有效性和模型的稳健性。本文综述提供了一个全面的资源，为概念擦除的发展景观、挑战和未来方向提供见解。 

---
# FOCUS on Contamination: A Geospatial Deep Learning Framework with a Noise-Aware Loss for Surface Water PFAS Prediction 

**Title (ZH)**: 聚焦污染：一种具有噪声意识损失函数的地理空间深度学习框架，用于地表水PFAS预测 

**Authors**: Jowaria Khan, Alexa Friedman, Sydney Evans, Runzi Wang, Kaley Beins, David Andrews, Elizabeth Bondi-Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2502.14894)  

**Abstract**: Per and polyfluoroalkyl substances (PFAS), chemicals found in products like non-stick cookware, are unfortunately persistent environmental pollutants with severe health risks. Accurately mapping PFAS contamination is crucial for guiding targeted remediation efforts and protecting public and environmental health, yet detection across large regions remains challenging due to the cost of testing and the difficulty of simulating their spread. In this work, we introduce FOCUS, a geospatial deep learning framework with a label noise-aware loss function, to predict PFAS contamination in surface water over large regions. By integrating hydrological flow data, land cover information, and proximity to known PFAS sources, our approach leverages both spatial and environmental context to improve prediction accuracy. We evaluate the performance of our approach through extensive ablation studies and comparative analyses against baselines like sparse segmentation, as well as existing scientific methods, including Kriging and pollutant transport simulations. Results highlight our framework's potential for scalable PFAS monitoring. 

**Abstract (ZH)**: 含全氟和多氟烷基物质（PFAS）的化学物质存在于不粘锅等产品中，是持久性环境污染物，对人体健康构成严重风险。准确地mapping PFAS污染对于指导有针对性的修复努力并保护公共和环境健康至关重要，但由于测试成本高昂和模拟其传播难度大，对大区域范围的检测仍然具有挑战性。本文介绍了一种具有标签噪声感知损失函数的地学深度学习框架FOCUS，用于预测大面积地区地表水中的PFAS污染。通过整合水文流数据、土地覆盖信息以及与已知PFAS来源的接近程度，我们的方法利用空间和环境上下文来提高预测准确性。我们通过广泛的消融研究和与稀疏分割、以及现有科学方法（如克里格法和污染物传输模拟）的对比分析，评估了我们方法的性能。实验结果突显了本框架在PFAS监测方面的规模化应用潜力。 

---
# NOTA: Multimodal Music Notation Understanding for Visual Large Language Model 

**Title (ZH)**: 注意：面向视觉大型语言模型的多模态音乐符号理解 

**Authors**: Mingni Tang, Jiajia Li, Lu Yang, Zhiqiang Zhang, Jinghao Tian, Zuchao Li, Lefei Zhang, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14893)  

**Abstract**: Symbolic music is represented in two distinct forms: two-dimensional, visually intuitive score images, and one-dimensional, standardized text annotation sequences. While large language models have shown extraordinary potential in music, current research has primarily focused on unimodal symbol sequence text. Existing general-domain visual language models still lack the ability of music notation understanding. Recognizing this gap, we propose NOTA, the first large-scale comprehensive multimodal music notation dataset. It consists of 1,019,237 records, from 3 regions of the world, and contains 3 tasks. Based on the dataset, we trained NotaGPT, a music notation visual large language model. Specifically, we involve a pre-alignment training phase for cross-modal alignment between the musical notes depicted in music score images and their textual representation in ABC notation. Subsequent training phases focus on foundational music information extraction, followed by training on music notation analysis. Experimental results demonstrate that our NotaGPT-7B achieves significant improvement on music understanding, showcasing the effectiveness of NOTA and the training pipeline. Our datasets are open-sourced at this https URL. 

**Abstract (ZH)**: 符号音乐以两种不同的形式表示：二维、直观的乐谱图和一维、标准化的文本注释序列。虽然大型语言模型在音乐方面展现出了巨大的潜力，但当前研究主要集中在单一模态的符号序列文本上。现有的通用领域视觉语言模型在音乐记谱理解方面仍存在不足。鉴于这一空白，我们提出了NOTA，这是第一个大规模综合多模态音乐记谱数据集。该数据集包含1,019,237条记录，来自三个不同的地区，并包含三个任务。基于该数据集，我们训练了NotaGPT，这是一种音乐记谱视觉大型语言模型。具体而言，我们在预对齐训练阶段涉及跨模态对齐，将音乐谱图中表示的乐谱符号与其ABC表示的文本形式进行对齐。后续的训练阶段集中在基础音乐信息提取上，随后进行音乐记谱分析的训练。实验结果表明，我们的NotaGPT-7B在音乐理解方面取得了显著的改进，证明了NOTA和训练管道的有效性。我们的数据集在此处开放获取：[这里请提供正确的URL链接]。 

---
# EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild 

**Title (ZH)**: EgoSpeak：学习自中心对话代理在野生环境下的发言时机 

**Authors**: Junhyeok Kim, Min Soo Kim, Jiwan Chung, Jungbin Cho, Jisoo Kim, Sungwoong Kim, Gyeongbo Sim, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14892)  

**Abstract**: Predicting when to initiate speech in real-world environments remains a fundamental challenge for conversational agents. We introduce EgoSpeak, a novel framework for real-time speech initiation prediction in egocentric streaming video. By modeling the conversation from the speaker's first-person viewpoint, EgoSpeak is tailored for human-like interactions in which a conversational agent must continuously observe its environment and dynamically decide when to talk. Our approach bridges the gap between simplified experimental setups and complex natural conversations by integrating four key capabilities: (1) first-person perspective, (2) RGB processing, (3) online processing, and (4) untrimmed video processing. We also present YT-Conversation, a diverse collection of in-the-wild conversational videos from YouTube, as a resource for large-scale pretraining. Experiments on EasyCom and Ego4D demonstrate that EgoSpeak outperforms random and silence-based baselines in real time. Our results also highlight the importance of multimodal input and context length in effectively deciding when to speak. 

**Abstract (ZH)**: 在实际环境中介定何时启动对话仍然是对话代理面临的一项基本挑战。我们提出了EgoSpeak，这是一种新型框架，用于实现实时自视角流媒体视频中的启动对话预测。通过从发言人的第一人称视角建模对话，EgoSpeak特别适用于人类交互场景，其中对话代理必须持续观察环境并动态决定何时进行对话。我们的方法通过整合四项关键技术特性，弥合了简化实验设置与复杂自然对话之间的差距：（1）第一人称视角、（2）RGB图像处理、（3）在线处理和（4）未剪辑视频处理。我们还介绍了YT-Conversation，这是一个从YouTube收集的多样化的野外对话视频集合，作为大规模预训练的资源。在EasyCom和Ego4D上的实验结果表明，EgoSpeak在实时情况下优于随机和沉默基线。我们的结果还强调了多模态输入和上下文长度在有效决定何时发言方面的重要性。 

---
# CoDiff: Conditional Diffusion Model for Collaborative 3D Object Detection 

**Title (ZH)**: CoDiff：条件扩散模型在协作三维物体检测中的应用 

**Authors**: Zhe Huang, Shuo Wang, Yongcai Wang, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14891)  

**Abstract**: Collaborative 3D object detection holds significant importance in the field of autonomous driving, as it greatly enhances the perception capabilities of each individual agent by facilitating information exchange among multiple agents. However, in practice, due to pose estimation errors and time delays, the fusion of information across agents often results in feature representations with spatial and temporal noise, leading to detection errors. Diffusion models naturally have the ability to denoise noisy samples to the ideal data, which motivates us to explore the use of diffusion models to address the noise problem between multi-agent systems. In this work, we propose CoDiff, a novel robust collaborative perception framework that leverages the potential of diffusion models to generate more comprehensive and clearer feature representations. To the best of our knowledge, this is the first work to apply diffusion models to multi-agent collaborative perception. Specifically, we project high-dimensional feature map into the latent space of a powerful pre-trained autoencoder. Within this space, individual agent information serves as a condition to guide the diffusion model's sampling. This process denoises coarse feature maps and progressively refines the fused features. Experimental study on both simulated and real-world datasets demonstrates that the proposed framework CoDiff consistently outperforms existing relevant methods in terms of the collaborative object detection performance, and exhibits highly desired robustness when the pose and delay information of agents is with high-level noise. 

**Abstract (ZH)**: 协作三维物体检测在自动驾驶领域具有重要意义，因为它能够通过促进多个代理之间的信息交换，极大地提升每个个体代理的感知能力。然而，在实践中，由于姿态估计误差和时间延迟等原因，代理之间信息的融合经常产生空间和时间噪声，从而导致检测错误。扩散模型天然具有将噪声样本恢复到理想数据结构的能力，这促使我们探索利用扩散模型解决多代理系统中的噪声问题。在本研究中，我们提出了一种新颖的鲁棒协作感知框架CoDiff，该框架利用扩散模型的潜力生成更为全面和清晰的特征表示。据我们所知，这是首次将扩散模型应用于多代理协作感知。具体来说，我们将在强预训练自动编码器的隐空间中投影高维特征图。在这个空间内，个体代理的信息作为条件引导扩散模型的采样过程，这一过程可以去噪并逐步细化融合后的特征。通过对仿真和真实世界数据集进行实验研究，表明CoDiff框架在协作物体检测性能上始终优于现有相关方法，并且在代理的姿态和延迟信息存在高级噪声时表现出高度的鲁棒性。 

---
# Narrowing Information Bottleneck Theory for Multimodal Image-Text Representations Interpretability 

**Title (ZH)**: 窄化信息瓶颈理论在多模态图像-文本表示可解释性中的应用 

**Authors**: Zhiyu Zhu, Zhibo Jin, Jiayu Zhang, Nan Yang, Jiahao Huang, Jianlong Zhou, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.14889)  

**Abstract**: The task of identifying multimodal image-text representations has garnered increasing attention, particularly with models such as CLIP (Contrastive Language-Image Pretraining), which demonstrate exceptional performance in learning complex associations between images and text. Despite these advancements, ensuring the interpretability of such models is paramount for their safe deployment in real-world applications, such as healthcare. While numerous interpretability methods have been developed for unimodal tasks, these approaches often fail to transfer effectively to multimodal contexts due to inherent differences in the representation structures. Bottleneck methods, well-established in information theory, have been applied to enhance CLIP's interpretability. However, they are often hindered by strong assumptions or intrinsic randomness. To overcome these challenges, we propose the Narrowing Information Bottleneck Theory, a novel framework that fundamentally redefines the traditional bottleneck approach. This theory is specifically designed to satisfy contemporary attribution axioms, providing a more robust and reliable solution for improving the interpretability of multimodal models. In our experiments, compared to state-of-the-art methods, our approach enhances image interpretability by an average of 9%, text interpretability by an average of 58.83%, and accelerates processing speed by 63.95%. Our code is publicly accessible at this https URL. 

**Abstract (ZH)**: 多模态图像-文本表示的识别任务正逐渐受到广泛关注，特别是在CLIP（对比语言-图像预训练）等模型的应用中，这些模型在学习图像与文本之间的复杂关联方面表现出色。尽管取得了这些进展，确保此类模型的可解释性对于在现实世界应用（如医疗健康领域）中的安全部署依然至关重要。尽管已经开发了许多用于单模态任务的可解释性方法，但这些方法往往难以有效地推广到多模态上下文，原因在于两者在表示结构上的固有差异。信息理论中已验证的瓶颈方法曾被用来增强CLIP的可解释性，但它们常常受到强假设或内在随机性的影响。为了克服这些挑战，我们提出了信息瓶颈理论收窄方法的新框架，该框架从根本上重新定义了传统的瓶颈方法。该理论特别设计以满足当前的归因公理，提供了一个更稳健和可靠的方法来提高多模态模型的可解释性。在我们的实验中，与最先进的方法相比，我们提出的方法在图像可解释性方面平均提高了9%，在文本可解释性方面平均提高了58.83%，并且使处理速度加快了63.95%。我们的代码可以在此处获取：[此 https URL] 

---
# The Multi-Faceted Monosemanticity in Multimodal Representations 

**Title (ZH)**: 多模态表示中的多面向单义性 

**Authors**: Hanqi Yan, Xiangxiang Cui, Lu Yin, Paul Pu Liang, Yulan He, Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14888)  

**Abstract**: In this paper, we leverage recent advancements in feature monosemanticity to extract interpretable features from deep multimodal models, offering a data-driven understanding of modality gaps. Specifically, we investigate CLIP (Contrastive Language-Image Pretraining), a prominent visual-language representation model trained on extensive image-text pairs. Building upon interpretability tools developed for single-modal models, we extend these methodologies to assess multi-modal interpretability of CLIP features. Additionally, we introduce the Modality Dominance Score (MDS) to attribute the interpretability of each feature to its respective modality. Next, we transform CLIP features into a more interpretable space, enabling us to categorize them into three distinct classes: vision features (single-modal), language features (single-modal), and visual-language features (cross-modal). Our findings reveal that this categorization aligns closely with human cognitive understandings of different modalities. We also demonstrate significant use cases of this modality-specific features including detecting gender bias, adversarial attack defense and text-to-image model editing. These results indicate that large-scale multimodal models, equipped with task-agnostic interpretability tools, offer valuable insights into key connections and distinctions between different modalities. 

**Abstract (ZH)**: 在本文中，我们充分利用了特征单_semanticity（语义性）的最新进展，从深层多模态模型中提取可解释特征，从而提供数据驱动的模态间隙理解。具体而言，我们研究了CLIP（对比语言-图像预训练）模型，这是一种在大量图像-文本配对上训练的突出视觉-语言表示模型。在借鉴单模态模型解释工具的基础上，我们将其拓展到评估CLIP特征的多模态解释性。此外，我们引入了模态主导分数（MDS）来将每个特征的解释性归因于相应的模态。随后，我们将CLIP特征转换到一个更易解释的空间中，使我们能够将其分为三个不同的类别：视觉特征（单一模态）、语言特征（单一模态）和视觉-语言特征（跨模态）。我们的研究结果表明，这种分类与不同模态的人类认知理解高度一致。我们还展示了这类模态特定特征的重大应用案例，包括检测性别偏见、对抗性攻击防御和文本到图像模型编辑。这些发现表明，具有任务无关解释工具的大规模多模态模型提供了关于不同模态之间关键关联和区分的重要洞察。 

---
# Vision-Enhanced Time Series Forecasting via Latent Diffusion Models 

**Title (ZH)**: 基于潜扩散模型的视觉增强时间序列预测 

**Authors**: Weilin Ruan, Siru Zhong, Haomin Wen, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14887)  

**Abstract**: Diffusion models have recently emerged as powerful frameworks for generating high-quality images. While recent studies have explored their application to time series forecasting, these approaches face significant challenges in cross-modal modeling and transforming visual information effectively to capture temporal patterns. In this paper, we propose LDM4TS, a novel framework that leverages the powerful image reconstruction capabilities of latent diffusion models for vision-enhanced time series forecasting. Instead of introducing external visual data, we are the first to use complementary transformation techniques to convert time series into multi-view visual representations, allowing the model to exploit the rich feature extraction capabilities of the pre-trained vision encoder. Subsequently, these representations are reconstructed using a latent diffusion model with a cross-modal conditioning mechanism as well as a fusion module. Experimental results demonstrate that LDM4TS outperforms various specialized forecasting models for time series forecasting tasks. 

**Abstract (ZH)**: 近年来，扩散模型已经展现出生成高质量图像的强大框架。尽管近期的研究已经探索了它们在时间序列预测中的应用，但这些方法在跨模态建模和有效转换视觉信息以捕捉时间模式方面面临着重大挑战。在本文中，我们提出了一种名为LDM4TS的新框架，该框架利用潜在扩散模型的强大图像重建能力，结合视觉增强的时间序列预测。不同于引入外部视觉数据，我们首次采用互补的转换技术，将时间序列转换为多视角的视觉表示，使得模型能够利用预训练视觉编码器的丰富特征提取能力。随后，这些表示通过具有跨模态条件机制和融合模块的潜在扩散模型进行重建。实验结果表明，LDM4TS在时间序列预测任务中优于多种专门的时间序列预测模型。 

---
# Can LVLMs and Automatic Metrics Capture Underlying Preferences of Blind and Low-Vision Individuals for Navigational Aid? 

**Title (ZH)**: LVLMs 和自动评价指标能否捕捉盲人和低视力个体对导航辅助工具的内在偏好？ 

**Authors**: Na Min An, Eunki Kim, Wan Ju Kang, Sangryul Kim, Hyunjung Shim, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2502.14883)  

**Abstract**: Vision is a primary means of how humans perceive the environment, but Blind and Low-Vision (BLV) people need assistance understanding their surroundings, especially in unfamiliar environments. The emergence of semantic-based systems as assistance tools for BLV users has motivated many researchers to explore responses from Large Vision-Language Models (LVLMs). However, it has yet been studied preferences of BLV users on diverse types/styles of responses from LVLMs, specifically for navigational aid. To fill this gap, we first construct Eye4B dataset, consisting of human-validated 1.1k curated outdoor/indoor scenes with 5-10 relevant requests per scene. Then, we conduct an in-depth user study with eight BLV users to evaluate their preferences on six LVLMs from five perspectives: Afraidness, Nonactionability, Sufficiency, and Conciseness. Finally, we introduce Eye4B benchmark for evaluating alignment between widely used model-based image-text metrics and our collected BLV preferences. Our work can be set as a guideline for developing BLV-aware LVLMs towards a Barrier-Free AI system. 

**Abstract (ZH)**: 视觉是人类感知环境的主要方式，但视障人士和低视力（BLV）人士需要帮助理解周围的环境，尤其是在不熟悉环境中。基于语义的系统作为辅助工具来帮助BLV用户出现后，许多研究者开始探索大型视觉-语言模型（LVLMs）的响应方式。然而，尚未研究BLV用户对LVLMs不同类型/风格响应的偏好，特别是用于导航辅助的偏好。为了填补这一空白，我们首先构建了Eye4B数据集，该数据集包含1100个经过人工验证的户外和室内场景，并为每个场景提供了5-10个相关请求。然后，我们进行了一项深入的用户研究，与八位BLV用户一起评估他们对六种LVLMs在四个方面的偏好：惧怕感（Afraidness）、不可行动性（Nonactionability）、充分性（Sufficiency）和简洁性（Conciseness）。最后，我们介绍了Eye4B基准测试，用于评估广泛使用的基于模型的图像-文本度量标准与我们收集的BLV用户偏好之间的契合度。我们的工作可以作为开发BLV感知的LVLMs的一份指南，以实现无障碍的人工智能系统。 

---
# KKA: Improving Vision Anomaly Detection through Anomaly-related Knowledge from Large Language Models 

**Title (ZH)**: KKA：通过大型语言模型中的异常相关知识提高视觉异常检测性能 

**Authors**: Dong Chen, Zhengqing Hu, Peiguang Fan, Yueting Zhuang, Yafei Li, Qidong Liu, Xiaoheng Jiang, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14880)  

**Abstract**: Vision anomaly detection, particularly in unsupervised settings, often struggles to distinguish between normal samples and anomalies due to the wide variability in anomalies. Recently, an increasing number of studies have focused on generating anomalies to help detectors learn more effective boundaries between normal samples and anomalies. However, as the generated anomalies are often derived from random factors, they frequently lack realism. Additionally, randomly generated anomalies typically offer limited support in constructing effective boundaries, as most differ substantially from normal samples and lie far from the boundary. To address these challenges, we propose Key Knowledge Augmentation (KKA), a method that extracts anomaly-related knowledge from large language models (LLMs). More specifically, KKA leverages the extensive prior knowledge of LLMs to generate meaningful anomalies based on normal samples. Then, KKA classifies the generated anomalies as easy anomalies and hard anomalies according to their similarity to normal samples. Easy anomalies exhibit significant differences from normal samples, whereas hard anomalies closely resemble normal samples. KKA iteratively updates the generated anomalies, and gradually increasing the proportion of hard anomalies to enable the detector to learn a more effective boundary. Experimental results show that the proposed method significantly improves the performance of various vision anomaly detectors while maintaining low generation costs. The code for CMG can be found at this https URL. 

**Abstract (ZH)**: 视觉异常检测，特别是在无监督设置中，往往难以区分正常样本和异常样本，因为异常样本的变异性很大。近期，越来越多的研究专注于生成异常样本以帮助检测器学习更有效的正常样本与异常样本之间的界限。然而，由于生成的异常样本通常源自随机因素，它们往往缺乏现实感。此外，随机生成的异常样本通常在构建有效界限方面提供的支持有限，因为大多数异常样本与正常样本差异很大，且远离界限。为了解决这些挑战，我们提出了一种名为关键知识增强（Key Knowledge Augmentation, KKA）的方法，该方法从大型语言模型（LLMs）中提取与异常相关的知识。具体而言，KKA 利用大型语言模型丰富的先验知识，基于正常样本生成有意义的异常样本。然后，KKA 根据生成的异常样本与正常样本的相似程度将其分类为简单异常和复杂异常。简单异常与正常样本有显著差异，而复杂异常则与正常样本非常相似。KKA 通过迭代更新生成的异常样本，并逐渐增加复杂异常的比例，从而帮助检测器学习更有效的界限。实验结果表明，所提出的方法在保持低生成成本的同时显著提升了各种视觉异常检测器的性能。CMG的代码可以在此找到：[此链接]。 

---
# Is Mathematics Obsolete? 

**Title (ZH)**: 《数学过时了吗？》

这一标题是对于数学在未来的重要性及应用前景的质疑和探讨，将其翻译为《数学过时了吗？》既保留了原文的批判性，又符合中文学术文章的表达习惯。 

**Authors**: Jeremy Avigad  

**Link**: [PDF](https://arxiv.org/pdf/2502.14874)  

**Abstract**: This is an essay about the value of mathematical and symbolic reasoning in the age of AI. 

**Abstract (ZH)**: 这是一篇关于在人工智能时代数学与符号推理价值的文章。 

---
# Why do Experts Disagree on Existential Risk and P(doom)? A Survey of AI Experts 

**Title (ZH)**: 为什么专家在存在风险和“末日”概率上存在分歧？对AI专家的调查 

**Authors**: Severin Field  

**Link**: [PDF](https://arxiv.org/pdf/2502.14870)  

**Abstract**: The development of artificial general intelligence (AGI) is likely to be one of humanity's most consequential technological advancements. Leading AI labs and scientists have called for the global prioritization of AI safety citing existential risks comparable to nuclear war. However, research on catastrophic risks and AI alignment is often met with skepticism, even by experts. Furthermore, online debate over the existential risk of AI has begun to turn tribal (e.g. name-calling such as "doomer" or "accelerationist"). Until now, no systematic study has explored the patterns of belief and the levels of familiarity with AI safety concepts among experts. I surveyed 111 AI experts on their familiarity with AI safety concepts, key objections to AI safety, and reactions to safety arguments. My findings reveal that AI experts cluster into two viewpoints -- an "AI as controllable tool" and an "AI as uncontrollable agent" perspective -- diverging in beliefs toward the importance of AI safety. While most experts (78%) agreed or strongly agreed that "technical AI researchers should be concerned about catastrophic risks", many were unfamiliar with specific AI safety concepts. For example, only 21% of surveyed experts had heard of "instrumental convergence," a fundamental concept in AI safety predicting that advanced AI systems will tend to pursue common sub-goals (such as self-preservation). The least concerned participants were the least familiar with concepts like this, suggesting that effective communication of AI safety should begin with establishing clear conceptual foundations in the field. 

**Abstract (ZH)**: 人工通用智能（AGI）的发展很可能是人类最具重要意义的技术进步之一。领先的AI实验室和科学家们呼吁全球优先考虑AI安全，因为这关乎到类似于核战争的生存风险。然而，对于灾难性风险和AI对齐的研究常常受到怀疑，即使是专家也不例外。此外，关于AI生存风险的在线辩论已经开始变得部落化（例如，使用诸如“末日论者”或“加速主义者”之类的贬称）。到目前为止，还没有系统的研究来探索专家们关于AI安全信念模式及其对AI安全概念熟悉程度的规律。我调查了111名AI专家，了解他们在AI安全概念方面的熟悉程度、对AI安全的主要反对意见以及对安全论点的反应。我的发现表明，AI专家分为两类观点——“可控工具”和“不可控代理”的看法——在AI安全的重要性方面持有不同的信念。尽管大多数专家（78%）同意或非常同意“技术AI研究人员应关注灾难性风险”，但许多人对具体的AI安全概念不够熟悉。例如，只有21%的受访专家听说过“工具性趋同”，这是一个在AI安全领域中基本的概念，预测先进AI系统倾向于追求共同的子目标（如自我保存）。最不关心的参与者对于这种概念最不熟悉，这表明有效的AI安全沟通工作应该从在该领域建立清晰的概念基础开始。 

---
# Envisioning Stakeholder-Action Pairs to Mitigate Negative Impacts of AI: A Participatory Approach to Inform Policy Making 

**Title (ZH)**: 预见利益相关者行动对减轻人工智能负面影响的设想：一种参与式方法以指导政策制定 

**Authors**: Julia Barnett, Kimon Kieslich, Natali Helberger, Nicholas Diakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.14869)  

**Abstract**: The potential for negative impacts of AI has rapidly become more pervasive around the world, and this has intensified a need for responsible AI governance. While many regulatory bodies endorse risk-based approaches and a multitude of risk mitigation practices are proposed by companies and academic scholars, these approaches are commonly expert-centered and thus lack the inclusion of a significant group of stakeholders. Ensuring that AI policies align with democratic expectations requires methods that prioritize the voices and needs of those impacted. In this work we develop a participative and forward-looking approach to inform policy-makers and academics that grounds the needs of lay stakeholders at the forefront and enriches the development of risk mitigation strategies. Our approach (1) maps potential mitigation and prevention strategies of negative AI impacts that assign responsibility to various stakeholders, (2) explores the importance and prioritization thereof in the eyes of laypeople, and (3) presents these insights in policy fact sheets, i.e., a digestible format for informing policy processes. We emphasize that this approach is not targeted towards replacing policy-makers; rather our aim is to present an informative method that enriches mitigation strategies and enables a more participatory approach to policy development. 

**Abstract (ZH)**: 随着人工智能（AI）潜在负面影响的全球范围迅速扩大，强化负责任的AI治理的需求变得更为迫切。尽管许多监管机构支持基于风险的方法，并且众多风险管理措施由企业和学术学者提出，但这些方法往往是专家导向的，缺乏对重要利益相关者的广泛参与。确保AI政策符合民主期望需要优先考虑受影响者的声音和需求的方法。本研究开发了一种参与性和前瞻性的方法，旨在为政策制定者和学者提供支持，确保普通利益相关者的需要处于核心地位，并丰富风险管理策略的制定。我们的方法包括：（1）绘制分配给不同利益相关者的责任的潜在缓解和预防策略；（2）探讨这些策略在普通公众眼中的重要性和优先级；（3）以政策事实简报的形式呈现这些见解，即一种便于政策过程的信息格式。我们强调，本方法并非旨在取代政策制定者；而是希望通过提供一种信息丰富的方法来丰富风险管理策略，并促进更加参与性的政策制定过程。 

---
# Unlocking the Black Box: Analysing the EU Artificial Intelligence Act's Framework for Explainability in AI 

**Title (ZH)**: 解锁黑箱：分析欧盟人工智能法案中的AI解释性框架 

**Authors**: Georgios Pavlidis  

**Link**: [PDF](https://arxiv.org/pdf/2502.14868)  

**Abstract**: The lack of explainability of Artificial Intelligence (AI) is one of the first obstacles that the industry and regulators must overcome to mitigate the risks associated with the technology. The need for eXplainable AI (XAI) is evident in fields where accountability, ethics and fairness are critical, such as healthcare, credit scoring, policing and the criminal justice system. At the EU level, the notion of explainability is one of the fundamental principles that underpin the AI Act, though the exact XAI techniques and requirements are still to be determined and tested in practice. This paper explores various approaches and techniques that promise to advance XAI, as well as the challenges of implementing the principle of explainability in AI governance and policies. Finally, the paper examines the integration of XAI into EU law, emphasising the issues of standard setting, oversight, and enforcement. 

**Abstract (ZH)**: 人工智能（AI）缺乏可解释性是产业界和监管机构必须克服的第一个障碍，以减轻该技术带来的风险。在问责制、伦理和公平性至关重要的领域，可解释性AI（XAI）的需求尤为明显，例如医疗保健、信用评分、警务以及刑事司法系统。在欧盟层面，可解释性是AI法案基本原则之一，尽管具体的XAI技术和要求仍需在实践中确定和测试。本文探讨了各种能够推动XAI发展的方法和技术，以及在AI治理和政策中实施可解释性原则所面临的挑战。最后，本文研究了XAI在欧盟法律中的整合，强调标准制定、监督和执行方面的问题。 

---
# d-Sketch: Improving Visual Fidelity of Sketch-to-Image Translation with Pretrained Latent Diffusion Models without Retraining 

**Title (ZH)**: d-Sketch：无需重新训练预训练潜在扩散模型以提高草图到图像转换的视觉保真度 

**Authors**: Prasun Roy, Saumik Bhattacharya, Subhankar Ghosh, Umapada Pal, Michael Blumenstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.14007)  

**Abstract**: Structural guidance in an image-to-image translation allows intricate control over the shapes of synthesized images. Generating high-quality realistic images from user-specified rough hand-drawn sketches is one such task that aims to impose a structural constraint on the conditional generation process. While the premise is intriguing for numerous use cases of content creation and academic research, the problem becomes fundamentally challenging due to substantial ambiguities in freehand sketches. Furthermore, balancing the trade-off between shape consistency and realistic generation contributes to additional complexity in the process. Existing approaches based on Generative Adversarial Networks (GANs) generally utilize conditional GANs or GAN inversions, often requiring application-specific data and optimization objectives. The recent introduction of Denoising Diffusion Probabilistic Models (DDPMs) achieves a generational leap for low-level visual attributes in general image synthesis. However, directly retraining a large-scale diffusion model on a domain-specific subtask is often extremely difficult due to demanding computation costs and insufficient data. In this paper, we introduce a technique for sketch-to-image translation by exploiting the feature generalization capabilities of a large-scale diffusion model without retraining. In particular, we use a learnable lightweight mapping network to achieve latent feature translation from source to target domain. Experimental results demonstrate that the proposed method outperforms the existing techniques in qualitative and quantitative benchmarks, allowing high-resolution realistic image synthesis from rough hand-drawn sketches. 

**Abstract (ZH)**: 在图像到图像转换中提供结构指导可以使合成图像的形状控制更加精细。从用户的草图手绘草图生成高质量的逼真图像就是这样一个任务，旨在对条件生成过程施加结构约束。虽然这一前提对内容创作和学术研究中的许多应用场景都颇具吸引力，但由于自由hand绘制草图中的大量歧义性，问题变得从根本上更具挑战性。此外，形状一致性与真实生成之间的权衡进一步增加了这一过程的复杂性。基于生成对抗网络（GANs）的现有方法通常采用条件GAN或GAN逆运算，常常需要特定应用的数据和优化目标。最近提出的去噪扩散概率模型（DDPMs）在一般图像合成中的低级视觉属性生成方面实现了质的飞跃。然而，直接在特定领域任务上重新训练大规模扩散模型通常是由于计算成本高昂和数据不足而极其困难。在本文中，我们提出了一个技术，通过利用大规模扩散模型的特征泛化能力来进行草图到图像的转换，而无需重新训练。特别地，我们使用一个可学习的轻量级映射网络来实现从源域到目标域的潜在特征转换。实验结果表明，提出的这种方法在定性和定量基准测试中均优于现有技术，能够从草图手绘草图中生成高分辨率的逼真图像。 

---
# High Quality Segmentation for Ultra High-resolution Images 

**Title (ZH)**: 超高清图像的高品質分割 

**Authors**: Tiancheng Shen, Yuechen Zhang, Lu Qi, Jason Kuen, Xingyu Xie, Jianlong Wu, Zhe Lin, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2111.14482)  

**Abstract**: To segment 4K or 6K ultra high-resolution images needs extra computation consideration in image segmentation. Common strategies, such as down-sampling, patch cropping, and cascade model, cannot address well the balance issue between accuracy and computation cost. Motivated by the fact that humans distinguish among objects continuously from coarse to precise levels, we propose the Continuous Refinement Model~(CRM) for the ultra high-resolution segmentation refinement task. CRM continuously aligns the feature map with the refinement target and aggregates features to reconstruct these images' details. Besides, our CRM shows its significant generalization ability to fill the resolution gap between low-resolution training images and ultra high-resolution testing ones. We present quantitative performance evaluation and visualization to show that our proposed method is fast and effective on image segmentation refinement. Code will be released at this https URL. 

**Abstract (ZH)**: 对4K或6K超高清图像进行分割时，需要额外考虑图像分割中的计算复杂性问题。常见的策略，如下采样、图像块裁剪和级联模型，无法很好地解决准确性和计算成本之间的平衡问题。受人类从粗略到精细持续区分物体的启发，我们提出了一种连续精炼模型（Continuous Refinement Model, CRM），用于超高清图像分割的精炼任务。CRM持续地将特征图与精炼目标对齐，并聚合特征以重建图像的细节。此外，我们的CRM展示了其显著的泛化能力，能够填补低分辨率训练图像与超高清测试图像之间的分辨率差距。我们通过定量性能评估和可视化展示了所提出的方法在图像分割精炼方面的高效性和有效性。代码将会发布在以下链接。 

---
