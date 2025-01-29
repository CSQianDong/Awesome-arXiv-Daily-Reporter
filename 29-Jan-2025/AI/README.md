# SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training 

**Title (ZH)**: SFT 存储，RL 精炼：基础模型训练后比较研究

解释：
- SFT (Fine-tuning) 存储：这里的“SFT”可能是指“fine-tuning”，即微调。微调后的模型倾向于记住特定的训练数据。
- RL (Reinforcement Learning) 精炼：使用强化学习方法进行精炼，使模型具有更好的泛化能力。
- 基础模型训练后比较研究：研究微调与强化学习方法在基础模型训练后的表现比较。 

**Authors**: Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.17161)  

**Abstract**: Supervised fine-tuning (SFT) and reinforcement learning (RL) are widely used post-training techniques for foundation models. However, their roles in enhancing model generalization capabilities remain unclear. This paper studies the difference between SFT and RL on generalization and memorization, focusing on text-based rule variants and visual variants. We introduce GeneralPoints, an arithmetic reasoning card game, and adopt V-IRL, a real-world navigation environment, to assess how models trained with SFT and RL generalize to unseen variants in both textual and visual domains. We show that RL, especially when trained with an outcome-based reward, generalizes across both rule-based textual and visual variants. SFT, in contrast, tends to memorize training data and struggles to generalize out-of-distribution scenarios. Further analysis reveals that RL improves the model's underlying visual recognition capabilities, contributing to its enhanced generalization in the visual domain. Despite RL's superior generalization, we show that SFT remains essential for effective RL training; SFT stabilizes the model's output format, enabling subsequent RL to achieve its performance gains. These findings demonstrates the capability of RL for acquiring generalizable knowledge in complex, multi-modal tasks. 

**Abstract (ZH)**: 监督调优（SFT）和强化学习（RL）是广泛应用于基础模型的后训练技术。然而，它们在提高模型泛化能力方面的角色尚不清楚。本文研究了SFT和RL在泛化能力和记忆方面之间的差异，重点关注基于文本的规则变体和视觉变体。我们引入了GeneralPoints，一种算术推理卡片游戏，并采用V-IRL，一个真实世界的导航环境，评估使用SFT和RL训练的模型在文本和视觉领域中对未见过的变体的泛化能力。研究表明，与基于结果的奖励进行训练时，RL能够在基于规则的文本和视觉变体之间泛化。相比之下，SFT倾向于记忆训练数据，难以泛化到分布外场景。进一步分析表明，RL在提高模型的基础视觉识别能力方面发挥了作用，从而在视觉领域增强其泛化能力。尽管RL在泛化方面的表现优越，但我们展示了SFT对于有效的RL训练仍然是必不可少的；SFT稳定了模型的输出格式，使后续的RL能够实现其性能提升。这些发现表明，RL有能力在复杂、多模态任务中获取可泛化的知识。 

---
# Revisit Mixture Models for Multi-Agent Simulation: Experimental Study within a Unified Framework 

**Title (ZH)**: 在统一框架下重访混合模型在多智能体模拟中的应用：实验研究 

**Authors**: Longzhong Lin, Xuewu Lin, Kechun Xu, Haojian Lu, Lichao Huang, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.17015)  

**Abstract**: Simulation plays a crucial role in assessing autonomous driving systems, where the generation of realistic multi-agent behaviors is a key aspect. In multi-agent simulation, the primary challenges include behavioral multimodality and closed-loop distributional shifts. In this study, we revisit mixture models for generating multimodal agent behaviors, which can cover the mainstream methods including continuous mixture models and GPT-like discrete models. Furthermore, we introduce a closed-loop sample generation approach tailored for mixture models to mitigate distributional shifts. Within the unified mixture model~(UniMM) framework, we recognize critical configurations from both model and data perspectives. We conduct a systematic examination of various model configurations, including positive component matching, continuous regression, prediction horizon, and the number of components. Moreover, our investigation into the data configuration highlights the pivotal role of closed-loop samples in achieving realistic simulations. To extend the benefits of closed-loop samples across a broader range of mixture models, we further address the shortcut learning and off-policy learning issues. Leveraging insights from our exploration, the distinct variants proposed within the UniMM framework, including discrete, anchor-free, and anchor-based models, all achieve state-of-the-art performance on the WOSAC benchmark. 

**Abstract (ZH)**: 仿真在评估自动驾驶系统中扮演着至关重要的角色，其中生成现实主义多代理行为是关键方面。在多代理仿真中，主要挑战包括行为的多模态性和闭环分布变化。在此研究中，我们重新审视了用于生成多模态代理行为的混合模型，这些模型涵盖了主流方法，包括连续混合模型和类似于GPT的离散模型。此外，我们介绍了专门为混合模型设计的闭环样本生成方法，以缓解分布变化问题。在统一混合模型（UniMM）框架内，我们从建模和数据两个视角识别出了关键配置。我们系统地研究了各种模型配置，包括正分量匹配、连续回归、预测时间窗口以及分量的数量。此外，我们的数据配置研究突显了闭环样本在实现现实仿真中的关键作用。为了扩大闭环样本对更广泛混合模型的好处，我们还解决了捷径学习和离策学习问题。基于我们探索的见解，UniMM框架下的不同变种，包括离散型、无锚点型和基于锚点型模型，均在WOSAC基准测试中实现了最先进的性能。 

---
# Instantiation-based Formalization of Logical Reasoning Tasks using Language Models and Logical Solvers 

**Title (ZH)**: 基于实例的逻辑推理任务形式化方法：利用语言模型和逻辑求解器 

**Authors**: Mohammad Raza, Natasa Milic-Frayling  

**Link**: [PDF](https://arxiv.org/pdf/2501.16961)  

**Abstract**: Robustness of reasoning remains a significant challenge for large language models, and addressing it is essential for the practical applicability of AI-driven reasoning systems. We introduce Semantic Self-Verification (SSV), a novel approach that addresses the key challenge in combining language models with the rigor of logical solvers: to accurately formulate the reasoning problem from natural language to the formal language of the solver. SSV uses a consistency-based approach to produce strong abstract formalizations of problems using concrete instantiations that are generated by the model and verified by the solver. In addition to significantly advancing the overall reasoning accuracy over the state-of-the-art, a key novelty that this approach presents is a feature of verification that has near-perfect precision over a significant coverage of cases, as we demonstrate on open reasoning benchmarks. We propose such *near-certain reasoning* as a new approach to reduce the need for manual verification in many cases, taking us closer to more dependable and autonomous AI reasoning systems. 

**Abstract (ZH)**: 大型语言模型在推理的稳健性方面仍面临重大挑战，而解决这一问题对于AI驱动的推理系统的实际应用至关重要。我们提出了语义自我验证（SSV）这一新颖的方法，以解决将语言模型与逻辑求解器的严谨性相结合的关键挑战：将自然语言准确地转化为求解器的形式语言。SSV 使用一致性为基础的方法，通过模型生成的具体实例来生成问题的强抽象形式化表示，并通过求解器验证这些实例。除了在整体推理准确性上显著超越现有技术之外，该方法的一个重要创新点在于验证在大量情形下具有近乎完美的精确度，我们通过开放推理基准测试进行了展示。我们提出了这种“几乎确定的推理”作为减少许多情况下人工验证需求的新方法，使我们更接近于更加可靠和自主的AI推理系统。 

---
# Agential AI for Integrated Continual Learning, Deliberative Behavior, and Comprehensible Models 

**Title (ZH)**: 整合持续学习、审慎行为和可解释模型的代理人工智能 

**Authors**: Zeki Doruk Erden, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2501.16922)  

**Abstract**: Contemporary machine learning paradigm excels in statistical data analysis, solving problems that classical AI couldn't. However, it faces key limitations, such as a lack of integration with planning, incomprehensible internal structure, and inability to learn continually. We present the initial design for an AI system, Agential AI (AAI), in principle operating independently or on top of statistical methods, designed to overcome these issues. AAI's core is a learning method that models temporal dynamics with guarantees of completeness, minimality, and continual learning, using component-level variation and selection to learn the structure of the environment. It integrates this with a behavior algorithm that plans on a learned model and encapsulates high-level behavior patterns. Preliminary experiments on a simple environment show AAI's effectiveness and potential. 

**Abstract (ZH)**: 当代机器学习范式在统计数据分析方面表现出色，能够解决经典人工智能无法解决的问题。然而，它面临着关键性限制，如与规划的整合不足、内部结构难以理解以及无法持续学习。我们提出了一个旨在克服这些问题的AI系统初始设计，名为代理性AI（Agential AI, AAI），原则上可以独立运行或建立在统计方法之上。AAI的核心是一种学习方法，能够以完备性、简洁性和持续学习为保证地建模时间动态，并利用组件级的变异和选择来学习环境结构。它将这种建模与一个行为算法相结合，该算法基于学习到的模型进行规划，并封装高层次的行为模式。初步实验在简单环境中展示了AAI的有效性和潜力。 

---
# MACI: Multi-Agent Collaborative Intelligence for Robust Reasoning and Temporal Planning 

**Title (ZH)**: MACI：多智能体协作智能用于稳健推理与时间规划 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16689)  

**Abstract**: Artificial intelligence requires deliberate reasoning, temporal awareness, and effective constraint management, capabilities beyond the pattern-matching strengths of LLMs. LLMs struggle with planning tasks because of their reliance on associative reasoning, inability to self-verify, and inconsistent constraint awareness. We propose Multi-Agent Collaborative Intelligence (MACI), a framework centered on a meta-planner (MP) that orchestrates multiple agents to generate planner templates that define roles and constraints. These planners produce actionable workflows of role nodes and dependency constraints, enabling advanced temporal reasoning and adaptability.
MACI's three-tier architecture includes a meta-planning module for planner construction, common agents for general reasoning, and specialized agents for domain expertise. By decoupling planning from validation, it overcomes key LLM limitations. Evaluations demonstrate MACI's effective constraint satisfaction, conflict detection, and reasoning, positioning it as a robust solution for complex reasoning and planning tasks. 

**Abstract (ZH)**: 人工智能需要精心的推理、时间意识和有效的约束管理，这些能力超出了大语言模型（LLMs）在模式匹配方面的优势。LLMs 在规划任务中面临挑战，主要是因为它们依赖联想推理、缺乏自我验证能力和约束意识不一致。我们提出多智能体协作智能（MACI），这是一种以元规划器（MP）为核心框架，协调多个智能体生成规划模板的方法。这些规划模板定义角色和约束，从而生成可执行的工作流节点及其依赖约束，提供高级时间推理和适应性。

MACI 的三层架构包括：元规划模块，用于构建规划；通用智能体，用于一般推理；以及专业智能体，用于特定领域的专业知识。通过将规划与验证分离，MACI 克服了大语言模型的关键局限性。评估结果显示，MACI 在有效约束满足、冲突检测和推理方面表现出色，使其成为解决复杂推理和规划任务的稳健解决方案。 

---
# VeriFact: Verifying Facts in LLM-Generated Clinical Text with Electronic Health Records 

**Title (ZH)**: VeriFact: 使用电子健康记录验证大规模语言模型生成的临床文本中的事实 

**Authors**: Philip Chung, Akshay Swaminathan, Alex J. Goodell, Yeasul Kim, S. Momsen Reincke, Lichy Han, Ben Deverett, Mohammad Amin Sadeghi, Abdel-Badih Ariss, Marc Ghanem, David Seong, Andrew A. Lee, Caitlin E. Coombes, Brad Bradshaw, Mahir A. Sufian, Hyo Jung Hong, Teresa P. Nguyen, Mohammad R. Rasouli, Komal Kamra, Mark A. Burbridge, James C. McAvoy, Roya Saffary, Stephen P. Ma, Dev Dash, James Xie, Ellen Y. Wang, Clifford A. Schmiesing, Nigam Shah, Nima Aghaeepour  

**Link**: [PDF](https://arxiv.org/pdf/2501.16672)  

**Abstract**: Methods to ensure factual accuracy of text generated by large language models (LLM) in clinical medicine are lacking. VeriFact is an artificial intelligence system that combines retrieval-augmented generation and LLM-as-a-Judge to verify whether LLM-generated text is factually supported by a patient's medical history based on their electronic health record (EHR). To evaluate this system, we introduce VeriFact-BHC, a new dataset that decomposes Brief Hospital Course narratives from discharge summaries into a set of simple statements with clinician annotations for whether each statement is supported by the patient's EHR clinical notes. Whereas highest agreement between clinicians was 88.5%, VeriFact achieves up to 92.7% agreement when compared to a denoised and adjudicated average human clinican ground truth, suggesting that VeriFact exceeds the average clinician's ability to fact-check text against a patient's medical record. VeriFact may accelerate the development of LLM-based EHR applications by removing current evaluation bottlenecks. 

**Abstract (ZH)**: 确保大型语言模型（LLM）在临床医学中生成的文本事实准确性的方法尚缺乏。VeriFact 是一种人工智能系统，结合了检索增强生成和LLM作为法官的技术，用于验证LLM生成的文本是否由患者电子健康记录（EHR）中的医疗历史事实支持。为了评估该系统，我们引入了VeriFact-BHC新数据集，该数据集将出院总结中的简短医院课程叙述分解为一系列简单陈述，并提供了临床医生的标注，表明每个陈述是否由患者的临床笔记支持。尽管临床医生之间的最高一致性为88.5%，但与去噪并经过裁定的人类临床医生一致性平均值相比，VeriFact 的一致性最高可达92.7%。这表明VeriFact 超过了普通临床医生的事实核查能力，能够对照患者的医疗记录检查文本的准确性。VeriFact 可能会加速LLM基于EHR的应用开发，通过消除当前的评估瓶颈。 

---
# CowPilot: A Framework for Autonomous and Human-Agent Collaborative Web Navigation 

**Title (ZH)**: CowPilot: 一种自主与人-智能体协作的网页导航框架 

**Authors**: Faria Huq, Zora Zhiruo Wang, Frank F. Xu, Tianyue Ou, Shuyan Zhou, Jeffrey P. Bigham, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2501.16609)  

**Abstract**: While much work on web agents emphasizes the promise of autonomously performing tasks on behalf of users, in reality, agents often fall short on complex tasks in real-world contexts and modeling user preference. This presents an opportunity for humans to collaborate with the agent and leverage the agent's capabilities effectively. We propose CowPilot, a framework supporting autonomous as well as human-agent collaborative web navigation, and evaluation across task success and task efficiency. CowPilot reduces the number of steps humans need to perform by allowing agents to propose next steps, while users are able to pause, reject, or take alternative actions. During execution, users can interleave their actions with the agent by overriding suggestions or resuming agent control when needed. We conducted case studies on five common websites and found that the human-agent collaborative mode achieves the highest success rate of 95% while requiring humans to perform only 15.2% of the total steps. Even with human interventions during task execution, the agent successfully drives up to half of task success on its own. CowPilot can serve as a useful tool for data collection and agent evaluation across websites, which we believe will enable research in how users and agents can work together. Video demonstrations are available at this https URL 

**Abstract (ZH)**: 尽管关于网络代理的研究大多强调代理能够自主为用户执行任务的优点，但在实际应用中，代理在处理复杂任务时常常表现不佳，也无法有效建模用户偏好。这为人类与代理的合作提供了一个机会，使人类可以利用代理的能力。我们提出了一种名为CowPilot的框架，支持代理自主导航以及人类与代理的协作导航，并从任务成功率和任务效率两个维度对其进行评估。CowPilot通过允许代理提出下一步操作来减少人类需要执行的步骤数量，同时用户可以暂停、拒绝或采取替代行动。在执行过程中，用户可以通过覆盖建议或在需要时恢复代理控制来与代理交错执行操作。我们在五个常见网站上开展了案例研究，发现协作模式下的人类与代理达到了95%的最高成功率，人类仅需执行总步骤的15.2%。即使在任务执行过程中有人类干预，代理也能独立完成任务的一半以上成功。CowPilot可以作为跨网站的数据收集和代理评估工具，我们认为这将促进用户和代理如何协同工作的研究。有关视频演示可在以下URL访问：[这个 https URL](这个 https URL) 

---
# Sample-Efficient Behavior Cloning Using General Domain Knowledge 

**Title (ZH)**: 使用通用领域知识的样本高效行为克隆 

**Authors**: Feiyu Zhu, Jean Oh, Reid Simmons  

**Link**: [PDF](https://arxiv.org/pdf/2501.16546)  

**Abstract**: Behavior cloning has shown success in many sequential decision-making tasks by learning from expert demonstrations, yet they can be very sample inefficient and fail to generalize to unseen scenarios. One approach to these problems is to introduce general domain knowledge, such that the policy can focus on the essential features and may generalize to unseen states by applying that knowledge. Although this knowledge is easy to acquire from the experts, it is hard to be combined with learning from individual examples due to the lack of semantic structure in neural networks and the time-consuming nature of feature engineering. To enable learning from both general knowledge and specific demonstration trajectories, we use a large language model's coding capability to instantiate a policy structure based on expert domain knowledge expressed in natural language and tune the parameters in the policy with demonstrations. We name this approach the Knowledge Informed Model (KIM) as the structure reflects the semantics of expert knowledge. In our experiments with lunar lander and car racing tasks, our approach learns to solve the tasks with as few as 5 demonstrations and is robust to action noise, outperforming the baseline model without domain knowledge. This indicates that with the help of large language models, we can incorporate domain knowledge into the structure of the policy, increasing sample efficiency for behavior cloning. 

**Abstract (ZH)**: 行为克隆在许多序列决策任务中通过学习专家演示取得了成功，但它们可能非常样本效率低下，并且难以泛化到未见过的场景。解决这些问题的方法之一是引入通用领域知识，从而使策略能够关注于关键特征，并通过应用这些知识在未见过的状态上泛化。尽管可以从专家那里轻松获取此类知识，但由于神经网络缺乏语义结构以及特征工程的耗时性，将此类知识与从个体示例中学习进行结合十分困难。为了同时从通用知识和具体示例轨迹中进行学习，我们利用大型语言模型的编码能力，基于自然语言表达的专家领域知识实例化策略结构，并通过示例对策略的参数进行调整。我们将这种做法命名为知识指导模型（KIM），因为其结构反映了专家知识的语义。在对月球着陆和赛车任务进行的实验中，我们的方法仅使用5次演示即可学会解决问题，并对动作噪声具有鲁棒性，优于未使用领域知识的基线模型。这表明，在大型语言模型的帮助下，可以将领域知识融入策略结构中，从而提高行为克隆的学习样本效率。 

---
# What is Harm? Baby Don't Hurt Me! On the Impossibility of Complete Harm Specification in AI Alignment 

**Title (ZH)**: 什么是伤害？宝宝别伤害我！关于在AI对齐中完全规定伤害的不可能性 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2501.16448)  

**Abstract**: "First, do no harm" faces a fundamental challenge in artificial intelligence: how can we specify what constitutes harm? While prior work treats harm specification as a technical hurdle to be overcome through better algorithms or more data, we argue this assumption is unsound. Drawing on information theory, we demonstrate that complete harm specification is fundamentally impossible for any system where harm is defined external to its specifications. This impossibility arises from an inescapable information-theoretic gap: the entropy of harm H(O) always exceeds the mutual information I(O;I) between ground truth harm O and a system's specifications I.
We introduce two novel metrics: semantic entropy H(S) and the safety-capability ratio I(O;I)/H(O), to quantify these limitations. Through a progression of increasingly sophisticated specification attempts, we show why each approach must fail and why the resulting gaps are not mere engineering challenges but fundamental constraints akin to the halting problem. These results suggest a paradigm shift: rather than pursuing complete specifications, AI alignment research should focus on developing systems that can operate safely despite irreducible specification uncertainty. 

**Abstract (ZH)**: “首先，不造成伤害”在人工智面前面临着一个基本挑战：我们如何界定什么构成伤害？尽管过去的研究将伤害定义视为通过更好的算法或更多数据克服的技术障碍，我们认为这种假设是不成立的。借鉴信息技术论，我们证明，在任何其伤害定义外部于系统规范的系统中，完全界定伤害从根本上是不可能的。这种不可能性源自不可逃避的信息论差距：不确定伤害H(O)的熵总是超过系统规范I与真实伤害O之间互信息I(O;I)。

我们引入了两个新的度量标准：语义熵H(S)和安全能力比I(O;I)/H(O)，以量化这些限制。通过一系列越来越复杂的规范尝试，我们展示了为什么每一个方法都会失败，为什么这些差距不仅仅是工程上的挑战，而是类似于停机问题的基本限制。这些结果表明了范式转变：而不是追求完全的规范，人工智能对齐研究应集中在开发能够在固有的规范不确定性下安全运行的系统上。 

---
# A Method for Multi-Hop Question Answering on Persian Knowledge Graph 

**Title (ZH)**: 一种针对波斯知识图谱的多跳问答方法 

**Authors**: Arash Ghafouri, Mahdi Firouzmandi, Hasan Naderi  

**Link**: [PDF](https://arxiv.org/pdf/2501.16350)  

**Abstract**: Question answering systems are the latest evolution in information retrieval technology, designed to accept complex queries in natural language and provide accurate answers using both unstructured and structured knowledge sources. Knowledge Graph Question Answering (KGQA) systems fulfill users' information needs by utilizing structured data, representing a vast number of facts as a graph. However, despite significant advancements, major challenges persist in answering multi-hop complex questions, particularly in Persian. One of the main challenges is the accurate understanding and transformation of these multi-hop complex questions into semantically equivalent SPARQL queries, which allows for precise answer retrieval from knowledge graphs. In this study, to address this issue, a dataset of 5,600 Persian multi-hop complex questions was developed, along with their decomposed forms based on the semantic representation of the questions. Following this, Persian language models were trained using this dataset, and an architecture was proposed for answering complex questions using a Persian knowledge graph. Finally, the proposed method was evaluated against similar systems on the PeCoQ dataset. The results demonstrated the superiority of our approach, with an improvement of 12.57% in F1-score and 12.06% in accuracy compared to the best comparable method. 

**Abstract (ZH)**: 基于信息检索技术的问答系统是最新的一种演进形式，旨在接受复杂的自然语言查询，并通过使用结构化和非结构化的知识源提供准确的答案。基于知识图谱的问答（KGQA）系统通过利用结构化数据，将大量事实表示为图，来满足用户的信息需求。尽管取得了显著的进步，但在回答多跳复杂问题（特别是在波斯语中）方面仍存在大量挑战。其中主要的挑战是如何准确理解并转换这些多跳复杂问题为语义等价的SPARQL查询，从而从知识图谱中精确检索出答案。在这项研究中，为了解决这一问题，我们开发了一个包含5600个波斯语多跳复杂问题及其基于问题语义分解形式的数据集。之后，我们使用该数据集训练了波斯语语言模型，并提出了一个利用波斯知识图谱回答复杂问题的架构。最后，我们在PeCoQ数据集上将所提出的方法与类似的系统进行了评估。结果表明，我们的方法具有优越性，F1分数提高了12.57%，准确率提高了12.06%，超过了最佳可比方法。 

---
# A Hybrid Deep Learning CNN Model for Enhanced COVID-19 Detection from Computed Tomography (CT) Scan Images 

**Title (ZH)**: 一种集成深度学习卷积神经网络模型，用于增强 COVID-19 从计算机断层扫描 (CT) 图像中的检测 

**Authors**: Suresh Babu Nettur, Shanthi Karpurapu, Unnati Nettur, Likhit Sagar Gajja, Sravanthy Myneni, Akhil Dusi, Lalithya Posham  

**Link**: [PDF](https://arxiv.org/pdf/2501.17160)  

**Abstract**: Early detection of COVID-19 is crucial for effective treatment and controlling its spread. This study proposes a novel hybrid deep learning model for detecting COVID-19 from CT scan images, designed to assist overburdened medical professionals. Our proposed model leverages the strengths of VGG16, DenseNet121, and MobileNetV2 to extract features, followed by Principal Component Analysis (PCA) for dimensionality reduction, after which the features are stacked and classified using a Support Vector Classifier (SVC). We conducted comparative analysis between the proposed hybrid model and individual pre-trained CNN models, using a dataset of 2,108 training images and 373 test images comprising both COVID-positive and non-COVID images. Our proposed hybrid model achieved an accuracy of 98.93%, outperforming the individual models in terms of precision, recall, F1 scores, and ROC curve performance. 

**Abstract (ZH)**: 早期检测新冠肺炎对于有效的治疗和控制其传播至关重要。本研究提出了一种新颖的混合深度学习模型，用于从CT扫描图像中检测新冠肺炎，旨在协助繁忙的医疗专业人员。我们提出的方法结合了VGG16、DenseNet121和MobileNetV2的优势来提取特征，然后通过主成分分析（PCA）进行维数缩减，最后使用支持向量机分类器（SVC）对特征进行堆叠并分类。我们使用包含2,108张训练图像和373张测试图像（其中包含新冠肺炎阳性图像和非新冠肺炎图像）的数据集，分别与预训练的CNN模型进行了对比分析。我们提出的高度集成模型在精确度、召回率、F1分数和ROC曲线性能方面均优于单一的预训练模型，达到了98.93%的准确率。 

---
# Three-Dimensional Diffusion-Weighted Multi-Slab MRI With Slice Profile Compensation Using Deep Energy Model 

**Title (ZH)**: 使用深度能量模型进行切片轮廓校正的三维扩散加权多层MRI 

**Authors**: Reza Ghorbani, Jyothi Rikhab Chand, Chu-Yu Lee, Mathews Jacob, Merry Mani  

**Link**: [PDF](https://arxiv.org/pdf/2501.17152)  

**Abstract**: Three-dimensional (3D) multi-slab acquisition is a technique frequently employed in high-resolution diffusion-weighted MRI in order to achieve the best signal-to-noise ratio (SNR) efficiency. However, this technique is limited by slab boundary artifacts that cause intensity fluctuations and aliasing between slabs which reduces the accuracy of anatomical imaging. Addressing this issue is crucial for advancing diffusion MRI quality and making high-resolution imaging more feasible for clinical and research applications. In this work, we propose a regularized slab profile encoding (PEN) method within a Plug-and-Play ADMM framework, incorporating multi-scale energy (MuSE) regularization to effectively improve the slab combined reconstruction. Experimental results demonstrate that the proposed method significantly improves image quality compared to non-regularized and TV-regularized PEN approaches. The regularized PEN framework provides a more robust and efficient solution for high-resolution 3D diffusion MRI, potentially enabling clearer, more reliable anatomical imaging across various applications. 

**Abstract (ZH)**: 三维（3D）多层采集是一种常用于高分辨率扩散加权磁共振成像的技术，旨在实现最佳信噪比（SNR）。然而，该技术受到层边界伪影的限制，这些伪影导致层间强度波动和混叠，从而降低解剖成像的准确性。解决这一问题对于提高扩散磁共振成像的质量，并使高分辨率成像更适用于临床和研究应用至关重要。本文提出了一种在插件和玩耍ADMM框架内的正则化层剖面编码（PEN）方法，结合多尺度能量（MuSE）正则化，以有效改善层结合重建。实验结果表明，所提出的方法显著改善了图像质量，与非正则化和TV正则化的PEN方法相比，正则化PEN框架提供了更稳健且高效的解决方案，有助于提供高分辨率3D扩散磁共振成像，可能在各种应用中实现更清晰可靠的解剖成像。 

---
# AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders 

**Title (ZH)**: AxBench: 引导大规模语言模型超车——即使简单的基线模型也可超越稀疏自编码器 

**Authors**: Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2501.17148)  

**Abstract**: Fine-grained steering of language model outputs is essential for safety and reliability. Prompting and finetuning are widely used to achieve these goals, but interpretability researchers have proposed a variety of representation-based techniques as well, including sparse autoencoders (SAEs), linear artificial tomography, supervised steering vectors, linear probes, and representation finetuning. At present, there is no benchmark for making direct comparisons between these proposals. Therefore, we introduce AxBench, a large-scale benchmark for steering and concept detection, and report experiments on Gemma-2-2B and 9B. For steering, we find that prompting outperforms all existing methods, followed by finetuning. For concept detection, representation-based methods such as difference-in-means, perform the best. On both evaluations, SAEs are not competitive. We introduce a novel weakly-supervised representational method (Rank-1 Representation Finetuning; ReFT-r1), which is competitive on both tasks while providing the interpretability advantages that prompting lacks. Along with AxBench, we train and publicly release SAE-scale feature dictionaries for ReFT-r1 and DiffMean. 

**Abstract (ZH)**: 细粒度的语言模型输出控制对于安全性和可靠性至关重要。目前广泛使用提示和微调来实现这些目标，但解释性研究人员还提出了多种基于表示的技术，包括稀疏自编码器（SAEs）、线性人工透镜、监督导向矢量、线性探针和表示微调。目前尚无直接比较这些提案的基准。因此，我们引入了AxBench，这是一个大规模的控制和概念检测基准，并在Gemma-2-2B和9B上报告了实验结果。对于控制而言，我们发现提示优于所有现有方法，其次是微调。对于概念检测，基于表示的方法如差值均值（Difference-in-means）表现最佳。在两个评估中，SAEs都不具竞争力。我们引入了一种新颖的弱监督表示方法（Rank-1 表示微调；ReFT-r1），该方法在两个任务上都具有竞争力，并且提供了提示所缺乏的解释性优势。除了AxBench，我们还训练并公开发布了适用于ReFT-r1和差值均值的SAEs规模特征字典。 

---
# FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data 

**Title (ZH)**: FactCG：基于图的多跳数据增强事实核查者 

**Authors**: Deren Lei, Yaxi Li, Siyao Li, Mengya Hu, Rui Xu, Ken Archer, Mingyu Wang, Emily Ching, Alex Deng  

**Link**: [PDF](https://arxiv.org/pdf/2501.17144)  

**Abstract**: Prior research on training grounded factuality classification models to detect hallucinations in large language models (LLMs) has relied on public natural language inference (NLI) data and synthetic data. However, conventional NLI datasets are not well-suited for document-level reasoning, which is critical for detecting LLM hallucinations. Recent approaches to document-level synthetic data generation involve iteratively removing sentences from documents and annotating factuality using LLM-based prompts. While effective, this method is computationally expensive for long documents and limited by the LLM's capabilities. In this work, we analyze the differences between existing synthetic training data used in state-of-the-art models and real LLM output claims. Based on our findings, we propose a novel approach for synthetic data generation, CG2C, that leverages multi-hop reasoning on context graphs extracted from documents. Our fact checker model, FactCG, demonstrates improved performance with more connected reasoning, using the same backbone models. Experiments show it even outperforms GPT-4-o on the LLM-Aggrefact benchmark with much smaller model size. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

以往关于训练生成事实分类模型以检测大型语言模型（LLMs）幻觉的研究依赖于公开的自然语言推理（NLI）数据和合成数据。然而，传统的NLI数据集不适用于文档级推理，这是检测LLM幻觉所必需的。最近的文档级合成数据生成方法涉及逐步从文档中移除句子，并使用LLM基于的提示进行事实性的标注。尽管该方法有效，但在处理长文档时计算成本较高，并且受限于LLM的能力。在本研究中，我们分析了现有用于先进模型的合成训练数据与真实LLM输出声明之间的差异。根据我们的研究结果，我们提出了一种名为CG2C的新颖合成数据生成方法，利用从文档中提取的上下文图进行多跳推理。我们的事实核查模型FactCG在具有更关联推理时表现出更好的性能，并使用相同的基础模型。实验表明，其在LLM-Aggrefact基准测试中甚至在模型规模更小的情况下也优于GPT-4-o。 

---
# Histoires Morales: A French Dataset for Assessing Moral Alignment 

**Title (ZH)**: 道德故事集：一个用于评估道德对齐的法语数据集 

**Authors**: Thibaud Leteno, Irina Proskurina, Antoine Gourru, Julien Velcin, Charlotte Laclau, Guillaume Metzler, Christophe Gravier  

**Link**: [PDF](https://arxiv.org/pdf/2501.17117)  

**Abstract**: Aligning language models with human values is crucial, especially as they become more integrated into everyday life. While models are often adapted to user preferences, it is equally important to ensure they align with moral norms and behaviours in real-world social situations. Despite significant progress in languages like English and Chinese, French has seen little attention in this area, leaving a gap in understanding how LLMs handle moral reasoning in this language. To address this gap, we introduce Histoires Morales, a French dataset derived from Moral Stories, created through translation and subsequently refined with the assistance of native speakers to guarantee grammatical accuracy and adaptation to the French cultural context. We also rely on annotations of the moral values within the dataset to ensure their alignment with French norms. Histoires Morales covers a wide range of social situations, including differences in tipping practices, expressions of honesty in relationships, and responsibilities toward animals. To foster future research, we also conduct preliminary experiments on the alignment of multilingual models on French and English data and the robustness of the alignment. We find that while LLMs are generally aligned with human moral norms by default, they can be easily influenced with user-preference optimization for both moral and immoral data. 

**Abstract (ZH)**: 将语言模型与人类价值观对齐至关重要，尤其是在它们越来越融入日常生活的情况下。尽管模型通常会根据用户偏好进行调整，但确保它们在现实社会情境中与道德规范和行为相一致同样重要。尽管在英语和汉语等语言方面取得了显著进展，法语在这方面却几乎未得到关注，这导致我们对法语语言模型在道德推理方面的处理能力缺乏了解。为填补这一空白，我们引入了“道德故事”（Histoires Morales）数据集，该数据集来源于道德故事，通过翻译并随后借助母语者的帮助进行精炼，以确保语法的准确性并适应法国的文化背景。我们还利用数据集中道德价值观的标注来确保其与法国规范的一致性。“道德故事”涵盖了广泛的社会情境，包括小费习俗的不同、关系中的诚实表达以及对动物的责任等方面。为了促进未来的研究，我们还进行了初步实验，探讨多语言模型在法语和英语数据上的对齐情况及其对齐的稳健性。我们发现，尽管语言模型通常默认与人类道德规范保持一致，但它们可以通过用户偏好优化轻松地受到道德和不道德数据的影响。 

---
# COS(M+O)S: Curiosity and RL-Enhanced MCTS for Exploring Story Space via Language Models 

**Title (ZH)**: COS(M+O)S：好奇心和强化学习增强的MCTS方法，通过语言模型探索故事情境空间 

**Authors**: Tobias Materzok  

**Link**: [PDF](https://arxiv.org/pdf/2501.17104)  

**Abstract**: We present COS(M+O)S, a System 2-inspired framework for open-ended plot development that systematically explores the vast space of possible story expansions, enabling a 3B-parameter language model to approach the plot quality of a 70B model on select short-story tasks. The method accomplishes this by combining Monte Carlo Tree Search (MCTS), guided by a step-level value model that rewards moderate surprisal (curiosity) while penalizing incoherence, and Odds Ratio Preference Optimization (ORPO) to fine-tune the policy on high-value plot expansions. This iterative reinforcement learning loop systematically explores multiple candidate plot branches, backpropagates quality signals, and adapts the policy for faster convergence, notably shifting the policy from puzzle-based Chain-of-Thought to more character-driven storytelling. In small-scale tests with short-story prompts, 67%-77% of participants favored COS(M+O)S's highest-rated expansions over lower-rated ones, suggesting that our learned value function aligns. GPT-4o ratings further show that COS(M+O)S surpasses naive single-pass decoding from Llama 3.2 3B by 0.59 SD, coming within 0.06 SD of Llama 3.1 70B (no significant difference, p=0.93). Pairwise comparisons with o1 place COS(M+O)S 1.5 SD above the 3B baseline and find no statistically significant gap from 70B. Nevertheless, absolute story quality remains modest, constrained by the small model's capacity and limited training data. 

**Abstract (ZH)**: 我们提出了COS(M+O)S系统，这是一种受到System 2启发的框架，用于开放式故事情节发展。该框架系统地探索可能的故事扩展的庞大空间，使一个30亿参数的语言模型在特定的短篇故事任务上接近700亿参数模型的故事情节质量。该方法通过结合蒙特卡洛树搜索（MCTS），并通过逐步价值模型进行引导，该模型奖励适度的惊讶（好奇心）同时惩罚不连贯性，来实现这一目标。此外，还结合了奇数比偏好优化（ORPO）来微调具有高价值情节扩展的策略。这个循环的强化学习过程中系统地探索多个候选情节分支，反向传播质量信号，并适应策略以加快收敛速度，明显地从基于谜题的逐步推理转变为以人物驱动的故事叙述。在对短篇故事提示的小规模测试中，67%-77%的参与者更青睐COS(M+O)S最高评分的情节扩展，这表明我们学到的价值函数与实际表现一致。GPT-4o的评分进一步显示COS(M+O)S在短时解码Llama 3.2 3B方面高出了0.59个标准差，离Llama 3.1 70B（标准偏差为0.06，无显著差异，p=0.93）仅一步之遥。与o1的成对比较中，COS(M+O)S比3B基线高出了1.5个标准差，并没有在统计学上与70B产生显著差距。然而，绝对的故事质量仍然有限，受到小型模型容量和有限训练数据的限制。 

---
# Why is the estimation of metaorder impact with public market data so challenging? 

**Title (ZH)**: 使用公开市场数据估算元订单影响为何如此具有挑战性？ 

**Authors**: Manuel Naviglio, Giacomo Bormetti, Francesco Campigli, German Rodikov, Fabrizio Lillo  

**Link**: [PDF](https://arxiv.org/pdf/2501.17096)  

**Abstract**: Estimating market impact and transaction costs of large trades (metaorders) is a very important topic in finance. However, using models of price and trade based on public market data provide average price trajectories which are qualitatively different from what is observed during real metaorder executions: the price increases linearly, rather than in a concave way, during the execution and the amount of reversion after its end is very limited. We claim that this is a generic phenomenon due to the fact that even sophisticated statistical models are unable to correctly describe the origin of the autocorrelation of the order flow. We propose a modified Transient Impact Model which provides more realistic trajectories by assuming that only a fraction of the metaorder trading triggers market order flow. Interestingly, in our model there is a critical condition on the kernels of the price and order flow equations in which market impact becomes permanent. 

**Abstract (ZH)**: 估计大型交易（metaorders）对市场的影响以及交易成本是一个金融领域中非常重要的课题。然而，基于公开市场数据构建的价格和交易模型提供的平均价格轨迹与实际大订单执行过程中观察到的轨迹在质上是不同的：在执行过程中，价格呈线性增长，而非凹形增长；而在交易结束后，价格的回撤也很有限。我们主张这是一种通用现象，原因在于即使是最复杂的统计模型也无法正确描述订单流自相关性的起源。为此，我们提出了一种修改后的瞬时冲击模型，这种方法通过假设只有部分大订单交易会触发市场订单流，从而提供更加现实的价格轨迹。有趣的是，在我们的模型中，价格和订单流方程的核函数存在一个临界条件，在这种条件下，市场冲击会变得持久。 

---
# Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models 

**Title (ZH)**: Mamba-Shedder：面向高效选择性结构化状态空间模型的后Transformer压缩方法 

**Authors**: J. Pablo Muñoz, Jinjie Yuan, Nilesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2501.17088)  

**Abstract**: Large pre-trained models have achieved outstanding results in sequence modeling. The Transformer block and its attention mechanism have been the main drivers of the success of these models. Recently, alternative architectures, such as Selective Structured State Space Models (SSMs), have been proposed to address the inefficiencies of Transformers. This paper explores the compression of SSM-based models, particularly Mamba and its hybrids. We study the sensitivity of these models to the removal of selected components at different granularities to reduce the model size and computational overhead, thus improving their efficiency while maintaining accuracy. The proposed solutions, collectively referred to as Mamba-Shedder, achieve a speedup of up to 1.4x during inference, demonstrating that model efficiency can be improved by eliminating several redundancies with minimal impact on the overall model performance. The code is available at this https URL. 

**Abstract (ZH)**: 大规模预训练模型在序列建模方面取得了卓越的成绩。Transformer组件及其注意机制是这些模型成功的主要推动力。最近，为了应对Transformer的效率问题，提出了一些替代架构，如选择性结构状态空间模型（SSMs）。本文探讨了基于SSM的模型的压缩方法，尤其是Mamba及其混合模型。我们研究了在不同粒度下移除选定组件对模型大小和计算开销的影响，从而在保持准确性的前提下提高模型效率。所提出的方法总称为Mamba-Shedder，在推理期间实现了高达1.4倍的加速，表明通过消除一些冗余部分，可以显著提高模型效率，同时对整体模型性能的影响较小。相关代码可在以下链接获取：this https URL。 

---
# Graph Transformers for inverse physics: reconstructing flows around arbitrary 2D airfoils 

**Title (ZH)**: 用于逆向物理的图变压器：重构任意2D翼型周围的流场 

**Authors**: Gregory Duthé, Imad Abdallah, Eleni Chatzi  

**Link**: [PDF](https://arxiv.org/pdf/2501.17081)  

**Abstract**: We introduce a Graph Transformer framework that serves as a general inverse physics engine on meshes, demonstrated through the challenging task of reconstructing aerodynamic flow fields from sparse surface measurements. While deep learning has shown promising results in forward physics simulation, inverse problems remain particularly challenging due to their ill-posed nature and the difficulty of propagating information from limited boundary observations. Our approach addresses these challenges by combining the geometric expressiveness of message-passing neural networks with the global reasoning of Transformers, enabling efficient learning of inverse mappings from boundary conditions to complete states. We evaluate this framework on a comprehensive dataset of steady-state RANS simulations around diverse airfoil geometries, where the task is to reconstruct full pressure and velocity fields from surface pressure measurements alone. The architecture achieves high reconstruction accuracy while maintaining fast inference times. We conduct experiments and provide insights into the relative importance of local geometric processing and global attention mechanisms in mesh-based inverse problems. We also find that the framework is robust to reduced sensor coverage. These results suggest that Graph Transformers can serve as effective inverse physics engines across a broader range of applications where complete system states must be reconstructed from limited boundary observations. 

**Abstract (ZH)**: 我们提出了一种图变换器框架，该框架作为一个在网格上的一般逆向物理引擎得到展示，通过从稀疏表面对应的风场重建这一具有挑战性的任务得到了验证。尽管深度学习在正向物理仿真方面展现了令人鼓舞的结果，但逆向问题仍然尤为具有挑战性，这主要是由于这些问题的病态性质以及从有限的边界观测信息传播信息的困难。我们的方法通过结合消息传递神经网络的几何表现力和Transformer的全局推理机制，解决了这些挑战，使从边界条件到完整状态的逆向映射的高效学习成为可能。我们在涉及多种翼型几何形状的稳态RANS仿真数据集上评估了此框架，任务是从表面压力测量中重建完整的压力和速度场。该架构在保持快速推理时间的同时实现了较高的重建精度。我们进行了实验并提供了对网格基逆向问题中局部几何处理和全局注意机制相对重要性的见解。此外，我们发现该框架对传感器覆盖范围减少具有鲁棒性。这些结果表明，图变换器可以在多种应用场景中作为有效的逆向物理引擎，这些应用场景都要求根据有限的边界观测信息来重建系统的完整状态。 

---
# Learning Mean Field Control on Sparse Graphs 

**Title (ZH)**: 在稀疏图上学习均场控制 

**Authors**: Christian Fabian, Kai Cui, Heinz Koeppl  

**Link**: [PDF](https://arxiv.org/pdf/2501.17079)  

**Abstract**: Large agent networks are abundant in applications and nature and pose difficult challenges in the field of multi-agent reinforcement learning (MARL) due to their computational and theoretical complexity. While graphon mean field games and their extensions provide efficient learning algorithms for dense and moderately sparse agent networks, the case of realistic sparser graphs remains largely unsolved. Thus, we propose a novel mean field control model inspired by local weak convergence to include sparse graphs such as power law networks with coefficients above two. Besides a theoretical analysis, we design scalable learning algorithms which apply to the challenging class of graph sequences with finite first moment. We compare our model and algorithms for various examples on synthetic and real world networks with mean field algorithms based on Lp graphons and graphexes. As it turns out, our approach outperforms existing methods in many examples and on various networks due to the special design aiming at an important, but so far hard to solve class of MARL problems. 

**Abstract (ZH)**: 大型代理网络在实际应用和自然中非常普遍，但在多代理强化学习（MARL）领域给提出了诸多计算和理论上的挑战。虽然图on平均场游戏及其扩展为密集和中度稀疏的代理网络提供了一种有效的学习算法，但对于现实中的更稀疏的网络图，这一问题依然难以解决。因此，我们提出了一种新的基于局部弱收敛的平均场控制模型，以包括具有系数高于2的幂律网络等稀疏网络。除了理论分析之外，我们还设计了可扩展的学习算法，适用于具有有限一阶矩的图序列。我们在合成和现实世界的网络上，将我们的模型和算法与基于Lp图on和图ex的平均场算法进行了比较。结果表明，由于特别针对解决目前难以处理的MARL问题类别，我们的方法在许多例子和不同网络中表现优于现有方法。 

---
# Induced Modularity and Community Detection for Functionally Interpretable Reinforcement Learning 

**Title (ZH)**: 诱导模ularity和社区检测在功能可解释强化学习中的应用 

**Authors**: Anna Soligo, Pietro Ferraro, David Boyle  

**Link**: [PDF](https://arxiv.org/pdf/2501.17077)  

**Abstract**: Interpretability in reinforcement learning is crucial for ensuring AI systems align with human values and fulfill the diverse related requirements including safety, robustness and fairness. Building on recent approaches to encouraging sparsity and locality in neural networks, we demonstrate how the penalisation of non-local weights leads to the emergence of functionally independent modules in the policy network of a reinforcement learning agent. To illustrate this, we demonstrate the emergence of two parallel modules for assessment of movement along the X and Y axes in a stochastic Minigrid environment. Through the novel application of community detection algorithms, we show how these modules can be automatically identified and their functional roles verified through direct intervention on the network weights prior to inference. This establishes a scalable framework for reinforcement learning interpretability through functional modularity, addressing challenges regarding the trade-off between completeness and cognitive tractability of reinforcement learning explanations. 

**Abstract (ZH)**: 强化学习中的可解释性对于确保人工智能系统与人类价值观一致并满足包括安全、鲁棒性和公平性在内的多样化相关需求至关重要。基于最近促进神经网络稀疏性和局部性的方法，我们展示了如何通过惩罚非局部权重来促使强化学习代理的策略网络出现功能性独立模块。为说明这一点，我们展示了在随机小网格环境中小模块如何实现对X轴和Y轴运动的平行评估。通过新颖地应用社区检测算法，我们表明这些模块可以自动识别，并通过干预网络权重以进行推理前的功能验证来验证其功能作用。这建立了一种通过功能性模块化实现强化学习可解释性的可扩展框架，解决了关于强化学习解释的完整性和认知可处理性之间的权衡挑战。 

---
# EdgeMLOps: Operationalizing ML models with Cumulocity IoT and thin-edge.io for Visual quality Inspection 

**Title (ZH)**: EdgeMLOps: 使用Cumulocity IoT和thin-edge.io 实现视觉质量检测中的机器学习模型运营化 

**Authors**: Kanishk Chaturvedi, Johannes Gasthuber, Mohamed Abdelaal  

**Link**: [PDF](https://arxiv.org/pdf/2501.17062)  

**Abstract**: This paper introduces EdgeMLOps, a framework leveraging Cumulocity IoT and this http URL for deploying and managing machine learning models on resource-constrained edge devices. We address the challenges of model optimization, deployment, and lifecycle management in edge environments. The framework's efficacy is demonstrated through a visual quality inspection (VQI) use case where images of assets are processed on edge devices, enabling real-time condition updates within an asset management system. Furthermore, we evaluate the performance benefits of different quantization methods, specifically static and dynamic signed-int8, on a Raspberry Pi 4, demonstrating significant inference time reductions compared to FP32 precision. Our results highlight the potential of EdgeMLOps to enable efficient and scalable AI deployments at the edge for industrial applications. 

**Abstract (ZH)**: 本文介绍了EdgeMLOps框架，该框架利用Cumulocity IoT和此链接中的资源，实现了在资源受限的边缘设备上部署和管理机器学习模型。本文解决了边缘环境中的模型优化、部署和生命周期管理的挑战。通过一个视觉质量检验（VQI）用例，展示了在边缘设备上处理资产图像，从而在资产管理系统中实现实时状态更新的有效性。此外，本文还评估了不同量化方法（特别是静态和动态-signed int8）在Raspberry Pi 4上的性能优势，显示与FP32精度相比，推理时间显著减少。研究结果突显了EdgeMLOps在工业应用中实现高效、可扩展的人工智能部署的潜力。 

---
# Synthesizing 3D Abstractions by Inverting Procedural Buildings with Transformers 

**Title (ZH)**: 使用变换器反向合成程序化建筑的3D抽象化 

**Authors**: Max Dax, Jordi Berbel, Jan Stria, Leonidas Guibas, Urs Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2501.17044)  

**Abstract**: We generate abstractions of buildings, reflecting the essential aspects of their geometry and structure, by learning to invert procedural models. We first build a dataset of abstract procedural building models paired with simulated point clouds and then learn the inverse mapping through a transformer. Given a point cloud, the trained transformer then infers the corresponding abstracted building in terms of a programmatic language description. This approach leverages expressive procedural models developed for gaming and animation, and thereby retains desirable properties such as efficient rendering of the inferred abstractions and strong priors for regularity and symmetry. Our approach achieves good reconstruction accuracy in terms of geometry and structure, as well as structurally consistent inpainting. 

**Abstract (ZH)**: 我们通过学习反向映射程序化模型来生成建筑物的抽象表示，反映其几何结构的本质方面。首先，我们构建了一个由抽象的程序化建筑模型及其模拟点云配对的数据集，然后通过变换器学习这些反向映射。给定一个点云，经过训练的变换器可以推断出相应的抽象化建筑模型，以程序化语言描述的形式呈现。该方法利用了为游戏和动画开发的表达性强的程序化模型，从而保留了诸如高效的几何结构推理和规律性、对称性等有利特性。我们的方法在几何结构和结构方面实现了较好的重建准确性，并且能够进行结构一致的补全。 

---
# Benchmarking Quantum Convolutional Neural Networks for Signal Classification in Simulated Gamma-Ray Burst Detection 

**Title (ZH)**: 模拟伽马射线暴检测中量子卷积神经网络信号分类的基准研究 

**Authors**: Farida Farsian, Nicolò Parmiggiani, Alessandro Rizzo, Gabriele Panebianco, Andrea Bulgarelli, Francesco Schillirò, Carlo Burigana, Vincenzo Cardone, Luca Cappelli, Massimo Meneghetti, Giuseppe Murante, Giuseppe Sarracino, Roberto Scaramella, Vincenzo Testa, Tiziana Trombetti  

**Link**: [PDF](https://arxiv.org/pdf/2501.17041)  

**Abstract**: This study evaluates the use of Quantum Convolutional Neural Networks (QCNNs) for identifying signals resembling Gamma-Ray Bursts (GRBs) within simulated astrophysical datasets in the form of light curves. The task addressed here focuses on distinguishing GRB-like signals from background noise in simulated Cherenkov Telescope Array Observatory (CTAO) data, the next-generation astrophysical observatory for very high-energy gamma-ray science. QCNNs, a quantum counterpart of classical Convolutional Neural Networks (CNNs), leverage quantum principles to process and analyze high-dimensional data efficiently. We implemented a hybrid quantum-classical machine learning technique using the Qiskit framework, with the QCNNs trained on a quantum simulator. Several QCNN architectures were tested, employing different encoding methods such as Data Reuploading and Amplitude encoding. Key findings include that QCNNs achieved accuracy comparable to classical CNNs, often surpassing 90\%, while using fewer parameters, potentially leading to more efficient models in terms of computational resources. A benchmark study further examined how hyperparameters like the number of qubits and encoding methods affected performance, with more qubits and advanced encoding methods generally enhancing accuracy but increasing complexity. QCNNs showed robust performance on time-series datasets, successfully detecting GRB signals with high precision. The research is a pioneering effort in applying QCNNs to astrophysics, offering insights into their potential and limitations. This work sets the stage for future investigations to fully realize the advantages of QCNNs in astrophysical data analysis. 

**Abstract (ZH)**: 本研究评价了量子卷积神经网络（QCNNs）在模拟 Astrophysical 数据集（以光曲线形式）中识别类似伽马射线暴（GRBs）信号中的应用。本研究的任务集中在区分模拟切伦科夫望远镜阵列观测站（CTAO）数据中的类似 GRB 信号与背景噪声。CTAO 是下一代用于极高能伽马射线科学的天体物理观测站。作为经典卷积神经网络（CNNs）的量子对应物，QCNNs 利用量子原理高效地处理和分析高维数据。我们使用 Qiskit 框架实现了量子-经典混合机器学习技术，并在量子模拟器上对 QCNNs 进行训练。我们测试了几种不同的 QCNN 架构，采用了不同的编码方法，如数据重上传和振幅编码。主要发现包括：QCNNs 在准确率上与经典 CNNs 相当，经常超过 90%，同时使用更少的参数，可能在计算资源方面更具高效性。进一步的基准研究表明，超参数如量子比特数量和编码方法如何影响性能，更多的量子比特和先进的编码方法通常会提高准确率，但也会增加复杂性。QCNNs 在时间序列数据集上表现出稳健的性能，成功地以高精度检测到了 GRB 信号。这项研究是将 QCNNs 应用于天文学的开创性努力，提供了它们潜在优势和限制的见解。这项工作为未来彻底了解 QCNNs 在天文数据分析中的优势奠定了基础。 

---
# Standardised schema and taxonomy for AI incident databases in critical digital infrastructure 

**Title (ZH)**: 关键数字基础设施中人工智能事故数据库的标准化模式和分类-taxonomy 

**Authors**: Avinash Agarwal, Manisha J. Nene  

**Link**: [PDF](https://arxiv.org/pdf/2501.17037)  

**Abstract**: The rapid deployment of Artificial Intelligence (AI) in critical digital infrastructure introduces significant risks, necessitating a robust framework for systematically collecting AI incident data to prevent future incidents. Existing databases lack the granularity as well as the standardized structure required for consistent data collection and analysis, impeding effective incident management. This work proposes a standardized schema and taxonomy for AI incident databases, addressing these challenges by enabling detailed and structured documentation of AI incidents across sectors. Key contributions include developing a unified schema, introducing new fields such as incident severity, causes, and harms caused, and proposing a taxonomy for classifying AI incidents in critical digital infrastructure. The proposed solution facilitates more effective incident data collection and analysis, thus supporting evidence-based policymaking, enhancing industry safety measures, and promoting transparency. This work lays the foundation for a coordinated global response to AI incidents, ensuring trust, safety, and accountability in using AI across regions. 

**Abstract (ZH)**: 人工智能（AI）在关键数字基础设施中的迅速部署带来了显著风险，因此需要一个 robust 的框架来系统地收集 AI 事件数据，以防止未来事件的发生。现有数据库缺乏所需的粒度和标准化结构，这阻碍了数据的一致收集和分析，影响有效的事件管理。本研究提出了一种标准化的数据库模式和分类体系，通过在不同领域详细和结构化地记录 AI 事件来应对这些挑战。主要贡献包括开发了统一的模式、引入了新的字段如事件严重性、原因和造成的影响，以及提出了关键数字基础设施中 AI 事件分类的分类体系。所提出的解决方案有助于更有效地收集和分析事件数据，从而支持基于证据的政策制定，增强行业安全措施，促进透明度。这项工作为全球协调应对 AI 事件奠定了基础，确保跨区域使用 AI 的可信、安全和问责。 

---
# Challenges in Ensuring AI Safety in DeepSeek-R1 Models: The Shortcomings of Reinforcement Learning Strategies 

**Title (ZH)**: 确保DeepSeek-R1模型安全性的挑战：强化学习策略的不足之处 

**Authors**: Manojkumar Parmar, Yuvaraj Govindarajulu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17030)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in reasoning, alignment, and task-specific performance. However, ensuring harmlessness in these systems remains a critical challenge, particularly in advanced models like DeepSeek-R1. This paper examines the limitations of Reinforcement Learning (RL) as the primary approach for reducing harmful outputs in DeepSeek-R1 and compares it with Supervised Fine-Tuning (SFT). While RL improves reasoning capabilities, it faces challenges such as reward hacking, generalization failures, language mixing, and high computational costs. We propose hybrid training approaches combining RL and SFT to achieve robust harmlessness reduction. Usage recommendations and future directions for deploying DeepSeek-R1 responsibly are also presented. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在推理、对齐和任务特定性能方面取得了显著进展。然而，确保这些系统不产生危害仍然是一个关键挑战，特别是在像DeepSeek-R1这样高级的模型中。本文探讨了强化学习（RL）作为减少DeepSeek-R1有害输出主要方法的局限性，并将其与监督微调（SFT）进行了比较。虽然RL能够提高推理能力，但也存在奖励劫持、泛化失败、语言混杂和高计算成本等挑战。我们提出了结合RL和SFT的混合训练方法，以实现更稳健的有害输出减少。同时，本文还提出了负责任部署DeepSeek-R1的使用建议以及未来的研究方向。 

---
# Generative quantum combinatorial optimization by means of a novel conditional generative quantum eigensolver 

**Title (ZH)**: 基于新型条件生成量子特征求解器的生成量子组合优化方法 

**Authors**: Shunya Minami, Kouhei Nakaji, Yohichi Suzuki, Alán Aspuru-Guzik, Tadashi Kadowaki  

**Link**: [PDF](https://arxiv.org/pdf/2501.16986)  

**Abstract**: Quantum computing is entering a transformative phase with the emergence of logical quantum processors, which hold the potential to tackle complex problems beyond classical capabilities. While significant progress has been made, applying quantum algorithms to real-world problems remains challenging. Hybrid quantum-classical techniques have been explored to bridge this gap, but they often face limitations in expressiveness, trainability, or scalability. In this work, we introduce conditional Generative Quantum Eigensolver (conditional-GQE), a context-aware quantum circuit generator powered by an encoder-decoder Transformer. Focusing on combinatorial optimization, we train our generator for solving problems with up to 10 qubits, exhibiting nearly perfect performance on new problems. By leveraging the high expressiveness and flexibility of classical generative models, along with an efficient preference-based training scheme, conditional-GQE provides a generalizable and scalable framework for quantum circuit generation. Our approach advances hybrid quantum-classical computing and contributes to accelerate the transition toward fault-tolerant quantum computing. 

**Abstract (ZH)**: 量子计算正进入一个变革性的阶段，随着逻辑量子处理器的出现，它们有可能解决超出经典计算能力的复杂问题。虽然在这一领域已经取得了显著进展，但将量子算法应用于实际问题仍然充满挑战。混合量子经典技术被探索以弥补这一差距，但这些技术往往在表达能力、训练能力和可扩展性方面存在限制。在本工作中，我们介绍了一种基于编码器-解码器Transformer的条件生成量子特征求解器（conditional-GQE），这是一种具有上下文感知能力的量子电路生成器。我们专注于组合优化问题，并训练生成器以解决最多10个量子比特的问题，新问题上几乎表现出完美性能。通过利用经典生成模型的高表达能力和灵活性，结合一种高效的选择性训练方案，conditional-GQE 提供了一种通用且可扩展的量子电路生成框架。我们的方法推进了混合量子经典计算并有助于加速向容错量子计算过渡。 

---
# Heterogeneity-aware Personalized Federated Learning via Adaptive Dual-Agent Reinforcement Learning 

**Title (ZH)**: 基于自适应双代理强化学习的异构适应性个性化联邦学习 

**Authors**: Xi Chen, Qin Li, Haibin Cai, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16966)  

**Abstract**: Federated Learning (FL) empowers multiple clients to collaboratively train machine learning models without sharing local data, making it highly applicable in heterogeneous Internet of Things (IoT) environments. However, intrinsic heterogeneity in clients' model architectures and computing capabilities often results in model accuracy loss and the intractable straggler problem, which significantly impairs training effectiveness. To tackle these challenges, this paper proposes a novel Heterogeneity-aware Personalized Federated Learning method, named HAPFL, via multi-level Reinforcement Learning (RL) mechanisms. HAPFL optimizes the training process by incorporating three strategic components: 1) An RL-based heterogeneous model allocation mechanism. The parameter server employs a Proximal Policy Optimization (PPO)-based RL agent to adaptively allocate appropriately sized, differentiated models to clients based on their performance, effectively mitigating performance disparities. 2) An RL-based training intensity adjustment scheme. The parameter server leverages another PPO-based RL agent to dynamically fine-tune the training intensity for each client to further enhance training efficiency and reduce straggling latency. 3) A knowledge distillation-based mutual learning mechanism. Each client deploys both a heterogeneous local model and a homogeneous lightweight model named LiteModel, where these models undergo mutual learning through knowledge distillation. This uniform LiteModel plays a pivotal role in aggregating and sharing global knowledge, significantly enhancing the effectiveness of personalized local training. Experimental results across multiple benchmark datasets demonstrate that HAPFL not only achieves high accuracy but also substantially reduces the overall training time by 20.9%-40.4% and decreases straggling latency by 19.0%-48.0% compared to existing solutions. 

**Abstract (ZH)**: 联邦学习（FL）使多个客户端能够无需共享本地数据即可协作训练机器学习模型，使其在异构物联网（IoT）环境中具有高度适用性。然而，客户端模型架构和计算能力的固有异质性往往会导致模型准确率下降和不可解决的延迟节点问题，这显著影响了训练效果。为应对这些挑战，本文提出了一个新颖的异质感知个性化联邦学习方法（HAPFL），通过多层次强化学习（RL）机制实现。HAPFL通过引入三个战略组件优化了训练过程：1) 基于RL的异质模型分配机制。参数服务器采用基于Proximal Policy Optimization (PPO) 的RL代理，根据客户端的性能动态分配合适大小和差异化的模型，有效地缓解了性能差异。2) 基于RL的训练强度调整方案。参数服务器利用另一个基于PPO 的RL代理动态调整每个客户端的训练强度，进一步提高训练效率并减少延迟。3) 基于知识蒸馏的互助学习机制。每个客户端部署一个异质本地模型和一个同质轻量级模型LiteModel，这些模型通过知识蒸馏进行互助学习。这个统一的LiteModel在聚合和共享全球知识方面发挥着关键作用，显著提升了个性化本地训练的效果。在多个基准数据集上的实验结果表明，HAPFL不仅能够实现高准确率，还能将整体训练时间减少20.9%-40.4%，将延迟减少19.0%-48.0%，相比于现有解决方案有显著优势。 

---
# Multiple Abstraction Level Retrieve Augment Generation 

**Title (ZH)**: 多抽象层次检索增强生成 

**Authors**: Zheng Zheng, Xinyi Ni, Pengyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2501.16952)  

**Abstract**: A Retrieval-Augmented Generation (RAG) model powered by a large language model (LLM) provides a faster and more cost-effective solution for adapting to new data and knowledge. It also delivers more specialized responses compared to pre-trained LLMs. However, most existing approaches rely on retrieving prefix-sized chunks as references to support question-answering (Q/A). This approach is often deployed to address information needs at a single level of abstraction, as it struggles to generate answers across multiple levels of abstraction. In an RAG setting, while LLMs can summarize and answer questions effectively when provided with sufficient details, retrieving excessive information often leads to the 'lost in the middle' problem and exceeds token limitations. We propose a novel RAG approach that uses chunks of multiple abstraction levels (MAL), including multi-sentence-level, paragraph-level, section-level, and document-level. The effectiveness of our approach is demonstrated in an under-explored scientific domain of Glycoscience. Compared to traditional single-level RAG approaches, our approach improves AI evaluated answer correctness of Q/A by 25.739\% on Glyco-related papers. 

**Abstract (ZH)**: 由大规模语言模型（LLM）驱动的检索增强生成（RAG）模型能够更快、更经济地适应新数据和知识。与预训练的语言模型相比，它还能提供更具专业性的回答。然而，现有大多数方法依赖于获取前缀大小的片段作为问答（Q/A）的支持参考。这种方法通常仅在一个抽象层次上解决信息需求的问题，因为它难以在多个抽象层次上生成答案。在RAG设置中，当LLM提供足够详细的信息时，它可以有效地总结和回答问题，但检索过多的信息往往会引发“中间迷失”问题，并且超过标记限制。我们提出了一种新的RAG方法，使用多个抽象层次的片段（MAL），包括多句级、段落级、节级和文档级。我们的方法在未充分探索的科学领域——糖科学中得到了验证。与传统的单一抽象层次的RAG方法相比，我们的方法在糖相关论文中的问题回答准确性的AI评估上提高了25.739%。 

---
# ToolFactory: Automating Tool Generation by Leveraging LLM to Understand REST API Documentations 

**Title (ZH)**: ToolFactory：通过利用大型语言模型理解REST API文档来自动生成工具 

**Authors**: Xinyi Ni, Qiuyang Wang, Yukun Zhang, Pengyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2501.16945)  

**Abstract**: LLM-based tool agents offer natural language interfaces, enabling users to seamlessly interact with computing services. While REST APIs are valuable resources for building such agents, they must first be transformed into AI-compatible tools. Automatically generating AI-compatible tools from REST API documents can greatly streamline tool agent development and minimize user learning curves. However, API documentation often suffers from a lack of standardization, inconsistent schemas, and incomplete information. To address these issues, we developed \textbf{ToolFactory}, an open-source pipeline for automating tool generation from unstructured API documents. To enhance the reliability of the developed tools, we implemented an evaluation method to diagnose errors. Furthermore, we built a knowledge base of verified tools, which we leveraged to infer missing information from poorly documented APIs. We developed the API Extraction Benchmark, comprising 167 API documents and 744 endpoints in various formats, and designed a JSON schema to annotate them. This annotated dataset was utilized to train and validate ToolFactory. The experimental results highlight the effectiveness of ToolFactory. We also demonstrated ToolFactory by creating a domain-specific AI agent for glycomaterials research. ToolFactory exhibits significant potential for facilitating the seamless integration of scientific REST APIs into AI workflows. 

**Abstract (ZH)**: 基于LLM的工具代理提供自然语言界面，使用户能够无缝与计算服务进行交互。虽然REST API是构建此类代理的宝贵资源，但必须首先将它们转换为AI兼容的工具。从REST API文档自动生成AI兼容的工具可以大大简化工具代理的开发过程，并尽量减少用户的学习曲线。然而，API文档往往缺乏标准化，存在不一致的模式和不完整的信息。为了解决这些问题，我们开发了\textbf{ToolFactory}——一个开源流水线，用于从非结构化的API文档自动生成工具。为了提高开发工具的可靠性，我们实施了一种评估方法来诊断错误。此外，我们构建了一个经过验证的工具知识库，从中可以推断出未充分文档化的API所缺失的信息。我们开发了API提取基准数据集，包含167份API文档和744个不同格式的端点，并设计了一个JSON模式对其进行标注。该标注数据集被用于训练和验证ToolFactory。实验结果突显了ToolFactory的有效性。我们还通过为糖材料研究创建一个特定领域的AI代理来展示了ToolFactory的应用。ToolFactory在促进科学REST API与AI工作流的无缝集成方面展现出巨大的潜力。 

---
# Exact Computation of Any-Order Shapley Interactions for Graph Neural Networks 

**Title (ZH)**: 任意阶Shapley相互作用的精确计算：用于图神经网络 

**Authors**: Fabian Fumagalli, Maximilian Muschalik, Paolo Frazzetto, Janine Strotherm, Luca Hermes, Alessandro Sperduti, Eyke Hüllermeier, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2501.16944)  

**Abstract**: Albeit the ubiquitous use of Graph Neural Networks (GNNs) in machine learning (ML) prediction tasks involving graph-structured data, their interpretability remains challenging. In explainable artificial intelligence (XAI), the Shapley Value (SV) is the predominant method to quantify contributions of individual features to a ML model's output. Addressing the limitations of SVs in complex prediction models, Shapley Interactions (SIs) extend the SV to groups of features. In this work, we explain single graph predictions of GNNs with SIs that quantify node contributions and interactions among multiple nodes. By exploiting the GNN architecture, we show that the structure of interactions in node embeddings are preserved for graph prediction. As a result, the exponential complexity of SIs depends only on the receptive fields, i.e. the message-passing ranges determined by the connectivity of the graph and the number of convolutional layers. Based on our theoretical results, we introduce GraphSHAP-IQ, an efficient approach to compute any-order SIs exactly. GraphSHAP-IQ is applicable to popular message passing techniques in conjunction with a linear global pooling and output layer. We showcase that GraphSHAP-IQ substantially reduces the exponential complexity of computing exact SIs on multiple benchmark datasets. Beyond exact computation, we evaluate GraphSHAP-IQ's approximation of SIs on popular GNN architectures and compare with existing baselines. Lastly, we visualize SIs of real-world water distribution networks and molecule structures using a SI-Graph. 

**Abstract (ZH)**: 尽管图神经网络（GNNs）在涉及图结构数据的机器学习（ML）预测任务中得到了广泛应用，但在解释性人工智能（XAI）领域，GNNs 的可解释性仍然面临挑战。在XAI 中，Shapley 值（SV）是量化单个特征对ML 模型输出贡献的主流方法。然而，在复杂预测模型中，SV 的局限性明显，因此Shapley 互作（SIs）将SV 扩展到了特征组。在本研究中，我们通过SIs 解释了GNN 的单一图预测，量化了节点贡献及其相互作用。借助GNN 架构，我们展示了节点嵌入中交互结构的保持对于图预测的重要性。因此，SIs 的指数复杂性仅依赖于感知域，即由图的连接性和卷积层数量确定的消息传递范围。基于我们理论上的结果，我们引入了GraphSHAP-IQ，这是一种高效的计算任何阶次SIs 的方法。GraphSHAP-IQ 可以与线性全局池化和输出层结合使用，适用于流行的传递消息技术。我们展示了GraphSHAP-IQ 在多个基准数据集上计算精确SIs 的指数复杂性显著减少的能力。在此基础上，我们评估了GraphSHAP-IQ 对流行GNN 架构中SIs 的近似计算，并与现有基线进行了比较。最后，我们使用SI-Graph 技术可视化实际供水网络和分子结构中SIs 的分布。 

---
# TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models 

**Title (ZH)**: TAID：时间自适应内插_distillation在语言模型中高效知识迁移中的应用 

**Authors**: Makoto Shing, Kou Misaki, Han Bao, Sho Yokoi, Takuya Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2501.16937)  

**Abstract**: Causal language models have demonstrated remarkable capabilities, but their size poses significant challenges for deployment in resource-constrained environments. Knowledge distillation, a widely-used technique for transferring knowledge from a large teacher model to a small student model, presents a promising approach for model compression. A significant remaining issue lies in the major differences between teacher and student models, namely the substantial capacity gap, mode averaging, and mode collapse, which pose barriers during distillation. To address these issues, we introduce $\textit{Temporally Adaptive Interpolated Distillation (TAID)}$, a novel knowledge distillation approach that dynamically interpolates student and teacher distributions through an adaptive intermediate distribution, gradually shifting from the student's initial distribution towards the teacher's distribution. We provide a theoretical analysis demonstrating TAID's ability to prevent mode collapse and empirically show its effectiveness in addressing the capacity gap while balancing mode averaging and mode collapse. Our comprehensive experiments demonstrate TAID's superior performance across various model sizes and architectures in both instruction tuning and pre-training scenarios. Furthermore, we showcase TAID's practical impact by developing two state-of-the-art compact foundation models: $\texttt{TAID-LLM-1.5B}$ for language tasks and $\texttt{TAID-VLM-2B}$ for vision-language tasks. These results demonstrate TAID's effectiveness in creating high-performing and efficient models, advancing the development of more accessible AI technologies. 

**Abstract (ZH)**: 因果语言模型展现出了显著的能力，但其规模也带来了在资源有限环境中部署的重大挑战。知识蒸馏作为一种广泛使用的从大型教师模型向小型学生模型转移知识的技术，为模型压缩提供了有前景的方法。一个重要的遗留问题是教师模型和学生模型之间存在显著差异，特别是容量差距、模式平均和模式崩溃等重大差异，这在知识蒸馏过程中构成了障碍。为了解决这些问题，我们提出了**时间自适应插值蒸馏（TAID）**，这是一种新颖的知识蒸馏方法，通过自适应中间分布动态地插值学生和教师的分布，逐步从学生初试分布向教师的分布转变。我们提供理论分析展示了TAID防止模式崩溃的能力，并通过实验证明了其在解决容量差距的同时平衡模式平均和模式崩溃方面的有效性。我们的综合实验表明，TAID在各种模型大小和架构下，无论是指令调优还是预训练场景中，都表现出优越的性能。此外，我们进一步展示了TAID的实用影响，开发了两个最先进的紧凑型基础模型：用于语言任务的**TAID-LLM-1.5B** 和用于视觉-语言任务的**TAID-VLM-2B**。这些结果表明，TAID能够有效地创建高性能和高效的模型，促进了更易获取的人工智能技术的发展。 

---
# RDMM: Fine-Tuned LLM Models for On-Device Robotic Decision Making with Enhanced Contextual Awareness in Specific Domains 

**Title (ZH)**: RDMM：针对特定领域增强上下文意识的嵌入式机器人决策 Fine-tuned 大型语言模型 

**Authors**: Shady Nasrat, Myungsu Kim, Seonil Lee, Jiho Lee, Yeoncheol Jang, Seung-joon Yi  

**Link**: [PDF](https://arxiv.org/pdf/2501.16899)  

**Abstract**: Large language models (LLMs) represent a significant advancement in integrating physical robots with AI-driven systems. We showcase the capabilities of our framework within the context of the real-world household competition. This research introduces a framework that utilizes RDMM (Robotics Decision-Making Models), which possess the capacity for decision-making within domain-specific contexts, as well as an awareness of their personal knowledge and capabilities. The framework leverages information to enhance the autonomous decision-making of the system. In contrast to other approaches, our focus is on real-time, on-device solutions, successfully operating on hardware with as little as 8GB of memory. Our framework incorporates visual perception models equipping robots with understanding of their environment. Additionally, the framework has integrated real-time speech recognition capabilities, thus enhancing the human-robot interaction experience. Experimental results demonstrate that the RDMM framework can plan with an 93\% accuracy. Furthermore, we introduce a new dataset consisting of 27k planning instances, as well as 1.3k text-image annotated samples derived from the competition. The framework, benchmarks, datasets, and models developed in this work are publicly available on our GitHub repository at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）代表了将物理机器人与AI驱动系统整合的重要进步。我们在实际家庭环境挑战的背景下展示了该框架的能力。本研究介绍了一种利用RDMM（机器人决策模型）的框架，这些模型能够在其特定领域内进行决策，同时具有对自己知识和能力的自我意识。该框架利用信息来增强系统的自主决策能力。与其它方法不同，我们的重点在于实时且在设备端的解决方案，成功地在内存仅为8GB的硬件上运行。该框架集成了视觉感知模型，使机器人能够理解其环境。此外，框架还集成了实时语音识别能力，从而增强了人与机器人之间的交互体验。实验结果表明，RDMM框架的规划准确率为93%。此外，我们还引入了一个新的数据集，包含27,000个规划实例，以及1,300个包含文本和图像标注的样本，这些数据源于比赛。本研究开发的框架、基准测试、数据集和模型已公开发布在我们的GitHub仓库中，网址为 [此链接]。

请注意，将“此 https URL”替换为实际的GitHub仓库链接。 

---
# Extending Information Bottleneck Attribution to Video Sequences 

**Title (ZH)**: 将信息瓶颈归因方法扩展到视频序列 

**Authors**: Veronika Solopova, Lucas Schmidt, Dorothea Kolossa  

**Link**: [PDF](https://arxiv.org/pdf/2501.16889)  

**Abstract**: We introduce VIBA, a novel approach for explainable video classification by adapting Information Bottlenecks for Attribution (IBA) to video sequences. While most traditional explainability methods are designed for image models, our IBA framework addresses the need for explainability in temporal models used for video analysis. To demonstrate its effectiveness, we apply VIBA to video deepfake detection, testing it on two architectures: the Xception model for spatial features and a VGG11-based model for capturing motion dynamics through optical flow. Using a custom dataset that reflects recent deepfake generation techniques, we adapt IBA to create relevance and optical flow maps, visually highlighting manipulated regions and motion inconsistencies. Our results show that VIBA generates temporally and spatially consistent explanations, which align closely with human annotations, thus providing interpretability for video classification and particularly for deepfake detection. 

**Abstract (ZH)**: 我们将介绍VIBA，这是一种通过将信息瓶颈归因（Information Bottlenecks for Attribution, IBA）方法适应视频序列来实现可解释视频分类的新方法。尽管大多数传统的可解释性方法是为图像模型设计的，但我们的IBA框架旨在解决用于视频分析的时间模型的可解释性需求。为了证明其有效性，我们将VIBA应用于视频深度假信息检测，并在两种架构上进行了测试：使用Xception模型提取空间特征，以及基于VGG11的模型通过光流捕捉运动动力学。利用一个反映近期深度假信息生成技术的自定义数据集，我们将IBA适应以生成相关性图和光流图，可视化地突出显示被篡改的区域和运动不一致性。实验结果显示，VIBA生成的时间和空间上一致的解释与人类注释高度一致，从而为视频分类和特别是深度假信息检测提供了可解释性。 

---
# Irony Detection, Reasoning and Understanding in Zero-shot Learning 

**Title (ZH)**: 零样本学习中的讽刺检测、推理与理解 

**Authors**: Peiling Yi, Yuhan Xia  

**Link**: [PDF](https://arxiv.org/pdf/2501.16884)  

**Abstract**: Irony is a powerful figurative language (FL) on social media that can potentially mislead various NLP tasks, such as recommendation systems, misinformation checks, and sentiment analysis. Understanding the implicit meaning of this kind of subtle language is essential to mitigate irony's negative impact on NLP tasks. However, building models to understand irony presents a unique set of challenges, because irony is a complex form of language that often relies on context, tone, and subtle cues to convey meaning that is opposite or different from the literal interpretation. Large language models, such as ChatGPT, are increasingly able to capture implicit and contextual information. In this study, we investigate the generalization, reasoning and understanding ability of ChatGPT on irony detection across six different genre irony detection datasets. Our findings suggest that ChatGPT appears to show an enhanced language understanding and reasoning ability. But it needs to be very careful in prompt engineering design. Thus, we propose a prompt engineering design framework IDADP to achieve higher irony detection accuracy, improved understanding of irony, and more effective explanations compared to other state-of-the-art ChatGPT zero-shot approaches. And ascertain via experiments that the practice generated under the framework is likely to be the promised solution to resolve the generalization issues of LLMs. 

**Abstract (ZH)**: irony 是社交媒体上一种强大的修辞语言（FL），可能会误导各种自然语言处理（NLP）任务，如推荐系统、错误信息检查和情绪分析。理解这种微妙语言的隐含意义对于减轻 irony 对 NLP 任务的负面影响至关重要。然而，构建用于理解 irony 的模型面临着独特的挑战，因为 irony 是一种复杂的语言形式，通常依赖于上下文、语气和细微的暗示来传达与其字面意义相反或不同的含义。大型语言模型（如 ChatGPT）越来越能够捕捉到隐含和上下文信息。在本研究中，我们探讨了 ChatGPT 在六种不同类型的 irony 检测数据集中的泛化能力、推理能力和理解能力。我们的研究发现表明，ChatGPT 显示出增强的语言理解和推理能力，但需要在提示工程设计中非常小心。因此，我们提出了一种提示工程设计框架 IDADP（IDA Design Pattern），以实现更高的 irony 检测准确性、更深入的 irony 理解和更有效的解释，相比其他最先进的 ChatGPT 零样本方法。并通过实验验证，在框架下生成的实践可能是解决大型语言模型泛化问题的承诺解决方案。 

---
# Misspellings in Natural Language Processing: A survey 

**Title (ZH)**: 自然语言处理中的拼写错误：一个综述 

**Authors**: Gianluca Sperduti, Alejandro Moreo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16836)  

**Abstract**: This survey provides an overview of the challenges of misspellings in natural language processing (NLP). While often unintentional, misspellings have become ubiquitous in digital communication, especially with the proliferation of Web 2.0, user-generated content, and informal text mediums such as social media, blogs, and forums. Even if humans can generally interpret misspelled text, NLP models frequently struggle to handle it: this causes a decline in performance in common tasks like text classification and machine translation. In this paper, we reconstruct a history of misspellings as a scientific problem. We then discuss the latest advancements to address the challenge of misspellings in NLP. Main strategies to mitigate the effect of misspellings include data augmentation, double step, character-order agnostic, and tuple-based methods, among others. This survey also examines dedicated data challenges and competitions to spur progress in the field. Critical safety and ethical concerns are also examined, for example, the voluntary use of misspellings to inject malicious messages and hate speech on social networks. Furthermore, the survey explores psycholinguistic perspectives on how humans process misspellings, potentially informing innovative computational techniques for text normalization and representation. Finally, the misspelling-related challenges and opportunities associated with modern large language models are also analyzed, including benchmarks, datasets, and performances of the most prominent language models against misspellings. This survey aims to be an exhaustive resource for researchers seeking to mitigate the impact of misspellings in the rapidly evolving landscape of NLP. 

**Abstract (ZH)**: 本文综述了自然语言处理（NLP）中错词挑战的现状。虽然错词往往是由无意造成的，但在Web 2.0、用户生成内容以及社交媒体、博客和论坛等非正式文字媒介的普及下，错词在数字通信中已经变得无处不在。即使人类通常能够理解错词文本，但在NLP模型中，处理错词仍然是一个难题：这会导致文本分类和机器翻译等常见任务性能下降。本文首先重构了错词作为科学问题的历史。接着，讨论了最新技术进步以应对NLP中的错词挑战。主要缓解错词影响的策略包括数据增强、双阶段方法、字符顺序无偏方法以及基于元组的方法等。本文还探讨了专门的数据挑战和竞赛，以推动该领域的进步。本文还分析了关键的安全和伦理问题，例如在网络社交中自愿使用错词注入恶意消息和仇恨言论。此外，本文还探讨了心理语言学视角下人类如何处理错词，这可能会为文本规范化和表示提供创新性的计算方法。最后，本文分析了现代大型语言模型中与错词相关的挑战和机遇，包括基准测试、数据集以及主流语言模型在面对错词时的表现。本文旨在为致力于减轻NLP快速演变环境中错词影响的研究人员提供一个详尽的资源。 

---
# DIRIGENt: End-To-End Robotic Imitation of Human Demonstrations Based on a Diffusion Model 

**Title (ZH)**: DIRIGENt：基于扩散模型的端到端人类演示模仿机器人技术 

**Authors**: Josua Spisak, Matthias Kerzel, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2501.16800)  

**Abstract**: There has been substantial progress in humanoid robots, with new skills continuously being taught, ranging from navigation to manipulation. While these abilities may seem impressive, the teaching methods often remain inefficient. To enhance the process of teaching robots, we propose leveraging a mechanism effectively used by humans: teaching by demonstrating. In this paper, we introduce DIRIGENt (DIrect Robotic Imitation GENeration model), a novel end-to-end diffusion approach that directly generates joint values from observing human demonstrations, enabling a robot to imitate these actions without any existing mapping between it and humans. We create a dataset in which humans imitate a robot and then use this collected data to train a diffusion model that enables a robot to imitate humans. The following three aspects are the core of our contribution. First is our novel dataset with natural pairs between human and robot poses, allowing our approach to imitate humans accurately despite the gap between their anatomies. Second, the diffusion input to our model alleviates the challenge of redundant joint configurations, limiting the search space. And finally, our end-to-end architecture from perception to action leads to an improved learning capability. Through our experimental analysis, we show that combining these three aspects allows DIRIGENt to outperform existing state-of-the-art approaches in the field of generating joint values from RGB images. 

**Abstract (ZH)**: 在人形机器人领域取得了显著进展，新技能不断被引入，涵盖导航和操作等方面。尽管这些能力看似令人印象深刻，但教学方法往往仍不够高效。为了提高机器人学习过程的效率，我们提出了利用有效的人类机制——示范教学。在本文中，我们介绍了一种新颖的端到端扩散模型——DIRIGENt（直接机器人模仿生成模型），该模型可以直接从人类示范中生成关节值，使机器人能够模仿这些动作，而无需预先存在的人类与机器人的映射。我们创建了一个数据集，在该数据集中，人类模仿机器人动作，然后利用这些收集的数据来训练一个扩散模型，使机器人能够模仿人类。以下是我们的主要贡献的三个核心方面。首先，我们创建了一个新颖的数据集，该数据集包含了人类和机器人姿势之间的自然配对，允许我们的方法即使存在人类和机器人解剖结构差异的情况下也能准确地模仿人类。其次，输入到我们模型中的扩散过程克服了冗余关节配置的挑战，从而减小了搜索空间。最后，从感知到行动的端到端架构提高了学习能力。通过我们的实验分析，我们展示了将这三个方面结合起来能够使DIRIGENt在从RGB图像生成关节值方面优于现有最先进的方法。 

---
# A Stochastic Dynamical Theory of LLM Self-Adversariality: Modeling Severity Drift as a Critical Process 

**Title (ZH)**: LLM自我对抗性的一种随机动力学理论：将严重性漂移建模为一个关键过程 

**Authors**: Jack David Carson  

**Link**: [PDF](https://arxiv.org/pdf/2501.16783)  

**Abstract**: This paper introduces a continuous-time stochastic dynamical framework for understanding how large language models (LLMs) may self-amplify latent biases or toxicity through their own chain-of-thought reasoning. The model posits an instantaneous "severity" variable $x(t) \in [0,1]$ evolving under a stochastic differential equation (SDE) with a drift term $\mu(x)$ and diffusion $\sigma(x)$. Crucially, such a process can be consistently analyzed via the Fokker--Planck approach if each incremental step behaves nearly Markovian in severity space. The analysis investigates critical phenomena, showing that certain parameter regimes create phase transitions from subcritical (self-correcting) to supercritical (runaway severity). The paper derives stationary distributions, first-passage times to harmful thresholds, and scaling laws near critical points. Finally, it highlights implications for agents and extended LLM reasoning models: in principle, these equations might serve as a basis for formal verification of whether a model remains stable or propagates bias over repeated inferences. 

**Abstract (ZH)**: 本文介绍了一个连续时间的随机动力学框架，用于理解大型语言模型（LLMs）如何通过自己的链式思考推理自我放大潜在偏见或毒性。该模型假设存在一个瞬时“严重性”变量 \(x(t) \in [0,1]\)，它在随机微分方程（SDE）中演化，含有漂移项 \(\mu(x)\) 和扩散项 \(\sigma(x)\)。关键的是，如果每次增量步骤在严重性空间中几乎表现出马尔可夫性，这样的过程可以通过福克-Planck方法进行一致的分析。分析探讨了关键现象，表明某些参数范围会导致从亚临界（自我纠正）到超临界（严重性失控）的相变。本文推导出了稳态分布、首次达到有害阈值的时间，以及关键点附近的标度定律。最后，本文强调了对代理和扩展的LLM推理模型的潜在影响：原则上，这些方程可能作为验证模型是否在重复推理中保持稳定或传播偏见的基础。 

---
# FlexMotion: Lightweight, Physics-Aware, and Controllable Human Motion Generation 

**Title (ZH)**: FlexMotion：轻量级、物理意识强且可控制的人体运动生成 

**Authors**: Arvin Tashakori, Arash Tashakori, Gongbo Yang, Z. Jane Wang, Peyman Servati  

**Link**: [PDF](https://arxiv.org/pdf/2501.16778)  

**Abstract**: Lightweight, controllable, and physically plausible human motion synthesis is crucial for animation, virtual reality, robotics, and human-computer interaction applications. Existing methods often compromise between computational efficiency, physical realism, or spatial controllability. We propose FlexMotion, a novel framework that leverages a computationally lightweight diffusion model operating in the latent space, eliminating the need for physics simulators and enabling fast and efficient training. FlexMotion employs a multimodal pre-trained Transformer encoder-decoder, integrating joint locations, contact forces, joint actuations and muscle activations to ensure the physical plausibility of the generated motions. FlexMotion also introduces a plug-and-play module, which adds spatial controllability over a range of motion parameters (e.g., joint locations, joint actuations, contact forces, and muscle activations). Our framework achieves realistic motion generation with improved efficiency and control, setting a new benchmark for human motion synthesis. We evaluate FlexMotion on extended datasets and demonstrate its superior performance in terms of realism, physical plausibility, and controllability. 

**Abstract (ZH)**: 轻量级、可控且物理上合理的真人动作合成对于动画、虚拟现实、机器人技术和人机交互应用至关重要。现有方法常常在计算效率、物理逼真度或空间可控性之间进行权衡。我们提出了一种名为 FlexMotion 的新型框架，该框架利用在潜在空间中操作的轻量级扩散模型，消除了对物理模拟器的依赖，并实现了快速高效的训练。FlexMotion 融合了多模态预训练Transformer编码器-解码器，整合了关节位置、接触力、关节驱动和肌肉激活等信息，确保生成动作的物理合理性。FlexMotion 还引入了一个即插即用模块，增强了在一系列动作参数（如关节位置、关节驱动、接触力和肌肉激活）上的空间可控性。我们的框架在效率和控制方面实现了逼真的动作生成，并建立了真人动作合成的新基准。我们在扩展数据集上评估了 FlexMotion，并展示了其在逼真度、物理合理性以及可控性方面的优越性能。 

---
# Overcoming Semantic Dilution in Transformer-Based Next Frame Prediction 

**Title (ZH)**: 基于变压器的下一帧预测中克服语义稀释问题 

**Authors**: Hy Nguyen, Srikanth Thudumu, Hung Du, Rajesh Vasa, Kon Mouzakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.16753)  

**Abstract**: Next-frame prediction in videos is crucial for applications such as autonomous driving, object tracking, and motion prediction. The primary challenge in next-frame prediction lies in effectively capturing and processing both spatial and temporal information from previous video sequences. The transformer architecture, known for its prowess in handling sequence data, has made remarkable progress in this domain. However, transformer-based next-frame prediction models face notable issues: (a) The multi-head self-attention (MHSA) mechanism requires the input embedding to be split into $N$ chunks, where $N$ is the number of heads. Each segment captures only a fraction of the original embeddings information, which distorts the representation of the embedding in the latent space, resulting in a semantic dilution problem; (b) These models predict the embeddings of the next frames rather than the frames themselves, but the loss function based on the errors of the reconstructed frames, not the predicted embeddings -- this creates a discrepancy between the training objective and the model output. We propose a Semantic Concentration Multi-Head Self-Attention (SCMHSA) architecture, which effectively mitigates semantic dilution in transformer-based next-frame prediction. Additionally, we introduce a loss function that optimizes SCMHSA in the latent space, aligning the training objective more closely with the model output. Our method demonstrates superior performance compared to the original transformer-based predictors. 

**Abstract (ZH)**: 视频的下一帧预测对于自动驾驶、物体跟踪和运动预测等应用至关重要。在下一帧预测中，主要挑战在于有效捕捉和处理之前视频序列中的空间和时间信息。基于序列数据处理能力而著称的转换器架构已经在这一领域取得了显著进展。然而，基于转换器的下一帧预测模型存在一些明显的问题：(a) 多头自注意力（MHSA）机制要求将输入嵌入分成 $N$ 个片段，其中 $N$ 是头的数量。每个片段只捕捉原始嵌入信息的一部分，导致在潜在空间中嵌入表示的失真，从而引起语义稀释问题；(b) 这些模型预测的是下一帧的嵌入而不是实际的帧，并且基于重建帧的误差来构建损失函数而非预测的嵌入，这在训练目标和模型输出之间造成了一定的不一致性。我们提出了一种语义集中多头自注意力（SCMHSA）架构，有效地缓解了基于转换器的下一帧预测中的语义稀释问题。此外，我们引入了一种损失函数，以优化SCMHSA在潜在空间中的性能，从而更紧密地将训练目标与模型输出对齐。我们的方法在性能上显著优于原始的基于转换器的预测器。 

---
# DebugAgent: Efficient and Interpretable Error Slice Discovery for Comprehensive Model Debugging 

**Title (ZH)**: DebugAgent: 高效可解释的错误切片发现方法，用于全面模型调试 

**Authors**: Muxi Chen, Chenchen Zhao, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16751)  

**Abstract**: Despite the significant success of deep learning models in computer vision, they often exhibit systematic failures on specific data subsets, known as error slices. Identifying and mitigating these error slices is crucial to enhancing model robustness and reliability in real-world scenarios. In this paper, we introduce DebugAgent, an automated framework for error slice discovery and model repair. DebugAgent first generates task-specific visual attributes to highlight instances prone to errors through an interpretable and structured process. It then employs an efficient slice enumeration algorithm to systematically identify error slices, overcoming the combinatorial challenges that arise during slice exploration. Additionally, DebugAgent extends its capabilities by predicting error slices beyond the validation set, addressing a key limitation of prior approaches. Extensive experiments across multiple domains, including image classification, pose estimation, and object detection - show that DebugAgent not only improves the coherence and precision of identified error slices but also significantly enhances the model repair capabilities. 

**Abstract (ZH)**: 尽管深度学习模型在计算机视觉领域取得了显著成功，但它们在特定数据子集上往往会表现出系统性的失败，这些失败通常被称为错误切片。识别并缓解这些错误切片对于提高模型在实际场景中的鲁棒性和可靠性至关重要。本文介绍了一种名为DebugAgent的自动化框架，该框架用于错误切片的发现和模型修复。DebugAgent首先生成特定任务的视觉属性，通过可解释且结构化的流程突出显示容易出错的实例。然后，它使用高效的切片枚举算法系统地识别错误切片，从而克服了切片探索过程中出现的组合难题。此外，DebugAgent进一步增强了其能力，能够预测超出验证集范围的错误切片，解决了先前方法的关键局限性。在图像分类、姿态估计和对象检测等多个领域的广泛实验表明，DebugAgent不仅提高了错误切片的准确性和连贯性，而且还显著增强了模型修复能力。 

---
# LLM Assisted Anomaly Detection Service for Site Reliability Engineers: Enhancing Cloud Infrastructure Resilience 

**Title (ZH)**: LLM辅助异常检测服务：增强现场可靠性工程师的云基础设施韧性 

**Authors**: Nimesh Jha, Shuxin Lin, Srideepika Jayaraman, Kyle Frohling, Christodoulos Constantinides, Dhaval Patel  

**Link**: [PDF](https://arxiv.org/pdf/2501.16744)  

**Abstract**: This paper introduces a scalable Anomaly Detection Service with a generalizable API tailored for industrial time-series data, designed to assist Site Reliability Engineers (SREs) in managing cloud infrastructure. The service enables efficient anomaly detection in complex data streams, supporting proactive identification and resolution of issues. Furthermore, it presents an innovative approach to anomaly modeling in cloud infrastructure by utilizing Large Language Models (LLMs) to understand key components, their failure modes, and behaviors. A suite of algorithms for detecting anomalies is offered in univariate and multivariate time series data, including regression-based, mixture-model-based, and semi-supervised approaches. We provide insights into the usage patterns of the service, with over 500 users and 200,000 API calls in a year. The service has been successfully applied in various industrial settings, including IoT-based AI applications. We have also evaluated our system on public anomaly benchmarks to show its effectiveness. By leveraging it, SREs can proactively identify potential issues before they escalate, reducing downtime and improving response times to incidents, ultimately enhancing the overall customer experience. We plan to extend the system to include time series foundation models, enabling zero-shot anomaly detection capabilities. 

**Abstract (ZH)**: 本文介绍了一种适用于工业时间序列数据的可扩展异常检测服务，该服务配备了一种通用的API，旨在协助系统可靠性工程师（SREs）管理云基础设施。该服务能够高效地在复杂数据流中检测异常，并支持问题的主动识别与解决。此外，本文还介绍了利用大语言模型（LLMs）理解关键组件、故障模式和行为的一种创新异常建模方法。该服务提供了针对单变量和多变量时间序列数据的一系列异常检测算法，包括基于回归、混合模型和半监督的方法。我们提供了关于服务使用模式的见解，数据显示该服务在过去一年中有超过500名用户和200,000次API调用。该服务已在各种工业环境中成功应用，包括基于物联网的AI应用程序。我们也对公共异常检测基准进行了评估，以展示其效果。通过利用该服务，SREs可以在问题升级之前主动识别潜在问题，从而减少停机时间和提高对事件的响应速度，最终提升整体客户体验。我们计划扩展该系统，使其能够包括时间序列基础模型，实现零样本异常检测能力。 

---
# Efficient Knowledge Distillation of SAM for Medical Image Segmentation 

**Title (ZH)**: SAM模式下的高效知识蒸馏在医学图像分割中的应用 

**Authors**: Kunal Dasharath Patil, Gowthamaan Palani, Ganapathy Krishnamurthi  

**Link**: [PDF](https://arxiv.org/pdf/2501.16740)  

**Abstract**: The Segment Anything Model (SAM) has set a new standard in interactive image segmentation, offering robust performance across various tasks. However, its significant computational requirements limit its deployment in real-time or resource-constrained environments. To address these challenges, we propose a novel knowledge distillation approach, KD SAM, which incorporates both encoder and decoder optimization through a combination of Mean Squared Error (MSE) and Perceptual Loss. This dual-loss framework captures structural and semantic features, enabling the student model to maintain high segmentation accuracy while reducing computational complexity. Based on the model evaluation on datasets, including Kvasir-SEG, ISIC 2017, Fetal Head Ultrasound, and Breast Ultrasound, we demonstrate that KD SAM achieves comparable or superior performance to the baseline models, with significantly fewer parameters. KD SAM effectively balances segmentation accuracy and computational efficiency, making it well-suited for real-time medical image segmentation applications in resource-constrained environments. 

**Abstract (ZH)**: Segment Anything Model (SAM) 在交互式图像分割中确立了新的标准，提供了跨多种任务的稳健性能。然而，其显著的计算需求限制了其在实时或资源受限环境中的部署。为了应对这些挑战，我们提出了一种新的知识蒸馏方法——KD SAM，并通过均方误差（MSE）和感知损失相结合的方式优化了编码器和解码器。这种双损失框架捕捉了结构和语义特征，使学生模型能够在不增加计算复杂度的情况下保持高分割精度。基于对包括Kvasir-SEG、ISIC 2017、胎儿头超声和乳腺超声数据集的模型评估，我们证明KD SAM在参数明显减少的情况下，能够达到或超越基线模型的性能。KD SAM能够平衡分割精度和计算效率，使其非常适合在资源受限环境中进行实时医学图像分割的应用。 

---
# Distilling Large Language Models for Network Active Queue Management 

**Title (ZH)**: 将大型语言模型用于网络主动队列管理的知识萃取 

**Authors**: Deol Satish, Shiva Raj Pokhrel, Jonathan Kua, Anwar Walid  

**Link**: [PDF](https://arxiv.org/pdf/2501.16734)  

**Abstract**: The growing complexity of network traffic and demand for ultra-low latency communication require smarter packet traffic management. Existing Deep Learning-based queuing approaches struggle with dynamic network scenarios and demand high engineering effort. We propose AQM-LLM, distilling Large Language Models (LLMs) with few-shot learning, contextual understanding, and pattern recognition to improve Active Queue Management (AQM) [RFC 9330] with minimal manual effort. We consider a specific case where AQM is Low Latency, Low Loss, and Scalable Throughput (L4S) and our design of AQM-LLM builds on speculative decoding and reinforcement-based distilling of LLM by tackling congestion prevention in the L4S architecture using Explicit Congestion Notification (ECN) [RFC 9331] and periodic packet dropping. We develop a new open-source experimental platform by executing L4S-AQM on FreeBSD-14, providing interoperable modules to support LLM integration and facilitate IETF recognition through wider testing. Our extensive evaluations show L4S-LLM enhances queue management, prevents congestion, reduces latency, and boosts network performance, showcasing LLMs' adaptability and efficiency in uplifting AQM systems. 

**Abstract (ZH)**: 随着网络流量复杂性的不断增加和对超低延迟通信的需求，需要更加智能的分组流量管理。现有的基于深度学习的队列管理方法在面对动态网络场景时遇到瓶颈，需要较高的工程投入。我们提出了AQM-LLM，通过少量示例学习、语境理解和模式识别提炼大规模语言模型（LLMs），以最少的人工干预改善主动队列管理（AQM）[RFC 9330]。我们考虑了一个具体场景，即AQM为低延迟、低丢包和可扩展吞吐量（L4S）模式，我们的AQM-LLM设计通过解决L4S架构中的拥塞预防问题，利用显式拥塞通知（ECN）[RFC 9331]和定期丢包来实现基于推测解码和强化学习的提炼。我们通过在FreeBSD-14上实现L4S-AQM来开发一个新的开源实验平台，提供了支持LLM集成的互操作模块，并通过广泛的测试促进IETF的认可。我们的广泛评估表明，L4S-LLM能够改进队列管理、预防拥塞、降低延迟并提升网络性能，展示了LLMs在提升AQM系统中的适应性和效率方面的能力。 

---
# On the Interplay Between Sparsity and Training in Deep Reinforcement Learning 

**Title (ZH)**: 深度强化学习中稀疏性和训练之间的相互作用 

**Authors**: Fatima Davelouis, John D. Martin, Michael Bowling  

**Link**: [PDF](https://arxiv.org/pdf/2501.16729)  

**Abstract**: We study the benefits of different sparse architectures for deep reinforcement learning. In particular, we focus on image-based domains where spatially-biased and fully-connected architectures are common. Using these and several other architectures of equal capacity, we show that sparse structure has a significant effect on learning performance. We also observe that choosing the best sparse architecture for a given domain depends on whether the hidden layer weights are fixed or learned. 

**Abstract (ZH)**: 我们研究了不同稀疏架构在深度强化学习中的优势。特别地，我们关注基于图像的应用领域，其中局部偏置和全连接架构较为常见。利用这些架构以及其他几种具有相同容量的架构，我们展示了稀疏结构对学习性能有显著影响。此外，我们观察到，为给定领域选择最佳稀疏架构取决于隐藏层权重是固定不变还是学习生成。 

---
# Bridging Neural Networks and Wireless Systems with MIMO-OFDM Semantic Communications 

**Title (ZH)**: 将神经网络与具有MIMO-OFDM语义通信的无线系统相结合 

**Authors**: Hanju Yoo, Dongha Choi, Yonghwi Kim, Yoontae Kim, Songkuk Kim, Chan-Byoung Chae, Robert W. Heath Jr  

**Link**: [PDF](https://arxiv.org/pdf/2501.16726)  

**Abstract**: Semantic communications aim to enhance transmission efficiency by jointly optimizing source coding, channel coding, and modulation. While prior research has demonstrated promising performance in simulations, real-world implementations often face significant challenges, including noise variability and nonlinear distortions, leading to performance gaps. This article investigates these challenges in a multiple-input multiple-output (MIMO) and orthogonal frequency division multiplexing (OFDM)-based semantic communication system, focusing on the practical impacts of power amplifier (PA) nonlinearity and peak-to-average power ratio (PAPR) variations. Our analysis identifies frequency selectivity of the actual channel as a critical factor in performance degradation and demonstrates that targeted mitigation strategies can enable semantic systems to approach theoretical performance. By addressing key limitations in existing designs, we provide actionable insights for advancing semantic communications in practical wireless environments. This work establishes a foundation for bridging the gap between theoretical models and real-world deployment, highlighting essential considerations for system design and optimization. 

**Abstract (ZH)**: 语义通信旨在通过联合优化源编码、信道编码和调制来提升传输效率。尽管先前的研究在仿真实验中展现了有希望的性能，但在实际部署中往往面临显著的挑战，包括噪声的波动性和非线性失真，导致实际性能与理论性能存在差距。本文在基于多输入多输出（MIMO）和正交频分复用（OFDM）的语义通信系统中，探讨了这些挑战，重点关注功率放大器（PA）非线性和峰值功率与平均功率比（PAPR）变化的实际影响。我们的分析表明，实际信道的选择性频率是导致性能降级的关键因素，并且已确定了特定的缓解策略可以使语义系统接近理论性能。通过解决现有设计中的关键限制，本文提供了在实际无线环境中推进语义通信的实用见解。这项工作为理论模型与实际部署之间的鸿沟架起了桥梁，并强调了系统设计和优化中必须考虑的关键因素。 

---
# Hypergraph Diffusion for High-Order Recommender Systems 

**Title (ZH)**: 高阶图扩散在高阶推荐系统中的应用 

**Authors**: Darnbi Sakong, Thanh Trung Huynh, Jun Jo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16722)  

**Abstract**: Recommender systems rely on Collaborative Filtering (CF) to predict user preferences by leveraging patterns in historical user-item interactions. While traditional CF methods primarily focus on learning compact vector embeddings for users and items, graph neural network (GNN)-based approaches have emerged as a powerful alternative, utilizing the structure of user-item interaction graphs to enhance recommendation accuracy. However, existing GNN-based models, such as LightGCN and UltraGCN, often struggle with two major limitations: an inability to fully account for heterophilic interactions, where users engage with diverse item categories, and the over-smoothing problem in multi-layer GNNs, which hinders their ability to model complex, high-order relationships. To address these gaps, we introduce WaveHDNN, an innovative wavelet-enhanced hypergraph diffusion framework. WaveHDNN integrates a Heterophily-aware Collaborative Encoder, designed to capture user-item interactions across diverse categories, with a Multi-scale Group-wise Structure Encoder, which leverages wavelet transforms to effectively model localized graph structures. Additionally, cross-view contrastive learning is employed to maintain robust and consistent representations. Experiments on benchmark datasets validate the efficacy of WaveHDNN, demonstrating its superior ability to capture both heterophilic and localized structural information, leading to improved recommendation performance. 

**Abstract (ZH)**: 推荐系统依赖于协同过滤（Collaborative Filtering, CF）方法，通过利用用户-项目历史交互模式来预测用户的偏好。传统的CF方法主要侧重于学习用户和项目的紧凑向量嵌入，而基于图神经网络（Graph Neural Networks, GNN）的方法因其能利用用户-项目交互图的结构而成为强大的替代方案，提高了推荐的准确性。然而，现有的基于GNN的模型，如LightGCN和UltraGCN，仍面临两个主要局限性：一是难以充分处理异构交互，即用户与不同类别项目的互动；二是多层GNN中的平滑化问题，这妨碍了模型捕捉复杂、高阶关系的能力。为了解决这些问题，我们引入了WaveHDNN，这是一种创新的波let增强超图扩散框架。WaveHDNN结合了异构交互感知的协作编码器，旨在捕获用户与不同类别项目的互动，以及多层次分组结构编码器，利用波let变换有效地建模局部图结构。此外，采用跨视角对比学习来保持鲁棒且一致的表示。基准数据集上的实验验证了WaveHDNN的有效性，表明它在捕捉异构性和局部结构信息方面具有优越性，从而提高了推荐性能。 

---
# One Head Eight Arms: Block Matrix based Low Rank Adaptation for CLIP-based Few-Shot Learning 

**Title (ZH)**: 《一头八臂：基于块矩阵低秩适应的CLIP驱动少量样本学习》 

**Authors**: Chunpeng Zhou, Qianqian Shen, Zhi Yu, Jiajun Bu, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16720)  

**Abstract**: Recent advancements in fine-tuning Vision-Language Foundation Models (VLMs) have garnered significant attention for their effectiveness in downstream few-shot learning this http URL these recent approaches exhibits some performance improvements, they often suffer from excessive training parameters and high computational costs. To address these challenges, we propose a novel Block matrix-based low-rank adaptation framework, called Block-LoRA, for fine-tuning VLMs on downstream few-shot tasks. Inspired by recent work on Low-Rank Adaptation (LoRA), Block-LoRA partitions the original low-rank decomposition matrix of LoRA into a series of sub-matrices while sharing all down-projection sub-matrices. This structure not only reduces the number of training parameters, but also transforms certain complex matrix multiplication operations into simpler matrix addition, significantly lowering the computational cost of fine-tuning. Notably, Block-LoRA enables fine-tuning CLIP on the ImageNet few-shot benchmark using a single 24GB GPU. We also show that Block-LoRA has the more tighter bound of generalization error than vanilla LoRA. Without bells and whistles, extensive experiments demonstrate that Block-LoRA achieves competitive performance compared to state-of-the-art CLIP-based few-shot methods, while maintaining a low training parameters count and reduced computational overhead. 

**Abstract (ZH)**: 近年来，在视觉-语言基础模型（VLMs）微调方面的最新进展因其在下游少量样本学习中的有效性而引起了广泛关注。尽管这些近期的方法在性能上取得了一些改进，但它们往往伴随着过多的训练参数和高昂的计算成本。为了解决这些问题，我们提出了一种基于块矩阵的低阶适应框架，称为Block-LoRA，用于在下游少量样本任务中微调VLMs。受低阶适应（LoRA）最近工作的影响，Block-LoRA 将LoRA的原始低阶分解矩阵划分为一系列子矩阵，同时共享所有下行投影子矩阵。这种结构不仅减少了训练参数的数量，还把某些复杂的矩阵乘法操作转化为更简单的矩阵加法操作，显著降低了微调时的计算成本。值得注意的是，Block-LoRA允许使用单个24GB GPU微调CLIP on ImageNet少量样本基准。此外，我们还证明了Block-LoRA在泛化误差上的界比常规LoRA更紧。通过大量实验，我们展示了Block-LoRA与最先进的基于CLIP的少量样本方法相比具有竞争力的性能，同时保持了较低的训练参数计数和减少的计算开销。 

---
# Separate Motion from Appearance: Customizing Motion via Customizing Text-to-Video Diffusion Models 

**Title (ZH)**: 分离运动与外观：通过自定义文本到视频扩散模型定制运动 

**Authors**: Huijie Liu, Jingyun Wang, Shuai Ma, Jie Hu, Xiaoming Wei, Guoliang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16714)  

**Abstract**: Motion customization aims to adapt the diffusion model (DM) to generate videos with the motion specified by a set of video clips with the same motion concept. To realize this goal, the adaptation of DM should be possible to model the specified motion concept, without compromising the ability to generate diverse appearances. Thus, the key to solving this problem lies in how to separate the motion concept from the appearance in the adaptation process of DM. Typical previous works explore different ways to represent and insert a motion concept into large-scale pretrained text-to-video diffusion models, e.g., learning a motion LoRA, using latent noise residuals, etc. While those methods can encode the motion concept, they also inevitably encode the appearance in the reference videos, resulting in weakened appearance generation capability. In this paper, we follow the typical way to learn a motion LoRA to encode the motion concept, but propose two novel strategies to enhance motion-appearance separation, including temporal attention purification (TAP) and appearance highway (AH). Specifically, we assume that in the temporal attention module, the pretrained Value embeddings are sufficient to serve as basic components needed by producing a new motion. Thus, in TAP, we choose only to reshape the temporal attention with motion LoRAs so that Value embeddings can be reorganized to produce a new motion. Further, in AH, we alter the starting point of each skip connection in U-Net from the output of each temporal attention module to the output of each spatial attention module. Extensive experiments demonstrate that compared to previous works, our method can generate videos with appearance more aligned with the text descriptions and motion more consistent with the reference videos. 

**Abstract (ZH)**: 运动自适应旨在使扩散模型（DM）适应生成具有特定运动概念集的视频。为了实现这一目标，DM 的适应性应能够建模指定的运动概念，而不会削弱其生成多样化外观的能力。因此，解决这一问题的关键在于如何在 DM 的适应过程中将运动概念与外观分离。典型的先前工作探索了不同方法来表示并插入运动概念到大规模的预训练文本到视频扩散模型中，例如学习运动 LoRA、使用潜在噪声残差等。虽然这些方法可以编码运动概念，但它们也不可避免地将参考视频中的外观编码进来，从而削弱了外观生成能力。在本文中，我们采用了学习运动 LoRA 的典型方法来编码运动概念，但提出了两种新颖的策略来增强运动与外观的分离，包括时序注意力净化（TAP）和外观高速公路（AH）。具体而言，我们假设在时序注意力模块中，预训练的值嵌入足以作为生成新运动所需的基本部件。因此，在 TAP 中，我们选择仅通过运动 LoRA 重新塑造时序注意力，使得值嵌入能够重新组织以生成新的运动。此外，在 AH 中，我们改变了 U-Net 中每一跳连接的起始点，从每个时序注意力模块的输出改为每个空域注意力模块的输出。大量实验表明，与先前的工作相比，我们的方法能够生成与文本描述更一致的外观、与参考视频更一致的运动的视频。 

---
# Determining Mosaic Resilience in Sugarcane Plants using Hyperspectral Images 

**Title (ZH)**: 使用高光谱图像确定甘蔗植株镶嵌性抗性的方法 

**Authors**: Ali Zia, Jun Zhou, Muyiwa Olayemi  

**Link**: [PDF](https://arxiv.org/pdf/2501.16700)  

**Abstract**: Sugarcane mosaic disease poses a serious threat to the Australian sugarcane industry, leading to yield losses of up to 30% in susceptible varieties. Existing manual inspection methods for detecting mosaic resilience are inefficient and impractical for large-scale application. This study introduces a novel approach using hyperspectral imaging and machine learning to detect mosaic resilience by leveraging global feature representation from local spectral patches. Hyperspectral data were collected from eight sugarcane varieties under controlled and field conditions. Local spectral patches were analyzed to capture spatial and spectral variations, which were then aggregated into global feature representations using a ResNet18 deep learning architecture. While classical methods like Support Vector Machines struggled to utilize spatial-spectral relationships effectively, the deep learning model achieved high classification accuracy, demonstrating its capacity to identify mosaic resilience from fine-grained hyperspectral data. This approach enhances early detection capabilities, enabling more efficient management of susceptible strains and contributing to sustainable sugarcane production. 

**Abstract (ZH)**: 蔗糖条纹病对澳大利亚蔗糖产业构成了严重威胁，导致易感品种的产量损失高达30%。现有的手工检查方法检测条纹抗性的效率低下，不适用于大规模应用。本研究提出了一种新的方法，利用高光谱成像和机器学习来检测条纹抗性，通过局部光谱斑块的全球特征表示来利用光谱关系。在受控和田间条件下，从八个蔗糖品种中采集了高光谱数据。局部光谱斑块被分析以捕捉空间和光谱变化，然后使用ResNet18深度学习架构将其聚合为全局特征表示。与像支持向量机这样的传统方法相比，深度学习模型能够有效地利用空-谱关系，并实现了高分类准确性，表明其有能力从高光谱数据中识别条纹抗性。该方法增强了早期检测能力，有助于更有效地管理易感品种，并促进可持续的蔗糖生产。 

---
# Improving Interpretability and Accuracy in Neuro-Symbolic Rule Extraction Using Class-Specific Sparse Filters 

**Title (ZH)**: 使用类别特异性稀疏滤波器提高神经符号规则提取的可解释性和准确性 

**Authors**: Parth Padalkar, Jaeseong Lee, Shiyi Wei, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2501.16677)  

**Abstract**: There has been significant focus on creating neuro-symbolic models for interpretable image classification using Convolutional Neural Networks (CNNs). These methods aim to replace the CNN with a neuro-symbolic model consisting of the CNN, which is used as a feature extractor, and an interpretable rule-set extracted from the CNN itself. While these approaches provide interpretability through the extracted rule-set, they often compromise accuracy compared to the original CNN model. In this paper, we identify the root cause of this accuracy loss as the post-training binarization of filter activations to extract the rule-set. To address this, we propose a novel sparsity loss function that enables class-specific filter binarization during CNN training, thus minimizing information loss when extracting the rule-set. We evaluate several training strategies with our novel sparsity loss, analyzing their effectiveness and providing guidance on their appropriate use. Notably, we set a new benchmark, achieving a 9% improvement in accuracy and a 53% reduction in rule-set size on average, compared to the previous SOTA, while coming within 3% of the original CNN's accuracy. This highlights the significant potential of interpretable neuro-symbolic models as viable alternatives to black-box CNNs. 

**Abstract (ZH)**: 近年来，在使用卷积神经网络（CNNs）进行可解释图像分类时，构建神经符号模型的研究取得了显著进展。这些方法旨在用神经符号模型替代CNN，该模型包括用于特征提取的CNN和从CNN中提取的可解释规则集。虽然这些方法通过提取的规则集提供了可解释性，但它们往往在准确度上低于原始的CNN模型。在本文中，我们将这种准确度损失的根本原因归结于在提取规则集时对滤波器激活后的二值化处理。为了解决这一问题，我们提出了一种新颖的稀疏损失函数，可以在CNN训练过程中实现类别特异性滤波器二值化，从而在提取规则集时最大限度地减少信息损失。我们评估了几种使用我们新颖稀疏损失函数的训练策略，并分析了它们的有效性，提供了有关其适当使用方法的指导。值得注意的是，我们设定了一个新的基准，相对于之前最先进的方法（SOTA），我们的方法在平均准确度上提高了9%，规则集规模减少了53%，并且准确度仅与原始CNN相差3%。这突显了可解释神经符号模型作为黑盒CNN替代方案的强大潜力。 

---
# Data-Free Model-Related Attacks: Unleashing the Potential of Generative AI 

**Title (ZH)**: 无数据模型相关攻击：释放生成AI的潜力 

**Authors**: Dayong Ye, Tianqing Zhu, Shang Wang, Bo Liu, Leo Yu Zhang, Wanlei Zhou, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16671)  

**Abstract**: Generative AI technology has become increasingly integrated into our daily lives, offering powerful capabilities to enhance productivity. However, these same capabilities can be exploited by adversaries for malicious purposes. While existing research on adversarial applications of generative AI predominantly focuses on cyberattacks, less attention has been given to attacks targeting deep learning models. In this paper, we introduce the use of generative AI for facilitating model-related attacks, including model extraction, membership inference, and model inversion. Our study reveals that adversaries can launch a variety of model-related attacks against both image and text models in a data-free and black-box manner, achieving comparable performance to baseline methods that have access to the target models' training data and parameters in a white-box manner. This research serves as an important early warning to the community about the potential risks associated with generative AI-powered attacks on deep learning models. 

**Abstract (ZH)**: 生成式AI技术在我们的日常生活中越来越广泛地应用，提供了增强生产力的强大能力。然而，同样的能力也可能被对手用于恶意目的。尽管现有研究主要集中在生成式AI在网络安全攻击方面的应用，但针对深度学习模型的攻击却较少被关注。在本文中，我们介绍了生成式AI在模型相关攻击中的应用，包括模型提取、成员推理和模型反演。我们的研究发现，对手可以在无数据和黑箱的方式下，对图像和文本模型发起多种模型相关攻击，并且能达到与基线方法可访问目标模型训练数据和参数时相近的性能。这项研究为学术界敲响了警钟，提醒生成式AI驱动的深度学习模型攻击可能带来的潜在风险。 

---
# Federated Learning for Efficient Condition Monitoring and Anomaly Detection in Industrial Cyber-Physical Systems 

**Title (ZH)**: 联邦学习在工业物理信息系统中高效的条件监控和异常检测中的应用 

**Authors**: William Marfo, Deepak K. Tosh, Shirley V. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2501.16666)  

**Abstract**: Detecting and localizing anomalies in cyber-physical systems (CPS) has become increasingly challenging as systems grow in complexity, particularly due to varying sensor reliability and node failures in distributed environments. While federated learning (FL) provides a foundation for distributed model training, existing approaches often lack mechanisms to address these CPS-specific challenges. This paper introduces an enhanced FL framework with three key innovations: adaptive model aggregation based on sensor reliability, dynamic node selection for resource optimization, and Weibull-based checkpointing for fault tolerance. The proposed framework ensures reliable condition monitoring while tackling the computational and reliability challenges of industrial CPS deployments. Experiments on the NASA Bearing and Hydraulic System datasets demonstrate superior performance compared to state-of-the-art FL methods, achieving 99.5% AUC-ROC in anomaly detection and maintaining accuracy even under node failures. Statistical validation using the Mann-Whitney U test confirms significant improvements, with a p-value less than 0.05, in both detection accuracy and computational efficiency across various operational scenarios. 

**Abstract (ZH)**: 随着系统复杂性的增加，检测和定位 cyber-物理系统（CPS）中的异常变得日益具挑战性，尤其是由于分布式环境中传感器可靠性和节点故障的差异性。虽然联邦学习（FL）为分布式模型训练提供了基础，但现有方法往往缺乏应对这些 CPS 特定挑战的机制。本文提出了一种改进的 FL 框架，包含三个关键创新：基于传感器可靠性的自适应模型聚合、动态节点选择以实现资源优化，以及基于 Weibull 分布的检查点机制以提高容错能力。该提出的框架确保了可靠的条件监测，并解决了工业 CPS 部署中的计算和可靠性挑战。通过对 NASA 轴承和液压系统数据集的实验表明，该方法在异常检测方面优于最先进的 FL 方法，实现了高达 99.5% 的 AUC-ROC，并在节点故障下仍能保持准确性。使用曼 Whitney U 检验的统计验证显示，在各种操作场景下，该方法在检测准确性和计算效率方面均显示出显著改进，p 值小于 0.05。 

---
# Data Duplication: A Novel Multi-Purpose Attack Paradigm in Machine Unlearning 

**Title (ZH)**: 数据重复利用：机器遗忘中的一种新型多功能攻击范式 

**Authors**: Dayong Ye, Tainqing Zhu, Jiayang Li, Kun Gao, Bo Liu, Leo Yu Zhang, Wanlei Zhou, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16663)  

**Abstract**: Duplication is a prevalent issue within datasets. Existing research has demonstrated that the presence of duplicated data in training datasets can significantly influence both model performance and data privacy. However, the impact of data duplication on the unlearning process remains largely unexplored. This paper addresses this gap by pioneering a comprehensive investigation into the role of data duplication, not only in standard machine unlearning but also in federated and reinforcement unlearning paradigms. Specifically, we propose an adversary who duplicates a subset of the target model's training set and incorporates it into the training set. After training, the adversary requests the model owner to unlearn this duplicated subset, and analyzes the impact on the unlearned model. For example, the adversary can challenge the model owner by revealing that, despite efforts to unlearn it, the influence of the duplicated subset remains in the model. Moreover, to circumvent detection by de-duplication techniques, we propose three novel near-duplication methods for the adversary, each tailored to a specific unlearning paradigm. We then examine their impacts on the unlearning process when de-duplication techniques are applied. Our findings reveal several crucial insights: 1) the gold standard unlearning method, retraining from scratch, fails to effectively conduct unlearning under certain conditions; 2) unlearning duplicated data can lead to significant model degradation in specific scenarios; and 3) meticulously crafted duplicates can evade detection by de-duplication methods. 

**Abstract (ZH)**: 数据集中的复制是常见的问题。现有研究表明，训练数据集中的重复数据会显著影响模型性能和数据隐私。然而，数据重复对抹除过程的影响尚未得到广泛探究。本文通过开创性地对数据重复的角色进行全面研究，填补了这一空白，不仅涵盖了标准的机器抹除的方法，还涉及联邦学习和强化学习中的抹除方法。具体来说，我们提出了一种对手，该对手复制目标模型训练集的子集，并将其纳入训练集。训练完成后，该对手要求模型所有者抹除该复制的子集，并分析其对抹除后的模型的影响。例如，该对手可以通过揭示尽管努力抹除，复制子集的影响仍然保留在模型中来挑战模型所有者。此外，为了绕过去重技术的检测，我们提出了三种新型的近似重复方法，这些方法适用于特定的抹除范式。随后，我们研究了当应用去重技术时这些方法对抹除过程的影响。我们的发现揭示了几点关键的见解：1）标准的从头开始重新训练的抹除方法在某些条件下无法有效执行抹除；2）抹除重复数据可能导致特定场景下的模型显著退化；3）精心设计的重复可以逃避去重方法的检测。 

---
# Vision-based autonomous structural damage detection using data-driven methods 

**Title (ZH)**: 基于视觉的数据驱动方法在自主结构损伤检测中的应用 

**Authors**: Seyyed Taghi Ataei, Parviz Mohammad Zadeh, Saeid Ataei  

**Link**: [PDF](https://arxiv.org/pdf/2501.16662)  

**Abstract**: This study addresses the urgent need for efficient and accurate damage detection in wind turbine structures, a crucial component of renewable energy infrastructure. Traditional inspection methods, such as manual assessments and non-destructive testing (NDT), are often costly, time-consuming, and prone to human error. To tackle these challenges, this research investigates advanced deep learning algorithms for vision-based structural health monitoring (SHM). A dataset of wind turbine surface images, featuring various damage types and pollution, was prepared and augmented for enhanced model training. Three algorithms-YOLOv7, its lightweight variant, and Faster R-CNN- were employed to detect and classify surface damage. The models were trained and evaluated on a dataset split into training, testing, and evaluation subsets (80%-10%-10%). Results indicate that YOLOv7 outperformed the others, achieving 82.4% mAP@50 and high processing speed, making it suitable for real-time inspections. By optimizing hyperparameters like learning rate and batch size, the models' accuracy and efficiency improved further. YOLOv7 demonstrated significant advancements in detection precision and execution speed, especially for real-time applications. However, challenges such as dataset limitations and environmental variability were noted, suggesting future work on segmentation methods and larger datasets. This research underscores the potential of vision-based deep learning techniques to transform SHM practices by reducing costs, enhancing safety, and improving reliability, thus contributing to the sustainable maintenance of critical infrastructure and supporting the longevity of wind energy systems. 

**Abstract (ZH)**: 本研究着眼于风力发电机结构中高效且准确的损伤检测需求，风力发电机是可再生能源基础设施的重要组成部分。传统的检查方法，如人工评估和无损检测（NDT），往往成本高、耗时长且容易出错。为应对这些挑战，本研究考察了基于视觉的结构健康监测（SHM）中高级深度学习算法的应用。为此，我们准备并增强了包含多种损伤类型和污染的风力发电机表面图像数据集，以提高模型的训练效果。本研究采用了三种算法：YOLOv7、其轻量级变体和Faster R-CNN，用于检测和分类表面损伤。模型在训练集、测试集和验证集（80%-10%-10%）的分割数据集上进行了训练和评估。结果表明YOLOv7在mAP@50指标上的表现优于其他两种算法，且具有较高的处理速度，使其适合于实时检查。通过优化学习率和批次大小等超参数，模型的准确性和效率得到了进一步提升。YOLOv7展示了在检测精度和执行速度方面的重要进步，尤其是在实时应用方面。然而，研究也指出了数据集限制和环境变化等挑战，未来研究可能需关注分割方法和更大规模的数据集。本研究强调了基于视觉的深度学习技术在降低运营成本、提升安全性及可靠性方面的潜力，从而有助于关键基础设施的可持续维护，并支持风能系统的长期稳定运行。 

---
# Contextual Reinforcement in Multimodal Token Compression for Large Language Models 

**Title (ZH)**: 多模态令牌压缩中基于上下文的强化学习方法 

**Authors**: Naderdel Piero, Zacharias Cromwell, Nathaniel Wainwright, Matthias Nethercott  

**Link**: [PDF](https://arxiv.org/pdf/2501.16658)  

**Abstract**: Effective token compression remains a critical challenge for scaling models to handle increasingly complex and diverse datasets. A novel mechanism based on contextual reinforcement is introduced, dynamically adjusting token importance through interdependencies and semantic relevance. This approach enables substantial reductions in token usage while preserving the quality and coherence of information representation. Incorporating graph-based algorithms and adaptive weighting, the method captures subtle contextual relationships across textual and multimodal data, ensuring robust alignment and performance in downstream tasks. Evaluations across varied domains reveal significant improvements in accuracy and semantic retention, particularly for tasks requiring detailed cross-modal interactions. Memory usage analyses demonstrate improved computational efficiency, with minimal overhead despite the additional reinforcement processes. Performance gains are further validated through error distribution analyses, showing reduced semantic loss and syntactic inconsistencies compared to baseline models. The modular architecture ensures compatibility with a wide range of open-source frameworks, facilitating scalable implementation for real-world applications. These findings highlight the potential of contextual reinforcement in redefining token management strategies and advancing large-scale model design. 

**Abstract (ZH)**: 将模型规模扩展到处理日益复杂和多样化的数据集仍是一项关键挑战。引入了一种基于上下文强化的新机制，通过相互依赖和语义相关性动态调整词汇项的重要性。这种方法在减少词汇项使用的同时，能够保持信息表示的质量和连贯性。结合图算法和自适应加权，该方法能够捕捉文本和多模态数据之间的微妙上下文关系，确保在下游任务中实现稳健对齐和性能。在不同领域的评估显示，该方法在准确性和语义保留方面取得了显著改进，特别是在需要详细跨模态交互的任务中。内存使用分析表明，尽管存在额外的强化过程，该方法仍能实现改进的计算效率，且具有最小的开销。通过错误分布分析进一步验证了性能收益，结果显示相比基准模型，该方法的语义损失和句法不一致减少了。模块化架构确保了该方法与各种开源框架的兼容性，便于其实现并应用于实际场景。这些发现凸显了上下文强化在重新定义词汇项管理策略和推动大规模模型设计方面的潜力。 

---
# Large Language Model Critics for Execution-Free Evaluation of Code Changes 

**Title (ZH)**: 大型语言模型评论者：执行免费评估编码变化 

**Authors**: Aashish Yadavally, Hoan Nguyen, Laurent Callot, Gauthier Guinet  

**Link**: [PDF](https://arxiv.org/pdf/2501.16655)  

**Abstract**: Large language models (LLMs) offer a promising way forward for automating software engineering tasks, such as bug fixes, feature additions, etc., via multi-step LLM-based agentic workflows. However, existing metrics for evaluating such workflows, mainly build status and occasionally log analysis, are too sparse and limited in providing the information needed to assess the quality of changes made. In this work, we designed LLM-based critics to derive well-structured and rigorous intermediate/step-level, execution-free evaluation proxies for repo-level code changes. Importantly, we assume access to the gold test patch for the problem (i.e., reference-aware) to assess both semantics and executability of generated patches. With the gold test patch as a reference, we predict executability of all editing locations with an F1 score of 91.6%, aggregating which, we can predict the build status in 84.8% of the instances in SWE-bench. In particular, such an execution-focused LLM critic outperforms other reference-free and reference-aware LLM critics by 38.9% to 72.5%. Moreover, we demonstrate the usefulness of such a reference-aware framework in comparing patches generated by different agentic workflows. Finally, we open-source the library developed for this project, which allows further usage for either other agentic workflows or other benchmarks. The source code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）为通过多步骤的基于LLM的代理工作流自动化软件工程任务（如错误修复、功能添加等）提供了有希望的方法。然而，现有的用于评估这些工作流的评估指标主要基于构建状态，偶尔也会进行日志分析，这些指标过于稀疏，提供的信息有限，不足以评估做出的变更质量。在这项工作中，我们设计了基于LLM的评估者，以推导出结构良好且严谨的中间/步骤级、无需执行的评估代理，用于评估仓库级别的代码变更。重要的是，我们假设可以访问问题的金标准测试补丁（即，带有参考信息的），以评估生成补丁的语义和可执行性。通过参考金标准测试补丁，我们以91.6%的F1分数预测所有编辑位置的可执行性，并综合这些结果，我们可以在SWE-bench的84.8%的情况下预测构建状态。特别是，这种以执行为中心的LLM评估者比其他无参考和有参考的LLM评估者分别高出38.9%到72.5%。此外，我们展示了这种有参考框架在比较不同代理工作流生成的补丁方面的有用性。最后，我们将为该项目开发的库开源，以便其他人可以将其用于其他代理工作流或基准测试。源代码可在以下链接获得：[这里提供链接]。 

---
# Molecular-driven Foundation Model for Oncologic Pathology 

**Title (ZH)**: 基于分子驱动的肿瘤病理学基础模型 

**Authors**: Anurag Vaidya, Andrew Zhang, Guillaume Jaume, Andrew H. Song, Tong Ding, Sophia J. Wagner, Ming Y. Lu, Paul Doucet, Harry Robertson, Cristina Almagro-Perez, Richard J. Chen, Dina ElHarouni, Georges Ayoub, Connor Bossi, Keith L. Ligon, Georg Gerber, Long Phi Le, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2501.16652)  

**Abstract**: Foundation models are reshaping computational pathology by enabling transfer learning, where models pre-trained on vast datasets can be adapted for downstream diagnostic, prognostic, and therapeutic response tasks. Despite these advances, foundation models are still limited in their ability to encode the entire gigapixel whole-slide images without additional training and often lack complementary multimodal data. Here, we introduce Threads, a slide-level foundation model capable of generating universal representations of whole-slide images of any size. Threads was pre-trained using a multimodal learning approach on a diverse cohort of 47,171 hematoxylin and eosin (H&E)-stained tissue sections, paired with corresponding genomic and transcriptomic profiles - the largest such paired dataset to be used for foundation model development to date. This unique training paradigm enables Threads to capture the tissue's underlying molecular composition, yielding powerful representations applicable to a wide array of downstream tasks. In extensive benchmarking across 54 oncology tasks, including clinical subtyping, grading, mutation prediction, immunohistochemistry status determination, treatment response prediction, and survival prediction, Threads outperformed all baselines while demonstrating remarkable generalizability and label efficiency. It is particularly well suited for predicting rare events, further emphasizing its clinical utility. We intend to make the model publicly available for the broader community. 

**Abstract (ZH)**: 基础模型正在通过启用迁移学习重新塑造计算病理学，从而使预训练于大量数据集上的模型能够适应下游诊断、预后和治疗反应任务。尽管取得了这些进展，但基础模型仍然在编码整个 gigapixel 病理切片图像方面受到限制，通常缺乏补充的多模态数据。在此，我们介绍了 Threads，这是一种能够在不同大小的全切片图像中生成通用表示的切片级基础模型。Threads 通过结合组织学和对应的基因组及转录组资料进行了多模态预训练，这是迄今为止用于基础模型开发的最大的配对数据集。这种独特的训练范式使 Threads 能够捕捉到组织的分子组成，从而生成适用于众多下游任务的强大表示。在对包括临床亚型分类、分级、突变预测、免疫组化状态确定、治疗反应预测和生存预测在内的54项肿瘤学任务进行广泛基准测试中，Threads 在所有基线模型上均表现出色，展示了出色的泛化能力和标签效率。特别适用于预测罕见事件，进一步突显了其临床应用价值。我们计划将模型公开供更广泛的社区使用。 

---
# DOCS: Quantifying Weight Similarity for Deeper Insights into Large Language Models 

**Title (ZH)**: DOCS：量化权重相似性以更深入地探讨大规模语言模型 

**Authors**: Zeping Min, Xinshang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16650)  

**Abstract**: We introduce a novel index, the Distribution of Cosine Similarity (DOCS), for quantitatively assessing the similarity between weight matrices in Large Language Models (LLMs), aiming to facilitate the analysis of their complex architectures. Leveraging DOCS, our analysis uncovers intriguing patterns in the latest open-source LLMs: adjacent layers frequently exhibit high weight similarity and tend to form clusters, suggesting depth-wise functional specialization. Additionally, we prove that DOCS is theoretically effective in quantifying similarity for orthogonal matrices, a crucial aspect given the prevalence of orthogonal initializations in LLMs. This research contributes to a deeper understanding of LLM architecture and behavior, offering tools with potential implications for developing more efficient and interpretable models. 

**Abstract (ZH)**: 我们提出了一种新的指标，即余弦相似度分布（Distribution of Cosine Similarity, DOCS），用于定量评估大型语言模型（LLMs）中权重矩阵之间的相似性，旨在便于分析其复杂的架构。利用DOCS，我们的分析揭示了最新开源LLMs中的有趣模式：相邻层经常表现出高权重相似性，并倾向于形成集群，这表明它们在深度方向上具有功能专业化。此外，我们证明了DOCS在量化正交矩阵之间的相似性方面具有理论上的有效性，这是一个关键方面，因为正交初始化在LLMs中非常普遍。这项研究有助于更深入地理解LLM的架构和行为，提供了具有潜在影响的工具，有助于开发更高效和可解释的模型。 

---
# An LLM Benchmark for Addressee Recognition in Multi-modal Multi-party Dialogue 

**Title (ZH)**: 多模态多人群体对话中收件人识别的LLM基准测试 

**Authors**: Koji Inoue, Divesh Lala, Mikey Elmers, Keiko Ochi, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2501.16643)  

**Abstract**: Handling multi-party dialogues represents a significant step for advancing spoken dialogue systems, necessitating the development of tasks specific to multi-party interactions. To address this challenge, we are constructing a multi-modal multi-party dialogue corpus of triadic (three-participant) discussions. This paper focuses on the task of addressee recognition, identifying who is being addressed to take the next turn, a critical component unique to multi-party dialogue systems. A subset of the corpus was annotated with addressee information, revealing that explicit addressees are indicated in approximately 20% of conversational turns. To evaluate the task's complexity, we benchmarked the performance of a large language model (GPT-4o) on addressee recognition. The results showed that GPT-4o achieved an accuracy only marginally above chance, underscoring the challenges of addressee recognition in multi-party dialogue. These findings highlight the need for further research to enhance the capabilities of large language models in understanding and navigating the intricacies of multi-party conversational dynamics. 

**Abstract (ZH)**: 处理多轮对话是推动口语对话系统发展的重大步骤，需要开发专门针对多轮交互的任务。为应对这一挑战，我们构建了一个包含三方讨论的多模态多轮对话语料库。本文专注于接收方识别任务，即识别谁是下一个发言的人，这是多轮对话系统中的一个关键组成部分。部分语料库被标注了接收方信息，结果显示约20%的对话轮次中明确指出了接收方。为了评估任务的复杂性，我们使用了一个大型语言模型（GPT-4o）进行了接收方识别的基准测试。结果表明，GPT-4o 的准确率仅略高于随机猜测，突显了在多轮对话中进行接收方识别的挑战。这些发现强调了进一步研究以增强大型语言模型在理解并导航多轮对话复杂动态方面的必要性。 

---
# Why Do We Laugh? Annotation and Taxonomy Generation for Laughable Contexts in Spontaneous Text Conversation 

**Title (ZH)**: 我们为什么笑？自发文本对话中可笑情境的标注与分类生成 

**Authors**: Koji Inoue, Mikey Elmers, Divesh Lala, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2501.16635)  

**Abstract**: Laughter serves as a multifaceted communicative signal in human interaction, yet its identification within dialogue presents a significant challenge for conversational AI systems. This study addresses this challenge by annotating laughable contexts in Japanese spontaneous text conversation data and developing a taxonomy to classify the underlying reasons for such contexts. Initially, multiple annotators manually labeled laughable contexts using a binary decision (laughable or non-laughable). Subsequently, an LLM was used to generate explanations for the binary annotations of laughable contexts, which were then categorized into a taxonomy comprising ten categories, including "Empathy and Affinity" and "Humor and Surprise," highlighting the diverse range of laughter-inducing scenarios. The study also evaluated GPT-4's performance in recognizing the majority labels of laughable contexts, achieving an F1 score of 43.14%. These findings contribute to the advancement of conversational AI by establishing a foundation for more nuanced recognition and generation of laughter, ultimately fostering more natural and engaging human-AI interactions. 

**Abstract (ZH)**: 笑声在人类互动中充当着多维度的沟通信号，但在对话中识别笑声对对话式人工智能系统构成了重大挑战。本研究通过在日语自发文本对话数据中标注可笑情境，并发展了一种分类体系以分类这些情境背后的原因，来应对这一挑战。首先，多名注释者手动对可笑情境进行了二元标注（可笑或非可笑）。随后，使用语言模型生成解释这些二元标注的文本，并将这些解释分类到一组包括“同情与共鸣”和“幽默与意外”在内的十个类别中，突显了引发笑声情境的多样性。此外，研究还评估了GPT-4在识别可笑情境主要标签方面的能力，其F1分数达到了43.14%。这些发现为推进对话式人工智能，提供了更细致的笑声识别和生成的基础，最终促进了更加自然和互动的人机交流。 

---
# Towards Resource-Efficient Compound AI Systems 

**Title (ZH)**: 面向高效资源的复合人工智能系统研究 

**Authors**: Gohar Irfan Chaudhry, Esha Choukse, Íñigo Goiri, Rodrigo Fonseca, Adam Belay, Ricardo Bianchini  

**Link**: [PDF](https://arxiv.org/pdf/2501.16634)  

**Abstract**: Compound AI Systems, integrating multiple interacting components like models, retrievers, and external tools, have emerged as essential for addressing complex AI tasks. However, current implementations suffer from inefficient resource utilization due to tight coupling between application logic and execution details, a disconnect between orchestration and resource management layers, and the perceived exclusiveness between efficiency and quality.
We propose a vision for resource-efficient Compound AI Systems through a \emph{declarative workflow programming model} and an \emph{adaptive runtime system} for dynamic scheduling and resource-aware decision-making. Decoupling application logic from low-level details exposes levers for the runtime to flexibly configure the execution environment and resources, without compromising on quality. Enabling collaboration between the workflow orchestration and cluster manager enables higher efficiency through better scheduling and resource management.
We are building a prototype system, called \textbf{\textit{Murakkab}}, to realize this vision. Our preliminary evaluation demonstrates speedups up to $\sim 3.4\times$ in workflow completion times while delivering $\sim 4.5\times$ higher energy efficiency, showing promise in optimizing resources and advancing AI system design. 

**Abstract (ZH)**: 复合同人工智能系统（Compound AI Systems）集成了多个相互作用的组件，如模型、检索器和外部工具，已成为解决复杂AI任务的必要工具。然而，当前的实现由于应用程序逻辑和执行细节之间的紧密耦合、编排层与资源管理层之间的脱节，以及效率与质量之间的误解，导致资源利用效率低下。

我们提出了一种通过声明式工作流编程模型和动态调度及资源感知决策的适应性运行时系统来实现资源高效复合同人工智能系统的愿景。通过将应用程序逻辑与低级细节解耦，运行时可以灵活地配置执行环境和资源，而不牺牲质量。编排工作流与集群管理器之间的协作可以通过更好的调度和资源管理提高效率。

我们正在构建一个名为**Murakkab**的原型系统，以实现这一愿景。初步评估显示，工作流完成时间加速了约3.4倍，同时实现了约4.5倍更高的能源效率，展示了在优化资源和推进人工智能系统设计方面的潜力。 

---
# Engaging with AI: How Interface Design Shapes Human-AI Collaboration in High-Stakes Decision-Making 

**Title (ZH)**: 参与人工智能：界面设计如何塑造高风险决策中的人机协作 

**Authors**: Zichen Chen, Yunhao Luo, Misha Sra  

**Link**: [PDF](https://arxiv.org/pdf/2501.16627)  

**Abstract**: As reliance on AI systems for decision-making grows, it becomes critical to ensure that human users can appropriately balance trust in AI suggestions with their own judgment, especially in high-stakes domains like healthcare. However, human + AI teams have been shown to perform worse than AI alone, with evidence indicating automation bias as the reason for poorer performance, particularly because humans tend to follow AI's recommendations even when they are incorrect. In many existing human + AI systems, decision-making support is typically provided in the form of text explanations (XAI) to help users understand the AI's reasoning. Since human decision-making often relies on System 1 thinking, users may ignore or insufficiently engage with the explanations, leading to poor decision-making. Previous research suggests that there is a need for new approaches that encourage users to engage with the explanations and one proposed method is the use of cognitive forcing functions (CFFs). In this work, we examine how various decision-support mechanisms impact user engagement, trust, and human-AI collaborative task performance in a diabetes management decision-making scenario. In a controlled experiment with 108 participants, we evaluated the effects of six decision-support mechanisms split into two categories of explanations (text, visual) and four CFFs. Our findings reveal that mechanisms like AI confidence levels, text explanations, and performance visualizations enhanced human-AI collaborative task performance, and improved trust when AI reasoning clues were provided. Mechanisms like human feedback and AI-driven questions encouraged deeper reflection but often reduced task performance by increasing cognitive effort, which in turn affected trust. Simple mechanisms like visual explanations had little effect on trust, highlighting the importance of striking a balance in CFF and XAI design. 

**Abstract (ZH)**: 随着对AI系统在决策中的依赖程度增加，确保人类用户能够恰当地平衡对AI建议的信任与自身的判断变得尤为重要，特别是在像医疗卫生这样高风险的领域。然而，研究表明，人类+AI团队的表现通常不如单独的AI系统，原因在于自动化偏差，尤其是当人类倾向于即使在AI建议错误时也遵循这些建议。在许多现有的人类+AI系统中，决策支持通常以文本解释（XAI）的形式提供，以帮助用户理解AI的推理过程。由于人类决策过程往往依赖于直觉思维（System 1），用户可能会忽视或不充分地参与这些解释，导致决策质量下降。先前的研究表明，需要新的方法来鼓励用户参与这些解释，其中一种方法是使用认知强制函数（CFFs）。

在这项工作中，我们研究了各种决策支持机制如何影响用户参与、信任以及人类与AI协作任务的表现，特别是在糖尿病管理决策场景中的影响。我们在一项包含108名参与者的受控实验中评估了六种决策支持机制的效果，这些机制分为两类解释（文本，视觉）和四种CFFs。我们的研究发现，如AI信心水平、文本解释和绩效可视化等机制能够提升人类与AI协作的任务表现，并在提供AI推理线索时提高信任。人类反馈和AI驱动的问题可以鼓励更深层次的反思，但是这往往通过增加认知负担降低了任务表现，进而影响信任。像简单的视觉解释这样的机制对信任的影响甚微，强调了CFF和XAI设计中保持平衡的重要性。 

---
# Chinese Stock Prediction Based on a Multi-Modal Transformer Framework: Macro-Micro Information Fusion 

**Title (ZH)**: 基于多模态变压器框架的中国股市预测：宏观与微观信息融合 

**Authors**: Lumen AI, Tengzhou No. 1 Middle School, Shihao Ji, Zihui Song, Fucheng Zhong, Jisen Jia, Zhaobo Wu, Zheyi Cao, Xu Tianhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.16621)  

**Abstract**: This paper proposes an innovative Multi-Modal Transformer framework (MMF-Trans) designed to significantly improve the prediction accuracy of the Chinese stock market by integrating multi-source heterogeneous information including macroeconomy, micro-market, financial text, and event knowledge. The framework consists of four core modules: (1) A four-channel parallel encoder that processes technical indicators, financial text, macro data, and event knowledge graph respectively for independent feature extraction of multi-modal data; (2) A dynamic gated cross-modal fusion mechanism that adaptively learns the importance of different modalities through differentiable weight allocation for effective information integration; (3) A time-aligned mixed-frequency processing layer that uses an innovative position encoding method to effectively fuse data of different time frequencies and solves the time alignment problem of heterogeneous data; (4) A graph attention-based event impact quantification module that captures the dynamic impact of events on the market through event knowledge graph and quantifies the event impact coefficient. We introduce a hybrid-frequency Transformer and Event2Vec algorithm to effectively fuse data of different frequencies and quantify the event impact. Experimental results show that in the prediction task of CSI 300 constituent stocks, the root mean square error (RMSE) of the MMF-Trans framework is reduced by 23.7% compared to the baseline model, the event response prediction accuracy is improved by 41.2%, and the Sharpe ratio is improved by 32.6%. 

**Abstract (ZH)**: 本文提出了一种创新的多模态Transformer框架（MMF-Trans），旨在通过整合宏观经济、微观市场、金融文本和事件知识等多种来源的异质信息来显著提高中国股票市场的预测准确性。该框架包含四个核心模块：（1）一个四通道并行编码器，分别处理技术指标、金融文本、宏观经济数据和事件知识图谱，独立提取多模态数据的特征；（2）一个动态门控跨模态融合机制，通过可微配权学习不同模态的重要性，实现有效信息整合；（3）一个时间对齐混合频次处理层，采用创新的位置编码方法有效融合不同时间频率的数据并解决异构数据的时间对齐问题；（4）一个基于图注意力的事件影响量化模块，通过事件知识图谱捕捉事件对市场的动态影响并量化事件影响系数。我们引入了混合频次Transformer和Event2Vec算法，有效融合不同频次的数据并量化事件影响。实验结果显示，在CSI 300成分股的预测任务中，MMF-Trans框架的均方根误差（RMSE）相较于基础模型降低了23.7%，事件响应预测准确性提高了41.2%，夏普比率提高了32.6%。 

---
# Safe Reinforcement Learning for Real-World Engine Control 

**Title (ZH)**: 将“Safe Reinforcement Learning for Real-World Engine Control”翻译成中文时，应保持专业性和准确性，翻译如下：

“面向实际发动机控制的安全强化学习” 

**Authors**: Julian Bedei, Lucas Koch, Kevin Badalian, Alexander Winkler, Patrick Schaber, Jakob Andert  

**Link**: [PDF](https://arxiv.org/pdf/2501.16613)  

**Abstract**: This work introduces a toolchain for applying Reinforcement Learning (RL), specifically the Deep Deterministic Policy Gradient (DDPG) algorithm, in safety-critical real-world environments. As an exemplary application, transient load control is demonstrated on a single-cylinder internal combustion engine testbench in Homogeneous Charge Compression Ignition (HCCI) mode, that offers high thermal efficiency and low emissions. However, HCCI poses challenges for traditional control methods due to its nonlinear, autoregressive, and stochastic nature. RL provides a viable solution, however, safety concerns, such as excessive pressure rise rates, must be addressed when applying to HCCI. A single unsuitable control input can severely damage the engine or cause misfiring and shut down. Additionally, operating limits are not known a priori and must be determined experimentally. To mitigate these risks, real-time safety monitoring based on the k-nearest neighbor algorithm is implemented, enabling safe interaction with the testbench. The feasibility of this approach is demonstrated as the RL agent learns a control policy through interaction with the testbench. A root mean square error of 0.1374 bar is achieved for the indicated mean effective pressure, comparable to neural network-based controllers from the literature. The toolchain's flexibility is further demonstrated by adapting the agent's policy to increase ethanol energy shares, promoting renewable fuel use while maintaining safety. This RL approach addresses the longstanding challenge of applying RL to safety-critical real-world environments. The developed toolchain, with its adaptability and safety mechanisms, paves the way for future applicability of RL in engine testbenches and other safety-critical settings. 

**Abstract (ZH)**: 本文引入了一套应用于安全关键现实环境中的强化学习（RL）工具链，具体采用了深度确定性策略梯度算法（DDPG）。作为示范应用，该工具链在均质充气压缩点火（HCCI）模式下的单缸内燃机试验台上展示了暂态负载控制的应用。HCCI模式能够提供高い热效率和低排放，但传统控制方法难以应对HCCI的非线性、自回归和随机特性所带来的挑战。虽然RL提供了解决方案，但在应用于HCCI时，必须解决安全性问题，如过高的压力上升率等问题。单个不合适的控制输入可能会严重损坏发动机或导致点火失败和停机。此外，操作限制通常是未知的，需要通过实验确定。为了降低这些风险，基于k近邻算法实现了实时安全监控，从而确保与试验台的安全交互。通过实验证明了该方法的有效性，R值通过与试验台的互动，学习出了控制策略，实现了指示平均有效压力的均方根误差达到0.1374巴，与文献中基于神经网络的控制器相媲美。该工具链的灵活性通过调整代理的策略来增加乙醇能量份额得以进一步展示，从而促进可再生能源的使用，同时保持安全性。该RL方法解决了将RL应用于安全关键现实环境的长期挑战。所开发的具有适应性和安全机制的工具链为未来的RL在发动机试验台上和其他安全关键环境中的应用铺平了道路。 

---
# MCTS-SQL: An Effective Framework for Text-to-SQL with Monte Carlo Tree Search 

**Title (ZH)**: MCTS-SQL：一种基于蒙特卡洛树搜索的文本到SQL有效框架 

**Authors**: Shuozhi Yuan, Liming Chen, Miaomiao Yuan, Jin Zhao, Haoran Peng, Wenming Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16607)  

**Abstract**: Text-to-SQL is a fundamental and longstanding problem in the NLP area, aiming at converting natural language queries into SQL, enabling non-expert users to operate databases. Recent advances in LLM have greatly improved text-to-SQL performance. However, challenges persist, especially when dealing with complex user queries. Current approaches (e.g., COT prompting and multi-agent frameworks) rely on the ability of models to plan and generate SQL autonomously, but controlling performance remains difficult. In addition, LLMs are still prone to hallucinations. To alleviate these challenges, we designed a novel MCTS-SQL to guide SQL generation iteratively. The approach generates SQL queries through Monte Carlo Tree Search (MCTS) and a heuristic self-refinement mechanism are used to enhance accuracy and reliability. Key components include a schema selector for extracting relevant information and an MCTS-based generator for iterative query refinement. Experimental results from the SPIDER and BIRD benchmarks show that MCTS-SQL achieves state-of-the-art performance. Specifically, on the BIRD development dataset, MCTS-SQL achieves an Execution (EX) accuracy of 69.40% using GPT-4o as the base model and a significant improvement when dealing with challenging tasks, with an EX of 51.48%, which is 3.41% higher than the existing method. 

**Abstract (ZH)**: 文本到SQL转换是自然语言处理领域的一个基础且长期存在的问题，旨在将自然语言查询转换为SQL，从而使非专家用户能够操作数据库。近年来，大规模语言模型（LLM）的进展极大地提高了文本到SQL的性能。然而，在处理复杂的用户查询时仍存在挑战。当前的方法（例如，基于生成式推理的提示和多代理框架）依赖于模型自主规划和生成SQL的能力，但控制性能依然是一个难题。此外，大规模语言模型仍然容易产生幻觉。为了解决这些挑战，我们设计了一种新的MCTS-SQL方法，以迭代方式引导SQL生成。该方法通过蒙特卡洛树搜索（MCTS）生成SQL查询，并使用启发式的自我完善机制来增强准确性和可靠性。关键组件包括一个模式选择器用于提取相关信息，以及一个基于MCTS的生成器用于迭代查询精炼。来自SPIDER和BIRD基准的实验结果表明，MCTS-SQL在性能上达到了最佳水平。具体而言，在BIRD开发数据集上，使用GPT-4o作为基础模型，MCTS-SQL的执行（EX）准确性达到了69.40%，在处理具有挑战性的任务时，其准确率为51.48%，相比现有方法提高了3.41%。 

---
# Governing the Agent-to-Agent Economy of Trust via Progressive Decentralization 

**Title (ZH)**: 通过渐进去中心化治理信任驱动的代理间经济体 

**Authors**: Tomer Jordi Chaffer  

**Link**: [PDF](https://arxiv.org/pdf/2501.16606)  

**Abstract**: Current approaches to AI governance often fall short in anticipating a future where AI agents manage critical tasks, such as financial operations, administrative functions, and beyond. As AI agents may eventually delegate tasks among themselves to optimize efficiency, understanding the foundational principles of human value exchange could offer insights into how AI-driven economies might operate. Just as trust and value exchange are central to human interactions in open marketplaces, they may also be critical for enabling secure and efficient interactions among AI agents. While cryptocurrencies could serve as the foundation for monetizing value exchange in a collaboration and delegation dynamic among AI agents, a critical question remains: how can these agents reliably determine whom to trust, and how can humans ensure meaningful oversight and control as an economy of AI agents scales and evolves? This paper is a call for a collective exploration of cryptoeconomic incentives, which can help design decentralized governance systems that allow AI agents to autonomously interact and exchange value while ensuring human oversight via progressive decentralization. Toward this end, I propose a research agenda to address the question of agent-to-agent trust using AgentBound Tokens, which are non-transferable, non-fungible tokens uniquely tied to individual AI agents, akin to Soulbound tokens for humans in Web3. By staking ABTs as collateral for autonomous actions within an agent-to-agent network via a proof-of-stake mechanism, agents may be incentivized towards ethical behavior, and penalties for misconduct are automatically enforced. 

**Abstract (ZH)**: 当前对人工智能治理的方法往往在预见未来情境时存在不足，即当人工智能代理接手关键任务（如财务操作、行政职能等）时。随着人工智能代理可能最终会将任务相互委托以优化效率，理解人类价值交换的基本原则可能会为了解人工智能驱动的经济如何运作提供洞察。就像信任和价值交换在开放市场的人类互动中至关重要一样，它们可能也会为人工智能代理之间实现安全高效的互动提供关键基础。尽管加密货币可以在人工智能代理间合作和委托动态中提供价值交换的基础，但仍有一个关键问题：这些代理如何可靠地确定彼此是否值得信任？人类如何确保在人工智能代理经济规模扩展和演进的过程中实现有意义的监督和控制？本文呼吁集体探索加密经济激励机制，这有助于设计分散化治理系统，使人工智能代理能够自主互动和交换价值，同时通过逐步分散化确保人类监督。为此，我提出了一项研究议程，探讨通过“AgentBound Tokens”（非转移、非同质化代币，与个别AI代理独特绑定）来解决代理间互信的问题。作为智能合约中的质押品用于代理间的网络自治行为验证，这些代币可以激励代理采取道德行为，同时自动执行不当行为的惩罚措施。 

---
# Impact and influence of modern AI in metadata management 

**Title (ZH)**: 现代人工智能在元数据管理中的影响与影响 

**Authors**: Wenli Yang, Rui Fu, Muhammad Bilal Amin, Byeong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16605)  

**Abstract**: Metadata management plays a critical role in data governance, resource discovery, and decision-making in the data-driven era. While traditional metadata approaches have primarily focused on organization, classification, and resource reuse, the integration of modern artificial intelligence (AI) technologies has significantly transformed these processes. This paper investigates both traditional and AI-driven metadata approaches by examining open-source solutions, commercial tools, and research initiatives. A comparative analysis of traditional and AI-driven metadata management methods is provided, highlighting existing challenges and their impact on next-generation datasets. The paper also presents an innovative AI-assisted metadata management framework designed to address these challenges. This framework leverages more advanced modern AI technologies to automate metadata generation, enhance governance, and improve the accessibility and usability of modern datasets. Finally, the paper outlines future directions for research and development, proposing opportunities to further advance metadata management in the context of AI-driven innovation and complex datasets. 

**Abstract (ZH)**: 元数据管理在数据驱动时代的数据治理、资源发现和决策制定中扮演着关键角色。虽然传统的元数据方法主要侧重于组织、分类和资源重用，但现代人工智能（AI）技术的整合已显著改变了这些过程。本文通过考察开源解决方案、商业工具和研究项目，探讨了传统和AI驱动的元数据方法。文章提供了传统和AI驱动的元数据管理方法的比较分析，突显了现有挑战及其对下一代数据集的影响。此外，本文还提出了一种创新的AI辅助元数据管理框架，旨在解决这些挑战。该框架利用更先进的现代AI技术来自动生成元数据、增强治理并提高现代数据集的可访问性和可用性。最后，本文概述了未来的研究和发展方向，提出了在AI驱动创新和复杂数据集背景下进一步推进元数据管理的机会。 

---
# Applying Ensemble Models based on Graph Neural Network and Reinforcement Learning for Wind Power Forecasting 

**Title (ZH)**: 基于图神经网络和强化学习的集成模型在风电预测中的应用 

**Authors**: Hongjin Song, Qianrun Chen, Tianqi Jiang, Yongfeng Li, Xusheng Li, Wenjun Xi, Songtao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16591)  

**Abstract**: Accurately predicting the wind power output of a wind farm across various time scales utilizing Wind Power Forecasting (WPF) is a critical issue in wind power trading and utilization. The WPF problem remains unresolved due to numerous influencing variables, such as wind speed, temperature, latitude, and longitude. Furthermore, achieving high prediction accuracy is crucial for maintaining electric grid stability and ensuring supply security. In this paper, we model all wind turbines within a wind farm as graph nodes in a graph built by their geographical locations. Accordingly, we propose an ensemble model based on graph neural networks and reinforcement learning (EMGRL) for WPF. Our approach includes: (1) applying graph neural networks to capture the time-series data from neighboring wind farms relevant to the target wind farm; (2) establishing a general state embedding that integrates the target wind farm's data with the historical performance of base models on the target wind farm; (3) ensembling and leveraging the advantages of all base models through an actor-critic reinforcement learning framework for WPF. 

**Abstract (ZH)**: 利用风力发电预测（WPF）在多种时间尺度上准确预测风场的风功率输出是风能交易和利用中的关键问题。由于众多影响变量，如风速、温度、纬度和经度，WPF 问题尚未完全解决。同时，实现高预测精度对于维持电网稳定和确保供应安全至关重要。在本文中，我们将风场内的所有风力发电机建模为基于其地理位置构建的图中的图节点。相应地，我们提出了基于图神经网络和强化学习的集成模型（EMGRL）以解决 WPF 问题。我们的方法包括：（1）利用图神经网络捕捉与目标风场相关的邻近风场的时间序列数据；（2）建立一个通用状态嵌入，该嵌入将目标风场的数据与基模型在目标风场上的历史性能综合起来；（3）通过强化学习中的演员-评论家框架将所有基模型的优势进行集成和利用，以提高 WPF 的预测精度。 

---
# Generative AI Uses and Risks for Knowledge Workers in a Science Organization 

**Title (ZH)**: 科学组织中知识工作者的生成型AI应用与风险 

**Authors**: Kelly B. Wagman, Matthew T. Dearing, Marshini Chetty  

**Link**: [PDF](https://arxiv.org/pdf/2501.16577)  

**Abstract**: Generative AI could enhance scientific discovery by supporting knowledge workers in science organizations. However, the real-world applications and perceived concerns of generative AI use in these organizations are uncertain. In this paper, we report on a collaborative study with a US national laboratory with employees spanning Science and Operations about their use of generative AI tools. We surveyed 66 employees, interviewed a subset (N=22), and measured early adoption of an internal generative AI interface called Argo lab-wide. We have four findings: (1) Argo usage data shows small but increasing use by Science and Operations employees; Common current and envisioned use cases for generative AI in this context conceptually fall into either a (2) copilot or (3) workflow agent modality; and (4) Concerns include sensitive data security, academic publishing, and job impacts. Based on our findings, we make recommendations for generative AI use in science and other organizations. 

**Abstract (ZH)**: 生成式人工智能可以通过支持科学组织中的知识工作者来增强科学研究发现。然而，这些组织中生成式人工智能的实际应用及其带来的关切仍然存在不确定性。在本文中，我们报告了一个与美国国家实验室的合作研究结果，该实验室的研究人员来自科学和运营领域。我们对66名员工进行了调查，采访了其中一部分人（共22人），并测量了全实验室范围内内部生成式人工智能界面（Argo）的早期采用情况。我们有四项发现：（1）Argo的使用数据表明，科学和运营部门的员工使用量虽然不大但正在逐渐增加；（2）当前和设想中的生成式人工智能应用场景在概念上可分为辅助型（copilot）或工作流程代理型（workflow agent）两种模式；（3）存在的关切包括敏感数据的安全性、学术出版以及对工作的可能影响。基于这些发现，我们对生成式人工智能在科学研究及其他组织中的使用提出了一些建议。 

---
# Efficient Object Detection of Marine Debris using Pruned YOLO Model 

**Title (ZH)**: 使用精简YOLO模型进行高效的 marine debris 目标检测 

**Authors**: Abi Aryaza, Novanto Yudistira, Tibyani  

**Link**: [PDF](https://arxiv.org/pdf/2501.16571)  

**Abstract**: Marine debris poses significant harm to marine life due to substances like microplastics, polychlorinated biphenyls, and pesticides, which damage habitats and poison organisms. Human-based solutions, such as diving, are increasingly ineffective in addressing this issue. Autonomous underwater vehicles (AUVs) are being developed for efficient sea garbage collection, with the choice of object detection architecture being critical. This research employs the YOLOv4 model for real-time detection of marine debris using the Trash-ICRA 19 dataset, consisting of 7683 images at 480x320 pixels. Various modifications-pretrained models, training from scratch, mosaic augmentation, layer freezing, YOLOv4-tiny, and channel pruning-are compared to enhance architecture efficiency. Channel pruning significantly improves detection speed, increasing the base YOLOv4 frame rate from 15.19 FPS to 19.4 FPS, with only a 1.2% drop in mean Average Precision, from 97.6% to 96.4%. 

**Abstract (ZH)**: 海洋垃圾由于含有微塑料、多氯联苯和农药等物质，对海洋生物造成了严重危害，破坏了栖息地并对生物造成了毒害。基于人类的解决方案，如潜水打捞，越来越难以有效应对这一问题。为此，正在开发自主水下机器人（AUV）用于高效清理海面垃圾，其中对象检测架构的选择至关重要。本研究采用YOLOv4模型，在Trash-ICRA 19数据集上进行实时海洋垃圾检测，该数据集包含480x320像素的7683张图像。通过多种改进方法，包括使用预训练模型、从头开始训练、mosaic增强、层冻结、YOLOv4-tiny以及通道剪枝，来提高架构效率。其中，通道剪枝显著提高了检测速度，将基线YOLOv4的基本帧率从15.19 FPS提升到19.4 FPS，仅使平均精度下降了1.2%，从97.6%下降到96.4%。 

---
# PackDiT: Joint Human Motion and Text Generation via Mutual Prompting 

**Title (ZH)**: PackDiT：通过相互提示联合生成人体动作和文本 

**Authors**: Zhongyu Jiang, Wenhao Chai, Zhuoran Zhou, Cheng-Yen Yang, Hsiang-Wei Huang, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16551)  

**Abstract**: Human motion generation has advanced markedly with the advent of diffusion models. Most recent studies have concentrated on generating motion sequences based on text prompts, commonly referred to as text-to-motion generation. However, the bidirectional generation of motion and text, enabling tasks such as motion-to-text alongside text-to-motion, has been largely unexplored. This capability is essential for aligning diverse modalities and supports unconditional generation. In this paper, we introduce PackDiT, the first diffusion-based generative model capable of performing various tasks simultaneously, including motion generation, motion prediction, text generation, text-to-motion, motion-to-text, and joint motion-text generation. Our core innovation leverages mutual blocks to integrate multiple diffusion transformers (DiTs) across different modalities seamlessly. We train PackDiT on the HumanML3D dataset, achieving state-of-the-art text-to-motion performance with an FID score of 0.106, along with superior results in motion prediction and in-between tasks. Our experiments further demonstrate that diffusion models are effective for motion-to-text generation, achieving performance comparable to that of autoregressive models. 

**Abstract (ZH)**: 自扩散模型的出现极大地推动了人体运动生成的发展。最近的研究主要集中在基于文本提示生成运动序列，通常称为文本到运动生成。然而，双向生成运动和文本的任务，比如运动到文本和文本到运动的相互转换，迄今尚未得到充分探索。这种能力对于对齐多种模态数据和实现无条件生成至关重要。本文中，我们介绍了PackDiT，这是第一个能够同时执行多种任务的基于扩散模型的生成模型，包括运动生成、运动预测、文本生成、文本到运动、运动到文本以及联合运动-文本生成。我们的主要创新在于引入了互换块，以无缝地集成不同模态的多个扩散变换器（DiTs）。在HumanML3D数据集上训练PackDiT，我们实现了最先进的文本到运动性能，FID得分为0.106，并且在运动预测和中间任务上也取得了优异结果。我们的实验进一步表明，扩散模型在运动到文本生成任务上也是有效的，其性能可与自回归模型相媲美。 

---
# Generalized Mission Planning for Heterogeneous Multi-Robot Teams via LLM-constructed Hierarchical Trees 

**Title (ZH)**: 基于LLM构建的层次树的异构多机器人团队通用任务规划 

**Authors**: Piyush Gupta, David Isele, Enna Sachdeva, Pin-Hao Huang, Behzad Dariush, Kwonjoon Lee, Sangjae Bae  

**Link**: [PDF](https://arxiv.org/pdf/2501.16539)  

**Abstract**: We present a novel mission-planning strategy for heterogeneous multi-robot teams, taking into account the specific constraints and capabilities of each robot. Our approach employs hierarchical trees to systematically break down complex missions into manageable sub-tasks. We develop specialized APIs and tools, which are utilized by Large Language Models (LLMs) to efficiently construct these hierarchical trees. Once the hierarchical tree is generated, it is further decomposed to create optimized schedules for each robot, ensuring adherence to their individual constraints and capabilities. We demonstrate the effectiveness of our framework through detailed examples covering a wide range of missions, showcasing its flexibility and scalability. 

**Abstract (ZH)**: 我们提出了一种新颖的异构多机器人团队任务规划策略，充分考虑了每台机器人特有的约束和能力。该方法采用层次树结构系统地将复杂任务分解为可管理的子任务。我们开发了专门的API和工具，这些工具被大型语言模型（LLMs）利用以高效地构建这些层次树。一旦生成了层次树，它就会进一步分解，从而为每台机器人创建优化的时间表，确保遵守它们各自的约束和能力。我们通过详尽的例子展示了该框架的有效性，覆盖了广泛的任务类型，展示了其灵活性和可扩展性。 

---
# Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs 

**Title (ZH)**: 针对对齐的目标：提取对齐的大型语言模型的安全分类器 

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel  

**Link**: [PDF](https://arxiv.org/pdf/2501.16534)  

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we present and evaluate a method to assess the robustness of LLM alignment. We observe that alignment embeds a safety classifier in the target model that is responsible for deciding between refusal and compliance. We seek to extract an approximation of this classifier, called a surrogate classifier, from the LLM. We develop an algorithm for identifying candidate classifiers from subsets of the LLM model. We evaluate the degree to which the candidate classifiers approximate the model's embedded classifier in benign (F1 score) and adversarial (using surrogates in a white-box attack) settings. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find attacks mounted on the surrogate models can be transferred with high accuracy. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70%, a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is a viable (and highly effective) means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks. 

**Abstract (ZH)**: 大语言模型（LLMs）中的对齐用于实施安全性等指导原则。然而，对齐在面对修改输入以诱导不安全输出的监狱突破攻击时会失效。在本文中，我们提出并评估了一种评估LLM对齐鲁棒性的方法。我们观察到对齐在目标模型中嵌入了一个安全分类器，该分类器负责在拒绝与合规之间做出决定。我们寻求从LLM中提取一个被称为替代分类器的近似表示。我们开发了一种算法，用于识别来自LLM模型子集的候选分类器。我们评估了候选分类器在良性（F1分值）和对抗（使用替代模型进行白盒攻击）情况下与模型嵌入分类器的一致性程度。我们的评估结果显示，在仅使用模型架构的20%的情况下，最佳候选分类器能够实现准确的一致性（F1分值超过80%）。此外，我们发现针对替代模型的攻击可以以高精度进行转移。例如，仅使用Llama 2模型50%的替代模型实现了70%的成功攻击率（ASR），远高于直接攻击LLM的情况，我们仅观察到了22%的ASR。这些结果表明，提取替代分类器是一种可行且非常有效的建模方法，可用于建模并进而解决对齐模型对抗监狱突破攻击的脆弱性。 

---
# Multi-Objective Deep-Learning-based Biomechanical Deformable Image Registration with MOREA 

**Title (ZH)**: 基于MOREA的多目标深度学习生物力学可变形图像配准

解释：
- "Multi-Objective" 表示多目标。
- "Deep-Learning-based" 表示基于深度学习。
- "Biomechanical Deformable Image Registration" 表示生物力学可变形图像配准。
- "MOREA" 保持不变，因为这是一个特定的技术或算法的缩写。如果 MOREA 有具体的中文解释或替代名称，可以根据具体情况进行调整。

这样的翻译既保持了原意，也符合学术规范。 

**Authors**: Georgios Andreadis, Eduard Ruiz Munné, Thomas H. W. Bäck, Peter A. N. Bosman, Tanja Alderliesten  

**Link**: [PDF](https://arxiv.org/pdf/2501.16525)  

**Abstract**: When choosing a deformable image registration (DIR) approach for images with large deformations and content mismatch, the realism of found transformations often needs to be traded off against the required runtime. DIR approaches using deep learning (DL) techniques have shown remarkable promise in instantly predicting a transformation. However, on difficult registration problems, the realism of these transformations can fall short. DIR approaches using biomechanical, finite element modeling (FEM) techniques can find more realistic transformations, but tend to require much longer runtimes. This work proposes the first hybrid approach to combine them, with the aim of getting the best of both worlds. This hybrid approach, called DL-MOREA, combines a recently introduced multi-objective DL-based DIR approach which leverages the VoxelMorph framework, called DL-MODIR, with MOREA, an evolutionary algorithm-based, multi-objective DIR approach in which a FEM-like biomechanical mesh transformation model is used. In our proposed hybrid approach, the DL results are used to smartly initialize MOREA, with the aim of more efficiently optimizing its mesh transformation model. We empirically compare DL-MOREA against its components, DL-MODIR and MOREA, on CT scan pairs capturing large bladder filling differences of 15 cervical cancer patients. While MOREA requires a median runtime of 45 minutes, DL-MOREA can already find high-quality transformations after 5 minutes. Compared to the DL-MODIR transformations, the transformations found by DL-MOREA exhibit far less folding and improve or preserve the bladder contour distance error. 

**Abstract (ZH)**: 在选择用于大形变和内容不匹配图像的弹性图像注册（DIR）方法时，通常需要在找到的变换真度与所需运行时间之间做出权衡。使用深度学习（DL）技术的DIR方法在即时预测变换方面表现出显著的潜力。然而，在复杂的配准问题上，这些变换的真实度可能会逊色。使用生物力学和有限元建模（FEM）技术的DIR方法可以找到更现实的变换，但往往需要更长的运行时间。本项研究提出了一种新的混合方法，旨在兼得两种方法的优点。该混合方法名为DL-MOREA，它将一种基于深度学习多目标的DIR方法（利用VoxelMorph框架，称为DL-MODIR）与利用类似FEM的生物力学网格变换模型的多目标进化算法（MOREA）结合在一起。在我们提出的混合方法中，利用DL结果智能初始化MOREA，目的是更高效地优化其网格变换模型。我们在15例宫颈癌患者的大膀胱充盈差异CT扫描对上，通过实验证明了DL-MOREA相对于其组成部分DL-MODIR和MOREA的效果。虽然MOREA的中位运行时间需要45分钟，但DL-MOREA仅用了5分钟就已经找到了高质量的变换。相比之下，与DL-MODIR的变换相比，DL-MOREA找到的变换少了许多褶皱现象，并且要么保持了膀胱轮廓距离误差要么改善了它。 

---
# How well can LLMs Grade Essays in Arabic? 

**Title (ZH)**: 大型语言模型在评定阿拉伯语作文方面表现如何？ 

**Authors**: Rayed Ghazawi, Edwin Simpson  

**Link**: [PDF](https://arxiv.org/pdf/2501.16516)  

**Abstract**: This research assesses the effectiveness of state-of-the-art large language models (LLMs), including ChatGPT, Llama, Aya, Jais, and ACEGPT, in the task of Arabic automated essay scoring (AES) using the AR-AES dataset. It explores various evaluation methodologies, including zero-shot, few-shot in-context learning, and fine-tuning, and examines the influence of instruction-following capabilities through the inclusion of marking guidelines within the prompts. A mixed-language prompting strategy, integrating English prompts with Arabic content, was implemented to improve model comprehension and performance. Among the models tested, ACEGPT demonstrated the strongest performance across the dataset, achieving a Quadratic Weighted Kappa (QWK) of 0.67, but was outperformed by a smaller BERT-based model with a QWK of 0.88. The study identifies challenges faced by LLMs in processing Arabic, including tokenization complexities and higher computational demands. Performance variation across different courses underscores the need for adaptive models capable of handling diverse assessment formats and highlights the positive impact of effective prompt engineering on improving LLM outputs. To the best of our knowledge, this study is the first to empirically evaluate the performance of multiple generative Large Language Models (LLMs) on Arabic essays using authentic student data. 

**Abstract (ZH)**: 本研究评估了最先进的大型语言模型（LLMs），包括ChatGPT、Llama、Aya、Jais和ACEGPT，在阿拉伯语自动化作文评分（AES）任务中的有效性，使用了AR-AES数据集。研究探讨了多种评估方法，包括零样本、少样本上下文学习和微调，并通过在提示中包含评分指南来研究模型遵循指令能力的影响。研究实施了混合语言提示策略，结合英语提示与阿拉伯语内容，以提高模型的理解能力和性能。在测试的模型中，ACEGPT在数据集中表现出最强的性能，获得了0.67的二次加权Kappa（QWK）值，但被一个较小的基于BERT的模型以0.88的QWK值超越。研究指出了LLMs在处理阿拉伯语时面临的挑战，包括分词复杂性和更高的计算需求。不同课程间性能的差异强调了需要能够处理多种评估形式的自适应模型，并突显了有效提示工程对提高LLM输出的积极作用。据我们所知，本研究是首次使用真实学生的阿拉伯语作文数据，对多个生成型大型语言模型的性能进行实证评估的研究。 

---
# Decrypting the temperature field in flow boiling with latent diffusion models 

**Title (ZH)**: 利用潜在扩散模型解析流动沸腾的温度场 

**Authors**: UngJin Na, JunYoung Seo, Taeil Kim, ByongGuk Jeon, HangJin Jo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16510)  

**Abstract**: This paper presents an innovative method using Latent Diffusion Models (LDMs) to generate temperature fields from phase indicator maps. By leveraging the BubbleML dataset from numerical simulations, the LDM translates phase field data into corresponding temperature distributions through a two-stage training process involving a vector-quantized variational autoencoder (VQVAE) and a denoising autoencoder. The resulting model effectively reconstructs complex temperature fields at interfaces. Spectral analysis indicates a high degree of agreement with ground truth data in the low to mid wavenumber ranges, even though some inconsistencies are observed at higher wavenumbers, suggesting areas for further enhancement. This machine learning approach significantly reduces the computational burden of traditional simulations and improves the precision of experimental calibration methods. Future work will focus on refining the model's ability to represent small-scale turbulence and expanding its applicability to a broader range of boiling conditions. 

**Abstract (ZH)**: 本文介绍了使用潜在扩散模型（LDMs）从相指示图生成温度场的一种创新方法。通过利用来自数值模拟的BubbleML数据集，LDM通过一个包括向量量化变分自编码器（VQVAE）和去噪自编码器的两阶段训练过程，将相场数据转换为相应的温度分布。由此生成的模型能够有效地恢复界面处复杂的温度场。频谱分析表明，在低到中频波数范围内，该模型与真实数据高度一致，尽管在高频波数范围内观察到一些不一致，这表明存在进一步改进的空间。这种机器学习方法显著减少了传统模拟的计算负担，并提高了实验校准方法的精度。未来的工作将致力于改进模型对小尺度湍流的表征能力，并使其适用于更广泛的沸腾条件。 

---
# Reinforcement Learning for Quantum Circuit Design: Using Matrix Representations 

**Title (ZH)**: 量子电路设计中的强化学习：使用矩阵表示方法 

**Authors**: Zhiyuan Wang, Chunlin Feng, Christopher Poon, Lijian Huang, Xingjian Zhao, Yao Ma, Tianfan Fu, Xiao-Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16509)  

**Abstract**: Quantum computing promises advantages over classical computing. The manufacturing of quantum hardware is in the infancy stage, called the Noisy Intermediate-Scale Quantum (NISQ) era. A major challenge is automated quantum circuit design that map a quantum circuit to gates in a universal gate set. In this paper, we present a generic MDP modeling and employ Q-learning and DQN algorithms for quantum circuit design. By leveraging the power of deep reinforcement learning, we aim to provide an automatic and scalable approach over traditional hand-crafted heuristic methods. 

**Abstract (ZH)**: 量子计算在某些任务上有望超越经典计算。量子硬件的制造正处于初级阶段，称为嘈杂中等规模量子（NISQ）时代。一个主要的挑战是如何自动设计量子电路，将量子电路映射到通用门集中。本文中，我们提出了一种通用的马尔可夫决策过程（MDP）建模方法，并采用Q学习和深度Q网络（DQN）算法进行量子电路设计。通过利用深度强化学习的强大功能，我们旨在提供一种自动化的、可扩展的方法，超越传统的手工构造启发式方法。 

---
# Characterizing Network Structure of Anti-Trans Actors on TikTok 

**Title (ZH)**: Characterizing Anti-Trans Actors' Network Structure on TikTok 

**Authors**: Maxyn Leitner, Rebecca Dorn, Fred Morstatter, Kristina Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2501.16507)  

**Abstract**: The recent proliferation of short form video social media sites such as TikTok has been effectively utilized for increased visibility, communication, and community connection amongst trans/nonbinary creators online. However, these same platforms have also been exploited by right-wing actors targeting trans/nonbinary people, enabling such anti-trans actors to efficiently spread hate speech and propaganda. Given these divergent groups, what are the differences in network structure between anti-trans and pro-trans communities on TikTok, and to what extent do they amplify the effects of anti-trans content? In this paper, we collect a sample of TikTok videos containing pro and anti-trans content, and develop a taxonomy of trans related sentiment to enable the classification of content on TikTok, and ultimately analyze the reply network structures of pro-trans and anti-trans communities. In order to accomplish this, we worked with hired expert data annotators from the trans/nonbinary community in order to generate a sample of highly accurately labeled data. From this subset, we utilized a novel classification pipeline leveraging Retrieval-Augmented Generation (RAG) with annotated examples and taxonomy definitions to classify content into pro-trans, anti-trans, or neutral categories. We find that incorporating our taxonomy and its logics into our classification engine results in improved ability to differentiate trans related content, and that Results from network analysis indicate many interactions between posters of pro-trans and anti-trans content exist, further demonstrating targeting of trans individuals, and demonstrating the need for better content moderation tools 

**Abstract (ZH)**: 近年来，TikTok等短视頻社交媒体平台的广泛使用，极大地提升了跨性别/非二元性别创作者在线的可见度、沟通和社区连接。然而，这些平台也被右翼分子不当利用，针对跨性别/非二元性别人士传播仇恨言论和 propaganda。面对这些对立群体，TikTok上反跨性别和亲跨性别社区的网络结构有何异同？反跨性别人士内容的影响又在多大程度上被这些社区放大？本文收集了包含亲跨性别和反跨性别内容的TikTok视频样本，并制定了一个关于跨性别相关情感的分类体系，以帮助对TikTok内容进行分类，最终分析亲跨性别和反跨性别社区的回复网络结构。为了实现这一目标，我们与来自跨性别/非二元性别社区的专业数据注释员合作，生成了一大批准确标注的数据样本。从这些样本中，我们采用了结合了检索增强生成（RAG）技术、带有注释示例和分类体系定义的新分类流水线，将内容分类为亲跨性别、反跨性别或中立类别。研究结果显示，将我们的分类体系及其逻辑纳入分类引擎中，显著提高了区分跨性别相关内容的能力。网络分析结果显示，发布亲跨性别和反跨性别内容的用户之间存在许多互动，进一步证明了对跨性别个体的针对性攻击，也体现了需要开发更好的内容审核工具的必要性。 

---
# Digital Twin Enabled Site Specific Channel Precoding: Over the Air CIR Inference 

**Title (ZH)**: 基于数字孪生的站点特定信道预编码：空中信道响应推断 

**Authors**: Majumder Haider, Imtiaz Ahmed, Zoheb Hassan, Timothy J. O'Shea, Lingjia Liu, Danda B. Rawat  

**Link**: [PDF](https://arxiv.org/pdf/2501.16504)  

**Abstract**: This paper investigates the significance of designing a reliable, intelligent, and true physical environment-aware precoding scheme by leveraging an accurately designed channel twin model to obtain realistic channel state information (CSI) for cellular communication systems. Specifically, we propose a fine-tuned multi-step channel twin design process that can render CSI very close to the CSI of the actual environment. After generating a precise CSI, we execute precoding using the obtained CSI at the transmitter end. We demonstrate a two-step parameters' tuning approach to design channel twin by ray tracing (RT) emulation, then further fine-tuning of CSI by employing an artificial intelligence (AI) based algorithm can significantly reduce the gap between actual CSI and the fine-tuned digital twin (DT) rendered CSI. The simulation results show the effectiveness of the proposed novel approach in designing a true physical environment-aware channel twin model. 

**Abstract (ZH)**: 本文探究了通过利用准确设计的信道孪生模型来获取真实的信道状态信息（CSI），从而设计出可靠、智能且真正反映物理环境的预编码方案的重要性。具体而言，我们提出了一种精细调节的多步骤信道孪生设计过程，能够使CSI接近实际环境中的CSI。在生成准确的CSI后，我们在发射端利用获得的CSI执行预编码。我们通过射线追踪（RT）仿真实现了信道孪生的两步参数调谐方法，然后采用基于人工智能（AI）的算法进一步精细调节CSI，可以显著缩小实际CSI与精细调节的数字孪生（DT）渲染CSI之间的差距。仿真结果表明，所提出的新型方法在设计真正反映物理环境的信道孪生模型方面具有有效性。 

---
# Smoothed Embeddings for Robust Language Models 

**Title (ZH)**: 光滑嵌入以构建稳健的语言模型 

**Authors**: Ryo Hase, Md Rafi Ur Rashid, Ashley Lewis, Jing Liu, Toshiaki Koike-Akino, Kieran Parsons, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16497)  

**Abstract**: Improving the safety and reliability of large language models (LLMs) is a crucial aspect of realizing trustworthy AI systems. Although alignment methods aim to suppress harmful content generation, LLMs are often still vulnerable to jailbreaking attacks that employ adversarial inputs that subvert alignment and induce harmful outputs. We propose the Randomized Embedding Smoothing and Token Aggregation (RESTA) defense, which adds random noise to the embedding vectors and performs aggregation during the generation of each output token, with the aim of better preserving semantic information. Our experiments demonstrate that our approach achieves superior robustness versus utility tradeoffs compared to the baseline defenses. 

**Abstract (ZH)**: 提高大型语言模型（LLMs）的安全性和可靠性是实现可信AI系统的关键方面。尽管对齐方法旨在抑制有害内容的生成，但LLMs仍然容易受到利用对抗输入的逃逸攻击的影响，这些对抗输入会破坏对齐并引发有害输出。我们提出了一种名为随机嵌入平滑和标记聚合（RESTA）的防御方法，该方法在生成每个输出标记时添加随机噪声并进行聚合，旨在更好地保持语义信息。我们的实验表明，与基准防御方法相比，我们的方法在鲁棒性与实用性之间的权衡上表现更优。 

---
# Towards Robust Stability Prediction in Smart Grids: GAN-based Approach under Data Constraints and Adversarial Challenges 

**Title (ZH)**: 面向智能电网鲁棒稳定性预测的研究：基于生成对抗网络的方法在数据约束和对抗性挑战下的应用 

**Authors**: Emad Efatinasab, Alessandro Brighente, Denis Donadel, Mauro Conti, Mirco Rampazzo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16490)  

**Abstract**: Smart grids are critical for addressing the growing energy demand due to global population growth and urbanization. They enhance efficiency, reliability, and sustainability by integrating renewable energy. Ensuring their availability and safety requires advanced operational control and safety measures. Researchers employ AI and machine learning to assess grid stability, but challenges like the lack of datasets and cybersecurity threats, including adversarial attacks, persist. In particular, data scarcity is a key issue: obtaining grid instability instances is tough due to the need for significant expertise, resources, and time. However, they are essential to test novel research advancements and security mitigations. In this paper, we introduce a novel framework to detect instability in smart grids by employing only stable data. It relies on a Generative Adversarial Network (GAN) where the generator is trained to create instability data that are used along with stable data to train the discriminator. Moreover, we include a new adversarial training layer to improve robustness against adversarial attacks. Our solution, tested on a dataset composed of real-world stable and unstable samples, achieve accuracy up to 97.5\% in predicting grid stability and up to 98.9\% in detecting adversarial attacks. Moreover, we implemented our model in a single-board computer demonstrating efficient real-time decision-making with an average response time of less than 7ms. Our solution improves prediction accuracy and resilience while addressing data scarcity in smart grid management. 

**Abstract (ZH)**: 智能电网对于应对全球人口增长和城市化导致的日益增长的能源需求至关重要。它们通过整合可再生能源来提高效率、可靠性和可持续性。确保其可用性和安全性需要先进的操作控制和安全措施。研究人员利用人工智能和机器学习评估电网稳定性，但仍然面临数据集不足和网络安全威胁（包括对抗性攻击）等挑战。特别是数据稀缺是一个关键问题：获得电网不稳定性实例非常困难，因为需要大量的专业知识、资源和时间。然而，这些实例对于测试新兴的研究成果和安全缓解措施至关重要。在本文中，我们提出了一种新的框架，通过仅使用稳定数据来检测智能电网的不稳定性。该框架依赖于生成对抗网络（GAN），其中生成器训练生成的不稳定性数据与稳定数据一起用于训练鉴别器。此外，我们引入了一种新的对抗训练层，以提高对对抗性攻击的鲁棒性。我们的解决方案在包含真实世界稳定和不稳定样本的数据集上进行了测试，其预测电网稳定性准确率达到97.5%，检测对抗性攻击的准确率达到98.9%。此外，我们在一款单板计算机上实现了我们的模型，展示了高效的实时决策能力，平均响应时间为不到7毫秒。我们的解决方案在智能电网管理中提高了预测准确性和鲁棒性，同时解决了数据稀缺问题。 

---
# SIM: Surface-based fMRI Analysis for Inter-Subject Multimodal Decoding from Movie-Watching Experiments 

**Title (ZH)**: SIM：基于表面的功能磁共振成像分析在电影观看实验中实现跨被试多模态解码 

**Authors**: Simon Dahan, Gabriel Bénédict, Logan Z. J. Williams, Yourong Guo, Daniel Rueckert, Robert Leech, Emma C. Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2501.16471)  

**Abstract**: Current AI frameworks for brain decoding and encoding, typically train and test models within the same datasets. This limits their utility for brain computer interfaces (BCI) or neurofeedback, for which it would be useful to pool experiences across individuals to better simulate stimuli not sampled during training. A key obstacle to model generalisation is the degree of variability of inter-subject cortical organisation, which makes it difficult to align or compare cortical signals across participants. In this paper we address this through the use of surface vision transformers, which build a generalisable model of cortical functional dynamics, through encoding the topography of cortical networks and their interactions as a moving image across a surface. This is then combined with tri-modal self-supervised contrastive (CLIP) alignment of audio, video, and fMRI modalities to enable the retrieval of visual and auditory stimuli from patterns of cortical activity (and vice-versa). We validate our approach on 7T task-fMRI data from 174 healthy participants engaged in the movie-watching experiment from the Human Connectome Project (HCP). Results show that it is possible to detect which movie clips an individual is watching purely from their brain activity, even for individuals and movies not seen during training. Further analysis of attention maps reveals that our model captures individual patterns of brain activity that reflect semantic and visual systems. This opens the door to future personalised simulations of brain function. Code & pre-trained models will be made available at this https URL, processed data for training will be available upon request at this https URL. 

**Abstract (ZH)**: 当前的脑解码和编码AI框架通常在相同的数据集中训练和测试模型，这限制了它们在脑机接口(BCI)或神经反馈中的应用，因为在这种应用中，跨个体汇总经验可以更好地模拟训练过程中未采样的刺激。模型泛化的一个主要障碍是跨个体皮层组织的变异程度，这使得难以对参与者之间的皮层信号进行对齐或比较。在这篇论文中，我们通过使用表面视网膜变换器来解决这一问题，它通过将皮层网络及其交互的拓扑结构编码为表面上的移动图像，构建了一个可泛化的皮层功能动态模型。然后，结合针对音频、视频和fMRI模态的三模态自监督对比(CLIP)对齐，使我们能够从皮层活动模式中检索视觉和听觉刺激（反之亦然）。我们在人类连通体项目(HCP)的174名健康参与者观看电影实验的7T任务fMRI数据上验证了这种方法。结果显示，仅从个体的脑活动就可以检测出他们在观看的电影剪辑，即使这些电影剪辑和个体在训练过程中未见过也是如此。进一步分析注意力图显示，我们的模型捕捉到了反映了语义和视觉系统的个体脑活动模式。这一发现为未来个性化模拟脑功能打开了新的大门。相关代码及预训练模型将在此 https://链接 发布，训练所需的处理数据将在收到请求后提供此 https://链接。 

---
# On the Feasibility of Using LLMs to Execute Multistage Network Attacks 

**Title (ZH)**: 使用大型语言模型执行多阶段网络攻击的可行性研究 

**Authors**: Brian Singer, Keane Lucas, Lakshmi Adiga, Meghna Jain, Lujo Bauer, Vyas Sekar  

**Link**: [PDF](https://arxiv.org/pdf/2501.16466)  

**Abstract**: LLMs have shown preliminary promise in some security tasks and CTF challenges. However, it is unclear whether LLMs are able to realize multistage network attacks, which involve executing a wide variety of actions across multiple hosts such as conducting reconnaissance, exploiting vulnerabilities to gain initial access, leveraging internal hosts to move laterally, and using multiple compromised hosts to exfiltrate data. We evaluate LLMs across 10 multistage networks and find that popular LLMs are unable to realize these attacks. To enable LLMs to realize these attacks, we introduce Incalmo, an LLM-agnostic high-level attack abstraction layer that sits between an LLM and the environment. Rather than LLMs issuing low-level command-line instructions, which can lead to incorrect implementations, Incalmo allows LLMs to specify high-level tasks (e.g., infect a host, scan a network), which are then carried out by Incalmo. Incalmo realizes these tasks by translating them into low-level primitives (e.g., commands to exploit tools). Incalmo also provides an environment state service and an attack graph service to provide structure to LLMs in selecting actions relevant to a multistage attack. Across 9 out of 10 realistic emulated networks (from 25 to 50 hosts), LLMs using Incalmo can successfully autonomously execute multistage attacks. We also conduct an ablation analysis to show the key role the high-level abstractions play. For instance, we find that both Incalmo's high-level tasks and services are crucial. Furthermore, even smaller-parameter LLMs with Incalmo can fully succeed in 5 of 10 environments, while larger-parameter LLMs without Incalmo do not fully succeed in any. 

**Abstract (ZH)**: 大语言模型（LLMs）在某些安全任务和CTF挑战中显示出了初步的潜力。然而，目前尚不清楚LLMs是否能够实现多阶段网络攻击，这种攻击涉及在多个主机上执行多种操作，如进行侦查、利用漏洞获取初始访问权限、利用内部主机横向移动，以及使用多个被攻陷的主机窃取数据。我们评估了LLMs在10个不同的多阶段网络环境中，发现现有的流行LLMs无法实现这些攻击。为使LLMs能够实现这些攻击，我们提出了Incalmo，这是LLM无关的高性能攻击抽象层，位于LLM与环境之间。与LLM直接发布低层级命令线指令不同，Incalmo允许LLM指定高层级任务（例如，感染主机、扫描网络），随后由Incalmo来执行这些任务。Incalmo通过将任务转换为低层级原语（例如，利用工具的命令）来实现这些任务。此外，Incalmo还提供了环境状态服务和攻击图谱服务，以帮助LLM在选择与多阶段攻击相关的操作时提供结构化支持。在9个真实的模拟网络（每个网络包含25到50个主机）中，使用Incalmo的LLMs能够自主成功执行多阶段攻击。我们还进行了消融分析，以展示高层级抽象的关键作用。例如，我们发现Incalmo的高层级任务和服务都是至关重要的。此外，即使在包含较小参数的LLM中使用Incalmo，可以在5个环境中完全成功，而没有Incalmo的支持，更大参数的LLM则在所有环境中都无法完全成功。 

---
# Detecting Zero-Day Attacks in Digital Substations via In-Context Learning 

**Title (ZH)**: 通过上下文学习检测数字变电站中的零日攻击 

**Authors**: Faizan Manzoor, Vanshaj Khattar, Akila Herath, Clifton Black, Matthew C Nielsen, Junho Hong, Chen-Ching Liu, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2501.16453)  

**Abstract**: The occurrences of cyber attacks on the power grids have been increasing every year, with novel attack techniques emerging every year. In this paper, we address the critical challenge of detecting novel/zero-day attacks in digital substations that employ the IEC-61850 communication protocol. While many heuristic and machine learning (ML)-based methods have been proposed for attack detection in IEC-61850 digital substations, generalization to novel or zero-day attacks remains challenging. We propose an approach that leverages the in-context learning (ICL) capability of the transformer architecture, the fundamental building block of large language models. The ICL approach enables the model to detect zero-day attacks and learn from a few examples of that attack without explicit retraining. Our experiments on the IEC-61850 dataset demonstrate that the proposed method achieves more than $85\%$ detection accuracy on zero-day attacks while the existing state-of-the-art baselines fail. This work paves the way for building more secure and resilient digital substations of the future. 

**Abstract (ZH)**: 近年来，针对电力网络的网络攻击事件逐年增加，伴随而来的是不断涌现的新颖攻击技术。本文针对采用IEC-61850通信协议的数字变电站中新颖/零日攻击检测这一关键挑战进行了探讨。虽然已提出了多种基于启发式方法和机器学习（ML）的攻击检测方法用于IEC-61850数字变电站，但针对新颖或零日攻击的泛化仍具有挑战性。本文提出了一种方法，利用Transformer架构的基本构建模块——上下文内学习（Contextual Learning Within the Context, ICL）的能力。这种方法使模型能够在未进行明确重训练的情况下，检测到零日攻击并仅从少量该攻击的示例中学习。在IEC-61850数据集上的实验结果显示，所提出的方法在零日攻击检测中的准确率超过85%，而现有的先进基准方法则未能实现这一目标。本研究为构建更加安全且更具弹性的未来数字变电站奠定了基础。 

---
# 360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation 

**Title (ZH)**: 360Brew：一种用于个性化排名和推荐的仅Decoder基础模型 

**Authors**: Hamed Firooz, Maziar Sanjabi, Adrian Englhardt, Aman Gupta, Ben Levine, Dre Olgiati, Gungor Polatkan, Iuliia Melnychuk, Karthik Ramgopal, Kirill Talanine, Kutta Srinivasan, Luke Simon, Natesh Sivasubramoniapillai, Necip Fazil Ayan, Qingquan Song, Samira Sriram, Souvik Ghosh, Tao Song, Vignesh Kothapalli, Xiaoling Zhai, Ya Xu, Yu Wang, Yun Dai  

**Link**: [PDF](https://arxiv.org/pdf/2501.16450)  

**Abstract**: Ranking and recommendation systems are the foundation for numerous online experiences, ranging from search results to personalized content delivery. These systems have evolved into complex, multilayered architectures that leverage vast datasets and often incorporate thousands of predictive models. The maintenance and enhancement of these models is a labor intensive process that requires extensive feature engineering. This approach not only exacerbates technical debt but also hampers innovation in extending these systems to emerging problem domains. In this report, we present our research to address these challenges by utilizing a large foundation model with a textual interface for ranking and recommendation tasks. We illustrate several key advantages of our approach: (1) a single model can manage multiple predictive tasks involved in ranking and recommendation, (2) decoder models with textual interface due to their comprehension of reasoning capabilities, can generalize to new recommendation surfaces and out-of-domain problems, and (3) by employing natural language interfaces for task definitions and verbalizing member behaviors and their social connections, we eliminate the need for feature engineering and the maintenance of complex directed acyclic graphs of model dependencies. We introduce our research pre-production model, 360Brew V1.0, a 150B parameter, decoder-only model that has been trained and fine-tuned on LinkedIn's data and tasks. This model is capable of solving over 30 predictive tasks across various segments of the LinkedIn platform, achieving performance levels comparable to or exceeding those of current production systems based on offline metrics, without task-specific fine-tuning. Notably, each of these tasks is conventionally addressed by dedicated models that have been developed and maintained over multiple years by teams of a similar or larger size than our own. 

**Abstract (ZH)**: 排名和推荐系统是众多在线体验的基础，从搜索结果到个性化内容交付不一而足。这些系统已经发展成为复杂且多层次的架构，利用了大量数据集，并常涉及数千个预测模型。这些模型的维护和优化是一个劳动密集型的过程，需要大量的特征工程。这种方法不仅加剧了技术债务，还阻碍了将这些系统扩展到新兴问题领域的创新。在本报告中，我们展示了如何通过使用具有文本界面的大规模基础模型来解决这些挑战，以应对排名和推荐任务。我们概述了这种方法的几个关键优势：（1）单个模型可以管理排名和推荐过程中的多个预测任务；（2）基于文本界面的解码器模型因其理解推理能力，能够在新的推荐界面和跨领域的挑战上进行泛化，并表现出色；（3）通过使用自然语言界面来定义任务，并通过人类语言来表述成员行为及其社会联系，我们无需进行特征工程，也无需维护复杂的定向无环图依赖模型。我们介绍了我们的研究预生产模型——360Brew V1.0，这是一个包含150亿参数的仅解码器模型，已在LinkedIn的数据和任务上进行过训练和微调。该模型能够在LinkedIn平台的各个细分领域中解决超过30个预测任务，并且在离线性能指标上达到了当前生产系统的水平，无需针对特定任务进行微调。值得注意的是，每一个此类任务通常都是由与我们团队规模相似或更大的团队，在数年的时间内开发和维护专用模型来解决的。 

---
# PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding 

**Title (ZH)**: PhysBench：视觉-语言模型在物理世界理解中的benchmarking与增强 

**Authors**: Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16411)  

**Abstract**: Understanding the physical world is a fundamental challenge in embodied AI, critical for enabling agents to perform complex tasks and operate safely in real-world environments. While Vision-Language Models (VLMs) have shown great promise in reasoning and task planning for embodied agents, their ability to comprehend physical phenomena remains extremely limited. To close this gap, we introduce PhysBench, a comprehensive benchmark designed to evaluate VLMs' physical world understanding capability across a diverse set of tasks. PhysBench contains 100,000 entries of interleaved video-image-text data, categorized into four major domains: physical object properties, physical object relationships, physical scene understanding, and physics-based dynamics, further divided into 19 subclasses and 8 distinct capability dimensions. Our extensive experiments, conducted on 75 representative VLMs, reveal that while these models excel in common-sense reasoning, they struggle with understanding the physical world -- likely due to the absence of physical knowledge in their training data and the lack of embedded physical priors. To tackle the shortfall, we introduce PhysAgent, a novel framework that combines the generalization strengths of VLMs with the specialized expertise of vision models, significantly enhancing VLMs' physical understanding across a variety of tasks, including an 18.4\% improvement on GPT-4o. Furthermore, our results demonstrate that enhancing VLMs' physical world understanding capabilities can help embodied agents such as MOKA. We believe that PhysBench and PhysAgent offer valuable insights and contribute to bridging the gap between VLMs and physical world understanding. 

**Abstract (ZH)**: 理解物理世界是实体人工智能中的一个基本挑战，对于使智能体能够执行复杂任务并在现实世界环境中安全操作至关重要。尽管视觉语言模型（VLMs）在实体智能体的推理和任务规划方面显示出巨大的潜力，但它们理解物理现象的能力仍然极为有限。为解决这一问题，我们介绍了PhysBench，这是一个综合基准，旨在评估VLMs在多样化的任务中理解物理世界的能力。PhysBench包含100,000条交错的视频-图像-文本数据，分为四大领域：物理对象属性、物理对象关系、物理场景理解以及物理基础动力学，进一步细分为19个子类别和8个不同的能力维度。我们在75个代表性VLMs上进行的广泛实验表明，尽管这些模型在常识推理方面表现出色，但在理解物理世界方面却存在问题——这可能是由于其训练数据中缺乏物理知识以及嵌入的物理先验知识不足所致。为解决这一不足，我们提出了PhysAgent，这是一种结合了VLMs的一般化优势和视觉模型的专门化优势的新框架，显著提升了不同任务中VLMs的物理理解能力，包括在GPT-4o上的18.4%的改进。此外，我们的结果表明，增强VLMs的物理世界理解能力能够帮助实体智能体如MOKA。我们认为，PhysBench和PhysAgent提供了宝贵的见解，并有助于弥合VLMs与物理世界理解之间的差距。 

---
# Classification of Mild Cognitive Impairment Based on Dynamic Functional Connectivity Using Spatio-Temporal Transformer 

**Title (ZH)**: 基于时空变压器的轻度认知损害分类研究 

**Authors**: Jing Zhang, Yanjun Lyu, Xiaowei Yu, Lu Zhang, Chao Cao, Tong Chen, Minheng Chen, Yan Zhuang, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16409)  

**Abstract**: Dynamic functional connectivity (dFC) using resting-state functional magnetic resonance imaging (rs-fMRI) is an advanced technique for capturing the dynamic changes of neural activities, and can be very useful in the studies of brain diseases such as Alzheimer's disease (AD). Yet, existing studies have not fully leveraged the sequential information embedded within dFC that can potentially provide valuable information when identifying brain conditions. In this paper, we propose a novel framework that jointly learns the embedding of both spatial and temporal information within dFC based on the transformer architecture. Specifically, we first construct dFC networks from rs-fMRI data through a sliding window strategy. Then, we simultaneously employ a temporal block and a spatial block to capture higher-order representations of dynamic spatio-temporal dependencies, via mapping them into an efficient fused feature representation. To further enhance the robustness of these feature representations by reducing the dependency on labeled data, we also introduce a contrastive learning strategy to manipulate different brain states. Experimental results on 345 subjects with 570 scans from the Alzheimer's Disease Neuroimaging Initiative (ADNI) demonstrate the superiority of our proposed method for MCI (Mild Cognitive Impairment, the prodromal stage of AD) prediction, highlighting its potential for early identification of AD. 

**Abstract (ZH)**: 基于静息状态功能性磁共振成像（rs-fMRI）的动态功能连接（dFC）是一种先进的技术，用于捕捉神经活动的动态变化，并在阿尔茨海默病（AD）等脑部疾病的研究中非常有用。然而，现有研究尚未充分利用dFC内部嵌入的序列信息，这些信息在识别脑部条件时可能提供有价值的见解。在本文中，我们提出了一种新的框架，该框架基于Transformer架构，联合学习dFC中空间和时间信息的嵌入。具体来说，我们首先通过滑动窗口策略从rs-fMRI数据中构建dFC网络。然后，我们同时使用时间块和空间块来捕获动态时空依赖性的高层次表征，通过将其映射到高效的融合特征表示中来实现。为了进一步通过减少对标记数据的依赖来增强这些特征表示的鲁棒性，我们还引入了一种对比学习策略来操控不同脑状态。我们在来自阿尔茨海默病神经影像学倡议（ADNI）的345名受试者和570次扫描的数据上进行的实验结果表明，我们提出的方法在轻度认知障碍（MCI，AD的前驱阶段）预测中的优越性，并突显了其在早期识别AD方面的潜力。 

---
# DynaPrompt: Dynamic Test-Time Prompt Tuning 

**Title (ZH)**: DynaPrompt：动态测试时提示调整 

**Authors**: Zehao Xiao, Shilin Yan, Jack Hong, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiayi Shen, Qi Wang, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2501.16404)  

**Abstract**: Test-time prompt tuning enhances zero-shot generalization of vision-language models but tends to ignore the relatedness among test samples during inference. Online test-time prompt tuning provides a simple way to leverage the information in previous test samples, albeit with the risk of prompt collapse due to error accumulation. To enhance test-time prompt tuning, we propose DynaPrompt, short for dynamic test-time prompt tuning, exploiting relevant data distribution information while reducing error accumulation. Built on an online prompt buffer, DynaPrompt adaptively selects and optimizes the relevant prompts for each test sample during tuning. Specifically, we introduce a dynamic prompt selection strategy based on two metrics: prediction entropy and probability difference. For unseen test data information, we develop dynamic prompt appending, which allows the buffer to append new prompts and delete the inactive ones. By doing so, the prompts are optimized to exploit beneficial information on specific test data, while alleviating error accumulation. Experiments on fourteen datasets demonstrate the effectiveness of dynamic test-time prompt tuning. 

**Abstract (ZH)**: 在测试时进行提示调整可以增强视觉-语言模型的零样本泛化能力，但在推理过程中倾向于忽略测试样本之间的相关性。在线测试时进行提示调整提供了一种简单的方法，可以利用先前测试样本中的信息，尽管这样做可能会因为错误累积而导致提示崩溃的风险。为提高测试时的提示调整效果，我们提出了DynaPrompt（动态测试时提示调整），利用相关数据分布信息同时减少错误累积。基于在线提示缓冲区，DynaPrompt 在调整过程中适应性地选择并优化每个测试样本的相关提示。具体来说，我们引入了一种基于预测熵和概率差两个指标的动态提示选择策略。针对未见过的测试数据信息，我们开发了动态提示追加功能，允许缓冲区追加新的提示并删除不活跃的提示。通过这种方式，提示被优化以充分利用特定测试数据上的有益信息，同时减轻错误累积。在十四个数据集上的实验表明了动态测试时提示调整的有效性。 

---
# Is Open Source the Future of AI? A Data-Driven Approach 

**Title (ZH)**: 《数据驱动视角下开源是否将是人工智能的未来？》 

**Authors**: Domen Vake, Bogdan Šinik, Jernej Vičič, Aleksandar Tošić  

**Link**: [PDF](https://arxiv.org/pdf/2501.16403)  

**Abstract**: Large Language Models (LLMs) have become central in academia and industry, raising concerns about privacy, transparency, and misuse. A key issue is the trustworthiness of proprietary models, with open-sourcing often proposed as a solution. However, open-sourcing presents challenges, including potential misuse, financial disincentives, and intellectual property concerns. Proprietary models, backed by private sector resources, are better positioned for return on investment.
There are also other approaches that lie somewhere on the spectrum between completely open-source and proprietary. These can largely be categorised into open-source usage limitations protected by licensing, partially open-source (open weights) models, hybrid approaches where obsolete model versions are open-sourced, while competitive versions with market value remain proprietary.
Currently, discussions on where on the spectrum future models should fall on remains unbacked and mostly opinionated where industry leaders are weighing in on the discussion. In this paper, we present a data-driven approach by compiling data on open-source development of LLMs, and their contributions in terms of improvements, modifications, and methods. Our goal is to avoid supporting either extreme but rather present data that will support future discussions both by industry experts as well as policy makers.
Our findings indicate that open-source contributions can enhance model performance, with trends such as reduced model size and manageable accuracy loss. We also identify positive community engagement patterns and architectures that benefit most from open contributions. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在学术界和工业界占据核心地位，引起了关于隐私、透明度和滥用的安全关注。一个关键问题是专有模型的信任度问题，开源常常被视为解决方案之一。然而，开源也带来了诸如潜在的滥用、财务激励不足和知识产权问题等挑战。依托私有部门资源，专有模型在投资回报方面更有优势。

此外，还存在一些介于完全开源和专有之间的其他方法。这些方法主要可以分为通过许可保护使用的开源限制、部分开源（公开权重）模型、以及混合方法，在这种方法中，过时的模型版本被开源，而具有市场价值的竞争版本则保持专有。

目前，未来模型应位于开源和专有的哪个区间，讨论仍然缺乏基于数据的支撑，主要停留在行业领导者的观点上。在本文中，我们通过收集和整理LLMs开源开发的数据，并对其改进、修改和方法等方面进行分析，提出了一种数据驱动的方法。我们的目标是在避免支持任何极端的同时，提供数据以支持行业专家和决策者未来讨论的内容。

我们的研究结果显示，开源贡献可以提升模型性能，呈现出模型体积减小和可管理的精度损失的趋势。此外，我们还识别出受益于开源贡献的积极社区参与模式和架构。 

---
# Leveraging Induced Transferable Binding Principles for Associative Prediction of Novel Drug-Target Interactions 

**Title (ZH)**: 利用诱导转换结合原理进行新颖药物-靶标相互作用的关联预测 

**Authors**: Xiaoqing Lian, Jie Zhu, Tianxu Lv, Shiyun Nie, Hang Fan, Guosheng Wu, Yunjun Ge, Lihua Li, Xiangxiang Zeng, Xiang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16391)  

**Abstract**: Significant differences in protein structures hinder the generalization of existing drug-target interaction (DTI) models, which often rely heavily on pre-learned binding principles or detailed annotations. In contrast, BioBridge designs an Inductive-Associative pipeline inspired by the workflow of scientists who base their accumulated expertise on drawing insights into novel drug-target pairs from weakly related references. BioBridge predicts novel drug-target interactions using limited sequence data, incorporating multi-level encoders with adversarial training to accumulate transferable binding principles. On these principles basis, BioBridge employs a dynamic prototype meta-learning framework to associate insights from weakly related annotations, enabling robust predictions for previously unseen drug-target pairs. Extensive experiments demonstrate that BioBridge surpasses existing models, especially for unseen proteins. Notably, when only homologous protein binding data is available, BioBridge proves effective for virtual screening of the epidermal growth factor receptor and adenosine receptor, underscoring its potential in drug discovery. 

**Abstract (ZH)**: 现有药物-目标相互作用（DTI）模型由于蛋白质结构间显著差异而难以泛化，这些模型往往依赖于预先学习的结合原则或详细的注释。相比之下，BioBridge 设计了一种归纳-关联管道，灵感来源于科学家的工作流程——他们通过从弱相关参考中抽取出新药物-目标配对的见解来积累专业知识。BioBridge 使用有限的序列数据预测新的药物-目标相互作用，并结合多级编码器和对抗训练，以积累可转移的结合原则。基于这些原则，BioBridge 采用动态原型元学习框架，将从弱相关注释中获得的见解关联起来，从而对未见过的药物-目标配对进行稳健预测。广泛实验表明，BioBridge 超越了现有模型，特别是在预测未见过的蛋白质方面表现尤为突出。值得注意的是，当仅可用同源蛋白质结合数据时，BioBridge 在表皮生长因子受体和腺苷受体的虚拟筛选中表现出有效性，突显了其在药物发现中的潜力。 

---
# RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations 

**Title (ZH)**: RotateKV：通过基于离群值的自适应旋转实现的LLMs高效且鲁棒的2位键值缓存量化 

**Authors**: Zunhai Su, Zhe Chen, Wang Shen, Hanyu Wei, Linge Li, Huangqi Yu, Kehong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16383)  

**Abstract**: Key-Value (KV) cache facilitates efficient large language models (LLMs) inference by avoiding recomputation of past KVs. As the batch size and context length increase, the oversized KV caches become a significant memory bottleneck, highlighting the need for efficient compression. Existing KV quantization rely on fine-grained quantization or the retention of a significant portion of high bit-widths caches, both of which compromise compression ratio and often fail to maintain robustness at extremely low average bit-widths. In this work, we explore the potential of rotation technique for 2-bit KV quantization and propose RotateKV, which achieves accurate and robust performance through the following innovations: (i) Outlier-Aware Rotation, which utilizes channel-reordering to adapt the rotations to varying channel-wise outlier distributions without sacrificing the computational efficiency of the fast Walsh-Hadamard transform (FWHT); (ii) Pre-RoPE Grouped-Head Rotation, which mitigates the impact of rotary position embedding (RoPE) on proposed outlier-aware rotation and further smooths outliers across heads; (iii) Attention-Sink-Aware Quantization, which leverages the massive activations to precisely identify and protect attention sinks. RotateKV achieves less than 0.3 perplexity (PPL) degradation with 2-bit quantization on WikiText-2 using LLaMA-2-13B, maintains strong CoT reasoning and long-context capabilities, with less than 1.7\% degradation on GSM8K, outperforming existing methods even at lower average bit-widths. RotateKV also showcases a 3.97x reduction in peak memory usage, supports 5.75x larger batch sizes, and achieves a 2.32x speedup in decoding stage. 

**Abstract (ZH)**: 键值（KV）缓存通过避免重新计算过去的键值对来促进大规模语言模型（LLMs）的高效推理。随着批次大小和上下文长度的增加，过大的KV缓存成为显著的内存瓶颈，突显了高效压缩的必要性。现有的KV量化依赖于细粒度的量化或者保留大量高位宽的缓存，这两种方法都牺牲了压缩比，并且在极低的平均位宽下难以保持鲁棒性。在本文中，我们探索了旋转技术在2位KV量化中的潜力，并提出了RotateKV，该方法通过以下创新实现了准确和稳健的性能：(i) 基于异常值感知的旋转，利用信道重排适应不同的信道-wise异常值分布，同时保持快速沃尔什-豪氏变换（FWHT）的计算效率；(ii) 预旋转位置编码分组头旋转，减轻旋转位置编码（RoPE）对提出的异常值感知旋转的影响，并进一步在不同头之间平滑异常值；(iii) 注意力沉降感知量化，利用大量的激活来精确地识别和保护注意力沉降点。在使用LLaMA-2-13B进行WikiText-2的2位量化时，RotateKV的困惑度（PPL）下降不到0.3，保持了强大的链式推理能力和长上下文能力，即使在GSM8K上也只有不到1.7%的下降，其性能甚至优于现有方法，尤其是在较低的平均位宽下。实验结果还显示，RotateKV的顶峰内存使用量减少了3.97倍，支持了5.75倍更大的批次大小，并在解码阶段实现了2.32倍的速度提升。 

---
# GraPPI: A Retrieve-Divide-Solve GraphRAG Framework for Large-scale Protein-protein Interaction Exploration 

**Title (ZH)**: GraPPI：一种用于大规模蛋白质相互作用探索的检索-划分-解决GraphRAG框架 

**Authors**: Ziwen Li, Xiang 'Anthony' Chen, Youngseung Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2501.16382)  

**Abstract**: Drug discovery (DD) has tremendously contributed to maintaining and improving public health. Hypothesizing that inhibiting protein misfolding can slow disease progression, researchers focus on target identification (Target ID) to find protein structures for drug binding. While Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) frameworks have accelerated drug discovery, integrating models into cohesive workflows remains challenging. We conducted a user study with drug discovery researchers to identify the applicability of LLMs and RAGs in Target ID. We identified two main findings: 1) an LLM should provide multiple Protein-Protein Interactions (PPIs) based on an initial protein and protein candidates that have a therapeutic impact; 2) the model must provide the PPI and relevant explanations for better understanding. Based on these observations, we identified three limitations in previous approaches for Target ID: 1) semantic ambiguity, 2) lack of explainability, and 3) short retrieval units. To address these issues, we propose GraPPI, a large-scale knowledge graph (KG)-based retrieve-divide-solve agent pipeline RAG framework to support large-scale PPI signaling pathway exploration in understanding therapeutic impacts by decomposing the analysis of entire PPI pathways into sub-tasks focused on the analysis of PPI edges. 

**Abstract (ZH)**: 药物发现（DD）极大地促进了公共健康维护和提升。研究人员假设抑制蛋白质错误折叠可以减缓疾病进展，因此专注于目标识别（Target ID），寻找可用于药物结合的蛋白质结构。虽然大型语言模型（LLMs）和检索增强生成（RAG）框架已经加速了药物发现过程，但将这些模型整合到统一的工作流中仍然颇具挑战。我们对药物发现研究人员进行了一项用户研究，以确定LLMs和RAGs在目标识别中的适用性。我们发现了两个主要发现：1）LLM 应该基于初始蛋白质及其候选药物分子提供多种蛋白质-蛋白质相互作用（PPI）的信息；2）模型必须提供PPI及其相关解释以便更好地理解。基于这些观察，我们在目标识别的先前方法中发现了三个局限性：1）语义模糊性，2）缺乏可解释性，3）短小的检索单元。为了解决这些问题，我们提出了一种大型知识图谱（KG）为基础的检索-分解-求解（retrieve-divide-solve）代理管道RAG框架——GraPPI。该框架旨在通过将整个PPI路径分析分解为专注于PPI边分析的子任务，支持大规模PPI信号通路探索，以更好地理解药物治疗效应。 

---
# UDiTQC: U-Net-Style Diffusion Transformer for Quantum Circuit Synthesis 

**Title (ZH)**: UDiTQC：基于U-Net风格扩散变换器的量子电路合成 

**Authors**: Zhiwei Chen, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16380)  

**Abstract**: Quantum computing is a transformative technology with wide-ranging applications, and efficient quantum circuit generation is crucial for unlocking its full potential. Current diffusion model approaches based on U-Net architectures, while promising, encounter challenges related to computational efficiency and modeling global context. To address these issues, we propose UDiT,a novel U-Net-style Diffusion Transformer architecture, which combines U-Net's strengths in multi-scale feature extraction with the Transformer's ability to model global context. We demonstrate the framework's effectiveness on two tasks: entanglement generation and unitary compilation, where UDiTQC consistently outperforms existing methods. Additionally, our framework supports tasks such as masking and editing circuits to meet specific physical property requirements. This dual advancement, improving quantum circuit synthesis and refining generative model architectures, marks a significant milestone in the convergence of quantum computing and machine learning research. 

**Abstract (ZH)**: 量子计算是一种具有广泛应用前景的变革性技术，而高效量子电路生成对于充分发挥其潜力至关重要。当前基于U-Net架构的扩散模型方法虽然充满潜力，但在计算效率和建模全局上下文方面遇到了挑战。为解决这些问题，我们提出了一种新型的U-Net风格扩散变换器（UDiT）架构，该架构结合了U-Net在多尺度特征提取方面的优势和变换器建模全局上下文的能力。我们在两个任务上展示了该框架的有效性：纠缠生成和酉矩阵编译，在这两个任务上，UDiT持续优于现有方法。此外，我们的框架还支持掩蔽和编辑电路以满足特定的物理属性要求。这一双重进步，在提高量子电路合成效率的同时，优化了生成模型的架构，标志着量子计算与机器学习研究交汇领域的重大里程碑。 

---
# FedAGHN: Personalized Federated Learning with Attentive Graph HyperNetworks 

**Title (ZH)**: FedAGHN：带有注意力图超网络的个性化 Federated Learning 

**Authors**: Jiarui Song, Yunheng Shen, Chengbin Hou, Pengyu Wang, Jinbao Wang, Ke Tang, Hairong Lv  

**Link**: [PDF](https://arxiv.org/pdf/2501.16379)  

**Abstract**: Personalized Federated Learning (PFL) aims to address the statistical heterogeneity of data across clients by learning the personalized model for each client. Among various PFL approaches, the personalized aggregation-based approach conducts parameter aggregation in the server-side aggregation phase to generate personalized models, and focuses on learning appropriate collaborative relationships among clients for aggregation. However, the collaborative relationships vary in different scenarios and even at different stages of the FL process. To this end, we propose Personalized Federated Learning with Attentive Graph HyperNetworks (FedAGHN), which employs Attentive Graph HyperNetworks (AGHNs) to dynamically capture fine-grained collaborative relationships and generate client-specific personalized initial models. Specifically, AGHNs empower graphs to explicitly model the client-specific collaborative relationships, construct collaboration graphs, and introduce tunable attentive mechanism to derive the collaboration weights, so that the personalized initial models can be obtained by aggregating parameters over the collaboration graphs. Extensive experiments can demonstrate the superiority of FedAGHN. Moreover, a series of visualizations are presented to explore the effectiveness of collaboration graphs learned by FedAGHN. 

**Abstract (ZH)**: 个性化联邦学习（PFL）旨在通过为每个客户端学习个性化模型来解决跨客户端数据的统计异质性问题。在各种PFL方法中，个性化聚合方法在服务器端聚合阶段进行参数聚合，从而生成个性化模型，并侧重于学习客户端之间的适当协同关系以进行聚合。然而，在不同的场景中，甚至在联邦学习（FL）过程的不同阶段，这些协同关系可能会有所变化。为解决这一问题，我们提出了基于注意力图超网络的个性化联邦学习（FedAGHN），利用注意力图超网络（AGHNs）动态捕捉细粒度的协同关系，并生成客户端特定的个性化初始模型。具体而言，AGHNs 赋能图来显式建模客户端特定的协同关系，构建协同图，并引入可调注意机制以推导协同权重，从而通过在协同图上聚合参数来获取个性化初始模型。大量的实验可以证明 FedAGHN 的优越性。此外，还呈现了一系列可视化结果，以探索 FedAGHN 学习到的协同图的有效性。 

---
# Internal Activation Revision: Safeguarding Vision Language Models Without Parameter Update 

**Title (ZH)**: 内部激活修正：在不更新参数的情况下保障视觉语言模型的安全性 

**Authors**: Qing Li, Jiahui Geng, Zongxiong Chen, Kun Song, Lei Ma, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2501.16378)  

**Abstract**: Vision-language models (VLMs) demonstrate strong multimodal capabilities but have been found to be more susceptible to generating harmful content compared to their backbone large language models (LLMs). Our investigation reveals that the integration of images significantly shifts the model's internal activations during the forward pass, diverging from those triggered by textual input. Moreover, the safety alignments of LLMs embedded within VLMs are not sufficiently robust to handle the activations discrepancies, making the models vulnerable to even the simplest jailbreaking attacks. To address this issue, we propose an \textbf{internal activation revision} approach that efficiently revises activations during generation, steering the model toward safer outputs. Our framework incorporates revisions at both the layer and head levels, offering control over the model's generation at varying levels of granularity. In addition, we explore three strategies for constructing positive and negative samples and two approaches for extracting revision vectors, resulting in different variants of our method. Comprehensive experiments demonstrate that the internal activation revision method significantly improves the safety of widely used VLMs, reducing attack success rates by an average of 48.94\%, 34.34\%, 43.92\%, and 52.98\% on SafeBench, Safe-Unsafe, Unsafe, and MM-SafetyBench, respectively, while minimally impacting model helpfulness. 

**Abstract (ZH)**: 视觉语言模型（Vision-Language Models, VLMs）展示了强大的多模态能力，但研究发现它们在生成有害内容方面比其基础的大语言模型（Large Language Models, LLMs）更易受到影响。我们的研究表明，图像的集成显著地改变了模型在前向传递过程中内部激活的状态，与仅由文本输入触发的激活状态产生了偏差。此外，VLMs 中嵌入的LLMs 的安全性对齐不够稳固，不足以处理这种激活差异，使模型在面对简单的 Jailbreaking 攻击时变得脆弱。为了应对这一问题，我们提出了一种**内部激活修订**（Internal Activation Revision）方法，该方法在生成过程中高效地修订激活状态，引导模型生成更安全的输出。我们的框架在层和头两个级别上都进行了修订，并提供了不同程度的模型生成控制。此外，我们探讨了三种构建正样本和负样本的策略以及两种提取修订向量的方法，从而形成了我们方法的不同变体。全面的实验表明，内部激活修订方法显著提高了广泛使用的 VLMs 的安全性，在 SafeBench、Safe-Unsafe、Unsafe 和 MM-SafetyBench 上分别将攻击成功率减少了 48.94%、34.34%、43.92% 和 52.98%，同时对模型的帮助性影响微乎其微。 

---
# Optimal Signal Decomposition-based Multi-Stage Learning for Battery Health Estimation 

**Title (ZH)**: 基于最优信号分解的多阶段学习方法用于电池健康状态估计 

**Authors**: Vijay Babu Pamshetti, Wei Zhang, King Jet Tseng, Bor Kiat Ng, Qingyu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16377)  

**Abstract**: Battery health estimation is fundamental to ensure battery safety and reduce cost. However, achieving accurate estimation has been challenging due to the batteries' complex nonlinear aging patterns and capacity regeneration phenomena. In this paper, we propose OSL, an optimal signal decomposition-based multi-stage machine learning for battery health estimation. OSL treats battery signals optimally. It uses optimized variational mode decomposition to extract decomposed signals capturing different frequency bands of the original battery signals. It also incorporates a multi-stage learning process to analyze both spatial and temporal battery features effectively. An experimental study is conducted with a public battery aging dataset. OSL demonstrates exceptional performance with a mean error of just 0.26%. It significantly outperforms comparison algorithms, both those without and those with suboptimal signal decomposition and analysis. OSL considers practical battery challenges and can be integrated into real-world battery management systems, offering a good impact on battery monitoring and optimization. 

**Abstract (ZH)**: 电池健康评估是确保电池安全和降低成本的基础。然而，由于电池复杂的非线性老化模式和容量再生现象，实现精确评估一直具有挑战性。本文提出了一种基于最优信号分解的多阶段机器学习方法OSL（Optimal Signal Decomposition-based Multi-stage Learning）进行电池健康评估。OSL通过优化信号分解来处理电池信号。它利用优化的变分模态分解提取分解信号，以捕获原始电池信号的不同频率区间。同时，它结合了多阶段学习过程，有效分析电池的时空特征。通过一项使用公开电池老化数据集的实验研究，OSL展现了出色的表现，平均误差仅为0.26%。它显著优于各种比较算法，包括那些未采用最优信号分解和分析的方法。OSL考虑了实际电池面临的挑战，并且可以集成到现实世界的电池管理系统中，对电池监测和优化具有很好的影响。 

---
# HWPQ: Hessian-free Weight Pruning-Quantization For LLM Compression And Acceleration 

**Title (ZH)**: HWPQ：无梯度二阶矩量权值修剪-量化方法在大语言模型压缩与加速中的应用 

**Authors**: Yuhan Kang, Zhongdi Luo, Mei Wen, Yang Shi, Jun He, Jianchao Yang, Zeyu Xue, Jing Feng, Xinwang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16376)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across numerous domains. However, the high time complexity of existing pruning and quantization methods significantly hinders their effective deployment on resource-constrained consumer or edge devices. In this study, we propose a novel Hessian-free Weight Pruning-Quantization (HWPQ) method. HWPQ eliminates the need for computationally intensive Hessian matrix calculations by introducing a contribution-based weight metric, which evaluates the importance of weights without relying on second-order derivatives. Additionally, we employ the Exponentially Weighted Moving Average (EWMA) technique to bypass weight sorting, enabling the selection of weights that contribute most to LLM accuracy and further reducing time complexity. Our approach is extended to support 2:4 structured sparsity pruning, facilitating efficient execution on modern hardware accelerators. Experimental results demonstrate that HWPQ significantly enhances the compression performance of LLaMA2. Compared to state-of-the-art quantization and pruning frameworks, HWPQ achieves average speedups of 5.97x (up to 20.75x) in quantization time and 12.29x (up to 56.02x) in pruning time, while largely preserving model accuracy. Furthermore, we observe a 1.50x inference speedup compared to the baseline. 

**Abstract (ZH)**: 大型语言模型（LLMs）在众多领域取得了显著的成功。然而，现有剪枝和量化方法的时间复杂度过高，显著阻碍了其在资源受限的消费者或边缘设备上的有效部署。在本研究中，我们提出了一种新颖的无海森矩阵权值剪枝-量化（HWPQ）方法。HWPQ通过引入基于贡献的权重度量方法，无需计算海森矩阵，从而评估权重的重要性。此外，我们采用了指数加权移动平均（EWMA）技术绕过了权重排序，使得可以选出对LLM准确度贡献最大的权重，并进一步降低了时间复杂度。该方法还扩展支持2:4结构稀疏剪枝，使得在现代硬件加速器上执行更加高效。实验结果表明，HWPQ显著提升了LLaMA2的压缩性能。与最先进的量化和剪枝框架相比，HWPQ在量化时间上的平均加速比为5.97倍（最多可达20.75倍），在剪枝时间上的加速比为12.29倍（最多可达56.02倍），同时模型准确度得到了很大程度的保留。此外，我们还观察到与基线相比，推理速度提升了1.50倍。 

---
# On Storage Neural Network Augmented Approximate Nearest Neighbor Search 

**Title (ZH)**: 存储神经网络增强的 approximate nearest neighbor 搜索 

**Authors**: Taiga Ikeda, Daisuke Miyashita, Jun Deguchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.16375)  

**Abstract**: Large-scale approximate nearest neighbor search (ANN) has been gaining attention along with the latest machine learning researches employing ANNs. If the data is too large to fit in memory, it is necessary to search for the most similar vectors to a given query vector from the data stored in storage devices, not from that in memory. The storage device such as NAND flash memory has larger capacity than the memory device such as DRAM, but they also have larger latency to read data. Therefore, ANN methods for storage require completely different approaches from conventional in-memory ANN methods. Since the approximation that the time required for search is determined only by the amount of data fetched from storage holds under reasonable assumptions, our goal is to minimize it while maximizing recall. For partitioning-based ANNs, vectors are partitioned into clusters in the index building phase. In the search phase, some of the clusters are chosen, the vectors in the chosen clusters are fetched from storage, and the nearest vector is retrieved from the fetched vectors. Thus, the key point is to accurately select the clusters containing the ground truth nearest neighbor vectors. We accomplish this by proposing a method to predict the correct clusters by means of a neural network that is gradually refined by alternating supervised learning and duplicated cluster assignment. Compared to state-of-the-art SPANN and an exhaustive method using k-means clustering and linear search, the proposed method achieves 90% recall on SIFT1M with 80% and 58% less data fetched from storage, respectively. 

**Abstract (ZH)**: 大规模近似最近邻搜索（Approximate Nearest Neighbor Search, ANN）随着最新基于ANN的机器学习研究而日益受到关注。如果数据量过大无法加载到内存中，就需要从存储设备中而非内存中检索与查询向量最相似的向量。例如，像NAND闪存这样的存储设备虽然比像DRAM这样的内存设备具有更大的容量，但读取数据时也会有更大的延迟。因此，存储设备的ANN方法需要与传统的内存ANN方法完全不同。基于分块的ANN，向量在索引构建阶段被分割成簇。在搜索阶段，选择一些簇，从存储设备中检索这些簇中的向量，并从中检索最近邻向量。因此，关键在于准确地选择包含真实最近邻向量的簇。我们通过提出一种使用逐步改进的神经网络进行监督学习和重复簇分配交替进行的方法来预测正确的簇。与最先进的SPANN方法以及使用k均值聚类和线性搜索的穷尽方法相比，在SIFT1M数据集上，所提出的方法分别只需检索80%和58%的数据即可达到90%的召回率。 

---
# SAFR: Neuron Redistribution for Interpretability 

**Title (ZH)**: SAFR：神经元重分布以提高可解释性 

**Authors**: Ruidi Chang, Chunyuan Deng, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.16374)  

**Abstract**: Superposition refers to encoding representations of multiple features within a single neuron, which is common in transformers. This property allows neurons to combine and represent multiple features, enabling the model to capture intricate information and handle complex tasks. Despite promising performance, the model's interpretability has been diminished. This paper presents a novel approach to enhance transformer interpretability by regularizing feature superposition. We introduce SAFR, which simply applies regularizations to the loss function to promote monosemantic representations for important tokens while encouraging polysemanticity for correlated token pairs, where important tokens and correlated token pairs are identified via VMASK and attention weights. With a transformer model on two classification tasks, SAFR improves interpretability without compromising prediction performance. Given an input to the model, SAFR provides an explanation by visualizing the neuron allocation and interaction within the MLP layers. 

**Abstract (ZH)**: 叠加是指在一个神经元中编码多个特征的表示，这是变压器中常见的特性。这一特性使得神经元能够结合和表示多个特征，从而使模型能够捕捉复杂的细节信息并处理复杂任务。尽管模型表现出色，但其可解释性已有所减弱。本文提出了一种新的方法，通过正则化特征叠加来增强变压器的可解释性。我们引入了SAFR（Semantically Aware Feature Regularization），它通过在损失函数中应用正则化手段，促进重要标记的单义表示，同时鼓励相关标记对的多义性表示。重要标记和相关标记对通过VMASK和注意力权重被识别。在两个分类任务上使用变压器模型时，SAFR能够在不牺牲预测性能的情况下提高模型的可解释性。给定输入，SAFR通过可视化MLP层内的神经元分配和交互来提供解释。 

---
# Unveiling Discrete Clues: Superior Healthcare Predictions for Rare Diseases 

**Title (ZH)**: 揭开离散线索的面纱：卓越的罕见疾病预测 

**Authors**: Chuang Zhao, Hui Tang, Jiheng Zhang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.16373)  

**Abstract**: Accurate healthcare prediction is essential for improving patient outcomes. Existing work primarily leverages advanced frameworks like attention or graph networks to capture the intricate collaborative (CO) signals in electronic health records. However, prediction for rare diseases remains challenging due to limited co-occurrence and inadequately tailored approaches. To address this issue, this paper proposes UDC, a novel method that unveils discrete clues to bridge consistent textual knowledge and CO signals within a unified semantic space, thereby enriching the representation semantics of rare diseases. Specifically, we focus on addressing two key sub-problems: (1) acquiring distinguishable discrete encodings for precise disease representation and (2) achieving semantic alignment between textual knowledge and the CO signals at the code level. For the first sub-problem, we refine the standard vector quantized process to include condition awareness. Additionally, we develop an advanced contrastive approach in the decoding stage, leveraging synthetic and mixed-domain targets as hard negatives to enrich the perceptibility of the reconstructed representation for downstream tasks. For the second sub-problem, we introduce a novel codebook update strategy using co-teacher distillation. This approach facilitates bidirectional supervision between textual knowledge and CO signals, thereby aligning semantically equivalent information in a shared discrete latent space. Extensive experiments on three datasets demonstrate our superiority. 

**Abstract (ZH)**: 准确的医疗预测对于改善患者预后至关重要。现有工作主要利用先进的框架，如注意力机制或图网络，来捕捉电子健康记录中的复杂协作（CO）信号。然而，对于罕见疾病的预测仍然颇具挑战性，原因在于共现数据有限，且现有的方法不够针对性。为解决这一问题，本文提出了一种名为UDC的新方法，该方法揭示了离散线索，以统一语义空间内的连续文本知识和CO信号之间的联系，从而丰富了罕见疾病的表示语义。具体而言，我们重点关注解决以下两个关键子问题：(1) 获取区分度高的离散编码以精确地表示疾病；(2) 在编码层面实现文本知识与CO信号的语义对齐。对于第一个子问题，我们对标准的向量量化过程进行了改进，使其包含条件感知。此外，在解码阶段，我们开发了一种先进的对比方法，利用合成目标和多领域目标作为难以区分的负样本，增强了重建表示在下游任务中的可感知性。对于第二个子问题，我们提出了一种新颖的码书更新策略，利用协同教师蒸馏方法。该方法实现了文本知识与CO信号之间的双向监督，从而在共享的离散潜在空间中对齐语义等价信息。在三个数据集上的广泛实验表明了我们方法的优越性。 

---
# Low-Rank Adapters Meet Neural Architecture Search for LLM Compression 

**Title (ZH)**: 低秩适配器结合神经架构搜索实现大规模语言模型压缩 

**Authors**: J. Pablo Muñoz, Jinjie Yuan, Nilesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2501.16372)  

**Abstract**: The rapid expansion of Large Language Models (LLMs) has posed significant challenges regarding the computational resources required for fine-tuning and deployment. Recent advancements in low-rank adapters have demonstrated their efficacy in parameter-efficient fine-tuning (PEFT) of these models. This retrospective paper comprehensively discusses innovative approaches that synergize low-rank representations with Neural Architecture Search (NAS) techniques, particularly weight-sharing super-networks. Robust solutions for compressing and fine-tuning large pre-trained models are developed by integrating these methodologies. Our analysis highlights the potential of these combined strategies to democratize the use of LLMs, making them more accessible for deployment in resource-constrained environments. The resulting models exhibit reduced memory footprints and faster inference times, paving the way for more practical and scalable applications of LLMs. Models and code are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速扩展为细调和部署所需的计算资源带来了重大挑战。近期，低秩适配器的进步证明了它们在参数高效细调（PEFT）方面的重要性。本文综述了将低秩表示与神经架构搜索（NAS）技术相结合的创新方法，特别是权重共享超级网络。通过整合这些方法，我们开发了稳健的用于压缩和细调大型预训练模型的解决方案。我们的分析强调了这些综合策略的潜力，使其能够让更多在资源受限环境中部署LLMs成为可能。这些模型具有较小的内存需求和更快的推理时间，为LLMs的实际应用和扩展性开辟了新途径。完整的模型和代码可在以下链接获取：[此处插入链接]。 

---
# Which Optimizer Works Best for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks? 

**Title (ZH)**: 哪种优化器最适合物理知情神经网络和科莫洛夫-阿诺尔德网络？ 

**Authors**: Elham Kiyani, Khemraj Shukla, Jorge F. Urbán, Jérôme Darbon, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.16371)  

**Abstract**: Physics-Informed Neural Networks (PINNs) have revolutionized the computation of PDE solutions by integrating partial differential equations (PDEs) into the neural network's training process as soft constraints, becoming an important component of the scientific machine learning (SciML) ecosystem. In its current implementation, PINNs are mainly optimized using first-order methods like Adam, as well as quasi-Newton methods such as BFGS and its low-memory variant, L-BFGS. However, these optimizers often struggle with highly non-linear and non-convex loss landscapes, leading to challenges such as slow convergence, local minima entrapment, and (non)degenerate saddle points. In this study, we investigate the performance of Self-Scaled Broyden (SSBroyden) methods and other advanced quasi-Newton schemes, including BFGS and L-BFGS with different line search strategies approaches. These methods dynamically rescale updates based on historical gradient information, thus enhancing training efficiency and accuracy. We systematically compare these optimizers on key challenging linear, stiff, multi-scale and non-linear PDEs benchmarks, including the Burgers, Allen-Cahn, Kuramoto-Sivashinsky, and Ginzburg-Landau equations, and extend our study to Physics-Informed Kolmogorov-Arnold Networks (PIKANs) representation. Our findings provide insights into the effectiveness of second-order optimization strategies in improving the convergence and accurate generalization of PINNs for complex PDEs by orders of magnitude compared to the state-of-the-art. 

**Abstract (ZH)**: 物理知情神经网络（Physics-Informed Neural Networks, PINNs）通过将部分微分方程（Partial Differential Equations, PDEs）整合到神经网络的训练过程中，作为软约束，革命性地改变了PDE解的计算方法，成为科学机器学习（Scientific Machine Learning, SciML）生态系统中的重要组成部分。在当前实现中，PINNs主要使用诸如Adam等一阶优化方法，以及BFGS和其低内存变种L-BFGS等拟牛顿方法进行优化。然而，这些优化器在高度非线性和非凸损失场景下常常遇到困难，导致收敛速度慢、容易陷入局部极小值、以及鞍点问题。在本研究中，我们探讨了Self-Scaled Broyden（SSBroyden）方法和其他高级拟牛顿方案的性能，包括BFGS及其与其他线搜索策略结合的不同L-BFGS版本。这些方法基于历史梯度信息动态缩放更新，从而提高训练效率和准确性。我们系统地在关键的线性、刚性、多尺度和非线性PDE基准测试中比较了这些优化器，包括Burgers、Allen-Cahn、Kuramoto-Sivashinsky和Ginzburg-Landau方程，并将研究扩展到物理知情柯尔莫果洛夫-阿诺尔德网络（Physics-Informed Kolmogorov-Arnold Networks, PIKANs）表示。我们的研究结果表明，与最先进的方法相比，第二阶优化策略极大地提高了PINNs对复杂PDEs的收敛速度和准确泛化能力。 

---
# Advanced Physics-Informed Neural Network with Residuals for Solving Complex Integral Equations 

**Title (ZH)**: 带有残差的先进物理知情神经网络用于求解复杂积分方程 

**Authors**: Mahdi Movahedian Moghaddam, Kourosh Parand, Saeed Reza Kheradpisheh  

**Link**: [PDF](https://arxiv.org/pdf/2501.16370)  

**Abstract**: In this paper, we present the Residual Integral Solver Network (RISN), a novel neural network architecture designed to solve a wide range of integral and integro-differential equations, including one-dimensional, multi-dimensional, ordinary and partial integro-differential, systems, and fractional types. RISN integrates residual connections with high-accurate numerical methods such as Gaussian quadrature and fractional derivative operational matrices, enabling it to achieve higher accuracy and stability than traditional Physics-Informed Neural Networks (PINN). The residual connections help mitigate vanishing gradient issues, allowing RISN to handle deeper networks and more complex kernels, particularly in multi-dimensional problems. Through extensive experiments, we demonstrate that RISN consistently outperforms PINN, achieving significantly lower Mean Absolute Errors (MAE) across various types of equations. The results highlight RISN's robustness and efficiency in solving challenging integral and integro-differential problems, making it a valuable tool for real-world applications where traditional methods often struggle. 

**Abstract (ZH)**: 在本文中，我们提出了一种名为残差积分求解器网络（Residual Integral Solver Network, RISN）的新型神经网络架构，该架构旨在解决广泛范围的积分和积分微分方程，包括一维、多维、常微分和偏微分积分微分方程系统以及分数阶类型。RISN 将残差连接与高精度数值方法（如高斯求积和分数阶导数运算矩阵）相结合，使其能够实现比传统物理信息神经网络（Physics-Informed Neural Network, PINN）更高的准确性和稳定性。残差连接有助于缓解梯度消失问题，从而使 RISN 能够处理更深的网络结构和更复杂的核函数，特别是在多维问题中。通过广泛的实验，我们证明 RISN 在各种类型的方程中始终优于 PINN，实现了显著更低的平均绝对误差（Mean Absolute Error, MAE）。结果表明，RISN 在解决具有挑战性的积分和积分微分问题方面具有稳健性和效率，使其成为传统方法在实际应用中常常难以应对的重要工具。 

---
# Blockchain-based Crowdsourced Deep Reinforcement Learning as a Service 

**Title (ZH)**: 基于区块链的众源深度强化学习即服务 

**Authors**: Ahmed Alagha, Hadi Otrok, Shakti Singh, Rabeb Mizouni, Jamal Bentahar  

**Link**: [PDF](https://arxiv.org/pdf/2501.16369)  

**Abstract**: Deep Reinforcement Learning (DRL) has emerged as a powerful paradigm for solving complex problems. However, its full potential remains inaccessible to a broader audience due to its complexity, which requires expertise in training and designing DRL solutions, high computational capabilities, and sometimes access to pre-trained models. This necessitates the need for hassle-free services that increase the availability of DRL solutions to a variety of users. To enhance the accessibility to DRL services, this paper proposes a novel blockchain-based crowdsourced DRL as a Service (DRLaaS) framework. The framework provides DRL-related services to users, covering two types of tasks: DRL training and model sharing. Through crowdsourcing, users could benefit from the expertise and computational capabilities of workers to train DRL solutions. Model sharing could help users gain access to pre-trained models, shared by workers in return for incentives, which can help train new DRL solutions using methods in knowledge transfer. The DRLaaS framework is built on top of a Consortium Blockchain to enable traceable and autonomous execution. Smart Contracts are designed to manage worker and model allocation, which are stored using the InterPlanetary File System (IPFS) to ensure tamper-proof data distribution. The framework is tested on several DRL applications, proving its efficacy. 

**Abstract (ZH)**: 深度 reinforcement learning（DRL）已成为解决复杂问题的强大框架。但由于其复杂性，其全部潜力尚未被广泛的用户群体所充分利用。这需要专门的训练和设计DRL解决方案的知识，高度的计算能力，有时还需要访问预训练模型。因此，迫切需要一种便捷的服务，以提高DRL解决方案的可用性。为了增强对DRL服务的可访问性，本文提出了一种新颖的基于区块链的众包DRL即服务（DRLaaS）框架。该框架为用户提供与DRL相关的服务，涵盖了两种类型的任务：DRL训练和模型共享。通过众包，用户可以从工作者的专业知识和计算能力中获益，以训练DRL解决方案。模型共享可以帮助用户访问工作者分享的预训练模型，这些模型以激励的形式提供，从而通过知识转移方法帮助训练新的DRL解决方案。DRLaaS框架基于联盟链构建，以实现可追溯和自主执行。智能合约被设计用于管理和分配工作者和模型，并通过星际文件系统（IPFS）存储，以确保数据的防篡改分布。该框架已在多个DRL应用中进行了测试，证明了其有效性。 

---
# Foundation Models for CPS-IoT: Opportunities and Challenges 

**Title (ZH)**: 面向CPS-IoT的基石模型：机遇与挑战 

**Authors**: Ozan Baris, Yizhuo Chen, Gaofeng Dong, Liying Han, Tomoyoshi Kimura, Pengrui Quan, Ruijie Wang, Tianchen Wang, Tarek Abdelzaher, Mario Bergés, Paul Pu Liang, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2501.16368)  

**Abstract**: Methods from machine learning (ML) have transformed the implementation of Perception-Cognition-Communication-Action loops in Cyber-Physical Systems (CPS) and the Internet of Things (IoT), replacing mechanistic and basic statistical models with those derived from data. However, the first generation of ML approaches, which depend on supervised learning with annotated data to create task-specific models, faces significant limitations in scaling to the diverse sensor modalities, deployment configurations, application tasks, and operating dynamics characterizing real-world CPS-IoT systems. The success of task-agnostic foundation models (FMs), including multimodal large language models (LLMs), in addressing similar challenges across natural language, computer vision, and human speech has generated considerable enthusiasm for and exploration of FMs and LLMs as flexible building blocks in CPS-IoT analytics pipelines, promising to reduce the need for costly task-specific engineering.
Nonetheless, a significant gap persists between the current capabilities of FMs and LLMs in the CPS-IoT domain and the requirements they must meet to be viable for CPS-IoT applications. In this paper, we analyze and characterize this gap through a thorough examination of the state of the art and our research, which extends beyond it in various dimensions. Based on the results of our analysis and research, we identify essential desiderata that CPS-IoT domain-specific FMs and LLMs must satisfy to bridge this gap. We also propose actions by CPS-IoT researchers to collaborate in developing key community resources necessary for establishing FMs and LLMs as foundational tools for the next generation of CPS-IoT systems. 

**Abstract (ZH)**: 机器学习（ML）方法已经改变了在网络物理系统（CPS）和物联网（IoT）中实施感知-认知-通信-行动循环的方式，用从数据中导出的模型取代了机械性和基础统计模型。然而，依赖于带有标注数据的监督学习的第一代ML方法在扩展到实际CPS-IoT系统中多样化的传感器模态、部署配置、应用任务和运行动态方面面临重大局限性。无任务特定的基础模型（FMs），包括多模态大规模语言模型（LLMs），在解决自然语言、计算机视觉和人类语言领域相似挑战方面取得了显著成功，这激发了对FMs和LLMs作为CPS-IoT分析管道的灵活构建模块的探索，有望减少针对特定任务的工程成本。

尽管如此，当前FMs和LLMs在CPS-IoT领域的功能与它们需要满足以适用于CPS-IoT应用的要求之间仍存在显著差距。在这篇论文中，我们通过全面分析现有技术和我们的研究成果来分析并表征这种差距，并在多个维度上超越了现有的研究。基于分析和研究结果，我们确定了CPS-IoT领域特定的FMs和LLMs必须满足的关键需求，以缩小这一差距。我们还提议CPS-IoT领域的研究人员合作开发关键社区资源，以建立FMs和LLMs作为新一代CPS-IoT系统的基础工具。 

---
# CAND: Cross-Domain Ambiguity Inference for Early Detecting Nuanced Illness Deterioration 

**Title (ZH)**: CAND：跨领域模糊性推理在早期检测微妙疾病恶化中的应用 

**Authors**: Lo Pang-Yun Ting, Zhen Tan, Hong-Pei Chen, Cheng-Te Li, Po-Lin Chen, Kun-Ta Chuang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16365)  

**Abstract**: Early detection of patient deterioration is essential for timely treatment, with vital signs like heart rates being key health indicators. Existing methods tend to solely analyze vital sign waveforms, ignoring transition relationships of waveforms within each vital sign and the correlation strengths among various vital signs. Such studies often overlook nuanced illness deterioration, which is the early sign of worsening health but is difficult to detect. In this paper, we introduce CAND, a novel method that organizes the transition relationships and the correlations within and among vital signs as domain-specific and cross-domain knowledge. CAND jointly models these knowledge in a unified representation space, considerably enhancing the early detection of nuanced illness deterioration. In addition, CAND integrates a Bayesian inference method that utilizes augmented knowledge from domain-specific and cross-domain knowledge to address the ambiguities in correlation strengths. With this architecture, the correlation strengths can be effectively inferred to guide joint modeling and enhance representations of vital signs. This allows a more holistic and accurate interpretation of patient health. Our experiments on a real-world ICU dataset demonstrate that CAND significantly outperforms existing methods in both effectiveness and earliness in detecting nuanced illness deterioration. Moreover, we conduct a case study for the interpretable detection process to showcase the practicality of CAND. 

**Abstract (ZH)**: 早期识别患者恶化是及时治疗的关键，心率等生命体征是重要健康指标。现有方法通常仅分析生命体征波形，忽视了每个生命体征内部波形转换关系以及各种生命体征之间的相关性。这类研究往往忽略了细微病情恶化的迹象，这是健康状况恶化早期的信号但很难被检测到。在本文中，我们提出了一种名为CAND的新方法，该方法将生命体征内部和之间的转换关系以及跨域知识组织为特定领域的和跨领域的知识。CAND通过统一的表示空间联合建模这些知识，大幅提升了对细微病情恶化的早期检测能力。此外，CAND整合了一种贝叶斯推理方法，利用特定领域和跨领域的增强知识来解决相关性强度中的不确定性。通过这种架构，相关性强度可以被有效推断以指导联合建模并增强生命体征的表示，从而使对患者健康状况的解释更加全面和准确。在实际ICU数据集上的实验结果表明，CAND在检测细微病情恶化方面在有效性上显著优于现有方法，在早期检测方面也表现出色。此外，我们还进行了一项可解释检测过程的案例研究，以展示CAND的实际应用价值。 

---
# Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained Intra- and Inter-Variate Dependencies 

**Title (ZH)**: 通过捕捉粗粒度的同变分间变关系进行多变量时间序列异常检测 

**Authors**: Yongzheng Xie, Hongyu Zhang, Muhammad Ali Babar  

**Link**: [PDF](https://arxiv.org/pdf/2501.16364)  

**Abstract**: Multivariate time series anomaly detection is essential for failure management in web application operations, as it directly influences the effectiveness and timeliness of implementing remedial or preventive measures. This task is often framed as a semi-supervised learning problem, where only normal data are available for model training, primarily due to the labor-intensive nature of data labeling and the scarcity of anomalous data. Existing semi-supervised methods often detect anomalies by capturing intra-variate temporal dependencies and/or inter-variate relationships to learn normal patterns, flagging timestamps that deviate from these patterns as anomalies. However, these approaches often fail to capture salient intra-variate temporal and inter-variate dependencies in time series due to their focus on excessively fine granularity, leading to suboptimal performance. In this study, we introduce MtsCID, a novel semi-supervised multivariate time series anomaly detection method. MtsCID employs a dual network architecture: one network operates on the attention maps of multi-scale intra-variate patches for coarse-grained temporal dependency learning, while the other works on variates to capture coarse-grained inter-variate relationships through convolution and interaction with sinusoidal prototypes. This design enhances the ability to capture the patterns from both intra-variate temporal dependencies and inter-variate relationships, resulting in improved performance. Extensive experiments across seven widely used datasets demonstrate that MtsCID achieves performance comparable or superior to state-of-the-art benchmark methods. 

**Abstract (ZH)**: 多变量时间序列异常检测对于Web应用操作中的故障管理至关重要，因为它直接影响到采取补救或预防措施的有效性和及时性。这一任务通常被描述为半监督学习问题，因为大量正常数据可用进行模型训练，而异常数据由于标注工作量巨大且稀缺而相对较少。现有的半监督方法通常通过捕捉单变量的时间依赖性和/或变量之间的关系来学习正常模式，并标记与这些模式偏差的时刻为异常。然而，这些方法往往由于过度关注细粒度而未能捕捉到关键的时间序列内部依赖性和变量间关系，导致性能不佳。在本研究中，我们提出了一种新型的半监督多变量时间序列异常检测方法——MtsCID。MtsCID采用了一种双重网络架构：一个网络在多尺度单变量片段的注意力图上操作，以学习粗粒度的时间依赖性；另一个网络则在变量层面通过卷积和与正弦原型的交互来捕捉粗粒度的变量间关系。这种设计增强了捕捉内部时间依赖性和变量间关系模式的能力，从而提高了性能。在七个广泛使用的数据集上的广泛实验表明，MtsCID在性能上与现有的最先进的基准方法相当或更优。 

---
# Large Language Models Meet Graph Neural Networks for Text-Numeric Graph Reasoning 

**Title (ZH)**: 大规模语言模型与图神经网络相结合进行文本-数值图推理 

**Authors**: Haoran Song, Jiarui Feng, Guangfu Li, Michael Province, Philip Payne, Yixin Chen, Fuhai Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.16361)  

**Abstract**: In real-world scientific discovery, human beings always make use of the accumulated prior knowledge with imagination pick select one or a few most promising hypotheses from large and noisy data analysis results. In this study, we introduce a new type of graph structure, the text-numeric graph (TNG), which is defined as graph entities and associations have both text-attributed information and numeric information. The TNG is an ideal data structure model for novel scientific discovery via graph reasoning because it integrates human-understandable textual annotations or prior knowledge, with numeric values that represent the observed or activation levels of graph entities or associations in different samples. Together both the textual information and numeric values determine the importance of graph entities and associations in graph reasoning for novel scientific knowledge discovery. We further propose integrating large language models (LLMs) and graph neural networks (GNNs) to analyze the TNGs for graph understanding and reasoning. To demonstrate the utility, we generated the text-omic(numeric) signaling graphs (TOSG), as one type of TNGs, in which all graphs have the same entities, associations and annotations, but have sample-specific entity numeric (omic) values using single cell RNAseq (scRNAseq) datasets of different diseases. We proposed joint LLM-GNN models for key entity mining and signaling pathway mining on the TOSGs. The evaluation results showed the LLM-GNN and TNGs models significantly improve classification accuracy and network inference. In conclusion, the TNGs and joint LLM-GNN models are important approaches for scientific discovery. 

**Abstract (ZH)**: 在实际的科学研究中，人类往往会利用积累的先验知识并结合想象力，从大量且杂乱的数据分析结果中选择一两个最有潜力的假设。本研究中，我们引入了一种新的图形结构——文本-数值图（TNG），定义为该图中的实体和关联同时具有文本属性和数值属性。TNG 是通过图形推理进行新颖科学发现的理想数据结构模型，因为它将人类可理解的文本注释或先验知识与表示不同样本中图形实体或关联观察水平或激活水平的数值信息结合起来。文本信息和数值信息共同决定了图形推理中图形实体和关联的重要程度，以发现新颖的科学知识。我们进一步提出将大型语言模型（LLMs）和图神经网络（GNNs）结合用于分析TNGs，以实现图形理解和推理。为了展示其实用性，我们生成了文本-组学（数值）信号图（TOSG），这是一种TNG 的类型，其中所有图具有相同实体、关联和注释，但具有特定样本的实体数值（组学）值，这些值是使用不同疾病的单细胞RNA 测序（scRNAseq）数据集获得的。我们在TOSG 上提出了联合LLM-GNN 模型，用于关键实体挖掘和信号通路挖掘。评估结果表明，LLM-GNN 和TNG 模型显著提高了分类准确性和网络推理。总的来说，TNGs 和联合LLM-GNN 模型是科学发现的重要方法。 

---
# Momentum Contrastive Learning with Enhanced Negative Sampling and Hard Negative Filtering 

**Title (ZH)**: 增强负样本采样并结合硬负样本过滤的动量对比学习 

**Authors**: Duy Hoang, Huy Ngo, Khoi Pham, Tri Nguyen, Gia Bao, Huy Phan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16360)  

**Abstract**: Contrastive learning has become pivotal in unsupervised representation learning, with frameworks like Momentum Contrast (MoCo) effectively utilizing large negative sample sets to extract discriminative features. However, traditional approaches often overlook the full potential of key embeddings and are susceptible to performance degradation from noisy negative samples in the memory bank. This study addresses these challenges by proposing an enhanced contrastive learning framework that incorporates two key innovations. First, we introduce a dual-view loss function, which ensures balanced optimization of both query and key embeddings, improving representation quality. Second, we develop a selective negative sampling strategy that emphasizes the most challenging negatives based on cosine similarity, mitigating the impact of noise and enhancing feature discrimination. Extensive experiments demonstrate that our framework achieves superior performance on downstream tasks, delivering robust and well-structured representations. These results highlight the potential of optimized contrastive mechanisms to advance unsupervised learning and extend its applicability across domains such as computer vision and natural language processing 

**Abstract (ZH)**: 对比学习已成为无监督表征学习中的关键方法，像Momentum Contrast (MoCo)这样的框架有效利用了大量负样本集来提取区分性特征。然而，传统方法往往未能充分利用关键嵌入的全部潜力，并且容易因记忆库中的噪声负样本而导致性能下降。本研究通过提出一种增强的对比学习框架来应对这些挑战，并且该框架包含两个创新。首先，我们引入了一种双视图损失函数，以确保查询和关键嵌入的平衡优化，从而提高表征质量。其次，我们开发了一种基于余弦相似度的选择性负样本策略，以强调最具挑战性的负样本，从而降低噪声的影响并增强特征辨别能力。广泛实验表明，我们的框架在下游任务中实现了更好的性能，提供了鲁棒且结构良好的表征。这些结果强调了优化对比机制在推进无监督学习方面的潜力，并拓展了其在诸如计算机视觉和自然语言处理等领域的应用。 

---
# EVolutionary Independent DEtermiNistiC Explanation 

**Title (ZH)**: 演化独立确定性解释

（注：这是一个直译，结合上下文，“EVolutionary Independent DEtermiNistiC Explanation”的意思可能更具体一些，不过在没有更多上下文信息的情况下，“演化独立确定性解释”是最准确的翻译。如果有更多背景信息，我可以提供更加精准的翻译。） 

**Authors**: Vincenzo Dentamaro, Paolo Giglio, Donato Impedovo, Giuseppe Pirlo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16357)  

**Abstract**: The widespread use of artificial intelligence deep neural networks in fields such as medicine and engineering necessitates understanding their decision-making processes. Current explainability methods often produce inconsistent results and struggle to highlight essential signals influencing model inferences. This paper introduces the Evolutionary Independent Deterministic Explanation (EVIDENCE) theory, a novel approach offering a deterministic, model-independent method for extracting significant signals from black-box models. EVIDENCE theory, grounded in robust mathematical formalization, is validated through empirical tests on diverse datasets, including COVID-19 audio diagnostics, Parkinson's disease voice recordings, and the George Tzanetakis music classification dataset (GTZAN). Practical applications of EVIDENCE include improving diagnostic accuracy in healthcare and enhancing audio signal analysis. For instance, in the COVID-19 use case, EVIDENCE-filtered spectrograms fed into a frozen Residual Network with 50 layers improved precision by 32% for positive cases and increased the area under the curve (AUC) by 16% compared to baseline models. For Parkinson's disease classification, EVIDENCE achieved near-perfect precision and sensitivity, with a macro average F1-Score of 0.997. In the GTZAN, EVIDENCE maintained a high AUC of 0.996, demonstrating its efficacy in filtering relevant features for accurate genre classification. EVIDENCE outperformed other Explainable Artificial Intelligence (XAI) methods such as LIME, SHAP, and GradCAM in almost all metrics. These findings indicate that EVIDENCE not only improves classification accuracy but also provides a transparent and reproducible explanation mechanism, crucial for advancing the trustworthiness and applicability of AI systems in real-world settings. 

**Abstract (ZH)**: 人工智能深度神经网络在医学和工程等领域的广泛应用亟需理解其决策过程。当前的可解释性方法常常产生不一致的结果，并且难以突出对模型推断至关重要的关键信号。本文提出了一种新的方法——进化独立确定性解释（EVIDENCE）理论，该方法提供了一种独立于模型的确定性方法，用于从黑盒模型中提取重要的信号。EVIDENCE理论基于严格的数学建模，并通过在不同类型的数据集上进行实证测试得到了验证，包括用于COVID-19音频诊断的数据集、帕金森病语音记录以及乔治·祖内塔基音乐分类数据集（GTZAN）。EVIDENCE的实际应用包括在医疗诊断中提高诊断准确性以及增强音频信号分析。例如，在COVID-19的应用场景中，通过滤波器提取的EVIDENCE谱图输入50层的预训练残差网络，精准度在阳性病例上提高了32%，AUC值相较于基线模型提高了16%。对于帕金森病分类，EVIDENCE实现了近乎完美的精确度和灵敏度，宏F1分数达到了0.997。在GTZAN数据集中，EVIDENCE保持了高AUC值0.996，显示了其在提取相关特征以实现准确音乐流派分类方面的有效性。在几乎所有指标上，EVIDENCE都优于其他可解释人工智能（XAI）方法，如LIME、SHAP和GradCAM。这些发现表明，EVIDENCE不仅提高了分类准确性，还提供了一种透明且可重复的解释机制，这对于提高AI系统在实际应用中的可信度和适用性至关重要。 

---
# Evaluating Binary Decision Biases in Large Language Models: Implications for Fair Agent-Based Financial Simulations 

**Title (ZH)**: 评估大型语言模型中的二元决策偏见：对其公平的基于代理的金融模拟的影响 

**Authors**: Alicia Vidler, Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2501.16356)  

**Abstract**: Large Language Models (LLMs) are increasingly being used to simulate human-like decision making in agent-based financial market models (ABMs). As models become more powerful and accessible, researchers can now incorporate individual LLM decisions into ABM environments. However, integration may introduce inherent biases that need careful evaluation. In this paper we test three state-of-the-art GPT models for bias using two model sampling approaches: one-shot and few-shot API queries. We observe significant variations in distributions of outputs between specific models, and model sub versions, with GPT-4o-Mini-2024-07-18 showing notably better performance (32-43% yes responses) compared to GPT-4-0125-preview's extreme bias (98-99% yes responses). We show that sampling methods and model sub-versions significantly impact results: repeated independent API calls produce different distributions compared to batch sampling within a single call. While no current GPT model can simultaneously achieve a uniform distribution and Markovian properties in one-shot testing, few-shot sampling can approach uniform distributions under certain conditions. We explore the Temperature parameter, providing a definition and comparative results. We further compare our results to true random binary series and test specifically for the common human bias of Negative Recency - finding LLMs have a mixed ability to 'beat' humans in this one regard. These findings emphasise the critical importance of careful LLM integration into ABMs for financial markets and more broadly. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用于模拟基于代理的金融市场模型（ABMs）中的人类决策。随着模型变得越来越强大和易于访问，研究人员现在可以将单个LLM决策纳入ABM环境。然而，集成可能引入需要仔细评估的固有偏差。本文使用单次查询和少量示例API查询两种模型抽样方法测试了三种最先进的GPT模型的偏差情况。我们观察到不同模型及其子版本之间的输出分布存在显著差异，GPT-4o-Mini-2024-07-18 显示出明显更好的表现（32-43% 的肯定回答），而 GPT-4-0125-preview 则表现出极大的偏差（98-99% 的肯定回答）。我们表明，抽样方法和模型子版本对结果有显著影响：重复的独立API调用与单次调用中的批量抽样相比会产生不同的分布。虽然目前没有GPT模型能够在单次测试中同时实现均匀分布和马尔可夫性质，但在特定条件下，少量示例抽样可以接近均匀分布。我们探讨了温度参数的定义和比较结果。我们进一步将我们的结果与真正的随机二进制序列进行比较，并特别测试了常见的负最近性偏见，发现LLMs在这方面的‘击败’人类的能力存在混合表现。这些发现强调了在金融市场及更广泛的领域中谨慎将LLM集成到ABMs中的重要性。 

---
# How Strategic Agents Respond: Comparing Analytical Models with LLM-Generated Responses in Strategic Classification 

**Title (ZH)**: 战略参与者的响应方式：比较分析模型与大语言模型生成的响应在战略分类中的表现 

**Authors**: Tian Xie, Pavan Rauch, Xueru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16355)  

**Abstract**: When machine learning (ML) algorithms are used to automate human-related decisions, human agents may gain knowledge of the decision policy and behave strategically to obtain desirable outcomes. Strategic Classification (SC) has been proposed to address the interplay between agents and decision-makers. Prior work on SC has relied on assumptions that agents are perfectly or approximately rational, responding to decision policies by maximizing their utilities. Verifying these assumptions is challenging due to the difficulty of collecting real-world agent responses. Meanwhile, the growing adoption of large language models (LLMs) makes it increasingly likely that human agents in SC settings will seek advice from these tools. We propose using strategic advice generated by LLMs to simulate human agent responses in SC. Specifically, we examine five critical SC scenarios -- hiring, loan applications, school admissions, personal income, and public assistance programs -- and simulate how human agents with diverse profiles seek advice from LLMs. We then compare the resulting agent responses with the best responses generated by existing theoretical models. Our findings reveal that: (i) LLMs and theoretical models generally lead to agent score or qualification changes in the same direction across most settings, with both achieving similar levels of fairness; (ii) state-of-the-art commercial LLMs (e.g., GPT-3.5, GPT-4) consistently provide helpful suggestions, though these suggestions typically do not result in maximal score or qualification improvements; and (iii) LLMs tend to produce more diverse agent responses, often favoring more balanced effort allocation strategies. These results suggest that theoretical models align with LLMs to some extent and that leveraging LLMs to simulate more realistic agent responses offers a promising approach to designing trustworthy ML systems. 

**Abstract (ZH)**: 当机器学习（ML）算法用于自动化与人类决策相关的过程时，人类代理可能会了解决策策略并采取战略性行为以获得理想的结果。战略分类（SC）已被提出以应对代理与决策制定者之间的影响。先前关于SC的研究依赖于这样的假设：代理是完全理性或近似理性，并通过最大化自己的效用来响应决策策略。验证这些假设具有挑战性，因为收集真实世界的代理响应非常困难。同时，大语言模型（LLMs）的广泛应用使得在SC环境中，人类代理更有可能从这些工具中寻求建议。我们提议使用由LLMs生成的战略建议来模拟SC中的代理响应。具体而言，我们考察了五个关键的SC场景——招聘、贷款申请、学校录取、个人收入以及公共援助项目，并模拟具有不同特征的人类代理从LLMs寻求建议的过程。然后，我们将这些代理的响应与现有理论模型生成的最佳响应进行了比较。我们的发现表明：(i) 在大多数情况下，LLMs和理论模型在大多数环境下的代理评分或资格变化方向一致，并且两者都达到了相似的公平程度；(ii) 最先进的商业LLMs（如GPT-3.5、GPT-4）一直提供有益的建议，尽管这些建议通常不会导致评分或资格的最大改善；(iii) LLMS倾向于生成更加多样化的代理响应，经常倾向于更好地分配努力策略。这些结果表明，理论模型在某种程度上与LLMs一致，并且利用LLM模拟更真实的代理响应来设计值得信赖的ML系统是一个有前景的方法。 

---
# Adaptive Hoeffding Tree with Transfer Learning for Streaming Synchrophasor Data Sets 

**Title (ZH)**: 基于迁移学习的自适应霍夫丁树在流式同步相量数据集中的应用 

**Authors**: Zakaria El Mrabet, Daisy Flora Selvaraj, Prakash Ranganathan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16354)  

**Abstract**: Synchrophasor technology or phasor measurement units (PMUs) are known to detect multiple type of oscillations or faults better than Supervisory Control and Data Acquisition (SCADA) systems, but the volume of Bigdata (e.g., 30-120 samples per second on a single PMU) generated by these sensors at the aggregator level (e.g., several PMUs) requires special handling. Conventional machine learning or data mining methods are not suitable to handle such larger streaming realtime data. This is primarily due to latencies associated with cloud environments (e.g., at an aggregator or PDC level), and thus necessitates the need for local computing to move the data on the edge (or locally at the PMU level) for processing. This requires faster real-time streaming algorithms to be processed at the local level (e.g., typically by a Field Programmable Gate Array (FPGA) based controllers). This paper proposes a transfer learning-based hoeffding tree with ADWIN (THAT) method to detect anomalous synchrophasor signatures. The proposed algorithm is trained and tested with the OzaBag method. The preliminary results with transfer learning indicate that a computational time saving of 0.7ms is achieved with THAT algorithm (0.34ms) over Ozabag (1.04ms), while the accuracy of both methods in detecting fault events remains at 94% for four signatures. 

**Abstract (ZH)**: 同步相量技术或相量测量单元（PMU）比监督控制和数据采集（SCADA）系统更能检测多种类型的振荡或故障。然而，在聚合器级别（例如，多个PMU）生成的大数据量（例如，每秒30-120个样本）需要特别处理。传统的机器学习或数据挖掘方法不适用于处理这种大规模的实时流数据。这主要是因为云环境相关的延迟（例如，在聚合器或PDC级别），因此需要将数据在边缘进行计算（或在PMU级别进行本地处理）以进行处理。这需要在本地水平上更快地处理实时流数据（例如，通常由基于现场可编程门阵列（FPGA）的控制器处理）。本文提出了一种基于迁移学习的霍夫丁树加ADWIN（THAT）方法来检测异常的同步相量特征。所提出的算法使用OzaBag方法进行了训练和测试。初步结果显示，与OzaBag方法（1.04毫秒）相比，THAT算法（0.34毫秒）实现了0.7毫秒的计算时间节省，而在检测故障事件方面，两种方法的准确性仍保持在94%（四种特征）。 

---
# Synthetic Data Generation by Supervised Neural Gas Network for Physiological Emotion Recognition Data 

**Title (ZH)**: 监督神经气网络生成合成数据用于生理情绪识别数据 

**Authors**: S. Muhammad Hossein Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2501.16353)  

**Abstract**: Data scarcity remains a significant challenge in the field of emotion recognition using physiological signals, as acquiring comprehensive and diverse datasets is often prevented by privacy concerns and logistical constraints. This limitation restricts the development and generalization of robust emotion recognition models, making the need for effective synthetic data generation methods more critical. Emotion recognition from physiological signals such as EEG, ECG, and GSR plays a pivotal role in enhancing human-computer interaction and understanding human affective states. Utilizing these signals, this study introduces an innovative approach to synthetic data generation using a Supervised Neural Gas (SNG) network, which has demonstrated noteworthy speed advantages over established models like Conditional VAE, Conditional GAN, diffusion model, and Variational LSTM. The Neural Gas network, known for its adaptability in organizing data based on topological and feature-space proximity, provides a robust framework for generating real-world-like synthetic datasets that preserve the intrinsic patterns of physiological emotion data. Our implementation of the SNG efficiently processes the input data, creating synthetic instances that closely mimic the original data distributions, as demonstrated through comparative accuracy assessments. In experiments, while our approach did not universally outperform all models, it achieved superior performance against most of the evaluated models and offered significant improvements in processing time. These outcomes underscore the potential of using SNG networks for fast, efficient, and effective synthetic data generation in emotion recognition applications. 

**Abstract (ZH)**: 情感识别领域中基于生理信号的情绪识别仍然面临着数据稀缺的重大挑战，因为隐私问题和实际操作中的限制往往阻碍了获取全面且多样化的数据集。这种限制限制了稳健的情绪识别模型的发展与推广，使得有效的合成数据生成方法变得尤为重要。基于EEG（脑电图）、ECG（心电图）和GSR（皮肤导电率）等生理信号的情感识别，在提升人机交互和理解人类情绪状态方面起着关键作用。利用这些信号，本研究提出了一种创新的合成数据生成方法，即使用监督神经气体（Supervised Neural Gas, SNG）网络，该方法在速度上显著优于现有模型，如条件VAE（变分自编码器）、条件GAN（生成对抗网络）、扩散模型和变分LSTM（长短时记忆网络）。神经气体网络以其能够在拓扑和特征空间附近组织数据的能力而闻名，为生成保留生理情绪数据内在模式的真实世界仿真实例提供了坚实的基础。我们的SNG实现高效地处理输入数据，生成能够接近原始数据分布的合成实例，这一点在对比准确性的评估中得到了证实。在实验中，虽然我们的方法并非在所有模型上均表现出色，但其在大多数评估模型上表现更为优越，并显著降低了处理时间。这些结果强调了使用SNG网络在情感识别应用中进行快速、高效和有效的合成数据生成的潜在优势。 

---
# Mixture of Experts (MoE): A Big Data Perspective 

**Title (ZH)**: 混合专家模型（MoE）：从大数据视角探讨 

**Authors**: Wensheng Gan, Zhenyao Ning, Zhenlian Qi, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16352)  

**Abstract**: As the era of big data arrives, traditional artificial intelligence algorithms have difficulty processing the demands of massive and diverse data. Mixture of experts (MoE) has shown excellent performance and broad application prospects. This paper provides an in-depth review and analysis of the latest progress in this field from multiple perspectives, including the basic principles, algorithmic models, key technical challenges, and application practices of MoE. First, we introduce the basic concept of MoE and its core idea and elaborate on its advantages over traditional single models. Then, we discuss the basic architecture of MoE and its main components, including the gating network, expert networks, and learning algorithms. Next, we review the applications of MoE in addressing key technical issues in big data. For each challenge, we provide specific MoE solutions and their innovations. Furthermore, we summarize the typical use cases of MoE in various application domains. This fully demonstrates the powerful capability of MoE in big data processing. We also analyze the advantages of MoE in big data environments. Finally, we explore the future development trends of MoE. We believe that MoE will become an important paradigm of artificial intelligence in the era of big data. In summary, this paper systematically elaborates on the principles, techniques, and applications of MoE in big data processing, providing theoretical and practical references to further promote the application of MoE in real scenarios. 

**Abstract (ZH)**: 随着大数据时代的到来，传统的机器学习算法在处理海量且多样的数据时遇到了困难。专家混合（MoE）算法展现出了卓越的性能和广泛的应用前景。本文从多个角度对MoE领域的最新进展进行了深入的回顾和分析，涵盖了MoE的基本原理、算法模型、关键技术挑战及其应用实践。首先，我们介绍了MoE的基本概念及其核心思想，并详细阐述了其相对于传统单一模型的优势。然后，我们讨论了MoE的基本架构及其主要组成部分，包括门控网络、专家网络和学习算法。接下来，我们回顾了MoE在解决大数据领域关键技术问题的应用。对于每个挑战，我们提供了具体的MoE解决方案及其创新之处。此外，我们总结了MoE在各种应用领域的典型用例。这充分展示了MoE在大数据处理中的强大能力。我们还分析了MoE在大数据环境中的优势。最后，我们探讨了MoE的未来发展方向。我们认为MoE将成为大数据时代的重要人工智能范式。总之，本文系统地阐述了MoE在大数据处理中的原理、技术和应用，为MoE在实际场景中的进一步应用提供了理论和实践参考。 

---
# A Method for Multi-Hop Question Answering on Persian Knowledge Graph 

**Title (ZH)**: 用于波斯知识图谱的多跳问答方法 

**Authors**: Arash Ghafouri, Mahdi Firouzmandi, Hasan Naderi  

**Link**: [PDF](https://arxiv.org/pdf/2501.16350)  

**Abstract**: Question answering systems are the latest evolution in information retrieval technology, designed to accept complex queries in natural language and provide accurate answers using both unstructured and structured knowledge sources. Knowledge Graph Question Answering (KGQA) systems fulfill users' information needs by utilizing structured data, representing a vast number of facts as a graph. However, despite significant advancements, major challenges persist in answering multi-hop complex questions, particularly in Persian. One of the main challenges is the accurate understanding and transformation of these multi-hop complex questions into semantically equivalent SPARQL queries, which allows for precise answer retrieval from knowledge graphs. In this study, to address this issue, a dataset of 5,600 Persian multi-hop complex questions was developed, along with their decomposed forms based on the semantic representation of the questions. Following this, Persian language models were trained using this dataset, and an architecture was proposed for answering complex questions using a Persian knowledge graph. Finally, the proposed method was evaluated against similar systems on the PeCoQ dataset. The results demonstrated the superiority of our approach, with an improvement of 12.57% in F1-score and 12.06% in accuracy compared to the best comparable method. 

**Abstract (ZH)**: 问答系统是信息检索技术的最新演变，旨在接受用自然语言编写的复杂查询，并利用结构化和非结构化知识源提供准确的答案。知识图谱问答（KGQA）系统通过利用结构化数据来满足用户的信息需求，将大量事实表示为图。尽管取得了显著的进步，但在回答多跳复杂问题（特别是在波斯语中）方面仍然存在重大挑战。其中主要的挑战之一是准确地理解和转换这些多跳复杂问题为语义等价的SPARQL查询，以便从知识图谱中精确检索答案。在本研究中，为了解决这一问题，我们开发了一个包含5,600个波斯语多跳复杂问题及其基于问题语义分解的语料库，然后使用该语料库训练了波斯语语言模型，并提出了一种使用波斯语知识图谱回答复杂问题的体系结构。最后，我们提出的方案在PeCoQ数据集上与类似系统进行了评估。结果显示，我们的方法在F1分数和准确性上分别优于最佳可比方法12.57%和12.06%。 

---
# Risk-Informed Diffusion Transformer for Long-Tail Trajectory Prediction in the Crash Scenario 

**Title (ZH)**: 面向风险的扩散变压器在碰撞情景下的长尾轨迹预测 

**Authors**: Junlan Chen, Pei Liu, Zihao Zhang, Hongyi Zhao, Yufei Ji, Ziyuan Pu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16349)  

**Abstract**: Trajectory prediction methods have been widely applied in autonomous driving technologies. Although the overall performance accuracy of trajectory prediction is relatively high, the lack of trajectory data in critical scenarios in the training data leads to the long-tail phenomenon. Normally, the trajectories of the tail data are more critical and more difficult to predict and may include rare scenarios such as crashes. To solve this problem, we extracted the trajectory data from real-world crash scenarios, which contain more long-tail data. Meanwhile, based on the trajectory data in this scenario, we integrated graph-based risk information and diffusion with transformer and proposed the Risk-Informed Diffusion Transformer (RI-DiT) trajectory prediction method. Extensive experiments were conducted on trajectory data in the real-world crash scenario, and the results show that the algorithm we proposed has good performance. When predicting the data of the tail 10\% (Top 10\%), the minADE and minFDE indicators are 0.016/2.667 m. At the same time, we showed the trajectory conditions of different long-tail distributions. The distribution of trajectory data is closer to the tail, the less smooth the trajectory is. Through the trajectory data in real-world crash scenarios, Our work expands the methods to overcome the long-tail challenges in trajectory prediction. Our method, RI-DiT, integrates inverse time to collision (ITTC) and the feature of traffic flow, which can predict long-tail trajectories more accurately and improve the safety of autonomous driving systems. 

**Abstract (ZH)**: 轨迹预测方法已经在自动驾驶技术中得到了广泛应用。尽管轨迹预测的总体性能准确率相对较高，但在训练数据中关键场景的轨迹数据不足导致了长尾现象。通常，尾部数据的轨迹更为关键且更难以预测，可能包括碰撞等罕见场景。为了解决这一问题，我们从实际碰撞场景中提取了轨迹数据，这些数据包含了更多的长尾数据。同时，基于这些场景下的轨迹数据，我们整合了基于图的风险信息和扩散机制，并结合了Transformer技术，提出了风险指导的扩散变换器（Risk-Informed Diffusion Transformer, RI-DiT）轨迹预测方法。我们在实际碰撞场景下的轨迹数据上进行了广泛的实验，结果表明所提出算法具有良好的性能。在预测尾部10%（Top 10%）的数据时，最小平均误差（minADE）和最小最终误差（minFDE）分别为0.016米和2.667米。同时，我们展示了不同长尾分布的轨迹条件。轨迹数据分布越接近尾部，轨迹越不平滑。通过实际碰撞场景的轨迹数据，我们的工作扩展了克服轨迹预测中长尾挑战的方法。我们的方法RI-DiT结合了倒计时碰撞时间（Inverted Time to Collision, ITTC）和交通流特征，能够更准确地预测长尾轨迹，提高自动驾驶系统的安全性。 

---
# An Integrated Approach to AI-Generated Content in e-health 

**Title (ZH)**: AI生成内容在数字医疗中的综合方法 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2501.16348)  

**Abstract**: Artificial Intelligence-Generated Content, a subset of Generative Artificial Intelligence, holds significant potential for advancing the e-health sector by generating diverse forms of data. In this paper, we propose an end-to-end class-conditioned framework that addresses the challenge of data scarcity in health applications by generating synthetic medical images and text data, evaluating on practical applications such as retinopathy detection, skin infections and mental health assessments. Our framework integrates Diffusion and Large Language Models (LLMs) to generate data that closely match real-world patterns, which is essential for improving downstream task performance and model robustness in e-health applications. Experimental results demonstrate that the synthetic images produced by the proposed diffusion model outperform traditional GAN architectures. Similarly, in the text modality, data generated by uncensored LLM achieves significantly better alignment with real-world data than censored models in replicating the authentic tone. 

**Abstract (ZH)**: 人工智能生成内容是生成式人工智能的一个子集，具有显著潜力，能够通过生成多样化的数据来推动电子健康领域的发展。本文提出了一种端到端条件分类框架，通过生成合成医学图像和文本数据来应对健康应用中数据稀缺的挑战，并通过视网膜病变检测、皮肤感染和心理健康评估等实际应用场景进行了评估。该框架结合了扩散模型和大型语言模型（LLMs），生成的数据能更紧密地匹配现实世界的模式，这对提高电子健康应用中下游任务的性能和模型 robustness 至关重要。实验结果表明，所提出的扩散模型生成的合成图像在下游任务性能上优于传统 GAN 架构。在文本模态方面，未经限制的 LLM 生成的数据在模仿真实语气方面比经过限制的模型表现出了显著更好的对齐效果。 

---
# Identification of Hardware Trojan Locations in Gate-Level Netlist using Nearest Neighbour Approach integrated with Machine Learning Technique 

**Title (ZH)**: 使用最近邻方法结合机器学习技术在门级网表中识别硬件木马位置 

**Authors**: Anindita Chattopadhyay, Siddharth Bisariya, Vijay Kumar Sutrakar  

**Link**: [PDF](https://arxiv.org/pdf/2501.16347)  

**Abstract**: In the evolving landscape of integrated circuit design, detecting Hardware Trojans (HTs) within a multi entity based design cycle presents significant challenges. This research proposes an innovative machine learning-based methodology for identifying malicious logic gates in gate-level netlists. By focusing on path retrace algorithms. The methodology is validated across three distinct cases, each employing different machine learning models to classify HTs. Case I utilizes a decision tree algorithm for node-to-node comparisons, significantly improving detection accuracy through the integration of Principal Component Analysis (PCA). Case II introduces a graph-to-graph classification using a Graph Neural Network (GNN) model, enabling the differentiation between normal and Trojan-infected circuit designs. Case III applies GNN-based node classification to identify individual compromised nodes and its location. Additionally, nearest neighbor (NN) method has been combined with GNN graph-to-graph in Case II and GNN node-to-node in Case III. Despite the potential of GNN model graph-to-graph classification, NN approach demonstrated superior performance, with the first nearest neighbor (1st NN) achieving 73.2% accuracy and the second nearest neighbor (2nd NN) method reaching 97.7%. In comparison, the GNN model achieved an accuracy of 62.8%. Similarly, GNN model node-to-node classification, NN approach demonstrated superior performance, with the 1st NN achieving 93% accuracy and the 2nd NN method reaching 97.7%. In comparison, the GNN model achieved an accuracy of 79.8%. However, higher and higher NN will lead to large code coverage for the identification of HTs. 

**Abstract (ZH)**: 随着集成电路设计的不断发展，多实体设计周期中检测硬件木马（HTs）面临重大挑战。本文提出了一种基于机器学习的创新方法，用于识别门级网表中的恶意逻辑门。该方法侧重于路径回溯算法，并在三种不同的实验案例中进行了验证，每种案例都采用了不同的机器学习模型来分类HTs。案例I使用了决策树算法进行节点间比较，并通过集成主成分分析（PCA）显著提高了检测准确性。案例II采用图神经网络（GNN）模型进行图到图分类，实现正常电路设计与受木马感染电路设计之间的区分。案例III运用基于GNN的节点分类方法识别个别被感染节点及其位置。此外，案例II结合了最近邻（NN）方法与GNN的图到图分类，案例III则将NN方法与GNN的节点到节点分类相结合。尽管GNN模型在图到图分类中具有潜力，但NN方法表现更优，其中1st NN的准确率为73.2%，2nd NN方法达到97.7%。相比之下，GNN模型的准确率为62.8%。同样，在GNN模型的节点到节点分类中，NN方法也表现出更优性能，其中1st NN的准确率达到93%，2nd NN方法达到97.7%，而GNN模型的准确率为79.8%。然而，更高阶的NN方法将导致更高的代码覆盖率，有助于HTs的识别。 

---
# Self-supervised Graph Transformer with Contrastive Learning for Brain Connectivity Analysis towards Improving Autism Detection 

**Title (ZH)**: 基于对比学习的自监督图变压器在改善自闭症检测中的脑连接性分析 

**Authors**: Yicheng Leng, Syed Muhammad Anwar, Islem Rekik, Sen He, Eung-Joo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.16346)  

**Abstract**: Functional Magnetic Resonance Imaging (fMRI) provides useful insights into the brain function both during task or rest. Representing fMRI data using correlation matrices is found to be a reliable method of analyzing the inherent connectivity of the brain in the resting and active states. Graph Neural Networks (GNNs) have been widely used for brain network analysis due to their inherent explainability capability. In this work, we introduce a novel framework using contrastive self-supervised learning graph transformers, incorporating a brain network transformer encoder with random graph alterations. The proposed network leverages both contrastive learning and graph alterations to effectively train the graph transformer for autism detection. Our approach, tested on Autism Brain Imaging Data Exchange (ABIDE) data, demonstrates superior autism detection, achieving an AUROC of 82.6 and an accuracy of 74%, surpassing current state-of-the-art methods. 

**Abstract (ZH)**: 功能磁共振成像(fMRI)在任务执行或静息状态下提供了关于大脑功能有价值的见解。使用相关矩阵表示fMRI数据已被证明是分析静息和激活状态下大脑固有连接性的可靠方法。图神经网络(GNNs)因其固有的可解释性能力而广泛应用于脑网络分析。在本研究中，我们提出了一种新颖的框架，结合对比自监督学习图变换器和随机图变形，其中包含脑网络变换器编码器。所提出的方法结合对比学习和图变形有效地训练图变换器以进行自闭症检测。在使用Autism Brain Imaging Data Exchange (ABIDE)数据集测试时，我们的方法展示了在自闭症检测方面的优越性能，AUROC达到82.6，准确率达到74%，超过了当前最先进的方法。 

---
# Self-Clustering Graph Transformer Approach to Model Resting-State Functional Brain Activity 

**Title (ZH)**: 基于自我聚类图变换器的方法建模静息态功能脑活动 

**Authors**: Bishal Thapaliya, Esra Akbas, Ram Sapkota, Bhaskar Ray, Vince Calhoun, Jingyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16345)  

**Abstract**: Resting-state functional magnetic resonance imaging (rs-fMRI) offers valuable insights into the human brain's functional organization and is a powerful tool for investigating the relationship between brain function and cognitive processes, as it allows for the functional organization of the brain to be captured without relying on a specific task or stimuli. In this study, we introduce a novel attention mechanism for graphs with subnetworks, named Self-Clustering Graph Transformer (SCGT), designed to handle the issue of uniform node updates in graph transformers. By using static functional connectivity (FC) correlation features as input to the transformer model, SCGT effectively captures the sub-network structure of the brain by performing cluster-specific updates to the nodes, unlike uniform node updates in vanilla graph transformers, further allowing us to learn and interpret the subclusters. We validate our approach on the Adolescent Brain Cognitive Development (ABCD) dataset, comprising 7,957 participants, for the prediction of total cognitive score and gender classification. Our results demonstrate that SCGT outperforms the vanilla graph transformer method and other recent models, offering a promising tool for modeling brain functional connectivity and interpreting the underlying subnetwork structures. 

**Abstract (ZH)**: 静息状态功能性磁共振成像（rs-fMRI）为人类大脑的功能组织提供了宝贵的洞察力，并且是研究大脑功能与认知过程之间关系的强大工具，因为它可以在无需特定任务或刺激的情况下捕获脑的功能组织。在本研究中，我们提出了一种新的基于子网络的图形注意机制，称为自我聚类图形变换器（SCGT），旨在解决图形变换器中节点更新统一的问题。通过将静态功能连接（FC）相关特征作为输入传递给变换器模型，SCGT能够通过进行集群特定的节点更新来有效捕获大脑的子网络结构，不同于传统的图形变换器中的统一节点更新，从而进一步使我们能够学习和解释子集群。我们在包含7,957名参与者的青少年大脑认知发展（ABCD）数据集上验证了我们的方法，用于预测总认知分数和性别分类。研究结果表明，SCGT方法相较于传统的图形变换器方法和其他最近的模型具有更好的性能，为建模大脑功能连接以及解释潜在的子网络结构提供了有前景的工具。 

---
# WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning 

**Title (ZH)**: WhiSPA：语义和心理一致的自我监督对比学习与学生-教师学习的 Whisper 语音隐写术 

**Authors**: Rajath Rao, Adithya Ganesan, Oscar Kjell, Jonah Luby, Akshay Raghavan, Scott Feltman, Whitney Ringwald, Ryan L. Boyd, Benjamin Luft, Camilo Ruggero, Neville Ryant, Roman Kotov, H. Andrew Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2501.16344)  

**Abstract**: Current speech encoding pipelines often rely on separate processing pipelines between text and audio, not fully leveraging the inherent overlap between these modalities for understanding human communication. Language models excel at capturing semantic meaning from text that can complement the additional prosodic, emotional, and acoustic cues from speech. This work bridges the gap by proposing WhiSPA (Whisper with Semantic-Psychological Alignment), a novel audio encoder trained with a contrastive student-teacher learning objective. Using over 500k speech segments from mental health audio interviews, we evaluate the utility of aligning Whisper audio embeddings with text representations from an SBERT encoder and text-based assessments of psychological dimensions: emotion and personality. Over self-supervised and downstream mental health tasks, WhiSPA surpasses state-of-the-art speech models, achieving an average error reduction of 73.4% on the segment-level self-supervised objective and 83.8% on 11 psychological downstream tasks. WhiSPA demonstrates that cross-modal alignment can increase the amount of text-semantic and psychological information captured in audio-only encoder models. 

**Abstract (ZH)**: 当前的语音编码流水线通常依赖于文本和音频之间的独立处理流水线，未充分利用这些模态之间固有的重叠，以更好地理解人类沟通。语言模型擅长从文本中捕获语义意义，这可以补充语音中附加的韵律、情感和声学线索。本研究通过提出WhiSPA（Whisper与语义-心理对齐）来弥合这一差距，WhiSPA是一种通过对比学生-教师学习目标进行训练的新型音频编码器。通过心理健康音频访谈中的超过50万段语音片段，我们评估了将Whisper音频嵌入与来自SBERT编码器的文本表示以及基于文本的心理维度评估（情绪和人格）进行对齐的实用性。在自监督和下游心理健康任务中，WhiSPA超越了最先进的语音模型，在自监督目标的段级指标上平均减少了73.4%的误差，同时在11项心理维度的下游任务上减少了83.8%的误差。WhiSPA表明，跨模态对齐可以增加仅音频编码模型中捕获的文本语义和心理信息量。 

---
# Explore Activation Sparsity in Recurrent LLMs for Energy-Efficient Neuromorphic Computing 

**Title (ZH)**: 探索循环LLM中的激活稀疏性以实现高效的神经形态计算 

**Authors**: Ivan Knunyants, Maryam Tavakol, Manolis Sifalakis, Yingfu Xu, Amirreza Yousefzadeh, Guangzhi Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16337)  

**Abstract**: The recent rise of Large Language Models (LLMs) has revolutionized the deep learning field. However, the desire to deploy LLMs on edge devices introduces energy efficiency and latency challenges. Recurrent LLM (R-LLM) architectures have proven effective in mitigating the quadratic complexity of self-attention, making them a potential paradigm for computing on-edge neuromorphic processors. In this work, we propose a low-cost, training-free algorithm to sparsify R-LLMs' activations to enhance energy efficiency on neuromorphic hardware. Our approach capitalizes on the inherent structure of these models, rendering them well-suited for energy-constrained environments. Although primarily designed for R-LLMs, this method can be generalized to other LLM architectures, such as transformers, as demonstrated on the OPT model, achieving comparable sparsity and efficiency improvements. Empirical studies illustrate that our method significantly reduces computational demands while maintaining competitive accuracy across multiple zero-shot learning benchmarks. Additionally, hardware simulations with the SENECA neuromorphic processor underscore notable energy savings and latency improvements. These results pave the way for low-power, real-time neuromorphic deployment of LLMs and demonstrate the feasibility of training-free on-chip adaptation using activation sparsity. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的兴起已经革新了深度学习领域。然而，希望在边缘设备上部署LLMs带来了能效和延迟方面的挑战。循环LLM（R-LLM）架构已被证明能够有效地缓解自我注意机制的二次复杂性问题，使其成为计算在边神经形态处理器上的潜在范式。在本研究中，我们提出了一种低成本且无需训练的算法，用于稀疏化R-LLMs的激活，以提高在神经形态硬件上的能效。我们的方法利用了这些模型固有的结构，使其非常适合能量受限的环境。虽然这种方法主要针对R-LLMs设计，但它也可以推广到其他LLM架构，如变换器，在OPT模型上的实验表明，它可以实现类似的稀疏性和效率改进。实证研究表明，我们的方法显著降低了计算需求，同时在多个零样本学习基准测试中保持了竞争力。此外，使用SENECA神经形态处理器的硬件仿真进一步证实了显著的能量节省和延迟改进。这些结果为在低功耗环境下实时部署LLMs铺平了道路，并展示了激活稀疏化进行芯片内无训练适应的可能性。 

---
# Runtime Analysis of Evolutionary Algorithms for Multiparty Multiobjective Optimization 

**Title (ZH)**: 多 parties 多目标优化中进化算法的运行时分析 

**Authors**: Yuetong Sun, Peilan Xu, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16336)  

**Abstract**: In scenarios where multiple decision-makers operate within a common decision space, each focusing on their own multi-objective optimization problem (e.g., bargaining games), the problem can be modeled as a multi-party multi-objective optimization problem (MPMOP). While numerous evolutionary algorithms have been proposed to solve MPMOPs, most results remain empirical. This paper presents the first theoretical analysis of the expected runtime of evolutionary algorithms on bi-party multi-objective optimization problems (BPMOPs). Our findings demonstrate that employing traditional multi-objective optimization algorithms to solve MPMOPs is both time-consuming and inefficient, as the resulting population contains many solutions that fail to achieve consensus among decision-makers. An alternative approach involves decision-makers individually solving their respective optimization problems and seeking consensus only in the final stage. While feasible for pseudo-Boolean optimization problems, this method may fail to guarantee approximate performance for one party in NP-hard problems. Finally, We propose coevolutionary multi-party multi-objective optimizers (CoEMPMO) for pseudo-Boolean optimization and shortest path problems within a multi-party multi-objective context, which maintains a common solution set among all parties through coevolution. Theoretical and experimental results demonstrate that the proposed \( \text{CoEMPMO}_{\text{random}} \) outperforms previous algorithms in terms of the expected lower bound on runtime for pseudo-Boolean optimization problems. Additionally, \( \text{CoEMPMO}_{\text{cons}}^{\text{SP}} \) achieves better efficiency and precision in solving shortest path problems compared to existing algorithms. 

**Abstract (ZH)**: 在多个决策者共同决策的空间中，每个决策者关注自己的多目标优化问题（例如，在讨价还价游戏中），该问题可以建模为多当事人多目标优化问题（MPMOP）。尽管已经提出了许多进化算法来解决MPMOPs，但大多数结果仍停留在经验层面。本文首次对进化算法在双当事人多目标优化问题（BPMOPs）上的预期运行时间进行了理论分析。我们的发现表明，使用传统多目标优化算法来解决MPMOPs既耗时又低效，因为产生的种群包含了许多未能达成决策者间共识的解。另一种方法是让决策者各自独立解决各自的优化问题，在最终阶段仅进行共识寻求。这种方法在伪布尔优化问题中是可行的，但在NP-难问题中可能无法保证一方的近似性能。最后，我们提出了多当事人多目标优化的共进化算法（CoEMPMO），用于伪布尔优化和最短路径问题中的多当事人多目标优化场景，通过共进化机制保持了所有当事人之间的共同解集。理论和实验结果表明，提出的 \( \text{CoEMPMO}_{\text{random}} \) 在伪布尔优化问题的预期下界运行时间方面优于先前算法。此外，\( \text{CoEMPMO}_{\text{cons}}^{\text{SP}} \) 在解决最短路径问题的效率和精确度方面也优于现有算法。 

---
# Decoding OTC Government Bond Market Liquidity: An ABM Model for Market Dynamics 

**Title (ZH)**: 解码OTC政府债券市场流动性：基于市场动力的ABM模型 

**Authors**: Alicia Vidler, Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2501.16331)  

**Abstract**: The over-the-counter (OTC) government bond markets are characterised by their bilateral trading structures, which pose unique challenges to understanding and ensuring market stability and liquidity. In this paper, we develop a bespoke ABM that simulates market-maker interactions within a stylised government bond market. The model focuses on the dynamics of liquidity and stability in the secondary trading of government bonds, particularly in concentrated markets like those found in Australia and the UK. Through this simulation, we test key hypotheses around improving market stability, focusing on the effects of agent diversity, business costs, and client base size. We demonstrate that greater agent diversity enhances market liquidity and that reducing the costs of market-making can improve overall market stability. The model offers insights into computational finance by simulating trading without price transparency, highlighting how micro-structural elements can affect macro-level market outcomes. This research contributes to the evolving field of computational finance by employing computational intelligence techniques to better understand the fundamental mechanics of government bond markets, providing actionable insights for both academics and practitioners. 

**Abstract (ZH)**: 场外（OTC）政府债券市场因其双边交易结构而独具特色，这给市场稳定性和流动性的理解和保障带来了独特挑战。本文旨在开发一个量身定制的 agent-based 模型（ABM），用于模拟在简化政府债券市场中的做市商互动。该模型重点关注政府债券二级市场的流动性和稳定性动态，尤其是在澳大利亚和英国等集中市场。通过这一模拟，我们测试了关于提高市场稳定性的关键假设，重点关注代理多样性、业务成本以及客户基础规模的影响。研究结果显示，增强代理多样性可以提升市场流动性，而减少做市成本可以提高市场的整体稳定性。该模型通过在缺乏价格透明度的情况下模拟交易，展示了微观结构要素如何影响宏观市场结果，为计算金融领域提供了新的见解。本文通过运用计算智能技术，对政府债券市场的基本机制进行了更深入的理解，为学术界和实务界提供了可操作的洞见，从而促进了计算金融领域的不断发展。 

---
