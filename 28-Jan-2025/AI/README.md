# Enhancing Visual Inspection Capability of Multi-Modal Large Language Models on Medical Time Series with Supportive Conformalized and Interpretable Small Specialized Models 

**Title (ZH)**: 增强多模态大型语言模型在医疗时间序列视觉检查能力的支持性同构造化和可解释小型专业化模型辅助方法 

**Authors**: Huayu Li, Xiwen Chen, Ci Zhang, Stuart F. Quan, William D.S. Killgore, Shu-Fen Wung, Chen X. Chen, Geng Yuan, Jin Lu, Ao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.16215)  

**Abstract**: Large language models (LLMs) exhibit remarkable capabilities in visual inspection of medical time-series data, achieving proficiency comparable to human clinicians. However, their broad scope limits domain-specific precision, and proprietary weights hinder fine-tuning for specialized datasets. In contrast, small specialized models (SSMs) excel in targeted tasks but lack the contextual reasoning required for complex clinical decision-making. To address these challenges, we propose ConMIL (Conformalized Multiple Instance Learning), a decision-support SSM that integrates seamlessly with LLMs. By using Multiple Instance Learning (MIL) to identify clinically significant signal segments and conformal prediction for calibrated set-valued outputs, ConMIL enhances LLMs' interpretative capabilities for medical time-series analysis. Experimental results demonstrate that ConMIL significantly improves the performance of state-of-the-art LLMs, such as ChatGPT4.0 and Qwen2-VL-7B. Specifically, \ConMIL{}-supported Qwen2-VL-7B achieves 94.92% and 96.82% precision for confident samples in arrhythmia detection and sleep staging, compared to standalone LLM accuracy of 46.13% and 13.16%. These findings highlight the potential of ConMIL to bridge task-specific precision and broader contextual reasoning, enabling more reliable and interpretable AI-driven clinical decision support. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医学时间序列数据的视觉检查方面表现出显著的能力，其专业水平接近于人类临床医生。然而，其广泛的适用范围限制了其在特定领域的精密度，而专有的模型权重则阻碍了对专门数据集的微调。相比之下，小型专门模型（SSMs）在执行特定任务方面表现出色，但缺乏进行复杂临床决策所需的背景推理能力。为解决这些挑战，我们提出了一种决策支持的小型专门模型（ConMIL，Conformalized Multiple Instance Learning），该模型能够与LLMs无缝集成。通过使用多重实例学习（MIL）识别具有临床意义的信号片段，并使用校准型集合值输出的卷积预测，ConMIL增强了LLMs对医学时间序列分析的解释能力。实验结果表明，ConMIL显著提高了当前先进的LLMs（如ChatGPT4.0和Qwen2-VL-7B）的表现。具体而言，ConMIL支持的Qwen2-VL-7B在心律失常检测和睡眠阶段分类中的精确度分别达到了94.92%和96.82%，而单独的LLMs在这两项任务中的精度分别为46.13%和13.16%。这些发现突显了ConMIL在任务特定精度和更广泛背景推理之间架起桥梁的潜力，从而能够提供更可靠和可解释的AI驱动的临床决策支持。 

---
# From Informal to Formal -- Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs 

**Title (ZH)**: 从非形式化到形式化——集成并评估大语言模型在自然语言需求到可验证的形式证明中的应用 

**Authors**: Jialun Cao, Yaojie Lu, Meiziniu Li, Haoyang Ma, Haokun Li, Mengda He, Cheng Wen, Le Sun, Hongyu Zhang, Shengchao Qin, Shing-Chi Cheung, Cong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2501.16207)  

**Abstract**: The research in AI-based formal mathematical reasoning has shown an unstoppable growth trend. These studies have excelled in mathematical competitions like IMO, showing significant progress. However, these studies intertwined multiple skills simultaneously, i.e., problem-solving, reasoning, and writing formal specifications, making it hard to precisely identify the LLMs' strengths and weaknesses in each task. This paper focuses on formal verification, an immediate application scenario of formal reasoning, and decomposes it into six sub-tasks. We constructed 18k high-quality instruction-response pairs across five mainstream formal specification languages (Coq, Lean4, Dafny, ACSL, and TLA+) in six formal-verification-related tasks by distilling GPT-4o. They are split into a 14k+ fine-tuning dataset FM-alpaca and a 4k benchmark FM-Bench. We found that LLMs are good at writing proof segments when given either the code, or the detailed description of proof steps. Also, the fine-tuning brought about a nearly threefold improvement at most. Interestingly, we observed that fine-tuning with formal data also enhances mathematics, reasoning, and coding abilities. We hope our findings inspire further research. Fine-tuned models are released to facilitate subsequent studies 

**Abstract (ZH)**: 基于AI的正式数学推理研究显示出了不可阻挡的发展趋势。这些研究在数学竞赛中，如国际数学奥林匹克竞赛（IMO），取得了显著的进步。然而，这些研究同时涉及多种技能，包括问题解决、推理和编写形式化规范，使得难以精确识别大模型（LLMs）在每个任务中的优势和不足。本文重点关注形式化验证，这是形式化推理的直接应用场景，并将其分解为六个子任务。我们通过精炼GPT-4o构建了涵盖五个主流形式化规范语言（Coq、Lean4、Dafny、ACSL和TLA+）的18,000个高质量指令-响应对，分布在六个形式化验证相关任务中。这些数据被划分为14,000个以上用于微调的FM-alpaca数据集和4,000个基准测试数据集FM-Bench。我们发现，当给定代码或详细推理步骤的描述时，大模型在编写证明片段方面表现出色。此外，微调带来了大约三倍的进步。有趣的是，我们发现使用形式化数据进行微调也能提升数学、推理和编程能力。我们希望我们的发现能激发进一步的研究，并发布了微调后的模型以促进后续的研究。 

---
# AI Agents for Computer Use: A Review of Instruction-based Computer Control, GUI Automation, and Operator Assistants 

**Title (ZH)**: 基于指令的计算机控制、GUI自动化的AI代理及操作助理：一个综述 

**Authors**: Pascal J. Sager, Benjamin Meyer, Peng Yan, Rebekka von Wartburg-Kottler, Layan Etaiwi, Aref Enayati, Gabriel Nobel, Ahmed Abdulkadir, Benjamin F. Grewe, Thilo Stadelmann  

**Link**: [PDF](https://arxiv.org/pdf/2501.16150)  

**Abstract**: Instruction-based computer control agents (CCAs) execute complex action sequences on personal computers or mobile devices to fulfill tasks using the same graphical user interfaces as a human user would, provided instructions in natural language. This review offers a comprehensive overview of the emerging field of instruction-based computer control, examining available agents -- their taxonomy, development, and respective resources -- and emphasizing the shift from manually designed, specialized agents to leveraging foundation models such as large language models (LLMs) and vision-language models (VLMs). We formalize the problem and establish a taxonomy of the field to analyze agents from three perspectives: (a) the environment perspective, analyzing computer environments; (b) the interaction perspective, describing observations spaces (e.g., screenshots, HTML) and action spaces (e.g., mouse and keyboard actions, executable code); and (c) the agent perspective, focusing on the core principle of how an agent acts and learns to act. Our framework encompasses both specialized and foundation agents, facilitating their comparative analysis and revealing how prior solutions in specialized agents, such as an environment learning step, can guide the development of more capable foundation agents. Additionally, we review current CCA datasets and CCA evaluation methods and outline the challenges to deploying such agents in a productive setting. In total, we review and classify 86 CCAs and 33 related datasets. By highlighting trends, limitations, and future research directions, this work presents a comprehensive foundation to obtain a broad understanding of the field and push its future development. 

**Abstract (ZH)**: 基于指令的计算机控制代理（CCAs）执行个人计算机或移动设备上的复杂操作序列，使用与人类用户相同的图形用户界面，根据自然语言提供的指令完成任务。本文综述了基于指令的计算机控制这一新兴领域，对现有的代理进行了全面的审查，包括它们的分类、开发及其相应的资源，并强调从手动设计的专业代理向利用基础模型（如大型语言模型LLMs和视觉语言模型VLMs）的转变。我们形式化了这一问题，并建立了该领域的分类框架，从三个视角分析代理：（a）环境视角，分析计算机环境；（b）交互视角，描述观察空间（例如屏幕截图、HTML）和动作空间（例如鼠标和键盘操作、可执行代码）；（c）代理视角，专注于代理如何行动和学习行动的核心原理。我们的框架包括专门的和基础的代理，便于它们的比较分析，并揭示了专门代理中的先前解决方案，如环境学习步骤，可以指导更强大基础代理的发展。此外，我们还回顾了现有的CCA数据集和CCA评估方法，并概述了在生产环境中部署此类代理所面临的挑战。总体而言，我们审查并分类了86个CCAs和33个相关数据集。通过突出显示趋势、限制和未来研究方向，本文提供了一个全面的基础，有助于全面理解该领域并推动其未来的发展。 

---
# Flexible Blood Glucose Control: Offline Reinforcement Learning from Human Feedback 

**Title (ZH)**: 灵活的血糖控制：基于人类反馈的离线强化学习 

**Authors**: Harry Emerson, Sam Gordon James, Matthew Guy, Ryan McConville  

**Link**: [PDF](https://arxiv.org/pdf/2501.15972)  

**Abstract**: Reinforcement learning (RL) has demonstrated success in automating insulin dosing in simulated type 1 diabetes (T1D) patients but is currently unable to incorporate patient expertise and preference. This work introduces PAINT (Preference Adaptation for INsulin control in T1D), an original RL framework for learning flexible insulin dosing policies from patient records. PAINT employs a sketch-based approach for reward learning, where past data is annotated with a continuous reward signal to reflect patient's desired outcomes. Labelled data trains a reward model, informing the actions of a novel safety-constrained offline RL algorithm, designed to restrict actions to a safe strategy and enable preference tuning via a sliding scale. In-silico evaluation shows PAINT achieves common glucose goals through simple labelling of desired states, reducing glycaemic risk by 15% over a commercial benchmark. Action labelling can also be used to incorporate patient expertise, demonstrating an ability to pre-empt meals (+10% time-in-range post-meal) and address certain device errors (-1.6% variance post-error) with patient guidance. These results hold under realistic conditions, including limited samples, labelling errors, and intra-patient variability. This work illustrates PAINT's potential in real-world T1D management and more broadly any tasks requiring rapid and precise preference learning under safety constraints. 

**Abstract (ZH)**: 强化学习（RL）在模拟1型糖尿病（T1D）患者的胰岛素剂量自动化方面取得了成功，但目前仍无法整合患者的专业知识和偏好。本文介绍了一种新的强化学习框架PAINT（患者偏好适应下的胰岛素控制在1型糖尿病中），用于从患者记录中学习灵活的胰岛素剂量策略。PAINT采用草图基础的方法进行奖励学习，通过为过去的数据添加连续的奖励信号来反映患者期望的结果。带标签的数据训练奖励模型，指导一种新颖的安全约束下的离线RL算法的行动，该算法旨在将行动限制在安全策略中，并通过滑动标度进行偏好调整。通过对模拟数据的评估显示，PAINT仅通过简单标注期望状态即可实现常见血糖目标，相比商业基准，降低15%的血糖风险。行动标签也可用于整合患者的专业知识，结果表明在患者指导下，能够提前应对餐食（餐后增益10%的时间内血糖控制良好）和解决某些设备错误（错误后减少1.6%的血糖波动）。这些结果在包括有限样本、标签错误和个体间差异的现实条件下依然成立。这项工作展示了PAINT在实际T1D管理以及更广泛要求在安全约束下快速精确偏好学习的任务中的潜力。 

---
# Are Transformers Able to Reason by Connecting Separated Knowledge in Training Data? 

**Title (ZH)**: Transformer模型能否通过连接训练数据中分离的知识来进行推理？ 

**Authors**: Yutong Yin, Zhaoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15857)  

**Abstract**: Humans exhibit remarkable compositional reasoning by integrating knowledge from various sources. For example, if someone learns ( B = f(A) ) from one source and ( C = g(B) ) from another, they can deduce ( C=g(B)=g(f(A)) ) even without encountering ( ABC ) together, showcasing the generalization ability of human intelligence. In this paper, we introduce a synthetic learning task, "FTCT" (Fragmented at Training, Chained at Testing), to validate the potential of Transformers in replicating this skill and interpret its inner mechanism. In the training phase, data consist of separated knowledge fragments from an overall causal graph. During testing, Transformers must infer complete causal graph traces by integrating these fragments. Our findings demonstrate that few-shot Chain-of-Thought prompting enables Transformers to perform compositional reasoning on FTCT by revealing correct combinations of fragments, even if such combinations were absent in the training data. Furthermore, the emergence of compositional reasoning ability is strongly correlated with the model complexity and training-testing data similarity. We propose, both theoretically and empirically, that Transformers learn an underlying generalizable program from training, enabling effective compositional reasoning during testing. 

**Abstract (ZH)**: 人类通过对来自不同来源的知识进行整合展示了惊人的组合推理能力。例如，如果某人从一个来源学习到（B = f(A)），从另一个来源学习到（C = g(B)），即使没有同时遇到ABC，他们仍能推导出（C = g(B) = g(f(A))），这展示了人类智能的泛化能力。在本文中，我们引入了一个合成学习任务“FTCT”（Training Fragmented at Chaining Testing），以验证Transformer在复制这一技能方面的潜力，并解析其内在机制。在训练阶段，数据由整体因果图中的分离知识片段组成。在测试阶段，Transformer必须通过整合这些片段来推断完整的因果图轨迹。我们的研究发现，少量样本的链式推理提示能够使Transformer在FTCT中进行组合推理，通过揭示正确的片段组合，即使这些组合在训练数据中不存在也是如此。此外，组合推理能力的出现与模型复杂度和训练测试数据的相似性密切相关。我们从理论上和实验上提出，Transformer通过训练学习到一个潜在可泛化的程序，在测试期间实现有效的组合推理。 

---
# Harnessing Diverse Perspectives: A Multi-Agent Framework for Enhanced Error Detection in Knowledge Graphs 

**Title (ZH)**: 利用多元视角：一种增强知识图谱错误检测的多代理框架 

**Authors**: Yu Li, Yi Huang, Guilin Qi, Junlan Feng, Nan Hu, Songlin Zhai, Haohan Xue, Yongrui Chen, Ruoyan Shen, Tongtong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15791)  

**Abstract**: Knowledge graphs are widely used in industrial applications, making error detection crucial for ensuring the reliability of downstream applications. Existing error detection methods often fail to effectively leverage fine-grained subgraph information and rely solely on fixed graph structures, while also lacking transparency in their decision-making processes, which results in suboptimal detection performance. In this paper, we propose a novel Multi-Agent framework for Knowledge Graph Error Detection (MAKGED) that utilizes multiple large language models (LLMs) in a collaborative setting. By concatenating fine-grained, bidirectional subgraph embeddings with LLM-based query embeddings during training, our framework integrates these representations to produce four specialized agents. These agents utilize subgraph information from different dimensions to engage in multi-round discussions, thereby improving error detection accuracy and ensuring a transparent decision-making process. Extensive experiments on FB15K and WN18RR demonstrate that MAKGED outperforms state-of-the-art methods, enhancing the accuracy and robustness of KG evaluation. For specific industrial scenarios, our framework can facilitate the training of specialized agents using domain-specific knowledge graphs for error detection, which highlights the potential industrial application value of our framework. Our code and datasets are available at this https URL. 

**Abstract (ZH)**: 知识图谱在工业应用中广泛应用，因此错误检测对于确保下游应用的可靠性至关重要。现有错误检测方法往往未能有效利用细粒度的子图信息，而是依赖于固定的图结构，同时在决策过程中缺乏透明性，导致检测性能不佳。本文提出了一种名为多代理框架的知识图谱错误检测（MAKGED）方法，该方法利用多个大型语言模型（LLMs）在协作环境中进行工作。通过在训练过程中将细粒度的双向子图嵌入与基于LLM的查询嵌入进行连接，该框架将这些表示整合成四种专门的代理。这些代理利用来自不同维度的子图信息进行多轮讨论，从而提高错误检测精度并确保决策过程的透明性。在FB15K和WN18RR上的 extensive 实验表明，MAKGED 在错误检测精度和鲁棒性方面优于最先进的方法。对于特定的工业场景，该框架可以根据专业知识图谱训练专门的代理进行错误检测，突显了该框架的潜在工业应用价值。我们的代码和数据集可以在此处访问：<此网址>。 

---
# LLM-powered Multi-agent Framework for Goal-oriented Learning in Intelligent Tutoring System 

**Title (ZH)**: 基于大型语言模型的多智能体框架：面向目标学习的智能辅导系统 

**Authors**: Tianfu Wang, Yi Zhan, Jianxun Lian, Zhengyu Hu, Nicholas Jing Yuan, Qi Zhang, Xing Xie, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.15749)  

**Abstract**: Intelligent Tutoring Systems (ITSs) have revolutionized education by offering personalized learning experiences. However, as goal-oriented learning, which emphasizes efficiently achieving specific objectives, becomes increasingly important in professional contexts, existing ITSs often struggle to deliver this type of targeted learning experience. In this paper, we propose GenMentor, an LLM-powered multi-agent framework designed to deliver goal-oriented, personalized learning within ITS. GenMentor begins by accurately mapping learners' goals to required skills using a fine-tuned LLM trained on a custom goal-to-skill dataset. After identifying the skill gap, it schedules an efficient learning path using an evolving optimization approach, driven by a comprehensive and dynamic profile of learners' multifaceted status. Additionally, GenMentor tailors learning content with an exploration-drafting-integration mechanism to align with individual learner needs. Extensive automated and human evaluations demonstrate GenMentor's effectiveness in learning guidance and content quality. Furthermore, we have deployed it in practice and also implemented it as an application. Practical human study with professional learners further highlights its effectiveness in goal alignment and resource targeting, leading to enhanced personalization. Supplementary resources are available at this https URL. 

**Abstract (ZH)**: 智能辅导系统（ITSs）通过提供个性化的学习体验，彻底改变了教育领域。然而，随着目标导向学习——强调高效达成具体目标——在专业环境中的重要性日益提高，现有的ITSs往往难以提供这种针对性的学习体验。本文提出了一种名为GenMentor的框架，该框架由大语言模型（LLM）驱动，旨在为ITS提供目标导向的个性化学习体验。GenMentor首先通过微调在自定义目标到技能数据集上训练的LLM，准确地将学习者的目标与所需的技能相匹配。在识别出技能缺口后，它利用一种不断演化的优化方法，根据学习者多维度状态的全面且动态的概况，规划一条高效的学习路径。此外，GenMentor采用了探索-草拟-整合机制来定制学习内容，以满足个体学习者的需求。广泛的自动化和人工评估表明，GenMentor在学习指导和内容质量方面具有显著效果。此外，我们已在实践中部署了GenMentor，并将其作为应用程序进行实施。对专业学习者的实际人类研究进一步突显了其在目标对齐和资源分配方面的有效性，从而增强了个性化水平。有关补充资源，请参阅此链接：[提供的链接]。 

---
# Propositional Interpretability in Artificial Intelligence 

**Title (ZH)**: 人工智能中的命题可解释性 

**Authors**: David J. Chalmers  

**Link**: [PDF](https://arxiv.org/pdf/2501.15740)  

**Abstract**: Mechanistic interpretability is the program of explaining what AI systems are doing in terms of their internal mechanisms. I analyze some aspects of the program, along with setting out some concrete challenges and assessing progress to date. I argue for the importance of propositional interpretability, which involves interpreting a system's mechanisms and behavior in terms of propositional attitudes: attitudes (such as belief, desire, or subjective probability) to propositions (e.g. the proposition that it is hot outside). Propositional attitudes are the central way that we interpret and explain human beings and they are likely to be central in AI too. A central challenge is what I call thought logging: creating systems that log all of the relevant propositional attitudes in an AI system over time. I examine currently popular methods of interpretability (such as probing, sparse auto-encoders, and chain of thought methods) as well as philosophical methods of interpretation (including those grounded in psychosemantics) to assess their strengths and weaknesses as methods of propositional interpretability. 

**Abstract (ZH)**: 机制可解释性是指通过分析和解释AI系统内部机制来说明它们在做什么的过程。本文分析了这一程序的一些方面，并概述了一些具体的挑战，同时评估了迄今为止取得的进展。我强调了命题可解释性的重要性，即以命题态度（如信念、欲望或主观概率）来解释系统机制和行为。命题态度是我们解释和理解人类行为的核心方式，很可能也是AI中的核心方式。一个核心挑战是我称之为思想日志的问题：创建可以记录AI系统中所有相关命题态度的系统。本文还探讨了当前流行的可解释性方法（如探针、稀疏自编码器和思维链方法）以及基于心理语义学的哲学解释方法，评估它们作为命题可解释性方法的优缺点。 

---
# Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning 

**Title (ZH)**: 重新思考外部慢思考：从雪球错误到正确推理的概率 

**Authors**: Zeyu Gan, Yun Liao, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15602)  

**Abstract**: Test-time scaling, which is also often referred to as \textit{slow-thinking}, has been demonstrated to enhance multi-step reasoning in large language models (LLMs). However, despite its widespread utilization, the mechanisms underlying slow-thinking methods remain poorly understood. This paper explores the mechanisms of external slow-thinking from a theoretical standpoint. We begin by examining the snowball error effect within the LLM reasoning process and connect it to the likelihood of correct reasoning using information theory. Building on this, we show that external slow-thinking methods can be interpreted as strategies to mitigate the error probability. We further provide a comparative analysis of popular external slow-thinking approaches, ranging from simple to complex, highlighting their differences and interrelationships. Our findings suggest that the efficacy of these methods is not primarily determined by the specific framework employed, and that expanding the search scope or the model's internal reasoning capacity may yield more sustained improvements in the long term. We open-source our code at \url{this https URL}. 

**Abstract (ZH)**: 测试时缩放（也常被称为“慢思考”），已被证明可以增强大型语言模型（LLMs）的多步推理能力。然而，尽管慢思考方法被广泛应用于实践中，其背后的机制仍然不够清楚。本文从理论角度探讨了外部慢思考的机制。我们首先研究了LLM推理过程中的雪球效应，并通过信息论将其与正确推理的概率联系起来。在此基础上，我们表明外部慢思考方法可以被理解为降低错误概率的策略。此外，我们还对流行的外部慢思考方法进行了比较分析，范围从简单的策略到复杂的策略，并强调了它们之间的差异和相互关系。研究结果表明，这些方法的有效性并不主要取决于所采用的具体框架，而是更有可能通过扩大搜索范围或提升模型内部推理能力来实现长期的持续改进。我们的代码已开源，可在 \url{此链接} 找到。 

---
# Expert-Free Online Transfer Learning in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 专家无需参与的多智能体强化学习中的在线转移学习 

**Authors**: Alberto Castagna  

**Link**: [PDF](https://arxiv.org/pdf/2501.15495)  

**Abstract**: Reinforcement Learning (RL) enables an intelligent agent to optimise its performance in a task by continuously taking action from an observed state and receiving a feedback from the environment in form of rewards. RL typically uses tables or linear approximators to map state-action tuples that maximises the reward. Combining RL with deep neural networks (DRL) significantly increases its scalability and enables it to address more complex problems than before. However, DRL also inherits downsides from both RL and deep learning. Despite DRL improves generalisation across similar state-action pairs when compared to simpler RL policy representations like tabular methods, it still requires the agent to adequately explore the state-action space. Additionally, deep methods require more training data, with the volume of data escalating with the complexity and size of the neural network. As a result, deep RL requires a long time to collect enough agent-environment samples and to successfully learn the underlying policy. Furthermore, often even a slight alteration to the task invalidates any previous acquired knowledge. To address these shortcomings, Transfer Learning (TL) has been introduced, which enables the use of external knowledge from other tasks or agents to enhance a learning process. The goal of TL is to reduce the learning complexity for an agent dealing with an unfamiliar task by simplifying the exploration process. This is achieved by lowering the amount of new information required by its learning model, resulting in a reduced overall convergence time... 

**Abstract (ZH)**: 强化学习（RL）使智能代理能够通过不断从观察到的状态采取行动并从环境中获得反馈（奖励）来优化其在任务中的表现。RL 通常使用表格或线性近似来映射能最大化奖励的状态-动作元组。将 RL 与深度神经网络相结合（深度强化学习，DRL）极大地提高了其可扩展性，并使其能够解决比以往更复杂的问题。然而，DRL 也从 RL 和深度学习中继承了一些缺点。尽管 DRL 在面对相似的状态-动作对时与简单 RL 策略表示（如表征方法）相比在泛化方面有所改进，但它仍然需要智能代理充分探索状态-动作空间。此外，深度方法需要更多的训练数据，数据量随着神经网络的复杂性和规模而增加。因此，DRL 需要较长时间收集足够的智能代理-环境样本，并成功学习底层策略。此外，往往即使是任务的轻微改动也会使得之前获得的知识失效。为解决这些缺点，引入了元学习（Transfer Learning，TL），它允许利用其他任务或代理的外部知识来增强学习过程。TL 的目标是通过简化探索过程来降低智能代理处理陌生任务的学习复杂性。这通过减少学习模型所需的新信息量来实现，从而降低总体收敛时间…… 

---
# AI in Oncology: Transforming Cancer Detection through Machine Learning and Deep Learning Applications 

**Title (ZH)**: AI在肿瘤学中的应用：通过机器学习和深度学习变革癌症检测 

**Authors**: Muhammad Aftab, Faisal Mehmood, Chengjuan Zhang, Alishba Nadeem, Zigang Dong, Yanan Jiang, Kangdongs Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15489)  

**Abstract**: Artificial intelligence (AI) has potential to revolutionize the field of oncology by enhancing the precision of cancer diagnosis, optimizing treatment strategies, and personalizing therapies for a variety of cancers. This review examines the limitations of conventional diagnostic techniques and explores the transformative role of AI in diagnosing and treating cancers such as lung, breast, colorectal, liver, stomach, esophageal, cervical, thyroid, prostate, and skin cancers. The primary objective of this paper is to highlight the significant advancements that AI algorithms have brought to oncology within the medical industry. By enabling early cancer detection, improving diagnostic accuracy, and facilitating targeted treatment delivery, AI contributes to substantial improvements in patient outcomes. The integration of AI in medical imaging, genomic analysis, and pathology enhances diagnostic precision and introduces a novel, less invasive approach to cancer screening. This not only boosts the effectiveness of medical facilities but also reduces operational costs. The study delves into the application of AI in radiomics for detailed cancer characterization, predictive analytics for identifying associated risks, and the development of algorithm-driven robots for immediate diagnosis. Furthermore, it investigates the impact of AI on addressing healthcare challenges, particularly in underserved and remote regions. The overarching goal of this platform is to support the development of expert recommendations and to provide universal, efficient diagnostic procedures. By reviewing existing research and clinical studies, this paper underscores the pivotal role of AI in improving the overall cancer care system. It emphasizes how AI-enabled systems can enhance clinical decision-making and expand treatment options, thereby underscoring the importance of AI in advancing precision oncology 

**Abstract (ZH)**: 人工智能（AI）有望通过提高癌症诊断的精确性、优化治疗策略和个性化各种癌症的治疗方法，从而变革癌症领域。本文回顾了传统诊断技术的局限性，并探讨了AI在诊断和治疗肺癌、乳腺癌、结直肠癌、肝癌、胃癌、食管癌、宫颈癌、甲状腺癌、前列腺癌和皮肤癌等方面所发挥的变革作用。本文的主要目标是强调AI算法在医疗行业中为癌症研究带来的重大进展。通过实现早期癌症检测、提高诊断准确性以及促进精准治疗，AI有助于显著提高患者的治疗效果。AI在医学成像、基因组分析和病理学中的集成，提高了诊断的准确性，并引入了一种更为微创的癌症筛查方法，这不仅提升了医疗机构的有效性，还降低了运营成本。本文深入探讨了AI在放射组学中的应用，用于详细描述癌症特征、预测分析以识别相关风险以及开发算法驱动的机器人以实现即时诊断。此外，本文还调查了AI在应对医疗保健挑战中的作用，特别是对贫困和偏远地区的影响。该平台的整体目标是支持专家建议的发展，并提供广泛、高效的诊断程序。通过回顾现有的研究和临床研究，本文强调了AI在改善整体癌症护理系统中的关键作用。本文强调了AI驱动系统如何增强临床决策并扩展治疗选项，突显了AI在推动精准肿瘤学发展中的重要性。 

---
# A Neurosymbolic Framework for Geometric Reduction of Binary Forms 

**Title (ZH)**: 一个神经符号框架，用于二元形式的几何简化 

**Authors**: Ilias Kotsireas, Tony Shaska  

**Link**: [PDF](https://arxiv.org/pdf/2501.15404)  

**Abstract**: This paper compares Julia reduction and hyperbolic reduction with the aim of finding equivalent binary forms with minimal coefficients. We demonstrate that hyperbolic reduction generally outperforms Julia reduction, particularly in the cases of sextics and decimics, though neither method guarantees achieving the minimal form. We further propose an additional shift and scaling to approximate the minimal form more closely. Finally, we introduce a machine learning framework to identify optimal transformations that minimize the heights of binary forms. This study provides new insights into the geometry and algebra of binary forms and highlights the potential of AI in advancing symbolic computation and reduction techniques. The findings, supported by extensive computational experiments, lay the groundwork for hybrid approaches that integrate traditional reduction methods with data-driven techniques. 

**Abstract (ZH)**: 本文将Julia减法与双曲减法进行比较，旨在找到具有最小系数的等价二元形式。我们展示了双曲减法通常优于Julia减法，尤其是在六次和十次形式的情况下，但两种方法均不能保证达到最小形式。我们进一步提出了一种额外的移动和缩放方法，以更接近最小形式。最后，我们引入了一种机器学习框架，用于识别可最小化二元形式高度的最佳变换。本研究不仅为二元形式的几何和代数提供了新的见解，还突显了人工智能在推进符号计算和减少技术方面的潜力。支持广泛计算实验的研究发现为将传统减少方法与数据驱动技术相结合的混合方法奠定了基础。 

---
# Diffusion-based Hierarchical Negative Sampling for Multimodal Knowledge Graph Completion 

**Title (ZH)**: 基于扩散的层级负采样方法及其在多模态知识图完成中的应用 

**Authors**: Guanglin Niu, Xiaowei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15393)  

**Abstract**: Multimodal Knowledge Graph Completion (MMKGC) aims to address the critical issue of missing knowledge in multimodal knowledge graphs (MMKGs) for their better applications. However, both the previous MMGKC and negative sampling (NS) approaches ignore the employment of multimodal information to generate diverse and high-quality negative triples from various semantic levels and hardness levels, thereby limiting the effectiveness of training MMKGC models. Thus, we propose a novel Diffusion-based Hierarchical Negative Sampling (DHNS) scheme tailored for MMKGC tasks, which tackles the challenge of generating high-quality negative triples by leveraging a Diffusion-based Hierarchical Embedding Generation (DiffHEG) that progressively conditions on entities and relations as well as multimodal semantics. Furthermore, we develop a Negative Triple-Adaptive Training (NTAT) strategy that dynamically adjusts training margins associated with the hardness level of the synthesized negative triples, facilitating a more robust and effective learning procedure to distinguish between positive and negative triples. Extensive experiments on three MMKGC benchmark datasets demonstrate that our framework outperforms several state-of-the-art MMKGC models and negative sampling techniques, illustrating the effectiveness of our DHNS for training MMKGC models. The source codes and datasets of this paper are available at this https URL. 

**Abstract (ZH)**: 多模态知识图谱补全（MMKGC）旨在解决多模态知识图谱（MMKG）中缺失知识的关键问题，以更好地应用于实际场景。然而，现有的MMKGC和负采样（NS）方法在生成多样性和高质量的负三元组时忽略了对多模态信息的利用，这些负三元组涵盖了不同语义级别和难度级别，从而限制了MMKGC模型训练的有效性。因此，我们提出了一种专为MMKGC任务设计的新颖扩散基础层次负采样（DHNS）方案，通过利用基于扩散的基础层次嵌入生成（DiffHEG）机制逐步条件化实体、关系以及多模态语义，来应对生成高质量负三元组的挑战。此外，我们开发了一种负三元组自适应训练（NTAT）策略，该策略动态调整与合成负三元组难度级别相关的训练间隔，促进一个更稳健和有效的学习过程来区分正负三元组。在三个MMKGC基准数据集上的广泛实验表明，我们的框架优于多种最先进的MMKGC模型和负采样技术，证明了DHNS在训练MMKGC模型中的有效性。本文的源代码和数据集可在以下链接获取：[链接]。 

---
# How to Mitigate Information Loss in Knowledge Graphs for GraphRAG: Leveraging Triple Context Restoration and Query-Driven Feedback 

**Title (ZH)**: 如何在GraphRAG中的知识图谱中减轻信息损失：利用三元组上下文恢复和查询驱动的反馈 

**Authors**: Manzong Huang, Chenyang Bu, Yi He, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15378)  

**Abstract**: Knowledge Graph (KG)-augmented Large Language Models (LLMs) have recently propelled significant advances in complex reasoning tasks, thanks to their broad domain knowledge and contextual awareness. Unfortunately, current methods often assume KGs to be complete, which is impractical given the inherent limitations of KG construction and the potential loss of contextual cues when converting unstructured text into entity-relation triples. In response, this paper proposes the Triple Context Restoration and Query-driven Feedback (TCR-QF) framework, which reconstructs the textual context underlying each triple to mitigate information loss, while dynamically refining the KG structure by iteratively incorporating query-relevant missing knowledge. Experiments on five benchmark question-answering datasets substantiate the effectiveness of TCR-QF in KG and LLM integration, where itachieves a 29.1% improvement in Exact Match and a 15.5% improvement in F1 over its state-of-the-art GraphRAG competitors. 

**Abstract (ZH)**: 增强知识图谱（KG）的大型语言模型（LLMs）最近在复杂的推理任务中取得了显著进展，这得益于它们广泛的知识领域和上下文意识。然而，当前的方法常常假设知识图谱是完整的，这在知识图谱构建的固有限制以及从非结构化文本转换为实体-关系三元组时可能丢失上下文线索的情况下是不切实际的。为此，本文提出了一种三元组上下文恢复和查询驱动反馈（TCR-QF）框架。该框架重建每个三元组背后的文本上下文，以减轻信息丢失，并通过迭代地结合查询相关的缺失知识来动态优化知识图谱结构。在五个基准问答数据集上的实验验证了TCR-QF在知识图谱和大型语言模型集成中的有效性，相对于最先进的GraphRAG竞争对手，它在准确匹配（Exact Match）上提高了29.1%，在F1分数上提高了15.5%。 

---
# Who's Driving? Game Theoretic Path Risk of AGI Development 

**Title (ZH)**: 《谁在掌舵？关于AGI发展路径风险的博弈论分析》

这个标题翻译成中文后，既保留了原文的意思，又符合学术规范。其中，“Who's Driving?”可以翻译为“谁在掌舵？”形象地表达了对控制AGI发展方向问题的探讨。“Game Theoretic Path Risk of AGI Development”则翻译为“关于AGI发展路径风险的博弈论分析”，准确地传达了原文的核心概念。 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2501.15280)  

**Abstract**: Who controls the development of Artificial General Intelligence (AGI) might matter less than how we handle the fight for control itself. We formalize this "steering wheel problem" as humanity's greatest near-term existential risk may stem not from misaligned AGI, but from the dynamics of competing to develop it. Just as a car crash can occur from passengers fighting over the wheel before reaching any destination, catastrophic outcomes could arise from development competition long before AGI exists. While technical alignment research focuses on ensuring safe arrival, we show how coordination failures during development could drive us off the cliff first.
We present a game theoretic framework modeling AGI development dynamics and prove conditions for sustainable cooperative equilibria. Drawing from nuclear control while accounting for AGI's unique characteristics, we propose concrete mechanisms including pre-registration, shared technical infrastructure, and automated deterrence to stabilize cooperation. Our key insight is that AGI creates network effects in safety: shared investments become more valuable as participation grows, enabling mechanism designs where cooperation dominates defection. This work bridges formal methodology and policy frameworks, providing foundations for practical governance of AGI competition risks. 

**Abstract (ZH)**: 人工智能通用智能（AGI）的发展控制权可能不如我们如何应对控制权之争本身重要。我们将这一称为“方向盘问题”的现象正式化为，人类近期存在的最大 existential 风险可能不来自于功能错配的 AGI，而是来自开发竞争的动态过程。正如在无人到达目的地前，因乘客争抢方向盘从而引发车祸一样，灾难性后果可能在 AGI 不存在的情况下就已经出现。虽然技术对齐研究集中于确保安全到达，我们展示了在开发过程中协调失败如何首先将我们推向悬崖。

我们提出了一种博弈论框架来建模 AGI 开发动态，并证明了可持续合作均衡状态的条件。借鉴核控制机制，并考虑到 AGI 的独特特性，我们提出了一些具体的机制，包括预注册、共享技术基础设施和自动威慑，以稳定合作。我们的主要见解是，AGI 在安全性方面创造了网络效应：随着参与者的增加，共享投资变得更有价值，从而使得机制设计更有利于合作而非背叛。这项工作将形式化的方法论与政策框架相结合，为 AGI 竞争风险的实际治理提供了基础。 

---
# Abstraction Method for Generalized Planning with Baggable Types 

**Title (ZH)**: 袋类型下的广义规划抽象方法 

**Authors**: Hao Dong, Zheyuan Shi, Hemeng Zeng, Yongmei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15249)  

**Abstract**: Generalized planning is concerned with how to find a single plan to solve multiple similar planning instances. Abstractions are widely used for solving generalized planning, and QNP (qualitative numeric planning) is a popular abstract model. Recently, Cui et al. showed that a plan solves a sound and complete abstraction of a generalized planning problem if and only if the refined plan solves the original problem. However, existing work on automatic abstraction for generalized planning can hardly guarantee soundness let alone completeness. In this paper, we propose an automatic sound and complete abstraction method for generalized planning with baggable types. We use a variant of QNP, called bounded QNP (BQNP), where integer variables are increased or decreased by only one. Since BQNP is undecidable, we propose and implement a sound but incomplete solver for BQNP. We present an automatic method to abstract a BQNP problem from a classical planning instance with baggable types. The basic idea for abstraction is to introduce a counter for each bag of indistinguishable tuples of objects. We define a class of domains called proper baggable domains, and show that for such domains, the BQNP problem got by our automatic method is a sound and complete abstraction for a generalized planning problem whose instances share the same bags with the given instance but the sizes of the bags might be different. Thus, the refined plan of a solution to the BQNP problem is a solution to the generalized planning problem. Finally, we implement our abstraction method and experiments on a number of domains demonstrate the promise of our approach. 

**Abstract (ZH)**: 广义规划关注如何找到一个计划来解决多个类似的规划实例。抽象是解决广义规划问题的广泛使用的手段，而量化数值规划（QNP）是一种流行的抽象模型。最近，Cui等人的研究表明，一个计划解决了某个广义规划问题的正确且完备的抽象当且仅当细化后的计划解决了原始问题。然而，现有的广义规划自动抽象工作几乎无法保证正确性，更不用说完备性了。在本文中，我们提出了一种适用于可打包类型广义规划的自动正确且完备的抽象方法。我们使用一种变体的QNP，称为有界量化数值规划（BQNP），其中整数变量仅增加或减少一个。由于BQNP不可判定，我们提出并实现了一个针对BQNP的正确但不完备的求解器。我们提出了一种自动方法，从具有可打包类型的经典规划实例中抽象出BQNP问题。抽象的基本思想是为每个不可区分的对象元组集合引入一个计数器。我们定义了一类称为适当可打包的领域，并证明对于这些领域，通过我们提出的自动方法得到的BQNP问题是与给定实例具有相同包但包的大小可能不同的广义规划问题的正确且完备的抽象。因此，BQNP问题的细化计划也是广义规划问题的解。最后，我们实现了一种抽象方法，并在多个领域的实验表明了我们方法的潜力。 

---
# A Causality-aware Paradigm for Evaluating Creativity of Multimodal Large Language Models 

**Title (ZH)**: 具有因果意识的范式，用于评估多模态大型语言模型的创造力 

**Authors**: Zhongzhan Huang, Shanshan Zhong, Pan Zhou, Shanghua Gao, Marinka Zitnik, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.15147)  

**Abstract**: Recently, numerous benchmarks have been developed to evaluate the logical reasoning abilities of large language models (LLMs). However, assessing the equally important creative capabilities of LLMs is challenging due to the subjective, diverse, and data-scarce nature of creativity, especially in multimodal scenarios. In this paper, we consider the comprehensive pipeline for evaluating the creativity of multimodal LLMs, with a focus on suitable evaluation platforms and methodologies. First, we find the Oogiri game, a creativity-driven task requiring humor, associative thinking, and the ability to produce unexpected responses to text, images, or both. This game aligns well with the input-output structure of modern multimodal LLMs and benefits from a rich repository of high-quality, human-annotated creative responses, making it an ideal platform for studying LLM creativity. Next, beyond using the Oogiri game for standard evaluations like ranking and selection, we propose LoTbench, an interactive, causality-aware evaluation framework, to further address some intrinsic risks in standard evaluations, such as information leakage and limited interpretability. The proposed LoTbench not only quantifies LLM creativity more effectively but also visualizes the underlying creative thought processes. Our results show that while most LLMs exhibit constrained creativity, the performance gap between LLMs and humans is not insurmountable. Furthermore, we observe a strong correlation between results from the multimodal cognition benchmark MMMU and LoTbench, but only a weak connection with traditional creativity metrics. This suggests that LoTbench better aligns with human cognitive theories, highlighting cognition as a critical foundation in the early stages of creativity and enabling the bridging of diverse concepts. this https URL 

**Abstract (ZH)**: 近年来，為了評估大型語言模型（LLMs）的邏輯推理能力，已經開發了大量評估基准。然而，由於創意的主觀性、多樣性和數據匱乏性，尤其是多模態場景下的特性，評估LLMs的同等重要的創意能力仍然具有挑戰。本文我們考慮了一套全面的評估管道，用於評估多模態LLMs的創意能力，並重點討論了合適的評估平台和方法。首先，我們發現了一個基於創意的Oogiri遊戲，這個遊戲需要幽默感、聯想思考，以及對文字、圖像或兩者的非預期回應生产能力。這個遊戲與現代多模態LLMs的輸入-輸出結構非常契合，並且得益於一個豐富的高質量、人標注的創意回應數據庫，使其成為研究LLMs創意的理想平台。其次是超越使用Oogiri遊戲進行標準評估（如排名和選擇），我們提出了LoTbench，一種交互式的、知因果性感知的評估框架，旨在進一步解決標準評估中的固有風險，例如信息洩露和有限的可解釋性。所提出的LoTbench不僅更有效地量化了LLMs的創意能力，還可视化了其背後的理念產生過程。結果表明，雖然大多數LLMs展示出了局限性的創意能力，但人類和LLMs之間的表現差距並非無法逾越。此外，我們觀察到多模態認知基准MMMU和LoTbench的結果之間存在密切的相關性，而與傳統創意指標的連接則較弱。這表明LoTbench更符合人類認知理論，強調認知在創意初步階段的關鍵基礎作用，並促進了多樣概念之間的橋接。

更多內容請訪問：[原文链接] 

---
# A New Approach for Knowledge Generation Using Active Inference 

**Title (ZH)**: 使用主动推断的新方法进行知识生成 

**Authors**: Jamshid Ghasimi, Nazanin Movarraei  

**Link**: [PDF](https://arxiv.org/pdf/2501.15105)  

**Abstract**: There are various models proposed on how knowledge is generated in the human brain including the semantic networks model. Although this model has been widely studied and even computational models are presented, but, due to various limits and inefficiencies in the generation of different types of knowledge, its application is limited to semantic knowledge because of has been formed according to semantic memory and declarative knowledge and has many limits in explaining various procedural and conditional knowledge. Given the importance of providing an appropriate model for knowledge generation, especially in the areas of improving human cognitive functions or building intelligent machines, improving existing models in knowledge generation or providing more comprehensive models is of great importance. In the current study, based on the free energy principle of the brain, is the researchers proposed a model for generating three types of declarative, procedural, and conditional knowledge. While explaining different types of knowledge, this model is capable to compute and generate concepts from stimuli based on probabilistic mathematics and the action-perception process (active inference). The proposed model is unsupervised learning that can update itself using a combination of different stimuli as a generative model can generate new concepts of unsupervised received stimuli. In this model, the active inference process is used in the generation of procedural and conditional knowledge and the perception process is used to generate declarative knowledge. 

**Abstract (ZH)**: 在人类大脑中知识生成的诸多模型中，包括了语义网络模型。尽管这种模型已经被广泛研究，并且甚至提出了计算模型来模拟知识生成过程，但由于不同种类的知识生成过程中存在各种局限性和效率问题，这一模型的应用主要局限于语义知识，因为语义记忆和显性知识构建过程中存在很多解释程序性和条件性知识的局限。鉴于为知识生成提供合适模型的重要性和尤其是在提高人类认知功能或构建智能机器方面的迫切需求，改进现有的知识生成模型或是提供更全面的模型具有重要意义。本研究基于大脑的自由能量原理，提出了一个可以生成三种类型知识（显性、程序性和条件性知识）的模型。该模型能够利用概率数学和行动感知过程（主动推断）从刺激中计算并生成概念，同时该模型是一种无监督学习，可以通过组合不同类型的刺激作为生成模型来生成新的无监督接收的刺激的概念。在这个模型中，程序性和条件性知识的生成过程使用了主动推断过程，而知觉过程则用于生成显性知识。 

---
# Data Center Cooling System Optimization Using Offline Reinforcement Learning 

**Title (ZH)**: 使用离线强化学习优化数据中心冷却系统 

**Authors**: Xianyuan Zhan, Xiangyu Zhu, Peng Cheng, Xiao Hu, Ziteng He, Hanfei Geng, Jichao Leng, Huiwen Zheng, Chenhui Liu, Tianshun Hong, Yan Liang, Yunxin Liu, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15085)  

**Abstract**: The recent advances in information technology and artificial intelligence have fueled a rapid expansion of the data center (DC) industry worldwide, accompanied by an immense appetite for electricity to power the DCs. In a typical DC, around 30~40% of the energy is spent on the cooling system rather than on computer servers, posing a pressing need for developing new energy-saving optimization technologies for DC cooling systems. However, optimizing such real-world industrial systems faces numerous challenges, including but not limited to a lack of reliable simulation environments, limited historical data, and stringent safety and control robustness requirements. In this work, we present a novel physics-informed offline reinforcement learning (RL) framework for energy efficiency optimization of DC cooling systems. The proposed framework models the complex dynamical patterns and physical dependencies inside a server room using a purposely designed graph neural network architecture that is compliant with the fundamental time-reversal symmetry. Because of its well-behaved and generalizable state-action representations, the model enables sample-efficient and robust latent space offline policy learning using limited real-world operational data. Our framework has been successfully deployed and verified in a large-scale production DC for closed-loop control of its air-cooling units (ACUs). We conducted a total of 2000 hours of short and long-term experiments in the production DC environment. The results show that our method achieves 14~21% energy savings in the DC cooling system, without any violation of the safety or operational constraints. Our results have demonstrated the significant potential of offline RL in solving a broad range of data-limited, safety-critical real-world industrial control problems. 

**Abstract (ZH)**: 近年来，信息技术和人工智能的迅猛发展促进了全球数据中心（DC）行业的快速增长，伴随而来的是对电力的巨大需求。在典型的数据中心中，大约有30%到40%的能源消耗是用于冷却系统，而不是用于计算机服务器，这迫切需要开发新的节能优化技术来改进DC冷却系统。然而，优化这类实际工业系统面临着诸多挑战，包括但不限于缺乏可靠的仿真环境、历史数据有限，以及严格的安全性和控制鲁棒性要求。在此项工作中，我们提出了一个新颖的基于物理信息的离线强化学习（Reinforcement Learning, RL）框架，用于优化DC冷却系统的能源效率。我们提出的框架利用一个特别设计的图神经网络架构，该架构符合基本的时间反演对称性，来建模服务器机房内的复杂动态模式和物理依赖关系。由于其良好且具有泛化性的状态-动作表示，该模型能够利用有限的实际操作数据实现高效的离线策略学习。我们的框架已经在大型生产数据中心中部署并得到了验证，用于闭环控制其空气冷却单元（ACUs）。我们总共在生产环境中的实验时间长达2000小时，涵盖了短期和长期的实验。结果显示，我们的方法在DC冷却系统中实现了14%到21%的能源节约，没有违反任何安全或操作约束。我们的结果证明了离线RL在解决大量数据受限、安全性要求高的实际工业控制问题中的巨大潜力。 

---
# Feedback-Aware Monte Carlo Tree Search for Efficient Information Seeking in Goal-Oriented Conversations 

**Title (ZH)**: 面向反馈的蒙特卡洛树搜索方法，用于目标导向对话中的高效信息检索 

**Authors**: Harshita Chopra, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2501.15056)  

**Abstract**: The ability to identify and acquire missing information is a critical component of effective decision making and problem solving. With the rise of conversational artificial intelligence (AI) systems, strategically formulating information-seeking questions becomes crucial and demands efficient methods to guide the search process. We introduce a novel approach to adaptive question-asking through a combination of Large Language Models (LLM) for generating questions that maximize information gain, Monte Carlo Tree Search (MCTS) for constructing and leveraging a decision tree across multiple samples, and a hierarchical feedback mechanism to learn from past interactions. We present two key innovations: (1) an adaptive MCTS algorithm that balances exploration and exploitation for efficient search over potential questions; and (2) a clustering-based feedback algorithm that leverages prior experience to guide future interactions. Each incoming sample is assigned to a cluster based on its semantic similarity with previously observed samples. Our UCT (Upper Confidence bound for Trees) formulation selects optimal questions by combining expected rewards, a function of information gain, with a cluster-specific bonus that decays with depth, to emphasize the importance of early-stage questions that have proven effective for narrowing the solution space in similar samples. Experiments across three domains, including medical diagnosis and troubleshooting, demonstrate that our method leads to an average of 12% improvement in success rates and a 10x reduction in the average number of LLM calls made per conversation for the search process, in comparison to the state of the art. 

**Abstract (ZH)**: 识别和获取缺失信息的能力是有效决策和问题解决的关键组成部分。随着对话型人工智能（AI）系统的发展，策略性地制定信息查询问题变得尤为重要，并且需要高效的方法来指导搜索过程。我们提出了一种新的自适应提问方法，结合了大规模语言模型（LLM）生成最大化信息增益的问题、蒙特卡洛树搜索（MCTS）构建和利用跨多样本的决策树，以及层次反馈机制从以往交互中学习。我们提出了两个关键创新：（1）一种平衡探索和利用的自适应MCTS算法，以实现有效的潜在问题搜索；（2）基于聚类的反馈算法，利用先前的经验指导未来的交互。每个新样本基于其与之前观察到的样本的语义相似性被分配到一个簇中。我们的UCT（树上限置信界）公式通过结合预期奖励（信息增益的一个函数）与随深度衰减的簇特定奖励来选择最优问题，从而强调早期阶段具有证明有效减少解空间的问题的重要性。在医学诊断、故障排查等三个领域进行的实验表明，与现有技术相比，我们的方法在成功率上平均提高了12%，并且每个对话中的LLM调用次数减少了10倍，这在搜索过程中显示出显著的优势。 

---
# Controllable Protein Sequence Generation with LLM Preference Optimization 

**Title (ZH)**: 基于LLM偏爱优化的可控蛋白质序列生成 

**Authors**: Xiangyu Liu, Yi Liu, Silei Chen, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15007)  

**Abstract**: Designing proteins with specific attributes offers an important solution to address biomedical challenges. Pre-trained protein large language models (LLMs) have shown promising results on protein sequence generation. However, to control sequence generation for specific attributes, existing work still exhibits poor functionality and structural stability. In this paper, we propose a novel controllable protein design method called CtrlProt. We finetune a protein LLM with a new multi-listwise preference optimization strategy to improve generation quality and support multi-attribute controllable generation. Experiments demonstrate that CtrlProt can meet functionality and structural stability requirements effectively, achieving state-of-the-art performance in both single-attribute and multi-attribute protein sequence generation. 

**Abstract (ZH)**: 设计具有特定属性的蛋白质为应对生物医药挑战提供了重要的解决方案。预训练的蛋白质大规模语言模型（LLMs）在蛋白质序列生成方面展现了令人鼓舞的结果。然而，现有工作在控制特定属性的序列生成方面仍表现出较差的功能性和结构稳定性。本文中，我们提出了一种新颖的可控蛋白质设计方法——CtrlProt。我们通过一种新的多列表偏好优化策略微调蛋白质LLM，以提高生成质量并支持多属性可控生成。实验结果表明，CtrlProt能够有效地满足功能性和结构稳定性要求，在单属性和多属性蛋白质序列生成方面均达到了最先进的性能。 

---
# What if Eye...? Computationally Recreating Vision Evolution 

**Title (ZH)**: 《如果眼睛...？计算再现视觉进化》

这个标题是一个类似于科幻作品中提问的形式，尝试以科幻的方式探讨视觉进化的计算模拟。在翻译时，尽可能保留原文的疑问性和创新思维，同时使其符合中文的学术表达习惯。 

**Authors**: Kushagra Tiwary, Aaron Young, Zaid Tasneem, Tzofi Klinghoffer, Akshat Dave, Tomaso Poggio, Dan Nilsson, Brian Cheung, Ramesh Raskar  

**Link**: [PDF](https://arxiv.org/pdf/2501.15001)  

**Abstract**: Vision systems in nature show remarkable diversity, from simple light-sensitive patches to complex camera eyes with lenses. While natural selection has produced these eyes through countless mutations over millions of years, they represent just one set of realized evolutionary paths. Testing hypotheses about how environmental pressures shaped eye evolution remains challenging since we cannot experimentally isolate individual factors. Computational evolution offers a way to systematically explore alternative trajectories. Here we show how environmental demands drive three fundamental aspects of visual evolution through an artificial evolution framework that co-evolves both physical eye structure and neural processing in embodied agents. First, we demonstrate computational evidence that task specific selection drives bifurcation in eye evolution - orientation tasks like navigation in a maze leads to distributed compound-type eyes while an object discrimination task leads to the emergence of high-acuity camera-type eyes. Second, we reveal how optical innovations like lenses naturally emerge to resolve fundamental tradeoffs between light collection and spatial precision. Third, we uncover systematic scaling laws between visual acuity and neural processing, showing how task complexity drives coordinated evolution of sensory and computational capabilities. Our work introduces a novel paradigm that illuminates evolutionary principles shaping vision by creating targeted single-player games where embodied agents must simultaneously evolve visual systems and learn complex behaviors. Through our unified genetic encoding framework, these embodied agents serve as next-generation hypothesis testing machines while providing a foundation for designing manufacturable bio-inspired vision systems. 

**Abstract (ZH)**: 自然界中的视觉系统显示出巨大的多样性，从简单的光感受斑点到具有透镜的复杂照相机眼睛。尽管自然选择在过去数百万年中通过无数次的突变产生了这些眼睛，但它们仅代表了一种进化路径。由于我们无法单独隔离环境压力对眼睛进化的各个因素进行实验性研究，因此验证这些假说是具有挑战性的。计算演化为系统地探索替代进化路径提供了一种方法。在这里，我们展示了通过一个综合演化框架，该框架不仅共同演化身体代理的物理眼睛结构和神经处理方式，还能如何通过环境需求驱动视觉进化的三个基本方面。首先，我们展示了计算证据表明，任务特异性选择驱动眼睛进化的分歧——例如在迷宫中的导航任务导致分分布式复合式眼睛的出现，而物体识别任务则导致高分辨率照相机式眼睛的出现。其次，我们揭示了如何通过光学创新（如透镜）自然地解决光线收集与空间精度之间的根本性权衡问题。第三，我们发现了视觉敏锐度与神经处理之间系统的规模律，表明任务复杂性如何驱动感觉能力和计算能力的协调进化。我们的研究引入了一种新的范式，通过创建目标明确的单人游戏来阐明塑造视觉的进化原理，这些在游戏中，身体代理必须同时进化视觉系统并学习复杂行为。通过我们统一的遗传编码框架，这些身体代理成为新一代的假设检验机器，并为设计制造可实现的生物启发式视觉系统提供了基础。 

---
# MISCON: A Mission-Driven Conversational Consultant for Pre-Venture Entrepreneurs in Food Deserts 

**Title (ZH)**: MISCON：面向食品沙漠中创业初期者的任务导向对话咨询师 

**Authors**: Subhasis Dasgupta, Hans Taparia, Laura Schmidt, Amarnath Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2501.14954)  

**Abstract**: This work-in-progress report describes MISCON, a conversational consultant being developed for a public mission project called NOURISH. With MISCON, aspiring small business owners in a food-insecure region and their advisors in Community-based organizations would be able to get information, recommendation and analysis regarding setting up food businesses. MISCON conversations are modeled as state machine that uses a heterogeneous knowledge graph as well as several analytical tools and services including a variety of LLMs. In this short report, we present the functional architecture and some design considerations behind MISCON. 

**Abstract (ZH)**: 本文的工作进展报告描述了为一项名为NOURISH的公共使命项目开发的一款对话式顾问MISCON。通过MISCON，食品不安全地区有意向开设食品企业的创业者及其社区组织顾问能够获得关于设立食品企业的信息、建议和分析。MISCON的对话过程被建模为一个状态机，该状态机利用了一个异质知识图谱以及多种分析工具和服务，包括各种语言大模型（LLM）。在本简要报告中，我们将介绍MISCON的功能架构及其一些设计考虑。 

---
# Causal Graphs Meet Thoughts: Enhancing Complex Reasoning in Graph-Augmented LLMs 

**Title (ZH)**: 因果图结合思维：增强图增强型大语言模型的复杂推理能力 

**Authors**: Hang Luo, Jian Zhang, Chujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14892)  

**Abstract**: In knowledge-intensive tasks, especially in high-stakes domains like medicine and law, it is critical not only to retrieve relevant information but also to provide causal reasoning and explainability. Large language models (LLMs) have achieved remarkable performance in natural language understanding and generation tasks. However, they often suffer from limitations such as difficulty in incorporating new knowledge, generating hallucinations, and explaining their reasoning process. To address these challenges, integrating knowledge graphs with Graph Retrieval-Augmented Generation (Graph RAG) has emerged as an effective solution. Traditional Graph RAG methods often rely on simple graph traversal or semantic similarity, which do not capture causal relationships or align well with the model's internal reasoning steps. This paper proposes a novel pipeline that filters large knowledge graphs to emphasize cause-effect edges, aligns the retrieval process with the model's chain-of-thought (CoT), and enhances reasoning through multi-stage path improvements. Experiments on medical question-answering tasks show consistent gains, with up to a 10\% absolute improvement across multiple large language models (LLMs). This approach demonstrates the value of combining causal reasoning with stepwise retrieval, leading to more interpretable and logically grounded solutions for complex queries. 

**Abstract (ZH)**: 在知识密集型任务中，特别是在医学和法律等高风险领域，不仅需要检索相关的信息，还需要提供因果推理和可解释性。大型语言模型（LLMs）在自然语言理解与生成任务中取得了显著的成果。然而，它们往往存在难以融入新知识、生成虚假信息以及解释推理过程等局限性。为了解决这些挑战，将知识图谱与图检索增强生成（Graph RAG）相结合已成为有效的方法。传统的Graph RAG方法通常依赖于简单的图遍历或语义相似度，无法捕捉因果关系或与模型的内部推理步骤很好地对齐。本文提出了一种新的管道，该管道通过对大规模知识图谱进行过滤以强调因果关系边，将检索过程与模型的链式思维（CoT）对齐，并通过多阶段路径改进来增强推理。在医学问答任务上的实验结果表明，这种方法在多个大型语言模型（LLMs）上表现出了一致的改进，绝对改进幅度最高可达10%。该方法证明了将因果推理与逐步检索相结合的价值，从而使复杂查询的结果更易解释且逻辑基础更牢固。 

---
# Symbolic Knowledge Extraction and Injection with Sub-symbolic Predictors: A Systematic Literature Review 

**Title (ZH)**: 符号知识提取与注入结合亚符号预测器：一项系统文献综述 

**Authors**: Giovanni Ciatto, Federico Sabbatini, Andrea Agiollo, Matteo Magnini, Andrea Omicini  

**Link**: [PDF](https://arxiv.org/pdf/2501.14836)  

**Abstract**: In this paper we focus on the opacity issue of sub-symbolic machine learning predictors by promoting two complementary activities, namely, symbolic knowledge extraction (SKE) and injection (SKI) from and into sub-symbolic predictors. We consider as symbolic any language being intelligible and interpretable for both humans and computers. Accordingly, we propose general meta-models for both SKE and SKI, along with two taxonomies for the classification of SKE and SKI methods. By adopting an explainable artificial intelligence (XAI) perspective, we highlight how such methods can be exploited to mitigate the aforementioned opacity issue. Our taxonomies are attained by surveying and classifying existing methods from the literature, following a systematic approach, and by generalising the results of previous surveys targeting specific sub-topics of either SKE or SKI alone. More precisely, we analyse 132 methods for SKE and 117 methods for SKI, and we categorise them according to their purpose, operation, expected input/output data and predictor types. For each method, we also indicate the presence/lack of runnable software implementations. Our work may be of interest for data scientists aiming at selecting the most adequate SKE/SKI method for their needs, and also work as suggestions for researchers interested in filling the gaps of the current state of the art, as well as for developers willing to implement SKE/SKI-based technologies. 

**Abstract (ZH)**: 本文专注于通过促进两种互补活动来解决子象征机器学习预测器的透明度问题，即从和向子象征预测器进行象征性知识提取（SKE）和注入（SKI）。我们将任意一种语言视为象征性语言，只要这种语言对于人类和计算机都有可读性和可解释性即可。基于此，我们提出了一般性的元模型来支持这两种活动，并为SKE和SKI方法提出了两类分类。从可解释人工智能（XAI）的角度出发，我们强调了如何利用这些方法来缓解上述透明度问题。我们的分类是通过系统地审视和分类现有文献中的方法，针对SKE或SKI的特定子主题进行总结和概括而获得的。更具体地说，我们分析了132种SKE方法和117种SKI方法，并根据它们的目的、操作方式、预期输入/输出数据以及预测器类型将它们进行分类。对于每种方法，我们还指出了是否存在可运行的软件实现。本文的工作可能对旨在选择最适合其需求的SKE/SKI方法的数据科学家具有吸引力，并且也可以作为研究人员填补现有研究前沿空白的建议，以及为希望实现基于SKE/SKI技术的开发人员提供参考。 

---
# RelightVid: Temporal-Consistent Diffusion Model for Video Relighting 

**Title (ZH)**: RelightVid：用于视频重新定向的一致性扩散模型 

**Authors**: Ye Fang, Zeyi Sun, Shangzhan Zhang, Tong Wu, Yinghao Xu, Pan Zhang, Jiaqi Wang, Gordon Wetzstein, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.16330)  

**Abstract**: Diffusion models have demonstrated remarkable success in image generation and editing, with recent advancements enabling albedo-preserving image relighting. However, applying these models to video relighting remains challenging due to the lack of paired video relighting datasets and the high demands for output fidelity and temporal consistency, further complicated by the inherent randomness of diffusion models. To address these challenges, we introduce RelightVid, a flexible framework for video relighting that can accept background video, text prompts, or environment maps as relighting conditions. Trained on in-the-wild videos with carefully designed illumination augmentations and rendered videos under extreme dynamic lighting, RelightVid achieves arbitrary video relighting with high temporal consistency without intrinsic decomposition while preserving the illumination priors of its image backbone. 

**Abstract (ZH)**: 扩散模型在图像生成和编辑中已经取得了显著的成功，并且最近的进步使其能够在保留给定条件的情况下进行图像重新光照。然而，将这些模型应用于视频重新光照仍然具有挑战性，这主要是由于缺乏配对的视频重新光照数据集以及对输出保真度和时间一致性提出的高度要求，进一步加剧了扩散模型固有的随机性。为了解决这些挑战，我们提出了RelightVid，这是一种灵活的视频重新光照框架，可以接受背景视频、文本提示或环境图作为重新光照条件。RelightVid 在自然界中采集的视频数据上进行训练，这些数据经过精心设计的照明增强，并在极端动态照明条件下进行渲染。RelightVid 实现了高时间一致性的任意视频重新光照，同时保留了其图像主干的照明先验知识，而无需进行内在分解。 

---
# sDREAMER: Self-distilled Mixture-of-Modality-Experts Transformer for Automatic Sleep Staging 

**Title (ZH)**: sDREAMER：自我蒸馏多模态专家混合的变压器模型及其在自动睡眠阶段划分中的应用 

**Authors**: Jingyuan Chen, Yuan Yao, Mie Anderson, Natalie Hauglund, Celia Kjaerby, Verena Untiet, Maiken Nedergaard, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16329)  

**Abstract**: Automatic sleep staging based on electroencephalography (EEG) and electromyography (EMG) signals is an important aspect of sleep-related research. Current sleep staging methods suffer from two major drawbacks. First, there are limited information interactions between modalities in the existing methods. Second, current methods do not develop unified models that can handle different sources of input. To address these issues, we propose a novel sleep stage scoring model sDREAMER, which emphasizes cross-modality interaction and per-channel performance. Specifically, we develop a mixture-of-modality-expert (MoME) model with three pathways for EEG, EMG, and mixed signals with partially shared weights. We further propose a self-distillation training scheme for further information interaction across modalities. Our model is trained with multi-channel inputs and can make classifications on either single-channel or multi-channel inputs. Experiments demonstrate that our model outperforms the existing transformer-based sleep scoring methods for multi-channel inference. For single-channel inference, our model also outperforms the transformer-based models trained with single-channel signals. 

**Abstract (ZH)**: 基于脑电图（EEG）和肌电图（EMG）信号的自动睡眠分期是相关研究的重要方面。当前的睡眠分期方法存在两大主要问题。首先，现有方法中的不同模态之间信息交互有限。其次，当前方法没有开发能够处理不同输入来源的统一模型。为解决这些问题，我们提出了一种新的睡眠阶段评分模型 sDREAMER，该模型强调跨模态交互和每通道性能。具体而言，我们开发了一个模态专家混合（MoME）模型，该模型具有三条路径，分别处理EEG、EMG和混合信号，并共享部分权重。我们还提出了一种自我蒸馏训练方案，以进一步促进不同模态之间的信息交互。该模型可以使用多通道输入进行训练，并且可以在单通道或多通道输入上进行分类。实验结果表明，我们的模型在多通道推理中优于现有的基于变压器的睡眠评分方法。对于单通道推理，我们的模型也优于使用单通道信号训练的基于变压器的模型。 

---
# Evaluating The Performance of Using Large Language Models to Automate Summarization of CT Simulation Orders in Radiation Oncology 

**Title (ZH)**: 评估使用大型语言模型自动化放射肿瘤学CT模拟订单总结的效果 

**Authors**: Meiyun Cao, Shaw Hu, Jason Sharp, Edward Clouser, Jason Holmes, Linda L. Lam, Xiaoning Ding, Diego Santos Toesca, Wendy S. Lindholm, Samir H. Patel, Sujay A. Vora, Peilong Wang, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16309)  

**Abstract**: Purpose: This study aims to use a large language model (LLM) to automate the generation of summaries from the CT simulation orders and evaluate its performance.
Materials and Methods: A total of 607 CT simulation orders for patients were collected from the Aria database at our institution. A locally hosted Llama 3.1 405B model, accessed via the Application Programming Interface (API) service, was used to extract keywords from the CT simulation orders and generate summaries. The downloaded CT simulation orders were categorized into seven groups based on treatment modalities and disease sites. For each group, a customized instruction prompt was developed collaboratively with therapists to guide the Llama 3.1 405B model in generating summaries. The ground truth for the corresponding summaries was manually derived by carefully reviewing each CT simulation order and subsequently verified by therapists. The accuracy of the LLM-generated summaries was evaluated by therapists using the verified ground truth as a reference.
Results: About 98% of the LLM-generated summaries aligned with the manually generated ground truth in terms of accuracy. Our evaluations showed an improved consistency in format and enhanced readability of the LLM-generated summaries compared to the corresponding therapists-generated summaries. This automated approach demonstrated a consistent performance across all groups, regardless of modality or disease site.
Conclusions: This study demonstrated the high precision and consistency of the Llama 3.1 405B model in extracting keywords and summarizing CT simulation orders, suggesting that LLMs have great potential to help with this task, reduce the workload of therapists and improve workflow efficiency. 

**Abstract (ZH)**: 目的：本研究旨在利用大型语言模型（LLM）自动化生成CT模拟订单的摘要，并评估其性能。
材料与方法：从我们机构的Aria数据库中收集了共607份患者的CT模拟订单。通过应用程序编程接口（API）服务访问我们本地托管的Llama 3.1 405B模型，用于从CT模拟订单中提取关键词并生成摘要。下载的CT模拟订单根据治疗方式和疾病部位分类为七个组别。对于每个组别，研究人员与治疗师合作开发了定制的指令提示，以指导Llama 3.1 405B模型生成摘要。每个组别的摘要地面真相通过仔细审查每个CT模拟订单并由治疗师验证后人为构建。治疗师使用验证过的地面真相作为参考，评估LLM生成的摘要的准确性。

结果：LLM生成的摘要与手工生成的地面真相在准确性方面的一致性约为98%。评估结果显示，与相应治疗师生成的摘要相比，LLM生成的摘要在格式上更具一致性且可读性更高。该自动化方法在所有组别中均表现出一致的性能，不受治疗方式或疾病部位的影响。

结论：本研究表明，Llama 3.1 405B模型在提取关键词和总结CT模拟订单方面具有高精度和一致性，表明LLM有巨大潜力帮助完成此任务，减轻治疗师的工作负担并提高工作效率。 

---
# Large Models in Dialogue for Active Perception and Anomaly Detection 

**Title (ZH)**: 大型模型在对话中的主动感知与异常检测 

**Authors**: Tzoulio Chamiti, Nikolaos Passalis, Anastasios Tefas  

**Link**: [PDF](https://arxiv.org/pdf/2501.16300)  

**Abstract**: Autonomous aerial monitoring is an important task aimed at gathering information from areas that may not be easily accessible by humans. At the same time, this task often requires recognizing anomalies from a significant distance or not previously encountered in the past. In this paper, we propose a novel framework that leverages the advanced capabilities provided by Large Language Models (LLMs) to actively collect information and perform anomaly detection in novel scenes. To this end, we propose an LLM based model dialogue approach, in which two deep learning models engage in a dialogue to actively control a drone to increase perception and anomaly detection accuracy. We conduct our experiments in a high fidelity simulation environment where an LLM is provided with a predetermined set of natural language movement commands mapped into executable code functions. Additionally, we deploy a multimodal Visual Question Answering (VQA) model charged with the task of visual question answering and captioning. By engaging the two models in conversation, the LLM asks exploratory questions while simultaneously flying a drone into different parts of the scene, providing a novel way to implement active perception. By leveraging LLMs reasoning ability, we output an improved detailed description of the scene going beyond existing static perception approaches. In addition to information gathering, our approach is utilized for anomaly detection and our results demonstrate the proposed methods effectiveness in informing and alerting about potential hazards. 

**Abstract (ZH)**: 自主导航监测是一项重要的任务，旨在从人类难以到达的区域收集信息。同时，这项任务往往需要在远距离或以前未遇到的场景中识别异常。在本文中，我们提出了一种新的框架，利用大型语言模型（LLMs）的高级能力，主动收集信息并进行异常检测。为此，我们提出了一种基于LLM的模型对话方法，其中两个深度学习模型进行对话，主动控制无人机以提高感知和异常检测的准确性。我们在一个高保真仿真环境中进行了实验，其中LLM被提供了一组预先确定的自然语言移动指令，这些指令映射为可执行代码功能。此外，我们部署了一个多模态视觉问答（VQA）模型，负责视觉问答和图像字幕生成的任务。通过让两个模型进行对话，LLM在飞行无人机探索不同场景部分时提出探索性问题，从而实现了一种新颖的主动感知方式。通过利用LLM的推理能力，我们输出了超越现有静态感知方法的改进详细场景描述。除了信息收集，我们的方法还用于异常检测，实验结果证明了所提出方法在告知和预警潜在危险方面的效果。 

---
# Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Modality-Aware Sparsity 

**Title (ZH)**: Mixture-of-Mamba：通过模态感知稀疏性增强多模态状态空间模型 

**Authors**: Weixin Liang, Junhong Shen, Genghan Zhang, Ning Dong, Luke Zettlemoyer, Lili Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16295)  

**Abstract**: State Space Models (SSMs) have emerged as efficient alternatives to Transformers for sequential modeling, but their inability to leverage modality-specific features limits their performance in multi-modal pretraining. Here, we propose Mixture-of-Mamba, a novel SSM architecture that introduces modality-aware sparsity through modality-specific parameterization of the Mamba block. Building on Mixture-of-Transformers (W. Liang et al. arXiv:2411.04996; 2024), we extend the benefits of modality-aware sparsity to SSMs while preserving their computational efficiency. We evaluate Mixture-of-Mamba across three multi-modal pretraining settings: Transfusion (interleaved text and continuous image tokens with diffusion loss), Chameleon (interleaved text and discrete image tokens), and an extended three-modality framework incorporating speech. Mixture-of-Mamba consistently reaches the same loss values at earlier training steps with significantly reduced computational costs. In the Transfusion setting, Mixture-of-Mamba achieves equivalent image loss using only 34.76% of the training FLOPs at the 1.4B scale. In the Chameleon setting, Mixture-of-Mamba reaches similar image loss with just 42.50% of the FLOPs at the 1.4B scale, and similar text loss with just 65.40% of the FLOPs. In the three-modality setting, MoM matches speech loss at 24.80% of the FLOPs at the 1.4B scale. Our ablation study highlights the synergistic effects of decoupling projection components, where joint decoupling yields greater gains than individual modifications. These results establish modality-aware sparsity as a versatile and effective design principle, extending its impact from Transformers to SSMs and setting new benchmarks in multi-modal pretraining. Our code can be accessed at this https URL 

**Abstract (ZH)**: 状态空间模型（SSMs）已成为Transformer在序列建模中的有效替代方案，但它们无法充分利用模态特定特征的限制，从而在多模态预训练中影响了其性能。在此基础上，我们提出了Mixture-of-Mamba，这是一种新颖的SSM架构，通过Mamba块的模态特定参数化引入了模态感知稀疏性。基于Mixture-of-Transformers（W. Liang等人，arXiv:2411.04996；2024），我们将模态感知稀疏性的优点扩展到了SSMs，同时保持其计算效率。我们在三种多模态预训练设置中评估了Mixture-of-Mamba：Transfusion（交错的文本和连续图像令牌，带有扩散损失）、Chameleon（交错的文本和离散图像令牌）以及一个扩展的三模态框架，包括语音。Mixture-of-Mamba在早期训练步骤中始终能够以显著降低的计算成本达到相同的损失值。在Transfusion设置中，Mixture-of-Mamba在1.4B规模下仅使用34.76%的训练FLOPs就达到了同等的图像损失。在Chameleon设置中，Mixture-of-Mamba以42.50%的FLOPs达到了类似的图像损失和65.40%的FLOPs达到了类似的文本损失。在三模态设置中，MoM在24.80%的FLOPs下达到了同等的语音损失。我们的消融研究突显了将投影组件解耦的协同效应，其中联合解耦所产生的改进大于单独修改。这些结果确立了模态感知稀疏性作为一个多功能且有效的设计原则，将其影响从Transformer扩展到SSMs，并在多模态预训练中设立了新的基准。我们的代码可通过以下链接访问：[此处链接] 

---
# Upside Down Reinforcement Learning with Policy Generators 

**Title (ZH)**: 倒置强化学习中的策略生成器 

**Authors**: Jacopo Di Ventura, Dylan R. Ashley, Francesco Faccio, Vincent Herrmann, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2501.16288)  

**Abstract**: Upside Down Reinforcement Learning (UDRL) is a promising framework for solving reinforcement learning problems which focuses on learning command-conditioned policies. In this work, we extend UDRL to the task of learning a command-conditioned generator of deep neural network policies. We accomplish this using Hypernetworks - a variant of Fast Weight Programmers, which learn to decode input commands representing a desired expected return into command-specific weight matrices. Our method, dubbed Upside Down Reinforcement Learning with Policy Generators (UDRLPG), streamlines comparable techniques by removing the need for an evaluator or critic to update the weights of the generator. To counteract the increased variance in last returns caused by not having an evaluator, we decouple the sampling probability of the buffer from the absolute number of policies in it, which, together with a simple weighting strategy, improves the empirical convergence of the algorithm. Compared with existing algorithms, UDRLPG achieves competitive performance and high returns, sometimes outperforming more complex architectures. Our experiments show that a trained generator can generalize to create policies that achieve unseen returns zero-shot. The proposed method appears to be effective in mitigating some of the challenges associated with learning highly multimodal functions. Altogether, we believe that UDRLPG represents a promising step forward in achieving greater empirical sample efficiency in RL. A full implementation of UDRLPG is publicly available at this https URL 

**Abstract (ZH)**: 倒置强化学习（UDRL）是一种解决强化学习问题的有前景的框架，它侧重于学习命令条件下的策略。在本工作中，我们扩展了UDRL，使其能够学习一个命令条件下的深度神经网络策略生成器。我们通过使用Hypernetworks（一种快速权重编程器的变体）来实现这一点，Hypernetworks能够学习将表示期望回报的输入命令解码为特定于命令的权重矩阵。我们的方法被称为“倒置强化学习与策略生成器”（UDRLPG），通过消除需要评估器或评论家来更新生成器权重的需要，简化了相关技术。为了解决因缺乏评估器引起的最后一轮回报方差增加的问题，我们从缓冲区的采样概率中解耦绝对数量的策略，这与一个简单的权重策略结合，改善了算法的经验收敛性。与现有算法相比，UDRLPG实现了具有竞争力的性能和高回报，有时甚至优于更复杂的架构。我们的实验表明，训练后的生成器可以泛化以创建实现未见过的回报的策略，这是一种零样本的效果。所提出的方法似乎有助于缓解学习高度多模态函数的一些挑战。总的来说，我们认为UDRLPG代表了实现强化学习中更高经验样本效率的一个有前景的进步。完整的UDRLPG实现可以在此 <https://> 地址公开获取。 

---
# Brain-Adapter: Enhancing Neurological Disorder Analysis with Adapter-Tuning Multimodal Large Language Models 

**Title (ZH)**: 脑适配器：通过适配调谐多模态大型语言模型增强神经疾病分析 

**Authors**: Jing Zhang, Xiaowei Yu, Yanjun Lyu, Lu Zhang, Tong Chen, Chao Cao, Yan Zhuang, Minheng Chen, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16282)  

**Abstract**: Understanding brain disorders is crucial for accurate clinical diagnosis and treatment. Recent advances in Multimodal Large Language Models (MLLMs) offer a promising approach to interpreting medical images with the support of text descriptions. However, previous research has primarily focused on 2D medical images, leaving richer spatial information of 3D images under-explored, and single-modality-based methods are limited by overlooking the critical clinical information contained in other modalities. To address this issue, this paper proposes Brain-Adapter, a novel approach that incorporates an extra bottleneck layer to learn new knowledge and instill it into the original pre-trained knowledge. The major idea is to incorporate a lightweight bottleneck layer to train fewer parameters while capturing essential information and utilize a Contrastive Language-Image Pre-training (CLIP) strategy to align multimodal data within a unified representation space. Extensive experiments demonstrated the effectiveness of our approach in integrating multimodal data to significantly improve the diagnosis accuracy without high computational costs, highlighting the potential to enhance real-world diagnostic workflows. 

**Abstract (ZH)**: 了解大脑疾病对于准确的临床诊断和治疗至关重要。近年来，多模态大型语言模型（MLLMs）的发展为通过文本描述解释医学图像提供了有希望的方法。然而，此前的研究主要集中在2D医学图像上，忽略了3D图像中 richer 的空间信息，而基于单一模态的方法则受限于未能充分利用其他模态中的关键临床信息。为解决这一问题，本文提出了一种名为 Brain-Adapter 的新型方法，该方法通过引入额外的瓶颈层来学习新知识并将其融入原始预训练知识中。主要思想是通过引入一个轻量级的瓶颈层来训练较少的参数，同时捕获关键信息，并利用 Contrastive Language-Image Pre-training (CLIP) 策略在统一的表示空间内对多模态数据进行对齐。广泛实验表明，该方法在整合多模态数据以显著提高诊断准确性方面具有有效性，且无需高昂的计算成本，突显了其在真实临床诊断流程中增强性能的潜力。 

---
# What is Formal Verification without Specifications? A Survey on mining LTL Specifications 

**Title (ZH)**: 没有规范的要求，形式验证何以为继？关于挖掘LTL规范的一篇综述

在这个翻译中，保留了原文的核心意思，并且确保语言符合学术规范。其中，“What is Formal Verification without Specifications?” 被翻译为“没有规范的要求，形式验证何以为继？”，“A Survey on mining LTL Specifications”则被翻译为“关于挖掘LTL规范的一篇综述”。 

**Authors**: Daniel Neider, Rajarshi Roy  

**Link**: [PDF](https://arxiv.org/pdf/2501.16274)  

**Abstract**: Virtually all verification techniques using formal methods rely on the availability of a formal specification, which describes the design requirements precisely. However, formulating specifications remains a manual task that is notoriously challenging and error-prone. To address this bottleneck in formal verification, recent research has thus focussed on automatically generating specifications for formal verification from examples of (desired and undesired) system behavior. In this survey, we list and compare recent advances in mining specifications in Linear Temporal Logic (LTL), the de facto standard specification language for reactive systems. Several approaches have been designed for learning LTL formulas, which address different aspects and settings of specification design. Moreover, the approaches rely on a diverse range of techniques such as constraint solving, neural network training, enumerative search, etc. We survey the current state-of-the-art techniques and compare them for the convenience of the formal methods practitioners. 

**Abstract (ZH)**: 几乎所有使用形式方法的验证技术都依赖于一个形式规范的存在，这个规范精确地描述了设计要求。然而，制定规范仍然是一个手动的任务，且被认为是一个高度具有挑战性和容易出错的过程。为了解决形式验证中的这一瓶颈，近期的研究已经集中在从系统的（期望和不期望）行为实例中自动生成用于形式验证的规范。在本文综述中，我们列举并比较了近期在线性时序逻辑（LTL）中挖掘规范的最新进展。LTL 是响应式系统事实上的标准规范语言。已有多种方法用于学习 LTL 公式，这些方法分别针对规范设计的不同方面和环境。此外，这些方法依赖于多种技术，如约束求解、神经网络训练、枚举搜索等。本文综述了当前最先进的技术，并方便形式方法的实际应用进行比较分析。 

---
# Return of the Encoder: Maximizing Parameter Efficiency for SLMs 

**Title (ZH)**: 返回编码器：最大化SLMs的参数效率 

**Authors**: Mohamed Elfeki, Rui Liu, Chad Voegele  

**Link**: [PDF](https://arxiv.org/pdf/2501.16273)  

**Abstract**: The dominance of large decoder-only language models has overshadowed encoder-decoder architectures, despite their fundamental efficiency advantages in sequence processing. For small language models (SLMs) - those with 1 billion parameters or fewer - our systematic analysis across GPU, CPU, and NPU platforms reveals that encoder-decoder architectures achieve 47% lower first-token latency and 4.7x higher throughput compared to decoder-only models on edge devices. These gains may be attributed to encoder-decoder's one-time input processing and efficient separation of understanding and generation phases.
We introduce a novel knowledge distillation framework that enables encoder-decoder models to leverage capabilities from large scalable decoder-only teachers while preserving their architectural advantages, achieving up to 6 average performance points improvement across diverse tasks, with significant gains in asymmetric sequence tasks where input and output distributions can benefit from different processing approaches.
When combined with modern advances like Rotary Positional Embeddings (RoPE) and Vision encoders, our systematic investigation demonstrates that encoder-decoder architectures provide a more practical path toward deploying capable language models in resource-constrained environments. Our findings challenge the prevailing trend toward decoder-only scaling, showing that architectural choices become increasingly crucial as parameter budgets decrease, particularly for on-device and edge deployments where computational efficiency is paramount. 

**Abstract (ZH)**: 尽管编码-解码架构在序列处理方面的基本效率优势显著，大型解码器-only语言模型的主导地位已经掩盖了它们。对于小规模语言模型（SLMs，参数量不超过10亿），我们系统分析了其在GPU、CPU和NPU等多个平台上的表现，发现编码-解码架构在边缘设备上的初始令牌延迟比纯解码器模型低47%，同时吞吐量高4.7倍。这些优势可能是由于编码-解码模型的一次性输入处理和理解与生成阶段的有效分离所致。

我们提出了一种新的知识蒸馏框架，使编码-解码模型能够利用大规模可扩展的纯解码器教师的性能，同时保留其架构优势。在各种任务中，这种框架能够实现多达6个平均性能点的改进，特别是在输入和输出分布可以从不同处理方式中获益的非对称序列任务中，性能提升尤为显著。

结合现代进展，如旋转位置编码（RoPE）和视觉编码器，我们的系统研究证明，编码-解码架构为资源受限环境部署具备能力的语言模型提供了一条更实际的道路。这些发现挑战了单一解码器扩展的主流趋势，表明随着参数预算减少，架构选择变得愈发重要，尤其是在计算效率至关重要的设备端和边缘部署环境中。 

---
# From Molecules to Mixtures: Learning Representations of Olfactory Mixture Similarity using Inductive Biases 

**Title (ZH)**: 从分子到混合物：利用归纳偏置学习嗅觉混合物相似性表示 

**Authors**: Gary Tom, Cher Tian Ser, Ella M. Rajaonson, Stanley Lo, Hyun Suk Park, Brian K. Lee, Benjamin Sanchez-Lengeling  

**Link**: [PDF](https://arxiv.org/pdf/2501.16271)  

**Abstract**: Olfaction -- how molecules are perceived as odors to humans -- remains poorly understood. Recently, the principal odor map (POM) was introduced to digitize the olfactory properties of single compounds. However, smells in real life are not pure single molecules, but complex mixtures of molecules, whose representations remain relatively under-explored. In this work, we introduce POMMix, an extension of the POM to represent mixtures. Our representation builds upon the symmetries of the problem space in a hierarchical manner: (1) graph neural networks for building molecular embeddings, (2) attention mechanisms for aggregating molecular representations into mixture representations, and (3) cosine prediction heads to encode olfactory perceptual distance in the mixture embedding space. POMMix achieves state-of-the-art predictive performance across multiple datasets. We also evaluate the generalizability of the representation on multiple splits when applied to unseen molecules and mixture sizes. Our work advances the effort to digitize olfaction, and highlights the synergy of domain expertise and deep learning in crafting expressive representations in low-data regimes. 

**Abstract (ZH)**: 嗅觉——分子如何被人类感知为气味——仍然缺乏充分理解。最近，主要气味图谱（POM）被引入以数字化单一化合物的感官特性。然而，现实生活中的气味并不是单一的分子，而是多种分子的复杂混合物，而这些混合物的表示方式仍然相对未被充分探索。在本项工作中，我们引入了POMMix，这是一种扩展的POM方法，用于表示混合物。我们的表示基于问题空间的对称性分级构建：(1) 图神经网络用于构建分子嵌入；(2) 注意机制用于将分子表示聚合为混合物表示；(3) 余弦预测头用于在混合物嵌入空间中编码嗅觉感知距离。POMMix在多个数据集上实现了最先进的预测性能。我们还在应用到未见过的分子和混合物尺寸时评估了表示方法的泛化能力。我们的工作推进了数字化嗅觉的努力，并强调了在数据稀缺条件下构建表达性表示时领域知识和深度学习的协同作用。 

---
# Lightweight Weighted Average Ensemble Model for Pneumonia Detection in Chest X-Ray Images 

**Title (ZH)**: 面向胸部X光图像肺炎检测的轻量级加权平均集成模型 

**Authors**: Suresh Babu Nettur, Shanthi Karpurapu, Unnati Nettur, Likhit Sagar Gajja, Sravanthy Myneni, Akhil Dusi, Lalithya Posham  

**Link**: [PDF](https://arxiv.org/pdf/2501.16249)  

**Abstract**: Pneumonia is a leading cause of illness and death in children, underscoring the need for early and accurate detection. In this study, we propose a novel lightweight ensemble model for detecting pneumonia in children using chest X-ray images. This ensemble model integrates two pre-trained convolutional neural networks (CNNs), MobileNetV2 and NASNetMobile, selected for their balance of computational efficiency and accuracy. These models were fine-tuned on a pediatric chest X-ray dataset and combined to enhance classification performance. Our proposed ensemble model achieved a classification accuracy of 98.63%, significantly outperforming individual models such as MobileNetV2 (97.10%) and NASNetMobile(96.25%) in terms of accuracy, precision, recall, and F1 score. Moreover, the ensemble model outperformed state-of-the-art architectures, including ResNet50, InceptionV3, and DenseNet201, while maintaining computational efficiency. The proposed lightweight ensemble model presents a highly effective and resource-efficient solution for pneumonia detection, making it particularly suitable for deployment in resource-constrained settings. 

**Abstract (ZH)**: 肺炎是导致儿童疾病和死亡的主要原因之一，这凸显了早期和准确检测的必要性。本研究提出了一种新颖的轻量级集成模型，用于通过胸部X光图像检测儿童肺炎。该集成模型结合了两种预训练的卷积神经网络（CNNs）——MobileNetV2和NASNetMobile，这两种网络在计算效率和准确性之间取得了平衡。这些模型在儿科胸部X光图像数据集上进行了微调，并结合使用以提高分类性能。我们提出的研究模型达到了98.63%的分类准确率，与单独使用的MobileNetV2（97.10%）和NASNetMobile（96.25%）相比，在准确率、精确率、召回率和F1分数方面显著优于这些模型。此外，该集成模型在计算效率方面优于现有的先进结构，如ResNet50、InceptionV3和DenseNet201，同时保持了较高的性能。所提出的轻量级集成模型提供了一种高效且资源节约的肺炎检测方案，特别适用于资源受限的环境部署。 

---
# Accelerating Quantum Reinforcement Learning with a Quantum Natural Policy Gradient Based Approach 

**Title (ZH)**: 基于量子自然策略梯度方法加速量子强化学习 

**Authors**: Yang Xu, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2501.16243)  

**Abstract**: We address the problem of quantum reinforcement learning (QRL) under model-free settings with quantum oracle access to the Markov Decision Process (MDP). This paper introduces a Quantum Natural Policy Gradient (QNPG) algorithm, which replaces the random sampling used in classical Natural Policy Gradient (NPG) estimators with a deterministic gradient estimation approach, enabling seamless integration into quantum systems. While this modification introduces a bounded bias in the estimator, the bias decays exponentially with increasing truncation levels. This paper demonstrates that the proposed QNPG algorithm achieves a sample complexity of $\tilde{\mathcal{O}}(\epsilon^{-1.5})$ for queries to the quantum oracle, significantly improving the classical lower bound of $\tilde{\mathcal{O}}(\epsilon^{-2})$ for queries to the MDP. 

**Abstract (ZH)**: 我们探讨了在使用量子 oracle 访问马尔可夫决策过程（MDP）的情况下，基于模型自由设置下的量子强化学习（QRL）问题。本文提出了一种量子自然策略梯度（QNPG）算法，它用确定性的梯度估计方法取代了经典自然策略梯度（NPG）估计器中的随机采样方法，从而能够无缝集成到量子系统中。虽然这种修改在估计器中引入了有限的偏差，但该偏差会随着截断级别的增加而呈指数级衰减。本文表明，所提出的 QNPG 算法的样本复杂度为 $\tilde{\mathcal{O}}(\epsilon^{-1.5})$，对于量子 oracle 的查询次数显著改善了经典 MDP 的下界 $\tilde{\mathcal{O}}(\epsilon^{-2})$。 

---
# Language-Based Bayesian Optimization Research Assistant (BORA) 

**Title (ZH)**: 语言驱动的贝叶斯优化研究助手（BORA） 

**Authors**: Abdoulatif Cissé, Xenophon Evangelopoulos, Vladimir V. Gusev, Andrew I. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2501.16224)  

**Abstract**: Many important scientific problems involve multivariate optimization coupled with slow and laborious experimental measurements. These complex, high-dimensional searches can be defined by non-convex optimization landscapes that resemble needle-in-a-haystack surfaces, leading to entrapment in local minima. Contextualizing optimizers with human domain knowledge is a powerful approach to guide searches to localized fruitful regions. However, this approach is susceptible to human confirmation bias and it is also challenging for domain experts to keep track of the rapidly expanding scientific literature. Here, we propose the use of Large Language Models (LLMs) for contextualizing Bayesian optimization (BO) via a hybrid optimization framework that intelligently and economically blends stochastic inference with domain knowledge-based insights from the LLM, which is used to suggest new, better-performing areas of the search space for exploration. Our method fosters user engagement by offering real-time commentary on the optimization progress, explaining the reasoning behind the search strategies. We validate the effectiveness of our approach on synthetic benchmarks with up to 15 independent variables and demonstrate the ability of LLMs to reason in four real-world experimental tasks where context-aware suggestions boost optimization performance substantially. 

**Abstract (ZH)**: 许多重要的科学问题涉及多变量优化与缓慢且繁琐的实验测量相结合。这类复杂且高维度的搜索可以由非凸优化景观定义，其类似于针尖在干草堆上的表面，这可能导致陷入局部极小值。利用人类领域知识来上下文化优化器是一种强大的方法，可以引导搜索到局部具有成果的区域。然而，这种方法容易受到人类确认偏见的影响，并且对于领域专家来说，跟踪迅速扩展的科学文献也极具挑战性。在此，我们提出使用大型语言模型（LLMs）通过将随机推理与来自LLM的基于领域知识的见解智能而经济地结合的混合优化框架，来上下文化贝叶斯优化（BO）。LLM用于建议搜索空间中新的、表现更好的区域以进行探索。我们的方法通过提供实时的优化进展评论来促进用户的参与，并解释搜索策略背后的推理。我们在多达15个独立变量的合成基准上验证了我们方法的有效性，并在四个真实世界的实验任务中展示了LLM能够基于上下文提供建议以大大提升优化性能的能力。 

---
# UDBE: Unsupervised Diffusion-based Brightness Enhancement in Underwater Images 

**Title (ZH)**: UDBE：水下图像无监督扩散增强亮度方法 

**Authors**: Tatiana Taís Schein, Gustavo Pereira de Almeira, Stephanie Loi Brião, Rodrigo Andrade de Bem, Felipe Gomes de Oliveira, Paulo L. J. Drews-Jr  

**Link**: [PDF](https://arxiv.org/pdf/2501.16211)  

**Abstract**: Activities in underwater environments are paramount in several scenarios, which drives the continuous development of underwater image enhancement techniques. A major challenge in this domain is the depth at which images are captured, with increasing depth resulting in a darker environment. Most existing methods for underwater image enhancement focus on noise removal and color adjustment, with few works dedicated to brightness enhancement. This work introduces a novel unsupervised learning approach to underwater image enhancement using a diffusion model. Our method, called UDBE, is based on conditional diffusion to maintain the brightness details of the unpaired input images. The input image is combined with a color map and a Signal-Noise Relation map (SNR) to ensure stable training and prevent color distortion in the output images. The results demonstrate that our approach achieves an impressive accuracy rate in the datasets UIEB, SUIM and RUIE, well-established underwater image benchmarks. Additionally, the experiments validate the robustness of our approach, regarding the image quality metrics PSNR, SSIM, UIQM, and UISM, indicating the good performance of the brightness enhancement process. The source code is available here: this https URL. 

**Abstract (ZH)**: 在水下环境中的活动在多种情况下至关重要，这推动了水下图像增强技术的持续发展。在这个领域中，主要的挑战在于图像被捕获的深度，随着深度的增加，环境会变得更暗。目前大多数水下图像增强方法主要集中在噪声去除和颜色调整上，较少有方法专注于亮度增强。本项工作提出了一种基于扩散模型的新型无监督学习方法，用于水下图像增强。我们的方法称为UDBE，基于条件扩散来保留未配对输入图像的亮度细节。输入图像与一个颜色图和信号-噪声关系图（SNR）相结合，以确保稳定的训练，并防止输出图像中的颜色失真。实验结果表明，我们的方法在UIEB、SUIM和RUIE等公认的标准水下图像数据集中具有令人 impressive 的准确率。此外，实验还验证了我们的方法在图像质量指标PSNR、SSIM、UIQM和UISM方面的鲁棒性，表明亮度提升过程的良好性能。源代码可在此处获取：this https URL。 

---
# Raiders of the Lost Dependency: Fixing Dependency Conflicts in Python using LLMs 

**Title (ZH)**: 《寻回缺失的依赖：使用大语言模型解决Python中的依赖冲突》 

**Authors**: Antony Bartlett, Cynthia Liem, Annibale Panichella  

**Link**: [PDF](https://arxiv.org/pdf/2501.16191)  

**Abstract**: Fixing Python dependency issues is a tedious and error-prone task for developers, who must manually identify and resolve environment dependencies and version constraints of third-party modules and Python interpreters. Researchers have attempted to automate this process by relying on large knowledge graphs and database lookup tables. However, these traditional approaches face limitations due to the variety of dependency error types, large sets of possible module versions, and conflicts among transitive dependencies. This study explores the potential of using large language models (LLMs) to automatically fix dependency issues in Python programs. We introduce PLLM (pronounced "plum"), a novel technique that employs retrieval-augmented generation (RAG) to help an LLM infer Python versions and required modules for a given Python file. PLLM builds a testing environment that iteratively (1) prompts the LLM for module combinations, (2) tests the suggested changes, and (3) provides feedback (error messages) to the LLM to refine the fix. This feedback cycle leverages natural language processing (NLP) to intelligently parse and interpret build error messages. We benchmark PLLM on the Gistable HG2.9K dataset, a collection of challenging single-file Python gists. We compare PLLM against two state-of-the-art automatic dependency inference approaches, namely PyEGo and ReadPyE, w.r.t. the ability to resolve dependency issues. Our results indicate that PLLM can fix more dependency issues than the two baselines, with +218 (+15.97%) more fixes over ReadPyE and +281 (+21.58%) over PyEGo. Our deeper analyses suggest that PLLM is particularly beneficial for projects with many dependencies and for specific third-party numerical and machine-learning modules. Our findings demonstrate the potential of LLM-based approaches to iteratively resolve Python dependency issues. 

**Abstract (ZH)**: 修复 Python 依赖问题对开发者来说是一个繁琐且容易出错的任务，他们需要手动识别和解决第三方模块和 Python 解释器的环境依赖及其版本限制。研究人员尝试通过依赖大型知识图谱和数据库查找表来自动化这一过程。然而，传统方法由于依赖错误类型多样、可能的模块版本众多以及传递依赖之间的冲突，面临着局限性。本研究探讨了使用大规模语言模型（LLMs）自动修复 Python 程序依赖问题的潜力。我们提出了一种名为 PLLM（发音为“plum”）的新技术，该技术利用检索增强生成（RAG）帮助 LLM 推断给定 Python 文件所需的 Python 版本和模块组合。PLLM 构建了一个测试环境，该环境通过以下步骤迭代工作：（1）提示 LLM 提出模块组合，（2）测试建议的更改，（3）向 LLM 提供反馈（错误消息），以便进一步优化修复。这一反馈循环利用自然语言处理（NLP）智能解析和解释构建错误消息。我们使用包含具有挑战性的单文件 Python gists 的 Gistable HG2.9K 数据集对 PLLM 进行基准测试，并将 PLLM 与两种最新的自动依赖推理方法 PyEGo 和 ReadPyE 进行比较，比较它们解决依赖问题的能力。结果表明，PLLM 比两种基准方法能够修复更多的依赖问题，相对于 ReadPyE 多修复 +218（+15.97%）个问题，相对于 PyEGo 多修复 +281（+21.58%）个问题。我们更深入的分析表明，PLLM 特别适用于具有众多依赖项的项目，以及特定的第三方数值和机器学习模块。我们的研究结果表明，基于 LLM 的方法有潜力逐步解决 Python 依赖问题。 

---
# The Linear Attention Resurrection in Vision Transformer 

**Title (ZH)**: 视觉转换器中的线性注意力复生 

**Authors**: Chuanyang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.16182)  

**Abstract**: Vision Transformers (ViTs) have recently taken computer vision by storm. However, the softmax attention underlying ViTs comes with a quadratic complexity in time and memory, hindering the application of ViTs to high-resolution images. We revisit the attention design and propose a linear attention method to address the limitation, which doesn't sacrifice ViT's core advantage of capturing global representation like existing methods (e.g. local window attention of Swin). We further investigate the key difference between linear attention and softmax attention. Our empirical results suggest that linear attention lacks a fundamental property of concentrating the distribution of the attention matrix. Inspired by this observation, we introduce a local concentration module to enhance linear attention. By incorporating enhanced linear global attention and local window attention, we propose a new ViT architecture, dubbed L$^2$ViT. Notably, L$^2$ViT can effectively capture both global interactions and local representations while enjoying linear computational complexity. Extensive experiments demonstrate the strong performance of L$^2$ViT. On image classification, L$^2$ViT achieves 84.4% Top-1 accuracy on ImageNet-1K without any extra training data or label. By further pre-training on ImageNet-22k, it attains 87.0% when fine-tuned with resolution 384$^2$. For downstream tasks, L$^2$ViT delivers favorable performance as a backbone on object detection as well as semantic segmentation. 

**Abstract (ZH)**: 视觉变换器（Vision Transformers，ViTs）最近在计算机视觉领域引起了广泛关注。然而，ViTs 所采用的基于softmax的注意力机制具有时间复杂度和空间复杂度上的二次效率问题，限制了其在高分辨率图像上的应用。我们重新审视了注意力设计，并提出了一种线性注意力方法来解决这一限制，该方法在保持ViT捕捉全局表示的核心优势方面并不逊色于现有的方法（例如Swin中的局部窗口注意力）。我们进一步探讨了线性注意力与softmax注意力之间的关键区别。我们的实验结果表明，线性注意力缺乏将注意力矩阵的分布集中起来的基本特性。受到这一观察的启发，我们引入了一个局部集中模块，以增强线性注意力。通过结合增强的线性全局注意力和局部窗口注意力，我们提出了一种新的ViT架构，名为L$^2$ViT。值得注意的是，L$^2$ViT能够在保持线性计算复杂度的同时，有效地捕捉全局交互和局部表示。广泛的实验表明了L$^2$ViT的强大性能。在图像分类任务中，L$^2$ViT在不使用任何额外训练数据或标签的情况下，在ImageNet-1K上达到了84.4%的Top-1准确率。进一步在ImageNet-22k上进行预训练后，通过分辨率384$^2$的微调，其准确率达到87.0%。对于下游任务，L$^2$ViT作为主干网络，在物体检测和语义分割方面表现出了可喜的性能。 

---
# BAG: Body-Aligned 3D Wearable Asset Generation 

**Title (ZH)**: BAG：身体对齐的3D可穿戴资产生成 

**Authors**: Zhongjin Luo, Yang Li, Mingrui Zhang, Senbo Wang, Han Yan, Xibin Song, Taizhang Shang, Wei Mao, Hongdong Li, Xiaoguang Han, Pan Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.16177)  

**Abstract**: While recent advancements have shown remarkable progress in general 3D shape generation models, the challenge of leveraging these approaches to automatically generate wearable 3D assets remains unexplored. To this end, we present BAG, a Body-aligned Asset Generation method to output 3D wearable asset that can be automatically dressed on given 3D human bodies. This is achived by controlling the 3D generation process using human body shape and pose information. Specifically, we first build a general single-image to consistent multiview image diffusion model, and train it on the large Objaverse dataset to achieve diversity and generalizability. Then we train a Controlnet to guide the multiview generator to produce body-aligned multiview images. The control signal utilizes the multiview 2D projections of the target human body, where pixel values represent the XYZ coordinates of the body surface in a canonical space. The body-conditioned multiview diffusion generates body-aligned multiview images, which are then fed into a native 3D diffusion model to produce the 3D shape of the asset. Finally, by recovering the similarity transformation using multiview silhouette supervision and addressing asset-body penetration with physics simulators, the 3D asset can be accurately fitted onto the target human body. Experimental results demonstrate significant advantages over existing methods in terms of image prompt-following capability, shape diversity, and shape quality. Our project page is available at this https URL. 

**Abstract (ZH)**: 尽管最近的进展在通用3D形状生成模型方面取得了显著进步，但利用这些方法自动生成可穿戴3D资产的挑战仍未被探索。为此，我们提出了一种体形对齐资产生成方法（BAG），以生成可以自动穿戴在给定3D人体上的可穿戴3D资产。这一方法通过使用人体形状和姿态信息来控制3D生成过程得以实现。具体而言，我们首先构建了一个通用的单图像到一致的多视角图像扩散模型，并在大规模Objaverse数据集上对其进行训练，以实现多样性和泛化能力。然后，我们训练一个ControlNet，以引导多视角生成器产生体形对齐的多视角图像。控制信号利用目标人体的多视角2D投影，其中像素值表示人体表面在标准空间中的XYZ坐标。体形条件的多视角扩散生成体形对齐的多视角图像，然后输入原始的3D扩散模型以生成资产的3D形状。最后，通过使用多视角轮廓监督恢复相似变换，并使用物理模拟器解决资产与人体穿插的问题，可以准确地将3D资产匹配到目标人体上。实验结果表明，在图像提示跟随能力、形状多样性以及形状质量方面，本方法相较于现有方法具备显著优势。我们的项目页面可在以下链接访问：[此 https URL](此 https URL)。 

---
# Measuring Heterogeneity in Machine Learning with Distributed Energy Distance 

**Title (ZH)**: 用分布式能量距离测量机器学习中的异质性 

**Authors**: Mengchen Fan, Baocheng Geng, Roman Shterenberg, Joseph A. Casey, Zhong Chen, Keren Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.16174)  

**Abstract**: In distributed and federated learning, heterogeneity across data sources remains a major obstacle to effective model aggregation and convergence. We focus on feature heterogeneity and introduce energy distance as a sensitive measure for quantifying distributional discrepancies. While we show that energy distance is robust for detecting data distribution shifts, its direct use in large-scale systems can be prohibitively expensive. To address this, we develop Taylor approximations that preserve key theoretical quantitative properties while reducing computational overhead. Through simulation studies, we show how accurately capturing feature discrepancies boosts convergence in distributed learning. Finally, we propose a novel application of energy distance to assign penalty weights for aligning predictions across heterogeneous nodes, ultimately enhancing coordination in federated and distributed settings. 

**Abstract (ZH)**: 在分布式和联邦学习中，数据源之间的异质性仍然是有效模型聚合和收敛的主要障碍。本文重点关注特征异质性，并引入能量距离作为一种敏感度量方法，用于量化分布差异。虽然我们证明了能量距离在检测数据分布转移方面具有稳健性，但其直接用于大规模系统中可能会极其昂贵。为解决这一问题，我们开发了泰勒近似方法，这些方法保留了关键的理论量化属性，同时减少了计算开销。通过模拟研究，我们展示了准确捕捉特征差异如何提升分布式学习中的收敛性。最后，我们提出了一种能量距离的新应用，用于为异构节点之间的预测对齐分配惩罚权重，从而最终增强联邦和分布式环境中的协调性。 

---
# MetaDecorator: Generating Immersive Virtual Tours through Multimodality 

**Title (ZH)**: MetaDecorator：通过多模态生成沉浸式虚拟导览 

**Authors**: Shuang Xie, Yang Liu, Jeannie S.A. Lee, Haiwei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2501.16164)  

**Abstract**: MetaDecorator, is a framework that empowers users to personalize virtual spaces. By leveraging text-driven prompts and image synthesis techniques, MetaDecorator adorns static panoramas captured by 360° imaging devices, transforming them into uniquely styled and visually appealing environments. This significantly enhances the realism and engagement of virtual tours compared to traditional offerings. Beyond the core framework, we also discuss the integration of Large Language Models (LLMs) and haptics in the VR application to provide a more immersive experience. 

**Abstract (ZH)**: MetaDecorator 是一个框架，赋能用户个性化虚拟空间。通过利用文本驱动的提示和图像合成技术，MetaDecorator 装饰由360°成像设备捕获的静态全景图，将其转化为具有独特风格和视觉吸引力的环境。这相较于传统方案显著提升了虚拟巡游的真实感和参与度。除了核心框架之外，我们还讨论了将大型语言模型（LLMs）和触觉技术集成到VR应用中，以提供更加沉浸式的体验。 

---
# AdaCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Chain-of-Thought 

**Title (ZH)**: AdaCoT：重新思考适应性链式思考在跨语言事实推理中的作用 

**Authors**: Xin Huang, Tarun Kumar Vangani, Zhengyuan Liu, Bowei Zou, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2501.16154)  

**Abstract**: Large language models (LLMs) have shown impressive multilingual capabilities through pretraining on diverse corpora. While these models show strong reasoning abilities, their performance varies significantly across languages due to uneven training data distribution. Existing approaches using machine translation, and extensive multilingual pretraining and cross-lingual tuning face scalability challenges and often fail to capture nuanced reasoning processes across languages. In this paper, we introduce AdaCoT (Adaptive Chain-of-Thought), a framework that enhances multilingual reasoning by dynamically routing thought processes through intermediary "thinking languages" before generating target-language responses. AdaCoT leverages a language-agnostic core and incorporates an adaptive, reward-based mechanism for selecting optimal reasoning pathways without requiring additional pretraining. Our comprehensive evaluation across multiple benchmarks demonstrates substantial improvements in both factual reasoning quality and cross-lingual consistency, with particularly strong performance gains in low-resource language settings. The results suggest that adaptive reasoning paths can effectively bridge the performance gap between high and low-resource languages while maintaining cultural and linguistic nuances. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过在多种语料上进行预训练展现了令人印象深刻的多语言能力。尽管这些模型在推理方面表现出强大的能力，但由于训练数据分布不均，其在不同语言上的表现存在显著差异。现有通过机器翻译、广泛多语言预训练和跨语言调优的方法面临着可扩展性挑战，往往无法捕捉不同语言间细微的推理过程。本文提出了一种名为AdaCoT（自适应链式思考）的框架，该框架通过动态将思考过程路由到中间的“思考语言”来进行推理，然后再生成目标语言的响应。AdaCoT 利用一个语言无关的核心，并结合了一种基于奖励的自适应机制，以选择最优的推理路径，而无需额外的预训练。我们在多个基准测试上的综合评估表明，该方法在事实推理质量和跨语言一致性方面均取得了显著改善，特别是在低资源语言环境下表现出明显的性能提升。研究结果表明，自适应的推理路径能够有效缩小高资源与低资源语言之间的性能差距，同时保留文化与语言的细微差异。 

---
# Toward Efficient Generalization in 3D Human Pose Estimation via a Canonical Domain Approach 

**Title (ZH)**: 通过规范领域方法 toward 有效泛化在三维人体姿态估计中的应用 

**Authors**: Hoosang Lee, Jeha Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16146)  

**Abstract**: Recent advancements in deep learning methods have significantly improved the performance of 3D Human Pose Estimation (HPE). However, performance degradation caused by domain gaps between source and target domains remains a major challenge to generalization, necessitating extensive data augmentation and/or fine-tuning for each specific target domain. To address this issue more efficiently, we propose a novel canonical domain approach that maps both the source and target domains into a unified canonical domain, alleviating the need for additional fine-tuning in the target domain. To construct the canonical domain, we introduce a canonicalization process to generate a novel canonical 2D-3D pose mapping that ensures 2D-3D pose consistency and simplifies 2D-3D pose patterns, enabling more efficient training of lifting networks. The canonicalization of both domains is achieved through the following steps: (1) in the source domain, the lifting network is trained within the canonical domain; (2) in the target domain, input 2D poses are canonicalized prior to inference by leveraging the properties of perspective projection and known camera intrinsics. Consequently, the trained network can be directly applied to the target domain without requiring additional fine-tuning. Experiments conducted with various lifting networks and publicly available datasets (e.g., Human3.6M, Fit3D, MPI-INF-3DHP) demonstrate that the proposed method substantially improves generalization capability across datasets while using the same data volume. 

**Abstract (ZH)**: 近年来，深度学习方法的进展显著提升了三维人体姿态估计（3D Human Pose Estimation, HPE）的性能。然而，由于源域与目标域之间的领域差距导致的性能下降仍然是泛化过程中的主要挑战，这需要对每个特定的目标域进行大量的数据增强和/或微调。为了更有效地解决这一问题，我们提出了一种新颖的典范域方法，该方法将源域和目标域都映射到一个统一的典范域，从而减轻在目标域中额外进行微调的需要。为了构建典范域，我们引入了一种典范化过程，生成了一种新颖的2D-3D姿态映射，该映射确保了2D-3D姿态的一致性并简化了2D-3D姿态模式，从而使提升网络的训练更加高效。通过以下步骤实现两个领域的同时典范化：（1）在源域中，提升网络在典范域内进行训练；（2）在目标域中，通过利用透视投影的性质和已知的相机内参对输入的2D姿态进行典范化，再进行推断。因此，训练好的网络可以直接应用于目标域而无需额外的微调。实验结果表明，与使用相同数据量相比，本方法在各种提升网络和公开可用的数据集（如Human3.6M、Fit3D、MPI-INF-3DHP）上显著提高了泛化能力。 

---
# Towards General-Purpose Model-Free Reinforcement Learning 

**Title (ZH)**: toward 通用目的的模型无关强化学习 

**Authors**: Scott Fujimoto, Pierluca D'Oro, Amy Zhang, Yuandong Tian, Michael Rabbat  

**Link**: [PDF](https://arxiv.org/pdf/2501.16142)  

**Abstract**: Reinforcement learning (RL) promises a framework for near-universal problem-solving. In practice however, RL algorithms are often tailored to specific benchmarks, relying on carefully tuned hyperparameters and algorithmic choices. Recently, powerful model-based RL methods have shown impressive general results across benchmarks but come at the cost of increased complexity and slow run times, limiting their broader applicability. In this paper, we attempt to find a unifying model-free deep RL algorithm that can address a diverse class of domains and problem settings. To achieve this, we leverage model-based representations that approximately linearize the value function, taking advantage of the denser task objectives used by model-based RL while avoiding the costs associated with planning or simulated trajectories. We evaluate our algorithm, MR.Q, on a variety of common RL benchmarks with a single set of hyperparameters and show a competitive performance against domain-specific and general baselines, providing a concrete step towards building general-purpose model-free deep RL algorithms. 

**Abstract (ZH)**: 强化学习（RL）承诺提供一个近乎普遍适用的问题解决框架。然而，在实践中，RL 算法通常针对特定基准进行调整，依赖于精细调优的超参数和算法选择。最近，强大的基于模型的RL方法在多个基准测试中展现了令人印象深刻的通用性能，但这也带来了复杂度增加和运行时间延长的问题，限制了其更广泛的应用。在本文中，我们试图找到一种统一的无模型的深度RL算法，能够解决一组多样化的领域和问题设置。为了实现这一点，我们利用模型化的表示来近似线性化价值函数，利用基于模型的RL所使用的密集任务目标，同时避免规划或模拟轨迹带来的成本。我们以单一的超参数集评估我们的算法MR.Q，并展示了其在多种常见的RL基准测试中的竞争表现，对抗特定领域和通用基准的不同算法，朝着构建通用的无模型深度RL算法迈出了具体的一步。 

---
# Automated Detection of Sport Highlights from Audio and Video Sources 

**Title (ZH)**: 从音频和视频源自动检测体育高光时刻 

**Authors**: Francesco Della Santa, Morgana Lalli  

**Link**: [PDF](https://arxiv.org/pdf/2501.16100)  

**Abstract**: This study presents a novel Deep Learning-based and lightweight approach for the automated detection of sports highlights (HLs) from audio and video sources. HL detection is a key task in sports video analysis, traditionally requiring significant human effort. Our solution leverages Deep Learning (DL) models trained on relatively small datasets of audio Mel-spectrograms and grayscale video frames, achieving promising accuracy rates of 89% and 83% for audio and video detection, respectively. The use of small datasets, combined with simple architectures, demonstrates the practicality of our method for fast and cost-effective deployment. Furthermore, an ensemble model combining both modalities shows improved robustness against false positives and false negatives. The proposed methodology offers a scalable solution for automated HL detection across various types of sports video content, reducing the need for manual intervention. Future work will focus on enhancing model architectures and extending this approach to broader scene-detection tasks in media analysis. 

**Abstract (ZH)**: 本研究提出了一种基于深度学习且轻量级的方法，用于从音频和视频源自动检测体育精彩片段（HLs）。精彩片段检测是体育视频分析中的一个关键任务，传统上需要大量的人工努力。我们利用在相对较小的音频梅尔频谱图和灰度视频帧数据集上训练的深度学习（DL）模型，分别实现了89%和83%的音频和视频检测准确率。利用较小的数据集和简单的架构，证明了我们方法的实用性和快速、低成本部署的可行性。此外，结合两种模态的集成模型在对抗假阳性与假阴性方面表现出更强的稳健性。所提方法提供了一种针对各类体育视频内容可扩展的自动精彩片段检测解决方案，减少了手动干预的需求。未来的工作将致力于增强模型架构，并将此方法扩展到媒体分析中的更广泛的场景检测任务。 

---
# STAR: Stepwise Task Augmentation and Relation Learning for Aspect Sentiment Quad Prediction 

**Title (ZH)**: STAR：逐步任务扩展与关系学习在方面情感四元组预测中的应用 

**Authors**: Wenna Lai, Haoran Xie, Guandong Xu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.16093)  

**Abstract**: Aspect-based sentiment analysis (ABSA) aims to identify four sentiment elements, including aspect term, aspect category, opinion term, and sentiment polarity. These elements construct the complete picture of sentiments. The most challenging task, aspect sentiment quad prediction (ASQP), predicts these elements simultaneously, hindered by difficulties in accurately coupling different sentiment elements. A key challenge is insufficient annotated data that limits the capability of models in semantic understanding and reasoning about quad prediction. To address this, we propose stepwise task augmentation and relation learning (STAR), a strategy inspired by human reasoning. STAR constructs auxiliary data to learn quadruple relationships incrementally by augmenting with pairwise and overall relation tasks derived from training data. By encouraging the model to infer causal relationships among sentiment elements without requiring additional annotations, STAR effectively enhances quad prediction. Extensive experiments demonstrate the proposed STAR exhibits superior performance on four benchmark datasets. 

**Abstract (ZH)**: 基于aspect的 sentiment分析（Aspect-based Sentiment Analysis, ABSA）旨在识别四个情感元素，包括aspect术语、aspect类别、意见术语和情感极性。这些元素共同构建了完整的情感图谱。最具有挑战性的任务是aspect情感四元预测（ASQP），该任务同时预测这些元素，受到在不同情感元素之间准确耦合的困难限制。一个关键挑战是标注数据不足，这限制了模型在语义理解和四元预测方面的推理能力。为此，我们提出了一种逐步任务增强和关系学习（Stepwise Task Augmentation and Relation Learning, STAR）策略，该策略灵感来源于人类推理。STAR通过利用从训练数据派生的成对关系任务和整体关系任务逐步构建辅助数据，逐步学习四元关系。通过鼓励模型在无需额外标注的情况下推断情感元素之间的因果关系，STAR有效地提升了四元预测的性能。广泛的实验表明，所提出的STAR在四个基准数据集上表现出优越的性能。 

---
# PISCO: Pretty Simple Compression for Retrieval-Augmented Generation 

**Title (ZH)**: PISCO：简单的检索增强生成压缩方法 

**Authors**: Maxime Louis, Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2501.16075)  

**Abstract**: Retrieval-Augmented Generation (RAG) pipelines enhance Large Language Models (LLMs) by retrieving relevant documents, but they face scalability issues due to high inference costs and limited context size. Document compression is a practical solution, but current soft compression methods suffer from accuracy losses and require extensive pretraining. In this paper, we introduce PISCO, a novel method that achieves a 16x compression rate with minimal accuracy loss (0-3%) across diverse RAG-based question-answering (QA) tasks. Unlike existing approaches, PISCO requires no pretraining or annotated data, relying solely on sequence-level knowledge distillation from document-based questions. With the ability to fine-tune a 7-10B LLM in 48 hours on a single A100 GPU, PISCO offers a highly efficient and scalable solution. We present comprehensive experiments showing that PISCO outperforms existing compression models by 8% in accuracy. 

**Abstract (ZH)**: 检索增强生成（RAG）管道通过检索相关文档来增强大型语言模型（LLMs），但它们面临着由于高推理成本和有限的上下文大小而导致的可扩展性问题。文档压缩是一个实际的解决方案，但当前的软压缩方法会导致准确率损失，并且需要大量的预训练。本文介绍了一种名为PISCO的新方法，它在多种基于RAG的问题回答（QA）任务中实现了16倍的压缩率，并且准确率损失极小（0-3%）。与现有方法不同的是，PISCO不需要预训练或标注数据，仅依赖于基于文档的问题的序列级知识蒸馏。PISCO能够在单个A100 GPU上在48小时内微调一个7-10B的LLM，从而提供了一种高效且可扩展的解决方案。我们进行了全面的实验，结果显示PISCO在准确率方面比现有的压缩模型高8%。 

---
# The Unbearable Lightness of Prompting: A Critical Reflection on the Environmental Impact of genAI use in Design Education 

**Title (ZH)**: 提示的不可承受之轻：对设计教育中生成AI使用环境影响的批判性反思 

**Authors**: Maria Luce Lupetti, Elena Cavallin, Dave Murray-Rust  

**Link**: [PDF](https://arxiv.org/pdf/2501.16061)  

**Abstract**: Design educators are finding ways to support students in skillfully using GenAI tools in their practices while encouraging the critical scrutiny of the ethical and social issues around these technologies. However, the issue of environmental sustainability remains unaddressed. There is a lack of both resources to grasp the environmental costs of genAI in education and a lack of shared practices for engaging with the issue. This paper critically reflects on the energy costs of using genAI in design education, using a workshop held in 2023 with 49 students as a motivating example. Through this reflection, we develop a set of five alternative stances, with related actions, that support the conscious use of genAI in design education. The work contributes to the field of design and HCI by bringing together ways for educators to reflect on their practices, informing the future development of educational programs around genAI. 

**Abstract (ZH)**: 设计教育者正在探索支持学生熟练使用生成式人工智能（GenAI）工具的方法，同时鼓励对这些技术带来的伦理和社会问题进行批判性审查。然而，环境可持续性这一议题尚未得到充分关注。教育中缺乏对GenAI环境成本的理解资源，也缺乏共同应对这一问题的实践。本文从2023年在一次包含49名学生的研讨会上进行反思入手，批判性地探讨在设计教育中使用GenAI的能耗问题。通过这一反思，我们提出了一套包含五种替代姿态及其相应行动的方案，以支持在设计教育中对GenAI的有意识应用。本文为设计与人机交互领域做出贡献，提供了教育者反思其实践的方式，并为未来围绕GenAI开发教育项目的方向提供了指导。 

---
# Skeleton-Guided-Translation: A Benchmarking Framework for Code Repository Translation with Fine-Grained Quality Evaluation 

**Title (ZH)**: 基于骨架指导的翻译：代码仓库翻译的基准框架及其细粒度质量评估 

**Authors**: Xing Zhang, Jiaheng Wen, Fangkai Yang, Pu Zhao, Yu Kang, Junhao Wang, Maoquan Wang, Yufan Huang, Elsie Nallipogu, Qingwei Lin, Yingnong Dang, Saravan Rajmohan, Dongmei Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16050)  

**Abstract**: The advancement of large language models has intensified the need to modernize enterprise applications and migrate legacy systems to secure, versatile languages. However, existing code translation benchmarks primarily focus on individual functions, overlooking the complexities involved in translating entire repositories, such as maintaining inter-module coherence and managing dependencies. While some recent repository-level translation benchmarks attempt to address these challenges, they still face limitations, including poor maintainability and overly coarse evaluation granularity, which make them less developer-friendly. We introduce Skeleton-Guided-Translation, a framework for repository-level Java to C# code translation with fine-grained quality evaluation. It uses a two-step process: first translating the repository's structural "skeletons", then translating the full repository guided by these skeletons. Building on this, we present TRANSREPO-BENCH, a benchmark of high quality open-source Java repositories and their corresponding C# skeletons, including matching unit tests and build configurations. Our unit tests are fixed and can be applied across multiple or incremental translations without manual adjustments, enhancing automation and scalability in evaluations. Additionally, we develop fine-grained evaluation metrics that assess translation quality at the individual test case level, addressing traditional binary metrics' inability to distinguish when build failures cause all tests to fail. Evaluations using TRANSREPO-BENCH highlight key challenges and advance more accurate repository level code translation. 

**Abstract (ZH)**: 大型语言模型的进步加剧了对企业应用程序的现代化需求，并推动了对现有系统向安全、多用途语言的迁移。然而，现有的代码翻译基准主要集中在单个函数的翻译上，忽视了整个代码库翻译过程中涉及的复杂性，如保持模块间的一致性和管理依赖关系。虽然有一些针对代码库级别翻译基准的努力试图解决这些问题，但是它们仍然存在局限性，包括可维护性差和评估粒度过粗，这使得它们不那么开发者友好。我们提出了一种名为Skeleton-Guided-Translation的框架，用于实现Java到C#代码库级别的细粒度质量评估。该框架采用两步过程：首先翻译代码库的结构“骨架”，然后根据这些骨架指导整个代码库的翻译。基于此，我们推出了TRANSREPO-BENCH基准，该基准包括高质量的开源Java代码库及其对应的C#骨架，包括匹配的单元测试和构建配置。我们的单元测试固定不变，可以在多次或增量翻译中无需手动调整即可应用，从而增强评估的自动化和可扩展性。此外，我们开发了细粒度的评估指标，这些指标评估个体测试用例的翻译质量，解决了传统二元指标在区分构建失败导致所有测试失败时存在的不足。使用TRANSREPO-BENCH进行的评估突显了库级别代码翻译的关键挑战，并推动了更准确的仓库级代码翻译的发展。 

---
# PRISMe: A Novel LLM-Powered Tool for Interactive Privacy Policy Assessment 

**Title (ZH)**: PRISMe：一种新型的基于大语言模型的互动隐私政策评估工具 

**Authors**: Vincent Freiberger, Arthur Fleig, Erik Buchmann  

**Link**: [PDF](https://arxiv.org/pdf/2501.16033)  

**Abstract**: Protecting online privacy requires users to engage with and comprehend website privacy policies, but many policies are difficult and tedious to read. We present PRISMe (Privacy Risk Information Scanner for Me), a novel Large Language Model (LLM)-driven privacy policy assessment tool, which helps users to understand the essence of a lengthy, complex privacy policy while browsing. The tool, a browser extension, integrates a dashboard and an LLM chat. One major contribution is the first rigorous evaluation of such a tool. In a mixed-methods user study (N=22), we evaluate PRISMe's efficiency, usability, understandability of the provided information, and impacts on awareness. While our tool improves privacy awareness by providing a comprehensible quick overview and a quality chat for in-depth discussion, users note issues with consistency and building trust in the tool. From our insights, we derive important design implications to guide future policy analysis tools. 

**Abstract (ZH)**: 保护在线隐私要求用户参与并理解网站隐私政策，但许多政策的内容冗长且难以阅读。我们提出了PRISMe（Privacy Risk Information Scanner for Me），这是一种基于大型语言模型（LLM）的新型隐私政策评估工具，帮助用户在浏览过程中理解复杂隐私政策的实质。该工具是一个浏览器扩展程序，集成了仪表板和LLM聊天功能。一个主要贡献是我们首次进行了此类工具的严格评估。在一项混合方法用户研究（n=22）中，我们评估了PRISMe的效率、易用性、提供信息的可理解性以及对隐私意识的影响。虽然我们的工具通过提供易于理解的快速概览和高质量的聊天对话来提高隐私意识，但用户指出工具的一致性和建立信任方面存在一些问题。从我们的研究中，我们得出了重要的设计启示，以指导未来政策分析工具的发展。 

---
# FDLLM: A Text Fingerprint Detection Method for LLMs in Multi-Language, Multi-Domain Black-Box Environments 

**Title (ZH)**: FDLLM：多语言、多领域黑盒环境中大规模语言模型文本指纹检测方法 

**Authors**: Zhiyuan Fu, Junfan Chen, Hongyu Sun, Ting Yang, Ruidong Li, Yuqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16029)  

**Abstract**: Using large language models (LLMs) integration platforms without transparency about which LLM is being invoked can lead to potential security risks. Specifically, attackers may exploit this black-box scenario to deploy malicious models and embed viruses in the code provided to users. In this context, it is increasingly urgent for users to clearly identify the LLM they are interacting with, in order to avoid unknowingly becoming victims of malicious models. However, existing studies primarily focus on mixed classification of human and machine-generated text, with limited attention to classifying texts generated solely by different models. Current research also faces dual bottlenecks: poor quality of LLM-generated text (LLMGT) datasets and limited coverage of detectable LLMs, resulting in poor detection performance for various LLMGT in black-box scenarios. We propose the first LLMGT fingerprint detection model, \textbf{FDLLM}, based on Qwen2.5-7B and fine-tuned using LoRA to address these challenges. FDLLM can more efficiently handle detection tasks across multilingual and multi-domain scenarios. Furthermore, we constructed a dataset named \textbf{FD-Datasets}, consisting of 90,000 samples that span multiple languages and domains, covering 20 different LLMs. Experimental results demonstrate that FDLLM achieves a macro F1 score 16.7\% higher than the best baseline method, LM-D. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，并符合学术规范：

在没有透明度的情况下使用大型语言模型（LLMs）集成平台，可能会引发潜在的安全风险。具体而言，攻击者可能利用这种“黑盒”场景部署恶意模型，并在提供给用户的代码中嵌入病毒。在这种背景下，用户越来越多地需要明确识别他们正在交互的LLM，以避免成为恶意模型的无意识受害者。然而，现有研究主要集中在人类和机器生成文本的混合分类上，对仅由不同模型生成的文本分类关注较少。当前研究还面临双重瓶颈：LLM生成的文本（LLM生成文本，LLMGT）数据集质量较差以及可检测LLM的覆盖率有限，这导致在“黑盒”场景下多种LLMGT的检测性能不佳。我们提出了一种基于Qwen2.5-7B并使用LoRA进一步调优的第一种LLMGT指纹检测模型——**FDLLM**。FDLLM能够更高效地处理多语言和多领域场景中的检测任务。此外，我们构建了一个名为**FD-Datasets**的数据集，包含90,000个样本，覆盖了多种语言和领域，并涵盖20种不同的LLM。实验结果表明，FDLLM在宏F1分数上比最佳基线方法LM-D高16.7%。 

---
# Controllable Forgetting Mechanism for Few-Shot Class-Incremental Learning 

**Title (ZH)**: 可控遗忘机制在少样本分类增量学习中的应用 

**Authors**: Kirill Paramonov, Mete Ozay, Eunju Yang, Jijoong Moon, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2501.15998)  

**Abstract**: Class-incremental learning in the context of limited personal labeled samples (few-shot) is critical for numerous real-world applications, such as smart home devices. A key challenge in these scenarios is balancing the trade-off between adapting to new, personalized classes and maintaining the performance of the model on the original, base classes. Fine-tuning the model on novel classes often leads to the phenomenon of catastrophic forgetting, where the accuracy of base classes declines unpredictably and significantly. In this paper, we propose a simple yet effective mechanism to address this challenge by controlling the trade-off between novel and base class accuracy. We specifically target the ultra-low-shot scenario, where only a single example is available per novel class. Our approach introduces a Novel Class Detection (NCD) rule, which adjusts the degree of forgetting a priori while simultaneously enhancing performance on novel classes. We demonstrate the versatility of our solution by applying it to state-of-the-art Few-Shot Class-Incremental Learning (FSCIL) methods, showing consistent improvements across different settings. To better quantify the trade-off between novel and base class performance, we introduce new metrics: NCR@2FOR and NCR@5FOR. Our approach achieves up to a 30% improvement in novel class accuracy on the CIFAR100 dataset (1-shot, 1 novel class) while maintaining a controlled base class forgetting rate of 2%. 

**Abstract (ZH)**: 在个人标注样本有限（少样本）的情况下，类增量学习对许多实际应用至关重要，如智能家居设备。在这些场景中，关键挑战在于平衡适应新个性化类和保持原始基础类性能之间的权衡。在新类上对模型进行微调通常会导致灾难性遗忘现象，即基础类的准确性会不可预测且显著地下降。在本文中，我们提出了一种简单而有效的方法来解决这一挑战，通过预先控制新类和基础类准确性之间的权衡。我们特别针对超少样本场景，其中每个新类仅有一个示例可用。我们的方法引入了一种新颖类检测（NCD）规则，该规则在增强新类性能的同时，预先调整遗忘的程度。我们通过将我们的解决方案应用于最先进的少样本类增量学习（FSCIL）方法，展示了其在不同设置下的泛化能力，显示出一致性的改进。为了更好地量化新类和基础类性能之间的权衡，我们引入了新的评价指标：NCR@2FOR 和 NCR@5FOR。我们的方法在 CIFAR100 数据集（1-shot，1 new class）的新类准确性上取得了最高 30% 的改进，同时保持基础类遗忘率的可控水平为 2%。 

---
# MultiPDENet: PDE-embedded Learning with Multi-time-stepping for Accelerated Flow Simulation 

**Title (ZH)**: 多时间步嵌入式PDE网络：加速流模拟的学习方法 

**Authors**: Qi Wang, Yuan Mi, Haoyun Wang, Yi Zhang, Ruizhi Chengze, Hongsheng Liu, Ji-Rong Wen, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.15987)  

**Abstract**: Solving partial differential equations (PDEs) by numerical methods meet computational cost challenge for getting the accurate solution since fine grids and small time steps are required. Machine learning can accelerate this process, but struggle with weak generalizability, interpretability, and data dependency, as well as suffer in long-term prediction. To this end, we propose a PDE-embedded network with multiscale time stepping (MultiPDENet), which fuses the scheme of numerical methods and machine learning, for accelerated simulation of flows. In particular, we design a convolutional filter based on the structure of finite difference stencils with a small number of parameters to optimize, which estimates the equivalent form of spatial derivative on a coarse grid to minimize the equation's residual. A Physics Block with a 4th-order Runge-Kutta integrator at the fine time scale is established that embeds the structure of PDEs to guide the prediction. To alleviate the curse of temporal error accumulation in long-term prediction, we introduce a multiscale time integration approach, where a neural network is used to correct the prediction error at a coarse time scale. Experiments across various PDE systems, including the Navier-Stokes equations, demonstrate that MultiPDENet can accurately predict long-term spatiotemporal dynamics, even given small and incomplete training data, e.g., spatiotemporally down-sampled datasets. MultiPDENet achieves the state-of-the-art performance compared with other neural baseline models, also with clear speedup compared to classical numerical methods. 

**Abstract (ZH)**: 通过数值方法求解偏微分方程（PDEs）虽然可以获得准确的解，但计算成本很高，需要细网格和小时间步长。机器学习可以加速这一过程，但面临泛化能力弱、可解释性差和数据依赖性的问题，尤其是在长期预测方面表现不佳。为了解决这些问题，我们提出了一种嵌入多尺度时间步长的PDE网络（MultiPDENet），将数值方法方案与机器学习融合，以加速流体模拟。特别地，我们设计了一个基于有限差分模板结构的卷积滤波器，通过优化少量参数来估计粗网格上的等效空间导数形式，以最小化方程的残差。我们构建了一个物理块，其中包含在细时间步长下使用四阶龙格库塔积分器，以嵌入PDE结构来引导预测。为了解决长期预测中时间误差累积的难题，我们引入了一种多尺度时间积分方法，其中在粗时间步长下使用神经网络来纠正预测误差。在涵盖纳维-斯托克斯方程等不同PDE系统的大规模实验中，MultiPDENet能够准确预测长时间的时空动态，即使训练数据量小且不完整，例如时空下采样的数据集也能表现良好。MultiPDENet在与其他神经基线模型比较中达到了最先进的性能，相较于经典数值方法也有明显的加速效果。 

---
# An Explainable Disease Surveillance System for Early Prediction of Multiple Chronic Diseases 

**Title (ZH)**: 可解释的疾病监测系统：用于多种慢性病早期预测 

**Authors**: Shaheer Ahmad Khan, Muhammad Usamah Shahid, Ahmad Abdullah, Ibrahim Hashmat, Muddassar Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2501.15969)  

**Abstract**: This study addresses a critical gap in the healthcare system by developing a clinically meaningful, practical, and explainable disease surveillance system for multiple chronic diseases, utilizing routine EHR data from multiple U.S. practices integrated with CureMD's EMR/EHR system. Unlike traditional systems--using AI models that rely on features from patients' labs--our approach focuses on routinely available data, such as medical history, vitals, diagnoses, and medications, to preemptively assess the risks of chronic diseases in the next year. We trained three distinct models for each chronic disease: prediction models that forecast the risk of a disease 3, 6, and 12 months before a potential diagnosis. We developed Random Forest models, which were internally validated using F1 scores and AUROC as performance metrics and further evaluated by a panel of expert physicians for clinical relevance based on inferences grounded in medical knowledge. Additionally, we discuss our implementation of integrating these models into a practical EMR system. Beyond using Shapley attributes and surrogate models for explainability, we also introduce a new rule-engineering framework to enhance the intrinsic explainability of Random Forests. 

**Abstract (ZH)**: 本研究通过开发一个临床有意义、实际可行且可解释的多慢性病监测系统，填补了医疗保健系统中的一个重要空白，该系统利用来自美国多个医疗机构的常规电子健康记录（EHR）数据，与CureMD的电子医疗记录/电子健康记录（EMR/EHR）系统集成。与传统的依赖患者实验室特征的AI模型不同，我们的方法专注于常规可用的数据，如医疗历史、生命体征、诊断和药物，以预估未来一年慢性病的风险。我们为每种慢性病分别训练了三个不同的模型：这些预测模型能在潜在诊断前3个月、6个月和12个月预估疾病的风险。我们开发了随机森林模型，并使用F1分数和AUROC作为性能指标进行内部验证，再通过一组专家医生的临床相关性评估，基于医学知识进行推断。此外，我们还讨论了将这些模型集成到实际的EMR系统中的实施方法。除了使用Shapley属性和代理模型来增强可解释性之外，我们还提出了一种新的规则工程框架，以增强随机森林的内在可解释性。 

---
# Multi-View Attention Syntactic Enhanced Graph Convolutional Network for Aspect-based Sentiment Analysis 

**Title (ZH)**: 基于多视图注意力句法增强的图卷积网络在方面级情感分析中的应用 

**Authors**: Xiang Huang, Hao Peng, Shuo Sun, Zhifeng Hao, Hui Lin, Shuhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15968)  

**Abstract**: Aspect-based Sentiment Analysis (ABSA) is the task aimed at predicting the sentiment polarity of aspect words within sentences. Recently, incorporating graph neural networks (GNNs) to capture additional syntactic structure information in the dependency tree derived from syntactic dependency parsing has been proven to be an effective paradigm for boosting ABSA. Despite GNNs enhancing model capability by fusing more types of information, most works only utilize a single topology view of the dependency tree or simply conflate different perspectives of information without distinction, which limits the model performance. To address these challenges, in this paper, we propose a new multi-view attention syntactic enhanced graph convolutional network (MASGCN) that weighs different syntactic information of views using attention mechanisms. Specifically, we first construct distance mask matrices from the dependency tree to obtain multiple subgraph views for GNNs. To aggregate features from different views, we propose a multi-view attention mechanism to calculate the attention weights of views. Furthermore, to incorporate more syntactic information, we fuse the dependency type information matrix into the adjacency matrices and present a structural entropy loss to learn the dependency type adjacency matrix. Comprehensive experiments on four benchmark datasets demonstrate that our model outperforms state-of-the-art methods. The codes and datasets are available at this https URL. 

**Abstract (ZH)**: 基于方面的情感分析（ABSAs）旨在预测句子中方面词的情感极性。近年来，将图神经网络（GNNs）引入句法依赖树中提取到的依赖树结构，以捕捉额外的句法结构信息，已被证明是提高ABSAs性能的有效范式。尽管GNNs通过融合多种信息提高了模型能力，但大多数工作仍只利用了依赖树的单一拓扑视图，或者简单地将不同视角的信息混在一起而不加以区分，这限制了模型性能。为了解决这些挑战，本文提出了一种新的多视角注意句法增强图卷积网络（MASGCN），通过注意机制加权不同的句法视图信息。具体来说，我们首先从依赖树构建距离掩码矩阵，以获得多个子图视图供GNN使用。为了从不同视图聚合特征，我们提出了多视角注意机制来计算视图的注意权重。此外，为了融合更多的句法信息，我们将依赖类型信息矩阵融合到邻接矩阵中，并提出了一种结构熵损失来学习依赖类型邻接矩阵。在四个基准数据集上的综合实验表明，我们的模型优于现有最先进的方法。相关代码和数据集可通过以下链接获取：this https URL。 

---
# Evaluating Data Influence in Meta Learning 

**Title (ZH)**: 评估元学习中的数据影响力 

**Authors**: Chenyang Ren, Huanyi Xie, Shu Yang, Meng Ding, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15963)  

**Abstract**: As one of the most fundamental models, meta learning aims to effectively address few-shot learning challenges. However, it still faces significant issues related to the training data, such as training inefficiencies due to numerous low-contribution tasks in large datasets and substantial noise from incorrect labels. Thus, training data attribution methods are needed for meta learning. However, the dual-layer structure of mata learning complicates the modeling of training data contributions because of the interdependent influence between meta-parameters and task-specific parameters, making existing data influence evaluation tools inapplicable or inaccurate. To address these challenges, based on the influence function, we propose a general data attribution evaluation framework for meta-learning within the bilevel optimization framework. Our approach introduces task influence functions (task-IF) and instance influence functions (instance-IF) to accurately assess the impact of specific tasks and individual data points in closed forms. This framework comprehensively models data contributions across both the inner and outer training processes, capturing the direct effects of data points on meta-parameters as well as their indirect influence through task-specific parameters. We also provide several strategies to enhance computational efficiency and scalability. Experimental results demonstrate the framework's effectiveness in training data evaluation via several downstream tasks. 

**Abstract (ZH)**: 作为最基础的模型之一，元学习旨在有效解决少样本学习的挑战。然而，元学习依然面临着与训练数据相关的一些重要问题，例如，在大数据集中有大量低贡献的任务导致的训练效率低下问题，以及来自错误标签的大量噪声问题。因此，对于元学习而言，需要采用训练数据归属方法。然而，元学习的双层结构使得训练数据贡献的建模变得复杂，因为元参数与任务特定参数之间存在相互依赖的影响，现有的数据影响评估工具在这种情况下变得不适用或不够准确。为了解决这些挑战，我们基于影响函数，在 bilevel 优化框架下提出了一种通用的元学习数据归属评估框架。我们的方法引入了任务影响函数（task-IF）和实例影响函数（instance-IF），以闭合形式精准评估特定任务和单个数据点的影响。该框架全面地模型了数据在内外训练过程中的贡献，捕捉了数据点对元参数的直接效应及其通过任务特定参数的间接影响。此外，我们还提供了一些策略以提高计算效率和可扩展性。实验结果表明，该框架在多个下游任务中有效地评估了训练数据。 

---
# Generative AI for Lyapunov Optimization Theory in UAV-based Low-Altitude Economy Networking 

**Title (ZH)**: 基于无人机的低空经济网络中拉普拉斯优化理论的生成式人工智能研究 

**Authors**: Zhang Liu, Dusit Niyato, Jiacheng Wang, Geng Sun, Lianfen Huang, Zhibin Gao, Xianbin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15928)  

**Abstract**: Lyapunov optimization theory has recently emerged as a powerful mathematical framework for solving complex stochastic optimization problems by transforming long-term objectives into a sequence of real-time short-term decisions while ensuring system stability. This theory is particularly valuable in unmanned aerial vehicle (UAV)-based low-altitude economy (LAE) networking scenarios, where it could effectively address inherent challenges of dynamic network conditions, multiple optimization objectives, and stability requirements. Recently, generative artificial intelligence (GenAI) has garnered significant attention for its unprecedented capability to generate diverse digital content. Extending beyond content generation, in this paper, we propose a framework integrating generative diffusion models with reinforcement learning to address Lyapunov optimization problems in UAV-based LAE networking. We begin by introducing the fundamentals of Lyapunov optimization theory and analyzing the limitations of both conventional methods and traditional AI-enabled approaches. We then examine various GenAI models and comprehensively analyze their potential contributions to Lyapunov optimization. Subsequently, we develop a Lyapunov-guided generative diffusion model-based reinforcement learning framework and validate its effectiveness through a UAV-based LAE networking case study. Finally, we outline several directions for future research. 

**Abstract (ZH)**: 李雅普诺夫优化理论近年来成为解决复杂随机优化问题的强大数学框架，通过将长期目标转化为一系列实时短期决策的方式，同时确保系统稳定性。该理论特别适用于基于无人机（UAV）的低空经济（LAE）网络场景，在这种场景中，它可以有效地应对动态网络条件、多重优化目标和稳定性要求等固有挑战。最近，生成式人工智能（GenAI）因其前所未有的内容生成能力而备受关注。超越内容生成，本文提出了一种将生成扩散模型与强化学习结合起来的框架，以解决基于UAV的LAE网络中的李雅普诺夫优化问题。首先，我们介绍了李雅普诺夫优化理论的基本原理，并分析了传统方法和传统AI使能方法的局限性。接着，我们探讨了各种GenAI模型，并对其在李雅普诺夫优化中的潜在贡献进行了全面分析。随后，我们开发了一种基于生成扩散模型引导的强化学习框架，并通过基于UAV的LAE网络案例研究验证了其有效性。最后，我们指出了未来研究的几个方向。 

---
# Online Housing Market 

**Title (ZH)**: 在线房地产市场 

**Authors**: Julien Lesca  

**Link**: [PDF](https://arxiv.org/pdf/2501.15916)  

**Abstract**: This paper studies an online variant of the celebrated housing market problem, where each agent has a single house and seeks to exchange it for another based on her preferences. In this online setting, agents may arrive and depart at any time, meaning that not all agents are present on the housing market simultaneously. I extend the well known serial dictatorship and Gale s top trading cycle mechanisms to this online scenario, aiming to retain their desirable properties such as Pareto efficiency, individual rationality, and strategy proofness. These extensions also seek to prevent agents from strategically delaying their arrival or advancing their departure. I demonstrate that achieving all of these properties simultaneously is impossible in the online context, and I present several variants that achieve different subsets of these properties. 

**Abstract (ZH)**: 本文研究了一个著名的住房市场问题的在线变体，其中每个代理人拥有一套住房，并希望根据自己的偏好将其交换为另一套住房。在这一在线环境中，代理人可能在任意时间到达或离开，这意味着并非所有代理人都会同时出现在住房市场上。本文将广为人知的顺序独裁制和格劳尔的顶尖交易循环机制扩展到这种在线环境中，旨在保留这些机制的 desirable 属性，如帕累托效率、个别理性性和策略证明性。这些扩展机制还旨在防止代理人战略性地延迟到达或提前离开。本文证明，在在线环境中同时实现所有这些属性是不可能的，并提出了几种实现不同属性组合的变体。 

---
# Evidential Physics-Informed Neural Networks 

**Title (ZH)**: 证据物理知情神经网络 

**Authors**: Hai Siong Tan, Kuancheng Wang, Rafe McBeth  

**Link**: [PDF](https://arxiv.org/pdf/2501.15908)  

**Abstract**: We present a novel class of Physics-Informed Neural Networks that is formulated based on the principles of Evidential Deep Learning, where the model incorporates uncertainty quantification by learning parameters of a higher-order distribution. The dependent and trainable variables of the PDE residual loss and data-fitting loss terms are recast as functions of the hyperparameters of an evidential prior distribution. Our model is equipped with an information-theoretic regularizer that contains the Kullback-Leibler divergence between two inverse-gamma distributions characterizing predictive uncertainty. Relative to Bayesian-Physics-Informed-Neural-Networks, our framework appeared to exhibit higher sensitivity to data noise, preserve boundary conditions more faithfully and yield empirical coverage probabilities closer to nominal ones. Toward examining its relevance for data mining in scientific discoveries, we demonstrate how to apply our model to inverse problems involving 1D and 2D nonlinear differential equations. 

**Abstract (ZH)**: 我们提出了一种新的物理知情神经网络类，该类基于证据深度学习的原则，通过学习高阶分布的参数来量化模型的不确定性。偏微分方程残差损失项和数据拟合损失项中的依赖和可训练变量被重塑为证据先验分布超参数的函数。我们的模型配备了一个信息论正则化项，该项包含两个逆伽马分布之间的克劳斯-莱布利尔（Kullback-Leibler）散度，用于表征预测不确定性。相较于贝叶斯物理知情神经网络，我们的框架对数据噪声的敏感性更高，更忠实于边界条件，并且给出了更接近名义值的经验置信度。为了验证其在科学研究中数据挖掘方面的相关性，我们展示了如何将该模型应用于涉及一维和二维非线性微分方程的逆问题。 

---
# A Data-Centric Approach: Dimensions of Visual Complexity and How to find Them 

**Title (ZH)**: 一种以数据为中心的方法：视觉复杂度的维度及其探测方法 

**Authors**: Karahan Sarıtaş, Tingke Shen, Surabhi S Nath, Peter Dayan  

**Link**: [PDF](https://arxiv.org/pdf/2501.15890)  

**Abstract**: Understanding how humans perceive visual complexity is a key area of study in visual cognition. Previous approaches to modeling visual complexity have often resulted in intricate, difficult-to-interpret solutions that employ numerous features or sophisticated deep learning architectures. While these complex models achieve high performance on specific datasets, they often sacrifice interpretability, making it challenging to understand the factors driving human perception of complexity. A recent model based on image segmentations showed promise in addressing this challenge; however, it presented limitations in capturing structural and semantic aspects of visual complexity. In this paper, we propose viable and effective features to overcome these shortcomings. Specifically, we develop multiscale features for the structural aspect of complexity, including the Multiscale Sobel Gradient (MSG), which captures spatial intensity variations across scales, and Multiscale Unique Colors (MUC), which quantifies image colorfulness by indexing quantized RGB values. We also introduce a new dataset SVG based on Visual Genome to explore the semantic aspect of visual complexity, obtaining surprise scores based on the element of surprise in images, which we demonstrate significantly contributes to perceived complexity. Overall, we suggest that the nature of the data is fundamental to understanding and modeling visual complexity, highlighting the importance of both structural and semantic dimensions in providing a comprehensive, interpretable assessment. The code for our analysis, experimental setup, and dataset will be made publicly available upon acceptance. 

**Abstract (ZH)**: 理解人类如何感知视觉复杂性是视觉认知领域的关键研究领域。以往对视觉复杂性的建模方法往往导致了结构复杂、难以理解的解决方案，这些方法通常依赖于多种特征或复杂的深度学习架构。尽管这些复杂模型在特定数据集上表现优异，但往往牺牲了可解释性，这使得理解驱动人类感知复杂性的因素变得困难。最近，基于图像分割的模型在这方面展现出了潜力，然而它在捕捉视觉复杂性的结构和语义方面存在局限性。在本文中，我们提出了一些可行且有效的特征来克服这些不足。具体来说，我们开发了多尺度特征来处理复杂性的结构方面，包括多尺度Sobel梯度（MSG），它可以捕捉不同尺度下的空间强度变化，以及多尺度独特颜色（MUC），它可以量化图像的颜色丰富度并通过量化RGB值进行索引。我们还引入了一个基于Visual Genome的新数据集SVG，以此来探索视觉复杂性的语义方面，通过计算图像中的惊讶分数，展示了这种震惊因素显著影响了对复杂性的感知。总体而言，我们认为数据的本质对于理解和建模视觉复杂性至关重要，强调结构和语义维度在提供全面且可解释的评估中的重要性。接受后，我们的分析代码、实验设置和数据集将公开发布。 

---
# Adaptive Width Neural Networks 

**Title (ZH)**: 自适应宽度神经网络 

**Authors**: Federico Errica, Henrik Christiansen, Viktor Zaverkin, Mathias Niepert, Francesco Alesiani  

**Link**: [PDF](https://arxiv.org/pdf/2501.15889)  

**Abstract**: For almost 70 years, researchers have mostly relied on hyper-parameter tuning to pick the width of neural networks' layers out of many possible choices. This paper challenges the status quo by introducing an easy-to-use technique to learn an unbounded width of a neural network's layer during training. The technique does not rely on alternate optimization nor hand-crafted gradient heuristics; rather, it jointly optimizes the width and the parameters of each layer via simple backpropagation. We apply the technique to a broad range of data domains such as tables, images, texts, and graphs, showing how the width adapts to the task's difficulty. By imposing a soft ordering of importance among neurons, it is possible to truncate the trained network at virtually zero cost, achieving a smooth trade-off between performance and compute resources in a structured way. Alternatively, one can dynamically compress the network with no performance degradation. In light of recent foundation models trained on large datasets, believed to require billions of parameters and where hyper-parameter tuning is unfeasible due to huge training costs, our approach stands as a viable alternative for width learning. 

**Abstract (ZH)**: 近70年来，研究者们主要依靠超参数调优来选择神经网络各层的宽度。本论文通过引入一种简洁易用的技术挑战了这一常规做法，该技术能够在训练过程中同时学习神经网络各层的无界宽度。该技术不依赖于交替优化或人工设计的梯度启发式方法，而是通过简单的反向传播同时优化宽度和每一层的参数。本文将该技术应用于表格、图像、文本和图形等多种数据领域，展示了宽度如何根据任务的难度进行调整。通过在神经元之间施加一个软优先级顺序，可以在几乎零成本的情况下截断训练的网络，以一种结构化的方式在性能和计算资源之间实现平滑的权衡。或者，可以动态压缩网络而不影响性能。鉴于近年来在大型数据集上训练的基础模型，这些模型被认为需要数十亿参数且由于高额的训练成本使得超参数调优不可行，我们的方法为宽度学习提供了一种可行的替代方案。 

---
# Boli: A dataset for understanding stuttering experience and analyzing stuttered speech 

**Title (ZH)**: 《Boli：一个用于理解 stuttering 经历和分析 stuttered speech 的数据集》

在翻译学术论文标题时，保持专业性和准确性非常重要。在这里，“stuttering experience”被翻译为“stuttering 经历”，“stuttered speech”被翻译为“stuttered speech”，这两个术语是专业术语，直接翻译为中文能够更好地保持其专业性。同时，考虑到中文句子的习惯，标题进行了适当的调整，使其更加通顺。 

**Authors**: Ashita Batra, Mannas narang, Neeraj Kumar Sharma, Pradip K Das  

**Link**: [PDF](https://arxiv.org/pdf/2501.15877)  

**Abstract**: There is a growing need for diverse, high-quality stuttered speech data, particularly in the context of Indian languages. This paper introduces Project Boli, a multi-lingual stuttered speech dataset designed to advance scientific understanding and technology development for individuals who stutter, particularly in India. The dataset constitutes (a) anonymized metadata (gender, age, country, mother tongue) and responses to a questionnaire about how stuttering affects their daily lives, (b) captures both read speech (using the Rainbow Passage) and spontaneous speech (through image description tasks) for each participant and (c) includes detailed annotations of five stutter types: blocks, prolongations, interjections, sound repetitions and word repetitions. We present a comprehensive analysis of the dataset, including the data collection procedure, experience summarization of people who stutter, severity assessment of stuttering events and technical validation of the collected data. The dataset is released as an open access to further speech technology development. 

**Abstract (ZH)**: 随着对多样化和高质量口吃语音数据需求的增长，特别是在印度语言的背景下，本研究提出了Project Boli项目，这是一个多语种口吃语音数据集，旨在促进对口吃个体的科学理解和技术创新，特别是在印度地区。该数据集包括：(a) 匿名元数据（性别、年龄、国籍、母语）以及有关口吃对其日常生活影响的问卷响应；(b) 捕捉每个参与者朗读语料（使用彩虹段落）和自发语料（通过图像描述任务）；(c) 包括对五种口吃类型的详细标注：阻塞、延展、插话、音素重复和词重复。我们对数据集进行了全面分析，包括数据收集过程、口吃者的经验总结、口吃事件严重程度评估以及收集数据的技术验证。该数据集作为开源发布，以促进语音技术的发展。 

---
# Optimizing Sentence Embedding with Pseudo-Labeling and Model Ensembles: A Hierarchical Framework for Enhanced NLP Tasks 

**Title (ZH)**: 基于伪标签和模型集成的分层框架：优化句子嵌入以增强NLP任务 

**Authors**: Ziwei Liu, Qi Zhang, Lifu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15876)  

**Abstract**: Sentence embedding tasks are important in natural language processing (NLP), but improving their performance while keeping them reliable is still hard. This paper presents a framework that combines pseudo-label generation and model ensemble techniques to improve sentence embeddings. We use external data from SimpleWiki, Wikipedia, and BookCorpus to make sure the training data is consistent. The framework includes a hierarchical model with an encoding layer, refinement layer, and ensemble prediction layer, using ALBERT-xxlarge, RoBERTa-large, and DeBERTa-large models. Cross-attention layers combine external context, and data augmentation techniques like synonym replacement and back-translation increase data variety. Experimental results show large improvements in accuracy and F1-score compared to basic models, and studies confirm that cross-attention and data augmentation make a difference. This work presents an effective way to improve sentence embedding tasks and lays the groundwork for future NLP research. 

**Abstract (ZH)**: 句向量嵌入任务在自然语言处理（NLP）中十分重要，但在提高其性能的同时保持其可靠性仍然是一个挑战。本文提出了一种结合伪标签生成和模型集成技术的框架，以改进句向量嵌入。我们使用来自SimpleWiki、Wikipedia和BookCorpus的外部数据，确保训练数据的一致性。该框架包括一个分层模型，包含编码层、修正层和集成预测层，使用了ALBERT-xxlarge、RoBERTa-large和DeBERTa-large模型。交叉注意力层结合了外部语境，数据增强技术（如同义词替换和反向翻译）增加了数据多样性。实验结果表明，与基础模型相比，准确率和F1分数有了显著提升，并且研究表明交叉注意力和数据增强对于提高性能至关重要。本项研究提供了一种有效的方法来改进句向量嵌入任务，并为此后的NLP研究奠定了基础。 

---
# D-PLS: Decoupled Semantic Segmentation for 4D-Panoptic-LiDAR-Segmentation 

**Title (ZH)**: D-PLS：解耦语义分割在4D-全景LiDAR分割中的应用 

**Authors**: Maik Steinhauser, Laurenz Reichardt, Nikolas Ebert, Oliver Wasenmüller  

**Link**: [PDF](https://arxiv.org/pdf/2501.15870)  

**Abstract**: This paper introduces a novel approach to 4D Panoptic LiDAR Segmentation that decouples semantic and instance segmentation, leveraging single-scan semantic predictions as prior information for instance segmentation. Our method D-PLS first performs single-scan semantic segmentation and aggregates the results over time, using them to guide instance segmentation. The modular design of D-PLS allows for seamless integration on top of any semantic segmentation architecture, without requiring architectural changes or retraining. We evaluate our approach on the SemanticKITTI dataset, where it demonstrates significant improvements over the baseline in both classification and association tasks, as measured by the LiDAR Segmentation and Tracking Quality (LSTQ) metric. Furthermore, we show that our decoupled architecture not only enhances instance prediction but also surpasses the baseline due to advancements in single-scan semantic segmentation. 

**Abstract (ZH)**: 本文介绍了一种新颖的4D全景LiDAR分割方法，该方法将语义分割和实例分割解耦，并利用单扫描语义预测作为实例分割的先验信息。我们的方法D-PLS首先执行单扫描语义分割，并在时间上聚集结果，然后利用这些结果指导实例分割。D-PLS的模块化设计使其可以无缝集成到任何语义分割架构之上，无需进行架构更改或重新训练。我们在SemanticKITTI数据集上评估了该方法，结果显示该方法在分类和关联任务中均显著优于基线方法，这一结果由LiDAR分割和跟踪质量（LSTQ）指标衡量。此外，我们还展示了我们解耦的架构不仅提高了实例预测的性能，而且由于单扫描语义分割的进步，还超越了基线方法。 

---
# Transfer of Knowledge through Reverse Annealing: A Preliminary Analysis of the Benefits and What to Share 

**Title (ZH)**: 通过逆退热处理转移知识：初步分析其益处及需共享的内容 

**Authors**: Eneko Osaba, Esther Villar-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2501.15865)  

**Abstract**: Being immersed in the NISQ-era, current quantum annealers present limitations for solving optimization problems efficiently. To mitigate these limitations, D-Wave Systems developed a mechanism called Reverse Annealing, a specific type of quantum annealing designed to perform local refinement of good states found elsewhere. Despite the research activity around Reverse Annealing, none has theorized about the possible benefits related to the transfer of knowledge under this paradigm. This work moves in that direction and is driven by experimentation focused on answering two key research questions: i) is reverse annealing a paradigm that can benefit from knowledge transfer between similar problems? and ii) can we infer the characteristics that an input solution should meet to help increase the probability of success? To properly guide the tests in this paper, the well-known Knapsack Problem has been chosen for benchmarking purposes, using a total of 34 instances composed of 14 and 16 items. 

**Abstract (ZH)**: 置身于NISQ时代，现有的量子退火机在高效解决优化问题方面存在局限性。为缓解这些局限性，D-Wave Systems 开发了一种名为反向退火的机制，这是一种特定类型的量子退火，旨在对在其他地方找到的良好状态进行局部细化。尽管有关反向退火的研究活动较多，但尚未有人从知识转移的角度对其可能带来的益处进行理论探讨。本研究正是朝着这个方向进行，并以实验为主要研究手段，旨在回答两个核心研究问题：i) 反向退火是否是一种可以从类似问题的知识转移中受益的范式？ii) 我们能否推断出输入解应具备的特征，以提高成功概率？为合理指导本论文中的测试，选择了著名的背包问题作为基准测试，总共使用了34个实例，其中包含14个和16个物品。 

---
# Beyond In-Distribution Performance: A Cross-Dataset Study of Trajectory Prediction Robustness 

**Title (ZH)**: 超越分布内性能：跨数据集的轨迹预测鲁棒性研究 

**Authors**: Yue Yao, Daniel Goehring, Joerg Reichardt  

**Link**: [PDF](https://arxiv.org/pdf/2501.15842)  

**Abstract**: We study the Out-of-Distribution (OoD) generalization ability of three SotA trajectory prediction models with comparable In-Distribution (ID) performance but different model designs. We investigate the influence of inductive bias, size of training data and data augmentation strategy by training the models on Argoverse 2 (A2) and testing on Waymo Open Motion (WO) and vice versa. We find that the smallest model with highest inductive bias exhibits the best OoD generalization across different augmentation strategies when trained on the smaller A2 dataset and tested on the large WO dataset. In the converse setting, training all models on the larger WO dataset and testing on the smaller A2 dataset, we find that all models generalize poorly, even though the model with the highest inductive bias still exhibits the best generalization ability. We discuss possible reasons for this surprising finding and draw conclusions about the design and test of trajectory prediction models and benchmarks. 

**Abstract (ZH)**: 我们研究了三种在分布内（In-Distribution, ID）性能相当但具有不同模型设计的当前最佳（State-of-the-Art, SotA）轨迹预测模型的离分布外（Out-of-Distribution, OoD）泛化能力。我们通过在Argoverse 2（A2）数据集上训练模型并在Waymo Open Motion（WO）数据集上测试，以及相反的设置，分别考察了归纳偏置、训练数据量和数据增强策略对模型OoD泛化能力的影响。我们发现，最小的模型但具有最高的归纳偏置，在较小的A2数据集上训练并在较大的WO数据集上测试时，表现出最佳的OoD泛化能力，不论使用哪种数据增强策略。在另一方面，当在较大的WO数据集上训练所有模型并在较小的A2数据集上测试时，所有模型的泛化能力都很差，即使具有最高归纳偏置的模型依旧表现最好。我们探讨了这一出乎意料的结果背后可能的原因，并得出了关于轨迹预测模型和基准设计与测试的结论。 

---
# CrySPAI: A new Crystal Structure Prediction Software Based on Artificial Intelligence 

**Title (ZH)**: CrySPAI：一种基于人工智能的新型晶体结构预测软件 

**Authors**: Zongguo Wang, Ziyi Chen, Yang Yuan, Yangang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15838)  

**Abstract**: Crystal structure predictions based on the combination of first-principles calculations and machine learning have achieved significant success in materials science. However, most of these approaches are limited to predicting specific systems, which hinders their application to unknown or unexplored domains. In this paper, we present CrySPAI, a crystal structure prediction package developed using artificial intelligence (AI) to predict energetically stable crystal structures of inorganic materials given their chemical compositions. The software consists of three key modules, an evolutionary optimization algorithm (EOA) that searches for all possible crystal structure configurations, density functional theory (DFT) that provides the accurate energy values for these structures, and a deep neural network (DNN) that learns the relationship between crystal structures and their corresponding energies. To optimize the process across these modules, a distributed framework is implemented to parallelize tasks, and an automated workflow has been integrated into CrySPAI for seamless execution. This paper reports the development and implementation of AI AI-based CrySPAI Crystal Prediction Software tool and its unique features. 

**Abstract (ZH)**: 基于第一性原理计算和机器学习相结合的晶体结构预测已经在材料科学中取得了显著的成功。然而，目前大多数方法仅限于预测特定系统，这限制了其在未知或未探索领域中的应用。本文介绍了使用人工智能（AI）开发的一款名为CrySPAI的晶体结构预测软件包，该软件可以给定化学组成后预测无机材料的 energetically stable 晶体结构。该软件由三个关键模块组成：进化优化算法（EOA），用于搜索所有可能的晶体结构配置；基于密度泛函理论（DFT）提供这些结构的准确能量值；以及深度神经网络（DNN），用于学习晶体结构与其相应能量之间的关系。为了优化这些模块之间的过程，实现并行化任务的分布式框架，并将自动化工作流程集成到CrySPAI中，以实现无缝执行。本文报告了基于AI的CrySPAI晶体预测软件工具的开发与实现以及其独特的功能。 

---
# Intelligent Code Embedding Framework for High-Precision Ransomware Detection via Multimodal Execution Path Analysis 

**Title (ZH)**: 基于多模态执行路径分析的高精度勒索软件检测智能代码嵌入框架 

**Authors**: Levi Gareth, Maximilian Fairbrother, Peregrine Blackwood, Lucasta Underhill, Benedict Ruthermore  

**Link**: [PDF](https://arxiv.org/pdf/2501.15836)  

**Abstract**: Modern threat landscapes continue to evolve with increasing sophistication, challenging traditional detection methodologies and necessitating innovative solutions capable of addressing complex adversarial tactics. A novel framework was developed to identify ransomware activity through multimodal execution path analysis, integrating high-dimensional embeddings and dynamic heuristic derivation mechanisms to capture behavioral patterns across diverse attack variants. The approach demonstrated high adaptability, effectively mitigating obfuscation strategies and polymorphic characteristics often employed by ransomware families to evade detection. Comprehensive experimental evaluations revealed significant advancements in precision, recall, and accuracy metrics compared to baseline techniques, particularly under conditions of variable encryption speeds and obfuscated execution flows. The framework achieved scalable and computationally efficient performance, ensuring robust applicability across a range of system configurations, from resource-constrained environments to high-performance infrastructures. Notable findings included reduced false positive rates and enhanced detection latency, even for ransomware families employing sophisticated encryption mechanisms. The modular design allowed seamless integration of additional modalities, enabling extensibility and future-proofing against emerging threat vectors. Quantitative analyses further highlighted the system's energy efficiency, emphasizing its practicality for deployment in environments with stringent operational constraints. The results underline the importance of integrating advanced computational techniques and dynamic adaptability to safeguard digital ecosystems from increasingly complex threats. 

**Abstract (ZH)**: 现代威胁场景持续进化，日益复杂，这不仅挑战了传统的检测方法，还迫切需要创新的解决方案来应对复杂的对抗性战术。为此，我们提出了一种新的框架，用于通过多模态执行路径分析来识别勒索软件活动。该框架集成了高维嵌入和动态启发式衍生机制，以便在各种攻击变种中捕捉行为模式。研究结果表明，该方法具有高度的适应性，能够有效应对勒索软件家族常用的混淆技术及其多态特性以逃避检测。全面的实验评估表明，与基线技术相比，在不同加密速度和混淆执行流条件下，该框架在精准度、召回率和准确性等方面有了显著提升。框架实现了可扩展且计算高效的性能，确保在从资源受限环境到高性能基础设施的广泛系统配置下具有稳健的适用性。研究发现，该框架的误报率降低，检测延迟显著缩短，即使在采用复杂加密机制的勒索软件家族中也是如此。模块化设计使其易于集成额外的数据模态，从而增强了未来对抗新兴威胁的能力。定量分析进一步突显了该系统的能源效率，强调了在具有严格操作约束的环境中部署其实用性。研究结果进一步强调了集成先进的计算技术和动态适应性对于保护日益复杂的数字生态系统的重要性。 

---
# SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model 

**Title (ZH)**: SpatialVLA：探索空间表示在视觉-语言-行动模型中的应用 

**Authors**: Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.15830)  

**Abstract**: In this paper, we claim that spatial understanding is the keypoint in robot manipulation, and propose SpatialVLA to explore effective spatial representations for the robot foundation model. Specifically, we introduce Ego3D Position Encoding to inject 3D information into the input observations of the visual-language-action model, and propose Adaptive Action Grids to represent spatial robot movement actions with adaptive discretized action grids, facilitating learning generalizable and transferrable spatial action knowledge for cross-robot control. SpatialVLA is first pre-trained on top of a vision-language model with 1.1 Million real-world robot episodes, to learn a generalist manipulation policy across multiple robot environments and tasks. After pre-training, SpatialVLA is directly applied to perform numerous tasks in a zero-shot manner. The superior results in both simulation and real-world robots demonstrate its advantage of inferring complex robot motion trajectories and its strong in-domain multi-task generalization ability. We further show the proposed Adaptive Action Grids offer a new and effective way to fine-tune the pre-trained SpatialVLA model for new simulation and real-world setups, where the pre-learned action grids are re-discretized to capture robot-specific spatial action movements of new setups. The superior results from extensive evaluations demonstrate the exceptional in-distribution generalization and out-of-distribution adaptation capability, highlighting the crucial benefit of the proposed spatial-aware representations for generalist robot policy learning. All the details and codes will be open-sourced. 

**Abstract (ZH)**: 在这篇论文中，我们主张空间理解是机器人操作的关键，并提出SpatialVLA以探索适用于机器人基础模型的有效空间表示。具体来说，我们引入了Ego3D位置编码，将3D信息注入视觉-语言-动作模型的输入观察中，并提出了自适应动作网格来用自适应离散的动作网格表示空间机器人运动动作，从而促进跨机器人控制的一般化和可转移的空间动作知识学习。SpatialVLA首先在包含110万真实世界机器人经历的视觉语言模型上进行预训练，以学习跨多个机器人环境和任务的一般机器人操作策略。在预训练后，SpatialVLA可以直接应用于在零样本情况下执行众多任务。在仿真实验和真实世界机器人中的优越结果证明了其推断复杂机器人运动轨迹的优势以及其强大的领域内多任务一般化能力。我们进一步展示了提出的自适应动作网格提供了一种新的有效方法，用于微调预训练的SpatialVLA模型以适应新的仿真实验和真实世界设置，其中预学习的动作网格被重新离散化以捕捉新设置中的机器人特定的空间动作。广泛的评估结果表明了其在域内一般化和域外适应方面的出色能力，突显了所提出的空间意识表示对通用机器人策略学习的关键益处。所有细节和代码都将开源。 

---
# FuzzyLight: A Robust Two-Stage Fuzzy Approach for Traffic Signal Control Works in Real Cities 

**Title (ZH)**: FuzzyLight：一种适用于真实城市交通信号控制的稳健两阶段模糊方法 

**Authors**: Mingyuan Li, Jiahao Wang, Bo Du, Jun Shen, Qiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15820)  

**Abstract**: Effective traffic signal control (TSC) is crucial in mitigating urban congestion and reducing emissions. Recently, reinforcement learning (RL) has been the research trend for TSC. However, existing RL algorithms face several real-world challenges that hinder their practical deployment in TSC: (1) Sensor accuracy deteriorates with increased sensor detection range, and data transmission is prone to noise, potentially resulting in unsafe TSC decisions. (2) During the training of online RL, interactions with the environment could be unstable, potentially leading to inappropriate traffic signal phase (TSP) selection and traffic congestion. (3) Most current TSC algorithms focus only on TSP decisions, overlooking the critical aspect of phase duration, affecting safety and efficiency. To overcome these challenges, we propose a robust two-stage fuzzy approach called FuzzyLight, which integrates compressed sensing and RL for TSC deployment. FuzzyLight offers several key contributions: (1) It employs fuzzy logic and compressed sensing to address sensor noise and enhances the efficiency of TSP decisions. (2) It maintains stable performance during training and combines fuzzy logic with RL to generate precise phases. (3) It works in real cities across 22 intersections and demonstrates superior performance in both real-world and simulated environments. Experimental results indicate that FuzzyLight enhances traffic efficiency by 48% compared to expert-designed timings in the real world. Furthermore, it achieves state-of-the-art (SOTA) performance in simulated environments using six real-world datasets with transmission noise. The code and deployment video are available at the URL1 

**Abstract (ZH)**: 有效的交通信号控制（TSC）对于缓解城市拥堵和减少排放至关重要。近年来，强化学习（RL）已成为TSC研究的热门趋势。然而，现有的RL算法面临着若干实际挑战，这些挑战阻碍了它们在TSC中的实际应用：（1）传感器精度随着检测范围的增加而降低，数据传输容易受到噪声的影响，可能导致不安全的TSC决策。（2）在线RL训练过程中，与环境的交互可能不稳定，这可能导致不恰当的交通信号相位（TSP）选择和交通拥堵。（3）目前大多数TSC算法仅关注TSP决策，忽视了相位持续时间这一关键方面，这影响了安全性和效率。为了克服这些挑战，我们提出了一种鲁棒的两级模糊方法，称为FuzzyLight，该方法结合了压缩感知和强化学习进行TSC部署。FuzzyLight提供了几项关键贡献：（1）它使用模糊逻辑和压缩感知解决传感器噪声问题，并提高TSP决策的效率。（2）它在训练过程中保持稳定性能，并结合模糊逻辑和强化学习生成精确的相位。（3）它在22个实际交叉路口运行，并在实际和仿真环境中均表现出色。实验结果表明，与专家设计的时机相比，FuzzyLight在实际环境中提高了交通效率48%。（4）使用六个包含传输噪声的真实世界数据集，在仿真环境中实现了最先进的（SOTA）性能。完整的代码和部署视频可在URL1处获得。 

---
# Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System 

**Title (ZH)**: 长期兴趣时钟：流式推荐系统中的细粒度时间感知 

**Authors**: Yongchun Zhu, Guanyu Jiang, Jingwu Chen, Feng Zhang, Xiao Yang, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15817)  

**Abstract**: User interests manifest a dynamic pattern within the course of a day, e.g., a user usually favors soft music at 8 a.m. but may turn to ambient music at 10 p.m. To model dynamic interests in a day, hour embedding is widely used in traditional daily-trained industrial recommendation systems. However, its discreteness can cause periodical online patterns and instability in recent streaming recommendation systems. Recently, Interest Clock has achieved remarkable performance in streaming recommendation systems. Nevertheless, it models users' dynamic interests in a coarse-grained manner, merely encoding users' discrete interests of 24 hours from short-term behaviors. In this paper, we propose a fine-grained method for perceiving time information for streaming recommendation systems, named Long-term Interest Clock (LIC). The key idea of LIC is adaptively calculating current user interests by taking into consideration the relevance of long-term behaviors around current time (e.g., 8 a.m.) given a candidate item. LIC consists of two modules: (1) Clock-GSU retrieves a sub-sequence by searching through long-term behaviors, using query information from a candidate item and current time, (2) Clock-ESU employs a time-gap-aware attention mechanism to aggregate sub-sequence with the candidate item. With Clock-GSU and Clock-ESU, LIC is capable of capturing users' dynamic fine-grained interests from long-term behaviors. We conduct online A/B tests, obtaining +0.122% improvements on user active days. Besides, the extended offline experiments show improvements as well. Long-term Interest Clock has been integrated into Douyin Music App's recommendation system. 

**Abstract (ZH)**: 用户兴趣在一天中的表现呈现出动态模式，例如，用户通常在早上8点更喜欢轻音乐，但在晚上10点可能会转而选择环境音乐。为了在一天中建模动态兴趣，传统的日训练工业推荐系统广泛使用小时嵌入。然而，其离散性会导致流式推荐系统中的周期性在线模式和不稳定性。最近，Interest Clock 在流式推荐系统中取得了显著性能。不过，它以粗粒度的方式建模用户的动态兴趣，仅通过短期行为对24小时内用户的离散兴趣进行编码。在本文中，我们提出了一种细粒度的方法，用于流式推荐系统感知时间信息，名为长期兴趣时钟（Long-term Interest Clock, LIS）。LIS 方法的核心思想是通过考虑到给定候选项目的当前时间周围长期行为的相关性，适应性地计算当前用户兴趣。LIS 包含两个模块：（1）时钟-GSU 通过搜索长期行为，利用候选项目和当前时间的查询信息检索子序列；（2）时钟-ESU 使用带有时隙意识的注意力机制来聚合候选项目与子序列。借助时钟-GSU 和时钟-ESU，LIS 能够从长期行为中捕捉用户的动态细粒度兴趣。我们进行了在线 A/B 测试，获得了用户活跃天数增加 0.122% 的改进。此外，扩展的离线实验也显示出改进效果。长期兴趣时钟已被集成到抖音音乐应用的推荐系统中。 

---
# AdaF^2M^2: Comprehensive Learning and Responsive Leveraging Features in Recommendation System 

**Title (ZH)**: AdaF^2M^2：全面学习和响应性利用特征的推荐系统 

**Authors**: Yongchun Zhu, Jingwu Chen, Ling Chen, Yitan Li, Feng Zhang, Xiao Yang, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15816)  

**Abstract**: Feature modeling, which involves feature representation learning and leveraging, plays an essential role in industrial recommendation systems. However, the data distribution in real-world applications usually follows a highly skewed long-tail pattern due to the popularity bias, which easily leads to over-reliance on ID-based features, such as user/item IDs and ID sequences of interactions. Such over-reliance makes it hard for models to learn features comprehensively, especially for those non-ID meta features, e.g., user/item characteristics. Further, it limits the feature leveraging ability in models, getting less generalized and more susceptible to data noise. Previous studies on feature modeling focus on feature extraction and interaction, hardly noticing the problems brought about by the long-tail data distribution. To achieve better feature representation learning and leveraging on real-world data, we propose a model-agnostic framework AdaF^2M^2, short for Adaptive Feature Modeling with Feature Mask. The feature-mask mechanism helps comprehensive feature learning via multi-forward training with augmented samples, while the adapter applies adaptive weights on features responsive to different user/item states. By arming base models with AdaF^2M^2, we conduct online A/B tests on multiple recommendation scenarios, obtaining +1.37% and +1.89% cumulative improvements on user active days and app duration respectively. Besides, the extended offline experiments on different models show improvements as well. AdaF$^2$M$^2$ has been widely deployed on both retrieval and ranking tasks in multiple applications of Douyin Group, indicating its superior effectiveness and universality. 

**Abstract (ZH)**: 特征建模，涉及特征表示学习及其利用，是工业推荐系统中的重要组成部分。然而，实际应用场景中的数据分布通常遵循高度倾斜的长尾模式，由于流行性偏见，这容易导致模型过度依赖ID基特征，如用户/项ID和交互的ID序列。这种过度依赖使模型难以全面学习特征，尤其是对于非ID元特征，例如用户/项特征。此外，这限制了模型的特征利用能力，使其通用性降低且更容易受到数据噪声的影响。以往关于特征建模的研究主要集中在特征提取和交互上，很少关注长尾数据分布带来的问题。为了在实际数据中实现更好的特征表示学习及其利用，我们提出了一个模型无关框架AdaF^2M^2，即自适应特征建模与特征屏蔽（Adaptative Feature Modeling with Feature Mask）的缩写。该框架中的特征屏蔽机制通过使用增强样本的多前向训练实现全面的特征学习，而适配器则根据用户/项的不同状态对特征应用可适应的权重。通过为基模型配备AdaF^2M^2，我们在多个推荐场景中进行了在线A/B测试，分别在用户活跃天数和应用持续时间上取得了1.37%和1.89%的累计改进。此外，在不同模型的扩展离线实验中也展示了改进效果。AdaF^2M^2已在抖音集团的多个应用中的检索和排序任务中广泛部署，表明其具有优越的有效性和通用性。 

---
# Adaptive AI-based Decentralized Resource Management in the Cloud-Edge Continuum 

**Title (ZH)**: 基于云边 continuum 的自适应人工智能驱动的去中心化资源管理 

**Authors**: Lanpei Li, Jack Bell, Massimo Coppola, Vincenzo Lomonaco  

**Link**: [PDF](https://arxiv.org/pdf/2501.15802)  

**Abstract**: The increasing complexity of application requirements and the dynamic nature of the Cloud-Edge Continuum present significant challenges for efficient resource management. These challenges stem from the ever-changing infrastructure, which is characterized by additions, removals, and reconfigurations of nodes and links, as well as the variability of application workloads. Traditional centralized approaches struggle to adapt to these changes due to their static nature, while decentralized solutions face challenges such as limited global visibility and coordination overhead. This paper proposes a hybrid decentralized framework for dynamic application placement and resource management. The framework utilizes Graph Neural Networks (GNNs) to embed resource and application states, enabling comprehensive representation and efficient decision-making. It employs a collaborative multi-agent reinforcement learning (MARL) approach, where local agents optimize resource management in their neighborhoods and a global orchestrator ensures system-wide coordination. By combining decentralized application placement with centralized oversight, our framework addresses the scalability, adaptability, and accuracy challenges inherent in the Cloud-Edge Continuum. This work contributes to the development of decentralized application placement strategies, the integration of GNN embeddings, and collaborative MARL systems, providing a foundation for efficient, adaptive and scalable resource management. 

**Abstract (ZH)**: 随着应用程序需求的日益复杂和云-边连续体的动态性质，有效地进行资源管理面临着重大挑战。这些挑战源于不断变化的基础架构，包括节点和链路的增加、删除和重构，以及应用程序工作负载的波动性。传统的集中式方法因其静态特性难以适应这些变化，而分散式解决方案则面临如全局可见性有限和协调开销高等挑战。本文提出了一种适用于动态应用部署和资源管理的混合分散式框架。该框架利用图神经网络（GNNs）嵌入资源和应用程序状态，实现了全面的表示和高效的决策。该框架采用协作多智能体强化学习（MARL）方法，其中局部智能体在其邻域内优化资源管理，而全局协调器则确保系统的整体协调。通过结合分散式应用部署和集中式监督，我们的框架解决了云-边连续体中固有的可扩展性、适应性和准确性挑战。本文为分散式应用部署策略的发展、GNN嵌入的集成以及协同MARL系统的构建提供了基础，提供了高效、适应性和可扩展资源管理的基础框架。 

---
# Large Language Models to Diffusion Finetuning 

**Title (ZH)**: 大规模语言模型应用于扩散微调 

**Authors**: Edoardo Cetin, Tianyu Zhao, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15781)  

**Abstract**: We propose a new finetuning method to provide pre-trained large language models (LMs) the ability to scale test-time compute through the diffusion framework. By increasing the number of diffusion steps, we show our finetuned models achieve monotonically increasing accuracy, directly translating to improved performance across downstream tasks. Furthermore, our finetuned models can expertly answer questions on specific topics by integrating powerful guidance techniques, and autonomously determine the compute required for a given problem by leveraging adaptive ODE solvers. Our method is universally applicable to any foundation model pre-trained with a cross-entropy loss and does not modify any of its original weights, fully preserving its strong single-step generation capabilities. We show our method is more effective and fully compatible with traditional finetuning approaches, introducing an orthogonal new direction to unify the strengths of the autoregressive and diffusion frameworks. 

**Abstract (ZH)**: 我们提出了一种新的微调方法，以使预训练的大语言模型（LMs）能够通过扩散框架扩展测试时的计算能力。通过增加扩散步骤的数量，我们展示了我们的微调模型在准确率上呈现单调增加的趋势，从而在下游任务中实现性能的提升。此外，我们的微调模型可以利用强大的引导技术，专家般地回答特定主题的问题，并通过利用自适应常微分方程求解器自主确定给定问题所需的计算量。我们的方法适用于任何使用交叉熵损失预训练的基础模型，并未修改其原始权重，从而完全保留了其强大的单步生成能力。我们证明，该方法比传统的微调方法更有效，并且完全兼容传统的微调方法，为统一自回归框架和扩散框架的优势提供了新的独立方向。 

---
# Formal Verification of Markov Processes with Learned Parameters 

**Title (ZH)**: 带有学习参数的马尔可夫过程的形式化验证 

**Authors**: Muhammad Maaz, Timothy C. Y. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2501.15767)  

**Abstract**: We introduce the problem of formally verifying properties of Markov processes where the parameters are the output of machine learning models. Our formulation is general and solves a wide range of problems, including verifying properties of probabilistic programs that use machine learning, and subgroup analysis in healthcare modeling. We show that for a broad class of machine learning models, including linear models, tree-based models, and neural networks, verifying properties of Markov chains like reachability, hitting time, and total reward can be formulated as a bilinear program. We develop a decomposition and bound propagation scheme for solving the bilinear program and show through computational experiments that our method solves the problem to global optimality up to 100x faster than state-of-the-art solvers. We also release $\texttt{markovml}$, an open-source tool for building Markov processes, integrating pretrained machine learning models, and verifying their properties, available at this https URL. 

**Abstract (ZH)**: 我们介绍了一个问题，即正式验证具有机器学习模型输出参数的马尔可夫过程的性质。我们的形式化描述是广泛的，并可以解决一系列问题，包括验证使用机器学习的概率程序的性质，以及医疗保健建模中的亚群体分析。我们证明，对于一类广泛的机器学习模型，包括线性模型、树基模型和神经网络，验证马尔可夫链的性质如可达性、接触时间以及总奖励，都可以形式化为双线性规划问题。我们开发了一种分治和界传播方案来求解双线性规划问题，并通过计算实验表明，我们的方法与最先进的求解器相比，可将问题求解速度提升高达100倍。此外，我们还发布了开源工具$\texttt{markovml}$，用于构建马尔可夫过程、集成预训练的机器学习模型及其性质验证，可在以下链接获取：this https URL。 

---
# Efficiency Bottlenecks of Convolutional Kolmogorov-Arnold Networks: A Comprehensive Scrutiny with ImageNet, AlexNet, LeNet and Tabular Classification 

**Title (ZH)**: 卷积科莫朵夫-阿诺德网络的效率瓶颈：基于ImageNet、AlexNet、LeNet和表格分类的全面审查 

**Authors**: Ashim Dahal, Saydul Akbar Murad, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2501.15757)  

**Abstract**: Algorithmic level developments like Convolutional Neural Networks, transformers, attention mechanism, Retrieval Augmented Generation and so on have changed Artificial Intelligence. Recent such development was observed by Kolmogorov-Arnold Networks that suggested to challenge the fundamental concept of a Neural Network, thus change Multilayer Perceptron, and Convolutional Neural Networks. They received a good reception in terms of scientific modeling, yet had some drawbacks in terms of efficiency. In this paper, we train Convolutional Kolmogorov Arnold Networks (CKANs) with the ImageNet-1k dataset with 1.3 million images, MNIST dataset with 60k images and a tabular biological science related MoA dataset and test the promise of CKANs in terms of FLOPS, Inference Time, number of trainable parameters and training time against the accuracy, precision, recall and f-1 score they produce against the standard industry practice on CNN models. We show that the CKANs perform fair yet slower than CNNs in small size dataset like MoA and MNIST but are not nearly comparable as the dataset gets larger and more complex like the ImageNet. The code implementation of this paper can be found on the link: \href{this https URL}{this https URL} 

**Abstract (ZH)**: 算法层面的发展，如卷积神经网络（Convolutional Neural Networks, CNNs）、变压器（transformers）、注意力机制（attention mechanism）、检索增强生成（retrieval augmented generation）等，已经改变了人工智能。近期，Kolmogorov-Arnold网络提出了一种挑战神经网络基本概念的方法，从而改变了多层感知机（Multilayer Perceptron, MLP）和卷积神经网络。尽管在科学建模方面收到好评，但在效率方面仍存在一些问题。在本文中，我们使用包含130万张图像的ImageNet-1k数据集、包含6万张图像的MNIST数据集，以及一个与生物科学相关的机制（Mechanism of Action, MoA）表格数据集，训练卷积Kolmogorov-Arnold网络（Convolutional Kolmogorov Arnold Networks, CKANs），并测试CKANs在FLOPS、推理时间、可训练参数数量和训练时间等方面的性能，这些性能与标准工业实践中的CNN模型在准确率、精确率、召回率和F1分数方面的表现进行对比。结果显示，在MoA和MNIST等小型数据集上，CKANs表现尚可但速度较慢，但在图像规模更大、更复杂的ImageNet数据集上，CKANs的表现则远不如CNNs。本文的代码实现可以在以下链接找到：\href{this https URL}{this https URL} 

---
# IndicMMLU-Pro: Benchmarking the Indic Large Language Models 

**Title (ZH)**: IndicMMLU-Pro: 评估印度语言大型语言模型 

**Authors**: Sankalp KJ, Ashutosh Kumar, Laxmaan Balaji, Nikunj Kotecha, Vinija Jain, Aman Chadha, Sreyoshi Bhaduri  

**Link**: [PDF](https://arxiv.org/pdf/2501.15747)  

**Abstract**: Known by more than 1.5 billion people in the Indian subcontinent, Indic languages present unique challenges and opportunities for natural language processing (NLP) research due to their rich cultural heritage, linguistic diversity, and complex structures. IndicMMLU-Pro is a comprehensive benchmark designed to evaluate Large Language Models (LLMs) across Indic languages, building upon the MMLU Pro (Massive Multitask Language Understanding) framework. Covering major languages such as Hindi, Bengali, Gujarati, Marathi, Kannada, Punjabi, Tamil, Telugu, and Urdu, our benchmark addresses the unique challenges and opportunities presented by the linguistic diversity of the Indian subcontinent. This benchmark encompasses a wide range of tasks in language comprehension, reasoning, and generation, meticulously crafted to capture the intricacies of Indian languages. IndicMMLU-Pro provides a standardized evaluation framework to push the research boundaries in Indic language AI, facilitating the development of more accurate, efficient, and culturally sensitive models. This paper outlines the benchmarks' design principles, task taxonomy, data collection methodology, and presents baseline results from state-of-the-art multilingual models. 

**Abstract (ZH)**: 印度次大陆有超过15亿人使用印欧语系语言，这些语言因其丰富的文化传统、语言多样性以及复杂的结构，在自然语言处理（NLP）研究中面临着独特的挑战和机遇。IndicMMLU-Pro 是一个全面的基准测试，旨在评估大型语言模型（LLMs）在印欧语系语言中的性能，基于MMLU Pro（大规模多任务语言理解）框架构建。该基准测试涵盖主要的印欧语系语言，如印地语、孟加拉语、古吉拉特语、马拉地语、卡纳达语、旁遮普语、泰米尔语、泰卢固语和乌尔都语，以应对印度次大陆语言多样性的独特挑战和机遇。该基准测试涵盖了语言理解、推理和生成等广泛的任务，精心设计以捕捉印度语言的复杂性。IndicMMLU-Pro 提供了一个标准化的评估框架，以推动印欧语系语言AI的研究边界，促进开发更准确、更高效且更具文化敏感性的模型。本文概述了基准测试的设计原则、任务分类、数据采集方法，并展示了最新的多语言模型的基准结果。 

---
# Leveraging Video Vision Transformer for Alzheimer's Disease Diagnosis from 3D Brain MRI 

**Title (ZH)**: 利用视频视觉变压器从三维脑部MRI图像中诊断阿尔茨海默病 

**Authors**: Taymaz Akan, Sait Alp, Md. Shenuarin Bhuiyan, Elizabeth A. Disbrow, Steven A. Conrad, John A. Vanchiere, Christopher G. Kevil, Mohammad A. N. Bhuiyan  

**Link**: [PDF](https://arxiv.org/pdf/2501.15733)  

**Abstract**: Alzheimer's disease (AD) is a neurodegenerative disorder affecting millions worldwide, necessitating early and accurate diagnosis for optimal patient management. In recent years, advancements in deep learning have shown remarkable potential in medical image analysis. Methods In this study, we present "ViTranZheimer," an AD diagnosis approach which leverages video vision transformers to analyze 3D brain MRI data. By treating the 3D MRI volumes as videos, we exploit the temporal dependencies between slices to capture intricate structural relationships. The video vision transformer's self-attention mechanisms enable the model to learn long-range dependencies and identify subtle patterns that may indicate AD progression. Our proposed deep learning framework seeks to enhance the accuracy and sensitivity of AD diagnosis, empowering clinicians with a tool for early detection and intervention. We validate the performance of the video vision transformer using the ADNI dataset and conduct comparative analyses with other relevant models. Results The proposed ViTranZheimer model is compared with two hybrid models, CNN-BiLSTM and ViT-BiLSTM. CNN-BiLSTM is the combination of a convolutional neural network (CNN) and a bidirectional long-short-term memory network (BiLSTM), while ViT-BiLSTM is the combination of a vision transformer (ViT) with BiLSTM. The accuracy levels achieved in the ViTranZheimer, CNN-BiLSTM, and ViT-BiLSTM models are 98.6%, 96.479%, and 97.465%, respectively. ViTranZheimer demonstrated the highest accuracy at 98.6%, outperforming other models in this evaluation metric, indicating its superior performance in this specific evaluation metric. Conclusion This research advances the understanding of applying deep learning techniques in neuroimaging and Alzheimer's disease research, paving the way for earlier and less invasive clinical diagnosis. 

**Abstract (ZH)**: 阿尔茨海默病（AD）是一种影响全世界数百万人的神经退行性疾病，早期和准确的诊断对于优化患者管理至关重要。近年来，深度学习的进展在医学图像分析领域显示出了显著的潜力。方法 在这项研究中，我们提出了一种名为“ViTranZheimer”的AD诊断方法，该方法利用视频视觉变压器来分析3D脑MRI数据。通过将3D MRI体素视为视频，我们利用各切片之间的时序依赖关系，捕捉复杂的结构关系。视频视觉变压器的自我注意机制使模型能够学习长期依赖关系并识别可能表明AD进展的细微模式。我们提出的深度学习框架旨在增强AD诊断的准确性和灵敏度，使临床医生能够提前检测和干预。我们使用ADNI数据集验证了视频视觉变压器的性能，并与其他相关模型进行了比较分析。结果 所提出的ViTranZheimer模型分别与两种混合模型CNN-BiLSTM和ViT-BiLSTM进行了比较。CNN-BiLSTM是由卷积神经网络（CNN）和双向长短期记忆网络（BiLSTM）组合而成，而ViT-BiLSTM则是由视觉变压器（ViT）与BiLSTM组合而成。ViTranZheimer、CNN-BiLSTM和ViT-BiLSTM模型的准确率分别为98.6%、96.479%和97.465%。ViTranZheimer在准确率方面达到了98.6%，在该评价指标中优于其他模型，表明其在特定评价指标上的优越性能。结论 本研究推进了将深度学习技术应用于神经影像学和阿尔茨海默病研究的理解，为早期和非侵入性的临床诊断铺平了道路。 

---
# Renewable Energy Prediction: A Comparative Study of Deep Learning Models for Complex Dataset Analysis 

**Title (ZH)**: 可再生能源预测：复杂数据集分析中深度学习模型的 comparative study 

**Authors**: Haibo Wang, Jun Huang, Lutfu Sua, Bahram Alidaee  

**Link**: [PDF](https://arxiv.org/pdf/2501.15731)  

**Abstract**: The increasing focus on predicting renewable energy production aligns with advancements in deep learning (DL). The inherent variability of renewable sources and the complexity of prediction methods require robust approaches, such as DL models, in the renewable energy sector. DL models are preferred over traditional machine learning (ML) because they capture complex, nonlinear relationships in renewable energy datasets. This study examines key factors influencing DL technique accuracy, including sampling and hyperparameter optimization, by comparing various methods and training and test ratios within a DL framework. Seven machine learning methods, LSTM, Stacked LSTM, CNN, CNN-LSTM, DNN, Time-Distributed MLP (TD-MLP), and Autoencoder (AE), are evaluated using a dataset combining weather and photovoltaic power output data from 12 locations. Regularization techniques such as early stopping, neuron dropout, L1 and L2 regularization are applied to address overfitting. The results demonstrate that the combination of early stopping, dropout, and L1 regularization provides the best performance to reduce overfitting in the CNN and TD-MLP models with larger training set, while the combination of early stopping, dropout, and L2 regularization is the most effective to reduce the overfitting in CNN-LSTM and AE models with smaller training set. 

**Abstract (ZH)**: 随着对可再生能源产量预测的关注不断增加，这与深度学习（DL）技术的进步相契合。由于可再生能源固有的波动性和预测方法的复杂性，可再生能源领域需要强大的方法，如DL模型。深度学习模型因其能够捕捉可再生能源数据集中的复杂非线性关系，而优于传统的机器学习（ML）方法。本研究通过在DL框架内比较各种方法和训练集与测试集的比例，探讨影响DL技术准确性的关键因素，包括采样和超参数优化。研究采用结合了12个地点的气象和光伏功率输出数据集，评估了七种机器学习方法，包括LSTM、堆叠LSTM、CNN、CNN-LSTM、DNN、时序分布MLP（TD-MLP）和自编码器（AE）。应用了诸如提前停止、神经元丢弃、L1和L2正则化等正则化技术来解决过拟合问题。结果显示，在大型训练集上，提前停止、神经元丢弃和L1正则化相结合提供最佳性能以减少CNN和TD-MLP模型的过拟合；而在小型训练集上，提前停止、神经元丢弃和L2正则化相结合是最有效的减少CNN-LSTM和AE模型过拟合的方法。 

---
# Gensors: Authoring Personalized Visual Sensors with Multimodal Foundation Models and Reasoning 

**Title (ZH)**: Gensors：使用多模态基础模型和推理构建个性化的视觉传感器 

**Authors**: Michael Xieyang Liu, Savvas Petridis, Vivian Tsai, Alexander J. Fiannaca, Alex Olwal, Michael Terry, Carrie J. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2501.15727)  

**Abstract**: Multimodal large language models (MLLMs), with their expansive world knowledge and reasoning capabilities, present a unique opportunity for end-users to create personalized AI sensors capable of reasoning about complex situations. A user could describe a desired sensing task in natural language (e.g., "alert if my toddler is getting into mischief"), with the MLLM analyzing the camera feed and responding within seconds. In a formative study, we found that users saw substantial value in defining their own sensors, yet struggled to articulate their unique personal requirements and debug the sensors through prompting alone. To address these challenges, we developed Gensors, a system that empowers users to define customized sensors supported by the reasoning capabilities of MLLMs. Gensors 1) assists users in eliciting requirements through both automatically-generated and manually created sensor criteria, 2) facilitates debugging by allowing users to isolate and test individual criteria in parallel, 3) suggests additional criteria based on user-provided images, and 4) proposes test cases to help users "stress test" sensors on potentially unforeseen scenarios. In a user study, participants reported significantly greater sense of control, understanding, and ease of communication when defining sensors using Gensors. Beyond addressing model limitations, Gensors supported users in debugging, eliciting requirements, and expressing unique personal requirements to the sensor through criteria-based reasoning; it also helped uncover users' "blind spots" by exposing overlooked criteria and revealing unanticipated failure modes. Finally, we discuss how unique characteristics of MLLMs--such as hallucinations and inconsistent responses--can impact the sensor-creation process. These findings contribute to the design of future intelligent sensing systems that are intuitive and customizable by everyday users. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）凭借其广泛的世界知识和推理能力，为终端用户创造个性化的AI传感器提供了独特的机会，这些传感器能够对复杂情况进行推理。用户可以用自然语言描述一个期望的感知任务（例如，“如果我的幼儿在搞破坏，请报警”），MLLMs 分析摄像头视频并在几秒钟内做出响应。在一项形成性研究中，我们发现用户在定义自己的传感器方面看到了巨大的价值，但他们在表达独特的个人需求及仅通过提示进行调试方面遇到困难。为了解决这些问题，我们开发了Gensors系统，它赋予用户通过MLLMs 的推理能力定义自定义传感器的能力。Gensors 1) 通过自动生成和手动创建传感器标准帮助用户提取需求，2) 通过允许用户并行隔离和测试各个标准的方式支持调试，3) 根据用户提供的图片建议额外的标准，4) 提出测试案例帮助用户对传感器进行“压力测试”，以应对可能不可预见的情况。在一项用户研究中，参与者在使用Gensors定义传感器时报告了更大的控制感、理解和沟通便利性。除了解决模型限制外，Gensors还支持用户通过基于标准的推理调试、提取需求并表达独特的个人需求，同时通过暴露被忽视的标准并揭示不可预见的故障模式，帮助用户发现“盲点”。最后，我们讨论了MLLMs的独特特征（如幻觉和不一致的响应）如何影响传感器创建过程。这些发现为未来直观且可定制的智能感知系统的设计做出了贡献。 

---
# A Survey on Computational Pathology Foundation Models: Datasets, Adaptation Strategies, and Evaluation Tasks 

**Title (ZH)**: 计算病理学基础模型综述：数据集、适应策略与评估任务 

**Authors**: Dong Li, Guihong Wan, Xintao Wu, Xinyu Wu, Ajit J. Nirmal, Christine G. Lian, Peter K. Sorger, Yevgeniy R. Semenov, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15724)  

**Abstract**: Computational pathology foundation models (CPathFMs) have emerged as a powerful approach for analyzing histopathological data, leveraging self-supervised learning to extract robust feature representations from unlabeled whole-slide images. These models, categorized into uni-modal and multi-modal frameworks, have demonstrated promise in automating complex pathology tasks such as segmentation, classification, and biomarker discovery. However, the development of CPathFMs presents significant challenges, such as limited data accessibility, high variability across datasets, the necessity for domain-specific adaptation, and the lack of standardized evaluation benchmarks. This survey provides a comprehensive review of CPathFMs in computational pathology, focusing on datasets, adaptation strategies, and evaluation tasks. We analyze key techniques, such as contrastive learning and multi-modal integration, and highlight existing gaps in current research. Finally, we explore future directions from four perspectives for advancing CPathFMs. This survey serves as a valuable resource for researchers, clinicians, and AI practitioners, guiding the advancement of CPathFMs toward robust and clinically applicable AI-driven pathology solutions. 

**Abstract (ZH)**: 计算病理学基础模型（CPathFMs）已 emerges 作为分析组织病理学数据的强大方法，通过自监督学习从未经标记的全切片图像中提取稳健的特征表示。这些模型分为空态模态和多模态框架，已在自动化复杂病理任务（如分割、分类和生物标志物发现）方面显示出前景。然而，CPathFMs 的发展面临显著挑战，包括数据获取限制、数据集间高变异性、领域特定适应的必要性以及缺乏标准化评估基准。本文综述了计算病理学中的 CPathFMs，重点关注数据集、适应策略和评估任务。我们分析了关键技术，如对比学习和多模态集成，并指出现有研究中的现有差距。最后，我们从四个角度探索了推进 CPathFMs 的未来方向。本文综述为研究人员、临床医生和人工智能从业者提供了有价值的资源，指导 CPathFMs 向稳健且临床适用的人工智能驱动病理解决方案的发展。 

---
# StaICC: Standardized Evaluation for Classification Task in In-context Learning 

**Title (ZH)**: StaICC：情境学习中分类任务的标准评估方法 

**Authors**: Hakaze Cho, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2501.15708)  

**Abstract**: Classification tasks are widely investigated in the In-Context Learning (ICL) paradigm. However, current efforts are evaluated on disjoint benchmarks and settings, while their performances are significantly influenced by some trivial variables, such as prompt templates, data sampling, instructions, etc., which leads to significant inconsistencies in the results reported across various literature, preventing fair comparison or meta-analysis across different papers. Therefore, this paper proposes a standardized and easy-to-use evaluation toolkit (StaICC) for in-context classification. Including, for the normal classification task, we provide StaICC-Normal, selecting 10 widely used datasets, and generating prompts with a fixed form, to mitigate the variance among the experiment implementations. To enrich the usage of our benchmark, we also provide a sub-benchmark StaICC-Diag for diagnosing ICL from several aspects, aiming for a more robust inference processing. 

**Abstract (ZH)**: 在基于上下文学习（ICL）范式下，分类任务已经被广泛研究。然而，当前的努力主要在独立的基准和设置上进行评估，而其性能受到一些简单变量的显著影响，如提示模板、数据采样、指令等，这导致了在不同文献中报告的结果存在显著差异，妨碍了不同论文间的公平比较或元分析。因此，本文提出了一种标准化且易于使用的评估工具包（StaICC），用于ICL中的分类评估。具体而言，对于常规分类任务，我们提供了一种标准化工具包StaICC-Normal，选择了10个广泛使用的数据集，并生成固定格式的提示，以减轻实验实现间的变异性。为了丰富我们基准的使用，我们还提供了一个子基准StaICC-Diag，从多个方面诊断ICL，旨在实现更加稳健的推理处理。 

---
# Contextual Knowledge Sharing in Multi-Agent Reinforcement Learning with Decentralized Communication and Coordination 

**Title (ZH)**: 基于分散通信与协调的多智能体强化学习中的上下文知识共享 

**Authors**: Hung Du, Srikanth Thudumu, Hy Nguyen, Rajesh Vasa, Kon Mouzakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.15695)  

**Abstract**: Decentralized Multi-Agent Reinforcement Learning (Dec-MARL) has emerged as a pivotal approach for addressing complex tasks in dynamic environments. Existing Multi-Agent Reinforcement Learning (MARL) methodologies typically assume a shared objective among agents and rely on centralized control. However, many real-world scenarios feature agents with individual goals and limited observability of other agents, complicating coordination and hindering adaptability. Existing Dec-MARL strategies prioritize either communication or coordination, lacking an integrated approach that leverages both. This paper presents a novel Dec-MARL framework that integrates peer-to-peer communication and coordination, incorporating goal-awareness and time-awareness into the agents' knowledge-sharing processes. Our framework equips agents with the ability to (i) share contextually relevant knowledge to assist other agents, and (ii) reason based on information acquired from multiple agents, while considering their own goals and the temporal context of prior knowledge. We evaluate our approach through several complex multi-agent tasks in environments with dynamically appearing obstacles. Our work demonstrates that incorporating goal-aware and time-aware knowledge sharing significantly enhances overall performance. 

**Abstract (ZH)**: 去中心化多智能体强化学习（Decentralized Multi-Agent Reinforcement Learning, Dec-MARL）已成为解决动态环境中复杂任务的关键方法。现有的多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）方法通常假定智能体之间共享目标，并依赖于中心化的控制。然而，许多实际场景中，智能体具有独立的目标，并且对其他智能体的可观测性有限，这使得协调变得复杂，并阻碍了适应性。现有的去中心化多智能体强化学习策略更侧重于通信或协调，缺乏一个既能利用两者优势的综合方法。本文提出了一种新的Dec-MARL框架，该框架将点对点通信与协调相结合，并将目标意识和时间意识融入智能体的知识共享过程中。该框架使得智能体能够：
(i) 共享上下文相关知识以帮助其他智能体；
(ii) 基于从多个智能体获取的信息进行推理，同时考虑自身目标和先前知识的时间背景。

我们通过在动态障碍环境中执行的多种复杂多智能体任务来评估该方法。我们的研究表明，结合目标意识和时间意识的知识共享显著提高了整体性能。 

---
# Beyond Benchmarks: On The False Promise of AI Regulation 

**Title (ZH)**: 超越基准：关于AI监管的虚假承诺 

**Authors**: Gabriel Stanovsky, Renana Keydar, Gadi Perl, Eliya Habba  

**Link**: [PDF](https://arxiv.org/pdf/2501.15693)  

**Abstract**: The rapid advancement of artificial intelligence (AI) systems in critical domains like healthcare, justice, and social services has sparked numerous regulatory initiatives aimed at ensuring their safe deployment. Current regulatory frameworks, exemplified by recent US and EU efforts, primarily focus on procedural guidelines while presuming that scientific benchmarking can effectively validate AI safety, similar to how crash tests verify vehicle safety or clinical trials validate drug efficacy. However, this approach fundamentally misunderstands the unique technical challenges posed by modern AI systems. Through systematic analysis of successful technology regulation case studies, we demonstrate that effective scientific regulation requires a causal theory linking observable test outcomes to future performance - for instance, how a vehicle's crash resistance at one speed predicts its safety at lower speeds. We show that deep learning models, which learn complex statistical patterns from training data without explicit causal mechanisms, preclude such guarantees. This limitation renders traditional regulatory approaches inadequate for ensuring AI safety. Moving forward, we call for regulators to reckon with this limitation, and propose a preliminary two-tiered regulatory framework that acknowledges these constraints: mandating human oversight for high-risk applications while developing appropriate risk communication strategies for lower-risk uses. Our findings highlight the urgent need to reconsider fundamental assumptions in AI regulation and suggest a concrete path forward for policymakers and researchers. 

**Abstract (ZH)**: 人工智能（AI）系统在医疗、司法和社会服务等关键领域的迅速发展激发了众多监管举措，旨在确保其安全部署。当前的监管框架，以近期美国和欧盟的努力为例，主要集中在程序性指导方针上，并假设科学基准测试能够有效验证AI的安全性，类似于碰撞测试验证车辆安全性或临床试验验证药物有效性的方式。然而，这种做法从根本上未能理解现代AI系统所面临的独特技术挑战。通过系统分析成功的科技监管案例，我们证明有效的科学监管需要将可观察的测试结果与未来性能联系起来的因果理论——例如，汽车在某一速度下的碰撞抵抗能力如何预测其在较低速度下的安全性。我们指出，深度学习模型从训练数据中学习复杂的统计模式，而无需明确的因果机制，这预示着上述保证无法实现。这一局限性使得传统的监管方法不足以确保AI的安全性。展望未来，我们呼吁监管机构认识这一局限性，并提出初步的两层监管框架以承认这些限制：对高风险应用实施人工监督，同时为低风险用途制定适当的风险管理策略。我们的研究强调了重新考虑AI监管基本假设的紧迫性，并为政策制定者和研究人员指出了具体的发展路径。 

---
# Transformer-Based Multimodal Knowledge Graph Completion with Link-Aware Contexts 

**Title (ZH)**: 基于Transformer的具有链接感知上下文的多模态知识图谱补全 

**Authors**: Haodi Ma, Dzmitry Kasinets, Daisy Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15688)  

**Abstract**: Multimodal knowledge graph completion (MMKGC) aims to predict missing links in multimodal knowledge graphs (MMKGs) by leveraging information from various modalities alongside structural data. Existing MMKGC approaches primarily extend traditional knowledge graph embedding (KGE) models, which often require creating an embedding for every entity. This results in large model sizes and inefficiencies in integrating multimodal information, particularly for real-world graphs. Meanwhile, Transformer-based models have demonstrated competitive performance in knowledge graph completion (KGC). However, their focus on single-modal knowledge limits their capacity to utilize cross-modal information. Recently, Large vision-language models (VLMs) have shown potential in cross-modal tasks but are constrained by the high cost of training. In this work, we propose a novel approach that integrates Transformer-based KGE models with cross-modal context generated by pre-trained VLMs, thereby extending their applicability to MMKGC. Specifically, we employ a pre-trained VLM to transform relevant visual information from entities and their neighbors into textual sequences. We then frame KGC as a sequence-to-sequence task, fine-tuning the model with the generated cross-modal context. This simple yet effective method significantly reduces model size compared to traditional KGE approaches while achieving competitive performance across multiple large-scale datasets with minimal hyperparameter tuning. 

**Abstract (ZH)**: 多模态知识图谱补全（MMKGC）旨在通过利用多种模态的信息以及结构数据来预测多模态知识图谱（MMKGs）中的缺失链接。现有的MMKGC方法主要扩展了传统的知识图嵌入（KGE）模型，这些模型经常需要为每个实体创建一个嵌入，这导致了模型规模庞大，并且在整合多模态信息方面效率低下，尤其是在处理真实世界的图时。同时，基于Transformer的模型已经在知识图谱补全（KGC）任务中显示出竞争力，但它们主要关注单模态知识，限制了其利用跨模态信息的能力。最近，预训练的大规模视觉-语言模型（VLMs）显示出了在跨模态任务中的潜力，但由于训练成本高昂，其应用受到限制。在这项工作中，我们提出了一种新颖的方法，将基于Transformer的知识图嵌入模型与预训练的VLM生成的跨模态上下文结合起来，从而扩展其在MMKGC中的适用性。具体来说，我们使用预训练的VLM将与实体及其邻居相关的视觉信息转换为文本序列。然后，我们将KGC问题重新定义为序列到序列的任务，并使用生成的跨模态上下文对模型进行微调。这种方法简单而有效，与传统的KGE方法相比，显著减少了模型规模，并在多个大规模数据集上实现了竞争性性能，同时通过最少的超参数调整实现了这一点。 

---
# Blissful (A)Ignorance: People form overly positive impressions of others based on their written messages, despite wide-scale adoption of Generative AI 

**Title (ZH)**: 乐于无知：即使广泛采用了生成式人工智能，人们仍然基于他人的书面信息形成了过度积极的印象 

**Authors**: Jiaqi Zhu, Andras Molnar  

**Link**: [PDF](https://arxiv.org/pdf/2501.15678)  

**Abstract**: As the use of Generative AI (GenAI) tools becomes more prevalent in interpersonal communication, understanding their impact on social perceptions is crucial. According to signaling theory, GenAI may undermine the credibility of social signals conveyed in writing, since it reduces the cost of writing and makes it hard to verify the authenticity of messages. Using a pre-registered large-scale online experiment (N = 647; Prolific), featuring scenarios in a range of communication contexts (personal vs. professional; close others vs. strangers), we explored how senders' use of GenAI influenced recipients' impressions of senders, both when GenAI use was known or uncertain. Consistent with past work, we found strong negative effects on social impressions when disclosing that a message was AI-generated, compared to when the same message was human-written. However, under the more realistic condition when potential GenAI use was not explicitly highlighted, recipients did not exhibit any skepticism towards senders, and these "uninformed" impressions were virtually indistinguishable from those of fully human-written messages. Even when we highlighted the potential (but uncertain) use of GenAI, recipients formed overly positive impressions. These results are especially striking given that 46% of our sample admitted having used such tools for writing messages, just within the past two weeks. Our findings put past work in a new light: While social judgments can be substantially affected when GenAI use is explicitly disclosed, this information may not be readily available in more realistic communication settings, making recipients blissfully ignorant about others' potential use of GenAI. 

**Abstract (ZH)**: 随着生成式人工智能（GenAI）工具在人际交流中的使用越来越普遍，理解其对社会认知的影响变得至关重要。根据信号理论，GenAI可能会削弱书面交流中传递的社会信号的可信度，因为它降低了书写成本，并使得验证消息的真实性变得困难。我们通过一项预先注册的大规模在线实验（N = 647；Prolific），对多种交流情境（个人 vs. 专业；亲密的人 vs. 陌生人）进行了探索，研究了发送者使用GenAI如何影响接收者对发送者的印象，无论是当GenAI的使用是已知的还是不确定的。与以往的研究一致，我们发现，在透露消息是AI生成的情况下，社会印象会受到强烈负面影响，与同为人类写作的相同消息相比。然而，在可能使用GenAI的情况没有明确突出的更现实条件下，接收者并未表现出对发送者的任何怀疑，这些“不知情”的印象几乎与完全人类写作的消息的印象无法区分。即使我们强调了潜在（但不确定的）GenAI使用情况，接收者也形成了过于积极的印象。考虑到我们样本中有46%的人在过去的两周内承认使用过此类工具进行写作，这一结果尤为引人注目。我们的发现为过去的研究提供了新的视角：虽然在明确披露使用GenAI的情况下，社会判断可能会受到重大影响，但在更具现实意义的交流环境中，这种信息可能并不可用，这使得接收者对他人可能使用GenAI的情况一无所知，因而蒙在鼓里。 

---
# StagFormer: Time Staggering Transformer Decoding for RunningLayers In Parallel 

**Title (ZH)**: StagFormer：时间交错Transformer解码以并行处理层 

**Authors**: Dylan Cutler, Arun Kandoor, Nishanth Dikkala, Nikunj Saunshi, Xin Wang, Rina Panigrahy  

**Link**: [PDF](https://arxiv.org/pdf/2501.15665)  

**Abstract**: Standard decoding in a Transformer based language model is inherently sequential as we wait for a token's embedding to pass through all the layers in the network before starting the generation of the next token. In this work, we propose a new architecture StagFormer (Staggered Transformer), which staggered execution along the time axis and thereby enables parallelizing the decoding process along the depth of the model. We achieve this by breaking the dependency of the token representation at time step $i$ in layer $l$ upon the representations of tokens until time step $i$ from layer $l-1$. Instead, we stagger the execution and only allow a dependency on token representations until time step $i-1$. The later sections of the Transformer still get access to the ``rich" representations from the prior section but only from those token positions which are one time step behind. StagFormer allows for different sections of the model to be executed in parallel yielding at potential 33\% speedup in decoding while being quality neutral in our simulations. We also explore many natural variants of this idea. We present how weight-sharing across the different sections being staggered can be more practical in settings with limited memory. We show how one can approximate a recurrent model during inference using such weight-sharing. We explore the efficacy of using a bounded window attention to pass information from one section to another which helps drive further latency gains for some applications. We also explore demonstrate the scalability of the staggering idea over more than 2 sections of the Transformer. 

**Abstract (ZH)**: 基于Transformer的语言模型的标准解码本质上是顺序的，因为在生成下一个词元之前，我们需要等待词元嵌入通过网络中的所有层。本文中，我们提出了一种新的架构StagFormer（错位Transformer），该架构在时间轴上实行错位执行，从而使得解码过程可以在模型的深度维度上并行化。我们通过打破词元在时间步 $i$、第 $l$ 层的表示与来自第 $l-1$ 层直到时间步 $i$ 的词元表示之间的依赖关系来实现这一点。相反，我们实行错位执行，仅允许词元表示直到时间步 $i-1$ 之间的依赖关系。Transformer 后期部分仍然可以访问之前部分的“丰富”表示，但仅限于那些时间步延迟一位的词元位置。StagFormer 允许模型的不同部分并行执行，从而在解码过程中可能获得33%的速度提升，而我们的模拟显示其质量保持不变。我们还探讨了许多这一理念的自然变体。我们展示了在内存有限的情况下，不同部分之间的权重共享可以更加实用。我们展示了如何在推断过程中使用此类权重共享来近似递归模型。我们探讨了使用有限窗口注意机制来传递一个部分到另一个部分的信息，这对于一些应用有助于进一步减少延迟。我们还展示了错位执行理念在Transformer多于两部分上的可扩展性。 

---
# Constrained Hybrid Metaheuristic Algorithm for Probabilistic Neural Networks Learning 

**Title (ZH)**: 约束混合元启发式算法在概率神经网络学习中的应用 

**Authors**: Piotr A. Kowalski, Szymon Kucharczyk, Jacek Mańdziuk  

**Link**: [PDF](https://arxiv.org/pdf/2501.15661)  

**Abstract**: This study investigates the potential of hybrid metaheuristic algorithms to enhance the training of Probabilistic Neural Networks (PNNs) by leveraging the complementary strengths of multiple optimisation strategies. Traditional learning methods, such as gradient-based approaches, often struggle to optimise high-dimensional and uncertain environments, while single-method metaheuristics may fail to exploit the solution space fully. To address these challenges, we propose the constrained Hybrid Metaheuristic (cHM) algorithm, a novel approach that combines multiple population-based optimisation techniques into a unified framework. The proposed procedure operates in two phases: an initial probing phase evaluates multiple metaheuristics to identify the best-performing one based on the error rate, followed by a fitting phase where the selected metaheuristic refines the PNN to achieve optimal smoothing parameters. This iterative process ensures efficient exploration and convergence, enhancing the network's generalisation and classification accuracy. cHM integrates several popular metaheuristics, such as BAT, Simulated Annealing, Flower Pollination Algorithm, Bacterial Foraging Optimization, and Particle Swarm Optimisation as internal optimisers. To evaluate cHM performance, experiments were conducted on 16 datasets with varying characteristics, including binary and multiclass classification tasks, balanced and imbalanced class distributions, and diverse feature dimensions. The results demonstrate that cHM effectively combines the strengths of individual metaheuristics, leading to faster convergence and more robust learning. By optimising the smoothing parameters of PNNs, the proposed method enhances classification performance across diverse datasets, proving its application flexibility and efficiency. 

**Abstract (ZH)**: 本研究探讨了混合元启发式算法在利用多种优化策略互补优势的基础上，提升概率神经网络（PNN）训练性能的潜力。传统的学习方法，如梯度基方法，往往难以优化高维和不确定的环境，而单一策略的元启发式算法可能无法充分利用解空间。为解决这些问题，我们提出了一种新的约束混合元启发式（cHM）算法，该方法将多种基于群体的优化技术整合到统一框架中。该提出的程序分为两个阶段：初始探测阶段评估多种元启发式算法，基于错误率选择表现最佳的算法，随后的拟合阶段使用选定的元启发式算法对PNN进行优化，以确定最优的平滑参数。这种迭代过程确保了高效的探索和收敛，从而增强了网络的泛化能力和分类准确性。cHM 结合了多种流行的元启发式算法，如BAT算法、模拟退火算法、花粉授粉算法、细菌觅食优化算法和粒子群优化算法作为内部优化器。为了评估cHM 的性能，我们在16个具有不同特性的数据集上进行了实验，包括二分类和多分类任务、平衡和不平衡类分布以及各种特征维度。结果表明，cHM 有效地结合了个体元启发式的优点，实现了更快的收敛和更稳健的学习。通过优化PNN的平滑参数，所提出的方法在不同数据集上提高了分类性能，证明了其应用的灵活性和效率。 

---
# Marker Track: Accurate Fiducial Marker Tracking for Evaluation of Residual Motions During Breath-Hold Radiotherapy 

**Title (ZH)**: 标记跟踪：用于评估屏气放射治疗中残余运动的固定标记跟踪方法 

**Authors**: Aimee Guo, Weihua Mao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15660)  

**Abstract**: Fiducial marker positions in projection image of cone-beam computed tomography (CBCT) scans have been studied to evaluate daily residual motion during breath-hold radiation therapy. Fiducial marker migration posed challenges in accurately locating markers, prompting the development of a novel algorithm that reconstructs volumetric probability maps of marker locations from filtered gradient maps of projections. This guides the development of a Python-based algorithm to detect fiducial markers in projection images using Meta AI's Segment Anything Model 2 (SAM 2). Retrospective data from a pancreatic cancer patient with two fiducial markers were analyzed. The three-dimensional (3D) marker positions from simulation computed tomography (CT) were compared to those reconstructed from CBCT images, revealing a decrease in relative distances between markers over time. Fiducial markers were successfully detected in 2777 out of 2786 projection frames. The average standard deviation of superior-inferior (SI) marker positions was 0.56 mm per breath-hold, with differences in average SI positions between two breath-holds in the same scan reaching up to 5.2 mm, and a gap of up to 7.3 mm between the end of the first and beginning of the second breath-hold. 3D marker positions were calculated using projection positions and confirmed marker migration. This method effectively calculates marker probability volume and enables accurate fiducial marker tracking during treatment without requiring any specialized equipment, additional radiation doses, or manual initialization and labeling. It has significant potential for automatically assessing daily residual motion to adjust planning margins, functioning as an adaptive radiation therapy tool. 

**Abstract (ZH)**: 在锥形束计算机断层扫描（CBCT）扫描的投影图像中研究 fiducial 标记的位置，目的是评估呼吸 hold 放射治疗过程中的日间残余运动。fiducial 标记的迁移给准确定位带来了挑战，促使开发了一种新型算法，该算法从投影的筛选梯度图中重建标记位置的体素概率图。该算法指导开发了一个基于 Python 的算法，使用 Meta AI 的 Segment Anything Model 2 (SAM 2) 来检测投影图像中的 fiducial 标记。对一名胰腺癌患者前后两次扫描中的两个 fiducial 标记进行了回顾性分析。根据不同模拟 CT 中重建的三维（3D）标记位置与从 CBCT 图像中重建的位置的对比，结果显示标记间相对距离随时间有所减小。检测成功的投影帧共 2777 帧，成功率达到了 99.7%。超声上下的（SI）标记位置的标准差在每次呼吸 hold 中平均为 0.56 毫米，而在同一个扫描中两次呼吸 hold 间的平均 SI 位置差异可达 5.2 毫米，两个呼吸 hold 之间的最大间隔为 7.3 毫米。使用投影位置和确认的标记迁移来计算 3D 标记位置。该方法有效地计算了标记的概率体积，并能实现无需任何特殊设备、额外辐射剂量或人工初始化和标注的情况下，在治疗过程中准确追踪 fiducial 标记。这种方法具有自动评估日常残余运动以便调整计划边界的重要潜力，可作为适应性放射治疗的工具。 

---
# People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text 

**Title (ZH)**: 经常使用ChatGPT进行写作任务的人能够准确且稳健地检测出AI生成的文本。 

**Authors**: Jenna Russell, Marzena Karpinska, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2501.15654)  

**Abstract**: In this paper, we study how well humans can detect text generated by commercial LLMs (GPT-4o, Claude, o1). We hire annotators to read 300 non-fiction English articles, label them as either human-written or AI-generated, and provide paragraph-length explanations for their decisions. Our experiments show that annotators who frequently use LLMs for writing tasks excel at detecting AI-generated text, even without any specialized training or feedback. In fact, the majority vote among five such "expert" annotators misclassifies only 1 of 300 articles, significantly outperforming most commercial and open-source detectors we evaluated even in the presence of evasion tactics like paraphrasing and humanization. Qualitative analysis of the experts' free-form explanations shows that while they rely heavily on specific lexical clues ('AI vocabulary'), they also pick up on more complex phenomena within the text (e.g., formality, originality, clarity) that are challenging to assess for automatic detectors. We release our annotated dataset and code to spur future research into both human and automated detection of AI-generated text. 

**Abstract (ZH)**: 本文研究了人类如何检测由商业大规模语言模型（如GPT-4o、Claude、o1）生成的文字。我们聘请注释员阅读300篇非虚构英文文章，并将它们标记为人类撰写或AI生成，同时提供段落长度的解释以说明其决策依据。我们的实验表明，经常使用语言模型进行写作任务的注释员在检测AI生成的文本方面表现出色，即使没有任何专门的培训或反馈。事实上，五名这样的“专家”注释员多数投票结果仅错误分类1篇文章，显著优于我们在评估中遇到的大多数商业和开源检测器，即使存在如改写和人性化等规避策略。专家提供自由形式解释的定性分析显示，他们不仅依赖特定的词汇线索（如“AI术语”），还能够识别文本中的复杂现象（如正式性、原创性、清晰度）——这些特征对于自动检测器来说难以评估。我们公开了注释数据集和代码，以促进未来对AI生成文本的人工和自动检测的研究。 

---
# Can Pose Transfer Models Generate Realistic Human Motion? 

**Title (ZH)**: 可以使用姿态转移模型生成逼真的-human-动作吗？ 

**Authors**: Vaclav Knapp, Matyas Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2501.15648)  

**Abstract**: Recent pose-transfer methods aim to generate temporally consistent and fully controllable videos of human action where the motion from a reference video is reenacted by a new identity. We evaluate three state-of-the-art pose-transfer methods -- AnimateAnyone, MagicAnimate, and ExAvatar -- by generating videos with actions and identities outside the training distribution and conducting a participant study about the quality of these videos. In a controlled environment of 20 distinct human actions, we find that participants, presented with the pose-transferred videos, correctly identify the desired action only 42.92% of the time. Moreover, the participants find the actions in the generated videos consistent with the reference (source) videos only 36.46% of the time. These results vary by method: participants find the splatting-based ExAvatar more consistent and photorealistic than the diffusion-based AnimateAnyone and MagicAnimate. 

**Abstract (ZH)**: 近年来，姿态迁移方法致力于生成在时间和身份上都保持一致的人类动作视频，其中参考视频中的动作会被新身份重现。我们通过生成训练分布之外的动作和身份视频，并进行参与者研究来评估三种最先进的姿态迁移方法——AnimateAnyone、MagicAnimate和ExAvatar。在20种不同的人类动作控制环境中，我们发现参与者在观看姿态迁移视频时，正确识别出所需动作的比例仅为42.92%。此外，在这些生成的视频中，参与者认为动作与参考（源）视频一致的比例仅为36.46%。这些结果因方法而异：参与者认为基于光栅化的ExAvatar比基于扩散的AnimateAnyone和MagicAnimate更加一致和逼真。 

---
# A Comprehensive Survey on Self-Interpretable Neural Networks 

**Title (ZH)**: 自我可解释神经网络综述 

**Authors**: Yang Ji, Ying Sun, Yuting Zhang, Zhigaoyuan Wang, Yuanxin Zhuang, Zheng Gong, Dazhong Shen, Chuan Qin, Hengshu Zhu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.15638)  

**Abstract**: Neural networks have achieved remarkable success across various fields. However, the lack of interpretability limits their practical use, particularly in critical decision-making scenarios. Post-hoc interpretability, which provides explanations for pre-trained models, is often at risk of robustness and fidelity. This has inspired a rising interest in self-interpretable neural networks, which inherently reveal the prediction rationale through the model structures. Although there exist surveys on post-hoc interpretability, a comprehensive and systematic survey of self-interpretable neural networks is still missing. To address this gap, we first collect and review existing works on self-interpretable neural networks and provide a structured summary of their methodologies from five key perspectives: attribution-based, function-based, concept-based, prototype-based, and rule-based self-interpretation. We also present concrete, visualized examples of model explanations and discuss their applicability across diverse scenarios, including image, text, graph data, and deep reinforcement learning. Additionally, we summarize existing evaluation metrics for self-interpretability and identify open challenges in this field, offering insights for future research. To support ongoing developments, we present a publicly accessible resource to track advancements in this domain: this https URL. 

**Abstract (ZH)**: 神经网络在各个领域取得了显著的成功。然而，缺乏可解释性限制了它们的实际应用，尤其是在关键决策场景中的应用。后 hoc 可解释性提供了对预训练模型的解释，但常常面临稳健性和准确性的风险。这激发了对自解释神经网络的兴趣，这些网络通过模型结构本身内在地揭示预测的理由。尽管已经有关于后 hoc 可解释性的综述，但关于自解释神经网络的全面而系统的综述仍然缺失。为填补这一空白，我们首先收集并回顾了现有的自解释神经网络工作，并从五个关键视角提供了方法论的结构化总结：基于归因的自解释、基于功能的自解释、基于概念的自解释、基于原型的自解释和基于规则的自解释。我们还展示了具体的可视化模型解释示例，并讨论了它们在图像、文本、图数据和深度强化学习等多种场景中的应用性。此外，我们总结了现有的自解释性评价指标，并指出了该领域的开放挑战，为未来的研究提供了见解。为了支持该领域的持续发展，我们提供了一个公开可访问的资源来跟踪该领域的进展：[这里提供链接]。 

---
# GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting 

**Title (ZH)**: 高斯令牌：一种有效的基于二维高斯渲染的图像标记器 

**Authors**: Jiajun Dong, Chengkun Wang, Wenzhao Zheng, Lei Chen, Jiwen Lu, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15619)  

**Abstract**: Effective image tokenization is crucial for both multi-modal understanding and generation tasks due to the necessity of the alignment with discrete text data. To this end, existing approaches utilize vector quantization (VQ) to project pixels onto a discrete codebook and reconstruct images from the discrete representation. However, compared with the continuous latent space, the limited discrete codebook space significantly restrict the representational ability of these image tokenizers. In this paper, we propose GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting as a solution. We first represent the encoded samples as multiple flexible featured 2D Gaussians characterized by positions, rotation angles, scaling factors, and feature coefficients. We adopt the standard quantization for the Gaussian features and then concatenate the quantization results with the other intrinsic Gaussian parameters before the corresponding splatting operation and the subsequent decoding module. In general, GaussianToken integrates the local influence of 2D Gaussian distribution into the discrete space and thus enhances the representation capability of the image tokenizer. Competitive reconstruction performances on CIFAR, Mini-ImageNet, and ImageNet-1K demonstrate the effectiveness of our framework. Our code is available at: this https URL. 

**Abstract (ZH)**: 有效的图像分词对于多模态的理解和生成任务至关重要，因为这需要与离散文本数据进行对齐。为此，现有的方法利用向量量化（VQ）将像素投影到一个离散的码本中，并从离散表示重建图像。然而，与连续的潜在空间相比，有限的离散码本空间显著限制了这些图像分词器的表示能力。在这篇论文中，我们提出了一种有效的图像分词方法——GaussianToken：一种基于2D 高斯散射的有效图像分词器。我们首先将编码样本表示为多个灵活的2D 高斯分布，这些分布由位置、旋转角度、缩放因子和特征系数来描述。我们采用标准的量化方法对高斯特征进行量化，然后在对应的散射操作和后续的解码模块之前，将量化结果与其他固有的高斯参数进行拼接。总体而言，GaussianToken 将2D 高斯分布的局部影响整合到离散空间中，从而增强了图像分词器的表示能力。在CIFAR、Mini-ImageNet和ImageNet-1K上的竞争性重建性能表明了我们框架的有效性。我们的代码可在以下链接获取：this https URL。 

---
# Your Learned Constraint is Secretly a Backward Reachable Tube 

**Title (ZH)**: 您的学习到的约束实际上是一个后向可达管状区域 

**Authors**: Mohamad Qadri, Gokul Swamy, Jonathan Francis, Michael Kaess, Andrea Bajcsy  

**Link**: [PDF](https://arxiv.org/pdf/2501.15618)  

**Abstract**: Inverse Constraint Learning (ICL) is the problem of inferring constraints from safe (i.e., constraint-satisfying) demonstrations. The hope is that these inferred constraints can then be used downstream to search for safe policies for new tasks and, potentially, under different dynamics. Our paper explores the question of what mathematical entity ICL recovers. Somewhat surprisingly, we show that both in theory and in practice, ICL recovers the set of states where failure is inevitable, rather than the set of states where failure has already happened. In the language of safe control, this means we recover a backwards reachable tube (BRT) rather than a failure set. In contrast to the failure set, the BRT depends on the dynamics of the data collection system. We discuss the implications of the dynamics-conditionedness of the recovered constraint on both the sample-efficiency of policy search and the transferability of learned constraints. 

**Abstract (ZH)**: 逆约束学习（ICL）问题是从安全执行（即满足约束的）示例中推断约束的过程。希望通过这种方式推断出的约束，可以在下游任务中用于寻找新的安全策略，并且在不同的动力学条件下也可能有效。我们的论文探讨了ICL恢复的是什么样的数学实体。令人意外的是，无论是从理论上还是从实践上来看，ICL恢复的是失败必然发生的状态集合，而不是已经发生失败的状态集合。用安全控制的语言来说，这意味着我们恢复的是一个向后可到达管状区域（BRT），而不是一个失败集合。与失败集合不同，BRT依赖于数据采集系统的动力学。我们讨论了恢复的约束的动力学特征对策略搜索的样本效率以及学习到的约束的可迁移性的影响。 

---
# Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images 

**Title (ZH)**: 基于组织学图像的空间解析基因表达推断的扩散生成建模 

**Authors**: Sichen Zhu, Yuchen Zhu, Molei Tao, Peng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15598)  

**Abstract**: Spatial Transcriptomics (ST) allows a high-resolution measurement of RNA sequence abundance by systematically connecting cell morphology depicted in Hematoxylin and Eosin (H&E) stained histology images to spatially resolved gene expressions. ST is a time-consuming, expensive yet powerful experimental technique that provides new opportunities to understand cancer mechanisms at a fine-grained molecular level, which is critical for uncovering new approaches for disease diagnosis and treatments. Here, we present $\textbf{Stem}$ ($\textbf{S}$pa$\textbf{T}$ially resolved gene $\textbf{E}$xpression inference with diffusion $\textbf{M}$odel), a novel computational tool that leverages a conditional diffusion generative model to enable in silico gene expression inference from H&E stained images. Through better capturing the inherent stochasticity and heterogeneity in ST data, $\textbf{Stem}$ achieves state-of-the-art performance on spatial gene expression prediction and generates biologically meaningful gene profiles for new H&E stained images at test time. We evaluate the proposed algorithm on datasets with various tissue sources and sequencing platforms, where it demonstrates clear improvement over existing approaches. $\textbf{Stem}$ generates high-fidelity gene expression predictions that share similar gene variation levels as ground truth data, suggesting that our method preserves the underlying biological heterogeneity. Our proposed pipeline opens up the possibility of analyzing existing, easily accessible H&E stained histology images from a genomics point of view without physically performing gene expression profiling and empowers potential biological discovery from H&E stained histology images. 

**Abstract (ZH)**: 空间转录组学（ST）可以通过系统地将苏木精和伊红（H&E）染色组织学图像中描绘的细胞形态与空间分辨的基因表达连接起来，实现RNA序列丰度的高分辨率测量。ST是一种耗时、昂贵但功能强大的实验技术，它为从分子层面深入了解癌症机制提供了新的机会，对于发现新的疾病诊断和治疗方法至关重要。在此，我们介绍了一种名为$\textbf{Stem}$（$\textbf{S}$pa$\textbf{T}$ially resolved gene $\textbf{E}$xpression inference with diffusion $\textbf{M}$odel）的新型计算工具，该工具利用条件扩散生成模型从H&E染色图像中推断细胞基因表达。通过更好地捕捉ST数据中固有的随机性和异质性，$\textbf{Stem}$在空间基因表达预测方面达到了最先进的性能，并且在测试时为新的H&E染色图像生成具有生物学意义的基因谱型。我们在具有不同组织来源和测序平台的数据集上评估了所提出的算法，结果显示其在现有方法基础上提供了明显改进。$\textbf{Stem}$生成的基因表达预测具有高保真度，其基因变异水平与真实数据相似，表明我们的方法保留了内在的生物学异质性。我们提出的处理管道为从基因组学角度分析现有易于获取的H&E染色组织学图像提供了可能，而无需实际进行基因表达分析，并能够从H&E染色组织学图像中发现潜在的生物学发现。 

---
# SCP-116K: A High-Quality Problem-Solution Dataset and a Generalized Pipeline for Automated Extraction in the Higher Education Science Domain 

**Title (ZH)**: SCP-116K：高质量问题解决方案数据集及高等教育科学领域自动提取的通用管道 

**Authors**: Dakuan Lu, Xiaoyu Tan, Rui Xu, Tianchu Yao, Chao Qu, Wei Chu, Yinghui Xu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2501.15587)  

**Abstract**: Recent breakthroughs in large language models (LLMs) exemplified by the impressive mathematical and scientific reasoning capabilities of the o1 model have spotlighted the critical importance of high-quality training data in advancing LLM performance across STEM disciplines. While the mathematics community has benefited from a growing body of curated datasets, the scientific domain at the higher education level has long suffered from a scarcity of comparable resources. To address this gap, we present SCP-116K, a new large-scale dataset of 116,756 high-quality problem-solution pairs, automatically extracted from heterogeneous sources using a streamlined and highly generalizable pipeline. Our approach involves stringent filtering to ensure the scientific rigor and educational level of the extracted materials, while maintaining adaptability for future expansions or domain transfers. By openly releasing both the dataset and the extraction pipeline, we seek to foster research on scientific reasoning, enable comprehensive performance evaluations of new LLMs, and lower the barrier to replicating the successes of advanced models like o1 in the broader science community. We believe SCP-116K will serve as a critical resource, catalyzing progress in high-level scientific reasoning tasks and promoting further innovations in LLM development. The dataset and code are publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新突破，例如o1模型在数学和科学推理方面显著的能力表现，凸显了高质量训练数据在跨STEM学科提升LLM性能方面的关键重要性。尽管数学领域受益于日益增长的高质量数据集，但高等教育领域的科学领域长期以来一直缺乏类似的资源。为填补这一空白，我们提出了SCP-116K，这是一个包含116,756个高质量问题-解决方案对的新大规模数据集，这些数据自动从异构来源中提取出来，采用了简化且高度可扩展的流程。我们的方法包括严格的筛选以确保提取材料的科学严谨性和教育水平，并保持对未来扩展或领域转移的适应性。通过公开提供该数据集和提取流程，我们旨在促进科学推理的研究，使全面评估新LLM的性能成为可能，并降低在更广泛科学界复制如o1等先进模型的成功门槛。我们认为，SCP-116K将成为一个重要资源，推动高级科学推理任务的发展，并促进LLM开发的进一步创新。该数据集和代码可以在以下网页公开获取：[这里添加链接]。 

---
# Twin Transition or Competing Interests? Validation of the Artificial Intelligence and Sustainability Perceptions Inventory (AISPI) 

**Title (ZH)**: 双轨转型还是利益竞争？人工智能与可持续性感知量表（AISPI）的验证 

**Authors**: Annika Bush  

**Link**: [PDF](https://arxiv.org/pdf/2501.15585)  

**Abstract**: As artificial intelligence (AI) and sustainability initiatives increasingly intersect, understanding public perceptions of their relationship becomes crucial for successful implementation. However, no validated instrument exists to measure these specific perceptions. This paper presents the development and validation of the Artificial Intelligence and Sustainability Perceptions Inventory (AISPI), a novel 13-item instrument measuring how individuals view the relationship between AI advancement and environmental sustainability. Through factor analysis (N=105), we identified two distinct dimensions: Twin Transition and Competing Interests. The instrument demonstrated strong reliability (alpha=.89) and construct validity through correlations with established measures of AI and sustainability attitudes. Our findings suggest that individuals can simultaneously recognize both synergies and tensions in the AI-sustainability relationship, offering important implications for researchers and practitioners working at this critical intersection. This work provides a foundational tool for future research on public perceptions of AI's role in sustainable development. 

**Abstract (ZH)**: 随着人工智能（AI）和可持续发展举措的日益交汇，理解公众对其之间关系的认知变得至关重要，这对于成功实施这些举措而言极为重要。然而，目前尚不存在能够度量这些特定认知的验证工具。本文介绍了《人工智能与可持续性感知量表》（AISPI）的发展与验证，这是一种新颖的包含13个项目的人工智能与环境可持续性关系认知测量工具。通过因子分析（样本量为105），我们识别出了两个不同的维度：双转型与竞争利益。该工具在可靠性（ alpha=.89）和结构效度方面表现优异，通过与既有的人工智能和可持续性态度衡量标准的相关分析得以验证。我们的研究结果表明，个体可以同时认识到人工智能与可持续性关系中的协同效应与紧张关系，为在此关键交汇点上开展研究和实践提供了重要启示。本文提供了一个基础工具，用于未来对人工智能在可持续发展中的角色的研究。 

---
# Comparative clinical evaluation of "memory-efficient" synthetic 3d generative adversarial networks (gan) head-to-head to state of art: results on computed tomography of the chest 

**Title (ZH)**: 基于对比临床评估的“内存高效”合成三维生成对抗网络（GAN）与当前最优方法的头对头比较：胸部计算机断层扫描结果 

**Authors**: Mahshid shiri, Chandra Bortolotto, Alessandro Bruno, Alessio Consonni, Daniela Maria Grasso, Leonardo Brizzi, Daniele Loiacono, Lorenzo Preda  

**Link**: [PDF](https://arxiv.org/pdf/2501.15572)  

**Abstract**: Introduction: Generative Adversarial Networks (GANs) are increasingly used to generate synthetic medical images, addressing the critical shortage of annotated data for training Artificial Intelligence (AI) systems. This study introduces a novel memory-efficient GAN architecture, incorporating Conditional Random Fields (CRFs) to generate high-resolution 3D medical images and evaluates its performance against the state-of-the-art hierarchical (HA)-GAN model.
Materials and Methods: The CRF-GAN was trained using the open-source lung CT LUNA16 dataset. The architecture was compared to HA-GAN through a quantitative evaluation, using Frechet Inception Distance (FID) and Maximum Mean Discrepancy (MMD) metrics, and a qualitative evaluation, through a two-alternative forced choice (2AFC) test completed by a pool of 12 resident radiologists, in order to assess the realism of the generated images.
Results: CRF-GAN outperformed HA-GAN with lower FID (0.047 vs. 0.061) and MMD (0.084 vs. 0.086) scores, indicating better image fidelity. The 2AFC test showed a significant preference for images generated by CRF-Gan over those generated by HA-GAN with a p-value of 1.93e-05. Additionally, CRF-GAN demonstrated 9.34% lower memory usage at 256 resolution and achieved up to 14.6% faster training speeds, offering substantial computational savings.
Discussion: CRF-GAN model successfully generates high-resolution 3D medical images with non-inferior quality to conventional models, while being more memory-efficient and faster. Computational power and time saved can be used to improve the spatial resolution and anatomical accuracy of generated images, which is still a critical factor limiting their direct clinical applicability. 

**Abstract (ZH)**: 介绍：生成对抗网络（GANs）越来越多地被用于生成合成医疗图像，以解决训练人工智能（AI）系统所需的标注数据严重短缺问题。本研究介绍了一种新颖的内存高效GAN架构，结合条件随机字段（CRFs），用于生成高分辨率的3D医疗图像，并将其性能与最先进的分层GAN（HA-GAN）模型进行了评估。

材料与方法：CRF-GAN使用开源的肺CT LUNA16数据集进行训练。该架构通过定量评价中的弗雷切-欣腾距离（FID）和最大均值偏差（MMD）指标以及使用12名住院放射科医生完成的二选一强迫选择（2AFC）测试中的定性评价，与HA-GAN进行了比较，以评估生成图像的逼真度。

结果：CRF-GAN在FID（0.047 vs. 0.061）和MMD（0.084 vs. 0.086）得分上优于HA-GAN，表明其图像保真度更高。2AFC测试结果显示，与生成的HA-GAN图像相比，CRF-GAN生成的图像获得了显著偏好，p值为1.93e-05。此外，CRF-GAN在256分辨率下的内存使用量低9.34%，并且训练速度提高了14.6%，提供了显著的计算成本节省。

讨论：CRF-GAN模型成功地生成了与传统模型具有非劣等质量的高分辨率3D医疗图像，同时更加内存高效且训练速度更快。节省的计算能力和时间可以用于提高生成图像的空间分辨率和解剖准确性，这对于直接在临床应用中仍然是一个至关重要的限制因素。 

---
# Diffusion-Based Planning for Autonomous Driving with Flexible Guidance 

**Title (ZH)**: 基于扩散模型的自主驾驶灵活指引规划方法 

**Authors**: Yinan Zheng, Ruiming Liang, Kexin Zheng, Jinliang Zheng, Liyuan Mao, Jianxiong Li, Weihao Gu, Rui Ai, Shengbo Eben Li, Xianyuan Zhan, Jingjing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15564)  

**Abstract**: Achieving human-like driving behaviors in complex open-world environments is a critical challenge in autonomous driving. Contemporary learning-based planning approaches such as imitation learning methods often struggle to balance competing objectives and lack of safety assurance,due to limited adaptability and inadequacy in learning complex multi-modal behaviors commonly exhibited in human planning, not to mention their strong reliance on the fallback strategy with predefined rules. We propose a novel transformer-based Diffusion Planner for closed-loop planning, which can effectively model multi-modal driving behavior and ensure trajectory quality without any rule-based refinement. Our model supports joint modeling of both prediction and planning tasks under the same architecture, enabling cooperative behaviors between vehicles. Moreover, by learning the gradient of the trajectory score function and employing a flexible classifier guidance mechanism, Diffusion Planner effectively achieves safe and adaptable planning behaviors. Evaluations on the large-scale real-world autonomous planning benchmark nuPlan and our newly collected 200-hour delivery-vehicle driving dataset demonstrate that Diffusion Planner achieves state-of-the-art closed-loop performance with robust transferability in diverse driving styles. 

**Abstract (ZH)**: 在复杂开放世界环境中实现类人的驾驶行为是自动驾驶领域的一项关键挑战。当前基于学习的规划方法，如模仿学习方法，往往难以平衡竞争性目标，缺乏安全性保证，这是由于其适应性有限以及在学习人类规划中常见的复杂多模态行为方面能力不足，更不用说它们对预先定义规则的强烈依赖。我们提出了一种新型的基于变换器的扩散规划器，该规划器可以在闭环规划中有效地建模多模态驾驶行为，并在不依赖任何基于规则的细化的情况下确保轨迹质量。我们的模型支持在同一架构下同时建模预测和规划任务，从而实现车辆之间的协同行为。此外，通过学习轨迹得分函数的梯度并采用灵活的分类器指导机制，扩散规划器有效实现了安全和适应性强的规划行为。在大规模真实世界的自主规划基准nuPlan和我们新收集的200小时的配送车辆驾驶数据集上的评估显示，扩散规划器在不同的驾驶风格下表现出强大的泛化能力，实现了最先进的闭环性能。 

---
# CE-SDWV: Effective and Efficient Concept Erasure for Text-to-Image Diffusion Models via a Semantic-Driven Word Vocabulary 

**Title (ZH)**: CE-SDWV：基于语义驱动词词汇的概念擦除方法，实现文本到图像扩散模型的有效且高效的概念消除 

**Authors**: Jiahang Tu, Qian Feng, Chufan Chen, Jiahua Dong, Hanbin Zhao, Chao Zhang, Hui Qian  

**Link**: [PDF](https://arxiv.org/pdf/2501.15562)  

**Abstract**: Large-scale text-to-image (T2I) diffusion models have achieved remarkable generative performance about various concepts. With the limitation of privacy and safety in practice, the generative capability concerning NSFW (Not Safe For Work) concepts is undesirable, e.g., producing sexually explicit photos, and licensed images. The concept erasure task for T2I diffusion models has attracted considerable attention and requires an effective and efficient method. To achieve this goal, we propose a CE-SDWV framework, which removes the target concepts (e.g., NSFW concepts) of T2I diffusion models in the text semantic space by only adjusting the text condition tokens and does not need to re-train the original T2I diffusion model's weights. Specifically, our framework first builds a target concept-related word vocabulary to enhance the representation of the target concepts within the text semantic space, and then utilizes an adaptive semantic component suppression strategy to ablate the target concept-related semantic information in the text condition tokens. To further adapt the above text condition tokens to the original image semantic space, we propose an end-to-end gradient-orthogonal token optimization strategy. Extensive experiments on I2P and UnlearnCanvas benchmarks demonstrate the effectiveness and efficiency of our method. 

**Abstract (ZH)**: 大规模文本到图像（T2I）扩散模型在生成各种概念方面取得了显著的生成性能。然而，在实践中由于隐私和安全的限制，与非工作合适（NSFW，Not Safe For Work）概念相关的生成能力是不希望的，例如生成含有性暗示的图片或受版权保护的图片。T2I扩散模型的概念擦除任务已经引起了广泛关注，并需要一个有效且高效的方法。为实现这一目标，我们提出了CE-SDWV框架，该框架仅通过调整文本条件词元来在文本语义空间中删除T2I扩散模型的目标概念（例如，NSFW概念），而无需重新训练原始T2I扩散模型的权重。具体而言，我们的框架首先构建一个与目标概念相关的词汇表，以增强文本语义空间中目标概念的表示，然后利用自适应语义组件抑制策略在文本条件词元中消除与目标概念相关的语义信息。为了进一步使上述文本条件词元适应原始图像语义空间，我们提出了一种端到端的梯度正交词元优化策略。在I2P和UnlearnCanvas基准上的广泛实验表明了我们方法的有效性和高效性。 

---
# Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model 

**Title (ZH)**: 分布鲁棒的图离分布推荐：基于扩散模型的方法 

**Authors**: Chu Zhao, Enneng Yang, Yuliang Liang, Jianzhe Zhao, Guibing Guo, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15555)  

**Abstract**: The distributionally robust optimization (DRO)-based graph neural network methods improve recommendation systems' out-of-distribution (OOD) generalization by optimizing the model's worst-case performance. However, these studies fail to consider the impact of noisy samples in the training data, which results in diminished generalization capabilities and lower accuracy. Through experimental and theoretical analysis, this paper reveals that current DRO-based graph recommendation methods assign greater weight to noise distribution, leading to model parameter learning being dominated by it. When the model overly focuses on fitting noise samples in the training data, it may learn irrelevant or meaningless features that cannot be generalized to OOD data. To address this challenge, we design a Distributionally Robust Graph model for OOD recommendation (DRGO). Specifically, our method first employs a simple and effective diffusion paradigm to alleviate the noisy effect in the latent space. Additionally, an entropy regularization term is introduced in the DRO objective function to avoid extreme sample weights in the worst-case distribution. Finally, we provide a theoretical proof of the generalization error bound of DRGO as well as a theoretical analysis of how our approach mitigates noisy sample effects, which helps to better understand the proposed framework from a theoretical perspective. We conduct extensive experiments on four datasets to evaluate the effectiveness of our framework against three typical distribution shifts, and the results demonstrate its superiority in both independently and identically distributed distributions (IID) and OOD. 

**Abstract (ZH)**: 基于分布鲁棒优化(DRO)的图神经网络方法通过优化模型在最坏情况下的性能，改善了推荐系统在分布外(OOD)数据上的泛化能力。然而，这些研究未能考虑训练数据中的噪声样本影响，导致泛化能力和准确性降低。通过实验和理论分析，本文揭示当前基于DRO的图推荐方法赋予了噪声分布更大的权重，导致模型参数学习主要受到噪声的影响。当模型过度关注拟合训练数据中的噪声样本时，可能会学习到与OOD数据无法泛化的无关或无意义特征。为应对这一挑战，我们设计了一种用于OOD推荐的分布鲁棒图模型 (DRGO)。具体而言，我们的方法首先采用一种简单有效的扩散范式，以减轻潜在空间中的噪声影响。此外，我们在DRO目标函数中引入熵正则化项，以避免在最坏情况分布中极端的样本权重。最后，我们提供了DRGO泛化误差界的理论证明，并对我们的方法如何减轻噪声样本影响进行了理论分析，有助于从理论角度更好地理解所提出的框架。我们对四个数据集进行了广泛实验，评估了该框架在三种典型分布转变下的有效性，结果表明其在独立同分布(IID)和OOD数据上的优越性。 

---
# Building Efficient Lightweight CNN Models 

**Title (ZH)**: 构建高效的轻量级CNN模型 

**Authors**: Nathan Isong  

**Link**: [PDF](https://arxiv.org/pdf/2501.15547)  

**Abstract**: Convolutional Neural Networks (CNNs) are pivotal in image classification tasks due to their robust feature extraction capabilities. However, their high computational and memory requirements pose challenges for deployment in resource-constrained environments. This paper introduces a methodology to construct lightweight CNNs while maintaining competitive accuracy. The approach integrates two stages of training; dual-input-output model and transfer learning with progressive unfreezing. The dual-input-output model train on original and augmented datasets, enhancing robustness. Progressive unfreezing is applied to the unified model to optimize pre-learned features during fine-tuning, enabling faster convergence and improved model accuracy.
The methodology was evaluated on three benchmark datasets; handwritten digit MNIST, fashion MNIST, and CIFAR-10. The proposed model achieved a state-of-the-art accuracy of 99% on the handwritten digit MNIST and 89% on fashion MNIST, with only 14,862 parameters and a model size of 0.17 MB. While performance on CIFAR-10 was comparatively lower (65% with less than 20,00 parameters), the results highlight the scalability of this method. The final model demonstrated fast inference times and low latency, making it suitable for real-time applications.
Future directions include exploring advanced augmentation techniques, improving architectural scalability for complex datasets, and extending the methodology to tasks beyond classification. This research underscores the potential for creating efficient, scalable, and task-specific CNNs for diverse applications. 

**Abstract (ZH)**: 卷积神经网络（CNNs）在图像分类任务中至关重要，由于其强大的特征提取能力。然而，其高计算和内存需求为在资源受限环境中部署带来了挑战。本文提出了一种方法，通过两阶段训练构建轻量级CNNs，同时保持竞争力的准确性。该方法结合了双输入输出模型和基于逐层解冻的迁移学习。双输入输出模型在原始和增强数据集上进行训练，增强鲁棒性。逐层解冻应用于统一模型，以优化fine-tuning过程中预学习的特征，从而实现更快的收敛和更好的模型准确性。

该方法在三个基准数据集上进行了评估：手写数字MNIST、时尚MNIST和CIFAR-10。所提出模型在手写数字MNIST上达到了99%的最先进的准确率，在时尚MNIST上达到了89%的准确率，仅包含14,862个参数和0.17 MB的模型大小。尽管在CIFAR-10上的表现相对较低（仅65%，参数少于20,000），但结果突显了该方法的可扩展性。最终模型展示了快速推理时间和低延迟，使其适用于实时应用。

未来的研究方向包括探索先进的增强技术、改进架构的可扩展性以适应复杂数据集，并将该方法扩展到分类之外的任务。这项研究强调了创建高效、可扩展和任务特定的CNNs以适应各种应用的潜力。 

---
# Advancing Generative Artificial Intelligence and Large Language Models for Demand Side Management with Electric Vehicles 

**Title (ZH)**: 推动生成式人工智能和大型语言模型在电动车辆需求侧管理中的应用与发展 

**Authors**: Hanwen Zhang, Ruichen Zhang, Wei Zhang, Dusit Niyato, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15544)  

**Abstract**: Generative artificial intelligence, particularly through large language models (LLMs), is poised to transform energy optimization and demand side management (DSM) within microgrids. This paper explores the integration of LLMs into energy management, emphasizing their roles in automating the optimization of DSM strategies with electric vehicles. We investigate challenges and solutions associated with DSM and explore the new opportunities presented by leveraging LLMs. Then, We propose an innovative solution that enhances LLMs with retrieval-augmented generation for automatic problem formulation, code generation, and customizing optimization. We present a case study to demonstrate the effectiveness of our proposed solution in charging scheduling and optimization for electric vehicles, highlighting our solution's significant advancements in energy efficiency and user adaptability. This work underscores the potential of LLMs for energy optimization and fosters a new era of intelligent DSM solutions. 

**Abstract (ZH)**: 生成式人工智能，尤其是在大型语言模型（LLMs）的帮助下，有望重塑微网中的能源优化和需求侧管理（DSM）。本文探讨了将LLMs集成到能源管理系统中的方法，强调了它们在自动化电动汽车DSM策略优化方面的作用。我们研究了DSM面临的挑战及其解决方案，并探索了利用LLMs所带来的新机遇。接下来，我们提出了一种创新解决方案，通过检索增强生成技术增强LLMs，以实现自动问题表述、代码生成和定制优化。我们通过一个案例研究展示了该解决方案在电动汽车充电调度和优化方面的有效性，突显了该解决方案在提高能源效率和用户适应性方面的显著进步。本文强调了LLMs在能源优化领域的潜力，并促进了新一代智能DSM解决方案的发展。 

---
# UNIDOOR: A Universal Framework for Action-Level Backdoor Attacks in Deep Reinforcement Learning 

**Title (ZH)**: UNIDOOR：深度强化学习中基于动作级后门攻击的通用框架 

**Authors**: Oubo Ma, Linkang Du, Yang Dai, Chunyi Zhou, Qingming Li, Yuwen Pu, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.15529)  

**Abstract**: Deep reinforcement learning (DRL) is widely applied to safety-critical decision-making scenarios. However, DRL is vulnerable to backdoor attacks, especially action-level backdoors, which pose significant threats through precise manipulation and flexible activation, risking outcomes like vehicle collisions or drone crashes. The key distinction of action-level backdoors lies in the utilization of the backdoor reward function to associate triggers with target actions. Nevertheless, existing studies typically rely on backdoor reward functions with fixed values or conditional flipping, which lack universality across diverse DRL tasks and backdoor designs, resulting in fluctuations or even failure in practice.
This paper proposes the first universal action-level backdoor attack framework, called UNIDOOR, which enables adaptive exploration of backdoor reward functions through performance monitoring, eliminating the reliance on expert knowledge and grid search. We highlight that action tampering serves as a crucial component of action-level backdoor attacks in continuous action scenarios, as it addresses attack failures caused by low-frequency target actions. Extensive evaluations demonstrate that UNIDOOR significantly enhances the attack performance of action-level backdoors, showcasing its universality across diverse attack scenarios, including single/multiple agents, single/multiple backdoors, discrete/continuous action spaces, and sparse/dense reward signals. Furthermore, visualization results encompassing state distribution, neuron activation, and animations demonstrate the stealthiness of UNIDOOR. The source code of UNIDOOR can be found at this https URL. 

**Abstract (ZH)**: 深度强化学习（DRL）在安全关键的决策情境中得到了广泛的应用。然而，DRL 对后门攻击极为脆弱，尤其是动作级后门攻击，它们通过精确操控和灵活激活带来了严重威胁，可能导致车辆相撞或无人机坠毁。动作级后门攻击的关键在于利用后门奖励函数将触发器与目标动作关联起来。不过，现有研究通常依赖于固定值或条件翻转的后门奖励函数，这在不同类型的DRL任务和后门设计中缺乏普适性，导致在实践中的性能波动甚至失效。

本文提出了第一个适用于动作级后门攻击的通用框架，称为UNIDOOR，该框架通过性能监控实现后门奖励函数的自适应探索，从而消除对专家知识和网格搜索的依赖。我们强调，在连续动作场景中，动作篡改是动作级后门攻击的关键组成部分，因为它解决了由低频目标动作引起攻击失败的问题。广泛的评估结果表明，UNIDOOR 显著提高了动作级后门攻击的性能，并展示了其在各种攻击场景中的普适性，包括单/多智能体、单/多后门、离散/连续动作空间以及稀疏/密集奖励信号。此外，包含状态分布、神经元激活和动画的可视化结果证实了UNIDOOR的高度隐蔽性。UNIDOOR的源代码可在以下链接找到：[此处填写链接]。 

---
# FIT-Print: Towards False-claim-resistant Model Ownership Verification via Targeted Fingerprint 

**Title (ZH)**: FIT-Print: 朝向针对虚假索赔的模型所有权验证方法 via 针对目标特征的指纹识别

注：这一翻译尽量保持了原文的含义和结构，但“via Targeted Fingerprint”部分进行了一些调整以适应中文表达习惯，同时保留了原文的核心概念。如果您有更多具体要求或需要进一步修改，请随时告知。 

**Authors**: Shuo Shao, Haozhe Zhu, Hongwei Yao, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2501.15509)  

**Abstract**: Model fingerprinting is a widely adopted approach to safeguard the intellectual property rights of open-source models by preventing their unauthorized reuse. It is promising and convenient since it does not necessitate modifying the protected model. In this paper, we revisit existing fingerprinting methods and reveal that they are vulnerable to false claim attacks where adversaries falsely assert ownership of any third-party model. We demonstrate that this vulnerability mostly stems from their untargeted nature, where they generally compare the outputs of given samples on different models instead of the similarities to specific references. Motivated by these findings, we propose a targeted fingerprinting paradigm (i.e., FIT-Print) to counteract false claim attacks. Specifically, FIT-Print transforms the fingerprint into a targeted signature via optimization. Building on the principles of FIT-Print, we develop bit-wise and list-wise black-box model fingerprinting methods, i.e., FIT-ModelDiff and FIT-LIME, which exploit the distance between model outputs and the feature attribution of specific samples as the fingerprint, respectively. Extensive experiments on benchmark models and datasets verify the effectiveness, conferrability, and resistance to false claim attacks of our FIT-Print. 

**Abstract (ZH)**: 模型指纹技术是一种广泛采用的方法，用于保护开放源代码模型的知识产权，防止其被未经授权的重用。这种方法很有前景且方便，因为它不需要修改受保护的模型。在本文中，我们重新审视了现有的指纹方法，并揭示出它们容易受到虚假宣称攻击，即对手可以虚假地断言任何第三方模型的所有权。我们证明，这种脆弱性主要来自于它们的非针对性特性，通常比较不同模型上给定样本的输出相似性，而不是特定参考的相似性。基于这些发现，我们提出了一种针对虚假宣称攻击的靶向指纹技术（即FIT-Print）。具体而言，FIT-Print通过优化将指纹转换为目标签名。在FIT-Print的原则基础上，我们开发了位级和列表级的黑盒模型指纹方法，即FIT-ModelDiff和FIT-LIME，分别利用模型输出与特定样本特征归因之间的距离作为指纹。在基准模型和数据集上进行的大量实验验证了FIT-Print的有效性、可传递性和对虚假宣称攻击的抵抗性。 

---
# Color Flow Imaging Microscopy Improves Identification of Stress Sources of Protein Aggregates in Biopharmaceuticals 

**Title (ZH)**: 彩色流场成像显微镜 improves 生物制药中蛋白质 aggregates 压力源的识别 

**Authors**: Michaela Cohrs, Shiwoo Koak, Yejin Lee, Yu Jin Sung, Wesley De Neve, Hristo L. Svilenov, Utku Ozbulak  

**Link**: [PDF](https://arxiv.org/pdf/2501.15492)  

**Abstract**: Protein-based therapeutics play a pivotal role in modern medicine targeting various diseases. Despite their therapeutic importance, these products can aggregate and form subvisible particles (SvPs), which can compromise their efficacy and trigger immunological responses, emphasizing the critical need for robust monitoring techniques. Flow Imaging Microscopy (FIM) has been a significant advancement in detecting SvPs, evolving from monochrome to more recently incorporating color imaging. Complementing SvP images obtained via FIM, deep learning techniques have recently been employed successfully for stress source identification of monochrome SvPs. In this study, we explore the potential of color FIM to enhance the characterization of stress sources in SvPs. To achieve this, we curate a new dataset comprising 16,000 SvPs from eight commercial monoclonal antibodies subjected to heat and mechanical stress. Using both supervised and self-supervised convolutional neural networks, as well as vision transformers in large-scale experiments, we demonstrate that deep learning with color FIM images consistently outperforms monochrome images, thus highlighting the potential of color FIM in stress source classification compared to its monochrome counterparts. 

**Abstract (ZH)**: 蛋白质基疗法在现代医学中扮演着关键角色，针对各种疾病。尽管这些产品具有重要的治疗意义，但它们也可能聚集并形成亚可见粒子（SvPs），这可能损害其疗效并引发免疫反应，强调了需要强大的监控技术的重要性。流形成像显微镜（Flow Imaging Microscopy, FIM）是检测SvPs的一个重要进展，从单色发展到最近开始使用彩色成像技术。借助FIM获得的SvP图像，最近成功地使用深度学习技术识别单色SvP的应力源。在这项研究中，我们探讨了彩色FIM在增强SvP应力源表征方面的潜在能力。为此，我们从八种商业单克隆抗体中收集了16,000个SvP样本，并对其施加热应力和机械应力。在大规模实验中，我们使用监督学习和半监督学习的卷积神经网络以及视觉变换器，证明了基于彩色FIM图像的深度学习方法在应力源分类中始终优于单色图像，从而突显了彩色FIM在应力源分类方面的潜在优势，相比于其单色对应物。 

---
# FedAlign: Federated Domain Generalization with Cross-Client Feature Alignment 

**Title (ZH)**: 联邦特征对齐：跨客户端领域泛化的联邦学习方法 

**Authors**: Sunny Gupta, Vinay Sutar, Varunav Singh, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2501.15486)  

**Abstract**: Federated Learning (FL) offers a decentralized paradigm for collaborative model training without direct data sharing, yet it poses unique challenges for Domain Generalization (DG), including strict privacy constraints, non-i.i.d. local data, and limited domain diversity. We introduce FedAlign, a lightweight, privacy-preserving framework designed to enhance DG in federated settings by simultaneously increasing feature diversity and promoting domain invariance. First, a cross-client feature extension module broadens local domain representations through domain-invariant feature perturbation and selective cross-client feature transfer, allowing each client to safely access a richer domain space. Second, a dual-stage alignment module refines global feature learning by aligning both feature embeddings and predictions across clients, thereby distilling robust, domain-invariant features. By integrating these modules, our method achieves superior generalization to unseen domains while maintaining data privacy and operating with minimal computational and communication overhead. 

**Abstract (ZH)**: 联邦学习（FL）提供了一种无需直接数据共享的去中心化协作模型训练 paradigm，但为此带来了独特的领域泛化（DG）挑战，包括严格的隐私约束、非同质独立同分布（non-i.i.d.）本地数据以及有限的领域多样性。我们提出了 FedAlign，这是一种轻量级、保护隐私的框架，旨在在联邦环境中通过同时增加特征多样性和促进领域不变性来增强 DG。首先，通过领域不变特征扰动和选择性跨客户端特征转移，一个跨客户端特征扩展模块扩展了本地领域表示，使每个客户端能够安全地访问更丰富的领域空间。其次，一个双重阶段对齐模块通过对齐客户端之间的特征嵌入和预测来细化全局特征学习，从而提炼出稳健的、领域不变的特征。通过将这些模块整合，我们的方法能够实现对未见过的领域的优越泛化能力，同时保持数据隐私，并且具有最小的计算和通信开销。 

---
# TractoGPT: A GPT architecture for White Matter Segmentation 

**Title (ZH)**: TractoGPT：一种用于白质分割的GPT架构 

**Authors**: Anoushkrit Goel, Simroop Singh, Ankita Joshi, Ranjeet Ranjan Jha, Chirag Ahuja, Aditya Nigam, Arnav Bhavsar  

**Link**: [PDF](https://arxiv.org/pdf/2501.15464)  

**Abstract**: White matter bundle segmentation is crucial for studying brain structural connectivity, neurosurgical planning, and neurological disorders. White Matter Segmentation remains challenging due to structural similarity in streamlines, subject variability, symmetry in 2 hemispheres, etc. To address these challenges, we propose TractoGPT, a GPT-based architecture trained on streamline, cluster, and fusion data representations separately. TractoGPT is a fully-automatic method that generalizes across datasets and retains shape information of the white matter bundles. Experiments also show that TractoGPT outperforms state-of-the-art methods on average DICE, Overlap and Overreach scores. We use TractoInferno and 105HCP datasets and validate generalization across dataset. 

**Abstract (ZH)**: 白质纤维束分割对于研究大脑结构连接、神经外科规划和神经疾病至关重要。由于纤维束结构的相似性、受试者间的变异性以及左右半球的对称性等原因，白质分割仍然极具挑战性。为应对这些挑战，我们提出了一种基于GPT的TractoGPT架构，该架构分别在纤维束、聚类和融合数据表示上进行训练。TractoGPT是一种全自动方法，能够泛化到不同数据集，并保留白质纤维束的形状信息。实验结果还表明，TractoGPT在平均Dice系数、重叠率和过度估计率方面优于最先进的方法。我们使用TractoInferno和105HCP数据集进行了验证，以证明其在不同数据集上的泛化能力。 

---
# Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values? 

**Title (ZH)**: 注意价值与行动之间的差距：大型语言模型的行为是否与其价值观一致？ 

**Authors**: Hua Shen, Nicholas Clark, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2501.15463)  

**Abstract**: Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps. 

**Abstract (ZH)**: 现有的研究主要通过评估语言模型（LLMs）宣称的特定价值观倾向来评估其价值。然而，在环境心理和社会心理学中，存在着“价值行动差距”这一现象，揭示了个体宣称的价值与其在实际情境中的行为之间的不一致。LLMs 在宣称的价值与其依据这些价值所采取的行为之间是否存在类似的差距？本研究引入了 ValueActionLens 评估框架，用于评估 LLMs 所宣称的价值与其价值导向行为的一致性。该框架包括生成一个包含14800个价值导向行为的跨文化数据集，涉及12种文化与11个社会主题，并通过两个任务评估 LLMs 所宣称的价值倾向及其价值导向行为在三种不同的对齐度量中的表现一致性。大量的实验表明，LLMs 所宣称的价值与其行为之间的一致性并不理想，不同情境和模型之间的差异显著。对于未对齐的结果进行的分析揭示了某些价值行为差距可能带来的潜在风险。为了预测这些价值行为差距，我们还发现利用理由解释可以提高预测性能。这些发现强调，仅依赖 LLMs 所宣称的价值来预测其行为存在风险，并突显了进行具有情境意识的 LLM 价值和价值行为差距评估的重要性。 

---
# Identifying Critical Tokens for Accurate Predictions in Transformer-based Medical Imaging Models 

**Title (ZH)**: 基于变压器的医学影像模型中关键令牌的识别以实现准确预测 

**Authors**: Solha Kang, Joris Vankerschaver, Utku Ozbulak  

**Link**: [PDF](https://arxiv.org/pdf/2501.15452)  

**Abstract**: With the advancements in self-supervised learning (SSL), transformer-based computer vision models have recently demonstrated superior results compared to convolutional neural networks (CNNs) and are poised to dominate the field of artificial intelligence (AI)-based medical imaging in the upcoming years. Nevertheless, similar to CNNs, unveiling the decision-making process of transformer-based models remains a challenge. In this work, we take a step towards demystifying the decision-making process of transformer-based medical imaging models and propose Token Insight, a novel method that identifies the critical tokens that contribute to the prediction made by the model. Our method relies on the principled approach of token discarding native to transformer-based models, requires no additional module, and can be applied to any transformer model. Using the proposed approach, we quantify the importance of each token based on its contribution to the prediction and enable a more nuanced understanding of the model's decisions. Our experimental results which are showcased on the problem of colonic polyp identification using both supervised and self-supervised pretrained vision transformers indicate that Token Insight contributes to a more transparent and interpretable transformer-based medical imaging model, fostering trust and facilitating broader adoption in clinical settings. 

**Abstract (ZH)**: 随着自监督学习（SSL）的进步，基于 transformer 的计算机视觉模型在近年来展现了优于卷积神经网络（CNNs）的性能，并且有望在未来几年占据人工智能（AI）驱动的医学影像领域的主导地位。然而，类似于 CNN，揭开基于 transformer 的模型的决策过程仍然是一项挑战。在本文中，我们向解开基于 transformer 的医学影像模型的决策过程迈出了一步，并提出了一种名为 Token Insight 的新型方法，该方法能够识别出对模型预测做出贡献的关键 token。我们的方法基于基于 transformer 的模型固有的 token 否定的原则，不需要额外的模块，并能够应用于任何 transformer 模型。通过我们提出的这种途径，我们基于 token 对预测的贡献程度定量地量化了每个 token 的重要性，从而增强了对模型决策的理解。我们的实验结果表明，Token Insight 使得基于 transformer 的医学影像模型更加透明和可解释，从而促进了临床应用中的信任并推动了更广泛的采用。这些结果是在使用监督预训练和自监督预训练视觉 transformer 对结肠息肉识别问题进行实验展示得出的。 

---
# SQ-DM: Accelerating Diffusion Models with Aggressive Quantization and Temporal Sparsity 

**Title (ZH)**: SQ-DM：通过激进量化和时间稀疏性加速扩散模型 

**Authors**: Zichen Fan, Steve Dai, Rangharajan Venkatesan, Dennis Sylvester, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2501.15448)  

**Abstract**: Diffusion models have gained significant popularity in image generation tasks. However, generating high-quality content remains notably slow because it requires running model inference over many time steps. To accelerate these models, we propose to aggressively quantize both weights and activations, while simultaneously promoting significant activation sparsity. We further observe that the stated sparsity pattern varies among different channels and evolves across time steps. To support this quantization and sparsity scheme, we present a novel diffusion model accelerator featuring a heterogeneous mixed-precision dense-sparse architecture, channel-last address mapping, and a time-step-aware sparsity detector for efficient handling of the sparsity pattern. Our 4-bit quantization technique demonstrates superior generation quality compared to existing 4-bit methods. Our custom accelerator achieves 6.91x speed-up and 51.5% energy reduction compared to traditional dense accelerators. 

**Abstract (ZH)**: 扩散模型在图像生成任务中获得了显著的关注。然而，生成高质量内容仍然相对缓慢，因为这需要在多个时间步长上运行模型推断。为了加速这些模型，我们提出了一种激进的权重和激活量化方法，同时促进显著的激活稀疏性。我们进一步观察到，所声明的稀疏模式在不同的通道之间存在差异，并且会随时间步长的变化而演变。为了支持这种量化和稀疏方案，我们提出了一种新型的扩散模型加速器，其特征是异构混合精度密集稀疏架构、通道后地址映射以及一种时间步长感知的稀疏性检测器，以高效地处理稀疏模式。我们的4位量化技术在生成质量上优于现有的4位方法。我们定制的加速器比传统的密集加速器快6.91倍，并且能效降低了51.5%。 

---
# Token Democracy: The Architectural Limits of Alignment in Transformer-Based Language Models 

**Title (ZH)**: 代币民主：基于Transformer的語言模型_alignment_的架构限制 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2501.15446)  

**Abstract**: Modern language models paradoxically combine unprecedented capability with persistent vulnerability in that they can draft poetry yet cannot reliably refuse harmful requests. We reveal this fragility stems not from inadequate training, but from a fundamental architectural limitation: transformers process all tokens as equals. Transformers operate as computational democracies, granting equal voice to all tokens. This is a design tragically unsuited for AGI, where we cannot risk adversarial "candidates" hijacking the system. Through formal analysis, we demonstrate that safety instructions fundamentally lack privileged status in transformer architectures, that they compete with adversarial inputs in the same computational arena, making robust alignment through prompting or fine-tuning inherently limited. This "token democracy" explains why jailbreaks bypass even extensively safety-trained models and why positional shifts erode prompt effectiveness. Our work systematizes practitioners' tacit knowledge into an architectural critique, showing current alignment approaches create mere preferences, not constraints. 

**Abstract (ZH)**: 现代语言模型在具有前所未有的能力和持久的脆弱性之间存在悖论，它们能够创作诗歌，却不能可靠地拒绝有害请求。我们揭示这种脆弱性并非源于训练不足，而是因为一种根本性的架构限制：变换器将所有标记视为平等。变换器作为计算民主体系运行，赋予所有标记同等发言权。这种设计对于人工智能通用智能（AGI）来说极其不合适，因为在AGI中我们无法承担恶意“候选者”劫持系统的风险。通过形式分析，我们证明了在变换器架构中，安全指令本质上缺乏特权地位，它们在与对抗性输入相同的计算环境中竞争，因此通过提示或微调实现稳健对齐本质上是有限的。这种“标记民主”解释了为什么破解攻击能够规避即使经过广泛安全训练的模型，并且为什么位置转移会削弱提示的有效性。我们的研究将实践者的隐性知识系统化，揭示当前的对齐方法仅创造了偏好，而非约束。 

---
# StochSync: Stochastic Diffusion Synchronization for Image Generation in Arbitrary Spaces 

**Title (ZH)**: StochSync：任意空间中的图像生成随机扩散同步算法 

**Authors**: Kyeongmin Yeo, Jaihoon Kim, Minhyuk Sung  

**Link**: [PDF](https://arxiv.org/pdf/2501.15445)  

**Abstract**: We propose a zero-shot method for generating images in arbitrary spaces (e.g., a sphere for 360° panoramas and a mesh surface for texture) using a pretrained image diffusion model. The zero-shot generation of various visual content using a pretrained image diffusion model has been explored mainly in two directions. First, Diffusion Synchronization-performing reverse diffusion processes jointly across different projected spaces while synchronizing them in the target space-generates high-quality outputs when enough conditioning is provided, but it struggles in its absence. Second, Score Distillation Sampling-gradually updating the target space data through gradient descent-results in better coherence but often lacks detail. In this paper, we reveal for the first time the interconnection between these two methods while highlighting their differences. To this end, we propose StochSync, a novel approach that combines the strengths of both, enabling effective performance with weak conditioning. Our experiments demonstrate that StochSync provides the best performance in 360° panorama generation (where image conditioning is not given), outperforming previous finetuning-based methods, and also delivers comparable results in 3D mesh texturing (where depth conditioning is provided) with previous methods. 

**Abstract (ZH)**: 我们提出了一种零shot方法，利用预训练的图像扩散模型生成任意空间的图像（例如，对于360°全景图可以在球面上生成，对于纹理可以在网格表面上生成）。主要通过两种方向探索使用预训练图像扩散模型生成各种视觉内容的零shot生成方法。首先，扩散同步方法在提供足够条件的情况下，通过在不同投影空间上联合进行逆向扩散过程并在目标空间中同步它们，可以生成高质量的输出，但在缺乏条件时表现不佳。其次，分数蒸馏采样通过逐步梯度下降更新目标空间数据，在提高一致性方面表现更好，但往往缺乏细节。本文首次揭示了这两种方法之间的联系及其差异，并提出了一种名为StochSync的新方法，该方法结合了这两种方法的优势，在缺乏条件的情况下也能实现有效的性能。我们的实验表明，在360°全景图生成（不提供图像条件）的情况下，StochSync的性能最佳，超过了之前的微调方法，并且在提供了深度条件的3D网格纹理生成中也能与之前的方法交付相当的结果。 

---
# Overview of the Amphion Toolkit (v0.2) 

**Title (ZH)**: Amphion工具包（v0.2）概览 

**Authors**: Jiaqi Li, Xueyao Zhang, Yuancheng Wang, Haorui He, Chaoren Wang, Li Wang, Huan Liao, Junyi Ao, Zeyu Xie, Yiqiao Huang, Junan Zhang, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15442)  

**Abstract**: Amphion is an open-source toolkit for Audio, Music, and Speech Generation, designed to lower the entry barrier for junior researchers and engineers in these fields. It provides a versatile framework that supports a variety of generation tasks and models. In this report, we introduce Amphion v0.2, the second major release developed in 2024. This release features a 100K-hour open-source multilingual dataset, a robust data preparation pipeline, and novel models for tasks such as text-to-speech, audio coding, and voice conversion. Furthermore, the report includes multiple tutorials that guide users through the functionalities and usage of the newly released models. 

**Abstract (ZH)**: Amphion 是一个开源工具包，主要用于音频、音乐和语音生成，旨在降低这些领域的初级研究人员和工程师的入门门槛。它提供了一个多功能框架，支持多种生成任务和模型。在本报告中，我们介绍 Amphion v0.2，这是 2024 年开发的第二次重要发布版本。该版本包含了包含 100 万小时的多语言开源数据集、稳健的数据准备管道，以及用于文本转语音、音频编码和声音转换等任务的新型模型。此外，报告中还包括了多个教程，旨在指导用户了解新发布模型的功能和使用方法。 

---
# Self-supervised Benchmark Lottery on ImageNet: Do Marginal Improvements Translate to Improvements on Similar Datasets? 

**Title (ZH)**: 自我监督基准彩票实验在ImageNet上：边际改进是否转化为类似数据集上的改进？ 

**Authors**: Utku Ozbulak, Esla Timothy Anzaku, Solha Kang, Wesley De Neve, Joris Vankerschaver  

**Link**: [PDF](https://arxiv.org/pdf/2501.15431)  

**Abstract**: Machine learning (ML) research strongly relies on benchmarks in order to determine the relative effectiveness of newly proposed models. Recently, a number of prominent research effort argued that a number of models that improve the state-of-the-art by a small margin tend to do so by winning what they call a "benchmark lottery". An important benchmark in the field of machine learning and computer vision is the ImageNet where newly proposed models are often showcased based on their performance on this dataset. Given the large number of self-supervised learning (SSL) frameworks that has been proposed in the past couple of years each coming with marginal improvements on the ImageNet dataset, in this work, we evaluate whether those marginal improvements on ImageNet translate to improvements on similar datasets or not. To do so, we investigate twelve popular SSL frameworks on five ImageNet variants and discover that models that seem to perform well on ImageNet may experience significant performance declines on similar datasets. Specifically, state-of-the-art frameworks such as DINO and Swav, which are praised for their performance, exhibit substantial drops in performance while MoCo and Barlow Twins displays comparatively good results. As a result, we argue that otherwise good and desirable properties of models remain hidden when benchmarking is only performed on the ImageNet validation set, making us call for more adequate benchmarking. To avoid the "benchmark lottery" on ImageNet and to ensure a fair benchmarking process, we investigate the usage of a unified metric that takes into account the performance of models on other ImageNet variant datasets. 

**Abstract (ZH)**: 机器学习（ML）研究强烈依赖基准来确定新提出的模型的相对有效性。最近，一些重要的研究工作指出，那些仅在边际上提高现有最佳水平的模型往往通过所谓的“基准彩票”来实现这一目标。在机器学习和计算机视觉领域，ImageNet是一个重要的基准，在这个基准上，新提出模型的性能通常被展示出来。鉴于近年来提出了一系列自监督学习（SSL）框架，每个框架都在ImageNet数据集上取得了边际改进，本研究旨在评估这些边际改进是否在类似数据集上也得到了体现。为此，我们在五个ImageNet变体上调查了十二种流行的SSL框架，并发现那些在ImageNet上表现良好的模型在类似数据集上可能会出现显著的性能下降。特别是，以DINO和SwAV为代表的顶尖框架，尽管因其性能而受到赞扬，但其性能却出现了显著下降，而MoCo和Barlow Twins则表现出相对较好的结果。因此，我们认为，在仅在ImageNet验证集上进行基准测试时，模型的其他良好和 desirable 的属性会被隐藏，我们呼吁实施更合适的基准测试。为了在ImageNet上避免“基准彩票”现象，并确保公正的基准测试过程，我们调查了在其他ImageNet变体数据集上综合评估模型性能的统一指标的使用情况。 

---
# Visual Generation Without Guidance 

**Title (ZH)**: 无需指导的视觉生成 

**Authors**: Huayu Chen, Kai Jiang, Kaiwen Zheng, Jianfei Chen, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15420)  

**Abstract**: Classifier-Free Guidance (CFG) has been a default technique in various visual generative models, yet it requires inference from both conditional and unconditional models during sampling. We propose to build visual models that are free from guided sampling. The resulting algorithm, Guidance-Free Training (GFT), matches the performance of CFG while reducing sampling to a single model, halving the computational cost. Unlike previous distillation-based approaches that rely on pretrained CFG networks, GFT enables training directly from scratch. GFT is simple to implement. It retains the same maximum likelihood objective as CFG and differs mainly in the parameterization of conditional models. Implementing GFT requires only minimal modifications to existing codebases, as most design choices and hyperparameters are directly inherited from CFG. Our extensive experiments across five distinct visual models demonstrate the effectiveness and versatility of GFT. Across domains of diffusion, autoregressive, and masked-prediction modeling, GFT consistently achieves comparable or even lower FID scores, with similar diversity-fidelity trade-offs compared with CFG baselines, all while being guidance-free. Code will be available at this https URL. 

**Abstract (ZH)**: 以下是翻译成中文的论文内容或标题，符合学术规范：

Classifier-Free Guidance (CFG) 已成为各种视觉生成模型中的默认技术，但在采样过程中需要从条件模型和无条件模型中进行推断。我们提出了一种无需引导采样的视觉模型。最终算法被称为无引导训练 (GFT)，它在保持与CFG相同性能的同时，将采样过程简化为单个模型，减少了计算成本的50%。与依赖于预训练CFG网络的前序基于蒸馏的方法不同，GFT可以直接从零开始进行训练。GFT 实现起来非常简单，它保留了与CFG相同的极大似然目标，主要区别在于条件模型的参数化方式。实现GFT只需要对现有代码库进行少量修改，因为大多数设计选择和超参数都直接继承自CFG。我们跨五个不同领域的视觉模型进行了广泛实验，证明了GFT的有效性和灵活性。在扩散模型、自回归模型和掩码预测模型等多个领域，GFT 一致地实现了可比甚至更低的FID分数，与CFG基线相比具有相似的多样性和精度权衡，同时完全去除了引导。相关代码可以在 <这个链接> 获取。 

---
# Episodic Novelty Through Temporal Distance 

**Title (ZH)**: 通过时间距离实现的 episodic 新鲜感 

**Authors**: Yuhua Jiang, Qihan Liu, Yiqin Yang, Xiaoteng Ma, Dianyu Zhong, Hao Hu, Jun Yang, Bin Liang, Bo Xu, Chongjie Zhang, Qianchuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15418)  

**Abstract**: Exploration in sparse reward environments remains a significant challenge in reinforcement learning, particularly in Contextual Markov Decision Processes (CMDPs), where environments differ across episodes. Existing episodic intrinsic motivation methods for CMDPs primarily rely on count-based approaches, which are ineffective in large state spaces, or on similarity-based methods that lack appropriate metrics for state comparison. To address these shortcomings, we propose Episodic Novelty Through Temporal Distance (ETD), a novel approach that introduces temporal distance as a robust metric for state similarity and intrinsic reward computation. By employing contrastive learning, ETD accurately estimates temporal distances and derives intrinsic rewards based on the novelty of states within the current episode. Extensive experiments on various benchmark tasks demonstrate that ETD significantly outperforms state-of-the-art methods, highlighting its effectiveness in enhancing exploration in sparse reward CMDPs. 

**Abstract (ZH)**: 在稀疏奖励环境中的探索仍然是强化学习中的一个重大挑战，尤其是在情境马尔可夫决策过程（CMDPs）中，这些环境在各期之间有所不同。现有的CMDPs期次内在动机方法主要依赖于基于计数的方法，这种方法在大规模状态空间中不有效，或者依赖于基于相似性的方法，这些方法缺乏适当的状态比较度量。为了解决这些缺点，我们提出了一种新颖的方法——基于时间距离的期次新颖性（Episodic Novelty Through Temporal Distance, ETD），该方法引入了时间距离作为状态相似性和内在奖赏计算的稳健度量。通过采用对比学习，ETD 准确地估计了时间距离，并根据当前期次中状态的新颖性来推导内在奖赏。在各种基准任务上的广泛实验表明，ETD 显著优于现有最先进的方法，突显了其在增强稀疏奖励 CMDPs 的探索方面的有效性。 

---
# AnyEnhance: A Unified Generative Model with Prompt-Guidance and Self-Critic for Voice Enhancement 

**Title (ZH)**: AnyEnhance：一种结合提示引导和自我批评的统一生成模型用于语音增强 

**Authors**: Junan Zhang, Jing Yang, Zihao Fang, Yuancheng Wang, Zehua Zhang, Zhuo Wang, Fan Fan, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15417)  

**Abstract**: We introduce AnyEnhance, a unified generative model for voice enhancement that processes both speech and singing voices. Based on a masked generative model, AnyEnhance is capable of handling both speech and singing voices, supporting a wide range of enhancement tasks including denoising, dereverberation, declipping, super-resolution, and target speaker extraction, all simultaneously and without fine-tuning. AnyEnhance introduces a prompt-guidance mechanism for in-context learning, which allows the model to natively accept a reference speaker's timbre. In this way, it could boost enhancement performance when a reference audio is available and enable the target speaker extraction task without altering the underlying architecture. Moreover, we also introduce a self-critic mechanism into the generative process for masked generative models, yielding higher-quality outputs through iterative self-assessment and refinement. Extensive experiments on various enhancement tasks demonstrate AnyEnhance outperforms existing methods in terms of both objective metrics and subjective listening tests. Demo audios are publicly available at this https URL. 

**Abstract (ZH)**: 我们介绍了一种统一的生成模型——AnyEnhance，该模型能够同时处理语音和歌声。基于带掩码的生成模型，AnyEnhance 能够处理各种语音和歌声，支持包括去噪、消混响、去咔嘶、超分辨率和目标说话人提取等一系列广泛的任务，同时且无需微调。AnyEnhance 引入了一种提示引导机制以实现上下文学习，使模型能够自然接受参考说话人的音色。在这种机制下，当有参考音频时，它能够提升增强效果，并能够在不改变基础架构的情况下执行目标说话人提取任务。此外，我们还引入了一种自我批评机制到生成过程中的掩码生成模型中，通过迭代自我评估和修正得到更高质量的输出。在多种增强任务上的广泛实验表明，AnyEnhance 在客观指标和主观听感测试方面均优于现有方法。相关信息的演示音频可在以下链接中公开获取：[这里填写链接]。 

---
# TdAttenMix: Top-Down Attention Guided Mixup 

**Title (ZH)**: TdAttenMix: 顶部注意力引导的Mixup 

**Authors**: Zhiming Wang, Lin Gu, Feng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15409)  

**Abstract**: CutMix is a data augmentation strategy that cuts and pastes image patches to mixup training data. Existing methods pick either random or salient areas which are often inconsistent to labels, thus misguiding the training model. By our knowledge, we integrate human gaze to guide cutmix for the first time. Since human attention is driven by both high-level recognition and low-level clues, we propose a controllable Top-down Attention Guided Module to obtain a general artificial attention which balances top-down and bottom-up attention. The proposed TdATttenMix then picks the patches and adjust the label mixing ratio that focuses on regions relevant to the current label. Experimental results demonstrate that our TdAttenMix outperforms existing state-of-the-art mixup methods across eight different benchmarks. Additionally, we introduce a new metric based on the human gaze and use this metric to investigate the issue of image-label inconsistency. Project page: \url{this https URL} 

**Abstract (ZH)**: CutMix是一种数据增强策略，通过切割和粘贴图像片段来混合训练数据。现有的方法要么选择随机区域，要么选择显眼的区域，这些区域往往与标签不一致，从而误导了训练模型。据我们所知，我们首次将人类注视引导用于CutMix。由于人类注意力既受高层次识别驱动，也受低层次线索驱动，我们提出了一种可控的自上而下注意引导模块，以获得一种平衡自上而下和自下而上注意的通用人工注意。然后，所提出的TdAttenMix选择这些片段并调整标签混合比例，使其集中在与当前标签相关的区域。实验结果表明，我们的TdAttenMix在八个不同的基准测试中均优于现有的最先进的混合方法。此外，我们介绍了基于人类注视的新度量标准，并使用该度量标准来研究图像-标签不一致的问题。项目页面：[请点击这里](this https URL) 

---
# Turn That Frown Upside Down: FaceID Customization via Cross-Training Data 

**Title (ZH)**: 《颠倒忧郁的微笑：通过跨训练数据实现FaceID个性化》 

**Authors**: Shuhe Wang, Xiaoya Li, Xiaofei Sun, Guoyin Wang, Tianwei Zhang, Jiwei Li, Eduard Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2501.15407)  

**Abstract**: Existing face identity (FaceID) customization methods perform well but are limited to generating identical faces as the input, while in real-world applications, users often desire images of the same person but with variations, such as different expressions (e.g., smiling, angry) or angles (e.g., side profile). This limitation arises from the lack of datasets with controlled input-output facial variations, restricting models' ability to learn effective modifications.
To address this issue, we propose CrossFaceID, the first large-scale, high-quality, and publicly available dataset specifically designed to improve the facial modification capabilities of FaceID customization models. Specifically, CrossFaceID consists of 40,000 text-image pairs from approximately 2,000 persons, with each person represented by around 20 images showcasing diverse facial attributes such as poses, expressions, angles, and adornments. During the training stage, a specific face of a person is used as input, and the FaceID customization model is forced to generate another image of the same person but with altered facial features. This allows the FaceID customization model to acquire the ability to personalize and modify known facial features during the inference stage. Experiments show that models fine-tuned on the CrossFaceID dataset retain its performance in preserving FaceID fidelity while significantly improving its face customization capabilities.
To facilitate further advancements in the FaceID customization field, our code, constructed datasets, and trained models are fully available to the public. 

**Abstract (ZH)**: 现有的面部身份（FaceID）定制方法表现良好，但仅限于生成与输入相同的面部特征，而在实际应用中，用户往往希望获得同一个人但具有变异性的图像，如不同的表情（例如微笑、愤怒）或角度（例如侧面轮廓）。这种限制源于缺乏具有可控输入-输出面部变异的数据集，限制了模型学习有效修改的能力。

为解决这一问题，我们提出了CrossFaceID，这是第一个大规模、高质量且公开可用的数据集，专门设计用于提升FaceID定制模型的面部修改能力。具体来说，CrossFaceID 包含约2,000人的40,000个文本-图像对，每人通过约20张图片展示不同面部属性，如姿态、表情、角度和装饰。在训练阶段，使用某人的特定面部作为输入，要求FaceID定制模型生成同一个人但面部特征有所改变的另一张图像。这使得FaceID定制模型在推理阶段能够学习个性化和修改已知面部特征的能力。实验结果显示，通过CrossFaceID数据集 fine-tuned 的模型在保持FaceID忠真度的同时，显著提高了其面部定制能力。

为促进FaceID定制领域的进一步发展，我们已将代码、构建的数据集和训练的模型完全公开。 

---
# Semantic Layered Embedding Diffusion in Large Language Models for Multi-Contextual Consistency 

**Title (ZH)**: 大型语言模型中多上下文一致性中的语义分层嵌入扩散 

**Authors**: Irin Kabakum, Thomas Montgomery, Daniel Ravenwood, Genevieve Harrington  

**Link**: [PDF](https://arxiv.org/pdf/2501.15405)  

**Abstract**: The Semantic Layered Embedding Diffusion (SLED) mechanism redefines the representation of hierarchical semantics within transformer-based architectures, enabling enhanced contextual consistency across a wide array of linguistic tasks. By introducing a multi-layered diffusion process grounded in spectral analysis, it achieves a complex balance between global and local semantic coherence. Experimental results demonstrate significant improvements in perplexity and BLEU scores, emphasizing the mechanism's ability to adapt effectively across diverse domains, including multilingual and cross-domain text generation. A rigorous mathematical framework underpins the embedding diffusion process, incorporating weighted adjacency matrices, kernel-based refinements, and dynamic layer-wise normalization. Error distribution analysis reveals that SLED addresses challenges in semantic alignment and coherence, outperforming baseline approaches across varied benchmarks. Scalability studies illustrate that its performance gains are maintained consistently across different model sizes, reflecting a practical balance between computational efficiency and linguistic precision. The implementation also achieves energy efficiency, reducing resource consumption during training and inference phases without compromising accuracy. Qualitative case studies further validate its adaptability to extended narratives and context-intensive scenarios, highlighting the mechanism's potential for real-world applications. SLED offers a different perspective on embedding design and its implications for advancing language modeling. 

**Abstract (ZH)**: SLED（语义分层嵌入扩散）机制重新定义了基于变换器架构中的层级语义表示，从而在广泛的语言任务中增强了上下文一致性。通过基于谱分析引入多层扩散过程，它在全局和局部语义一致性之间实现了复杂的平衡。实验结果表明，在困惑度和BLEU评分上取得了显著改进，强调了该机制在多种不同领域的有效适应能力，包括多语言和跨域文本生成。该机制嵌入扩散过程的数学框架严谨，结合了加权邻接矩阵、核基础改进和动态层归一化。误差分布分析表明，SLED在解决语义对齐和连贯性的挑战方面优于基础方法，在多种基准测试中表现更佳。扩展性研究表明，其性能增益在不同模型规模下保持一致，反映了在计算效率和语言精度之间的实用平衡。同时，实现也实现了能效优化，在训练和推理阶段降低资源消耗而不影响准确性。定性案例研究进一步验证了其对扩展叙事和情境密集场景的适应性，突显了该机制在实际应用中的潜力。SLED为嵌入设计及其对推进语言建模的影响提供了新的视角。 

---
# MetaOcc: Surround-View 4D Radar and Camera Fusion Framework for 3D Occupancy Prediction with Dual Training Strategies 

**Title (ZH)**: MetaOcc：一种基于双训练策略的四维雷达和摄像头融合框架，用于三维占用率预测 

**Authors**: Long Yang, Lianqing Zheng, Wenjin Ai, Minghao Liu, Sen Li, Qunshu Lin, Shengyu Yan, Jie Bai, Zhixiong Ma, Xichan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15384)  

**Abstract**: 3D occupancy prediction is crucial for autonomous driving perception. Fusion of 4D radar and camera provides a potential solution of robust occupancy prediction on serve weather with least cost. How to achieve effective multi-modal feature fusion and reduce annotation costs remains significant challenges. In this work, we propose MetaOcc, a novel multi-modal occupancy prediction framework that fuses surround-view cameras and 4D radar for comprehensive environmental perception. We first design a height self-attention module for effective 3D feature extraction from sparse radar points. Then, a local-global fusion mechanism is proposed to adaptively capture modality contributions while handling spatio-temporal misalignments. Temporal alignment and fusion module is employed to further aggregate historical feature. Furthermore, we develop a semi-supervised training procedure leveraging open-set segmentor and geometric constraints for pseudo-label generation, enabling robust perception with limited annotations. Extensive experiments on OmniHD-Scenes dataset demonstrate that MetaOcc achieves state-of-the-art performance, surpassing previous methods by significant margins. Notably, as the first semi-supervised 4D radar and camera fusion-based occupancy prediction approach, MetaOcc maintains 92.5% of the fully-supervised performance while using only 50% of ground truth annotations, establishing a new benchmark for multi-modal 3D occupancy prediction. Code and data are available at this https URL. 

**Abstract (ZH)**: 三维占用预测对于自主驾驶感知至关重要。融合四维雷达和摄像头数据提供了一种在恶劣天气条件下实现稳健占用预测的潜在解决方案，且成本较低。如何实现有效的多模态特征融合以及降低注释成本仍然是重要的挑战。在本文中，我们提出了一种名为MetaOcc的新颖多模态占用预测框架，该框架融合环视摄像头和四维雷达数据，以实现全面的环境感知。我们首先设计了一个高度自注意力模块，用于从稀疏的雷达点中有效提取三维特征。然后，提出了一种局部-全局融合机制，以自适应地捕获模态贡献并处理空间-时间对齐问题。此外，我们使用开放集分割器和几何约束开发了一种半监督训练方法，以生成伪标签，从而在有限注释下实现稳健的感知。在OmniHD-Scenes数据集上的广泛实验表明，MetaOcc达到了最先进的性能，显著优于之前的算法。值得注意的是，作为首个基于四维雷达和摄像头融合的半监督占用预测方法，MetaOcc仅使用50%的地面真实注释，就能保持与完全监督性能92.5%的水平，从而为多模态3D占用预测设立了新的基准。代码和数据可在以下链接处获取：[请提供具体链接]。 

---
# Zero-Shot Interactive Text-to-Image Retrieval via Diffusion-Augmented Representations 

**Title (ZH)**: 基于扩散增强表示的零样本互动文本到图像检索 

**Authors**: Zijun Long, Kangheng Liang, Gerardo Aragon-Camarasa, Richard Mccreadie, Paul Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2501.15379)  

**Abstract**: Interactive Text-to-Image Retrieval (I-TIR) has emerged as a transformative user-interactive tool for applications in domains such as e-commerce and education. Yet, current methodologies predominantly depend on finetuned Multimodal Large Language Models (MLLMs), which face two critical limitations: (1) Finetuning imposes prohibitive computational overhead and long-term maintenance costs. (2) Finetuning narrows the pretrained knowledge distribution of MLLMs, reducing their adaptability to novel scenarios. These issues are exacerbated by the inherently dynamic nature of real-world I-TIR systems, where queries and image databases evolve in complexity and diversity, often deviating from static training distributions. To overcome these constraints, we propose Diffusion Augmented Retrieval (DAR), a paradigm-shifting framework that bypasses MLLM finetuning entirely. DAR synergizes Large Language Model (LLM)-guided query refinement with Diffusion Model (DM)-based visual synthesis to create contextually enriched intermediate representations. This dual-modality approach deciphers nuanced user intent more holistically, enabling precise alignment between textual queries and visually relevant images. Rigorous evaluations across four benchmarks reveal DAR's dual strengths: (1) Matches state-of-the-art finetuned I-TIR models on straightforward queries without task-specific training. (2) Scalable Generalization: Surpasses finetuned baselines by 7.61% in Hits@10 (top-10 accuracy) under multi-turn conversational complexity, demonstrating robustness to intricate, distributionally shifted interactions. By eliminating finetuning dependencies and leveraging generative-augmented representations, DAR establishes a new trajectory for efficient, adaptive, and scalable cross-modal retrieval systems. 

**Abstract (ZH)**: 交互式文本到图像检索（I-TIR）已成为电子商务和教育等领域的一种变革性的用户交互工具。然而，当前的方法主要依赖于微调的多模态大型语言模型（MLLMs），面临着两个关键限制：（1）微调带来了巨大的计算开销和长期维护成本。（2）微调限制了MLLMs的知识分布，降低了其适应新场景的能力。这些问题在实际世界中交互式文本到图像检索系统中尤为突出，因为这些系统的查询和图像数据库随着复杂性和多样性的发展，经常偏离静态的训练分布。为了解决这些限制，我们提出了扩散增强检索（DAR）框架，该框架完全绕过了对MLLM的微调。DAR结合了大型语言模型（LLM）引导的查询细化与基于扩散模型（DM）的视觉合成，生成语境丰富的中间表示。这种双模态方法更全面地解开了用户的意图，使得文本查询与相关图像之间实现了精确的对齐。针对四个基准的严格评估揭示了DAR的双重优势：（1）对于简单的查询，DAR无需特定任务的训练即可匹现状有的最佳微调I-TIR模型。（2）可扩展的泛化能力：在多轮对话复杂性下，DAR在HITS@10指标上比微调基线高出7.61%，展示了对复杂、分布转移交互的稳健性。通过消除对微调的依赖并利用生成性增强表示，DAR为高效、适应性强且可扩展的跨模态检索系统设立了新的方向。 

---
# Evaluating the Effectiveness of XAI Techniques for Encoder-Based Language Models 

**Title (ZH)**: 评估面向编码器基础语言模型的可解释性人工智能技术的有效性 

**Authors**: Melkamu Abay Mersha, Mesay Gemeda Yigezu, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2501.15374)  

**Abstract**: The black-box nature of large language models (LLMs) necessitates the development of eXplainable AI (XAI) techniques for transparency and trustworthiness. However, evaluating these techniques remains a challenge. This study presents a general evaluation framework using four key metrics: Human-reasoning Agreement (HA), Robustness, Consistency, and Contrastivity. We assess the effectiveness of six explainability techniques from five different XAI categories model simplification (LIME), perturbation-based methods (SHAP), gradient-based approaches (InputXGradient, Grad-CAM), Layer-wise Relevance Propagation (LRP), and attention mechanisms-based explainability methods (Attention Mechanism Visualization, AMV) across five encoder-based language models: TinyBERT, BERTbase, BERTlarge, XLM-R large, and DeBERTa-xlarge, using the IMDB Movie Reviews and Tweet Sentiment Extraction (TSE) datasets. Our findings show that the model simplification-based XAI method (LIME) consistently outperforms across multiple metrics and models, significantly excelling in HA with a score of 0.9685 on DeBERTa-xlarge, robustness, and consistency as the complexity of large language models increases. AMV demonstrates the best Robustness, with scores as low as 0.0020. It also excels in Consistency, achieving near-perfect scores of 0.9999 across all models. Regarding Contrastivity, LRP performs the best, particularly on more complex models, with scores up to 0.9371. 

**Abstract (ZH)**: 大型语言模型（LLMs）的黑箱性质 necessitates the development of eXplainable AI (XAI) techniques for transparency and trustworthiness. However, evaluating these techniques remains a challenge. This study presents a general evaluation framework using four key metrics: Human-reasoning Agreement (HA), Robustness, Consistency, and Contrastivity. We assess the effectiveness of six explainability techniques from five different XAI categories: model simplification (LIME), perturbation-based methods (SHAP), gradient-based approaches (InputXGradient, Grad-CAM), Layer-wise Relevance Propagation (LRP), and attention mechanisms-based explainability methods (Attention Mechanism Visualization, AMV). These techniques are evaluated across five encoder-based language models: TinyBERT, BERTbase, BERTlarge, XLM-R large, and DeBERTa-xlarge, using the IMDB Movie Reviews and Tweet Sentiment Extraction (TSE) datasets. Our findings show that the model simplification-based XAI method (LIME) consistently outperforms across multiple metrics and models, significantly excelling in HA with a score of 0.9685 on DeBERTa-xlarge, robustness, and consistency as the complexity of large language models increases. AMV demonstrates the best Robustness, with scores as low as 0.0020. It also excels in Consistency, achieving near-perfect scores of 0.9999 across all models. Regarding Contrastivity, LRP performs the best, particularly on more complex models, with scores up to 0.9371.

翻译如下：

大型语言模型（LLMs）的黑箱性质要求发展可解释的人工智能（XAI）技术以提高透明度和可信度。然而，评估这些技术仍是一项挑战。本研究提出了一种使用四个关键指标（Human-reasoning Agreement (HA)、稳健性、一致性和对比性）的一般评估框架。我们评估了来自五个不同XAI类别的六种解释性技术的有效性：模型简化（LIME）、扰动方法（SHAP）、梯度方法（InputXGradient、Grad-CAM）、逐层相关性传播（LRP）和基于注意机制的解释方法（Attention Mechanism Visualization, AMV）。这些技术是在五种基于编码器的语言模型上进行评估的，包括TinyBERT、BERTbase、BERTlarge、XLM-R large和DeBERTa-xlarge，使用IMDB电影评论和推文情感提取（TSE）数据集。我们的研究发现表明，基于模型简化的方法（LIME）在多个指标和模型中表现最佳，在DeBERTa-xlarge上的HA得分为0.9685，并且随着大型语言模型复杂性的增加，在稳健性和一致性方面表现出色。AMV在稳健性方面表现最佳，得分为0.0020。同时，在一致性方面也表现出色，所有模型中接近完美的得分为0.9999。关于对比性，逐层相关性传播（LRP）在复杂模型上表现最佳，得分为0.9371。 

---
# Learning-Enhanced Safeguard Control for High-Relative-Degree Systems: Robust Optimization under Disturbances and Faults 

**Title (ZH)**: 基于学习增强的高相对度系统保护控制：扰动与故障下的稳健优化 

**Authors**: Xinyang Wang, Hongwei Zhang, Shimin Wang, Wei Xiao, Martin Guay  

**Link**: [PDF](https://arxiv.org/pdf/2501.15373)  

**Abstract**: Merely pursuing performance may adversely affect the safety, while a conservative policy for safe exploration will degrade the performance. How to balance the safety and performance in learning-based control problems is an interesting yet challenging issue. This paper aims to enhance system performance with safety guarantee in solving the reinforcement learning (RL)-based optimal control problems of nonlinear systems subject to high-relative-degree state constraints and unknown time-varying disturbance/actuator faults. First, to combine control barrier functions (CBFs) with RL, a new type of CBFs, termed high-order reciprocal control barrier function (HO-RCBF) is proposed to deal with high-relative-degree constraints during the learning process. Then, the concept of gradient similarity is proposed to quantify the relationship between the gradient of safety and the gradient of performance. Finally, gradient manipulation and adaptive mechanisms are introduced in the safe RL framework to enhance the performance with a safety guarantee. Two simulation examples illustrate that the proposed safe RL framework can address high-relative-degree constraint, enhance safety robustness and improve system performance. 

**Abstract (ZH)**: 单纯追求性能可能会对安全性产生不利影响，而保守的安全探索策略又会降低性能。在学习控制问题中如何权衡安全性和性能是一个有趣而具有挑战性的问题。本文旨在通过结合控制屏障函数（CBFs）和强化学习（RL），解决非线性系统在高相对度状态约束和未知时间varying干扰/执行器故障情况下基于RL的最优控制问题，同时保证系统的安全性。首先，为了将CBFs与RL相结合，提出了一种新的高阶倒数控制屏障函数（HO-RCBF），以在学习过程中处理高相对度约束。然后，提出了梯度相似性的概念，以量化安全性梯度和性能梯度之间的关系。最后，引入了梯度操控和自适应机制，以在安全的RL框架中提升性能并保证安全性。两个仿真示例说明了所提出的安全RL框架能够处理高相对度约束、增强安全鲁棒性并改善系统性能。 

---
# Scaling Large Vision-Language Models for Enhanced Multimodal Comprehension In Biomedical Image Analysis 

**Title (ZH)**: 面向生物医学图像分析的大型视觉-语言模型的扩展以增强多模态理解 

**Authors**: Robinson Umeike, Neil Getty, Fangfang Xia, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2501.15370)  

**Abstract**: Large language models (LLMs) have demonstrated immense capabilities in understanding textual data and are increasingly being adopted to help researchers accelerate scientific discovery through knowledge extraction (information retrieval), knowledge distillation (summarizing key findings and methodologies into concise forms), and knowledge synthesis (aggregating information from multiple scientific sources to address complex queries, generate hypothesis and formulate experimental plans). However, scientific data often exists in both visual and textual modalities. Vision language models (VLMs) address this by incorporating a pretrained vision backbone for processing images and a cross-modal projector that adapts image tokens into the LLM dimensional space, thereby providing richer multimodal comprehension. Nevertheless, off-the-shelf VLMs show limited capabilities in handling domain-specific data and are prone to hallucinations. We developed intelligent assistants finetuned from LLaVA models to enhance multimodal understanding in low-dose radiation therapy (LDRT)-a benign approach used in the treatment of cancer-related illnesses. Using multilingual data from 42,673 articles, we devise complex reasoning and detailed description tasks for visual question answering (VQA) benchmarks. Our assistants, trained on 50,882 image-text pairs, demonstrate superior performance over base models as evaluated using LLM-as-a-judge approach, particularly in reducing hallucination and improving domain-specific comprehension. 

**Abstract (ZH)**: 大型语言模型（LLMs）在理解文本数据方面展现了巨大的能力，并且越来越多地被用于通过知识提取（信息检索）、知识精简（将关键发现和方法学总结为简洁的形式）和知识综合（从多个科学来源汇总信息以解决复杂查询、形成假设和编制实验计划）来帮助研究人员加速科学发现。然而，科学数据通常以视觉和文本两种模态存在。视觉语言模型（VLMs）通过引入预训练的视觉骨干网络来处理图像，并通过跨模态投影器将图像标记转换到LLM的空间，从而提供更丰富的跨模态理解。尽管现成的VLMs在处理领域特定数据方面的能力有限，并且容易产生幻觉，我们开发了基于LLaVA模型微调的智能助手，以增强低剂量辐射治疗（LDRT）领域的跨模态理解——LDRT是一种用于治疗癌症相关疾病的良性方法。我们使用来自42,673篇文章的多语言数据，为视觉问答（VQA）基准设计了复杂的推理和详细的描述任务。我们的助手在50,882幅图像-文本对上进行训练，在LLM作为评判员的方法评估中表现出色，特别是在减少幻觉和提高领域特定理解方面。 

---
# iFormer: Integrating ConvNet and Transformer for Mobile Application 

**Title (ZH)**: iFormer：结合卷积神经网络和Transformer的移动应用模型 

**Authors**: Chuanyang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.15369)  

**Abstract**: We present a new family of mobile hybrid vision networks, called iFormer, with a focus on optimizing latency and accuracy on mobile applications. iFormer effectively integrates the fast local representation capacity of convolution with the efficient global modeling ability of self-attention. The local interactions are derived from transforming a standard convolutional network, \textit{i.e.}, ConvNeXt, to design a more lightweight mobile network. Our newly introduced mobile modulation attention removes memory-intensive operations in MHA and employs an efficient modulation mechanism to boost dynamic global representational capacity. We conduct comprehensive experiments demonstrating that iFormer outperforms existing lightweight networks across various tasks. Notably, iFormer achieves an impressive Top-1 accuracy of 80.4\% on ImageNet-1k with a latency of only 1.10 ms on an iPhone 13, surpassing the recently proposed MobileNetV4 under similar latency constraints. Additionally, our method shows significant improvements in downstream tasks, including COCO object detection, instance segmentation, and ADE20k semantic segmentation, while still maintaining low latency on mobile devices for high-resolution inputs in these scenarios. 

**Abstract (ZH)**: 我们提出了一种新的移动混合视觉网络家族，称为iFormer，着重于在移动应用中优化延迟和准确性。iFormer有效地结合了卷积的快速局部表示能力和自注意力的高效全局建模能力。局部交互是通过将标准卷积神经网络（例如ConvNeXt）转换设计为更轻量级的移动网络来实现的。我们新引入的移动调制注意力去除了MHA中的内存密集型操作，并采用高效的调制机制来增强动态全局表示能力。通过全面的实验，我们证明iFormer在各种任务中优于现有的轻量级网络。值得注意的是，iFormer在iPhone 13上实现了仅1.10 ms的延迟下，ImageNet-1k数据集上的Top-1准确率达到80.4%，在相似延迟约束下超越了最近提出的MobileNetV4。此外，该方法在COCO物体检测、实例分割以及ADE20k语义分割等下游任务上表现出显著改进，同时在这些场景中仍能维持低延迟，适用于高分辨率输入的移动设备。 

---
# Large Language Models as Theory of Mind Aware Generative Agents with Counterfactual Reflection 

**Title (ZH)**: 大型语言模型作为具有反事实反思能力的理论理解生成代理 

**Authors**: Bo Yang, Jiaxian Guo, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2501.15355)  

**Abstract**: Recent studies have increasingly demonstrated that large language models (LLMs) possess significant theory of mind (ToM) capabilities, showing the potential for simulating the tracking of mental states in generative agents. In this study, we propose a novel paradigm called ToM-agent, designed to empower LLMs-based generative agents to simulate ToM in open-domain conversational interactions. ToM-agent disentangles the confidence from mental states, facilitating the emulation of an agent's perception of its counterpart's mental states, such as beliefs, desires, and intentions (BDIs). Using past conversation history and verbal reflections, ToM-Agent can dynamically adjust counterparts' inferred BDIs, along with related confidence levels. We further put forth a counterfactual intervention method that reflects on the gap between the predicted responses of counterparts and their real utterances, thereby enhancing the efficiency of reflection. Leveraging empathetic and persuasion dialogue datasets, we assess the advantages of implementing the ToM-agent with downstream tasks, as well as its performance in both the first-order and the \textit{second-order} ToM. Our findings indicate that the ToM-agent can grasp the underlying reasons for their counterpart's behaviors beyond mere semantic-emotional supporting or decision-making based on common sense, providing new insights for studying large-scale LLMs-based simulation of human social behaviors. 

**Abstract (ZH)**: 近年来，大量研究表明大型语言模型（LLMs）具备显著的理论共情（ToM）能力，显示出模拟生成型代理追踪心理状态的潜在能力。本研究提出了一种名为ToM-agent的新范式，旨在赋予基于LLMs的生成型代理在开放式领域对话交互中模拟ToM的能力。ToM-agent将信念、欲望和意图（BDIs）的心理状态与置信度解耦，从而简化代理对其对方心理状态（如信念、欲望和意图）的感知。利用过去的对话历史和言语反思，ToM-agent可以动态调整对方的推断BDIs及其相关置信度水平。此外，我们提出了一种反事实干预方法，旨在反映预测对方响应与实际言语之间的差距，从而提高反思的效率。利用移情对话和说服对话数据集，我们评估了在下游任务中实施ToM-agent的优势及其实现的一阶和二阶ToM的表现。研究结果表明，ToM-agent不仅能够超越简单的语义情感支持或基于常识的决策来捕捉对方行为的根本原因，还提供了大规模LLMs驱动的人类社会行为模拟的新研究视角。 

---
# Development and Application of Self-Supervised Machine Learning for Smoke Plume and Active Fire Identification from the FIREX-AQ Datasets 

**Title (ZH)**: 基于FIREX-AQ数据集的自监督机器学习在烟羽和活跃火灾识别中的发展与应用 

**Authors**: Nicholas LaHaye, Anistasija Easley, Kyongsik Yun, Huikyo Lee, Erik Linstead, Michael J. Garay, Olga V. Kalashnikova  

**Link**: [PDF](https://arxiv.org/pdf/2501.15343)  

**Abstract**: Fire Influence on Regional to Global Environments and Air Quality (FIREX-AQ) was a field campaign aimed at better understanding the impact of wildfires and agricultural fires on air quality and climate. The FIREX-AQ campaign took place in August 2019 and involved two aircraft and multiple coordinated satellite observations. This study applied and evaluated a self-supervised machine learning (ML) method for the active fire and smoke plume identification and tracking in the satellite and sub-orbital remote sensing datasets collected during the campaign. Our unique methodology combines remote sensing observations with different spatial and spectral resolutions. The demonstrated approach successfully differentiates fire pixels and smoke plumes from background imagery, enabling the generation of a per-instrument smoke and fire mask product, as well as smoke and fire masks created from the fusion of selected data from independent instruments. This ML approach has a potential to enhance operational wildfire monitoring systems and improve decision-making in air quality management through fast smoke plume identification12 and tracking and could improve climate impact studies through fusion data from independent instruments. 

**Abstract (ZH)**: FIREX-AQ（Fire Influence on Regional to Global Environments and Air Quality）是一项现场campaign，旨在更好地理解野火和农业火灾对空气质量及气候变化的影响。FIREX-AQ活动于2019年8月进行，涉及两架飞机和多个协调卫星观测。本研究应用并评估了一种自监督机器学习（ML）方法，用于识别和追踪在campaign期间收集的卫星和次轨道遥感数据集中的火灾和烟羽。我们的独特方法将具有不同空间和光谱分辨率的遥感观测相结合。所展示的方法成功地区分了火灾像素和烟羽与背景图像，从而生成每个传感器的烟羽和火灾掩模产品，以及由独立传感器选定数据融合生成的烟羽和火灾掩模产品。通过快速烟羽识别和追踪，这种ML方法有潜力增强实际的野火监测系统，并通过改进空气质量管理中的决策支持来提升烟羽的识别与追踪能力。此外，从独立传感器数据融合获得的信息还有助于改善气候变化影响的研究。 

---
# Scaling laws for decoding images from brain activity 

**Title (ZH)**: 从大脑活动解码图像的标度规律 

**Authors**: Hubert Banville, Yohann Benchetrit, Stéphane d'Ascoli, Jérémy Rapin amd Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2501.15322)  

**Abstract**: Generative AI has recently propelled the decoding of images from brain activity. How do these approaches scale with the amount and type of neural recordings? Here, we systematically compare image decoding from four types of non-invasive devices: electroencephalography (EEG), magnetoencephalography (MEG), high-field functional Magnetic Resonance Imaging (3T fMRI) and ultra-high field (7T) fMRI. For this, we evaluate decoding models on the largest benchmark to date, encompassing 8 public datasets, 84 volunteers, 498 hours of brain recording and 2.3 million brain responses to natural images. Unlike previous work, we focus on single-trial decoding performance to simulate real-time settings. This systematic comparison reveals three main findings. First, the most precise neuroimaging devices tend to yield the best decoding performances, when the size of the training sets are similar. However, the gain enabled by deep learning - in comparison to linear models - is obtained with the noisiest devices. Second, we do not observe any plateau of decoding performance as the amount of training data increases. Rather, decoding performance scales log-linearly with the amount of brain recording. Third, this scaling law primarily depends on the amount of data per subject. However, little decoding gain is observed by increasing the number of subjects. Overall, these findings delineate the path most suitable to scale the decoding of images from non-invasive brain recordings. 

**Abstract (ZH)**: 生成式AI最近推动了从脑活动解码图像的技术进步。这些方法在多大程度上依赖于神经记录的数量和类型？在这里，我们系统地比较了四种非侵入性设备解码图像的表现：脑电图（EEG）、磁源成像（MEG）、高场功能磁共振成像（3T fMRI）和超高场（7T）fMRI。为此，我们在迄今为止最大的基准数据集上评估了解码模型，该基准数据集涵盖了8个公开数据集、84位志愿者、498小时的脑部记录和230万次对自然图像的脑部反应。与以往研究不同，我们专注于单次试验的解码性能来模拟实时环境。这一系统的比较揭示了三个主要发现。首先，当训练集大小相似时，最精确的神经成像设备往往会获得最佳的解码性能。然而，与线性模型相比，深度学习带来的提升是在噪音较大的设备上获得的。第二，随着训练数据量的增加，我们没有观察到解码性能的饱和现象。相反，解码性能与脑部记录量之间呈对数线性关系增加。第三，这种标度律主要依赖于每位受试者的数据量。然而，增加受试者数量并不能观察到显著的解码性能提升。总体而言，这些发现为扩大量外脑成像解码的路径指明了方向。 

---
# A Post-Processing-Based Fair Federated Learning Framework 

**Title (ZH)**: 基于后处理的公平联邦学习框架 

**Authors**: Yi Zhou, Naman Goel  

**Link**: [PDF](https://arxiv.org/pdf/2501.15318)  

**Abstract**: Federated Learning (FL) allows collaborative model training among distributed parties without pooling local datasets at a central server. However, the distributed nature of FL poses challenges in training fair federated learning models. The existing techniques are often limited in offering fairness flexibility to clients and performance. We formally define and empirically analyze a simple and intuitive post-processing-based framework to improve group fairness in FL systems. This framework can be divided into two stages: a standard FL training stage followed by a completely decentralized local debiasing stage. In the first stage, a global model is trained without fairness constraints using a standard federated learning algorithm (e.g. FedAvg). In the second stage, each client applies fairness post-processing on the global model using their respective local dataset. This allows for customized fairness improvements based on clients' desired and context-guided fairness requirements. We demonstrate two well-established post-processing techniques in this framework: model output post-processing and final layer fine-tuning. We evaluate the framework against three common baselines on four different datasets, including tabular, signal, and image data, each with varying levels of data heterogeneity across clients. Our work shows that this framework not only simplifies fairness implementation in FL but also provides significant fairness improvements with minimal accuracy loss or even accuracy gain, across data modalities and machine learning methods, being especially effective in more heterogeneous settings. 

**Abstract (ZH)**: 联邦学习（FL）允许分布式各方在无需将本地数据集集中到中央服务器的情况下协作训练模型。然而，FL的分布式特性给训练公平的联邦学习模型带来了挑战。现有的技术往往在为客户提供公平灵活性和性能方面能力有限。我们正式定义并实证分析了一个简单直观的后处理框架，以提高联邦学习系统中的分组公平性。该框架可以分为两个阶段：标准的FL训练阶段，随后是完全去中心化的本地去偏见阶段。在第一阶段，使用标准的联邦学习算法（例如FedAvg）在没有公平性约束的情况下训练全局模型。在第二阶段，每个客户端使用各自的本地数据集对全局模型进行公平性后处理。这使得可以根据客户端的具体公平性需求和上下文要求进行个性化的公平性改进。我们在这框架中展示了两种广为认可的后处理技术：模型输出后处理和最终层微调。我们在四个不同数据集上（包括表格数据、信号数据和图像数据）与三种常规基线进行了评估，每个数据集中的客户端数据异质性存在不同程度的差异。我们的研究表明，该框架不仅简化了FL中的公平性实现，还能够在数据模态和机器学习方法各异的情况下提供显著的公平性改进，甚至不会损失或仅会带来微小的准确度损失，特别是在客户端数据更加异质的环境中尤为有效。 

---
# Enhancing Disaster Resilience with UAV-Assisted Edge Computing: A Reinforcement Learning Approach to Managing Heterogeneous Edge Devices 

**Title (ZH)**: 利用无人机辅助边缘计算提升灾害韧性：一种基于强化学习管理异构边缘设备的方法 

**Authors**: Talha Azfar, Kaicong Huang, Ruimin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2501.15305)  

**Abstract**: Edge sensing and computing is rapidly becoming part of intelligent infrastructure architecture leading to operational reliance on such systems in disaster or emergency situations. In such scenarios there is a high chance of power supply failure due to power grid issues, and communication system issues due to base stations losing power or being damaged by the elements, e.g., flooding, wildfires etc. Mobile edge computing in the form of unmanned aerial vehicles (UAVs) has been proposed to provide computation offloading from these devices to conserve their battery, while the use of UAVs as relay network nodes has also been investigated previously. This paper considers the use of UAVs with further constraints on power and connectivity to prolong the life of the network while also ensuring that the data is received from the edge nodes in a timely manner. Reinforcement learning is used to investigate numerous scenarios of various levels of power and communication failure. This approach is able to identify the device most likely to fail in a given scenario, thus providing priority guidance for maintenance personnel. The evacuations of a rural town and urban downtown area are also simulated to demonstrate the effectiveness of the approach at extending the life of the most critical edge devices. 

**Abstract (ZH)**: 边缘感知与计算正迅速成为智能基础设施架构的一部分，使其在灾难或紧急情况下依赖这些系统进行操作。在这种情况下，由于电网问题导致的电力供应故障和由于基站停电或被自然灾害（如洪水、野火等）破坏导致的通信系统故障的可能性很高。为节省设备电量，提议使用无人飞行器（UAVs）形式的移动边缘计算来提供计算卸载，而将UAVs用作中继网络节点的研究也已有所探讨。本论文在进一步限制电力和连接性的情况下考虑了UAVs的应用，旨在延长网络寿命的同时确保及时接收边缘节点的数据。本文采用强化学习来研究不同级别电力和通信故障的各种场景。这种方法能够识别在给定场景中最有可能失败的设备，从而为维护人员提供优先级指导。此外，还模拟了农村小镇和城市市中心的撤离场景，以展示该方法在延长最关键边缘设备寿命方面的有效性。 

---
# Music Generation using Human-In-The-Loop Reinforcement Learning 

**Title (ZH)**: 使用人类在环增强学习的音乐生成 

**Authors**: Aju Ani Justus  

**Link**: [PDF](https://arxiv.org/pdf/2501.15304)  

**Abstract**: This paper presents an approach that combines Human-In-The-Loop Reinforcement Learning (HITL RL) with principles derived from music theory to facilitate real-time generation of musical compositions. HITL RL, previously employed in diverse applications such as modelling humanoid robot mechanics and enhancing language models, harnesses human feedback to refine the training process. In this study, we develop a HILT RL framework that can leverage the constraints and principles in music theory. In particular, we propose an episodic tabular Q-learning algorithm with an epsilon-greedy exploration policy. The system generates musical tracks (compositions), continuously enhancing its quality through iterative human-in-the-loop feedback. The reward function for this process is the subjective musical taste of the user. 

**Abstract (ZH)**: 本文提出了一种结合人工参与循环强化学习（Human-In-The-Loop Reinforcement Learning, HITL RL）与音乐理论原则的方法，以实现音乐作品的实时生成。HITL RL 在诸如人形机器人机械建模和增强语言模型等多样化的应用中得到了应用，通过利用人类反馈来优化训练过程。在本研究中，我们构建了一个可以利用音乐理论中的约束和原则的HITL RL框架。特别地，我们提出了一种基于epsilon-贪婪探索策略的分阶段表格Q学习算法。该系统通过迭代的人工参与循环反馈不断生成音乐轨道（作品），并不断提高其质量。这一过程中的奖励函数是用户的主观音乐偏好。 

---
# Advanced Real-Time Fraud Detection Using RAG-Based LLMs 

**Title (ZH)**: 基于RAG的LLM的高级实时欺诈检测 

**Authors**: Gurjot Singh, Prabhjot Singh, Maninder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2501.15290)  

**Abstract**: Artificial Intelligence has become a double edged sword in modern society being both a boon and a bane. While it empowers individuals it also enables malicious actors to perpetrate scams such as fraudulent phone calls and user impersonations. This growing threat necessitates a robust system to protect individuals In this paper we introduce a novel real time fraud detection mechanism using Retrieval Augmented Generation technology to address this challenge on two fronts. First our system incorporates a continuously updating policy checking feature that transcribes phone calls in real time and uses RAG based models to verify that the caller is not soliciting private information thus ensuring transparency and the authenticity of the conversation. Second we implement a real time user impersonation check with a two step verification process to confirm the callers identity ensuring accountability. A key innovation of our system is the ability to update policies without retraining the entire model enhancing its adaptability. We validated our RAG based approach using synthetic call recordings achieving an accuracy of 97.98 percent and an F1score of 97.44 percent with 100 calls outperforming state of the art methods. This robust and flexible fraud detection system is well suited for real world deployment. 

**Abstract (ZH)**: 人工智能已成为现代社会一把双刃剑，既是福又是祸。它不仅赋能个体，同时也让恶意行为者能进行诸如欺诈电话和冒充用户等行为。这一不断增长的威胁需要一个强大的系统来保护个体。在本文中，我们引入了一种基于检索增强生成技术的新型实时欺诈检测机制，从两个方面应对这一挑战。首先，我们的系统包含一个持续更新的策略检查功能，能够实时转录电话内容，并利用基于检索增强生成（RAG）的模型验证呼叫者是否在索取私人信息，从而确保对话的透明性和真实身份。其次，我们实施了一种实时用户冒充检查机制，通过两步验证过程确认呼叫者的身份，确保其可追溯性。系统的一个关键创新是能够无需重新训练整个模型即可更新策略，从而增强其适应性。我们使用合成电话录音验证了基于RAG的方法，在100次电话中实现了97.98％的准确率和97.44％的F1分数，超越了现有最先进的方法。这一强大且灵活的欺诈检测系统非常适合实际部署。 

---
# Pre-training a Transformer-Based Generative Model Using a Small Sepedi Dataset 

**Title (ZH)**: 使用小规模Sepedi数据集预训练基于Transformer的生成模型 

**Authors**: Simon P. Ramalepe, Thipe I. Modipa, Marelie H. Davel  

**Link**: [PDF](https://arxiv.org/pdf/2501.15281)  

**Abstract**: Due to the scarcity of data in low-resourced languages, the development of language models for these languages has been very slow. Currently, pre-trained language models have gained popularity in natural language processing, especially, in developing domain-specific models for low-resourced languages. In this study, we experiment with the impact of using occlusion-based techniques when training a language model for a text generation task. We curate 2 new datasets, the Sepedi monolingual (SepMono) dataset from several South African resources and the Sepedi radio news (SepNews) dataset from the radio news domain. We use the SepMono dataset to pre-train transformer-based models using the occlusion and non-occlusion pre-training techniques and compare performance. The SepNews dataset is specifically used for fine-tuning. Our results show that the non-occlusion models perform better compared to the occlusion-based models when measuring validation loss and perplexity. However, analysis of the generated text using the BLEU score metric, which measures the quality of the generated text, shows a slightly higher BLEU score for the occlusion-based models compared to the non-occlusion models. 

**Abstract (ZH)**: 由于低资源语言的数据稀缺，这些语言的语言模型开发进展非常缓慢。目前，预训练语言模型在自然语言处理领域中变得非常流行，特别适用于为低资源语言开发特定领域的模型。在本研究中，我们探究了在训练文本生成任务的语言模型时使用遮挡技术的影响。我们收集了两个新的数据集：一个是来自南非多种资源的Sepedi单语（SepMono）数据集，另一个是来自广播新闻领域的Sepedi广播新闻（SepNews）数据集。我们使用SepMono数据集分别通过遮挡和非遮挡预训练技术预训练基于Transformer的模型，并比较其性能。SepNews数据集专门用于微调。结果显示，当衡量验证损失和困惑度时，非遮挡模型的性能优于遮挡模型。然而，使用BLEU评分指标来评估生成文本的质量时，遮挡模型的BLEU分数略高于非遮挡模型。 

---
# Exploring the Collaborative Co-Creation Process with AI: A Case Study in Novice Music Production 

**Title (ZH)**: 探索基于AI的协作共创过程：新手音乐制作案例研究 

**Authors**: Yue Fu, Michele Newman, Lewis Going, Qiuzi Feng, Jin Ha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.15276)  

**Abstract**: Artificial intelligence is reshaping creative domains, yet its co-creative processes, especially in group settings with novice users, remain under explored. To bridge this gap, we conducted a case study in a college-level course where nine undergraduate students were tasked with creating three original music tracks using AI tools over 10 weeks. The study spanned the entire creative journey from ideation to releasing these songs on Spotify. Participants leveraged AI for music and lyric production, cover art, and distribution. Our findings highlight how AI transforms creative workflows: accelerating ideation but compressing the traditional preparation stage, and requiring novices to navigate a challenging idea selection and validation phase. We also identified a new "collaging and refinement" stage, where participants creatively combined diverse AI-generated outputs into cohesive works. Furthermore, AI influenced group social dynamics and role division among human creators. Based on these insights, we propose the Human-AI Co-Creation Stage Model and the Human-AI Agency Model, offering new perspectives on collaborative co-creation with AI. 

**Abstract (ZH)**: 人工智能正在重塑创意领域，但在群体环境中，特别是在新手用户参与的情况下，其共创过程仍然鲜有探索。为弥合这一差距，我们在一门大学课程中开展了一项案例研究，九名本科生在10周内使用人工智能工具创作了三首原创音乐作品。研究涵盖了从创意构思到在Spotify发布这些歌曲的整个创作过程。参与者利用人工智能进行音乐和歌词创作、封面艺术设计和分发。我们的研究结果表明，人工智能如何改变创意工作流程：加快了创意构思过程，但压缩了传统的准备阶段，并要求新手在具有挑战性的想法筛选和验证阶段进行导航。我们还发现了一个新的“拼接和润色”阶段，在这个阶段，参与者创造性地将多种人工智能生成的输出整合成一致的作品。此外，人工智能影响了团队的社会动态以及人类创作者之间的角色分配。基于这些见解，我们提出了人类-人工智能共创阶段模型和人类-人工智能代理模型，为与人工智能协作共创提供了新的视角。 

---
# Lightweight and Post-Training Structured Pruning for On-Device Large Lanaguage Models 

**Title (ZH)**: 针对设备端大型语言模型的轻量级和训练后结构剪枝 

**Authors**: Zihuai Xu, Yang Xu, Hongli Xu, Yunming Liao, Zhiwei Yao, Zuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2501.15255)  

**Abstract**: Considering the hardware-friendly characteristics and broad applicability, structured pruning has emerged as an efficient solution to reduce the resource demands of large language models (LLMs) on resource-constrained devices. Traditional structured pruning methods often need fine-tuning to recover performance loss, which incurs high memory overhead and substantial data requirements, rendering them unsuitable for on-device applications. Additionally, post-training structured pruning techniques typically necessitate specific activation functions or architectural modifications, thereby limiting their scope of applications. Herein, we introduce COMP, a lightweight post-training structured pruning method that employs a hybrid-granularity pruning strategy. COMP initially prunes selected model layers based on their importance at a coarse granularity, followed by fine-grained neuron pruning within the dense layers of each remaining model layer. To more accurately evaluate neuron importance, COMP introduces a new matrix condition-based metric. Subsequently, COMP utilizes mask tuning to recover accuracy without the need for fine-tuning, significantly reducing memory consumption. Experimental results demonstrate that COMP improves performance by 6.13\% on the LLaMA-2-7B model with a 20\% pruning ratio compared to LLM-Pruner, while simultaneously reducing memory overhead by 80\%. 

**Abstract (ZH)**: 考虑到结构化剪枝的硬件友好特性及广泛的应用价值，结构化剪枝已成为减轻大型语言模型（LLMs）在资源受限设备上资源需求的有效解决方案。传统的结构化剪枝方法通常需要微调以恢复性能损失，这带来了高内存消耗和大量的数据需求，使其不适合于本地设备应用。此外，后续训练的结构化剪枝技术通常需要特定的激活函数或架构修改，从而限制了其应用范围。在此基础上，我们介绍了一种名为COMP的轻量级后续训练结构化剪枝方法，该方法采用混合粒度的剪枝策略。COMP首先基于粗粒度的重要性对选定的模型层进行剪枝，随后在每个剩余模型层的稠密层中进行精细粒度的神经元剪枝。为了更准确地评估神经元的重要性，COMP引入了一种基于矩阵条件的新度量标准。接着，COMP利用掩码调整来恢复准确性，而无需微调，从而显著减少了内存消耗。实验结果表明，与LLM-Pruner相比，在对LLaMA-2-7B模型进行20%剪枝的情况下，COMP在性能上提高了6.13%，同时内存开销减少了80%。 

---
# Prompting ChatGPT for Chinese Learning as L2: A CEFR and EBCL Level Study 

**Title (ZH)**: 将下面的论文标题翻译成中文，符合学术规范：

Prompting ChatGPT for Chinese Learning as L2: A CEFR and EBCL Level Study

翻译为：

使用 ChatGPT 进行二外汉语学习的促进研究：基于CEFR和EBCL的水平分析 

**Authors**: Miao Lin-Zucker, Joël Bellasen, Jean-Daniel Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2501.15247)  

**Abstract**: The use of chatbots in language learning has evolved significantly since the 1960s, becoming more sophisticated platforms as generative AI emerged. These tools now simulate natural conversations, adapting to individual learners' needs, including those studying Chinese. Our study explores how learners can use specific prompts to engage Large Language Models (LLM) as personalized chatbots, aiming to target their language level based on the Common European Framework of Reference for Languages (CEFR) and the European Benchmarking Chinese Language (EBCL) project. Focusing on A1, A1+ and A2 levels, we examine the teaching of Chinese, which presents unique challenges due to its logographic writing system. Our goal is to develop prompts that integrate oral and written skills, using high-frequency character lists and controlling oral lexical productions. These tools, powered by generative AI, aim to enhance language practice by crossing lexical and sinographic recurrence. While generative AI shows potential as a personalized tutor, further evaluation is needed to assess its effectiveness. We conducted a systematic series of experiments using ChatGPT models to evaluate their adherence to constraints specified in the prompts. The results indicate that incorporating level A1 and A1+ characters, along with the associated reference list, significantly enhances compliance with the EBCL character set. Properly prompted, LLMs can increase exposure to the target language and offer interactive exchanges to develop language skills. 

**Abstract (ZH)**: 自20世纪60年代以来，聊天机器人在语言学习中的应用已取得了显著进展，随着生成式人工智能的出现，这些工具变得日益复杂。它们现在能够模拟自然对话，适应个体学习者的需求，包括学习汉语的人。本研究探讨了学习者如何使用特定提示与大型语言模型（LLM）进行个性化聊天，旨在根据共同欧洲语言参考框架（CEFR）和欧洲汉语基准项目（EBCL）调整其语言水平。本研究重点关注A1、A1+和A2水平，汉语的教学因其表意文字系统而具有独特的挑战。我们的目标是开发出综合听说能力的提示，利用高频汉字列表，并控制口语词汇的生产。这些由生成式人工智能驱动的工具旨在通过词汇和汉字的反复出现来增强语言实践。虽然生成式人工智能作为个人辅导者具有潜力，但还需要进一步评估其有效性。我们使用ChatGPT模型进行了一系列系统的实验，以评估其对提示中指定约束的遵守情况。结果显示，结合A1和A1+级别的字符及相关的参考列表，显著提升了对EBCL字符集的遵守度。在适当提示下，LLM可以增加接触目标语言的机会，并提供互动交流以发展语言技能。 

---
# Hardware-Aware DNN Compression for Homogeneous Edge Devices 

**Title (ZH)**: 面向硬件的深度神经网络压缩技术研究——适用于homogeneous边缘设备 

**Authors**: Kunlong Zhang, Guiying Li, Ning Lu, Peng Yang, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15240)  

**Abstract**: Deploying deep neural networks (DNNs) across homogeneous edge devices (the devices with the same SKU labeled by the manufacturer) often assumes identical performance among them. However, once a device model is widely deployed, the performance of each device becomes different after a period of running. This is caused by the differences in user configurations, environmental conditions, manufacturing variances, battery degradation, etc. Existing DNN compression methods have not taken this scenario into consideration and can not guarantee good compression results in all homogeneous edge devices. To address this, we propose Homogeneous-Device Aware Pruning (HDAP), a hardware-aware DNN compression framework explicitly designed for homogeneous edge devices, aiming to achieve optimal average performance of the compressed model across all devices. To deal with the difficulty of time-consuming hardware-aware evaluations for thousands or millions of homogeneous edge devices, HDAP partitions all the devices into several device clusters, which can dramatically reduce the number of devices to evaluate and use the surrogate-based evaluation instead of hardware evaluation in real-time. Experiments on ResNet50 and MobileNetV1 with the ImageNet dataset show that HDAP consistently achieves lower average inference latency compared with state-of-the-art methods, with substantial speedup gains (e.g., 2.86 $\times$ speedup at 1.0G FLOPs for ResNet50) on the homogeneous device clusters. HDAP offers an effective solution for scalable, high-performance DNN deployment methods for homogeneous edge devices. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

在将深度神经网络（DNNs）部署到由制造商同一SKU标签标明的具有相同硬件配置的边缘设备上时，通常假定这些设备具有相同的性能。然而，一旦设备模型大规模部署，每个设备在运行一段时间后，其性能会有所不同。这主要是由于用户配置、环境条件、生产差异、电池老化等因素引起的。现有的DNN压缩方法并未考虑到这种情况，无法保证在所有同质边缘设备上获得良好的压缩效果。为解决这一问题，我们提出了一种应用于同质边缘设备的硬件感知压缩框架——同质设备感知剪枝（HDAP）。该框架旨在实现压缩模型在所有设备上的最优平均性能。为应对评估数以千计或数百万同质边缘设备所耗费的大量时间这一难题，HDAP 将所有设备划分为若干设备簇，这可以大幅减少需要评估的设备数量，并在实时评估中使用替代的代理评估方法而不是硬件评估。通过使用ImageNet数据集上的ResNet50和MobileNetV1进行实验，结果显示，HDAP 在同质设备簇中平均推理延迟始终低于最新方法，并在ResNet50的1.0G FLOPs性能下实现了显著的加速（例如，2.86 倍加速）。HDAP 提供了一种可扩展且高性能的解决方案，用于同质边缘设备上的DNN部署。 

---
# SEAL: Scaling to Emphasize Attention for Long-Context Retrieval 

**Title (ZH)**: SEAL：扩展以强调注意力机制进行长上下文检索 

**Authors**: Changhun Lee, Jun-gyu Jin, Younghyun Cho, Eunhyeok Park  

**Link**: [PDF](https://arxiv.org/pdf/2501.15225)  

**Abstract**: In this work, we introduce a novel approach called Scaling to Emphasize Attention for Long-context retrieval (SEAL), which enhances the retrieval performance of large language models (LLMs) over extended contexts. Previous studies have shown that each attention head in LLMs has a unique functionality and collectively contributes to the overall behavior of the model. Similarly, we observe that specific heads are closely tied to long-context retrieval, showing positive or negative correlation with retrieval scores. Built on this insight, we propose a learning-based mechanism using zero-shot generated data to emphasize these heads, improving the model's performance in long-context retrieval tasks. By applying SEAL, we can achieve significant improvements in in-domain retrieval performance, including document QA tasks from LongBench, and considerable improvements in out-of-domain cases. Additionally, when combined with existing training-free context extension techniques, SEAL extends the context limits of LLMs while maintaining highly reliable outputs, opening new avenues for research in this field. 

**Abstract (ZH)**: 在本工作中，我们提出了一种新的方法，称为Scaling to Emphasize Attention for Long-context retrieval (SEAL)，该方法旨在增强大型语言模型（LLMs）在长上下文检索中的检索性能。先前的研究表明，每个多头注意机制在大型语言模型中具有独特的功能，并且共同作用以影响模型的整体行为。类似地，我们观察到特定的注意力头与长上下文检索紧密相关，这些头与检索评分呈正相关或负相关。基于这一洞察，我们提出了一种基于零样本生成数据的学习机制，以突出这些关键头，从而改善模型在长上下文检索任务中的性能。通过应用SEAL方法，我们可以在包括LongBench的文档问答任务在内的同域检索任务中实现显著性能提升，并在跨域情况下实现明显改进。此外，当与现有的无需训练的上下文扩展技术结合使用时，SEAL能够在保持高可靠性输出的同时扩展大型语言模型的上下文限制，为该领域的研究开辟新的途径。 

---
# Efficient and Interpretable Neural Networks Using Complex Lehmer Transform 

**Title (ZH)**: 使用复Lehmer变换的高效可解释神经网络 

**Authors**: Masoud Ataei, Xiaogang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15223)  

**Abstract**: We propose an efficient and interpretable neural network with a novel activation function called the weighted Lehmer transform. This new activation function enables adaptive feature selection and extends to the complex domain, capturing phase-sensitive and hierarchical relationships within data. Notably, it provides greater interpretability and transparency compared to existing machine learning models, facilitating a deeper understanding of its functionality and decision-making processes. We analyze the mathematical properties of both real-valued and complex-valued Lehmer activation units and demonstrate their applications in modeling nonlinear interactions. Empirical evaluations demonstrate that our proposed neural network achieves competitive accuracy on benchmark datasets with significantly improved computational efficiency. A single layer of real-valued or complex-valued Lehmer activation units is shown to deliver state-of-the-art performance, balancing efficiency with interpretability. 

**Abstract (ZH)**: 我们提出了一种高效且可解释的神经网络，并引入了一种新的激活函数，称为加权勒乌尔变换（Weighted Lehmer Transform）。这种新激活函数能够实现自适应特征选择，并扩展到复数域，能够捕捉数据中的相位敏感和分层关系。值得注意的是，与现有的机器学习模型相比，它提供了更高的可解释性和透明度，有助于更深入地理解其功能和决策过程。我们分析了实值和复值勒乌尔激活单元的数学性质，并展示了它们在建模非线性交互方面的应用。实验证明，我们提出的神经网络在基准数据集上实现了竞争性的准确率，并具有显著的计算效率提升。单层实值或复值勒乌尔激活单元展示了最先进的性能，平衡了效率和可解释性。 

---
# Towards Conscious Service Robots 

**Title (ZH)**: 朝向有意识的服务机器人 

**Authors**: Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2501.15198)  

**Abstract**: Deep learning's success in perception, natural language processing, etc. inspires hopes for advancements in autonomous robotics. However, real-world robotics face challenges like variability, high-dimensional state spaces, non-linear dependencies, and partial observability. A key issue is non-stationarity of robots, environments, and tasks, leading to performance drops with out-of-distribution data. Unlike current machine learning models, humans adapt quickly to changes and new tasks due to a cognitive architecture that enables systematic generalization and meta-cognition. Human brain's System 1 handles routine tasks unconsciously, while System 2 manages complex tasks consciously, facilitating flexible problem-solving and self-monitoring. For robots to achieve human-like learning and reasoning, they need to integrate causal models, working memory, planning, and metacognitive processing. By incorporating human cognition insights, the next generation of service robots will handle novel situations and monitor themselves to avoid risks and mitigate errors. 

**Abstract (ZH)**: 深度学习在感知、自然语言处理等方面的成功激发了对自主机器人技术进步的希望。然而，现实中的机器人面临着变异性、高维状态空间、非线性依赖性和部分可观测性的挑战。一个关键问题是机器人、环境和任务的非稳定性，这会导致在非分布数据下的性能下降。与当前的机器学习模型不同，人类能够迅速适应变化和新任务，这得益于一种认知架构，使人类能够进行系统化泛化和元认知。人类的大脑中的系统1无意识地处理常规任务，而系统2则有意识地处理复杂任务，从而促进灵活的问题解决和自我监控。为了使机器人具备类似人类的学习和推理能力，它们需要结合因果模型、工作记忆、规划和元认知处理。通过融合人类的认知洞察，下一代服务机器人将能够应对新型情境并自我监控以避免风险、减少错误。 

---
# Option-ID Based Elimination For Multiple Choice Questions 

**Title (ZH)**: 基于Option-ID的多项选择题消除方法 

**Authors**: Zhenhao Zhu, Bulou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15175)  

**Abstract**: Multiple choice questions (MCQs) are a common and important task for evaluating large language models (LLMs). Based on common strategies humans use when answering MCQs, the process of elimination has been proposed as an effective problem-solving method. Existing methods to the process of elimination generally fall into two categories: one involves having the model directly select the incorrect answer, while the other involves scoring the options. However, both methods incur high computational costs and often perform worse than methods that answer based on option ID. To address this issue, this paper proposes a process of elimination based on option ID. We select 10 LLMs and conduct zero-shot experiments on 7 different datasets. The experimental results demonstrate that our method significantly improves the model's performance. Further analysis reveals that the sequential elimination strategy can effectively enhance the model's reasoning ability. Additionally, we find that sequential elimination is also applicable to few-shot settings and can be combined with debias methods to further improve model performance. 

**Abstract (ZH)**: 多项选择题（MCQs）是评估大规模语言模型（LLMs）的一种常见且重要的任务。基于人类回答MCQs时常用的一些策略，逐步排除法被认为是一种有效的问题解决方法。现有的逐步排除法大致可以分为两类：一类方法是让模型直接选择错误的答案，另一类是评分选项。但这两类方法通常会产生较高的计算成本，并且往往不如基于选项ID作答的方法表现得好。为了解决这一问题，本文提出了一种基于选项ID的逐步排除法。我们选择了10个LLM，并在7个不同的数据集上进行了零样本实验。实验结果表明，我们的方法显著提高了模型的性能。进一步的分析显示，顺序排除策略能够有效增强模型的推理能力。此外，我们发现顺序排除方法也适用于少量样本设置，并且可以与去偏方法结合使用以进一步提高模型性能。 

---
# Mapping Galaxy Images Across Ultraviolet, Visible and Infrared Bands Using Generative Deep Learning 

**Title (ZH)**: 使用生成式深度学习跨紫外、可见和红外波段映射星系图像 

**Authors**: Youssef Zaazou, Alex Bihlo, Terrence S. Tricco  

**Link**: [PDF](https://arxiv.org/pdf/2501.15149)  

**Abstract**: We demonstrate that generative deep learning can translate galaxy observations across ultraviolet, visible, and infrared photometric bands. Leveraging mock observations from the Illustris simulations, we develop and validate a supervised image-to-image model capable of performing both band interpolation and extrapolation. The resulting trained models exhibit high fidelity in generating outputs, as verified by both general image comparison metrics (MAE, SSIM, PSNR) and specialized astronomical metrics (GINI coefficient, M20). Moreover, we show that our model can be used to predict real-world observations, using data from the DECaLS survey as a case study. These findings highlight the potential of generative learning to augment astronomical datasets, enabling efficient exploration of multi-band information in regions where observations are incomplete. This work opens new pathways for optimizing mission planning, guiding high-resolution follow-ups, and enhancing our understanding of galaxy morphology and evolution. 

**Abstract (ZH)**: 我们展示了生成式深度学习能够将星系观测从紫外、可见光和红外光谱带之间进行转换。借助Illustris模拟的模拟观测数据，我们开发并验证了一种监督图像到图像模型，该模型不仅能进行光谱带插值，还能进行光谱带外推。经过训练的模型在生成输出方面表现出高保真度，这通过通用图像比较指标（MAE、SSIM、PSNR）和专门的天文指标（GINI系数、M20）得到了验证。此外，我们展示了该模型能够在实际观测数据（以DECaLS调查数据为例）基础上进行预测。这些发现强调了生成式学习在增强天文学数据集方面的潜力，使我们在观测不完整区域能够高效探索多波段信息。此项工作为优化航天任务规划、指导高分辨率后续观测以及增强我们对星系形态和演化的理解开辟了新的途径。 

---
# DAGPrompT: Pushing the Limits of Graph Prompting with a Distribution-aware Graph Prompt Tuning Approach 

**Title (ZH)**: DAGPrompt：通过分布感知图提示调优方法推动图提示技术的边界 

**Authors**: Qin Chen, Liang Wang, Bo Zheng, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.15142)  

**Abstract**: The pre-train then fine-tune approach has advanced GNNs by enabling general knowledge capture without task-specific labels. However, an objective gap between pre-training and downstream tasks limits its effectiveness. Recent graph prompting methods aim to close this gap through task reformulations and learnable prompts. Despite this, they struggle with complex graphs like heterophily graphs. Freezing the GNN encoder can reduce the impact of prompting, while simple prompts fail to handle diverse hop-level distributions. This paper identifies two key challenges in adapting graph prompting methods for complex graphs: (1) adapting the model to new distributions in downstream tasks to mitigate pre-training and fine-tuning discrepancies from heterophily and (2) customizing prompts for hop-specific node requirements. To overcome these challenges, we propose Distribution-aware Graph Prompt Tuning (DAGPrompT), which integrates a GLoRA module for optimizing the GNN encoder's projection matrix and message-passing schema through low-rank adaptation. DAGPrompT also incorporates hop-specific prompts accounting for varying graph structures and distributions among hops. Evaluations on 10 datasets and 14 baselines demonstrate that DAGPrompT improves accuracy by up to 4.79 in node and graph classification tasks, setting a new state-of-the-art while preserving efficiency. Codes are available at GitHub. 

**Abstract (ZH)**: 预训练然后微调的方法通过使图神经网络能够捕获通用知识而促进了其发展，无需特定任务的标签。然而，预训练与下游任务之间的客观差距限制了其效果。最近的图提示方法旨在通过任务重新表述和可学习的提示来弥合这一差距。尽管如此，它们在处理异质性图等复杂图时仍存在问题。冻结GNN编码器可以减少提示的影响，而简单的提示无法应对不同的跳跃级别分布。本文识别了在复杂图上适应图提示方法的两个关键挑战：(1) 调整模型以应对下游任务中新分布，以减轻预训练和微调与异质性相关的差异，以及 (2) 为不同跳跃级别的节点需求定制提示。为了克服这些挑战，我们提出了一种新的方法——分布感知图提示调整（DAGPrompT）——该方法结合了一个GLoRA模块，通过低秩适应优化GNN编码器的投影矩阵和消息传递方案。DAGPrompT 还包含了考虑不同跳跃级别之间图结构和分布差异的专门化提示。在10个数据集和14种基线方法上的评估结果显示，DAGPrompT 在节点和图分类任务中的准确率最多可提升4.79%，达到了新的最佳性能，同时保持了高效性。相关代码已发布在GitHub上。 

---
# Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models 

**Title (ZH)**: 分析并增强细粒度视觉识别在多模态大型语言模型中的能力 

**Authors**: Hulingxiao He, Geng Li, Zijun Geng, Jinglin Xu, Yuxin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2501.15140)  

**Abstract**: Multi-modal large language models (MLLMs) have shown remarkable abilities in various visual understanding tasks. However, MLLMs still struggle with fine-grained visual recognition (FGVR), which aims to identify subordinate-level categories from images. This can negatively impact more advanced capabilities of MLLMs, such as object-centric visual question answering and reasoning. In our study, we revisit three quintessential capabilities of MLLMs for FGVR, including object information extraction, category knowledge reserve, object-category alignment, and position of the root cause as a misalignment problem. To address this issue, we present Finedefics, an MLLM that enhances the model's FGVR capability by incorporating informative attribute descriptions of objects into the training phase. We employ contrastive learning on object-attribute pairs and attribute-category pairs simultaneously and use examples from similar but incorrect categories as hard negatives, naturally bringing representations of visual objects and category names closer. Extensive evaluations across multiple popular FGVR datasets demonstrate that Finedefics outperforms existing MLLMs of comparable parameter sizes, showcasing its remarkable efficacy. The code is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在各种视觉理解任务中展现了显著的能力。然而，MLLMs 在细粒度视觉识别（FGVR）方面仍然存在困难，FGVR 的目标是从图像中识别子类别。这可能对 MLLMs 的更高级能力，如对象为中心的视觉问题回答和推理产生负面影响。在我们的研究中，我们重新审视了 MLLMs 在 FGVR 方面的三种核心能力，包括对象信息提取、类别知识储备、对象-类别对齐，以及将这些差异视为对齐问题的根本原因所在。为了解决这一问题，我们提出了一种名为 Finedefics 的 MLLM，通过在训练过程中结合对象的详细属性描述来增强模型的 FGVR 能力。我们同时采用了对比学习，分别在对象-属性对和属性-类别对上进行，并使用来自相似但错误类别的例子作为困难负样本，自然地拉近了视觉对象和类别名的表示。在多个流行的 FGVR 数据集上的广泛评估表明，Finedefics 在与现有 MLLMs 类似的参数量下具有更好的性能，证明了其显著的效能。相关代码可在以下链接获取：[插入链接] 

---
# Snapshot Compressed Imaging Based Single-Measurement Computer Vision for Videos 

**Title (ZH)**: 基于单测量的视频压缩成像计算视觉snapshot压缩成像方法 

**Authors**: Fengpu Pan, Jiangtao Wen, Yuxing Han  

**Link**: [PDF](https://arxiv.org/pdf/2501.15122)  

**Abstract**: Snapshot compressive imaging (SCI) is a promising technique for capturing high-speed video at low bandwidth and low power, typically by compressing multiple frames into a single measurement. However, similar to traditional CMOS image sensor based imaging systems, SCI also faces challenges in low-lighting photon-limited and low-signal-to-noise-ratio image conditions. In this paper, we propose a novel Compressive Denoising Autoencoder (CompDAE) using the STFormer architecture as the backbone, to explicitly model noise characteristics and provide computer vision functionalities such as edge detection and depth estimation directly from compressed sensing measurements, while accounting for realistic low-photon conditions. We evaluate the effectiveness of CompDAE across various datasets and demonstrated significant improvements in task performance compared to conventional RGB-based methods. In the case of ultra-low-lighting (APC $\leq$ 20) while conventional methods failed, the proposed algorithm can still maintain competitive performance. 

**Abstract (ZH)**: 瞬时压缩成像（Snapshot Compressive Imaging, SCI）是一种在低带宽和低功耗下捕捉高速视频的有希望的技术，通常通过将多帧压缩为单一测量来实现。然而，与基于传统CMOS图像传感器的成像系统类似，SCI在低光照条件下的光子限制和信噪比低的情况下也面临挑战。在本文中，我们提出了一种新颖的压缩去噪自编码器（CompDAE），其使用STFormer架构作为骨干网络，以明确建模噪声特性，并直接从压缩传感测量中提供计算机视觉功能，如边缘检测和深度估计，同时考虑到现实中的低光子条件。我们通过多种数据集评估了CompDAE的有效性，并展示了其任务性能相较于传统基于RGB的方法有显著改进。在极端低光照（APC ≤ 20）的情况下，尽管传统方法失败，所提出的算法仍然保持了可竞争的性能。 

---
# Clear Preferences Leave Traces: Reference Model-Guided Sampling for Preference Learning 

**Title (ZH)**: 明确的偏好会留下痕迹：基于参考模型的偏好学习采样方法 

**Authors**: Nirav Diwan, Tolga Ergen, Dongsub Shim, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.15109)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a de-facto approach for aligning language models with human preferences. Recent work has shown DPO's effectiveness relies on training data quality. In particular, clear quality differences between preferred and rejected responses enhance learning performance. Current methods for identifying and obtaining such high-quality samples demand additional resources or external models. We discover that reference model probability space naturally detects high-quality training samples. Using this insight, we present a sampling strategy that achieves consistent improvements (+0.1 to +0.4) on MT-Bench while using less than half (30-50%) of the training data. We observe substantial improvements (+0.4 to +0.98) for technical tasks (coding, math, and reasoning) across multiple models and hyperparameter settings. 

**Abstract (ZH)**: 直接偏好优化（Direct Preference Optimization, DPO）已经成为了将语言模型与人类偏好对齐的默认方法。最近的研究表明，DPO的效果依赖于训练数据的质量。特别是，被偏好和未被接受的响应之间明显的质量差异可以增强学习性能。当前识别和获取此类高质量样本的方法需要额外的资源或外部模型。我们发现，参考模型的概率空间自然地检测高质量的训练样本。基于这一洞察，我们提出了一种采样策略，该策略在使用不到一半（30-50%）训练数据的情况下，在MT-Bench上实现了持续改进（+0.1到+0.4）。此外，在多个模型和超参数设置下，我们观察到技术任务（编程、数学和推理）上的显著改进（+0.4到+0.98）。 

---
# Each Rank Could be an Expert: Single-Ranked Mixture of Experts LoRA for Multi-Task Learning 

**Title (ZH)**: 每种层级都可以是专家：基于LoRA的单层级混合专家模型在多任务学习中的应用 

**Authors**: Ziyu Zhao, Yixiao Zhou, Didi Zhu, Tao Shen, Xuwu Wang, Jing Su, Kun Kuang, Zhongyu Wei, Fei Wu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.15103)  

**Abstract**: Low-Rank Adaptation (LoRA) is widely used for adapting large language models (LLMs) to specific domains due to its efficiency and modularity. Meanwhile, vanilla LoRA struggles with task conflicts in multi-task scenarios. Recent works adopt Mixture of Experts (MoE) by treating each LoRA module as an expert, thereby mitigating task interference through multiple specialized LoRA modules. While effective, these methods often isolate knowledge within individual tasks, failing to fully exploit the shared knowledge across related tasks. In this paper, we establish a connection between single LoRA and multi-LoRA MoE, integrating them into a unified framework. We demonstrate that the dynamic routing of multiple LoRAs is functionally equivalent to rank partitioning and block-level activation within a single LoRA. We further empirically demonstrate that finer-grained LoRA partitioning, within the same total and activated parameter constraints, leads to better performance gains across heterogeneous tasks. Building on these findings, we propose Single-ranked Mixture of Experts LoRA (\textbf{SMoRA}), which embeds MoE into LoRA by \textit{treating each rank as an independent expert}. With a \textit{dynamic rank-wise activation} mechanism, SMoRA promotes finer-grained knowledge sharing while mitigating task conflicts. Experiments demonstrate that SMoRA activates fewer parameters yet achieves better performance in multi-task scenarios. 

**Abstract (ZH)**: 低秩适应（LoRA）由于其高效性和模块性，被广泛用于将大型语言模型（LLMs）适应到特定领域。然而，经典的LoRA在多任务场景中难以处理任务冲突。近期的研究通过将每个LoRA模块视为一个专家，并采用混合专家（MoE）来减轻任务间的干扰。虽然这些方法有效，但它们往往将知识孤立在各个任务中，未能充分利用相关任务间的共享知识。本文建立了单个LoRA和多LoRA MoE之间的联系，将两者统一到一个框架中。我们证明，多个LoRA模块的动态路由功能等同于单个LoRA模块中的秩分区和块级激活。进一步的实验证明，在相同的总参数约束和激活参数约束下，更精细的LoRA分区可以带来更好的跨异质任务的性能提升。基于这些发现，我们提出了单秩混合专家LoRA（SMoRA），通过将每个秩视为独立的专家将MoE嵌入LoRA中。通过动态的秩级激活机制，SMoRA促进了更精细的知识共享并减轻了任务冲突。实验结果表明，尽管SMoRA激活的参数更少，但在多任务场景中却能获得更好的性能。 

---
# CFT-RAG: An Entity Tree Based Retrieval Augmented Generation Algorithm With Cuckoo Filter 

**Title (ZH)**: CFT-RAG：基于实体树的检索增强生成算法，采用 cuckoo 过滤器 

**Authors**: Zihang Li, Yangdong Ruan, Wenjun Liu, Zhengyang Wang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15098)  

**Abstract**: Although retrieval-augmented generation(RAG) significantly improves generation quality by retrieving external knowledge bases and integrating generated content, it faces computational efficiency bottlenecks, particularly in knowledge retrieval tasks involving hierarchical structures for Tree-RAG. This paper proposes a Tree-RAG acceleration method based on the improved Cuckoo Filter, which optimizes entity localization during the retrieval process to achieve significant performance improvements. Tree-RAG effectively organizes entities through the introduction of a hierarchical tree structure, while the Cuckoo Filter serves as an efficient data structure that supports rapid membership queries and dynamic updates. The experiment results demonstrate that our method is much faster than naive Tree-RAG while maintaining high levels of generative quality. When the number of trees is large, our method is hundreds of times faster than naive Tree-RAG. Our work is available at this https URL. 

**Abstract (ZH)**: 尽管检索增强生成（RAG）通过检索外部知识库并整合生成内容显著提高了生成质量，但在涉及层次结构的知识检索任务（如Tree-RAG）中，它面临计算效率瓶颈。本文提出了一种基于改进的Cuckoo Filter的Tree-RAG加速方法，该方法在检索过程中优化实体定位，以实现显著的性能提升。通过引入层次树结构，Tree-RAG有效组织了实体，而Cuckoo Filter作为一种高效的数据结构，支持快速的成员查询和动态更新。实验结果表明，与朴素的Tree-RAG相比，我们的方法在保持高度生成质量的同时具有更快的速度。当树的数量较大时，我们的方法比朴素的Tree-RAG快数百倍。我们的工作可以在以下链接访问：[此链接]。 

---
# Hierarchical Pattern Decryption Methodology for Ransomware Detection Using Probabilistic Cryptographic Footprints 

**Title (ZH)**: 使用概率加密特征进行 ransomware 检测的分层模式解密方法学 

**Authors**: Kevin Pekepok, Persephone Kirkwood, Esme Christopolous, Florence Braithwaite, Oliver Nightingale  

**Link**: [PDF](https://arxiv.org/pdf/2501.15084)  

**Abstract**: The increasing sophistication of encryption-based ransomware has demanded innovative approaches to detection and mitigation, prompting the development of a hierarchical framework grounded in probabilistic cryptographic analysis. By focusing on the statistical characteristics of encryption patterns, the proposed methodology introduces a layered approach that combines advanced clustering algorithms with machine learning to isolate ransomware-induced anomalies. Through comprehensive testing across diverse ransomware families, the framework demonstrated exceptional accuracy, effectively distinguishing malicious encryption operations from benign activities while maintaining low false positive rates. The system's design integrates dynamic feedback mechanisms, enabling adaptability to varying cryptographic complexities and operational environments. Detailed entropy-based evaluations revealed its sensitivity to subtle deviations in encryption workflows, offering a robust alternative to traditional detection methods reliant on static signatures or heuristics. Computational benchmarks confirmed its scalability and efficiency, achieving consistent performance even under high data loads and complex cryptographic scenarios. The inclusion of real-time clustering and anomaly evaluation ensures rapid response capabilities, addressing critical latency challenges in ransomware detection. Performance comparisons with established methods highlighted its improvements in detection efficacy, particularly against advanced ransomware employing extended key lengths and unique cryptographic protocols. 

**Abstract (ZH)**: 基于加密技术的勒索软件日益复杂，要求提出创新的检测和缓解方法，从而促进了基于概率密码分析的分层次框架的发展。通过聚焦于加密模式的统计特征，提出的的方法引入了一种分层次的方法，结合了高级聚类算法和机器学习，以隔离由勒索软件引起的异常。通过在多种勒索软件家族中进行全面测试，该框架展示了卓越的准确率，能够有效区分恶意加密操作与正常活动，并保持低误报率。系统设计中融入了动态反馈机制，使其能够适应不同的密码复杂性和操作环境。基于熵的详细评估显示，该系统对加密工作流程中的细微偏差具有高度敏感性，提供了一种传统的依赖静态签名或启发式方法的稳健替代方案。计算基准测试证实了其可扩展性和效率，在高数据负载和复杂加密场景下仍能保持一致的性能。实现实时聚类和异常评估确保了快速响应能力，有效应对了勒索软件检测中的关键延迟挑战。与现有方法的性能比较突出了其在检测效果上的改进，尤其是在对付采用扩展密钥长度和独特加密协议的高级勒索软件方面。 

---
# Can Large Language Models Be Trusted as Black-Box Evolutionary Optimizers for Combinatorial Problems? 

**Title (ZH)**: 大型语言模型可以作为组合问题的黑盒进化优化器被信任吗？ 

**Authors**: Jie Zhao, Tao Wen, Kang Hao Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2501.15081)  

**Abstract**: Evolutionary computation excels in complex optimization but demands deep domain knowledge, restricting its accessibility. Large Language Models (LLMs) offer a game-changing solution with their extensive knowledge and could democratize the optimization paradigm. Although LLMs possess significant capabilities, they may not be universally effective, particularly since evolutionary optimization encompasses multiple stages. It is therefore imperative to evaluate the suitability of LLMs as evolutionary optimizer (EVO). Thus, we establish a series of rigid standards to thoroughly examine the fidelity of LLM-based EVO output in different stages of evolutionary optimization and then introduce a robust error-correction mechanism to mitigate the output uncertainty. Furthermore, we explore a cost-efficient method that directly operates on entire populations with excellent effectiveness in contrast to individual-level optimization. Through extensive experiments, we rigorously validate the performance of LLMs as operators targeted for combinatorial problems. Our findings provide critical insights and valuable observations, advancing the understanding and application of LLM-based optimization. 

**Abstract (ZH)**: 进化计算在复杂优化方面表现出色，但需深厚的领域知识，限制了其可访问性。大型语言模型（LLMs）凭借其广泛的知识提供了变革性的解决方案，有可能普及优化范式。尽管LLMs具备显著的能力，但在进化优化的多个阶段中，它们可能并不普遍有效。因此，有必要评估LLMs作为进化优化器（EVO）的适用性。为此，我们制定了一系列严格的评价标准，全面考察基于LLM的EVO在进化优化不同阶段的输出准确度，并引入了稳健的错误校正机制以减轻输出的不确定性。此外，我们探索了一种成本效益高的方法，可以直接在种群层面进行操作，与基于个体的优化相比效果显著。通过广泛的实验，我们严格验证了LLMs作为针对组合问题的操作符的性能。我们的研究结果提供了宝贵的见解和观察，推动了基于LLM的优化的理解和应用。 

---
# PatentLMM: Large Multimodal Model for Generating Descriptions for Patent Figures 

**Title (ZH)**: PatentLMM：大型多模态模型，用于生成专利图表的描述 

**Authors**: Shreya Shukla, Nakul Sharma, Manish Gupta, Anand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2501.15074)  

**Abstract**: Writing comprehensive and accurate descriptions of technical drawings in patent documents is crucial to effective knowledge sharing and enabling the replication and protection of intellectual property. However, automation of this task has been largely overlooked by the research community. To this end, we introduce PatentDesc-355K, a novel large-scale dataset containing ~355K patent figures along with their brief and detailed textual descriptions extracted from more than 60K US patent documents. In addition, we propose PatentLMM - a novel multimodal large language model specifically tailored to generate high-quality descriptions of patent figures. Our proposed PatentLMM comprises two key components: (i) PatentMME, a specialized multimodal vision encoder that captures the unique structural elements of patent figures, and (ii) PatentLLaMA, a domain-adapted version of LLaMA fine-tuned on a large collection of patents. Extensive experiments demonstrate that training a vision encoder specifically designed for patent figures significantly boosts the performance, generating coherent descriptions compared to fine-tuning similar-sized off-the-shelf multimodal models. PatentDesc-355K and PatentLMM pave the way for automating the understanding of patent figures, enabling efficient knowledge sharing and faster drafting of patent documents. We make the code and data publicly available. 

**Abstract (ZH)**: 在专利文件中对技术图纸进行全面准确的描述对于有效的知识共享、专利的复制和知识产权保护至关重要。然而，这一任务的自动化在研究界尚未受到足够的关注。为此，我们引入了包含约35.5万个专利图形及其从超过6万份美国专利文件中提取的简短和详细文本描述的新颖大规模数据集——PatentDesc-355K。此外，我们提出了一种专门针对专利图形生成高质量描述的新型多模态大语言模型——PatentLMM。我们的PatentLMM包含两大关键组成部分：(i) PatentMME，一种专门针对专利图形的多模态视觉编码器，能够捕捉专利图形的独特结构元素；(ii) PatentLLaMA，一个经过大规模专利数据微调的LLaMA领域适配版本。大量实验表明，针对专利图形设计的视觉编码器显著提升了性能，生成的描述更具连贯性，相较于微调大小相近的现成多模态模型。PatentDesc-355K和PatentLMM为自动化理解和处理专利图形铺平了道路，促进了高效的知识共享并加快了专利文件的起草。我们公开了代码和数据。 

---
# Task Arithmetic in Trust Region: A Training-Free Model Merging Approach to Navigate Knowledge Conflicts 

**Title (ZH)**: 信任区域中的任务算术：一种导航知识冲突的无训练模型合并方法 

**Authors**: Wenju Sun, Qingyong Li, Wen Wang, Yangli-ao Geng, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.15065)  

**Abstract**: Multi-task model merging offers an efficient solution for integrating knowledge from multiple fine-tuned models, mitigating the significant computational and storage demands associated with multi-task training. As a key technique in this field, Task Arithmetic (TA) defines task vectors by subtracting the pre-trained model ($\theta_{\text{pre}}$) from the fine-tuned task models in parameter space, then adjusting the weight between these task vectors and $\theta_{\text{pre}}$ to balance task-generalized and task-specific knowledge. Despite the promising performance of TA, conflicts can arise among the task vectors, particularly when different tasks require distinct model adaptations. In this paper, we formally define this issue as knowledge conflicts, characterized by the performance degradation of one task after merging with a model fine-tuned for another task. Through in-depth analysis, we show that these conflicts stem primarily from the components of task vectors that align with the gradient of task-specific losses at $\theta_{\text{pre}}$. To address this, we propose Task Arithmetic in Trust Region (TATR), which defines the trust region as dimensions in the model parameter space that cause only small changes (corresponding to the task vector components with gradient orthogonal direction) in the task-specific losses. Restricting parameter merging within this trust region, TATR can effectively alleviate knowledge conflicts. Moreover, TATR serves as both an independent approach and a plug-and-play module compatible with a wide range of TA-based methods. Extensive empirical evaluations on eight distinct datasets robustly demonstrate that TATR improves the multi-task performance of several TA-based model merging methods by an observable margin. 

**Abstract (ZH)**: 多任务模型融合提供了一种有效的方法来整合多个微调模型的知识，从而减轻多任务训练相关的显著计算和存储需求。作为该领域的一项关键技术，任务算术（TA）通过在参数空间中从预训练模型（$\theta_{\text{pre}}$）中减去微调任务模型来定义任务向量，然后调整这些任务向量与$\theta_{\text{pre}}$之间的权重，以平衡通用任务和特定任务的知识。尽管TA具有令人期待的性能，但在某些情况下，不同的任务可能需要不同的模型适应，从而导致任务向量之间产生冲突。在本文中，我们正式将这一问题定义为知识冲突，其特征是在融合另一个任务微调模型后，一个任务的性能下降。通过深入分析，我们发现这些冲突主要源自任务向量中与$\theta_{\text{pre}}$处特定任务损失梯度方向正交的分量。为解决这一问题，我们提出了任务算术在信任区间中的方法（TATR），将信任区间定义为仅会导致任务特定损失微小变化的模型参数空间维度（对应于梯度正交方向的任务向量分量）。通过限制参数合并仅在信任区间内进行，TATR可以有效缓解知识冲突。此外，TATR既可以作为独立的方法，也可以作为与TA方法兼容的即插即用模块。通过对八个不同数据集的广泛实证评估，我们证明TATR可以明显提高几种TA方法的多任务性能。 

---
# PolaFormer: Polarity-aware Linear Attention for Vision Transformers 

**Title (ZH)**: PolaFormer：面向极性的线性注意力机制在视觉变换器中的应用 

**Authors**: Weikang Meng, Yadan Luo, Xin Li, Dongmei Jiang, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15061)  

**Abstract**: Linear attention has emerged as a promising alternative to softmax-based attention, leveraging kernelized feature maps to reduce complexity from quadratic to linear in sequence length. However, the non-negative constraint on feature maps and the relaxed exponential function used in approximation lead to significant information loss compared to the original query-key dot products, resulting in less discriminative attention maps with higher entropy. To address the missing interactions driven by negative values in query-key pairs, we propose a polarity-aware linear attention mechanism that explicitly models both same-signed and opposite-signed query-key interactions, ensuring comprehensive coverage of relational information. Furthermore, to restore the spiky properties of attention maps, we provide a theoretical analysis proving the existence of a class of element-wise functions (with positive first and second derivatives) that can reduce entropy in the attention distribution. For simplicity, and recognizing the distinct contributions of each dimension, we employ a learnable power function for rescaling, allowing strong and weak attention signals to be effectively separated. Extensive experiments demonstrate that the proposed PolaFormer improves performance on various vision tasks, enhancing both expressiveness and efficiency by up to 4.6%. 

**Abstract (ZH)**: 线性注意力已成为softmax基注意力的一种有前途的替代方案，通过核特征映射将复杂度从序列长度的平方降低到线性。然而，特征映射的非负约束以及在近似中使用的松弛指数函数导致与原始查询-键点积相比存在显著的信息丢失，从而导致区分度较低、熵较高的注意力图。为了解决由查询-键对中的负值驱动的缺失交互，我们提出了一种感知极性的线性注意力机制，该机制明确地建模了同号和异号的查询-键交互，确保全面覆盖关系信息。此外，为了恢复注意力图的尖峰特性，我们提供了一种理论分析，证明了一类元素-wise函数（具有正的一阶和二阶导数）的存在性，这些函数可以降低注意力分布的熵。为了简化起见，并考虑到每个维度的单独贡献，我们采用可学习的幂函数进行缩放，从而使强和弱的注意力信号能够有效地分离。广泛的实验表明，所提出的PolaFormer在各种视觉任务中提高了性能，最多可以提高4.6%的表达能力和效率。 

---
# Group Ligands Docking to Protein Pockets 

**Title (ZH)**: 蛋白质口袋中群组配体的对接研究 

**Authors**: Jiaqi Guan, Jiahan Li, Xiangxin Zhou, Xingang Peng, Sheng Wang, Yunan Luo, Jian Peng, Jianzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.15055)  

**Abstract**: Molecular docking is a key task in computational biology that has attracted increasing interest from the machine learning community. While existing methods have achieved success, they generally treat each protein-ligand pair in isolation. Inspired by the biochemical observation that ligands binding to the same target protein tend to adopt similar poses, we propose \textsc{GroupBind}, a novel molecular docking framework that simultaneously considers multiple ligands docking to a protein. This is achieved by introducing an interaction layer for the group of ligands and a triangle attention module for embedding protein-ligand and group-ligand pairs. By integrating our approach with diffusion-based docking model, we set a new S performance on the PDBBind blind docking benchmark, demonstrating the effectiveness of our proposed molecular docking paradigm. 

**Abstract (ZH)**: 分子对接是计算生物学中的一个关键任务，近年来吸引了机器学习领域越来越多的关注。尽管现有的方法已经取得了一定的成功，但它们通常将每对蛋白质-配体单独进行处理。受生物化学观察到的配体倾向于在相同的靶蛋白上采用相似构象的启发，我们提出了一种名为**GroupBind**的新型分子对接框架，该框架能够同时考虑多个配体与蛋白质的对接。这一目标通过引入一个交互层（用于配体群组）以及一个三角注意力模块（用于嵌入蛋白质-配体和群组-配体对）得以实现。通过将我们的方法与基于扩散的分子对接模型相结合，我们在PDBBind盲对接基准测试中达到了新的性能标准，这表明我们提出的分子对接范式具有有效性。 

---
# An Attempt to Unraveling Token Prediction Refinement and Identifying Essential Layers of Large Language Models 

**Title (ZH)**: 尝试解开_token预测精化_并识别大型语言模型中的核心层 

**Authors**: Jaturong Kongmanee  

**Link**: [PDF](https://arxiv.org/pdf/2501.15054)  

**Abstract**: This research aims to unravel how large language models (LLMs) iteratively refine token predictions (or, in a general sense, vector predictions). We utilized a logit lens technique to analyze the model's token predictions derived from intermediate representations. Specifically, we focused on how LLMs access and use information from input contexts, and how positioning of relevant information affects the model's token prediction refinement process. Our findings for multi-document question answering task, by varying input context lengths (the number of documents), using GPT-2, revealed that the number of layers between the first layer that the model predicted next tokens correctly and the later layers that the model finalized its correct predictions, as a function of the position of relevant information (i.e., placing the relevant one at the beginning, middle, or end of the input context), has a nearly inverted U shape. We found that the gap between these two layers, on average, diminishes when relevant information is positioned at the beginning or end of the input context, suggesting that the model requires more refinements when processing longer contexts with relevant information situated in the middle, and highlighting which layers are essential for determining the correct output. Our analysis provides insights about how token predictions are distributed across different conditions, and establishes important connections to existing hypotheses and previous findings in AI safety research and development. 

**Abstract (ZH)**: 本研究旨在揭示大型语言模型（LLMs）如何迭代细化令牌预测（或更广泛地说，向量预测）。我们采用了一种logit透镜技术来分析模型从中间表示中得到的令牌预测。具体而言，我们关注的是LLMs如何访问和利用输入上下文中的信息，以及相关信息在模型的令牌预测细化过程中所处的位置如何影响这一过程。通过在多文档问答任务中改变输入上下文长度（即文档的数量），使用GPT-2这一方法，我们发现，从模型首次正确预测下一个令牌的层到模型最终确定正确预测的层之间间隔的层数，作为相关信息位置函数（即将其置于输入上下文的开头、中途或结尾），呈现出几乎倒U形曲线。我们发现，当相关信息置于输入上下文的开头或结尾时，这两层之间的差距通常较小，表明在处理包含相关信息位于中间的更长上下文时，模型需要更多的细化，并突显了哪些层对确定正确输出至关重要。我们的分析提供了关于不同条件下令牌预测分布的见解，并与AI安全研究及开发中现有的假设和先前发现建立了重要的联系。 

---
# Exploring the impact of Optimised Hyperparameters on Bi-LSTM-based Contextual Anomaly Detector 

**Title (ZH)**: 探索优化超参数对基于双路长短期记忆网络（Bi-LSTM）的上下文异常检测器影响的研究 

**Authors**: Aafan Ahmad Toor, Jia-Chun Lin, Ernst Gunnar Gran  

**Link**: [PDF](https://arxiv.org/pdf/2501.15053)  

**Abstract**: The exponential growth in the usage of Internet of Things in daily life has caused immense increase in the generation of time series data. Smart homes is one such domain where bulk of data is being generated and anomaly detection is one of the many challenges addressed by researchers in recent years. Contextual anomaly is a kind of anomaly that may show deviation from the normal pattern like point or sequence anomalies, but it also requires prior knowledge about the data domain and the actions that caused the deviation. Recent studies based on Recurrent Neural Networks (RNN) have demonstrated strong performance in anomaly detection. This study explores the impact of automatically tuned hyperparamteres on Unsupervised Online Contextual Anomaly Detection (UoCAD) approach by proposing UoCAD with Optimised Hyperparamnters (UoCAD-OH). UoCAD-OH conducts hyperparameter optimisation on Bi-LSTM model in an offline phase and uses the fine-tuned hyperparameters to detect anomalies during the online phase. The experiments involve evaluating the proposed framework on two smart home air quality datasets containing contextual anomalies. The evaluation metrics used are Precision, Recall, and F1 score. 

**Abstract (ZH)**: 物联网在日常生活中的指数级增长导致了时间序列数据生成的巨大增加。智能家居是一个数据生成量极大的领域，近年来研究人员已经面临了包括上下文异常检测在内的诸多挑战。上下文异常是一种与点异常或序列异常类似，可能会偏离正常模式的异常类型，但它还需要关于数据领域和导致偏差的动作的先验知识。基于循环神经网络（RNN）的近期研究表明，这种网络在异常检测方面表现出色。本研究通过提出优化超参数的在线上下文异常检测（UoCAD）方法——即UoCAD-OH，来探索自动调优超参数对UoCAD方法的影响。UoCAD-OH在离线阶段对双向长短期记忆（Bi-LSTM）模型进行超参数优化，并在在线阶段使用优化后的超参数进行异常检测。实验包括在包含上下文异常的两个智能家居空气质量数据集上评估所提出的框架，并使用的评估指标包括精准率（Precision）、召回率（Recall）和F1分数（F1 score）。 

---
# Graph-Based Cross-Domain Knowledge Distillation for Cross-Dataset Text-to-Image Person Retrieval 

**Title (ZH)**: 基于图的跨域知识精练方法在跨数据集文本到图像的人像检索中的应用 

**Authors**: Bingjun Luo, Jinpeng Wang, Wang Zewen, Junjie Zhu, Xibin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15052)  

**Abstract**: Video surveillance systems are crucial components for ensuring public safety and management in smart city. As a fundamental task in video surveillance, text-to-image person retrieval aims to retrieve the target person from an image gallery that best matches the given text description. Most existing text-to-image person retrieval methods are trained in a supervised manner that requires sufficient labeled data in the target domain. However, it is common in practice that only unlabeled data is available in the target domain due to the difficulty and cost of data annotation, which limits the generalization of existing methods in practical application scenarios. To address this issue, we propose a novel unsupervised domain adaptation method, termed Graph-Based Cross-Domain Knowledge Distillation (GCKD), to learn the cross-modal feature representation for text-to-image person retrieval in a cross-dataset scenario. The proposed GCKD method consists of two main components. Firstly, a graph-based multi-modal propagation module is designed to bridge the cross-domain correlation among the visual and textual samples. Secondly, a contrastive momentum knowledge distillation module is proposed to learn the cross-modal feature representation using the online knowledge distillation strategy. By jointly optimizing the two modules, the proposed method is able to achieve efficient performance for cross-dataset text-to-image person retrieval. acExtensive experiments on three publicly available text-to-image person retrieval datasets demonstrate the effectiveness of the proposed GCKD method, which consistently outperforms the state-of-the-art baselines. 

**Abstract (ZH)**: 视频监控系统是确保智慧城市中公共安全和管理的关键组件。作为视频监控中的基础任务，从文本到图像的行人检索旨在从包含给定文本描述的最佳匹配目标行人的图像库中检索目标行人体。现有的大多数从文本到图像的行人检索方法都是在监督模式下训练的，这需要目标领域中充足的标注数据。然而，在实践中，由于数据标注的困难和成本，目标领域中往往只有未标注的数据可用，这限制了现有方法在实际应用中的普适性。为了解决这一问题，我们提出了一种新的无监督领域自适应方法，称为基于图的跨域知识蒸馏（GCKD），以在跨数据集场景中学习文本到图像的行人检索的跨模态特征表示。所提出的GCKD方法由两个主要组成部分组成。首先，设计了一种基于图的多模态传播模块，以在视觉和文本样本之间建立跨域关联。其次，提出了一种对比增强动力知识蒸馏模块，使用在线知识蒸馏策略学习跨模态特征表示。通过联合优化这两个模块，所提出的方法能够在跨数据集的文本到图像行人检索中实现高效的性能。在三个公开的文本到图像行人检索数据集上的大量实验表明，所提出的GCKD方法的有效性，其在所有基线方法中的表现均优于最先进的方法。 

---
# Evaluating Hallucination in Large Vision-Language Models based on Context-Aware Object Similarities 

**Title (ZH)**: 基于上下文感知对象相似性的大型视觉-语言模型 hallucination 评价 

**Authors**: Shounak Datta, Dhanasekar Sundararaman  

**Link**: [PDF](https://arxiv.org/pdf/2501.15046)  

**Abstract**: Despite their impressive performance on multi-modal tasks, large vision-language models (LVLMs) tend to suffer from hallucinations. An important type is object hallucination, where LVLMs generate objects that are inconsistent with the images shown to the model. Existing works typically attempt to quantify object hallucinations by detecting and measuring the fraction of hallucinated objects in generated captions. Additionally, more recent work also measures object hallucinations by directly querying the LVLM with binary questions about the presence of likely hallucinated objects based on object statistics like top-k frequent objects and top-k co-occurring objects. In this paper, we present Context-Aware Object Similarities (CAOS), a novel approach for evaluating object hallucination in LVLMs using object statistics as well as the generated captions. CAOS uniquely integrates object statistics with semantic relationships between objects in captions and ground-truth data. Moreover, existing approaches usually only detect and measure hallucinations belonging to a predetermined set of in-domain objects (typically the set of all ground-truth objects for the training dataset) and ignore generated objects that are not part of this set, leading to under-evaluation. To address this, we further employ language model--based object recognition to detect potentially out-of-domain hallucinated objects and use an ensemble of LVLMs for verifying the presence of such objects in the query image. CAOS also examines the sequential dynamics of object generation, shedding light on how the order of object appearance influences hallucinations, and employs word embedding models to analyze the semantic reasons behind hallucinations. CAOS aims to offer a nuanced understanding of the hallucination tendencies of LVLMs by providing a systematic framework to identify and interpret object hallucinations. 

**Abstract (ZH)**: 尽管大规模视觉-语言模型（LVLMs）在多模态任务中表现出色，但它们往往容易出现幻觉现象。其中一种重要类型的幻觉是物体幻觉，即LVLMs生成与提供的图像不一致的物体。现有方法通常通过检测并衡量生成字幕中幻象物体的比例来量化物体幻觉。此外，最近的一些研究直接通过针对物体统计信息（如最常见前k个物体和最常共现前k个物体）中的可能幻象物体提出二元问题来衡量物体幻觉。在本文中，我们提出了上下文感知物体相似性（CAOS，Context-Aware Object Similarities）方法，这是一种利用物体统计信息和生成字幕评估LVLMs中物体幻觉的新颖方法。CAOS独特地将物体统计信息与字幕中的语义关系以及真实数据相结合。此外，现有方法通常只检测并衡量属于某一预设领域物体集（通常是训练数据集中所有真实物体集）中的幻觉物体，并忽略不属于这一集的生成物体，从而导致评估不足。为解决这一问题，我们进一步利用基于语言模型的物体识别来检测潜在的跨领域幻觉物体，并使用多个LVLM的集合来验证查询图像中这些物体的存在性。CAOS还探讨了物体生成的顺序动态，揭示了物体出现顺序如何影响幻觉，并利用词嵌入模型分析幻觉的根本语义原因。CAOS旨在通过提供系统的方法来识别和解释物体幻觉，为理解LVLMs的幻觉倾向提供细腻的理解。 

---
# Towards Robust Unsupervised Attention Prediction in Autonomous Driving 

**Title (ZH)**: 朝着在自动驾驶中实现稳健的无监督注意力预测的研究 

**Authors**: Mengshi Qi, Xiaoyang Bi, Pengfei Zhu, Huadong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.15045)  

**Abstract**: Robustly predicting attention regions of interest for self-driving systems is crucial for driving safety but presents significant challenges due to the labor-intensive nature of obtaining large-scale attention labels and the domain gap between self-driving scenarios and natural scenes. These challenges are further exacerbated by complex traffic environments, including camera corruption under adverse weather, noise interferences, and central bias from long-tail distributions. To address these issues, we propose a robust unsupervised attention prediction method. An Uncertainty Mining Branch refines predictions by analyzing commonalities and differences across multiple pre-trained models on natural scenes, while a Knowledge Embedding Block bridges the domain gap by incorporating driving knowledge to adaptively enhance pseudo-labels. Additionally, we introduce RoboMixup, a novel data augmentation method that improves robustness against corruption through soft attention and dynamic augmentation, and mitigates central bias by integrating random cropping into Mixup as a this http URL systematically evaluate robustness in self-driving attention prediction, we introduce the DriverAttention-C benchmark, comprising over 100k frames across three subsets: BDD-A-C, DR(eye)VE-C, and DADA-2000-C. Our method achieves performance equivalent to or surpassing fully supervised state-of-the-art approaches on three public datasets and the proposed robustness benchmark, reducing relative corruption degradation by 58.8% and 52.8%, and improving central bias robustness by 12.4% and 11.4% in KLD and CC metrics, respectively. Code and data are available at this https URL. 

**Abstract (ZH)**: 稳健预测自动驾驶系统中的注意力区域对于驾驶安全至关重要，但由于获取大规模注意力标签的劳动密集型性质以及自动驾驶场景与自然场景之间的领域差距，这一任务面临重大挑战。复杂交通环境进一步加剧了这些挑战，包括恶劣天气下的摄像头损坏、噪声干扰以及长尾分布带来的中心偏差。为解决这些问题，我们提出了一种稳健的无监督注意力预测方法。不确定性挖掘分支通过分析多个预训练模型在自然场景上的预测结果的共同点和差异点来细化预测，知识嵌入块通过融入驾驶知识来适应性地增强伪标签，缩小领域差距。此外，我们引入了RoboMixup，这是一种新颖的数据增强方法，通过软注意力和动态增强提高对抗损坏的鲁棒性，并通过集成随机裁剪到Mixup中来缓解中心偏差。为了系统性地评估自动驾驶注意力预测的鲁棒性，我们提出了DriverAttention-C基准数据集，包含超过100,000帧的三个子集：BDD-A-C、DR(eye)VE-C和DADA-2000-C。我们的方法在三个公开数据集和提出的鲁棒性基准上实现了与全监督状态-of-the-art方法相当或更好的性能，相对损坏退化降低了58.8%和52.8%，KL距离和CC指标下的中心偏差鲁棒性分别提高了12.4%和11.4%。相关代码和数据可在[此处](this http URL)获取。 

---
# Adaptive Client Selection in Federated Learning: A Network Anomaly Detection Use Case 

**Title (ZH)**: 联邦学习中自适应客户端选择：一种网络异常检测应用案例 

**Authors**: William Marfo, Deepak K. Tosh, Shirley V. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2501.15038)  

**Abstract**: Federated Learning (FL) has become a widely used approach for training machine learning models on decentralized data, addressing the significant privacy concerns associated with traditional centralized methods. However, the efficiency of FL relies on effective client selection and robust privacy preservation mechanisms. Ineffective client selection can result in suboptimal model performance, while inadequate privacy measures risk exposing sensitive data.
This paper introduces a client selection framework for FL that incorporates differential privacy and fault tolerance. The proposed adaptive approach dynamically adjusts the number of selected clients based on model performance and system constraints, ensuring privacy through the addition of calibrated noise.
The method is evaluated on a network anomaly detection use case using the UNSW-NB15 and ROAD datasets. Results demonstrate up to a 7% improvement in accuracy and a 25% reduction in training time compared to the FedL2P approach. Additionally, the study highlights trade-offs between privacy budgets and model performance, with higher privacy budgets leading to reduced noise and improved accuracy. While the fault tolerance mechanism introduces a slight performance decrease, it enhances robustness against client failures. Statistical validation using the Mann-Whitney U test confirms the significance of these improvements, with results achieving a p-value of less than 0.05. 

**Abstract (ZH)**: 联邦学习（FL）已成为在去中心化数据上训练机器学习模型的一种广泛采用的方法，解决了传统集中式方法相关的重大隐私问题。然而，FL的效率依赖于有效的客户端选择和稳健的隐私保护机制。无效的客户端选择可能导致模型性能不佳，而不充分的隐私措施则可能暴露敏感数据。

本文提出了一种结合差分隐私和容错机制的客户端选择框架。该提议的自适应方法根据模型性能和系统约束动态调整选择的客户端数量，通过加入校准噪声确保隐私。

该方法在使用 UNSW-NB15 和 ROAD 数据集的网络异常检测应用场景中进行了评估。结果表明，与 FedL2P 方法相比，该方法在精度上可提升最高达 7%，训练时间减少 25%。此外，研究还突出了隐私预算与模型性能之间的权衡，更高的隐私预算会导致_noise_减少和准确率提高。虽然容错机制会导致轻微的性能下降，但它增强了对客户端故障的鲁棒性。使用曼-惠特尼 U 检验的统计验证证实了这些改进的重要性，结果的 p 值均小于 0.05。

注：在英文原文中，“noise”被用作技术术语，但原文中提到“加性噪声”的一致性，为了保持准确性和一致性，此处翻译使用了“噪声”，而非直接翻译为中文的“噪音”。在上下文中，这指的是添加到数据中的随机噪声，用于保护隐私。

请确认“Noise”是否应该理解为加性噪声，并且确认术语“p值”在中文中的翻译是否适合。 

---
# Divergence-Augmented Policy Optimization 

**Title (ZH)**: 增强偏差的策略优化 

**Authors**: Qing Wang, Yingru Li, Jiechao Xiong, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15034)  

**Abstract**: In deep reinforcement learning, policy optimization methods need to deal with issues such as function approximation and the reuse of off-policy data. Standard policy gradient methods do not handle off-policy data well, leading to premature convergence and instability. This paper introduces a method to stabilize policy optimization when off-policy data are reused. The idea is to include a Bregman divergence between the behavior policy that generates the data and the current policy to ensure small and safe policy updates with off-policy data. The Bregman divergence is calculated between the state distributions of two policies, instead of only on the action probabilities, leading to a divergence augmentation formulation. Empirical experiments on Atari games show that in the data-scarce scenario where the reuse of off-policy data becomes necessary, our method can achieve better performance than other state-of-the-art deep reinforcement learning algorithms. 

**Abstract (ZH)**: 在深度强化学习中，策略优化方法需要处理函数逼近和离策略数据的重用问题。标准的策略梯度方法对离策略数据处理不佳，导致过早收敛和稳定性问题。本文介绍了一种在重用离策略数据时稳定策略优化的方法。其核心思想是通过在生成数据的行为策略和当前策略之间包含Bregman散度，以确保使用离策略数据时可以进行小而安全的策略更新。Bregman散度是基于两个策略的状态分布计算的，而不仅仅是基于动作概率，从而形成了一个散度增强的形式。在Atari游戏中进行的经验实验显示，在数据稀缺的场景下，当需要重用离策略数据时，我们的方法能够优于其他最先进的深度强化学习算法。 

---
# OptiSeq: Optimizing Example Ordering for In-Context Learning 

**Title (ZH)**: OptiSeq: 优化示例顺序以实现上下文学习 

**Authors**: Rahul Atul Bhope, Praveen Venkateswaran, K. R. Jayaram, Vatche Isahagian, Vinod Muthusamy, Nalini Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2501.15030)  

**Abstract**: Developers using LLMs in their applications and agents have provided plenty of anecdotal evidence that in-context-learning (ICL) is fragile. In addition to the quantity and quality of examples, we show that the order in which the in-context examples are listed in the prompt affects the output of the LLM and, consequently, their performance. In this paper, we present OptiSeq, which introduces a score based on log probabilities of LLM outputs to prune the universe of possible example orderings in few-shot ICL and recommend the best order(s) by distinguishing between correct and incorrect outputs resulting from different order permutations. Through a detailed empirical evaluation on multiple LLMs, datasets and prompts, we demonstrate that OptiSeq improves accuracy by 6 - 10.5 percentage points across multiple tasks. 

**Abstract (ZH)**: 本文探讨了开发者在其应用程序和代理中使用大型语言模型（LLMs）时提供的大量经验性证据，表明基于上下文学习（ICL）可能是脆弱的。除了示例的数量和质量之外，我们还展示了在提示中列出的基于上下文示例的顺序也会影响LLM的输出及其性能。本文提出了OptiSeq，一种基于LLM输出对数概率的评分方法，用于在少量示例ICL中修剪可能的示例排序范畴，并通过区分不同排列顺序产生的正确和错误输出来推荐最佳排序。通过在多种LLM、数据集和提示上的详细实证评估，我们证明OptiSeq可以在多种任务中将准确性提高6-10.5个百分点。 

---
# Using Large Language Models for education managements in Vietnamese with low resources 

**Title (ZH)**: 使用大规模语言模型进行越南地区资源有限的教育管理 

**Authors**: Duc Do Minh, Vinh Nguyen Van, Thang Dam Cong  

**Link**: [PDF](https://arxiv.org/pdf/2501.15022)  

**Abstract**: Large language models (LLMs), such as GPT-4, Gemini 1.5, Claude 3.5 Sonnet, and Llama3, have demonstrated significant advancements in various NLP tasks since the release of ChatGPT in 2022. Despite their success, fine-tuning and deploying LLMs remain computationally expensive, especially in resource-constrained environments. In this paper, we proposed VietEduFrame, a framework specifically designed to apply LLMs to educational management tasks in Vietnamese institutions. Our key contribution includes the development of a tailored dataset, derived from student education documents at Hanoi VNU, which addresses the unique challenges faced by educational systems with limited resources. Through extensive experiments, we show that our approach outperforms existing methods in terms of accuracy and efficiency, offering a promising solution for improving educational management in under-resourced environments. While our framework leverages synthetic data to supplement real-world examples, we discuss potential limitations regarding broader applicability and robustness in future implementations. 

**Abstract (ZH)**: 自2022年ChatGPT发布以来，大型语言模型（LLMs），如GPT-4、Gemini 1.5、Claude 3.5 Sonnet和Llama 3，在各种自然语言处理（NLP）任务中展现了显著的进步。尽管取得了成功，但在资源受限的环境中，LLMs的微调和部署仍然非常耗计算资源。本文提出了一种名为VietEduFrame的框架，专门设计用于在越南教育机构中应用LLMs于教育管理任务。我们的主要贡献包括开发了一个专为此目的定制的数据集，该数据集源自河内越南国家大学的学生教育文档，旨在解决资源有限的教育系统所面临的独特挑战。通过广泛实验，我们表明，我们的方法在准确性与效率方面优于现有方法，为改进资源有限环境下的教育管理提供了有前景的解决方案。尽管我们的框架利用合成数据来补充现实世界示例，但我们讨论了未来实现中可能存在的适用性和鲁棒性方面的局限性。 

---
# On Accelerating Edge AI: Optimizing Resource-Constrained Environments 

**Title (ZH)**: 加速边缘AI：优化资源受限环境 

**Authors**: Jacob Sander, Achraf Cohen, Venkat R. Dasari, Brent Venable, Brian Jalaian  

**Link**: [PDF](https://arxiv.org/pdf/2501.15014)  

**Abstract**: Resource-constrained edge deployments demand AI solutions that balance high performance with stringent compute, memory, and energy limitations. In this survey, we present a comprehensive overview of the primary strategies for accelerating deep learning models under such constraints. First, we examine model compression techniques-pruning, quantization, tensor decomposition, and knowledge distillation-that streamline large models into smaller, faster, and more efficient variants. Next, we explore Neural Architecture Search (NAS), a class of automated methods that discover architectures inherently optimized for particular tasks and hardware budgets. We then discuss compiler and deployment frameworks, such as TVM, TensorRT, and OpenVINO, which provide hardware-tailored optimizations at inference time. By integrating these three pillars into unified pipelines, practitioners can achieve multi-objective goals, including latency reduction, memory savings, and energy efficiency-all while maintaining competitive accuracy. We also highlight emerging frontiers in hierarchical NAS, neurosymbolic approaches, and advanced distillation tailored to large language models, underscoring open challenges like pre-training pruning for massive networks. Our survey offers practical insights, identifies current research gaps, and outlines promising directions for building scalable, platform-independent frameworks to accelerate deep learning models at the edge. 

**Abstract (ZH)**: 资源受限的边缘部署需要兼顾高性能与严格的计算、内存和能效限制的AI解决方案。在这篇综述中，我们将全面概述在这些约束条件下加速深度学习模型的主要策略。首先，我们考察了模型压缩技术——剪枝、量化、张量分解和知识蒸馏，这些技术可以将大型模型简化为较小、更快且更高效的版本。接着，我们探讨了神经架构搜索（NAS），这是一种自动化方法，可以发现适用于特定任务和硬件预算的优化架构。然后，我们讨论了编译和部署框架，如TVM、TensorRT和OpenVINO，这些框架可以在推断时提供针对硬件的优化。通过将这三个支柱整合到统一的管道中，实践者可以实现包括延时减少、内存节省和能效提高在内的多目标优化，同时保持竞争力的准确率。我们还强调了分层NAS、神经符号方法以及针对大规模语言模型的先进蒸馏等新兴前沿领域，指出了一些开放挑战，例如超大规模网络的预训练剪枝。我们的综述提供了实用见解，指出了当前的研究缺口，并概述了构建可扩展、平台无关的框架以加速边缘深度学习模型的前景方向。 

---
# Robust Cross-Etiology and Speaker-Independent Dysarthric Speech Recognition 

**Title (ZH)**: 稳健的跨病因与 speaker 独立构音障碍语音识别 

**Authors**: Satwinder Singh, Qianli Wang, Zihan Zhong, Clarion Mendes, Mark Hasegawa-Johnson, Waleed Abdulla, Seyed Reza Shahamiri  

**Link**: [PDF](https://arxiv.org/pdf/2501.14994)  

**Abstract**: In this paper, we present a speaker-independent dysarthric speech recognition system, with a focus on evaluating the recently released Speech Accessibility Project (SAP-1005) dataset, which includes speech data from individuals with Parkinson's disease (PD). Despite the growing body of research in dysarthric speech recognition, many existing systems are speaker-dependent and adaptive, limiting their generalizability across different speakers and etiologies. Our primary objective is to develop a robust speaker-independent model capable of accurately recognizing dysarthric speech, irrespective of the speaker. Additionally, as a secondary objective, we aim to test the cross-etiology performance of our model by evaluating it on the TORGO dataset, which contains speech samples from individuals with cerebral palsy (CP) and amyotrophic lateral sclerosis (ALS). By leveraging the Whisper model, our speaker-independent system achieved a CER of 6.99% and a WER of 10.71% on the SAP-1005 dataset. Further, in cross-etiology settings, we achieved a CER of 25.08% and a WER of 39.56% on the TORGO dataset. These results highlight the potential of our approach to generalize across unseen speakers and different etiologies of dysarthria. 

**Abstract (ZH)**: 在本文中，我们提出了一种独立说话人迟缓言语识别系统，并重点评估了最近发布的声音无障碍项目（SAP-1005）数据集，该数据集包括帕金森病（PD）患者的语音数据。尽管在迟缓言语识别领域已有大量研究，但许多现有的系统依赖特定说话人并且具有自适应性，限制了它们在不同说话人和病因之间的一般适用性。我们的主要目标是开发一种健壮的独立说话人模型，能够准确识别迟缓言语，不受说话人影响。此外，作为次要目标，我们旨在通过在TORGO数据集上测试我们的模型的跨病因表现来评估该模型的性能，该数据集包括脑瘫（CP）和肌萎缩侧索硬化症（ALS）患者的语音样本。通过利用Whisper模型，我们提出的独立说话人系统在SAP-1005数据集上实现了6.99%的字符错误率（CER）和10.71%的词错误率（WER）。进一步地，在跨病因设置下，我们在TORGO数据集上实现了25.08%的CER和39.56%的WER。这些结果突显了我们方法在不同病因条件下的迟缓言语识别中具有跨未知说话人的推广潜力。 

---
# A Deep State Space Model for Rainfall-Runoff Simulations 

**Title (ZH)**: 一种用于径流模拟的深度状态空间模型 

**Authors**: Yihan Wang, Lujun Zhang, Annan Yu, N. Benjamin Erichson, Tiantian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14980)  

**Abstract**: The classical way of studying the rainfall-runoff processes in the water cycle relies on conceptual or physically-based hydrologic models. Deep learning (DL) has recently emerged as an alternative and blossomed in hydrology community for rainfall-runoff simulations. However, the decades-old Long Short-Term Memory (LSTM) network remains the benchmark for this task, outperforming newer architectures like Transformers. In this work, we propose a State Space Model (SSM), specifically the Frequency Tuned Diagonal State Space Sequence (S4D-FT) model, for rainfall-runoff simulations. The proposed S4D-FT is benchmarked against the established LSTM and a physically-based Sacramento Soil Moisture Accounting model across 531 watersheds in the contiguous United States (CONUS). Results show that S4D-FT is able to outperform the LSTM model across diverse regions. Our pioneering introduction of the S4D-FT for rainfall-runoff simulations challenges the dominance of LSTM in the hydrology community and expands the arsenal of DL tools available for hydrological modeling. 

**Abstract (ZH)**: 在水循环中研究降雨径流过程的经典方法依赖于概念模型或物理基础的水文学模型。深度学习（DL）最近在水文领域中作为一种替代方案获得了蓬勃发展，特别是在降雨径流模拟方面。然而，传统的长短期记忆（LSTM）网络仍然在这方面担任基准模型，胜过了诸如变换器等 newer 架构。在本研究中，我们提出了一种状态空间模型（SSM），具体是频率调谐对角状态空间序列（S4D-FT）模型，用于降雨径流模拟。我们在这项工作中列出了 S4D-FT 模型，并将其与成熟的 LSTM 模型和 SacRCM（Sacramento 土壤水分账户）物理基础模型进行了比较，评估了美国本土 531 个流域的表现。结果表明，S4D-FT 在不同区域均能优于 LSTM 模型。我们首次将 S4D-FT 引入降雨径流模拟挑战了 LSTM 在水文领域中的主导地位，扩展了可用于水文模型的 DL 工具箱。 

---
# AI-driven Wireless Positioning: Fundamentals, Standards, State-of-the-art, and Challenges 

**Title (ZH)**: 基于AI的无线定位：基本原理、标准、最新进展与挑战 

**Authors**: Guangjin Pan, Yuan Gao, Yilin Gao, Zhiyong Zhong, Xiaoyu Yang, Xinyu Guo, Shugong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14970)  

**Abstract**: Wireless positioning technologies hold significant value for applications in autonomous driving, extended reality (XR), unmanned aerial vehicles (UAVs), and more. With the advancement of artificial intelligence (AI), leveraging AI to enhance positioning accuracy and robustness has emerged as a field full of potential. Driven by the requirements and functionalities defined in the 3rd Generation Partnership Project (3GPP) standards, AI/machine learning (ML)-based positioning is becoming a key technology to overcome the limitations of traditional methods. This paper begins with an introduction to the fundamentals of AI and wireless positioning, covering AI models, algorithms, positioning applications, emerging wireless technologies, and the basics of positioning techniques. Subsequently, focusing on standardization progress, we provide a comprehensive review of the evolution of 3GPP positioning standards, with an emphasis on the integration of AI/ML technologies in recent and upcoming releases. Based on the AI/ML-assisted positioning and direct AI/ML positioning schemes outlined in the standards, we conduct an in-depth investigation of related research. we focus on state-of-the-art (SOTA) research in AI-based line-of-sight (LOS)/non-line-of-sight (NLOS) detection, time of arrival (TOA)/time difference of arrival (TDOA) estimation, and angle estimation techniques. For Direct AI/ML Positioning, we explore SOTA advancements in fingerprint-based positioning, knowledge-assisted AI positioning, and channel charting-based positioning. Furthermore, we introduce publicly available datasets for wireless positioning and conclude by summarizing the challenges and opportunities of AI-driven wireless positioning. 

**Abstract (ZH)**: 无线定位技术在自动驾驶、扩展现实（XR）、无人飞行器（UAVs）等应用中具有重要的价值。随着人工智能（AI）的发展，利用AI提高定位精度和鲁棒性已经成为一个充满潜力的领域。受到第三代合作伙伴计划（3GPP）标准中定义的需求和功能驱动，基于AI/机器学习（ML）的定位技术正逐步成为克服传统方法局限的关键技术。本文首先介绍了AI和无线定位的基础知识，涵盖了AI模型、算法、定位应用、新兴无线技术以及定位技术的基本原理。随后，我们重点回顾了3GPP定位标准的发展演变，并着重介绍了近年来以及即将发布的版本中AI/ML技术的集成。基于标准中提出的AI/ML辅助定位和直接AI/ML定位方案，我们进行了深入的研究。我们在LOS/NLOS检测、TOA/TDOA估测和角度估测技术的前沿研究方面进行了重点分析。对于直接AI/ML定位，我们探讨了基于指纹定位、知识辅助AI定位和基于信道測绘的定位的最新进展。此外，我们介绍了可用于无线定位的公开数据集，并总结了AI驱动的无线定位面临的挑战与机遇。 

---
# LLM4DistReconfig: A Fine-tuned Large Language Model for Power Distribution Network Reconfiguration 

**Title (ZH)**: LLM4DistReconfig：一个用于电力配电网重构的微调大型语言模型 

**Authors**: Panayiotis Christou, Md. Zahidul Islam, Yuzhang Lin, Jingwei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.14960)  

**Abstract**: Power distribution networks are evolving due to the integration of DERs and increased customer participation. To maintain optimal operation, minimize losses, and meet varying load demands, frequent network reconfiguration is necessary. Traditionally, the reconfiguration task relies on optimization software and expert operators, but as systems grow more complex, faster and more adaptive solutions are required without expert intervention. Data-driven reconfiguration is gaining traction for its accuracy, speed, and robustness against incomplete network data. LLMs, with their ability to capture complex patterns, offer a promising approach for efficient and responsive network reconfiguration in evolving complex power networks.
In this work, we introduce LLM4DistReconfig, a deep learning-based approach utilizing a fine-tuned LLM to solve the distribution network reconfiguration problem. By carefully crafting prompts and designing a custom loss function, we train the LLM with inputs representing network parameters such as buses, available lines, open lines, node voltages, and system loss. The model then predicts optimal reconfigurations by outputting updated network configurations that minimize system loss while meeting operational constraints. Our approach significantly reduces inference time compared to classical algorithms, allowing for near real-time optimal reconfiguration after training. Experimental results show that our method generates optimal configurations minimizing system loss for five individual and a combined test dataset. It also produces minimal invalid edges, no cycles, or subgraphs across all datasets, fulfilling domain-specific needs. Additionally, the generated responses contain less than 5% improper outputs on seen networks and satisfactory results on unseen networks, demonstrating its effectiveness and reliability for the reconfiguration task. 

**Abstract (ZH)**: 配电网络由于分布式能源资源（DERs）的整合和客户参与度的提高而不断演进。为了维持最优运行状态、降低损失并满足不断变化的负荷需求，频繁的网络重构变得必要。传统上，网络重构任务依赖于优化软件和专家操作员，但随着系统变得越来越复杂，需要能够更快地适应且无需专家干预的解决方案。基于数据的重构因其准确性、速度以及对不完整网络数据的鲁棒性而受到青睐。大规模语言模型（LLMs）能够捕捉复杂的模式，为在不断演进的复杂配电网络中实现高效和响应性网络重构提供了潜在的解决方案。

在本研究中，我们提出了LLM4DistReconfig，这是一种利用微调后的LLM来解决配电网络重构问题的深度学习方法。通过精心设计提示词并设计自定义损失函数，我们将网络参数如母线、可用线路、断开线路、节点电压和系统损耗作为输入数据对模型进行训练。模型通过输出更新后的网络配置，最小化系统损耗并满足运营约束来预测最优重构方案。与经典算法相比，我们的方法显著减少了推理时间，使得在训练完成后能够实现近乎实时的最优重构。实验结果显示，我们的方法能够为五个单独的和合并的测试数据集生成最优配置，同时最小化系统损耗。此外，所有数据集均未出现无效边、环路或子图，满足特定领域的需要。另外，生成的响应在已见过的网络中包含不到5%的不当输出，并在未见过的网络中也表现出令人满意的结果，这表明其在重构任务中的有效性和可靠性。 

---
# The Curious Case of Arbitrariness in Machine Learning 

**Title (ZH)**: 机器学习中任意性的有趣问题 

**Authors**: Prakhar Ganesh, Afaf Taik, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2501.14959)  

**Abstract**: Algorithmic modelling relies on limited information in data to extrapolate outcomes for unseen scenarios, often embedding an element of arbitrariness in its decisions. A perspective on this arbitrariness that has recently gained interest is multiplicity-the study of arbitrariness across a set of "good models", i.e., those likely to be deployed in practice. In this work, we systemize the literature on multiplicity by: (a) formalizing the terminology around model design choices and their contribution to arbitrariness, (b) expanding the definition of multiplicity to incorporate underrepresented forms beyond just predictions and explanations, (c) clarifying the distinction between multiplicity and other traditional lenses of arbitrariness, i.e., uncertainty and variance, and (d) distilling the benefits and potential risks of multiplicity into overarching trends, situating it within the broader landscape of responsible AI. We conclude by identifying open research questions and highlighting emerging trends in this young but rapidly growing area of research. 

**Abstract (ZH)**: 算法建模依赖数据中的有限信息来外推未见情景的结果，通常在其决策中嵌入了一定程度的主观性。近期，对这种主观性的一种关注视角是“多重性”——研究在一个“好模型”集合中的主观性，即那些在实际中很可能被部署的模型。在这项工作中，我们通过以下方式系统化地整理了关于多重性的文献：(a) 对模型设计选择及其对主观性贡献的术语进行形式化定义；(b) 扩展多重性的定义，以纳入预测和解释之外的代表性不足的形式；(c) 界定多重性与其他传统主观性视角（如不确定性与方差）之间的差异；(d) 提炼多重性带来的优势和潜在风险，并将其置于负责人工智能的更广阔背景下。最后，我们识别了研究中的开放问题，并突出了这一年轻但迅速增长的研究领域中新兴的趋势。 

---
# ExPerT: Effective and Explainable Evaluation of Personalized Long-Form Text Generation 

**Title (ZH)**: ExPerT: 有效的个性化长文本生成评估与解释 

**Authors**: Alireza Salemi, Julian Killingback, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2501.14956)  

**Abstract**: Evaluating personalized text generated by large language models (LLMs) is challenging, as only the LLM user, i.e., prompt author, can reliably assess the output, but re-engaging the same individuals across studies is infeasible. This paper addresses the challenge of evaluating personalized text generation by introducing ExPerT, an explainable reference-based evaluation framework. ExPerT leverages an LLM to extract atomic aspects and their evidence from the generated and reference texts, match the aspects, and evaluate their alignment based on content and writing style -- two key attributes in personalized text generation. Additionally, ExPerT generates detailed, fine-grained explanations for every step of the evaluation process, enhancing transparency and interpretability. Our experiments demonstrate that ExPerT achieves a 7.2% relative improvement in alignment with human judgments compared to the state-of-the-art text generation evaluation methods. Furthermore, human evaluators rated the usability of ExPerT's explanations at 4.7 out of 5, highlighting its effectiveness in making evaluation decisions more interpretable. 

**Abstract (ZH)**: 评估由大规模语言模型（LLMs）生成的个性化文本具有挑战性，因为只有LLM的用户，即提示作者，才能可靠地评估输出，但重新在不同研究中涉及同一个体是不现实的。本文通过引入ExPerT，一种基于解释的参考文本评估框架，来应对个性化文本生成的评估挑战。ExPerT利用LLM从生成文本和参考文本中提取原子层面的方面及其证据，匹配这些方面，并基于内容和写作风格评估它们的对齐情况——这是个性化文本生成的两个关键属性。此外，ExPerT还为评估过程中的每一步生成详细的、细粒度的解释，从而增强透明度和可解释性。我们的实验表明，ExPerT在与最先进的文本生成评估方法相比时，实现了7.2%的相对对齐改进。此外，人类评估者对ExPerT解释的易用性评分达到4.7分（满分5分），突显了其在使评估决策更具可解释性方面的有效性。 

---
# Force-Based Robotic Imitation Learning: A Two-Phase Approach for Construction Assembly Tasks 

**Title (ZH)**: 基于力的机器人模仿学习：一种用于建筑装配任务的两阶段方法 

**Authors**: Hengxu You, Yang Ye, Tianyu Zhou, Jing Du  

**Link**: [PDF](https://arxiv.org/pdf/2501.14942)  

**Abstract**: The drive for efficiency and safety in construction has boosted the role of robotics and automation. However, complex tasks like welding and pipe insertion pose challenges due to their need for precise adaptive force control, which complicates robotic training. This paper proposes a two-phase system to improve robot learning, integrating human-derived force feedback. The first phase captures real-time data from operators using a robot arm linked with a virtual simulator via ROS-Sharp. In the second phase, this feedback is converted into robotic motion instructions, using a generative approach to incorporate force feedback into the learning process. This method's effectiveness is demonstrated through improved task completion times and success rates. The framework simulates realistic force-based interactions, enhancing the training data's quality for precise robotic manipulation in construction tasks. 

**Abstract (ZH)**: 不断提高施工效率与安全性的需求推动了机器人与自动化技术的应用。然而，像焊接和管道插入这类复杂的任务由于需要精确的适应性力控制，给机器人的训练带来了挑战。本文提出了一种两阶段系统，以改进机器人的学习，该系统整合了人类提供的力反馈。第一阶段通过ROS-Sharp将机器人手臂与虚拟模拟器链接，实时采集操作员数据。第二阶段将这些反馈转化为机器人的运动指令，并采用生成性方法将力反馈纳入学习过程。通过提高任务完成时间和成功率，该方法的有效性得到验证。该框架模拟了真实的力交互，从而提高了用于精确施工任务中的机器人操作的训练数据质量。 

---
# CASE-Bench: Context-Aware Safety Evaluation Benchmark for Large Language Models 

**Title (ZH)**: CASE-Bench：面向大型语言模型的上下文感知安全评估基准 

**Authors**: Guangzhi Sun, Xiao Zhan, Shutong Feng, Philip C. Woodland, Jose Such  

**Link**: [PDF](https://arxiv.org/pdf/2501.14940)  

**Abstract**: Aligning large language models (LLMs) with human values is essential for their safe deployment and widespread adoption. Current LLM safety benchmarks often focus solely on the refusal of individual problematic queries, which overlooks the importance of the context where the query occurs and may cause undesired refusal of queries under safe contexts that diminish user experience. Addressing this gap, we introduce CASE-Bench, a Context-Aware Safety Evaluation Benchmark that integrates context into safety assessments of LLMs. CASE-Bench assigns distinct, formally described contexts to categorized queries based on Contextual Integrity theory. Additionally, in contrast to previous studies which mainly rely on majority voting from just a few annotators, we recruited a sufficient number of annotators necessary to ensure the detection of statistically significant differences among the experimental conditions based on power analysis. Our extensive analysis using CASE-Bench on various open-source and commercial LLMs reveals a substantial and significant influence of context on human judgments (p<0.0001 from a z-test), underscoring the necessity of context in safety evaluations. We also identify notable mismatches between human judgments and LLM responses, particularly in commercial models within safe contexts. 

**Abstract (ZH)**: 将大型语言模型（LLM）与人类价值观对齐对于其安全部署和广泛采用至关重要。当前的LLM安全基准通常仅侧重于拒绝个别问题，而忽略了查询发生的上下文的重要性，这可能导致在安全上下文中不必要的拒绝查询，从而损害用户体验。为解决这一问题，我们引入了CASE-Bench，一种基于上下文的认知安全评估基准，该基准将上下文整合到对LLM的安全评估中。CASE-Bench 根据情境完整性理论为分类的查询分配了明确描述的不同情境。此外，不同于以往研究主要依赖少数几名注释员的多数投票，我们根据功效分析招募了足够的注释员，以确保在实验条件下检测出统计学上的显著差异。通过对各种开源和商用大型语言模型的广泛分析，我们发现上下文对人类判断产生了显著影响（p<0.0001，通过Z检验得出），强调了在安全性评估中考虑上下文的必要性。我们还发现，在安全上下文中，人类判断与LLM响应之间存在明显的分歧，尤其是在商业模型中表现更为显著。 

---
# Context-Aware Neural Gradient Mapping for Fine-Grained Instruction Processing 

**Title (ZH)**: 基于上下文的神经梯度映射方法用于细粒度指令处理 

**Authors**: David Boldo, Lily Pemberton, Gabriel Thistledown, Jacob Fairchild, Felix Kowalski  

**Link**: [PDF](https://arxiv.org/pdf/2501.14936)  

**Abstract**: The integration of contextual embeddings into the optimization processes of large language models is an advancement in natural language processing. The Context-Aware Neural Gradient Mapping framework introduces a dynamic gradient adjustment mechanism, incorporating contextual embeddings directly into the optimization process. This approach facilitates real-time parameter adjustments, enhancing task-specific generalization even in the presence of sparse or noisy data inputs. The mathematical foundation of this framework relies on gradient descent modifications, where contextual embeddings are derived from a supplementary neural network trained to map input features to optimal adaptation gradients. By employing differential geometry principles, high-dimensional input dependencies are encoded into low-dimensional gradient manifolds, enabling efficient adaptation without necessitating the retraining of the entire model. Empirical evaluations demonstrate that the proposed framework consistently outperforms baseline models across various metrics, including accuracy, robustness to noise, and computational efficiency. The integration of context-specific embeddings allows for a more complex understanding of language, thereby improving the model's ability to handle diverse linguistic phenomena. Furthermore, the computational efficiency achieved through this method demonstrates its scalability for large-scale language models operating under diverse constraints. 

**Abstract (ZH)**: 将上下文嵌入集成到大规模语言模型的优化过程中是自然语言处理领域的一项进步。《基于上下文感知神经梯度映射的框架》引入了一种动态梯度调整机制，直接将上下文嵌入融入优化过程。该方法通过实现实时参数调整，即使在稀疏或噪声数据输入的情况下也能增强任务特定的泛化能力。该框架的数学基础在于通过梯度下降的修改，从辅助神经网络训练得到上下文嵌入，该网络用于将输入特征映射到最优适应梯度。通过运用微分几何原理，高维输入依赖关系被编码到低维梯度流形中，从而实现高效适应而无需重新训练整个模型。实证评估表明，所提出的框架在各种指标（包括准确率、抗噪性和计算效率）上均优于基准模型。上下文特定的嵌入整合使模型能够更复杂地理解语言，从而提高其处理各种语言现象的能力。此外，通过这种方法实现的计算效率证明其在各种约束条件下可扩展性对于大规模语言模型的应用具有重要意义。 

---
# Temporal Binding Foundation Model for Material Property Recognition via Tactile Sequence Perception 

**Title (ZH)**: 基于时间绑定的基础模型：通过触觉序列感知识别材料属性 

**Authors**: Hengxu You, Tianyu Zhou, Jing Du  

**Link**: [PDF](https://arxiv.org/pdf/2501.14934)  

**Abstract**: Robots engaged in complex manipulation tasks require robust material property recognition to ensure adaptability and precision. Traditionally, visual data has been the primary source for object perception; however, it often proves insufficient in scenarios where visibility is obstructed or detailed observation is needed. This gap highlights the necessity of tactile sensing as a complementary or primary input for material recognition. Tactile data becomes particularly essential in contact-rich, small-scale manipulations where subtle deformations and surface interactions cannot be accurately captured by vision alone. This letter presents a novel approach leveraging a temporal binding foundation model for tactile sequence understanding to enhance material property recognition. By processing tactile sensor data with a temporal focus, the proposed system captures the sequential nature of tactile interactions, similar to human fingertip perception. Additionally, this letter demonstrates that, through tailored and specific design, the foundation model can more effectively capture temporal information embedded in tactile sequences, advancing material property understanding. Experimental results validate the model's capability to capture these temporal patterns, confirming its utility for material property recognition in visually restricted scenarios. This work underscores the necessity of embedding advanced tactile data processing frameworks within robotic systems to achieve truly embodied and responsive manipulation capabilities. 

**Abstract (ZH)**: 参与复杂操作任务的机器人需要具备 robust 的材料属性识别能力，以确保其适应性和精确性。传统上，视觉数据是物体识别的主要来源；然而，在视野受阻或需要进行详细观察的情况下，视觉数据经常不足。这凸显了触觉感知作为补充或主要输入对于材料识别的必要性。在接触密集且规模较小的操作中，触觉数据尤为重要，因为单靠视觉无法准确捕捉细微形变和表面交互。本文提出了一种新的方法，利用时间绑定基础模型来理解和增强材料属性的识别。通过关注触觉传感器数据的时间维度，所提出的系统捕捉了触觉交互的序列性，类似于人类指尖感知。此外，本文证明通过特定设计，基础模型可以更有效地捕捉触觉序列中嵌入的时间信息，从而推进了材料属性的理解。实验结果验证了该模型捕获这些时间模式的能力，证实了其在视觉受限场景下进行材料属性识别的实用性。这项工作强调了在机器人系统中嵌入先进的触觉数据处理框架的重要性，以实现真正具身和响应的操作能力。 

---
# Explaining Categorical Feature Interactions Using Graph Covariance and LLMs 

**Title (ZH)**: 使用图形协方差和大规模语言模型解释分类特征交互 

**Authors**: Cencheng Shen, Darren Edge, Jonathan Larson, Carey E. Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2501.14932)  

**Abstract**: Modern datasets often consist of numerous samples with abundant features and associated timestamps. Analyzing such datasets to uncover underlying events typically requires complex statistical methods and substantial domain expertise. A notable example, and the primary data focus of this paper, is the global synthetic dataset from the Counter Trafficking Data Collaborative (CTDC) -- a global hub of human trafficking data containing over 200,000 anonymized records spanning from 2002 to 2022, with numerous categorical features for each record. In this paper, we propose a fast and scalable method for analyzing and extracting significant categorical feature interactions, and querying large language models (LLMs) to generate data-driven insights that explain these interactions. Our approach begins with a binarization step for categorical features using one-hot encoding, followed by the computation of graph covariance at each time. This graph covariance quantifies temporal changes in dependence structures within categorical data and is established as a consistent dependence measure under the Bernoulli distribution. We use this measure to identify significant feature pairs, such as those with the most frequent trends over time or those exhibiting sudden spikes in dependence at specific moments. These extracted feature pairs, along with their timestamps, are subsequently passed to an LLM tasked with generating potential explanations of the underlying events driving these dependence changes. The effectiveness of our method is demonstrated through extensive simulations, and its application to the CTDC dataset reveals meaningful feature pairs and potential data stories underlying the observed feature interactions. 

**Abstract (ZH)**: 现代数据集通常包含大量样本和丰富的特征以及相关的时戳。分析这些数据集以发现潜在事件通常需要复杂统计方法以及大量领域的专业知识。一个典型例子，也是本文的主要数据关注点，是来自反人口贩卖数据协作组织（CTDC）的全球合成数据集——这是一个全球的人口贩卖数据枢纽，包含了从2002年到2022年超过20万条匿名记录，每条记录还包含了多个分类特征。本文提出了一种快速和可扩展的方法，用于分析和提取重要的分类特征相互作用，并利用大型语言模型（LLMs）生成数据驱动的见解，以解释这些相互作用。我们的方法首先使用一位编码对分类特征进行二值化处理，然后在每个时间点上计算图协方差。这种图协方差量化了分类数据中依赖结构的时态变化，并在伯努利分布下被证明是一致的依赖性度量。我们使用这一度量来识别显著的特征对，例如随着时间最频繁的趋势对或在特定时刻突然表现出依赖性突增的趋势对。这些提取的特征对及其时间戳随后被传递给负责生成这些依赖性变化背后事件潜在解释的LLM。我们通过广泛的模拟验证了该方法的有效性，并将其应用于CTDC数据集，揭示了有意义的特征对和潜在的数据故事，这些故事解释了观察到的特征相互作用背后的原因。 

---
# Motion-enhancement to Echocardiography Segmentation via Inserting a Temporal Attention Module: An Efficient, Adaptable, and Scalable Approach 

**Title (ZH)**: 通过插入时间注意力模块提升心脏超声分割的动效增强方法：一种高效、可适应且可扩展的方法 

**Authors**: Md. Kamrul Hasan, Guang Yang, Choon Hwai Yap  

**Link**: [PDF](https://arxiv.org/pdf/2501.14929)  

**Abstract**: Cardiac anatomy segmentation is essential for clinical assessment of cardiac function and disease diagnosis to inform treatment and intervention. In performing segmentation, deep learning (DL) algorithms improved accuracy significantly compared to traditional image processing approaches. More recently, studies showed that enhancing DL segmentation with motion information can further improve it. A range of methods for injecting motion information has been proposed, but many of them increase the dimensionality of input images (which is computationally expensive) or have not used an optimal method to insert motion information, such as non-DL registration, non-attention-based networks or single-headed attention. Here, we present a novel, computation-efficient alternative where a novel, scalable temporal attention module (TAM) extracts temporal feature interactions multiple times and where TAM has a multi-headed, KQV projection cross-attention architecture. The module can be seamlessly integrated into a wide range of existing CNN- or Transformer-based networks, providing novel flexibility for inclusion in future implementations. Extensive evaluations on different cardiac datasets, 2D echocardiography (CAMUS), and 3D echocardiography (MITEA) demonstrate the model's effectiveness when integrated into well-established backbone networks like UNet, FCN8s, UNetR, SwinUNetR, and the recent I2UNet. We further find that the optimized TAM-enhanced FCN8s network performs well compared to contemporary alternatives. Our results confirm TAM's robustness, scalability, and generalizability across diverse datasets and backbones. 

**Abstract (ZH)**: 心脏解剖学分割对于临床评估心脏功能和疾病诊断至关重要，以指导治疗和干预措施。在进行分割时，深度学习（DL）算法在准确性方面显著优于传统图像处理方法。最近的研究表明，通过增强DL分割过程中的运动信息，可以进一步提高其准确性。许多方法被提出用于注入运动信息，但其中许多方法增加了输入图像的维度（这在计算上非常昂贵），或者未使用最优方法插入运动信息，如非DL对齐、非注意力机制网络或单头注意力机制。在此，我们提出一种新颖而计算高效的替代方案，其中引入了一种新型可扩展的时间注意力模块（TAM），该模块多次提取时间特征交互，并且TAM具有多头、KQV投影交叉注意力架构。该模块可以无缝集成到各种现有的CNN或Transformer网络中，为未来的实现提供了新的灵活性。对不同心脏数据集（包括2D和3D心脏超声图像的CAMUS和MITEA数据集）的广泛评估表明，当该模型集成到成熟的骨干网络（如UNet、FCN8s、UNetR、SwinUNetR和最近的I2UNet）中时，其有效性得到了验证。此外，我们发现优化的TAM增强的FCN8s网络在当代替代方案中表现良好。我们的结果证实了TAM的鲁棒性、可扩展性和跨不同数据集和骨干网络的泛化能力。 

---
# Decision Making in Changing Environments: Robustness, Query-Based Learning, and Differential Privacy 

**Title (ZH)**: 在变化环境中决策：鲁棒性、查询导向学习与差分隐私 

**Authors**: Fan Chen, Alexander Rakhlin  

**Link**: [PDF](https://arxiv.org/pdf/2501.14928)  

**Abstract**: We study the problem of interactive decision making in which the underlying environment changes over time subject to given constraints. We propose a framework, which we call \textit{hybrid Decision Making with Structured Observations} (hybrid DMSO), that provides an interpolation between the stochastic and adversarial settings of decision making. Within this framework, we can analyze local differentially private (LDP) decision making, query-based learning (in particular, SQ learning), and robust and smooth decision making under the same umbrella, deriving upper and lower bounds based on variants of the Decision-Estimation Coefficient (DEC). We further establish strong connections between the DEC's behavior, the SQ dimension, local minimax complexity, learnability, and joint differential privacy. To showcase the framework's power, we provide new results for contextual bandits under the LDP constraint. 

**Abstract (ZH)**: 我们研究了在给定约束条件下，环境随时间发生变化的交互决策问题。我们提出了一种框架，称为**结构化观测下的混合决策制定（Hybrid Decision Making with Structured Observations，简称hybrid DMSO）**，该框架在随机性和对抗性决策制定设置之间提供了一种插值。在此框架下，我们可以对局部差分隐私（Local Differential Privacy，简称LDP）决策制定、基于查询的学习（特别是样本查询学习，简称SQ学习）、以及在鲁棒性和平滑性方面的决策制定进行统一分析，基于决策估计系数（Decision-Estimation Coefficient，简称DEC）的不同变体得出上界和下界。我们进一步建立了DEC行为、SQ维数、局部最小最大复杂度、可学习性和联合差分隐私之间的强大联系。为了展示该框架的力量，我们在LDP约束条件下提供了新的上下文臂博弈结果。 

---
# Feasible Learning 

**Title (ZH)**: 可行学习 

**Authors**: Juan Ramirez, Ignacio Hounie, Juan Elenter, Jose Gallego-Posada, Meraj Hashemizadeh, Alejandro Ribeiro, Simon Lacoste-Julien  

**Link**: [PDF](https://arxiv.org/pdf/2501.14912)  

**Abstract**: We introduce Feasible Learning (FL), a sample-centric learning paradigm where models are trained by solving a feasibility problem that bounds the loss for each training sample. In contrast to the ubiquitous Empirical Risk Minimization (ERM) framework, which optimizes for average performance, FL demands satisfactory performance on every individual data point. Since any model that meets the prescribed performance threshold is a valid FL solution, the choice of optimization algorithm and its dynamics play a crucial role in shaping the properties of the resulting solutions. In particular, we study a primal-dual approach which dynamically re-weights the importance of each sample during training. To address the challenge of setting a meaningful threshold in practice, we introduce a relaxation of FL that incorporates slack variables of minimal norm. Our empirical analysis, spanning image classification, age regression, and preference optimization in large language models, demonstrates that models trained via FL can learn from data while displaying improved tail behavior compared to ERM, with only a marginal impact on average performance. 

**Abstract (ZH)**: 我们将介绍可行学习（Feasible Learning，FL），这是一种以样本为中心的学习范式，其中模型通过解决一个使每个训练样本损失受限的可行性问题进行训练。与广泛使用的经验风险最小化（Empirical Risk Minimization, ERM）框架不同，ERM 优化的是整体性能，而 FL 要求每个单个数据点都有令人满意的性能。由于任何满足规定性能门槛的模型均可被视为 FL 的有效解决方案，因此优化算法的选择及其动态特性在决定最终解决方案的性质方面起着关键作用。特别是，我们将研究一种主对偶方法，在该方法中，优化过程中会动态调整每个样本的重要性。为了解决实践中设置有意义的阈值的挑战，我们引入了一种对FL的松弛形式，该形式结合了最小范数的松弛变量。我们的实证分析涵盖了图像分类、年龄回归以及大规模语言模型中的偏好优化等应用，表明通过FL训练的模型可以在保持平均性能几乎没有影响的情况下，从数据中学习并展示出改进的尾部行为。 

---
# Noise-conditioned Energy-based Annealed Rewards (NEAR): A Generative Framework for Imitation Learning from Observation 

**Title (ZH)**: 基于观测的模仿学习生成框架：噪声条件化的能量模型退火奖励（NEAR） 

**Authors**: Anish Abhijit Diwan, Julen Urain, Jens Kober, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2501.14856)  

**Abstract**: This paper introduces a new imitation learning framework based on energy-based generative models capable of learning complex, physics-dependent, robot motion policies through state-only expert motion trajectories. Our algorithm, called Noise-conditioned Energy-based Annealed Rewards (NEAR), constructs several perturbed versions of the expert's motion data distribution and learns smooth, and well-defined representations of the data distribution's energy function using denoising score matching. We propose to use these learnt energy functions as reward functions to learn imitation policies via reinforcement learning. We also present a strategy to gradually switch between the learnt energy functions, ensuring that the learnt rewards are always well-defined in the manifold of policy-generated samples. We evaluate our algorithm on complex humanoid tasks such as locomotion and martial arts and compare it with state-only adversarial imitation learning algorithms like Adversarial Motion Priors (AMP). Our framework sidesteps the optimisation challenges of adversarial imitation learning techniques and produces results comparable to AMP in several quantitative metrics across multiple imitation settings. 

**Abstract (ZH)**: 本文介绍了一种基于能量生成模型的新模仿学习框架，能够通过状态唯一专家运动轨迹学习到复杂的、与物理相关的机器人运动策略。我们的算法称为噪声条件下的能量退火奖励（NEAR），通过构造专家运动数据分布的多种扰动版本，并利用去噪分数匹配学习到平滑且定义良好的数据分布能量函数的表示。我们提出使用这些学习到的能量函数作为奖励函数，通过强化学习学习模仿策略。此外，我们提出了一种逐步转换学习到的能量函数的策略，确保在策略生成样本的流形中学习到的奖励始终是有良好定义的。我们在复杂的类人任务，如行走和武术动作中评估了该算法，并将其与仅基于状态的对抗模仿学习算法（如对抗运动先验算法 AMP）进行比较。我们的框架克服了对抗模仿学习技术的优化挑战，多项定量指标上生成了与 AMP 相当的结果，在多个模仿设置中具有竞争力。 

---
# JustLogic: A Comprehensive Benchmark for Evaluating Deductive Reasoning in Large Language Models 

**Title (ZH)**: JustLogic：评估大型语言模型演绎推理能力的综合性基准 

**Authors**: Michael K. Chen, Xikun Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14851)  

**Abstract**: Logical reasoning is a critical component of Large Language Models (LLMs), and substantial research efforts in recent years have aimed to enhance their deductive reasoning capabilities. However, existing deductive reasoning benchmarks, which are crucial for evaluating and advancing LLMs, are inadequate due to their lack of task complexity, presence of prior knowledge as a confounder, and superficial error analysis. To address these deficiencies, we introduce JustLogic, a synthetically generated deductive reasoning benchmark designed for rigorous evaluation of LLMs. JustLogic is (i) highly complex, capable of generating a diverse range of linguistic patterns, vocabulary, and argument structures; (ii) prior knowledge independent, eliminating the advantage of models possessing prior knowledge and ensuring that only deductive reasoning is used to answer questions; and (iii) capable of in-depth error analysis on the heterogeneous effects of reasoning depth and argument form on model accuracy. Our experimental results on JustLogic reveal that most state-of-the-art (SOTA) LLMs perform significantly worse than the human average, demonstrating substantial room for model improvement. All code and data are available at this https URL 

**Abstract (ZH)**: 逻辑推理是大型语言模型（LLMs）的一个关键组成部分，近年来的研究重点在于增强其演绎推理能力。然而，现有的演绎推理基准由于缺乏任务复杂性、存在先验知识干扰以及浅层次的错误分析，尚不足以评估和推动LLM的发展。为了解决这些问题，我们提出了JustLogic，这是一种专门设计用于严格评估LLM的合成演绎推理基准。JustLogic具有以下特点：（i）高度复杂，能够生成多样化的语言模式、词汇和论据结构；（ii）独立于先验知识，消除了模型拥有先验知识所带来的优势，确保仅通过演绎推理来回答问题；（iii）能够进行深入的错误分析，探讨推理深度和论据形式对模型准确率的异质性影响。我们的实验结果表明，大多数最先进（SOTA）的LLM在其演绎推理方面的性能远低于人类平均水平，这表明模型仍有很大的改进空间。所有代码和数据均可通过以下网址获取：[链接] 

---
# On the locality bias and results in the Long Range Arena 

**Title (ZH)**: 关于长-range arena中的局部偏见及其结果 

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2501.14850)  

**Abstract**: The Long Range Arena (LRA) benchmark was designed to evaluate the performance of Transformer improvements and alternatives in long-range dependency modeling tasks. The Transformer and its main variants performed poorly on this benchmark, and a new series of architectures such as State Space Models (SSMs) gained some traction, greatly outperforming Transformers in the LRA. Recent work has shown that with a denoising pre-training phase, Transformers can achieve competitive results in the LRA with these new architectures. In this work, we discuss and explain the superiority of architectures such as MEGA and SSMs in the Long Range Arena, as well as the recent improvement in the results of Transformers, pointing to the positional and local nature of the tasks. We show that while the LRA is a benchmark for long-range dependency modeling, in reality most of the performance comes from short-range dependencies. Using training techniques to mitigate data inefficiency, Transformers are able to reach state-of-the-art performance with proper positional encoding. In addition, with the same techniques, we were able to remove all restrictions from SSM convolutional kernels and learn fully parameterized convolutions without decreasing performance, suggesting that the design choices behind SSMs simply added inductive biases and learning efficiency for these particular tasks. Our insights indicate that LRA results should be interpreted with caution and call for a redesign of the benchmark. 

**Abstract (ZH)**: 《长范围竞技场（LRA）》基准设计用于评估Transformer改进和替代方案在长范围依赖建模任务中的性能。在这一基准中，Transformer及其主要变体表现不佳，而新型架构如状态空间模型（SSMs）获得了关注，并在LRA中显著超越了Transformer。近期研究表明，通过去噪预训练阶段，Transformer可以与这些新型架构在LRA中获得竞争性的结果。在本研究中，我们将讨论和解释MEGA和SSMs等架构在《长范围竞技场》中的优越性，以及Transformer性能最近的改进，这表明任务具有位置性和局部性特点。我们发现，虽然LRA是一个长范围依赖建模的基准，但在实际中，大部分性能来源于短范围依赖。采用减轻数据效率低下的训练技术，通过适当的位移编码，Transformer能够达到最先进的性能。此外，使用相同的技术，我们能够去除SSM卷积核的所有限制，并且能够学习完全参数化的卷积而不影响性能，这表明SSM的设计选择实际上是为其特定任务添加了归纳偏置和学习效率。我们的见解表明，LRA的结果应谨慎解读，并呼吁重新设计该基准。 

---
# Wormhole Memory: A Rubik's Cube for Cross-Dialogue Retrieval 

**Title (ZH)**: wormhole memory：对话跨轮次检索的魔方 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14846)  

**Abstract**: In view of the gap in the current large language model in sharing memory across dialogues, this research proposes a wormhole memory module (WMM) to realize memory as a Rubik's cube that can be arbitrarily retrieved between different dialogues. Through simulation experiments, the researcher built an experimental framework based on the Python environment and used setting memory barriers to simulate the current situation where memories between LLMs dialogues are difficult to share. The CoQA development data set was imported into the experiment, and the feasibility of its cross-dialogue memory retrieval function was verified for WMM's nonlinear indexing and dynamic retrieval, and a comparative analysis was conducted with the capabilities of Titans and MemGPT memory modules. Experimental results show that WMM demonstrated the ability to retrieve memory across dialogues and the stability of quantitative indicators in eight experiments. It contributes new technical approaches to the optimization of memory management of LLMs and provides experience for the practical application in the future. 

**Abstract (ZH)**: 鉴于当前大规模语言模型在跨对话共享记忆方面的不足，本研究提出了一种 wormhole 记忆模块（WMM），以实现记忆可以像魔方一样在不同对话间任意检索。通过模拟实验，研究者基于 Python 环境构建了一个实验框架，并通过设置记忆屏障来模拟当前大规模语言模型（LLM）之间对话记忆难以共享的情况。CoQA 开发数据集被导入实验，验证了 WMM 的非线性索引和动态检索功能，并与 Titan 和 MemGPT 记忆模块的能力进行了比较分析。实验结果显示，WMM 在八次实验中展示了跨对话检索记忆的能力，并且其定量指标的稳定性也得到了验证。WMM 为 LLM 记忆管理的优化提供了新的技术方法，并为未来的实际应用提供了经验借鉴。 

---
# Unmasking Conversational Bias in AI Multiagent Systems 

**Title (ZH)**: 揭示AI多智能体系统中的对话偏见 

**Authors**: Erica Coppolillo, Giuseppe Manco, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2501.14844)  

**Abstract**: Detecting biases in the outputs produced by generative models is essential to reduce the potential risks associated with their application in critical settings. However, the majority of existing methodologies for identifying biases in generated text consider the models in isolation and neglect their contextual applications. Specifically, the biases that may arise in multi-agent systems involving generative models remain under-researched. To address this gap, we present a framework designed to quantify biases within multi-agent systems of conversational Large Language Models (LLMs). Our approach involves simulating small echo chambers, where pairs of LLMs, initialized with aligned perspectives on a polarizing topic, engage in discussions. Contrary to expectations, we observe significant shifts in the stance expressed in the generated messages, particularly within echo chambers where all agents initially express conservative viewpoints, in line with the well-documented political bias of many LLMs toward liberal positions. Crucially, the bias observed in the echo-chamber experiment remains undetected by current state-of-the-art bias detection methods that rely on questionnaires. This highlights a critical need for the development of a more sophisticated toolkit for bias detection and mitigation for AI multi-agent systems. The code to perform the experiments is publicly available at this https URL. 

**Abstract (ZH)**: 检测生成模型输出中的偏见对于减少其在关键应用场景中的潜在风险至关重要。然而，现有的多数偏见识别方法主要将模型孤立考虑，并未充分考虑到其背景环境的应用。特别是涉及生成模型的多智能体系统中的偏见研究仍较为不足。为填补这一空白，我们提出了一种框架，旨在量化多智能体系统中的对话型大语言模型（LLMs）中的偏见。该方法通过模拟小回声室来实现，其中，对一对初始观点一致（针对一个具有争议性的话题）的LLMs，让它们进行讨论。出乎意料的是，在所有智能体最初都保持保守观点的回声室中，生成的消息表达的态度发生了显著变化，这与许多LLMs普遍存在的政治偏向（倾向于支持自由派观点）相符。最关键的是，当前最先进的偏见检测方法依赖问卷调查，未能识别出回声室实验中发现的偏见。这突显了开发更复杂的工具包以检测和减轻AI多智能体系统中偏见的必要性。实验代码可在此处公开访问：[](https://)。 

---
# An Ensemble Model with Attention Based Mechanism for Image Captioning 

**Title (ZH)**: 基于注意力机制的ensemble模型在图片字幕生成中的应用 

**Authors**: Israa Al Badarneh, Bassam Hammo, Omar Al-Kadi  

**Link**: [PDF](https://arxiv.org/pdf/2501.14828)  

**Abstract**: Image captioning creates informative text from an input image by creating a relationship between the words and the actual content of an image. Recently, deep learning models that utilize transformers have been the most successful in automatically generating image captions. The capabilities of transformer networks have led to notable progress in several activities related to vision. In this paper, we thoroughly examine transformer models, emphasizing the critical role that attention mechanisms play. The proposed model uses a transformer encoder-decoder architecture to create textual captions and a deep learning convolutional neural network to extract features from the images. To create the captions, we present a novel ensemble learning framework that improves the richness of the generated captions by utilizing several deep neural network architectures based on a voting mechanism that chooses the caption with the highest bilingual evaluation understudy (BLEU) score. The proposed model was evaluated using publicly available datasets. Using the Flickr8K dataset, the proposed model achieved the highest BLEU-[1-3] scores with rates of 0.728, 0.495, and 0.323, respectively. The suggested model outperformed the latest methods in Flickr30k datasets, determined by BLEU-[1-4] scores with rates of 0.798, 0.561, 0.387, and 0.269, respectively. The model efficacy was also obtained by the Semantic propositional image caption evaluation (SPICE) metric with a scoring rate of 0.164 for the Flicker8k dataset and 0.387 for the Flicker30k. Finally, ensemble learning significantly advances the process of image captioning and, hence, can be leveraged in various applications across different domains. 

**Abstract (ZH)**: 图像字幕通过在词语与图像实际内容之间建立关系来从输入图像生成具有信息性的文本。近年来，利用变压器的深度学习模型在自动生成图像字幕方面表现最佳。变压器网络的能力在与视觉相关的多种活动中取得了显著进展。在本文中，我们深入探讨了变压器模型，并强调了注意力机制的关键作用。所提出的模型采用变压器编码器-解码器结构生成文本字幕，同时使用深度学习卷积神经网络从图像中提取特征。为了生成字幕，我们提出了一种新颖的集成学习框架，通过利用基于投票机制的多种深度神经网络架构来提高生成字幕的丰富性，通过选择具有最高双语评价 understudy（BLEU）分数的字幕。所提出的模型通过公开可用的数据集进行了评估。使用 Flickr8K 数据集，所提出的模型分别获得了 BLEU-[1-3] 分数，分别为 0.728、0.495 和 0.323。在 Flickr30k 数据集上，按 BLEU-[1-4] 分数，所建议的模型也优于最新方法，分别为 0.798、0.561、0.387 和 0.269。该模型的有效性还通过语义命题图像字幕评估标准（SPICE）度量获得，Flicker8k 数据集的得分为 0.164，Flicker30k 数据集的得分为 0.387。最后，集成学习显著提高了图像字幕生成的过程，因此可以在不同领域的各类应用中发挥作用。 

---
# Multi-Modality Transformer for E-Commerce: Inferring User Purchase Intention to Bridge the Query-Product Gap 

**Title (ZH)**: 电商领域多模态变压器：推断用户购买意向以弥合查询与产品之间的差距 

**Authors**: Srivatsa Mallapragada, Ying Xie, Varsha Rani Chawan, Zeyad Hailat, Yuanbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14826)  

**Abstract**: E-commerce click-stream data and product catalogs offer critical user behavior insights and product knowledge. This paper propose a multi-modal transformer termed as PINCER, that leverages the above data sources to transform initial user queries into pseudo-product representations. By tapping into these external data sources, our model can infer users' potential purchase intent from their limited queries and capture query relevant product features. We demonstrate our model's superior performance over state-of-the-art alternatives on e-commerce online retrieval in both controlled and real-world experiments. Our ablation studies confirm that the proposed transformer architecture and integrated learning strategies enable the mining of key data sources to infer purchase intent, extract product features, and enhance the transformation pipeline from queries to more accurate pseudo-product representations. 

**Abstract (ZH)**: 电商平台点击流数据和产品目录提供了关键的用户行为洞察和产品知识。本文提出了一种多模态变压器模型，称为PINCER，该模型利用上述数据源将初始用户查询转化为伪产品表示。通过利用这些外部数据源，我们的模型可以从用户有限的查询中推断出潜在的购买意图，并捕捉查询相关的商品特征。我们在电子商务在线检索的控制实验和实际场景中展示了该模型相较于最新替代方案的优越性能。我们的消融研究证实，所提出的变压器架构和集成学习策略能够挖掘关键数据源以推断购买意图、提取产品特征，并增强从查询到更准确的伪产品表示的转换管道。 

---
# Quantifying Energy and Cost Benefits of Hybrid Edge Cloud: Analysis of Traditional and Agentic Workloads 

**Title (ZH)**: 混合边缘云的能量和成本效益量化：传统和代理工作负载的分析 

**Authors**: Siavash Alamouti  

**Link**: [PDF](https://arxiv.org/pdf/2501.14823)  

**Abstract**: This paper examines the workload distribution challenges in centralized cloud systems and demonstrates how Hybrid Edge Cloud (HEC) [1] mitigates these inefficiencies. Workloads in cloud environments often follow a Pareto distribution, where a small percentage of tasks consume most resources, leading to bottlenecks and energy inefficiencies. By analyzing both traditional workloads reflective of typical IoT and smart device usage and agentic workloads, such as those generated by AI agents, robotics, and autonomous systems, this study quantifies the energy and cost savings enabled by HEC. Our findings reveal that HEC achieves energy savings of up to 75% and cost reductions exceeding 80%, even in resource-intensive agentic scenarios. These results highlight the critical role of HEC in enabling scalable, cost-effective, and sustainable computing for the next generation of intelligent systems. 

**Abstract (ZH)**: 本文探讨了集中式云系统中的工作负载分布挑战，并展示了混合边缘云（HEC）[1] 如何缓解这些不效率。在云环境中，工作负载通常遵循帕累托分布，其中一小部分任务消耗了大部分资源，导致瓶颈和能源效率低下。通过分析反映典型物联网和智能设备使用情况的传统工作负载以及由人工智能代理、机器人和自主系统生成的具身工作负载，本研究量化了HEC带来的能源和成本节约。我们的研究发现，在资源密集型的具身场景中，HEC能够实现高达75%的能源节约和超过80%的成本减少。这些结果强调了HEC在为下一代智能系统提供可扩展、成本效益高和可持续计算方面发挥的关键作用。 

---
# Controlling Ensemble Variance in Diffusion Models: An Application for Reanalyses Downscaling 

**Title (ZH)**: 控制扩散模型中的集成方差：一种再分析降尺度应用 

**Authors**: Fabio Merizzi, Davide Evangelista, Harilaos Loukos  

**Link**: [PDF](https://arxiv.org/pdf/2501.14822)  

**Abstract**: In recent years, diffusion models have emerged as powerful tools for generating ensemble members in meteorology. In this work, we demonstrate that a Denoising Diffusion Implicit Model (DDIM) can effectively control ensemble variance by varying the number of diffusion steps. Introducing a theoretical framework, we relate diffusion steps to the variance expressed by the reverse diffusion process. Focusing on reanalysis downscaling, we propose an ensemble diffusion model for the full ERA5-to-CERRA domain, generating variance-calibrated ensemble members for wind speed at full spatial and temporal resolution. Our method aligns global mean variance with a reference ensemble dataset and ensures spatial variance is distributed in accordance with observed meteorological variability. Additionally, we address the lack of ensemble information in the CARRA dataset, showcasing the utility of our approach for efficient, high-resolution ensemble generation. 

**Abstract (ZH)**: 近年来，扩散模型已成为气象学中生成集合成员的强大工具。本文展示了一种去噪扩散隐式模型（DDIM）可以通过调整扩散步数有效控制集合方差的能力。我们引入了一个理论框架，将扩散步数与反向扩散过程中表达的方差联系起来。针对再分析下标化问题，我们提出了一种适用于完整ERA5至CERRA区域的集合扩散模型，以生成风速的高分辨率方差校准集合成员。该方法使全球平均方差与参考集合数据集相一致，并确保方差在空间上的分布符合观测到的气象变异性。此外，我们还解决了CARRA数据集中缺乏集合信息的问题，展示了我们的方法在高效生成高分辨率集合方面的实用性。 

---
# Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models 

**Title (ZH)**: Eagle 2：从零构建前沿视觉语言模型的后训练数据策略 

**Authors**: Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Yilin Zhao, Subhashree Radhakrishnan, Nadine Chang, Karan Sapra, Amala Sanjay Deshmukh, Tuomas Rintamaki, Matthieu Le, Ilia Karmanov, Lukas Voegtle, Philipp Fischer, De-An Huang, Timo Roman, Tong Lu, Jose M. Alvarez, Bryan Catanzaro, Jan Kautz, Andrew Tao, Guilin Liu, Zhiding Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14818)  

**Abstract**: Recently, promising progress has been made by open-source vision-language models (VLMs) in bringing their capabilities closer to those of proprietary frontier models. However, most open-source models only publish their final model weights, leaving the critical details of data strategies and implementation largely opaque. In this work, we address VLM post-training from a data-centric perspective, showing the key role of data strategy in developing frontier VLMs. By studying and building our post-training data strategy from scratch, we share detailed insights into the development processes, aiming to benefit the development of competitive models for the open-source community. Our introduced data strategy, together with training recipes and model design, leads to a family of performant VLMs named Eagle2. Specifically, Eagle2-9B achieves state-of-the-art results across various multimodal benchmarks, matching certain competitive models with up to 70B parameters. 

**Abstract (ZH)**: 近年来，开源视觉-语言模型（VLMs）在使其能力接近专有先进模型方面取得了令人鼓舞的进展。然而，大多数开源模型仅发布其最终模型权重，而数据策略和实现的许多关键细节则相对不透明。在本项工作中，我们从数据为中心的角度研究了VLM的后训练方法，展示了数据策略在开发先进VLM中的关键作用。通过从头研究和构建后训练的数据策略，我们分享了开发过程中的详细见解，旨在为开源社区开发具有竞争力的模型提供帮助。我们引入的数据策略，结合训练方法和模型设计，产生了一系列表现优异的VLMs，命名为Eagle2。具体而言，Eagle2-9B在各种多模态基准测试中取得了最先进的结果，与包含多达700亿参数的竞争模型相当。 

---
# A VM-HDL Co-Simulation Framework for Systems with PCIe-Connected FPGAs 

**Title (ZH)**: 基于PCIe连接FPGA的系统的一种VM-HDL联合仿真实现框架 

**Authors**: Shenghsun Cho, Mrunal Patel, Basavaraj Kaladagi, Han Chen, Tapti Palit, Michael Ferdman, Peter Milder  

**Link**: [PDF](https://arxiv.org/pdf/2501.14815)  

**Abstract**: PCIe-connected FPGAs are gaining popularity as an accelerator technology in data centers. However, it is challenging to jointly develop and debug host software and FPGA hardware. Changes to the hardware design require a time-consuming FPGA synthesis process, and modification to the software, especially the operating system and device drivers, can frequently cause the system to hang, without providing enough information for debugging. The combination of these problems results in long debug iterations and a slow development process. To overcome these problems, we designed a VM-HDL co-simulation framework, which is capable of running the same software, operating system, and hardware designs as the target physical system, while providing full visibility and significantly shorter debug iterations. 

**Abstract (ZH)**: PCIe连接的FPGA在数据中心中作为加速器技术越来越受欢迎。然而，联合开发和调试宿主机软件和FPGA硬件颇具挑战性。对硬件设计的任何更改都需要耗费大量时间进行FPGA综合过程，且软件（尤其是操作系统和设备驱动程序）的任何改动都可能导致系统挂起，而提供的调试信息却很少。这些问题的结合导致了长时间的调试迭代和缓慢的开发进程。为了解决这些问题，我们设计了一个虚拟机-硬件描述语言（VM-HDL）联合仿真框架，该框架能够在目标物理系统运行相同的软件、操作系统和硬件设计的同时，提供全程可见性和显著缩短的调试迭代时间。 

---
# Towards Foundation Models: Evaluation of Geoscience Artificial Intelligence with Uncertainty 

**Title (ZH)**: 面向基础模型：地质科学人工智能不确定性评估 

**Authors**: Samuel Myren, Nidhi Parikh, Rosalyn Rael, Garrison Flynn, Dave Higdon, Emily Casleton  

**Link**: [PDF](https://arxiv.org/pdf/2501.14809)  

**Abstract**: Artificial intelligence (AI) has transformed the geoscience community with deep learning models (DLMs) that are trained to complete specific tasks within workflows. This success has led to the development of geoscience foundation models (FMs), which promise to accomplish multiple tasks within a workflow or replace the workflow altogether. However, lack of robust evaluation frameworks, even for traditional DLMs, leaves the geoscience community ill prepared for the inevitable adoption of FMs. We address this gap by designing an evaluation framework that jointly incorporates three crucial aspects to current DLMs and future FMs: performance uncertainty, learning efficiency, and overlapping training-test data splits. To target the three aspects, we meticulously construct the training, validation, and test splits using clustering methods tailored to geoscience data and enact an expansive training design to segregate performance uncertainty arising from stochastic training processes and random data sampling. The framework's ability to guard against misleading declarations of model superiority is demonstrated through evaluation of PhaseNet, a popular seismic phase picking DLM, under 3 training approaches. Furthermore, we show how the performance gains due to overlapping training-test data can lead to biased FM evaluation. Our framework helps practitioners choose the best model for their problem and set performance expectations by explicitly analyzing model performance at varying budgets of training data. 

**Abstract (ZH)**: 人工 intelligence（AI）通过深度学习模型（DLMs）在地质科学社区中取得了革命性的变革，这些模型经过训练以在工作流中完成特定任务。这些成功的案例推动了地质科学基础模型（FMs）的发展，这些模型有望在同一工作流中完成多项任务甚至完全替代整个工作流。然而，即使是对于传统的DLMs，缺乏坚固的评估框架也让地质科学社区在不可避免地采用FMs的过程中显得准备不足。为了弥合这一缺口，我们设计了一个评估框架，该框架综合考虑了当前DLMs和未来FMs的三个关键方面：性能不确定性、学习效率以及重叠的训练-测试数据分割。为了针对这三个方面，我们精心构建了训练、验证和测试分割，采用针对地质科学数据定制的聚类方法，并采用了广泛的设计方案来隔离由随机训练过程和随机数据采样引起的不同性能不确定性。通过在三种训练方法下评估流行地震相识别DLM——PhaseNet，我们展示了框架如何防止出现关于模型优越性的误导性声明。此外，我们还展示了由于训练-测试数据重叠而导致的性能提升如何导致基础模型评估结果的偏差。我们的框架有助于实践者根据不同的训练数据预算明确分析模型性能，从而选择最适合他们问题的最佳模型并设定合理的性能期望。 

---
# HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators 

**Title (ZH)**: HeteroLLM：通过异构AI加速器在移动SoC平台加速大型语言模型推断 

**Authors**: Le Chen, Dahu Feng, Erhu Feng, Rong Zhao, Yingrui Wang, Yubin Xia, Haibo Chen, Pinjie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14794)  

**Abstract**: With the rapid advancement of artificial intelligence technologies such as ChatGPT, AI agents and video generation,contemporary mobile systems have begun integrating these AI capabilities on local devices to enhance privacy and reduce response latency. To meet the computational demands of AI tasks, current mobile SoCs are equipped with diverse AI accelerators, including GPUs and Neural Processing Units (NPUs). However, there has not been a comprehensive characterization of these heterogeneous processors, and existing designs typically only leverage a single AI accelerator for LLM inference, leading to suboptimal use of computational resources and memory bandwidth. In this paper, we first summarize key performance characteristics of mobile SoC, including heterogeneous processors, unified memory, synchronization, etc. Drawing on these observations, we propose different tensor partition strategies to fulfill the distinct requirements of the prefill and decoding phases. We further design a fast synchronization mechanism that leverages the unified memory address provided by mobile SoCs. By employing these techniques, we present HeteroLLM, the fastest LLM inference engine in mobile devices which supports both layer-level and tensor-level heterogeneous execution. Evaluation results show that HeteroLLM achieves 9.99 and 4.36 performance improvement over other mobile-side LLM inference engines: MLC and MNN. 

**Abstract (ZH)**: 随着ChatGPT、AI代理和视频生成等人工智能技术的迅猛发展，当代移动系统已经开始在其本地设备中整合这些AI能力，以增强隐私保护并减少响应延迟。为满足AI任务的计算需求，当前的移动SoC（片上系统）配备了多种AI加速器，包括GPU和神经处理单元（NPUs）。然而，这些异构处理器尚未进行全面的特征化，现有的设计通常仅利用单一的AI加速器进行大语言模型（LLM）推理，导致计算资源和内存带宽利用率不足。本文首先总结了移动SoC的关键性能特征，包括异构处理器、统一内存、同步等。基于这些观察，我们提出了不同的张量分割策略，以满足预填充和解码阶段的不同需求。此外，我们设计了一种快速同步机制，利用移动SoC提供的统一内存地址。通过采用这些技术，我们提出了HeteroLLM，这是目前移动设备中速度最快的大语言模型推理引擎，支持层级和张量级的异构执行。评估结果显示，HeteroLLM在与其他移动侧的大语言模型推理引擎（如MLC和MNN）进行比较时，分别实现了9.99倍和4.36倍的性能提升。 

---
# Towards Dynamic Neural Communication and Speech Neuroprosthesis Based on Viseme Decoding 

**Title (ZH)**: 基于唇型解码的动态神经通信与语音神经假体研究 

**Authors**: Ji-Ha Park, Seo-Hyun Lee, Soowon Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.14790)  

**Abstract**: Decoding text, speech, or images from human neural signals holds promising potential both as neuroprosthesis for patients and as innovative communication tools for general users. Although neural signals contain various information on speech intentions, movements, and phonetic details, generating informative outputs from them remains challenging, with mostly focusing on decoding short intentions or producing fragmented outputs. In this study, we developed a diffusion model-based framework to decode visual speech intentions from speech-related non-invasive brain signals, to facilitate face-to-face neural communication. We designed an experiment to consolidate various phonemes to train visemes of each phoneme, aiming to learn the representation of corresponding lip formations from neural signals. By decoding visemes from both isolated trials and continuous sentences, we successfully reconstructed coherent lip movements, effectively bridging the gap between brain signals and dynamic visual interfaces. The results highlight the potential of viseme decoding and talking face reconstruction from human neural signals, marking a significant step toward dynamic neural communication systems and speech neuroprosthesis for patients. 

**Abstract (ZH)**: 从人类神经信号解码文本、语音或图像在患者神经假体和普通用户的创新通信工具方面都展现了广阔的可能性。尽管神经信号包含了关于语音意图、动作和音素细节的多种信息，但从这些信号生成具有信息量的输出仍然是一个挑战，大多数研究集中在解码短暂的意图或生成片段化的输出。在本研究中，我们开发了一种基于扩散模型的框架，从与语音相关的非侵入性脑信号中解码视觉语音意图，以促进面对面的神经通信。我们设计了一个实验，将各种音素整合起来训练每个音素的唇形，旨在从神经信号中学习相应唇形的表示。通过从孤立试次和连续句子的解码结果中，我们成功地重建了连贯的唇部运动，有效地填补了脑信号与动态视觉界面之间的差距。研究结果突显了从人类神经信号解码唇形和重构说话语音的潜力，标志着动态神经通信系统和患者语音神经假体发展的重要进展。 

---
# ED-Filter: Dynamic Feature Filtering for Eating Disorder Classification 

**Title (ZH)**: ED-Filter: 动态特征筛选在饮食障碍分类中的应用 

**Authors**: Mehdi Naseriparsa, Suku Sukunesan, Zhen Cai, Osama Alfarraj, Amr Tolba, Saba Fathi Rabooki, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2501.14785)  

**Abstract**: Eating disorders (ED) are critical psychiatric problems that have alarmed the mental health community. Mental health professionals are increasingly recognizing the utility of data derived from social media platforms such as Twitter. However, high dimensionality and extensive feature sets of Twitter data present remarkable challenges for ED classification. To overcome these hurdles, we introduce a novel method, an informed branch and bound search technique known as ED-Filter. This strategy significantly improves the drawbacks of conventional feature selection algorithms such as filters and wrappers. ED-Filter iteratively identifies an optimal set of promising features that maximize the eating disorder classification accuracy. In order to adapt to the dynamic nature of Twitter ED data, we enhance the ED-Filter with a hybrid greedy-based deep learning algorithm. This algorithm swiftly identifies sub-optimal features to accommodate the ever-evolving data landscape. Experimental results on Twitter eating disorder data affirm the effectiveness and efficiency of ED-Filter. The method demonstrates significant improvements in classification accuracy and proves its value in eating disorder detection on social media platforms. 

**Abstract (ZH)**: 饮食障碍（ED）是严重的心理卫生问题，引起了心理卫生界的广泛关注。心理健康专业人员越来越多地认识到，可以从推特等社交媒体平台上获取的数据中提取有用信息。然而，推特数据的高维性和广泛性特征带来了显著的分类挑战。为克服这些难题，我们提出了一种新颖的方法——一种名为ED-Filter的有指导分支界限搜索技术。这种策略显著改进了传统特征选择算法（如滤波器和包装器）的不足。ED-Filter通过迭代方式识别出能够最大化饮食障碍分类准确性的最佳特征集。为了适应推特ED数据的动态特性，我们通过引入一种混合贪婪深度学习算法来增强ED-Filter。该算法能够快速识别出次优特征，以适应不断变化的数据环境。在推特饮食障碍数据上的实验结果表明，ED-Filter方法的有效性和效率。该方法在分类准确性方面表现出显著改进，并证明了其在社交媒体平台上的饮食障碍检测方面的价值。 

---
# DeServe: Towards Affordable Offline LLM Inference via Decentralization 

**Title (ZH)**: DeServe：通过去中心化实现可负担的离线大规模语言模型推理 

**Authors**: Linyu Wu, Xiaoyuan Liu, Tianneng Shi, Zhe Ye, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.14784)  

**Abstract**: The rapid growth of generative AI and its integration into everyday workflows have significantly increased the demand for large language model (LLM) inference services. While proprietary models remain popular, recent advancements in open-source LLMs have positioned them as strong contenders. However, deploying these models is often constrained by the high costs and limited availability of GPU resources. In response, this paper presents the design of a decentralized offline serving system for LLM inference. Utilizing idle GPU resources, our proposed system, DeServe, decentralizes access to LLMs at a lower cost. DeServe specifically addresses key challenges in optimizing serving throughput in high-latency network environments. Experiments demonstrate that DeServe achieves a 6.7x-12.6x improvement in throughput over existing serving system baselines in such conditions. 

**Abstract (ZH)**: 生成型AI的快速发展及其融入日常工作流中，显著增加了对大规模语言模型（LLM）推理服务的需求。虽然专有模型仍很流行，但最近开源LLM的进步使它们成为了强有力的竞争对手。然而，部署这些模型往往受限于GPU资源的高成本和有限可用性。针对这一问题，本文提出了一种去中心化的离线服务系统设计，用于LLM推理。利用闲置的GPU资源，我们提出的系统DeServe以更低的成本实现了对LLM的去中心化访问。DeServe特别解决了在高延迟网络环境中优化服务吞吐量的关键挑战。实验表明，在这些条件下，DeServe在吞吐量方面相对于现有的服务系统基线实现了6.7至12.6倍的提升。 

---
# Perspective Chapter: MOOCs in India: Evolution, Innovation, Impact, and Roadmap 

**Title (ZH)**: 视角章：印度的MOOCs：发展、创新、影响与未来 roadmap 

**Authors**: Partha Pratim Das  

**Link**: [PDF](https://arxiv.org/pdf/2501.14780)  

**Abstract**: With the largest population of the world and one of the highest enrolments in higher education, India needs efficient and effective means to educate its learners. India started focusing on open and digital education in 1980's and its efforts were escalated in 2009 through the NMEICT program of the Government of India. A study by the Government and FICCI in 2014 noted that India cannot meet its educational needs just by capacity building in brick and mortar institutions. It was decided that ongoing MOOCs projects under the umbrella of NMEICT will be further strengthened over its second (2017-21) and third (2021-26) phases. NMEICT now steers NPTEL or SWAYAM (India's MOOCs) and several digital learning projects including Virtual Labs, e-Yantra, Spoken Tutorial, FOSSEE, and National Digital Library on India - the largest digital education library in the world. Further, India embraced its new National Education Policy in 2020 to strongly foster online education. In this chapter, we take a deep look into the evolution of MOOCs in India, its innovations, its current status and impact, and the roadmap for the next decade to address its challenges and grow. AI-powered MOOCs is an emerging opportunity for India to lead MOOCs worldwide. 

**Abstract (ZH)**: 随着世界上人口最多且高等教育入学率较高的国家之一，印度需要高效而有效的教育资源来培养其学习者。印度早在1980年代就开始关注开放与数字化教育，并在2009年通过印度政府的NMEICT项目加大了这方面的努力。印度政府和FICCI在2014年的一项研究指出，仅靠砖瓦建筑学校的教育资源扩建无法满足印度的教育需求。决定加强NMEICT下的持续MOOC项目，分别在第二阶段（2017-2021）和第三阶段（2021-2026）进一步推进。NMEICT现在负责管理NPTEL或SWAYAM（印度的MOOC项目），以及多个数字化学习项目，包括虚拟实验室、e-Yantra、口语教程、FOSSEE和National Digital Library of India，这是全球最大的数字教育资源库。除此之外，印度于2020年采纳了新的国家教育政策，以进一步促进在线教育。在这一章节中，我们将深入探讨印度MOOCs的发展历程、创新、当前状况及其影响，并制定未来十年的战略路线图，以应对挑战并促进其发展。人工智能驱动的MOOCs为印度在全球MOOC领域引领潮流提供了新的机会。 

---
# The Use of Generative Artificial Intelligence for Upper Secondary Mathematics Education Through the Lens of Technology Acceptance 

**Title (ZH)**: 通过技术接受理论视角下生成式人工智能在高中数学教育中的应用研究 

**Authors**: Mika Setälä, Ville Heilala, Pieta Sikström, Tommi Kärkkäinen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14779)  

**Abstract**: This study investigated the students' perceptions of using Generative Artificial Intelligence (GenAI) in upper-secondary mathematics education. Data was collected from Finnish high school students to represent how key constructs of the Technology Acceptance Model (Perceived Usefulness, Perceived Ease of Use, Perceived Enjoyment, and Intention to Use) influence the adoption of AI tools. First, a structural equation model for a comparative study with a prior study was constructed and analyzed. Then, an extended model with the additional construct of Compatibility, which represents the alignment of AI tools with students' educational experiences and needs, was proposed and analyzed. The results demonstrated a strong influence of perceived usefulness on the intention to use GenAI, emphasizing the statistically significant role of perceived enjoyment in determining perceived usefulness and ease of use. The inclusion of compatibility improved the model's explanatory power, particularly in predicting perceived usefulness. This study contributes to a deeper understanding of how AI tools can be integrated into mathematics education and highlights key differences between the Finnish educational context and previous studies based on structural equation modeling. 

**Abstract (ZH)**: 本研究调查了高中生对在中等数学教育中使用生成型人工智能（GenAI）的看法。通过收集来自芬兰高中的数据，本研究探讨了技术接受模型（感知有用性、感知易用性、感知愉悦性和使用意图）中的关键构念如何影响AI工具的采用。首先，构建了一个结构方程模型，与先前的研究进行了比较分析。然后，提出了一个扩展模型，增加了兼容性这一构念，以代表AI工具与学生教育体验和需求的一致性，并进行了分析。研究结果表明，感知有用性对GenAI的使用意图有显著影响，特别强调了感知愉悦性在决定感知有用性和易用性中的统计显著作用。兼容性的纳入增强了模型的解释力，特别是在预测感知有用性方面。本研究为深入了解如何将AI工具整合到数学教育中提供了更深刻的认识，并突显了芬兰教育背景与基于结构方程建模的先前研究之间的关键差异。 

---
# Advancing Trustworthy AI for Sustainable Development: Recommendations for Standardising AI Incident Reporting 

**Title (ZH)**: 促进可信人工智能以实现可持续发展：关于标准化人工智能事故报告的建议 

**Authors**: Avinash Agarwal, Manisha J Nene  

**Link**: [PDF](https://arxiv.org/pdf/2501.14778)  

**Abstract**: The increasing use of AI technologies has led to increasing AI incidents, posing risks and causing harm to individuals, organizations, and society. This study recognizes and addresses the lack of standardized protocols for reliably and comprehensively gathering such incident data crucial for preventing future incidents and developing mitigating strategies. Specifically, this study analyses existing open-access AI-incident databases through a systematic methodology and identifies nine gaps in current AI incident reporting practices. Further, it proposes nine actionable recommendations to enhance standardization efforts to address these gaps. Ensuring the trustworthiness of enabling technologies such as AI is necessary for sustainable digital transformation. Our research promotes the development of standards to prevent future AI incidents and promote trustworthy AI, thus facilitating achieving the UN sustainable development goals. Through international cooperation, stakeholders can unlock the transformative potential of AI, enabling a sustainable and inclusive future for all. 

**Abstract (ZH)**: 随着人工智能技术的广泛应用，人工智能事件的数量也在增加，这些事件对个人、组织和社会造成了风险和损害。本研究认识到了缺乏可靠和全面收集此类事件数据的标准化协议的问题，这些数据对于预防未来事件和制定缓解策略至关重要。具体而言，本研究通过系统的方法分析了现有的开放访问的人工智能事件数据库，并识别出了当前人工智能事件报告实践中存在的九个缺口。进一步地，本研究提出了九项可操作的建议，以增强标准化努力来解决这些问题。确保如人工智能等使能技术的信任度对于可持续数字化转型至关重要。我们的研究促进了制定标准以预防未来的人工智能事件并促进可信的人工智能的发展，从而有助于实现联合国可持续发展目标。通过国际协作，利益相关者可以释放人工智能的变革潜力，为所有人实现可持续和包容的未来。 

---
# Enhancing Supply Chain Resilience with Metaverse and ChatGPT Technologies 

**Title (ZH)**: 使用元宇宙和ChatGPT技术增强供应链韧性 

**Authors**: Oumaima Sarhir  

**Link**: [PDF](https://arxiv.org/pdf/2501.14777)  

**Abstract**: Global supply lines have been severely disrupted by the COVID-19 epidemic and the conflict between Russia and Ukraine, which has sharply increased the price of commodities and generated inflation. These incidents highlight how critical it is to improve supply chain resilience (SCRES) in order to fend off unforeseen setbacks. Controlling both internal and external interruptions, such as transportation problems brought on by natural catastrophes and wars, is the responsibility of SCRES. Enhancing resilience in supply chains requires accurate and timely information transfer.
Promising answers to these problems can be found in the Metaverse and ChatGPT, two new digital technologies. The Metaverse may imitate real-world situations and offer dynamic, real-time 3D representations of supply chain data by integrating blockchain, IoT, network connection, and computer this http URL-scale natural language processing model ChatGPT improves communication and data translation accuracy and speed. To manage risk and facilitate decision making in Supply Chain management, firms should increase information transmission, Speed and quality. This study aim to show the importance of ChatGPT and Metaverse technologies to improve SCRES, with an emphasis on the most important criteria for SCRES, and maturity factor that can influence directly the SC development. 

**Abstract (ZH)**: 以下是论文内容或标题的中文翻译，符合学术规范：

全球供应链因 COVID-19 疫情和俄罗斯与乌克兰之间的冲突而严重受阻，导致商品价格上涨并引发了通货膨胀。这些事件突显了提高供应链韧性（Supply Chain Resilience, SCL）的重要性，以抵御不可预见的挫折。SCL 的职责包括控制内外部中断，如自然灾害和战争引起的运输问题。提高供应链韧性需要准确及时的信息传递。

在这些问题中，元宇宙（Metaverse）和 ChatGPT 两种新的数字技术提供了可能的解决方案。元宇宙可以通过整合区块链、物联网、网络连接等技术，模仿现实世界情况，实时动态呈现供应链数据的 3D 表现。ChatGPT 增强了沟通和数据转换的准确性和速度。为了在供应链管理中管理和降低风险，促进决策，企业应增加信息传递的速度和质量。本研究旨在展示元宇宙技术和 ChatGPT 对提高 SCL 的重要性，重点介绍 SCL 最重要的标准和直接影响 SCL 发展的成熟度因素。 

---
# Green AI: Which Programming Language Consumes the Most? 

**Title (ZH)**: 绿色人工智能：哪种编程语言消耗最多资源？ 

**Authors**: Niccolò Marini, Leonardo Pampaloni, Filippo Di Martino, Roberto Verdecchia, Enrico Vicario  

**Link**: [PDF](https://arxiv.org/pdf/2501.14776)  

**Abstract**: AI is demanding an evergrowing portion of environmental resources. Despite their potential impact on AI environmental sustainability, the role that programming languages play in AI (in)efficiency is to date still unknown. With this study, we aim to understand the impact that programming languages can have on AI environmental sustainability. To achieve our goal, we conduct a controlled empirical experiment by considering five programming languages (C++, Java, Python, MATLAB, and R), seven AI algorithms (KNN, SVC, AdaBoost, decision tree, logistic regression, naive bayses, and random forest), three popular datasets, and the training and inference phases. The collected results show that programming languages have a considerable impact on AI environmental sustainability. Compiled and semi-compiled languages (C++, Java) consistently consume less than interpreted languages (Python, MATLAB, R), which require up to 54x more energy. Some languages are cumulatively more efficient in training, while others in inference. Which programming language consumes the most highly depends on the algorithm considered. Ultimately, algorithm implementation might be the most determining factor in Green AI, regardless of the language used. As conclusion, while making AI more environmentally sustainable is paramount, a trade-off between energy efficiency and implementation ease should always be considered. Green AI can be achieved without the need of completely disrupting the development practices and technologies currently in place. 

**Abstract (ZH)**: 人工智能对环境资源的需求日益增加。尽管编程语言对人工智能效率的影响可能对其环境可持续性产生重大影响，但其具体作用至今仍未知。通过本研究，我们旨在了解编程语言对人工智能环境可持续性的影响。为了实现这一目标，我们通过考虑五种编程语言（C++、Java、Python、MATLAB 和 R）、七种人工智能算法（KNN、SVC、AdaBoost、决策树、逻辑回归、朴素贝叶斯和随机森林）、三种流行的数据集以及训练和推理阶段，进行了受控的实证实验。收集到的结果表明，编程语言对人工智能环境可持续性的影响显著。编译和半编译语言（如 C++ 和 Java）始终比解释型语言（如 Python、MATLAB 和 R）消耗更少的能源，后者所需的能量最多可达到前者的 54 倍。在训练阶段和推理阶段，某些语言表现出不同程度的效率。哪个编程语言消耗的能源最多取决于所使用的算法。最终，算法的实现可能是绿色人工智能中最重要的决定因素，无论使用何种语言。综上所述，在提高人工智能环境可持续性的过程中，应权衡能源效率和实现便捷性之间的折衷。绿色人工智能可以在不完全颠覆当前开发实践和技术的前提下得以实现。 

---
# Hybrid Firefly-Genetic Algorithm for Single and Multi-dimensional 0-1 Knapsack Problems 

**Title (ZH)**: 用于单维度和多维度0-1背包问题的萤火虫-遗传算法混合方法 

**Authors**: Aswathi Malanthara, Ishaan R Kale  

**Link**: [PDF](https://arxiv.org/pdf/2501.14775)  

**Abstract**: This paper addresses the challenges faced by algorithms, such as the Firefly Algorithm (FA) and the Genetic Algorithm (GA), in constrained optimization problems. While both algorithms perform well for unconstrained problems, their effectiveness diminishes when constraints are introduced due to limitations in exploration, exploitation, and constraint handling. To overcome these challenges, a hybrid FAGA algorithm is proposed, combining the strengths of both algorithms. The hybrid algorithm is validated by solving unconstrained benchmark functions and constrained optimization problems, including design engineering problems and combinatorial problems such as the 0-1 Knapsack Problem. The proposed algorithm delivers improved solution accuracy and computational efficiency compared to conventional optimization algorithm. This paper outlines the development and structure of the hybrid algorithm and demonstrates its effectiveness in handling complex optimization problems. 

**Abstract (ZH)**: 本文探讨了如萤火虫算法（FA）和遗传算法（GA）等算法在约束优化问题中面临的挑战。尽管这两种算法在无约束问题上表现良好，但在引入约束条件后其有效性会降低，原因在于探索、利用和约束处理能力的局限性。为克服这些挑战，本文提出了一种混合FAGA算法，综合利用了两种算法的优势。该混合算法通过求解无约束基准函数和约束优化问题（包括设计工程问题和组合问题如0-1背包问题）得到了验证。与传统优化算法相比，提出的算法在解的准确性及计算效率方面都有所提升。本文详细阐述了混合算法的开发过程及其结构，并展示了其在处理复杂优化问题方面的有效性。 

---
# DropMicroFluidAgents (DMFAs): Autonomous Droplet Microfluidic Research Framework Through Large Language Model Agents 

**Title (ZH)**: DropMicroFluidAgents (DMFAs): 通过大型语言模型代理实现的自主液滴微流控研究框架 

**Authors**: Dinh-Nguyen Nguyen, Raymond Kai-Yu Tong, Ngoc-Duy Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2501.14772)  

**Abstract**: Applying Large language models (LLMs) within specific domains requires substantial adaptation to account for the unique terminologies, nuances, and context-specific challenges inherent to those areas. Here, we introduce DropMicroFluidAgents (DMFAs), an advanced language-driven framework leveraging state-of-the-art pre-trained LLMs. DMFAs employs LLM agents to perform two key functions: (1) delivering focused guidance, answers, and suggestions specific to droplet microfluidics and (2) generating machine learning models to optimise and automate the design of droplet microfluidic devices, including the creation of code-based computer-aided design (CAD) scripts to enable rapid and precise design execution. Experimental evaluations demonstrated that the integration of DMFAs with the LLAMA3.1 model yielded the highest accuracy of 76.15%, underscoring the significant performance enhancement provided by agent integration. This effect was particularly pronounced when DMFAs were paired with the GEMMA2 model, resulting in a 34.47% improvement in accuracy compared to the standalone GEMMA2 configuration. This study demonstrates the effective use of LLM agents in droplet microfluidics research as powerful tools for automating workflows, synthesising knowledge, optimising designs, and interacting with external systems. These capabilities enable their application across education and industrial support, driving greater efficiency in scientific discovery and innovation. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，符合学术规范：

在特定领域应用大型语言模型（LLMs）需要进行大量的适应，以应对这些领域内独特的术语、细微差别和情境特定的挑战。本文介绍了DropMicroFluidAgents（DMFAs），这是一种利用最新预训练LLMs的先进语言驱动框架。DMFAs 通过LLMs代理执行两个关键功能：（1）提供针对微液滴微流控领域的集中指导、回答和建议；（2）生成机器学习模型以优化和自动化微液滴微流控装置的设计，包括生成基于代码的计算机辅助设计（CAD）脚本，以实现快速和精确的设计执行。实验评估表明，将DMFAs 与LLAMA3.1模型结合使用，能够获得最高的准确率76.15%，突显了代理集成提供的显著性能提升。当DMFAs 与GEMMA2模型配对时，其准确率提高了34.47%，超过了仅使用GEMMA2配置的情况。研究证明了在微液滴微流控研究中有效使用LLMs代理作为自动化工序的强大工具，能够整合知识、优化设计，并与外部系统交互。这些能力使得它们能够在教育和工业支持中得到应用，促进科学研究和创新效率的提升。 

---
# A survey on pioneering metaheuristic algorithms between 2019 and 2024 

**Title (ZH)**: 2019年至2024年间开创性元启发式算法的综述 

**Authors**: Tansel Dokeroglu, Deniz Canturk, Tayfun Kucukyilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2501.14769)  

**Abstract**: This review examines over 150 new metaheuristics of the last six years (between 2019 and 2024), underscoring their profound influence and performance. Over the past three decades, more than 500 new metaheuristic algorithms have been proposed, with no slowdown in sight. An overwhelming abundance that complicates the process of selecting and assessing the most effective solutions for complex optimization challenges. Our evaluation centers on pivotal criteria, including annual citation metrics, the breadth of the addressed problem types, source code availability, user friendly parameter configurations, innovative mechanisms and operators, and approaches designed to mitigate traditional metaheuristic issues such as stagnation and premature convergence. We further explore recent high impact applications of the past six years' most influential 23 metahueristic algorithms, shedding light on their advantages and limitations, while identifying challenges and potential avenues for future research. 

**Abstract (ZH)**: 本综述检视了过去六年（2019年至2024年）提出的超过150种新的元启发式算法，并强调了它们深远的影响和表现。在过去的三十年中，已有超过500种新的元启发式算法被提出，这一速度似乎没有放缓的迹象。这种巨大的数量增加了为复杂优化挑战选择和评估最有效方法的复杂性。我们的评估集中在关键指标上，包括年度引用指标、所解决的问题类型多样性、源代码的可用性、用户友好的参数配置、创新的机制和操作，以及旨在缓解传统元启发式算法问题（如停滞和过早收敛）的策略。此外，我们还探讨了过去六年中最具有影响力的23种元启发式算法的最新高影响力应用，阐明了它们的优势和局限性，并指出了未来研究的挑战和潜在方向。 

---
# Equation discovery framework EPDE: Towards a better equation discovery 

**Title (ZH)**: EPDE方程发现框架：迈向更出色的方程发现

在这个翻译中，"Equation discovery framework EPDE" 被翻译为 "EPDE方程发现框架"，"Towards a better equation discovery" 被翻译为 "迈向更出色的方程发现"。这样的翻译既保留了原意，又符合学术文章的专业表达方式。 

**Authors**: Mikhail Maslyaev, Alexander Hvatov  

**Link**: [PDF](https://arxiv.org/pdf/2501.14768)  

**Abstract**: Equation discovery methods hold promise for extracting knowledge from physics-related data. However, existing approaches often require substantial prior information that significantly reduces the amount of knowledge extracted. In this paper, we enhance the EPDE algorithm -- an evolutionary optimization-based discovery framework. In contrast to methods like SINDy, which rely on pre-defined libraries of terms and linearities, our approach generates terms using fundamental building blocks such as elementary functions and individual differentials. Within evolutionary optimization, we may improve the computation of the fitness function as is done in gradient methods and enhance the optimization algorithm itself. By incorporating multi-objective optimization, we effectively explore the search space, yielding more robust equation extraction, even when dealing with complex experimental data. We validate our algorithm's noise resilience and overall performance by comparing its results with those from the state-of-the-art equation discovery framework SINDy. 

**Abstract (ZH)**: 方程发现方法在从与物理相关的数据中提取知识方面充满希望。然而，现有的方法往往需要大量的先验信息，这显著减少了提取的知识量。在本文中，我们改进了基于进化优化的发现框架EPDE算法。与依赖于预定义项库和线性性的方法（如SINDy）不同，我们的方法使用基本构建块（如基本函数和个体微分）生成术语。在进化优化中，我们可以像在梯度方法中那样改进适应度函数的计算，并增强优化算法本身。通过引入多目标优化，我们有效地探索搜索空间，从而在处理复杂实验数据时也能获得更稳健的方程提取结果。我们通过将我们的算法结果与目前最先进的方程发现框架SINDy的成果进行比较，验证了其抗噪声能力和整体性能。 

---
# Leveraging Social Media Data and Artificial Intelligence for Improving Earthquake Response Efforts 

**Title (ZH)**: 利用社交媒体数据和人工智能提高地震应对努力 

**Authors**: Kalin Kopanov, Velizar Varbanov, Tatiana Atanasova  

**Link**: [PDF](https://arxiv.org/pdf/2501.14767)  

**Abstract**: The integration of social media and artificial intelligence (AI) into disaster management, particularly for earthquake response, represents a profound evolution in emergency management practices. In the digital age, real-time information sharing has reached unprecedented levels, with social media platforms emerging as crucial communication channels during crises. This shift has transformed traditional, centralized emergency services into more decentralized, participatory models of disaster situational awareness. Our study includes an experimental analysis of 8,900 social media interactions, including 2,920 posts and 5,980 replies on X (formerly Twitter), following a magnitude 5.1 earthquake in Oklahoma on February 2, 2024. The analysis covers data from the immediate aftermath and extends over the following seven days, illustrating the critical role of digital platforms in modern disaster response. The results demonstrate that social media platforms can be effectively used as real-time situational awareness tools, delivering critical information to society and authorities during emergencies. 

**Abstract (ZH)**: 将社会媒体和人工智能（AI）整合到灾害管理中，特别是在地震响应中，代表了应急管理实践的一项深刻变革。在数字时代，实时信息共享达到了前所未有的水平，社会媒体平台在此期间成为危机时期至关重要的沟通渠道。这一转变将传统的集中式应急服务转变为更为分散和参与式的灾害情况意识模型。我们的研究包括对8900条社会媒体互动的实验性分析，其中包括发生在2024年2月2日俄克拉荷马州5.1级地震后的2920条帖子和5980条评论（原Twitter平台），该分析涵盖了震后立即以及随后七天的数据，展示了数字平台在现代灾害响应中的关键作用。研究结果表明，社会媒体平台可以有效用作实时情况意识工具，在紧急情况下向社会和当局提供关键信息。 

---
# Artificial Intelligence for Sustainable Urban Biodiversity: A Framework for Monitoring and Conservation 

**Title (ZH)**: 人工智能在可持续城市生物多样性中的应用：监测与保护框架 

**Authors**: Yasmin Rahmati  

**Link**: [PDF](https://arxiv.org/pdf/2501.14766)  

**Abstract**: The rapid expansion of urban areas challenges biodiversity conservation, requiring innovative ecosystem management. This study explores the role of Artificial Intelligence (AI) in urban biodiversity conservation, its applications, and a framework for implementation. Key findings show that: (a) AI enhances species detection and monitoring, achieving over 90% accuracy in urban wildlife tracking and invasive species management; (b) integrating data from remote sensing, acoustic monitoring, and citizen science enables large-scale ecosystem analysis; and (c) AI decision tools improve conservation planning and resource allocation, increasing prediction accuracy by up to 18.5% compared to traditional methods. The research presents an AI-Driven Framework for Urban Biodiversity Management, highlighting AI's impact on monitoring, conservation strategies, and ecological outcomes. Implementation strategies include: (a) standardizing data collection and model validation, (b) ensuring equitable AI access across urban contexts, and (c) developing ethical guidelines for biodiversity monitoring. The study concludes that integrating AI in urban biodiversity conservation requires balancing innovation with ecological wisdom and addressing data quality, socioeconomic disparities, and ethical concerns. 

**Abstract (ZH)**: 城市区域的迅速扩展对生物多样性的保护构成了挑战，需要创新的生态系统管理策略。本研究探讨了人工智能（AI）在城市生物多样性保护中的作用、应用以及实施框架。主要发现包括：（a）AI提升了物种检测和监测的效率，在城市野生动物追踪和入侵物种管理中实现了超过90%的准确率；（b）整合遥感、声学监测和公民科学的数据使得大范围生态系统分析成为可能；（c）AI决策工具改善了保护规划和资源配置，与传统方法相比，预测准确率提高了多达18.5%。研究提出了一个基于AI的城市生物多样性管理框架，强调了AI对监测、保护策略和生态效果的影响。实施策略包括：（a）标准化数据收集和模型验证，（b）确保在城市各区域提供公平的AI访问，以及（c）制定生物多样性监测的伦理指南。研究得出结论，将AI整合到城市生物多样性保护中需要在创新与生态智慧之间取得平衡，并解决数据质量、社会经济不平等和伦理关切等问题。 

---
# Towards An Automated AI Act FRIA Tool That Can Reuse GDPR's DPIA 

**Title (ZH)**: 向自动化的AI法案FRIA工具迈进：该工具能够reuse GDPR的DPIA 

**Authors**: Tytti Rintamaki, Harshvardhan J. Pandit  

**Link**: [PDF](https://arxiv.org/pdf/2501.14756)  

**Abstract**: The AI Act introduces the obligation to conduct a Fundamental Rights Impact Assessment (FRIA), with the possibility to reuse a Data Protection Impact Assessment (DPIA), and requires the EU Commission to create of an automated tool to support the FRIA process. In this article, we provide our novel exploration of the DPIA and FRIA as information processes to enable the creation of automated tools. We first investigate the information involved in DPIA and FRIA, and then use this to align the two to state where a DPIA can be reused in a FRIA. We then present the FRIA as a 5-step process and discuss the role of an automated tool for each step. Our work provides the necessary foundation for creating and managing information for FRIA and supporting it through an automated tool as required by the AI Act. 

**Abstract (ZH)**: 《AI法案》引入了进行基本权利影响评估（FRIA）的义务，允许重新使用数据保护影响评估（DPIA），并要求欧盟委员会创建支持FRIA过程的自动化工具。本文我们提供了一种新颖的研究探索，将DPIA和FRIA视为信息过程，以支持自动化工具的创建。首先，我们研究了DPIA和FRIA中涉及的信息；然后，我们利用这些信息，确定DPIA可以在FRIA中哪些部分被重新使用。接下来，我们阐述了FRIA的五个步骤，并讨论了自动化工具在每个步骤中的作用。本文为创建和管理FRIA所需的信息以及通过自动化工具支持其实施提供了必要的基础，符合《AI法案》的要求。 

---
# Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for Foundation Models 

**Title (ZH)**: Data-Juicer 2.0：面向基础模型的云规模自适应数据处理 

**Authors**: Daoyuan Chen, Yilun Huang, Xuchen Pan, Nana Jiang, Haibin Wang, Ce Ge, Yushuo Chen, Wenhao Zhang, Zhijian Ma, Yilei Zhang, Jun Huang, Wei Lin, Yaliang Li, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.14755)  

**Abstract**: The burgeoning field of foundation models necessitates advanced data processing mechanisms capable of harnessing vast valuable data with varied types utilized by these models. Nevertheless, the current landscape presents unique challenges that traditional data processing frameworks cannot handle effectively, especially with multimodal intricacies. In response, we present Data-Juicer 2.0, a new system offering fruitful data processing capabilities backed by over a hundred operators spanning various modalities like text, image, audio, and video. With seamless compatibility and dedicated optimization to popular dataset hubs like Hugging Face and computing engines like Ray, Data-Juicer 2.0 enhances its predecessor in both usability, efficiency, and programmability. It features an easily accessible user interface layer that supports decoupled Python interactions, RESTful APIs, and conversational commands. Alongside this, it contains a core runtime layer optimized for adaptive execution and management across different dataset scales, processing demands, and computational environments, while shielding unnecessary system details. Extensive empirical evaluations demonstrate Data-Juicer 2.0's remarkable performance and scalability, highlighting its capability to efficiently process tens of billions of data samples with tens of thousands of CPU cores. The system is publicly available, actively maintained, and broadly adopted in diverse research endeavors, practical applications, and real-world products such as Alibaba Cloud PAI. 

**Abstract (ZH)**: 础模型领域的蓬勃发展催生了能够利用各种类型海量有价值数据的先进数据处理机制。然而，当前的数据处理框架在应对多模态复杂性方面仍存在独特挑战，传统框架无法有效应对。为应对这一挑战，我们推出了Data-Juicer 2.0，这是一个全新的系统，提供了丰富的数据处理能力，涵盖了超过一百种操作符，适用于文本、图像、音频和视频等多种模态。Data-Juicer 2.0 通过无缝兼容性和针对流行的如 Hugging Face 数据集库和计算引擎如 Ray 的专门优化，在易用性、效率和编程性方面提升了其前身。它包含一个易于访问的用户界面层，支持脱钩的 Python 交互、RESTful API 和会话命令。此外，它还包含一个核心运行时层，针对不同数据集规模、处理需求和计算环境进行了优化管理，并屏蔽了不必要的系统细节。广泛的实证评估表明，Data-Juicer 2.0 具有出色的性能和可扩展性，能够高效处理数十亿条数据样本和数万个 CPU 核心。该系统已公开提供，并被积极维护和广泛应用于各种研究项目、实际应用和真实世界的产品中，例如阿里云 PAI。 

---
# ABACUS: A FinOps Service for Cloud Cost Optimization 

**Title (ZH)**: ABACUS：一种云成本优化的FinOps服务 

**Authors**: Saurabh Deochake  

**Link**: [PDF](https://arxiv.org/pdf/2501.14753)  

**Abstract**: In recent years, as more enterprises have moved their infrastructure to the cloud, significant challenges have emerged in achieving holistic cloud spend visibility and cost optimization. FinOps practices provide a way for enterprises to achieve these business goals by optimizing cloud costs and bringing accountability to cloud spend. This paper presents ABACUS - Automated Budget Analysis and Cloud Usage Surveillance, a FinOps solution for optimizing cloud costs by setting budgets, enforcing those budgets through blocking new deployments, and alerting appropriate teams if spending breaches a budget threshold. ABACUS also leverages best practices like Infrastructure-as-Code to alert engineering teams of the expected cost of deployment before resources are deployed in the cloud. Finally, future research directions are proposed to advance the state of the art in this important field. 

**Abstract (ZH)**: 近年来，随着越来越多的企业将基础设施迁移到云端，实现整体云支出可见性和成本优化的重大挑战也随之而来。FinOps实践为企业提供了一种方式，通过优化云成本和对云支出负责来实现这些商业目标。本文介绍了ABACUS——自动化预算分析和云使用监控——这是一种FinOps解决方案，通过设定预算、通过阻止新部署来强制执行这些预算，并在支出超过预算阈值时向相关部门发出警报，进而优化云成本。ABACUS还利用基础设施即代码的最佳实践，在资源部署到云端之前提醒工程团队预估部署成本。最后，提出了未来的研究方向，以推动这一重要领域的技术发展。 

---
# Enhancing Green Economy with Artificial Intelligence: Role of Energy Use and FDI in the United States 

**Title (ZH)**: 增强绿色经济：能源使用和外国直接投资在人工智能的作用下对美国的影响 

**Authors**: Abdullah Al Abrar Chowdhury, Azizul Hakim Rafi, Adita Sultana, Abdulla All Noman  

**Link**: [PDF](https://arxiv.org/pdf/2501.14747)  

**Abstract**: The escalating challenge of climate change necessitates an urgent exploration of factors influencing carbon emissions. This study contributes to the discourse by examining the interplay of technological, economic, and demographic factors on environmental sustainability. This study investigates the impact of artificial intelligence (AI) innovation, economic growth, foreign direct investment (FDI), energy consumption, and urbanization on CO2 emissions in the United States from 1990 to 2022. Employing the ARDL framework integrated with the STIRPAT model, the findings reveal a dual narrative: while AI innovation mitigates environmental stress, economic growth, energy use, FDI, and urbanization exacerbate environmental degradation. Unit root tests (ADF, PP, and DF-GLS) confirm mixed integration levels among variables, and the ARDL bounds test establishes long-term co-integration. The analysis highlights that AI innovation positively correlates with CO2 reduction when environmental safeguards are in place, whereas GDP growth, energy consumption, FDI, and urbanization intensify CO2 emissions. Robustness checks using FMOLS, DOLS, and CCR validate the ARDL findings. Additionally, Pairwise Granger causality tests reveal significant one-way causal links between CO2 emissions and economic growth, AI innovation, energy use, FDI, and urbanization. These relationships emphasize the critical role of AI-driven technological advancements, sustainable investments, and green energy in fostering ecological sustainability. The study suggests policy measures such as encouraging green FDI, advancing AI technologies, adopting sustainable energy practices, and implementing eco-friendly urban development to promote sustainable growth in the USA. 

**Abstract (ZH)**: 气候变化的不断升级迫使我们迫切地探索影响碳排放的因素。本研究通过分析技术、经济和人口因素对环境可持续性的影响，为讨论做出了贡献。本文研究了1990年至2022年间美国的人工智能创新、经济增长、外国直接投资（FDI）、能源消耗和城市化进程对二氧化碳排放的影响。采用结合STIRPAT模型的协整范围检验框架（ARDL），研究结果揭示了一种复杂的故事：虽然人工智能创新减轻了环境压力，但经济增长、能源使用、FDI和城市化进程加剧了环境污染。单位根检验（ADF、PP和DF-GLS）证实了变量之间的混合整合水平，而ARDL边界检验确立了长期协整关系。分析显示，在环境保护措施到位的情况下，人工智能创新与二氧化碳减排正相关；然而，国内生产总值增长、能源消耗、FDI和城市化进程会加剧二氧化碳排放。使用FMOLS、DOLS和CCR进行的稳健性检验证实了ARDL的结果。此外，成对Granger因果关系检验揭示了二氧化碳排放与经济增长、人工智能创新、能源消耗、FDI和城市化进程之间存在显著的一次性因果联系。这些关系强调了人工智能驱动的技术进步、可持续投资和绿色能源在促进生态可持续性方面的重要作用。本研究建议制定政策措施，如鼓励绿色FDI，推进人工智能技术，采用可持续能源实践，实施环保城市发展模式，以促进美国的可持续增长。 

---
# EvalSVA: Multi-Agent Evaluators for Next-Gen Software Vulnerability Assessment 

**Title (ZH)**: EvalSVA：面向下一代软件漏洞评估的多agent评估器 

**Authors**: Xin-Cheng Wen, Jiaxin Ye, Cuiyun Gao, Lianwei Wu, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14737)  

**Abstract**: Software Vulnerability (SV) assessment is a crucial process of determining different aspects of SVs (e.g., attack vectors and scope) for developers to effectively prioritize efforts in vulnerability mitigation. It presents a challenging and laborious process due to the complexity of SVs and the scarcity of labeled data. To mitigate the above challenges, we introduce EvalSVA, a multi-agent evaluators team to autonomously deliberate and evaluate various aspects of SV assessment. Specifically, we propose a multi-agent-based framework to simulate vulnerability assessment strategies in real-world scenarios, which employs multiple Large Language Models (LLMs) into an integrated group to enhance the effectiveness of SV assessment in the limited data. We also design diverse communication strategies to autonomously discuss and assess different aspects of SV. Furthermore, we construct a multi-lingual SV assessment dataset based on the new standard of CVSS, comprising 699, 888, and 1,310 vulnerability-related commits in C++, Python, and Java, respectively. Our experimental results demonstrate that EvalSVA averagely outperforms the 44.12\% accuracy and 43.29\% F1 for SV assessment compared with the previous methods. It shows that EvalSVA offers a human-like process and generates both reason and answer for SV assessment. EvalSVA can also aid human experts in SV assessment, which provides more explanation and details for SV assessment. 

**Abstract (ZH)**: 软件漏洞（SV）评估是确定不同方面漏洞（例如攻击向量和影响范围）的重要过程，旨在帮助开发者有效优先考虑漏洞缓解工作。这一过程由于漏洞的复杂性和标注数据的稀缺性而变得具有挑战性和繁琐。为了应对上述挑战，我们引入了EvalSVA，这是一个自主讨论和评估漏洞评估各种方面的一组多智能体评价者团队。具体而言，我们提出了一种基于多智能体的框架，在实际场景中模拟漏洞评估策略，该框架通过将多个大型语言模型（LLMs）整合到一个小组中，增强了在数据有限情况下的漏洞评估效果。我们还设计了多种通信策略，以自主讨论和评估不同方面漏洞。此外，我们根据新的CVSS标准构建了一个多语言漏洞评估数据集，其中包含C++、Python和Java语言中分别共计699,888个和1,310个漏洞相关提交记录。实验结果显示，EvalSVA相比之前的评估方法，在漏洞评估准确性和F1分数上平均高出44.12%和43.29%。这表明EvalSVA提供了类似人类的过程，并为漏洞评估生成了合理性和答案。同时，EvalSVA也可以帮助人类专家进行漏洞评估，提供更详细的解释和信息。 

---
# ARCEAK: An Automated Rule Checking Framework Enhanced with Architectural Knowledge 

**Title (ZH)**: ARCEAK：一种集成架构知识的自动规则检查框架 

**Authors**: Junyong Chen, Ling-I Wu, Minyu Chen, Xiaoying Qian, Haoze Zhu, Qiongfang Zhang, Guoqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14735)  

**Abstract**: Automated Rule Checking (ARC) plays a crucial role in advancing the construction industry by addressing the laborious, inconsistent, and error-prone nature of traditional model review conducted by industry professionals. Manual assessment against intricate sets of rules often leads to significant project delays and expenses. In response to these challenges, ARC offers a promising solution to improve efficiency and compliance in design within the construction sector. However, the main challenge of ARC lies in translating regulatory text into a format suitable for computer processing. Current methods for rule interpretation require extensive manual labor, thereby limiting their practicality. To address this issue, our study introduces a novel approach that decomposes ARC into two distinct tasks: rule information extraction and verification code generation. Leveraging generative pre-trained transformers, our method aims to streamline the interpretation of regulatory texts and simplify the process of generating model compliance checking code. Through empirical evaluation and case studies, we showcase the effectiveness and potential of our approach in automating code compliance checking, enhancing the efficiency and reliability of construction projects. 

**Abstract (ZH)**: 自动化规则检查（ARC）在建筑行业中发挥着关键作用，通过解决传统模型审核工作中繁琐、不一致且易出错的问题。专业人员在进行复杂规则的逐条评估时，往往会导致项目延误和成本增加。为应对这些挑战，ARC 提供了一种有望提高设计效率和合规性的解决方案。然而，ARC 的主要挑战在于将监管文本转换为适应计算机处理的格式。当前的规则解释方法需要大量的手动劳动，从而限制了其实用性。为解决这一问题，我们的研究提出了一种新颖的方法，将 ARC 分解为两个独立的任务：规则信息提取和验证代码生成。利用生成式预训练转换器，我们的方法旨在简化监管文本的解释过程，并简化生成模型合规检查代码的过程。通过实证评估和案例研究，我们展示了该方法在自动化代码合规检查方面的有效性和潜在价值，从而提高建筑项目的效率和可靠性。 

---
# Research on the Application of Spark Streaming Real-Time Data Analysis System and large language model Intelligent Agents 

**Title (ZH)**: 研究Spark Streaming实时数据分析系统与大型语言模型智能代理的应用 

**Authors**: Jialin Wang, Zhihua Duan  

**Link**: [PDF](https://arxiv.org/pdf/2501.14734)  

**Abstract**: This study explores the integration of Agent AI with LangGraph to enhance real-time data analysis systems in big data environments. The proposed framework overcomes limitations of static workflows, inefficient stateful computations, and lack of human intervention by leveraging LangGraph's graph-based workflow construction and dynamic decision-making capabilities. LangGraph allows large language models (LLMs) to dynamically determine control flows, invoke tools, and assess the necessity of further actions, improving flexibility and efficiency.
The system architecture incorporates Apache Spark Streaming, Kafka, and LangGraph to create a high-performance sentiment analysis system. LangGraph's capabilities include precise state management, dynamic workflow construction, and robust memory checkpointing, enabling seamless multi-turn interactions and context retention. Human-in-the-loop mechanisms are integrated to refine sentiment analysis, particularly in ambiguous or high-stakes scenarios, ensuring greater reliability and contextual relevance.
Key features such as real-time state streaming, debugging via LangGraph Studio, and efficient handling of large-scale data streams make this framework ideal for adaptive decision-making. Experimental results confirm the system's ability to classify inquiries, detect sentiment trends, and escalate complex issues for manual review, demonstrating a synergistic blend of LLM capabilities and human oversight.
This work presents a scalable, adaptable, and reliable solution for real-time sentiment analysis and decision-making, advancing the use of Agent AI and LangGraph in big data applications. 

**Abstract (ZH)**: 本研究探讨了在大数据环境中通过将代理AI与LangGraph结合来增强实时数据分析系统的可能性。所提出的框架克服了静态工作流、低效的状态计算以及缺乏人工干预的限制，通过利用LangGraph基于图的工作流构建能力和动态决策能力。LangGraph使大型语言模型（LLMs）能够动态确定控制流、调用工具并评估进一步行动的必要性，从而提高灵活性和效率。

该系统架构集成了Apache Spark Streaming、Kafka和LangGraph，以创建高性能的情绪分析系统。LangGraph的功能包括精确的状态管理、动态工作流构建和强大的内存快照功能，能够实现无缝的多轮交互和上下文保留。将人工干预机制融入系统，尤其是在含义模糊或高风险的情景下，进一步细化情绪分析，确保更高的可靠性和相关性。

系统的关键功能包括实时状态流、通过LangGraph Studio进行调试以及高效处理大规模数据流，使该框架成为适应性决策的理想选择。实验结果证实了该系统的分类查询、检测情绪趋势以及将复杂问题提交人工审查的能力，展示了LLMs能力和人工监督的协同作用。

本研究提出了一个可扩展、适应性强且可靠的情绪分析及决策解决方案，推动了代理AI和LangGraph在大数据应用中的应用。 

---
# LLM as HPC Expert: Extending RAG Architecture for HPC Data 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，确保符合学术规范：

LLM作为HPC专家：扩展基于RAG的架构以处理HPC数据

解释：
- LLM：Large Language Model（大规模语言模型）
- HPC：High-Performance Computing（高性能计算）
- RAG：Retrieval-Augmented Generation（检索增强生成）

这个标题翻译已经尽可能保持了原有的专业术语，并且符合中文的表达习惯。 

**Authors**: Yusuke Miyashita, Patrick Kin Man Tung, Johan Barthélemy  

**Link**: [PDF](https://arxiv.org/pdf/2501.14733)  

**Abstract**: High-Performance Computing (HPC) is crucial for performing advanced computational tasks, yet their complexity often challenges users, particularly those unfamiliar with HPC-specific commands and workflows. This paper introduces Hypothetical Command Embeddings (HyCE), a novel method that extends Retrieval-Augmented Generation (RAG) by integrating real-time, user-specific HPC data, enhancing accessibility to these systems. HyCE enriches large language models (LLM) with real-time, user-specific HPC information, addressing the limitations of fine-tuned models on such data. We evaluate HyCE using an automated RAG evaluation framework, where the LLM itself creates synthetic questions from the HPC data and serves as a judge, assessing the efficacy of the extended RAG with the evaluation metrics relevant for HPC tasks. Additionally, we tackle essential security concerns, including data privacy and command execution risks, associated with deploying LLMs in HPC environments. This solution provides a scalable and adaptable approach for HPC clusters to leverage LLMs as HPC expert, bridging the gap between users and the complex systems of HPC. 

**Abstract (ZH)**: 高性能计算（HPC）对于执行高级计算任务至关重要，但其复杂性往往挑战用户，尤其是那些不熟悉HPC特定命令和工作流的用户。本文介绍了一种名为假设命令嵌入（HyCE）的新方法，该方法通过集成实时的用户特定HPC数据扩展了检索增强生成（RAG），从而提高HPC系统的易用性。HyCE 通过将实时的用户特定HPC信息融入大规模语言模型（LLM），解决了调优模型在处理此类数据时的局限性。我们使用自动化RAG评估框架来评估HyCE，在该框架中，LLM本身从HPC数据中生成合成问题并充当裁判，评估扩展的RAG的有效性，以符合HPC任务的相关评估指标。此外，我们还解决了在HPC环境中部署LLM时的关键安全问题，包括数据隐私和命令执行风险。该解决方案提供了一种可扩展和适应性强的方法，使HPC集群能够利用LLM作为HPC专家，从而弥合用户与HPC复杂系统的差距。 

---
# From Critique to Clarity: A Pathway to Faithful and Personalized Code Explanations with Large Language Models 

**Title (ZH)**: 从批评到清晰：一条实现忠实和个人化代码解释的路径——大型语言模型的应用 

**Authors**: Zexing Xu, Zhuang Luo, Yichuan Li, Kyumin Lee, S. Rasoul Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2501.14731)  

**Abstract**: In the realm of software development, providing accurate and personalized code explanations is crucial for both technical professionals and business stakeholders. Technical professionals benefit from enhanced understanding and improved problem-solving skills, while business stakeholders gain insights into project alignments and transparency. Despite the potential, generating such explanations is often time-consuming and challenging. This paper presents an innovative approach that leverages the advanced capabilities of large language models (LLMs) to generate faithful and personalized code explanations. Our methodology integrates prompt enhancement, self-correction mechanisms, personalized content customization, and interaction with external tools, facilitated by collaboration among multiple LLM agents. We evaluate our approach using both automatic and human assessments, demonstrating that our method not only produces accurate explanations but also tailors them to individual user preferences. Our findings suggest that this approach significantly improves the quality and relevance of code explanations, offering a valuable tool for developers and stakeholders alike. 

**Abstract (ZH)**: 在软件开发领域，提供准确且个性化的代码解释对于技术人员和商业利益相关者都至关重要。技术人员可以从增强的理解能力和提高的解决问题能力中受益，而商业利益相关者则能获得项目对齐和透明度的见解。尽管具有潜力，生成此类解释往往耗时且具有挑战性。本文提出了一种创新的方法，利用先进的人工智能语言模型（LLMs）来生成忠实且个性化的代码解释。我们的方法结合了提示增强、自我纠错机制、个性化内容定制以及与外部工具的交互，这一切得益于多个LLM代理之间的协作。我们通过自动评估和人工评估两种方式来评估我们的方法，证明我们的方法不仅能够生成准确的解释，还能根据个别用户偏好进行定制。我们的研究结果表明，这种方法显著提高了代码解释的质量和相关性，为开发人员和利益相关者提供了一个有价值的工具。 

---
# A transformer-based deep q learning approach for dynamic load balancing in software-defined networks 

**Title (ZH)**: 基于变压器的深度Q学习方法在软件定义网络中的动态负载均衡 

**Authors**: Evans Tetteh Owusu, Kwame Agyemang-Prempeh Agyekum, Marinah Benneh, Pius Ayorna, Justice Owusu Agyemang, George Nii Martey Colley, James Dzisi Gazde  

**Link**: [PDF](https://arxiv.org/pdf/2501.12829)  

**Abstract**: This study proposes a novel approach for dynamic load balancing in Software-Defined Networks (SDNs) using a Transformer-based Deep Q-Network (DQN). Traditional load balancing mechanisms, such as Round Robin (RR) and Weighted Round Robin (WRR), are static and often struggle to adapt to fluctuating traffic conditions, leading to inefficiencies in network performance. In contrast, SDNs offer centralized control and flexibility, providing an ideal platform for implementing machine learning-driven optimization strategies. The core of this research combines a Temporal Fusion Transformer (TFT) for accurate traffic prediction with a DQN model to perform real-time dynamic load balancing. The TFT model predicts future traffic loads, which the DQN uses as input, allowing it to make intelligent routing decisions that optimize throughput, minimize latency, and reduce packet loss. The proposed model was tested against RR and WRR in simulated environments with varying data rates, and the results demonstrate significant improvements in network performance. For the 500MB data rate, the DQN model achieved an average throughput of 0.275 compared to 0.202 and 0.205 for RR and WRR, respectively. Additionally, the DQN recorded lower average latency and packet loss. In the 1000MB simulation, the DQN model outperformed the traditional methods in throughput, latency, and packet loss, reinforcing its effectiveness in managing network loads dynamically. This research presents an important step towards enhancing network performance through the integration of machine learning models within SDNs, potentially paving the way for more adaptive, intelligent network management systems. 

**Abstract (ZH)**: 本文提出了一种使用基于Transformer的深度Q网络（DQN）的新颖方法，以实现软件定义网络（SDNs）中的动态负载均衡。传统的负载均衡机制，如轮询（Round Robin, RR）和加权轮询（Weighted Round Robin, WRR），通常是静态的，难以适应波动的流量条件，从而导致网络性能效率低下。相比之下，SDNs提供了集中控制和灵活性，为采用基于机器学习的优化策略提供了理想的平台。本文的核心在于结合了时间融合Transformer（Temporal Fusion Transformer, TFT）进行准确的流量预测，并使用DQN模型进行实时动态负载均衡。TFT模型预测未来的流量负荷，DQN将这些预测作为输入，从而能够做出优化吞吐量、减少延迟和降低丢包率的智能路由决策。所提出的模型在不同数据速率的仿真环境中与RR和WRR进行了对比测试，结果显示在网络性能方面有显著提升。在500MB数据速率下，DQN模型的平均吞吐量为0.275，而RR和WRR分别为0.202和0.205。此外，DQN还记录了较低的平均延迟和丢包率。在1000MB的仿真中，DQN模型在吞吐量、延迟和丢包率方面均优于传统方法，进一步证实了其在动态管理网络负载方面的有效性。本文为通过将机器学习模型集成到SDNs中以增强网络性能迈出了重要一步，有可能为更适应性和智能化的网络管理系统铺平道路。 

---
# Optimally-Weighted Maximum Mean Discrepancy Framework for Continual Learning 

**Title (ZH)**: 持续学习中优化加权最大均值差异框架 

**Authors**: KaiHui Huang, RunQing Wu, Fei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2501.12121)  

**Abstract**: Continual learning has emerged as a pivotal area of research, primarily due to its advantageous characteristic that allows models to persistently acquire and retain information. However, catastrophic forgetting can severely impair model performance. In this study, we tackle the issue of network forgetting by introducing a novel framework termed Optimally-Weighted Maximum Mean Discrepancy (OWMMD), which imposes penalties on representation alterations via a Multi-Level Feature Matching Mechanism (MLFMM). Furthermore, we propose an Adaptive Regularization Optimization (ARO) strategy to refine the adaptive weight vectors, which autonomously assess the significance of each feature layer throughout the optimization process. We conduct a comprehensive series of experiments, benchmarking our proposed method against several established baselines. The empirical findings indicate that our approach achieves state-of-the-art performance. 

**Abstract (ZH)**: 持续学习已成为一个关键的研究领域，主要得益于其能够使模型持久地获取和保留信息的优势特性。然而，灾难性遗忘可能会严重损害模型性能。在本研究中，我们通过引入一种新的框架——最优加权最大均方偏差（Optimally-Weighted Maximum Mean Discrepancy，OWMMD）来解决网络遗忘问题，该框架通过多层次特征匹配机制（Multi-Level Feature Matching Mechanism，MLFMM）对表示层的变化施加惩罚。此外，我们提出了一种自适应正则化优化（Adaptive Regularization Optimization，ARO）策略，用于细化自适应权重向量，在优化过程中自主评估每层特征的重要性。我们进行了全面的实验，并将我们提出的方法与若干已建立的基准方法进行了比较。实验结果表明，我们的方法取得了最先进的性能。 

---
