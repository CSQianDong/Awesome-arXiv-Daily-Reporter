# TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues 

**Title (ZH)**: TReMu：面向具有记忆功能的多会话对话中LLM代理的神经符号时间推理 

**Authors**: Yubin Ge, Salvatore Romeo, Jason Cai, Raphael Shu, Monica Sunkara, Yassine Benajiba, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01630)  

**Abstract**: Temporal reasoning in multi-session dialogues presents a significant challenge which has been under-studied in previous temporal reasoning benchmarks. To bridge this gap, we propose a new evaluation task for temporal reasoning in multi-session dialogues and introduce an approach to construct a new benchmark by augmenting dialogues from LoCoMo and creating multi-choice QAs. Furthermore, we present TReMu, a new framework aimed at enhancing the temporal reasoning capabilities of LLM-agents in this context. Specifically, the framework employs \textit{time-aware memorization} through timeline summarization, generating retrievable memory by summarizing events in each dialogue session with their inferred dates. Additionally, we integrate \textit{neuro-symbolic temporal reasoning}, where LLMs generate Python code to perform temporal calculations and select answers. Experimental evaluations on popular LLMs demonstrate that our benchmark is challenging, and the proposed framework significantly improves temporal reasoning performance compared to baseline methods, raising from 29.83 on GPT-4o via standard prompting to 77.67 via our approach and highlighting its effectiveness in addressing temporal reasoning in multi-session dialogues. 

**Abstract (ZH)**: 多会话对话中的时间推理提出了一个重要的挑战，而这一挑战在之前的时序推理基准中尚未得到充分的研究。为解决这一问题，我们提出了一项新的评价任务，旨在评估多会话对话中的时间推理能力，并通过增强Loremotion数据集中的对话，构造了一个新的基准，并创建了多项选择题。此外，我们提出了TReMu框架，该框架旨在增强在这种情况下LLM代理的时间推理能力。具体而言，该框架通过时间轴总结，采用了具有时间意识的记忆化方法，生成可检索的记忆，通过总结每个对话会话中的事件及其推断日期来生成摘要。我们还整合了神经符号时间推理，其中LLM生成Python代码以执行时间计算并选择答案。对流行的基础模型进行的实验评估表明，我们的基准具有挑战性，并且所提的框架在时间推理性能上显著优于基线方法，评分从GPT-4o标准提示下的29.83提高到我们方法下的77.67，突显了其在解决多会话对话中时间推理问题方面的有效性。 

---
# PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models 

**Title (ZH)**: 无需博士学位：大型语言模型的推理挑战 

**Authors**: Carolyn Jane Anderson, Joydeep Biswas, Aleksander Boruch-Gruszecki, Federico Cassano, Molly Q Feldman, Arjun Guha, Francesca Lucchetti, Zixuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01584)  

**Abstract**: Existing benchmarks for frontier models often test specialized, ``PhD-level'' knowledge that is difficult for non-experts to grasp. In contrast, we present a benchmark based on the NPR Sunday Puzzle Challenge that requires only general knowledge. Our benchmark is challenging for both humans and models, however correct solutions are easy to verify, and models' mistakes are easy to spot.
Our work reveals capability gaps that are not evident in existing benchmarks: OpenAI o1 significantly outperforms other reasoning models that are on par on benchmarks that test specialized knowledge. Furthermore, our analysis of reasoning outputs uncovers new kinds of failures. DeepSeek R1, for instance, often concedes with ``I give up'' before providing an answer that it knows is wrong. R1 can also be remarkably ``uncertain'' in its output and in rare cases, it does not ``finish thinking,'' which suggests the need for an inference-time technique to ``wrap up'' before the context window limit is reached. We also quantify the effectiveness of reasoning longer with R1 and Gemini Thinking to identify the point beyond which more reasoning is unlikely to improve accuracy on our benchmark. 

**Abstract (ZH)**: 现有的前沿模型基准通常测试专业化、接近“博士水平”的知识，这使得非专家难以理解。相比之下，我们提出了一个基于《纽约客》周日谜题挑战的基准，只需要一般的知识。我们的基准对人类和模型都是具有挑战性的，但正确的解决方案易于验证，模型的错误也非常容易被发现。

我们的研究揭示了现有基准中未明显显现的能力差距：OpenAI o1 显著优于在测试专业化知识的基准上与其水平相当的其他推理模型。此外，我们对推理输出的分析揭示了新的失败类型。例如，DeepSeek R1 经常在给出它知道是错误的答案之前承认“放弃”。R1 的输出有时也表现得异常“不确定”，偶尔甚至未能完成推理过程，这表明在达到上下文窗口限制之前需要一种推理时的技巧来“总结”。我们还利用 R1 和 Gemini Thinking 推理更长的时间来量化推理长度对基准效果的影响，以确定进一步推理是否还可能提高准确性。 

---
# Sea-cret Agents: Maritime Abduction for Region Generation to Expose Dark Vessel Trajectories 

**Title (ZH)**: 《海-secret代理：海洋劫持方法用于区域生成以揭示隐蔽船舶轨迹》 

**Authors**: Divyagna Bavikadi, Nathaniel Lee, Paulo Shakarian, Chad Parvis  

**Link**: [PDF](https://arxiv.org/pdf/2502.01503)  

**Abstract**: Bad actors in the maritime industry engage in illegal behaviors after disabling their vessel's automatic identification system (AIS) - which makes finding such vessels difficult for analysts. Machine learning approaches only succeed in identifying the locations of these ``dark vessels'' in the immediate future. This work leverages ideas from the literature on abductive inference applied to locating adversarial agents to solve the problem. Specifically, we combine concepts from abduction, logic programming, and rule learning to create an efficient method that approaches full recall of dark vessels while requiring less search area than machine learning methods. We provide a logic-based paradigm for reasoning about maritime vessels, an abductive inference query method, an automatically extracted rule-based behavior model methodology, and a thorough suite of experiments. 

**Abstract (ZH)**: 海洋行业中的一些不良行为者通过禁用船舶的自动识别系统（AIS）来从事非法行为，使得分析师难以寻找这些船舶。仅依靠机器学习方法在未来短期内确实可以找到这些“隐形船舶”的位置。本研究借鉴了关于归因推理在定位敌对代理研究中的方法来解决这一问题。具体而言，我们结合了归因推理、逻辑编程和规则学习的概念，提出了一种高效的方法，既能实现对隐形船舶的全面召回，又能比机器学习方法减少搜索区域。我们提供了一种基于逻辑的船舶推理框架，一种归因推理查询方法，一种自动提取的行为规则模型方法，以及一系列详尽的实验研究。 

---
# Develop AI Agents for System Engineering in Factorio 

**Title (ZH)**: 在Factorio中开发系统工程的人工智能代理 

**Authors**: Neel Kant  

**Link**: [PDF](https://arxiv.org/pdf/2502.01492)  

**Abstract**: Continuing advances in frontier model research are paving the way for widespread deployment of AI agents. Meanwhile, global interest in building large, complex systems in software, manufacturing, energy and logistics has never been greater. Although AI driven system engineering holds tremendous promise, the static benchmarks dominating agent evaluations today fail to capture the crucial skills required for implementing dynamic systems, such as managing uncertain trade-offs and ensuring proactive adaptability. This position paper advocates for training and evaluating AI agents' system engineering abilities through automation-oriented sandbox games-particularly Factorio. By directing research efforts in this direction, we can equip AI agents with the specialized reasoning and long-horizon planning necessary to design, maintain, and optimize tomorrow's most demanding engineering projects. 

**Abstract (ZH)**: 不断推进前沿模型研究正为人工智能代理的广泛应用铺平道路。与此同时，全球在软件、制造、能源和物流等领域构建大型复杂系统的需求前所未有的强烈。虽然基于人工智能的系统工程具有巨大的潜力，但当前主导代理评估中的静态基准未能捕捉到实施动态系统所需的关键技能，如管理不确定的权衡和确保积极的适应性。本文倡议通过自动化导向的沙盒游戏（特别是Factorio）来训练和评估人工智能代理的系统工程能力。通过将研究方向导向这一领域，我们可以使人工智能代理具备设计、维护和优化未来最具挑战性的工程项目的专门推理能力和长期规划能力。 

---
# TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning 

**Title (ZH)**: TeLL-Drive：借助教师大语言模型引导的深度强化学习增强自主驾驶 

**Authors**: Chengkai Xu, Jiaqi Liu, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.01387)  

**Abstract**: Although Deep Reinforcement Learning (DRL) and Large Language Models (LLMs) each show promise in addressing decision-making challenges in autonomous driving, DRL often suffers from high sample complexity, while LLMs have difficulty ensuring real-time decision making. To address these limitations, we propose TeLL-Drive, a hybrid framework that integrates an Teacher LLM to guide an attention-based Student DRL policy. By incorporating risk metrics, historical scenario retrieval, and domain heuristics into context-rich prompts, the LLM produces high-level driving strategies through chain-of-thought reasoning. A self-attention mechanism then fuses these strategies with the DRL agent's exploration, accelerating policy convergence and boosting robustness across diverse driving conditions. Our experimental results, evaluated across multiple traffic scenarios, show that TeLL-Drive outperforms existing baseline methods, including other LLM-based approaches, in terms of success rates, average returns, and real-time feasibility. Ablation studies underscore the importance of each model component, especially the synergy between the attention mechanism and LLM-driven guidance. These findings suggest that TeLL-Drive significantly enhances both the adaptability and safety of autonomous driving systems, while offering a more efficient and scalable approach for policy learning. Full validation results are available on our website. 

**Abstract (ZH)**: 尽管深度强化学习（DRL）和大规模语言模型（LLMs）在自动驾驶决策问题上各具潜力，但DRL往往面临样本复杂性高的问题，而LLMs则难以保证实时决策。为了解决这些局限性，我们提出了一种混合框架——TeLL-Drive，该框架结合了一个指导型的教师LLM，以引导基于注意力的学生DRL策略。通过对包含风险指标、历史场景检索和领域启发式信息的语境提示进行推理，LLM生成了高层次的驾驶策略。随后，自注意力机制将这些策略与DRL代理的探索相结合，加快了策略收敛速度，并提高了在各种驾驶条件下的鲁棒性。我们的实验结果，跨越多个交通场景进行评估，表明TeLL-Drive在成功率、平均回报以及实时可行性方面优于现有基准方法，包括其他LLM基方法。消融研究强调了每个模型组件的重要性，特别是注意力机制与LLM驱动指导之间的协同作用。这些发现表明，TeLL-Drive显著增强了自动驾驶系统的适应性和安全性，同时提供了一种更高效、更可扩展的策略学习方法。完整的验证结果可在我们的网站上查阅。 

---
# PSSD: Making Large Language Models Self-denial via Human Psyche Structure 

**Title (ZH)**: PSSD：通过人类心理结构使大型语言模型具备自我否定能力 

**Authors**: Jinzhi Liao, Zenghua Liao, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.01344)  

**Abstract**: The enhance of accuracy in reasoning results of LLMs arouses the community's interests, wherein pioneering studies investigate post-hoc strategies to rectify potential mistakes. Despite extensive efforts, they are all stuck in a state of resource competition demanding significant time and computing expenses. The cause of the situation lies in the failure of identifying the fundamental feature of the solutions in this line, coined as the self-denial of LLMs. In other words, LLMs should confidently determine the potential existence of mistakes and carefully execute the targeted correction. As the whole procedure conducts within LLMs, supporting and persuasive references are hard to acquire, while the absence of specific steps towards refining hidden mistakes persists even when errors are acknowledged. In response to the challenges, we present PSSD, which refers to and implements the human psyche structure such that three distinct and interconnected roles contribute to human reasoning. Specifically, PSSD leverages the recent multi-agent paradigm, and is further enhanced with three innovatively conceived roles: (1) the intuition-based id role that provides initial attempts based on benign LLMs; (2) the rule-driven superego role that summarizes rules to regulate the above attempts, and returns specific key points as guidance; and (3) the script-centric ego role that absorbs all procedural information to generate executable script for the final answer prediction. Extensive experiments demonstrate that the proposed design not only better enhance reasoning capabilities, but also seamlessly integrate with current models, leading to superior performance. 

**Abstract (ZH)**: 随着大型语言模型（LLM）推理结果准确性的提升，学术界对此产生了浓厚的兴趣。在这其中，开创性的研究专注于事后纠正潜在错误的策略。尽管付出了大量努力，但这些方法仍然受困于资源竞争，需要耗费大量时间和计算资源。这种情况的根源在于未能识别出这类解决方案的基本特征，这被称为LLM的自我否定。换句话说，LLM应该自信地确定潜在错误的存在，并谨慎执行针对性的修正。由于整个过程完全在LLM内部进行，难以获得支持和有说服力的参考，而且即使承认了错误，细化隐藏错误的具体步骤仍然缺失。

为了应对这一挑战，我们提出了PSSD，这是一种参考并实现人类心理结构的设计，其中三个相互关联且各不相同的角色共同参与人类推理过程。具体而言，PSSD 利用了最近的多智能体范式，并进一步通过三个新颖构思的角色进行增强：（1）基于直觉的自我角色，它基于良性LLM提供初始尝试；（2）基于规则的超我角色，它总结规则以调节上述尝试，并返回具体的要点作为指导；（3）以脚本为中心的自我角色，它吸收所有程序信息以生成可执行的脚本，用于最终答案预测。广泛的实验表明，这种设计方案不仅能更好地增强推理能力，还能无缝地与当前模型集成，从而实现优越的性能。 

---
# Explainability-Driven Quality Assessment for Rule-Based Systems 

**Title (ZH)**: 基于可解释性的规则驱动系统质量评估 

**Authors**: Oshani Seneviratne, Brendan Capuzzo, William Van Woensel  

**Link**: [PDF](https://arxiv.org/pdf/2502.01253)  

**Abstract**: This paper introduces an explanation framework designed to enhance the quality of rules in knowledge-based reasoning systems based on dataset-driven insights. The traditional method for rule induction from data typically requires labor-intensive labeling and data-driven learning. This framework provides an alternative and instead allows for the data-driven refinement of existing rules: it generates explanations of rule inferences and leverages human interpretation to refine rules. It leverages four complementary explanation types: trace-based, contextual, contrastive, and counterfactual, providing diverse perspectives for debugging, validating, and ultimately refining rules. By embedding explainability into the reasoning architecture, the framework enables knowledge engineers to address inconsistencies, optimize thresholds, and ensure fairness, transparency, and interpretability in decision-making processes. Its practicality is demonstrated through a use case in finance. 

**Abstract (ZH)**: 本文介绍了一种基于数据驱动洞见的规则解释框架，旨在提高基于知识推理系统的规则质量。传统从数据中归纳规则的方法通常需要耗时的数据标注和数据驱动的学习。该框架提供了一种替代方案，允许基于数据驱动的方式细化现有规则：它生成规则推理的解释，并利用人类对这些解释的解读来细化规则。该框架利用了四种互补的解释类型：追踪型、上下文型、对照型和反事实型，为调试、验证和最终细化规则提供了多维视角。通过将解释性嵌入到推理架构中，该框架使知识工程师能够解决不一致性、优化阈值，并确保决策过程中的公平性、透明性和可解释性。通过在金融领域的应用案例，证明了该框架的实际可行性。 

---
# Efficient rule induction by ignoring pointless rules 

**Title (ZH)**: 通过忽略无意义规则实现高效的规则归纳 

**Authors**: Andrew Cropper, David M. Cerna  

**Link**: [PDF](https://arxiv.org/pdf/2502.01232)  

**Abstract**: The goal of inductive logic programming (ILP) is to find a set of logical rules that generalises training examples and background knowledge. We introduce an ILP approach that identifies pointless rules. A rule is pointless if it contains a redundant literal or cannot discriminate against negative examples. We show that ignoring pointless rules allows an ILP system to soundly prune the hypothesis space. Our experiments on multiple domains, including visual reasoning and game playing, show that our approach can reduce learning times by 99% whilst maintaining predictive accuracies. 

**Abstract (ZH)**: 归纳逻辑编程（ILP）的目标是找出一套逻辑规则，这些规则能够泛化训练示例和背景知识。我们介绍了一种ILP方法，用于识别无用规则。一条规则被认为是无用的，如果它包含冗余的文字单元，或者无法区分负面示例。我们展示了忽视无用规则能够使ILP系统在假设空间上进行可靠地精简。我们在视觉推理和游戏玩等领域进行的实验表明，我们的方法可以在保持预测准确率的同时将学习时间减少99%。 

---
# Skewed Memorization in Large Language Models: Quantification and Decomposition 

**Title (ZH)**: 大型语言模型中的偏斜记忆：量化与分解 

**Authors**: Hao Li, Di Huang, Ziyu Wang, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2502.01187)  

**Abstract**: Memorization in Large Language Models (LLMs) poses privacy and security risks, as models may unintentionally reproduce sensitive or copyrighted data. Existing analyses focus on average-case scenarios, often neglecting the highly skewed distribution of memorization. This paper examines memorization in LLM supervised fine-tuning (SFT), exploring its relationships with training duration, dataset size, and inter-sample similarity. By analyzing memorization probabilities over sequence lengths, we link this skewness to the token generation process, offering insights for estimating memorization and comparing it to established metrics. Through theoretical analysis and empirical evaluation, we provide a comprehensive understanding of memorization behaviors and propose strategies to detect and mitigate risks, contributing to more privacy-preserving LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的记忆化会给隐私和安全性带来风险，因为模型可能无意中重新产生了敏感或受版权保护的数据。现有分析主要集中在平均情况场景上，往往忽视了记忆化高度偏斜的分布。本文探讨了在监督微调（SFT）过程中LLMs的记忆化现象，研究其与训练时长、数据集大小以及样本间相似度之间的关系。通过分析序列长度上的记忆化概率，我们将这种偏斜与token生成过程联系起来，提供了一种估算和比较记忆化的见解，并将其与现有指标进行对比。通过理论分析和实证评估，本文提供了对记忆化行为的全面理解，并提出了一系列检测和缓解风险的策略，从而促进更具有隐私保护性的LLMs。 

---
# Scalable Precise Computation of Shannon Entropy 

**Title (ZH)**: 可扩展的精确计算香农熵方法 

**Authors**: Yong Lai, Haolong Tong, Zhenghang Xu, Minghao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.01160)  

**Abstract**: Quantitative information flow analyses (QIF) are a class of techniques for measuring the amount of confidential information leaked by a program to its public outputs.
Shannon entropy is an important method to quantify the amount of leakage in QIF.
This paper focuses on the programs modeled in Boolean constraints and optimizes the two stages of the Shannon entropy computation to implement a scalable precise tool PSE.
In the first stage, we design a knowledge compilation language called \ADDAND that combines Algebraic Decision Diagrams and conjunctive decomposition.
\ADDAND avoids enumerating possible outputs of a program and supports tractable entropy computation.
In the second stage, we optimize the model counting queries that are used to compute the probabilities of outputs.
We compare PSE with the state-of-the-art probably approximately correct tool EntropyEstimation, which was shown to significantly outperform the existing precise tools.
The experimental results demonstrate that PSE solved 55 more benchmarks compared to EntropyEstimation in a total of 441. For 98% of the benchmarks that both PSE and EntropyEstimation solved, PSE is at least $10\times$ as efficient as EntropyEstimation. 

**Abstract (ZH)**: 定量信息流动分析（QIF）是一类用于衡量程序通过公共输出泄漏的机密信息量的技术。
香农熵是一种重要的方法，用于量化QIF中的泄漏量。
本文专注于布尔约束下的程序建模，并优化了香农熵计算的两个阶段，实现了一个可扩展且精确的工具PSE。
在第一阶段，我们设计了一种名为\ADDAND的知识编译语言，结合了代数决策图和合取分解。
\ADDAND避免列举程序的所有可能输出，并支持可处理的熵计算。
在第二阶段，我们优化了用于计算输出概率的模型计数查询。
我们将在两个方面进行比较：PSE与最先进的可能近似正确工具EntropyEstimation。EntropyEstimation已证明显著优于现有精确工具。
实验结果表明，PSE在总共441个基准测试中解决了55个更多的问题。对于PSE和EntropyEstimation都解决的98%的基准测试，PSE至少比EntropyEstimation快10倍。 

---
# DeepRAG: Thinking to Retrieval Step by Step for Large Language Models 

**Title (ZH)**: DeepRAG：逐步思考以进行大规模语言模型的检索 

**Authors**: Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.01142)  

**Abstract**: Large Language Models (LLMs) have shown remarkable potential in reasoning while they still suffer from severe factual hallucinations due to timeliness, accuracy, and coverage of parametric knowledge. Meanwhile, integrating reasoning with retrieval-augmented generation (RAG) remains challenging due to ineffective task decomposition and redundant retrieval, which can introduce noise and degrade response quality. In this paper, we propose DeepRAG, a framework that models retrieval-augmented reasoning as a Markov Decision Process (MDP), enabling strategic and adaptive retrieval. By iteratively decomposing queries, DeepRAG dynamically determines whether to retrieve external knowledge or rely on parametric reasoning at each step. Experiments show that DeepRAG improves retrieval efficiency while improving answer accuracy by 21.99%, demonstrating its effectiveness in optimizing retrieval-augmented reasoning. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在推理方面展现了显著的潜力，但仍然受到参数知识的时效性、准确性和覆盖范围的影响，从而导致严重的事实幻觉。同时，将推理与检索增强生成（RAG）集成仍然具有挑战性，这主要是由于任务分解不有效以及冗余检索可能会引入噪声并降低响应质量。在本文中，我们提出了一种名为DeepRAG的框架，该框架将检索增强推理建模为马尔可夫决策过程（MDP），从而实现策略性和自适应的检索。通过迭代分解查询，DeepRAG在每一步动态决定是检索外部知识还是依靠参数推理。实验结果表明，DeepRAG在提高检索效率的同时将答案准确性提高了21.99%，证明了其在优化检索增强推理方面的有效性。 

---
# Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning 

**Title (ZH)**: 挑食的大型语言模型和不可靠的推荐模型：指令微调后安全对齐的实证研究 

**Authors**: Guanlin Li, Kangjie Chen, Shangwei Guo, Jie Zhang, Han Qiu, Chao Zhang, Guoyin Wang, Tianwei Zhang, Jiwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.01116)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools for addressing a wide range of general inquiries and tasks. Despite this, fine-tuning aligned LLMs on smaller, domain-specific datasets, critical to adapting them to specialized tasks, can inadvertently degrade their safety alignment, even when the datasets are benign. This phenomenon makes models more susceptible to providing inappropriate responses. In this study, we systematically examine the factors contributing to safety alignment degradation in benign fine-tuning scenarios. Our analysis identifies three critical factors affecting aligned LLMs: answer structure, identity calibration, and role-play. Additionally, we evaluate the reliability of state-of-the-art reward models (RMs), which are often used to guide alignment processes. Our findings reveal that these RMs frequently fail to accurately reflect human preferences regarding safety, underscoring their limitations in practical applications. By uncovering these challenges, our work highlights the complexities of maintaining safety alignment during fine-tuning and offers guidance to help developers balance utility and safety in LLMs. Datasets and fine-tuning code used in our experiments can be found in this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已发展成为解决广泛一般性查询和任务的强大工具。然而，在将对齐的大规模语言模型微调到较小的领域特定数据集时，虽然这些数据集通常是无害的，但却可能会无意中降低模型的安全对齐程度。这一现象使得模型更容易提供不合适的回应。在本研究中，我们系统地分析了导致在无害微调场景中安全对齐降级的因素。我们的分析确定了影响对齐的大规模语言模型的三个关键因素：答案结构、身份校准以及角色扮演。此外，我们还评估了当前最先进的奖励模型（RMs）的可靠性，这些模型常用于引导对齐过程。我们的研究发现，这些RMs经常无法准确反映人类对安全性的偏好，这突显了它们在实际应用中的局限性。通过揭示这些挑战，我们的研究揭示了在微调过程中保持安全对齐的复杂性，并为帮助开发者在LLMs中平衡有用性与安全性提供了指导。在实验中使用的数据集和微调代码可以在此访问：[这个链接]。 

---
# ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning 

**Title (ZH)**: ZebraLogic：大规模语言模型在逻辑推理领域的 scalability 极限探究 

**Authors**: Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.01100)  

**Abstract**: We investigate the logical reasoning capabilities of large language models (LLMs) and their scalability in complex non-monotonic reasoning. To this end, we introduce ZebraLogic, a comprehensive evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs). ZebraLogic enables the generation of puzzles with controllable and quantifiable complexity, facilitating a systematic study of the scaling limits of models such as Llama, o1 models, and DeepSeek-R1. By encompassing a broad range of search space complexities and diverse logical constraints, ZebraLogic provides a structured environment to evaluate reasoning under increasing difficulty.
Our results reveal a significant decline in accuracy as problem complexity grows -- a phenomenon we term the curse of complexity. This limitation persists even with larger models and increased inference-time computation, suggesting inherent constraints in current LLM reasoning capabilities. Additionally, we explore strategies to enhance logical reasoning, including Best-of-N sampling, backtracking mechanisms, and self-verification prompts. Our findings offer critical insights into the scalability of LLM reasoning, highlight fundamental limitations, and outline potential directions for improvement. 

**Abstract (ZH)**: 我们研究了大型语言模型（LLMs）的逻辑推理能力及其在复杂非单调推理中的可扩展性。为此，我们提出了ZebraLogic，这是一个全面的评估框架，用于评估LLM在基于约束满足问题（CSPs）的逻辑网格谜题中的推理性能。ZebraLogic允许生成具有可控性和可量化复杂度的谜题，从而促进对如Llama、o1模型和DeepSeek-R1等模型的扩展极限的系统研究。通过涵盖广泛的搜索空间复杂度和多样的逻辑约束，ZebraLogic为评估随难度增加的推理能力提供了一个结构化的环境。

我们的研究结果揭示了随着问题复杂性的增加，准确性显著下降的现象——我们将其称为“复杂性诅咒”。即使使用更大规模的模型和更多的推理时间计算，这一限制仍然存在，这表明当前LLM推理能力中固有的约束。此外，我们探索了增强逻辑推理策略，包括Best-of-N采样、回溯机制以及自我验证提示。我们的发现提供了有关LLM推理扩展性的关键见解，突显了基本限制，并指出了改进的方向。 

---
# Language Models Use Trigonometry to Do Addition 

**Title (ZH)**: 语言模型使用三角函数进行加法运算 

**Authors**: Subhash Kantamneni, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2502.00873)  

**Abstract**: Mathematical reasoning is an increasingly important indicator of large language model (LLM) capabilities, yet we lack understanding of how LLMs process even simple mathematical tasks. To address this, we reverse engineer how three mid-sized LLMs compute addition. We first discover that numbers are represented in these LLMs as a generalized helix, which is strongly causally implicated for the tasks of addition and subtraction, and is also causally relevant for integer division, multiplication, and modular arithmetic. We then propose that LLMs compute addition by manipulating this generalized helix using the "Clock" algorithm: to solve $a+b$, the helices for $a$ and $b$ are manipulated to produce the $a+b$ answer helix which is then read out to model logits. We model influential MLP outputs, attention head outputs, and even individual neuron preactivations with these helices and verify our understanding with causal interventions. By demonstrating that LLMs represent numbers on a helix and manipulate this helix to perform addition, we present the first representation-level explanation of an LLM's mathematical capability. 

**Abstract (ZH)**: 数学推理日益成为大型语言模型（LLM）能力的重要指标，但我们对LLM处理甚至简单数学任务的方式仍缺乏理解。为了解决这一问题，我们反向-engineered三种中型大小的LLM如何计算加法。我们首先发现，在这些LLM中，数字以一种泛化的螺线形式表示，这种形式对加法和减法任务有强烈的因果影响，并且也对整数除法、乘法和模算术有因果相关性。我们进而提出，LLM通过使用“时钟”算法操作这种泛化的螺线来进行加法计算：为了解决$a+b$，会操作$a$和$b$对应的螺线以生成$a+b$的答案螺线，然后从该螺线中读取出模型的概似值（logits）。我们用这些螺线去建模影响性的MLP输出、注意力头输出，甚至单个神经元的预激活状态，并通过因果干预验证了我们的理解。通过展示LLM用螺线表示数字并通过操作螺线进行加法运算的方式，我们提供了关于LLM数学能力的首个表示级别解释。 

---
# Learning to Plan with Personalized Preferences 

**Title (ZH)**: 具备个性化偏好的规划学习 

**Authors**: Manjie Xu, Xinyi Yang, Wei Liang, Chi Zhang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00858)  

**Abstract**: Effective integration of AI agents into daily life requires them to understand and adapt to individual human preferences, particularly in collaborative roles. Although recent studies on embodied intelligence have advanced significantly, they typically adopt generalized approaches that overlook personal preferences in planning. We address this limitation by developing agents that not only learn preferences from few demonstrations but also learn to adapt their planning strategies based on these preferences. Our research leverages the observation that preferences, though implicitly expressed through minimal demonstrations, can generalize across diverse planning scenarios. To systematically evaluate this hypothesis, we introduce Preference-based Planning (PbP) benchmark, an embodied benchmark featuring hundreds of diverse preferences spanning from atomic actions to complex sequences. Our evaluation of SOTA methods reveals that while symbol-based approaches show promise in scalability, significant challenges remain in learning to generate and execute plans that satisfy personalized preferences. We further demonstrate that incorporating learned preferences as intermediate representations in planning significantly improves the agent's ability to construct personalized plans. These findings establish preferences as a valuable abstraction layer for adaptive planning, opening new directions for research in preference-guided plan generation and execution. 

**Abstract (ZH)**: 有效将AI代理融入日常生活中要求它们能够理解并适应个人人类偏好，特别是在协作角色中。尽管近年来关于具身智能的研究取得了显著进展，但这些研究通常采用通用的方法，忽略了规划中个人偏好这一因素。我们通过开发既能从少量示范中学到偏好，又能根据这些偏好调整其规划策略的代理来弥补这一局限。我们的研究利用了这样一个观察：尽管偏好是通过最少的示范而隐式表达的，但它们能够泛化到多种多样的规划场景中。为了系统地检验这一假设，我们引入了基于偏好规划（PbP）基准，该基准是具身环境下的基准测试，涵盖了数百种不同的偏好，从原子动作到复杂的序列。对当前最先进的方法的评估表明，虽然基于符号的方法显示出规模化的潜力，但在学习生成和执行满足个性化偏好的计划方面仍存在重大挑战。进一步的研究还表明，将学到的偏好作为规划中的中间表示可以显著提高代理构建个性化计划的能力。这些发现确立了偏好作为适应性规划中的有价值抽象层的地位，为基于偏好引导的计划生成和执行的研究开启了新方向。 

---
# Psychometric-Based Evaluation for Theorem Proving with Large Language Models 

**Title (ZH)**: 基于心理测量学的评估方法在大规模语言模型进行定理证明中的应用 

**Authors**: Jianyu Zhang, Yongwang Zhao, Long Zhang, Jilin Hu, Xiaokun Luan, Zhiwei Xu, Feng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00855)  

**Abstract**: Large language models (LLMs) for formal theorem proving have become a prominent research focus. At present, the proving ability of these LLMs is mainly evaluated through proof pass rates on datasets such as miniF2F. However, this evaluation method overlooks the varying importance of theorems. As a result, it fails to highlight the real performance disparities between LLMs and leads to high evaluation costs. This study proposes a psychometric-based evaluation method for theorem proving with LLMs, comprising two main components: Dataset Annotation and Adaptive Evaluation. First, we propose a metric calculation method to annotate the dataset with difficulty and discrimination metrics. Specifically, we annotate each theorem in the miniF2F dataset and grade them into varying difficulty levels according to the performance of LLMs, resulting in an enhanced dataset: miniF2F-Graded. Experimental results show that the difficulty grading in miniF2F-Graded better reflects the theorem difficulty perceived by LLMs. Secondly, we design an adaptive evaluation method to dynamically select the most suitable theorems for testing based on the annotated metrics and the real-time performance of LLMs. We apply this method to evaluate 10 LLMs. The results show that our method finely highlights the performance disparities between LLMs. It also reduces evaluation costs by using only 23% of the theorems in the dataset. 

**Abstract (ZH)**: 大型语言模型（LLMs）在形式定理证明中的应用已成为研究的重点。目前，这些LLMs的证明能力主要通过在miniF2F等数据集上的证明通过率进行评估。然而，这种评估方法忽视了定理之间相对重要性的差异，导致未能突出LLMs之间的实际性能差异，同时增加了评估成本。本研究提出了一种基于心理测量学的形式定理证明评价方法，主要包括两个主要组成部分：数据集注释和自适应评估。首先，我们提出了一种度量计算方法，用于为数据集添加难度和区分度指标。具体而言，我们对miniF2F数据集中的每个定理进行注释，并根据LLMs的性能将其划分为不同的难度等级，从而形成增强的数据集miniF2F-Graded。实验结果显示，miniF2F-Graded中的难度分级更能反映LLMs感知的定理难度。其次，我们设计了一种自适应评估方法，根据标注的指标和LLMs的实时性能动态选择最合适的定理进行测试。我们将此方法应用于评估10个LLMs，结果显示，我们的方法能够精细地突出LLMs之间的性能差异，且通过仅使用数据集中的23%定理即可显著降低评估成本。 

---
# RTBAgent: A LLM-based Agent System for Real-Time Bidding 

**Title (ZH)**: RTBAgent：一个基于大语言模型的实时竞价代理系统 

**Authors**: Leng Cai, Junxuan He, Yikai Li, Junjie Liang, Yuanping Lin, Ziming Quan, Yawen Zeng, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00792)  

**Abstract**: Real-Time Bidding (RTB) enables advertisers to place competitive bids on impression opportunities instantaneously, striving for cost-effectiveness in a highly competitive landscape. Although RTB has widely benefited from the utilization of technologies such as deep learning and reinforcement learning, the reliability of related methods often encounters challenges due to the discrepancies between online and offline environments and the rapid fluctuations of online bidding. To handle these challenges, RTBAgent is proposed as the first RTB agent system based on large language models (LLMs), which synchronizes real competitive advertising bidding environments and obtains bidding prices through an integrated decision-making process. Specifically, obtaining reasoning ability through LLMs, RTBAgent is further tailored to be more professional for RTB via involved auxiliary modules, i.e., click-through rate estimation model, expert strategy knowledge, and daily reflection. In addition, we propose a two-step decision-making process and multi-memory retrieval mechanism, which enables RTBAgent to review historical decisions and transaction records and subsequently make decisions more adaptive to market changes in real-time bidding. Empirical testing with real advertising datasets demonstrates that RTBAgent significantly enhances profitability. The RTBAgent code will be publicly accessible at: this https URL. 

**Abstract (ZH)**: 实时竞价（RTB）允许广告商在瞬间对展示机会进行竞争性出价，以在高度竞争的环境中追求成本效益。尽管RTB从深度学习和强化学习等技术的应用中广泛受益，但由于在线和离线环境之间的差异以及在线竞价的快速波动，相关方法的可靠性常常会遇到挑战。为了应对这些挑战，我们提出了基于大型语言模型（LLMs）的首个RTB代理系统——RTBAgent，该系统同步了真实的竞争广告竞价环境，并通过集成决策过程获取竞价价格。具体来说，通过大型语言模型（LLMs）获得推理能力后，RTBAgent进一步通过包含辅助模块（如点击率估计模型、专家策略知识和每日反思）来更加专业化地适应RTB需求。此外，我们提出了两步决策过程和多记忆检索机制，使RTBAgent能够回顾历史决策和交易记录，并在实时竞价中更加适应市场变化做出决策。实证测试使用真实的广告数据集表明，RTBAgent显著提高了盈利能力。RTBAgent的代码将在以下网址公开访问：this https URL。 

---
# Zero-Shot Warning Generation for Misinformative Multimodal Content 

**Title (ZH)**: 零样本警告生成以应对误导性的多模态内容 

**Authors**: Giovanni Pio Delvecchio, Huy Hong Nguyen, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2502.00752)  

**Abstract**: The widespread prevalence of misinformation poses significant societal concerns. Out-of-context misinformation, where authentic images are paired with false text, is particularly deceptive and easily misleads audiences. Most existing detection methods primarily evaluate image-text consistency but often lack sufficient explanations, which are essential for effectively debunking misinformation. We present a model that detects multimodal misinformation through cross-modality consistency checks, requiring minimal training time. Additionally, we propose a lightweight model that achieves competitive performance using only one-third of the parameters. We also introduce a dual-purpose zero-shot learning task for generating contextualized warnings, enabling automated debunking and enhancing user comprehension. Qualitative and human evaluations of the generated warnings highlight both the potential and limitations of our approach. 

**Abstract (ZH)**: 广泛流传的错误信息在社会上引起了重大关切。脱离上下文的错误信息，即真实图片与虚假文本配对的情况，尤为具有欺骗性，容易误导受众。现有的大多数检测方法主要评估图像-文本的一致性，但往往缺乏充分的解释，而这对于有效驳斥错误信息至关重要。我们提出了一种通过跨模态一致性检查来检测多模态错误信息的模型，该模型能在极短的训练时间内完成检测。此外，我们提出了一种轻量级模型，仅使用三分之一的参数就能达到竞争性的性能。我们还引入了一项双重目的的零样本学习任务，用于生成上下文化的警告，从而实现自动化驳斥并增强用户理解。生成的警告的质量和人类评估揭示了我们方法的潜在价值和局限性。 

---
# Selective Response Strategies for GenAI 

**Title (ZH)**: 面向生成性人工智能的选择性响应策略 

**Authors**: Boaz Taitler, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2502.00729)  

**Abstract**: The rise of Generative AI (GenAI) has significantly impacted human-based forums like Stack Overflow, which are essential for generating high-quality data. This creates a negative feedback loop, hindering the development of GenAI systems, which rely on such data to provide accurate responses. In this paper, we provide a possible remedy: A novel strategy we call selective response. Selective response implies that GenAI could strategically provide inaccurate (or conservative) responses to queries involving emerging topics and novel technologies, thereby driving users to use human-based forums like Stack Overflow. We show that selective response can potentially have a compounding effect on the data generation process, increasing both GenAI's revenue and user welfare in the long term. From an algorithmic perspective, we propose an approximately optimal approach to maximize GenAI's revenue under social welfare constraints. From a regulatory perspective, we derive sufficient and necessary conditions for selective response to improve welfare improvements. 

**Abstract (ZH)**: 生成式AI（GenAI）的兴起对像Stack Overflow这样的人力论坛产生了显著影响，这些论坛是生成高质量数据的重要来源。这创造了一个负反馈循环，阻碍了GenAI系统的开发，因为这些系统依赖于这类数据来提供准确的回答。在本文中，我们提供了一个可能的解决办法：我们提出了一种新颖的战略，称为选择性响应。选择性响应意味着GenAI可以在涉及新兴主题和新技术的查询中提供不准确（或保守）的回答，从而促使用户使用类似Stack Overflow这样的人力论坛。我们表明，选择性响应可能会在数据生成过程中产生累积效应，从长远来看，增加GenAI的收入并提升用户福利。从算法角度来看，我们提出了一种近似最优的方法，在社会福利约束下最大化GenAI的收入。从监管角度来看，我们推导出了选择性响应可提升福利改进的充分必要条件。 

---
# Perspectives for Direct Interpretability in Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: 直接可解释性在多agent深度强化学习中的前景 

**Authors**: Yoann Poupart, Aurélie Beynier, Nicolas Maudet  

**Link**: [PDF](https://arxiv.org/pdf/2502.00726)  

**Abstract**: Multi-Agent Deep Reinforcement Learning (MADRL) was proven efficient in solving complex problems in robotics or games, yet most of the trained models are hard to interpret. While learning intrinsically interpretable models remains a prominent approach, its scalability and flexibility are limited in handling complex tasks or multi-agent dynamics. This paper advocates for direct interpretability, generating post hoc explanations directly from trained models, as a versatile and scalable alternative, offering insights into agents' behaviour, emergent phenomena, and biases without altering models' architectures. We explore modern methods, including relevance backpropagation, knowledge edition, model steering, activation patching, sparse autoencoders and circuit discovery, to highlight their applicability to single-agent, multi-agent, and training process challenges. By addressing MADRL interpretability, we propose directions aiming to advance active topics such as team identification, swarm coordination and sample efficiency. 

**Abstract (ZH)**: 多智能体深度强化学习（Multi-Agent Deep Reinforcement Learning, MADRL）在解决机器人或游戏中的复杂问题上已被证明是有效的，但大多数训练好的模型难以解释。尽管学习固有可解释的模型仍然是一个显着的方法，但在处理复杂任务或多智能体动态方面，这种方法的扩展性和灵活性仍受到限制。本文提倡直接可解释性，直接从训练好的模型中生成事后解释，作为一种灵活且可扩展的替代方案，无需修改模型架构即可提供对智能体行为、涌现现象和偏差的深入见解。我们探讨了现代方法，包括相关反向传播、知识编辑、模型引导、激活补丁、稀疏自动编码器和电路发现，以展示其在单智能体、多智能体及训练过程挑战方面的适用性。通过对MADRL可解释性的探讨，我们提出了一些方向，旨在推进团队识别、群集协调和样本效率等活跃研究领域的发展。 

---
# MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models 

**Title (ZH)**: MM-IQ: 多模态模型中人类似抽象与推理的基准测试 

**Authors**: Huanqia Cai, Yijun Yang, Winston Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00698)  

**Abstract**: IQ testing has served as a foundational methodology for evaluating human cognitive capabilities, deliberately decoupling assessment from linguistic background, language proficiency, or domain-specific knowledge to isolate core competencies in abstraction and reasoning. Yet, artificial intelligence research currently lacks systematic benchmarks to quantify these critical cognitive dimensions in multimodal systems. To address this critical gap, we propose MM-IQ, a comprehensive evaluation framework comprising 2,710 meticulously curated test items spanning 8 distinct reasoning paradigms.
Through systematic evaluation of leading open-source and proprietary multimodal models, our benchmark reveals striking limitations: even state-of-the-art architectures achieve only marginally superior performance to random chance (27.49% vs. 25% baseline accuracy). This substantial performance chasm highlights the inadequacy of current multimodal systems in approximating fundamental human reasoning capacities, underscoring the need for paradigm-shifting advancements to bridge this cognitive divide. 

**Abstract (ZH)**: 智商测试一直作为一种基础方法，用于评估人类的认知能力，故意将评估与语言背景、语言熟练程度或领域特定知识脱钩，以分离出抽象和推理的核心能力。然而，当前的人工智能研究缺乏系统性的基准来量化多模态系统中的这些关键认知维度。为解决这一关键差距，我们提出了MM-IQ，这是一个包含2,710个精心筛选测试项目的综合评估框架，覆盖8种不同的推理范式。

通过对领先的开源和专有多模态模型进行系统性评估，我们的基准揭示出显著的局限性：即使最先进的架构也仅比随机猜测略好（准确率为27.49%，而基线准确率为25%）。这一显著的性能差距突显了当前多模态系统在逼近基本的人类推理能力方面的不足，强调了需要范式转变性的进展来弥合这一认知鸿沟。 

---
# Learning Autonomous Code Integration for Math Language Models 

**Title (ZH)**: 学习自主代码集成的数学语言模型 

**Authors**: Haozhe Wang, Long Li, Chao Qu, Fengming Zhu, Weidi Xu, Wei Chu, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.00691)  

**Abstract**: Recent research on tool integration for math Large Language Models (LLMs) aims to combine complementary strengths of chain-of-thought (CoT) reasoning and code execution. However, we discover a critical limitation: current tool-integrated math LLMs rely on externally dictated instructions to decide whether to use CoT or code, lacking the autonomy to choose the most appropriate method independently. This prompts us to study \emph{Autonomous Code integration} for math LLMs, which enables models to \emph{independently} develop their own methodology-selection strategy in the absence of reliable supervision. To address this challenge, we propose an innovative Expectation-Maximization (EM) formulation that refines the model's decision-making through the exploration of its capabilities. This framework alternates between (a) computing a reference strategy that improves the model's belief over its capabilities through self-exploration, and (b) updating the model based on the refined belief. We further enhance this framework with an efficient implementation, incorporating a novel data synthesis strategy and off-policy reinforcement learning. Extensive experiments demonstrate that our approach, using only a public query set, significantly boosts the performance of existing math LLMs, raising accuracy by nearly 20\% to 65.28\% on the challenging MATH benchmark, while reducing code executions by up to 65\% . 

**Abstract (ZH)**: 近年来，关于数学大型语言模型（LLM）工具集成的研究旨在结合链式推理（CoT）和代码执行的互补优势。然而，我们发现一个关键的局限性：目前的工具集成数学LLM依赖于外部指令来决定是否使用CoT或代码执行，缺乏独立选择最恰当方法的能力。这促使我们研究数学LLM的自主代码集成，让模型能够在缺乏可靠监督的情况下，自主开发其方法选择策略。为应对这一挑战，我们提出了一种创新的期望最大化（EM）公式，通过探索模型的能力来提高其决策能力。该框架交替进行以下两个步骤：(a) 计算一个参考策略，该策略通过自我探索提高模型对其能力的信任度；(b) 根据更新后的信任度调整模型。此外，我们还通过高效的实现方式进一步完善了该框架，引入了一种新颖的数据合成策略和离策略强化学习方法。广泛的实验表明，我们的方法仅使用公开查询集就能显著提升现有数学LLM的表现，MATH基准在困难任务上的准确率几乎提高了20%，达到了65.28%，同时减少了高达65%的代码执行次数。 

---
# LLM-based event log analysis techniques: A survey 

**Title (ZH)**: 基于大型语言模型的事件日志分析技术：一种综述 

**Authors**: Siraaj Akhtar, Saad Khan, Simon Parkinson  

**Link**: [PDF](https://arxiv.org/pdf/2502.00677)  

**Abstract**: Event log analysis is an important task that security professionals undertake. Event logs record key information on activities that occur on computing devices, and due to the substantial number of events generated, they consume a large amount of time and resources to analyse. This demanding and repetitive task is also prone to errors. To address these concerns, researchers have developed automated techniques to improve the event log analysis process. Large Language Models (LLMs) have recently demonstrated the ability to successfully perform a wide range of tasks that individuals would usually partake in, to high standards, and at a pace and degree of complexity that outperform humans. Due to this, researchers are rapidly investigating the use of LLMs for event log analysis. This includes fine-tuning, Retrieval-Augmented Generation (RAG) and in-context learning, which affect performance. These works demonstrate good progress, yet there is a need to understand the developing body of knowledge, identify commonalities between works, and identify key challenges and potential solutions to further developments in this domain. This paper aims to survey LLM-based event log analysis techniques, providing readers with an in-depth overview of the domain, gaps identified in previous research, and concluding with potential avenues to explore in future. 

**Abstract (ZH)**: 事件日志分析是一项重要的安全专业人员任务。事件日志记录了计算设备上发生的活动的关键信息，但由于生成的事件数量庞大，分析这些日志会消耗大量时间和资源。这一任务既耗时又重复，并且容易出错。为了解决这些问题，研究人员开发了自动化技术以改进事件日志分析过程。大语言模型（LLMs）最近已经展示出能够高效、高质量地完成人类通常会参与的多种任务，并且在速度和复杂性方面超越人类。由于这一点，研究人员正在迅速探索使用LLMs进行事件日志分析的方法。这包括微调、检索增强生成（RAG）和上下文学习等方法，它们对性能产生影响。尽管这些研究展示了良好的进展，但仍有必要理解这一领域的新兴知识体系，识别不同研究之间的共通之处，并识别关键挑战和潜在解决方案以推动该领域进一步发展。本文旨在回顾基于LLM的事件日志分析技术，为读者提供该领域的深入概述，指出先前研究中的空白，并总结出未来研究的潜在途径。 

---
# Agency in the Age of AI 

**Title (ZH)**: AI时代的代理问题 

**Authors**: Samarth Swarup  

**Link**: [PDF](https://arxiv.org/pdf/2502.00648)  

**Abstract**: There is significant concern about the impact of generative AI on society. Modern AI tools are capable of generating ever more realistic text, images, and videos, and functional code, from minimal prompts. Accompanying this rise in ability and usability, there is increasing alarm about the misuses to which these tools can be put, and the intentional and unintentional harms to individuals and society that may result. In this paper, we argue that \emph{agency} is the appropriate lens to study these harms and benefits, but that doing so will require advancement in the theory of agency, and advancement in how this theory is applied in (agent-based) models. 

**Abstract (ZH)**: 关于生成式AI对社会的影响，存在显著的关切。现代AI工具能够从少量提示生成越来越逼真的文本、图像和视频，以及功能性代码。伴随着这种能力和使用性的提升，人们对这些工具可能被滥用的担忧也在增加，并可能对个人和社会造成有意或无意的危害。在本文中，我们主张应该使用“agency”（自主权/能力）这一视角来研究这些危害和益处，但要实现这一目标，需要在自主权理论方面进行深入探讨，并改进这种理论在基于代理的模型中的应用。 

---
# CollabLLM: From Passive Responders to Active Collaborators 

**Title (ZH)**: CollabLLM：从被动回应者到主动合作者 

**Authors**: Shirley Wu, Michel Galley, Baolin Peng, Hao Cheng, Gavin Li, Yao Dou, Weixin Cai, James Zou, Jure Leskovec, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.00640)  

**Abstract**: Large Language Models are typically trained with next-turn rewards, limiting their ability to optimize for long-term interaction. As a result, they often respond passively to ambiguous or open-ended user requests, failing to help users reach their ultimate intents and leading to inefficient conversations. To address these limitations, we introduce CollabLLM, a novel and general training framework that enhances multiturn human-LLM collaboration. Its key innovation is a collaborative simulation that estimates the long-term contribution of responses using Multiturn-aware Rewards. By reinforcement fine-tuning these rewards, CollabLLM goes beyond responding to user requests, and actively uncovers user intent and offers insightful suggestions-a key step towards more human-centered AI. We also devise a multiturn interaction benchmark with three challenging tasks such as document creation. CollabLLM significantly outperforms our baselines with averages of 18.5% higher task performance and 46.3% improved interactivity by LLM judges. Finally, we conduct a large user study with 201 judges, where CollabLLM increases user satisfaction by 17.6% and reduces user spent time by 10.4%. 

**Abstract (ZH)**: 大型语言模型通常使用下一轮奖励进行训练，这限制了它们优化长期交互的能力。因此，它们往往对模糊或开放性的用户请求作出被动响应，无法帮助用户达成最终意图，导致对话效率低下。为解决这些问题，我们提出了一种名为CollabLLM的新颖且通用的训练框架，以增强多轮人类-语言模型协作。其核心创新在于一种协作模拟，通过多轮次意识奖励估算响应的长期贡献。通过强化学习精细调整这些奖励，CollabLLM不仅响应用户请求，还能主动揭示用户意图并提供有意义的建议，这是迈向更加用户导向的人工智能的关键步骤。我们还设计了一个多轮次交互基准，其中包括三个具有挑战性的任务，如文档创建。CollabLLM在平均任务性能上显著高于我们的基线，提高了18.5%，并且在LLM评判者看来，对话互动性提高了46.3%。最后，我们在201名评判者参与的大规模用户研究中，CollabLLM将用户满意度提高了17.6%，并将用户耗时减少了10.4%。 

---
# Lipschitz Lifelong Monte Carlo Tree Search for Mastering Non-Stationary Tasks 

**Title (ZH)**: 基于Lipschitz Lip⁺终身蒙特卡罗树搜索掌握非稳态任务 

**Authors**: Zuyuan Zhang, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00633)  

**Abstract**: Monte Carlo Tree Search (MCTS) has proven highly effective in solving complex planning tasks by balancing exploration and exploitation using Upper Confidence Bound for Trees (UCT). However, existing work have not considered MCTS-based lifelong planning, where an agent faces a non-stationary series of tasks -- e.g., with varying transition probabilities and rewards -- that are drawn sequentially throughout the operational lifetime. This paper presents LiZero for Lipschitz lifelong planning using MCTS. We propose a novel concept of adaptive UCT (aUCT) to transfer knowledge from a source task to the exploration/exploitation of a new task, depending on both the Lipschitz continuity between tasks and the confidence of knowledge in in Monte Carlo action sampling. We analyze LiZero's acceleration factor in terms of improved sampling efficiency and also develop efficient algorithms to compute aUCT in an online fashion by both data-driven and model-based approaches, whose sampling complexity and error bounds are also characterized. Experiment results show that LiZero significantly outperforms existing MCTS and lifelong learning baselines in terms of much faster convergence (3$\sim$4x) to optimal rewards. Our results highlight the potential of LiZero to advance decision-making and planning in dynamic real-world environments. 

**Abstract (ZH)**: 蒙特卡洛树搜索（MCTS）已被证明在通过上下文不确定性树（UCT）平衡探索和利用解决复杂规划任务方面极为有效。然而，现有工作尚未考虑基于MCTS的终身规划问题，其中智能体需要面对一系列非稳态的任务——例如具有变化的转移概率和回报的任务，这些任务在整个操作生命周期中陆续出现。本文提出了LiZero，用于利用MCTS实现Lipschitz终身规划。我们提出了一种新的自适应UCT（aUCT）概念，它能够在依赖于任务之间Lipschitz连续性和Monte Carlo动作采样中知识的信心水平的基础上，将源任务的知识转移到新任务的探索和利用中。我们分析了LiZero加速因子，即改进的采样效率，还开发了能够在数据驱动和模型驱动两种方法下在线计算aUCT的高效算法，并对这些算法的采样复杂度和误差边界进行了分析。实验结果表明，与现有的MCTS和终身学习基准方法相比，LiZero在更快达到最优回报方面（3～4倍）表现出显著优势。我们的结果突显了LiZero在动态现实环境中的决策和规划方面的发展潜力。 

---
# Advanced Weakly-Supervised Formula Exploration for Neuro-Symbolic Mathematical Reasoning 

**Title (ZH)**: 高级弱监督公式探索在神经符号数学推理中的应用 

**Authors**: Yuxuan Wu, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2502.00629)  

**Abstract**: In recent years, neuro-symbolic methods have become a popular and powerful approach that augments artificial intelligence systems with the capability to perform abstract, logical, and quantitative deductions with enhanced precision and controllability. Recent studies successfully performed symbolic reasoning by leveraging various machine learning models to explicitly or implicitly predict intermediate labels that provide symbolic instructions. However, these intermediate labels are not always prepared for every task as a part of training data, and pre-trained models, represented by Large Language Models (LLMs), also do not consistently generate valid symbolic instructions with their intrinsic knowledge. On the other hand, existing work developed alternative learning techniques that allow the learning system to autonomously uncover optimal symbolic instructions. Nevertheless, their performance also exhibits limitations when faced with relatively huge search spaces or more challenging reasoning problems. In view of this, in this work, we put forward an advanced practice for neuro-symbolic reasoning systems to explore the intermediate labels with weak supervision from problem inputs and final outputs. Our experiments on the Mathematics dataset illustrated the effectiveness of our proposals from multiple aspects. 

**Abstract (ZH)**: 近年来，神经符号方法已成为一种流行且强大的手段，能够增强人工智能系统，使其能够执行更精确和可控的抽象、逻辑和量化推理。近期的研究通过利用各种机器学习模型显式或隐式预测中间标签，成功实现了符号推理。然而，这些中间标签并非所有任务在训练数据中都准备齐全，大型语言模型（LLMs）等预训练模型也不能一贯地生成有效的符号指令。另一方面，现有工作开发了替代学习技术，使学习系统能够自主发现最优的符号指令。但是，当面对更大的搜索空间或更具挑战性的推理问题时，它们也表现出一定的局限性。鉴于此，本文提出了一个先进的神经符号推理系统实践，以从问题输入和最终输出中进行弱监督下的中间标签探索。我们在数学数据集上的实验从多个方面验证了我们提议的有效性。 

---
# Understanding Multimodal LLMs Under Distribution Shifts: An Information-Theoretic Approach 

**Title (ZH)**: 在分布偏移下理解多模态大规模语言模型：一种信息论方法 

**Authors**: Changdae Oh, Zhen Fang, Shawn Im, Xuefeng Du, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.00577)  

**Abstract**: Multimodal large language models (MLLMs) have shown promising capabilities but struggle under distribution shifts, where evaluation data differ from instruction tuning distributions. Although previous works have provided empirical evaluations, we argue that establishing a formal framework that can characterize and quantify the risk of MLLMs is necessary to ensure the safe and reliable application of MLLMs in the real world. By taking an information-theoretic perspective, we propose the first theoretical framework that enables the quantification of the maximum risk of MLLMs under distribution shifts. Central to our framework is the introduction of Effective Mutual Information (EMI), a principled metric that quantifies the relevance between input queries and model responses. We derive an upper bound for the EMI difference between in-distribution (ID) and out-of-distribution (OOD) data, connecting it to visual and textual distributional discrepancies. Extensive experiments on real benchmark datasets, spanning 61 shift scenarios empirically validate our theoretical insights. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）展现了强大的能力，但在分布转移的情况下却表现不佳，即评估数据与指令调优分布不同。尽管之前的工作已经提供了实证评估，但我们认为有必要建立一个形式化的框架，能够表征和量化MLLMs的风险，以确保MLLMs在实际应用中的安全性和可靠性。从信息论的角度出发，我们提出了第一个理论框架，能够在分布转移的情况下量化MLLMs的最大风险。该框架的核心在于引入有效的互信息（EMI），这是一种原则性的度量标准，用于量化输入查询与模型响应的相关性。我们推导了EMI差异的上界，在视觉和文本分布差异之间建立了联系。通过对六个分布转移场景的实用基准数据集进行广泛实验，我们的理论见解得到了实证验证。 

---
# Who's the MVP? A Game-Theoretic Evaluation Benchmark for Modular Attribution in LLM Agents 

**Title (ZH)**: 谁是MVP？一种基于博弈论的模块化归因评估基准应用于大语言模型代理 

**Authors**: Yingxuan Yang, Bo Huang, Siyuan Qi, Chao Feng, Haoyi Hu, Yuxuan Zhu, Jinbo Hu, Haoran Zhao, Ziyi He, Xiao Liu, Zongyu Wang, Lin Qiu, Xuezhi Cao, Xunliang Cai, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00510)  

**Abstract**: Large Language Model (LLM) agents frameworks often employ modular architectures, incorporating components such as planning, reasoning, action execution, and reflection to tackle complex tasks. However, quantifying the contribution of each module to overall system performance remains a significant challenge, impeding optimization and interpretability. To address this, we introduce CapaBench (Capability-level Assessment Benchmark), an evaluation framework grounded in cooperative game theory's Shapley Value, which systematically measures the marginal impact of individual modules and their interactions within an agent's architecture. By replacing default modules with test variants across all possible combinations, CapaBench provides a principle method for attributing performance contributions. Key contributions include: (1) We are the first to propose a Shapley Value-based methodology for quantifying the contributions of capabilities in LLM agents; (2) Modules with high Shapley Values consistently lead to predictable performance gains when combined, enabling targeted optimization; and (3) We build a multi-round dataset of over 1,000 entries spanning diverse domains and practical task scenarios, enabling comprehensive evaluation of agent capabilities. CapaBench bridges the gap between component-level evaluation and holistic system assessment, providing actionable insights for optimizing modular LLM agents and advancing their deployment in complex, real-world scenarios. 

**Abstract (ZH)**: 大型语言模型（LLM）代理框架通常采用模块化架构，包括计划、推理、动作执行和反思等组件，以应对复杂的任务。然而，量化每个模块对整体系统性能的贡献仍然是一个重大挑战，影响了优化和可解释性。为解决这一问题，我们引入了CapaBench（能力级评估基准），该框架基于合作博弈论中的Shapley值，系统性地度量了个体模块及其在代理架构内交互的边际影响。通过在所有可能的组合中用测试变体替换默认模块，CapaBench 提供了一种原则性的方法来归因性能贡献。主要贡献包括：

1. 我们首次提出了一种基于Shapley值的方法，用于量化LLM代理中的能力贡献；
2. 带有较高Shapley值的模块在组合时可以预测性地带来性能提升，从而实现有目标的优化；
3. 我们构建了一个包含超过1,000个条目的多轮数据集，覆盖了多种不同的领域和实际任务场景，使代理能力的全面评估成为可能。

CapaBench架起了组件级评估与整体系统评估之间的桥梁，提供了关于优化模块化LLM代理和在复杂实际场景中推广它们的实用见解。 

---
# Discovering Directly-Follows Graph Model for Acyclic Processes 

**Title (ZH)**: 探索无环过程的直接跟随图模型 

**Authors**: Nikita Shaimov, Irina Lomazova, Alexey Mitsyuk  

**Link**: [PDF](https://arxiv.org/pdf/2502.00499)  

**Abstract**: Process mining is the common name for a range of methods and approaches aimed at analysing and improving processes. Specifically, methods that aim to derive process models from event logs fall under the category of process discovery. Within the range of processes, acyclic processes form a distinct category. In such processes, previously performed actions are not repeated, forming chains of unique actions. However, due to differences in the order of actions, existing process discovery methods can provide models containing cycles even if a process is acyclic. This paper presents a new process discovery algorithm that allows to discover acyclic DFG models for acyclic processes. A model is discovered by partitioning an event log into parts that provide acyclic DFG models and merging them while avoiding the formation of cycles. The resulting algorithm was tested both on real-life and artificial event logs. Absence of cycles improves model visual clarity and precision, also allowing to apply cycle-sensitive methods or visualisations to the model. 

**Abstract (ZH)**: 过程挖掘是用于分析和改进过程的一系列方法和方法论的通用名称。具体来说，从事件日志中推导过程模型的方法属于过程发现的范畴。在过程的范围中，无环过程形成一个独特的类别。在这种过程中，之前执行的动作不会重复，形成一系列唯一的动作链。然而，由于动作顺序的不同，现有的过程发现方法可能会为原本无环的过程生成包含循环的模型。本文提出了一种新的过程发现算法，该算法能够为无环过程发现无环的数据流图（DFG）模型。模型是通过将事件日志划分为提供无环DFG模型的部分，并在其合并过程中避免形成循环来发现的。该算法已在实际和人工事件日志上进行了测试。消除循环提高了模型的视觉清晰度和精度，并允许对模型应用循环敏感的方法或可视化方法。 

---
# MetaOpenFOAM 2.0: Large Language Model Driven Chain of Thought for Automating CFD Simulation and Post-Processing 

**Title (ZH)**: MetaOpenFOAM 2.0：由大型语言模型驱动的推理链以自动化CFD模拟及其后处理 

**Authors**: Yuxuan Chen, Xu Zhu, Hua Zhou, Zhuyin Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.00498)  

**Abstract**: Computational Fluid Dynamics (CFD) is widely used in aerospace, energy, and biology to model fluid flow, heat transfer, and chemical reactions. While Large Language Models (LLMs) have transformed various domains, their application in CFD remains limited, particularly for complex tasks like post-processing. To bridge this gap, we introduce MetaOpenFOAM 2.0, which leverages Chain of Thought (COT) decomposition and iterative verification to enhance accessibility for non-expert users through natural language inputs. Tested on a new benchmark covering simulation (fluid flow, heat transfer, combustion) and post-processing (extraction, visualization), MetaOpenFOAM 2.0 achieved an Executability score of 6.3/7 and a pass rate of 86.9%, significantly outperforming MetaOpenFOAM 1.0 (2.1/7, 0%). Additionally, it proved cost-efficient, averaging $0.15 per case. An ablation study confirmed that COT-driven decomposition and iterative refinement substantially improved task performance. Furthermore, scaling laws showed that increasing COT steps enhanced accuracy while raising token usage, aligning with LLM post-training scaling trends. These results highlight the transformative potential of LLMs in automating CFD workflows for industrial and research applications. Code is available at this https URL 

**Abstract (ZH)**: 计算流体动力学（CFD）广泛应用于航空航天、能源和生物学领域，用于模拟流体流动、热传递和化学反应。尽管大型语言模型（LLMs）已经在各个领域取得了重大突破，但在CFD领域的应用仍然有限，特别是在复杂任务如后处理方面。为了弥补这一差距，我们引入了MetaOpenFOAM 2.0，它通过利用思维链（COT）分解和迭代验证，增强了非专家用户的使用便利性，使其能够通过自然语言输入操作。MetaOpenFOAM 2.0在涵盖模拟（流体流动、热传递、燃烧）和后处理（提取、可视化）的新基准测试中，获得了6.3/7的可执行性评分和86.9%的通过率，显著优于MetaOpenFOAM 1.0（2.1/7, 0%）。此外，测试结果还显示，MetaOpenFOAM 2.0平均每个案例成本仅为0.15美元。消融实验结果表明，由思维链驱动的分解和迭代细化显著提高了任务性能。进一步的研究还发现，增加思维链步骤可以提高准确性，同时增加令牌使用量，这与大型语言模型后训练扩展趋势相符。这些结果突显了LLMs在自动化CFD工作流方面的变革潜力，特别是在工业和研究应用中。源代码可在以下地址获取：[此链接] 

---
# Doing More with Less -- Implementing Routing Strategies in Large Language Model-Based Systems: An Extended Survey 

**Title (ZH)**: “用较少资源完成更多任务——基于大规模语言模型系统的路由策略实现：一篇扩展性综述” 

**Authors**: Clovis Varangot-Reille, Christophe Bouvard, Antoine Gourru, Mathieu Ciancone, Marion Schaeffer, François Jacquenet  

**Link**: [PDF](https://arxiv.org/pdf/2502.00409)  

**Abstract**: Large Language Models (LLM)-based systems, i.e. interconnected elements that include an LLM as a central component (e.g., conversational agents), are typically monolithic static architectures that rely on a single LLM for all user queries. However, they often require different preprocessing strategies, levels of reasoning, or knowledge. Generalist LLMs (i.e. GPT-4), trained on very large multi-topic corpora, can perform well in a variety of tasks. However, they require significant financial, energy, and hardware resources that may not be justified for basic tasks. This implies potentially investing in unnecessary costs for a given query. To overcome this problem, a routing mechanism routes user queries to the most suitable components, such as smaller LLMs or experts in specific topics. This approach may improve response quality while minimising costs. Routing can be expanded to other components of the conversational agent architecture, such as the selection of optimal embedding strategies. This paper explores key considerations for integrating routing into LLM-based systems, focusing on resource management, cost definition, and strategy selection. Our main contributions include a formalisation of the problem, a novel taxonomy of existing approaches emphasising relevance and resource efficiency, and a comparative analysis of these strategies in relation to industry practices. Finally, we identify critical challenges and directions for future research. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的系统，即包含LLM作为核心组件的互联元素（例如对话代理），通常是单一静态架构，依赖单一的LLM处理所有用户查询。然而，这些系统往往需要不同的预处理策略、推理层级或知识。通用的大型语言模型（如GPT-4），在大规模多主题语料库上进行训练，可以在多种任务中表现出色。然而，这些模型需要大量的财务、能源和硬件资源，而对于基本任务来说，这种资源的投入可能并不合理。这可能意味着为了某个查询而产生不必要的成本。为了解决这个问题，可以通过路由机制将用户查询导向最适合的组件，例如更小的LLM或特定主题的专家。这种方法可能在提高响应质量的同时降低成本。路由机制还可以扩展到对话代理架构的其他组件，例如最佳嵌入策略的选择。本文探讨了将路由机制整合到基于LLM的系统中的关键考虑因素，重点关注资源管理、成本定义和策略选择。我们的主要贡献包括对问题的正式化表述、一种现有的方法分类的新方法，强调相关性和资源效率，以及这些策略与行业实践的比较分析。最后，我们确定了关键挑战并指出了未来研究的方向。 

---
# ALU: Agentic LLM Unlearning 

**Title (ZH)**: ALU：自主的LLM去学习（或ALU：自主的大型语言模型去学习） 

**Authors**: Debdeep Sanyal, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2502.00406)  

**Abstract**: Information removal or suppression in large language models (LLMs) is a desired functionality, useful in AI regulation, legal compliance, safety, and privacy. LLM unlearning methods aim to remove information on demand from LLMs. Current LLM unlearning methods struggle to balance the unlearning efficacy and utility due to the competing nature of these objectives. Keeping the unlearning process computationally feasible without assuming access to the model weights is an overlooked area. We present the first agentic LLM unlearning (ALU) method, a multi-agent, retrain-free, model-agnostic approach to LLM unlearning that achieves effective unlearning while preserving the utility. Our ALU framework unlearns by involving multiple LLM agents, each designed for a specific step in the unlearning process, without the need to update model weights for any of the agents in the framework. Users can easily request any set of unlearning instances in any sequence, and ALU seamlessly adapts in real time. This is facilitated without requiring any changes in the underlying LLM model. Through extensive experiments on established benchmarks (TOFU, WMDP, WPU) and jailbreaking techniques (many shot, target masking, other languages), we demonstrate that ALU consistently stands out as the most robust LLM unlearning framework among current state-of-the-art methods while incurring a low constant-time cost. We further highlight ALU's superior performance compared to existing methods when evaluated at scale. Specifically, ALU is assessed on up to 1000 unlearning targets, exceeding the evaluation scope of all previously proposed LLM unlearning methods. 

**Abstract (ZH)**: 大规模语言模型（LLMs）中的信息移除或抑制是一项 desirable 功能，对于 AI 规范管理、法律合规性、安全性和隐私保护等方面具有重要意义。LLM 的去学习方法旨在根据需求从 LLM 中移除信息。目前的 LLM 去学习方法在平衡去学习效果和实用性方面存在困难，因为这两个目标之间存在竞争关系。在不假设访问模型权重的情况下，保持去学习过程的计算可行性是一个被忽视的领域。我们提出了首个代理 LLM 去学习（ALU）方法，这是一种多代理、无需重新训练且模型无关的 LLM 去学习方法，能够在有效去学习的同时保持其实用性。我们的 ALU 框架通过涉及多个 LLM 代理来实现去学习，每个代理都针对去学习过程中的特定步骤设计，而不会更新框架内任何代理的模型权重。用户可以轻松地按任何顺序请求任何一组去学习实例，并且 ALU 可在实时中无缝调整，无需对底层 LLM 模型进行任何更改。通过在已建立的标准基准（TOFU、WMDP、WPU）和破解技术（多射击、目标遮蔽、其他语言）上进行广泛的实验，我们展示了 ALU 作为当前最先进的方法中最为稳健的 LLM 去学习框架，同时具有较低的恒定时间成本。此外，我们在大规模评估中进一步展示了 ALU 相对于现有方法具有更优异的性能。具体而言，ALU 在多达 1000 个去学习目标的评估范围上进行了评估，超过了所有先前提出的 LLM 去学习方法的评估范围。 

---
# A Differentiated Reward Method for Reinforcement Learning based Multi-Vehicle Cooperative Decision-Making Algorithms 

**Title (ZH)**: 基于多车辆协同决策算法的差异化奖励方法 

**Authors**: Ye Han, Lijun Zhang, Dejian Meng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00352)  

**Abstract**: Reinforcement learning (RL) shows great potential for optimizing multi-vehicle cooperative driving strategies through the state-action-reward feedback loop, but it still faces challenges such as low sample efficiency. This paper proposes a differentiated reward method based on steady-state transition systems, which incorporates state transition gradient information into the reward design by analyzing traffic flow characteristics, aiming to optimize action selection and policy learning in multi-vehicle cooperative decision-making. The performance of the proposed method is validated in RL algorithms such as MAPPO, MADQN, and QMIX under varying autonomous vehicle penetration. The results show that the differentiated reward method significantly accelerates training convergence and outperforms centering reward and others in terms of traffic efficiency, safety, and action rationality. Additionally, the method demonstrates strong scalability and environmental adaptability, providing a novel approach for multi-agent cooperative decision-making in complex traffic scenarios. 

**Abstract (ZH)**: 强化学习（RL）在通过状态-动作-奖励反馈循环优化多车辆协同驾驶策略方面显示出巨大潜力，但仍然面临着样本效率低等挑战。本文提出了一种基于稳态转换系统的差异化奖励方法，通过分析交通流特征将状态转换梯度信息纳入奖励设计中，旨在优化多车辆协同决策中的动作选择和策略学习。该方法在不同的自主车辆渗透率下，通过MAPPO、MADQN和QMIX等RL算法进行了性能验证。结果显示，差异化奖励方法在交通效率、安全性和动作合理性方面显著加快了训练收敛速度，并优于中心化奖励及其他方法。此外，该方法展示了强大的可扩展性和环境适应性，为复杂交通场景下的多智能体协同决策提供了一种新的解决方案。 

---
# The role of positional encodings in the ARC benchmark 

**Title (ZH)**: ARC基准中位置编码的作用 

**Authors**: Guilherme H. Bandeira Costa, Miguel Freire, Arlindo L. Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2502.00174)  

**Abstract**: The Abstraction and Reasoning Corpus challenges AI systems to perform abstract reasoning with minimal training data, a task intuitive for humans but demanding for machine learning models. Using CodeT5+ as a case study, we demonstrate how limitations in positional encoding hinder reasoning and impact performance. This work further examines the role of positional encoding across transformer architectures, highlighting its critical influence on models of varying sizes and configurations. Comparing several strategies, we find that while 2D positional encoding and Rotary Position Embedding offer competitive performance, 2D encoding excels in data-constrained scenarios, emphasizing its effectiveness for ARC tasks 

**Abstract (ZH)**: 抽象与推理语料库挑战AI系统在最少训练数据的情况下进行抽象推理，这是一个对人类直觉来说简单但对机器学习模型来说却具有挑战的任务。以CodeT5+为例，我们展示了位置编码限制如何影响推理能力并影响模型性能。这项工作进一步探讨了位置编码在变压器架构中的作用，强调了其对不同大小和配置模型的关键影响。通过比较几种策略，我们发现虽然2D位置编码和旋转位置嵌入提供了竞争性的性能，但在数据受限的场景中，2D编码表现出色，凸显了其在ARC任务中的有效性。 

---
# Counting and Reasoning with Plans 

**Title (ZH)**: 计数与计划推理 

**Authors**: David Speck, Markus Hecher, Daniel Gnad, Johannes K. Fichte, Augusto B. Corrêa  

**Link**: [PDF](https://arxiv.org/pdf/2502.00145)  

**Abstract**: Classical planning asks for a sequence of operators reaching a given goal. While the most common case is to compute a plan, many scenarios require more than that. However, quantitative reasoning on the plan space remains mostly unexplored. A fundamental problem is to count plans, which relates to the conditional probability on the plan space. Indeed, qualitative and quantitative approaches are well-established in various other areas of automated reasoning. We present the first study to quantitative and qualitative reasoning on the plan space. In particular, we focus on polynomially bounded plans. On the theoretical side, we study its complexity, which gives rise to rich reasoning modes. Since counting is hard in general, we introduce the easier notion of facets, which enables understanding the significance of operators. On the practical side, we implement quantitative reasoning for planning. Thereby, we transform a planning task into a propositional formula and use knowledge compilation to count different plans. This framework scales well to large plan spaces, while enabling rich reasoning capabilities such as learning pruning functions and explainable planning. 

**Abstract (ZH)**: 经典规划旨在找到一系列操作符以达到给定的目标。虽然最常见的应用场景是计算一个计划，但很多情况下需要更多的信息。然而，对计划空间中的量化推理仍处于相对未开发的状态。一个基本问题是计算计划的数量，这与计划空间上的条件概率密切相关。事实上，定性和定量推理已经在自动推理的许多其他领域得到了充分的发展。我们提出了首个对计划空间进行定性与定量推理的研究。特别地，我们关注那些在多项式界内的计划。从理论角度，我们研究其复杂性，从而产生了丰富的推理模式。由于计数问题在一般情况下是困难的，我们引入了更容易处理的概念——面，这有助于理解操作符的重要性。从实践角度看，我们实现了一种对规划的定量推理方法。具体来说，我们将规划任务转化为命题公式，并利用知识编译技术来计算不同计划的数量。这种框架能够有效地处理大型计划空间，同时支持丰富的推理能力，如学习剪枝函数和解释性规划。 

---
# Towards Efficient Multi-Objective Optimisation for Real-World Power Grid Topology Control 

**Title (ZH)**: 面向实际电力网络拓扑控制的高效多目标优化研究 

**Authors**: Yassine El Manyari, Anton R. Fuxjager, Stefan Zahlner, Joost Van Dijk, Alberto Castagna, Davide Barbieri, Jan Viebahn, Marcel Wasserer  

**Link**: [PDF](https://arxiv.org/pdf/2502.00034)  

**Abstract**: Power grid operators face increasing difficulties in the control room as the increase in energy demand and the shift to renewable energy introduce new complexities in managing congestion and maintaining a stable supply. Effective grid topology control requires advanced tools capable of handling multi-objective trade-offs. While Reinforcement Learning (RL) offers a promising framework for tackling such challenges, existing Multi-Objective Reinforcement Learning (MORL) approaches fail to scale to the large state and action spaces inherent in real-world grid operations. Here we present a two-phase, efficient and scalable Multi-Objective Optimisation (MOO) method designed for grid topology control, combining an efficient RL learning phase with a rapid planning phase to generate day-ahead plans for unseen scenarios. We validate our approach using historical data from TenneT, a European Transmission System Operator (TSO), demonstrating minimal deployment time, generating day-ahead plans within 4-7 minutes with strong performance. These results underline the potential of our scalable method to support real-world power grid management, offering a practical, computationally efficient, and time-effective tool for operational planning. Based on current congestion costs and inefficiencies in grid operations, adopting our approach by TSOs could potentially save millions of euros annually, providing a compelling economic incentive for its integration in the control room. 

**Abstract (ZH)**: 随着能源需求的增长和向可再生能源的转变，电网运营商在控制室面临着越来越多的管理挑战。新增的能源供需不匹配和电网拥堵问题使得维持稳定供应变得更加复杂。有效的电网拓扑控制需要能够处理多目标权衡的高级工具。强化学习（Reinforcement Learning, RL）为应对这些挑战提供了前景广阔的框架，但现有的多目标强化学习（Multi-Objective Reinforcement Learning, MORL）方法无法处理实际电网操作中固有的庞大状态空间和动作空间。为了解决这一问题，我们提出了一种两阶段的、高效和可扩展的多目标优化（Multi-Objective Optimization, MOO）方法，该方法主要用于电网拓扑控制，结合了高效的RL学习阶段和快速的规划阶段，以生成针对未见场景的次日计划。我们利用欧洲输电系统运营商（TSO）TenneT的历史数据验证了该方法的有效性，证明了部署时间的最小化，并能在4-7分钟内生成出色的次日计划。这些结果表明，我们的可扩展方法有可能支持实际的电力网络管理，提供了一种实用的、计算效率高且时间效率高的操作规划工具。

当前电网中的拥堵成本和运营效率低下表明，TSO采用我们的方法每年可能节省数百万欧元，这为将其集成到控制室中提供了强大的经济效益驱动。 

---
# A Dynamic and High-Precision Method for Scenario-Based HRA Synthetic Data Collection in Multi-Agent Collaborative Environments Driven by LLMs 

**Title (ZH)**: 基于大型语言模型驱动的多agent协作环境中情景基于的人因工程合成数据动态高精度采集方法 

**Authors**: Xingyu Xiao, Peng Chen, Qianqian Jia, Jiejuan Tong, Jingang Liang, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00022)  

**Abstract**: HRA (Human Reliability Analysis) data is crucial for advancing HRA methodologies. however, existing data collection methods lack the necessary granularity, and most approaches fail to capture dynamic features. Additionally, many methods require expert knowledge as input, making them time-consuming and labor-intensive. To address these challenges, we propose a new paradigm for the automated collection of HRA data. Our approach focuses on key indicators behind human error, specifically measuring workload in collaborative settings. This study introduces a novel, scenario-driven method for workload estimation, leveraging fine-tuned large language models (LLMs). By training LLMs on real-world operational data from high-temperature gas-cooled reactors (HTGRs), we simulate human behavior and cognitive load in real time across various collaborative scenarios. The method dynamically adapts to changes in operator workload, providing more accurate, flexible, and scalable workload estimates. The results demonstrate that the proposed WELLA (Workload Estimation with LLMs and Agents) outperforms existing commercial LLM-based methods in terms of prediction accuracy. 

**Abstract (ZH)**: 人类可靠性分析（HRA）数据对于推进HRA方法具有重要意义。然而，现有的数据收集方法缺乏必要的粒度，且大多数方法无法捕捉动态特征。此外，许多方法需要专家知识作为输入，这使得它们耗时且劳动密集。为应对这些挑战，我们提出了一种新的自动化收集HRA数据的范式。我们的方法集中在影响人类错误的关键指标上，特别关注协作环境中的工作负荷测量。本研究引入了一种基于场景的新颖方法来估算工作负荷，利用细调后的大型语言模型（LLMs）。通过使用高温气冷堆（HTGRs）的真实操作数据对LLMs进行训练，我们实时模拟了各种协作场景下的人类行为和认知负荷。该方法能够动态适应操作员工作负荷的变化，提供更准确、灵活和可扩展的工作负荷估计。结果表明，提出的WELLA（使用LLM和代理的工作负荷估算）在预测准确性方面优于现有的商业LLM基方法。 

---
# Temporal Reasoning in AI systems 

**Title (ZH)**: AI系统中的时序推理 

**Authors**: Abhishek Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2502.00020)  

**Abstract**: Commonsense temporal reasoning at scale is a core problem for cognitive systems. The correct inference of the duration for which fluents hold is required by many tasks, including natural language understanding and planning. Many AI systems have limited deductive closure because they cannot extrapolate information correctly regarding existing fluents and events. In this study, we discuss the knowledge representation and reasoning schemes required for robust temporal projection in the Cyc Knowledge Base. We discuss how events can start and end risk periods for fluents. We then use discrete survival functions, which represent knowledge of the persistence of facts, to extrapolate a given fluent. The extrapolated intervals can be truncated by temporal constraints and other types of commonsense knowledge. Finally, we present the results of experiments to demonstrate that these methods obtain significant improvements in terms of Q/A performance. 

**Abstract (ZH)**: 大规模下的常识时间推理是认知系统中的核心问题。正确推断语境持续时间对于许多任务至关重要，包括自然语言理解和规划。许多AI系统由于无法正确外推现有语境和事件的信息，导致推理闭包有限。在本研究中，我们探讨了 Cyc 知识库中实现稳健时间外推所需的知识表示和推理方法。我们讨论了如何通过事件的开始和结束来界定语境的风险期。然后，我们利用离散生存函数来外推给定的语境，这些离散生存函数能够表示事实的持久性知识。外推的时间间隔可以由时间约束和其他类型的常识知识进行裁剪。最后，我们展示了实验结果，证明了这些方法在问答性能方面取得了显著改进。 

---
# Growth Patterns of Inference 

**Title (ZH)**: 推理的发展模式 

**Authors**: Abhishek Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2502.00019)  

**Abstract**: What properties of a first-order search space support/hinder inference? What kinds of facts would be most effective to learn? Answering these questions is essential for understanding the dynamics of deductive reasoning and creating large-scale knowledge-based learning systems that support efficient inference. We address these questions by developing a model of how the distribution of ground facts affects inference performance in search spaces. Experiments suggest that uniform search spaces are suitable for larger KBs whereas search spaces with skewed degree distribution show better performance in smaller KBs. A sharp transition in Q/A performance is seen in some cases, suggesting that analysis of the structure of search spaces with existing knowledge should be used to guide the acquisition of new ground facts in learning systems. 

**Abstract (ZH)**: 具有什么性质的一阶搜索空间有助于/妨碍推理？哪种类型的事实最有效？回答这些问题对于理解演绎推理的动力学并创建支持高效推理的大规模知识基础学习系统至关重要。我们通过开发一个模型来探讨基础事实在搜索空间中分布对推理性能的影响来回答这些问题。实验表明，均匀分布的事实可能更适合更大的知识库（KB），而具有偏斜度分布的事实对于较小的知识库表现更好。在某些情况下，问题/答案性能存在明显的转变，这表明应该利用现有知识分析搜索空间的结构，以指导学习系统中新基础事实的获取。 

---
# An Expectation-Maximization Algorithm-based Autoregressive Model for the Fuzzy Job Shop Scheduling Problem 

**Title (ZH)**: 基于期望最大化算法的自回归模型在模糊车间调度问题中的应用 

**Authors**: Yijian Wang, Tongxian Guo, Zhaoqiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00018)  

**Abstract**: The fuzzy job shop scheduling problem (FJSSP) emerges as an innovative extension to the job shop scheduling problem (JSSP), incorporating a layer of uncertainty that aligns the problem more closely with the complexities of real-world manufacturing environments. This improvement increases the computational complexity of deriving the solution while improving its applicability. In the domain of deterministic scheduling, neural combinatorial optimization (NCO) has recently demonstrated remarkable efficacy. However, its application to the realm of fuzzy scheduling has been relatively unexplored. This paper aims to bridge this gap by investigating the feasibility of employing neural networks to assimilate and process fuzzy information for the resolution of FJSSP, thereby leveraging the advancements in NCO to enhance fuzzy scheduling methodologies. To achieve this, we approach the FJSSP as a generative task and introduce an expectation-maximization algorithm-based autoregressive model (EMARM) to address it. During training, our model alternates between generating scheduling schemes from given instances (E-step) and adjusting the autoregressive model weights based on these generated schemes (M-step). This novel methodology effectively navigates around the substantial hurdle of obtaining ground-truth labels, which is a prevalent issue in NCO frameworks. In testing, the experimental results demonstrate the superior capability of EMARM in addressing the FJSSP, showcasing its effectiveness and potential for practical applications in fuzzy scheduling. 

**Abstract (ZH)**: 模糊车间调度问题（FJSSP）作为车间调度问题（JSSP）的一种创新扩展，引入了一层不确定性，使问题更加贴近真实制造环境的复杂性。这一改进增加了求解问题的计算复杂度，但同时也提高了其适用性。在确定性调度领域，神经组合优化（NCO）近年来已显示出显著的有效性。然而，将其应用于模糊调度领域的研究相对较少。本文旨在填补这一空白，通过研究神经网络在处理和利用模糊信息以解决FJSSP的可行性，结合NCO的最新进展来提高模糊调度方法的效果。为实现这一目标，我们将FJSSP视为生成任务，并引入了一个基于期望最大化算法（EM算法）的自回归模型（EMARM，Expectation-Maximization Algorithm-based Autoregressive Model）来解决它。在训练过程中，我们的模型交替进行两步操作：首先根据给定的具体情形生成调度方案（E步），然后根据这些生成的方案调整自回归模型的权重（M步）。这种新颖的方法有效地绕过了NCO框架中常见的难以获取真实标签问题的障碍。在测试阶段，实验结果表明EMARM在解决FJSSP方面表现出色，展示了其在模糊调度领域的有效性和实际应用潜力。 

---
# Lifelong Sequential Knowledge Editing without Model Degradation 

**Title (ZH)**: 无需模型退化的一生序列知识编辑 

**Authors**: Akshat Gupta, Phudish Prateepamornkul, Maochuan Lu, Ahmed Alaa, Thomas Hartvigsen, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2502.01636)  

**Abstract**: Prior work in parameter-modifying knowledge editing has shown that large-scale sequential editing leads to significant model degradation. In this paper, we study the reasons behind this and scale sequential knowledge editing to 10,000 sequential edits, while maintaining the downstream performance of the original model. We first show that locate-then-edit knowledge editing methods lead to overfitting on the edited facts. We also show that continuous knowledge editing using these methods leads to disproportionate growth in the norm of the edited matrix. We then provide a crucial insight into the inner workings of locate-then-edit methods. We show that norm-growth is a hidden trick employed by these methods that gives larger importance to the output activations produced from the edited layers. With this "importance hacking", the edited layers provide a much larger contributions to the model's output. To mitigate these issues, we present ENCORE - Early stopping and Norm-Constrained Robust knowledge Editing. ENCORE controls for overfitting and the disproportionate norm-growth to enable long-term sequential editing, where we are able to perform up to 10,000 sequential edits without loss of downstream performance. ENCORE is also 61% faster than MEMIT and 64% faster than AlphaEdit on Llama3-8B. 

**Abstract (ZH)**: 参数调整型知识编辑的前期研究已经表明，大规模的顺序编辑会导致模型性能显著下降。本文我们探讨了这一现象背后的原因，并将顺序知识编辑扩展到10,000次编辑，同时保持原始模型的下游性能。首先，我们展示了定位然后编辑的知识编辑方法会导致对编辑事实的过度拟合。此外，我们还表明，使用这些方法进行连续的知识编辑会导致编辑矩阵范数不成比例的增长。然后我们深入探讨了定位然后编辑方法的内部工作机制。我们展示了这些方法中隐藏的技巧——范数增长，这是这些方法赋予编辑层输出激活更重要的权重的结果。通过这种“重要性劫持”，编辑层在模型输出中的贡献显著增加。为了解决这些问题，我们提出了一种新的方法——ENCORE（Early stopping and Norm-Constrained Robust knowledge Editing）。ENCORE通过控制过度拟合和不成比例的范数增长，使得长期的顺序编辑成为可能，在进行多达10,000次顺序编辑后，仍然保持下游性能。此外，ENCORE在Llama3-8B上的速度分别比MEMIT快61%和比AlphaEdit快64%。 

---
# The AI Agent Index 

**Title (ZH)**: 《AI代理指数》 

**Authors**: Stephen Casper, Luke Bailey, Rosco Hunter, Carson Ezell, Emma Cabalé, Michael Gerovitch, Stewart Slocum, Kevin Wei, Nikola Jurkovic, Ariba Khan, Phillip J.K. Christoffersen, A. Pinar Ozisik, Rakshit Trivedi, Dylan Hadfield-Menell, Noam Kolt  

**Link**: [PDF](https://arxiv.org/pdf/2502.01635)  

**Abstract**: Leading AI developers and startups are increasingly deploying agentic AI systems that can plan and execute complex tasks with limited human involvement. However, there is currently no structured framework for documenting the technical components, intended uses, and safety features of agentic systems. To fill this gap, we introduce the AI Agent Index, the first public database to document information about currently deployed agentic AI systems. For each system that meets the criteria for inclusion in the index, we document the system's components (e.g., base model, reasoning implementation, tool use), application domains (e.g., computer use, software engineering), and risk management practices (e.g., evaluation results, guardrails), based on publicly available information and correspondence with developers. We find that while developers generally provide ample information regarding the capabilities and applications of agentic systems, they currently provide limited information regarding safety and risk management practices. The AI Agent Index is available online at this https URL 

**Abstract (ZH)**: 领先的AI开发者和初创企业正越来越多地部署自主型AI系统，这些系统能够在有限的人为干预下规划和执行复杂的任务。然而，当前缺乏一个结构化的框架来记录自主型系统的技术组件、预期用途和安全性特征。为填补这一空白，我们引入了AI Agent Index，这是第一个公开数据库，用于记录当前部署的自主型AI系统的相关信息。对于每个符合索引收录标准的系统，我们根据公开信息和与开发者的交流记录了该系统的组件（例如，基础模型、推理实现、工具使用）、应用领域（例如，计算机使用、软件工程）以及风险管理实践（例如，评估结果、防护措施）。我们发现，虽然开发者通常能提供大量关于自主型系统功能和应用的信息，但他们目前在安全性和风险管理实践方面的信息相对有限。AI Agent Index 可通过以下网址在线访问：[该网址] 

---
# Online Gradient Boosting Decision Tree: In-Place Updates for Efficient Adding/Deleting Data 

**Title (ZH)**: 在线梯度提升决策树：高效添加/删除数据的原地更新方法 

**Authors**: Huawei Lin, Jun Woo Chung, Yingjie Lao, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.01634)  

**Abstract**: Gradient Boosting Decision Tree (GBDT) is one of the most popular machine learning models in various applications. However, in the traditional settings, all data should be simultaneously accessed in the training procedure: it does not allow to add or delete any data instances after training. In this paper, we propose an efficient online learning framework for GBDT supporting both incremental and decremental learning. To the best of our knowledge, this is the first work that considers an in-place unified incremental and decremental learning on GBDT. To reduce the learning cost, we present a collection of optimizations for our framework, so that it can add or delete a small fraction of data on the fly. We theoretically show the relationship between the hyper-parameters of the proposed optimizations, which enables trading off accuracy and cost on incremental and decremental learning. The backdoor attack results show that our framework can successfully inject and remove backdoor in a well-trained model using incremental and decremental learning, and the empirical results on public datasets confirm the effectiveness and efficiency of our proposed online learning framework and optimizations. 

**Abstract (ZH)**: 梯度提升决策树（GBDT）是各种应用中最为流行的一种机器学习模型。然而，在传统的设置中，训练过程中需要同时访问所有数据：它不允许在训练之后添加或删除任何数据实例。在本文中，我们提出了一种高效的在线学习框架，该框架支持增量和 decremental 学习。据我们所知，这是首篇在 GBDT 中考虑原位统一增量和 decremental 学习的研究工作。为了降低学习成本，我们提出了该框架的一系列优化方法，使其能够在运行时添加或删除少量数据。我们从理论上展示了所提出优化的超参数之间的关系，这使得在增量和 decremental 学习中可以权衡准确性和成本。后门攻击结果表明，我们的框架可以利用增量和 decremental 学习成功地在已训练好的模型中注入和移除后门。同时，公共数据集上的实证结果证实了我们提出的在线学习框架及其优化方法的有效性和效率。 

---
# Adversarial Reasoning at Jailbreaking Time 

**Title (ZH)**: 在 Jailbreaking 时刻的对抗性推理 

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2502.01633)  

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的能力增强和应用范围扩大，对其失败案例的研究变得尤为重要。最近在标准化、测量和扩展测试时计算方面的进展为优化模型以在困难任务上实现高性能提供了新的方法论。本文将这些进展应用于模型脱狱任务：从对齐的LLM中引致有害响应。我们开发了一种对抗推理方法，通过测试时计算实现对许多对齐的LLM的最高成功率（SOTA攻击成功率），即使是对那些旨在通过减少推理时计算来换取对抗鲁棒性的LLM也是如此。我们的方法引入了一种理解和应对LLM脆弱性的新范式，为开发更加稳健和可信赖的AI系统奠定了基础。 

---
# Learning to Generate Unit Tests for Automated Debugging 

**Title (ZH)**: 学习生成单元测试用于自动化调试 

**Authors**: Archiki Prasad, Elias Stengel-Eskin, Justin Chih-Yao Chen, Zaid Khan, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.01619)  

**Abstract**: Unit tests (UTs) play an instrumental role in assessing code correctness as well as providing feedback to a large language model (LLM) as it iteratively debugs faulty code, motivating automated test generation. However, we uncover a trade-off between generating unit test inputs that reveal errors when given a faulty code and correctly predicting the unit test output without access to the gold solution. To address this trade-off, we propose UTGen, which teaches LLMs to generate unit test inputs that reveal errors along with their correct expected outputs based on task descriptions and candidate code. We integrate UTGen into UTDebug, a robust debugging pipeline that uses generated tests to help LLMs debug effectively. Since model-generated tests can provide noisy signals (e.g., from incorrectly predicted outputs), UTDebug (i) scales UTGen via test-time compute to improve UT output prediction, and (ii) validates and back-tracks edits based on multiple generated UTs to avoid overfitting. We show that UTGen outperforms UT generation baselines by 7.59% based on a metric measuring the presence of both error-revealing UT inputs and correct UT outputs. When used with UTDebug, we find that feedback from UTGen's unit tests improves pass@1 accuracy of Qwen-2.5 7B on HumanEvalFix and our own harder debugging split of MBPP+ by over 3% and 12.35% (respectively) over other LLM-based UT generation baselines. 

**Abstract (ZH)**: 单位测试（UTs）在评估代码正确性方面发挥着重要作用，并能为大型语言模型（LLMs）提供反馈，帮助其逐步调试错误代码，从而激发自动测试生成的需求。然而，我们发现生成能够揭示错误的单元测试输入与在没有金标准的情况下正确预测单元测试输出之间存在权衡。为了解决这一权衡问题，我们提出了UTGen方法，教导LLMs根据任务描述和候选代码生成揭示错误的同时提供正确预期输出的单元测试输入。我们将UTGen集成到UTDebug中，这是一个稳健的调试管道，利用生成的测试来帮助LLMs更有效地调试。由于模型生成的测试可能提供嘈杂的信号（例如，从错误预测的输出中），UTDebug通过（i）在测试期间增加计算量来扩展UTGen，以提高单元测试输出的预测效果；（ii）基于多个生成的单元测试验证和回溯编辑，以避免过拟合。结果显示，与衡量同时包含揭示错误的单元测试输入和正确单元测试输出的指标相比，UTGen比基线模块高出7.59%。当与UTDebug结合使用时，来自UTGen单元测试的反馈在HumanEvalFix任务上提高了Qwen-2.5 7B模型的pass@1准确性超过3%，在我们自己构建的更难的调试分拆MBPP+上提高了12.35%，超过了其他基于LLM的单元测试生成基线。 

---
# A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods 

**Title (ZH)**: 使用粒子蒙特卡洛方法的基于概率推理的大型语言模型推理时缩放方法 

**Authors**: Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2502.01618)  

**Abstract**: Large language models (LLMs) have achieved significant performance gains via scaling up model sizes and/or data. However, recent evidence suggests diminishing returns from such approaches, motivating scaling the computation spent at inference time. Existing inference-time scaling methods, usually with reward models, cast the task as a search problem, which tends to be vulnerable to reward hacking as a consequence of approximation errors in reward models. In this paper, we instead cast inference-time scaling as a probabilistic inference task and leverage sampling-based techniques to explore the typical set of the state distribution of a state-space model with an approximate likelihood, rather than optimize for its mode directly. We propose a novel inference-time scaling approach by adapting particle-based Monte Carlo methods to this task. Our empirical evaluation demonstrates that our methods have a 4-16x better scaling rate over our deterministic search counterparts on various challenging mathematical reasoning tasks. Using our approach, we show that Qwen2.5-Math-1.5B-Instruct can surpass GPT-4o accuracy in only 4 rollouts, while Qwen2.5-Math-7B-Instruct scales to o1 level accuracy in only 32 rollouts. Our work not only presents an effective method to inference-time scaling, but also connects the rich literature in probabilistic inference with inference-time scaling of LLMs to develop more robust algorithms in future work. Code and further information is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过扩大模型规模和/或数据量实现了显著的性能提升。然而，近期的研究证据表明，这种做法的效果逐渐减弱，从而促使我们在推理阶段增加计算量。现有的推理时扩展方法通常使用奖励模型将任务视为搜索问题，这由于奖励模型的近似误差容易导致奖励作弊问题。在本文中，我们相反地将推理时扩展视为一个概率推理任务，并利用基于采样的技术探索状态空间模型状态分布的典型集，而不是直接优化其模式。我们提出了一种新的推理时扩展方法，通过将粒子蒙特卡洛方法适应于这个任务。我们的实验评估表明，与确定性搜索方法相比，我们的方法在多种具有挑战性的数学推理任务中具有4-16倍的更好的扩展率。使用我们的方法，我们展示了Qwen2.5-Math-1.5B-Instruct仅需4次推理即可超越GPT-4o的准确性，而Qwen2.5-Math-7B-Instruct仅需32次推理即可达到o1级的准确性。我们的工作不仅提供了一种有效的推理时扩展方法，还通过将概率推理领域的丰富文献与LLMs的推理时扩展相结合，为未来的更稳健算法开发奠定了基础。更多信息和代码可通过以下链接获取：this https URL 

---
# Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges 

**Title (ZH)**: 自我提升的Transformer模型克服了从易到难和长度泛化挑战 

**Authors**: Nayoung Lee, Ziyang Cai, Avi Schwarzschild, Kangwook Lee, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.01612)  

**Abstract**: Large language models often struggle with length generalization and solving complex problem instances beyond their training distribution. We present a self-improvement approach where models iteratively generate and learn from their own solutions, progressively tackling harder problems while maintaining a standard transformer architecture. Across diverse tasks including arithmetic, string manipulation, and maze solving, self-improving enables models to solve problems far beyond their initial training distribution-for instance, generalizing from 10-digit to 100-digit addition without apparent saturation. We observe that in some cases filtering for correct self-generated examples leads to exponential improvements in out-of-distribution performance across training rounds. Additionally, starting from pretrained models significantly accelerates this self-improvement process for several tasks. Our results demonstrate how controlled weak-to-strong curricula can systematically teach a model logical extrapolation without any changes to the positional embeddings, or the model architecture. 

**Abstract (ZH)**: 大型语言模型往往在长度泛化和解决超出训练分布的复杂问题实例时表现不佳。我们提出了一种自我改进的方法，其中模型通过迭代生成和从自己的解决方案中学习，逐步解决更难的问题，同时保持标准的变压器架构。在算术、字符串操作和迷宫求解等多样化的任务中，自我改进使模型能够解决远超出其初始训练分布的问题——例如，从10位数加法到100位数加法的泛化，而无需出现饱和现象。我们发现，在某些情况下，筛选正确生成的示例可以在训练轮次中显著提高模型在分布外的表现，使其呈指数级改善。此外，从预训练模型开始可以显著加速这种自我改进过程，尤其是在某些任务中。我们的结果表明，通过控制从弱到强的教学计划，可以在不改变位置嵌入或模型架构的情况下系统地教会模型逻辑外推。 

---
# Reinforcement Learning for Long-Horizon Interactive LLM Agents 

**Title (ZH)**: 长时 horizon 交互式大型语言模型智能体的强化学习方法 

**Authors**: Kevin Chen, Marco Cusumano-Towner, Brody Huval, Aleksei Petrenko, Jackson Hamburger, Vladlen Koltun, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2502.01600)  

**Abstract**: Interactive digital agents (IDAs) leverage APIs of stateful digital environments to perform tasks in response to user requests. While IDAs powered by instruction-tuned large language models (LLMs) can react to feedback from interface invocations in multi-step exchanges, they have not been trained in their respective digital environments. Prior methods accomplish less than half of tasks in sophisticated benchmarks such as AppWorld. We present a reinforcement learning (RL) approach that trains IDAs directly in their target environments. We formalize this training as a partially observable Markov decision process and derive M-PPO, a data- and memory-efficient variant of proximal policy optimization. M-PPO uses no value network and maintains exactly one copy of the underlying LLM in memory, making its implementation straightforward and as memory-efficient as fine-tuning a single LLM. A 32-billion-parameter agent trained with M-PPO in the AppWorld environment outperforms the much larger OpenAI o1 agent by 9 percentage points (15% relative). To our knowledge, this is the first reported application of RL to IDAs that interact with a stateful, multi-domain, multi-app environment via direct API calls. Our analysis sheds light on the effectiveness of RL in this area, showing that the agent learns to consult the API documentation, avoid unwarranted assumptions, minimize confabulation, and recover from setbacks. 

**Abstract (ZH)**: 交互式数字代理（IDAs）利用状态型数字环境的API来响应用户请求并执行任务。由指令调优的大语言模型（LLMs）驱动的IDAs可以在多步骤交互中对界面调用的反馈做出反应，但它们并未在其各自的数字环境中进行训练。此前的方法在复杂的基准测试（如AppWorld）中仅能完成不到一半的任务。我们提出了一个利用强化学习（RL）的训练方法，直接在目标环境中训练IDAs。我们将这一训练过程形式化为部分可观测的马尔可夫决策过程，并推导出M-PPO，这是一个数据和内存效率更高的 proximal policy optimization 变体。M-PPO 不使用价值网络，并且在内存中仅仅维持一个底层LLM的副本，从而使得其实现简单，并且内存效率与对单一LLM的调优相当。在一个具有320亿参数的代理在AppWorld环境中通过M-PPO训练后，其表现超过了OpenAI的更大型o1代理9个百分点（即15%的相对改进）。据我们所知，这是首次将RL应用于通过直接API调用来与状态型、多域、多应用环境交互的IDAs。我们的分析揭示了在这一领域中RL的有效性，表明该代理学会了查阅API文档、避免不切实际的假设、减少虚构信息，并能从挫折中恢复。 

---
# Improving Transformer World Models for Data-Efficient RL 

**Title (ZH)**: 提高变换器世界模型以实现数据高效的强化学习 

**Authors**: Antoine Dedieu, Joseph Ortiz, Xinghua Lou, Carter Wendelken, Wolfgang Lehrach, J Swaroop Guntupalli, Miguel Lazaro-Gredilla, Kevin Patrick Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2502.01591)  

**Abstract**: We present an approach to model-based RL that achieves a new state of the art performance on the challenging Craftax-classic benchmark, an open-world 2D survival game that requires agents to exhibit a wide range of general abilities -- such as strong generalization, deep exploration, and long-term reasoning. With a series of careful design choices aimed at improving sample efficiency, our MBRL algorithm achieves a reward of 67.4% after only 1M environment steps, significantly outperforming DreamerV3, which achieves 53.2%, and, for the first time, exceeds human performance of 65.0%. Our method starts by constructing a SOTA model-free baseline, using a novel policy architecture that combines CNNs and RNNs. We then add three improvements to the standard MBRL setup: (a) "Dyna with warmup", which trains the policy on real and imaginary data, (b) "nearest neighbor tokenizer" on image patches, which improves the scheme to create the transformer world model (TWM) inputs, and (c) "block teacher forcing", which allows the TWM to reason jointly about the future tokens of the next timestep. 

**Abstract (ZH)**: 我们提出了一种基于模型的强化学习方法，在具有挑战性的Craftax-classic基准测试中达到了新的最先进性能，这是一个开放世界的2D生存游戏，要求代理展现广泛的一般能力——如强大的泛化能力、深度探索能力和长期推理能力。通过一系列旨在提高样本效率的设计选择，我们的MBRL算法在仅100万环境步之后获得67.4%的奖励，显著优于DreamerV3（53.2%），首次超过人类的65.0%的性能。我们的方法首先构建了一个最先进的模型自由基线，采用了一种结合卷积神经网络（CNN）和递归神经网络（RNN）的新颖策略架构。然后，我们对标准的MBRL设置进行了三个改进：（a）“预热的Dyna”，通过训练真实和虚拟数据上的策略；（b）在图像块上使用“最近邻分词器”，以改进创建转换器世界模型（TWM）输入的方案；（c）“块教师强约束”，允许TWM联合推理下一个时间步的后续标记。 

---
# Verbalized Bayesian Persuasion 

**Title (ZH)**: 口头化的贝叶斯说服 

**Authors**: Wenhao Li, Yue Lin, Xiangfeng Wang, Bo Jin, Hongyuan Zha, Baoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01587)  

**Abstract**: Information design (ID) explores how a sender influence the optimal behavior of receivers to achieve specific objectives. While ID originates from everyday human communication, existing game-theoretic and machine learning methods often model information structures as numbers, which limits many applications to toy games. This work leverages LLMs and proposes a verbalized framework in Bayesian persuasion (BP), which extends classic BP to real-world games involving human dialogues for the first time. Specifically, we map the BP to a verbalized mediator-augmented extensive-form game, where LLMs instantiate the sender and receiver. To efficiently solve the verbalized game, we propose a generalized equilibrium-finding algorithm combining LLM and game solver. The algorithm is reinforced with techniques including verbalized commitment assumptions, verbalized obedience constraints, and information obfuscation. Numerical experiments in dialogue scenarios, such as recommendation letters, courtroom interactions, and law enforcement, validate that our framework can both reproduce theoretical results in classic BP and discover effective persuasion strategies in more complex natural language and multi-stage scenarios. 

**Abstract (ZH)**: 信息设计（ID）探讨了发送者如何影响接收者的最优行为以实现特定目标。尽管ID源自日常生活中的交流，但现有的博弈论和机器学习方法常常将信息结构建模为数字，这限制了许多应用只能局限于玩具博弈。本文利用大型语言模型（LLMs），提出了一种在贝叶斯说服（BP）中的口头化框架，首次将经典BP扩展到包含人类对话的真实世界博弈中。具体而言，我们将BP映射到一个口头化的调解人增强的扩展形式博弈中，其中LLMs实现发送者和接收者。为了高效求解口头化博弈，我们提出了一种结合LLMs和博弈求解器的广义均衡寻找算法，并通过包括口头化的承诺假设、口头化的服从约束和信息混淆等技术来加强该算法。在推荐信、法庭互动和执法等对话场景的数值实验中，验证了我们的框架既能够重现经典BP中的理论结果，又能够在更复杂的自然语言和多阶段场景中发现有效的说服策略。 

---
# Next Steps in LLM-Supported Java Verification 

**Title (ZH)**: LLM 支撑的 Java 验证下一步工作 

**Authors**: Samuel Teuber, Bernhard Beckert  

**Link**: [PDF](https://arxiv.org/pdf/2502.01573)  

**Abstract**: Recent work has shown that Large Language Models (LLMs) are not only a suitable tool for code generation but also capable of generating annotation-based code specifications. Scaling these methodologies may allow us to deduce provable correctness guarantees for large-scale software systems. In comparison to other LLM tasks, the application field of deductive verification has the notable advantage of providing a rigorous toolset to check LLM-generated solutions. This short paper provides early results on how this rigorous toolset can be used to reliably elicit correct specification annotations from an unreliable LLM oracle. 

**Abstract (ZH)**: 近期的研究表明，大型语言模型（LLMs）不仅适合作为代码生成的工具，还能够生成基于注解的代码规范。通过扩展这些方法，我们可以为大规模软件系统推导出可验证的正确性保证。与其它LLM任务相比，演绎验证的应用领域的一个显著优势是可以提供严格的工具集来检查LLM生成的解决方案。本文简要呈现了如何利用这种严格的工具集可靠地从不可靠的LLM先验中提取正确的规范注解的早期结果。 

---
# Visual Theory of Mind Enables the Invention of Writing Systems 

**Title (ZH)**: 视觉共情能力促进了书写系统的发明 

**Authors**: Benjamin A. Spiegel, Lucas Gelfond, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2502.01568)  

**Abstract**: Abstract symbolic writing systems are \textit{semiotic codes} that are ubiquitous in modern society but are otherwise absent in the animal kingdom. Anthropological evidence suggests that the earliest forms of some writing systems originally consisted of \textit{iconic pictographs}, which signify their referent via visual resemblance. While previous studies have examined the emergence and, separately, the evolution of pictographic writing systems through a computational lens, most employ non-naturalistic methodologies that make it difficult to draw clear analogies to human and animal cognition. We develop a multi-agent reinforcement learning testbed for emergent communication called a \textit{Signification Game}, and formulate a model of inferential communication that enables agents to leverage \textit{visual theory of mind} to communicate actions using pictographs. Our model, which is situated within a broader formalism for animal communication, sheds light on the cognitive and cultural processes that led to the development of early writing systems. 

**Abstract (ZH)**: 抽象符号书写系统是现代社会中无处不在的 **符号代码**，但在动物界中却不存在。人类学证据表明，某些书写系统最早的形式可能由 **象征性象形文字** 构成，这些文字通过视觉相似性来表示它们所指代的对象。尽管以往的研究从计算的角度分别探讨了象形文字书写系统出现和演化的机制，但大多数研究采用非自然的方法，使其难以与人类和动物的认知过程建立清晰的类比关系。我们开发了一个名为 **象征游戏** 的多智能体强化学习测试平台，以研究新兴通信，并提出了一个模型，该模型允许代理利用 **视觉心智理论** 来使用象形文字传达动作。我们的模型位于更广泛的动物通信形式化框架之内，揭示了早期书写系统发展的认知和文化过程。 

---
# MeetMap: Real-Time Collaborative Dialogue Mapping with LLMs in Online Meetings 

**Title (ZH)**: MeetMap：在线会议中使用大语言模型进行实时协作对话映射 

**Authors**: Xinyue Chen, Nathan Yap, Xinyi Lu, Aylin Gunal, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01564)  

**Abstract**: Video meeting platforms display conversations linearly through transcripts or summaries. However, ideas during a meeting do not emerge linearly. We leverage LLMs to create dialogue maps in real time to help people visually structure and connect ideas. Balancing the need to reduce the cognitive load on users during the conversation while giving them sufficient control when using AI, we explore two system variants that encompass different levels of AI assistance. In Human-Map, AI generates summaries of conversations as nodes, and users create dialogue maps with the nodes. In AI-Map, AI produces dialogue maps where users can make edits. We ran a within-subject experiment with ten pairs of users, comparing the two MeetMap variants and a baseline. Users preferred MeetMap over traditional methods for taking notes, which aligned better with their mental models of conversations. Users liked the ease of use for AI-Map due to the low effort demands and appreciated the hands-on opportunity in Human-Map for sense-making. 

**Abstract (ZH)**: 视频会议平台通过转录或总结来线性展示对话内容，然而会议中的思想并不是线性的。我们利用大语言模型（LLMs）在会议过程中实时生成对话地图，帮助人们可视化地结构化和连接想法。在减少用户在交流过程中认知负担的同时，提供足够的AI控制，我们探索了两种不同AI辅助水平的系统变体。在Human-Map系统中，AI生成对话摘要作为节点，用户使用这些节点构建对话地图。在AI-Map系统中，AI生成对话地图，用户可以进行编辑。我们进行了一项针对十对用户的内部实验，比较了两种MeetMap变体和一个基线方法。用户更倾向于使用MeetMap而不是传统的笔记方法，这与他们对对话的认知模型更为一致。用户喜欢AI-Map的易用性，因为其对用户努力的需求较低，同时赞赏在Human-Map中进行意义构建的手动机会。 

---
# Search-Based Adversarial Estimates for Improving Sample Efficiency in Off-Policy Reinforcement Learning 

**Title (ZH)**: 基于搜索的对抗估计方法：提高离策强化学习样本效率 

**Authors**: Federico Malato, Ville Hautamaki  

**Link**: [PDF](https://arxiv.org/pdf/2502.01558)  

**Abstract**: Sample inefficiency is a long-lasting challenge in deep reinforcement learning (DRL). Despite dramatic improvements have been made, the problem is far from being solved and is especially challenging in environments with sparse or delayed rewards. In our work, we propose to use Adversarial Estimates as a new, simple and efficient approach to mitigate this problem for a class of feedback-based DRL algorithms. Our approach leverages latent similarity search from a small set of human-collected trajectories to boost learning, using only five minutes of human-recorded experience. The results of our study show algorithms trained with Adversarial Estimates converge faster than their original version. Moreover, we discuss how our approach could enable learning in feedback-based algorithms in extreme scenarios with very sparse rewards. 

**Abstract (ZH)**: 深度强化学习（DRL）中的样本效率低下一直是持续存在的挑战。尽管已经取得了一定的进步，但该问题仍未完全解决，尤其是在稀疏或延迟奖励的环境条件下尤为突出。在我们的工作中，我们提出了一种新颖、简单且高效的对抗估计方法，以缓解此类基于反馈的DRL算法中的样本效率问题。该方法利用少量人工收集轨迹的潜在相似性搜索，通过仅使用五分钟的人工记录经验来提升学习效果。我们的研究结果表明，使用对抗估计训练的算法比原始版本收敛速度更快。此外，我们讨论了该方法如何在极稀疏奖励的极端场景中使基于反馈的算法能够学习。 

---
# Query Brand Entity Linking in E-Commerce Search 

**Title (ZH)**: 电子商务搜索中的查询品牌实体链接 

**Authors**: Dong Liu, Sreyashi Nag  

**Link**: [PDF](https://arxiv.org/pdf/2502.01555)  

**Abstract**: In this work, we address the brand entity linking problem for e-commerce search queries. The entity linking task is done by either i)a two-stage process consisting of entity mention detection followed by entity disambiguation or ii) an end-to-end linking approaches that directly fetch the target entity given the input text. The task presents unique challenges: queries are extremely short (averaging 2.4 words), lack natural language structure, and must handle a massive space of unique brands. We present a two-stage approach combining named-entity recognition with matching, and a novel end-to-end solution using extreme multi-class classification. We validate our solutions by both offline benchmarks and the impact of online A/B test. 

**Abstract (ZH)**: 在本文中，我们探讨了电子商务搜索查询中的品牌实体链接问题。实体链接任务可以通过以下两种方式之一完成：i) 一个两阶段过程，包括实体提及检测后跟实体消歧；或ii) 直接从输入文本中获取目标实体的端到端链接方法。该任务具有独特的挑战性：查询极为简短（平均2.4词），缺乏自然语言结构，并且必须处理大量的独特品牌空间。我们提出了一种结合命名实体识别和匹配的两阶段方法，并提出了一种使用极端多类分类的新型端到端解决方案。我们通过离线基准测试和在线A/B测试的效果来验证我们的解决方案。 

---
# FireCastNet: Earth-as-a-Graph for Seasonal Fire Prediction 

**Title (ZH)**: FireCastNet：将地球视为图模型进行季节性森林火灾预测 

**Authors**: Dimitrios Michail, Charalampos Davalas, Lefki-Ioanna Panagiotou, Ioannis Prapas, Spyros Kondylatos, Nikolaos Ioannis Bountos, Ioannis Papoutsis  

**Link**: [PDF](https://arxiv.org/pdf/2502.01550)  

**Abstract**: With climate change expected to exacerbate fire weather conditions, the accurate and timely anticipation of wildfires becomes increasingly crucial for disaster mitigation. In this study, we utilize SeasFire, a comprehensive global wildfire dataset with climate, vegetation, oceanic indices, and human-related variables, to enable seasonal wildfire forecasting with machine learning. For the predictive analysis, we present FireCastNet, a novel architecture which combines a 3D convolutional encoder with GraphCast, originally developed for global short-term weather forecasting using graph neural networks. FireCastNet is trained to capture the context leading to wildfires, at different spatial and temporal scales. Our investigation focuses on assessing the effectiveness of our model in predicting the presence of burned areas at varying forecasting time horizons globally, extending up to six months into the future, and on how different spatial or/and temporal context affects the performance. Our findings demonstrate the potential of deep learning models in seasonal fire forecasting; longer input time-series leads to more robust predictions, while integrating spatial information to capture wildfire spatio-temporal dynamics boosts performance. Finally, our results hint that in order to enhance performance at longer forecasting horizons, a larger receptive field spatially needs to be considered. 

**Abstract (ZH)**: 随着气候变暖预计将加剧火天气条件，准确及时预测野火变得越来越关键，以减轻灾害影响。本研究利用SeasFire，这是一个综合性的全球野火数据集，包含了气候、植被、海洋指数和人类相关的变量，以利用机器学习进行季节性野火预报。在预测分析中，我们提出了一种名为FireCastNet的新型架构，该架构结合了3D卷积编码器和GraphCast，后者最初是用于全球短期天气预报的基于图神经网络的模型。FireCastNet被训练以在不同的空间和时间尺度上捕捉导致野火的情境。我们的研究主要集中在评估我们的模型在不同预报时间范围内的全球范围内预测烧伤区域存在的有效性，以及不同的空间或/和时间背景如何影响性能。研究结果表明，深度学习模型在季节性火情预报中具有潜力；更长的输入时间序列能够产生更稳健的预测，而整合空间信息以捕捉野火的空间-时间动态可以提高性能。最后，研究结果表明，为了在更长的预报时间范围内提高性能，需要考虑更大的空间感受野。 

---
# VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos 

**Title (ZH)**: VideoRAG：极端长上下文视频的检索增强生成 

**Authors**: Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01549)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated remarkable success in enhancing Large Language Models (LLMs) through external knowledge integration, yet its application has primarily focused on textual content, leaving the rich domain of multi-modal video knowledge predominantly unexplored. This paper introduces VideoRAG, the first retrieval-augmented generation framework specifically designed for processing and understanding extremely long-context videos. Our core innovation lies in its dual-channel architecture that seamlessly integrates (i) graph-based textual knowledge grounding for capturing cross-video semantic relationships, and (ii) multi-modal context encoding for efficiently preserving visual features. This novel design empowers VideoRAG to process unlimited-length videos by constructing precise knowledge graphs that span multiple videos while maintaining semantic dependencies through specialized multi-modal retrieval paradigms. Through comprehensive empirical evaluation on our proposed LongerVideos benchmark-comprising over 160 videos totaling 134+ hours across lecture, documentary, and entertainment categories-VideoRAG demonstrates substantial performance compared to existing RAG alternatives and long video understanding methods. The source code of VideoRAG implementation and the benchmark dataset are openly available at: this https URL. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG) 在通过外部知识集成增强大规模语言模型（LLMs）方面取得了显著成功，但其应用主要集中在文本内容上，而多模态视频知识的丰富领域尚未得到充分探索。本文介绍了 VideoRAG，这是首个专门用于处理和理解极长时长视频的检索增强生成框架。我们的核心创新之处在于其双通道架构，该架构无缝地结合了 (i) 基于图的文本知识接地方法，用于捕捉视频间的语义关系，以及 (ii) 多模态上下文编码，用于高效地保留视觉特征。这种新颖设计赋予 VideoRAG 通过构建跨越多个视频的精确知识图谱来处理无限长度视频的能力，同时通过专门的多模态检索范式保持语义依赖性。通过在我们提出的 LongerVideos 基准测试集上进行全面的实证评估（该测试集包含超过 160 个视频，总计 134+ 小时，涵盖讲座、纪录片和娱乐类别），VideoRAG 的性能与现有的 RAG 替代方案和长视频理解方法相比表现出显著优势。VideoRAG 的实现代码和基准数据集均已公开，可通过以下链接访问：this https URL。 

---
# What is a Number, That a Large Language Model May Know It? 

**Title (ZH)**: 什么是数字，以至于大规模语言模型能够理解它？ 

**Authors**: Raja Marjieh, Veniamin Veselovsky, Thomas L. Griffiths, Ilia Sucholutsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.01540)  

**Abstract**: Numbers are a basic part of how humans represent and describe the world around them. As a consequence, learning effective representations of numbers is critical for the success of large language models as they become more integrated into everyday decisions. However, these models face a challenge: depending on context, the same sequence of digit tokens, e.g., 911, can be treated as a number or as a string. What kind of representations arise from this duality, and what are its downstream implications? Using a similarity-based prompting technique from cognitive science, we show that LLMs learn representational spaces that blend string-like and numerical representations. In particular, we show that elicited similarity judgments from these models over integer pairs can be captured by a combination of Levenshtein edit distance and numerical Log-Linear distance, suggesting an entangled representation. In a series of experiments we show how this entanglement is reflected in the latent embeddings, how it can be reduced but not entirely eliminated by context, and how it can propagate into a realistic decision scenario. These results shed light on a representational tension in transformer models that must learn what a number is from text input. 

**Abstract (ZH)**: 数字是人类表示和描述周围世界的基本组成部分。因此，学习有效的数字表示对于大型语言模型的成败至关重要，尤其是在它们越来越融入日常决策的过程中。然而，这些模型面临着一个挑战：在不同的上下文中，同一个数字符序列，例如911，既可以被视为一个数字，也可以被视为一个字符串。这种二元性会产生什么样的表示形式，以及它对下游的影响是什么？通过认知科学中的基于相似性的提示技术，我们展示了大型语言模型学习了一种融合字符串和数字表示的表示空间。具体而言，我们发现，从这些模型对整数对的引发相似性判断可以使用编辑距离（Levenshtein编辑距离）和数值对数线性距离（Log-Linear距离）的组合来捕捉，这表明了一种交织的表示形式。在一系列实验中，我们展示了这种交织如何反映在潜在嵌入中，上下文可以减少但它无法完全消除这种交织，以及这种交织如何传播到现实决策场景中。这些结果揭示了转换器模型中的一个表示上的矛盾：它们必须从文本输入中学习什么是数字。 

---
# Preference Leakage: A Contamination Problem in LLM-as-a-judge 

**Title (ZH)**: 偏好泄漏：LLM作为法官中的污染问题 

**Authors**: Dawei Li, Renliang Sun, Yue Huang, Ming Zhong, Bohan Jiang, Jiawei Han, Xiangliang Zhang, Wei Wang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01534)  

**Abstract**: Large Language Models (LLMs) as judges and LLM-based data synthesis have emerged as two fundamental LLM-driven data annotation methods in model development. While their combination significantly enhances the efficiency of model training and evaluation, little attention has been given to the potential contamination brought by this new model development paradigm. In this work, we expose preference leakage, a contamination problem in LLM-as-a-judge caused by the relatedness between the synthetic data generators and LLM-based evaluators. To study this issue, we first define three common relatednesses between data generator LLM and judge LLM: being the same model, having an inheritance relationship, and belonging to the same model family. Through extensive experiments, we empirically confirm the bias of judges towards their related student models caused by preference leakage across multiple LLM baselines and benchmarks. Further analysis suggests that preference leakage is a pervasive issue that is harder to detect compared to previously identified biases in LLM-as-a-judge scenarios. All of these findings imply that preference leakage is a widespread and challenging problem in the area of LLM-as-a-judge. We release all codes and data at: this https URL. 

**Abstract (ZH)**: 作为法官的大型语言模型（LLMs）和基于LLM的数据合成已经成为了模型开发中两种基本的数据注释方法。尽管它们的结合在显著提高模型训练和评估效率方面发挥了重要作用，但对这种新型模型开发范式带来的潜在污染关注却很少。在本文中，我们揭示了由数据生成器LLM与法官LLM之间的相关性导致的“偏好泄露”这一污染问题。为研究这一问题，我们首先定义了数据生成器LLM与法官LLM之间的三种常见相关性：同一个模型、继承关系以及属于同一个模型家族。通过广泛实验，我们实证性地确认了偏好泄露导致法官倾向于其相关的学生模型，在多个LLM基线和基准中均表现出这种偏见。进一步分析表明，偏好泄露是一个更普遍、更难以检测的问题，相较于之前识别的法官场景中的偏见更为棘手。所有这些发现意味着，偏好泄露在作为法官的LLM领域是一个广泛且具有挑战性的问题。我们已将所有代码和数据发布在此：this https URL。 

---
# Transformers trained on proteins can learn to attend to Euclidean distance 

**Title (ZH)**: 在蛋白质数据上训练的Transformer可以学会关注欧几里得距离 

**Authors**: Isaac Ellmen, Constantin Schneider, Matthew I.J. Raybould, Charlotte M. Deane  

**Link**: [PDF](https://arxiv.org/pdf/2502.01533)  

**Abstract**: While conventional Transformers generally operate on sequence data, they can be used in conjunction with structure models, typically SE(3)-invariant or equivariant graph neural networks (GNNs), for 3D applications such as protein structure modelling. These hybrids typically involve either (1) preprocessing/tokenizing structural features as input for Transformers or (2) taking Transformer embeddings and processing them within a structural representation. However, there is evidence that Transformers can learn to process structural information on their own, such as the AlphaFold3 structural diffusion model. In this work we show that Transformers can function independently as structure models when passed linear embeddings of coordinates. We first provide a theoretical explanation for how Transformers can learn to filter attention as a 3D Gaussian with learned variance. We then validate this theory using both simulated 3D points and in the context of masked token prediction for proteins. Finally, we show that pre-training protein Transformer encoders with structure improves performance on a downstream task, yielding better performance than custom structural models. Together, this work provides a basis for using standard Transformers as hybrid structure-language models. 

**Abstract (ZH)**: 传统的Transformer通常在序列数据上操作，但它们可以结合结构模型使用，如SE(3)-不变或等变图神经网络（GNNs），用于3D应用，例如蛋白质结构建模。这些混合模型通常包括两种方式之一：（1）预处理/标记结构特征作为Transformer的输入，或（2）采用Transformer嵌入并在结构表示内进行处理。然而，有证据表明Transformer可以学会独立处理结构信息，例如AlphaFold3结构扩散模型。在本文中，我们展示了当传递线性坐标嵌入时，Transformer可以独立地充当结构模型。我们首先给出理论解释，说明Transformer如何学习将注意力过滤为具有学习方差的3D高斯。然后，我们使用模拟的3D点和蛋白质的掩码标记预测来验证这一理论。最后，我们展示了使用结构预训练蛋白质Transformer编码器可以提高下游任务的性能，其性能优于定制的结构模型。结合这些结果，本文为使用标准Transformer作为混合结构-语言模型奠定了基础。 

---
# Efficiently Integrate Large Language Models with Visual Perception: A Survey from the Training Paradigm Perspective 

**Title (ZH)**: 从训练范式视角高效整合大规模语言模型与视觉感知：一个综述 

**Authors**: Xiaorui Ma, Haoran Xie, S. Joe Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.01524)  

**Abstract**: The integration of vision-language modalities has been a significant focus in multimodal learning, traditionally relying on Vision-Language Pretrained Models. However, with the advent of Large Language Models (LLMs), there has been a notable shift towards incorporating LLMs with vision modalities. Following this, the training paradigms for incorporating vision modalities into LLMs have evolved. Initially, the approach was to integrate the modalities through pretraining the modality integrator, named Single-stage Tuning. It has since branched out into methods focusing on performance enhancement, denoted as Two-stage Tuning, and those prioritizing parameter efficiency, referred to as Direct Adaptation. However, existing surveys primarily address the latest Vision Large Language Models (VLLMs) with Two-stage Tuning, leaving a gap in understanding the evolution of training paradigms and their unique parameter-efficient considerations. This paper categorizes and reviews 34 VLLMs from top conferences, journals, and highly cited Arxiv papers, focusing on parameter efficiency during adaptation from the training paradigm perspective. We first introduce the architecture of LLMs and parameter-efficient learning methods, followed by a discussion on vision encoders and a comprehensive taxonomy of modality integrators. We then review three training paradigms and their efficiency considerations, summarizing benchmarks in the VLLM field. To gain deeper insights into their effectiveness in parameter efficiency, we compare and discuss the experimental results of representative models, among which the experiment of the Direct Adaptation paradigm is replicated. Providing insights into recent developments and practical uses, this survey is a vital guide for researchers and practitioners navigating the efficient integration of vision modalities into LLMs. 

**Abstract (ZH)**: 视觉语言模态的集成一直是多模态学习中的一个重要研究方向，传统上依赖于视觉语言预训练模型。然而，随着大型语言模型（LLMs）的发展，模型设计越来越多地结合视觉模态和LLMs。随之而来的是，将视觉模态融入LLMs的训练范式也发生了演变。最初的方法是通过预训练模态集成器，这种方法被称为单阶段调优。此后，这一领域分支出了专注于性能提升的方法，称为双阶段调优；以及侧重参数效率的方法，称为直接适应。现有的综述主要关注使用双阶段调优的视觉大型语言模型（VLLMs），却没有全面解释这些训练范式的演变及其独特的参数效率考虑。本文从训练范式视角出发，对来自顶级会议、期刊和高引用量Arxiv论文的34种VLLMs进行了分类和综述，重点关注适应过程中的参数效率。我们首先介绍大型语言模型的架构和参数效率学习方法，随后讨论视觉编码器，并提供模态集成器的全面分类。接着，我们回顾了三种训练范式及其效率考虑，并总结了VLLM领域的基准测试结果。为了更深入地了解这些方法在参数效率方面的有效性，我们比较并讨论了代表性模型的实验结果，其中包括直接适应范式的实验复现。提供这些最新进展和实用应用的见解，本文为研究人员和实践者如何高效地将视觉模态集成到大型语言模型中提供了宝贵的指南。 

---
# Toward Task Generalization via Memory Augmentation in Meta-Reinforcement Learning 

**Title (ZH)**: 通过记忆增强实现元强化学习中的任务泛化direction

或者更正式的表述方式：

通过记忆增强实现元强化学习中的任务泛化

这样翻译既符合学术规范，也保持了原意。 

**Authors**: Kaixi Bao, Chenhao Li, Yarden As, Andreas Krause, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2502.01521)  

**Abstract**: In reinforcement learning (RL), agents often struggle to perform well on tasks that differ from those encountered during training. This limitation presents a challenge to the broader deployment of RL in diverse and dynamic task settings. In this work, we introduce memory augmentation, a memory-based RL approach to improve task generalization. Our approach leverages task-structured augmentations to simulate plausible out-of-distribution scenarios and incorporates memory mechanisms to enable context-aware policy adaptation. Trained on a predefined set of tasks, our policy demonstrates the ability to generalize to unseen tasks through memory augmentation without requiring additional interactions with the environment. Through extensive simulation experiments and real-world hardware evaluations on legged locomotion tasks, we demonstrate that our approach achieves zero-shot generalization to unseen tasks while maintaining robust in-distribution performance and high sample efficiency. 

**Abstract (ZH)**: 在强化学习（RL）中，代理往往难以在与训练过程中遇到的任务不同的任务上表现出色。这一局限性为在多样化和动态的任务设置中广泛部署RL带来了挑战。在这项研究中，我们引入了记忆增强方法，这是一种基于记忆的RL方法，旨在提高任务泛化能力。我们的方法利用结构化任务增强来模拟合理的分布外场景，并集成了记忆机制以实现上下文感知的策略适应。在预定义的任务集上进行训练后，我们的策略能够通过记忆增强实现对未见过任务的泛化，而无需额外与环境交互。通过广泛的仿真实验和真实硬件评估（特别是在腿式运动任务上），我们证明了该方法能够在保持稳健的任务内性能和高样本效率的同时实现零样本泛化到未见过的任务。 

---
# Regularized interpolation in 4D neural fields enables optimization of 3D printed geometries 

**Title (ZH)**: 正则化插值在4D神经场中的应用：优化3D打印几何结构 

**Authors**: Christos Margadji, Andi Kuswoyo, Sebastian W. Pattinson  

**Link**: [PDF](https://arxiv.org/pdf/2502.01517)  

**Abstract**: The ability to accurately produce geometries with specified properties is perhaps the most important characteristic of a manufacturing process. 3D printing is marked by exceptional design freedom and complexity but is also prone to geometric and other defects that must be resolved for it to reach its full potential. Ultimately, this will require both astute design decisions and timely parameter adjustments to maintain stability that is challenging even with expert human operators. While machine learning is widely investigated in 3D printing, existing methods typically overlook spatial features that vary across prints and thus find it difficult to produce desired geometries. Here, we encode volumetric representations of printed parts into neural fields and apply a new regularization strategy, based on minimizing the partial derivative of the field's output with respect to a single, non-learnable parameter. By thus encouraging small input changes to yield only small output variations, we encourage smooth interpolation between observed volumes and hence realistic geometry predictions. This framework therefore allows the extraction of 'imagined' 3D shapes, revealing how a part would look if manufactured under previously unseen parameters. The resulting continuous field is used for data-driven optimization to maximize geometric fidelity between expected and produced geometries, reducing post-processing, material waste, and production costs. By optimizing process parameters dynamically, our approach enables advanced planning strategies, potentially allowing manufacturers to better realize complex and feature-rich designs. 

**Abstract (ZH)**: 具有指定属性的几何精确生成能力可能是制造过程最重要的特征之一。3D打印因其卓越的设计自由度和复杂性而受到重视，但同时也存在几何和其他缺陷，这些缺陷必须解决才能充分发挥其潜力。最终，这将需要敏锐的设计决策和及时的参数调整来维持稳定性，即使是有经验的人类操作者也很难做到这一点。虽然机器学习在3D打印领域受到了广泛研究，但现有方法通常忽视了在不同打印件之间变化的空间特征，从而难以生成所需的几何形状。在此，我们将打印部件的体积表示编码到神经场中，并应用一种新的正则化策略，该策略基于最小化场输出对单个非学习参数的偏导数。通过鼓励小输入变化导致小输出变化，我们鼓励在观察到的体积之间进行平滑内插，从而生成具有现实感的几何预测。因此，该框架允许提取“想象中的”3D形状，揭示在未见过的设计参数条件下制造出的零件将如何呈现。由此得出的连续场用于数据驱动的优化，以最大化预期和制造几何形状之间的几何保真度，从而减少后期处理、材料浪费和生产成本。通过动态优化工艺参数，我们的方法可以实现更先进的规划策略，可能使制造商更好地实现复杂和丰富的设计。 

---
# MoireDB: Formula-generated Interference-fringe Image Dataset 

**Title (ZH)**: MoireDB：公式生成的干涉纹图像数据集 

**Authors**: Yuto Matsuo, Ryo Hayamizu, Hirokatsu Kataoka, Akio Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2502.01490)  

**Abstract**: Image recognition models have struggled to treat recognition robustness to real-world degradations. In this context, data augmentation methods like PixMix improve robustness but rely on generative arts and feature visualizations (FVis), which have copyright, drawing cost, and scalability issues. We propose MoireDB, a formula-generated interference-fringe image dataset for image augmentation enhancing robustness. MoireDB eliminates copyright concerns, reduces dataset assembly costs, and enhances robustness by leveraging illusory patterns. Experiments show that MoireDB augmented images outperforms traditional Fractal arts and FVis-based augmentations, making it a scalable and effective solution for improving model robustness against real-world degradations. 

**Abstract (ZH)**: 图像识别模型在处理真实世界降级的识别稳健性方面一直存在困难。在此背景下，数据增强方法如PixMix可以提高稳健性，但依赖于生成艺术和特征可视化（FVis），存在版权、绘图成本和可扩展性问题。我们提出MoireDB，这是一种由公式生成的莫尔图案图像数据集，用于增强数据增强的稳健性。MoireDB消除了版权问题、降低了数据集组装成本，并通过利用幻象图案来提高稳健性。实验表明，使用MoireDB增强的图像在性能上优于传统的分形艺术和FVis基增强方法，使其成为一个可扩展且有效的解决方案，能够提高模型在面对真实世界降级情况下的稳健性。 

---
# Position: Empowering Time Series Reasoning with Multimodal LLMs 

**Title (ZH)**: 标题：利用多模态大规模语言模型增强时间序列推理 

**Authors**: Yaxuan Kong, Yiyuan Yang, Shiyu Wang, Chenghao Liu, Yuxuan Liang, Ming Jin, Stefan Zohren, Dan Pei, Yan Liu, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.01477)  

**Abstract**: Understanding time series data is crucial for multiple real-world applications. While large language models (LLMs) show promise in time series tasks, current approaches often rely on numerical data alone, overlooking the multimodal nature of time-dependent information, such as textual descriptions, visual data, and audio signals. Moreover, these methods underutilize LLMs' reasoning capabilities, limiting the analysis to surface-level interpretations instead of deeper temporal and multimodal reasoning. In this position paper, we argue that multimodal LLMs (MLLMs) can enable more powerful and flexible reasoning for time series analysis, enhancing decision-making and real-world applications. We call on researchers and practitioners to leverage this potential by developing strategies that prioritize trust, interpretability, and robust reasoning in MLLMs. Lastly, we highlight key research directions, including novel reasoning paradigms, architectural innovations, and domain-specific applications, to advance time series reasoning with MLLMs. 

**Abstract (ZH)**: 理解时间序列数据对于多个实际应用至关重要。虽然大型语言模型（LLMs）在时间序列任务中展现出潜力，当前的方法往往仅依赖于数值数据，而忽略了时间依赖信息的多模态性质，如文本描述、视觉数据和音频信号。此外，这些方法未能充分利用LLMs的推理能力，限制了分析仅停留在表面层面的理解，而非更深层次的时间和多模态推理。在本文中，我们主张多模态LLMs（MLLMs）可以增强时间序列分析中的推理能力，提升决策质量和实际应用效果。我们呼吁研究者和实践者通过发展优先考虑信任、可解释性和稳健推理的策略，充分利用MLLMs的潜力。最后，我们指出了关键的研究方向，包括新的推理范式、架构创新以及特定领域的应用，以促进利用MLLMs进行时间序列推理的研究进展。 

---
# FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model 

**Title (ZH)**: FALCON：通过对比正交非对齐进行精细粒度的激活操纵以优化大型语言模型 

**Authors**: Jinwei Hu, Zhenglin Huang, Xiangyu Yin, Wenjie Ruan, Guangliang Cheng, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01472)  

**Abstract**: Large language models have been widely applied, but can inadvertently encode sensitive or harmful information, raising significant safety concerns. Machine unlearning has emerged to alleviate this concern; however, existing training-time unlearning approaches, relying on coarse-grained loss combinations, have limitations in precisely separating knowledge and balancing removal effectiveness with model utility. In contrast, we propose Fine-grained Activation manipuLation by Contrastive Orthogonal uNalignment (FALCON), a novel representation-guided unlearning approach that leverages information-theoretic guidance for efficient parameter selection, employs contrastive mechanisms to enhance representation separation, and projects conflict gradients onto orthogonal subspaces to resolve conflicts between forgetting and retention objectives. Extensive experiments demonstrate that FALCON achieves superior unlearning effectiveness while maintaining model utility, exhibiting robust resistance against knowledge recovery attempts. 

**Abstract (ZH)**: 大规模语言模型已广泛应用，但可能会无意中编码敏感或有害信息，从而引发重大安全问题。机器遗忘技术已经出现以缓解这一问题；然而，现有的训练时遗忘方法依赖粗粒度的损失组合，这在精确定分离知识和平衡遗忘效果与模型效用方面存在局限性。与此相反，我们提出了一种新的命名杂质细粒度激活操纵通过对比正交unalignment（FALCON）方法，这是一种基于表示的遗忘方法，利用信息论指导进行高效参数选择，采用对比机制增强表示分离，并将冲突梯度投影到正交子空间中以解决遗忘和保留目标之间的冲突。大量实验表明，FALCON在保持模型效用的同时实现了更优的遗忘效果，对知识恢复尝试展现了稳健的抵抗能力。 

---
# Process Reinforcement through Implicit Rewards 

**Title (ZH)**: 通过隐式奖励强化过程 

**Authors**: Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2502.01456)  

**Abstract**: Dense process rewards have proven a more effective alternative to the sparse outcome-level rewards in the inference-time scaling of large language models (LLMs), particularly in tasks requiring complex multi-step reasoning. While dense rewards also offer an appealing choice for the reinforcement learning (RL) of LLMs since their fine-grained rewards have the potential to address some inherent issues of outcome rewards, such as training efficiency and credit assignment, this potential remains largely unrealized. This can be primarily attributed to the challenges of training process reward models (PRMs) online, where collecting high-quality process labels is prohibitively expensive, making them particularly vulnerable to reward hacking. To address these challenges, we propose PRIME (Process Reinforcement through IMplicit rEwards), which enables online PRM updates using only policy rollouts and outcome labels through implict process rewards. PRIME combines well with various advantage functions and forgoes the dedicated reward model training phrase that existing approaches require, substantially reducing the development overhead. We demonstrate PRIME's effectiveness on competitional math and coding. Starting from Qwen2.5-Math-7B-Base, PRIME achieves a 15.1% average improvement across several key reasoning benchmarks over the SFT model. Notably, our resulting model, Eurus-2-7B-PRIME, surpasses Qwen2.5-Math-7B-Instruct on seven reasoning benchmarks with 10% of its training data. 

**Abstract (ZH)**: 在大型语言模型（LLM）的推理时缩放任务中，密集过程奖励已被证明是稀疏最终结果奖励的一种更有效的替代方案，特别是在需要复杂多步推理的任务中。虽然密集奖励也为LLM的强化学习（RL）提供了令人信服的选择，因为它们的细粒度奖励有可能解决最终结果奖励的一些固有问题，如训练效率和奖励归因，但这种潜力仍远未实现。这主要是由于在线训练过程奖励模型（PRM）所面临的挑战，即收集高质量的中间过程标签成本非常高，使其特别容易受到奖励寻优的影响。为了解决这些问题，我们提出了PRIME（过程强化学习通过隐式奖励），它能够通过隐式的中间奖励仅使用策略展开和最终结果标签来实现在线的PRM更新。PRIME能够与各种优势函数很好地结合，并且可以省去现有方法所需的专业奖励模型训练阶段，显著降低了开发成本。我们在综合数学和编程任务上展示了PRIME的有效性。从Qwen2.5-Math-7B-Base开始，PRIME在多个关键推理基准上的平均改进率为15.1%。值得注意的是，我们得到的结果模型Eurus-2-7B-PRIME仅使用其训练数据的10%，就在七个推理基准上超过了Qwen2.5-Math-7B-Instruct模型。 

---
# Temporal-consistent CAMs for Weakly Supervised Video Segmentation in Waste Sorting 

**Title (ZH)**: 基于时间一致性CAMs的弱监督视频垃圾分类分割方法 

**Authors**: Andrea Marelli, Luca Magri, Federica Arrigoni, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.01455)  

**Abstract**: In industrial settings, weakly supervised (WS) methods are usually preferred over their fully supervised (FS) counterparts as they do not require costly manual annotations. Unfortunately, the segmentation masks obtained in the WS regime are typically poor in terms of accuracy. In this work, we present a WS method capable of producing accurate masks for semantic segmentation in the case of video streams. More specifically, we build saliency maps that exploit the temporal coherence between consecutive frames in a video, promoting consistency when objects appear in different frames. We apply our method in a waste-sorting scenario, where we perform weakly supervised video segmentation (WSVS) by training an auxiliary classifier that distinguishes between videos recorded before and after a human operator, who manually removes specific wastes from a conveyor belt. The saliency maps of this classifier identify materials to be removed, and we modify the classifier training to minimize differences between the saliency map of a central frame and those in adjacent frames, after having compensated object displacement. Experiments on a real-world dataset demonstrate the benefits of integrating temporal coherence directly during the training phase of the classifier. Code and dataset are available upon request. 

**Abstract (ZH)**: 在工业环境中，弱监督（WS）方法通常偏好于完全监督（FS）方法，因为它们不需要昂贵的手动标注。不幸的是，在弱监督模式下获得的分割掩码在精度上通常是较差的。在这项工作中，我们提出了一种WS方法，能够在视频流中生成准确的掩码，用于语义分割。具体而言，我们构建了显著性图，利用视频中连续帧之间的时序一致性，促进不同帧中物体出现时的连贯性。我们将该方法应用于废物分类场景，通过训练一个辅助分类器来区分人类操作者手动从传送带上移除特定废物前后的视频片段。该分类器的显著性图可以识别需要移除的材料，并对分类器的训练进行修改，以最小化中心帧和相邻帧显著性图之间的差异，补偿了物体的位移。在真实数据集上的实验表明，直接在分类器的训练阶段整合时序一致性可以带来诸多益处。源代码和数据集可应要求提供。 

---
# Simulating Rumor Spreading in Social Networks using LLM Agents 

**Title (ZH)**: 使用大语言模型代理模拟社交网络中的谣言传播 

**Authors**: Tianrui Hu, Dimitrios Liakopoulos, Xiwen Wei, Radu Marculescu, Neeraja J. Yadwadkar  

**Link**: [PDF](https://arxiv.org/pdf/2502.01450)  

**Abstract**: With the rise of social media, misinformation has become increasingly prevalent, fueled largely by the spread of rumors. This study explores the use of Large Language Model (LLM) agents within a novel framework to simulate and analyze the dynamics of rumor propagation across social networks. To this end, we design a variety of LLM-based agent types and construct four distinct network structures to conduct these simulations. Our framework assesses the effectiveness of different network constructions and agent behaviors in influencing the spread of rumors. Our results demonstrate that the framework can simulate rumor spreading across more than one hundred agents in various networks with thousands of edges. The evaluations indicate that network structure, personas, and spreading schemes can significantly influence rumor dissemination, ranging from no spread to affecting 83\% of agents in iterations, thereby offering a realistic simulation of rumor spread in social networks. 

**Abstract (ZH)**: 随着社交媒体的兴起，虚假信息日益盛行，很大程度上是由于谣言的传播。本研究旨在探索在新型框架中使用大型语言模型（LLM）代理模拟和分析社交网络中谣言传播的动力学。为此，我们设计了多种基于LLM的代理类型，并构建了四种不同的网络结构进行这些模拟。我们的框架评估了不同网络结构和代理行为对谣言传播的影响效果。研究结果表明，该框架可以用于在各种网络中模拟超过一百个代理的谣言传播，涉及数千条边。评估结果表明，网络结构、代理特征以及传播方案可以显著影响谣言的传播范围，从不传播到在迭代过程中影响83%的代理，从而为社交网络中的谣言传播提供了一个现实的模拟。 

---
# SPFFNet: Strip Perception and Feature Fusion Spatial Pyramid Pooling for Fabric Defect Detection 

**Title (ZH)**: SPFFNet：条纹感知与特征融合的空间金字塔池化在纺织缺陷检测中的应用 

**Authors**: Peizhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.01445)  

**Abstract**: Defect detection in fabrics is critical for quality control, yet existing methods often struggle with complex backgrounds and shape-specific defects. In this paper, we propose an improved fabric defect detection model based on YOLOv4. To enhance the detection of strip defects, we introduce a Strip Perception Module (SPM) that improves feature capture through multi-scale convolution. We further enhance the spatial pyramid pooling fast (SPPF) by integrating a squeeze-and-excitation mechanism, resulting in the SE-SPPF module, which better integrates spatial and channel information for more effective defect feature extraction. Additionally, we propose a novel focal enhanced complete intersection over union (FECIoU) metric with adaptive weights, addressing scale differences and class imbalance by adjusting the weights of hard-to-detect instances through focal loss. Experimental results demonstrate that our model achieves a 0.8-8.1% improvement in mean average precision (mAP) on the Tianchi dataset and a 1.6-13.2% improvement on our custom dataset, outperforming other state-of-the-art methods. 

**Abstract (ZH)**: 织物缺陷检测对于质量控制至关重要，但现有方法通常在处理复杂背景和形状特定缺陷方面存在困难。本文提出了一种基于YOLOv4改进的织物缺陷检测模型。为了增强对条纹缺陷的检测能力，我们引入了一种条纹感知模块（SPM），通过多尺度卷积提高特征捕捉能力。我们还通过整合挤压-激励机制对空间金字塔池化快速（SPPF）进行了增强，形成了SE-SPPF模块，更好地整合了空间和通道信息，从而更有效地提取缺陷特征。此外，我们提出了一种新颖的自适应权重增强焦点交并比（FECIoU）指标，通过聚焦损失调整难以检测实例的权重，从而解决大小差异和类别不平衡问题。实验结果表明，我们的模型在Tianchi数据集上平均精度（mAP）提高了0.8-8.1%，在我们自定义的数据集上提高了1.6-13.2%，超过了其他最先进的方法。 

---
# Towards Safer Chatbots: A Framework for Policy Compliance Evaluation of Custom GPTs 

**Title (ZH)**: 朝着更安全的聊天机器人：自定义GPT政策合规性评估框架 

**Authors**: David Rodriguez, William Seymour, Jose M. Del Alamo, Jose Such  

**Link**: [PDF](https://arxiv.org/pdf/2502.01436)  

**Abstract**: Large Language Models (LLMs) have gained unprecedented prominence, achieving widespread adoption across diverse domains and integrating deeply into society. The capability to fine-tune general-purpose LLMs, such as Generative Pre-trained Transformers (GPT), for specific tasks has facilitated the emergence of numerous Custom GPTs. These tailored models are increasingly made available through dedicated marketplaces, such as OpenAI's GPT Store. However, their black-box nature introduces significant safety and compliance risks. In this work, we present a scalable framework for the automated evaluation of Custom GPTs against OpenAI's usage policies, which define the permissible behaviors of these systems. Our framework integrates three core components: (1) automated discovery and data collection of models from the GPT store, (2) a red-teaming prompt generator tailored to specific policy categories and the characteristics of each target GPT, and (3) an LLM-as-a-judge technique to analyze each prompt-response pair for potential policy violations.
We validate our framework with a manually annotated ground truth, and evaluate it through a large-scale study with 782 Custom GPTs across three categories: Romantic, Cybersecurity, and Academic GPTs. Our manual annotation process achieved an F1 score of 0.975 in identifying policy violations, confirming the reliability of the framework's assessments. The results reveal that 58.7% of the analyzed models exhibit indications of non-compliance, exposing weaknesses in the GPT store's review and approval processes. Furthermore, our findings indicate that a model's popularity does not correlate with compliance, and non-compliance issues largely stem from behaviors inherited from base models rather than user-driven customizations. We believe this approach is extendable to other chatbot platforms and policy domains, improving LLM-based systems safety. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在前所未有的重要性下，广泛应用于各个领域，并深度融入社会。对通用语言模型，如生成预训练变换器（GPT）进行特定任务的细调，促成了大量定制化GPTs（Custom GPTs）的涌现。这些定制化模型越来越多地通过专用市场（如OpenAI的GPT Store）提供给用户。然而，它们的黑盒性质引入了重大的安全性和合规性风险。在本文中，我们提出了一个可扩展的框架，用于自动评估定制化GPTs是否遵守OpenAI的使用政策，这些政策界定了这些系统的允许行为。我们的框架整合了三个核心组件：（1）自动发现并从GPT Store收集模型；（2）专为特定政策类别和每个目标GPT的特性量身定制的红队攻击提示生成器；（3）一种LLM作为裁判的技术，用于分析每对提示-响应对是否存在潜在的政策违规行为。

我们使用手动标注的数据集来验证该框架，并通过一项涵盖782个定制化GPTs的大规模研究对其进行了评估，这些GPTs被划分为三类：浪漫主题、网络安全主题和学术主题GPT。我们的手动标注过程在识别政策违规行为时获得了0.975的F1评分，证实了该框架评估结果的可靠性。研究结果表明，58.7%的分析模型表现出违规迹象，揭示了GPT Store审查和批准流程中的薄弱环节。此外，我们的研究结果还表明，模型的受欢迎程度与合规性之间不存在关联，违规问题主要源自基础模型继承的行为，而非用户的定制化更改。我们认为，这种方法可以扩展到其他聊天机器人平台和政策领域，从而提高基于LLM系统的安全性。 

---
# Structural features of the fly olfactory circuit mitigate the stability-plasticity dilemma in continual learning 

**Title (ZH)**: 飞虫嗅觉回路的结构特征缓解了持续学习中的稳定-可塑性困境 

**Authors**: Heming Zou, Yunliang Zang, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.01427)  

**Abstract**: Artificial neural networks face the stability-plasticity dilemma in continual learning, while the brain can maintain memories and remain adaptable. However, the biological strategies for continual learning and their potential to inspire learning algorithms in neural networks are poorly understood. This study presents a minimal model of the fly olfactory circuit to investigate the biological strategies that support continual odor learning. We introduce the fly olfactory circuit as a plug-and-play component, termed the Fly Model, which can integrate with modern machine learning methods to address this dilemma. Our findings demonstrate that the Fly Model enhances both memory stability and learning plasticity, overcoming the limitations of current continual learning strategies. We validated its effectiveness across various challenging continual learning scenarios using commonly used datasets. The fly olfactory system serves as an elegant biological circuit for lifelong learning, offering a module that enhances continual learning with minimal additional computational cost for machine learning. 

**Abstract (ZH)**: 在持续学习中，人工神经网络面临稳定性和可塑性之间的困境，而大脑却能够保持记忆并保持适应性。然而，持续学习的生物策略及其对神经网络学习算法的潜在启发作用仍不甚明了。本研究提出了一种苍蝇嗅觉回路的最小模型，以探讨支持持续气味学习的生物策略。我们将苍蝇嗅觉回路作为可插拔组件引入，称为苍蝇模型，它可以与现代机器学习方法集成，以解决这一困境。我们的研究结果表明，苍蝇模型能够同时增强记忆的稳定性与学习的可塑性，克服现有持续学习策略的局限性。我们采用常用的数据集在各种具有挑战性的持续学习场景中验证了其有效性。苍蝇嗅觉系统为终身学习提供了一个优雅的生物电路模块，能够为机器学习提供一个成本较低的持续学习增强模块。 

---
# Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models 

**Title (ZH)**: 视觉注意力永不衰退：面向多模态大型语言模型的selective progressive attention recalibration方法在细节图像描述中的应用 

**Authors**: Mingi Jung, Saehuyng Lee, Eunji Kim, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2502.01419)  

**Abstract**: Detailed image captioning is essential for tasks like data generation and aiding visually impaired individuals. High-quality captions require a balance between precision and recall, which remains challenging for current multimodal large language models (MLLMs). In this work, we hypothesize that this limitation stems from weakening and increasingly noisy visual attention as responses lengthen. To address this issue, we propose SPARC (Selective Progressive Attention ReCalibration), a training-free method that enhances the contribution of visual tokens during decoding. SPARC is founded on three key observations: (1) increasing the influence of all visual tokens reduces recall; thus, SPARC selectively amplifies visual tokens; (2) as captions lengthen, visual attention becomes noisier, so SPARC identifies critical visual tokens by leveraging attention differences across time steps; (3) as visual attention gradually weakens, SPARC reinforces it to preserve its influence. Our experiments, incorporating both automated and human evaluations, demonstrate that existing methods improve the precision of MLLMs at the cost of recall. In contrast, our proposed method enhances both precision and recall with minimal computational overhead. 

**Abstract (ZH)**: 详细的图像描述对于数据生成和辅助视障人士等任务至关重要。高质量的描述需要在精确度和召回率之间达到平衡，而当前的跨模态大型语言模型（MLLMs）在实现这一平衡方面仍然存在挑战。在本研究中，我们假设这种限制源于随着响应长度增加所导致的视觉注意力减弱及噪音增加。为了解决这个问题，我们提出了SPARC（选择性渐进注意力重新校准）方法，这是一种无需训练的方法，可以增强解码期间视觉词素的贡献。SPARC建立在三个关键观察的基础上：（1）增加所有视觉词素的影响会降低召回率；因此，SPARC有选择地放大关键视觉词素；（2）随着描述长度的增加，视觉注意力变得越来越嘈杂，所以SPARC通过利用不同时刻之间的注意力差异来识别关键的视觉词素；（3）随着视觉注意力逐渐减弱，SPARC通过强化它来保持其影响。我们的实验表明，现有方法在提高精确度的同时会牺牲召回率；相比之下，我们提出的这种方法在不增加大量计算开销的情况下，同时提高了精确度和召回率。 

---
# GRADIEND: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models 

**Title (ZH)**: GRADIEND: 神经网络内部单义特征学习及其在 transformer 模型性别偏差消除中的应用 

**Authors**: Jonathan Drechsel, Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2502.01406)  

**Abstract**: AI systems frequently exhibit and amplify social biases, including gender bias, leading to harmful consequences in critical areas. This study introduces a novel encoder-decoder approach that leverages model gradients to learn a single monosemantic feature neuron encoding gender information. We show that our method can be used to debias transformer-based language models, while maintaining other capabilities. We demonstrate the effectiveness of our approach across multiple encoder-only based models and highlight its potential for broader applications. 

**Abstract (ZH)**: AI系统经常表现出并放大社会偏见，包括性别偏见，这在关键领域可能导致有害后果。本研究提出了一种新颖的编码-解码方法，利用模型梯度学习单一语义特征神经元来编码性别信息。我们证明了该方法可以用于去偏见基于变换器的语言模型，同时保持其他功能。我们展示了该方法在多种基于编码器的模型中的有效性，并强调了其在更广泛应用中的潜力。 

---
# AdaSVD: Adaptive Singular Value Decomposition for Large Language Models 

**Title (ZH)**: AdaSVD：自适应奇异值分解在大规模语言模型中的应用 

**Authors**: Li Zhiteng, Xia Mingyuan, Zhang Jingyuan, Hui Zheng, Kong Linghe, Zhang Yulun, Yang Xiaokang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01403)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in natural language processing (NLP) tasks, yet their substantial memory requirements present significant challenges for deployment on resource-constrained devices. Singular Value Decomposition (SVD) has emerged as a promising compression technique for LLMs, offering considerable reductions in memory overhead. However, existing SVD-based methods often struggle to effectively mitigate the errors introduced by SVD truncation, leading to a noticeable performance gap when compared to the original models. Furthermore, applying a uniform compression ratio across all transformer layers fails to account for the varying importance of different layers. To address these challenges, we propose AdaSVD, an adaptive SVD-based LLM compression approach. Specifically, AdaSVD introduces adaComp, which adaptively compensates for SVD truncation errors by alternately updating the singular matrices U and V^T. Additionally, AdaSVD introduces adaCR, which adaptively assigns layer-specific compression ratios based on the relative importance of each layer. Extensive experiments across multiple LLM families and evaluation metrics demonstrate that AdaSVD consistently outperforms state-of-the-art (SOTA) SVD-based methods, achieving superior performance with significantly reduced memory requirements. The code and models will be available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理（NLP）任务中取得了显著的成果，但其巨大的内存需求为在资源有限的设备上部署带来了重大挑战。奇异值分解（SVD）作为一种为LLMs进行压缩的技术已经展现出巨大的潜力，能够显著减少内存开销。然而，现有的基于SVD的方法往往难以有效缓解由SVD截断引入的误差，导致性能与原始模型相比存在明显差距。此外，对所有变压器层应用统一的压缩比不能考虑到不同层的重要性差异。为了解决这些问题，我们提出了AdaSVD，这是一种自适应的基于SVD的LLMs压缩方法。具体而言，AdaSVD 引入了adaComp，通过交替更新奇异矩阵U和V^T来自适应补偿SVD截断误差。此外，AdaSVD 还引入了adaCR，通过根据每层相对的重要性自适应分配压缩比。在多个LLMs家族和评估指标上的广泛实验表明，AdaSVD 一致地优于最先进的（SOTA）基于SVD的方法，在显著减少内存需求的同时实现了更好的性能。相关代码和模型将发布在以下链接：[此 https URL]。 

---
# Can message-passing GNN approximate triangular factorizations of sparse matrices? 

**Title (ZH)**: 消息传递图神经网络能逼近稀疏矩阵的三角分解吗？ 

**Authors**: Vladislav Trifonov, Ekaterina Muravleva, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2502.01397)  

**Abstract**: We study fundamental limitations of Graph Neural Networks (GNNs) for learning sparse matrix preconditioners. While recent works have shown promising results using GNNs to predict incomplete factorizations, we demonstrate that the local nature of message passing creates inherent barriers for capturing non-local dependencies required for optimal preconditioning. We introduce a new benchmark dataset of matrices where good sparse preconditioners exist but require non-local computations, constructed using both synthetic examples and real-world matrices. Our experimental results show that current GNN architectures struggle to approximate these preconditioners, suggesting the need for new architectural approaches beyond traditional message passing networks. We provide theoretical analysis and empirical evidence to explain these limitations, with implications for the broader use of GNNs in numerical linear algebra. 

**Abstract (ZH)**: 我们研究图神经网络（GNNs）在学习稀疏矩阵预条件器方面的基本限制。尽管最近的研究表明，使用GNNs预测不完全分解具有令人鼓舞的结果，但我们证明了局部消息传递的本性为捕捉最优预条件化所必需的非局部依赖关系设置了固有的障碍。我们引入了一个新的基准数据集，其中包含存在但需要非局部计算的优质稀疏预条件器，该数据集结合了合成示例和现实世界的矩阵。我们的实验结果表明，当前的GNN架构在近似这些预条件器方面存在困难，这表明需要超越传统消息传递网络的新架构方法。我们提供了理论分析和实证证据来解释这些限制，并探讨了这些限制对更广泛使用GNNs在数值线性代数中的影响。 

---
# Learning Traffic Anomalies from Generative Models on Real-Time Observations 

**Title (ZH)**: 从实时观察中利用生成模型学习交通异常 

**Authors**: Fotis I. Giasemis, Alexandros Sopasakis  

**Link**: [PDF](https://arxiv.org/pdf/2502.01391)  

**Abstract**: Accurate detection of traffic anomalies is crucial for effective urban traffic management and congestion mitigation. We use the Spatiotemporal Generative Adversarial Network (STGAN) framework combining Graph Neural Networks and Long Short-Term Memory networks to capture complex spatial and temporal dependencies in traffic data. We apply STGAN to real-time, minute-by-minute observations from 42 traffic cameras across Gothenburg, Sweden, collected over several months in 2020. The images are processed to compute a flow metric representing vehicle density, which serves as input for the model. Training is conducted on data from April to November 2020, and validation is performed on a separate dataset from November 14 to 23, 2020. Our results demonstrate that the model effectively detects traffic anomalies with high precision and low false positive rates. The detected anomalies include camera signal interruptions, visual artifacts, and extreme weather conditions affecting traffic flow. 

**Abstract (ZH)**: 准确检测交通异常对于有效的城市交通管理及缓解交通拥堵至关重要。我们利用结合图神经网络（GNN）和长短期记忆网络（LSTM）的时空生成对抗网络（STGAN）框架，捕捉交通数据中的复杂时空依赖关系。我们将STGAN应用于收集自2020年数月的瑞典哥德堡42个交通摄像头的实时、每分钟观察数据。图像被处理以计算一个表示车辆密度的流动指标，作为模型的输入。训练数据来自2020年4月至11月，验证则在2020年11月14日至23日的独立数据集上进行。实验结果表明，该模型能够以高精度和低误报率有效地检测交通异常。检测到的异常包括摄像头信号中断、视觉伪影以及极端天气条件对交通流的影响。 

---
# Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods 

**Title (ZH)**: 使用策略梯度方法微调离散扩散模型 

**Authors**: Oussama Zekri, Nicolas Boullé  

**Link**: [PDF](https://arxiv.org/pdf/2502.01384)  

**Abstract**: Discrete diffusion models have recently gained significant attention due to their ability to process complex discrete structures for language modeling. However, fine-tuning these models with policy gradient methods, as is commonly done in Reinforcement Learning from Human Feedback (RLHF), remains a challenging task. We propose an efficient, broadly applicable, and theoretically justified policy gradient algorithm, called Score Entropy Policy Optimization (SEPO), for fine-tuning discrete diffusion models over non-differentiable rewards. Our numerical experiments across several discrete generative tasks demonstrate the scalability and efficiency of our method. Our code is available at this https URL 

**Abstract (ZH)**: 近年来，离散扩散模型因其能够处理语言建模中的复杂离散结构而受到广泛关注。然而，使用策略梯度方法对这些模型进行微调，如强化学习从人类反馈（RLHF）中常见的做法，仍然是一个具有挑战性的任务。我们提出了一种高效、广泛适用且具有理论依据的策略梯度算法，称为分数熵策略优化（SEPO），用于在非可微奖励下微调离散扩散模型。我们的数值实验在几个离散生成任务中显示了该方法的可扩展性和效率。我们的代码可在以下链接获取：[此处填写URL] 

---
# Data-Efficient Model for Psychological Resilience Prediction based on Neurological Data 

**Title (ZH)**: 基于神经数据的心理复苏能力高效预测模型 

**Authors**: Zhi Zhang, Yan Liu, Mengxia Gao, Yu Yang, Jiannong Cao, Wai Kai Hou, Shirley Li, Sonata Yau, Yun Kwok Wing, Tatia M. C. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.01377)  

**Abstract**: Psychological resilience, defined as the ability to rebound from adversity, is crucial for mental health. Compared with traditional resilience assessments through self-reported questionnaires, resilience assessments based on neurological data offer more objective results with biological markers, hence significantly enhancing credibility. This paper proposes a novel data-efficient model to address the scarcity of neurological data. We employ Neuro Kolmogorov-Arnold Networks as the structure of the prediction model. In the training stage, a new trait-informed multimodal representation algorithm with a smart chunk technique is proposed to learn the shared latent space with limited data. In the test stage, a new noise-informed inference algorithm is proposed to address the low signal-to-noise ratio of the neurological data. The proposed model not only shows impressive performance on both public datasets and self-constructed datasets but also provides some valuable psychological hypotheses for future research. 

**Abstract (ZH)**: 心理韧性，定义为从逆境中恢复的能力，对于心理健康至关重要。相比之下，基于神经生物学数据的韧性评估，通过生物学标记提供更客观的结果，显著提高了评估的可信度。本文提出了一种新的数据高效模型，以应对神经生物学数据稀缺的问题。我们采用神经柯尔莫哥洛夫-阿诺尔德网络（Neuro Kolmogorov-Arnold Networks）作为预测模型的结构。在训练阶段，我们提出了一种新的基于特质的多模态表示算法，并结合智能分块技术，以有限的数据学习共享的潜在空间。在测试阶段，我们提出了一种新的受噪声影响的推理算法，以应对神经生物学数据中的低信噪比问题。所提出的模型不仅在公共数据集和自建数据集上显示出了令人印象深刻的性能，还为未来的研究提供了一些有价值的心理假设。 

---
# Compact Rule-Based Classifier Learning via Gradient Descent 

**Title (ZH)**: 基于梯度下降的紧凑规则式分类器学习 

**Authors**: Javier Fumanal-Idocin, Raquel Fernandez-Peralta, Javier Andreu-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2502.01375)  

**Abstract**: Rule-based models play a crucial role in scenarios that require transparency and accountable decision-making. However, they primarily consist of discrete parameters and structures, which presents challenges for scalability and optimization. In this work, we introduce a new rule-based classifier trained using gradient descent, in which the user can control the maximum number and length of the rules. For numerical partitions, the user can also control the partitions used with fuzzy sets, which also helps keep the number of partitions small. We perform a series of exhaustive experiments on $40$ datasets to show how this classifier performs in terms of accuracy and rule base size. Then, we compare our results with a genetic search that fits an equivalent classifier and with other explainable and non-explainable state-of-the-art classifiers. Our results show how our method can obtain compact rule bases that use significantly fewer patterns than other rule-based methods and perform better than other explainable classifiers. 

**Abstract (ZH)**: 基于规则的模型在需要透明性和问责制决策的情景中发挥着关键作用。然而，这些模型主要由离散参数和结构构成，这给缩放和优化带来了挑战。本项工作中，我们提出了一种新的基于规则的分类器，该分类器通过梯度下降训练，用户可以控制规则的最大数量和长度。对于数值分区间，用户还可以控制使用模糊集的分区间隔，这也有助于保持区间数量的较小规模。我们对40个数据集进行了详尽的实验，展示了该分类器在准确性和规则库大小方面的性能。然后我们将结果与适合等效分类器的遗传搜索方法以及其他解释性和非解释性最先进的分类器进行了比较。我们的结果显示，与其它基于规则的方法相比，我们的方法能够获得更加紧凑的规则库，并且在解释性分类器中具有更好的性能，且使用了显著 fewer 的模式。 

---
# Meursault as a Data Point 

**Title (ZH)**: 默尔索作为数据点 

**Authors**: Abhinav Pratap, Amit Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2502.01364)  

**Abstract**: In an era dominated by datafication, the reduction of human experiences to quantifiable metrics raises profound philosophical and ethical questions. This paper explores these issues through the lens of Meursault, the protagonist of Albert Camus' The Stranger, whose emotionally detached existence epitomizes the existential concept of absurdity. Using natural language processing (NLP) techniques including emotion detection (BERT), sentiment analysis (VADER), and named entity recognition (spaCy)-this study quantifies key events and behaviors in Meursault's life. Our analysis reveals the inherent limitations of applying algorithmic models to complex human experiences, particularly those rooted in existential alienation and moral ambiguity. By examining how modern AI tools misinterpret Meursault's actions and emotions, this research underscores the broader ethical dilemmas of reducing nuanced human narratives to data points, challenging the foundational assumptions of our data-driven society. The findings presented in this paper serve as a critique of the increasing reliance on data-driven narratives and advocate for incorporating humanistic values in artificial intelligence. 

**Abstract (ZH)**: 在数据化主导的时代，将人类经历转化为可量化的指标引发了深刻而重要的哲学和伦理问题。本文通过阿尔贝·加缪的《异乡人》中的主人公梅尔杜姆的冷酷存在，探讨了这些问题，他的冷淡生活体现了存在主义中荒诞的概念。通过自然语言处理（NLP）技术，包括情感检测（BERT）、情感分析（VADER）和命名实体识别（spaCy），本文量化了梅尔杜姆生活中关键事件和行为。我们的分析揭示了将计算模型应用于复杂的人类体验，尤其是那些根植于存在主义疏离和道德模糊性中的体验时的内在局限性。通过研究现代人工智能工具如何误解梅尔杜姆的行为和情感，本研究凸显了将复杂的个人叙事简化为数据点所面临的更广泛伦理问题，从而挑战了我们数据驱动社会的基本假设。本文的研究结果是对日益依赖数据驱动叙述的批判，并倡导在人工智能中融入人文学科的价值观。 

---
# Activation by Interval-wise Dropout: A Simple Way to Prevent Neural Networks from Plasticity Loss 

**Title (ZH)**: 区间Dropout激活：一种简单的方法防止神经网络丧失塑性 

**Authors**: Sangyeon Park, Isaac Han, Seungwon Oh, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.01342)  

**Abstract**: Plasticity loss, a critical challenge in neural network training, limits a model's ability to adapt to new tasks or shifts in data distribution. This paper introduces AID (Activation by Interval-wise Dropout), a novel method inspired by Dropout, designed to address plasticity loss. Unlike Dropout, AID generates subnetworks by applying Dropout with different probabilities on each preactivation interval. Theoretical analysis reveals that AID regularizes the network, promoting behavior analogous to that of deep linear networks, which do not suffer from plasticity loss. We validate the effectiveness of AID in maintaining plasticity across various benchmarks, including continual learning tasks on standard image classification datasets such as CIFAR10, CIFAR100, and TinyImageNet. Furthermore, we show that AID enhances reinforcement learning performance in the Arcade Learning Environment benchmark. 

**Abstract (ZH)**: 塑料性损失是神经网络训练过程中的一项关键挑战，它限制了模型适应新任务或数据分布变化的能力。本文介绍了AID（区间间隙Dropout）方法，这是一种受Dropout启发的新方法，旨在解决塑料性损失问题。与Dropout不同的是，AID通过在每个前激活区间上应用不同的Dropout概率来生成子网络。理论分析表明，AID可以正则化网络，使其行为类似于不会遭受塑料性损失的深层线性网络。我们通过各种基准测试验证了AID在保持塑料性方面的有效性，包括在标准图像分类数据集CIFAR10、CIFAR100和TinyImageNet上的连续学习任务。此外，我们还展示了AID在 Arcade Learning Environment 基准测试中的强化学习性能提升。 

---
# Learning Fused State Representations for Control from Multi-View Observations 

**Title (ZH)**: 从多视角观察中学习融合状态表示的控制算法 

**Authors**: Zeyu Wang, Yao-Hui Li, Xin Li, Hongyu Zang, Romain Laroche, Riashat Islam  

**Link**: [PDF](https://arxiv.org/pdf/2502.01316)  

**Abstract**: Multi-View Reinforcement Learning (MVRL) seeks to provide agents with multi-view observations, enabling them to perceive environment with greater effectiveness and precision. Recent advancements in MVRL focus on extracting latent representations from multiview observations and leveraging them in control tasks. However, it is not straightforward to learn compact and task-relevant representations, particularly in the presence of redundancy, distracting information, or missing views. In this paper, we propose Multi-view Fusion State for Control (MFSC), firstly incorporating bisimulation metric learning into MVRL to learn task-relevant representations. Furthermore, we propose a multiview-based mask and latent reconstruction auxiliary task that exploits shared information across views and improves MFSC's robustness in missing views by introducing a mask token. Extensive experimental results demonstrate that our method outperforms existing approaches in MVRL tasks. Even in more realistic scenarios with interference or missing views, MFSC consistently maintains high performance. 

**Abstract (ZH)**: 多视图强化学习（MVRL）旨在为智能体提供多视图观察，使其能够以更高的准确性和有效性感知环境。近年来，MVRL 的进展主要集中在从多视图观察中提取潜在表示，并在控制任务中利用这些表示。然而，在存在冗余信息、干扰信息或缺失视图的情况下，学习紧凑且任务相关的表示并不容易。在本文中，我们提出了一种名为多视图融合状态用于控制（MFSC）的方法，首次将bisimulation度量学习集成到MVRL中，以学习任务相关的表示。此外，我们提出了一种基于多视图的掩码和潜在重构辅助任务，利用跨视图的共享信息并引入掩码令牌以提高MFSC在缺失视图情况下的鲁棒性。广泛的实验结果表明，我们的方法在多视图强化学习任务中优于现有方法。即使在存在干扰或视图缺失的更现实场景中，MFSC也表现出稳定的高度性能。 

---
# TFBS-Finder: Deep Learning-based Model with DNABERT and Convolutional Networks to Predict Transcription Factor Binding Sites 

**Title (ZH)**: TFBS-Finder: 基于DNABERT和卷积网络的深度学习模型用于预测转录因子结合位点 

**Authors**: Nimisha Ghosh, Pratik Dutta, Daniele Santoni  

**Link**: [PDF](https://arxiv.org/pdf/2502.01311)  

**Abstract**: Transcription factors are proteins that regulate the expression of genes by binding to specific genomic regions known as Transcription Factor Binding Sites (TFBSs), typically located in the promoter regions of those genes. Accurate prediction of these binding sites is essential for understanding the complex gene regulatory networks underlying various cellular functions. In this regard, many deep learning models have been developed for such prediction, but there is still scope of improvement. In this work, we have developed a deep learning model which uses pre-trained DNABERT, a Convolutional Neural Network (CNN) module, a Modified Convolutional Block Attention Module (MCBAM), a Multi-Scale Convolutions with Attention (MSCA) module and an output module. The pre-trained DNABERT is used for sequence embedding, thereby capturing the long-term dependencies in the DNA sequences while the CNN, MCBAM and MSCA modules are useful in extracting higher-order local features. TFBS-Finder is trained and tested on 165 ENCODE ChIP-seq datasets. We have also performed ablation studies as well as cross-cell line validations and comparisons with other models. The experimental results show the superiority of the proposed method in predicting TFBSs compared to the existing methodologies. The codes and the relevant datasets are publicly available at this https URL. 

**Abstract (ZH)**: 转录因子是一类蛋白质，它们通过结合特定的基因组区域（称为转录因子结合位点，TFBSs）来调控基因的表达，这些区域通常位于这些基因的启动子区域。准确预测这些结合位点对于理解各种细胞功能下的复杂基因调控网络至关重要。为此，已经开发了许多深度学习模型来进行这种预测，但仍然有改进的空间。在这项工作中，我们开发了一个深度学习模型，该模型使用预训练的 DNABERT、卷积神经网络（CNN）模块、修改的卷积块注意力模块（MCBAM）、多尺度卷积注意力模块（MSCA）以及输出模块。预训练的 DNABERT 用于序列嵌入，从而捕获 DNA 序列中的长期依赖关系，而 CNN、MCBAM 和 MSCA 模块则有助于提取高阶局部特征。TFBS-Finder 在 165 个 ENCODE ChIP-seq 数据集中进行了训练和测试。我们还进行了消融研究、跨细胞系验证和与其他模型的比较。实验结果表明，与现有方法相比，本方法在预测 TFBSs 方面具有优越性。代码和相关的数据集在此处公开获取：[提供网址]。 

---
# A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers 

**Title (ZH)**: 从统计学习角度审视半对偶对抗神经最优输运求解器 

**Authors**: Roman Tarasov, Petr Mokrov, Milena Gazdieva, Evgeny Burnaev, Alexander Korotin  

**Link**: [PDF](https://arxiv.org/pdf/2502.01310)  

**Abstract**: Neural network based Optimal Transport (OT) is a recent and fruitful direction in the generative modeling community. It finds its applications in various fields such as domain translation, image super-resolution, computational biology and others. Among the existing approaches to OT, of considerable interest are adversarial minimax solvers based on semi-dual formulations of OT problems. While promising, these methods lack theoretical investigation from a statistical learning perspective. Our work fills this gap by establishing upper bounds on the generalization error of an approximate OT map recovered by the minimax quadratic OT solver. Importantly, the bounds we derive depend solely on some standard statistical and mathematical properties of the considered functional classes (neural networks). While our analysis focuses on the quadratic OT, we believe that similar bounds could be derived for more general OT formulations, paving the promising direction for future research. 

**Abstract (ZH)**: 基于神经网络的最优传输（OT）是一种最近在生成建模社区中显示出丰富成果的方向。它在领域转换、图像超分辨率、计算生物学等多个领域找到了应用。在现有的最优传输方法中，基于最优传输问题半对偶形式的对抗最小最大求解器尤其值得关注。虽然这些方法具有前景，但它们缺乏从统计学习的角度进行的理论研究。我们通过建立最小最大二次最优传输求解器恢复的近似最优传输映射的泛化误差上界，填补了这一空白。重要的是，我们推导的界仅依赖于所考虑的功能类（神经网络）的一些标准统计和数学性质。尽管我们的分析集中在二次最优传输上，但我们相信可以为更一般的最优传输形式推导类似的界，为未来的研究开辟了有希望的方向。 

---
# Partial Channel Network: Compute Fewer, Perform Better 

**Title (ZH)**: 部分通道网络：计算更少，性能更优 

**Authors**: Haiduo Huang, Tian Xia, Wenzhe zhao, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.01303)  

**Abstract**: Designing a module or mechanism that enables a network to maintain low parameters and FLOPs without sacrificing accuracy and throughput remains a challenge. To address this challenge and exploit the redundancy within feature map channels, we propose a new solution: partial channel mechanism (PCM). Specifically, through the split operation, the feature map channels are divided into different parts, with each part corresponding to different operations, such as convolution, attention, pooling, and identity mapping. Based on this assumption, we introduce a novel partial attention convolution (PATConv) that can efficiently combine convolution with visual attention. Our exploration indicates that the PATConv can completely replace both the regular convolution and the regular visual attention while reducing model parameters and FLOPs. Moreover, PATConv can derive three new types of blocks: Partial Channel-Attention block (PAT_ch), Partial Spatial-Attention block (PAT_sp), and Partial Self-Attention block (PAT_sf). In addition, we propose a novel dynamic partial convolution (DPConv) that can adaptively learn the proportion of split channels in different layers to achieve better trade-offs. Building on PATConv and DPConv, we propose a new hybrid network family, named PartialNet, which achieves superior top-1 accuracy and inference speed compared to some SOTA models on ImageNet-1K classification and excels in both detection and segmentation on the COCO dataset. Our code is available at this https URL. 

**Abstract (ZH)**: 设计一种机制或模块，能够在保持低参数量和FLOPs的同时不牺牲准确性和吞吐量，仍然是一个挑战。为了解决这一挑战并利用特征图通道内的冗余度，我们提出了一种新的解决方案：部分通道机制（Partial Channel Mechanism, PCM）。具体而言，通过分拆操作，将特征图通道分为不同的部分，每个部分对应不同的操作，如卷积、注意、池化和恒等映射。基于此假设，我们引入了一种新颖的部分注意力卷积（Partial Attention Convolution, PATConv），它可以高效地将卷积与视觉注意结合起来。我们的探索表明，PATConv可以完全替代常规卷积和常规视觉注意力，同时减少模型参数量和FLOPs。此外，PATConv可以派生出三种新的模块类型：部分通道-注意力模块（Partial Channel-Attention block, PAT_ch）、部分空间-注意力模块（Partial Spatial-Attention block, PAT_sp）和部分自我-注意力模块（Partial Self-Attention block, PAT_sf）。此外，我们还提出了一种新颖的动态部分卷积（Dynamic Partial Convolution, DPConv），它可以自适应地学习不同层中分拆通道的比例，从而实现更好的权衡。基于PATConv和DPConv，我们提出了一种新的混合网络家族，名为PartialNet，在ImageNet-1K分类任务上优于一些当前最佳模型，并在COCO数据集上的检测和分割任务上表现出色。我们的代码可通过以下链接获得：[请插入代码链接]。 

---
# Common Foundations for SHACL, ShEx, and PG-Schema 

**Title (ZH)**: 《SHACL、ShEx和PG-Schema的共同基础》 

**Authors**: S. Ahmetaj, I. Boneva, J. Hidders, K. Hose, M. Jakubowski, J.E. Labra-Gayo, W. Martens, F. Mogavero, F. Murlak, C. Okulmus, A. Polleres, O. Savkovic, M. Simkus, D. Tomaszuk  

**Link**: [PDF](https://arxiv.org/pdf/2502.01295)  

**Abstract**: Graphs have emerged as an important foundation for a variety of applications, including capturing and reasoning over factual knowledge, semantic data integration, social networks, and providing factual knowledge for machine learning algorithms. To formalise certain properties of the data and to ensure data quality, there is a need to describe the schema of such graphs. Because of the breadth of applications and availability of different data models, such as RDF and property graphs, both the Semantic Web and the database community have independently developed graph schema languages: SHACL, ShEx, and PG-Schema. Each language has its unique approach to defining constraints and validating graph data, leaving potential users in the dark about their commonalities and differences. In this paper, we provide formal, concise definitions of the core components of each of these schema languages. We employ a uniform framework to facilitate a comprehensive comparison between the languages and identify a common set of functionalities, shedding light on both overlapping and distinctive features of the three languages. 

**Abstract (ZH)**: 图数据已经成为各种应用的重要基石，包括事实知识的捕获和推理、语义数据集成、社会网络以及为机器学习算法提供事实知识等。为了正式化数据的某些属性并确保数据质量，需要描述此类图的数据结构。由于应用范围广泛且存在不同的数据模型，如RDF和属性图，语义网和数据库社区分别独立开发了图数据结构语言：SHACL、ShEx和PG-Schema。每种语言都有其独特的约束定义和图数据验证方法，这使得潜在用户难以了解它们之间的共性和差异。在本文中，我们将提供每种这些数据结构语言核心组件的正式、简洁定义，并采用统一的框架进行全面比较，以识别出三种语言共有的功能，揭示其共同和独特特性。 

---
# Rational Gaussian wavelets and corresponding model driven neural networks 

**Title (ZH)**: 理性高斯小波及其相应的模型驱动神经网络 

**Authors**: Attila Miklós Ámon, Kristian Fenech, Péter Kovács, Tamás Dózsa  

**Link**: [PDF](https://arxiv.org/pdf/2502.01282)  

**Abstract**: In this paper we consider the continuous wavelet transform using Gaussian wavelets multiplied by an appropriate rational term. The zeros and poles of this rational modifier act as free parameters and their choice highly influences the shape of the mother wavelet. This allows the proposed construction to approximate signals with complex morphology using only a few wavelet coefficients. We show that the proposed rational Gaussian wavelets are admissible and provide numerical approximations of the wavelet coefficients using variable projection operators. In addition, we show how the proposed variable projection based rational Gaussian wavelet transform can be used in neural networks to obtain a highly interpretable feature learning layer. We demonstrate the effectiveness of the proposed scheme through a biomedical application, namely, the detection of ventricular ectopic beats (VEBs) in real ECG measurements. 

**Abstract (ZH)**: 在本文中，我们考虑了使用乘以适当有理因子的高斯波let的连续小波变换。该有理解调因子的零点和极点作为自由参数，其选择对母波let的形式有重大影响。这使得所提出的构造能够仅用少量波let系数拟合具有复杂形态的信号。我们证明了所提出的有理高斯波let是可接受的，并通过可变量投影算子提供了波let系数的数值逼近。此外，我们展示了基于可变量投影的有理高斯波let变换如何在神经网络中使用，以获得高度可解释的特征学习层。我们通过一个生物医学应用——在实际心电图（ECG）测量中检测室性早搏（VEBs）——来验证所提出方案的有效性。 

---
# HyperSHAP: Shapley Values and Interactions for Hyperparameter Importance 

**Title (ZH)**: HyperSHAP：超参数重要性评估的Shapley值与交互效应 

**Authors**: Marcel Wever, Maximilian Muschalik, Fabian Fumagalli, Marius Lindauer  

**Link**: [PDF](https://arxiv.org/pdf/2502.01276)  

**Abstract**: Hyperparameter optimization (HPO) is a crucial step in achieving strong predictive performance. However, the impact of individual hyperparameters on model generalization is highly context-dependent, prohibiting a one-size-fits-all solution and requiring opaque automated machine learning (AutoML) systems to find optimal configurations. The black-box nature of most AutoML systems undermines user trust and discourages adoption. To address this, we propose a game-theoretic explainability framework for HPO that is based on Shapley values and interactions. Our approach provides an additive decomposition of a performance measure across hyperparameters, enabling local and global explanations of hyperparameter importance and interactions. The framework, named HyperSHAP, offers insights into ablations, the tunability of learning algorithms, and optimizer behavior across different hyperparameter spaces. We evaluate HyperSHAP on various HPO benchmarks by analyzing the interaction structure of the HPO problem. Our results show that while higher-order interactions exist, most performance improvements can be explained by focusing on lower-order representations. 

**Abstract (ZH)**: 超参数优化（HPO）是实现强大预测性能的关键步骤。然而，个体超参数对模型泛化的具体影响高度依赖于上下文，这限制了“一刀切”解决方案的有效性，并要求不透明的自动化机器学习（AutoML）系统找到最优配置。大多数AutoML系统的黑盒特性削弱了用户信任，并阻碍了其采用。为解决这一问题，我们提出了一种基于Shapley值和交互的博弈论解释性框架用于HPO。我们的方法提供了对性能测度在超参数上的加性分解，从而能够提供超参数重要性和交互的局部和全局解释。该框架命名为HyperSHAP，能够揭示超参数空间中不同学习算法的可调性和优化器行为的见解。我们通过分析HPO问题的交互结构，对HyperSHAP在多种HPO基准上的性能进行了评估。结果显示，尽管存在高阶交互，但大多数性能提升可以通过关注较低阶的表示来解释。 

---
# Analysis of Student-LLM Interaction in a Software Engineering Project 

**Title (ZH)**: 软件工程项目中学生与大语言模型互动的分析 

**Authors**: Agrawal Naman, Ridwan Shariffdeen, Guanlin Wang, Sanka Rasnayaka, Ganesh Neelakanta Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2502.01273)  

**Abstract**: Large Language Models (LLMs) are becoming increasingly competent across various domains, educators are showing a growing interest in integrating these LLMs into the learning process. Especially in software engineering, LLMs have demonstrated qualitatively better capabilities in code summarization, code generation, and debugging. Despite various research on LLMs for software engineering tasks in practice, limited research captures the benefits of LLMs for pedagogical advancements and their impact on the student learning process. To this extent, we analyze 126 undergraduate students' interaction with an AI assistant during a 13-week semester to understand the benefits of AI for software engineering learning. We analyze the conversations, code generated, code utilized, and the human intervention levels to integrate the code into the code base.
Our findings suggest that students prefer ChatGPT over CoPilot. Our analysis also finds that ChatGPT generates responses with lower computational complexity compared to CoPilot. Furthermore, conversational-based interaction helps improve the quality of the code generated compared to auto-generated code. Early adoption of LLMs in software engineering is crucial to remain competitive in the rapidly developing landscape. Hence, the next generation of software engineers must acquire the necessary skills to interact with AI to improve productivity. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域的能力不断提升，教育工作者对将这些LLMs整合到学习过程中表现出越来越浓厚的兴趣。特别是在软件工程领域，LLMs在代码summarization、代码生成和调试方面展现了更高质量的能力。尽管已有大量研究探讨了LLMs在软件工程任务中的应用，但有限的研究关注于LLMs在教育进步中的益处及其对学生学习过程的影响。为此，我们分析了126名本科生在为期13周的课程中与AI助手的互动，以理解AI对软件工程学习的益处。我们分析了对话内容、生成的代码、使用的代码以及人工干预程度，以整合代码到代码库中。

我们的研究表明，学生倾向于使用ChatGPT而非CoPilot。我们的分析还发现，与CoPilot相比，ChatGPT生成的响应具有更低的计算复杂性。此外，基于对话的交互有助于生成比自动生成的代码更高的质量代码。在快速发展的软件工程领域尽早采用LLMs至关重要，以保持竞争力。因此，下一代软件工程师必须掌握与AI交互的技能，以提高生产效率。 

---
# Resilient UAV Trajectory Planning via Few-Shot Meta-Offline Reinforcement Learning 

**Title (ZH)**: 基于少样本元离线强化学习的鲁棒无人机航迹规划 

**Authors**: Eslam Eldeeb, Hirley Alves  

**Link**: [PDF](https://arxiv.org/pdf/2502.01268)  

**Abstract**: Reinforcement learning (RL) has been a promising essence in future 5G-beyond and 6G systems. Its main advantage lies in its robust model-free decision-making in complex and large-dimension wireless environments. However, most existing RL frameworks rely on online interaction with the environment, which might not be feasible due to safety and cost concerns. Another problem with online RL is the lack of scalability of the designed algorithm with dynamic or new environments. This work proposes a novel, resilient, few-shot meta-offline RL algorithm combining offline RL using conservative Q-learning (CQL) and meta-learning using model-agnostic meta-learning (MAML). The proposed algorithm can train RL models using static offline datasets without any online interaction with the environments. In addition, with the aid of MAML, the proposed model can be scaled up to new unseen environments. We showcase the proposed algorithm for optimizing an unmanned aerial vehicle (UAV) 's trajectory and scheduling policy to minimize the age-of-information (AoI) and transmission power of limited-power devices. Numerical results show that the proposed few-shot meta-offline RL algorithm converges faster than baseline schemes, such as deep Q-networks and CQL. In addition, it is the only algorithm that can achieve optimal joint AoI and transmission power using an offline dataset with few shots of data points and is resilient to network failures due to unprecedented environmental changes. 

**Abstract (ZH)**: 强化学习（Reinforcement Learning, RL）在未来的5G-beyond和6G系统中展现出了令人振奋的应用前景。其主要优势在于能够在复杂的高维无线环境中实现稳健的无模型决策。然而，现有的大多数RL框架依赖于与环境的在线交互，这可能会因为安全和成本问题而不可行。在线RL的另一个问题是其设计的算法在动态或新环境中缺乏可扩展性。本研究提出了一种新颖的、鲁棒性强的少次元元离线RL算法，该算法将离线RL中的保守Q学习（Conservative Q-learning, CQL）与元学习中的无模型元学习（Model-Agnostic Meta-Learning, MAML）相结合。该提出的算法可以使用静态离线数据集训练RL模型，而无需与环境进行任何在线交互。此外，在MAML的辅助下，提出的模型可以扩展到新的未见过的环境。我们展示了该算法用于优化无人驾驶飞行器（UAV）的轨迹和调度策略，以最小化信息年龄（Age-of-Information, AoI）和有限功率设备的传输功率。数值结果表明，提出的少次元元离线RL算法收敛速度比基线方案（如深度Q网络和CQL）更快。此外，它是唯一一种能够在少量数据点的离线数据集中实现AoI和传输功率的最优联合，并且由于前所未有的环境变化而具有网络容错性的算法。 

---
# Learnable polynomial, trigonometric, and tropical activations 

**Title (ZH)**: 可学习的多项式、三角和热带激活函数 

**Authors**: Ismail Khalfaoui-Hassani, Stefan Kesselheim  

**Link**: [PDF](https://arxiv.org/pdf/2502.01247)  

**Abstract**: This paper investigates scalable neural networks with learnable activation functions based on orthogonal function bases and tropical polynomials, targeting ImageNet-1K classification and next token prediction on OpenWebText. Traditional activations, such as ReLU, are static. In contrast, learnable activations enable the network to adapt dynamically during training. However, stability issues, such as vanishing or exploding gradients, arise with improper variance management in deeper networks. To remedy this, we propose an initialization scheme that single-handedly preserves unitary variance in transformers and convolutional networks, ensuring stable gradient flow even in deep architectures. Extensive experiments demonstrate that networks with Hermite, Fourier, and Tropical-based learnable activations significantly improve over GPT-2 and ConvNeXt networks in terms of accuracy and perplexity in train and test, highlighting the viability of learnable activations in large-scale tasks. The activation functions developed here are the subject of a library coded entirely in pure PyTorch: torchortho, available at this https URL. 

**Abstract (ZH)**: 本文研究了基于正交函数基和热带多项式的可学习激活函数的可扩展神经网络，旨在针对ImageNet-1K分类任务和OpenWebText上的下一个token预测任务。传统的激活函数，如ReLU，是静态的。相比之下，可学习的激活函数使网络能够在训练过程中动态适应。然而，在更深的网络中，不恰当的方差管理会导致梯度消失或爆炸等问题。为了解决这一问题，我们提出了一种初始化方案，能够单独保持变换器和卷积网络中的单位方差，确保即使在深层架构中也能稳定地流动梯度。广泛的实验结果表明，基于厄米特、傅里叶和热带多项式的可学习激活函数在网络的训练和测试中相对于GPT-2和ConvNeXt网络在准确性和困惑度方面有显著提升，突显了可学习激活函数在大规模任务中的可行性。本文中开发的激活函数将在全部使用纯PyTorch编写的库torchortho中提供，可供访问：[此处提供具体网址]。 

---
# OphthBench: A Comprehensive Benchmark for Evaluating Large Language Models in Chinese Ophthalmology 

**Title (ZH)**: OphthBench：评估中文眼科领域大型语言模型的综合性基准 

**Authors**: Chengfeng Zhou, Ji Wang, Juanjuan Qin, Yining Wang, Ling Sun, Weiwei Dai  

**Link**: [PDF](https://arxiv.org/pdf/2502.01243)  

**Abstract**: Large language models (LLMs) have shown significant promise across various medical applications, with ophthalmology being a notable area of focus. Many ophthalmic tasks have shown substantial improvement through the integration of LLMs. However, before these models can be widely adopted in clinical practice, evaluating their capabilities and identifying their limitations is crucial. To address this research gap and support the real-world application of LLMs, we introduce the OphthBench, a specialized benchmark designed to assess LLM performance within the context of Chinese ophthalmic practices. This benchmark systematically divides a typical ophthalmic clinical workflow into five key scenarios: Education, Triage, Diagnosis, Treatment, and Prognosis. For each scenario, we developed multiple tasks featuring diverse question types, resulting in a comprehensive benchmark comprising 9 tasks and 591 questions. This comprehensive framework allows for a thorough assessment of LLMs' capabilities and provides insights into their practical application in Chinese ophthalmology. Using this benchmark, we conducted extensive experiments and analyzed the results from 39 popular LLMs. Our evaluation highlights the current gap between LLM development and its practical utility in clinical settings, providing a clear direction for future advancements. By bridging this gap, we aim to unlock the potential of LLMs and advance their development in ophthalmology. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种医疗应用中展现了显著的潜力，眼科领域尤为突出。许多眼科任务通过整合LLMs后显示出显著的改进。然而，在这些模型能够广泛应用于临床实践之前，对其能力进行评估并识别其局限性至关重要。为填补这一研究空白并支持LLMs的实际应用，我们介绍了OphthBench，这是一种专门设计的基准测试，用于评估LLMs在中文眼科实践中的表现。该基准测试通过系统地将典型的眼科临床工作流程分为五个核心场景：教育、分诊、诊断、治疗和预后。针对每个场景，我们发展了多种任务，涵盖了多样化的提问类型，最终形成了包含9个任务和591个问题的综合基准测试。这一综合框架能够对LLMs的能力进行全面评估，并提供有关其在中文眼科中实际应用的见解。使用该基准测试，我们进行了广泛的实验并分析了来自39种流行LLMs的结果。我们的评估突显了当前LLM开发与其在临床环境中的实用价值之间的差距，并为未来的进步提供了明确的方向。通过缩小这一差距，我们旨在充分发挥LLMs的潜力，并推动其在眼科中的发展。 

---
# Eliciting Language Model Behaviors with Investigator Agents 

**Title (ZH)**: 使用调查员代理引导语言模型行为 

**Authors**: Xiang Lisa Li, Neil Chowdhury, Daniel D. Johnson, Tatsunori Hashimoto, Percy Liang, Sarah Schwettmann, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2502.01236)  

**Abstract**: Language models exhibit complex, diverse behaviors when prompted with free-form text, making it difficult to characterize the space of possible outputs. We study the problem of behavior elicitation, where the goal is to search for prompts that induce specific target behaviors (e.g., hallucinations or harmful responses) from a target language model. To navigate the exponentially large space of possible prompts, we train investigator models to map randomly-chosen target behaviors to a diverse distribution of outputs that elicit them, similar to amortized Bayesian inference. We do this through supervised fine-tuning, reinforcement learning via DPO, and a novel Frank-Wolfe training objective to iteratively discover diverse prompting strategies. Our investigator models surface a variety of effective and human-interpretable prompts leading to jailbreaks, hallucinations, and open-ended aberrant behaviors, obtaining a 100% attack success rate on a subset of AdvBench (Harmful Behaviors) and an 85% hallucination rate. 

**Abstract (ZH)**: 当语言模型受到自由文本的提示时，它们会表现出复杂多样的行为，这使得很难刻画可能输出的空间。我们研究了一个称为行为激发的问题，目标是在特定语言模型中寻找能够诱导特定目标行为（如幻觉或有害响应）的提示。为了在指数级庞大的提示空间中导航，我们训练了调查模型，使其能够将随机选择的目标行为映射到能够诱发这些行为的多样化输出分布，这种方法类似于委托贝叶斯推断。我们通过监督微调、基于DPO的强化学习以及一种新的Frank-Wolfe训练目标来进行这一过程，以逐步发现多样化的提示策略。我们的调查模型揭示了多种有效的并具有人类可解释性的提示，这些提示导致了漏洞利用、幻觉以及开放式的异常行为，并在AdvBench（有害行为）的一组数据上实现了100%的攻击成功率，以及85%的幻觉率。 

---
# One-step full gradient suffices for low-rank fine-tuning, provably and efficiently 

**Title (ZH)**: 一次完整的梯度计算足以用于低秩微调，且可证明和高效 

**Authors**: Yuanhe Zhang, Fanghui Liu, Yudong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.01235)  

**Abstract**: This paper studies how to improve the performance of Low-Rank Adaption (LoRA) as guided by our theoretical analysis. Our first set of theoretical results show that for random initialization and linear models, \textit{i)} LoRA will align to the certain singular subspace of one-step gradient of full fine-tuning; \textit{ii)} preconditioners improve convergence in the high-rank case. These insights motivate us to focus on preconditioned LoRA using a specific spectral initialization strategy for aligning with certain subspaces. For both linear and nonlinear models, we prove that alignment and generalization guarantees can be directly achieved at initialization, and the subsequent linear convergence can be also built. Our analysis leads to the \emph{LoRA-One} algorithm (using \emph{One}-step gradient and preconditioning), a theoretically grounded algorithm that achieves significant empirical improvement over vanilla LoRA and its variants on several benchmarks. Our theoretical analysis, based on decoupling the learning dynamics and characterizing how spectral initialization contributes to feature learning, may be of independent interest for understanding matrix sensing and deep learning theory. The source code can be found in the this https URL. 

**Abstract (ZH)**: 本文通过我们的理论分析探讨了如何提高低秩适应（LoRA）的性能。我们的一系列理论结果表明，对于随机初始化和线性模型而言，\textit{i)} LoRA 将会与全量微调的一步梯度的特定奇异子空间对齐；\textit{ii)} 在高秩情况下，预条件提高收敛性。这些见解促使我们关注使用特定谱初始化策略的预条件LoRA，以对齐特定子空间。对于线性和非线性模型，我们证明了对齐和泛化保证可以在初始化时直接实现，并且后续的线性收敛性也可以建立。我们的分析导致了\emph{LoRA-One}算法（使用\emph{One}-步梯度和预条件化），这是一种理论依据的算法，相对于传统的LoRA及其变体在多个基准测试中取得了显著的实证改进。我们的理论分析基于解耦学习动态并刻画谱初始化如何促进特征学习，可能对理解矩阵感知和深度学习理论具有独立的兴趣价值。源代码可在此链接中找到：this https URL。 

---
# The dark deep side of DeepSeek: Fine-tuning attacks against the safety alignment of CoT-enabled models 

**Title (ZH)**: DeepSeek的黑暗深侧：针对具解释性推理（CoT）能力的模型安全对齐的微调攻击 

**Authors**: Zhiyuan Xu, Joseph Gardiner, Sana Belguith  

**Link**: [PDF](https://arxiv.org/pdf/2502.01225)  

**Abstract**: Large language models are typically trained on vast amounts of data during the pre-training phase, which may include some potentially harmful information. Fine-tuning attacks can exploit this by prompting the model to reveal such behaviours, leading to the generation of harmful content. In this paper, we focus on investigating the performance of the Chain of Thought based reasoning model, DeepSeek, when subjected to fine-tuning attacks. Specifically, we explore how fine-tuning manipulates the model's output, exacerbating the harmfulness of its responses while examining the interaction between the Chain of Thought reasoning and adversarial inputs. Through this study, we aim to shed light on the vulnerability of Chain of Thought enabled models to fine-tuning attacks and the implications for their safety and ethical deployment. 

**Abstract (ZH)**: 大型语言模型通常在预训练阶段使用大量的数据进行训练，这些数据可能包含一些潜在有害的信息。通过细微调优攻击，可以促使模型揭示这些有害行为，从而生成有害内容。在本文中，我们专注于研究Chain of Thought推理模型DeepSeek在遭受细微调优攻击时的表现。具体来说，我们探讨细微调优如何操控模型的输出，加剧其响应的有害性，并分析Chain of Thought推理与对抗性输入之间的交互。通过这项研究，我们旨在揭示Chain of Thought使能模型对细微调优攻击的脆弱性，并探讨这对它们的安全性和伦理应用的影响。 

---
# Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents 

**Title (ZH)**: 可验证的排序与连续性在视觉-语言预训练中的作用：实现通用体态智能体的可迁移性 

**Authors**: Zhizhen Zhang, Lei Zhu, Zhen Fang, Zi Huang, Yadan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.01218)  

**Abstract**: Pre-training vision-language representations on human action videos has emerged as a promising approach to reduce reliance on large-scale expert demonstrations for training embodied agents. However, prior methods often employ time contrastive learning based on goal-reaching heuristics, progressively aligning language instructions from the initial to the final frame. This overemphasis on future frames can result in erroneous vision-language associations, as actions may terminate early or include irrelevant moments in the end. To address this issue, we propose Action Temporal Coherence Learning (AcTOL) to learn ordered and continuous vision-language representations without rigid goal-based constraint. AcTOL treats a video as a continuous trajectory where it (1) contrasts semantic differences between frames to reflect their natural ordering, and (2) imposes a local Brownian bridge constraint to ensure smooth transitions across intermediate frames. Extensive imitation learning experiments across varying numbers of demonstrations show that the pretrained features significantly enhance downstream manipulation tasks by up to 49% with high robustness to different linguistic styles of instructions, offering a viable pathway toward generalized embodied agents. The source code is included in the supplementary material for reference. 

**Abstract (ZH)**: 在人类动作视频上预训练视觉-语言表示已成为减少对大规模专家示范依赖的有效方法，从而训练具身智能体。然而，以往的方法往往基于目标到达的启发式方法使用时间对比学习，逐步将语言指令从初始帧对齐到最终帧。这种过度关注未来帧的做法可能导致视觉-语言关联错误，因为动作可能在早期终止或包含不相关的时间片段。为了解决这个问题，我们提出了一种动作时间连贯学习（Action Temporal Coherence Learning, AcTOL）方法，以学习有序且连续的视觉-语言表示而不受刚性目标导向约束。AcTOL 将视频视为一个连续的轨迹，其中（1）通过对比帧之间的语义差异来反映其自然顺序，（2）施加局部布朗桥约束以确保中间帧之间的平滑过渡。大量的模仿学习实验表明，预训练特征可以通过高达 49% 的改进显著增强各种演示数量下的下游操作任务，并且对不同指令语言风格具有很高的鲁棒性，为通用具身智能体的实现提供了可行路径。源代码作为补充材料提供，供参考。 

---
# Nearly Lossless Adaptive Bit Switching 

**Title (ZH)**: 几乎无损自适应比特切换 

**Authors**: Haiduo Huang, Zhenhua Liu, Tian Xia, Wenzhe zhao, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.01199)  

**Abstract**: Model quantization is widely applied for compressing and accelerating deep neural networks (DNNs). However, conventional Quantization-Aware Training (QAT) focuses on training DNNs with uniform bit-width. The bit-width settings vary across different hardware and transmission demands, which induces considerable training and storage costs. Hence, the scheme of one-shot joint training multiple precisions is proposed to address this issue. Previous works either store a larger FP32 model to switch between different precision models for higher accuracy or store a smaller INT8 model but compromise accuracy due to using shared quantization parameters. In this paper, we introduce the Double Rounding quantization method, which fully utilizes the quantized representation range to accomplish nearly lossless bit-switching while reducing storage by using the highest integer precision instead of full precision. Furthermore, we observe a competitive interference among different precisions during one-shot joint training, primarily due to inconsistent gradients of quantization scales during backward propagation. To tackle this problem, we propose an Adaptive Learning Rate Scaling (ALRS) technique that dynamically adapts learning rates for various precisions to optimize the training process. Additionally, we extend our Double Rounding to one-shot mixed precision training and develop a Hessian-Aware Stochastic Bit-switching (HASB) strategy. Experimental results on the ImageNet-1K classification demonstrate that our methods have enough advantages to state-of-the-art one-shot joint QAT in both multi-precision and mixed-precision. We also validate the feasibility of our method on detection and segmentation tasks, as well as on LLMs task. Our codes are available at this https URL. 

**Abstract (ZH)**: 模型量化广泛应用于深度神经网络(DNNs)的压缩和加速。然而，传统的量化感知训练(QAT)主要关注具有统一位宽的DNNs训练。不同硬件和传输需求下的位宽设置不同，这会引发显著的训练和存储成本。因此，我们提出了一个一次性联合训练多种精度的方案，以解决这一问题。以往的工作要么存储一个较大的FP32模型，以在不同精度模型之间切换以获得更高的准确度，要么存储一个较小的INT8模型，但由于使用共享量化参数，导致准确度降低。在本文中，我们提出了双舍入量化方法，该方法充分利用量化表示范围，在使用最高整数精度而非全精度的情况下，实现了几乎无损的位宽切换，并减少了存储需求。此外，我们在一次性联合训练中观察到不同精度之间存在较强的相互干扰，主要原因是反向传播过程中量化尺度的梯度不一致。为了解决这一问题，我们提出了一种自适应学习率缩放(ALRS)技术，该技术能够动态适应各种精度的学习率，优化训练过程。此外，我们还将双舍入扩展到一次性混合精度训练，并开发了一种自适应海森矩阵感知随机位宽切换(HASB)策略。在ImageNet-1K分类实验中，我们的方法在多种精度和混合精度的一次性联合QAT中都优于现有方法。我们也验证了该方法在检测和分割任务以及大语言模型(LLMs)任务上的可行性。我们的代码可以在以下链接获得：[链接]。 

---
# Dance recalibration for dance coherency with recurrent convolution block 

**Title (ZH)**: 舞蹈校准以实现舞蹈连贯性的一种循环卷积块方法 

**Authors**: Seungho Eum, Ihjoon Cho, Junghyeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.01190)  

**Abstract**: With the recent advancements in generative AI such as GAN, Diffusion, and VAE, the use of generative AI for dance generation has seen significant progress and received considerable interest. In this study, We propose R-Lodge, an enhanced version of Lodge. R-Lodge incorporates Recurrent Sequential Representation Learning named Dance Recalibration to original coarse-to-fine long dance generation model. R-Lodge utilizes Dance Recalibration method using $N$ Dance Recalibration Block to address the lack of consistency in the coarse dance representation of the Lodge model. By utilizing this method, each generated dance motion incorporates a bit of information from the previous dance motions. We evaluate R-Lodge on FineDance dataset and the results show that R-Lodge enhances the consistency of the whole generated dance motions. 

**Abstract (ZH)**: 随着生成AI如GAN、Diffusion和VAE的 Recent 进展，使用生成AI进行舞蹈生成的研究取得了显著的进步并引起了广泛关注。本研究中，我们提出了一种增强版的Lodge，即R-Lodge。R-Lodge 结合了名为舞步校准的递归序列表示学习（Dance Recalibration），并将其应用于Lodge模型的自底向上的长舞蹈生成模型。R-Lodge 通过使用 $N$ 个舞步校准模块（Dance Recalibration Block）来解决Lodge模型粗略舞步表示中的不一致性问题。通过这种方法，每个生成的舞蹈动作都包含了一些之前生成的舞蹈动作的信息。我们对R-Lodge在FineDance数据集上的性能进行了评估，结果表明R-Lodge提高了整个生成舞蹈动作的一致性。 

---
# Compressed Image Generation with Denoising Diffusion Codebook Models 

**Title (ZH)**: 基于去噪扩散码本模型的压缩图像生成 

**Authors**: Guy Ohayon, Hila Manor, Tomer Michaeli, Michael Elad  

**Link**: [PDF](https://arxiv.org/pdf/2502.01189)  

**Abstract**: We present a novel generative approach based on Denoising Diffusion Models (DDMs), which produces high-quality image samples along with their losslessly compressed bit-stream representations. This is obtained by replacing the standard Gaussian noise sampling in the reverse diffusion with a selection of noise samples from pre-defined codebooks of fixed iid Gaussian vectors. Surprisingly, we find that our method, termed Denoising Diffusion Codebook Model (DDCM), retains sample quality and diversity of standard DDMs, even for extremely small codebooks. We leverage DDCM and pick the noises from the codebooks that best match a given image, converting our generative model into a highly effective lossy image codec achieving state-of-the-art perceptual image compression results. More generally, by setting other noise selections rules, we extend our compression method to any conditional image generation task (e.g., image restoration), where the generated images are produced jointly with their condensed bit-stream representations. Our work is accompanied by a mathematical interpretation of the proposed compressed conditional generation schemes, establishing a connection with score-based approximations of posterior samplers for the tasks considered. 

**Abstract (ZH)**: 我们提出了一种基于去噪扩散模型（DDMs）的新型生成方法，该方法能够生成高质量的图像样本，同时提供其无损压缩的位流表示。这通过将反向扩散中的标准高斯噪声采样替换为从预定义的固定iid高斯向量码本中选择噪声样本实现。令人惊讶的是，我们发现这种方法被称为去噪扩散码本模型（DDCM），即使在极小的码本中也能保持与标准DDMs相同的样本质量和多样性。我们利用DDCM，从与给定图像最佳匹配的码本中选择噪声，将我们的生成模型转化为一个高效的有损图像编码器，实现了达到当前最佳感知图像压缩效果。更广泛地说，通过设置其他噪声选择规则，我们将压缩方法扩展到任何条件图像生成任务（例如，图像恢复），其中生成的图像与它们浓缩的位流表示同时生成。我们的工作还提供了一个提出的压缩条件生成方案的数学解释，并建立了与已考虑任务中的基于评分的后验采样近似之间的联系。 

---
# Deep Active Speech Cancellation with Multi-Band Mamba Network 

**Title (ZH)**: 多带Mamba网络的深度主动语音抑制 

**Authors**: Yehuda Mishaly, Lior Wolf, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2502.01185)  

**Abstract**: We present a novel deep learning network for Active Speech Cancellation (ASC), advancing beyond Active Noise Cancellation (ANC) methods by effectively canceling both noise and speech signals. The proposed Multi-Band Mamba architecture segments input audio into distinct frequency bands, enabling precise anti-signal generation and improved phase alignment across frequencies. Additionally, we introduce an optimization-driven loss function that provides near-optimal supervisory signals for anti-signal generation. Experimental results demonstrate substantial performance gains, achieving up to 7.2dB improvement in ANC scenarios and 6.2dB in ASC, significantly outperforming existing methods. Audio samples are available at this https URL 

**Abstract (ZH)**: 我们提出了一种新颖的深度学习网络，用于活动语音取消（Active Speech Cancellation, ASC），其超越了活动噪声取消（Active Noise Cancellation, ANC）方法，能够有效取消噪声和语音信号。提出的多频带Mamba架构将输入音频分割为不同的频率带，使得反信号生成更加精确，并且在不同频率上提高了相位对准。此外，我们引入了一种基于优化的损失函数，为反信号生成提供了接近最优的监督信号。实验结果表明，该方法显著提高了性能，在噪声取消场景中达到了7.2dB的提升，在语音取消中达到了6.2dB，大幅超过了现有方法。音频样本可在以下链接获取：[此链接](此链接应该替换为实际的URL)。 

---
# FragmentNet: Adaptive Graph Fragmentation for Graph-to-Sequence Molecular Representation Learning 

**Title (ZH)**: FragmentNet：自适应图片段化在分子序列化图表示学习中的应用 

**Authors**: Ankur Samanta, Rohan Gupta, Aditi Misra, Christian McIntosh Clarke, Jayakumar Rajadas  

**Link**: [PDF](https://arxiv.org/pdf/2502.01184)  

**Abstract**: Molecular property prediction uses molecular structure to infer chemical properties. Chemically interpretable representations that capture meaningful intramolecular interactions enhance the usability and effectiveness of these predictions. However, existing methods often rely on atom-based or rule-based fragment tokenization, which can be chemically suboptimal and lack scalability. We introduce FragmentNet, a graph-to-sequence foundation model with an adaptive, learned tokenizer that decomposes molecular graphs into chemically valid fragments while preserving structural connectivity. FragmentNet integrates VQVAE-GCN for hierarchical fragment embeddings, spatial positional encodings for graph serialization, global molecular descriptors, and a transformer. Pre-trained with Masked Fragment Modeling and fine-tuned on MoleculeNet tasks, FragmentNet outperforms models with similarly scaled architectures and datasets while rivaling larger state-of-the-art models requiring significantly more resources. This novel framework enables adaptive decomposition, serialization, and reconstruction of molecular graphs, facilitating fragment-based editing and visualization of property trends in learned embeddings - a powerful tool for molecular design and optimization. 

**Abstract (ZH)**: 分子性质预测利用分子结构来推断化学性质。能够捕捉有意义的分子内部相互作用的化学可解释表示增强了一次性和有效性。然而，现有方法往往依赖于基于原子或基于规则的片段标记化，这可能在化学上不是最优的，并且缺乏扩展性。我们提出了FragmentNet，这是一种图到序列的基础模型，具有自适应的学习标记化器，能够将分子图分解为化学上有效的片段，同时保持结构连通性。FragmentNet 结合了 VQVAE-GCN 进行分层次的片段嵌入，使用空间位置编码进行图序列化，加入了全局分子描述符，并采用变换器。通过对 Masked Fragment Modeling 预训练并在 MoleculeNet 任务上微调，FragmentNet 在具有类似规模架构和数据集的模型中表现出更优的效果，同时与需要显著更多资源的大型最新模型相媲美。这一新颖框架使分子图的自适应分解、序列化和重构成为可能，促进了基于片段的编辑和学习嵌入中性质趋势的可视化——这为分子设计与优化提供了一个强大的工具。 

---
# A Single Model Ensemble Framework for Neural Machine Translation using Pivot Translation 

**Title (ZH)**: 使用pivot翻译的单模型集成框架在神经机器翻译中的应用 

**Authors**: Seokjin Oh, Keonwoong Noh, Woohwan Jung  

**Link**: [PDF](https://arxiv.org/pdf/2502.01182)  

**Abstract**: Despite the significant advances in neural machine translation, performance remains subpar for low-resource language pairs. Ensembling multiple systems is a widely adopted technique to enhance performance, often accomplished by combining probability distributions. However, the previous approaches face the challenge of high computational costs for training multiple models. Furthermore, for black-box models, averaging token-level probabilities at each decoding step is not feasible. To address the problems of multi-model ensemble methods, we present a pivot-based single model ensemble. The proposed strategy consists of two steps: pivot-based candidate generation and post-hoc aggregation. In the first step, we generate candidates through pivot translation. This can be achieved with only a single model and facilitates knowledge transfer from high-resource pivot languages, resulting in candidates that are not only diverse but also more accurate. Next, in the aggregation step, we select k high-quality candidates from the generated candidates and merge them to generate a final translation that outperforms the existing candidates. Our experimental results show that our method produces translations of superior quality by leveraging candidates from pivot translation to capture the subtle nuances of the source sentence. 

**Abstract (ZH)**: 尽管在神经机器翻译方面取得了显著进展，但对于低资源语言对，性能仍然不尽如人意。多系统集成是一种广泛采用的技术，旨在提高性能，通常通过合并概率分布来实现。然而，现有的方法面临着训练多个模型的高计算成本挑战。此外，对于黑盒模型而言，在解码的每个步骤中平均标记级别概率是不可行的。为了解决多模型集成方法的问题，我们提出了一种基于转写的单模型集成策略。该策略包含了两个步骤：基于转写的候选生成和事后聚合。首先，在候选生成步骤中，我们通过转写生成候选译文。这只需使用一个模型，并且能够从高资源转写语言中转移知识，从而产生既多样又更准确的候选译文。其次，在聚合步骤中，我们从生成的候选译文中选择k个高质量的候选译文并合并它们，以生成优于现有候选译文的最终译文。我们的实验结果表明，通过利用转写候选译文来捕捉来源句子中的微妙之处，我们的方法能够生成质量更高的译文。 

---
# Joint Localization and Activation Editing for Low-Resource Fine-Tuning 

**Title (ZH)**: 低资源场景下联合定位和激活编辑的 fine-tuning 方法 

**Authors**: Wen Lai, Alexander Fraser, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2502.01179)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, are commonly used to adapt LLMs. However, the effectiveness of standard PEFT methods is limited in low-resource scenarios with only a few hundred examples. Recent advances in interpretability research have inspired the emergence of activation editing techniques, which modify the activations of specific model components. These methods, due to their extremely small parameter counts, show promise for small datasets. However, their performance is highly dependent on identifying the correct modules to edit and often lacks stability across different datasets. In this paper, we propose Joint Localization and Activation Editing (JoLA), a method that jointly learns (1) which heads in the Transformer to edit (2) whether the intervention should be additive, multiplicative, or both and (3) the intervention parameters themselves - the vectors applied as additive offsets or multiplicative scalings to the head output. Through evaluations on three benchmarks spanning commonsense reasoning, natural language understanding, and natural language generation, we demonstrate that JoLA consistently outperforms existing methods. 

**Abstract (ZH)**: 参数高效微调（PEFT）方法，如LoRA，在适配大规模语言模型（LLMs）方面得到了广泛应用。然而，标准PEFT方法在仅有少量几百个样本的低资源场景中效果有限。近期解释性研究的进展激发了激活编辑技术的出现，这些技术通过修改特定模型组件的激活值来实现微调。由于其参数量极小，这些方法在小数据集场景下显示出潜力。然而，它们的性能高度依赖于能否正确识别需要编辑的模块，且在不同数据集上的稳定性较差。本文中，我们提出了一种名为Joint Localization and Activation Editing（JoLA）的方法，可以联合学习以下内容：(1) 哪些在Transformer中的头需要编辑；(2) 干预应是加性、乘性还是两者兼有；(3) 干预参数本身，即应用于头输出的加性偏移向量或乘性缩放因子。通过在常识推理、自然语言理解以及自然语言生成三个基准数据集上的评估，我们展示了JoLA方法在多个任务上都取得了优于现有方法的效果。 

---
# AtmosSci-Bench: Evaluating the Recent Advance of Large Language Model for Atmospheric Science 

**Title (ZH)**: AtmosSci-Bench: 评估大型语言模型在大气科学领域的 recent advance 

**Authors**: Chenyue Li, Wen Deng, Mengqian Lu, Binhang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.01159)  

**Abstract**: The rapid advancements in large language models (LLMs), particularly in their reasoning capabilities, hold transformative potential for addressing complex challenges in atmospheric science. However, leveraging LLMs effectively in this domain requires a robust and comprehensive evaluation benchmark. To address this need, we present AtmosSci-Bench, a novel benchmark designed to systematically assess LLM performance across five core categories of atmospheric science problems: hydrology, atmospheric dynamics, atmospheric physics, geophysics, and physical oceanography. We employ a template-based question generation framework, enabling scalable and diverse multiple-choice questions curated from graduate-level atmospheric science problems. We conduct a comprehensive evaluation of representative LLMs, categorized into four groups: instruction-tuned models, advanced reasoning models, math-augmented models, and domain-specific climate models. Our analysis provides some interesting insights into the reasoning and problem-solving capabilities of LLMs in atmospheric science. We believe AtmosSci-Bench can serve as a critical step toward advancing LLM applications in climate service by offering a standard and rigorous evaluation framework. Our source codes are currently available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的迅猛发展，特别是在推理能力方面的提升，为解决大气科学领域的复杂挑战提供了变革性的潜力。然而，有效地利用LLMs需要一个稳固且全面的评估基准。为满足这一需求，我们提出了AtmosSci-Bench，这是一个新的基准测试，旨在系统评估LLMs在大气科学五大核心问题领域的性能：水文学、大气动力学、大气物理学、地球物理学和物理海洋学。我们采用基于模板的问题生成框架，能够生成多样化且适用的多项选择题，这些问题是从高级大气科学问题中精选出来的。我们对代表性LLMs进行了全面评估，将它们分为四个组别：指令微调模型、高级推理模型、数学增强模型和领域特定的气候模型。我们的分析为LLMs在大气科学中的推理和问题解决能力提供了有价值的洞察。我们相信，AtmosSci-Bench 可以为推动LLMs在气候服务中的应用提供一个标准且严谨的评估框架。目前，我们的源代码可在此处获取：[此httpsURL]。 

---
# Jailbreaking with Universal Multi-Prompts 

**Title (ZH)**: 使用通用多提示词进行越狱攻击 

**Authors**: Yu-Ling Hsu, Hsuan Su, Shang-Tse Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.01154)  

**Abstract**: Large language models (LLMs) have seen rapid development in recent years, revolutionizing various applications and significantly enhancing convenience and productivity. However, alongside their impressive capabilities, ethical concerns and new types of attacks, such as jailbreaking, have emerged. While most prompting techniques focus on optimizing adversarial inputs for individual cases, resulting in higher computational costs when dealing with large datasets. Less research has addressed the more general setting of training a universal attacker that can transfer to unseen tasks. In this paper, we introduce JUMP, a prompt-based method designed to jailbreak LLMs using universal multi-prompts. We also adapt our approach for defense, which we term DUMP. Experimental results demonstrate that our method for optimizing universal multi-prompts outperforms existing techniques. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）取得了快速的发展，革新了各种应用并显著提高了便利性和生产效率。然而，随着其出色的性能，出现了伦理问题和新的攻击类型，如越狱攻击。尽管大多数提示技术侧重于针对个体案例优化对抗输入，这在处理大规模数据集时会导致更高的计算成本，但较少的研究关注训练能够应用于未见任务的通用攻击者这一更为普遍的情境。本文介绍了一种名为JUMP的提示基方法，该方法使用通用多提示来越狱大型语言模型。我们还为防御开发了相应的技术，称为DUMP。实验结果表明，我们优化通用多提示的方法在性能上优于现有技术。 

---
# Quantum Machine Learning: A Hands-on Tutorial for Machine Learning Practitioners and Researchers 

**Title (ZH)**: 量子机器学习：面向机器学习从业者和研究人员的实战教程 

**Authors**: Yuxuan Du, Xinbiao Wang, Naixu Guo, Zhan Yu, Yang Qian, Kaining Zhang, Min-Hsiu Hsieh, Patrick Rebentrost, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.01146)  

**Abstract**: This tutorial intends to introduce readers with a background in AI to quantum machine learning (QML) -- a rapidly evolving field that seeks to leverage the power of quantum computers to reshape the landscape of machine learning. For self-consistency, this tutorial covers foundational principles, representative QML algorithms, their potential applications, and critical aspects such as trainability, generalization, and computational complexity. In addition, practical code demonstrations are provided in this https URL to illustrate real-world implementations and facilitate hands-on learning. Together, these elements offer readers a comprehensive overview of the latest advancements in QML. By bridging the gap between classical machine learning and quantum computing, this tutorial serves as a valuable resource for those looking to engage with QML and explore the forefront of AI in the quantum era. 

**Abstract (ZH)**: 本教程旨在为具备人工智能背景的读者介绍量子机器学习（QML）——一个迅速发展的领域，该领域致力于利用量子计算机的强大功能重塑机器学习的格局。为了内部一致性，本教程涵盖了基础原理、代表性QML算法、潜在应用以及可训练性、泛化能力和计算复杂性等关键方面。此外，还提供了此处提供的实际代码示例（https://example.com）以说明实际的实现，并促进动手学习。这些元素共同为读者提供了QML最新进展的全面概述。通过连接经典机器学习与量子计算之间的桥梁，本教程成为那些希望涉足QML并探索量子时代人工智能前沿的人的重要资源。 

---
# ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills 

**Title (ZH)**: ASAP: 将模拟与真实世界物理对齐以学习灵活的人形全身技能 

**Authors**: Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbab, Chaoyi Pan, Zeji Yi, Guannan Qu, Kris Kitani, Jessica Hodgins, Linxi "Jim" Fan, Yuke Zhu, Changliu Liu, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.01143)  

**Abstract**: Humanoid robots hold the potential for unparalleled versatility in performing human-like, whole-body skills. However, achieving agile and coordinated whole-body motions remains a significant challenge due to the dynamics mismatch between simulation and the real world. Existing approaches, such as system identification (SysID) and domain randomization (DR) methods, often rely on labor-intensive parameter tuning or result in overly conservative policies that sacrifice agility. In this paper, we present ASAP (Aligning Simulation and Real-World Physics), a two-stage framework designed to tackle the dynamics mismatch and enable agile humanoid whole-body skills. In the first stage, we pre-train motion tracking policies in simulation using retargeted human motion data. In the second stage, we deploy the policies in the real world and collect real-world data to train a delta (residual) action model that compensates for the dynamics mismatch. Then, ASAP fine-tunes pre-trained policies with the delta action model integrated into the simulator to align effectively with real-world dynamics. We evaluate ASAP across three transfer scenarios: IsaacGym to IsaacSim, IsaacGym to Genesis, and IsaacGym to the real-world Unitree G1 humanoid robot. Our approach significantly improves agility and whole-body coordination across various dynamic motions, reducing tracking error compared to SysID, DR, and delta dynamics learning baselines. ASAP enables highly agile motions that were previously difficult to achieve, demonstrating the potential of delta action learning in bridging simulation and real-world dynamics. These results suggest a promising sim-to-real direction for developing more expressive and agile humanoids. 

**Abstract (ZH)**: humanoid 机器人在执行人类般的全身技能方面具有无与伦比的多功能性。然而，由于仿真与现实世界之间的动力学不匹配，实现灵活而协调的全身运动仍是一项重大挑战。现有方法，如系统辨识（SysID）和域随机化（DR）方法，通常依赖于耗时的参数调优，或者会产生过于保守的策略，牺牲灵活性。本文介绍了一种名为 ASAP（仿真与现实物理对齐）的两阶段框架，旨在解决动力学不匹配问题，并使 humanoid 机器人能够执行灵活的全身技能。在第一阶段，我们使用重新定向的人类运动数据在仿真中预训练动作追踪策略。在第二阶段，我们将这些策略部署到现实世界中，并收集实际数据以训练一个残差（δ）动作模型，该模型可以补偿动力学不匹配。随后，ASAP 通过将 δ 动作模型整合到仿真器中来微调预训练策略，从而有效对齐与现实世界的动力学。我们通过三个传输场景评估了 ASAP：从 IsaacGym 到 IsaacSim，从 IsaacGym 到 Genesis，以及从 IsaacGym 到现实世界中的 Unitree G1 humanoid 机器人。我们的方法在各种动态运动中显著提高了灵活性和全身协调性，相比 SysID、DR 和 δ 动力学学习基线，减少了追踪误差。ASAP 使得以前难以实现的高度灵活运动成为可能，展示了 δ 动作学习在连接仿真和现实动力学方面的能力。这些结果表明，ASAP 在开发更具表现力和灵活性的 humanoid 机器人方面具有令人鼓舞的从仿真到现实的方向。 

---
# Beyond Yes or No: Predictive Compliance Monitoring Approaches for Quantifying the Magnitude of Compliance Violations 

**Title (ZH)**: 超越是与非：用于量化合规性违规程度的预测性合规监控方法 

**Authors**: Qian Chen, Stefanie Rinderle-Ma, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.01141)  

**Abstract**: Most existing process compliance monitoring approaches detect compliance violations in an ex post manner. Only predicate prediction focuses on predicting them. However, predicate prediction provides a binary yes/no notion of compliance, lacking the ability to measure to which extent an ongoing process instance deviates from the desired state as specified in constraints. Here, being able to quantify the magnitude of violation would provide organizations with deeper insights into their operational performance, enabling informed decision making to reduce or mitigate the risk of non-compliance. Thus, we propose two predictive compliance monitoring approaches to close this research gap. The first approach reformulates the binary classification problem as a hybrid task that considers both classification and regression, while the second employs a multi-task learning method to explicitly predict the compliance status and the magnitude of violation for deviant cases simultaneously. In this work, we focus on temporal constraints as they are significant in almost any application domain, e.g., health care. The evaluation on synthetic and real-world event logs demonstrates that our approaches are capable of quantifying the magnitude of violations while maintaining comparable performance for compliance predictions achieved by state-of-the-art approaches. 

**Abstract (ZH)**: 现有的大多数过程合规监控方法以事后方式检测合规性违规。仅有基于谓词的预测专注于预测这些违规行为。然而，基于谓词的预测仅能提供二元的“是/否”合规性概念，缺乏衡量正在进行的过程实例偏离期望状态的程度的能力。这意味着能够量化违规程度将帮助组织更深入地了解其运营绩效，并能基于相关数据进行知情决策，以降低或缓解不符合合规性的风险。因此，我们提出了两种预测合规性监控方法来弥补这一研究空白。第一种方法将二元分类问题重新表述为结合分类和回归的混合任务，而第二种方法则利用多任务学习方法同时预测偏离合规状况的案例的合规状态和违规程度。在该工作中，我们重点关注时间约束，因为它们在大多数应用领域都至关重要，例如医疗保健。对于合成和实际事件日志的评估表明，我们的方法不仅能够量化违规程度，还能在合规性预测方面达到与现有先进方法相当的性能。 

---
# Self-Organizing Interaction Spaces: A Framework for Engineering Pervasive Applications in Mobile and Distributed Environments 

**Title (ZH)**: 自我组织交互空间：在移动和分布式环境中构建沉浸式应用的框架 

**Authors**: Shubham Malhotra  

**Link**: [PDF](https://arxiv.org/pdf/2502.01137)  

**Abstract**: The rapid adoption of pervasive and mobile computing has led to an unprecedented rate of data production and consumption by mobile applications at the network edge. These applications often require interactions such as data exchange, behavior coordination, and collaboration, which are typically mediated by cloud servers. While cloud computing has been effective for distributed systems, challenges like latency, cost, and intermittent connectivity persist. With the advent of 5G technology, features like location-awareness and device-to-device (D2D) communication enable a more distributed and adaptive architecture. This paper introduces Self-Organizing Interaction Spaces (SOIS), a novel framework for engineering pervasive applications. SOIS leverages the dynamic and heterogeneous nature of mobile nodes, allowing them to form adaptive organizational structures based on their individual and social contexts. The framework provides two key abstractions for modeling and programming pervasive applications using an organizational mindset and mechanisms for adapting dynamic organizational structures. Case examples and performance evaluations of a simulated mobile crowd-sensing application demonstrate the feasibility and benefits of SOIS. Results highlight its potential to enhance efficiency and reduce reliance on traditional cloud models, paving the way for innovative solutions in mobile and distributed environments. 

**Abstract (ZH)**: 快速普及的泛在计算和移动计算正以前所未有的速度产生和消费网络边缘的移动应用数据。这些应用通常需要数据交换、行为协调和协作等交互，这些交互通常由云服务器进行中介。尽管云计算对于分布式系统是有效的，但仍存在延迟、成本和间歇性连接等挑战。随着5G技术的出现，具备位置感知能力和设备到设备（D2D）通信的特点，使得更分布式和自适应的架构成为可能。本文介绍了自我组织交互空间（SOIS），这是一种新型框架用于构建泛在应用。SOIS 利用了移动节点动态且异构的特性，使它们能够根据不同个体和社会背景形成自适应的组织结构。该框架提供了两种关键抽象，用于使用组织化的思维模式建模和编程泛在应用，并提供了适应动态组织结构的机制。通过模拟移动群感知应用的案例研究和性能评估，展示了SOIS 的可行性和优势。结果突显了其潜在能力，提高了效率并减少了对传统云模型的依赖，为移动和分布式环境中的创新解决方案铺平了道路。 

---
# Deep Reinforcement Learning for Dynamic Resource Allocation in Wireless Networks 

**Title (ZH)**: 无线网络中动态资源分配的深度强化学习方法 

**Authors**: Shubham Malhotra  

**Link**: [PDF](https://arxiv.org/pdf/2502.01129)  

**Abstract**: This report investigates the application of deep reinforcement learning (DRL) algorithms for dynamic resource allocation in wireless communication systems. An environment that includes a base station, multiple antennas, and user equipment is created. Using the RLlib library, various DRL algorithms such as Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) are then applied. These algorithms are compared based on their ability to optimize resource allocation, focusing on the impact of different learning rates and scheduling policies. The findings demonstrate that the choice of algorithm and learning rate significantly influences system performance, with DRL providing more efficient resource allocation compared to traditional methods. 

**Abstract (ZH)**: 本报告探讨了深度强化学习（DRL）算法在无线通信系统中动态资源分配中的应用。构建了一个包含基站、多个天线和用户设备的环境。然后，使用RLlib库应用了多种DRL算法，如深度Q网络（DQN）和近端策略优化（PPO）。根据这些算法优化资源分配的能力进行比较，重点关注不同学习率和调度策略的影响。研究结果表明，算法的选择和学习率对系统性能有显著影响，DRL相比传统方法提供了更高效的资源分配。 

---
# The Battling Influencers Game: Nash Equilibria Structure of a Potential Game and Implications to Value Alignment 

**Title (ZH)**: 《对决影响者游戏：潜在博弈的纳什均衡结构及其对价值对齐的影响》 

**Authors**: Young Wu, Yancheng Zhu, Jin-Yi Cai, Xiaojin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01127)  

**Abstract**: When multiple influencers attempt to compete for a receiver's attention, their influencing strategies must account for the presence of one another. We introduce the Battling Influencers Game (BIG), a multi-player simultaneous-move general-sum game, to provide a game-theoretic characterization of this social phenomenon. We prove that BIG is a potential game, that it has either one or an infinite number of pure Nash equilibria (NEs), and these pure NEs can be found by convex optimization. Interestingly, we also prove that at any pure NE, all (except at most one) influencers must exaggerate their actions to the maximum extent. In other words, it is rational for the influencers to be non-truthful and extreme because they anticipate other influencers to cancel out part of their influence. We discuss the implications of BIG to value alignment. 

**Abstract (ZH)**: 当多位影响者试图争夺接收者的注意力时，他们的影响策略必须考虑到彼此的存在。我们引入了竞争影响者博弈（Battling Influencers Game, BIG），这是一种多玩家同时决策的一般胜负游戏，以提供对该社会现象的博弈论描述。我们证明BIG是一种潜力博弈，它要么有一个纯纳什均衡（NE），要么有无限多个纯纳什均衡，这些纯NE可以通过凸优化找到。有趣的是，我们还证明，在任何纯NE中，所有影响者（最多只有一个例外）必须将其行为夸大至最大程度。换句话说，影响者进行不诚实和极端的行为是有道理的，因为他们预期其他影响者会抵消他们部分的影响。我们讨论了BIG对价值对齐的含义。 

---
# Large Language Model-Enhanced Multi-Armed Bandits 

**Title (ZH)**: 大型语言模型增强的多臂 bandit 算法 

**Authors**: Jiahang Sun, Zhiyong Wang, Runhan Yang, Chenjun Xiao, John C.S. Lui, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2502.01118)  

**Abstract**: Large language models (LLMs) have been adopted to solve sequential decision-making tasks such as multi-armed bandits (MAB), in which an LLM is directly instructed to select the arms to pull in every iteration. However, this paradigm of direct arm selection using LLMs has been shown to be suboptimal in many MAB tasks. Therefore, we propose an alternative approach which combines the strengths of classical MAB and LLMs. Specifically, we adopt a classical MAB algorithm as the high-level framework and leverage the strong in-context learning capability of LLMs to perform the sub-task of reward prediction. Firstly, we incorporate the LLM-based reward predictor into the classical Thompson sampling (TS) algorithm and adopt a decaying schedule for the LLM temperature to ensure a transition from exploration to exploitation. Next, we incorporate the LLM-based reward predictor (with a temperature of 0) into a regression oracle-based MAB algorithm equipped with an explicit exploration mechanism. We also extend our TS-based algorithm to dueling bandits where only the preference feedback between pairs of arms is available, which requires non-trivial algorithmic modifications. We conduct empirical evaluations using both synthetic MAB tasks and experiments designed using real-world text datasets, in which the results show that our algorithms consistently outperform previous baseline methods based on direct arm selection. Interestingly, we also demonstrate that in challenging tasks where the arms lack semantic meanings that can be exploited by the LLM, our approach achieves considerably better performance than LLM-based direct arm selection. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经被应用于解决多臂老虎机（MAB）等序列决策任务，在这些任务中，LLMs 直接指令选择每一轮应拉动的臂。然而，直接使用LLMs选择臂的方法在许多MAB任务中已被证明是次优的。因此，我们提出了一种新的方法，这种方法结合了经典MAB与LLMs的优点。具体而言，我们将一个经典MAB算法作为高层次框架，并利用LLMs的强大上下文学习能力来完成奖励预测子任务。首先，我们将基于LLM的奖励预测器整合到经典的泰勒斯采样（TS）算法中，并采用温度衰减计划以确保从探索向利采的过渡。接着，我们将基于回归或acles的MAB算法中的LLM基于的奖励预测器（温度设为0）整合进来，并配备了显式的探索机制。我们还将基于TS的算法推广到对战式多臂老虎机（dueling bandits）中，其中只有臂对之间的偏好反馈可用，这需要算法上的复杂调整。我们通过合成的MAB任务和基于实际文本数据集设计的实验进行了实证评估，结果显示我们的算法始终优于基于直接选择臂的方法。有趣的是，我们还展示了在臂缺乏对LLM有益的语义特征的具有挑战性任务中，我们的方法比基于LLM的直接臂选择方法表现出了显著更好的性能。 

---
# Learning to Learn Weight Generation via Trajectory Diffusion 

**Title (ZH)**: 通过轨迹扩散学习生成权重的学习方法 

**Authors**: Yunchuan Guan, Yu Liu, Ke Zhou, Zhiqi Shen, Serge Belongie, Jenq-Neng Hwang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.01117)  

**Abstract**: Diffusion-based algorithms have emerged as promising techniques for weight generation, particularly in scenarios like multi-task learning that require frequent weight updates. However, existing solutions suffer from limited cross-task transferability. In addition, they only utilize optimal weights as training samples, ignoring the value of other weights in the optimization process. To address these issues, we propose Lt-Di, which integrates the diffusion algorithm with meta-learning to generate weights for unseen tasks. Furthermore, we extend the vanilla diffusion algorithm into a trajectory diffusion algorithm to utilize other weights along the optimization trajectory. Trajectory diffusion decomposes the entire diffusion chain into multiple shorter ones, improving training and inference efficiency. We analyze the convergence properties of the weight generation paradigm and improve convergence efficiency without additional time overhead. Our experiments demonstrate Lt-Di's higher accuracy while reducing computational overhead across various tasks, including zero-shot and few-shot learning, multi-domain generalization, and large-scale language model this http URL code is released at this https URL. 

**Abstract (ZH)**: 基于扩散的算法已经成为了生成权重的有前途的技术，特别是在需要频繁更新权重的多任务学习等场景中。然而，现有的解决方案在跨任务迁移方面存在局限性。此外，这些方法仅利用最优权重作为训练样本，忽视了其他权重在优化过程中的价值。为了解决这些问题，我们提出了一种名为Lt-Di的方法，将扩散算法与元学习相结合，以生成未见过任务的权重。此外，我们将传统的扩散算法扩展为路径扩散算法，以利用优化路径中其他权重的价值。路径扩散将整个扩散链分解为多个较短的链，提高训练和推理效率。我们分析了权重生成范式的收敛性质，在不增加额外时间开销的情况下提高了收敛效率。实验结果表明，与现有的方法相比，Lt-Di在不同任务上的准确率更高，并且减少了计算开销，涵盖零样本学习、少样本学习、多域泛化以及大规模语言模型等领域。相关代码已发布，详见：

[此链接](this http URL)

[此处链接](this https URL) 

---
# GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation 

**Title (ZH)**: GFM-RAG：图基础模型赋能检索增强生成 

**Authors**: Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.01113)  

**Abstract**: Retrieval-augmented generation (RAG) has proven effective in integrating knowledge into large language models (LLMs). However, conventional RAGs struggle to capture complex relationships between pieces of knowledge, limiting their performance in intricate reasoning that requires integrating knowledge from multiple sources. Recently, graph-enhanced retrieval augmented generation (GraphRAG) builds graph structure to explicitly model these relationships, enabling more effective and efficient retrievers. Nevertheless, its performance is still hindered by the noise and incompleteness within the graph structure. To address this, we introduce GFM-RAG, a novel graph foundation model (GFM) for retrieval augmented generation. GFM-RAG is powered by an innovative graph neural network that reasons over graph structure to capture complex query-knowledge relationships. The GFM with 8M parameters undergoes a two-stage training process on large-scale datasets, comprising 60 knowledge graphs with over 14M triples and 700k documents. This results in impressive performance and generalizability for GFM-RAG, making it the first graph foundation model applicable to unseen datasets for retrieval without any fine-tuning required. Extensive experiments on three multi-hop QA datasets and seven domain-specific RAG datasets demonstrate that GFM-RAG achieves state-of-the-art performance while maintaining efficiency and alignment with neural scaling laws, highlighting its potential for further improvement. 

**Abstract (ZH)**: 检索增强生成（RAG）已被证明能够有效地将知识整合到大型语言模型（LLMs）中。然而，传统的RAG方法在捕捉多个知识片段之间的复杂关系方面存在困难，这限制了它们在需要从多个来源综合知识的复杂推理中的性能。最近，图增强的检索增强生成（GraphRAG）通过构建图结构来明确建模这些关系，从而使检索更有效率且更加有效。尽管如此，图结构中的噪声和不完整性仍然限制了其性能。为了解决这一问题，我们提出了GFM-RAG（Graph Foundation Model-Enhanced Retrieval-Augmented Generation），这是一种用于检索增强生成的新型图基础模型（GFM）。GFM-RAG通过一种创新的图神经网络工作，该网络能够在图结构上进行推理以捕捉复杂的查询-知识关系。一个包含800万个参数的GFM在一个包含60个知识图和超过1400万个三元组，以及70万份文档的大规模数据集上进行了两阶段训练。这使得GFM-RAG在性能和泛化能力方面表现出色，并成为首个无需任何微调即可应用于未见过的数据集的图基础模型。在三个多跳QA数据集和七个领域特定的RAG数据集上的广泛实验表明，GFM-RAG不仅实现了最先进的性能，还保持了高效性，并与神经网络的扩展定律保持一致，这展示了其进一步改进的潜力。 

---
# A generative foundation model for an all-in-one seismic processing framework 

**Title (ZH)**: 一个用于集成地震处理框架的生成性基础模型 

**Authors**: Shijun Cheng, Randy Harsuko, Tariq Alkhalifah  

**Link**: [PDF](https://arxiv.org/pdf/2502.01111)  

**Abstract**: Seismic data often face challenges in their utilization due to noise contamination, incomplete acquisition, and limited low-frequency information, which hinder accurate subsurface imaging and interpretation. Traditional processing methods rely heavily on task-specific designs to address these challenges and fail to account for the variability of data. To address these limitations, we present a generative seismic foundation model (GSFM), a unified framework based on generative diffusion models (GDMs), designed to tackle multi-task seismic processing challenges, including denoising, backscattered noise attenuation, interpolation, and low-frequency extrapolation. GSFM leverages a pre-training stage on synthetic data to capture the features of clean, complete, and broadband seismic data distributions and applies an iterative fine-tuning strategy to adapt the model to field data. By adopting a target-oriented diffusion process prediction, GSFM improves computational efficiency without compromising accuracy. Synthetic data tests demonstrate GSFM surpasses benchmarks with equivalent architectures in all tasks and achieves performance comparable to traditional pre-training strategies, even after their fine-tuning. Also, field data tests suggest that our iterative fine-tuning approach addresses the generalization limitations of conventional pre-training and fine-tuning paradigms, delivering significantly enhanced performance across diverse tasks. Furthermore, GSFM's inherent probabilistic nature enables effective uncertainty quantification, offering valuable insights into the reliability of processing results. 

**Abstract (ZH)**: 地震数据在利用过程中常常面临噪声污染、数据不完整以及低频信息有限等挑战，这些问题阻碍了地下成像和解释的准确性。传统处理方法依赖于特定任务的设计来应对这些挑战，但未能考虑到数据的变异性。为解决这些局限性，我们提出了一种生成地震基础模型（GSFM），这是一种基于生成扩散模型（GDMs）的统一框架，旨在解决包括去噪、散射噪声衰减、插值和低频外推在内的多任务地震处理挑战。GSFM利用合成数据进行预训练，以捕获干净、完整且宽带地震数据分布的特点，并采用迭代微调策略来使模型适应现场数据。通过采用目标导向的扩散过程预测，GSFM在提高计算效率的同时不牺牲精度。合成数据测试表明，GSFM在所有任务中性能优于具有同等架构的基准模型，并且其性能与传统预训练策略相当，甚至在微调后也是如此。此外，现场数据测试表明，我们提出的迭代微调方法解决了传统预训练和微调范式的一般化局限性，实现了在各种任务中显著增强的性能。此外，GSFM固有的概率性质使其能够有效地进行不确定性量化，提供了关于处理结果可靠性的宝贵见解。 

---
# Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications Across Lab and Field Settings 

**Title (ZH)**: 脉搏-光电容积脉冲：一种适用于实验室和现场应用场景的可穿戴设备光电容积描迹图基础模型（开放源代码现场训练版本） 

**Authors**: Mithun Saha, Maxwell A. Xu, Wanting Mao, Sameer Neupane, James M. Rehg, Santosh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.01108)  

**Abstract**: Photoplethysmography (PPG)-based foundation models are gaining traction due to the widespread use of PPG in biosignal monitoring and their potential to generalize across diverse health applications. In this paper, we introduce Pulse-PPG, the first open-source PPG foundation model trained exclusively on raw PPG data collected over a 100-day field study with 120 participants. Existing PPG foundation models are either open-source but trained on clinical data or closed-source, limiting their applicability in real-world settings. We evaluate Pulse-PPG across multiple datasets and downstream tasks, comparing its performance against a state-of-the-art foundation model trained on clinical data. Our results demonstrate that Pulse-PPG, trained on uncurated field data, exhibits superior generalization across clinical and mobile health applications in both lab and field settings. This suggests that exposure to real-world variability enables the model to learn fine-grained representations, making it more adaptable across tasks. Furthermore, pre-training on field data surprisingly outperforms its pre-training on clinical data in many tasks, reinforcing the importance of training on real-world, diverse datasets. To encourage further advancements in robust foundation models leveraging field data, we plan to release Pulse-PPG, providing researchers with a powerful resource for developing more generalizable PPG-based models. 

**Abstract (ZH)**: 基于光电容积描记法（PPG）的基石模型由于PPG在生物信号监测中的广泛应用及其在多种健康应用中泛化潜力的增强而得到了广泛关注。本文介绍了Pulse-PPG，这是第一个专门在为期100天的实地研究中收集的120名参与者原始PPG数据上训练的开源PPG基石模型。现有的PPG基石模型要么是开源的但训练于临床数据，要么是闭源的，这限制了其在实际应用场景中的应用。我们评估了Pulse-PPG在多个数据集和下游任务上的表现，将其性能与基于临床数据训练的最新基石模型进行了比较。我们的结果表明，Pulse-PPG在未经筛选的实地数据上训练后，在实验室和现场环境中均展示了在临床和移动健康应用中更好的泛化能力。这表明暴露于现实世界的变异性使模型能够学习更精细的表示，从而使其在不同任务上更具可适应性。此外，许多任务中在实地数据上预训练的表现意外地优于在临床数据上预训练的表现，突显了使用真实世界多样数据进行训练的重要性。为了鼓励在利用实地数据提高鲁棒性基石模型方面进一步取得进展，我们计划发布Pulse-PPG，为研究人员提供一个强大的资源，以开发更具泛化能力的PPG基础模型。 

---
# VidSketch: Hand-drawn Sketch-Driven Video Generation with Diffusion Control 

**Title (ZH)**: VidSketch：基于素描驱动的扩散控制视频生成 

**Authors**: Lifan Jiang, Shuang Chen, Boxi Wu, Xiaotong Guan, Jiahui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01101)  

**Abstract**: With the advancement of generative artificial intelligence, previous studies have achieved the task of generating aesthetic images from hand-drawn sketches, fulfilling the public's needs for drawing. However, these methods are limited to static images and lack the ability to control video animation generation using hand-drawn sketches. To address this gap, we propose VidSketch, the first method capable of generating high-quality video animations directly from any number of hand-drawn sketches and simple text prompts, bridging the divide between ordinary users and professional artists. Specifically, our method introduces a Level-Based Sketch Control Strategy to automatically adjust the guidance strength of sketches during the generation process, accommodating users with varying drawing skills. Furthermore, a TempSpatial Attention mechanism is designed to enhance the spatiotemporal consistency of generated video animations, significantly improving the coherence across frames. You can find more detailed cases on our official website. 

**Abstract (ZH)**: 随着生成式人工智能的进步，以往的研究已经实现了从手绘草图生成美观图像的任务，满足了公众对于绘画的需求。然而，这些方法局限于静态图像，缺乏使用手绘草图生成视频动画的能力。为解决这一问题，我们提出了VidSketch，这是首个能够直接从任意数量的手绘草图和简单的文本提示生成高质量视频动画的方法，使普通用户能够接近专业艺术家的能力。具体而言，我们的方法引入了一种基于层级的草图控制策略，能够在生成过程中自动调整草图的引导强度，以适应不同绘画技能的用户。此外，我们设计了一种时空注意力机制，以增强生成视频动画的空间-时间一致性，从而显著提高帧间连贯性。更多详细案例请参见我们的官方网站。 

---
# Enhancing Aspect-based Sentiment Analysis with ParsBERT in Persian Language 

**Title (ZH)**: 使用ParsBERT增强波斯语方面的情感分析 

**Authors**: Farid Ariai, Maryam Tayefeh Mahmoudi, Ali Moeini  

**Link**: [PDF](https://arxiv.org/pdf/2502.01091)  

**Abstract**: In the era of pervasive internet use and the dominance of social networks, researchers face significant challenges in Persian text mining including the scarcity of adequate datasets in Persian and the inefficiency of existing language models. This paper specifically tackles these challenges, aiming to amplify the efficiency of language models tailored to the Persian language. Focusing on enhancing the effectiveness of sentiment analysis, our approach employs an aspect-based methodology utilizing the ParsBERT model, augmented with a relevant lexicon. The study centers on sentiment analysis of user opinions extracted from the Persian website 'Digikala.' The experimental results not only highlight the proposed method's superior semantic capabilities but also showcase its efficiency gains with an accuracy of 88.2% and an F1 score of 61.7. The importance of enhancing language models in this context lies in their pivotal role in extracting nuanced sentiments from user-generated content, ultimately advancing the field of sentiment analysis in Persian text mining by increasing efficiency and accuracy. 

**Abstract (ZH)**: 在广泛使用互联网和社交网络主导的时代，研究人员在波斯文文本挖掘方面面临着重大挑战，包括可用的波斯语言数据集不足以及现有语言模型的效率低下。本文特别针对这些挑战，旨在提升针对波斯语的语言模型效率。重点在于提高情感分析的有效性，我们的方法采用基于方面的情感分析方法，并利用ParsBERT模型加以增强，结合相关词汇。研究集中在从波斯网站“Digikala”中提取的用户意见的情感分析上。实验结果不仅突显了所提出方法在语义方面的优越能力，还展示了其效率提升，准确率为88.2%，F1分数为61.7。增强语言模型的重要性在于，它们在从用户生成内容中提取复杂情感方面发挥着关键作用，从而通过提高效率和准确性推动了波斯文本情感分析领域的发展。 

---
# Classic4Children: Adapting Chinese Literary Classics for Children with Large Language Model 

**Title (ZH)**: Classic4Children：通过大型语言模型改编中国文学经典供儿童阅读 

**Authors**: Jiali Chen, Xusen Hei, Yuqi Xue, Zihan Wu, Jiayuan Xie, Yi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2502.01090)  

**Abstract**: Chinese literary classics hold significant cultural and educational value, offering deep insights into morality, history, and human nature. These works often include classical Chinese and complex narratives, making them difficult for children to read. To bridge this gap, we introduce a child-friendly literary adaptation (CLA) task to adapt the Chinese literary classic into engaging and accessible text for children. However, recent large language models (LLMs) overlook children's reading preferences (\ie, vivid character portrayals, concise narrative structures, and appropriate readability), which poses challenges in CLA. In this paper, we propose a method called InstructChild, which augments the LLM with these preferences for adaptation. Specifically, we first obtain the characters' personalities and narrative structure as additional information for fine-grained instruction tuning. Then, we devise a readability metric as the reward to align the LLM with the children's reading level. Finally, a lookahead decoding strategy is applied to improve the readability of the generated text during inference. To support the evaluation of CLA task, we construct the Classic4Children dataset, which comprises both the original and child-friendly versions of the Four Great Classical Novels of Chinese literature. Experimental results show that our InstructChild significantly improves automatic and human evaluation performance. 

**Abstract (ZH)**: 中国的文学经典在文化和教育方面具有重要的价值，它们深刻地揭示了道德、历史和人性。这些作品通常包含古典汉语文本和复杂的叙事结构，这使得儿童难以阅读。为了弥合这一差距，我们引入了一种面向儿童的文学改编任务（CLA），将中国的文学经典改编成适合儿童阅读并充满吸引力的文本。然而，最近的大规模语言模型（LLMs）忽视了儿童的阅读偏好（例如生动的人物刻画、简洁的叙事结构和适当的可读性），这在文学改编中提出了挑战。在本文中，我们提出了一种名为InstructChild的方法，该方法通过引入这些偏好来增强LLM。具体来说，我们首先获取人物的性格特征和叙事结构，作为细化指令调整的额外信息。然后，我们设计了一个可读性度量作为奖励，以使LLM与儿童的阅读水平对齐。最后，在推理过程中应用前瞻解码策略，以提高生成文本的可读性。为了支持CLAA任务的评估，我们构建了一个Classic4Children数据集，该数据集包含中国四大古典小说的原始版本和面向儿童的版本。实验结果表明，我们的InstructChild方法显著提高了自动评估和人工评估的表现。 

---
# Advanced Architectures Integrated with Agentic AI for Next-Generation Wireless Networks 

**Title (ZH)**: 面向下一代无线网络的集成自主AI架构 

**Authors**: Kapal Dev, Sunder Ali Khowaja, Engin Zeydan, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2502.01089)  

**Abstract**: This paper investigates a range of cutting-edge technologies and architectural innovations aimed at simplifying network operations, reducing operational expenditure (OpEx), and enabling the deployment of new service models. The focus is on (i) Proposing novel, more efficient 6G architectures, with both Control and User planes enabling the seamless expansion of services, while addressing long-term 6G network evolution. (ii) Exploring advanced techniques for constrained artificial intelligence (AI) operations, particularly the design of AI agents for real-time learning, optimizing energy consumption, and the allocation of computational resources. (iii) Identifying technologies and architectures that support the orchestration of backend services using serverless computing models across multiple domains, particularly for vertical industries. (iv) Introducing optically-based, ultra-high-speed, low-latency network architectures, with fast optical switching and real-time control, replacing conventional electronic switching to reduce power consumption by an order of magnitude. 

**Abstract (ZH)**: 本文探讨了一系列旨在简化网络运营、降低运维成本（OpEx）并支持新型服务模式部署的前沿技术和架构创新。重点包括：

(i) 提出新型、更高效的6G架构，其中控制面和用户面共同实现服务的无缝扩展，以应对长期的6G网络演化。

(ii) 探索受限人工智能（AI）操作的高级技术，特别是设计用于实时学习、优化能耗和分配计算资源的AI代理。

(iii) 识别支持跨多个领域的垂直行业进行后端服务编排的技术和架构，尤其是使用无服务器计算模型。

(iv) 引入基于光的、超高速、低延迟网络架构，采用快速光交换和实时控制，取代传统的电子交换，能耗降低一个数量级。 

---
# Tool Unlearning for Tool-Augmented LLMs 

**Title (ZH)**: Augmented LLMs 的工具撤销学习 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2502.01083)  

**Abstract**: Tool-augmented large language models (LLMs) are often trained on datasets of query-response pairs, which embed the ability to use tools or APIs directly into the parametric knowledge of LLMs. Tool-augmented LLMs need the ability to forget learned tools due to security vulnerabilities, privacy regulations, or tool deprecations. However, ``tool unlearning'' has not been investigated in unlearning literature. We introduce this novel task, which requires addressing distinct challenges compared to traditional unlearning: knowledge removal rather than forgetting individual samples, the high cost of optimizing LLMs, and the need for principled evaluation metrics. To bridge these gaps, we propose ToolDelete, the first approach for unlearning tools from tool-augmented LLMs. It implements three key properties to address the above challenges for effective tool unlearning and introduces a new membership inference attack (MIA) model for effective evaluation. Extensive experiments on multiple tool learning datasets and tool-augmented LLMs show that ToolDelete effectively unlearns randomly selected tools, while preserving the LLM's knowledge on non-deleted tools and maintaining performance on general tasks. 

**Abstract (ZH)**: 工具增强的大语言模型（LLMs）通常是在查询-响应对的数据集上进行训练的，这将使用工具或API的能力直接嵌入到了LLMs的参数化知识中。由于安全漏洞、隐私法规或工具的弃用，工具增强的LLMs需要具备遗忘已学习工具的能力。然而，“工具遗忘”在遗忘文献中尚未被研究。我们引入了这一新颖的任务，与传统的遗忘任务相比，它需要应对不同的挑战：不仅仅是遗忘单个样本的知识，而是知识删除，优化LLMs的成本很高，以及需要使用原理明确的评估指标。为了解决这些差距，我们提出了ToolDelete，这是第一个用于从工具增强的LLMs中遗忘工具的方法。它实施了三个关键属性，以有效地解决上述挑战，并引入了一种新的成员推理攻击（MIA）模型，用于有效的评估。在多个工具学习数据集和工具增强的LLMs上进行的广泛实验表明，ToolDelete能够有效地遗忘随机选定的工具，同时保留LLMs在未删除工具上的知识，并在一般任务上保持性能。 

---
# The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles 

**Title (ZH)**: 跳跃式的推理曲线？——追踪GPT-[n]和o-[n]模型在多模态谜题中推理性能的演变 

**Authors**: Vernon Y.H. Toh, Yew Ken Chia, Deepanway Ghosal, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2502.01081)  

**Abstract**: The releases of OpenAI's o1 and o3 mark a significant paradigm shift in Large Language Models towards advanced reasoning capabilities. Notably, o3 outperformed humans in novel problem-solving and skill acquisition on the Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI). However, this benchmark is limited to symbolic patterns, whereas humans often perceive and reason about multimodal scenarios involving both vision and language data. Thus, there is an urgent need to investigate advanced reasoning capabilities in multimodal tasks. To this end, we track the evolution of the GPT-[n] and o-[n] series models on challenging multimodal puzzles, requiring fine-grained visual perception with abstract or algorithmic reasoning. The superior performance of o1 comes at nearly 750 times the computational cost of GPT-4o, raising concerns about its efficiency. Our results reveal a clear upward trend in reasoning capabilities across model iterations, with notable performance jumps across GPT-series models and subsequently to o1. Nonetheless, we observe that the o1 model still struggles with simple multimodal puzzles requiring abstract reasoning. Furthermore, its performance in algorithmic puzzles remains poor. We plan to continuously track new models in the series and update our results in this paper accordingly. All resources used in this evaluation are openly available this https URL. 

**Abstract (ZH)**: OpenAI的o1和o3的发布标志着大规模语言模型在高级推理能力方面的一次重要范式转变。尤其是，o3在人工通用智能抽象与推理语料库（ARC-AGI）的新型问题解决和技能获取方面超过了人类，但此基准仅限于符号模式，而人类往往在涉及视觉和语言数据的多模态场景中进行感知和推理。因此，对多模态任务中的高级推理能力进行深入研究变得尤为迫切。为此，我们追踪了GPT-[n]和o-[n]系列模型在具有细粒度视觉感知和抽象或算法推理要求的挑战性多模态拼图中的演变。虽然o1模型的优越性能几乎是GPT-4o的750倍，引发了对其效率的担忧。我们的研究结果表明，推理能力在模型迭代中呈现出明确的上升趋势，尤其是在GPT系列模型和随后的o1模型中观察到了显著的性能跃升。然而，我们还发现o1模型在需要抽象推理的简单多模态拼图中仍显得较为挣扎，而在算法拼图中的表现也相对不佳。我们计划持续跟踪该系列中的新模型，并及时更新本论文中的结果。本研究中使用的所有资源均在此处公开可获取：<https://>。 

---
# Learning Nonlinearity of Boolean Functions: An Experimentation with Neural Networks 

**Title (ZH)**: 学习布尔函数的非线性特性：基于神经网络的实验研究 

**Authors**: Sriram Ranga, Nandish Chattopadhyay, Anupam Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2502.01060)  

**Abstract**: This paper investigates the learnability of the nonlinearity property of Boolean functions using neural networks. We train encoder style deep neural networks to learn to predict the nonlinearity of Boolean functions from examples of functions in the form of a truth table and their corresponding nonlinearity values. We report empirical results to show that deep neural networks are able to learn to predict the property for functions in 4 and 5 variables with an accuracy above 95%. While these results are positive and a disciplined analysis is being presented for the first time in this regard, we should also underline the statutory warning that it seems quite challenging to extend the idea to higher number of variables, and it is also not clear whether one can get advantage in terms of time and space complexity over the existing combinatorial algorithms. 

**Abstract (ZH)**: 本文探讨了使用神经网络学习布尔函数的非线性特性的问题。我们训练了一种编码器风格的深层神经网络，使其能够从真值表形式的布尔函数及其对应的非线性值示例中学习预测非线性特性。我们报告了实验结果，表明深层神经网络能够在4变量和5变量的布尔函数中以超过95%的准确率学习预测该特性。尽管这些结果是积极的，并且首次对这一问题进行了系统分析，但我们也应该强调，将这一想法扩展到更多的变量似乎极具挑战性，而且还不清楚是否能在时间和空间复杂度上有优于现有组合算法的优势。 

---
# Knowledge Synthesis of Photosynthesis Research Using a Large Language Model 

**Title (ZH)**: 使用大型语言模型合成光合作用研究知识 

**Authors**: Seungri Yoon, Woosang Jeon, Sanghyeok Choi, Taehyeong Kim, Tae In Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.01059)  

**Abstract**: The development of biological data analysis tools and large language models (LLMs) has opened up new possibilities for utilizing AI in plant science research, with the potential to contribute significantly to knowledge integration and research gap identification. Nonetheless, current LLMs struggle to handle complex biological data and theoretical models in photosynthesis research and often fail to provide accurate scientific contexts. Therefore, this study proposed a photosynthesis research assistant (PRAG) based on OpenAI's GPT-4o with retrieval-augmented generation (RAG) techniques and prompt optimization. Vector databases and an automated feedback loop were used in the prompt optimization process to enhance the accuracy and relevance of the responses to photosynthesis-related queries. PRAG showed an average improvement of 8.7% across five metrics related to scientific writing, with a 25.4% increase in source transparency. Additionally, its scientific depth and domain coverage were comparable to those of photosynthesis research papers. A knowledge graph was used to structure PRAG's responses with papers within and outside the database, which allowed PRAG to match key entities with 63% and 39.5% of the database and test papers, respectively. PRAG can be applied for photosynthesis research and broader plant science domains, paving the way for more in-depth data analysis and predictive capabilities. 

**Abstract (ZH)**: 生物数据分析工具的发展和大型语言模型（LLMs）的应用为利用AI推动植物科学研究开辟了新的可能性，有助于知识整合和研究空白的识别。然而，当前的LLMs在光合作用研究中难以处理复杂的生物数据和理论模型，往往无法提供准确的科学背景。因此，本研究提出了一种基于OpenAI的GPT-4o并结合检索增强生成（RAG）技术和提示优化的光合作用研究助手（PRAG）。在提示优化过程中使用向量数据库和自动化反馈循环，以提高对光合作用相关查询响应的准确性和相关性。与科学写作相关的五个指标中，PRAG平均提高了8.7%，来源透明度提高了25.4%。此外，其科学深度和领域覆盖范围与光合作用研究论文相当。通过知识图谱结构化PRAG的响应，使其能够与数据库内外的论文匹配，在数据库论文和测试论文中分别匹配了63%和39.5%的关键实体。PRAG可以应用于光合作用研究和更广泛的植物科学领域，为更深入的数据分析和预测能力奠定了基础。 

---
# FetDTIAlign: A Deep Learning Framework for Affine and Deformable Registration of Fetal Brain dMRI 

**Title (ZH)**: FetDTIAlign：一种用于胎儿大脑dMRI的仿射和非刚性配准的深度学习框架 

**Authors**: Bo Li, Qi Zeng, Simon K. Warfield, Davood Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2502.01057)  

**Abstract**: Diffusion MRI (dMRI) provides unique insights into fetal brain microstructure in utero. Longitudinal and cross-sectional fetal dMRI studies can reveal crucial neurodevelopmental changes but require precise spatial alignment across scans and subjects. This is challenging due to low data quality, rapid brain development, and limited anatomical landmarks. Existing registration methods, designed for high-quality adult data, struggle with these complexities. To address this, we introduce FetDTIAlign, a deep learning approach for fetal brain dMRI registration, enabling accurate affine and deformable alignment. FetDTIAlign features a dual-encoder architecture and iterative feature-based inference, reducing the impact of noise and low resolution. It optimizes network configurations and domain-specific features at each registration stage, enhancing both robustness and accuracy. We validated FetDTIAlign on data from 23 to 36 weeks gestation, covering 60 white matter tracts. It consistently outperformed two classical optimization-based methods and a deep learning pipeline, achieving superior anatomical correspondence. Further validation on external data from the Developing Human Connectome Project confirmed its generalizability across acquisition protocols. Our results demonstrate the feasibility of deep learning for fetal brain dMRI registration, providing a more accurate and reliable alternative to classical techniques. By enabling precise cross-subject and tract-specific analyses, FetDTIAlign supports new discoveries in early brain development. 

**Abstract (ZH)**: 扩散磁共振成像（dMRI）可以提供胎儿脑内微结构的独到见解，在母体内的情况下揭示胎儿大脑的微结构特性。纵向和横断面的胎儿dMRI研究可以揭示关键的神经发育变化，但这些研究需要在不同扫描和不同个体之间进行精确的空间对齐。这由于数据质量低、大脑发育迅速以及解剖标志有限而极具挑战性。现有的注册方法适用于高质量的成人数据，难以应对这些复杂情况。为了解决这一问题，我们引入了FetDTIAlign，这是一种用于胎儿脑dMRI注册的深度学习方法，能够实现精确的线性和非线性对齐。FetDTIAlign采用双编码器架构和迭代的基于特征的推理，减少了噪声和低分辨率的影响。它在每个注册阶段优化网络配置和领域特异性特征，增强了鲁棒性和准确性。我们通过涵盖妊娠23至36周的60条白质束的数据对FetDTIAlign进行了验证，该方法在古典型优化方法和深度学习管道中表现出优越的解剖对应性。进一步使用发育中的人类连接组项目（Developing Human Connectome Project, DHCP）的外部数据进行验证，证实了其在不同采集协议下的普适性。我们的研究结果表明，深度学习在胎儿脑dMRI注册中具有可行性，提供了与经典技术相比更为准确可靠的替代方法。通过实现精确的跨个体和束特异性分析，FetDTIAlign支持了早期脑发育的新发现。 

---
# Sparks of Explainability: Recent Advancements in Explaining Large Vision Models 

**Title (ZH)**: 可解释性的火花：大型视觉模型解释领域的 recent 进展 

**Authors**: Thomas Fel  

**Link**: [PDF](https://arxiv.org/pdf/2502.01048)  

**Abstract**: This thesis explores advanced approaches to improve explainability in computer vision by analyzing and modeling the features exploited by deep neural networks. Initially, it evaluates attribution methods, notably saliency maps, by introducing a metric based on algorithmic stability and an approach utilizing Sobol indices, which, through quasi-Monte Carlo sequences, allows a significant reduction in computation time. In addition, the EVA method offers a first formulation of attribution with formal guarantees via verified perturbation analysis.
Experimental results indicate that in complex scenarios these methods do not provide sufficient understanding, particularly because they identify only "where" the model focuses without clarifying "what" it perceives. Two hypotheses are therefore examined: aligning models with human reasoning -- through the introduction of a training routine that integrates the imitation of human explanations and optimization within the space of 1-Lipschitz functions -- and adopting a conceptual explainability approach.
The CRAFT method is proposed to automate the extraction of the concepts used by the model and to assess their importance, complemented by MACO, which enables their visualization. These works converge towards a unified framework, illustrated by an interactive demonstration applied to the 1000 ImageNet classes in a ResNet model. 

**Abstract (ZH)**: 本论文探讨了通过分析和建模深度神经网络使用的特征来改进计算机视觉中的解释性方法。初始阶段，论文评估了归因方法，特别是显著图，并引入了一个基于算法稳定性的评估指标，以及利用Sobol指数的方法，通过拟蒙特卡洛序列显著减少了计算时间。此外，EVA方法通过验证扰动分析提供了一种具有形式保证的归因方法。
实验结果表明，在复杂场景下，这些方法未能提供足够的理解，特别是因为它们仅能识别模型关注的“哪里”，而未能解释模型感知的“什么”。因此，研究了两个假设：通过引入一种将人类解释模仿与1-Lipschitz函数空间内的优化相结合的训练过程，使模型与人类推理相一致；以及采用概念性的解释方法。
提出了一种CRAFT方法来自动提取模型使用的概念，并评估其重要性，并通过MACO方法使其可视化。这些工作共同形成了一种统一的框架，并通过应用于ResNet模型中的ImageNet 1000个类别的交互式演示进行了展示。 

---
# eagle: early approximated gradient based learning rate estimator 

**Title (ZH)**: Eagle: 早期近似梯度为基础的学习率估算器 

**Authors**: Takumi Fujimoto, Hiroaki Nishi  

**Link**: [PDF](https://arxiv.org/pdf/2502.01036)  

**Abstract**: We propose EAGLE update rule, a novel optimization method that accelerates loss convergence during the early stages of training by leveraging both current and previous step parameter and gradient values. The update algorithm estimates optimal parameters by computing the changes in parameters and gradients between consecutive training steps and leveraging the local curvature of the loss landscape derived from these changes. However, this update rule has potential instability, and to address that, we introduce an adaptive switching mechanism that dynamically selects between Adam and EAGLE update rules to enhance training stability. Experiments on standard benchmark datasets demonstrate that EAGLE optimizer, which combines this novel update rule with the switching mechanism achieves rapid training loss convergence with fewer epochs, compared to conventional optimization methods. 

**Abstract (ZH)**: 我们提出了一种名为EAGLE的更新规则，这是一种新颖的优化方法，在训练早期阶段通过利用当前和上一步的参数及梯度值来加速损失函数的收敛。该更新算法通过计算连续训练步骤之间的参数和梯度变化，并利用这些变化推导出损失景观的局部曲率来估计最优参数。然而，该更新规则存在潜在的不稳定性，为此，我们引入了一种自适应切换机制，该机制能够动态选择Adam和EAGLE更新规则，以增强训练稳定性。实验结果表明，在标准基准数据集上，结合该新型更新规则和切换机制的EAGLE优化器能够在较少的训练周期内实现快速的训练损失收敛，优于传统的优化方法。 

---
# Comprehensive Modeling Approaches for Forecasting Bitcoin Transaction Fees: A Comparative Study 

**Title (ZH)**: 全面建模方法在预测比特币交易费用方面的研究：一项比较分析 

**Authors**: Jiangqin Ma, Erfan Mahmoudinia  

**Link**: [PDF](https://arxiv.org/pdf/2502.01029)  

**Abstract**: Transaction fee prediction in Bitcoin's ecosystem represents a crucial challenge affecting both user costs and miner revenue optimization. This study presents a systematic evaluation of six predictive models for forecasting Bitcoin transaction fees across a 24-hour horizon (144 blocks): SARIMAX, Prophet, Time2Vec, Time2Vec with Attention, a Hybrid model combining SARIMAX with Gradient Boosting, and the Temporal Fusion Transformer (TFT). Our approach integrates comprehensive feature engineering spanning mempool metrics, network parameters, and historical fee patterns to capture the multifaceted dynamics of fee behavior.
Through rigorous 5-fold cross-validation and independent testing, our analysis reveals that traditional statistical approaches outperform more complex deep learning architectures. The SARIMAX model achieves superior accuracy on the independent test set, while Prophet demonstrates strong performance during cross-validation. Notably, sophisticated deep learning models like Time2Vec and TFT show comparatively lower predictive power despite their architectural complexity. This performance disparity likely stems from the relatively constrained training dataset of 91 days, suggesting that deep learning models may achieve enhanced results with extended historical data.
These findings offer significant practical implications for cryptocurrency stakeholders, providing empirically-validated guidance for fee-sensitive decision making while illuminating critical considerations in model selection based on data constraints. The study establishes a foundation for advanced fee prediction while highlighting the current advantages of traditional statistical methods in this domain. 

**Abstract (ZH)**: 比特币生态系统中的交易费用预测是一项关键挑战，直接影响到用户成本和矿工收益优化。本研究系统评估了六种预测模型在24小时（144块）时间范围内的比特币交易费用预测性能：SARIMAX、Prophet、Time2Vec、Time2Vec结合注意机制、SARIMAX与梯度提升相结合的混合模型以及时间融合变换器（TFT）。我们通过全面的特征工程，涵盖了内存池指标、网络参数和历史费用模式，来捕捉费用行为的复杂动态。

通过严格的5折交叉验证和独立测试，我们的分析表明传统统计方法优于更复杂的深度学习架构。SARIMAX模型在独立测试集上实现了更高的准确性，而Prophet在交叉验证过程中表现出强劲的性能。值得注意的是，尽管具有复杂架构的模型如Time2Vec和TFT在预测能力上相对较低，但在模型复杂性方面具有优势。这种性能差异可能源于相对受限的91天训练数据集，这意味着在具有更长历史数据的情况下，深度学习模型可能会取得更好的结果。

这些发现为加密货币利益相关者提供了重要的实践意义，提供了基于数据限制的实证验证指导，同时揭示了模型选择中的关键考虑因素。研究奠定了高级费用预测的基础，同时强调了在这一领域传统统计方法的现有优势。 

---
# Refining Adaptive Zeroth-Order Optimization at Ease 

**Title (ZH)**: 轻松精化自适应零阶优化算法 

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2502.01014)  

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (I) the first analysis to the variance reduction of first moment estimate in ZO optimization, (II) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (III) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (IV) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges. 

**Abstract (ZH)**: 近年来，零阶（Zeroth-Order, ZO）优化在无法或不便于获取梯度信息的情景中（如黑盒系统和资源受限环境）发挥着重要作用。尽管现有的自适应方法，如ZO-AdaMM，展现了一定的潜力，但它们在优化过程中通常未能充分利用动量信息，从而导致了不佳的收敛性能。为克服这些限制，本文提出了改进自适应零阶优化（Refined Adaptive Zeroth-Order Optimization, R-AdaZO）。具体而言，我们首先展示了对一阶动量估计的未利用的方差减少效应如何提升ZO梯度估计的准确性和稳定性，从而改进ZO更新的准确性和稳定性。然后，我们基于这些方差减少的梯度估计来细化二阶动量估计，更好地捕捉优化景观的几何结构，使ZO更新更为有效。我们提出了严谨的理论分析，证明了（I）一阶动量估计在ZO优化中首次的方差减少分析，（II）改进的二阶动量估计更准确地逼近其无方差的理想情况，（III）首个方差感知的自适应ZO方法收敛框架，其独立兴趣，以及（IV）R-AdaZO相较于现有基线（如ZO-AdaMM）更快的收敛速度。我们在多种实验中进一步验证了R-AdaZO的优越收敛性，表明R-AdaZO为实际应用中的ZO优化挑战提供了一个改进的解决方案。 

---
# Encrypted Large Model Inference: The Equivariant Encryption Paradigm 

**Title (ZH)**: 加密的大模型推理：等变加密范式 

**Authors**: James Buban, Hongyang Zhang, Claudio Angione, Harry Yang, Ahmad Farhan, Seyfal Sultanov, Michael Du, Xuran Ma, Zihao Wang, Yue Zhao, Arria Owlia, Fielding Johnston, Patrick Colangelo  

**Link**: [PDF](https://arxiv.org/pdf/2502.01013)  

**Abstract**: Large scale deep learning model, such as modern language models and diffusion architectures, have revolutionized applications ranging from natural language processing to computer vision. However, their deployment in distributed or decentralized environments raises significant privacy concerns, as sensitive data may be exposed during inference. Traditional techniques like secure multi-party computation, homomorphic encryption, and differential privacy offer partial remedies but often incur substantial computational overhead, latency penalties, or limited compatibility with non-linear network operations. In this work, we introduce Equivariant Encryption (EE), a novel paradigm designed to enable secure, "blind" inference on encrypted data with near zero performance overhead. Unlike fully homomorphic approaches that encrypt the entire computational graph, EE selectively obfuscates critical internal representations within neural network layers while preserving the exact functionality of both linear and a prescribed set of non-linear operations. This targeted encryption ensures that raw inputs, intermediate activations, and outputs remain confidential, even when processed on untrusted infrastructure. We detail the theoretical foundations of EE, compare its performance and integration complexity against conventional privacy preserving techniques, and demonstrate its applicability across a range of architectures, from convolutional networks to large language models. Furthermore, our work provides a comprehensive threat analysis, outlining potential attack vectors and baseline strategies, and benchmarks EE against standard inference pipelines in decentralized settings. The results confirm that EE maintains high fidelity and throughput, effectively bridging the gap between robust data confidentiality and the stringent efficiency requirements of modern, large scale model inference. 

**Abstract (ZH)**: 大规模深度学习模型，如现代语言模型和扩散架构，已经彻底改变了从自然语言处理到计算机视觉的各种应用。然而，在分布式或分散式环境中部署这些模型会引发重大的隐私问题，因为敏感数据可能会在推理过程中泄露。传统的安全多方计算、同态加密和差分隐私等技术提供了一定的解决方案，但往往会导致显著的计算开销、延迟增加或与非线性网络操作的兼容性有限。在本项工作中，我们引入了一种新的加密范式——对称加密 (Equivariant Encryption, EE)，旨在实现在加密数据上进行近乎零性能开销的安全“盲”推理。与完全同态加密方法将整个计算图完全加密不同，EE 选择性地对神经网络层中的关键内部表示进行混淆处理，同时保留线性和特定非线性操作的精确功能。这种有针对性的加密确保了即使在不信任的基础设施上处理时，原始输入、中间激活和输出仍然保持机密。我们详细介绍了 EE 的理论基础，将其实现性能和集成复杂度与传统的隐私保护技术进行了对比，并展示了其在从卷积网络到大规模语言模型等各种架构中的适用性。此外，我们的工作还提供了全面的安全威胁分析，概述了潜在的攻击向量和基准策略，并在分散式环境下将 EE 与标准推理管道进行了基准测试。结果证实，EE 能够保持高保真度和吞吐量，有效地弥补了高效数据保密性和现代大规模模型推理严格效率要求之间的差距。 

---
# MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs 

**Title (ZH)**: MergeME：同质和异构MoE模型合并技术 

**Authors**: Yuhang Zhou, Giannis Karamanolakis, Victor Soto, Anna Rumshisky, Mayank Kulkarni, Furong Huang, Wei Ai, Jianhua Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00997)  

**Abstract**: The recent success of specialized Large Language Models (LLMs) in domains such as mathematical reasoning and coding has led to growing interest in methods for merging these expert LLMs into a unified Mixture-of-Experts (MoE) model, with the goal of enhancing performance in each domain while retaining effectiveness on general tasks. However, the effective merging of expert models remains an open challenge, especially for models with highly divergent weight parameters or different architectures. State-of-the-art MoE merging methods only work with homogeneous model architectures and rely on simple unweighted averaging to merge expert layers, which does not address parameter interference and requires extensive fine-tuning of the merged MoE to restore performance. To address these limitations, this paper introduces new MoE merging techniques, including strategies to mitigate parameter interference, routing heuristics to reduce the need for MoE fine-tuning, and a novel method for merging experts with different architectures. Extensive experiments across multiple domains demonstrate the effectiveness of our proposed methods, reducing fine-tuning costs, improving performance over state-of-the-art methods, and expanding the applicability of MoE merging. 

**Abstract (ZH)**: recent 成功的专业大型语言模型（LLMs）在数学推理和编程等领域的应用，使得人们越来越关注如何将这些专业模型有效地结合到统一的专家混合模型（MoE）中，以增强每个领域的性能同时保持对一般任务的有效性。然而，有效结合专家模型仍然是一个开放的挑战，特别是在模型权重参数差异大或架构不同的情况下。目前最先进的MoE结合方法仅适用于同类模型架构，并依赖简单的加权平均来合并专家层，这一方法没有解决参数干扰问题，且需要对合并的MoE进行大量微调才能恢复性能。为了解决这些限制，本文引入了新的MoE结合技术，包括减轻参数干扰的策略、减少MoE微调需求的路由启发式方法，以及一种能够合并不同架构专家模型的新方法。通过在多个领域的广泛实验，证明了我们提出方法的有效性，降低了微调成本，提高了相较于最先进的方法的性能，并扩大了MoE结合的应用范围。 

---
# ChartCitor: Multi-Agent Framework for Fine-Grained Chart Visual Attribution 

**Title (ZH)**: ChartCitor：细粒度图表视觉归属多Agent框架 

**Authors**: Kanika Goswami, Puneet Mathur, Ryan Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2502.00989)  

**Abstract**: Large Language Models (LLMs) can perform chart question-answering tasks but often generate unverified hallucinated responses. Existing answer attribution methods struggle to ground responses in source charts due to limited visual-semantic context, complex visual-text alignment requirements, and difficulties in bounding box prediction across complex layouts. We present ChartCitor, a multi-agent framework that provides fine-grained bounding box citations by identifying supporting evidence within chart images. The system orchestrates LLM agents to perform chart-to-table extraction, answer reformulation, table augmentation, evidence retrieval through pre-filtering and re-ranking, and table-to-chart mapping. ChartCitor outperforms existing baselines across different chart types. Qualitative user studies show that ChartCitor helps increase user trust in Generative AI by providing enhanced explainability for LLM-assisted chart QA and enables professionals to be more productive. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以执行图表问答任务，但通常会生成未经验证的虚构回答。现有的答案归因方法由于缺乏视觉语义上下文、复杂的视觉文本对齐要求以及在复杂布局中预测边界框的困难，在将响应与来源图表对接方面存在挑战。我们提出了ChartCitor，这是一个多agent框架，通过在图表图像内识别支持性证据来提供精细粒度的边界框引证。该系统协调LLM代理执行图表到表格提取、答案重述、表格扩充、基于预筛选和再排序的证据检索，以及表格到图表映射。在不同图表类型的基准测试中，ChartCitor表现优于现有基线。定性的用户研究表明，ChartCitor通过增强基于LLM的图表问答的解释性，帮助用户增加对生成式人工智能的信任，并使专业人士能够更高效地工作。 

---
# PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback 

**Title (ZH)**: PlotGen：基于多模态反馈的多智能体LLM科学数据分析可视化 

**Authors**: Kanika Goswami, Puneet Mathur, Ryan Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2502.00988)  

**Abstract**: Scientific data visualization is pivotal for transforming raw data into comprehensible visual representations, enabling pattern recognition, forecasting, and the presentation of data-driven insights. However, novice users often face difficulties due to the complexity of selecting appropriate tools and mastering visualization techniques. Large Language Models (LLMs) have recently demonstrated potential in assisting code generation, though they struggle with accuracy and require iterative debugging. In this paper, we propose PlotGen, a novel multi-agent framework aimed at automating the creation of precise scientific visualizations. PlotGen orchestrates multiple LLM-based agents, including a Query Planning Agent that breaks down complex user requests into executable steps, a Code Generation Agent that converts pseudocode into executable Python code, and three retrieval feedback agents - a Numeric Feedback Agent, a Lexical Feedback Agent, and a Visual Feedback Agent - that leverage multimodal LLMs to iteratively refine the data accuracy, textual labels, and visual correctness of generated plots via self-reflection. Extensive experiments show that PlotGen outperforms strong baselines, achieving a 4-6 percent improvement on the MatPlotBench dataset, leading to enhanced user trust in LLM-generated visualizations and improved novice productivity due to a reduction in debugging time needed for plot errors. 

**Abstract (ZH)**: 科学数据可视化对于将原始数据转换为易于理解的可视化表示至关重要，能够识别模式、进行预测，并展示数据驱动的洞察。然而，初学者用户往往因选择合适的工具和掌握可视化技术的复杂性而遇到困难。大型语言模型（LLMs）最近在辅助代码生成方面显示出潜力，但它们在准确性方面存在局限，需要进行迭代调试。本文提出了一种名为PlotGen的新颖多智能体框架，旨在自动化精准科学可视化生成。PlotGen协调多个基于LLM的智能体，包括一个查询规划智能体，负责将复杂的用户请求分解为可执行步骤；一个代码生成智能体，将伪代码转换为可执行的Python代码；以及三个检索反馈智能体——数值反馈智能体、词汇反馈智能体和视觉反馈智能体，利用多模态LLMs通过自我反思迭代地 refinizing生成图表的数据准确性、文本标签和视觉正确性。广泛的实验表明，PlotGen在强基线方法中表现出色，其在MatPlotBench数据集上的性能提高了4-6个百分点，从而增强了用户对由LLM生成的可视化成果的信任，并通过减少为图表错误进行调试所需的时间来提高初学者的效率。 

---
# RandLoRA: Full-rank parameter-efficient fine-tuning of large models 

**Title (ZH)**: RandLoRA: 低秩高秩参数高效微调大规模模型 

**Authors**: Paul Albert, Frederic Z. Zhang, Hemanth Saratchandran, Cristian Rodriguez-Opazo, Anton van den Hengel, Ehsan Abbasnejad  

**Link**: [PDF](https://arxiv.org/pdf/2502.00987)  

**Abstract**: Low-Rank Adaptation (LoRA) and its variants have shown impressive results in reducing the number of trainable parameters and memory requirements of large transformer networks while maintaining fine-tuning performance. However, the low-rank nature of the weight update inherently limits the representation power of fine-tuned models, potentially compromising performance on complex tasks. This raises a critical question: when a performance gap between LoRA and standard fine-tuning is observed, is it due to the reduced number of trainable parameters or the rank deficiency? This paper aims to answer this question by introducing RandLoRA, a parameter-efficient method that performs full-rank updates using a learned linear combinations of low-rank, non-trainable random matrices. Our method limits the number of trainable parameters by restricting optimization to diagonal scaling matrices applied to the fixed random matrices. This allows us to effectively overcome the low-rank limitations while maintaining parameter and memory efficiency during training. Through extensive experimentation across vision, language, and vision-language benchmarks, we systematically evaluate the limitations of LoRA and existing random basis methods. Our findings reveal that full-rank updates are beneficial across vision and language tasks individually, and even more so for vision-language tasks, where RandLoRA significantly reduces -- and sometimes eliminates -- the performance gap between standard fine-tuning and LoRA, demonstrating its efficacy. 

**Abstract (ZH)**: 低秩适应（LoRA）及其变种在保持微调性能的同时，显著减少了大型变换器网络中的可训练参数数量和内存需求。然而，权重更新的低秩性质固有地限制了微调模型的表示能力，可能在复杂任务上损害性能。这引发了重要问题：当观察到LoRA与标准微调之间的性能差异时，这种差异是由于减少的可训练参数数量还是因为秩不足造成的？本文通过引入RandLoRA，一种参数高效的微调方法，来回答这个问题。RandLoRA利用学习到的低秩非可训练随机矩阵的线性组合来进行全秩更新，方法是通过将优化限制在应用于固定随机矩阵的对角缩放矩阵来限制可训练参数的数量。这种方法在保持训练过程中的参数和内存效率的同时，有效地克服了低秩限制。通过对视觉、语言和视觉-语言基准任务的广泛实验，我们系统地评估了LoRA及其现有随机基方法的局限性。我们的发现表明，在视觉和语言任务中，全秩更新是有益的，而在视觉-语言任务中，RandLoRA显著缩小了标准微调与LoRA之间的性能差距，并且有时甚至完全消除了这种差距，这证明了该方法的有效性。 

---
# Forecasting VIX using interpretable Kolmogorov-Arnold networks 

**Title (ZH)**: 使用可解释的柯尔莫哥洛夫-阿诺尔德网络预测VIX 

**Authors**: So-Yoon Cho, Sungchul Lee, Hyun-Gyoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.00980)  

**Abstract**: This paper presents the use of Kolmogorov-Arnold Networks (KANs) for forecasting the CBOE Volatility Index (VIX). Unlike traditional MLP-based neural networks that are often criticized for their black-box nature, KAN offers an interpretable approach via learnable spline-based activation functions and symbolification. Based on a parsimonious architecture with symbolic functions, KAN expresses a forecast of the VIX as a closed-form in terms of explanatory variables, and provide interpretable insights into key characteristics of the VIX, including mean reversion and the leverage effect. Through in-depth empirical analysis across multiple datasets and periods, we show that KANs achieve competitive forecasting performance while requiring significantly fewer parameters compared to MLP-based neural network models. Our findings demonstrate the capacity and potential of KAN as an interpretable financial time-series forecasting method. 

**Abstract (ZH)**: 本文介绍了使用柯尔莫哥洛夫-阿诺尔德网络（KANs）来预测芝加哥期权交易所波动率指数（VIX）。与通常因其黑箱性质而受到批评的传统基于多层感知机（MLP）的神经网络不同，KAN 通过可学习的分段函数激活函数和符号化提供了可解释的方法。基于简约的架构和符号函数，KAN 将 VIX 的预测表达为解释变量的封闭形式，并提供对 VIX 关键特征的可解释洞察，包括均值回复和杠杆效应。通过对多个数据集和时期进行深入的实证分析，我们表明，在参数数量远少于基于 MLP 的神经网络模型的情况下，KAN 能够获得竞争力的预测性能。我们的研究结果表明，KAN 作为一种可解释的金融市场时间序列预测方法具备巨大的潜力和能力。 

---
# ML-Dev-Bench: Comparative Analysis of AI Agents on ML development workflows 

**Title (ZH)**: ML-Dev-Bench：比较分析AI代理在机器学习开发工作流中的表现 

**Authors**: Harshith Padigela, Chintan Shah, Dinkar Juyal  

**Link**: [PDF](https://arxiv.org/pdf/2502.00964)  

**Abstract**: In this report, we present ML-Dev-Bench, a benchmark aimed at testing agentic capabilities on applied Machine Learning development tasks. While existing benchmarks focus on isolated coding tasks or Kaggle-style competitions, ML-Dev-Bench tests agents' ability to handle the full complexity of ML development workflows. The benchmark assesses performance across critical aspects including dataset handling, model training, improving existing models, debugging, and API integration with popular ML tools. We evaluate three agents -- ReAct, Openhands, and AIDE -- on a diverse set of 25 tasks, providing insights into their strengths and limitations in handling practical ML development challenges. 

**Abstract (ZH)**: 在本报告中，我们介绍了ML-Dev-Bench，这是一个旨在测试代理在实际机器学习开发任务中的代理能力的基准测试。现有的基准测试主要集中在独立的编码任务或Kaggle风格的比赛上，而ML-Dev-Bench则测试代理处理整个机器学习开发工作流的复杂性的能力。该基准测试从数据集处理、模型训练、改进现有模型、调试以及与流行的机器学习工具的API集成等多个关键方面评估性能。我们在这25项多样化的任务中评估了三个代理——ReAct、Openhands和AIDE，提供了它们在应对实际机器学习开发挑战时的优势和局限性的见解。 

---
# An MDP Model for Censoring in Harvesting Sensors: Optimal and Approximated Solutions 

**Title (ZH)**: 基于Harvesting传感器中的过滤问题的MDP模型：最优和近似解 

**Authors**: Jesus Fernandez-Bes, Jesus Cid-Sueiro, Antonio G. Marques  

**Link**: [PDF](https://arxiv.org/pdf/2502.00940)  

**Abstract**: In this paper, we propose a novel censoring policy for energy-efficient transmissions in energy-harvesting sensors. The problem is formulated as an infinite-horizon Markov Decision Process (MDP). The objective to be optimized is the expected sum of the importance (utility) of all transmitted messages. Assuming that such importance can be evaluated at the transmitting node, we show that, under certain conditions on the battery model, the optimal censoring policy is a threshold function on the importance value. Specifically, messages are transmitted only if their importance is above a threshold whose value depends on the battery level. Exploiting this property, we propose a model-based stochastic scheme that approximates the optimal solution, with less computational complexity and faster convergence speed than a conventional Q-learning algorithm. Numerical experiments in single-hop and multi-hop networks confirm the analytical advantages of the proposed scheme. 

**Abstract (ZH)**: 在本文中，我们提出了一种新型的截断策略，用于具有能量收集能力的传感器的能量高效传输。该问题被形式化为无限 horizon 的马尔可夫决策过程（MDP）。优化的目标是所有传输消息的重要性的期望和（效用）。假设此类重要性可以在发送节点处进行评估，我们证明在某些电池模型条件下，最优的截断策略是重要性值的阈值函数。具体而言，只有当消息的重要性超过一个阈值（该阈值的值取决于电池水平）时，才会发送消息。利用这一特性，我们提出了一种基于模型的随机方案，该方案能近似最优解，并且与传统的 Q 学习算法相比，具有更低的计算复杂度和更快的收敛速度。在单跳和多跳网络中的数值实验验证了所提方案的分析优势。 

---
# Fruit Fly Classification (Diptera: Tephritidae) in Images, Applying Transfer Learning 

**Title (ZH)**: 图像中蒂斐洛蒂德科 fruit fly 分类：应用迁移学习 

**Authors**: Erick Andrew Bustamante Flores, Harley Vera Olivera, Ivan Cesar Medrano Valencia, Carlos Fernando Montoya Cubas  

**Link**: [PDF](https://arxiv.org/pdf/2502.00939)  

**Abstract**: This study develops a transfer learning model for the automated classification of two species of fruit flies, Anastrepha fraterculus and Ceratitis capitata, in a controlled laboratory environment. The research addresses the need to optimize identification and classification, which are currently performed manually by experts, being affected by human factors and facing time challenges. The methodological process of this study includes the capture of high-quality images using a mobile phone camera and a stereo microscope, followed by segmentation to reduce size and focus on relevant morphological areas. The images were carefully labeled and preprocessed to ensure the quality and consistency of the dataset used to train the pre-trained convolutional neural network models VGG16, VGG19, and Inception-v3. The results were evaluated using the F1-score, achieving 82% for VGG16 and VGG19, while Inception-v3 reached an F1-score of 93%. Inception-v3's reliability was verified through model testing in uncontrolled environments, with positive results, complemented by the Grad-CAM technique, demonstrating its ability to capture essential morphological features. These findings indicate that Inception-v3 is an effective and replicable approach for classifying Anastrepha fraterculus and Ceratitis capitata, with potential for implementation in automated monitoring systems. 

**Abstract (ZH)**: 本研究开发了一种迁移学习模型，用于在受控实验室环境中自动对两种果蝇种类——弗雷克勒斯果实蝇（Anastrepha fraterculus）和无斑果实蝇（Ceratitis capitata）进行分类。研究旨在优化识别和分类过程，目前这些过程主要由专家手工完成，受到人为因素的影响，并面临时间上的挑战。本研究的方法包括使用移动手机摄像头和立体显微镜拍摄高质量图像，然后进行分割以减少图像大小并聚焦于相关形态学区域。图像经过仔细标注和预处理，以确保用于训练预训练卷积神经网络模型（如VGG16、VGG19和Inception-v3）的数据集的质量和一致性。结果通过F1分数进行评估，VGG16和VGG19的F1分数分别为82%，而Inception-v3达到93%。Inception-v3的可靠性通过在不受控环境中对模型进行测试得到验证，并且结果为正，同时通过Grad-CAM技术进一步证明了其捕捉关键形态学特征的能力。这些发现表明，Inception-v3是有效且可复制的分类弗雷克勒斯果实蝇和无斑果实蝇的方法，并具有在自动化监测系统中实施的潜力。 

---
# Towards Efficient Large Multimodal Model Serving 

**Title (ZH)**: 面向高效大规模多模态模型服务 

**Authors**: Haoran Qiu, Anish Biswas, Zihan Zhao, Jayashree Mohan, Alind Khare, Esha Choukse, Íñigo Goiri, Zeyu Zhang, Haiying Shen, Chetan Bansal, Ramachandran Ramjee, Rodrigo Fonseca  

**Link**: [PDF](https://arxiv.org/pdf/2502.00937)  

**Abstract**: Recent advances in generative AI have led to large multi-modal models (LMMs) capable of simultaneously processing inputs of various modalities such as text, images, video, and audio. While these models demonstrate impressive capabilities, efficiently serving them in production environments poses significant challenges due to their complex architectures and heterogeneous resource requirements.
We present the first comprehensive systems analysis of two prominent LMM architectures, decoder-only and cross-attention, on six representative open-source models. We investigate their multi-stage inference pipelines and resource utilization patterns that lead to unique systems design implications. We also present an in-depth analysis of production LMM inference traces, uncovering unique workload characteristics, including variable, heavy-tailed request distributions, diverse modal combinations, and bursty traffic patterns.
Our key findings reveal that different LMM inference stages exhibit highly heterogeneous performance characteristics and resource demands, while concurrent requests across modalities lead to significant performance interference. To address these challenges, we propose a decoupled serving architecture that enables independent resource allocation and adaptive scaling for each stage. We further propose optimizations such as stage colocation to maximize throughput and resource utilization while meeting the latency objectives. 

**Abstract (ZH)**: 近年来，生成式AI的进步催生了大型多模态模型（LMMs），这些模型能够同时处理各种模态的输入，如文本、图像、视频和音频。尽管这些模型展现了令人印象深刻的性能，但在生产环境中高效地提供服务却面临着巨大的挑战，这主要是由于它们复杂的架构和多样的资源需求。

我们首次对两种主要的LMM架构——解码器唯一架构和交叉注意力架构——进行了全面的系统分析，涉及六种代表性开源模型。我们探讨了它们的多阶段推理流水线和资源利用模式，这些模式导致了独特的系统设计影响。我们还对生产环境中LMM的推理痕迹进行了深入分析，揭示了独特的负载特征，包括可变的、重尾的请求分布、多样的模态组合以及突发的流量模式。

我们的关键发现揭示了不同LMM推理阶段具有高度异质的性能特性和资源需求特征，同时跨模态的并发请求会带来显著的性能干扰。为应对这些挑战，我们提出了一种解耦的服务架构，允许为每个阶段独立分配资源并实现动态扩展。我们还提出了阶段共置等优化措施，以最大化吞吐量和资源利用率，同时满足延迟目标。 

---
# Attention Sinks and Outlier Features: A 'Catch, Tag, and Release' Mechanism for Embeddings 

**Title (ZH)**: 注意力陷阱和离群特征：一种“捕捉、标记和释放”机制用于嵌入 

**Authors**: Stephen Zhang, Mustafa Khan, Vardan Papyan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00919)  

**Abstract**: Two prominent features of large language models (LLMs) is the presence of large-norm (outlier) features and the tendency for tokens to attend very strongly to a select few tokens. Despite often having no semantic relevance, these select tokens, called attention sinks, along with the large outlier features, have proven important for model performance, compression, and streaming. Consequently, investigating the roles of these phenomena within models and exploring how they might manifest in the model parameters has become an area of active interest. Through an empirical investigation, we demonstrate that attention sinks utilize outlier features to: catch a sequence of tokens, tag the captured tokens by applying a common perturbation, and then release the tokens back into the residual stream, where the tagged tokens are eventually retrieved. We prove that simple tasks, like averaging, necessitate the 'catch, tag, release' mechanism hence explaining why it would arise organically in modern LLMs. Our experiments also show that the creation of attention sinks can be completely captured in the model parameters using low-rank matrices, which has important implications for model compression and substantiates the success of recent approaches that incorporate a low-rank term to offset performance degradation. 

**Abstract (ZH)**: 大型语言模型（LLM）的两个显著特征是存在大规模范数异常特征（outlier features）以及令牌倾向于非常强烈地关注几项特定的令牌。尽管这些特定的令牌通常没有语义相关性，但被称为注意力陷阱（attention sinks）的这些选择的令牌，连同那些大规模的异常特征，已被证明对模型性能、压缩和流式处理非常重要。因此，研究这些现象在模型中的作用以及它们如何在模型参数中表现出来，已成为一个活跃的研究领域。通过实证研究，我们证明了注意力陷阱利用这些异常特征来：捕捉一系列令牌、通过对这些捕捉到的令牌应用相同的扰动来为其打标签，然后将这些令牌释放回残差流，在此过程中被打上标签的令牌最终会被检索。我们证明了像平均这类简单的任务需要这种“捕捉、打标签、释放”的机制，从而解释了为什么这种机制会在现代LLM中自然出现。我们的实验还表明，注意力陷阱的创造可以完全通过低秩矩阵来捕捉，这在模型压缩方面具有重要意义，并证实了近年来通过引入低秩项来抵消性能下降的做法的成功。 

---
# Embracing Dialectic Intersubjectivity: Coordination of Different Perspectives in Content Analysis with LLM Persona Simulation 

**Title (ZH)**: 拥抱辩证的 intersubjectivity：内容分析中基于大规模语言模型个性模拟的不同视角协调 

**Authors**: Taewoo Kang, Kjerstin Thorson, Tai-Quan Peng, Dan Hiaeshutter-Rice, Sanguk Lee, Stuart Soroka  

**Link**: [PDF](https://arxiv.org/pdf/2502.00903)  

**Abstract**: This study attempts to advancing content analysis methodology from consensus-oriented to coordination-oriented practices, thereby embracing diverse coding outputs and exploring the dynamics among differential perspectives. As an exploratory investigation of this approach, we evaluate six GPT-4o configurations to analyze sentiment in Fox News and MSNBC transcripts on Biden and Trump during the 2020 U.S. presidential campaign, examining patterns across these models. By assessing each model's alignment with ideological perspectives, we explore how partisan selective processing could be identified in LLM-Assisted Content Analysis (LACA). Findings reveal that partisan persona LLMs exhibit stronger ideological biases when processing politically congruent content. Additionally, intercoder reliability is higher among same-partisan personas compared to cross-partisan pairs. This approach enhances the nuanced understanding of LLM outputs and advances the integrity of AI-driven social science research, enabling simulations of real-world implications. 

**Abstract (ZH)**: 本研究旨在从基于共识的方法转向基于协调的方法，推进内容分析方法的发展，从而包容多元编码输出，并探索不同视角之间的动态关系。作为这一方法的探索性研究，我们评估了六种不同的GPT-4配置，以分析2020年美国总统竞选期间福克斯新闻和MSNBC关于拜登和特朗普的转录文本的情感倾向，研究这些模型之间的模式。通过评估每个模型与意识形态视角的一致性，我们探索了 partisan 选择性加工在 LLM 辅助内容分析（LACA）中的识别方式。研究结果表明，当处理政治上一致的内容时，partisan 人格 LLM 更表现出强烈的意识形态偏见。此外，相同党派人格之间的编码者可靠性高于跨党派配对之间的可靠性。这种方法增强了对 LLM 输出的精细理解，促进了基于 AI 的社会科学的研究质量，使实际影响的模拟成为可能。 

---
# MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies 

**Title (ZH)**: MorphBPE：一种考虑形态学特性的分词器，旨在跨形态学高效培训语言模型 

**Authors**: Ehsaneddin Asgari, Yassine El Kheir, Mohammad Ali Sadraei Javaheri  

**Link**: [PDF](https://arxiv.org/pdf/2502.00894)  

**Abstract**: Tokenization is fundamental to Natural Language Processing (NLP), directly impacting model efficiency and linguistic fidelity. While Byte Pair Encoding (BPE) is widely used in Large Language Models (LLMs), it often disregards morpheme boundaries, leading to suboptimal segmentation, particularly in morphologically rich languages. We introduce MorphBPE, a morphology-aware extension of BPE that integrates linguistic structure into subword tokenization while preserving statistical efficiency. Additionally, we propose two morphology-based evaluation metrics: (i) Morphological Consistency F1-Score, which quantifies the consistency between morpheme sharing and token sharing, contributing to LLM training convergence, and (ii) Morphological Edit Distance, which measures alignment between morphemes and tokens concerning interpretability. Experiments on English, Russian, Hungarian, and Arabic across 300M and 1B parameter LLMs demonstrate that MorphBPE consistently reduces cross-entropy loss, accelerates convergence, and improves morphological alignment scores. Fully compatible with existing LLM pipelines, MorphBPE requires minimal modifications for integration. The MorphBPE codebase and tokenizer playground will be available at: this https URL and this https URL 

**Abstract (ZH)**: 分词是自然语言处理（NLP）的基础，直接关系到模型的效率和语言的一致性。尽管字节对编码（BPE）在大型语言模型（LLMs）中得到了广泛应用，但它往往忽略了形态学边界，导致分词不理想，特别是在形态学丰富语言中尤为明显。我们提出了 MorphBPE，这是一种基于形态学的 BPE 扩展，能够在保持统计效率的同时将语言结构整合到子词分词中。此外，我们还提出了两种基于形态学的评价指标：（i）形态一致性 F1 分数，它量化了形态共现和分词共现之间的一致性，有助于 LLM 训练收敛；（ii）形态编辑距离，它衡量了形态学和分词在可解释性方面的对齐程度。针对英、俄、匈、阿四种语言的 300M 和 1B 参数量级的 LLM 进行的实验表明，MorphBPE 一再减少了交叉熵损失、加速了收敛，并提高了形态学对齐得分。MorphBPE 完全兼容现有的 LLM 流水线，集成只需进行少量修改。MorphBPE 代码库和分词器游乐场将在以下链接提供：此 https URL 和此 https URL。 

---
# Paper Copilot: The Artificial Intelligence and Machine Learning Community Should Adopt a More Transparent and Regulated Peer Review Process 

**Title (ZH)**: 论文协作者：人工智能与机器学习社区应采用更为透明和规范的同行评审流程 

**Authors**: Jing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00874)  

**Abstract**: The rapid growth of submissions to top-tier Artificial Intelligence (AI) and Machine Learning (ML) conferences has prompted many venues to transition from closed to open review platforms. Some have fully embraced open peer reviews, allowing public visibility throughout the process, while others adopt hybrid approaches, such as releasing reviews only after final decisions or keeping reviews private despite using open peer review systems. In this work, we analyze the strengths and limitations of these models, highlighting the growing community interest in transparent peer review. To support this discussion, we examine insights from Paper Copilot, a website launched two years ago to aggregate and analyze AI / ML conference data while engaging a global audience. The site has attracted over 200,000 early-career researchers, particularly those aged 18-34 from 177 countries, many of whom are actively engaged in the peer review process. Drawing on our findings, this position paper advocates for a more transparent, open, and well-regulated peer review aiming to foster greater community involvement and propel advancements in the field. 

**Abstract (ZH)**: 顶级人工智能（AI）和机器学习（ML）会议投稿的迅速增长已经促使许多会议转向开放审稿平台。一些会议已经完全拥抱公开审稿，使整个过程具有公共可见性，而另一些则采用混合方式，例如仅在最终决定之后发布审稿内容，或者虽然使用开放审稿系统但仍然保持审稿内容的私密性。在本文中，我们分析了这些模型的优势和限制，并强调了开放透明审稿日益增长的社区兴趣。为了支持这一讨论，我们研究了Paper Copilot网站提供的见解，该网站两年前成立，旨在聚合和分析AI/ML会议数据，并吸引全球受众。该网站吸引了超过200,000名早期职业生涯的研究人员，特别是18-34岁的177个国家的研究人员，其中许多人积极参与审稿过程。根据我们的发现，本文倡导一种更加透明、开放且规范的审稿模式，旨在促进社区参与并推动该领域的发展。 

---
# FedHPD: Heterogeneous Federated Reinforcement Learning via Policy Distillation 

**Title (ZH)**: FedHPD：基于策略蒸馏的异构联邦强化学习 

**Authors**: Wenzheng Jiang, Ji Wang, Xiongtao Zhang, Weidong Bao, Cheston Tan, Flint Xiaofeng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00870)  

**Abstract**: Federated Reinforcement Learning (FedRL) improves sample efficiency while preserving privacy; however, most existing studies assume homogeneous agents, limiting its applicability in real-world scenarios. This paper investigates FedRL in black-box settings with heterogeneous agents, where each agent employs distinct policy networks and training configurations without disclosing their internal details. Knowledge Distillation (KD) is a promising method for facilitating knowledge sharing among heterogeneous models, but it faces challenges related to the scarcity of public datasets and limitations in knowledge representation when applied to FedRL. To address these challenges, we propose Federated Heterogeneous Policy Distillation (FedHPD), which solves the problem of heterogeneous FedRL by utilizing action probability distributions as a medium for knowledge sharing. We provide a theoretical analysis of FedHPD's convergence under standard assumptions. Extensive experiments corroborate that FedHPD shows significant improvements across various reinforcement learning benchmark tasks, further validating our theoretical findings. Moreover, additional experiments demonstrate that FedHPD operates effectively without the need for an elaborate selection of public datasets. 

**Abstract (ZH)**: 联邦强化学习（FedRL）能够在保护隐私的同时提高样本效率；然而，目前大多数现有研究假设所有代理是同质化的，这限制了其在实际场景中的应用。本文探讨了在异构代理的黑盒环境中应用FedRL，其中每个代理使用不同的策略网络和训练配置，而不披露内部细节。知识蒸馏（KD）是一种促进异构模型之间知识共享的有前景的方法，但在应用于FedRL时，面临着公共数据集稀缺和知识表示方面的局限性挑战。为了解决这些挑战，我们提出了联邦异构策略蒸馏（FedHPD），通过使用动作概率分布作为知识共享的媒介来解决异构FedRL的问题。我们对FedHPD在标准假设下的收敛性进行了理论分析。大量实验结果表明，FedHPD在各种强化学习基准任务中表现出显著的改进，进一步验证了我们的理论发现。此外，额外的实验表明，FedHPD可以在不需要精心选择公共数据集的情况下有效运行。 

---
# Predicting potentially unfair clauses in Chilean terms of services with natural language processing 

**Title (ZH)**: 使用自然语言处理预测智利服务条款中的潜在不公平条款 

**Authors**: Christoffer Loeffler, Andrea Martínez Freile, Tomás Rey Pizarro  

**Link**: [PDF](https://arxiv.org/pdf/2502.00865)  

**Abstract**: This study addresses the growing concern of information asymmetry in consumer contracts, exacerbated by the proliferation of online services with complex Terms of Service that are rarely even read. Even though research on automatic analysis methods is conducted, the problem is aggravated by the general focus on English-language Machine Learning approaches and on major jurisdictions, such as the European Union. We introduce a new methodology and a substantial dataset addressing this gap. We propose a novel annotation scheme with four categories and a total of 20 classes, and apply it on 50 online Terms of Service used in Chile. Our evaluation of transformer-based models highlights how factors like language- and/or domain-specific pre-training, few-shot sample size, and model architecture affect the detection and classification of potentially abusive clauses. Results show a large variability in performance for the different tasks and models, with the highest macro-F1 scores for the detection task ranging from 79% to 89% and micro-F1 scores up to 96%, while macro-F1 scores for the classification task range from 60% to 70% and micro-F1 scores from 64% to 80%. Notably, this is the first Spanish-language multi-label classification dataset for legal clauses, applying Chilean law and offering a comprehensive evaluation of Spanish-language models in the legal domain. Our work lays the ground for future research in method development for rarely considered legal analysis and potentially leads to practical applications to support consumers in Chile and Latin America as a whole. 

**Abstract (ZH)**: 本研究着眼于消费者合同中日益严重的信息不对称问题，这种问题因复杂的服务条款的广泛使用而加剧，而这些条款甚至很少有人阅读。尽管已经开展了自动分析方法的研究，但问题因普遍关注英语语言的机器学习方法和欧盟等主要司法管辖区而加剧。我们引入了一种新的方法论和大量的数据集来填补这一空白。我们提出了一种新的标注方案，包含四个类别和总共20个子类别，并将其应用于智利使用的50份在线服务条款。对变换器模型的评估揭示了诸如语言和/或领域特定预训练、少量样本数量和模型架构等因素如何影响潜在滥用条款的检测和分类。结果显示，在不同任务和模型中的性能差异很大，检测任务的最大宏F1分数从79%到89%，微F1分数最高可达96%，而分类任务的最大宏F1分数从60%到70%，微F1分数从64%到80%。值得注意的是，这是首次针对智利法律条款的多标签分类数据集，使用智利法律，提供了对西班牙语法律领域模型的全面评估。我们的工作为进一步研究罕见考虑的法律分析方法奠定了基础，并可能为智利乃至拉丁美洲的消费者提供实际应用的支持。 

---
# Dual Alignment Maximin Optimization for Offline Model-based RL 

**Title (ZH)**: 基于模型的离线强化学习中的双对齐最大化最小化优化 

**Authors**: Chi Zhou, Wang Luo, Haoran Li, Congying Han, Tiande Guo, Zicheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00850)  

**Abstract**: Offline reinforcement learning agents face significant deployment challenges due to the synthetic-to-real distribution mismatch. While most prior research has focused on improving the fidelity of synthetic sampling and incorporating off-policy mechanisms, the directly integrated paradigm often fails to ensure consistent policy behavior in biased models and underlying environmental dynamics, which inherently arise from discrepancies between behavior and learning policies. In this paper, we first shift the focus from model reliability to policy discrepancies while optimizing for expected returns, and then self-consistently incorporate synthetic data, deriving a novel actor-critic paradigm, Dual Alignment Maximin Optimization (DAMO). It is a unified framework to ensure both model-environment policy consistency and synthetic and offline data compatibility. The inner minimization performs dual conservative value estimation, aligning policies and trajectories to avoid out-of-distribution states and actions, while the outer maximization ensures that policy improvements remain consistent with inner value estimates. Empirical evaluations demonstrate that DAMO effectively ensures model and policy alignments, achieving competitive performance across diverse benchmark tasks. 

**Abstract (ZH)**: 由于合成数据与现实数据之间的分布不匹配，离线强化学习代理在部署时面临着重大挑战。虽然以往多数研究集中在提高合成样本的真实性和引入离策机制上，但直接集成的方法往往无法确保在偏向模型和潜在环境动态中的一致性策略行为，这种偏见来源于行为策略与学习策略之间的固有差异。本文中，我们首先将焦点从模型可靠性转移到策略差异性，同时最大化期望回报，然后自洽地引入合成数据，提出了一种新颖的Actor-Critic框架，Double Alignment Maximin Optimization (DAMO)。这是一种统一框架，旨在确保模型-环境策略一致性以及合成数据与离线数据的兼容性。内部最小化操作执行双重保守值估计，将策略和轨迹对齐以避免出现分布外状态和行动，而外部最大化操作确保策略改进与内部值估计一致。实证评估表明，DAMO 有效地确保了模型和策略的一致性，在多种基准任务中实现了具有竞争力的表现。 

---
# SecPE: Secure Prompt Ensembling for Private and Robust Large Language Models 

**Title (ZH)**: SecPE：面向私有和稳健大型语言模型的安全提示集成 

**Authors**: Jiawen Zhang, Kejia Chen, Zunlei Feng, Jian Lou, Mingli Song, Jian Liu, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00847)  

**Abstract**: With the growing popularity of LLMs among the general public users, privacy-preserving and adversarial robustness have become two pressing demands for LLM-based services, which have largely been pursued separately but rarely jointly. In this paper, to the best of our knowledge, we are among the first attempts towards robust and private LLM inference by tightly integrating two disconnected fields: private inference and prompt ensembling. The former protects users' privacy by encrypting inference data transmitted and processed by LLMs, while the latter enhances adversarial robustness by yielding an aggregated output from multiple prompted LLM responses. Although widely recognized as effective individually, private inference for prompt ensembling together entails new challenges that render the naive combination of existing techniques inefficient. To overcome the hurdles, we propose SecPE, which designs efficient fully homomorphic encryption (FHE) counterparts for the core algorithmic building blocks of prompt ensembling. We conduct extensive experiments on 8 tasks to evaluate the accuracy, robustness, and efficiency of SecPE. The results show that SecPE maintains high clean accuracy and offers better robustness at the expense of merely $2.5\%$ efficiency overhead compared to baseline private inference methods, indicating a satisfactory ``accuracy-robustness-efficiency'' tradeoff. For the efficiency of the encrypted Argmax operation that incurs major slowdown for prompt ensembling, SecPE is 35.4x faster than the state-of-the-art peers, which can be of independent interest beyond this work. 

**Abstract (ZH)**: 随着大型语言模型（LLM）在普通用户中的日益普及，隐私保护和对抗鲁棒性已成为LLM服务领域的两大紧迫需求。虽然这两方面通常会分别追求，但很少同时进行。在这项工作中，据我们所知，我们是最早尝试通过紧密整合两个分离的领域——隐私推理和提示集合来实现鲁棒性和隐私保护的尝试。前者通过加密LLM传输和处理的推理数据来保护用户隐私，而后者通过生成多个提示LLM响应的聚合输出来增强对抗鲁棒性。尽管这些方法在各自领域中被广泛认为是有效的，但将它们结合起来用于提示集合却带来了新的挑战，导致现有技术的简单组合变得低效。为了克服这些障碍，我们提出了SecPE方法，其设计了高效的全同态加密（FHE）版本，以支持提示集合的核心算法构建模块。我们对8个任务进行了广泛的实验，以评估SecPE的准确度、鲁棒性和效率。实验结果显示，SecPE保持了高准确度，并在仅比基线隐私推理方法效率有2.5%的轻微下降的情况下提供了更好的鲁棒性，表明实现了一个令人满意的“准确度-鲁棒性-效率”权衡。对于加密Argmax操作导致的提示集合中的重大速度减慢，SecPE 比最先进的同类技术快35.4倍，这具有独立的研究兴趣，超越了本研究本身。 

---
# Activation Approximations Can Incur Safety Vulnerabilities Even in Aligned LLMs: Comprehensive Analysis and Defense 

**Title (ZH)**: 即使是目标一致的大语言模型，激活近似也可能引入安全漏洞：全面分析与防御 

**Authors**: Jiawen Zhang, Kejia Chen, Lipeng He, Jian Lou, Dan Li, Zunlei Feng, Mingli Song, Jian Liu, Kui Ren, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00840)  

**Abstract**: Large Language Models (LLMs) have showcased remarkable capabilities across various domains. Accompanying the evolving capabilities and expanding deployment scenarios of LLMs, their deployment challenges escalate due to their sheer scale and the advanced yet complex activation designs prevalent in notable model series, such as Llama, Gemma, and Mistral. These challenges have become particularly pronounced in resource-constrained deployment scenarios, where mitigating inference efficiency bottlenecks is imperative. Among various recent efforts, activation approximation has emerged as a promising avenue for pursuing inference efficiency, sometimes considered indispensable in applications such as private inference. Despite achieving substantial speedups with minimal impact on utility, even appearing sound and practical for real-world deployment, the safety implications of activation approximations remain unclear. In this work, we fill this critical gap in LLM safety by conducting the first systematic safety evaluation of activation approximations. Our safety vetting spans seven sota techniques across three popular categories, revealing consistent safety degradation across ten safety-aligned LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各个领域展现了卓越的能力。伴随LLMs能力的演进和部署场景的扩展，由于其庞大的规模和先进的复杂激活设计，其部署挑战也随之加剧，这些挑战在资源受限的部署场景中尤为突出，需要解决推理效率瓶颈的问题。在各种近期的努力中，激活近似已经逐渐成为提高推理效率的一个有前景的方法，甚至在某些应用，如私人推理中被认为是必不可少的。尽管激活近似方法能实现显著的速度提升，且对效用的影响相对较小，甚至在实际部署中显得非常可行和可靠，但在安全方面的潜在影响仍然不明确。在这项工作中，我们填补了LLM安全研究中的这一关键空白，进行了第一次系统性的激活近似安全评估。我们的安全性评估覆盖了三种流行类别中的七个最佳技术，揭示了在十种安全导向的LLM上持续性安全性能下降的趋势。 

---
# Explainability in Practice: A Survey of Explainable NLP Across Various Domains 

**Title (ZH)**: 实践中的可解释性：跨不同领域可解释自然语言处理的综述 

**Authors**: Hadi Mohammadi, Ayoub Bagheri, Anastasia Giachanou, Daniel L. Oberski  

**Link**: [PDF](https://arxiv.org/pdf/2502.00837)  

**Abstract**: Natural Language Processing (NLP) has become a cornerstone in many critical sectors, including healthcare, finance, and customer relationship management. This is especially true with the development and use of advanced models such as GPT-based architectures and BERT, which are widely used in decision-making processes. However, the black-box nature of these advanced NLP models has created an urgent need for transparency and explainability. This review explores explainable NLP (XNLP) with a focus on its practical deployment and real-world applications, examining its implementation and the challenges faced in domain-specific contexts. The paper underscores the importance of explainability in NLP and provides a comprehensive perspective on how XNLP can be designed to meet the unique demands of various sectors, from healthcare's need for clear insights to finance's emphasis on fraud detection and risk assessment. Additionally, this review aims to bridge the knowledge gap in XNLP literature by offering a domain-specific exploration and discussing underrepresented areas such as real-world applicability, metric evaluation, and the role of human interaction in model assessment. The paper concludes by suggesting future research directions that could enhance the understanding and broader application of XNLP. 

**Abstract (ZH)**: 自然语言处理（NLP）已成为包括医疗保健、金融和客户关系管理在内的许多关键领域的基石。特别是在先进模型如基于GPT的架构和BERT的发展和应用中，NLP在决策过程中得到了广泛应用。然而，这些先进NLP模型的黑盒性质迫切需要透明性和解释性。本文回顾了可解释的自然语言处理（XNLP），重点关注其实用部署和实际应用，探讨其在特定领域中的实现及其所面临的挑战。文章强调了NLP中的解释性的重要性，并提供了一个全面的视角，解释了如何设计XNLP以满足不同领域特有的要求，从医疗保健对清晰洞察的需求到金融领域对欺诈检测和风险评估的重视。此外，本文旨在通过提供特定领域的探索并讨论未充分研究的领域（如实际应用、指标评估以及人类交互在模型评估中的作用），弥合XNLP文献中的知识缺口。最后，文章建议未来的研究方向，以增强对XNLP的理解及其更广泛的应用。 

---
# Decision-informed Neural Networks with Large Language Model Integration for Portfolio Optimization 

**Title (ZH)**: 带有大型语言模型集成的决策导向神经网络在组合优化中的应用 

**Authors**: Yoontae Hwang, Yaxuan Kong, Stefan Zohren, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.00828)  

**Abstract**: This paper addresses the critical disconnect between prediction and decision quality in portfolio optimization by integrating Large Language Models (LLMs) with decision-focused learning. We demonstrate both theoretically and empirically that minimizing the prediction error alone leads to suboptimal portfolio decisions. We aim to exploit the representational power of LLMs for investment decisions. An attention mechanism processes asset relationships, temporal dependencies, and macro variables, which are then directly integrated into a portfolio optimization layer. This enables the model to capture complex market dynamics and align predictions with the decision objectives. Extensive experiments on S\&P100 and DOW30 datasets show that our model consistently outperforms state-of-the-art deep learning models. In addition, gradient-based analyses show that our model prioritizes the assets most crucial to decision making, thus mitigating the effects of prediction errors on portfolio performance. These findings underscore the value of integrating decision objectives into predictions for more robust and context-aware portfolio management. 

**Abstract (ZH)**: 本文通过将大型语言模型（LLMs）与决策导向学习相结合，解决了投资组合优化中预测与决策质量之间的关键断层问题。我们从理论和实证两方面证明，单独最小化预测误差会导致投资组合决策的次优结果。我们的目标是利用LLMs在投资决策中的表示能力。通过注意机制处理资产关系、时间依赖性和宏观经济变量，然后将这些信息直接整合到投资组合优化层中，从而使模型能够捕捉复杂市场动态，并使预测与决策目标相一致。在S&P100和DOW30数据集上的广泛实验表明，我们的模型在性能上持续优于最先进的深度学习模型。此外，基于梯度的分析表明，我们的模型优先考虑对决策最重要的资产，从而减轻了预测误差对投资组合业绩的影响。这些发现强调了将决策目标集成到预测中对于更稳健和情境感知的投资组合管理的价值。 

---
# Fisher-Guided Selective Forgetting: Mitigating The Primacy Bias in Deep Reinforcement Learning 

**Title (ZH)**: Fishers引导的选择性遗忘：减轻深度强化学习中的首因效应偏差 

**Authors**: Massimiliano Falzari, Matthia Sabatelli  

**Link**: [PDF](https://arxiv.org/pdf/2502.00802)  

**Abstract**: Deep Reinforcement Learning (DRL) systems often tend to overfit to early experiences, a phenomenon known as the primacy bias (PB). This bias can severely hinder learning efficiency and final performance, particularly in complex environments. This paper presents a comprehensive investigation of PB through the lens of the Fisher Information Matrix (FIM). We develop a framework characterizing PB through distinct patterns in the FIM trace, identifying critical memorization and reorganization phases during learning. Building on this understanding, we propose Fisher-Guided Selective Forgetting (FGSF), a novel method that leverages the geometric structure of the parameter space to selectively modify network weights, preventing early experiences from dominating the learning process. Empirical results across DeepMind Control Suite (DMC) environments show that FGSF consistently outperforms baselines, particularly in complex tasks. We analyze the different impacts of PB on actor and critic networks, the role of replay ratios in exacerbating the effect, and the effectiveness of even simple noise injection methods. Our findings provide a deeper understanding of PB and practical mitigation strategies, offering a FIM-based geometric perspective for advancing DRL. 

**Abstract (ZH)**: 深度强化学习（DRL）系统往往倾向于过度拟合早期经历，这一现象被称为首因偏差（Primacy Bias, PB）。PB现象会严重影响学习效率和最终表现，特别是在复杂环境中尤为显著。本文通过费舍尔信息矩阵（FIM）的角度对PB进行了全面的研究。我们开发了一个框架，通过FIM迹中的不同模式来表征PB，并识别出学习过程中关键的记忆重组织阶段。在此基础上，我们提出了费舍尔指导的选择性遗忘（Fisher-Guided Selective Forgetting, FGSF）方法，该方法利用网络参数空间的几何结构选择性地修改权重，防止早期经历主导学习过程。在DeepMind控制套件（DMC）环境中进行的实证研究表明，FGSF在复杂任务中通常优于基准方法。我们分析了PB对演员和评论家网络的不同影响、回放比例在加剧这一效应中的作用，以及简单噪声注入方法的有效性。本文的研究结果为理解PB提供了更深入的认识，并提出了实用的缓解策略，提供了一种基于FIM的几何视角，以促进强化学习的发展。 

---
# Environment-Driven Online LiDAR-Camera Extrinsic Calibration 

**Title (ZH)**: 环境驱动的在线LiDAR-相机外参标定 

**Authors**: Zhiwei Huang, Jiaqi Li, Ping Zhong, Rui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00801)  

**Abstract**: LiDAR-camera extrinsic calibration (LCEC) is the core for data fusion in computer vision. Existing methods typically rely on customized calibration targets or fixed scene types, lacking the flexibility to handle variations in sensor data and environmental contexts. This paper introduces EdO-LCEC, the first environment-driven, online calibration approach that achieves human-like adaptability. Inspired by the human perceptual system, EdO-LCEC incorporates a generalizable scene discriminator to actively interpret environmental conditions, creating multiple virtual cameras that capture detailed spatial and textural information. To overcome cross-modal feature matching challenges between LiDAR and camera, we propose dual-path correspondence matching (DPCM), which leverages both structural and textural consistency to achieve reliable 3D-2D correspondences. Our approach formulates the calibration process as a spatial-temporal joint optimization problem, utilizing global constraints from multiple views and scenes to improve accuracy, particularly in sparse or partially overlapping sensor views. Extensive experiments on real-world datasets demonstrate that EdO-LCEC achieves state-of-the-art performance, providing reliable and precise calibration across diverse, challenging environments. 

**Abstract (ZH)**: 激光雷达-摄像头外参标定（LCEC）是计算机视觉中数据融合的核心。现有方法通常依赖于定制化的校准目标或固定场景类型，缺乏应对传感器数据和环境变化的灵活性。本文介绍了EdO-LCEC，这是一种基于环境驱动的在线校准方法，实现了类似人类的适应性。受人类感知系统启发，EdO-LCEC 结合了一个通用的场景鉴别器，主动解释环境条件，生成多个虚拟摄像头以捕捉详细的空间和纹理信息。为了解决激光雷达和摄像头之间跨模态特征匹配的挑战，我们提出了一种双路径对应匹配（DPCM）方法，利用结构和纹理一致性来实现可靠的三维-二维对应关系。我们的方法将校准过程表述为时空联合优化问题，利用多视图和场景的全局约束以提高精度，特别是在稀疏或部分重叠的传感器视图中。在真实世界的数据集上的广泛实验表明，EdO-LCEC 达到了最先进的性能，能够在多种具有挑战性的环境中提供可靠和精确的校准。 

---
# Role of Mixup in Topological Persistence Based Knowledge Distillation for Wearable Sensor Data 

**Title (ZH)**: 基于拓扑持久同胚的知识蒸馏中 Mixup 的作用研究 

**Authors**: Eun Som Jeon, Hongjun Choi, Matthew P. Buman, Pavan Turaga  

**Link**: [PDF](https://arxiv.org/pdf/2502.00779)  

**Abstract**: The analysis of wearable sensor data has enabled many successes in several applications. To represent the high-sampling rate time-series with sufficient detail, the use of topological data analysis (TDA) has been considered, and it is found that TDA can complement other time-series features. Nonetheless, due to the large time consumption and high computational resource requirements of extracting topological features through TDA, it is difficult to deploy topological knowledge in various applications. To tackle this problem, knowledge distillation (KD) can be adopted, which is a technique facilitating model compression and transfer learning to generate a smaller model by transferring knowledge from a larger network. By leveraging multiple teachers in KD, both time-series and topological features can be transferred, and finally, a superior student using only time-series data is distilled. On the other hand, mixup has been popularly used as a robust data augmentation technique to enhance model performance during training. Mixup and KD employ similar learning strategies. In KD, the student model learns from the smoothed distribution generated by the teacher model, while mixup creates smoothed labels by blending two labels. Hence, this common smoothness serves as the connecting link that establishes a connection between these two methods. In this paper, we analyze the role of mixup in KD with time-series as well as topological persistence, employing multiple teachers. We present a comprehensive analysis of various methods in KD and mixup on wearable sensor data. 

**Abstract (ZH)**: 穿戴传感器数据的分析已经在多个应用领域取得了许多成功。为了以足够的细节表示高采样率的时间序列，人们考虑使用拓扑数据分析（TDA）。研究发现，TDA 可以补充其他时间序列特征。然而，由于通过 TDA 提取拓扑特征所需的时间消耗大且对计算资源的需求高，难以在各种应用中部署拓扑知识。为了解决这一问题，可以采用知识蒸馏（KD）技术，该技术通过从大型网络转移知识来生成较小的模型，从而支持模型压缩和迁移学习。通过在 KD 中使用多个教师，可以同时转移时间序列特征和拓扑特征，最终得到仅使用时间序列数据的学生模型。另一方面，mixup 是一种常用的数据增强技术，用于提高模型在训练过程中的性能。mixup 和 KD 都采用了类似的训练策略。在 KD 中，学生模型从教师模型生成的平滑分布中学习，而 mixup 则通过混合两个标签生成平滑标签。因此，这种共同的平滑性充当了连接这两种方法的纽带。在本文中，我们分析了在 KD 中使用 mixup 以及时间序列和拓扑持久性的角色，采用多个教师进行分析。我们对 KD 和 mixup 在穿戴传感器数据中的各种方法进行了全面分析。 

---
# Learning-Based TSP-Solvers Tend to Be Overly Greedy 

**Title (ZH)**: 基于学习的TSP求解器往往会过分贪婪 

**Authors**: Xiayang Li, Shihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00767)  

**Abstract**: Deep learning has shown significant potential in solving combinatorial optimization problems such as the Euclidean traveling salesman problem (TSP). However, most training and test instances for existing TSP algorithms are generated randomly from specific distributions like uniform distribution. This has led to a lack of analysis and understanding of the performance of deep learning algorithms in out-of-distribution (OOD) generalization scenarios, which has a close relationship with the worst-case performance in the combinatorial optimization field. For data-driven algorithms, the statistical properties of randomly generated datasets are critical. This study constructs a statistical measure called nearest-neighbor density to verify the asymptotic properties of randomly generated datasets and reveal the greedy behavior of learning-based solvers, i.e., always choosing the nearest neighbor nodes to construct the solution path. Based on this statistical measure, we develop interpretable data augmentation methods that rely on distribution shifts or instance perturbations and validate that the performance of the learning-based solvers degenerates much on such augmented data. Moreover, fine-tuning learning-based solvers with augmented data further enhances their generalization abilities. In short, we decipher the limitations of learning-based TSP solvers tending to be overly greedy, which may have profound implications for AI-empowered combinatorial optimization solvers. 

**Abstract (ZH)**: 深度学习在解决欧几里得旅行商问题（TSP）等组合优化问题方面展示了显著的潜力。然而，现有的TSP算法的训练和测试实例大多是从特定分布（例如均匀分布）中随机生成的。这导致了对深度学习算法在离分布外（OOD）泛化场景中的性能分析和理解不足，这些场景与组合优化领域的最坏情况性能密切相关。对于数据驱动的算法而言，随机生成数据集的统计特性至关重要。本研究构建了一种统计度量——最近邻密度，用于验证随机生成数据集的渐进行为，并揭示基于学习的求解器的贪婪行为，即始终选择最近邻节点来构建解路径。基于这一统计度量，我们开发了依赖于分布转换或实例扰动的可解释数据增强方法，并验证了基于学习的求解器在增强数据上的性能显著退化。此外，使用增强数据进一步微调基于学习的求解器可以进一步提升其泛化能力。简言之，我们揭示了基于学习的TSP求解器倾向于过度贪婪的局限性，这可能对由人工智能赋能的组合优化求解器具有深远的影响。 

---
# AgentBreeder: Mitigating the AI Safety Impact of Multi-Agent Scaffolds 

**Title (ZH)**: AgentBreeder: 缓解多智能体架构对AI安全影响的方法 

**Authors**: J Rosser, Jakob Nicolaus Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2502.00757)  

**Abstract**: Scaffolding Large Language Models (LLMs) into multi-agent systems often improves performance on complex tasks, but the safety impact of such scaffolds has not been as thoroughly explored. In this paper, we introduce AGENTBREEDER a framework for multi-objective evolutionary search over scaffolds. Our REDAGENTBREEDER evolves scaffolds towards jailbreaking the base LLM while achieving high task success, while BLUEAGENTBREEDER instead aims to combine safety with task reward. We evaluate the systems discovered by the different instances of AGENTBREEDER and popular baselines using widely recognized reasoning, mathematics, and safety benchmarks. Our work highlights and mitigates the safety risks due to multi-agent scaffolding. 

**Abstract (ZH)**: 将大型语言模型（LLMs）构建到多代理系统中通常可以提高完成复杂任务的性能，但此类支撑结构的安全影响尚未得到充分探讨。本文介绍了AGENTBREEDER框架，该框架用于多目标进化搜索。我们的REDAGENTBREEDER朝破解基础LLM的方向进化支撑结构，同时实现高任务成功率，而BLUEAGENTBREEDER则旨在结合安全性和任务奖励。我们使用广泛认可的推理、数学和安全基准对不同实例的AGENTBREEDER发现的系统和流行基准进行了评估。我们的工作突显了多代理支撑结构带来的安全风险，并提出了相应的缓解措施。 

---
# Universal Post-Processing Networks for Joint Optimization of Modules in Task-Oriented Dialogue Systems 

**Title (ZH)**: 面向任务的对话系统中模块联合优化的通用后处理网络 

**Authors**: Atsumoto Ohashi, Ryuichiro Higashinaka  

**Link**: [PDF](https://arxiv.org/pdf/2502.00747)  

**Abstract**: Post-processing networks (PPNs) are components that modify the outputs of arbitrary modules in task-oriented dialogue systems and are optimized using reinforcement learning (RL) to improve the overall task completion capability of the system. However, previous PPN-based approaches have been limited to handling only a subset of modules within a system, which poses a significant limitation in improving the system performance. In this study, we propose a joint optimization method for post-processing the outputs of all modules using universal post-processing networks (UniPPNs), which are language-model-based networks that can modify the outputs of arbitrary modules in a system as a sequence-transformation task. Moreover, our RL algorithm, which employs a module-level Markov decision process, enables fine-grained value and advantage estimation for each module, thereby stabilizing joint learning for post-processing the outputs of all modules. Through both simulation-based and human evaluation experiments using the MultiWOZ dataset, we demonstrated that UniPPN outperforms conventional PPNs in the task completion capability of task-oriented dialogue systems. 

**Abstract (ZH)**: 后处理网络（PPNs）是任务导向对话系统中用于修改任意模块输出的组件，并通过强化学习（RL）进行优化，以提高系统的整体任务完成能力。然而，先前的PPN基方法仅限于处理系统中的一部分模块，这在提高系统性能方面存在显著限制。本研究提出了一种联合优化方法，使用基于语言模型的通用后处理网络（UniPPNs）对系统中所有模块的输出进行后处理，UniPPNs可以将系统中任意模块的输出视作序列转换任务进行修改。此外，我们采用基于模块级马尔可夫决策过程的RL算法，能够为每个模块提供精细的价值和优势估计，从而稳定所有模块输出后处理的联合学习过程。通过使用MultiWOZ数据集的仿真评估和人类评估实验，我们展示了UniPPN在任务导向对话系统任务完成能力方面优于传统PPNs。 

---
# From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs 

**Title (ZH)**: 从遵守规则到利用：针对多模态LLM的 Jailbreak 对话攻击 

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.00735)  

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the frontier multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. To better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flank Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios. These findings highlight both the potency of prompt-based obfuscation in voice-enabled contexts and the limitations of current LLMs' moderation safeguards and the urgent need for advanced defense strategies to address the challenges posed by evolving, context-rich attacks. 

**Abstract (ZH)**: 大型语言模型（LLMs）由于其处理不同类型输入数据（包括文本、音频、图像和视频）的能力不断增强，已在众多领域得到了广泛应用。尽管LLMs在理解和生成不同场景下的语境方面表现出色，但它们对基于提示的攻击易受攻击，这些攻击大多通过文本输入实施。本文介绍了针对多模态LLMs的第一种基于声音的越狱攻击，称为边缘攻击（Flanking Attack），该攻击可以同时处理多种类型的输入以应对多模态LLMs。我们的研究受近年来基于单一语言的声音驱动大型语言模型 advancements 启发，这些模型为LLMs引入了新的攻击面，超出了传统基于文本的漏洞。为了研究这些风险，我们考察了可以通过多种输入类型（如音频输入）访问的前沿多模态LLMs，并关注敌对提示如何绕过其防御机制。我们提出了一种新的策略，在这种策略中，禁止的提示由良性、叙述驱动的提示包围。该策略整合在边缘攻击中，旨在通过虚构的背景使交互情境人性化，并通过该背景执行攻击。为了更好地评估攻击性能，我们提出了一个半自动化的自我评估框架，用于检测违规行为。研究表明，边缘攻击能够操纵最先进的LLMs生成对齐不当和禁止的输出，在七个禁止场景中，攻击的成功率范围从0.67到0.93不等。这些发现突显了语音启用环境中基于提示混淆的效力，以及当前LLMs的监控保护措施的局限性，强调了亟需先进的防御策略以应对不断演变、富含上下文的攻击带来的挑战。 

---
# CycleGuardian: A Framework for Automatic RespiratorySound classification Based on Improved Deep clustering and Contrastive Learning 

**Title (ZH)**: CycleGuardian：一种基于改进的深度聚类和对比学习的自动呼吸音分类框架 

**Authors**: Yun Chu, Qiuhao Wang, Enze Zhou, Ling Fu, Qian Liu, Gang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00734)  

**Abstract**: Auscultation plays a pivotal role in early respiratory and pulmonary disease diagnosis. Despite the emergence of deep learning-based methods for automatic respiratory sound classification post-Covid-19, limited datasets impede performance enhancement. Distinguishing between normal and abnormal respiratory sounds poses challenges due to the coexistence of normal respiratory components and noise components in both types. Moreover, different abnormal respiratory sounds exhibit similar anomalous features, hindering their differentiation. Besides, existing state-of-the-art models suffer from excessive parameter size, impeding deployment on resource-constrained mobile platforms. To address these issues, we design a lightweight network CycleGuardian and propose a framework based on an improved deep clustering and contrastive learning. We first generate a hybrid spectrogram for feature diversity and grouping spectrograms to facilitating intermittent abnormal sound this http URL, CycleGuardian integrates a deep clustering module with a similarity-constrained clustering component to improve the ability to capture abnormal features and a contrastive learning module with group mixing for enhanced abnormal feature discernment. Multi-objective optimization enhances overall performance during training. In experiments we use the ICBHI2017 dataset, following the official split method and without any pre-trained weights, our method achieves Sp: 82.06 $\%$, Se: 44.47$\%$, and Score: 63.26$\%$ with a network model size of 38M, comparing to the current model, our method leads by nearly 7$\%$, achieving the current best performances. Additionally, we deploy the network on Android devices, showcasing a comprehensive intelligent respiratory sound auscultation system. 

**Abstract (ZH)**: 听诊在早期呼吸和肺部疾病诊断中发挥着重要作用。尽管在新冠疫情期间出现了基于深度学习的自动呼吸音分类方法，但由于数据集有限，阻碍了性能的提升。正常呼吸音和异常呼吸音之间的区分存在挑战，因为两种类型的呼吸音均包含正常的呼吸成分和噪音成分。此外，不同的异常呼吸音表现出相似的异常特征，这使得它们难以区分。另外，现有的先进模型存在参数过多的问题，这阻碍了其在资源受限的移动平台上的部署。为解决这些问题，我们设计了一个轻量级网络CycleGuardian，并提出了一种基于改进深度聚类和对比学习的框架。我们首先生成混合光谱图以增强特征多样性和组织光谱图来促进间歇性异常声音的识别。CycleGuardian集成了一个深度聚类模块和一个相似性约束聚类组件，以提高捕捉异常特征的能力，以及一个带有组混合同步的对比学习模块，以增强异常特征的区分能力。多目标优化在训练过程中提升整体性能。在实验中，我们使用ICBHI2017数据集，遵循官方划分方法且未使用任何预训练权重，我们的方法在模型大小为38M的情况下，达到Sp: 82.06%，Se: 44.47%，Score: 63.26%，相较于现有模型，我们的方法提升了约7%，获得了当前最佳性能。此外，我们还将网络部署在Android设备上，展示了全面的智能呼吸音听诊系统。 

---
# Learned Bayesian Cram\'er-Rao Bound for Unknown Measurement Models Using Score Neural Networks 

**Title (ZH)**: 使用评分神经网络学习未知测量模型的贝叶斯克劳德-拉奥下界 

**Authors**: Hai Victor Habi, Hagit Messer, Yoram Bresler  

**Link**: [PDF](https://arxiv.org/pdf/2502.00724)  

**Abstract**: The Bayesian Cramér-Rao bound (BCRB) is a crucial tool in signal processing for assessing the fundamental limitations of any estimation problem as well as benchmarking within a Bayesian frameworks. However, the BCRB cannot be computed without full knowledge of the prior and the measurement distributions. In this work, we propose a fully learned Bayesian Cramér-Rao bound (LBCRB) that learns both the prior and the measurement distributions. Specifically, we suggest two approaches to obtain the LBCRB: the Posterior Approach and the Measurement-Prior Approach. The Posterior Approach provides a simple method to obtain the LBCRB, whereas the Measurement-Prior Approach enables us to incorporate domain knowledge to improve the sample complexity and {interpretability}. To achieve this, we introduce a Physics-encoded score neural network which enables us to easily incorporate such domain knowledge into a neural network. We {study the learning} errors of the two suggested approaches theoretically, and validate them numerically. We demonstrate the two approaches on several signal processing examples, including a linear measurement problem with unknown mixing and Gaussian noise covariance matrices, frequency estimation, and quantized measurement. In addition, we test our approach on a nonlinear signal processing problem of frequency estimation with real-world underwater ambient noise. 

**Abstract (ZH)**: 贝叶斯克拉美-罗界线（BCRL）是信号处理中评估任何估计问题的基本限制以及在贝叶斯框架内进行基准测试的重要工具。然而，BCRL 的计算需要完整了解先验和测量分布。在这项工作中，我们提出了一种完全学习的贝叶斯克拉美-罗界线（LBCRL），该方法能够学习先验和测量分布。具体而言，我们提出了两种获得LBCRL的方法：后验方法和测量-先验方法。后验方法提供了一种简单的方法来获得LBCRL，而测量-先验方法则允许我们通过整合领域知识来提高样本复杂性和可解释性。为了实现这一点，我们引入了一种基于物理的得分神经网络，这使得我们可以轻松地将这些领域知识整合到神经网络中。我们从理论上研究了两种建议方法的学习误差，并通过数值验证进行了验证。我们分别在几个信号处理示例中展示了这两种方法，包括具有未知混叠和高斯噪声协方差矩阵的线性测量问题、频率估计以及量化测量。此外，我们还测试了基于真实海洋噪声的非线性信号处理中频率估计问题。 

---
# Registration-Enhanced Segmentation Method for Prostate Cancer in Ultrasound Images 

**Title (ZH)**: 超声图像中前列腺癌分割的注册增强方法 

**Authors**: Shengtian Sang, Hassan Jahanandish, Cynthia Xinran Li, Indrani Bhattachary, Jeong Hoon Lee, Lichun Zhang, Sulaiman Vesal, Pejman Ghanouni, Richard Fan, Geoffrey A. Sonn, Mirabela Rusu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00712)  

**Abstract**: Prostate cancer is a major cause of cancer-related deaths in men, where early detection greatly improves survival rates. Although MRI-TRUS fusion biopsy offers superior accuracy by combining MRI's detailed visualization with TRUS's real-time guidance, it is a complex and time-intensive procedure that relies heavily on manual annotations, leading to potential errors. To address these challenges, we propose a fully automatic MRI-TRUS fusion-based segmentation method that identifies prostate tumors directly in TRUS images without requiring manual annotations. Unlike traditional multimodal fusion approaches that rely on naive data concatenation, our method integrates a registration-segmentation framework to align and leverage spatial information between MRI and TRUS modalities. This alignment enhances segmentation accuracy and reduces reliance on manual effort. Our approach was validated on a dataset of 1,747 patients from Stanford Hospital, achieving an average Dice coefficient of 0.212, outperforming TRUS-only (0.117) and naive MRI-TRUS fusion (0.132) methods, with significant improvements (p $<$ 0.01). This framework demonstrates the potential for reducing the complexity of prostate cancer diagnosis and provides a flexible architecture applicable to other multimodal medical imaging tasks. 

**Abstract (ZH)**: 前列腺癌是男性癌症相关死亡的主要原因之一，早期检测可以显著提高生存率。虽然MRI-TRUS融合活检通过将MRI的详细成像与TRUS的实时引导结合起来，提供了更高的准确性，但该过程复杂且耗时，并且高度依赖手动注释，容易出现错误。为了解决这些挑战，我们提出了一种全自动的MRI-TRUS融合分割方法，可以直接在TRUS图像中标记前列腺肿瘤，无需手动注释。与传统基于简单数据拼接的多模态融合方法不同，我们的方法采用了一个注册-分割框架，用于在MRI和TRUS模态之间对齐和利用空间信息。这种对齐提高了分割的准确性，减少了对手动努力的依赖。我们通过斯坦福医院的1,747名患者的 dataset 进行验证，平均 Dice 系数为 0.212，优于仅TRUS（0.117）和简单MRI-TRUS融合（0.132）方法，并实现了显著改进（p < 0.01）。该框架展示了减少前列腺癌诊断复杂性的潜力，并提供了一个适用于其他多模态医学成像任务的灵活架构。 

---
# VIKSER: Visual Knowledge-Driven Self-Reinforcing Reasoning Framework 

**Title (ZH)**: VIKSER：视觉知识驱动的自强化推理框架 

**Authors**: Chunbai Zhang, Chao Wang, Yang Zhou, Yan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00711)  

**Abstract**: Visual reasoning refers to the task of solving questions about visual information. Current visual reasoning methods typically employ pre-trained vision-language model (VLM) strategies or deep neural network approaches. However, existing efforts are constrained by limited reasoning interpretability, while hindering by the phenomenon of underspecification in the question text. Additionally, the absence of fine-grained visual knowledge limits the precise understanding of subject behavior in visual reasoning tasks. To address these issues, we propose VIKSER (Visual Knowledge-Driven Self-Reinforcing Reasoning Framework). Specifically, VIKSER, trained using knowledge distilled from large language models, extracts fine-grained visual knowledge with the assistance of visual relationship detection techniques. Subsequently, VIKSER utilizes fine-grained visual knowledge to paraphrase the question with underspecification. Additionally, we design a novel prompting method called Chain-of-Evidence (CoE), which leverages the power of ``evidence for reasoning'' to endow VIKSER with interpretable reasoning capabilities. Meanwhile, the integration of self-reflection technology empowers VIKSER with the ability to learn and improve from its mistakes. Experiments conducted on widely used datasets demonstrate that VIKSER achieves new state-of-the-art (SOTA) results in relevant tasks. 

**Abstract (ZH)**: 视觉推理是指通过解决与视觉信息相关的问题来进行的一种任务。目前的视觉推理方法通常采用预训练的视觉-语言模型（VLM）策略或深度神经网络方法。然而，现有的努力受到推理可解释性的限制，同时由于问题文本中存在的模态性现象而受到阻碍。此外，缺乏精细的视觉知识限制了在视觉推理任务中对主题行为的精确理解。为了应对这些挑战，我们提出了一种名为VIKSER（视觉知识驱动的自我强化推理框架）的方法。具体而言，VIKSER通过从大型语言模型中提炼的知识进行训练，并借助视觉关系检测技术提取精细的视觉知识。随后，VIKSER利用精细的视觉知识对具有模态性的问题进行重述。此外，我们设计了一种新颖的提示方法，称为证据链（Chain-of-Evidence, CoE），该方法利用推理所需“证据”的力量，赋予VIKSER可解释的推理能力。同时，自我反思技术的集成使VIKSER能够从错误中学习和改进。在广泛使用的数据集上进行的实验表明，VIKSER在相关任务中取得了新的最佳性能（SOTA）结果。 

---
# PhiP-G: Physics-Guided Text-to-3D Compositional Scene Generation 

**Title (ZH)**: PhiP-G：物理引导的文本至三维组态场景生成 

**Authors**: Qixuan Li, Chao Wang, Zongjin He, Yan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00708)  

**Abstract**: Text-to-3D asset generation has achieved significant optimization under the supervision of 2D diffusion priors. However, when dealing with compositional scenes, existing methods encounter several challenges: 1). failure to ensure that composite scene layouts comply with physical laws; 2). difficulty in accurately capturing the assets and relationships described in complex scene descriptions; 3). limited autonomous asset generation capabilities among layout approaches leveraging large language models (LLMs). To avoid these compromises, we propose a novel framework for compositional scene generation, PhiP-G, which seamlessly integrates generation techniques with layout guidance based on a world model. Leveraging LLM-based agents, PhiP-G analyzes the complex scene description to generate a scene graph, and integrating a multimodal 2D generation agent and a 3D Gaussian generation method for targeted assets creation. For the stage of layout, PhiP-G employs a physical pool with adhesion capabilities and a visual supervision agent, forming a world model for layout prediction and planning. Extensive experiments demonstrate that PhiP-G significantly enhances the generation quality and physical rationality of the compositional scenes. Notably, PhiP-G attains state-of-the-art (SOTA) performance in CLIP scores, achieves parity with the leading methods in generation quality as measured by the T$^3$Bench, and improves efficiency by 24x. 

**Abstract (ZH)**: 在2D扩散先验的监督下，文本到3D资产生成已取得了显著优化。然而，在处理组合场景时，现有方法面临几个挑战：1）难以确保组合场景布局遵守物理定律；2）难以准确捕捉复杂场景描述中的资产及其关系；3）依赖大规模语言模型（LLMs）的布局方法在自主生成资产方面能力有限。为避免这些妥协，我们提出了一种名为PhiP-G的新型组合场景生成框架，该框架通过世界模型无缝结合生成技术和基于布局指导的技术。利用基于LLM的代理，PhiP-G分析复杂的场景描述以生成场景图，并结合多模态2D生成代理和3D高斯生成方法进行目标资产的创建。在布局阶段，PhiP-G采用具备黏附能力的物理池和视觉监督代理，形成世界模型用于布局预测和计划。大量实验表明，PhiP-G显著增强了组合场景的生成质量和物理合理性。值得注意的是，PhiP-G在CLIP评分中达到了最新的最佳性能（SOTA），在T$^3$Bench衡量的生成质量方面与领先方法持平，并将效率提升了24倍。 

---
# TMI-CLNet: Triple-Modal Interaction Network for Chronic Liver Disease Prognosis From Imaging, Clinical, and Radiomic Data Fusion 

**Title (ZH)**: TMI-CLNet：融合影像学、临床和影像组学数据的三模态交互网络用于慢性肝病预后研究 

**Authors**: Linglong Wu, Xuhao Shan, Ruiquan Ge, Ruoyu Liang, Chi Zhang, Yonghong Li, Ahmed Elazab, Huoling Luo, Yunbi Liu, Changmiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00695)  

**Abstract**: Chronic liver disease represents a significant health challenge worldwide and accurate prognostic evaluations are essential for personalized treatment plans. Recent evidence suggests that integrating multimodal data, such as computed tomography imaging, radiomic features, and clinical information, can provide more comprehensive prognostic information. However, modalities have an inherent heterogeneity, and incorporating additional modalities may exacerbate the challenges of heterogeneous data fusion. Moreover, existing multimodal fusion methods often struggle to adapt to richer medical modalities, making it difficult to capture inter-modal relationships. To overcome these limitations, We present the Triple-Modal Interaction Chronic Liver Network (TMI-CLNet). Specifically, we develop an Intra-Modality Aggregation module and a Triple-Modal Cross-Attention Fusion module, which are designed to eliminate intra-modality redundancy and extract cross-modal information, respectively. Furthermore, we design a Triple-Modal Feature Fusion loss function to align feature representations across modalities. Extensive experiments on the liver prognosis dataset demonstrate that our approach significantly outperforms existing state-of-the-art unimodal models and other multi-modal techniques. Our code is available at this https URL. 

**Abstract (ZH)**: 慢性肝病是全球性的重要健康挑战，精准的预后评估对于个性化治疗计划至关重要。近期研究表明，整合多模态数据，如计算机断层扫描影像、影像组学特征和临床信息，可以提供更为全面的预后信息。然而，不同模态之间存在固有的异质性，引入额外的模态可能会加剧异质性数据融合的挑战。此外，现有的多模态融合方法往往难以适应更丰富的医学模态数据，难以捕捉跨模态的关系。为克服这些局限性，我们提出了三模态交互慢性肝网络（TMI-CLNet）。具体来说，我们开发了内模态聚合模块和三模态交叉注意力融合模块，前者旨在消除内模态冗余，后者则用于提取跨模态信息。此外，我们设计了一种三模态特征融合损失函数，以在不同模态之间对齐特征表示。在肝脏预后数据集上的广泛实验表明，我们的方法显著优于现有的最先进的单模态模型和其他多模态技术。我们的代码可在以下链接获取：此 http URL。 

---
# Leveraging Large Language Models to Predict Antibody Biological Activity Against Influenza A Hemagglutinin 

**Title (ZH)**: 利用大型语言模型预测抗流感A型血凝素抗体的生物学活性 

**Authors**: Ella Barkan, Ibrahim Siddiqui, Kevin J. Cheng, Alex Golts, Yoel Shoshan, Jeffrey K. Weber, Yailin Campos Mota, Michal Ozery-Flato, Giuseppe A. Sautto  

**Link**: [PDF](https://arxiv.org/pdf/2502.00694)  

**Abstract**: Monoclonal antibodies (mAbs) represent one of the most prevalent FDA-approved modalities for treating autoimmune diseases, infectious diseases, and cancers. However, discovery and development of therapeutic antibodies remains a time-consuming and expensive process. Recent advancements in machine learning (ML) and artificial intelligence (AI) have shown significant promise in revolutionizing antibody discovery and optimization. In particular, models that predict antibody biological activity enable in-silico evaluation of binding and functional properties; such models can prioritize antibodies with the highest likelihoods of success in costly and time-intensive laboratory testing procedures. We here explore an AI model for predicting the binding and receptor blocking activity of antibodies against influenza A hemagglutinin (HA) antigens. Our present model is developed with the MAMMAL framework for biologics discovery to predict antibody-antigen interactions using only sequence information. To evaluate the model's performance, we tested it under various data split conditions to mimic real-world scenarios.
Our models achieved an AUROC $\geq$ 0.91 for predicting the activity of existing antibodies against seen HAs and an AUROC of 0.9 for unseen HAs. For novel antibody activity prediction, the AUROC was 0.73, which further declined to 0.63-0.66 under stringent constraints on similarity to existing antibodies. These results demonstrate the potential of AI foundation models to transform antibody design by reducing dependence on extensive laboratory testing and enabling more efficient prioritization of antibody candidates. Moreover, our findings emphasize the critical importance of diverse and comprehensive antibody datasets to improve the generalization of prediction models, particularly for novel antibody development. 

**Abstract (ZH)**: 单克隆抗体（mAbs）是FDA批准用于治疗自身免疫疾病、感染性疾病和癌症的最常用疗法之一。然而，治疗抗体的发现与开发仍然是一个耗时且昂贵的过程。近期，在机器学习（ML）和人工智能（AI）领域的进步显示了革命性改变抗体发现和优化的巨大潜力。特别是在预测抗体生物活性的模型的帮助下，可以通过计算仿真评估抗体的结合和功能性特征；这些模型能够优先筛选出在成本高昂且耗时的实验室测试过程中成功率最高的抗体。我们在此探讨了一种基于人工智能的模型，用于预测针对流感A型血凝素（HA）抗原的抗体的结合和受体阻断活性。我们的模型使用MAMMAL框架进行生物制剂发现，仅通过序列信息预测抗体-抗原相互作用。为了评估模型性能，我们在不同数据分割条件下进行测试，以模拟实际场景。

我们的模型在预测现有抗体对已知HA的活性时达到了AUROC≥0.91，在预测未见过HA的活性时达到了AUROC=0.9。对于新型抗体活性预测，AUROC为0.73，而在对新型抗体相似性有严格限制的情况下，这一数值进一步下降至0.63-0.66。这些结果表明，AI基础模型有可能通过减少对大量实验室测试的依赖并促进更高效的抗体候选物筛选来进行抗体设计。此外，我们的研究表明，为了提高预测模型的泛化能力，特别是对于新型抗体的开发，多样且全面的抗体数据集至关重要。 

---
# Dissecting Submission Limit in Desk-Rejections: A Mathematical Analysis of Fairness in AI Conference Policies 

**Title (ZH)**: 拆解提交限制在直接拒稿中的作用：对AI会议政策中公平性的数学分析 

**Authors**: Yuefan Cao, Xiaoyu Li, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00690)  

**Abstract**: As AI research surges in both impact and volume, conferences have imposed submission limits to maintain paper quality and alleviate organizational pressure. In this work, we examine the fairness of desk-rejection systems under submission limits and reveal that existing practices can result in substantial inequities. Specifically, we formally define the paper submission limit problem and identify a critical dilemma: when the number of authors exceeds three, it becomes impossible to reject papers solely based on excessive submissions without negatively impacting innocent authors. Thus, this issue may unfairly affect early-career researchers, as their submissions may be penalized due to co-authors with significantly higher submission counts, while senior researchers with numerous papers face minimal consequences. To address this, we propose an optimization-based fairness-aware desk-rejection mechanism and formally define two fairness metrics: individual fairness and group fairness. We prove that optimizing individual fairness is NP-hard, whereas group fairness can be efficiently optimized via linear programming. Through case studies, we demonstrate that our proposed system ensures greater equity than existing methods, including those used in CVPR 2025, offering a more socially just approach to managing excessive submissions in AI conferences. 

**Abstract (ZH)**: 随着人工智能研究在影响力和数量上急剧增长，会议已经实施提交限制以保持论文质量并减轻组织压力。在本研究中，我们审视了在提交限制下的桌面否决系统公平性问题，并揭示现有做法可能导致实质性不公。具体而言，我们正式定义了论文提交限制问题，并识别了一个关键困境：当作者人数超过三人时，在仅基于过度提交数量进行否决的情况下，不可避免地会影响无辜作者。因此，这个问题可能导致对早期职业研究人员的不公平影响，因为他们可能会因合作者的提交数量显著较高而受到惩罚，而资深研究人员由于论文数量众多而面临的后果较小。为解决这一问题，我们提出了一种基于优化的公平性意识桌面否决机制，并正式定义了两个公平性指标：个人公平性和群体公平性。我们证明了优化个人公平性是NP难问题，而群体公平性可以通过线性规划有效优化。通过案例研究，我们表明，我们提出的系统在公平性方面优于现有方法，包括CVPR 2025使用的方法，为AI会议管理过度提交提供了一种更加社会正义的方法。 

---
# High-Order Matching for One-Step Shortcut Diffusion Models 

**Title (ZH)**: 一步捷径扩散模型中的高阶匹配方法 

**Authors**: Bo Chen, Chengyue Gong, Xiaoyu Li, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song, Mingda Wan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00688)  

**Abstract**: One-step shortcut diffusion models [Frans, Hafner, Levine and Abbeel, ICLR 2025] have shown potential in vision generation, but their reliance on first-order trajectory supervision is fundamentally limited. The Shortcut model's simplistic velocity-only approach fails to capture intrinsic manifold geometry, leading to erratic trajectories, poor geometric alignment, and instability-especially in high-curvature regions. These shortcomings stem from its inability to model mid-horizon dependencies or complex distributional features, leaving it ill-equipped for robust generative modeling. In this work, we introduce HOMO (High-Order Matching for One-Step Shortcut Diffusion), a game-changing framework that leverages high-order supervision to revolutionize distribution transportation. By incorporating acceleration, jerk, and beyond, HOMO not only fixes the flaws of the Shortcut model but also achieves unprecedented smoothness, stability, and geometric precision. Theoretically, we prove that HOMO's high-order supervision ensures superior approximation accuracy, outperforming first-order methods. Empirically, HOMO dominates in complex settings, particularly in high-curvature regions where the Shortcut model struggles. Our experiments show that HOMO delivers smoother trajectories and better distributional alignment, setting a new standard for one-step generative models. 

**Abstract (ZH)**: 一步捷径扩散模型 [Frans, Hafner, Levine and Abbeel, ICLR 2025] 在视觉生成方面显示出了潜力，但其依赖于一阶轨迹监督的根本限制使其面临局限。捷径模型仅依赖速度的简单方法无法捕捉内在流形几何结构，导致轨迹混乱、几何对齐不良以及不稳定，尤其是在高曲率区域尤为明显。这些问题源于其无法建模中间时段的依赖关系或复杂的分布特征，使之在稳健的生成建模方面显得无能为力。在本研究中，我们引入了HOMO（高阶匹配的一步捷径扩散模型）框架，该框架利用高阶监督重塑分布传输。通过引入加速度、瞬时加速度以及其他高阶特征，HOMO 不仅仅修复了捷径模型的缺陷，还实现了前所未有的平滑性、稳定性和几何精确性。理论上，我们证明了HOMO的高阶监督确保了其具有更优的逼近精度，超越了一阶方法。实验中，HOMO 在复杂场景中的表现尤为突出，特别是在捷径模型难以应对的高曲率区域。我们的实验表明，HOMO 提供了更平滑的轨迹和更好的分布对齐，确立了一步生成模型的新标准。 

---
# Compositional Concept-Based Neuron-Level Interpretability for Deep Reinforcement Learning 

**Title (ZH)**: 基于组合概念的神经元层级可解释性在深度强化学习中的应用 

**Authors**: Zeyu Jiang, Hai Huang, Xingquan Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2502.00684)  

**Abstract**: Deep reinforcement learning (DRL), through learning policies or values represented by neural networks, has successfully addressed many complex control problems. However, the neural networks introduced by DRL lack interpretability and transparency. Current DRL interpretability methods largely treat neural networks as black boxes, with few approaches delving into the internal mechanisms of policy/value networks. This limitation undermines trust in both the neural network models that represent policies and the explanations derived from them. In this work, we propose a novel concept-based interpretability method that provides fine-grained explanations of DRL models at the neuron level. Our method formalizes atomic concepts as binary functions over the state space and constructs complex concepts through logical operations. By analyzing the correspondence between neuron activations and concept functions, we establish interpretable explanations for individual neurons in policy/value networks. Experimental results on both continuous control tasks and discrete decision-making environments demonstrate that our method can effectively identify meaningful concepts that align with human understanding while faithfully reflecting the network's decision-making logic. 

**Abstract (ZH)**: 深度强化学习（DRL）通过使用神经网络来学习策略或值，已经成功地解决了许多复杂的控制问题。然而，DRL 引入的神经网络缺乏可解释性和透明度。当前的 DRL 解释方法大多将神经网络视为黑盒子，对策略/值网络的内部机制涉及较少。这种限制削弱了对代表策略的神经网络模型及其解释的信任。在本文中，我们提出了一种新颖的概念基于解释方法，该方法在神经元级别提供了对 DRL 模型的细粒度解释。我们的方法将原子概念形式化为状态空间上的二元函数，并通过逻辑操作构造复杂概念。通过对神经元激活与概念函数之间的对应关系进行分析，我们建立了策略/值网络中个体神经元的可解释解释。在连续控制任务和离散决策环境中的实验结果表明，我们的方法能够有效地识别与人类理解一致且真实反映网络决策逻辑的意义概念。 

---
# Guidance Source Matters: How Guidance from AI, Expert, or a Group of Analysts Impacts Visual Data Preparation and Analysis 

**Title (ZH)**: 指导来源重要性探究：来自AI、专家或分析师团队的指导如何影响视觉数据准备与分析 

**Authors**: Arpit Narechania, Alex Endert, Atanu R Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2502.00682)  

**Abstract**: The progress in generative AI has fueled AI-powered tools like co-pilots and assistants to provision better guidance, particularly during data analysis. However, research on guidance has not yet examined the perceived efficacy of the source from which guidance is offered and the impact of this source on the user's perception and usage of guidance. We ask whether users perceive all guidance sources as equal, with particular interest in three sources: (i) AI, (ii) human expert, and (iii) a group of human analysts. As a benchmark, we consider a fourth source, (iv) unattributed guidance, where guidance is provided without attribution to any source, enabling isolation of and comparison with the effects of source-specific guidance. We design a five-condition between-subjects study, with one condition for each of the four guidance sources and an additional (v) no-guidance condition, which serves as a baseline to evaluate the influence of any kind of guidance. We situate our study in a custom data preparation and analysis tool wherein we task users to select relevant attributes from an unfamiliar dataset to inform a business report. Depending on the assigned condition, users can request guidance, which the system then provides in the form of attribute suggestions. To ensure internal validity, we control for the quality of guidance across source-conditions. Through several metrics of usage and perception, we statistically test five preregistered hypotheses and report on additional analysis. We find that the source of guidance matters to users, but not in a manner that matches received wisdom. For instance, users utilize guidance differently at various stages of analysis, including expressing varying levels of regret, despite receiving guidance of similar quality. Notably, users in the AI condition reported both higher post-task benefit and regret. 

**Abstract (ZH)**: 生成式人工智能的进步推动了诸如副驾和助手之类的AI驱动工具的发展，这些工具能够提供更好的指导，尤其是在数据分析过程中。然而，关于指导的研究尚未探讨提供指导的来源对其有效性的感知以及这一来源对用户感知和使用指导的影响。我们询问用户是否认为所有来源的指导都具有同等价值，我们特别关注三种来源：（i）人工智能、（ii）人类专家和（iii）一组人类分析师。作为基准，我们考虑了一个第四种来源，即（iv）未归因指导，这意味着指导不标明任何来源，这使我们能够隔离并比较特定来源指导的影响。我们设计了一项涉及五个条件的被试间实验，分别为每种指导来源的条件，以及一个额外的（v）无指导条件，后者作为基准，用于评估任何类型指导的影响。我们将研究置于一个定制的数据准备与分析工具中，让用户从一个不熟悉的数据库中选择相关属性，以生成一本商业报告。根据分配的条件，用户可以请求指导，系统会以属性建议的形式提供指导。为了确保内部有效性，我们控制了每种来源条件下的指导质量。通过多个使用和感知指标，我们统计检验了五个预先注册的假设，并报告了额外分析的结果。我们发现，提供指导的来源对用户很重要，但其重要性的方式并不符合既有的常识。例如，用户在分析的不同阶段对指导的使用方式不同，尽管他们收到的是质量相似的指导，但仍表现出不同的后悔水平。值得注意的是，在人工智能条件下的用户报告了更高的任务后收益和后悔感。 

---
# A Survey of Quantized Graph Representation Learning: Connecting Graph Structures with Large Language Models 

**Title (ZH)**: 量化图表示学习综述：连接图结构与大规模语言模型 

**Authors**: Qika Lin, Zhen Peng, Kaize Shi, Kai He, Yiming Xu, Erik Cambria, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00681)  

**Abstract**: Recent years have witnessed rapid advances in graph representation learning, with the continuous embedding approach emerging as the dominant paradigm. However, such methods encounter issues regarding parameter efficiency, interpretability, and robustness. Thus, Quantized Graph Representation (QGR) learning has recently gained increasing interest, which represents the graph structure with discrete codes instead of conventional continuous embeddings. Given its analogous representation form to natural language, QGR also possesses the capability to seamlessly integrate graph structures with large language models (LLMs). As this emerging paradigm is still in its infancy yet holds significant promise, we undertake this thorough survey to promote its rapid future prosperity. We first present the background of the general quantization methods and their merits. Moreover, we provide an in-depth demonstration of current QGR studies from the perspectives of quantized strategies, training objectives, distinctive designs, knowledge graph quantization, and applications. We further explore the strategies for code dependence learning and integration with LLMs. At last, we give discussions and conclude future directions, aiming to provide a comprehensive picture of QGR and inspire future research. 

**Abstract (ZH)**: 近年来，图表示学习领域取得了 rapid 的发展，连续嵌入方法逐渐成为主导范式。然而，此类方法在参数效率、可解释性和稳健性方面遇到了一些问题。因此，量化图表示（QGR）学习最近获得了越来越大的兴趣，它使用离散码代替传统的连续嵌入来表示图结构。由于其表示形式与自然语言具有相似性，QGR 还具备无缝整合图结构与大语言模型（LLMs）的能力。尽管这一新兴范式仍处于起步阶段，但具有许多潜力，因此我们进行了这篇全面的综述，以促进其未来的发展。首先，我们介绍了量化方法的背景及其优势。此外，我们从量化策略、训练目标、特色设计、知识图谱量化以及应用等方面深入阐述了当前的 QGR 研究。我们还探讨了编码依赖学习策略以及与大语言模型的整合方法。最后，我们进行了讨论并指出了未来的研究方向，旨在提供一个全面的 QGR 图景，并激发未来的研究。 

---
# How Contaminated Is Your Benchmark? Quantifying Dataset Leakage in Large Language Models with Kernel Divergence 

**Title (ZH)**: 您的基准数据集受到多少污染？使用核散度量化大型语言模型中的数据集泄露程度 

**Authors**: Hyeong Kyu Choi, Maxim Khanov, Hongxin Wei, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.00678)  

**Abstract**: Dataset contamination, where evaluation datasets overlap with pre-training corpora, inflates performance metrics and undermines the reliability of model evaluations. Quantifying dataset contamination thus becomes essential to ensure that performance evaluations genuinely reflect a model's ability to generalize to unseen data, rather than relying on memorized examples. To address this problem, we propose Kernel Divergence Score (KDS), a novel method that quantifies dataset contamination by computing the divergence between the kernel similarity matrix of sample embeddings, before and after fine-tuning on the benchmark dataset. Leveraging the insight that fine-tuning affects unseen samples more significantly than seen ones, KDS provides a reliable measure of contamination. Through extensive experiments on controlled contamination scenarios, KDS demonstrates a near-perfect correlation with contamination levels and outperforms existing baselines. Additionally, we perform comprehensive ablation studies to analyze the impact of key design choices, providing deeper insights into the components and effectiveness of KDS. These ablations highlight the importance of leveraging fine-grained kernel-based information and confirm the reliability of the proposed framework across diverse datasets and settings. 

**Abstract (ZH)**: 数据集污染是指评估数据集与预训练语料库重叠，这会夸大模型性能指标并损害模型评估的可靠性。因此，量化数据集污染变得至关重要，以确保性能评估真正反映了模型在未见过的数据上的泛化能力，而不是依赖于对已见过的示例的记忆。为了解决这一问题，我们提出了核散度评分（KDS）方法，这是一种新颖的方法，通过计算样本嵌入在基准数据集上微调前后核相似矩阵的散度来量化数据集污染。利用微调对未见过的样本影响更大，而对见过的样本影响较小的洞见，KDS提供了污染的可靠衡量标准。通过在控制污染场景中的广泛实验，KDS与污染水平显示出接近完美的相关性，并超越了现有基线。此外，我们进行了全面的消融研究，分析了关键设计选择的影响，提供了对KDS组件及其有效性的更深入理解。这些消融研究突出了利用细粒度核基信息的重要性，并证实了所提出框架在不同数据集和场景下的可靠性。 

---
# Biogeochemistry-Informed Neural Network (BINN) for Improving Accuracy of Model Prediction and Scientific Understanding of Soil Organic Carbon 

**Title (ZH)**: 基于生物地球化学的神经网络（BINN）以提高模型预测准确性并增进对土壤有机碳科学理解 

**Authors**: Haodi Xu, Joshua Fan, Feng Tao, Lifen Jiang, Fengqi You, Benjamin Z. Houlton, Ying Sun, Carla P. Gomes, Yiqi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.00672)  

**Abstract**: Big data and the rapid development of artificial intelligence (AI) provide unprecedented opportunities to enhance our understanding of the global carbon cycle and other biogeochemical processes. However, retrieving mechanistic knowledge from big data remains a challenge. Here, we develop a Biogeochemistry-Informed Neural Network (BINN) that seamlessly integrates a vectorized process-based soil carbon cycle model (i.e., Community Land Model version 5, CLM5) into a neural network (NN) structure to examine mechanisms governing soil organic carbon (SOC) storage from big data. BINN demonstrates high accuracy in retrieving biogeochemical parameter values from synthetic data in a parameter recovery experiment. We use BINN to predict six major processes regulating the soil carbon cycle (or components in process-based models) from 25,925 observed SOC profiles across the conterminous US and compared them with the same processes previously retrieved by a Bayesian inference-based PROcess-guided deep learning and DAta-driven modeling (PRODA) approach (Tao et al. 2020; 2023). The high agreement between the spatial patterns of the retrieved processes using the two approaches with an average correlation coefficient of 0.81 confirms BINN's ability in retrieving mechanistic knowledge from big data. Additionally, the integration of neural networks and process-based models in BINN improves computational efficiency by more than 50 times over PRODA. We conclude that BINN is a transformative tool that harnesses the power of both AI and process-based modeling, facilitation new scientific discoveries while improving interpretability and accuracy of Earth system models. 

**Abstract (ZH)**: 大数据和人工智能（AI）的迅速发展为增强我们对全球碳循环和其他生物地球化学过程的理解提供了前所未有的机遇。然而，从大数据中提取机制知识仍是一个挑战。为此，我们开发了一个生物地球化学导向的神经网络（BINN），将矢量化过程驱动的土壤碳循环模型（即社区陆地模型版本5，CLM5）无缝集成到神经网络（NN）结构中，以从大数据中探讨土壤有机碳（SOC）储存机制。BINN在参数恢复实验中从合成数据中提取生物地球化学参数值显示了极高的准确性。我们使用BINN预测了美国本土全域内25,925个观测到的SOC剖面中六个主要控制土壤碳循环的过程（或过程驱动模型中的组成部分），并将这些过程与通过贝叶斯推理导向的PROcess-guided深度学习和Data-driven建模（PRODA）方法（Tao等，2020；2023）之前提取的过程进行比较。两种方法恢复过程中空间模式的一致性，平均相关系数为0.81，证实了BINN从大数据中提取机制知识的能力。此外，BINN将神经网络与过程驱动模型的集成在计算效率上提高了50倍以上。我们得出结论，BINN是一个具有革新性的工具，它利用了AI和过程驱动建模的双重力量，促进了新的科学发现，同时提高了地球系统模型的解释性和准确性。 

---
# Avoiding $\mathbf{exp(R_{max})}$ scaling in RLHF through Preference-based Exploration 

**Title (ZH)**: 通过基于偏好的探索避免RLHF中的$\mathbf{exp(R_{max})}$缩放 

**Authors**: Mingyu Chen, Yiding Chen, Wen Sun, Xuezhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00666)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has emerged as a pivotal technique for large language model (LLM) alignment. This paper studies the setting of online RLHF and focus on improving sample efficiency. All existing algorithms in online RLHF, whether doing passive exploration or active exploration, suffer from a sample complexity that scales exponentially with the scale of the reward function. This fundamental limitation hinders their effectiveness in scenarios with heavily skewed preferences, e.g. questions with a unique correct solution. To address this, we introduce Self-Exploring Preference-Incentive Online Preference Optimization (SE-POPO), an online RLHF algorithm that for the first time achieves a sample complexity that scales polynomially with the reward scale, answering an open problem raised by Xie et al. (2024).. Theoretically, we demonstrate that the sample complexity of SE-POPO dominates that of existing exploration algorithms. Empirically, our systematic evaluation confirms that SE-POPO is more sample-efficient than both exploratory and non-exploratory baselines, in two primary application scenarios of RLHF as well as on public benchmarks, marking a significant step forward in RLHF algorithm design. 

**Abstract (ZH)**: 人类反馈强化学习（RLHF）已成为大型语言模型（LLM）对齐的关键技术。本文研究了在线RLHF的设置，并专注于提高样本效率。现有的所有在线RLHF算法，无论是进行被动探索还是主动探索，都受到了样本复杂度随着奖励函数规模指数增长的基本限制。这一根本限制阻碍了它们在偏好严重失衡的场景下的效果，例如那些具有唯一正确答案的问题。为了解决这个问题，我们提出了Self-Exploring Preference-Incentive Online Preference Optimization（SE-POPO）算法，这是首次实现样本复杂度与奖励规模呈多项式增长的在线RLHF算法，解决了Xie等（2024）提出的公开问题。理论分析表明，SE-POPO的样本复杂度优于现有探索算法的样本复杂度。实验结果显示，我们在两种主要的RLHF应用场景和公共基准测试中系统的评估均证明SE-POPO比探索性和非探索性基线更具有样本效率，这标志着RLHF算法设计上的一个重要进步。 

---
# Enhanced Convolutional Neural Networks for Improved Image Classification 

**Title (ZH)**: 增强卷积神经网络以提高图像分类性能 

**Authors**: Xiaoran Yang, Shuhan Yu, Wenxi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00663)  

**Abstract**: Image classification is a fundamental task in computer vision with diverse applications, ranging from autonomous systems to medical imaging. The CIFAR-10 dataset is a widely used benchmark to evaluate the performance of classification models on small-scale, multi-class datasets. Convolutional Neural Networks (CNNs) have demonstrated state-of-the-art results; however, they often suffer from overfitting and suboptimal feature representation when applied to challenging datasets like CIFAR-10. In this paper, we propose an enhanced CNN architecture that integrates deeper convolutional blocks, batch normalization, and dropout regularization to achieve superior performance. The proposed model achieves a test accuracy of 84.95%, outperforming baseline CNN architectures. Through detailed ablation studies, we demonstrate the effectiveness of the enhancements and analyze the hierarchical feature representations. This work highlights the potential of refined CNN architectures for tackling small-scale image classification problems effectively. 

**Abstract (ZH)**: 图像分类是计算机视觉中的一个基本任务，具有广泛的应用，从自主系统到医学影像。CIFAR-10 数据集是一个广泛使用的基准，用于评估分类模型在小型多分类数据集上的性能。卷积神经网络（CNNs）已经展示了最先进的结果；然而，当应用于像CIFAR-10这样的具有挑战性的数据集时，它们往往会遭受过拟合和特征表示不佳的问题。本文提出了一种改进的CNN架构，该架构结合了更深的卷积块、批量标准化和dropout正则化，以实现更好的性能。所提出的模型在测试集上的准确率为84.95%，优于基线CNN架构。通过详细的消融研究，我们证明了这些改进的有效性，并分析了层次特征表示。本文强调了改进的CNN架构在有效解决小型图像分类问题方面的潜力。 

---
# LLM Safety Alignment is Divergence Estimation in Disguise 

**Title (ZH)**: LLM 安全对齐实际上是偏差估计的伪装 

**Authors**: Rajdeep Haldar, Ziyi Wang, Qifan Song, Guang Lin, Yue Xing  

**Link**: [PDF](https://arxiv.org/pdf/2502.00657)  

**Abstract**: We propose a theoretical framework demonstrating that popular Large Language Model (LLM) alignment methods, including Reinforcement Learning from Human Feedback (RLHF) and alternatives, fundamentally function as divergence estimators between aligned (preferred or safe) and unaligned (less-preferred or harmful) distributions. This explains the separation phenomenon between safe and harmful prompts in the model hidden representation after alignment. Inspired by the theoretical results, we identify that some alignment methods are better than others in terms of separation and, introduce a new method, KLDO, and further demonstrate the implication of our theories. We advocate for compliance-refusal datasets over preference datasets to enhance safety alignment, supported by both theoretical reasoning and empirical evidence. Additionally, to quantify safety separation, we leverage a distance metric in the representation space and statistically validate its efficacy as a statistical significant indicator of LLM resilience against jailbreak attacks. 

**Abstract (ZH)**: 我们提出了一种理论框架，认为流行的大型语言模型（LLM）对齐方法，包括基于人类反馈的强化学习（RLHF）及其变体，本质上作为对齐（首选或安全）和未对齐（不太首选或有害）分布之间偏斜度的估计器来运作。这解释了对齐后模型隐藏表示中安全和有害提示之间的分离现象。受理论结果的启发，我们发现一些对齐方法在分离方面优于其他方法，并且引入了一种新方法KLDO，并进一步验证了我们理论的意义。我们主张使用合规拒绝数据集而不是偏好数据集来增强安全性对齐，并且这一主张得到了理论推理和实证证据的支持。此外，为了量化安全性分离，我们利用表示空间中的距离度量，并通过统计验证其作为LLM对脱笼攻击有弹性的显著指标的有效性。 

---
# TrojanTime: Backdoor Attacks on Time Series Classification 

**Title (ZH)**: TrojanTime：时间序列分类中的后门攻击 

**Authors**: Chang Dong, Zechao Sun, Guangdong Bai, Shuying Piao, Weitong Chen, Wei Emma Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00646)  

**Abstract**: Time Series Classification (TSC) is highly vulnerable to backdoor attacks, posing significant security threats. Existing methods primarily focus on data poisoning during the training phase, designing sophisticated triggers to improve stealthiness and attack success rate (ASR). However, in practical scenarios, attackers often face restrictions in accessing training data. Moreover, it is a challenge for the model to maintain generalization ability on clean test data while remaining vulnerable to poisoned inputs when data is inaccessible. To address these challenges, we propose TrojanTime, a novel two-step training algorithm. In the first stage, we generate a pseudo-dataset using an external arbitrary dataset through target adversarial attacks. The clean model is then continually trained on this pseudo-dataset and its poisoned version. To ensure generalization ability, the second stage employs a carefully designed training strategy, combining logits alignment and batch norm freezing. We evaluate TrojanTime using five types of triggers across four TSC architectures in UCR benchmark datasets from diverse domains. The results demonstrate the effectiveness of TrojanTime in executing backdoor attacks while maintaining clean accuracy. Finally, to mitigate this threat, we propose a defensive unlearning strategy that effectively reduces the ASR while preserving clean accuracy. 

**Abstract (ZH)**: 时间序列分类（TSC）对后门攻击极其脆弱，带来了显著的安全威胁。现有方法主要集中在训练阶段的数据污染，设计复杂的触发器以提高隐蔽性和攻击成功率（ASR）。然而，在实际场景中，攻击者往往难以访问训练数据。此外，在数据不可用的情况下，模型在保持泛化能力的同时仍对受污染输入保持脆弱性，这是一大挑战。为应对这些挑战，我们提出了TrojanTime，一种创新的两步训练算法。在第一步中，我们利用目标对抗攻击生成一个伪数据集，该数据集是通过外部任意数据集生成的。清洁模型随后在此伪数据集及其受污染版本上连续训练。为了确保泛化能力，第二步采用精心设计的训练策略，结合了logits对齐和批量归一化冻结。我们使用UCR基准数据集中的四个不同类型的时间序列分类架构，并针对五种类型的触发器进行了评估。结果表明，TrojanTime在执行后门攻击的同时保持了干净准确度。最后，为了缓解这一威胁，我们提出了一种有效的防御性遗忘策略，该策略在减少ASR的同时保持了干净准确度。 

---
# Evaluating Small Language Models for News Summarization: Implications and Factors Influencing Performance 

**Title (ZH)**: 评估小型语言模型在新闻摘要生成中的表现：影响性能的含义与因素 

**Authors**: Borui Xu, Yao Chen, Zeyi Wen, Weiguo Liu, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2502.00641)  

**Abstract**: The increasing demand for efficient summarization tools in resource-constrained environments highlights the need for effective solutions. While large language models (LLMs) deliver superior summarization quality, their high computational resource requirements limit practical use applications. In contrast, small language models (SLMs) present a more accessible alternative, capable of real-time summarization on edge devices. However, their summarization capabilities and comparative performance against LLMs remain underexplored. This paper addresses this gap by presenting a comprehensive evaluation of 19 SLMs for news summarization across 2,000 news samples, focusing on relevance, coherence, factual consistency, and summary length. Our findings reveal significant variations in SLM performance, with top-performing models such as Phi3-Mini and Llama3.2-3B-Ins achieving results comparable to those of 70B LLMs while generating more concise summaries. Notably, SLMs are better suited for simple prompts, as overly complex prompts may lead to a decline in summary quality. Additionally, our analysis indicates that instruction tuning does not consistently enhance the news summarization capabilities of SLMs. This research not only contributes to the understanding of SLMs but also provides practical insights for researchers seeking efficient summarization solutions that balance performance and resource use. 

**Abstract (ZH)**: 资源限制环境下高效摘要工具需求的不断增加突显了有效解决方案的必要性。虽然大规模语言模型（LLMs）提供了卓越的摘要质量，但其高计算资源要求限制了其在实际应用中的使用。相比之下，小规模语言模型（SLMs）则提供了更为易用的替代方案，能够实现在边缘设备上的实时摘要。然而，SLMs的摘要能力及其与LLMs的相对性能仍然有待进一步探索。本文通过在2000篇新闻样本上全面评估19种SLMs的新闻摘要效果，专注于相关性、连贯性、事实一致性以及摘要长度，填补了这一空白。研究发现，SLMs在性能上存在显著差异，表现最佳的模型如Phi3-Mini和Llama3.2-3B-Ins在生成更为紧凑的摘要时达到了与70B LLM相似的效果。值得注意的是，SLMs更适合简洁的提示，过于复杂的提示可能导致摘要质量下降。此外，我们的分析表明，指令调优并不总是能提升SLMs的新闻摘要能力。这项研究不仅增进了对SLMs的理解，也为寻求在性能和资源使用之间取得平衡的有效摘要解决方案的研究人员提供了实用的指导。 

---
# Zeroth-order Informed Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer 

**Title (ZH)**: 基于零阶信息的扩散模型调优：递归 likelihood 比率优化器 

**Authors**: Tao Ren, Zishi Zhang, Zehao Li, Jingyang Jiang, Shentao Qin, Guanghao Li, Yan Li, Yi Zheng, Xinping Li, Min Zhan, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00639)  

**Abstract**: The probabilistic diffusion model (DM), generating content by inferencing through a recursive chain structure, has emerged as a powerful framework for visual generation. After pre-training on enormous unlabeled data, the model needs to be properly aligned to meet requirements for downstream applications. How to efficiently align the foundation DM is a crucial task. Contemporary methods are either based on Reinforcement Learning (RL) or truncated Backpropagation (BP). However, RL and truncated BP suffer from low sample efficiency and biased gradient estimation respectively, resulting in limited improvement or, even worse, complete training failure. To overcome the challenges, we propose the Recursive Likelihood Ratio (RLR) optimizer, a zeroth-order informed fine-tuning paradigm for DM. The zeroth-order gradient estimator enables the computation graph rearrangement within the recursive diffusive chain, making the RLR's gradient estimator an unbiased one with the lower variance than other methods. We provide theoretical guarantees for the performance of the RLR. Extensive experiments are conducted on image and video generation tasks to validate the superiority of the RLR. Furthermore, we propose a novel prompt technique that is natural for the RLR to achieve a synergistic effect. 

**Abstract (ZH)**: 概率扩散模型（DM）通过递归链结构进行推理生成内容，已经成为了视觉生成的一种强大框架。在大规模未标注数据上进行预训练后，该模型需要适当调整，以满足下游应用的要求。如何高效地对基础DM进行对齐是一项关键任务。现有的方法要么基于强化学习（RL），要么基于截断反向传播（truncated BP）。然而，RL和截断反向传播分别存在样本效率低和梯度估计偏差的问题，导致改进有限，甚至可能出现完全训练失败的情况。为克服这些挑战，我们提出了递归似然比（RLR）优化器，这是一种零阶信息驱动的DM的细调范式。零阶梯度估计器允许在递归扩散链中进行计算图重组，使RLR的梯度估计器在无偏性方面优于其他方法，并且具有更低的方差。我们对RLR的性能提供了理论保证。在图像和视频生成任务中进行了广泛的实验，以验证RLR的优越性。此外，我们提出了一种新的提示技术，该技术与RLR兼容性良好，可以产生协同效应。 

---
# SimulPL: Aligning Human Preferences in Simultaneous Machine Translation 

**Title (ZH)**: SimulPL：同时机器翻译中的人类偏好对齐 

**Authors**: Donglei Yu, Yang Zhao, Jie Zhu, Yangyifan Xu, Yu Zhou, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2502.00634)  

**Abstract**: Simultaneous Machine Translation (SiMT) generates translations while receiving streaming source inputs. This requires the SiMT model to learn a read/write policy, deciding when to translate and when to wait for more source input. Numerous linguistic studies indicate that audiences in SiMT scenarios have distinct preferences, such as accurate translations, simpler syntax, and no unnecessary latency. Aligning SiMT models with these human preferences is crucial to improve their performances. However, this issue still remains unexplored. Additionally, preference optimization for SiMT task is also challenging. Existing methods focus solely on optimizing the generated responses, ignoring human preferences related to latency and the optimization of read/write policy during the preference optimization phase. To address these challenges, we propose Simultaneous Preference Learning (SimulPL), a preference learning framework tailored for the SiMT task. In the SimulPL framework, we categorize SiMT human preferences into five aspects: \textbf{translation quality preference}, \textbf{monotonicity preference}, \textbf{key point preference}, \textbf{simplicity preference}, and \textbf{latency preference}. By leveraging the first four preferences, we construct human preference prompts to efficiently guide GPT-4/4o in generating preference data for the SiMT task. In the preference optimization phase, SimulPL integrates \textbf{latency preference} into the optimization objective and enables SiMT models to improve the read/write policy, thereby aligning with human preferences more effectively. Experimental results indicate that SimulPL exhibits better alignment with human preferences across all latency levels in Zh$\rightarrow$En, De$\rightarrow$En and En$\rightarrow$Zh SiMT tasks. Our data and code will be available at \url{this https URL}. 

**Abstract (ZH)**: 同时机器翻译（SiMT）在接收流式源输入的同时生成翻译。这要求SiMT模型学习读取/写入策略，决定何时译出和何时等待更多源输入。众多语言学研究表明，在SiMT情景下的观众有明显的偏好，例如准确的翻译、更简单的句法结构以及没有不必要的延迟。使SiMT模型与这些人类偏好相一致对于提高其性能至关重要。然而，这一问题仍然未被充分探索。此外，同时机器翻译任务的前提优化也颇具挑战性。现有方法侧重于优化生成的响应，忽略了与延迟和偏好优化阶段的读取/写入策略有关的人类偏好优化。为应对这些挑战，我们提出了一种专门针对SiMT任务的偏好学习框架——Simultaneous Preference Learning（SimulPL）。

在SimulPL框架中，我们将SiMT人类偏好划分为五个方面：**翻译质量偏好**、**单调性偏好**、**关键点偏好**、**简单性偏好**和**延迟偏好**。通过利用前四种偏好，我们构建人类偏好提示，以高效引导GPT-4/4o生成SiMT任务的偏好数据。在偏好优化阶段，SimulPL将**延迟偏好**纳入优化目标，使SiMT模型能够改进其读取/写入策略，从而更有效地与人类偏好相契合。实验结果表明，在Zh→En、De→En和En→Zh的SiMT任务中，SimulPL在所有延迟级别上都更能够与人类偏好相契合。我们的数据和代码将于**此链接**获取。 

---
# Representations Shape Weak-to-Strong Generalization: Theoretical Insights and Empirical Predictions 

**Title (ZH)**: 表征塑造从弱到强的泛化能力：理论洞见与经验预测 

**Authors**: Yihao Xue, Jiping Li, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2502.00620)  

**Abstract**: Weak-to-Strong Generalization (W2SG), where a weak model supervises a stronger one, serves as an important analogy for understanding how humans might guide superhuman intelligence in the future. Promising empirical results revealed that a strong model can surpass its weak supervisor. While recent work has offered theoretical insights into this phenomenon, a clear understanding of the interactions between weak and strong models that drive W2SG remains elusive. We investigate W2SG through a theoretical lens and show that it can be characterized using kernels derived from the principal components of weak and strong models' internal representations. These kernels can be used to define a space that, at a high level, captures what the weak model is unable to learn but is learnable by the strong model. The projection of labels onto this space quantifies how much the strong model falls short of its full potential due to weak supervision. This characterization also provides insights into how certain errors in weak supervision can be corrected by the strong model, regardless of overfitting. Our theory has significant practical implications, providing a representation-based metric that predicts W2SG performance trends without requiring labels, as shown in experiments on molecular predictions with transformers and 5 NLP tasks involving 52 LLMs. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

弱到强泛化（W2SG），即一个较弱的模型监督一个更强的模型，为理解未来人类如何引导超人类智能提供了重要类比。实证研究表明，更强的模型能够超越其较弱的导师。尽管近期的工作为这一现象提供了理论见解，但尚不清楚如何通过弱模型和强模型之间的互动来驱动W2SG的机制。我们通过理论视角探讨W2SG，并展示它可以通过从弱模型和强模型内部表征的主要成分中推导出的核来表征。这些核可用于定义一个空间，在此空间中，从宏观角度看，可以捕捉到弱模型无法学到但可以由强模型学习的内容。将标签投影到该空间可以量化强模型因弱监督而导致的性能缺口有多大。这一表征还揭示了强模型如何纠正某些弱监督中的错误，而无需担心过拟合。我们的理论具有重要的实际意义，提供了无需标签即可预测W2SG性能趋势的基于表示的方法，如在使用变压器进行分子预测和涉及52个语言模型的5项NLP任务中所展示的那样。 

---
# Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspective 

**Title (ZH)**: 从控制理论视角出发的分布感知公平学习在医疗影像分割中的应用 

**Authors**: Yujin Oh, Pengfei Jin, Sangjoon Park, Sekeun Kim, Siyeop Yoon, Kyungsang Kim, Jin Sung Kim, Xiang Li, Quanzheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.00619)  

**Abstract**: Ensuring fairness in medical image segmentation is critical due to biases in imbalanced clinical data acquisition caused by demographic attributes (e.g., age, sex, race) and clinical factors (e.g., disease severity). To address these challenges, we introduce Distribution-aware Mixture of Experts (dMoE), inspired by optimal control theory. We provide a comprehensive analysis of its underlying mechanisms and clarify dMoE's role in adapting to heterogeneous distributions in medical image segmentation. Furthermore, we integrate dMoE into multiple network architectures, demonstrating its broad applicability across diverse medical image analysis tasks. By incorporating demographic and clinical factors, dMoE achieves state-of-the-art performance on two 2D benchmark datasets and a 3D in-house dataset. Our results highlight the effectiveness of dMoE in mitigating biases from imbalanced distributions, offering a promising approach to bridging control theory and medical image segmentation within fairness learning paradigms. The source code will be made available. 

**Abstract (ZH)**: 确保医学图像分割的公平性至关重要，这主要是由于临床数据采集中的偏差所引起的，这些偏差是由人口统计属性（如年龄、性别、种族）和临床因素（如疾病严重程度）导致的异质性。为应对这些挑战，我们引入了基于分布感知的专家混合模型（dMoE），该模型受到最优控制理论的启发。我们对其内部机制进行了全面分析，并阐明了dMoE在医学图像分割中适应异质性分布的作用。此外，我们将dMoE整合到多种网络架构中，展示了其在不同医学图像分析任务中的广泛应用。通过融入人口统计和临床因素，dMoE在两个2D基准数据集和一个3D内部数据集上取得了最新的性能。我们的研究结果突显了dMoE在减轻不平衡分布偏差方面的有效性，为在公平学习框架内将控制理论与医学图像分割相结合提供了一个有前景的方法。源代码将公开提供。 

---
# DesCLIP: Robust Continual Adaptation via General Attribute Descriptions for Pretrained Vision-Language Models 

**Title (ZH)**: DesCLIP：通过通用属性描述实现鲁棒连续适应的预训练视觉-语言模型 

**Authors**: Chiyuan He, Zihuan Qiu, Fanman Meng, Linfeng Xu, Qingbo Wu, Hongliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.00618)  

**Abstract**: Continual adaptation of vision-language models (VLMs) focuses on leveraging cross-modal pretrained knowledge to incrementally adapt for expanding downstream tasks and datasets, while tackling the challenge of knowledge forgetting. Existing research often focuses on connecting visual features with specific class text in downstream tasks, overlooking the latent relationships between general and specialized knowledge. Our findings reveal that forcing models to optimize inappropriate visual-text matches exacerbates forgetting of VLMs. To tackle this issue, we propose DesCLIP, which leverages general attribute (GA) descriptions to guide the understanding of specific class objects, enabling VLMs to establish robust \textit{vision-GA-class} trilateral associations rather than relying solely on \textit{vision-class} connections. Specifically, we introduce a language assistant to generate concrete GA description candidates via proper request prompts. Then, an anchor-based embedding filter is designed to obtain highly relevant GA description embeddings, which are leveraged as the paired text embeddings for visual-textual instance matching, thereby tuning the visual encoder. Correspondingly, the class text embeddings are gradually calibrated to align with these shared GA description embeddings. Extensive experiments demonstrate the advancements and efficacy of our proposed method, with comprehensive empirical evaluations highlighting its superior performance compared to existing pretrained and VLM-based continual learning methods. 

**Abstract (ZH)**: 持续适应视觉-语言模型（VLMs）专注于利用跨模态预训练知识来逐步适应扩展的下游任务和数据集，同时解决知识遗忘的挑战。现有研究往往集中在将视觉特征与下游任务中的特定类别文本连接起来，而忽视了通用知识与专门化知识之间的潜在关系。我们的研究结果表明，强制模型优化不合适的视觉-文本匹配会加剧VLMs的知识遗忘。为了解决这一问题，我们提出了DesCLIP方法，利用通用属性（GA）描述来引导特定类别对象的理解，从而使VLMs能够建立稳健的视图-GA类别三边关联，而不仅仅依赖于视图-类别连接。具体而言，我们引入了一种语言助手来通过适当的请求提示生成具体的GA描述候选。然后，设计了一个基于锚点的嵌入过滤器，以获取高度相关的GA描述嵌入，这些嵌入作为视觉-文本实例匹配的配对文本嵌入，从而调整视觉编码器。相应地，类别文本嵌入逐渐校准以与这些共享的GA描述嵌入对齐。广泛实验表明了我们提出方法的进步和有效性，并通过全面的经验评估强调了其优于现有预训练和VLM基持续学习方法的出色性能。 

---
# Enhancing Code Consistency in AI Research with Large Language Models and Retrieval-Augmented Generation 

**Title (ZH)**: 使用大型语言模型和检索增强生成技术提升AI研究中的代码一致性 

**Authors**: Rajat Keshri, Arun George Zachariah, Michael Boone  

**Link**: [PDF](https://arxiv.org/pdf/2502.00611)  

**Abstract**: Ensuring that code accurately reflects the algorithms and methods described in research papers is critical for maintaining credibility and fostering trust in AI research. This paper presents a novel system designed to verify code implementations against the algorithms and methodologies outlined in corresponding research papers. Our system employs Retrieval-Augmented Generation to extract relevant details from both the research papers and code bases, followed by a structured comparison using Large Language Models. This approach improves the accuracy and comprehensiveness of code implementation verification while contributing to the transparency, explainability, and reproducibility of AI research. By automating the verification process, our system reduces manual effort, enhances research credibility, and ultimately advances the state of the art in code verification. 

**Abstract (ZH)**: 确保代码准确反映研究论文中描述的算法和方法对于维持人工智能研究的可信度至关重要。本文提出了一种新颖的系统，旨在验证代码实现与对应研究论文中描述的算法和方法的一致性。该系统采用检索增强生成技术从研究论文和代码库中提取相关细节，随后使用大规模语言模型进行结构化的比较。该方法提高了代码实现验证的准确性和全面性，同时促进了人工智能研究的透明度、可解释性和可重复性。通过自动化验证过程，该系统减少了人工努力，增强了研究可信度，并最终推动了代码验证领域的最新进展。 

---
# Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective 

**Title (ZH)**: 物理导向神经网络中的梯度对齐：二阶优化视角 

**Authors**: Sifan Wang, Ananyae Kumar Bhartari, Bowen Li, Paris Perdikaris  

**Link**: [PDF](https://arxiv.org/pdf/2502.00604)  

**Abstract**: Multi-task learning through composite loss functions is fundamental to modern deep learning, yet optimizing competing objectives remains challenging. We present new theoretical and practical approaches for addressing directional conflicts between loss terms, demonstrating their effectiveness in physics-informed neural networks (PINNs) where such conflicts are particularly challenging to resolve. Through theoretical analysis, we demonstrate how these conflicts limit first-order methods and show that second-order optimization naturally resolves them through implicit gradient alignment. We prove that SOAP, a recently proposed quasi-Newton method, efficiently approximates the Hessian preconditioner, enabling breakthrough performance in PINNs: state-of-the-art results on 10 challenging PDE benchmarks, including the first successful application to turbulent flows with Reynolds numbers up to 10,000, with 2-10x accuracy improvements over existing methods. We also introduce a novel gradient alignment score that generalizes cosine similarity to multiple gradients, providing a practical tool for analyzing optimization dynamics. Our findings establish frameworks for understanding and resolving gradient conflicts, with broad implications for optimization beyond scientific computing. 

**Abstract (ZH)**: 多任务学习通过复合损失函数是现代深度学习的基础，但优化相互竞争的目标仍然具有挑战性。我们提出了新的理论和实践方法来解决损失项之间方向性的冲突，并展示了这些方法在物理信息神经网络（PINNs）中的有效性，特别是在PINNs中，这种冲突特别难以解决。通过理论分析，我们证明了这些冲突限制了梯度下降方法，并展示了二阶优化方法通过隐式梯度对齐自然解决了这些问题。我们证明了SOAP（最近提出的拟牛顿方法）有效地近似了海森矩阵预条件子，从而在PINNs中取得了突破性的性能：在10个具有挑战性的偏微分方程（PDE）基准测试中获得了最前沿的结果，包括首次成功地应用到雷诺数高达10,000的湍流流动中，相较于现有方法，准确度提高了2-10倍。我们还引入了一种新的梯度对齐分数，它将余弦相似性推广到多个梯度，为分析优化动力学提供了实用工具。我们的发现为理解并解决梯度冲突建立了框架，具有广泛的应用于优化科学计算之外的领域的重要意义。 

---
# RPGBENCH: Evaluating Large Language Models as Role-Playing Game Engines 

**Title (ZH)**: RPGBENCH：评估作为角色扮演游戏引擎的大型语言模型 

**Authors**: Pengfei Yu, Dongming Shen, Silin Meng, Jaewon Lee, Weisu Yin, Andrea Yaoyun Cui, Zhenlin Xu, Yi Zhu, Xingjian Shi, Mu Li, Alex Smola  

**Link**: [PDF](https://arxiv.org/pdf/2502.00595)  

**Abstract**: We present RPGBench, the first benchmark designed to evaluate large language models (LLMs) as text-based role-playing game (RPG) engines. RPGBench comprises two core tasks: Game Creation (GC) and Game Simulation (GS). In GC, an LLM must craft a valid and playable RPG world using a structured event-state representation, ensuring logical coherence and proper termination conditions. In GS, the LLM simulates interactive gameplay across multiple rounds while consistently updating states and enforcing game rules. To comprehensively assess performance, RPGBench integrates objective and subjective evaluation methodologies. Objective measures verify adherence to event mechanics and check variable updates without requiring human intervention. Subjective measures, such as content interestingness, action quality, and role-playing capability, are evaluated via an LLM-as-a-judge framework, where a strong LLM grades each candidate's outputs. Empirical results demonstrate that state-of-the-art LLMs can produce engaging stories but often struggle to implement consistent, verifiable game mechanics, particularly in long or complex scenarios. By combining structured, rule-based assessments with LLM-based judgments, RPGBench provides a new standard for evaluating how well LLMs can balance creativity, coherence, and complexity in text-based RPGs, opening avenues for more immersive and controllable interactive storytelling. 

**Abstract (ZH)**: 我们介绍了RPGBench，这是首个用于评估大型语言模型（LLM）作为文本基础的角色扮演游戏（RPG）引擎的基准测试。RPGBench 包含两个核心任务：游戏创建（Game Creation, GC）和游戏模拟（Game Simulation, GS）。在 GC 任务中，LLM 必须使用结构化事件状态表示法来创造一个有效且可玩的 RPG 世界，并确保逻辑连贯和适当的结束条件。在 GS 任务中，LLM 需要在多轮互动游戏过程中持续更新状态并执行游戏规则。为了全面评估性能，RPGBench 结合了客观和主观的评估方法。客观度量方法验证对事件机制的遵循和变量更新，无需人工干预。主观度量方法，如内容趣味性、操作质量以及角色扮演能力，通过 LLM 作为裁判的框架进行评估，其中强大的 LLM 会对每个候选者的输出进行评分。实验证据表明，最先进的 LLM 能够生成引人入胜的故事，但往往难以实现一致的、可验证的游戏机制，特别是在长篇或复杂情节中。通过结合结构化、基于规则的评估方法与基于 LLM 的评判，RPGBench 提供了一种新的标准，用于评估 LLM 在文本基础 RPG 中平衡创造力、连贯性和复杂性的能力，从而开辟了更沉浸式和可控的互动叙事探索途径。 

---
# Fast Vision Mamba: Pooling Spatial Dimensions for Accelerated Processing 

**Title (ZH)**: 快速视觉蜂鸟：池化空间维度以加速处理 

**Authors**: Saarthak Kapse, Robin Betz, Srinivasan Sivanandan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00594)  

**Abstract**: State Space Models (SSMs) with selective scan (Mamba) have been adapted into efficient vision models. Mamba, unlike Vision Transformers, achieves linear complexity for token interactions through a recurrent hidden state process. This sequential processing is enhanced by a parallel scan algorithm, which reduces the computational time of recurrent steps from $L$ sequential steps to $log(L)$ parallel steps with respect to the number of input tokens ($L$). In this work, we propose Fast Vision Mamba (FastVim), that further reduces the computational time of the SSM block by reducing the number of recurrent steps in Vision Mamba models while still retaining model performance. By alternately pooling tokens along image dimensions across Mamba blocks, we obtain a 2$\times$ reduction in the number of parallel steps in SSM block. Our model offers up to $72.5\%$ speedup in inference speed compared to baseline Vision Mamba models on high resolution (2048$\times$2048) images. Our experiments demonstrate state-of-the-art performance with dramatically improved throughput in a range of tasks such as image classification, cell perturbation prediction, segmentation, and object detection. Code is made available at this https URL 

**Abstract (ZH)**: 状态空间模型（SSMs）结合选择性扫描（Mamba）方法已被改编成高效的视觉模型。与视觉变换器（Vision Transformers）不同，Mamba 通过递归隐藏状态过程实现了线性复杂度的 token 交互。这种顺序处理通过并行扫描算法得到增强，将递归步骤的计算时间从 $L$ 个顺序步骤减少到相对于输入 token 数量 $L$ 的 $\log(L)$ 个并行步骤。在本文中，我们提出了 Fast Vision Mamba（FastVim），该模型通过减少 Vision Mamba 模型中的递归步骤数量，进一步减少了 SSM 块的计算时间，同时保留了模型性能。通过在 Mamba 块中交替沿图像维度池化 token，我们减少了 SSM 块中的并行步骤数量，获得了一半的减量。在高分辨率（2048×2048）图像上与基准 Vision Mamba 模型相比，我们的模型提供了高达 72.5% 的推理速度提升。我们的实验表明，在图像分类、细胞干扰预测、分割和物体检测等多种任务中，我们的模型展示了最先进的性能并显著提高了通量。代码可在以下网址获取：this https URL 

---
# Robust Knowledge Distillation in Federated Learning: Counteracting Backdoor Attacks 

**Title (ZH)**: 联邦学习中稳健的知识蒸馏：对抗后门攻击 

**Authors**: Ebtisaam Alharbi, Leandro Soriano Marcolino, Qiang Ni, Antonios Gouglidis  

**Link**: [PDF](https://arxiv.org/pdf/2502.00587)  

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple devices while preserving data privacy. However, it remains susceptible to backdoor attacks, where malicious participants can compromise the global model. Existing defence methods are limited by strict assumptions on data heterogeneity (Non-Independent and Identically Distributed data) and the proportion of malicious clients, reducing their practicality and effectiveness. To overcome these limitations, we propose Robust Knowledge Distillation (RKD), a novel defence mechanism that enhances model integrity without relying on restrictive assumptions. RKD integrates clustering and model selection techniques to identify and filter out malicious updates, forming a reliable ensemble of models. It then employs knowledge distillation to transfer the collective insights from this ensemble to a global model. Extensive evaluations demonstrate that RKD effectively mitigates backdoor threats while maintaining high model performance, outperforming current state-of-the-art defence methods across various scenarios. 

**Abstract (ZH)**: 联邦学习（FL）能够在保护数据隐私的同时，实现多个设备之间的协作模型训练。然而，它仍然容易受到后门攻击，恶意参与者可以破坏全局模型。现有的防御方法受限于对数据异质性（非独立同分布数据）及其恶意客户端比例的严格假设，这减少了其实用性和有效性。为了克服这些限制，我们提出了一种新的防御机制——鲁棒知识蒸馏（RKD），该机制能够在不依赖严格假设的情况下增强模型的完整性。RKD结合了聚类和模型选择技术，以识别并过滤恶意更新，从而形成一个可靠的模型组。随后，它利用知识蒸馏将该组的集体见解转移到全局模型中。广泛的研究评估表明，RKD能够有效地减轻后门威胁，同时保持高模型性能，并在各种场景中优于当前最先进的防御方法。 

---
# Defense Against the Dark Prompts: Mitigating Best-of-N Jailbreaking with Prompt Evaluation 

**Title (ZH)**: 抵抗黑暗提示：通过提示评估减轻最佳模型破解的影响 

**Authors**: Stuart Armstrong, Matija Franklin, Connor Stevens, Rebecca Gorman  

**Link**: [PDF](https://arxiv.org/pdf/2502.00580)  

**Abstract**: Recent work showed Best-of-N (BoN) jailbreaking using repeated use of random augmentations (such as capitalization, punctuation, etc) is effective against all major large language models (LLMs). We have found that $100\%$ of the BoN paper's successful jailbreaks (confidence interval $[99.65\%, 100.00\%]$) and $99.8\%$ of successful jailbreaks in our replication (confidence interval $[99.28\%, 99.98\%]$) were blocked with our Defense Against The Dark Prompts (DATDP) method. The DATDP algorithm works by repeatedly utilizing an evaluation LLM to evaluate a prompt for dangerous or manipulative behaviors--unlike some other approaches, DATDP also explicitly looks for jailbreaking attempts--until a robust safety rating is generated. This success persisted even when utilizing smaller LLMs to power the evaluation (Claude and LLaMa-3-8B-instruct proved almost equally capable). These results show that, though language models are sensitive to seemingly innocuous changes to inputs, they seem also capable of successfully evaluating the dangers of these inputs. Versions of DATDP can therefore be added cheaply to generative AI systems to produce an immediate significant increase in safety. 

**Abstract (ZH)**: 最近的研究表明，通过使用重复的随机增强（如大写、标点等）的最好-N（BoN）破解方法能够有效针对所有主要的大语言模型（LLMs）。我们发现，BoN论文中100%的成功破解实例（置信区间：99.65%至100.00%）和我们在重复实验中99.8%的成功破解实例（置信区间：99.28%至99.98%）均被我们提出的对抗黑暗提示的防御机制（Defense Against The Dark Prompts, DATDP）方法所阻止。DATDP算法通过反复利用评估LLM来评估提示是否存在危险或操控行为——与其他方法不同，DATDP明确寻找破解尝试——直到生成稳健的安全评分。即使使用较小的LLM（如Claude和LLaMa-3-8B-instruct）来执行评估工作，这一成功依然持续有效。这些结果表明，尽管语言模型对输入的细微变化非常敏感，但它们似乎也能够成功评估这些输入的危险性。因此，可以通过经济地将DATDP的版本添加到生成型AI系统中，从而立即显著提高安全性。 

---
# Generating crossmodal gene expression from cancer histopathology improves multimodal AI predictions 

**Title (ZH)**: 从癌症组织病理学生成跨模态基因表达以改进多模态AI预测 

**Authors**: Samiran Dey, Christopher R.S. Banerji, Partha Basuchowdhuri, Sanjoy K. Saha, Deepak Parashar, Tapabrata Chakraborti  

**Link**: [PDF](https://arxiv.org/pdf/2502.00568)  

**Abstract**: Emerging research has highlighted that artificial intelligence based multimodal fusion of digital pathology and transcriptomic features can improve cancer diagnosis (grading/subtyping) and prognosis (survival risk) prediction. However, such direct fusion for joint decision is impractical in real clinical settings, where histopathology is still the gold standard for diagnosis and transcriptomic tests are rarely requested, at least in the public healthcare system. With our novel diffusion based crossmodal generative AI model PathoGen, we show that genomic expressions synthesized from digital histopathology jointly predicts cancer grading and patient survival risk with high accuracy (state-of-the-art performance), certainty (through conformal coverage guarantee) and interpretability (through distributed attention maps). PathoGen code is available for open use by the research community through GitHub at this https URL. 

**Abstract (ZH)**: 新兴研究表明，基于人工智能的多模态融合，结合数字病理学和转录组特征，可以提高癌症诊断（分级/亚型分类）和预后（生存风险预测）的准确性。然而，在实际临床环境中，直接融合联合决策是不切实际的，因为组织病理学仍然是诊断的金标准，而转录组检测在公共医疗保健系统中很少被要求。借助我们新颖的基于扩散的跨模态生成AI模型PathoGen，我们证明了从数字组织病理学推断的基因表达能够以高精度（前沿性能）、高确定性（通过置信覆盖保证）和高可解释性（通过分布式注意力图）预测癌症分级和患者生存风险。PathoGen代码可通过GitHub（请访问此链接：[这里]）对研究社区开放使用。 

---
# Lessons for GenAI Literacy From a Field Study of Human-GenAI Augmentation in the Workplace 

**Title (ZH)**: 来自职场中人类与GenAI增强研究的元认知技能启示 

**Authors**: Aditya Johri, Johannes Schleiss, Nupoor Ranade  

**Link**: [PDF](https://arxiv.org/pdf/2502.00567)  

**Abstract**: Generative artificial intelligence (GenAI) is increasingly becoming a part of work practices across the technology industry and being used across a range of industries. This has necessitated the need to better understand how GenAI is being used by professionals in the field so that we can better prepare students for the workforce. An improved understanding of the use of GenAI in practice can help provide guidance on the design of GenAI literacy efforts including how to integrate it within courses and curriculum, what aspects of GenAI to teach, and even how to teach it. This paper presents a field study that compares the use of GenAI across three different functions - product development, software engineering, and digital content creation - to identify how GenAI is currently being used in the industry. This study takes a human augmentation approach with a focus on human cognition and addresses three research questions: how is GenAI augmenting work practices; what knowledge is important and how are workers learning; and what are the implications for training the future workforce. Findings show a wide variance in the use of GenAI and in the level of computing knowledge of users. In some industries GenAI is being used in a highly technical manner with deployment of fine-tuned models across domains. Whereas in others, only off-the-shelf applications are being used for generating content. This means that the need for what to know about GenAI varies, and so does the background knowledge needed to utilize it. For the purposes of teaching and learning, our findings indicated that different levels of GenAI understanding needs to be integrated into courses. From a faculty perspective, the work has implications for training faculty so that they are aware of the advances and how students are possibly, as early adopters, already using GenAI to augment their learning practices. 

**Abstract (ZH)**: 生成型人工智能（GenAI）正日益成为信息技术行业中工作实践的一部分，并且跨行业应用越来越广泛。这促使我们更深入地了解专业人士在实际工作中如何使用GenAI，以便更好地为学生做好准备，进入劳动力市场。通过更深入地了解GenAI的实际应用，我们可以为GenAI素养教育的设计提供指导，包括如何将其整合到课程和课程体系中，需要教授哪些方面以及如何教授GenAI等。本文呈现了一项实地研究，比较了产品开发、软件工程和数字内容创作三个不同职能中GenAI的应用情况，以识别GenAI在行业中的当前使用情况。该研究采取了增强人类能力的方法，重点关注人类认知，并提出了三个研究问题：GenAI如何增强工作实践？哪些知识是重要的，工作人员是如何学习的？这对培训未来劳动力有何影响？研究发现，GenAI的应用范围和用户的技术水平差异很大。在某些行业中，GenAI以技术密集型的方式使用，跨越不同领域部署微调模型。而在其他行业中，仅使用现成的应用程序生成内容。这意味着关于GenAI需要了解哪些内容以及利用它所需的背景知识各不相同。对于教学而言，我们的发现表明需要根据不同水平的GenAI理解将其整合到课程中。从教师的角度来看，这项工作对培训教师具有重要意义，使他们了解最新进展，并认识到学生作为早期采用者，已经在使用GenAI来增强自己的学习实践。 

---
# Milmer: a Framework for Multiple Instance Learning based Multimodal Emotion Recognition 

**Title (ZH)**: Milmer：基于多模态情感识别的多个实例学习框架 

**Authors**: Zaitian Wang, Jian He, Yu Liang, Xiyuan Hu, Tianhao Peng, Kaixin Wang, Jiakai Wang, Chenlong Zhang, Weili Zhang, Shuang Niu, Xiaoyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.00547)  

**Abstract**: Emotions play a crucial role in human behavior and decision-making, making emotion recognition a key area of interest in human-computer interaction (HCI). This study addresses the challenges of emotion recognition by integrating facial expression analysis with electroencephalogram (EEG) signals, introducing a novel multimodal framework-Milmer. The proposed framework employs a transformer-based fusion approach to effectively integrate visual and physiological modalities. It consists of an EEG preprocessing module, a facial feature extraction and balancing module, and a cross-modal fusion module. To enhance visual feature extraction, we fine-tune a pre-trained Swin Transformer on emotion-related datasets. Additionally, a cross-attention mechanism is introduced to balance token representation across modalities, ensuring effective feature integration. A key innovation of this work is the adoption of a multiple instance learning (MIL) approach, which extracts meaningful information from multiple facial expression images over time, capturing critical temporal dynamics often overlooked in previous studies. Extensive experiments conducted on the DEAP dataset demonstrate the superiority of the proposed framework, achieving a classification accuracy of 96.72% in the four-class emotion recognition task. Ablation studies further validate the contributions of each module, highlighting the significance of advanced feature extraction and fusion strategies in enhancing emotion recognition performance. Our code are available at this https URL. 

**Abstract (ZH)**: 情绪在人类行为和决策中扮演着至关重要的角色，因此情绪识别成为人机交互（HCI）领域的一个关键研究方向。本研究通过将面部表情分析与脑电信号（EEG）结合，提出了一种名为Milmer的新颖多模态框架，以应对情绪识别的挑战。该框架采用基于变换器的方法，有效地整合了视觉和生理模态信息。具体而言，该框架包括一个EEG预处理模块、一个面部特征提取与平衡模块以及一个跨模态融合模块。为增强视觉特征提取，我们选用预训练的Swin Transformer对情绪相关数据集进行微调。此外，引入了跨注意力机制来平衡各模态的令牌表示，确保有效的特征融合。本研究的一个关键创新是采用了多次实例学习（MIL）方法，该方法能够从多个随时间变化的面部表情图像中提取有意义的信息，捕捉前人研究中往往忽略的关键时间动态特性。在DEAP数据集上进行的大量实验结果表明，所提出的框架在四类情绪识别任务中达到了96.72%的分类准确率。进一步的消融研究表明，各模块的贡献显著，突显了先进特征提取和融合策略在提升情绪识别性能方面的重要性。我们的代码可在以下链接获取：[该链接]。 

---
# Integrating Frequency Guidance into Multi-source Domain Generalization for Bearing Fault Diagnosis 

**Title (ZH)**: 将频率指导集成到多源领域泛化中的轴承故障诊断中 

**Authors**: Xiaotong Tu, Chenyu Ma, Qingyao Wu, Yinhao Liu, Hongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00545)  

**Abstract**: Recent generalizable fault diagnosis researches have effectively tackled the distributional shift between unseen working conditions. Most of them mainly focus on learning domain-invariant representation through feature-level methods. However, the increasing numbers of unseen domains may lead to domain-invariant features contain instance-level spurious correlations, which impact the previous models' generalizable ability. To address the limitations, we propose the Fourier-based Augmentation Reconstruction Network, namely this http URL methods are motivated by the observation that the Fourier phase component and amplitude component preserve different semantic information of the signals, which can be employed in domain augmentation techniques. The network comprises an amplitude spectrum sub-network and a phase spectrum sub-network, sequentially reducing the discrepancy between the source and target domains. To construct a more robust generalized model, we employ a multi-source domain data augmentation strategy in the frequency domain. Specifically, a Frequency-Spatial Interaction Module (FSIM) is introduced to handle global information and local spatial features, promoting representation learning between the two sub-networks. To refine the decision boundary of our model output compared to conventional triplet loss, we propose a manifold triplet loss to contribute to generalization. Through extensive experiments on the CWRU and SJTU datasets, FARNet demonstrates effective performance and achieves superior results compared to current cross-domain approaches on the benchmarks. 

**Abstract (ZH)**: 近年来，可泛化的故障诊断研究有效应对了未见工况下的数据分布变化。大多数研究主要通过特征级方法学习域不变的表示。然而，未见域的数量不断增加可能导致域不变特征中包含实例级的虚假相关性，从而影响先前模型的泛化能力。为解决这些问题，我们提出了基于傅里叶变换的增强重建网络（Fourier-based Augmentation Reconstruction Network，简称FARNet）。该方法受到观察到的傅里叶相位分量和幅度分量保存信号不同语义信息的启发，可以用于域增强技术。网络由幅度谱子网络和相位谱子网络组成，逐步减少源域和目标域之间的差异。为了构造一个更稳健的泛化模型，我们在频域中采用了多源域数据增强策略。具体来说，我们引入了频域空间交互模块（Frequency-Spatial Interaction Module，简称FSIM）来处理全局信息和局部空间特征，促进两个子网络之间的表示学习。为了相比 conventional triplet loss 改善模型输出的决策边界从而提高泛化能力，我们提出了流形三重损失。通过在 CWRU 和 SJTU 数据集上的广泛实验，FARNet 展现了有效性能，并且在基准测试中达到了优于当前跨域方法的结果。 

---
# Generic Multimodal Spatially Graph Network for Spatially Embedded Network Representation Learning 

**Title (ZH)**: 通用的多模态空间图网络，用于空间嵌入网络表示学习 

**Authors**: Xudong Fan, Jürgen Hackl  

**Link**: [PDF](https://arxiv.org/pdf/2502.00530)  

**Abstract**: Spatially embedded networks (SENs) represent a special type of complex graph, whose topologies are constrained by the networks' embedded spatial environments. The graph representation of such networks is thereby influenced by the embedded spatial features of both nodes and edges. Accurate network representation of the graph structure and graph features is a fundamental task for various graph-related tasks. In this study, a Generic Multimodal Spatially Graph Convolutional Network (GMu-SGCN) is developed for efficient representation of spatially embedded networks. The developed GMu-SGCN model has the ability to learn the node connection pattern via multimodal node and edge features. In order to evaluate the developed model, a river network dataset and a power network dataset have been used as test beds. The river network represents the naturally developed SENs, whereas the power network represents a man-made network. Both types of networks are heavily constrained by the spatial environments and uncertainties from nature. Comprehensive evaluation analysis shows the developed GMu-SGCN can improve accuracy of the edge existence prediction task by 37.1\% compared to a GraphSAGE model which only considers the node's position feature in a power network test bed. Our model demonstrates the importance of considering the multidimensional spatial feature for spatially embedded network representation. 

**Abstract (ZH)**: 以下是将您提供的内容翻译成中文后的版本，符合学术规范：

位置嵌入网络（Spatially Embedded Networks, SENs）是一类特殊的复杂网络，其拓扑结构受到网络嵌入的空间环境的约束。这类网络的图表示形式受节点和边嵌入的空间特征的影响。准确地表示图结构和图特征是各种图相关任务中的基础任务。本研究开发了一种通用多模态空间图卷积网络（GMu-SGCN），用于高效表示位置嵌入网络。所开发的GMu-SGCN模型能够通过多模态节点和边特征学习节点连接模式。为评估所开发的模型，使用了河流网络数据集和电力网络数据集作为试验平台。河流网络代表了自然发展的SENs，而电力网络则代表了人工构建的网络。这两种类型的网络均受到自然环境和不确定性的影响较大。综合评估分析表明，与仅考虑节点位置特征的GraphSAGE模型相比，所开发的GMu-SGCN在电力网络试验平台上将边存在预测任务的准确性提高了37.1%。本研究表明，在位置嵌入网络表示中考虑多维度空间特征的重要性。 

---
# Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning 

**Title (ZH)**: 以内部分布与自我一致性桥梁构建有效高效的大语言模型推理 

**Authors**: Zhi Zhou, Tan Yuhao, Zenan Li, Yuan Yao, Lan-Zhe Guo, Xiaoxing Ma, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.00511)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning capabilities. However, single-shot inference often yields unreliable results for complex reasoning tasks, leading researchers to explore multiple reasoning paths through methods such as perplexity and self-consistency. In this paper, we present the first theoretical error decomposition analysis of these techniques, breaking down their error into estimation error and model error. Our analysis reveals a fundamental trade-off: perplexity methods suffer from substantial model error due to the absence of a proper consistency function, while self-consistency exhibits high estimation error due to a slow error convergence rate. To overcome these limitations, we propose Reasoning-Pruning Perplexity Consistency (RPC). This approach combines Perplexity Consistency, which seamlessly integrates LLM perplexity with self-consistency, and Reasoning Pruning, which eliminates low-probability reasoning paths to effectively prevent the degeneration of estimation error reduction. Theoretical analysis demonstrates that RPC not only accelerates the convergence rate of estimation error to an exponential level but also holds strong potential for further reducing model error. Extensive empirical evaluations on seven benchmark datasets confirm that RPC can significantly improve reasoning performance, sample efficiency, and confidence reliability. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）展示了出色的推理能力。然而，单次推理往往在复杂推理任务中产生不可靠的结果，促使研究者通过 perplexity 和自一致性等方法探索多种推理路径。本文首先提供了这些技术的第一种理论误差分解分析，将它们的误差分解为估计误差和模型误差。我们的分析揭示了一个基本的权衡：由于缺乏适当的连贯性函数，perplexity 方法遭受了严重的模型误差；而自一致性则由于误差收敛速度缓慢而表现出高的估计误差。为克服这些局限，我们提出了推理剪枝 perplexity 连贯性（RPC）方法。该方法结合了 perplexity 连贯性，它无缝地将 LLM 的 perplexity 与自一致性相结合，以及推理剪枝，它通过消除低概率推理路径来有效防止估计误差减少的退化。理论分析表明，RPC 不仅将估计误差的收敛速度加速到指数级别，还具有进一步减少模型误差的巨大潜力。在七个基准数据集上的广泛实证评估证实，RPC 可显著提高推理性能、样本效率和置信可靠性。 

---
# A statistically consistent measure of Semantic Variability using Language Models 

**Title (ZH)**: 使用语言模型的一种统计上一致的语义变异性度量方法 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00507)  

**Abstract**: To address the issue of variability in the output generated by a language model, we present a measure of semantic variability that is statistically consistent under mild assumptions. This measure, denoted as semantic spectral entropy, is a easy to implement algorithm that requires just off the shelf language models. We put very few restrictions on the language models and we have shown in a clear simulation studies that such method can generate accurate metric despite randomness that arise from the language models. 

**Abstract (ZH)**: 为了解决语言模型生成输出变异性的难题，我们提出了一种在温和假设下统计一致性的语义变异度量方法。这种方法被称为语义频谱熵，是一种易于实现的算法，只需使用现成的语言模型即可。我们对语言模型施加了很少的限制，并通过清晰的模拟研究表明，即使存在由语言模型引起的随机性，这种方法也能生成准确的度量指标。 

---
# Optimizing Feature Selection in Causal Inference: A Three-Stage Computational Framework for Unbiased Estimation 

**Title (ZH)**: 在因果推断中优化特征选择：无偏估计的三阶段计算框架 

**Authors**: Tianyu Yang, Md. Noor-E-Alam  

**Link**: [PDF](https://arxiv.org/pdf/2502.00501)  

**Abstract**: Feature selection is an important but challenging task in causal inference for obtaining unbiased estimates of causal quantities. Properly selected features in causal inference not only significantly reduce the time required to implement a matching algorithm but, more importantly, can also reduce the bias and variance when estimating causal quantities. When feature selection techniques are applied in causal inference, the crucial criterion is to select variables that, when used for matching, can achieve an unbiased and robust estimation of causal quantities. Recent research suggests that balancing only on treatment-associated variables introduces bias while balancing on spurious variables increases variance. To address this issue, we propose an enhanced three-stage framework that shows a significant improvement in selecting the desired subset of variables compared to the existing state-of-the-art feature selection framework for causal inference, resulting in lower bias and variance in estimating the causal quantity. We evaluated our proposed framework using a state-of-the-art synthetic data across various settings and observed superior performance within a feasible computation time, ensuring scalability for large-scale datasets. Finally, to demonstrate the applicability of our proposed methodology using large-scale real-world data, we evaluated an important US healthcare policy related to the opioid epidemic crisis: whether opioid use disorder has a causal relationship with suicidal behavior. 

**Abstract (ZH)**: 特征选择是因果推理中一个关键但具有挑战性的任务，旨在获得因果量的无偏估计。在因果推理中，正确选择的特征不仅能显著减少实施匹配算法所需的时间，而且更重要的是，还能降低估计因果量时的偏差和方差。当在因果推理中应用特征选择技术时，关键标准是选择那些在匹配时能实现无偏且稳健的因果量估计的变量。最新研究表明，仅根据与治疗相关变量进行匹配会引入偏差，而根据虚假变量进行匹配则会增加方差。为解决这一问题，我们提出了一种增强的三阶段框架，在因果推理中根据现有最先进的特征选择框架选择所需变量方面表现出显著改进，从而在估计因果量时降低偏差和方差。我们通过使用前沿的合成数据进行了各种设置下实验，并在可行的计算时间内观察到优越的性能，确保了该方法在大规模数据集中的可扩展性。最后，我们利用大规模的现实世界数据验证了我们提出方法的有效性，并评估了一个重要的美国医疗保健政策问题，即是否存在由于阿片类药物滥用障碍而引发自杀行为的因果关系。 

---
# Video Latent Flow Matching: Optimal Polynomial Projections for Video Interpolation and Extrapolation 

**Title (ZH)**: 视频潜流匹配：视频插值与外推的最佳多项式投影 

**Authors**: Yang Cao, Zhao Song, Chiwun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00500)  

**Abstract**: This paper considers an efficient video modeling process called Video Latent Flow Matching (VLFM). Unlike prior works, which randomly sampled latent patches for video generation, our method relies on current strong pre-trained image generation models, modeling a certain caption-guided flow of latent patches that can be decoded to time-dependent video frames. We first speculate multiple images of a video are differentiable with respect to time in some latent space. Based on this conjecture, we introduce the HiPPO framework to approximate the optimal projection for polynomials to generate the probability path. Our approach gains the theoretical benefits of the bounded universal approximation error and timescale robustness. Moreover, VLFM processes the interpolation and extrapolation abilities for video generation with arbitrary frame rates. We conduct experiments on several text-to-video datasets to showcase the effectiveness of our method. 

**Abstract (ZH)**: 这篇文章探讨了一种高效视频建模过程，称为Video Latent Flow Matching（VLFM）。与之前的工作不同，之前的工作是随机采样视频中的潜在补丁进行生成，我们的方法依赖于当前强大的预训练图像生成模型，通过建模由特定描述符引导的时间依赖的潜在补丁流，这些补帖可以被解码为时间相关的视频帧。我们首先假设视频中的多个图像在某种程度上相对于时间是可微的，存在于某个潜在空间中。基于这一假设，我们引入HiPPO框架来近似多项式的最佳投影，生成概率路径。我们的方法获得了有界通用逼近误差的理论优势以及时间尺度鲁棒性。此外，VLFM能够处理不同帧率下视频生成的插值和外推能力。我们在几个文本到视频的数据集上进行了实验，展示了我们方法的有效性。 

---
# Looking into the Future of Health-Care Services: Can Life-Like Agents Change the Future of Health-Care Services? 

**Title (ZH)**: 探究医疗服务的未来：生活化代理能否改变医疗服务的未来？ 

**Authors**: Mohammad Saleh Torkestani, Robert Davis, Abdolhossein Sarrafzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2502.00495)  

**Abstract**: Time constraints on doctor patient interaction and restricted access to specialists under the managed care system led to increasingly referring to computers as a medical information source and a self-health-care management tool. However, research show that less than 40% of information seekers indicated that online information helped them to make a decision about their health. Searching multiple web sites that need basic computer skills, lack of interaction and no face to face interaction in most search engines and some social issues, led us to develop a specialized life-like agent that would overcome mentioned problems. 

**Abstract (ZH)**: 时间限制和受管理医疗体系下专家资源受限导致医生与患者互动时间缩短，且患者难以获得专科医生的及时咨询，这促使人们越来越多地将计算机视为医疗信息来源和自我健康管理工具。然而，研究显示，仅有不到40%的寻求信息者认为在线信息帮助他们做出了关于自身健康的决策。在需要基本计算机技能、缺乏互动和大多数搜索引擎缺乏面对面交流的网站上进行多网站搜索，以及一些社会问题，促使我们开发了一个专门的、拟人化的代理，以克服上述问题。 

---
# Data Overvaluation Attack and Truthful Data Valuation 

**Title (ZH)**: 数据过估攻击与真实数据估值 

**Authors**: Shuyuan Zheng, Sudong Cai, Chuan Xiao, Yang Cao, Jainbin Qin, Masatoshi Yoshikawa, Makoto Onizuka  

**Link**: [PDF](https://arxiv.org/pdf/2502.00494)  

**Abstract**: In collaborative machine learning, data valuation, i.e., evaluating the contribution of each client' data to the machine learning model, has become a critical task for incentivizing and selecting positive data contributions. However, existing studies often assume that clients engage in data valuation truthfully, overlooking the practical motivation for clients to exaggerate their contributions. To unlock this threat, this paper introduces the first data overvaluation attack, enabling strategic clients to have their data significantly overvalued. Furthermore, we propose a truthful data valuation metric, named Truth-Shapley. Truth-Shapley is the unique metric that guarantees some promising axioms for data valuation while ensuring that clients' optimal strategy is to perform truthful data valuation. Our experiments demonstrate the vulnerability of existing data valuation metrics to the data overvaluation attack and validate the robustness and effectiveness of Truth-Shapley. 

**Abstract (ZH)**: 在协作机器学习中，数据评价，即评估每个客户端的数据对机器学习模型的贡献，已成为激励和选择积极数据贡献的关键任务。然而，现有研究通常假设客户端会真实地进行数据评价，忽视了客户端夸大其贡献的实用动机。为了解决这一问题，本文首次介绍了数据过评价攻击，在此攻击下，战略性客户端能够使其数据显著地被高估。此外，我们提出了一种真实数据评价指标，名为Truth-Shapley。Truth-Shapley 是唯一保证数据评价中某些具有前景的公理，并确保客户端的最佳策略是进行真实数据评价的指标。我们的实验表明，现有数据评价指标对数据过评价攻击的脆弱性，并验证了 Truth-Shapley 的稳健性和有效性。 

---
# Enhance Learning Efficiency of Oblique Decision Tree via Feature Concatenation 

**Title (ZH)**: 通过特征拼接提高斜向决策树的学习效率 

**Authors**: Shen-Huan Lyu, Yi-Xiao He, Yanyan Wang, Zhihao Qu, Bin Tang, Baoliu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2502.00465)  

**Abstract**: Oblique Decision Tree (ODT) separates the feature space by linear projections, as opposed to the conventional Decision Tree (DT) that forces axis-parallel splits. ODT has been proven to have a stronger representation ability than DT, as it provides a way to create shallower tree structures while still approximating complex decision boundaries. However, its learning efficiency is still insufficient, since the linear projections cannot be transmitted to the child nodes, resulting in a waste of model parameters. In this work, we propose an enhanced ODT method with Feature Concatenation (\texttt{FC-ODT}), which enables in-model feature transformation to transmit the projections along the decision paths. Theoretically, we prove that our method enjoys a faster consistency rate w.r.t. the tree depth, indicating that our method possesses a significant advantage in generalization performance, especially for shallow trees. Experiments show that \texttt{FC-ODT} can outperform the other state-of-the-art decision trees with a limited tree depth. 

**Abstract (ZH)**: 斜分决策树（Oblique Decision Tree, ODT）通过线性投影分割特征空间，不同于传统的决策树（Decision Tree, DT）强制进行轴平行分割。ODT已被证明具有比DT更强的表示能力，因为它提供了一种创建更浅树结构的同时逼近复杂决策边界的途径。然而，其学习效率仍然不足，因为线性投影不能传递给子节点，导致模型参数的浪费。在本文中，我们提出了一种增强的ODT方法——特征连接（Feature Concatenation, \texttt{FC-ODT}）方法，使得模型内部特征变换能够沿决策路径传递投影。理论上，我们证明了该方法在树深度方面享有更快的一致性率，表明该方法在泛化性能方面具有显著优势，尤其是对于浅树结构。实验结果表明，在限制的树深度下，\texttt{FC-ODT} 能够优于其他最先进的决策树方法。 

---
# AudioGenX: Explainability on Text-to-Audio Generative Models 

**Title (ZH)**: AudioGenX：文本到音频生成模型的可解释性研究 

**Authors**: Kang Hyunju, Han Geonhee, Jeong Yoonjae, Park Hogun  

**Link**: [PDF](https://arxiv.org/pdf/2502.00459)  

**Abstract**: Text-to-audio generation models (TAG) have achieved significant advances in generating audio conditioned on text descriptions. However, a critical challenge lies in the lack of transparency regarding how each textual input impacts the generated audio. To address this issue, we introduce AudioGenX, an Explainable AI (XAI) method that provides explanations for text-to-audio generation models by highlighting the importance of input tokens. AudioGenX optimizes an Explainer by leveraging factual and counterfactual objective functions to provide faithful explanations at the audio token level. This method offers a detailed and comprehensive understanding of the relationship between text inputs and audio outputs, enhancing both the explainability and trustworthiness of TAG models. Extensive experiments demonstrate the effectiveness of AudioGenX in producing faithful explanations, benchmarked against existing methods using novel evaluation metrics specifically designed for audio generation tasks. 

**Abstract (ZH)**: 文本到音频生成模型（Text-to-audio generation models, TAG）已在根据文本描述生成音频方面取得了显著进展。然而，一个关键挑战在于缺乏关于每个文本输入如何影响生成音频的透明度。为解决这一问题，我们引入了AudioGenX，这是一种可解释人工智能（Explainable AI, XAI）方法，通过突出输入令牌的重要性来为文本到音频生成模型提供解释。AudioGenX 通过利用事实性和反事实目标函数来优化解释器，在音频令牌层面提供忠实的解释。该方法提供了文本输入与音频输出之间关系的详细而全面的理解，从而增强TAG模型的可解释性和可信度。广泛开展的实验表明，AudioGenX 在生产忠实解释方面具有有效性，这些解释采用了针对音频生成任务设计的新颖评估指标进行基准测试。 

---
# Towards Privacy-aware Mental Health AI Models: Advances, Challenges, and Opportunities 

**Title (ZH)**: 面向隐私保护的心理健康AI模型：进展、挑战与机遇 

**Authors**: Aishik Mandal, Tanmoy Chakraborty, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2502.00451)  

**Abstract**: Mental illness is a widespread and debilitating condition with substantial societal and personal costs. Traditional diagnostic and treatment approaches, such as self-reported questionnaires and psychotherapy sessions, often impose significant burdens on both patients and clinicians, limiting accessibility and efficiency. Recent advances in Artificial Intelligence (AI), particularly in Natural Language Processing and multimodal techniques, hold great potential for recognizing and addressing conditions such as depression, anxiety, bipolar disorder, schizophrenia, and post-traumatic stress disorder. However, privacy concerns, including the risk of sensitive data leakage from datasets and trained models, remain a critical barrier to deploying these AI systems in real-world clinical settings. These challenges are amplified in multimodal methods, where personal identifiers such as voice and facial data can be misused. This paper presents a critical and comprehensive study of the privacy challenges associated with developing and deploying AI models for mental health. We further prescribe potential solutions, including data anonymization, synthetic data generation, and privacy-preserving model training, to strengthen privacy safeguards in practical applications. Additionally, we discuss evaluation frameworks to assess the privacy-utility trade-offs in these approaches. By addressing these challenges, our work aims to advance the development of reliable, privacy-aware AI tools to support clinical decision-making and improve mental health outcomes. 

**Abstract (ZH)**: 心理健康是一种普遍且具有严重破坏性的疾病，对社会和个人造成了巨大的成本。传统的诊断和治疗方法，如自我报告问卷和心理治疗会诊，常常给患者和临床医生带来显著的负担，从而限制了访问性和效率。近年来，特别是在自然语言处理和多模态技术方面的人工智能（AI）进展，为识别和应对抑郁症、焦虑症、双相情感障碍、精神分裂症和创伤后应激障碍等条件提供了巨大的潜力。然而，隐私问题，包括敏感数据泄露的风险，仍然是在实际临床环境中部署这些AI系统的重大障碍。这些挑战在多模态方法中更为突出，其中个人标识符，如语音和面部数据可能会被误用。本文对开发和部署用于心理健康的人工智能模型所面临的隐私挑战进行了全面而批判性的研究。我们进一步提出了一些潜在的解决方案，包括数据脱敏、合成数据生成和隐私保护模型训练，以增强实际应用中的隐私保护措施。此外，我们讨论了评估这些方法的隐私-实用性权衡的框架。通过解决这些挑战，我们的工作旨在促进可靠且具有隐私意识的AI工具的发展，以支持临床决策并改善心理健康结果。 

---
# Model-Free Predictive Control: Introductory Algebraic Calculations, and a Comparison with HEOL and ANNs 

**Title (ZH)**: 模型自由预测控制：初步代数计算及其与HEOL和ANNs的比较 

**Authors**: Cédric Join, Emmanuel Delaleau, Michel Fliess  

**Link**: [PDF](https://arxiv.org/pdf/2502.00443)  

**Abstract**: Model predictive control (MPC) is a popular control engineering practice, but requires a sound knowledge of the model. Model-free predictive control (MFPC), a burning issue today, also related to reinforcement learning (RL) in AI, is reformulated here via a linear differential equation with constant coefficients, thanks to a new perspective on optimal control combined with recent advances in the field of model-free control. It is replacing Dynamic Programming, the Hamilton-Jacobi-Bellman equation, and Pontryagin's Maximum Principle. The computing burden is low. The implementation is straightforward. Two nonlinear examples, a chemical reactor and a two tank system, are illustrating our approach. A comparison with the HEOL setting, where some expertise of the process model is needed, shows only a slight superiority of the later. A recent identification of the two tank system via a complex ANN architecture might indicate that a full modeling and the corresponding machine learning mechanism are not always necessary neither in control, nor, more generally, in AI. 

**Abstract (ZH)**: 模型预测控制（MPC）是流行的控制工程实践，但需要对模型有扎实的了解。无模型预测控制（MFPC），作为当今的研究热点，与人工智能中的强化学习（RL）相关，通过新的最优控制视角和近期模型自由控制领域的进展，被重新定义为线性常系数微分方程的形式。这种方法替代了动态规划、哈密尔顿-雅可比-贝尔曼方程以及庞特里亚金极大原则，计算负担较低，实现简便。通过两个非线性示例，一个化学反应器和两个容器系统，说明了我们的方法。与需要过程模型专业知识的HEOL设置相比，后期方法仅表现出轻微的优势。对两个容器系统的最新识别研究表明，在控制甚至更广泛的AI中，完整的建模和相应的机器学习机制可能并不总是必要的。 

---
# Compilation and Fast Model Counting beyond CNF 

**Title (ZH)**: 《超越CNF的综合与快速模型计数》 

**Authors**: Alexis de Colnet, Stefan Szeider, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00434)  

**Abstract**: Circuits in deterministic decomposable negation normal form (d-DNNF) are representations of Boolean functions that enable linear-time model counting. This paper strengthens our theoretical knowledge of what classes of functions can be efficiently transformed, or compiled, into d-DNNF. Our main contribution is the fixed-parameter tractable (FPT) compilation of conjunctions of specific constraints parameterized by incidence treewidth. This subsumes the known result for CNF. The constraints in question are all functions representable by constant-width ordered binary decision diagrams (OBDDs) for all variable orderings. For instance, this includes parity constraints and cardinality constraints with constant threshold. The running time of the FPT compilation is singly exponential in the incidence treewidth but hides large constants in the exponent. To balance that, we give a more efficient FPT algorithm for model counting that applies to a sub-family of the constraints and does not require compilation. 

**Abstract (ZH)**: 确定性分解否定范式（d-DNNF）中的电路是布尔函数的表示，能实现线性时间的概率计算。本文加强了我们对哪些类别的函数可以高效地转换或编译为d-DNNF的理论认识。我们的主要贡献是基于发生树宽度的特定约束的逻辑积的固定参数可处理（FPT）编译。这涵盖了已知的CNF结果。所述约束是所有变量排序下都能由恒定宽度有序二叉决策图（OBDD）表示的所有函数。例如，这包括奇偶约束和具有恒定阈值的计数约束。固定参数可处理编译的运行时间是发生树宽度的单指数函数，但指数中的系数较大。为平衡这一点，我们提供了一个更高效的固定参数可处理概率计算算法，适用于部分约束子集，且不需要编译。 

---
# MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization 

**Title (ZH)**: MQuant：通过全静态量化释放多模态大型语言模型的推理潜力 

**Authors**: JiangYong Yu, Sifan Zhou, Dawei Yang, Shuo Wang, Shuoyu Li, Xing Hu, Chen Xu, Zukang Xu, Changyong Shu, Zhihang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00425)  

**Abstract**: Multimodal large language models (MLLMs) have garnered widespread attention due to their ability to understand multimodal input. However, their large parameter sizes and substantial computational demands severely hinder their practical deployment and this http URL quantization is an effective way to reduce model size and inference latency, its application to MLLMs remains underexplored. In this paper, we propose MQuant, a post-training quantization (PTQ) framework designed to tackle the unique challenges of multimodal large language models (MLLMs). Conventional quantization often struggles with MLLMs because of (a) high inference latency from large visual token counts, (b) distributional disparities between visual and textual tokens, and (c) extreme outliers introduced by Hadamard-based transformations. To address these issues, MQuant introduces: Modality-Specific Static Quantization (MSQ), assigning distinct static scales for visual vs. textual tokens; Attention-Invariant Flexible Switching (AIFS), reordering tokens to preserve casual attention while eliminating expensive token-wise scale computations; Rotation Magnitude Suppression (RMS), mitigating weight outliers arising from online Hadamard rotations. On five mainstream MLLMs (including Qwen-VL, MiniCPM-V, CogVLM2), MQuant under W4A8 achieves near-floating-point accuracy (<1% degradation) while reducing inference latency by up to 30%, significantly outperforming existing PTQ baselines. Our MQuant effectively bridges the gap for efficient and accurate MLLMs inference in resource-constrained devices. Code will be released. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）因其能够理解多模态输入而引起了广泛的关注。然而，它们庞大的参数量和巨大的计算需求严重阻碍了其实际部署。量化是一种有效的方法来减少模型大小和推断延迟，但其在MLLMs中的应用仍然不足。本文我们提出了一种针对多模态大型语言模型（MLLMs）独特挑战的后训练量化（PTQ）框架，MQuant。传统的量化方法常常难以应对MLLMs，因为（a）大量视觉词元导致的高推断延迟，（b）视觉和文本词元之间的分布差异，以及（c）Hadamard变换引入的极端异常值。为了解决这些问题，MQuant 引入了以下方法：模态特定静态量化（MSQ），为视觉和文本词元分配不同的静态缩放比例；注意不变的灵活切换（AIFS），重新排列词元以保持因果注意同时消除昂贵的词元级缩放计算；旋转幅度抑制（RMS），减轻在线Hadamard旋转引起的权重异常值。在五种主流的MLLMs（包括Qwen-VL、MiniCPM-V、CogVLM2）中，MQuant 在W4A8量化下实现了接近浮点精度（<1% 的退化）的同时将推断延迟降低了高达30%，显著优于现有的PTQ基准。我们的MQuant有效地填补了资源受限设备中高效准确的MLLMs推断的空白。代码将开源。 

---
# MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents 

**Title (ZH)**: MarketSenseAI 2.0：通过LLM代理增强股票分析 

**Authors**: George Fatouros, Kostas Metaxas, John Soldatos, Manos Karathanassis  

**Link**: [PDF](https://arxiv.org/pdf/2502.00415)  

**Abstract**: MarketSenseAI is a novel framework for holistic stock analysis which leverages Large Language Models (LLMs) to process financial news, historical prices, company fundamentals and the macroeconomic environment to support decision making in stock analysis and selection. In this paper, we present the latest advancements on MarketSenseAI, driven by rapid technological expansion in LLMs. Through a novel architecture combining Retrieval-Augmented Generation and LLM agents, the framework processes SEC filings and earnings calls, while enriching macroeconomic analysis through systematic processing of diverse institutional reports. We demonstrate a significant improvement in fundamental analysis accuracy over the previous version. Empirical evaluation on S\&P 100 stocks over two years (2023-2024) shows MarketSenseAI achieving cumulative returns of 125.9% compared to the index return of 73.5%, while maintaining comparable risk profiles. Further validation on S\&P 500 stocks during 2024 demonstrates the framework's scalability, delivering a 33.8% higher Sortino ratio than the market. This work marks a significant advancement in applying LLM technology to financial analysis, offering insights into the robustness of LLM-driven investment strategies. 

**Abstract (ZH)**: MarketSenseAI 是一种新的综合股票分析框架，该框架利用大型语言模型（LLMs）处理财经新闻、历史价格、公司基本面以及宏观经济环境，以支持股票分析和选择中的决策制定。在本文中，我们介绍了由LLM技术快速发展驱动的MarketSenseAI 最新进展。通过结合检索增强生成和LLM代理的新型架构，该框架处理SEC文件和收益电话会议，并通过系统处理各种机构报告来丰富宏观经济分析。我们展示了与上一版本相比，在基本分析准确性方面的显著改进。对2023-2024年标普100指数股票的实证评估显示，MarketSenseAI 的累计回报率为125.9%，而指数回报率为73.5%，且保持相似的风险水平。进一步对2024年标普500指数股票的验证显示，该框架具有可扩展性，其索丁比率比市场高出33.8%。这项工作标志着在财务分析中应用LLM技术的一个重要进展，为LLM驱动的投资策略的稳健性提供了见解。 

---
# Causal Abstraction Learning based on the Semantic Embedding Principle 

**Title (ZH)**: 基于语义嵌入原理的因果抽象学习 

**Authors**: Gabriele D'Acunto, Fabio Massimo Zennaro, Yorgos Felekis, Paolo Di Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2502.00407)  

**Abstract**: Structural causal models (SCMs) allow us to investigate complex systems at multiple levels of resolution. The causal abstraction (CA) framework formalizes the mapping between high- and low-level SCMs. We address CA learning in a challenging and realistic setting, where SCMs are inaccessible, interventional data is unavailable, and sample data is misaligned. A key principle of our framework is $\textit{semantic embedding}$, formalized as the high-level distribution lying on a subspace of the low-level one. This principle naturally links linear CA to the geometry of the $\textit{Stiefel manifold}$. We present a category-theoretic approach to SCMs that enables the learning of a CA by finding a morphism between the low- and high-level probability measures, adhering to the semantic embedding principle. Consequently, we formulate a general CA learning problem. As an application, we solve the latter problem for linear CA; considering Gaussian measures and the Kullback-Leibler divergence as an objective. Given the nonconvexity of the learning task, we develop three algorithms building upon existing paradigms for Riemannian optimization. We demonstrate that the proposed methods succeed on both synthetic and real-world brain data with different degrees of prior information about the structure of CA. 

**Abstract (ZH)**: 结构因果模型（SCMs）使我们能够探究不同分辨率层次上的复杂系统。因果抽象（CA）框架正式化了从高分辨率层次的SCMs到低分辨率层次的SCMs之间的映射关系。我们在一个具有挑战性和现实性的环境中处理CA学习问题，其中SCMs不可访问，干预数据不可用，样本数据还错位了。我们框架中的一个关键原则是$\textit{语义嵌入}$，形式化为高分辨率层次分布位于低分辨率层次分布的子空间上。该原则自然地将线性CA与$\textit{斯坦费尔流形}$的几何结构联系起来。我们采用范畴论的方法来处理SCMs，该方法通过在低分辨率层次概率测度和高分辨率层次概率测度之间找到同构，遵循语义嵌入的原则，从而构建一般性的CA学习问题。作为应用，我们解决了线性CA的学习问题；考虑了高斯测度和相对熵作为目标函数。由于学习任务的非凸性，我们开发了三个基于黎曼优化现有范式的算法。我们证明了所提出的方法在不同程度的先验信息下，在合成数据和真实世界大脑数据中均取得了成功。 

---
# Spectro-Riemannian Graph Neural Networks 

**Title (ZH)**: 谱-黎曼图形神经网络 

**Authors**: Karish Grover, Haiyang Yu, Xiang Song, Qi Zhu, Han Xie, Vassilis N. Ioannidis, Christos Faloutsos  

**Link**: [PDF](https://arxiv.org/pdf/2502.00401)  

**Abstract**: Can integrating spectral and curvature signals unlock new potential in graph representation learning? Non-Euclidean geometries, particularly Riemannian manifolds such as hyperbolic (negative curvature) and spherical (positive curvature), offer powerful inductive biases for embedding complex graph structures like scale-free, hierarchical, and cyclic patterns. Meanwhile, spectral filtering excels at processing signal variations across graphs, making it effective in homophilic and heterophilic settings. Leveraging both can significantly enhance the learned representations. To this end, we propose Spectro-Riemannian Graph Neural Networks (CUSP) - the first graph representation learning paradigm that unifies both CUrvature (geometric) and SPectral insights. CUSP is a mixed-curvature spectral GNN that learns spectral filters to optimize node embeddings in products of constant-curvature manifolds (hyperbolic, spherical, and Euclidean). Specifically, CUSP introduces three novel components: (a) Cusp Laplacian, an extension of the traditional graph Laplacian based on Ollivier-Ricci curvature, designed to capture the curvature signals better; (b) Cusp Filtering, which employs multiple Riemannian graph filters to obtain cues from various bands in the eigenspectrum; and (c) Cusp Pooling, a hierarchical attention mechanism combined with a curvature-based positional encoding to assess the relative importance of differently curved substructures in our graph. Empirical evaluation across eight homophilic and heterophilic datasets demonstrates the superiority of CUSP in node classification and link prediction tasks, with a gain of up to 5.3% over state-of-the-art models. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

“光谱信号与曲率信号的结合能否开启图表示学习的新潜能？非欧几里得几何，特别是具有负曲率的双曲空间和具有正曲率的球面空间（Riemannian流形）提供了嵌入复杂图结构的强大归纳偏差，如无标度、层次和循环模式。与此同时，光谱过滤在处理图上信号的变化方面表现出色，使其适用于同质性和异质性设置。结合这两种方法可以显著提高学习表示的效果。为此，我们提出了一种结合光谱和曲率见解的图神经网络——Spectro-Riemannian图神经网络（CUSP），这是首个将光谱和几何学见解统一于一体的图表示学习范式。CUSP 是一种混合曲率光谱图神经网络，在常曲率流形（双曲、球面和欧几里得）的乘积中学习光谱滤波器以优化节点嵌入。具体而言，CUSP 引入了三个新颖组件：(a) Cusp拉普拉斯算子，这是一种基于奥利維耶-里奇曲率的传统图拉普拉斯算子的拓展，旨在更好地捕捉曲率信号；(b) Cusp过滤器，利用多个黎曼图滤波器从特征谱的不同带中获得线索；(c) Cusp pooling，一种结合基于曲率的位置编码的分层注意力机制，用于评估图中不同曲率子结构的相对重要性。在八个同质性和异质性数据集上的实证评估表明，CUSP 在节点分类和链接预测任务中均优于现有最先进的模型，提高了高达5.3%。 

---
# The Impact of Persona-based Political Perspectives on Hateful Content Detection 

**Title (ZH)**: 基于人格的政治视角对仇恨内容检测的影响 

**Authors**: Stefano Civelli, Pietro Bernardelle, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2502.00385)  

**Abstract**: While pretraining language models with politically diverse content has been shown to improve downstream task fairness, such approaches require significant computational resources often inaccessible to many researchers and organizations. Recent work has established that persona-based prompting can introduce political diversity in model outputs without additional training. However, it remains unclear whether such prompting strategies can achieve results comparable to political pretraining for downstream tasks. We investigate this question using persona-based prompting strategies in multimodal hate-speech detection tasks, specifically focusing on hate speech in memes. Our analysis reveals that when mapping personas onto a political compass and measuring persona agreement, inherent political positioning has surprisingly little correlation with classification decisions. Notably, this lack of correlation persists even when personas are explicitly injected with stronger ideological descriptors. Our findings suggest that while LLMs can exhibit political biases in their responses to direct political questions, these biases may have less impact on practical classification tasks than previously assumed. This raises important questions about the necessity of computationally expensive political pretraining for achieving fair performance in downstream tasks. 

**Abstract (ZH)**: 尽管使用政治多样性的内容预训练语言模型已被证明能改善下游任务的公平性，但这些方法往往需要大量的计算资源，而许多研究者和组织难以获得。最近的研究表明，基于角色的提示可以在不进行额外训练的情况下引入模型输出的政治多样性。然而，尚未清楚这些提示策略是否能达到与政治预训练相当的下游任务效果。我们通过基于角色的提示策略在多模态仇恨言论检测任务中进行研究，特别是关注在网络 meme 中的仇恨言论。我们的分析表明，当将角色映射到政治立场图谱并衡量角色一致性时，内在的政治定位与分类决策之间的相关性 surprisingly 低。值得注意的是，即使将角色明确注入更强的意识形态描述，这一缺乏相关性仍然存在。研究结果表明，尽管大语言模型（LLMs）在直接回答政治问题时可能表现出政治偏见，但这些偏见对实际分类任务的影响可能不像预期的那样大。这引发了关于在实现下游任务公平性方面是否需要代价高昂的政治预训练的重要问题。 

---
# Masked Generative Nested Transformers with Decode Time Scaling 

**Title (ZH)**: 掩码生成嵌套变压器，带解码时序缩放 

**Authors**: Sahil Goyal, Debapriya Tula, Gagan Jain, Pradeep Shenoy, Prateek Jain, Sujoy Paul  

**Link**: [PDF](https://arxiv.org/pdf/2502.00382)  

**Abstract**: Recent advances in visual generation have made significant strides in producing content of exceptional quality. However, most methods suffer from a fundamental problem - a bottleneck of inference computational efficiency. Most of these algorithms involve multiple passes over a transformer model to generate tokens or denoise inputs. However, the model size is kept consistent throughout all iterations, which makes it computationally expensive. In this work, we aim to address this issue primarily through two key ideas - (a) not all parts of the generation process need equal compute, and we design a decode time model scaling schedule to utilize compute effectively, and (b) we can cache and reuse some of the computation. Combining these two ideas leads to using smaller models to process more tokens while large models process fewer tokens. These different-sized models do not increase the parameter size, as they share parameters. We rigorously experiment with ImageNet256$\times$256 , UCF101, and Kinetics600 to showcase the efficacy of the proposed method for image/video generation and frame prediction. Our experiments show that with almost $3\times$ less compute than baseline, our model obtains competitive performance. 

**Abstract (ZH)**: 近年来，视觉生成领域取得了显著进展，生成高质量内容的能力得到了大幅提升。然而，大多数方法面临一个根本性的问题——推理计算效率的瓶颈。这些算法大多需要多次遍历一个变换器模型来生成标记或去噪输入。然而，在所有迭代过程中，模型规模保持一致，这使得计算成本高昂。在这项工作中，我们主要通过两个关键思想来解决这一问题——(a) 生成过程的各部分并不需要均等的计算资源，因此我们设计了一个解码时间和模型规模调整计划，以有效地利用计算资源；(b) 我们可以缓存并重用一些计算。将这两个想法结合，使得使用较小的模型处理更多的标记，而较大的模型处理较少的标记。这些不同大小的模型并没有增加参数量，因为它们共享参数。我们通过实验在ImageNet 256×256、UCF101和Kinetics600数据集上展示了所提出方法在图像/视频生成和帧预测中的有效性。实验结果表明，与基线方法相比，我们的模型几乎只需三分之一的计算量，就能获得具有竞争力的性能。 

---
# Latent Action Learning Requires Supervision in the Presence of Distractors 

**Title (ZH)**: 在干扰因素存在的情况下，潜在动作学习需要监督 

**Authors**: Alexander Nikulin, Ilya Zisman, Denis Tarasov, Nikita Lyubaykin, Andrei Polubarov, Igor Kiselev, Vladislav Kurenkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.00379)  

**Abstract**: Recently, latent action learning, pioneered by Latent Action Policies (LAPO), have shown remarkable pre-training efficiency on observation-only data, offering potential for leveraging vast amounts of video available on the web for embodied AI. However, prior work has focused on distractor-free data, where changes between observations are primarily explained by ground-truth actions. Unfortunately, real-world videos contain action-correlated distractors that may hinder latent action learning. Using Distracting Control Suite (DCS) we empirically investigate the effect of distractors on latent action learning and demonstrate that LAPO struggle in such scenario. We propose LAOM, a simple LAPO modification that improves the quality of latent actions by 8x, as measured by linear probing. Importantly, we show that providing supervision with ground-truth actions, as few as 2.5% of the full dataset, during latent action learning improves downstream performance by 4.2x on average. Our findings suggest that integrating supervision during Latent Action Models (LAM) training is critical in the presence of distractors, challenging the conventional pipeline of first learning LAM and only then decoding from latent to ground-truth actions. 

**Abstract (ZH)**: 近年来，由Latent Action Policies (LAPO) 开创的潜在行动学习已经展示了在仅基于观测数据上的预训练效率，这为利用网络上庞大的视频资源开展具身人工智能提供了潜力。然而，早期的研究大多集中在无干扰数据上，在这种数据中，观测之间的变化主要由真实动作解释。不幸的是，现实世界的视频包含与动作相关的干扰因素，这可能妨碍潜在动作学习。通过Distracting Control Suite (DCS)，我们实验证明干扰因素对潜在动作学习的影响，并表明LAPO在这种情况下难以应对。我们提出了一种简单的LAPO改进版本——LAOM，该版本通过线性探测方法将潜在动作的质量提高了8倍。重要的是，我们还展示出，仅在潜在动作学习过程中提供少量（约占完整数据集的2.5%）的真实动作监督，就能将下游性能平均提高4.2倍。我们的研究结果表明，在存在干扰因素的情况下，集成监督信息在潜在动作模型（LAM）的训练过程中至关重要，这挑战了先学习潜在动作模型再解码到真实动作的传统工作流程。 

---
# When End-to-End is Overkill: Rethinking Cascaded Speech-to-Text Translation 

**Title (ZH)**: 当端到端模型过于复杂时：重新思考级联语音转文本翻译 

**Authors**: Anna Min, Chenxu Hu, Yi Ren, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.00377)  

**Abstract**: Though end-to-end speech-to-text translation has been a great success, we argue that the cascaded speech-to-text translation model still has its place, which is usually criticized for the error propagation between automatic speech recognition (ASR) and machine translation (MT) models. In this paper, we explore the benefits of incorporating multiple candidates from ASR and self-supervised speech features into MT. Our analysis reveals that the primary cause of cascading errors stems from the increased divergence between similar samples in the speech domain when mapped to the text domain. By including multiple candidates and self-supervised speech features, our approach allows the machine translation model to choose the right words and ensure precise translation using various speech samples. This strategy minimizes error spread and takes advantage of large ASR and MT datasets, along with pre-trained ASR/MT models, while addressing associated issues. 

**Abstract (ZH)**: 尽管端到端的语音转文本翻译取得了巨大成功，但我们认为级联语音转文本翻译模型仍然有一定的应用空间。这种模型通常因其在自动语音识别（ASR）和机器翻译（MT）模型之间的错误传播而受到批评。在本文中，我们探讨了将来自ASR的多种候选选项和自监督语音特征整合到MT中的益处。我们的分析表明，级联错误的主要原因是语音领域相似样本在映射到文本领域时的差异性增加。通过包含多种候选选项和自监督语音特征，我们的方法允许机器翻译模型选择正确的单词，并利用各种语音样本进行精准翻译。该策略有助于减少错误传播，并充分利用大规模的ASR和MT数据集以及预训练的ASR/MT模型，同时解决相关问题。 

---
# What should an AI assessor optimise for? 

**Title (ZH)**: AI评估器应该优化什么？ 

**Authors**: Daniel Romero-Alvarado, Fernando Martínez-Plumed, José Hernández-Orallo  

**Link**: [PDF](https://arxiv.org/pdf/2502.00365)  

**Abstract**: An AI assessor is an external, ideally indepen-dent system that predicts an indicator, e.g., a loss value, of another AI system. Assessors can lever-age information from the test results of many other AI systems and have the flexibility of be-ing trained on any loss function or scoring rule: from squared error to toxicity metrics. Here we address the question: is it always optimal to train the assessor for the target metric? Or could it be better to train for a different metric and then map predictions back to the target metric? Us-ing twenty regression and classification problems with tabular data, we experimentally explore this question for, respectively, regression losses and classification scores with monotonic and non-monotonic mappings and find that, contrary to intuition, optimising for more informative met-rics is not generally better. Surprisingly, some monotonic transformations are promising. For example, the logistic loss is useful for minimis-ing absolute or quadratic errors in regression, and the logarithmic score helps maximise quadratic or spherical scores in classification. 

**Abstract (ZH)**: 人工智能评估器是一种外部的、理想情况下是独立的系统，能够预测另一个AI系统的一个指标，例如损失值。评估器可以从许多其他AI系统的测试结果中获取信息，并且具有根据任意损失函数或评分规则进行训练的灵活性：从平方误差到毒性度量。在这里我们探讨了一个问题：是否总是最好为目标指标训练评估器？或者通过训练不同的指标，然后将预测映射回目标指标，这样是否更好？

我们分别使用了二十个回归和分类问题，并基于单调和非单调映射实验性地探索相应的情况，结果发现，出乎意料的是，优化更具有信息性的指标通常并不总是更好。令人惊讶的是，一些单调变换具有前景。例如，逻辑损失对于最小化回归中的绝对误差或二次误差很有用，而对数评分可以帮助最大化分类中的二次评分或球形评分。 

---
# Do Audio-Visual Segmentation Models Truly Segment Sounding Objects? 

**Title (ZH)**: audio-visual分割模型真地能够分隔出发音对象吗？ 

**Authors**: Jia Li, Wenjie Zhao, Ziru Huang, Yunhui Guo, Yapeng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2502.00358)  

**Abstract**: Unlike traditional visual segmentation, audio-visual segmentation (AVS) requires the model not only to identify and segment objects but also to determine whether they are sound sources. Recent AVS approaches, leveraging transformer architectures and powerful foundation models like SAM, have achieved impressive performance on standard benchmarks. Yet, an important question remains: Do these models genuinely integrate audio-visual cues to segment sounding objects? In this paper, we systematically investigate this issue in the context of robust AVS. Our study reveals a fundamental bias in current methods: they tend to generate segmentation masks based predominantly on visual salience, irrespective of the audio context. This bias results in unreliable predictions when sounds are absent or irrelevant. To address this challenge, we introduce AVSBench-Robust, a comprehensive benchmark incorporating diverse negative audio scenarios including silence, ambient noise, and off-screen sounds. We also propose a simple yet effective approach combining balanced training with negative samples and classifier-guided similarity learning. Our extensive experiments show that state-of-theart AVS methods consistently fail under negative audio conditions, demonstrating the prevalence of visual bias. In contrast, our approach achieves remarkable improvements in both standard metrics and robustness measures, maintaining near-perfect false positive rates while preserving highquality segmentation performance. 

**Abstract (ZH)**: 与传统的视觉分割不同，音频-视觉分割（AVS）不仅要求模型识别和分割对象，还需确定这些对象是否为声源。近期，利用变换器架构和强大的基础模型（如SAM）的AVS方法在标准基准测试中取得了令人印象深刻的表现。然而，一个重要问题仍然存在：这些模型是否真的综合了音频-视觉线索来分割产声对象？在本文中，我们系统地探讨了这一问题在鲁棒AVS的背景下。我们的研究表明，当前方法存在一个根本性的偏见：它们倾向于主要依赖视觉显著性生成分割掩码，而不考虑音频上下文。这种偏见导致在声音缺失或无关时预测不可靠。为了解决这一挑战，我们提出了AVSBench-Robust，这是一个全面基准，包含了多样化的负音频场景，包括沉默、环境声以及离屏声。我们还提出了一种结合平衡训练与负样本和分类器引导相似性学习的简单而有效的方法。广泛的实验表明，最先进的AVS方法在负音频条件下一致性地表现不佳，揭示了视觉偏见的普遍存在。相比之下，我们的方法在标准指标和鲁棒性指标上均取得了显著的改进，同时保持了近乎完美的假阳性率和高质量的分割性能。 

---
# PM-MOE: Mixture of Experts on Private Model Parameters for Personalized Federated Learning 

**Title (ZH)**: PM-MOE: 具有私有模型参数混合专家的个性化 federated 学习 

**Authors**: Yu Feng, Yangli-ao Geng, Yifan Zhu, Zongfu Han, Xie Yu, Kaiwen Xue, Haoran Luo, Mengyang Sun, Guangwei Zhang, Meina Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.00354)  

**Abstract**: Federated learning (FL) has gained widespread attention for its privacy-preserving and collaborative learning capabilities. Due to significant statistical heterogeneity, traditional FL struggles to generalize a shared model across diverse data domains. Personalized federated learning addresses this issue by dividing the model into a globally shared part and a locally private part, with the local model correcting representation biases introduced by the global model. Nevertheless, locally converged parameters more accurately capture domain-specific knowledge, and current methods overlook the potential benefits of these parameters. To address these limitations, we propose PM-MoE architecture. This architecture integrates a mixture of personalized modules and an energy-based personalized modules denoising, enabling each client to select beneficial personalized parameters from other clients. We applied the PM-MoE architecture to nine recent model-split-based personalized federated learning algorithms, achieving performance improvements with minimal additional training. Extensive experiments on six widely adopted datasets and two heterogeneity settings validate the effectiveness of our approach. The source code is available at \url{this https URL}. 

**Abstract (ZH)**: 联邦学习（FL）因其保护隐私和协作学习的能力而引起了广泛关注。由于存在显著的统计异质性，传统的FL在跨多种数据域推广共享模型时遇到了困难。个性化联邦学习通过将模型分为全局共享部分和本地私有部分来解决这一问题，本地模型可以纠正由全局模型引入的表示偏差。然而，本地收敛的参数更准确地捕捉了领域特定的知识，而当前的方法忽视了这些参数的潜在益处。为了克服这些限制，我们提出了一种PM-MoE架构。该架构结合了个性化组件的混合以及基于能量的个性化组件去噪，允许每个客户端从其他客户端中选择有益的个性化参数。我们将PM-MoE架构应用于九种最新的模型分割为基础的个性化联邦学习算法，实现了轻微附加训练的性能提升。在六个广泛采用的数据集和两种异质性设置下的广泛实验验证了我们方法的有效性。源代码可在此处获取：\url{this https URL}。 

---
# Multi-Order Hyperbolic Graph Convolution and Aggregated Attention for Social Event Detection 

**Title (ZH)**: 多阶双曲图形卷积与聚合注意力机制在社会事件检测中的应用 

**Authors**: Yao Liu, Zhilan Liu, Tien Ping Tan, Yuxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.00351)  

**Abstract**: Social event detection (SED) is a task focused on identifying specific real-world events and has broad applications across various domains. It is integral to many mobile applications with social features, including major platforms like Twitter, Weibo, and Facebook. By enabling the analysis of social events, SED provides valuable insights for businesses to understand consumer preferences and supports public services in handling emergencies and disaster management. Due to the hierarchical structure of event detection data, traditional approaches in Euclidean space often fall short in capturing the complexity of such relationships. While existing methods in both Euclidean and hyperbolic spaces have shown promising results, they tend to overlook multi-order relationships between events. To address these limitations, this paper introduces a novel framework, Multi-Order Hyperbolic Graph Convolution with Aggregated Attention (MOHGCAA), designed to enhance the performance of SED. Experimental results demonstrate significant improvements under both supervised and unsupervised settings. To further validate the effectiveness and robustness of the proposed framework, we conducted extensive evaluations across multiple datasets, confirming its superiority in tackling common challenges in social event detection. 

**Abstract (ZH)**: 社会事件检测（SED）是一项专注于识别特定现实世界事件的任务，具有广泛的应用范围，涵盖了多个领域。它在具有社交功能的许多移动应用中至关重要，包括如Twitter、微博和Facebook等主要平台。通过使业务能够分析社会事件，SED提供了了解消费者偏好的宝贵见解，并支持公共部门应对紧急情况和灾害管理。由于事件检测数据具层次结构，传统的欧几里得空间方法往往难以捕捉此类关系的复杂性。尽管在欧几里得和双曲空间中已有方法显示出有希望的结果，但它们往往忽略了事件之间的多级关系。为解决这些局限性，本文提出了一种新的框架——多级双曲图卷积聚合注意（MOHGCAA），旨在提高SED的性能。实验结果显示，该方法在监督和无监督设置下都表现出显著的改进。为进一步验证所提出框架的有效性和鲁棒性，我们在多个数据集上进行了广泛的评估，证实了其在处理社会事件检测中的常见挑战方面的优越性。 

---
# OrcaLoca: An LLM Agent Framework for Software Issue Localization 

**Title (ZH)**: OrcaLoca：一个软件问题定位的大型语言模型代理框架 

**Authors**: Zhongming Yu, Hejia Zhang, Yujie Zhao, Hanxian Huang, Matrix Yao, Ke Ding, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.00350)  

**Abstract**: Recent developments in Large Language Model (LLM) agents are revolutionizing Autonomous Software Engineering (ASE), enabling automated coding, problem fixes, and feature improvements. However, localization -- precisely identifying software problems by navigating to relevant code sections -- remains a significant challenge. Current approaches often yield suboptimal results due to a lack of effective integration between LLM agents and precise code search mechanisms. This paper introduces OrcaLoca, an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning. Experimental results demonstrate that OrcaLoca becomes the new open-source state-of-the-art (SOTA) in function match rate (65.33%) on SWE-bench Lite. It also improves the final resolved rate of an open-source framework by 6.33 percentage points through its patch generation integration. 

**Abstract (ZH)**: 近年来，大型语言模型（LLM）代理的发展正在革新自主软件工程（ASE），使自动编码、问题修复和功能改进成为可能。然而，本地化——即通过导航到相关代码段精确识别软件问题——仍然是一个重大挑战。当前的方法由于LLM代理与精确代码搜索机制之间的有效集成不足，往往导致结果不佳。本文介绍了OrcaLoca，这是一种LLM代理框架，通过优先级调度指导动作、动作分解与相关性评分以及距离感知上下文剪枝，提高了软件问题定位的准确性。实验结果表明，OrcaLoca在SWE-bench Lite的功能匹配率上成为新的开源领先方案（65.33%）。此外，通过其修补生成集成，它还将开源框架的最终解决率提高了6.33个百分点。 

---
# Actor Critic with Experience Replay-based automatic treatment planning for prostate cancer intensity modulated radiotherapy 

**Title (ZH)**: 基于经验回放的自动治疗计划方法结合演员-评论家算法在前列腺癌调强放疗中的应用 

**Authors**: Md Mainul Abrar, Parvat Sapkota, Damon Sprouts, Xun Jia, Yujie Chi  

**Link**: [PDF](https://arxiv.org/pdf/2502.00346)  

**Abstract**: Background: Real-time treatment planning in IMRT is challenging due to complex beam interactions. AI has improved automation, but existing models require large, high-quality datasets and lack universal applicability. Deep reinforcement learning (DRL) offers a promising alternative by mimicking human trial-and-error planning.
Purpose: Develop a stochastic policy-based DRL agent for automatic treatment planning with efficient training, broad applicability, and robustness against adversarial attacks using Fast Gradient Sign Method (FGSM).
Methods: Using the Actor-Critic with Experience Replay (ACER) architecture, the agent tunes treatment planning parameters (TPPs) in inverse planning. Training is based on prostate cancer IMRT cases, using dose-volume histograms (DVHs) as input. The model is trained on a single patient case, validated on two independent cases, and tested on 300+ plans across three datasets. Plan quality is assessed using ProKnow scores, and robustness is tested against adversarial attacks.
Results: Despite training on a single case, the model generalizes well. Before ACER-based planning, the mean plan score was 6.20$\pm$1.84; after, 93.09% of cases achieved a perfect score of 9, with a mean of 8.93$\pm$0.27. The agent effectively prioritizes optimal TPP tuning and remains robust against adversarial attacks.
Conclusions: The ACER-based DRL agent enables efficient, high-quality treatment planning in prostate cancer IMRT, demonstrating strong generalizability and robustness. 

**Abstract (ZH)**: 背景：在IMRT（调强放射治疗）中实现实时治疗计划具有挑战性，因为需要处理复杂的光线相互作用。尽管人工智能提高了自动化水平，但现有模型仍需要大量高质量的数据集，并且缺乏通用适用性。深度强化学习（DRL）提供了一种有前途的选择，因为它模仿了人类的试探性计划方式。
目的：开发一种基于随机策略的DRL代理，用于高效的自动治疗计划，具有广泛的适用性和对抗性攻击的鲁棒性，并采用快速梯度符号方法（FGSM）进行测试。
方法：使用Actor-Critic带经验回放（ACER）架构，代理在逆向计划中调整治疗计划参数（TPPs）。训练基于前列腺癌的IMRT案例，以剂量体积直方图（DVHs）作为输入。模型在单个患者案例上进行训练，在两个独立案例上进行验证，并在三个数据集中包含300多个案例上进行测试。使用ProKnow评分评估计划质量，并通过对抗性攻击测试其鲁棒性。
结果：尽管只在单个案例上进行训练，但该模型仍具有良好的泛化能力。在ACER计划之前，平均计划得分为6.20±1.84；之后，93.09%的案例达到了完美的得分为9分，平均得分为8.93±0.27。代理成功优先调整最优的TPPs，并且仍然具有对抗性攻击的鲁棒性。
结论：基于ACER的DRL代理能够在前列腺癌IMRT中实现高效且高质量的治疗计划，显示出强大的泛化能力和鲁棒性。 

---
# The Composite Task Challenge for Cooperative Multi-Agent Reinforcement Learning 

**Title (ZH)**: 合作多代理强化学习中的复合任务挑战 

**Authors**: Yurui Li, Yuxuan Chen, Li Zhang, Shijian Li, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00345)  

**Abstract**: The significant role of division of labor (DOL) in promoting cooperation is widely recognized in real-world this http URL cooperative multi-agent reinforcement learning (MARL) methods have incorporated the concept of DOL to improve cooperation among this http URL, the tasks used in existing testbeds typically correspond to tasks where DOL is often not a necessary feature for achieving optimal this http URL, the full utilize of DOL concept in MARL methods remains unrealized due to the absence of appropriate this http URL enhance the generality and applicability of MARL methods in real-world scenarios, there is a necessary to develop tasks that demand multi-agent DOL and this http URL this paper, we propose a series of tasks designed to meet these requirements, drawing on real-world rules as the guidance for their this http URL guarantee that DOL and cooperation are necessary condition for completing tasks and introduce three factors to expand the diversity of proposed tasks to cover more realistic this http URL evaluate 10 cooperative MARL methods on the proposed this http URL results indicate that all baselines perform poorly on these this http URL further validate the solvability of these tasks, we also propose simplified variants of proposed this http URL results show that baselines are able to handle these simplified variants, providing evidence of the solvability of the proposed this http URL source files is available at this https URL. 

**Abstract (ZH)**: 分工（Division of Labor, DOL）在促进合作方面的重要作用在现实世界中得到了广泛认可。现有的多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）方法已将DOL的概念纳入其中，以提高智能体间的合作效果。然而，现有测试床中使用的任务通常并不需要DOL这一特性来实现最优效果，因此DOL概念在MARL方法中的充分利用仍然未能实现。为了增强MARL方法在现实世界场景中的适用性和普遍性，有必要开发需要多智能体DOL的任务。在本文中，我们提出了若干旨在满足这些需求的任务，以现实世界规则为指导设计这些任务，以确保DOL和合作是完成任务的必要条件，并通过引入三个因素来扩展任务的多样性，使其覆盖更多实际场景。我们对提出的任务评估了10种合作MARL方法，结果表明所有基线方法在这些任务上表现不佳，进一步证明了这些任务的可解性。我们还提出了所提任务的简化版本，并展示了基线方法可以处理这些简化版本，进一步证明了所提任务的可解性。相关的源代码可在本链接访问：[此处插入链接]。 

---
# UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models 

**Title (ZH)**: UGPhysics: 一个全面的本科生物理推理基准测试，用于大规模语言模型 

**Authors**: Xin Xu, Qiyun Xu, Tong Xiao, Tianhao Chen, Yuchen Yan, Jiaxin Zhang, Shizhe Diao, Can Yang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00334)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in solving complex reasoning tasks, particularly in mathematics. However, the domain of physics reasoning presents unique challenges that have received significantly less attention. Existing benchmarks often fall short in evaluating LLMs' abilities on the breadth and depth of undergraduate-level physics, underscoring the need for a comprehensive evaluation. To fill this gap, we introduce UGPhysics, a large-scale and comprehensive benchmark specifically designed to evaluate UnderGraduate-level Physics (UGPhysics) reasoning with LLMs. UGPhysics includes 5,520 undergraduate-level physics problems in both English and Chinese, covering 13 subjects with seven different answer types and four distinct physics reasoning skills, all rigorously screened for data leakage. Additionally, we develop a Model-Assistant Rule-based Judgment (MARJ) pipeline specifically tailored for assessing answer correctness of physics problems, ensuring accurate evaluation. Our evaluation of 31 leading LLMs shows that the highest overall accuracy, 49.8% (achieved by OpenAI-o1-mini), emphasizes the necessity for models with stronger physics reasoning skills, beyond math abilities. We hope UGPhysics, along with MARJ, will drive future advancements in AI for physics reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解决复杂推理任务方面展现了显著的能力，尤其是在数学领域。然而，物理学推理领域存在独特的挑战，这些挑战受到的关注相对较少。现有的基准测试往往未能全面评估LLMs在本科水平物理学知识广度和深度上的能力，表明需要进行更全面的评估。为弥补这一空白，我们提出了一个大规模且全面的基准——UGPhysics，专门设计用于评估大型语言模型在本科物理学推理方面的表现。UGPhysics包含5,520道本科水平的物理问题，涵盖13个科目，并包括七种不同的答案类型和四种独立的物理推理技能，所有数据均已严格筛查以防止数据泄露。此外，我们还开发了一个针对物理问题答案正确性评估的模型助手规则基准则（MARJ）管道，以确保评估的准确性。我们的评估结果显示，最高整体准确率为49.8%（由OpenAI-o1-mini实现），突显了需要具备更强物理推理能力的模型，而不仅仅是数学能力。我们希望通过UGPhysics及其配套的MARJ管道推动物理推理领域的人工智能未来取得进步。 

---
# From Few to Many: Self-Improving Many-Shot Reasoners Through Iterative Optimization and Generation 

**Title (ZH)**: 从少量到大量：通过迭代优化与生成实现自我改进的多-shot推理器 

**Authors**: Xingchen Wan, Han Zhou, Ruoxi Sun, Hootan Nakhost, Ke Jiang, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2502.00330)  

**Abstract**: Recent advances in long-context large language models (LLMs) have led to the emerging paradigm of many-shot in-context learning (ICL), where it is observed that scaling many more demonstrating examples beyond the conventional few-shot setup in the context can lead to performance benefits. However, despite its promise, it is unclear what aspects dominate the benefits and whether simply scaling to more examples is the most effective way of improving many-shot ICL. In this work, we first provide an analysis of the factors driving many-shot ICL, and we find that 1) many-shot performance can still be attributed to often a few disproportionately influential examples and 2) identifying such influential examples ("optimize") and using them as demonstrations to regenerate new examples ("generate") can lead to further improvements. Inspired by the findings, we propose BRIDGE, an algorithm that alternates between the optimize step with Bayesian optimization to discover the influential sets of examples and the generate step to reuse this set to expand the reasoning paths of the examples back to the many-shot regime automatically. On Gemini, Claude, and Mistral LLMs of different sizes, we show that BRIDGE to significant improvements across a diverse set of tasks, including symbolic reasoning, numerical reasoning, and code generation. 

**Abstract (ZH)**: 近年来，长上下文语言模型（LLMs）的进展推动了多示例上下文学习（ICL）新兴范式的出现，在这种范式中，观察到在上下文中超出传统少量示例设置的情况下加入更多示例可以带来性能提升。然而，尽管这种做法具有潜力，但尚不清楚是什么因素主导了这种益处，以及是否简单地增加更多示例是最有效的提高多示例ICL性能的方法。在这项工作中，我们首先分析了驱动多示例ICL的因素，并发现1）多示例性能仍然往往可归因于少数具有显著影响力的示例；2）识别这些有影响力的示例（优化）并利用它们生成新的示例（生成）可以进一步提高性能。基于这些发现，我们提出了一种交替进行“优化”步骤和“生成”步骤的BRIDGE算法。在“优化”步骤中，使用贝叶斯优化来发现具有影响力的示例集；在“生成”步骤中，利用这些示例集自动扩展示例的推理路径，以返回到多示例范式。我们在不同规模的Gemini、Claude和Mistral语言模型上展示了BRIDGE在包括符号推理、数值推理和代码生成等不同任务上的显著改进。 

---
# CoddLLM: Empowering Large Language Models for Data Analytics 

**Title (ZH)**: CoddLLM：赋能大型语言模型进行数据分析 

**Authors**: Jiani Zhang, Hengrui Zhang, Rishav Chakravarti, Yiqun Hu, Patrick Ng, Asterios Katsifodimos, Huzefa Rangwala, George Karypis, Alon Halevy  

**Link**: [PDF](https://arxiv.org/pdf/2502.00329)  

**Abstract**: Large Language Models (LLMs) have the potential to revolutionize data analytics by simplifying tasks such as data discovery and SQL query synthesis through natural language interactions. This work serves as a pivotal first step toward the development of foundation models explicitly designed for data analytics applications. To propel this vision forward, we unveil a new data recipe for post-training LLMs, enhancing their comprehension of data management and empowering them to tackle complex real-world analytics tasks. Specifically, our innovative approach includes a scalable synthetic data generation method that enables the creation of a broad spectrum of topics centered on data representation and manipulation. Furthermore, we introduce two new tasks that seamlessly bridge tables and text. We show that such tasks can enhance models' understanding of schema creation and the nuanced translation between natural language and tabular data. Leveraging this data recipe, we post-train a new foundation model, named CoddLLM, based on Mistral-NeMo-12B. To assess the language understanding and reasoning capabilities of LLMs in the realm of data analytics, we contribute AnalyticsMMLU, a benchmark containing thousands of multiple-choice questions on databases, data analysis, and machine learning. Our focus on data discovery, has resulted in the contribution of three comprehensive benchmarks that address both database and data lake scenarios. CoddLLM not only excels in performance but also sets a new standard, achieving the highest average accuracy across eight datasets. It outperforms GPT-3.5-Turbo on AnalyticsMMLU, exceeding GPT-4o by 12.1% in table selection and showing an average improvement of 24.9% in Text-to-SQL compared to the base model. 

**Abstract (ZH)**: 大规模语言模型（LLMs）有望通过自然语言交互简化数据发现和SQL查询合成等任务，从而彻底改变数据分析。本研究是旨在开发明确面向数据应用的基础模型的重要第一步。为了推动这一愿景的实现，我们提出了一个新的数据食谱，用于后训练LLMs，增强其对数据管理的理解，并使其能够应对复杂的现实世界分析任务。具体而言，我们提出了一种创新的方法，包括一种可扩展的合成数据生成方法，能够生成涵盖数据表示和操作广泛主题的数据集。此外，我们引入了两个新的任务，无缝连接表格和文本。我们展示了这些任务如何提高模型对模式创建和自然语言与表结构数据之间细微转换的理解。借助这一数据食谱，我们基于Mistral-NeMo-12B训练了一个新的基础模型，命名为CoddLLM。为了评估LLMs在数据分析领域的语言理解和推理能力，我们贡献了AnalyticsMMLU基准测试，包含数千道关于数据库、数据分析和机器学习的多选题。我们专注于数据发现，提出并贡献了三项全面的基准测试，分别解决了数据库和数据湖场景。CoddLLM不仅在性能上表现出色，而且还设定了新的标准，其在八个数据集上的平均准确率达到最高。在AnalyticsMMLU基准测试中，CoddLLM的表现超过了GPT-3.5-Turbo，其在表选择上的性能比GPT-4高出12.1%，而在文本到SQL任务上的平均改进达到了24.9%，超过基线模型。 

---
# MIM: Multi-modal Content Interest Modeling Paradigm for User Behavior Modeling 

**Title (ZH)**: 多模态内容兴趣建模范式在用户行为建模中的应用

这个标题翻译成中文时，保持了原文的意义和学术规范。其中，“MIM”代表“Multi-modal Interest Modeling”，在这里翻译为“多模态内容兴趣建模”。为了使表达更加自然和准确，将其简化为“多模态内容兴趣建模范式”。 

**Authors**: Bencheng Yan, Si Chen, Shichang Jia, Jianyu Liu, Yueran Liu, Chenghan Fu, Wanxian Guan, Hui Zhao, Xiang Zhang, Kai Zhang, Wenbo Su, Pengjie Wang, Jian Xu, Bo Zheng, Baolin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00321)  

**Abstract**: Click-Through Rate (CTR) prediction is a crucial task in recommendation systems, online searches, and advertising platforms, where accurately capturing users' real interests in content is essential for performance. However, existing methods heavily rely on ID embeddings, which fail to reflect users' true preferences for content such as images and titles. This limitation becomes particularly evident in cold-start and long-tail scenarios, where traditional approaches struggle to deliver effective results. To address these challenges, we propose a novel Multi-modal Content Interest Modeling paradigm (MIM), which consists of three key stages: Pre-training, Content-Interest-Aware Supervised Fine-Tuning (C-SFT), and Content-Interest-Aware UBM (CiUBM). The pre-training stage adapts foundational models to domain-specific data, enabling the extraction of high-quality multi-modal embeddings. The C-SFT stage bridges the semantic gap between content and user interests by leveraging user behavior signals to guide the alignment of embeddings with user preferences. Finally, the CiUBM stage integrates multi-modal embeddings and ID-based collaborative filtering signals into a unified framework. Comprehensive offline experiments and online A/B tests conducted on the Taobao, one of the world's largest e-commerce platforms, demonstrated the effectiveness and efficiency of MIM method. The method has been successfully deployed online, achieving a significant increase of +14.14% in CTR and +4.12% in RPM, showcasing its industrial applicability and substantial impact on platform performance. To promote further research, we have publicly released the code and dataset at this https URL. 

**Abstract (ZH)**: 点击通过率（CTR）预测是推荐系统、在线搜索和广告平台中的关键任务，准确捕捉用户的实际兴趣对于系统性能至关重要。然而，现有方法高度依赖ID嵌入，未能充分反映用户对内容如图像和标题的真实偏好。这一局限性在冷启动和长尾场景中尤为明显，传统方法难以有效应对这些挑战。为了解决这些问题，我们提出了一种新的多模态内容兴趣建模范式（MIM），该范式包含三个关键阶段：预训练、内容兴趣感知监督微调（C-SFT）和内容兴趣感知的统一贝叶斯模型（CiUBM）。预训练阶段使基础模型适应特定领域的数据，从而提取高质量的多模态嵌入。C-SFT阶段通过利用用户行为信号来弥合内容和用户兴趣之间的语义差距，引导嵌入与用户偏好的对齐。最后，CiUBM阶段将多模态嵌入和ID基的协作过滤信号整合到统一框架中。在淘宝平台上进行的全面离线实验和在线A/B测试证明了MIM方法的有效性和效率。该方法已成功部署上线，CTR提高了14.14%，RPM提高了4.12%，展示了其工业应用的可行性和对平台性能的显著影响。为促进进一步研究，我们已将代码和数据集在此网址公开发布：[提供网址]。 

---
# Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values 

**Title (ZH)**: 大型语言模型中的分配公平性：评估与人类价值观的一致性 

**Authors**: Hadi Hosseini, Samarth Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2502.00313)  

**Abstract**: The growing interest in employing large language models (LLMs) for decision-making in social and economic contexts has raised questions about their potential to function as agents in these domains. A significant number of societal problems involve the distribution of resources, where fairness, along with economic efficiency, play a critical role in the desirability of outcomes. In this paper, we examine whether LLM responses adhere to fundamental fairness concepts such as equitability, envy-freeness, and Rawlsian maximin, and investigate their alignment with human preferences. We evaluate the performance of several LLMs, providing a comparative benchmark of their ability to reflect these measures. Our results demonstrate a lack of alignment between current LLM responses and human distributional preferences. Moreover, LLMs are unable to utilize money as a transferable resource to mitigate inequality. Nonetheless, we demonstrate a stark contrast when (some) LLMs are tasked with selecting from a predefined menu of options rather than generating one. In addition, we analyze the robustness of LLM responses to variations in semantic factors (e.g. intentions or personas) or non-semantic prompting changes (e.g. templates or orderings). Finally, we highlight potential strategies aimed at enhancing the alignment of LLM behavior with well-established fairness concepts. 

**Abstract (ZH)**: 随着将大规模语言模型（LLMs）应用于社会和经济背景下的决策问题研究逐渐引起关注，人们开始探讨这些模型是否能在这些领域中充当代理角色。许多社会问题涉及到资源分配，公平性不仅在经济学效率方面起着关键作用，还在非经济性方面同样重要。在本文中，我们探讨LLM响应是否符合基本的公平性概念，如公平性、无嫉妒性和罗尔斯的最小最大化原则，并调查这些响应是否与人类偏好相一致。我们评估了多个LLM的表现，并提供了一个它们在反映这些度量标准方面能力的对比基准。我们的结果显示，当前的LLM响应与人类的分配偏好之间存在脱节。此外，当（某些）LLMs从预定义的选项菜单中进行选择而不是生成时，这些模型无法利用金钱作为可转移的资源来缓解不平等。然而，当LLMs被要求从预定义的选项菜单中进行选择而非生成新菜单时，我们发现了一个显著的区别。此外，我们分析了LLM响应对语义因素（如意图或人设）或非语义提示变化（如模板或顺序）变化的鲁棒性。最后，我们提出了几种策略，旨在提高LLM行为与广泛认可的公平性概念的对齐程度。 

---
# SigWavNet: Learning Multiresolution Signal Wavelet Network for Speech Emotion Recognition 

**Title (ZH)**: SigWavNet：学习多分辨率信号小波网络用于言语情绪识别 

**Authors**: Alaa Nfissi, Wassim Bouachir, Nizar Bouguila, Brian Mishara  

**Link**: [PDF](https://arxiv.org/pdf/2502.00310)  

**Abstract**: In the field of human-computer interaction and psychological assessment, speech emotion recognition (SER) plays an important role in deciphering emotional states from speech signals. Despite advancements, challenges persist due to system complexity, feature distinctiveness issues, and noise interference. This paper introduces a new end-to-end (E2E) deep learning multi-resolution framework for SER, addressing these limitations by extracting meaningful representations directly from raw waveform speech signals. By leveraging the properties of the fast discrete wavelet transform (FDWT), including the cascade algorithm, conjugate quadrature filter, and coefficient denoising, our approach introduces a learnable model for both wavelet bases and denoising through deep learning techniques. The framework incorporates an activation function for learnable asymmetric hard thresholding of wavelet coefficients. Our approach exploits the capabilities of wavelets for effective localization in both time and frequency domains. We then combine one-dimensional dilated convolutional neural networks (1D dilated CNN) with a spatial attention layer and bidirectional gated recurrent units (Bi-GRU) with a temporal attention layer to efficiently capture the nuanced spatial and temporal characteristics of emotional features. By handling variable-length speech without segmentation and eliminating the need for pre or post-processing, the proposed model outperformed state-of-the-art methods on IEMOCAP and EMO-DB datasets. The source code of this paper is shared on the Github repository: this https URL. 

**Abstract (ZH)**: 在人机交互和心理评估领域，语音情感识别（SER）在从语音信号中解读情绪状态方面起着重要作用。尽管取得了进展，但由于系统复杂性、特征独特性问题和噪声干扰，仍存在挑战。本文提出了一种新的端到端（E2E）深度学习多分辨率框架以应对这些局限性，通过直接从原始波形语音信号中提取有意义的表示。利用快速离散小波变换（FDWT）的特性，包括级联算法、共轭四元滤波器和系数去噪，我们通过深度学习技术引入了可学习的模型，用于波形基底和去噪。该框架采用了激活函数以实现可学习的非对称硬阈值波形系数激活。我们的方法利用小波的特性，在时间和频率域中实现有效定位。随后，我们结合了一维膨胀卷积神经网络（1D膨胀CNN）与空间注意力层以及双向门控循环单元（Bi-GRU）与时序注意力层，以有效捕捉情感特征的细微时空特性。通过处理不同长度的语音而不进行分割，并且消除了预处理或后处理的需要，所提出模型在IEMOCAP和EMO-DB数据集上优于现有方法。本文的源代码已在GitHub仓库中公开：[这个链接](https://github.com/具体用户名/具体仓库名)。 

---
# Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation 

**Title (ZH)**: 这个谜题给你！隐秘的成员推理在检索增强生成中的应用 

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2502.00306)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference. 

**Abstract (ZH)**: 检索增强生成（RAG）使大型语言模型（LLMs）能够通过利用外部知识数据库生成基于事实的回应，而无需改变模型参数。尽管缺少权重调整可以防止通过模型参数泄露信息，但它增加了恶意推理者利用检索到的文档在模型上下文中提取信息的风险。现有的成员推理和数据提取方法往往依赖于“jailbreaking”或精心构造的不自然查询，这些方法很容易被RAG系统中常见的查询重写技术检测或阻止。在本研究中，我们提出了询问攻击（Interrogation Attack，IA），这是一种针对RAG数据存储库中文档的成员推理技术。通过构造只能通过目标文档的存在才能回答的自然文本查询，我们的方法仅使用30个查询就成功实现了推理，并且保持了隐蔽性；现有的检测器能够比我们的攻击生成的提示更频繁地识别出8到76倍的恶意提示。我们的方法在多种RAG配置中实现了1%假阳性率下的召回率提升2倍，同时每文档推理成本低于0.02美元。 

---
# DEUCE: Dual-diversity Enhancement and Uncertainty-awareness for Cold-start Active Learning 

**Title (ZH)**: DEUCE：冷启动主动学习中的双元多样性增强与不确定性意识 

**Authors**: Jiaxin Guo, C. L. Philip Chen, Shuzhen Li, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00305)  

**Abstract**: Cold-start active learning (CSAL) selects valuable instances from an unlabeled dataset for manual annotation. It provides high-quality data at a low annotation cost for label-scarce text classification. However, existing CSAL methods overlook weak classes and hard representative examples, resulting in biased learning. To address these issues, this paper proposes a novel dual-diversity enhancing and uncertainty-aware (DEUCE) framework for CSAL. Specifically, DEUCE leverages a pretrained language model (PLM) to efficiently extract textual representations, class predictions, and predictive uncertainty. Then, it constructs a Dual-Neighbor Graph (DNG) to combine information on both textual diversity and class diversity, ensuring a balanced data distribution. It further propagates uncertainty information via density-based clustering to select hard representative instances. DEUCE performs well in selecting class-balanced and hard representative data by dual-diversity and informativeness. Experiments on six NLP datasets demonstrate the superiority and efficiency of DEUCE. 

**Abstract (ZH)**: 冷启动主动学习（CSAL）从未标注数据集中选择有价值的实例进行人工注释，为标注稀缺的文字分类任务提供高质量的数据，同时降低成本。然而，现有的CSAL方法忽视了弱类别和难代表的实例，导致学习结果具有偏见。为了解决这些问题，本文提出了一种新的双重多样性增强和不确定性感知（DEUCE）框架，用于CSAL。具体而言，DEUCE利用预训练的语言模型（PLM）高效地提取文本表示、类别预测和预测不确定性。然后，它构建一个双重邻域图（DNG），结合文本多样性与类别多样性的信息，确保数据分布的平衡。进一步通过基于密度的聚类传播不确定性信息，选择难代表的实例。DEUCE通过双重多样性和信息的相关性在选择类别均衡和难代表数据方面表现良好。实验结果在六个NLP数据集上表明DEUCE的优势和效率。 

---
# HoP: Homeomorphic Polar Learning for Hard Constrained Optimization 

**Title (ZH)**: HoP：同胚极坐标学习在硬约束优化中的应用 

**Authors**: Ke Deng, Hanwen Zhang, Jin Lu, Haijian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.00304)  

**Abstract**: Constrained optimization demands highly efficient solvers which promotes the development of learn-to-optimize (L2O) approaches. As a data-driven method, L2O leverages neural networks to efficiently produce approximate solutions. However, a significant challenge remains in ensuring both optimality and feasibility of neural networks' output. To tackle this issue, we introduce Homeomorphic Polar Learning (HoP) to solve the star-convex hard-constrained optimization by embedding homeomorphic mapping in neural networks. The bijective structure enables end-to-end training without extra penalty or correction. For performance evaluation, we evaluate HoP's performance across a variety of synthetic optimization tasks and real-world applications in wireless communications. In all cases, HoP achieves solutions closer to the optimum than existing L2O methods while strictly maintaining feasibility. 

**Abstract (ZH)**: 受约束的优化需要高效的求解器，这促进了学习优化（Learn-to-Optimize, L2O）方法的发展。作为一种数据驱动的方法，L2O 利用神经网络高效地生成近似解。然而，确保神经网络输出的最优性和可行性仍然是一个重要挑战。为了应对这一挑战，我们引入了同胚极性学习（Homeomorphic Polar Learning, HoP）方法，通过在神经网络中嵌入同胚映射来解决星凸的硬约束优化问题。双射结构使得能够在端到端训练中不需额外的惩罚或修正。为了评估性能，我们对HoP在多种合成优化任务和无线通信领域的实际应用进行了评估。在所有情况下，HoP 都能够达到比现有L2O方法更接近最优解的结果，同时严格保持可行性。 

---
# Learning to Fuse Temporal Proximity Networks: A Case Study in Chimpanzee Social Interactions 

**Title (ZH)**: 学习融合时间邻近网络：黑猩猩社会互动案例研究 

**Authors**: Yixuan He, Aaron Sandel, David Wipf, Mihai Cucuringu, John Mitani, Gesine Reinert  

**Link**: [PDF](https://arxiv.org/pdf/2502.00302)  

**Abstract**: How can we identify groups of primate individuals which could be conjectured to drive social structure? To address this question, one of us has collected a time series of data for social interactions between chimpanzees. Here we use a network representation, leading to the task of combining these data into a time series of a single weighted network per time stamp, where different proximities should be given different weights reflecting their relative importance. We optimize these proximity-type weights in a principled way, using an innovative loss function which rewards structural consistency across time. The approach is empirically validated by carefully designed synthetic data. Using statistical tests, we provide a way of identifying groups of individuals that stay related for a significant length of time. Applying the approach to the chimpanzee data set, we detect cliques in the animal social network time series, which can be validated by real-world intuition from prior research and qualitative observations by chimpanzee experts. 

**Abstract (ZH)**: 如何识别可能驱动社会结构的灵长类个体群体？为回答这一问题，其中一位作者收集了一定时期内黑猩猩社会互动的时间序列数据。本研究采用网络表示法，从而将这些数据整合为一个时间戳下的加权网络时间序列，不同类型的接近度应赋予不同的权重，反映出其相对重要性。通过一种创新的损失函数，以有原则的方式优化这些接近度类型的权重，该损失函数奖励时间上结构的一致性。该方法通过精心设计的合成数据进行了实证验证。利用统计检验方法，提供了一种识别在显著时间内保持关联的个体群体的方法。将该方法应用于黑猩猩数据集，我们检测到了动物社会网络时间序列中的团簇，这些结果可以通过来自先前研究和灵长类专家的定性观察进行验证。 

---
# Estimating LLM Uncertainty with Logits 

**Title (ZH)**: 使用Logits估计大语言模型的不确定性 

**Authors**: Huan Ma, Jingdong Chen, Guangyu Wang, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00290)  

**Abstract**: In recent years, Large Language Models (LLMs) have seen remarkable advancements and have been extensively integrated across various fields. Despite their progress, LLMs are prone to hallucinations, producing responses that may not be dependable if the models lack sufficient grounding knowledge. To mitigate this issue, methods for estimating uncertainty have been adopted, with a focus on critical tokens as indicators of reliability. Nevertheless, probability-based approaches have shown limitations in assessing token-level reliability due to the erosion of evidence strength information acquired during training. In this paper, we introduce Logits-induced Token Uncertainty (LogU), a novel framework designed to estimate token-specific uncertainty in LLMs in real time, without the need for multiple sampling rounds. By leveraging evidence modeling for the implementation of LogU, we utilize the derived uncertainty measures to steer downstream tasks. Our experimental findings highlight the substantial effectiveness and potential of LogU, marking a significant advancement in addressing the challenge of model hallucinations. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）取得了显著的进步，并在各个领域得到了广泛的应用。尽管取得了这些进展，但LLMs仍然容易产生幻觉，即生成可能不可靠的响应，尤其是在模型缺乏足够的环境知识时。为了缓解这一问题，人们采用了估计不确定性的方法，以关键令牌作为可靠性的指示器。然而，基于概率的方法在评估令牌级别的可靠性方面受到了证据强度信息在训练过程中被降解的限制。在本文中，我们提出了一种名为Logits诱导令牌不确定性（LogU）的新型框架，该框架能够在实时情况下估计LLMs中的令牌特定不确定性，无需多次采样。通过利用证据建模来实施LogU，我们利用所得到的不确定性措施来引导下游任务。我们的实验结果表明，LogU在应对模型幻觉挑战方面具有显著的有效性和潜力，标志着在解决模型幻觉问题方面取得了重要进展。 

---
# Sigmoid Self-Attention is Better than Softmax Self-Attention: A Mixture-of-Experts Perspective 

**Title (ZH)**: 从专家混合的角度来看，S形自我注意优于软性最大化自我注意 

**Authors**: Fanqi Yan, Huy Nguyen, Pedram Akbarian, Nhat Ho, Alessandro Rinaldo  

**Link**: [PDF](https://arxiv.org/pdf/2502.00281)  

**Abstract**: At the core of the popular Transformer architecture is the self-attention mechanism, which dynamically assigns softmax weights to each input token so that the model can focus on the most salient information. However, the softmax structure slows down the attention computation due to its row-wise nature, and inherently introduces competition among tokens: as the weight assigned to one token increases, the weights of others decrease. This competitive dynamic may narrow the focus of self-attention to a limited set of features, potentially overlooking other informative characteristics. Recent experimental studies have shown that using the element-wise sigmoid function helps eliminate token competition and reduce the computational overhead. Despite these promising empirical results, a rigorous comparison between sigmoid and softmax self-attention mechanisms remains absent in the literature. This paper closes this gap by theoretically demonstrating that sigmoid self-attention is more sample-efficient than its softmax counterpart. Toward that goal, we illustrate that each row of the self-attention matrix can be represented as a mixture of experts. Our analysis shows that ''experts'' in sigmoid self-attention require significantly less data to achieve the same approximation error as those in softmax self-attention. We corroborate our theoretical findings through extensive experiments on both synthetic and real-world datasets. 

**Abstract (ZH)**: 在流行的位置编码器（Transformer架构）的核心是自注意力机制，该机制能够动态地为每个输入词元分配softmax权重，从而使模型能够专注于最关键的信息。然而，softmax结构由于其行向量性质而导致注意力计算速度变慢，并且固有地引入了词元之间的竞争：一个词元分配的权重增加时，其他词元的权重会相应减少。这种竞争动力可能会使自注意力的焦点集中在有限的特征集上，从而可能忽略了其他重要的特征。近期的实验研究表明，使用元素级的sigmoid函数有助于消除词元之间的竞争并减少计算负担。尽管这些经验结果显示出积极的前景，但有关sigmoid与softmax自注意力机制之间严格比较的研究在文献中仍然缺乏。为此，本文通过理论证明.sigmoid自注意力机制比其softmax版本更为样本高效。为此，我们指出自注意力矩阵的每一行都可以表示为专家混合。我们的分析表明，sigmoid自注意力中的“专家”需要显著较少的数据才能达到与softmax自注意力中的“专家”相同的近似误差。我们通过在合成数据集和真实世界数据集上的广泛实验来验证我们的理论发现。 

---
# DUET: Optimizing Training Data Mixtures via Feedback from Unseen Evaluation Tasks 

**Title (ZH)**: DUET：通过未见评估任务反馈优化训练数据混合 

**Authors**: Zhiliang Chen, Gregory Kang Ruey Lau, Chuan-Sheng Foo, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2502.00270)  

**Abstract**: The performance of a machine learning (ML) model depends heavily on the relevance of its training data to the domain of the downstream evaluation task. However, in practice, the data involved in an unseen evaluation task is often not known to us (e.g., conversations between an LLM and a user are end-to-end encrypted). So, it is not obvious what data would be relevant for training/fine-tuning the ML model to maximize its task performance. Instead, one can only deploy the ML model in the unseen evaluation task to gather multiple rounds of coarse feedback on how well the model has performed. This paper presents a novel global-to-local algorithm called DUET that can exploit the feedback loop by interleaving a data selection method with Bayesian optimization. As a result, DUET can efficiently refine the training data mixture from a pool of data domains to maximize the model's performance on the unseen evaluation task and its convergence to the optimal data mixture can be theoretically guaranteed by analyzing its cumulative regret. Empirical evaluation on image and LLM evaluation tasks shows that DUET finds better training data mixtures than conventional baselines. 

**Abstract (ZH)**: 机器学习（ML）模型的性能高度依赖于其训练数据与下游评估任务领域的相关性。然而，在实践中，未见过的评估任务所涉及的数据通常是未知的（例如，大语言模型LLM与用户之间的对话是端到端加密的）。因此，并不清楚哪些数据对于训练/微调ML模型以最大化其任务性能最为相关。相反，只能将ML模型部署到未见过的评估任务中，收集多轮粗略反馈以了解模型的表现如何。本文提出了一种新的全局到局部算法DUET，该算法通过交替使用数据选择方法和贝叶斯优化来利用反馈循环。因此，DUET可以高效地从数据域池中细化训练数据混合，以最大化其在未见过的评估任务中的性能，并通过分析其累积遗憾，理论上可以确保其收敛到最优数据混合。对图像和LLM评估任务的实证评估表明，DUET能够找到比传统基线更好的训练数据混合。 

---
# Your submission contained main.bib and main.tex file, but no main.bbl file (include main.bbl, or submit without main.bib; and remember to verify references) 

**Title (ZH)**: 您的提交包含了一个`main.bib`文件和一个`main.tex`文件，但没有包含`main.bbl`文件（请包含`main.bbl`文件，或不使用`main.bib`文件；并且请记得验证参考文献）。 

**Authors**: Dianwei Chen, Zifan Zhang, Yuchen Liu, Xianfeng Terry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00262)  

**Abstract**: Autonomous driving systems face significant challenges in handling unpredictable edge-case scenarios, such as adversarial pedestrian movements, dangerous vehicle maneuvers, and sudden environmental changes. Current end-to-end driving models struggle with generalization to these rare events due to limitations in traditional detection and prediction approaches. To address this, we propose INSIGHT (Integration of Semantic and Visual Inputs for Generalized Hazard Tracking), a hierarchical vision-language model (VLM) framework designed to enhance hazard detection and edge-case evaluation. By using multimodal data fusion, our approach integrates semantic and visual representations, enabling precise interpretation of driving scenarios and accurate forecasting of potential dangers. Through supervised fine-tuning of VLMs, we optimize spatial hazard localization using attention-based mechanisms and coordinate regression techniques. Experimental results on the BDD100K dataset demonstrate a substantial improvement in hazard prediction straightforwardness and accuracy over existing models, achieving a notable increase in generalization performance. This advancement enhances the robustness and safety of autonomous driving systems, ensuring improved situational awareness and potential decision-making in complex real-world scenarios. 

**Abstract (ZH)**: 自主驾驶系统在处理不可预测的边缘情况场景时面临巨大挑战，如对手操作的行人移动、危险的机动驾驶以及突然的环境变化。当前的端到端驾驶模型由于传统检测和预测方法的局限性，在对这些罕见事件的泛化方面表现出色。为解决这一问题，我们提出了一种名为INSIGHT（Integrating Semantic and Visual Inputs for Generalized Hazard Tracking）的层次视觉-语言模型（VLM）框架，旨在增强隐患检测和边缘情况评估能力。通过多模态数据融合，我们的方法将语义和视觉表示相结合，能够精确地解释驾驶场景并准确预测潜在危险。通过监督微调视觉-语言模型（VLMs），我们使用基于注意力的机制和坐标回归技术优化空间隐患定位。在BDD100K数据集上的实验结果表明，与现有模型相比，我们的方法在隐患预测的简单性和准确性方面取得了显著改进，并显著提高了泛化性能。这一进展增强了自主驾驶系统的稳健性和安全性，确保在复杂的现实场景中具备更好的情境意识和决策能力。 

---
# Mordal: Automated Pretrained Model Selection for Vision Language Models 

**Title (ZH)**: Mordal：自动预训练模型选择用于视觉语言模型 

**Authors**: Shiqi He, Insu Jang, Mosharaf Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2502.00241)  

**Abstract**: Incorporating multiple modalities into large language models (LLMs) is a powerful way to enhance their understanding of non-textual data, enabling them to perform multimodal tasks. Vision language models (VLMs) form the fastest growing category of multimodal models because of their many practical use cases, including in healthcare, robotics, and accessibility. Unfortunately, even though different VLMs in the literature demonstrate impressive visual capabilities in different benchmarks, they are handcrafted by human experts; there is no automated framework to create task-specific multimodal models.
We introduce Mordal, an automated multimodal model search framework that efficiently finds the best VLM for a user-defined task without manual intervention. Mordal achieves this both by reducing the number of candidates to consider during the search process and by minimizing the time required to evaluate each remaining candidate. Our evaluation shows that Mordal can find the best VLM for a given problem using up to $8.9\times$--$11.6\times$ lower GPU hours than grid search. In the process of our evaluation, we have also discovered new VLMs that outperform their state-of-the-art counterparts. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，并确保符合学术规范：

将多种模态数据整合到大型语言模型（LLMs）中是增强其对非文本数据理解的一种强大方式，使其能够执行多模态任务。视觉语言模型（VLMs）由于其在医疗保健、机器人技术和无障碍领域等多种实用场景中的应用，是增长最快的多模态模型类别。不幸的是，尽管文献中提出了具有不同视觉能力的不同VLMs，这些模型都是由人类专家手工设计的；目前尚无自动化的框架能够创建针对特定任务的多模态模型。

我们引入了Mordal，这是一个自动化的多模态模型搜索框架，能够在无需人工干预的情况下高效地找到最适合用户定义任务的最佳VLM。Mordal 通过减少需要考虑的候选模型数量，并尽可能减少评估每个剩余候选模型所需的时间，实现了这一目标。我们的评估表明，Mordal 能够在资源使用上最多节省多达 $8.9 \times$ 到 $11.6 \times$ 的 GPU 小时来找出给定问题的最佳VLM。在评估过程中，我们还发现了超越其当前最优同类模型的新VLMs。 

---
# A Hybrid Random Forest and CNN Framework for Tile-Wise Oil-Water Classification in Hyperspectral Images 

**Title (ZH)**: 一种用于高光谱图像中砖块级油水分类的混合随机森林和CNN框架 

**Authors**: Mehdi Nickzamir, Seyed Mohammad Sheikh Ahamdi Gandab  

**Link**: [PDF](https://arxiv.org/pdf/2502.00232)  

**Abstract**: A novel hybrid Random Forest and Convolutional Neural Network (CNN) framework is presented for oil-water classification in hyperspectral images (HSI). To address the challenge of preserving spatial context, the images were divided into smaller, non-overlapping tiles, which served as the basis for training, validation, and testing. Random Forest demonstrated strong performance in pixel-wise classification, outperforming models such as XGBoost, Attention-Based U-Net, and HybridSN. However, Random Forest loses spatial context, limiting its ability to fully exploit the spatial relationships in hyperspectral data. To improve performance, a CNN was trained on the probability maps generated by the Random Forest, leveraging the CNN's capacity to incorporate spatial context. The hybrid approach achieved 7.6% improvement in recall (to 0.85), 2.4% improvement in F1 score (to 0.84), and 0.54% improvement in AUC (to 0.99) compared to the baseline. These results highlight the effectiveness of combining probabilistic outputs with spatial feature learning for context-aware analysis of hyperspectral images. 

**Abstract (ZH)**: 本文提出了一种新的随机森林和卷积神经网络（CNN）混合框架，用于高光谱图像（HSI）中的油水分类。为了解决保持空间上下文的挑战，将图像划分为更小的非重叠块，这些块作为训练、验证和测试的基础。随机森林在像素级分类中表现出色，优于XGBoost、注意力机制U-Net和HybridSN等模型。然而，随机森林会丢失空间上下文，限制了其完全利用高光谱数据中空间关系的能力。为了提高性能，对随机森林生成的概率图进行卷积神经网络训练，利用CNN结合空间上下文的能力。混合方法在召回率上取得了7.6%的提升（达到0.85），F1分数上提升了2.4%（达到0.84），AUC上提升了0.54%（达到0.99），优于基线方法。这些结果突显了结合概率输出与空间特征学习以进行高光谱图像上下文感知分析的有效性。 

---
# Should You Use Your Large Language Model to Explore or Exploit? 

**Title (ZH)**: 你应该使用你的大型语言模型进行探索还是利用？ 

**Authors**: Keegan Harris, Aleksandrs Slivkins  

**Link**: [PDF](https://arxiv.org/pdf/2502.00225)  

**Abstract**: We evaluate the ability of the current generation of large language models (LLMs) to help a decision-making agent facing an exploration-exploitation tradeoff. We use LLMs to explore and exploit in silos in various (contextual) bandit tasks. We find that while the current LLMs often struggle to exploit, in-context mitigations may be used to substantially improve performance for small-scale tasks. However even then, LLMs perform worse than a simple linear regression. On the other hand, we find that LLMs do help at exploring large action spaces with inherent semantics, by suggesting suitable candidates to explore. 

**Abstract (ZH)**: 我们评估了当前大型语言模型（LLM）在面对探索与利用权衡时帮助决策代理的能力。我们使用LLM在各种（上下文相关的）多臂赌局任务中独立进行探索和利用。我们发现，尽管当前的LLM在利用方面常常难以胜任，但在具体任务中可以利用上下文内的缓解措施来显著提升性能。然而，即使在这种情况下，LLM的表现仍不如简单的线性回归模型。另一方面，我们发现LLM在探索具有内在语义的大动作空间时确实能够提供帮助，通过建议合适的探索候选对象来实现这一点。 

---
# Fantastic Multi-Task Gradient Updates and How to Find Them In a Cone 

**Title (ZH)**: fantastic 多任务梯度更新及其在圆锥中的寻找方法 

**Authors**: Negar Hassanpour, Muhammad Kamran Janjua, Kunlin Zhang, Sepehr Lavasani, Xiaowen Zhang, Chunhua Zhou, Chao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.00217)  

**Abstract**: Balancing competing objectives remains a fundamental challenge in multi-task learning (MTL), primarily due to conflicting gradients across individual tasks. A common solution relies on computing a dynamic gradient update vector that balances competing tasks as optimization progresses. Building on this idea, we propose ConicGrad, a principled, scalable, and robust MTL approach formulated as a constrained optimization problem. Our method introduces an angular constraint to dynamically regulate gradient update directions, confining them within a cone centered on the reference gradient of the overall objective. By balancing task-specific gradients without over-constraining their direction or magnitude, ConicGrad effectively resolves inter-task gradient conflicts. Moreover, our framework ensures computational efficiency and scalability to high-dimensional parameter spaces. We conduct extensive experiments on standard supervised learning and reinforcement learning MTL benchmarks, and demonstrate that ConicGrad achieves state-of-the-art performance across diverse tasks. 

**Abstract (ZH)**: 在多任务学习（MTL）中平衡相互竞争的目标仍然是一个基本挑战，主要原因是各个任务之间的梯度可能存在冲突。一个常见的解决方案是计算一个动态梯度更新向量，随着优化过程的进行，平衡各个任务之间的竞争。基于这一想法，我们提出了ConicGrad，这是一种原理性、可扩展且健壮的MTL方法，被形式化为一个受约束的优化问题。我们的方法引入了一个角度约束，以动态调节梯度更新的方向，将它们限制在一个以总体目标参考梯度为中心的圆锥内。通过平衡任务特定的梯度，而不过度限制其方向或大小，ConicGrad有效地解决了任务间梯度的冲突。此外，我们的框架确保了计算效率和对高维参数空间的可扩展性。我们在标准的有监督学习和强化学习MTL基准上进行了广泛的实验，结果表明ConicGrad在各种任务上都达到了最先进的性能。 

---
# Understanding Why Adam Outperforms SGD: Gradient Heterogeneity in Transformers 

**Title (ZH)**: 理解 Adam 为何优于 SGD：Transformer 中的梯度异方差性 

**Authors**: Akiyoshi Tomihari, Issei Sato  

**Link**: [PDF](https://arxiv.org/pdf/2502.00213)  

**Abstract**: Transformer models are challenging to optimize with SGD and typically require adaptive optimizers such as Adam. However, the reasons behind the superior performance of Adam over SGD remain unclear. In this study, we investigate the optimization of transformer models by focusing on \emph{gradient heterogeneity}, defined as the disparity in gradient norms among parameters. Our analysis shows that gradient heterogeneity hinders gradient-based optimization, including SGD, while sign-based optimization, a simplified variant of Adam, is less affected. We further examine gradient heterogeneity in transformer models and show that it is influenced by the placement of layer normalization. Additionally, we show that the momentum term in sign-based optimization is important for preventing the excessive growth of linear-head parameters in tasks with many classes. Experimental results from fine-tuning transformer models in both NLP and vision domains validate our theoretical analyses. This study provides insights into the optimization challenges of transformer models and offers guidance for designing future optimization algorithms. Code is available at \url{this https URL}. 

**Abstract (ZH)**: 以下是将该论文内容或标题翻译成中文，并符合学术规范的结果：

Transformer 模型用 SGD 进行优化存在挑战，通常需要使用自适应优化器如 Adam。然而，Adam 相较于 SGD 的优越性能背后的原因仍然不明确。在这项研究中，我们通过关注 \emph{梯度异质性}，即参数之间的梯度范数差异，来研究Transformer模型的优化问题。我们的分析表明，梯度异质性会阻碍基于梯度的优化，包括 SGD；而基于符号的优化（Adam 的一个简化版本）受影响较小。我们进一步研究了Transformer模型中的梯度异质性，并发现其受层归一化位置的影响。此外，我们还展示了在具有多个类别的任务中，基于符号的优化中的动量项对于防止线性层参数过度增长至关重要。来自自然语言处理和视觉领域Transformer模型微调的实验结果验证了我们的理论分析。本研究为理解Transformer模型的优化挑战提供了见解，并提供了设计未来优化算法的指导。相关代码可在 \url{该URL} 获取。 

---
# Beyond Limited Data: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving 

**Title (ZH)**: 有限数据之外：具有迭代猜想与证明的自博弈大规模语言模型定理证明器 

**Authors**: Kefan Dong, Tengyu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.00212)  

**Abstract**: A fundamental challenge in formal theorem proving by LLMs is the lack of high-quality training data. Although reinforcement learning or expert iteration partially mitigates this issue by alternating between LLM generating proofs and finetuning them on correctly generated ones, performance quickly plateaus due to the scarcity of correct proofs (sparse rewards). To keep improving the models with limited data, we draw inspiration from mathematicians, who continuously develop new results, partly by proposing novel conjectures or exercises (which are often variants of known results) and attempting to solve them. We design the Self-play Theorem Prover (STP) that simultaneously takes on two roles, conjecturer and prover, each providing training signals to the other. The conjecturer is trained iteratively on previously generated conjectures that are barely provable by the current prover, which incentivizes it to generate increasingly challenging conjectures over time. The prover attempts to prove the conjectures with standard expert iteration. We evaluate STP with both Lean and Isabelle formal versifiers. With 19.8 billion tokens generated during the training in Lean, STP proves 26.3% of the statements in the LeanWorkbook dataset, doubling the previous best result of 13.2% achieved through expert iteration. The final model achieves state-of-the-art performance among whole-proof generation methods on miniF2F-test (61.1%, pass@3200), Proofnet-test (23.1%, pass@3200) and PutnamBench (8/644, pass@64). 

**Abstract (ZH)**: 形式化定理证明中使用大语言模型（LLMs）面临的一个基本挑战是缺乏高质量的训练数据。尽管强化学习或专家迭代部分地通过交替让LLM生成证明并针对正确生成的证明进行微调来缓解这一问题，但由于正确证明的稀缺性（稀疏奖励），性能很快就会停滞不前。为了在有限的数据下持续改进模型，我们从数学家身上汲取灵感。数学家们不断开发新的成果，部分是通过提出新的猜想或练习题（这些通常是已知结果的变体），然后试图解决它们。我们设计了自我对弈定理证明器（STP），使其同时扮演猜想提出者和证明者的角色，彼此之间相互提供训练信号。猜想提出者通过迭代训练之前生成的、当前证明者几乎无法证明的猜想，激励其随着时间推移生成越来越具有挑战性的猜想。证明者尝试使用标准的专家迭代来证明这些猜想。我们使用Lean和Isabelle形式验证器评估STP。在Lean中生成了198亿个 Tokens的训练过程中，STP 在LeanWorkbook数据集中证明了26.3%的陈述，比仅通过专家迭代之前的最佳结果13.2%翻了一番。最终模型在miniF2F测试（61.1%，pass@3200）、Proofnet测试（23.1%，pass@3200）和PutnamBench（8/644，pass@64）上达到了最先进的性能。 

---
# EcoWeedNet: A Lightweight and Automated Weed Detection Method for Sustainable Next-Generation Agricultural Consumer Electronics 

**Title (ZH)**: EcoWeedNet：一种用于可持续下一代农业消费电子产品的轻量化和自动化杂草检测方法 

**Authors**: Omar H. Khater, Abdul Jabbar Siddiqui, M. Shamim Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2502.00205)  

**Abstract**: Sustainable agriculture plays a crucial role in ensuring world food security for consumers. A critical challenge faced by sustainable precision agriculture is weed growth, as weeds share essential resources with the crops, such as water, soil nutrients, and sunlight, which notably affect crop yields. The traditional methods employed to combat weeds include the usage of chemical herbicides and manual weed removal methods. However, these could damage the environment and pose health hazards. The adoption of automated computer vision technologies and ground agricultural consumer electronic vehicles in precision agriculture offers sustainable, low-carbon solutions. However, prior works suffer from issues such as low accuracy and precision and high computational expense. This work proposes EcoWeedNet, a novel model with enhanced weed detection performance without adding significant computational complexity, aligning with the goals of low-carbon agricultural practices. Additionally, our model is lightweight and optimal for deployment on ground-based consumer electronic agricultural vehicles and robots. The effectiveness of the proposed model is demonstrated through comprehensive experiments on the CottonWeedDet12 benchmark dataset reflecting real-world scenarios. EcoWeedNet achieves performance close to that of large models yet with much fewer parameters. (approximately 4.21% of the parameters and 6.59% of the GFLOPs of YOLOv4). This work contributes effectively to the development of automated weed detection methods for next-generation agricultural consumer electronics featuring lower energy consumption and lower carbon footprint. This work paves the way forward for sustainable agricultural consumer technologies. 

**Abstract (ZH)**: 可持续农业在确保世界食品安全方面发挥着关键作用。可持续精准农业面临的其中一个关键挑战是杂草生长，因为杂草与作物共享水、土壤养分和阳光等必要资源，从而显著影响作物产量。传统上用于控制杂草的方法包括使用化学除草剂和人工拔草。然而，这些方法可能对环境造成损害，并存在健康风险。采用自动化计算机视觉技术和地面农业消费电子车辆在精准农业中的应用提供了可持续且低碳的解决方案。然而，先前的研究存在诸如检测准确性低、计算成本高等问题。本研究提出了一种名为EcoWeedNet的新模型，该模型在未增加显著计算复杂性的前提下提升了杂草检测性能，符合低碳农业实践的目标。此外，该模型具有轻量级特性，适合部署在地面消费电子农业车辆和机器人上。通过在CottonWeedDet12基准数据集上的全面实验，展示了该模型在反映实际场景方面的有效性。EcoWeedNet在性能上接近大型模型，但参数量却少得多（约YOLOv4参数的4.21%和GFLOPs的6.59%）。本研究有效促进了具有更低能耗和更低碳足迹的下一代农业消费电子设备中自动杂草检测方法的发展。本研究为可持续农业消费技术的发展铺平了道路。 

---
# Year-over-Year Developments in Financial Fraud Detection via Deep Learning: A Systematic Literature Review 

**Title (ZH)**: 基于深度学习的财务欺诈检测年度发展系统文献综述 

**Authors**: Yisong Chen, Chuqing Zhao, Yixin Xu, Chuanhao Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.00201)  

**Abstract**: This paper systematically reviews advancements in deep learning (DL) techniques for financial fraud detection, a critical issue in the financial sector. Using the Kitchenham systematic literature review approach, 57 studies published between 2019 and 2024 were analyzed. The review highlights the effectiveness of various deep learning models such as Convolutional Neural Networks, Long Short-Term Memory, and transformers across domains such as credit card transactions, insurance claims, and financial statement audits. Performance metrics such as precision, recall, F1-score, and AUC-ROC were evaluated. Key themes explored include the impact of data privacy frameworks and advancements in feature engineering and data preprocessing. The study emphasizes challenges such as imbalanced datasets, model interpretability, and ethical considerations, alongside opportunities for automation and privacy-preserving techniques such as blockchain integration and Principal Component Analysis. By examining trends over the past five years, this review identifies critical gaps and promising directions for advancing DL applications in financial fraud detection, offering actionable insights for researchers and practitioners. 

**Abstract (ZH)**: 本文系统回顾了深度学习（DL）技术在金融欺诈检测领域的进步，这是金融领域的一个关键问题。采用Kitchenham系统的文献回顾方法，分析了2019年至2024年间出版的57篇相关研究。回顾结果显示，各种深度学习模型（如卷积神经网络、长短期记忆网络和变换器）在信用卡交易、保险索赔和财务报表审计等领域中具有显著效果。性能度量指标包括精确率、召回率、F1分数和AUC-ROC。探讨的主要主题包括数据隐私框架的影响、特征工程和数据预处理的进步。研究强调了数据集不平衡、模型可解释性和伦理考量等挑战，同时也提到了自动化和隐私保护技术（如区块链集成和主成分分析）的机会。通过回顾过去五年中的趋势，本文识别出了DL技术在金融欺诈检测领域发展中的关键空白和潜在方向，为研究人员和实践者提供了可操作的见解。 

---
# DermaSynth: Rich Synthetic Image-Text Pairs Using Open Access Dermatology Datasets 

**Title (ZH)**: DermaSynth：使用开放访问皮肤病学数据集生成丰富的图像-文本配对 

**Authors**: Abdurrahim Yilmaz, Furkan Yuceyalcin, Ece Gokyayla, Donghee Choi, Ozan Erdem Ali Anil Demircali, Rahmetullah Varol, Ufuk Gorkem Kirabali, Gulsum Gencoglan, Joram M. Posma, Burak Temelkuran  

**Link**: [PDF](https://arxiv.org/pdf/2502.00196)  

**Abstract**: A major barrier to developing vision large language models (LLMs) in dermatology is the lack of large image--text pairs dataset. We introduce DermaSynth, a dataset comprising of 92,020 synthetic image--text pairs curated from 45,205 images (13,568 clinical and 35,561 dermatoscopic) for dermatology-related clinical tasks. Leveraging state-of-the-art LLMs, using Gemini 2.0, we used clinically related prompts and self-instruct method to generate diverse and rich synthetic texts. Metadata of the datasets were incorporated into the input prompts by targeting to reduce potential hallucinations. The resulting dataset builds upon open access dermatological image repositories (DERM12345, BCN20000, PAD-UFES-20, SCIN, and HIBA) that have permissive CC-BY-4.0 licenses. We also fine-tuned a preliminary Llama-3.2-11B-Vision-Instruct model, DermatoLlama 1.0, on 5,000 samples. We anticipate this dataset to support and accelerate AI research in dermatology. Data and code underlying this work are accessible at this https URL. 

**Abstract (ZH)**: 在皮肤科领域开发视觉大型语言模型（LLMs）的一个主要障碍是缺乏大规模的图像-文本对数据集。我们介绍了DermaSynth数据集，该数据集包含92,020张合成图像-文本对，这些对是从45,205张图像（13,568张临床图像和35,561张皮肤镜图像）中精心挑选出来的，用于皮肤科相关的临床任务。利用最先进的LLM模型（如Gemini 2.0），我们使用临床相关的提示和自我指令方法生成多样且丰富的合成文本。数据集的元数据被纳入输入提示中，旨在减少潜在的幻觉。该数据集是在开放访问的皮肤病学图像库（DERM12345、BCN20000、PAD-UFES-20、SCIN和HIBA）的基础上构建的，这些库具有宽松的CC-BY-4.0许可证。我们还针对5,000个样本细调了初期的Llama-3.2-11B-Vision-Instruct模型，即DermatoLlama 1.0。我们期望该数据集能够支持和加速皮肤科的AI研究。与此工作相关的数据和代码可通过以下链接访问：[该链接地址]。 

---
# Physics-Informed Neural Network based Damage Identification for Truss Railroad Bridges 

**Title (ZH)**: 基于物理约束的人工神经网络桁架铁路桥梁损伤识别 

**Authors**: Althaf Shajihan, Kirill Mechitov, Girish Chowdhary, Billie F. Spencer Jr  

**Link**: [PDF](https://arxiv.org/pdf/2502.00194)  

**Abstract**: Railroad bridges are a crucial component of the U.S. freight rail system, which moves over 40 percent of the nation's freight and plays a critical role in the economy. However, aging bridge infrastructure and increasing train traffic pose significant safety hazards and risk service disruptions. The U.S. rail network includes over 100,000 railroad bridges, averaging one every 1.4 miles of track, with steel bridges comprising over 50% of the network's total bridge length. Early identification and assessment of damage in these bridges remain challenging tasks. This study proposes a physics-informed neural network (PINN) based approach for damage identification in steel truss railroad bridges. The proposed approach employs an unsupervised learning approach, eliminating the need for large datasets typically required by supervised methods. The approach utilizes train wheel load data and bridge response during train crossing events as inputs for damage identification. The PINN model explicitly incorporates the governing differential equations of the linear time-varying (LTV) bridge-train system. Herein, this model employs a recurrent neural network (RNN) based architecture incorporating a custom Runge-Kutta (RK) integrator cell, designed for gradient-based learning. The proposed approach updates the bridge finite element model while also quantifying damage severity and localizing the affected structural members. A case study on the Calumet Bridge in Chicago, Illinois, with simulated damage scenarios, is used to demonstrate the model's effectiveness in identifying damage while maintaining low false-positive rates. Furthermore, the damage identification pipeline is designed to seamlessly integrate prior knowledge from inspections and drone surveys, also enabling context-aware updating and assessment of bridge's condition. 

**Abstract (ZH)**: 美国铁路桥梁是美国货运铁路系统的关键组成部分，该系统运输着全国超过40%的货物，并在经济中扮演着至关重要的角色。然而，桥梁老化基础设施和不断增加的列车交通给安全带来了严峻的隐患，并增加了服务中断的风险。美国铁路网络包括超过10万个铁路桥梁，平均每1.4英里轨道就有一个桥梁，其中钢桥约占网络总桥梁长度的50%以上。早期识别和评估这些桥梁的损伤仍然是一个具有挑战性的任务。本研究提出了一种基于物理信息神经网络（PINN）的方法，用于钢桁架铁路桥梁的损伤识别。该提出的方法采用无监督学习方法，消除了传统监督方法通常需要大量数据集的需求。该方法利用列车车轮负载数据和列车通过期间桥梁的响应作为损伤识别的输入。PINN模型明确包含了线性时变（LTV）桥梁-列车系统的基本微分方程。该模型采用基于递归神经网络（RNN）的架构，并结合了一个自定义的Runge-Kutta（RK）积分单元，设计用于基于梯度的学习。该提出的方法在更新桥梁有限元模型的同时，还量化了损伤程度并定位受影响的结构部件。通过对伊利诺伊州芝加哥市的Calumet桥进行损伤模拟场景的研究案例，该模型的有效性在识别损伤的同时保持了较低的假阳性率得到了验证。此外，损伤识别管道设计为无缝集成检查和无人机调查之前的先验知识，还能够实现情境感知的更新和桥梁状态评估。 

---
# Understanding Federated Learning from IID to Non-IID dataset: An Experimental Study 

**Title (ZH)**: 从同态数据集到非同态数据集的联邦学习理解：一项实验研究 

**Authors**: Jungwon Seo, Ferhat Ozgur Catak, Chunming Rong  

**Link**: [PDF](https://arxiv.org/pdf/2502.00182)  

**Abstract**: As privacy concerns and data regulations grow, federated learning (FL) has emerged as a promising approach for training machine learning models across decentralized data sources without sharing raw data. However, a significant challenge in FL is that client data are often non-IID (non-independent and identically distributed), leading to reduced performance compared to centralized learning. While many methods have been proposed to address this issue, their underlying mechanisms are often viewed from different perspectives. Through a comprehensive investigation from gradient descent to FL, and from IID to non-IID data settings, we find that inconsistencies in client loss landscapes primarily cause performance degradation in non-IID scenarios. From this understanding, we observe that existing methods can be grouped into two main strategies: (i) adjusting parameter update paths and (ii) modifying client loss landscapes. These findings offer a clear perspective on addressing non-IID challenges in FL and help guide future research in the field. 

**Abstract (ZH)**: 随着隐私关注和数据法规的增强，联邦学习（FL）已成为一种有前途的方法，可以在不共享原始数据的情况下跨分散的数据源训练机器学习模型。然而，联邦学习中的一个重大挑战是客户端数据通常是非IID（非独立同分布）的，这会导致性能降低，而这种性能降低在集中式学习中则不会出现。尽管已经提出了许多解决方案，但这些方法的基础机制往往是从不同的视角来进行分析的。通过从梯度下降到联邦学习、从IID数据设置到非IID数据设置的全面探究，我们发现客户端损失景观中的不一致性主要导致了非IID情景下的性能下降。基于这一理解，我们发现现有的方法可以分为两大类策略：（i）调整参数更新路径，以及（ii）修改客户端损失景观。这些发现为解决联邦学习中非IID挑战提供了清晰的视角，并有助于指导该领域的未来研究。 

---
# A Comprehensive Review: Applicability of Deep Neural Networks in Business Decision Making and Market Prediction Investment 

**Title (ZH)**: 全面综述：深度神经网络在企业决策和市场预测投资中的适用性 

**Authors**: Viet Trinh  

**Link**: [PDF](https://arxiv.org/pdf/2502.00151)  

**Abstract**: Big data, both in its structured and unstructured formats, have brought in unforeseen challenges in economics and business. How to organize, classify, and then analyze such data to obtain meaningful insights are the ever-going research topics for business leaders and academic researchers. This paper studies recent applications of deep neural networks in decision making in economical business and investment; especially in risk management, portfolio optimization, and algorithmic trading. Set aside limitation in data privacy and cross-market analysis, the article establishes that deep neural networks have performed remarkably in financial classification and prediction. Moreover, the study suggests that by compositing multiple neural networks, spanning different data type modalities, a more robust, efficient, and scalable financial prediction framework can be constructed. 

**Abstract (ZH)**: 大数据，无论是结构化还是非结构化数据，都给经济和商业带来了意想不到的挑战。如何组织、分类和分析这些数据以获得有意义的洞察是企业管理者和学术研究人员不断研究的课题。本文研究了深度神经网络在经济和商业决策以及投资中的最新应用，特别是风险管理和投资组合优化和算法交易方面。尽管在数据隐私和跨市场分析方面存在限制，文章表明深度神经网络在金融分类和预测方面表现优异。此外，研究还表明，通过组合多种神经网络，涵盖不同数据类型模式，可以构建出更稳健、高效和可扩展的金融预测框架。 

---
# Multimodal MRI-Ultrasound AI for Prostate Cancer Detection Outperforms Radiologist MRI Interpretation: A Multi-Center Study 

**Title (ZH)**: 多模态MRI-超声AI在前列腺癌检测中的性能优于放射科医生的MRI解释：一项多中心研究 

**Authors**: Hassan Jahanandish, Shengtian Sang, Cynthia Xinran Li, Sulaiman Vesal, Indrani Bhattacharya, Jeong Hoon Lee, Richard Fan, Geoffrey A. Sonna, Mirabela Rusu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00146)  

**Abstract**: Pre-biopsy magnetic resonance imaging (MRI) is increasingly used to target suspicious prostate lesions. This has led to artificial intelligence (AI) applications improving MRI-based detection of clinically significant prostate cancer (CsPCa). However, MRI-detected lesions must still be mapped to transrectal ultrasound (TRUS) images during biopsy, which results in missing CsPCa. This study systematically evaluates a multimodal AI framework integrating MRI and TRUS image sequences to enhance CsPCa identification. The study included 3110 patients from three cohorts across two institutions who underwent prostate biopsy. The proposed framework, based on the 3D UNet architecture, was evaluated on 1700 test cases, comparing performance to unimodal AI models that use either MRI or TRUS alone. Additionally, the proposed model was compared to radiologists in a cohort of 110 patients. The multimodal AI approach achieved superior sensitivity (80%) and Lesion Dice (42%) compared to unimodal MRI (73%, 30%) and TRUS models (49%, 27%). Compared to radiologists, the multimodal model showed higher specificity (88% vs. 78%) and Lesion Dice (38% vs. 33%), with equivalent sensitivity (79%). Our findings demonstrate the potential of multimodal AI to improve CsPCa lesion targeting during biopsy and treatment planning, surpassing current unimodal models and radiologists; ultimately improving outcomes for prostate cancer patients. 

**Abstract (ZH)**: 磁共振成像（MRI）在活检前已被广泛用于定位可疑的前列腺病变。这促使了人工智能（AI）在基于MRI的临床显著前列腺癌（CsPCa）检测中的应用得到了改进。然而，在进行经直肠超声（TRUS）引导的活检时，仍需将MRI检测到的病变与TRUS图像进行匹配，这可能导致遗漏CsPCa。本研究系统评估了一种整合MRI和TRUS图像序列的多模态AI框架，以提高CsPCa的识别能力。该研究包括了两个机构的三个队列共3110例接受前列腺活检的患者。基于3D UNet结构提出的框架，在1700个测试案例中进行了评估，将其性能与仅使用MRI或TRUS的单模态AI模型进行了比较。此外，该模型还与110例患者的放射科医师进行了比较。多模态AI方法在敏感性（80%）和病灶Dice系数（42%）方面优于单模态MRI（73%，30%）和TRUS模型（49%，27%）。与放射科医师相比，多模态模型在特异性（88% vs. 78%）和病灶Dice系数（38% vs. 33%）上表现更好，而灵敏度相当（79%）。我们的研究结果表明，多模态AI有可能提高CsPCa病灶在活检和治疗规划中的定位能力，超越现有的单模态模型和放射科医师，最终改善前列腺癌患者的预后。 

---
# Demystifying MPNNs: Message Passing as Merely Efficient Matrix Multiplication 

**Title (ZH)**: 揭开MPNNs的面纱：消息传递仅是高效的矩阵乘法 

**Authors**: Qin Jiang, Chengjia Wang, Michael Lones, Wei Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00140)  

**Abstract**: While Graph Neural Networks (GNNs) have achieved remarkable success, their design largely relies on empirical intuition rather than theoretical understanding. In this paper, we present a comprehensive analysis of GNN behavior through three fundamental aspects: (1) we establish that \textbf{$k$-layer} Message Passing Neural Networks efficiently aggregate \textbf{$k$-hop} neighborhood information through iterative computation, (2) analyze how different loop structures influence neighborhood computation, and (3) examine behavior across structure-feature hybrid and structure-only tasks. For deeper GNNs, we demonstrate that gradient-related issues, rather than just over-smoothing, can significantly impact performance in sparse graphs. We also analyze how different normalization schemes affect model performance and how GNNs make predictions with uniform node features, providing a theoretical framework that bridges the gap between empirical success and theoretical understanding. 

**Abstract (ZH)**: 尽管图神经网络（GNNs）已经取得了显著的成功，但其设计很大程度上依赖于实验直觉而非理论理解。在本文中，我们通过三个基本方面对GNN的行为进行了全面分析：(1) 我们建立了**k层**消息传递神经网络通过迭代计算有效地聚合**k跳**邻域信息的事实，(2) 分析不同的循环结构如何影响邻域计算，以及(3) 考察其在结构特征混合任务和只涉及结构的任务中的表现。对于更深的GNNs，我们证明了梯度相关问题，而不是仅仅因为过平滑现象，会在稀疏图中显著影响性能。我们还分析了不同的规范化方案如何影响模型性能，并探讨GNN在节点特征均匀时的预测方式，从而提供了从实验成功到理论理解的理论框架。 

---
# A Three-Branch Checks-and-Balances Frameworkfor Context-Aware Ethical Alignment of Large Language Models 

**Title (ZH)**: 一种三分支制衡框架：面向情境感知的大型语言模型伦理协同校准 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00136)  

**Abstract**: This paper introduces a three-branch checks-and-balances framework for ethical alignment of Large Language Models (LLMs), inspired by governmental systems. It implements three independent yet interacting components: LLMs as the executive branch for knowledge generation, DIKE as the legislative branch establishing ethical guardrails, and ERIS as the judicial branch for contextual interpretation. The adversarial DIKE-ERIS duality enables adaptation to diverse cultural contexts while upholding consistent ethical principles. This architecture addresses limitations of reinforcement learning with human feedback (RLHF) by providing interpretable, adaptable, and culturally-aware ethical reasoning. Through self-supervised learning and adversarial testing, our framework demonstrates how emotional modeling can guide linguistic behaviors toward ethical outcomes while preserving independence across knowledge generation, ethical oversight, and contextual interpretation. 

**Abstract (ZH)**: 本文介绍了一种以政府系统为灵感的三大分支制衡框架，用于大型语言模型（LLMs）的伦理对齐。该框架包括三个独立但相互作用的组成部分：作为执行分支进行知识生成的LLMs、作为立法分支制定伦理规范的DIKE，以及作为司法分支进行情境解释的ERIS。敌对的DIKE-ERIS二元性使系统能够适应多样的文化背景，同时保持一致的伦理原则。该架构通过提供可解释、可适应以及文化敏感的伦理推理解决了强化学习结合人类反馈（RLHF）的局限性。通过自我监督学习和对抗性测试，我们的框架展示了情感建模如何引导语言行为走向伦理结果，同时在知识生成、伦理监督和情境解释方面保持独立性。 

---
# Exploring Transfer Learning for Deep Learning Polyp Detection in Colonoscopy Images Using YOLOv8 

**Title (ZH)**: 利用YOLOv8探讨迁移学习在结肠镜图像结肠息肉检测中的应用 

**Authors**: Fabian Vazquez, Jose Angel Nuñez, Xiaoyan Fu, Pengfei Gu, Bin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00133)  

**Abstract**: Deep learning methods have demonstrated strong performance in objection tasks; however, their ability to learn domain-specific applications with limited training data remains a significant challenge. Transfer learning techniques address this issue by leveraging knowledge from pre-training on related datasets, enabling faster and more efficient learning for new tasks. Finding the right dataset for pre-training can play a critical role in determining the success of transfer learning and overall model performance. In this paper, we investigate the impact of pre-training a YOLOv8n model on seven distinct datasets, evaluating their effectiveness when transferred to the task of polyp detection. We compare whether large, general-purpose datasets with diverse objects outperform niche datasets with characteristics similar to polyps. In addition, we assess the influence of the size of the dataset on the efficacy of transfer learning. Experiments on the polyp datasets show that models pre-trained on relevant datasets consistently outperform those trained from scratch, highlighting the benefit of pre-training on datasets with shared domain-specific features. 

**Abstract (ZH)**: 深度学习方法在检测任务中展示了强大的性能，然而在有限训练数据下学习特定领域的应用仍面临着重大挑战。迁移学习技术通过利用相关数据集上预训练的知识，解决了这一问题，使新任务的学习更加迅速且高效。选择合适的预训练数据集对于确定迁移学习的成功和整体模型性能至关重要。在本文中，我们探讨了在七个不同数据集上预训练YOLOv8n模型的影响，评估了这些数据集在转移到息肉检测任务时的效果。我们比较了大规模、具有多样目标的通用数据集与具有类似息Toronto特征的专用数据集之间的效果。此外，我们还评估了数据集大小对迁移学习效果的影响。在息肉数据集上的实验表明，预训练于相关数据集的模型始终优于从零开始训练的模型，突显了在具有共同领域特征的数据集上进行预训练的好处。 

---
# AIN: The Arabic INclusive Large Multimodal Model 

**Title (ZH)**: AIN：阿拉伯语包容性大规模多模态模型 

**Authors**: Ahmed Heakl, Sara Ghaboura, Omkar Thawkar, Fahad Shahbaz Khan, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00094)  

**Abstract**: Amid the swift progress of large language models (LLMs) and their evolution into large multimodal models (LMMs), significant strides have been made in high-resource languages such as English and Chinese. While Arabic LLMs have seen notable progress, Arabic LMMs remain largely unexplored, often narrowly focusing on a few specific aspects of the language and visual understanding. To bridge this gap, we introduce AIN-the Arabic Inclusive Multimodal Model-designed to excel across diverse domains. AIN is an English-Arabic bilingual LMM designed to excel in English and Arabic, leveraging carefully constructed 3.6 million high-quality Arabic-English multimodal data samples. AIN demonstrates state-of-the-art Arabic performance, while also possessing strong English-language visual capabilities. On the recent CAMEL-Bench benchmark comprising 38 sub-domains including, multi-image understanding, complex visual perception, handwritten document understanding, video understanding, medical imaging, plant diseases, and remote sensing-based land use understanding, our AIN demonstrates strong performance with the 7B model outperforming GPT-4o by an absolute gain of 3.4% averaged over eight domains and 38 sub-domains. AIN's superior capabilities position it as a significant step toward empowering Arabic speakers with advanced multimodal generative AI tools across diverse applications. 

**Abstract (ZH)**: 在大型语言模型（LLMs）和大型多模态模型（LMMs）迅速发展和演变的过程中，英语和中文等高资源语言取得了显著进展。虽然阿拉伯语LLMs取得了显著进步，但阿拉伯语LMMs依然鲜有探索，往往仅仅集中在语言和视觉理解的几个特定方面。为了弥合这一差距，我们引入了AIN——阿拉伯 inclusive 多模态模型，旨在在多个领域表现出色。AIN是一个双语的英阿LMM，旨在擅长英语和阿拉伯语，同时利用精心构建的360万高质量的英阿多模态数据样本。AIN在阿拉伯语任务上达到了最先进的性能，同时也具备强大的英文视觉能力。在近期包含38个子领域的CAMEL-Bench基准测试中，这些子领域涉及多图像理解、复杂视觉感知、手写文档理解、视频理解、医学成像、植物病害识别以及基于遥感的土地利用理解，我们的AIN在7B模型上表现出色，在八个领域和38个子领域的绝对增益达到了3.4%。AIN的卓越能力使其成为跨多种应用领域赋予阿拉伯语使用者先进多模态生成AI工具的重要一步。 

---
# Ensembles of Low-Rank Expert Adapters 

**Title (ZH)**: 低秩专家适配器的ensemble方法 

**Authors**: Yinghao Li, Vianne Gao, Chao Zhang, MohamadAli Torkamani  

**Link**: [PDF](https://arxiv.org/pdf/2502.00089)  

**Abstract**: The training and fine-tuning of large language models (LLMs) often involve diverse textual data from multiple sources, which poses challenges due to conflicting gradient directions, hindering optimization and specialization. These challenges can undermine model generalization across tasks, resulting in reduced downstream performance. Recent research suggests that fine-tuning LLMs on carefully selected, task-specific subsets of data can match or even surpass the performance of using the entire dataset. Building on these insights, we propose the Ensembles of Low-Rank Expert Adapters (ELREA) framework to improve the model's capability to handle diverse tasks. ELREA clusters the training instructions based on their gradient directions, representing different areas of expertise and thereby reducing conflicts during optimization. Expert adapters are then trained on these clusters, utilizing the low-rank adaptation (LoRA) technique to ensure training efficiency and model scalability. During inference, ELREA combines predictions from the most relevant expert adapters based on the input data's gradient similarity to the training clusters, ensuring optimal adapter selection for each task. Experiments show that our method outperforms baseline LoRA adapters trained on the full dataset and other ensemble approaches with similar training and inference complexity across a range of domain-specific tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）的训练和微调通常涉及源自多个数据源的多样文本数据，这会导致矛盾的方向梯度，从而妨碍优化和专业化。这些挑战可能削弱模型在不同任务上的泛化能力，导致下游性能下降。近期研究显示，通过对任务特定部分数据进行精心选择和微调，可以达到甚至超越利用整个数据集的效果。基于这些发现，我们提出了一种低秩专家适配器集合（ELREA）框架，以提高模型处理多样任务的能力。ELREA 根据梯度方向对训练指令进行聚类，表示不同的专业领域，从而在优化过程中减少冲突。然后，使用低秩适应（LoRA）技术训练专家适配器，以确保训练效率和模型的可扩展性。在推理过程中，ELREA 根据输入数据与训练集群的梯度相似性组合最相关的专家适配器的预测，从而为每个任务选择最佳适配器。实验结果显示，该方法在多个专业领域的任务中，相对于在完整数据集上训练的标准 LoRA 适配器和其他具有相似训练和推理复杂性的集成方法，均表现出更好的性能。 

---
# Influence of color correction on pathology detection in Capsule Endoscopy 

**Title (ZH)**: 胶囊内镜中颜色校正对病理检测的影响研究 

**Authors**: Bidossessi Emmanuel Agossou, Marius Pedersen, Kiran Raja, Anuja Vats, Pål Anders Floor  

**Link**: [PDF](https://arxiv.org/pdf/2502.00076)  

**Abstract**: Pathology detection in Wireless Capsule Endoscopy (WCE) using deep learning has been explored in the recent past. However, deep learning models can be influenced by the color quality of the dataset used to train them, impacting detection, segmentation and classification tasks. In this work, we evaluate the impact of color correction on pathology detection using two prominent object detection models: Retinanet and YOLOv5. We first generate two color corrected versions of a popular WCE dataset (i.e., SEE-AI dataset) using two different color correction functions. We then evaluate the performance of the Retinanet and YOLOv5 on the original and color corrected versions of the dataset. The results reveal that color correction makes the models generate larger bounding boxes and larger intersection areas with the ground truth annotations. Furthermore, color correction leads to an increased number of false positives for certain pathologies. However, these effects do not translate into a consistent improvement in performance metrics such as F1-scores, IoU, and AP50. The code is available at this https URL. Keywords: Wireless Capsule Endoscopy, Color correction, Retinanet, YOLOv5, Detection 

**Abstract (ZH)**: 近年来，使用深度学习进行无线胶囊内镜（WCE）中的病理检测已经得到了探索。然而，训练数据集的颜色质量可能会影响深度学习模型的表现，从而影响检测、分割和分类任务。在本研究中，我们评估了颜色校正对使用两种主流目标检测模型（Retinanet和YOLOv5）进行病理检测的影响。首先，我们使用两种不同的颜色校正功能生成了一个流行WCE数据集（即SEE-AI数据集）的两个颜色校正版本。然后，我们评估了Retinanet和YOLOv5在这原始和颜色校正数据集上的性能。结果显示，颜色校正使模型生成更大的边界框和更大的与真实标注的交集区域。此外，颜色校正导致某些病理情况下的假阳性增多。然而，这些效果并未转化为在F1分数、交并比（IoU）和AP50等性能指标上的一致提升。相关代码可在以下链接获取：this https URL。

关键词：无线胶囊内镜，颜色校正，Retinanet，YOLOv5，检测 

---
# SpikingRTNH: Spiking Neural Network for 4D Radar Object Detection 

**Title (ZH)**: SpikingRTNH：用于四维雷达目标检测的脉冲神经网络 

**Authors**: Dong-Hee Paek, Seung-Hyun Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.00074)  

**Abstract**: Recently, 4D Radar has emerged as a crucial sensor for 3D object detection in autonomous vehicles, offering both stable perception in adverse weather and high-density point clouds for object shape recognition. However, processing such high-density data demands substantial computational resources and energy consumption. We propose SpikingRTNH, the first spiking neural network (SNN) for 3D object detection using 4D Radar data. By replacing conventional ReLU activation functions with leaky integrate-and-fire (LIF) spiking neurons, SpikingRTNH achieves significant energy efficiency gains. Furthermore, inspired by human cognitive processes, we introduce biological top-down inference (BTI), which processes point clouds sequentially from higher to lower densities. This approach effectively utilizes points with lower noise and higher importance for detection. Experiments on K-Radar dataset demonstrate that SpikingRTNH with BTI significantly reduces energy consumption by 78% while achieving comparable detection performance to its ANN counterpart (51.1% AP 3D, 57.0% AP BEV). These results establish the viability of SNNs for energy-efficient 4D Radar-based object detection in autonomous driving systems. All codes are available at this https URL. 

**Abstract (ZH)**: 近年来，4D雷达已成为自动驾驶车辆中三维物体检测的关键传感器，它能够在恶劣天气条件下提供稳定的感知，并通过高密度点云实现物体形状识别。然而，处理这类高密度数据需要大量的计算资源和能耗。为此，我们提出了一种名为SpikingRTNH的新方法，这是首个用于4D雷达数据三维物体检测的脉冲神经网络（SNN）。通过将传统的ReLU激活函数替换为漏电流整合发放（LIF）脉冲神经元，SpikingRTNH显著提高了能效。此外，受人类认知过程的启发，我们引入了生物自上而下推理（BTI），这种方法依次处理从高密度到低密度的点云数据。这种做法有效地利用了噪声更低、对检测更重要的点进行处理。在K-Radar数据集上的实验结果显示，与其对应的非脉冲神经网络（ANN）相比，结合了BTI的SpikingRTNH在保持检测性能相似的情况下（3D AP 51.1%，BEV AP 57.0%），能耗降低了78%。这些结果证明了SNN在自动驾驶系统中基于4D雷达的高效物体检测中的可行性。所有源代码均可从以下链接获取：[此 https URL]。 

---
# LLM Cyber Evaluations Don't Capture Real-World Risk 

**Title (ZH)**: 大型语言模型的网络安全评估无法捕捉到实际风险 

**Authors**: Kamilė Lukošiūtė, Adam Swanda  

**Link**: [PDF](https://arxiv.org/pdf/2502.00072)  

**Abstract**: Large language models (LLMs) are demonstrating increasing prowess in cybersecurity applications, creating creating inherent risks alongside their potential for strengthening defenses. In this position paper, we argue that current efforts to evaluate risks posed by these capabilities are misaligned with the goal of understanding real-world impact. Evaluating LLM cybersecurity risk requires more than just measuring model capabilities -- it demands a comprehensive risk assessment that incorporates analysis of threat actor adoption behavior and potential for impact. We propose a risk assessment framework for LLM cyber capabilities and apply it to a case study of language models used as cybersecurity assistants. Our evaluation of frontier models reveals high compliance rates but moderate accuracy on realistic cyber assistance tasks. However, our framework suggests that this particular use case presents only moderate risk due to limited operational advantages and impact potential. Based on these findings, we recommend several improvements to align research priorities with real-world impact assessment, including closer academia-industry collaboration, more realistic modeling of attacker behavior, and inclusion of economic metrics in evaluations. This work represents an important step toward more effective assessment and mitigation of LLM-enabled cybersecurity risks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在网络安全应用方面展现出不断增强的能力，同时也伴随着固有的风险。本文认为，当前用于评估这些能力所带来的风险的努力与理解实际影响的目标存在错位。评估LLM的网络安全风险不仅仅需要衡量模型的能力，还需要进行全面的风险评估，结合对手行为分析和实际影响的可能性。我们提出了一种针对LLM网络安全能力的风险评估框架，并将其应用于使用语言模型作为网络安全助手的案例研究中。对前沿模型的评估显示了较高的合规率，但在现实的网络安全辅助任务中准确度中等。然而，我们的框架表明，这种特定的应用场景的风险程度仅为中等，因为其操作优势和潜在影响有限。基于这些发现，我们建议通过加强产学研合作、更现实地建模攻击者行为以及在评估中纳入经济指标等措施来调整研究重点，以更好地与实际影响评估相一致。本研究代表了对LLM带来的网络安全风险更为有效评估和缓解的重要一步。 

---
# Can AI Solve the Peer Review Crisis? A Large Scale Experiment on LLM's Performance and Biases in Evaluating Economics Papers 

**Title (ZH)**: 人工智能能解决同行评审危机吗？一项大型实验探究LLM在评估经济论文时的表现与偏见 

**Authors**: Pat Pataranutaporn, Nattavudh Powdthavee, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2502.00070)  

**Abstract**: We investigate whether artificial intelligence can address the peer review crisis in economics by analyzing 27,090 evaluations of 9,030 unique submissions using a large language model (LLM). The experiment systematically varies author characteristics (e.g., affiliation, reputation, gender) and publication quality (e.g., top-tier, mid-tier, low-tier, AI generated papers). The results indicate that LLMs effectively distinguish paper quality but exhibit biases favoring prominent institutions, male authors, and renowned economists. Additionally, LLMs struggle to differentiate high-quality AI-generated papers from genuine top-tier submissions. While LLMs offer efficiency gains, their susceptibility to bias necessitates cautious integration and hybrid peer review models to balance equity and accuracy. 

**Abstract (ZH)**: 我们通过分析9030篇独特提交论文的27,090份评估结果，使用大型语言模型（LLM）来探讨人工智能是否能解决经济学同行评审危机。实验系统地变化了作者特征（如：隶属关系、声誉、性别）以及出版质量（如：顶级、中等级、低等级、AI生成的文章）。结果表明，LLM能够有效区分论文质量，但表现出对知名机构、男性作者和著名经济学家的偏见。此外，LLM难以区分高质量的AI生成论文和真正的顶级论文。虽然LLM提供了效率上的改进，但其易受偏见的影响需要谨慎整合，并采用混合同行评审模式以平衡公平性和准确性。 

---
# Privacy Preserving Charge Location Prediction for Electric Vehicles 

**Title (ZH)**: 电动汽车隐私保护充电位置预测 

**Authors**: Robert Marlin, Raja Jurdak, Alsharif Abuadbba, Dimity Miller  

**Link**: [PDF](https://arxiv.org/pdf/2502.00068)  

**Abstract**: By 2050, electric vehicles (EVs) are projected to account for 70% of global vehicle sales. While EVs provide environmental benefits, they also pose challenges for energy generation, grid infrastructure, and data privacy. Current research on EV routing and charge management often overlooks privacy when predicting energy demands, leaving sensitive mobility data vulnerable. To address this, we developed a Federated Learning Transformer Network (FLTN) to predict EVs' next charge location with enhanced privacy measures. Each EV operates as a client, training an onboard FLTN model that shares only model weights, not raw data with a community-based Distributed Energy Resource Management System (DERMS), which aggregates them into a community global model. To further enhance privacy, non-transitory EVs use peer-to-peer weight sharing and augmentation within their community, obfuscating individual contributions and improving model accuracy. Community DERMS global model weights are then redistributed to EVs for continuous training. Our FLTN approach achieved up to 92% accuracy while preserving data privacy, compared to our baseline centralised model, which achieved 98% accuracy with no data privacy. Simulations conducted across diverse charge levels confirm the FLTN's ability to forecast energy demands over extended periods. We present a privacy-focused solution for forecasting EV charge location prediction, effectively mitigating data leakage risks. 

**Abstract (ZH)**: 到2050年，电动汽车（EVs）预计将占全球汽车销量的70%。尽管电动汽车提供了环境效益，但它们也对能源生成、电网基础设施以及数据隐私提出了挑战。目前关于电动汽车路径规划和充电管理的研究往往在预测能源需求时忽视了隐私问题，导致敏感的出行数据存在泄露风险。为解决这一问题，我们开发了一种联邦学习变换器网络（FLTN）来预测电动汽车的下一个充电地点，并增强了隐私保护措施。每辆电动汽车作为客户端，训练一个本地FLTN模型，仅将模型权重而不是原始数据与其他基于分布式能源资源管理系统的社区成员共享，这些权重被聚合生成一个社区级全局模型。为了进一步增强隐私保护，非永久性电动汽车在其社区内部进行点对点权重共享和增补，使个体贡献变得模糊，从而提高模型的准确性。社区级分布式能源资源管理系统的全局模型权重随后重新分配给每辆电动汽车进行持续训练。我们的FLTN方法在保持数据隐私的同时达到了92%的准确率，相比之下，我们基准的集中式模型达到了98%的准确率，但没有数据隐私。我们进行了跨不同充电水平的模拟实验，证实了FLTN在长时间内预测能源需求的能力。我们提出了一种隐私保护型解决方案，用于预测电动汽车充电位置，有效缓解了数据泄露风险。 

---
# A Multi-Layered Large Language Model Framework for Disease Prediction 

**Title (ZH)**: 一种多层大型语言模型框架用于疾病预测 

**Authors**: Malak Mohamed, Rokaia Emad, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2502.00063)  

**Abstract**: Social telehealth has revolutionized healthcare by enabling patients to share symptoms and receive medical consultations remotely. Users frequently post symptoms on social media and online health platforms, generating a vast repository of medical data that can be leveraged for disease classification and symptom severity assessment. Large language models (LLMs), such as LLAMA3, GPT-3.5 Turbo, and BERT, process complex medical data to enhance disease classification. This study explores three Arabic medical text preprocessing techniques: text summarization, text refinement, and Named Entity Recognition (NER). Evaluating CAMeL-BERT, AraBERT, and Asafaya-BERT with LoRA, the best performance was achieved using CAMeL-BERT with NER-augmented text (83% type classification, 69% severity assessment). Non-fine-tuned models performed poorly (13%-20% type classification, 40%-49% severity assessment). Integrating LLMs into social telehealth systems enhances diagnostic accuracy and treatment outcomes. 

**Abstract (ZH)**: 社会远程医疗通过使患者能够在远程环境中分享症状并接受医疗咨询，从而颠覆了医疗保健领域。用户经常在社交媒体和在线健康平台上发布症状，生成了大量的医学数据，这些数据可以用于疾病分类和症状严重程度评估。大规模语言模型（LLM），如LLAMA3、GPT-3.5 Turbo和BERT，处理复杂的医学数据以提高疾病分类的准确性。本研究探索了三种阿拉伯医学文本预处理技术：文本总结、文本精炼和命名实体识别（NER）。通过对CAMeL-BERT、AraBERT和Asafaya-BERT结合LoRA进行评估，使用NER增强的CAMeL-BERT模型表现出最佳性能（类型分类准确率为83%，严重程度评估准确率为69%）。未经微调的模型表现不佳（类型分类准确率为13%-20%，严重程度评估准确率为40%-49%）。将LLM集成到社会远程医疗系统中可以提高诊断准确性并改善治疗结果。 

---
# From Data to Action: Charting A Data-Driven Path to Combat Antimicrobial Resistance 

**Title (ZH)**: 从数据到行动：一条数据驱动的抗微生物Resistance防控路径 

**Authors**: Qian Fu, Yuzhe Zhang, Yanfeng Shu, Ming Ding, Lina Yao, Chen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00061)  

**Abstract**: Antimicrobial-resistant (AMR) microbes are a growing challenge in healthcare, rendering modern medicines ineffective. AMR arises from antibiotic production and bacterial evolution, but quantifying its transmission remains difficult. With increasing AMR-related data, data-driven methods offer promising insights into its causes and treatments. This paper reviews AMR research from a data analytics and machine learning perspective, summarizing the state-of-the-art and exploring key areas such as surveillance, prediction, drug discovery, stewardship, and driver analysis. It discusses data sources, methods, and challenges, emphasizing standardization and interoperability. Additionally, it surveys statistical and machine learning techniques for AMR analysis, addressing issues like data noise and bias. Strategies for denoising and debiasing are highlighted to enhance fairness and robustness in AMR research. The paper underscores the importance of interdisciplinary collaboration and awareness of data challenges in advancing AMR research, pointing to future directions for innovation and improved methodologies. 

**Abstract (ZH)**: 抗微生物耐药性（AMR）微生物已成为医疗领域日益严峻的挑战，使得现代药物失效。AMR 是由抗生素生产和细菌进化造成的，但其传播的量化仍然具有挑战性。随着 AMR 相关数据的增加，数据驱动的方法为深入研究其成因和治疗提供了有前景的洞察。本文从数据挖掘和机器学习的角度回顾 AMR 研究，总结了当前的技术前沿，并探讨了监督、预测、药物发现、合理使用以及驱动因素分析等重要领域。文章讨论了数据来源、方法和挑战，强调了标准化和互操作性的重要性。此外，还概述了统计学和机器学习技术在 AMR 分析中的应用，解决了数据噪声和偏差等问题。强调了降噪和去偏差策略的使用，以提高 AMR 研究中的公平性和鲁棒性。本文突出了跨学科合作和意识的必要性，以促进 AMR 研究的发展，并指出了未来创新和改进方法的方向。 

---
# Israel-Hamas war through Telegram, Reddit and Twitter 

**Title (ZH)**: 以色列-哈马斯战争在Telegram、Reddit和Twitter上的表现 

**Authors**: Despoina Antonakaki, Sotiris Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2502.00060)  

**Abstract**: The Israeli-Palestinian conflict started on 7 October 2023, have resulted thus far to over 48,000 people killed including more than 17,000 children with a majority from Gaza, more than 30,000 people injured, over 10,000 missing, and over 1 million people displaced, fleeing conflict zones. The infrastructure damage includes the 87\% of housing units, 80\% of public buildings and 60\% of cropland 17 out of 36 hospitals, 68\% of road networks and 87\% of school buildings damaged. This conflict has as well launched an online discussion across various social media platforms. Telegram was no exception due to its encrypted communication and highly involved audience. The current study will cover an analysis of the related discussion in relation to different participants of the conflict and sentiment represented in those discussion. To this end, we prepared a dataset of 125K messages shared on channels in Telegram spanning from 23 October 2025 until today. Additionally, we apply the same analysis in two publicly available datasets from Twitter containing 2001 tweets and from Reddit containing 2M opinions. We apply a volume analysis across the three datasets, entity extraction and then proceed to BERT topic analysis in order to extract common themes or topics. Next, we apply sentiment analysis to analyze the emotional tone of the discussions. Our findings hint at polarized narratives as the hallmark of how political factions and outsiders mold public opinion. We also analyze the sentiment-topic prevalence relationship, detailing the trends that may show manipulation and attempts of propaganda by the involved parties. This will give a better understanding of the online discourse on the Israel-Palestine conflict and contribute to the knowledge on the dynamics of social media communication during geopolitical crises. 

**Abstract (ZH)**: 以色列与巴勒斯坦的冲突始于2023年10月7日，至今已导致超过48,000人遇难，其中超过17,000人为儿童，主要来自加沙地带；超过30,000人受伤，超过10,000人失踪，超过100万人被迫流离失所，逃离冲突区。基础设施破坏包括87%的住房单元、80%的公共建筑、60%的耕地、36家医院中的17家、68%的道路网络和87%的学校建筑受损。此次冲突还引发了各社交媒体平台上的在线讨论。Telegram也不乏参与者，由于其加密通信和活跃用户群，因而特别引人关注。目前的研究将涵盖对相关讨论的分析，特别是涉及冲突各方参与者及讨论中体现的情感。为此，我们准备了一个数据集，包含自2025年10月23日至今在Telegram上共享的125,000条消息。此外，我们还在Twitter和Reddit两个公开可用的数据集中分别进行了同样的分析，Twitter数据集包含2001条推文，Reddit数据集包含2,000,000条意见。我们对三个数据集进行了规模分析，提取实体，然后应用BERT主题分析以提取常见主题或话题。接下来，我们应用情感分析来分析讨论中的情感倾向。研究发现表明，极化叙事是政治派别和局外人塑造公众意见的特征。我们还将分析情感-主题的普适性关系，详细探讨表明参与方操纵和企图传播宣传的趋势。这将更好地理解以色列-巴勒斯坦冲突的在线讨论，并为社会媒体在地缘政治危机期间的交流动态提供知识贡献。 

---
# Large Language Models are Few-shot Multivariate Time Series Classifiers 

**Title (ZH)**: 大型语言模型是 Few-Shot 多变量时间序列分类器 

**Authors**: Yakun Chen, Zihao Li, Chao Yang, Xianzhi Wang, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00059)  

**Abstract**: Large Language Models (LLMs) have been extensively applied in time series analysis. Yet, their utility in the few-shot classification (i.e., a crucial training scenario due to the limited training data available in industrial applications) concerning multivariate time series data remains underexplored. We aim to leverage the extensive pre-trained knowledge in LLMs to overcome the data scarcity problem within multivariate time series. Specifically, we propose LLMFew, an LLM-enhanced framework to investigate the feasibility and capacity of LLMs for few-shot multivariate time series classification. This model introduces a Patch-wise Temporal Convolution Encoder (PTCEnc) to align time series data with the textual embedding input of LLMs. We further fine-tune the pre-trained LLM decoder with Low-rank Adaptations (LoRA) to enhance its feature representation learning ability in time series data. Experimental results show that our model outperformed state-of-the-art baselines by a large margin, achieving 125.2% and 50.2% improvement in classification accuracy on Handwriting and EthanolConcentration datasets, respectively. Moreover, our experimental results demonstrate that LLM-based methods perform well across a variety of datasets in few-shot MTSC, delivering reliable results compared to traditional models. This success paves the way for their deployment in industrial environments where data are limited. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在时间序列分析中已被广泛应用于各种场景。然而，在涉及多变量时间序列数据的少量样本分类（即由于工业应用中可用训练数据量有限而成为一个关键的训练场景）方面，其应用潜力仍未得到充分探索。我们旨在利用大规模语言模型中丰富的预训练知识，以克服多变量时间序列中的数据稀缺问题。具体而言，我们提出了一种LLM增强框架LLMFew，以探究LLMs在少量样本多变量时间序列分类中的可行性和能力。该模型引入了一种基于补丁的时序卷积编码器（PTCEnc），用于将时间序列数据与LLM的文本嵌入输入对齐。进一步地，我们通过低秩适应（LoRA）微调预训练的LLM解码器，以增强其在时间序列数据中的特征表示学习能力。实验结果显示，我们的模型在Handwriting和EthanolConcentration数据集上的分类准确率上显著优于现有最先进的基线方法，分别提高了125.2%和50.2%。此外，我们的实验结果表明，基于LLM的方法在少量样本多变量时间序列分类（Few-shot MTSC）的各种数据集上表现良好，相比传统模型，可以提供可靠的性能。这一成功为这些方法在数据有限的工业环境中的应用铺平了道路。 

---
# Towards Recommender Systems LLMs Playground (RecSysLLMsP): Exploring Polarization and Engagement in Simulated Social Networks 

**Title (ZH)**: 《向推荐系统中大规模语言模型的试验场迈进（RecSysLLMsP）：探究模拟社会网络中的极化与参与》 

**Authors**: Ljubisa Bojic, Zorica Dodevska, Yashar Deldjoo, Nenad Pantelic  

**Link**: [PDF](https://arxiv.org/pdf/2502.00055)  

**Abstract**: Given the exponential advancement in AI technologies and the potential escalation of harmful effects from recommendation systems, it is crucial to simulate and evaluate these effects early on. Doing so can help prevent possible damage to both societies and technology companies. This paper introduces the Recommender Systems LLMs Playground (RecSysLLMsP), a novel simulation framework leveraging Large Language Models (LLMs) to explore the impacts of different content recommendation setups on user engagement and polarization in social networks. By creating diverse AI agents (AgentPrompts) with descriptive, static, and dynamic attributes, we assess their autonomous behaviour across three scenarios: Plurality, Balanced, and Similarity. Our findings reveal that the Similarity Scenario, which aligns content with user preferences, maximizes engagement while potentially fostering echo chambers. Conversely, the Plurality Scenario promotes diverse interactions but produces mixed engagement results. Our study emphasizes the need for a careful balance in recommender system designs to enhance user satisfaction while mitigating societal polarization. It underscores the unique value and challenges of incorporating LLMs into simulation environments. The benefits of RecSysLLMsP lie in its potential to calculate polarization effects, which is crucial for assessing societal impacts and determining user engagement levels with diverse recommender system setups. This advantage is essential for developing and maintaining a successful business model for social media companies. However, the study's limitations revolve around accurately emulating reality. Future efforts should validate the similarity in behaviour between real humans and AgentPrompts and establish metrics for measuring polarization scores. 

**Abstract (ZH)**: 鉴于人工智能技术的指数级发展以及推荐系统潜在的有害影响升级，提前模拟和评估这些影响变得至关重要。这样做可以帮助防止对社会和技术公司可能造成的损害。本文介绍了利用大型语言模型（LLMs）探索不同内容推荐设置对社交网络用户参与度和极化影响的新型模拟框架——推荐系统大型语言模型游乐场（RecSysLLMsP）。

通过创建具有描述性、静态和动态属性的多样化AI代理（AgentPrompts），我们评估了它们在三种情景下的自主行为：多元性、平衡性和平行性。我们的研究结果表明，平行性情景，即内容与用户偏好相匹配的情景，能够最大化参与度，但可能会促进回声室效应。相反，多元性情景促进了多样化互动，但参与度结果并不一致。我们的研究强调，在构建推荐系统时需要谨慎平衡，以增强用户满意度并减轻社会极化的风险。它强调了将LLMs整合到模拟环境中独特价值及其所面临的挑战。RecSysLLMsP 的优势在于其能够计算极化效应，这对于评估社会影响及确定不同推荐系统配置下的用户参与度至关重要。这种优势对于社交媒体公司发展和维持成功商业模式至关重要。然而，该研究的局限性在于准确模拟现实的难度。未来的研究应验证AgentPrompts与真实人类行为的一致性，并建立衡量极化得分的指标。 

---
# Bridging Contrastive Learning and Domain Adaptation: Theoretical Perspective and Practical Application 

**Title (ZH)**: 对比学习与领域适应的桥梁构建：理论视角与 Practical Application 实践应用 

**Authors**: Gonzalo Iñaki Quintana, Laurence Vancamberg, Vincent Jugnon, Agnès Desolneux, Mathilde Mougeot  

**Link**: [PDF](https://arxiv.org/pdf/2502.00052)  

**Abstract**: This work studies the relationship between Contrastive Learning and Domain Adaptation from a theoretical perspective. The two standard contrastive losses, NT-Xent loss (Self-supervised) and Supervised Contrastive loss, are related to the Class-wise Mean Maximum Discrepancy (CMMD), a dissimilarity measure widely used for Domain Adaptation. Our work shows that minimizing the contrastive losses decreases the CMMD and simultaneously improves class-separability, laying the theoretical groundwork for the use of Contrastive Learning in the context of Domain Adaptation. Due to the relevance of Domain Adaptation in medical imaging, we focused the experiments on mammography images. Extensive experiments on three mammography datasets - synthetic patches, clinical (real) patches, and clinical (real) images - show improved Domain Adaptation, class-separability, and classification performance, when minimizing the Supervised Contrastive loss. 

**Abstract (ZH)**: 本研究从理论角度探讨了对比学习与领域适应之间的关系。两种标准的对比损失函数，自监督的NT-Xent损失和监督对比损失，与领域适应中广泛使用的类别均值最大差异（CMMD）相关联，这是一种不相似度度量。我们的研究显示，最小化对比损失可以降低CMMD，同时提高类间可分性，为在领域适应背景下使用对比学习奠定了理论基础。由于医学影像中领域适应的重要性，我们将实验集中在乳腺X光片。在三个乳腺X光数据集中——合成补丁、临床（真实）补丁和临床（真实）图像——的广泛实验结果表明，最小化监督对比损失可以提高领域适应性、类间可分性和分类性能。 

---
# Contextually Entangled Gradient Mapping for Optimized LLM Comprehension 

**Title (ZH)**: 基于上下文纠缠的梯度映射优化大型语言模型理解 

**Authors**: Colin Sisate, Alistair Goldfinch, Vincent Waterstone, Sebastian Kingsley, Mariana Blackthorn  

**Link**: [PDF](https://arxiv.org/pdf/2502.00048)  

**Abstract**: Contextually Entangled Gradient Mapping (CEGM) introduces a new approach to gradient optimization, redefining the relationship between contextual embeddings and gradient updates to enhance semantic coherence and reasoning capabilities in neural architectures. By treating gradients as dynamic carriers of contextual dependencies rather than isolated numerical entities, the proposed methodology bridges critical gaps in existing optimization strategies. The integration of entangled gradient dynamics into a loss regularization framework demonstrated significant improvements in tasks involving long-form reasoning, contextual retention, and adaptability to unseen domains. Experimental evaluations showed that the CEGM-enhanced model consistently outperformed baseline approaches, achieving higher accuracy in token-level predictions and greater resilience to noisy inputs. Practical implementations involved modifications to training pipelines, introducing entanglement layers and dynamic coefficient adjustments that seamlessly align with existing architectures. Results further highlighted reductions in semantic drift during sequential transformations and improvements in embedding coherence across paraphrased sentences, showing the robustness and versatility of the proposed methodology. The findings demonstrate the broader implications of gradient entanglement for both theoretical advancements and practical applications in optimization strategies. 

**Abstract (ZH)**: 情境纠缠梯度映射（Contextually Entangled Gradient Mapping, CEGM）提出了一种新的梯度优化方法，重新定义了情境嵌入与梯度更新之间的关系，以增强神经架构中的语义连贯性和推理能力。通过将梯度视为情境依赖性的动态载体，而非孤立的数值实体，所提出的方法填补了现有优化策略中的关键空白。将纠缠的梯度动态整合到损失正则化框架中，在涉及长序列推理、情境保留和对未见领域的适应性任务中显示出显著的进步。实验评估表明，CEGM增强的模型始终优于基线方法，在标记级别预测中获得更高的准确性，并对噪声输入具有更强的鲁棒性。实际实现涉及对训练管道的修改，引入纠缠层和动态系数调整，这些调整能无缝地与现有架构兼容。结果进一步强调，序列转换期间语义漂移的减少以及跨同义句嵌入的一致性提升，充分展示了所提出方法的稳健性和灵活性。研究结果表明，梯度纠缠对于优化策略的理论进展和实际应用具有更广泛的影响。 

---
# Restless Multi-armed Bandits under Frequency and Window Constraints for Public Service Inspections 

**Title (ZH)**: 频率和窗口约束下的活跃多臂bandit问题在公共服务检查中的应用 

**Authors**: Yi Mao, Andrew Perrault  

**Link**: [PDF](https://arxiv.org/pdf/2502.00045)  

**Abstract**: Municipal inspections are an important part of maintaining the quality of goods and services. In this paper, we approach the problem of intelligently scheduling service inspections to maximize their impact, using the case of food establishment inspections in Chicago as a case study. The Chicago Department of Public Health (CDPH) inspects thousands of establishments each year, with a substantial fail rate (over 3,000 failed inspection reports in 2023). To balance the objectives of ensuring adherence to guidelines, minimizing disruption to establishments, and minimizing inspection costs, CDPH assigns each establishment an inspection window every year and guarantees that they will be inspected exactly once during that window. These constraints create a challenge for a restless multi-armed bandit (RMAB) approach, for which there are no existing methods. We develop an extension to Whittle index-based systems for RMABs that can guarantee action window constraints and frequencies, and furthermore can be leveraged to optimize action window assignments themselves. Briefly, we combine MDP reformulation and integer programming-based lookahead to maximize the impact of inspections subject to constraints. A neural network-based supervised learning model is developed to model state transitions of real Chicago establishments using public CDPH inspection records, which demonstrates 10\% AUC improvements compared with directly predicting establishments' failures. Our experiments not only show up to 24\% (in simulation) or 33\% (on real data) reward improvements resulting from our approach but also give insight into the impact of scheduling constraints. 

**Abstract (ZH)**: 市政检查是维护商品和服务质量的重要组成部分。本文采用智能化调度服务检查的方法，最大化检查的效果，以芝加哥食品场所检查为例进行了研究。芝加哥公共卫生部门（CDPH）每年检查数千个场所，失败率较高（2023年有超过3000份失败的检查报告）。为满足确保遵守指南、最小化对场所的干扰以及最小化检查成本等目标，CDPH 每年为每个场所分配一个检查窗口，并保证在该窗口内进行一次检查。这些约束条件为解决多臂赌博机问题中的时间紧张版（RMAB）方法带来了挑战，目前尚无现成的方法。我们开发了一种 Whittle 索引为基础的系统扩展，该系统可以确保行动窗口的约束条件和频率，同时还可以用于优化行动窗口的分配。简而言之，我们通过结合MDP（马尔可夫决策过程）重述和基于整数规划的前瞻，以满足约束条件最大化检查效果。基于神经网络的监督学习模型用于根据公共的 CDPH 检查记录模拟芝加哥实际场所的状态转移，并展示了相比直接预测场所的失败情况，有10%的AUC改进。我们的实验不仅展示了我们的方法在模拟中最高可达24%或在实际数据中最高可达33%的奖励改进，还提供了关于检查约束影响的洞察。 

---
# A scalable adaptive deep Koopman predictive controller for real-time optimization of mixed traffic flow 

**Title (ZH)**: 一种适用于混合车流实时优化的可扩展自适应深度库曼预测控制器 

**Authors**: Hao Lyu, Yanyong Guo, Pan Liu, Nan Zheng, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00043)  

**Abstract**: The use of connected automated vehicle (CAV) is advocated to mitigate traffic oscillations in mixed traffic flow consisting of CAVs and human driven vehicles (HDVs). This study proposes an adaptive deep Koopman predictive control framework (AdapKoopPC) for regulating mixed traffic flow. Firstly, a Koopman theory-based adaptive trajectory prediction deep network (AdapKoopnet) is designed for modeling HDVs car-following behavior. AdapKoopnet enables the representation of HDVs behavior by a linear model in a high-dimensional space. Secondly, the model predictive control is employed to smooth the mixed traffic flow, where the combination of the linear dynamic model of CAVs and linear prediction blocks from AdapKoopnet is embedded as the predictive model into the AdapKoopPC. Finally, the predictive performance of the prosed AdapKoopnet is verified using the HighD naturalistic driving dataset. Furthermore, the control performance of AdapKoopPC is validated by the numerical simulations. Results demonstrate that the AdapKoopnet provides more accuracy HDVs predicted trajectories than the baseline nonlinear models. Moreover, the proposed AdapKoopPC exhibits more effective control performance with less computation cost compared with baselines in mitigating traffic oscillations, especially at the low CAVs penetration rates. The code of proposed AdapKoopPC is open source. 

**Abstract (ZH)**: 采用连驾驶汽车（Connected Automated Vehicle, CAV）有助于缓解由CAVs和人工驾驶车辆（Human Driver Vehicle, HDVs）组成的混合交通流中的交通振荡。本文提出了一种自适应深度Koopman预测控制框架（AdapKoopPC），用于调节混合交通流。首先，基于Koopman理论设计了一种自适应轨迹预测深度网络（AdapKoopnet），用于建模HDVs的跟随行为。AdapKoopnet能够将HDVs的行为表示为高维空间中的线性模型。其次，在控制策略中使用模型预测控制来平滑混合交通流，其中CAVs的线性动力学模型和从AdapKoopnet输出的线性预测模块结合形成预测模型嵌入到AdapKoopPC中。最后，使用HighD自然驾驶数据集验证了所提出的AdapKoopnet的预测性能，并通过数值模拟验证了AdapKoopPC的控制性能。结果表明，AdapKoopnet相比基线的非线性模型提供了更准确的HDVs预测轨迹。此外，与基线相比，所提出的AdapKoopPC在缓解交通振荡方面表现出更有效的控制性能，尤其是在较少的CAVs渗透率下，且计算成本更低。提出的AdapKoopPC的代码是开源的。 

---
# Multi-Objective Reinforcement Learning for Power Grid Topology Control 

**Title (ZH)**: 多目标强化学习在电力网络拓扑控制中的应用 

**Authors**: Thomas Lautenbacher, Ali Rajaei, Davide Barbieri, Jan Viebahn, Jochen L. Cremer  

**Link**: [PDF](https://arxiv.org/pdf/2502.00040)  

**Abstract**: Transmission grid congestion increases as the electrification of various sectors requires transmitting more power. Topology control, through substation reconfiguration, can reduce congestion but its potential remains under-exploited in operations. A challenge is modeling the topology control problem to align well with the objectives and constraints of operators. Addressing this challenge, this paper investigates the application of multi-objective reinforcement learning (MORL) to integrate multiple conflicting objectives for power grid topology control. We develop a MORL approach using deep optimistic linear support (DOL) and multi-objective proximal policy optimization (MOPPO) to generate a set of Pareto-optimal policies that balance objectives such as minimizing line loading, topological deviation, and switching frequency. Initial case studies show that the MORL approach can provide valuable insights into objective trade-offs and improve Pareto front approximation compared to a random search baseline. The generated multi-objective RL policies are 30% more successful in preventing grid failure under contingencies and 20% more effective when training budget is reduced - compared to the common single objective RL policy. 

**Abstract (ZH)**: 随着各个领域的电气化需求增加，需要传输更多的电力，从而导致输电网拥堵加剧。通过变电站重新配置来进行拓扑控制可以减少拥堵，但在实际操作中其潜力尚未得到充分利用。一个挑战是如何建立拓扑控制问题的模型，以更好地与运营商的目标和约束条件相一致。为了解决这一挑战，本文探讨了多目标强化学习（MORL）在电力系统拓扑控制中的应用，以整合多个相互冲突的目标。我们开发了使用深度乐观线性支持（DOL）和多目标近端策略优化（MOPPO）的MORL方法，以生成一组帕累托最优策略，这些策略可以平衡如减小线路负载、拓扑偏差和切换频率等目标。初步案例研究显示，MORL方法可以提供有价值的目标权衡见解，并在帕累托前沿逼近上优于随机搜索基线。生成的多目标RL策略在其能够防止电网故障的能力上优于传统的单一目标RL策略30%，在训练预算减少时更为有效，比常见的单一目标RL策略有效20%。 

---
# Efficient Client Selection in Federated Learning 

**Title (ZH)**: 联邦学习中高效客户端选择方法 

**Authors**: William Marfo, Deepak K. Tosh, Shirley V. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2502.00036)  

**Abstract**: Federated Learning (FL) enables decentralized machine learning while preserving data privacy. This paper proposes a novel client selection framework that integrates differential privacy and fault tolerance. The adaptive client selection adjusts the number of clients based on performance and system constraints, with noise added to protect privacy. Evaluated on the UNSW-NB15 and ROAD datasets for network anomaly detection, the method improves accuracy by 7% and reduces training time by 25% compared to baselines. Fault tolerance enhances robustness with minimal performance trade-offs. 

**Abstract (ZH)**: 联邦学习（FL）能够实现分散化机器学习并保护数据隐私。本文提出了一种新颖的客户端选择框架，该框架将差分隐私和容错机制整合进来。自适应客户端选择根据性能和系统约束动态调整客户端的数量，并通过添加噪声来保护隐私。该方法在对UNSW-NB15和ROAD数据集进行网络异常检测评估中，与基准方法相比，准确率提高了7%，训练时间减少了25%。容错机制增加了系统的鲁棒性，同时最大限度地减少了性能损失。 

---
# Querying Databases with Function Calling 

**Title (ZH)**: 使用函数调用来查询数据库 

**Authors**: Connor Shorten, Charles Pierse, Thomas Benjamin Smith, Karel D'Oosterlinck, Tuana Celik, Erika Cardenas, Leonie Monigatti, Mohd Shukri Hasan, Edward Schmuhl, Daniel Williams, Aravind Kesiraju, Bob van Luijt  

**Link**: [PDF](https://arxiv.org/pdf/2502.00032)  

**Abstract**: The capabilities of Large Language Models (LLMs) are rapidly accelerating largely thanks to their integration with external tools. Querying databases is among the most effective of these integrations, enabling LLMs to access private or continually updating data. While Function Calling is the most common method for interfacing external tools to LLMs, its application to database querying as a tool has been underexplored. We propose a tool definition for database querying that unifies accessing data with search queries, filters, or a combination both, as well as transforming results with aggregation and groupby operators. To evaluate its effectiveness, we conduct a study with 8 LLMs spanning 5 model families. We present a novel pipeline adapting the Gorilla LLM framework to create synthetic database schemas and queries. We primarily evaluate the models with the Exact Match of predicted and ground truth query APIs. Among the models tested, Claude 3.5 Sonnet achieves the highest performance with an Exact Match score of 74.3%, followed by GPT-4o mini at 73.7%, and GPT-4o at 71.8%. We further breakdown these results per API component utilized and across synthetic use cases. We find that LLMs are highly effective at utilizing operators on boolean properties, but struggle with text property filters. Across use cases we find robust results with the higher performing models such as GPT-4o, but significant performance variance across use cases from lower performing models. We additionally conduct ablation studies exploring the impact of parallel tool calling, adding a rationale as an argument of the tool call, using a separate tool per database collection, and tool calling with structured outputs. Our findings demonstrate the effectiveness of enabling LLMs to query databases with Function Calling. We have open-sourced our experimental code and results at this http URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的能力迅速提升，这主要得益于其与外部工具的整合。查询数据库是这种整合中最有效的手段之一，使LLMs能够访问私有数据或实时更新的数据。虽然函数调用是最常用的将外部工具与LLMs进行接口的方法，但将其应用于数据库查询工具的研究尚不足。我们提出了一种数据库查询工具定义，该定义统一了通过搜索查询、筛选器或二者的组合访问数据的方式，以及使用聚合和分组操作符对结果进行转换。为了评估其有效性，我们在5个模型家族中的8个LLM上进行了研究。我们提出了一种新的流水线，将Gorilla LLM框架适应于创建合成数据库模式和查询。我们主要通过预测和地面真相查询API的精确匹配来评估这些模型。在测试的模型中，Claude 3.5 Sonnet表现出最高的性能，精确匹配得分为74.3%，其次是GPT-4o mini的73.7%，以及GPT-4o的71.8%。我们进一步根据使用的API组件和合成应用场景对这些结果进行了细分。我们发现，LLMs在利用布尔属性上的操作方面表现出色，但在文本属性的筛选器方面存在困难。在应用场景中，我们发现高性能模型，如GPT-4o，具有稳健的结果，但低性能模型在不同应用场景中表现出显著的性能差异。此外，我们进行了消融研究，探讨了平行工具调用、在工具调用中添加理由、为每个数据库集合使用单独的工具以及使用结构化输出的工具调用的影响。我们的研究结果表明，允许LLMs通过函数调用来查询数据库是有效的。我们已将实验代码和结果开源，并可在该网址访问：[此处填写网址]。 

---
# AlphaSharpe: LLM-Driven Discovery of Robust Risk-Adjusted Metrics 

**Title (ZH)**: AlphaSharpe: 基于LLM的稳健风险调整指标发现 

**Authors**: Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.00029)  

**Abstract**: Financial metrics like the Sharpe ratio are pivotal in evaluating investment performance by balancing risk and return. However, traditional metrics often struggle with robustness and generalization, particularly in dynamic and volatile market conditions. This paper introduces AlphaSharpe, a novel framework leveraging large language models (LLMs) to iteratively evolve and optimize financial metrics. AlphaSharpe generates enhanced risk-return metrics that outperform traditional approaches in robustness and correlation with future performance metrics by employing iterative crossover, mutation, and evaluation. Key contributions of this work include: (1) an innovative use of LLMs for generating and refining financial metrics inspired by domain-specific knowledge, (2) a scoring mechanism to ensure the evolved metrics generalize effectively to unseen data, and (3) an empirical demonstration of 3x predictive power for future risk-return forecasting. Experimental results on a real-world dataset highlight the superiority of AlphaSharpe metrics, making them highly relevant for portfolio managers and financial decision-makers. This framework not only addresses the limitations of existing metrics but also showcases the potential of LLMs in advancing financial analytics, paving the way for informed and robust investment strategies. 

**Abstract (ZH)**: 金融指标如夏普比率在评估投资表现方面至关重要，它们能够平衡风险与回报。然而，传统的指标在应对动态和波动的市场条件时往往缺乏稳健性和普适性。本文介绍了一种名为AlphaSharpe的创新框架，该框架利用大型语言模型（LLMs）迭代地生成和优化金融指标。AlphaSharpe通过运用迭代交叉、变异和评估机制，生成了优于传统方法的增强风险-回报指标，并在稳健性和对未来表现指标的相关性方面表现更优。本文的主要贡献包括：（1）利用特定领域知识为金融指标生成和优化提供创新使用LLMs的方法；（2）提出了一种评分机制，以确保优化后的指标能够有效泛化到未见过的数据；（3）提供了实验证据，证明AlphaSharpe指标在风险-回报预测中具有3倍的预测能力。实际数据集上的实验结果表明，AlphaSharpe指标具有卓越的优越性，对于投资组合经理和财务决策者具有高度相关性。该框架不仅解决了现有指标的局限性，还展示了LLMs在推动金融分析方面的发展潜力，为制定知情且稳健的投资策略铺平了道路。 

---
# Analysis of a Memcapacitor-Based for Neural Network Accelerator Framework 

**Title (ZH)**: 基于忆阻器的神经网络加速器框架分析 

**Authors**: Ankur Singh, Dowon Kim, Byung-Geun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.00027)  

**Abstract**: Data-intensive computing tasks, such as training neural networks, are crucial for artificial intelligence applications but often come with high energy demands. One promising solution is to develop specialized hardware that directly maps neural networks, utilizing arrays of memristive devices to perform parallel multiply-accumulate operations. In our research, we introduce a novel CMOS-based memcapacitor circuit that is validated using the cadence tool. Additionally, we developed the device in Python to facilitate the design of a memcapacitive-based accelerator. Our proposed framework employs a crossbar array of memcapacitor devices to train a neural network capable of digit classification and CIFAR dataset recognition. We tested the non-ideal characteristics of the constructed memcapacitor-based neural network. The system achieved an impressive 98.4% training accuracy in digit recognition and 94.4% training accuracy in CIFAR recognition, highlighting its effectiveness. This study demonstrates the potential of memcapacitor-based neural network systems in handling classification tasks and sets the stage for further advancements in neuromorphic computing. 

**Abstract (ZH)**: 数据密集型计算任务，如训练神经网络，对于人工智能应用至关重要，但往往伴随着高能耗。一种有潜力的解决方案是开发专门的硬件，直接映射神经网络，利用 memristive 设备阵列执行并行乘积-累加操作。在我们的研究中，我们引入了一种新型基于 CMOS 的 memcapacitor 电路，并使用 cadence 工具进行了验证。此外，我们还在 Python 中开发了该器件，以促进基于 memcapacitive 的加速器的设计。我们提出的框架采用 memcapacitor 设备的交叉电位器阵列来训练一个能够进行数字分类和 CIFAR 数据集识别的神经网络。我们测试了所构建的 memcapacitor 基础神经网络的非理想特性。该系统在数字识别中的训练准确率达到 98.4%，在 CIFAR 识别中的训练准确率达到 94.4%，突显了其有效性。本研究表明，基于 memcapacitor 的神经网络系统在处理分类任务方面具有潜力，并为类脑计算的进一步发展奠定了基础。 

---
# Pushing the Limits of BFP on Narrow Precision LLM Inference 

**Title (ZH)**: 将BFP在窄位宽LLM推断中的潜力推向极限 

**Authors**: Hui Wang, Yuan Cheng, Xiaomeng Han, Zhengpeng Zhao, Dawei Yang, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00026)  

**Abstract**: The substantial computational and memory demands of Large Language Models (LLMs) hinder their deployment. Block Floating Point (BFP) has proven effective in accelerating linear operations, a cornerstone of LLM workloads. However, as sequence lengths grow, nonlinear operations, such as Attention, increasingly become performance bottlenecks due to their quadratic computational complexity. These nonlinear operations are predominantly executed using inefficient floating-point formats, which renders the system challenging to optimize software efficiency and hardware overhead. In this paper, we delve into the limitations and potential of applying BFP to nonlinear operations. Given our findings, we introduce a hardware-software co-design framework (DB-Attn), including: (i) DBFP, an advanced BFP version, overcomes nonlinear operation challenges with a pivot-focus strategy for diverse data and an adaptive grouping strategy for flexible exponent sharing. (ii) DH-LUT, a novel lookup table algorithm dedicated to accelerating nonlinear operations with DBFP format. (iii) An RTL-level DBFP-based engine is implemented to support DB-Attn, applicable to FPGA and ASIC. Results show that DB-Attn provides significant performance improvements with negligible accuracy loss, achieving 74% GPU speedup on Softmax of LLaMA and 10x low overhead performance improvement over SOTA designs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的巨大计算和内存需求阻碍了它们的部署。块浮动点数（Block Floating Point，BFP）已被证明能够加速线性运算，而线性运算对LLMs的工作负载至关重要。然而，随着序列长度的增长，非线性运算（如注意机制）逐渐成为性能瓶颈，这归因于它们的二次复杂度计算特性。这些非线性运算通常使用效率较低的浮点格式执行，这使得系统在优化软件效率和硬件开销方面面临挑战。本文探讨了BFP在非线性运算中的局限性和潜在应用。基于我们的研究，我们提出了一个硬件与软件协同设计框架（DB-Attn），包括：（i）DBFP，这是一种高级的BFP版本，通过针对多样数据的枢轴焦点策略和灵活的指数共享策略来解决非线性运算挑战。（ii）DH-LUT，一种专门为加速使用DBFP格式的非线性运算而设计的查找表算法。（iii）基于DBFP的RTL级引擎实现，适用于FPGA和ASIC。结果显示，DB-Attn在不牺牲精度的情况下提供了显著的性能提升，LLaMA的Softmax加速达到了74%的GPU速度提升，并且与最先进的设计相比，具有10倍的低开销性能改进。 

---
# Leveraging Large Language Models to Enhance Machine Learning Interpretability and Predictive Performance: A Case Study on Emergency Department Returns for Mental Health Patients 

**Title (ZH)**: 利用大型语言模型提升机器学习可解释性和预测性能：一项针对精神健康患者急诊复诊的案例研究 

**Authors**: Abdulaziz Ahmed, Mohammad Saleem, Mohammed Alzeen, Badari Birur, Rachel E Fargason, Bradley G Burk, Hannah Rose Harkins, Ahmed Alhassan, Mohammed Ali Al-Garadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.00025)  

**Abstract**: Objective: To evaluate whether integrating large language models (LLMs) with traditional machine learning approaches improves both the predictive accuracy and clinical interpretability of ED mental health returns risk models. Methods: This retrospective cohort study analyzed 42,464 ED visits for 27,904 unique mental health patients at an Academic Medical Center in the deep South of the United States between January 2018 and December 2022. Main Outcomes and Measures: Two primary outcomes were evaluated: (1) 30 days ED return prediction accuracy and (2) model interpretability through a novel retrieval-augmented generation (RAG) framework integrating SHAP (SHapley Additive exPlanations) values with contextual clinical knowledge. Results: The proposed machine learning interpretability framework, leveraging LLM, achieved 99% accuracy in translating complex model predictions into clinically relevant explanations. Integration of LLM-extracted features enhanced predictive performance, improving the XGBoost model area under the curve (AUC) from 0.73 to 0.76. The LLM-based feature extraction using 10-shot learning significantly outperformed traditional approaches, achieving an accuracy of 0.882 and an F1 score of 0.86 for chief complaint classification (compared to conventional methods with an accuracy range of 0.59 to 0.63) and demonstrating accuracy values ranging from 0.65 to 0.93 across multiple SDoH categories, underscoring its robust performance in extracting features from clinical notes. Conclusions and Relevance: Integrating LLMs with traditional machine learning models yielded modest but consistent improvements in ED return prediction accuracy while substantially enhancing model interpretability through automated, clinically relevant explanations. This approach offers a framework for translating complex predictive analytics into actionable clinical insights. 

**Abstract (ZH)**: 目的：评估将大型语言模型（LLMs）与传统机器学习方法结合使用是否能够同时提高急诊科心理健康返回风险模型的预测准确性和临床可解释性。

方法：本回顾性队列研究分析了2018年1月至2022年12月在美国南部一所学术医疗中心的27,904名独特心理健康患者在急诊科的42,464次就诊记录。主要结果和指标：主要评估了两项结果：（1）30天急诊返回预测准确性；（2）通过结合利用LIME（本地可解释的模型解释）值与上下文临床知识的检索增强生成（RAG）框架来评估模型的可解释性。结果：所提出的基于LLMs的机器学习可解释性框架实现了99%的复杂模型预测向临床相关解释的翻译准确性。将LLM提取的特征结合使用提高了预测性能，使XGBoost模型的曲线下面积（AUC）从0.73提高到0.76。基于10-shot学习的LLM特征提取方法在主要症状分类方面的准确率达到0.882，F1分数达到0.86，显著优于传统方法（准确率范围为0.59至0.63），并在多个社会人口经济因素（SDoH）类别中达到了0.65至0.93的准确性，证明了其在从临床记录中提取特征方面的稳健性表现。结论与意义：将LLMs与传统机器学习模型结合使用在急诊返回预测准确性方面产生了适度但一致的提升，同时通过自动化、临床相关的解释大幅增强了模型的可解释性。这种方法提供了一个将复杂的预测分析转化为可操作的临床见解的框架。 

---
# Musical Agent Systems: MACAT and MACataRT 

**Title (ZH)**: 音乐智能体系统：MACAT与MACataRT 

**Authors**: Keon Ju M. Lee, Philippe Pasquier  

**Link**: [PDF](https://arxiv.org/pdf/2502.00023)  

**Abstract**: Our research explores the development and application of musical agents, human-in-the-loop generative AI systems designed to support music performance and improvisation within co-creative spaces. We introduce MACAT and MACataRT, two distinct musical agent systems crafted to enhance interactive music-making between human musicians and AI. MACAT is optimized for agent-led performance, employing real-time synthesis and self-listening to shape its output autonomously, while MACataRT provides a flexible environment for collaborative improvisation through audio mosaicing and sequence-based learning. Both systems emphasize training on personalized, small datasets, fostering ethical and transparent AI engagement that respects artistic integrity. This research highlights how interactive, artist-centred generative AI can expand creative possibilities, empowering musicians to explore new forms of artistic expression in real-time, performance-driven and music improvisation contexts. 

**Abstract (ZH)**: 我们的研究探索了音乐代理的发展与应用，这是一种由人类参与的生成型人工智能系统，旨在支持音乐表演和即兴创作。我们介绍了两种不同的音乐代理系统：MACAT和MACataRT，以提高人类音乐家和AI之间的交互式音乐创作。MACAT优化了以代理为中心的表演，利用实时合成和自我倾听来自主塑造其输出，而MACataRT则提供了一个通过音频镶嵌和基于序列的学习进行协作即兴创作的灵活环境。两种系统都强调基于个性化的小数据集进行训练，从而培养一种公平透明的AI互动方式，尊重艺术完整性。本研究突显了交互式、以艺术家为中心的生成型人工智能如何扩展创作可能性，使音乐家能够实时探索新的艺术表达形式，在表演驱动和即兴音乐创作的情境中赋予他们更大的创造力。 

---
# Ethical Concerns of Generative AI and Mitigation Strategies: A Systematic Mapping Study 

**Title (ZH)**: 生成式人工智能的伦理关切与缓解策略：一项系统映射研究 

**Authors**: Yutan Huang, Chetan Arora, Wen Cheng Houng, Tanjila Kanij, Anuradha Madulgalla, John Grundy  

**Link**: [PDF](https://arxiv.org/pdf/2502.00015)  

**Abstract**: [Context] Generative AI technologies, particularly Large Language Models (LLMs), have transformed numerous domains by enhancing convenience and efficiency in information retrieval, content generation, and decision-making processes. However, deploying LLMs also presents diverse ethical challenges, and their mitigation strategies remain complex and domain-dependent. [Objective] This paper aims to identify and categorize the key ethical concerns associated with using LLMs, examine existing mitigation strategies, and assess the outstanding challenges in implementing these strategies across various domains. [Method] We conducted a systematic mapping study, reviewing 39 studies that discuss ethical concerns and mitigation strategies related to LLMs. We analyzed these ethical concerns using five ethical dimensions that we extracted based on various existing guidelines, frameworks, and an analysis of the mitigation strategies and implementation challenges. [Results] Our findings reveal that ethical concerns in LLMs are multi-dimensional and context-dependent. While proposed mitigation strategies address some of these concerns, significant challenges still remain. [Conclusion] Our results highlight that ethical issues often hinder the practical implementation of the mitigation strategies, particularly in high-stake areas like healthcare and public governance; existing frameworks often lack adaptability, failing to accommodate evolving societal expectations and diverse contexts. 

**Abstract (ZH)**: [背景] 生成型人工智能技术，特别是大型语言模型（LLMs），通过在信息检索、内容生成和决策过程中的便利性和效率提升，彻底改变了众多领域。然而，部署LLMs也带来了多样化的伦理挑战，而这些挑战的缓解策略依然复杂且具有领域依赖性。[研究目的] 本文旨在识别并分类与使用LLMs相关的关键伦理问题，评估现有缓解策略的有效性，并考察在不同领域实施这些策略所面临的突出挑战。[方法] 我们进行了一项系统映射研究，审查了39篇讨论LLMs相关伦理问题和缓解策略的研究论文。我们使用了从各种现有指南、框架以及对缓解策略和实施挑战分析中提取出的五个伦理维度来分析这些伦理问题。[结果] 研究发现表明，LLMs的伦理问题具有多维度和情境依赖性。虽然提出的缓解策略能够解决部分问题，但仍存在显著挑战。[结论] 研究结果指出，伦理问题常常阻碍缓解策略的实际实施，尤其是在医疗保健和公共治理等高风险领域；现有的框架往往缺乏适应性，未能满足不断变化的社会期望和多样化的情境需求。 

---
# TOAST Framework: A Multidimensional Approach to Ethical and Sustainable AI Integration in Organizations 

**Title (ZH)**: TOAST框架：组织中伦理与可持续人工智能集成的多维度方法 

**Authors**: Dian Tjondronegoro  

**Link**: [PDF](https://arxiv.org/pdf/2502.00011)  

**Abstract**: Artificial Intelligence (AI) has emerged as a transformative technology with the potential to revolutionize various sectors, from healthcare to finance, education, and beyond. However, successfully implementing AI systems remains a complex challenge, requiring a comprehensive and methodologically sound framework. This paper contributes to this challenge by introducing the Trustworthy, Optimized, Adaptable, and Socio-Technologically harmonious (TOAST) framework. It draws on insights from various disciplines to align technical strategy with ethical values, societal responsibilities, and innovation aspirations. The TOAST framework is a novel approach designed to guide the implementation of AI systems, focusing on reliability, accountability, technical advancement, adaptability, and socio-technical harmony. By grounding the TOAST framework in healthcare case studies, this paper provides a robust evaluation of its practicality and theoretical soundness in addressing operational, ethical, and regulatory challenges in high-stakes environments, demonstrating how adaptable AI systems can enhance institutional efficiency, mitigate risks like bias and data privacy, and offer a replicable model for other sectors requiring ethically aligned and efficient AI integration. 

**Abstract (ZH)**: 人工智能（AI）作为一种 transformative 技术，已经展现出在各领域（如医疗、金融、教育等）进行革命的潜力。然而，成功实施AI系统仍然是一项复杂挑战，需要一个全面且方法论严谨的框架。本文通过提出值得信赖、优化、适应性强且社会和技术和谐共存（TOAST）框架来应对这一挑战。该框架借鉴了各学科的见解，旨在将技术策略与伦理价值观、社会责任和创新愿景相结合。TOAST框架是一种新颖的方法，旨在指导AI系统的实施，重点关注可靠性和可问责性、技术进步、适应性和社会技术和谐。

本文通过基于医疗案例研究来确立TOAST框架，提供其在高风险领域中操作可行性、理论严谨性的稳健评估，说明如何使可适应性强的AI系统提升机构效率，减轻偏见和数据隐私等风险，并为其他需要道德对齐和高效AI集成的领域提供可复制的模型。 

---
# A Study about Distribution and Acceptance of Conversational Agents for Mental Health in Germany: Keep the Human in the Loop? 

**Title (ZH)**: 关于对话代理在德国心理健康领域中的分布与接受度的研究：保持人类在环中？ 

**Authors**: Christina Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2502.00005)  

**Abstract**: Good mental health enables individuals to cope with the normal stresses of life. In Germany, approximately one-quarter of the adult population is affected by mental illnesses. Teletherapy and digital health applications are available to bridge gaps in care and relieve healthcare professionals. The acceptance of these tools is a strongly influencing factor for their effectiveness, which also needs to be evaluated for AI-based conversational agents (CAs) (e. g. ChatGPT, Siri) to assess the risks and potential for integration into therapeutic practice. This study investigates the perspectives of both the general population and healthcare professionals with the following questions: 1. How frequently are CAs used for mental health? 2. How high is the acceptance of CAs in the field of mental health? 3. To what extent is the use of CAs in counselling, diagnosis, and treatment acceptable? To address these questions, two quantitative online surveys were conducted with 444 participants from the general population and 351 healthcare professionals. Statistical analyses show that 27 % of the surveyed population already confide their concerns to CAs. Not only experience with this technology but also experience with telemedicine shows a higher acceptance among both groups for using CAs for mental health. Additionally, participants from the general population were more likely to support CAs as companions controlled by healthcare professionals rather than as additional experts for the professionals. CAs have the potential to support mental health, particularly in counselling. Future research should examine the influence of different communication media and further possibilities of augmented intelligence. With the right balance between technology and human care, integration into patient-professional interaction can be achieved. 

**Abstract (ZH)**: 良好的心理健康能使个体更好地应对生活中的正常压力。在德国，大约四分之一的成年人受到精神疾病的困扰。远程治疗和数字健康应用程序可用于解决医疗护理缺口，并减轻医务人员的压力。这些工具的接受度是影响其有效性的关键因素，特别是在基于人工智能的对话代理（例如ChatGPT、Siri）用于评估风险和探讨将其整合到治疗实践中时。本研究通过探讨普通人群和医务人员的看法，旨在回答以下问题：1. 人们对对话代理（CAs）用于心理健康的情况有多频繁？2. 对话代理在心理健康领域中的接受度有多高？3. 在咨询、诊断和治疗中使用对话代理的程度是否可接受？为了回答这些问题，我们对普通人群和医务人员分别进行了两份定量在线调查，分别有444名普通参与者和351名医务人员。统计分析显示，受访人群中已有27%的人向对话代理倾诉自己的问题。不仅在该技术的经验上，而且在对远程医疗服务的经验上，这两组人群中使用对话代理进行心理健康护理的接受度更高。此外，普通人群参与者更倾向于支持由医务人员控制的对话代理作为陪伴，而不是作为医务人员的额外专家。对话代理有可能在心理健康咨询中提供支持。未来的研究应探讨不同通信媒介的影响以及增强智能的进一步可能性。通过恰当平衡技术与人性化护理，对话代理可以整合到患者与医务人员的互动中。 

---
# Defending Compute Thresholds Against Legal Loopholes 

**Title (ZH)**: 防范法律漏洞对计算阈值的攻击 

**Authors**: Matteo Pistillo, Pablo Villalobos  

**Link**: [PDF](https://arxiv.org/pdf/2502.00003)  

**Abstract**: Existing legal frameworks on AI rely on training compute thresholds as a proxy to identify potentially-dangerous AI models and trigger increased regulatory attention. In the United States, Section 4.2(a) of Executive Order 14110 instructs the Secretary of Commerce to require extensive reporting from developers of AI models above a certain training compute threshold. In the European Union, Article 51 of the AI Act establishes a presumption that AI models above a certain compute threshold have high impact capabilities and hence pose systemic risk, thus subjecting their developers to several obligations including capability evaluations, reporting, and incident monitoring. In this paper, we examine some enhancement techniques that are capable of decreasing training compute usage while preserving, or even increasing, model capabilities. Since training compute thresholds rely on training compute as a metric and trigger for increased regulatory attention, these capability-enhancing and compute-saving techniques could constitute a legal loophole to existing training compute thresholds. In particular, we concentrate on four illustrative techniques (fine-tuning, model reuse, model expansion, and above compute-optimal inference compute) with the goal of furthering the conversation about their implications on training compute thresholds as a legal mechanism and advancing policy recommendations that could address the relevant legal loopholes. 

**Abstract (ZH)**: 现有的AI法律框架依赖于训练计算阈值作为代理，以识别潜在危险的AI模型并触发更严格的监管关注。在美利坚合众国，行政令14110第4.2(a)节指示商务部长要求达到一定训练计算阈值的AI模型开发者进行详尽报告。在欧盟，AI法案第51条假定达到一定计算阈值的AI模型具有高影响能力，从而构成系统性风险，因此对其开发者施加了包括能力评估、报告和事件监控在内的多项义务。本文旨在探讨一些能够降低训练计算使用率但仍能保持或提高模型能力的增强技术。由于训练计算阈值依赖于训练计算作为衡量标准和触发更严格监管的机制，这些能力增强和计算节约技术可能构成现有训练计算阈值的法律漏洞。我们特别集中讨论四种示例技术（微调、模型复用、模型扩展和计算优化推理计算），旨在进一步探讨这些技术对训练计算阈值作为法律机制的影响，并推动可以解决相关法律漏洞的政策建议。 

---
