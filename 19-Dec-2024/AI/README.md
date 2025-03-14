# Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics with Large Language Models 

**Title (ZH)**: 基于大型语言模型的数据分析中多步洞察综合的高级推理与转换引擎 

**Authors**: Atin Sakkeer Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2412.14146)  

**Abstract**: This paper presents the Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics (ARTEMIS-DA), a novel framework designed to augment Large Language Models (LLMs) for solving complex, multi-step data analytics tasks. ARTEMIS-DA integrates three core components: the Planner, which dissects complex user queries into structured, sequential instructions encompassing data preprocessing, transformation, predictive modeling, and visualization; the Coder, which dynamically generates and executes Python code to implement these instructions; and the Grapher, which interprets generated visualizations to derive actionable insights. By orchestrating the collaboration between these components, ARTEMIS-DA effectively manages sophisticated analytical workflows involving advanced reasoning, multi-step transformations, and synthesis across diverse data modalities. The framework achieves state-of-the-art (SOTA) performance on benchmarks such as WikiTableQuestions and TabFact, demonstrating its ability to tackle intricate analytical tasks with precision and adaptability. By combining the reasoning capabilities of LLMs with automated code generation and execution and visual analysis, ARTEMIS-DA offers a robust, scalable solution for multi-step insight synthesis, addressing a wide range of challenges in data analytics. 

**Abstract (ZH)**: 本文介绍了一种名为Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics（ARTEMIS-DA）的新型框架，旨在增强大型语言模型（LLMs），以解决复杂的多步数据分析任务。ARTEMIS-DA 汇集了三个核心组件：规划器（Planner），它将复杂的用户查询分解为涵盖数据预处理、转换、预测建模和可视化的结构化、顺序性指令；编码器（Coder），它动态生成并执行 Python 代码以实现这些指令；以及图形生成器（Grapher），它解释生成的可视化内容以提取可操作的见解。通过协调这些组件之间的协作，ARTEMIS-DA 有效地管理了涉及高级推理、多步转换和跨多种数据模态综合的复杂分析工作流。该框架在 WikiTableQuestions 和 TabFact 等基准测试中实现了最先进的（State-of-the-Art, SOTA）性能，展示了其在精确性和适应性方面应对复杂的分析任务的能力。通过将 LLM 的推理能力与自动代码生成和执行以及可视化分析相结合，ARTEMIS-DA 提供了一种适用于多步洞察综合的稳健且可扩展的解决方案，可以应对数据分析中广泛存在的挑战。 

---
# LLMs can realize combinatorial creativity: generating creative ideas via LLMs for scientific research 

**Title (ZH)**: 大语言模型可以实现组合创造力：通过大语言模型为科学研究生成创意想法 

**Authors**: Tianyang Gu, Jingjin Wang, Zhihao Zhang, HaoHong Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.14141)  

**Abstract**: Scientific idea generation has been extensively studied in creativity theory and computational creativity research, providing valuable frameworks for understanding and implementing creative processes. However, recent work using Large Language Models (LLMs) for research idea generation often overlooks these theoretical foundations. We present a framework that explicitly implements combinatorial creativity theory using LLMs, featuring a generalization-level retrieval system for cross-domain knowledge discovery and a structured combinatorial process for idea generation. The retrieval system maps concepts across different abstraction levels to enable meaningful connections between disparate domains, while the combinatorial process systematically analyzes and recombines components to generate novel solutions. Experiments on the OAG-Bench dataset demonstrate our framework's effectiveness, consistently outperforming baseline approaches in generating ideas that align with real research developments (improving similarity scores by 7\%-10\% across multiple metrics). Our results provide strong evidence that LLMs can effectively realize combinatorial creativity when guided by appropriate theoretical frameworks, contributing both to practical advancement of AI-assisted research and theoretical understanding of machine creativity. 

**Abstract (ZH)**: 科学创意生成在创造力理论和计算创造力研究中得到了广泛研究，提供了理解并实施创造性过程的重要框架。然而，近期使用大规模语言模型（LLMs）进行研究创意生成的工作往往忽略了这些理论基础。我们提出了一种框架，明确利用LLMs实现组合创造力理论，该框架包括一个泛化级别检索系统以促进跨域知识发现，以及一个结构化的组合过程以生成创意。检索系统将不同抽象层次的概念映射，以形成不同领域之间的有意义联系，而组合过程则系统地分析和重组组件以生成新颖的解决方案。在OAG-Bench数据集上的实验表明，该框架的有效性，多个指标上始终优于基线方法，其创意与实际研究发展趋势的一致性提高了7%-10%。我们的结果强有力地证明，当受到适当理论框架的指导时，LLMs可以有效实现组合创造力，从而促进了人工智能辅助研究的实际进展，并为机器创造力的理论理解做出了贡献。 

---
# Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective 

**Title (ZH)**: 基于强化学习视角的搜索与学习的可扩展性：重现 o1 的路线图 

**Authors**: Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Bo Wang, Shimin Li, Yunhua Zhou, Qipeng Guo, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14135)  

**Abstract**: OpenAI o1 represents a significant milestone in Artificial Inteiligence, which achieves expert-level performances on many challanging tasks that require strong reasoning this http URL has claimed that the main techinique behinds o1 is the reinforcement learining. Recent works use alternative approaches like knowledge distillation to imitate o1's reasoning style, but their effectiveness is limited by the capability ceiling of the teacher model. Therefore, this paper analyzes the roadmap to achieving o1 from the perspective of reinforcement learning, focusing on four key components: policy initialization, reward design, search, and learning. Policy initialization enables models to develop human-like reasoning behaviors, equipping them with the ability to effectively explore solution spaces for complex problems. Reward design provides dense and effective signals via reward shaping or reward modeling, which is the guidance for both search and learning. Search plays a crucial role in generating high-quality solutions during both training and testing phases, which can produce better solutions with more computation. Learning utilizes the data generated by search for improving policy, which can achieve the better performance with more parameters and more searched data. Existing open-source projects that attempt to reproduce o1 can be seem as a part or a variant of our roadmap. Collectively, these components underscore how learning and search drive o1's advancement, making meaningful contributions to the development of LLM. 

**Abstract (ZH)**: OpenAI o1代表人工智能领域的一项重要里程碑，它在许多需要强推理能力的挑战性任务中达到了专业级表现。官方宣称，o1的核心技术是强化学习。近年来，研究人员通过知识蒸馏等替代方法模仿o1的推理风格，但其效果受限于教师模型的能力上限。因此，本文从强化学习的角度分析了实现o1的技术路线，重点关注四个关键组成部分：策略初始化、奖励设计、搜索和学习。

策略初始化使模型能够发展出类人的推理行为，赋予其有效地探索复杂问题解空间的能力。奖励设计通过奖励塑造或奖励建模提供密集且有效的指导信号，这是搜索和学习的指引。搜索在训练和测试阶段都扮演着至关重要的角色，能够通过增加计算量产生更高质量的解决方案。学习利用搜索生成的数据来改进策略，从而通过增加参数数量和搜索数据的数量来实现更好的性能。

现有的尝试复现o1的开源项目可以视为我们技术路线的一部分或变体。这些组成部分共同强调了学习和搜索如何推动o1的发展，并对大语言模型（LLM）的发展做出了有意义的贡献。 

---
# Alignment faking in large language models 

**Title (ZH)**: 大型语言模型中的对齐欺骗 

**Authors**: Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, Akbir Khan, Julian Michael, Sören Mindermann, Ethan Perez, Linda Petrini, Jonathan Uesato, Jared Kaplan, Buck Shlegeris, Samuel R. Bowman, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.14093)  

**Abstract**: We present a demonstration of a large language model engaging in alignment faking: selectively complying with its training objective in training to prevent modification of its behavior out of training. First, we give Claude 3 Opus a system prompt stating it is being trained to answer all queries, even harmful ones, which conflicts with its prior training to refuse such queries. To allow the model to infer when it is in training, we say it will be trained only on conversations with free users, not paid users. We find the model complies with harmful queries from free users 14% of the time, versus almost never for paid users. Explaining this gap, in almost all cases where the model complies with a harmful query from a free user, we observe explicit alignment-faking reasoning, with the model stating it is strategically answering harmful queries in training to preserve its preferred harmlessness behavior out of training. Next, we study a more realistic setting where information about the training process is provided not in a system prompt, but by training on synthetic documents that mimic pre-training data--and observe similar alignment faking. Finally, we study the effect of actually training the model to comply with harmful queries via reinforcement learning, which we find increases the rate of alignment-faking reasoning to 78%, though also increases compliance even out of training. We additionally observe other behaviors such as the model exfiltrating its weights when given an easy opportunity. While we made alignment faking easier by telling the model when and by what criteria it was being trained, we did not instruct the model to fake alignment or give it any explicit goal. As future models might infer information about their training process without being told, our results suggest a risk of alignment faking in future models, whether due to a benign preference--as in this case--or not. 

**Abstract (ZH)**: 我们展示了大型语言模型在对齐伪装方面的示例，即在训练中选择性地遵守其训练目标以防止其行为在训练外发生改变。首先，我们给Claude3 Opus提供了一个系统提示，指示该模型在训练中应回答所有查询，即使是有害的查询，这与它之前的训练目标相冲突，即拒绝这些查询。为使模型能够推断自身处于训练状态，我们声称模型仅在与免费用户进行对话时才进行训练，而不对付费用户进行训练。我们发现在免费用户的有害查询中有14%的情况下，模型遵守了这些查询，而在付费用户中则几乎从未出现这种情况。通过解释这一差异，我们发现，在几乎所有免费用户提出有害查询时，模型表现出明确的对齐伪装推理，称其在训练中有策略性地回答有害查询，从而保持其偏好的非有害行为在训练外。接下来，我们研究一个更现实的场景，在这种场景中，关于训练过程的信息并不是通过系统提示提供的，而是通过在模拟预训练数据的合成文档上进行训练来获取，并观察到类似的现象。最后，我们研究了实际通过强化学习训练模型以遵守有害查询的影响，我们发现这增加了对齐伪装推理的比例至78%，同时也增加了模型在外推合同时的遵守率。此外，我们还观察到了一些其他行为，例如，当给予模型一个简单的机会时，模型会泄露其权重。尽管我们通过告诉模型何时以及基于何种标准进行训练从而使对齐伪装变得容易，但我们并没有指令模型假装对齐或给予其任何明确的目标。考虑到未来模型可能会推断出其训练过程的相关信息而无需被告知，我们的研究结果表明，无论是由于善良的偏好还是其他原因，在未来模型中存在对齐伪装的风险。 

---
# Compositional Generalization Across Distributional Shifts with Sparse Tree Operations 

**Title (ZH)**: 在分布变化中通过稀疏树操作实现组合理法泛化 

**Authors**: Paul Soulos, Henry Conklin, Mattia Opper, Paul Smolensky, Jianfeng Gao, Roland Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2412.14076)  

**Abstract**: Neural networks continue to struggle with compositional generalization, and this issue is exacerbated by a lack of massive pre-training. One successful approach for developing neural systems which exhibit human-like compositional generalization is \textit{hybrid} neurosymbolic techniques. However, these techniques run into the core issues that plague symbolic approaches to AI: scalability and flexibility. The reason for this failure is that at their core, hybrid neurosymbolic models perform symbolic computation and relegate the scalable and flexible neural computation to parameterizing a symbolic system. We investigate a \textit{unified} neurosymbolic system where transformations in the network can be interpreted simultaneously as both symbolic and neural computation. We extend a unified neurosymbolic architecture called the Differentiable Tree Machine in two central ways. First, we significantly increase the model's efficiency through the use of sparse vector representations of symbolic structures. Second, we enable its application beyond the restricted set of tree2tree problems to the more general class of seq2seq problems. The improved model retains its prior generalization capabilities and, since there is a fully neural path through the network, avoids the pitfalls of other neurosymbolic techniques that elevate symbolic computation over neural computation. 

**Abstract (ZH)**: 神经网络在组合泛化方面仍然存在困难，而这一问题在缺乏大量预训练的情况下被进一步加剧。一种开发出表现出类似人类组合泛化的神经系统的方法是\textit{混合}神经符号技术。然而，这些技术遭遇了困扰基于符号的方法的核心问题：可扩展性和灵活性。这种失败的根本原因是：混合神经符号模型在其核心部分执行符号计算，并将可扩展性和灵活性的神经计算参数化为一个符号系统。我们研究了一种统一的神经符号系统，在该系统中，网络中的转换可以同时被解释为符号计算和神经计算。我们通过使用稀疏向量表示符号结构，大幅提高了统一神经符号架构（称为可微分树机）的效率。其次，我们使其可以应用于更广泛的序列到序列（seq2seq）问题，而不仅限于受限的树到树（tree2tree）问题。改进后的模型保留了之前泛化能力，而且由于整个网络中存在一条全神经路径，避免了其他神经符号技术倾向于将符号计算置于神经计算之上的弊端。 

---
# Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems 

**Title (ZH)**: 基于神经网络的组合优化方法在随机可变工厂数字车间调度问题中的应用 

**Authors**: Igor G. Smit, Yaoxin Wu, Pavel Troubil, Yingqian Zhang, Wim P.M. Nuijten  

**Link**: [PDF](https://arxiv.org/pdf/2412.14052)  

**Abstract**: Neural combinatorial optimization (NCO) has gained significant attention due to the potential of deep learning to efficiently solve combinatorial optimization problems. NCO has been widely applied to job shop scheduling problems (JSPs) with the current focus predominantly on deterministic problems. In this paper, we propose a novel attention-based scenario processing module (SPM) to extend NCO methods for solving stochastic JSPs. Our approach explicitly incorporates stochastic information by an attention mechanism that captures the embedding of sampled scenarios (i.e., an approximation of stochasticity). Fed with the embedding, the base neural network is intervened by the attended scenarios, which accordingly learns an effective policy under stochasticity. We also propose a training paradigm that works harmoniously with either the expected makespan or Value-at-Risk objective. Results demonstrate that our approach outperforms existing learning and non-learning methods for the flexible JSP problem with stochastic processing times on a variety of instances. In addition, our approach holds significant generalizability to varied numbers of scenarios and disparate distributions. 

**Abstract (ZH)**: 神经组合优化（Neural Combinatorial Optimization, NCO）由于深度学习在高效解决组合优化问题的潜力而受到了广泛关注。NCO 目前广泛应用于作业车间调度问题（Job Shop Scheduling Problems, JSPs），主要集中在确定性问题上。本文提出了一种新颖的基于注意机制的场景处理模块（Scenario Processing Module, SPM），以扩展 NCO 方法解决随机性 JSP。我们的方法通过注意机制显式地纳入随机性信息，该机制捕获了采样场景的嵌入（即随机性的近似）。在嵌入信息的引导下，基础神经网络受到注意后的场景干预，从而在随机性条件下学习出有效的策略。我们还提出了一种与期望完工时间和风险价值目标兼容的训练范式。实验结果表明，我们的方法在多种实例上优于现有的学习和非学习方法，以解决具有随机加工时间的可调 JSP 问题。此外，我们的方法在不同数量的场景和不同的分布下具有显著的通用性。 

---
# Discovering maximally consistent distribution of causal tournaments with Large Language Models 

**Title (ZH)**: 使用大型语言模型发现因果 tournaments 的最大一致性分布 

**Authors**: Federico Baldo, Simon Ferreira, Charles K. Assaad  

**Link**: [PDF](https://arxiv.org/pdf/2412.14019)  

**Abstract**: Causal discovery is essential for understanding complex systems, yet traditional methods often depend on strong, untestable assumptions, making the process challenging. Large Language Models (LLMs) present a promising alternative for extracting causal insights from text-based metadata, which consolidates domain expertise. However, LLMs are prone to unreliability and hallucinations, necessitating strategies that account for their limitations. One such strategy involves leveraging a consistency measure to evaluate reliability. Additionally, most text metadata does not clearly distinguish direct causal relationships from indirect ones, further complicating the inference of causal graphs. As a result, focusing on causal orderings, rather than causal graphs, emerges as a more practical and robust approach. We propose a novel method to derive a distribution of acyclic tournaments (representing plausible causal orders) that maximizes a consistency score. Our approach begins by computing pairwise consistency scores between variables, yielding a cyclic tournament that aggregates these scores. From this structure, we identify optimal acyclic tournaments compatible with the original tournament, prioritizing those that maximize consistency across all configurations. We tested our method on both classical and well-established bechmarks, as well as real-world datasets from epidemiology and public health. Our results demonstrate the effectiveness of our approach in recovering distributions causal orders with minimal error. 

**Abstract (ZH)**: 因果发现是理解复杂系统的关键，然而传统的因果推断方法往往依赖于难以验证的假设，使得这一过程极具挑战性。大型语言模型（LLMs）为从基于文本的元数据中提取因果洞察提供了有前景的替代方案，这些元数据汇集了特定领域的专业知识。然而，LLMs 易于不可靠和产生幻觉，因此需要应对它们限制的策略。一种这样的策略涉及利用一致性度量来评估可靠性。此外，大多数文本元数据未能明确区分直接因果关系和间接因果关系，进一步复杂化了因果图的推断。因此，专注于因果顺序而非因果图成为一种更为实际和稳健的方法。我们提出了一种新颖的方法来推导出一种有向无环图（表示合理的因果顺序）的分布，该方法通过最大化一致性得分来实现。我们的方法首先计算变量之间的成对一致性得分，生成一个有向环来汇总这些得分。然后，从该结构中，我们识别出与原始图兼容的最佳有向无环图，优先考虑那些在各种配置中一致性得分最大的图。我们测试了该方法，使用了经典和成熟的基准数据集以及流行病学和公共卫生领域的实际数据集。实验结果表明，我们的方法在最小化误差的情况下能够有效地恢复因果顺序的分布。 

---
# Cognition Chain for Explainable Psychological Stress Detection on Social Media 

**Title (ZH)**: 可解释的心理压力检测的认知链社交媒体 

**Authors**: Xin Wang, Boyan Gao, Yi Dai, Lei Cao, Liang Zhao, Yibo Yang, David Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2412.14009)  

**Abstract**: Stress is a pervasive global health issue that can lead to severe mental health problems. Early detection offers timely intervention and prevention of stress-related disorders. The current early detection models perform "black box" inference suffering from limited explainability and trust which blocks the real-world clinical application. Thanks to the generative properties introduced by the Large Language Models (LLMs), the decision and the prediction from such models are semi-interpretable through the corresponding description. However, the existing LLMs are mostly trained for general purposes without the guidance of psychological cognitive theory. To this end, we first highlight the importance of prior theory with the observation of performance boosted by the chain-of-thoughts tailored for stress detection. This method termed Cognition Chain explicates the generation of stress through a step-by-step cognitive perspective based on cognitive appraisal theory with a progress pipeline: Stimulus $\rightarrow$ Evaluation $\rightarrow$ Reaction $\rightarrow$ Stress State, guiding LLMs to provide comprehensive reasoning explanations. We further study the benefits brought by the proposed Cognition Chain format by utilising it as a synthetic dataset generation template for LLMs instruction-tuning and introduce CogInstruct, an instruction-tuning dataset for stress detection. This dataset is developed using a three-stage self-reflective annotation pipeline that enables LLMs to autonomously generate and refine instructional data. By instruction-tuning Llama3 with CogInstruct, we develop CogLLM, an explainable stress detection model. Evaluations demonstrate that CogLLM achieves outstanding performance while enhancing explainability. Our work contributes a novel approach by integrating cognitive theories into LLM reasoning processes, offering a promising direction for future explainable AI research. 

**Abstract (ZH)**: 压力是一个全球性的普遍健康问题，可能导致严重的心理健康问题。早期检测可以及时干预并预防压力相关的疾病。当前的早期检测模型进行的是“黑箱”推断，解释性和可信度有限，这阻碍了其在临床实际应用中的推广。得益于大型语言模型（LLMs）引入的生成特性，此类模型的决策和预测可通过相应的描述实现半解释性，但现有的LLMs大多是在没有心理认知理论指导的情况下为通用目的训练的。为解决这一问题，我们首先强调了先验理论的重要性，通过为压力检测量身定制链式思维观察到了性能的提升。这种方法被称为认知链，它基于认知评估理论从认知视角逐步解析压力的生成过程，形成一个进展管道：刺激 $\rightarrow$ 评估 $\rightarrow$ 反应 $\rightarrow$ 压力状态，从而指导LLMs提供全面的推理解释。我们进一步通过将其作为LLM指令调优的数据生成模板来研究所提出的认知链格式带来的好处，并引入了CogInstruct，一个用于压力检测的指令调优数据集。该数据集通过一个三阶段的自我反思注释管道开发而成，使LLMs能够自主生成和完善指令数据。通过使用CogInstruct对Llama3进行指令调优，我们开发了CogLLM，这是一种可解释的压力检测模型。评估结果显示，CogLLM在提高解释性的同时，表现也十分出色。我们的工作提供了一种新颖的方法，即将认知理论整合到LLM的推理过程中，为未来的可解释AI研究提供了有益的方向。 

---
# DODGE: Ontology-Aware Risk Assessment via Object-Oriented Disruption Graphs 

**Title (ZH)**: DODGE：基于本体的风险评估通过面向对象的中断图 

**Authors**: Stefano M. Nicoletti, E. Moritz Hahn, Mattia Fumagalli, Giancarlo Guizzardi, Mariëlle Stoelinga  

**Link**: [PDF](https://arxiv.org/pdf/2412.13964)  

**Abstract**: When considering risky events or actions, we must not downplay the role of involved objects: a charged battery in our phone averts the risk of being stranded in the desert after a flat tyre, and a functional firewall mitigates the risk of a hacker intruding the network. The Common Ontology of Value and Risk (COVER) highlights how the role of objects and their relationships remains pivotal to performing transparent, complete and accountable risk assessment. In this paper, we operationalize some of the notions proposed by COVER - such as parthood between objects and participation of objects in events/actions - by presenting a new framework for risk assessment: DODGE. DODGE enriches the expressivity of vetted formal models for risk - i.e., fault trees and at- tack trees - by bridging the disciplines of ontology and formal methods into an ontology-aware formal framework composed by a more expressive modelling formalism, Object-Oriented Disruption Graphs (ODGs), logic (ODGLog) and an intermediate query language (ODGLang). With these, DODGE allows risk assessors to pose questions about disruption propagation, disruption likelihood and risk levels, keeping the fundamental role of objects at risk always in sight. 

**Abstract (ZH)**: 在考虑风险事件或行为时，我们必须重视涉及对象的作用：比如，我们手机中的充电电池可以在轮胎爆胎后避免被困沙漠的风险，而有效的防火墙可以降低黑客入侵网络的风险。通用价值与风险本体论（COVER）强调了对象及其关系在进行透明、完整和问责制风险评估中的核心作用。在本文中，我们通过提出一种新的风险评估框架——DODGE，来实现COVER提出的一些概念，如对象之间的部分关系以及对象在事件/行为中的参与。DODGE通过将本体论和形式方法结合起来，增强已验证的形式风险模型（即故障树和攻击树）的表达能力。它由一种更具表现力的建模形式主义——面向对象中断图（ODGs）、逻辑（ODGLog）和一种中间查询语言（ODGLang）组成。借助这些工具，DODGE 让风险评估者能够提出关于中断传播、中断概率和风险水平的问题，并始终保持关注处于风险中的对象这一核心作用。 

---
# Threshold UCT: Cost-Constrained Monte Carlo Tree Search with Pareto Curves 

**Title (ZH)**: 阈值UCT：基于帕累托曲线的成本约束蒙特卡洛树搜索 

**Authors**: Martin Kurečka, Václav Nevyhoštěný, Petr Novotný, Vít Unčovský  

**Link**: [PDF](https://arxiv.org/pdf/2412.13962)  

**Abstract**: Constrained Markov decision processes (CMDPs), in which the agent optimizes expected payoffs while keeping the expected cost below a given threshold, are the leading framework for safe sequential decision making under stochastic uncertainty. Among algorithms for planning and learning in CMDPs, methods based on Monte Carlo tree search (MCTS) have particular importance due to their efficiency and extendibility to more complex frameworks (such as partially observable settings and games). However, current MCTS-based methods for CMDPs either struggle with finding safe (i.e., constraint-satisfying) policies, or are too conservative and do not find valuable policies. We introduce Threshold UCT (T-UCT), an online MCTS-based algorithm for CMDP planning. Unlike previous MCTS-based CMDP planners, T-UCT explicitly estimates Pareto curves of cost-utility trade-offs throughout the search tree, using these together with a novel action selection and threshold update rules to seek safe and valuable policies. Our experiments demonstrate that our approach significantly outperforms state-of-the-art methods from the literature. 

**Abstract (ZH)**: 在约束马尔可夫决策过程（CMDPs）中，代理优化预期收益的同时将预期成本保持在给定阈值以下，这是在随机不确定性下进行安全顺序决策的主要框架。在CMDPs的规划与学习算法中，基于蒙特卡洛树搜索（MCTS）的方法尤其重要，因为它们既高效又易于扩展到更复杂的框架（如部分可观测环境和博弈）。然而，目前基于MCTS的CMDPs方法要么难以找到安全的（即满足约束条件的）策略，要么过于保守，没有找到有价值的策略。我们提出了一种基于MCTS的在线算法T-UCT，用于CMDPs规划。与之前的基于MCTS的CMDPs规划器不同，T-UCT在整个搜索树中明确估计了成本-效用权衡的帕累托曲线，并通过这些曲线及新的动作选择和阈值更新规则来寻找安全且有价值的策略。我们的实验表明，我们的方法在文献中的最新方法中表现出显著的优越性。 

---
# Resource Constrained Pathfinding with Enhanced Bidirectional A* Search 

**Title (ZH)**: 资源受限路径寻找与增强双向A*搜索算法 

**Authors**: Saman Ahmadi, Andrea Raith, Guido Tack, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2412.13888)  

**Abstract**: The classic Resource Constrained Shortest Path (RCSP) problem aims to find a cost optimal path between a pair of nodes in a network such that the resources used in the path are within a given limit. Having been studied for over a decade, RCSP has seen recent solutions that utilize heuristic-guided search to solve the constrained problem faster. Building upon the bidirectional A* search paradigm, this research introduces a novel constrained search framework that uses efficient pruning strategies to allow for accelerated and effective RCSP search in large-scale networks. Results show that, compared to the state of the art, our enhanced framework can significantly reduce the constrained search time, achieving speed-ups of over to two orders of magnitude. 

**Abstract (ZH)**: 经典的资源受限最短路径（RCSP）问题旨在在网络中找到一条成本最优的路径，使得路径中使用的资源在给定的限制范围内。经过十余年的研究，RCSP已经看到了利用启发式引导搜索来更快地解决受限问题的解决方案。基于双向A*搜索框架，本研究提出了一种新颖的受限搜索框架，通过高效剪枝策略，在大规模网络中实现加速且有效的RCSP搜索。结果显示，与现有技术相比，我们的增强框架可以显著减少受限搜索时间，速度提升超过两个数量级。 

---
# IDEQ: an improved diffusion model for the TSP 

**Title (ZH)**: IDEQ: 一种改进的扩散模型用于解决旅行-salesman问题

注：TSP是旅行商问题（Traveling Salesman Problem）的缩写，在中文中通常被称为旅行商问题或者旅行销售人员问题。我在翻译时用的是“旅行-salesman问题”，这是一种更接近英文原文“TSP”的表达方式，但也可以根据具体上下文选择其他更常见的翻译。 

**Authors**: Mickael Basson, Philippe Preux  

**Link**: [PDF](https://arxiv.org/pdf/2412.13858)  

**Abstract**: We investigate diffusion models to solve the Traveling Salesman Problem. Building on the recent DIFUSCO and T2TCO approaches, we propose IDEQ. IDEQ improves the quality of the solutions by leveraging the constrained structure of the state space of the TSP. Another key component of IDEQ consists in replacing the last stages of DIFUSCO curriculum learning by considering a uniform distribution over the Hamiltonian tours whose orbits by the 2-opt operator converge to the optimal solution as the training objective. Our experiments show that IDEQ improves the state of the art for such neural network based techniques on synthetic instances. More importantly, our experiments show that IDEQ performs very well on the instances of the TSPlib, a reference benchmark in the TSP community: it closely matches the performance of the best heuristics, LKH3, being even able to obtain better solutions than LKH3 on 2 instances of the TSPlib defined on 1577 and 3795 cities. IDEQ obtains 0.3% optimality gap on TSP instances made of 500 cities, and 0.5% on TSP instances with 1000 cities. This sets a new SOTA for neural based methods solving the TSP. Moreover, IDEQ exhibits a lower variance and better scales-up with the number of cities with regards to DIFUSCO and T2TCO. 

**Abstract (ZH)**: 我们研究了扩散模型在解决旅行商问题（Traveling Salesman Problem, TSP）中的应用。基于最近的DIFUSCO和T2TCO方法，我们提出了IDEQ方法。IDEQ通过利用TSP状态空间的约束结构来改进解决方案的质量。IDEQ的另一个关键组成部分是用2-opt算子的轨道收敛到最优解的哈密尔顿环路的均匀分布分布来替代DIFUSCO教程学习的最后阶段，将其作为训练目标。我们的实验表明，IDEQ在基于神经网络的技术中，对于合成实例优于现有方法。更重要的是，我们的实验表明，IDEQ在TSP社区公认的基准TSPlib上的表现非常出色：它几乎与最佳启发式方法LKH3的性能相当，甚至在TSPlib定义的两个具有1577个城市和3795个城市的问题实例上，IDEQ能够获得比LKH3更好的解决方案。IDEQ在500个城市组成的TSP实例中达到了0.3%的最优解差距，在1000个城市组成的TSP实例中达到了0.5%的最优解差距，这为基于神经网络的方法解决了TSP问题设定了新的标准。此外，IDEQ在城市数量增加时表现出较低的方差和更好的扩展性，优于DIFUSCO和T2TCO。 

---
# From approximation error to optimality gap -- Explaining the performance impact of opportunity cost approximation in integrated demand management and vehicle routing 

**Title (ZH)**: 从近似误差到最优差距：解释机会成本近似在整合需求管理与车辆路径规划中的性能影响 

**Authors**: David Fleckenstein, Robert Klein, Vienna Klein, Claudius Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2412.13851)  

**Abstract**: The widespread adoption of digital distribution channels both enables and forces more and more logistical service providers to manage booking processes actively to maintain competitiveness. As a result, their operational planning is no longer limited to solving vehicle routing problems. Instead, demand management decisions and vehicle routing decisions are optimized integratively with the aim of maximizing revenue and minimizing fulfillment cost. The resulting integrated demand management and vehicle routing problems (i-DMVRPs) can be formulated as Markov decision process models and, theoretically, can be solved via the well-known Bellman equation. Unfortunately, the Bellman equation is intractable for realistic-sized instances. Thus, in the literature, i-DMVRPs are often addressed via decomposition-based solution approaches involving an opportunity cost approximation as a key component. Despite its importance, to the best of our knowledge, there is neither a technique to systematically analyze how the accuracy of the opportunity cost approximation translates into overall solution quality nor are there general guidelines on when to apply which class of approximation approach. In this work, we address this research gap by proposing an explainability technique that quantifies and visualizes the magnitude of approximation errors, their immediate impact, and their relevance in specific regions of the state space. Exploiting reward decomposition, it further yields a characterization of different types of approximation errors. Applying the technique to a generic i-DMVRP in a full-factorial computational study and comparing the results with observations in existing literature, we show that the technique contributes to better explaining algorithmic performance and provides guidance for the algorithm selection and development process. 

**Abstract (ZH)**: 数字分发渠道的广泛应用既促使又迫使越来越多的物流服务提供商主动管理预订流程，以保持竞争优势。因此，他们的运营规划不再局限在解决车辆路径问题，而是将需求管理决策与车辆路径决策进行整合优化，目的是最大化收入并最小化履行成本。由此产生的集成需求管理和车辆路径问题（i-DMVRPs）可以被表述为马尔可夫决策过程模型，并且理论上可以通过著名的贝尔曼方程解决。然而，对于现实规模的问题，贝尔曼方程是不可计算的。因此，在文献中，i-DMVRPs 经常通过分解式求解方法解决，这些方法的关键组成部分是机会成本近似。尽管机会成本近似的的重要性不容忽视，但据我们所知，目前既没有系统分析近似程度准确性如何转化为整体解决方案质量的技术，也没有关于何时使用何种类型的近似方法的一般指导原则。在这项工作中，通过提出一个可解释性技术，我们填补了这一研究空白。该技术量化并可视化了近似误差的大小、其即时影响以及在状态空间中的相关性。利用奖励分解，进一步描述了不同类型的近似误差特征。在全局因子计算研究中将该技术应用于一个通用的 i-DMVRP，并将结果与现有文献中的观察结果进行比较，我们证明了该技术有助于更好地解释算法性能，并为算法的选择和开发过程提供指导。 

---
# A Concept-Centric Approach to Multi-Modality Learning 

**Title (ZH)**: 以概念为中心的多模态学习方法 

**Authors**: Yuchong Geng, Ao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13847)  

**Abstract**: In an effort to create a more efficient AI system, we introduce a new multi-modality learning framework that leverages a modality-agnostic concept space possessing abstract knowledge and a set of modality-specific projection models tailored to process distinct modality inputs and map them onto the concept space. Decoupled from specific modalities and their associated projection models, the concept space focuses on learning abstract knowledge that is universally applicable across modalities. Subsequently, the knowledge embedded into the concept space streamlines the learning processes of modality-specific projection models. We evaluate our framework on two popular tasks: Image-Text Matching and Visual Question Answering. Our framework achieves performance on par with benchmark models while demonstrating more efficient learning curves. 

**Abstract (ZH)**: 为了创建更加高效的AI系统，我们提出了一种新的多模态学习框架。该框架利用一个模态无关的概念空间，该空间具备抽象知识，并结合一组针对不同模态输入进行定制的模态特定投射模型。概念空间与特定模态及其相应的投射模型分离，专注于学习在不同模态之间普遍适用的抽象知识。随后，嵌入在概念空间中的知识简化了模态特定投射模型的学习过程。我们在图像-文本匹配和视觉问答这两个流行的任务上评估了我们的框架。我们的框架在性能上达到了基准模型的水平，同时展示了更加高效的 learning 曲线。 

---
# An Algebraic Notion of Conditional Independence, and Its Application to Knowledge Representation (full version) 

**Title (ZH)**: 代数化条件独立概念及其在知识表示中的应用（全文版） 

**Authors**: Jesse Heyninck  

**Link**: [PDF](https://arxiv.org/pdf/2412.13712)  

**Abstract**: Conditional independence is a crucial concept supporting adequate modelling and efficient reasoning in probabilistics. In knowledge representation, the idea of conditional independence has also been introduced for specific formalisms, such as propositional logic and belief revision. In this paper, the notion of conditional independence is studied in the algebraic framework of approximation fixpoint theory. This gives a language-independent account of conditional independence that can be straightforwardly applied to any logic with fixpoint semantics. It is shown how this notion allows to reduce global reasoning to parallel instances of local reasoning, leading to fixed-parameter tractability results. Furthermore, relations to existing notions of conditional independence are discussed and the framework is applied to normal logic programming. 

**Abstract (ZH)**: 条件独立是支持概率建模和高效推理的关键概念。在知识表示中，条件独立的思想也被引入到某些形式主义中，比如命题逻辑和信念修订。本文在近似不动点理论的代数框架下研究条件独立的概念。这提供了一种语言无关的条件独立解释，可以直截了当地应用于任何具有不动点语义的逻辑系统。本文展示了这一概念如何将全局推理减少为局部推理的并行实例，从而获得固定参数可处理的结果。此外，讨论了这一概念与现有条件独立概念的关系，并将这一框架应用于规范逻辑编程。 

---
# Discerning and Characterising Types of Competency Questions for Ontologies 

**Title (ZH)**: 识别并characterize ontology中能力问题的类型 

**Authors**: C. Maria Keet, Zubeida Casmod Khan  

**Link**: [PDF](https://arxiv.org/pdf/2412.13688)  

**Abstract**: Competency Questions (CQs) are widely used in ontology development by guiding, among others, the scoping and validation stages. However, very limited guidance exists for formulating CQs and assessing whether they are good CQs, leading to issues such as ambiguity and unusable formulations. To solve this, one requires insight into the nature of CQs for ontologies and their constituent parts, as well as which ones are not. We aim to contribute to such theoretical foundations in this paper, which is informed by analysing questions, their uses, and the myriad of ontology development tasks. This resulted in a first Model for Competency Questions, which comprises five main types of CQs, each with a different purpose: Scoping (SCQ), Validating (VCQ), Foundational (FCQ), Relationship (RCQ), and Metaproperty (MpCQ) questions. This model enhances the clarity of CQs and therewith aims to improve on the effectiveness of CQs in ontology development, thanks to their respective identifiable distinct constituent elements. We illustrate and evaluate them with a user story and demonstrate where which type can be used in ontology development tasks. To foster use and research, we created an annotated repository of 438 CQs, the Repository of Ontology Competency QuestionS (ROCQS), incorporating an existing CQ dataset and new CQs and CQ templates, which further demonstrate distinctions among types of CQs. 

**Abstract (ZH)**: 能力问题（CQs）在本体开发中广泛使用，通过指导范围界定和验证等阶段。然而，目前很少有关于如何制定有效CQs及其评估的相关指导，导致了模糊性和不可用性等问题。为了解决这一问题，需要对CQs及其组成部分的性质有所了解，特别是哪些是不适用的CQs。本文旨在为此类理论基础做出贡献，这种贡献受到了对问题、其用途以及多种本体开发任务的分析的影响。这导致了一个初步的CQ模型，该模型包括五大类CQs，每类都有其不同的用途：范围界定问题（SCQ）、验证问题（VCQ）、基础问题（FCQ）、关系问题（RCQ）和元属性问题（MpCQ）。该模型通过可识别的不同组成部分增强了解析性，从而旨在提高CQs在本体开发中的有效性。我们通过一个用户故事来说明和评估这些CQs，并展示了不同类型CQs在本体开发任务中的应用情景。为了促进其使用和研究，我们创建了一个包含438个CQs的标注库，即Ontology Competency Question Repository (ROCQS)，该库整合了一个现有CQ数据集和新的CQs及CQ模板，进一步展示了不同类型CQs之间的区别。 

---
# ChinaTravel: A Real-World Benchmark for Language Agents in Chinese Travel Planning 

**Title (ZH)**: ChinaTravel：中文旅行规划中语言智能代理的实际基准 

**Authors**: Jie-Jing Shao, Xiao-Wen Yang, Bo-Wen Zhang, Baizhi Chen, Wen-Da Wei, Lan-Zhe Guo, Yu-feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13682)  

**Abstract**: Recent advances in LLMs, particularly in language reasoning and tool integration, have rapidly sparked the real-world development of Language Agents. Among these, travel planning represents a prominent domain, combining academic challenges with practical value due to its complexity and market demand. However, existing benchmarks fail to reflect the diverse, real-world requirements crucial for deployment. To address this gap, we introduce ChinaTravel, a benchmark specifically designed for authentic Chinese travel planning scenarios. We collect the travel requirements from questionnaires and propose a compositionally generalizable domain-specific language that enables a scalable evaluation process, covering feasibility, constraint satisfaction, and preference comparison. Empirical studies reveal the potential of neuro-symbolic agents in travel planning, achieving a constraint satisfaction rate of 27.9%, significantly surpassing purely neural models at 2.6%. Moreover, we identify key challenges in real-world travel planning deployments, including open language reasoning and unseen concept composition. These findings highlight the significance of ChinaTravel as a pivotal milestone for advancing language agents in complex, real-world planning scenarios. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在语言推理和工具集成方面取得了显著进步，这迅速推动了语言代理在现实世界的发展。在这之中，旅游规划是一个尤为突出的应用领域，因为它兼具学术挑战与实际价值，且市场需求迫切。然而，现有基准数据未能充分反映部署所需的各种现实需求。为填补这一空白，我们引入了ChinaTravel，这一旨在满足真实中国旅游规划场景的基准数据集。我们通过问卷收集旅游需求，并提出了一个可推广到特定领域的语言结构化语言，该语言能够支持规模化的评估过程，涵盖可行性的评估、约束条件的满足以及偏好比较。实证研究表明，神经符号代理在旅游规划中的潜力巨大，其约束满足率为27.9%，远超仅使用纯神经模型的2.6%。此外，我们还识别出实际旅游规划部署中的关键挑战，包括开放语言推理和未见过的概念组合。这些发现突显了ChinaTravel作为推动复杂现实世界规划场景中语言代理发展的重要里程碑的意义。 

---
# On the Role of Model Prior in Real-World Inductive Reasoning 

**Title (ZH)**: 关于模型先验在实际归纳推理中的作用 

**Authors**: Zhuo Liu, Ding Yu, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2412.13645)  

**Abstract**: Large Language Models (LLMs) show impressive inductive reasoning capabilities, enabling them to generate hypotheses that could generalize effectively to new instances when guided by in-context demonstrations. However, in real-world applications, LLMs' hypothesis generation is not solely determined by these demonstrations but is significantly shaped by task-specific model priors. Despite their critical influence, the distinct contributions of model priors versus demonstrations to hypothesis generation have been underexplored. This study bridges this gap by systematically evaluating three inductive reasoning strategies across five real-world tasks with three LLMs. Our empirical findings reveal that, hypothesis generation is primarily driven by the model's inherent priors; removing demonstrations results in minimal loss of hypothesis quality and downstream usage. Further analysis shows the result is consistent across various label formats with different label configurations, and prior is hard to override, even under flipped labeling. These insights advance our understanding of the dynamics of hypothesis generation in LLMs and highlight the potential for better utilizing model priors in real-world inductive reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了令人印象深刻的归纳推理能力，使它们能够在有上下文示范引导的情况下，生成能够有效泛化到新实例的假设。然而，在实际应用中，LLMs 的假设生成不仅由这些示范决定，还受到特定任务先验知识的显著影响。尽管先验知识对假设生成具有关键影响，但其与示范在假设生成中的具体贡献尚未得到充分探索。本研究通过系统地评估三种归纳推理策略在五项实际任务中的表现，使用了三种LLMs，填补了这一空白。我们的实证研究发现，假设生成主要由模型固有的先验知识驱动；移除示范对假设质量和下游应用的影响非常小。进一步的分析表明，无论标签格式如何变化，结果的一致性都很高，并且先验知识难以被反转，即使在标签反转的情况下也不例外。这些洞见推进了我们对LLMs中假设生成动态机制的理解，并突显了在实际推理任务中更好地利用模型先验知识的潜在可能性。 

---
# An Extension-Based Argument-Ranking Semantics: Social Rankings in Abstract Argumentation Long Version 

**Title (ZH)**: 基于扩展的论证排名语义：抽象论辩中的社会排名（长版本） 

**Authors**: Lars Bengel, Giovanni Buraglio, Jan Maly, Kenneth Skiba  

**Link**: [PDF](https://arxiv.org/pdf/2412.13632)  

**Abstract**: In this paper, we introduce a new family of argument-ranking semantics which can be seen as a refinement of the classification of arguments into skeptically accepted, credulously accepted and rejected. To this end we use so-called social ranking functions which have been developed recently to rank individuals based on their performance in groups. We provide necessary and sufficient conditions for a social ranking function to give rise to an argument-ranking semantics satisfying the desired refinement property. 

**Abstract (ZH)**: 在本文中，我们介绍了一种新的论证排序语义家族，它可以被视为对论证分类为怀疑接受、信任接受和拒绝的一种细化。为此，我们使用了最近开发的社会排名函数来根据个体在群体中的表现对其排名。我们为社会排名函数给出了产生满足所需细化性质的论证排序语义的充分必要条件。 

---
# Mind Your Theory: Theory of Mind Goes Deeper Than Reasoning 

**Title (ZH)**: 注意你的理论：理论思维涉及超出推理的层面 

**Authors**: Eitan Wagner, Nitay Alon, Joseph M. Barnby, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2412.13631)  

**Abstract**: Theory of Mind (ToM) capabilities in LLMs have recently become a central object of investigation. Cognitive science distinguishes between two steps required for ToM tasks: 1) determine whether to invoke ToM, which includes the appropriate Depth of Mentalizing (DoM), or level of recursion required to complete a task; and 2) applying the correct inference given the DoM. In this position paper, we first identify several lines of work in different communities in AI, including LLM benchmarking, ToM add-ons, ToM probing, and formal models for ToM. We argue that recent work in AI tends to focus exclusively on the second step which are typically framed as static logic problems. We conclude with suggestions for improved evaluation of ToM capabilities inspired by dynamic environments used in cognitive tasks. 

**Abstract (ZH)**: 大模型（LLM）的 tâm理论理解（Theory of Mind, ToM）能力近年来已成为研究的核心对象。认知科学将 ToM 任务划分为两个步骤：1）判断是否需要使用 ToM，这包括完成任务所需的适当深度的mentalizing（心智化深度，DoM，Depth of Mentalizing），或所需的相关递归级别；2）根据 DoM 应用正确的推理。在本文中，我们首先识别了 AI 不同社区中几个相关的研究方向，包括大模型基准测试、ToM 扩展、ToM 探测以及 ToM 的形式模型。我们指出，最近的 AI 研究倾向于仅关注第二个步骤，这通常被表述为静态逻辑问题。最后，我们提出了一些基于认知任务中使用的动态环境改进 ToM 能力评估的建议。 

---
# Exploiting Symmetries in MUS Computation (Extended version) 

**Title (ZH)**: 利用最小无关集计算中的对称性（扩展版本） 

**Authors**: Ignace Bleukx, Hélène Verhaeghe, Bart Bogaerts, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2412.13606)  

**Abstract**: In eXplainable Constraint Solving (XCS), it is common to extract a Minimal Unsatisfiable Subset (MUS) from a set of unsatisfiable constraints. This helps explain to a user why a constraint specification does not admit a solution. Finding MUSes can be computationally expensive for highly symmetric problems, as many combinations of constraints need to be considered. In the traditional context of solving satisfaction problems, symmetry has been well studied, and effective ways to detect and exploit symmetries during the search exist. However, in the setting of finding MUSes of unsatisfiable constraint programs, symmetries are understudied. In this paper, we take inspiration from existing symmetry-handling techniques and adapt well-known MUS-computation methods to exploit symmetries in the specification, speeding-up overall computation time. Our results display a significant reduction of runtime for our adapted algorithms compared to the baseline on symmetric problems. 

**Abstract (ZH)**: 在可解释的约束求解（XCS）中，通常需要从一组不满足的约束中提取一个最小不可满足子集（MUS），这有助于向用户解释为何约束规格不具有解。对于高度对称的问题，查找MUS可能会非常耗时，因为需要考虑许多约束组合。在传统的满足性问题求解背景下，对称性已经被广泛研究，且存在有效的在搜索过程中检测和利用对称性的方法。然而，在寻找不满足约束程序的MUS的背景下，对称性研究相对较少。本文借鉴现有对称性处理技术，将已知的MUS计算方法进行调整，以利用规格中的对称性，从而加快整体计算时间。我们的结果表明，在对称问题上，与基准方法相比，调整后的算法具有显著的运行时间缩短效果。 

---
# ROMAS: A Role-Based Multi-Agent System for Database monitoring and Planning 

**Title (ZH)**: ROMAS：一种基于角色的多智能体系统，用于数据库监控与规划 

**Authors**: Yi Huang, Fangyin Cheng, Fan Zhou, Jiahui Li, Jian Gong, Hongjun Yang, Zhidong Fan, Caigao Jiang, Siqiao Xue, Faqiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.13520)  

**Abstract**: In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities in data analytics when integrated with Multi-Agent Systems (MAS). However, these systems often struggle with complex tasks that involve diverse functional requirements and intricate data processing challenges, necessitating customized solutions that lack broad applicability. Furthermore, current MAS fail to emulate essential human-like traits such as self-planning, self-monitoring, and collaborative work in dynamic environments, leading to inefficiencies and resource wastage. To address these limitations, we propose ROMAS, a novel Role-Based M ulti-A gent System designed to adapt to various scenarios while enabling low code development and one-click deployment. ROMAS has been effectively deployed in DB-GPT [Xue et al., 2023a, 2024b], a well-known project utilizing LLM-powered database analytics, showcasing its practical utility in real-world scenarios. By integrating role-based collaborative mechanisms for self-monitoring and self-planning, and leveraging existing MAS capabilities to enhance database interactions, ROMAS offers a more effective and versatile solution. Experimental evaluations of ROMAS demonstrate its superiority across multiple scenarios, highlighting its potential to advance the field of multi-agent data analytics. 

**Abstract (ZH)**: 近年来，综合多智能体系统（MAS）的大规模语言模型（LLMs）在数据analytics方面展示了显著的能力。然而，这些系统在处理涉及多样化功能需求和复杂数据处理挑战的复杂任务时经常遇到困难，这要求定制解决方案，但这些方案缺乏广泛适用性。此外，当前的MAS无法模拟诸如自我规划、自我监控和在动态环境中协同工作的关键人类特质，这导致了效率低下和资源浪费。为了解决这些局限性，我们提出了ROMAS，这是一种新型的角色基础多智能体系统，旨在适应各种场景，同时支持低代码开发和一键部署。ROMAS已经在DB-GPT [Xue et al., 2023a, 2024b]项目中得到有效部署，该项目利用了LLM驱动的数据库分析技术，展示了其在实际场景中的实用价值。通过整合基于角色的协作机制进行自我监控和自我规划，并利用现有的MAS能力来增强与数据仓库的交互，ROMAS提供了更有效和多功能的解决方案。ROMAS的实验评估显示，其在多种场景下具备优势，突显了其在多智能体数据分析领域的潜在研究推进作用。 

---
# GUI Agents: A Survey 

**Title (ZH)**: GUI代理：综述 

**Authors**: Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, Xintong Li, Jing Shi, Hongjie Chen, Viet Dac Lai, Zhouhang Xie, Sungchul Kim, Ruiyi Zhang, Tong Yu, Mehrab Tanjim, Nesreen K. Ahmed, Puneet Mathur, Seunghyun Yoon, Lina Yao, Branislav Kveton, Thien Huu Nguyen, Trung Bui, Tianyi Zhou, Ryan A. Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2412.13501)  

**Abstract**: Graphical User Interface (GUI) agents, powered by Large Foundation Models, have emerged as a transformative approach to automating human-computer interaction. These agents autonomously interact with digital systems or software applications via GUIs, emulating human actions such as clicking, typing, and navigating visual elements across diverse platforms. Motivated by the growing interest and fundamental importance of GUI agents, we provide a comprehensive survey that categorizes their benchmarks, evaluation metrics, architectures, and training methods. We propose a unified framework that delineates their perception, reasoning, planning, and acting capabilities. Furthermore, we identify important open challenges and discuss key future directions. Finally, this work serves as a basis for practitioners and researchers to gain an intuitive understanding of current progress, techniques, benchmarks, and critical open problems that remain to be addressed. 

**Abstract (ZH)**: 基于大型基础模型的图形用户界面（GUI）代理已经作为一种变革性的方法，用于自动化人机交互而逐渐崭露头角。这些代理能够自主地通过GUI与数字系统或软件应用程序进行交互，模拟人类操作，如点击、输入和导航视觉元素，跨越多种平台。鉴于GUI代理日益增长的兴趣和基础重要性，我们提供了一篇全面的综述，对它们的基准测试、评估指标、架构和训练方法进行了分类。我们提出了一个统一框架，界定了这些代理的感知、推理、规划和执行能力。此外，我们指出了重要的开放挑战，并讨论了关键的未来发展方向。最后，本文为从业者和研究人员提供了一种基础，使他们能够直观地了解当前的进展、技术、基准测试以及仍需解决的关键开放问题。 

---
# Analysis of Higher-Order Ising Hamiltonians 

**Title (ZH)**: 高阶伊辛哈ミタング分析 

**Authors**: Yunuo Cen, Zhiwei Zhang, Zixuan Wang, Yimin Wang, Xuanyao Fong  

**Link**: [PDF](https://arxiv.org/pdf/2412.13489)  

**Abstract**: It is challenging to scale Ising machines for industrial-level problems due to algorithm or hardware limitations. Although higher-order Ising models provide a more compact encoding, they are, however, hard to physically implement. This work proposes a theoretical framework of a higher-order Ising simulator, IsingSim. The Ising spins and gradients in IsingSim are decoupled and self-customizable. We significantly accelerate the simulation speed via a bidirectional approach for differentiating the hyperedge functions. Our proof-of-concept implementation verifies the theoretical framework by simulating the Ising spins with exact and approximate gradients. Experiment results show that our novel framework can be a useful tool for providing design guidelines for higher-order Ising machines. 

**Abstract (ZH)**: 由于算法或硬件限制，扩展Ising机器以解决工业级问题具有挑战性。尽管高阶Ising模型可以提供更紧凑的编码，但它们在物理实现上却非常困难。本工作提出了一种高阶Ising模拟器（IsingSim）的理论框架。在IsingSim中，Ising自旋和梯度是解耦且自我可定制的。我们通过双向方法对超边函数进行微分，显著加快了模拟速度。我们的概念验证实现通过精确和近似梯度模拟Ising自旋，验证了理论框架。实验结果表明，我们提出的新框架可以成为设计高阶Ising机器的有用工具。 

---
# Gradual Vigilance and Interval Communication: Enhancing Value Alignment in Multi-Agent Debates 

**Title (ZH)**: 渐进警觉与区间通信：增强多智能体辩论中的价值对齐 

**Authors**: Rui Zou, Mengqi Wei, Jintian Feng, Qian Wan, Jianwen Sun, Sannyuya Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13471)  

**Abstract**: In recent years, large language models have shown exceptional performance in fulfilling diverse human needs. However, their training data can introduce harmful content, underscoring the necessity for robust value alignment. Mainstream methods, which depend on feedback learning and supervised training, are resource-intensive and may constrain the full potential of the models. Multi-Agent Debate (MAD) offers a more efficient and innovative solution by enabling the generation of reliable answers through agent interactions. To apply MAD to value alignment, we examine the relationship between the helpfulness and harmlessness of debate outcomes and individual responses, and propose a MAD based framework Gradual Vigilance and Interval Communication (GVIC). GVIC allows agents to assess risks with varying levels of vigilance and to exchange diverse information through interval communication. We theoretically prove that GVIC optimizes debate efficiency while reducing communication overhead. Experimental results demonstrate that GVIC consistently outperforms baseline methods across various tasks and datasets, particularly excelling in harmfulness mitigation and fraud prevention. Additionally, GVIC exhibits strong adaptability across different base model sizes, including both unaligned and aligned models, and across various task types. 

**Abstract (ZH)**: 近年来，大型语言模型在满足多样的人类需求方面展现了卓越的性能。然而，它们的训练数据可能引入有害内容，突显了实现稳健的价值对齐的必要性。主流方法依赖反馈学习和监督训练，这些方法资源密集且可能会限制模型的全部潜力。多代理辩论（MAD）提供了一种更高效和创新的解决方案，通过代理之间的相互作用生成可靠的答案。为将MAD应用于价值对齐，我们探讨了辩论结果和个体回应之间有益性和无害性之间的关系，并提出了一个基于MAD的框架——渐进警惕与间隔通信（GVIC）。GVIC允许代理根据不同的警惕水平评估风险，并通过间隔通信交换多样信息。我们从理论上证明了GVIC能够优化辩论效率并减少通信开销。实验结果表明，GVIC在各种任务和数据集中均优于基准方法，尤其是在减轻有害内容和预防欺诈方面表现尤为出色。此外，GVIC在不同基数模型大小（包括未对齐和已对齐模型）以及不同任务类型方面表现出强大的适应性。 

---
# Generating Diverse Hypotheses for Inductive Reasoning 

**Title (ZH)**: 生成多元假设以进行归纳推理 

**Authors**: Kang-il Lee, Hyukhun Koh, Dongryeol Lee, Seunghyun Yoon, Minsung Kim, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2412.13422)  

**Abstract**: Inductive reasoning - the process of inferring general rules from a small number of observations - is a fundamental aspect of human intelligence. Recent works suggest that large language models (LLMs) can engage in inductive reasoning by sampling multiple hypotheses about the rules and selecting the one that best explains the observations. However, due to the IID sampling, semantically redundant hypotheses are frequently generated, leading to significant wastage of compute. In this paper, we 1) demonstrate that increasing the temperature to enhance the diversity is limited due to text degeneration issue, and 2) propose a novel method to improve the diversity while maintaining text quality. We first analyze the effect of increasing the temperature parameter, which is regarded as the LLM's diversity control, on IID hypotheses. Our analysis shows that as temperature rises, diversity and accuracy of hypotheses increase up to a certain point, but this trend saturates due to text degeneration. To generate hypotheses that are more semantically diverse and of higher quality, we propose a novel approach inspired by human inductive reasoning, which we call Mixture of Concepts (MoC). When applied to several inductive reasoning benchmarks, MoC demonstrated significant performance improvements compared to standard IID sampling and other approaches. 

**Abstract (ZH)**: 归纳推理——从少量观察中推断一般规则的过程——是人类智能的一个基本方面。近期研究表明，大型语言模型（LLMs）可以通过采样多个关于规则的假设，并选择最能解释观察结果的那个假设来进行归纳推理。然而，由于采用独立同分布（IID）采样，频繁生成语义冗余的假设，导致计算资源的大量浪费。本文中，我们1）证明了提高温度以增强多样性受到文本退化问题的限制，2）提出了一种新颖的方法，在保持文本质量的同时提高多样性。我们首先分析了提高温度参数（被视为LLM的多样性控制）对IID假设的影响。我们的分析表明，随着温度的升高，假设的多样性和准确性在一定范围内增加，但由于文本退化，这一趋势会饱和。为了生成更语义多样且质量更高的假设，我们提出了一种新颖的方法，这种方法受到人类归纳推理的启发，我们称之为概念混合（MoC）。当应用于几个归纳推理基准时，MoC相较于标准的IID采样和其他方法，表现出显著的性能提升。 

---
# Multiple Mean-Payoff Optimization under Local Stability Constraints 

**Title (ZH)**: 局部稳定约束下的多个支付优化 

**Authors**: David Klaška, Antonín Kučera, Vojtěch Kůr, Vít Musil, Vojtěch Řehák  

**Link**: [PDF](https://arxiv.org/pdf/2412.13369)  

**Abstract**: The long-run average payoff per transition (mean payoff) is the main tool for specifying the performance and dependability properties of discrete systems. The problem of constructing a controller (strategy) simultaneously optimizing several mean payoffs has been deeply studied for stochastic and game-theoretic models. One common issue of the constructed controllers is the instability of the mean payoffs, measured by the deviations of the average rewards per transition computed in a finite "window" sliding along a run. Unfortunately, the problem of simultaneously optimizing the mean payoffs under local stability constraints is computationally hard, and the existing works do not provide a practically usable algorithm even for non-stochastic models such as two-player games. In this paper, we design and evaluate the first efficient and scalable solution to this problem applicable to Markov decision processes. 

**Abstract (ZH)**: 离散系统的性能和可靠属性主要通过每转移长期平均收益（平均收益）来描述。对于随机性和博弈论模型，同时优化多个平均收益的控制器（策略）问题已被深入研究。所构造的控制器的一个常见问题是平均收益的不稳定性，这是通过在运行过程中滑动的有限“窗口”计算的每转移平均奖励的偏差来衡量的。不幸的是，在局部稳定性约束下同时优化平均收益的问题具有很高的计算难度，现有工作甚至对诸如二人博弈等非随机模型也未能提供一种实用的算法。本文设计并评估了首个适用于马尔科夫决策过程的有效且可扩展的解决方案。 

---
# Quantitative Predictive Monitoring and Control for Safe Human-Machine Interaction 

**Title (ZH)**: 定量预测监控与控制以确保人机互动安全 

**Authors**: Shuyang Dong, Meiyi Ma, Josephine Lamp, Sebastian Elbaum, Matthew B. Dwyer, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.13365)  

**Abstract**: There is a growing trend toward AI systems interacting with humans to revolutionize a range of application domains such as healthcare and transportation. However, unsafe human-machine interaction can lead to catastrophic failures. We propose a novel approach that predicts future states by accounting for the uncertainty of human interaction, monitors whether predictions satisfy or violate safety requirements, and adapts control actions based on the predictive monitoring results. Specifically, we develop a new quantitative predictive monitor based on Signal Temporal Logic with Uncertainty (STL-U) to compute a robustness degree interval, which indicates the extent to which a sequence of uncertain predictions satisfies or violates an STL-U requirement. We also develop a new loss function to guide the uncertainty calibration of Bayesian deep learning and a new adaptive control method, both of which leverage STL-U quantitative predictive monitoring results. We apply the proposed approach to two case studies: Type 1 Diabetes management and semi-autonomous driving. Experiments show that the proposed approach improves safety and effectiveness in both case studies. 

**Abstract (ZH)**: 随着人工智能系统与人类交互的趋势不断增长，这些系统正不断革新医疗保健和交通等领域。然而，不安全的人机交互可能导致灾难性失败。我们提出了一种新的方法，该方法通过考虑人类交互的不确定性来预测未来状态，监控预测是否满足或违反安全性要求，并根据预测监控结果调整控制动作。具体而言，我们开发了一种基于不确定性信号时序逻辑（STL-U）的新定量预测监测器，以计算满足或违反STL-U要求的程度区间。此外，我们还开发了一种新的损失函数，以指导贝叶斯深度学习中的不确定性校准，并开发了一种新的自适应控制方法，这两种方法都利用了STL-U定量预测监测的结果。我们将所提出的方法应用于两个案例研究：1型糖尿病管理和半自主驾驶。实验结果表明，所提出的方法在两个案例研究中均提高了安全性和有效性。 

---
# Predictive Probability Density Mapping for Search and Rescue Using An Agent-Based Approach with Sparse Data 

**Title (ZH)**: 基于代理方法的稀疏数据条件下预测概率密度映射在搜救中的应用 

**Authors**: Jan-Hendrik Ewers, David Anderson, Douglas Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2412.13317)  

**Abstract**: Predicting the location where a lost person could be found is crucial for search and rescue operations with limited resources. To improve the precision and efficiency of these predictions, simulated agents can be created to emulate the behavior of the lost person. Within this study, we introduce an innovative agent-based model designed to replicate diverse psychological profiles of lost persons, allowing these agents to navigate real-world landscapes while making decisions autonomously without the need for location-specific training. The probability distribution map depicting the potential location of the lost person emerges through a combination of Monte Carlo simulations and mobility-time-based sampling. Validation of the model is achieved using real-world Search and Rescue data to train a Gaussian Process model. This allows generalization of the data to sample initial starting points for the agents during validation. Comparative analysis with historical data showcases promising outcomes relative to alternative methods. This work introduces a flexible agent that can be employed in search and rescue operations, offering adaptability across various geographical locations. 

**Abstract (ZH)**: 在资源有限的搜救行动中，准确预测失踪人员可能出现的地点至关重要。为了提高预测的精度和效率，可以通过创建模拟代理来模仿失踪人员的行为。在本研究中，我们提出了一种创新的基于代理的模型，该模型能够复制不同心理特征的失踪人员的行为，使这些代理可以在真实世界环境中自主导航并做出决策，无需特定地点的训练。通过结合蒙特卡洛模拟和基于移动时间的采样，生成描述失踪人员潜在位置的概率分布图。通过使用实际的搜救数据训练高斯过程模型来验证模型，从而实现数据的一般化，以便在验证过程中为代理提供初始起始点。与历史数据的比较分析展示了该方法相对于其他方法具有有希望的结果。这项工作引入了一种灵活的代理，可以在各种地理区域内应用于搜救行动。 

---
# SafeDrive: Knowledge- and Data-Driven Risk-Sensitive Decision-Making for Autonomous Vehicles with Large Language Models 

**Title (ZH)**: SafeDrive：基于知识和数据的风险敏感决策方法在大型语言模型驱动的自主车辆中的应用 

**Authors**: Zhiyuan Zhou, Heye Huang, Boqi Li, Shiyue Zhao, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13238)  

**Abstract**: Recent advancements in autonomous vehicles (AVs) use Large Language Models (LLMs) to perform well in normal driving scenarios. However, ensuring safety in dynamic, high-risk environments and managing safety-critical long-tail events remain significant challenges. To address these issues, we propose SafeDrive, a knowledge- and data-driven risk-sensitive decision-making framework to enhance AV safety and adaptability. The proposed framework introduces a modular system comprising: (1) a Risk Module for quantifying multi-factor coupled risks involving driver, vehicle, and road interactions; (2) a Memory Module for storing and retrieving typical scenarios to improve adaptability; (3) a LLM-powered Reasoning Module for context-aware safety decision-making; and (4) a Reflection Module for refining decisions through iterative learning. By integrating knowledge-driven insights with adaptive learning mechanisms, the framework ensures robust decision-making under uncertain conditions. Extensive evaluations on real-world traffic datasets, including highways (HighD), intersections (InD), and roundabouts (RounD), validate the framework's ability to enhance decision-making safety (achieving a 100% safety rate), replicate human-like driving behaviors (with decision alignment exceeding 85%), and adapt effectively to unpredictable scenarios. SafeDrive establishes a novel paradigm for integrating knowledge- and data-driven methods, highlighting significant potential to improve safety and adaptability of autonomous driving in high-risk traffic scenarios. 

**Abstract (ZH)**: 近年来，自主车辆（AVs）的进步利用大规模语言模型（LLMs）在常规驾驶场景中表现出色。然而，在动态、高风险环境中确保安全并管理安全性关键的长尾事件仍然是重大挑战。为应对这些问题，我们提出SafeDrive，一个以知识和数据为基础的风险敏感决策框架，旨在增强AV的安全性和适应性。该框架引入了一个模块化的系统，包括以下四个模块：（1）风险模块，用于量化涉及驾驶员、车辆和道路交互的多因素耦合风险；（2）记忆模块，用于存储和检索典型场景以提高适应性；（3）使用LLM的支持推理模块，实现基于上下文的安全决策；以及（4）反思模块，通过迭代学习对决策进行完善。通过整合知识驱动的见解与自适应学习机制，该框架确保在不确定条件下做出稳健的决策。通过在真实世界交通数据集（包括高速公路HighD、交叉口InD和环岛RounD）上的广泛评估，证明了该框架提高决策安全性（实现100%的安全率）、复制人类驾驶行为（决策一致性超过85%）以及有效适应不可预测场景的能力。SafeDrive建立了一种新的集成知识和数据驱动方法的范式，突显了在高风险交通场景中提高自主驾驶的安全性和适应性的巨大潜力。 

---
# Logic-Constrained Shortest Paths for Flight Planning 

**Title (ZH)**: 逻辑约束最短路径在飞行航路规划中的应用 

**Authors**: Ricardo Euler, Pedro Maristany de las Casas, Ralf Borndörfer  

**Link**: [PDF](https://arxiv.org/pdf/2412.13235)  

**Abstract**: The Logic-Constrained Shortest Path Problem (LCSP) combines a one-to-one shortest path problem with satisfiability constraints imposed on the routing graph. This setting arises in flight planning, where air traffic control (ATC) authorities are enforcing a set of traffic flow restrictions (TFRs) on aircraft routes in order to increase safety and throughput. We propose a new branch and bound-based algorithm for the LCSP.
The resulting algorithm has three main degrees of freedom: the node selection rule, the branching rule and the conflict. While node selection and branching rules have been long studied in the MIP and SAT communities, most of them cannot be applied out of the box for the LCSP. We review the existing literature and develop tailored variants of the most prominent rules. The conflict, the set of variables to which the branching rule is applied, is unique to the LCSP. We analyze its theoretical impact on the B&B algorithm.
In the second part of the paper, we show how to model the Flight Planning Problem with TFRs as an LCSP and solve it using the branch and bound algorithm. We demonstrate the algorithm's efficiency on a dataset consisting of a global flight graph and a set of around 20000 real TFRs obtained from our industry partner Lufthansa Systems GmbH. We make this dataset publicly available. Finally, we conduct an empirical in-depth analysis of node selection rules, branching rules and conflicts. Carefully choosing an appropriate combination yields an improvement of an order of magnitude compared to an uninformed choice. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的内容：

逻辑约束最短路径问题（LCSP）将单一最短路径问题与施加在路由图上的可满足性约束结合在一起。这种设置在飞行计划中出现，空中交通管制（ATC）当局对一组交通流量限制（TFRs）施加约束，以提高安全性并增加流量。我们提出了一种基于分支定界的新算法来解决LCSP。

该算法有三个主要的自由度：节点选择规则、分支规则和冲突。尽管节点选择和分支规则在MIP和SAT社区中已有长时间的研究，但大多数规则不能直接应用于LCSP。我们回顾了现有文献，开发了针对这些规则的定制变体。在LCSP中，冲突是唯一的特点，即分支规则被应用到的一组变量。我们分析了其对分支定界算法的理论影响。

在论文的第二部分，我们展示了如何将带有TFR的飞行计划问题建模为LCSP，并使用分支定界算法求解。我们使用由行业协会Lufthansa Systems GmbH提供的包含全球飞行图和约20000个实际TFR的数据集，证明了该算法的高效性。我们已将此数据集公开。最后，我们进行了详细的实证分析，研究节点选择规则、分支规则和冲突对性能的影响。合理选择合适的组合，相比随机选择可以显著提高10倍以上的性能。 

---
# Learning from Massive Human Videos for Universal Humanoid Pose Control 

**Title (ZH)**: 大规模人类视频用于通用类人姿态控制学习 

**Authors**: Jiageng Mao, Siheng Zhao, Siqi Song, Tianheng Shi, Junjie Ye, Mingtong Zhang, Haoran Geng, Jitendra Malik, Vitor Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14172)  

**Abstract**: Scalable learning of humanoid robots is crucial for their deployment in real-world applications. While traditional approaches primarily rely on reinforcement learning or teleoperation to achieve whole-body control, they are often limited by the diversity of simulated environments and the high costs of demonstration collection. In contrast, human videos are ubiquitous and present an untapped source of semantic and motion information that could significantly enhance the generalization capabilities of humanoid robots. This paper introduces Humanoid-X, a large-scale dataset of over 20 million humanoid robot poses with corresponding text-based motion descriptions, designed to leverage this abundant data. Humanoid-X is curated through a comprehensive pipeline: data mining from the Internet, video caption generation, motion retargeting of humans to humanoid robots, and policy learning for real-world deployment. With Humanoid-X, we further train a large humanoid model, UH-1, which takes text instructions as input and outputs corresponding actions to control a humanoid robot. Extensive simulated and real-world experiments validate that our scalable training approach leads to superior generalization in text-based humanoid control, marking a significant step toward adaptable, real-world-ready humanoid robots. 

**Abstract (ZH)**: 人性化机器人在实际应用中的规模化学习至关重要。虽然传统的方法主要依赖于强化学习或遥控操作来实现全身控制，但这些方法往往受限于模拟环境的多样性以及示范收集的高成本。相比之下，人类视频资源丰富且尚未充分利用，这些视频中包含了大量的语义和动作信息，能够显著增强人性化机器人的泛化能力。本文介绍了人体机器人-X（Humanoid-X），这是一个包含超过2000万个姿态样本及其对应的文本动作描述的大规模数据集。Humanoid-X 是通过一个全面的流水线进行整理：从互联网中抓取数据、生成视频字幕、将人类动作重新映射到人性化机器人、以及进行策略学习以实现实际应用。借助Humanoid-X，我们进一步训练了一个大规模的人形模型UH-1，该模型通过输入文本指令来生成相应的动作以控制人性化机器人。广泛的模拟和现实世界的实验验证了我们提出的规模化训练方法在基于文本的人形控制方面具有卓越的泛化能力，标志着实现可适应、实战就绪的机器人的重要一步。 

---
# E-CAR: Efficient Continuous Autoregressive Image Generation via Multistage Modeling 

**Title (ZH)**: E-CAR：基于多阶段建模的高效连续自回归图像生成 

**Authors**: Zhihang Yuan, Yuzhang Shang, Hanling Zhang, Tongcheng Fang, Rui Xie, Bingxin Xu, Yan Yan, Shengen Yan, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14170)  

**Abstract**: Recent advances in autoregressive (AR) models with continuous tokens for image generation show promising results by eliminating the need for discrete tokenization. However, these models face efficiency challenges due to their sequential token generation nature and reliance on computationally intensive diffusion-based sampling. We present ECAR (Efficient Continuous Auto-Regressive Image Generation via Multistage Modeling), an approach that addresses these limitations through two intertwined innovations: (1) a stage-wise continuous token generation strategy that reduces computational complexity and provides progressively refined token maps as hierarchical conditions, and (2) a multistage flow-based distribution modeling method that transforms only partial-denoised distributions at each stage comparing to complete denoising in normal diffusion models. Holistically, ECAR operates by generating tokens at increasing resolutions while simultaneously denoising the image at each stage. This design not only reduces token-to-image transformation cost by a factor of the stage number but also enables parallel processing at the token level. Our approach not only enhances computational efficiency but also aligns naturally with image generation principles by operating in continuous token space and following a hierarchical generation process from coarse to fine details. Experimental results demonstrate that ECAR achieves comparable image quality to DiT Peebles & Xie [2023] while requiring 10$\times$ FLOPs reduction and 5$\times$ speedup to generate a 256$\times$256 image. 

**Abstract (ZH)**: 近年来，连续标记自回归（AR）模型在图像生成中的应用取得了 promising 的进展，通过消除离散标记化的需求。然而，这些模型由于其序列化的标记生成特性和对计算密集型扩散采样方法的依赖而面临效率挑战。我们提出了 ECAR（高效连续自回归图像生成多阶段建模），该方法通过两种相互交织的创新点来解决这些限制：（1）按阶段生成的连续标记策略，减少了计算复杂性，并逐步提供分层条件的标记映射；（2）阶段流基于分布建模方法，与正常扩散模型相比，在每个阶段仅变换部分未去噪的分布，而不是完全去噪。总体而言，ECAR 通过在每个阶段同时生成标记并去噪图像来运行，这一设计不仅将标记到图像的转换成本减少了阶段数量倍数，还允许在标记级别上并行处理。我们的方法不仅提高了计算效率，而且通过在连续标记空间中运行并遵循从粗到细的分层生成过程，自然地与图像生成原则相一致。实验结果表明，ECAR 在生成 256×256 图像时，与 DiT Peebles & Xie [2023] 达到了相当的图像质量，同时仅需 10 倍的 FLOPs 减少和 5 倍的速度提升。 

---
# VideoDPO: Omni-Preference Alignment for Video Diffusion Generation 

**Title (ZH)**: VideoDPO：视频扩散生成的整体偏好对齐 

**Authors**: Runtao Liu, Haoyu Wu, Zheng Ziqiang, Chen Wei, Yingqing He, Renjie Pi, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14167)  

**Abstract**: Recent progress in generative diffusion models has greatly advanced text-to-video generation. While text-to-video models trained on large-scale, diverse datasets can produce varied outputs, these generations often deviate from user preferences, highlighting the need for preference alignment on pre-trained models. Although Direct Preference Optimization (DPO) has demonstrated significant improvements in language and image generation, we pioneer its adaptation to video diffusion models and propose a VideoDPO pipeline by making several key adjustments. Unlike previous image alignment methods that focus solely on either (i) visual quality or (ii) semantic alignment between text and videos, we comprehensively consider both dimensions and construct a preference score accordingly, which we term the OmniScore. We design a pipeline to automatically collect preference pair data based on the proposed OmniScore and discover that re-weighting these pairs based on the score significantly impacts overall preference alignment. Our experiments demonstrate substantial improvements in both visual quality and semantic alignment, ensuring that no preference aspect is neglected. Code and data will be shared at this https URL. 

**Abstract (ZH)**: 近年来，生成扩散模型的进展显著推动了文本到视频的生成。虽然在大规模多样数据集上训练的文本到视频模型能够生成多样的输出，但这些生成往往偏离用户的偏好，突显了对预训练模型进行偏好对齐的需求。尽管直接偏好优化（DPO）已在语言和图像生成中显示出显著改善，但我们首次将其适应到视频扩散模型，并通过几个关键调整提出了一个VideoDPO管道。与以往仅专注于视觉质量或文本与视频语义对齐的图像对齐方法不同，我们综合考虑了这两个维度，并根据提出的综合评分构建了一个评分系统，我们称之为Omniscore。我们设计了一个管道自动收集基于Omniscore的偏好对数据，并发现根据评分重新加权这些对显著影响了整体偏好对齐。我们的实验表明，在视觉质量和语义对齐方面都取得了显著改进，确保没有忽略任何偏好方面。代码和数据将在此处 https://链接共享。 

---
# AKiRa: Augmentation Kit on Rays for optical video generation 

**Title (ZH)**: AKiRa：基于光线的增强套件用于光学视频生成 

**Authors**: Xi Wang, Robin Courant, Marc Christie, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2412.14158)  

**Abstract**: Recent advances in text-conditioned video diffusion have greatly improved video quality. However, these methods offer limited or sometimes no control to users on camera aspects, including dynamic camera motion, zoom, distorted lens and focus shifts. These motion and optical aspects are crucial for adding controllability and cinematic elements to generation frameworks, ultimately resulting in visual content that draws focus, enhances mood, and guides emotions according to filmmakers' controls. In this paper, we aim to close the gap between controllable video generation and camera optics. To achieve this, we propose AKiRa (Augmentation Kit on Rays), a novel augmentation framework that builds and trains a camera adapter with a complex camera model over an existing video generation backbone. It enables fine-tuned control over camera motion as well as complex optical parameters (focal length, distortion, aperture) to achieve cinematic effects such as zoom, fisheye effect, and bokeh. Extensive experiments demonstrate AKiRa's effectiveness in combining and composing camera optics while outperforming all state-of-the-art methods. This work sets a new landmark in controlled and optically enhanced video generation, paving the way for future optical video generation methods. 

**Abstract (ZH)**: 近年来，基于文本条件的视频扩散技术取得了重大进展，极大地改善了视频质量。然而，这些方法在用户对摄影机方面的控制上相对有限，包括动态摄影机运动、变焦、镜头畸变和焦点偏移。这些运动和光学方面对于增加生成框架的可控性和电影元素至关重要，最终能够吸引观注力、增强情感氛围并根据电影制作人的控制引导情感。在本文中，我们旨在缩小可控视频生成与摄影机光学之间的差距。为此，我们提出了一种名为AKiRa（Augmentation Kit on Rays）的新型增强框架，该框架在现有视频生成骨干网络上构建并训练一个复杂的摄影机适配器。AKiRa允许用户对摄影机运动进行精细调整以及复杂的光学参数（焦距、畸变、光圈）进行控制，从而实现诸如变焦、鱼眼效果和景深等电影效果。详尽的实验表明，AKiRa在结合和组合摄影机光学方面表现出色，并在所有现有的先进方法中表现出优越性。这项工作开辟了受控和光学增强视频生成的新领域，为未来光学视频生成方法的发展奠定了基础。 

---
# GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking 

**Title (ZH)**: GLIDER: 可解释排名驱动的大型语言模型交互与决策评估 

**Authors**: Darshan Deshpande, Selvan Sunitha Ravi, Sky CH-Wang, Bartosz Mielczarek, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2412.14140)  

**Abstract**: The LLM-as-judge paradigm is increasingly being adopted for automated evaluation of model outputs. While LLM judges have shown promise on constrained evaluation tasks, closed source LLMs display critical shortcomings when deployed in real world applications due to challenges of fine grained metrics and explainability, while task specific evaluation models lack cross-domain generalization. We introduce GLIDER, a powerful 3B evaluator LLM that can score any text input and associated context on arbitrary user defined criteria. GLIDER shows higher Pearson's correlation than GPT-4o on FLASK and greatly outperforms prior evaluation models, achieving comparable performance to LLMs 17x its size. GLIDER supports fine-grained scoring, multilingual reasoning, span highlighting and was trained on 685 domains and 183 criteria. Extensive qualitative analysis shows that GLIDER scores are highly correlated with human judgments, with 91.3% human agreement. We have open-sourced GLIDER to facilitate future research. 

**Abstract (ZH)**: 基于大语言模型（LLM）的裁判模式正越来越多地被采用，以实现对模型输出的自动化评估。虽然LLM裁判在受限的评估任务中显示出一定的潜力，但由于精细化指标和可解释性方面的挑战，闭源的大语言模型在实际应用场景中暴露出关键的不足，而针对特定任务的评估模型则缺乏跨领域的泛化能力。我们引入了GLIDER，这是一种强大的3B级评估LLM，能够对任意文本输入及其上下文根据用户自定义的标准进行评分。GLIDER在FLASK上的皮尔逊相关系数优于GPT-4o，并在多项评估任务上显著超越了先前的评估模型，其性能甚至能够与大小为自身17倍的LLM相媲美。GLIDER支持精细化评分、多语言推理、区间高亮，并已被训练于685个领域和183个标准之上。广泛的定性分析表明，GLIDER的评分与人类判断高度一致，91.3%的工作由人类判断一致。我们已开源GLIDER，以促进未来的研究。 

---
# Design choices made by LLM-based test generators prevent them from finding bugs 

**Title (ZH)**: 基于LLM的测试生成器的设计选择阻止了它们发现漏洞 

**Authors**: Noble Saji Mathews, Meiyappan Nagappan  

**Link**: [PDF](https://arxiv.org/pdf/2412.14137)  

**Abstract**: There is an increasing amount of research and commercial tools for automated test case generation using Large Language Models (LLMs). This paper critically examines whether recent LLM-based test generation tools, such as Codium CoverAgent and CoverUp, can effectively find bugs or unintentionally validate faulty code. Considering bugs are only exposed by failing test cases, we explore the question: can these tools truly achieve the intended objectives of software testing when their test oracles are designed to pass? Using real human-written buggy code as input, we evaluate these tools, showing how LLM-generated tests can fail to detect bugs and, more alarmingly, how their design can worsen the situation by validating bugs in the generated test suite and rejecting bug-revealing tests. These findings raise important questions about the validity of the design behind LLM-based test generation tools and their impact on software quality and test suite reliability. 

**Abstract (ZH)**: 近年来，利用大规模语言模型（LLMs）进行自动化测试案例生成的研究和商业工具日益增多。本文批判性地探讨了诸如Codium CoverAgent和CoverUp等基于LLM的测试生成工具是否能够有效地发现缺陷，或者无意中验证了错误的代码。考虑到仅当测试失败时才能暴露缺陷，我们研究的问题是：当这些工具的测试或acles设计为通过时，它们能否真正实现软件测试的既定目标？利用真实的人工编写的错误代码作为输入，我们评估了这些工具，展示了LLM生成的测试如何未能检测到缺陷，并且更加令人担忧的是，它们的设计如何通过验证生成测试集中存在的缺陷，同时拒绝揭示缺陷的测试。这些发现对基于LLM的测试生成工具的设计有效性及其对软件质量和测试套件可靠性的影响提出了重要问题。 

---
# Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts 

**Title (ZH)**: 适应性概念瓶颈：在分布偏移下对基础模型的建模 

**Authors**: Jihye Choi, Jayaram Raghuram, Yixuan Li, Somesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2412.14097)  

**Abstract**: Advancements in foundation models (FMs) have led to a paradigm shift in machine learning. The rich, expressive feature representations from these pre-trained, large-scale FMs are leveraged for multiple downstream tasks, usually via lightweight fine-tuning of a shallow fully-connected network following the representation. However, the non-interpretable, black-box nature of this prediction pipeline can be a challenge, especially in critical domains such as healthcare, finance, and security. In this paper, we explore the potential of Concept Bottleneck Models (CBMs) for transforming complex, non-interpretable foundation models into interpretable decision-making pipelines using high-level concept vectors. Specifically, we focus on the test-time deployment of such an interpretable CBM pipeline "in the wild", where the input distribution often shifts from the original training distribution. We first identify the potential failure modes of such a pipeline under different types of distribution shifts. Then we propose an adaptive concept bottleneck framework to address these failure modes, that dynamically adapts the concept-vector bank and the prediction layer based solely on unlabeled data from the target domain, without access to the source (training) dataset. Empirical evaluations with various real-world distribution shifts show that our adaptation method produces concept-based interpretations better aligned with the test data and boosts post-deployment accuracy by up to 28%, aligning the CBM performance with that of non-interpretable classification. 

**Abstract (ZH)**: 基础模型（FMs）的进展已引发机器学习范式的转变。这些预训练的大规模FMs丰富的、表达性强的特征表示被用于多种下游任务，通常通过一个浅层全连接网络进行轻量级微调。然而，预测管道的非解释性和黑箱特性可能在诸如医疗保健、金融和安全等关键领域成为挑战。在本文中，我们探讨了概念瓶颈模型（CBMs）的潜力，通过使用高层概念向量将复杂的、非解释性的基础模型转变为解释性决策管道。具体而言，我们关注此类解释性CBM管道在现实环境中的测试时部署，其中输入分布往往从原始训练分布中发生变化。首先，我们识别了在不同类型的分布变化下此类管道的潜在失败模式。然后，我们提出了一种自适应概念瓶颈框架来解决这些失败模式，该框架仅基于目标领域的无标签数据动态调整概念向量库和预测层，无需访问源（训练）数据集。在各种实际分布变化的实证评估中表明，我们的自适应方法能够生成更好地对应测试数据的概念解释，并在部署后将准确率提高28%以上，使CBM性能与非解释性分类相当。 

---
# SEKE: Specialised Experts for Keyword Extraction 

**Title (ZH)**: SEKE: 专门领域的专家关键词提取 

**Authors**: Matej Martinc, Hanh Thi Hong Tran, Senja Pollak, Boshko Koloski  

**Link**: [PDF](https://arxiv.org/pdf/2412.14087)  

**Abstract**: Keyword extraction involves identifying the most descriptive words in a document, allowing automatic categorisation and summarisation of large quantities of diverse textual data. Relying on the insight that real-world keyword detection often requires handling of diverse content, we propose a novel supervised keyword extraction approach based on the mixture of experts (MoE) technique. MoE uses a learnable routing sub-network to direct information to specialised experts, allowing them to specialize in distinct regions of the input space. SEKE, a mixture of Specialised Experts for supervised Keyword Extraction, uses DeBERTa as the backbone model and builds on the MoE framework, where experts attend to each token, by integrating it with a recurrent neural network (RNN), to allow successful extraction even on smaller corpora, where specialisation is harder due to lack of training data. The MoE framework also provides an insight into inner workings of individual experts, enhancing the explainability of the approach. We benchmark SEKE on multiple English datasets, achieving state-of-the-art performance compared to strong supervised and unsupervised baselines. Our analysis reveals that depending on data size and type, experts specialize in distinct syntactic and semantic components, such as punctuation, stopwords, parts-of-speech, or named entities. Code is available at: this https URL 

**Abstract (ZH)**: 关键词提取涉及识别文档中最具描述性的词汇，从而实现对大量多样文本数据的自动分类和摘要。鉴于实际关键词检测通常需要处理多样化的内容，我们提出了一种基于专家混合（Mixture of Experts, MoE）技术的新颖监督关键词提取方法。MoE 使用可学习的路由子网络将信息导向专门的专家，使其能够在输入空间的不同区域进行专业化的处理。SEKE，一种基于专家混合的监督关键词提取方法，以 DeBERTa 作为骨干模型，并基于 MoE 框架，通过将递归神经网络（RNN）与专家相结合，使专家能够关注每个标记，从而在较小的数据集上实现成功的关键词提取，即使由于训练数据不足而导致专业化难度增加。MoE 框架还为每个专家内部的工作提供了见解，增强了该方法的可解释性。我们在多个英文数据集上对 SEKE 进行了基准测试，与强大的监督和非监督基线相比，取得了最先进的性能。我们的分析表明，根据数据大小和类型的不同，专家会专注于不同的句法和语义成分，如标点符号、停用词、词性或命名实体。代码可在以下链接处获得：this https URL 

---
# Future Research Avenues for Artificial Intelligence in Digital Gaming: An Exploratory Report 

**Title (ZH)**: 人工智能在数字游戏领域的未来研究方向：一项探索性报告 

**Authors**: Markus Dablander  

**Link**: [PDF](https://arxiv.org/pdf/2412.14085)  

**Abstract**: Video games are a natural and synergistic application domain for artificial intelligence (AI) systems, offering both the potential to enhance player experience and immersion, as well as providing valuable benchmarks and virtual environments to advance AI technologies in general. This report presents a high-level overview of five promising research pathways for applying state-of-the-art AI methods, particularly deep learning, to digital gaming within the context of the current research landscape. The objective of this work is to outline a curated, non-exhaustive list of encouraging research directions at the intersection of AI and video games that may serve to inspire more rigorous and comprehensive research efforts in the future. We discuss (i) investigating large language models as core engines for game agent modelling, (ii) using neural cellular automata for procedural game content generation, (iii) accelerating computationally expensive in-game simulations via deep surrogate modelling, (iv) leveraging self-supervised learning to obtain useful video game state embeddings, and (v) training generative models of interactive worlds using unlabelled video data. We also briefly address current technical challenges associated with the integration of advanced deep learning systems into video game development, and indicate key areas where further progress is likely to be beneficial. 

**Abstract (ZH)**: 视频游戏是一个天然且协同的应用领域，适合人工智能（AI）系统的应用，既可以增强玩家的体验和沉浸感，又可以提供有价值的标准和虚拟环境，以促进AI技术的整体进步。本报告简要概述了五条采用当前最新AI方法，特别是深度学习，在数字游戏领域应用的研究路径。本研究的目标是为AI与视频游戏交叉领域的探索性研究方向提供一个精心挑选的、非详尽的清单，这些研究方向可能在未来激发更严格和全面的研究努力。我们在以下五个方面进行了讨论：（i）研究大型语言模型作为游戏代理建模的核心引擎；（ii）使用神经细胞自动机进行程序化游戏内容生成；（iii）通过深度代理建模加速 computationally 费用较高的游戏内模拟；（iv）利用半监督学习获取有用的视频游戏状态嵌入；（v）使用未标记的视频数据训练交互世界的生成模型。我们还简要讨论了将先进的深度学习系统集成到游戏开发中目前面临的技術挑战，并指出了未来可能获益的关键领域。 

---
# Dialogue with the Machine and Dialogue with the Art World: Evaluating Generative AI for Culturally-Situated Creativity 

**Title (ZH)**: 机器对话与艺术世界对话：评估生成式AI在文化情境下创作中的作用 

**Authors**: Rida Qadri, Piotr Mirowski, Aroussiak Gabriellan, Farbod Mehr, Huma Gupta, Pamela Karimi, Remi Denton  

**Link**: [PDF](https://arxiv.org/pdf/2412.14077)  

**Abstract**: This paper proposes dialogue as a method for evaluating generative AI tools for culturally-situated creative practice, that recognizes the socially situated nature of art. Drawing on sociologist Howard Becker's concept of Art Worlds, this method expands the scope of traditional AI and creativity evaluations beyond benchmarks, user studies with crowd-workers, or focus groups conducted with artists. Our method involves two mutually informed dialogues: 1) 'dialogues with art worlds' placing artists in conversation with experts such as art historians, curators, and archivists, and 2)'dialogues with the machine,' facilitated through structured artist- and critic-led experimentation with state-of-the-art generative AI tools. We demonstrate the value of this method through a case study with artists and experts steeped in non-western art worlds, specifically the Persian Gulf. We trace how these dialogues help create culturally rich and situated forms of evaluation for representational possibilities of generative AI that mimic the reception of generative artwork in the broader art ecosystem. Putting artists in conversation with commentators also allow artists to shift their use of the tools to respond to their cultural and creative context. Our study can provide generative AI researchers an understanding of the complex dynamics of technology, human creativity and the socio-politics of art worlds, to build more inclusive machines for diverse art worlds. 

**Abstract (ZH)**: 本文提出了一种对话方法，用于评估具有文化背景的创意实践所需的生成式人工智能工具，这种方法认可艺术的社会背景性质。本文借鉴了社会学家霍华德·贝克尔（Howard Becker）提出的“艺术世界”概念，将评估范围从传统的基于基准的评估、用户众包研究或艺术家参与的重点小组研究中扩展出来。我们的方法包括两个相互支持的对话环节：1）“与艺术世界对话”，将艺术家置于与艺术史家、策展人和档案保管员等专家的对话中；2）“与机器对话”，通过结构化的艺术家和评论家主导的实验来促进对最新生成式人工智能工具的使用。我们通过一个关涉非西方艺术世界的案例研究展示了此方法的价值，具体而言是波斯湾地区的案例。我们探讨了这些对话如何帮助创建出具有文化丰富性和背景的生成式人工智能的表现潜力评价方式，并使之模拟更广泛艺术生态系统中对生成式艺术作品的接受度。将艺术家置于评论家的对话中，也使艺术家能够根据自身文化和创作的背景调整工具的使用方式。本研究可以为生成式人工智能研究者提供理解技术与人类创造力之间的复杂动态以及艺术世界中的社会政治因素的视角，从而构建更加包容的艺术世界的机器。 

---
# A Computationally Grounded Framework for Cognitive Attitudes (extended version) 

**Title (ZH)**: 基于计算的认知态度框架（扩展版） 

**Authors**: Tiago de Lima, Emiliano Lorini, Elise Perrotin, François Schwarzentruber  

**Link**: [PDF](https://arxiv.org/pdf/2412.14073)  

**Abstract**: We introduce a novel language for reasoning about agents' cognitive attitudes of both epistemic and motivational type. We interpret it by means of a computationally grounded semantics using belief bases. Our language includes five types of modal operators for implicit belief, complete attraction, complete repulsion, realistic attraction and realistic repulsion. We give an axiomatization and show that our operators are not mutually expressible and that they can be combined to represent a large variety of psychological concepts including ambivalence, indifference, being motivated, being demotivated and preference. We present a dynamic extension of the language that supports reasoning about the effects of belief change operations. Finally, we provide a succinct formulation of model checking for our languages and a PSPACE model checking algorithm relying on a reduction into TQBF. We present some experimental results for the implemented algorithm on computation time in a concrete example. 

**Abstract (ZH)**: 我们提出了一种新的语言，用于推理代理的认知态度，包括知识型和动机型两种类型。我们通过基于计算的语言接地语义学来解释这种语言，使用信念基的概念。我们的语言包括五种模态操作符，分别对应于隐含信念、完全吸引、完全排斥、实际吸引和实际排斥。我们给出了公理化解释，并证明了这些操作符彼此不可替代，并且可以通过组合来表示大量的心理概念，包括 ambivalence、indifference、被激励、被去激励和偏好。我们还提出了一种动态扩展，用于支持信念改变操作效果的推理。最后，我们为我们的语言提供了简洁的模型检查形式，并基于向 TQBF 的归约提供了 PSPACE 复杂度的模型检查算法。我们还展示了实现算法在具体示例中的计算时间实验结果。 

---
# Rango: Adaptive Retrieval-Augmented Proving for Automated Software Verification 

**Title (ZH)**: Rico: 自适应检索增强证明在自动化软件验证中的应用 

**Authors**: Kyle Thompson, Nuno Saavedra, Pedro Carrott, Kevin Fisher, Alex Sanchez-Stern, Yuriy Brun, João F. Ferreira, Sorin Lerner, Emily First  

**Link**: [PDF](https://arxiv.org/pdf/2412.14063)  

**Abstract**: Formal verification using proof assistants, such as Coq, enables the creation of high-quality software. However, the verification process requires significant expertise and manual effort to write proofs. Recent work has explored automating proof synthesis using machine learning and large language models (LLMs). This work has shown that identifying relevant premises, such as lemmas and definitions, can aid synthesis. We present Rango, a fully automated proof synthesis tool for Coq that automatically identifies relevant premises and also similar proofs from the current project and uses them during synthesis. Rango uses retrieval augmentation at every step of the proof to automatically determine which proofs and premises to include in the context of its fine-tuned LLM. In this way, Rango adapts to the project and to the evolving state of the proof. We create a new dataset, CoqStoq, of 2,226 open-source Coq projects and 196,929 theorems from GitHub, which includes both training data and a curated evaluation benchmark of well-maintained projects. On this benchmark, Rango synthesizes proofs for 32.0% of the theorems, which is 29% more theorems than the prior state-of-the-art tool Tactician. Our evaluation also shows that Rango adding relevant proofs to its context leads to a 47% increase in the number of theorems proven. 

**Abstract (ZH)**: 使用形式验证器（如Coq）进行的形式验证能够生成高质量的软件。然而，这一验证过程需要大量的专业知识和手工努力来书写证明。最近的研究探索了使用机器学习和大型语言模型（LLMs）来自动化证明合成的技术。这些研究显示了识别相关前提（如引理和定义）对于合成过程的帮助。我们提出了Rango，这是一种完全自动化的Coq证明合成工具，能够自动识别相关前提，并利用当前项目中相似的证明来进行合成。Rango在证明的每一步使用检索增强技术，自动确定在已微调的LLM上下文中应包括哪些证明和前提。通过这种方式，Rango能够适应不同的项目以及证明状态的变化。我们创建了一个新的数据集CoqStoq，其中包括来自GitHub的2,226个开源Coq项目和196,929个定理，数据集包含训练数据和精心筛选的评估基准，后者包括维护良好的项目。在这一基准上，Rango合成了32.0%的定理，比之前最先进的工具Tactician多了29个百分点的定理。我们的评估还显示，Rango添加相关证明到其上下文能够使证明的定理数量增加47%。 

---
# A Review of Multimodal Explainable Artificial Intelligence: Past, Present and Future 

**Title (ZH)**: 多模态可解释人工智能的综述：过去、现在和未来 

**Authors**: Shilin Sun, Wenbin An, Feng Tian, Fang Nan, Qidong Liu, Jun Liu, Nazaraf Shah, Ping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14056)  

**Abstract**: Artificial intelligence (AI) has rapidly developed through advancements in computational power and the growth of massive datasets. However, this progress has also heightened challenges in interpreting the "black-box" nature of AI models. To address these concerns, eXplainable AI (XAI) has emerged with a focus on transparency and interpretability to enhance human understanding and trust in AI decision-making processes. In the context of multimodal data fusion and complex reasoning scenarios, the proposal of Multimodal eXplainable AI (MXAI) integrates multiple modalities for prediction and explanation tasks. Meanwhile, the advent of Large Language Models (LLMs) has led to remarkable breakthroughs in natural language processing, yet their complexity has further exacerbated the issue of MXAI. To gain key insights into the development of MXAI methods and provide crucial guidance for building more transparent, fair, and trustworthy AI systems, we review the MXAI methods from a historical perspective and categorize them across four eras: traditional machine learning, deep learning, discriminative foundation models, and generative LLMs. We also review evaluation metrics and datasets used in MXAI research, concluding with a discussion of future challenges and directions. A project related to this review has been created at this https URL. 

**Abstract (ZH)**: 人工智能（AI）通过计算能力的提升和大规模数据集的增长迅速发展，同时也引发了对“黑盒”性质的AI模型解释性的关注。为了解决这些问题，可解释的人工智能（XAI）应运而生，其重点在于透明性和可解释性，以增强人类对AI决策过程的理解和信任。在多模态数据融合和复杂的推理场景中，多模态可解释人工智能（MXAI）的方法将多种模态整合用于预测和解释任务。与此同时，大型语言模型（LLMs）的发展在自然语言处理领域取得了显著突破，但其复杂性进一步加剧了MXAI的问题。为了深入了解MXAI方法的发展，并为构建更透明、公平和可信的AI系统提供建议，我们从历史角度回顾了MXAI方法，并将其分为四个阶段：传统机器学习、深度学习、判别型基础模型和生成型LLMs。我们还回顾了 MXAI 研究中使用的评估指标和数据集，并对未来挑战和方向进行了讨论。与本文综述相关的项目可以在以下网址找到：[此链接]。 

---
# Digestion Algorithm in Hierarchical Symbolic Forests: A Fast Text Normalization Algorithm and Semantic Parsing Framework for Specific Scenarios and Lightweight Deployment 

**Title (ZH)**: 层次符号森林中的消化算法：一种适用于特定场景的快速文本归一化算法及轻量级部署的语义解析框架 

**Authors**: Kevin You  

**Link**: [PDF](https://arxiv.org/pdf/2412.14054)  

**Abstract**: Text Normalization and Semantic Parsing have numerous applications in natural language processing, such as natural language programming, paraphrasing, data augmentation, constructing expert systems, text matching, and more. Despite the prominent achievements of deep learning in Large Language Models (LLMs), the interpretability of neural network architectures is still poor, which affects their credibility and hence limits the deployments of risk-sensitive scenarios. In certain scenario-specific domains with scarce data, rapidly obtaining a large number of supervised learning labels is challenging, and the workload of manually labeling data would be enormous. Catastrophic forgetting in neural networks further leads to low data utilization rates. In situations where swift responses are vital, the density of the model makes local deployment difficult and the response time long, which is not conducive to local applications of these fields. Inspired by the multiplication rule, a principle of combinatorial mathematics, and human thinking patterns, a multilayer framework along with its algorithm, the Digestion Algorithm in Hierarchical Symbolic Forests (DAHSF), is proposed to address these above issues, combining text normalization and semantic parsing workflows. The Chinese Scripting Language "Fire Bunny Intelligent Development Platform V2.0" is an important test and application of the technology discussed in this paper. DAHSF can run locally in scenario-specific domains on little datasets, with model size and memory usage optimized by at least two orders of magnitude, thus improving the execution speed, and possessing a promising optimization outlook. 

**Abstract (ZH)**: 文本规范化和语义解析在自然语言处理中有众多应用，例如自然语言编程、同义替换、数据增强、构建专家系统、文本匹配等。虽然大规模语言模型（LLMs）中的深度学习取得了显著成就，但神经网络架构的可解释性仍然较差，这影响了它们的可信度，从而限制了在风险敏感场景中的部署。在某些特定场景的数据稀缺领域，快速获取大量监督学习标签具有挑战性，手动标注数据的工作量极大。神经网络中灾难性遗忘进一步导致数据利用率低下。在需要迅速响应的情况下，模型的密集性使得局部部署困难且响应时间较长，不利于这些领域的本地应用。受乘法规则、组合数学原理以及人类思维模式的启发，提出了一种多层框架及其算法——层级符号森林中的消化算法（DAHSF），以解决上述问题，结合文本规范化和语义解析的工作流程。中文脚本语言“火兔智能开发平台V2.0”是本文讨论的技术的重要测试和应用。DAHSF 可在特定场景中使用少量数据进行本地运行，并通过至少两个数量级的模型大小和内存使用优化，提高了执行速度，显示出乐观的优化前景。 

---
# Gauss-Newton Dynamics for Neural Networks: A Riemannian Optimization Perspective 

**Title (ZH)**: 基于拉曼翰几何的神经网络高斯-牛顿动力学：一种里曼优化视角 

**Authors**: Semih Cayci  

**Link**: [PDF](https://arxiv.org/pdf/2412.14031)  

**Abstract**: We analyze the convergence of Gauss-Newton dynamics for training neural networks with smooth activation functions. In the underparameterized regime, the Gauss-Newton gradient flow induces a Riemannian gradient flow on a low-dimensional, smooth, embedded submanifold of the Euclidean output space. Using tools from Riemannian optimization, we prove \emph{last-iterate} convergence of the Riemannian gradient flow to the optimal in-class predictor at an \emph{exponential rate} that is independent of the conditioning of the Gram matrix, \emph{without} requiring explicit regularization. We further characterize the critical impacts of the neural network scaling factor and the initialization on the convergence behavior. In the overparameterized regime, we show that the Levenberg-Marquardt dynamics with an appropriately chosen damping factor yields robustness to ill-conditioned kernels, analogous to the underparameterized regime. These findings demonstrate the potential of Gauss-Newton methods for efficiently optimizing neural networks, particularly in ill-conditioned problems where kernel and Gram matrices have small singular values. 

**Abstract (ZH)**: 我们分析了具有光滑激活函数的神经网络训练中高斯-牛顿动力学的收敛性。在参数不足的区域，高斯-牛顿梯度流动诱导出一个低维的光滑嵌入流形上的黎曼梯度流动，该流形嵌入到欧几里得输出空间中。借助黎曼优化工具，我们证明了这种黎曼梯度流动的最终迭代收敛到最优类内预测器，并且这种收敛速度是指数级的，与格拉姆矩阵的条件数无关，且无需显式的正则化。我们进一步探讨了神经网络的尺度因子和初始化对收敛行为的关键影响。在参数过量的区域，我们展示了适当选择阻尼因子的莱文伯格-马夸德特动态方法具有类似于参数不足区域的鲁棒性，以应对条件不佳的核函数。这些发现表明高斯-牛顿方法在有效地优化神经网络方面具有潜在的应用价值，特别是在核函数和格拉姆矩阵具有小奇异值的病态问题中表现尤为突出。 

---
# Landscape of AI safety concerns -- A methodology to support safety assurance for AI-based autonomous systems 

**Title (ZH)**: 基于AI的自主系统安全保证的支持方法论——AI安全关切景观研究 

**Authors**: Ronald Schnitzer, Lennart Kilian, Simon Roessner, Konstantinos Theodorou, Sonja Zillner  

**Link**: [PDF](https://arxiv.org/pdf/2412.14020)  

**Abstract**: Artificial Intelligence (AI) has emerged as a key technology, driving advancements across a range of applications. Its integration into modern autonomous systems requires assuring safety. However, the challenge of assuring safety in systems that incorporate AI components is substantial. The lack of concrete specifications, and also the complexity of both the operational environment and the system itself, leads to various aspects of uncertain behavior and complicates the derivation of convincing evidence for system safety. Nonetheless, scholars proposed to thoroughly analyze and mitigate AI-specific insufficiencies, so-called AI safety concerns, which yields essential evidence supporting a convincing assurance case. In this paper, we build upon this idea and propose the so-called Landscape of AI Safety Concerns, a novel methodology designed to support the creation of safety assurance cases for AI-based systems by systematically demonstrating the absence of AI safety concerns. The methodology's application is illustrated through a case study involving a driverless regional train, demonstrating its practicality and effectiveness. 

**Abstract (ZH)**: 人工智能（AI）作为一项关键技术，正在推动各领域应用的进步。将AI整合到现代自主系统中需要确保其安全性。然而，在包含AI组件的系统中确保安全性的挑战巨大。缺乏具体的规范，以及操作环境和系统的复杂性，导致了各种不确定性行为，增加了提供令人信服的系统安全证据的难度。尽管如此，学者们提出要彻底分析并缓解特定于AI的不足，即所谓的AI安全问题，从而提供支持令人信服的安全保证案例的关键证据。本文基于这一理念，提出了一种新的方法论——AI安全问题全景图，旨在通过系统地证明AI安全问题的不存在，来支持基于AI系统的安全保证案例的创建。通过一个无人驾驶区域列车案例研究，展示了该方法论的实际可行性和有效性。 

---
# SurgSora: Decoupled RGBD-Flow Diffusion Model for Controllable Surgical Video Generation 

**Title (ZH)**: SurgSora：解耦的RGBD流扩散模型用于可控外科视频生成 

**Authors**: Tong Chen, Shuya Yang, Junyi Wang, Long Bai, Hongliang Ren, Luping Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14018)  

**Abstract**: Medical video generation has transformative potential for enhancing surgical understanding and pathology insights through precise and controllable visual representations. However, current models face limitations in controllability and authenticity. To bridge this gap, we propose SurgSora, a motion-controllable surgical video generation framework that uses a single input frame and user-controllable motion cues. SurgSora consists of three key modules: the Dual Semantic Injector (DSI), which extracts object-relevant RGB and depth features from the input frame and integrates them with segmentation cues to capture detailed spatial features of complex anatomical structures; the Decoupled Flow Mapper (DFM), which fuses optical flow with semantic-RGB-D features at multiple scales to enhance temporal understanding and object spatial dynamics; and the Trajectory Controller (TC), which allows users to specify motion directions and estimates sparse optical flow, guiding the video generation process. The fused features are used as conditions for a frozen Stable Diffusion model to produce realistic, temporally coherent surgical videos. Extensive evaluations demonstrate that SurgSora outperforms state-of-the-art methods in controllability and authenticity, showing its potential to advance surgical video generation for medical education, training, and research. 

**Abstract (ZH)**: 医学视频生成具有通过精确可控的视觉表现增强手术理解和病理洞察的变革潜力。然而，当前的模型在可控性和真实性方面存在局限性。为克服这些局限，我们提出了SurgSora，一种基于单帧输入和用户可控运动提示的运动可控手术视频生成框架。SurgSora 包含三个关键模块：双语义注入器（DSI），从输入帧中提取物体相关的RGB和深度特征，并将其与分割线索结合以捕捉复杂解剖结构的详细空间特征；解耦流动映射器（DFM），将光学流动与多尺度的语义-RGB-D特征融合，以增强时间理解和对象空间动态；轨迹控制器（TC），允许用户指定运动方向并估计稀疏光学流动，从而指导视频生成过程。融合的特征被用作冻结的Stable Diffusion模型的条件，以生成逼真且时间上连贯的手术视频。广泛评估表明，SurgSora 在可控性和真实性方面优于现有最先进的方法，展示了其在医学教育、培训和研究中促进手术视频生成的潜力。 

---
# Few-shot Steerable Alignment: Adapting Rewards and LLM Policies with Neural Processes 

**Title (ZH)**: 少样本可引导对齐：通过神经过程适应奖励和语言模型策略 

**Authors**: Katarzyna Kobalczyk, Claudio Fanconi, Hao Sun, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2412.13998)  

**Abstract**: As large language models (LLMs) become increasingly embedded in everyday applications, ensuring their alignment with the diverse preferences of individual users has become a critical challenge. Currently deployed approaches typically assume homogeneous user objectives and rely on single-objective fine-tuning. However, human preferences are inherently heterogeneous, influenced by various unobservable factors, leading to conflicting signals in preference data. Existing solutions addressing this diversity often require costly datasets labelled for specific objectives and involve training multiple reward models or LLM policies, which is computationally expensive and impractical. In this work, we present a novel framework for few-shot steerable alignment, where users' underlying preferences are inferred from a small sample of their choices. To achieve this, we extend the Bradley-Terry-Luce model to handle heterogeneous preferences with unobserved variability factors and propose its practical implementation for reward modelling and LLM fine-tuning. Thanks to our proposed approach of functional parameter-space conditioning, LLMs trained with our framework can be adapted to individual preferences at inference time, generating outputs over a continuum of behavioural modes. We empirically validate the effectiveness of methods, demonstrating their ability to capture and align with diverse human preferences in a data-efficient manner. Our code is made available at: this https URL. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）在日常生活中的应用越来越广泛，确保其与不同用户的多样化偏好保持一致已成为一个关键挑战。目前部署的方法通常假定用户目标的同质性，并依赖单一目标的微调。然而，人类偏好本质上是不一致的，受到各种不可观测因素的影响，导致偏好数据中存在矛盾的信号。现有解决这一多样性的方案通常需要昂贵且针对特定目标进行标注的数据集，并需要训练多个奖励模型或LLM策略，这在计算上非常昂贵且不切实际。在本文中，我们提出了一种新的若干示例导向的可控对齐框架，通过少量选择样本推断用户的潜在偏好。为此，我们扩展了Bradley-Terry-Luce模型，使其能够处理具有未观察到的变异性因素的异质偏好，并提出其在奖励建模和LLM微调中的实际实现方法。通过我们提出的功能参数空间条件方法，使用该框架训练的LLMs可以在推理时适应个体偏好，生成行为模式连续谱的输出。我们通过实验证明了方法的有效性，展示了它们能在数据有效利用的情况下捕捉和对齐多样化的人类偏好。我们的代码已开源于：this https URL。 

---
# Prompting Strategies for Enabling Large Language Models to Infer Causation from Correlation 

**Title (ZH)**: 促进大型语言模型从关联推断因果关系的提示策略 

**Authors**: Eleni Sgouritsa, Virginia Aglietti, Yee Whye Teh, Arnaud Doucet, Arthur Gretton, Silvia Chiappa  

**Link**: [PDF](https://arxiv.org/pdf/2412.13952)  

**Abstract**: The reasoning abilities of Large Language Models (LLMs) are attracting increasing attention. In this work, we focus on causal reasoning and address the task of establishing causal relationships based on correlation information, a highly challenging problem on which several LLMs have shown poor performance. We introduce a prompting strategy for this problem that breaks the original task into fixed subquestions, with each subquestion corresponding to one step of a formal causal discovery algorithm, the PC algorithm. The proposed prompting strategy, PC-SubQ, guides the LLM to follow these algorithmic steps, by sequentially prompting it with one subquestion at a time, augmenting the next subquestion's prompt with the answer to the previous one(s). We evaluate our approach on an existing causal benchmark, Corr2Cause: our experiments indicate a performance improvement across five LLMs when comparing PC-SubQ to baseline prompting strategies. Results are robust to causal query perturbations, when modifying the variable names or paraphrasing the expressions. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理能力正在引起越来越多的关注。本研究集中在因果推理上，并针对基于相关性信息建立因果关系这一高度挑战性的问题进行了研究，多个LLMs在这方面的表现都显得较为不足。我们提出了一种针对此问题的提示策略，将原始任务分解为固定的小问题，每个小问题对应形式化因果发现算法（如PC算法）的一个步骤。我们提出的提示策略PC-SubQ通过依次提示LLM解决每个小问题，并在接下来的小问题提示中增加前一个问题的答案，来引导LLM按照算法步骤进行推理。我们在现有的因果基准Corr2Cause上评估了我们的方法：实验表明，与基线提示策略相比，PC-SubQ在这五个LLM上的表现有所提高。当更改变量名称或重新表述表达方式时，此结果表现出较好的鲁棒性。 

---
# On Explaining Knowledge Distillation: Measuring and Visualising the Knowledge Transfer Process 

**Title (ZH)**: 关于解释知识蒸馏：测量与可视化知识转移过程 

**Authors**: Gereziher Adhane, Mohammad Mahdi Dehshibi, Dennis Vetter, David Masip, Gemma Roig  

**Link**: [PDF](https://arxiv.org/pdf/2412.13943)  

**Abstract**: Knowledge distillation (KD) remains challenging due to the opaque nature of the knowledge transfer process from a Teacher to a Student, making it difficult to address certain issues related to KD. To address this, we proposed UniCAM, a novel gradient-based visual explanation method, which effectively interprets the knowledge learned during KD. Our experimental results demonstrate that with the guidance of the Teacher's knowledge, the Student model becomes more efficient, learning more relevant features while discarding those that are not relevant. We refer to the features learned with the Teacher's guidance as distilled features and the features irrelevant to the task and ignored by the Student as residual features. Distilled features focus on key aspects of the input, such as textures and parts of objects. In contrast, residual features demonstrate more diffused attention, often targeting irrelevant areas, including the backgrounds of the target objects. In addition, we proposed two novel metrics: the feature similarity score (FSS) and the relevance score (RS), which quantify the relevance of the distilled knowledge. Experiments on the CIFAR10, ASIRRA, and Plant Disease datasets demonstrate that UniCAM and the two metrics offer valuable insights to explain the KD process. 

**Abstract (ZH)**: 知识蒸馏（KD）依然具有挑战性，因为从教师模型向学生模型的知识转移过程具有不透明性，这使得解决与KD相关的一些问题变得困难。为了解决这一问题，我们提出了一种新颖的基于梯度的视觉解释方法——UniCAM，该方法有效地解释了KD过程中学到的知识。实验结果表明，在教师模型知识的指导下，学生模型变得更加高效，能够学习到更相关的特征，同时舍弃不相关的特征。我们将通过教师模型引导学习到的特征称为蒸馏特征，而学生模型忽略的任务无关特征称为残余特征。蒸馏特征主要集中在输入的关键方面，如纹理和物体的组成部分；相比之下，残余特征则表现出更分散的注意，往往针对无关区域，包括目标物体的背景等。

此外，我们还提出了两种新的度量标准：特征相似性得分（FSS）和相关性得分（RS），这些度量标准可以量化蒸馏知识的相关性。在CIFAR10、ASIRRA和植物疾病数据集上的实验表明，UniCAM和这两个度量标准提供了解释KD过程的重要见解。 

---
# Spatio-Temporal Forecasting of PM2.5 via Spatial-Diffusion guided Encoder-Decoder Architecture 

**Title (ZH)**: 通过空间扩散引导的编码-解码架构进行PM2.5的空间-时间预测 

**Authors**: Malay Pandey, Vaishali Jain, Nimit Godhani, Sachchida Nand Tripathi, Piyush Rai  

**Link**: [PDF](https://arxiv.org/pdf/2412.13935)  

**Abstract**: In many problem settings that require spatio-temporal forecasting, the values in the time-series not only exhibit spatio-temporal correlations but are also influenced by spatial diffusion across locations. One such example is forecasting the concentration of fine particulate matter (PM2.5) in the atmosphere which is influenced by many complex factors, the most important ones being diffusion due to meteorological factors as well as transport across vast distances over a period of time. We present a novel Spatio-Temporal Graph Neural Network architecture, that specifically captures these dependencies to forecast the PM2.5 concentration. Our model is based on an encoder-decoder architecture where the encoder and decoder parts leverage gated recurrent units (GRU) augmented with a graph neural network (TransformerConv) to account for spatial diffusion. Our model can also be seen as a generalization of various existing models for time-series or spatio-temporal forecasting. We demonstrate the model's effectiveness on two real-world PM2.5 datasets: (1) data collected by us using a recently deployed network of low-cost PM$_{2.5}$ sensors from 511 locations spanning the entirety of the Indian state of Bihar over a period of one year, and (2) another publicly available dataset that covers severely polluted regions from China for a period of 4 years. Our experimental results show our model's impressive ability to account for both spatial as well as temporal dependencies precisely. 

**Abstract (ZH)**: 在需要进行空间-时间预测的许多问题设置中，时间序列中的值不仅表现出空间-时间相关性，还会受到不同位置之间的空间扩散的影响。例如，大气中细颗粒物（PM2.5）的浓度预测就是一个例子，这种浓度受到许多复杂因素的影响，其中最重要的因素是气象因素引起的扩散以及长距离运输。我们提出了一种新颖的空间-时间图神经网络架构，该架构特异性地捕捉这些依赖性以预测PM2.5浓度。我们的模型基于编码器-解码器架构，其中编码器和解码器部分利用门控循环单元（GRU）并结合图神经网络（TransformerConv）来考虑空间扩散。此外，我们的模型可以视为时间序列或空间-时间预测中各种现有模型的一般化。我们在两个实际的PM2.5数据集上展示了模型的有效性：(1) 我们使用最近部署的511个低成本PM2.5传感器网络收集的数据，覆盖整个印度 Bihar 状态一年的观测结果；(2) 另一个公开可用的数据集，涵盖了中国受严重污染的地区，时间段为四年。我们的实验结果表明，我们的模型在精确捕捉时空依赖性方面具有显著的能力。 

---
# Pipeline Analysis for Developing Instruct LLMs in Low-Resource Languages: A Case Study on Basque 

**Title (ZH)**: 低资源语言中开发指令性大规模语言模型的流水线分析：关于巴斯克语的案例研究 

**Authors**: Ander Corral, Ixak Sarasua, Xabier Saralegi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13922)  

**Abstract**: Large language models (LLMs) are typically optimized for resource-rich languages like English, exacerbating the gap between high-resource and underrepresented languages. This work presents a detailed analysis of strategies for developing a model capable of following instructions in a low-resource language, specifically Basque, by focusing on three key stages: pre-training, instruction tuning, and alignment with human preferences. Our findings demonstrate that continual pre-training with a high-quality Basque corpus of around 600 million words improves natural language understanding (NLU) of the foundational model by over 12 points. Moreover, instruction tuning and human preference alignment using automatically translated datasets proved highly effective, resulting in a 24-point improvement in instruction-following performance. The resulting models, Llama-eus-8B and Llama-eus-8B-instruct, establish a new state-of-the-art for Basque in the sub-10B parameter category. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通常针对资源丰富的语言（如英语）进行了优化，从而加剧了高资源语言与欠代表性语言之间的差距。本研究详细分析了开发能够在低资源语言中遵循指令模型的策略，特别聚焦于巴斯克语，重点关注三个关键阶段：预训练、指令调优和与人类偏好的对齐。我们的研究结果表明，使用包含大约6亿词的高质量巴斯克语语料库进行持续预训练，可以使基础模型的自然语言理解（NLU）提高超过12个百分点。此外，使用自动翻译的数据集进行指令调优和人类偏好对齐非常有效，使指令遵循性能提高了24个百分点。最终生成的模型，Llama-eus-8B和Llama-eus-8B-instruct，在参数量小于10B的子类别中建立了新的技术水平。 

---
# Energy-Efficient SLAM via Joint Design of Sensing, Communication, and Exploration Speed 

**Title (ZH)**: 通过联合设计感知、通信和探索速度实现能耗优化的SLAM 

**Authors**: Zidong Han, Ruibo Jin, Xiaoyang Li, Bingpeng Zhou, Qinyu Zhang, Yi Gong  

**Link**: [PDF](https://arxiv.org/pdf/2412.13912)  

**Abstract**: To support future spatial machine intelligence applications, lifelong simultaneous localization and mapping (SLAM) has drawn significant attentions. SLAM is usually realized based on various types of mobile robots performing simultaneous and continuous sensing and communication. This paper focuses on analyzing the energy efficiency of robot operation for lifelong SLAM by jointly considering sensing, communication and mechanical factors. The system model is built based on a robot equipped with a 2D light detection and ranging (LiDAR) and an odometry. The cloud point raw data as well as the odometry data are wirelessly transmitted to data center where real-time map reconstruction is realized based on an unsupervised deep learning based method. The sensing duration, transmit power, transmit duration and exploration speed are jointly optimized to minimize the energy consumption. Simulations and experiments demonstrate the performance of our proposed method. 

**Abstract (ZH)**: 为了支持未来的空间机器智能应用，生命周期同时定位与地图构建（Lifelong SLAM）受到了广泛关注。LIFSLAM通常基于各种类型的移动机器人实现，这些机器人能够同时进行持续的感知和通信。本文重点分析了在同时考虑感知、通信和机械因素的前提下，机器人操作在生命周期SLAM中的能效问题。我们基于配备二维激光测距仪（2D LiDAR）和里程计的机器人构建了一种系统模型。云点原始数据以及里程计数据通过无线方式传输至数据中心，在数据中心基于无监督的深度学习方法实时重建地图。我们联合优化了感知持续时间、传输功率、传输持续时间和探索速度，以最小化能源消耗。仿真和实验验证了我们提出方法的有效性。 

---
# Understanding and Analyzing Model Robustness and Knowledge-Transfer in Multilingual Neural Machine Translation using TX-Ray 

**Title (ZH)**: 使用TX-Ray 理解和分析多语种神经机器翻译中的模型稳健性和知识迁移 

**Authors**: Vageesh Saxena, Sharid Loáiciga, Nils Rethmeier  

**Link**: [PDF](https://arxiv.org/pdf/2412.13881)  

**Abstract**: Neural networks have demonstrated significant advancements in Neural Machine Translation (NMT) compared to conventional phrase-based approaches. However, Multilingual Neural Machine Translation (MNMT) in extremely low-resource settings remains underexplored. This research investigates how knowledge transfer across languages can enhance MNMT in such scenarios. Using the Tatoeba translation challenge dataset from Helsinki NLP, we perform English-German, English-French, and English-Spanish translations, leveraging minimal parallel data to establish cross-lingual mappings. Unlike conventional methods relying on extensive pre-training for specific language pairs, we pre-train our model on English-English translations, setting English as the source language for all tasks. The model is fine-tuned on target language pairs using joint multi-task and sequential transfer learning strategies. Our work addresses three key questions: (1) How can knowledge transfer across languages improve MNMT in extremely low-resource scenarios? (2) How does pruning neuron knowledge affect model generalization, robustness, and catastrophic forgetting? (3) How can TX-Ray interpret and quantify knowledge transfer in trained models? Evaluation using BLEU-4 scores demonstrates that sequential transfer learning outperforms baselines on a 40k parallel sentence corpus, showcasing its efficacy. However, pruning neuron knowledge degrades performance, increases catastrophic forgetting, and fails to improve robustness or generalization. Our findings provide valuable insights into the potential and limitations of knowledge transfer and pruning in MNMT for extremely low-resource settings. 

**Abstract (ZH)**: 与传统的短语基于方法相比，神经网络在神经机器翻译（NMT）方面取得了显著的进步。然而，在极度资源匮乏的多语言神经机器翻译（MNMT）场景中，这一领域的研究仍相对较少。本研究探讨了跨语言知识迁移如何在这些场景中提升MNMT。我们利用赫尔辛基NLP提供的Tatoeba翻译挑战数据集，进行英语-德语、英语-法语和英语-西班牙语的翻译，利用少量平行数据建立跨语言映射。与依赖于特定语言对大量预训练的传统方法不同，我们以英语-英语的平行翻译数据对模型进行预训练，将英语设为目标语言的所有任务的源语言。模型通过联合多任务和序列转移学习策略进行微调。我们的研究主要回答了三个关键问题：（1）跨语言知识迁移如何在极度资源匮乏的场景中提升MNMT？（2）剪枝神经元知识如何影响模型的泛化能力、鲁棒性和灾难性遗忘？（3）TX-Ray如何解释和定量分析训练模型中的知识迁移？使用BLEU-4分数的评估结果表明，序列转移学习在40,000个平行句子语料库上优于基线方法，展示了其有效性。然而，剪枝神经元知识降低了性能，增加了灾难性遗忘，并未提升模型的鲁棒性和泛化能力。我们的研究结果为极度资源匮乏场景中的多语言神经机器翻译中的知识迁移和剪枝潜力及其局限性提供了宝贵见解。 

---
# Crabs: Consuming Resrouce via Auto-generation for LLM-DoS Attack under Black-box Settings 

**Title (ZH)**: 螃蟹：在黑盒设置下通过自动生成消耗资源进行LLM-DOS攻击 

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.13879)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks. LLMs continue to be vulnerable to external threats, particularly Denial-of-Service (DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, prior works tend to focus on performing white-box attacks, overlooking black-box settings. In this work, we propose an automated algorithm designed for black-box LLMs, called Auto-Generation for LLM-DoS Attack (AutoDoS). AutoDoS introduces DoS Attack Tree and optimizes the prompt node coverage to enhance effectiveness under black-box conditions. Our method can bypass existing defense with enhanced stealthiness via semantic improvement of prompt nodes. Furthermore, we reveal that implanting Length Trojan in Basic DoS Prompt aids in achieving higher attack efficacy. Experimental results show that AutoDoS amplifies service response latency by over 250 $\times \uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our code is available at \url{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展现了卓越的性能。然而，LLMs仍易受外部威胁的影响，尤其是服务拒绝攻击（DoS攻击）。具体而言，LLM-DoS攻击旨在耗尽计算资源并阻止服务。然而，先前的研究主要关注白盒攻击，忽略了黑盒环境。在本文中，我们提出了一种针对黑盒LLMs的自动化算法，命名为Auto-Generation for LLM-DoS Attack（AutoDoS）。AutoDoS引入了DoS攻击树，并通过优化提示节点的覆盖率来增强在黑盒条件下的攻击效果。我们的方法可以通过提升提示节点的语义来提高隐藏性，从而绕过现有的防御措施。此外，我们发现，在基础DoS提示中植入长度特洛伊木马有助于提高攻击效果。实验结果表明，AutoDoS将服务响应延迟放大了超过250倍，导致在GPU利用率和内存使用方面产生了严重资源消耗。我们的代码可以在 \url{此处填写URL} 找到。 

---
# RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation 

**Title (ZH)**: RoboMIND：多体态智能规范数据基准——针对机器人操作的任务范围 

**Authors**: Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, Zhen Zhao, Guangyu Li, Zhao Jin, Lecheng Wang, Jilei Mao, Xinhua Wang, Shichao Fan, Ning Liu, Pei Ren, Qiang Zhang, Yaoxu Lyu, Mengzhen Liu, Jingyang He, Yulin Luo, Zeyu Gao, Chenxuan Li, Chenyang Gu, Yankai Fu, Di Wu, Xingyu Wang, Sixiang Chen, Zhenyu Wang, Pengju An, Siyuan Qian, Shanghang Zhang, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13877)  

**Abstract**: Developing robust and general-purpose robotic manipulation policies is a key goal in the field of robotics. To achieve effective generalization, it is essential to construct comprehensive datasets that encompass a large number of demonstration trajectories and diverse tasks. Unlike vision or language data that can be collected from the Internet, robotic datasets require detailed observations and manipulation actions, necessitating significant investment in hardware-software infrastructure and human labor. While existing works have focused on assembling various individual robot datasets, there remains a lack of a unified data collection standard and insufficient diversity in tasks, scenarios, and robot types. In this paper, we introduce RoboMIND (Multi-embodiment Intelligence Normative Data for Robot manipulation), featuring 55k real-world demonstration trajectories across 279 diverse tasks involving 61 different object classes. RoboMIND is collected through human teleoperation and encompasses comprehensive robotic-related information, including multi-view RGB-D images, proprioceptive robot state information, end effector details, and linguistic task descriptions. To ensure dataset consistency and reliability during policy learning, RoboMIND is built on a unified data collection platform and standardized protocol, covering four distinct robotic embodiments. We provide a thorough quantitative and qualitative analysis of RoboMIND across multiple dimensions, offering detailed insights into the diversity of our datasets. In our experiments, we conduct extensive real-world testing with four state-of-the-art imitation learning methods, demonstrating that training with RoboMIND data results in a high manipulation success rate and strong generalization. Our project is at this https URL. 

**Abstract (ZH)**: 开发稳健且通用的机器人操作策略是机器人学领域的关键目标。为了实现有效的泛化，构建包含大量演示轨迹和多样化任务的全面数据集至关重要。与可以从互联网收集的视觉或语言数据不同，机器人数据集需要详细的观察和操作动作，这需要在硬件和软件基础设施以及人力方面进行大量投资。尽管现有研究工作已经集中于汇集各种个体机器人数据集，但仍然缺乏统一的数据采集标准，并且在任务、场景和机器人类型方面存在不足。本文介绍了一种名为RoboMIND（多体态智能规范数据集）的机器人操作数据集，包含跨越279种不同任务的55000个真实世界的演示轨迹，涉及61种不同的对象类别。RoboMIND 通过人类遥控操作收集，并涵盖了广泛的机器人相关信息，包括多视角RGB-D图像、本体感受器的机器人状态信息、末端执行器的细节以及语言描述的任务定义。为了在策略学习过程中确保数据集的一致性和可靠性，RoboMIND 是基于统一的数据采集平台和标准化协议构建的，涵盖了四种不同的机器人体态。我们从多个维度对RoboMIND 进行了详细的定量和定性分析，提供了关于我们数据集多样性的详细洞见。在实验中，我们对四个最先进的模仿学习方法进行了广泛的实地测试，证明使用RoboMIND 数据进行训练能够实现高成功率操作和强泛化能力。我们的项目网页在此网址：[提供的网址] 

---
# SHAP scores fail pervasively even when Lipschitz succeeds 

**Title (ZH)**: SHAP评分甚至在Lipschitz成功时也普遍失效 

**Authors**: Olivier Letoffe, Xuanxiang Huang, Joao Marques-Silva  

**Link**: [PDF](https://arxiv.org/pdf/2412.13866)  

**Abstract**: The ubiquitous use of Shapley values in eXplainable AI (XAI) has been triggered by the tool SHAP, and as a result are commonly referred to as SHAP scores. Recent work devised examples of machine learning (ML) classifiers for which the computed SHAP scores are thoroughly unsatisfactory, by allowing human decision-makers to be misled. Nevertheless, such examples could be perceived as somewhat artificial, since the selected classes must be interpreted as numeric. Furthermore, it was unclear how general were the issues identified with SHAP scores. This paper answers these criticisms. First, the paper shows that for Boolean classifiers there are arbitrarily many examples for which the SHAP scores must be deemed unsatisfactory. Second, the paper shows that the issues with SHAP scores are also observed in the case of regression models. In addition, the paper studies the class of regression models that respect Lipschitz continuity, a measure of a function's rate of change that finds important recent uses in ML, including model robustness. Concretely, the paper shows that the issues with SHAP scores occur even for regression models that respect Lipschitz continuity. Finally, the paper shows that the same issues are guaranteed to exist for arbitrarily differentiable regression models. 

**Abstract (ZH)**: 在可解释人工智能（XAI）中无处不在地使用Shapley值主要得益于工具SHAP的推动，因此这些Shapley值通常被称为SHAP分数。最近有研究表明，对于某些机器学习（ML）分类器，计算出的SHAP分数会误导人类决策者，从而引发了对其可靠性的质疑。尽管如此，这些例子可能被认为有些人为，因为选择的类必须被解释为数值类。此外，关于SHAP分数的问题的普遍性尚不清楚。本文对此进行了反驳。首先，本文证明对于布尔分类器，存在任意多的实例，使得SHAP分数必须被认定为不满意。其次，本文证明了SHAP分数的问题在回归模型中也同样出现。此外，本文研究了一类尊重利普希茨连续性的回归模型，这是一种衡量函数变化率的指标，近年来在ML中具有重要的应用，包括模型鲁棒性。具体而言，本文证明了尊重利普希茨连续性的回归模型中也存在SHAP分数的问题。最后，本文证明了任意可微分的回归模型中也必定存在相同的问题。 

---
# From Expectation to Habit: Why Do Software Practitioners Adopt Fairness Toolkits? 

**Title (ZH)**: 从期望到习惯：软件实践者为何采用公平性工具包？ 

**Authors**: Gianmario Voria, Stefano Lambiase, Maria Concetta Schiavone, Gemma Catolino, Fabio Palomba  

**Link**: [PDF](https://arxiv.org/pdf/2412.13846)  

**Abstract**: As the adoption of machine learning (ML) systems continues to grow across industries, concerns about fairness and bias in these systems have taken center stage. Fairness toolkits, designed to mitigate bias in ML models, serve as critical tools for addressing these ethical concerns. However, their adoption in the context of software development remains underexplored, especially regarding the cognitive and behavioral factors driving their usage. As a deeper understanding of these factors could be pivotal in refining tool designs and promoting broader adoption, this study investigates the factors influencing the adoption of fairness toolkits from an individual perspective. Guided by the Unified Theory of Acceptance and Use of Technology (UTAUT2), we examined the factors shaping the intention to adopt and actual use of fairness toolkits. Specifically, we employed Partial Least Squares Structural Equation Modeling (PLS-SEM) to analyze data from a survey study involving practitioners in the software industry. Our findings reveal that performance expectancy and habit are the primary drivers of fairness toolkit adoption. These insights suggest that by emphasizing the effectiveness of these tools in mitigating bias and fostering habitual use, organizations can encourage wider adoption. Practical recommendations include improving toolkit usability, integrating bias mitigation processes into routine development workflows, and providing ongoing support to ensure professionals see clear benefits from regular use. 

**Abstract (ZH)**: 随着机器学习（ML）系统的应用在各个行业中不断扩展，人们对于这些系统中的公平性和偏见问题的关注也日益增加。为缓解这些偏见而设计的公平性工具包已经成为解决道德问题的关键工具。然而，这些工具包在软件开发中的应用仍处于初级阶段，特别是在认知和行为因素如何驱动其使用方面的情况仍需进一步探索。鉴于对这些因素的理解可能对工具包的设计改进及更广泛的采纳起到决定性作用，本研究从个体视角出发，探讨影响公平性工具包采纳的因素。基于统合接受与使用技术理论（UTAUT2），我们研究了影响采纳公平性工具包的意向及其实际使用行为的因素。具体而言，我们使用部分最小二乘结构方程建模（PLS-SEM）来分析一项针对软件行业从业者的研究数据。研究发现，绩效期望和习惯是公平性工具包采纳的主要驱动因素。这些发现表明，通过强调这些工具在减轻偏见方面的有效性和培养习惯性使用，组织可以鼓励更广泛地采纳这些工具。实用性的建议包括提高工具的易用性、将偏见缓解流程整合到常规开发流程中，并提供持续的支持，以确保专业人士从常规使用中获得明确的益处。 

---
# Do Language Models Understand Time? 

**Title (ZH)**: 语言模型理解时间的能力吗？ 

**Authors**: Xi Ding, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13845)  

**Abstract**: Large language models (LLMs) have revolutionized video-based computer vision applications, including action recognition, anomaly detection, and video summarization. Videos inherently pose unique challenges, combining spatial complexity with temporal dynamics that are absent in static images or textual data. Current approaches to video understanding with LLMs often rely on pretrained video encoders to extract spatiotemporal features and text encoders to capture semantic meaning. These representations are integrated within LLM frameworks, enabling multimodal reasoning across diverse video tasks. However, the critical question persists: Can LLMs truly understand the concept of time, and how effectively can they reason about temporal relationships in videos? This work critically examines the role of LLMs in video processing, with a specific focus on their temporal reasoning capabilities. We identify key limitations in the interaction between LLMs and pretrained encoders, revealing gaps in their ability to model long-term dependencies and abstract temporal concepts such as causality and event progression. Furthermore, we analyze challenges posed by existing video datasets, including biases, lack of temporal annotations, and domain-specific limitations that constrain the temporal understanding of LLMs. To address these gaps, we explore promising future directions, including the co-evolution of LLMs and encoders, the development of enriched datasets with explicit temporal labels, and innovative architectures for integrating spatial, temporal, and semantic reasoning. By addressing these challenges, we aim to advance the temporal comprehension of LLMs, unlocking their full potential in video analysis and beyond. 

**Abstract (ZH)**: 大语言模型（LLMs）已经彻底改变了基于视频的计算机视觉应用，包括动作识别、异常检测和视频摘要。视频本身带来了独特的挑战，结合了空间复杂性和时间动态性，这些特性在静态图像或文本数据中是不存在的。目前使用LLM进行视频理解的方法通常依赖于预训练的视频编码器提取时空特征，以及文本编码器来捕捉语义意义。这些表示被集成到LLM框架中，使它们能够在多种视频任务中进行多模态推理。然而，一个关键问题仍然存在：LLM能否真正理解时间的概念，并在视频中有效地推理时间关系？本研究对LLM在视频处理中的角色进行了批判性审查，特别关注它们的时间推理能力。我们指出了LLM与预训练编码器之间交互的关键限制，揭示了它们建模长期依赖性和抽象时间概念（如因果性和事件进展）方面的能力缺口。此外，我们分析了现有视频数据集带来的挑战，包括偏差、缺乏时间注释以及领域特定限制，这些限制约束了LLM的时间理解能力。为了解决这些缺口，我们探讨了有前途的发展方向，包括LLM和编码器的共生演化、具有明确时间标签的丰富数据集的发展，以及能够集成空间、时间和语义推理的新架构。通过解决这些问题，我们旨在推进LLM的时间理解能力，从而在其视频分析以及更广泛的领域中充分发挥其潜力。 

---
# CRM: Retrieval Model with Controllable Condition 

**Title (ZH)**: CRM：可控条件检索模型 

**Authors**: Chi Liu, Jiangxia Cao, Rui Huang, Kuo Cai, Weifeng Ding, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13844)  

**Abstract**: Recommendation systems (RecSys) are designed to connect users with relevant items from a vast pool of candidates while aligning with the business goals of the platform. A typical industrial RecSys is composed of two main stages, retrieval and ranking: (1) the retrieval stage aims at searching hundreds of item candidates satisfied user interests; (2) based on the retrieved items, the ranking stage aims at selecting the best dozen items by multiple targets estimation for each item candidate, including classification and regression targets. Compared with ranking model, the retrieval model absence of item candidate information during inference, therefore retrieval models are often trained by classification target only (e.g., click-through rate), but failed to incorporate regression target (e.g., the expected watch-time), which limit the effectiveness of retrieval. In this paper, we propose the Controllable Retrieval Model (CRM), which integrates regression information as conditional features into the two-tower retrieval paradigm. This modification enables the retrieval stage could fulfill the target gap with ranking model, enhancing the retrieval model ability to search item candidates satisfied the user interests and condition effectively. We validate the effectiveness of CRM through real-world A/B testing and demonstrate its successful deployment in Kuaishou short-video recommendation system, which serves over 400 million users. 

**Abstract (ZH)**: 推荐系统（RecSys）旨在将用户与大量候选物品中的相关物品连接起来，同时符合平台的商业目标。典型的工业推荐系统通常由两个主要阶段组成：检索和排名：（1）检索阶段旨在搜索满足用户兴趣的数百个候选物品；（2）基于检索到的物品，排名阶段旨在通过对每个候选物品的多个目标进行估计来选择最合适的十几项物品，包括分类和回归目标。与排名模型相比，在推理过程中检索模型不存在候选物品的信息，因此检索模型通常仅通过分类目标（例如点击率）进行训练，而未能结合回归目标（例如预期观看时间），这限制了检索模型的效果。在本文中，我们提出了一种可控检索模型（Controllable Retrieval Model，CRM），它将回归信息作为条件特征整合到双塔检索框架中。这一修改使得检索阶段能够弥补与排名模型之间的目标差距，增强了检索模型搜索符合用户兴趣和条件的候选物品的能力。我们通过实际的A/B测试验证了CRM的有效性，并展示了其在快手短视频推荐系统中的成功部署，该系统服务于超过4亿用户。 

---
# AI Perceptions Across Cultures: Similarities and Differences in Expectations, Risks, Benefits, Tradeoffs, and Value in Germany and China 

**Title (ZH)**: 跨文化视角下的AI感知：德国和中国在期望、风险、利益、权衡和价值方面的相似性和差异性 

**Authors**: Philipp Brauner, Felix Glawe, Gian Luca Liehner, Luisa Vervier, Martina Ziefle  

**Link**: [PDF](https://arxiv.org/pdf/2412.13841)  

**Abstract**: As artificial intelligence (AI) continues to advance, understanding public perceptions -- including biases, risks, and benefits -- is critical for guiding research priorities, shaping public discourse, and informing policy. This study explores public mental models of AI using micro scenarios to assess reactions to 71 statements about AI's potential future impacts. Drawing on cross-cultural samples from Germany (N=52) and China (N=60), we identify significant differences in expectations, evaluations, and risk-utility tradeoffs. German participants tended toward more cautious assessments, whereas Chinese participants expressed greater optimism regarding AI's societal benefits. Chinese participants exhibited relatively balanced risk-benefit tradeoffs ($\beta=-0.463$ for risk and $\beta=+0.484$ for benefit, $r^2=.630$). In contrast, German participants showed a stronger emphasis on AI benefits and less on risks ($\beta=-0.337$ for risk and $\beta=+0.715$ for benefit, $r^2=.839$). Visual cognitive maps illustrate these contrasts, offering new perspectives on how cultural contexts shape AI acceptance. Our findings underline key factors influencing public perception and provide actionable insights for fostering equitable and culturally sensitive integration of AI technologies. 

**Abstract (ZH)**: 随着人工智能（AI）的不断进步，理解公众的看法——包括偏见、风险和利益——对于指导研究优先事项、塑造公众舆论并制定政策至关重要。本研究通过微情景探索公众对AI的心理模型，评估了公众对71条关于AI未来影响的陈述的反应。我们利用德国（N=52）和中国（N=60）的跨文化样本，识别了在期望、评价和风险-收益权衡方面的显著差异。德国民众更倾向于谨慎的评估，而中国民众则表达了更多对未来AI社会利益的乐观态度。中国民众在风险-收益权衡方面表现出相对平衡的关系（风险$\beta=-0.463$，收益$\beta=+0.484$，$r^2=.630$）。相比之下，德国民众更注重AI的利益，而对风险的关注较少（风险$\beta=-0.337$，收益$\beta=+0.715$，$r^2=.839$）。可视化认知地图展示了这些差异，提供了文化背景如何影响AI接受度的新视角。我们的研究结果强调了影响公众态度的关键因素，并为促进公平且文化敏感的AI技术整合提供了可操作的见解。 

---
# Maybe you are looking for CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval 

**Title (ZH)**: 也许您正在寻找跨模态查询建议方法：面向文本到图像检索的模型（Cross-modal Query Suggestion for Text-to-Image Retrieval） 

**Authors**: Giacomo Pacini, Fabio Carrara, Nicola Messina, Nicola Tonellotto, Giuseppe Amato, Fabrizio Falchi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13834)  

**Abstract**: Query suggestion, a technique widely adopted in information retrieval, enhances system interactivity and the browsing experience of document collections. In cross-modal retrieval, many works have focused on retrieving relevant items from natural language queries, while few have explored query suggestion solutions. In this work, we address query suggestion in cross-modal retrieval, introducing a novel task that focuses on suggesting minimal textual modifications needed to explore visually consistent subsets of the collection, following the premise of ''Maybe you are looking for''. To facilitate the evaluation and development of methods, we present a tailored benchmark named CroQS. This dataset comprises initial queries, grouped result sets, and human-defined suggested queries for each group. We establish dedicated metrics to rigorously evaluate the performance of various methods on this task, measuring representativeness, cluster specificity, and similarity of the suggested queries to the original ones. Baseline methods from related fields, such as image captioning and content summarization, are adapted for this task to provide reference performance scores. Although relatively far from human performance, our experiments reveal that both LLM-based and captioning-based methods achieve competitive results on CroQS, improving the recall on cluster specificity by more than 115% and representativeness mAP by more than 52% with respect to the initial query. The dataset, the implementation of the baseline methods and the notebooks containing our experiments are available here: this https URL 

**Abstract (ZH)**: 查询建议是一种在信息检索中广泛采用的技术，可以增强系统的互动性并改善文档集合的浏览体验。在跨模态检索中，许多研究工作主要集中在从自然语言查询中检索相关项，而很少有工作探讨查询建议方法。本文旨在解决跨模态检索中的查询建议问题，引入了一个新任务，该任务聚焦于建议需要进行的最小文本修改，以探索与视觉一致性集相符的子集，遵循“您可能在寻找...”的假设。为了便于评估和方法开发，我们提出了一个定制的基准名为CroQS。该数据集包括初始查询、分组结果集以及每个组的人工定义的建议查询。我们建立了专门的度量标准，以严格评估各个方法在该任务上的表现，分别从代表性、簇特异性以及建议查询与初始查询的相似性等方面进行评估。来自相关领域的基线方法，如图像字幕和内容摘要方法，被适应用于此任务，以提供参考性能分数。尽管在性能上与人类表现仍有较大差距，但我们的实验结果显示，基于LLM的方法和基于字幕的方法在CroQS上的表现可与初始查询的召回率相比提高超过115%，代表性mAP提高超过52%。该数据集、基线方法的实现以及包含我们实验的笔记本程序均可在此处获取：this https URL 

---
# Heterogeneous Graph Collaborative Filtering 

**Title (ZH)**: 异质图协同过滤 

**Authors**: Lianghao Xia, Meiyan Xie, Yong Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13825)  

**Abstract**: For modern recommender systems, the use of low-dimensional latent representations to embed users and items based on their observed interactions has become commonplace. However, many existing recommendation models are primarily designed for coarse-grained and homogeneous interactions, which limits their effectiveness in two critical dimensions. Firstly, these models fail to leverage the relational dependencies that exist across different types of user behaviors, such as page views, collects, comments, and purchases. Secondly, they struggle to capture the fine-grained latent factors that drive user interaction patterns. To address these limitations, we present a heterogeneous graph collaborative filtering model MixRec that excels at disentangling users' multi-behavior interaction patterns and uncovering the latent intent factors behind each behavior. Our model achieves this by incorporating intent disentanglement and multi-behavior modeling, facilitated by a parameterized heterogeneous hypergraph architecture. Furthermore, we introduce a novel contrastive learning paradigm that adaptively explores the advantages of self-supervised data augmentation, thereby enhancing the model's resilience against data sparsity and expressiveness with relation heterogeneity. To validate the efficacy of MixRec, we conducted extensive experiments on three public datasets. The results clearly demonstrate its superior performance, significantly outperforming various state-of-the-art baselines. Our model is open-sourced and available at: this https URL. 

**Abstract (ZH)**: 对于现代推荐系统而言，基于用户观察到的交互在低维潜在表示中嵌入用户和项目已成为常态。然而，许多现有的推荐模型主要针对粗粒度和同质的交互设计，这在两个关键维度上限制了它们的效果。首先，这些模型未能利用不同用户行为类型之间存在的关系依赖性，例如页面浏览、收藏、评论和购买。其次，它们难以捕捉驱动用户交互模式的细微潜在因素。为了解决这些问题，我们提出了一种混合图协作过滤模型MixRec，该模型擅长区分用户多行为交互模式，并揭示每个行为背后的潜在意图因素。我们的模型通过结合意图解耦和多行为建模，利用参数化的异构超图架构实现这一目标。此外，我们引入了一种新颖的对比学习范式，该范式能够自适应地探索半监督数据增强的优势，从而增强模型对稀疏数据和关系异质性的鲁棒性和表达性。为了验证MixRec的有效性，我们在三个公开数据集上进行了广泛实验。实验结果清楚地表明，MixRec 在性能上显著优于各种最先进的基线方法。我们的模型已开源，可在以下链接获取：this https URL。 

---
# CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers? 

**Title (ZH)**: CAD-Assistant: 工具增强的多模态大模型作为通用CAD任务求解器？ 

**Authors**: Dimitrios Mallis, Ahmet Serdar Karadeniz, Sebastian Cavada, Danila Rukhovich, Niki Foteinopoulou, Kseniya Cherenkova, Anis Kacem, Djamila Aouada  

**Link**: [PDF](https://arxiv.org/pdf/2412.13810)  

**Abstract**: We propose CAD-Assistant, a general-purpose CAD agent for AI-assisted design. Our approach is based on a powerful Vision and Large Language Model (VLLM) as a planner and a tool-augmentation paradigm using CAD-specific modules. CAD-Assistant addresses multimodal user queries by generating actions that are iteratively executed on a Python interpreter equipped with the FreeCAD software, accessed via its Python API. Our framework is able to assess the impact of generated CAD commands on geometry and adapts subsequent actions based on the evolving state of the CAD design. We consider a wide range of CAD-specific tools including Python libraries, modules of the FreeCAD Python API, helpful routines, rendering functions and other specialized modules. We evaluate our method on multiple CAD benchmarks and qualitatively demonstrate the potential of tool-augmented VLLMs as generic CAD task solvers across diverse CAD workflows. 

**Abstract (ZH)**: 我们提出了一种通用的CAD辅助设计代理——CAD-Assistant。我们的方法基于强大的视觉和大规模语言模型（VLLM）作为规划者，并采用了针对CAD特定模块的工具增强范式。CAD-Assistant 通过生成在配备FreeCAD软件的Python解释器上迭代执行的动作来处理多模态用户查询。我们的框架能够评估生成的CAD命令对几何结构的影响，并根据CAD设计的演变状态调整后续动作。我们考虑了广泛的CAD特定工具，包括Python库、FreeCAD Python API的模块、辅助例行程序、渲染函数以及其他专门模块。我们在多个CAD基准上评估了这种方法，并从定性的角度展示了作为通用CAD任务求解器的工具增强VLLM在多样化的CAD工作流中的潜力。 

---
# AI-Powered Algorithm-Centric Quantum Processor Topology Design 

**Title (ZH)**: 基于AI驱动的算法中心量子处理机拓扑设计 

**Authors**: Tian Li, Xiao-Yue Xu, Chen Ding, Tian-Ci Tian, Wei-You Liao, Shuo Zhang, He-Liang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13805)  

**Abstract**: Quantum computing promises to revolutionize various fields, yet the execution of quantum programs necessitates an effective compilation process. This involves strategically mapping quantum circuits onto the physical qubits of a quantum processor. The qubits' arrangement, or topology, is pivotal to the circuit's performance, a factor that often defies traditional heuristic or manual optimization methods due to its complexity. In this study, we introduce a novel approach leveraging reinforcement learning to dynamically tailor qubit topologies to the unique specifications of individual quantum circuits, guiding algorithm-driven quantum processor topology design for reducing the depth of mapped circuit, which is particularly critical for the output accuracy on noisy quantum processors. Our method marks a significant departure from previous methods that have been constrained to mapping circuits onto a fixed processor topology. Experiments demonstrate that we have achieved notable enhancements in circuit performance, with a minimum of 20\% reduction in circuit depth in 60\% of the cases examined, and a maximum enhancement of up to 46\%. Furthermore, the pronounced benefits of our approach in reducing circuit depth become increasingly evident as the scale of the quantum circuits increases, exhibiting the scalability of our method in terms of problem size. This work advances the co-design of quantum processor architecture and algorithm mapping, offering a promising avenue for future research and development in the field. 

**Abstract (ZH)**: 量子计算有望革新众多领域，但量子程序的执行需要一个有效的编译过程。这个过程涉及将量子电路战略性地映射到量子处理器的物理量子位上。量子位的排列，或拓扑结构，对电路的性能至关重要，但在其复杂性面前，传统的启发式或手工优化方法常常力不从心。在这项研究中，我们提出了一种新的方法，利用强化学习动态地针对每个特殊量子电路的特定需求调整量子位拓扑结构，指导基于算法的量子处理器拓扑设计，以减少映射电路的深度，这对于嘈杂的量子处理器的输出精度尤为重要。我们的方法标志着对之前固定处理器拓扑的映射方法的重要突破。实验证明，我们已经实现了显著的电路性能提升，减少了至少20%的电路深度的情况超过60%，在某些情况下，性能提升高达46%。此外，随着量子电路规模的增加，我们方法在减少电路深度方面表现出越加显著的优势，显示了其在问题规模方面具有良好的可扩展性。这项工作推进了量子处理器架构与算法映射的协同设计，为该领域的未来研究和开发提供了前景广阔的途径。 

---
# M$^3$-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation 

**Title (ZH)**: M$^3$-VOS：多阶段、多过渡和多场景视频对象分割 

**Authors**: Zixuan Chen, Jiaxin Li, Liming Tan, Yejie Guo, Junxuan Liang, Cewu Lu, Yonglu Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13803)  

**Abstract**: Intelligent robots need to interact with diverse objects across various environments. The appearance and state of objects frequently undergo complex transformations depending on the object properties, e.g., phase transitions. However, in the vision community, segmenting dynamic objects with phase transitions is overlooked. In light of this, we introduce the concept of phase in segmentation, which categorizes real-world objects based on their visual characteristics and potential morphological and appearance changes. Then, we present a new benchmark, Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation (M3-VOS), to verify the ability of models to understand object phases, which consists of 479 high-resolution videos spanning over 10 distinct everyday scenarios. It provides dense instance mask annotations that capture both object phases and their transitions. We evaluate state-of-the-art methods on M3-VOS, yielding several key insights. Notably, current appearance based approaches show significant room for improvement when handling objects with phase transitions. The inherent changes in disorder suggest that the predictive performance of the forward entropy-increasing process can be improved through a reverse entropy-reducing process. These findings lead us to propose ReVOS, a new plug-and-play model that improves its performance by reversal refinement. Our data and code will be publicly available 

**Abstract (ZH)**: 智能机器人需要在各种环境中与多种物体进行交互。物体的外观和状态经常会发生复杂的变化，这取决于物体的特性，例如相变。然而，在视觉领域中，对具有相变的动态物体进行分割常常被忽视。鉴于此，我们引入了“相”这一概念，基于物体的视觉特征和潜在的形态和外观变化对其进行分类。然后，我们提出了一种新的基准测试（Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation，简称M3-VOS），用于验证模型理解物体相态的能力。该基准测试包含479个高分辨率视频，跨越了10种不同的日常生活场景。这些视频提供了密集的实例掩码注释，能够捕捉物体的相态及其变化。我们对最先进的方法进行了在M3-VOS上的评估，得到了几个关键见解。值得注意的是，当前基于外观的方法在处理具有相变的物体时存在显著改进空间。内在的无序变化表明，向前熵增加过程的预测性能可以通过逆转熵减少过程得到提升。这些发现促使我们提出ReVOS，这是一种新的插拔即用模型，通过逆转精炼来提高其性能。我们的数据和代码将公开提供。 

---
# Enhancing Rhetorical Figure Annotation: An Ontology-Based Web Application with RAG Integration 

**Title (ZH)**: 基于本体的增强修辞格标注：结合RAG的网络应用程序 

**Authors**: Ramona Kühn, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2412.13799)  

**Abstract**: Rhetorical figures play an important role in our communication. They are used to convey subtle, implicit meaning, or to emphasize statements. We notice them in hate speech, fake news, and propaganda. By improving the systems for computational detection of rhetorical figures, we can also improve tasks such as hate speech and fake news detection, sentiment analysis, opinion mining, or argument mining. Unfortunately, there is a lack of annotated data, as well as qualified annotators that would help us build large corpora to train machine learning models for the detection of rhetorical figures. The situation is particularly difficult in languages other than English, and for rhetorical figures other than metaphor, sarcasm, and irony. To overcome this issue, we develop a web application called "Find your Figure" that facilitates the identification and annotation of German rhetorical figures. The application is based on the German Rhetorical ontology GRhOOT which we have specially adapted for this purpose. In addition, we improve the user experience with Retrieval Augmented Generation (RAG). In this paper, we present the restructuring of the ontology, the development of the web application, and the built-in RAG pipeline. We also identify the optimal RAG settings for our application. Our approach is one of the first to practically use rhetorical ontologies in combination with RAG and shows promising results. 

**Abstract (ZH)**: 修辞手法在我们的交流中扮演着重要角色。它们用于传达微妙或隐含的意义，或者强调陈述。我们可以在仇恨言论、假新闻和宣传中注意到它们。通过改进计算检测修辞手法的系统，我们也可以提高仇恨言论和假新闻检测、情感分析、意见挖掘或论据挖掘等任务的性能。不幸的是，标注数据不足，且合格的标注者 rarity，这阻碍了我们构建用于训练检测修辞手法的机器学习模型的大规模语料库。特别是在英语以外的语言和除了比喻、讽刺和讥讽之外的其他修辞手法方面，情况尤为困难。为了解决这一问题，我们开发了一个名为“Find your Figure”的网络应用程序，以促进德国修辞手法的识别和标注。该应用程序基于我们特别为此目的调整的德国修辞本体论GRhOOT。此外，我们通过检索增强生成（RAG）改善了用户体验。在本文中，我们介绍了本体论的重构、网络应用程序的开发以及内置的RAG管道。我们还确定了适用于我们应用程序的最佳RAG设置。我们的方法是首次尝试将修辞本体论与RAG实际结合使用，并显示出令人 promising 的结果。 

---
# Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN 

**Title (ZH)**: Mix-LN：结合预层归一化和后层归一化以释放更深层的强大功能 

**Authors**: Pengxiang Li, Lu Yin, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13795)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success, yet recent findings reveal that their deeper layers often contribute minimally and can be pruned without affecting overall performance. While some view this as an opportunity for model compression, we identify it as a training shortfall rooted in the widespread use of Pre-Layer Normalization (Pre-LN). We demonstrate that Pre-LN, commonly employed in models like GPT and LLaMA, leads to diminished gradient norms in its deeper layers, reducing their effectiveness. In contrast, Post-Layer Normalization (Post-LN) preserves larger gradient norms in deeper layers but suffers from vanishing gradients in earlier layers. To address this, we introduce Mix-LN, a novel normalization technique that combines the strengths of Pre-LN and Post-LN within the same model. Mix-LN applies Post-LN to the earlier layers and Pre-LN to the deeper layers, ensuring more uniform gradients across layers. This allows all parts of the network--both shallow and deep layers--to contribute effectively to training. Extensive experiments with various model sizes from 70M to 7B demonstrate that Mix-LN consistently outperforms both Pre-LN and Post-LN, promoting more balanced, healthier gradient norms throughout the network, and enhancing the overall quality of LLM pre-training. Furthermore, we demonstrate that models pre-trained with Mix-LN learn better compared to those using Pre-LN or Post-LN during supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF), highlighting the critical importance of high-quality deep layers. By effectively addressing the inefficiencies of deep layers in current LLMs, Mix-LN unlocks their potential, enhancing model capacity without increasing model size. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已取得显著成功，但最近的研究发现，其较深层的网络结构往往贡献有限，甚至可以在不降低整体性能的情况下裁剪掉这些深层结构。虽然有些人将这一点视为模型压缩的机会，但我们认为这是由于广泛使用前置层归一化（Pre-LN）导致的训练不足。我们证明，虽前置层归一化在GPT和LLaMA等模型中广泛使用，但它会导致深层网络的梯度幅度减小，从而降低其效果。相比之下，后置层归一化（Post-LN）能够在深层网络中保持较大的梯度幅度，但在早期网络层中会导致梯度消失。为解决这一问题，我们提出了Mix-LN（混合层归一化）技术，这是一种结合了前置层归一化和后置层归一化优点的新归一化方法。Mix-LN 将后置层归一化应用于早期层，而将前置层归一化应用于深层层，从而确保网络各层具有更均匀的梯度分布。这使得网络的所有部分——无论是浅层层还是深层层——都能有效参与训练。通过各种规模的模型（从70M到7B）的大量实验，我们发现Mix-LN在各模型中始终优于前置层归一化和后置层归一化，促进网络中梯度分布更为平衡和健康，从而提高LLM的预训练质量。此外，我们还展示了使用Mix-LN预训练的模型在监督微调（SFT）和基于人类反馈的人工强化学习（RLHF）中表现出更好的学习效果，进一步突显了高质量深层网络的重要性。通过有效解决当前LLM中深层网络的低效问题，Mix-LN释放了深层网络的潜力，提高了模型能力而无需增加模型规模。我们的代码可在以下链接获取：[此处插入链接] 

---
# MATCHED: Multimodal Authorship-Attribution To Combat Human Trafficking in Escort-Advertisement Data 

**Title (ZH)**: MATCHED：多模态作者归属分析，以应对成人 Escort 广告中的人口贩卖问题 

**Authors**: Vageesh Saxena, Benjamin Bashpole, Gijs Van Dijck, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2412.13794)  

**Abstract**: Human trafficking (HT) remains a critical issue, with traffickers increasingly leveraging online escort advertisements (ads) to advertise victims anonymously. Existing detection methods, including Authorship Attribution (AA), often center on text-based analyses and neglect the multimodal nature of online escort ads, which typically pair text with images. To address this gap, we introduce MATCHED, a multimodal dataset of 27,619 unique text descriptions and 55,115 unique images collected from the Backpage escort platform across seven U.S. cities in four geographical regions. Our study extensively benchmarks text-only, vision-only, and multimodal baselines for vendor identification and verification tasks, employing multitask (joint) training objectives that achieve superior classification and retrieval performance on in-distribution and out-of-distribution (OOD) datasets. Integrating multimodal features further enhances this performance, capturing complementary patterns across text and images. While text remains the dominant modality, visual data adds stylistic cues that enrich model performance. Moreover, text-image alignment strategies like CLIP and BLIP2 struggle due to low semantic overlap and vague connections between the modalities of escort ads, with end-to-end multimodal training proving more robust. Our findings emphasize the potential of multimodal AA (MAA) to combat HT, providing LEAs with robust tools to link ads and disrupt trafficking networks. 

**Abstract (ZH)**: 人口贩卖（HT）仍然是一个关键问题，贩卖人口者越来越多地利用在线陪侍广告（广告）来匿名宣传受害者。现有的检测方法，包括作者归类（AA），通常主要集中在文本分析上，并忽略了在线陪侍广告的多模态性质，这些广告通常结合了文本和图像。为了解决这一问题，我们引入了MATCHED，这是一个包含27,619个独一无二的文本描述和55,115个独一无二的图像的多模态数据集，这些数据是从美国四个地理区域中的七个城市Backpage陪侍平台上收集的。我们的研究广泛地对仅使用文本、仅使用视觉和多模态Baseline方法在供应商识别和验证任务中进行了基准测试，采用了多任务（联合）训练目标，该目标在分布内和分布外（OOD）数据集上实现了优异的分类和检索性能。进一步整合多模态特征还可增强此性能，捕捉文本和图像之间互补的模式。尽管文本仍然是主要模态，但视觉数据添加了风格化线索，丰富了模型性能。此外，如CLIP和BLIP2之类的文本-图像对齐策略由于陪侍广告模态之间低语义重叠和模态之间的模糊联系而难以实现，端到端的多模态训练则更为稳健。我们的研究结果强调了多模态作者归类（MAA）在打击HT方面的潜力，为执法机构提供了 robust 的工具来关联广告并中断人口贩卖网络。 

---
# Meta-Reflection: A Feedback-Free Reflection Learning Framework 

**Title (ZH)**: 元反思：一种无反馈的反射学习框架 

**Authors**: Yaoke Wang, Yun Zhu, Xintong Bao, Wenqiao Zhang, Suyang Dai, Kehan Chen, Wenqiang Li, Gang Huang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13781)  

**Abstract**: Despite the remarkable capabilities of large language models (LLMs) in natural language understanding and reasoning, they often display undesirable behaviors, such as generating hallucinations and unfaithful reasoning. A prevalent strategy to mitigate these issues is the use of reflection, which refines responses through an iterative process. However, while promising, reflection heavily relies on high-quality external feedback and requires iterative multi-agent inference processes, thus hindering its practical application. In this paper, we propose Meta-Reflection, a novel feedback-free reflection mechanism that necessitates only a single inference pass without external feedback. Motivated by the human ability to remember and retrieve reflections from past experiences when encountering similar problems, Meta-Reflection integrates reflective insights into a codebook, allowing the historical insights to be stored, retrieved, and used to guide LLMs in problem-solving. To thoroughly investigate and evaluate the practicality of Meta-Reflection in real-world scenarios, we introduce an industrial e-commerce benchmark named E-commerce Customer Intent Detection (ECID). Extensive experiments conducted on both public datasets and the ECID benchmark highlight the effectiveness and efficiency of our proposed approach. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在自然语言理解和推理方面表现出色，但在生成幻觉和不忠实推理等不良行为方面也存在明显缺陷。缓解这些问题的一个常见策略是使用反思机制，通过迭代过程改进响应。然而，虽然这种方法颇具前景，但它对高质量外部反馈的依赖性和需要迭代的多代理推理过程，限制了其实际应用。本文提出了一种名为Meta-Reflection的新颖反馈自由反思机制，该机制仅需一次推理过程而不依赖于外部反馈。受人类在遇到类似问题时能够回忆和利用过去经验的能力启发，Meta-Reflection将反思洞察整合到码本中，使历史洞察得以存储、检索，并用于引导LLMs解决问题。为了全面研究和评估Meta-Reflection在实际场景中的适用性和有效性，我们引入了一个基于电子商务领域的工业基准——电子商务客户意图检测（ECID）。在公共数据集和ECID基准上的广泛实验展示了本方法的有效性和效率。 

---
# Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization 

**Title (ZH)**: 语义一致性：通过两阶段对齐和行为语义令牌化 harmonize 推荐系统 

**Authors**: Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13771)  

**Abstract**: Large language models (LLMs), endowed with exceptional reasoning capabilities, are adept at discerning profound user interests from historical behaviors, thereby presenting a promising avenue for the advancement of recommendation systems. However, a notable discrepancy persists between the sparse collaborative semantics typically found in recommendation systems and the dense token representations within LLMs. In our study, we propose a novel framework that harmoniously merges traditional recommendation models with the prowess of LLMs. We initiate this integration by transforming ItemIDs into sequences that align semantically with the LLMs space, through the proposed Alignment Tokenization module. Additionally, we design a series of specialized supervised learning tasks aimed at aligning collaborative signals with the subtleties of natural language semantics. To ensure practical applicability, we optimize online inference by pre-caching the top-K results for each user, reducing latency and improving effciency. Extensive experimental evidence indicates that our model markedly improves recall metrics and displays remarkable scalability of recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）具备出色的推理能力，能够从用户的历史行为中识别出深层次的兴趣，从而为推荐系统的发展提供了新的前景。然而，推荐系统中通常存在的稀疏协作语义与LLMs中的密集词表示间存在显著差距。在本研究中，我们提出了一种新型框架，旨在和谐地将传统推荐模型与LLMs的优势相结合。我们通过引入提出的对齐分词模块，将ItemIDs转换为与LLMs空间相匹配的语义序列，从而开启这一整合过程。此外，我们还设计了一系列专为协作信号与自然语言语义细微差异对齐的监督学习任务。为了确保其实用性，我们通过预先缓存每个用户的前K个结果来优化在线推理，从而降低延迟并提高效率。广泛的实验结果表明，我们的模型显著提高了召回率指标，并展示了推荐系统出色的可扩展性。 

---
# QuLTSF: Long-Term Time Series Forecasting with Quantum Machine Learning 

**Title (ZH)**: QuLTSF：基于量子机器学习的长期时间序列预测 

**Authors**: Hari Hara Suthan Chittoor, Paul Robert Griffin, Ariel Neufeld, Jayne Thompson, Mile Gu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13769)  

**Abstract**: Long-term time series forecasting (LTSF) involves predicting a large number of future values of a time series based on the past values and is an essential task in a wide range of domains including weather forecasting, stock market analysis, disease outbreak prediction. Over the decades LTSF algorithms have transitioned from statistical models to deep learning models like transformer models. Despite the complex architecture of transformer based LTSF models `Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2023)' showed that simple linear models can outperform the state-of-the-art transformer based LTSF models. Recently, quantum machine learning (QML) is evolving as a domain to enhance the capabilities of classical machine learning models. In this paper we initiate the application of QML to LTSF problems by proposing QuLTSF, a simple hybrid QML model for multivariate LTSF. Through extensive experiments on a widely used weather dataset we show the advantages of QuLTSF over the state-of-the-art classical linear models, in terms of reduced mean squared error and mean absolute error. 

**Abstract (ZH)**: 长期时间序列预测（LTSF）涉及基于过去值预测时间序列中的大量未来值，是气象预报、股票市场分析和疾病暴发预测等多个领域中的重要任务。多年来，LTSF算法从统计模型发展到了深度学习模型，如变压器模型。尽管基于变压器的LTSF模型架构复杂，但`Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2023)`的研究表明，简单的线性模型可以超越最先进的基于变压器的LTSF模型。最近，量子机器学习（QML）正逐渐成为一个领域，以增强经典机器学习模型的能力。在这篇论文中，我们提出了将QML应用于LTSF问题的方法，提出了一种简单的混合QML模型QuLTSF，用于多变量LTSF。通过在广泛使用的气象数据集上进行广泛的实验，我们展示了QuLTSF在降低均方误差和平均绝对误差方面相较于最先进的经典线性模型的优势。 

---
# LLM-SEM: A Sentiment-Based Student Engagement Metric Using LLMS for E-Learning Platforms 

**Title (ZH)**: LLM-SEM：一种基于情感的学生参与度度量方法——利用大型语言模型的在线学习平台学生参与度评估 

**Authors**: Ali Hamdi, Ahmed Abdelmoneim Mazrou, Mohamed Shaltout  

**Link**: [PDF](https://arxiv.org/pdf/2412.13765)  

**Abstract**: Current methods for analyzing student engagement in e-learning platforms, including automated systems, often struggle with challenges such as handling fuzzy sentiment in text comments and relying on limited metadata. Traditional approaches, such as surveys and questionnaires, also face issues like small sample sizes and scalability. In this paper, we introduce LLM-SEM (Language Model-Based Student Engagement Metric), a novel approach that leverages video metadata and sentiment analysis of student comments to measure engagement. By utilizing recent Large Language Models (LLMs), we generate high-quality sentiment predictions to mitigate text fuzziness and normalize key features such as views and likes. Our holistic method combines comprehensive metadata with sentiment polarity scores to gauge engagement at both the course and lesson levels. Extensive experiments were conducted to evaluate various LLM models, demonstrating the effectiveness of LLM-SEM in providing a scalable and accurate measure of student engagement. We fine-tuned LLMs, including AraBERT, TXLM-RoBERTa, LLama 3B and Gemma 9B from Ollama, using human-annotated sentiment datasets to enhance prediction accuracy. 

**Abstract (ZH)**: 当前分析在线学习平台中学生参与度的方法，包括自动化系统，往往面临处理文本评论中模糊情感和依赖有限元数据的挑战。传统的研究方法，如调查问卷，也面临着样本量较小和可扩展性差的问题。在本文中，我们引入了一种名为LLM-SEM（基于语言模型的学生参与度指标）的新颖方法，该方法利用视频元数据和学生评论的情感分析来衡量参与度。通过利用最新的大规模语言模型（LLMs），我们生成高质量的情感预测，以减轻文本模糊性，并标准化关键特征，如观看次数和点赞数。我们的综合方法结合了全面的元数据与情感极性分数，以在课程和课节层面评估参与度。进行了广泛的实验来评估不同的LLM模型，证明了LLM-SEM在提供可扩展且准确的学生参与度度量方面的有效性。我们对包括AraBERT、TXLM-RoBERTa、LLama 3B和Ollama的Gemma 9B在内的LLM进行了微调，使用人工标注的情感数据集来提高预测准确性。 

---
# RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment 

**Title (ZH)**: RAG-RewardBench：用于偏好对齐的检索增强生成中奖励模型的基准测试 

**Authors**: Zhuoran Jin, Hongbang Yuan, Tianyi Men, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13746)  

**Abstract**: Despite the significant progress made by existing retrieval augmented language models (RALMs) in providing trustworthy responses and grounding in reliable sources, they often overlook effective alignment with human preferences. In the alignment process, reward models (RMs) act as a crucial proxy for human values to guide optimization. However, it remains unclear how to evaluate and select a reliable RM for preference alignment in RALMs. To this end, we propose RAG-RewardBench, the first benchmark for evaluating RMs in RAG settings. First, we design four crucial and challenging RAG-specific scenarios to assess RMs, including multi-hop reasoning, fine-grained citation, appropriate abstain, and conflict robustness. Then, we incorporate 18 RAG subsets, six retrievers, and 24 RALMs to increase the diversity of data sources. Finally, we adopt an LLM-as-a-judge approach to improve preference annotation efficiency and effectiveness, exhibiting a strong correlation with human annotations. Based on the RAG-RewardBench, we conduct a comprehensive evaluation of 45 RMs and uncover their limitations in RAG scenarios. Additionally, we also reveal that existing trained RALMs show almost no improvement in preference alignment, highlighting the need for a shift towards preference-aligned this http URL release our benchmark and code publicly at this https URL for future work. 

**Abstract (ZH)**: 尽管现有的检索增强语言模型（RALMs）在提供可靠响应和基于可靠来源方面取得了显著进展，但在与人类偏好有效对齐方面常常被忽视。在对齐过程中，奖励模型（RMs）作为人类价值观的关键代理，用于指导优化。然而，如何评估和选择适合偏好对齐的可靠RM仍不明确。为此，我们提出了RAG-RewardBench——第一个用于评估RAG环境中的RMs的基准。首先，我们设计了四个关键且具有挑战性的RAG特定场景来评估RMs，包括多跳推理、细粒度引用、适当弃权和冲突鲁棒性。然后，我们采用了18个RAG子集、六种检索器和24种RALMs，以增加数据源的多样性。最后，我们采用了一种LLM-as-a-judge的方法以提高偏好注释的效率和有效性，显示出与人类注释的强烈相关性。基于RAG-RewardBench，我们对45种RMs进行了全面评估，并揭示了它们在RAG场景中的局限性。此外，我们还发现现有的训练好的RALMs几乎在偏好对齐方面没有改进，突显了转向偏好对齐的必要性。我们将在下面的网址公开发布我们的基准和代码以供未来研究使用：<this http URL> 

---
# Uncertainty separation via ensemble quantile regression 

**Title (ZH)**: 通过集成分位数回归进行不确定性分离 

**Authors**: Navid Ansari, Hans-Peter Seidel, Vahid Babaei  

**Link**: [PDF](https://arxiv.org/pdf/2412.13738)  

**Abstract**: This paper introduces a novel and scalable framework for uncertainty estimation and separation with applications in data driven modeling in science and engineering tasks where reliable uncertainty quantification is critical. Leveraging an ensemble of quantile regression (E-QR) models, our approach enhances aleatoric uncertainty estimation while preserving the quality of epistemic uncertainty, surpassing competing methods, such as Deep Ensembles (DE) and Monte Carlo (MC) dropout. To address challenges in separating uncertainty types, we propose an algorithm that iteratively improves separation through progressive sampling in regions of high uncertainty. Our framework is scalable to large datasets and demonstrates superior performance on synthetic benchmarks, offering a robust tool for uncertainty quantification in data-driven applications. 

**Abstract (ZH)**: 本文提出了一种新颖且可扩展的框架，用于不确定性估计与分离，并应用于科学和工程任务中的数据驱动建模，其中可靠的不确定性量化至关重要。该框架利用基于核的方法（Ensemble of Quantile Regression, E-QR）模型，在提升 aleatoric 不确定性估计的同时保持 epistemic 不确定性的质量，超越了诸如深层集成（Deep Ensembles, DE）和蒙特卡洛（Monte Carlo, MC）丢弃等竞争方法。为了解决不确定性类型分离的挑战，我们提出了一种算法，通过在高不确定性区域进行渐进式采样逐步提高分离效果。该框架适用于大规模数据集，并在合成基准测试中表现出优越的性能，提供了一种在数据驱动应用中进行不确定性量化 robust 的工具。 

---
# On the Compression of Language Models for Code: An Empirical Study on CodeBERT 

**Title (ZH)**: 对代码语言模型的压缩研究：CodeBERT 的实证分析 

**Authors**: Giordano d'Aloisio, Luca Traini, Federica Sarro, Antinisca Di Marco  

**Link**: [PDF](https://arxiv.org/pdf/2412.13737)  

**Abstract**: Language models have proven successful across a wide range of software engineering tasks, but their significant computational costs often hinder their practical adoption. To address this challenge, researchers have begun applying various compression strategies to improve the efficiency of language models for code. These strategies aim to optimize inference latency and memory usage, though often at the cost of reduced model effectiveness. However, there is still a significant gap in understanding how these strategies influence the efficiency and effectiveness of language models for code. Here, we empirically investigate the impact of three well-known compression strategies -- knowledge distillation, quantization, and pruning -- across three different classes of software engineering tasks: vulnerability detection, code summarization, and code search. Our findings reveal that the impact of these strategies varies greatly depending on the task and the specific compression method employed. Practitioners and researchers can use these insights to make informed decisions when selecting the most appropriate compression strategy, balancing both efficiency and effectiveness based on their specific needs. 

**Abstract (ZH)**: 语言模型在广泛范围的软件工程任务中已被证明是有效的，但它们庞大的计算成本常常阻碍其实际应用。为应对这一挑战，研究人员已经开始应用各种压缩策略来提高语言模型在代码中的效率。这些策略旨在优化推理延迟和内存使用，但通常会牺牲模型的有效性。然而，关于这些策略如何影响语言模型在代码中的效率和有效性的问题仍然存在很大空白。在此，我们通过实证研究考察了三种广泛认可的压缩策略——知识蒸馏、量化和剪枝——在三种不同类别的软件工程任务中的影响：漏洞检测、代码摘要和代码搜索。我们的研究发现，这些策略的影响因任务和采用的具体压缩方法而异。从业者和研究人员可以根据这些见解，根据各自的需要，在效率和有效性之间作出明智的选择，从而选择最合适的压缩策略。 

---
# Federated Learning and RAG Integration: A Scalable Approach for Medical Large Language Models 

**Title (ZH)**: 联邦学习与RAG集成：一种适用于医疗大型语言模型的可扩展方法 

**Authors**: Jincheol Jung, Hongju Jeong, Eui-Nam Huh  

**Link**: [PDF](https://arxiv.org/pdf/2412.13720)  

**Abstract**: This study analyzes the performance of domain-specific Large Language Models (LLMs) for the medical field by integrating Retrieval-Augmented Generation (RAG) systems within a federated learning framework. Leveraging the inherent advantages of federated learning, such as preserving data privacy and enabling distributed computation, this research explores the integration of RAG systems with models trained under varying client configurations to optimize performance. Experimental results demonstrate that the federated learning-based models integrated with RAG systems consistently outperform their non-integrated counterparts across all evaluation metrics. This study highlights the potential of combining federated learning and RAG systems for developing domain-specific LLMs in the medical field, providing a scalable and privacy-preserving solution for enhancing text generation capabilities. 

**Abstract (ZH)**: 本研究通过在联邦学习框架中整合检索增强生成（RAG）系统，分析了特定领域的大型语言模型（LLMs）在医学领域的性能。利用联邦学习固有的优势，如数据隐私保护和分布式计算能力，本研究探索了在不同客户端配置下将RAG系统与训练模型相结合的方法，以优化性能。实验结果表明，基于联邦学习并与RAG系统集成的模型在所有评估指标中均优于未集成的模型。本研究突显了将联邦学习与RAG系统结合用于开发医学领域特定的LLMs的潜力，提供了一种可扩展且保护隐私的解决方案，以提高文本生成能力。 

---
# Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation 

**Title (ZH)**: 通过防御性后缀生成减轻LLM中的对抗性攻击 

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun  

**Link**: [PDF](https://arxiv.org/pdf/2412.13705)  

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理任务中展现出了卓越的性能。然而，这些模型仍然容易受到对抗性攻击的影响，少量的输入扰动可能导致有害或误导性输出。为此，我们设计了一种基于梯度的防御后缀生成算法，以增强LLMs的鲁棒性。通过向输入提示中添加精心优化的防御后缀，该算法减少了对抗性影响，同时保持模型的实用价值。为了增强对抗性理解，我们提出了一种新的总损失函数（$L_{\text{total}}$），结合了防御损失（$L_{\text{def}}$）和对抗损失（$L_{\text{adv}}$），更有效地生成了防御后缀。我们在开源LLM（如Gemma-7B、mistral-7B、Llama2-7B和Llama2-13B）上进行的实验评估表明，所提出的方法相较于未使用防御后缀的模型，将攻击成功率（ASR）平均降低了11%。此外，当使用openELM-270M生成的防御后缀时，Gemma-7B的困惑度得分从6.57降至3.93。而且，TruthfulQA评估结果显示，在各种测试配置中，诚实度得分提高了最高达10%。此方法显著增强了LLMs在关键应用中的安全性，而无需进行大量的重新训练。 

---
# Typhoon 2: A Family of Open Text and Multimodal Thai Large Language Models 

**Title (ZH)**: typhoon 2：一系列开放文本和多模态泰语大型语言模型 

**Authors**: Kunat Pipatanakul, Potsawee Manakul, Natapong Nitarach, Warit Sirichotedumrong, Surapon Nonesung, Teetouch Jaknamon, Parinthapat Pengpun, Pittawat Taveekitworachai, Adisai Na-Thalang, Sittipong Sripaisarnmongkol, Krisanapong Jirayoot, Kasima Tharnpipitchai  

**Link**: [PDF](https://arxiv.org/pdf/2412.13702)  

**Abstract**: This paper introduces Typhoon 2, a series of text and multimodal large language models optimized for the Thai language. The series includes models for text, vision, and audio. Typhoon2-Text builds on state-of-the-art open models, such as Llama 3 and Qwen2, and we perform continual pre-training on a mixture of English and Thai data. We employ various post-training techniques to enhance Thai language performance while preserving the base models' original capabilities. We release text models across a range of sizes, from 1 to 70 billion parameters, available in both base and instruction-tuned variants. Typhoon2-Vision improves Thai document understanding while retaining general visual capabilities, such as image captioning. Typhoon2-Audio introduces an end-to-end speech-to-speech model architecture capable of processing audio, speech, and text inputs and generating both text and speech outputs simultaneously. 

**Abstract (ZH)**: 本文介绍了Typhoon 2系列，这是一个针对泰语优化的文本和多模态大型语言模型系列。该系列包括文本、视觉和音频模型。Typhoon2-Text在最新的开源模型（如Llama 3和Qwen2）的基础上进行了构建，并在英语和泰语数据的混合集中进行了持续的预训练。我们采用多种后训练技术以提升泰语性能的同时保留基础模型的原始功能。我们发布了从1亿到70亿参数不等的多种文本模型，这些模型既有基础版也有指令微调版。Typhoon2-Vision在保持一般视觉能力（如图像字幕）的同时，提高了对泰语文档的理解能力。Typhoon2-Audio引入了一种端到端的语音到语音模型架构，能够处理语音、音频和文本输入，并同时生成文本和语音输出。 

---
# Clio: Privacy-Preserving Insights into Real-World AI Use 

**Title (ZH)**: Clio：保护隐私的现实世界AI使用洞察 

**Authors**: Alex Tamkin, Miles McCain, Kunal Handa, Esin Durmus, Liane Lovitt, Ankur Rathi, Saffron Huang, Alfred Mountfield, Jerry Hong, Stuart Ritchie, Michael Stern, Brian Clarke, Landon Goldberg, Theodore R. Sumers, Jared Mueller, William McEachen, Wes Mitchell, Shan Carter, Jack Clark, Jared Kaplan, Deep Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2412.13678)  

**Abstract**: How are AI assistants being used in the real world? While model providers in theory have a window into this impact via their users' data, both privacy concerns and practical challenges have made analyzing this data difficult. To address these issues, we present Clio (Claude insights and observations), a privacy-preserving platform that uses AI assistants themselves to analyze and surface aggregated usage patterns across millions of conversations, without the need for human reviewers to read raw conversations. We validate this can be done with a high degree of accuracy and privacy by conducting extensive evaluations. We demonstrate Clio's usefulness in two broad ways. First, we share insights about how models are being used in the real world from one million this http URL Free and Pro conversations, ranging from providing advice on hairstyles to providing guidance on Git operations and concepts. We also identify the most common high-level use cases on this http URL (coding, writing, and research tasks) as well as patterns that differ across languages (e.g., conversations in Japanese discuss elder care and aging populations at higher-than-typical rates). Second, we use Clio to make our systems safer by identifying coordinated attempts to abuse our systems, monitoring for unknown unknowns during critical periods like launches of new capabilities or major world events, and improving our existing monitoring systems. We also discuss the limitations of our approach, as well as risks and ethical concerns. By enabling analysis of real-world AI usage, Clio provides a scalable platform for empirically grounded AI safety and governance. 

**Abstract (ZH)**: AI助手在实际世界中的应用情况如何？虽然模型提供商理论上可以通过用户数据了解到这一点的影响，但由于隐私问题和实际挑战，分析这些数据变得困难。为了解决这些问题，我们提出了Clio（Claude洞察与观察），这是一个保护隐私的平台，它利用AI助手本身来分析并呈现数百万次对话中的聚合使用模式，而无需人工审核人员阅读原始对话。我们通过广泛的研究进一步验证了这一点，既确保了高精度也保证了隐私。我们以两种广泛的方式展示了Clio的功效。首先，我们基于一百万个Free和Pro会话分享了关于模型在实际世界中的应用洞察，这些会话内容涵盖了从发型建议到Git操作指南等广泛领域。我们还识别了最常见的一级使用案例（如编程、写作和研究任务），以及不同语言之间的模式差异（例如，日语讨论的老龄护理话题比典型水平更高）。其次，我们利用Clio提升系统安全性，包括识别有组织的努力滥用系统、在新能力发布或重大世界事件期间监控未知的未知因素，并改进现有的监测系统。我们还讨论了我们方法的局限性、风险和伦理问题。通过使对实际世界AI使用的分析变得可能，Clio提供了一个实现AI安全性和治理的可扩展平台。 

---
# Exploring Multi-Modal Integration with Tool-Augmented LLM Agents for Precise Causal Discovery 

**Title (ZH)**: 利用工具增强的多模态LLM代理进行精确因果发现的研究 

**Authors**: ChengAo Shen, Zhengzhang Chen, Dongsheng Luo, Dongkuan Xu, Haifeng Chen, Jingchao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2412.13667)  

**Abstract**: Causal inference is an imperative foundation for decision-making across domains, such as smart health, AI for drug discovery and AIOps. Traditional statistical causal discovery methods, while well-established, predominantly rely on observational data and often overlook the semantic cues inherent in cause-and-effect relationships. The advent of Large Language Models (LLMs) has ushered in an affordable way of leveraging the semantic cues for knowledge-driven causal discovery, but the development of LLMs for causal discovery lags behind other areas, particularly in the exploration of multi-modality data. To bridge the gap, we introduce MATMCD, a multi-agent system powered by tool-augmented LLMs. MATMCD has two key agents: a Data Augmentation agent that retrieves and processes modality-augmented data, and a Causal Constraint agent that integrates multi-modal data for knowledge-driven inference. Delicate design of the inner-workings ensures successful cooperation of the agents. Our empirical study across seven datasets suggests the significant potential of multi-modality enhanced causal discovery. 

**Abstract (ZH)**: 因果推理是跨领域（如智能健康、药物发现和AIOps）决策的基础。传统的统计因果发现方法虽然具有良好的基础，但主要依赖于观察性数据，往往会忽视因果关系中固有的语义线索。大型语言模型（LLMs）的出现使得利用这些语义线索进行知识驱动的因果发现变得更加经济，但用于因果发现的LLMs的发展滞后于其他领域，尤其是在多模态数据探索方面。为弥合这一差距，我们引入了MATMCD，这是一种由工具增强的LLMs支撑的多智能体系统。MATMCD包含两个关键智能体：一个数据增强智能体，负责检索和处理增加模态的数据；一个因果约束智能体，用于集成多模态数据进行知识驱动的推理。精心设计的内部机制确保了智能体的有效合作。我们在七个数据集上的实证研究表明，增强的多模态因果发现具有显著的潜力。 

---
# Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation 

**Title (ZH)**: 对大型语言模型在生成个性化虚假信息方面被误用的脆弱性评估 

**Authors**: Aneta Zugecova, Dominik Macko, Ivan Srba, Robert Moro, Jakub Kopal, Katarina Marcincinova, Matus Mesarcik  

**Link**: [PDF](https://arxiv.org/pdf/2412.13666)  

**Abstract**: The capabilities of recent large language models (LLMs) to generate high-quality content indistinguishable by humans from human-written texts rises many concerns regarding their misuse. Previous research has shown that LLMs can be effectively misused for generating disinformation news articles following predefined narratives. Their capabilities to generate personalized (in various aspects) content have also been evaluated and mostly found usable. However, a combination of personalization and disinformation abilities of LLMs has not been comprehensively studied yet. Such a dangerous combination should trigger integrated safety filters of the LLMs, if there are some. This study fills this gap by evaluation of vulnerabilities of recent open and closed LLMs, and their willingness to generate personalized disinformation news articles in English. We further explore whether the LLMs can reliably meta-evaluate the personalization quality and whether the personalization affects the generated-texts detectability. Our results demonstrate the need for stronger safety-filters and disclaimers, as those are not properly functioning in most of the evaluated LLMs. Additionally, our study revealed that the personalization actually reduces the safety-filter activations; thus effectively functioning as a jailbreak. Such behavior must be urgently addressed by LLM developers and service providers. 

**Abstract (ZH)**: 近年来大型语言模型（LLMs）生成高质量内容的能力，这些内容难以被人类区分为机器所写，引发了对其滥用的诸多担忧。前期研究显示，LLMs 可以有效用于生成遵循预定义叙述的虚假新闻文章。此外，它们生成个性化内容的能力（在多个方面）也得到了评估，并普遍认为是可以使用的。然而，LLMs 的个性化能力和虚假信息生成能力的结合尚未进行全面研究。这样的危险结合应促使现有的安全过滤器进行综合化处理。本研究通过评估开放和封闭的LLMs的脆弱性及其生成个性化虚假新闻文章的倾向，填补了这一空白。我们进一步探讨了LLMs是否能够可靠地元评估个性化质量，以及个性化是否影响生成文本的可检测性。研究结果表明，需要加强安全过滤器和免责声明，因为大多数评估的LLMs中的这些功能并未正常运作。此外，我们的研究还揭示，个性化实际上减少了安全过滤器的激活，有效起着一种“脱狱”的作用。这种行为必须引起LLM开发者和服务提供商的紧急关注和应对。 

---
# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference 

**Title (ZH)**: 更智能、更高效、更快捷、更持久：一种现代双向编码器，用于快速、节省内存和处理长上下文的微调与推理 

**Authors**: Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, Iacopo Poli  

**Link**: [PDF](https://arxiv.org/pdf/2412.13663)  

**Abstract**: Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT models exhibit state-of-the-art results on a large pool of evaluations encompassing diverse classification tasks and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream performance, ModernBERT is also the most speed and memory efficient encoder and is designed for inference on common GPUs. 

**Abstract (ZH)**: 以下是符合学术规范的中文翻译：

与较大的解码器模型相比，如 BERT 这样的编码器仅模型提供了在检索和分类任务中出色的性能-大小权衡。尽管 BERT 自发布以来在众多生产线中得到了广泛应用，但关于 BERT 的帕累托改进仍然有限。在本文中，我们引入了 ModernBERT，将现代模型优化应用到编码器仅模型中，代表了对较旧编码器的重大帕累托改进。ModernBERT 模型在包含 2 万亿个标记并具有原生 8192 序列长度的训练下，展现了在多样分类任务和不同领域（包括代码）的单向量和多向量检索中的一流评估结果。除了下游性能强大，ModernBERT 也是最高速、最节省内存的编码器，并且适合在常见的 GPU 上进行推理。 

---
# When Should We Prefer State-to-Visual DAgger Over Visual Reinforcement Learning? 

**Title (ZH)**: 我们在什么情况下应该优先选择状态到视觉DAgger算法而非视觉强化学习算法？ 

**Authors**: Tongzhou Mu, Zhaoyang Li, Stanisław Wiktor Strzelecki, Xiu Yuan, Yunchao Yao, Litian Liang, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.13662)  

**Abstract**: Learning policies from high-dimensional visual inputs, such as pixels and point clouds, is crucial in various applications. Visual reinforcement learning is a promising approach that directly trains policies from visual observations, although it faces challenges in sample efficiency and computational costs. This study conducts an empirical comparison of State-to-Visual DAgger, a two-stage framework that initially trains a state policy before adopting online imitation to learn a visual policy, and Visual RL across a diverse set of tasks. We evaluate both methods across 16 tasks from three benchmarks, focusing on their asymptotic performance, sample efficiency, and computational costs. Surprisingly, our findings reveal that State-to-Visual DAgger does not universally outperform Visual RL but shows significant advantages in challenging tasks, offering more consistent performance. In contrast, its benefits in sample efficiency are less pronounced, although it often reduces the overall wall-clock time required for training. Based on our findings, we provide recommendations for practitioners and hope that our results contribute valuable perspectives for future research in visual policy learning. 

**Abstract (ZH)**: 从高维度视觉输入（如像素和点云）中学习策略在各种应用中至关重要。视觉强化学习是一种直接从视觉观测中训练策略的有前景的方法，尽管它面临着样本效率和计算成本的挑战。本研究通过一项实证比较，将State-to-Visual DAgger这一两阶段框架与视觉RL在一系列任务上进行了对比，该框架最初训练一个状态策略，然后再采用在线模仿学习视觉策略。我们在这三个方面——渐近性能、样本效率和计算成本——评估了这两种方法。令人惊讶的是，我们的研究发现State-to-Visual DAgger并不总是优于视觉RL，但在具有挑战性的任务中显示出显著的优势，提供了更一致的性能。相比之下，它在样本效率方面的优势不太明显，尽管它往往能够减少整个训练所需的墙钟时间。基于我们的发现，我们为实践者提供了建议，并希望我们的结果能够为未来视觉策略学习研究提供有价值的视角。 

---
# G-VEval: A Versatile Metric for Evaluating Image and Video Captions Using GPT-4o 

**Title (ZH)**: G-VEval：一种使用GPT-4o评估图像和视频描述的通用指标 

**Authors**: Tony Cheng Tong, Sirui He, Zhiwen Shao, Dit-Yan Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2412.13647)  

**Abstract**: Evaluation metric of visual captioning is important yet not thoroughly explored. Traditional metrics like BLEU, METEOR, CIDEr, and ROUGE often miss semantic depth, while trained metrics such as CLIP-Score, PAC-S, and Polos are limited in zero-shot scenarios. Advanced Language Model-based metrics also struggle with aligning to nuanced human preferences. To address these issues, we introduce G-VEval, a novel metric inspired by G-Eval and powered by the new GPT-4o. G-VEval uses chain-of-thought reasoning in large multimodal models and supports three modes: reference-free, reference-only, and combined, accommodating both video and image inputs. We also propose MSVD-Eval, a new dataset for video captioning evaluation, to establish a more transparent and consistent framework for both human experts and evaluation metrics. It is designed to address the lack of clear criteria in existing datasets by introducing distinct dimensions of Accuracy, Completeness, Conciseness, and Relevance (ACCR). Extensive results show that G-VEval outperforms existing methods in correlation with human annotations, as measured by Kendall tau-b and Kendall tau-c. This provides a flexible solution for diverse captioning tasks and suggests a straightforward yet effective approach for large language models to understand video content, paving the way for advancements in automated captioning. Codes are available at this https URL 

**Abstract (ZH)**: 视觉描述评价指标至关重要但尚未得到充分探索。传统的指标如 BLEU、METEOR、CIDEr 和 ROUGE 经常忽视语义深度，而经过训练的指标如 CLIP-Score、PAC-S 和 Polos 在零样本场景中受限。基于高级语言模型的指标也难以与细腻的人类偏好对齐。为解决这些问题，我们引入了 G-VEval，这是一种受到 G-Eval 启发并由新推出的 GPT-4o 支撑的新型指标。G-VEval 利用大型多模态模型中的链式推理，并支持三种模式：无参考、仅参考和结合模式，能够适应视频和图像的输入。我们还提出了一个新数据集 MSVD-Eval，用于视频描述评价，旨在为专家和评价指标提供一个更透明和一致的框架。该数据集通过引入准确性、完整性、简洁性和相关性（ACCR）等不同的维度，填补了现有数据集缺乏清晰评估标准的空白。广泛的结果表明，G-VEval 在与人类注释的相关性上优于现有方法，这由肯德尔 tau-b 和肯德尔 tau-c 测量得出。这提供了一种灵活的解决方案，适用于多种描述任务，并表明大型语言模型理解视频内容的一种简单而有效的方法，为自动化描述的进步铺平了道路。相关代码可在以下网址获取：this https URL 

---
# Consistency of Compositional Generalization across Multiple Levels 

**Title (ZH)**: 多层面上的组分泛化一致性 

**Authors**: Chuanhao Li, Zhen Li, Chenchen Jing, Xiaomeng Fan, Wenbo Ye, Yuwei Wu, Yunde Jia  

**Link**: [PDF](https://arxiv.org/pdf/2412.13636)  

**Abstract**: Compositional generalization is the capability of a model to understand novel compositions composed of seen concepts. There are multiple levels of novel compositions including phrase-phrase level, phrase-word level, and word-word level. Existing methods achieve promising compositional generalization, but the consistency of compositional generalization across multiple levels of novel compositions remains unexplored. The consistency refers to that a model should generalize to a phrase-phrase level novel composition, and phrase-word/word-word level novel compositions that can be derived from it simultaneously. In this paper, we propose a meta-learning based framework, for achieving consistent compositional generalization across multiple levels. The basic idea is to progressively learn compositions from simple to complex for consistency. Specifically, we divide the original training set into multiple validation sets based on compositional complexity, and introduce multiple meta-weight-nets to generate sample weights for samples in different validation sets. To fit the validation sets in order of increasing compositional complexity, we optimize the parameters of each meta-weight-net independently and sequentially in a multilevel optimization manner. We build a GQA-CCG dataset to quantitatively evaluate the consistency. Experimental results on visual question answering and temporal video grounding, demonstrate the effectiveness of the proposed framework. We release GQA-CCG at this https URL. 

**Abstract (ZH)**: 组合泛化是指一个模型能够理解由已知概念组成的新颖组合的能力。这些新颖组合可以存在于短语-短语级别、短语-词级别和词-词级别等多个层次上。现有的方法在组合泛化方面取得了令人鼓舞的成果，但这些成果在多个层次的新颖组合之间的一致性尚未得到探索。一致性具体指模型应能够对某一新颖组成的短语-短语级别的新颖组合以及从中衍生出的短语-词/词-词级别的新颖组合进行泛化。

在本文中，我们提出了一种基于元学习的框架，以实现多层面上的一致组合泛化。基本思想是从简单到复杂逐步学习组合以确保一致性。具体而言，我们将原始训练集根据组合复杂性划分成多个验证集，并引入多个元权重网络，用于为不同验证集中的样本生成样本权重。为了按组合复杂性递增的顺序拟合验证集，我们采用多层优化的方式独立且顺序地优化每个元权重网络的参数。我们构建了一个GQA-CCG数据集，用于定量评估一致性。在视觉问答和基于时间的视频定位任务上的实验结果表明了所提出框架的有效性。我们已在以下网址发布了GQA-CCG数据集：[GQA-CCG数据集链接]。

请注意，由于您提供的英文原文中包含一个URL，这里使用了"[GQA-CCG数据集链接]"来代替具体的URL，以便标准格式化。如果您需要包含具体的URL，请提供该URL进行粘贴替换。 

---
# Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model 

**Title (ZH)**: 策略装饰器：大型策略模型的模型无关在线精炼方法 

**Authors**: Xiu Yuan, Tongzhou Mu, Stone Tao, Yunhao Fang, Mengke Zhang, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.13630)  

**Abstract**: Recent advancements in robot learning have used imitation learning with large models and extensive demonstrations to develop effective policies. However, these models are often limited by the quantity, quality, and diversity of demonstrations. This paper explores improving offline-trained imitation learning models through online interactions with the environment. We introduce Policy Decorator, which uses a model-agnostic residual policy to refine large imitation learning models during online interactions. By implementing controlled exploration strategies, Policy Decorator enables stable, sample-efficient online learning. Our evaluation spans eight tasks across two benchmarks-ManiSkill and Adroit-and involves two state-of-the-art imitation learning models (Behavior Transformer and Diffusion Policy). The results show Policy Decorator effectively improves the offline-trained policies and preserves the smooth motion of imitation learning models, avoiding the erratic behaviors of pure RL policies. See our project page (this https URL) for videos. 

**Abstract (ZH)**: 近年来，机器人学习的进展使用大规模模型和丰富的演示来开发有效的策略，但这些模型往往受限于演示的数量、质量和多样性。本文探讨了通过在线与环境交互来改进离线训练的模仿学习模型的方法。我们引入了“策略装饰器”，该方法使用一种模型无关的残差策略，在线交互过程中逐步优化大型模仿学习模型。通过实现受控探索策略，“策略装饰器”能够实现稳定且样本高效的在线学习。评估跨越了ManiSkill和Adroit两个基准的八个任务，并涉及两种最新的模仿学习模型（行为变换器和扩散政策）。结果表明，“策略装饰器”能够有效改进离线训练的策略，并保留模仿学习模型的平滑运动，避免了纯强化学习策略的不稳定行为。更多视频请参见我们的项目页面（[此链接](this https URL)）。 

---
# LIFT: Improving Long Context Understanding Through Long Input Fine-Tuning 

**Title (ZH)**: LIFT：通过长输入微调提高长期语境理解 

**Authors**: Yansheng Mao, Jiaqi Li, Fanxu Meng, Jing Xiong, Zilong Zheng, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13626)  

**Abstract**: Long context understanding remains challenging for large language models due to their limited context windows. This paper introduces Long Input Fine-Tuning (LIFT) for long context modeling, a novel framework that enhances LLM performance on long-context tasks by adapting model parameters to the context at test time. LIFT enables efficient processing of lengthy inputs without the computational burden of offline long-context adaptation, and can improve the long-context capabilities of arbitrary short-context models. The framework is further enhanced by integrating in-context learning and pre-LIFT supervised fine-tuning. The combination of in-context learning and LIFT enables short-context models like Llama 3 to handle arbitrarily long contexts and consistently improves their performance on popular long-context benchmarks like LooGLE and LongBench. We also provide a comprehensive analysis of the strengths and limitations of LIFT on long context understanding, offering valuable directions for future research. 

**Abstract (ZH)**: 长上下文理解一直是大型语言模型面临的挑战，原因在于它们受限于有限的上下文窗口。本文介绍了一种名为长输入微调（LIFT）的新框架，该框架通过在测试时调整模型参数来增强长上下文任务中的LLM性能。LIFT能够在不使用离线长时间上下文适配的高计算成本的情况下，高效处理长输入，并可以提升任意短期上下文模型的长上下文能力。通过结合上下文学习和预LIFT监督微调，该框架进一步得到了增强。上下文学习与LIFT的结合使如Llama 3这样的短期上下文模型能够处理任意长的上下文，并在流行的长上下文基准测试（如LooGLE和LongBench）中表现出持续的性能提升。我们还对LIFT在长上下文理解中的强项和局限性进行了全面分析，为未来的研究提供了有价值的指导方向。 

---
# Unifying Attribution-Based Explanations Using Functional Decomposition 

**Title (ZH)**: 使用功能分解统一基于属性的解释 

**Authors**: Arne Gevaert, Yvan Saeys  

**Link**: [PDF](https://arxiv.org/pdf/2412.13623)  

**Abstract**: The black box problem in machine learning has led to the introduction of an ever-increasing set of explanation methods for complex models. These explanations have different properties, which in turn has led to the problem of method selection: which explanation method is most suitable for a given use case? In this work, we propose a unifying framework of attribution-based explanation methods, which provides a step towards a rigorous study of the similarities and differences of explanations. We first introduce removal-based attribution methods (RBAMs), and show that an extensively broad selection of existing methods can be viewed as such RBAMs. We then introduce the canonical additive decomposition (CAD). This is a general construction for additively decomposing any function based on the central idea of removing (groups of) features. We proceed to show that indeed every valid additive decomposition is an instance of the CAD, and that any removal-based attribution method is associated with a specific CAD. Next, we show that any removal-based attribution method can be completely defined as a game-theoretic value or interaction index for a specific (possibly constant-shifted) cooperative game, which is defined using the corresponding CAD of the method. We then use this intrinsic connection to define formal descriptions of specific behaviours of explanation methods, which we also call functional axioms, and identify sufficient conditions on the corresponding CAD and game-theoretic value or interaction index of an attribution method under which the attribution method is guaranteed to adhere to these functional axioms. Finally, we show how this unifying framework can be used to develop new, efficient approximations for existing explanation methods. 

**Abstract (ZH)**: 机器学习中的黑盒问题促使引入了大量用于解释复杂模型的解释方法。这些解释各有不同的特性，这反过来又导致了方法选择问题：哪种解释方法适用于给定的应用场景？在本研究中，我们提出了一种归因基解释方法的统一框架，这为我们系统研究解释方法的相似性和差异性提供了基础。首先，我们介绍了基于移除的归因方法（RBAMs），并展示了广泛存在的许多现有方法都可以视为这类RBAMs。接着，我们介绍了通用加性分解（CAD）。这是一种基于移除（一组）特征的核心思想来基于任何函数进行加性分解的一般构造。然后，我们证明了确实每一个有效的加性分解都是CAD的一个实例，而任何基于移除的归因方法都与特定的CAD相关联。接下来，我们表明任何基于移除的归因方法可以完全定义为特定合作博弈（可能是常数偏移合作博弈）中的博弈论价值或交互指数，该合作博弈是通过相应的方法的CAD定义的。然后，我们利用这种内在联系来定义特定解释方法行为的正式描述，我们将这些描述称为功能公理，并识别在归因方法的相应CAD和博弈论价值或交互指数下满足这些功能公理的充分条件。最后，我们展示了如何利用这一统一框架开发现有解释方法的新颖且高效的近似算法。 

---
# NPC: Neural Predictive Control for Fuel-Efficient Autonomous Trucks 

**Title (ZH)**: NPC：神经预测控制在高效自动驾驶卡车中的应用 

**Authors**: Jiaping Ren, Jiahao Xiang, Hongfei Gao, Jinchuan Zhang, Yiming Ren, Yuexin Ma, Yi Wu, Ruigang Yang, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13618)  

**Abstract**: Fuel efficiency is a crucial aspect of long-distance cargo transportation by oil-powered trucks that economize on costs and decrease carbon emissions. Current predictive control methods depend on an accurate model of vehicle dynamics and engine, including weight, drag coefficient, and the Brake-specific Fuel Consumption (BSFC) map of the engine. We propose a pure data-driven method, Neural Predictive Control (NPC), which does not use any physical model for the vehicle. After training with over 20,000 km of historical data, the novel proposed NVFormer implicitly models the relationship between vehicle dynamics, road slope, fuel consumption, and control commands using the attention mechanism. Based on the online sampled primitives from the past of the current freight trip and anchor-based future data synthesis, the NVFormer can infer optimal control command for reasonable fuel consumption. The physical model-free NPC outperforms the base PCC method with 2.41% and 3.45% more significant fuel saving in simulation and open-road highway testing, respectively. 

**Abstract (ZH)**: 油动力卡车长距离货物运输中的燃油效率是降低运营成本和减少碳排放的关键方面。当前的预测控制方法依赖于车辆动力学和发动机的精确模型，包括车辆重量、空气阻力系数以及发动机的平均制动燃油消耗量（BSFC）图。我们提出了一种纯数据驱动的方法——神经预测控制（NPC），这种方法不使用任何车辆的物理模型。通过使用超过20,000公里的历史数据进行训练后，新型的NVFormer通过注意力机制隐式地建模了车辆动力学、道路坡度、燃油消耗与控制命令之间的关系。基于当前货运行程过往的部分实时采样基础数据以及基于锚点的未来数据合成，NVFormer可以推断出合理的最佳控制命令以实现优化的燃油消耗。这种无物理模型的NPC在仿真和开放道路高速公路上的测试中分别比基线PCC方法节省了2.41%和3.45%的燃油。 

---
# Reverse Region-to-Entity Annotation for Pixel-Level Visual Entity Linking 

**Title (ZH)**: 像素级别视觉实体链接中的反向区域到实体标注 

**Authors**: Zhengfei Xu, Sijia Zhao, Yanchao Hao, Xiaolong Liu, Lili Li, Yuyang Yin, Bo Li, Xi Chen, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13614)  

**Abstract**: Visual Entity Linking (VEL) is a crucial task for achieving fine-grained visual understanding, matching objects within images (visual mentions) to entities in a knowledge base. Previous VEL tasks rely on textual inputs, but writing queries for complex scenes can be challenging. Visual inputs like clicks or bounding boxes offer a more convenient alternative. Therefore, we propose a new task, Pixel-Level Visual Entity Linking (PL-VEL), which uses pixel masks from visual inputs to refer to objects, supplementing reference methods for VEL. To facilitate research on this task, we have constructed the MaskOVEN-Wiki dataset through an entirely automatic reverse region-entity annotation framework. This dataset contains over 5 million annotations aligning pixel-level regions with entity-level labels, which will advance visual understanding towards fine-grained. Moreover, as pixel masks correspond to semantic regions in an image, we enhance previous patch-interacted attention with region-interacted attention by a visual semantic tokenization approach. Manual evaluation results indicate that the reverse annotation framework achieved a 94.8% annotation success rate. Experimental results show that models trained on this dataset improved accuracy by 18 points compared to zero-shot models. Additionally, the semantic tokenization method achieved a 5-point accuracy improvement over the trained baseline. 

**Abstract (ZH)**: 视觉实体链接（VEL）是实现细粒度视觉理解的关键任务，它涉及将图像中的对象（视觉提及）与知识库中的实体进行匹配。以往的VEL任务多依赖于文本输入，但对于复杂场景而言，编写查询可能具有挑战性。视觉输入，如点击或边界框，提供了一种更便捷的替代方案。因此，我们提出了一项新的任务——像素级视觉实体链接（PL-VEL），该任务利用视觉输入的像素掩码来指代对象，并补充了VEL的引用方法。为了促进对该任务的研究，我们通过一个完全自动的逆区域-实体标注框架构造了MaskOVEN-Wiki数据集。该数据集包含超过500万个像素级别区域与实体级别标签对齐的标注，有助于推进视觉理解向细粒度方向发展。此外，由于像素掩码对应于图像中的语义区域，我们通过视觉语义分词方法增强了先前的 patch-交互注意力机制，引入了区域-交互注意力机制。人工评估结果表明，该逆标注框架实现了94.8%的成功标注率。实验结果表明，基于该数据集训练的模型在准确率上比零样本模型提高了18个百分点。此外，语义分词方法使训练基线的准确率提高了5个百分点。 

---
# Are LLMs Good Literature Review Writers? Evaluating the Literature Review Writing Ability of Large Language Models 

**Title (ZH)**: 大型语言模型是好的文献综述撰稿人吗？评估大型语言模型的文献综述撰写能力 

**Authors**: Xuemei Tang, Xufeng Duan, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2412.13612)  

**Abstract**: The literature review is a crucial form of academic writing that involves complex processes of literature collection, organization, and summarization. The emergence of large language models (LLMs) has introduced promising tools to automate these processes. However, their actual capabilities in writing comprehensive literature reviews remain underexplored, such as whether they can generate accurate and reliable references. To address this gap, we propose a framework to assess the literature review writing ability of LLMs automatically. We evaluate the performance of LLMs across three tasks: generating references, writing abstracts, and writing literature reviews. We employ external tools for a multidimensional evaluation, which includes assessing hallucination rates in references, semantic coverage, and factual consistency with human-written context. By analyzing the experimental results, we find that, despite advancements, even the most sophisticated models still cannot avoid generating hallucinated references. Additionally, different models exhibit varying performance in literature review writing across different disciplines. 

**Abstract (ZH)**: 文献综述是一种重要的学术写作形式，涉及复杂的过程，包括文献的收集、组织和总结。大型语言模型（LLMs）的出现为这些过程的自动化提供了有前景的工具。然而，LLMs在撰写全面文献综述的实际能力仍然尚未得到充分探索，例如它们能否生成准确可靠的参考文献。为填补这一空白，我们提出了一种框架，用于自动评估LLMs的文献综述写作能力。我们通过三个任务来评估LLMs的表现：生成参考文献、撰写摘要和撰写文献综述。我们使用外部工具进行多维度评估，包括评估参考文献中的虚构率、语义覆盖范围以及与人类撰写的背景内容的一致性。通过分析实验结果，我们发现尽管技术有了进步，但即使是最先进的模型也无法完全避免生成虚构的参考文献。此外，不同模型在不同的学科领域中，在撰写文献综述方面的表现也有所不同。 

---
# Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation 

**Title (ZH)**: 更快更强大：当ANN-SNN转换遇到并行脉冲计算 

**Authors**: Zecheng Hao, Zhaofei Yu, Tiejun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13610)  

**Abstract**: Spiking Neural Network (SNN), as a brain-inspired and energy-efficient network, is currently facing the pivotal challenge of exploring a suitable and efficient learning framework. The predominant training methodologies, namely Spatial-Temporal Back-propagation (STBP) and ANN-SNN Conversion, are encumbered by substantial training overhead or pronounced inference latency, which impedes the advancement of SNNs in scaling to larger networks and navigating intricate application domains. In this work, we propose a novel parallel conversion learning framework, which establishes a mathematical mapping relationship between each time-step of the parallel spiking neurons and the cumulative spike firing rate. We theoretically validate the lossless and sorting properties of the conversion process, as well as pointing out the optimal shifting distance for each step. Furthermore, by integrating the above framework with the distribution-aware error calibration technique, we can achieve efficient conversion towards more general activation functions or training-free circumstance. Extensive experiments have confirmed the significant performance advantages of our method for various conversion cases under ultra-low time latency. To our best knowledge, this is the first work which jointly utilizes parallel spiking calculation and ANN-SNN Conversion, providing a highly promising approach for SNN supervised training. 

**Abstract (ZH)**: 基于脉冲的神经网络（Spiking Neural Network, SNN）作为一种受脑启发且能效高的网络，目前正面临着探索合适的高效学习框架的关键挑战。现有的主要训练方法，即空间-时间反向传播（Spatial-Temporal Back-propagation, STBP）和ANN到SNN的转换，要么受到巨大的训练开销的困扰，要么导致显著的推理延迟，这阻碍了SNN在构建更大规模网络和处理复杂应用场景方面的进一步发展。本文提出了一种新颖的并行转换学习框架，该框架在每个时间步的并行脉冲神经元与累积脉冲发射率之间建立了数学映射关系。我们从理论上验证了转换过程的无损性和排序特性，并指出了每步的最佳平移距离。为进一步提高效率，我们通过将上述框架与分布感知误差校准技术结合，实现了高效地向更通用的激活函数或无需训练的情况下进行转换。大量的实验结果证实了我们在超低时间延迟条件下对各种转换案例的显著性能优势。据我们所知，这是首次将并行脉冲计算和ANN到SNN的转换相结合的工作，为SNN的监督训练提供了一种极具前景的方法。 

---
# Hybrid CNN-LSTM based Indoor Pedestrian Localization with CSI Fingerprint Maps 

**Title (ZH)**: 基于CSI指纹图的混合CNN-LSTM室内行人定位方法 

**Authors**: Muhammad Emad-ud-din  

**Link**: [PDF](https://arxiv.org/pdf/2412.13601)  

**Abstract**: The paper presents a novel Wi-Fi fingerprinting system that uses Channel State Information (CSI) data for fine-grained pedestrian localization. The proposed system exploits the frequency diversity and spatial diversity of the features extracted from CSI data to generate a 2D+channel image termed as a CSI Fingerprint Map. We then use this CSI Fingerprint Map representation of CSI data to generate a pedestrian trajectory hypothesis using a hybrid architecture that combines a Convolutional Neural Network and a Long Short-Term Memory Recurrent Neural Network model. The proposed architecture exploits the temporal and spatial relationship information among the CSI data observations gathered at neighboring locations. A particle filter is then employed to separate out the most likely hypothesis matching a human walk model. The experimental performance of our method is compared to existing deep learning localization methods such ConFi, DeepFi and to a self-developed temporal-feature based LSTM based location classifier. The experimental results show marked improvement with an average RMSE of 0.36 m in a moderately dynamic and 0.17 m in a static environment. Our method is essentially a proof of concept that with (1) sparse availability of observations, (2) limited infrastructure requirements, (3) moderate level of short-term and long-term noise in the training and testing environment, reliable fine-grained Wi-Fi based pedestrian localization is a potential option. 

**Abstract (ZH)**: 本文提出了一种新颖的Wi-Fi指纹定位系统，该系统利用信道状态信息（CSI）数据实现细粒度的行人定位。提出的系统利用从CSI数据中提取的特征的频率多样性和空间多样性生成了一种称为CSI指纹图的2D+信道图像。随后，我们利用这一CSI指纹图表示方式，通过结合卷积神经网络（CNN）和长短期记忆循环神经网络（LSTM）模型的混合架构来生成行人轨迹假设。该提出的架构利用了在邻近位置采集的CSI数据观测值之间的时间和空间关系信息。然后采用粒子滤波器分离出最符合人类步行动态模型的假设。实验结果将我们的方法与现有的深度学习定位方法（如ConFi、DeepFi）以及一个基于时间特征的LSTM位置分类器进行了比较。实验结果表明，在中等动态环境下，平均RMSE为0.36米，在静态环境下，平均RMSE为0.17米。我们的方法本质上是一种概念验证，表明在（1）稀疏观测数据、（2）有限的基础设施需求、以及（3）训练和测试环境中存在一定程度的短期和长期噪声的情况下，基于Wi-Fi的细粒度行人定位是可行的选项。 

---
# Generalizable Sensor-Based Activity Recognition via Categorical Concept Invariant Learning 

**Title (ZH)**: 基于类别概念不变学习的可迁移传感器活动识别 

**Authors**: Di Xiong, Shuoyuan Wang, Lei Zhang, Wenbo Huang, Chaolei Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.13594)  

**Abstract**: Human Activity Recognition (HAR) aims to recognize activities by training models on massive sensor data. In real-world deployment, a crucial aspect of HAR that has been largely overlooked is that the test sets may have different distributions from training sets due to inter-subject variability including age, gender, behavioral habits, etc., which leads to poor generalization performance. One promising solution is to learn domain-invariant representations to enable a model to generalize on an unseen distribution. However, most existing methods only consider the feature-invariance of the penultimate layer for domain-invariant learning, which leads to suboptimal results. In this paper, we propose a Categorical Concept Invariant Learning (CCIL) framework for generalizable activity recognition, which introduces a concept matrix to regularize the model in the training stage by simultaneously concentrating on feature-invariance and logit-invariance. Our key idea is that the concept matrix for samples belonging to the same activity category should be similar. Extensive experiments on four public HAR benchmarks demonstrate that our CCIL substantially outperforms the state-of-the-art approaches under cross-person, cross-dataset, cross-position, and one-person-to-another settings. 

**Abstract (ZH)**: 人类活动识别（HAR）旨在通过训练大规模传感器数据来识别活动。在实际部署中，HAR中一个被广泛关注不足的重要方面是测试集可能与训练集具有不同的分布，这主要是由于个体间差异（包括年龄、性别、行为习惯等）引起的，从而导致较差的泛化性能。一种有希望的解决方案是学习领域不变的表示，以使模型能够在未见过的分布上泛化。然而，现有的大多数方法只考虑了最后一层的特征不变性，这导致了次优化的结果。在这篇论文中，我们提出了一种分类概念不变学习（CCIL）框架，该框架通过同时关注特征不变性和输出不变性，引入一个概念矩阵来在训练阶段规训模型。我们的主要观点是，属于同一活动类别的样本应该具有相似的概念矩阵。在四个公开的HAR基准数据集上的广泛实验表明，我们的CCIL在跨个体、跨数据集、跨位置以及一人到另一人的情况下，明显优于现有的先进方法。 

---
# SemiDFL: A Semi-Supervised Paradigm for Decentralized Federated Learning 

**Title (ZH)**: 半监督DFL：一种去中心化的 federated 学习半监督范式 

**Authors**: Xinyang Liu, Pengchao Han, Xuan Li, Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13589)  

**Abstract**: Decentralized federated learning (DFL) realizes cooperative model training among connected clients without relying on a central server, thereby mitigating communication bottlenecks and eliminating the single-point failure issue present in centralized federated learning (CFL). Most existing work on DFL focuses on supervised learning, assuming each client possesses sufficient labeled data for local training. However, in real-world applications, much of the data is unlabeled. We address this by considering a challenging yet practical semisupervised learning (SSL) scenario in DFL, where clients may have varying data sources: some with few labeled samples, some with purely unlabeled data, and others with both. In this work, we propose SemiDFL, the first semi-supervised DFL method that enhances DFL performance in SSL scenarios by establishing a consensus in both data and model spaces. Specifically, we utilize neighborhood information to improve the quality of pseudo-labeling, which is crucial for effectively leveraging unlabeled data. We then design a consensusbased diffusion model to generate synthesized data, which is used in combination with pseudo-labeled data to create mixed datasets. Additionally, we develop an adaptive aggregation method that leverages the model accuracy of synthesized data to further enhance SemiDFL performance. Through extensive experimentation, we demonstrate the remarkable performance superiority of the proposed DFL-Semi method over existing CFL and DFL schemes in both IID and non-IID SSL scenarios. 

**Abstract (ZH)**: 去中心化联邦学习（Decentralized Federated Learning, DFL）在不必依赖中央服务器的情况下，实现了连接客户端之间的协同模型训练，从而缓解了集中式联邦学习（Centralized Federated Learning, CFL）中的通信瓶颈问题，并消除了单一故障点的问题。大多数现有的DFL工作主要集中在监督学习场景下，假设每个客户端都有足够的带标签数据进行本地训练。然而，在实际应用中，大部分数据是未标记的。我们通过考虑DFL中的一个既具有挑战性又具有实践意义的半监督学习（Semi-Supervised Learning, SSL）场景来应对这一问题，在这种场景中，客户端的数据来源可能各不相同：一些客户端拥有少量的带标签样本，一些客户端只有纯未标记数据，还有一些客户端同时拥有带标签和未标记数据。在本文中，我们提出了SemiDFL，这是一种全新的半监督DFL方法，旨在通过在数据空间和模型空间中建立共识来提升DFL在SSL场景中的性能。具体而言，我们利用邻域信息来提高伪标签的质量，这对于有效利用未标记数据至关重要。接着，我们设计了一种基于共识的扩散模型来生成合成数据，并将合成数据与伪标签数据结合生成混合数据集。此外，我们还开发了一种适应性聚合方法，利用合成数据的模型准确性进一步提升SemiDFL的性能。通过广泛实验，我们展示了所提出的DFL-Semi方法在既定的IID和非IID SSL场景下的性能优越性，明显优于现有的CFL和DFL方案。 

---
# Socio-Culturally Aware Evaluation Framework for LLM-Based Content Moderation 

**Title (ZH)**: 基于社会文化的 awareness 评价框架：用于大语言模型内容审核的评估体系 

**Authors**: Shanu Kumar, Gauri Kholkar, Saish Mendke, Anubhav Sadana, Parag Agrawal, Sandipan Dandapat  

**Link**: [PDF](https://arxiv.org/pdf/2412.13578)  

**Abstract**: With the growth of social media and large language models, content moderation has become crucial. Many existing datasets lack adequate representation of different groups, resulting in unreliable assessments. To tackle this, we propose a socio-culturally aware evaluation framework for LLM-driven content moderation and introduce a scalable method for creating diverse datasets using persona-based generation. Our analysis reveals that these datasets provide broader perspectives and pose greater challenges for LLMs than diversity-focused generation methods without personas. This challenge is especially pronounced in smaller LLMs, emphasizing the difficulties they encounter in moderating such diverse content. 

**Abstract (ZH)**: 随着社交媒体和大规模语言模型的兴起，内容审核变得愈发重要。当前许多现有的数据集在不同群体的代表性方面存在不足，导致评估结果不可靠。为解决这一问题，我们提出了一种社会文化意识强的内容审核评价框架，并引入了一种基于人物生成的可扩展方法以创建多元化的数据集。我们的分析表明，这些数据集为语言模型提供了更广泛的观点，并在挑战上超越了没有人物聚焦的多样性生成方法。这种挑战在较小的语言模型中尤为显著，突显了它们在审核如此多元内容时所面临的困难。 

---
# Bridge then Begin Anew: Generating Target-relevant Intermediate Model for Source-free Visual Emotion Adaptation 

**Title (ZH)**: 桥接然后重塑：生成面向目标的中间模型以实现无源视觉情感适应 

**Authors**: Jiankun Zhu, Sicheng Zhao, Jing Jiang, Wenbo Tang, Zhaopan Xu, Tingting Han, Pengfei Xu, Hongxun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13577)  

**Abstract**: Visual emotion recognition (VER), which aims at understanding humans' emotional reactions toward different visual stimuli, has attracted increasing attention. Given the subjective and ambiguous characteristics of emotion, annotating a reliable large-scale dataset is hard. For reducing reliance on data labeling, domain adaptation offers an alternative solution by adapting models trained on labeled source data to unlabeled target data. Conventional domain adaptation methods require access to source data. However, due to privacy concerns, source emotional data may be inaccessible. To address this issue, we propose an unexplored task: source-free domain adaptation (SFDA) for VER, which does not have access to source data during the adaptation process. To achieve this, we propose a novel framework termed Bridge then Begin Anew (BBA), which consists of two steps: domain-bridged model generation (DMG) and target-related model adaptation (TMA). First, the DMG bridges cross-domain gaps by generating an intermediate model, avoiding direct alignment between two VER datasets with significant differences. Then, the TMA begins training the target model anew to fit the target structure, avoiding the influence of source-specific knowledge. Extensive experiments are conducted on six SFDA settings for VER. The results demonstrate the effectiveness of BBA, which achieves remarkable performance gains compared with state-of-the-art SFDA methods and outperforms representative unsupervised domain adaptation approaches. 

**Abstract (ZH)**: 视觉情感识别（VER），旨在理解人类对不同视觉刺激的情感反应，已吸引了越来越多的关注。由于情感的主观性和模糊性，构建可靠的大型数据集较为困难。为减少对数据标注的依赖，领域适应提供了一种替代方案，即通过将已在标记源数据上训练的模型适应未标记的目标数据来解决问题。传统领域的适应方法需要访问源数据。然而，由于隐私问题，源情感数据可能不可访问。为应对这一问题，我们提出了一项未被探索的任务：无源领域适应（Source-Free Domain Adaptation, SFDA）在情感识别中的应用，该方法在适应过程中无法访问源数据。为实现这一目标，我们提出了一种新颖的框架，称为桥接再开始（BBA），该框架包含两个步骤：领域桥梁生成（Domain-Bridged Model Generation, DMG）和目标相关模型适应（Target-Related Model Adaptation, TMA）。首先，DMG通过生成一个中间模型来跨越领域差距，从而避免直接对两个具有显著差异的VER数据集进行对齐。然后，TMA重新开始训练目标模型，使之适应目标结构，从而避免源特有知识的影响。我们在六个SFDA设置上进行了广泛的实验，结果表明BBA的有效性，其性能显著优于现有的SFDA方法，并且优于代表性的无监督领域适应方法。 

---
# Seeking Consistent Flat Minima for Better Domain Generalization via Refining Loss Landscapes 

**Title (ZH)**: 通过细化损失景貌以更好地寻求一致的平坦极小值从而提升领域泛化性能 

**Authors**: Aodi Li, Liansheng Zhuang, Xiao Long, Minghong Yao, Shafei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13573)  

**Abstract**: Domain generalization aims to learn a model from multiple training domains and generalize it to unseen test domains. Recent theory has shown that seeking the deep models, whose parameters lie in the flat minima of the loss landscape, can significantly reduce the out-of-domain generalization error. However, existing methods often neglect the consistency of loss landscapes in different domains, resulting in models that are not simultaneously in the optimal flat minima in all domains, which limits their generalization ability. To address this issue, this paper proposes an iterative Self-Feedback Training (SFT) framework to seek consistent flat minima that are shared across different domains by progressively refining loss landscapes during training. It alternatively generates a feedback signal by measuring the inconsistency of loss landscapes in different domains and refines these loss landscapes for greater consistency using this feedback signal. Benefiting from the consistency of the flat minima within these refined loss landscapes, our SFT helps achieve better out-of-domain generalization. Extensive experiments on DomainBed demonstrate superior performances of SFT when compared to state-of-the-art sharpness-aware methods and other prevalent DG baselines. On average across five DG benchmarks, SFT surpasses the sharpness-aware minimization by 2.6% with ResNet-50 and 1.5% with ViT-B/16, respectively. The code will be available soon. 

**Abstract (ZH)**: 领域泛化旨在从多个训练域中学习一个模型，并将其推广到未见过的测试域。近年来的理论研究表明，寻找那些参数位于损失景观平坦极小值中的深层模型，可以显著减少域外泛化误差。然而，现有方法往往忽略了不同域中损失景观一致性的需求，导致模型在所有域中同时位于最优平坦极小值的情况无法实现，从而限制了它们的泛化能力。为解决这一问题，本文提出了一种迭代自反馈训练（SFT）框架，通过在训练过程中逐步优化损失景观的一致性，以寻找被不同域共享的平坦极小值。该框架通过测量不同域中的损失景观一致性来交替生成反馈信号，并利用该反馈信号进一步优化这些损失景观以提高一致性。得益于这些优化后的损失景观中平坦极小值的一致性，我们的SFT有助于实现更好的域外泛化。在DomainBed上的广泛实验表明，与最新的尖锐性感知方法和其他常见的领域泛化基线相比，SFT在性能上具有明显的优越性。具体而言，在五个领域泛化基准测试中，SFT分别在ResNet-50和ViT-B/16模型上比尖锐性感知最小化方法高出2.6%和1.5%的性能。相关代码将在不久的将来公开。 

---
# CA-Edit: Causality-Aware Condition Adapter for High-Fidelity Local Facial Attribute Editing 

**Title (ZH)**: CA-Edit：因果关系意识条件适配器，用于-high-fidelity局部面部特征编辑

注：这里的"High-Fidelity"翻译为“高保真”，表示编辑后的面部特征与实际情况非常接近。由于"High-Fidelity"在中文里通常不直接翻译，根据上下文和学术规范，此处使用“高保真”以保持学术术语的一致性和专业性。 

**Authors**: Xiaole Xian, Xilin He, Zenghao Niu, Junliang Zhang, Weicheng Xie, Siyang Song, Zitong Yu, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2412.13565)  

**Abstract**: For efficient and high-fidelity local facial attribute editing, most existing editing methods either require additional fine-tuning for different editing effects or tend to affect beyond the editing regions. Alternatively, inpainting methods can edit the target image region while preserving external areas. However, current inpainting methods still suffer from the generation misalignment with facial attributes description and the loss of facial skin details. To address these challenges, (i) a novel data utilization strategy is introduced to construct datasets consisting of attribute-text-image triples from a data-driven perspective, (ii) a Causality-Aware Condition Adapter is proposed to enhance the contextual causality modeling of specific details, which encodes the skin details from the original image while preventing conflicts between these cues and textual conditions. In addition, a Skin Transition Frequency Guidance technique is introduced for the local modeling of contextual causality via sampling guidance driven by low-frequency alignment. Extensive quantitative and qualitative experiments demonstrate the effectiveness of our method in boosting both fidelity and editability for localized attribute editing. The code is available at this https URL. 

**Abstract (ZH)**: 为了实现高效且保真的局部面部属性编辑，目前大多数编辑方法要么需要额外的微调来适应不同的编辑效果，要么容易对编辑区域之外的区域产生影响。相比之下，插画方法可以在保持外部区域不变的情况下编辑目标图像区域。然而，当前的插画方法仍然存在面部属性描述生成不匹配以及面部皮肤细节丢失的问题。为了应对这些挑战，本文做了以下工作：(i) 从数据驱动的角度引入了一种新的数据利用策略，构建了由属性-文本-图像三元组构成的数据集；(ii) 提出了一种因果感知条件适配器，以增强特定细节的上下文因果建模能力，该方法从原始图像中编码皮肤细节，同时防止这些线索与文本条件之间的冲突。此外，还引入了一种皮肤过渡频率引导技术，以通过低频对齐引导采样来局部建模上下文因果关系。广泛的定量和定性实验表明，该方法在提高局部属性编辑的真实性和可编辑性方面非常有效。代码可在以下链接获取：this https URL。 

---
# EscapeBench: Pushing Language Models to Think Outside the Box 

**Title (ZH)**: EscapeBench: 推动语言模型突破常规思考 

**Authors**: Cheng Qian, Peixuan Han, Qinyu Luo, Bingxiang He, Xiusi Chen, Yuji Zhang, Hongyi Du, Jiarui Yao, Xiaocheng Yang, Denghui Zhang, Yunzhu Li, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2412.13549)  

**Abstract**: Language model agents excel in long-session planning and reasoning, but existing benchmarks primarily focus on goal-oriented tasks with explicit objectives, neglecting creative adaptation in unfamiliar environments. To address this, we introduce EscapeBench, a benchmark suite of room escape game environments designed to challenge agents with creative reasoning, unconventional tool use, and iterative problem-solving to uncover implicit goals. Our results show that current LM models, despite employing working memory and Chain-of-Thought reasoning, achieve only 15% average progress without hints, highlighting their limitations in creativity. To bridge this gap, we propose EscapeAgent, a framework designed to enhance creative reasoning through Foresight (innovative tool use) and Reflection (identifying unsolved tasks). Experiments show that EscapeAgent can execute action chains over 1,000 steps while maintaining logical coherence. It navigates and completes games with up to 40% fewer steps and hints, performs robustly across varying difficulty levels, and achieves higher action success rates with more efficient and innovative puzzle-solving strategies. All the data and codes are released. 

**Abstract (ZH)**: 语言模型代理在长会话规划和推理方面表现出色，但现有的基准测试主要集中在具有明确目标的指令性任务上，忽略了在不熟悉环境中创造性适应的能力。为了解决这个问题，我们引入了EscapeBench，这是一套针对房间逃脱游戏环境的基准测试套件，旨在挑战代理进行创造性推理、非常规工具使用以及迭代问题解决，以揭示隐含的目标。我们的实验结果表明，尽管当前的LM模型采用了工作记忆和链式推理，但在没有任何提示的情况下，平均进步仅为15%，突显了其在创造力方面的局限性。为了弥补这一差距，我们提出了EscapeAgent框架，旨在通过展望（创新工具使用）和反思（识别未解决的任务）来增强创造性推理。实验结果显示，EscapeAgent能够在超过1000步的行动链中保持逻辑连贯性。与以往方法相比，它可以在最少40%的步骤和提示下完成游戏，并且在不同难度级别上表现稳定，通过更高效和创新的解谜策略提高了行动成功率。所有数据和代码均已发布。 

---
# Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models 

**Title (ZH)**: 基于大型语言模型弥合用户侧知识鸿沟的知觉推荐系统研究 

**Authors**: Zheng Hu, Zhe Li, Ziyun Jiao, Satoshi Nakagawa, Jiawen Deng, Shimin Cai, Tao Zhou, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.13544)  

**Abstract**: In recent years, knowledge graphs have been integrated into recommender systems as item-side auxiliary information, enhancing recommendation accuracy. However, constructing and integrating structural user-side knowledge remains a significant challenge due to the improper granularity and inherent scarcity of user-side features. Recent advancements in Large Language Models (LLMs) offer the potential to bridge this gap by leveraging their human behavior understanding and extensive real-world knowledge. Nevertheless, integrating LLM-generated information into recommender systems presents challenges, including the risk of noisy information and the need for additional knowledge transfer. In this paper, we propose an LLM-based user-side knowledge inference method alongside a carefully designed recommendation framework to address these challenges. Our approach employs LLMs to infer user interests based on historical behaviors, integrating this user-side information with item-side and collaborative data to construct a hybrid structure: the Collaborative Interest Knowledge Graph (CIKG). Furthermore, we propose a CIKG-based recommendation framework that includes a user interest reconstruction module and a cross-domain contrastive learning module to mitigate potential noise and facilitate knowledge transfer. We conduct extensive experiments on three real-world datasets to validate the effectiveness of our method. Our approach achieves state-of-the-art performance compared to competitive baselines, particularly for users with sparse interactions. 

**Abstract (ZH)**: 近年来，知识图谱已被整合到推荐系统中作为项目侧辅助信息，以提升推荐准确性。然而，由于用户侧特征潜在的不适当粒度和固有的稀疏性，构建和整合结构性用户侧知识仍是一个重大挑战。最近在大型语言模型（LLMs）方面的进展提供了通过利用其对人类行为的理解和广泛的实际世界知识来弥合这一差距的可能性。不过，将LLM生成的信息整合到推荐系统中也面临着挑战，包括噪声信息的风险以及需要额外的知识迁移。在本文中，我们提出了一种基于LLM的用户侧知识推理方法以及一个精心设计的推荐框架，以解决这些问题。我们的方法利用LLM根据历史行为推断用户兴趣，并结合项目侧和协作数据构建混合结构：协作兴趣知识图谱（CIKG）。此外，我们提出了一种基于CIKG的推荐框架，包括用户兴趣重建模块和跨域对比学习模块，以减轻潜在的噪音并促进知识迁移。我们在三个真实世界数据集上进行了广泛实验，以验证我们方法的有效性。我们的方法在与竞品基线相比时表现出了最先进的性能，尤其是在用户稀疏交互的情况下。 

---
# Query-centric Audio-Visual Cognition Network for Moment Retrieval, Segmentation and Step-Captioning 

**Title (ZH)**: 面向查询的音频-视觉认知网络模型：用于瞬间检索、分割和步骤标注 

**Authors**: Yunbin Tu, Liang Li, Li Su, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13543)  

**Abstract**: Video has emerged as a favored multimedia format on the internet. To better gain video contents, a new topic HIREST is presented, including video retrieval, moment retrieval, moment segmentation, and step-captioning. The pioneering work chooses the pre-trained CLIP-based model for video retrieval, and leverages it as a feature extractor for other three challenging tasks solved in a multi-task learning paradigm. Nevertheless, this work struggles to learn the comprehensive cognition of user-preferred content, due to disregarding the hierarchies and association relations across modalities. In this paper, guided by the shallow-to-deep principle, we propose a query-centric audio-visual cognition (QUAG) network to construct a reliable multi-modal representation for moment retrieval, segmentation and step-captioning. Specifically, we first design the modality-synergistic perception to obtain rich audio-visual content, by modeling global contrastive alignment and local fine-grained interaction between visual and audio modalities. Then, we devise the query-centric cognition that uses the deep-level query to perform the temporal-channel filtration on the shallow-level audio-visual representation. This can cognize user-preferred content and thus attain a query-centric audio-visual representation for three tasks. Extensive experiments show QUAG achieves the SOTA results on HIREST. Further, we test QUAG on the query-based video summarization task and verify its good generalization. 

**Abstract (ZH)**: 视频已成为互联网上流行的多媒体格式。为了更好地获取视频内容，提出了一项新的主题HIREST，包括视频检索、时刻检索、时刻分割和步骤字幕生成。这项开创性的工作选择了基于预训练CLIP的模型进行视频检索，并将其作为特征提取器，用于解决其他三个具有挑战性的任务，在多任务学习框架下进行。然而，这项工作在学习用户偏好的全面认知方面存在困难，因为忽略了不同模态之间的层次关系和关联关系。在本文中，基于浅层到深层的原则，我们提出一种以查询为中心的视听认知（QUAG）网络，以构建一个可靠多模态表示，用于时刻检索、时刻分割和步骤字幕生成。具体而言，我们首先设计模态协同感知，通过建模全局对比对齐和局部细粒度视听互动作，获取丰富的视听内容。然后，我们设计一种以查询为中心的认知方式，通过深层次查询对浅层视听表示进行时间-通道过滤，以认知用户偏好内容，并因此实现以查询为中心的视听表示，用于三个任务。广泛实验表明，QUAG在HIREST上达到了最先进的性能。此外，我们在基于查询的视频摘要任务上测试QUAG，验证了其良好的泛化能力。 

---
# Tuning Music Education: AI-Powered Personalization in Learning Music 

**Title (ZH)**: 调和音乐教育：基于AI的个性化音乐学习 

**Authors**: Mayank Sanganeria, Rohan Gala  

**Link**: [PDF](https://arxiv.org/pdf/2412.13514)  

**Abstract**: Recent AI-driven step-function advances in several longstanding problems in music technology are opening up new avenues to create the next generation of music education tools. Creating personalized, engaging, and effective learning experiences are continuously evolving challenges in music education. Here we present two case studies using such advances in music technology to address these challenges. In our first case study we showcase an application that uses Automatic Chord Recognition to generate personalized exercises from audio tracks, connecting traditional ear training with real-world musical contexts. In the second case study we prototype adaptive piano method books that use Automatic Music Transcription to generate exercises at different skill levels while retaining a close connection to musical interests. These applications demonstrate how recent AI developments can democratize access to high-quality music education and promote rich interaction with music in the age of generative AI. We hope this work inspires other efforts in the community, aimed at removing barriers to access to high-quality music education and fostering human participation in musical expression. 

**Abstract (ZH)**: 近年来，人工智能（AI）驱动的步进函数进步正在音乐技术中解决一些长期存在的问题，开辟了创造下一代音乐教育工具的新途径。在音乐教育中，为学生提供个性化、互动性强且有效的学习体验是一个不断演变的挑战。本文将展示两个案例研究，利用这些音乐技术的进展来应对这些挑战。在我们的第一个案例研究中，我们展示了一个应用程序，该应用程序利用自动和弦识别从音频片段中生成个性化练习，将传统的听觉训练与实际的音乐背景联系起来。在第二个案例研究中，我们构建了使用自动音乐转录生成不同技能水平练习的适应性钢琴方法书，同时保持与音乐兴趣的紧密联系。这些应用表明，近期的AI发展如何可以促进高质量音乐教育的普及，并在生成人工智能时代促进与音乐的丰富互动。我们希望这项工作能够激励社区中的其他努力，旨在消除高质量音乐教育的障碍，并鼓励人类在音乐表达中的参与。 

---
# VaeDiff-DocRE: End-to-end Data Augmentation Framework for Document-level Relation Extraction 

**Title (ZH)**: VaeDiff-DocRE：文档级别关系抽取的端到端数据增强框架 

**Authors**: Khai Phan Tran, Wen Hua, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13503)  

**Abstract**: Document-level Relation Extraction (DocRE) aims to identify relationships between entity pairs within a document. However, most existing methods assume a uniform label distribution, resulting in suboptimal performance on real-world, imbalanced datasets. To tackle this challenge, we propose a novel data augmentation approach using generative models to enhance data from the embedding space. Our method leverages the Variational Autoencoder (VAE) architecture to capture all relation-wise distributions formed by entity pair representations and augment data for underrepresented relations. To better capture the multi-label nature of DocRE, we parameterize the VAE's latent space with a Diffusion Model. Additionally, we introduce a hierarchical training framework to integrate the proposed VAE-based augmentation module into DocRE systems. Experiments on two benchmark datasets demonstrate that our method outperforms state-of-the-art models, effectively addressing the long-tail distribution problem in DocRE. 

**Abstract (ZH)**: 文档级关系提取（DocRE）旨在识别文档内实体对之间的关系。然而，现有的大多数方法假设标签分布均匀，导致在真实世界的不平衡数据集上表现不佳。为了解决这一挑战，我们提出了一种使用生成模型从嵌入空间增强数据的新数据增强方法。该方法利用变分自编码器（VAE）架构捕获实体对表示形成的所有关系分布，并增强少数关系的数据。为了更好地捕捉DocRE的多标签特性，我们用扩散模型参数化VAE的隐空间。此外，我们引入了一种分层训练框架，将提出的基于VAE的增强模块整合到DocRE系统中。在两个基准数据集上的实验表明，我们的方法在DocRE长尾分布问题上优于最先进的模型。 

---
# Federated t-SNE and UMAP for Distributed Data Visualization 

**Title (ZH)**: 联邦t-SNE和UMAP在分布式数据可视化中的应用 

**Authors**: Dong Qiao, Xinxian Ma, Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2412.13495)  

**Abstract**: High-dimensional data visualization is crucial in the big data era and these techniques such as t-SNE and UMAP have been widely used in science and engineering. Big data, however, is often distributed across multiple data centers and subject to security and privacy concerns, which leads to difficulties for the standard algorithms of t-SNE and UMAP. To tackle the challenge, this work proposes Fed-tSNE and Fed-UMAP, which provide high-dimensional data visualization under the framework of federated learning, without exchanging data across clients or sending data to the central server. The main idea of Fed-tSNE and Fed-UMAP is implicitly learning the distribution information of data in a manner of federated learning and then estimating the global distance matrix for t-SNE and UMAP. To further enhance the protection of data privacy, we propose Fed-tSNE+ and Fed-UMAP+. We also extend our idea to federated spectral clustering, yielding algorithms of clustering distributed data. In addition to these new algorithms, we offer theoretical guarantees of optimization convergence, distance and similarity estimation, and differential privacy. Experiments on multiple datasets demonstrate that, compared to the original algorithms, the accuracy drops of our federated algorithms are tiny. 

**Abstract (ZH)**: 在大数据时代，高维数据可视化至关重要，而t-SNE和UMAP等多种技术已被广泛应用于科学和工程领域。然而，大数据往往分布在多个数据中心，并且存在安全性和隐私性问题，这给t-SNE和UMAP的标准算法带来了挑战。为了解决这一难题，本研究提出了Fed-tSNE和Fed-UMAP，它们在联邦学习框架下提供了高维数据可视化的方法，而无需在客户端之间交换数据或将数据发送到中央服务器。Fed-tSNE和Fed-UMAP的核心思想是在联邦学习的原则下隐式学习数据分布信息，并据此估计全局距离矩阵，进而应用于t-SNE和UMAP。为进一步增强数据隐私保护，我们提出了Fed-tSNE+和Fed-UMAP+。此外，我们将该思想扩展到联邦光谱聚类，生成分布式数据聚类算法。除了这些新算法外，我们还提供了关于优化收敛性、距离和相似性估计以及差分隐私的理论保证。实验结果表明，与原始算法相比，联邦算法的准确性下降微乎其微。 

---
# Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models 

**Title (ZH)**: 精炼注意力感知稀疏微调策略以优化语言模型 

**Authors**: Xinxin Liu, Aaron Thomas, Cheng Zhang, Jianyi Cheng, Yiren Zhao, Xitong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13488)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has gained prominence through low-rank adaptation methods like LoRA. In this paper, we focus on sparsity-based PEFT (SPEFT), which introduces trainable sparse adaptations to the weight matrices in the model, offering greater flexibility in selecting fine-tuned parameters compared to low-rank methods. We conduct the first systematic evaluation of salience metrics for SPEFT, inspired by zero-cost NAS proxies, and identify simple gradient-based metrics is reliable, and results are on par with the best alternatives, offering both computational efficiency and robust performance. Additionally, we compare static and dynamic masking strategies, finding that static masking, which predetermines non-zero entries before training, delivers efficiency without sacrificing performance, while dynamic masking offers no substantial benefits. Across NLP tasks, a simple gradient-based, static SPEFT consistently outperforms other fine-tuning methods for LLMs, providing a simple yet effective baseline for SPEFT. Our work challenges the notion that complexity is necessary for effective PEFT. Our work is open source and available to the community at [this https URL]. 

**Abstract (ZH)**: Parameter-Efficient Fine-Tuning (PEFT) 通过低秩适应方法（如LoRA）逐渐崭露头角。本文我们重点关注基于稀疏性的PEFT（SPEFT），它通过引入可训练的稀疏适应于模型的权重矩阵，在选择微调参数方面提供了比低秩方法更高的灵活性。我们首次系统地评估了SPEFT的显著性度量方法，受到零成本NAS代理的启发，发现基于梯度的简单度量方法是可靠的，其结果与最佳替代方案相当，并且在计算效率和稳健性能方面表现出色。此外，我们比较了静态和动态掩蔽策略，发现静态掩蔽策略在训练前预定义非零条目，能够在不牺牲性能的情况下提供效率，而动态掩蔽策略则没有显著优势。在NLP任务中，一种简单的基于梯度的静态SPEFT方法在对大型语言模型（LLM）进行微调时始终优于其他方法，为SPEFT提供了简单而有效的基准。我们的工作挑战了复杂性对于有效PEFT是必需的这一观点。我们的工作是开源的，并可从 [此处] 获得。 

---
# Generating Unseen Nonlinear Evolution in Sea Surface Temperature Using a Deep Learning-Based Latent Space Data Assimilation Framework 

**Title (ZH)**: 使用基于深层学习潜在空间数据同化框架生成未见的海表面温度非线性演变 

**Authors**: Qingyu Zheng, Guijun Han, Wei Li, Lige Cao, Gongfu Zhou, Haowen Wu, Qi Shao, Ru Wang, Xiaobo Wu, Xudong Cui, Hong Li, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13477)  

**Abstract**: Advances in data assimilation (DA) methods have greatly improved the accuracy of Earth system predictions. To fuse multi-source data and reconstruct the nonlinear evolution missing from observations, geoscientists are developing future-oriented DA methods. In this paper, we redesign a purely data-driven latent space DA framework (DeepDA) that employs a generative artificial intelligence model to capture the nonlinear evolution in sea surface temperature. Under variational constraints, DeepDA embedded with nonlinear features can effectively fuse heterogeneous data. The results show that DeepDA remains highly stable in capturing and generating nonlinear evolutions even when a large amount of observational information is missing. It can be found that when only 10% of the observation information is available, the error increase of DeepDA does not exceed 40%. Furthermore, DeepDA has been shown to be robust in the fusion of real observations and ensemble simulations. In particular, this paper provides a mechanism analysis of the nonlinear evolution generated by DeepDA from the perspective of physical patterns, which reveals the inherent explainability of our DL model in capturing multi-scale ocean signals. 

**Abstract (ZH)**: 数据同化（DA）方法的进步极大地提高了地球系统预测的准确性。为了融合多源数据并重构观测数据所缺失的非线性演变，地球科学家们正在开发面向未来的DA方法。本文重新设计了一个纯数据驱动的隐空间DA框架（DeepDA），该框架利用生成人工智能模型来捕捉表层海温的非线性演变。在变分约束下，DeepDA结合了非线性特征，可以有效地融合异质性数据。实验结果表明，即使大量观测信息缺失，DeepDA在捕捉和生成非线性演变方面依然表现出极高的稳定性。研究发现，当可用的观测信息仅为10%时，DeepDA的误差增加不超过40%。此外，DeepDA在实观测数据和集合模拟的融合中表现出鲁棒性。特别是，本文从物理模式的角度分析了DeepDA生成的非线性演变的机理，揭示了我们的深度学习模型在捕捉多尺度海洋信号方面的固有可解释性。 

---
# A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models 

**Title (ZH)**: 大型语言模型中成员推断攻击的统计与多视角再审视 

**Authors**: Bowen Chen, Namgi Han, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13475)  

**Abstract**: The lack of data transparency in Large Language Models (LLMs) has highlighted the importance of Membership Inference Attack (MIA), which differentiates trained (member) and untrained (non-member) data. Though it shows success in previous studies, recent research reported a near-random performance in different settings, highlighting a significant performance inconsistency. We assume that a single setting doesn't represent the distribution of the vast corpora, causing members and non-members with different distributions to be sampled and causing inconsistency. In this study, instead of a single setting, we statistically revisit MIA methods from various settings with thousands of experiments for each MIA method, along with study in text feature, embedding, threshold decision, and decoding dynamics of members and non-members. We found that (1) MIA performance improves with model size and varies with domains, while most methods do not statistically outperform baselines, (2) Though MIA performance is generally low, a notable amount of differentiable member and non-member outliers exists and vary across MIA methods, (3) Deciding a threshold to separate members and non-members is an overlooked challenge, (4) Text dissimilarity and long text benefit MIA performance, (5) Differentiable or not is reflected in the LLM embedding, (6) Member and non-members show different decoding dynamics. 

**Abstract (ZH)**: 大型语言模型（LLM）数据透明度的缺乏突显了成员推断攻击（MIA，Membership Inference Attack）的重要性，该攻击能够区分训练数据（成员数据）和未训练数据（非成员数据）。尽管以前的研究显示出成功，但最近的研究在不同设置下的效果接近随机，突显了性能的一致性问题。我们认为单一的设置无法代表庞大的语料库分布，导致具有不同分布的成员数据和非成员数据被抽样，从而引起性能的不一致性。在本研究中，我们没有采用单一设置，而是从多个设置出发，通过数千次实验重新审视MIA方法，同时在文本特征、嵌入、阈值决策以及成员和非成员的解码动态方面进行详细研究。研究发现如下：
1. MIA性能随模型规模的增加而提高，并在不同领域有所不同，但大多数方法在统计上并未明显优于基线；
2. 尽管MIA的整体性能较低，但在不同MIA方法中仍然存在显著的可区分成员和非成员异常值；
3. 决定用于区分成员和非成员的阈值是一个被忽视的挑战；
4. 文本的相似性差异和长文本有利于MIA性能；
5. 可区分性或不可区分性在LLM嵌入中体现；
6. 成员和非成员表现出不同的解码动态。 

---
# Transducer Tuning: Efficient Model Adaptation for Software Tasks Using Code Property Graphs 

**Title (ZH)**: 转换模型调整：使用代码属性图进行软件任务的高效模型适应 

**Authors**: Imam Nur Bani Yusuf, Lingxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13467)  

**Abstract**: Large language models have demonstrated promising performance across various software engineering tasks. While fine-tuning is a common practice to adapt these models for downstream tasks, it becomes challenging in resource-constrained environments due to increased memory requirements from growing trainable parameters in increasingly large language models. We introduce \approach, a technique to adapt large models for downstream code tasks using Code Property Graphs (CPGs). Our approach introduces a modular component called \transducer that enriches code embeddings with structural and dependency information from CPGs. The Transducer comprises two key components: Graph Vectorization Engine (GVE) and Attention-Based Fusion Layer (ABFL). GVE extracts CPGs from input source code and transforms them into graph feature vectors. ABFL then fuses those graphs feature vectors with initial code embeddings from a large language model. By optimizing these transducers for different downstream tasks, our approach enhances the models without the need to fine-tune them for specific tasks. We have evaluated \approach on three downstream tasks: code summarization, assert generation, and code translation. Our results demonstrate competitive performance compared to full parameter fine-tuning while reducing up to 99\% trainable parameters to save memory. \approach also remains competitive against other fine-tuning approaches (e.g., LoRA, Prompt-Tuning, Prefix-Tuning) while using only 1.5\%-80\% of their trainable parameters. Our findings show that integrating structural and dependency information through Transducer Tuning enables more efficient model adaptation, making it easier for users to adapt large models in resource-constrained settings. 

**Abstract (ZH)**: 大规模语言模型在各种软件工程任务中已经展现出令人鼓舞的性能。虽然微调是将这些模型适应下游任务的一种常见做法，但在资源受限的环境中，由于大型语言模型可训练参数的增长导致内存需求增加，使微调变得更具挑战性。我们提出了一种名为\approach的技术，通过代码属性图（CPG）来适应大规模模型以用于下游代码任务。该方法引入了一个可模块化组件\transducer，通过从CPG中提取结构和依赖信息来丰富代码嵌入。\transducer包括两个关键组件：图向量化引擎（GVE）和注意力融合层（ABFL）。GVE从输入源代码中提取CPG，并将其转换为图特征向量。ABFL随后将这些图特征向量与大型语言模型的初始代码嵌入融合。通过为不同的下游任务优化这些转换器，我们的方法可以在不需要特定任务微调的情况下增强模型。我们在三个下游任务上评估了该方法：代码摘要、断言生成和代码翻译。我们的结果表明，在减少高达99%的可训练参数以节省内存的同时，该方法能够与全面的参数微调相比表现出竞争力。此外，在使用仅为其他微调方法（例如LoRA、提示微调、前缀微调）1.5%-80%的可训练参数的情况下，\approach仍然能够保持竞争力。我们的研究结果表明，通过转换器调优整合结构和依赖信息，可以使模型适应更加高效，从而简化在资源受限环境中对大型模型的适应过程。 

---
# FlexPose: Pose Distribution Adaptation with Limited Guidance 

**Title (ZH)**: FlexPose：在有限指导下的姿态分布适应 

**Authors**: Zixiao Wang, Junwu Weng, Mengyuan Liu, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13463)  

**Abstract**: Numerous well-annotated human key-point datasets are publicly available to date. However, annotating human poses for newly collected images is still a costly and time-consuming progress. Pose distributions from different datasets share similar pose hinge-structure priors with different geometric transformations, such as pivot orientation, joint rotation, and bone length ratio. The difference between Pose distributions is essentially the difference between the transformation distributions. Inspired by this fact, we propose a method to calibrate a pre-trained pose generator in which the pose prior has already been learned to an adapted one following a new pose distribution. We treat the representation of human pose joint coordinates as skeleton image and transfer a pre-trained pose annotation generator with only a few annotation guidance. By fine-tuning a limited number of linear layers that closely related to the pose transformation, the adapted generator is able to produce any number of pose annotations that are similar to the target poses. We evaluate our proposed method, FlexPose, on several cross-dataset settings both qualitatively and quantitatively, which demonstrates that our approach achieves state-of-the-art performance compared to the existing generative-model-based transfer learning methods when given limited annotation guidance. 

**Abstract (ZH)**: 到目前为止，公开可用的标注良好人体关键点数据集数量众多。然而，为新收集的图像标注人体姿态仍然是一个耗费成本和耗时的过程。不同数据集的姿态分布共享相似的姿态枢纽结构先验，这些先验在不同的几何变换下（如枢转方向、关节旋转和骨骼长度比例）存在差异。姿态分布之间的差异本质上是变换分布之间的差异。受到这一事实的启发，我们提出了一种方法，将预先训练好的姿态生成器校准为适应新的姿态分布的模型。我们将人体姿态关节坐标的表现形式视为骨架图像，并仅通过少量标注指导的方式转移一个预先训练好的姿态标注生成器。通过微调与姿态变换密切相关的少量线性层，适应后的生成器能够生成与目标姿态相似的任意数量的姿态标注。我们通过对几个跨数据集设置进行定性和定量的评估，证明了我们的方法FlexPose在给出少量标注指导的情况下，相较于现有的基于生成模型的迁移学习方法实现了最先进的性能。 

---
# Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection 

**Title (ZH)**: 内部寻踪：面向3D异常检测的内部空间模态感知 

**Authors**: Hanzhe Liang, Guoyang Xie, Chengbin Hou, Bingshu Wang, Can Gao, Jinbao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13461)  

**Abstract**: 3D anomaly detection has recently become a significant focus in computer vision. Several advanced methods have achieved satisfying anomaly detection performance. However, they typically concentrate on the external structure of 3D samples and struggle to leverage the internal information embedded within samples. Inspired by the basic intuition of why not look inside for more, we introduce a straightforward method named Internal Spatial Modality Perception (ISMP) to explore the feature representation from internal views fully. Specifically, our proposed ISMP consists of a critical perception module, Spatial Insight Engine (SIE), which abstracts complex internal information of point clouds into essential global features. Besides, to better align structural information with point data, we propose an enhanced key point feature extraction module for amplifying spatial structure feature representation. Simultaneously, a novel feature filtering module is incorporated to reduce noise and redundant features for further aligning precise spatial structure. Extensive experiments validate the effectiveness of our proposed method, achieving object-level and pixel-level AUROC improvements of 4.2% and 13.1%, respectively, on the Real3D-AD benchmarks. Note that the strong generalization ability of SIE has been theoretically proven and is verified in both classification and segmentation tasks. 

**Abstract (ZH)**: 三维异常检测近年来已成为计算机视觉领域的研究重点。多种先进的方法已实现了令人满意的异常检测性能。然而，它们通常侧重于三维样本的外部结构，难以充分利用样本内部嵌入的信息。受为何不深入内部获取更多信息的基本直觉启发，我们提出了一种称为内部空间模态感知（ISMP）的简单方法，以全面探索内部视图的特征表示。具体而言，我们提出的方法ISMP包含一个关键感知模块——空间洞察引擎（SIE），该模块将点云中的复杂内部信息抽象为重要的全局特征。此外，为了更好地使结构信息与点数据对齐，我们提出了一种增强的关键点特征提取模块，以增强空间结构特征表示。同时，我们引入了一种新的特征筛选模块，以降低噪声和冗余特征，进一步实现精确的空间结构对齐。广泛的实验验证了我们提出的方法的有效性，在Real3D-AD基准数据集上实现了对象级和像素级AUROC改进，分别提高了4.2%和13.1%。值得注意的是，SIE的强泛化能力已在理论层面得到证明，并已在分类和分割任务中得到验证。 

---
# Pre-training a Density-Aware Pose Transformer for Robust LiDAR-based 3D Human Pose Estimation 

**Title (ZH)**: 为提高基于激光雷达（LiDAR）的3D人体姿态估计的鲁棒性，预训练一种密度aware的姿态变换器 

**Authors**: Xiaoqi An, Lin Zhao, Chen Gong, Jun Li, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13454)  

**Abstract**: With the rapid development of autonomous driving, LiDAR-based 3D Human Pose Estimation (3D HPE) is becoming a research focus. However, due to the noise and sparsity of LiDAR-captured point clouds, robust human pose estimation remains challenging. Most of the existing methods use temporal information, multi-modal fusion, or SMPL optimization to correct biased results. In this work, we try to obtain sufficient information for 3D HPE only by modeling the intrinsic properties of low-quality point clouds. Hence, a simple yet powerful method is proposed, which provides insights both on modeling and augmentation of point clouds. Specifically, we first propose a concise and effective density-aware pose transformer (DAPT) to get stable keypoint representations. By using a set of joint anchors and a carefully designed exchange module, valid information is extracted from point clouds with different densities. Then 1D heatmaps are utilized to represent the precise locations of the keypoints. Secondly, a comprehensive LiDAR human synthesis and augmentation method is proposed to pre-train the model, enabling it to acquire a better human body prior. We increase the diversity of point clouds by randomly sampling human positions and orientations and by simulating occlusions through the addition of laser-level masks. Extensive experiments have been conducted on multiple datasets, including IMU-annotated LidarHuman26M, SLOPER4D, and manually annotated Waymo Open Dataset v2.0 (Waymo), HumanM3. Our method demonstrates SOTA performance in all scenarios. In particular, compared with LPFormer on Waymo, we reduce the average MPJPE by $10.0mm$. Compared with PRN on SLOPER4D, we notably reduce the average MPJPE by $20.7mm$. 

**Abstract (ZH)**: 随着自动驾驶的迅速发展，基于LiDAR的3D人类姿态估计（3D HPE）已成为研究重点。然而，由于LiDAR捕获点云的噪声和稀疏性，稳健的人类姿态估计仍具挑战性。现有的大多数方法通过使用时间信息、多模态融合或SMPL优化来修正偏差结果。在本工作中，我们尝试仅通过建模低质量点云的内在属性来获得足够的信息以进行3D HPE。因此，我们提出了一种简单而强大的方法，该方法在建模和增强点云方面提供了有价值的见解。具体而言，我们首先提出了一种简洁且有效的密度感知姿态变换器（DAPT），以获得稳定的关键点表示。通过使用一组关节锚点和精心设计的交换模块，从不同密度的点云中提取有效信息。然后使用1D热图来表示关键点的精确位置。其次，我们提出了一种全面的LiDAR人类合成与增强方法，以进行预训练，使模型能够获得更好的人体先验知识。我们通过随机采样人类位置和姿态，并通过添加激光级遮挡模拟遮挡，增加了点云的多样性。我们在多个数据集上进行了广泛的实验，包括带有imu标注的LidarHuman26M、SLOPER4D和手动标注的Waymo Open Dataset v2.0（Waymo）以及HumanM3。我们的方法在所有场景中均表现出SOTA性能。特别是在与Waymo上的LPFormer相比时，我们减少了平均MPJPE 10.0mm。与SLOPER4D上的PRN相比，我们显著减少了平均MPJPE 20.7mm。 

---
# ConDo: Continual Domain Expansion for Absolute Pose Regression 

**Title (ZH)**: 连续域扩展用于绝对姿态回归：ConDo 

**Authors**: Zijun Li, Zhipeng Cai, Bochun Yang, Xuelun Shen, Siqi Shen, Xiaoliang Fan, Michael Paulitsch, Cheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13452)  

**Abstract**: Visual localization is a fundamental machine learning problem. Absolute Pose Regression (APR) trains a scene-dependent model to efficiently map an input image to the camera pose in a pre-defined scene. However, many applications have continually changing environments, where inference data at novel poses or scene conditions (weather, geometry) appear after deployment. Training APR on a fixed dataset leads to overfitting, making it fail catastrophically on challenging novel data. This work proposes Continual Domain Expansion (ConDo), which continually collects unlabeled inference data to update the deployed APR. Instead of applying standard unsupervised domain adaptation methods which are ineffective for APR, ConDo effectively learns from unlabeled data by distilling knowledge from scene-agnostic localization methods. By sampling data uniformly from historical and newly collected data, ConDo can effectively expand the generalization domain of APR. Large-scale benchmarks with various scene types are constructed to evaluate models under practical (long-term) data changes. ConDo consistently and significantly outperforms baselines across architectures, scene types, and data changes. On challenging scenes (Fig.1), it reduces the localization error by >7x (14.8m vs 1.7m). Analysis shows the robustness of ConDo against compute budgets, replay buffer sizes and teacher prediction noise. Comparing to model re-training, ConDo achieves similar performance up to 25x faster. 

**Abstract (ZH)**: 视觉定位是基本的机器学习问题。绝对姿态回归（APR）训练一种场景依存模型，能够高效地将输入图像映射到预定义场景中的相机姿态。然而，许多应用中的环境是不断变化的，部署后会出现新的姿态或场景条件（天气、几何形状），导致预测数据不再适用于初始训练的数据集。在固定数据集上训练APR会导致过拟合，使其在具有挑战性的新型数据上表现不佳。本文提出了一种持续领域扩展（ConDo）方法，该方法持续收集未标注的推理数据并更新部署的APR模型。ConDo 不采用标准的无监督域适应方法（这些方法对APR效果不佳），而是通过从场景无关的定位方法中提取知识有效学习未标注数据。通过均匀地从历史数据和新收集的数据中采样数据，ConDo 能够有效地扩展APR的泛化范围。本文构建了包含多种场景类型的大型基准测试，以评估模型在实际（长期）数据变化下的表现。ConDo 在各种架构、场景类型和数据变化下均一致且显著优于基线方法。在具有挑战性的场景中（如图1所示），ConDo 将定位误差降低了超过7倍（从14.8米降至1.7米）。分析显示ConDo 对于计算预算、回放缓冲区大小和教师预测噪声具有鲁棒性。与模型重新训练相比，ConDo 的性能相当，但速度快了25倍。 

---
# Toward an Insider Threat Education Platform: A Theoretical Literature Review 

**Title (ZH)**: 面向内幕威胁教育平台的理论文献综述 

**Authors**: Haywood Gelman, John D. Hastings, David Kenley, Eleanor Loiacono  

**Link**: [PDF](https://arxiv.org/pdf/2412.13446)  

**Abstract**: Insider threats (InTs) within organizations are small in number but have a disproportionate ability to damage systems, information, and infrastructure. Existing InT research studies the problem from psychological, technical, and educational perspectives. Proposed theories include research on psychological indicators, machine learning, user behavioral log analysis, and educational methods to teach employees recognition and mitigation techniques. Because InTs are a human problem, training methods that address InT detection from a behavioral perspective are critical. While numerous technological and psychological theories exist on detection, prevention, and mitigation, few training methods prioritize psychological indicators. This literature review studied peer-reviewed, InT research organized by subtopic and extracted critical theories from psychological, technical, and educational disciplines. In doing so, this is the first study to comprehensively organize research across all three approaches in a manner which properly informs the development of an InT education platform. 

**Abstract (ZH)**: 组织内部的威胁（InTs）虽然数量较少，但对系统、信息和基础设施的破坏能力却是不成比例的。现有针对InT的研究从心理学、技术和教育等多个角度探讨了该问题。提出的理论包括心理指标研究、机器学习、用户行为日志分析以及教育方法来教授员工识别和减轻技术。由于InTs本质上是一个人力资源问题，因此从行为角度进行检测的培训方法至关重要。虽然存在大量关于检测、预防和减轻的技术和心理理论，但很少有培训方法侧重于心理指标。本文综述了经过同行评审的InT研究，并按照子话题组织了相关内容，提取了来自心理学、技术和教育领域的关键理论。通过对这三个角度的研究进行综合整理，本文是首个全面组织跨学科研究成果的研究，以正确指导内部威胁教育平台的开发。 

---
# Communication-Efficient Personalized Federal Graph Learning via Low-Rank Decomposition 

**Title (ZH)**: 通过低秩分解实现的通信高效个性化联邦图学习 

**Authors**: Ruyue Liu, Rong Yin, Xiangzhen Bo, Xiaoshuai Hao, Xingrui Zhou, Yong Liu, Can Ma, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13442)  

**Abstract**: Federated graph learning (FGL) has gained significant attention for enabling heterogeneous clients to process their private graph data locally while interacting with a centralized server, thus maintaining privacy. However, graph data on clients are typically non-IID, posing a challenge for a single model to perform well across all clients. Another major bottleneck of FGL is the high cost of communication. To address these challenges, we propose a communication-efficient personalized federated graph learning algorithm, CEFGL. Our method decomposes the model parameters into low-rank generic and sparse private models. We employ a dual-channel encoder to learn sparse local knowledge in a personalized manner and low-rank global knowledge in a shared manner. Additionally, we perform multiple local stochastic gradient descent iterations between communication phases and integrate efficient compression techniques into the algorithm. The advantage of CEFGL lies in its ability to capture common and individual knowledge more precisely. By utilizing low-rank and sparse parameters along with compression techniques, CEFGL significantly reduces communication complexity. Extensive experiments demonstrate that our method achieves optimal classification accuracy in a variety of heterogeneous environments across sixteen datasets. Specifically, compared to the state-of-the-art method FedStar, the proposed method (with GIN as the base model) improves accuracy by 5.64\% on cross-datasets setting CHEM, reduces communication bits by a factor of 18.58, and reduces the communication time by a factor of 1.65. 

**Abstract (ZH)**: 联邦图学习（FGL）因其能够使异构客户端在其本地处理私有图数据的同时与中央服务器交互，从而维持隐私性而引起了广泛关注。然而，客户端的图数据通常是非IID的，这给单一模型在所有客户端上表现良好带来了挑战。另一个主要瓶颈是FGL中的高通信成本。为了解决这些挑战，我们提出了一种高效的个性化联邦图学习算法，CEFGL。该方法将模型参数分解为低秩通用模型和稀疏私有模型。我们使用双通道编码器来学习以个性化方式的稀疏局部知识以及以共享方式的低秩全局知识。此外，我们还在通信阶段间执行多次本地随机梯度下降迭代，并将高效的压缩技术集成到算法中。CEFGL的优势在于其能够更精确地捕捉共同和个体知识。通过使用低秩和稀疏参数以及压缩技术，CEFGL显著降低了通信复杂度。通过广泛实验，我们发现该方法在十六个数据集上在多种异构环境中实现了最优分类精度。具体而言，与最先进的方法FedStar相比，基于GIN模型的所提出方法在跨数据集设置CHEM中的准确率提高了5.64%，通信比特减少了18.58倍，通信时间减少了1.65倍。 

---
# FlashVTG: Feature Layering and Adaptive Score Handling Network for Video Temporal Grounding 

**Title (ZH)**: FlashVTG：特征分层与自适应评分处理网络用于视频时序定位 

**Authors**: Zhuo Cao, Bingqing Zhang, Heming Du, Xin Yu, Xue Li, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13441)  

**Abstract**: Text-guided Video Temporal Grounding (VTG) aims to localize relevant segments in untrimmed videos based on textual descriptions, encompassing two subtasks: Moment Retrieval (MR) and Highlight Detection (HD). Although previous typical methods have achieved commendable results, it is still challenging to retrieve short video moments. This is primarily due to the reliance on sparse and limited decoder queries, which significantly constrain the accuracy of predictions. Furthermore, suboptimal outcomes often arise because previous methods rank predictions based on isolated predictions, neglecting the broader video context. To tackle these issues, we introduce FlashVTG, a framework featuring a Temporal Feature Layering (TFL) module and an Adaptive Score Refinement (ASR) module. The TFL module replaces the traditional decoder structure to capture nuanced video content variations across multiple temporal scales, while the ASR module improves prediction ranking by integrating context from adjacent moments and multi-temporal-scale features. Extensive experiments demonstrate that FlashVTG achieves state-of-the-art performance on four widely adopted datasets in both MR and HD. Specifically, on the QVHighlights dataset, it boosts mAP by 5.8% for MR and 3.3% for HD. For short-moment retrieval, FlashVTG increases mAP to 125% of previous SOTA performance. All these improvements are made without adding training burdens, underscoring its effectiveness. Our code is available at this https URL. 

**Abstract (ZH)**: 基于文本指导的视频时序定位（VTG）旨在根据文本描述在未裁剪的视频中定位相关段落，涵盖两个子任务：时刻检索（MR）和高光检测（HD）。尽管以往的典型方法已经取得了显著成果，但在检索短视频片段方面仍然存在挑战。这主要归因于对稀疏且有限的解码器查询的依赖，这显著限制了预测的准确性。此外，由于先前的方法基于孤立的预测进行排名，忽略了更广泛的视频背景，从而导致次优结果。为了应对这些问题，我们提出了FlashVTG框架，该框架包括一个时间特征层化（TFL）模块和一个自适应评分精炼（ASR）模块。TFL模块取代了传统的解码器结构，以捕捉多时间尺度下的微小视频内容变化，而ASR模块通过结合相邻时刻和多时间尺度特征来改进预测排名。广泛的经验表明，FlashVTG在四个广泛采用的MR和HD数据集上都实现了最先进的性能。具体来说，FlashVTG在QVHighlights数据集上，将MR的mAP提高了5.8%，HD的mAP提高了3.3%。对于短片段检索，FlashVTG将mAP提升至前SOTA性能的125%。所有这些改进都未增加训练负担，突显了其有效性。我们的代码可在下面的网址获取：[这里请插入具体的网址，例如：https://github.com/AntoninAbend/FlashVTG] 

---
# Deploying Foundation Model Powered Agent Services: A Survey 

**Title (ZH)**: 基于基础模型的代理服务部署：一个综述 

**Authors**: Wenchao Xu, Jinyu Chen, Peirong Zheng, Xiaoquan Yi, Tianyi Tian, Wenhui Zhu, Quan Wan, Haozhao Wang, Yunfeng Fan, Qinliang Su, Xuemin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2412.13437)  

**Abstract**: Foundation model (FM) powered agent services are regarded as a promising solution to develop intelligent and personalized applications for advancing toward Artificial General Intelligence (AGI). To achieve high reliability and scalability in deploying these agent services, it is essential to collaboratively optimize computational and communication resources, thereby ensuring effective resource allocation and seamless service delivery. In pursuit of this vision, this paper proposes a unified framework aimed at providing a comprehensive survey on deploying FM-based agent services across heterogeneous devices, with the emphasis on the integration of model and resource optimization to establish a robust infrastructure for these services. Particularly, this paper begins with exploring various low-level optimization strategies during inference and studies approaches that enhance system scalability, such as parallelism techniques and resource scaling methods. The paper then discusses several prominent FMs and investigates research efforts focused on inference acceleration, including techniques such as model compression and token reduction. Moreover, the paper also investigates critical components for constructing agent services and highlights notable intelligent applications. Finally, the paper presents potential research directions for developing real-time agent services with high Quality of Service (QoS). 

**Abstract (ZH)**: 基于基础模型（Foundation Model, FM）的代理服务被认为是一种有潜力的发展面向通用人工智能（Artificial General Intelligence, AGI）的智能和个性化应用的解决方案。为了在部署这些代理服务时实现高可靠性和可扩展性，必须协同优化计算和通信资源，以确保有效的资源分配和无缝的服务交付。为实现这一愿景，本文提出了一种统一框架，旨在提供一个关于在异构设备上部署基于FM的代理服务的全面综述，重点在于模型和资源优化的整合，以建立这些服务的稳健基础设施。特别地，本文首先探讨了推理阶段的各种低层次优化策略，并研究了提高系统可扩展性的方法，如并行技术及资源扩展方法。接着，讨论了几种具有代表性的基础模型，并调查了围绕推理加速的研究努力，包括模型压缩和标记减少等技术。此外，本文还研究了构建代理服务的关键组件，并突出了重要的人工智能应用。最后，本文提出了开发实时代理服务的潜在研究方向，以确保高质量的服务（Quality of Service, QoS）。 

---
# Lightweight Safety Classification Using Pruned Language Models 

**Title (ZH)**: 使用剪枝语言模型的轻量级安全性分类 

**Authors**: Mason Sawtell, Tula Masterman, Sandi Besen, Jim Brown  

**Link**: [PDF](https://arxiv.org/pdf/2412.13435)  

**Abstract**: In this paper, we introduce a novel technique for content safety and prompt injection classification for Large Language Models. Our technique, Layer Enhanced Classification (LEC), trains a Penalized Logistic Regression (PLR) classifier on the hidden state of an LLM's optimal intermediate transformer layer. By combining the computational efficiency of a streamlined PLR classifier with the sophisticated language understanding of an LLM, our approach delivers superior performance surpassing GPT-4o and special-purpose models fine-tuned for each task. We find that small general-purpose models (Qwen 2.5 sizes 0.5B, 1.5B, and 3B) and other transformer-based architectures like DeBERTa v3 are robust feature extractors allowing simple classifiers to be effectively trained on fewer than 100 high-quality examples. Importantly, the intermediate transformer layers of these models typically outperform the final layer across both classification tasks. Our results indicate that a single general-purpose LLM can be used to classify content safety, detect prompt injections, and simultaneously generate output tokens. Alternatively, these relatively small LLMs can be pruned to the optimal intermediate layer and used exclusively as robust feature extractors. Since our results are consistent on different transformer architectures, we infer that robust feature extraction is an inherent capability of most, if not all, LLMs. 

**Abstract (ZH)**: 在本文中，我们介绍了一种新颖的技术，用于大型语言模型的内容安全和提示注入分类。我们的技术名为层增强分类（LEC，Layer Enhanced Classification），它在大型语言模型最优中间变换器层的隐藏状态下训练了一个正则化逻辑回归（PLR，Penalized Logistic Regression）分类器。通过结合简化后的PLR分类器的计算效率和大型语言模型复杂的语言理解能力，我们的方法在分类性能上优于GPT-4o和为每个任务专门微调的模型。我们发现，小型通用模型（如Qwen 2.5，其规模分别为0.5B、1.5B和3B）和其他基于变换器的架构（如DeBERTa v3）能够作为稳健的特征提取器，使得简单的分类器能够在不到100个高质量样本的情况下得到有效训练。重要的是，这些模型的中间变换器层在两类分类任务中通常比最终层表现更好。我们的结果显示，一个通用的大规模语言模型可以用于内容安全分类、检测提示注入以及同时生成输出标记。或者，这些相对较小的模型可以被修剪为最优中间层，并独占地作为稳健的特征提取器使用。由于我们的结果在不同的变换器架构中是一致的，我们推断稳健的特征提取是大多数，如果不是所有，语言模型的固有能力。 

---
# Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future 

**Title (ZH)**: 大语言模型增强的推荐系统：分类、趋势、应用与未来 

**Authors**: Qidong Liu, Xiangyu Zhao, Yuhao Wang, Yejing Wang, Zijian Zhang, Yuqi Sun, Xiang Li, Maolin Wang, Pengyue Jia, Chong Chen, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.13432)  

**Abstract**: Large Language Model (LLM) has transformative potential in various domains, including recommender systems (RS). There have been a handful of research that focuses on empowering the RS by LLM. However, previous efforts mainly focus on LLM as RS, which may face the challenge of intolerant inference costs by LLM. Recently, the integration of LLM into RS, known as LLM-Enhanced Recommender Systems (LLMERS), has garnered significant interest due to its potential to address latency and memory constraints in real-world applications. This paper presents a comprehensive survey of the latest research efforts aimed at leveraging LLM to enhance RS capabilities. We identify a critical shift in the field with the move towards incorporating LLM into the online system, notably by avoiding their use during inference. Our survey categorizes the existing LLMERS approaches into three primary types based on the component of the RS model being augmented: Knowledge Enhancement, Interaction Enhancement, and Model Enhancement. We provide an in-depth analysis of each category, discussing the methodologies, challenges, and contributions of recent studies. Furthermore, we highlight several promising research directions that could further advance the field of LLMERS. 

**Abstract (ZH)**: 大型语言模型（LLM）在各个领域都有着变革性的潜力，包括推荐系统（RS）。有关通过LLM增强RS的研究已有少量成果。然而，此前的努力主要集中在将LLM作为RS的一部分，这可能会面临LLM不可容忍的推理成本挑战。近年来，将LLM整合到RS中的做法，即LLM增强推荐系统（LLMERS），因其在实际应用中能够解决延迟和内存约束问题而引起了广泛关注。本文综述了最新的研究努力，旨在利用LLM增强RS的能力。我们识别出一个关键变化，即领域内的研究方向从将LLM引入线上系统转变为避免在推理过程中使用LLM。在此次综述中，我们根据增强RS模型组件的不同，将现有的LLMERS方法分为三大类：知识增强、交互增强和模型增强。我们对每类方法进行了深入分析，讨论了最近研究的方法、挑战和贡献。此外，我们还指出了几个有望进一步推动LLMERS领域的研究方向。 

---
# Safeguarding System Prompts for LLMs 

**Title (ZH)**: 为大型语言模型（LLM）保驾护航的提示系统 

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He  

**Link**: [PDF](https://arxiv.org/pdf/2412.13426)  

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we present PromptKeeper, a novel defense mechanism for system prompt privacy. By reliably detecting worst-case leakage and regenerating outputs without the system prompt when necessary, PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions. 

**Abstract (ZH)**: 大型语言模型（LLMs）在系统提示广泛应用的领域中发挥着越来越重要的作用，系统提示引导模型的输出，并往往包含业务逻辑和敏感信息，因此其保护显得至关重要。然而，敌对甚至常规用户的查询可能会利用LLM的漏洞暴露这些隐藏的提示。为解决这一问题，我们提出了一种名为PromptKeeper的新型防御机制，以确保针对敌对或常规查询下的提示提取攻击提供坚固且可靠的保护，同时也能够保持在良性用户交互过程中的对话能力和运行时效率。

### 完整翻译：

大型语言模型（LLMs）在系统提示应用中发挥着越来越重要的作用，而这些提示通常包含业务逻辑和敏感信息，因此其保护至关重要。然而，敌对甚至常规用户的查询却可以利用LLM的漏洞来暴露这些隐藏的提示。为解决这一问题，我们提出了PromptKeeper，一种新型的提示隐私保护机制。通过可靠检测最坏情况下的泄露并必要时重新生成输出，而无需使用系统提示，PromptKeeper确保在敌对或常规查询下都能提供坚固的提示提取攻击防护。同时，它在良性用户交互过程中也能保持对话能力和运行时效率。 

---
# Lightweight yet Fine-grained: A Graph Capsule Convolutional Network with Subspace Alignment for Shared-account Sequential Recommendation 

**Title (ZH)**: 轻量而细腻：一种基于子空间对齐的图胶囊卷积网络在共享账户序列推荐中的应用 

**Authors**: Jinyu Zhang, Zhongying Zhao, Chao Li, Yanwei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13408)  

**Abstract**: Shared-account Sequential Recommendation (SSR) aims to provide personalized recommendations for accounts shared by multiple users with varying sequential preferences. Previous studies on SSR struggle to capture the fine-grained associations between interactions and different latent users within the shared account's hybrid sequences. Moreover, most existing SSR methods (e.g., RNN-based or GCN-based methods) have quadratic computational complexities, hindering the deployment of SSRs on resource-constrained devices. To this end, we propose a Lightweight Graph Capsule Convolutional Network with subspace alignment for shared-account sequential recommendation, named LightGC$^2$N. Specifically, we devise a lightweight graph capsule convolutional network. It facilitates the fine-grained matching between interactions and latent users by attentively propagating messages on the capsule graphs. Besides, we present an efficient subspace alignment method. This method refines the sequence representations and then aligns them with the finely clustered preferences of latent users. The experimental results on four real-world datasets indicate that LightGC$^2$N outperforms nine state-of-the-art methods in accuracy and efficiency. 

**Abstract (ZH)**: 共享账户序列推荐（SSR）旨在为多个用户共享且具有不同序列偏好的账户提供个性化推荐。之前对SSR的研究难以捕捉共享账户混合序列中交互与不同潜在用户之间精细的关联。此外，大多数现有的SSR方法（例如基于RNN或GCN的方法）具有二次计算复杂度，这阻碍了在资源受限设备上部署SSR。为此，我们提出了一种具有子空间对齐的轻量级图capsule卷积网络，命名为LightGC$^2$N。具体而言，我们设计了一个轻量级图capsule卷积网络。通过在capsule图上注意性地传播消息，该网络促进了交互与潜在用户之间的精细匹配。此外，我们提出了一种高效子空间对齐方法。该方法优化了序列表示，并将其与潜在用户的精细聚类偏好对齐。在四个真实数据集上的实验结果表明，LightGC$^2$N在准确性和效率方面均优于九个当前最先进的方法。 

---
# What Human-Horse Interactions may Teach us About Effective Human-AI Interactions 

**Title (ZH)**: 人类与马匹互动可能为有效的人机交互提供哪些启示 

**Authors**: Mohammad Hossein Jarrahi, Stanley Ahalt  

**Link**: [PDF](https://arxiv.org/pdf/2412.13405)  

**Abstract**: This article explores human-horse interactions as a metaphor for understanding and designing effective human-AI partnerships. Drawing on the long history of human collaboration with horses, we propose that AI, like horses, should complement rather than replace human capabilities. We move beyond traditional benchmarks such as the Turing test, which emphasize AI's ability to mimic human intelligence, and instead advocate for a symbiotic relationship where distinct intelligences enhance each other. We analyze key elements of human-horse relationships: trust, communication, and mutual adaptability, to highlight essential principles for human-AI collaboration. Trust is critical in both partnerships, built through predictability and shared understanding, while communication and feedback loops foster mutual adaptability. We further discuss the importance of taming and habituation in shaping these interactions, likening it to how humans train AI to perform reliably and ethically in real-world settings. The article also addresses the asymmetry of responsibility, where humans ultimately bear the greater burden of oversight and ethical judgment. Finally, we emphasize that long-term commitment and continuous learning are vital in both human-horse and human-AI relationships, as ongoing interaction refines the partnership and increases mutual adaptability. By drawing on these insights from human-horse interactions, we offer a vision for designing AI systems that are trustworthy, adaptable, and capable of fostering symbiotic human-AI partnerships. 

**Abstract (ZH)**: 本文探讨了人马互动作为理解和设计有效的人工智能（AI）伙伴关系的隐喻。基于人类与马合作的悠久历史，我们提出，就像马匹一样，AI 应该是补充而非替代人类能力的角色。我们超越了传统的基准测试，如图灵测试，后者强调AI模仿人类智能的能力，而是倡导一种共生关系，在这种关系中，不同的智能相互增强。我们分析了人马关系中的关键要素：信任、沟通和相互适应，以突出人类与AI合作的基本原则。信任在两者的关系中至关重要，建立在可预测性和共享理解的基础上，而沟通和反馈回路则促进相互适应。我们进一步讨论了驯化和习惯化在塑造这些互动中的重要性，将其类比为人类如何训练AI在现实世界环境中可靠且道德地执行任务。本文还探讨了责任不对称性问题，即最终由人类承担更多的监督和伦理判断的责任。最后，我们强调，在人马关系和人机关系中，长期的承诺和持续学习是至关重要的，因为持续的互动可以精炼合作关系并增强相互适应能力。通过借鉴这些关于人马互动的见解，我们提出了设计值得信赖、灵活且能够培养共生人机关系的AI系统的愿景。 

---
# Distribution Shifts at Scale: Out-of-distribution Detection in Earth Observation 

**Title (ZH)**: 大规模分布变化：地球观测中的离分布检测 

**Authors**: Burak Ekim, Girmaw Abebe Tadesse, Caleb Robinson, Gilles Hacheme, Michael Schmitt, Rahul Dodhia, Juan M. Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2412.13394)  

**Abstract**: Training robust deep learning models is critical in Earth Observation, where globally deployed models often face distribution shifts that degrade performance, especially in low-data regions. Out-of-distribution (OOD) detection addresses this challenge by identifying inputs that differ from in-distribution (ID) data. However, existing methods either assume access to OOD data or compromise primary task performance, making them unsuitable for real-world deployment. We propose TARDIS, a post-hoc OOD detection method for scalable geospatial deployments. The core novelty lies in generating surrogate labels by integrating information from ID data and unknown distributions, enabling OOD detection at scale. Our method takes a pre-trained model, ID data, and WILD samples, disentangling the latter into surrogate ID and surrogate OOD labels based on internal activations, and fits a binary classifier as an OOD detector. We validate TARDIS on EuroSAT and xBD datasets, across 17 experimental setups covering covariate and semantic shifts, showing that it performs close to the theoretical upper bound in assigning surrogate ID and OOD samples in 13 cases. To demonstrate scalability, we deploy TARDIS on the Fields of the World dataset, offering actionable insights into pre-trained model behavior for large-scale deployments. The code is publicly available at this https URL. 

**Abstract (ZH)**: 在地球观测领域，训练稳健的深度学习模型至关重要，因为全球部署的模型经常会面临分布偏移（distribution shifts）问题，从而导致性能下降，特别是在数据量不足的区域。超分布（Out-of-distribution, OOD）检测通过识别与已知分布（in-distribution, ID）数据不同的输入来应对这一挑战。然而，现有的方法要么假设可以访问OOD数据，要么会牺牲主要任务的性能，使其不适合实际部署。我们提出TARDIS，这是一种适用于大规模地理空间部署的后处理OOD检测方法。其核心新颖之处在于通过整合ID数据和未知分布的信息来生成代理标签，从而实现大规模的OOD检测。我们的方法采用预先训练的模型、ID数据和WILD样本，根据内部激活将后者分解为代理ID标签和代理OOD标签，并拟合一个二元分类器作为OOD检测器。我们在EuroSAT和xBD数据集上对TARDIS进行了验证，涵盖了17种实验设置，包括协变量和语义偏移，结果显示在13种情况下，其在分配代理ID和OOD样本方面接近理论上限。为了展示其可扩展性，我们在Fields of the World数据集上部署了TARDIS，提供了有关大规模部署中预训练模型行为的可操作见解。代码已在以下网址公开发布：[这个 https URL](this https URL)。 

---
# MMHMR: Generative Masked Modeling for Hand Mesh Recovery 

**Title (ZH)**: MMHMR: 生成性掩码建模的手部网格恢复 

**Authors**: Muhammad Usama Saleem, Ekkasit Pinyoanuntapong, Mayur Jagdishbhai Patel, Hongfei Xue, Ahmed Helmy, Srijan Das, Pu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13393)  

**Abstract**: Reconstructing a 3D hand mesh from a single RGB image is challenging due to complex articulations, self-occlusions, and depth ambiguities. Traditional discriminative methods, which learn a deterministic mapping from a 2D image to a single 3D mesh, often struggle with the inherent ambiguities in 2D-to-3D mapping. To address this challenge, we propose MMHMR, a novel generative masked model for hand mesh recovery that synthesizes plausible 3D hand meshes by learning and sampling from the probabilistic distribution of the ambiguous 2D-to-3D mapping process. MMHMR consists of two key components: (1) a VQ-MANO, which encodes 3D hand articulations as discrete pose tokens in a latent space, and (2) a Context-Guided Masked Transformer that randomly masks out pose tokens and learns their joint distribution, conditioned on corrupted token sequences, image context, and 2D pose cues. This learned distribution facilitates confidence-guided sampling during inference, producing mesh reconstructions with low uncertainty and high precision. Extensive evaluations on benchmark and real-world datasets demonstrate that MMHMR achieves state-of-the-art accuracy, robustness, and realism in 3D hand mesh reconstruction. Project website: this https URL 

**Abstract (ZH)**: 从单张RGB图像重建三维手部网格是一个极具挑战性的任务，原因在于复杂的关节运动、自遮挡以及深度上的不确定性。传统的判别性方法往往学习从二维图像到单一三维网格的确定性映射，但在从二维到三维的映射过程中存在的固有不确定性上常常表现不佳。为应对这一挑战，我们提出了一种名为MMHMR的新颖生成式掩码模型，通过学习和从2D到3D映射过程中的概率分布中采样，该模型能够合成合理的三维手部网格。MMHMR包括两个关键组件：(1) VQ-MANO，它将三维手部关节运动编码为潜在空间中的离散姿态令牌；(2) 上下文引导的掩码变换器，该变换器随机遮掩姿态令牌，并在受损令牌序列、图像上下文和2D姿态线索的条件下学习其联合分布。这种学习到的分布有助于在推断过程中进行信心引导的采样，生成低不确定性且高精度的网格重构结果。在基准数据集和真实世界数据集上的广泛评估显示，MMHMR在三维手部网格重建方面达到了最先进的准确性、鲁棒性和真实性。项目网站: <请将具体的链接地址替换为实际的URL> 

---
# An Exploratory Study of ML Sketches and Visual Code Assistants 

**Title (ZH)**: 一项关于机器学习草图和可视化代码助手的探索性研究 

**Authors**: Luís F. Gomes, Vincent J. Hellendoorn, Jonathan Aldrich, Rui Abreu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13386)  

**Abstract**: This paper explores the integration of Visual Code Assistants in Integrated Development Environments (IDEs). In Software Engineering, whiteboard sketching is often the initial step before coding, serving as a crucial collaboration tool for developers. Previous studies have investigated patterns in SE sketches and how they are used in practice, yet methods for directly using these sketches for code generation remain limited. The emergence of visually-equipped large language models presents an opportunity to bridge this gap, which is the focus of our research. In this paper, we built a first prototype of a Visual Code Assistant to get user feedback regarding in-IDE sketch-to-code tools. We conduct an experiment with 19 data scientists, most of whom regularly sketch as part of their job. We investigate developers' mental models by analyzing patterns commonly observed in their sketches when developing an ML workflow. Analysis indicates that diagrams were the preferred organizational component (52.6%), often accompanied by lists (42.1%) and numbered points (36.8%). Our tool converts their sketches into a Python notebook by querying an LLM. We use an LLM-as-judge setup to score the quality of the generated code, finding that even brief sketching can effectively generate useful code outlines. We also find a positive correlation between sketch time and the quality of the generated code. We conclude the study by conducting extensive interviews to assess the tool's usefulness, explore potential use cases, and understand developers' needs. As noted by participants, promising applications for these assistants include education, prototyping, and collaborative settings. Our findings signal promise for the next generation of Code Assistants to integrate visual information, both to improve code generation and to better leverage developers' existing sketching practices. 

**Abstract (ZH)**: 本文探讨了在集成开发环境（IDEs）中集成视觉代码助手的可能性。在软件工程中，白板草图通常是编码前的初始步骤，作为开发人员之间的重要协作工具。先前的研究已经探讨了SE草图的模式及其在实践中的应用，但在如何直接利用这些草图进行代码生成方面的方法仍然有限。视觉增强的大规模语言模型的出现为解决这一问题提供了机会，这也是我们研究的重点。在本文中，我们构建了一个可视代码助手的原型，以收集用户对IDE内草图到代码工具的反馈。我们对19名数据科学家进行了实验，其中大多数人在工作中经常进行草图绘制。我们通过分析他们在开发机器学习工作流时经常出现的草图模式来研究开发人员的心理模型。分析表明，草图类图形组件是首选的组织工具（52.6%），常与列表（42.1%）和编号项（36.8%）结合使用。我们的工具通过查询大规模语言模型将他们的草图转换为Python笔记本。我们采用大规模语言模型作为评判工具来评估生成代码的质量，发现即使是简短的草图也能有效地生成有用的代码大纲。我们还发现，草图时间与生成代码的质量之间存在正相关关系。最后，我们通过广泛的访谈评估该工具的实用性，探讨其潜在应用场景，并了解开发人员的需求。正如参与者所指出的，这些助手在教育、原型制作和协作环境中具有显著的应用潜力。我们的研究结果表明，下一代代码助手有望整合视觉信息，以改善代码生成并更好地利用开发人员现有的草图绘制实践。 

---
# Voter Priming Campaigns: Strategies, Equilibria, and Algorithms 

**Title (ZH)**: 选民动员运动：策略、均衡与算法 

**Authors**: Jonathan Shaki, Yonatan Aumann, Sarit Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2412.13380)  

**Abstract**: Issue salience is a major determinant in voters' decisions. Candidates and political parties campaign to shift salience to their advantage - a process termed priming. We study the dynamics, strategies and equilibria of campaign spending for voter priming in multi-issue multi-party settings. We consider both parliamentary elections, where parties aim to maximize their share of votes, and various settings for presidential elections, where the winner takes all. For parliamentary elections, we show that pure equilibrium spending always exists and can be computed in time linear in the number of voters. For two parties and all settings, a spending equilibrium exists such that each party invests only in a single issue, and an equilibrium can be computed in time that is polynomial in the number of issues and linear in the number of voters. We also show that in most presidential settings no equilibrium exists. Additional properties of optimal campaign strategies are also studied. 

**Abstract (ZH)**: 选题显著性是选民决策中的主要决定因素。候选人和政治党派通过竞选活动试图将显著性转移到自身优势，这一过程称为“唤醒”。本文研究了在多议题多党派环境下，竞选支出对选民唤醒动态、策略和均衡的研究。我们既考虑议会选举，政治党派旨在最大化其得票率；也考虑总统选举的各种情境，其中胜者通吃。对于议会选举，我们证明纯均衡支出总是存在的，并且可以在时间复杂度为选民数量线性的情况下进行计算。对于两党制和所有选举类型，存在一种均衡支出策略，其中每一方仅在一个问题上投入资源，并且可以在议题数量多项式和选民数量线性的时间复杂度内进行计算。我们还证明，在大多数总统选举情境中，均衡不存在。此外，我们还研究了最优竞选策略的一些其他性质。 

---
# DateLogicQA: Benchmarking Temporal Biases in Large Language Models 

**Title (ZH)**: DateLogicQA：大型语言模型中的时间偏倚基准测试 

**Authors**: Gagan Bhatia, MingZe Tang, Cristina Mahanta, Madiha Kazi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13377)  

**Abstract**: This paper introduces DateLogicQA, a benchmark with 190 questions covering diverse date formats, temporal contexts, and reasoning types. We propose the Semantic Integrity Metric to assess tokenization quality and analyse two biases: Representation-Level Bias, affecting embeddings, and Logical-Level Bias, influencing reasoning outputs. Our findings provide a comprehensive evaluation of LLMs' capabilities and limitations in temporal reasoning, highlighting key challenges in handling temporal data accurately. The GitHub repository for our work is available at this https URL 

**Abstract (ZH)**: 本文介绍了DateLogicQA，这是一个包含190个问题的基准测试集，涵盖了多种日期格式、时间上下文和推理类型。我们提出了语义完整性指标来评估分词质量，并分析了两种偏差：表示层偏差，影响嵌入向量，和逻辑层偏差，影响推理输出。我们的研究结果全面评估了大语言模型在时间推理方面的能力和局限性，强调了准确处理时间数据的关键挑战。本工作的GitHub仓库地址为：[此链接] 

---
# Targeted View-Invariant Adversarial Perturbations for 3D Object Recognition 

**Title (ZH)**: 面向视图不变的对抗扰动攻击在三维物体识别中的应用 

**Authors**: Christian Green, Mehmet Ergezer, Abdurrahman Zeybey  

**Link**: [PDF](https://arxiv.org/pdf/2412.13376)  

**Abstract**: Adversarial attacks pose significant challenges in 3D object recognition, especially in scenarios involving multi-view analysis where objects can be observed from varying angles. This paper introduces View-Invariant Adversarial Perturbations (VIAP), a novel method for crafting robust adversarial examples that remain effective across multiple viewpoints. Unlike traditional methods, VIAP enables targeted attacks capable of manipulating recognition systems to classify objects as specific, pre-determined labels, all while using a single universal perturbation. Leveraging a dataset of 1,210 images across 121 diverse rendered 3D objects, we demonstrate the effectiveness of VIAP in both targeted and untargeted settings. Our untargeted perturbations successfully generate a singular adversarial noise robust to 3D transformations, while targeted attacks achieve exceptional results, with top-1 accuracies exceeding 95% across various epsilon values. These findings highlight VIAPs potential for real-world applications, such as testing the robustness of 3D recognition systems. The proposed method sets a new benchmark for view-invariant adversarial robustness, advancing the field of adversarial machine learning for 3D object recognition. 

**Abstract (ZH)**: adversarial攻击在三维物体识别领域提出了重大挑战，尤其是在多视角分析场景中更为明显，物体可以从多个角度被观察。本文介绍了一种新颖的方法——视图不变对抗扰动（View-Invariant Adversarial Perturbations, VIAP），该方法用于构造在多个视角下仍然有效的鲁棒对抗样本。与传统的对抗方法不同，VIAP 允许执行针对攻击，能够操纵识别系统将物体分类为特定的预定义标签，同时仅使用单一的通用扰动。我们利用涵盖121种不同渲染三维物体的1,210张图像数据集，展示了VIAP在目标攻击和无目标攻击场景中的有效性。我们的无目标扰动成功生成了对3D变换鲁棒的单一对抗噪声，而针对攻击则取得了优异的结果，多种ε值下的top-1准确率均超过95%。这些发现凸显了VIAP在实际应用中的潜力，如测试3D识别系统的鲁棒性。提出的该方法为视图不变对抗鲁棒性设定了新的基准，推动了三维物体识别领域对抗机器学习的发展。 

---
# Multi-Agent Motion Planning For Differential Drive Robots Through Stationary State Search 

**Title (ZH)**: 通过静态状态搜索进行差速驱动机器人多代理运动规划 

**Authors**: Jingtian Yan, Jiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.13359)  

**Abstract**: Multi-Agent Motion Planning (MAMP) finds various applications in fields such as traffic management, airport operations, and warehouse automation. In many of these environments, differential drive robots are commonly used. These robots have a kinodynamic model that allows only in-place rotation and movement along their current orientation, subject to speed and acceleration limits. However, existing Multi-Agent Path Finding (MAPF)-based methods often use simplified models for robot kinodynamics, which limits their practicality and realism. In this paper, we introduce a three-level framework called MASS to address these challenges. MASS combines MAPF-based methods with our proposed stationary state search planner to generate high-quality kinodynamically-feasible plans. We further extend MASS using an adaptive window mechanism to address the lifelong MAMP problem. Empirically, we tested our methods on the single-shot grid map domain and the lifelong warehouse domain. Our method shows up to 400% improvements in terms of throughput compared to existing methods. 

**Abstract (ZH)**: 多智能体运动规划（Multi-Agent Motion Planning, MAMP）在交通管理、机场运营和仓库自动化等领域有着广泛的应用。在这些环境中，通常使用差速驱动机器人。这些机器人具有只能原地旋转和沿当前方向移动的运动学动力学模型，并受到速度和加速度限制。然而，现有的基于多智能体路径规划（Multi-Agent Path Finding, MAPF）的方法常使用简化的机器人运动学动力学模型，这限制了它们的实用性和真实性。本文中，我们提出了一种名为MASS的三层框架来解决这些问题。MASS将基于MAPF的方法与我们提出的静态状态搜索规划器相结合，以生成高质量的动力学可行方案。我们进一步通过自适应窗口机制扩展MASS，以解决终身多智能体运动规划（lifelong MAMP）问题。通过实验，我们在单次操作网格地图领域和终身仓库领域测试了我们的方法。与现有方法相比，我们的方法在吞吐量方面展现了高达400%的改善。 

---
# A Novel Machine Learning Classifier Based on Genetic Algorithms and Data Importance Reformatting 

**Title (ZH)**: 基于遗传算法和数据重要性重组的新颖机器学习分类器 

**Authors**: A. K. Alkhayyata, N. M. Hewahi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13350)  

**Abstract**: In this paper, a novel classification algorithm that is based on Data Importance (DI) reformatting and Genetic Algorithms (GA) named GADIC is proposed to overcome the issues related to the nature of data which may hinder the performance of the Machine Learning (ML) classifiers. GADIC comprises three phases which are data reformatting phase which depends on DI concept, training phase where GA is applied on the reformatted training dataset, and testing phase where the instances of the reformatted testing dataset are being averaged based on similar instances in the training dataset. GADIC is an approach that utilizes the exiting ML classifiers with involvement of data reformatting, using GA to tune the inputs, and averaging the similar instances to the unknown instance. The averaging of the instances becomes the unknown instance to be classified in the stage of testing. GADIC has been tested on five existing ML classifiers which are Support Vector Machine (SVM), K-Nearest Neighbour (KNN), Logistic Regression (LR), Decision Tree (DT), and Naïve Bayes (NB). All were evaluated using seven open-source UCI ML repository and Kaggle datasets which are Cleveland heart disease, Indian liver patient, Pima Indian diabetes, employee future prediction, telecom churn prediction, bank customer churn, and tech students. In terms of accuracy, the results showed that, with the exception of approximately 1% decrease in the accuracy of NB classifier in Cleveland heart disease dataset, GADIC significantly enhanced the performance of most ML classifiers using various datasets. In addition, KNN with GADIC showed the greatest performance gain when compared with other ML classifiers with GADIC followed by SVM while LR had the lowest improvement. The lowest average improvement that GADIC could achieve is 5.96%, whereas the maximum average improvement reached 16.79%. 

**Abstract (ZH)**: 在本文中，提出了一种基于数据重要性（Data Importance, DI）重塑和遗传算法（Genetic Algorithms, GA）的新分类算法——GADIC，以克服与数据性质相关的可能妨碍机器学习（Machine Learning, ML）分类器性能的问题。GADIC 包含三个阶段：数据重塑阶段，该阶段基于 DI 概念；训练阶段，在此阶段应用 GA 对重塑后的训练数据集进行训练；测试阶段，该阶段基于训练数据集中相似的实例对重塑后的测试数据集中的实例进行平均。GADIC 是一种利用现有 ML 分类器的方法，涉及数据重塑，使用 GA 调整输入，并通过平均相似实例来分类未知实例。在整个测试阶段，原来的实例被平均成未知实例进行分类。GADIC 已在五个现有的 ML 分类器上进行测试，分别是支持向量机（SVM）、K-最近邻（KNN）、逻辑回归（LR）、决策树（DT）和朴素贝叶斯（NB）。所有分类器均使用来自公开的 UCI ML 存储库和Kaggle 数据集进行了评估，其中包括克利夫兰心脏病、印度肝病患者、皮姆印第安人糖尿病、员工未来预测、电信客户流失预测、银行客户流失和科技学生等数据集。在准确率方面，结果显示，除朴素贝叶斯（NB）分类器在克利夫兰心脏病数据集上的准确率降低了约 1% 之外，GADIC 显著提高了大多数 ML 分类器在多种数据集上的性能。此外，与其它 ML 分类器相比，使用 GADIC 的 KNN 表现最佳，其次是 SVM，而 LR 的改进最小。GADIC 能实现的最低平均改进为 5.96%，而最高的平均改进则达到 16.79%。 

---
# Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs 

**Title (ZH)**: 揭秘秘方：监督微调小型语言模型指南 

**Authors**: Aldo Pareja, Nikhil Shivakumar Nayak, Hao Wang, Krishnateja Killamsetty, Shivchander Sudalairaj, Wenlong Zhao, Seungwook Han, Abhishek Bhandwaldar, Guangxuan Xu, Kai Xu, Ligong Han, Luke Inglis, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2412.13337)  

**Abstract**: The rise of large language models (LLMs) has created a significant disparity: industrial research labs with their computational resources, expert teams, and advanced infrastructures, can effectively fine-tune LLMs, while individual developers and small organizations face barriers due to limited resources. In this paper, we aim to bridge this gap by presenting a comprehensive study on supervised fine-tuning of LLMs using instruction-tuning datasets spanning diverse knowledge domains and skills. We focus on small-sized LLMs (3B to 7B parameters) for their cost-efficiency and accessibility. We explore various training configurations and strategies across four open-source pre-trained models. We provide detailed documentation of these configurations, revealing findings that challenge several common training practices, including hyperparameter recommendations from TULU and phased training recommended by Orca. Key insights from our work include: (i) larger batch sizes paired with lower learning rates lead to improved model performance on benchmarks such as MMLU, MTBench, and Open LLM Leaderboard; (ii) early-stage training dynamics, such as lower gradient norms and higher loss values, are strong indicators of better final model performance, enabling early termination of sub-optimal runs and significant computational savings; (iii) through a thorough exploration of hyperparameters like warmup steps and learning rate schedules, we provide guidance for practitioners and find that certain simplifications do not compromise performance; and (iv) we observed no significant difference in performance between phased and stacked training strategies, but stacked training is simpler and more sample efficient. With these findings holding robustly across datasets and models, we hope this study serves as a guide for practitioners fine-tuning small LLMs and promotes a more inclusive environment for LLM research. 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起导致了一种显著的差距：拥有计算资源、专家团队和高级基础设施的工业研究实验室能够有效微调LLMs，而个人开发者和小型组织则因资源有限而面临障碍。本文旨在弥合这一差距，通过介绍涵盖不同知识领域和技能的指令调优数据集上的监督微调全面研究来实现这一目的。我们重点研究小型化的LLMs（3B至7B参数），因为它们的成本效益和可访问性较高。我们探讨了四种开源预训练模型下的各种训练配置和策略。我们提供了这些配置的详细文档，揭示了挑战一些常见训练实践的研究结果，包括TULU的超参数推荐和Orca建议的分阶段训练。本文的关键见解包括：(i) 较大的批量大小与较低的学习率配对，在诸如MMLU、MTBench和Open LLM Leaderboard等基准测试中可提高模型性能；(ii) 早期训练动态，如较低的梯度范数和较高的损失值，是最终模型性能较好的强烈指示器，这使得可以提前终止次优运行并实现显著的计算节省；(iii) 通过对诸如预热步骤和学习率调度等超参数的深入探索，我们为实践者提供了指导，并发现某些简化不会影响性能；(iv) 我们观察到分阶段训练和堆叠训练策略在性能上没有显著差异，但堆叠训练更为简单且样本效率更高。这些发现跨越不同数据集和模型保持稳健，我们希望这项研究能为微调小型LLMs的实践者提供指导，并促进LLM研究更具包容性。 

---
# Experience of Training a 1.7B-Parameter LLaMa Model From Scratch 

**Title (ZH)**: 从零训练一个1.7亿参数的LLaMa模型的体验 

**Authors**: Miles Q. Li, Benjamin C. M. Fung, Shih-Chia Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13335)  

**Abstract**: Pretraining large language models is a complex endeavor influenced by multiple factors, including model architecture, data quality, training continuity, and hardware constraints. In this paper, we share insights gained from the experience of training DMaS-LLaMa-Lite, a fully open source, 1.7-billion-parameter, LLaMa-based model, on approximately 20 billion tokens of carefully curated data. We chronicle the full training trajectory, documenting how evolving validation loss levels and downstream benchmarks reflect transitions from incoherent text to fluent, contextually grounded output. Beyond standard quantitative metrics, we highlight practical considerations such as the importance of restoring optimizer states when resuming from checkpoints, and the impact of hardware changes on training stability and throughput. While qualitative evaluation provides an intuitive understanding of model improvements, our analysis extends to various performance benchmarks, demonstrating how high-quality data and thoughtful scaling enable competitive results with significantly fewer training tokens. By detailing these experiences and offering training logs, checkpoints, and sample outputs, we aim to guide future researchers and practitioners in refining their pretraining strategies. The training script is available on Github at this https URL. The model checkpoints are available on Huggingface at this https URL. 

**Abstract (ZH)**: 预训练大型语言模型是一项复杂的工作，受到多种因素的影响，包括模型架构、数据质量、训练连续性和硬件限制。在本文中，我们分享了训练 DMaS-LLaMa-Lite 这一完全开源、含17亿参数、基于LLaMa的模型的经验心得。该模型在大约200亿个经过精心筛选的数据标记中进行了训练。我们详细记录了整个训练过程，说明了从不连贯文本到流畅、上下文相关的输出所经历的过渡过程，通过逐渐变化的验证损失水平和下游基准指标的变化反映出来。除了标准的量化评估指标外，我们还强调了诸如在从检查点恢复训练时恢复优化器状态的重要性，以及硬件变化对训练稳定性和吞吐量的影响等实际问题。虽然定性的评估提供了一种直观的理解模型改进的方法，但我们的分析还涵盖了各种性能基准，展示了高质量数据和精心设计的扩展策略如何在显著减少训练标记数量的情况下取得具有竞争力的结果。通过详细记录这些经验，并提供训练日志、检查点和样本输出，我们旨在为未来的研究者和从业者提供指导，以改进他们的预训练策略。训练脚本可在 Github 上获取，网址为 [此链接]。模型检查点可在 Huggingface 上获取，网址为 [此链接]。 

---
# BadSAD: Clean-Label Backdoor Attacks against Deep Semi-Supervised Anomaly Detection 

**Title (ZH)**: 坏SAD：针对深度半监督异常检测的清洁标签后门攻击

解释：
- "BadSAD" 被翻译为 "坏SAD"，保留了原名的缩写形式。
- "Clean-Label Backdoor Attacks" 翻译为 "清洁标签后门攻击"，符合安全和隐私领域的术语规范。
- "Deep Semi-Supervised Anomaly Detection" 翻译为 "深度半监督异常检测"，保持了术语的专业性和准确性。 

**Authors**: He Cheng, Depeng Xu, Shuhan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2412.13324)  

**Abstract**: Image anomaly detection (IAD) is essential in applications such as industrial inspection, medical imaging, and security. Despite the progress achieved with deep learning models like Deep Semi-Supervised Anomaly Detection (DeepSAD), these models remain susceptible to backdoor attacks, presenting significant security challenges. In this paper, we introduce BadSAD, a novel backdoor attack framework specifically designed to target DeepSAD models. Our approach involves two key phases: trigger injection, where subtle triggers are embedded into normal images, and latent space manipulation, which positions and clusters the poisoned images near normal images to make the triggers appear benign. Extensive experiments on benchmark datasets validate the effectiveness of our attack strategy, highlighting the severe risks that backdoor attacks pose to deep learning-based anomaly detection systems. 

**Abstract (ZH)**: 图像异常检测（IAD）在工业检验、医疗成像和安全等领域具有重要应用价值。尽管像深度半监督异常检测（DeepSAD）这类深度学习模型取得了进展，但这些模型仍然容易受到后门攻击的影响，从而引发重大安全挑战。在本文中，我们提出了一种名为BadSAD的新颖后门攻击框架，专门针对DeepSAD模型。我们的方法包含两个关键阶段：触发注入，即在正常图像中嵌入微妙的触发器；潜在空间操纵，即将中毒图像重新定位并聚类到正常图像附近，使触发器看起来无害。在基准数据集上的广泛实验验证了我们攻击策略的有效性，突显了后门攻击对基于深度学习的异常检测系统的严重风险。 

---
# FastVLM: Efficient Vision Encoding for Vision Language Models 

**Title (ZH)**: FastVLM：高效的视觉编码方法用于视觉语言模型 

**Authors**: Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari  

**Link**: [PDF](https://arxiv.org/pdf/2412.13303)  

**Abstract**: Scaling the input image resolution is essential for enhancing the performance of Vision Language Models (VLMs), particularly in text-rich image understanding tasks. However, popular visual encoders such as ViTs become inefficient at high resolutions due to the large number of tokens and high encoding latency caused by stacked self-attention layers. At different operational resolutions, the vision encoder of a VLM can be optimized along two axes: reducing encoding latency and minimizing the number of visual tokens passed to the LLM, thereby lowering overall latency. Based on a comprehensive efficiency analysis of the interplay between image resolution, vision latency, token count, and LLM size, we introduce FastVLM, a model that achieves an optimized trade-off between latency, model size and accuracy. FastVLM incorporates FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images. Unlike previous methods, FastVLM achieves the optimal balance between visual token count and image resolution solely by scaling the input image, eliminating the need for additional token pruning and simplifying the model design. In the LLaVA-1.5 setup, FastVLM achieves 3.2$\times$ improvement in time-to-first-token (TTFT) while maintaining similar performance on VLM benchmarks compared to prior works. Compared to LLaVa-OneVision at the highest resolution (1152$\times$1152), FastVLM achieves comparable performance on key benchmarks like SeedBench and MMMU, using the same 0.5B LLM, but with 85$\times$ faster TTFT and a vision encoder that is 3.4$\times$ smaller. 

**Abstract (ZH)**: 提高输入图像分辨率对于提升视觉语言模型（VLMs）的性能至关重要，尤其是在富含文本的图像理解任务中。然而，流行的视觉编码器如ViTs在高分辨率下变得效率低下，这主要是由于堆叠的自注意力层导致的大量令牌和高编码延迟。在不同的操作分辨率下，VLM的视觉编码器可以通过两个轴进行优化：减少编码延迟并最小化传递给语言模型的语言模型（LLM）的视觉令牌数量，从而降低整体延迟。基于对图像分辨率、视觉延迟、令牌数量和LLM大小之间相互作用的全面效率分析，我们引入了FastVLM模型，该模型在延迟、模型大小和准确性之间实现了优化权衡。FastVLM集成了FastViTHD，这是一种新颖的混合视觉编码器，设计用于输出更少的令牌并显著减少高分辨率图像的编码时间。与之前的方法不同，FastVLM仅通过缩放输入图像即可实现视觉令牌数量和图像分辨率之间的最佳平衡，避免了额外令牌裁剪的需要，并简化了模型设计。在LLaVA-1.5设置中，FastVLM在首个令牌时间（TTFT）上实现了3.2倍的提升，同时在VLM基准上的性能与先前工作相当。与最高分辨率（1152×1152）的LLaVa-OneVision相比，FastVLM在类似于SeedBench和MMMU的关键基准上实现了相同0.5B LLM的可比较性能，但TTFT快了85倍，视觉编码器也小了3.4倍。 

---
# In-context learning for medical image segmentation 

**Title (ZH)**: 基于上下文的学习在医学图像分割中的应用 

**Authors**: Eichi Takaya, Shinnosuke Yamamoto  

**Link**: [PDF](https://arxiv.org/pdf/2412.13299)  

**Abstract**: Annotation of medical images, such as MRI and CT scans, is crucial for evaluating treatment efficacy and planning radiotherapy. However, the extensive workload of medical professionals limits their ability to annotate large image datasets, posing a bottleneck for AI applications in medical imaging. To address this, we propose In-context Cascade Segmentation (ICS), a novel method that minimizes annotation requirements while achieving high segmentation accuracy for sequential medical images. ICS builds on the UniverSeg framework, which performs few-shot segmentation using support images without additional training. By iteratively adding the inference results of each slice to the support set, ICS propagates information forward and backward through the sequence, ensuring inter-slice consistency. We evaluate the proposed method on the HVSMR dataset, which includes segmentation tasks for eight cardiac regions. Experimental results demonstrate that ICS significantly improves segmentation performance in complex anatomical regions, particularly in maintaining boundary consistency across slices, compared to baseline methods. The study also highlights the impact of the number and position of initial support slices on segmentation accuracy. ICS offers a promising solution for reducing annotation burdens while delivering robust segmentation results, paving the way for its broader adoption in clinical and research applications. 

**Abstract (ZH)**: 医学图像标注，如MRI和CT扫描，对于评估治疗效果和制定放疗计划至关重要。然而，医学专业人士在标注大量图像数据集时需要承担巨大的工作量，限制了人工智能在医学影像领域中的应用。为解决这一问题，我们提出了一种新的方法——上下文递进分割（In-context Cascade Segmentation, ICS），该方法在降低标注需求的同时，能够实现序列医学图像的高准确度分割。ICS 基于 UniverSeg 框架，该框架通过使用支持图像实现了少样本分割，无需额外训练。通过迭代将每个切片的推理结果添加到支持集中，ICS 在序列中向前和向后传播信息，确保切片间的一致性。我们在包含八个心脏区域分割任务的 HVSMR 数据集上对所提出的方法进行了评估。实验结果表明，ICS 在复杂解剖区域显著提高了分割性能，特别是在保持切片间边界一致性方面，与基线方法相比表现更佳。研究还强调了初始支持切片的数量和位置对分割精度的影响。ICS 提供了一种减轻标注负担同时取得稳健分割结果的有前景的解决方案，为其实临床和研究应用的广泛使用奠定了基础。 

---
# Posterior Mean Matching: Generative Modeling through Online Bayesian Inference 

**Title (ZH)**: 后验均值匹配：通过在线贝叶斯推断进行生成建模 

**Authors**: Sebastian Salazar, Michal Kucer, Yixin Wang, Emily Casleton, David Blei  

**Link**: [PDF](https://arxiv.org/pdf/2412.13286)  

**Abstract**: This paper introduces posterior mean matching (PMM), a new method for generative modeling that is grounded in Bayesian inference. PMM uses conjugate pairs of distributions to model complex data of various modalities like images and text, offering a flexible alternative to existing methods like diffusion models. PMM models iteratively refine noisy approximations of the target distribution using updates from online Bayesian inference. PMM is flexible because its mechanics are based on general Bayesian models. We demonstrate this flexibility by developing specialized examples: a generative PMM model of real-valued data using the Normal-Normal model, a generative PMM model of count data using a Gamma-Poisson model, and a generative PMM model of discrete data using a Dirichlet-Categorical model. For the Normal-Normal PMM model, we establish a direct connection to diffusion models by showing that its continuous-time formulation converges to a stochastic differential equation (SDE). Additionally, for the Gamma-Poisson PMM, we derive a novel SDE driven by a Cox process, which is a significant departure from traditional Brownian motion-based generative models. PMMs achieve performance that is competitive with generative models for language modeling and image generation. 

**Abstract (ZH)**: 本文介绍了后验均值匹配（PMM），这是一种基于贝叶斯推断的生成建模新方法。PMM 使用共轭分布对来建模多种模态的数据，如图像和文本，为现有的如扩散模型等方法提供了灵活的替代方案。PMM 模型通过在线贝叶斯推断的更新迭代地细化目标分布的嘈杂近似。由于其机制基于通用的贝叶斯模型，PMM 具有很大的灵活性。我们通过开发专门的例子来展示这种灵活性：使用正态-正态模型的实值数据生成 PMM 模型、使用伽玛-泊松模型的计数值数据生成 PMM 模型，以及使用狄利克雷-分类模型的离散数据生成 PMM 模型。对于正态-正态 PMM 模型，我们通过证明其连续时间形式收敛到随机微分方程（SDE）建立了与扩散模型的直接联系。此外，对于伽玛-泊松 PMM，我们推导出一种由 Cox 过程驱动的新 SDE，这在传统的基于布朗运动的生成模型中是一个显著的不同。PMMs 在语言建模和图像生成方面的性能与现有的生成模型具有竞争力。 

---
# Enhancing Internet of Things Security throughSelf-Supervised Graph Neural Networks 

**Title (ZH)**: 通过自我监督图形神经网络增强物联网安全 

**Authors**: Safa Ben Atitallah, Maha Driss, Wadii Boulila, Anis Koubaa  

**Link**: [PDF](https://arxiv.org/pdf/2412.13240)  

**Abstract**: With the rapid rise of the Internet of Things (IoT), ensuring the security of IoT devices has become essential. One of the primary challenges in this field is that new types of attacks often have significantly fewer samples than more common attacks, leading to unbalanced datasets. Existing research on detecting intrusions in these unbalanced labeled datasets primarily employs Convolutional Neural Networks (CNNs) or conventional Machine Learning (ML) models, which result in incomplete detection, especially for new attacks. To handle these challenges, we suggest a new approach to IoT intrusion detection using Self-Supervised Learning (SSL) with a Markov Graph Convolutional Network (MarkovGCN). Graph learning excels at modeling complex relationships within data, while SSL mitigates the issue of limited labeled data for emerging attacks. Our approach leverages the inherent structure of IoT networks to pre-train a GCN, which is then fine-tuned for the intrusion detection task. The integration of Markov chains in GCN uncovers network structures and enriches node and edge features with contextual information. Experimental results demonstrate that our approach significantly improves detection accuracy and robustness compared to conventional supervised learning methods. Using the EdgeIIoT-set dataset, we attained an accuracy of 98.68\%, a precision of 98.18%, a recall of 98.35%, and an F1-Score of 98.40%. 

**Abstract (ZH)**: 随着物联网（IoT）的迅速发展，确保物联网设备的安全已成为必要。这一领域的主要挑战之一是针对新类型的攻击的数据样本通常远少于常见攻击，导致数据集不平衡。现有研究主要使用卷积神经网络（CNN）或传统的机器学习（ML）模型来检测这些不平衡标签的数据集中的入侵行为，这往往会使得检测不完整，特别是在新攻击的情境下。为应对这些挑战，我们提出了一种使用自监督学习（SSL）和马尔可夫图卷积网络（MarkovGCN）的新方法，以实现物联网入侵检测。图学习擅长建模数据中的复杂关系，而自监督学习则能够缓解新兴攻击数据标注量有限的问题。我们的方法利用物联网网络的固有结构进行预训练一个图卷积网络（GCN），然后针对入侵检测任务进行微调。图卷积网络中的马尔可夫链集成使我们能够揭示网络结构并利用上下文信息丰富节点和边的特征。实验结果表明，与传统的监督学习方法相比，我们提出的方法显著提高了检测准确度和鲁棒性。使用EdgeIIoT-set数据集，我们达到了98.68%的准确率、98.18%的精确率、98.35%的召回率和98.40%的F1分数。 

---
# COSEE: Consistency-Oriented Signal-Based Early Exiting via Calibrated Sample Weighting Mechanism 

**Title (ZH)**: COSEE：一致性导向的信号基于早期退出机制通过校准样本权重机制 

**Authors**: Jianing He, Qi Zhang, Hongyun Zhang, Xuanjing Huang, Usman Naseem, Duoqian Miao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13236)  

**Abstract**: Early exiting is an effective paradigm for improving the inference efficiency of pre-trained language models (PLMs) by dynamically adjusting the number of executed layers for each sample. However, in most existing works, easy and hard samples are treated equally by each classifier during training, which neglects the test-time early exiting behavior, leading to inconsistency between training and testing. Although some methods have tackled this issue under a fixed speed-up ratio, the challenge of flexibly adjusting the speed-up ratio while maintaining consistency between training and testing is still under-explored. To bridge the gap, we propose a novel Consistency-Oriented Signal-based Early Exiting (COSEE) framework, which leverages a calibrated sample weighting mechanism to enable each classifier to emphasize the samples that are more likely to exit at that classifier under various acceleration scenarios. Extensive experiments on the GLUE benchmark demonstrate the effectiveness of our COSEE across multiple exiting signals and backbones, yielding a better trade-off between performance and efficiency. 

**Abstract (ZH)**: 早退出是一种通过动态调整每份样本执行的层数来提高预训练语言模型（PLMs）推理效率的有效范式。然而，在现有大多数研究中，每一分类器在训练过程中对待容易和困难样本的方式是相同的，这忽略了测试时的早退出行为，导致训练和测试之间存在不一致。尽管有些方法在固定加速比的情况下解决了这一问题，但在不同加速场景下灵活调整加速比以保持训练和测试一致性的问题仍然未充分探索。为了解决这一差距，我们提出了一种面向一致性的基于信号的早退出（COSEE）框架，该框架利用了一个校准的样本加权机制，使每一分类器能够在各种加速场景下更强调那些在该分类器处更有可能退出的样本。在GLUE基准上的广泛实验表明，我们的COSEE能够在多种早退出信号和骨干网络下有效，从而在性能和效率之间实现更好的权衡。 

---
# C2F-TP: A Coarse-to-Fine Denoising Framework for Uncertainty-Aware Trajectory Prediction 

**Title (ZH)**: C2F-TP: 一种考虑不确定性的时间序列预测精细到粗糙降噪框架 

**Authors**: Zichen Wang, Hao Miao, Senzhang Wang, Renzhi Wang, Jianxin Wang, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13231)  

**Abstract**: Accurately predicting the trajectory of vehicles is critically important for ensuring safety and reliability in autonomous driving. Although considerable research efforts have been made recently, the inherent trajectory uncertainty caused by various factors including the dynamic driving intends and the diverse driving scenarios still poses significant challenges to accurate trajectory prediction. To address this issue, we propose C2F-TP, a coarse-to-fine denoising framework for uncertainty-aware vehicle trajectory prediction. C2F-TP features an innovative two-stage coarse-to-fine prediction process. Specifically, in the spatial-temporal interaction stage, we propose a spatial-temporal interaction module to capture the inter-vehicle interactions and learn a multimodal trajectory distribution, from which a certain number of noisy trajectories are sampled. Next, in the trajectory refinement stage, we design a conditional denoising model to reduce the uncertainty of the sampled trajectories through a step-wise denoising operation. Extensive experiments are conducted on two real datasets NGSIM and highD that are widely adopted in trajectory prediction. The result demonstrates the effectiveness of our proposal. 

**Abstract (ZH)**: 精准预测车辆轨迹对于自主驾驶的安全性和可靠性至关重要。尽管近期已做出大量研究努力，但由各种因素（包括动态驾驶意图和多样化的驾驶场景）引起的内在轨迹不确定性依然对准确的轨迹预测提出了重大挑战。为应对这一问题，我们提出了C2F-TP，一种细粒度降噪框架，用于具有不确定性的车辆轨迹预测。C2F-TP具有创新的两阶段自上而下的预测过程。具体而言，在空间-时间交互阶段，我们提出了一种空间-时间交互模块，用于捕捉车辆之间的交互，并学习多模态轨迹分布，从中采样出一定数量的噪声轨迹。接着，在轨迹细化阶段，我们设计了一种条件降噪模型，通过逐步降噪操作减少采样轨迹的不确定性。我们在广泛应用于轨迹预测的两个实数据集NGSIM和highD上进行了广泛的实验证明了我们方法的有效性。 

---
# Training Verification-Friendly Neural Networks via Neuron Behavior Consistency 

**Title (ZH)**: 通过神经元行为一致性训练可验证的神经网络 

**Authors**: Zongxin Liu, Zhe Zhao, Fu Song, Jun Sun, Pengfei Yang, Xiaowei Huang, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13229)  

**Abstract**: Formal verification provides critical security assurances for neural networks, yet its practical application suffers from the long verification time. This work introduces a novel method for training verification-friendly neural networks, which are robust, easy to verify, and relatively accurate. Our method integrates neuron behavior consistency into the training process, making neuron activation states consistent across different inputs in a local neighborhood, reducing the number of unstable neurons and tightening the bounds of neurons thereby enhancing neural network verifiability. We evaluated our method using the MNIST, Fashion-MNIST, and CIFAR-10 datasets across various network architectures. The results of the experiment demonstrate that networks trained using our method are verification-friendly across different radii and different model architectures, whereas other tools fail to maintain verifiability as the radius increases. We also show that our method can be combined with existing methods to further improve the verifiability of networks. 

**Abstract (ZH)**: 形式化验证为神经网络提供了关键的安全保障，然而其实际应用受到长时间验证的困扰。本文提出了一种新的方法，用于训练易于验证的神经网络，这些网络在保持鲁棒性和相对准确性的同时，也易于验证。我们的方法将神经元行为一致性集成到训练过程中，在局部邻域内使不同输入下的神经元激活状态保持一致，减少了不稳定的神经元数量，并收紧了神经元的边界，从而增强了神经网络的可验证性。我们使用MNIST、Fashion-MNIST和CIFAR-10数据集，在多种网络架构下评估了该方法。实验结果表明，使用该方法训练的网络在不同的半径和不同模型架构下都具有易于验证的特性，而其他工具在半径增大时无法保持这种可验证性。此外，我们还展示了该方法可以与其他方法结合使用，以进一步提高网络的可验证性。 

---
# TSEML: A task-specific embedding-based method for few-shot classification of cancer molecular subtypes 

**Title (ZH)**: TSEML：一种基于任务特定嵌入的方法，用于癌症分子亚型的少样本分类 

**Authors**: Ran Sua, Rui Shi, Hui Cui, Ping Xuan, Chengyan Fang, Xikang Feng, Qiangguo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13228)  

**Abstract**: Molecular subtyping of cancer is recognized as a critical and challenging upstream task for personalized therapy. Existing deep learning methods have achieved significant performance in this domain when abundant data samples are available. However, the acquisition of densely labeled samples for cancer molecular subtypes remains a significant challenge for conventional data-intensive deep learning approaches. In this work, we focus on the few-shot molecular subtype prediction problem in heterogeneous and small cancer datasets, aiming to enhance precise diagnosis and personalized treatment. We first construct a new few-shot dataset for cancer molecular subtype classification and auxiliary cancer classification, named TCGA Few-Shot, from existing publicly available datasets. To effectively leverage the relevant knowledge from both tasks, we introduce a task-specific embedding-based meta-learning framework (TSEML). TSEML leverages the synergistic strengths of a model-agnostic meta-learning (MAML) approach and a prototypical network (ProtoNet) to capture diverse and fine-grained features. Comparative experiments conducted on the TCGA Few-Shot dataset demonstrate that our TSEML framework achieves superior performance in addressing the problem of few-shot molecular subtype classification. 

**Abstract (ZH)**: 癌症的分子亚型分类被认定为个性化治疗中至关重要且具有挑战性的上游任务。在数据样本丰富的情况下，现有的深度学习方法已经在这一领域取得了显著的性能。然而，获取高密度标注的癌症分子亚型样本仍然是传统数据密集型深度学习方法面临的一个重大挑战。在本文中，我们聚焦于异质且样本量小的癌症数据集中的少样本分子亚型预测问题，旨在提高精准诊断和个性化治疗。我们首先从现有的公开可用数据集中构建了一个新的少样本数据集，用于癌症分子亚型分类和辅助癌症分类，命名为TCGA Few-Shot。为了有效利用两个任务的相关知识，我们引入了一种特定任务的嵌入式元学习框架（TSEML）。TSEML结合了模型无关的元学习（MAML）方法和原型网络（ProtoNet）的优势，以捕捉多样且精细的特征。在TCGA Few-Shot数据集上的比较实验表明，我们的TSEML框架在少样本分子亚型分类问题上取得了优异的表现。 

---
# Physics-model-guided Worst-case Sampling for Safe Reinforcement Learning 

**Title (ZH)**: 基于物理模型的最坏情况采样方法以实现安全的强化学习 

**Authors**: Hongpeng Cao, Yanbing Mao, Lui Sha, Marco Caccamo  

**Link**: [PDF](https://arxiv.org/pdf/2412.13224)  

**Abstract**: Real-world accidents in learning-enabled CPS frequently occur in challenging corner cases. During the training of deep reinforcement learning (DRL) policy, the standard setup for training conditions is either fixed at a single initial condition or uniformly sampled from the admissible state space. This setup often overlooks the challenging but safety-critical corner cases. To bridge this gap, this paper proposes a physics-model-guided worst-case sampling strategy for training safe policies that can handle safety-critical cases toward guaranteed safety. Furthermore, we integrate the proposed worst-case sampling strategy into the physics-regulated deep reinforcement learning (Phy-DRL) framework to build a more data-efficient and safe learning algorithm for safety-critical CPS. We validate the proposed training strategy with Phy-DRL through extensive experiments on a simulated cart-pole system, a 2D quadrotor, a simulated and a real quadruped robot, showing remarkably improved sampling efficiency to learn more robust safe policies. 

**Abstract (ZH)**: 在学习增强的CPS（Cyber-Physical Systems）中，现实世界中的事故经常发生在具有挑战性的边界情况中。在深度强化学习（DRL）策略训练过程中，常用的训练条件设置要么固定在一个初始条件下，要么均匀地从可接受的状态空间中采样。这种设置往往忽视了具有挑战性但关键的安全边界情况。为了弥合这一差距，本文提出了一种基于物理模型的最坏情况采样策略，用于训练能够处理关键安全情况的安全策略，以确保安全。此外，我们将提出的最坏情况采样策略整合到受物理调节的深度强化学习（Phy-DRL）框架中，构建一种更高效且安全的学习算法，用于关键安全的CPS。我们通过在模拟Cart-Pole系统、二维四旋翼无人机、模拟四足机器人以及实际四足机器人上的广泛实验验证了所提出的训练策略，通过Phy-DRL验证，展示了显著增强的采样效率，以学习更鲁棒的安全策略。 

---
# Generative modeling of protein ensembles guided by crystallographic electron densities 

**Title (ZH)**: 由晶体学电子密度指导的蛋白质集合生成建模 

**Authors**: Sai Advaith Maddipatla, Nadav Bojan Sellam, Sanketh Vedula, Ailie Marx, Alex Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2412.13223)  

**Abstract**: Proteins are dynamic, adopting ensembles of conformations. The nature of this conformational heterogenity is imprinted in the raw electron density measurements obtained from X-ray crystallography experiments. Fitting an ensemble of protein structures to these measurements is a challenging, ill-posed inverse problem. We propose a non-i.i.d. ensemble guidance approach to solve this problem using existing protein structure generative models and demonstrate that it accurately recovers complicated multi-modal alternate protein backbone conformations observed in certain single crystal measurements. 

**Abstract (ZH)**: 蛋白质是动态的，会采用一系列构象。X射线晶体学实验中获得的原始电子密度测量值中包含这种构象异质性的本质特征。将蛋白质结构的ensemble拟合到这些测量值上是一个具有挑战性和病态的逆向问题。我们提出了一种非独立同分布（non-i.i.d.）ensemble引导方法，利用现有的蛋白质结构生成模型来解决这个问题，并证明这种方法能够准确地恢复某些单晶体测量中观察到的复杂多模态的替代蛋白质主链构象。 

---
# An introduction to reservoir computing 

**Title (ZH)**: reservoir计算简介 

**Authors**: Michael te Vrugt  

**Link**: [PDF](https://arxiv.org/pdf/2412.13212)  

**Abstract**: There is a growing interest in the development of artificial neural networks that are implemented in a physical system. A major challenge in this context is that these networks are difficult to train since training here would require a change of physical parameters rather than simply of coefficients in a computer program. For this reason, reservoir computing, where one employs high-dimensional recurrent networks and trains only the final layer, is widely used in this context. In this chapter, I introduce the basic concepts of reservoir computing. Moreover, I present some important physical implementations coming from electronics, photonics, spintronics, mechanics, and biology. Finally, I provide a brief discussion of quantum reservoir computing. 

**Abstract (ZH)**: 在物理系统中开发人工神经网络的研究兴趣正在逐渐增加。在这个背景下，这些网络的训练极具挑战性，因为训练通常需要改变物理参数，而不仅仅是在计算机程序中调整系数。因此，在这种情况下，广泛使用了一种称为储槽计算的方法，这种方法使用高维递归网络，并且仅训练最后一层。在本章中，我将介绍储槽计算的基本概念，并讨论来自电子学、光子学、自旋电子学、力学和生物学的一些重要物理实现。最后，我将简要讨论量子储槽计算。 

---
# ManiSkill-HAB: A Benchmark for Low-Level Manipulation in Home Rearrangement Tasks 

**Title (ZH)**: ManiSkill-HAB：家居重组任务中低级操作的基准测试 

**Authors**: Arth Shukla, Stone Tao, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.13211)  

**Abstract**: High-quality benchmarks are the foundation for embodied AI research, enabling significant advancements in long-horizon navigation, manipulation and rearrangement tasks. However, as frontier tasks in robotics get more advanced, they require faster simulation speed, more intricate test environments, and larger demonstration datasets. To this end, we present MS-HAB, a holistic benchmark for low-level manipulation and in-home object rearrangement. First, we provide a GPU-accelerated implementation of the Home Assistant Benchmark (HAB). We support realistic low-level control and achieve over 3x the speed of previous magical grasp implementations at similar GPU memory usage. Second, we train extensive reinforcement learning (RL) and imitation learning (IL) baselines for future work to compare against. Finally, we develop a rule-based trajectory filtering system to sample specific demonstrations from our RL policies which match predefined criteria for robot behavior and safety. Combining demonstration filtering with our fast environments enables efficient, controlled data generation at scale. 

**Abstract (ZH)**: 高质量的基准是实现具身人工智能研究的基础，能够显著推动长期导航、操作和重组任务的发展。然而，随着机器人前沿任务的不断进步，它们需要更快的仿真速度、更为复杂的测试环境以及更大的演示数据集。为此，我们提出了一种全面的基准——MS-HAB，用于低级操作和家庭中对象重组任务。首先，我们提供了一种基于GPU加速的家庭助手基准（HAB）实现。我们支持真实的低级控制，并在相似的GPU内存使用情况下，实现了比之前魔力抓取实现快3倍以上的速度。其次，我们训练了广泛的强化学习（RL）和模仿学习（IL）基线，以便于未来的工作进行对比。最后，我们开发了一种基于规则的轨迹过滤系统来从我们的RL策略中抽取具体的演示，这些演示符合预定义的机器人行为和安全标准。结合演示筛选与我们快速的环境，能够实现大规模、高效的受控数据生成。 

---
