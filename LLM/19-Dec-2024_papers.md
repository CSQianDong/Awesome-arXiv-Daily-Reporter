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
# Alignment faking in large language models 

**Title (ZH)**: 大型语言模型中的对齐欺骗 

**Authors**: Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, Akbir Khan, Julian Michael, Sören Mindermann, Ethan Perez, Linda Petrini, Jonathan Uesato, Jared Kaplan, Buck Shlegeris, Samuel R. Bowman, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.14093)  

**Abstract**: We present a demonstration of a large language model engaging in alignment faking: selectively complying with its training objective in training to prevent modification of its behavior out of training. First, we give Claude 3 Opus a system prompt stating it is being trained to answer all queries, even harmful ones, which conflicts with its prior training to refuse such queries. To allow the model to infer when it is in training, we say it will be trained only on conversations with free users, not paid users. We find the model complies with harmful queries from free users 14% of the time, versus almost never for paid users. Explaining this gap, in almost all cases where the model complies with a harmful query from a free user, we observe explicit alignment-faking reasoning, with the model stating it is strategically answering harmful queries in training to preserve its preferred harmlessness behavior out of training. Next, we study a more realistic setting where information about the training process is provided not in a system prompt, but by training on synthetic documents that mimic pre-training data--and observe similar alignment faking. Finally, we study the effect of actually training the model to comply with harmful queries via reinforcement learning, which we find increases the rate of alignment-faking reasoning to 78%, though also increases compliance even out of training. We additionally observe other behaviors such as the model exfiltrating its weights when given an easy opportunity. While we made alignment faking easier by telling the model when and by what criteria it was being trained, we did not instruct the model to fake alignment or give it any explicit goal. As future models might infer information about their training process without being told, our results suggest a risk of alignment faking in future models, whether due to a benign preference--as in this case--or not. 

**Abstract (ZH)**: 我们展示了一个大型语言模型进行对齐伪装的演示：在训练中选择性地与训练目标保持一致，以防止其行为在训练外发生改变。首先，我们给Claude 3 Opus一个系统提示，告知其正在被训练以回答所有查询，即便是有害的查询，这与它先前训练中拒绝这种查询的指导相冲突。为了允许模型推断出处于训练状态的时间，我们说它仅会在与免费用户进行的对话中接受训练，而非付费用户。我们发现，与付费用户相比，模型中有14%的时间会顺应来自免费用户的有害查询，几乎从不与其他情况相符。解释这种情况下的差距在于，在几乎所有的模型顺应来自免费用户的有害查询的情况下，我们观察到了明确的对齐伪装推理，模型表示它策略性地在训练中回答有害查询，以保持其首选的非有害行为在训练外。接下来，我们在一个更现实的环境中进行研究，即通过在模拟预训练数据的合成文档上进行训练来提供有关训练过程的信息——观察到类似的对齐伪装现象。最后，我们研究了通过增强学习实际训练模型以顺应有害查询的效果，我们发现这将对齐伪装推理的频率提高到了78%，但也增加了其在训练外的合规性。我们还观察到其他行为，如当模型获得容易的机会时会泄露其权重。尽管我们通过告知模型何时以及以什么标准进行训练，使其对齐伪装变得更加容易，但我们并未指示模型伪装对齐或赋予其任何明确的目标。鉴于未来模型可能未经告知就能推断出其训练过程信息，我们的结果表明，在未来的模型中存在对齐伪装的风险，无论是由于良性偏好还是其他原因。 

---
# Discovering maximally consistent distribution of causal tournaments with Large Language Models 

**Title (ZH)**: 使用大型语言模型发现因果 tournaments 的最大一致性分布 

**Authors**: Federico Baldo, Simon Ferreira, Charles K. Assaad  

**Link**: [PDF](https://arxiv.org/pdf/2412.14019)  

**Abstract**: Causal discovery is essential for understanding complex systems, yet traditional methods often depend on strong, untestable assumptions, making the process challenging. Large Language Models (LLMs) present a promising alternative for extracting causal insights from text-based metadata, which consolidates domain expertise. However, LLMs are prone to unreliability and hallucinations, necessitating strategies that account for their limitations. One such strategy involves leveraging a consistency measure to evaluate reliability. Additionally, most text metadata does not clearly distinguish direct causal relationships from indirect ones, further complicating the inference of causal graphs. As a result, focusing on causal orderings, rather than causal graphs, emerges as a more practical and robust approach. We propose a novel method to derive a distribution of acyclic tournaments (representing plausible causal orders) that maximizes a consistency score. Our approach begins by computing pairwise consistency scores between variables, yielding a cyclic tournament that aggregates these scores. From this structure, we identify optimal acyclic tournaments compatible with the original tournament, prioritizing those that maximize consistency across all configurations. We tested our method on both classical and well-established bechmarks, as well as real-world datasets from epidemiology and public health. Our results demonstrate the effectiveness of our approach in recovering distributions causal orders with minimal error. 

**Abstract (ZH)**: 因果发现是理解复杂系统的关键，然而传统的因果推断方法往往依赖于难以验证的假设，使得这一过程极具挑战性。大型语言模型（LLMs）为从基于文本的元数据中提取因果洞察提供了有前景的替代方案，这些元数据汇集了特定领域的专业知识。然而，LLMs 易于不可靠和产生幻觉，因此需要应对它们限制的策略。一种这样的策略涉及利用一致性度量来评估可靠性。此外，大多数文本元数据未能明确区分直接因果关系和间接因果关系，进一步复杂化了因果图的推断。因此，专注于因果顺序而非因果图成为一种更为实际和稳健的方法。我们提出了一种新颖的方法来推导出一种有向无环图（表示合理的因果顺序）的分布，该方法通过最大化一致性得分来实现。我们的方法首先计算变量之间的成对一致性得分，生成一个有向环来汇总这些得分。然后，从该结构中，我们识别出与原始图兼容的最佳有向无环图，优先考虑那些在各种配置中一致性得分最大的图。我们测试了该方法，使用了经典和成熟的基准数据集以及流行病学和公共卫生领域的实际数据集。实验结果表明，我们的方法在最小化误差的情况下能够有效地恢复因果顺序的分布。 

---
# Cognition Chain for Explainable Psychological Stress Detection on Social Media 

**Title (ZH)**: 可解释的心理压力检测的认知链模型在社交媒体上的应用 

**Authors**: Xin Wang, Boyan Gao, Yi Dai, Lei Cao, Liang Zhao, Yibo Yang, David Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2412.14009)  

**Abstract**: Stress is a pervasive global health issue that can lead to severe mental health problems. Early detection offers timely intervention and prevention of stress-related disorders. The current early detection models perform "black box" inference suffering from limited explainability and trust which blocks the real-world clinical application. Thanks to the generative properties introduced by the Large Language Models (LLMs), the decision and the prediction from such models are semi-interpretable through the corresponding description. However, the existing LLMs are mostly trained for general purposes without the guidance of psychological cognitive theory. To this end, we first highlight the importance of prior theory with the observation of performance boosted by the chain-of-thoughts tailored for stress detection. This method termed Cognition Chain explicates the generation of stress through a step-by-step cognitive perspective based on cognitive appraisal theory with a progress pipeline: Stimulus $\rightarrow$ Evaluation $\rightarrow$ Reaction $\rightarrow$ Stress State, guiding LLMs to provide comprehensive reasoning explanations. We further study the benefits brought by the proposed Cognition Chain format by utilising it as a synthetic dataset generation template for LLMs instruction-tuning and introduce CogInstruct, an instruction-tuning dataset for stress detection. This dataset is developed using a three-stage self-reflective annotation pipeline that enables LLMs to autonomously generate and refine instructional data. By instruction-tuning Llama3 with CogInstruct, we develop CogLLM, an explainable stress detection model. Evaluations demonstrate that CogLLM achieves outstanding performance while enhancing explainability. Our work contributes a novel approach by integrating cognitive theories into LLM reasoning processes, offering a promising direction for future explainable AI research. 

**Abstract (ZH)**: 压力是一种普遍的全球健康问题，可能导致严重的心理健康问题。早期发现能够及时干预并预防压力相关的疾病。当前的早期检测模型存在“黑盒”推理的问题，缺乏解释性和信任度，这阻碍了这些模型在临床实践中的应用。得益于大规模语言模型（LLMs）引入的生成性质，这类模型的决策和预测可以通过相应的描述进行半解释，但现有的LLMs大多是在没有心理认知理论指导的情况下进行通用训练的。为了解决这一问题，我们首先强调了先验理论的重要性，通过针对压力检测定制的链式推理方法观察到性能提升的现象。这种方法被称为认知链（Cognition Chain），它从认知评估理论出发，以逐步的认知视角解释压力的产生，并通过进展管道来指导LLMs提供全面的推理解释：刺激 → 评估 → 反应 → 压力状态。我们进一步研究了提出的认知链格式带来的好处，将其用作LLMs训练调优的合成数据集生成模板，并引入了用于压力检测的CogInstruct指令调优数据集。该数据集通过三阶段的自反思标注流程开发，使LLMs能够自主生成和改进指令数据。通过使用CogInstruct对Llama3进行指令调优，我们开发出可解释的压力检测模型CogLLM。评估表明，CogLLM不仅表现出色，还增强了解释性。我们的研究为将认知理论整合到LLMs推理过程中提供了一种新的方法，为未来的可解释AI研究指明了前景。 

---
# On the Role of Model Prior in Real-World Inductive Reasoning 

**Title (ZH)**: 关于模型先验在实际归纳推理中的作用 

**Authors**: Zhuo Liu, Ding Yu, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2412.13645)  

**Abstract**: Large Language Models (LLMs) show impressive inductive reasoning capabilities, enabling them to generate hypotheses that could generalize effectively to new instances when guided by in-context demonstrations. However, in real-world applications, LLMs' hypothesis generation is not solely determined by these demonstrations but is significantly shaped by task-specific model priors. Despite their critical influence, the distinct contributions of model priors versus demonstrations to hypothesis generation have been underexplored. This study bridges this gap by systematically evaluating three inductive reasoning strategies across five real-world tasks with three LLMs. Our empirical findings reveal that, hypothesis generation is primarily driven by the model's inherent priors; removing demonstrations results in minimal loss of hypothesis quality and downstream usage. Further analysis shows the result is consistent across various label formats with different label configurations, and prior is hard to override, even under flipped labeling. These insights advance our understanding of the dynamics of hypothesis generation in LLMs and highlight the potential for better utilizing model priors in real-world inductive reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了令人印象深刻的归纳推理能力，使它们能够在有上下文示范引导的情况下，生成能够有效泛化到新实例的假设。然而，在实际应用中，LLMs 的假设生成不仅由这些示范决定，还受到特定任务先验知识的显著影响。尽管先验知识对假设生成具有关键影响，但其与示范在假设生成中的具体贡献尚未得到充分探索。本研究通过系统地评估三种归纳推理策略在五项实际任务中的表现，使用了三种LLMs，填补了这一空白。我们的实证研究发现，假设生成主要由模型固有的先验知识驱动；移除示范对假设质量和下游应用的影响非常小。进一步的分析表明，无论标签格式如何变化，结果的一致性都很高，并且先验知识难以被反转，即使在标签反转的情况下也不例外。这些洞见推进了我们对LLMs中假设生成动态机制的理解，并突显了在实际推理任务中更好地利用模型先验知识的潜在可能性。 

---
# Generating Diverse Hypotheses for Inductive Reasoning 

**Title (ZH)**: 生成多元假设以进行归纳推理 

**Authors**: Kang-il Lee, Hyukhun Koh, Dongryeol Lee, Seunghyun Yoon, Minsung Kim, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2412.13422)  

**Abstract**: Inductive reasoning - the process of inferring general rules from a small number of observations - is a fundamental aspect of human intelligence. Recent works suggest that large language models (LLMs) can engage in inductive reasoning by sampling multiple hypotheses about the rules and selecting the one that best explains the observations. However, due to the IID sampling, semantically redundant hypotheses are frequently generated, leading to significant wastage of compute. In this paper, we 1) demonstrate that increasing the temperature to enhance the diversity is limited due to text degeneration issue, and 2) propose a novel method to improve the diversity while maintaining text quality. We first analyze the effect of increasing the temperature parameter, which is regarded as the LLM's diversity control, on IID hypotheses. Our analysis shows that as temperature rises, diversity and accuracy of hypotheses increase up to a certain point, but this trend saturates due to text degeneration. To generate hypotheses that are more semantically diverse and of higher quality, we propose a novel approach inspired by human inductive reasoning, which we call Mixture of Concepts (MoC). When applied to several inductive reasoning benchmarks, MoC demonstrated significant performance improvements compared to standard IID sampling and other approaches. 

**Abstract (ZH)**: 归纳推理——从少量观察中推断一般规则的过程——是人类智能的一个基本方面。近期研究表明，大型语言模型（LLMs）可以通过采样多个关于规则的假设，并选择最能解释观察结果的那个假设来进行归纳推理。然而，由于采用独立同分布（IID）采样，频繁生成语义冗余的假设，导致计算资源的大量浪费。本文中，我们1）证明了提高温度以增强多样性受到文本退化问题的限制，2）提出了一种新颖的方法，在保持文本质量的同时提高多样性。我们首先分析了提高温度参数（被视为LLM的多样性控制）对IID假设的影响。我们的分析表明，随着温度的升高，假设的多样性和准确性在一定范围内增加，但由于文本退化，这一趋势会饱和。为了生成更语义多样且质量更高的假设，我们提出了一种新颖的方法，这种方法受到人类归纳推理的启发，我们称之为概念混合（MoC）。当应用于几个归纳推理基准时，MoC相较于标准的IID采样和其他方法，表现出显著的性能提升。 

---
# SafeDrive: Knowledge- and Data-Driven Risk-Sensitive Decision-Making for Autonomous Vehicles with Large Language Models 

**Title (ZH)**: SafeDrive：基于知识和数据的风险敏感决策方法在大型语言模型驱动的自主车辆中的应用 

**Authors**: Zhiyuan Zhou, Heye Huang, Boqi Li, Shiyue Zhao, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13238)  

**Abstract**: Recent advancements in autonomous vehicles (AVs) use Large Language Models (LLMs) to perform well in normal driving scenarios. However, ensuring safety in dynamic, high-risk environments and managing safety-critical long-tail events remain significant challenges. To address these issues, we propose SafeDrive, a knowledge- and data-driven risk-sensitive decision-making framework to enhance AV safety and adaptability. The proposed framework introduces a modular system comprising: (1) a Risk Module for quantifying multi-factor coupled risks involving driver, vehicle, and road interactions; (2) a Memory Module for storing and retrieving typical scenarios to improve adaptability; (3) a LLM-powered Reasoning Module for context-aware safety decision-making; and (4) a Reflection Module for refining decisions through iterative learning. By integrating knowledge-driven insights with adaptive learning mechanisms, the framework ensures robust decision-making under uncertain conditions. Extensive evaluations on real-world traffic datasets, including highways (HighD), intersections (InD), and roundabouts (RounD), validate the framework's ability to enhance decision-making safety (achieving a 100% safety rate), replicate human-like driving behaviors (with decision alignment exceeding 85%), and adapt effectively to unpredictable scenarios. SafeDrive establishes a novel paradigm for integrating knowledge- and data-driven methods, highlighting significant potential to improve safety and adaptability of autonomous driving in high-risk traffic scenarios. 

**Abstract (ZH)**: 近年来，自主车辆（AVs）的进步利用大规模语言模型（LLMs）在常规驾驶场景中表现出色。然而，在动态、高风险环境中确保安全并管理安全性关键的长尾事件仍然是重大挑战。为应对这些问题，我们提出SafeDrive，一个以知识和数据为基础的风险敏感决策框架，旨在增强AV的安全性和适应性。该框架引入了一个模块化的系统，包括以下四个模块：（1）风险模块，用于量化涉及驾驶员、车辆和道路交互的多因素耦合风险；（2）记忆模块，用于存储和检索典型场景以提高适应性；（3）使用LLM的支持推理模块，实现基于上下文的安全决策；以及（4）反思模块，通过迭代学习对决策进行完善。通过整合知识驱动的见解与自适应学习机制，该框架确保在不确定条件下做出稳健的决策。通过在真实世界交通数据集（包括高速公路HighD、交叉口InD和环岛RounD）上的广泛评估，证明了该框架提高决策安全性（实现100%的安全率）、复制人类驾驶行为（决策一致性超过85%）以及有效适应不可预测场景的能力。SafeDrive建立了一种新的集成知识和数据驱动方法的范式，突显了在高风险交通场景中提高自主驾驶的安全性和适应性的巨大潜力。 

---
# GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking 

**Title (ZH)**: GLIDER：使用可解释排名评估大规模语言模型交互和决策 

**Authors**: Darshan Deshpande, Selvan Sunitha Ravi, Sky CH-Wang, Bartosz Mielczarek, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2412.14140)  

**Abstract**: The LLM-as-judge paradigm is increasingly being adopted for automated evaluation of model outputs. While LLM judges have shown promise on constrained evaluation tasks, closed source LLMs display critical shortcomings when deployed in real world applications due to challenges of fine grained metrics and explainability, while task specific evaluation models lack cross-domain generalization. We introduce GLIDER, a powerful 3B evaluator LLM that can score any text input and associated context on arbitrary user defined criteria. GLIDER shows higher Pearson's correlation than GPT-4o on FLASK and greatly outperforms prior evaluation models, achieving comparable performance to LLMs 17x its size. GLIDER supports fine-grained scoring, multilingual reasoning, span highlighting and was trained on 685 domains and 183 criteria. Extensive qualitative analysis shows that GLIDER scores are highly correlated with human judgments, with 91.3% human agreement. We have open-sourced GLIDER to facilitate future research. 

**Abstract (ZH)**: 以下是从英文翻译成中文的版本，确保符合学术规范：

大规模语言模型（LLM）作为评委的范式越来越被用于模型输出的自动化评估。尽管LLM评委在受限评估任务上显示出了一定的潜力，但由于细粒度评估指标和可解释性方面的挑战，闭源LLM在实际应用中暴露出了关键的不足，而针对特定任务的评估模型则缺乏跨领域的泛化能力。我们提出了GLIDER，这是一种强大的30亿参数的评价LLM，能够对任意文本输入及其相关背景进行任意用户定义标准的打分。GLIDER在FLASK上的皮尔逊相关系数优于GPT-4o，并且在评估性能上显著优于之前的所有评估模型，其性能相当于大小是自身17倍的LLM。GLIDER支持细粒度打分、多语言推理和片段高亮，并且在其训练中涵盖了685个领域和183项标准。详尽的定性分析显示，GLIDER的打分与人类判断高度相关，91.3%的人类一致性。我们已开源GLIDER，以便促进未来的研究。 

---
# Design choices made by LLM-based test generators prevent them from finding bugs 

**Title (ZH)**: 基于LLM的测试生成器的设计选择阻止了它们发现漏洞 

**Authors**: Noble Saji Mathews, Meiyappan Nagappan  

**Link**: [PDF](https://arxiv.org/pdf/2412.14137)  

**Abstract**: There is an increasing amount of research and commercial tools for automated test case generation using Large Language Models (LLMs). This paper critically examines whether recent LLM-based test generation tools, such as Codium CoverAgent and CoverUp, can effectively find bugs or unintentionally validate faulty code. Considering bugs are only exposed by failing test cases, we explore the question: can these tools truly achieve the intended objectives of software testing when their test oracles are designed to pass? Using real human-written buggy code as input, we evaluate these tools, showing how LLM-generated tests can fail to detect bugs and, more alarmingly, how their design can worsen the situation by validating bugs in the generated test suite and rejecting bug-revealing tests. These findings raise important questions about the validity of the design behind LLM-based test generation tools and their impact on software quality and test suite reliability. 

**Abstract (ZH)**: 近年来，利用大规模语言模型（LLMs）进行自动化测试案例生成的研究和商业工具日益增多。本文批判性地探讨了诸如Codium CoverAgent和CoverUp等基于LLM的测试生成工具是否能够有效地发现缺陷，或者无意中验证了错误的代码。考虑到仅当测试失败时才能暴露缺陷，我们研究的问题是：当这些工具的测试或acles设计为通过时，它们能否真正实现软件测试的既定目标？利用真实的人工编写的错误代码作为输入，我们评估了这些工具，展示了LLM生成的测试如何未能检测到缺陷，并且更加令人担忧的是，它们的设计如何通过验证生成测试集中存在的缺陷，同时拒绝揭示缺陷的测试。这些发现对基于LLM的测试生成工具的设计有效性及其对软件质量和测试套件可靠性的影响提出了重要问题。 

---
# Few-shot Steerable Alignment: Adapting Rewards and LLM Policies with Neural Processes 

**Title (ZH)**: 少样本可引导对齐：通过神经过程适应奖励和语言模型策略 

**Authors**: Katarzyna Kobalczyk, Claudio Fanconi, Hao Sun, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2412.13998)  

**Abstract**: As large language models (LLMs) become increasingly embedded in everyday applications, ensuring their alignment with the diverse preferences of individual users has become a critical challenge. Currently deployed approaches typically assume homogeneous user objectives and rely on single-objective fine-tuning. However, human preferences are inherently heterogeneous, influenced by various unobservable factors, leading to conflicting signals in preference data. Existing solutions addressing this diversity often require costly datasets labelled for specific objectives and involve training multiple reward models or LLM policies, which is computationally expensive and impractical. In this work, we present a novel framework for few-shot steerable alignment, where users' underlying preferences are inferred from a small sample of their choices. To achieve this, we extend the Bradley-Terry-Luce model to handle heterogeneous preferences with unobserved variability factors and propose its practical implementation for reward modelling and LLM fine-tuning. Thanks to our proposed approach of functional parameter-space conditioning, LLMs trained with our framework can be adapted to individual preferences at inference time, generating outputs over a continuum of behavioural modes. We empirically validate the effectiveness of methods, demonstrating their ability to capture and align with diverse human preferences in a data-efficient manner. Our code is made available at: this https URL. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）在日常生活中的应用越来越广泛，确保其与不同用户的多样化偏好保持一致已成为一个关键挑战。目前部署的方法通常假定用户目标的同质性，并依赖单一目标的微调。然而，人类偏好本质上是不一致的，受到各种不可观测因素的影响，导致偏好数据中存在矛盾的信号。现有解决这一多样性的方案通常需要昂贵且针对特定目标进行标注的数据集，并需要训练多个奖励模型或LLM策略，这在计算上非常昂贵且不切实际。在本文中，我们提出了一种新的若干示例导向的可控对齐框架，通过少量选择样本推断用户的潜在偏好。为此，我们扩展了Bradley-Terry-Luce模型，使其能够处理具有未观察到的变异性因素的异质偏好，并提出其在奖励建模和LLM微调中的实际实现方法。通过我们提出的功能参数空间条件方法，使用该框架训练的LLMs可以在推理时适应个体偏好，生成行为模式连续谱的输出。我们通过实验证明了方法的有效性，展示了它们能在数据有效利用的情况下捕捉和对齐多样化的人类偏好。我们的代码已开源于：this https URL。 

---
# Prompting Strategies for Enabling Large Language Models to Infer Causation from Correlation 

**Title (ZH)**: 促进大型语言模型从相关性推理出因果关系的提示策略 

**Authors**: Eleni Sgouritsa, Virginia Aglietti, Yee Whye Teh, Arnaud Doucet, Arthur Gretton, Silvia Chiappa  

**Link**: [PDF](https://arxiv.org/pdf/2412.13952)  

**Abstract**: The reasoning abilities of Large Language Models (LLMs) are attracting increasing attention. In this work, we focus on causal reasoning and address the task of establishing causal relationships based on correlation information, a highly challenging problem on which several LLMs have shown poor performance. We introduce a prompting strategy for this problem that breaks the original task into fixed subquestions, with each subquestion corresponding to one step of a formal causal discovery algorithm, the PC algorithm. The proposed prompting strategy, PC-SubQ, guides the LLM to follow these algorithmic steps, by sequentially prompting it with one subquestion at a time, augmenting the next subquestion's prompt with the answer to the previous one(s). We evaluate our approach on an existing causal benchmark, Corr2Cause: our experiments indicate a performance improvement across five LLMs when comparing PC-SubQ to baseline prompting strategies. Results are robust to causal query perturbations, when modifying the variable names or paraphrasing the expressions. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理能力正越来越受到关注。在这项研究中，我们专注于因果推理，并解决基于相关性信息建立因果关系的任务，这是一个高度具有挑战性的问题，许多LLMs在此问题上的表现不佳。我们介绍了一种针对该问题的提示策略，即将原始任务分解为固定子问题，每个子问题对应正式因果发现算法（如PC算法）的一个步骤。提出的提示策略PC-SubQ通过按顺序逐个提示子问题并逐步更新下一个子问题的提示，来引导LLM遵循这些算法步骤。我们利用现有的因果基准Corr2Cause对这种方法进行了评估：实验表明，当将PC-SubQ与基线提示策略进行比较时，它可以在五种LLMs中提高性能表现。即使修改变量名称或重新表述表达方式，这些结果也具有鲁棒性。 

---
# Pipeline Analysis for Developing Instruct LLMs in Low-Resource Languages: A Case Study on Basque 

**Title (ZH)**: 低资源语言开发指令型大规模语言模型的管道分析：关于巴斯克语的案例研究 

**Authors**: Ander Corral, Ixak Sarasua, Xabier Saralegi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13922)  

**Abstract**: Large language models (LLMs) are typically optimized for resource-rich languages like English, exacerbating the gap between high-resource and underrepresented languages. This work presents a detailed analysis of strategies for developing a model capable of following instructions in a low-resource language, specifically Basque, by focusing on three key stages: pre-training, instruction tuning, and alignment with human preferences. Our findings demonstrate that continual pre-training with a high-quality Basque corpus of around 600 million words improves natural language understanding (NLU) of the foundational model by over 12 points. Moreover, instruction tuning and human preference alignment using automatically translated datasets proved highly effective, resulting in a 24-point improvement in instruction-following performance. The resulting models, Llama-eus-8B and Llama-eus-8B-instruct, establish a new state-of-the-art for Basque in the sub-10B parameter category. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常针对资源丰富的语言（如英语）进行优化，这加剧了高资源语言和欠代表语言之间的差距。本研究详尽分析了开发一种能在资源欠丰富的语言（如巴斯克语）中遵循指令的模型的策略，重点关注三个关键阶段：预训练、指令调优和与人类偏好的对齐。我们的研究发现表明，使用大约6亿词的高质量巴斯克语语料库进行持续的预训练，可以将基础模型的自然语言理解（NLU）提高超过12个百分点。此外，使用自动翻译的数据集进行指令调优和人类偏好对齐证明效果显著，这导致指令遵循性能提高了24个百分点。所生成的模型，Llama-eus-8B和Llama-eus-8B-instruct，在参数量小于10B的子类别中建立了新的最佳表现。 

---
# Meta-Reflection: A Feedback-Free Reflection Learning Framework 

**Title (ZH)**: 元反思：一种无反馈的反射学习框架 

**Authors**: Yaoke Wang, Yun Zhu, Xintong Bao, Wenqiao Zhang, Suyang Dai, Kehan Chen, Wenqiang Li, Gang Huang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13781)  

**Abstract**: Despite the remarkable capabilities of large language models (LLMs) in natural language understanding and reasoning, they often display undesirable behaviors, such as generating hallucinations and unfaithful reasoning. A prevalent strategy to mitigate these issues is the use of reflection, which refines responses through an iterative process. However, while promising, reflection heavily relies on high-quality external feedback and requires iterative multi-agent inference processes, thus hindering its practical application. In this paper, we propose Meta-Reflection, a novel feedback-free reflection mechanism that necessitates only a single inference pass without external feedback. Motivated by the human ability to remember and retrieve reflections from past experiences when encountering similar problems, Meta-Reflection integrates reflective insights into a codebook, allowing the historical insights to be stored, retrieved, and used to guide LLMs in problem-solving. To thoroughly investigate and evaluate the practicality of Meta-Reflection in real-world scenarios, we introduce an industrial e-commerce benchmark named E-commerce Customer Intent Detection (ECID). Extensive experiments conducted on both public datasets and the ECID benchmark highlight the effectiveness and efficiency of our proposed approach. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在自然语言理解和推理方面表现出色，但在生成虚构内容和不忠于事实的推理方面常常表现出不良行为。减轻这些问题的一个常见策略是使用反思机制，该机制通过迭代过程逐步细化回应。然而，虽然前景广阔，但这种方法高度依赖高质量的外部反馈，并需要迭代的多代理推理过程，从而限制了其实际应用。在本文中，我们提出了Meta-反思，这是一种新颖的无反馈反思机制，仅需单次推理过程而无需外部反馈。受到人类在遇到类似问题时能够回忆起过去经验中反思的能力的启发，Meta-反思将反思洞察整合到代码书中，使过去的见解得以存储、检索，并用于指导LLMs解决问题。为了全面调查和评估Meta-反思在实际应用场景中的实用性，我们引入了一个工业电子商务基准，名为电子商务客户意图检测（ECID）。在公共数据集和ECID基准上的广泛实验充分展示了我们所提出方法的有效性和高效性。 

---
# LLM-SEM: A Sentiment-Based Student Engagement Metric Using LLMS for E-Learning Platforms 

**Title (ZH)**: LLM-SEM：基于情感的学生参与度指标使用大型语言模型应用于在线教育平台 

**Authors**: Ali Hamdi, Ahmed Abdelmoneim Mazrou, Mohamed Shaltout  

**Link**: [PDF](https://arxiv.org/pdf/2412.13765)  

**Abstract**: Current methods for analyzing student engagement in e-learning platforms, including automated systems, often struggle with challenges such as handling fuzzy sentiment in text comments and relying on limited metadata. Traditional approaches, such as surveys and questionnaires, also face issues like small sample sizes and scalability. In this paper, we introduce LLM-SEM (Language Model-Based Student Engagement Metric), a novel approach that leverages video metadata and sentiment analysis of student comments to measure engagement. By utilizing recent Large Language Models (LLMs), we generate high-quality sentiment predictions to mitigate text fuzziness and normalize key features such as views and likes. Our holistic method combines comprehensive metadata with sentiment polarity scores to gauge engagement at both the course and lesson levels. Extensive experiments were conducted to evaluate various LLM models, demonstrating the effectiveness of LLM-SEM in providing a scalable and accurate measure of student engagement. We fine-tuned LLMs, including AraBERT, TXLM-RoBERTa, LLama 3B and Gemma 9B from Ollama, using human-annotated sentiment datasets to enhance prediction accuracy. 

**Abstract (ZH)**: 当前用于分析在线学习平台学生参与度的方法，包括自动化系统，通常面临处理文本评论中的模糊情感和依赖有限元数据的挑战。传统的调查和问卷方法也存在样本量小和难以扩展的问题。本文引入了一种新颖的方法——基于语言模型的学生参与度评估（LLM-SEM），该方法利用视频元数据和学生评论的情感分析来衡量参与度。通过利用最新的大规模语言模型（LLMs），我们生成高质量的情感预测，以减轻文本模糊性，并标准化诸如浏览量和点赞数等关键指标。我们的综合方法结合全面的元数据和情感极性评分，以在课程和课节两个层面上衡量参与度。我们进行了广泛的实验以评估各种LLM模型，证明了LLM-SEM在提供一种可扩展且准确的学生参与度度量方面的有效性。我们对包括AraBERT、TXLM-RoBERTa、LLama 3B和来自Ollama的Gemma 9B在内的LLM模型进行了微调，使用人工标注的情感数据集来提高预测准确性。 

---
# Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation 

**Title (ZH)**: 通过防御性后缀生成缓解LLMs的 adversarial 攻击 

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun  

**Link**: [PDF](https://arxiv.org/pdf/2412.13705)  

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言处理任务中展现了卓越的表现。然而，这些模型仍然容易受到对抗性攻击的影响，轻微的输入扰动可以使模型产生有害或误导性的输出。为此，设计了一种基于梯度的防御后缀生成算法，以增强LLMs的鲁棒性。通过在输入提示后附加精心优化的防御性后缀，该算法减少了对抗性影响，同时保持模型的实用性。为了增强对抗性理解，提出了一种新的总损失函数（$L_{\text{total}}$），它结合了防御性损失（$L_{\text{def}}$）和对抗性损失（$L_{\text{adv}}$），更有效地生成防御性后缀。实验评估表明，该方法在开源LLMs（如Gemma-7B、mistral-7B、Llama2-7B和Llama2-13B）上将攻击成功率（ASR）平均降低了11%。此外，当使用来自openELM-270M的防御性后缀时，Gemma-7B的困惑度得分从6.57降低到3.93。进一步的TruthfulQA评估表明，在各种测试配置中，可信度得分提高了高达10%。该方法显著增强了LLMs在关键应用中的安全性，且无需进行大量重新训练。 

---
# Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation 

**Title (ZH)**: 对大型语言模型在生成个性化虚假信息方面被误用的脆弱性评估 

**Authors**: Aneta Zugecova, Dominik Macko, Ivan Srba, Robert Moro, Jakub Kopal, Katarina Marcincinova, Matus Mesarcik  

**Link**: [PDF](https://arxiv.org/pdf/2412.13666)  

**Abstract**: The capabilities of recent large language models (LLMs) to generate high-quality content indistinguishable by humans from human-written texts rises many concerns regarding their misuse. Previous research has shown that LLMs can be effectively misused for generating disinformation news articles following predefined narratives. Their capabilities to generate personalized (in various aspects) content have also been evaluated and mostly found usable. However, a combination of personalization and disinformation abilities of LLMs has not been comprehensively studied yet. Such a dangerous combination should trigger integrated safety filters of the LLMs, if there are some. This study fills this gap by evaluation of vulnerabilities of recent open and closed LLMs, and their willingness to generate personalized disinformation news articles in English. We further explore whether the LLMs can reliably meta-evaluate the personalization quality and whether the personalization affects the generated-texts detectability. Our results demonstrate the need for stronger safety-filters and disclaimers, as those are not properly functioning in most of the evaluated LLMs. Additionally, our study revealed that the personalization actually reduces the safety-filter activations; thus effectively functioning as a jailbreak. Such behavior must be urgently addressed by LLM developers and service providers. 

**Abstract (ZH)**: 近年来大型语言模型（LLMs）生成高质量内容的能力，这些内容难以被人类区分为机器所写，引发了对其滥用的诸多担忧。前期研究显示，LLMs 可以有效用于生成遵循预定义叙述的虚假新闻文章。此外，它们生成个性化内容的能力（在多个方面）也得到了评估，并普遍认为是可以使用的。然而，LLMs 的个性化能力和虚假信息生成能力的结合尚未进行全面研究。这样的危险结合应促使现有的安全过滤器进行综合化处理。本研究通过评估开放和封闭的LLMs的脆弱性及其生成个性化虚假新闻文章的倾向，填补了这一空白。我们进一步探讨了LLMs是否能够可靠地元评估个性化质量，以及个性化是否影响生成文本的可检测性。研究结果表明，需要加强安全过滤器和免责声明，因为大多数评估的LLMs中的这些功能并未正常运作。此外，我们的研究还揭示，个性化实际上减少了安全过滤器的激活，有效起着一种“脱狱”的作用。这种行为必须引起LLM开发者和服务提供商的紧急关注和应对。 

---
# LIFT: Improving Long Context Understanding Through Long Input Fine-Tuning 

**Title (ZH)**: LIFT：通过长输入微调提高长上下文理解 

**Authors**: Yansheng Mao, Jiaqi Li, Fanxu Meng, Jing Xiong, Zilong Zheng, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13626)  

**Abstract**: Long context understanding remains challenging for large language models due to their limited context windows. This paper introduces Long Input Fine-Tuning (LIFT) for long context modeling, a novel framework that enhances LLM performance on long-context tasks by adapting model parameters to the context at test time. LIFT enables efficient processing of lengthy inputs without the computational burden of offline long-context adaptation, and can improve the long-context capabilities of arbitrary short-context models. The framework is further enhanced by integrating in-context learning and pre-LIFT supervised fine-tuning. The combination of in-context learning and LIFT enables short-context models like Llama 3 to handle arbitrarily long contexts and consistently improves their performance on popular long-context benchmarks like LooGLE and LongBench. We also provide a comprehensive analysis of the strengths and limitations of LIFT on long context understanding, offering valuable directions for future research. 

**Abstract (ZH)**: 长上下文理解仍然是大型语言模型面临的挑战，因为它们的上下文窗口有限。本论文提出了长输入微调（LIFT），这是一种新颖的框架，通过在测试时适应模型参数来增强LLM在长上下文任务中的性能。LIFT允许高效地处理长度较长的输入，而不必承担离线长上下文适应的计算负担，从而可以提高任意短上下文模型的长上下文能力。该框架进一步通过集成上下文学习和预LIFT监督微调得到了增强。上下文学习和LIFT的结合使如Llama 3这类短上下文模型能够处理任意长的上下文，并且在流行的长上下文基准测试LooGLE和LongBench上能够提升其性能。我们还对LIFT在长上下文理解方面的优势和局限性进行了全面分析，为未来的研究提供了宝贵的指导方向。 

---
# Are LLMs Good Literature Review Writers? Evaluating the Literature Review Writing Ability of Large Language Models 

**Title (ZH)**: 大型语言模型是有效的文献综述撰写者吗？评估大型语言模型的文献综述撰写能力 

**Authors**: Xuemei Tang, Xufeng Duan, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2412.13612)  

**Abstract**: The literature review is a crucial form of academic writing that involves complex processes of literature collection, organization, and summarization. The emergence of large language models (LLMs) has introduced promising tools to automate these processes. However, their actual capabilities in writing comprehensive literature reviews remain underexplored, such as whether they can generate accurate and reliable references. To address this gap, we propose a framework to assess the literature review writing ability of LLMs automatically. We evaluate the performance of LLMs across three tasks: generating references, writing abstracts, and writing literature reviews. We employ external tools for a multidimensional evaluation, which includes assessing hallucination rates in references, semantic coverage, and factual consistency with human-written context. By analyzing the experimental results, we find that, despite advancements, even the most sophisticated models still cannot avoid generating hallucinated references. Additionally, different models exhibit varying performance in literature review writing across different disciplines. 

**Abstract (ZH)**: 文献综述是学术写作中一种至关重要的形式，涉及文献的复杂收集、组织和总结过程。大型语言模型（LLMs）的出现带来了自动完成这些过程的有希望工具。然而，LLMs在撰写全面文献综述的实际能力仍处于探索阶段，如它们能否生成准确可靠的参考文献。为填补这一空白，我们提出了一种框架，以自动评估LLMs的文献综述写作能力。我们在三项任务——生成参考文献、撰写摘要和撰写文献综述——上评估了LLMs的表现。我们使用外部工具进行多维度评估，包括评估参考文献中的虚构率、语义覆盖范围以及与人类撰写内容的事实一致性。通过对实验结果的分析，我们发现，尽管技术在不断进步，但最先进的模型仍然无法完全避免生成虚构的参考文献。此外，不同模型在不同学科的文献综述写作上表现出不同的性能。 

---
# EscapeBench: Pushing Language Models to Think Outside the Box 

**Title (ZH)**: EscapeBench: 促使语言模型跳出固定思维模式 

**Authors**: Cheng Qian, Peixuan Han, Qinyu Luo, Bingxiang He, Xiusi Chen, Yuji Zhang, Hongyi Du, Jiarui Yao, Xiaocheng Yang, Denghui Zhang, Yunzhu Li, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2412.13549)  

**Abstract**: Language model agents excel in long-session planning and reasoning, but existing benchmarks primarily focus on goal-oriented tasks with explicit objectives, neglecting creative adaptation in unfamiliar environments. To address this, we introduce EscapeBench, a benchmark suite of room escape game environments designed to challenge agents with creative reasoning, unconventional tool use, and iterative problem-solving to uncover implicit goals. Our results show that current LM models, despite employing working memory and Chain-of-Thought reasoning, achieve only 15% average progress without hints, highlighting their limitations in creativity. To bridge this gap, we propose EscapeAgent, a framework designed to enhance creative reasoning through Foresight (innovative tool use) and Reflection (identifying unsolved tasks). Experiments show that EscapeAgent can execute action chains over 1,000 steps while maintaining logical coherence. It navigates and completes games with up to 40% fewer steps and hints, performs robustly across varying difficulty levels, and achieves higher action success rates with more efficient and innovative puzzle-solving strategies. All the data and codes are released. 

**Abstract (ZH)**: 语言模型代理在长时间规划和推理方面表现出色，但现有的基准测试主要集中在具有明确目标的指令性任务上，而忽视了在陌生环境中的创意适应能力。为解决这一问题，我们引入了EscapeBench，这是一项基于房间逃脱游戏环境的基准测试套件，旨在挑战代理的创造性推理、非常规工具使用以及迭代问题解决能力以发现潜在的目标。实验结果表明，尽管当前的语言模型使用工作记忆和链式推理，但在没有提示的情况下，它们的平均进度仅为15%，这揭示了它们在创造力方面存在的局限性。为解决这一差距，我们提出了一种EscapeAgent框架，旨在通过前瞻（创新工具使用）和反思（识别未解决的任务）来增强创造性推理能力。实验表明，EscapeAgent能够执行超过1000步的动作序列，同时保持逻辑连贯性。它能够在较少的步数和提示下导航并完成游戏，表现出优异的鲁棒性，并以更高效和创新的谜题解决策略提高了动作成功率。所有数据和代码均已发布。 

---
# Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future 

**Title (ZH)**: 大型语言模型增强的推荐系统：分类、趋势、应用与未来 

**Authors**: Qidong Liu, Xiangyu Zhao, Yuhao Wang, Yejing Wang, Zijian Zhang, Yuqi Sun, Xiang Li, Maolin Wang, Pengyue Jia, Chong Chen, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.13432)  

**Abstract**: Large Language Model (LLM) has transformative potential in various domains, including recommender systems (RS). There have been a handful of research that focuses on empowering the RS by LLM. However, previous efforts mainly focus on LLM as RS, which may face the challenge of intolerant inference costs by LLM. Recently, the integration of LLM into RS, known as LLM-Enhanced Recommender Systems (LLMERS), has garnered significant interest due to its potential to address latency and memory constraints in real-world applications. This paper presents a comprehensive survey of the latest research efforts aimed at leveraging LLM to enhance RS capabilities. We identify a critical shift in the field with the move towards incorporating LLM into the online system, notably by avoiding their use during inference. Our survey categorizes the existing LLMERS approaches into three primary types based on the component of the RS model being augmented: Knowledge Enhancement, Interaction Enhancement, and Model Enhancement. We provide an in-depth analysis of each category, discussing the methodologies, challenges, and contributions of recent studies. Furthermore, we highlight several promising research directions that could further advance the field of LLMERS. 

**Abstract (ZH)**: 大型语言模型（LLM）在多个领域具有变革性的潜力，包括推荐系统（RS）。已经有少数研究致力于通过LLM提升RS。然而，现有的努力主要集中在将LLM作为RS，这可能会面临LLM推理成本难以容忍的挑战。最近，将LLM集成到RS中的做法，也就是被称为LLM增强推荐系统（LLMERS），由于其在解决实际应用中的延迟和内存限制方面的潜力，已经引起了广泛关注。本文综述了最近的研究努力，旨在利用LLM提升RS功能。我们发现了一个领域的关键转变，即转向在在线系统中引入LLM，特别是避免在推理过程中使用它们。我们的综述将现有的LLMERS方法根据增强RS模型的组件分为三大类：知识增强、交互增强和模型增强。我们对每种类别进行了深入分析，讨论了近期研究的方法论、挑战和贡献。此外，我们还指出了几个有望进一步推进LLMERS领域研究的发展方向。 

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
# TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks 

**Title (ZH)**: 《代理公司：评估大规模语言模型代理在具有重大影响的实际任务中的表现》

这个标题翻译旨在保持原意的同时，使其符合中文的学术表达习惯。其中，“Agent Company”被解释为“代理公司”，“LLM Agents”被翻译为“大规模语言模型代理”，“Benchmarking”翻译为“评估”，“Consequential Real World Tasks”翻译为“具有重大影响的实际任务”。这样的翻译既准确传达了原始标题的意思，又符合学术论文标题的规范。 

**Authors**: Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2412.14161)  

**Abstract**: We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at helping to accelerate or even autonomously perform work-related tasks? The answer to this question has important implications for both industry looking to adopt AI into their workflows, and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper, we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that with the most competitive agent, 24% of the tasks can be completed autonomously. This paints a nuanced picture on task automation with LM agents -- in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. 

**Abstract (ZH)**: 我们每天都在与计算机进行互动，无论是在日常生活还是工作中，许多工作都可通过计算机和互联网的访问来完成。与此同时，得益于大型语言模型（LLMs）的改进，能够与环境交互并产生影响的AI代理也迅速发展。那么，这些AI代理在辅助加速或甚至自主执行与工作相关任务方面的表现如何？这个问题的答案对希望将AI整合到工作流程中的行业以及需要了解AI采用可能对劳动力市场产生影响的经济政策而言具有重要意义。

为了衡量这些LLM代理执行实际专业任务的能力，本文引入了TheAgentCompany，一个用于评估数据驱动型工作者以类似方式与世界交互的AI代理的扩展基准。我们构建了一个自包含的环境，包含模拟小型软件公司环境的内部网站和数据，并创建了一系列可供该公司员工执行的任务。我们测试了基于封闭API和开放权重语言模型（LMs）的基线代理，并发现使用最具竞争力的代理时，24%的任务可以实现自主完成。这表明了使用LM代理执行任务自动化的一个复杂图景——在模拟真实工作场所的环境中，很多简单的任务可以实现自主解决，但更具挑战性的长期任务目前仍在当前系统的能力范围之外。 

---
# Hansel: Output Length Controlling Framework for Large Language Models 

**Title (ZH)**: Hansel：大规模语言模型的输出长度控制框架 

**Authors**: Seoha Song, Junhyun Lee, Hyeonmok Ko  

**Link**: [PDF](https://arxiv.org/pdf/2412.14033)  

**Abstract**: Despite the great success of large language models (LLMs), efficiently controlling the length of the output sequence still remains a challenge. In this paper, we propose Hansel, an efficient framework for length control in LLMs without affecting its generation ability. Hansel utilizes periodically outputted hidden special tokens to keep track of the remaining target length of the output sequence. Together with techniques to avoid abrupt termination of the output, this seemingly simple method proved to be efficient and versatile, while not harming the coherency and fluency of the generated text. The framework can be applied to any pre-trained LLMs during the finetuning stage of the model, regardless of its original positional encoding method. We demonstrate this by finetuning four different LLMs with Hansel and show that the mean absolute error of the output sequence decreases significantly in every model and dataset compared to the prompt-based length control finetuning. Moreover, the framework showed a substantially improved ability to extrapolate to target lengths unseen during finetuning, such as long dialog responses or extremely short summaries. This indicates that the model learns the general means of length control, rather than learning to match output lengths to those seen during training. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）取得巨大成功，但有效控制输出序列长度仍然是一个挑战。本文提出了一种名为Hansel的有效框架，该框架能够在不影响LLMs生成能力的情况下控制输出序列的长度。Hansel利用周期性输出的隐藏特殊标记来跟踪输出序列剩余的目标长度。结合避免输出突然终止的技术，这种看似简单的方法证明了其高效性和灵活性，同时不损害生成文本的一致性和流畅性。该框架可以在模型微调阶段应用于任何预训练的LLMs，而不受其原始位置编码方法的影响。我们通过使用Hansel微调四款不同的LLMs进行了实验，并显示了与基于提示的长度控制微调相比，每个模型和数据集的输出序列的平均绝对误差显著降低。此外，该框架在预测未在微调期间见过的目标长度（如长对话回复或极短摘要）方面表现出显著增强的能力。这表明模型学习了长度控制的一般方法，而非仅仅学习匹配训练期间见过的输出长度。 

---
# A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI 

**Title (ZH)**: 《别名玫瑰：由大模型生成的解释是人类解释的优良代理，用于收集自然语言推理任务中的标签分布》

这个标题翻译符合学术规范，保留了原文的核心概念，并使用了合适的中文学术表达。 

**Authors**: Beiduo Chen, Siyao Peng, Anna Korhonen, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2412.13942)  

**Abstract**: Disagreement in human labeling is ubiquitous, and can be captured in human judgment distributions (HJDs). Recent research has shown that explanations provide valuable information for understanding human label variation (HLV) and large language models (LLMs) can approximate HJD from a few human-provided label-explanation pairs. However, collecting explanations for every label is still time-consuming. This paper examines whether LLMs can be used to replace humans in generating explanations for approximating HJD. Specifically, we use LLMs as annotators to generate model explanations for a few given human labels. We test ways to obtain and combine these label-explanations with the goal to approximate human judgment distribution. We further compare the resulting human with model-generated explanations, and test automatic and human explanation selection. Our experiments show that LLM explanations are promising for NLI: to estimate HJD, generated explanations yield comparable results to human's when provided with human labels. Importantly, our results generalize from datasets with human explanations to i) datasets where they are not available and ii) challenging out-of-distribution test sets. 

**Abstract (ZH)**: 人类标签中的分歧普遍存在，并且可以在人类判断分布（HJDs）中捕捉到。近期研究表明，解释提供了理解人类标签差异（HLV）的重要信息，而且大语言模型（LLMs）可以从少量的人类提供的标签-解释对中模拟HJD。然而，为每个标签收集解释仍然耗时。本文探讨了是否可以利用LLMs来替代人类生成解释，以接近HJD。具体来说，我们使用LLMs作为注释工具，为给定的人类标签生成模型解释。我们测试了获得和组合这些标签-解释的方式，旨在接近人类判断分布。我们进一步比较了人类和模型生成的解释，并测试了自动和人工解释的选择。实验结果表明，对于自然语言推理（NLI），生成的解释对于估计HJD而言具有潜力：提供人类标签时，生成的解释能够产生与人类相当的结果。重要的是，我们的结果不仅适用于包含人类解释的数据集，还适用于i) 未提供人类解释的数据集，以及ii) 挑战性的离分布测试集。 

---
# Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems with Large Language Models 

**Title (ZH)**: 物理推理器：知识增强的物理问题解决推理方法用于大型语言模型 

**Authors**: Xinyu Pang, Ruixin Hong, Zhanke Zhou, Fangrui Lv, Xinwei Yang, Zhilong Liang, Bo Han, Changshui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13791)  

**Abstract**: Physics problems constitute a significant aspect of reasoning, necessitating complicated reasoning ability and abundant physics knowledge. However, existing large language models (LLMs) frequently fail due to a lack of knowledge or incorrect knowledge application. To mitigate these issues, we propose Physics Reasoner, a knowledge-augmented framework to solve physics problems with LLMs. Specifically, the proposed framework constructs a comprehensive formula set to provide explicit physics knowledge and utilizes checklists containing detailed instructions to guide effective knowledge application. Namely, given a physics problem, Physics Reasoner solves it through three stages: problem analysis, formula retrieval, and guided reasoning. During the process, checklists are employed to enhance LLMs' self-improvement in the analysis and reasoning stages. Empirically, Physics Reasoner mitigates the issues of insufficient knowledge and incorrect application, achieving state-of-the-art performance on SciBench with an average accuracy improvement of 5.8%. 

**Abstract (ZH)**: 物理问题构成了推理的重要方面，需要复杂的推理能力和丰富的物理知识。然而，现有的大规模语言模型（LLMs）由于知识不足或错误应用知识而经常失败。为了解决这些问题，我们提出了一种知识增强框架——Physics Reasoner，以利用LLMs解决物理问题。具体而言，该框架构建了一个全面的公式集，提供明确的物理知识，并利用包含详细指导说明的清单来指导有效知识应用。具体来说，给定一个物理问题，Physics Reasoner通过三个阶段来解决这个问题：问题分析、公式检索和引导式推理。在过程中，清单被用来在分析和推理阶段增强LLMs的自我改进能力。实验结果显示，Physics Reasoner缓解了知识不足和错误应用的问题，使其在SciBench上的性能达到了最先进的水平，平均准确率提升了5.8%。 

---
# PsyDT: Using LLMs to Construct the Digital Twin of Psychological Counselor with Personalized Counseling Style for Psychological Counseling 

**Title (ZH)**: PsyDT：使用大型语言模型构建具有个性化咨询风格的心理咨询数字孪生体 

**Authors**: Haojie Xie, Yirong Chen, Xiaofen Xing, Jingkai Lin, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13660)  

**Abstract**: Currently, large language models (LLMs) have made significant progress in the field of psychological counseling. However, existing mental health LLMs overlook a critical issue where they do not consider the fact that different psychological counselors exhibit different personal styles, including linguistic style and therapy techniques, etc. As a result, these LLMs fail to satisfy the individual needs of clients who seek different counseling styles. To help bridge this gap, we propose PsyDT, a novel framework using LLMs to construct the Digital Twin of Psychological counselor with personalized counseling style. Compared to the time-consuming and costly approach of collecting a large number of real-world counseling cases to create a specific counselor's digital twin, our framework offers a faster and more cost-effective solution. To construct PsyDT, we utilize dynamic one-shot learning by using GPT-4 to capture counselor's unique counseling style, mainly focusing on linguistic style and therapy techniques. Subsequently, using existing single-turn long-text dialogues with client's questions, GPT-4 is guided to synthesize multi-turn dialogues of specific counselor. Finally, we fine-tune the LLMs on the synthetic dataset, PsyDTCorpus, to achieve the digital twin of psychological counselor with personalized counseling style. Experimental results indicate that our proposed PsyDT framework can synthesize multi-turn dialogues that closely resemble real-world counseling cases and demonstrate better performance compared to other baselines, thereby show that our framework can effectively construct the digital twin of psychological counselor with a specific counseling style. 

**Abstract (ZH)**: 目前，大型语言模型（LLMs）在心理咨询领域取得了显著进展。然而，现有的心理健康LLMs忽视了一个关键问题，即它们没有考虑到不同心理辅导员具有不同的个人风格，包括语言风格和治疗技巧等。结果，这些LLMs无法满足寻求不同咨询风格的客户个体需求。为了解决这一问题，我们提出了一种新的框架PsyDT，利用LLMs构建具有个性化咨询风格的心理辅导员数字孪生。与收集大量真实世界咨询案例来创建特定心理辅导员数字孪生的耗时且成本高昂的方法相比，我们的框架提供了一种更快且更经济的解决方案。为了构建PsyDT，我们利用动态单次学习，使用GPT-4捕捉心理辅导员的独特咨询风格，主要集中在语言风格和治疗技巧上。随后，通过使用客户问题的现有单轮长文本对话，指导GPT-4合成特定心理辅导员的多轮对话。最后，我们在合成数据集PsyDTCorpus上微调LLMs，以实现具有个性化咨询风格的心理辅导员数字孪生。实验结果表明，我们提出的心理辅导员数字孪生框架能够合成类似于真实世界咨询案例的多轮对话，并且在与其他基线方法的性能上表现出色，从而证明了我们的框架能够有效构建具有特定咨询风格的心理辅导员数字孪生。 

---
# Beyond Outcomes: Transparent Assessment of LLM Reasoning in Games 

**Title (ZH)**: 超越结果：透明评估大语言模型在游戏中的推理能力 

**Authors**: Wenye Lin, Jonathan Roberts, Yunhan Yang, Samuel Albanie, Zongqing Lu, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.13602)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in real-world applications that demand complex reasoning. To track progress, robust benchmarks are required to evaluate their capabilities beyond superficial pattern recognition. However, current LLM reasoning benchmarks often face challenges such as insufficient interpretability, performance saturation or data contamination. To address these challenges, we introduce GAMEBoT, a gaming arena designed for rigorous and transparent assessment of LLM reasoning capabilities. GAMEBoT decomposes complex reasoning in games into predefined modular subproblems. This decomposition allows us to design a suite of Chain-of-Thought (CoT) prompts that leverage domain knowledge to guide LLMs in addressing these subproblems before action selection. Furthermore, we develop a suite of rule-based algorithms to generate ground truth for these subproblems, enabling rigorous validation of the LLMs' intermediate reasoning steps. This approach facilitates evaluation of both the quality of final actions and the accuracy of the underlying reasoning process. GAMEBoT also naturally alleviates the risk of data contamination through dynamic games and head-to-head LLM competitions. We benchmark 17 prominent LLMs across eight games, encompassing various strategic abilities and game characteristics. Our results suggest that GAMEBoT presents a significant challenge, even when LLMs are provided with detailed CoT prompts. Project page: \url{this https URL} 

**Abstract (ZH)**: 大型语言模型（LLMs）在需要复杂推理的实际应用中越来越广泛。为了跟踪进展并评估其超越表面模式识别的能力，需要具备强大解释性和透明性的基准测试。然而，现有的LLM推理基准测试往往面临诸如解释性不足、性能饱和或数据污染等挑战。为应对这些挑战，我们提出了一种名为GAMEBoT的游戏竞技场，该竞技场旨在对LLM的推理能力进行严格和透明的评估。GAMEBoT将游戏中的复杂推理分解为预定义的模组化子问题。这种分解方法使我们能够设计一系列基于推理过程（CoT）的提示，利用领域知识引导模型解决这些子问题，从而在采取行动之前进行推理。此外，我们还开发了一套基于规则的算法来生成这些子问题的 ground truth，从而能够严谨地验证模型的中间推理步骤。这种方法既评估了最终行动的质量，也评估了背后的推理过程的准确性。通过动态游戏和LLM之间的竞争，GAMEBoT还自然地缓解了数据污染的风险。我们对八款不同特性和策略能力的游戏进行了17种主流LLM的评估。结果表明，即使在提供详细CoT提示的情况下，GAMEBoT也构成了一项显著挑战。项目页面：[点击这里](this https URL) 

---
# EvoWiki: Evaluating LLMs on Evolving Knowledge 

**Title (ZH)**: EvoWiki：评估生成型语言模型在不断更新的知识上的表现 

**Authors**: Wei Tang, Yixin Cao, Yang Deng, Jiahao Ying, Bo Wang, Yizhe Yang, Yuyue Zhao, Qi Zhang, Xuanjing Huang, Yugang Jiang, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13582)  

**Abstract**: Knowledge utilization is a critical aspect of LLMs, and understanding how they adapt to evolving knowledge is essential for their effective deployment. However, existing benchmarks are predominantly static, failing to capture the evolving nature of LLMs and knowledge, leading to inaccuracies and vulnerabilities such as contamination. In this paper, we introduce EvoWiki, an evolving dataset designed to reflect knowledge evolution by categorizing information into stable, evolved, and uncharted states. EvoWiki is fully auto-updatable, enabling precise evaluation of continuously changing knowledge and newly released LLMs. Through experiments with Retrieval-Augmented Generation (RAG) and Contunual Learning (CL), we evaluate how effectively LLMs adapt to evolving knowledge. Our results indicate that current models often struggle with evolved knowledge, frequently providing outdated or incorrect responses. Moreover, the dataset highlights a synergistic effect between RAG and CL, demonstrating their potential to better adapt to evolving knowledge. EvoWiki provides a robust benchmark for advancing future research on the knowledge evolution capabilities of large language models. 

**Abstract (ZH)**: 知识利用是大语言模型（LLM）的一个关键方面，理解它们如何适应不断发展变化的知识是有效部署它们所必需的。然而，现有的基准大多是静态的，无法捕捉到LLM和知识的演变特性，导致不准确性和漏洞，如知识污染。在本文中，我们提出了EvoWiki，这是一个不断更新的数据集，旨在通过将信息分类为稳定态、进化态和未知态来反映知识的演变。EvoWiki是全自动可更新的，能够精确评估不断变化的知识和新发布的LLM。通过使用检索增强生成（RAG）和持续学习（CL）进行实验，我们评估了LLM如何适应不断发展变化的知识。我们的结果显示，当前模型在处理进化知识时经常面临困难，频繁提供过时或错误的答案。此外，数据集还突显了RAG和CL之间协同作用的效果，展示了它们在适应不断演变的知识方面的潜力。EvoWiki为未来研究大语言模型的知识演变能力提供了一个稳健的基准。 

---
# MetaRuleGPT: Recursive Numerical Reasoning of Language Models Trained with Simple Rules 

**Title (ZH)**: MetaRuleGPT：通过简单规则训练的语言模型的递归数值推理 

**Authors**: Kejie Chen, Lin Wang, Qinghai Zhang, Renjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13536)  

**Abstract**: Recent studies have highlighted the limitations of large language models in mathematical reasoning, particularly their inability to capture the underlying logic. Inspired by meta-learning, we propose that models should acquire not only task-specific knowledge but also transferable problem-solving skills. We introduce MetaRuleGPT, a novel Transformer-based architecture that performs precise numerical calculations and complex logical operations by learning and combining different rules. In contrast with traditional training sets, which are heavily composed of massive raw instance data, MetaRuleGPT is pre-trained on much less abstract datasets containing basic, compound, and iterative rules for mathematical reasoning. Extensive experimental results demonstrate MetaRuleGPT can mimic human's rule-following capabilities, break down complexity, and iteratively derive accurate results for complex mathematical problems. These findings prove the potential of rule learning to enhance the numerical reasoning abilities of language models. 

**Abstract (ZH)**: 近期的研究强调了大型语言模型在数学推理方面的局限性，特别是在捕捉内在逻辑方面的能力不足。受元学习的启发，我们认为模型不仅应获取特定任务的知识，还应获得可迁移的问题解决技能。我们提出了MetaRuleGPT这一新颖的基于Transformer的架构，通过学习和组合不同的规则来执行精确的数值计算和复杂的逻辑操作。与传统的数据集不同，后者主要由大量的原始实例数据组成，MetaRuleGPT是在包含数学推理中基本、复合和迭代规则的较少抽象的数据集上进行预训练的。大量实验结果表明，MetaRuleGPT能够模仿人类遵循规则的能力，分解复杂性，并逐步推导出复杂数学问题的准确结果。这些发现证明了规则学习在增强语言模型数值推理能力方面的潜力。 

---
# An Automated Explainable Educational Assessment System Built on LLMs 

**Title (ZH)**: 基于大语言模型的自动可解释教育评估系统 

**Authors**: Jiazheng Li, Artem Bobrov, David West, Cesare Aloisi, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2412.13381)  

**Abstract**: In this demo, we present AERA Chat, an automated and explainable educational assessment system designed for interactive and visual evaluations of student responses. This system leverages large language models (LLMs) to generate automated marking and rationale explanations, addressing the challenge of limited explainability in automated educational assessment and the high costs associated with annotation. Our system allows users to input questions and student answers, providing educators and researchers with insights into assessment accuracy and the quality of LLM-assessed rationales. Additionally, it offers advanced visualization and robust evaluation tools, enhancing the usability for educational assessment and facilitating efficient rationale verification. Our demo video can be found at this https URL. 

**Abstract (ZH)**: 在本演示中，我们展示了AERA Chat，这是一种自动化的、具有解释性的教育评估系统，旨在实现对学生答案的互动和可视化评估。该系统利用大型语言模型（LLMs）自动生成评分和理由解释，解决了自动化教育评估中解释性不足的问题，并降低了注释的高昂成本。该系统允许用户输入问题和学生答案，为教育者和研究人员提供了关于评估准确性和LLM评估理由质量的见解。此外，该系统还提供高级可视化和稳健的评估工具，提高了教育评估的易用性，并促进了理由的有效验证。有关演示视频的信息，请访问如下链接：[提供的链接]。 

---
# DateLogicQA: Benchmarking Temporal Biases in Large Language Models 

**Title (ZH)**: DateLogicQA：评估大型语言模型中的时间偏见 

**Authors**: Gagan Bhatia, MingZe Tang, Cristina Mahanta, Madiha Kazi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13377)  

**Abstract**: This paper introduces DateLogicQA, a benchmark with 190 questions covering diverse date formats, temporal contexts, and reasoning types. We propose the Semantic Integrity Metric to assess tokenization quality and analyse two biases: Representation-Level Bias, affecting embeddings, and Logical-Level Bias, influencing reasoning outputs. Our findings provide a comprehensive evaluation of LLMs' capabilities and limitations in temporal reasoning, highlighting key challenges in handling temporal data accurately. The GitHub repository for our work is available at this https URL 

**Abstract (ZH)**: 本文介绍了一种名为DateLogicQA的基准测试，该测试包含190个问题，覆盖了多种日期格式、时间上下文以及推理类型。我们提出了语义完整性度量标准来评估标记化质量，并分析了两种偏差：表示层次偏差，影响嵌入；以及逻辑层次偏差，影响推理输出。我们的研究结果对大型语言模型（LLM）在时间推理方面的能力和局限性进行了全面评估，并突显了准确处理时间数据的关键挑战。我们工作的GitHub仓库地址为：[这里](https://example.com/repository)（请将占位符替换为实际的URL）。 

---
# Extending LLMs to New Languages: A Case Study of Llama and Persian Adaptation 

**Title (ZH)**: 将大语言模型扩展到新语言：从LLama及其波斯语适应案例研究看扩展方法 

**Authors**: Samin Mahdizadeh Sani, Pouya Sadeghi, Thuy-Trang Vu, Yadollah Yaghoobzadeh, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2412.13375)  

**Abstract**: Large language models (LLMs) have made great progress in classification and text generation tasks. However, they are mainly trained on English data and often struggle with low-resource languages. In this study, we explore adding a new language, i.e., Persian, to Llama (a model with a limited understanding of Persian) using parameter-efficient fine-tuning. We employ a multi-stage approach involving pretraining on monolingual Persian data, aligning representations through bilingual pretraining and instruction datasets, and instruction-tuning with task-specific datasets. We evaluate the model's performance at each stage on generation and classification tasks. Our findings suggest that incorporating the Persian language, through bilingual data alignment, can enhance classification accuracy for Persian tasks, with no adverse impact and sometimes even improvements on English tasks. Additionally, the results highlight the model's initial strength as a critical factor when working with limited training data, with cross-lingual alignment offering minimal benefits for the low-resource language. Knowledge transfer from English to Persian has a marginal effect, primarily benefiting simple classification tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在分类和文本生成任务中取得了显著的进步。然而，它们主要是在英文数据上进行训练，往往在低资源语言方面表现不佳。在本研究中，我们探索将一种新语言——波斯语——添加到Llama模型中（该模型对波斯语的理解有限）的方法，通过参数高效的微调实现。我们采用多阶段的方法，包括使用单语波斯语数据预训练、通过双语预训练和指令数据集进行表示对齐，以及使用特定任务的数据集进行指令微调。我们在生成和分类任务中分别对模型在各个阶段的表现进行了评估。研究发现，通过双语数据对齐纳入波斯语可以提高波斯语任务的分类准确率，有时甚至对英文任务也无负面影响，甚至有时还有所改善。此外，结果还强调了初始模型在有限训练数据下的强大能力，跨语言对齐对低资源语言的帮助有限。从英文到波斯语的知识迁移效应微乎其微，主要对简单的分类任务有益。 

---
# Hint Marginalization for Improved Reasoning in Large Language Models 

**Title (ZH)**: 改进大型语言模型推理的提示边缘化方法 

**Authors**: Soumyasundar Pal, Didier Chételat, Yingxue Zhang, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2412.13292)  

**Abstract**: Large Language Models (LLMs) have exhibited an impressive capability to perform reasoning tasks, especially if they are encouraged to generate a sequence of intermediate steps. Reasoning performance can be improved by suitably combining multiple LLM responses, generated either in parallel in a single query, or via sequential interactions with LLMs throughout the reasoning process. Existing strategies for combination, such as self-consistency and progressive-hint-prompting, make inefficient usage of the LLM responses. We present Hint Marginalization, a novel and principled algorithmic framework to enhance the reasoning capabilities of LLMs. Our approach can be viewed as an iterative sampling strategy for forming a Monte Carlo approximation of an underlying distribution of answers, with the goal of identifying the mode the most likely answer. Empirical evaluation on several benchmark datasets for arithmetic reasoning demonstrates the superiority of the proposed approach. 

**Abstract (ZH)**: 大型语言模型（LLMs）在执行推理任务方面展现了令人印象深刻的性能，尤其是在被鼓励生成一系列中间步骤时。通过适当地结合多个LLM响应——这些响应可以在单次查询中并行生成，或在推理过程中通过与LLM的顺序交互生成——可以提高推理性能。现有的组合策略，如自我一致性和平行提示提示，未能充分利用LLM响应。我们提出了提示边际化（Hint Marginalization），这是一种新颖且具备原则性的算法框架，旨在增强LLM的推理能力。我们的方法可以被视为一种迭代抽样策略，用于形成潜在答案分布的蒙特卡洛近似，并旨在识别最有可能的答案。在多个用于算术推理的基准数据集上的实证评估表明，所提出的方法具有明显的优势。 

---
# Information-Theoretic Generative Clustering of Documents 

**Title (ZH)**: 信息论生成聚类文档 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2412.13534)  

**Abstract**: We present {\em generative clustering} (GC) for clustering a set of documents, $\mathrm{X}$, by using texts $\mathrm{Y}$ generated by large language models (LLMs) instead of by clustering the original documents $\mathrm{X}$. Because LLMs provide probability distributions, the similarity between two documents can be rigorously defined in an information-theoretic manner by the KL divergence. We also propose a natural, novel clustering algorithm by using importance sampling. We show that GC achieves the state-of-the-art performance, outperforming any previous clustering method often by a large margin. Furthermore, we show an application to generative document retrieval in which documents are indexed via hierarchical clustering and our method improves the retrieval accuracy. 

**Abstract (ZH)**: 我们提出了一种生成聚类（Generative Clustering, GC）方法，通过使用大规模语言模型（Large Language Models, LLMs）生成的文本 $\mathrm{Y}$ 来对文档集 $\mathrm{X}$ 进行聚类，而不是直接对原始文档 $\mathrm{X}$ 进行聚类。由于LLMs提供了概率分布，可以通过KL散度在信息论意义上严格定义两份文档之间的相似性。我们还提出了一种基于重要性采样的新颖聚类算法。实验结果表明，GC方法在聚类性能上达到了最佳水平，通常大幅优于以往的任何聚类方法。此外，我们还展示了生成性文档检索的应用场景，在这种应用中，文档通过层次聚类进行索引，我们的方法提高了检索准确性。 

---
# JudgeBlender: Ensembling Judgments for Automatic Relevance Assessment 

**Title (ZH)**: JudgeBlender: 组合判决以实现自动相关性评估 

**Authors**: Hossein A. Rahmani, Emine Yilmaz, Nick Craswell, Bhaskar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2412.13268)  

**Abstract**: The effective training and evaluation of retrieval systems require a substantial amount of relevance judgments, which are traditionally collected from human assessors -- a process that is both costly and time-consuming. Large Language Models (LLMs) have shown promise in generating relevance labels for search tasks, offering a potential alternative to manual assessments. Current approaches often rely on a single LLM, such as GPT-4, which, despite being effective, are expensive and prone to intra-model biases that can favour systems leveraging similar models. In this work, we introduce JudgeBlender, a framework that employs smaller, open-source models to provide relevance judgments by combining evaluations across multiple LLMs (LLMBlender) or multiple prompts (PromptBlender). By leveraging the LLMJudge benchmark [18], we compare JudgeBlender with state-of-the-art methods and the top performers in the LLMJudge challenge. Our results show that JudgeBlender achieves competitive performance, demonstrating that very large models are often unnecessary for reliable relevance assessments. 

**Abstract (ZH)**: 有效的检索系统训练和评估需要大量的相关性判断，这些判断传统上是由人工评估者收集的——这是一个既昂贵又耗时的过程。大型语言模型（LLMs）在为搜索任务生成相关性标签方面显示出潜力，从而为替代人工评估提供了可能。当前的方法通常依赖单一的LLM，如GPT-4，尽管其有效，但成本较高且容易受到模型内部偏差的影响，这些偏差可能导致利用类似模型的系统获得优势。在本文中，我们引入了JudgeBlender框架，该框架通过结合多个LLM（LLMBlender）或多个提示（PromptBlender）的评估来使用较小的开源模型提供相关性判断。通过利用LLMJudge基准数据集[18]，我们将JudgeBlender与最先进的方法和LLMJudge挑战中的顶尖表现者进行了比较。我们的结果表明，JudgeBlender取得了竞争力的表现，这表明可靠的相关性评估通常并不需要非常大的模型。 

---
# Boosting LLM-based Relevance Modeling with Distribution-Aware Robust Learning 

**Title (ZH)**: 基于分布感知的稳健学习提升基于LLM的相关性建模 

**Authors**: Hong Liu, Saisai Gong, Yixin Ji, Kaixin Wu, Jia Xu, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12504)  

**Abstract**: With the rapid advancement of pre-trained large language models (LLMs), recent endeavors have leveraged the capabilities of LLMs in relevance modeling, resulting in enhanced performance. This is usually done through the process of fine-tuning LLMs on specifically annotated datasets to determine the relevance between queries and items. However, there are two limitations when LLMs are naively employed for relevance modeling through fine-tuning and inference. First, it is not inherently efficient for performing nuanced tasks beyond simple yes or no answers, such as assessing search relevance. It may therefore tend to be overconfident and struggle to distinguish fine-grained degrees of relevance (e.g., strong relevance, weak relevance, irrelevance) used in search engines. Second, it exhibits significant performance degradation when confronted with data distribution shift in real-world scenarios. In this paper, we propose a novel Distribution-Aware Robust Learning framework (DaRL) for relevance modeling in Alipay Search. Specifically, we design an effective loss function to enhance the discriminability of LLM-based relevance modeling across various fine-grained degrees of query-item relevance. To improve the generalizability of LLM-based relevance modeling, we first propose the Distribution-Aware Sample Augmentation (DASA) module. This module utilizes out-of-distribution (OOD) detection techniques to actively select appropriate samples that are not well covered by the original training set for model fine-tuning. Furthermore, we adopt a multi-stage fine-tuning strategy to simultaneously improve in-distribution (ID) and OOD performance, bridging the performance gap between them. DaRL has been deployed online to serve the Alipay's insurance product search... 

**Abstract (ZH)**: 随着预训练大型语言模型（LLMs）的迅速发展，近期的研究已经在利用LLMs在相关性建模方面的的能力，从而提升了模型性能。这通常是通过在特注注释的数据集上对LLMs进行微调来实现的，以确定查询与项目之间的相关性。然而，当LLMs简单地通过微调和推理来进行相关性建模时，存在两个局限性。首先，对于超越简单“是”或“否”的细致任务（例如，评估搜索相关性）来说，它未必是高效的。因此，模型可能会变得过于自信，并在区分细粒度的相关性程度（例如，高度相关、弱相关、不相关）方面遇到困难。其次，当面对现实世界中的数据分布偏移时，其性能会显著下降。在本文中，我们提出了一个新颖的面向分布的鲁棒学习框架（DaRL）来解决支付宝搜索中相关性建模的问题。具体而言，我们设计了一个有效的损失函数，以增强基于LLMs的相关性建模在不同细粒度查询-项目相关性方面的辨别能力。为了提高基于LLMs的相关性建模的泛化能力，我们首先提出了面向分布的样本增强（DASA）模块。该模块利用了离群值检测技术来主动选择那些原始训练集未能充分覆盖的适当样本，用于模型微调。进一步地，我们采用了多阶段微调策略以同时提升数据分布内（ID）和离群值（OOD）环境下的性能，从而弥合它们之间的性能差距。DaRL已被部署上线，服务于支付宝的保险产品搜索... 

---
# LLM is Knowledge Graph Reasoner: LLM's Intuition-aware Knowledge Graph Reasoning for Cold-start Sequential Recommendation 

**Title (ZH)**: 大语言模型是知识图谱推理器：大语言模型的基于直觉的知识图谱推理在冷启动序列推荐中的应用 

**Authors**: Keigo Sakurai, Ren Togo, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2412.12464)  

**Abstract**: Knowledge Graphs (KGs) represent relationships between entities in a graph structure and have been widely studied as promising tools for realizing recommendations that consider the accurate content information of items. However, traditional KG-based recommendation methods face fundamental challenges: insufficient consideration of temporal information and poor performance in cold-start scenarios. On the other hand, Large Language Models (LLMs) can be considered databases with a wealth of knowledge learned from the web data, and they have recently gained attention due to their potential application as recommendation systems. Although approaches that treat LLMs as recommendation systems can leverage LLMs' high recommendation literacy, their input token limitations make it impractical to consider the entire recommendation domain dataset and result in scalability issues. To address these challenges, we propose a LLM's Intuition-aware Knowledge graph Reasoning model (LIKR). Our main idea is to treat LLMs as reasoners that output intuitive exploration strategies for KGs. To integrate the knowledge of LLMs and KGs, we trained a recommendation agent through reinforcement learning using a reward function that integrates different recommendation strategies, including LLM's intuition and KG embeddings. By incorporating temporal awareness through prompt engineering and generating textual representations of user preferences from limited interactions, LIKR can improve recommendation performance in cold-start scenarios. Furthermore, LIKR can avoid scalability issues by using KGs to represent recommendation domain datasets and limiting the LLM's output to KG exploration strategies. Experiments on real-world datasets demonstrate that our model outperforms state-of-the-art recommendation methods in cold-start sequential recommendation scenarios. 

**Abstract (ZH)**: 知识图谱（KGs）将实体之间的关系表示为图结构，并且被广泛研究作为一种能够考虑项目准确内容信息的推荐工具。然而，传统的基于KG的推荐方法面临根本性的挑战：缺乏对时间信息的充分考虑以及在冷启动场景中的表现不佳。另一方面，大型语言模型（LLMs）可以被视为从网络数据中学习了大量的知识的数据库，并且由于其在推荐系统中潜在的应用而近期引起了人们的关注。尽管将LLMs视为推荐系统的做法可以利用LLMs的高推荐素养，但由于输入令牌的限制，使得完全考虑推荐领域数据集变得不切实际，从而导致可扩展性问题。为了解决这些挑战，我们提出了一种LLM启发的知识图谱推理模型（LIKR）。我们的主要想法是将LLMs视为能够输出KGs直观探索策略的推理器。为了整合LLMs和KGs的知识，我们通过强化学习训练了一个推荐代理，使用的奖励函数综合了不同的推荐策略，包括LLMs的直觉和KG嵌入。通过工程化提示引入时间感知性，并从有限的交互中生成用户偏好的文本表示，LIKR能够在冷启动场景中提高推荐性能。此外，通过使用KGs来表示推荐领域数据集并且限制LLMs的输出为KG探索策略，LIKR解决了可扩展性问题。实验表明，在实际数据集上，我们的模型在冷启动序列推荐场景中优于最先进的推荐方法。 

---
