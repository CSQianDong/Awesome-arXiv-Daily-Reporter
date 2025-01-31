# Investigating Tax Evasion Emergence Using Dual Large Language Model and Deep Reinforcement Learning Powered Agent-based Simulation 

**Title (ZH)**: 使用双大型语言模型和深度强化学习驱动的基于代理的仿真研究逃税行为的 emergence 

**Authors**: Teddy Lazebnik, Labib Shami  

**Link**: [PDF](https://arxiv.org/pdf/2501.18177)  

**Abstract**: Tax evasion, usually the largest component of an informal economy, is a persistent challenge over history with significant socio-economic implications. Many socio-economic studies investigate its dynamics, including influencing factors, the role and influence of taxation policies, and the prediction of the tax evasion volume over time. These studies assumed such behavior is given, as observed in the real world, neglecting the "big bang" of such activity in a population. To this end, computational economy studies adopted developments in computer simulations, in general, and recent innovations in artificial intelligence (AI), in particular, to simulate and study informal economy appearance in various socio-economic settings. This study presents a novel computational framework to examine the dynamics of tax evasion and the emergence of informal economic activity. Employing an agent-based simulation powered by Large Language Models and Deep Reinforcement Learning, the framework is uniquely designed to allow informal economic behaviors to emerge organically, without presupposing their existence or explicitly signaling agents about the possibility of evasion. This provides a rigorous approach for exploring the socio-economic determinants of compliance behavior. The experimental design, comprising model validation and exploratory phases, demonstrates the framework's robustness in replicating theoretical economic behaviors. Findings indicate that individual personality traits, external narratives, enforcement probabilities, and the perceived efficiency of public goods provision significantly influence both the timing and extent of informal economic activity. The results underscore that efficient public goods provision and robust enforcement mechanisms are complementary; neither alone is sufficient to curtail informal activity effectively. 

**Abstract (ZH)**: 税收 evasion 通常是非正式经济中最大的组成部分，一直是历史上一个持久的挑战，具有显著的经济社会影响。许多经济社会研究探讨了税收 evasion 的动态，包括影响因素、税收政策的作用和影响，以及税收 evasion 规模的预测。这些研究假设这种行为是给定的，基于现实世界中的观察，忽略了这种活动在人群中的“突变”。为此，计算经济学研究采用了计算机模拟的一般发展，尤其是最近人工智能（AI）的创新成果，来模拟和研究在不同经济社会环境中的非正式经济活动出现。

本研究提出了一种新的计算框架，以探讨税收 evasion 的动态及其非正式经济活动的产生。通过基于大型语言模型和深度强化学习的代理基础仿真，该框架独特地设计了无需假设其存在或明确指示代理存在逃税可能性的方法，从而使其有机地产生非正式经济行为。这种方法为探索合规行为的经济社会决定因素提供了严谨的方法。实验设计包括模型验证和探索阶段，展示了该框架在重现理论经济行为方面的稳健性。研究结果表明，个体人格特质、外部叙述、实施可能性以及公共物品提供效率显著影响非正式经济活动的时间和规模。研究结果强调了高效的公共物品提供和严格的执法机制之间的互补性；两者单独均不足以有效遏制非正式活动。 

---
# Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method 

**Title (ZH)**: 当然，以下是翻译后的标题和内容，符合学术规范：

标题：
一次检索所有内容可行吗？ARM：一种基于语言模型的对齐导向检索方法

内容摘要：
本研究探讨了一次性检索所有相关信息的可能性。我们提出了一种名为ARM的方法，该方法基于语言模型，并采用对齐导向的策略。 

**Authors**: Peter Baile Chen, Yi Zhang, Michael Cafarella, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2501.18539)  

**Abstract**: Real-world open-domain questions can be complicated, particularly when answering them involves information from multiple information sources. LLMs have demonstrated impressive performance in decomposing complex tasks into simpler steps, and previous work has used it for better retrieval in support of complex questions. However, LLM's decomposition of questions is unaware of what data is available and how data is organized, often leading to a sub-optimal retrieval performance. Recent effort in agentic RAG proposes to perform retrieval in an iterative fashion, where a followup query is derived as an action based on previous rounds of retrieval. While this provides one way of interacting with the data collection, agentic RAG's exploration of data is inefficient because successive queries depend on previous results rather than being guided by the organization of available data in the collection. To address this problem, we propose an LLM-based retrieval method -- ARM, that aims to better align the question with the organization of the data collection by exploring relationships among data objects beyond matching the utterance of the query, thus leading to a retrieve-all-at-once solution for complex queries. We evaluated ARM on two datasets, Bird and OTT-QA. On Bird, it outperforms standard RAG with query decomposition by up to 5.2 pt in execution accuracy and agentic RAG (ReAct) by up to 15.9 pt. On OTT-QA, it achieves up to 5.5 pt and 19.3 pt higher F1 match scores compared to these approaches. 

**Abstract (ZH)**: 现实世界的开放式问题可能非常复杂，尤其是当回答这些问题需要从多个信息源获取信息时。大规模语言模型（LLMs）在分解复杂任务为更简单步骤方面展现了出色的性能，先前的工作已经利用这一点来提高复杂问题的支持检索效果。然而，LLMs对问题的分解并不了解哪些数据可用以及数据是如何组织的，这常常导致检索性能不佳。近期关于代理型检索-生成（RAG）的努力提出了一种迭代检索的方法，其中后续查询是基于前几轮检索的结果而采取的一种行动。虽然这种方法提供了一种与数据集交互的方式，但代理型RAG对数据的探索不够高效，因为后续查询依赖于前几轮的结果而非数据集内可用数据的组织结构。为了解决这一问题，我们提出了一种基于LLM的检索方法——ARM，其目的是通过探索数据对象之间的关系，而不仅仅是匹配查询的表述，更好地将问题与数据集的组织结构对齐，从而为复杂问题提供一个一次性检索所有信息的解决方案。

我们使用两个数据集BIRD和OTT-QA对ARM进行了评估。在BIRD数据集上，ARM在执行准确性上比具有查询分解标准RAG方法提高了最多5.2个百分点，比代理型RAG（ReAct）提高了最多15.9个百分点。在OTT-QA数据集上，ARM的F1匹配得分比这些方法分别提高了最多5.5个百分点和19.3个百分点。 

---
# Conversation Games and a Strategic View of the Turing Test 

**Title (ZH)**: 对话游戏与图灵测试的战略视角 

**Authors**: Kaveh Aryan  

**Link**: [PDF](https://arxiv.org/pdf/2501.18455)  

**Abstract**: Although many game-theoretic models replicate real interactions that often rely on natural language, explicit study of games where language is central to strategic interaction remains limited. This paper introduces the \emph{conversation game}, a multi-stage, extensive-form game based on linguistic strategic interaction. We focus on a subset of the games, called verdict games. In a verdict game, two players alternate to contribute to a conversation, which is evaluated at each stage by a non-strategic judge who may render a conclusive binary verdict, or a decision to continue the dialogue. The game ends once a limit is reached or a verdict is given. We show many familiar processes, such as interrogation or a court process fall under this category. We also, show that the Turing test is an instance of verdict game, and discuss the significance of a strategic view of the Turing test in the age of advanced AI deception. We show the practical relevance of the proposed concepts by simulation experiments, and show that a strategic agent outperforms a naive agent by a high margin. 

**Abstract (ZH)**: 尽管许多博弈论模型能够模拟常常依赖自然语言的真实互动，对于语言在战略互动中起核心作用的游戏的明确研究仍然相对有限。本文介绍了\emph{对话博弈}，这是一种基于语言战略互动的多阶段、扩展型博弈。我们重点关注这类博弈中的一个子集，称为裁决博弈。在裁决博弈中，两名玩家交替贡献于对话，每一步由一位非战略性的法官进行评估，这位法官可能会作出最终的二元裁决或决定继续对话。博弈在达到上限或给出裁决后结束。我们表明许多熟悉的流程，如审问或法庭程序都属于此类。我们还证明图灵测试是此类博弈的一个实例，并讨论在先进人工智能欺诈时代重新审视图灵测试的战略视角的重要性。我们通过模拟实验展示了所提出概念的实用相关性，结果表明战略智能体的性能远远优于非明智智能体。 

---
# Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents 

**Title (ZH)**: Gravity-Bench-v1：智能体在引力物理发现方面的基准测试 

**Authors**: Nolan Koblischke, Hyunseok Jang, Kristen Menou, Mohamad Ali-Dib  

**Link**: [PDF](https://arxiv.org/pdf/2501.18411)  

**Abstract**: Modern science emerged from reasoning over repeatedly-observed planetary motions. We present Gravity-Bench-v1, an environment-based benchmark that challenges AI agents on tasks that parallel this historical development. Gravity-Bench-v1 evaluates agents on the discovery of physics concealed within a dynamic environment, using rigorous gravitational dynamics simulations. Gravity-Bench includes out-of-distribution cases, i.e. with physics that deviates from the real world, to evaluate true scientific generalization capabilities. Agents must plan to collect data within an experimental budget and must perform a dynamic form of data analysis and reasoning to solve tasks efficiently. Our benchmark admits an open-ended space of solutions. PhD-level solutions for each task are provided, to calibrate AI performance against human expertise. Technically at an upper-undergraduate level, our benchmark proves challenging to baseline AI agents. Gravity-Bench-v1 and planned extensions should help map out AI progress towards scientific discovery capabilities. 

**Abstract (ZH)**: 现代科学研究起源于对反复观测到的行星运动的推理。本文介绍了基于环境的基准测试Gravity-Bench-v1，该基准测试挑战AI代理完成类似于这一历史发展的任务。Gravity-Bench-v1 评估代理在动态环境中发现隐藏的物理现象的能力，通过严格的引力动力学模拟进行评估。Gravity-Bench 包括未见过分布（out-of-distribution）情况，即包含与现实世界不符的物理现象，以评估真正的科学泛化能力。代理必须在实验预算内计划收集数据，并通过动态的数据分析和推理高效地解决问题。我们的基准测试允许广泛的方法空间。每项任务都提供了博士水平的解决方案，以便根据人类专业知识校准AI性能。技术上而言，我们的基准测试对基础模型构成较大挑战。Gravity-Bench-v1 及其未来扩展应有助于绘制AI在科学发现能力方面的进步。 

---
# Leveraging LLM Agents for Automated Optimization Modeling for SASP Problems: A Graph-RAG based Approach 

**Title (ZH)**: 利用大语言模型代理进行SASP问题的自动化优化建模：基于Graph-RAG的方法 

**Authors**: Tianpeng Pan, Wenqiang Pu, Licheng Zhao, Rui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.18320)  

**Abstract**: Automated optimization modeling (AOM) has evoked considerable interest with the rapid evolution of large language models (LLMs). Existing approaches predominantly rely on prompt engineering, utilizing meticulously designed expert response chains or structured guidance. However, prompt-based techniques have failed to perform well in the sensor array signal processing (SASP) area due the lack of specific domain knowledge. To address this issue, we propose an automated modeling approach based on retrieval-augmented generation (RAG) technique, which consists of two principal components: a multi-agent (MA) structure and a graph-based RAG (Graph-RAG) process. The MA structure is tailored for the architectural AOM process, with each agent being designed based on principles of human modeling procedure. The Graph-RAG process serves to match user query with specific SASP modeling knowledge, thereby enhancing the modeling result. Results on ten classical signal processing problems demonstrate that the proposed approach (termed as MAG-RAG) outperforms several AOM benchmarks. 

**Abstract (ZH)**: 自动化优化建模（AOM）随着大型语言模型（LLMs）的迅速发展引起了广泛的关注。现有的方法主要依赖于提示工程，利用精心设计的专家响应链或结构化的指导。然而，在传感器阵列信号处理（SASP）领域，提示基础的方法由于缺乏特定领域的知识而表现不佳。为了解决这一问题，我们提出了一种基于检索增强生成（RAG）技术的自动化建模方法，该方法主要包括两个主要组成部分：一个多代理（MA）结构和基于图的RAG（Graph-RAG）过程。多代理结构专门针对架构化的AOM流程，其中每个代理都是基于人类建模过程的原则设计的。基于图的RAG过程用于匹配用户查询与特定的SASP建模知识，从而提高建模结果。在十个经典信号处理问题上的实验结果显示，所提出的方法（称为MAG-RAG）在几个AOM基准中表现更优。 

---
# Model-Free RL Agents Demonstrate System 1-Like Intentionality 

**Title (ZH)**: 无模型 RL 剂量表现出类似系统 1 的意图性 

**Authors**: Hal Ashton, Matija Franklin  

**Link**: [PDF](https://arxiv.org/pdf/2501.18299)  

**Abstract**: This paper argues that model-free reinforcement learning (RL) agents, while lacking explicit planning mechanisms, exhibit behaviours that can be analogised to System 1 ("thinking fast") processes in human cognition. Unlike model-based RL agents, which operate akin to System 2 ("thinking slow") reasoning by leveraging internal representations for planning, model-free agents react to environmental stimuli without anticipatory modelling. We propose a novel framework linking the dichotomy of System 1 and System 2 to the distinction between model-free and model-based RL. This framing challenges the prevailing assumption that intentionality and purposeful behaviour require planning, suggesting instead that intentionality can manifest in the structured, reactive behaviours of model-free agents. By drawing on interdisciplinary insights from cognitive psychology, legal theory, and experimental jurisprudence, we explore the implications of this perspective for attributing responsibility and ensuring AI safety. These insights advocate for a broader, contextually informed interpretation of intentionality in RL systems, with implications for their ethical deployment and regulation. 

**Abstract (ZH)**: 本文 argues，尽管模型自由强化学习（RL）代理缺乏显式规划机制，但它们表现出的行为可以类比于人类认知中的系统1（“快速思考”）过程。与依赖内部表示进行规划、操作类似于系统2（“慢速思考”）推理的模型依赖RL代理不同，模型自由代理对环境刺激作出反应，而不进行预先建模。我们提出了一种新的框架，将系统1与系统2的区别与模型自由和模型依赖RL的区别联系起来。这种框架挑战了先前假设，即意图性和有目的行为需要规划，表明意图性可以在模型自由代理的结构化、反应性行为中体现。通过借鉴认知心理学、法律理论和实验法律学的跨学科见解，我们探讨了这一视角对于归因责任和确保AI安全的影响。这些见解提倡在RL系统中对意图性采用更为广泛、情境化的解释，这有助于其伦理部署和监管。 

---
# Investigating an Intelligent System to Monitor \& Explain Abnormal Activity Patterns of Older Adults 

**Title (ZH)**: 研究一种智能系统以监控和解释老年人异常活动模式 

**Authors**: Min Hun Lee, Daniel P. Siewiorek, Alexandre Bernardino  

**Link**: [PDF](https://arxiv.org/pdf/2501.18108)  

**Abstract**: Despite the growing potential of older adult care technologies, the adoption of these technologies remains challenging. In this work, we conducted a focus-group session with family caregivers to scope designs of the older adult care technology. We then developed a high-fidelity prototype and conducted its qualitative study with professional caregivers and older adults to understand their perspectives on the system functionalities. This system monitors abnormal activity patterns of older adults using wireless motion sensors and machine learning models and supports interactive dialogue responses to explain abnormal activity patterns of older adults to caregivers and allow older adults proactively sharing their status with caregivers for an adequate intervention. Both older adults and professional caregivers appreciated that our system can provide a faster, personalized service while proactively controlling what information is to be shared through interactive dialogue responses. We further discuss other considerations to realize older adult technology in practice. 

**Abstract (ZH)**: 尽管老年照护技术的发展潜力日益增大，但这些技术的应用仍面临诸多挑战。本研究通过与家庭护理人员进行焦点小组讨论，初步探讨了老年照护技术的设计。随后，我们开发了一个高保真原型，并通过向专业护理人员和老年人进行定性研究，以了解他们对该系统的功能看法。该系统利用无线运动传感器和机器学习模型监测老年人的异常活动模式，并支持交互式对话响应，以解释老年人的异常活动模式给护理人员，并允许老年人主动向护理人员分享其状态，以便进行适当的干预。老年受试者和专业护理人员均认为，该系统可以提供更快、更个性化的服务，并通过交互式对话响应主动控制要分享的信息。此外，我们还讨论了在实践中实现老年人照护技术时需要注意的其他事项。 

---
# Belief Roadmaps with Uncertain Landmark Evanescence 

**Title (ZH)**: 含不确定地标消失信念 roadmap 

**Authors**: Erick Fuentes, Jared Strader, Ethan Fahnestock, Nicholas Roy  

**Link**: [PDF](https://arxiv.org/pdf/2501.17982)  

**Abstract**: We would like a robot to navigate to a goal location while minimizing state uncertainty. To aid the robot in this endeavor, maps provide a prior belief over the location of objects and regions of interest. To localize itself within the map, a robot identifies mapped landmarks using its sensors. However, as the time between map creation and robot deployment increases, portions of the map can become stale, and landmarks, once believed to be permanent, may disappear. We refer to the propensity of a landmark to disappear as landmark evanescence. Reasoning about landmark evanescence during path planning, and the associated impact on localization accuracy, requires analyzing the presence or absence of each landmark, leading to an exponential number of possible outcomes of a given motion plan. To address this complexity, we develop BRULE, an extension of the Belief Roadmap. During planning, we replace the belief over future robot poses with a Gaussian mixture which is able to capture the effects of landmark evanescence. Furthermore, we show that belief updates can be made efficient, and that maintaining a random subset of mixture components is sufficient to find high quality solutions. We demonstrate performance in simulated and real-world experiments. Software is available at this https URL. 

**Abstract (ZH)**: 我们希望让机器人在导航到目标位置的同时，尽量减少状态不确定性。为了帮助机器人实现这一目标，地图提供了物体和感兴趣区域的先验信念。为了在地图中定位自身，机器人使用其传感器识别已映射的地标。然而，随着地图创建与机器人部署之间的时间增加，地图的某些部分可能会变得过时，原本认为是永久存在的地标也可能会消失。我们称这种地标消失的倾向为地标蒸发性。在路径规划过程中考虑地标蒸发性，并对其对定位准确性的影响进行推理，需要分析每个地标的存在或不存在情况，从而导致一组给定运动计划可能的结果呈指数增长。为了解决这种复杂性，我们开发了BRULE，它是信念路网的扩展。在规划过程中，我们用高斯混合模型来替换对未来机器人姿态的信念，从而能够捕捉地标蒸发性的影响。此外，我们展示了信念更新可以实现高效化，并且维持混合模型中的随机子集足以找到高质量的解决方案。我们通过模拟和真实世界的实验展示了性能。软件可以在以下链接获取：[提供链接处] 

---
# Free Agent in Agent-Based Mixture-of-Experts Generative AI Framework 

**Title (ZH)**: 基于代理混合专家生成人工智能框架中的自由代理 

**Authors**: Jung-Hua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17903)  

**Abstract**: Multi-agent systems commonly distribute tasks among specialized, autonomous agents, yet they often lack mechanisms to replace or reassign underperforming agents in real time. Inspired by the free-agency model of Major League Baseball, the Reinforcement Learning Free Agent (RLFA) algorithm introduces a reward-based mechanism to detect and remove agents exhibiting persistent underperformance and seamlessly insert more capable ones. Each agent internally uses a mixture-of-experts (MoE) approach, delegating incoming tasks to specialized sub-models under the guidance of a gating function. A primary use case is fraud detection, where RLFA promptly swaps out an agent whose detection accuracy dips below a preset threshold. A new agent is tested in a probationary mode, and upon demonstrating superior performance, fully replaces the underperformer. This dynamic, free-agency cycle ensures sustained accuracy, quicker adaptation to emerging threats, and minimal disruption to ongoing operations. By continually refreshing its roster of agents, the system fosters ongoing improvements and more resilient collaboration in multi-agent Generative AI environments. 

**Abstract (ZH)**: 多智能体系统通常将任务分配给专门的、自主的代理，但它们往往缺乏能够实时检测和更换表现不佳代理的机制。受到美国职业棒球大联盟自由球员机制的启发，Reinforcement Learning Free Agent（RLFA）算法引入了一种基于奖励的机制，用于检测持续表现不佳的代理并实时更换，同时无缝地插入更胜任的代理。每个代理内部使用混合专家（MoE）方法，通过门控函数的指导将到来的任务委派给专门的子模型。一个主要的应用案例是欺诈检测，在这种情况下，RLFA可以迅速替换检测精度下降到预设阈值以下的代理。新的代理将处于试用模式，若表现出色，则完全替代表现较差的代理。这种动态的自由代理循环确保了持续的准确性、更快地适应新兴威胁并且对持续操作的影响最小。通过不断更新其代理列表，系统促进了多代理生成AI环境中的持续改进和更稳健的合作。 

---
