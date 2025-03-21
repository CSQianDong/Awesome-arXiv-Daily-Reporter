# If Multi-Agent Debate is the Answer, What is the Question? 

**Title (ZH)**: 如果多智能体辩论是答案，那么问题是什么？ 

**Authors**: Hangfan Zhang, Zhiyao Cui, Xinrun Wang, Qiaosheng Zhang, Zhen Wang, Dinghao Wu, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08788)  

**Abstract**: Multi-agent debate (MAD) has emerged as a promising approach to enhance the factual accuracy and reasoning quality of large language models (LLMs) by engaging multiple agents in iterative discussions during inference. Despite its potential, we argue that current MAD research suffers from critical shortcomings in evaluation practices, including limited dataset overlap and inconsistent baselines, raising significant concerns about generalizability. Correspondingly, this paper presents a systematic evaluation of five representative MAD methods across nine benchmarks using four foundational models. Surprisingly, our findings reveal that MAD methods fail to reliably outperform simple single-agent baselines such as Chain-of-Thought and Self-Consistency, even when consuming additional inference-time computation. From our analysis, we found that model heterogeneity can significantly improve MAD frameworks. We propose Heter-MAD enabling a single LLM agent to access the output from heterogeneous foundation models, which boosts the performance of current MAD frameworks. Finally, we outline potential directions for advancing MAD, aiming to spark a broader conversation and inspire future work in this area. 

**Abstract (ZH)**: 多智能体辩论（MAD）已成为通过在推理过程中让多个智能体进行迭代讨论来增强大型语言模型（LLMs）的事实准确性及推理质量的一种有前景的方法。尽管具有潜力，但我们认为当前的MAD研究在评估实践方面存在严重的不足，包括数据集重叠有限以及基线不一致，这严重地引发了对其普遍适用性的担忧。相应地，本文对五种代表性的MAD方法在九个基准上进行了系统性评估，使用了四种基础模型。令人惊讶的是，我们的发现表明，即使消耗了更多的推理时间计算，MAD方法也无法可靠地超越简单的单智能体基线，如Chain-of-Thought和Self-Consistency。通过对这些结果的分析，我们发现模型异质性可以显著改善MAD框架。我们提出了Heter-MAD，允许单一LLM智能体访问来自不同基础模型的输出，从而提升当前MAD框架的性能。最后，我们概述了MAD未来发展的一些潜在方向，旨在激发更广泛的讨论并激发未来在此领域的研究工作。 

---
# EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents 

**Title (ZH)**: EmbodiedBench：综合评估面向视觉驱动体感代理的多模态大型语言模型 

**Authors**: Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09560)  

**Abstract**: Leveraging Multi-modal Large Language Models (MLLMs) to create embodied agents offers a promising avenue for tackling real-world tasks. While language-centric embodied agents have garnered substantial attention, MLLM-based embodied agents remain underexplored due to the lack of comprehensive evaluation frameworks. To bridge this gap, we introduce EmbodiedBench, an extensive benchmark designed to evaluate vision-driven embodied agents. EmbodiedBench features: (1) a diverse set of 1,128 testing tasks across four environments, ranging from high-level semantic tasks (e.g., household) to low-level tasks involving atomic actions (e.g., navigation and manipulation); and (2) six meticulously curated subsets evaluating essential agent capabilities like commonsense reasoning, complex instruction understanding, spatial awareness, visual perception, and long-term planning. Through extensive experiments, we evaluated 13 leading proprietary and open-source MLLMs within EmbodiedBench. Our findings reveal that: MLLMs excel at high-level tasks but struggle with low-level manipulation, with the best model, GPT-4o, scoring only 28.9% on average. EmbodiedBench provides a multifaceted standardized evaluation platform that not only highlights existing challenges but also offers valuable insights to advance MLLM-based embodied agents. Our code is available at this https URL. 

**Abstract (ZH)**: 利用多模态大型语言模型（MLLMs）创建具身智能体为解决现实世界任务提供了有前途的途径。虽然以语言为中心的具身智能体已受到广泛关注，但由于缺乏全面的评估框架，基于MLLM的具身智能体仍然未得到充分探索。为了弥合这一差距，我们提出了EmbodiedBench——一个广泛的基准测试，旨在评估以视觉驱动的具身智能体。EmbodiedBench的特点包括：(1) 涵盖四个环境的1,128个测试任务，从高层语义任务（如家庭环境）到涉及原子动作的低层任务（如导航和操作）；(2) 评估六个精心策划的子集，这些子集涵盖了常识推理、复杂指令理解、空间意识、视觉感知和长期规划等核心智能体能力。通过大量的实验，我们在EmbodiedBench上评估了13个领先的专有和开源MLLM。我们的研究结果表明：MLLMs在高层任务上表现优异，但在低层操作方面存在困难，最优模型GPT-4o的平均得分仅为28.9%。EmbodiedBench提供了一个多方面的标准评估平台，不仅突出了现有的挑战，还为推动基于MLLM的具身智能体提供了宝贵见解。我们的代码可在以下链接获取：[此 https URL]。 

---
# Language Agents as Digital Representatives in Collective Decision-Making 

**Title (ZH)**: 语言代理作为集体决策中的数字代表 

**Authors**: Daniel Jarrett, Miruna Pîslar, Michiel A. Bakker, Michael Henry Tessler, Raphael Köster, Jan Balaguer, Romuald Elie, Christopher Summerfield, Andrea Tacchetti  

**Link**: [PDF](https://arxiv.org/pdf/2502.09369)  

**Abstract**: Consider the process of collective decision-making, in which a group of individuals interactively select a preferred outcome from among a universe of alternatives. In this context, "representation" is the activity of making an individual's preferences present in the process via participation by a proxy agent -- i.e. their "representative". To this end, learned models of human behavior have the potential to fill this role, with practical implications for multi-agent scenario studies and mechanism design. In this work, we investigate the possibility of training \textit{language agents} to behave in the capacity of representatives of human agents, appropriately expressing the preferences of those individuals whom they stand for. First, we formalize the setting of \textit{collective decision-making} -- as the episodic process of interaction between a group of agents and a decision mechanism. On this basis, we then formalize the problem of \textit{digital representation} -- as the simulation of an agent's behavior to yield equivalent outcomes from the mechanism. Finally, we conduct an empirical case study in the setting of \textit{consensus-finding} among diverse humans, and demonstrate the feasibility of fine-tuning large language models to act as digital representatives. 

**Abstract (ZH)**: 考虑集体决策过程，在这个过程中，一群个体通过代理（代表）的参与，相互选择最满意的结果。在这种背景下，“代表”是指通过代理来展现个体的偏好。为此，通过学习的人类行为模型可能在这个角色中发挥作用，对多智能体场景的研究和机制设计具有实际意义。在本研究中，我们探讨了训练“语言代理”以作为人类代理的代表，适当地表达被代表个体的偏好的可能性。首先，我们正式定义了“集体决策”——即一群智能体与决策机制之间交互的阶段过程。在此基础上，我们进一步定义了“数字代表”问题——即模拟一个代理的行为，使其能够从机制中产生等效的结果。最后，我们在多样的人类之间达成共识的背景下进行了实证研究案例分析，并证明了微调大型语言模型作为数字代表的可行性。 

---
# Reliable Conversational Agents under ASP Control that Understand Natural Language 

**Title (ZH)**: 在逻辑编程控制下可靠的对话代理及其对自然语言的理解 

**Authors**: Yankai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.09237)  

**Abstract**: Efforts have been made to make machines converse like humans in the past few decades. The recent techniques of Large Language Models (LLMs) make it possible to have human-like conversations with machines, but LLM's flaws of lacking understanding and reliability are well documented. We believe that the best way to eliminate this problem is to use LLMs only as parsers to translate text to knowledge and vice versa and carry out the conversation by reasoning over this knowledge using the answer set programming. I have been developing a framework based on LLMs and ASP to realize reliable chatbots that "understand" human conversation. This framework has been used to develop task-specific chatbots as well as socialbots. My future research is focused on making these chatbots scalable and trainable. 

**Abstract (ZH)**: 在过去的几十年中，人们一直在努力使机器的对话方式像人类一样。最近的大规模语言模型（LLMs）技术使得与机器进行像人类一样的对话成为可能，但LLMs在理解能力和可靠性方面存在的缺陷已经被充分记录。我们相信，解决这一问题的最佳方法是仅将LLMs用作解析器，用于文本到知识的翻译和反之亦然，并利用答案集编程对这些知识进行推理以进行对话。我一直致力于开发基于LLMs和ASP的框架，以实现能够“理解”人类对话的可靠聊天机器人。该框架已用于开发特定任务的聊天机器人和社会机器人。我的未来研究重点在于使这些聊天机器人具有可扩展性和可训练性。 

---
# Mind the Gaps: Logical English, Prolog, and Multi-agent Systems for Autonomous Vehicles 

**Title (ZH)**: 填补空白：逻辑英语、Prolog 与多代理系统在自主车辆中的应用 

**Authors**: Galileo Sartor, Adam Wyner, Giuseppe Contissa  

**Link**: [PDF](https://arxiv.org/pdf/2502.09216)  

**Abstract**: In this paper, we present a modular system for representing and reasoning with legal aspects of traffic rules for autonomous vehicles. We focus on a subset of the United Kingdom's Highway Code (HC) related to junctions. As human drivers and automated vehicles (AVs) will interact on the roads, especially in urban environments, we claim that an accessible, unitary, high-level computational model should exist and be applicable to both users. Autonomous vehicles introduce a shift in liability that should not bring disadvantages or increased burden on human drivers. We develop a system "in silico" of the model.  The proposed system is built of three main components: a natural language interface, using Logical English, which encodes the rules; an internal representation of the rules in Prolog; and an multi-agent-based simulation environment, built in NetLogo. The three components interact: Logical English is translated into and out of Prolog (along with some support code); Prolog and NetLogo interface via predicates. Such a modular approach enables the different components to carry different "burdens" in the overall system; it also allows swapping of modules. Given NetLogo, we can visualize the effect of the modeled rules as well as validate the system with a simple dynamic running scenario. Designated agents monitor the behaviour of the vehicles for compliance and record potential violations where they occur. The information on potential violations is then utilized by Validators, to determine whether the violation is punishable, differentiating between exceptions and cases. 

**Abstract (ZH)**: 在本文中，我们提出了一种模块化系统，用于表示和推理关于自动驾驶车辆交通规则的法律方面。我们专注于英国公路守则（HC）中与交叉路口相关内容的一个子集。鉴于人类驾驶员和自动驾驶车辆（AVs）将共用道路，特别是在城市环境中，我们认为应该存在一个便于访问的、统一的、高层次的计算模型，该模型适用于两者。自动驾驶车辆引入了责任转移，这种转移不应给人类驾驶员带来不利或额外的负担。我们对模型进行了“在硅中”的开发。该系统主要由三个组件构成：一个使用逻辑英语的自然语言接口，该接口编码了规则；用Prolog表示规则的内部表示；以及基于NetLogo构建的多代理仿真环境。这三个组件相互作用：逻辑英语在与Prolog交互时会进行翻译（同时包含一些辅助代码）；Prolog和NetLogo通过谓词进行接口交互。这种模块化的方法使得不同组件在整体系统中承担不同的“负担”；它还允许模块的更换。借助NetLogo，我们能够可视化模型规则的效果，并通过简单的动态运行场景验证系统。指定的代理监控车辆行为以确保遵守规则，并记录违规行为的发生。然后，验证器利用违规行为的信息来确定是否应对违规行为进行惩罚，区分情况与例外情况。 

---
# FLAME: Flexible LLM-Assisted Moderation Engine 

**Title (ZH)**: FLAME：灵活的大型语言模型辅助审核引擎 

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2502.09175)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展在管理用户-模型互动方面带来了重大挑战。尽管LLMs展现了显著的能力，但在对抗性攻击面前，特别是“逃逸攻击”技术，它们仍然脆弱，这些技术可以绕过内容安全措施。当前的内容审核系统主要依赖输入提示过滤，但这些方法已证明不够充分，如“最佳N项”（BoN）逃逸攻击技术对流行LLMs的成功率高达80%以上。在本文中，我们介绍了一种新的灵活的大语言模型辅助审核引擎（FLAME）：该方法将焦点从输入过滤转向输出审核。不同于传统的方法分析用户查询，FLAME评估模型响应，具有以下几个关键优势：（1）训练和推理中的计算效率高；（2）增强对BoN逃逸攻击的抵抗力；（3）通过可定制的主题过滤定义和更新安全标准的灵活性。我们的实验表明，FLAME显著优于现有的审核系统。例如，FLAME将针对GPT-4o-mini和DeepSeek-v3的攻击成功率降低了约9倍，同时保持了较低的计算开销。我们对多种LLMs进行了全面评估，并分析了该引擎在最新逃逸攻击中的效率。这项工作为LLMs开发更稳健和适应性更强的内容审核系统做出了贡献。 

---
# PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology 

**Title (ZH)**: PathFinder：一种应用于组织病理学的多模态多agent系统，用于医学诊断决策辅助 

**Authors**: Fatemeh Ghezloo, Mehmet Saygin Seyfioglu, Rustin Soraki, Wisdom O. Ikezogwo, Beibin Li, Tejoram Vivekanandan, Joann G. Elmore, Ranjay Krishna, Linda Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2502.08916)  

**Abstract**: Diagnosing diseases through histopathology whole slide images (WSIs) is fundamental in modern pathology but is challenged by the gigapixel scale and complexity of WSIs. Trained histopathologists overcome this challenge by navigating the WSI, looking for relevant patches, taking notes, and compiling them to produce a final holistic diagnostic. Traditional AI approaches, such as multiple instance learning and transformer-based models, fail short of such a holistic, iterative, multi-scale diagnostic procedure, limiting their adoption in the real-world. We introduce PathFinder, a multi-modal, multi-agent framework that emulates the decision-making process of expert pathologists. PathFinder integrates four AI agents, the Triage Agent, Navigation Agent, Description Agent, and Diagnosis Agent, that collaboratively navigate WSIs, gather evidence, and provide comprehensive diagnoses with natural language explanations. The Triage Agent classifies the WSI as benign or risky; if risky, the Navigation and Description Agents iteratively focus on significant regions, generating importance maps and descriptive insights of sampled patches. Finally, the Diagnosis Agent synthesizes the findings to determine the patient's diagnostic classification. Our Experiments show that PathFinder outperforms state-of-the-art methods in skin melanoma diagnosis by 8% while offering inherent explainability through natural language descriptions of diagnostically relevant patches. Qualitative analysis by pathologists shows that the Description Agent's outputs are of high quality and comparable to GPT-4o. PathFinder is also the first AI-based system to surpass the average performance of pathologists in this challenging melanoma classification task by 9%, setting a new record for efficient, accurate, and interpretable AI-assisted diagnostics in pathology. Data, code and models available at this https URL 

**Abstract (ZH)**: 通过对大尺寸切片图像（WSI）进行组织病理学诊断是现代病理学的基础，但WSI的高像素和复杂性构成了挑战。经过训练的病理学家通过导航WSI，寻找相关区域，做笔记，并整理这些信息以生成最终的整体诊断。传统的AI方法，如多个实例学习和基于Transformer的模型，在实现这样一种综合性、迭代性和多尺度的诊断程序方面力有未逮，限制了其在实际中的应用。我们提出了一种多模态、多代理框架PathFinder，模拟了专家病理学家的决策过程。PathFinder 综合了四个AI代理：分诊代理、导航代理、描述代理和诊断代理，它们协同工作，导航WSI，收集证据，并以自然语言提供全面的诊断结果。分诊代理将WSI分类为良性或有风险；如果有风险，导航代理和描述代理会迭代地聚焦于重要的区域，生成重要性地图和描述性见解，进一步分析采样的切片区域。最后，诊断代理综合这些发现，确定患者的诊断分类。实验结果显示，PathFinder 在皮肤黑色素瘤诊断上比最先进的方法高出8%，并通过自然语言描述诊断相关的切片区域实现了内置的可解释性。病理学家的定性分析表明，描述代理的输出质量较高，与GPT-4o相当。PathFinder 也是第一个在这一具有挑战性的黑色素瘤分类任务上超过平均病理学家表现的基于AI的系统，使其在病理学中成为高效、准确和可解释的AI辅助诊断的新标准。更多信息、代码和模型请参见此链接：[此 https URL] 

---
# Can a Single Model Master Both Multi-turn Conversations and Tool Use? CALM: A Unified Conversational Agentic Language Model 

**Title (ZH)**: 单个模型能否同时掌握多轮对话和工具使用？统一对话型代理语言模型——CALM 

**Authors**: Emre Can Acikgoz, Jeremiah Greer, Akul Datta, Ze Yang, William Zeng, Oussama Elachqar, Emmanouil Koukoumidis, Dilek Hakkani-Tür, Gokhan Tur  

**Link**: [PDF](https://arxiv.org/pdf/2502.08820)  

**Abstract**: Large Language Models (LLMs) with API-calling capabilities enabled building effective Language Agents (LA), while also revolutionizing the conventional task-oriented dialogue (TOD) paradigm. However, current approaches face a critical dilemma: TOD systems are often trained on a limited set of target APIs, requiring new data to maintain their quality when interfacing with new services, while LAs are not trained to maintain user intent over multi-turn conversations. Because both robust multi-turn management and advanced function calling are crucial for effective conversational agents, we evaluate these skills on three popular benchmarks: MultiWOZ 2.4 (TOD), BFCL V3 (LA), and API-Bank (LA), and our analyses reveal that specialized approaches excel in one domain but underperform in the other. To bridge this chasm, we introduce CALM (Conversational Agentic Language Model), a unified approach that integrates both conversational and agentic capabilities. We created CALM-IT, a carefully constructed multi-task dataset that interleave multi-turn ReAct reasoning with complex API usage. Using CALM-IT, we train three models CALM 8B, CALM 70B, and CALM 405B, which outperform top domain-specific models, including GPT-4o, across all three benchmarks. 

**Abstract (ZH)**: larg语言模型（LLMs）具备API调用能力，能够构建有效的语言代理（LAs），同时也在传统任务导向对话（TOD）范式上进行了革命。然而，当前的方法面临一个关键的困境：TOD系统通常仅在有限的目标API集上进行训练，当与新的服务交互时，需要新数据来保持其质量，而LAs则未被训练以在多轮对话中保持用户意图。由于强大的多轮管理和高级功能调用对于有效的对话代理至关重要，我们在三个流行的基准测试上评估了这些技能：MultiWOZ 2.4（TOD）、BFCL V3（LA）和API-Bank（LA），分析结果显示专门的方法在某一领域表现出色，但在另一领域则表现不佳。为弥合这一差距，我们提出了CALM（Conversational Agentic Language Model），这是一种统一的方法，融合了对话能力和代理功能。我们创建了CALM-IT，这是一个精心构建的多任务数据集，结合了多轮ReAct推理和复杂API使用。使用CALM-IT，我们训练了三个模型：CALM 8B、CALM 70B和CALM 405B，它们在三个基准测试上均表现出色，超越了包括GPT-4o在内的顶尖领域特定模型。 

---
# KIMAs: A Configurable Knowledge Integrated Multi-Agent System 

**Title (ZH)**: KIMAs：一种可配置的知识集成多智能体系统 

**Authors**: Zitao Li, Fei Wei, Yuexiang Xie, Dawei Gao, Weirui Kuang, Zhijian Ma, Bingchen Qian, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2502.09596)  

**Abstract**: Knowledge-intensive conversations supported by large language models (LLMs) have become one of the most popular and helpful applications that can assist people in different aspects. Many current knowledge-intensive applications are centered on retrieval-augmented generation (RAG) techniques. While many open-source RAG frameworks facilitate the development of RAG-based applications, they often fall short in handling practical scenarios complicated by heterogeneous data in topics and formats, conversational context management, and the requirement of low-latency response times. This technical report presents a configurable knowledge integrated multi-agent system, KIMAs, to address these challenges. KIMAs features a flexible and configurable system for integrating diverse knowledge sources with 1) context management and query rewrite mechanisms to improve retrieval accuracy and multi-turn conversational coherency, 2) efficient knowledge routing and retrieval, 3) simple but effective filter and reference generation mechanisms, and 4) optimized parallelizable multi-agent pipeline execution. Our work provides a scalable framework for advancing the deployment of LLMs in real-world settings. To show how KIMAs can help developers build knowledge-intensive applications with different scales and emphases, we demonstrate how we configure the system to three applications already running in practice with reliable performance. 

**Abstract (ZH)**: 由大型语言模型（LLMs）支持的知识密集型对话已成为各种应用场景中最受欢迎和最有帮助的应用之一。许多当前的知识密集型应用主要集中在检索增强生成（RAG）技术上。虽然许多开源RAG框架促进了基于RAG的应用程序的开发，但它们在处理由异构数据主题和格式、对话上下文管理以及低延迟响应时间要求带来的复杂场景时往往难以胜任。本技术报告提出了一种可配置的知识集成多代理系统（KIMAs），以应对这些挑战。KIMAs具备以下特点：1）灵活且可配置的系统，用于集成多种知识来源，包括上下文管理和查询重写机制，以提高检索准确性和多轮对话的连贯性；2）高效的知识路由和检索；3）简单而有效的过滤和参考生成机制；4）优化的可并行化的多代理流水线执行。我们的工作提供了一种可扩展的框架，以促进LLM在实际场景中的部署。为了展示KIMAs如何帮助开发者构建具有不同规模和侧重点的知识密集型应用，我们演示了如何将系统配置应用于三个已成功运行的实际案例，并展示了其可靠的性能。 

---
# MDCrow: Automating Molecular Dynamics Workflows with Large Language Models 

**Title (ZH)**: MDCrow：使用大规模语言模型自动化分子动力学工作流 

**Authors**: Quintina Campbell, Sam Cox, Jorge Medina, Brittany Watterson, Andrew D. White  

**Link**: [PDF](https://arxiv.org/pdf/2502.09565)  

**Abstract**: Molecular dynamics (MD) simulations are essential for understanding biomolecular systems but remain challenging to automate. Recent advances in large language models (LLM) have demonstrated success in automating complex scientific tasks using LLM-based agents. In this paper, we introduce MDCrow, an agentic LLM assistant capable of automating MD workflows. MDCrow uses chain-of-thought over 40 expert-designed tools for handling and processing files, setting up simulations, analyzing the simulation outputs, and retrieving relevant information from literature and databases. We assess MDCrow's performance across 25 tasks of varying required subtasks and difficulty, and we evaluate the agent's robustness to both difficulty and prompt style. \texttt{gpt-4o} is able to complete complex tasks with low variance, followed closely by \texttt{llama3-405b}, a compelling open-source model. While prompt style does not influence the best models' performance, it has significant effects on smaller models. 

**Abstract (ZH)**: 分子动力学（MD）模拟对于理解生物分子系统至关重要，但自动化的实现仍然具有挑战性。近年来，大规模语言模型（LLM）在使用基于LLM的代理自动化复杂科学任务方面取得了显著成功。在本文中，我们介绍了MDCrow，这是一种能够自动化MD工作流程的代理型LLM辅助工具。MDCrow通过使用40个专家设计的工具进行文件处理和处理、设置模拟、分析模拟输出，并从文献和数据库中检索相关的信息，实现了这一目标。我们对MDCrow进行了评估，涉及25项不同子任务和难度的任务，评估了该代理在难度和指令风格方面的一致性。结果显示，\texttt{gpt-4o}能够以极低的方差完成复杂任务，紧随其后的是\texttt{llama3-405b}，这是一个有吸引力的开源模型。虽然指令风格对最佳模型的性能影响不大，但对较小模型则有显著影响。 

---
# Commonsense Reasoning-Aided Autonomous Vehicle Systems 

**Title (ZH)**: 基于常识推理的自主车辆系统 

**Authors**: Keegan Kimbrell  

**Link**: [PDF](https://arxiv.org/pdf/2502.09233)  

**Abstract**: Autonomous Vehicle (AV) systems have been developed with a strong reliance on machine learning techniques. While machine learning approaches, such as deep learning, are extremely effective at tasks that involve observation and classification, they struggle when it comes to performing higher level reasoning about situations on the road. This research involves incorporating commonsense reasoning models that use image data to improve AV systems. This will allow AV systems to perform more accurate reasoning while also making them more adjustable, explainable, and ethical. This paper will discuss the findings so far and motivate its direction going forward. 

**Abstract (ZH)**: 自动驾驶（AV）系统依赖机器学习技术得到了快速发展。尽管深度学习等机器学习方法在涉及观察和分类的任务中非常有效，但在执行道路情况的高层次推理方面却表现出局限性。本研究旨在结合使用图像数据的常识推理模型，以提升AV系统的性能。这将使AV系统能够在进行更准确的推理的同时，变得更具可调性、可解释性和伦理性。本文将讨论迄今为止的研究发现，并激发未来研究的方向。 

---
# Reinforced Large Language Model is a formal theorem prover 

**Title (ZH)**: 强化型大型语言模型是一个形式定理证明器 

**Authors**: Zhiling Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.08908)  

**Abstract**: To take advantage of Large Language Model in theorem formalization and proof, we propose a reinforcement learning framework to iteratively optimize the pretrained LLM by rolling out next tactics and comparing them with the expected ones. The experiment results show that it helps to achieve a higher accuracy compared with directly fine-tuned LLM. 

**Abstract (ZH)**: 为了利用大型语言模型在定理形式化和证明中的优势，我们提出了一种强化学习框架，通过迭代优化预训练的大语言模型，生成下一个策略并将其与期望策略进行比较。实验结果表明，这种方法在准确性方面优于直接微调大语言模型。 

---
# Architecture for Simulating Behavior Mode Changes in Norm-Aware Autonomous Agents 

**Title (ZH)**: 规范翻译后的标题为：

面向规范意识自主agent行为模式变化仿真架构

这个翻译保持了原标题的学术性和专业性，确保了中文表达的准确性和流畅性。 

**Authors**: Sean Glaze, Daniela Inclezan  

**Link**: [PDF](https://arxiv.org/pdf/2502.09215)  

**Abstract**: This paper presents an architecture for simulating the actions of a norm-aware intelligent agent whose behavior with respect to norm compliance is set, and can later be changed, by a human controller. Updating an agent's behavior mode from a norm-abiding to a riskier one may be relevant when the agent is involved in time-sensitive rescue operations, for example. We base our work on the Authorization and Obligation Policy Language AOPL designed by Gelfond and Lobo for the specification of norms. We introduce an architecture and a prototype software system that can be used to simulate an agent's plans under different behavior modes that can later be changed by the controller. We envision such software to be useful to policy makers, as they can more readily understand how agents may act in certain situations based on the agents' attitudes towards norm-compliance. Policy makers may then refine their policies if simulations show unwanted consequences. 

**Abstract (ZH)**: 本文提出了一种架构，用于模拟一个遵守规范的智能代理的行为，其行为模式在遵守规范方面由人类控制器设定，并且可以在之后由控制器进行更改。当代理参与时间敏感的救援操作时，从遵守规范的行为模式更新到更危险的行为模式可能是相关的。我们的工作基于Gelfond和Lobo为规范的规范性规定设计的授权与义务政策语言（Authorization and Obligation Policy Language, AOPL）。我们提出了一种架构和一个原型软件系统，该系统可以在不同的行为模式下模拟代理的计划，并且这些模式可以由控制器在之后进行更改。我们设想这种软件对于政策制定者来说可能很有用，因为通过模拟可以看到代理在特定情况下可能如何行动，基于代理对其遵守规范的态度。如果模拟显示了不希望的后果，政策制定者可以据此调整其政策。 

---
# AIDE: Agentically Improve Visual Language Model with Domain Experts 

**Title (ZH)**: AIDE：由领域专家提升的代理改善视觉语言模型 

**Authors**: Ming-Chang Chiu, Fuxiao Liu, Karan Sapra, Andrew Tao, Yaser Jacoob, Xuezhe Ma, Zhiding Yu, Guilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09051)  

**Abstract**: The enhancement of Visual Language Models (VLMs) has traditionally relied on knowledge distillation from larger, more capable models. This dependence creates a fundamental bottleneck for improving state-of-the-art systems, particularly when no superior models exist. We introduce AIDE (Agentic Improvement through Domain Experts), a novel framework that enables VLMs to autonomously enhance their capabilities by leveraging specialized domain expert models. AIDE operates through a four-stage process: (1) identifying instances for refinement, (2) engaging domain experts for targeted analysis, (3) synthesizing expert outputs with existing data, and (4) integrating enhanced instances into the training pipeline. Experiments on multiple benchmarks, including MMMU, MME, MMBench, etc., demonstrate AIDE's ability to achieve notable performance gains without relying on larger VLMs nor human supervision. Our framework provides a scalable, resource-efficient approach to continuous VLM improvement, addressing critical limitations in current methodologies, particularly valuable when larger models are unavailable to access. 

**Abstract (ZH)**: 视觉语言模型（VLMs）的传统增强依赖于从更大、更强大的模型中提取知识蒸馏。这种依赖性为改善最先进的系统设定了一个根本性的瓶颈，尤其是在没有更优模型可用的情况下。我们提出了AIDE（通过领域专家的自主提升），这是一种新型框架，它使VLMs能够自主增强其能力，通过利用专门的领域专家模型。AIDE 通过四阶段过程运作：（1）识别需要改进的实例，（2）邀请领域专家进行针对性分析，（3）将专家输出与现有数据相结合，以及（4）将增强的实例整合到训练管道中。在多个基准测试中，包括MMMU、MME、MMbench等，实验结果表明AIDE能够在不依赖更大规模的VLMs和人类监督的情况下实现显著的性能提升。我们的框架提供了一种可扩展且资源高效的持续改进VLM的方法，解决了当前方法的关键限制问题，尤其是在无法访问更大模型的情况下尤为宝贵。 

---
# Few is More: Task-Efficient Skill-Discovery for Multi-Task Offline Multi-Agent Reinforcement Learning 

**Title (ZH)**: 更少也更多：多任务 Offline 多智能体强化学习中的任务高效技能发现 

**Authors**: Xun Wang, Zhuoran Li, Hai Zhong, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08985)  

**Abstract**: As a data-driven approach, offline MARL learns superior policies solely from offline datasets, ideal for domains rich in historical data but with high interaction costs and risks. However, most existing methods are task-specific, requiring retraining for new tasks, leading to redundancy and inefficiency. To address this issue, in this paper, we propose a task-efficient multi-task offline MARL algorithm, Skill-Discovery Conservative Q-Learning (SD-CQL). Unlike existing offline skill-discovery methods, SD-CQL discovers skills by reconstructing the next observation. It then evaluates fixed and variable actions separately and employs behavior-regularized conservative Q-learning to execute the optimal action for each skill. This approach eliminates the need for local-global alignment and enables strong multi-task generalization from limited small-scale source tasks. Substantial experiments on StarCraftII demonstrates the superior generalization performance and task-efficiency of SD-CQL. It achieves the best performance on $\textbf{10}$ out of $14$ task sets, with up to $\textbf{65%}$ improvement on individual task sets, and is within $4\%$ of the best baseline on the remaining four. 

**Abstract (ZH)**: 作为一种数据驱动的方法，离线多智能体 reinforcement 学习（MARL）仅从离线数据集中学习出优质的策略，特别适用于历史数据丰富但交互成本和风险较高的领域。然而，现有的大多数方法都是针对特定任务设计的，需要为新任务重新训练，导致冗余和低效。为解决这一问题，本文提出了一种高效的多任务离线 MARL 算法——技能发现保守 Q 学习（SD-CQL）。与现有的离线技能发现方法不同，SD-CQL 通过重构下一个观测来发现技能，然后分别评估固定和可变动作，并采用行为正则化的保守 Q 学习来执行每项技能下的最优动作。这种方法消除了局部与全局对齐的需求，并且可以从少量的小规模源任务中实现强大的多任务泛化能力。在 StarCraft II 的大量实验中，SD-CQL 展现出卓越的泛化能力和任务效率。它在 14 项任务集中取得了 10 项最佳性能，单个任务集的性能提升最高可达 65%，在剩余的四个任务集中，其性能与最佳基线相差不到 4%。 

---
# SkyRover: A Modular Simulator for Cross-Domain Pathfinding 

**Title (ZH)**: SkyRover：一种跨域路径规划模块化模拟器 

**Authors**: Wenhui Ma, Wenhao Li, Bo Jin, Changhong Lu, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08969)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) and Automated Guided Vehicles (AGVs) increasingly collaborate in logistics, surveillance, inspection tasks and etc. However, existing simulators often focus on a single domain, limiting cross-domain study. This paper presents the SkyRover, a modular simulator for UAV-AGV multi-agent pathfinding (MAPF). SkyRover supports realistic agent dynamics, configurable 3D environments, and convenient APIs for external solvers and learning methods. By unifying ground and aerial operations, it facilitates cross-domain algorithm design, testing, and benchmarking. Experiments highlight SkyRover's capacity for efficient pathfinding and high-fidelity simulations in UAV-AGV coordination. Project is available at this https URL. 

**Abstract (ZH)**: 无人驾驶航空器（UAVs）和自动引导车（AGVs）在物流、 surveillance、检查等任务中越来越多地协同工作。然而，现有的模拟器往往专注于单一领域，限制了跨域研究。本文介绍了SkyRover，这是一种用于多智能体路径规划（MAPF）的UAV-AGV模块化模拟器。SkyRover支持真实的智能体动力学，可配置的3D环境，并提供了方便的外部求解器和学习方法的API。通过统一地面和空中操作，它促进了跨域算法的设计、测试和基准测试。实验结果显示，SkyRover在UAV-AGV协调中的高效路径规划和高保真模拟能力。项目详情可访问此链接：[项目链接]。 

---
# RTBAS: Defending LLM Agents Against Prompt Injection and Privacy Leakage 

**Title (ZH)**: RTBAS: 防御提示注入和隐私泄露的大语言模型代理 

**Authors**: Peter Yong Zhong, Siyuan Chen, Ruiqi Wang, McKenna McCall, Ben L. Titzer, Heather Miller  

**Link**: [PDF](https://arxiv.org/pdf/2502.08966)  

**Abstract**: Tool-Based Agent Systems (TBAS) allow Language Models (LMs) to use external tools for tasks beyond their standalone capabilities, such as searching websites, booking flights, or making financial transactions. However, these tools greatly increase the risks of prompt injection attacks, where malicious content hijacks the LM agent to leak confidential data or trigger harmful actions. Existing defenses (OpenAI GPTs) require user confirmation before every tool call, placing onerous burdens on users. We introduce Robust TBAS (RTBAS), which automatically detects and executes tool calls that preserve integrity and confidentiality, requiring user confirmation only when these safeguards cannot be ensured. RTBAS adapts Information Flow Control to the unique challenges presented by TBAS. We present two novel dependency screeners, using LM-as-a-judge and attention-based saliency, to overcome these challenges. Experimental results on the AgentDojo Prompt Injection benchmark show RTBAS prevents all targeted attacks with only a 2% loss of task utility when under attack, and further tests confirm its ability to obtain near-oracle performance on detecting both subtle and direct privacy leaks. 

**Abstract (ZH)**: 基于工具的代理系统（TBAS）使语言模型（LMs）能够利用外部工具完成超出其独立能力的任务，如网上搜索、预订航班或进行金融交易。然而，这些工具大大增加了提示注入攻击的风险，即恶意内容劫持LM代理以泄露机密数据或触发有害行为。现有的防御措施（如OpenAI的GPT）在每次调用工具之前都需要用户确认，给用户带来了沉重的负担。我们提出了鲁棒TBAS（RTBAS），它能够自动检测和执行保持完整性和保密性的工具调用，仅在这些保障措施无法确保时才要求用户确认。RTBAS将信息流控制方法应用于TBAS的独特挑战中。我们提出了两种新的依赖性筛选器，利用LM作为法官以及基于注意力的显著性，以克服这些挑战。在AgentDojo提示注入基准测试上的实验结果表明，RTBAS在受攻击情况下仅损失2%的任务实用性即可防止所有针对攻击，并且进一步测试证明它能够获得接近Oracle性能的检测细微和直接隐私泄露的能力。 

---
# Exploring Emotion-Sensitive LLM-Based Conversational AI 

**Title (ZH)**: 探索情绪感知的大语言模型驱动的对话人工智能 

**Authors**: Antonin Brun, Ruying Liu, Aryan Shukla, Frances Watson, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2502.08920)  

**Abstract**: Conversational AI chatbots have become increasingly common within the customer service industry. Despite improvements in their emotional development, they often lack the authenticity of real customer service interactions or the competence of service providers. By comparing emotion-sensitive and emotion-insensitive LLM-based chatbots across 30 participants, we aim to explore how emotional sensitivity in chatbots influences perceived competence and overall customer satisfaction in service interactions. Additionally, we employ sentiment analysis techniques to analyze and interpret the emotional content of user inputs. We highlight that perceptions of chatbot trustworthiness and competence were higher in the case of the emotion-sensitive chatbot, even if issue resolution rates were not affected. We discuss implications of improved user satisfaction from emotion-sensitive chatbots and potential applications in support services. 

**Abstract (ZH)**: 对话式人工智能聊天机器人在客户服务行业中越来越普遍。尽管在情感发展方面取得了进步，但它们仍然缺乏真实客户服务互动的真诚性，或者服务提供者的专业性。通过在30名参与者中比较情感敏感和非情感敏感的大规模语言模型（LLM）聊天机器人，我们旨在探讨聊天机器人的情感敏感性如何影响客户服务中感知的专业性和总体客户满意度。此外，我们还采用情感分析技术来分析和解释用户输入中的情感内容。我们强调，即使问题解决率未受影响，情感敏感聊天机器人的可信度和专业性感知更高。我们讨论了情感敏感聊天机器人改善用户满意度的含义及其在支持服务中的潜在应用。 

---
# AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society 

**Title (ZH)**: 《AgentSociety：由大模型驱动的生成式智能体大规模仿真促进对人类行为和社会的理解》

这个标题翻译旨在保持原文的学术规范和研究主题的一致性，同时确保中文表达的自然流畅。 

**Authors**: Jinghua Piao, Yuwei Yan, Jun Zhang, Nian Li, Junbo Yan, Xiaochong Lan, Zhihong Lu, Zhiheng Zheng, Jing Yi Wang, Di Zhou, Chen Gao, Fengli Xu, Fang Zhang, Ke Rong, Jun Su, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.08691)  

**Abstract**: Understanding human behavior and society is a central focus in social sciences, with the rise of generative social science marking a significant paradigmatic shift. By leveraging bottom-up simulations, it replaces costly and logistically challenging traditional experiments with scalable, replicable, and systematic computational approaches for studying complex social dynamics. Recent advances in large language models (LLMs) have further transformed this research paradigm, enabling the creation of human-like generative social agents and realistic simulacra of society. In this paper, we propose AgentSociety, a large-scale social simulator that integrates LLM-driven agents, a realistic societal environment, and a powerful large-scale simulation engine. Based on the proposed simulator, we generate social lives for over 10k agents, simulating their 5 million interactions both among agents and between agents and their environment. Furthermore, we explore the potential of AgentSociety as a testbed for computational social experiments, focusing on four key social issues: polarization, the spread of inflammatory messages, the effects of universal basic income policies, and the impact of external shocks such as hurricanes. These four issues serve as valuable cases for assessing AgentSociety's support for typical research methods -- such as surveys, interviews, and interventions -- as well as for investigating the patterns, causes, and underlying mechanisms of social issues. The alignment between AgentSociety's outcomes and real-world experimental results not only demonstrates its ability to capture human behaviors and their underlying mechanisms, but also underscores its potential as an important platform for social scientists and policymakers. 

**Abstract (ZH)**: 理解人类行为和社会现象是社会科学的核心关注点，而生成性社会科学的兴起标志着一个重要的范式转变。通过利用自底向上的模拟方法，它取代了传统实验中的高昂成本和实施难度，采用了可扩展、可重复和系统性的计算方法来研究复杂的社会动态。近年来，大型语言模型（LLMs）的进展进一步改变了这一研究范式，使其能够生成类人的生成性社会代理和现实的社会模拟。在本文中，我们提出了一种大规模社会模拟器AgentSociety，该模拟器结合了由大型语言模型驱动的代理、现实的社会环境以及强大的大规模仿真引擎。基于该模拟器，我们为超过10000个代理生成了社会生活，模拟了它们之间的500万次交互，包括代理之间的交互以及代理与其环境之间的交互。此外，我们探讨了AgentSociety作为计算社会实验的试验场的潜力，重点关注四个关键的社会问题：极化、煽动性信息的传播、普遍基本收入政策的影响以及外部冲击如飓风等的影响。这四个问题为评估AgentSociety对典型研究方法的支持——如调查、访谈和干预——以及探讨社会问题的模式、原因和潜在机制提供了宝贵的案例。AgentSociety的结果与现实世界实验结果的一致性不仅展示了其捕捉人类行为及其潜在机制的能力，而且还强调了其作为社会科学家和政策制定者重要平台的潜力。 

---
# Centrally Coordinated Multi-Agent Reinforcement Learning for Power Grid Topology Control 

**Title (ZH)**: 中心协调的多Agent强化学习在电力网络拓扑控制中的应用 

**Authors**: Barbera de Mol, Davide Barbieri, Jan Viebahn, Davide Grossi  

**Link**: [PDF](https://arxiv.org/pdf/2502.08681)  

**Abstract**: Power grid operation is becoming more complex due to the increase in generation of renewable energy. The recent series of Learning To Run a Power Network (L2RPN) competitions have encouraged the use of artificial agents to assist human dispatchers in operating power grids. However, the combinatorial nature of the action space poses a challenge to both conventional optimizers and learned controllers. Action space factorization, which breaks down decision-making into smaller sub-tasks, is one approach to tackle the curse of dimensionality. In this study, we propose a centrally coordinated multi-agent (CCMA) architecture for action space factorization. In this approach, regional agents propose actions and subsequently a coordinating agent selects the final action. We investigate several implementations of the CCMA architecture, and benchmark in different experimental settings against various L2RPN baseline approaches. The CCMA architecture exhibits higher sample efficiency and superior final performance than the baseline approaches. The results suggest high potential of the CCMA approach for further application in higher-dimensional L2RPN as well as real-world power grid settings. 

**Abstract (ZH)**: 随着可再生能源发电量的增加，电网运行正变得日益复杂。近期举办的多次“学习运行电力网络”（L2RPN）竞赛鼓励使用人工代理来协助人类调度员进行电力网络的操作。然而，动作空间的组合性质对传统的优化器和学习控制器都构成了挑战。动作空间分解，即将决策过程分解为更小的子任务，是应对维数灾的一个方法。在本研究中，我们提出了一种集中协调的多代理（CCMA）架构来处理动作空间分解问题。在该方法中，区域代理提出动作，随后协调代理选择最终动作。我们研究了几种CCMA架构的实现方式，并在不同的实验环境中与L2RPN的各种基线方法进行了 Benchmark。研究结果表明，CCMA架构在样本效率和最终性能上优于基线方法。结果表明，CCMA方法在更高维的L2RPN以及实际电力网络环境中有很高的应用潜力。 

---
# Refining Positive and Toxic Samples for Dual Safety Self-Alignment of LLMs with Minimal Human Interventions 

**Title (ZH)**: 减少人类干预以精炼正向和有毒样本，实现大规模语言模型的双重安全自我对齐 

**Authors**: Jingxin Xu, Guoshun Nan, Sheng Guan, Sicong Leng, Yilian Liu, Zixiao Wang, Yuyang Ma, Zhili Zhou, Yanzhao Hou, Xiaofeng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.08657)  

**Abstract**: Recent AI agents, such as ChatGPT and LLaMA, primarily rely on instruction tuning and reinforcement learning to calibrate the output of large language models (LLMs) with human intentions, ensuring the outputs are harmless and helpful. Existing methods heavily depend on the manual annotation of high-quality positive samples, while contending with issues such as noisy labels and minimal distinctions between preferred and dispreferred response data. However, readily available toxic samples with clear safety distinctions are often filtered out, removing valuable negative references that could aid LLMs in safety alignment. In response, we propose PT-ALIGN, a novel safety self-alignment approach that minimizes human supervision by automatically refining positive and toxic samples and performing fine-grained dual instruction tuning. Positive samples are harmless responses, while toxic samples deliberately contain extremely harmful content, serving as a new supervisory signals. Specifically, we utilize LLM itself to iteratively generate and refine training instances by only exploring fewer than 50 human annotations. We then employ two losses, i.e., maximum likelihood estimation (MLE) and fine-grained unlikelihood training (UT), to jointly learn to enhance the LLM's safety. The MLE loss encourages an LLM to maximize the generation of harmless content based on positive samples. Conversely, the fine-grained UT loss guides the LLM to minimize the output of harmful words based on negative samples at the token-level, thereby guiding the model to decouple safety from effectiveness, directing it toward safer fine-tuning objectives, and increasing the likelihood of generating helpful and reliable content. Experiments on 9 popular open-source LLMs demonstrate the effectiveness of our PT-ALIGN for safety alignment, while maintaining comparable levels of helpfulness and usefulness. 

**Abstract (ZH)**: 近年来，诸如ChatGPT和LLaMA之类的AI代理主要依赖于指令微调和强化学习来校准大语言模型（LLMs）的输出以符合人类意图，从而确保输出既无害又有益。现有方法高度依赖于高质量正样本的手动标注，同时面临标签噪声大和偏好响应与非偏好响应数据区分度小等问题。然而，容易获得的具有明确安全区分的有毒样本往往被过滤掉，从而消除了能够帮助LLMs实现安全对齐的重要负样本参考。为应对这一问题，我们提出了一种新颖的安全自我对齐方法——PT-ALIGN，该方法通过自动优化正样本和有毒样本并进行细粒度的双指令微调来最大限度地减少人类监督。正样本为无害的响应，而有毒样本故意包含极端有害的内容，作为新的监督信号。具体而言，我们利用LLM本身通过探索不到50个人标注实例进行迭代生成和优化训练实例。然后，我们使用两种损失，即最大似然估计（MLE）和细粒度的不可能性训练（UT），以联合学习方式提升LLM的安全性。MLE损失鼓励LLM根据正样本生成尽可能多的无害内容。相反，细粒度的UT损失在 token 级别引导LLM减少有害词汇的输出，从而指导模型安全性和效果脱钩，使其朝着更安全的微调目标发展，并增加生成有益和可靠内容的可能性。实验表明，与9个流行的开源LLM相比，PT-ALIGN在安全对齐方面表现出有效性，同时保持了相近的有益性和实用性。 

---
