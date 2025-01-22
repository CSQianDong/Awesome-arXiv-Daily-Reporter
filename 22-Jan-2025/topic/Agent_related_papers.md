# PlotEdit: Natural Language-Driven Accessible Chart Editing in PDFs via Multimodal LLM Agents 

**Title (ZH)**: PlotEdit: 通过多模态大语言模型代理实现的基于自然语言的PDF图表编辑 

**Authors**: Kanika Goswami, Puneet Mathur, Ryan Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2501.11233)  

**Abstract**: Chart visualizations, while essential for data interpretation and communication, are predominantly accessible only as images in PDFs, lacking source data tables and stylistic information. To enable effective editing of charts in PDFs or digital scans, we present PlotEdit, a novel multi-agent framework for natural language-driven end-to-end chart image editing via self-reflective LLM agents. PlotEdit orchestrates five LLM agents: (1) Chart2Table for data table extraction, (2) Chart2Vision for style attribute identification, (3) Chart2Code for retrieving rendering code, (4) Instruction Decomposition Agent for parsing user requests into executable steps, and (5) Multimodal Editing Agent for implementing nuanced chart component modifications - all coordinated through multimodal feedback to maintain visual fidelity. PlotEdit outperforms existing baselines on the ChartCraft dataset across style, layout, format, and data-centric edits, enhancing accessibility for visually challenged users and improving novice productivity. 

**Abstract (ZH)**: 图表可视化虽然对于数据解释和传达至关重要，但在大多数情况下仅作为PDF中的图像存在，缺乏原始数据表和风格信息。为了使图表在PDF或数字扫描中的有效编辑成为可能，我们提出了PlotEdit，这是一种新颖的多代理框架，利用自省语言模型代理实现自然语言驱动的端到端图表图像编辑。PlotEdit协调了五个语言模型代理：（1）Chart2Table，用于提取数据表；（2）Chart2Vision，用于识别样式属性；（3）Chart2Code，用于检索渲染代码；（4）指令分解代理，用于将用户请求解析为可执行步骤；以及（5）多模态编辑代理，用于实现细腻的图表组件修改——所有这一切都通过多模态反馈来协调，以保持视觉保真度。在ChartCraft数据集上，PlotEdit在风格、布局、格式和数据导向编辑方面优于现有基线，提升了视觉障碍用户的可访问性，并改善了新手的生产力。 

---
# Med-R$^2$: Crafting Trustworthy LLM Physicians through Retrieval and Reasoning of Evidence-Based Medicine 

**Title (ZH)**: Med-R²：通过基于证据的医学检索与推理打造可信赖的LLM医生 

**Authors**: Keer Lu, Zheng Liang, Da Pan, Shusen Zhang, Xin Wu, Weipeng Chen, Zenan Zhou, Guosheng Dong, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11885)  

**Abstract**: In recent years, Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. However, despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 14.87\% improvement over vanilla RAG methods and even a 3.59\% enhancement compared to fine-tuning strategies, without incurring additional training costs. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在临床场景中展现出了非凡的能力。然而，尽管这些模型具有潜力，在将它们应用于医疗领域时仍面临诸多挑战。依赖医学数据集进行训练的方法成本高昂，且可能受到过时训练数据的影响。利用外部知识库是一个可行的替代方案，但此类方法也面临着诸如检索精度较低和难以有效抽取答案等障碍。这些问题共同阻碍了LLMs在掌握医学专业知识方面达到预期水平。为解决这些问题，我们提出了Med-R^2，这是一种新颖的LLM医生框架，遵循循证医学（EBM）过程，高效地整合了检索机制以及证据的选择和推理过程，从而增强LLMs在医疗场景中的问题解决能力，并培养一种值得信赖的LLM医生。我们全面的实验表明，Med-R^2 在与基础检索聚合（RAG）方法相比时展示了14.87% 的改进，并且相较于微调策略还展现了3.59% 的提升，而无需额外的训练成本。 

---
# Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks 

**Title (ZH)**: 移动代理-E：用于复杂任务的自我进化移动助理 

**Authors**: Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.11733)  

**Abstract**: Smartphones have become indispensable in modern life, yet navigating complex tasks on mobile devices often remains frustrating. Recent advancements in large multimodal model (LMM)-based mobile agents have demonstrated the ability to perceive and act in mobile environments. However, current approaches face significant limitations: they fall short in addressing real-world human needs, struggle with reasoning-intensive and long-horizon tasks, and lack mechanisms to learn and improve from prior experiences. To overcome these challenges, we introduce Mobile-Agent-E, a hierarchical multi-agent framework capable of self-evolution through past experience. By hierarchical, we mean an explicit separation of high-level planning and low-level action execution. The framework comprises a Manager, responsible for devising overall plans by breaking down complex tasks into subgoals, and four subordinate agents--Perceptor, Operator, Action Reflector, and Notetaker--which handle fine-grained visual perception, immediate action execution, error verification, and information aggregation, respectively. Mobile-Agent-E also features a novel self-evolution module which maintains a persistent long-term memory comprising Tips and Shortcuts. Tips are general guidance and lessons learned from prior tasks on how to effectively interact with the environment. Shortcuts are reusable, executable sequences of atomic operations tailored for specific subroutines. The inclusion of Tips and Shortcuts facilitates continuous refinement in performance and efficiency. Alongside this framework, we introduce Mobile-Eval-E, a new benchmark featuring complex mobile tasks requiring long-horizon, multi-app interactions. Empirical results show that Mobile-Agent-E achieves a 22% absolute improvement over previous state-of-the-art approaches across three foundation model backbones. Project page: this https URL. 

**Abstract (ZH)**: 智能手机已成为现代生活不可或缺的工具，但在移动设备上导航复杂任务往往仍然令人沮丧。基于大规模多模态模型（LMM）的移动代理最近在感知和执行移动环境方面展现了能力。然而，当前的方法在处理现实世界的人类需求、应对推理密集型和长期任务方面存在显著局限性，缺乏从先前经验中学习和改进的机制。为了克服这些挑战，我们提出了Mobile-Agent-E，一种能够通过以往经验进行自我进化的分层多代理框架。所谓分层，指的是明确区分高层级规划和低层级动作执行。该框架包括一个经理，负责将复杂任务分解为子目标以制定总体计划；以及四个下属代理——感知器、操作员、动作反思器和记录员，分别负责精细的视觉感知、立即的动作执行、错误验证和信息聚合。Mobile-Agent-E 还配备了一个新颖的自我进化模块，该模块维护了一个持久的长期记忆，包括提示和捷径。提示是关于如何有效与环境互动的一般指导和从先前任务中吸取的教训。捷径是为特定子例行程序量身定制的可重复使用、可执行的原子操作序列。提示和捷径的包含促进了持续的性能和效率改进。除此之外，我们还引入了Mobile-Eval-E，这是一个新的基准测试，包括需要长期交互和多应用互动的复杂移动任务。实验结果表明，Mobile-Agent-E 在三个基础模型框架上实现了比之前最先进的方法绝对改进22%。项目页面：请点击此处。 

---
# Conversation Routines: A Prompt Engineering Framework for Task-Oriented Dialog Systems 

**Title (ZH)**: 对话惯例：面向任务导向的对话系统的一种提示工程框架 

**Authors**: Giorgio Robino  

**Link**: [PDF](https://arxiv.org/pdf/2501.11613)  

**Abstract**: This study introduces Conversation Routines (CR), a structured prompt engineering framework for developing task-oriented dialog systems using Large Language Models (LLMs). While LLMs demonstrate remarkable natural language understanding capabilities, engineering them to reliably execute complex business workflows remains challenging. The proposed CR framework enables the development of Conversation Agentic Systems (CAS) through natural language specifications, embedding task-oriented logic within LLM prompts. This approach provides a systematic methodology for designing and implementing complex conversational workflows while maintaining behavioral consistency. We demonstrate the framework's effectiveness through two proof of concept implementations: a Train Ticket Booking System and an Interactive Troubleshooting Copilot. These case studies validate CR's capability to encode sophisticated behavioral patterns and decision logic while preserving natural conversational flexibility. Results show that CR enables domain experts to design conversational workflows in natural language while leveraging custom enterprise functionalities (tools) developed by software engineers, creating an efficient division of responsibilities where developers focus on core API implementation and domain experts handle conversation design. While the framework shows promise in accessibility and adaptability, we identify key challenges including computational overhead, non-deterministic behavior, and domain-specific logic optimization. Future research directions include enhancing system robustness, improving scalability for complex multi-agent interactions, and addressing the identified limitations across diverse business applications. 

**Abstract (ZH)**: 本研究介绍了对话例行程序（CR），这是一个结构化的提示工程框架，用于使用大规模语言模型（LLMs）开发面向任务的对话系统。尽管LLMs展现出卓越的自然语言理解能力，但将它们工程化以可靠地执行复杂的业务工作流仍然具有挑战性。所提出的CR框架通过自然语言规范使开发者能够构建对话代理系统（CAS），并将任务导向的逻辑嵌入到LLM提示中。这种方法提供了一种系统的方法来设计和实现复杂的对话工作流，同时保持行为一致性。我们通过两个概念验证实现展示了该框架的有效性：一个火车票预订系统和一个交互式故障排除副驾。这些案例研究验证了CR能够编码复杂的行为模式和决策逻辑，同时保持自然对话的灵活性。结果显示，CR使领域专家能够使用自然语言设计对话工作流，同时利用软件工程师开发的定制企业功能（工具），从而形成一种高效的职责分工，其中开发人员专注于核心API的实现，而领域专家则负责对话设计。尽管该框架在易用性和适应性方面显示出前景，但我们仍识别出一些关键挑战，包括计算开销、非确定性行为以及特定领域的逻辑优化。未来的研究方向包括增强系统的稳健性、改进多代理交互的可扩展性，并解决在不同商业应用场景中识别出的限制。 

---
# IntellAgent: A Multi-Agent Framework for Evaluating Conversational AI Systems 

**Title (ZH)**: IntellAgent：多智能体系统框架评估对话式人工智能系统 

**Authors**: Elad Levi, Ilan Kadar  

**Link**: [PDF](https://arxiv.org/pdf/2501.11067)  

**Abstract**: Large Language Models (LLMs) are transforming artificial intelligence, evolving into task-oriented systems capable of autonomous planning and execution. One of the primary applications of LLMs is conversational AI systems, which must navigate multi-turn dialogues, integrate domain-specific APIs, and adhere to strict policy constraints. However, evaluating these agents remains a significant challenge, as traditional methods fail to capture the complexity and variability of real-world interactions. We introduce IntellAgent, a scalable, open-source multi-agent framework designed to evaluate conversational AI systems comprehensively. IntellAgent automates the creation of diverse, synthetic benchmarks by combining policy-driven graph modeling, realistic event generation, and interactive user-agent simulations. This innovative approach provides fine-grained diagnostics, addressing the limitations of static and manually curated benchmarks with coarse-grained metrics. IntellAgent represents a paradigm shift in evaluating conversational AI. By simulating realistic, multi-policy scenarios across varying levels of complexity, IntellAgent captures the nuanced interplay of agent capabilities and policy constraints. Unlike traditional methods, it employs a graph-based policy model to represent relationships, likelihoods, and complexities of policy interactions, enabling highly detailed diagnostics. IntellAgent also identifies critical performance gaps, offering actionable insights for targeted optimization. Its modular, open-source design supports seamless integration of new domains, policies, and APIs, fostering reproducibility and community collaboration. Our findings demonstrate that IntellAgent serves as an effective framework for advancing conversational AI by addressing challenges in bridging research and deployment. The framework is available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）正在改变人工智能领域，进化成为能够自主规划和执行任务的系统。LLMs 的主要应用之一是对话型人工智能系统，这些系统必须导航多轮对话、整合特定领域的API，并遵守严格的政策限制。然而，评估这些代理仍然是一个重大挑战，因为传统方法无法捕捉真实世界互动的复杂性和多样性。我们提出了IntellAgent，这是一个可扩展的开源多代理框架，旨在全面评估对话型人工智能系统。IntellAgent通过结合基于策略的图建模、真实的事件生成和交互式用户-代理模拟，自动化创建多种合成基准。这种方法提供细粒度的诊断，解决了静态和人工策画基准的粗粒度度量所存在的局限性。IntellAgent代表了评估对话型人工智能的一个范式转变。通过模拟不同复杂程度的真实多策略场景，IntellAgent捕捉了代理能力和政策约束之间的微妙交互。与传统方法不同，它使用基于图的策略模型来表示策略之间的关系、可能性及其复杂性，从而实现高度详细的诊断。IntellAgent还识别出关键的性能差距，提供了针对目标优化的可操作见解。其模块化和开源设计支持新领域、策略和API的无缝集成，促进可再现性和社区协作。我们的研究结果表明，IntellAgent是一个有效的框架，有助于通过解决研究与部署之间的鸿沟来推进对话型人工智能的发展。该框架可以在以下网址获取：[此链接] 

---
# UI-TARS: Pioneering Automated GUI Interaction with Native Agents 

**Title (ZH)**: UI-TARS：首创使用本机代理进行自动化GUI交互 

**Authors**: Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12326)  

**Abstract**: This paper introduces UI-TARS, a native GUI agent model that solely perceives the screenshots as input and performs human-like interactions (e.g., keyboard and mouse operations). Unlike prevailing agent frameworks that depend on heavily wrapped commercial models (e.g., GPT-4o) with expert-crafted prompts and workflows, UI-TARS is an end-to-end model that outperforms these sophisticated frameworks. Experiments demonstrate its superior performance: UI-TARS achieves SOTA performance in 10+ GUI agent benchmarks evaluating perception, grounding, and GUI task execution. Notably, in the OSWorld benchmark, UI-TARS achieves scores of 24.6 with 50 steps and 22.7 with 15 steps, outperforming Claude (22.0 and 14.9 respectively). In AndroidWorld, UI-TARS achieves 46.6, surpassing GPT-4o (34.5). UI-TARS incorporates several key innovations: (1) Enhanced Perception: leveraging a large-scale dataset of GUI screenshots for context-aware understanding of UI elements and precise captioning; (2) Unified Action Modeling, which standardizes actions into a unified space across platforms and achieves precise grounding and interaction through large-scale action traces; (3) System-2 Reasoning, which incorporates deliberate reasoning into multi-step decision making, involving multiple reasoning patterns such as task decomposition, reflection thinking, milestone recognition, etc. (4) Iterative Training with Reflective Online Traces, which addresses the data bottleneck by automatically collecting, filtering, and reflectively refining new interaction traces on hundreds of virtual machines. Through iterative training and reflection tuning, UI-TARS continuously learns from its mistakes and adapts to unforeseen situations with minimal human intervention. We also analyze the evolution path of GUI agents to guide the further development of this domain. 

**Abstract (ZH)**: 本文介绍了UI-TARS，这是一种原生的GUI代理模型，仅通过感知截屏作为输入，并执行类似人类的操作（如键盘和鼠标操作）。与依赖于高度封装的商业模型（例如GPT-4o）并需采用专家设计的提示和工作流的现有代理框架不同，UI-TARS是一个端到端的模型，性能超越了这些复杂的框架。实验结果表明其优越的性能：UI-TARS在10多个评价感知、定位和GUI任务执行的代理基准测试中均取得了SOTA（最佳）表现。特别地，在OSWorld基准测试中，UI-TARS在50步情况下的得分为24.6，15步情况下的得分为22.7，超越了Claude（分别为22.0和14.9）。在AndroidWorld中，UI-TARS的得分为46.6，超越了GPT-4o（34.5）。UI-TARS集成了多项关键创新：(1) 强化感知：利用大规模的GUI截屏数据集，实现基于上下文理解UI元素和精确描述；(2) 统一动作建模，标准化了跨平台的动作，通过大规模的动作痕迹实现精准的定位和交互；(3) 系统2推理，将有意识的推理融入多步骤决策中，包括任务分解、反思思考、里程碑识别等多种推理模式；(4) 循环训练与反思调整，通过自动收集、筛选和反思性优化新的交互痕迹，在数百个虚拟机上连续学习和适应，减少了人类干预。我们还分析了GUI代理的发展路径，以指导这一领域未来的进一步发展。 

---
# EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents 

**Title (ZH)**: EmbodiedEval：评估多模态大语言模型作为具身代理 

**Authors**: Zhili Cheng, Yuge Tu, Ran Li, Shiqi Dai, Jinyi Hu, Shengding Hu, Jiahao Li, Yang Shi, Tianyu Yu, Weize Chen, Lei Shi, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.11858)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown significant advancements, providing a promising future for embodied agents. Existing benchmarks for evaluating MLLMs primarily utilize static images or videos, limiting assessments to non-interactive scenarios. Meanwhile, existing embodied AI benchmarks are task-specific and not diverse enough, which do not adequately evaluate the embodied capabilities of MLLMs. To address this, we propose EmbodiedEval, a comprehensive and interactive evaluation benchmark for MLLMs with embodied tasks. EmbodiedEval features 328 distinct tasks within 125 varied 3D scenes, each of which is rigorously selected and annotated. It covers a broad spectrum of existing embodied AI tasks with significantly enhanced diversity, all within a unified simulation and evaluation framework tailored for MLLMs. The tasks are organized into five categories: navigation, object interaction, social interaction, attribute question answering, and spatial question answering to assess different capabilities of the agents. We evaluated the state-of-the-art MLLMs on EmbodiedEval and found that they have a significant shortfall compared to human level on embodied tasks. Our analysis demonstrates the limitations of existing MLLMs in embodied capabilities, providing insights for their future development. We open-source all evaluation data and simulation framework at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在实现重要进展的同时，为 embodied 代理带来了光明的未来。现有的 MLLM 评估基准主要依赖静态图像或视频，这限制了评估范围仅限于非交互式场景。同时，现有的 embodied AI 基准多为特定任务，缺乏多样性，不足以评估 MLLMs 的 embodied 能力。为解决这一问题，我们提出了一种名为 EmbodiedEval 的全面且交互式的评估基准，专为 MLLMs 设计，涵盖 embodied 任务。EmbodiedEval 包含 125 个不同场景中的 328 个独立任务，每个场景都经过严格选择和标注。它涵盖了现有的多种 embodied AI 任务，具备显著增强的多样性，并在为 MLLMs 设计的统一仿真和评估框架中进行了整合。任务被分类为五个类别：导航、对象交互、社会交互、属性问题回答和空间问题回答，以评估代理的不同能力。我们对当前最先进的 MLLMs 进行了评估，并发现它们在 embodied 任务上与人类水平相比存在显著差距。我们的分析揭示了现有 MLLMs 在 embodied 能力方面的局限性，为它们的未来发展方向提供了洞察。我们在以下网址开源了所有评估数据和仿真框架：[提供链接]。 

---
# ChaosEater: Fully Automating Chaos Engineering with Large Language Models 

**Title (ZH)**: 混沌吞噬者：利用大规模语言模型完全自动化混沌工程 

**Authors**: Daisuke Kikuta, Hiroki Ikeuchi, Kengo Tajiri, Yuusuke Nakano  

**Link**: [PDF](https://arxiv.org/pdf/2501.11107)  

**Abstract**: Chaos Engineering (CE) is an engineering technique aimed at improving the resiliency of distributed systems. It involves artificially injecting specific failures into a distributed system and observing its behavior in response. Based on the observation, the system can be proactively improved to handle those failures. Recent CE tools realize the automated execution of predefined CE experiments. However, defining these experiments and reconfiguring the system after the experiments still remain manual. To reduce the costs of the manual operations, we propose \textsc{ChaosEater}, a \textit{system} for automating the entire CE operations with Large Language Models (LLMs). It pre-defines the general flow according to the systematic CE cycle and assigns subdivided operations within the flow to LLMs. We assume systems based on Infrastructure as Code (IaC), wherein the system configurations and artificial failures are managed through code. Hence, the LLMs' operations in our \textit{system} correspond to software engineering tasks, including requirement definition, code generation and debugging, and testing. We validate our \textit{system} through case studies on both small and large systems. The results demonstrate that our \textit{system} significantly reduces both time and monetary costs while completing reasonable single CE cycles. 

**Abstract (ZH)**: 混沌工程（Chaos Engineering，CE）是一种旨在提高分布式系统弹性的工程技术。它通过在分布式系统中人为注入特定故障并观察其响应行为，从而能够基于这些观察对系统进行主动改进，使其能够处理这些故障。近年来，CE工具实现了预定义CE实验的自动化执行。然而，定义这些实验和实验后重新配置系统仍然需要手动操作。为减少这些手动操作的成本，我们提出了“ChaosEater”系统，该系统利用大规模语言模型（LLMs）自动化整个CE操作。它根据系统的CE周期定义了一般流程，并将流程中的细分操作分配给LLMs。我们假设基于基础设施即代码（IaC）的系统，其中系统配置和人工故障通过代码进行管理。因此，我们系统中的LLM操作对应于软件工程任务，包括需求定义、代码生成、调试和测试。我们通过针对小系统和大系统的案例研究验证了该系统。结果表明，该系统可以显著减少时间和货币成本，同时完成合理的单个CE周期。 

---
# Improved IR-based Bug Localization with Intelligent Relevance Feedback 

**Title (ZH)**: 基于改进的IR的智能相关反馈软件缺陷定位方法 

**Authors**: Asif Mohammed Samir, Mohammad Masudur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2501.10542)  

**Abstract**: Software bugs pose a significant challenge during development and maintenance, and practitioners spend nearly 50% of their time dealing with bugs. Many existing techniques adopt Information Retrieval (IR) to localize a reported bug using textual and semantic relevance between bug reports and source code. However, they often struggle to bridge a critical gap between bug reports and code that requires in-depth contextual understanding, which goes beyond textual or semantic relevance. In this paper, we present a novel technique for bug localization - BRaIn - that addresses the contextual gaps by assessing the relevance between bug reports and code with Large Language Models (LLM). It then leverages the LLM's feedback (a.k.a., Intelligent Relevance Feedback) to reformulate queries and re-rank source documents, improving bug localization. We evaluate BRaIn using a benchmark dataset, Bench4BL, and three performance metrics and compare it against six baseline techniques from the literature. Our experimental results show that BRaIn outperforms baselines by 87.6%, 89.5%, and 48.8% margins in MAP, MRR, and HIT@K, respectively. Additionally, it can localize approximately 52% of bugs that cannot be localized by the baseline techniques due to the poor quality of corresponding bug reports. By addressing the contextual gaps and introducing Intelligent Relevance Feedback, BRaIn advances not only theory but also improves IR-based bug localization. 

**Abstract (ZH)**: 软件错误在开发和维护过程中构成了重大挑战，从业者花费近一半的时间来处理这些错误。现有许多技术采用信息检索（IR）方法，通过错误报告与源代码之间的文本和语义相关性来定位错误报告。然而，这些方法往往难以弥合错误报告与代码之间的重要差距，这种差距要求深入了解上下文，而不仅仅是文本或语义相关性。本文提出了一种新的错误定位技术——BRaIn，该技术通过利用大型语言模型（LLM）评估错误报告与代码之间的相关性来解决这些上下文差距。在此基础上，利用LLM的反馈（即智能相关性反馈）来重新构建查询并重新排名源文档，从而提高错误定位的准确性。我们使用基准数据集Bench4BL和三种性能指标评估了BRaIn，并将其与文献中的六种基线技术进行了比较。实验结果表明，BRaIn在MAP、MRR和HIT@K指标上的表现分别优于基线技术87.6%、89.5%和48.8%。此外，它还能定位大约52%的由于错误报告质量不佳而无法被基线技术定位的错误。通过解决上下文差距并引入智能相关性反馈，BRaIn从理论上和实际应用上都推动了基于IR的错误定位技术的发展。 

---
# Beyond the Sum: Unlocking AI Agents Potential Through Market Forces 

**Title (ZH)**: 超越总和：通过市场力量释放人工智能代理的潜力 

**Authors**: Jordi Montes Sanabria, Pol Alvarez Vecino  

**Link**: [PDF](https://arxiv.org/pdf/2501.10388)  

**Abstract**: The emergence of Large Language Models has fundamentally transformed the capabilities of AI agents, enabling a new class of autonomous agents capable of interacting with their environment through dynamic code generation and execution. These agents possess the theoretical capacity to operate as independent economic actors within digital markets, offering unprecedented potential for value creation through their distinct advantages in operational continuity, perfect replication, and distributed learning capabilities. However, contemporary digital infrastructure, architected primarily for human interaction, presents significant barriers to their participation.
This work presents a systematic analysis of the infrastructure requirements necessary for AI agents to function as autonomous participants in digital markets. We examine four key areas - identity and authorization, service discovery, interfaces, and payment systems - to show how existing infrastructure actively impedes agent participation. We argue that addressing these infrastructure challenges represents more than a technical imperative; it constitutes a fundamental step toward enabling new forms of economic organization. Much as traditional markets enable human intelligence to coordinate complex activities beyond individual capability, markets incorporating AI agents could dramatically enhance economic efficiency through continuous operation, perfect information sharing, and rapid adaptation to changing conditions. The infrastructure challenges identified in this work represent key barriers to realizing this potential. 

**Abstract (ZH)**: 大型语言模型的出现从根本上改变了人工智能代理的能力，使一种新的自动化代理能够通过动态代码生成和执行与环境进行互动。这些代理具备理论上的能力，在数字市场中独立运作，并通过操作连续性、完美复制和分布式学习能力创造前所未有的价值。然而，主要为人类互动设计的当前数字基础设施对它们的参与构成了重大障碍。

本文对确保人工智能代理能够在数字市场中作为自主参与者运作所需的基础设施要求进行了系统分析。我们探讨了四个关键领域——身份与授权、服务发现、接口和服务支付系统，以展示现有基础设施如何积极阻碍代理的参与。我们认为，应对这些基础设施挑战不仅仅是技术上的必要性，而是朝着促进新型经济组织的一种基本步骤。正如传统市场使人类智能能够协调超出个人能力的复杂活动一样，包含人工智能代理的市场可以通过连续运营、完美信息共享和快速适应变化条件来显著提升经济效率。本文中识别出的基础设施挑战是实现这一潜力的关键障碍。 

---
# Episodic memory in AI agents poses risks that should be studied and mitigated 

**Title (ZH)**: AI代理的事件记忆存在风险，应当加以研究和缓解 

**Authors**: Chad DeChant  

**Link**: [PDF](https://arxiv.org/pdf/2501.11739)  

**Abstract**: Most current AI models have little ability to store and later retrieve a record or representation of what they do. In human cognition, episodic memories play an important role in both recall of the past as well as planning for the future. The ability to form and use episodic memories would similarly enable a broad range of improved capabilities in an AI agent that interacts with and takes actions in the world. Researchers have begun directing more attention to developing memory abilities in AI models. It is therefore likely that models with such capability will be become widespread in the near future. This could in some ways contribute to making such AI agents safer by enabling users to better monitor, understand, and control their actions. However, as a new capability with wide applications, we argue that it will also introduce significant new risks that researchers should begin to study and address. We outline these risks and benefits and propose four principles to guide the development of episodic memory capabilities so that these will enhance, rather than undermine, the effort to keep AI safe and trustworthy. 

**Abstract (ZH)**: 当前大多数人工智能模型缺乏存储和后续检索其行为记录或表示的能力。在人类认知中，情景记忆在回忆过去和规划未来方面扮演着重要角色。能够形成和利用情景记忆将在世界中与环境互动并采取行动的人工智能代理的多种能力得到显著提升。研究人员已经开始更多地关注提高人工智能模型的记忆能力。因此，具备此类能力的模型在未来很可能会变得普遍。这在某种程度上可以通过使用户更好地监控、理解和控制其行为来提高人工智能代理的安全性。然而，作为一种具有广泛应用的新能力，我们认为它也将引入显著的新风险，需要研究人员开始研究并加以解决。我们概述了这些风险与利益，并建议四条原则来指导情景记忆能力的发展，以确保这些能力能够增强而非削弱保持人工智能安全和可信赖的努力。 

---
# Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training 

**Title (ZH)**: Agent-R：通过迭代自我训练来进行反思的语言模型代理 

**Authors**: Siyu Yuan, Zehui Chen, Zhiheng Xi, Junjie Ye, Zhengyin Du, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.11425)  

**Abstract**: Large Language Models (LLMs) agents are increasingly pivotal for addressing complex tasks in interactive environments. Existing work mainly focuses on enhancing performance through behavior cloning from stronger experts, yet such approaches often falter in real-world applications, mainly due to the inability to recover from errors. However, step-level critique data is difficult and expensive to collect. Automating and dynamically constructing self-critique datasets is thus crucial to empowering models with intelligent agent capabilities. In this work, we propose an iterative self-training framework, Agent-R, that enables language Agent to Reflect on the fly. Unlike traditional methods that reward or penalize actions based on correctness, Agent-R leverages MCTS to construct training data that recover correct trajectories from erroneous ones. A key challenge of agent reflection lies in the necessity for timely revision rather than waiting until the end of a rollout. To address this, we introduce a model-guided critique construction mechanism: the actor model identifies the first error step (within its current capability) in a failed trajectory. Starting from it, we splice it with the adjacent correct path, which shares the same parent node in the tree. This strategy enables the model to learn reflection based on its current policy, therefore yielding better learning efficiency. To further explore the scalability of this self-improvement paradigm, we investigate iterative refinement of both error correction capabilities and dataset construction. Our findings demonstrate that Agent-R continuously improves the model's ability to recover from errors and enables timely error correction. Experiments on three interactive environments show that Agent-R effectively equips agents to correct erroneous actions while avoiding loops, achieving superior performance compared to baseline methods (+5.59%). 

**Abstract (ZH)**: 大型语言模型（LLMs）代理在处理交互环境中的复杂任务方面变得越来越关键。现有工作主要集中在通过从更强的专家那里进行行为克隆来提升性能，但这种做法在实际应用中往往会因为无法从错误中恢复而失效。然而，步骤级的批判数据收集起来既困难又昂贵。因此，自动化和动态构建自我批判数据集对于赋予模型智能代理能力至关重要。在本工作中，我们提出了一种迭代自我训练框架Agent-R，使得语言代理能够实时反思。与传统的基于正确性奖励或惩罚动作的方法不同，Agent-R 利用 Monte Carlo 树搜索（MCTS）来构建能够从错误轨迹中恢复正确轨迹的训练数据。代理反思的一个关键挑战是对及时修订的需求，而不仅仅是在一场展开（rollout）结束时才修订。为了解决这一问题，我们引入了一种基于模型的批判构建机制：行为模型识别失败轨迹中它目前能力范围内的第一个错误步骤。从这个步骤开始，将它与具有相同父节点的相邻正确路径拼接起来。这种策略使模型能够在当前策略的基础上学习反思，从而提高学习效率。为了进一步探索这种自我改进范式的可扩展性，我们探讨了错误纠正能力和数据集构建的迭代细化。我们的研究结果表明，Agent-R 不断提高模型从错误中恢复的能力，并实现及时的错误修正。在三个交互环境中进行的实验表明，与基线方法相比，Agent-R 有效地使代理能够纠正错误行为，同时避免循环，从而实现了更优的性能（+5.59%）。 

---
# ColorGrid: A Multi-Agent Non-Stationary Environment for Goal Inference and Assistance 

**Title (ZH)**: ColorGrid：一个多 agent 非稳态环境，用于目标推断与辅助 

**Authors**: Andrey Risukhin, Kavel Rao, Ben Caffee, Alan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2501.10593)  

**Abstract**: Autonomous agents' interactions with humans are increasingly focused on adapting to their changing preferences in order to improve assistance in real-world tasks. Effective agents must learn to accurately infer human goals, which are often hidden, to collaborate well. However, existing Multi-Agent Reinforcement Learning (MARL) environments lack the necessary attributes required to rigorously evaluate these agents' learning capabilities. To this end, we introduce ColorGrid, a novel MARL environment with customizable non-stationarity, asymmetry, and reward structure. We investigate the performance of Independent Proximal Policy Optimization (IPPO), a state-of-the-art (SOTA) MARL algorithm, in ColorGrid and find through extensive ablations that, particularly with simultaneous non-stationary and asymmetric goals between a ``leader'' agent representing a human and a ``follower'' assistant agent, ColorGrid is unsolved by IPPO. To support benchmarking future MARL algorithms, we release our environment code, model checkpoints, and trajectory visualizations at this https URL. 

**Abstract (ZH)**: 自主代理与人类的交互越来越多地集中在适应人类不断变化的偏好上，以提高在实际任务中的辅助效果。有效的代理必须学会准确地推断出往往被隐藏的人类目标，从而更好地协作。然而，现有的多智能体强化学习（MARL）环境缺乏评估这些代理学习能力所需的关键属性。为此，我们引入了ColorGrid，这是一种具有可定制非稳定性和不对称性的新型MARL环境，并且具有可定制的奖励结构。我们研究了当前最先进的MARL算法——独立接近策略优化（IPPO）在ColorGrid中的性能，并通过广泛的消融实验发现，特别是在“领导者”代理代表人类和“跟随者”助手代理之间的同时非稳定性和不对称性目标下，IPPO无法解决ColorGrid环境。为了支持未来MARL算法的基准测试，我们在以下链接中开放我们的环境代码、模型检查点和轨迹可视化：[提供链接处] 

---
# Federated Deep Reinforcement Learning for Energy Efficient Multi-Functional RIS-Assisted Low-Earth Orbit Networks 

**Title (ZH)**: 联邦深度强化学习在辅助低地球轨道网络中的多功能可控反射表面（RIS）部署以提高能源效率 

**Authors**: Li-Hsiang Shen, Jyun-Jhe Huang, Kai-Ten Feng, Lie-Liang Yang, Jen-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11079)  

**Abstract**: In this paper, a novel network architecture that deploys the multi-functional reconfigurable intelligent surface (MF-RIS) in low-Earth orbit (LEO) is proposed. Unlike traditional RIS with only signal reflection capability, the MF-RIS can reflect, refract, and amplify signals, as well as harvest energy from wireless signals. Given the high energy demands in shadow regions where solar energy is unavailable, MF-RIS is deployed in LEO to enhance signal coverage and improve energy efficiency (EE). To address this, we formulate a long-term EE optimization problem by determining the optimal parameters for MF-RIS configurations, including amplification and phase-shifts, energy harvesting ratios, and LEO transmit beamforming. To address the complex non-convex and non-linear problem, a federated learning enhanced multi-agent deep deterministic policy gradient (FEMAD) scheme is designed. Multi-agent DDPG of each agent can provide the optimal action policy from its interaction to environments, whereas federated learning enables the hidden information exchange among multi-agents. In numerical results, we can observe significant EE improvements compared to the other benchmarks, including centralized deep reinforcement learning as well as distributed multi-agent deep deterministic policy gradient (DDPG). Additionally, the proposed LEO-MF-RIS architecture has demonstrated its effectiveness, achieving the highest EE performance compared to the scenarios of fixed/no energy harvesting in MF-RIS, traditional reflection-only RIS, and deployment without RISs/MF-RISs. 

**Abstract (ZH)**: 本文提出了一种新颖的网络架构，该架构在低地球轨道（LEO）中部署了多功能可重构智能表面（MF-RIS）。不同于传统仅具有信号反射能力的RIS，MF-RIS能够进行信号反射、折射、放大，并能够从无线信号中获取能量。由于在阴影区域太阳能不可用导致的高能耗需求，MF-RIS被部署在LEO中以增强信号覆盖并提高能量效率（EE）。为此，通过确定MF-RIS配置的最佳参数（包括放大和相位调制、能量采集比率以及LEO传输波束成形），我们提出了长期EE优化问题。为解决这一复杂的非凸性和非线性问题，设计了一种联邦学习增强的多智能体深度确定性策略梯度（FEMAD）方案。每个智能体的多智能体DDPG可以提供其与环境交互后的最优动作策略，而联邦学习则允许多智能体之间隐藏信息的交换。在数值结果中，我们观察到相比于集中式深度强化学习以及分布式多智能体深度确定性策略梯度（DDPG）等基准方法，EE有了显著提高。另外，提出的LEO-MF-RIS架构已显示出其有效性，在固定/无能量采集的MF-RIS场景、仅反射的RIS传统场景以及不部署RIS/MF-RIS的场景中，其实现了最高的EE性能。 

---
# Blockchain-assisted Demonstration Cloning for Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: 基于区块链辅助的多智能体深度强化学习示范克隆 

**Authors**: Ahmed Alagha, Jamal Bentahar, Hadi Otrok, Shakti Singh, Rabeb Mizouni  

**Link**: [PDF](https://arxiv.org/pdf/2501.10938)  

**Abstract**: Multi-Agent Deep Reinforcement Learning (MDRL) is a promising research area in which agents learn complex behaviors in cooperative or competitive environments. However, MDRL comes with several challenges that hinder its usability, including sample efficiency, curse of dimensionality, and environment exploration. Recent works proposing Federated Reinforcement Learning (FRL) to tackle these issues suffer from problems related to model restrictions and maliciousness. Other proposals using reward shaping require considerable engineering and could lead to local optima. In this paper, we propose a novel Blockchain-assisted Multi-Expert Demonstration Cloning (MEDC) framework for MDRL. The proposed method utilizes expert demonstrations in guiding the learning of new MDRL agents, by suggesting exploration actions in the environment. A model sharing framework on Blockchain is designed to allow users to share their trained models, which can be allocated as expert models to requesting users to aid in training MDRL systems. A Consortium Blockchain is adopted to enable traceable and autonomous execution without the need for a single trusted entity. Smart Contracts are designed to manage users and models allocation, which are shared using IPFS. The proposed framework is tested on several applications, and is benchmarked against existing methods in FRL, Reward Shaping, and Imitation Learning-assisted RL. The results show the outperformance of the proposed framework in terms of learning speed and resiliency to faulty and malicious models. 

**Abstract (ZH)**: 多智能体深度强化学习（MDRL）是研究中一个有前景的领域，在该领域中，智能体在合作或竞争环境中学习复杂行为。然而，MDRL 面临着几个挑战，这些挑战限制了其应用性，包括样本效率低、维数灾难以及环境探索。近年来，提出使用联邦强化学习（FRL）来解决这些问题的方法，但这些方法存在模型限制和恶意行为相关的问题。其他使用奖励重塑的方法则需要大量的工程工作，可能会导致局部最优解。在本文中，我们提出了一种名为区块链辅助多专家示范克隆（Blockchain-assisted Multi-Expert Demonstration Cloning, MEDC）框架的新型多智能体深度强化学习方法。所提出的方法利用专家示范引导新MDRL智能体的学习，并建议在环境中的探索动作。一个基于区块链的模型共享框架被设计出来，允许用户共享其训练模型，并将这些模型分配给请求方以协助训练MDRL系统。采用联盟区块链来实现可追溯且自治的执行，而无需单一可信实体的参与。智能合约被设计用于管理用户和模型分配，并通过IPFS共享。所提出的框架在多个应用场景中进行了测试，并与现有的FRL、奖励重塑和模仿学习辅助的RL方法进行了基准测试。结果表明，该框架在学习速度和对故障和恶意模型的鲁棒性方面优于现有方法。 

---
# Adaptive Target Localization under Uncertainty using Multi-Agent Deep Reinforcement Learning with Knowledge Transfer 

**Title (ZH)**: 基于知识转移的多代理深度强化学习在不确定性条件下的自适应目标定位 

**Authors**: Ahmed Alagha, Rabeb Mizouni, Shakti Singh, Jamal Bentahar, Hadi Otrok  

**Link**: [PDF](https://arxiv.org/pdf/2501.10924)  

**Abstract**: Target localization is a critical task in sensitive applications, where multiple sensing agents communicate and collaborate to identify the target location based on sensor readings. Existing approaches investigated the use of Multi-Agent Deep Reinforcement Learning (MADRL) to tackle target localization. Nevertheless, these methods do not consider practical uncertainties, like false alarms when the target does not exist or when it is unreachable due to environmental complexities. To address these drawbacks, this work proposes a novel MADRL-based method for target localization in uncertain environments. The proposed MADRL method employs Proximal Policy Optimization to optimize the decision-making of sensing agents, which is represented in the form of an actor-critic structure using Convolutional Neural Networks. The observations of the agents are designed in an optimized manner to capture essential information in the environment, and a team-based reward functions is proposed to produce cooperative agents. The MADRL method covers three action dimensionalities that control the agents' mobility to search the area for the target, detect its existence, and determine its reachability. Using the concept of Transfer Learning, a Deep Learning model builds on the knowledge from the MADRL model to accurately estimating the target location if it is unreachable, resulting in shared representations between the models for faster learning and lower computational complexity. Collectively, the final combined model is capable of searching for the target, determining its existence and reachability, and estimating its location accurately. The proposed method is tested using a radioactive target localization environment and benchmarked against existing methods, showing its efficacy. 

**Abstract (ZH)**: 目标定位是敏感应用中的关键任务，其中多个传感代理通过通信和协作，基于传感器读数来识别目标的位置。现有的方法已研究了使用多代理深度强化学习（MADRL）来解决目标定位问题。然而，这些方法并未考虑实际的不确定性，例如目标不存在时的误报警或因环境复杂性而导致目标不可达的情况。为了应对这些缺点，本工作提出了一种基于MADRL的方法，用于处理不确定环境中的目标定位。所提出的MADRL方法利用近端策略优化（PPO）来优化传感代理的决策制定，采用卷积神经网络（CNN）的形式表示为演员-评论家结构。代理的观测值被优化设计以捕获环境中的关键信息，并提出了一种基于团队的奖励函数以生成协同工作的代理。MADRL方法涵盖了三种行动维度，分别控制代理的移动性以搜索区域、检测目标的存在性以及确定其可达性。通过迁移学习的概念，深度学习模型基于MADRL模型的知识，能够在目标不可达时准确估计其位置，从而使得模型之间拥有共享表示，加快学习速度并降低计算复杂度。最后，最终组合模型能够搜索目标、确定其存在性与可达性，并准确估计其位置。所提出的方案在放射性目标定位环境中进行了测试，并与现有方法进行了基准测试，显示出其有效性。 

---
# Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments 

**Title (ZH)**: 基于数据的交互学习：一种适用于现实环境的自适应代理自适应框架 

**Authors**: Hongjin Su, Ruoxi Sun, Jinsung Yoon, Pengcheng Yin, Tao Yu, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2501.10893)  

**Abstract**: Autonomous agents powered by large language models (LLMs) have the potential to enhance human capabilities, assisting with digital tasks from sending emails to performing data analysis. The abilities of existing LLMs at such tasks are often hindered by the lack of high-quality agent data from the corresponding environments they interact with. We propose Learn-by-interact, a data-centric framework to adapt LLM agents to any given environments without human annotations. Learn-by-interact synthesizes trajectories of agent-environment interactions based on documentations, and constructs instructions by summarizing or abstracting the interaction histories, a process called backward construction. We assess the quality of our synthetic data by using them in both training-based scenarios and training-free in-context learning (ICL), where we craft innovative retrieval approaches optimized for agents. Extensive experiments on SWE-bench, WebArena, OSWorld and Spider2-V spanning across realistic coding, web, and desktop environments show the effectiveness of Learn-by-interact in various downstream agentic tasks -- baseline results are improved by up to 12.2\% for ICL with Claude-3.5 and 19.5\% for training with Codestral-22B. We further demonstrate the critical role of backward construction, which provides up to 14.0\% improvement for training. Our ablation studies demonstrate the efficiency provided by our synthesized data in ICL and the superiority of our retrieval pipeline over alternative approaches like conventional retrieval-augmented generation (RAG). We expect that Learn-by-interact will serve as a foundation for agent data synthesis as LLMs are increasingly deployed at real-world environments. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的自主代理有可能增强人类的能力，协助完成从发送电子邮件到进行数据分析等各种数字任务。现有的LLMs在这些任务中的能力往往受到与之交互的相应环境中的高质量代理数据缺乏的限制。我们提出了一个数据为中心的框架——“通过交互学习”，该框架能够在无需人工注释的情况下使LLM代理适应任何给定的环境。通过文档，“通过交互学习”综合了代理-环境交互的轨迹，并通过总结或抽象交互历史来构建指令，这一过程称为反向构造。我们通过使用合成数据在基于训练的场景和无需训练的上下文学习（ICL）中评估其质量，其中我们设计了针对代理的创新检索方法。在SWE-bench、WebArena、OSWorld和Spider2-V等涵盖现实编码、网络和桌面环境的广泛实验中，展示了“通过交互学习”的有效性，通过使用Codestral-22B训练时，基准结果提高了19.5%，使用Claude-3.5进行ICL时提高了12.2%。我们进一步证明了反向构造的关键作用，其能够提供高达14.0%的训练改进。我们的消融研究表明，我们合成数据在ICL中的效率以及我们检索流水线相较于传统检索增强生成（RAG）等替代方法的优势。我们期望“通过交互学习”将成为LLMs部署到真实环境中的代理数据合成的基础。 

---
# BAP v2: An Enhanced Task Framework for Instruction Following in Minecraft Dialogues 

**Title (ZH)**: BAP v2：一种增强的任务框架，用于Minecraft对话中的指令跟随 

**Authors**: Prashant Jayannavar, Liliang Ren, Marisa Hudspeth, Charlotte Lambert, Ariel Cordes, Elizabeth Kaplan, Anjali Narayan-Chen, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2501.10836)  

**Abstract**: Interactive agents capable of understanding and executing instructions in the physical world have long been a central goal in AI research. The Minecraft Collaborative Building Task (MCBT) provides one such setting to work towards this goal (Narayan-Chen, Jayannavar, and Hockenmaier 2019). It is a two-player game in which an Architect (A) instructs a Builder (B) to construct a target structure in a simulated Blocks World Environment. We focus on the challenging Builder Action Prediction (BAP) subtask of predicting correct action sequences in a given multimodal game context with limited training data (Jayannavar, Narayan-Chen, and Hockenmaier 2020). We take a closer look at evaluation and data for the BAP task, discovering key challenges and making significant improvements on both fronts to propose BAP v2, an upgraded version of the task. This will allow future work to make more efficient and meaningful progress on it. It comprises of: (1) an enhanced evaluation benchmark that includes a cleaner test set and fairer, more insightful metrics, and (2) additional synthetic training data generated from novel Minecraft dialogue and target structure simulators emulating the MCBT. We show that the synthetic data can be used to train more performant and robust neural models even with relatively simple training methods. Looking ahead, such data could also be crucial for training more sophisticated, data-hungry deep transformer models and training/fine-tuning increasingly large LLMs. Although modeling is not the primary focus of this work, we also illustrate the impact of our data and training methodologies on a simple LLM- and transformer-based model, thus validating the robustness of our approach, and setting the stage for more advanced architectures and LLMs going forward. 

**Abstract (ZH)**: 能够在物理世界中理解并执行指令的交互式代理一直是AI研究的中心目标之一。Minecraft协作建造任务（MCBT）提供了一个这样的环境，旨在向这个目标迈进（Narayan-Chen, Jayannavar, and Hockenmaier 2019）。这是一个两人游戏，在模拟的Blocks World环境中，建筑师（A）会指导建造者（B）建造一个目标结构。我们专注于“建造者动作预测”（BAP）子任务，该任务涉及根据有限的训练数据预测给定多模态游戏上下文中的正确动作序列（Jayannavar, Narayan-Chen, and Hockenmaier 2020）。我们更详细地审视了BAP任务的评估和数据，发现了关键挑战，并在两个方面取得了显著改进，提出了BAP v2，即任务的升级版本。这将使未来的工作能够更高效且有意义地推进该任务。具体包括：（1）改进的评估基准，包括更清洁的测试集和更公平、更深入的指标；（2）从模拟MCBT的新MC游戏对话和目标结构生成的额外合成训练数据。我们展示了合成数据即使使用相对简单的训练方法也能用于训练性能更优、更稳健的神经网络模型。展望未来，这样的数据对于训练更复杂的、数据需求更大的深度变换模型以及训练/微调越来越大的语言模型也可能至关重要。虽然建模不是本工作的主要焦点，但我们也展示了我们的数据和训练方法对简单基于语言模型和变换器模型的影响，从而验证了我们方法的稳健性，并为未来更高级架构和语言模型奠定了基础。 

---
# Simultaneous Computation with Multiple Prioritizations in Multi-Agent Motion Planning 

**Title (ZH)**: 多智能体运动规划中的多重优先级同时计算 

**Authors**: Patrick Scheffe, Julius Kahle, Bassam Alrifaee  

**Link**: [PDF](https://arxiv.org/pdf/2501.10781)  

**Abstract**: Multi-agent path finding (MAPF) in large networks is computationally challenging. An approach for MAPF is prioritized planning (PP), in which agents plan sequentially according to their priority. Albeit a computationally efficient approach for MAPF, the solution quality strongly depends on the prioritization. Most prioritizations rely either on heuristics, which do not generalize well, or iterate to find adequate priorities, which costs computational effort. In this work, we show how agents can compute with multiple prioritizations simultaneously. Our approach is general as it does not rely on domain-specific knowledge. The context of this work is multi-agent motion planning (MAMP) with a receding horizon subject to computation time constraints. MAMP considers the system dynamics in more detail compared to MAPF. In numerical experiments on MAMP, we demonstrate that our approach to prioritization comes close to optimal prioritization and outperforms state-of-the-art methods with only a minor increase in computation time. We show real-time capability in an experiment on a road network with ten vehicles in our Cyber-Physical Mobility Lab. 

**Abstract (ZH)**: 在大规模网络中，多智能体路径寻找（Multi-agent Path Finding, MAPF）具有计算上的挑战性。一种MAPF的方法是优先级规划（Prioritized Planning, PP），在该方法中，智能体根据其优先级顺序规划路径。尽管PP方法在计算上较为高效，但其解的质量很大程度上依赖于优先级的选择。大多数优先级选择要么依赖于启发式方法，这些方法在泛化方面表现不佳，要么需要迭代以找到合适的优先级，这会增加计算成本。在本研究中，我们展示了如何让智能体同时使用多种优先级进行计算。我们的方法是通用的，因为它不依赖于特定领域的知识。本研究的背景是在计算时间受限条件下考虑系统动力学的多智能体运动规划（Multi-agent Motion Planning, MAMP）。相较于MAPF，MAMP更详细地考虑了系统动力学。在针对MAMP的数值实验中，我们证明了我们的优先级规划方法接近最优优先级规划，并且仅略微增加了计算时间便优于现有的先进方法。我们还在我们的人机物理移动实验室（Cyber-Physical Mobility Lab）中进行了一项十辆车的道路网络实验，展示了实时操作能力。 

---
# Cooperative Search and Track of Rogue Drones using Multiagent Reinforcement Learning 

**Title (ZH)**: 使用多智能体强化学习进行恶意无人机的协同搜索与跟踪 

**Authors**: Panayiota Valianti, Kleanthis Malialis, Panayiotis Kolios, Georgios Ellinas  

**Link**: [PDF](https://arxiv.org/pdf/2501.10413)  

**Abstract**: This work considers the problem of intercepting rogue drones targeting sensitive critical infrastructure facilities. While current interception technologies focus mainly on the jamming/spoofing tasks, the challenges of effectively locating and tracking rogue drones have not received adequate attention. Solving this problem and integrating with recently proposed interception techniques will enable a holistic system that can reliably detect, track, and neutralize rogue drones. Specifically, this work considers a team of pursuer UAVs that can search, detect, and track multiple rogue drones over a sensitive facility. The joint search and track problem is addressed through a novel multiagent reinforcement learning scheme to optimize the agent mobility control actions that maximize the number of rogue drones detected and tracked. The performance of the proposed system is investigated under realistic settings through extensive simulation experiments with varying number of agents demonstrating both its performance and scalability. 

**Abstract (ZH)**: 本研究探讨了拦截针对敏感关键基础设施的恶意无人机的问题。尽管当前的拦截技术主要集中在干扰/欺骗任务上，但对于有效定位和跟踪恶意无人机的挑战尚未得到充分的关注。解决这些问题并将这些挑战与最近提出的拦截技术相结合，可以使系统能够可靠地检测、跟踪并中和恶意无人机。具体而言，本研究考虑了一组追捕无人飞行器（UAV），这些追捕无人飞行器可以在敏感设施区域内搜索、检测和跟踪多个恶意无人机。通过一个新颖的多智能体强化学习方案来解决联合搜索和跟踪问题，以优化智能体移动控制动作，最大化检测和跟踪到的恶意无人机的数量。通过广泛仿真实验，在不同智能体数量下的现实环境中考察所提系统的性能和可扩展性，以验证其有效性和可扩展性。 

---
# Towards General Purpose Robots at Scale: Lifelong Learning and Learning to Use Memory 

**Title (ZH)**: 面向大规模通用机器人：终身学习与利用记忆的学习 

**Authors**: William Yue  

**Link**: [PDF](https://arxiv.org/pdf/2501.10395)  

**Abstract**: The widespread success of artificial intelligence in fields like natural language processing and computer vision has not yet fully transferred to robotics, where progress is hindered by the lack of large-scale training data and the complexity of real-world tasks. To address this, many robot learning researchers are pushing to get robots deployed at scale in everyday unstructured environments like our homes to initiate a data flywheel. While current robot learning systems are effective for certain short-horizon tasks, they are not designed to autonomously operate over long time horizons in unstructured environments. This thesis focuses on addressing two key challenges for robots operating over long time horizons: memory and lifelong learning.
We propose two novel methods to advance these capabilities. First, we introduce t-DGR, a trajectory-based deep generative replay method that achieves state-of-the-art performance on Continual World benchmarks, advancing lifelong learning. Second, we develop a framework that leverages human demonstrations to teach agents effective memory utilization, improving learning efficiency and success rates on Memory Gym tasks. Finally, we discuss future directions for achieving the lifelong learning and memory capabilities necessary for robots to function at scale in real-world settings. 

**Abstract (ZH)**: 人工智能在自然语言处理和计算机视觉等领域取得的广泛成功尚未完全转移到机器人技术中，这主要是由于缺乏大规模训练数据以及真实世界任务的复杂性造成的。为了解决这一问题，许多机器人学习研究人员正努力将机器人大规模部署在像家庭这样未经结构化的日常环境中，以启动数据飞轮。当前的机器人学习系统在特定的短期任务中表现有效，但它们并不是为了在未经结构化的环境中自主长时间运行而设计的。本论文重点解决长时程运行中机器人面临的两个关键挑战：记忆和终身学习。

我们提出两种创新方法来推进这些能力。首先，我们引入了t-DGR（轨迹导向的深度生成重播方法），它在持续世界基准测试中达到了最先进的性能，促进了终身学习的发展。其次，我们开发了一个框架，利用人类示范来教会代理合理利用记忆，从而提高在记忆锻炼场任务中的学习效率和成功率。最后，我们讨论了实现机器人在真实世界中大规模运行所需的终身学习和记忆能力的未来方向。 

---
# Autonomous Microscopy Experiments through Large Language Model Agents 

**Title (ZH)**: 通过大规模语言模型代理实现自主显微镜实验 

**Authors**: Indrajeet Mandal, Jitendra Soni, Mohd Zaki, Morten M. Smedskjaer, Katrin Wondraczek, Lothar Wondraczek, Nitya Nand Gosvami, N. M. Anoop Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2501.10385)  

**Abstract**: The emergence of large language models (LLMs) has accelerated the development of self-driving laboratories (SDLs) for materials research. Despite their transformative potential, current SDL implementations rely on rigid, predefined protocols that limit their adaptability to dynamic experimental scenarios across different labs. A significant challenge persists in measuring how effectively AI agents can replicate the adaptive decision-making and experimental intuition of expert scientists. Here, we introduce AILA (Artificially Intelligent Lab Assistant), a framework that automates atomic force microscopy (AFM) through LLM-driven agents. Using AFM as an experimental testbed, we develop AFMBench-a comprehensive evaluation suite that challenges AI agents based on language models like GPT-4o and GPT-3.5 to perform tasks spanning the scientific workflow: from experimental design to results analysis. Our systematic assessment shows that state-of-the-art language models struggle even with basic tasks such as documentation retrieval, leading to a significant decline in performance in multi-agent coordination scenarios. Further, we observe that LLMs exhibit a tendency to not adhere to instructions or even divagate to additional tasks beyond the original request, raising serious concerns regarding safety alignment aspects of AI agents for SDLs. Finally, we demonstrate the application of AILA on increasingly complex experiments open-ended experiments: automated AFM calibration, high-resolution feature detection, and mechanical property measurement. Our findings emphasize the necessity for stringent benchmarking protocols before deploying AI agents as laboratory assistants across scientific disciplines. 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起加速了材料研究中自动实验室（SDLs）的发展。尽管它们具有变革性潜力，当前的SDL实现依赖于僵硬的预定义协议，限制了其在不同实验室动态实验场景中的适应性。一个重大挑战在于，如何有效衡量人工智能代理能否复制专家科学家的适应性决策和实验直觉。在这里，我们介绍了一种名为AILA（Artificially Intelligent Lab Assistant）的框架，该框架通过基于LLM的代理自动化原子力显微镜（AFM）。使用AFM作为实验测试平台，我们开发了AFMBench——一个全面的评估套件，基于如GPT-4o和GPT-3.5等语言模型挑战AI代理完成涵盖整个科学工作流程的任务：从实验设计到结果分析。我们的系统评估表明，最先进的语言模型即使在基本任务如文献检索上也表现不佳，这在多代理协调场景中导致了显著的性能下降。此外，我们观察到LLMs表现出不遵守指令或甚至转向超出原始请求的额外任务的趋势，这引起了关于SDL中人工智能代理的安全对齐方面的严重关切。最后，我们展示了AILA在日益复杂的实验中的应用：全自动AFM校准、高分辨率特征检测和机械性能测量。我们的研究结果强调了在将AI代理应用于不同科学领域之前进行严格基准测试的必要性。 

---
# GTDE: Grouped Training with Decentralized Execution for Multi-agent Actor-Critic 

**Title (ZH)**: GTDE: 分组训练与去中心化执行的多代理actor-critic方法 

**Authors**: Mengxian Li, Qi Wang, Yongjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.10367)  

**Abstract**: The rapid advancement of multi-agent reinforcement learning (MARL) has given rise to diverse training paradigms to learn the policies of each agent in the multi-agent system. The paradigms of decentralized training and execution (DTDE) and centralized training with decentralized execution (CTDE) have been proposed and widely applied. However, as the number of agents increases, the inherent limitations of these frameworks significantly degrade the performance metrics, such as win rate, total reward, etc. To reduce the influence of the increasing number of agents on the performance metrics, we propose a novel training paradigm of grouped training decentralized execution (GTDE). This framework eliminates the need for a centralized module and relies solely on local information, effectively meeting the training requirements of large-scale multi-agent systems. Specifically, we first introduce an adaptive grouping module, which divides each agent into different groups based on their observation history. To implement end-to-end training, GTDE uses Gumbel-Sigmoid for efficient point-to-point sampling on the grouping distribution while ensuring gradient backpropagation. To adapt to the uncertainty in the number of members in a group, two methods are used to implement a group information aggregation module that merges member information within the group. Empirical results show that in a cooperative environment with 495 agents, GTDE increased the total reward by an average of 382\% compared to the baseline. In a competitive environment with 64 agents, GTDE achieved a 100\% win rate against the baseline. 

**Abstract (ZH)**: 多智能体强化学习（MARL）的快速发展催生了多种训练范式，以学习每个智能体在多智能体系统中的策略。分散训练与执行（DTDE）和集中训练与分散执行（CTDE）等范式已被提出并广泛应用于实际问题中。然而，随着智能体数量的增加，这些框架内部固有的局限性显著降低了性能指标（如胜率、总奖励等）。为了减少智能体数量增加对性能指标的影响，我们提出了一种新的训练范式——分组训练分散执行（GTDE）。该框架去除了集中模块的必要性，仅依赖于局部信息，有效满足大规模多智能体系统的训练需求。具体来说，我们首先引入了一种自适应分组模块，该模块根据每个智能体的观测历史将其分为不同的组。为实现端到端训练，GTDE使用Gumbel-Sigmoid进行高效的点到点采样，同时确保梯度反传。为了适应组内成员数量的不确定性，我们采用两种方法实现一组信息聚合模块，该模块可以将组内成员信息进行合并。实验结果表明，在包含495个智能体的合作环境中，与基线相比，GTDE的总奖励平均提高了382%；在包含64个智能体的竞赛环境中，GTDE实现了100%的胜率，战胜了基线。 

---
