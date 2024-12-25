# Multi-Agent Norm Perception and Induction in Distributed Healthcare 

**Title (ZH)**: 多智能体规范感知与诱导在分布式医疗中的应用 

**Authors**: Chao Li, Olga Petruchik, Elizaveta Grishanina, Sergey Kovalchuk  

**Link**: [PDF](https://arxiv.org/pdf/2412.18454)  

**Abstract**: This paper presents a Multi-Agent Norm Perception and Induction Learning Model aimed at facilitating the integration of autonomous agent systems into distributed healthcare environments through dynamic interaction processes. The nature of the medical norm system and its sharing channels necessitates distinct approaches for Multi-Agent Systems to learn two types of norms. Building on this foundation, the model enables agents to simultaneously learn descriptive norms, which capture collective tendencies, and prescriptive norms, which dictate ideal behaviors. Through parameterized mixed probability density models and practice-enhanced Markov games, the multi-agent system perceives descriptive norms in dynamic interactions and captures emergent prescriptive norms. We conducted experiments using a dataset from a neurological medical center spanning from 2016 to 2020. 

**Abstract (ZH)**: 本文提出了一个面向分布式医疗环境自主代理系统整合的多Agent规范感知与归纳学习模型。医疗规范系统及其共享渠道的本性要求多Agent系统采用不同的方法来学习两类规范。在此基础上，该模型使智能体能够同时学习描述性规范（捕捉集体倾向）和指令性规范（规定理想行为）。通过参数化混合概率密度模型和实践增强的马尔可夫游戏，多Agent系统在动态交互过程中感知描述性规范并捕获新兴的指令性规范。我们使用2016年至2020年间神经医学中心的数据集进行了实验。 

---
# Explainable Multi-Modal Data Exploration in Natural Language via LLM Agent 

**Title (ZH)**: 通过大规模语言模型代理进行可解释的多模态数据语言探索 

**Authors**: Farhad Nooralahzadeh, Yi Zhang, Jonathan Furst, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.18428)  

**Abstract**: International enterprises, organizations, or hospitals collect large amounts of multi-modal data stored in databases, text documents, images, and videos. While there has been recent progress in the separate fields of multi-modal data exploration as well as in database systems that automatically translate natural language questions to database query languages, the research challenge of querying database systems combined with other unstructured modalities such as images in natural language is widely unexplored.
In this paper, we propose XMODE - a system that enables explainable, multi-modal data exploration in natural language. Our approach is based on the following research contributions: (1) Our system is inspired by a real-world use case that enables users to explore multi-modal information systems. (2) XMODE leverages a LLM-based agentic AI framework to decompose a natural language question into subtasks such as text-to-SQL generation and image analysis. (3) Experimental results on multi-modal datasets over relational data and images demonstrate that our system outperforms state-of-the-art multi-modal exploration systems, excelling not only in accuracy but also in various performance metrics such as query latency, API costs, planning efficiency, and explanation quality, thanks to the more effective utilization of the reasoning capabilities of LLMs. 

**Abstract (ZH)**: 国际企业、组织或医院收集了大量的多模态数据，这些数据存储在数据库、文本文件、图像和视频中。虽然在多模态数据探索和自动将自然语言问题转换为数据库查询语言的数据库系统方面已经取得了进展，但将数据库系统与其他未结构化的模态（如图像）结合起来使用自然语言进行查询的研究挑战仍未得到广泛探索。

本文提出了一种名为XMODE的系统，该系统能够使用自然语言进行可解释的多模态数据探索。我们的方法基于以下几个研究贡献：（1）我们的系统受到了实际应用场景的启发，使得用户能够探索多模态信息系统。（2）XMODE利用基于大模型的代理型AI框架，将自然语言问题分解为子任务，例如文本到SQL生成和图像分析。（3）在关系数据和图像的多模态数据集上的实验结果表明，我们的系统在准确性和查询延迟、API费用、规划效率和解释质量等多个性能指标上都明显优于现有的多模态探索系统，这得益于对大模型推理能力更有效的利用。 

---
# GUI Testing Arena: A Unified Benchmark for Advancing Autonomous GUI Testing Agent 

**Title (ZH)**: GUI测试竞技场：一种促进自主GUI测试代理发展的统一基准 

**Authors**: Kangjia Zhao, Jiahui Song, Leigang Sha, Haozhan Shen, Zhi Chen, Tiancheng Zhao, Xiubo Liang, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2412.18426)  

**Abstract**: Nowadays, research on GUI agents is a hot topic in the AI community. However, current research focuses on GUI task automation, limiting the scope of applications in various GUI scenarios. In this paper, we propose a formalized and comprehensive environment to evaluate the entire process of automated GUI Testing (GTArena), offering a fair, standardized environment for consistent operation of diverse multimodal large language models. We divide the testing process into three key subtasks: test intention generation, test task execution, and GUI defect detection, and construct a benchmark dataset based on these to conduct a comprehensive evaluation. It evaluates the performance of different models using three data types: real mobile applications, mobile applications with artificially injected defects, and synthetic data, thoroughly assessing their capabilities in this relevant task. Additionally, we propose a method that helps researchers explore the correlation between the performance of multimodal language large models in specific scenarios and their general capabilities in standard benchmark tests. Experimental results indicate that even the most advanced models struggle to perform well across all sub-tasks of automated GUI Testing, highlighting a significant gap between the current capabilities of Autonomous GUI Testing and its practical, real-world applicability. This gap provides guidance for the future direction of GUI Agent development. Our code is available at this https URL. 

**Abstract (ZH)**: 如今，GUI代理的研究是人工智能社区的热点话题。然而，现有的研究主要集中在GUI任务自动化上，这限制了其在各种GUI场景中的应用范围。本文提出了一种形式化和全面的环境（GTArena），用于评估整个自动化GUI测试过程，提供了一个公平和标准化的环境，以确保不同多模态大型语言模型的一致运行。我们将测试过程分为三个关键子任务：测试意图生成、测试任务执行和GUI缺陷检测，并基于这些子任务构建基准数据集进行全面评估。该评估使用三种类型的数据：真实移动应用、人工注入缺陷的移动应用以及合成数据，全面评估其在相关任务中的能力。此外，我们提出了一种方法，帮助研究人员探索多模态语言大模型在特定场景中的性能与其在标准基准测试中的通用能力之间的关系。实验结果表明，即使是最先进的模型，在自动化GUI测试的所有子任务中也难以表现出色，凸显了当前自主GUI测试能力与其实用、现实世界适用性之间的显著差距。这一差距为GUI代理未来的发展方向提供了指导。我们的代码可通过以下链接获取：[这里](this https URL)。 

---
# The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence 

**Title (ZH)**: 千脑计划：一种新的感motor智能范式 

**Authors**: Viviane Clay, Niels Leadholm, Jeff Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2412.18354)  

**Abstract**: Artificial intelligence has advanced rapidly in the last decade, driven primarily by progress in the scale of deep-learning systems. Despite these advances, the creation of intelligent systems that can operate effectively in diverse, real-world environments remains a significant challenge. In this white paper, we outline the Thousand Brains Project, an ongoing research effort to develop an alternative, complementary form of AI, derived from the operating principles of the neocortex. We present an early version of a thousand-brains system, a sensorimotor agent that is uniquely suited to quickly learn a wide range of tasks and eventually implement any capabilities the human neocortex has. Core to its design is the use of a repeating computational unit, the learning module, modeled on the cortical columns found in mammalian brains. Each learning module operates as a semi-independent unit that can model entire objects, represents information through spatially structured reference frames, and both estimates and is able to effect movement in the world. Learning is a quick, associative process, similar to Hebbian learning in the brain, and leverages inductive biases around the spatial structure of the world to enable rapid and continual learning. Multiple learning modules can interact with one another both hierarchically and non-hierarchically via a "cortical messaging protocol" (CMP), creating more abstract representations and supporting multimodal integration. We outline the key principles motivating the design of thousand-brains systems and provide details about the implementation of Monty, our first instantiation of such a system. Code can be found at this https URL, along with more detailed documentation at this https URL. 

**Abstract (ZH)**: 在过去十年中，人工智能取得了 rapid 的进步，主要得益于深度学习系统规模的扩大。尽管取得了这些进展，如何创建能够在多种多变的实际环境中有效运作的智能系统仍然是一个重大挑战。在这份白皮书中，我们概述了千脑计划，这是一个正在进行的研究项目，旨在开发一种新的替代性互补形式的 AI，该形式的AI源自新皮层的运作原理。我们介绍了千脑系统的早期版本，这是一种传感器-执行器代理，能够快速学习多种任务，并最终实现与人类新皮层相媲美的所有能力。该系统的核心在于使用重复的计算单元，即学习模块，这种模块模拟了哺乳动物大脑中的皮层柱。每个学习模块作为半独立单元运作，可以建模整个对象、通过空间结构化的参照框架表达信息，并且既能估计又能影响世界中的运动。学习是一个快速关联的过程，类似于大脑中的 Hebbsian 学习，并利用了对世界空间结构的归纳先验，从而实现快速且持续的学习。多个学习模块可以通过“皮层消息协议”（CMP）以层次化和非层次化的方式相互交互，从而生成更抽象的表示，支持多种模态的整合。我们概述了驱动千脑系统设计的关键原则，并详细介绍了我们第一个此类系统的实现，蒙蒂（Monty）。相关代码可以在此找到：https://github.com/…，更详细的文档在此：https://github.com/…。 

---
# MinsStudio: A Streamlined Package for Minecraft AI Agent Development 

**Title (ZH)**: MinsStudio：一种简化版的Minecraft AI代理开发包 

**Authors**: Shaofei Cai, Zhancun Mu, Kaichen He, Bowei Zhang, Xinyue Zheng, Anji Liu, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18293)  

**Abstract**: Minecraft has emerged as a valuable testbed for embodied intelligence and sequential decision-making research, yet the development and validation of novel agents remains hindered by significant engineering challenges. This paper presents MineStudio, an open-source software package designed to streamline embodied policy development in Minecraft. MineStudio represents the first comprehensive integration of seven critical engineering components: simulator, data, model, offline pretraining, online finetuning, inference, and benchmark, thereby allowing users to concentrate their efforts on algorithm innovation. We provide a user-friendly API design accompanied by comprehensive documentation and tutorials. The complete codebase is publicly available at this https URL. 

**Abstract (ZH)**: Minecraft 凭借其在具身智能和顺序决策研究方面的价值，逐渐成为了一个重要的实验平台，然而，新型代理的开发与验证仍然受到重大工程挑战的阻碍。本文介绍了 MineStudio，这是一个开源软件包，旨在简化 Minecraft 中具身策略的开发流程。MineStudio 是首个全面集成七个关键工程组件的平台：模拟器、数据、模型、离线预训练、在线微调、推理和基准测试，从而允许用户将精力聚焦于算法创新。我们提供了用户友好的 API 设计，并配备了详尽的文档和教程。完整的代码库可以在以下网址进行访问：[提供网址]。

注：翻译中“[提供网址]”需替换为实际的网址链接，以便读者可以直接访问代码库。 

---
# VISION: A Modular AI Assistant for Natural Human-Instrument Interaction at Scientific User Facilities 

**Title (ZH)**: VISION：一种适用于科学用户设施的模块化人工智能助手，用于自然的人-仪器交互 

**Authors**: Shray Mathur, Noah van der Vleuten, Kevin Yager, Esther Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2412.18161)  

**Abstract**: Scientific user facilities, such as synchrotron beamlines, are equipped with a wide array of hardware and software tools that require a codebase for human-computer-interaction. This often necessitates developers to be involved to establish connection between users/researchers and the complex instrumentation. The advent of generative AI presents an opportunity to bridge this knowledge gap, enabling seamless communication and efficient experimental workflows. Here we present a modular architecture for the Virtual Scientific Companion (VISION) by assembling multiple AI-enabled cognitive blocks that each scaffolds large language models (LLMs) for a specialized task. With VISION, we performed LLM-based operation on the beamline workstation with low latency and demonstrated the first voice-controlled experiment at an X-ray scattering beamline. The modular and scalable architecture allows for easy adaptation to new instrument and capabilities. Development on natural language-based scientific experimentation is a building block for an impending future where a science exocortex -- a synthetic extension to the cognition of scientists -- may radically transform scientific practice and discovery. 

**Abstract (ZH)**: 科学用户设施，如同步辐射光束线，配备了大量硬件和软件工具，需要代码库来支持人机交互。这通常需要开发人员参与，以建立用户/研究人员与复杂仪器之间的连接。随着生成式AI的发展，为弥补这一知识差距提供了机会，促进了无缝通信和高效的实验流程。本文提出了一种模块化的虚拟科学伴侣（VISION）架构，通过集合多个AI-enabled认知模块，为特定任务搭建大型语言模型（LLMs）。利用VISION，我们实现了基于LLM的光束线工作站操作，并展示了首个语音控制的X射线散射实验。这种模块化和可扩展的架构允许轻松适应新的仪器和功能。基于自然语言的科学实验开发是向着一个即将到来的未来发展的基石，即科学外脑——一种合成扩展科学家认知能力的技术——可能对科学研究和发现产生根本性的影响。 

---
# AutoDroid-V2: Boosting SLM-based GUI Agents via Code Generation 

**Title (ZH)**: AutoDroid-V2：基于代码生成增强SLM方法的GUI代理性能 

**Authors**: Hao Wen, Shizuo Tian, Borislav Pavlov, Wenjie Du, Yixuan Li, Ge Chang, Shanhui Zhao, Jiacheng Liu, Yunxin Liu, Ya-Qin Zhang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.18116)  

**Abstract**: Large language models (LLMs) have brought exciting new advances to mobile UI agents, a long-standing research field that aims to complete arbitrary natural language tasks through mobile UI interactions. However, existing UI agents usually demand high reasoning capabilities of powerful large models that are difficult to be deployed locally on end-users' devices, which raises huge concerns about user privacy and centralized serving cost. One way to reduce the required model size is to customize a smaller domain-specific model with high-quality training data, e.g. large-scale human demonstrations of diverse types of apps and tasks, while such datasets are extremely difficult to obtain. Inspired by the remarkable coding abilities of recent small language models (SLMs), we propose to convert the UI task automation problem to a code generation problem, which can be effectively solved by an on-device SLM and efficiently executed with an on-device code interpreter. Unlike normal coding tasks that can be extensively pretrained with public datasets, generating UI automation code is challenging due to the diversity, complexity, and variability of target apps. Therefore, we adopt a document-centered approach that automatically builds fine-grained API documentation for each app and generates diverse task samples based on this documentation. By guiding the agent with the synthetic documents and task samples, it learns to generate precise and efficient scripts to complete unseen tasks. Based on detailed comparisons with state-of-the-art mobile UI agents, our approach effectively improves the mobile task automation with significantly higher success rates and lower latency/token consumption. Code will be open-sourced. 

**Abstract (ZH)**: 大规模语言模型（LLMs）为移动UI代理带来了令人兴奋的新进展，这是一个长期研究的领域，旨在通过移动UI交互完成任意自然语言任务。然而，现有的UI代理通常需要具备高推理能力的强大模型，这些模型在终端用户的设备上本地部署非常困难，这引发了用户隐私和集中式服务成本方面的巨大担忧。减少所需模型大小的一种方法是使用高质量的训练数据自定义一个小规模领域特定模型，例如大型人类示范各种类型的应用和任务，但这类数据集极其难以获得。受近期小型语言模型（SLMs）卓越编码能力的启发，我们提出了将UI任务自动化问题转化为一个代码生成问题，该问题可以通过设备上的SLM有效地解决，并通过设备上的代码解释器高效执行。与可以广泛使用公开数据集进行预训练的常规编码任务不同，生成UI自动化代码具有挑战性，因为目标应用的多样性、复杂性和变异性。因此，我们采用一种文档中心的方法，自动为每个应用构建细粒度的API文档，并基于这些文档生成多样化的任务样本。通过使用合成文档和任务样本指导代理，它学习生成用于完成未见任务的精确和高效的脚本。基于与最先进的移动UI代理的详细对比，我们的方法显著提高了移动任务自动化的成功率和降低了延迟/令牌消耗。代码将开源。 

---
# Real-world Deployment and Evaluation of PErioperative AI CHatbot (PEACH) -- a Large Language Model Chatbot for Perioperative Medicine 

**Title (ZH)**: 在实际环境中的部署与评估：PEPerioperative AI Chatbot (PEACH)——一个用于围术期医学的大语言模型聊天机器人 

**Authors**: Yu He Ke, Liyuan Jin, Kabilan Elangovan, Bryan Wen Xi Ong, Chin Yang Oh, Jacqueline Sim, Kenny Wei-Tsen Loh, Chai Rick Soh, Jonathan Ming Hua Cheng, Aaron Kwang Yang Lee, Daniel Shu Wei Ting, Nan Liu, Hairil Rizal Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2412.18096)  

**Abstract**: Large Language Models (LLMs) are emerging as powerful tools in healthcare, particularly for complex, domain-specific tasks. This study describes the development and evaluation of the PErioperative AI CHatbot (PEACH), a secure LLM-based system integrated with local perioperative guidelines to support preoperative clinical decision-making. PEACH was embedded with 35 institutional perioperative protocols in the secure Claude 3.5 Sonet LLM framework within Pair Chat (developed by Singapore Government) and tested in a silent deployment with real-world data. Accuracy, safety, and usability were assessed. Deviations and hallucinations were categorized based on potential harm, and user feedback was evaluated using the Technology Acceptance Model (TAM). Updates were made after the initial silent deployment to amend one protocol.
In 240 real-world clinical iterations, PEACH achieved a first-generation accuracy of 97.5% (78/80) and an overall accuracy of 96.7% (232/240) across three iterations. The updated PEACH demonstrated improved accuracy of 97.9% (235/240), with a statistically significant difference from the null hypothesis of 95% accuracy (p = 0.018, 95% CI: 0.952-0.991). Minimal hallucinations and deviations were observed (both 1/240 and 2/240, respectively). Clinicians reported that PEACH expedited decisions in 95% of cases, and inter-rater reliability ranged from kappa 0.772-0.893 within PEACH and 0.610-0.784 among attendings.
PEACH is an accurate, adaptable tool that enhances consistency and efficiency in perioperative decision-making. Future research should explore its scalability across specialties and its impact on clinical outcomes. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在成为医疗保健领域中强大的工具，特别是在处理复杂且领域特定的任务方面。本研究描述了PE亟术AI聊天机器人（PEACH）的开发与评估，PEACH是一种结合了当地围手术期指南的安全LLM系统，旨在支持术前临床决策。PEACH嵌入了35个机构的围手术期协议，使用了安全的Claude 3.5 Sonet LLM框架，并在新加坡政府开发的Pair Chat中进行了无声部署，使用真实世界数据进行了测试。评估了其准确度、安全性和可用性。基于潜在危害，对偏差和幻觉进行了分类，并使用技术接受模型（TAM）评估了用户反馈。在首次无声部署后进行了更新，修正了一个协议。

在240例真实世界的临床迭代中，PEACH的第一代准确率为97.5%（78/80），整体准确率为96.7%（232/240）。更新后的PEACH的准确率提高至97.9%（235/240），与95%的准确率（虚无假设）相比有显著性差异（p=0.018，95% CI：0.952-0.991）。观察到的幻觉和偏差很少（分别为1/240和2/240）。临床医生报告称，在95%的情况下，PEACH加速了决策过程，并且PEACH内的信度范围为Kappa 0.772-0.893，而主治医生间的信度范围为Kappa 0.610-0.784。

PEACH是一款准确且可适应的工具，可增强围手术期决策的一致性和效率。未来的研究应探索其在各个专科的可扩展性及其对临床结果的影响。 

---
# Dynamic Multi-Agent Orchestration and Retrieval for Multi-Source Question-Answer Systems using Large Language Models 

**Title (ZH)**: 使用大型语言模型的多源问答系统中动态多agent编排与检索方法 

**Authors**: Antony Seabra, Claudio Cavalcante, Joao Nepomuceno, Lucas Lago, Nicolaas Ruberg, Sergio Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17964)  

**Abstract**: We propose a methodology that combines several advanced techniques in Large Language Model (LLM) retrieval to support the development of robust, multi-source question-answer systems. This methodology is designed to integrate information from diverse data sources, including unstructured documents (PDFs) and structured databases, through a coordinated multi-agent orchestration and dynamic retrieval approach. Our methodology leverages specialized agents-such as SQL agents, Retrieval-Augmented Generation (RAG) agents, and router agents - that dynamically select the most appropriate retrieval strategy based on the nature of each query. To further improve accuracy and contextual relevance, we employ dynamic prompt engineering, which adapts in real time to query-specific contexts. The methodology's effectiveness is demonstrated within the domain of Contract Management, where complex queries often require seamless interaction between unstructured and structured data. Our results indicate that this approach enhances response accuracy and relevance, offering a versatile and scalable framework for developing question-answer systems that can operate across various domains and data sources. 

**Abstract (ZH)**: 我们提出了一种方法论，结合了大型语言模型（LLM）检索中的多项高级技术，以支持稳健的多源问答系统的开发。该方法论旨在通过协调多代理编排和动态检索方法，整合来自多种数据源的信息，包括未结构化的文档（如PDF）和结构化的数据库。此方法论利用了专用于SQL代理、检索增强生成（RAG）代理和路由器代理等特定任务的智能代理，它们能够根据每个查询的性质动态选择最合适的检索策略。为了进一步提高准确性和上下文相关性，我们采用了动态提示工程，它能够根据查询的具体上下文实时调整。该方法论在合同管理领域得到了验证，在该领域复杂的查询往往需要无缝地处理未结构化和结构化数据之间的交互。实验结果表明，该方法增强了响应的准确性和相关性，提供了一个在不同领域和数据源上操作的灵活和可扩展框架，用于开发问答系统。 

---
# Contrato360 2.0: A Document and Database-Driven Question-Answer System using Large Language Models and Agents 

**Title (ZH)**: Contrato360 2.0：一种基于文档和数据库的大型语言模型及智能代理驱动的问答系统 

**Authors**: Antony Seabra, Claudio Cavalcante, Joao Nepomuceno, Lucas Lago, Nicolaas Ruberg, Sergio Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17942)  

**Abstract**: We present a question-and-answer (Q\&A) application designed to support the contract management process by leveraging combined information from contract documents (PDFs) and data retrieved from contract management systems (database). This data is processed by a large language model (LLM) to provide precise and relevant answers. The accuracy of these responses is further enhanced through the use of Retrieval-Augmented Generation (RAG), text-to-SQL techniques, and agents that dynamically orchestrate the workflow. These techniques eliminate the need to retrain the language model. Additionally, we employed Prompt Engineering to fine-tune the focus of responses. Our findings demonstrate that this multi-agent orchestration and combination of techniques significantly improve the relevance and accuracy of the answers, offering a promising direction for future information systems. 

**Abstract (ZH)**: 我们提出了一种基于问题回答（Q&A）的应用程序，旨在通过利用合同文件（PDFs）和从合同管理系统中检索的数据（数据库）的综合信息来支持合同管理流程。这些数据由大型语言模型（LLM）处理，以提供精准和相关的答案。通过使用检索增强生成（RAG）、文本到SQL技术以及能够动态调度工作流程的代理，这些答案的准确性得以进一步提高。这些技术消除了重新训练语言模型的必要性。此外，我们采用了提示工程技术来精细调整答案的焦点。我们的研究结果表明，这种多代理调度及其技术组合显著提高了答案的相关性和准确性，并为未来的信息系统提供了令人鼓舞的方向。 

---
# Decentralized Intelligence in GameFi: Embodied AI Agents and the Convergence of DeFi and Virtual Ecosystems 

**Title (ZH)**: GameFi中去中心化智能：具身人工智能代理及其与去中心化金融和虚拟生态系统融合的研究 

**Authors**: Fernando Jia, Jade Zheng, Florence Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.18601)  

**Abstract**: In the rapidly evolving landscape of GameFi, a fusion of gaming and decentralized finance (DeFi), there exists a critical need to enhance player engagement and economic interaction within gaming ecosystems. Our GameFi ecosystem aims to fundamentally transform this landscape by integrating advanced embodied AI agents into GameFi platforms. These AI agents, developed using cutting-edge large language models (LLMs), such as GPT-4 and Claude AI, are capable of proactive, adaptive, and contextually rich interactions with players. By going beyond traditional scripted responses, these agents become integral participants in the game's narrative and economic systems, directly influencing player strategies and in-game economies. We address the limitations of current GameFi platforms, which often lack immersive AI interactions and mechanisms for community engagement or creator monetization. Through the deep integration of AI agents with blockchain technology, we establish a consensus-driven, decentralized GameFi ecosystem. This ecosystem empowers creators to monetize their contributions and fosters democratic collaboration among players and creators. Furthermore, by embedding DeFi mechanisms into the gaming experience, we enhance economic participation and provide new opportunities for financial interactions within the game. Our approach enhances player immersion and retention and advances the GameFi ecosystem by bridging traditional gaming with Web3 technologies. By integrating sophisticated AI and DeFi elements, we contribute to the development of more engaging, economically robust, and community-centric gaming environments. This project represents a significant advancement in the state-of-the-art in GameFi, offering insights and methodologies that can be applied throughout the gaming industry. 

**Abstract (ZH)**: 在GameFi这一快速发展的领域中，GameFi结合了游戏和去中心化金融（DeFi），存在着增强玩家参与度和经济互动的迫切需求。我们的GameFi生态系统旨在通过将先进的具身AI代理整合到GameFi平台中，从根本上改变这一现状。这些AI代理采用最新的人工智能语言模型（如GPT-4和Claude AI）开发，能够在与玩家的互动中展现出主动、适应性及丰富的情境交互能力。通过超越传统的预设回应，这些代理能够成为游戏叙事和经济系统的积极参与者，直接影响玩家的策略和游戏内的经济体系。我们解决了当前GameFi平台中常见的沉浸式AI互动缺乏和社区参与机制不足的问题。通过将AI代理与区块链技术深度融合，我们建立了一个基于共识的去中心化GameFi生态系统。该生态系统赋予创作者们通过自己的贡献获取收益的能力，并促进了玩家和创作者之间的民主化合作。此外，通过将DeFi机制嵌入游戏体验中，我们增强了经济参与度并为游戏内的金融互动提供了新的机会。我们的方法增强了玩家的沉浸感和留存率，并通过将传统游戏与Web3技术相结合，推动了GameFi生态系统的进步。通过整合复杂的AI和DeFi元素，我们为更具吸引力、经济稳定性更强且更注重社区的 gaming环境做出了贡献。本项目代表了GameFi领域的重大进展，提供了可用于整个游戏行业的见解和方法论。 

---
# A Paragraph is All It Takes: Rich Robot Behaviors from Interacting, Trusted LLMs 

**Title (ZH)**: 一节文字足矣：来自相互信任的大型语言模型的丰富机器人行为 

**Authors**: OpenMind, Shaohong Zhong, Adam Zhou, Boyuan Chen, Homin Luo, Jan Liphardt  

**Link**: [PDF](https://arxiv.org/pdf/2412.18588)  

**Abstract**: Large Language Models (LLMs) are compact representations of all public knowledge of our physical environment and animal and human behaviors. The application of LLMs to robotics may offer a path to highly capable robots that perform well across most human tasks with limited or even zero tuning. Aside from increasingly sophisticated reasoning and task planning, networks of (suitably designed) LLMs offer ease of upgrading capabilities and allow humans to directly observe the robot's thinking. Here we explore the advantages, limitations, and particularities of using LLMs to control physical robots. The basic system consists of four LLMs communicating via a human language data bus implemented via web sockets and ROS2 message passing. Surprisingly, rich robot behaviors and good performance across different tasks could be achieved despite the robot's data fusion cycle running at only 1Hz and the central data bus running at the extremely limited rates of the human brain, of around 40 bits/s. The use of natural language for inter-LLM communication allowed the robot's reasoning and decision making to be directly observed by humans and made it trivial to bias the system's behavior with sets of rules written in plain English. These rules were immutably written into Ethereum, a global, public, and censorship resistant Turing-complete computer. We suggest that by using natural language as the data bus among interacting AIs, and immutable public ledgers to store behavior constraints, it is possible to build robots that combine unexpectedly rich performance, upgradability, and durable alignment with humans. 

**Abstract (ZH)**: 大规模语言模型（LLMs）是对我们物理环境以及动物和人类行为的全部公共知识的紧凑表示。将LLMs应用于机器人技术可能为创建高度能胜任、能够在大多数人类任务中表现出色的机器人提供一条路径，即使缺乏或完全没有调优。除了日益复杂的推理和任务规划外，适当地设计的LLM网络提供了能力更新的简便性和允许人类直接观察机器人思维的独特性。本文探讨了使用LLMs控制物理机器人的优势、局限性和特定性。基本系统由四个通过WebSocket和ROS2消息传递实现的人类语言数据总线进行通信的LLM组成。令人惊讶的是，尽管机器人数据融合周期仅为每秒1Hz，中央数据总线的运行速率也极其受限，大约为人类大脑的40比特/秒，但仍然能够实现丰富的机器人行为和不同任务的良好表现。自然语言作为LLM间通信介质使得人类可以直接观察机器人的推理和决策过程，并且通过以简单英语撰写的规则集轻松地偏置系统的行为。这些规则被不可变地写入以太坊中，这是一个全球性的、公开的和抗审查的图灵完备计算机。我们建议，通过在交互的AI之间使用自然语言作为数据总线，并使用不可变的公共账本来存储行为约束，有可能构建出结合丰富性能、易于更新和持久的人类对齐的机器人。 

---
# GeAR: Graph-enhanced Agent for Retrieval-augmented Generation 

**Title (ZH)**: GeAR：基于图的代理增强检索增强生成

这个翻译符合学术规范，同时保持了原文的意思和结构。在这里，“Graph-enhanced”被翻译为“基于图的”，“agent”翻译为“代理”，“retrieval-augmented generation”翻译为“检索增强生成”，以确保术语的专业性和准确性。 

**Authors**: Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Piramanayagam, Damien Graux, Dandan Tu, Zeren Jiang, Ruofei Lai, Yang Ren, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18431)  

**Abstract**: Retrieval-augmented generation systems rely on effective document retrieval capabilities. By design, conventional sparse or dense retrievers face challenges in multi-hop retrieval scenarios. In this paper, we present GeAR, which advances RAG performance through two key innovations: (i) graph expansion, which enhances any conventional base retriever, such as BM25, and (ii) an agent framework that incorporates graph expansion. Our evaluation demonstrates GeAR's superior retrieval performance on three multi-hop question answering datasets. Additionally, our system achieves state-of-the-art results with improvements exceeding 10% on the challenging MuSiQue dataset, while requiring fewer tokens and iterations compared to other multi-step retrieval systems. 

**Abstract (ZH)**: 检索增强生成系统依赖于有效的文档检索能力。从设计上讲，传统的稀疏或密集检索器在多跳检索场景中面临挑战。本文中，我们提出了GeAR，通过两项关键创新来提升RAG（检索增强生成）的表现：(i) 图扩展，该方法可以增强任何传统的基线检索器，例如BM25；(ii) 一个代理框架，该框架结合了图扩展。我们的评估结果显示，GeAR在三个多跳问答数据集上的检索性能优于其他方法。此外，在具有挑战性的MuSiQue数据集上，我们的系统取得了当前最佳结果，相比其他多步检索系统，所需token数量和迭代次数更少，性能提升超过10%。 

---
# ChaI-TeA: A Benchmark for Evaluating Autocompletion of Interactions with LLM-based Chatbots 

**Title (ZH)**: ChaI-TeA: 一种评估基于大语言模型的聊天机器人交互自动完成性能的基准 

**Authors**: Shani Goren, Oren Kalinsky, Tomer Stav, Yuri Rapoport, Yaron Fairstein, Ram Yazdy, Nachshon Cohen, Alexander Libov, Guy Kushilevitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.18377)  

**Abstract**: The rise of LLMs has deflected a growing portion of human-computer interactions towards LLM-based chatbots. The remarkable abilities of these models allow users to interact using long, diverse natural language text covering a wide range of topics and styles. Phrasing these messages is a time and effort consuming task, calling for an autocomplete solution to assist users. We introduce the task of chatbot interaction autocomplete. We present ChaI-TeA: CHat InTEraction Autocomplete; An autcomplete evaluation framework for LLM-based chatbot interactions. The framework includes a formal definition of the task, coupled with suitable datasets and metrics. We use the framework to evaluate After formally defining the task along with suitable datasets and metrics, we test 9 models on the defined auto completion task, finding that while current off-the-shelf models perform fairly, there is still much room for improvement, mainly in ranking of the generated suggestions. We provide insights for practitioners working on this task and open new research directions for researchers in the field. We release our framework to serve as a foundation for future research. 

**Abstract (ZH)**: 生成型大模型（LLM）的兴起使得越来越多的人机交互转向基于大模型的对话机器人。这些模型的卓越能力使得用户能够使用长篇、多样的自然语言文本，涵盖广泛的主题和风格进行互动。编写这些消息是一个既耗时又费力的任务，因此需要一个自动补全解决方案来辅助用户。本文介绍了对话机器人交互自动补全的任务。我们提出了ChaI-TeA：对话交互自动补全；一个用于基于大模型对话机器人交互的自动补全评估框架。该框架包括对任务的正式定义，以及适用的数据集和评估指标。我们使用该框架对定义的自动补全任务进行了评估，并测试了9个模型，发现尽管现有的现成模型表现尚可，但在生成建议的排名方面仍有很大的改进空间。我们为在这个任务上工作的实践者提供了见解，并为该领域研究人员打开了新的研究方向。我们发布了该框架，以作为未来研究的基础。 

---
# Multi-Agents Based on Large Language Models for Knowledge-based Visual Question Answering 

**Title (ZH)**: 基于大型语言模型的多agents知识驱动视觉问答系统 

**Authors**: Zhongjian Hu, Peng Yang, Bing Li, Zhenqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18351)  

**Abstract**: Large Language Models (LLMs) have achieved impressive results in knowledge-based Visual Question Answering (VQA). However existing methods still have challenges: the inability to use external tools autonomously, and the inability to work in teams. Humans tend to know whether they need to use external tools when they encounter a new question, e.g., they tend to be able to give a direct answer to a familiar question, whereas they tend to use tools such as search engines when they encounter an unfamiliar question. In addition, humans also tend to collaborate and discuss with others to get better answers. Inspired by this, we propose the multi-agent voting framework. We design three LLM-based agents that simulate different levels of staff in a team, and assign the available tools according to the levels. Each agent provides the corresponding answer, and finally all the answers provided by the agents are voted to get the final answer. Experiments on OK-VQA and A-OKVQA show that our approach outperforms other baselines by 2.2 and 1.0, respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在基于知识的视觉问答（VQA）任务上取得了显著成果。然而，现有的方法仍然存在一些挑战：自主使用外部工具的能力不足，以及无法团队协作工作。人类在遇到新问题时通常能判断是否需要使用外部工具，例如，他们往往能直接回答熟悉的问题，而遇到不熟悉的问题时会使用诸如搜索引擎等工具。此外，人类还倾向于与他人协作讨论以获得更好的答案。受此启发，我们提出了多Agent投票框架。我们设计了三个基于LLM的Agent，模拟团队中不同级别的工作人员，并根据其级别分配可用的工具。每个Agent提供相应的答案，最后通过投票将所有Agent提供的答案结合以获得最终答案。实验结果表明，我们的方法在OK-VQA和A-OKVQA数据集上分别优于其他基线方法2.2和1.0的分数。 

---
# VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks 

**Title (ZH)**: VLABench：一种用于长时序推理任务的基于语言条件的机器人操纵大规模基准测试 

**Authors**: Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18194)  

**Abstract**: General-purposed embodied agents are designed to understand the users' natural instructions or intentions and act precisely to complete universal tasks. Recently, methods based on foundation models especially Vision-Language-Action models (VLAs) have shown a substantial potential to solve language-conditioned manipulation (LCM) tasks well. However, existing benchmarks do not adequately meet the needs of VLAs and relative algorithms. To better define such general-purpose tasks in the context of LLMs and advance the research in VLAs, we present VLABench, an open-source benchmark for evaluating universal LCM task learning. VLABench provides 100 carefully designed categories of tasks, with strong randomization in each category of task and a total of 2000+ objects. VLABench stands out from previous benchmarks in four key aspects: 1) tasks requiring world knowledge and common sense transfer, 2) natural language instructions with implicit human intentions rather than templates, 3) long-horizon tasks demanding multi-step reasoning, and 4) evaluation of both action policies and language model capabilities. The benchmark assesses multiple competencies including understanding of mesh\&texture, spatial relationship, semantic instruction, physical laws, knowledge transfer and reasoning, etc. To support the downstream finetuning, we provide high-quality training data collected via an automated framework incorporating heuristic skills and prior information. The experimental results indicate that both the current state-of-the-art pretrained VLAs and the workflow based on VLMs face challenges in our tasks. 

**Abstract (ZH)**: 通用型躯体化代理旨在理解和执行用户的自然指令或意图，以精确地完成通用任务。近期，基于基础模型的方法，尤其是视觉-语言-动作模型（VLAs），在解决语言条件下的操作任务（LCM）方面展现了巨大的潜力。然而，现有的基准测试并没有充分满足VLAs及其相关算法的需求。为更好地在大语言模型（LLMs）的背景下定义这类通用任务，并推进VLAs的研究，我们提出了VLABench，这是一个开源基准测试，用于评估通用LCM任务的学习。VLABench 提供了100个精心设计的任务类别，每个类别都有较强的随机化，并包含了超过2000个对象。与之前的基准测试相比，VLABench 在四个方面脱颖而出：1) 要求世界知识和常识的转移；2) 自然语言指令中隐含着人类意图，而非模板；3) 需要多步推理的长期任务；4) 评估动作策略和语言模型能力。基准测试评估了多个能力，包括网格与纹理的理解、空间关系、语义指令、物理定律、知识转移和推理等。为了支持下游微调，我们提供了一套高质量的训练数据，这些数据是通过结合启发式技能和先验信息的自动化框架收集的。实验结果表明，当前最先进的预训练VLAs和基于VLMs的工作流程在我们的任务中都面临着挑战。 

---
# EvoPat: A Multi-LLM-based Patents Summarization and Analysis Agent 

**Title (ZH)**: EvoPat：一个基于多层次语言模型的专利总结与分析代理 

**Authors**: Suyuan Wang, Xueqian Yin, Menghao Wang, Ruofeng Guo, Kai Nan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18100)  

**Abstract**: The rapid growth of scientific techniques and knowledge is reflected in the exponential increase in new patents filed annually. While these patents drive innovation, they also present significant burden for researchers and engineers, especially newcomers. To avoid the tedious work of navigating a vast and complex landscape to identify trends and breakthroughs, researchers urgently need efficient tools to summarize, evaluate, and contextualize patents, revealing their innovative contributions and underlying scientific this http URL address this need, we present EvoPat, a multi-LLM-based patent agent designed to assist users in analyzing patents through Retrieval-Augmented Generation (RAG) and advanced search strategies. EvoPat leverages multiple Large Language Models (LLMs), each performing specialized roles such as planning, identifying innovations, and conducting comparative evaluations. The system integrates data from local databases, including patents, literature, product catalogous, and company repositories, and online searches to provide up-to-date insights. The ability to collect information not included in original database automatically is also implemented. Through extensive testing in the natural language processing (NLP) domain, we demonstrate that EvoPat outperforms GPT-4 in tasks such as patent summarization, comparative analysis, and technical evaluation. EvoPat represents a significant step toward creating AI-powered tools that empower researchers and engineers to efficiently navigate the complexities of the patent landscape. 

**Abstract (ZH)**: 科学技术与知识的迅速增长体现在每年提交的新专利数量呈指数级增长上。尽管这些专利推动了创新，但也给研究人员和工程师带来了巨大的负担，尤其是新手。为了避免在庞大而复杂的环境中进行繁琐的工作来识别趋势和突破，研究人员迫切需要高效的工具来总结、评估和语境化专利，揭示其创新贡献和背后的科学意义。为了解决这一需求，我们提出了EvoPat，这是一种基于多大规模语言模型（LLM）的专利代理，旨在通过检索增强生成（RAG）和高级搜索策略帮助用户分析专利。EvoPat利用多个大规模语言模型，每个模型分别承担规划、识别创新和进行对比评价等专业化角色。该系统整合了本地数据库中的数据，包括专利、文献、产品目录和公司库，并结合在线搜索，提供最新的见解。同时，系统还具备自动收集不在原始数据库中但相关的数据信息的能力。通过在自然语言处理（NLP）领域的广泛测试，我们证明EvoPat在专利总结、对比分析和技术评估等任务上优于GPT-4。EvoPat代表了向创建能够帮助研究人员和工程师高效导航专利复杂性的AI工具迈出的重要一步。 

---
# Uncertainty-Aware Critic Augmentation for Hierarchical Multi-Agent EV Charging Control 

**Title (ZH)**: 具有不确定性意识的评论者增强方法用于层次化多代理电动汽车充电控制 

**Authors**: Lo Pang-Yun Ting, Ali Şenol, Huan-Yang Wang, Hsu-Chao Lai, Kun-Ta Chuang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18047)  

**Abstract**: The advanced bidirectional EV charging and discharging technology, aimed at supporting grid stability and emergency operations, has driven a growing interest in workplace applications. It not only effectively reduces electricity expenses but also enhances the resilience of handling practical issues, such as peak power limitation, fluctuating energy prices, and unpredictable EV departures. However, existing EV charging strategies have yet to fully consider these factors in a way that benefits both office buildings and EV users simultaneously. To address these issues, we propose HUCA, a novel real-time charging control for regulating energy demands for both the building and electric vehicles. HUCA employs hierarchical actor-critic networks to dynamically reduce electricity costs in buildings, accounting for the needs of EV charging in the dynamic pricing scenario. To tackle the uncertain EV departures, a new critic augmentation is introduced to account for departure uncertainties in evaluating the charging decisions, while maintaining the robustness of the charging control. Experiments on real-world electricity datasets under both simulated certain and uncertain departure scenarios demonstrate that HUCA outperforms baselines in terms of total electricity costs while maintaining competitive performance in fulfilling EV charging requirements. A case study also manifests that HUCA effectively balances energy supply between the building and EVs based on real-time information. 

**Abstract (ZH)**: 先进双向电动汽车充电和放电技术旨在支持电网稳定和应急操作，正引发对工作场所应用的兴趣增长。它不仅有效降低电费开支，还能增强处理实际问题的韧性，如高峰功率限制、波动的能源价格和不可预测的电动汽车离开情况。然而，现有的电动汽车充电策略尚未以同时惠及办公楼和电动汽车用户的方式充分考虑这些因素。为解决这些问题，我们提出了一种名为HUCA的新型实时充电控制方法，用于调节建筑和电动汽车的能源需求。HUCA采用分层的行为-批评网络，根据动态定价场景下的电动汽车充电需求，动态降低建筑物的电费成本。为应对电动汽车不确定的离开情况，引入了一种新的批评增强方法，以评估充电决策时考虑到离开不确定性，同时保持充电控制的鲁棒性。在实际电力数据集下的模拟确定和不确定离开场景实验中，HUCA在总电费成本方面表现优于基准模型，同时在满足电动汽车充电需求方面保持了竞争力。案例研究表明，HUCA能够根据实时信息有效平衡办公楼和电动汽车之间的能源供应。 

---
# More than Chit-Chat: Developing Robots for Small-Talk Interactions 

**Title (ZH)**: 不仅仅是闲聊：开发用于短对话的机器人 

**Authors**: Rebecca Ramnauth, Dražen Brščić, Brian Scassellati  

**Link**: [PDF](https://arxiv.org/pdf/2412.18023)  

**Abstract**: Beyond mere formality, small talk plays a pivotal role in social dynamics, serving as a verbal handshake for building rapport and understanding. For conversational AI and social robots, the ability to engage in small talk enhances their perceived sociability, leading to more comfortable and natural user interactions. In this study, we evaluate the capacity of current Large Language Models (LLMs) to drive the small talk of a social robot and identify key areas for improvement. We introduce a novel method that autonomously generates feedback and ensures LLM-generated responses align with small talk conventions. Through several evaluations -- involving chatbot interactions and human-robot interactions -- we demonstrate the system's effectiveness in guiding LLM-generated responses toward realistic, human-like, and natural small-talk exchanges. 

**Abstract (ZH)**: 超越单纯的礼仪，闲聊在社交动态中发挥着关键作用，作为建立共鸣和沟通理解的口头握手。对于对话式人工智能和社交机器人而言，具备进行闲聊的能力可以增强它们的社交感知度，从而实现更为舒适和自然的用户互动。在本研究中，我们评估了当前大规模语言模型（LLM）在驱动社交机器人闲聊方面的能力，并确定了需要改进的关键领域。我们提出了一种新颖的方法，能够自主生成反馈并确保LLM生成的回复符合闲聊规范。通过多项评估——包括聊天机器人互动和人机互动——我们证明了该系统在引导LLM生成的回复向真实、人类般的自然闲聊交流方向发展的有效性。 

---
# Multi-Agent Path Finding in Continuous Spaces with Projected Diffusion Models 

**Title (ZH)**: 在连续空间中使用投影扩散模型的多代理路径规划 

**Authors**: Jinhao Liang, Jacob K. Christopher, Sven Koenig, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2412.17993)  

**Abstract**: Multi-Agent Path Finding (MAPF) is a fundamental problem in robotics, requiring the computation of collision-free paths for multiple agents moving from their respective start to goal positions. Coordinating multiple agents in a shared environment poses significant challenges, especially in continuous spaces where traditional optimization algorithms struggle with scalability. Moreover, these algorithms often depend on discretized representations of the environment, which can be impractical in image-based or high-dimensional settings. Recently, diffusion models have shown promise in single-agent path planning, capturing complex trajectory distributions and generating smooth paths that navigate continuous, high-dimensional spaces. However, directly extending diffusion models to MAPF introduces new challenges since these models struggle to ensure constraint feasibility, such as inter-agent collision avoidance. To overcome this limitation, this work proposes a novel approach that integrates constrained optimization with diffusion models for MAPF in continuous spaces. This unique combination directly produces feasible multi-agent trajectories that respect collision avoidance and kinematic constraints. The effectiveness of our approach is demonstrated across various challenging simulated scenarios of varying dimensionality. 

**Abstract (ZH)**: 多智能体路径寻找（Multi-Agent Path Finding, MAPF）是机器人领域的一项基础问题，要求为多个智能体从各自的起始位置到目标位置生成无碰撞的路径。在共享环境中协调多个智能体移动带来了显著的挑战，尤其是在连续空间中，传统的优化算法往往在可扩展性方面存在困难。此外，这些算法常依赖于环境的离散表示，这在基于图像或高维设置的情况下可能不切实际。最近，扩散模型在单智能体路径规划方面显示了潜力，能够捕捉复杂的轨迹分布并生成平滑路径，以导航连续的高维空间。然而，直接将扩散模型扩展到MAPF引入了新的挑战，因为这些模型难以确保满足约束条件，例如避免智能体间的碰撞。为此，本文提出了一种新颖的方法，结合约束优化与扩散模型在连续空间中的MAPF问题中。这种独特的结合可以直接生成满足碰撞避免和动力学约束的可行多智能体轨迹。我们的方法在各种具有不同维度的挑战性模拟场景中得到了有效性验证。 

---
# Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types 

**Title (ZH)**: 评估并提升针对多轮文本到SQL转换的大型语言模型的性能，涵盖多种问题类型 

**Authors**: Ziming Guo, Chao Ma, Yinggang Sun, Tiancheng Zhao, Guangyao Wang, Hai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17867)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly advanced text-to-SQL systems. However, most LLM-based methods often narrowly focus on SQL generation, neglecting the complexities of real-world conversational queries. This oversight can lead to unreliable responses, particularly for ambiguous questions that cannot be directly addressed with SQL. To bridge this gap, we propose MMSQL, a comprehensive test suite designed to evaluate the question classification and SQL generation capabilities of LLMs by simulating real-world scenarios with diverse question types and multi-turn Q\&A interactions. Using MMSQL, we assessed the performance of popular LLMs, including both open-source and closed-source models, and identified key factors impacting their performance in such scenarios. Moreover, we introduce an LLM-based multi-agent framework that employs specialized agents to identify question types and determine appropriate answering strategies. Our experiments demonstrate that this approach significantly enhances the model's ability to navigate the complexities of conversational dynamics, effectively handling the diverse and complex nature of user queries. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进展显著提升了文本到SQL系统的性能。然而，大多数基于LLM的方法往往仅专注于SQL生成，忽视了真实世界对话查询的复杂性。这种忽视可能导致不可靠的响应，特别是对于那些不能直接用SQL回答的含糊不清的问题。为了弥补这一差距，我们提出了MMSQL，这是一个全面的测试套件，通过模拟具有多种问题类型和多轮问答交互的真实场景，评估LLM的问题分类和SQL生成能力。通过MMSQL，我们评估了几种流行的LLM，包括开源和封闭源模型，并确定了影响其在这种情况下表现的关键因素。此外，我们引入了一种基于LLM的多代理框架，该框架使用专门的代理来识别问题类型并确定合适的回答策略。我们的实验表明，这种方法显著增强了模型适应对话动态复杂性的能力，有效处理了用户查询的多样性和复杂性。 

---
# Coordinated Power Smoothing Control for Wind Storage Integrated System with Physics-informed Deep Reinforcement Learning 

**Title (ZH)**: 基于物理信息深度强化学习的风储一体化系统协调功率平滑控制 

**Authors**: Shuyi Wang, Huan Zhao, Yuji Cao, Zibin Pan, Guolong Liu, Gaoqi Liang, Junhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.17838)  

**Abstract**: The Wind Storage Integrated System with Power Smoothing Control (PSC) has emerged as a promising solution to ensure both efficient and reliable wind energy generation. However, existing PSC strategies overlook the intricate interplay and distinct control frequencies between batteries and wind turbines, and lack consideration of wake effect and battery degradation cost. In this paper, a novel coordinated control framework with hierarchical levels is devised to address these challenges effectively, which integrates the wake model and battery degradation model. In addition, after reformulating the problem as a Markov decision process, the multi-agent reinforcement learning method is introduced to overcome the bi-level characteristic of the problem. Moreover, a Physics-informed Neural Network-assisted Multi-agent Deep Deterministic Policy Gradient (PAMA-DDPG) algorithm is proposed to incorporate the power fluctuation differential equation and expedite the learning process. The effectiveness of the proposed methodology is evaluated through simulations conducted in four distinct scenarios using WindFarmSimulator (WFSim). The results demonstrate that the proposed algorithm facilitates approximately an 11% increase in total profit and a 19% decrease in power fluctuation compared to the traditional methods, thereby addressing the dual objectives of economic efficiency and grid-connected energy reliability. 

**Abstract (ZH)**: 具有功率平滑控制（PSC）的风储一体化系统已经成为了确保风能高效可靠生成的一种有前景的解决方案。然而，现有的PSC策略忽视了电池与风力涡轮机之间的复杂交互和不同的控制频率，并且没有考虑到尾流效应和电池退化成本。在本文中，我们设计了一种分层式的新型协调控制框架以有效应对这些挑战，该框架整合了尾流模型和电池退化模型。此外，在将问题重新表述为马尔可夫决策过程之后，我们引入了多智能体强化学习方法以克服该问题的多级特性。同时，我们提出了一种基于物理信息神经网络的多智能体深度确定性策略梯度算法（PAMA-DDPG），以结合功率波动微分方程并加速学习过程。通过使用WindFarmSimulator (WFSim) 在四个不同的场景下进行仿真，我们评估了所提出的方法的有效性。结果表明，所提出的算法在总利润方面约提高了11%，在功率波动方面约减少了19%，从而实现了经济效率和并网能源可靠性双重目标。 

---
# Extracting triples from dialogues for conversational social agents 

**Title (ZH)**: 将对话中的三元组提取用于会话社会代理 

**Authors**: Piek Vossen, Selene Báez Santamaría, Lenka Bajčetić, Thomas Belluci  

**Link**: [PDF](https://arxiv.org/pdf/2412.18364)  

**Abstract**: Obtaining an explicit understanding of communication within a Hybrid Intelligence collaboration is essential to create controllable and transparent agents. In this paper, we describe a number of Natural Language Understanding models that extract explicit symbolic triples from social conversation. Triple extraction has mostly been developed and tested for Knowledge Base Completion using Wikipedia text and data for training and testing. However, social conversation is very different as a genre in which interlocutors exchange information in sequences of utterances that involve statements, questions, and answers. Phenomena such as co-reference, ellipsis, coordination, and implicit and explicit negation or confirmation are more prominent in conversation than in Wikipedia text. We therefore describe an attempt to fill this gap by releasing data sets for training and testing triple extraction from social conversation. We also created five triple extraction models and tested them in our evaluation data. The highest precision is 51.14 for complete triples and 69.32 for triple elements when tested on single utterances. However, scores for conversational triples that span multiple turns are much lower, showing that extracting knowledge from true conversational data is much more challenging. 

**Abstract (ZH)**: 了解混合智能协作中通信的显式理解对于创建可控和透明的智能体至关重要。在本文中，我们描述了一些自然语言理解模型，这些模型可以从社会对话中提取显式的符号三元组。三元组提取主要在知识库完成任务中进行开发和测试，通常使用维基百科的文本和数据进行训练和测试。然而，社会对话作为一种体裁存在巨大差异，在这种对话中，对话参与者通过一系列包括陈述、提问和回答在内的话语交换信息。在对话中，共指、省略、并列、以及隐式和显式的否定或确认现象更为明显。因此，我们尝试填补这一空白，通过发布社会对话数据集来训练和测试三元组提取。我们还创建了五个三元组提取模型，并在我们的评估数据集上进行了测试。在单独话语上测试时，完整三元组的最高精度为51.14%，三元组元素的最高精度为69.32%。然而，跨多个回合的对话三元组的得分明显较低，这表明从真实对话数据中提取知识更具挑战性。 

---
# Molly: Making Large Language Model Agents Solve Python Problem More Logically 

**Title (ZH)**: 莫莉：使大型语言模型代理更逻辑地解决Python问题 

**Authors**: Rui Xiao, Jiong Wang, Lu Han, Na Zong, Han Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18093)  

**Abstract**: Applying large language models (LLMs) as teaching assists has attracted much attention as an integral part of intelligent education, particularly in computing courses. To reduce the gap between the LLMs and the computer programming education expert, fine-tuning and retrieval augmented generation (RAG) are the two mainstream methods in existing researches. However, fine-tuning for specific tasks is resource-intensive and may diminish the model`s generalization capabilities. RAG can perform well on reducing the illusion of LLMs, but the generation of irrelevant factual content during reasoning can cause significant confusion for learners. To address these problems, we introduce the Molly agent, focusing on solving the proposed problem encountered by learners when learning Python programming language. Our agent automatically parse the learners' questioning intent through a scenario-based interaction, enabling precise retrieval of relevant documents from the constructed knowledge base. At generation stage, the agent reflect on the generated responses to ensure that they not only align with factual content but also effectively answer the user's queries. Extensive experimentation on a constructed Chinese Python QA dataset shows the effectiveness of the Molly agent, indicating an enhancement in its performance for providing useful responses to Python questions. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

将大型语言模型（LLMs）用作教学助手，作为智能教育，尤其是在计算课程中的一项重要组成部分，已经引起了广泛关注。为减少LLMs与计算机编程教育专家之间的差距，现有的研究主要采用了微调和检索增强生成（RAG）两种方法。然而，针对特定任务的微调耗时且可能削弱模型的一般化能力。RAG 可在减少LLMs的幻觉方面表现出色，但在推理过程中生成的相关事实内容不足可能导致学习者的困惑。为解决这些问题，我们引入了Molly代理，专注于解决学习Python编程语言过程中学习者遇到的问题。我们的代理通过基于场景的交互自动解析学习者的提问意图，从而从构建的知识库中精确检索相关文档。在生成阶段，代理反思生成的回答，以确保它们不仅与事实内容一致，还能有效地回答用户的问题。通过对构建的中文Python问答数据集进行广泛的实验，展示了Molly代理的有效性，表明其性能在提供有用的Python问题回答方面有所提升。 

---
# Contrastive Representation for Interactive Recommendation 

**Title (ZH)**: 交互推荐的对比表示方法 

**Authors**: Jingyu Li, Zhiyong Feng, Dongxiao He, Hongqi Chen, Qinghang Gao, Guoli Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18396)  

**Abstract**: Interactive Recommendation (IR) has gained significant attention recently for its capability to quickly capture dynamic interest and optimize both short and long term objectives. IR agents are typically implemented through Deep Reinforcement Learning (DRL), because DRL is inherently compatible with the dynamic nature of IR. However, DRL is currently not perfect for IR. Due to the large action space and sample inefficiency problem, training DRL recommender agents is challenging. The key point is that useful features cannot be extracted as high-quality representations for the recommender agent to optimize its policy. To tackle this problem, we propose Contrastive Representation for Interactive Recommendation (CRIR). CRIR efficiently extracts latent, high-level preference ranking features from explicit interaction, and leverages the features to enhance users' representation. Specifically, the CRIR provides representation through one representation network, and refines it through our proposed Preference Ranking Contrastive Learning (PRCL). The key insight of PRCL is that it can perform contrastive learning without relying on computations involving high-level representations or large potential action sets. Furthermore, we also propose a data exploiting mechanism and an agent training mechanism to better adapt CRIR to the DRL backbone. Extensive experiments have been carried out to show our method's superior improvement on the sample efficiency while training an DRL-based IR agent. 

**Abstract (ZH)**: 交互推荐（IR）最近受到了广泛关注，因为其能够迅速捕捉动态兴趣，并优化短期和长期目标。IR代理通常通过深度强化学习（DRL）实现，因为DRL天生与IR的动态性质相兼容。然而，DRL目前尚未完全适用于IR。由于动作空间庞大和样本效率问题，训练DRL推荐代理极具挑战性。关键问题是，有用的特征无法被提取为高质量的表示，以优化推荐代理的策略。为解决这一问题，我们提出了对比表示的交互推荐（CRIR）。CRIR有效从显式交互中抽取潜在的高度偏好排序特征，并利用这些特征增强用户的表示。具体而言，CRIR通过一个表示网络提供表示，并通过我们提出的偏好排序对比学习（PRCL）进行细化。PRCL的关键洞察是，它可以在不依赖于涉及高级表示或大规模潜在动作集的计算的情况下执行对比学习。此外，我们还提出了一种数据利用机制和代理训练机制，以更好地使CRIR适应DRL框架。我们进行了大量实验，证明了CRIR在训练基于DRL的IR代理时在样本效率方面具有显著的改进优势。 

---
