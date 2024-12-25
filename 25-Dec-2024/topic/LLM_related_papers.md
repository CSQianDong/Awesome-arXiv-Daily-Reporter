# Explainable Multi-Modal Data Exploration in Natural Language via LLM Agent 

**Title (ZH)**: 通过大规模语言模型代理进行可解释的多模态数据语言探索 

**Authors**: Farhad Nooralahzadeh, Yi Zhang, Jonathan Furst, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.18428)  

**Abstract**: International enterprises, organizations, or hospitals collect large amounts of multi-modal data stored in databases, text documents, images, and videos. While there has been recent progress in the separate fields of multi-modal data exploration as well as in database systems that automatically translate natural language questions to database query languages, the research challenge of querying database systems combined with other unstructured modalities such as images in natural language is widely unexplored.
In this paper, we propose XMODE - a system that enables explainable, multi-modal data exploration in natural language. Our approach is based on the following research contributions: (1) Our system is inspired by a real-world use case that enables users to explore multi-modal information systems. (2) XMODE leverages a LLM-based agentic AI framework to decompose a natural language question into subtasks such as text-to-SQL generation and image analysis. (3) Experimental results on multi-modal datasets over relational data and images demonstrate that our system outperforms state-of-the-art multi-modal exploration systems, excelling not only in accuracy but also in various performance metrics such as query latency, API costs, planning efficiency, and explanation quality, thanks to the more effective utilization of the reasoning capabilities of LLMs. 

**Abstract (ZH)**: 国际企业、组织或医院收集了大量的多模态数据，这些数据存储在数据库、文本文件、图像和视频中。虽然在多模态数据探索和自动将自然语言问题转换为数据库查询语言的数据库系统方面已经取得了进展，但将数据库系统与其他未结构化的模态（如图像）结合起来使用自然语言进行查询的研究挑战仍未得到广泛探索。

本文提出了一种名为XMODE的系统，该系统能够使用自然语言进行可解释的多模态数据探索。我们的方法基于以下几个研究贡献：（1）我们的系统受到了实际应用场景的启发，使得用户能够探索多模态信息系统。（2）XMODE利用基于大模型的代理型AI框架，将自然语言问题分解为子任务，例如文本到SQL生成和图像分析。（3）在关系数据和图像的多模态数据集上的实验结果表明，我们的系统在准确性和查询延迟、API费用、规划效率和解释质量等多个性能指标上都明显优于现有的多模态探索系统，这得益于对大模型推理能力更有效的利用。 

---
# Improving Multi-Step Reasoning Abilities of Large Language Models with Direct Advantage Policy Optimization 

**Title (ZH)**: 利用直接优势策略优化提升大型语言模型的多步推理能力 

**Authors**: Jiacai Liu, Chaojie Wang, Chris Yuhao Liu, Liang Zeng, Rui Yan, Yiwen Sun, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.18279)  

**Abstract**: The role of reinforcement learning (RL) in enhancing the reasoning of large language models (LLMs) is becoming increasingly significant. Despite the success of RL in many scenarios, there are still many challenges in improving the reasoning of LLMs. One challenge is the sparse reward, which makes optimization difficult for RL and necessitates a large amount of data samples. Another challenge stems from the inherent instability of RL, particularly when using Actor-Critic (AC) methods to derive optimal policies, which often leads to unstable training processes. To address these issues, we introduce Direct Advantage Policy Optimization (DAPO), an novel step-level offline RL algorithm. Unlike standard alignment that rely solely outcome rewards to optimize policies (such as DPO), DAPO employs a critic function to predict the reasoning accuracy at each step, thereby generating dense signals to refine the generation strategy. Additionally, the Actor and Critic components in DAPO are trained independently, avoiding the co-training instability observed in standard AC algorithms like PPO. We train DAPO on mathematical and code query datasets and then evaluate its performance on multiple benchmarks. Our results show that DAPO can effectively enhance the mathematical and code capabilities on both SFT models and RL models, demonstrating the effectiveness of DAPO. 

**Abstract (ZH)**: 强化学习（RL）在提高大规模语言模型（LLMs）推理能力方面的作用日益重要。尽管在许多场景中RL取得了成功，但在提高LLMs推理能力方面仍存在许多挑战。其中一个挑战是稀疏奖励，这使得RL的优化变得困难，需要大量的数据样本。另一个挑战来源于RL固有的不稳定性，尤其是在使用Actor-Critic（AC）方法来推导最优策略时，通常会导致训练过程不稳定。为了解决这些问题，我们提出了一种新颖的离线RL算法——直接优势策略优化（DAPO）。与仅依赖于结果奖励来优化策略的标准对齐方法（如DPO）不同，DAPO使用了一个评论家函数来预测每一步的推理准确性，从而生成密集信号以改进生成策略。此外，DAPO中的Actor和 Critic组件是独立训练的，从而避免了标准AC算法（如PPO）中出现的协同训练不稳定性。我们使用数学和代码查询数据集训练DAPO，并在多个基准测试中评估其性能。结果显示，DAPO可以有效地增强两者即自适应训练模型（SFT）和RL模型的数学和代码能力，证明了DAPO的有效性。 

---
# Annotating References to Mythological Entities in French Literature 

**Title (ZH)**: 法国文学中神话实体的标注研究 

**Authors**: Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2412.18270)  

**Abstract**: In this paper, we explore the relevance of large language models (LLMs) for annotating references to Roman and Greek mythological entities in modern and contemporary French literature. We present an annotation scheme and demonstrate that recent LLMs can be directly applied to follow this scheme effectively, although not without occasionally making significant analytical errors. Additionally, we show that LLMs (and, more specifically, ChatGPT) are capable of offering interpretative insights into the use of mythological references by literary authors. However, we also find that LLMs struggle to accurately identify relevant passages in novels (when used as an information retrieval engine), often hallucinating and generating fabricated examples-an issue that raises significant ethical concerns. Nonetheless, when used carefully, LLMs remain valuable tools for performing annotations with high accuracy, especially for tasks that would be difficult to annotate comprehensively on a large scale through manual methods alone. 

**Abstract (ZH)**: 在本文中，我们探讨了大规模语言模型（LLMs）在标注现代和当代法语文献中关于罗马和希腊神话实体的引用方面的相关性。我们呈现了一种标注方案，并证明了最近的LLMs可以直接应用于遵循该方案，尽管偶尔会产生显著的分析错误。此外，我们还展示了LLMs（更具体地说是ChatGPT）能够为文学作者在使用神话引用方面提供解释性的见解。然而，我们发现，当将LLMs用作信息检索引擎时，它们在准确识别小说中的相关段落方面存在困难，经常会产生幻觉并生成虚构的示例—这一问题引发了重大的伦理关切。尽管如此，只要谨慎使用，LLMs仍然是在大规模手动方法难以全面标注的任务中进行标注的有价值的工具，尤其是对于那些能够实现高精度标注的任务。 

---
# VISION: A Modular AI Assistant for Natural Human-Instrument Interaction at Scientific User Facilities 

**Title (ZH)**: VISION：一种适用于科学用户设施的模块化人工智能助手，用于自然的人-仪器交互 

**Authors**: Shray Mathur, Noah van der Vleuten, Kevin Yager, Esther Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2412.18161)  

**Abstract**: Scientific user facilities, such as synchrotron beamlines, are equipped with a wide array of hardware and software tools that require a codebase for human-computer-interaction. This often necessitates developers to be involved to establish connection between users/researchers and the complex instrumentation. The advent of generative AI presents an opportunity to bridge this knowledge gap, enabling seamless communication and efficient experimental workflows. Here we present a modular architecture for the Virtual Scientific Companion (VISION) by assembling multiple AI-enabled cognitive blocks that each scaffolds large language models (LLMs) for a specialized task. With VISION, we performed LLM-based operation on the beamline workstation with low latency and demonstrated the first voice-controlled experiment at an X-ray scattering beamline. The modular and scalable architecture allows for easy adaptation to new instrument and capabilities. Development on natural language-based scientific experimentation is a building block for an impending future where a science exocortex -- a synthetic extension to the cognition of scientists -- may radically transform scientific practice and discovery. 

**Abstract (ZH)**: 科学用户设施，如同步辐射光束线，配备了大量硬件和软件工具，需要代码库来支持人机交互。这通常需要开发人员参与，以建立用户/研究人员与复杂仪器之间的连接。随着生成式AI的发展，为弥补这一知识差距提供了机会，促进了无缝通信和高效的实验流程。本文提出了一种模块化的虚拟科学伴侣（VISION）架构，通过集合多个AI-enabled认知模块，为特定任务搭建大型语言模型（LLMs）。利用VISION，我们实现了基于LLM的光束线工作站操作，并展示了首个语音控制的X射线散射实验。这种模块化和可扩展的架构允许轻松适应新的仪器和功能。基于自然语言的科学实验开发是向着一个即将到来的未来发展的基石，即科学外脑——一种合成扩展科学家认知能力的技术——可能对科学研究和发现产生根本性的影响。 

---
# AIGT: AI Generative Table Based on Prompt 

**Title (ZH)**: AIGT：基于提示的AI生成表格 

**Authors**: Mingming Zhang, Zhiqing Xiao, Guoshan Lu, Sai Wu, Weiqiang Wang, Xing Fu, Can Yi, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18111)  

**Abstract**: Tabular data, which accounts for over 80% of enterprise data assets, is vital in various fields. With growing concerns about privacy protection and data-sharing restrictions, generating high-quality synthetic tabular data has become essential. Recent advancements show that large language models (LLMs) can effectively gener-ate realistic tabular data by leveraging semantic information and overcoming the challenges of high-dimensional data that arise from one-hot encoding. However, current methods do not fully utilize the rich information available in tables. To address this, we introduce AI Generative Table (AIGT) based on prompt enhancement, a novel approach that utilizes meta data information, such as table descriptions and schemas, as prompts to generate ultra-high quality synthetic data. To overcome the token limit constraints of LLMs, we propose long-token partitioning algorithms that enable AIGT to model tables of any scale. AIGT achieves state-of-the-art performance on 14 out of 20 public datasets and two real industry datasets within the Alipay risk control system. 

**Abstract (ZH)**: 表格数据占企业数据资产的80%以上，在多个领域中至关重要。随着对隐私保护和数据共享限制的关注不断增加，生成高质量的合成表格数据变得至关重要。最近的研究表明，大型语言模型（LLMs）可以通过利用语义信息并克服基于独热编码的一维高维数据带来的挑战，有效地生成逼真的表格数据。然而，当前的方法并未充分利用表格中丰富的信息。为此，我们基于提示增强引入了AI生成表格（AIGT）方法，这是一种新颖的方法，利用表描述和表结构等元数据信息作为提示来生成超高质量的合成数据。为克服LLMs的令牌限制约束，我们提出了长令牌分割算法，使AIGT能够建模任意规模的表格。在20个公开数据集的14个以及两个实际行业数据集（来自支付宝风控系统）中，AIGT达到了目前的最优性能。 

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
# How Well Do LLMs Generate Code for Different Application Domains? Benchmark and Evaluation 

**Title (ZH)**: 不同应用领域中大语言模型生成代码的效果如何？基准测试与评估 

**Authors**: Dewu Zheng, Yanlin Wang, Ensheng Shi, Hongyu Zhang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.18573)  

**Abstract**: Recently, an increasing number of AI-driven programming assistants powered by code LLMs have been integrated into various real-world software development environments, significantly boosting developer productivity. However, existing code generation benchmarks primarily focus on general-purpose scenarios, leaving the code generation performance of LLMs for specific application domains largely unknown. In this paper, we introduce a new benchmark, MultiCodeBench, to fill this gap. MultiCodeBench comprises 2,400 programming tasks, covering 12 popular software development domains and 15 programming languages. Specifically, we perform in-depth research to identify these 12 application domains. Given that each domain may involve multiple technical frameworks, and that different frameworks present distinct challenges in the coding process, we categorize the commonly used frameworks and platforms within each domain. We then sample programming problems from GitHub repositories related to these subdomains. To ensure the quality of the tasks and mitigate data leakage issues, we invite annotators to rewrite the docstrings for each task in MultiCodeBench. Additionally, we build a static analysis-based dependency parsing tool to extract the dependencies in the ground truth for each task, enabling deeper performance analysis. Through extensive experiments on MultiCodeBench with eleven representative mainstream LLMs, we reveal the code generation performance of the LLMs across different application domains, providing practical insights for developers in downstream fields when selecting LLMs. Furthermore, we analyze the reasons behind the models' failures in completing software application development tasks, offering guidance for model developers to enhance domain-specific code generation capabilities. 

**Abstract (ZH)**: 近年来，越来越多由代码LLM驱动的AI编程助手被集成到各种实际软件开发环境中，显著提升了开发者的生产力。然而，现有的代码生成基准主要关注通用场景，使得LLM在特定应用领域中的代码生成性能仍然知之甚少。在本文中，我们介绍了一个新的基准——MultiCodeBench，以填补这一空白。MultiCodeBench 包含2,400个编程任务，涵盖了12个流行的软件开发领域和15种编程语言。具体而言，我们深入研究了这些12个应用领域的选择。鉴于每个领域可能涉及多种技术框架，而不同框架在编码过程中会带来不同的挑战，我们对每个领域的常用框架和平台进行了分类。然后，我们从与这些子领域相关的GitHub仓库中抽取编程问题。为了确保任务的质量并减少数据泄露的问题，我们邀请注释者为MultiCodeBench中的每个任务重写文档字符串。此外，我们建立了一种基于静态分析的依赖关系解析工具，以提取每个任务的真实依赖关系，从而实现更深入的性能分析。通过在MultiCodeBench上对11种主流的代表LLM进行广泛的实验，我们揭示了LLM在不同应用领域的代码生成性能，为下游开发者在选择LLM时提供了实用的洞察。进一步地，我们分析了模型在完成软件应用开发任务时失败的原因，为模型开发者提供指导以增强领域特定的代码生成能力。 

---
# Token-Budget-Aware LLM Reasoning 

**Title (ZH)**: 面向token预算的大型语言模型推理 

**Authors**: Tingxu Han, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen, Zhenting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18547)  

**Abstract**: Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework, which dynamically estimates token budgets for different problems based on reasoning complexity and uses the estimated token budgets to guide the reasoning process. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: this https URL. 

**Abstract (ZH)**: 推理对于大型语言模型（LLMs）在广泛任务中的卓越表现至关重要。虽然像逐步推理（Chain-of-Thought, CoT）这样的方法通过将问题分解为中间步骤来增强LLM的性能，但这也带来了显著的令牌使用量 overhead，从而增加了成本。我们发现当前LLMs的推理过程过长，可以通过在提示中包含合理的令牌预算来压缩推理过程，但令牌预算的选择在实际压缩效果中起着关键作用。随后，我们提出了一种意识令牌预算的LLM推理框架，该框架根据推理复杂度动态估计不同问题的令牌预算，并使用估算的令牌预算来引导推理过程。实验结果显示，我们的方法在CoT推理中有效降低了令牌成本，仅轻微降低了性能，提供了一种平衡效率和准确性的实用解决方案。代码：[这里提供链接] 

---
# Consistency Checks for Language Model Forecasters 

**Title (ZH)**: 语言模型预测器的一致性检查 

**Authors**: Daniel Paleka, Abhimanyu Pallavi Sudhir, Alejandro Alvarez, Vineeth Bhat, Adam Shen, Evan Wang, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2412.18544)  

**Abstract**: Forecasting is a task that is difficult to evaluate: the ground truth can only be known in the future. Recent work showing LLM forecasters rapidly approaching human-level performance begs the question: how can we benchmark and evaluate these forecasters instantaneously? Following the consistency check framework, we measure the performance of forecasters in terms of the consistency of their predictions on different logically-related questions. We propose a new, general consistency metric based on arbitrage: for example, if a forecasting AI illogically predicts that both the Democratic and Republican parties have 60% probability of winning the 2024 US presidential election, an arbitrageur can trade against the forecaster's predictions and make a profit. We build an automated evaluation system that generates a set of base questions, instantiates consistency checks from these questions, elicits the predictions of the forecaster, and measures the consistency of the predictions. We then build a standard, proper-scoring-rule forecasting benchmark, and show that our (instantaneous) consistency metrics correlate with LLM forecasters' ground truth Brier scores (which are only known in the future). We also release a consistency benchmark that resolves in 2028, providing a long-term evaluation tool for forecasting. 

**Abstract (ZH)**: 预测是一项难以评估的任务：真实情况只有在未来才能得知。最近的研究表明，大型语言模型（LLM）预测器正在迅速接近人类水平的表现，这引发了另一个问题：我们如何能够即时地对这些预测器进行基准测试和评估？我们遵循一致性检验框架，衡量预测器在其对不同逻辑相关问题的预测中的一致性。我们提出了一种基于套利的新的一般一致性度量方法：例如，如果一个预测AI逻辑上错误地预测民主党与共和党在2024年美国总统大选中获胜的概率均为60%，那么套利者可以通过对预测的反向操作获利。我们构建了一个自动评估系统，该系统生成一组基础问题，基于这些问题实例化一致性检验，征求预测器的预测，并衡量这些预测的一致性。然后，我们构建了一个标准的、符合评分规则的预测基准，并展示了我们的一致性度量指标与LLM预测器的未来已知真实布雷尔评分（Brier scores）之间的相关性。此外，我们还发布了一个将在2028年揭晓的一致性基准，为预测提供了一个长期评估工具。 

---
# Multilingual Mathematical Reasoning: Advancing Open-Source LLMs in Hindi and English 

**Title (ZH)**: 多语言数学推理：推进印地语和英语开源大语言模型的发展 

**Authors**: Avinash Anand, Kritarth Prasad, Chhavi Kirtani, Ashwin R Nair, Manvendra Kumar Nema, Raj Jaiswal, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2412.18415)  

**Abstract**: Large Language Models (LLMs) excel in linguistic tasks but struggle with mathematical reasoning, particularly in non English languages like Hindi. This research aims to enhance the mathematical reasoning skills of smaller, resource efficient open-source LLMs in both Hindi and English. We evaluate models like OpenHathi 7B, LLaMA-2 7B, WizardMath 7B, Mistral 7B, LLeMMa 7B, MAmmoTH 7B, Gemini Pro, and GPT-4 using zero-shot, few-shot chain-of-thought (CoT) methods, and supervised fine-tuning. Our approach incorporates curriculum learning, progressively training models on increasingly difficult problems, a novel Decomposition Strategy to simplify complex arithmetic operations, and a Structured Solution Design that divides solutions into phases. Our experiments result in notable performance enhancements. WizardMath 7B exceeds Gemini's accuracy on English datasets by +6% and matches Gemini's performance on Hindi datasets. Adopting a bilingual approach that combines English and Hindi samples achieves results comparable to individual language models, demonstrating the capability to learn mathematical reasoning in both languages. This research highlights the potential for improving mathematical reasoning in open-source LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言任务上表现出色，但在数学推理方面存在困难，尤其是在非英语语言如印地语中。本研究旨在增强较小的、资源高效的开源LLMs在印地语和英语中的数学推理能力。我们使用零样本、少量样本的思考链（CoT）方法和监督微调来评估OpenHathi 7B、LLaMA-2 7B、WizardMath 7B、Mistral 7B、LLeMMa 7B、MAmmoTH 7B、Gemini Pro和GPT-4等模型。我们的方法结合了逐渐学习，通过逐步训练模型解决越来越难的问题，提出了一种新颖的分解策略来简化复杂的算术运算，并采用结构化解决方案设计将解决方案划分为多个阶段。实验结果显示显著的性能提升。WizardMath 7B在英语数据集上的准确率比Gemini高出6%，在印地语数据集上与其性能相当。采用双语方法结合英语和印地语样本，其结果与单一语言模型相当，表明模型可以在两种语言中学习数学推理的能力。本研究突显了提高开源LLMs的数学推理能力的潜力。 

---
# A Statistical Framework for Ranking LLM-Based Chatbots 

**Title (ZH)**: 基于统计框架的大型语言模型驱动的聊天机器人排名方法 

**Authors**: Siavash Ameli, Siyuan Zhuang, Ion Stoica, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2412.18407)  

**Abstract**: Large language models (LLMs) have transformed natural language processing, with frameworks like Chatbot Arena providing pioneering platforms for evaluating these models. By facilitating millions of pairwise comparisons based on human judgments, Chatbot Arena has become a cornerstone in LLM evaluation, offering rich datasets for ranking models in open-ended conversational tasks. Building upon this foundation, we propose a statistical framework that incorporates key advancements to address specific challenges in pairwise comparison analysis. First, we introduce a factored tie model that enhances the ability to handle ties -- an integral aspect of human-judged comparisons -- significantly improving the model's fit to observed data. Second, we extend the framework to model covariance between competitors, enabling deeper insights into performance relationships and facilitating intuitive groupings into performance tiers. Third, we resolve optimization challenges arising from parameter non-uniqueness by introducing novel constraints, ensuring stable and interpretable parameter estimation. Through rigorous evaluation and extensive experimentation, our framework demonstrates substantial improvements over existing methods in modeling pairwise comparison data. To support reproducibility and practical adoption, we release leaderbot, an open-source Python package implementing our models and analyses. 

**Abstract (ZH)**: 大语言模型（LLMs）已经彻底改变了自然语言处理领域，而类似Chatbot Arena这样的框架提供了评估这些模型的先驱平台。通过基于人类判断进行数百万次的成对比较，Chatbot Arena已成为LLM评估领域的重要基石，提供了丰富的数据集，用于评估模型在开放对话任务中的表现。在此基础上，我们提出了一种统计框架，结合关键的最新进展来应对成对比较分析中的特定挑战。首先，我们引入了一个事实上的平局模型，增强了处理平局——人类判断对比中的关键方面——的能力，显著提升了模型拟合观察数据的效果。其次，我们扩展了框架以建模竞争者之间的协方差，这有助于更深入地理解性能关系，并促进对性能层级的直观分组。第三，我们通过引入新颖的约束条件解决了由参数非唯一性引起的优化挑战，从而确保了参数估计的稳定性和可解释性。通过严格的评估和广泛的实验，我们的框架在建模成对比较数据方面展示了对现有方法的显著改进。为了支持可再现性和实际应用，我们发布了领导者机器人（leaderbot），这是一个开源的Python软件包，实现了我们的模型和分析。 

---
# ChaI-TeA: A Benchmark for Evaluating Autocompletion of Interactions with LLM-based Chatbots 

**Title (ZH)**: ChaI-TeA：基于大语言模型的聊天机器人互动自动补全评估基准 

**Authors**: Shani Goren, Oren Kalinsky, Tomer Stav, Yuri Rapoport, Yaron Fairstein, Ram Yazdy, Nachshon Cohen, Alexander Libov, Guy Kushilevitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.18377)  

**Abstract**: The rise of LLMs has deflected a growing portion of human-computer interactions towards LLM-based chatbots. The remarkable abilities of these models allow users to interact using long, diverse natural language text covering a wide range of topics and styles. Phrasing these messages is a time and effort consuming task, calling for an autocomplete solution to assist users. We introduce the task of chatbot interaction autocomplete. We present ChaI-TeA: CHat InTEraction Autocomplete; An autcomplete evaluation framework for LLM-based chatbot interactions. The framework includes a formal definition of the task, coupled with suitable datasets and metrics. We use the framework to evaluate After formally defining the task along with suitable datasets and metrics, we test 9 models on the defined auto completion task, finding that while current off-the-shelf models perform fairly, there is still much room for improvement, mainly in ranking of the generated suggestions. We provide insights for practitioners working on this task and open new research directions for researchers in the field. We release our framework to serve as a foundation for future research. 

**Abstract (ZH)**: 大规模语言模型（LLM）的兴起正逐渐将越来越多的人机交互转向基于LLM的聊天机器人。这些模型的强大能力使用户能够使用长篇且多样化的自然语言文本，涵盖广泛的主题和风格进行交互。撰写这些消息是一个既耗费时间又费力的任务，因此需要一种自动补全解决方案来辅助用户。我们提出了聊天机器人交互自动补全任务。我们提出ChaI-TeA：聊天互动自动补全；一种用于基于LLM的聊天机器人交互的自动补全评估框架。该框架包括对该任务的正式定义，以及配套的数据集和评估指标。我们使用该框架来评估任务，并测试了9个模型在定义的自动补全任务上的表现，发现尽管现有的现成模型表现尚可，但在生成的建议排序上仍有很大的改进空间。我们为从事该任务的实践者提供了见解，并为该领域的研究人员开辟了新的研究方向。我们发布该框架，作为未来研究的基础。 

---
# Multi-Agents Based on Large Language Models for Knowledge-based Visual Question Answering 

**Title (ZH)**: 基于大型语言模型的多agents知识驱动视觉问答系统 

**Authors**: Zhongjian Hu, Peng Yang, Bing Li, Zhenqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18351)  

**Abstract**: Large Language Models (LLMs) have achieved impressive results in knowledge-based Visual Question Answering (VQA). However existing methods still have challenges: the inability to use external tools autonomously, and the inability to work in teams. Humans tend to know whether they need to use external tools when they encounter a new question, e.g., they tend to be able to give a direct answer to a familiar question, whereas they tend to use tools such as search engines when they encounter an unfamiliar question. In addition, humans also tend to collaborate and discuss with others to get better answers. Inspired by this, we propose the multi-agent voting framework. We design three LLM-based agents that simulate different levels of staff in a team, and assign the available tools according to the levels. Each agent provides the corresponding answer, and finally all the answers provided by the agents are voted to get the final answer. Experiments on OK-VQA and A-OKVQA show that our approach outperforms other baselines by 2.2 and 1.0, respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在基于知识的视觉问答（VQA）任务上取得了显著成果。然而，现有的方法仍然存在一些挑战：自主使用外部工具的能力不足，以及无法团队协作工作。人类在遇到新问题时通常能判断是否需要使用外部工具，例如，他们往往能直接回答熟悉的问题，而遇到不熟悉的问题时会使用诸如搜索引擎等工具。此外，人类还倾向于与他人协作讨论以获得更好的答案。受此启发，我们提出了多Agent投票框架。我们设计了三个基于LLM的Agent，模拟团队中不同级别的工作人员，并根据其级别分配可用的工具。每个Agent提供相应的答案，最后通过投票将所有Agent提供的答案结合以获得最终答案。实验结果表明，我们的方法在OK-VQA和A-OKVQA数据集上分别优于其他基线方法2.2和1.0的分数。 

---
# Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search 

**Title (ZH)**: 桑葚: 通过集体蒙特卡洛树搜索增强MLLM的o1-like推理与反思能力 

**Authors**: Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18319)  

**Abstract**: In this work, we aim to develop an MLLM that understands and solves questions by learning to create each intermediate step of the reasoning involved till the final answer. To this end, we propose Collective Monte Carlo Tree Search (CoMCTS), a new learning-to-reason method for MLLMs, which introduces the concept of collective learning into ``tree search'' for effective and efficient reasoning-path searching and learning. The core idea of CoMCTS is to leverage collective knowledge from multiple models to collaboratively conjecture, search and identify effective reasoning paths toward correct answers via four iterative operations including Expansion, Simulation and Error Positioning, Backpropagation, and Selection. Using CoMCTS, we construct Mulberry-260k, a multimodal dataset with a tree of rich, explicit and well-defined reasoning nodes for each question. With Mulberry-260k, we perform collective SFT to train our model, Mulberry, a series of MLLMs with o1-like step-by-step Reasoning and Reflection capabilities. Extensive experiments demonstrate the superiority of our proposed methods on various benchmarks. Code will be available at this https URL 

**Abstract (ZH)**: 在本项工作中，我们旨在开发一种多模块语言模型（MLLM），使其能够通过学习生成推理过程中的每个中间步骤，直至最终答案来理解并解决问题。为此，我们提出了集体蒙特卡洛树搜索（CoMCTS），这是一种新的MLLM推理学习方法，将“树搜索”中的集体学习概念引入，以实现更有效的推理路径搜索和学习。CoMCTS的核心思想是利用多个模型的集体知识，通过扩张、模拟和错误定位、反向传播和选择等四个迭代操作，协作地猜测、搜索和识别通向正确答案的有效推理路径。

利用CoMCTS，我们构建了Mulberry-260k，这是一个包含每个问题丰富的明确和规范推理节点的多模态数据集。通过Mulberry-260k，我们进行集体监督 fine-tuning (SFT) 来训练我们的模型——具有类似o1的逐步推理和反思能力的一系列MLLMs，即Mulberry。广泛的实验表明，我们提出的方法在各种基准测试中具有优越性。代码将在以下链接提供：[请在此插入URL] 

---
# M-Ped: Multi-Prompt Ensemble Decoding for Large Language Models 

**Title (ZH)**: M-Ped：面向大规模语言模型的多提示集成解码 

**Authors**: Jiaxin Guo, Daimeng Wei, Yuanchang Luo, Shimin Tao, Hengchao Shang, Zongyao Li, Shaojun Li, Jinlong Yang, Zhanglin Wu, Zhiqiang Rao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18299)  

**Abstract**: With the widespread application of Large Language Models (LLMs) in the field of Natural Language Processing (NLP), enhancing their performance has become a research hotspot. This paper presents a novel multi-prompt ensemble decoding approach designed to bolster the generation quality of LLMs by leveraging the aggregation of outcomes from multiple prompts. Given a unique input $X$, we submit $n$ variations of prompts with $X$ to LLMs in batch mode to decode and derive probability distributions. For each token prediction, we calculate the ensemble probability by averaging the $n$ probability distributions within the batch, utilizing this aggregated probability to generate the token. This technique is dubbed Inner-Batch Ensemble. To facilitate efficient batch inference, we implement a Left-Padding strategy to maintain uniform input lengths across the n prompts. Through extensive experimentation on diverse NLP tasks, including machine translation, code generation, and text simplification, we demonstrate the efficacy of our method in enhancing LLM performance. The results show substantial improvements in BLEU scores, pass@$k$ rates, and LENS metrics over conventional methods. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）在自然语言处理（NLP）领域的广泛应用，提高其性能已成为研究的热点。本文提出了一种新颖的多提示集ensemble解码方法，该方法通过聚合多个提示的输出结果，以增强LLMs的生成质量。对于给定的唯一输入$X$，我们将$X$的$n$个不同版本的提示提交给LLMs进行批量解码，并推导出概率分布。对于每个令牌预测，我们通过计算批量内的$n$个概率分布的平均值来计算集成概率，并利用该集成概率生成令牌。该方法称为内部批量ensemble。为了促进高效的批量推理，我们采用左填充策略来保持$n$个提示之间输入长度的一致性。通过在机器翻译、代码生成和文本简化等多种NLP任务上的广泛实验，我们展示了该方法在提高LLMs性能方面的有效性。结果表明，该方法在BLEU分数、pass@$k$率和LENS度量方面比传统方法有显著提高。 

---
# DeepCRCEval: Revisiting the Evaluation of Code Review Comment Generation 

**Title (ZH)**: DeepCRCEval：重新审视代码审查评论生成的评估方法 

**Authors**: Junyi Lu, Xiaojia Li, Zihan Hua, Lei Yu, Shiqi Cheng, Li Yang, Fengjun Zhang, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2412.18291)  

**Abstract**: Code review is a vital but demanding aspect of software development, generating significant interest in automating review comments. Traditional evaluation methods for these comments, primarily based on text similarity, face two major challenges: inconsistent reliability of human-authored comments in open-source projects and the weak correlation of text similarity with objectives like enhancing code quality and detecting defects.
This study empirically analyzes benchmark comments using a novel set of criteria informed by prior research and developer interviews. We then similarly revisit the evaluation of existing methodologies. Our evaluation framework, DeepCRCEval, integrates human evaluators and Large Language Models (LLMs) for a comprehensive reassessment of current techniques based on the criteria set. Besides, we also introduce an innovative and efficient baseline, LLM-Reviewer, leveraging the few-shot learning capabilities of LLMs for a target-oriented comparison.
Our research highlights the limitations of text similarity metrics, finding that less than 10% of benchmark comments are high quality for automation. In contrast, DeepCRCEval effectively distinguishes between high and low-quality comments, proving to be a more reliable evaluation mechanism. Incorporating LLM evaluators into DeepCRCEval significantly boosts efficiency, reducing time and cost by 88.78% and 90.32%, respectively. Furthermore, LLM-Reviewer demonstrates significant potential of focusing task real targets in comment generation. 

**Abstract (ZH)**: 代码审查是软件开发中一个重要但富有挑战性的环节，引起了对自动化审查评论的广泛关注。传统对这些评论的评估方法，主要基于文本相似性，面临两大主要挑战：开源项目中人工编写的评论的一致性和可靠性不足，以及文本相似性与提升代码质量、检测缺陷等目标之间的弱关联性。

本研究通过采用一套新的评价标准，结合先前研究和开发者访谈的结果进行实证分析，重新评估现有的评估方法。我们的评估框架DeepCRCEval结合了人力评估者和大型语言模型（LLM），基于提出的标准对现有技术进行全面的重新评估。此外，我们还引入了一种创新且高效的基线模型LLM-Reviewer，利用LLM的少量示例学习能力进行目标导向的比较。

我们的研究揭示了文本相似性度量的局限性，发现基准评论中仅有不到10%的评论适合自动化。相比之下，DeepCRCEval能够有效地区分高质量和低质量的评论，证明了其作为更可靠的评估机制的有效性。将LLM评估者纳入DeepCRCEval显著提高了效率，分别降低了88.78%的时间和90.32%的成本。此外，LLM-Reviewer展示了在评论生成中集中目标任务的巨大潜力。 

---
# An Automatic Graph Construction Framework based on Large Language Models for Recommendation 

**Title (ZH)**: 基于大型语言模型的自动图构建推荐框架 

**Authors**: Rong Shan, Jianghao Lin, Chenxu Zhu, Bo Chen, Menghui Zhu, Kangning Zhang, Jieming Zhu, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18241)  

**Abstract**: Graph neural networks (GNNs) have emerged as state-of-the-art methods to learn from graph-structured data for recommendation. However, most existing GNN-based recommendation methods focus on the optimization of model structures and learning strategies based on pre-defined graphs, neglecting the importance of the graph construction stage. Earlier works for graph construction usually rely on speciffic rules or crowdsourcing, which are either too simplistic or too labor-intensive. Recent works start to utilize large language models (LLMs) to automate the graph construction, in view of their abundant open-world knowledge and remarkable reasoning capabilities. Nevertheless, they generally suffer from two limitations: (1) invisibility of global view (e.g., overlooking contextual information) and (2) construction inefficiency. To this end, we introduce AutoGraph, an automatic graph construction framework based on LLMs for recommendation. Specifically, we first use LLMs to infer the user preference and item knowledge, which is encoded as semantic vectors. Next, we employ vector quantization to extract the latent factors from the semantic vectors. The latent factors are then incorporated as extra nodes to link the user/item nodes, resulting in a graph with in-depth global-view semantics. We further design metapath-based message aggregation to effectively aggregate the semantic and collaborative information. The framework is model-agnostic and compatible with different backbone models. Extensive experiments on three real-world datasets demonstrate the efficacy and efffciency of AutoGraph compared to existing baseline methods. We have deployed AutoGraph in Huawei advertising platform, and gain a 2.69% improvement on RPM and a 7.31% improvement on eCPM in the online A/B test. Currently AutoGraph has been used as the main trafffc model, serving hundreds of millions of people. 

**Abstract (ZH)**: 图神经网络（GNNs）已经成了从图结构数据中进行推荐的最先进方法。然而，现有大多数基于GNN的推荐方法集中于模型结构和基于预定义图的学習策略的优化，忽视了图构建阶段的重要性。早期的图构建工作通常依赖于特定规则或众包，这些方法要么过于简单，要么过于耗时。最近的研究开始利用大型语言模型（LLMs）来自动化图的构建，鉴于它们丰富的开放式知识和出色的推理能力。然而，这些方法通常存在两个局限性：（1）全局视图的不可见性（例如，忽略了上下文信息）（2）构建效率低下。为此，我们引入了AutoGraph，这是一种基于LLMs的推荐图自动生成框架。具体来说，我们首先利用LLMs推断用户偏好和项目知识，并将这些知识编码为语义向量。接下来，利用向量量化从语义向量中提取隐因子，并将这些隐因子作为额外节点与用户/项目节点相连，从而使生成的图具有深层次的全局视图语义。我们进一步设计了基于元路径的消息聚合，以有效地整合语义和协同信息。该框架具有模型无关性，并兼容不同的骨干模型。在三个实际数据集上的广泛实验表明，AutoGraph相较于现有基准方法具有更高的有效性和效率。我们已在华为广告平台部署了AutoGraph，并在在线A/B测试中获得了2.69%的RPM提升和7.31%的eCPM提升。目前AutoGraph已成为主要流量模型，服务着数亿用户。 

---
# TextMatch: Enhancing Image-Text Consistency Through Multimodal Optimization 

**Title (ZH)**: TextMatch：通过多模态优化增强图像-文本一致性 

**Authors**: Yucong Luo, Mingyue Cheng, Jie Ouyang, Xiaoyu Tao, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18185)  

**Abstract**: Text-to-image generative models excel in creating images from text but struggle with ensuring alignment and consistency between outputs and prompts. This paper introduces TextMatch, a novel framework that leverages multimodal optimization to address image-text discrepancies in text-to-image (T2I) generation and editing. TextMatch employs a scoring strategy powered by large language models (LLMs) and visual question-answering (VQA) models to evaluate semantic consistency between prompts and generated images. By integrating multimodal in-context learning and chain of thought reasoning, our method dynamically refines prompts through iterative optimization. This process ensures that the generated images better capture user intent of, resulting in higher fidelity and relevance. Extensive experiments demonstrate that TextMatch significantly improves text-image consistency across multiple benchmarks, establishing a reliable framework for advancing the capabilities of text-to-image generative models. Our code is available at this https URL. 

**Abstract (ZH)**: 文本到图像生成模型在从文本生成图像方面表现出色，但在确保输出与提示之间的对齐和一致性方面存在困难。本文介绍了一种名为TextMatch的新框架，该框架利用多模态优化来解决文本到图像（T2I）生成与编辑中的图像-文本不一致问题。TextMatch采用一种基于大型语言模型（LLM）和视觉问答（VQA）模型的评分策略，评估提示与生成图像之间的语义一致性。通过将多模态上下文学习与chain of thought推理相结合，我们的方法通过迭代优化动态细化提示。这一过程确保生成的图像更好地捕捉用户的意图，从而提高生成图像的准确性和相关性。广泛的实验表明，TextMatch在多个基准测试中显著提高了文本-图像的一致性，为提升文本到图像生成模型的能力建立了可靠的框架。我们的代码可通过以下链接获取：[此链接]。 

---
# Molar: Multimodal LLMs with Collaborative Filtering Alignment for Enhanced Sequential Recommendation 

**Title (ZH)**: Molar：具有协作过滤对齐的多模态语言模型以增强序列推荐 

**Authors**: Yucong Luo, Qitao Qin, Hao Zhang, Mingyue Cheng, Ruiran Yan, Kefan Wang, Jie Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18176)  

**Abstract**: Sequential recommendation (SR) systems have evolved significantly over the past decade, transitioning from traditional collaborative filtering to deep learning approaches and, more recently, to large language models (LLMs). While the adoption of LLMs has driven substantial advancements, these models inherently lack collaborative filtering information, relying primarily on textual content data neglecting other modalities and thus failing to achieve optimal recommendation performance. To address this limitation, we propose Molar, a Multimodal large language sequential recommendation framework that integrates multiple content modalities with ID information to capture collaborative signals effectively. Molar employs an MLLM to generate unified item representations from both textual and non-textual data, facilitating comprehensive multimodal modeling and enriching item embeddings. Additionally, it incorporates collaborative filtering signals through a post-alignment mechanism, which aligns user representations from content-based and ID-based models, ensuring precise personalization and robust performance. By seamlessly combining multimodal content with collaborative filtering insights, Molar captures both user interests and contextual semantics, leading to superior recommendation accuracy. Extensive experiments validate that Molar significantly outperforms traditional and LLM-based baselines, highlighting its strength in utilizing multimodal data and collaborative signals for sequential recommendation tasks. The source code is available at this https URL. 

**Abstract (ZH)**: 在过去的十年里，序列推荐（SR）系统经历了显著的发展，从传统的协同过滤方法过渡到深度学习方法，最近则转向了大型语言模型（LLMs）。尽管LLMs的应用推动了显著的进步，但这些模型本身缺乏协同过滤的信息，主要依赖于文本内容数据，忽视了其他模态的数据，因而未能实现最佳的推荐性能。为了解决这一限制，我们提出了Molar——一个多模态大型语言序列推荐框架，该框架整合了多种内容模态和ID信息，以有效捕捉协同信号。Molar采用多模态大型语言模型（MLLM）从文本和非文本数据中生成统一的项目表示，促进全面的多模态建模并丰富项目嵌入。此外，Molar通过后对齐机制整合了协同过滤信号，这种机制对基于内容和基于ID的用户表示进行对齐，以确保精确的个性化和稳健的性能。通过无缝结合多模态内容与协同过滤洞察，Molar能够同时捕获用户兴趣和上下文语义，从而提高推荐精度。大量实验验证了Molar在传统方法和LLM基线模型中显著优越的表现，突显了其利用多模态数据和协同信号进行序列推荐任务的优势。相关源代码可在以下链接获取：[请提供具体链接]。 

---
# INVESTORBENCH: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent 

**Title (ZH)**: 投资者基准：基于大语言模型代理的金融决策任务基准 

**Authors**: Haohang Li, Yupeng Cao, Yangyang Yu, Shashidhar Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen Jiang, Zining Zhu, Koduvayur Subbalakshmi, Guojun Xiong, Jimin Huang, Lingfei Qian, Xueqing Peng, Qianqian Xie, Jordan W. Suchow  

**Link**: [PDF](https://arxiv.org/pdf/2412.18174)  

**Abstract**: Recent advancements have underscored the potential of large language model (LLM)-based agents in financial decision-making. Despite this progress, the field currently encounters two main challenges: (1) the lack of a comprehensive LLM agent framework adaptable to a variety of financial tasks, and (2) the absence of standardized benchmarks and consistent datasets for assessing agent performance. To tackle these issues, we introduce \textsc{InvestorBench}, the first benchmark specifically designed for evaluating LLM-based agents in diverse financial decision-making contexts. InvestorBench enhances the versatility of LLM-enabled agents by providing a comprehensive suite of tasks applicable to different financial products, including single equities like stocks, cryptocurrencies and exchange-traded funds (ETFs). Additionally, we assess the reasoning and decision-making capabilities of our agent framework using thirteen different LLMs as backbone models, across various market environments and tasks. Furthermore, we have curated a diverse collection of open-source, multi-modal datasets and developed a comprehensive suite of environments for financial decision-making. This establishes a highly accessible platform for evaluating financial agents' performance across various scenarios. 

**Abstract (ZH)**: 近年来的研究凸显了基于大规模语言模型（LLM）的代理在金融决策中的潜力。尽管取得了这些进展，该领域目前面临两个主要挑战：（1）缺乏一个能够适应各种金融任务的全面LLM代理框架，和（2）缺乏标准化的基准和一致的数据集来评估代理性能。为应对这些问题，我们提出了InvestorBench，这是首个专门用于评估应用于多种金融决策场景的LLM代理的基准。InvestorBench通过提供适用于不同金融产品的一系列综合任务，增强了LLM代理的通用性，这些金融产品包括个股（如股票）、加密货币和交易所交易基金（ETF）。此外，我们使用十三种不同的LLM作为主干模型，评估我们的代理框架在各种市场环境和任务下的推理和决策能力。同时，我们整理了一组多元化的开源多模态数据集，并开发了一整套金融决策环境。这为在各种场景下评估金融代理的性能提供了一个高度可访问的平台。 

---
# scReader: Prompting Large Language Models to Interpret scRNA-seq Data 

**Title (ZH)**: scReader：引导大型语言模型解释单细胞RNA测序数据 

**Authors**: Cong Li, Qingqing Long, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18156)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable advancements, primarily due to their capabilities in modeling the hidden relationships within text sequences. This innovation presents a unique opportunity in the field of life sciences, where vast collections of single-cell omics data from multiple species provide a foundation for training foundational models. However, the challenge lies in the disparity of data scales across different species, hindering the development of a comprehensive model for interpreting genetic data across diverse organisms. In this study, we propose an innovative hybrid approach that integrates the general knowledge capabilities of LLMs with domain-specific representation models for single-cell omics data interpretation. We begin by focusing on genes as the fundamental unit of representation. Gene representations are initialized using functional descriptions, leveraging the strengths of mature language models such as LLaMA-2. By inputting single-cell gene-level expression data with prompts, we effectively model cellular representations based on the differential expression levels of genes across various species and cell types. In the experiments, we constructed developmental cells from humans and mice, specifically targeting cells that are challenging to annotate. We evaluated our methodology through basic tasks such as cell annotation and visualization analysis. The results demonstrate the efficacy of our approach compared to other methods using LLMs, highlighting significant improvements in accuracy and interoperability. Our hybrid approach enhances the representation of single-cell data and offers a robust framework for future research in cross-species genetic analysis. 

**Abstract (ZH)**: 大型语言模型（LLMs）在模拟文本序列中隐藏的关系方面取得了显著的进步。这种创新为生命科学领域带来了独特的机遇，其中来自多个物种的大量单细胞组学数据为训练基础模型提供了基础。然而，不同物种间数据规模的差异阻碍了跨物种遗传数据解释的全面模型的发展。在此研究中，我们提出了一种创新的融合方法，将LLMs的一般知识能力与针对单细胞组学数据的领域特定表示模型相结合，用于解释基因数据。我们首先以基因作为表示的基本单元进行研究。基因表示利用功能描述初始化，利用成熟语言模型如LLaMA-2的优势。通过输入单细胞基因水平表达数据并结合提示，我们有效构建了基于不同物种和细胞类型之间基因差异表达水平的细胞表示。在实验中，我们构建了来自人类和小鼠的发育细胞，并专门针对那些难以标注的细胞进行研究。我们通过基本任务如细胞注释和可视化分析评估了该方法。结果表明，与使用LLMs的其他方法相比，我们的方法具有更高的准确性和互操作性。我们的融合方法提高了单细胞数据的表示能力，并为跨物种遗传分析提供了稳健的框架。 

---
# GeneSUM: Large Language Model-based Gene Summary Extraction 

**Title (ZH)**: GeneSUM：基于大型语言模型的基因摘要提取 

**Authors**: Zhijian Chen, Chuan Hu, Min Wu, Qingqing Long, Xuezhi Wang, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18154)  

**Abstract**: Emerging topics in biomedical research are continuously expanding, providing a wealth of information about genes and their function. This rapid proliferation of knowledge presents unprecedented opportunities for scientific discovery and formidable challenges for researchers striving to keep abreast of the latest advancements. One significant challenge is navigating the vast corpus of literature to extract vital gene-related information, a time-consuming and cumbersome task. To enhance the efficiency of this process, it is crucial to address several key challenges: (1) the overwhelming volume of literature, (2) the complexity of gene functions, and (3) the automated integration and generation. In response, we propose GeneSUM, a two-stage automated gene summary extractor utilizing a large language model (LLM). Our approach retrieves and eliminates redundancy of target gene literature and then fine-tunes the LLM to refine and streamline the summarization process. We conducted extensive experiments to validate the efficacy of our proposed framework. The results demonstrate that LLM significantly enhances the integration of gene-specific information, allowing more efficient decision-making in ongoing research. 

**Abstract (ZH)**: 生物医学研究中的新兴领域正在不断发展，为基因及其功能提供了丰富的信息。这种知识的快速增长为科学发现提供了前所未有的机遇，同时也给研究人员带来了巨大挑战，他们需要努力跟上最新的进展。一个重要的挑战是如何有效导航大量的文献，提取关键的基因相关信息，这是一项耗时且繁琐的工作。为了提高这一过程的效率，必须解决几个关键问题：(1) 文献的庞大数量，(2) 基因功能的复杂性，以及(3) 自动化的整合与生成能力。为此，我们提出了一种名为GeneSUM的两阶段自动化基因摘要提取器，该工具利用大型语言模型（LLM）。我们的方法首先检索并去除目标基因文献中的冗余信息，然后通过微调LLM来优化和简化摘要生成过程。我们进行了大量实验来验证我们所提出的框架的有效性。结果表明，大型语言模型显著增强了基因特定信息的整合能力，使研究人员在当前研究中能够更高效地作出决策。 

---
# EvoPat: A Multi-LLM-based Patents Summarization and Analysis Agent 

**Title (ZH)**: EvoPat：一个基于多层次语言模型的专利总结与分析代理 

**Authors**: Suyuan Wang, Xueqian Yin, Menghao Wang, Ruofeng Guo, Kai Nan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18100)  

**Abstract**: The rapid growth of scientific techniques and knowledge is reflected in the exponential increase in new patents filed annually. While these patents drive innovation, they also present significant burden for researchers and engineers, especially newcomers. To avoid the tedious work of navigating a vast and complex landscape to identify trends and breakthroughs, researchers urgently need efficient tools to summarize, evaluate, and contextualize patents, revealing their innovative contributions and underlying scientific this http URL address this need, we present EvoPat, a multi-LLM-based patent agent designed to assist users in analyzing patents through Retrieval-Augmented Generation (RAG) and advanced search strategies. EvoPat leverages multiple Large Language Models (LLMs), each performing specialized roles such as planning, identifying innovations, and conducting comparative evaluations. The system integrates data from local databases, including patents, literature, product catalogous, and company repositories, and online searches to provide up-to-date insights. The ability to collect information not included in original database automatically is also implemented. Through extensive testing in the natural language processing (NLP) domain, we demonstrate that EvoPat outperforms GPT-4 in tasks such as patent summarization, comparative analysis, and technical evaluation. EvoPat represents a significant step toward creating AI-powered tools that empower researchers and engineers to efficiently navigate the complexities of the patent landscape. 

**Abstract (ZH)**: 科学技术与知识的迅速增长体现在每年提交的新专利数量呈指数级增长上。尽管这些专利推动了创新，但也给研究人员和工程师带来了巨大的负担，尤其是新手。为了避免在庞大而复杂的环境中进行繁琐的工作来识别趋势和突破，研究人员迫切需要高效的工具来总结、评估和语境化专利，揭示其创新贡献和背后的科学意义。为了解决这一需求，我们提出了EvoPat，这是一种基于多大规模语言模型（LLM）的专利代理，旨在通过检索增强生成（RAG）和高级搜索策略帮助用户分析专利。EvoPat利用多个大规模语言模型，每个模型分别承担规划、识别创新和进行对比评价等专业化角色。该系统整合了本地数据库中的数据，包括专利、文献、产品目录和公司库，并结合在线搜索，提供最新的见解。同时，系统还具备自动收集不在原始数据库中但相关的数据信息的能力。通过在自然语言处理（NLP）领域的广泛测试，我们证明EvoPat在专利总结、对比分析和技术评估等任务上优于GPT-4。EvoPat代表了向创建能够帮助研究人员和工程师高效导航专利复杂性的AI工具迈出的重要一步。 

---
# Generating Traffic Scenarios via In-Context Learning to Learn Better Motion Planner 

**Title (ZH)**: 通过情境学习生成交通场景以改善运动规划器性能 

**Authors**: Aizierjiang Aiersilan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18086)  

**Abstract**: Motion planning is a crucial component in autonomous driving. State-of-the-art motion planners are trained on meticulously curated datasets, which are not only expensive to annotate but also insufficient in capturing rarely seen critical scenarios. Failing to account for such scenarios poses a significant risk to motion planners and may lead to incidents during testing. An intuitive solution is to manually compose such scenarios by programming and executing a simulator (e.g., CARLA). However, this approach incurs substantial human costs. Motivated by this, we propose an inexpensive method for generating diverse critical traffic scenarios to train more robust motion planners. First, we represent traffic scenarios as scripts, which are then used by the simulator to generate traffic scenarios. Next, we develop a method that accepts user-specified text descriptions, which a Large Language Model (LLM) translates into scripts using in-context learning. The output scripts are sent to the simulator that produces the corresponding traffic scenarios. As our method can generate abundant safety-critical traffic scenarios, we use them as synthetic training data for motion planners. To demonstrate the value of generated scenarios, we train existing motion planners on our synthetic data, real-world datasets, and a combination of both. Our experiments show that motion planners trained with our data significantly outperform those trained solely on real-world data, showing the usefulness of our synthetic data and the effectiveness of our data generation method. Our source code is available at this https URL. 

**Abstract (ZH)**: 自主驾驶中的运动规划是其关键组成部分。最新的运动规划器在精心策划的数据集上进行训练，这些数据集不仅标注成本高昂，而且难以涵盖罕见但至关重要的应用场景。未能考虑这些场景会显著增加运动规划器的风险，并可能在测试过程中引发事故。一个直观的解决方案是通过编程和执行模拟器（例如CARLA）手动组合这些场景。然而，这种方法会带来显著的人工成本。受到这一问题的启发，我们提出了一种低成本的方法，用于生成多样化的关键交通场景，以训练更稳健的运动规划器。首先，我们将交通场景表示为脚本，然后将这些脚本用于模拟器生成交通场景。接着，我们开发了一种方法，该方法接受用户指定的文本描述，由大型语言模型（LLM）通过上下文学习将其转化为脚本。生成的脚本随后发送给模拟器，以生成相应的交通场景。由于我们的方法能够生成大量的安全关键交通场景，我们将其用作运动规划器的合成训练数据。为了展示生成场景的价值，我们使用了我们的合成数据、真实世界数据，以及两者的组合来训练现有的运动规划器。实验结果显示，使用我们数据训练的运动规划器在性能上显著优于仅使用真实世界数据进行训练的运动规划器，这表明了我们合成数据的有效性以及生成方法的有效性。我们的源代码可以在以下网址获取：这个 https URL。 

---
# More than Chit-Chat: Developing Robots for Small-Talk Interactions 

**Title (ZH)**: 不仅仅是闲聊：开发用于短对话的机器人 

**Authors**: Rebecca Ramnauth, Dražen Brščić, Brian Scassellati  

**Link**: [PDF](https://arxiv.org/pdf/2412.18023)  

**Abstract**: Beyond mere formality, small talk plays a pivotal role in social dynamics, serving as a verbal handshake for building rapport and understanding. For conversational AI and social robots, the ability to engage in small talk enhances their perceived sociability, leading to more comfortable and natural user interactions. In this study, we evaluate the capacity of current Large Language Models (LLMs) to drive the small talk of a social robot and identify key areas for improvement. We introduce a novel method that autonomously generates feedback and ensures LLM-generated responses align with small talk conventions. Through several evaluations -- involving chatbot interactions and human-robot interactions -- we demonstrate the system's effectiveness in guiding LLM-generated responses toward realistic, human-like, and natural small-talk exchanges. 

**Abstract (ZH)**: 超越单纯的礼仪，闲聊在社交动态中发挥着关键作用，作为建立共鸣和沟通理解的口头握手。对于对话式人工智能和社交机器人而言，具备进行闲聊的能力可以增强它们的社交感知度，从而实现更为舒适和自然的用户互动。在本研究中，我们评估了当前大规模语言模型（LLM）在驱动社交机器人闲聊方面的能力，并确定了需要改进的关键领域。我们提出了一种新颖的方法，能够自主生成反馈并确保LLM生成的回复符合闲聊规范。通过多项评估——包括聊天机器人互动和人机互动——我们证明了该系统在引导LLM生成的回复向真实、人类般的自然闲聊交流方向发展的有效性。 

---
# Trustworthy and Efficient LLMs Meet Databases 

**Title (ZH)**: 可信且高效的大型语言模型与数据库相遇 

**Authors**: Kyoungmin Kim, Anastasia Ailamaki  

**Link**: [PDF](https://arxiv.org/pdf/2412.18022)  

**Abstract**: In the rapidly evolving AI era with large language models (LLMs) at the core, making LLMs more trustworthy and efficient, especially in output generation (inference), has gained significant attention. This is to reduce plausible but faulty LLM outputs (a.k.a hallucinations) and meet the highly increased inference demands. This tutorial explores such efforts and makes them transparent to the database community. Understanding these efforts is essential in harnessing LLMs in database tasks and adapting database techniques to LLMs. Furthermore, we delve into the synergy between LLMs and databases, highlighting new opportunities and challenges in their intersection. This tutorial aims to share with database researchers and practitioners essential concepts and strategies around LLMs, reduce the unfamiliarity of LLMs, and inspire joining in the intersection between LLMs and databases. 

**Abstract (ZH)**: 在以大规模语言模型（LLMs）为核心的快速演化的AI时代，提高LLMs的可靠性和效率，特别是在输出生成（推理）方面，已经引起了广泛关注。这旨在减少可能但虚假的LLMs输出（即幻觉），以满足高度增加的推理需求。本教程探讨了这些努力，并使数据库社区对此了解透明。理解这些努力对于在数据库任务中利用LLMs以及将数据库技术与LLMs相结合至关重要。此外，我们还探讨了LLMs与数据库之间的协同作用，突显了它们交集中的新机遇和挑战。本教程旨在与数据库研究人员和实践者分享有关LLMs的基本概念和策略，减少对LLMs的陌生感，并激发他们参与到LLMs与数据库的交集中来。 

---
# LMV-RPA: Large Model Voting-based Robotic Process Automation 

**Title (ZH)**: LMV-RPA：基于大型模型投票的机器人流程自动化 

**Authors**: Osama Abdellatif, Ahmed Ayman, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2412.17965)  

**Abstract**: Automating high-volume unstructured data processing is essential for operational efficiency. Optical Character Recognition (OCR) is critical but often struggles with accuracy and efficiency in complex layouts and ambiguous text. These challenges are especially pronounced in large-scale tasks requiring both speed and precision. This paper introduces LMV-RPA, a Large Model Voting-based Robotic Process Automation system to enhance OCR workflows. LMV-RPA integrates outputs from OCR engines such as Paddle OCR, Tesseract OCR, Easy OCR, and DocTR with Large Language Models (LLMs) like LLaMA 3 and Gemini-1.5-pro. Using a majority voting mechanism, it processes OCR outputs into structured JSON formats, improving accuracy, particularly in complex layouts. The multi-phase pipeline processes text extracted by OCR engines through LLMs, combining results to ensure the most accurate outputs. LMV-RPA achieves 99 percent accuracy in OCR tasks, surpassing baseline models with 94 percent, while reducing processing time by 80 percent. Benchmark evaluations confirm its scalability and demonstrate that LMV-RPA offers a faster, more reliable, and efficient solution for automating large-scale document processing tasks. 

**Abstract (ZH)**: 大规模自动化无结构数据处理对于提高操作效率至关重要。光学字符识别（OCR）是关键步骤，但往往在复杂布局和模糊文本中面临着准确性和效率的挑战。这些挑战在需要快速和精准的大型任务中尤为突出。本文介绍了基于大型模型投票的机器人流程自动化系统（LMV-RPA），以增强OCR工作流程。LMV-RPA 将 Paddle OCR、Tesseract OCR、Easy OCR 和 DocTR 等OCR引擎的输出与 LLaMA 3 和 Gemini-1.5-pro 等大型语言模型（LLMs）整合在一起。利用多数投票机制，它将OCR输出转换为结构化的JSON格式，特别是在复杂布局中提高了准确性。多阶段管道流程将由OCR引擎提取的文本通过LLMs处理，并结合结果以确保最准确的输出。LMV-RPA 的OCR任务准确率达到99%，超过基线模型94%的准确率，同时将处理时间减少80%。基准评估证实了其可扩展性，并表明LMV-RPA 提供了一种更快、更可靠且更高效的自动化大规模文档处理任务的解决方案。 

---
# Evaluating LLM Reasoning in the Operations Research Domain with ORQA 

**Title (ZH)**: 使用ORQA评估大型语言模型在运筹学领域的推理能力 

**Authors**: Mahdi Mostajabdaveh, Timothy T. Yu, Samarendra Chandan Bindu Dash, Rindranirina Ramamonjison, Jabo Serge Byusa, Giuseppe Carenini, Zirui Zhou, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17874)  

**Abstract**: In this paper, we introduce and apply Operations Research Question Answering (ORQA), a new benchmark designed to assess the generalization capabilities of Large Language Models (LLMs) in the specialized technical domain of Operations Research (OR). This benchmark evaluates whether LLMs can emulate the knowledge and reasoning skills of OR experts when confronted with diverse and complex optimization problems. The dataset, developed by OR experts, features real-world optimization problems that demand multistep reasoning to construct their mathematical models. Our evaluations of various open source LLMs, such as LLaMA 3.1, DeepSeek, and Mixtral, reveal their modest performance, highlighting a gap in their ability to generalize to specialized technical domains. This work contributes to the ongoing discourse on LLMs generalization capabilities, offering valuable insights for future research in this area. The dataset and evaluation code are publicly available. 

**Abstract (ZH)**: 在这篇论文中，我们引入并应用了运筹学问答（ORQA），一个新的基准，旨在评估大型语言模型（LLM）在运筹学（OR）这一专门技术领域的泛化能力。该基准评估LLM是否能在面对多样化和复杂优化问题时，模仿运筹学专家的知识和推理技能。该数据集由运筹学专家精心设计，包含需要多步推理来构建其数学模型的实际优化问题。我们对多种开源LLM（如LLaMA 3.1、DeepSeek和Mixtral）的评估显示，它们的性能较为有限，这突显了它们在泛化到专门技术领域方面的能力差距。这项工作为LLM泛化能力的持续讨论做出了贡献，并为该领域的未来研究提供了宝贵见解。该数据集和评估代码已公开提供。 

---
# Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types 

**Title (ZH)**: 评估并提升针对多轮文本到SQL转换的大型语言模型的性能，涵盖多种问题类型 

**Authors**: Ziming Guo, Chao Ma, Yinggang Sun, Tiancheng Zhao, Guangyao Wang, Hai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17867)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly advanced text-to-SQL systems. However, most LLM-based methods often narrowly focus on SQL generation, neglecting the complexities of real-world conversational queries. This oversight can lead to unreliable responses, particularly for ambiguous questions that cannot be directly addressed with SQL. To bridge this gap, we propose MMSQL, a comprehensive test suite designed to evaluate the question classification and SQL generation capabilities of LLMs by simulating real-world scenarios with diverse question types and multi-turn Q\&A interactions. Using MMSQL, we assessed the performance of popular LLMs, including both open-source and closed-source models, and identified key factors impacting their performance in such scenarios. Moreover, we introduce an LLM-based multi-agent framework that employs specialized agents to identify question types and determine appropriate answering strategies. Our experiments demonstrate that this approach significantly enhances the model's ability to navigate the complexities of conversational dynamics, effectively handling the diverse and complex nature of user queries. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进展显著提升了文本到SQL系统的性能。然而，大多数基于LLM的方法往往仅专注于SQL生成，忽视了真实世界对话查询的复杂性。这种忽视可能导致不可靠的响应，特别是对于那些不能直接用SQL回答的含糊不清的问题。为了弥补这一差距，我们提出了MMSQL，这是一个全面的测试套件，通过模拟具有多种问题类型和多轮问答交互的真实场景，评估LLM的问题分类和SQL生成能力。通过MMSQL，我们评估了几种流行的LLM，包括开源和封闭源模型，并确定了影响其在这种情况下表现的关键因素。此外，我们引入了一种基于LLM的多代理框架，该框架使用专门的代理来识别问题类型并确定合适的回答策略。我们的实验表明，这种方法显著增强了模型适应对话动态复杂性的能力，有效处理了用户查询的多样性和复杂性。 

---
# The Rosetta Paradox: Domain-Specific Performance Inversions in Large Language Models 

**Title (ZH)**: 罗塞塔悖论：大型语言模型中的领域特定性能反转 

**Authors**: Basab Jha, Ujjwal Puri  

**Link**: [PDF](https://arxiv.org/pdf/2412.17821)  

**Abstract**: While large language models, such as GPT and BERT, have already demonstrated unprecedented skills in everything from natural language processing to domain-specific applications, there came an unexplored phenomenon we term the Rosetta Paradox. The Rosetta Paradox characterizes the counterintuitive performance inversions across domains of knowledge. This paradox captures how such LLMs can excel in highly specialized fields but do poorly on tasks which require general, everyday knowledge. This paper formalizes the definition of the Rosetta Paradox and introduces a panoramic analysis framework that includes both a Domain Specificity Index (DSI) and a Performance Inversion Metric (PIM) for consistent quantification of domain-specific behavior in LLMs.
We adopt this paradox and conduct a series of investigations through extensive experiments across diverse models and knowledge domains, ranging from rich technical areas to common-sense reasoning. Our findings indicate that the Rosetta Paradox is likely not a mere artifact of data distribution but an intrinsic architectural and emergent property of deep neural networks. We present comparative analyses across different model architectures, sizes, and training methodologies that shed light into the peculiar ways this paradox manifests itself and challenge the standard evaluation metrics. 

**Abstract (ZH)**: 尽管诸如GPT和BERT这样的大规模语言模型已经在自然语言处理以及领域特定应用中展现了前所未有的技能，但其中出现了一种未被探索的现象，我们称之为罗塞塔悖论。罗塞塔悖论指出了知识领域间非直观的性能倒置现象。这种悖论揭示了这些语言模型在高度专业化领域表现出色，但在需要广泛日常知识的任务上则表现不佳。本文正式定义了罗塞塔悖论，并引入了一个全景分析框架，该框架包括领域特定性指数（DSI）和性能倒置度量（PIM），用于量化LLM在不同领域的特定行为。

我们采用了这一悖论，并通过广泛的实验，对多种模型和知识领域（从丰富的技术领域到常识推理）进行了深入研究。我们的研究结果表明，罗塞塔悖论很可能不是数据分布的偶然现象，而是一种深层次神经网络架构的固有和新兴属性。我们对不同模型架构、规模和训练方法进行了比较分析，揭示了这一悖论表现出来的独特方式，并挑战了传统的评估标准。 

---
# Inductive Linguistic Reasoning with Large Language Models 

**Title (ZH)**: 使用大规模语言模型进行归纳语言推理 

**Authors**: Raghav Ramji, Keshav Ramji  

**Link**: [PDF](https://arxiv.org/pdf/2412.17819)  

**Abstract**: Evaluating large language models (LLMs) on their linguistic reasoning capabilities is an important task to understand the gaps in their skills that may surface during large-scale adoption. In this work, we investigate the abilities of such models to perform abstract multilingual reasoning through the lens of linguistic puzzles on extremely low-resource languages. As these translation tasks involve inductive and deductive reasoning from reference instances, we examine whether diverse auxiliary demonstrations can be automatically induced from seed exemplars, through analogical prompting. We employ a two-stage procedure, first generating analogical exemplars with a language model, and then applying them in-context along with provided target language exemplars. Our results on the modeLing dataset show that analogical prompting is effective in eliciting models' knowledge of language grammar similarities, boosting the performance of GPT-4o by as much as 8.1% and Llama-3.1-405B-Instruct by 5.9% over chain-of-thought approaches. These gains are attributable to the analogical demonstrations, both when self-generated as well as when produced by weaker multilingual models. Furthermore, we demonstrate that our method generalizes to other tasks present in Linguistics Olympiad competitions, achieving sizable improvements across all problem types and difficulty levels included in the LINGOLY dataset with GPT-4o. We also report several findings about interesting phenomena which drive linguistic reasoning performance, suggesting that such puzzles are a valuable benchmark for new reasoning methods. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）的语言推理能力是一项重要任务，有助于理解其在大规模应用中可能暴露出的能力差距。在本研究中，我们通过极端低资源语言的语言谜题，探讨这些模型在进行抽象多语言推理方面的能力。由于这些翻译任务涉及归纳和演绎推理，我们研究是否可以从种子示例中自动诱导出多样性的辅助示范，通过类比提示进行。我们采用两阶段的过程，首先使用语言模型生成类比示例，然后在提供目标语言示例的上下文中应用它们。我们在modeLing数据集上的结果显示，类比提示在激发模型对语言语法相似性的知识方面是有效的，提高了GPT-4o的性能8.1%，以及Llama-3.1-405B-Instruct的性能5.9%，超过了基于链式思考的方法。这些提升归因于无论是自动生成的还是由较弱的多语言模型生成的类比示范。此外，我们展示了本方法在其他包括在Linguistics Olympiad竞赛中的任务上的泛化能力，在LINGOLY数据集中的所有问题类型和难度级别上，GPT-4o均实现了显著的改进。我们还报告了一些关于影响语言推理性能有趣现象的研究发现，这些发现表明该类谜题是评估新推理方法的有价值的基准。 

---
# Distilling Fine-grained Sentiment Understanding from Large Language Models 

**Title (ZH)**: 从大规模语言模型中提炼细粒度情感理解 

**Authors**: Yice Zhang, Guangyu Xie, Hongling Xu, Kaiheng Hou, Jianzhu Bao, Qianlong Wang, Shiwei Chen, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18552)  

**Abstract**: Fine-grained sentiment analysis (FSA) aims to extract and summarize user opinions from vast opinionated text. Recent studies demonstrate that large language models (LLMs) possess exceptional sentiment understanding capabilities. However, directly deploying LLMs for FSA applications incurs high inference costs. Therefore, this paper investigates the distillation of fine-grained sentiment understanding from LLMs into small language models (SLMs). We prompt LLMs to examine and interpret the sentiments of given reviews and then utilize the generated content to pretrain SLMs. Additionally, we develop a comprehensive FSA benchmark to evaluate both SLMs and LLMs. Extensive experiments on this benchmark reveal that: (1) distillation significantly enhances the performance of SLMs in FSA tasks, achieving a 6.00\% improvement in $F_1$-score, and the distilled model can outperform Llama-2-7b with only 220M parameters; (2) distillation equips SLMs with excellent zero-shot sentiment classification capabilities, enabling them to match or even exceed their teacher models. These results suggest that distillation from LLMs is a highly promising direction for FSA. We will release our code, data, and pretrained model weights at \url{this https URL}. 

**Abstract (ZH)**: 精细粒度情感分析（Fine-grained Sentiment Analysis, FSA）旨在从大量的意见性文本中提取和总结用户观点。近期的研究表明，大型语言模型（Large Language Models, LLMs）具有出色的情感理解能力。然而，直接将LLMs部署到FSA应用中会产生较高的推理成本。因此，本文探讨了将LLMs的情感理解能力精简到小型语言模型（Small Language Models, SLMs）中的方法。我们促使LLMs审查和解释给定评论的情感，并利用生成的内容对SLMs进行预训练。此外，我们还开发了一个全面的FSA基准，用于评估SLMs和LLMs。在该基准上的广泛实验表明：(1) 精简显著提高了SLMs在FSA任务中的性能，实现了$F_1$-分数6.00%的提升，并且精简后的模型在仅有220M参数的情况下仍能优于Llama-2-7b；(2) 精简赋予了SLMs出色的零样本情感分类能力，使它们能够与甚至超越其教师模型。这些结果表明，从LLMs中提取情感理解能力是一种非常有前景的方向。我们将在 \url{this https URL} 释放我们的代码、数据和预训练模型权重。 

---
# Harnessing Large Language Models for Knowledge Graph Question Answering via Adaptive Multi-Aspect Retrieval-Augmentation 

**Title (ZH)**: 利用自适应多方面检索增强技术解锁大规模语言模型在知识图谱问答中的潜力 

**Authors**: Derong Xu Xinhang Li, Ziheng Zhang, Zhenxi Lin, Zhihong Zhu, Zhi Zheng, Xian Wu, Xiangyu Zhao, Tong Xu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.18537)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities, yet struggle with hallucination and outdated knowledge when tasked with complex knowledge reasoning, resulting in factually incorrect outputs. Previous studies have attempted to mitigate it by retrieving factual knowledge from large-scale knowledge graphs (KGs) to assist LLMs in logical reasoning and prediction of answers. However, this kind of approach often introduces noise and irrelevant data, especially in situations with extensive context from multiple knowledge aspects. In this way, LLM attention can be potentially mislead from question and relevant information. In our study, we introduce an Adaptive Multi-Aspect Retrieval-augmented over KGs (Amar) framework. This method retrieves knowledge including entities, relations, and subgraphs, and converts each piece of retrieved text into prompt embeddings. The Amar framework comprises two key sub-components: 1) a self-alignment module that aligns commonalities among entities, relations, and subgraphs to enhance retrieved text, thereby reducing noise interference; 2) a relevance gating module that employs a soft gate to learn the relevance score between question and multi-aspect retrieved data, to determine which information should be used to enhance LLMs' output, or even filtered altogether. Our method has achieved state-of-the-art performance on two common datasets, WebQSP and CWQ, showing a 1.9\% improvement in accuracy over its best competitor and a 6.6\% improvement in logical form generation over a method that directly uses retrieved text as context prompts. These results demonstrate the effectiveness of Amar in improving the reasoning of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了显著的能力，但在处理复杂知识推理任务时，它们会遇到幻觉和过时知识的问题，导致事实错误的输出。以往的研究尝试通过从大规模知识图谱（KGs）中检索事实知识来辅助LLMs的逻辑推理和答案预测，以减轻这些问题。然而，这种做法往往引入了噪声和无关的数据，尤其是在涉及多方面知识背景的广泛上下文情况下。这样可能会误导LLMs的注意力，使其偏离问题和相关信息。在本研究中，我们提出了一种自适应的多方面检索增强知识图谱框架（Amar）。该方法检索包括实体、关系和子图的知识，并将每一段检索到的文字转换成提示嵌入。Amar框架包含两个关键子模块：1) 自对齐模块，该模块通过对齐实体、关系和子图之间的共同点来增强检索文本，从而减少噪声干扰；2) 相关性闸门模块，该模块使用软闸门来学习问题与多方面检索数据的相关性评分，从而决定哪些信息应该用于增强LLMs的输出，甚至可以过滤掉。我们的方法在两个常用数据集WebQSP和CWQ上达到了最先进的性能，通过与最佳竞争对手相比，在准确率上提高了1.9%，在逻辑形式生成上提高了6.6%。这些结果证明了Amar在提高LLMs推理能力方面的有效性。 

---
# Think or Remember? Detecting and Directing LLMs Towards Memorization or Generalization 

**Title (ZH)**: 思考还是记忆？检测并引导大语言模型向记忆或泛化方向发展 

**Authors**: Yi-Fu Fu, Yu-Chieh Tu, Tzu-Ling Cheng, Cheng-Yu Lin, Yi-Ting Yang, Heng-Yi Liu, Keng-Te Liao, Da-Cheng Juan, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.18497)  

**Abstract**: In this paper, we explore the foundational mechanisms of memorization and generalization in Large Language Models (LLMs), inspired by the functional specialization observed in the human brain. Our investigation serves as a case study leveraging specially designed datasets and experimental-scale LLMs to lay the groundwork for understanding these behaviors. Specifically, we aim to first enable LLMs to exhibit both memorization and generalization by training with the designed dataset, then (a) examine whether LLMs exhibit neuron-level spatial differentiation for memorization and generalization, (b) predict these behaviors using model internal representations, and (c) steer the behaviors through inference-time interventions. Our findings reveal that neuron-wise differentiation of memorization and generalization is observable in LLMs, and targeted interventions can successfully direct their behavior. 

**Abstract (ZH)**: 在本文中，我们探讨了大型语言模型（LLMs）中记忆和泛化的基本机制，受到了人类大脑功能专业化观察的启发。我们的研究通过利用特别设计的数据集和实验规模的LLM作为案例研究，为理解这些行为奠定了基础。具体而言，我们旨在通过使用设计的数据集对LLM进行训练，首先让LLM展现出记忆和泛化的双重特性，然后（a）考察LLM是否在神经元级别上表现出记忆和泛化的空间分化，（b）通过模型内部表示预测这些行为，并（c）通过推理时的干预引导这些行为。我们的发现表明，神经元级别的记忆和泛化分化在LLM中是可以观察到的，并且有针对性的干预可以成功地指导其行为。 

---
# Is Large Language Model Good at Triple Set Prediction? An Empirical Study 

**Title (ZH)**: 大型语言模型在三元组集预测方面表现如何？一项实证研究 

**Authors**: Yuan Yuan, Yajing Xu, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18443)  

**Abstract**: The core of the Knowledge Graph Completion (KGC) task is to predict and complete the missing relations or nodes in a KG. Common KGC tasks are mostly about inferring unknown elements with one or two elements being known in a triple. In comparison, the Triple Set Prediction (TSP) task is a more realistic knowledge graph completion task. It aims to predict all elements of unknown triples based on the information from known triples. In recent years, large language models (LLMs) have exhibited significant advancements in language comprehension, demonstrating considerable potential for KGC tasks. However, the potential of LLM on the TSP task has not yet to be investigated. Thus in this paper we proposed a new framework to explore the strengths and limitations of LLM in the TSP task. Specifically, the framework consists of LLM-based rule mining and LLM-based triple set prediction. The relation list of KG embedded within rich semantic information is first leveraged to prompt LLM in the generation of rules. This process is both efficient and independent of statistical information, making it easier to mine effective and realistic rules. For each subgraph, the specified rule is applied in conjunction with the relevant triples within that subgraph to guide the LLM in predicting the missing triples. Subsequently, the predictions from all subgraphs are consolidated to derive the complete set of predicted triples on KG. Finally, the method is evaluated on the relatively complete CFamily dataset. The experimental results indicate that when LLMs are required to adhere to a large amount of factual knowledge to predict missing triples, significant hallucinations occurs, leading to a noticeable decline in performance. To further explore the causes of this phenomenon, this paper presents a comprehensive analysis supported by a detailed case study. 

**Abstract (ZH)**: 知识图谱补全（Knowledge Graph Completion, KGC）任务的核心在于预测和补全知识图谱中缺失的关系或节点。常见的KGC任务通常是在已知一个或两个元素的情况下，推断三元组中的未知元素。相比之下，三元组集预测（Triple Set Prediction, TSP）任务是一种更加现实的KGC任务，它旨在基于已知三元组的信息预测所有未知三元组的元素。近年来，大型语言模型（Large Language Models, LLMs）在语言理解方面取得了显著进步，展现了在KGC任务中的巨大潜力。然而，LLM在TSP任务中的潜力尚未得到充分研究。因此，在本文中，我们提出了一种新的框架，以探索LLM在TSP任务中的优点和局限性。具体而言，该框架由基于LLM的规则挖掘和基于LLM的三元组集预测两部分组成。首先，嵌入丰富语义信息的知识图谱关系列表被利用来提示LLM生成规则，这一过程既高效又独立于统计信息，使得有效且真实的规则易于挖掘。对于每个子图，特定的规则与该子图相关的三元组结合使用，以指导LLM预测缺失的三元组。随后，从所有子图的预测中整合出完整的预测三元组集。最后，该方法在相对完整的CFamily数据集上进行了评估。实验结果表明，当LLM需要依据大量事实知识预测缺失的三元组时，会出现显著的虚构现象，导致性能明显下降。为进一步探讨这一现象的原因，本文通过详细的案例研究进行了一项全面的分析。 

---
# Robustness-aware Automatic Prompt Optimization 

**Title (ZH)**: Robustness-aware 自动提示优化 

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Fan Yang, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18196)  

**Abstract**: The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）的性能取决于提示的质量以及输入数据在语义和结构完整性方面的信息。然而，当前的提示生成方法主要集中在生成适用于干净输入数据的提示上，往往忽视了扰动输入对提示性能的影响。为了解决这一限制，我们提出了BATprompt（通过对抗训练生成提示）——一种设计用于抵御输入扰动（例如输入中的拼写错误）的新型提示生成方法。受到对抗训练技术的启发，BATprompt 通过两步过程在扰动任务上表现出强大的性能：对抗扰动和基于LLM对未受扰输入的迭代优化。与传统的对抗攻击方法不同，BATprompt 避免了对真实梯度或模型参数的依赖。相反，它利用了LLM的高级推理、语言理解和自我反思能力来模拟梯度，引导对抗扰动的生成并优化提示性能。在我们的实验中，我们在语言理解和生成任务的多个数据集上评估了BATprompt。结果表明，BATprompt 在各种扰动场景下的鲁棒性和性能均优于现有的提示生成方法。 

---
# Molly: Making Large Language Model Agents Solve Python Problem More Logically 

**Title (ZH)**: 莫莉：使大型语言模型代理更逻辑地解决Python问题 

**Authors**: Rui Xiao, Jiong Wang, Lu Han, Na Zong, Han Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18093)  

**Abstract**: Applying large language models (LLMs) as teaching assists has attracted much attention as an integral part of intelligent education, particularly in computing courses. To reduce the gap between the LLMs and the computer programming education expert, fine-tuning and retrieval augmented generation (RAG) are the two mainstream methods in existing researches. However, fine-tuning for specific tasks is resource-intensive and may diminish the model`s generalization capabilities. RAG can perform well on reducing the illusion of LLMs, but the generation of irrelevant factual content during reasoning can cause significant confusion for learners. To address these problems, we introduce the Molly agent, focusing on solving the proposed problem encountered by learners when learning Python programming language. Our agent automatically parse the learners' questioning intent through a scenario-based interaction, enabling precise retrieval of relevant documents from the constructed knowledge base. At generation stage, the agent reflect on the generated responses to ensure that they not only align with factual content but also effectively answer the user's queries. Extensive experimentation on a constructed Chinese Python QA dataset shows the effectiveness of the Molly agent, indicating an enhancement in its performance for providing useful responses to Python questions. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

将大型语言模型（LLMs）用作教学助手，作为智能教育，尤其是在计算课程中的一项重要组成部分，已经引起了广泛关注。为减少LLMs与计算机编程教育专家之间的差距，现有的研究主要采用了微调和检索增强生成（RAG）两种方法。然而，针对特定任务的微调耗时且可能削弱模型的一般化能力。RAG 可在减少LLMs的幻觉方面表现出色，但在推理过程中生成的相关事实内容不足可能导致学习者的困惑。为解决这些问题，我们引入了Molly代理，专注于解决学习Python编程语言过程中学习者遇到的问题。我们的代理通过基于场景的交互自动解析学习者的提问意图，从而从构建的知识库中精确检索相关文档。在生成阶段，代理反思生成的回答，以确保它们不仅与事实内容一致，还能有效地回答用户的问题。通过对构建的中文Python问答数据集进行广泛的实验，展示了Molly代理的有效性，表明其性能在提供有用的Python问题回答方面有所提升。 

---
# Improving Factuality with Explicit Working Memory 

**Title (ZH)**: 提高事实准确性与显性工作记忆的关系 

**Authors**: Mingda Chen, Yang Li, Karthik Padthe, Rulin Shao, Alicia Sun, Luke Zettlemoyer, Gargi Gosh, Wen-tau Yih  

**Link**: [PDF](https://arxiv.org/pdf/2412.18069)  

**Abstract**: Large language models can generate factually inaccurate content, a problem known as hallucination. Recent works have built upon retrieved-augmented generation to improve factuality through iterative prompting but these methods are limited by the traditional RAG design. To address these challenges, we introduce EWE (Explicit Working Memory), a novel approach that enhances factuality in long-form text generation by integrating a working memory that receives real-time feedback from external resources. The memory is refreshed based on online fact-checking and retrieval feedback, allowing EWE to rectify false claims during the generation process and ensure more accurate and reliable outputs. Our experiments demonstrate that Ewe outperforms strong baselines on four fact-seeking long-form generation datasets, increasing the factuality metric, VeriScore, by 2 to 10 points absolute without sacrificing the helpfulness of the responses. Further analysis reveals that the design of rules for memory updates, configurations of memory units, and the quality of the retrieval datastore are crucial factors for influencing model performance. 

**Abstract (ZH)**: 大型语言模型可以生成事实不准确的内容，这一问题被称为幻觉。近年来的研究通过迭代提示来增强检索增强生成方法以提高事实准确性，但这些方法受限于传统的RAG设计。为了解决这些问题，我们提出了一种名为EWE（显式工作记忆）的新颖方法，该方法通过集成一个实时从外部资源接收反馈的工作记忆来增强长文本生成的事实准确性。该工作记忆根据在线事实核查和检索反馈进行更新，使EWE在生成过程中能够更正虚假声明，从而确保输出更加准确可靠。我们的实验表明，EWE在四个事实寻求的长文本生成数据集上表现优于强大的基线方法，在不牺牲回复的有用性的情况下，VeriScore准确性指标提高了2到10个百分点。进一步的分析表明，记忆更新规则的设计、记忆单元的配置以及检索数据存储的质量对模型性能有重要影响。 

---
# Factuality or Fiction? Benchmarking Modern LLMs on Ambiguous QA with Citations 

**Title (ZH)**: 事实还是虚构？现代大规模语言模型在有歧义的问答任务上的基准测试及引文分析 

**Authors**: Maya Patel, Aditi Anand  

**Link**: [PDF](https://arxiv.org/pdf/2412.18051)  

**Abstract**: Benchmarking modern large language models (LLMs) on complex and realistic tasks is critical to advancing their development. In this work, we evaluate the factual accuracy and citation performance of state-of-the-art LLMs on the task of Question Answering (QA) in ambiguous settings with source citations. Using three recently published datasets-DisentQA-DupliCite, DisentQA-ParaCite, and AmbigQA-Cite-featuring a range of real-world ambiguities, we analyze the performance of two leading LLMs, GPT-4o-mini and Claude-3.5. Our results show that larger, recent models consistently predict at least one correct answer in ambiguous contexts but fail to handle cases with multiple valid answers. Additionally, all models perform equally poorly in citation generation, with citation accuracy consistently at 0. However, introducing conflict-aware prompting leads to large improvements, enabling models to better address multiple valid answers and improve citation accuracy, while maintaining their ability to predict correct answers. These findings highlight the challenges and opportunities in developing LLMs that can handle ambiguity and provide reliable source citations. Our benchmarking study provides critical insights and sets a foundation for future improvements in trustworthy and interpretable QA systems. 

**Abstract (ZH)**: 在复杂且现实的任务中基准测试现代大型语言模型（LLMs）对于推动其发展至关重要。本研究旨在评估最先进LLMs在具有源引文的模糊环境中问答（QA）任务中的事实在准确性及引文性能。我们使用了三种最近发布的数据集——DisentQA-DupliCite、DisentQA-ParaCite 和 AmbigQA-Cite，涵盖了各种现实世界的模糊性。我们分析了两种领先的LLM，GPT-4o-mini 和 Claude-3.5，在这些数据集上的表现。研究结果表明，较大的、较新的模型在模糊环境中一致能够预测至少一个正确答案，但难以处理具有多个正确答案的情况。此外，所有模型在引文生成方面的表现都极其不佳，引文准确性始终为0。然而，引入冲突感知的提示对模型表现产生了显著改善，使模型更好地处理多重正确答案并提高引文准确性，同时保持预测正确答案的能力。这些发现突显了开发能够处理模糊性和提供可靠引文数据的LLMs所面临的挑战和机遇。我们的基准测试研究提供了重要的见解，并为未来的可信和可解释的QA系统改进奠定了基础。 

---
# StructTest: Benchmarking LLMs' Reasoning through Compositional Structured Outputs 

**Title (ZH)**: StructTest: 通过组成结构化输出对LLMs推理能力进行基准测试 

**Authors**: Hailin Chen, Fangkai Jiao, Mathieu Ravaut, Nawshad Farruque, Xuan Phi Nguyen, Chengwei Qin, Manan Dey, Bosheng Ding, Caiming Xiong, Shafiq Joty, Yingbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.18011)  

**Abstract**: The rapid development of large language models (LLMs) necessitates robust, unbiased, and scalable methods for evaluating their capabilities. However, human annotations are expensive to scale, model-based evaluations are prone to biases in answer style, while target-answer-based benchmarks are vulnerable to data contamination and cheating. To address these limitations, we propose StructTest, a novel benchmark that evaluates LLMs on their ability to produce compositionally specified structured outputs as an unbiased, cheap-to-run and difficult-to-cheat measure. The evaluation is done deterministically by a rule-based evaluator, which can be easily extended to new tasks. By testing structured outputs across diverse task domains -- including Summarization, Code, HTML and Math -- we demonstrate that StructTest serves as a good proxy for general reasoning abilities, as producing structured outputs often requires internal logical reasoning. We believe that StructTest offers a critical, complementary approach to objective and robust model evaluation. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速发展迫切需要稳健、无偏见且可扩展的评估方法来衡量其能力。然而，人工标注成本高昂，基于模型的评估容易受到答案风格偏见的影响，而基于目标答案的基准则容易受到数据污染和作弊的威胁。为解决这些限制，我们提出了一种名为StructTest的新基准方法，该方法通过一种无偏见、低成本且难以作弊的方式评估LLMs生成组合性结构化输出的能力。评估通过基于规则的评估器完成，可以轻松扩展到新的任务。通过在包括摘要、代码、HTML和数学在内的多个任务领域测试结构化输出，我们证明了StructTest可以作为一般推理能力的良好代理指标，因为生成结构化输出通常需要内部逻辑推理。我们认为，StructTest提供了客观且稳健模型评估的一种关键补充方法。 

---
# CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language Models 

**Title (ZH)**: CARL-GT：评估大型语言模型的因果推理能力 

**Authors**: Ruibo Tu, Hedvig Kjellström, Gustav Eje Henter, Cheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17970)  

**Abstract**: Causal reasoning capabilities are essential for large language models (LLMs) in a wide range of applications, such as education and healthcare. But there is still a lack of benchmarks for a better understanding of such capabilities. Current LLM benchmarks are mainly based on conversational tasks, academic math tests, and coding tests. Such benchmarks evaluate LLMs in well-regularized settings, but they are limited in assessing the skills and abilities to solve real-world problems. In this work, we provide a benchmark, named by CARL-GT, which evaluates CAusal Reasoning capabilities of large Language models using Graphs and Tabular data. The benchmark has a diverse range of tasks for evaluating LLMs from causal graph reasoning, knowledge discovery, and decision-making aspects. In addition, effective zero-shot learning prompts are developed for the tasks. In our experiments, we leverage the benchmark for evaluating open-source LLMs and provide a detailed comparison of LLMs for causal reasoning abilities. We found that LLMs are still weak in casual reasoning, especially with tabular data to discover new insights. Furthermore, we investigate and discuss the relationships of different benchmark tasks by analyzing the performance of LLMs. The experimental results show that LLMs have different strength over different tasks and that their performance on tasks in different categories, i.e., causal graph reasoning, knowledge discovery, and decision-making, shows stronger correlation than tasks in the same category. 

**Abstract (ZH)**: 因果推理能力对于大型语言模型（LLMs）在各种应用中，如教育和医疗领域，至关重要。然而，目前尚缺乏相关的基准测试以更好地理解这种能力。当前的LLM基准主要基于对话任务、学术数学测试和编程测试。这些基准测试在设定良好的规则环境中评估LLM的能力，但它们在评估解决实际问题所需技能和能力方面存在局限性。在本研究中，我们提供了一个名为CARL-GT的基准测试，该基准测试使用图形和表格数据评估大型语言模型的因果推理能力。基准测试涵盖了一系列多样化的任务，从因果图推理、知识发现和决策制定方面评估LLM。此外，还开发了有效的零样本学习提示以应用于这些任务。在实验中，我们利用基准测试评估开源LLM，并详细比较了LLM在因果推理能力方面的表现。我们发现LLM在因果推理方面仍然较弱，特别是在从表格数据中发现新见解方面表现不佳。此外，我们通过分析LLM在不同基准测试任务中的表现，探讨了不同任务之间的关系。实验结果表明，LLM在不同任务上的表现各异，并且它们在不同类别任务（因果图推理、知识发现和决策制定）上的表现相关性更强，而不是在同一个类别内的任务。 

---
# Path-of-Thoughts: Extracting and Following Paths for Robust Relational Reasoning with Large Language Models 

**Title (ZH)**: 基于思维路径：从大型语言模型中提取和追踪路径以实现稳健的关系推理 

**Authors**: Ge Zhang, Mohammad Ali Alomrani, Hongjian Gu, Jiaming Zhou, Yaochen Hu, Bin Wang, Qun Liu, Mark Coates, Yingxue Zhang, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2412.17963)  

**Abstract**: Large language models (LLMs) possess vast semantic knowledge but often struggle with complex reasoning tasks, particularly in relational reasoning problems such as kinship or spatial reasoning. In this paper, we present Path-of-Thoughts (PoT), a novel framework designed to tackle relation reasoning by decomposing the task into three key stages: graph extraction, path identification, and reasoning. Unlike previous approaches, PoT efficiently extracts a task-agnostic graph that identifies crucial entities, relations, and attributes within the problem context. Subsequently, PoT identifies relevant reasoning chains within the graph corresponding to the posed question, facilitating inference of potential answers. Experimental evaluations on four benchmark datasets, demanding long reasoning chains, demonstrate that PoT surpasses state-of-the-art baselines by a significant margin (maximum 21.3%) without necessitating fine-tuning or extensive LLM calls. Furthermore, as opposed to prior neuro-symbolic methods, PoT exhibits improved resilience against LLM errors by leveraging the compositional nature of graphs. 

**Abstract (ZH)**: 大语言模型（LLMs）蕴含了广泛的语言知识，但在处理复杂的推理任务时常常表现不佳，特别是在亲缘关系或空间推理等关系推理问题上。本文提出了一种名为路径思维（Path-of-Thoughts, PoT）的新型框架，旨在通过将任务分解为三个关键阶段来应对关系推理问题：图提取、路径识别和推理。与之前的几种方法不同，PoT 有效地提取了一个任务无关的图，该图能够识别问题上下文中关键实体、关系和属性。随后，PoT 在图中识别出与提出的问题相关的推理链，从而有助于潜在答案的推断。通过在四个基准数据集上进行实证评估，这些数据集要求处理较长的推理链，结果显示PoT 在不需微调或大量调用LLM的情况下显著超越了现有最先进的基线方法（最高超过21.3%）。此外，与之前的神经符号方法相比，PoT 通过利用图的组合性质展示了更好的鲁棒性，能够更好地抵御LLM的错误。 

---
# Enhancing Knowledge Distillation for LLMs with Response-Priming Prompting 

**Title (ZH)**: 用响应引导提示增强知识蒸馏以优化大规模语言模型 

**Authors**: Vijay Goyal, Mustafa Khan, Aprameya Tirupati, Harveer Saini, Michael Lam, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17846)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing (NLP) tasks. However, these models are often difficult to deploy due to significant computational requirements and resource constraints. Knowledge distillation (KD) is an effective technique for transferring the performance of larger LLMs to smaller models. Traditional KD methods primarily focus on the direct output of the teacher model, with little emphasis on the role of prompting during knowledge transfer. In this paper, we propose a set of novel response-priming prompting strategies applied in the knowledge distillation pipeline to enhance the performance of student models. Our approach fine-tunes a smaller Llama 3.1 8B Instruct model by distilling knowledge from a quantized Llama 3.1 405B Instruct teacher model. We apply LoRA optimization and evaluate on the GSM8K benchmark. Experimental results demonstrate that integrating reasoning-eliciting prompting into the proposed KD pipeline significantly improves student model performance, offering an efficient way to deploy powerful models in resource-constrained environments. We find that Ground Truth prompting results in a 55\% performance increase on GSM8K for a distilled Llama 3.1 8B Instruct compared to the same model distilled without prompting. A thorough investigation into the self-attention layers of the student models indicates that the more successful prompted models tend to exhibit certain positive behaviors inside their attention heads which can be tied to their increased accuracy. Our implementation can be found at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的语言处理任务中表现出色。然而，这些模型由于计算需求大和资源限制，往往难以部署。知识蒸馏（KD）是一种有效的方法，可以将大型LLM的性能转移到较小的模型中。传统的方法主要关注教师模型的直接输出，而对知识转移过程中提示的作用关注较少。本文提出了一套新颖的响应促进提示策略，应用于知识蒸馏管道中，以提高学生模型的性能。我们的方法通过对量化后的Llama 3.1 405B Instruct教师模型进行知识蒸馏，来微调一个较小的Llama 3.1 8B Instruct学生模型，并应用LoRA优化，在GSM8K基准上进行评估。实验结果表明，将推理促进提示整合到所提出的KD管道中，显著提高了学生模型的性能，为在资源受限环境中部署强大模型提供了有效途径。我们发现，使用Ground Truth提示可以将蒸馏后的Llama 3.1 8B Instruct模型在GSM8K上的性能提高55%。通过对学生模型的自我注意层进行深入研究，发现更成功的提示模型通常在其注意头中表现出某些积极的行为，这些行为与其更高的准确性有关。我们的实现代码可以在这个链接访问：[这里提供一个链接]。 

---
# Look Ahead Text Understanding and LLM Stitching 

**Title (ZH)**: 展望文本理解与大规模语言模型整合 

**Authors**: Junlin Julian Jiang, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.17836)  

**Abstract**: This paper proposes a look ahead text understanding problem with look ahead section identification (LASI) as an example. This problem may appear in generative AI as well as human interactions, where we want to understand the direction of a developing text or conversation. We tackle the problem using transformer-based LLMs. We show that LASI is more challenging than classic section identification (SI). We argue that both bidirectional contextual information (e.g., BERT) and unidirectional predictive ability (e.g., GPT) will benefit the task. We propose two approaches to stitch together BERT and GPT. Experiments show that our approach outperforms the established models, especially when there is noise in the text (which is often the case for developing text in generative AI). Our paper sheds light on other look ahead text understanding tasks that are important to social media, such as look ahead sentiment classification, and points out the opportunities to leverage pre-trained LLMs through stitching. 

**Abstract (ZH)**: 本文以前瞻段落识别（LASI）为例，提出了一个前瞻文本理解问题。该问题可能出现在生成型AI和人际交往中，我们希望通过理解正在发展的文本或对话的方向来解决这一问题。我们使用基于变换器的大型语言模型（LLM）来解决这一问题。研究表明，LASI 比经典的段落识别（SI）更具挑战性。我们认为，双向上下文信息（如 BERT）和单向预测能力（如 GPT）都将有助于这一任务。我们提出了两种方法将BERT和GPT结合在一起。实验结果表明，我们的方法在文本存在噪声（在生成型AI中的发展文本通常存在这种情况）的情况下优于现有模型。我们的研究为社交媒体中的其他前瞻文本理解任务提供了新的视角，如前瞻情感分类，并指出了通过结合预训练的大型语言模型来利用这些机会的可能性。 

---
