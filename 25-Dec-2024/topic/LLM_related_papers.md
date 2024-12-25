# Explainable Multi-Modal Data Exploration in Natural Language via LLM Agent 

**Title (ZH)**: 通过大语言模型代理进行可解释的多模态数据自然语言探索 

**Authors**: Farhad Nooralahzadeh, Yi Zhang, Jonathan Furst, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2412.18428)  

**Abstract**: International enterprises, organizations, or hospitals collect large amounts of multi-modal data stored in databases, text documents, images, and videos. While there has been recent progress in the separate fields of multi-modal data exploration as well as in database systems that automatically translate natural language questions to database query languages, the research challenge of querying database systems combined with other unstructured modalities such as images in natural language is widely unexplored.
In this paper, we propose XMODE - a system that enables explainable, multi-modal data exploration in natural language. Our approach is based on the following research contributions: (1) Our system is inspired by a real-world use case that enables users to explore multi-modal information systems. (2) XMODE leverages a LLM-based agentic AI framework to decompose a natural language question into subtasks such as text-to-SQL generation and image analysis. (3) Experimental results on multi-modal datasets over relational data and images demonstrate that our system outperforms state-of-the-art multi-modal exploration systems, excelling not only in accuracy but also in various performance metrics such as query latency, API costs, planning efficiency, and explanation quality, thanks to the more effective utilization of the reasoning capabilities of LLMs. 

**Abstract (ZH)**: 国际企业、组织或医院收集了大量的多模态数据，这些数据存储在数据库、文本文档、图像和视频中。虽然在多模态数据探索和自然语言到数据库查询语言自动生成方面各自取得了一定进展，但将数据库系统与其他未结构化的模态如图像结合查询的研究挑战仍然没有得到广泛探索。

在本文中，我们提出了一种名为XMODE的系统，旨在通过自然语言实现可解释的多模态数据探索。我们的方法主要基于以下研究贡献：（1）我们的系统借鉴了一个实际应用案例，使用户能够探索多模态信息系统。（2）XMODE利用一个基于大语言模型（LLM）的代理AI框架，将自然语言问题分解为子任务，例如文本到SQL生成和图像分析。（3）通过针对关系数据和图像的多模态数据集进行实验，结果显示我们的系统在准确性和多种性能指标（如查询延迟、API成本、计划效率和解释质量）方面均优于现有最先进的多模态探索系统，得益于更有效利用了LLMs的推理能力。 

---
# Improving Multi-Step Reasoning Abilities of Large Language Models with Direct Advantage Policy Optimization 

**Title (ZH)**: 通过直接优势策略优化提升大型语言模型的多步推理能力 

**Authors**: Jiacai Liu, Chaojie Wang, Chris Yuhao Liu, Liang Zeng, Rui Yan, Yiwen Sun, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.18279)  

**Abstract**: The role of reinforcement learning (RL) in enhancing the reasoning of large language models (LLMs) is becoming increasingly significant. Despite the success of RL in many scenarios, there are still many challenges in improving the reasoning of LLMs. One challenge is the sparse reward, which makes optimization difficult for RL and necessitates a large amount of data samples. Another challenge stems from the inherent instability of RL, particularly when using Actor-Critic (AC) methods to derive optimal policies, which often leads to unstable training processes. To address these issues, we introduce Direct Advantage Policy Optimization (DAPO), an novel step-level offline RL algorithm. Unlike standard alignment that rely solely outcome rewards to optimize policies (such as DPO), DAPO employs a critic function to predict the reasoning accuracy at each step, thereby generating dense signals to refine the generation strategy. Additionally, the Actor and Critic components in DAPO are trained independently, avoiding the co-training instability observed in standard AC algorithms like PPO. We train DAPO on mathematical and code query datasets and then evaluate its performance on multiple benchmarks. Our results show that DAPO can effectively enhance the mathematical and code capabilities on both SFT models and RL models, demonstrating the effectiveness of DAPO. 

**Abstract (ZH)**: reinforcement 学习（RL）在增强大语言模型（LLMs）推理能力方面的作用变得越来越重要。尽管在许多场景中RL取得了成功，但在提高LLMs的推理能力方面仍然存在许多挑战。其中一个挑战是稀疏奖励，这使得RL的优化变得困难，并需要大量的数据样本。另一个挑战源自于RL内在的不稳定性，特别是在使用Actor-Critic（AC）方法推导最优策略时，通常会导致训练过程的不稳定。为解决这些问题，我们引入了一种新颖的离线RL算法——直接优势策略优化（DAPO）。与依赖单一结果奖励进行策略优化的标准对齐方法（如DPO）不同，DAPO 使用一个评论器函数来预测每一步的推理准确性，从而生成密集信号以改进生成策略。此外，DAPO 中的Actor和Critic组件是独立训练的，避免了标准AC算法（如PPO）中观察到的协同训练不稳定现象。我们使用数学和代码查询数据集训练DAPO，并在多个基准上评估其性能。结果表明，DAPO 能够有效地增强SFT模型和RL模型的数学和代码能力，证明了DAPO的有效性。 

---
# Annotating References to Mythological Entities in French Literature 

**Title (ZH)**: 法国文学中神话实体的标注研究 

**Authors**: Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2412.18270)  

**Abstract**: In this paper, we explore the relevance of large language models (LLMs) for annotating references to Roman and Greek mythological entities in modern and contemporary French literature. We present an annotation scheme and demonstrate that recent LLMs can be directly applied to follow this scheme effectively, although not without occasionally making significant analytical errors. Additionally, we show that LLMs (and, more specifically, ChatGPT) are capable of offering interpretative insights into the use of mythological references by literary authors. However, we also find that LLMs struggle to accurately identify relevant passages in novels (when used as an information retrieval engine), often hallucinating and generating fabricated examples-an issue that raises significant ethical concerns. Nonetheless, when used carefully, LLMs remain valuable tools for performing annotations with high accuracy, especially for tasks that would be difficult to annotate comprehensively on a large scale through manual methods alone. 

**Abstract (ZH)**: 在本文中，我们探讨了大规模语言模型（LLMs）在标注现代及当代法语文坛中提到的罗马和希腊神话实体方面的相关性。我们提出了一种标注方案，并展示了近期的LLMs可以直接应用于遵循该方案，尽管在分析过程中偶尔会出现严重的分析错误。此外，我们还证明了LLMs（更具体地说，是ChatGPT）能够为文学作者在使用神话引用方面的解读提供有价值的见解。然而，我们也发现，当LLMs作为信息检索工具使用时，它们在准确识别小说中相关段落时遇到很大的困难，常常会产生幻觉并生成虚假例子——这一问题引起了重大的伦理关注。尽管如此，当谨慎使用时，LLMs仍然是进行高精度标注的重要工具，尤其是对于那些通过手工方法难以全面标注的任务。 

---
# VISION: A Modular AI Assistant for Natural Human-Instrument Interaction at Scientific User Facilities 

**Title (ZH)**: VISION：一种适用于科学用户设施的模块化人工智能助手，实现自然的人机乐器交互 

**Authors**: Shray Mathur, Noah van der Vleuten, Kevin Yager, Esther Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2412.18161)  

**Abstract**: Scientific user facilities, such as synchrotron beamlines, are equipped with a wide array of hardware and software tools that require a codebase for human-computer-interaction. This often necessitates developers to be involved to establish connection between users/researchers and the complex instrumentation. The advent of generative AI presents an opportunity to bridge this knowledge gap, enabling seamless communication and efficient experimental workflows. Here we present a modular architecture for the Virtual Scientific Companion (VISION) by assembling multiple AI-enabled cognitive blocks that each scaffolds large language models (LLMs) for a specialized task. With VISION, we performed LLM-based operation on the beamline workstation with low latency and demonstrated the first voice-controlled experiment at an X-ray scattering beamline. The modular and scalable architecture allows for easy adaptation to new instrument and capabilities. Development on natural language-based scientific experimentation is a building block for an impending future where a science exocortex -- a synthetic extension to the cognition of scientists -- may radically transform scientific practice and discovery. 

**Abstract (ZH)**: 科学用户设施，如同步加速器光束线，配备了广泛的硬件和软件工具，需要一套用于人机交互的代码库。这通常需要开发人员参与，以建立用户/研究人员与复杂仪器之间的连接。生成式人工智能的出现为弥合这一知识差距提供了机会，使沟通更加顺畅并提高实验流程的效率。在此，我们通过组装多个AI赋能的认知模块来构建多模块架构的虚拟科学伴侣（VISION），这些模块分别专门配置大型语言模型（LLMs）以执行特定任务。借助VISION，我们在光束线工作站上实现了基于LLM的操作，并首次在X射线散射光束线上展示了语音控制的实验。这种模块化和可扩展的架构允许轻松适应新的仪器和功能。基于自然语言的科学实验开发是构建未来科学外皮（科学认知的合成扩展）的关键组成部分，它可能彻底改变科学研究和发现的方式。 

---
# AIGT: AI Generative Table Based on Prompt 

**Title (ZH)**: AIGT：基于提示的AI生成表格 

**Authors**: Mingming Zhang, Zhiqing Xiao, Guoshan Lu, Sai Wu, Weiqiang Wang, Xing Fu, Can Yi, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18111)  

**Abstract**: Tabular data, which accounts for over 80% of enterprise data assets, is vital in various fields. With growing concerns about privacy protection and data-sharing restrictions, generating high-quality synthetic tabular data has become essential. Recent advancements show that large language models (LLMs) can effectively gener-ate realistic tabular data by leveraging semantic information and overcoming the challenges of high-dimensional data that arise from one-hot encoding. However, current methods do not fully utilize the rich information available in tables. To address this, we introduce AI Generative Table (AIGT) based on prompt enhancement, a novel approach that utilizes meta data information, such as table descriptions and schemas, as prompts to generate ultra-high quality synthetic data. To overcome the token limit constraints of LLMs, we propose long-token partitioning algorithms that enable AIGT to model tables of any scale. AIGT achieves state-of-the-art performance on 14 out of 20 public datasets and two real industry datasets within the Alipay risk control system. 

**Abstract (ZH)**: 表数据约占企业数据资产的80%，在各个领域都具有重要作用。随着对隐私保护和数据共享限制的日益关注，生成高质量的合成表数据变得至关重要。最近的研究表明，大规模语言模型（LLMs）可以通过利用语义信息并克服一-hot编码带来的高维数据挑战，有效地生成具有现实感的表数据。然而，当前的方法并未充分利用表格中丰富的信息。为解决这一问题，我们提出了基于提示增强的AI生成表格（AIGT）方法，这是一种新型的方法，通过利用元数据信息（如表描述和结构）作为提示来生成超高质量的合成数据。为了克服LLMs的标记限制，我们提出了长标记分割算法，从而使AIGT能够模型任何规模的表格。实验结果显示，AIGT 在20个公开数据集中的14个以及支付宝风控系统中的两个真实企业数据集上取得了最先进的性能。 

---
# Real-world Deployment and Evaluation of PErioperative AI CHatbot (PEACH) -- a Large Language Model Chatbot for Perioperative Medicine 

**Title (ZH)**: 在临床 perioperative 医学中大规模语言模型聊天机器人 PErioperative AI CHatbot (PEACH) 的实际部署与评估 

**Authors**: Yu He Ke, Liyuan Jin, Kabilan Elangovan, Bryan Wen Xi Ong, Chin Yang Oh, Jacqueline Sim, Kenny Wei-Tsen Loh, Chai Rick Soh, Jonathan Ming Hua Cheng, Aaron Kwang Yang Lee, Daniel Shu Wei Ting, Nan Liu, Hairil Rizal Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2412.18096)  

**Abstract**: Large Language Models (LLMs) are emerging as powerful tools in healthcare, particularly for complex, domain-specific tasks. This study describes the development and evaluation of the PErioperative AI CHatbot (PEACH), a secure LLM-based system integrated with local perioperative guidelines to support preoperative clinical decision-making. PEACH was embedded with 35 institutional perioperative protocols in the secure Claude 3.5 Sonet LLM framework within Pair Chat (developed by Singapore Government) and tested in a silent deployment with real-world data. Accuracy, safety, and usability were assessed. Deviations and hallucinations were categorized based on potential harm, and user feedback was evaluated using the Technology Acceptance Model (TAM). Updates were made after the initial silent deployment to amend one protocol.
In 240 real-world clinical iterations, PEACH achieved a first-generation accuracy of 97.5% (78/80) and an overall accuracy of 96.7% (232/240) across three iterations. The updated PEACH demonstrated improved accuracy of 97.9% (235/240), with a statistically significant difference from the null hypothesis of 95% accuracy (p = 0.018, 95% CI: 0.952-0.991). Minimal hallucinations and deviations were observed (both 1/240 and 2/240, respectively). Clinicians reported that PEACH expedited decisions in 95% of cases, and inter-rater reliability ranged from kappa 0.772-0.893 within PEACH and 0.610-0.784 among attendings.
PEACH is an accurate, adaptable tool that enhances consistency and efficiency in perioperative decision-making. Future research should explore its scalability across specialties and its impact on clinical outcomes. 

**Abstract (ZH)**: 大型语言模型（LLMs）正逐渐成为医疗领域中强有力的工具，尤其是在处理复杂且领域特定的任务时。本研究描述了PErioperative AI CHatbot（PEACH）系统的发展与评价，PEACH是一种集成了本地围手术期指南的安全LLM系统，旨在支持术前临床决策。PEACH系统在安全的Claude 3.5 Sonet LLM框架（由新加坡政府开发的Pair Chat中的一个集成部分）内嵌入了35家机构的围手术期协议，并在真实数据环境下进行了静默部署测试。评估了其准确度、安全性与可用性。根据潜在的危害性，对偏离和幻觉进行了分类，并通过技术接受模型（TAM）评估了用户反馈。在首次静默部署后，对协议进行了更新。

在240次真实临床迭代中，PEACH的一代准确率为97.5%（78/80），整体准确率为96.7%（232/240）。更新后的PEACH准确率为97.9%（235/240），显著优于95%的零假设（p = 0.018，95% CI：0.952-0.991）。观察到的幻觉和偏差最少（分别为1/240和2/240）。临床医生报告，在95%的情况下，PEACH加速了决策过程，PEACH内部的临诊者者一致可靠性范围为κ值0.772-0.893，临诊者间的可靠性范围为0.610-0.784。

PEACH是一种准确且适应性强的工具，能增强围手术期决策的一致性和效率。未来研究应探讨其在不同专科领域的扩展性和对临床结果的影响。 

---
# Dynamic Multi-Agent Orchestration and Retrieval for Multi-Source Question-Answer Systems using Large Language Models 

**Title (ZH)**: 使用大型语言模型的动态多Agent协同与检索以构建多源问答系统 

**Authors**: Antony Seabra, Claudio Cavalcante, Joao Nepomuceno, Lucas Lago, Nicolaas Ruberg, Sergio Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17964)  

**Abstract**: We propose a methodology that combines several advanced techniques in Large Language Model (LLM) retrieval to support the development of robust, multi-source question-answer systems. This methodology is designed to integrate information from diverse data sources, including unstructured documents (PDFs) and structured databases, through a coordinated multi-agent orchestration and dynamic retrieval approach. Our methodology leverages specialized agents-such as SQL agents, Retrieval-Augmented Generation (RAG) agents, and router agents - that dynamically select the most appropriate retrieval strategy based on the nature of each query. To further improve accuracy and contextual relevance, we employ dynamic prompt engineering, which adapts in real time to query-specific contexts. The methodology's effectiveness is demonstrated within the domain of Contract Management, where complex queries often require seamless interaction between unstructured and structured data. Our results indicate that this approach enhances response accuracy and relevance, offering a versatile and scalable framework for developing question-answer systems that can operate across various domains and data sources. 

**Abstract (ZH)**: 我们提出了一种方法论，该方法论结合了大型语言模型（LLM）检索领域的多项先进技术，以支持稳健的多源问答系统的发展。该方法论旨在通过协调多智能体编排和动态检索方式，整合来自多种数据源的信息，包括未结构化的文档（PDF）和结构化数据库。该方法论利用了专门的智能体——如SQL智能体、检索增强生成（RAG）智能体和路由智能体——这些智能体能够根据每个查询的性质动态选择最合适的检索策略。为进一步提高准确性和上下文相关性，我们采用了动态提示工程，这种技术能够实时适应查询特定的上下文。该方法论的有效性在合同管理领域得到验证，该领域中复杂的查询往往需要无缝地交互未结构化和结构化数据。我们的结果表明，这种方法提高了响应的准确性和相关性，提供了一个适用于多种领域和数据源的灵活且可扩展的框架，用于开发问答系统。 

---
# Contrato360 2.0: A Document and Database-Driven Question-Answer System using Large Language Models and Agents 

**Title (ZH)**: Contrato360 2.0：一种基于文档和数据库的大语言模型及智能体驱动的问答系统 

**Authors**: Antony Seabra, Claudio Cavalcante, Joao Nepomuceno, Lucas Lago, Nicolaas Ruberg, Sergio Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17942)  

**Abstract**: We present a question-and-answer (Q\&A) application designed to support the contract management process by leveraging combined information from contract documents (PDFs) and data retrieved from contract management systems (database). This data is processed by a large language model (LLM) to provide precise and relevant answers. The accuracy of these responses is further enhanced through the use of Retrieval-Augmented Generation (RAG), text-to-SQL techniques, and agents that dynamically orchestrate the workflow. These techniques eliminate the need to retrain the language model. Additionally, we employed Prompt Engineering to fine-tune the focus of responses. Our findings demonstrate that this multi-agent orchestration and combination of techniques significantly improve the relevance and accuracy of the answers, offering a promising direction for future information systems. 

**Abstract (ZH)**: 我们提出了一种问答（Q&A）应用，该应用通过结合合同文件（PDF）和从合同管理系统（数据库）中检索的数据来支持合同管理流程。这些数据经过大规模语言模型（LLM）处理，提供精准且相关的答案。通过使用检索增强生成（RAG）、文本到SQL技术以及能够动态协调工作流的代理，这些方法进一步提高了答案的准确性。这些技术消除了重新训练语言模型的需要。此外，我们采用了提示工程（Prompt Engineering）来细化答案的焦点。我们的研究发现，这种多代理协调及技术组合显著提高了答案的相关性和准确性，为未来信息系统的发展提供了有前景的方向。 

---
# Decentralized Intelligence in GameFi: Embodied AI Agents and the Convergence of DeFi and Virtual Ecosystems 

**Title (ZH)**: GameFi中去中心化智能：具身人工智能代理及DeFi与虚拟生态系统融合 

**Authors**: Fernando Jia, Jade Zheng, Florence Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.18601)  

**Abstract**: In the rapidly evolving landscape of GameFi, a fusion of gaming and decentralized finance (DeFi), there exists a critical need to enhance player engagement and economic interaction within gaming ecosystems. Our GameFi ecosystem aims to fundamentally transform this landscape by integrating advanced embodied AI agents into GameFi platforms. These AI agents, developed using cutting-edge large language models (LLMs), such as GPT-4 and Claude AI, are capable of proactive, adaptive, and contextually rich interactions with players. By going beyond traditional scripted responses, these agents become integral participants in the game's narrative and economic systems, directly influencing player strategies and in-game economies. We address the limitations of current GameFi platforms, which often lack immersive AI interactions and mechanisms for community engagement or creator monetization. Through the deep integration of AI agents with blockchain technology, we establish a consensus-driven, decentralized GameFi ecosystem. This ecosystem empowers creators to monetize their contributions and fosters democratic collaboration among players and creators. Furthermore, by embedding DeFi mechanisms into the gaming experience, we enhance economic participation and provide new opportunities for financial interactions within the game. Our approach enhances player immersion and retention and advances the GameFi ecosystem by bridging traditional gaming with Web3 technologies. By integrating sophisticated AI and DeFi elements, we contribute to the development of more engaging, economically robust, and community-centric gaming environments. This project represents a significant advancement in the state-of-the-art in GameFi, offering insights and methodologies that can be applied throughout the gaming industry. 

**Abstract (ZH)**: 在快速发展的GameFi（将游戏和去中心化金融DeFi相结合）领域，迫切需要提升玩家参与度和经济互动。我们的GameFi生态系统旨在通过将先进的具身人工智能代理整合到GameFi平台中，从根本上改变这一领域。这些基于前沿大型语言模型（LLMs）如GPT-4和Claude AI开发的人工智能代理，能够在玩家之间实现主动、适应性强且情景丰富的互动。通过超越传统的预编程响应，这些代理将成为游戏叙事和经济系统的积极参与者，直接影响玩家策略和游戏内经济。

我们解决了当前GameFi平台普遍存在的问题，这些平台往往缺乏沉浸式的人工智能互动和社区参与或创作者变现的机制。通过深度整合人工智能代理与区块链技术，我们建立了一个共识驱动的、去中心化的GameFi生态系统。该生态系统赋予创作者变现其贡献的机会，并促进玩家和创作者之间的民主合作。此外，通过将DeFi机制嵌入游戏体验中，我们增强了经济参与并为游戏中的金融互动提供了新的机会。通过整合先进的人工智能和DeFi元素，我们的方法提升了玩家的沉浸感和留存率，并通过传统游戏和Web3技术的结合，推动了GameFi生态系统的进步。该项目在GameFi前沿技术方面实现了重要突破，提供了可用于整个游戏行业的见解和方法论。

总之，我们的研究和创新为更加吸引人、经济上更稳健且更具社区导向的游戏环境做出了贡献。该项目代表了GameFi领域的重大进展，提供了可应用于整个游戏行业的洞见和方法。 

---
# A Paragraph is All It Takes: Rich Robot Behaviors from Interacting, Trusted LLMs 

**Title (ZH)**: 仅需一段文本：来自相互信任的大型语言模型丰富机器人行为 

**Authors**: OpenMind, Shaohong Zhong, Adam Zhou, Boyuan Chen, Homin Luo, Jan Liphardt  

**Link**: [PDF](https://arxiv.org/pdf/2412.18588)  

**Abstract**: Large Language Models (LLMs) are compact representations of all public knowledge of our physical environment and animal and human behaviors. The application of LLMs to robotics may offer a path to highly capable robots that perform well across most human tasks with limited or even zero tuning. Aside from increasingly sophisticated reasoning and task planning, networks of (suitably designed) LLMs offer ease of upgrading capabilities and allow humans to directly observe the robot's thinking. Here we explore the advantages, limitations, and particularities of using LLMs to control physical robots. The basic system consists of four LLMs communicating via a human language data bus implemented via web sockets and ROS2 message passing. Surprisingly, rich robot behaviors and good performance across different tasks could be achieved despite the robot's data fusion cycle running at only 1Hz and the central data bus running at the extremely limited rates of the human brain, of around 40 bits/s. The use of natural language for inter-LLM communication allowed the robot's reasoning and decision making to be directly observed by humans and made it trivial to bias the system's behavior with sets of rules written in plain English. These rules were immutably written into Ethereum, a global, public, and censorship resistant Turing-complete computer. We suggest that by using natural language as the data bus among interacting AIs, and immutable public ledgers to store behavior constraints, it is possible to build robots that combine unexpectedly rich performance, upgradability, and durable alignment with humans. 

**Abstract (ZH)**: 大规模语言模型（LLMs）是所有关于我们物理环境以及动物和人类行为的公开知识的紧凑表示。将LLMs应用于机器人技术可能会提供一条途径，即构建出能够在大多数人类任务上表现优异的机器人，而无需或只需最少的微调。除了不断增强的推理和任务规划能力外，适配设计的LLM网络还提供了升级能力的简便性，并允许人类直接观察机器人的思考过程。本文探讨了使用LLMs控制物理机器人的优势、限制和独特之处。基本系统由四个LLM通过WebSocket和ROS2消息传递实现的人类语言数据总线进行通信。令人惊讶的是，尽管机器人数据融合循环的运行频率仅为1Hz，且中心数据总线的运行速率受限于类似人类大脑的极低速率（约40比特/秒），但仍然能够实现丰富的机器人行为和在不同任务中的良好表现。利用自然语言促进LLM之间的通信，使得人类可以直接观察机器人的推理和决策过程，并且使用简单的英语书写规则即可轻松偏移系统的行为。这些规则被不可变更地写入了以太坊这样一个全球性、公众性和抗审查的图灵完备计算机中。我们认为，通过在交互AI之间使用自然语言作为数据总线，并将行为约束存储在不可变的公共账本中，有可能构建出兼具惊人性能、易于升级以及持久与人类一致性的机器人。 

---
# How Well Do LLMs Generate Code for Different Application Domains? Benchmark and Evaluation 

**Title (ZH)**: 不同应用场景下大型语言模型生成代码的能力如何？基准测试与评估 

**Authors**: Dewu Zheng, Yanlin Wang, Ensheng Shi, Hongyu Zhang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.18573)  

**Abstract**: Recently, an increasing number of AI-driven programming assistants powered by code LLMs have been integrated into various real-world software development environments, significantly boosting developer productivity. However, existing code generation benchmarks primarily focus on general-purpose scenarios, leaving the code generation performance of LLMs for specific application domains largely unknown. In this paper, we introduce a new benchmark, MultiCodeBench, to fill this gap. MultiCodeBench comprises 2,400 programming tasks, covering 12 popular software development domains and 15 programming languages. Specifically, we perform in-depth research to identify these 12 application domains. Given that each domain may involve multiple technical frameworks, and that different frameworks present distinct challenges in the coding process, we categorize the commonly used frameworks and platforms within each domain. We then sample programming problems from GitHub repositories related to these subdomains. To ensure the quality of the tasks and mitigate data leakage issues, we invite annotators to rewrite the docstrings for each task in MultiCodeBench. Additionally, we build a static analysis-based dependency parsing tool to extract the dependencies in the ground truth for each task, enabling deeper performance analysis. Through extensive experiments on MultiCodeBench with eleven representative mainstream LLMs, we reveal the code generation performance of the LLMs across different application domains, providing practical insights for developers in downstream fields when selecting LLMs. Furthermore, we analyze the reasons behind the models' failures in completing software application development tasks, offering guidance for model developers to enhance domain-specific code generation capabilities. 

**Abstract (ZH)**: 近年来，由代码大型语言模型（Code LLMs）驱动的AI编程助手被越来越多地集成到各种实际软件开发环境中，显著提升了开发者的生产力。然而，现有的代码生成基准主要集中在通用场景上，使得代码生成性能在特定应用领域中的表现仍然 largely unknown。本文旨在填补这一空白，引入了一个新的基准测试MultCodeBench。MultCodeBench 包含2,400个编程任务，涵盖了12个流行的软件开发领域和15种编程语言。具体来说，我们进行了深入研究以识别这12个应用领域。鉴于每个领域可能存在多种技术框架，不同框架在编码过程中呈现出不同的挑战，我们对每个领域的常用框架和平台进行了分类。然后，我们从与这些子领域相关的GitHub仓库中抽取编程问题。为确保任务质量并减轻数据泄漏问题，我们邀请标注员为MultCodeBench中的每个任务重写文档字符串。此外，我们构建了一个基于静态分析的依赖关系解析工具，以提取每个任务的黄金标准中的依赖关系，从而实现更深入的性能分析。通过在MultCodeBench上进行广泛的实验，我们揭示了不同应用领域的LLM的代码生成性能，为下游开发者在选择LLM时提供了实际指导。同时，我们分析了模型未能完成软件应用开发任务的原因，为模型开发者提供指导，以增强领域特定的代码生成能力。 

---
# Token-Budget-Aware LLM Reasoning 

**Title (ZH)**: Token-Budget-Aware LLM推理 

**Authors**: Tingxu Han, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen, Zhenting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18547)  

**Abstract**: Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework, which dynamically estimates token budgets for different problems based on reasoning complexity and uses the estimated token budgets to guide the reasoning process. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: this https URL. 

**Abstract (ZH)**: 逻辑推理对于大型语言模型（LLMs）在广泛任务上的表现至关重要。虽然诸如链式思维（Chain-of-Thought, CoT）等方法通过将问题分解为中间步骤来提高LLM的性能，但也导致了显著的token使用量增加，从而提高了成本。我们发现当前LLMs的推理过程过长，并且可以通过在提示中包含合理的token预算来压缩，但token预算的选择在实际压缩效果中起着关键作用。我们随后提出了一种基于token预算的LLM推理框架，该框架根据推理复杂性动态估计不同问题的token预算，并使用估算的token预算来指导推理过程。实验表明，我们的方法在CoT推理中有效减少了token成本，同时仅造成轻微的性能下降，从而提供了一种在效率和准确性之间平衡的实际解决方案。代码：[此链接](this https URL)。 

---
# Consistency Checks for Language Model Forecasters 

**Title (ZH)**: 语言模型预测器的一致性检查 

**Authors**: Daniel Paleka, Abhimanyu Pallavi Sudhir, Alejandro Alvarez, Vineeth Bhat, Adam Shen, Evan Wang, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2412.18544)  

**Abstract**: Forecasting is a task that is difficult to evaluate: the ground truth can only be known in the future. Recent work showing LLM forecasters rapidly approaching human-level performance begs the question: how can we benchmark and evaluate these forecasters instantaneously? Following the consistency check framework, we measure the performance of forecasters in terms of the consistency of their predictions on different logically-related questions. We propose a new, general consistency metric based on arbitrage: for example, if a forecasting AI illogically predicts that both the Democratic and Republican parties have 60% probability of winning the 2024 US presidential election, an arbitrageur can trade against the forecaster's predictions and make a profit. We build an automated evaluation system that generates a set of base questions, instantiates consistency checks from these questions, elicits the predictions of the forecaster, and measures the consistency of the predictions. We then build a standard, proper-scoring-rule forecasting benchmark, and show that our (instantaneous) consistency metrics correlate with LLM forecasters' ground truth Brier scores (which are only known in the future). We also release a consistency benchmark that resolves in 2028, providing a long-term evaluation tool for forecasting. 

**Abstract (ZH)**: 预测是一项难以评估的任务：真实情况只能在未来知晓。近期工作表明，基于大模型的预测工具已经迅速接近人类水平的性能，这引发了一个问题：我们如何能够即时地对这些预测工具进行基准测试和评估？我们依据一致性检查框架，根据预测在不同逻辑相关问题上的一致性来评估预测工具的性能。我们提出了一种基于套利的新的一般性一致性度量方法：例如，如果有预测AI无逻辑地预测2024年美国总统选举中民主党与共和党获胜的可能性均为60%，那么套利者可以通过对冲预测结果实现盈利。我们构建了一个自动化评估系统，生成一系列基础问题，从这些问题中实例化一致性检查，获取预测者的预测，并测量预测的一致性。随后，我们构建了一个标准的拟合评分规则（Proper Scoring Rule）预测基准，并展示了我们（即时的）一致性度量与大模型预测者的未来已知布瑞尔得分（Brier score）相关。我们还发布了一个在2028年发布的基准，提供了一个长期的预测评估工具。 

---
# Multilingual Mathematical Reasoning: Advancing Open-Source LLMs in Hindi and English 

**Title (ZH)**: 多语言数学推理：推进印地语和英语中的开源大规模语言模型 

**Authors**: Avinash Anand, Kritarth Prasad, Chhavi Kirtani, Ashwin R Nair, Manvendra Kumar Nema, Raj Jaiswal, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2412.18415)  

**Abstract**: Large Language Models (LLMs) excel in linguistic tasks but struggle with mathematical reasoning, particularly in non English languages like Hindi. This research aims to enhance the mathematical reasoning skills of smaller, resource efficient open-source LLMs in both Hindi and English. We evaluate models like OpenHathi 7B, LLaMA-2 7B, WizardMath 7B, Mistral 7B, LLeMMa 7B, MAmmoTH 7B, Gemini Pro, and GPT-4 using zero-shot, few-shot chain-of-thought (CoT) methods, and supervised fine-tuning. Our approach incorporates curriculum learning, progressively training models on increasingly difficult problems, a novel Decomposition Strategy to simplify complex arithmetic operations, and a Structured Solution Design that divides solutions into phases. Our experiments result in notable performance enhancements. WizardMath 7B exceeds Gemini's accuracy on English datasets by +6% and matches Gemini's performance on Hindi datasets. Adopting a bilingual approach that combines English and Hindi samples achieves results comparable to individual language models, demonstrating the capability to learn mathematical reasoning in both languages. This research highlights the potential for improving mathematical reasoning in open-source LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言任务方面表现出色，但在数学推理方面存在不足，特别是在非英语语言（如印地语）中更为明显。本研究旨在提升较小且资源高效的开源LLMs在印地语和英语中的数学推理能力。我们使用零样本、少样本以及带有思维链（CoT）的方法和监督微调对OpenHathi 7B、LLaMA-2 7B、WizardMath 7B、Mistral 7B、LLeMMa 7B、MAmmoTH 7B、Gemini Pro和GPT-4等模型进行了评估。我们的方法包括采用递增式学习策略，逐步训练模型解决越来越难的问题；引入一种新颖的分解策略来简化复杂的算术运算；采用结构化的解题设计，将解决方案分为多个阶段。实验结果显示，这些方法显著提升了模型的能力。WizardMath 7B在英语数据集上的准确性超过了Gemini，高出6%，并且在印地语数据集上的表现与Gemini相当。采用双语方法，即结合英语和印地语样本，实现了与单语言模型相当的结果，表明模型能够在两种语言中学习数学推理。本研究强调了提高开源LLMs数学推理能力的潜力。 

---
# A Statistical Framework for Ranking LLM-Based Chatbots 

**Title (ZH)**: 基于统计框架的大型语言模型驱动的聊天机器人排名方法 

**Authors**: Siavash Ameli, Siyuan Zhuang, Ion Stoica, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2412.18407)  

**Abstract**: Large language models (LLMs) have transformed natural language processing, with frameworks like Chatbot Arena providing pioneering platforms for evaluating these models. By facilitating millions of pairwise comparisons based on human judgments, Chatbot Arena has become a cornerstone in LLM evaluation, offering rich datasets for ranking models in open-ended conversational tasks. Building upon this foundation, we propose a statistical framework that incorporates key advancements to address specific challenges in pairwise comparison analysis. First, we introduce a factored tie model that enhances the ability to handle ties -- an integral aspect of human-judged comparisons -- significantly improving the model's fit to observed data. Second, we extend the framework to model covariance between competitors, enabling deeper insights into performance relationships and facilitating intuitive groupings into performance tiers. Third, we resolve optimization challenges arising from parameter non-uniqueness by introducing novel constraints, ensuring stable and interpretable parameter estimation. Through rigorous evaluation and extensive experimentation, our framework demonstrates substantial improvements over existing methods in modeling pairwise comparison data. To support reproducibility and practical adoption, we release leaderbot, an open-source Python package implementing our models and analyses. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经重塑了自然语言处理领域，而像Chatbot Arena这样的框架则提供了评估这些模型的先驱平台。通过基于人类判断进行数百万次成对比较，Chatbot Arena已成为LLM评估的基石，提供了丰富数据集用于在开放对话任务中排名模型。在此基础上，我们提出了一种统计框架，该框架结合了关键技术进步，以解决成对比较分析中的特定挑战。首先，我们引入了一种分因素平局模型，增强了处理平局的能力——这是人类评判比较中的一个关键方面，显著提高了模型对观察数据的拟合度。其次，我们扩展了该框架以建模竞争者之间的协方差，这有助于更深入地了解性能关系，并促进直观的性能层次分组。第三，我们通过引入新型约束条件解决了由参数非唯一性引起的优化挑战，确保了参数估计的稳定性和可解释性。通过严格的评估和广泛的实验，我们的框架在建模成对比较数据方面表现出显著改进。为了支持可重复性和实际应用，我们发布了Leaderbot，这是一个开源的Python包，实现了我们的模型和分析。 

---
# ChaI-TeA: A Benchmark for Evaluating Autocompletion of Interactions with LLM-based Chatbots 

**Title (ZH)**: ChaI-TeA：基于LLM的聊天机器人交互自动补全评估基准 

**Authors**: Shani Goren, Oren Kalinsky, Tomer Stav, Yuri Rapoport, Yaron Fairstein, Ram Yazdy, Nachshon Cohen, Alexander Libov, Guy Kushilevitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.18377)  

**Abstract**: The rise of LLMs has deflected a growing portion of human-computer interactions towards LLM-based chatbots. The remarkable abilities of these models allow users to interact using long, diverse natural language text covering a wide range of topics and styles. Phrasing these messages is a time and effort consuming task, calling for an autocomplete solution to assist users. We introduce the task of chatbot interaction autocomplete. We present ChaI-TeA: CHat InTEraction Autocomplete; An autcomplete evaluation framework for LLM-based chatbot interactions. The framework includes a formal definition of the task, coupled with suitable datasets and metrics. We use the framework to evaluate After formally defining the task along with suitable datasets and metrics, we test 9 models on the defined auto completion task, finding that while current off-the-shelf models perform fairly, there is still much room for improvement, mainly in ranking of the generated suggestions. We provide insights for practitioners working on this task and open new research directions for researchers in the field. We release our framework to serve as a foundation for future research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的兴起已经越来越多地将人类计算机交互转向基于LLM的聊天机器人。这些模型的显著能力使得用户能够使用包含广泛主题和风格的长篇多样自然语言文本进行交互。撰写这些消息是一个耗时耗力的过程，因此需要一个自动补全文本解决方案来辅助用户。我们引入了聊天机器人交互自动补全这一任务，并提出了ChaI-TeA：聊天交互自动补全；一个用于基于LLM的聊天机器人交互的自动补全评估框架。该框架包括对任务的正式定义，以及相应的数据集和评估指标。我们使用此框架来评估九种模型在定义的自动补全任务上的表现，发现尽管现成的模型表现不错，但在生成建议的排名方面仍有很大的改进空间。我们为从事该任务的实践者提供了见解，并为该领域的研究人员开拓了新的研究方向。我们发布了该框架，为未来的研究提供基础。 

---
# Multi-Agents Based on Large Language Models for Knowledge-based Visual Question Answering 

**Title (ZH)**: 基于大型语言模型的多agents知识驱动视觉问答系统 

**Authors**: Zhongjian Hu, Peng Yang, Bing Li, Zhenqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18351)  

**Abstract**: Large Language Models (LLMs) have achieved impressive results in knowledge-based Visual Question Answering (VQA). However existing methods still have challenges: the inability to use external tools autonomously, and the inability to work in teams. Humans tend to know whether they need to use external tools when they encounter a new question, e.g., they tend to be able to give a direct answer to a familiar question, whereas they tend to use tools such as search engines when they encounter an unfamiliar question. In addition, humans also tend to collaborate and discuss with others to get better answers. Inspired by this, we propose the multi-agent voting framework. We design three LLM-based agents that simulate different levels of staff in a team, and assign the available tools according to the levels. Each agent provides the corresponding answer, and finally all the answers provided by the agents are voted to get the final answer. Experiments on OK-VQA and A-OKVQA show that our approach outperforms other baselines by 2.2 and 1.0, respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）在基于知识的视觉问答（VQA）任务上取得了显著成果。然而，现有方法仍然存在一些挑战：无法自主使用外部工具，以及无法协同工作。人类在遇到新的问题时往往能够判断是否需要使用外部工具，例如，他们通常能够直接回答熟悉的问题，而在面对不熟悉的问题时则倾向于使用搜索引擎等工具。此外，人类还倾向于与他人合作和讨论以获得更好的答案。受此启发，我们提出了多智能体投票框架。我们设计了三种基于LLM的智能体，模拟团队中不同级别的员工，并根据级别分配可用工具。每个智能体提供相应的答案，最终通过投票汇总所有智能体提供的答案以得到最终答案。在OK-VQA和A-OKVQA数据集上的实验表明，我们的方法分别比其他基线方法提高了2.2和1.0的性能。 

---
# Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search 

**Title (ZH)**: 桑树：通过集体蒙特卡洛树搜索增强MLLM的o1-like推理与反思能力 

**Authors**: Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18319)  

**Abstract**: In this work, we aim to develop an MLLM that understands and solves questions by learning to create each intermediate step of the reasoning involved till the final answer. To this end, we propose Collective Monte Carlo Tree Search (CoMCTS), a new learning-to-reason method for MLLMs, which introduces the concept of collective learning into ``tree search'' for effective and efficient reasoning-path searching and learning. The core idea of CoMCTS is to leverage collective knowledge from multiple models to collaboratively conjecture, search and identify effective reasoning paths toward correct answers via four iterative operations including Expansion, Simulation and Error Positioning, Backpropagation, and Selection. Using CoMCTS, we construct Mulberry-260k, a multimodal dataset with a tree of rich, explicit and well-defined reasoning nodes for each question. With Mulberry-260k, we perform collective SFT to train our model, Mulberry, a series of MLLMs with o1-like step-by-step Reasoning and Reflection capabilities. Extensive experiments demonstrate the superiority of our proposed methods on various benchmarks. Code will be available at this https URL 

**Abstract (ZH)**: 在本文中，我们旨在开发一种多模态逻辑学习模型（MLLM），该模型通过学习生成推理过程中的每个中间步骤直至最终答案来理解和解决问题。为此，我们提出了集体蒙特卡罗树搜索（CoMCTS），这是一种针对MLLM的新学习推理方法，它将集体学习的概念引入“树搜索”中，以实现有效的推理路径搜索和学习。CoMCTS的核心思想是利用多个模型的集体知识，通过扩展、模拟与错误定位、反向传播和选择四种迭代操作，协同猜测、搜索并识别通向正确答案的有效推理路径。

基于CoMCTS，我们构建了Mulberry-260k数据集，它为每个问题提供了一棵包含丰富、明确定义的推理节点的树。利用Mulberry-260k，我们进行了集体强化学习（SFT）以训练我们的模型Mulberry，该模型是一系列具有逐步推理与反思能力的MLLM。广泛的经验表明，我们的方法在各种基准测试中表现出优越性。代码将在以下链接处提供：[指定链接] 

---
# M-Ped: Multi-Prompt Ensemble Decoding for Large Language Models 

**Title (ZH)**: M-Ped：面向大型语言模型的多提示ensemble解码方法 

**Authors**: Jiaxin Guo, Daimeng Wei, Yuanchang Luo, Shimin Tao, Hengchao Shang, Zongyao Li, Shaojun Li, Jinlong Yang, Zhanglin Wu, Zhiqiang Rao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18299)  

**Abstract**: With the widespread application of Large Language Models (LLMs) in the field of Natural Language Processing (NLP), enhancing their performance has become a research hotspot. This paper presents a novel multi-prompt ensemble decoding approach designed to bolster the generation quality of LLMs by leveraging the aggregation of outcomes from multiple prompts. Given a unique input $X$, we submit $n$ variations of prompts with $X$ to LLMs in batch mode to decode and derive probability distributions. For each token prediction, we calculate the ensemble probability by averaging the $n$ probability distributions within the batch, utilizing this aggregated probability to generate the token. This technique is dubbed Inner-Batch Ensemble. To facilitate efficient batch inference, we implement a Left-Padding strategy to maintain uniform input lengths across the n prompts. Through extensive experimentation on diverse NLP tasks, including machine translation, code generation, and text simplification, we demonstrate the efficacy of our method in enhancing LLM performance. The results show substantial improvements in BLEU scores, pass@$k$ rates, and LENS metrics over conventional methods. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在自然语言处理（NLP）领域的广泛应用，提高其性能已成为研究热点。本文提出了一种新颖的多提示集束解码方法，旨在通过综合多个提示的结果来增强LLMs的生成质量。对于给定的输入 \(X\)，我们提交 \(n\) 个提示 \(X\) 的变体批次给LLMs，以进行解码并获得概率分布。对于每个令牌预测，我们通过计算批次内的 \(n\) 个概率分布的平均值来计算集束概率，并利用该聚合概率生成令牌。这种方法被称为内批次集束。为了实现高效的批次推理，我们实施了左填充策略，以确保批次中 \(n\) 个提示的输入长度一致。通过在包括机器翻译、代码生成和文本简化等不同NLP任务上的广泛实验，我们展示了该方法在增强LLMs性能方面的有效性。实验结果表明，该方法在BLEU分数、pass@k率和LENS指标上相较于传统方法有显著改进。 

---
# DeepCRCEval: Revisiting the Evaluation of Code Review Comment Generation 

**Title (ZH)**: DeepCRCEval: 重新审视代码审查评论生成的评估方法 

**Authors**: Junyi Lu, Xiaojia Li, Zihan Hua, Lei Yu, Shiqi Cheng, Li Yang, Fengjun Zhang, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2412.18291)  

**Abstract**: Code review is a vital but demanding aspect of software development, generating significant interest in automating review comments. Traditional evaluation methods for these comments, primarily based on text similarity, face two major challenges: inconsistent reliability of human-authored comments in open-source projects and the weak correlation of text similarity with objectives like enhancing code quality and detecting defects.
This study empirically analyzes benchmark comments using a novel set of criteria informed by prior research and developer interviews. We then similarly revisit the evaluation of existing methodologies. Our evaluation framework, DeepCRCEval, integrates human evaluators and Large Language Models (LLMs) for a comprehensive reassessment of current techniques based on the criteria set. Besides, we also introduce an innovative and efficient baseline, LLM-Reviewer, leveraging the few-shot learning capabilities of LLMs for a target-oriented comparison.
Our research highlights the limitations of text similarity metrics, finding that less than 10% of benchmark comments are high quality for automation. In contrast, DeepCRCEval effectively distinguishes between high and low-quality comments, proving to be a more reliable evaluation mechanism. Incorporating LLM evaluators into DeepCRCEval significantly boosts efficiency, reducing time and cost by 88.78% and 90.32%, respectively. Furthermore, LLM-Reviewer demonstrates significant potential of focusing task real targets in comment generation. 

**Abstract (ZH)**: 代码审查是软件开发中至关重要但又颇具挑战性的一项工作，已引起了大量关于自动化审查注释的兴趣。传统的注释评估方法主要基于文本相似性，存在两大问题：开源项目中人工编写的注释可靠性不一致以及文本相似性与提升代码质量、检测缺陷等目标之间的弱相关性。

本文通过一个由先前研究和开发者访谈启发的新颖标准，对基准注释进行了实证分析，并重新审视了现有方法的评估。我们构建了一个新的评估框架DeepCRCEval，该框架结合了人工评估者和大型语言模型（LLM），基于设定的标准对现有技术进行了全面的重新评估。此外，我们还引入了一个创新且高效的基准工具LLM-Reviewer，利用LLM的有限提示学习能力进行目标导向的比较。

研究结果揭示了文本相似性指标的局限性，发现少于10%的基准注释适合自动化。相比之下，DeepCRCEval能够有效地区分高质量和低质量的注释，证明了其作为更可靠的评估机制的能力。将LLM评估者纳入DeepCRCEval显著提高了效率，分别将时间和成本降低了88.78%和90.32%。此外，LLM-Reviewer展示了在注释生成任务中聚焦实际目标的巨大潜力。 

---
# An Automatic Graph Construction Framework based on Large Language Models for Recommendation 

**Title (ZH)**: 基于大型语言模型的自动图构建推荐框架 

**Authors**: Rong Shan, Jianghao Lin, Chenxu Zhu, Bo Chen, Menghui Zhu, Kangning Zhang, Jieming Zhu, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18241)  

**Abstract**: Graph neural networks (GNNs) have emerged as state-of-the-art methods to learn from graph-structured data for recommendation. However, most existing GNN-based recommendation methods focus on the optimization of model structures and learning strategies based on pre-defined graphs, neglecting the importance of the graph construction stage. Earlier works for graph construction usually rely on speciffic rules or crowdsourcing, which are either too simplistic or too labor-intensive. Recent works start to utilize large language models (LLMs) to automate the graph construction, in view of their abundant open-world knowledge and remarkable reasoning capabilities. Nevertheless, they generally suffer from two limitations: (1) invisibility of global view (e.g., overlooking contextual information) and (2) construction inefficiency. To this end, we introduce AutoGraph, an automatic graph construction framework based on LLMs for recommendation. Specifically, we first use LLMs to infer the user preference and item knowledge, which is encoded as semantic vectors. Next, we employ vector quantization to extract the latent factors from the semantic vectors. The latent factors are then incorporated as extra nodes to link the user/item nodes, resulting in a graph with in-depth global-view semantics. We further design metapath-based message aggregation to effectively aggregate the semantic and collaborative information. The framework is model-agnostic and compatible with different backbone models. Extensive experiments on three real-world datasets demonstrate the efficacy and efffciency of AutoGraph compared to existing baseline methods. We have deployed AutoGraph in Huawei advertising platform, and gain a 2.69% improvement on RPM and a 7.31% improvement on eCPM in the online A/B test. Currently AutoGraph has been used as the main trafffc model, serving hundreds of millions of people. 

**Abstract (ZH)**: 图神经网络（GNNs）已成为从图结构数据中进行推荐的最先进的方法。然而，现有的大多数基于GNN的推荐方法主要集中在模型结构和基于预定义图的学习策略的优化上，忽略了图构建阶段的重要性。早期的图构建工作通常依赖于特定规则或众包，这些方法要么过于简单，要么过于费时。最近的工作开始利用大语言模型（LLMs）来自动化图构建，利用它们丰富的开放世界知识和卓越的推理能力。尽管如此，它们通常存在两个局限性：（1）全局视图的不可见性（例如，忽视上下文信息）和（2）构建效率低下。为此，我们提出了AutoGraph，这是一种基于LLMs的推荐自动图构建框架。具体来说，我们首先使用LLMs推断用户偏好和项目知识，并将这些知识编码为语义向量。然后，我们使用向量量化来从语义向量中提取潜在因子。接着，将这些潜在因子作为额外节点加入到用户/项目节点中，构建出具有深入全局视图语义的图。我们还设计了基于元路径的消息聚合来有效聚合语义和协同信息。该框架具有模型无关性，并且与不同的骨干模型兼容。在三个真实世界数据集上的广泛实验表明，AutoGraph在与现有基线方法相比在有效性和效率上都表现出色。我们已在华为广告平台部署了AutoGraph，通过在线A/B测试，在RPM上获得了2.69%的提升，在eCPM上获得了7.31%的提升。目前，AutoGraph已成为主要流量模型，服务于数亿用户。 

---
# Molar: Multimodal LLMs with Collaborative Filtering Alignment for Enhanced Sequential Recommendation 

**Title (ZH)**: Molar：基于协作过滤对齐的多模态大语言模型以增强序贯推荐 

**Authors**: Yucong Luo, Qitao Qin, Hao Zhang, Mingyue Cheng, Ruiran Yan, Kefan Wang, Jie Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18176)  

**Abstract**: Sequential recommendation (SR) systems have evolved significantly over the past decade, transitioning from traditional collaborative filtering to deep learning approaches and, more recently, to large language models (LLMs). While the adoption of LLMs has driven substantial advancements, these models inherently lack collaborative filtering information, relying primarily on textual content data neglecting other modalities and thus failing to achieve optimal recommendation performance. To address this limitation, we propose Molar, a Multimodal large language sequential recommendation framework that integrates multiple content modalities with ID information to capture collaborative signals effectively. Molar employs an MLLM to generate unified item representations from both textual and non-textual data, facilitating comprehensive multimodal modeling and enriching item embeddings. Additionally, it incorporates collaborative filtering signals through a post-alignment mechanism, which aligns user representations from content-based and ID-based models, ensuring precise personalization and robust performance. By seamlessly combining multimodal content with collaborative filtering insights, Molar captures both user interests and contextual semantics, leading to superior recommendation accuracy. Extensive experiments validate that Molar significantly outperforms traditional and LLM-based baselines, highlighting its strength in utilizing multimodal data and collaborative signals for sequential recommendation tasks. The source code is available at this https URL. 

**Abstract (ZH)**: 在过去的十年中，序列推荐（SR）系统经历了显著的发展，从传统的协同过滤方法过渡到深度学习方法，并且最近转向了大型语言模型（LLMs）。虽然LLMs的应用带动了重大进步，但这些模型本质上缺乏协同过滤信息，主要依赖文本内容数据而忽视了其他类型的数据，因此未能实现最佳的推荐性能。为了解决这一局限，我们提出了Molar，这是一种多模态大型语言序列推荐框架，能够整合多种内容模态和ID信息，以有效捕获协作信号。Molar利用一个MLLM从文本和非文本数据生成统一的项目表示，促进全面的多模态建模并丰富项目嵌入。此外，它通过后对齐机制将基于内容和基于ID的模型用户表示对齐，以确保精确的个性化和稳健的性能。通过无缝结合多模态内容和协同过滤洞察，Molar能够捕捉用户兴趣和上下文语义，从而获得更优的推荐准确性。大量的实验验证了Molar显著优于传统的和基于LLM的方法，突显了其利用多模态数据和协作信号进行序列推荐任务的优势。源代码可在以下链接获取：this https URL。 

---
# INVESTORBENCH: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent 

**Title (ZH)**: 投资商衡：基于LLM的代理金融决策任务基准测试 

**Authors**: Haohang Li, Yupeng Cao, Yangyang Yu, Shashidhar Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen Jiang, Zining Zhu, Koduvayur Subbalakshmi, Guojun Xiong, Jimin Huang, Lingfei Qian, Xueqing Peng, Qianqian Xie, Jordan W. Suchow  

**Link**: [PDF](https://arxiv.org/pdf/2412.18174)  

**Abstract**: Recent advancements have underscored the potential of large language model (LLM)-based agents in financial decision-making. Despite this progress, the field currently encounters two main challenges: (1) the lack of a comprehensive LLM agent framework adaptable to a variety of financial tasks, and (2) the absence of standardized benchmarks and consistent datasets for assessing agent performance. To tackle these issues, we introduce \textsc{InvestorBench}, the first benchmark specifically designed for evaluating LLM-based agents in diverse financial decision-making contexts. InvestorBench enhances the versatility of LLM-enabled agents by providing a comprehensive suite of tasks applicable to different financial products, including single equities like stocks, cryptocurrencies and exchange-traded funds (ETFs). Additionally, we assess the reasoning and decision-making capabilities of our agent framework using thirteen different LLMs as backbone models, across various market environments and tasks. Furthermore, we have curated a diverse collection of open-source, multi-modal datasets and developed a comprehensive suite of environments for financial decision-making. This establishes a highly accessible platform for evaluating financial agents' performance across various scenarios. 

**Abstract (ZH)**: 近年来，大型语言模型（LLM）代理在财务决策中的潜力得到了强调。尽管取得了这些进展，该领域目前仍面临两个主要挑战：（1）缺乏一个适应各种财务任务的全面LLM代理框架，以及（2）缺乏标准化的基准测试和一致的数据集来评估代理性能。为应对这些问题，我们介绍了InvestorBench，这是第一个专门用于评估在不同财务决策情境下LLM代理的基准测试框架。InvestorBench通过提供适用于不同类型金融产品的综合任务套件，增强了LLM启用代理的灵活性，包括单一股票、加密货币和交易型开放式指数基金（ETFs）等单一证券产品。

此外，我们使用来自不同市场的十三种不同的LLM作为骨干模型，评估我们的代理框架在各种市场环境和任务中的推理和决策能力。我们还精选了多种开源多模态数据集，并开发了一整套用于财务决策的环境，从而建立了一个高度可访问的平台用于评估不同情境下金融代理的性能。 

---
# scReader: Prompting Large Language Models to Interpret scRNA-seq Data 

**Title (ZH)**: scReader：引导大规模语言模型解析单细胞RNA测序数据 

**Authors**: Cong Li, Qingqing Long, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18156)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable advancements, primarily due to their capabilities in modeling the hidden relationships within text sequences. This innovation presents a unique opportunity in the field of life sciences, where vast collections of single-cell omics data from multiple species provide a foundation for training foundational models. However, the challenge lies in the disparity of data scales across different species, hindering the development of a comprehensive model for interpreting genetic data across diverse organisms. In this study, we propose an innovative hybrid approach that integrates the general knowledge capabilities of LLMs with domain-specific representation models for single-cell omics data interpretation. We begin by focusing on genes as the fundamental unit of representation. Gene representations are initialized using functional descriptions, leveraging the strengths of mature language models such as LLaMA-2. By inputting single-cell gene-level expression data with prompts, we effectively model cellular representations based on the differential expression levels of genes across various species and cell types. In the experiments, we constructed developmental cells from humans and mice, specifically targeting cells that are challenging to annotate. We evaluated our methodology through basic tasks such as cell annotation and visualization analysis. The results demonstrate the efficacy of our approach compared to other methods using LLMs, highlighting significant improvements in accuracy and interoperability. Our hybrid approach enhances the representation of single-cell data and offers a robust framework for future research in cross-species genetic analysis. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出显著的进展，主要得益于它们在建模文本序列中隐藏关系方面的能力。这种创新为生命科学领域提供了独特的机会，该领域中来自多种物种的大量单细胞组学数据为训练基础模型奠定了基础。然而，不同物种之间数据规模的差异阻碍了跨多种生物体解析基因数据的全面模型的发展。在本研究中，我们提出了一种创新的混合方法，将LLMs的一般知识能力与其针对单细胞组学数据的特定领域表示模型相结合。我们首先以基因作为表示的基本单元。基因表示通过功能描述初始化，利用成熟语言模型（如LLaMA-2）的优势。通过将单细胞基因水平表达数据与提示输入，我们能够基于不同物种和细胞类型中基因表达差异来有效建模细胞表示。在实验中，我们构建了来自人类和小鼠的发育细胞，特别关注那些难以注释的细胞。我们通过基本任务如细胞注释和可视化分析来评估我们的方法。结果表明，与使用LLMs的其他方法相比，我们方法的有效性和兼容性显著提高。我们的混合方法增强了单细胞数据的表现，并为跨物种遗传分析的未来研究提供了一个稳健的框架。 

---
# GeneSUM: Large Language Model-based Gene Summary Extraction 

**Title (ZH)**: GeneSUM：基于大型语言模型的基因摘要提取 

**Authors**: Zhijian Chen, Chuan Hu, Min Wu, Qingqing Long, Xuezhi Wang, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18154)  

**Abstract**: Emerging topics in biomedical research are continuously expanding, providing a wealth of information about genes and their function. This rapid proliferation of knowledge presents unprecedented opportunities for scientific discovery and formidable challenges for researchers striving to keep abreast of the latest advancements. One significant challenge is navigating the vast corpus of literature to extract vital gene-related information, a time-consuming and cumbersome task. To enhance the efficiency of this process, it is crucial to address several key challenges: (1) the overwhelming volume of literature, (2) the complexity of gene functions, and (3) the automated integration and generation. In response, we propose GeneSUM, a two-stage automated gene summary extractor utilizing a large language model (LLM). Our approach retrieves and eliminates redundancy of target gene literature and then fine-tunes the LLM to refine and streamline the summarization process. We conducted extensive experiments to validate the efficacy of our proposed framework. The results demonstrate that LLM significantly enhances the integration of gene-specific information, allowing more efficient decision-making in ongoing research. 

**Abstract (ZH)**: 生物医学研究中的新兴领域不断扩展，提供了大量关于基因及其功能的信息。这种知识的快速增长为科学发现提供了前所未有的机遇，同时也为研究人员跟上最新进展带来了巨大的挑战。一个重大的挑战是如何在庞大的文献库中导航并提取关键的基因相关信息，这一过程耗时且繁琐。为了提高这一过程的效率，必须应对几个关键挑战：(1) 文献的庞大体量，(2) 基因功能的复杂性，以及(3) 自动化整合和生成。为此，我们提出了一种名为GeneSUM的两阶段自动化基因摘要提取器，利用大型语言模型（LLM）。我们的方法首先检索并消除目标基因文献的冗余，然后通过微调LLM来优化和简化摘要过程。我们进行了广泛实验，以验证我们提出框架的有效性。结果表明，大型语言模型显著增强了基因特异性信息的整合，使得在正在进行的研究中能够更高效地做出决策。 

---
# EvoPat: A Multi-LLM-based Patents Summarization and Analysis Agent 

**Title (ZH)**: EvoPat：一种基于多大型语言模型的专利总结与分析代理 

**Authors**: Suyuan Wang, Xueqian Yin, Menghao Wang, Ruofeng Guo, Kai Nan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18100)  

**Abstract**: The rapid growth of scientific techniques and knowledge is reflected in the exponential increase in new patents filed annually. While these patents drive innovation, they also present significant burden for researchers and engineers, especially newcomers. To avoid the tedious work of navigating a vast and complex landscape to identify trends and breakthroughs, researchers urgently need efficient tools to summarize, evaluate, and contextualize patents, revealing their innovative contributions and underlying scientific this http URL address this need, we present EvoPat, a multi-LLM-based patent agent designed to assist users in analyzing patents through Retrieval-Augmented Generation (RAG) and advanced search strategies. EvoPat leverages multiple Large Language Models (LLMs), each performing specialized roles such as planning, identifying innovations, and conducting comparative evaluations. The system integrates data from local databases, including patents, literature, product catalogous, and company repositories, and online searches to provide up-to-date insights. The ability to collect information not included in original database automatically is also implemented. Through extensive testing in the natural language processing (NLP) domain, we demonstrate that EvoPat outperforms GPT-4 in tasks such as patent summarization, comparative analysis, and technical evaluation. EvoPat represents a significant step toward creating AI-powered tools that empower researchers and engineers to efficiently navigate the complexities of the patent landscape. 

**Abstract (ZH)**: 随着科学技术和知识的迅速发展，每年新提交的专利数量呈指数级增长。虽然这些专利推动了创新，但也给研究人员和工程师，尤其是新入行者带来了巨大负担。为了避免在广阔的复杂领域中寻找趋势和突破时进行繁琐的工作，研究人员迫切需要高效的工具来总结、评估和上下文化专利，揭示其创新贡献和背后的科学原理。为了解决这一需求，我们提出了EvoPat，这是一种基于多大语言模型（LLMs）的专利代理，旨在通过检索增强生成（RAG）和高级搜索策略帮助用户分析专利。EvoPat利用多个大型语言模型，每个模型承担特定的角色，如计划、识别创新和进行比较评估。该系统整合了本地数据库中的数据，包括专利、文献、产品目录和公司仓库，以及在线搜索，以提供最新的洞见。还实现了自动收集未包含在原始数据库中的信息的能力。通过在自然语言处理（NLP）领域的广泛测试，我们证明EvoPat在专利总结、比较分析和技术评估等任务上优于GPT-4。EvoPat代表了朝着创建赋能研究人员和工程师高效导航专利landscape复杂性的AI工具迈出的重要一步。 

---
# Generating Traffic Scenarios via In-Context Learning to Learn Better Motion Planner 

**Title (ZH)**: 通过情境内学习生成交通场景以提高运动规划器性能 

**Authors**: Aizierjiang Aiersilan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18086)  

**Abstract**: Motion planning is a crucial component in autonomous driving. State-of-the-art motion planners are trained on meticulously curated datasets, which are not only expensive to annotate but also insufficient in capturing rarely seen critical scenarios. Failing to account for such scenarios poses a significant risk to motion planners and may lead to incidents during testing. An intuitive solution is to manually compose such scenarios by programming and executing a simulator (e.g., CARLA). However, this approach incurs substantial human costs. Motivated by this, we propose an inexpensive method for generating diverse critical traffic scenarios to train more robust motion planners. First, we represent traffic scenarios as scripts, which are then used by the simulator to generate traffic scenarios. Next, we develop a method that accepts user-specified text descriptions, which a Large Language Model (LLM) translates into scripts using in-context learning. The output scripts are sent to the simulator that produces the corresponding traffic scenarios. As our method can generate abundant safety-critical traffic scenarios, we use them as synthetic training data for motion planners. To demonstrate the value of generated scenarios, we train existing motion planners on our synthetic data, real-world datasets, and a combination of both. Our experiments show that motion planners trained with our data significantly outperform those trained solely on real-world data, showing the usefulness of our synthetic data and the effectiveness of our data generation method. Our source code is available at this https URL. 

**Abstract (ZH)**: 自主驾驶中的运动规划是其关键组成部分。当前最先进的运动规划算法是在精心标注的数据集上进行训练的，这些数据集不仅标注成本高昂，而且难以覆盖罕见但至关重要的场景。忽略这些场景会显著增加运动规划的风险，并可能导致测试过程中出现事故。一种直观的解决方案是通过编程和执行模拟器（例如CARLA）手动组成这些场景。然而，这种方法将付出高昂的人力成本。受此启发，我们提出了一种低成本的方法来生成多样化的关键交通场景，以训练更 robust 的运动规划器。首先，我们将交通场景转化为脚本，这些脚本随后被模拟器用来生成交通场景。接着，我们开发了一种方法，该方法接受用户指定的文字描述，然后通过上下文学习过程，由大规模语言模型（LLM）将其转化为符合要求的脚本。生成的脚本被发送给模拟器，以生产相应的交通场景。由于我们的方法能够生成大量的安全关键交通场景，我们使用这些场景作为运动规划器的合成训练数据。为了证明生成场景的价值，我们在合成数据、真实世界数据以及两者的组合上分别训练现有的运动规划器。实验结果表明，使用我们数据训练的运动规划器明显优于仅使用真实世界数据训练的运动规划器，这表明了我们合成数据的有效性以及数据生成方法的效果。我们的源代码可在此获取：[提供链接]。 

---
# More than Chit-Chat: Developing Robots for Small-Talk Interactions 

**Title (ZH)**: 不仅仅是闲聊：开发用于小型对话的机器人 

**Authors**: Rebecca Ramnauth, Dražen Brščić, Brian Scassellati  

**Link**: [PDF](https://arxiv.org/pdf/2412.18023)  

**Abstract**: Beyond mere formality, small talk plays a pivotal role in social dynamics, serving as a verbal handshake for building rapport and understanding. For conversational AI and social robots, the ability to engage in small talk enhances their perceived sociability, leading to more comfortable and natural user interactions. In this study, we evaluate the capacity of current Large Language Models (LLMs) to drive the small talk of a social robot and identify key areas for improvement. We introduce a novel method that autonomously generates feedback and ensures LLM-generated responses align with small talk conventions. Through several evaluations -- involving chatbot interactions and human-robot interactions -- we demonstrate the system's effectiveness in guiding LLM-generated responses toward realistic, human-like, and natural small-talk exchanges. 

**Abstract (ZH)**: 超越 mere formality，闲聊在社会动态中扮演着至关重要的角色，它作为一种口头“握手”手段，有助于建立关系和增进理解。对于对话式人工智能和社交机器人而言，具备进行闲聊的能力能够提升其社会性，从而使用户交互更加舒适和自然。在这项研究中，我们评估了当前大型语言模型（LLMs）驱动社交机器人闲聊的能力，并确定了改进的关键领域。我们提出了一种新颖的方法，可以自主生成反馈并确保大型语言模型生成的回应符合闲聊的惯例。通过几次评估——包括聊天机器人交互和人机交互——我们展示了该系统引导大型语言模型生成的回应向现实、人性化的自然闲聊交流方向发展的效果。 

---
# LMV-RPA: Large Model Voting-based Robotic Process Automation 

**Title (ZH)**: LMV-RPA：基于大型模型投票的机器人流程自动化

这个标题翻译成中文时，保持了原有的学术规范和专业术语。"Large Model Voting" 被翻译为“大型模型投票”，"Robotic Process Automation" 翻译为“机器人流程自动化”。确保了翻译的准确性和专业性。 

**Authors**: Osama Abdellatif, Ahmed Ayman, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2412.17965)  

**Abstract**: Automating high-volume unstructured data processing is essential for operational efficiency. Optical Character Recognition (OCR) is critical but often struggles with accuracy and efficiency in complex layouts and ambiguous text. These challenges are especially pronounced in large-scale tasks requiring both speed and precision. This paper introduces LMV-RPA, a Large Model Voting-based Robotic Process Automation system to enhance OCR workflows. LMV-RPA integrates outputs from OCR engines such as Paddle OCR, Tesseract OCR, Easy OCR, and DocTR with Large Language Models (LLMs) like LLaMA 3 and Gemini-1.5-pro. Using a majority voting mechanism, it processes OCR outputs into structured JSON formats, improving accuracy, particularly in complex layouts. The multi-phase pipeline processes text extracted by OCR engines through LLMs, combining results to ensure the most accurate outputs. LMV-RPA achieves 99 percent accuracy in OCR tasks, surpassing baseline models with 94 percent, while reducing processing time by 80 percent. Benchmark evaluations confirm its scalability and demonstrate that LMV-RPA offers a faster, more reliable, and efficient solution for automating large-scale document processing tasks. 

**Abstract (ZH)**: 大规模处理高 Volume 的非结构化数据对于提高运营效率至关重要。光学字符识别（OCR）至关重要，但在复杂布局和模糊文本的情况下经常面临准确性和效率的问题。这些挑战在需要同时速度和精度的大规模任务中尤为突出。本文介绍了一种基于大型模型投票的机器人流程自动化系统 LMV-RPA，以增强 OCR 工作流程。LMV-RPA 将来自 Paddle OCR、Tesseract OCR、Easy OCR 和 DocTR 等 OCR 引擎的输出与大型语言模型（LLMs）如 LLaMA 3 和 Gemini-1.5-pro 整合在一起。利用多数投票机制，它将 OCR 输出转化为结构化的 JSON 格式，特别是在复杂布局上提升了准确性。多阶段流水线对 OCR 引擎提取的文本通过 LLMs 处理，并结合结果确保最准确的输出。LMV-RPA 在 OCR 任务中的准确率达到 99%，比基线模型（准确率 94%）高，同时将处理时间减少了 80%。基准评估证实了其可扩展性，并表明 LMV-RPA 提供了一种更快、更可靠且更高效的大型文档处理自动化解决方案。 

---
# Evaluating LLM Reasoning in the Operations Research Domain with ORQA 

**Title (ZH)**: 使用ORQA评估大型语言模型在运筹学领域的推理能力 

**Authors**: Mahdi Mostajabdaveh, Timothy T. Yu, Samarendra Chandan Bindu Dash, Rindranirina Ramamonjison, Jabo Serge Byusa, Giuseppe Carenini, Zirui Zhou, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17874)  

**Abstract**: In this paper, we introduce and apply Operations Research Question Answering (ORQA), a new benchmark designed to assess the generalization capabilities of Large Language Models (LLMs) in the specialized technical domain of Operations Research (OR). This benchmark evaluates whether LLMs can emulate the knowledge and reasoning skills of OR experts when confronted with diverse and complex optimization problems. The dataset, developed by OR experts, features real-world optimization problems that demand multistep reasoning to construct their mathematical models. Our evaluations of various open source LLMs, such as LLaMA 3.1, DeepSeek, and Mixtral, reveal their modest performance, highlighting a gap in their ability to generalize to specialized technical domains. This work contributes to the ongoing discourse on LLMs generalization capabilities, offering valuable insights for future research in this area. The dataset and evaluation code are publicly available. 

**Abstract (ZH)**: 在本文中，我们介绍了并应用了运筹学问答（ORQA）基准，这是一个新的基准，旨在评估大规模语言模型（LLMs）在运筹学（OR）这一专门技术领域的泛化能力。该基准评估LLMs在面对多种多样且复杂的优化问题时，是否能够模拟运筹学专家的知识和推理能力。该数据集由运筹学专家开发，包含需要多步推理来构建其数学模型的实际优化问题。我们对多种开源LLM，如LLaMA 3.1、DeepSeek和Mixtral的评估显示，它们的性能较为有限，突显了它们在向专门技术领域泛化方面的不足。本文为持续进行的关于LLMs泛化能力的讨论做出了贡献，并为该领域的未来研究提供了宝贵见解。该数据集及其评价代码已对外公开。 

---
# Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types 

**Title (ZH)**: 评估和增强多轮文本到SQL转换的LLM模型，支持多种问题类型 

**Authors**: Ziming Guo, Chao Ma, Yinggang Sun, Tiancheng Zhao, Guangyao Wang, Hai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17867)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly advanced text-to-SQL systems. However, most LLM-based methods often narrowly focus on SQL generation, neglecting the complexities of real-world conversational queries. This oversight can lead to unreliable responses, particularly for ambiguous questions that cannot be directly addressed with SQL. To bridge this gap, we propose MMSQL, a comprehensive test suite designed to evaluate the question classification and SQL generation capabilities of LLMs by simulating real-world scenarios with diverse question types and multi-turn Q\&A interactions. Using MMSQL, we assessed the performance of popular LLMs, including both open-source and closed-source models, and identified key factors impacting their performance in such scenarios. Moreover, we introduce an LLM-based multi-agent framework that employs specialized agents to identify question types and determine appropriate answering strategies. Our experiments demonstrate that this approach significantly enhances the model's ability to navigate the complexities of conversational dynamics, effectively handling the diverse and complex nature of user queries. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新进展显著提升了文本到SQL系统的能力。然而，大多数基于LLM的方法往往狭隘地集中在SQL生成上，忽视了现实世界对话查询的复杂性。这种忽视可能导致不稳定的响应，尤其是对于那些不能直接用SQL解决的模糊问题。为了弥合这一差距，我们提出了MMSQL，这是一个全面的测试套件，通过模拟各种问题类型和多轮对话交互的现实场景，评估LLMs的问题分类和SQL生成能力。使用MMSQL，我们评估了包括开源和闭源在内的流行LLM模型的性能，并确定了影响它们在这类场景中表现的关键因素。此外，我们介绍了一种基于LLM的多智能体框架，该框架采用专门的智能体来识别问题类型并确定合适的回答策略。我们的实验表明，这种做法显著增强了模型导航对话动态复杂性的能力，有效地应对了用户的多样性和复杂查询。 

---
# The Rosetta Paradox: Domain-Specific Performance Inversions in Large Language Models 

**Title (ZH)**: 罗赛塔悖论：大型语言模型在专业领域中的性能倒置 

**Authors**: Basab Jha, Ujjwal Puri  

**Link**: [PDF](https://arxiv.org/pdf/2412.17821)  

**Abstract**: While large language models, such as GPT and BERT, have already demonstrated unprecedented skills in everything from natural language processing to domain-specific applications, there came an unexplored phenomenon we term the Rosetta Paradox. The Rosetta Paradox characterizes the counterintuitive performance inversions across domains of knowledge. This paradox captures how such LLMs can excel in highly specialized fields but do poorly on tasks which require general, everyday knowledge. This paper formalizes the definition of the Rosetta Paradox and introduces a panoramic analysis framework that includes both a Domain Specificity Index (DSI) and a Performance Inversion Metric (PIM) for consistent quantification of domain-specific behavior in LLMs.
We adopt this paradox and conduct a series of investigations through extensive experiments across diverse models and knowledge domains, ranging from rich technical areas to common-sense reasoning. Our findings indicate that the Rosetta Paradox is likely not a mere artifact of data distribution but an intrinsic architectural and emergent property of deep neural networks. We present comparative analyses across different model architectures, sizes, and training methodologies that shed light into the peculiar ways this paradox manifests itself and challenge the standard evaluation metrics. 

**Abstract (ZH)**: 尽管诸如GPT和BERT这样的大规模语言模型已经在自然语言处理以及特定领域的应用中展示了前所未有的能力，但仍出现了一种未被探索的现象，我们称之为罗塞塔悖论。罗塞塔悖论描述了知识领域间不直观的性能反转现象。这一悖论反映了这些LLM在专门领域的卓越表现，但在需要一般性和日常生活知识的任务上却表现不佳。本文正式定义了罗塞塔悖论，并引入了一个全景分析框架，包括领域特异性指数（Domain Specificity Index，DSI）和性能反转指标（Performance Inversion Metric，PIM），以一致地量化LLM在特定领域的行为。

我们采用了这一悖论，并通过广泛的实验对不同模型和知识领域进行了一系列调查，范围从丰富的技术领域到常识推理。我们的研究结果表明，罗塞塔悖论很可能不仅是一种数据分布的产物，而是深层神经网络的内在架构和涌现属性。我们对不同模型架构、规模和训练方法进行了比较分析，揭示了这一悖论特有的表现方式，并挑战了现有的评价指标。 

---
# Inductive Linguistic Reasoning with Large Language Models 

**Title (ZH)**: 大型语言模型中的归纳语义推理 

**Authors**: Raghav Ramji, Keshav Ramji  

**Link**: [PDF](https://arxiv.org/pdf/2412.17819)  

**Abstract**: Evaluating large language models (LLMs) on their linguistic reasoning capabilities is an important task to understand the gaps in their skills that may surface during large-scale adoption. In this work, we investigate the abilities of such models to perform abstract multilingual reasoning through the lens of linguistic puzzles on extremely low-resource languages. As these translation tasks involve inductive and deductive reasoning from reference instances, we examine whether diverse auxiliary demonstrations can be automatically induced from seed exemplars, through analogical prompting. We employ a two-stage procedure, first generating analogical exemplars with a language model, and then applying them in-context along with provided target language exemplars. Our results on the modeLing dataset show that analogical prompting is effective in eliciting models' knowledge of language grammar similarities, boosting the performance of GPT-4o by as much as 8.1% and Llama-3.1-405B-Instruct by 5.9% over chain-of-thought approaches. These gains are attributable to the analogical demonstrations, both when self-generated as well as when produced by weaker multilingual models. Furthermore, we demonstrate that our method generalizes to other tasks present in Linguistics Olympiad competitions, achieving sizable improvements across all problem types and difficulty levels included in the LINGOLY dataset with GPT-4o. We also report several findings about interesting phenomena which drive linguistic reasoning performance, suggesting that such puzzles are a valuable benchmark for new reasoning methods. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）在语言推理能力上的表现是理解其技能差距的重要任务，特别是在大规模应用时可能会浮现这些差距。在本文中，我们通过语言谜题的视角研究这些模型在极度低资源语言上的抽象多语言推理能力。由于这些翻译任务涉及从参照实例进行归纳和演绎推理，我们考察了是否可以通过类比提示自动生成多样化的辅助示例，并从种子示例中进行推导。我们采用两阶段的方法，首先使用语言模型生成类比示例，然后在提供目标语言示例的情况下将其应用于上下文。我们在modeLing数据集上的结果显示，类比提示在激发模型对语言语法规则相似性的了解方面是有效的，相较于链式思维方法，GPT-4o和Llama-3.1-405B-Instruct的性能分别提高了8.1%和5.9%。这些提升可以归因于无论是自动生成还是由较弱的多语言模型生成的类比示例。此外，我们展示了我们的方法可以推广到 Linguistics Olympiad 竞赛中的其他任务，在LINGOLY数据集中，所有问题类型和难度级别的表现均有所显著改善。我们还报告了一些有趣的发现，这些发现揭示了影响语言推理性能的现象，表明这些谜题对于新推理方法的基准具有重要价值。 

---
# Leveraging Memory Retrieval to Enhance LLM-based Generative Recommendation 

**Title (ZH)**: 利用记忆检索增强基于大规模语言模型的生成性推荐 

**Authors**: Chengbing Wang, Yang Zhang, Fengbin Zhu, Jizhi Zhang, Tianhao Shi, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17593)  

**Abstract**: Leveraging Large Language Models (LLMs) to harness user-item interaction histories for item generation has emerged as a promising paradigm in generative recommendation. However, the limited context window of LLMs often restricts them to focusing on recent user interactions only, leading to the neglect of long-term interests involved in the longer histories. To address this challenge, we propose a novel Automatic Memory-Retrieval framework (AutoMR), which is capable of storing long-term interests in the memory and extracting relevant information from it for next-item generation within LLMs. Extensive experimental results on two real-world datasets demonstrate the effectiveness of our proposed AutoMR framework in utilizing long-term interests for generative recommendation. 

**Abstract (ZH)**: 利用大型语言模型（LLMs）挖掘用户-项目互动历史以生成项目，在生成型推荐领域被认为是一种有前途的范式。然而，LLMs 较小的上下文窗口限制了它们仅关注最近的用户互动，而忽视了较长历史中涉及的长期兴趣。为了解决这一挑战，我们提出了一种新的自动记忆检索框架（AutoMR），该框架能在记忆中存储长期兴趣，并从记忆中提取相关信息用于后续项目的生成。在两个实际数据集上的广泛实验结果证明了我们所提出的 AutoMR 框架在利用长期兴趣进行生成型推荐方面的有效性。 

---
# SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval 

**Title (ZH)**: SyNeg: 由大模型驱动的合成否定样本用于密集检索 

**Authors**: Xiaopeng Li, Xiangyang Li, Hao Zhang, Zhaocheng Du, Pengyue Jia, Yichao Wang, Xiangyu Zhao, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17250)  

**Abstract**: The performance of Dense retrieval (DR) is significantly influenced by the quality of negative sampling. Traditional DR methods primarily depend on naive negative sampling techniques or on mining hard negatives through external retriever and meticulously crafted strategies. However, naive negative sampling often fails to adequately capture the accurate boundaries between positive and negative samples, whereas existing hard negative sampling methods are prone to false negatives, resulting in performance degradation and training instability. Recent advancements in large language models (LLMs) offer an innovative solution to these challenges by generating contextually rich and diverse negative samples. In this work, we present a framework that harnesses LLMs to synthesize high-quality hard negative samples. We first devise a \textit{multi-attribute self-reflection prompting strategy} to direct LLMs in hard negative sample generation. Then, we implement a \textit{hybrid sampling strategy} that integrates these synthetic negatives with traditionally retrieved negatives, thereby stabilizing the training process and improving retrieval performance. Extensive experiments on five benchmark datasets demonstrate the efficacy of our approach, and code is also publicly available. 

**Abstract (ZH)**: 密集检索(DR)的性能显著受到负样本质量的影响。传统DR方法主要依赖于简单的负样本采样技术或通过外部检索器和精心设计的方法挖掘难以负样本。然而，简单的负样本采样常常无法充分捕捉正负样本之间的准确界限，而现有的难以负样本采样方法则容易产生假阴性，导致性能下降和训练不稳定。近年来，大型语言模型(LLMs)的进步为解决这些问题提供了新的解决方案，通过生成丰富且多样化的负样本。在本工作中，我们提出了一种利用LLMs生成高质量难以负样本的框架。我们首先设计了一种**多属性自我反思提示策略**，以指导LLMs进行难以负样本的生成。然后，我们实施了一种**混合采样策略**，将这些合成的负样本与传统检索到的负样本结合，从而稳定训练过程并提高检索性能。在五个基准数据集上的大量实验中证明了我们方法的有效性，并且相关代码也已公开发布。 

---
# LLM-based relevance assessment still can't replace human relevance assessment 

**Title (ZH)**: 基于LLM的相关性评估仍无法替代人工相关性评估 

**Authors**: Charles L. A. Clarke, Laura Dietz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17156)  

**Abstract**: The use of large language models (LLMs) for relevance assessment in information retrieval has gained significant attention, with recent studies suggesting that LLM-based judgments provide comparable evaluations to human judgments. Notably, based on TREC 2024 data, Upadhyay et al. make a bold claim that LLM-based relevance assessments, such as those generated by the UMBRELA system, can fully replace traditional human relevance assessments in TREC-style evaluations. This paper critically examines this claim, highlighting practical and theoretical limitations that undermine the validity of this conclusion. First, we question whether the evidence provided by Upadhyay et al. really supports their claim, particularly if a test collection is used asa benchmark for future improvements. Second, through a submission deliberately intended to do so, we demonstrate the ease with which automatic evaluation metrics can be subverted, showing that systems designed to exploit these evaluations can achieve artificially high scores. Theoretical challenges -- such as the inherent narcissism of LLMs, the risk of overfitting to LLM-based metrics, and the potential degradation of future LLM performance -- must be addressed before LLM-based relevance assessments can be considered a viable replacement for human judgments. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在信息检索中的相关性评估应用已经引起广泛关注，最近的研究表明，基于LLM的判断可以提供与人类判断相媲美的评估结果。基于TREC 2024数据，Upadhyay等人提出了一个大胆的论点，即基于LLM的相关性评估，如UMBRELA系统生成的评估，完全可以在TREC风格的评估中取代传统的人类相关性评估。本文对此论点进行了批判性评估，指出了实际和理论上的局限性，这些局限性削弱了这一结论的有效性。首先，我们质疑Upadhyay等人提供的证据是否真正支持他们的论点，尤其是在未来的改进中使用测试集合作为基准的情况下。其次，通过一个特意设计用于此目的的提交，我们证明了自动评估指标可以被轻易破坏，展示了系统如何通过利用这些评估实现虚假高的评分。在大规模语言模型（LLMs）相关性评估被认为是一个可行的人类判断替代方案之前，必须解决诸如LLMs的固有自我中心倾向、过度拟合LLM指标的风险以及未来LLM性能可能降级等理论挑战。 

---
# LLM-Powered User Simulator for Recommender System 

**Title (ZH)**: LLM驱动的用户模拟器在推荐系统中的应用 

**Authors**: Zijian Zhang, Shuchang Liu, Ziru Liu, Rui Zhong, Qingpeng Cai, Xiangyu Zhao, Chunxu Zhang, Qidong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16984)  

**Abstract**: User simulators can rapidly generate a large volume of timely user behavior data, providing a testing platform for reinforcement learning-based recommender systems, thus accelerating their iteration and optimization. However, prevalent user simulators generally suffer from significant limitations, including the opacity of user preference modeling and the incapability of evaluating simulation accuracy. In this paper, we introduce an LLM-powered user simulator to simulate user engagement with items in an explicit manner, thereby enhancing the efficiency and effectiveness of reinforcement learning-based recommender systems training. Specifically, we identify the explicit logic of user preferences, leverage LLMs to analyze item characteristics and distill user sentiments, and design a logical model to imitate real human engagement. By integrating a statistical model, we further enhance the reliability of the simulation, proposing an ensemble model that synergizes logical and statistical insights for user interaction simulations. Capitalizing on the extensive knowledge and semantic generation capabilities of LLMs, our user simulator faithfully emulates user behaviors and preferences, yielding high-fidelity training data that enrich the training of recommendation algorithms. We establish quantifying and qualifying experiments on five datasets to validate the simulator's effectiveness and stability across various recommendation scenarios. 

**Abstract (ZH)**: 用户模拟器可以快速生成大量及时的用户行为数据，为基于强化学习的推荐系统提供测试平台，从而加速其迭代和优化。然而，目前广泛使用的用户模拟器通常存在显著的局限性，包括用户偏好建模的不透明性和评估模拟准确性的能力不足。在本文中，我们介绍了一个以LLM（大型语言模型）为驱动的用户模拟器，以显式方式模拟用户与项目之间的互动，从而提高基于强化学习的推荐系统训练的效率和有效性。具体来说，我们识别了用户偏好的显式逻辑，利用LLM分析项目特征和萃取用户情感，并设计了一个逻辑模型来模仿真实的人类互动。通过集成统计模型，我们进一步增强了模拟的可靠性，提出了一种结合逻辑和统计洞察的集成模型，用于用户互动模拟。借助LLM广泛的知识和语义生成能力，我们的用户模拟器能够忠实模拟用户行为和偏好，生成高保真度的训练数据，丰富推荐算法的训练。我们在五个数据集上建立了量化和定性的实验，以验证模拟器在各种推荐场景中的有效性和稳定性。 

---
