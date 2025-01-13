# LLM-MedQA: Enhancing Medical Question Answering through Case Studies in Large Language Models 

**Title (ZH)**: LLM-MedQA：通过大型语言模型案例研究增强医学问答能力 

**Authors**: Hang Yang, Hao Chen, Hui Guo, Yineng Chen, Ching-Sheng Lin, Shu Hu, Jinrong Hu, Xi Wu, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05464)  

**Abstract**: Accurate and efficient question-answering systems are essential for delivering high-quality patient care in the medical field. While Large Language Models (LLMs) have made remarkable strides across various domains, they continue to face significant challenges in medical question answering, particularly in understanding domain-specific terminologies and performing complex reasoning. These limitations undermine their effectiveness in critical medical applications. To address these issues, we propose a novel approach incorporating similar case generation within a multi-agent medical question-answering (MedQA) system. Specifically, we leverage the Llama3.1:70B model, a state-of-the-art LLM, in a multi-agent architecture to enhance performance on the MedQA dataset using zero-shot learning. Our method capitalizes on the model's inherent medical knowledge and reasoning capabilities, eliminating the need for additional training data. Experimental results show substantial performance gains over existing benchmark models, with improvements of 7% in both accuracy and F1-score across various medical QA tasks. Furthermore, we examine the model's interpretability and reliability in addressing complex medical queries. This research not only offers a robust solution for medical question answering but also establishes a foundation for broader applications of LLMs in the medical domain. 

**Abstract (ZH)**: 准确且高效的问答系统对于提供高质量的医疗服务至关重要。尽管大型语言模型（LLMs）已经在各个领域取得了显著进展，但在医疗领域的问答任务中，它们仍然面临重大挑战，特别是在理解和处理领域特定术语以及进行复杂推理方面。这些限制削弱了它们在关键医疗应用中的效果。为了解决这些问题，我们提出了一种新颖的方法，即在多智能体医疗问答（MedQA）系统中引入相似病例生成技术。具体而言，我们利用最先进的大型语言模型Llama3.1:70B，在多智能体架构中进行零样本学习，以增强MedQA数据集上的性能。我们的方法充分利用了模型内置的医学知识和推理能力，无需额外的训练数据。实验结果显示，与现有基准模型相比，我们的方法在各种医学问答任务中实现了显著的性能提升，准确率和F1分数分别提高了7%。此外，我们还考察了该模型在处理复杂医学查询时的可解释性和可靠性。这项研究不仅提供了一种稳健的解决方案以应对医学问答任务，还为我们探索大型语言模型在医疗领域的更广泛应用奠定了基础。 

---
# Strategy Masking: A Method for Guardrails in Value-based Reinforcement Learning Agents 

**Title (ZH)**: 价值导向强化学习代理中的策略屏蔽：一种护栏方法 

**Authors**: Jonathan Keane, Sam Keyser, Jeremy Kedziora  

**Link**: [PDF](https://arxiv.org/pdf/2501.05501)  

**Abstract**: The use of reward functions to structure AI learning and decision making is core to the current reinforcement learning paradigm; however, without careful design of reward functions, agents can learn to solve problems in ways that may be considered ``undesirable" or ``unethical. Without thorough understanding of the incentives a reward function creates, it can be difficult to impose principled yet general control mechanisms over its behavior. In this paper, we study methods for constructing guardrails for AI agents that use reward functions to learn decision making. We introduce a novel approach, which we call strategy masking, to explicitly learn and then suppress undesirable AI agent behavior. We apply our method to study lying in AI agents and show that strategy masking can effectively modify agent behavior by suppressing, or actively penalizing, the reward dimension for lying such that agents act more honestly while not compromising their ability to perform effectively. 

**Abstract (ZH)**: 使用奖励函数来构建AI的学习和决策过程是目前强化学习范式的核心；然而，如果没有精心设计奖励函数，智能体可能会以被认为“不合适”或“不道德”的方式学习解决问题。在不了解奖励函数所创造的激励机制的情况下，很难对其行为施加既原则性又通用的控制机制。在本文中，我们研究了为使用奖励函数学习决策的AI智能体构建“防护栏”的方法。我们介绍了一种新颖的方法，称之为策略遮蔽，这种方法旨在显式地学习并抑制不 desirable 的AI智能体行为。我们应用该方法研究了AI智能体中的谎言行为，并展示了策略遮蔽可以通过抑制或主动惩罚说谎的奖励维度，有效地修改智能体的行为，使其更加诚实，同时不牺牲其有效的执行能力。 

---
# Contextual ASR Error Handling with LLMs Augmentation for Goal-Oriented Conversational AI 

**Title (ZH)**: 基于LLM增强的上下文ASR错误处理方法在目标导向对话AI中的应用 

**Authors**: Yuya Asano, Sabit Hassan, Paras Sharma, Anthony Sicilia, Katherine Atwell, Diane Litman, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2501.06129)  

**Abstract**: General-purpose automatic speech recognition (ASR) systems do not always perform well in goal-oriented dialogue. Existing ASR correction methods rely on prior user data or named entities. We extend correction to tasks that have no prior user data and exhibit linguistic flexibility such as lexical and syntactic variations. We propose a novel context augmentation with a large language model and a ranking strategy that incorporates contextual information from the dialogue states of a goal-oriented conversational AI and its tasks. Our method ranks (1) n-best ASR hypotheses by their lexical and semantic similarity with context and (2) context by phonetic correspondence with ASR hypotheses. Evaluated in home improvement and cooking domains with real-world users, our method improves recall and F1 of correction by 34% and 16%, respectively, while maintaining precision and false positive rate. Users rated .8-1 point (out of 5) higher when our correction method worked properly, with no decrease due to false positives. 

**Abstract (ZH)**: 通用的自动语音识别（ASR）系统在目标导向对话中并不总是表现良好。现有的ASR校正方法依赖于先验用户数据或命名实体。我们扩展了校正范围，使其适用于没有先验用户数据且具有语言灵活性的任务，例如词汇和句法变体。我们提出了一种新的上下文增强方法，结合了大规模语言模型和一种综合目标导向对话AI及其任务的对话状态上下文信息的排序策略。我们的方法通过（1）根据与上下文的词汇和语义相似性来排名ASR的n-best假设；以及（2）通过与ASR假设的音节对应关系来排名上下文，来进行校正。我们在这项研究中在家居改进和烹饪领域进行了评估，结果显示，在真实用户参与的情况下，我们的方法在召回率和F1值上分别提高了34%和16%，同时保持了精确率和假阳性率。当我们的校正方法正常工作时，用户评分平均提高了0.8至1分（满分5分），并且没有因假阳性因素而导致评分下降。 

---
# Towards Developing Socially Compliant Automated Vehicles: State of the Art, Experts Expectations, and A Conceptual Framework 

**Title (ZH)**: 面向开发社会合规的自动驾驶车辆：现状、专家期望与概念框架 

**Authors**: Yongqi Dong, Bart van Arem, Haneen Farah  

**Link**: [PDF](https://arxiv.org/pdf/2501.06089)  

**Abstract**: Automated Vehicles (AVs) hold promise for revolutionizing transportation by improving road safety, traffic efficiency, and overall mobility. Despite the steady advancement in high-level AVs in recent years, the transition to full automation entails a period of mixed traffic, where AVs of varying automation levels coexist with human-driven vehicles (HDVs). Making AVs socially compliant and understood by human drivers is expected to improve the safety and efficiency of mixed traffic. Thus, ensuring AVs compatibility with HDVs and social acceptance is crucial for their successful and seamless integration into mixed traffic. However, research in this critical area of developing Socially Compliant AVs (SCAVs) remains sparse. This study carries out the first comprehensive scoping review to assess the current state of the art in developing SCAVs, identifying key concepts, methodological approaches, and research gaps. An expert interview was also conducted to identify critical research gaps and expectations towards SCAVs. Based on the scoping review and expert interview input, a conceptual framework is proposed for the development of SCAVs. The conceptual framework is evaluated using an online survey targeting researchers, technicians, policymakers, and other relevant professionals worldwide. The survey results provide valuable validation and insights, affirming the significance of the proposed conceptual framework in tackling the challenges of integrating AVs into mixed-traffic environments. Additionally, future research perspectives and suggestions are discussed, contributing to the research and development agenda of SCAVs. 

**Abstract (ZH)**: 自动驾驶车辆（AVs）有望通过提升道路安全、交通效率和整体移动性来改革交通运输。尽管近年来高级别AVs的技术不断进步，从部分自动化过渡到全面自动化需要一个混合交通的过渡期，在此期间不同自动化级别的AVs将与人类驾驶车辆（HDVs）共存。使AVs具备社会合规性并被人类驾驶员理解，有望提高混合交通的安全性和效率。因此，确保AVs与HDVs的兼容性和社会接受度对于其成功且无缝地集成到混合交通中至关重要。然而，这一关键领域的研究仍然相对缺乏。本研究首次进行全面的范围审查，评估当前在开发社会合规性自动驾驶车辆（SCAVs）方面的状态，识别核心概念、方法论方法和研究缺口。同时也进行了专家访谈，以识别关键的研究缺口和对SCAVs的期望。基于范围审查和专家访谈的输入，提出了一个概念框架来指导SCAVs的发展。该概念框架通过面向全球研究人员、技术人员、政策制定者和其他相关专业人士的在线调查进行了评估。调查结果提供了宝贵的验证和洞察，证实了所提出的概念框架在应对AVs集成到混合交通环境中的挑战方面的重要性。此外，还讨论了未来的研究视角和建议，为SCAVs的研究和发展议程做出了贡献。 

---
# Diffusion Models for Smarter UAVs: Decision-Making and Modeling 

**Title (ZH)**: 智能化无人机的扩散模型：决策与建模 

**Authors**: Yousef Emami, Hao Zhou, Luis Almeida, Kai Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.05819)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly adopted in modern communication networks. However, challenges in decision-making and digital modeling continue to impede their rapid advancement. Reinforcement Learning (RL) algorithms face limitations such as low sample efficiency and limited data versatility, further magnified in UAV communication scenarios. Moreover, Digital Twin (DT) modeling introduces substantial decision-making and data management complexities. RL models, often integrated into DT frameworks, require extensive training data to achieve accurate predictions. In contrast to traditional approaches that focus on class boundaries, Diffusion Models (DMs), a new class of generative AI, learn the underlying probability distribution from the training data and can generate trustworthy new patterns based on this learned distribution. This paper explores the integration of DMs with RL and DT to effectively address these challenges. By combining the data generation capabilities of DMs with the decision-making framework of RL and the modeling accuracy of DT, the integration improves the adaptability and real-time performance of UAV communication. Moreover, the study shows how DMs can alleviate data scarcity, improve policy networks, and optimize dynamic modeling, providing a robust solution for complex UAV communication scenarios. 

**Abstract (ZH)**: 无人机（UAVs）在现代通信网络中越来越受到重视。然而，决策和数字建模中的挑战依然阻碍了其快速进步。强化学习（Reinforcement Learning, RL）算法存在样本效率低和数据灵活性有限的问题，这些问题在无人机通信场景中被进一步放大。此外，数字孪生（Digital Twin, DT）建模引入了显著的决策和数据管理复杂性。RL 模型通常集成到 DT 框架中，需要大量的训练数据才能实现准确的预测。与传统方法关注类边界不同，生成型人工智能中的扩散模型（Diffusion Models, DMs）能够从训练数据中学习潜在的概率分布，并根据这个学习到的分布生成可信的新模式。本文探讨了将 DMs 与 RL 和 DT 相结合以有效应对这些挑战。通过结合 DMs 的数据生成能力、RL 的决策框架以及 DT 的建模精度，这种集成提高了无人机通信的适应性和实时性能。此外，研究表明 DMs 可以缓解数据稀缺性、改善策略网络并优化动态建模，为复杂的无人机通信场景提供了一个稳健的解决方案。 

---
# How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond 

**Title (ZH)**: 如何实现人类与自然语言处理模型之间有效的协作：原则、形式化方法及更多方面的综述 

**Authors**: Chen Huang, Yang Deng, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua, Jimmy Xiangji Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05714)  

**Abstract**: With the advancement of large language models (LLMs), intelligent models have evolved from mere tools to autonomous agents with their own goals and strategies for cooperating with humans. This evolution has birthed a novel paradigm in NLP, i.e., human-model cooperation, that has yielded remarkable progress in numerous NLP tasks in recent years. In this paper, we take the first step to present a thorough review of human-model cooperation, exploring its principles, formalizations, and open challenges. In particular, we introduce a new taxonomy that provides a unified perspective to summarize existing approaches. Also, we discuss potential frontier areas and their corresponding challenges. We regard our work as an entry point, paving the way for more breakthrough research in this regard. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的发展，智能模型已从单纯的工具演变为具有自身目标和与人类合作策略的自主代理。这一演变催生了一种新的自然语言处理（NLP）范式，即人类-模型合作，近年来在众多NLP任务中取得了显著进展。在本文中，我们首次对该领域进行了全面回顾，探讨了其原则、形式化方法以及面临的关键挑战。特别地，我们引入了一个新的分类体系，为总结现有方法提供统一视角。此外，我们讨论了潜在的研究前沿领域及其相应的挑战。我们认为，我们的工作是这一领域的入门点，为后续更突破性的研究铺平了道路。 

---
# Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains 

**Title (ZH)**: 多智能体微调：通过多样化推理链实现自我提升 

**Authors**: Vighnesh Subramaniam, Yilun Du, Joshua B. Tenenbaum, Antonio Torralba, Shuang Li, Igor Mordatch  

**Link**: [PDF](https://arxiv.org/pdf/2501.05707)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance in recent years but are fundamentally limited by the underlying training data. To improve models beyond the training data, recent works have explored how LLMs can be used to generate synthetic data for autonomous self-improvement. However, successive steps of self-improvement can reach a point of diminishing returns. In this work, we propose a complementary approach towards self-improvement where finetuning is applied to a multiagent society of language models. A group of language models, all starting from the same base model, are independently specialized by updating each one using data generated through multiagent interactions among the models. By training each model on independent sets of data, we illustrate how this approach enables specialization across models and diversification over the set of models. As a result, our overall system is able to preserve diverse reasoning chains and autonomously improve over many more rounds of fine-tuning than single-agent self-improvement methods. We quantitatively illustrate the efficacy of the approach across a wide suite of reasoning tasks. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在多个任务中取得了显著的性能，但其根本上仍受限于训练数据。为了超越训练数据的限制，近期的研究探索了如何利用LLMs生成合成数据以促进自主自适应。然而，连续的自适应步骤可能会达到边际收益递减的阶段。在本文中，我们提出了一种互补的自适应方法，在该方法中，对多智能体社会中的语言模型进行微调。一群从同一基础模型出发的语言模型通过模型间的多智能体交互生成数据，各自独立地进行专业化训练。通过让每个模型在独立的数据集上进行训练，本文展示了这种方法如何在模型之间实现专业化，并在模型集合中实现多样化。因此，我们的整体系统能够在多轮次的微调中实现更自主的进步，并且比单智能体自适应方法能够进行更多轮次的优化。我们通过广泛的心理任务验证了该方法的有效性。 

---
# RTLSquad: Multi-Agent Based Interpretable RTL Design 

**Title (ZH)**: RTLSquad: 基于多代理的可解释RTL设计 

**Authors**: Bowei Wang, Qi Xiong, Zeqing Xiang, Lei Wang, Renzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.05470)  

**Abstract**: Optimizing Register-Transfer Level (RTL) code is crucial for improving hardware PPA performance. Large Language Models (LLMs) offer new approaches for automatic RTL code generation and optimization. However, existing methods often lack decision interpretability (sufficient, understandable justification for decisions), making it difficult for hardware engineers to trust the generated results, thus preventing these methods from being integrated into the design process. To address this, we propose RTLSquad, a novel LLM-Based Multi-Agent system for interpretable RTL code generation. RTLSquad divides the design process into exploration, implementation, and verification & evaluation stages managed by specialized agent squads, generating optimized RTL code through inter-agent collaboration, and providing decision interpretability through the communication process. Experiments show that RTLSquad excels in generating functionally correct RTL code and optimizing PPA performance, while also having the capability to provide decision paths, demonstrating the practical value of our system. 

**Abstract (ZH)**: 优化寄存器传输级（RTL）代码对于提高硬件性能参数（PPA）至关重要。大型语言模型（LLMs）为自动RTL代码生成和优化提供了新的方法。然而，现有方法通常缺乏决策解释性（即，充分且可理解的决策依据），这使得硬件工程师难以信任生成的结果，从而导致这些方法难以集成到设计过程中。为解决这一问题，我们提出了一种名为RTLSquad的新型基于LLM的多智能体系统，用于可解释的RTL代码生成。RTLSquad将设计过程分为探索、实现、验证与评估阶段，并由专门的智能体战队管理这些阶段，通过智能体间的协作生成优化的RTL代码，并通过通信过程提供决策解释性。实验结果表明，RTLSquad在生成功能正确的RTL代码和优化PPA性能方面表现出色，同时还能提供决策路径，验证了我们系统的实际应用价值。 

---
# Language and Planning in Robotic Navigation: A Multilingual Evaluation of State-of-the-Art Models 

**Title (ZH)**: 机器人导航中的语言与规划：对最先进的模型进行多语言评估 

**Authors**: Malak Mansour, Ahmed Aly, Bahey Tharwat, Sarim Hashmi, Dong An, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2501.05478)  

**Abstract**: Large Language Models (LLMs) such as GPT-4, trained on huge amount of datasets spanning multiple domains, exhibit significant reasoning, understanding, and planning capabilities across various tasks. This study presents the first-ever work in Arabic language integration within the Vision-and-Language Navigation (VLN) domain in robotics, an area that has been notably underexplored in existing research. We perform a comprehensive evaluation of state-of-the-art multi-lingual Small Language Models (SLMs), including GPT-4o mini, Llama 3 8B, and Phi-3 medium 14B, alongside the Arabic-centric LLM, Jais. Our approach utilizes the NavGPT framework, a pure LLM-based instruction-following navigation agent, to assess the impact of language on navigation reasoning through zero-shot sequential action prediction using the R2R dataset. Through comprehensive experiments, we demonstrate that our framework is capable of high-level planning for navigation tasks when provided with instructions in both English and Arabic. However, certain models struggled with reasoning and planning in the Arabic language due to inherent limitations in their capabilities, sub-optimal performance, and parsing issues. These findings highlight the importance of enhancing planning and reasoning capabilities in language models for effective navigation, emphasizing this as a key area for further development while also unlocking the potential of Arabic-language models for impactful real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT-4，在跨多个领域的大量数据集上进行训练，展现出了在各种任务中显著的推理、理解和计划能力。本研究首次将阿拉伯语融入机器人学中的视觉-语言导航（VLN）领域，这一领域在现有研究中鲜有探讨。我们对最先进的多语言小语言模型（SLMs），包括GPT-4o mini、Llama 3 8B和Phi-3中型14B，以及阿拉伯语中心的LLM Jais进行了全面评估。我们的方法利用了NavGPT框架，这是一种完全基于LLM的指令跟随导航代理，通过零样本连续动作预测实验来评估语言对导航推理的影响，测试使用了R2R数据集。通过全面的实验，我们证明了当提供英文和阿拉伯文指令时，我们的框架能够完成高层次的导航任务计划。然而，某些模型在阿拉伯语推理和计划方面遇到了困难，这主要是由于这些模型固有的局限性、性能不佳和解析问题所导致。这些发现强调了增强语言模型的计划和推理能力对于实现有效导航的重要性，指出这是一个需要进一步发展的关键领域，同时也展现了阿拉伯语模型在现实世界应用中的潜在影响力。 

---
# LatteReview: A Multi-Agent Framework for Systematic Review Automation Using Large Language Models 

**Title (ZH)**: LatteReview：使用大规模语言模型进行系统评价自动化的一种多代理框架 

**Authors**: Pouria Rouzrokh, Moein Shariatnia  

**Link**: [PDF](https://arxiv.org/pdf/2501.05468)  

**Abstract**: Systematic literature reviews and meta-analyses are essential for synthesizing research insights, but they remain time-intensive and labor-intensive due to the iterative processes of screening, evaluation, and data extraction. This paper introduces and evaluates LatteReview, a Python-based framework that leverages large language models (LLMs) and multi-agent systems to automate key elements of the systematic review process. Designed to streamline workflows while maintaining rigor, LatteReview utilizes modular agents for tasks such as title and abstract screening, relevance scoring, and structured data extraction. These agents operate within orchestrated workflows, supporting sequential and parallel review rounds, dynamic decision-making, and iterative refinement based on user feedback. LatteReview's architecture integrates LLM providers, enabling compatibility with both cloud-based and locally hosted models. The framework supports features such as Retrieval-Augmented Generation (RAG) for incorporating external context, multimodal reviews, Pydantic-based validation for structured inputs and outputs, and asynchronous programming for handling large-scale datasets. The framework is available on the GitHub repository, with detailed documentation and an installable package. 

**Abstract (ZH)**: 系统文献综述和元分析是综合研究洞察力的重要工具，但由于筛选、评估和数据提取的迭代过程，它们仍耗时且劳动密集。本文介绍了并评估了LatteReview，这是一个基于Python的框架，利用大型语言模型（LLMs）和多智能体系统自动化系统综述过程中的关键要素。设计时旨在简化工作流程同时保持严格性，LatteReview使用模块化的代理程序来执行诸如标题和摘要筛选、相关性评分和结构化数据提取等任务。这些代理程序在协调的流程中运作，支持顺序和并行的审查轮次、动态决策和基于用户反馈的迭代改进。LatteReview的架构整合了LLM提供者，使得既兼容基于云的模型也兼容本地托管的模型。该框架支持诸如检索增强生成（RAG）、多模态审查、基于Pydantic的验证以确保输入和输出的结构化，以及异步编程以处理大规模数据集等功能。该框架在GitHub存储库中可用，配备了详细的文档和可安装的包。 

---
