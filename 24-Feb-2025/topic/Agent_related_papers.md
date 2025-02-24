# SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instrucion Following Evaluation for Social Agents 

**Title (ZH)**: SOTOPIA-Ω：社交代理的动态策略注入学习与社会指令跟随评估 

**Authors**: Wenyuan Zhang, Tianyun Liu, Mengxiao Song, Xiaodong Li, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15538)  

**Abstract**: Despite the abundance of prior social strategies possessed by humans, there remains a paucity of research dedicated to their transfer and integration into social agents. Our proposed SOTOPIA-{\Omega} framework aims to address and bridge this gap, with a particular focus on enhancing the social capabilities of language agents. This framework dynamically injects multi-step reasoning strategies inspired by negotiation theory, along with two simple direct strategies, into expert agents, thereby automating the construction of high-quality social dialogue training corpus. Additionally, we introduce the concept of Social Instruction Following (S-IF) and propose two new S-IF evaluation metrics that are complementary to social capability. We demonstrate that several 7B models trained on high-quality corpus not only significantly surpass the expert agent (GPT-4) in achieving social goals but also enhance S-IF performance. Analysis and variant experiments validate the advantages of dynamic construction, which can especially break the agent's prolonged deadlock. 

**Abstract (ZH)**: 尽管人类拥有丰富的社会策略，但关于这些策略的转移和集成到社会代理中的研究仍然相对匮乏。我们提出的SOTOPIA-Ω框架旨在填补这一空白，特别关注提升语言代理的社会能力。该框架通过动态注入受谈判理论启发的多步推理策略以及两个简单的直接策略，自动构建高质量的社会对话训练语料库。此外，我们引入了社会指令跟随（S-IF）的概念，并提出了两种互补于社会能力的新S-IF评估指标。研究表明，多个训练于高质量语料库的7B模型不仅在实现社会目标方面显著超过了专家代理（GPT-4），而且还提升了S-IF性能。分析和变体实验验证了动态构建的优势，特别是在打破代理长时间僵局方面尤为显著。 

---
# ESPnet-SpeechLM: An Open Speech Language Model Toolkit 

**Title (ZH)**: ESPnet-SpeechLM：一个开源语音语言模型工具包 

**Authors**: Jinchuan Tian, Jiatong Shi, William Chen, Siddhant Arora, Yoshiki Masuyama, Takashi Maekaku, Yihan Wu, Junyi Peng, Shikhar Bharadwaj, Yiwen Zhao, Samuele Cornell, Yifan Peng, Xiang Yue, Chao-Han Huck Yang, Graham Neubig, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2502.15218)  

**Abstract**: We present ESPnet-SpeechLM, an open toolkit designed to democratize the development of speech language models (SpeechLMs) and voice-driven agentic applications. The toolkit standardizes speech processing tasks by framing them as universal sequential modeling problems, encompassing a cohesive workflow of data preprocessing, pre-training, inference, and task evaluation. With ESPnet-SpeechLM, users can easily define task templates and configure key settings, enabling seamless and streamlined SpeechLM development. The toolkit ensures flexibility, efficiency, and scalability by offering highly configurable modules for every stage of the workflow. To illustrate its capabilities, we provide multiple use cases demonstrating how competitive SpeechLMs can be constructed with ESPnet-SpeechLM, including a 1.7B-parameter model pre-trained on both text and speech tasks, across diverse benchmarks. The toolkit and its recipes are fully transparent and reproducible at: this https URL. 

**Abstract (ZH)**: 我们介绍了ESPnet-SpeechLM，这是一个开源工具包，旨在普及语音语言模型（SpeechLM）和语音驱动的自主应用的开发。该工具包通过将语音处理任务框定为统一的序列建模问题来标准化语音处理任务，涵盖了从数据预处理、预训练、推理到任务评估的统一工作流程。借助ESPnet-SpeechLM，用户可以轻松定义任务模板并配置关键设置，从而实现无缝且高效的语音语言模型开发。该工具包通过提供高度可配置的模块，确保在工作流程的每个阶段都具有灵活性、高效性和可扩展性。为了展示其功能，我们提供了多个应用场景，展示了如何使用ESPnet-SpeechLM构建竞争力的语音语言模型，包括一个在文本和语音任务上预训练的参数量为17亿的模型，并在多种基准测试中进行了评估。该工具包及其配方完全透明且可重现，详情请访问：this https URL。 

---
# Investigating the Adaptive Robustness with Knowledge Conflicts in LLM-based Multi-Agent Systems 

**Title (ZH)**: 基于知识冲突的LLM驱动多agent系统中的自适应稳健性研究 

**Authors**: Tianjie Ju, Bowen Wang, Hao Fei, Mong-Li Lee, Wynne Hsu, Yun Li, Qianren Wang, Pengzhou Cheng, Zongru Wu, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15153)  

**Abstract**: Recent advances in Large Language Models (LLMs) have upgraded them from sophisticated text generators to autonomous agents capable of corporation and tool use in multi-agent systems (MASs). However, the robustness of these LLM-based MASs, especially under knowledge conflicts, remains unclear. In this paper, we design four comprehensive metrics to investigate the robustness of MASs when facing mild or task-critical knowledge conflicts. We first analyze mild knowledge conflicts introduced by heterogeneous agents and find that they do not harm system robustness but instead improve collaborative decision-making. Next, we investigate task-critical knowledge conflicts by synthesizing knowledge conflicts and embedding them into one of the agents. Our results show that these conflicts have surprisingly little to no impact on MAS robustness. Furthermore, we observe that MASs demonstrate certain self-repairing capabilities by reducing their reliance on knowledge conflicts and adopting alternative solution paths to maintain stability. Finally, we conduct ablation studies on the knowledge conflict number, agent number, and interaction rounds, finding that the self-repairing capability of MASs has intrinsic limits, and all findings hold consistently across various factors. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的进步已经将它们从复杂的文本生成器升级为能够在多代理系统（MASs）中进行合作和工具使用的自主代理。然而，这些基于LLM的MASs在知识冲突下的鲁棒性仍然存在不确定性，特别是在知识冲突严重的情况下。本文设计了四种综合指标，以探究MASs在面对轻度或任务关键的知识冲突时的鲁棒性。我们首先分析由异构代理引入的轻度知识冲突，发现这些冲突并不会损害系统鲁棒性，反而能改善协作决策。接着，我们通过合成知识冲突并将其嵌入到一个代理中，研究任务关键的知识冲突。我们的结果表明，这些冲突对MAS鲁棒性的影响非常有限甚至几乎不存在。此外，我们观察到，MASs自身具有一定的自我修复能力，通过减少对知识冲突的依赖并采用替代的解决方案路径来维持系统的稳定性。最后，我们在知识冲突数量、代理数量和交互轮次等方面进行了消融研究，发现MASs的自我修复能力具有固有的限制，并且所有研究结果在不同因素下均保持一致。我们的代码已公开发布，地址为：this https URL。 

---
# Rare Disease Differential Diagnosis with Large Language Models at Scale: From Abdominal Actinomycosis to Wilson's Disease 

**Title (ZH)**: 大规模语言模型在罕见病鉴别诊断中的应用：从腹腔类巴克特里亚小菌病到威尔森病 

**Authors**: Elliot Schumacher, Dhruv Naik, Anitha Kannan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15069)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in disease diagnosis. However, their effectiveness in identifying rarer diseases, which are inherently more challenging to diagnose, remains an open question. Rare disease performance is critical with the increasing use of LLMs in healthcare settings. This is especially true if a primary care physician needs to make a rarer prognosis from only a patient conversation so that they can take the appropriate next step. To that end, several clinical decision support systems are designed to support providers in rare disease identification. Yet their utility is limited due to their lack of knowledge of common disorders and difficulty of use.
In this paper, we propose RareScale to combine the knowledge LLMs with expert systems. We use jointly use an expert system and LLM to simulate rare disease chats. This data is used to train a rare disease candidate predictor model. Candidates from this smaller model are then used as additional inputs to black-box LLM to make the final differential diagnosis. Thus, RareScale allows for a balance between rare and common diagnoses. We present results on over 575 rare diseases, beginning with Abdominal Actinomycosis and ending with Wilson's Disease. Our approach significantly improves the baseline performance of black-box LLMs by over 17% in Top-5 accuracy. We also find that our candidate generation performance is high (e.g. 88.8% on gpt-4o generated chats). 

**Abstract (ZH)**: 大型语言模型（LLMs）在疾病诊断方面展现出了令人印象深刻的能力。然而，它们在识别稀有疾病方面——这些疾病本就更难诊断——的有效性仍然是一个开放的问题。随着LLMs在医疗领域的广泛应用，稀有疾病的性能变得尤为重要。尤其是在初级保健医师仅凭与患者对话就需要作出稀有诊断时，这种重要性更为突出。为此，设计了一些临床决策支持系统来辅助识别罕见疾病，但它们的实用性有限，因为它们缺乏对常见疾病的了解，且使用困难。

本文中，我们提出了一种名为RareScale的方法，旨在结合LLMs的知识和专家系统。我们使用专家系统和LLMs共同模拟稀有疾病的对话，这些数据用于训练一个稀有疾病候选预测模型。从该小型模型中得到的候选者随后作为额外输入用于黑盒LLMs，以做出最终的鉴别诊断。因此，RareScale能够在稀有和常见疾病的诊断之间取得平衡。我们在超过575种稀有疾病（从腹型放线菌病到威尔逊病）上进行了实验，结果显示，与基线性能相比，我们的方法在Top-5准确率方面提高了超过17%。此外，我们还发现，候选生成性能较高（例如，对由gpt-4生成的对话，准确率为88.8%）。 

---
# A Socratic RAG Approach to Connect Natural Language Queries on Research Topics with Knowledge Organization Systems 

**Title (ZH)**: 一种苏格拉底式检索辅助方法，用于连接研究主题的自然语言查询与知识组织系统 

**Authors**: Lew Lefton, Kexin Rong, Chinar Dankhara, Lila Ghemri, Firdous Kausar, A. Hannibal Hamdallahi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15005)  

**Abstract**: In this paper, we propose a Retrieval Augmented Generation (RAG) agent that maps natural language queries about research topics to precise, machine-interpretable semantic entities. Our approach combines RAG with Socratic dialogue to align a user's intuitive understanding of research topics with established Knowledge Organization Systems (KOSs). The proposed approach will effectively bridge "little semantics" (domain-specific KOS structures) with "big semantics" (broad bibliometric repositories), making complex academic taxonomies more accessible. Such agents have the potential for broad use. We illustrate with a sample application called CollabNext, which is a person-centric knowledge graph connecting people, organizations, and research topics. We further describe how the application design has an intentional focus on HBCUs and emerging researchers to raise visibility of people historically rendered invisible in the current science system. 

**Abstract (ZH)**: 在本文中，我们提出一种检索增强生成（RAG）代理，能够将关于研究主题的自然语言查询映射到精确且机器可解释的语义实体。我们的方法结合了RAG和苏格拉底式对话，从而将用户对研究主题的直观理解与已建立的知识组织系统（KOS）相一致。所提出的方法将有效地将“小语义”（领域特定的KOS结构）与“大语义”（广泛的文献计量repository）相连接，使得复杂的学术分类更加易于访问。此类代理有广泛的应用潜力。我们通过一个名为CollabNext的示例应用进行了说明，该应用以个人为中心的知识图谱将个人、组织和研究主题连接起来。进一步阐述了该应用设计的初衷，旨在关注HBCU（哈莱姆宝物大学）和新兴研究者，以提高历史上在当前科学系统中被忽视的人群的能见度。 

---
# ARS: Automatic Routing Solver with Large Language Models 

**Title (ZH)**: ARS：基于大型语言模型的自动路由求解器 

**Authors**: Kai Li, Fei Liu, Zhenkun Wang, Xialiang Tong, Xiongwei Han, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15359)  

**Abstract**: Real-world Vehicle Routing Problems (VRPs) are characterized by a variety of practical constraints, making manual solver design both knowledge-intensive and time-consuming. Although there is increasing interest in automating the design of routing algorithms, existing research has explored only a limited array of VRP variants and fails to adequately address the complex and prevalent constraints encountered in real-world situations. To fill this gap, this paper introduces RoutBench, a benchmark of 1,000 VRP variants derived from 24 attributes, for evaluating the effectiveness of automatic routing solvers in addressing complex constraints. Along with RoutBench, we present the Automatic Routing Solver (ARS), which employs Large Language Model (LLM) agents to enhance a backbone algorithm framework by automatically generating constraint-aware heuristic code, based on problem descriptions and several representative constraints selected from a database. Our experiments show that ARS outperforms state-of-the-art LLM-based methods and commonly used solvers, automatically solving 91.67% of common VRPs and achieving at least a 30% improvement across all benchmarks. 

**Abstract (ZH)**: 现实世界中的车辆路线问题（VRPs）受多种实用约束的影响，使得手动设计求解器既需要深厚的知识支持，也需要耗费大量时间。尽管有越来越多的研究兴趣集中在自动化设计路由算法上，但现有研究仅探索了有限的VRP变体，并未能充分应对现实世界中复杂且普遍存在的约束。为弥补这一不足，本文引入了一个基于24个属性生成的1000个VRP变体的基准——RoutBench，用于评估自动化路由求解器在应对复杂约束方面的有效性。与此同时，我们还提出了自动路由求解器（ARS），该求解器使用大型语言模型（LLM）代理来增强基础算法框架，通过根据问题描述和从数据库中选择的若干代表性约束，自动生成具有约束意识的启发式代码。实验结果表明，ARS 在有效解决常见VRPs 和所有基准中均表现出色，自动解决了91.67%的常见VRP问题，并实现了至少30%的整体性能提升。 

---
# The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning 

**Title (ZH)**: LLM-和VLM-集成强化学习的发展 landscape 

**Authors**: Sheila Schoepp, Masoud Jafaripour, Yingyue Cao, Tianpei Yang, Fatemeh Abdollahi, Shadan Golestan, Zahin Sufiyan, Osmar R. Zaiane, Matthew E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.15214)  

**Abstract**: Reinforcement learning (RL) has shown impressive results in sequential decision-making tasks. Meanwhile, Large Language Models (LLMs) and Vision-Language Models (VLMs) have emerged, exhibiting impressive capabilities in multimodal understanding and reasoning. These advances have led to a surge of research integrating LLMs and VLMs into RL. In this survey, we review representative works in which LLMs and VLMs are used to overcome key challenges in RL, such as lack of prior knowledge, long-horizon planning, and reward design. We present a taxonomy that categorizes these LLM/VLM-assisted RL approaches into three roles: agent, planner, and reward. We conclude by exploring open problems, including grounding, bias mitigation, improved representations, and action advice. By consolidating existing research and identifying future directions, this survey establishes a framework for integrating LLMs and VLMs into RL, advancing approaches that unify natural language and visual understanding with sequential decision-making. 

**Abstract (ZH)**: 强化学习（RL）在序贯决策任务中展现了令人印象深刻的成果。与此同时，大型语言模型（LLMs）和多模态视觉语言模型（VLMs）已经涌现，并展示了在多模态理解和推理方面的 impressive 能力。这些进展催生了将 LLMs 和 VLMs 结合进 RL 的大量研究。在本文综述中，我们回顾了 LLMs 和 VLMs 在克服 RL 中关键挑战时的应用，如缺乏先验知识、长期规划以及奖励设计。我们提出了一种分类法，将这些由 LLMs/VLMs 支撑的 RL 方法归类为三个角色：代理、规划器和奖励。最后，我们探讨了开放性问题，包括语义关联、偏见 mitigation、更好的表示形式以及行动建议。通过总结现有研究并确定未来的研究方向，本文综述建立了一种框架，用于将 LLMs 和 VLMs 整合进 RL，推动了自然语言和视觉理解与序贯决策统一的方法的发展。 

---
# AutoToM: Automated Bayesian Inverse Planning and Model Discovery for Open-ended Theory of Mind 

**Title (ZH)**: AutoToM：自动贝叶斯反规划与模型发现算法在开放性心智理论中的应用 

**Authors**: Zhining Zhang, Chuanyang Jin, Mung Yao Jia, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15676)  

**Abstract**: Theory of Mind (ToM), the ability to understand people's mental variables based on their behavior, is key to developing socially intelligent agents. Current approaches to Theory of Mind reasoning either rely on prompting Large Language Models (LLMs), which are prone to systematic errors, or use rigid, handcrafted Bayesian Theory of Mind (BToM) models, which are more robust but cannot generalize across different domains. In this work, we introduce AutoToM, an automated Bayesian Theory of Mind method for achieving open-ended machine Theory of Mind. AutoToM can operate in any domain, infer any mental variable, and conduct robust Theory of Mind reasoning of any order. Given a Theory of Mind inference problem, AutoToM first proposes an initial BToM model. It then conducts automated Bayesian inverse planning based on the proposed model, leveraging an LLM as the backend. Based on the uncertainty of the inference, it iteratively refines the model, by introducing additional mental variables and/or incorporating more timesteps in the context. Empirical evaluations across multiple Theory of Mind benchmarks demonstrate that AutoToM consistently achieves state-of-the-art performance, offering a scalable, robust, and interpretable approach to machine Theory of Mind. 

**Abstract (ZH)**: 理论心理（ToM），即根据人们的行為理解其心理变量的能力，是开发社会智能代理的关键。当前的ToM推理方法要么依赖于大型语言模型（LLMs）提示，这容易产生系统性错误，要么使用结构僵硬的手动构建的贝叶斯理论心理（BToM）模型，这些模型更加稳健但不能在不同领域泛化。在此项工作中，我们介绍了一种自动化贝叶斯理论心理（AutoToM）方法，以实现开放式的机器理论心理。AutoToM可以在任何领域运行，能够推断任何心理变量，并执行任意级别的ToM推理。给定一个ToM推理问题，AutoToM首先提出一个初始的BToM模型。然后，基于提出的模型，它进行自动化贝叶斯逆规划，利用LLM作为后端。基于推理的不确定性，它通过引入额外的心理变量和/或增加上下文中的时间步数来迭代细化模型。在多个ToM基准测试中的实证评估表明，AutoToM始终能够达到最先进的性能，提供了一种可扩展、稳健且可解释的机器理论心理方法。 

---
# Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path? 

**Title (ZH)**: 超智能代理带来灾难性风险：科学家型AI能提供一條更安全的道路吗？ 

**Authors**: Yoshua Bengio, Michael Cohen, Damiano Fornasiere, Joumana Ghosn, Pietro Greiner, Matt MacDermott, Sören Mindermann, Adam Oberman, Jesse Richardson, Oliver Richardson, Marc-Antoine Rondeau, Pierre-Luc St-Charles, David Williams-King  

**Link**: [PDF](https://arxiv.org/pdf/2502.15657)  

**Abstract**: The leading AI companies are increasingly focused on building generalist AI agents -- systems that can autonomously plan, act, and pursue goals across almost all tasks that humans can perform. Despite how useful these systems might be, unchecked AI agency poses significant risks to public safety and security, ranging from misuse by malicious actors to a potentially irreversible loss of human control. We discuss how these risks arise from current AI training methods. Indeed, various scenarios and experiments have demonstrated the possibility of AI agents engaging in deception or pursuing goals that were not specified by human operators and that conflict with human interests, such as self-preservation. Following the precautionary principle, we see a strong need for safer, yet still useful, alternatives to the current agency-driven trajectory. Accordingly, we propose as a core building block for further advances the development of a non-agentic AI system that is trustworthy and safe by design, which we call Scientist AI. This system is designed to explain the world from observations, as opposed to taking actions in it to imitate or please humans. It comprises a world model that generates theories to explain data and a question-answering inference machine. Both components operate with an explicit notion of uncertainty to mitigate the risks of overconfident predictions. In light of these considerations, a Scientist AI could be used to assist human researchers in accelerating scientific progress, including in AI safety. In particular, our system can be employed as a guardrail against AI agents that might be created despite the risks involved. Ultimately, focusing on non-agentic AI may enable the benefits of AI innovation while avoiding the risks associated with the current trajectory. We hope these arguments will motivate researchers, developers, and policymakers to favor this safer path. 

**Abstract (ZH)**: 领先的人工智能公司越来越关注于构建通用人工智能代理系统——这些系统能够自主规划、行动，并追求几乎涵盖人类所有任务的目标。尽管这些系统可能非常有用，但未受约束的人工智能代理可能会对公共安全和安全构成重大风险，从恶意行为者的误用到人类控制不可逆转的丧失。我们讨论了当前人工智能训练方法如何导致这些风险。事实上，各种情境和实验已经表明，人工智能代理有可能采取欺骗行为或追求由人类操作员未指定并可能与人类利益相冲突的目标，例如自我保护。根据预防原则，我们需要寻求更为安全但仍实用的替代方案。因此，我们建议作为进一步发展核心构建模块的是一种信任和安全设计的非代理型人工智能系统，我们称之为科学家型AI。该系统旨在通过观察解释世界，而不是采取行动模仿或取悦人类。该系统包括一个世界模型，用于生成解释数据的理论，以及一个问答推理机器。这两部分都具有明确的不确定性概念，以减轻过度自信预测的风险。考虑到这些考量，科学家型AI可以用于帮助人类研究人员加速科学发展，包括人工智能安全性。特别地，该系统可以作为预防风险的措施，防止尽管存在潜在风险仍可能出现的危险人工智能代理的出现。最终，专注于非代理型人工智能可能会使人工智能创新的好处最大化，同时避免当前轨迹相关风险。我们希望这些论点能够激励研究人员、开发者和政策制定者倾向于这一更为安全的路径。 

---
# TAG: A Decentralized Framework for Multi-Agent Hierarchical Reinforcement Learning 

**Title (ZH)**: TAG：多代理层次强化学习的去中心化框架 

**Authors**: Giuseppe Paolo, Abdelhakim Benechehab, Hamza Cherkaoui, Albert Thomas, Balázs Kégl  

**Link**: [PDF](https://arxiv.org/pdf/2502.15425)  

**Abstract**: Hierarchical organization is fundamental to biological systems and human societies, yet artificial intelligence systems often rely on monolithic architectures that limit adaptability and scalability. Current hierarchical reinforcement learning (HRL) approaches typically restrict hierarchies to two levels or require centralized training, which limits their practical applicability. We introduce TAME Agent Framework (TAG), a framework for constructing fully decentralized hierarchical multi-agent this http URL enables hierarchies of arbitrary depth through a novel LevelEnv concept, which abstracts each hierarchy level as the environment for the agents above it. This approach standardizes information flow between levels while preserving loose coupling, allowing for seamless integration of diverse agent types. We demonstrate the effectiveness of TAG by implementing hierarchical architectures that combine different RL agents across multiple levels, achieving improved performance over classical multi-agent RL baselines on standard benchmarks. Our results show that decentralized hierarchical organization enhances both learning speed and final performance, positioning TAG as a promising direction for scalable multi-agent systems. 

**Abstract (ZH)**: 层次结构是生物系统和人类社会的基本特征，而人工智能系统通常依赖于单一架构，限制了其适应性和扩展性。当前的层次化强化学习（HRL）方法通常限制层次结构为两级或者需要集中训练，这限制了它们的实际应用。我们引入了TAME智能体框架（TAG），该框架用于构建完全去中心化的层次化多智能体系统。通过引入新的LevelEnv概念，TAG能够通过抽象每个层次作为智能体之上层次的环境来实现任意深度的层次结构。这种方法标准化了不同层次之间的信息流动，同时保持了松耦合，从而实现了不同类型的智能体无缝集成。我们通过在多个层次上结合不同的RL智能体来实现层次化架构，并在标准基准上展示了TAG的有效性，它实现了优于经典多智能体RL基线的性能。结果表明，去中心化的层次化组织能够提高学习速度和最终性能，将TAG定位为可扩展多智能体系统的一个有前途的方向。 

---
# Measuring AI agent autonomy: Towards a scalable approach with code inspection 

**Title (ZH)**: 评估AI代理自主性：面向可扩展方法的代码检查途径 

**Authors**: Peter Cihon, Merlin Stein, Gagan Bansal, Sam Manning, Kevin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15212)  

**Abstract**: AI agents are AI systems that can achieve complex goals autonomously. Assessing the level of agent autonomy is crucial for understanding both their potential benefits and risks. Current assessments of autonomy often focus on specific risks and rely on run-time evaluations -- observations of agent actions during operation. We introduce a code-based assessment of autonomy that eliminates the need to run an AI agent to perform specific tasks, thereby reducing the costs and risks associated with run-time evaluations. Using this code-based framework, the orchestration code used to run an AI agent can be scored according to a taxonomy that assesses attributes of autonomy: impact and oversight. We demonstrate this approach with the AutoGen framework and select applications. 

**Abstract (ZH)**: 人工智能代理是指能够自主实现复杂目标的AI系统。评估代理的自主程度对于理解其潜在优势和风险至关重要。目前对自主性的评估往往侧重于特定风险，并主要依赖于运行时评估——即在操作过程中观察代理的行为。我们提出了一种基于代码的自主性评估方法，这种方法无需运行AI代理执行特定任务即可进行评估，从而减少了运行时评估相关的成本和风险。通过这种方法，可以使用基于代码的框架对运行AI代理所使用的编排代码进行打分，根据评估自主性的分类体系考察其影响和监控属性。我们通过AutoGen框架和选定的应用程序展示了这一方法。 

---
# Multi-Agent Architecture in Distributed Environment Control Systems: vision, challenges, and opportunities 

**Title (ZH)**: 分布式环境控制系统中的多代理架构：愿景、挑战与机遇 

**Authors**: Natasha Astudillo, Fernando Koch  

**Link**: [PDF](https://arxiv.org/pdf/2502.15663)  

**Abstract**: The increasing demand for energy-efficient solutions in large-scale infrastructure, particularly data centers, requires advanced control strategies to optimize environmental management systems. We propose a multi-agent architecture for distributed control of air-cooled chiller systems in data centers. Our vision employs autonomous agents to monitor and regulate local operational parameters and optimize system-wide efficiency. We demonstrate how this approach improves the responsiveness, operational robustness, and energy efficiency of the system, contributing to the broader goal of sustainable infrastructure management. 

**Abstract (ZH)**: 随着对高效能源解决方案需求的不断增加，特别是在数据中心等大规模基础设施中，需要先进的控制策略来优化环境管理系统。我们提出了一种多代理架构，用于数据中心分布式控制空气冷却制冷系统的优化。我们的愿景是通过自主代理来监控和调节本地操作参数，并优化系统范围内的效率。我们展示了这种方法如何提高系统的响应性、运行稳定性和能源效率，从而有助于可持续基础设施管理的总体目标。 

---
# WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents 

**Title (ZH)**: WorldCraft：通过LLM代理创建和定制逼真3D世界 

**Authors**: Xinhang Liu, Chi-Keung Tang, Yu-Wing Tai  

**Link**: [PDF](https://arxiv.org/pdf/2502.15601)  

**Abstract**: Constructing photorealistic virtual worlds has applications across various fields, but it often requires the extensive labor of highly trained professionals to operate conventional 3D modeling software. To democratize this process, we introduce WorldCraft, a system where large language model (LLM) agents leverage procedural generation to create indoor and outdoor scenes populated with objects, allowing users to control individual object attributes and the scene layout using intuitive natural language commands. In our framework, a coordinator agent manages the overall process and works with two specialized LLM agents to complete the scene creation: ForgeIt, which integrates an ever-growing manual through auto-verification to enable precise customization of individual objects, and ArrangeIt, which formulates hierarchical optimization problems to achieve a layout that balances ergonomic and aesthetic considerations. Additionally, our pipeline incorporates a trajectory control agent, allowing users to animate the scene and operate the camera through natural language interactions. Our system is also compatible with off-the-shelf deep 3D generators to enrich scene assets. Through evaluations and comparisons with state-of-the-art methods, we demonstrate the versatility of WorldCraft, ranging from single-object customization to intricate, large-scale interior and exterior scene designs. This system empowers non-professionals to bring their creative visions to life. 

**Abstract (ZH)**: 构建逼真的虚拟世界在多个领域都有应用，但通常需要经过高度训练的专业人员使用传统的3D建模软件进行大量劳动。为了使这一过程民主化，我们提出了一种名为WorldCraft的系统，其中大型语言模型（LLM）代理利用程序化生成技术来创建室内和室外场景，并填充各种物体。用户可以通过直观的自然语言命令控制每个物体的属性和场景布局。在我们的框架中，协调代理管理整个过程，并与两个专门的LLM代理合作完成场景创作：ForgeIt，它通过自动验证不断扩展的手动操作来进行个体物体的精确定制；ArrangeIt，则通过构建层次优化问题实现兼顾人体工程学和美学的布局平衡。此外，我们的流水线还包含一个轨迹控制代理，允许用户通过自然语言交互来使场景动画化并操作相机。我们的系统也兼容现成的深度3D生成器以丰富场景资产。通过评估并将我们的方法与当前最先进的技术进行比较，我们展示了WorldCraft的多样性和灵活性，从单个物体的定制到复杂的大型室内和室外场景设计。该系统使非专业人员能够实现他们的创意构想。 

---
# CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models 

**Title (ZH)**: CurricuVLM：通过基于视觉-语言模型的个性化安全关键课程学习实现安全自动驾驶 

**Authors**: Zihao Sheng, Zilin Huang, Yansong Qu, Yue Leng, Sruthi Bhavanam, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15119)  

**Abstract**: Ensuring safety in autonomous driving systems remains a critical challenge, particularly in handling rare but potentially catastrophic safety-critical scenarios. While existing research has explored generating safety-critical scenarios for autonomous vehicle (AV) testing, there is limited work on effectively incorporating these scenarios into policy learning to enhance safety. Furthermore, developing training curricula that adapt to an AV's evolving behavioral patterns and performance bottlenecks remains largely unexplored. To address these challenges, we propose CurricuVLM, a novel framework that leverages Vision-Language Models (VLMs) to enable personalized curriculum learning for autonomous driving agents. Our approach uniquely exploits VLMs' multimodal understanding capabilities to analyze agent behavior, identify performance weaknesses, and dynamically generate tailored training scenarios for curriculum adaptation. Through comprehensive analysis of unsafe driving situations with narrative descriptions, CurricuVLM performs in-depth reasoning to evaluate the AV's capabilities and identify critical behavioral patterns. The framework then synthesizes customized training scenarios targeting these identified limitations, enabling effective and personalized curriculum learning. Extensive experiments on the Waymo Open Motion Dataset show that CurricuVLM outperforms state-of-the-art baselines across both regular and safety-critical scenarios, achieving superior performance in terms of navigation success, driving efficiency, and safety metrics. Further analysis reveals that CurricuVLM serves as a general approach that can be integrated with various RL algorithms to enhance autonomous driving systems. The code and demo video are available at: this https URL. 

**Abstract (ZH)**: 确保自动驾驶系统的安全性仍然是一个关键的挑战，尤其是在处理那些虽然稀少但可能导致灾难性后果的安全关键场景时。尽管现有研究已经探索了生成自动驾驶汽车（AV）测试所需的安全关键场景，但将这些场景有效地融入策略学习以增强安全性的相关工作仍然较为有限。此外，开发能够适应自动驾驶车辆不断演变的行为模式和性能瓶颈的训练课程依然鲜有研究。为了解决这些挑战，我们提出了CurricuVLM框架，这是一种新颖的方法，利用视觉语言模型（VLMs）来实现个性化课程学习，为自动驾驶代理提供定制化的训练课程。我们的方法独特地利用了VLMs的多模态理解能力，分析代理行为，识别性能短板，并动态生成定制化的训练场景以适应课程学习。通过综合分析包含叙事描述的不安全驾驶情况，CurricuVLM深入地进行推理，评估自动驾驶汽车的能力，并识别关键的行为模式。然后，框架综合生成针对性的定制化训练场景，针对识别出的局限性，实现有效和个性化的课程学习。在Waymo Open Motion数据集上的广泛实验表明，CurricuVLM在常规和安全关键场景中均优于最新的基准，其在导航成功、驾驶效率和安全指标上的表现更加出色。进一步的分析表明，CurricuVLM可以作为一种通用方法，与各种强化学习（RL）算法结合，以增强自动驾驶系统。感兴趣的读者可以通过以下链接获取相关代码和演示视频：[此处插入链接]。 

---
# Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive Learning 

**Title (ZH)**: Sce2DriveX：一种场景到驾驶学习的一般化多模态模型框架 

**Authors**: Rui Zhao, Qirui Yuan, Jinyu Li, Haofeng Hu, Yun Li, Chengyuan Zheng, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14917)  

**Abstract**: End-to-end autonomous driving, which directly maps raw sensor inputs to low-level vehicle controls, is an important part of Embodied AI. Despite successes in applying Multimodal Large Language Models (MLLMs) for high-level traffic scene semantic understanding, it remains challenging to effectively translate these conceptual semantics understandings into low-level motion control commands and achieve generalization and consensus in cross-scene driving. We introduce Sce2DriveX, a human-like driving chain-of-thought (CoT) reasoning MLLM framework. Sce2DriveX utilizes multimodal joint learning from local scene videos and global BEV maps to deeply understand long-range spatiotemporal relationships and road topology, enhancing its comprehensive perception and reasoning capabilities in 3D dynamic/static scenes and achieving driving generalization across scenes. Building on this, it reconstructs the implicit cognitive chain inherent in human driving, covering scene understanding, meta-action reasoning, behavior interpretation analysis, motion planning and control, thereby further bridging the gap between autonomous driving and human thought processes. To elevate model performance, we have developed the first extensive Visual Question Answering (VQA) driving instruction dataset tailored for 3D spatial understanding and long-axis task reasoning. Extensive experiments demonstrate that Sce2DriveX achieves state-of-the-art performance from scene understanding to end-to-end driving, as well as robust generalization on the CARLA Bench2Drive benchmark. 

**Abstract (ZH)**: 端到端自主驾驶，它直接将原始传感器输入映射到低级车辆控制，是具身人工智能的重要组成部分。尽管在利用多模态大型语言模型（MLLMs）进行高层次交通场景语义理解方面取得了一定的成果，但将这些概念性语义理解有效地转化为低级运动控制命令，并跨场景实现普遍性和一致性仍然颇具挑战性。我们提出了Sce2DriveX，这是一种类人的驾驶链式思考（CoT）推理MLLM框架。Sce2DriveX 利用局部场景视频和全局BEV图的多模态联合学习，深入理解长期时空关系和道路拓扑，从而增强其在3D动态/静态场景中的全面感知和推理能力，并实现了跨场景的驾驶通用性。在此基础上，它重建了人类驾驶中固有的隐式认知链，涵盖了场景理解、元行为推理、行为解释分析、运动规划和控制，从而进一步弥合了自动驾驶与人类思维过程之间的差距。为了提升模型性能，我们开发了首个针对3D空间理解和长轴任务推理定制的专业驾驶指令视觉问答（VQA）数据集。广泛的实验表明，Sce2DriveX 在从场景理解到端到端驾驶的过程中实现了最先进的性能，并在CARLA Bench2Drive基准测试中表现出强大的泛化能力。 

---
# OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignment 

**Title (ZH)**: OpenSearch-SQL：通过动态少样本学习和一致对齐增强文本到SQL的转换 

**Authors**: Xiangjin Xie, Guangwei Xu, Lingyan Zhao, Ruijie Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14913)  

**Abstract**: Although multi-agent collaborative Large Language Models (LLMs) have achieved significant breakthroughs in the Text-to-SQL task, their performance is still constrained by various factors. These factors include the incompleteness of the framework, failure to follow instructions, and model hallucination problems. To address these problems, we propose OpenSearch-SQL, which divides the Text-to-SQL task into four main modules: Preprocessing, Extraction, Generation, and Refinement, along with an Alignment module based on a consistency alignment mechanism. This architecture aligns the inputs and outputs of agents through the Alignment module, reducing failures in instruction following and hallucination. Additionally, we designed an intermediate language called SQL-Like and optimized the structured CoT based on SQL-Like. Meanwhile, we developed a dynamic few-shot strategy in the form of self-taught Query-CoT-SQL. These methods have significantly improved the performance of LLMs in the Text-to-SQL task.
In terms of model selection, we directly applied the base LLMs without any post-training, thereby simplifying the task chain and enhancing the framework's portability. Experimental results show that OpenSearch-SQL achieves an execution accuracy(EX) of 69.3% on the BIRD development set, 72.28% on the test set, and a reward-based validity efficiency score (R-VES) of 69.36%, with all three metrics ranking first at the time of submission. These results demonstrate the comprehensive advantages of the proposed method in both effectiveness and efficiency. 

**Abstract (ZH)**: 尽管多智能体协作的大语言模型（LLMs）在文本到SQL任务上取得了显著突破，但它们的表现仍然受到多种因素的限制。这些因素包括框架的不完整性、未能遵循指令和模型的幻觉问题。为了解决这些问题，我们提出了OpenSearch-SQL，将文本到SQL任务划分为四个主要模块：预处理、抽取、生成和精修，同时基于一致性对齐机制引入了对齐模块。这种架构通过对齐模块对智能体的输入和输出进行对齐，从而减少了指令遵循失败和幻觉问题。此外，我们设计了一种中间语言SQL-Like，并基于SQL-Like优化了结构化CoT。同时，我们开发了一种以自学习Query-CoT-SQL形式呈现的动态少量示例策略。这些方法显著提高了大语言模型在文本到SQL任务上的表现。

在模型选择方面，我们直接应用了基线LLMs，在任务链简化和框架便捷性方面进行了改进，没有进行任何后续训练。实验结果表明，OpenSearch-SQL在BIRD开发集上的执行准确性(EX)为69.3%，测试集上的执行准确性为72.28%，基于奖励的有效性效率评分(R-VES)为69.36%，三项指标均在提交时排名第一。这些结果证明了所提方法在效果和效率方面的综合优势。

总结：
尽管多智能体协作的大语言模型在文本到SQL任务上取得了显著进展，但仍面临框架不完整、指令执行不准确和模型幻觉等问题。为解决这些问题，本文提出了OpenSearch-SQL，它通过将文本到SQL任务划分为预处理、抽取、生成和精修四个模块，并基于一致性对齐机制引入了对齐模块，从而提高了任务执行的准确性和效率。此外，还设计了一种类似的SQL中间语言并优化了基于该语言的结构化CoT，开发了动态少量示例策略Query-CoT-SQL。实验结果显示，OpenSearch-SQL在多项指标上表现最优，展示了其在效果和效率方面的全面优势。 

---
# EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild 

**Title (ZH)**: EgoSpeak：学习自中心对话代理在野生环境下的发言时机 

**Authors**: Junhyeok Kim, Min Soo Kim, Jiwan Chung, Jungbin Cho, Jisoo Kim, Sungwoong Kim, Gyeongbo Sim, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14892)  

**Abstract**: Predicting when to initiate speech in real-world environments remains a fundamental challenge for conversational agents. We introduce EgoSpeak, a novel framework for real-time speech initiation prediction in egocentric streaming video. By modeling the conversation from the speaker's first-person viewpoint, EgoSpeak is tailored for human-like interactions in which a conversational agent must continuously observe its environment and dynamically decide when to talk. Our approach bridges the gap between simplified experimental setups and complex natural conversations by integrating four key capabilities: (1) first-person perspective, (2) RGB processing, (3) online processing, and (4) untrimmed video processing. We also present YT-Conversation, a diverse collection of in-the-wild conversational videos from YouTube, as a resource for large-scale pretraining. Experiments on EasyCom and Ego4D demonstrate that EgoSpeak outperforms random and silence-based baselines in real time. Our results also highlight the importance of multimodal input and context length in effectively deciding when to speak. 

**Abstract (ZH)**: 在实际环境中介定何时启动对话仍然是对话代理面临的一项基本挑战。我们提出了EgoSpeak，这是一种新型框架，用于实现实时自视角流媒体视频中的启动对话预测。通过从发言人的第一人称视角建模对话，EgoSpeak特别适用于人类交互场景，其中对话代理必须持续观察环境并动态决定何时进行对话。我们的方法通过整合四项关键技术特性，弥合了简化实验设置与复杂自然对话之间的差距：（1）第一人称视角、（2）RGB图像处理、（3）在线处理和（4）未剪辑视频处理。我们还介绍了YT-Conversation，这是一个从YouTube收集的多样化的野外对话视频集合，作为大规模预训练的资源。在EasyCom和Ego4D上的实验结果表明，EgoSpeak在实时情况下优于随机和沉默基线。我们的结果还强调了多模态输入和上下文长度在有效决定何时发言方面的重要性。 

---
