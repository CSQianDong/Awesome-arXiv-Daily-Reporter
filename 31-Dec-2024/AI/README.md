# Aviary: training language agents on challenging scientific tasks 

**Title (ZH)**: Aviary：在具有挑战性的科学任务中训练语言代理 

**Authors**: Siddharth Narayanan, James D. Braza, Ryan-Rhys Griffiths, Manu Ponnapati, Albert Bou, Jon Laurent, Ori Kabeli, Geemi Wellawatte, Sam Cox, Samuel G. Rodriques, Andrew D. White  

**Link**: [PDF](https://arxiv.org/pdf/2412.21154)  

**Abstract**: Solving complex real-world tasks requires cycles of actions and observations. This is particularly true in science, where tasks require many cycles of analysis, tool use, and experimentation. Language agents are promising for automating intellectual tasks in science because they can interact with tools via natural language or code. Yet their flexibility creates conceptual and practical challenges for software implementations, since agents may comprise non-standard components such as internal reasoning, planning, tool usage, as well as the inherent stochasticity of temperature-sampled language models. Here, we introduce Aviary, an extensible gymnasium for language agents. We formalize agents as policies solving language-grounded partially observable Markov decision processes, which we term language decision processes. We then implement five environments, including three challenging scientific environments: (1) manipulating DNA constructs for molecular cloning, (2) answering research questions by accessing scientific literature, and (3) engineering protein stability. These environments were selected for their focus on multi-step reasoning and their relevance to contemporary biology research. Finally, with online training and scaling inference-time compute, we show that language agents backed by open-source, non-frontier LLMs can match and exceed both frontier LLM agents and human experts on multiple tasks at up to 100x lower inference cost. 

**Abstract (ZH)**: 解决复杂的实际任务需要一系列的动作和观察。这一点在科学领域表现尤为明显，因为科学任务通常需要多次分析、工具使用和实验循环。语言代理在科学领域自动化智力任务方面具有巨大潜力，因为它们可以通过自然语言或代码与工具进行交互。然而，语言代理的灵活性也给软件实现带来了概念和实践上的挑战，因为这些代理可能包括非标准组件，如内部推理、规划、工具使用以及基于温度采样的语言模型固有的随机性。在此背景下，我们介绍了Aviary，一个灵活的语言代理实验平台。我们将代理精确定义为解决语言驱动的部分可观测马尔可夫决策过程的策略，我们将其称为语言决策过程。然后，我们实现了五个环境，包括三个具有挑战性的科学环境：（1）进行分子克隆时的核酸结构操作；（2）通过访问科学文献回答研究问题；（3）工程蛋白质稳定性。这些环境之所以被选择，是因为它们强调多步推理，并且与当前的生物学研究密切相关。最后，通过在线训练和扩展推理时间计算资源，我们证明基于开源非前沿的大语言模型（LLM）的语言代理能够在多个任务上达到甚至超过前沿LLM代理和人类专家的表现，且推理成本最多可降低100倍。 

---
# On Parallel External-Memory Bidirectional Search 

**Title (ZH)**: 《基于并行外部存储的双向搜索算法》

这个标题翻译成中文时，为了符合学术规范和中文表达习惯，可以对部分内容进行调整，使其更加自然和准确。原文中的 "external-Memory" 在中文中通常翻译为“外部存储”，"bidirectional search" 翻译为“双向搜索”，“parallel”则保持不变，指并行计算。完整的翻译可以是：

《并行外部存储双向搜索算法》

这样翻译既保留了原意，又符合中文的表达习惯。 

**Authors**: ior Siag, Shahaf S. Shperberg, Ariel Felner, Nathan R. Sturtevant  

**Link**: [PDF](https://arxiv.org/pdf/2412.21104)  

**Abstract**: Parallelization and External Memory (PEM) techniques have significantly enhanced the capabilities of search algorithms when solving large-scale problems. Previous research on PEM has primarily centered on unidirectional algorithms, with only one publication on bidirectional PEM that focuses on the meet-in-the-middle (MM) algorithm. Building upon this foundation, this paper presents a framework that integrates both uni- and bi-directional best-first search algorithms into this framework. We then develop a PEM variant of the state-of-the-art bidirectional heuristic search (\BiHS) algorithm BAE* (PEM-BAE*). As previous work on \BiHS did not focus on scaling problem sizes, this work enables us to evaluate bidirectional algorithms on hard problems. Empirical evaluation shows that PEM-BAE* outperforms the PEM variants of A* and the MM algorithm, as well as a parallel variant of IDA*. These findings mark a significant milestone, revealing that bidirectional search algorithms clearly outperform unidirectional search algorithms across several domains, even when equipped with state-of-the-art heuristics. 

**Abstract (ZH)**: 并行处理与外部存储（PEM）技术显著提升了在解决大规模问题时搜索算法的能力。之前对PEM的研究主要集中在单向算法上，仅有一篇关于双向PEM的研究，专注于其中的“中间相遇”（Meet-in-the-Middle，MM）算法。在此基础上，本文提出了一种框架，将单向和双向最佳优先搜索算法整合其中。在此框架下，我们进一步开发了一种基于最先进的双向启发式搜索算法（Bidirectional Heuristic Search, \BiHS）的PEM变体（PEM-BAE*）。由于之前的\BiHS研究并未关注扩大问题规模，本研究使我们能够评估双向算法在难题上的性能。实验结果表明，PEM-BAE*在性能上优于A*的PEM变体、MM算法的PEM变体以及IDA*的并行变体。这些发现标志着一个重要的里程碑，揭示出即使在配备最先进的启发式函数的情况下，双向搜索算法在多个领域中明显优于单向搜索算法。 

---
# UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI 

**Title (ZH)**: UnrealZoo：丰富逼真的虚拟世界以促进具身人工智能 

**Authors**: Fangwei Zhong, Kui Wu, Churan Wang, Hao Chen, Hai Ci, Zhoujun Li, Yizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20977)  

**Abstract**: We introduce UnrealZoo, a rich collection of photo-realistic 3D virtual worlds built on Unreal Engine, designed to reflect the complexity and variability of the open worlds. Additionally, we offer a variety of playable entities for embodied AI agents. Based on UnrealCV, we provide a suite of easy-to-use Python APIs and tools for various potential applications, such as data collection, environment augmentation, distributed training, and benchmarking. We optimize the rendering and communication efficiency of UnrealCV to support advanced applications, such as multi-agent interaction. Our experiments benchmark agents in various complex scenes, focusing on visual navigation and tracking, which are fundamental capabilities for embodied visual intelligence. The results yield valuable insights into the advantages of diverse training environments for reinforcement learning (RL) agents and the challenges faced by current embodied vision agents, including those based on RL and large vision-language models (VLMs), in open worlds. These challenges involve latency in closed-loop control in dynamic scenes and reasoning about 3D spatial structures in unstructured terrain. 

**Abstract (ZH)**: 我们介绍了UnrealZoo，这是一个基于Unreal Engine构建的丰富多样的照片级真实感3D虚拟世界集合，旨在反映开放世界的复杂性和多样性。此外，我们还提供了一系列可玩的实体，以供具身体验人工智能代理使用。基于UnrealCV，我们提供了一系列易于使用的Python接口和工具，适用于多种潜在应用，如数据收集、环境增强、分布式训练和基准测试。我们对UnrealCV的渲染和通信效率进行了优化，以支持多代理交互等高级应用。我们的实验在各种复杂的场景中对代理进行了基准测试，重点关注视觉导航和跟踪，这是具身体验视觉智能的基本能力。实验结果提供了关于各种训练环境对强化学习（RL）代理的优势以及当前的具身体验视觉代理（包括基于RL和大型视觉语言模型的代理）在开放世界中所面临的挑战的重要见解。这些挑战包括动态场景中的闭环控制延迟问题以及对未经结构化地形的3D空间结构的推理问题。 

---
# Ontology-grounded Automatic Knowledge Graph Construction by LLM under Wikidata schema 

**Title (ZH)**: 基于维基数据模式的大型语言模型引导的知识图谱自动构建方法 

**Authors**: Xiaohan Feng, Xixin Wu, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2412.20942)  

**Abstract**: We propose an ontology-grounded approach to Knowledge Graph (KG) construction using Large Language Models (LLMs) on a knowledge base. An ontology is authored by generating Competency Questions (CQ) on knowledge base to discover knowledge scope, extracting relations from CQs, and attempt to replace equivalent relations by their counterpart in Wikidata. To ensure consistency and interpretability in the resulting KG, we ground generation of KG with the authored ontology based on extracted relations. Evaluation on benchmark datasets demonstrates competitive performance in knowledge graph construction task. Our work presents a promising direction for scalable KG construction pipeline with minimal human intervention, that yields high quality and human-interpretable KGs, which are interoperable with Wikidata semantics for potential knowledge base expansion. 

**Abstract (ZH)**: 我们提出了一种基于本体的方法，利用大型语言模型（LLMs）在一个知识库上构建知识图谱（KG）。本体的构建通过生成知识库上的能力问题（CQ）来发现知识范围，从中提取关系，并尝试用维基数据中的对应关系替换等效关系。为确保生成的KG在一致性和可 Interpretability 方面保持一致，我们将基于提取的关系建立的KG生成过程与所编写的本体相结合。在基准数据集上的测试表明，该方法在知识图谱构建任务中表现出了竞争力。我们的工作为在最少人工干预的情况下构建可扩展的KG管道提供了有 promise 的方向，这种管道能够生成高质量且易于人类理解的KG，这些KG能够与维基数据语义兼容，从而为知识库的扩展提供了潜力。 

---
# HUNYUANPROVER: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving 

**Title (ZH)**: 汇融证明器：一种可扩展的数据合成框架与引导树搜索的自动化定理证明方法 

**Authors**: Yang Li, Dong Du, Linfeng Song, Chen Li, Weikang Wang, Tao Yang, Haitao Mi  

**Link**: [PDF](https://arxiv.org/pdf/2412.20735)  

**Abstract**: We introduce HunyuanProver, an language model finetuned from the Hunyuan 7B for interactive automatic theorem proving with LEAN4. To alleviate the data sparsity issue, we design a scalable framework to iterative synthesize data with low cost. Besides, guided tree search algorithms are designed to enable effective ``system 2 thinking`` of the prover. HunyuanProver achieves state-of-the-art (SOTA) performances on major benchmarks. Specifically, it achieves a pass of 68.4% on the miniF2F-test compared to 65.9%, the current SOTA results. It proves 4 IMO statements (imo_1960_p2, imo_1962_p2}, imo_1964_p2 and imo_1983_p6) in miniF2F-test. To benefit the community, we will open-source a dataset of 30k synthesized instances, where each instance contains the original question in natural language, the converted statement by autoformalization, and the proof by HunyuanProver. 

**Abstract (ZH)**: 我们将介绍HunyuanProver，这是一种从Hunyuan 7B微调而来的大语言模型，用于与LEAN4进行交互式的自动定理证明。为了解决数据稀疏的问题，我们设计了一个可扩展的框架，用于以低成本迭代生成数据。此外，我们还设计了引导树搜索算法，以使证明器能够有效地进行“系统2思考”。HunyuanProver在主要基准测试中达到了最先进的性能。具体而言，它在miniF2F-test中的通过率为68.4%，优于目前最先进的结果65.9%。在miniF2F-test中，它成功证明了4个IMO命题（imo_1960_p2, imo_1962_p2, imo_1964_p2和imo_1983_p6）。为了促进社区的发展，我们将开源一个包含30,000个合成实例的数据集，每个实例包含原始问题的自然语言描述、自动形式化后的命题以及HunyuanProver的证明。 

---
# Predicting Long Term Sequential Policy Value Using Softer Surrogates 

**Title (ZH)**: 使用更柔和的替代目标预测长期顺序政策价值 

**Authors**: Hyunji Nam, Allen Nie, Ge Gao, Vasilis Syrgkanis, Emma Brunskill  

**Link**: [PDF](https://arxiv.org/pdf/2412.20638)  

**Abstract**: Performing policy evaluation in education, healthcare and online commerce can be challenging, because it can require waiting substantial amounts of time to observe outcomes over the desired horizon of interest. While offline evaluation methods can be used to estimate the performance of a new decision policy from historical data in some cases, such methods struggle when the new policy involves novel actions or is being run in a new decision process with potentially different dynamics. Here we consider how to estimate the full-horizon value of a new decision policy using only short-horizon data from the new policy, and historical full-horizon data from a different behavior policy. We introduce two new estimators for this setting, including a doubly robust estimator, and provide formal analysis of their properties. Our empirical results on two realistic simulators, of HIV treatment and sepsis treatment, show that our methods can often provide informative estimates of a new decision policy ten times faster than waiting for the full horizon, highlighting that it may be possible to quickly identify if a new decision policy, involving new actions, is better or worse than existing past policies. 

**Abstract (ZH)**: 在教育、医疗和在线商业等领域执行策略评估具有挑战性，因为这可能需要等待相当长的时间才能观察到在期望的时间框架内的结果。虽然在某些情况下可以使用离线评估方法从历史数据中估计新决策策略的表现，但在新策略涉及新的行动或在具有潜在不同动态的新决策过程中运行时，这些方法往往难以发挥作用。本文探讨如何仅使用新策略的短期数据和不同行为策略的历史长期数据来估计新决策策略的全时间框架价值。我们引入了两种新的估计器，包括双重鲁棒估计器，并对其性质进行了形式分析。我们在两种现实的模拟器（艾滋病治疗和脓毒症治疗）上的实验结果表明，我们的方法往往可以在等待完整时间框架所需时间的十分之一时间里提供有关新决策策略的信息性估计，突显了能够快速确定新涉及新行动的决策策略是否优于现有过去策略的可能性。 

---
# The intrinsic motivation of reinforcement and imitation learning for sequential tasks 

**Title (ZH)**: 强化学习和模仿学习在序列任务中的内在动机 

**Authors**: Sao Mai Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20573)  

**Abstract**: This work in the field of developmental cognitive robotics aims to devise a new domain bridging between reinforcement learning and imitation learning, with a model of the intrinsic motivation for learning agents to learn with guidance from tutors multiple tasks, including sequential tasks. The main contribution has been to propose a common formulation of intrinsic motivation based on empirical progress for a learning agent to choose automatically its learning curriculum by actively choosing its learning strategy for simple or sequential tasks: which task to learn, between autonomous exploration or imitation learning, between low-level actions or task decomposition, between several tutors. The originality is to design a learner that benefits not only passively from data provided by tutors, but to actively choose when to request tutoring and what and whom to ask. The learner is thus more robust to the quality of the tutoring and learns faster with fewer demonstrations. We developed the framework of socially guided intrinsic motivation with machine learning algorithms to learn multiple tasks by taking advantage of the generalisability properties of human demonstrations in a passive manner or in an active manner through requests of demonstrations from the best tutor for simple and composing subtasks. The latter relies on a representation of subtask composition proposed for a construction process, which should be refined by representations used for observational processes of analysing human movements and activities of daily living. With the outlook of a language-like communication with the tutor, we investigated the emergence of a symbolic representation of the continuous sensorimotor space and of tasks using intrinsic motivation. We proposed within the reinforcement learning framework, a reward function for interacting with tutors for automatic curriculum learning in multi-task learning. 

**Abstract (ZH)**: 本研究领域的发展认知机器人学旨在设计一种新的领域桥梁，该桥梁连接强化学习和模仿学习，并提出了一种内在动机模型，使学习代理能够在导师的指导下，主动选择学习策略以学习包括序列任务在内的多种任务。主要贡献在于提出了一种基于经验进步的内在动机共同形式，使学习代理能够自动其学习课程，通过主动选择学习策略（即选择学习哪个任务、自主探索还是模仿学习、低级动作还是任务分解、以及哪个导师）来决定学习过程。其新颖之处在于设计了一种学习者，不仅被动地从导师提供的数据中受益，还能主动选择何时请求指导以及请求什么样的指导和从哪个导师那里请求指导。因此，这种学习者更加 robust，并能通过较少的示范更快地学习。我们通过结合被动利用人类示范的一般化特性和主动请求最佳导师的示范来利用这些特性，开发了社会引导的内在动机框架，使用机器学习算法来学习多种任务。后者依赖于一种用于构造过程的任务子任务组合表示法，该表示法应通过用于分析人类运动和日常生活活动的观察过程中的表现来进一步完善。展望与导师之间类语言的交流，我们研究了内在动机驱动下连续的传感器运动空间及其任务的符号表示的产生。并提出了在强化学习框架下的奖励函数，用于多任务学习中的自动课程学习与导师互动。 

---
# Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning 

**Title (ZH)**: 规划、居住与评判：基于多智能体LLM框架的循环城市规划系统 

**Authors**: Hang Ni, Yuzhi Wang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20505)  

**Abstract**: Urban regeneration presents significant challenges within the context of urbanization, requiring adaptive approaches to tackle evolving needs. Leveraging advancements in large language models (LLMs), we propose Cyclical Urban Planning (CUP), a new paradigm that continuously generates, evaluates, and refines urban plans in a closed-loop. Specifically, our multi-agent LLM-based framework consists of three key components: (1) Planning, where LLM agents generate and refine urban plans based on contextual data; (2) Living, where agents simulate the behaviors and interactions of residents, modeling life in the urban environment; and (3) Judging, which involves evaluating plan effectiveness and providing iterative feedback for improvement. The cyclical process enables a dynamic and responsive planning approach. Experiments on the real-world dataset demonstrate the effectiveness of our framework as a continuous and adaptive planning process. 

**Abstract (ZH)**: 城市再开发在城市化进程中面临显著挑战，要求采用适应性策略来应对不断变化的需求。借助大型语言模型（LLMs）的进步，我们提出了一种新的范式——循环城市规划（Cyclical Urban Planning，CUP），该范式通过闭环不断生成、评估和优化城市规划。具体而言，我们的基于多智能体的大型语言模型框架包括三个关键组成部分：（1）规划阶段，LLM智能体基于上下文数据生成和优化城市规划；（2）生活阶段，智能体模拟居民的行为和互动，模型化城市环境中的人类生活；（3）评估阶段，涉及评估规划的有效性并提供迭代反馈以进行改进。循环过程使得规划方法具有动态和响应性。实验结果表明，我们的框架作为连续且适应性强的规划过程是有效的。 

---
# A Comprehensive Framework for Reliable Legal AI: Combining Specialized Expert Systems and Adaptive Refinement 

**Title (ZH)**: 可靠法律AI的综合框架：结合专门专家系统和自适应精炼 

**Authors**: Sidra Nasir, Qamar Abbas, Samita Bai, Rizwan Ahmed Khan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20468)  

**Abstract**: This article discusses the evolving role of artificial intelligence (AI) in the legal profession, focusing on its potential to streamline tasks such as document review, research, and contract drafting. However, challenges persist, particularly the occurrence of "hallucinations" in AI models, where they generate inaccurate or misleading information, undermining their reliability in legal contexts. To address this, the article proposes a novel framework combining a mixture of expert systems with a knowledge-based architecture to improve the precision and contextual relevance of AI-driven legal services. This framework utilizes specialized modules, each focusing on specific legal areas, and incorporates structured operational guidelines to enhance decision-making. Additionally, it leverages advanced AI techniques like Retrieval-Augmented Generation (RAG), Knowledge Graphs (KG), and Reinforcement Learning from Human Feedback (RLHF) to improve the system's accuracy. The proposed approach demonstrates significant improvements over existing AI models, showcasing enhanced performance in legal tasks and offering a scalable solution to provide more accessible and affordable legal services. The article also outlines the methodology, system architecture, and promising directions for future research in AI applications for the legal sector. 

**Abstract (ZH)**: 本文探讨了人工智能（AI）在法律职业中的不断演变的作用，着重讨论了其在文书审查、研究和合同起草等任务上的潜在效能。然而，挑战仍然存在，尤其是在AI模型中出现的“幻觉”现象，这些模型生成的信息不准确或误导，从而削弱了它们在法律环境中的可靠性。为了解决这一问题，本文提出了一种新颖的框架，该框架将专家系统与基于知识的架构相结合，以提高AI驱动的法律服务的精确度和上下文相关性。该框架利用专门模块，每个模块专注于特定的法律领域，并结合结构化操作指南以增强决策过程。此外，该框架利用了先进的AI技术，如检索增强生成（RAG）、知识图谱（KG）和基于人类反馈的强化学习（RLHF），以提高系统的准确性。所提出的方法在现有AI模型上表现出显著改进，在法律任务中的性能得到增强，并提供了一种可扩展的解决方案，以提供更普及和经济实惠的法律服务。本文还介绍了研究方法、系统架构以及未来研究中在法律领域应用AI的有希望的方向。 

---
# High-fidelity social learning via shared episodic memories enhances collaborative foraging through mnemonic convergence 

**Title (ZH)**: 高保真社会学习通过共享情境记忆增强记忆 convergence 从而促进协作觅食 

**Authors**: Ismael T. Freire, Paul Verschure  

**Link**: [PDF](https://arxiv.org/pdf/2412.20271)  

**Abstract**: Social learning, a cornerstone of cultural evolution, enables individuals to acquire knowledge by observing and imitating others. At the heart of its efficacy lies episodic memory, which encodes specific behavioral sequences to facilitate learning and decision-making. This study explores the interrelation between episodic memory and social learning in collective foraging. Using Sequential Episodic Control (SEC) agents capable of sharing complete behavioral sequences stored in episodic memory, we investigate how variations in the frequency and fidelity of social learning influence collaborative foraging performance. Furthermore, we analyze the effects of social learning on the content and distribution of episodic memories across the group. High-fidelity social learning is shown to consistently enhance resource collection efficiency and distribution, with benefits sustained across memory lengths. In contrast, low-fidelity learning fails to outperform nonsocial learning, spreading diverse but ineffective mnemonic patterns. Novel analyses using mnemonic metrics reveal that high-fidelity social learning also fosters mnemonic group alignment and equitable resource distribution, while low-fidelity conditions increase mnemonic diversity without translating to performance gains. Additionally, we identify an optimal range for episodic memory length in this task, beyond which performance plateaus. These findings underscore the critical effects of social learning on mnemonic group alignment and distribution and highlight the potential of neurocomputational models to probe the cognitive mechanisms driving cultural evolution. 

**Abstract (ZH)**: 社会学习是文化进化的基石，使个体通过观察和模仿他人来获取知识。其效果核心在于情景记忆，这种记忆编码特定的行为序列，以促进学习和决策。本研究探讨了情景记忆和社会学习在群体觅食中的相互关系。通过使用能够共享情景记忆中存储的完整行为序列的序列情景控制（SEC）代理，我们研究了社会学习频率和准确性变化如何影响协作觅食的表现。此外，我们还分析了社会学习对群体中情景记忆内容及其分布的影响。结果显示，高保真度的社会学习能够持续提高资源收集效率和分布，并且这种优势在记忆长度变化时都能得到维持。相比之下，低保真度的学习无法超越非社会学习，只会传播多样但无效的记忆模式。通过使用记忆度量的新分析表明，高保真度的社会学习也有助于促进记忆群体一致性和资源分配的公平性，而低保真度条件则增加了记忆多样性但未能转化为性能提升。此外，我们还确定了这一任务中情景记忆长度的优化范围，超过这个范围时，表现会停滞不前。这些发现强调了社会学习对记忆群体一致性和资源分布的决定性影响，并突出了神经计算模型在探索驱动文化进化的心智机制方面的潜力。 

---
# The Emotional Spectrum of LLMs: Leveraging Empathy and Emotion-Based Markers for Mental Health Support 

**Title (ZH)**: 大型语言模型的情感谱系：利用共情和基于情感的标志物为心理健康支持赋能 

**Authors**: Alessandro De Grandi, Federico Ravenda, Andrea Raballo, Fabio Crestani  

**Link**: [PDF](https://arxiv.org/pdf/2412.20068)  

**Abstract**: The increasing demand for mental health services has highlighted the need for innovative solutions, particularly in the realm of psychological conversational AI, where the availability of sensitive data is scarce. In this work, we explored the development of a system tailored for mental health support with a novel approach to psychological assessment based on explainable emotional profiles in combination with empathetic conversational models, offering a promising tool for augmenting traditional care, particularly where immediate expertise is unavailable. Our work can be divided into two main parts, intrinsecaly connected to each other. First, we present RACLETTE, a conversational system that demonstrates superior emotional accuracy compared to state-of-the-art benchmarks in both understanding users' emotional states and generating empathetic responses during conversations, while progressively building an emotional profile of the user through their interactions. Second, we show how the emotional profiles of a user can be used as interpretable markers for mental health assessment. These profiles can be compared with characteristic emotional patterns associated with different mental disorders, providing a novel approach to preliminary screening and support. 

**Abstract (ZH)**: 随着对心理健康服务需求的增长，创新解决方案的需求变得尤为突出，尤其是在心理对话型人工智能领域，敏感数据的可用性相对稀缺。本文中，我们探讨了一种针对心理健康支持的定制系统的发展，该系统采用了基于可解释情感档案的新颖心理评估方法，结合了具有同情心的对话模型，为传统护理模式提供了增强工具，尤其是在即时专业知识不可得的情况下。我们的研究可以分为两个紧密相连的主要部分。首先，我们介绍了RACLETTE，这是一个在理解用户情感状态和生成同情心回应方面均表现出色的对话系统，通过用户的互动逐渐构建其情感档案。其次，我们展示了用户的情感档案如何作为心理健康评估的可解释指标使用。这些档案可以与不同类型精神障碍的典型情感模式进行比较，提供了一种新的初步筛查和支持方法。 

---
# BaiJia: A Large Scale Role-Playing Agent Corpus of Chinese Historical Charcaters 

**Title (ZH)**: BaiJia：中国历史人物大规模角色扮演代理语料库 

**Authors**: Ting Bai, Jiazheng Kang, Jiayang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20024)  

**Abstract**: We introduce a comprehensive large-scale role-playing agent corpus, termed BaiJia, that comprises various Chinese historical characters. This corpus is noteworthy for being the pioneering compilation of low-resource data that can be utilized in large language models (LLMs) to engage in AI-driven historical role-playing agents. BaiJia addresses the challenges in terms of fragmented historical textual records in different forms and modalities, integrating various characters' information, including their biographical, literary, family relations, historical events, and so on. We conduct extensive experiments to demonstrate the effectiveness of our BaiJia agent corpus in bolstering the role-playing abilities of various foundational LLMs, and promoting the development and assessment of LLMs in the context of historical role-playing tasks. The agent corpus is available at this http URL. 

**Abstract (ZH)**: 我们介绍了涵盖各种中国历史人物的综合性大型角色扮演游戏语料库，命名为BaiJia。该语料库值得注意的是，它是用于大型语言模型（LLMs）进行AI驱动的历史角色扮演游戏的初级数据汇编。BaiJia克服了不同形式和模态的历史文本记录碎片化的问题，整合了各种人物的信息，包括生平、文学、家庭关系、历史事件等。我们通过广泛的实验展示了BaiJia代理语料库在增强各类基础LLM的角色扮演能力方面的有效性，并促进了历史角色扮演游戏任务背景下LLM的发展与评估。该代理语料库可在以下链接访问：[请将具体的URL链接替换到这里]。 

---
# Action-Agnostic Point-Level Supervision for Temporal Action Detection 

**Title (ZH)**: 基于动作无关的点级别监督的时空动作检测 

**Authors**: Shuhei M. Yoshida, Takashi Shibata, Makoto Terao, Takayuki Okatani, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2412.21205)  

**Abstract**: We propose action-agnostic point-level (AAPL) supervision for temporal action detection to achieve accurate action instance detection with a lightly annotated dataset. In the proposed scheme, a small portion of video frames is sampled in an unsupervised manner and presented to human annotators, who then label the frames with action categories. Unlike point-level supervision, which requires annotators to search for every action instance in an untrimmed video, frames to annotate are selected without human intervention in AAPL supervision. We also propose a detection model and learning method to effectively utilize the AAPL labels. Extensive experiments on the variety of datasets (THUMOS '14, FineAction, GTEA, BEOID, and ActivityNet 1.3) demonstrate that the proposed approach is competitive with or outperforms prior methods for video-level and point-level supervision in terms of the trade-off between the annotation cost and detection performance. 

**Abstract (ZH)**: 我们提出了一种适用于时间动作检测的动作无关点级（AAPL）监督方法，以实现使用少量标注数据的精确动作实例检测。在所提出的方法中，一小部分视频帧以无监督的方式进行采样并呈现给人类标注者，随后他们使用动作类别对这些帧进行标注。与点级监督不同，后者要求标注者在未剪裁的视频中搜索每个动作实例，而AAPL监督在选择待标注的帧时不依赖人工干预。此外，我们还提出了一种检测模型和学习方法，以有效利用AAPL标签。在THUMOS '14、FineAction、GTEA、BEOD和ActivityNet 1.3等多个数据集上的广泛实验表明，所提出的方法在标注成本和检测性能的权衡上与或优于早期的视频级和点级监督方法。 

---
# Adversarial Attack and Defense for LoRa Device Identification and Authentication via Deep Learning 

**Title (ZH)**: 利用深度学习的LoRa设备识别与认证的对抗攻击与防御 

**Authors**: Yalin E. Sagduyu, Tugba Erpek  

**Link**: [PDF](https://arxiv.org/pdf/2412.21164)  

**Abstract**: LoRa provides long-range, energy-efficient communications in Internet of Things (IoT) applications that rely on Low-Power Wide-Area Network (LPWAN) capabilities. Despite these merits, concerns persist regarding the security of LoRa networks, especially in situations where device identification and authentication are imperative to secure the reliable access to the LoRa networks. This paper explores a deep learning (DL) approach to tackle these concerns, focusing on two critical tasks, namely (i) identifying LoRa devices and (ii) classifying them to legitimate and rogue devices. Deep neural networks (DNNs), encompassing both convolutional and feedforward neural networks, are trained for these tasks using actual LoRa signal data. In this setting, the adversaries may spoof rogue LoRa signals through the kernel density estimation (KDE) method based on legitimate device signals that are received by the adversaries. Two cases are considered, (i) training two separate classifiers, one for each of the two tasks, and (ii) training a multi-task classifier for both tasks. The vulnerabilities of the resulting DNNs to manipulations in input samples are studied in form of untargeted and targeted adversarial attacks using the Fast Gradient Sign Method (FGSM). Individual and common perturbations are considered against single-task and multi-task classifiers for the LoRa signal analysis. To provide resilience against such attacks, a defense approach is presented by increasing the robustness of classifiers with adversarial training. Results quantify how vulnerable LoRa signal classification tasks are to adversarial attacks and emphasize the need to fortify IoT applications against these subtle yet effective threats. 

**Abstract (ZH)**: LoRa 在依赖低功耗广域网（LPWAN）能力的物联网（IoT）应用中提供了远距离和高效能的通信。尽管如此，人们仍在担心 LoRa 网络的安全性，特别是在需要对设备进行身份验证和认证以确保可靠接入网络的情况下。本文探讨了利用深度学习（DL）方法应对这些担忧，重点关注以下两个关键任务：一是识别 LoRa 设备，二是将它们分类为合法设备或恶意设备。卷积神经网络（CNN）和前馈神经网络（FNN）等深度神经网络（DNNs）利用实际的 LoRa 信号数据训练以完成这些任务。在此条件下，对手可以通过内核密度估计（KDE）方法伪造恶意 LoRa 信号而利用合法设备信号。本文考虑了两种情况：一是分别训练两个分类器，一个用于每个任务；二是同时训练一个针对两个任务的多任务分类器。通过使用快速梯度符号方法（FGSM）进行非目标和目标对抗攻击，研究了由此产生的 DNNs 对输入样本操纵的脆弱性。针对单任务和多任务分类器对 LoRa 信号进行分析时的不同扰动进行了研究。为增强其抵御此类攻击的能力，提出了一种防御方法，即通过对抗训练提高分类器的鲁棒性。结果量化了 LoRa 信号分类任务对对抗攻击的脆弱性，并强调了需要加强对 IoT 应用的防护，以抵御这些微妙但有效的威胁。 

---
# Open RAN-Enabled Deep Learning-Assisted Mobility Management for Connected Vehicles 

**Title (ZH)**: 基于Open RAN和深度学习辅助的车联网移动性管理 

**Authors**: Maria Barbosa, Kelvin Dias  

**Link**: [PDF](https://arxiv.org/pdf/2412.21161)  

**Abstract**: Connected Vehicles (CVs) can leverage the unique features of 5G and future 6G/NextG networks to enhance Intelligent Transportation System (ITS) services. However, even with advancements in cellular network generations, CV applications may experience communication interruptions in high-mobility scenarios due to frequent changes of serving base station, also known as handovers (HOs). This paper proposes the adoption of Open Radio Access Network (Open RAN/O-RAN) and deep learning models for decision-making to prevent Quality of Service (QoS) degradation due to HOs and to ensure the timely connectivity needed for CV services. The solution utilizes the O-RAN Software Community (OSC), an open-source O-RAN platform developed by the collaboration between the O-RAN Alliance and Linux Foundation, to develop xApps that are executed in the near-Real-Time RIC of OSC. To demonstrate the proposal's effectiveness, an integrated framework combining the OMNeT++ simulator and OSC was created. Evaluations used real-world datasets in urban application scenarios, such as video streaming transmission and over-the-air (OTA) updates. Results indicate that the proposal achieved superior performance and reduced latency compared to the standard 3GPP HO procedure. 

**Abstract (ZH)**: 自动驾驶车辆（CVs）可以利用5G和未来6G/NextG网络的独特特征来增强智能交通系统（ITS）服务。然而，即使在蜂窝网络技术不断进步的情况下，CV应用在高移动性场景中仍可能因频繁切换服务基站（称为切换或HOs）而经历通信中断。本文提出采用开放无线接入网络（Open RAN/O-RAN）和深度学习模型进行决策，以防止由于HOs引起的服务质量（QoS）下降，并确保CV服务所需的及时连接。该解决方案利用由O-RAN联盟和Linux基金会合作开发的O-RAN软件社区（OSC）平台开发的xApps，并在OSC的近实时RIC中执行。为证明提案的有效性，创建了一个结合OMNeT++仿真器和OSC的集成框架。评估使用了在城市应用场景下实际数据集，如视频流传输和空中（OTA）更新的数据。结果表明，该提案在HO过程的标准3GPP程序中实现了更优性能并降低了延迟。 

---
# PyG-SSL: A Graph Self-Supervised Learning Toolkit 

**Title (ZH)**: PyG-SSL：图自我监督学习工具包 

**Authors**: Lecheng Zheng, Baoyu Jing, Zihao Li, Zhichen Zeng, Tianxin Wei, Mengting Ai, Xinrui He, Lihui Liu, Dongqi Fu, Jiaxuan You, Hanghang Tong, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2412.21151)  

**Abstract**: Graph Self-Supervised Learning (SSL) has emerged as a pivotal area of research in recent years. By engaging in pretext tasks to learn the intricate topological structures and properties of graphs using unlabeled data, these graph SSL models achieve enhanced performance, improved generalization, and heightened robustness. Despite the remarkable achievements of these graph SSL methods, their current implementation poses significant challenges for beginners and practitioners due to the complex nature of graph structures, inconsistent evaluation metrics, and concerns regarding reproducibility hinder further progress in this field. Recognizing the growing interest within the research community, there is an urgent need for a comprehensive, beginner-friendly, and accessible toolkit consisting of the most representative graph SSL algorithms. To address these challenges, we present a Graph SSL toolkit named PyG-SSL, which is built upon PyTorch and is compatible with various deep learning and scientific computing backends. Within the toolkit, we offer a unified framework encompassing dataset loading, hyper-parameter configuration, model training, and comprehensive performance evaluation for diverse downstream tasks. Moreover, we provide beginner-friendly tutorials and the best hyper-parameters of each graph SSL algorithm on different graph datasets, facilitating the reproduction of results. The GitHub repository of the library is this https URL. 

**Abstract (ZH)**: 图自监督学习（SSL）近年来已成为研究中的一个关键领域。通过利用未标记数据执行预设任务来学习图的复杂拓扑结构和属性，这些图SSL模型实现了性能提升、泛化能力增强和鲁棒性的提高。尽管图SSL方法取得了显著的成就，但由于图结构的复杂性、评估指标的一致性问题以及重现性问题，其当前的实现方式对初学者和实践者构成了重大挑战。鉴于研究社区日益浓厚的兴趣，迫切需要一个全面、易于入门和使用的学习工具，其中包括最具有代表性的图SSL算法。为了解决这些挑战，我们提出了一个名为PyG-SSL的图SSL工具包，该工具包基于PyTorch，并兼容多种深度学习和科学计算后端。在该工具包中，我们提供了一个统一的框架，包括数据集加载、超参数配置、模型训练和全面的性能评估，涵盖各种下游任务。此外，我们还提供了初学者友好的教程和每个图SSL算法在不同图数据集上的最佳超参数，以促进结果的重现。该库的GitHub仓库地址为：https://github.com/your-repository-url。 

---
# Facilitating large language model Russian adaptation with Learned Embedding Propagation 

**Title (ZH)**: 使用学习嵌入传播促进大型语言模型的俄语适应 

**Authors**: Mikhail Tikhomirov, Daniil Chernyshev  

**Link**: [PDF](https://arxiv.org/pdf/2412.21140)  

**Abstract**: Rapid advancements of large language model (LLM) technologies led to the introduction of powerful open-source instruction-tuned LLMs that have the same text generation quality as the state-of-the-art counterparts such as GPT-4. While the emergence of such models accelerates the adoption of LLM technologies in sensitive-information environments the authors of such models don not disclose the training data necessary for replication of the results thus making the achievements model-exclusive. Since those open-source models are also multilingual this in turn reduces the benefits of training a language specific LLMs as improved inference computation efficiency becomes the only guaranteed advantage of such costly procedure. More cost-efficient options such as vocabulary extension and subsequent continued pre-training are also inhibited by the lack of access to high-quality instruction-tuning data since it is the major factor behind the resulting LLM task-solving capabilities. To address the limitations and cut the costs of the language adaptation pipeline we propose Learned Embedding Propagation (LEP). Unlike existing approaches our method has lower training data size requirements due to minimal impact on existing LLM knowledge which we reinforce using novel ad-hoc embedding propagation procedure that allows to skip the instruction-tuning step and instead implant the new language knowledge directly into any existing instruct-tuned variant. We evaluated four Russian vocabulary adaptations for LLaMa-3-8B and Mistral-7B, showing that LEP is competitive with traditional instruction-tuning methods, achieving performance comparable to OpenChat 3.5 and LLaMa-3-8B-Instruct, with further improvements via self-calibration and continued tuning enhancing task-solving capabilities. 

**Abstract (ZH)**: 大型语言模型（LLM）技术的迅速发展导致了具有与GPT-4等最先进的模型相同文本生成质量的强大开源指令调优LLM的出现。尽管这些模型的出现加速了在敏感信息环境中的LLM技术的采用，但这些模型的开发者并未披露必要的训练数据，以便复制这些成果，这使得这些成就成为了特定模型的专属成果。由于这些开源模型是多语种模型，这意味着训练特定语言的LLM的好处减少了，因为改进的推理计算效率成为了唯一可保证的优势。此外，由于缺乏高质量指令调优数据的访问权限，词汇扩展和后续继续预训练等更低成本的选择也受到了抑制，因为这些方法的进步取决于高质量的数据。为了解决这些限制并降低语言适应管道的成本，我们提出了学习嵌入传播（LEP）方法。与现有方法相比，我们的方法对现有LLM知识的影响较小，因此所需的训练数据量也较少。我们通过一种新颖的即用型嵌入传播程序增强了这一点，该程序允许跳过指令调优步骤，而是直接将新的语言知识植入任何现有的指令调优变体中。我们对LLaMa-3-8B和Mistral-7B的四种俄语词汇适应进行了评估，表明LEP在性能上与传统的指令调优方法相当，与OpenChat 3.5和LLaMa-3-8B-Instruct相当，进一步通过自我校准和继续调优，提高了任务解决能力。 

---
# Exploring and Controlling Diversity in LLM-Agent Conversation 

**Title (ZH)**: 探索和控制大规模语言模型-代理对话中的多样性 

**Authors**: KuanChao Chu, Yi-Pei Chen, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2412.21102)  

**Abstract**: Diversity is a critical aspect of multi-agent communication. In this paper, we focus on controlling and exploring diversity in the context of open-domain multi-agent conversations, particularly for world simulation applications. We propose Adaptive Prompt Pruning (APP), a novel method that dynamically adjusts the content of the utterance generation prompt to control diversity using a single parameter, lambda. Through extensive experiments, we show that APP effectively controls the output diversity across models and datasets, with pruning more information leading to more diverse output. We comprehensively analyze the relationship between prompt content and conversational diversity. Our findings reveal that information from all components of the prompt generally constrains the diversity of the output, with the Memory block exerting the most significant influence. APP is compatible with established techniques like temperature sampling and top-p sampling, providing a versatile tool for diversity management. To address the trade-offs of increased diversity, such as inconsistencies with omitted information, we incorporate a post-generation correction step, which effectively balances diversity enhancement with output consistency. Additionally, we examine how prompt structure, including component order and length, impacts diversity. This study addresses key questions surrounding diversity in multi-agent world simulation, offering insights into its control, influencing factors, and associated trade-offs. Our contributions lay the foundation for systematically engineering diversity in LLM-based multi-agent collaborations, advancing their effectiveness in real-world applications. 

**Abstract (ZH)**: 多样性是多agents通信的一个关键方面。本文重点探讨在开放领域多agents对话中控制和探索多样性，特别是在世界模拟应用程序中的应用。我们提出了一种名为自适应提示修剪（APP）的新方法，该方法通过调整单个参数lambda来动态调整生成语句的提示内容，从而控制多样性。通过广泛的实验，我们证明APP能够在不同模型和数据集中有效地控制输出多样性，修剪更多的信息会导致更广泛的输出。我们全面分析了提示内容与对话多样性的关系。研究结果表明，提示中所有部分的信息通常都会限制输出的多样性，而Memory块的影响最为显著。APP与现有的技术和方法（如温度采样和top-p采样）兼容，提供了一种广泛适用的多样性管理工具。为应对增加多样性所带来的矛盾，如遗漏信息导致的一致性问题，我们引入了事后生成校正步骤，这有效地平衡了多样性提升与输出一致性之间的关系。此外，我们还研究了提示结构（包括各组件的顺序和长度）对多样性的影响。本研究针对多agents世界模拟中多样性控制的关键问题提供了解析，并探讨了其影响因素及其权衡。我们提出的贡献为系统地构建基于LLM的多agents合作中的多样性奠定了基础，提高了其在实际应用中的效果。 

---
# Towards Effective Discrimination Testing for Generative AI 

**Title (ZH)**: 面向生成式AI的有效歧视性测试研究 

**Authors**: Thomas P. Zollo, Nikita Rajaneesh, Richard Zemel, Talia B. Gillis, Emily Black  

**Link**: [PDF](https://arxiv.org/pdf/2412.21052)  

**Abstract**: Generative AI (GenAI) models present new challenges in regulating against discriminatory behavior. In this paper, we argue that GenAI fairness research still has not met these challenges; instead, a significant gap remains between existing bias assessment methods and regulatory goals. This leads to ineffective regulation that can allow deployment of reportedly fair, yet actually discriminatory, GenAI systems. Towards remedying this problem, we connect the legal and technical literature around GenAI bias evaluation and identify areas of misalignment. Through four case studies, we demonstrate how this misalignment between fairness testing techniques and regulatory goals can result in discriminatory outcomes in real-world deployments, especially in adaptive or complex environments. We offer practical recommendations for improving discrimination testing to better align with regulatory goals and enhance the reliability of fairness assessments in future deployments. 

**Abstract (ZH)**: 生成式人工智能（GenAI）模型在监管具有歧视行为方面提出了新的挑战。在本文中，我们指出现有的GenAI公平性研究尚未解决这些挑战；相反，现有的偏见评估方法与监管目标之间仍存在显著差距。这导致了无效的监管，可能会允许部署表面上公平但实际上具有歧视性的GenAI系统。为了弥补这一问题，我们结合了法律和技术文献中关于GenAI偏见评估的讨论，并识别出其中的不一致之处。通过四个案例研究，我们展示了这种测试技术与监管目标之间的不一致如何导致实际部署中的歧视性结果，特别是在适应性或复杂环境中尤为明显。我们提出了一些实用建议，以改善歧视性测试，更好地与监管目标保持一致，并增强未来部署中公平性评估的可靠性。 

---
# Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense 

**Title (ZH)**: 向智能和安全的云迈进：大型语言模型赋能的主动防御 

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.21051)  

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods. 

**Abstract (ZH)**: 云计算技术的快速发展以及云应用程序的不断增加为日常生活带来了大量便利。然而，不同组件的多样性和复杂性给云计算安全带来了重大挑战，尤其是在应对复杂和先进的网络攻击时。近年来，生成基础模型（GFMs），尤其是大规模语言模型（LLMs）的进步为安全智能提供了有望的解决方案。通过利用其强大的语言理解、数据分析、任务推理、行动规划和代码生成能力，我们提出了一种名为LLM-PD的新颖主动防御架构，能够以主动的方式抵御各种威胁。LLM-PD能够通过全面的数据分析和顺序推理高效做出决策，并根据目标云的需求动态生成和部署可操作的防御机制。此外，它可以根据以往交互中获得的经验灵活自我进化，并在无需额外训练的情况下适应新的攻击场景。实验结果证明了其在防御有效性和效率方面的出色能力，并特别强调了与其他现有方法相比时取得的卓越成功率。 

---
# TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization 

**Title (ZH)**: TangoFlux：基于流匹配和拍次偏好优化的超高速高保真文本转音频生成 

**Authors**: Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Rafael Valle, Bryan Catanzaro, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2412.21037)  

**Abstract**: We introduce TangoFlux, an efficient Text-to-Audio (TTA) generative model with 515M parameters, capable of generating up to 30 seconds of 44.1kHz audio in just 3.7 seconds on a single A40 GPU. A key challenge in aligning TTA models lies in the difficulty of creating preference pairs, as TTA lacks structured mechanisms like verifiable rewards or gold-standard answers available for Large Language Models (LLMs). To address this, we propose CLAP-Ranked Preference Optimization (CRPO), a novel framework that iteratively generates and optimizes preference data to enhance TTA alignment. We demonstrate that the audio preference dataset generated using CRPO outperforms existing alternatives. With this framework, TangoFlux achieves state-of-the-art performance across both objective and subjective benchmarks. We open source all code and models to support further research in TTA generation. 

**Abstract (ZH)**: 我们介绍了TangoFlux，这是一种高效的文本到音频（Text-to-Audio, TTA）生成模型，参数量为515M，能够在单个A40 GPU上仅用3.7秒生成长达30秒、采样率为44.1kHz的音频。TTA模型在对齐方面面临的主要挑战之一是难以创建偏好对，因为TTA缺乏与大型语言模型（LLM）可用的验证奖励或黄金标准答案等结构化的机制。为了解决这一问题，我们提出了CLAP排名偏好优化（CLAP-Ranked Preference Optimization, CRPO）这一新颖框架，该框架通过迭代生成和优化偏好数据来增强TTA对齐。我们证明使用CRPO生成的音频偏好数据集优于现有替代方案。通过这一框架，TangoFlux在客观和主观基准测试中均实现了最先进的性能。我们开源了所有代码和模型，以支持TTA生成领域的进一步研究。 

---
# Plancraft: an evaluation dataset for planning with LLM agents 

**Title (ZH)**: PlanCraft：用于评估基于LLM代理的规划数据集 

**Authors**: Gautier Dagan, Frank Keller, Alex Lascarides  

**Link**: [PDF](https://arxiv.org/pdf/2412.21033)  

**Abstract**: We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as an oracle planner and oracle RAG information extractor, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and strategies on our task and compare their performance to a handcrafted planner. We find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and we offer suggestions on how to improve their capabilities. 

**Abstract (ZH)**: 我们介绍了Plancraft，这是一个针对大规模语言模型（LLM）代理的跨模态评估数据集。Plancraft 支持文本-only 和跨模态两种界面，基于《我的世界》（Minecraft）的制作用户界面（GUI）。我们包含了《我的世界》维基，用于评估工具使用和检索增强生成（RAG），同时也提供了一个 oracle 计划者和 oracle RAG 信息提取器，以消除现代代理架构中不同组件的影响。为了评估决策能力，Plancraft 还包括了一部分故意无法解决的示例，这些示例为代理提供了真实的挑战，不仅需要代理完成任务，还需要代理判断这些任务是否可解。我们对开源和封闭源代码的 LLM 和策略进行了基准测试，并将它们的表现与手工设计的计划者进行了比较。我们发现，LLM 和视觉-语言模型在处理 Plancraft 引入的规划问题时表现不佳，并提出了提高其能力的建议。 

---
# Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale via Principled Criteria 

**Title (ZH)**: 基于冗余性原则评估的verbosity意识摘要生成：有效减少冗余理由的方法 

**Authors**: Joonwon Jang, Jaehee Kim, Wonbin Kweon, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.21006)  

**Abstract**: Large Language Models (LLMs) rely on generating extensive intermediate reasoning units (e.g., tokens, sentences) to enhance final answer quality across a wide range of complex tasks. While generating multiple reasoning paths or iteratively refining rationales proves effective for improving performance, these approaches inevitably result in significantly higher inference costs. In this work, we propose a novel sentence-level rationale reduction training framework that leverages likelihood-based criteria, verbosity, to identify and remove redundant reasoning sentences. Unlike previous approaches that utilize token-level reduction, our sentence-level reduction framework maintains model performance while reducing generation length. This preserves the original reasoning abilities of LLMs and achieves an average 17.15% reduction in generation costs across various models and tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）依赖于生成大量的中间推理单元（例如，词元、句子）以在多种复杂的任务中提高最终答案的质量。尽管生成多条推理路径或迭代优化理由已被证明能有效提高性能，但这些方法不可避免地会导致显著增加推理成本。在本项研究中，我们提出了一种新颖的句子级理由缩减训练框架，该框架利用基于似然性的标准（如冗余度），以识别并移除冗余的推理句子。不同于以往依赖于词元级缩减的方法，我们的句子级缩减框架能够在保持模型性能的同时减少生成长度。这保留了LLMs的原始推理能力，并在各种模型和任务中实现平均每种模型17.15%的生成成本降低。 

---
# LEASE: Offline Preference-based Reinforcement Learning with High Sample Efficiency 

**Title (ZH)**: LEASE：基于偏好 Offline 强化学习的高样本效率方法 

**Authors**: Xiao-Yin Liu, Guotao Li, Xiao-Hu Zhou, Zeng-Guang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2412.21001)  

**Abstract**: Offline preference-based reinforcement learning (PbRL) provides an effective way to overcome the challenges of designing reward and the high costs of online interaction. However, since labeling preference needs real-time human feedback, acquiring sufficient preference labels is challenging. To solve this, this paper proposes a offLine prEference-bAsed RL with high Sample Efficiency (LEASE) algorithm, where a learned transition model is leveraged to generate unlabeled preference data. Considering the pretrained reward model may generate incorrect labels for unlabeled data, we design an uncertainty-aware mechanism to ensure the performance of reward model, where only high confidence and low variance data are selected. Moreover, we provide the generalization bound of reward model to analyze the factors influencing reward accuracy, and demonstrate that the policy learned by LEASE has theoretical improvement guarantee. The developed theory is based on state-action pair, which can be easily combined with other offline algorithms. The experimental results show that LEASE can achieve comparable performance to baseline under fewer preference data without online interaction. 

**Abstract (ZH)**: 基于 offline 偏好强化学习 (PbRL) 提供了一种有效的方法来克服设计奖励的挑战以及在线交互的高成本。然而，由于标签偏好需要实时的人类反馈，获取足够的偏好标签仍然是一个挑战。为了解决这个问题，本文提出了一种高效样本获取的基于 offline 偏好的 RL (LEASE, Low-Effort Sample Efficiency for Offline Preference-based RL) 算法，其中利用学习到的转换模型生成未标记的偏好数据。考虑到预先训练的奖励模型可能会为未标记数据生成错误的标签，我们设计了一种不确定性感知机制，以确保奖励模型的性能，在此机制下仅选择高置信度和低方差的数据。此外，我们提供了奖励模型的一般化界，分析影响奖励准确性因素，并证明了 LEASE 学习的策略具有理论上的改进保证。该开发理论基于状态-动作对，可以轻松与其它 offline 算法结合。实验结果显示，在较少的偏好数据下，LEASE 可以实现与基线方法相当的性能，且无需在线交互。 

---
# KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model's Reasoning Path Aggregation 

**Title (ZH)**: KARPA：一种无需训练的方法，将知识图谱适应为大型语言模型推理路径聚合的参考 

**Authors**: Siyuan Fang, Kaijing Ma, Tianyu Zheng, Xinrun Du, Ningxuan Lu, Ge Zhang, Qingkun Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20995)  

**Abstract**: Large language models (LLMs) demonstrate exceptional performance across a variety of tasks, yet they are often affected by hallucinations and the timeliness of knowledge. Leveraging knowledge graphs (KGs) as external knowledge sources has emerged as a viable solution, but existing methods for LLM-based knowledge graph question answering (KGQA) are often limited by step-by-step decision-making on KGs, restricting the global planning and reasoning capabilities of LLMs, or they require fine-tuning or pre-training on specific KGs. To address these challenges, we propose Knowledge graph Assisted Reasoning Path Aggregation (KARPA), a novel framework that harnesses the global planning abilities of LLMs for efficient and accurate KG reasoning. KARPA operates in three steps: pre-planning relation paths using the LLM's global planning capabilities, matching semantically relevant paths via an embedding model, and reasoning over these paths to generate answers. Unlike existing KGQA methods, KARPA avoids stepwise traversal, requires no additional training, and is adaptable to various LLM architectures. Extensive experimental results show that KARPA achieves state-of-the-art performance in KGQA tasks, delivering both high efficiency and accuracy. Our code will be available on Github. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色，但常常受到幻觉和知识时效性的影响。利用知识图谱（KGs）作为外部知识源已经出现了可行的解决方案，但现有基于LLM的知识图谱问答（KGQA）方法往往受限于逐级决策的过程，限制了LLM的全局规划和推理能力，或者需要针对特定的KG进行微调或预训练。为了解决这些问题，我们提出了一种名为知识图谱辅助推理路径聚合（KARPA）的新框架，该框架利用LLM的全局规划能力进行高效和准确的知识图谱推理。KARPA的工作流程分为三个步骤：利用LLM的全局规划能力预先规划关系路径，通过嵌入模型匹配语义相关路径，并在这些路径上进行推理生成答案。与现有的KGQA方法相比，KARPA避免了逐级遍历，不需要额外的训练，并且可以适应各种LLM架构。实验结果表明，KARPA在KGQA任务中达到了最先进的性能，既高效又准确。我们的代码将在GitHub上开源。 

---
# Conservation-informed Graph Learning for Spatiotemporal Dynamics Prediction 

**Title (ZH)**: 基于保护信息的图学习方法在空间-temporal动态预测中的应用 

**Authors**: Yuan Mi, Pu Ren, Hongteng Xu, Hongsheng Liu, Zidong Wang, Yike Guo, Ji-Rong Wen, Hao Sun, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20962)  

**Abstract**: Data-centric methods have shown great potential in understanding and predicting spatiotemporal dynamics, enabling better design and control of the object system. However, pure deep learning models often lack interpretability, fail to obey intrinsic physics, and struggle to cope with the various domains. While geometry-based methods, e.g., graph neural networks (GNNs), have been proposed to further tackle these challenges, they still need to find the implicit physical laws from large datasets and rely excessively on rich labeled data. In this paper, we herein introduce the conservation-informed GNN (CiGNN), an end-to-end explainable learning framework, to learn spatiotemporal dynamics based on limited training data. The network is designed to conform to the general conservation law via symmetry, where conservative and non-conservative information passes over a multiscale space enhanced by a latent temporal marching strategy. The efficacy of our model has been verified in various spatiotemporal systems based on synthetic and real-world datasets, showing superiority over baseline models. Results demonstrate that CiGNN exhibits remarkable accuracy and generalization ability, and is readily applicable to learning for prediction of various spatiotemporal dynamics in a spatial domain with complex geometry. 

**Abstract (ZH)**: 以数据为中心的方法在理解与预测时空动态方面展现了巨大的潜力，有助于更好地设计和控制对象系统。然而，纯粹的深度学习模型往往缺乏可解释性，不能遵守内在的物理定律，并且难以应对各种应用场景。而基于几何的方法，例如图神经网络（GNNs），虽然被提出以进一步解决这些问题，但它们仍然需要从大数据集中挖掘隐含的物理定律，并且过度依赖于丰富的标注数据。在本文中，我们介绍了一种端到端的可解释学习框架——守恒导向的GNN（CiGNN），旨在基于有限的训练数据学习时空动态。网络设计通过对称性来符合通用守恒定律，其中保守和非保守信息通过增强的潜藏时间推进策略在多尺度空间中传递。我们的模型在基于合成数据集和真实世界数据集的多种时空系统中得到了验证，表现出色于基线模型。结果表明，CiGNN 在时空动态预测方面具有显著的准确性和泛化能力，并且可以方便地应用于复杂几何空间域中各种时空动态的学习。 

---
# Rise of Generative Artificial Intelligence in Science 

**Title (ZH)**: 生成式人工智能在科学中的崛起 

**Authors**: Liangping Ding, Cornelia Lawson, Philip Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2412.20960)  

**Abstract**: Generative Artificial Intelligence (GenAI, generative AI) has rapidly become available as a tool in scientific research. To explore the use of generative AI in science, we conduct an empirical analysis using OpenAlex. Analyzing GenAI publications and other AI publications from 2017 to 2023, we profile growth patterns, the diffusion of GenAI publications across fields of study, and the geographical spread of scientific research on generative AI. We also investigate team size and international collaborations to explore whether GenAI, as an emerging scientific research area, shows different collaboration patterns compared to other AI technologies. The results indicate that generative AI has experienced rapid growth and increasing presence in scientific publications. The use of GenAI now extends beyond computer science to other scientific research domains. Over the study period, U.S. researchers contributed nearly two-fifths of global GenAI publications. The U.S. is followed by China, with several small and medium-sized advanced economies demonstrating relatively high levels of GenAI deployment in their research publications. Although scientific research overall is becoming increasingly specialized and collaborative, our results suggest that GenAI research groups tend to have slightly smaller team sizes than found in other AI fields. Furthermore, notwithstanding recent geopolitical tensions, GenAI research continues to exhibit levels of international collaboration comparable to other AI technologies. 

**Abstract (ZH)**: 生成式人工智能（GenAI，生成型AI）已迅速成为科学研究中的一个工具。为了探索生成式人工智能在科学中的应用，我们使用OpenAlex进行实证分析。通过对2017年至2023年发布的生成式AI论文及其他AI论文进行分析，我们描绘了生成式AI论文的增长模式，研究了生成式AI论文在各个研究领域中的扩散情况，以及科学界在生成式AI研究中的地理分布。我们还调查了团队规模和国际合作情况，以探讨生成式AI作为一个新兴科学研究领域是否表现出与其他AI技术不同的合作模式。研究结果表明，生成式AI在科学研究中的增长迅速且影响力持续增强。目前，生成式AI的应用已不仅局限于计算机科学，还扩展到了其他科学研究领域。在研究期间，美国研究人员贡献了全球近四分之一的生成式AI论文。美国之后是中国，一些较小和中等规模的先进经济体也显示出在其研究论文中相对高水平的生成式AI部署。尽管科学研究总体上越来越专业化和合作化，但我们的结果表明，生成式AI研究团队的规模略小于其他AI领域。此外，尽管存在近期的政治地缘紧张局势，生成式AI研究的国际合作水平仍与其它AI技术相当。 

---
# HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization 

**Title (ZH)**: HisynSeg：基于图像混合合成和一致性正则化的弱监督病理图像分割方法 

**Authors**: Zijie Fang, Yifeng Wang, Peizhang Xie, Zhi Wang, Yongbing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20924)  

**Abstract**: Tissue semantic segmentation is one of the key tasks in computational pathology. To avoid the expensive and laborious acquisition of pixel-level annotations, a wide range of studies attempt to adopt the class activation map (CAM), a weakly-supervised learning scheme, to achieve pixel-level tissue segmentation. However, CAM-based methods are prone to suffer from under-activation and over-activation issues, leading to poor segmentation performance. To address this problem, we propose a novel weakly-supervised semantic segmentation framework for histopathological images based on image-mixing synthesis and consistency regularization, dubbed HisynSeg. Specifically, synthesized histopathological images with pixel-level masks are generated for fully-supervised model training, where two synthesis strategies are proposed based on Mosaic transformation and Bézier mask generation. Besides, an image filtering module is developed to guarantee the authenticity of the synthesized images. In order to further avoid the model overfitting to the occasional synthesis artifacts, we additionally propose a novel self-supervised consistency regularization, which enables the real images without segmentation masks to supervise the training of the segmentation model. By integrating the proposed techniques, the HisynSeg framework successfully transforms the weakly-supervised semantic segmentation problem into a fully-supervised one, greatly improving the segmentation accuracy. Experimental results on three datasets prove that the proposed method achieves a state-of-the-art performance. Code is available at this https URL. 

**Abstract (ZH)**: 组织语义分割是计算病理学中的关键任务之一。为了避免在像素级标注上耗费昂贵的人力成本，许多研究尝试采用类激活图（CAM），一种弱监督学习方案，以实现像素级组织分割。然而，基于CAM的方法容易遭受激活不足和激活过度的问题，导致分割性能较差。为了解决这一问题，我们提出了一种基于图像混合合成和一致性正则化的新颖弱监督语义分割框架，称为HisynSeg。具体而言，通过Mosaic变换和贝塞尔掩膜生成两种合成策略，生成带有像素级掩膜的合成组织病理图像，用于完全监督模型的训练。此外，我们还开发了一种图像过滤模块，以确保合成图像的真实性。为了进一步避免模型对偶然出现的合成伪影过度适应，我们还提出了一种新颖的自监督一致性正则化方法，使没有分割掩膜的真实图像能够监督分割模型的训练。通过整合上述技术，HisynSeg框架成功地将弱监督语义分割问题转化为完全监督问题，显著提高了分割精度。在三个数据集上的实验结果证明了所提方法达到了最先进的性能。代码可在以下链接获得：this https URL。 

---
# WalkVLM:Aid Visually Impaired People Walking by Vision Language Model 

**Title (ZH)**: WalkVLM：通过视觉语言模型辅助视障人士行走 

**Authors**: Zhiqiang Yuan, Ting Zhang, Jiapei Zhang, Jie Zhou, Jinchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20903)  

**Abstract**: Approximately 200 million individuals around the world suffer from varying degrees of visual impairment, making it crucial to leverage AI technology to offer walking assistance for these people. With the recent progress of vision-language models (VLMs), employing VLMs to improve this field has emerged as a popular research topic. However, most existing methods are studied on self-built question-answering datasets, lacking a unified training and testing benchmark for walk guidance. Moreover, in blind walking task, it is necessary to perform real-time streaming video parsing and generate concise yet informative reminders, which poses a great challenge for VLMs that suffer from redundant responses and low inference efficiency. In this paper, we firstly release a diverse, extensive, and unbiased walking awareness dataset, containing 12k video-manual annotation pairs from Europe and Asia to provide a fair training and testing benchmark for blind walking task. Furthermore, a WalkVLM model is proposed, which employs chain of thought for hierarchical planning to generate concise but informative reminders and utilizes temporal-aware adaptive prediction to reduce the temporal redundancy of reminders. Finally, we have established a solid benchmark for blind walking task and verified the advantages of WalkVLM in stream video processing for this task compared to other VLMs. Our dataset and code will be released at anonymous link this https URL. 

**Abstract (ZH)**: 在全球范围内，大约有2亿人不同程度地遭受视力障碍的困扰，因此利用人工智能技术为这些人群提供行走辅助变得至关重要。随着视觉-语言模型（VLMs）的进展，使用VLMs来改善这一领域已经成为一个热门的研究方向。然而，现有的大多数方法主要在自我构建的问题-回答数据集上进行研究，缺乏一个统一的训练和测试基准来评估行走指导。此外，在盲人行走任务中，需要实时处理视频流并生成简洁但富有信息性的提醒，这给VLMs带来了很大的挑战，因为它们容易产生冗余响应和较低的推理效率。在本文中，我们首先发布了一个多样、广泛且不偏不倚的行走意识数据集，包含来自欧洲和亚洲的12000个视频-人工注释对，旨在为盲人行走任务提供一个公平的训练和测试基准。此外，我们提出了一种WalkVLM模型，该模型通过分层规划使用推理链生成简洁但富有信息性的提醒，并利用时空感知自适应预测来减少提醒中的时间冗余。最后，我们建立了一个盲人行走任务的基准，并验证了WalkVLM在处理该任务的实时视频流方面相比其他VLMs的优势。我们的数据集和代码将通过以下匿名链接发布：[https://anonymous.link]。 

---
# ILDiff: Generate Transparent Animated Stickers by Implicit Layout Distillation 

**Title (ZH)**: ILDiff: 通过隐式布局精炼生成透明动画贴纸 

**Authors**: Ting Zhang, Zhiqiang Yuan, Yeshuang Zhu, Jinchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20901)  

**Abstract**: High-quality animated stickers usually contain transparent channels, which are often ignored by current video generation models. To generate fine-grained animated transparency channels, existing methods can be roughly divided into video matting algorithms and diffusion-based algorithms. The methods based on video matting have poor performance in dealing with semi-open areas in stickers, while diffusion-based methods are often used to model a single image, which will lead to local flicker when modeling animated stickers. In this paper, we firstly propose an ILDiff method to generate animated transparent channels through implicit layout distillation, which solves the problems of semi-open area collapse and no consideration of temporal information in existing methods. Secondly, we create the Transparent Animated Sticker Dataset (TASD), which contains 0.32M high-quality samples with transparent channel, to provide data support for related fields. Extensive experiments demonstrate that ILDiff can produce finer and smoother transparent channels compared to other methods such as Matting Anything and Layer Diffusion. Our code and dataset will be released at link this https URL. 

**Abstract (ZH)**: 高质量的动画贴纸通常包含透明通道，而当前的视频生成模型往往忽略了这些透明通道。为了生成精细的动画透明通道，现有的方法可以大致分为视频分割算法和基于扩散的方法。基于视频分割的方法在处理贴纸中的半开区域时表现较差，而基于扩散的方法通常用于单一图像建模，在建模动画贴纸时会导致局部闪烁问题。在本文中，我们首先提出了一种ILDiff方法，通过隐式布局蒸馏生成动画透明通道，解决了现有方法在处理半开区域塌陷以及忽视时间信息的问题。其次，我们创建了包含0.32M高质量样本（带有透明通道）的透明动画贴纸数据集（TASD），为相关领域提供数据支持。大量的实验表明，ILDiff相较于Matting Anything和Layer Diffusion等方法可以生成更精细和更平滑的透明通道。我们的代码和数据集将在以下链接发布：[这个链接](https://example.com)。 

---
# Holistic Construction Automation with Modular Robots: From High-Level Task Specification to Execution 

**Title (ZH)**: 基于模块化机器人实现的整体化施工自动化：从高层任务规范到执行 

**Authors**: Jonathan Külz, Michael Terzer, Marco Magri, Andrea Giusti, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2412.20867)  

**Abstract**: In situ robotic automation in construction is challenging due to constantly changing environments, a shortage of robotic experts, and a lack of standardized frameworks bridging robotics and construction practices. This work proposes a holistic framework for construction task specification, optimization of robot morphology, and mission execution using a mobile modular reconfigurable robot. Users can specify and monitor the desired robot behavior through a graphical interface. Our framework identifies an optimized robot morphology and enables automatic real-world execution by integrating Building Information Modelling (BIM). By leveraging modular robot components, we ensure seamless and fast adaption to the specific demands of the construction task. Experimental validation demonstrates that our approach robustly enables the autonomous execution of robotic drilling. 

**Abstract (ZH)**: 由于施工环境不断变化、机器人专家短缺以及缺乏将机器人技术和施工实践标准化的框架，现场机器人自动化施工面临着挑战。本研究提出了一种全面框架，用于施工任务的规范、机器人形态的优化以及任务执行，采用的是移动式的可模块化重新配置机器人。用户可以通过图形界面指定和监控所需的机器人行为。本框架识别出优化的机器人形态，并通过集成建筑信息建模（BIM）实现自动的现实世界执行。通过利用模块化机器人组件，我们确保能够无缝且快速地适应特定的施工任务需求。实验验证表明，本方法能够稳健地实现自主钻孔机器人的执行。 

---
# Enhancing Annotated Bibliography Generation with LLM Ensembles 

**Title (ZH)**: 使用大型语言模型ensemble增强标注参考文献生成 

**Authors**: Sergio Bermejo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20864)  

**Abstract**: This work proposes a novel approach to enhancing annotated bibliography generation through Large Language Model (LLM) ensembles. In particular, multiple LLMs in different roles -- controllable text generation, evaluation, and summarization -- are introduced and validated using a systematic methodology to enhance model performance in scholarly tasks. Output diversity among the ensemble that generates text is obtained using different LLM parameters, followed by an LLM acting as a judge to assess relevance, accuracy, and coherence. Responses selected by several combining strategies are then merged and refined through summarization and redundancy removal techniques. The preliminary experimental validation demonstrates that the combined outputs from the LLM ensemble improve coherence and relevance compared to individual responses, leading to a 38% improvement in annotation quality and a 51% reduction in content redundancy, thus highlighting the potential for automating complex scholarly tasks while maintaining high-quality standards. 

**Abstract (ZH)**: 本文提出了一种通过大型语言模型（LLM）集成提升标注参考文献生成的新方法。具体而言，通过引入并在系统的方法论框架下验证了多个在不同角色中工作的LLM——包括可控文本生成、评估和总结——以提升模型在学术任务中的性能。通过使用不同的LLM参数生成文本，并通过一个评估模型来评估相关性、准确性和连贯性，从而实现集成生成的文本输出多样性。之后，通过总结和去除冗余的技术，采用多种组合策略选出的响应被合并和精炼。初步的实验验证表明，LLM集成组合输出在连贯性和相关性方面优于单个响应，导致注释质量提高了38%，冗余内容减少了51%。这不仅证明了自动执行复杂学术任务的可行性，还保持了高质量标准，突显了自动化在学术领域应用的潜力。 

---
# About rectified sigmoid function for enhancing the accuracy of Physics-Informed Neural Networks 

**Title (ZH)**: 关于修正的sigmoid函数用于提高物理知情神经网络的准确性 

**Authors**: Vasiliy A. Es'kin, Alexey O. Malkhanov, Mikhail E. Smorkalov  

**Link**: [PDF](https://arxiv.org/pdf/2412.20851)  

**Abstract**: The article is devoted to the study of neural networks with one hidden layer and a modified activation function for solving physical problems. A rectified sigmoid activation function has been proposed to solve physical problems described by the ODE with neural networks. Algorithms for physics-informed data-driven initialization of a neural network and a neuron-by-neuron gradient-free fitting method have been presented for the neural network with this activation function. Numerical experiments demonstrate the superiority of neural networks with a rectified sigmoid function over neural networks with a sigmoid function in the accuracy of solving physical problems (harmonic oscillator, relativistic slingshot, and Lorentz system). 

**Abstract (ZH)**: 本文致力于研究具有一个隐藏层和修改后的激活函数的神经网络，用于解决物理问题。文中提出了一种修正的Sigmoid激活函数，用于通过神经网络求解由常微分方程描述的物理问题。还介绍了适用于这种激活函数的神经网络的物理信息驱动初始化算法以及无需梯度的神经元逐个拟合方法。数值实验表明，使用修正Sigmoid函数的神经网络在求解物理问题（谐振子、相对论快递和洛伦兹系统）的准确性上优于使用标准Sigmoid函数的神经网络。 

---
# Analog Alchemy: Neural Computation with In-Memory Inference, Learning and Routing 

**Title (ZH)**: 模拟炼金术：具有内存内推断、学习和路由的神经计算 

**Authors**: Yigit Demirag  

**Link**: [PDF](https://arxiv.org/pdf/2412.20848)  

**Abstract**: As neural computation is revolutionizing the field of Artificial Intelligence (AI), rethinking the ideal neural hardware is becoming the next frontier. Fast and reliable von Neumann architecture has been the hosting platform for neural computation. Although capable, its separation of memory and computation creates the bottleneck for the energy efficiency of neural computation, contrasting the biological brain. The question remains: how can we efficiently combine memory and computation, while exploiting the physics of the substrate, to build intelligent systems? In this thesis, I explore an alternative way with memristive devices for neural computation, where the unique physical dynamics of the devices are used for inference, learning and routing. Guided by the principles of gradient-based learning, we selected functions that need to be materialized, and analyzed connectomics principles for efficient wiring. Despite non-idealities and noise inherent in analog physics, I will provide hardware evidence of adaptability of local learning to memristive substrates, new material stacks and circuit blocks that aid in solving the credit assignment problem and efficient routing between analog crossbars for scalable architectures. 

**Abstract (ZH)**: 随着神经计算正在重塑人工智能（AI）领域，重新思考理想的神经硬件已成为下一个前沿领域。快速且可靠的冯·诺伊曼架构（von Neumann architecture）一直是神经计算的承载平台。尽管如此，其计算与存储分离的设计限制了神经计算的能量效率，与生物大脑形成了对比。因此，问题在于：如何有效地集成计算与存储，同时利用底层物理特性，构建智能系统？在本论文中，我将探讨使用忆阻器进行神经计算的替代方案，其中通过利用器件的独特物理动力学来进行推断、学习和路由。基于梯度学习的原则，我们选择了需要实现的功能，并分析了连接组学原理以进行高效的布线。尽管类比物理中固有的非理想性和噪声，我将提供忆阻器底板上本地学习的适应性硬件证据，以及有助于解决归因问题和在可扩展架构中高效路由的新型材料堆栈和电路模块。 

---
# Dual-Space Augmented Intrinsic-LoRA for Wind Turbine Segmentation 

**Title (ZH)**: 双重空间增强内在LoRA方法在风力发电机分割中的应用 

**Authors**: Shubh Singhal, Raül Pérez-Gonzalo, Andreas Espersen, Antonio Agudo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20838)  

**Abstract**: Accurate segmentation of wind turbine blade (WTB) images is critical for effective assessments, as it directly influences the performance of automated damage detection systems. Despite advancements in large universal vision models, these models often underperform in domain-specific tasks like WTB segmentation. To address this, we extend Intrinsic LoRA for image segmentation, and propose a novel dual-space augmentation strategy that integrates both image-level and latent-space augmentations. The image-space augmentation is achieved through linear interpolation between image pairs, while the latent-space augmentation is accomplished by introducing a noise-based latent probabilistic model. Our approach significantly boosts segmentation accuracy, surpassing current state-of-the-art methods in WTB image segmentation. 

**Abstract (ZH)**: 准确的风力发电机叶片（WTB）图像分割对于有效评估至关重要，因为它直接影响自动化损伤检测系统的性能。尽管在大规模通用视觉模型方面取得了进展，但这些模型在诸如WTB分割这类特定领域任务中往往表现出色。为了解决这一问题，我们扩展了Intrinsic LoRA在图像分割中的应用，并提出了一种新颖的双空间增强策略，该策略结合了图像级和潜在空间的增强。图像空间增强通过图像对之间的线性插值实现，而潜在空间增强则是通过引入基于噪声的潜在概率模型实现。我们的方法显著提升了分割准确性，超越了当前WTB图像分割的领先方法。 

---
# Disentangling Preference Representation and Text Generation for Efficient Individual Preference Alignment 

**Title (ZH)**: 拆分偏好表示与文本生成以实现高效个体偏好对齐 

**Authors**: Jianfei Zhang, Jun Bai, Bei Li, Yanmeng Wang, Rumei Li, Chenghua Lin, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2412.20834)  

**Abstract**: Aligning Large Language Models (LLMs) with general human preferences has been proved crucial in improving the interaction quality between LLMs and human. However, human values are inherently diverse among different individuals, making it insufficient to align LLMs solely with general preferences. To address this, personalizing LLMs according to individual feedback emerges as a promising solution. Nonetheless, this approach presents challenges in terms of the efficiency of alignment algorithms. In this work, we introduce a flexible paradigm for individual preference alignment. Our method fundamentally improves efficiency by disentangling preference representation from text generation in LLMs. We validate our approach across multiple text generation tasks and demonstrate that it can produce aligned quality as well as or better than PEFT-based methods, while reducing additional training time for each new individual preference by $80\%$ to $90\%$ in comparison with them. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与一般人类偏好对齐已被证明对于提高LLMs与人类之间的交互质量至关重要。然而，人类价值观在不同个体之间固然是多样化的，仅仅将LLMs与一般偏好对齐是不够的。为了解决这一问题，根据个体反馈个性化调整LLMs被认为是一种有前途的解决方案。然而，这种方法在对齐算法的效率方面也面临着挑战。在本研究中，我们引入了一种灵活的个性化偏好对齐的范式。我们的方法通过分离偏好表示与文本生成，从根本上提高了对齐的效率。我们在多个文本生成任务中验证了该方法的有效性，并展示了它能够产生与基于PEFT的方法相当或更好的对齐质量，同时相较于基于PEFT的方法，可以将每个新个体偏好的额外训练时间减少80%到90%。 

---
# Fine-Tuning TransMorph with Gradient Correlation for Anatomical Alignment 

**Title (ZH)**: 使用梯度相关性微调TransMorph以实现解剖对齐 

**Authors**: Lukas Förner, Kartikay Tehlan, Thomas Wendler  

**Link**: [PDF](https://arxiv.org/pdf/2412.20822)  

**Abstract**: Unsupervised deep learning is a promising method in brain MRI registration to reduce the reliance on anatomical labels, while still achieving anatomically accurate transformations. For the Learn2Reg2024 LUMIR challenge, we propose fine-tuning of the pre-trained TransMorph model to improve the convergence stability as well as the deformation smoothness. The former is achieved through the FAdam optimizer, and consistency in structural changes is incorporated through the addition of gradient correlation in the similarity measure, improving anatomical alignment. The results show slight improvements in the Dice and HdDist95 scores, and a notable reduction in the NDV compared to the baseline TransMorph model. These are also confirmed by inspecting the boundaries of the tissue. Our proposed method highlights the effectiveness of including Gradient Correlation to achieve smoother and structurally consistent deformations for interpatient brain MRI registration. 

**Abstract (ZH)**: 无监督深度学习是脑MRI配准中减少对解剖标签依赖的一种有前景的方法，同时仍能实现解剖学上的精确变换。对于参加Learn2Reg2024 LUMIR挑战，我们提出对预训练的TransMorph模型进行微调，以提高收敛稳定性和变形平滑性。通过使用FAdam优化器实现前者的改进，并通过在相似性度量中添加梯度相关性来确保结构变化的一致性，从而提高解剖学对齐。结果显示，与基线TransMorph模型相比，Dice和HdDist95评分略有提高，且NDV显著降低。这些结果也通过检查组织边界得到了验证。我们提出的方法强调了梯度相关性在实现更平滑且结构一致变形方面的有效性，特别是在不同患者脑MRI配准中。 

---
# Length-Aware DETR for Robust Moment Retrieval 

**Title (ZH)**: 基于长度感知的DETR模型以提高稳健的时刻检索性能 

**Authors**: Seojeong Park, Jiho Choi, Kyungjune Baek, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2412.20816)  

**Abstract**: Video Moment Retrieval (MR) aims to localize moments within a video based on a given natural language query. Given the prevalent use of platforms like YouTube for information retrieval, the demand for MR techniques is significantly growing. Recent DETR-based models have made notable advances in performance but still struggle with accurately localizing short moments. Through data analysis, we identified limited feature diversity in short moments, which motivated the development of MomentMix. MomentMix employs two augmentation strategies: ForegroundMix and BackgroundMix, each enhancing the feature representations of the foreground and background, respectively. Additionally, our analysis of prediction bias revealed that short moments particularly struggle with accurately predicting their center positions of moments. To address this, we propose a Length-Aware Decoder, which conditions length through a novel bipartite matching process. Our extensive studies demonstrate the efficacy of our length-aware approach, especially in localizing short moments, leading to improved overall performance. Our method surpasses state-of-the-art DETR-based methods on benchmark datasets, achieving the highest R1 and mAP on QVHighlights and the highest R1@0.7 on TACoS and Charades-STA (such as a 2.46% gain in R1@0.7 and a 2.57% gain in mAP average for QVHighlights). The code is available at this https URL. 

**Abstract (ZH)**: 视频片段检索（MR）旨在根据给定的自然语言查询在视频中定位片段。由于像YouTube这样的平台广泛用于信息检索，对MR技术的需求正在显著增长。基于DETR的最近模型在性能上取得了显著进展，但仍难以准确定位短片段。通过数据分析，我们发现短片段中特征多样性有限，这促使我们开发了MomentMix。MomentMix采用了两种增强策略：ForegroundMix和BackgroundMix，分别增强前景和背景的特征表示。此外，我们对预测偏差的分析发现，短片段特别难以准确预测片段的中心位置。为此，我们提出了一种长度感知解码器，通过新颖的二分匹配过程对长度进行条件控制。我们的大量研究表明，我们的长度感知方法尤其在定位短片段方面效果显著，从而提高了整体性能。我们的方法在基准数据集上超过了最先进的基于DETR的方法，在QVHighlights上实现了最高的R1和mAP，在TACoS和Charades-STA上实现了最高的R1@0.7（例如，在QVHighlights上R1@0.7提高了2.46%，mAP平均提高了2.57%）。代码可在以下链接获得：this https URL。 

---
# Two Heads Are Better Than One: Averaging along Fine-Tuning to Improve Targeted Transferability 

**Title (ZH)**: 两个头胜过一个：沿微调过程进行平均以提高目标域转移性 

**Authors**: Hui Zeng, Sanshuai Cui, Biwei Chen, Anjie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2412.20807)  

**Abstract**: With much longer optimization time than that of untargeted attacks notwithstanding, the transferability of targeted attacks is still far from satisfactory. Recent studies reveal that fine-tuning an existing adversarial example (AE) in feature space can efficiently boost its targeted transferability. However, existing fine-tuning schemes only utilize the endpoint and ignore the valuable information in the fine-tuning trajectory. Noting that the vanilla fine-tuning trajectory tends to oscillate around the periphery of a flat region of the loss surface, we propose averaging over the fine-tuning trajectory to pull the crafted AE towards a more centered region. We compare the proposed method with existing fine-tuning schemes by integrating them with state-of-the-art targeted attacks in various attacking scenarios. Experimental results uphold the superiority of the proposed method in boosting targeted transferability. The code is available at this http URL. 

**Abstract (ZH)**: 尽管目标攻击的优化时间远长于非目标攻击，但目标攻击的迁移性仍然不尽如人意。最近的研究表明，在特征空间中对现有的对抗样本（AE）进行微调可以有效提升其目标迁移性。然而，现有的微调方案仅利用了微调的终端状态，而忽视了微调轨迹中的宝贵信息。注意到 vanilla 微调轨迹倾向于在损失曲面平坦区域的边缘附近振荡，我们提出通过平均微调轨迹来将构建的 AE 向损失曲面中心区域拉动。我们将所提出的方法与现有的微调方案结合，应用于各种攻击场景中的先进目标攻击方法进行对比。实验结果证明了所提出方法在提升目标迁移性方面的优越性。代码托管在此 <https://> 地址。 

---
# A Tale of Two Imperatives: Privacy and Explainability 

**Title (ZH)**: 两者的叙事：隐私与可解释性 

**Authors**: Supriya Manna, Niladri Sett  

**Link**: [PDF](https://arxiv.org/pdf/2412.20798)  

**Abstract**: Deep learning's preponderance across scientific domains has reshaped high-stakes decision-making, making it essential to follow rigorous operational frameworks that include both Right-to-Privacy (RTP) and Right-to-Explanation (RTE). This paper examines the complexities of combining these two requirements. For RTP, we focus on 'Differentially privacy' (DP), which is considered the current gold standard for privacy-preserving machine learning due to its strong quantitative guarantee of privacy. For RTE, we focus on post-hoc explainers: they are the go-to option for model auditing as they operate independently of model training. We formally investigate (DP) models and various commonly-used post-hoc explainers: how to evaluate these explainers subject to RTP, and analyze the intrinsic interactions between DP models and these explainers. Furthermore, our work throws light on how RTP and RTE can be effectively combined in high-stakes applications. Our study concludes by outlining an industrial software pipeline, with the example of a wildly used use-case, that respects both RTP and RTE requirements. 

**Abstract (ZH)**: 深度学习在各个科学领域中的广泛应用已经重塑了高风险决策过程，因此遵循严格的运营框架变得至关重要，这些框架既包括隐私权（Right-to-Privacy, RTP），也包括解释权（Right-to-Explanation, RTE）。本文探讨了将这两种要求结合起来的复杂性。对于RTP，我们重点关注“差分隐私”（Differentially Privacy, DP），因为它因其强大的隐私保护定量保障而被认为是当前私有化机器学习的标准。对于RTE，我们重点关注事后解释器：这些解释器是模型审计的首选方案，因为它们独立于模型训练过程。我们形式化地研究了DP模型以及各类常用的事后解释器：如何在RTP的约束下评估这些解释器，以及分析DP模型与这些解释器之间的内在交互。此外，我们的研究揭示了如何在高风险应用中有效结合RTP和RTE。我们的研究结论中概述了一个符合RTP和RTE要求的工业软件流水线，并以一个广泛应用的示例进行了说明。 

---
# Frequency-Masked Embedding Inference: A Non-Contrastive Approach for Time Series Representation Learning 

**Title (ZH)**: 频率掩蔽嵌入推理：一种用于时间序列表示学习的非对比度方法 

**Authors**: En Fu, Yanyan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20790)  

**Abstract**: Contrastive learning underpins most current self-supervised time series representation methods. The strategy for constructing positive and negative sample pairs significantly affects the final representation quality. However, due to the continuous nature of time series semantics, the modeling approach of contrastive learning struggles to accommodate the characteristics of time series data. This results in issues such as difficulties in constructing hard negative samples and the potential introduction of inappropriate biases during positive sample construction. Although some recent works have developed several scientific strategies for constructing positive and negative sample pairs with improved effectiveness, they remain constrained by the contrastive learning framework. To fundamentally overcome the limitations of contrastive learning, this paper introduces Frequency-masked Embedding Inference (FEI), a novel non-contrastive method that completely eliminates the need for positive and negative samples. The proposed FEI constructs 2 inference branches based on a prompting strategy: 1) Using frequency masking as prompts to infer the embedding representation of the target series with missing frequency bands in the embedding space, and 2) Using the target series as prompts to infer its frequency masking embedding. In this way, FEI enables continuous semantic relationship modeling for time series. Experiments on 8 widely used time series datasets for classification and regression tasks, using linear evaluation and end-to-end fine-tuning, show that FEI significantly outperforms existing contrastive-based methods in terms of generalization. This study provides new insights into self-supervised representation learning for time series. The code is available at this https URL. 

**Abstract (ZH)**: 对比学习是当前大多数自监督时间序列表示方法的基石。构造正样本和负样本对的策略对最终表示质量影响重大。然而，由于时间序列语义的连续性，对比学习的建模方法难以适应时间序列数据的特点。这导致了构建硬负样本困难以及正样本构建过程中可能引入不适当的偏差等问题。尽管一些最新的研究工作开发了多种科学策略来提高正负样本对构建的有效性，但它们仍然受到对比学习框架的限制。为了从根本上克服对比学习的局限性，本文引入了频率掩蔽嵌入推理（FEI），这是一种全新的非对比学习方法，完全消除了对正负样本的需求。所提出的FEI基于提示策略构建了2个推理分支：1) 使用频率掩蔽作为提示以推断目标系列在嵌入空间中缺失频率带的嵌入表示；2) 使用目标系列作为提示以推断其频率掩蔽嵌入。通过这种方式，FEI能够对时间序列进行连续语义关系建模。实验结果显示，在8个广泛使用的分类和回归任务时间序列数据集上，使用线性评测和端到端微调，FEI显著优于现有的基于对比学习的方法，在泛化性能上表现更佳。本文为时间序列的自监督表示学习提供了新的见解。代码可在以下链接获取：this https URL。 

---
# SecBench: A Comprehensive Multi-Dimensional Benchmarking Dataset for LLMs in Cybersecurity 

**Title (ZH)**: SecBench：全面的多维度大型语言模型基准数据集（在网络安全领域的应用） 

**Authors**: Pengfei Jing, Mengyun Tang, Xiaorong Shi, Xing Zheng, Sen Nie, Shi Wu, Yong Yang, Xiapu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20787)  

**Abstract**: Evaluating Large Language Models (LLMs) is crucial for understanding their capabilities and limitations across various applications, including natural language processing and code generation. Existing benchmarks like MMLU, C-Eval, and HumanEval assess general LLM performance but lack focus on specific expert domains such as cybersecurity. Previous attempts to create cybersecurity datasets have faced limitations, including insufficient data volume and a reliance on multiple-choice questions (MCQs). To address these gaps, we propose SecBench, a multi-dimensional benchmarking dataset designed to evaluate LLMs in the cybersecurity domain. SecBench includes questions in various formats (MCQs and short-answer questions (SAQs)), at different capability levels (Knowledge Retention and Logical Reasoning), in multiple languages (Chinese and English), and across various sub-domains. The dataset was constructed by collecting high-quality data from open sources and organizing a Cybersecurity Question Design Contest, resulting in 44,823 MCQs and 3,087 SAQs. Particularly, we used the powerful while cost-effective LLMs to (1). label the data and (2). constructing a grading agent for automatic evaluation of this http URL results on 13 SOTA LLMs demonstrate the usability of SecBench, which is arguably the largest and most comprehensive benchmark dataset for LLMs in cybersecurity. More information about SecBench can be found at our website, and the dataset can be accessed via the artifact link. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）对于理解其在各种应用中的能力和局限性至关重要，包括自然语言处理和代码生成。现有的基准测试如MMLU、C-Eval和HumanEval评估了LLM的一般性能，但缺乏对特定专家领域（如网络安全）的关注。此前尝试创建网络安全数据集的努力遇到了一些局限性，包括数据量不足以及依赖多项选择题（MCQs）的形式。为了弥补这些不足，我们提出了SecBench，这是一个多维度的基准测试数据集，旨在评估网络安全领域的LLM。SecBench包括多种形式的问题（多项选择题和简答题）、不同能力水平（知识保留和逻辑推理）、多语言（中文和英文）以及多个子领域的题目。

数据集的构建主要通过收集开放源中的高质量数据，并组织了一场网络安全问题设计竞赛，最终生成了44,823个多项选择题和3,087个简答题。特别地，我们使用了强大而成本效益高的LLM来（1）标注数据，以及（2）构建一个自动评分代理，用于评估这些结果。13种最先进的LLM的结果表明，SecBench可能是迄今为止规模最大、涵盖最全面的网络安全领域LLM基准数据集之一。更多关于SecBench的信息可以在我们的网站上找到，并可通过数据脚本链接获取数据集。

请注意，在翻译过程中，为了符合中文的学术规范和表达习惯，对一些句子的结构进行了调整和优化，同时确保了信息的准确性与完整性。 

---
# Sample Correlation for Fingerprinting Deep Face Recognition 

**Title (ZH)**: 标题翻译为：基于样本相关性的指纹识别深度面部识别方法

如果需要更详细的论文内容翻译，请提供具体内容段落。 

**Authors**: Jiyang Guan, Jian Liang, Yanbo Wang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2412.20768)  

**Abstract**: Face recognition has witnessed remarkable advancements in recent years, thanks to the development of deep learning this http URL, an off-the-shelf face recognition model as a commercial service could be stolen by model stealing attacks, posing great threats to the rights of the model this http URL fingerprinting, as a model stealing detection method, aims to verify whether a suspect model is stolen from the victim model, gaining more and more attention this http URL methods always utilize transferable adversarial examples as the model fingerprint, but this method is known to be sensitive to adversarial defense and transfer learning this http URL address this issue, we consider the pairwise relationship between samples instead and propose a novel yet simple model stealing detection method based on SAmple Correlation (SAC).Specifically, we present SAC-JC that selects JPEG compressed samples as model inputs and calculates the correlation matrix among their model this http URL results validate that SAC successfully defends against various model stealing attacks in deep face recognition, encompassing face verification and face emotion recognition, exhibiting the highest performance in terms of AUC, p-value and F1 this http URL, we extend our evaluation of SAC-JC to object recognition datasets including Tiny-ImageNet and CIFAR10, which also demonstrates the superior performance of SAC-JC to previous this http URL code will be available at \url{this https URL}. 

**Abstract (ZH)**: 近年来，受深度学习的发展推动，人脸识别取得了显著的进步。然而，商用的人脸识别模型可能因模型窃取攻击而被盗取，这对模型的所有权构成了巨大威胁。为此，模型窃取检测方法（如指纹分析）受到了越来越多的关注。当前的方法通常利用可转移的对抗样本作为模型指纹，但这种方法已知对对抗防御和迁移学习比较敏感。为了解决这一问题，我们考虑了样本之间的成对关系，并基于样本相关性（SAmple Correlation, SAC）提出了一种新颖且简单的模型窃取检测方法。具体来说，我们提出了SAC-JC，该方法选择JPEG压缩样本作为模型输入，并计算它们模型输出之间的相关矩阵。实验结果表明，SAC成功地抵御了各种模型窃取攻击，涵盖面部认证和面部表情识别，在AUC、p值和F1方面表现出最佳性能。此外，我们还将SAC-JC的评估扩展到了目标识别数据集，包括Tiny-ImageNet和CIFAR10，这也证明了SAC-JC在这些数据集上的优越性能。相关代码将在\url{https://github.com/username/SAC-JC}处提供。 

---
# KeyGS: A Keyframe-Centric Gaussian Splatting Method for Monocular Image Sequences 

**Title (ZH)**: KeyGS：一种以关键帧为中心的高斯渲染方法用于单目图像序列 

**Authors**: Keng-Wei Chang, Zi-Ming Wang, Shang-Hong Lai  

**Link**: [PDF](https://arxiv.org/pdf/2412.20767)  

**Abstract**: Reconstructing high-quality 3D models from sparse 2D images has garnered significant attention in computer vision. Recently, 3D Gaussian Splatting (3DGS) has gained prominence due to its explicit representation with efficient training speed and real-time rendering capabilities. However, existing methods still heavily depend on accurate camera poses for reconstruction. Although some recent approaches attempt to train 3DGS models without the Structure-from-Motion (SfM) preprocessing from monocular video datasets, these methods suffer from prolonged training times, making them impractical for many applications.
In this paper, we present an efficient framework that operates without any depth or matching model. Our approach initially uses SfM to quickly obtain rough camera poses within seconds, and then refines these poses by leveraging the dense representation in 3DGS. This framework effectively addresses the issue of long training times. Additionally, we integrate the densification process with joint refinement and propose a coarse-to-fine frequency-aware densification to reconstruct different levels of details. This approach prevents camera pose estimation from being trapped in local minima or drifting due to high-frequency signals. Our method significantly reduces training time from hours to minutes while achieving more accurate novel view synthesis and camera pose estimation compared to previous methods. 

**Abstract (ZH)**: 从稀疏2D图像重建高质量3D模型在计算机视觉领域引起了广泛关注。最近，3D高斯点表示（3DGS）由于其高效的训练速度和实时渲染能力而备受青睐。然而，现有的方法仍然高度依赖准确的相机姿态进行重建。尽管一些最近的方法尝试在单目视频数据集上训练3DGS模型而不进行结构从运动（SfM）预处理，但这些方法仍然面临着训练时间过长的问题，这使得它们在许多应用中实用性较低。

在本文中，我们提出了一种无需深度或匹配模型的高效框架。我们的方法首先利用SfM在几秒钟内快速获得粗略的相机姿态，然后通过利用3DGS中的密集表示来进一步精化这些姿态。该框架有效地解决了长时间训练的问题。此外，我们将密集化过程与联合精化相结合，并提出了从粗到细的频率感知密集化方法，用于重建不同级别的细节。这种方法可以防止相机姿态估计陷入局部极小值或因高频信号漂移。我们的方法将训练时间从数小时缩短到几分钟，同时在新颖视图合成和相机姿态估计的准确性上优于先前的方法。 

---
# Attributing Culture-Conditioned Generations to Pretraining Corpora 

**Title (ZH)**: 将文化条件下的代际差异归因于预训练语料库 

**Authors**: Huihan Li, Arnav Goel, Keyu He, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.20760)  

**Abstract**: In open-ended generative tasks like narrative writing or dialogue, large language models often exhibit cultural biases, showing limited knowledge and generating templated outputs for less prevalent cultures. Recent works show that these biases may stem from uneven cultural representation in pretraining corpora. This work investigates how pretraining leads to biased culture-conditioned generations by analyzing how models associate entities with cultures based on pretraining data patterns. We propose the MEMOed framework (MEMOrization from pretraining document) to determine whether a generation for a culture arises from memorization. Using MEMOed on culture-conditioned generations about food and clothing for 110 cultures, we find that high-frequency cultures in pretraining data yield more generations with memorized symbols, while some low-frequency cultures produce none. Additionally, the model favors generating entities with extraordinarily high frequency regardless of the conditioned culture, reflecting biases toward frequent pretraining terms irrespective of relevance. We hope that the MEMOed framework and our insights will inspire more works on attributing model performance on pretraining data. 

**Abstract (ZH)**: 在像叙事写作或对话这类开放生成任务中，大型语言模型常常表现出文化偏见，缺乏对较少见文化的了解，并生成模板化的输出。最近的研究表明，这些偏见可能源于预训练数据集中文化代表性不均。本文通过分析模型根据预训练数据模式将实体与文化关联的方式，探讨预训练导致文化条件生成偏见的过程。我们提出了MEMOed框架（从预训练文档中记忆化）来判断一个文化的生成是否来源于记忆化。在对110种文化的食品和服装进行文化条件生成时，我们发现高频文化在预训练数据中的生成更多包含记忆化的符号，而一些低频文化则没有任何生成。此外，模型倾向于生成异常高频的实体，无论这些实体是否与条件文化相关，这表明模型对高频预训练术语存在偏见，而不考虑其相关性。我们希望通过MEMOed框架和我们的洞见激发更多研究，探索模型在预训练数据上的性能归因。 

---
# Solar Filaments Detection using Active Contours Without Edges 

**Title (ZH)**: 使用无边线活跃轮廓进行日珥检测 

**Authors**: Sanmoy Bandyopadhyay, Vaibhav Pant  

**Link**: [PDF](https://arxiv.org/pdf/2412.20749)  

**Abstract**: In this article, an active contours without edges (ACWE)-based algorithm has been proposed for the detection of solar filaments in H-alpha full-disk solar images. The overall algorithm consists of three main steps of image processing. These are image pre-processing, image segmentation, and image post-processing. Here in the work, contours are initialized on the solar image and allowed to deform based on the energy function. As soon as the contour reaches the boundary of the desired object, the energy function gets reduced, and the contour stops evolving. The proposed algorithm has been applied to few benchmark datasets and has been compared with the classical technique of object detection. The results analysis indicates that the proposed algorithm outperforms the results obtained using the existing classical algorithm of object detection. 

**Abstract (ZH)**: 在本文中，提出了一种基于无边界的主动轮廓（ACWE）算法，用于太阳H-α全盘图像中太阳日珥的检测。该算法总体上包括三个主要的图像处理步骤：图像预处理、图像分割和图像后处理。在这项研究中，初始化轮廓并基于能量函数允许其变形。一旦轮廓达到目标对象的边界，能量函数便会减少，轮廓停止演化。所提出的算法应用于多个基准数据集，并与经典的物体检测技术进行了比较。结果分析表明，所提出的算法在物体检测方面的表现优于现有经典算法的检测结果。 

---
# Advancing Parkinson's Disease Progression Prediction: Comparing Long Short-Term Memory Networks and Kolmogorov-Arnold Networks 

**Title (ZH)**: 提升帕金森病进展预测：长短时记忆网络与柯尔莫哥洛夫-阿诺德网络比较 

**Authors**: Abhinav Roy, Bhavesh Gyanchandani, Aditya Oza, Abhishek Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2412.20744)  

**Abstract**: Parkinson's Disease (PD) is a degenerative neurological disorder that impairs motor and non-motor functions, significantly reducing quality of life and increasing mortality risk. Early and accurate detection of PD progression is vital for effective management and improved patient outcomes. Current diagnostic methods, however, are often costly, time-consuming, and require specialized equipment and expertise. This work proposes an innovative approach to predicting PD progression using regression methods, Long Short-Term Memory (LSTM) networks, and Kolmogorov Arnold Networks (KAN). KAN, utilizing spline-parametrized univariate functions, allows for dynamic learning of activation patterns, unlike traditional linear models.
The Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS) is a comprehensive tool for evaluating PD symptoms and is commonly used to measure disease progression. Additionally, protein or peptide abnormalities are linked to PD onset and progression. Identifying these associations can aid in predicting disease progression and understanding molecular changes.
Comparing multiple models, including LSTM and KAN, this study aims to identify the method that delivers the highest metrics. The analysis reveals that KAN, with its dynamic learning capabilities, outperforms other approaches in predicting PD progression. This research highlights the potential of AI and machine learning in healthcare, paving the way for advanced computational models to enhance clinical predictions and improve patient care and treatment strategies in PD management. 

**Abstract (ZH)**: 帕金森病（PD）是一种进行性神经退行性疾病，会影响运动和非运动功能，显著降低生活质量并增加死亡风险。早期和准确地检测PD的进展对于有效的管理和改善患者预后至关重要。然而，现有的诊断方法往往成本高、耗时且需要专门的设备和专业知识。本研究提出了一种创新的方法，通过回归方法、长短期记忆（LSTM）网络和柯尔莫哥洛夫-阿诺尔德网络（KAN）来预测PD的进展。KAN 利用分片参数化的单变量函数，能够动态学习激活模式，与传统的线性模型相比具有优势。

《运动障碍协会赞助修订的统一帕金森病评定量表》（MDS-UPDRS）是一个全面评估PD症状的工具，常用于衡量疾病进展。此外，蛋白质或肽的异常与PD的发病和进展有关。识别这些关联有助于预测疾病进展并理解分子变化。

本研究通过比较包括LSTM 和KAN在内的多个模型，旨在确定能够提供最高指标的方法。分析结果显示，KAN 由于其动态学习能力，在预测PD进展方面优于其他方法。该研究突显了人工智能和机器学习在医疗保健领域的潜力，为进一步开发先进的计算模型以提高临床预测能力并改善PD管理中的患者护理和治疗策略铺平了道路。 

---
# Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study 

**Title (ZH)**: 面向全国范围的分析型医疗基础设施：一种隐私保护增强的膝关节康复案例研究 

**Authors**: Boris Bačić, Claudiu Vasile, Chengwei Feng, Marian G. Ciucă  

**Link**: [PDF](https://arxiv.org/pdf/2412.20733)  

**Abstract**: The purpose of this paper is to contribute towards the near-future privacy-preserving big data analytical healthcare platforms, capable of processing streamed or uploaded timeseries data or videos from patients. The experimental work includes a real-life knee rehabilitation video dataset capturing a set of exercises from simple and personalised to more general and challenging movements aimed for returning to sport. To convert video from mobile into privacy-preserving diagnostic timeseries data, we employed Google MediaPipe pose estimation. The developed proof-of-concept algorithms can augment knee exercise videos by overlaying the patient with stick figure elements while updating generated timeseries plot with knee angle estimation streamed as CSV file format. For patients and physiotherapists, video with side-to-side timeseries visually indicating potential issues such as excessive knee flexion or unstable knee movements or stick figure overlay errors is possible by setting a-priori knee-angle parameters. To address adherence to rehabilitation programme and quantify exercise sets and repetitions, our adaptive algorithm can correctly identify (91.67%-100%) of all exercises from side- and front-view videos. Transparent algorithm design for adaptive visual analysis of various knee exercise patterns contributes towards the interpretable AI and will inform near-future privacy-preserving, non-vendor locking, open-source developments for both end-user computing devices and as on-premises non-proprietary cloud platforms that can be deployed within the national healthcare system. 

**Abstract (ZH)**: 本文旨在为即将出现的隐私保护的大数据分析医疗平台做出贡献，这些平台能够处理来自患者的流式传输或上传的时序数据或视频。实验工作包括一个现实生活中的膝关节康复视频数据集，记录了一系列从简单个性化到更具普遍性和挑战性的运动，目的是重返运动。为了将视频转化为隐私保护的诊断时序数据，我们采用了Google MediaPipe姿态估计技术。我们开发的概念验证算法可在视频上叠加人体模型元素，并将通过CSV文件格式流传输的膝关节角度估计值更新生成的时序图。对于患者和物理治疗师而言，可以通过设置先验的膝关节角度参数，将视频转换为侧面时序图，从而显示潜在问题，如膝关节过度弯曲或不稳定运动，或人体模型叠加错误。为了解决康复计划的依从性并量化运动组和重复次数，我们的自适应算法可以从侧面和正面视频中正确识别（91.67%-100%）所有运动。透明的算法设计有助于可解释的人工智能，并将促进未来隐私保护、非供应商锁定、开源的发展，既适用于终端用户的计算设备，也适用于内部非专有的云平台，可以在国家医疗体系内部署。 

---
# M$^3$oralBench: A MultiModal Moral Benchmark for LVLMs 

**Title (ZH)**: M$^3$oralBench: 一种面向多模态语言模型的道德基准测试 

**Authors**: Bei Yan, Jie Zhang, Zhiyuan Chen, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20718)  

**Abstract**: Recently, large foundation models, including large language models (LLMs) and large vision-language models (LVLMs), have become essential tools in critical fields such as law, finance, and healthcare. As these models increasingly integrate into our daily life, it is necessary to conduct moral evaluation to ensure that their outputs align with human values and remain within moral boundaries. Previous works primarily focus on LLMs, proposing moral datasets and benchmarks limited to text modality. However, given the rapid development of LVLMs, there is still a lack of multimodal moral evaluation methods. To bridge this gap, we introduce M$^3$oralBench, the first MultiModal Moral Benchmark for LVLMs. M$^3$oralBench expands the everyday moral scenarios in Moral Foundations Vignettes (MFVs) and employs the text-to-image diffusion model, SD3.0, to create corresponding scenario images. It conducts moral evaluation across six moral foundations of Moral Foundations Theory (MFT) and encompasses tasks in moral judgement, moral classification, and moral response, providing a comprehensive assessment of model performance in multimodal moral understanding and reasoning. Extensive experiments on 10 popular open-source and closed-source LVLMs demonstrate that M$^3$oralBench is a challenging benchmark, exposing notable moral limitations in current models. Our benchmark is publicly available. 

**Abstract (ZH)**: 近年来，大型基础模型，包括大型语言模型（LLMs）和大型视觉-语言模型（LVLMs），在法律、金融和医疗等关键领域已成为不可或缺的工具。随着这些模型越来越多地融入我们的日常生活，进行道德评估变得必要，以确保其输出与人类价值观相符，并保持在道德边界之内。以往的工作主要关注LLMs，提出了限于文本模态的道德数据集和基准测试。然而，鉴于LVLMs的快速发展，仍然缺乏多模态道德评估方法。为了弥补这一差距，我们引入了M$^3$oralBench，这是首个针对LVLMs的多模态道德基准测试。M$^3$oralBench 扩展了《道德基础情景》（MFVs）中的日常生活道德场景，并利用文本到图像扩散模型SD3.0创建相应的场景图像。该基准测试涵盖了《道德基础理论》（MFT）中的六种道德基础，并包含道德判断、道德分类和道德回应任务，提供了模型在多模态道德理解和推理方面的全面评估。针对10种流行开源和闭源LVLMs进行的广泛实验表明，M$^3$oralBench 是一个具有挑战性的基准测试，揭示了当前模型在道德方面的显著局限性。我们的基准测试已公开可用。 

---
# UBER: Uncertainty-Based Evolution with Large Language Models for Automatic Heuristic Design 

**Title (ZH)**: UBER：基于不确定性的大语言模型自动启发式设计演化方法 

**Authors**: Zijie Chen, Zhanchao Zhou, Yu Lu, Renjun Xu, Lili Pan, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20694)  

**Abstract**: NP-hard problem-solving traditionally relies on heuristics, but manually crafting effective heuristics for complex problems remains challenging. While recent work like FunSearch has demonstrated that large language models (LLMs) can be leveraged for heuristic design in evolutionary algorithm (EA) frameworks, their potential is not fully realized due to its deficiency in exploitation and exploration. We present UBER (Uncertainty-Based Evolution for Refinement), a method that enhances LLM+EA methods for automatic heuristic design by integrating uncertainty on top of the FunSearch framework. UBER introduces two key innovations: an Uncertainty-Inclusive Evolution Process (UIEP) for adaptive exploration-exploitation balance, and a principled Uncertainty-Inclusive Island Reset (UIIS) strategy for maintaining population diversity. Through extensive experiments on challenging NP-complete problems, UBER demonstrates significant improvements over FunSearch. Our work provides a new direction for the synergy of LLMs and EA, advancing the field of automatic heuristic design. 

**Abstract (ZH)**: 传统的NP-hard问题求解依赖于启发式方法，但为复杂问题手动设计有效的启发式方法仍然具有挑战性。虽然最近的研究，如FunSearch，已经证明大规模语言模型（LLMs）可以在进化算法（EAs）框架中用于启发式设计，但它们的潜力并未完全发挥，因为存在探索和利用能力的不足。我们提出了UBER（基于不确定性 refinement的进化算法），这是一种通过在FunSearch框架上集成不确定性来增强LLM+EA方法以实现自动启发式设计的方法。UBER引入了两项关键创新：一种包含不确定性的进化过程（UIEP），以实现自适应的探索和利用平衡，以及一种原则性的包含不确定性的岛群重启策略（UIIS），以维持种群多样性。通过在NP完全问题上的广泛实验，UBER在FunSearch的基础上取得了显著的进步。我们的工作为LLMs和EAs的协同作用提供了新的方向，推动了自动启发式设计领域的发展。 

---
# Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner 

**Title (ZH)**: 使用视觉LLM增强表格识别：一个基准及基于邻居指导的工具链推理器 

**Authors**: Yitong Zhou, Mingyue Cheng, Qingyang Mao, Qi Liu, Feiyang Xu, Xin Li, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20662)  

**Abstract**: Pre-trained foundation models have recently significantly progressed in structured table understanding and reasoning. However, despite advancements in areas such as table semantic understanding and table question answering, recognizing the structure and content of unstructured tables using Vision Large Language Models (VLLMs) remains under-explored. In this work, we address this research gap by employing VLLMs in a training-free reasoning paradigm. First, we design a benchmark with various hierarchical dimensions relevant to table recognition. Subsequently, we conduct in-depth evaluations using pre-trained VLLMs, finding that low-quality image input is a significant bottleneck in the recognition process. Drawing inspiration from these findings, we propose the Neighbor-Guided Toolchain Reasoner (NGTR) framework, which is characterized by integrating multiple lightweight models for low-level visual processing operations aimed at mitigating issues with low-quality input images. Specifically, we utilize a neighbor retrieval mechanism to guide the generation of multiple tool invocation plans, transferring tool selection experiences from similar neighbors to the given input, thereby facilitating suitable tool selection. Additionally, we introduce a reflection module to supervise the tool invocation process. Extensive experiments on public table recognition datasets demonstrate that our approach significantly enhances the recognition capabilities of the vanilla VLLMs. We believe that the designed benchmark and the proposed NGTR framework could provide an alternative solution in table recognition. 

**Abstract (ZH)**: 预训练基础模型最近在结构化表格理解和推理方面取得了显著进展。然而，尽管在表格语义理解和表格问答等领域的研究取得了进展，使用视觉大规模语言模型（VLLMs）识别非结构化表格的结构和内容依然缺乏探索。在本文中，我们通过在无训练推理范式中应用VLLMs来填补这一研究空白。首先，我们设计了一个包含多种与表格识别相关的层次维度的基准。随后，我们使用预训练的VLLMs进行了详细的评估，发现低质量图像输入是识别过程中的主要瓶颈。受到这些发现的启发，我们提出了一种名为Neighbor-Guided Toolchain Reasoner（NGTR）的框架，该框架通过集成多种轻量级模型，以低级视觉处理操作来减轻低质量输入图像的问题。具体而言，我们利用邻居检索机制来指导多种工具调用计划的生成，通过将相似邻居的工具选择经验转移到给定输入中，从而促进合适的工具选择。此外，我们引入了一个反馈模块来监督工具调用过程。在公共表格识别数据集上的大量实验表明，我们的方法显著提升了基础VLLMs的识别能力。我们相信，设计的基准和提出的NGTR框架可以在表格识别中提供一种替代解决方案。 

---
# Overcoming Class Imbalance: Unified GNN Learning with Structural and Semantic Connectivity Representations 

**Title (ZH)**: 克服类别不平衡：结构和语义连接表示的统一GNN学习 

**Authors**: Abdullah Alchihabi, Hao Yan, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20656)  

**Abstract**: Class imbalance is pervasive in real-world graph datasets, where the majority of annotated nodes belong to a small set of classes (majority classes), leaving many other classes (minority classes) with only a handful of labeled nodes. Graph Neural Networks (GNNs) suffer from significant performance degradation in the presence of class imbalance, exhibiting bias towards majority classes and struggling to generalize effectively on minority classes. This limitation stems, in part, from the message passing process, leading GNNs to overfit to the limited neighborhood of annotated nodes from minority classes and impeding the propagation of discriminative information throughout the entire graph. In this paper, we introduce a novel Unified Graph Neural Network Learning (Uni-GNN) framework to tackle class-imbalanced node classification. The proposed framework seamlessly integrates both structural and semantic connectivity representations through semantic and structural node encoders. By combining these connectivity types, Uni-GNN extends the propagation of node embeddings beyond immediate neighbors, encompassing non-adjacent structural nodes and semantically similar nodes, enabling efficient diffusion of discriminative information throughout the graph. Moreover, to harness the potential of unlabeled nodes within the graph, we employ a balanced pseudo-label generation mechanism that augments the pool of available labeled nodes from minority classes in the training set. Experimental results underscore the superior performance of our proposed Uni-GNN framework compared to state-of-the-art class-imbalanced graph learning baselines across multiple benchmark datasets. 

**Abstract (ZH)**: 真实世界中的图数据集普遍存在类别不平衡的问题，其中绝大多数标注节点属于少数几个类（多数类），而许多其他类（少数类）仅有少量标注节点。图神经网络（GNNs）在类别不平衡的情况下表现出显著的性能下降，偏向于多数类，并且在推广到少数类时效果欠佳。这一限制部分源于消息传递过程，导致GNNs在少数类标注节点的有限邻域上过度拟合，阻碍了区分性信息在整个图中的传播。

本文中，我们提出了一种新的统一图神经网络学习（Uni-GNN）框架，以应对类别不平衡的节点分类问题。所提出的框架通过语义节点编码器和结构节点编码器无缝地整合了语义和结构连接表示。通过结合这些连接类型，Uni-GNN能够将节点嵌入的传播范围延伸至非相邻的结构节点和语义相似的节点，从而在整个图中有效扩散区分性信息。

此外，为了利用图中未标注节点的潜力，我们采用了一种平衡的伪标签生成机制，在训练集中小规模扩充少数类的可用标注节点池。实验结果表明，与当前最先进的类别不平衡图学习基线相比，我们的Uni-GNN框架在多个基准数据集上表现出更优的性能。 

---
# Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis 

**Title (ZH)**: 扩散模型中用于反事实医疗图像合成的潜在漂移现象 

**Authors**: Yousef Yeganeh, Ioannis Charisiadis, Marta Hasny, Martin Hartenberger, Björn Ommer, Nassir Navab, Azade Farshad, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2412.20651)  

**Abstract**: Scaling by training on large datasets has been shown to enhance the quality and fidelity of image generation and manipulation with diffusion models; however, such large datasets are not always accessible in medical imaging due to cost and privacy issues, which contradicts one of the main applications of such models to produce synthetic samples where real data is scarce. Also, finetuning on pre-trained general models has been a challenge due to the distribution shift between the medical domain and the pre-trained models. Here, we propose Latent Drift (LD) for diffusion models that can be adopted for any fine-tuning method to mitigate the issues faced by the distribution shift or employed in inference time as a condition. Latent Drifting enables diffusion models to be conditioned for medical images fitted for the complex task of counterfactual image generation, which is crucial to investigate how parameters such as gender, age, and adding or removing diseases in a patient would alter the medical images. We evaluate our method on three public longitudinal benchmark datasets of brain MRI and chest X-rays for counterfactual image generation. Our results demonstrate significant performance gains in various scenarios when combined with different fine-tuning schemes. The source code of this work will be publicly released upon its acceptance. 

**Abstract (ZH)**: 通过在大规模数据集上进行训练，已经证明扩散模型在图像生成和修饰方面可以提高质量和真实性；然而，在医学成像中由于成本和隐私问题，这样的大规模数据集并不总是可获得的，这与这种模型的主要应用之一——在真实数据稀缺的情况下生成合成样本——相矛盾。此外，从预训练的一般模型进行微调也受到医学领域与预训练模型之间的分布转移的挑战。在这里，我们提出了一种适用于任何微调方法的Latent Drift (LD) 方法，可以缓解分布转移所面临的问题，或者在推理阶段作为条件进行应用。Latent Drifting 允许扩散模型在医学成像中进行条件化，以适应生成反事实图像这一复杂任务，这对于研究性别、年龄以及增加或移除患者疾病如何改变医学图像至关重要。我们使用三个公开的纵向基准数据集（脑MRI和胸部X光片）来评估我们的方法在反事实图像生成方面的性能。结果表明，结合不同的微调方案时，我们的方法在各种场景中都能实现显著的性能提升。该工作的源代码将在接受后公开发布。 

---
# NetFlowGen: Leveraging Generative Pre-training for Network Traffic Dynamics 

**Title (ZH)**: NetFlowGen：利用生成预训练技术分析网络流量动态

这个标题的翻译尽量保持了原文的意思和学术性，同时确保了中文表达的流畅和准确性。如果需要进一步的具体背景信息或其他方面的帮助，请随时告知。 

**Authors**: Jiawei Zhou, Woojeong Kim, Zhiying Xu, Alexander M. Rush, Minlan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20635)  

**Abstract**: Understanding the traffic dynamics in networks is a core capability for automated systems to monitor and analyze networking behaviors, reducing expensive human efforts and economic risks through tasks such as traffic classification, congestion prediction, and attack detection. However, it is still challenging to accurately model network traffic with machine learning approaches in an efficient and broadly applicable manner. Task-specific models trained from scratch are used for different networking applications, which limits the efficiency of model development and generalization of model deployment. Furthermore, while networking data is abundant, high-quality task-specific labels are often insufficient for training individual models. Large-scale self-supervised learning on unlabeled data provides a natural pathway for tackling these challenges. We propose to pre-train a general-purpose machine learning model to capture traffic dynamics with only traffic data from NetFlow records, with the goal of fine-tuning for different downstream tasks with small amount of labels. Our presented NetFlowGen framework goes beyond a proof-of-concept for network traffic pre-training and addresses specific challenges such as unifying network feature representations, learning from large unlabeled traffic data volume, and testing on real downstream tasks in DDoS attack detection. Experiments demonstrate promising results of our pre-training framework on capturing traffic dynamics and adapting to different networking tasks. 

**Abstract (ZH)**: 理解网络中的交通动态是自动系统监控和分析网络行为的核心能力，通过诸如流量分类、拥塞预测和攻击检测等任务，可以减少昂贵的人力投入和经济损失。然而，使用机器学习方法高效且广泛适用地建模网络流量仍然是一个具有挑战性的问题。针对不同网络应用，需要从头训练任务特定模型，这限制了模型开发的效率和模型部署的泛化能力。此外，尽管网络数据丰富，但高质量的任务特定标签往往不足以单独训练模型。大规模的无监督学习在未标记数据上提供了一条自然途径来应对这些挑战。我们提出预训练一个通用的机器学习模型，仅使用NetFlow记录中的流量数据来捕捉流量动态，然后通过少量标签对模型进行微调，以适应不同的下游任务。我们提出的NetFlowGen框架不仅超越了一个网络流量预训练的概念证明，还解决了统一网络特征表示、从大量未标记流量数据中学习以及在分布式拒绝服务（DDoS）攻击检测等真实下游任务上的测试等特定挑战。实验结果展示了我们的预训练框架在捕捉流量动态和适应不同网络任务方面的有希望的表现。 

---
# HALLUCINOGEN: A Benchmark for Evaluating Object Hallucination in Large Visual-Language Models 

**Title (ZH)**: 幻觉：评估大型视觉语言模型中对象幻觉的基准 

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2412.20622)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in performing complex multimodal tasks. However, they are still plagued by object hallucination: the misidentification or misclassification of objects present in images. To this end, we propose HALLUCINOGEN, a novel visual question answering (VQA) object hallucination attack benchmark that utilizes diverse contextual reasoning prompts to evaluate object hallucination in state-of-the-art LVLMs. We design a series of contextual reasoning hallucination prompts to evaluate LVLMs' ability to accurately identify objects in a target image while asking them to perform diverse visual-language tasks such as identifying, locating or performing visual reasoning around specific objects. Further, we extend our benchmark to high-stakes medical applications and introduce MED-HALLUCINOGEN, hallucination attacks tailored to the biomedical domain, and evaluate the hallucination performance of LVLMs on medical images, a critical area where precision is crucial. Finally, we conduct extensive evaluations of eight LVLMs and two hallucination mitigation strategies across multiple datasets to show that current generic and medical LVLMs remain susceptible to hallucination attacks. 

**Abstract (ZH)**: 大规模多模态语言视觉模型（Large Vision-Language Models, LVLMs）在执行复杂多模态任务方面展现了非凡的能力。然而，它们仍然受到物体幻觉的困扰：即将图像中存在的物体错误地识别或分类。为解决这一问题，我们提出了一种名为HALLUCINOGEN的新颖视觉问答（Visual Question Answering, VQA）物体幻觉攻击基准，该基准利用多样化的上下文推理提示来评估最先进的LVLMs中的物体幻觉情况。我们设计了一系列上下文推理幻觉提示，以评估LVLMs在执行诸如识别、定位或围绕特定物体进行视觉推理等多样视觉-语言任务时准确识别目标图像中物体的能力。此外，我们还将基准扩展到了高风险的医疗应用领域，并引入了MED-HALLUCINOGEN这一针对生物医学领域定制的幻觉攻击，以评估LVLMs在医疗图像中的幻觉性能，医疗领域对精确性要求极高。最后，我们在多个数据集中对八种LVLMs和两种幻觉缓解策略进行了详尽的评估，以证明当前通用和医疗专用的LVLMs仍然容易受到幻觉攻击的影响。 

---
# Towards Explaining Uncertainty Estimates in Point Cloud Registration 

**Title (ZH)**: 面向点云对齐中不确定性估计的解释研究 

**Authors**: Ziyuan Qin, Jongseok Lee, Rudolph Triebel  

**Link**: [PDF](https://arxiv.org/pdf/2412.20612)  

**Abstract**: Iterative Closest Point (ICP) is a commonly used algorithm to estimate transformation between two point clouds. The key idea of this work is to leverage recent advances in explainable AI for probabilistic ICP methods that provide uncertainty estimates. Concretely, we propose a method that can explain why a probabilistic ICP method produced a particular output. Our method is based on kernel SHAP (SHapley Additive exPlanations). With this, we assign an importance value to common sources of uncertainty in ICP such as sensor noise, occlusion, and ambiguous environments. The results of the experiment show that this explanation method can reasonably explain the uncertainty sources, providing a step towards robots that know when and why they failed in a human interpretable manner 

**Abstract (ZH)**: 迭代最近点（ICP）算法是一种常用的用于估计两个点云之间变换的方法。本文的关键思想是利用最近在可解释人工智能方面的进展，通过概率ICP方法来提供不确定性估计。具体而言，我们提出了一种方法，该方法能够解释为什么概率ICP方法产生了特定的输出。我们的方法基于核SHAP（SHapley Additive exPlanations）。通过这种方法，我们为ICP中的常见不确定性源（如传感器噪声、遮挡和复杂环境）分配了重要性值。实验结果表明，这种解释方法能够合理地解释不确定性来源，为以人类可解释的方式理解机器人何时以及为何失败迈出了一步。 

---
# MATEY: multiscale adaptive foundation models for spatiotemporal physical systems 

**Title (ZH)**: MATEY: 多尺度自适应基础模型在时空物理系统中的应用 

**Authors**: Pei Zhang, M. Paul Laiu, Matthew Norman, Doug Stefanski, John Gounley  

**Link**: [PDF](https://arxiv.org/pdf/2412.20601)  

**Abstract**: Accurate representation of the multiscale features in spatiotemporal physical systems using vision transformer (ViT) architectures requires extremely long, computationally prohibitive token sequences. To address this issue, we propose two adaptive tokenization schemes that dynamically adjust patch sizes based on local features: one ensures convergent behavior to uniform patch refinement, while the other offers better computational efficiency. Moreover, we present a set of spatiotemporal attention schemes, where the temporal or axial spatial dimensions are decoupled, and evaluate their computational and data efficiencies. We assess the performance of the proposed multiscale adaptive model, MATEY, in a sequence of experiments. The results show that adaptive tokenization schemes achieve improved accuracy without significantly increasing the length of the token sequence. Compared to a full spatiotemporal attention scheme or a scheme that decouples only the temporal dimension, we find that fully decoupled axial attention is less efficient and expressive, requiring more training time and model weights to achieve the same accuracy. Finally, we demonstrate in two fine-tuning tasks featuring different physics that models pretrained on PDEBench data outperform the ones trained from scratch, especially in the low data regime with frozen attention. 

**Abstract (ZH)**: 使用视觉转换器（ViT）架构准确表示时空物理系统中的多尺度特征需要非常长的计算成本高昂的标记序列。为解决这一问题，我们提出了一种自适应标记化方案，该方案基于局部特征动态调整拼块大小：一种确保收敛行为到均匀拼块细化，另一种则提供更好的计算效率。此外，我们还提出了一组时空注意机制，其中时间维度或轴向空间维度被解耦，并评估了它们的计算效率和数据效率。我们通过一系列实验评估所提出的自适应多尺度模型MATEY的性能。结果表明，自适应标记化方案在不显著增加标记序列长度的情况下提高了准确性。与时空完全解耦轴向注意机制或仅解耦时间维度的方案相比，我们发现完全解耦的轴向注意机制在表达能力和效率上较差，需要更多的训练时间和模型参数才能达到相同的准确性。最后，我们在两个不同的物理任务微调任务中显示，基于PDEBench数据预训练的模型优于从头训练的模型，特别是在数据稀缺且注意机制冻结的情况下。 

---
# Controlling Out-of-Domain Gaps in LLMs for Genre Classification and Generated Text Detection 

**Title (ZH)**: 针对体裁分类和生成文本检测的LLM领域外差距控制 

**Authors**: Dmitri Roussinov, Serge Sharoff, Nadezhda Puchnina  

**Link**: [PDF](https://arxiv.org/pdf/2412.20595)  

**Abstract**: This study demonstrates that the modern generation of Large Language Models (LLMs, such as GPT-4) suffers from the same out-of-domain (OOD) performance gap observed in prior research on pre-trained Language Models (PLMs, such as BERT). We demonstrate this across two non-topical classification tasks: 1) genre classification and 2) generated text detection. Our results show that when demonstration examples for In-Context Learning (ICL) come from one domain (e.g., travel) and the system is tested on another domain (e.g., history), classification performance declines significantly.
To address this, we introduce a method that controls which predictive indicators are used and which are excluded during classification. For the two tasks studied here, this ensures that topical features are omitted, while the model is guided to focus on stylistic rather than content-based attributes. This approach reduces the OOD gap by up to 20 percentage points in a few-shot setup. Straightforward Chain-of-Thought (CoT) methods, used as the baseline, prove insufficient, while our approach consistently enhances domain transfer performance. 

**Abstract (ZH)**: 本研究证明，现代大型语言模型（LLMs，如GPT-4）在领域外（OOD）性能上与先前对预训练语言模型（PLMs，如BERT）的研究中观察到的现象一致。我们在这两个非主题分类任务中展示了这一点：1）体裁分类；2）生成文本检测。研究结果显示，当基于上下文学习（ICL）的演示示例来自一个领域（如旅行），而系统在另一个领域（如历史）进行测试时，分类性能会显著下降。

为了应对这一问题，我们提出了一种方法，该方法在分类过程中控制了哪些预测指标被使用，哪些被排除。对于这里研究的两个任务，这种方法确保省略了主题特征，同时引导模型专注于风格而非内容属性。这种策略在少样本设置中最多可减少20个百分点的领域外性能差距。传统的简单链式思维（CoT）方法用作基准时证明是不足的，而我们的方法则能一致提升领域间迁移性能。 

---
# Kryptonite-N: Machine Learning Strikes Back 

**Title (ZH)**: Kryptonite-N：机器学习卷土重来 

**Authors**: Albus Li, Nathan Bailey, Will Sumerfield, Kira Kim  

**Link**: [PDF](https://arxiv.org/pdf/2412.20588)  

**Abstract**: Quinn et al propose challenge datasets in their work called ``Kryptonite-N". These datasets aim to counter the universal function approximation argument of machine learning, breaking the notation that machine learning can ``approximate any continuous function" \cite{original_paper}. Our work refutes this claim and shows that universal function approximations can be applied successfully; the Kryptonite datasets are constructed predictably, allowing logistic regression with sufficient polynomial expansion and L1 regularization to solve for any dimension N. 

**Abstract (ZH)**: Quinn等人在其作品《Kryptonite-N》中提出了挑战性数据集。这些数据集旨在反驳机器学习的普遍函数逼近论断，打破了机器学习能够“逼近任何连续函数”的这一传统观点 [原论文引用：《original_paper》]。我们的研究反驳了这一观点，并表明普遍函数逼近可以成功应用；通过合理构建Kryptonite数据集，使得足夠的多项式扩展和L1正则化下的逻辑回归能够解决任意维度N的问题。 

---
# Bridging the Gap: A Decade Review of Time-Series Clustering Methods 

**Title (ZH)**: 缩短差距：时间序列聚类方法十年综述 

**Authors**: John Paparrizos, Fan Yang, Haojun Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.20582)  

**Abstract**: Time series, as one of the most fundamental representations of sequential data, has been extensively studied across diverse disciplines, including computer science, biology, geology, astronomy, and environmental sciences. The advent of advanced sensing, storage, and networking technologies has resulted in high-dimensional time-series data, however, posing significant challenges for analyzing latent structures over extended temporal scales. Time-series clustering, an established unsupervised learning strategy that groups similar time series together, helps unveil hidden patterns in these complex datasets. In this survey, we trace the evolution of time-series clustering methods from classical approaches to recent advances in neural networks. While previous surveys have focused on specific methodological categories, we bridge the gap between traditional clustering methods and emerging deep learning-based algorithms, presenting a comprehensive, unified taxonomy for this research area. This survey highlights key developments and provides insights to guide future research in time-series clustering. 

**Abstract (ZH)**: 时间序列，作为序列数据最基础的表示形式之一，已在计算机科学、生物学、地质学、天文学和环境科学等多个学科领域得到了广泛研究。随着先进传感、存储和网络技术的发展，产生了高维的时间序列数据，这为长时间尺度上的潜在结构分析带来了巨大的挑战。时间序列聚类，作为一种成熟的无监督学习策略，能够将相似的时间序列归为一类，帮助揭示这些复杂数据集中的潜在模式。在本文综述中，我们从经典的聚类方法到近年来基于神经网络的技术进步，追溯了时间序列聚类方法的发展历程。尽管之前的综述主要集中在特定的方法学类别上，但本文旨在弥合传统聚类方法与新兴的深度学习算法之间的差距，提供一个全面且统一的分类框架。本文综述突出了关键的发展，并为未来的时间序列聚类研究提供指导性的见解。 

---
# A Survey on Time-Series Distance Measures 

**Title (ZH)**: 时间序列距离度量综述 

**Authors**: John Paparrizos, Haojun Li, Fan Yang, Kaize Wu, Jens E. d'Hondt, Odysseas Papapetrou  

**Link**: [PDF](https://arxiv.org/pdf/2412.20574)  

**Abstract**: Distance measures have been recognized as one of the fundamental building blocks in time-series analysis tasks, e.g., querying, indexing, classification, clustering, anomaly detection, and similarity search. The vast proliferation of time-series data across a wide range of fields has increased the relevance of evaluating the effectiveness and efficiency of these distance measures. To provide a comprehensive view of this field, this work considers over 100 state-of-the-art distance measures, classified into 7 categories: lock-step measures, sliding measures, elastic measures, kernel measures, feature-based measures, model-based measures, and embedding measures. Beyond providing comprehensive mathematical frameworks, this work also delves into the distinctions and applications across these categories for both univariate and multivariate cases. By providing comprehensive collections and insights, this study paves the way for the future development of innovative time-series distance measures. 

**Abstract (ZH)**: 距离度量已被认为是时间序列分析任务中的基本构建模块，例如查询、索引、分类、聚类、异常检测和相似性搜索。随着时间序列数据在各个领域的广泛应用，评估这些距离度量的有效性和效率的重要性也日益增加。为了提供该领域的全面视角，本文考虑了超过100种最先进的距离度量，并将其分类为7个类别：同步度量、滑动度量、弹性度量、核度量、特征基于度量、模型基于度量和嵌入度量。除了提供全面的数学框架之外，本文还探讨了这些类别之间的区别及其在单变量和多变量情况下的应用。通过提供全面的集合理论和见解，研究为未来创新的时间序列距离度量的发展铺平了道路。 

---
# Segmentation of Muscularis Propria in Colon Histopathology Images Using Vision Transformers for Hirschsprung's Disease 

**Title (ZH)**: 使用视觉变换器对肠肌层进行结肠病理图像分割以诊断霍乱性肠炎 

**Authors**: Youssef Megahed, Anthony Fuller, Saleh Abou-Alwan, Dina El Demellawy, Adrian D. C. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20571)  

**Abstract**: Hirschsprung's disease (HD) is a congenital birth defect diagnosed by identifying the lack of ganglion cells within the colon's muscularis propria, specifically within the myenteric plexus regions. There may be advantages for quantitative assessments of histopathology images of the colon, such as counting the ganglion and assessing their spatial distribution; however, this would be time-intensive for pathologists, costly, and subject to inter- and intra-rater variability. Previous research has demonstrated the potential for deep learning approaches to automate histopathology image analysis, including segmentation of the muscularis propria using convolutional neural networks (CNNs). Recently, Vision Transformers (ViTs) have emerged as a powerful deep learning approach due to their self-attention. This study explores the application of ViTs for muscularis propria segmentation in calretinin-stained histopathology images and compares their performance to CNNs and shallow learning methods. The ViT model achieved a DICE score of 89.9% and Plexus Inclusion Rate (PIR) of 100%, surpassing the CNN (DICE score of 89.2%; PIR of 96.0%) and k-means clustering method (DICE score of 80.7%; PIR 77.4%). Results assert that ViTs are a promising tool for advancing HD-related image analysis. 

**Abstract (ZH)**: 希里斯普朗格病（HD）是一种先天性出生缺陷，通过在结肠的肌层中识别缺乏神经节细胞，尤其是髓层区域的情况下进行诊断。虽然对结肠组织病理学图像进行定量评估，如计数神经节细胞并评估其空间分布等，可能具有优势，但这将耗费病理学家大量时间，成本较高，并且容易受到评价者间和评价者内差异的影响。以往研究已经展示了深度学习方法在自动组织病理学图像分析方面具有潜力，包括使用卷积神经网络（CNNs）对肌层进行分割。近年来，视觉变换器（ViTs）因其自注意力机制的出现而成为一种强大的深度学习方法。本研究探讨了ViTs在calretinin染色组织病理学图像中对肌层的分割应用，并将其性能与CNNs和浅层学习方法进行了比较。ViT模型取得了DICE分数为89.9%和Plexus包括率（PIR）为100%的结果，超过了CNN（DICE分数为89.2%；PIR为96.0%）和k-means聚类方法（DICE分数为80.7%；PIR为77.4%）。研究结果表明，ViTs是推进HD相关图像分析的一种有前景的工具。 

---
# Enhancing autonomous vehicle safety in rain: a data-centric approach for clear vision 

**Title (ZH)**: 提高雨天自动驾驶车辆安全性的数据为中心的方法以获得清晰视野 

**Authors**: Mark A. Seferian, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20565)  

**Abstract**: Autonomous vehicles face significant challenges in navigating adverse weather, particularly rain, due to the visual impairment of camera-based systems. In this study, we leveraged contemporary deep learning techniques to mitigate these challenges, aiming to develop a vision model that processes live vehicle camera feeds to eliminate rain-induced visual hindrances, yielding visuals closely resembling clear, rain-free scenes. Using the Car Learning to Act (CARLA) simulation environment, we generated a comprehensive dataset of clear and rainy images for model training and testing. In our model, we employed a classic encoder-decoder architecture with skip connections and concatenation operations. It was trained using novel batching schemes designed to effectively distinguish high-frequency rain patterns from low-frequency scene features across successive image frames. To evaluate the model performance, we integrated it with a steering module that processes front-view images as input. The results demonstrated notable improvements in steering accuracy, underscoring the model's potential to enhance navigation safety and reliability in rainy weather conditions. 

**Abstract (ZH)**: 自动驾驶车辆在恶劣天气条件下，尤其是降雨天气中，面临着显著的导航挑战，这主要是因为基于摄像头的视觉系统受到视觉干扰。本研究利用了当代的深度学习技术来应对这些挑战，旨在开发一种视觉模型，能够处理实时的车载摄像头数据，消除降雨引起的视觉障碍，使画面尽可能接近无雨的清晰场景。我们通过使用Car Learning to Act (CARLA)仿真环境生成了包含清晰和雨天图像的全面数据集，用于模型的训练和测试。在我们的模型中，我们采用了经典的编码器-解码器架构，并结合了跳跃连接和连接操作。该模型使用新颖的批量处理方案进行训练，以有效地区分连续图像帧中的高频降雨模式与低频场景特征。为了评估模型性能，我们将模型与一个方向控制模块结合，该模块以前方视角图像作为输入。结果表明，方向控制的准确性显著提高，这表明该模型有望在雨天条件下增强导航的安全性和可靠性。 

---
# Attacks on the neural network and defense methods 

**Title (ZH)**: 对神经网络的攻击与防御方法 

**Authors**: A. Korenev, G. Belokrylov, B. Lodonova, A. Novokhrestov  

**Link**: [PDF](https://arxiv.org/pdf/2412.20529)  

**Abstract**: This article will discuss the use of attacks on a neural network trained on audio data, as well as possible methods of protection against these attacks. FGSM, PGD and CW attacks, as well as data poisoning, will be considered. Within the framework of protection, Art-IBM and advertorch libraries will be considered. The obtained accuracy metrics within the framework of attack applications are presented 

**Abstract (ZH)**: 本文将讨论对基于音频数据训练的神经网络展开的攻击，以及可能的防护方法。将考虑FGSM、PGD和CW攻击，以及数据中毒等攻击方式。在防护方面，本文将讨论Art-IBM和advertorch库的相关方法。还将在攻击应用框架下展示获得的准确率指标。 

---
# Game Theory and Multi-Agent Reinforcement Learning : From Nash Equilibria to Evolutionary Dynamics 

**Title (ZH)**: 博弈论与多智能体强化学习：从纳什均衡到进化动力学 

**Authors**: Neil De La Fuente, Miquel Noguer i Alonso, Guim Casadellà  

**Link**: [PDF](https://arxiv.org/pdf/2412.20523)  

**Abstract**: This paper explores advanced topics in complex multi-agent systems building upon our previous work. We examine four fundamental challenges in Multi-Agent Reinforcement Learning (MARL): non-stationarity, partial observability, scalability with large agent populations, and decentralized learning. The paper provides mathematical formulations and analysis of recent algorithmic advancements designed to address these challenges, with a particular focus on their integration with game-theoretic concepts. We investigate how Nash equilibria, evolutionary game theory, correlated equilibrium, and adversarial dynamics can be effectively incorporated into MARL algorithms to improve learning outcomes. Through this comprehensive analysis, we demonstrate how the synthesis of game theory and MARL can enhance the robustness and effectiveness of multi-agent systems in complex, dynamic environments. 

**Abstract (ZH)**: 本文在我们之前工作的基础上，探讨了复杂多智能体系统的高级主题。我们研究了多智能体强化学习（MARL）中四个基本挑战：非平稳性、部分可观测性、大规模智能体群体的可扩展性以及去中心化学习。文章提供了对旨在解决这些挑战的最新算法进展的数学建模和分析，并特别关注这些算法与博弈论概念的集成。我们探讨了如何有效地将纳什均衡、进化博弈论、相关均衡和对抗动态纳入MARL算法中，以改善学习结果。通过对这些内容的全面分析，我们展示了将博弈论与MARL的结合如何增强复杂动态环境中多智能体系统的稳健性和有效性。 

---
# Goal-Conditioned Data Augmentation for Offline Reinforcement Learning 

**Title (ZH)**: 基于目标的离线强化学习数据增强方法 

**Authors**: Xingshuai Huang, Di Wu Member, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2412.20519)  

**Abstract**: Offline reinforcement learning (RL) enables policy learning from pre-collected offline datasets, relaxing the need to interact directly with the environment. However, limited by the quality of offline datasets, it generally fails to learn well-qualified policies in suboptimal datasets. To address datasets with insufficient optimal demonstrations, we introduce Goal-cOnditioned Data Augmentation (GODA), a novel goal-conditioned diffusion-based method for augmenting samples with higher quality. Leveraging recent advancements in generative modeling, GODA incorporates a novel return-oriented goal condition with various selection mechanisms. Specifically, we introduce a controllable scaling technique to provide enhanced return-based guidance during data sampling. GODA learns a comprehensive distribution representation of the original offline datasets while generating new data with selectively higher-return goals, thereby maximizing the utility of limited optimal demonstrations. Furthermore, we propose a novel adaptive gated conditioning method for processing noised inputs and conditions, enhancing the capture of goal-oriented guidance. We conduct experiments on the D4RL benchmark and real-world challenges, specifically traffic signal control (TSC) tasks, to demonstrate GODA's effectiveness in enhancing data quality and superior performance compared to state-of-the-art data augmentation methods across various offline RL algorithms. 

**Abstract (ZH)**: 离线强化学习（Offline RL）允许从预先收集的离线数据集中学习策略，从而减轻了直接与环境交互的需要。然而，受限于离线数据集的质量，它通常在质量较差的数据集上难以学到高质量的策略。为了解决缺乏足够优质演示数据集的问题，我们引入了一种新的目标条件数据增强方法——目标条件扩散增强（GODA），该方法能够生成更高质量的数据样本。利用生成建模领域的最新进展，GODA 引入了一种以回路为中心的目标条件机制，并结合了多种选择机制。具体而言，我们提出了一种可控缩放技术，在数据采样过程中提供增强的基于回路的指导。GODA 在保留原始离线数据集的同时，生成具有部分高回路收益目标的新数据，从而最大化有限优质演示的利用价值。此外，我们提出了一种新的自适应门控条件处理方法，用于处理嘈杂的输入和条件，增强了对目标导向指导的捕捉能力。我们在 D4RL 标准测试平台和现实世界挑战，特别是交通信号控制（TSC）任务中进行了实验，证明了GODA 在提高数据质量和各离线RL算法相较最先进的数据增强方法时的优越性能。 

---
# Dive into Time-Series Anomaly Detection: A Decade Review 

**Title (ZH)**: 深入时间序列异常检测：十年回顾 

**Authors**: Paul Boniol, Qinghua Liu, Mingyi Huang, Themis Palpanas, John Paparrizos  

**Link**: [PDF](https://arxiv.org/pdf/2412.20512)  

**Abstract**: Recent advances in data collection technology, accompanied by the ever-rising volume and velocity of streaming data, underscore the vital need for time series analytics. In this regard, time-series anomaly detection has been an important activity, entailing various applications in fields such as cyber security, financial markets, law enforcement, and health care. While traditional literature on anomaly detection is centered on statistical measures, the increasing number of machine learning algorithms in recent years call for a structured, general characterization of the research methods for time-series anomaly detection. This survey groups and summarizes anomaly detection existing solutions under a process-centric taxonomy in the time series context. In addition to giving an original categorization of anomaly detection methods, we also perform a meta-analysis of the literature and outline general trends in time-series anomaly detection research. 

**Abstract (ZH)**: 随着数据采集技术的 recent 进步以及流数据量和速度的不断增加，时间序列分析变得愈发重要。在此背景下，时间序列异常检测已成为一项重要活动，其在网络安全、金融市场、执法和医疗保健等领域都有着广泛的应用。尽管传统的异常检测文献主要关注统计方法，但近年来不断增加的机器学习算法促使我们对时间序列异常检测的研究方法进行结构化的、通用的概括。本文从过程导向的视角对现有异常检测解决方案进行了分类和总结。除了提出一种新的异常检测方法分类外，我们还对文献进行了元分析，概述了时间序列异常检测研究的一般趋势。 

---
# Stratify: Unifying Multi-Step Forecasting Strategies 

**Title (ZH)**: Stratify: 统一多步预测策略 

**Authors**: Riku Green, Grant Stevens, Zahraa Abdallah, Telmo M. Silva Filho  

**Link**: [PDF](https://arxiv.org/pdf/2412.20510)  

**Abstract**: A key aspect of temporal domains is the ability to make predictions multiple time steps into the future, a process known as multi-step forecasting (MSF). At the core of this process is selecting a forecasting strategy, however, with no existing frameworks to map out the space of strategies, practitioners are left with ad-hoc methods for strategy selection. In this work, we propose Stratify, a parameterised framework that addresses multi-step forecasting, unifying existing strategies and introducing novel, improved strategies. We evaluate Stratify on 18 benchmark datasets, five function classes, and short to long forecast horizons (10, 20, 40, 80). In over 84% of 1080 experiments, novel strategies in Stratify improved performance compared to all existing ones. Importantly, we find that no single strategy consistently outperforms others in all task settings, highlighting the need for practitioners explore the Stratify space to carefully search and select forecasting strategies based on task-specific requirements. Our results are the most comprehensive benchmarking of known and novel forecasting strategies. We make code available to reproduce our results. 

**Abstract (ZH)**: 时间域的一个关键方面是能够对未来多个时间步进行预测，这一过程被称为多步预测（Multi-step Forecasting, MSF）。这一过程的核心在于选择一个预测策略，然而至今尚未有框架来映射所有可用策略的空间，因此实践者只能依靠经验方法来选择策略。在本研究中，我们提出了Stratify，这是一个参数化的框架，旨在解决多步预测问题，统一了现有策略并引入了新的改进策略。我们在18个基准数据集中、五个函数类别以及从短期到长期的预测时间范围内（10步、20步、40步、80步）评估了Stratify。在84%以上的840项实验中，Stratify中的新型策略在性能上优于所有现有策略。值得注意的是，我们发现没有单一策略在所有任务设置中都能始终优于其他策略，这强调了需要实践者探索Stratify空间，根据特定任务需求仔细搜索和选择预测策略的重要性。我们的研究成果是对现有和新型预测策略的最全面基准评估。我们提供了代码以便其他人能够重现我们的结果。 

---
# A Multiparty Homomorphic Encryption Approach to Confidential Federated Kaplan Meier Survival Analysis 

**Title (ZH)**: 一种多方同态加密方法实现保密联邦 Kaplan Meier 生存分析 

**Authors**: Narasimha Raghavan Veeraragavan, Svetlana Boudko, Jan Franz Nygård  

**Link**: [PDF](https://arxiv.org/pdf/2412.20495)  

**Abstract**: The proliferation of healthcare data has expanded opportunities for collaborative research, yet stringent privacy regulations hinder pooling sensitive patient records. We propose a \emph{multiparty homomorphic encryption-based} framework for \emph{privacy-preserving federated Kaplan--Meier survival analysis}, offering native floating-point support, a theoretical model, and explicit reconstruction-attack mitigation. Compared to prior work, our framework ensures encrypted federated survival estimates closely match centralized outcomes, supported by formal utility-loss bounds that demonstrate convergence as aggregation and decryption noise diminish. Extensive experiments on the NCCTG Lung Cancer and synthetic Breast Cancer datasets confirm low \emph{mean absolute error (MAE)} and \emph{root mean squared error (RMSE)}, indicating negligible deviations between encrypted and non-encrypted survival curves. Log-rank and numerical accuracy tests reveal \emph{no significant difference} between federated encrypted and non-encrypted analyses, preserving statistical validity. A reconstruction-attack evaluation shows smaller federations (2--3 providers) with overlapping data between the institutions are vulnerable, a challenge mitigated by multiparty encryption. Larger federations (5--50 sites) degrade reconstruction accuracy further, with encryption improving confidentiality. Despite an 8--19$\times$ computational overhead, threshold-based homomorphic encryption is \emph{feasible for moderate-scale deployments}, balancing security and runtime. By providing robust privacy guarantees alongside high-fidelity survival estimates, our framework advances the state-of-the art in secure multi-institutional survival analysis. 

**Abstract (ZH)**: 医疗数据的激增为协作研究提供了更多的机会，但严格的隐私法规阻碍了敏感病人记录的合并。我们提议了一种基于多方同态加密的框架，用于保护隐私的联邦Kaplan-Meier生存分析，该框架提供本机浮点支持、理论模型以及明确的重构攻击缓解。与先前的工作相比，我们的框架确保加密的联邦生存估计值与集中式结果高度一致，由正式的效用损失边界支持，这些边界证明了随着聚合和解密噪声的减少而接近一致。在NCCTG肺癌和合成乳腺癌数据集上的广泛实验中，加密和非加密生存曲线之间的平均绝对误差（MAE）和均方根误差（RMSE）都很低，表明两者的偏差可以忽略不计。对卡方检验和数值精度测试显示，联邦加密和非加密分析之间没有显著差异，从而保持了统计有效性。重构攻击评估显示，2-3个提供者的小型联合体因机构间的数据重叠而易受攻击，这一挑战通过多方加密得以缓解。更大的联合体（5-50个站点）进一步降低了重构准确性，但加密增强了保密性。尽管计算开销增加了8-19倍，基于阈值的同态加密在中等规模部署中是可行的，平衡了安全性和运行时间。通过提供强大的隐私保证和高精度的生存估计，我们的框架推动了安全多机构生存分析的技术前沿。 

---
# Integrating Natural Language Processing Techniques of Text Mining Into Financial System: Applications and Limitations 

**Title (ZH)**: 将自然语言处理技术集成到金融系统中的文本挖掘：应用与局限性 

**Authors**: Denisa Millo, Blerina Vika, Nevila Baci  

**Link**: [PDF](https://arxiv.org/pdf/2412.20438)  

**Abstract**: The financial sector, a pivotal force in economic development, increasingly uses the intelligent technologies such as natural language processing to enhance data processing and insight extraction. This research paper through a review process of the time span of 2018-2023 explores the use of text mining as natural language processing techniques in various components of the financial system including asset pricing, corporate finance, derivatives, risk management, and public finance and highlights the need to address the specific problems in the discussion section. We notice that most of the research materials combined probabilistic with vector-space models, and text-data with numerical ones. The most used technique regarding information processing is the information classification technique and the most used algorithms include the long-short term memory and bidirectional encoder models. The research noticed that new specific algorithms are developed and the focus of the financial system is mainly on asset pricing component. The research also proposes a path from engineering perspective for researchers who need to analyze financial text. The challenges regarding text mining perspective such as data quality, context-adaption and model interpretability need to be solved so to integrate advanced natural language processing models and techniques in enhancing financial analysis and prediction. Keywords: Financial System (FS), Natural Language Processing (NLP), Software and Text Engineering, Probabilistic, Vector-Space, Models, Techniques, TextData, Financial Analysis. 

**Abstract (ZH)**: 金融部门作为经济发展的关键力量，越来越多地利用自然语言处理等智能技术来提升数据处理和洞察提取能力。本研究论文通过2018年至2023年间的时间跨度回顾，探讨了自然语言处理技术在金融系统各个组成部分中的应用，包括资产定价、公司金融、衍生品、风险管理及公共财政，并在讨论部分强调了解决特定问题的必要性。研究发现，大多数研究材料结合了概率模型与向量空间模型，以及文本数据与数值数据。有关信息处理的最常用技术是信息分类技术，最常用的算法包括长短时记忆（LSTM）和双向编码器表示（BiLSTM）模型。研究注意到，新算法正在不断开发，金融系统的重点主要集中在资产定价方面。研究还从工程角度提出了一条路径，供需要分析金融文本的研究人员参考。从文本挖掘的角度考虑，面临的挑战如数据质量、上下文适应性和模型可解释性需得到解决，以便将先进的自然语言处理模型和技术整合到金融分析和预测中。关键词：金融系统（FS）、自然语言处理（NLP）、软件与文本工程、概率模型、向量空间模型、技术、文本数据、金融分析。 

---
# Multi-Scenario Reasoning: Unlocking Cognitive Autonomy in Humanoid Robots for Multimodal Understanding 

**Title (ZH)**: 多场景推理：在类人机器人中实现多模态理解的认知自主性 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20429)  

**Abstract**: To improve the cognitive autonomy of humanoid robots, this research proposes a multi-scenario reasoning architecture to solve the technical shortcomings of multi-modal understanding in this field. It draws on simulation based experimental design that adopts multi-modal synthesis (visual, auditory, tactile) and builds a simulator "Maha" to perform the experiment. The findings demonstrate the feasibility of this architecture in multimodal data. It provides reference experience for the exploration of cross-modal interaction strategies for humanoid robots in dynamic environments. 

**Abstract (ZH)**: 为了提高类人机器人的情境认知自主能力，本研究提出了一种多场景推理架构，以解决该领域多模态理解的技术短板。该架构借鉴了基于仿真的实验设计，采用了多模态合成（视觉、听觉、触觉），并构建了一个名为“Maha”的模拟器来执行实验。研究发现表明了该架构在多模态数据上的可行性，为其在动态环境下类人机器人跨模态交互策略的探索提供了参考经验。 

---
# Comparative Performance of Advanced NLP Models and LLMs in Multilingual Geo-Entity Detection 

**Title (ZH)**: 多语言地理实体检测中先进NLP模型和大语言模型的性能比较 

**Authors**: Kalin Kopanov  

**Link**: [PDF](https://arxiv.org/pdf/2412.20414)  

**Abstract**: The integration of advanced Natural Language Processing (NLP) methodologies and Large Language Models (LLMs) has significantly enhanced the extraction and analysis of geospatial data from multilingual texts, impacting sectors such as national and international security. This paper presents a comprehensive evaluation of leading NLP models -- SpaCy, XLM-RoBERTa, mLUKE, GeoLM -- and LLMs, specifically OpenAI's GPT 3.5 and GPT 4, within the context of multilingual geo-entity detection. Utilizing datasets from Telegram channels in English, Russian, and Arabic, we examine the performance of these models through metrics such as accuracy, precision, recall, and F1 scores, to assess their effectiveness in accurately identifying geospatial references. The analysis exposes each model's distinct advantages and challenges, underscoring the complexities involved in achieving precise geo-entity identification across varied linguistic landscapes. The conclusions drawn from this experiment aim to direct the enhancement and creation of more advanced and inclusive NLP tools, thus advancing the field of geospatial analysis and its application to global security. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，应符合学术规范：

先进的自然语言处理（NLP）方法与大型语言模型（LLMs）的集成显著提高了从多语言文本中提取和分析地理位置数据的能力，对国家和国际安全等领域产生了重大影响。本文对SpaCy、XLM-RoBERTa、mLUKE、GeoLM等领先NLP模型以及OpenAI的GPT 3.5和GPT 4等LLMs，在多语言地名检测方面的表现进行了全面评估。通过使用来自Telegram频道的英文、俄文和阿拉伯文数据集，我们利用准确率、精确率、召回率和F1分数等指标，评估这些模型在准确识别地理位置参考方面的有效性。分析结果揭示了每个模型的独特优势和挑战，突显了在不同语言环境中实现精确地名识别的复杂性。本实验得出的结论旨在指导NLP工具的改进和完善，从而推动地理空间分析领域及其在国际安全应用中的发展。 

---
# Multi-Objective Large Language Model Unlearning 

**Title (ZH)**: 多目标大型语言模型去学习 

**Authors**: Zibin Pan, Shuwen Zhang, Yuesheng Zheng, Chi Li, Yuheng Cheng, Junhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.20412)  

**Abstract**: Machine unlearning in the domain of large language models (LLMs) has attracted great attention recently, which aims to effectively eliminate undesirable behaviors from LLMs without full retraining from scratch. In this paper, we explore the Gradient Ascent (GA) approach in LLM unlearning, which is a proactive way to decrease the prediction probability of the model on the target data in order to remove their influence. We analyze two challenges that render the process impractical: gradient explosion and catastrophic forgetting. To address these issues, we propose Multi-Objective Large Language Model Unlearning (MOLLM) algorithm. We first formulate LLM unlearning as a multi-objective optimization problem, in which the cross-entropy loss is modified to the unlearning version to overcome the gradient explosion issue. A common descent update direction is then calculated, which enables the model to forget the target data while preserving the utility of the LLM. Our empirical results verify that MoLLM outperforms the SOTA GA-based LLM unlearning methods in terms of unlearning effect and model utility preservation. 

**Abstract (ZH)**: 在大型语言模型（LLMs）领域的机器遗忘问题最近引起了广泛的关注，其目标是在不完全从头开始重新训练的情况下，有效消除大型语言模型中的不良行为。本文探讨了在LLMs遗忘中应用梯度上升（GA）方法，这是一种主动的手段，旨在通过降低模型对目标数据的预测概率来减少其影响。我们分析了导致这一过程不可行的两个挑战：梯度爆炸和灾难性遗忘。为了解决这些问题，我们提出了多目标大型语言模型遗忘算法（MOLLM）。我们首先将LLMs遗忘问题形式化为一个多目标优化问题，在其中通过修改交叉熵损失为遗忘版本来克服梯度爆炸问题。然后计算了一个共同的下降更新方向，使模型能够忘记目标数据同时保留大型语言模型的有用性。我们的实证结果验证了MOLLM在遗忘效果和模型有用性保留方面优于现有的基于GA的大型语言模型遗忘方法。 

---
# Natural Language Fine-Tuning 

**Title (ZH)**: 自然语言微调 

**Authors**: Jia Liu, Yue Wang, Zhiqi Lin, Min Chen, Yixue Hao, Long Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20382)  

**Abstract**: Large language model fine-tuning techniques typically depend on extensive labeled data, external guidance, and feedback, such as human alignment, scalar rewards, and demonstration. However, in practical application, the scarcity of specific knowledge poses unprecedented challenges to existing fine-tuning techniques. In this paper, focusing on fine-tuning tasks in specific domains with limited data, we introduce Natural Language Fine-Tuning (NLFT), which utilizes natural language for fine-tuning for the first time. By leveraging the strong language comprehension capability of the target LM, NLFT attaches the guidance of natural language to the token-level outputs. Then, saliency tokens are identified with calculated probabilities. Since linguistic information is effectively utilized in NLFT, our proposed method significantly reduces training costs. It markedly enhances training efficiency, comprehensively outperforming reinforcement fine-tuning algorithms in accuracy, time-saving, and resource conservation. Additionally, on the macro level, NLFT can be viewed as a token-level fine-grained optimization of SFT, thereby efficiently replacing the SFT process without the need for warm-up (as opposed to ReFT requiring multiple rounds of warm-up with SFT). Compared to SFT, NLFT does not increase the algorithmic complexity, maintaining O(n). Extensive experiments on the GSM8K dataset demonstrate that NLFT, with only 50 data instances, achieves an accuracy increase that exceeds SFT by 219%. Compared to ReFT, the time complexity and space complexity of NLFT are reduced by 78.27% and 92.24%, respectively. The superior technique of NLFT is paving the way for the deployment of various innovative LLM fine-tuning applications when resources are limited at network edges.
Our code has been released at this https URL. 

**Abstract (ZH)**: 大型语言模型微调技术通常依赖于大量标记数据、外部指引和反馈，例如人类对齐、标量奖励和示范。然而，在实际应用中，特定领域的知识稀缺性对现有的微调技术提出了前所未有的挑战。本文旨在解决在数据有限的特定领域下微调任务的问题，首次提出了自然语言微调（NLFT）技术。NLFT 利用自然语言对目标LM进行微调，并将其指导应用于token级别的输出。通过计算概率来识别显著token。由于NLFT中有效利用了语言信息，我们提出的方法显著降低了训练成本，极大地提升了训练效率，并在准确度、节省时间和资源方面全面超越了强化微调算法。在宏观层面，NLFT可以被视为SFT的token级细粒度优化，因此可以高效地替代SFT过程，而无需预热（不同于ReFT需要多轮SFT预热）。与SFT相比，NLFT并未增加算法复杂度，保持在O(n)水平。实验结果表明，使用仅50个数据实例，NLFT的准确度提升了超过219%，比ReFT分别在时间复杂度和空间复杂度上减少了78.27%和92.24%。NLFT的优越技术为在网络边缘资源有限的情况下部署各种创新的LLM微调应用铺平了道路。

我们的代码已在此处发布：[链接] 

---
# A Deep Subgrouping Framework for Precision Drug Repurposing via Emulating Clinical Trials on Real-world Patient Data 

**Title (ZH)**: 基于模拟临床试验的现实患者数据精细亚群分组框架以实现精确药物再利用 

**Authors**: Seungyeon Lee, Ruoqi Liu, Feixiong Cheng, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20373)  

**Abstract**: Drug repurposing identifies new therapeutic uses for existing drugs, reducing the time and costs compared to traditional de novo drug discovery. Most existing drug repurposing studies using real-world patient data often treat the entire population as homogeneous, ignoring the heterogeneity of treatment responses across patient subgroups. This approach may overlook promising drugs that benefit specific subgroups but lack notable treatment effects across the entire population, potentially limiting the number of repurposable candidates identified. To address this, we introduce STEDR, a novel drug repurposing framework that integrates subgroup analysis with treatment effect estimation. Our approach first identifies repurposing candidates by emulating multiple clinical trials on real-world patient data and then characterizes patient subgroups by learning subgroup-specific treatment effects. We deploy \model to Alzheimer's Disease (AD), a condition with few approved drugs and known heterogeneity in treatment responses. We emulate trials for over one thousand medications on a large-scale real-world database covering over 8 million patients, identifying 14 drug candidates with beneficial effects to AD in characterized subgroups. Experiments demonstrate STEDR's superior capability in identifying repurposing candidates compared to existing approaches. Additionally, our method can characterize clinically relevant patient subgroups associated with important AD-related risk factors, paving the way for precision drug repurposing. 

**Abstract (ZH)**: 药物再利用可以发现现有药物的新治疗用途，相比传统从头药物发现，可以大大缩短时间和降低成本。大多数使用真实患者数据进行药物再利用的研究往往将整个人群视为同质的，忽视了患者亚群体之间治疗反应的异质性。这种做法可能会忽视对特定亚群体有益但整个群体治疗效果不显著的药物，从而限制了发现再利用候选药物的数量。为解决这一问题，我们引入了STEDR（Subgroup-based Treatment Effect Disentanglement for Drug Repurposing），一种将亚组分析与治疗效果估计相结合的新颖药物再利用框架。我们的方法首先通过模拟实际临床试验来识别再利用候选药物，然后再通过学习亚组特定的治疗效果来表征患者亚组。我们部署该方法应用于阿尔茨海默病（AD），这是一种批准药物有限且治疗反应存在异质性的疾病。我们在涵盖超过800万名患者的大型真实世界数据库中模拟了超过一千种药物的试验，识别出14种在表征的亚组中对AD具有有益效果的药物候选物。实验结果显示，STEDR相比现有方法在识别再利用候选药物方面具有优越的能力。此外，我们的方法还可以表征与AD相关的重要临床风险因素相关的患者亚组，为精准药物再利用铺平了道路。 

---
# LLM2: Let Large Language Models Harness System 2 Reasoning 

**Title (ZH)**: LLM2：让大型语言模型运用系统二推理

注释：在翻译学术术语时，通常会保持与原文相近的缩写形式。"System 2 Reasoning" 是心理学中描述深入、耗时和逻辑化思考过程的概念，在翻译时，直接翻译为“系统二推理”更为准确和规范。 

**Authors**: Cheng Yang, Chufan Shi, Siheng Li, Bo Shui, Yujiu Yang, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2412.20372)  

**Abstract**: Large language models (LLMs) have exhibited impressive capabilities across a myriad of tasks, yet they occasionally yield undesirable outputs. We posit that these limitations are rooted in the foundational autoregressive architecture of LLMs, which inherently lacks mechanisms for differentiating between desirable and undesirable results. Drawing inspiration from the dual-process theory of human cognition, we introduce LLM2, a novel framework that combines an LLM (System 1) with a process-based verifier (System 2). Within LLM2, the LLM is responsible for generating plausible candidates, while the verifier provides timely process-based feedback to distinguish desirable and undesirable outputs. The verifier is trained with a pairwise comparison loss on synthetic process-supervision data generated through our token quality exploration strategy. Empirical results on mathematical reasoning benchmarks substantiate the efficacy of LLM2, exemplified by an accuracy enhancement from 50.3 to 57.8 (+7.5) for Llama3-1B on GSM8K. Furthermore, when combined with self-consistency, LLM2 achieves additional improvements, boosting major@20 accuracy from 56.2 to 70.2 (+14.0). 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务中展现了令人 impressive 的能力，但在某些情况下也会产生不良输出。我们认为这些限制源于LLMs的自回归基础架构，这种架构本身就缺乏区分有利和不利结果的机制。受到人类认知的双过程理论的启发，我们提出了一种名为LLM2的新框架，该框架将一个LLM（系统1）与基于过程的验证器（系统2）结合在一起。在LLM2中，LLM负责生成合理的候选答案，而验证器则提供及时的过程反馈，以区分有利和不利的输出。验证器通过我们的令牌质量探索策略生成的合成过程监督数据进行训练，并使用成对比较损失函数进行训练。数学推理基准数据集上的实验证明了LLM2的有效性，例如LLAMA3-1B在GSM8K上的准确性从50.3提高到了57.8（+7.5）。此外，当与自我一致性结合使用时，LLM2还实现了额外的改进，主要@20准确性从56.2提高到了70.2（+14.0）。 

---
# Safe Multiagent Coordination via Entropic Exploration 

**Title (ZH)**: 通过熵驱动的探索实现安全多智能体协调 

**Authors**: Ayhan Alp Aydeniz, Enrico Marchesini, Robert Loftin, Christopher Amato, Kagan Tumer  

**Link**: [PDF](https://arxiv.org/pdf/2412.20361)  

**Abstract**: Many real-world multiagent learning problems involve safety concerns. In these setups, typical safe reinforcement learning algorithms constrain agents' behavior, limiting exploration -- a crucial component for discovering effective cooperative multiagent behaviors. Moreover, the multiagent literature typically models individual constraints for each agent and has yet to investigate the benefits of using joint team constraints. In this work, we analyze these team constraints from a theoretical and practical perspective and propose entropic exploration for constrained multiagent reinforcement learning (E2C) to address the exploration issue. E2C leverages observation entropy maximization to incentivize exploration and facilitate learning safe and effective cooperative behaviors. Experiments across increasingly complex domains show that E2C agents match or surpass common unconstrained and constrained baselines in task performance while reducing unsafe behaviors by up to $50\%$. 

**Abstract (ZH)**: 许多现实世界的多代理学习问题涉及安全方面的考虑。在这种设置中，典型的安全强化学习算法会约束代理的行为，限制探索——这是发现有效合作行为的关键环节。此外，多代理文献通常为每个代理建模个体约束，尚未探讨使用联合团队约束所带来的益处。在本工作中，我们从理论和实践的角度分析了团队约束，并提出了受限多代理强化学习中的熵探索方法（E2C）来解决探索问题。E2C 利用观察熵最大化来激励探索，并促进学习安全有效的合作行为。实验结果表明，在逐渐复杂的任务领域中，E2C 代理在任务性能上与常见的未受约束和受约束基准相当甚至超越，并将不安全行为减少了高达 50%。 

---
# EmoReg: Directional Latent Vector Modeling for Emotional Intensity Regularization in Diffusion-based Voice Conversion 

**Title (ZH)**: EmoReg：基于扩散模型的声音转换中情感强度正则化的方向潜变量建模 

**Authors**: Ashishkumar Gudmalwar, Ishan D. Biyani, Nirmesh Shah, Pankaj Wasnik, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2412.20359)  

**Abstract**: The Emotional Voice Conversion (EVC) aims to convert the discrete emotional state from the source emotion to the target for a given speech utterance while preserving linguistic content. In this paper, we propose regularizing emotion intensity in the diffusion-based EVC framework to generate precise speech of the target emotion. Traditional approaches control the intensity of an emotional state in the utterance via emotion class probabilities or intensity labels that often lead to inept style manipulations and degradations in quality. On the contrary, we aim to regulate emotion intensity using self-supervised learning-based feature representations and unsupervised directional latent vector modeling (DVM) in the emotional embedding space within a diffusion-based framework. These emotion embeddings can be modified based on the given target emotion intensity and the corresponding direction vector. Furthermore, the updated embeddings can be fused in the reverse diffusion process to generate the speech with the desired emotion and intensity. In summary, this paper aims to achieve high-quality emotional intensity regularization in the diffusion-based EVC framework, which is the first of its kind work. The effectiveness of the proposed method has been shown across state-of-the-art (SOTA) baselines in terms of subjective and objective evaluations for the English and Hindi languages \footnote{Demo samples are available at the following URL: \url{this https URL}}. 

**Abstract (ZH)**: 情感语音转换（EVC）的目标是将给定语音片段的源情感状态转换为目标情感状态，同时保留语言内容。本文提出了一种在基于扩散的情感语音转换框架中正则化情感强度的方法，以生成精确的目标情感语音。传统的方法通过情绪类概率或强度标签来控制音节中的情感强度，这往往会导致不恰当的情感风格处理，并且在质量上有所下降。相比之下，我们希望通过自监督学习得到的特征表示和无监督的方向潜在向量建模（DVM）在情感嵌入空间中调节情感强度。这些情感嵌入可以根据给定的目标情感强度和相应的方向向量进行修改。此外，在反转扩散过程中，更新后的嵌入可以融合生成具有预期情感和强度的语音。总之，本文旨在在基于扩散的情感语音转换框架中实现高质量的情感强度正则化，这是该领域的首个工作。所提出的方法在英语和印地语的最新基准方法（SOTA）中分别进行了主观和客观评价，并展示了其有效性（附有演示样本：\url{请将网址替换为实际URL}）。 

---
# HindiLLM: Large Language Model for Hindi 

**Title (ZH)**: HindiLLM：印地语大型语言模型 

**Authors**: Sanjay Chouhan, Shubha Brata Nath, Aparajita Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2412.20357)  

**Abstract**: The advancements in the Large Language Model (LLM) have helped in solving several problems related to language processing. Most of the researches have focused on the English language only, because of its popularity and abundance on the internet. However, a high-performance language model for Hindi and other Indic languages is lacking in the literature. In this work, we have pre-trained two autoregressive LLM models for the Hindi language, namely HindiLLM-Small and HindiLLM-Medium. We use a two-step process comprising unsupervised pre-training and supervised fine-tuning. First, we create a large and high-quality text corpus for unsupervised pre-training. Next, we train a Byte-Pair Encoding, named HindiLLM tokenizer, using the pre-training text data. We then perform training on the unlabeled data, known as the pre-training step, to get the HindiLLM base models. Furthermore, we perform fine-tuning of the HindiLLM base models for different tasks like sentiment analysis, text classification, natural language inference, and multiple choice question-answer on popular labeled datasets to measure the real-world performance. The evaluation shows that the HindiLLM-based fine-tuned models outperform several models in most of the language related tasks. 

**Abstract (ZH)**: 大型语言模型（LLM）的进步在解决语言处理相关问题方面发挥了重要作用。大多数研究集中在英语上，因为英语因其在互联网上的流行性和丰富性而备受关注。然而，关于孟买语和其他印度语言的高性能语言模型的文献仍然不足。在这项工作中，我们预训练了两个自回归LLM模型，分别为HindiLLM-Small和HindiLLM-Medium，专门用于孟买语。我们采用两步过程，包括无监督预训练和有监督微调。首先，我们创建了一个大型和高质量的文本语料库用于无监督预训练。接着，我们利用预训练文本数据训练了一个名为HindiLLM的字节对编码器。然后，我们在未标记的数据上进行训练，即预训练步骤，以生成HindiLLM基础模型。此外，我们对HindiLLM基础模型进行了针对不同任务的微调，包括情感分析、文本分类、自然语言推理和多项选择问题回答，这些建立在流行的标记数据集之上，以衡量其实际性能。评估结果显示，基于HindiLLM的微调模型在大多数语言相关任务中超越了多种模型。 

---
# Distilling Desired Comments for Enhanced Code Review with Large Language Models 

**Title (ZH)**: 使用大型语言模型强化代码审查所需的评论精炼 

**Authors**: Yongda Yu, Lei Zhang, Guoping Rong, Haifeng Shen, Jiahao Zhang, Haoxiang Yan, Guohao Shi, Dong Shao, Ruiqi Pan, Yuan Li, Qiushi Wang, Zhao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.20340)  

**Abstract**: There has been a growing interest in using Large Language Models (LLMs) for code review thanks to their proven proficiency in code comprehension. The primary objective of most review scenarios is to generate desired review comments (DRCs) that explicitly identify issues to trigger code fixes. However, existing LLM-based solutions are not so effective in generating DRCs for various reasons such as hallucination. To enhance their code review ability, they need to be fine-tuned with a customized dataset that is ideally full of DRCs. Nevertheless, such a dataset is not yet available, while manual annotation of DRCs is too laborious to be practical. In this paper, we propose a dataset distillation method, Desiview, which can automatically construct a distilled dataset by identifying DRCs from a code review dataset. Experiments on the CodeReviewer dataset comprising more than 150K review entries show that Desiview achieves an impressive performance of 88.93%, 80.37%, 86.67%, and 84.44% in terms of Precision, Recall, Accuracy, and F1, respectively, surpassing state-of-the-art methods. To validate the effect of such a distilled dataset on enhancing LLMs' code review ability, we first fine-tune the latest LLaMA series (i.e., LLaMA 3 and LLaMA 3.1) to build model Desiview4FT. We then enhance the model training effect through KTO alignment by feeding those review comments identified as non-DRCs to the LLMs, resulting in model Desiview4FA. Verification results indicate that Desiview4FA slightly outperforms Desiview4FT, while both models have significantly improved against the base models in terms of generating DRCs. Human evaluation confirms that both models identify issues more accurately and tend to generate review comments that better describe the issues contained in the code than the base LLMs do. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在代码审查方面表现出色的语言理解能力，人们对使用LLMs进行代码审查的兴趣正在逐渐增长。大多数审查场景的主要目标是生成所需审查评论（DRCs），以明确标识问题并触发代码修复。然而，现有的基于LLM的解决方案在生成DRCs方面并不十分有效，这主要是由于幻觉等问题。为了增强其代码审查能力，它们需要使用一个自定义的数据集进行微调，该数据集应包含大量的DRCs。然而，这样的数据集目前尚未可用，而人工标注DRCs则过于劳力密集，难以实现。在本文中，我们提出了一种数据集精炼方法Desiview，该方法能够通过从代码审查数据集中识别DRCs来自动构建精炼数据集。实验结果表明，Desiview在精度、召回率、准确率和F1值方面的性能分别为88.93%、80.37%、86.67%和84.44%，超越了当前最先进的方法。为了验证这种精炼数据集对增强LLMs代码审查能力的效果，我们首先将最新的LLaMA系列模型（即LLaMA 3和LLaMA 3.1）进行微调，构建了模型Desiview4FT。然后，通过KTO对齐增强模型训练效果，通过输入被识别为非DRCs的审查评论来进一步训练LLMs，从而构建了Desiview4FA模型。验证结果表明，Desiview4FA在生成DRCs方面略微优于Desiview4FT，而两者在生成DRCs方面的表现都显著优于基础模型。人工评估证实，两者在识别问题方面更准确，且生成的审查评论更能够详细描述代码中的问题，优于基础LLMs的表现。 

---
# Mind the Data Gap: Bridging LLMs to Enterprise Data Integration 

**Title (ZH)**: 注意数据缺口：连接大规模语言模型与企业数据集成 

**Authors**: Moe Kayali, Fabian Wenz, Nesime Tatbul, Çağatay Demiralp  

**Link**: [PDF](https://arxiv.org/pdf/2412.20331)  

**Abstract**: Leading large language models (LLMs) are trained on public data. However, most of the world's data is dark data that is not publicly accessible, mainly in the form of private organizational or enterprise data. We show that the performance of methods based on LLMs seriously degrades when tested on real-world enterprise datasets. Current benchmarks, based on public data, overestimate the performance of LLMs. We release a new benchmark dataset, the GOBY Benchmark, to advance discovery in enterprise data integration. Based on our experience with this enterprise benchmark, we propose techniques to uplift the performance of LLMs on enterprise data, including (1) hierarchical annotation, (2) runtime class-learning, and (3) ontology synthesis. We show that, once these techniques are deployed, the performance on enterprise data becomes on par with that of public data. The Goby benchmark can be obtained at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）是在公共数据上进行训练的。然而，世界上大部分数据是未公开的数据（暗数据），主要以私有组织或企业数据的形式存在。我们发现，基于LLMs的方法在实际企业数据集上的性能显著下降。当前基于公共数据的基准测试过分高估了LLMs的性能。为推进企业数据集成的研究，我们发布了一个新的基准数据集——GOBY基准集。基于我们在该企业基准数据集上的经验，我们提出了提升LLMs在企业数据上性能的技术方法，包括（1）分层标注、（2）运行时类学习以及（3）本体合成。我们展示，一旦这些技术被部署，LLMs在企业数据上的表现将与公共数据相当。GOBY基准集可以从以下链接获取：https://this-https-url.com。 

---
# Protein Structure Prediction in the 3D HP Model Using Deep Reinforcement Learning 

**Title (ZH)**: 使用深度强化学习在3D HP模型中进行蛋白质结构预测 

**Authors**: Giovanny Espitia, Yui Tik Pang, James C. Gumbart  

**Link**: [PDF](https://arxiv.org/pdf/2412.20329)  

**Abstract**: We address protein structure prediction in the 3D Hydrophobic-Polar lattice model through two novel deep learning architectures. For proteins under 36 residues, our hybrid reservoir-based model combines fixed random projections with trainable deep layers, achieving optimal conformations with 25% fewer training episodes. For longer sequences, we employ a long short-term memory network with multi-headed attention, matching best-known energy values. Both architectures leverage a stabilized Deep Q-Learning framework with experience replay and target networks, demonstrating consistent achievement of optimal conformations while significantly improving training efficiency compared to existing methods. 

**Abstract (ZH)**: 我们通过两种新颖的深度学习架构，针对3D疏水-极性晶格模型中的蛋白质结构预测问题进行了研究。对于长度小于36个残基的蛋白质，我们提出了一种基于混合蓄水池的模型，该模型结合了固定随机投影与可训练的深层网络层，仅需25%的训练周期即可获得最优构象。对于更长的序列，我们采用了长短期记忆网络结合多头注意力机制，能够匹配最佳已知的能量值。两种架构均利用了稳定化的深度Q学习框架，并结合经验回放和目标网络，显示出在获得最优构象方面的一致性表现，同时显著提高了训练效率，相比现有方法有着显著改进。 

---
# Hypergraph-Based Dynamic Graph Node Classification 

**Title (ZH)**: 基于超图的动态图节点分类 

**Authors**: Xiaoxu Ma, Chen Zhao, Minglai Shao, Yujie Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.20321)  

**Abstract**: Node classification on static graphs has achieved significant success, but achieving accurate node classification on dynamic graphs where node topology, attributes, and labels change over time has not been well addressed. Existing methods based on RNNs and self-attention only aggregate features of the same node across different time slices, which cannot adequately address and capture the diverse dynamic changes in dynamic graphs. Therefore, we propose a novel model named Hypergraph-Based Multi-granularity Dynamic Graph Node Classification (HYDG). After obtaining basic node representations for each slice through a GNN backbone, HYDG models the representations of each node in the dynamic graph through two modules. The individual-level hypergraph captures the spatio-temporal node representations between individual nodes, while the group-level hypergraph captures the multi-granularity group temporal representations among nodes of the same class. Each hyperedge captures different temporal dependencies of varying lengths by connecting multiple nodes within specific time ranges. More accurate representations are obtained through weighted information propagation and aggregation by the hypergraph neural network. Extensive experiments on five real dynamic graph datasets using two GNN backbones demonstrate the superiority of our proposed framework. 

**Abstract (ZH)**: 静态图上的节点分类已经取得了显著的成果，但在处理节点拓扑、属性和标签随时间发生变化的动态图上的准确节点分类问题上，现有的方法并未得到很好的解决。基于RNN和自注意力机制的方法仅能聚合不同时间切片中同一节点的特征，这无法充分应对和捕捉动态图中多样的动态变化。因此，我们提出了一种名为基于超图的多粒度动态图节点分类（HYDG）的新模型。通过GNN骨干网络获取每个时间切片的基本节点表示后，HYDG通过两个模块对动态图中每个节点的表示进行建模。个体层面的超图捕获个体节点之间的空时节点表示，而群体层面的超图捕获同一类节点之间多粒度的时间序列表示。每条超边通过连接特定时间范围内多个节点，捕捉不同长度的时间依赖性，并通过超图神经网络进行加权信息传播和聚合，从而获得更准确的表示。在两个GNN骨干网络的辅助下，我们对五个实际的动态图数据集进行了广泛的实验，证明了我们所提出框架的优势。 

---
# EXAdam: The Power of Adaptive Cross-Moments 

**Title (ZH)**: EXAdam: 自适应交叉矩的强大力量 

**Authors**: Ahmed M. Adly  

**Link**: [PDF](https://arxiv.org/pdf/2412.20302)  

**Abstract**: This paper introduces EXAdam ($\textbf{EX}$tended $\textbf{Adam}$), a novel optimization algorithm that builds upon the widely-used Adam optimizer. EXAdam incorporates three key enhancements: (1) new debiasing terms for improved moment estimation, (2) a gradient-based acceleration mechanism for increased responsiveness to the current loss landscape, and (3) a dynamic step size formula that allows for continuous growth of the learning rate throughout training. These innovations work synergistically to address limitations of the original Adam algorithm, potentially offering improved convergence properties, enhanced ability to escape saddle points, and greater robustness to hyperparameter choices. I provide a theoretical analysis of EXAdam's components and their interactions, highlighting the algorithm's potential advantages in navigating complex optimization landscapes. Empirical evaluations demonstrate EXAdam's superiority over Adam, achieving 48.07% faster convergence and yielding improvements of 4.6%, 4.13%, and 2.39% in training, validation, and testing accuracies, respectively, when applied to a CNN trained on the CIFAR-10 dataset. While these results are promising, further empirical validation across diverse tasks is essential to fully gauge EXAdam's efficacy. Nevertheless, EXAdam represents a significant advancement in adaptive optimization techniques, with promising implications for a wide range of machine learning applications. This work aims to contribute to the ongoing development of more efficient, adaptive, and universally applicable optimization methods in the field of machine learning and artificial intelligence. 

**Abstract (ZH)**: 本文介绍了EXAdam（扩展的Adam），这是一种基于广泛使用的Adam优化器的新优化算法。EXAdam结合了三项关键增强：（1）新的去偏差项以改进动量估计，（2）基于梯度的加速机制以提高对当前损失景观的响应能力，（3）动态步长公式，允许学习率在整个训练过程中持续增长。这些创新协同作用，解决了一些原始Adam算法的限制，有可能提供更好的收敛性质、增强的鞍点逃逸能力以及对超参数选择的更强鲁棒性。本文对EXAdam的各个组成部分及其相互作用进行了理论分析，强调了该算法在复杂优化景观中导航的潜在优势。实证评估显示，EXAdam在性能上优于Adam，按CIFAR-10数据集上训练的CNN计算，其收敛速度快48.07%，并且在训练、验证和测试精度上分别提高了4.6%、4.13%和2.39%。尽管这些结果充满希望，但在不同任务上的进一步实证验证对于全面评估EXAdam的效果是必要的。不过，EXAdam代表了自适应优化技术的重要进步，具有广泛应用于机器学习和人工智能领域各种应用的前景。本文旨在为机器学习和人工智能领域更高效、更适应性、更广泛适用的优化方法的发展贡献一份力量。 

---
# Transformer-Based Contrastive Meta-Learning For Low-Resource Generalizable Activity Recognition 

**Title (ZH)**: 基于Transformer的对比元学习方法在低资源泛化活动识别中的应用 

**Authors**: Junyao Wang, Mohammad Abdullah Al Faruque  

**Link**: [PDF](https://arxiv.org/pdf/2412.20290)  

**Abstract**: Deep learning has been widely adopted for human activity recognition (HAR) while generalizing a trained model across diverse users and scenarios remains challenging due to distribution shifts. The inherent low-resource challenge in HAR, i.e., collecting and labeling adequate human-involved data can be prohibitively costly, further raising the difficulty of tackling DS. We propose TACO, a novel transformer-based contrastive meta-learning approach for generalizable HAR. TACO addresses DS by synthesizing virtual target domains in training with explicit consideration of model generalizability. Additionally, we extract expressive feature with the attention mechanism of Transformer and incorporate the supervised contrastive loss function within our meta-optimization to enhance representation learning. Our evaluation demonstrates that TACO achieves notably better performance across various low-resource DS scenarios. 

**Abstract (ZH)**: 深度学习在人体活动识别（HAR）中被广泛采用，但在不同的用户和场景下推广训练好的模型依然存在挑战，原因是分布转移问题。人体活动涉及的数据收集和标注成本高昂，本身就构成了一个资源有限的问题，这进一步增加了解决分布转移问题的难度。为此，我们提出了TACO，一种新颖的基于变换器的对比元学习方法，旨在实现可推广的人体活动识别。TACO通过在训练过程中合成虚拟目标域来解决分布转移问题，并明确考虑模型的泛化能力。此外，我们利用变换器的注意力机制提取具有表现力的特征，并在元优化过程中引入监督对比损失函数，以增强表示学习。我们的评估结果证明，TACO在各种低资源分布转移场景下取得了显著的性能提升。 

---
# How To Think About End-To-End Encryption and AI: Training, Processing, Disclosure, and Consent 

**Title (ZH)**: 如何思考端到端加密与AI：训练、处理、披露和同意 

**Authors**: Mallory Knodel, Andrés Fábrega, Daniella Ferrari, Jacob Leiken, Betty Li Hou, Derek Yen, Sam de Alfaro, Kyunghyun Cho, Sunoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2412.20231)  

**Abstract**: End-to-end encryption (E2EE) has become the gold standard for securing communications, bringing strong confidentiality and privacy guarantees to billions of users worldwide. However, the current push towards widespread integration of artificial intelligence (AI) models, including in E2EE systems, raises some serious security concerns. This work performs a critical examination of the (in)compatibility of AI models and E2EE applications. We explore this on two fronts: (1) the integration of AI "assistants" within E2EE applications, and (2) the use of E2EE data for training AI models. We analyze the potential security implications of each, and identify conflicts with the security guarantees of E2EE. Then, we analyze legal implications of integrating AI models in E2EE applications, given how AI integration can undermine the confidentiality that E2EE promises. Finally, we offer a list of detailed recommendations based on our technical and legal analyses, including: technical design choices that must be prioritized to uphold E2EE security; how service providers must accurately represent E2EE security; and best practices for the default behavior of AI features and for requesting user consent. We hope this paper catalyzes an informed conversation on the tensions that arise between the brisk deployment of AI and the security offered by E2EE, and guides the responsible development of new AI features. 

**Abstract (ZH)**: 端到端加密（E2EE）已成为保障通信安全的黄金标准，为全球数十亿用户提供强大的保密性和隐私保护。然而，当前在包括E2EE系统在内的广泛集成人工智能（AI）模型的趋势，引发了严重的安全问题。本文对AI模型和E2EE应用的（不）兼容性进行了关键性审查。我们从两个方面探讨了这一点：（1）在E2EE应用中整合AI“助手”，以及（2）利用E2EE数据训练AI模型。我们分析了每种方法可能的安全影响，指出了与E2EE安全保证的冲突。接着，我们分析了在E2EE应用中集成AI模型的法律影响，考虑到AI集成可能如何损害E2EE承诺的保密性。最后，基于我们的技术和法律分析，我们提出了详细的建议，包括：必须优先考虑的技术设计选择，以确保E2EE安全；服务提供商如何准确代表E2EE安全；以及AI功能的默认行为和请求用户同意的最佳实践。我们希望本文能促进对AI迅猛部署与E2EE提供的安全之间的紧张关系的有见地的讨论，并指导负责任地开发新的AI功能。 

---
# Leveraging Large Language Models for Enhancing Autonomous Vehicle Perception 

**Title (ZH)**: 利用大型语言模型增强自主车辆感知 

**Authors**: Athanasios Karagounis  

**Link**: [PDF](https://arxiv.org/pdf/2412.20230)  

**Abstract**: Autonomous vehicles (AVs) rely on sophisticated perception systems to interpret their surroundings, a cornerstone for safe navigation and decision-making. The integration of Large Language Models (LLMs) into AV perception frameworks offers an innovative approach to address challenges in dynamic environments, sensor fusion, and contextual reasoning. This paper presents a novel framework for incorporating LLMs into AV perception, enabling advanced contextual understanding, seamless sensor integration, and enhanced decision support. Experimental results demonstrate that LLMs significantly improve the accuracy and reliability of AV perception systems, paving the way for safer and more intelligent autonomous driving technologies. By expanding the scope of perception beyond traditional methods, LLMs contribute to creating a more adaptive and human-centric driving ecosystem, making autonomous vehicles more reliable and transparent in their operations. These advancements redefine the relationship between human drivers and autonomous systems, fostering trust through enhanced understanding and personalized decision-making. Furthermore, by integrating memory modules and adaptive learning mechanisms, LLMs introduce continuous improvement in AV perception, enabling vehicles to evolve with time and adapt to changing environments and user preferences. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）依赖于复杂的感知系统来解释其周围环境，这是确保安全导航和决策的基础。将大型语言模型（LLMs）整合到AV感知框架中为应对动态环境、传感器融合和上下文推理的挑战提供了一种创新的方法。本文提出了一种新的框架，用于将LLMs整合到AV感知中，从而实现高级上下文理解、无缝传感器集成和增强的决策支持。实验结果表明，LLMs显著提高了AV感知系统的准确性和可靠性，为更安全、更智能的自主驾驶技术铺平了道路。通过扩展感知范围超越传统方法，LLMs有助于创建一个更具适应性和以人为中心的驾驶生态系统，使自主车辆在操作中更为可靠和透明。这些进步重新定义了人类驾驶员与自主系统之间的关系，通过增强理解和个性化决策来培养信任。此外，通过整合记忆模块和适应性学习机制，LLMs为AV感知提供持续改进的能力，使车辆能够随着时间的推移而演变，并适应不断变化的环境和用户偏好。 

---
# Decoding Emotion: Speech Perception Patterns in Individuals with Self-reported Depression 

**Title (ZH)**: 解码情感：自我报告抑郁个体的语音感知模式 

**Authors**: Guneesh Vats, Priyanka Srivastava, Chiranjeevi Yarra  

**Link**: [PDF](https://arxiv.org/pdf/2412.20213)  

**Abstract**: The current study examines the relationship between self-reported depression and the perception of affective speech within the Indian population. PANAS and PHQ-9 were used to assess current mood and depression, respectively. Participants' emotional reactivity was recorded on a valence and arousal scale against the affective speech audio presented in a sequence. No significant differences between the depression and no-depression groups were observed for any of the emotional stimuli, except the audio file depicting neutral emotion. Significantly higher PANAS scores by the depression than the no-depression group indicate the impact of pre-disposed mood on the current mood status. Contrary to previous findings, this study did not observe reduced positive emotional reactivity by the depression group. However, the results demonstrated consistency in emotional reactivity for speech stimuli depicting sadness and anger across all measures of emotion perception. 

**Abstract (ZH)**: 本研究探讨了自我报告的抑郁症状与印度人群中情感语音感知之间的关系。采用PANAS量表评估当前情绪状态，采用PHQ-9量表评估抑郁水平。参与者的情绪反应按照情感语音音频的正性和唤醒程度进行记录。除中性情感的音频文件外，抑郁组和非抑郁组在所有情感刺激上均未观察到显著差异。抑郁组显著高于非抑郁组的PANAS评分表明，基线情绪对当前情绪状态有影响。与之前的研究结果不符的是，本研究未发现抑郁组存在正向情感反应降低的现象。然而，结果表明，对于表达悲伤和愤怒的情感语音刺激，所有情感感知措施中情绪反应的一致性是存在的。 

---
# Building a Rich Dataset to Empower the Persian Question Answering Systems 

**Title (ZH)**: 构建丰富的数据集以赋能波斯语问答系统 

**Authors**: Mohsen Yazdinejad, Marjan Kaedi  

**Link**: [PDF](https://arxiv.org/pdf/2412.20212)  

**Abstract**: Question answering systems provide short, precise, and specific answers to questions. So far, many robust question answering systems have been developed for English, while some languages with fewer resources, like Persian, have few numbers of standard dataset. In this study, a comprehensive open-domain dataset is presented for Persian. This dataset is called NextQuAD and has 7,515 contexts, including 23,918 questions and answers. Then, a BERT-based question answering model has been applied to this dataset using two pre-trained language models, including ParsBERT and XLM-RoBERTa. The results of these two models have been ensembled using mean logits. Evaluation on the development set shows 0.95 Exact Match (EM) and 0.97 Fl_score. Also, to compare the NextQuAD with other Persian datasets, our trained model on the NextQuAD, is evaluated on two other datasets named PersianQA and ParSQuAD. Comparisons show that the proposed model increased EM by 0.39 and 0.14 respectively in PersianQA and ParSQuAD-manual, while a slight EM decline of 0.007 happened in ParSQuAD-automatic. 

**Abstract (ZH)**: 问答系统能够提供简短、精确且具体的问题答案。到目前为止，已经为英语开发了许多稳健的问答系统，而资源较少的语言，如波斯语，标准数据集的数量则相对较少。在此研究中，我们为波斯语提供了一个全面的开放域数据集。该数据集名为NextQuAD，包含7,515个上下文，包括23,918个问题和答案。然后，我们使用两个预训练语言模型——ParsBERT和XLM-RoBERTa，基于BERT的问答模型对该数据集进行了应用，并使用均值logits对这两种模型的结果进行了集成。在开发集上的评估结果显示其Exact Match (EM) 为0.95，F1_score 为0.97。此外，为了与现有的其他波斯语数据集进行对比，我们使用训练在NextQuAD上的模型对另一个数据集波斯语问答数据集（PersianQA）和帕斯问答数据集（ParSQuAD）进行了评估。对比结果显示，与PersianQA相比，提出的模型分别提高了0.39和0.14的EM分数，而在ParSQuAD-自动版本中略有下降，为0.007分。 

---
# Towards Real-Time 2D Mapping: Harnessing Drones, AI, and Computer Vision for Advanced Insights 

**Title (ZH)**: 面向实时二维制图：利用无人机、人工智能和计算机视觉获取高级洞察 

**Authors**: Bharath Kumar Agnur  

**Link**: [PDF](https://arxiv.org/pdf/2412.20210)  

**Abstract**: Real-time 2D mapping is a vital tool in aerospace and defense, where accurate and timely geographic data is essential for operations like surveillance, reconnaissance, and target tracking. This project introduces a cutting-edge mapping system that integrates drone imagery with machine learning and computer vision to address challenges in processing speed, accuracy, and adaptability to diverse terrains. By automating feature detection, image matching, and stitching, the system generates seamless, high-resolution maps with minimal delay, providing strategic advantages in defense operations.
Implemented in Python, the system leverages OpenCV for image processing, NumPy for efficient computations, and this http URL for parallel processing. ORB (Oriented FAST and Rotated BRIEF) handles feature detection, while FLANN (Fast Library for Approximate Nearest Neighbors) ensures precise keypoint matching. Homography transformations align overlapping images, creating distortion-free maps in real time. This automated approach eliminates manual intervention, enabling live updates critical in dynamic environments. Designed for adaptability, the system performs well under varying light conditions and rugged terrains, making it highly effective in aerospace and defense scenarios. Testing demonstrates significant improvements in speed and accuracy compared to traditional methods, enhancing situational awareness and decision-making. This scalable solution leverages advanced technologies to deliver reliable, actionable data for mission-critical operations. 

**Abstract (ZH)**: 实时二维地图测绘是航空航天和国防领域中的重要工具，其中准确及时的地理数据对于监视、侦察和目标跟踪等操作至关重要。本项目介绍了一种集成无人机影像与机器学习和计算机视觉的先进技术系统，以解决处理速度、准确性和地形适应性等方面的挑战。通过自动化特征检测、图像匹配和缝合，该系统能够生成无缝、高分辨率地图，且延迟极小，为防御操作提供了战略优势。

该系统用Python实现，并利用OpenCV进行图像处理、NumPy进行高效计算，同时采用以下网址中的并行处理技术。ORB（定向FAST和旋转BRIEF）处理特征检测，FLANN（快速近似最近邻库）确保精确的关键点匹配。Homography变换对接边图像，实现实时无失真的地图生成。这种自动化方法消除了人工干预，能够在动态环境中提供关键的实时更新。该系统设计用于适应各种光照条件和崎岖地形，使其在航空航天和国防场景中具备高效性。测试结果显示，与传统方法相比，该系统在速度和准确性上有了显著提升，增强了态势感知和决策能力。该可扩展的解决方案利用先进技术提供可靠的操作数据，支持关键任务操作。 

---
# Injecting Explainability and Lightweight Design into Weakly Supervised Video Anomaly Detection Systems 

**Title (ZH)**: 将可解释性和轻量级设计注入弱监督视频异常检测系统 

**Authors**: Wen-Dong Jiang, Chih-Yung Chang, Hsiang-Chuan Chang, Ji-Yuan Chen, Diptendu Sinha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2412.20201)  

**Abstract**: Weakly Supervised Monitoring Anomaly Detection (WSMAD) utilizes weak supervision learning to identify anomalies, a critical task for smart city monitoring. However, existing multimodal approaches often fail to meet the real-time and interpretability requirements of edge devices due to their complexity. This paper presents TCVADS (Two-stage Cross-modal Video Anomaly Detection System), which leverages knowledge distillation and cross-modal contrastive learning to enable efficient, accurate, and interpretable anomaly detection on edge this http URL operates in two stages: coarse-grained rapid classification and fine-grained detailed analysis. In the first stage, TCVADS extracts features from video frames and inputs them into a time series analysis module, which acts as the teacher model. Insights are then transferred via knowledge distillation to a simplified convolutional network (student model) for binary classification. Upon detecting an anomaly, the second stage is triggered, employing a fine-grained multi-class classification model. This stage uses CLIP for cross-modal contrastive learning with text and images, enhancing interpretability and achieving refined classification through specially designed triplet textual relationships. Experimental results demonstrate that TCVADS significantly outperforms existing methods in model performance, detection efficiency, and interpretability, offering valuable contributions to smart city monitoring applications. 

**Abstract (ZH)**: 弱监督跨模态视频异常检测（WSMAD）利用弱监督学习来识别异常，这是智能城市监控中的一个关键任务。然而，现有的多模态方法由于其复杂性，往往无法满足边缘设备的实时性和可解释性要求。本文提出了两阶段跨模态视频异常检测系统（TCVADS），该系统结合知识蒸馏和跨模态对比学习，能够在边缘设备上实现高效的、准确的和可解释的异常检测。TCVADS 操作分为两个阶段：粗粒度快速分类和细粒度详细分析。在第一阶段，TCVADS 从视频帧中提取特征并将这些特征输入时间序列分析模块（教师模型）。然后，通过知识蒸馏将这些见解转移到简化卷积网络（学生模型）中进行二分类。在检测到异常后，触发第二阶段，使用细粒度的多分类模型。该阶段采用 CLIP 进行跨模态对比学习（结合文本和图像），并通过特别设计的三元组文本关系实现更精细的分类，从而提高可解释性。实验结果表明，TCVADS 在模型性能、检测效率和可解释性方面显著优于现有方法，为智能城市监控应用提供了有价值的贡献。 

---
# Federated Unlearning with Gradient Descent and Conflict Mitigation 

**Title (ZH)**: 联邦卸载：基于梯度下降和冲突缓解的方法 

**Authors**: Zibin Pan, Zhichao Wang, Chi Li, Kaiyan Zheng, Boqi Wang, Xiaoying Tang, Junhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.20200)  

**Abstract**: Federated Learning (FL) has received much attention in recent years. However, although clients are not required to share their data in FL, the global model itself can implicitly remember clients' local data. Therefore, it's necessary to effectively remove the target client's data from the FL global model to ease the risk of privacy leakage and implement ``the right to be forgotten". Federated Unlearning (FU) has been considered a promising way to remove data without full retraining. But the model utility easily suffers significant reduction during unlearning due to the gradient conflicts. Furthermore, when conducting the post-training to recover the model utility, the model is prone to move back and revert what has already been unlearned. To address these issues, we propose Federated Unlearning with Orthogonal Steepest Descent (FedOSD). We first design an unlearning Cross-Entropy loss to overcome the convergence issue of the gradient ascent. A steepest descent direction for unlearning is then calculated in the condition of being non-conflicting with other clients' gradients and closest to the target client's gradient. This benefits to efficiently unlearn and mitigate the model utility reduction. After unlearning, we recover the model utility by maintaining the achievement of unlearning. Finally, extensive experiments in several FL scenarios verify that FedOSD outperforms the SOTA FU algorithms in terms of unlearning and model utility. 

**Abstract (ZH)**: 联邦学习（FL）近年来引起了广泛关注。然而，尽管客户端在FL中不需要共享其数据，但全局模型本身仍可能隐含地记得客户端的本地数据。因此，有必要有效移除目标客户端的数据以减轻隐私泄露的风险并实现“被遗忘的权利”。联邦遗忘（FU）被认为是移除数据的一种有前景的方法，但因梯度冲突的问题，模型可用性易受到显著降低的影响。进一步，在进行后续训练以恢复模型可用性时，模型容易向原先已遗忘的方向回归。为解决这些问题，我们提出了联邦遗忘与正交最速下降（FedOSD）方法。我们首先设计了一种遗忘交叉熵损失以克服梯度上升的收敛问题。然后，在不与其他客户端的梯度发生冲突且与目标客户端的梯度最接近的条件下，计算出一种最速下降方向。这有助于高效地进行遗忘操作并减轻模型可用性的降低。在进行遗忘操作后，通过保持遗忘的成果来恢复模型可用性。最后，通过在多种联邦学习场景下进行的广泛实验表明，在遗忘和模型可用性方面，FedOSD在性能上优于当前最先进（SOTA）的联邦遗忘算法。 

---
# Lower bounds on transformers with infinite precision 

**Title (ZH)**: 具有无限精度的Transformer下的下界研究 

**Authors**: Alexander Kozachinskiy  

**Link**: [PDF](https://arxiv.org/pdf/2412.20195)  

**Abstract**: In this note, we use the VC dimension technique to prove the first lower bound against one-layer softmax transformers with infinite precision. We do so for two tasks: function composition, considered by Peng, Narayanan, and Papadimitriou, and the SUM$_2$ task, considered by Sanford, Hsu, and Telgarsky. 

**Abstract (ZH)**: 在本文中，我们使用VC维技术证明了一层softmax变换器在无限精度情况下的首个下界。我们针对两个任务进行了这项研究：第一个任务是函数组合，被彭、纳拉扬南和帕达米 triou 考虑过；第二个任务是SUM$_2$任务，被桑德福、胡和特尔加斯基考虑过。 

---
# Imitation Learning from Suboptimal Demonstrations via Meta-Learning An Action Ranker 

**Title (ZH)**: 通过元学习构建动作排名器来从次优示范进行 imitation 学习 

**Authors**: Jiangdong Fan, Hongcai He, Paul Weng, Hui Xu, Jie Shao  

**Link**: [PDF](https://arxiv.org/pdf/2412.20193)  

**Abstract**: A major bottleneck in imitation learning is the requirement of a large number of expert demonstrations, which can be expensive or inaccessible. Learning from supplementary demonstrations without strict quality requirements has emerged as a powerful paradigm to address this challenge. However, previous methods often fail to fully utilize their potential by discarding non-expert data. Our key insight is that even demonstrations that fall outside the expert distribution but outperform the learned policy can enhance policy performance. To utilize this potential, we propose a novel approach named imitation learning via meta-learning an action ranker (ILMAR). ILMAR implements weighted behavior cloning (weighted BC) on a limited set of expert demonstrations along with supplementary demonstrations. It utilizes the functional of the advantage function to selectively integrate knowledge from the supplementary demonstrations. To make more effective use of supplementary demonstrations, we introduce meta-goal in ILMAR to optimize the functional of the advantage function by explicitly minimizing the distance between the current policy and the expert policy. Comprehensive experiments using extensive tasks demonstrate that ILMAR significantly outperforms previous methods in handling suboptimal demonstrations. Code is available at this https URL. 

**Abstract (ZH)**: 模仿学习的一个主要瓶颈是需要大量的专家演示，这可能会非常昂贵或难以获得。在没有严格质量要求的情况下学习辅助演示已经成为应对这一挑战的强大范式。然而，之前的方法往往未能充分利用辅助演示的潜力，因为它们经常忽略了非专家数据。我们提出的见解是，即使那些超出专家分布但能够超越所学策略的演示也能增强策略性能。为了利用这一潜力，我们提出了一种新的方法，名为元学习动作排名的模仿学习（ILMAR）。ILMAR 在有限的专家演示和辅助演示上实现了加权行为克隆（加权 BC），并利用优势函数的功能来选择性地整合辅助演示的知识。为了更有效地利用辅助演示，我们在 ILMAR 中引入了元目标（meta-goal），通过显式地最小化当前策略和专家策略之间的距离来优化优势函数的功能。通过广泛的实验任务进行的综合实验表明，ILMAR 显著优于之前的方法，能够更好地处理次优演示。代码获取地址为：这个链接。 

---
# Real-time Calibration Model for Low-cost Sensor in Fine-grained Time series 

**Title (ZH)**: 低成本传感器在细粒度时间序列中的实时校准模型 

**Authors**: Seokho Ahn, Hyungjin Kim, Sungbok Shin, Young-Duk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20170)  

**Abstract**: Precise measurements from sensors are crucial, but data is usually collected from low-cost, low-tech systems, which are often inaccurate. Thus, they require further calibrations. To that end, we first identify three requirements for effective calibration under practical low-tech sensor conditions. Based on the requirements, we develop a model called TESLA, Transformer for effective sensor calibration utilizing logarithmic-binned attention. TESLA uses a high-performance deep learning model, Transformers, to calibrate and capture non-linear components. At its core, it employs logarithmic binning to minimize attention complexity. TESLA achieves consistent real-time calibration, even with longer sequences and finer-grained time series in hardware-constrained systems. Experiments show that TESLA outperforms existing novel deep learning and newly crafted linear models in accuracy, calibration speed, and energy efficiency. 

**Abstract (ZH)**: 传感器精确测量至关重要，但在实际应用中，数据通常是由低成本、技术简单的系统收集的，这些系统往往不够准确，因而需要进一步校准。为此，我们首先识别了在实际低成本传感器条件下进行有效校准的三个要求。基于这些要求，我们开发了一个名为TESLA（Transformer for Effective Sensor Calibration Utilizing Logarithmic-Binned Attention）的模型。TESLA利用Transformer模型进行校准和捕捉非线性成分，并通过使用对数间隔分箱来最小化注意力复杂度。即使在硬件受限的系统中，TESLA也能实现一致的实时校准，即使在较长的序列和更精细的时间序列下亦然。实验表明，相比现有的新颖深度学习模型和新设计的线性模型，TESLA在准确度、校准速度和能效方面均表现出色。 

---
# LoL-PIM: Long-Context LLM Decoding with Scalable DRAM-PIM System 

**Title (ZH)**: LoL-PIM：具有可扩展DRAM-PIM系统的长上下文LLM解码 

**Authors**: Hyucksung Kwon, Kyungmo Koo, Janghyeon Kim, Woongkyu Lee, Minjae Lee, Hyungdeok Lee, Yousub Jung, Jaehan Park, Yosub Song, Byeongsu Yang, Haerang Choi, Guhyun Kim, Jongsoon Won, Woojae Shin, Changhyun Kim, Gyeongcheol Shin, Yongkee Kwon, Ilkon Kim, Euicheol Lim, John Kim, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2412.20166)  

**Abstract**: The expansion of large language models (LLMs) with hundreds of billions of parameters presents significant challenges to computational resources, particularly data movement and memory bandwidth. Long-context LLMs, which process sequences of tens of thousands of tokens, further increase the demand on the memory system as the complexity in attention layers and key-value cache sizes is proportional to the context length. Processing-in-Memory (PIM) maximizes memory bandwidth by moving compute to the data and can address the memory bandwidth challenges; however, PIM is not necessarily scalable to accelerate long-context LLM because of limited per-module memory capacity and the inflexibility of fixed-functional unit PIM architecture and static memory management. In this work, we propose LoL-PIM which is a multi-node PIM architecture that accelerates long context LLM through hardware-software co-design. In particular, we propose how pipeline parallelism can be exploited across a multi-PIM module while a direct PIM access (DPA) controller (or DMA for PIM) is proposed that enables dynamic PIM memory management and results in efficient PIM utilization across a diverse range of context length. We developed an MLIR-based compiler for LoL-PIM extending a commercial PIM-based compiler where the software modifications were implemented and evaluated, while the hardware changes were modeled in the simulator. Our evaluations demonstrate that LoL-PIM significantly improves throughput and reduces latency for long-context LLM inference, outperforming both multi-GPU and GPU-PIM systems (up to 8.54x and 16.0x speedup, respectively), thereby enabling more efficient deployment of LLMs in real-world applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）参数量达到数十亿的扩展给计算资源带来了显著挑战，尤其是数据传输和内存带宽方面。处理数十万令牌的长上下文LLMs进一步增加了对内存系统的依赖，因为注意力层和键值缓存大小的复杂性与上下文长度成正比。计算-内存协同处理（Processing-in-Memory, PIM）可以通过将计算移到数据上来最大化内存带宽，并解决内存带宽挑战；然而，由于每个模块的内存容量有限以及固定功能单元PIM架构和静态内存管理的灵活性不足，PIM不一定能够加速长上下文LLMs。在此项工作中，我们提出了一种名为LoL-PIM的多节点PIM架构，通过硬件-软件协同设计来加速长上下文LLMs。我们提出了如何在多PIM模块之间利用流水线并行性，并提出了一种直接PIM访问（DPA）控制器（或PIM的DMA控制器），该控制器能够实现动态PIM内存管理，从而在不同上下文长度范围内有效地利用PIM。我们基于MLIR开发了一个LoL-PIM编译器，扩展了商用PIM基编译器的功能，在软件修改上进行了实现和评估，而硬件变更则在模拟器中建模。我们的评估结果表明，LoL-PIM在长上下文LLMs推理中显著提高了吞吐量并降低了延迟，相较于多GPU系统和GPU-PIM系统分别实现了最高8.54倍和16.0倍的速度提升，从而使得在实际应用中能够更有效地部署LLMs。 

---
# StyleAutoEncoder for manipulating image attributes using pre-trained StyleGAN 

**Title (ZH)**: 使用预训练StyleGAN操控图像属性的StyleAutoEncoder 

**Authors**: Andrzej Bedychaj, Jacek Tabor, Marek Śmieja  

**Link**: [PDF](https://arxiv.org/pdf/2412.20164)  

**Abstract**: Deep conditional generative models are excellent tools for creating high-quality images and editing their attributes. However, training modern generative models from scratch is very expensive and requires large computational resources. In this paper, we introduce StyleAutoEncoder (StyleAE), a lightweight AutoEncoder module, which works as a plugin for pre-trained generative models and allows for manipulating the requested attributes of images. The proposed method offers a cost-effective solution for training deep generative models with limited computational resources, making it a promising technique for a wide range of applications. We evaluate StyleAutoEncoder by combining it with StyleGAN, which is currently one of the top generative models. Our experiments demonstrate that StyleAutoEncoder is at least as effective in manipulating image attributes as the state-of-the-art algorithms based on invertible normalizing flows. However, it is simpler, faster, and gives more freedom in designing neural 

**Abstract (ZH)**: 深度条件生成模型是创造高质量图像和编辑其属性的优秀工具。然而，从头训练现代生成模型非常昂贵且需要大量计算资源。本文我们提出了StyleAutoEncoder（StyleAE），一个轻量级的自编码器模块，它可以作为一个插件用于预训练的生成模型，并允许用户操作所需图像的属性。所提出的方法为在有限计算资源下训练深度生成模型提供了成本效益高的解决方案，使其成为广泛应用场景中的有前途的技术。我们通过将StyleAutoEncoder与当前领先的生成模型之一StyleGAN结合来评估该方法。实验结果表明，StyleAutoEncoder在操作图像属性方面至少与基于可逆归一化流的最新算法一样有效。然而，它更简单、更快，并且在设计神经网络方面提供了更大的灵活性。 

---
# Topic-Aware Knowledge Graph with Large Language Models for Interoperability in Recommender Systems 

**Title (ZH)**: 基于主题的大型语言模型知识图谱：推荐系统中的 interoperability 应用 

**Authors**: Minhye Jeon, Seokho Ahn, Young-Duk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20163)  

**Abstract**: The use of knowledge graphs in recommender systems has become one of the common approaches to addressing data sparsity and cold start problems. Recent advances in large language models (LLMs) offer new possibilities for processing side and context information within knowledge graphs. However, consistent integration across various systems remains challenging due to the need for domain expert intervention and differences in system characteristics. To address these issues, we propose a consistent approach that extracts both general and specific topics from both side and context information using LLMs. First, general topics are iteratively extracted and updated from side information. Then, specific topics are extracted using context information. Finally, to address synonymous topics generated during the specific topic extraction process, a refining algorithm processes and resolves these issues effectively. This approach allows general topics to capture broad knowledge across diverse item characteristics, while specific topics emphasize detailed attributes, providing a more comprehensive understanding of the semantic features of items and the preferences of users. Experimental results demonstrate significant improvements in recommendation performance across diverse knowledge graphs. 

**Abstract (ZH)**: 知识图谱在推荐系统中的应用已成为解决数据稀疏性和冷启动问题的一种常见方法。大型语言模型（LLMs）的最新进展为处理知识图谱内的侧信息和上下文信息提供了新的可能性。然而，由于需要领域专家的干预以及系统特性的差异，一致性的整合仍面临挑战。为了解决这些问题，我们提出了一种一致的方法，利用LLMs从侧信息和上下文信息中提取一般和特定的主题。首先，通过迭代提取和更新，从侧信息中提取一般主题。然后，利用上下文信息提取特定主题。最后，针对特定主题提取过程中生成的同义主题，提出了一种改进算法，有效处理和解决这些问题。这种方法使得一般主题能够捕捉不同物品特性下的广泛知识，而特定主题则强调详细的属性，从而提供对物品语义特征和用户偏好更全面的理解。实验结果证明，这种方法在不同知识图谱中显著提高了推荐性能。 

---
# Stable-TTS: Stable Speaker-Adaptive Text-to-Speech Synthesis via Prosody Prompting 

**Title (ZH)**: 稳定-TTS：通过韵律提示实现稳定的说话人自适应文本到语音合成 

**Authors**: Wooseok Han, Minki Kang, Changhun Kim, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20155)  

**Abstract**: Speaker-adaptive Text-to-Speech (TTS) synthesis has attracted considerable attention due to its broad range of applications, such as personalized voice assistant services. While several approaches have been proposed, they often exhibit high sensitivity to either the quantity or the quality of target speech samples. To address these limitations, we introduce Stable-TTS, a novel speaker-adaptive TTS framework that leverages a small subset of a high-quality pre-training dataset, referred to as prior samples. Specifically, Stable-TTS achieves prosody consistency by leveraging the high-quality prosody of prior samples, while effectively capturing the timbre of the target speaker. Additionally, it employs a prior-preservation loss during fine-tuning to maintain the synthesis ability for prior samples to prevent overfitting on target samples. Extensive experiments demonstrate the effectiveness of Stable-TTS even under limited amounts of and noisy target speech samples. 

**Abstract (ZH)**: 基于说话人的文本到语音（TTS）合成因其广泛的应用领域，如个性化语音助手服务，引起了广泛关注。尽管已有多种方法被提出，但它们通常对目标语音样本的数量或质量表现出高度敏感性。为了解决这些限制，我们引入了稳定-TTS（Stable-TTS），这是一种新颖的基于说话人的TTS框架，该框架利用了一小部分高质前训练数据集，称为先验样本。具体而言，稳定-TTS通过利用先验样本的高质量语调实现语调一致性，同时有效地捕捉目标说话人的音色。此外，在微调过程中还采用了一种先验保真损失，以保持对先验样本的合成能力，从而防止过度拟合目标样本。大量实验表明，即使在目标语音样本数量有限且带噪声的情况下，稳定-TTS也能显示出其有效性。 

---
# TradingAgents: Multi-Agents LLM Financial Trading Framework 

**Title (ZH)**: 交易代理：多智能体LLM金融交易框架

注：在这个翻译中，“TradingAgents”被译为“交易代理”，“Multi-Agents”被译为“多智能体”，“LLM”被解释为“大语言模型”，考虑到具体上下文，“LLM”也有可能指的是“长期记忆模型”或其他特定含义，需要根据实际场景进一步确认。此处保持了“LLM”未译，保持原文中的缩写形式，同时在翻译中注释其可能的含义。总体来说，整句话翻译保持了原文的学术风格和专业术语。 

**Authors**: Yijia Xiao, Edward Sun, Di Luo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20138)  

**Abstract**: Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. 

**Abstract (ZH)**: 在使用大规模语言模型（LLMs）驱动的代理社会进行自动化问题解决方面已经取得显著进展。在金融领域，大部分努力主要集中在处理特定任务的单代理系统或独立收集数据的多代理框架上。然而，多代理系统在复制现实世界交易公司的协同动态方面具有巨大潜力，这一领域尚未充分探索。TradingAgents 提出了一种受交易公司启发的新型股票交易框架，该框架中的代理由LLM驱动，扮演不同的专业角色，如基本面分析师、情绪分析师、技术分析师和不同风险偏好级别的交易者。该框架包括牛市和熊市研究员代理评估市场状况，风险管理部门监控风险敞口，以及交易者通过辩论和历史数据综合获得的见解来做决策。通过模拟动态且协作的交易环境，该框架旨在提高交易表现。详细的架构和大量实验表明，与基准模型相比，该框架在累积回报、夏普比率和最大回撤等方面具有显著优势，这表明多代理LLM框架在金融市场交易中的潜力。 

---
# M-MAD: Multidimensional Multi-Agent Debate Framework for Fine-grained Machine Translation Evaluation 

**Title (ZH)**: M-MAD：细粒度机器翻译评估的多维度多代理辩论框架 

**Authors**: Zhaopeng Feng, Jiayuan Su, Jiamei Zheng, Jiahan Ren, Yan Zhang, Jian Wu, Hongwei Wang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20127)  

**Abstract**: Recent advancements in large language models (LLMs) have given rise to the LLM-as-a-judge paradigm, showcasing their potential to deliver human-like judgments. However, in the field of machine translation (MT) evaluation, current LLM-as-a-judge methods fall short of learned automatic metrics. In this paper, we propose Multidimensional Multi-Agent Debate (M-MAD), a systematic LLM-based multi-agent framework for advanced LLM-as-a-judge MT evaluation. Our findings demonstrate that M-MAD achieves significant advancements by (1) decoupling heuristic MQM criteria into distinct evaluation dimensions for fine-grained assessments; (2) employing multi-agent debates to harness the collaborative reasoning capabilities of LLMs; (3) synthesizing dimension-specific results into a final evaluation judgment to ensure robust and reliable outcomes. Comprehensive experiments show that M-MAD not only outperforms all existing LLM-as-a-judge methods but also competes with state-of-the-art reference-based automatic metrics, even when powered by a suboptimal model like GPT-4o mini. Detailed ablations and analysis highlight the superiority of our framework design, offering a fresh perspective for LLM-as-a-judge paradigm. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的最新进展催生了LLM-as-a-judge的范式，展示了它们在提供类人类判断方面的潜力。然而，在机器翻译（MT）评估领域，当前的LLM-as-a-judge方法未能超越已学习的自动度量标准。在本文中，我们提出了一种基于LLM的多维度多代理辩论框架（M-MAD），这是一种系统化的多代理框架，用于高级LLM-as-a-judge的MT评估。我们的研究结果显示，M-MAD通过以下三个方面实现了显著的进步：（1）将启发式MQM标准拆分为不同的评估维度，以实现精细评估；（2）利用多代理辩论来发挥LLMs的协作推理能力；（3）将各维度的具体结果综合成最终评估判断，以确保结果的稳健性和可靠性。综合实验表明，M-MAD不仅超过了所有现有的LLM-as-a-judge方法，而且即使在使用像GPT-4o mini这样的次优模型时，也能与最先进的基于参考的自动度量标准相媲美。详细的消融实验和分析突显了我们框架设计的优势，为LLM-as-a-judge范式提供了一个新的视角。我们的代码和数据已公开，可通过以下链接访问：[此处插入链接] 

---
# SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis 

**Title (ZH)**: SynDiff：同步运动扩散在多体人体-物体交互合成中的应用 

**Authors**: Wenkun He, Yun Liu, Ruitao Liu, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2412.20104)  

**Abstract**: Synthesizing realistic human-object interaction motions is a critical problem in VR/AR and human animation. Unlike the commonly studied scenarios involving a single human or hand interacting with one object, we address a more generic multi-body setting with arbitrary numbers of humans, hands, and objects. This complexity introduces significant challenges in synchronizing motions due to the high correlations and mutual influences among bodies. To address these challenges, we introduce SyncDiff, a novel method for multi-body interaction synthesis using a synchronized motion diffusion strategy. SyncDiff employs a single diffusion model to capture the joint distribution of multi-body motions. To enhance motion fidelity, we propose a frequency-domain motion decomposition scheme. Additionally, we introduce a new set of alignment scores to emphasize the synchronization of different body motions. SyncDiff jointly optimizes both data sample likelihood and alignment likelihood through an explicit synchronization strategy. Extensive experiments across four datasets with various multi-body configurations demonstrate the superiority of SyncDiff over existing state-of-the-art motion synthesis methods. 

**Abstract (ZH)**: 在虚拟现实/增强现实（VR/AR）和人体动画中，合成真实的人机交互动作是一个关键问题。不同于仅涉及单一人体或双手与一个物体交互的研究场景，我们关注更具通用性的多主体设置，其中包括任意数量的人体、手和物体。这种设置增加了同步动作的复杂性，因为它涉及到各主体间的高相关性和相互影响。为应对这些挑战，我们提出了一种名为SyncDiff的新方法，该方法采用同步运动扩散策略进行多主体交互合成。SyncDiff使用单一的扩散模型来捕捉多主体动作的联合分布。为了提高动作的保真度，我们提出了一种频率域运动分解方案。此外，我们引入了一组新的对齐评分，以强调不同主体动作的同步性。通过显式的同步策略，SyncDiff同时优化数据样本似然性和对齐似然性。在包含不同多主体配置的四个数据集上的广泛实验表明，与现有的最先进的动作合成方法相比，SyncDiff具有显著的优势。 

---
# RFPPO: Motion Dynamic RRT based Fluid Field - PPO for Dynamic TF/TA Routing Planning 

**Title (ZH)**: RFPPO：基于运动动态RRT的流场-强化学习（PPO）动态TF/TA 路由规划方法 

**Authors**: Rongkun Xue, Jing Yang, Yuyang Jiang, Yiming Feng, Zi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20098)  

**Abstract**: Existing local dynamic route planning algorithms, when directly applied to terrain following/terrain avoidance, or dynamic obstacle avoidance for large and medium-sized fixed-wing aircraft, fail to simultaneously meet the requirements of real-time performance, long-distance planning, and the dynamic constraints of large and medium-sized aircraft. To deal with this issue, this paper proposes the Motion Dynamic RRT based Fluid Field - PPO for dynamic TF/TA routing planning. Firstly, the action and state spaces of the proximal policy gradient algorithm are redesigned using disturbance flow fields and artificial potential field algorithms, establishing an aircraft dynamics model, and designing a state transition process based on this model. Additionally, a reward function is designed to encourage strategies for obstacle avoidance, terrain following, terrain avoidance, and safe flight. Experimental results on real DEM data demonstrate that our algorithm can complete long-distance flight tasks through collision-free trajectory planning that complies with dynamic constraints, without the need for prior global planning. 

**Abstract (ZH)**: 现有的局部动态路径规划算法在应用于地形跟随/地形避免或大型和中型固定翼飞机的动态障碍物避免时，无法同时满足实时性能、长距离规划和大型和中型飞机的动态约束要求。为解决这一问题，本文提出了一种基于流场的Motion Dynamic RRT与PPO相结合的动态地形跟随/地形避免路径规划方法。首先，利用干扰流场和人工势场算法重新设计了扰动流场算法和策略梯度算法的动作和状态空间，并建立了飞机动力学模型，在此基础上设计了状态转移过程。此外，设计了一个奖励函数来促进障碍物避免、地形跟随、地形避免和安全飞行的策略。实验结果表明，我们的算法能够在遵循动态约束条件下通过无碰撞轨迹规划完成长距离飞行任务，无需进行先验全局规划。 

---
# From Worms to Mice: Homeostasis Maybe All You Need 

**Title (ZH)**: 从蠕虫到老鼠：或许一切你都需要是稳态

这个标题翻译成中文后，保持了原意，并采用了更符合中文表达习惯的句式。如有更多具体内容需要翻译或进一步的修改，请告知。 

**Authors**: Jesus Marco de Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2412.20090)  

**Abstract**: In this brief and speculative commentary, we explore ideas inspired by neural networks in machine learning, proposing that a simple neural XOR motif, involving both excitatory and inhibitory connections, may provide the basis for a relevant mode of plasticity in neural circuits of living organisms, with homeostasis as the sole guiding principle. This XOR motif simply signals the discrepancy between incoming signals and reference signals, thereby providing a basis for a loss function in learning neural circuits, and at the same time regulating homeostasis by halting the propagation of these incoming signals. The core motif uses a 4:1 ratio of excitatory to inhibitory neurons, and supports broader neural patterns such as the well-known 'winner takes all' (WTA) mechanism. We examined the prevalence of the XOR motif in the published connectomes of various organisms with increasing complexity, and found that it ranges from tens (in C. elegans) to millions (in several Drosophila neuropils) and more than tens of millions (in mouse V1 visual cortex). If validated, our hypothesis identifies two of the three key components in analogy to machine learning models: the architecture and the loss function. And we propose that a relevant type of biological neural plasticity is simply driven by a basic control or regulatory system, which has persisted and adapted despite the increasing complexity of organisms throughout evolution. 

**Abstract (ZH)**: 在本文中，我们将探讨受机器学习中神经网络启发的想法，提出一个包含兴奋性和抑制性连接的简单神经XOR模体可能为基础生物体神经回路中的某种相关可塑性模式提供基础，而稳态则是唯一的指导原则。这个XOR模体简单地表示输入信号与参考信号之间的差异，从而为学习神经回路中的损失函数提供基础，并同时通过停止这些输入信号的传播来调节稳态。该核心模体采用兴奋性和抑制性神经元4:1的比例，并支持诸如广为人知的“一胜俱输”（Winner-Take-All，WTA）机制等更广泛神经模式。我们检查了各种复杂性不断增加的生物中公布的联接图中XOR模体的普遍存在程度，发现从秀丽隐杆线虫中的几十个，到果蝇的数百万个，再到小鼠初级视觉皮层中超过数千万个。如果这一假设得到验证，我们的假说将识别出模拟机器学习模型的三个关键组成部分中的两种：架构和损失函数。并且我们提出，一种相关类型的生物神经可塑性仅仅由一种基本的控制系统驱动，该系统在进化过程中尽管生物体复杂性不断增加，但仍然得以延续并适应。 

---
# An archaeological Catalog Collection Method Based on Large Vision-Language Models 

**Title (ZH)**: 基于大型视觉-语言模型的考古藏品目录编制方法 

**Authors**: Honglin Pang, Yi Chang, Tianjing Duan, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20088)  

**Abstract**: Archaeological catalogs, containing key elements such as artifact images, morphological descriptions, and excavation information, are essential for studying artifact evolution and cultural inheritance. These data are widely scattered across publications, requiring automated collection methods. However, existing Large Vision-Language Models (VLMs) and their derivative data collection methods face challenges in accurate image detection and modal matching when processing archaeological catalogs, making automated collection difficult. To address these issues, we propose a novel archaeological catalog collection method based on Large Vision-Language Models that follows an approach comprising three modules: document localization, block comprehension and block matching. Through practical data collection from the Dabagou and Miaozigou pottery catalogs and comparison experiments, we demonstrate the effectiveness of our approach, providing a reliable solution for automated collection of archaeological catalogs. 

**Abstract (ZH)**: 考古目录包含器物图像、形态描述和发掘信息等关键要素，对于研究器物演进和文化传承至关重要。这些数据广泛分散在各种出版物上，需要自动化的数据收集方法。然而，现有的大型视觉-语言模型（VLMs）及其衍生的数据收集方法在处理考古目录时面临着精准图像检测和模态匹配的挑战，导致自动收集变得困难。为了解决这些问题，我们提出了一种基于大型视觉-语言模型的新型考古目录收集方法，该方法包含三个模块：文档定位、块理解与块匹配。通过对达包口和苗子沟陶器目录的实际数据收集和对比实验，我们证明了该方法的有效性，提供了一种可靠的方法来实现考古目录的自动化收集。 

---
# On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs 

**Title (ZH)**: 针对LLM的对抗性攻击中传统脆弱性评分系统的有效性研究 

**Authors**: Atmane Ayoub Mansour Bahar, Ahmad Samer Wazan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20087)  

**Abstract**: This research investigates the effectiveness of established vulnerability metrics, such as the Common Vulnerability Scoring System (CVSS), in evaluating attacks against Large Language Models (LLMs), with a focus on Adversarial Attacks (AAs). The study explores the influence of both general and specific metric factors in determining vulnerability scores, providing new perspectives on potential enhancements to these metrics.
This study adopts a quantitative approach, calculating and comparing the coefficient of variation of vulnerability scores across 56 adversarial attacks on LLMs. The attacks, sourced from various research papers, and obtained through online databases, were evaluated using multiple vulnerability metrics. Scores were determined by averaging the values assessed by three distinct LLMs. The results indicate that existing scoring-systems yield vulnerability scores with minimal variation across different attacks, suggesting that many of the metric factors are inadequate for assessing adversarial attacks on LLMs. This is particularly true for context-specific factors or those with predefined value sets, such as those in CVSS. These findings support the hypothesis that current vulnerability metrics, especially those with rigid values, are limited in evaluating AAs on LLMs, highlighting the need for the development of more flexible, generalized metrics tailored to such attacks.
This research offers a fresh analysis of the effectiveness and applicability of established vulnerability metrics, particularly in the context of Adversarial Attacks on Large Language Models, both of which have gained significant attention in recent years. Through extensive testing and calculations, the study underscores the limitations of these metrics and opens up new avenues for improving and refining vulnerability assessment frameworks specifically tailored for LLMs. 

**Abstract (ZH)**: 本研究探讨了通用漏洞评估指标，如通用漏洞评分系统（CVSS）在评估大语言模型（LLMs）攻击中的有效性，特别是对抗性攻击（Adversarial Attacks, AAs）。研究探索了通用和特定指标因素对确定漏洞评分的影响，提供了改进现有指标的新视角。

本研究采用定量方法，计算并比较了56种不同的对抗性攻击对LLMs进行评估时的漏洞评分的变异系数。所使用的攻击数据来自多篇研究论文，并通过在线数据库获取。多种漏洞评估指标被用于评估每个攻击的评分。评分是根据三个不同的LLMs评估值的平均值得出。结果表明，现有的评分系统在不同攻击下产生的漏洞评分变化很小，说明许多指标因素不足以评估针对LLMs的对抗性攻击。这尤其适用于情境特定的因素或具有预定值集的因素，例如CVSS中的因素。这些发现支持了当前漏洞评估指标，特别是那些具有刚性值的指标，在评估LLMs的对抗性攻击时存在局限性的假设，强调了开发更加灵活和普适的指标以适应此类攻击的需求。

本研究提供了一种全新的分析方法，以评估和应用现有的漏洞评估指标，特别是在针对大语言模型的对抗性攻击方面，这种类型的攻击近年来受到了广泛关注。通过广泛的测试和计算，该研究突显了这些指标的局限性，并为针对LLMs而定制的漏洞评估框架的改进和优化打开了新的途径。 

---
# MAFT: Efficient Model-Agnostic Fairness Testing for Deep Neural Networks via Zero-Order Gradient Search 

**Title (ZH)**: MAFT：通过零阶梯度搜索实现的面向深度神经网络的高效模型不可知公平性测试 

**Authors**: Zhaohui Wang, Min Zhang, Jingran Yang, Bojie Shao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20086)  

**Abstract**: Deep neural networks (DNNs) have shown powerful performance in various applications and are increasingly being used in decision-making systems. However, concerns about fairness in DNNs always persist. Some efficient white-box fairness testing methods about individual fairness have been proposed. Nevertheless, the development of black-box methods has stagnated, and the performance of existing methods is far behind that of white-box methods. In this paper, we propose a novel black-box individual fairness testing method called Model-Agnostic Fairness Testing (MAFT). By leveraging MAFT, practitioners can effectively identify and address discrimination in DL models, regardless of the specific algorithm or architecture employed. Our approach adopts lightweight procedures such as gradient estimation and attribute perturbation rather than non-trivial procedures like symbol execution, rendering it significantly more scalable and applicable than existing methods. We demonstrate that MAFT achieves the same effectiveness as state-of-the-art white-box methods whilst improving the applicability to large-scale networks. Compared to existing black-box approaches, our approach demonstrates distinguished performance in discovering fairness violations w.r.t effectiveness (approximately 14.69 times) and efficiency (approximately 32.58 times). 

**Abstract (ZH)**: 深度神经网络（DNNs）在各种应用中表现出强大的性能，并且越来越多地被用于决策系统中。然而，关于DNNs公平性的担忧始终存在。已经提出了一些高效的白盒个体公平性测试方法。但是，黑盒方法的发展停滞不前，现有的黑盒方法的性能远落后于白盒方法。在本文中，我们提出了一种新的黑盒个体公平性测试方法，称为Model-Agnostic Fairness Testing（MAFT）。通过利用MAFT，实践者可以在不考虑所使用的具体算法或架构的情况下，有效地识别和解决深度学习模型中的歧视问题。我们的方法采用轻量级的程序如梯度估计和属性扰动，而不是符号执行等复杂程序，使其比现有方法更具可扩展性和适用性。我们证明，MAFT在有效性方面与最先进的白盒方法相同，但在适用性方面明显优于大型网络。与现有的黑盒方法相比，我们的方法在有效性方面（约14.69倍）和效率方面（约32.58倍）表现出显著的优势。 

---
# Extract Information from Hybrid Long Documents Leveraging LLMs: A Framework and Dataset 

**Title (ZH)**: 利用大语言模型提取混合长文档信息：一个框架与数据集 

**Authors**: Chongjian Yue, Xinrun Xu, Xiaojun Ma, Lun Du, Zhiming Ding, Shi Han, Dongmei Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20072)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional performance in textual understanding and tabular reasoning tasks. However, their ability to comprehend and analyze hybrid text, containing textual and tabular data, remains unexplored. The hybrid text often appears in the form of hybrid long documents (HLDs), which far exceed the token limit of LLMs. Consequently, we apply an Automated Information Extraction framework (AIE) to enable LLMs to process the HLDs and carry out experiments to analyse four important aspects of information extraction from HLDs. Given the findings: 1) The effective way to select and summarize the useful part of a HLD. 2) An easy table serialization way is enough for LLMs to understand tables. 3) The naive AIE has adaptability in many complex scenarios. 4) The useful prompt engineering to enhance LLMs on HLDs. To address the issue of dataset scarcity in HLDs and support future work, we also propose the Financial Reports Numerical Extraction (FINE) dataset. The dataset and code are publicly available in the attachments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本理解和表格推理任务中表现出色。然而，它们在理解并分析包含文本和表格数据的混合文本方面的能力尚未得到探索。混合文本通常以混合长文档（HLDs）的形式出现，这远超LLM的令牌限制。因此，我们应用了自动信息提取框架（AIE），使LLM能够处理HLDs，并进行实验以分析信息从HLDs中提取的四个重要方面。根据研究发现：1）有效选择和总结HLD中有用部分的方法。2）简单的表格序列化方式足以让LLM理解表格。3）原始AIE在许多复杂场景中具有适应性。4）有用的提示工程可以增强LLM在处理HLDs方面的表现。为了解决HLDs数据集稀缺的问题，同时也为了支持未来的研究，我们还提出了金融报告数值提取（FINE）数据集。数据集和代码已在附件中公开提供。 

---
# On the Compositional Generalization of Multimodal LLMs for Medical Imaging 

**Title (ZH)**: 多模态大语言模型在医学成像中的组成性泛化研究 

**Authors**: Zhenyang Cai, Junying Chen, Rongsheng Wang, Weihong Wang, Yonglin Deng, Dingjie Song, Yize Chen, Zixu Zhang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20070)  

**Abstract**: Multimodal large language models (MLLMs) hold significant potential in the medical field, but their capabilities are often limited by insufficient data in certain medical domains, highlighting the need for understanding what kinds of images can be used by MLLMs for generalization. Current research suggests that multi-task training outperforms single-task as different tasks can benefit each other, but they often overlook the internal relationships within these tasks, providing limited guidance on selecting datasets to enhance specific tasks. To analyze this phenomenon, we attempted to employ compositional generalization (CG)-the ability of models to understand novel combinations by recombining learned elements-as a guiding framework. Since medical images can be precisely defined by Modality, Anatomical area, and Task, naturally providing an environment for exploring CG. Therefore, we assembled 106 medical datasets to create Med-MAT for comprehensive experiments. The experiments confirmed that MLLMs can use CG to understand unseen medical images and identified CG as one of the main drivers of the generalization observed in multi-task training. Additionally, further studies demonstrated that CG effectively supports datasets with limited data and delivers consistent performance across different backbones, highlighting its versatility and broad applicability. Med-MAT is publicly available at this https URL. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在医疗领域具有巨大的潜力，但其能力往往受限于某些医疗领域的数据不足，突显出理解MLLMs可以用于泛化的类型是什么的重要性。现有研究表明，多任务训练优于单任务训练，因为不同任务之间可以互相受益，但它们往往忽略了这些任务之间的内部关系，对选择增强特定任务的数据集指导有限。为了分析这一现象，我们尝试采用组合泛化（CG）——模型通过重组学习元素来理解新组合的能力——作为指导框架。由于医学图像可以通过模态、解剖区域和任务精确定义，天然地为探索CG提供了环境。因此，我们收集了106个医学数据集，创建了Med-MAT进行全面实验。实验结果证实了MLLMs可以利用CG来理解未见过的医学图像，并将CG确定为多任务训练中观察到的泛化现象的主要驱动力之一。此外，进一步的研究表明，CG有效支持数据不足的数据集，并在不同的底层架构上表现出一致的性能，突显出其多样性和广泛的适用性。Med-MAT现已在该网址公开：[此链接地址]。 

---
# VELoRA: A Low-Rank Adaptation Approach for Efficient RGB-Event based Recognition 

**Title (ZH)**: VELoRA：一种高效的基于RGB-事件的识别的低秩适应方法 

**Authors**: Lan Chen, Haoxiang Yang, Pengpeng Shao, Haoyu Song, Xiao Wang, Zhicheng Zhao, Yaowei Wang, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.20064)  

**Abstract**: Pattern recognition leveraging both RGB and Event cameras can significantly enhance performance by deploying deep neural networks that utilize a fine-tuning strategy. Inspired by the successful application of large models, the introduction of such large models can also be considered to further enhance the performance of multi-modal tasks. However, fully fine-tuning these models leads to inefficiency and lightweight fine-tuning methods such as LoRA and Adapter have been proposed to achieve a better balance between efficiency and performance. To our knowledge, there is currently no work that has conducted parameter-efficient fine-tuning (PEFT) for RGB-Event recognition based on pre-trained foundation models. To address this issue, this paper proposes a novel PEFT strategy to adapt the pre-trained foundation vision models for the RGB-Event-based classification. Specifically, given the RGB frames and event streams, we extract the RGB and event features based on the vision foundation model ViT with a modality-specific LoRA tuning strategy. The frame difference of the dual modalities is also considered to capture the motion cues via the frame difference backbone network. These features are concatenated and fed into high-level Transformer layers for efficient multi-modal feature learning via modality-shared LoRA tuning. Finally, we concatenate these features and feed them into a classification head to achieve efficient fine-tuning. The source code and pre-trained models will be released on \url{this https URL}. 

**Abstract (ZH)**: 利用RGB和事件相机结合进行模式识别可以通过部署利用微调策略的深度神经网络显著提升性能。受大型模型成功应用的启发，引入这些大型模型也可以进一步提高多模态任务的性能。然而，对这些模型进行全面微调会导致效率下降，轻量化微调方法如LoRA和Adapter已被提出，以在效率与性能之间取得更好的平衡。据我们所知，目前尚无针对基于预训练基础模型的RGB-事件识别进行参数高效微调（PEFT）的研究工作。为了解决这一问题，本文提出了一种新的PEFT策略，以适应基于RGB-事件分类的基础视觉模型。具体而言，给定RGB帧和事件流，我们基于模态特定的LoRA调优策略从ViT基础视觉模型中提取RGB和事件特征。同时考虑双模态帧差，通过帧差主干网络捕捉运动线索。这些特征被连接并送入高层次的Transformer层，利用模态共享的LoRA调优策略进行高效的多模态特征学习。最后，我们将这些特征连接起来并送入分类头以实现高效的微调。源代码和预训练模型将在 \url{this https URL} 释放。 

---
# CrossSpeech++: Cross-lingual Speech Synthesis with Decoupled Language and Speaker Generation 

**Title (ZH)**: CrossSpeech++: 分离语言和说话人生成的跨语言语音合成 

**Authors**: Ji-Hoon Kim, Hong-Sun Yang, Yoon-Cheol Ju, Il-Hwan Kim, Byeong-Yeol Kim, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2412.20048)  

**Abstract**: The goal of this work is to generate natural speech in multiple languages while maintaining the same speaker identity, a task known as cross-lingual speech synthesis. A key challenge of cross-lingual speech synthesis is the language-speaker entanglement problem, which causes the quality of cross-lingual systems to lag behind that of intra-lingual systems. In this paper, we propose CrossSpeech++, which effectively disentangles language and speaker information and significantly improves the quality of cross-lingual speech synthesis. To this end, we break the complex speech generation pipeline into two simple components: language-dependent and speaker-dependent generators. The language-dependent generator produces linguistic variations that are not biased by specific speaker attributes. The speaker-dependent generator models acoustic variations that characterize speaker identity. By handling each type of information in separate modules, our method can effectively disentangle language and speaker representation. We conduct extensive experiments using various metrics, and demonstrate that CrossSpeech++ achieves significant improvements in cross-lingual speech synthesis, outperforming existing methods by a large margin. 

**Abstract (ZH)**: 本文的目标是在多种语言中生成自然的语音的同时保持相同的说话人身份，这一任务被称为跨语言语音合成。跨语言语音合成的一个关键挑战是语言-说话人纠缠问题，这使得跨语言系统的质量落后于同语言系统的质量。本文中，我们提出了一种名为CrossSpeech++的方法，它有效地解纠缠了语言和说话人信息，并显著提高了跨语言语音合成的质量。为此，我们将复杂的语音生成管道分解为两个简单组件：语言依赖性和说话人依赖性生成器。语言依赖性生成器产生不受特定说话人属性偏见的语言变异。说话人依赖性生成器则建模刻画说话人身份的音、韵变异特性。通过在单独的模块中处理每种类型的信息，我们的方法可以有效解纠缠语言和说话人的表示。我们使用多种指标进行了广泛的实验，并证明CrossSpeech++在跨语言语音合成方面取得了显著的改进，大幅优于现有方法。 

---
# Enhancing Diffusion Models for Inverse Problems with Covariance-Aware Posterior Sampling 

**Title (ZH)**: 增强协方差感知后验采样以改善反问题中的扩散模型 

**Authors**: Shayan Mohajer Hamidi, En-Hui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20045)  

**Abstract**: Inverse problems exist in many disciplines of science and engineering. In computer vision, for example, tasks such as inpainting, deblurring, and super resolution can be effectively modeled as inverse problems. Recently, denoising diffusion probabilistic models (DDPMs) are shown to provide a promising solution to noisy linear inverse problems without the need for additional task specific training. Specifically, with the prior provided by DDPMs, one can sample from the posterior by approximating the likelihood. In the literature, approximations of the likelihood are often based on the mean of conditional densities of the reverse process, which can be obtained using Tweedie formula. To obtain a better approximation to the likelihood, in this paper we first derive a closed form formula for the covariance of the reverse process. Then, we propose a method based on finite difference method to approximate this covariance such that it can be readily obtained from the existing pretrained DDPMs, thereby not increasing the complexity compared to existing approaches. Finally, based on the mean and approximated covariance of the reverse process, we present a new approximation to the likelihood. We refer to this method as covariance-aware diffusion posterior sampling (CA-DPS). Experimental results show that CA-DPS significantly improves reconstruction performance without requiring hyperparameter tuning. The code for the paper is put in the supplementary materials. 

**Abstract (ZH)**: 逆问题在科学和工程的许多领域中都存在。例如，在计算机视觉中，任务如图像修复、去模糊和超分辨率可以有效地被建模为逆问题。最近，去噪扩散概率模型（DDPMs）被证明能够在不需要额外任务特定训练的情况下提供解决噪声线性逆问题的有希望的解决方案。具体而言，通过DDPMs提供的先验，可以通过近似似然来从后验中采样。在文献中，似然的近似通常基于逆过程条件密度的均值，该均值可以通过Tweedie公式获得。为了得到更好的似然近似，本文首先推导出逆过程的协方差的闭合形式公式。然后，我们提出了一种基于有限差分法的方法来近似这种协方差，使得它可以方便地从现有的预训练DDPMs获得，从而不会增加与现有方法相比的复杂性。最后，基于逆过程的均值和近似协方差，我们提出了一种新的似然近似方法。我们将这种方法称为协方差意识扩散后验采样（CA-DPS）。实验结果表明，CA-DPS在不进行超参数调整的情况下显著提高了重建性能。论文的代码已作为补充材料提供。 

---
# Calibre: Towards Fair and Accurate Personalized Federated Learning with Self-Supervised Learning 

**Title (ZH)**: Calibre：面向公平和个性化联邦学习的自监督学习方法 

**Authors**: Sijia Chen, Ningxin Su, Baochun Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.20020)  

**Abstract**: In the context of personalized federated learning, existing approaches train a global model to extract transferable representations, based on which any client could train personalized models with a limited number of data samples. Self-supervised learning is considered a promising direction as the global model it produces is generic and facilitates personalization for all clients fairly. However, when data is heterogeneous across clients, the global model trained using SSL is unable to learn high-quality personalized models. In this paper, we show that when the global model is trained with SSL without modifications, its produced representations have fuzzy class boundaries. As a result, personalized learning within each client produces models with low accuracy. In order to improve SSL towards better accuracy without sacrificing its advantage in fairness, we propose Calibre, a new personalized federated learning framework designed to calibrate SSL representations by maintaining a suitable balance between more generic and more client-specific representations. Calibre is designed based on theoretically-sound properties, and introduces (1) a client-specific prototype loss as an auxiliary training objective; and (2) an aggregation algorithm guided by such prototypes across clients. Our experimental results in an extensive array of non-i.i.d.~settings show that Calibre achieves state-of-the-art performance in terms of both mean accuracy and fairness across clients. Code repo: this https URL. 

**Abstract (ZH)**: 在个性化联邦学习的背景下，现有方法训练一个全局模型以提取可转移的表示，基于这些表示，任何客户端都能够使用有限数量的数据样本训练个性化模型。自监督学习被认为是一个有前途的方向，因为它产生的全局模型具有通用性，能够公平地促进所有客户端的个性化学习。然而，当客户端之间的数据异构性较高时，使用自监督学习（SSL）训练的全局模型无法学习高质量的个性化模型。在本文中，我们展示了，在不做修改的情况下使用SSL训练全局模型时，其产生的表示具有模糊的类别边界。因此，每个客户端内部的个性化学习会产生低精度的模型。为了在不牺牲公平性优势的情况下提高自监督学习的精度，我们提出了Calibre，这是一种新的个性化联邦学习框架，旨在通过保持更通用和更客户端特定的表示之间的适当平衡来校准SSL表示。Calibre基于理论上的属性进行设计，并引入了（1）特定于客户端的原型损失作为辅助训练目标；以及（2）由此类原型指导的聚合算法。我们在广泛的非独立同分布（non-i.i.d.）设置中的实验结果显示，Calibre在均值准确性与客户端公平性方面均实现了最先进的性能。代码仓库：this https URL。 

---
# ProtCLIP: Function-Informed Protein Multi-Modal Learning 

**Title (ZH)**: ProtCLIP: 功能导向的蛋白质多模态学习 

**Authors**: Hanjing Zhou, Mingze Yin, Wei Wu, Mingyang Li, Kun Fu, Jintai Chen, Jian Wu, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20014)  

**Abstract**: Multi-modality pre-training paradigm that aligns protein sequences and biological descriptions has learned general protein representations and achieved promising performance in various downstream applications. However, these works were still unable to replicate the extraordinary success of language-supervised visual foundation models due to the ineffective usage of aligned protein-text paired data and the lack of an effective function-informed pre-training paradigm. To address these issues, this paper curates a large-scale protein-text paired dataset called ProtAnno with a property-driven sampling strategy, and introduces a novel function-informed protein pre-training paradigm. Specifically, the sampling strategy determines selecting probability based on the sample confidence and property coverage, balancing the data quality and data quantity in face of large-scale noisy data. Furthermore, motivated by significance of the protein specific functional mechanism, the proposed paradigm explicitly model protein static and dynamic functional segments by two segment-wise pre-training objectives, injecting fine-grained information in a function-informed manner. Leveraging all these innovations, we develop ProtCLIP, a multi-modality foundation model that comprehensively represents function-aware protein embeddings. On 22 different protein benchmarks within 5 types, including protein functionality classification, mutation effect prediction, cross-modal transformation, semantic similarity inference and protein-protein interaction prediction, our ProtCLIP consistently achieves SOTA performance, with remarkable improvements of 75% on average in five cross-modal transformation benchmarks, 59.9% in GO-CC and 39.7% in GO-BP protein function prediction. The experimental results verify the extraordinary potential of ProtCLIP serving as the protein multi-modality foundation model. 

**Abstract (ZH)**: 多模态预训练范式已经在对蛋白质序列和生物描述进行对齐后学习到了通用的蛋白质表示，并在多种下游应用中取得了显著的表现。然而，这些工作仍无法复制语言监督视觉基础模型的卓越成功，原因在于缺乏有效利用对齐的蛋白质-文本配对数据的有效方法和缺乏功能导向的预训练范式。为了解决这些问题，本论文采用一种以属性驱动的采样策略构建了一个大规模的蛋白质-文本配对数据集ProtAnno，并引入了一种新的功能导向的蛋白质预训练范式。具体而言，采样策略根据样本置信度和属性覆盖度来确定采样概率，平衡大规模噪声数据的数据质量和数据量。此外，鉴于蛋白质特异性功能机制的重要性，所提出的范式通过两个段级预训练目标明确建模蛋白质的静态和动态功能段，以功能导向的方式注入细粒度信息。利用这些创新，我们开发了ProtCLIP，这是一种综合表示功能意识蛋白质嵌入的多模态基础模型。在包含5种类型的22个不同蛋白质基准中的各类测试（包括蛋白质功能分类、突变效应预测、跨模态变换、语义相似性推理和蛋白质-蛋白质相互作用预测），我们的ProtCLIP在所有测试中均表现出一致的领先性能，特别是在5个跨模态变换基准中的平均提升为75%，在GO-CC（细胞组件）蛋白质功能预测中的提升为59.9%，在GO-BP（分子功能）蛋白质功能预测中的提升为39.7%。实验结果验证了ProtCLIP作为蛋白质多模态基础模型的出色潜力。 

---
# OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System 

**Title (ZH)**: OneKE：一种基于模式指导的LLM代理知识提取系统（Docker 化版本） 

**Authors**: Yujie Luo, Xiangyuan Ru, Kangwei Liu, Lin Yuan, Mengshu Sun, Ningyu Zhang, Lei Liang, Zhiqiang Zhang, Jun Zhou, Lanning Wei, Da Zheng, Haofen Wang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20005)  

**Abstract**: We introduce OneKE, a dockerized schema-guided knowledge extraction system, which can extract knowledge from the Web and raw PDF Books, and support various domains (science, news, etc.). Specifically, we design OneKE with multiple agents and a configure knowledge base. Different agents perform their respective roles, enabling support for various extraction scenarios. The configure knowledge base facilitates schema configuration, error case debugging and correction, further improving the performance. Empirical evaluations on benchmark datasets demonstrate OneKE's efficacy, while case studies further elucidate its adaptability to diverse tasks across multiple domains, highlighting its potential for broad applications. We have open-sourced the Code at this https URL and released a Video at this http URL. 

**Abstract (ZH)**: 我们介绍了OneKE，这是一个容器化的基于模式的知识提取系统，能够从网页和原始PDF书籍中提取知识，并支持多种领域（如科学、新闻等）。具体而言，我们设计了OneKE，使其包含多个代理和一个配置的知识库。各个代理承担不同的角色，从而支持多种提取场景。配置的知识库有助于模式配置、错误案例调试和修正，进而提高系统的性能。基准数据集上的实证评估展示了OneKE的有效性，而案例研究进一步阐明了其在多个领域的多种任务中适应性的特点，突显了其广泛的应用潜力。我们已将源代码开源发布在 <https://this-url.com/>，并在 <https://this-url.com/> 上传了演示视频。 

---
# Adaptive Parameter-Efficient Federated Fine-Tuning on Heterogeneous Devices 

**Title (ZH)**: 适应性参数高效跨设备异构联邦微调 

**Authors**: Jun Liu, Yunming Liao, Hongli Xu, Yang Xu, Jianchun Liu, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2412.20004)  

**Abstract**: Federated fine-tuning (FedFT) has been proposed to fine-tune the pre-trained language models in a distributed manner. However, there are two critical challenges for efficient FedFT in practical applications, i.e., resource constraints and system heterogeneity. Existing works rely on parameter-efficient fine-tuning methods, e.g., low-rank adaptation (LoRA), but with major limitations. Herein, based on the inherent characteristics of FedFT, we observe that LoRA layers with higher ranks added close to the output help to save resource consumption while achieving comparable fine-tuning performance. Then we propose a novel LoRA-based FedFT framework, termed LEGEND, which faces the difficulty of determining the number of LoRA layers (called, LoRA depth) and the rank of each LoRA layer (called, rank distribution). We analyze the coupled relationship between LoRA depth and rank distribution, and design an efficient LoRA configuration algorithm for heterogeneous devices, thereby promoting fine-tuning efficiency. Extensive experiments are conducted on a physical platform with 80 commercial devices. The results show that LEGEND can achieve a speedup of 1.5-2.8$\times$ and save communication costs by about 42.3% when achieving the target accuracy, compared to the advanced solutions. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的内容：

联邦微调（FedFT）已被提出用于以分布方式微调预训练语言模型。然而，在实际应用中，高效的FedFT面临两个关键挑战，即资源约束和系统异质性。现有工作依赖于参数高效微调方法，例如低秩适应（LoRA），但这些方法存在重大限制。在此基础上，我们根据FedFT的固有特性观察到，将更高秩的LoRA层靠近输出添加可以帮助节省资源消耗，同时实现相当的微调性能。因此，我们提出了一种基于LoRA的新型FedFT框架，称为LEGEND，该框架面临的困难在于确定LoRA层的数量（称为LoRA深度）以及每个LoRA层的秩（称为秩分布）。我们分析了LoRA深度与秩分布之间的耦合关系，并设计了一种适应异构设备的高效LoRA配置算法，从而促进微调效率。我们在包含80台商用设备的物理平台上进行了广泛的实验。结果表明，当达到目标精度时，与先进的解决方案相比，LEGEND可以实现1.5-2.8倍的速度提升，并降低通信成本约42.3%。 

---
# Comprehensive Review of EEG-to-Output Research: Decoding Neural Signals into Images, Videos, and Audio 

**Title (ZH)**: EEG到输出的综合综述：解码神经信号为图像、视频和音频 

**Authors**: Yashvir Sabharwal, Balaji Rama  

**Link**: [PDF](https://arxiv.org/pdf/2412.19999)  

**Abstract**: Electroencephalography (EEG) is an invaluable tool in neuroscience, offering insights into brain activity with high temporal resolution. Recent advancements in machine learning and generative modeling have catalyzed the application of EEG in reconstructing perceptual experiences, including images, videos, and audio. This paper systematically reviews EEG-to-output research, focusing on state-of-the-art generative methods, evaluation metrics, and data challenges. Using PRISMA guidelines, we analyze 1800 studies and identify key trends, challenges, and opportunities in the field. The findings emphasize the potential of advanced models such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Transformers, while highlighting the pressing need for standardized datasets and cross-subject generalization. A roadmap for future research is proposed that aims to improve decoding accuracy and broadening real-world applications. 

**Abstract (ZH)**: 脑电图（EEG）是神经科学中不可或缺的工具，能够以高时间分辨率揭示大脑活动。近期机器学习和生成建模的进展推动了EEG在重建感知体验方面的应用，包括图像、视频和音频。本文系统性地回顾了EEG到输出的研究，重点关注最新的生成方法、评估指标和数据挑战。我们按照PRISMA指南分析了1800项研究，并确定了该领域的关键趋势、挑战和机遇。研究结果强调了生成对抗网络（GANs）、变分自编码器（VAEs）和变换器等高级模型的潜力，同时也突显了标准化数据集和跨被试一般化方面的紧迫需求。本文提出了未来研究的路线图，旨在提高解码准确性并扩大实际应用的范围。 

---
# From Generalist to Specialist: A Survey of Large Language Models for Chemistry 

**Title (ZH)**: 从通才到专才：化学领域大型语言模型综述 

**Authors**: Yang Han, Ziping Wan, Lu Chen, Kai Yu, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.19994)  

**Abstract**: Large Language Models (LLMs) have significantly transformed our daily life and established a new paradigm in natural language processing (NLP). However, the predominant pretraining of LLMs on extensive web-based texts remains insufficient for advanced scientific discovery, particularly in chemistry. The scarcity of specialized chemistry data, coupled with the complexity of multi-modal data such as 2D graph, 3D structure and spectrum, present distinct challenges. Although several studies have reviewed Pretrained Language Models (PLMs) in chemistry, there is a conspicuous absence of a systematic survey specifically focused on chemistry-oriented LLMs. In this paper, we outline methodologies for incorporating domain-specific chemistry knowledge and multi-modal information into LLMs, we also conceptualize chemistry LLMs as agents using chemistry tools and investigate their potential to accelerate scientific research. Additionally, we conclude the existing benchmarks to evaluate chemistry ability of LLMs. Finally, we critically examine the current challenges and identify promising directions for future research. Through this comprehensive survey, we aim to assist researchers in staying at the forefront of developments in chemistry LLMs and to inspire innovative applications in the field. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在我们的日常生活中发挥了显著作用，并在自然语言处理（NLP）领域建立了新的范式。然而，LLMs主要通过广泛网络文本进行预训练，这在先进科学发现方面仍显不足，尤其是在化学领域。由于缺乏专门的化学数据，以及2D图形、3D结构和光谱等多模态数据的复杂性，这带来了独特的挑战。虽然已有几项研究回顾了化学领域的预训练语言模型（PLMs），但专门针对面向化学的LLMs的系统性调查仍然不足。本文概述了将领域特定的化学知识和多模态信息整合到LLMs中的方法，同时也将化学LLMs设想为使用化学工具的代理，并探讨了它们加速科学研究的潜力。此外，我们综合了现有的基准来评估化学能力，最后批判性地分析了当前面临的挑战，并指出了未来研究的有希望的方向。通过这一全面的综述，我们希望帮助研究人员保持对化学LLMs发展的前沿，并激发领域内的创新应用。 

---
# An Ordinary Differential Equation Sampler with Stochastic Start for Diffusion Bridge Models 

**Title (ZH)**: 用于扩散桥梁模型的随机起点常微分方程采样器 

**Authors**: Yuang Wang, Pengfei Jin, Li Zhang, Quanzheng Li, Zhiqiang Chen, Dufan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19992)  

**Abstract**: Diffusion bridge models have demonstrated promising performance in conditional image generation tasks, such as image restoration and translation, by initializing the generative process from corrupted images instead of pure Gaussian noise. However, existing diffusion bridge models often rely on Stochastic Differential Equation (SDE) samplers, which result in slower inference speed compared to diffusion models that employ high-order Ordinary Differential Equation (ODE) solvers for acceleration. To mitigate this gap, we propose a high-order ODE sampler with a stochastic start for diffusion bridge models. To overcome the singular behavior of the probability flow ODE (PF-ODE) at the beginning of the reverse process, a posterior sampling approach was introduced at the first reverse step. The sampling was designed to ensure a smooth transition from corrupted images to the generative trajectory while reducing discretization errors. Following this stochastic start, Heun's second-order solver is applied to solve the PF-ODE, achieving high perceptual quality with significantly reduced neural function evaluations (NFEs). Our method is fully compatible with pretrained diffusion bridge models and requires no additional training. Extensive experiments on image restoration and translation tasks, including super-resolution, JPEG restoration, Edges-to-Handbags, and DIODE-Outdoor, demonstrated that our sampler outperforms state-of-the-art methods in both visual quality and Frechet Inception Distance (FID). 

**Abstract (ZH)**: 扩散桥梁模型在基于受损图像初始化生成过程的条件图像生成任务（如图像修复和翻译）中表现出了令人鼓舞的性能。然而，现有的扩散桥梁模型往往依赖于随机微分方程（SDE）采样器，这导致了与使用高阶常微分方程（ODE）求解器加速的扩散模型相比，推理速度较慢。为缩小这一差距，我们提出了一种带有随机启动的高阶ODE采样器，用于扩散桥梁模型。为了解决概率流ODE（PF-ODE）在逆向过程初期的奇异行为，我们在首个逆向步骤中引入了后验采样方法。这种方法设计目的是确保从受损图像平滑过渡到生成轨迹，同时减少离散化误差。在该随机启动之后，我们应用Heun的二阶求解器来求解PF-ODE，从而在显著减少神经网络函数评估（NFEs）的情况下，实现高质量的感知质量。我们的方法完全兼容预训练的扩散桥梁模型，并不需要额外的训练。在图像修复和翻译任务中的广泛实验，包括超分辨率、JPEG修复、Edges-to-Handbags和DIODE-室外场景，表明我们的采样器在视觉质量和弗雷歇-_inception_距离（FID）方面优于最先进的方法。 

---
# Delayed Random Partial Gradient Averaging for Federated Learning 

**Title (ZH)**: 延迟随机部分梯度平均在 Federated Learning 中的应用 

**Authors**: Xinyi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19987)  

**Abstract**: Federated learning (FL) is a distributed machine learning paradigm that enables multiple clients to train a shared model collaboratively while preserving privacy. However, the scaling of real-world FL systems is often limited by two communication bottlenecks:(a) while the increasing computing power of edge devices enables the deployment of large-scale Deep Neural Networks (DNNs), the limited bandwidth constraints frequent transmissions over large DNNs; and (b) high latency cost greatly degrades the performance of FL. In light of these bottlenecks, we propose a Delayed Random Partial Gradient Averaging (DPGA) to enhance FL. Under DPGA, clients only share partial local model gradients with the server. The size of the shared part in a local model is determined by the update rate, which is coarsely initialized and subsequently refined over the temporal dimension. Moreover, DPGA largely reduces the system run time by enabling computation in parallel with communication. We conduct experiments on non-IID CIFAR-10/100 to demonstrate the efficacy of our method. 

**Abstract (ZH)**: 联邦学习（FL）是一种分布式机器学习范式，使多个客户端能够合作训练共享模型，同时保护隐私。然而，实际应用中的FL系统往往受限于两种通信瓶颈：(a) 尽管边缘设备的计算能力提升了大规模深度神经网络（DNNs）的部署，但有限的带宽限制了大规模DNNs的频繁传输；(b) 高延迟显著降低了FL的性能。考虑到这些瓶颈，我们提出了一种延迟随机部分梯度平均（DPGA）方法来增强FL。在DPGA中，客户端仅与服务器分享局部模型的部分梯度。每个局部模型中共享部分的大小由更新频率决定，在时间维度上被粗略初始化并进一步细化。此外，DPGA通过使计算与通信并行进行，大大减少了系统运行时间。我们通过在非iid的CIFAR-10/100数据集上进行实验，证明了该方法的有效性。 

---
# The Fifth International Verification of Neural Networks Competition (VNN-COMP 2024): Summary and Results 

**Title (ZH)**: 第五届国际神经网络验证竞赛（VNN-COMP 2024）：总结与结果 

**Authors**: Christopher Brix, Stanley Bak, Taylor T. Johnson, Haoze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19985)  

**Abstract**: This report summarizes the 5th International Verification of Neural Networks Competition (VNN-COMP 2024), held as a part of the 7th International Symposium on AI Verification (SAIV), that was collocated with the 36th International Conference on Computer-Aided Verification (CAV). VNN-COMP is held annually to facilitate the fair and objective comparison of state-of-the-art neural network verification tools, encourage the standardization of tool interfaces, and bring together the neural network verification community. To this end, standardized formats for networks (ONNX) and specification (VNN-LIB) were defined, tools were evaluated on equal-cost hardware (using an automatic evaluation pipeline based on AWS instances), and tool parameters were chosen by the participants before the final test sets were made public. In the 2024 iteration, 8 teams participated on a diverse set of 12 regular and 8 extended benchmarks. This report summarizes the rules, benchmarks, participating tools, results, and lessons learned from this iteration of this competition. 

**Abstract (ZH)**: 本报告总结了第五届国际神经网络验证竞赛（VNN-COMP 2024），该竞赛作为第7届人工智能验证国际研讨会（SAIV）的一部分，在第36届计算机辅助验证国际会议（CAV）同期举办。VNN-COMP 每年举办一次，旨在促进最先进的神经网络验证工具的公正客观比较，鼓励工具接口的标准制定，并聚集神经网络验证社区。为此，定义了标准化的网络格式（ONNX）和规范格式（VNN-LIB），所有工具都在同等成本的硬件上进行评估（使用基于AWS实例的自动评估管道），工具参数由参赛者在最终测试集公布前选定。在2024年这届比赛中，共有8支队伍在一个多样化的12项常规和8项扩展基准上进行了角逐。本报告总结了此次竞赛的规则、基准、参赛工具、结果以及从中获得的经验教训。 

---
# Will you donate money to a chatbot? The effect of chatbot anthropomorphic features and persuasion strategies on willingness to donate 

**Title (ZH)**: 你会为聊天机器人捐款吗？聊天机器人的拟人特征及其说服策略对其捐款意愿的影响 

**Authors**: Ekaterina Novozhilova, Jiacheng Huang, Le He, Ziling Li, James Cummings  

**Link**: [PDF](https://arxiv.org/pdf/2412.19976)  

**Abstract**: This work investigates the causal mechanism behind the effect of chatbot personification and persuasion strategies on users' perceptions and donation likelihood. In a 2 (personified vs. non-personified chatbot) x 2 (emotional vs. logical persuasion strategy) between-subjects experiment (N=76), participants engaged with a chatbot that represented a non-profit charitable organization. The results suggest that interaction with a personified chatbot evokes perceived anthropomorphism; however, it does not elicit greater willingness to donate. In fact, we found that commonly used anthropomorphic features, like name and narrative, led to negative attitudes toward an AI agent in the donation context. Our results showcase a preference for non-personified chatbots paired with logical persuasion appeal, emphasizing the significance of consistency in chatbot interaction, mirroring human-human engagement. We discuss the importance of moving from exploring the common scenario of a chatbot with machine identity vs. a chatbot with human identity in light of the recent regulations of AI systems. 

**Abstract (ZH)**: 本研究探讨了聊天机器人拟人化和说服策略对用户感知和捐款意愿背后因果机制的影响。在一项包含两个因素的被试间实验（48名参与者，2（拟人化聊天机器人 vs. 非拟人化聊天机器人）× 2（情感说服策略 vs. 理性说服策略））中，参与者与代表非营利慈善组织的聊天机器人进行了互动。研究结果表明，与拟人化聊天机器人交互会引发拟人化的感知，但并不会提高捐款意愿。实际上，我们发现，通常用于拟人化特征，如名字和叙述，导致了在捐款情境中对AI代理的负面态度。本研究结果展示了偏好非拟人化聊天机器人配以理性说服策略的倾向，强调了聊天机器人交互中一致性的意义，使其与人类互动相呼应。我们讨论了在近期对AI系统的监管背景下，从探索具有机器身份的聊天机器人与具有人类身份的聊天机器人之间的常见场景转向的重要性。 

---
# MobileNetV2: A lightweight classification model for home-based sleep apnea screening 

**Title (ZH)**: 基于MobileNetV2的家庭睡眠呼吸暂停筛查轻量级分类模型 

**Authors**: Hui Pan, Yanxuan Yu, Jilun Ye, Xu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.19967)  

**Abstract**: This study proposes a novel lightweight neural network model leveraging features extracted from electrocardiogram (ECG) and respiratory signals for early OSA screening. ECG signals are used to generate feature spectrograms to predict sleep stages, while respiratory signals are employed to detect sleep-related breathing abnormalities. By integrating these predictions, the method calculates the apnea-hypopnea index (AHI) with enhanced accuracy, facilitating precise OSA diagnosis.
The method was validated on three publicly available sleep apnea databases: the Apnea-ECG database, the UCDDB dataset, and the MIT-BIH Polysomnographic database. Results showed an overall OSA detection accuracy of 0.978, highlighting the model's robustness. Respiratory event classification achieved an accuracy of 0.969 and an area under the receiver operating characteristic curve (ROC-AUC) of 0.98. For sleep stage classification, in UCDDB dataset, the ROC-AUC exceeded 0.85 across all stages, with recall for Sleep reaching 0.906 and specificity for REM and Wake states at 0.956 and 0.937, respectively.
This study underscores the potential of integrating lightweight neural networks with multi-signal analysis for accurate, portable, and cost-effective OSA screening, paving the way for broader adoption in home-based and wearable health monitoring systems. 

**Abstract (ZH)**: 本文提出了一种利用心电图（ECG）和呼吸信号提取的特征构建的轻量级神经网络模型，用于早期睡眠呼吸暂停（OSA）筛查。ECG信号用于生成特征频谱图以预测睡眠阶段，而呼吸信号则用于检测睡眠相关的呼吸异常。通过结合这些预测，该方法可以更准确地计算呼吸暂停低通气指数（AHI），从而有助于准确诊断OSA。

该方法在三个公开的睡眠呼吸暂停数据库上进行了验证，包括Apnea-ECG数据库、UCDDB数据集和MIT-BIH多导睡眠图数据库。结果显示OSA检测准确性为0.978，突显了该模型的鲁棒性。呼吸事件分类的准确率为0.969，受试者操作特征曲线下面积（ROC-AUC）为0.98。对于睡眠阶段分类，在UCDDB数据集中，所有阶段的ROC-AUC均超过0.85，睡眠阶段的召回率为0.906，REM和清醒状态的特异性分别为0.956和0.937。

本文强调了将轻量级神经网络与多信号分析结合，用于精准、便携且成本效益高的OSA筛查的潜力，为家庭和穿戴式健康监测系统的大规模应用铺平了道路。 

---
# Bridging Context Gaps: Enhancing Comprehension in Long-Form Social Conversations Through Contextualized Excerpts 

**Title (ZH)**: 填补背景差距：通过上下文化摘录增强长篇社会对话的理解 

**Authors**: Shrestha Mohanty, Sarah Xuan, Jacob Jobraeel, Anurag Kumar, Deb Roy, Jad Kabbara  

**Link**: [PDF](https://arxiv.org/pdf/2412.19966)  

**Abstract**: We focus on enhancing comprehension in small-group recorded conversations, which serve as a medium to bring people together and provide a space for sharing personal stories and experiences on crucial social matters. One way to parse and convey information from these conversations is by sharing highlighted excerpts in subsequent conversations. This can help promote a collective understanding of relevant issues, by highlighting perspectives and experiences to other groups of people who might otherwise be unfamiliar with and thus unable to relate to these experiences. The primary challenge that arises then is that excerpts taken from one conversation and shared in another setting might be missing crucial context or key elements that were previously introduced in the original conversation. This problem is exacerbated when conversations become lengthier and richer in themes and shared experiences. To address this, we explore how Large Language Models (LLMs) can enrich these excerpts by providing socially relevant context. We present approaches for effective contextualization to improve comprehension, readability, and empathy. We show significant improvements in understanding, as assessed through subjective and objective evaluations. While LLMs can offer valuable context, they struggle with capturing key social aspects. We release the Human-annotated Salient Excerpts (HSE) dataset to support future work. Additionally, we show how context-enriched excerpts can provide more focused and comprehensive conversation summaries. 

**Abstract (ZH)**: 我们专注于增强小型团体录音对话中的理解力，这些对话作为一种媒介，可以促进人们之间的联系，并提供一个分享关于重要社会事务个人故事和经验的空间。通过在后续对话中分享突出显示的片段，可以增强其他小组成员对相关问题的集体理解，突出不同的视角和经验，使他们能够理解和关联这些经验。然而，随之而来的主要挑战是，从一个对话中提取的片段在另一环境中重新分享时，可能会缺少原始对话中之前引入的关键背景或要素。当对话变得越来越长，并且主题和共享经验更加丰富时，这一问题会更加凸显。为了解决这一问题，我们探索了大规模语言模型（LLMs）如何通过提供社会相关背景来丰富这些片段。我们提出了有效情境化的策略，以提高理解力、可读性和同情心。通过主观和客观评估，我们展示了显著的理解改进。尽管LLMs可以提供有价值的社会背景，但在捕捉关键的社会方面方面存在困难。我们发布了人类标注的重要片段（HSE）数据集，以支持未来的研究工作。此外，我们展示了情境丰富的片段可以提供更加集中和全面的对话总结。 

---
# DepthMamba with Adaptive Fusion 

**Title (ZH)**: 深度融合的DepthMamba 

**Authors**: Zelin Meng, Zhichen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.19964)  

**Abstract**: Multi-view depth estimation has achieved impressive performance over various benchmarks. However, almost all current multi-view systems rely on given ideal camera poses, which are unavailable in many real-world scenarios, such as autonomous driving. In this work, we propose a new robustness benchmark to evaluate the depth estimation system under various noisy pose settings. Surprisingly, we find current multi-view depth estimation methods or single-view and multi-view fusion methods will fail when given noisy pose settings. To tackle this challenge, we propose a two-branch network architecture which fuses the depth estimation results of single-view and multi-view branch. In specific, we introduced mamba to serve as feature extraction backbone and propose an attention-based fusion methods which adaptively select the most robust estimation results between the two branches. Thus, the proposed method can perform well on some challenging scenes including dynamic objects, texture-less regions, etc. Ablation studies prove the effectiveness of the backbone and fusion method, while evaluation experiments on challenging benchmarks (KITTI and DDAD) show that the proposed method achieves a competitive performance compared to the state-of-the-art methods. 

**Abstract (ZH)**: 多视角深度估计在各种基准上取得了令人印象深刻的性能。然而，目前几乎所有多视角系统都依赖于给定的理想相机姿态，而在许多现实场景中，如自动驾驶，这种理想姿态是不可用的。在这项工作中，我们提出一个新的鲁棒性基准，以评估在各种噪声姿态设置下的深度估计系统性能。令人惊讶的是，我们发现当前的多视角深度估计方法或单视角与多视角融合方法在给定噪声姿态设置时会失效。为了解决这一挑战，我们提出了一种双分支网络架构，该架构融合了单视角和多视角分支的深度估计结果。具体而言，我们引入了mamba作为特征提取骨干，并提出了一种基于注意力的融合方法，该方法能够在两个分支之间选择最稳健的估计结果。因此，所提出的方法在一些具有挑战性的场景中表现良好，包括动态对象、无纹理区域等。消融研究证明了骨干和融合方法的有效性，而针对具有挑战性的基准（KITTI和DDAD）的评估实验表明，所提出的方法在性能上与现有的最先进方法相当。 

---
# ErgoChat: a Visual Query System for the Ergonomic Risk Assessment of Construction Workers 

**Title (ZH)**: ErgoChat：一种用于评估建筑工人职业风险的可视化查询系统 

**Authors**: Chao Fan, Qipei Mei, Xiaonan Wang, Xinming Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.19954)  

**Abstract**: In the construction sector, workers often endure prolonged periods of high-intensity physical work and prolonged use of tools, resulting in injuries and illnesses primarily linked to postural ergonomic risks, a longstanding predominant health concern. To mitigate these risks, researchers have applied various technological methods to identify the ergonomic risks that construction workers face. However, traditional ergonomic risk assessment (ERA) techniques do not offer interactive feedback. The rapidly developing vision-language models (VLMs), capable of generating textual descriptions or answering questions about ergonomic risks based on image inputs, have not yet received widespread attention. This research introduces an interactive visual query system tailored to assess the postural ergonomic risks of construction workers. The system's capabilities include visual question answering (VQA), which responds to visual queries regarding workers' exposure to postural ergonomic risks, and image captioning (IC), which generates textual descriptions of these risks from images. Additionally, this study proposes a dataset designed for training and testing such methodologies. Systematic testing indicates that the VQA functionality delivers an accuracy of 96.5%. Moreover, evaluations using nine metrics for IC and assessments from human experts indicate that the proposed approach surpasses the performance of a method using the same architecture trained solely on generic datasets. This study sets a new direction for future developments in interactive ERA using generative artificial intelligence (AI) technologies. 

**Abstract (ZH)**: 在建筑行业，工人经常需要进行长时间的高强度体力工作，并经常使用工具，这导致了与姿势相关的人体工学风险相关的伤害和疾病，这是长期存在的主要健康问题之一。为了减轻这些风险，研究人员已经应用了各种技术方法来识别建筑工人面临的工学风险。然而，传统的工学风险评估（Ergonomic Risk Assessment, ERA）技术并没有提供互动反馈。基于图像输入生成文本描述或回答与工学风险有关的问题的能力日益增强的视觉-语言模型（Vision-Language Models, VLMs）尚未引起广泛关注。本研究介绍了专门用于评估建筑工人姿势工学风险的互动视觉查询系统。该系统的功能包括视觉问答（Visual Question Answering, VQA），它可以针对工人暴露于姿势工学风险的视觉查询做出回答，以及图像生成（Image Captioning, IC），它可以生成表示这些风险的文本描述。此外，本研究还提出了一种用于训练和测试此类方法的数据集。系统测试表明，VQA功能的准确率为96.5%。此外，使用九个指标对IC进行评估以及来自人类专家的评估表明，所提出的方法在使用相同架构但仅在通用数据集上训练的方法中表现出更优异的性能。本研究为使用生成性人工智能（AI）技术进行互动工学风险评估的未来开发指明了新的方向。 

---
# Standard-Deviation-Inspired Regularization for Improving Adversarial Robustness 

**Title (ZH)**: 基于标准差启发的正则化方法以提高对抗鲁棒性 

**Authors**: Olukorede Fakorede, Modeste Atsague, Jin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.19947)  

**Abstract**: Adversarial Training (AT) has been demonstrated to improve the robustness of deep neural networks (DNNs) against adversarial attacks. AT is a min-max optimization procedure where in adversarial examples are generated to train a more robust DNN. The inner maximization step of AT increases the losses of inputs with respect to their actual classes. The outer minimization involves minimizing the losses on the adversarial examples obtained from the inner maximization. This work proposes a standard-deviation-inspired (SDI) regularization term to improve adversarial robustness and generalization. We argue that the inner maximization in AT is similar to minimizing a modified standard deviation of the model's output probabilities. Moreover, we suggest that maximizing this modified standard deviation can complement the outer minimization of the AT framework. To support our argument, we experimentally show that the SDI measure can be used to craft adversarial examples. Additionally, we demonstrate that combining the SDI regularization term with existing AT variants enhances the robustness of DNNs against stronger attacks, such as CW and Auto-attack, and improves generalization. 

**Abstract (ZH)**: 对抗训练（AT）已经被证明能够提高深度神经网络（DNNs）对抗对抗攻击的稳健性。AT 是一个最小-最大优化过程，其中通过生成对抗样本来训练更为稳健的 DNN。AT 的内部最大化步骤增加了输入与实际类别的损失。外部最小化步骤则涉及最小化从内部最大化得到的对抗样本上的损失。本文提出了一种基于方差启发（SDI）的正则化项，以提高对抗稳健性和泛化能力。我们认为，AT 的内部最大化过程类似于最小化模型输出概率的修改后的方差。此外，我们建议通过最大化这种修改后的方差来补充AT框架中的外部最小化过程。为了支持我们的论点，我们实验证明SDI度量可以用于生成对抗样本。同时，我们还展示了将SDI正则化项与现有的AT变体结合使用，能够增强DNNs在更强的攻击（如CW和Auto-attack）下的稳健性，并提高泛化能力。 

---
# Towards Strong AI: Transformational Beliefs and Scientific Creativity 

**Title (ZH)**: 通向强人工智能：变革性信念与科学创造性 

**Authors**: Samuel J. Eschker, Chuanhai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19938)  

**Abstract**: Strong artificial intelligence (AI) is envisioned to possess general cognitive abilities and scientific creativity comparable to human intelligence, encompassing both knowledge acquisition and problem-solving. While remarkable progress has been made in weak AI, the realization of strong AI remains a topic of intense debate and critical examination. In this paper, we explore pivotal innovations in the history of astronomy and physics, focusing on the discovery of Neptune and the concept of scientific revolutions as perceived by philosophers of science. Building on these insights, we introduce a simple theoretical and statistical framework of weak beliefs, termed the Transformational Belief (TB) framework, designed as a foundation for modeling scientific creativity. Through selected illustrative examples in statistical science, we demonstrate the TB framework's potential as a promising foundation for understanding, analyzing, and even fostering creativity -- paving the way toward the development of strong AI. We conclude with reflections on future research directions and potential advancements. 

**Abstract (ZH)**: 本文将以下内容或标题翻译成中文，确保符合学术规范：

强大的人工智能（AI）被期望具备与人类智能相媲美的广泛认知能力和科学创造力，涵盖了知识获取和问题解决。尽管在弱AI方面取得了显著进展，强大的AI的实现仍然是一个激烈的辩论和批判性审视的话题。本文探讨了天文学和物理学历史上的一些关键创新，重点关注海王星的发现及其所体现的科学革命概念，按哲学家的观点。在此基础上，我们引入了一个简单理论和统计框架，称为转变性信念（Transformational Belief, TB）框架，旨在为其科学创造力的建模提供基础。通过统计科学中的选择性示例，我们展示了TB框架作为理解和激发创造力的有前途的基础的潜力，从而铺平了通往强大AI发展的道路。最后，我们对未来的研究方向和潜在进展进行了反思。 

---
# Hidformer: Transformer-Style Neural Network in Stock Price Forecasting 

**Title (ZH)**: Hidformer：用于股价预测的Transformer风格神经网络 

**Authors**: Kamil Ł. Szydłowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2412.19932)  

**Abstract**: This paper investigates the application of Transformer-based neural networks to stock price forecasting, with a special focus on the intersection of machine learning techniques and financial market analysis. The evolution of Transformer models, from their inception to their adaptation for time series analysis in financial contexts, is reviewed and discussed. Central to our study is the exploration of the Hidformer model, which is currently recognized for its promising performance in time series prediction. The primary aim of this paper is to determine whether Hidformer will also prove itself in the task of stock price prediction. This slightly modified model serves as the framework for our experiments, integrating the principles of technical analysis with advanced machine learning concepts to enhance stock price prediction accuracy. We conduct an evaluation of the Hidformer model's performance, using a set of criteria to determine its efficacy. Our findings offer additional insights into the practical application of Transformer architectures in financial time series forecasting, highlighting their potential to improve algorithmic trading strategies, including human decision making. 

**Abstract (ZH)**: 本文研究了基于Transformer的神经网络在股票价格预测中的应用，特别关注机器学习技术与金融市场分析的交集。从Transformer模型的起源到其适应金融时间序列分析的演变过程进行了回顾和讨论。本研究的核心是探索Hidformer模型，该模型目前因其在时间序列预测中的良好表现而受到认可。本文的主要目标是确定Hidformer是否也能在股票价格预测任务中表现出色。这一稍作修改的模型作为我们实验的基础框架，将技术分析的原则与先进的机器学习概念相结合，以提高股票价格预测的准确性。我们对Hidformer模型的性能进行了评估，并使用一系列标准来确定其有效性。我们的研究结果进一步揭示了Transformer架构在金融时间序列预测中的实际应用潜力，强调了其在改善算法交易策略中的作用，包括增强人类决策能力。 

---
# Pivoting B2B platform business models: From platform experimentation to multi-platform integration to ecosystem envelopment 

**Title (ZH)**: 转型面向企业的平台商业模式：从平台实验到多平台整合再到生态系统包覆 

**Authors**: Clara Filosa, Marin Jovanovic, Lara Agostini, Anna Nosella  

**Link**: [PDF](https://arxiv.org/pdf/2412.19931)  

**Abstract**: The landscape of digital servitization in the manufacturing sector is evolving, marked by a strategic shift from traditional product-centric to platform business models (BMs). Manufacturing firms often employ a blend of approaches to develop business-to-business (B2B) platforms, leading to significant reconfigurations in their BMs. However, they frequently encounter failures in their B2B platform development initiatives, leading them to abandon initial efforts and pivot to alternative platform strategies. Therefore, this study, through an in-depth case study of a manufacturer in the energy sector, articulates a three-phase pivoting framework for B2B platform BMs, including platform development and platform strategy. Initially, the manufacturer focused on asset-based product sales supplemented by asset maintenance services and followed an emergent platformization strategy characterized by the rise of multiple, independent B2B platforms catering to diverse functions. Next, focusing on the imposed customer journey strategy, the firm shifted towards a strategic multi-platform integration into an all-encompassing platform supported by artificial intelligence (AI), signaling a maturation of the platform BM to combine a wide range of services into an energy-performance-based contract. Finally, the last step of the firm's platform BM evolution consisted of a deliberate platform strategy open to external stakeholders and enveloping its data-driven offerings within a broader platform ecosystem. This article advances B2B platform BMs and digital servitization literature, highlighting the efficacy of a progressive approach and strategic pivoting. 

**Abstract (ZH)**: 制造业领域的数字服务化景观正在演变，从传统的以产品为中心的战略转向以平台为基础的商业模式（BM）。制造企业通常采用多种方式开发企业对企业（B2B）平台，导致其商业模式发生了显著重塑。然而，他们在B2B平台开发项目中常常遇到失败，迫使他们放弃初始努力并转向替代平台策略。因此，本文通过一个能源领域的制造商的深入案例研究，构建了一个包括平台开发和平台策略在内的三阶段转向框架。最初，该制造商专注于基于资产的产品销售，并提供资产维护服务，采用一种由多个独立B2B平台构成的随机制定的平台化策略，这些平台服务于不同的功能。随后，该企业重点转向强制性客户旅程策略，转向一种战略性的多平台集成，依托人工智能（AI）构建一个包容所有功能的平台，标志着平台BM向将多种服务整合到能效合同中的成熟阶段转变。最后，企业的平台BM演变的最终阶段是一种对外部相关方开放的平台策略，将其数据驱动的产品和服务纳入更广泛的平台生态系统之中。本文在B2B平台BM和数字服务化研究领域推进了相关研究，并强调了渐进式方法和战略转向的有效性。 

---
# Modeling Continuous Spatial-temporal Dynamics of Turbulent Flow with Test-time Refinement 

**Title (ZH)**: 基于测试时校准的湍流流动连续空时动态建模 

**Authors**: Shengyu Chen, Peyman Givi, Can Zheng, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2412.19927)  

**Abstract**: The precise simulation of turbulent flows holds immense significance across various scientific and engineering domains, including climate science, freshwater science, and energy-efficient manufacturing. Within the realm of simulating turbulent flows, large eddy simulation (LES) has emerged as a prevalent alternative to direct numerical simulation (DNS), offering computational efficiency. However, LES cannot accurately capture the full spectrum of turbulent transport scales and is present only at a lower spatial resolution. Reconstructing high-fidelity DNS data from the lower-resolution LES data is essential for numerous applications, but it poses significant challenges to existing super-resolution techniques, primarily due to the complex spatio-temporal nature of turbulent flows. This paper proposes a novel flow reconstruction approach that leverages physical knowledge to model flow dynamics. Different from traditional super-resolution techniques, the proposed approach uses LES data only in the testing phase through a degradation-based refinement approach to enforce physical constraints and mitigate cumulative reconstruction errors over time. Furthermore, a feature sampling strategy is developed to enable flow data reconstruction across different resolutions. The results on two distinct sets of turbulent flow data indicate the effectiveness of the proposed method in reconstructing high-resolution DNS data, preserving the inherent physical attributes of flow transport, and achieving DNS reconstruction at different resolutions. 

**Abstract (ZH)**: 流体中的湍流流动精确模拟在气候科学、淡水科学和能源高效制造等领域具有重要的科学和工程意义。在模拟湍流流动的领域中，大型涡尺度模拟（LES）已成为直接数值模拟（DNS）的一种流行替代方法，提供了计算效率。然而，LES不能准确捕捉湍流传输尺度的全频谱，并且其空间分辨率较低。从低分辨率的LES数据重建高保真的DNS数据对于许多应用至关重要，但现有的超分辨率技术面临显著挑战，主要是由于湍流流动的复杂时空特性。本文提出了一种新的流体重构方法，该方法利用物理知识来建模流体动力学。与传统的超分辨率技术不同，本文提出的方法仅在测试阶段使用LES数据，并通过基于退化的过程细化方法来施加物理约束，以减轻随时间积累的重构误差。此外，开发了一种特征采样策略，以实现不同分辨率的流体数据重构。在两组不同湍流流动数据集上的结果表明，所提出的方法在重建高分辨率DNS数据方面有效，能够保持流体传输的内在物理特性，并实现不同分辨率的DNS数据重构。 

---
# HADES: Hardware Accelerated Decoding for Efficient Speculation in Large Language Models 

**Title (ZH)**: HADES：硬件加速解码以在大型语言模型中高效推测 

**Authors**: Ze Yang, Yihong Jin, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.19925)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing by understanding and generating human-like text. However, the increasing demand for more sophisticated LLMs presents significant computational challenges due to their scale and complexity. This paper introduces Hardware Accelerated Decoding (HADES), a novel approach to enhance the performance and energy efficiency of LLMs. We address the design of an LLM accelerator with hardware-level speculative decoding support, a concept not previously explored in existing literature. Our work demonstrates how speculative decoding can significantly improve the efficiency of LLM operations, paving the way for more advanced and practical applications of these models. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过理解和生成类人类文本，已经彻底改变了自然语言处理。然而，对于更复杂和高级LLM的需求日益增长，给其规模和复杂性带来了显著的计算挑战。本文介绍了一种新的方法——硬件加速解码（HADES），旨在增强LLM的性能和能源效率。我们提出了一个具有硬件级推测性解码支持的LLM加速器设计，这是现有文献中未曾探索的概念。我们的研究证明了推测性解码如何显著提高LLM操作的效率，开启了这些模型更高级和实际应用的道路。 

---
# Identifying Cocoa Pollinators: A Deep Learning Dataset 

**Title (ZH)**: 识别可可授粉者：一个深度学习数据集 

**Authors**: Wenxiu Xu, Saba Ghorbani Bazegar, Dong Sheng, Manuel Toledo-Hernandez, ZhenZhong Lan, Thomas Cherico Wanger  

**Link**: [PDF](https://arxiv.org/pdf/2412.19915)  

**Abstract**: Cocoa is a multi-billion-dollar industry but research on improving yields through pollination remains limited. New embedded hardware and AI-based data analysis is advancing information on cocoa flower visitors, their identity and implications for yields. We present the first cocoa flower visitor dataset containing 5,792 images of Ceratopogonidae, Formicidae, Aphididae, Araneae, and Encyrtidae, and 1,082 background cocoa flower images. This dataset was curated from 23 million images collected over two years by embedded cameras in cocoa plantations in Hainan province, China. We exemplify the use of the dataset with different sizes of YOLOv8 models and by progressively increasing the background image ratio in the training set to identify the best-performing model. The medium-sized YOLOv8 model achieved the best results with 8% background images (F1 Score of 0.71, mAP50 of 0.70). Overall, this dataset is useful to compare the performance of deep learning model architectures on images with low contrast images and difficult detection targets. The data can support future efforts to advance sustainable cocoa production through pollination monitoring projects. 

**Abstract (ZH)**: 可可行业价值数百亿美元，但通过授粉提高产量的研究仍较为有限。新的嵌入式硬件和基于人工智能的数据分析正在推进对可可花访问者、其身份及其对产量影响的信息了解。我们呈现了首个包含5792张昆虫照片的数据集，这些昆虫属于Ceratopogonidae、Formicidae、Aphididae、Araneae和Encyrtidae科，以及1082张背景可可花朵照片。该数据集汇集了中国海南省可可种植园内嵌入式摄像头在两年内收集的共计2300万张图片。我们通过使用不同大小的YOLOv8模型，并逐步增加训练集中背景图片的比例，展示了该数据集的应用，并确定了性能最佳的模型。中型YOLOv8模型在8%背景图片条件下表现最佳，其F1分数为0.71，mAP50为0.70。总体而言，该数据集对于比较深度学习模型架构在低对比度图像和难以检测目标中的性能非常有用。数据可以支持未来通过授粉监测项目提高可持续可可生产的努力。 

---
# Leveraging Scene Geometry and Depth Information for Robust Image Deraining 

**Title (ZH)**: 利用场景几何结构和深度信息实现鲁棒去雨处理 

**Authors**: Ningning Xu, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.19913)  

**Abstract**: Image deraining holds great potential for enhancing the vision of autonomous vehicles in rainy conditions, contributing to safer driving. Previous works have primarily focused on employing a single network architecture to generate derained images. However, they often fail to fully exploit the rich prior knowledge embedded in the scenes. Particularly, most methods overlook the depth information that can provide valuable context about scene geometry and guide more robust deraining. In this work, we introduce a novel learning framework that integrates multiple networks: an AutoEncoder for deraining, an auxiliary network to incorporate depth information, and two supervision networks to enforce feature consistency between rainy and clear scenes. This multi-network design enables our model to effectively capture the underlying scene structure, producing clearer and more accurately derained images, leading to improved object detection for autonomous vehicles. Extensive experiments on three widely-used datasets demonstrated the effectiveness of our proposed method. 

**Abstract (ZH)**: 在雨天条件下增强自主驾驶车辆视觉的大雨去除技术具有巨大潜力，有助于提升驾驶安全性。先前的工作主要集中在使用单一网络架构生成去雨图像。然而，这些方法往往未能充分利用场景中嵌入的丰富先验知识。特别是，大多数方法忽视了深度信息的作用，这类信息能提供有关场景几何结构的重要上下文，并指导更稳健的去雨处理。在本文中，我们提出了一种新的学习框架，集成多个网络：一个自编码器用于去雨处理，一个辅助网络用于整合深度信息，以及两个监督网络以强制在雨天和清晰场景之间保持特征一致性。这种多网络设计使我们的模型能够有效地捕捉到场景的内在结构，生成更为清晰和准确的去雨图像，从而提高自主驾驶车辆的目标检测性能。在三个广泛使用的数据集上的大量实验验证了我们所提出方法的有效性。 

---
# Evaluate Summarization in Fine-Granularity: Auto Evaluation with LLM 

**Title (ZH)**: 在细粒度层面评估摘要生成：基于大语言模型的自动评估 

**Authors**: Dong Yuan, Eti Rastogi, Fen Zhao, Sagar Goyal, Gautam Naik, Sree Prasanna Rajagopal  

**Link**: [PDF](https://arxiv.org/pdf/2412.19906)  

**Abstract**: Due to the exponential growth of information and the need for efficient information consumption the task of summarization has gained paramount importance. Evaluating summarization accurately and objectively presents significant challenges, particularly when dealing with long and unstructured texts rich in content. Existing methods, such as ROUGE (Lin, 2004) and embedding similarities, often yield scores that have low correlation with human judgements and are also not intuitively understandable, making it difficult to gauge the true quality of the summaries. LLMs can mimic human in giving subjective reviews but subjective scores are hard to interpret and justify. They can be easily manipulated by altering the models and the tones of the prompts. In this paper, we introduce a novel evaluation methodology and tooling designed to address these challenges, providing a more comprehensive, accurate and interpretable assessment of summarization outputs. Our method (SumAutoEval) proposes and evaluates metrics at varying granularity levels, giving objective scores on 4 key dimensions such as completeness, correctness, Alignment and readability. We empirically demonstrate, that SumAutoEval enhances the understanding of output quality with better human correlation. 

**Abstract (ZH)**: 由于信息的指数级增长和高效信息消费的需要，总结任务的重要性日益凸显。准确和客观地评估总结是一项重大挑战，特别是在处理长且结构不规则、内容丰富的文本时。现有方法，如 ROUGE（Lin, 2004）和嵌入相似性，通常给出的评分与人类判断的相关性较低，且缺乏直观性，使得难以评估总结的真实质量。尽管大规模语言模型（LLMs）可以模拟人类给出的主观评价，但主观评分难以解释和验证，并且容易通过调整模型和提示的语气被操控。在本文中，我们提出了一种新的评估方法和工具，旨在解决上述挑战，提供更全面、准确和可解释的总结输出评估。我们的方法（SumAutoEval）提出了在不同粒度级别评估指标，并在四大关键维度（完整性、准确性、一致性与可读性）上给出客观评分。我们通过实证研究证明，SumAutoEval 提高了输出质量理解的直观性和与人类判断的相关性。 

---
# A Fully Hardware Implemented Accelerator Design in ReRAM Analog Computing without ADCs 

**Title (ZH)**: 一种在电阻式随机存取内存（ReRAM）模拟计算中不使用模数转换器（ADC）的全硬件实现加速器设计 

**Authors**: Peng Dang, Huawei Li, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.19869)  

**Abstract**: Emerging ReRAM-based accelerators process neural networks via analog Computing-in-Memory (CiM) for ultra-high energy efficiency. However, significant overhead in peripheral circuits and complex nonlinear activation modes constrain system energy efficiency improvements. This work explores the hardware implementation of the Sigmoid and SoftMax activation functions of neural networks with stochastically binarized neurons by utilizing sampled noise signals from ReRAM devices to achieve a stochastic effect. We propose a complete ReRAM-based Analog Computing Accelerator (RACA) that accelerates neural network computation by leveraging stochastically binarized neurons in combination with ReRAM crossbars. The novel circuit design removes significant sources of energy/area efficiency degradation, i.e., the Digital-to-Analog and Analog-to-Digital Converters (DACs and ADCs) as well as the components to explicitly calculate the activation functions. Experimental results show that our proposed design outperforms traditional architectures across all overall performance metrics without compromising inference accuracy. 

**Abstract (ZH)**: 基于ReRAM的新兴加速器通过存内模拟计算（Computing-in-Memory, CiM）处理神经网络，实现了超高的能效。然而，外围电路的显著开销和复杂的非线性激活模式限制了系统的能效改进。本研究探讨了利用ReRAM器件采样噪声信号实现随机化效应的方法，以硬件形式实现神经网络中的Sigmoid和SoftMax激活函数，使用随机二值化神经元。我们提出了一种基于ReRAM的模拟计算加速器（RACA），该加速器通过结合使用ReRAM交叉栏和随机二值化神经元加速神经网络计算。这种新型电路设计消除了能效/面积效率退化的显著来源，即数字-模拟转换器（DACs）和模拟-数字转换器（ADCs），以及显式计算激活函数的组件。实验结果表明，我们的设计在所有综合性能指标上优于传统架构，同时不牺牲推理准确性。 

---
# Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales 

**Title (ZH)**: 基于可学习比例系数的数据驱动分组全量化的Winograd卷积 

**Authors**: Shuokai Pan, Gerti Tuzi, Sudarshan Sreeram, Dibakar Gope  

**Link**: [PDF](https://arxiv.org/pdf/2412.19867)  

**Abstract**: Despite the revolutionary breakthroughs of large-scale textto-image diffusion models for complex vision and downstream tasks, their extremely high computational and storage costs limit their usability. Quantization of diffusion models has been explored in recent works to reduce compute costs and memory bandwidth usage. To further improve inference time, fast convolution algorithms such as Winograd can be used for convolution layers, which account for a significant portion of computations in diffusion models. However, the significant quality loss of fully quantized Winograd using existing coarser-grained post-training quantization methods, combined with the complexity and cost of finetuning the Winograd transformation matrices for such large models to recover quality, makes them unsuitable for large-scale foundation models. Motivated by the presence of a large range of values in them, we investigate the impact of finer-grained group-wise quantization in quantizing diffusion models. While group-wise quantization can largely handle the fully quantized Winograd convolution, it struggles to deal with the large distribution imbalance in a sizable portion of the Winograd domain computation. To reduce range differences in the Winograd domain, we propose finetuning only the scale parameters of the Winograd transform matrices without using any domain-specific training data. Because our method does not depend on any training data, the generalization performance of quantized diffusion models is safely guaranteed. For text-to-image generation task, the 8-bit fully-quantized diffusion model with Winograd provides near-lossless quality (FID and CLIP scores) in comparison to the full-precision model. For image classification, our method outperforms the state-of-the-art Winograd PTQ method by 1.62% and 2.56% in top-1 ImageNet accuracy on ResNet18 and ResNet-34, respectively, with Winograd F(6, 3). 

**Abstract (ZH)**: 尽管大规模文本到图像扩散模型在复杂视觉和下游任务方面取得了革命性的突破，但其极高的计算和存储成本限制了其实用性。最近的研究工作已经探索了扩散模型的量化方法，以降低计算成本和内存带宽使用。为了进一步缩短推理时间，可以使用Winograd等快速卷积算法处理卷积层，卷积层在扩散模型中的计算量占比较高。然而，现有粗粒度后训练量化方法导致的全量化Winograd显著的质量损失，以及对如此大规模模型微调Winograd变换矩阵以恢复质量的复杂性和成本，使得这种方法不适合大型基础模型。鉴于它们值的范围很大，我们探讨了在量化扩散模型中使用细粒度分组量化的影响。虽然分组量化可以很好地处理全量化的Winograd卷积，但在Winograd域计算中相当大部分的大分布不平衡部分，它无法有效应对。为了减少Winograd域中的范围差异，我们仅微调Winograd变换矩阵的比例参数，而不使用任何特定领域的训练数据。由于我们的方法不依赖任何训练数据，量化扩散模型的一般化性能得到了安全保证。对于文本到图像生成任务，使用Winograd的8位全量化扩散模型在FID和CLIP评分上几乎与全精度模型保持一致。对于图像分类，与使用Winograd F(6, 3)的ResNet18相比，我们的方法在ImageNet上的Top-1准确性提高了1.62%，而在ResNet-34上提高了2.56%，超过了最先进的Winograd后训练量化方法。 

---
# Fusion of Deep Learning and GIS for Advanced Remote Sensing Image Analysis 

**Title (ZH)**: 基于深度学习与地理信息系统融合的高级遥感图像分析 

**Authors**: Sajjad Afroosheh, Mohammadreza Askari  

**Link**: [PDF](https://arxiv.org/pdf/2412.19856)  

**Abstract**: This paper presents an innovative framework for remote sensing image analysis by fusing deep learning techniques, specifically Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks, with Geographic Information Systems (GIS). The primary objective is to enhance the accuracy and efficiency of spatial data analysis by overcoming challenges associated with high dimensionality, complex patterns, and temporal data processing. We implemented optimization algorithms, namely Particle Swarm Optimization (PSO) and Genetic Algorithms (GA), to fine-tune model parameters, resulting in improved performance metrics. Our findings reveal a significant increase in classification accuracy from 78% to 92% and a reduction in prediction error from 12% to 6% after optimization. Additionally, the temporal accuracy of the models improved from 75% to 88%, showcasing the frameworks capability to monitor dynamic changes effectively. The integration of GIS not only enriched the spatial analysis but also facilitated a deeper understanding of the relationships between geographical features. This research demonstrates that combining advanced deep learning methods with GIS and optimization strategies can significantly advance remote sensing applications, paving the way for future developments in environmental monitoring, urban planning, and resource management. 

**Abstract (ZH)**: 本文提出了一种创新框架，通过融合深度学习技术，特别是卷积神经网络（CNNs）和长短期记忆网络（LSTM），与地理信息系统（GIS）相结合，以改进遥感图像分析。主要目标是通过克服高维数据、复杂模式以及时空数据处理带来的挑战，提高空间数据分析的准确性和效率。我们采用优化算法，如粒子群优化（PSO）和遗传算法（GA），对模型参数进行了微调，从而提高了性能指标。研究结果表明，在优化后，分类准确率从78%提高到了92%，预测误差从12%降低到了6%。此外，模型的时效性准确率从75%提高到了88%，展示了该框架在有效监测动态变化方面的潜力。GIS的集成不仅丰富了空间分析的内容，还促进了对地理特征之间关系的更深入理解。本研究表明，结合先进的深度学习方法、GIS以及优化策略可以显著推进遥感应用的发展，为环境监测、城市规划和资源管理领域的未来技术发展铺平了道路。 

---
# Symbolic Disentangled Representations for Images 

**Title (ZH)**: 图像的象征性解耦表示 

**Authors**: Alexandr Korchemnyi, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2412.19847)  

**Abstract**: The idea of disentangled representations is to reduce the data to a set of generative factors that produce it. Typically, such representations are vectors in latent space, where each coordinate corresponds to one of the generative factors. The object can then be modified by changing the value of a particular coordinate, but it is necessary to determine which coordinate corresponds to the desired generative factor -- a difficult task if the vector representation has a high dimension. In this article, we propose ArSyD (Architecture for Symbolic Disentanglement), which represents each generative factor as a vector of the same dimension as the resulting representation. In ArSyD, the object representation is obtained as a superposition of the generative factor vector representations. We call such a representation a \textit{symbolic disentangled representation}. We use the principles of Hyperdimensional Computing (also known as Vector Symbolic Architectures), where symbols are represented as hypervectors, allowing vector operations on them. Disentanglement is achieved by construction, no additional assumptions about the underlying distributions are made during training, and the model is only trained to reconstruct images in a weakly supervised manner. We study ArSyD on the dSprites and CLEVR datasets and provide a comprehensive analysis of the learned symbolic disentangled representations. We also propose new disentanglement metrics that allow comparison of methods using latent representations of different dimensions. ArSyD allows to edit the object properties in a controlled and interpretable way, and the dimensionality of the object property representation coincides with the dimensionality of the object representation itself. 

**Abstract (ZH)**: 解缠表示的概念是将数据简化为一组生成因子，这些生成因子能够生成数据。通常情况下，这些表示是潜在空间中的向量，每个坐标对应一个生成因子。通过改变特定坐标的值，可以修改对象，但需要确定哪个坐标的值对应于所需的生成因子，这在向量表示具有高维度时是一项困难的任务。本文提出了一种名为ArSyD（符号解缠架构）的方法，其中每个生成因子表示为与最终表示维度相同的向量。在ArSyD中，对象的表示是生成因子向量表示的叠加。我们称这样的表示为“符号解缠表示”。我们采用了超维计算（也称为向量符号架构）的原则，其中符号以超向量的形式表示，允许对其执行向量操作。通过构造实现解缠，训练过程中不做关于潜在分布的额外假设，仅通过弱监督的方式训练模型以重建图像。我们对ArSyD在dSprites和CLEVR数据集上进行了研究，并对其学习到的符号解缠表示进行了全面分析。我们还提出了一种新的解缠度量标准，允许使用不同维度的潜在表示来比较方法。ArSyD允许以受控和可解释的方式修改对象属性，且对象属性表示的维度与其自身表示的维度相同。 

---
# Unveiling Secrets of Brain Function With Generative Modeling: Motion Perception in Primates & Cortical Network Organization in Mice 

**Title (ZH)**: 借助生成建模揭示大脑功能的秘密：灵长类动物的运动知觉与小鼠的皮层网络组织 

**Authors**: Hadi Vafaii  

**Link**: [PDF](https://arxiv.org/pdf/2412.19845)  

**Abstract**: This Dissertation is comprised of two main projects, addressing questions in neuroscience through applications of generative modeling.
Project #1 (Chapter 4) explores how neurons encode features of the external world. I combine Helmholtz's "Perception as Unconscious Inference" -- paralleled by modern generative models like variational autoencoders (VAE) -- with the hierarchical structure of the visual cortex. This combination leads to the development of a hierarchical VAE model, which I test for its ability to mimic neurons from the primate visual cortex in response to motion stimuli. Results show that the hierarchical VAE perceives motion similar to the primate brain. Additionally, the model identifies causal factors of retinal motion inputs, such as object- and self-motion, in a completely unsupervised manner. Collectively, these results suggest that hierarchical inference underlines the brain's understanding of the world, and hierarchical VAEs can effectively model this understanding.
Project #2 (Chapter 5) investigates the spatiotemporal structure of spontaneous brain activity and its reflection of brain states like rest. Using simultaneous fMRI and wide-field Ca2+ imaging data, this project demonstrates that the mouse cortex can be decomposed into overlapping communities, with around half of the cortical regions belonging to multiple communities. Comparisons reveal similarities and differences between networks inferred from fMRI and Ca2+ signals.
The introduction (Chapter 1) is divided similarly to this abstract: sections 1.1 to 1.8 provide background information about Project #1, and sections 1.9 to 1.13 are related to Project #2. Chapter 2 includes historical background, Chapter 3 provides the necessary mathematical background, and finally, Chapter 6 contains concluding remarks and future directions. 

**Abstract (ZH)**: 这篇论文由两个主要项目组成，通过生成模型的应用来解答神经科学方面的问题。

项目 #1（第 4 章）探讨了神经元如何编码外部世界的特征。我将赫尔姆霍茨的“感知即无意识推理”与现代生成模型（如变分自编码器 VAE）相提并论，并结合视觉皮层的层次结构。这一结合导致开发出一个层次结构的 VAE 模型，该模型被测试以模拟灵长类动物视觉皮层对运动刺激的反应。结果表明，层次结构的 VAE 在感知运动方面类似于灵长类大脑的功能。此外，该模型在未监督的情况下识别了视网膜运动输入的原因，例如物体-自身运动等。综上所述，这些结果表明层次结构的推理是大脑理解世界的基础，并且层次结构的 VAE 可有效地模拟这种理解。

项目 #2（第 5 章）探讨自发脑活动的空间-时间结构及其对大脑状态（如休息状态）的反映。本项目利用同时进行的 fMRI 和宽场 Ca2+ 成像数据表明，小鼠皮层可以被分解为重叠的社区，大约一半的皮层区域属于多个社区。通过比较 fMRI 信号和 Ca2+ 信号推断的网络，揭示了它们之间的相似性和差异性。

引言（第 1 章）的结构与本摘要相似：第 1.1 至 1.8 节提供了项目 #1 的背景信息，第 1.9 至 1.13 节与项目 #2 相关。第 2 章包含历史背景，第 3 章提供了必要的数学背景，最后，第 6 章包含了总结和未来方向。 

---
# A Review of Latent Representation Models in Neuroimaging 

**Title (ZH)**: 神经影像学中潜在表示模型的综述 

**Authors**: C. Vázquez-García, F. J. Martínez-Murcia, F. Segovia Román, Juan M. Górriz  

**Link**: [PDF](https://arxiv.org/pdf/2412.19844)  

**Abstract**: Neuroimaging data, particularly from techniques like MRI or PET, offer rich but complex information about brain structure and activity. To manage this complexity, latent representation models - such as Autoencoders, Generative Adversarial Networks (GANs), and Latent Diffusion Models (LDMs) - are increasingly applied. These models are designed to reduce high-dimensional neuroimaging data to lower-dimensional latent spaces, where key patterns and variations related to brain function can be identified. By modeling these latent spaces, researchers hope to gain insights into the biology and function of the brain, including how its structure changes with age or disease, or how it encodes sensory information, predicts and adapts to new inputs. This review discusses how these models are used for clinical applications, like disease diagnosis and progression monitoring, but also for exploring fundamental brain mechanisms such as active inference and predictive coding. These approaches provide a powerful tool for both understanding and simulating the brain's complex computational tasks, potentially advancing our knowledge of cognition, perception, and neural disorders. 

**Abstract (ZH)**: 神经成像数据，尤其是来自MRI或PET等技术的数据，提供了关于大脑结构和功能的丰富但复杂的信息。为了管理这种复杂性，潜在表征模型如自动编码器（Autoencoders）、生成对抗网络（GANs）和潜在扩散模型（LDMs）逐渐被应用于这一领域。这些模型旨在将高维神经成像数据简化为低维潜在空间，在这些潜在空间中可以识别出与大脑功能相关的关键模式和变化。通过建模这些潜在空间，研究人员希望能够深入了解大脑的生物学和功能，包括其结构如何随着年龄或疾病而变化，或如何编码感官信息、预测和适应新的输入。本文综述了这些模型在临床应用中的使用情况，例如疾病诊断和疾病进展监测，同时也探讨了它们在探索基本的大脑机制（如主动推理和预测编码）方面的应用。这些方法为理解和模拟大脑复杂计算任务提供了强大的工具，有望推动我们对认知、知觉和神经疾病的理解。 

---
# RoboSignature: Robust Signature and Watermarking on Network Attacks 

**Title (ZH)**: RoboSignature：网络攻击中的稳健签名与数字水印 

**Authors**: Aryaman Shaan, Garvit Banga, Raghav Mantri  

**Link**: [PDF](https://arxiv.org/pdf/2412.19834)  

**Abstract**: Generative models have enabled easy creation and generation of images of all kinds given a single prompt. However, this has also raised ethical concerns about what is an actual piece of content created by humans or cameras compared to model-generated content like images or videos. Watermarking data generated by modern generative models is a popular method to provide information on the source of the content. The goal is for all generated images to conceal an invisible watermark, allowing for future detection or identification. The Stable Signature finetunes the decoder of Latent Diffusion Models such that a unique watermark is rooted in any image produced by the decoder. In this paper, we present a novel adversarial fine-tuning attack that disrupts the model's ability to embed the intended watermark, exposing a significant vulnerability in existing watermarking methods. To address this, we further propose a tamper-resistant fine-tuning algorithm inspired by methods developed for large language models, tailored to the specific requirements of watermarking in LDMs. Our findings emphasize the importance of anticipating and defending against potential vulnerabilities in generative systems. 

**Abstract (ZH)**: 生成模型使得仅基于单个提示即可轻松创建和生成各种类型的图像成为可能。然而，这也引发了关于哪些图像实际上是人类或相机创作的内容，而哪些是由生成模型生成的内容（如图像或视频）的伦理问题。对现代生成模型生成的数据进行水印标记是提供内容来源信息的一种流行方法。目标是在所有生成的图像中隐藏一个不可见的水印，以供将来进行检测或识别。Stable Signature 通过调整潜在扩散模型（Latent Diffusion Models, LDMs）的解码器，使得任何由解码器生成的图像中都包含一个独特的水印。在本文中，我们提出了一种新的对抗微调攻击方法，旨在破坏模型嵌入预定水印的能力，揭示了现有水印方法中的显著漏洞。为应对这一问题，我们进一步提出了一种抗篡改的微调算法，该算法借鉴了为大型语言模型开发的方法，针对LDMs中水印的具体需求进行了定制。我们的研究强调了预见并防御生成系统潜在漏洞的重要性。 

---
# Multi-atlas Ensemble Graph Neural Network Model For Major Depressive Disorder Detection Using Functional MRI Data 

**Title (ZH)**: 使用功能磁共振成像数据的多图谱ensemble图神经网络模型在检测主要抑郁障碍中的应用 

**Authors**: Nojod M. Alotaibi, Areej M. Alhothali, Manar S. Ali  

**Link**: [PDF](https://arxiv.org/pdf/2412.19833)  

**Abstract**: Major depressive disorder (MDD) is one of the most common mental disorders, with significant impacts on many daily activities and quality of life. It stands as one of the most common mental disorders globally and ranks as the second leading cause of disability. The current diagnostic approach for MDD primarily relies on clinical observations and patient-reported symptoms, overlooking the diverse underlying causes and pathophysiological factors contributing to depression. Therefore, scientific researchers and clinicians must gain a deeper understanding of the pathophysiological mechanisms involved in MDD. There is growing evidence in neuroscience that depression is a brain network disorder, and the use of neuroimaging, such as magnetic resonance imaging (MRI), plays a significant role in identifying and treating MDD. Rest-state functional MRI (rs-fMRI) is among the most popular neuroimaging techniques used to study MDD. Deep learning techniques have been widely applied to neuroimaging data to help with early mental health disorder detection. Recent years have seen a rise in interest in graph neural networks (GNNs), which are deep neural architectures specifically designed to handle graph-structured data like rs-fMRI. This research aimed to develop an ensemble-based GNN model capable of detecting discriminative features from rs-fMRI images for the purpose of diagnosing MDD. Specifically, we constructed an ensemble model by combining features from multiple brain region segmentation atlases to capture brain complexity and detect distinct features more accurately than single atlas-based models. Further, the effectiveness of our model is demonstrated by assessing its performance on a large multi-site MDD dataset. The best performing model among all folds achieved an accuracy of 75.80%, a sensitivity of 88.89%, a specificity of 61.84%, a precision of 71.29%, and an F1-score of 79.12%. 

**Abstract (ZH)**: 重度抑郁症（MDD）是常见的精神疾病之一，对日常生活和生活质量产生了重要影响。在全球范围内，它是最常见的精神疾病之一，并且是导致残疾的第二大原因。目前，用于诊断MDD的主要方法依赖于临床观察和患者报告的症状，忽视了抑郁症背后多样化的潜在成因和病理生理机制。因此，科学家和临床医生需要更加深入地理解MDD的病理生理机制。神经科学领域的研究证据越来越多地表明，抑郁症是一种脑网络障碍，而神经影像技术，如磁共振成像（MRI），在识别和治疗MDD方面发挥了重要作用。静息态功能磁共振成像（rs-fMRI）是最常用的神经影像技术之一，用于研究MDD。深度学习技术已被广泛应用于神经影像数据，以帮助早期检测心理健康障碍。近年来，图神经网络（GNNs）引起了越来越多的研究兴趣，这是一种专门设计来处理图结构数据（如rs-fMRI）的深度神经架构。本研究旨在开发一种基于集成的GNN模型，该模型可以从rs-fMRI图像中检测出诊断MDD的辨别性特征。具体而言，我们通过结合多个脑区分割图谱的特点构建了一个集成模型，以捕捉脑部的复杂性并更准确地检测差异性特征，超越了基于单一图谱的模型。此外，通过在大型多中心MDD数据集上的性能评估，证明了我们模型的有效性。所有折叠中表现最佳的模型的准确性为75.80%，敏感性为88.89%，特异性为61.84%，精确度为71.29%，F1分数为79.12%。 

---
# Back To The Future: A Hybrid Transformer-XGBoost Model for Action-oriented Future-proofing Nowcasting 

**Title (ZH)**: 回到未来：一种混合Transformer-XGBoost模型的行动导向型未来证明现值预测

在这个翻译中，我尽量保持了原文的学术语气和专业术语，并进行了适当的中文表达优化，确保符合学术规范。原标题中的“Nowcasting”通常指的是“现值预测”或“即时预测”，在学术文章中可以直接使用这个术语。 

**Authors**: Ziheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2412.19832)  

**Abstract**: Inspired by the iconic movie Back to the Future, this paper explores an innovative adaptive nowcasting approach that reimagines the relationship between present actions and future outcomes. In the movie, characters travel through time to manipulate past events, aiming to create a better future. Analogously, our framework employs predictive insights about the future to inform and adjust present conditions. This dual-stage model integrates the forecasting power of Transformers (future visionary) with the interpretability and efficiency of XGBoost (decision maker), enabling a seamless loop of future prediction and present adaptation. Through experimentation with meteorological datasets, we demonstrate the framework's advantage in achieving more accurate forecasting while guiding actionable interventions for real-time applications. 

**Abstract (ZH)**: 受经典电影《回到未来》的启发，本文探讨了一种创新的自适应现在预测方法，重新定义了当前行为与未来结果之间的关系。在电影中，角色穿越时间来操控过去的事件，以期创造一个更美好的未来。与此类似，我们的框架利用对未来洞察来指导并调整当前条件。该双阶段模型结合了Transformer（未来预见者）的预测能力和XGBoost（决策制定者）的可解释性和效率，从而实现了未来预测与当前调整之间的无缝循环。通过气象数据集的实验，我们展示了该框架在获得更准确预测的同时，能够指导实时应用中的可操作干预措施的优势。 

---
# A Unified Framework for Context-Aware IoT Management and State-of-the-Art IoT Traffic Anomaly Detection 

**Title (ZH)**: 一种面向上下文的物联网管理统一框架及物联网流量异常检测现状 

**Authors**: Daniel Adu Worae, Athar Sheikh, Spyridon Mastorakis  

**Link**: [PDF](https://arxiv.org/pdf/2412.19830)  

**Abstract**: The rapid expansion of Internet of Things (IoT) ecosystems has introduced growing complexities in device management and network security. To address these challenges, we present a unified framework that combines context-driven large language models (LLMs) for IoT administrative tasks with a fine-tuned anomaly detection module for network traffic analysis. The framework streamlines administrative processes such as device management, troubleshooting, and security enforcement by harnessing contextual knowledge from IoT manuals and operational data. The anomaly detection model achieves state-of-the-art performance in identifying irregularities and threats within IoT traffic, leveraging fine-tuning to deliver exceptional accuracy. Evaluations demonstrate that incorporating relevant contextual information significantly enhances the precision and reliability of LLM-based responses for diverse IoT administrative tasks. Additionally, resource usage metrics such as execution time, memory consumption, and response efficiency demonstrate the framework's scalability and suitability for real-world IoT deployments. 

**Abstract (ZH)**: 物联网（IoT）生态系统的迅速扩张带来了设备管理和网络安全日益复杂的挑战。为应对这些挑战，我们提出了一种统一框架，该框架结合了基于上下文的大规模语言模型（LLMs）以处理IoT管理任务，以及微调的异常检测模块以分析网络流量。该框架通过利用IoT手册和运营数据中的上下文信息，简化了诸如设备管理、故障排除和安全控制等管理流程。异常检测模型在识别IoT流量中的不规则性和威胁方面达到了最先进的性能，通过微调实现卓越的准确性。评估结果表明，结合相关上下文信息显著提高了基于LLM的响应在各种IoT管理任务中的精确性和可靠性。此外，资源使用指标，如执行时间、内存消耗和响应效率，证明了该框架的可扩展性和对实际IoT部署的适用性。 

---
# AnalogXpert: Automating Analog Topology Synthesis by Incorporating Circuit Design Expertise into Large Language Models 

**Title (ZH)**: AnalogXpert: 通过将电路设计专业知识融入大型语言模型来自动化模拟拓扑合成 

**Authors**: Haoyi Zhang, Shizhao Sun, Yibo Lin, Runsheng Wang, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2412.19824)  

**Abstract**: Analog circuits are crucial in modern electronic systems, and automating their design has attracted significant research interest. One of major challenges is topology synthesis, which determines circuit components and their connections. Recent studies explore large language models (LLM) for topology synthesis. However, the scenarios addressed by these studies do not align well with practical applications. Specifically, existing work uses vague design requirements as input and outputs an ideal model, but detailed structural requirements and device-level models are more practical. Moreover, current approaches either formulate topology synthesis as graph generation or Python code generation, whereas practical topology design is a complex process that demands extensive design knowledge. In this work, we propose AnalogXpert, a LLM-based agent aiming at solving practical topology synthesis problem by incorporating circuit design expertise into LLMs. First, we represent analog topology as SPICE code and introduce a subcircuit library to reduce the design space, in the same manner as experienced designers. Second, we decompose the problem into two sub-task (i.e., block selection and block connection) through the use of CoT and incontext learning techniques, to mimic the practical design process. Third, we introduce a proofreading strategy that allows LLMs to incrementally correct the errors in the initial design, akin to human designers who iteratively check and adjust the initial topology design to ensure accuracy. Finally, we construct a high-quality benchmark containing both real data (30) and synthetic data (2k). AnalogXpert achieves 40% and 23% success rates on the synthetic dataset and real dataset respectively, which is markedly better than those of GPT-4o (3% on both the synthetic dataset and the real dataset). 

**Abstract (ZH)**: 现代电子系统中，模拟电路至关重要，其自动化设计吸引了大量研究兴趣。其中一项主要挑战是拓扑合成，它决定了电路元件及其连接方式。近期的研究探讨了通过大型语言模型（LLM）进行拓扑合成的可行性，但是现有研究中的应用场景与实际应用不完全匹配。具体来说，现有工作使用模糊的设计要求作为输入，输出理想模型，但实际上，详细的结构要求和器件级模型更为实用。此外，当前的方法将拓扑合成要么形式化为图生成，要么形式化为Python代码生成，而实际的拓扑设计是一个复杂的过程，需要广泛的设计知识。在本文中，我们提出了AnalogXpert，这是一种基于LLM的代理，旨在通过将电路设计专业知识整合到LLM中来解决实际的拓扑合成问题。首先，我们将模拟拓扑表示为SPICE代码，并引入子电路库以减少设计空间，类似于经验丰富的设计师的做法。其次，我们通过使用CoT和上下文学习技术将问题分解为两个子任务（即模块选择和模块连接），以模拟实际设计过程。第三，我们引入了一种校对策略，使LLM能够逐步修正初始设计中的错误，类似于人类设计师通过迭代检查和调整初始拓扑设计来确保准确性。最后，我们构建了一个高性能基准，包含实际数据（30个）和合成数据（2000个）。AnalogXpert在合成数据集和实际数据集上的成功率分别为40%和23%，这比GPT-4o的性能要好得多（在合成数据集和实际数据集上均为3%）。 

---
# A Survey on Large Language Models for Communication, Network, and Service Management: Application Insights, Challenges, and Future Directions 

**Title (ZH)**: 大型语言模型在通信、网络和服务管理中的综述：应用洞察、挑战及未来方向 

**Authors**: Gordon Owusu Boateng, Hani Sami, Ahmed Alagha, Hanae Elmekki, Ahmad Hammoud, Rabeb Mizouni, Azzam Mourad, Hadi Otrok, Jamal Bentahar, Sami Muhaidat, Chamseddine Talhi, Zbigniew Dziong, Mohsen Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2412.19823)  

**Abstract**: The rapid evolution of communication networks in recent decades has intensified the need for advanced Network and Service Management (NSM) strategies to address the growing demands for efficiency, scalability, enhanced performance, and reliability of these networks. Large Language Models (LLMs) have received tremendous attention due to their unparalleled capabilities in various Natural Language Processing (NLP) tasks and generating context-aware insights, offering transformative potential for automating diverse communication NSM tasks. Contrasting existing surveys that consider a single network domain, this survey investigates the integration of LLMs across different communication network domains, including mobile networks and related technologies, vehicular networks, cloud-based networks, and fog/edge-based networks. First, the survey provides foundational knowledge of LLMs, explicitly detailing the generic transformer architecture, general-purpose and domain-specific LLMs, LLM model pre-training and fine-tuning, and their relation to communication NSM. Under a novel taxonomy of network monitoring and reporting, AI-powered network planning, network deployment and distribution, and continuous network support, we extensively categorize LLM applications for NSM tasks in each of the different network domains, exploring existing literature and their contributions thus far. Then, we identify existing challenges and open issues, as well as future research directions for LLM-driven communication NSM, emphasizing the need for scalable, adaptable, and resource-efficient solutions that align with the dynamic landscape of communication networks. We envision that this survey serves as a holistic roadmap, providing critical insights for leveraging LLMs to enhance NSM. 

**Abstract (ZH)**: 近年来，通信网络的迅速发展加强了对高级网络与服务管理（NSM）策略的需求，这些策略旨在提高网络的效率、可扩展性、性能和可靠性。大规模语言模型（LLMs）因其在各种自然语言处理（NLP）任务中无与伦比的能力以及生成上下文相关洞察的能力，受到了广泛关注，为自动化通信NSM任务提供了变革性的潜力。本文不同于现有的仅局限于单一网络领域的综述，探讨了LLMs在不同通信网络领域的集成，包括移动网络及相关的技术、车载网络、基于云的网络以及雾/边缘计算网络。首先，本综述提供了一般性的LLMs基础知识，详细介绍了通用变压器架构、领域通用和特定领域模型的预训练与微调，以及它们与通信NSM的关系。在新的网络监控与报告分类体系下，包括人工智能赋能的网络规划、网络部署与分布，以及持续网络支持，我们对每个不同网络领域的NSM任务中的LLM应用进行了广泛的分类，并综述现有文献及其迄今为止的贡献。然后，我们指出了现有的挑战和开放问题，并提出了未来研究的方向，强调了需要能够适应通信网络动态环境的可扩展、具有适应性且资源高效的解决方案。我们期望本文综述能够提供一个全面的路线图，为利用LLMs增强NSM提供关键的见解。 

---
# Nanoscaling Floating-Point (NxFP): NanoMantissa, Adaptive Microexponents, and Code Recycling for Direct-Cast Compression of Large Language Models 

**Title (ZH)**: Nanoscaling 浮点表示（NxFP）：纳米尾数、自适应微指数以及代码回收以实现直接铸型压缩大规模语言模型 

**Authors**: Yun-Chen Lo, Gu-Yeon Wei, David Brooks  

**Link**: [PDF](https://arxiv.org/pdf/2412.19821)  

**Abstract**: As cutting-edge large language models (LLMs) continue to transform various industries, their fast-growing model size and sequence length have led to memory traffic and capacity challenges. Recently, AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm have proposed a Microscaling standard (Mx), which augments block floating-point with microexponents to achieve promising perplexity-to-footprint trade-offs. However, the Microscaling suffers from significant perplexity degradation on modern LLMs with less than six bits. This paper profiles modern LLMs and identifies three main challenges of low-bit Microscaling format, i.e., inaccurate tracking of outliers, vacant quantization levels, and wasted binary code. In response, Nanoscaling (NxFP) proposes three techniques, i.e., NanoMantissa, Adaptive Microexponent, and Code Recycling to enable better accuracy and smaller memory footprint than state-of-the-art MxFP. Experimental results on direct-cast inference across various modern LLMs demonstrate that our proposed methods outperform state-of-the-art MxFP by up to 0.64 in perplexity and by up to 30% in accuracy on MMLU benchmarks. Furthermore, NxFP reduces memory footprint by up to 16% while achieving comparable perplexity as MxFP. 

**Abstract (ZH)**: 随着前沿的大语言模型（LLMs）继续变革各个行业，它们快速增长的模型规模和序列长度导致了内存带宽和容量的挑战。最近，AMD、Arm、Intel、Meta、Microsoft、NVIDIA和Qualcomm提出了微缩标度标准（Mx），该标准通过将微指数与块浮点数相结合，实现了令人满意的困惑度与足迹之间的权衡。然而，微缩标度在现代LLMs中表现不佳，尤其是对于少于六位比特的LLMs，其困惑度显著下降。本文概述了现代LLMs，并识别出了低比特微缩标度格式的三项主要挑战，即外部值跟踪不准确、空的量化级别以及浪费的二进制代码。为此，Nanoscaling（NxFP）提出了三种技术，即NanoMantissa、自适应微指数和代码回收，以实现比现有最优微浮点数（MxFP）方法更好的准确性和更小的内存足迹。在多种现代LLMs的直接转换推理实验中，我们提出的方法在困惑度上优于现有最优的MxFP方法最多0.64，在MMLU基准测试上的准确率上则提高了最多30%。此外，NxFP将内存足迹减少了最多16%，同时保持与MxFP相当的困惑度。 

---
# GaLore$+$: Boosting Low-Rank Adaptation for LLMs with Cross-Head Projection 

**Title (ZH)**: GaLore$+$: 通过跨头投影提升低秩适应性大模型性能 

**Authors**: Xutao Liao, Shaohui Li, Yuhui Xu, Zhi Li, Yu Liu, You He  

**Link**: [PDF](https://arxiv.org/pdf/2412.19820)  

**Abstract**: Recent low-rank training methods, such as GaLore, have significantly reduced the memory required to optimize large language models (LLMs). However, these methods often suffer from time-consuming low-rank projection estimations. In particular, the singular value decomposition (SVD) in GaLore can consume more than 80\% of the total training time. To address this issue, we propose GaLore$+$, which uses cross-head low-rank projection to reduce the substantial time consumption in estimating low-rank projections for multi-head attention. In addition, we employ randomized subspace iteration to achieve fast SVD. To further enhance performance, we propose sparsely coded residuals to reduce the errors caused by low-rank approximation on the first- and second-order moments of the optimizers and weight updates. We evaluate GaLore$+$ on arithmetic reasoning and natural language generation datasets. Our experiments demonstrate that GaLore$+$ delivers superior performance while achieving approximately $4\times$ fine-tuning speed compared to vanilla GaLore. 

**Abstract (ZH)**: 近年来，低秩训练方法如GaLore显著减少了优化大规模语言模型（LLMs）所需的内存。然而，这些方法往往因低秩投影估计耗时而受到限制。特别是，在GaLore中，奇异值分解（SVD）可能占用了总训练时间的80%以上。为解决这一问题，我们提出了GaLore$+$，该方法使用交叉头低秩投影来减少多头注意力中低秩投影估计的大量时间消耗。此外，我们采用随机子空间迭代来实现快速SVD。为了进一步提升性能，我们提出了稀疏编码残差，以减少低秩近似在优化器和权重更新的一阶和二阶矩中的误差。我们对GaLore$+$在算术推理和自然语言生成数据集上进行了评估。实验结果表明，与vanilla GaLore相比，GaLore$+$在大约4倍的微调速度下实现了更优的性能。 

---
# ChipAlign: Instruction Alignment in Large Language Models for Chip Design via Geodesic Interpolation 

**Title (ZH)**: ChipAlign：通过测地插值在大型语言模型中进行芯片设计的指令对齐 

**Authors**: Chenhui Deng, Yunsheng Bai, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.19819)  

**Abstract**: Recent advancements in large language models (LLMs) have expanded their application across various domains, including chip design, where domain-adapted chip models like ChipNeMo have emerged. However, these models often struggle with instruction alignment, a crucial capability for LLMs that involves following explicit human directives. This limitation impedes the practical application of chip LLMs, including serving as assistant chatbots for hardware design engineers. In this work, we introduce ChipAlign, a novel approach that utilizes a training-free model merging strategy, combining the strengths of a general instruction-aligned LLM with a chip-specific LLM. By considering the underlying manifold in the weight space, ChipAlign employs geodesic interpolation to effectively fuse the weights of input LLMs, producing a merged model that inherits strong instruction alignment and chip expertise from the respective instruction and chip LLMs. Our results demonstrate that ChipAlign significantly enhances instruction-following capabilities of existing chip LLMs, achieving up to a 26.6% improvement on the IFEval benchmark, while maintaining comparable expertise in the chip domain. This improvement in instruction alignment also translates to notable gains in instruction-involved QA tasks, delivering performance enhancements of 3.9% on the OpenROAD QA benchmark and 8.25% on production-level chip QA benchmarks, surpassing state-of-the-art baselines. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的应用范围不断扩大，现已延伸至集成电路设计领域，其中包括专门为集成电路设计而调整的模型，如ChipNeMo。然而，这些模型在指令对齐方面常常遇到挑战，这是LLMs中的关键能力之一，涉及遵循明确的人类指令。这一限制阻碍了集成电路LLMs的实际应用，包括作为硬件设计工程师助手聊天机器人的应用。本文介绍了一种名为ChipAlign的创新方法，该方法采用了一种无需训练的模型合并策略，结合了一般指令对齐LLM和特定于集成电路的LLM的优点。通过考虑权重空间中的潜在流形，ChipAlign采用测地线插值有效地融合了输入LLM的权重，生成了一个合并模型，该模型继承了来自相应指令和集成电路LLM的强大指令对齐能力和专业知识。我们的实验结果表明，ChipAlign显著增强了现有集成电路LLMs的指令遵循能力，在IFEval基准测试中取得了高达26.6%的提升，同时在集成电路领域保持了相当的专业知识。这种指令对齐能力的提升也转化为指令涉及问答任务的显著改进，在OpenROAD问答基准测试中实现了3.9%的性能提升，在生产级集成电路问答基准测试中实现了8.25%的性能提升，超过了最先进的基线方法。 

---
# Predicting Human Brain States with Transformer 

**Title (ZH)**: 使用变换器预测人类大脑状态 

**Authors**: Yifei Sun, Mariano Cabezas, Jiah Lee, Chenyu Wang, Wei Zhang, Fernando Calamante, Jinglei Lv  

**Link**: [PDF](https://arxiv.org/pdf/2412.19814)  

**Abstract**: The human brain is a complex and highly dynamic system, and our current knowledge of its functional mechanism is still very limited. Fortunately, with functional magnetic resonance imaging (fMRI), we can observe blood oxygen level-dependent (BOLD) changes, reflecting neural activity, to infer brain states and dynamics. In this paper, we ask the question of whether the brain states rep-resented by the regional brain fMRI can be predicted. Due to the success of self-attention and the transformer architecture in sequential auto-regression problems (e.g., language modelling or music generation), we explore the possi-bility of the use of transformers to predict human brain resting states based on the large-scale high-quality fMRI data from the human connectome project (HCP). Current results have shown that our model can accurately predict the brain states up to 5.04s with the previous 21.6s. Furthermore, even though the prediction error accumulates for the prediction of a longer time period, the gen-erated fMRI brain states reflect the architecture of functional connectome. These promising initial results demonstrate the possibility of developing gen-erative models for fMRI data using self-attention that learns the functional or-ganization of the human brain. Our code is available at: this https URL. 

**Abstract (ZH)**: 人类大脑是一个复杂且高度动态的系统，我们对其功能机制的理解仍然非常有限。幸运的是，功能性磁共振成像（fMRI）使我们能够观察血氧水平依赖（BOLD）变化，这些变化反映了神经活动，从而推断出大脑状态和动态。本文提出的问题是，在局部脑fMRI所代表的大脑状态下，能否进行预测。由于自我注意力和变压器架构在序列自回归问题（如语言建模或音乐生成）中的成功应用，我们探讨了利用变压器预测人类大脑静息状态的可能性，基于人类连接组计划（HCP）提供的大规模高质量fMRI数据。目前的结果表明，我们的模型可以在前21.6秒的基础上，准确预测下到5.04秒的大脑状态。即使在预测更长时间段时，预测误差累积，但生成的fMRI大脑状态仍然反映了功能连接组的结构。这些初步而有希望的结果证明了使用自我注意力学习人类大脑功能组织从而开发生成模型的可能性。我们的代码可在此处获得：[此链接]。 

---
# LINKs: Large Language Model Integrated Management for 6G Empowered Digital Twin NetworKs 

**Title (ZH)**: LINKs：大型语言模型集成管理在6G赋能数字孪生网络中的应用 

**Authors**: Shufan Jiang, Bangyan Lin, Yue Wu, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.19811)  

**Abstract**: In the rapidly evolving landscape of digital twins (DT) and 6G networks, the integration of large language models (LLMs) presents a novel approach to network management. This paper explores the application of LLMs in managing 6G-empowered DT networks, with a focus on optimizing data retrieval and communication efficiency in smart city scenarios. The proposed framework leverages LLMs for intelligent DT problem analysis and radio resource management (RRM) in fully autonomous way without any manual intervention. Our proposed framework -- LINKs, builds up a lazy loading strategy which can minimize transmission delay by selectively retrieving the relevant data. Based on the data retrieval plan, LLMs transform the retrieval task into an numerical optimization problem and utilizing solvers to build an optimal RRM, ensuring efficient communication across the network. Simulation results demonstrate the performance improvements in data planning and network management, highlighting the potential of LLMs to enhance the integration of DT and 6G technologies. 

**Abstract (ZH)**: 在数字孪生（DT）和6G网络迅速发展的背景下，大型语言模型（LLMs）的集成为网络管理提供了一种新颖的方法。本文探讨了LLMs在管理6G赋能的数字孪生网络中的应用，重点关注在智能城市场景中优化数据检索和通信效率的方法。所提出的框架利用LLMs进行智能的DT问题分析和无线资源管理（RRM），以完全自主的方式进行，无需任何人工干预。我们提出的框架——LINKs，构建了一种懒加载策略，通过有选择地检索相关数据，从而最小化传输延迟。基于数据检索计划，LLMs将检索任务转换为一个数值优化问题，并利用求解器构建最优的RRM，确保网络中高效的通信。仿真结果表明，该方法在数据规划和网络管理方面的性能改进，突显了LLMs在增强DT和6G技术集成方面的能力。 

---
# AI-driven Automation as a Pre-condition for Eudaimonia 

**Title (ZH)**: AI驱动的自动化作为幸福的先决条件 

**Authors**: Anastasia Siapka  

**Link**: [PDF](https://arxiv.org/pdf/2412.19808)  

**Abstract**: The debate surrounding the 'future of work' is saturated with alarmist warnings about the loss of work as an intrinsically valuable activity. Instead, the present doctoral research approaches this debate from the perspective of human flourishing (eudaimonia). It articulates a neo-Aristotelian interpretation according to which the prospect of mass AI-driven automation, far from being a threat, is rather desirable insofar as it facilitates humans' flourishing and, subsequently, their engagement in leisure. Drawing on virtue jurisprudence, this research further explores what this desirability may imply for the current legal order. 

**Abstract (ZH)**: 关于“工作未来”的辩论充斥着对工作作为内在有价值活动的丧失的夸张警告。相反，本博士论文从人类幸福（euōdaimonia）的角度出发探讨这一辩论。它提出了一种新亚里士多德式的解释，认为大规模由人工智能驱动的自动化前景并非威胁，而是值得向往的，因为它有助于促进人类的幸福，并随后使人们能够投身于休闲活动。借助美德法学，本研究进一步探讨这种向往对当前法律秩序可能意味着什么。 

---
# exLong: Generating Exceptional Behavior Tests with Large Language Models 

**Title (ZH)**: ExLong：使用大型语言模型生成异常行为测试 

**Authors**: Jiyang Zhang, Yu Liu, Pengyu Nie, Junyi Jessy Li, Milos Gligoric  

**Link**: [PDF](https://arxiv.org/pdf/2405.14619)  

**Abstract**: Many popular programming languages, including C#, Java, and Python, support exceptions. Exceptions are thrown during program execution if an unwanted event happens, e.g., a method is invoked with an illegal argument value. Software developers write exceptional behavior tests (EBTs) to check that their code detects unwanted events and throws appropriate exceptions. Prior research studies have shown the importance of EBTs, but those studies also highlighted that developers put most of their efforts on "happy paths", e.g., paths without unwanted events. To help developers fill the gap, we present the first framework, dubbed exLong, that automatically generates EBTs. exLong is a large language model instruction fine-tuned from CodeLlama and embeds reasoning about traces that lead to throw statements, conditional expressions that guard throw statements, and non-exceptional behavior tests that execute similar traces. We compare exLong with the state-of-the-art models for test generation (CAT-LM) and one of the strongest foundation models (GPT-4o), as well as with analysis-based tools for test generation (Randoop and EvoSuite). Our results show that exLong outperforms existing models and tools. Furthermore, we contributed several pull requests to open-source projects and 23 EBTs generated by exLong were already accepted. 

**Abstract (ZH)**: 许多流行的编程语言，包括C#、Java和Python，都支持异常处理。如果发生不希望的事件，例如方法调用时传入了非法参数值，这些编程语言就会抛出异常。软件开发人员编写异常行为测试（EBTs）以检查代码是否能够检测到不希望的事件并正确抛出异常。先前的研究已经表明EBTs的重要性，但这些研究也指出，开发人员通常将大部分精力集中在所谓的“成功路径”上，即没有不希望事件的路径。为了帮助开发人员弥补这一不足，我们提出了第一个名为exLong的框架，能够自动生成EBTs。exLong基于CodeLlama进行指令微调，并嵌入了关于导致抛出语句的调用跟踪、保护抛出语句的条件表达式以及执行类似跟踪的非异常行为测试的推理。我们将exLong与最先进测试生成模型（CAT-LM）和一个最强大的基础模型（GPT-4o）进行了比较，并与基于分析的测试生成工具（Randoop和EvoSuite）进行了比较。结果显示，exLong在多个方面优于现有模型和工具。此外，我们还为开源项目提交了几个代码请求，并且已经有23个由exLong生成的EBTs被接受。 

---
