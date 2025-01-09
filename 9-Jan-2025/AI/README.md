# Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Though 

**Title (ZH)**: 向LLM中引入系统二推理：学习如何通过元链式思考进行思考 

**Authors**: Violet Xiang, Charlie Snell, Kanishk Gandhi, Alon Albalak, Anikait Singh, Chase Blagden, Duy Phung, Rafael Rafailov, Nathan Lile, Dakota Mahan, Louis Castricato, Jan-Philipp Franken, Nick Haber, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2501.04682)  

**Abstract**: We propose a novel framework, Meta Chain-of-Thought (Meta-CoT), which extends traditional Chain-of-Thought (CoT) by explicitly modeling the underlying reasoning required to arrive at a particular CoT. We present empirical evidence from state-of-the-art models exhibiting behaviors consistent with in-context search, and explore methods for producing Meta-CoT via process supervision, synthetic data generation, and search algorithms. Finally, we outline a concrete pipeline for training a model to produce Meta-CoTs, incorporating instruction tuning with linearized search traces and reinforcement learning post-training. Finally, we discuss open research questions, including scaling laws, verifier roles, and the potential for discovering novel reasoning algorithms. This work provides a theoretical and practical roadmap to enable Meta-CoT in LLMs, paving the way for more powerful and human-like reasoning in artificial intelligence. 

**Abstract (ZH)**: 我们提出了一种新颖的框架，即元思维链（Meta-CoT），它通过明确建模达到特定思维链（CoT）所需的底层推理，扩展了传统的思维链方法。我们展示了最先进的模型表现出类似上下文搜索的行为，并探讨了通过过程监督、合成数据生成和搜索算法来生成元CoT的方法。最后，我们概述了一个具体的训练模型生成元CoT的管道，该管道结合了指令调优、线性化搜索轨迹以及训练后的强化学习。最后，我们讨论了开放的研究问题，包括扩展律、验证者角色以及发现新型推理算法的可能性。这项工作为在大语言模型（LLM）中实现元CoT提供了一个理论和实践的路线图，为人工智能中更强大和类人的推理铺平了道路。 

---
# MedCoDi-M: A Multi-Prompt Foundation Model for Multimodal Medical Data Generation 

**Title (ZH)**: MedCoDi-M：一种用于多模态医疗数据生成的多提示基础模型 

**Authors**: Daniele Molino, Francesco Di Feola, Eliodoro Faiella, Deborah Fazzini, Domiziana Santucci, Linlin Shen, Valerio Guarrasi, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2501.04614)  

**Abstract**: Artificial Intelligence is revolutionizing medical practice, enhancing diagnostic accuracy and healthcare delivery. However, its adaptation in medical settings still faces significant challenges, related to data availability and privacy constraints. Synthetic data has emerged as a promising solution to mitigate these issues, addressing data scarcity while preserving privacy. Recently, Latent Diffusion Models have emerged as a powerful tool for generating high-quality synthetic data. Meanwhile, the integration of different modalities has gained interest, emphasizing the need of models capable of handle multimodal medical this http URL approaches struggle to integrate complementary information and lack the ability to generate modalities simultaneously. To address this challenge, we present MedCoDi-M, a 6.77-billion-parameter model, designed for multimodal medical data generation, that, following Foundation Model paradigm, exploits contrastive learning and large quantity of data to build a shared latent space which capture the relationships between different data modalities. Further, we introduce the Multi-Prompt training technique, which significantly boosts MedCoDi-M's generation under different settings. We extensively validate MedCoDi-M: first we benchmark it against five competitors on the MIMIC-CXR dataset, a state-of-the-art dataset for Chest X-ray and radiological report generation. Secondly, we perform a Visual Turing Test with expert radiologists to assess the realism and clinical relevance of the generated data, ensuring alignment with real-world scenarios. Finally, we assess the utility of MedCoDi-M in addressing key challenges in the medical field, such as anonymization, data scarcity and imbalance learning. The results are promising, demonstrating the applicability of MedCoDi-M in medical contexts. Project page is at this https URL. 

**Abstract (ZH)**: 人工智能正在革命性地改变医疗实践，提高了诊断准确性和医疗服务。然而，其在医疗环境中的应用仍然面临着数据可用性和隐私限制的重大挑战。合成数据已成为缓解这些问题的潜在解决方案，既解决了数据稀缺性问题，又保护了隐私。最近，潜在扩散模型（Latent Diffusion Models）已成为生成高质量合成数据的强大工具。同时，对不同模态的集成引起了广泛关注，强调了能够处理多模态医疗数据的模型的需求。现有的方法在整合互补信息方面存在困难，并且缺乏同时生成不同模态的能力。为了解决这一挑战，我们提出了MedCoDi-M，这是一个67.7亿参数的模型，旨在生成多模态医疗数据。该模型遵循基础模型范式，利用对比学习和大量数据构建共享的潜在空间，捕捉不同数据模态之间的关系。此外，我们还引入了多提示训练技术，这显著提升了MedCoDi-M在不同设置下的生成效果。我们广泛验证了MedCoDi-M：首先，我们在MIMIC-CXR数据集上（该数据集是用于胸部X光片和放射报告生成的前沿数据集）将其与五个竞争对手进行基准测试。其次，我们通过专家放射科医生进行视觉图灵测试，以评估生成数据的现实性和临床相关性，确保与实际场景保持一致。最后，我们评估了MedCoDi-M在应对医疗领域中的关键挑战（如匿名化、数据稀缺性和不均衡学习）方面的实用性。结果非常令人鼓舞，展示了MedCoDi-M在医疗环境中的应用潜力。项目页面链接为：这个 <https://>。 

---
# InfiGUIAgent: A Multimodal Generalist GUI Agent with Native Reasoning and Reflection 

**Title (ZH)**: InfiGUIAgent：一种具有原生推理和反思能力的多模态通用GUI代理 

**Authors**: Yuhang Liu, Pengxiang Li, Zishu Wei, Congkai Xie, Xueyu Hu, Xinchen Xu, Shengyu Zhang, Xiaotian Han, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.04575)  

**Abstract**: Graphical User Interface (GUI) Agents, powered by multimodal large language models (MLLMs), have shown great potential for task automation on computing devices such as computers and mobile phones. However, existing agents face challenges in multi-step reasoning and reliance on textual annotations, limiting their effectiveness. We introduce \textit{InfiGUIAgent}, an MLLM-based GUI Agent trained with a two-stage supervised fine-tuning pipeline. Stage 1 enhances fundamental skills such as GUI understanding and grounding, while Stage 2 integrates hierarchical reasoning and expectation-reflection reasoning skills using synthesized data to enable native reasoning abilities of the agents. \textit{InfiGUIAgent} achieves competitive performance on several GUI benchmarks, highlighting the impact of native reasoning skills in enhancing GUI interaction for automation tasks. Resources are available at \url{this https URL}. 

**Abstract (ZH)**: 由多模态大规模语言模型（MLLMs）驱动的图形用户界面（GUI）代理已经在计算设备如计算机和手机上展示了在任务自动化方面的巨大潜力。然而，现有的代理在多步推理和依赖文本标注方面面临挑战，限制了它们的效果。我们提出了基于MLLM的\textit{InfiGUIAgent}，这是一种通过两阶段监督微调管道进行训练的GUI代理。第一阶段增强基础技能，如GUI理解与定位，第二阶段则通过合成数据整合层级推理和预期-反思推理技能，使代理具备本体推理能力。在多个GUI基准测试中，\textit{InfiGUIAgent}取得了竞争力的表现，突显了本体推理技能在提升GUI交互以适应自动化任务方面的效果。更多资源可参见\url{this https URL}。 

---
# Research on environment perception and behavior prediction of intelligent UAV based on semantic communication 

**Title (ZH)**: 基于语义通信的智能无人驾驶航空器环境感知与行为预测研究 

**Authors**: Kechong Ren, Li Gao, Qi Guan  

**Link**: [PDF](https://arxiv.org/pdf/2501.04480)  

**Abstract**: The convergence of drone delivery systems, virtual worlds, and blockchain has transformed logistics and supply chain management, providing a fast, and environmentally friendly alternative to traditional ground transportation methods;Provide users with a real-world experience, virtual service providers need to collect up-to-the-minute delivery information from edge devices. To address this challenge, 1) a reinforcement learning approach is introduced to enable drones with fast training capabilities and the ability to autonomously adapt to new virtual scenarios for effective resource allocation.2) A semantic communication framework for meta-universes is proposed, which utilizes the extraction of semantic information to reduce the communication cost and incentivize the transmission of information for meta-universe services.3) In order to ensure that user information security, a lightweight authentication and key agreement scheme is designed between the drone and the user by introducing blockchain technology. In our experiments, the drone adaptation performance is improved by about 35\%, and the local offloading rate can reach 90\% with the increase of the number of base stations. The semantic communication system proposed in this paper is compared with the Cross Entropy baseline model. Introducing blockchain technology the throughput of the transaction is maintained at a stable value with different number of drones. 

**Abstract (ZH)**: 无人机配送系统、虚拟世界和区块链的融合已经改变了物流和供应链管理领域，提供了比传统地面运输方法更快、更环保的替代方案；为用户提供真实的体验，虚拟服务提供商需要从边缘设备中收集最新的配送信息。为应对这一挑战，1) 引入了强化学习方法，使无人机具备快速训练能力和自主适应新虚拟场景的能力，从而实现有效的资源分配。2) 提出了用于元宇宙的语义通信框架，利用语义信息的提取来减少通信成本，并激励信息在元宇宙服务中的传输。3) 为确保用户信息的安全，通过引入区块链技术，在无人机和用户之间设计了一种轻量级的身份认证和密钥协商方案。在我们的实验中，无人机自适应性能提高了约35%，随着基站数量的增加，本地卸载率可达90%。本文提出的语义通信系统与Cross Entropy基线模型进行了比较，在不同数量的无人机下，引入区块链技术后交易吞吐量保持稳定值。 

---
# Hybrid Artificial Intelligence Strategies for Drone Navigation 

**Title (ZH)**: 无人机导航的混合人工智能策略 

**Authors**: Rubén San-Segundo, Lucía Angulo, Manuel Gil-Martín, David Carramiñana, Ana M. Bernardos  

**Link**: [PDF](https://arxiv.org/pdf/2501.04472)  

**Abstract**: Objective: This paper describes the development of hybrid artificial intelligence strategies for drone navigation. Methods: The navigation module combines a deep learning model with a rule-based engine depending on the agent state. The deep learning model has been trained using reinforcement learning. The rule-based engine uses expert knowledge to deal with specific situations. The navigation module incorporates several strategies to explain the drone decision based on its observation space, and different mechanisms for including human decisions in the navigation process. Finally, this paper proposes an evaluation methodology based on defining several scenarios and analyzing the performance of the different strategies according to metrics adapted to each scenario. Results: Two main navigation problems have been studied. For the first scenario (reaching known targets), it has been possible to obtain a 90% task completion rate, reducing significantly the number of collisions thanks to the rule-based engine. For the second scenario, it has been possible to reduce 20% of the time required to locate all the targets using the reinforcement learning model. Conclusions: Reinforcement learning is a very good strategy to learn policies for drone navigation, but in critical situations, it is necessary to complement it with a rule-based module to increase task success rate. 

**Abstract (ZH)**: 目标：本文描述了用于无人机导航的混合人工智能策略的发展。

方法：导航模块结合了基于深度学习的模型和基于规则的引擎，具体取决于代理的状态。深度学习模型通过强化学习进行了训练。基于规则的引擎利用专家知识处理特定情况。导航模块整合了几种策略，以基于观察空间解释无人机的决策，并引入了多种机制将人类决策纳入导航过程。最后，本文提出了一种评估方法，该方法基于定义多个场景，并根据适应每个场景的指标分析不同策略的性能。

结果：两大主要导航问题得到了研究。对于第一个场景（到达已知目标），通过基于规则的引擎可以实现90%的任务完成率，显著减少了碰撞次数。对于第二个场景，使用强化学习模型可以将定位所有目标所需的时间减少20%。

结论：强化学习是一个非常有效的策略，用于学习无人机导航策略，但在关键情况下，有必要通过引入基于规则的模块来增强任务的成功率。 

---
# A Digital Shadow for Modeling, Studying and Preventing Urban Crime 

**Title (ZH)**: 一种数字影子模型，用于犯罪建模、研究与预防 

**Authors**: Juan Palma-Borda, Eduardo Guzmán, María-Victoria Belmonte  

**Link**: [PDF](https://arxiv.org/pdf/2501.04435)  

**Abstract**: Crime is one of the greatest threats to urban security. Around 80 percent of the world's population lives in countries with high levels of criminality. Most of the crimes committed in the cities take place in their urban environments. This paper presents the development and validation of a digital shadow platform for modeling and simulating urban crime. This digital shadow has been constructed using data-driven agent-based modeling and simulation techniques, which are suitable for capturing dynamic interactions among individuals and with their environment. Our approach transforms and integrates well-known criminological theories and the expert knowledge of law enforcement agencies (LEA), policy makers, and other stakeholders under a theoretical model, which is in turn combined with real crime, spatial (cartographic) and socio-economic data into an urban model characterizing the daily behavior of citizens. The digital shadow has also been instantiated for the city of Malaga, for which we had over 300,000 complaints available. This instance has been calibrated with those complaints and other geographic and socio-economic information of the city. To the best of our knowledge, our digital shadow is the first for large urban areas that has been calibrated with a large dataset of real crime reports and with an accurate representation of the urban environment. The performance indicators of the model after being calibrated, in terms of the metrics widely used in predictive policing, suggest that our simulated crime generation matches the general pattern of crime in the city according to historical data. Our digital shadow platform could be an interesting tool for modeling and predicting criminal behavior in an urban environment on a daily basis and, thus, a useful tool for policy makers, criminologists, sociologists, LEAs, etc. to study and prevent urban crime. 

**Abstract (ZH)**: 犯罪是城市安全面临的最大威胁之一。全球约80%的人口居住在犯罪率较高的国家。大多数城市犯罪行为发生在城市环境中。本文介绍了用于建模和模拟城市犯罪的数字阴影平台的发展与验证过程。该数字阴影是基于数据驱动的基于代理的建模与仿真技术构建的，适用于捕捉个体之间及其与环境的动态交互。我们的方法将知名的犯罪学理论与执法机构（LEA）、决策者及其他利益相关者的专业知识整合到一个理论模型中，并结合了真实的犯罪数据、空间（制图）和社会经济数据，形成了一个描述市民日常行为的城市模型。此外，该数字阴影还针对马拉加市进行了实例化，我们有超过30万起投诉数据。此实例化结果通过投诉数据及其他地理和社会经济信息进行了校准。据我们所知，这是首个使用大量真实犯罪报告数据并精确反映城市环境的大规模城市中的数字阴影实例。校准后的模型在预测警务广泛使用的指标方面表现出色，表明我们的模拟犯罪生成与历史数据中犯罪模式相符。我们的数字阴影平台可以作为一个有趣的工具，用于每日模拟和预测城市环境中的犯罪行为，从而成为政策制定者、犯罪学家、社会学家、LEA等人士研究和预防城市犯罪的有用工具。 

---
# NSA: Neuro-symbolic ARC Challenge 

**Title (ZH)**: NSA：神经符号ARC挑战 

**Authors**: Paweł Batorski, Jannik Brinkmann, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2501.04424)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) evaluates general reasoning capabilities that are difficult for both machine learning models and combinatorial search methods. We propose a neuro-symbolic approach that combines a transformer for proposal generation with combinatorial search using a domain-specific language. The transformer narrows the search space by proposing promising search directions, which allows the combinatorial search to find the actual solution in short time. We pre-train the trainsformer with synthetically generated data. During test-time we generate additional task-specific training tasks and fine-tune our model. Our results surpass comparable state of the art on the ARC evaluation set by 27% and compare favourably on the ARC train set. We make our code and dataset publicly available at this https URL. 

**Abstract (ZH)**: 抽象和推理语料库（ARC）评估了机器学习模型和组合搜索方法都难以实现的一般推理能力。我们提出了一种结合变分器和组合搜索的神经-符号方法，利用特定领域的语言进行组合搜索。变分器通过提出有前景的搜索方向来缩小搜索空间，这使得组合搜索能够在较短的时间内找到实际解。我们使用合成生成的数据对变分器进行预训练。在测试时，我们生成额外的任务特定训练任务并对模型进行微调。我们的结果在ARC评估集上超过了可比的最新技术水平27%，并在ARC训练集上表现良好。我们已在以下链接公开发布我们的代码和数据集：[这个 https URL](https://this-https-url.com)。 

---
# User Simulation in the Era of Generative AI: User Modeling, Synthetic Data Generation, and System Evaluation 

**Title (ZH)**: 生成式AI时代的用户模拟：用户建模、合成数据生成与系统评估 

**Authors**: Krisztian Balog, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2501.04410)  

**Abstract**: User simulation is an emerging interdisciplinary topic with multiple critical applications in the era of Generative AI. It involves creating an intelligent agent that mimics the actions of a human user interacting with an AI system, enabling researchers to model and analyze user behaviour, generate synthetic data for training, and evaluate interactive AI systems in a controlled and reproducible manner. User simulation has profound implications for diverse fields and plays a vital role in the pursuit of Artificial General Intelligence. This paper provides an overview of user simulation, highlighting its key applications, connections to various disciplines, and outlining future research directions to advance this increasingly important technology. 

**Abstract (ZH)**: 用户模拟是生成式人工智能时代的一个新兴跨学科课题，具有多种关键应用价值。它涉及创建能够模仿人类用户与人工智能系统交互行为的智能代理，使研究人员能够建模和分析用户行为、生成用于训练的合成数据，并在受控和可重复的环境中评估互动人工智能系统。用户模拟对多个领域具有深远的影响，并在追求通用人工智能方面扮演着至关重要的角色。本文概述了用户模拟的基本概念，突出了其关键应用、与各个学科的联系，并指出了未来研究方向，以推进这一日益重要的技术。 

---
# Implementing Systemic Thinking for Automatic Schema Matching: An Agent-Based Modeling Approach 

**Title (ZH)**: 基于代理建模的系统性思维在自动模式匹配中的应用 

**Authors**: Hicham Assoudi, Hakim Lounis  

**Link**: [PDF](https://arxiv.org/pdf/2501.04136)  

**Abstract**: Several approaches are proposed to deal with the problem of the Automatic Schema Matching (ASM). The challenges and difficulties caused by the complexity and uncertainty characterizing both the process and the outcome of Schema Matching motivated us to investigate how bio-inspired emerging paradigm can help with understanding, managing, and ultimately overcoming those challenges. In this paper, we explain how we approached Automatic Schema Matching as a systemic and Complex Adaptive System (CAS) and how we modeled it using the approach of Agent-Based Modeling and Simulation (ABMS). This effort gives birth to a tool (prototype) for schema matching called Reflex-SMAS. A set of experiments demonstrates the viability of our approach on two main aspects: (i) effectiveness (increasing the quality of the found matchings) and (ii) efficiency (reducing the effort required for this efficiency). Our approach represents a significant paradigm-shift, in the field of Automatic Schema Matching. 

**Abstract (ZH)**: 为了应对自动模式匹配（Automatic Schema Matching, ASM）问题，提出了几种方法。由于模式匹配过程及其结果所表现出的复杂性和不确定性带来的挑战和困难，我们探讨了如何利用生物启发的新兴范式来帮助理解、管理和最终克服这些挑战。在本文中，我们解释了我们将自动模式匹配视为系统性且复杂的自适应系统（Complex Adaptive System, CAS），并如何通过基于代理的建模和仿真（Agent-Based Modeling and Simulation, ABMS）方法对其进行建模。这一努力催生了一种名为Reflex-SMAS的模式匹配工具（原型）。一系列实验展示了我们在两个主要方面的可行性：（i）有效性（提高找到匹配的质量）和（ii）效率（减少所需的努力）。我们的方法代表了自动模式匹配领域的一项重要范式转变。 

---
# Planarian Neural Networks: Evolutionary Patterns from Basic Bilateria Shaping Modern Artificial Neural Network Architectures 

**Title (ZH)**: planarian 疣吻ripplingmania（一种水生扁平worm-like无脊椎扁虫），在这里可能是指水螅虫或类似的结构简单但具有再生能力的生物。为了更准确地翻译，可以将其视为水螅虫的类比形式，根据上下文推断最合适的翻译。

以下是符合学术规范的翻译：

水螅虫神经网络：从基础两侧动物演化的模式塑造现代人工神经网络架构

或更具体的翻译（考虑到 planarian 通常指扁形动物门的水螅虫）：

水螅虫神经网络：基础两侧动物的演化模式对现代人工神经网络架构的影响

这样翻译既保持了原意，又符合学术写作的规范和风格。 

**Authors**: Ziyuan Huang, Mark Newman, Maria Vaida, Srikar Bellur, Roozbeh Sadeghian, Andrew Siu, Hui Wang, Kevin Huggins  

**Link**: [PDF](https://arxiv.org/pdf/2501.04700)  

**Abstract**: This study examined the viability of enhancing the prediction accuracy of artificial neural networks (ANNs) in image classification tasks by developing ANNs with evolution patterns similar to those of biological neural networks. ResNet is a widely used family of neural networks with both deep and wide variants; therefore, it was selected as the base model for our investigation. The aim of this study is to improve the image classification performance of ANNs via a novel approach inspired by the biological nervous system architecture of planarians, which comprises a brain and two nerve cords. We believe that the unique neural architecture of planarians offers valuable insights into the performance enhancement of ANNs. The proposed planarian neural architecture-based neural network was evaluated on the CIFAR-10 and CIFAR-100 datasets. Our results indicate that the proposed method exhibits higher prediction accuracy than the baseline neural network models in image classification tasks. These findings demonstrate the significant potential of biologically inspired neural network architectures in improving the performance of ANNs in a wide range of applications. 

**Abstract (ZH)**: 本研究旨在通过开发具有类似于生物神经网络进化模式的人工神经网络（ANNs），探讨提高图像分类任务中ANN预测准确性的影响。ResNet是一类广泛使用的神经网络，具有深广度变体；因此，它被选作我们研究的基础模型。本研究的目标是通过一种受扁形动物脑神经系统结构灵感启发的新型方法，提高ANN的图像分类性能。我们认为，扁形动物独特的神经结构为提升ANN性能提供了宝贵见解。基于扁形动物神经结构的神经网络被分别在CIFAR-10和CIFAR-100数据集上进行评估。研究结果表明，与基线神经网络模型相比，所提出的方法在图像分类任务中表现出更高的预测准确性。这些发现证明了受生物启发的神经网络架构在改善ANN性能方面具有广泛的应用潜力。 

---
# Grokking at the Edge of Numerical Stability 

**Title (ZH)**: 在数值稳定性边缘的顿悟 

**Authors**: Lucas Prieto, Melih Barsbey, Pedro A.M. Mediano, Tolga Birdal  

**Link**: [PDF](https://arxiv.org/pdf/2501.04697)  

**Abstract**: Grokking, the sudden generalization that occurs after prolonged overfitting, is a surprising phenomenon challenging our understanding of deep learning. Although significant progress has been made in understanding grokking, the reasons behind the delayed generalization and its dependence on regularization remain unclear. In this work, we argue that without regularization, grokking tasks push models to the edge of numerical stability, introducing floating point errors in the Softmax function, which we refer to as Softmax Collapse (SC). We demonstrate that SC prevents grokking and that mitigating SC enables grokking without regularization. Investigating the root cause of SC, we find that beyond the point of overfitting, the gradients strongly align with what we call the naïve loss minimization (NLM) direction. This component of the gradient does not alter the model's predictions but decreases the loss by scaling the logits, typically by scaling the weights along their current direction. We show that this scaling of the logits explains the delay in generalization characteristic of grokking and eventually leads to SC, halting further learning. To validate our hypotheses, we introduce two key contributions that address the challenges in grokking tasks: StableMax, a new activation function that prevents SC and enables grokking without regularization, and $\perp$Grad, a training algorithm that promotes quick generalization in grokking tasks by preventing NLM altogether. These contributions provide new insights into grokking, elucidating its delayed generalization, reliance on regularization, and the effectiveness of existing grokking-inducing methods. Code for this paper is available at this https URL. 

**Abstract (ZH)**: 掌握（Grokking），一种在长期过拟合之后突然发生的泛化现象，是对深度学习理解的一个令人惊讶的现象。尽管对掌握的理解取得了显著进展，但延迟泛化的原因及其与正则化之间的依赖关系依然不是很清楚。在本文中，我们主张在没有正则化的情况下，掌握任务会将模型推向数值稳定性边缘，导致Softmax函数中的浮点误差，我们称之为Softmax崩溃（Softmax Collapse，SC）。我们展示了SC如何阻止掌握发生，并且通过缓解SC可以在没有正则化的情况下实现掌握。为了找出SC的根本原因，我们发现超过过拟合点之后，梯度强烈地指向我们称为朴素损失最小化（Naive Loss Minimization, NLM）的方向。该梯度的这一部分并不会改变模型的预测，而是通过缩放逻辑值（通常通过沿当前权重方向缩放权重）来降低损失。我们证明了逻辑值的这种缩放解释了掌握的泛化延迟特性，并最终导致SC，从而停止进一步学习。

为了验证我们的假设，我们介绍了两条关键贡献，以应对掌握任务中的挑战：StableMax，这是一种新的激活函数，能够防止SC并允许在没有正则化的情况下实现掌握；$\perp$Grad，这是一种训练算法，通过完全防止NLM，促进掌握任务的快速泛化。这些贡献为理解掌握提供了新的见解，阐明了其延迟泛化的机制，以及现有掌握诱导方法的有效性。本文的相关代码可从以下链接获取：[这里](this https URL)。 

---
# EpiCoder: Encompassing Diversity and Complexity in Code Generation 

**Title (ZH)**: EpiCoder：涵盖代码生成中的多样性和复杂性 

**Authors**: Yaoxiang Wang, Haoling Li, Xin Zhang, Jie Wu, Xiao Liu, Wenxiang Hu, Zhongxin Guo, Yangyu Huang, Ying Xin, Yujiu Yang, Jinsong Su, Qi Chen, Scarlett Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.04694)  

**Abstract**: Effective instruction tuning is indispensable for optimizing code LLMs, aligning model behavior with user expectations and enhancing model performance in real-world applications. However, most existing methods focus on code snippets, which are limited to specific functionalities and rigid structures, restricting the complexity and diversity of the synthesized data. To address these limitations, we introduce a novel feature tree-based synthesis framework inspired by Abstract Syntax Trees (AST). Unlike AST, which captures syntactic structure of code, our framework models semantic relationships between code elements, enabling the generation of more nuanced and diverse data. The feature tree is constructed from raw data and refined iteratively to increase the quantity and diversity of the extracted features. This process enables the identification of more complex patterns and relationships within the code. By sampling subtrees with controlled depth and breadth, our framework allows precise adjustments to the complexity of the generated code, supporting a wide range of tasks from simple function-level operations to intricate multi-file scenarios. We fine-tuned widely-used base models to create the EpiCoder series, achieving state-of-the-art performance at both the function and file levels across multiple benchmarks. Notably, empirical evidence indicates that our approach shows significant potential in synthesizing highly complex repository-level code data. Further analysis elucidates the merits of this approach by rigorously assessing data complexity and diversity through software engineering principles and LLM-as-a-judge method. 

**Abstract (ZH)**: 有效的指令调优对于优化代码LLM（大型语言模型）至关重要，它能够使模型行为与用户期望一致，并增强模型在实际应用中的性能。然而，现有的大多数方法集中在代码片段上，这些片段局限于特定功能和固定结构，限制了生成数据的复杂性和多样性。为了解决这些局限性，我们提出了一种基于特征树的新合成框架，该框架受到抽象语法树（AST）的启发。与仅捕获代码的语法结构的AST不同，我们的框架建模代码元素之间的语义关系，从而能够生成更为细致和多样化的数据。特征树从原始数据中构建，并在迭代中改进，以增加提取特征的数量和多样性。这一过程使我们能够识别代码中更复杂的模式和关系。通过具有可控深度和广度的子树采样，我们的框架可以在生成代码的复杂性上实现精确调整，从而支持从简单的函数级操作到复杂的多文件场景的广泛任务。我们对广泛使用的基模型进行了微调，创建了EpiCoder系列，在多个基准测试中均在功能和文件级别实现了最先进的性能。值得注意的是，实验证据表明，我们的方法在合成高度复杂的仓库级代码数据方面显示出巨大的潜力。进一步的分析通过对软件工程原则和LLM作为裁判的方法进行严格的数据复杂性和多样性的评估，阐明了这种方法的优势。 

---
# Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding 

**Title (ZH)**: 超越视觉：通过语言锚定细调具备异构传感器的通用机器人策略 

**Authors**: Joshua Jones, Oier Mees, Carmelo Sferrazza, Kyle Stachowicz, Pieter Abbeel, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2501.04693)  

**Abstract**: Interacting with the world is a multi-sensory experience: achieving effective general-purpose interaction requires making use of all available modalities -- including vision, touch, and audio -- to fill in gaps from partial observation. For example, when vision is occluded reaching into a bag, a robot should rely on its senses of touch and sound. However, state-of-the-art generalist robot policies are typically trained on large datasets to predict robot actions solely from visual and proprioceptive observations. In this work, we propose FuSe, a novel approach that enables finetuning visuomotor generalist policies on heterogeneous sensor modalities for which large datasets are not readily available by leveraging natural language as a common cross-modal grounding. We combine a multimodal contrastive loss with a sensory-grounded language generation loss to encode high-level semantics. In the context of robot manipulation, we show that FuSe enables performing challenging tasks that require reasoning jointly over modalities such as vision, touch, and sound in a zero-shot setting, such as multimodal prompting, compositional cross-modal prompting, and descriptions of objects it interacts with. We show that the same recipe is applicable to widely different generalist policies, including both diffusion-based generalist policies and large vision-language-action (VLA) models. Extensive experiments in the real world show that FuSeis able to increase success rates by over 20% compared to all considered baselines. 

**Abstract (ZH)**: 与世界交互是一种多感官体验：为了实现有效的通用交互，需要充分利用所有可用的模态，包括视觉、触觉和音频，以填补部分观察带来的空白。例如，在视线受阻的情况下从袋子里取东西时，机器人应当依赖触觉和听觉。然而，最先进的通用机器人策略通常只在大型数据集上进行训练，从视觉和本体感觉观察中预测机器人的行为。在本研究中，我们提出了FuSe，一种新颖的方法，通过利用自然语言作为跨模态的共同基础，使视觉运动的通用策略能够针对无法获得大量数据的异构传感器模态进行微调。我们结合了多模态对比损失和基于感官的语义生成损失，以编码高层次的语义。在机器人操作的背景下，我们展示了FuSe能够在零样本设置中执行需要联合推理多种模态（如视觉、触觉和音频）的挑战性任务，包括多模态提示、组成跨模态提示以及描述其交互对象的描述。我们表明，相同的配方适用于广泛不同的通用策略，包括基于扩散的通用策略和大型视觉-语言-动作（VLA）模型。在实际世界中的大量实验中，我们发现FuSe相较于所有考虑的基线方法，能够将成功率达到20%以上的提高。 

---
# URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics 

**Title (ZH)**: URSA：理解与验证多模态数学中的链式推理 

**Authors**: Ruilin Luo, Zhuofan Zheng, Yifan Wang, Yiyao Yu, Xinzhe Ni, Zicheng Lin, Jin Zeng, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.04686)  

**Abstract**: Chain-of-thought (CoT) reasoning has been widely applied in the mathematical reasoning of Large Language Models (LLMs). Recently, the introduction of derivative process supervision on CoT trajectories has sparked discussions on enhancing scaling capabilities during test time, thereby boosting the potential of these models. However, in multimodal mathematical reasoning, the scarcity of high-quality CoT training data has hindered existing models from achieving high-precision CoT reasoning and has limited the realization of reasoning potential during test time. In this work, we propose a three-module synthesis strategy that integrates CoT distillation, trajectory-format rewriting, and format unification. It results in a high-quality CoT reasoning instruction fine-tuning dataset in multimodal mathematics, MMathCoT-1M. We comprehensively validate the state-of-the-art (SOTA) performance of the trained URSA-7B model on multiple multimodal mathematical benchmarks. For test-time scaling, we introduce a data synthesis strategy that automatically generates process annotation datasets, known as DualMath-1.1M, focusing on both interpretation and logic. By further training URSA-7B on DualMath-1.1M, we transition from CoT reasoning capabilities to robust supervision abilities. The trained URSA-RM-7B acts as a verifier, effectively enhancing the performance of URSA-7B at test time. URSA-RM-7B also demonstrates excellent out-of-distribution (OOD) verifying capabilities, showcasing its generalization. Model weights, training data and code will be open-sourced. 

**Abstract (ZH)**: 链式思维（CoT）推理已在大规模语言模型（LLMs）的数学推理中广泛应用。最近，在CoT轨迹中引入微分过程监督引发了关于测试时增强扩展能力的讨论，从而提升了这些模型的潜力。然而，在多模态数学推理中，高质量CoT训练数据的稀缺性阻碍了现有模型实现高精度CoT推理，并且限制了其在测试时推理潜力的发挥。在本项工作中，我们提出了一种三模块综合策略，该策略结合了CoT蒸馏、轨迹格式重写和格式统一。这生成了一个高质量的多模态数学CoT推理指令微调数据集——MMathCoT-1M。我们全面验证了训练后的URSA-7B模型在多个多模态数学基准测试中的最先进（SOTA）性能。为了实现测试时的扩展，我们引入了一种数据合成策略，该策略可以自动生成过程注释数据集，称之为DualMath-1.1M，关注解释和逻辑的双重结合。通过进一步在DualMath-1.1M上训练URSA-7B，我们从CoT推理能力转向了稳健的监督能力。训练后的URSA-RM-7B作为验证器，有效提升了URSA-7B在测试时的表现。URSA-RM-7B还展示了出色的分布外（OOD）验证能力，展示了其泛化能力。模型权重、训练数据和代码将开源。 

---
# Enhancing Financial VQA in Vision Language Models using Intermediate Structured Representations 

**Title (ZH)**: 使用中间结构化表示增强视觉语言模型中的金融VQA 

**Authors**: Archita Srivastava, Abhas Kumar, Rajesh Kumar, Prabhakar Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2501.04675)  

**Abstract**: Chart interpretation is crucial for visual data analysis, but accurately extracting information from charts poses significant challenges for automated models. This study investigates the fine-tuning of DEPLOT, a modality conversion module that translates the image of a plot or chart to a linearized table, on a custom dataset of 50,000 bar charts. The dataset comprises simple, stacked, and grouped bar charts, targeting the unique structural features of these visualizations. The finetuned DEPLOT model is evaluated against its base version using a test set of 1,000 images and two metrics: Relative Mapping Similarity (RMS), which measures categorical mapping accuracy, and Relative Number Set Similarity (RNSS), which evaluates numerical interpretation accuracy. To further explore the reasoning capabilities of large language models (LLMs), we curate an additional set of 100 bar chart images paired with question answer sets. Our findings demonstrate that providing a structured intermediate table alongside the image significantly enhances LLM reasoning performance compared to direct image queries. 

**Abstract (ZH)**: 图表解读对于视觉数据分析至关重要，但准确从图表中提取信息对自动化模型构成了重大挑战。本研究考察了对 DEPLOT 进行微调的效果，DEPLOT 是一种模态转换模块，可以将图表或图示的图像转换为线性化表格。我们使用一个包含50,000个柱状图的自定义数据集进行微调，该数据集涵盖了简单的、叠加的和分组的柱状图，旨在瞄准这些可视化特有的结构特征。微调后的 DEPLOT 模型使用一个包含1,000张图像的测试集以及两种指标进行评估：相对取项相似度（RMS），用于衡量分类映射准确性；相对数值集合相似度（RNSS），用于评估数值解读准确性。为进一步探索大规模语言模型（LLM）的推理能力，我们还制作了一个包含100张柱状图及其问题和答案的额外数据集。我们的研究结果表明，与直接图像查询相比，提供一个结构化的中间表可以显著增强 LLM 的推理性能。 

---
# DRIVINGVQA: Analyzing Visual Chain-of-Thought Reasoning of Vision Language Models in Real-World Scenarios with Driving Theory Tests 

**Title (ZH)**: 车载视觉问答：在驾驶理论测试中分析视觉语言模型在现实场景中的视觉链式思维推理能力 

**Authors**: Charles Corbière, Simon Roburin, Syrielle Montariol, Antoine Bosselut, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2501.04671)  

**Abstract**: Large vision-language models (LVLMs) augment language models with visual understanding, enabling multimodal reasoning. However, due to the modality gap between textual and visual data, they often face significant challenges, such as over-reliance on text priors, hallucinations, and limited capacity for complex visual reasoning. Existing benchmarks to evaluate visual reasoning in LVLMs often rely on schematic or synthetic images and on imprecise machine-generated explanations. To bridge the modality gap, we present DrivingVQA, a new benchmark derived from driving theory tests to evaluate visual chain-of-thought reasoning in complex real-world scenarios. It offers 3,931 expert-crafted multiple-choice problems and interleaved explanations grounded with entities relevant to the reasoning process. We leverage this dataset to perform an extensive study of LVLMs' ability to reason about complex visual scenarios. Our experiments reveal that open-source and proprietary LVLMs struggle with visual chain-of-thought reasoning under zero-shot settings. We investigate training strategies that leverage relevant entities to improve visual reasoning. Notably, we observe a performance boost of up to 7\% when reasoning over image tokens of cropped regions tied to these entities. 

**Abstract (ZH)**: 大型跨模态语言模型（LVLMs）通过引入视觉理解来扩展语言模型的功能，从而实现多模态推理。然而，由于文本数据和视觉数据之间的模态差距，它们往往面临诸多挑战，例如过度依赖文本先验、幻觉以及复杂视觉推理能力有限。现有用于评估LVLMs视觉推理能力的基准测试往往依赖于简化的或合成的图像以及不精确的人工生成的解释。为了解决这一模态差距，我们提出了DrivingVQA，这是一种新的基准测试，源自驾驶理论测试，旨在评估复杂现实场景中的视觉链式推理能力。它提供了3,931个专家手工制作的多项选择题和嵌入了相关实体的解释。我们利用这个数据集来探讨LVLMs在复杂视觉场景推理方面的能力。我们的实验表明，在零样本设置下，开源和专有LVLMs在视觉链式推理方面存在困难。我们研究了利用相关实体进行训练的方法以提高视觉推理能力。值得注意的是，当我们对与这些实体相关的裁剪区域图像进行推理时，观察到最大7%的性能提升。 

---
# Assessing Language Comprehension in Large Language Models Using Construction Grammar 

**Title (ZH)**: 使用构建语法评估大规模语言模型的语言理解能力 

**Authors**: Wesley Scivetti, Melissa Torgbi, Austin Blodgett, Mollie Shichman, Taylor Hudson, Claire Bonial, Harish Tayyar Madabushi  

**Link**: [PDF](https://arxiv.org/pdf/2501.04661)  

**Abstract**: Large Language Models, despite their significant capabilities, are known to fail in surprising and unpredictable ways. Evaluating their true `understanding' of language is particularly challenging due to the extensive web-scale data they are trained on. Therefore, we construct an evaluation to systematically assess natural language understanding (NLU) in LLMs by leveraging Construction Grammar (CxG), which provides insights into the meaning captured by linguistic elements known as constructions (Cxns). CxG is well-suited for this purpose because provides a theoretical basis to construct targeted evaluation sets. These datasets are carefully constructed to include examples which are unlikely to appear in pre-training data, yet intuitive and easy for humans to understand, enabling a more targeted and reliable assessment. Our experiments focus on downstream natural language inference and reasoning tasks by comparing LLMs' understanding of the underlying meanings communicated through 8 unique Cxns with that of humans. The results show that while LLMs demonstrate some knowledge of constructional information, even the latest models including GPT-o1 struggle with abstract meanings conveyed by these Cxns, as demonstrated in cases where test sentences are dissimilar to their pre-training data. We argue that such cases provide a more accurate test of true language understanding, highlighting key limitations in LLMs' semantic capabilities. We make our novel dataset and associated experimental data including prompts and model responses publicly available. 

**Abstract (ZH)**: 尽管大规模语言模型拥有显著的能力，但它们在意外和不可预测的方式中往往会失败。由于这些模型是基于大规模网络数据训练的，因此评估它们真正理解语言的能力尤为具有挑战性。因此，我们构建了一种评估方法，利用构式语法（CxG）系统性地评估大规模语言模型（LLMs）的自然语言理解（NLU），CxG为理解语言元素（构式）所捕捉到的意义提供了见解。CxG非常适合这一目的，因为它为构建针对性的评估集提供了理论基础。这些数据集精心构建，包含了在预训练数据中不太可能出现但令人直观且易于人类理解的例子，从而能进行更精准和可靠的评估。我们的实验集中在下游自然语言推理和推理性任务上，通过比较LLMs理解和人类对通过8种独特构式（Cxns）传达的底层意义的理解，来考察模型的理解能力。结果显示，虽然LLMs在一些构造信息方面展示了知识，但即使是最新的模型（如GPT-o1）仍难以理解这些Cxns传达的抽象意义，特别是在测试句子与预训练数据显著不同时更为明显。我们认为，这些案例为评估真正的语言理解提供了更为精准的测试，突显了LLMs在语义能力方面的关键限制。我们公开发布了我们的新颖数据集及相关实验数据，包括提示和模型响应等。 

---
# Knowledge Retrieval Based on Generative AI 

**Title (ZH)**: 基于生成式AI的知识检索 

**Authors**: Te-Lun Yang, Jyi-Shane Liu, Yuen-Hsien Tseng, Jyh-Shing Roger Jang  

**Link**: [PDF](https://arxiv.org/pdf/2501.04635)  

**Abstract**: This study develops a question-answering system based on Retrieval-Augmented Generation (RAG) using Chinese Wikipedia and Lawbank as retrieval sources. Using TTQA and TMMLU+ as evaluation datasets, the system employs BGE-M3 for dense vector retrieval to obtain highly relevant search results and BGE-reranker to reorder these results based on query relevance. The most pertinent retrieval outcomes serve as reference knowledge for a Large Language Model (LLM), enhancing its ability to answer questions and establishing a knowledge retrieval system grounded in generative AI.
The system's effectiveness is assessed through a two-stage evaluation: automatic and assisted performance evaluations. The automatic evaluation calculates accuracy by comparing the model's auto-generated labels with ground truth answers, measuring performance under standardized conditions without human intervention. The assisted performance evaluation involves 20 finance-related multiple-choice questions answered by 20 participants without financial backgrounds. Initially, participants answer independently. Later, they receive system-generated reference information to assist in answering, examining whether the system improves accuracy when assistance is provided.
The main contributions of this research are: (1) Enhanced LLM Capability: By integrating BGE-M3 and BGE-reranker, the system retrieves and reorders highly relevant results, reduces hallucinations, and dynamically accesses authorized or public knowledge sources. (2) Improved Data Privacy: A customized RAG architecture enables local operation of the LLM, eliminating the need to send private data to external servers. This approach enhances data security, reduces reliance on commercial services, lowers operational costs, and mitigates privacy risks. 

**Abstract (ZH)**: 本研究基于检索增强生成（RAG）技术，开发了一个以中文维基百科和Lawbank为检索源的问题解答系统。该系统使用TTQA和TMMLU+作为评估数据集，并采用BGE-M3进行密集向量检索，获取高相关的搜索结果，使用BGE-reranker根据查询相关性对这些结果进行重新排序。最相关检索结果作为大型语言模型（LLM）的参考知识，增强其回答问题的能力，并建立基于生成型AI的知识检索系统。

该系统的有效性通过两个阶段的评估进行测试：自动和辅助性能评估。自动评估通过比较模型自动生成的标签与真实答案之间的准确性，评估系统的性能，并在没有人为干预的情况下标准化条件。辅助性能评估涉及20个金融相关的多项选择问题，由20名无金融背景的参与者独立作答。随后，参与者接收系统生成的参考信息来辅助作答，检查在提供帮助的情况下系统的准确性是否有所提升。

本研究的主要贡献包括：（1）增强的LLM能力：通过整合BGE-M3和BGE-reranker，系统能够检索和重新排序高相关的结果，减少虚构信息的产生，并动态访问授权或公共知识源。（2）改进的数据隐私保护：定制化的RAG架构允许本地运行LLM，无需将私人数据发送到外部服务器。这种做法增强了数据安全性，减少了对商业服务的依赖，降低了运营成本，并减少了隐私风险。 

---
# Federated-Continual Dynamic Segmentation of Histopathology guided by Barlow Continuity 

**Title (ZH)**: 基于Barlow Continuity引导的联邦持续动态病理分割 

**Authors**: Niklas Babendererde, Haozhe Zhu, Moritz Fuchs, Jonathan Stieber, Anirban Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2501.04588)  

**Abstract**: Federated- and Continual Learning have been established as approaches to enable privacy-aware learning on continuously changing data, as required for deploying AI systems in histopathology images. However, data shifts can occur in a dynamic world, spatially between institutions and temporally, due to changing data over time. This leads to two issues: Client Drift, where the central model degrades from aggregating data from clients trained on shifted data, and Catastrophic Forgetting, from temporal shifts such as changes in patient populations. Both tend to degrade the model's performance of previously seen data or spatially distributed training. Despite both problems arising from the same underlying problem of data shifts, existing research addresses them only individually. In this work, we introduce a method that can jointly alleviate Client Drift and Catastrophic Forgetting by using our proposed Dynamic Barlow Continuity that evaluates client updates on a public reference dataset and uses this to guide the training process to a spatially and temporally shift-invariant model. We evaluate our approach on the histopathology datasets BCSS and Semicol and prove our method to be highly effective by jointly improving the dice score as much as from 15.8% to 71.6% in Client Drift and from 42.5% to 62.8% in Catastrophic Forgetting. This enables Dynamic Learning by establishing spatio-temporal shift-invariance. 

**Abstract (ZH)**: 联邦学习和连续学习已被确立为在历史病理图像等不断变化数据中实现隐私意识学习的方法，这符合在AI系统中部署的需求。然而，在动态世界中，数据偏移现象可能在地域上（不同机构之间）和时间上（随时间变化的数据）发生。这导致了两个问题：客户漂移，即中央模型因聚合训练于偏移数据的客户端数据而退化；以及灾难性遗忘，因时间偏移如患者人群变化而引起。这两种情况都会影响模型对之前数据的性能或在地域上分布式训练的效果。尽管这两种问题都源于相同的基础问题——数据偏移，但现有研究仅分别解决了这些问题。在本工作中，我们提出了一种方法，该方法通过使用我们提出的动态Barlow Continuity，这种技术能够评估客户端更新并基于此指导训练过程，从而共同缓解客户漂移和灾难性遗忘问题。动态Barlow Continuity能够在公共参考数据集上评估客户端的更新，并通过这种方式引导训练过程，构建出在空间和时间上具有偏移不变性的模型。我们对BCSS和Semicol等病理学数据集进行了评估，并通过同时提高Dice分数来证明我们方法的有效性，具体而言，在客户漂移问题上从15.8%提高到71.6%，在灾难性遗忘问题上从42.5%提高到62.8%。这使得动态学习成为可能，从而实现时空偏移不变性。 

---
# A 65 nm Bayesian Neural Network Accelerator with 360 fJ/Sample In-Word GRNG for AI Uncertainty Estimation 

**Title (ZH)**: 一种采用360 fJ/样本内置字内高斯随机数发生器（In-Word GRNG）的65纳米贝叶斯神经网络加速器，用于AI不确定性估计 

**Authors**: Zephan M. Enciso, Boyang Cheng, Likai Pei, Jianbo Liu, Steven Davis, Ningyuan Cao, Michael Niemier  

**Link**: [PDF](https://arxiv.org/pdf/2501.04577)  

**Abstract**: Uncertainty estimation is an indispensable capability for AI-enabled, safety-critical applications, e.g. autonomous vehicles or medical diagnosis. Bayesian neural networks (BNNs) use Bayesian statistics to provide both classification predictions and uncertainty estimation, but they suffer from high computational overhead associated with random number generation and repeated sample iterations. Furthermore, BNNs are not immediately amenable to acceleration through compute-in-memory architectures due to the frequent memory writes necessary after each RNG operation. To address these challenges, we present an ASIC that integrates 360 fJ/Sample Gaussian RNG directly into the SRAM memory words. This integration reduces RNG overhead and enables fully-parallel compute-in-memory operations for BNNs. The prototype chip achieves 5.12 GSa/s RNG throughput and 102 GOp/s neural network throughput while occupying 0.45 mm2, bringing AI uncertainty estimation to edge computation. 

**Abstract (ZH)**: 不确定性估计是AI驱动的安全关键应用中不可或缺的能力，例如自动驾驶车辆或医疗诊断。贝叶斯神经网络（BNN）利用贝叶斯统计提供分类预测和不确定性估计，但它们受到随机数生成和重复样本迭代所引起的高计算开销的影响。此外，BNNs 无法立即通过计算在内存架构进行加速，因为每次随机数生成器（RNG）操作后都需要频繁地进行内存写入。为了解决这些问题，我们提出了一种ASIC，将360 fJ/样本的高斯RNG直接集成到SRAM存储单元中。这种集成减少了RNG开销，并使BNNs能够实现全并行计算在内存操作。原型芯片实现了5.12 GSa/s 的RNG吞吐量和102 GOp/s 的神经网络吞吐量，面积仅为0.45 mm²，为边缘计算带来了AI的不确定性估计能力。 

---
# Supervision-free Vision-Language Alignment 

**Title (ZH)**: 无监督视觉-语言对齐 

**Authors**: Giorgio Giannone, Ruoteng Li, Qianli Feng, Evgeny Perevodchikov, Rui Chen, Aleix Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2501.04568)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable potential in integrating visual and linguistic information, but their performance is often constrained by the need for extensive, high-quality image-text training data. Curation of these image-text pairs is both time-consuming and computationally expensive. To address this challenge, we introduce SVP (Supervision-free Visual Projection), a novel framework that enhances vision-language alignment without relying on curated data or preference annotation. SVP leverages self-captioning and a pre-trained grounding model as a feedback mechanism to elicit latent information in VLMs. We evaluate our approach across six key areas: captioning, referring, visual question answering, multitasking, hallucination control, and object recall. Results demonstrate significant improvements, including a 14% average improvement in captioning tasks, up to 12% increase in object recall, and substantial reduction in hallucination rates. Notably, a small VLM using SVP achieves hallucination reductions comparable to a model five times larger, while a VLM with initially poor referring capabilities more than doubles its performance, approaching parity with a model twice its size. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在整合视觉和语言信息方面展现了 remarkable 的潜力，但其性能往往受限于需要大量高质量的图像-文本训练数据。收集这些图像-文本对既耗时又需要高性能计算资源。为了解决这一挑战，我们提出了 SVP（无需监督的视觉投影），这是一种新颖的框架，能够在不依赖于标注数据或偏好标注的情况下增强视觉-语言对齐。SVP 利用自我 caption 生成和预训练的 grounding 模型作为反馈机制，以提取 VLMs 中潜藏的信息。我们从六个关键方面评估了我们的方法：captioning、referencing、视觉问答、多任务、幻觉控制和对象召回。结果表明，该方法在多个方面取得了显著的改进，包括 captioning 任务平均改进了 14%，对象召回增加了 12%，以及在幻觉率方面实现了大幅降低。值得注意的是，使用 SVP 的小型 VLM 在幻觉减少方面的效果与五倍大小的模型相当，而一个初始 referencing 能力较弱的 VLM 则将自己的性能提高了两倍以上，接近与两倍大小模型的性能相当。 

---
# Cyber-Physical Steganography in Robotic Motion Control 

**Title (ZH)**: 基于机器人的运动控制的网络物理隐写术 

**Authors**: Ching-Chun Chang, Yijie Lin, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2501.04541)  

**Abstract**: Steganography, the art of information hiding, has continually evolved across visual, auditory and linguistic domains, adapting to the ceaseless interplay between steganographic concealment and steganalytic revelation. This study seeks to extend the horizons of what constitutes a viable steganographic medium by introducing a steganographic paradigm in robotic motion control. Based on the observation of the robot's inherent sensitivity to changes in its environment, we propose a methodology to encode messages as environmental stimuli influencing the motions of the robotic agent and to decode messages from the resulting motion trajectory. The constraints of maximal robot integrity and minimal motion deviation are established as fundamental principles underlying secrecy. As a proof of concept, we conduct experiments in simulated environments across various manipulation tasks, incorporating robotic embodiments equipped with generalist multimodal policies. 

**Abstract (ZH)**: 信息隐藏的艺术——隐写术——已经不断在视觉、听觉和语言领域中演进，适应着隐写术隐藏与隐写分析揭示之间永无休止的互动。本研究旨在通过在机器人运动控制中引入隐写术范式，扩展可行隐写媒介的边界。基于对机器人对其环境变化高度敏感性的观察，我们提出了一种将信息编码为影响机器人代理运动的环境刺激的方法，并从结果运动轨迹中解码信息。最大机器人完整性和最小运动偏差被确立为保密性基础原则。为了验证概念，我们在多种操作任务的模拟环境中进行了实验，使用具备通用多模态策略的机器人实体。 

---
# Towards a Problem-Oriented Domain Adaptation Framework for Machine Learning 

**Title (ZH)**: 面向问题导向的领域适应机器学习框架研究 

**Authors**: Philipp Spitzer, Dominik Martin, Laurin Eichberger, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2501.04528)  

**Abstract**: Domain adaptation is a sub-field of machine learning that involves transferring knowledge from a source domain to perform the same task in the target domain. It is a typical challenge in machine learning that arises, e.g., when data is obtained from various sources or when using a data basis that changes over time. Recent advances in the field offer promising methods, but it is still challenging for researchers and practitioners to determine if domain adaptation is suitable for a given problem -- and, subsequently, to select the appropriate approach. This article employs design science research to develop a problem-oriented framework for domain adaptation, which is matured in three evaluation episodes. We describe a framework that distinguishes between five domain adaptation scenarios, provides recommendations for addressing each scenario, and offers guidelines for determining if a problem falls into one of these scenarios. During the multiple evaluation episodes, the framework is tested on artificial and real-world datasets and an experimental study involving 100 participants. The evaluation demonstrates that the framework has the explanatory power to capture any domain adaptation problem effectively. In summary, we provide clear guidance for researchers and practitioners who want to employ domain adaptation but lack in-depth knowledge of the possibilities. 

**Abstract (ZH)**: 领域适应是机器学习的一个子领域，涉及从源域转移知识以在目标域执行相同的任务。这是机器学习中一个典型的挑战，例如，当数据来自多种来源或数据基础随着时间变化时会遇到这种情况。领域的最新进展提供了有前景的方法，但研究人员和实践者仍然难以确定领域适应是否适合给定的问题——进而选择适当的方法。本文采用设计科学研究方法开发了一种面向问题的领域适应框架，并通过三个评估阶段成熟完善。我们描述了一种区分五种领域适应场景的框架，为每个场景提供应对策略，并提供指南以确定问题是否属于这些场景之一。在多个评估阶段中，该框架被应用于人工和真实世界的数据集以及涉及100名参与者的实验研究。评估表明，该框架具有解释力，能够有效捕捉任何领域适应问题。总之，我们为那些希望使用领域适应但缺乏深入了解可能性的研究人员和实践者提供了清晰的指导。 

---
# CGP-Tuning: Structure-Aware Soft Prompt Tuning for Code Vulnerability Detection 

**Title (ZH)**: CGP-调优：面向结构的软提示调优在代码漏洞检测中的应用 

**Authors**: Ruijun Feng, Hammond Pearce, Pietro Liguori, Yulei Sui  

**Link**: [PDF](https://arxiv.org/pdf/2501.04510)  

**Abstract**: Large language models (LLMs) have been proposed as powerful tools for detecting software vulnerabilities, where task-specific fine-tuning is typically employed to provide vulnerability-specific knowledge to the LLMs for this purpose. However, traditional full-parameter fine-tuning is inefficient for modern, complex LLMs, which contain billions of parameters.
Soft prompt tuning has been suggested as a more efficient alternative for fine-tuning LLMs in general cases. However, pure soft prompt tuning treats source code as plain text, losing structural information inherent in source code. Meanwhile, graph-enhanced soft prompt tuning methods, which aim to address this issue, are unable to preserve the rich semantic information within code graphs, as they are primarily designed for general graph-related tasks and focus more on adjacency information. They also fail to ensure computational efficiency while accounting for graph-text interactions.
This paper, therefore, introduces a new code graph-enhanced, structure-aware soft prompt tuning method for vulnerability detection, referred to as CGP-Tuning. It employs innovative type-aware embeddings to capture the rich semantic information within code graphs, along with a novel and efficient cross-modal alignment module that achieves linear computational cost while incorporating graph-text interactions. The proposed CGP-Tuning is evaluated on the latest DiverseVul dataset and the most recent open-source code LLMs, CodeLlama and CodeGemma. Experimental results demonstrate that CGP-Tuning outperforms the best state-of-the-art method by an average of 3.5 percentage points in accuracy, without compromising its vulnerability detection capabilities for long source code. 

**Abstract (ZH)**: 大型语言模型（LLMs）已被提议作为检测软件漏洞的强大工具，通常通过特定任务的微调将漏洞相关的知识注入LLMs。然而，传统的全参数微调对现代复杂的LLMs（含有数十亿参数）效率低下。
软提示微调已被建议作为一种更有效的替代方法，适用于一般情况下的LLMs微调。然而，单纯的软提示微调将源代码视为普通文本，丢失了源代码中的结构信息。与此同时，旨在解决该问题的图增强软提示微调方法，主要针对一般的图相关任务设计，侧重于邻接信息，无法保留代码图中的丰富语义信息。此外，这些方法在兼顾图-文本交互的同时，在计算效率上也显得不足。
本文提出了一种新的代码图增强、结构感知的软提示微调方法，命名为CGP-Tuning。该方法采用了创新的类型感知嵌入来捕捉代码图中的丰富语义信息，并采用了一种新颖且高效的跨模态对齐模块，该模块在考虑图-文本交互的同时实现了线性计算成本。所提出的CGP-Tuning方法在最新发布的DiverseVul数据集和最新的开源代码LLM，CodeLlama和CodeGemma上进行了评估。实验结果表明，CGP-Tuning在准确率方面平均高出最佳现有方法3.5个百分点，同时不牺牲其对长源代码的漏洞检测能力。 

---
# The Role of Machine Learning in Congenital Heart Disease Diagnosis: Datasets, Algorithms, and Insights 

**Title (ZH)**: 机器学习在先天性心脏病诊断中的作用：数据集、算法与洞察 

**Authors**: Khalil Khan, Farhan Ullah, Ikram Syed, Irfan Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2501.04493)  

**Abstract**: Congenital heart disease is among the most common fetal abnormalities and birth defects. Despite identifying numerous risk factors influencing its onset, a comprehensive understanding of its genesis and management across diverse populations remains limited. Recent advancements in machine learning have demonstrated the potential for leveraging patient data to enable early congenital heart disease detection. Over the past seven years, researchers have proposed various data-driven and algorithmic solutions to address this challenge. This paper presents a systematic review of congential heart disease recognition using machine learning, conducting a meta-analysis of 432 references from leading journals published between 2018 and 2024. A detailed investigation of 74 scholarly works highlights key factors, including databases, algorithms, applications, and solutions. Additionally, the survey outlines reported datasets used by machine learning experts for congenital heart disease recognition. Using a systematic literature review methodology, this study identifies critical challenges and opportunities in applying machine learning to congenital heart disease. 

**Abstract (ZH)**: 先天性心脏病是胎儿中最常见的异常和出生缺陷之一。尽管已经识别出许多影响其发生的危险因素，但对不同人群中的先天性心脏病的发生机制和管理措施仍缺乏全面的理解。近年来，机器学习的进步展示了通过利用患者数据实现早期先天性心脏病检测的潜力。在过去七年中，研究人员提出了多种数据驱动和算法解决方案来应对这一挑战。本文系统回顾了机器学习在先天性心脏病识别中的应用，通过分析2018年至2024年间发表的432篇顶尖期刊论文，进行了一项元分析。详细调查了74篇学术论文的关键因素，包括数据库、算法、应用和解决方案。此外，该调查还概述了机器学习专家用于先天性心脏病识别的报告数据集。采用系统文献综述的方法，本研究识别了将机器学习应用于先天性心脏病研究中的关键挑战和机遇。 

---
# Integrating remote sensing data assimilation, deep learning and large language model for interactive wheat breeding yield prediction 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

"整合遥感数据同化、深度学习和大型语言模型以实现交互式小麦育种产量预测" 

**Authors**: Guofeng Yang, Nanfei Jin, Wenjie Ai, Zhonghua Zheng, Yuhong He, Yong He  

**Link**: [PDF](https://arxiv.org/pdf/2501.04487)  

**Abstract**: Yield is one of the core goals of crop breeding. By predicting the potential yield of different breeding materials, breeders can screen these materials at various growth stages to select the best performing. Based on unmanned aerial vehicle remote sensing technology, high-throughput crop phenotyping data in breeding areas is collected to provide data support for the breeding decisions of breeders. However, the accuracy of current yield predictions still requires improvement, and the usability and user-friendliness of yield forecasting tools remain suboptimal. To address these challenges, this study introduces a hybrid method and tool for crop yield prediction, designed to allow breeders to interactively and accurately predict wheat yield by chatting with a large language model (LLM). First, the newly designed data assimilation algorithm is used to assimilate the leaf area index into the WOFOST model. Then, selected outputs from the assimilation process, along with remote sensing inversion results, are used to drive the time-series temporal fusion transformer model for wheat yield prediction. Finally, based on this hybrid method and leveraging an LLM with retrieval augmented generation technology, we developed an interactive yield prediction Web tool that is user-friendly and supports sustainable data updates. This tool integrates multi-source data to assist breeding decision-making. This study aims to accelerate the identification of high-yield materials in the breeding process, enhance breeding efficiency, and enable more scientific and smart breeding decisions. 

**Abstract (ZH)**: 产量是作物育种的核心目标之一。通过预测不同育种材料的潜在产量，育种者可以在作物生长的各个阶段筛选这些材料，以选择表现最佳的品种。基于无人飞行器遥感技术，收集育种区域的高通量作物表型数据，为育种者的决策提供数据支持。然而，当前的产量预测准确性仍需改进，产量预报工具的适用性和用户友好性也需改进。为解决这些挑战，本研究引入了一种结合方法和工具，旨在通过与大型语言模型（LLM）聊天的方式，使育种者能够交互并准确预测小麦产量。首先，设计的新数据同化算法将叶面积指数同化到WOFOST模型中。然后，同化过程中的选定输出及遥感反演结果用于驱动时间序列时序变换器模型进行小麦产量预测。最后，基于这种方法并利用具有检索增强生成技术的大型语言模型，我们开发了一个用户友好且支持可持续数据更新的互动产量预测Web工具，该工具整合了多源数据以辅助育种决策。本研究旨在加速育种过程中高产材料的识别、提高育种效率，并使更加科学和智能的育种决策成为可能。 

---
# A novel Facial Recognition technique with Focusing on Masked Faces 

**Title (ZH)**: 一种专注于遮罩面部的新型面部识别技术 

**Authors**: Dana A Abdullah, Dana Rasul Hamad, Hakem Beitollahi, Ismail Y Maolood, Abdulhady Abas Abdullah, Aso Khaleel Ameen  

**Link**: [PDF](https://arxiv.org/pdf/2501.04444)  

**Abstract**: Recognizing the same faces with and without masks is important for ensuring consistent identification in security, access control, and public safety. This capability is crucial in scenarios like law enforcement, healthcare, and surveillance, where accurate recognition must be maintained despite facial occlusion. This research focuses on the challenge of recognizing the same faces with and without masks by employing cosine similarity as the primary technique. With the increased use of masks, traditional facial recognition systems face significant accuracy issues, making it crucial to develop methods that can reliably identify individuals in masked conditions. For that reason, this study proposed Masked-Unmasked Face Matching Model (MUFM). This model employs transfer learning using the Visual Geometry Group (VGG16) model to extract significant facial features, which are subsequently classified utilizing the K-Nearest Neighbors (K-NN) algorithm. The cosine similarity metric is employed to compare masked and unmasked faces of the same individuals. This approach represents a novel contribution, as the task of recognizing the same individual with and without a mask using cosine similarity has not been previously addressed. By integrating these advanced methodologies, the research demonstrates effective identification of individuals despite the presence of masks, addressing a significant limitation in traditional systems. Using data is another essential part of this work, by collecting and preparing an image dataset from three different sources especially some of those data are real provided a comprehensive power of this research. The image dataset used were already collected in three different datasets of masked and unmasked for the same faces. 

**Abstract (ZH)**: 识别戴口罩和不戴口罩的相同面部对于确保安全、访问控制和公共安全中的一致性识别至关重要。这种能力在执法、医疗保健和监控等场景中尤为重要，即使面对面部遮挡，也需要保持准确的识别。本研究专注于通过使用余弦相似度作为主要技术手段来解决戴口罩和不戴口罩的相同面部识别难题。随着口罩使用的增加，传统的面部识别系统面临着显著的准确度问题，因此开发能够在戴口罩条件下可靠识别个体的方法变得至关重要。因此，本研究提出了戴口罩-不戴口罩面部匹配模型（MUFM）。该模型通过使用Visual Geometry Group (VGG16) 进行迁移学习以提取关键面部特征，随后利用K-最近邻（K-NN）算法进行分类。余弦相似度度量被用于比较相同个体的戴口罩和不戴口罩面部图像。这种方法是一项创新性贡献，因为在使用余弦相似度识别戴口罩和不戴口罩的相同个体方面，尚未有相关研究。通过整合这些先进的方法，本研究展示了即使在戴口罩的情况下也能有效识别个体，解决了传统系统中的重大限制。数据的使用是本研究的另一重要组成部分，通过从三个不同的数据来源收集和准备图像数据集，特别是其中一些数据源自真实场景，赋予了本研究全面的实证基础。所使用的图像数据集已经从三个不同数据集中收集了相同面部的戴口罩和不戴口罩的图像。 

---
# Effect of Information Technology on Job Creation to Support Economic: Case Studies of Graduates in Universities (2023-2024) of the KRG of Iraq 

**Title (ZH)**: 信息技术对支持经济增长的就业创造影响：伊拉克库尔德地区大学毕业生案例研究（2023-2024） 

**Authors**: Azhi Kh. Bapir, Ismail Y. Maolood, Dana A Abdullah, Aso K. Ameen, Abdulhady Abas Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2501.04438)  

**Abstract**: The aim of this study is to assess the impact of information technology (IT) on university graduates in terms of employment development, which will aid in economic issues. This study uses a descriptive research methodology and a quantitative approach to understand variables. The focus of this study is to ascertain how graduates of Kurdistan regional universities might use IT to secure employment and significantly contribute to the nation's economic revival. The sample size was established by the use of judgmental sampling procedure and consisted of 314 people. The researcher prepared the questionnaire to collect data, and then SPSS statistical software, version 22, and Excel 2010 were used to modify, compile, and tabulate the results. The study's outcome showed that information technology is incredibly inventive, has a promising future, and makes life much easier for everyone. It also proved that a deep academic understanding of information technology and its constituent parts helps graduates of Kurdistan Regional University find suitable careers. More importantly, though, anyone looking for work or a means of support will find great benefit from possessing credentials and understanding of IT. The study's final finding was that information technology has actively advanced the country's economy. Not only is IT helping to boost youth employment, but it is also turning into a worthwhile investment for economic growth. 

**Abstract (ZH)**: 本研究旨在评估信息技术（IT）对大学毕业生就业发展的影响，以助于经济发展问题。本研究采用描述性研究方法和定量分析来理解相关变量。本研究的重点是确定库尔德斯坦区域大学毕业生如何利用信息技术来获取就业机会，并显著促进国家经济复苏。样本大小通过判断抽样程序确定，共包含314人。研究者准备了一份问卷来收集数据，然后使用SPSS统计软件（版本22）和Excel 2010对结果进行了修改、汇总和统计。研究结果表明，信息技术极其创新、充满光明的未来，让每个人的生活变得更加便捷。同时，深入掌握信息技术及其组成部分的知识有助于库尔德斯坦区域大学的毕业生找到合适的职业。更重要的是，任何寻找工作或支持手段的人来说，掌握信息技术的资质和理解能力都将带来极大的益处。研究的最终结论是，信息技术正在积极促进国家经济的发展。不仅信息技术有助于提升青年就业率，而且也成为促进经济增长的一个值得投资的领域。 

---
# Integrating LLMs with ITS: Recent Advances, Potentials, Challenges, and Future Directions 

**Title (ZH)**: 将大型语言模型（LLMs）与智能教学系统（ITS）集成： recent advances, potentials, challenges, and future directions 

**Authors**: Doaa Mahmud, Hadeel Hajmohamed, Shamma Almentheri, Shamma Alqaydi, Lameya Aldhaheri, Ruhul Amin Khalil, Nasir Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2501.04437)  

**Abstract**: Intelligent Transportation Systems (ITS) are crucial for the development and operation of smart cities, addressing key challenges in efficiency, productivity, and environmental sustainability. This paper comprehensively reviews the transformative potential of Large Language Models (LLMs) in optimizing ITS. Initially, we provide an extensive overview of ITS, highlighting its components, operational principles, and overall effectiveness. We then delve into the theoretical background of various LLM techniques, such as GPT, T5, CTRL, and BERT, elucidating their relevance to ITS applications. Following this, we examine the wide-ranging applications of LLMs within ITS, including traffic flow prediction, vehicle detection and classification, autonomous driving, traffic sign recognition, and pedestrian detection. Our analysis reveals how these advanced models can significantly enhance traffic management and safety. Finally, we explore the challenges and limitations LLMs face in ITS, such as data availability, computational constraints, and ethical considerations. We also present several future research directions and potential innovations to address these challenges. This paper aims to guide researchers and practitioners through the complexities and opportunities of integrating LLMs in ITS, offering a roadmap to create more efficient, sustainable, and responsive next-generation transportation systems. 

**Abstract (ZH)**: 智能交通系统（ITS）对于智慧城市的发展和运营至关重要，能够解决效率、生产力和环境可持续性方面的关键挑战。本文全面回顾了大型语言模型（LLMs）在优化ITS方面的变革潜力。首先，我们提供了ITS的广泛概述，突出了其组成部分、运行原理及其整体效果。随后，我们深入探讨了各种LLM技术（如GPT、T5、CTRL和BERT）的理论背景，阐明了它们在ITS应用中的相关性。接着，我们考查了LLMs在ITS中的广泛应用，包括交通流量预测、车辆检测与分类、自主驾驶、交通标志识别及行人检测。我们的分析揭示了这些高级模型如何显著提升交通管理和安全性。最后，我们探讨了ITS中LLMs面临的挑战和限制，例如数据可用性问题、计算约束以及伦理考虑。我们还提出了几种未来研究方向和潜在创新，以应对这些挑战。本文旨在引导研究人员和从业者理解整合LLMs到ITS中的复杂性和机遇，提供一种路线图，以创建更高效、更可持续和更响应式的新一代交通系统。 

---
# Federated Fine-Tuning of LLMs: Framework Comparison and Research Directions 

**Title (ZH)**: 联邦微调大型语言模型：框架比较与研究方向 

**Authors**: Na Yan, Yang Su, Yansha Deng, Robert Schober  

**Link**: [PDF](https://arxiv.org/pdf/2501.04436)  

**Abstract**: Federated learning (FL) provides a privacy-preserving solution for fine-tuning pre-trained large language models (LLMs) using distributed private datasets, enabling task-specific adaptation while preserving data privacy. However, fine-tuning the extensive parameters in LLMs is particularly challenging in resource-constrained federated scenarios due to the significant communication and computational costs. To gain a deeper understanding of how these challenges can be addressed, this article conducts a comparative analysis three advanced federated LLM (FedLLM) frameworks that integrate knowledge distillation (KD) and split learning (SL) to mitigate these issues: 1) FedLLMs, where clients upload model parameters or gradients to enable straightforward and effective fine-tuning; 2) KD-FedLLMs, which leverage KD for efficient knowledge sharing via logits; and 3) Split-FedLLMs, which split the LLMs into two parts, with one part executed on the client and the other one on the server, to balance the computational load. Each framework is evaluated based on key performance metrics, including model accuracy, communication overhead, and client-side computational load, offering insights into their effectiveness for various federated fine-tuning scenarios. Through this analysis, we identify framework-specific optimization opportunities to enhance the efficiency of FedLLMs and discuss broader research directions, highlighting open opportunities to better adapt FedLLMs for real-world applications. A use case is presented to demonstrate the performance comparison of these three frameworks under varying configurations and settings. 

**Abstract (ZH)**: 联邦学习（FL）提供了一种在分布式私有数据集上微调预训练大型语言模型（LLMs）的同时保护数据隐私的隐私保护解决方案，从而实现任务特定的适应。然而，在资源限制的联邦场景中，由于显著的通信和计算成本，微调LLMs的大量参数特别具有挑战性。为了深入了解如何应对这些挑战，本文通过比较分析三种整合知识蒸馏（KD）和分割学习（SL）的先进联邦LLM（FedLLM）框架来进行研究：1）FedLLMs，其中客户端上传模型参数或梯度，以实现简单有效的微调；2）KD-FedLLMs，利用KD进行高效的知识共享，通过logits实现知识传递；3）Split-FedLLMs，将LLMs分为两部分，一部分在客户端执行，另一部分在服务器上执行，以平衡计算负载。每个框架将根据模型准确性、通信开销和客户端计算负载等关键性能指标进行评估，从而提供其在各种联邦微调场景中的有效性见解。通过这一分析，我们识别出针对每个框架的具体优化机会，以增强FedLLMs的效率，并讨论更广泛的研宄方向，强调更好地适应FedLLMs以满足实际应用需求的开放机会。还提供了一个用例，以在不同配置和设置下比较这三种框架的性能。 

---
# Dual-Force: Enhanced Offline Diversity Maximization under Imitation Constraints 

**Title (ZH)**: 双力：在模仿约束下的离线多样性最大化增强方法 

**Authors**: Pavel Kolev, Marin Vlastelica, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2501.04426)  

**Abstract**: While many algorithms for diversity maximization under imitation constraints are online in nature, many applications require offline algorithms without environment interactions. Tackling this problem in the offline setting, however, presents significant challenges that require non-trivial, multi-stage optimization processes with non-stationary rewards. In this work, we present a novel offline algorithm that enhances diversity using an objective based on Van der Waals (VdW) force and successor features, and eliminates the need to learn a previously used skill discriminator. Moreover, by conditioning the value function and policy on a pre-trained Functional Reward Encoding (FRE), our method allows for better handling of non-stationary rewards and provides zero-shot recall of all skills encountered during training, significantly expanding the set of skills learned in prior work. Consequently, our algorithm benefits from receiving a consistently strong diversity signal (VdW), and enjoys more stable and efficient training. We demonstrate the effectiveness of our method in generating diverse skills for two robotic tasks in simulation: locomotion of a quadruped and local navigation with obstacle traversal. 

**Abstract (ZH)**: 尽管许多在模仿约束下最大化多样性的问题采用在线算法，但许多实际应用需要不需要与环境交互的离线算法。然而，在离线设置中解决这个问题带来了显著的挑战，需要使用非平凡的多阶段优化过程来处理非平稳奖励。在本文中，我们提出了一种新型的离线算法，该算法使用基于范德瓦尔斯力（VdW力）的目标函数和嗣特征来增强多样性，并消除了对先前使用的技能鉴别器的依赖。此外，通过将价值函数和策略条件化于预训练的功能奖励编码（FRE），我们的方法能够更好地处理非平稳奖励，并在训练过程中记录所有遇到的技能，从而显著扩展了先前工作所学习的技能集。因此，我们的算法能够接收到一致强烈的多样性信号（VdW力），并享受更稳定和高效的训练过程。我们通过在仿真中生成两种机器人的多样技能，验证了该方法的有效性：四足运动和具有障碍穿越的局部导航任务。 

---
# On Computational Limits and Provably Efficient Criteria of Visual Autoregressive Models: A Fine-Grained Complexity Analysis 

**Title (ZH)**: 关于视觉自回归模型的计算极限及可证明高效标准的精细复杂性分析 

**Authors**: Yekun Ke, Xiaoyu Li, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.04377)  

**Abstract**: Recently, Visual Autoregressive ($\mathsf{VAR}$) Models introduced a groundbreaking advancement in the field of image generation, offering a scalable approach through a coarse-to-fine "next-scale prediction" paradigm. However, the state-of-the-art algorithm of $\mathsf{VAR}$ models in [Tian, Jiang, Yuan, Peng and Wang, NeurIPS 2024] takes $O(n^4)$ time, which is computationally inefficient. In this work, we analyze the computational limits and efficiency criteria of $\mathsf{VAR}$ Models through a fine-grained complexity lens. Our key contribution is identifying the conditions under which $\mathsf{VAR}$ computations can achieve sub-quadratic time complexity. Specifically, we establish a critical threshold for the norm of input matrices used in $\mathsf{VAR}$ attention mechanisms. Above this threshold, assuming the Strong Exponential Time Hypothesis ($\mathsf{SETH}$) from fine-grained complexity theory, a sub-quartic time algorithm for $\mathsf{VAR}$ models is impossible. To substantiate our theoretical findings, we present efficient constructions leveraging low-rank approximations that align with the derived criteria. This work initiates the study of the computational efficiency of the $\mathsf{VAR}$ model from a theoretical perspective. Our technique will shed light on advancing scalable and efficient image generation in $\mathsf{VAR}$ frameworks. 

**Abstract (ZH)**: 近年来，视觉自回归（$\mathsf{VAR}$）模型在图像生成领域取得了突破性进展，通过从粗到细的“下一阶段预测” paradigm 提供了可扩展的方法。然而，Tian等人的 $\mathsf{VAR}$ 模型在 NeurIPS 2024 的最新算法所需时间复杂度为 $O(n^4)$，这在计算效率方面表现不佳。在本文中，我们通过精细粒度的复杂性分析来探讨 $\mathsf{VAR}$ 模型的计算限制和效率标准。我们的主要贡献是确定了 $\mathsf{VAR}$ 计算能够在亚二次时间复杂度下实现的条件。具体而言，我们为 $\mathsf{VAR}$ 注意机制中使用的输入矩阵的范数设定了一个关键阈值。在这一阈值之上，假设细粒度复杂性理论中的强指数时间假设 ($\mathsf{SETH}$) 为真，对于 $\mathsf{VAR}$ 模型而言，不存在低于四次方的时间算法。为了验证我们的理论结果，我们提出了符合所推导标准的有效构造方法，利用低秩近似。本工作从理论角度开始研究 $\mathsf{VAR}$ 模型的计算效率，并提出的技术将为 $\mathsf{VAR}$ 框架中的可扩展和高效图像生成提供借鉴。 

---
# DispFormer: Pretrained Transformer for Flexible Dispersion Curve Inversion from Global Synthesis to Regional Applications 

**Title (ZH)**: DispFormer：用于从全球合成到区域应用的可调节 dispersion 曲线反转的预训练变换器 

**Authors**: Feng Liu, Bao Deng, Rui Su, Lei Bai, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2501.04366)  

**Abstract**: Surface wave dispersion curve inversion is essential for estimating subsurface Shear-wave velocity ($v_s$), yet traditional methods often struggle to balance computational efficiency with inversion accuracy. While deep learning approaches show promise, previous studies typically require large amounts of labeled data and struggle with real-world datasets that have varying period ranges, missing data, and low signal-to-noise ratios. This study proposes DispFormer, a transformer-based neural network for inverting the $v_s$ profile from Rayleigh-wave phase and group dispersion curves. DispFormer processes dispersion data at each period independently, thereby allowing it to handle data of varying lengths without requiring network modifications or alignment between training and testing data. The performance is demonstrated by pre-training it on a global synthetic dataset and testing it on two regional synthetic datasets using zero-shot and few-shot strategies. Results indicate that zero-shot DispFormer, even without any labeled data, produces inversion profiles that match well with the ground truth, providing a deployable initial model generator to assist traditional methods. When labeled data is available, few-shot DispFormer outperforms traditional methods with only a small number of labels. Furthermore, real-world tests indicate that DispFormer effectively handles varying length data, and yields lower data residuals than reference models. These findings demonstrate that DispFormer provides a robust foundation model for dispersion curve inversion and is a promising approach for broader applications. 

**Abstract (ZH)**: 表面波频散曲线反演对于估计地下剪切波速度（$v_s$）至关重要，然而传统的反演方法常常难以在计算效率与反演精度之间取得平衡。虽然深度学习方法展现出潜力，但以往研究通常需要大量的标注数据，并且在面对期程范围变化、缺失数据和信噪比低的现实数据集时表现不佳。本研究提出了一种基于变换器的神经网络DispFormer，用于从勒罗伊波相位和群延迟频散曲线中反演$v_s$剖面。DispFormer能够独立处理每个期程的频散数据，从而不需要对网络进行修改或对齐训练和测试数据的情况下即可处理长度不一的数据。通过在全局合成数据集上进行预训练，并使用零样本和少样本策略测试在两个区域合成数据集上，展示了其性能。结果表明，零样本的DispFormer即使在没有标注数据的情况下也能产生与真实情况吻合的反演剖面，可作为辅助传统方法的部署初始模型。当有标注数据时，少样本的DispFormer仅使用少量标注数据便能超越传统方法。此外，现实测试表明，DispFormer有效处理长度不一的数据，并且与参考模型相比，其数据残差较小。这些发现表明， DispFormer为频散曲线反演提供了稳健的基础模型，并且是广泛应用于更广泛领域的一个有前景的方法。 

---
# TimelineKGQA: A Comprehensive Question-Answer Pair Generator for Temporal Knowledge Graphs 

**Title (ZH)**: TimelineKGQA：面向时间知识图谱的全面问答对生成器 

**Authors**: Qiang Sun, Sirui Li, Du Huynh, Mark Reynolds, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.04343)  

**Abstract**: Question answering over temporal knowledge graphs (TKGs) is crucial for understanding evolving facts and relationships, yet its development is hindered by limited datasets and difficulties in generating custom QA pairs. We propose a novel categorization framework based on timeline-context relationships, along with \textbf{TimelineKGQA}, a universal temporal QA generator applicable to any TKGs. The code is available at: \url{this https URL} as an open source Python package. 

**Abstract (ZH)**: 时间知识图谱（TKG）上的问答对于理解 evolving 的事实和关系至关重要，但由于数据集有限以及自动生成定制问答对的困难，其发展受到了阻碍。我们提出了一种基于时间轴上下文关系的新颖分类框架，并开发了适用于任何 TKG 的通用时间问答生成器 \textbf{TimelineKGQA}。相关代码可在以下开源 Python 包中获取：\url{此链接 URL}。 

---
# RoRA: Efficient Fine-Tuning of LLM with Reliability Optimization for Rank Adaptation 

**Title (ZH)**: RoRA：针对排名适应性的可靠性能优化高效细调大语言模型 

**Authors**: Jun Liu, Zhenglun Kong, Peiyan Dong, Xuan Shen, Pu Zhao, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Xue Lin, Dong Huang, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.04315)  

**Abstract**: Fine-tuning helps large language models (LLM) recover degraded information and enhance task this http URL Low-Rank Adaptation (LoRA) is widely used and effective for fine-tuning, we have observed that its scaling factor can limit or even reduce performance as the rank size increases. To address this issue, we propose RoRA (Rank-adaptive Reliability Optimization), a simple yet effective method for optimizing LoRA's scaling factor. By replacing $\alpha/r$ with $\alpha/\sqrt{r}$, RoRA ensures improved performance as rank size increases. Moreover, RoRA enhances low-rank adaptation in fine-tuning uncompressed models and excels in the more challenging task of accuracy recovery when fine-tuning pruned models. Extensive experiments demonstrate the effectiveness of RoRA in fine-tuning both uncompressed and pruned models. RoRA surpasses the state-of-the-art (SOTA) in average accuracy and robustness on LLaMA-7B/13B, LLaMA2-7B, and LLaMA3-8B, specifically outperforming LoRA and DoRA by 6.5% and 2.9% on LLaMA-7B, respectively. In pruned model fine-tuning, RoRA shows significant advantages; for SHEARED-LLAMA-1.3, a LLaMA-7B with 81.4% pruning, RoRA achieves 5.7% higher average accuracy than LoRA and 3.9% higher than DoRA. 

**Abstract (ZH)**: 细调有助于大型语言模型（LLM）恢复降级信息并增强任务性能。我们观察到，随着秩大小增加，低秩适应（LoRA）的缩放因子可能会限制甚至降低性能。为了解决这一问题，我们提出了RoRA（Rank-adaptive Reliability Optimization），一种简单而有效的方法来优化LoRA的缩放因子。通过将$\alpha/r$替换为$\alpha/\sqrt{r}$，RoRA确保随秩大小增加时性能得到提升。此外，RoRA在细调未压缩模型时增强了低秩适应性，并在细调剪枝模型时取得了更高的准确度恢复效果。广泛的实验结果表明，RoRA在未压缩和剪枝模型的细调中都表现出有效性。在对LLaMA-7B/13B、LLaMA2-7B和LLaMA3-8B的平均准确度和鲁棒性测试中，RoRA分别比LoRA和DoRA高出6.5%和2.9%。在剪枝模型的细调中，RoRA表现出明显优势；对于SHEARED-LLAMA-1.3（去掉了81.4%的LLaMA-7B），RoRA的平均准确度比LoRA高出5.7%，比DoRA高出3.9%。 

---
# H-MBA: Hierarchical MamBa Adaptation for Multi-Modal Video Understanding in Autonomous Driving 

**Title (ZH)**: H-MBA：自主驾驶中多模态视频理解的分层MamBa自适应方法 

**Authors**: Siran Chen, Yuxiao Luo, Yue Ma, Yu Qiao, Yali Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.04302)  

**Abstract**: With the prevalence of Multimodal Large Language Models(MLLMs), autonomous driving has encountered new opportunities and challenges. In particular, multi-modal video understanding is critical to interactively analyze what will happen in the procedure of autonomous driving. However, videos in such a dynamical scene that often contains complex spatial-temporal movements, which restricts the generalization capacity of the existing MLLMs in this field. To bridge the gap, we propose a novel Hierarchical Mamba Adaptation (H-MBA) framework to fit the complicated motion changes in autonomous driving videos. Specifically, our H-MBA consists of two distinct modules, including Context Mamba (C-Mamba) and Query Mamba (Q-Mamba). First, C-Mamba contains various types of structure state space models, which can effectively capture multi-granularity video context for different temporal resolutions. Second, Q-Mamba flexibly transforms the current frame as the learnable query, and attentively selects multi-granularity video context into query. Consequently, it can adaptively integrate all the video contexts of multi-scale temporal resolutions to enhance video understanding. Via a plug-and-play paradigm in MLLMs, our H-MBA shows the remarkable performance on multi-modal video tasks in autonomous driving, e.g., for risk object detection, it outperforms the previous SOTA method with 5.5% mIoU improvement. 

**Abstract (ZH)**: 随着多模态大语言模型（MLLMs）的普及，自动驾驶面临着新的机遇与挑战。尤其在自动驾驶过程中，多模态视频理解对于实时分析即将发生的情况至关重要。然而，这些动态场景中的视频通常包含复杂的时空运动，这限制了现有MLLMs在这方面的泛化能力。为了解决这一问题，我们提出了一种新颖的层次Mamba自适应（H-MBA）框架，以适应自动驾驶视频中的复杂运动变化。具体而言，我们的H-MBA框架包含两个不同的模块，即上下文Mamba（C-Mamba）和查询Mamba（Q-Mamba）。首先，C-Mamba包含多种结构状态空间模型，能够有效捕捉不同时间分辨率下的多尺度视频上下文。其次，Q-Mamba灵活地将当前帧转换为可学习的查询，并注意力选择多尺度视频上下文作为查询。由此，它可以通过自适应地整合不同时间尺度下的所有视频上下文来增强视频理解。通过在MLLMs中的插拔式 Paradigm，我们的 H-MBA 在自动驾驶中的多模态视频任务中展现出显著性能，例如，在风险对象检测中，与之前的最先进的方法相比，取得了5.5%的mIoU改进。 

---
# Circuit Complexity Bounds for Visual Autoregressive Model 

**Title (ZH)**: 视觉自回归模型的电路复杂度界 

**Authors**: Yekun Ke, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.04299)  

**Abstract**: Understanding the expressive ability of a specific model is essential for grasping its capacity limitations. Recently, several studies have established circuit complexity bounds for Transformer architecture. Besides, the Visual AutoRegressive (VAR) model has risen to be a prominent method in the field of image generation, outperforming previous techniques, such as Diffusion Transformers, in generating high-quality images. We investigate the circuit complexity of the VAR model and establish a bound in this study. Our primary result demonstrates that the VAR model is equivalent to a simulation by a uniform $\mathsf{TC}^0$ threshold circuit with hidden dimension $d \leq O(n)$ and $\mathrm{poly}(n)$ precision. This is the first study to rigorously highlight the limitations in the expressive power of VAR models despite their impressive performance. We believe our findings will offer valuable insights into the inherent constraints of these models and guide the development of more efficient and expressive architectures in the future. 

**Abstract (ZH)**: 理解特定模型的表达能力对于把握其容量限制至关重要。最近，一些研究已经为Transformer架构设定了电路复杂度界限。此外，视觉自回归（VAR）模型已成为图像生成领域的突出方法，其生成高质量图像的表现优于之前的扩散Transformer等方法。在这篇文章中，我们研究了VAR模型的电路复杂度并在此建立了相应的界限。我们的主要结果表明，VAR模型等同于一个均匀的阈值电路$\mathsf{TC}^0$模拟，其隐藏维度$d \leq O(n)$且精度为$\mathrm{poly}(n)$。这是首次严格地指出尽管VAR模型表现令人印象深刻，其表达能力仍然存在的局限性。我们相信这些发现将为这些模型的固有约束提供有价值的见解，并指导未来更高效和更具表达能力的架构的发展。 

---
# MAD-UV: The 1st INTERSPEECH Mice Autism Detection via Ultrasound Vocalization Challenge 

**Title (ZH)**: MAD-UV：首届INTERSPECH老鼠自闭症检测超声 vocalization挑战赛

注释：
1. MAD-UV：这个缩写保持不变，因为它是挑战赛名称的一部分。
2. INTERSPECH：这是“INTERSPEECH”的误写，正确的拼写是“INTERSPEECH”，INTERSPEECH是国际语音通信会议的名称，此处应为正确的拼写。
3. mice autism detection via ultrasound vocalization：老鼠自闭症检测通过超声 vocalization。括号内的解释是提供的背景信息，没有翻译成中文，因为它已经清楚地表达了其含义。
4. 挑战赛名称通常保持原文，以便读者可以直接识别出这是哪个特定的竞赛。但为了整体文句通顺，可以做一些必要的调整。 

**Authors**: Zijiang Yang, Meishu Song, Xin Jing, Haojie Zhang, Kun Qian, Bin Hu, Kota Tamada, Toru Takumi, Björn W. Schuller, Yoshiharu Yamamoto  

**Link**: [PDF](https://arxiv.org/pdf/2501.04292)  

**Abstract**: The Mice Autism Detection via Ultrasound Vocalization (MAD-UV) Challenge introduces the first INTERSPEECH challenge focused on detecting autism spectrum disorder (ASD) in mice through their vocalizations. Participants are tasked with developing models to automatically classify mice as either wild-type or ASD models based on recordings with a high sampling rate. Our baseline system employs a simple CNN-based classification using three different spectrogram features. Results demonstrate the feasibility of automated ASD detection, with the considered audible-range features achieving the best performance (UAR of 0.600 for segment-level and 0.625 for subject-level classification). This challenge bridges speech technology and biomedical research, offering opportunities to advance our understanding of ASD models through machine learning approaches. The findings suggest promising directions for vocalization analysis and highlight the potential value of audible and ultrasound vocalizations in ASD detection. 

**Abstract (ZH)**: 《基于超声声学发声的实验鼠自闭症检测挑战（MAD-UV挑战）》引入了INTERSPEECH的第一个挑战，该挑战旨在通过实验鼠的发声检测自闭症谱系障碍（ASD）。参赛者需要开发模型，根据高速率录音自动将实验鼠分类为野生型或ASD模型。我们基线系统采用基于简单CNN的分类方法，使用三种不同的声谱图特征。实验结果表明，自动检测ASD的可能性是可行的，考虑的可在听觉范围内检测的特征表现出最佳性能（在片段级别分类的宏平均准确率为0.600，在个体级别分类的宏平均准确率为0.625）。该挑战将语音技术与生物医学研究结合起来，提供了通过机器学习方法推进对ASD模型理解的机会。研究结果表明，发声分析具有潜在的发展方向，并突出了可听和超声发声在ASD检测中的潜在价值。 

---
# Mapping the Edge of Chaos: Fractal-Like Boundaries in The Trainability of Decoder-Only Transformer Models 

**Title (ZH)**: 探索混沌边缘：解码器唯一大规模变压器模型可训练性的分形边界 

**Authors**: Bahman Torkamandi  

**Link**: [PDF](https://arxiv.org/pdf/2501.04286)  

**Abstract**: In the realm of fractal geometry, intricate structures emerge from simple iterative processes that partition parameter spaces into regions of stability and instability. Likewise, training large language models involves iteratively applying update functions, such as Adam, where even slight hyperparameter adjustments can shift the training process from convergence to divergence. Recent evidence from miniature neural networks suggests that the boundary separating these outcomes displays fractal characteristics [1]. Building on these insights, this study extends them to medium-sized, decoder-only transformer architectures by employing a more consistent convergence measure and examining the learning rate hyperparameter landscape for attention and fully connected layers. The results show that the trainability frontier is not a simple threshold; rather, it forms a self-similar yet seemingly random structure at multiple scales, with statistically consistent and repeating patterns. Within this landscape, a region of stable convergence is surrounded by a complex chaotic border, illustrating the sensitive nature of the underlying training dynamics. 

**Abstract (ZH)**: 在分形几何的领域中，从简单的迭代过程中会涌现复杂的结构，这些过程将参数空间划分为稳定区和不稳定区。类似地，在训练大规模语言模型时，通过迭代应用更新函数（如Adam），即使微小的超参数调整也可能导致训练过程从收敛转变为发散。最近来自小型神经网络的研究表明，分隔这些结果的边界具有分形特性 [1]。在此基础上，本研究将这些发现扩展到了中型的、仅解码器的Transformer架构中，通过使用更一致的收敛度量，并研究注意力层和全连接层的超参数学习率景观。研究结果表明，训练可实现的边界并不是一个简单的阈值；而是形成一种在多个尺度上呈现自相似但又看似随机的结构，具有统计上一致且重复的模式。在这个景观中，一个稳定的收敛区域被一个复杂混沌的边界所包围，展示了训练动力学的敏感性质。 

---
# Enhancing Scene Classification in Cloudy Image Scenarios: A Collaborative Transfer Method with Information Regulation Mechanism using Optical Cloud-Covered and SAR Remote Sensing Images 

**Title (ZH)**: 基于光学云覆盖图像和SAR遥感图像的协作迁移方法及其信息调节机制在云雾场景下场景分类能力的提升 

**Authors**: Yuze Wang, Rong Xiao, Haifeng Li, Mariana Belgiu, Chao Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.04283)  

**Abstract**: In remote sensing scene classification, leveraging the transfer methods with well-trained optical models is an efficient way to overcome label scarcity. However, cloud contamination leads to optical information loss and significant impacts on feature distribution, challenging the reliability and stability of transferred target models. Common solutions include cloud removal for optical data or directly using Synthetic aperture radar (SAR) data in the target domain. However, cloud removal requires substantial auxiliary data for support and pre-training, while directly using SAR disregards the unobstructed portions of optical data. This study presents a scene classification transfer method that synergistically combines multi-modality data, which aims to transfer the source domain model trained on cloudfree optical data to the target domain that includes both cloudy optical and SAR data at low cost. Specifically, the framework incorporates two parts: (1) the collaborative transfer strategy, based on knowledge distillation, enables the efficient prior knowledge transfer across heterogeneous data; (2) the information regulation mechanism (IRM) is proposed to address the modality imbalance issue during transfer. It employs auxiliary models to measure the contribution discrepancy of each modality, and automatically balances the information utilization of modalities during the target model learning process at the sample-level. The transfer experiments were conducted on simulated and real cloud datasets, demonstrating the superior performance of the proposed method compared to other solutions in cloud-covered scenarios. We also verified the importance and limitations of IRM, and further discussed and visualized the modality imbalance problem during the model transfer. Codes are available at this https URL 

**Abstract (ZH)**: 在遥感场景分类中，利用预训练光学模型的迁移方法是一种有效克服标签稀缺的方法。然而，云污染导致了光学信息的丢失，并对特征分布产生了显著影响，这挑战了目标模型的可靠性和稳定性。常见的解决方案包括对光学数据进行云去除或直接在目标域中使用合成孔径雷达（SAR）数据。但是，云去除需要大量的辅助数据支持和预训练，而直接使用SAR数据则忽视了光学数据中的清晰部分。本研究提出了一种结合多模态数据的场景分类迁移方法，旨在以低成本将训练于无云光学数据源域模型迁移到包括有云光学和SAR数据的目标域。具体而言，该框架包括两个部分：（1）基于知识蒸馏的协同迁移策略能够高效地在异构数据间转移先验知识；（2）提出了一种信息调节机制（IRM），以解决迁移过程中模态不平衡的问题。该机制利用辅助模型测量每种模态的贡献差异，并在目标模型学习过程中按样本级自动平衡不同模态的信息利用。我们在模拟和实际的云覆盖数据集上进行了迁移实验，结果显示与现有解决方案相比，所提出的方法在云覆盖场景中的性能更优。我们还验证了IRM的重要性及其限制，并进一步讨论和可视化了模型迁移过程中模态不平衡的问题。代码可在以下链接获取：[此处提供链接] 

---
# Scaling Large Language Model Training on Frontier with Low-Bandwidth Partitioning 

**Title (ZH)**: 在低带宽分区技术下的大规模语言模型训练扩展研究 

**Authors**: Lang Xu, Quentin Anthony, Jacob Hatef, Aamir Shafi, Hari Subramoni, Dhabaleswar K., Panda  

**Link**: [PDF](https://arxiv.org/pdf/2501.04266)  

**Abstract**: Scaling up Large Language Model(LLM) training involves fitting a tremendous amount of training parameters across a limited number of workers. However, methods like ZeRO-3 that drastically reduce GPU memory pressure often incur heavy communication to ensure global synchronization and consistency. Established efforts such as ZeRO++ use secondary partitions to avoid inter-node communications, given that intra-node GPU-GPU transfer generally has more bandwidth and lower latency than inter-node connections. However, as more capable infrastructure like Frontier, equipped with AMD GPUs, emerged with impressive computing capability, there is a need for investigations on the hardware topology and to develop targeted strategies to improve training efficiency. In this work, we propose a collection of communication and optimization strategies for ZeRO++ to reduce communication costs and improve memory utilization. In this paper, we propose a 3-level hierarchical partitioning specifically for the current Top-1 supercomputing cluster, Frontier, which aims at leveraging various bandwidths across layers of communications (GCD-GCD, GPU-GPU, and inter-node) to reduce communication overhead. For a 20B GPT model, we observe a 1.71x increase in TFLOPS per GPU when compared with ZeRO++ up to 384 GCDs and a scaling efficiency of 0.94 for up to 384 GCDs. To the best of our knowledge, our work is also the first effort to efficiently optimize LLM workloads on Frontier AMD GPUs. 

**Abstract (ZH)**: 扩展大规模语言模型（LLM）的训练涉及在有限数量的计算节点上分配大量的训练参数。然而，像ZeRO-3这样的方法虽然极大地减少了GPU内存压力，但往往需要大量的通信来确保全局同步与一致性。现有的解决方案，如ZeRO++，通过使用二级分区来避免节点间通信，考虑到节点内GPU到GPU的传输通常具有更高的带宽和更低的延迟。然而，随着诸如Frontier这样的先进基础设施的出现，配备了AMD GPU的计算能力更为强大的系统，有必要调查硬件拓扑并开发针对性策略以提高训练效率。在本研究中，我们提出了一种适用于当前排名首位的超级计算集群Frontier的三层级分区方法，旨在利用不同层次通信带宽（GCD-GCD、GPU-GPU和节点间）来减少通信开销。对于一个20B的GPT模型，与ZeRO++在最多384个GCD上的配置相比，我们观察到每GPU的TFLOPS提高了1.71倍，并且在最多384个GCD上的扩展效率为0.94。据我们所知，本研究也是第一个在Frontier的AMD GPU上高效优化语言模型负载的工作。 

---
# KN-LIO: Geometric Kinematics and Neural Field Coupled LiDAR-Inertial Odometry 

**Title (ZH)**: KN-LIO: 几何运动学与神经场耦合的激光雷达-惯性里程计技术 

**Authors**: Zhong Wang, Lele Ren, Yue Wen, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.04263)  

**Abstract**: Recent advancements in LiDAR-Inertial Odometry (LIO) have boosted a large amount of applications. However, traditional LIO systems tend to focus more on localization rather than mapping, with maps consisting mostly of sparse geometric elements, which is not ideal for downstream tasks. Recent emerging neural field technology has great potential in dense mapping, but pure LiDAR mapping is difficult to work on high-dynamic vehicles. To mitigate this challenge, we present a new solution that tightly couples geometric kinematics with neural fields to enhance simultaneous state estimation and dense mapping capabilities. We propose both semi-coupled and tightly coupled Kinematic-Neural LIO (KN-LIO) systems that leverage online SDF decoding and iterated error-state Kalman filtering to fuse laser and inertial data. Our KN-LIO minimizes information loss and improves accuracy in state estimation, while also accommodating asynchronous multi-LiDAR inputs. Evaluations on diverse high-dynamic datasets demonstrate that our KN-LIO achieves performance on par with or superior to existing state-of-the-art solutions in pose estimation and offers improved dense mapping accuracy over pure LiDAR-based methods. The relevant code and datasets will be made available at https://**. 

**Abstract (ZH)**: 近年来，LiDAR-惯性定位技术（LIO）的进步极大地推动了众多应用的发展。然而，传统LIO系统往往更侧重于定位而非建图，生成的图主要由稀疏的几何元素构成，这不利于下游任务的应用。近期兴起的神经场技术在稠密建图方面潜力巨大，但纯LiDAR建图在处理高动态车辆时较为困难。为解决这一挑战，我们提出了一种新的解决方案，即将几何运动学与神经场紧密耦合以增强同时状态估计和稠密建图的能力。我们提出了半耦合和紧密耦合的运动学-神经场LIO（KN-LIO）系统，利用在线体素距离场（SDF）解码和迭代误差状态卡尔曼滤波（IEKF）来融合激光和惯性数据。我们的KN-LIO在状态估计中最大限度地减少了信息损失，提高了准确性，并且能够适应异步多LiDAR输入。在多样化的高动态数据集上的评估表明，我们的KN-LIO在姿态估计上的性能与现有最先进的解决方案相当或更优，并且在基于纯LiDAR的方法上实现了更优的稠密建图精度。相关的代码和数据集将在https://** 供下载和使用。 

---
# Integrated Offline and Online Learning to Solve a Large Class of Scheduling Problems 

**Title (ZH)**: 集成离线和在线学习以解决一类大规模调度问题 

**Authors**: Anbang Liu, Zhi-Long Chen, Jinyang Jiang, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.04253)  

**Abstract**: In this paper, we develop a unified machine learning (ML) approach to predict high-quality solutions for single-machine scheduling problems with a non-decreasing min-sum objective function with or without release times. Our ML approach is novel in three major aspects. First, our approach is developed for the entire class of the aforementioned problems. To achieve this, we exploit the fact that the entire class of the problems considered can be formulated as a time-indexed formulation in a unified manner. We develop a deep neural network (DNN) which uses the cost parameters in the time-indexed formulation as the inputs to effectively predict a continuous solution to this formulation, based on which a feasible discrete solution is easily constructed. The second novel aspect of our approach lies in how the DNN model is trained. In view of the NP-hard nature of the problems, labels (i.e., optimal solutions) are hard to generate for training. To overcome this difficulty, we generate and utilize a set of special instances, for which optimal solutions can be found with little computational effort, to train the ML model offline. The third novel idea we employ in our approach is that we develop an online single-instance learning approach to fine tune the parameters in the DNN for a given online instance, with the goal of generating an improved solution for the given instance. To this end, we develop a feasibility surrogate that approximates the objective value of a given instance as a continuous function of the outputs of the DNN, which then enables us to derive gradients and update the learnable parameters in the DNN. Numerical results show that our approach can efficiently generate high-quality solutions for a variety of single-machine scheduling min-sum problems with up to 1000 jobs. 

**Abstract (ZH)**: 在本文中，我们提出了一种统一的机器学习（ML）方法，用于预测具有非递减最小和目标函数的单机调度问题的高质量解，该问题可以带有或不带有释放时间。我们的ML方法在三个方面具有创新性。首先，我们的方法适用于上述问题的整个类别。为此，我们利用整个问题类可以统一形式化为时间索引公式这一事实。我们开发了一个深度神经网络（DNN），该网络利用时间索引公式中的成本参数作为输入，以有效预测该公式的连续解，在此基础上很容易构造可行的离散解。其次，我们方法的第二个创新之处在于DNN模型的训练方式。鉴于问题的NP难性质，用于训练的标签（即最优解）很难生成。为了解决这一难题，我们生成并利用了一组特殊实例，对于这些实例，可以通过微小的计算努力找到最优解，从而预先训练ML模型。我们方法的第三个创新之处在于，我们开发了实时单实例学习方法，以针对给定的实时实例精细调整DNN中的参数，目标是为给定的实例生成更好的解。为此，我们开发了一个可行性的替代物，该替代物以DNN输出的连续函数近似给定实例的目标值，从而能够导出梯度并更新DNN中的可学习参数。数值结果表明，我们的方法可以高效地生成具有多达1000个任务的各种单机调度最小和问题的高质量解决方案。 

---
# Constraints as Rewards: Reinforcement Learning for Robots without Reward Functions 

**Title (ZH)**: 将约束作为奖励：无需奖励函数的机器人强化学习 

**Authors**: Yu Ishihara, Noriaki Takasugi, Kotaro Kawakami, Masaya Kinoshita, Kazumi Aoyama  

**Link**: [PDF](https://arxiv.org/pdf/2501.04228)  

**Abstract**: Reinforcement learning has become an essential algorithm for generating complex robotic behaviors. However, to learn such behaviors, it is necessary to design a reward function that describes the task, which often consists of multiple objectives that needs to be balanced. This tuning process is known as reward engineering and typically involves extensive trial-and-error. In this paper, to avoid this trial-and-error process, we propose the concept of Constraints as Rewards (CaR). CaR formulates the task objective using multiple constraint functions instead of a reward function and solves a reinforcement learning problem with constraints using the Lagrangian-method. By adopting this approach, different objectives are automatically balanced, because Lagrange multipliers serves as the weights among the objectives. In addition, we will demonstrate that constraints, expressed as inequalities, provide an intuitive interpretation of the optimization target designed for the task. We apply the proposed method to the standing-up motion generation task of a six-wheeled-telescopic-legged robot and demonstrate that the proposed method successfully acquires the target behavior, even though it is challenging to learn with manually designed reward functions. 

**Abstract (ZH)**: 强化学习已成为生成复杂机器人行为的关键算法。然而，为了学习这些行为，需要设计一个描述任务的奖励函数，该任务往往由多个需要平衡的目标组成。这一调整过程被称为奖励工程，通常涉及大量的试错过程。在本文中，为了避免这一试错过程，我们提出了一种约束作为奖励（Constraints as Rewards, CaR）的概念。CaR 使用多个约束函数而不是奖励函数来表示任务目标，并采用拉格朗日方法解决具有约束的强化学习问题。通过这种方法，不同的目标会自动实现平衡，因为拉格朗日乘子起到了目标之间的权重作用。此外，我们将证明，以不等式形式表示的约束为任务设计的优化目标提供了直观的解释。我们将所提出的方法应用于六轮伸缩腿机器人起立运动生成任务，并证明所提出的方法即使使用手动设计的奖励函数也难以学习，也能成功获得所需行为。 

---
# Agent Laboratory: Using LLM Agents as Research Assistants 

**Title (ZH)**: 代理实验室：将大型语言模型代理作为研究助理 

**Authors**: Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2501.04227)  

**Abstract**: Historically, scientific discovery has been a lengthy and costly process, demanding substantial time and resources from initial conception to final results. To accelerate scientific discovery, reduce research costs, and improve research quality, we introduce Agent Laboratory, an autonomous LLM-based framework capable of completing the entire research process. This framework accepts a human-provided research idea and progresses through three stages--literature review, experimentation, and report writing to produce comprehensive research outputs, including a code repository and a research report, while enabling users to provide feedback and guidance at each stage. We deploy Agent Laboratory with various state-of-the-art LLMs and invite multiple researchers to assess its quality by participating in a survey, providing human feedback to guide the research process, and then evaluate the final paper. We found that: (1) Agent Laboratory driven by o1-preview generates the best research outcomes; (2) The generated machine learning code is able to achieve state-of-the-art performance compared to existing methods; (3) Human involvement, providing feedback at each stage, significantly improves the overall quality of research; (4) Agent Laboratory significantly reduces research expenses, achieving an 84% decrease compared to previous autonomous research methods. We hope Agent Laboratory enables researchers to allocate more effort toward creative ideation rather than low-level coding and writing, ultimately accelerating scientific discovery. 

**Abstract (ZH)**: 历史上，科学研究是一个漫长且昂贵的过程，需要从初始构想到最终结果耗费大量的时间和资源。为了加快科学研究进程、降低研究成本并提高研究质量，我们引入了Agent Laboratory这一自主的基于语言模型的框架，能够完成整个研究过程。该框架接受人类提供的研究想法，并通过三个阶段——文献回顾、实验和报告撰写，生成全面的研究输出，包括代码仓库和研究报告，同时允许用户在每个阶段提供反馈和指导。我们采用多种最先进的语言模型部署Agent Laboratory，并邀请多位研究人员通过参与调查、提供人类反馈来指导研究过程，并最终评估最终论文的质量。我们发现：(1) 由o1-preview驱动的Agent Laboratory产生最佳的研究成果；(2) 生成的机器学习代码能够与现有方法相比达到最先进的性能；(3) 人类参与，每个阶段提供反馈，显著提高了整体研究质量；(4) Agent Laboratory显著降低了研究成本，与之前的自主研究方法相比，研究费用降低了84%。我们希望Agent Laboratory能够使研究人员将更多精力投入到富有创造力的构想中，而非低级别的编码和撰写工作，从而最终加速科学研究的进程。 

---
# Continual Self-supervised Learning Considering Medical Domain Knowledge in Chest CT Images 

**Title (ZH)**: 考虑医学领域知识的持续自监督学习在胸部CT图像中的应用 

**Authors**: Ren Tasai, Guang Li, Ren Togo, Minghui Tang, Takaaki Yoshimura, Hiroyuki Sugimori, Kenji Hirata, Takahiro Ogawa, Kohsuke Kudo, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2501.04217)  

**Abstract**: We propose a novel continual self-supervised learning method (CSSL) considering medical domain knowledge in chest CT images. Our approach addresses the challenge of sequential learning by effectively capturing the relationship between previously learned knowledge and new information at different stages. By incorporating an enhanced DER into CSSL and maintaining both diversity and representativeness within the rehearsal buffer of DER, the risk of data interference during pretraining is reduced, enabling the model to learn more richer and robust feature representations. In addition, we incorporate a mixup strategy and feature distillation to further enhance the model's ability to learn meaningful representations. We validate our method using chest CT images obtained under two different imaging conditions, demonstrating superior performance compared to state-of-the-art methods. 

**Abstract (ZH)**: 我们提出了一种新颖的持续自监督学习方法（CSSL），该方法考虑了在胸部CT图像中的医学领域知识。我们的方法通过有效地捕捉先前学习的知识与新信息之间的关系，解决了顺序学习的挑战。通过将增强的DER（回放增强）集成到CSSL中，并在DER的回放缓冲区中保持多样性和代表性的平衡，减少了预训练过程中数据干扰的风险，从而使模型能够学习到更丰富和稳健的特征表示。此外，我们引入了mixup策略和特征蒸馏，以进一步增强模型学习有意义表示的能力。我们使用在两种不同的成像条件下获得的胸部CT图像验证了该方法，结果显示其性能优于现有最先进的方法。 

---
# UPAQ: A Framework for Real-Time and Energy-Efficient 3D Object Detection in Autonomous Vehicles 

**Title (ZH)**: UPAQ：自主车辆中实时高效3D物体检测的框架 

**Authors**: Abhishek Balasubramaniam, Febin P Sunny, Sudeep Pasricha  

**Link**: [PDF](https://arxiv.org/pdf/2501.04213)  

**Abstract**: To enhance perception in autonomous vehicles (AVs), recent efforts are concentrating on 3D object detectors, which deliver more comprehensive predictions than traditional 2D object detectors, at the cost of increased memory footprint and computational resource usage. We present a novel framework called UPAQ, which leverages semi-structured pattern pruning and quantization to improve the efficiency of LiDAR point-cloud and camera-based 3D object detectors on resource-constrained embedded AV platforms. Experimental results on the Jetson Orin Nano embedded platform indicate that UPAQ achieves up to 5.62x and 5.13x model compression rates, up to 1.97x and 1.86x boost in inference speed, and up to 2.07x and 1.87x reduction in energy consumption compared to state-of-the-art model compression frameworks, on the Pointpillar and SMOKE models respectively. 

**Abstract (ZH)**: 为了增强自动驾驶车辆（AV）的感知能力，近期的研究重点集中在3D对象检测器上，这类检测器相较于传统的2D对象检测器可以提供更加全面的预测，但代价是增加了内存占用和计算资源的使用。我们提出了一种名为UPAQ的新型框架，该框架利用半结构化模式剪枝和量化技术，在资源受限的嵌入式AV平台上提高LiDAR点云和基于摄像头的3D对象检测器的效率。在嵌入式平台Jetson Orin Nano上的实验结果表明，与最先进的模型压缩框架相比，UPAQ在PointPillar和SMOKE模型上分别实现了高达5.62倍和5.13倍的模型压缩率、高达1.97倍和1.86倍的推理速度提升以及高达2.07倍和1.87倍的能量消耗降低。 

---
# CURing Large Models: Compression via CUR Decomposition 

**Title (ZH)**: CURing大型模型：基于CUR分解的压缩方法 

**Authors**: Sanghyeon Park, Soo-Mook Moon  

**Link**: [PDF](https://arxiv.org/pdf/2501.04211)  

**Abstract**: Large deep learning models have achieved remarkable success but are resource-intensive, posing challenges in computational cost and memory usage.
We introduce CURing, a novel model compression method based on CUR matrix decomposition, which approximates weight matrices as the product of selected columns (C) and rows (R), and a small linking matrix (U). We apply this decomposition to weights chosen based on the combined influence of their magnitudes and activations. By identifying and retaining informative rows and columns, CURing significantly reduces model size with minimal performance loss.
It preserves the original network's input/output structures, retains important features such as non-negativity, and the compressed model's activation patterns align with the original, thereby enhancing interpretability. 

**Abstract (ZH)**: 大型深度学习模型在各方面取得了显著的成果，但这些模型对计算成本和内存使用提出了挑战。
为此，我们提出了一种基于CUR矩阵分解的新型模型压缩方法——CURing。该方法通过将权重矩阵近似表示为选定列(C)和行(R)及其一个小链接矩阵(U)的乘积来进行模型压缩。我们根据权重幅度和激活值的综合影响选择这些权重。通过识别并保留有信息性的行和列，CURing显著减小了模型的大小，同时保持了最小的性能损失。
该方法保留了原始网络的输入/输出结构，保留了诸如非负性等重要的特征，并且压缩后的模型激活模式与原始模型相一致，这提高了模型的可解释性。 

---
# Generative Dataset Distillation Based on Self-knowledge Distillation 

**Title (ZH)**: 基于自我知识精炼的生成型数据集精炼 

**Authors**: Longzhen Li, Guang Li, Ren Togo, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2501.04202)  

**Abstract**: Dataset distillation is an effective technique for reducing the cost and complexity of model training while maintaining performance by compressing large datasets into smaller, more efficient versions. In this paper, we present a novel generative dataset distillation method that can improve the accuracy of aligning prediction logits. Our approach integrates self-knowledge distillation to achieve more precise distribution matching between the synthetic and original data, thereby capturing the overall structure and relationships within the data. To further improve the accuracy of alignment, we introduce a standardization step on the logits before performing distribution matching, ensuring consistency in the range of logits. Through extensive experiments, we demonstrate that our method outperforms existing state-of-the-art methods, resulting in superior distillation performance. 

**Abstract (ZH)**: 数据集蒸馏是一种有效的方法，能够在保持性能的前提下，通过压缩大规模数据集为更小、更高效的版本来降低模型训练的成本和复杂性。在本文中，我们提出了一种新颖的生成数据集蒸馏方法，以提高预测类别分布匹配的准确性。我们的方法通过集成自我知识蒸馏，实现了合成数据和原始数据之间更精确的分布匹配，从而捕捉数据的整体结构和关系。为了进一步提高匹配的准确性，我们在进行分布匹配之前引入了一个标准化步骤，确保类别评分（logits）的取值范围一致性。通过广泛的实验，我们证明了该方法优于现有最先进的方法，从而实现了更优的蒸馏性能。 

---
# GNN-based Decentralized Perception in Multirobot Systems for Predicting Worker Actions 

**Title (ZH)**: 基于GNN的多机器人系统去中心化感知方法及其在预测工人行动中的应用 

**Authors**: Ali Imran, Giovanni Beltrame, David St-Onge  

**Link**: [PDF](https://arxiv.org/pdf/2501.04193)  

**Abstract**: In industrial environments, predicting human actions is essential for ensuring safe and effective collaboration between humans and robots. This paper introduces a perception framework that enables mobile robots to understand and share information about human actions in a decentralized way. The framework first allows each robot to build a spatial graph representing its surroundings, which it then shares with other robots. This shared spatial data is combined with temporal information to track human behavior over time. A swarm-inspired decision-making process is used to ensure all robots agree on a unified interpretation of the human's actions. Results show that adding more robots and incorporating longer time sequences improve prediction accuracy. Additionally, the consensus mechanism increases system resilience, making the multi-robot setup more reliable in dynamic industrial settings. 

**Abstract (ZH)**: 在工业环境中，预测人类行为对于确保人类与机器人之间安全有效的协作至关重要。本文介绍了一种感知框架，使得移动机器人能够以去中心化的方式理解并共享人类行为信息。该框架首先允许每个机器人构建其周围环境的空间图，并将其与其他机器人共享。通过结合空间数据和时间信息，该框架跟踪人类行为随时间的变化。借鉴 swarm（群体智能）启发式的决策过程，确保所有机器人能够达成对人类行为一致的解释。实验结果表明，增加机器人数量并采用更长的时间序列能够提高预测准确性。此外，共识机制提高了系统的鲁棒性，在动态工业环境中使多机器人系统更加可靠。 

---
# Fixed Points of Deep Neural Networks: Emergence, Stability, and Applications 

**Title (ZH)**: 深度神经网络的不动点：涌现、稳定性和应用 

**Authors**: L. Berlyand, V. Slavin  

**Link**: [PDF](https://arxiv.org/pdf/2501.04182)  

**Abstract**: We present numerical and analytical results on the formation and stability of a family of fixed points of deep neural networks (DNNs). Such fixed points appear in a class of DNNs when dimensions of input and output vectors are the same. We demonstrate examples of applications of such networks in supervised, semi-supervised and unsupervised learning such as encoding/decoding of images, restoration of damaged images among others.
We present several numerical and analytical results. First, we show that for untrained DNN's with weights and biases initialized by normally distributed random variables the only one fixed point exists. This result holds for DNN with any depth (number of layers) $L$, any layer width $N$, and sigmoid-type activation functions. Second, it has been shown that for a DNN whose parameters (weights and biases) are initialized by ``light-tailed'' distribution of weights (e.g. normal distribution), after training the distribution of these parameters become ``heavy-tailed''. This motivates our study of DNNs with ``heavy-tailed'' initialization. For such DNNs we show numerically %existence and stability that training leads to emergence of $Q(N,L)$ fixed points, where $Q(N,L)$ is a positive integer which depends on the number of layers $L$ and layer width $N$. We further observe numerically that for fixed $N = N_0$ the function $Q(N_0, L)$ is non-monotone, that is it initially grows as $L$ increases and then decreases to 1.
This non-monotone behavior of $Q(N_0, L)$ is also obtained by analytical derivation of equation for Empirical Spectral Distribution (ESD) of input-output Jacobian followed by numerical solution of this equation. 

**Abstract (ZH)**: 我们呈现了关于深度神经网络（DNNs）的固定点形成及其稳定性的数值和分析结果。此类固定点出现在输入和输出向量维度相同的DNN类别中。我们展示了这些网络在有监督、半监督和无监督学习中的应用示例，如图像编码/解码、修复受损图像等。

我们呈现了几种数值和分析结果。首先，我们表明，在权重和偏置由正态分布的随机变量初始化的未训练的DNN中，仅存在一个固定点。这一结果适用于具有任意层数\(L\)和任意层宽\(N\)的任何深度DNN，并且激活函数为S型函数时也适用。其次，我们证明了，对于参数（权重和偏置）由“轻尾分布”（例如正态分布）初始化的DNN，在训练后，这些参数的分布变为“重尾分布”。这促使我们研究由重尾分布初始化的DNN。对于此类DNN，我们通过数值方法证明，训练会导致出现\(\text{Q}(N,L)\)个固定点，其中\(\text{Q}(N,L)\)是一个正整数，取决于层数\(L\)和层宽\(N\)。进一步观察发现，对于固定 \(N = N_0\)，函数\(\text{Q}(N_0, L)\)是非单调的，即在\(L\)增加时首先增长，然后减少至1。

非单调性\(\text{Q}(N_0, L)\)的行为还通过Empirical Spectral Distribution（ESD）的输入输出雅各比矩阵的经验谱分布方程的解析推导和数值解获取。 

---
# HIVEX: A High-Impact Environment Suite for Multi-Agent Research (extended version) 

**Title (ZH)**: HIVEX：多代理研究的高影响环境套件（扩展版） 

**Authors**: Philipp D. Siedler  

**Link**: [PDF](https://arxiv.org/pdf/2501.04180)  

**Abstract**: Games have been vital test beds for the rapid development of Agent-based research. Remarkable progress has been achieved in the past, but it is unclear if the findings equip for real-world problems. While pressure grows, some of the most critical ecological challenges can find mitigation and prevention solutions through technology and its applications. Most real-world domains include multi-agent scenarios and require machine-machine and human-machine collaboration. Open-source environments have not advanced and are often toy scenarios, too abstract or not suitable for multi-agent research. By mimicking real-world problems and increasing the complexity of environments, we hope to advance state-of-the-art multi-agent research and inspire researchers to work on immediate real-world problems. Here, we present HIVEX, an environment suite to benchmark multi-agent research focusing on ecological challenges. HIVEX includes the following environments: Wind Farm Control, Wildfire Resource Management, Drone-Based Reforestation, Ocean Plastic Collection, and Aerial Wildfire Suppression. We provide environments, training examples, and baselines for the main and sub-tasks. All trained models resulting from the experiments of this work are hosted on Hugging Face. We also provide a leaderboard on Hugging Face and encourage the community to submit models trained on our environment suite. 

**Abstract (ZH)**: 游戏一直是促进基于代理的（Agent-based）研究快速发展的关键平台。尽管以往取得了显著的进步，但这些发现是否适用于现实世界的问题仍不清楚。随着压力的增大，某些最关键生态挑战可以通过技术和其应用找到缓解和预防的解决方案。大多数现实世界领域包含多代理场景，并需要机器与机器之间的合作以及人类与机器之间的合作。开源环境尚未得到充分发展，且通常是过于简单的场景，过于抽象或不适合多代理研究。通过模拟现实世界问题并增加环境的复杂性，我们希望推进前沿的多代理研究，并激励研究人员面对现实世界的问题。在此，我们提出了HIVEX，这是一个环境套件，用于对以生态挑战为重点的多代理研究进行基准测试。HIVEX 包括以下环境：风力农场控制（Wind Farm Control）、野火资源管理（Wildfire Resource Management）、无人机植树造林（Drone-Based Reforestation）、海洋塑料收集（Ocean Plastic Collection）和空中灭火（Aerial Wildfire Suppression）。我们提供了用于主任务和辅助任务的环境、训练示例和基准。本研究中所有训练模型的结果托管在 Hugging Face 上。我们还在 Hugging Face 上提供了一个排行榜，并鼓励社区提交在我们的环境套件中训练的模型。 

---
# Multimodal Multihop Source Retrieval for Web Question Answering 

**Title (ZH)**: 多模态多跳源检索以实现网页问答 

**Authors**: Navya Yarrabelly, Saloni Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2501.04173)  

**Abstract**: This work deals with the challenge of learning and reasoning over multi-modal multi-hop question answering (QA). We propose a graph reasoning network based on the semantic structure of the sentences to learn multi-source reasoning paths and find the supporting facts across both image and text modalities for answering the question. In this paper, we investigate the importance of graph structure for multi-modal multi-hop question answering. Our analysis is centered on WebQA. We construct a strong baseline model, that finds relevant sources using a pairwise classification task. We establish that, with the proper use of feature representations from pre-trained models, graph structure helps in improving multi-modal multi-hop question answering. We point out that both graph structure and adjacency matrix are task-related prior knowledge, and graph structure can be leveraged to improve the retrieval performance for the task. Experiments and visualized analysis demonstrate that message propagation over graph networks or the entire graph structure can replace massive multimodal transformers with token-wise cross-attention. We demonstrated the applicability of our method and show a performance gain of \textbf{4.6$\%$} retrieval F1score over the transformer baselines, despite being a very light model. We further demonstrated the applicability of our model to a large scale retrieval setting. 

**Abstract (ZH)**: 本研究聚焦于多模态多跳问答（QA）中的学习与推理挑战。我们提出了一种基于句子语义结构的图推理网络，用于学习多源推理路径，并在图像和文本模态之间寻找支持事实以回答问题。在本文中，我们探讨了图结构在多模态多跳问答中的重要性。我们的分析集中在WebQA数据集上。我们构建了一个强基线模型，该模型通过一对分类任务来查找相关源。我们证明，通过恰当利用预训练模型的特征表示，图结构有助于提升多模态多跳问答的效果。我们指出，图结构和邻接矩阵都是与任务相关的先验知识，图结构可以被利用来改进任务的检索性能。实验和可视化分析表明，图网络中的消息传播或整个图结构可以替代大量多模态变换器，使用的是基于令牌的双向注意机制。我们展示了本方法的应用性，并在保持轻量模型的情况下，与变换器基线相比，取得了**4.6%**的检索F1分数提升。此外，我们进一步展示了该模型在大规模检索环境中的应用性。 

---
# Learning to Transfer Human Hand Skills for Robot Manipulations 

**Title (ZH)**: 学习将人类手部技能转移至机器人操作 

**Authors**: Sungjae Park, Seungho Lee, Mingi Choi, Jiye Lee, Jeonghwan Kim, Jisoo Kim, Hanbyul Joo  

**Link**: [PDF](https://arxiv.org/pdf/2501.04169)  

**Abstract**: We present a method for teaching dexterous manipulation tasks to robots from human hand motion demonstrations. Unlike existing approaches that solely rely on kinematics information without taking into account the plausibility of robot and object interaction, our method directly infers plausible robot manipulation actions from human motion demonstrations. To address the embodiment gap between the human hand and the robot system, our approach learns a joint motion manifold that maps human hand movements, robot hand actions, and object movements in 3D, enabling us to infer one motion component from others. Our key idea is the generation of pseudo-supervision triplets, which pair human, object, and robot motion trajectories synthetically. Through real-world experiments with robot hand manipulation, we demonstrate that our data-driven retargeting method significantly outperforms conventional retargeting techniques, effectively bridging the embodiment gap between human and robotic hands. Website at this https URL. 

**Abstract (ZH)**: 我们提出了一种方法，用于从人类手部动作示范中 teaching 机器人进行灵巧操作任务。与现有依赖于仅运动学信息而不考虑机器人与物体交互可行性的方法不同，我们的方法可以直接从人类手部动作中推断出合理的机器人操作动作。为了弥合人类手部与机器人系统之间的具身差距，我们的方法学习了一个关节运动流形，该流形将人类手部动作、机器人手部操作和物体运动在三维空间中进行映射，从而能够从其中一个运动成分中推断出其他运动成分。我们的核心思想是生成合成的伪监督三元组，即合成人类、物体和机器人运动轨迹的配对。通过机器人手部操作的现实世界实验，我们证明了我们的数据驱动重新定位方法显著优于传统的重新定位技术，有效地弥合了人类手部与机器人手部之间的具身差距。更多详情请访问此网址：[该网址]。 

---
# Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation 

**Title (ZH)**: 增强推理的自训练方法在长文本个性化生成中的应用 

**Authors**: Alireza Salemi, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Weize Kong, Tao Chen, Zhuowan Li, Michael Bendersky, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2501.04167)  

**Abstract**: Personalized text generation requires a unique ability of large language models (LLMs) to learn from context that they often do not encounter during their standard training. One way to encourage LLMs to better use personalized context for generating outputs that better align with the user's expectations is to instruct them to reason over the user's past preferences, background knowledge, or writing style. To achieve this, we propose Reasoning-Enhanced Self-Training for Personalized Text Generation (REST-PG), a framework that trains LLMs to reason over personal data during response generation. REST-PG first generates reasoning paths to train the LLM's reasoning abilities and then employs Expectation-Maximization Reinforced Self-Training to iteratively train the LLM based on its own high-reward outputs. We evaluate REST-PG on the LongLaMP benchmark, consisting of four diverse personalized long-form text generation tasks. Our experiments demonstrate that REST-PG achieves significant improvements over state-of-the-art baselines, with an average relative performance gain of 14.5% on the benchmark. 

**Abstract (ZH)**: 个性化文本生成需要大型语言模型（LLMs）具备一种独特的从上下文中学习的能力，而这些上下文在它们的标准训练中可能不会遇到。为了鼓励LLMs更好地利用个性化上下文以生成更好地符合用户期望的输出，可以通过指导它们推理用户的过去偏好、背景知识或写作风格来实现这一目标。为此，我们提出了一种增强推理的自主训练框架（Reasoning-Enhanced Self-Training for Personalized Text Generation，REST-PG），该框架在响应生成过程中使LLMs能够推理个人数据。首先，REST-PG生成推理路径以增强LLMs的推理能力，然后利用期望最大化强化自主训练来基于其自身的高奖励输出迭代训练LLMs。我们在LongLaMP基准数据集上评估了REST-PG，该基准数据集包含四个多样化的个性化长文本生成任务。实验结果表明，与现有的最佳基线方法相比，REST-PG在基准数据集上取得了显著的性能提升，平均相对性能提升达到了14.5%。 

---
# BiasGuard: Guardrailing Fairness in Machine Learning Production Systems 

**Title (ZH)**: BiasGuard：约束机器学习生产系统中公平性的护栏 

**Authors**: Nurit Cohen-Inger, Seffi Cohen, Neomi Rabaev, Lior Rokach, Bracha Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2501.04142)  

**Abstract**: As machine learning (ML) systems increasingly impact critical sectors such as hiring, financial risk assessments, and criminal justice, the imperative to ensure fairness has intensified due to potential negative implications. While much ML fairness research has focused on enhancing training data and processes, addressing the outputs of already deployed systems has received less attention. This paper introduces 'BiasGuard', a novel approach designed to act as a fairness guardrail in production ML systems. BiasGuard leverages Test-Time Augmentation (TTA) powered by Conditional Generative Adversarial Network (CTGAN), a cutting-edge generative AI model, to synthesize data samples conditioned on inverted protected attribute values, thereby promoting equitable outcomes across diverse groups. This method aims to provide equal opportunities for both privileged and unprivileged groups while significantly enhancing the fairness metrics of deployed systems without the need for retraining. Our comprehensive experimental analysis across diverse datasets reveals that BiasGuard enhances fairness by 31% while only reducing accuracy by 0.09% compared to non-mitigated benchmarks. Additionally, BiasGuard outperforms existing post-processing methods in improving fairness, positioning it as an effective tool to safeguard against biases when retraining the model is impractical. 

**Abstract (ZH)**: 随着机器学习（ML）系统在招聘、金融风险评估和刑事司法等关键领域的影响日益增强，确保公平性的需求因潜在的负面影响而变得更为迫切。尽管大量关于ML公平性的研究集中在提升训练数据和流程上，但对已部署系统的输出进行改进的关注度较低。本文介绍了“BiasGuard”这一创新方法，旨在作为生产中ML系统的公平性门槛。BiasGuard利用由条件生成对抗网络（CTGAN）驱动的测试时增强（TTA）技术，通过生成受反转受保护属性值条件限制的数据样本，促进了不同群体中的公平结果。该方法旨在为特权群体和非特权群体提供平等的机会，同时大幅提高已部署系统的公平性指标，而无需重新训练。我们对多种数据集进行全面的实验分析表明，与未处理基准相比，BiasGuard能够将公平性提高31%，同时仅将准确率降低0.09%。此外，BiasGuard在改进公平性方面优于现有后处理方法，使其成为在重新训练模型不可行时防止偏差的有效工具。 

---
# TrojanDec: Data-free Detection of Trojan Inputs in Self-supervised Learning 

**Title (ZH)**: TrojanDec：自我监督学习中恶意输入检测的无数据方法 

**Authors**: Yupei Liu, Yanting Wang, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2501.04108)  

**Abstract**: An image encoder pre-trained by self-supervised learning can be used as a general-purpose feature extractor to build downstream classifiers for various downstream tasks. However, many studies showed that an attacker can embed a trojan into an encoder such that multiple downstream classifiers built based on the trojaned encoder simultaneously inherit the trojan behavior. In this work, we propose TrojanDec, the first data-free method to identify and recover a test input embedded with a trigger. Given a (trojaned or clean) encoder and a test input, TrojanDec first predicts whether the test input is trojaned. If not, the test input is processed in a normal way to maintain the utility. Otherwise, the test input will be further restored to remove the trigger. Our extensive evaluation shows that TrojanDec can effectively identify the trojan (if any) from a given test input and recover it under state-of-the-art trojan attacks. We further demonstrate by experiments that our TrojanDec outperforms the state-of-the-art defenses. 

**Abstract (ZH)**: 通过自主监督学习预先训练的图像编码器可以作为通用特征提取器，用于构建多种下游任务的分类器。然而，许多研究表明，攻击者可以将特洛伊木马嵌入编码器之中，使得基于被污染编码器构建的多个下游分类器同时继承特洛伊木马的行为。在本项工作中，我们提出了TrojanDec，这是首个无需数据的方法来识别和恢复嵌有触发器的测试输入。给定一个（被污染的或干净的）编码器和一个测试输入，TrojanDec 首先预测测试输入是否被污染。如果没有被污染，则测试输入以正常方式处理以保持其效用。否则，测试输入将被进一步恢复以移除触发器。我们的广泛评估表明，TrojanDec 能够有效识别给定测试输入中的特洛伊木马（如果有）并在最先进的特洛伊木马攻击下对其进行恢复。此外，我们的实验进一步证明，TrojanDec 在性能上优于最先进的防御方法。 

---
# Enhancing Distribution and Label Consistency for Graph Out-of-Distribution Generalization 

**Title (ZH)**: 增强图数据的分布和标签一致性以提高域外泛化能力 

**Authors**: Song Wang, Xiaodong Yang, Rashidul Islam, Huiyuan Chen, Minghua Xu, Jundong Li, Yiwei Cai  

**Link**: [PDF](https://arxiv.org/pdf/2501.04102)  

**Abstract**: To deal with distribution shifts in graph data, various graph out-of-distribution (OOD) generalization techniques have been recently proposed. These methods often employ a two-step strategy that first creates augmented environments and subsequently identifies invariant subgraphs to improve generalizability. Nevertheless, this approach could be suboptimal from the perspective of consistency. First, the process of augmenting environments by altering the graphs while preserving labels may lead to graphs that are not realistic or meaningfully related to the origin distribution, thus lacking distribution consistency. Second, the extracted subgraphs are obtained from directly modifying graphs, and may not necessarily maintain a consistent predictive relationship with their labels, thereby impacting label consistency. In response to these challenges, we introduce an innovative approach that aims to enhance these two types of consistency for graph OOD generalization. We propose a modifier to obtain both augmented and invariant graphs in a unified manner. With the augmented graphs, we enrich the training data without compromising the integrity of label-graph relationships. The label consistency enhancement in our framework further preserves the supervision information in the invariant graph. We conduct extensive experiments on real-world datasets to demonstrate the superiority of our framework over other state-of-the-art baselines. 

**Abstract (ZH)**: 为了应对图数据中的分布偏移问题，最近提出了一系列图数据域外泛化（Out-of-Distribution, OOD）的方法。这些方法通常采用两步策略，首先是创建增强环境，随后识别不变子图以提高泛化能力。然而，这种做法从一致性角度来看可能并不最优。首先，通过改变图结构来保留标签标签的增强环境过程，可能会导致生成不现实或与原始分布无关的图，从而缺乏分布一致性。其次，提取的不变子图通常是直接修改原始图所得，可能无法保持与标签之间一致的预测关系，从而影响标签一致性。为应对这些挑战，我们提出了一种创新方法，旨在同时提升这两种类型的一致性以增强图数据的域外泛化能力。我们提出了一种改进措施，能够在统一框架下获取增强和不变的图。利用增强图，我们可以在不破坏标签-图关系完整性的前提下丰富训练数据。此外，我们框架中的标签一致性增强进一步在不变图中保留监督信息。我们在多种真实世界数据集上进行了广泛实验，以证明我们框架在域外泛化能力上的优越性，相较于其他最先进的基线方法。 

---
# Multi-armed Bandit and Backbone boost Lin-Kernighan-Helsgaun Algorithm for the Traveling Salesman Problems 

**Title (ZH)**: 多臂bandit算法和主干增强的Lin-Kernighan-Helsgaun算法在旅行商问题中的应用 

**Authors**: Long Wang, Jiongzhi Zheng, Zhengda Xiong, Kun He  

**Link**: [PDF](https://arxiv.org/pdf/2501.04072)  

**Abstract**: The Lin-Kernighan-Helsguan (LKH) heuristic is a classic local search algorithm for the Traveling Salesman Problem (TSP). LKH introduces an $\alpha$-value to replace the traditional distance metric for evaluating the edge quality, which leads to a significant improvement. However, we observe that the $\alpha$-value does not make full use of the historical information during the search, and single guiding information often makes LKH hard to escape from some local optima. To address the above issues, we propose a novel way to extract backbone information during the TSP local search process, which is dynamic and can be updated once a local optimal solution is found. We further propose to combine backbone information, $\alpha$-value, and distance to evaluate the edge quality so as to guide the search. Moreover, we abstract their different combinations to arms in a multi-armed bandit (MAB) and use an MAB model to help the algorithm select an appropriate evaluation metric dynamically. Both the backbone information and MAB can provide diverse guiding information and learn from the search history to suggest the best metric. We apply our methods to LKH and LKH-3, which is an extension version of LKH that can be used to solve about 40 variant problems of TSP and Vehicle Routing Problem (VRP). Extensive experiments show the excellent performance and generalization capability of our proposed method, significantly improving LKH for TSP and LKH-3 for two representative TSP and VRP variants, the Colored TSP (CTSP) and Capacitated VRP with Time Windows (CVRPTW). 

**Abstract (ZH)**: 林-凯南-海尔舒安（LKH）启发式算法是一种经典的用于旅行商问题（TSP）的局部搜索算法。LKH 引入了一个 $\alpha$ 值来替代传统的距离度量，以评估边的质量，这显著提高了算法性能。然而，我们发现 $\alpha$ 值在搜索过程中未能充分利用历史信息，单一指导信息常使 LKH 难以跳出局部最优解。为解决上述问题，我们提出了一种新的方法，在 TSP 局部搜索过程中动态提取骨干信息，一旦找到局部最优解即可更新这些信息。此外，我们将骨干信息、$\alpha$ 值和距离结合，以评估边的质量并指导搜索。我们还将它们的不同组合抽象为多臂 bandit（MAB）中的“臂”，并使用 MAB 模型帮助算法动态选择适当的评估度量。骨干信息和 MAB 均能提供多样的指导信息，并从搜索历史中学习以推荐最优度量。我们将该方法应用于 LKH 和 LKH-3。LKH-3 是 LKH 的扩展版本，可用于解决大约 40 种 TSP 和车辆路径问题（VRP）变体。广泛的实验表明，我们的方法表现出色且具有良好的泛化能力，显著改进了 LKH 在 TSP 中的表现以及 LKH-3 在两个代表性 TSP 和 VRP 变体（有色 TSP 和带时间窗的容量约束 VRP）中的表现。 

---
# More is not always better? Enhancing Many-Shot In-Context Learning with Differentiated and Reweighting Objectives 

**Title (ZH)**: 更多并不总能带来更好的效果？不同的和加权目标增强多样本上下文学习 

**Authors**: Xiaoqing Zhang, Ang Lv, Yuhan Liu, Flood Sung, Wei Liu, Shuo Shang, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2501.04070)  

**Abstract**: Large language models (LLMs) excel at few-shot in-context learning (ICL) without requiring parameter updates. However, as the number of ICL demonstrations increases from a few to many, performance tends to plateau and eventually decline. We identify two primary causes for this trend: the suboptimal negative log-likelihood (NLL) optimization objective and the incremental data noise. To address these issues, we introduce DR-ICL, a novel optimization method that enhances model performance through Differentiated Learning and advantage-based Reweighting objectives. Globally, DR-ICL utilizes differentiated learning to optimize the NLL objective, ensuring that many-shot performance surpasses zero-shot levels. Locally, it dynamically adjusts the weighting of many-shot demonstrations by leveraging cumulative advantages inspired by reinforcement learning, thereby improving generalization. This approach allows the model to handle varying numbers of shots effectively, mitigating the impact of noisy data. Recognizing the lack of multi-task datasets with diverse many-shot distributions, we develop the Many-Shot ICL Benchmark (MICLB)-a large-scale benchmark covering shot numbers from 1 to 350 within sequences of up to 8,000 tokens-for fine-tuning purposes. MICLB facilitates the evaluation of many-shot ICL strategies across seven prominent NLP tasks and 50 distinct datasets. Experimental results demonstrate that LLMs enhanced with DR-ICL achieve significant improvements in many-shot setups across various tasks, including both in-domain and out-of-domain scenarios. We release the code and benchmark dataset hoping to facilitate further research in many-shot ICL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在少量示例上下文学习（ICL）中表现出色，无需更新参数。然而，随着ICL示例数量从少数增加到大量，性能往往会面临瓶颈并最终下降。我们识别出这种趋势的两个主要原因：次优的负对数似然（NLL）优化目标以及增量数据噪声。为解决这些问题，我们提出了一种新的优化方法DR-ICL，通过差异化的学习和基于优势的重新加权目标来提升模型性能。全局上，DR-ICL 利用差异化的学习来优化NLL目标，确保多示例情况下的性能超过零示例水平。局部上，DR-ICL 通过借鉴强化学习中的累积优势动态调整多示例示例的权重，从而提高泛化能力。这种方法使模型能够有效地处理不同数量的示例，从而减轻噪声数据的影响。鉴于缺乏具有多种多示例分布的多任务数据集，我们开发了Many-Shot ICL基准（MICLB），这是一个大规模基准，覆盖了最多8,000个标记序列内的1到350个示例数，用于微调目的。MICLB 便于在七个主要的自然语言处理（NLP）任务和50个不同数据集上评估多示例ICL策略。实验结果表明，通过DR-ICL增强的大规模语言模型在多种任务的不同场景（包括领域内和领域外场景）中实现了显著改进。我们发布了代码和基准数据集，希望促进进一步研究多示例ICL领域。 

---
# Explainable Reinforcement Learning for Formula One Race Strategy 

**Title (ZH)**: 可解释的强化学习在一级方程式赛车战术中的应用 

**Authors**: Devin Thomas, Junqi Jiang, Avinash Kori, Aaron Russo, Steffen Winkler, Stuart Sale, Joseph McMillan, Francesco Belardinelli, Antonio Rago  

**Link**: [PDF](https://arxiv.org/pdf/2501.04068)  

**Abstract**: In Formula One, teams compete to develop their cars and achieve the highest possible finishing position in each race. During a race, however, teams are unable to alter the car, so they must improve their cars' finishing positions via race strategy, i.e. optimising their selection of which tyre compounds to put on the car and when to do so. In this work, we introduce a reinforcement learning model, RSRL (Race Strategy Reinforcement Learning), to control race strategies in simulations, offering a faster alternative to the industry standard of hard-coded and Monte Carlo-based race strategies. Controlling cars with a pace equating to an expected finishing position of P5.5 (where P1 represents first place and P20 is last place), RSRL achieves an average finishing position of P5.33 on our test race, the 2023 Bahrain Grand Prix, outperforming the best baseline of P5.63. We then demonstrate, in a generalisability study, how performance for one track or multiple tracks can be prioritised via training. Further, we supplement model predictions with feature importance, decision tree-based surrogate models, and decision tree counterfactuals towards improving user trust in the model. Finally, we provide illustrations which exemplify our approach in real-world situations, drawing parallels between simulations and reality. 

**Abstract (ZH)**: 在一级方程式赛车中，车队致力于开发自己的赛车，并在每一场比赛中争取达到最高的完赛位置。然而，在比赛中，车队无法改变赛车的设计，因此他们必须通过比赛策略来改善赛车的完赛位置，即优化选择什么轮胎化合物以及何时使用这些化合物。在本工作中，我们引入了一个强化学习模型，RSRL（Race Strategy Reinforcement Learning），用于在模拟中控制比赛策略，提供了一个比传统硬编码和蒙特卡洛方法更快的选择。使用与预期完赛车位P5.5相当的速度（其中P1代表第一名，P20代表最后一名），RSRL在我们测试的2023年巴林大奖赛中实现平均完赛车位P5.33，优于基准模型的平均完赛车位P5.63。随后，我们在泛化研究中展示了如何通过训练优先处理单一赛道或多个赛道的性能。此外，我们通过特征重要性、基于决策树的代理模型以及决策树反事实分析补充了模型预测，以提高用户对模型的信任度。最后，我们提供了实际应用示例图，通过将模拟与现实进行类比，展示了我们的方法在实际情境中的应用。 

---
# Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy 

**Title (ZH)**: 可解释的赛车轮胎能量时间序列预测在一级方程式赛车策略中的应用 

**Authors**: Jamie Todd, Junqi Jiang, Aaron Russo, Steffen Winkler, Stuart Sale, Joseph McMillan, Antonio Rago  

**Link**: [PDF](https://arxiv.org/pdf/2501.04067)  

**Abstract**: Formula One (F1) race strategy takes place in a high-pressure and fast-paced environment where split-second decisions can drastically affect race results. Two of the core decisions of race strategy are when to make pit stops (i.e. replace the cars' tyres) and which tyre compounds (hard, medium or soft, in normal conditions) to select. The optimal pit stop decisions can be determined by estimating the tyre degradation of these compounds, which in turn can be computed from the energy applied to each tyre, i.e. the tyre energy. In this work, we trained deep learning models, using the Mercedes-AMG PETRONAS F1 team's historic race data consisting of telemetry, to forecast tyre energies during races. Additionally, we fitted XGBoost, a decision tree-based machine learning algorithm, to the same dataset and compared the results, with both giving impressive performance. Furthermore, we incorporated two different explainable AI methods, namely feature importance and counterfactual explanations, to gain insights into the reasoning behind the forecasts. Our contributions thus result in an explainable, automated method which could assist F1 teams in optimising their race strategy. 

**Abstract (ZH)**: 一级方程式（F1）比赛策略处于一个高压且快节奏的环境中，每一秒的决策都可能对比赛结果产生巨大影响。比赛策略的核心决策之一是在何时进站（即更换轮胎），另一项关键决策是选择哪种轮胎化合物（正常条件下，分别为硬胎、中胎和软胎）。通过估算这些化合物的轮胎磨损情况，可以确定最佳的进站时机，这种轮胎磨损情况可以通过计算每个轮胎受到的能量来计算，即轮胎能量。在本研究中，我们利用梅赛德斯-AMG 弗拉诺石油 F1 车队的历史比赛数据（包括遥测数据）训练了深度学习模型，以预测比赛中的轮胎能量。此外，我们还使用了基于决策树的机器学习算法 XGBoost 对相同数据集进行了拟合，并对结果进行了比较，两者均表现出色。在此基础上，我们引入了两种不同的可解释AI方法——特征重要性和反事实解释——以深入了解模型预测背后的推理过程。我们的贡献因此提供了一种可解释的自动化方法，可以帮助F1车队优化其比赛策略。 

---
# ChronoLLM: A Framework for Customizing Large Language Model for Digital Twins generalization based on PyChrono 

**Title (ZH)**: ChronoLLM：一种基于PyChrono的数字孪生通用化大型语言模型自定义框架 

**Authors**: Jingquan Wang, Harry Zhang, Khailanii Slaton, Shu Wang, Radu Serban, Jinlong Wu, Dan Negrut  

**Link**: [PDF](https://arxiv.org/pdf/2501.04062)  

**Abstract**: Recently, the integration of advanced simulation technologies with artificial intelligence (AI) is revolutionizing science and engineering research. ChronoLlama introduces a novel framework that customizes the open-source LLMs, specifically for code generation, paired with PyChrono for multi-physics simulations. This integration aims to automate and improve the creation of simulation scripts, thus enhancing model accuracy and efficiency. This combination harnesses the speed of AI-driven code generation with the reliability of physics-based simulations, providing a powerful tool for researchers and engineers. Empirical results indicate substantial enhancements in simulation setup speed, accuracy of the generated codes, and overall computational efficiency. ChronoLlama not only expedites the development and testing of multibody systems but also spearheads a scalable, AI-enhanced approach to managing intricate mechanical simulations. This pioneering integration of cutting-edge AI with traditional simulation platforms represents a significant leap forward in automating and optimizing design processes in engineering applications. 

**Abstract (ZH)**: 近年来，先进的仿真技术与人工智能（AI）的集成正在重塑科学研究和工程研究。ChronoLlama 引入了一种新颖的框架，专门为代码生成定制开源的大规模语言模型（LLM），并结合 PyChrono 进行多物理场仿真。这种集成旨在自动化并提高仿真脚本的创建，从而提升模型的准确性和效率。这种组合利用了AI驱动的代码生成的快速性以及基于物理的仿真的可靠性，为研究人员和工程师提供了一个强大的工具。实验证明，这种方法在仿真设置速度、生成代码的准确性以及总体计算效率方面取得了显著的提升。ChronoLlama 不仅加速了多体系统的开发和测试，还引领了一种可扩展的、AI增强的管理复杂机械仿真过程的方法。这种将前沿AI与传统仿真平台相结合的开创性整合，代表了在工程应用中自动化和优化设计流程的重要进步。 

---
# Traits of a Leader: User Influence Level Prediction through Sociolinguistic Modeling 

**Title (ZH)**: 领导者特质：基于社会语言学建模的用户影响力等级预测 

**Authors**: Denys Katerenchuk, Rivka Levitan  

**Link**: [PDF](https://arxiv.org/pdf/2501.04046)  

**Abstract**: Recognition of a user's influence level has attracted much attention as human interactions move online. Influential users have the ability to sway others' opinions to achieve some goals. As a result, predicting users' level of influence can help to understand social networks, forecast trends, prevent misinformation, etc. However, predicting user influence is a challenging problem because the concept of influence is specific to a situation or a domain, and user communications are limited to text. In this work, we define user influence level as a function of community endorsement and develop a model that significantly outperforms the baseline by leveraging demographic and personality data. This approach consistently improves RankDCG scores across eight different domains. 

**Abstract (ZH)**: 将人类互动转移到网络上后，识别用户的影响水平已引起了广泛关注。具有影响力的人用户有能力影响他人意见以实现某些目标。因此，预测用户的影响力水平有助于理解社交网络、预测趋势、防止虚假信息等。然而，预测用户影响力是一个极具挑战性的问题，因为影响力的概念在不同的情况或领域中都有特定的含义，而用户之间的交流仅限于文本。在本项工作中，我们定义用户的影响水平为社区背书的函数，并开发了一个模型，该模型通过利用人口统计学和个性数据显著优于基线模型。该方法在八个不同的领域中持续提高了RankDCG分数。 

---
# A Survey on Large Language Models with some Insights on their Capabilities and Limitations 

**Title (ZH)**: 大型语言模型综述：关于其能力与局限性的某些见解 

**Authors**: Andrea Matarazzo, Riccardo Torlone  

**Link**: [PDF](https://arxiv.org/pdf/2501.04040)  

**Abstract**: The rapid advancement of artificial intelligence, particularly with the development of Large Language Models (LLMs) built on the transformer architecture, has redefined the capabilities of natural language processing. These models now exhibit remarkable performance across various language-related tasks, such as text generation, question answering, translation, and summarization, often rivaling human-like comprehension. More intriguingly, LLMs have demonstrated emergent abilities extending beyond their core functions, showing proficiency in tasks like commonsense reasoning, code generation, and arithmetic. This survey paper explores the foundational components, scaling mechanisms, and architectural strategies that drive these capabilities. Emphasizing models like GPT and LLaMA, we analyze the impact of exponential data and computational growth on LLM performance, while also addressing the trade-offs associated with scaling. We also examine LLM applications across sectors, such as healthcare, finance, education, and law, highlighting their adaptability and potential to solve domain-specific challenges. Central to this work are the questions of how LLMs generalize across diverse tasks, exhibit planning, and reasoning abilities, and whether these emergent abilities can be systematically elicited or enhanced. In particular, we provide some insights into the CoT (Chain of Thought) and PoT (Plan of Thought) abilities within LLMs, focusing on how pre-training data influences their emergence. Additionally, we investigate LLM-modulo frameworks that integrate external systems, allowing LLMs to handle complex, dynamic tasks. By analyzing these factors, this paper aims to foster the ongoing discussion on the capabilities and limits of LLMs, promoting their responsible development and application in novel and increasingly complex environments. 

**Abstract (ZH)**: 人工智能的迅速发展，特别是在基于Transformers架构构建的大规模语言模型（LLMs）的发展下，重新定义了自然语言处理的能力。这些模型在文本生成、问答、翻译和总结等多种语言相关任务中表现出卓越的性能，往往能够媲美人类的理解能力。更令人着迷的是，LLMs展示了超出其核心功能的新兴能力，展示了在常识推理、代码生成和算术任务上的专长。本文综述了这些能力的基础组件、扩展示例机制和架构策略。以GPT和LLaMA为代表的模型为例，我们分析了指数级数据和计算增长对LLMs性能的影响，并讨论了与扩展现有的权衡。此外，我们还探讨了LLMs在医疗保健、金融、教育和法律领域的广泛应用，突显了它们的适应能力和解决特定领域挑战的潜力。本文的关键问题在于LLMs在不同任务中的泛化能力、计划能力和推理能力，以及这些新兴能力能否系统地引发或增强。特别地，我们探讨了LLMs中CoT（思维链）和PoT（思维计划）能力，关注预训练数据对它们出现的影响。我们还研究了将外部系统集成到LLMs中的LLM-modulo框架，使LLMs能够处理复杂、动态的任务。通过对这些因素的分析，本文旨在促进对LLMs能力和限制的持续讨论，促进其负责任的开发和在日益复杂的新环境中应用。 

---
# Listening and Seeing Again: Generative Error Correction for Audio-Visual Speech Recognition 

**Title (ZH)**: 重新倾听与观看：生成式错误修正的视听语音识别 

**Authors**: Rui Liu, Hongyu Yuan, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.04038)  

**Abstract**: Unlike traditional Automatic Speech Recognition (ASR), Audio-Visual Speech Recognition (AVSR) takes audio and visual signals simultaneously to infer the transcription. Recent studies have shown that Large Language Models (LLMs) can be effectively used for Generative Error Correction (GER) in ASR by predicting the best transcription from ASR-generated N-best hypotheses. However, these LLMs lack the ability to simultaneously understand audio and visual, making the GER approach challenging to apply in AVSR. In this work, we propose a novel GER paradigm for AVSR, termed AVGER, that follows the concept of ``listening and seeing again''. Specifically, we first use the powerful AVSR system to read the audio and visual signals to get the N-Best hypotheses, and then use the Q-former-based Multimodal Synchronous Encoder to read the audio and visual information again and convert them into an audio and video compression representation respectively that can be understood by LLM. Afterward, the audio-visual compression representation and the N-Best hypothesis together constitute a Cross-modal Prompt to guide the LLM in producing the best transcription. In addition, we also proposed a Multi-Level Consistency Constraint training criterion, including logits-level, utterance-level and representations-level, to improve the correction accuracy while enhancing the interpretability of audio and visual compression representations. The experimental results on the LRS3 dataset show that our method outperforms current mainstream AVSR systems. The proposed AVGER can reduce the Word Error Rate (WER) by 24% compared to them. Code and models can be found at: this https URL. 

**Abstract (ZH)**: 与传统的自动语音识别（ASR）不同，音频-视觉语音识别（AVSR）同时利用音频和视觉信号来推断转写。 recent 研究表明，大语言模型（LLMs）能够有效用于ASR中的生成性错误纠正（GER），通过预测ASR生成的N-best假设中最优的转写结果。然而，这些LLMs缺乏同时理解音频和视觉信号的能力，使得GER方法在AVSR中的应用具有挑战性。在这项工作中，我们提出了一种新的AVSR的GER范式，称为AVGER，遵循“再次倾听并观察”的概念。具体而言，我们首先使用强大的AVSR系统阅读音频和视觉信号以获得N-best假设，然后使用基于Q-former的多模态同步编码器再次阅读音频和视觉信息，并分别将其转换为LLM能够理解的音频和视频压缩表示。随后，音频-视觉压缩表示和N-best假设共同构成交叉模态提示，以指导LLM生成最优的转写结果。此外，我们还提出了多级一致性约束训练准则，包括logits级、语句级和表示级，以提高纠错准确性并增强音频和视觉压缩表示的可解释性。在LRS3数据集上的实验证明了我们的方法优于当前主流的AVSR系统。所提出的方法在Word Error Rate（WER）上相比它们降低了24%。有关代码和模型可访问：this https URL。 

---
# AICat: An AI Cataloguing Approach to Support the EU AI Act 

**Title (ZH)**: AICat：支持欧盟人工智能法案的AI分类方法 

**Authors**: Delaram Golpayegani, Harshvardhan J. Pandit, Dave Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2501.04014)  

**Abstract**: The European Union's Artificial Intelligence Act (AI Act) requires providers and deployers of high-risk AI applications to register their systems into the EU database, wherein the information should be represented and maintained in an easily-navigable and machine-readable manner. Given the uptake of open data and Semantic Web-based approaches for other EU repositories, in particular the use of the Data Catalogue vocabulary Application Profile (DCAT-AP), a similar solution for managing the EU database of high-risk AI systems is needed. This paper introduces AICat - an extension of DCAT for representing catalogues of AI systems that provides consistency, machine-readability, searchability, and interoperability in managing open metadata regarding AI systems. This open approach to cataloguing ensures transparency, traceability, and accountability in AI application markets beyond the immediate needs of high-risk AI compliance in the EU. AICat is available online at this https URL under the CC-BY-4.0 license. 

**Abstract (ZH)**: 《欧洲联盟人工智能法案（AI法案）》要求提供和部署高风险人工智能应用的服务商将其系统注册到欧盟数据库中，信息应以易于导航和机器可读的形式呈现和维护。鉴于欧盟其他数据库采用了开放数据和基于语义网的方法，特别是使用了数据目录词汇表应用配置文件（DCAT-AP），对于管理欧盟高风险人工智能系统数据库，也需要类似的方法。本文介绍了一种名为AICat的解决方案，它是DCAT的扩展，用于表示人工智能系统的目录，提供了管理人工智能系统元数据的统一性、机器可读性、可搜索性和互操作性。这种开放式的目录方法确保了在欧盟高风险人工智能合规之外，人工智能应用市场中的透明性、可追溯性和可问责性。AICat可在以下网址在线获得：[此链接]，并采用CC-BY-4.0许可证。 

---
# A Generative AI-driven Metadata Modelling Approach 

**Title (ZH)**: 基于生成式人工智能的元数据建模方法 

**Authors**: Mayukh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.04008)  

**Abstract**: Since decades, the modelling of metadata has been core to the functioning of any academic library. Its importance has only enhanced with the increasing pervasiveness of Generative Artificial Intelligence (AI)-driven information activities and services which constitute a library's outreach. However, with the rising importance of metadata, there arose several outstanding problems with the process of designing a library metadata model impacting its reusability, crosswalk and interoperability with other metadata models. This paper posits that the above problems stem from an underlying thesis that there should only be a few core metadata models which would be necessary and sufficient for any information service using them, irrespective of the heterogeneity of intra-domain or inter-domain settings. To that end, this paper advances a contrary view of the above thesis and substantiates its argument in three key steps. First, it introduces a novel way of thinking about a library metadata model as an ontology-driven composition of five functionally interlinked representation levels from perception to its intensional definition via properties. Second, it introduces the representational manifoldness implicit in each of the five levels which cumulatively contributes to a conceptually entangled library metadata model. Finally, and most importantly, it proposes a Generative AI-driven Human-Large Language Model (LLM) collaboration based metadata modelling approach to disentangle the entanglement inherent in each representation level leading to the generation of a conceptually disentangled metadata model. Throughout the paper, the arguments are exemplified by motivating scenarios and examples from representative libraries handling cancer information. 

**Abstract (ZH)**: 自从几十年前起，元数据建模一直是任何学术图书馆运作的核心。随着生成型人工智能（AI）驱动的信息活动和服务的普及，这些服务构成了图书馆的延伸，元数据的重要性进一步提升。然而，随着元数据重要性的增强，设计图书馆元数据模型过程中出现了一些问题，这些问题影响了元数据模型的再利用性、转换兼容性和与其他元数据模型的互操作性。本文认为上述问题源于一个潜在的假设，即只应存在少数核心元数据模型，这些模型对任何使用它们的信息服务而言都是必要和充分的，无论是在域内还是跨域环境中存在异质性。为此，本文提出了一种相反的观点，并通过三个关键步骤来支持其论点。首先，它引入了一种新的思维方式，将图书馆元数据模型视为由感知到其扩展定义的五个功能上相互关联的表示层次构成的一种本体驱动的组合。其次，它介绍了每个五个层次中隐含的表示多样性，这些多样性累积起来共同构成了概念上交织的图书馆元数据模型。最后，而且最重要的是，它提出了一种基于生成型AI的人工智能-大型语言模型（LLM）协作的元数据建模方法，旨在解开每个表示层次中固有的交织，从而生成一种概念上解开的元数据模型。整篇论文通过代表图书馆处理癌症信息的相关情境和示例来例证这些论点。 

---
