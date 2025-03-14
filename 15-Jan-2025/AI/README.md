# ADAM-1: AI and Bioinformatics for Alzheimer's Detection and Microbiome-Clinical Data Integrations 

**Title (ZH)**: ADAM-1：人工智能和生物信息学在阿尔茨海默病检测及微生物群-临床数据整合中的应用 

**Authors**: Ziyuan Huang, Vishaldeep Kaur Sekhon, Ouyang Guo, Mark Newman, Roozbeh Sadeghian, Maria L. Vaida, Cynthia Jo, Doyle Ward, Vanni Bucci, John P. Haran  

**Link**: [PDF](https://arxiv.org/pdf/2501.08324)  

**Abstract**: The Alzheimer's Disease Analysis Model Generation 1 (ADAM) is a multi-agent large language model (LLM) framework designed to integrate and analyze multi-modal data, including microbiome profiles, clinical datasets, and external knowledge bases, to enhance the understanding and detection of Alzheimer's disease (AD). By leveraging retrieval-augmented generation (RAG) techniques along with its multi-agent architecture, ADAM-1 synthesizes insights from diverse data sources and contextualizes findings using literature-driven evidence. Comparative evaluation against XGBoost revealed similar mean F1 scores but significantly reduced variance for ADAM-1, highlighting its robustness and consistency, particularly in small laboratory datasets. While currently tailored for binary classification tasks, future iterations aim to incorporate additional data modalities, such as neuroimaging and biomarkers, to broaden the scalability and applicability for Alzheimer's research and diagnostics. 

**Abstract (ZH)**: 阿尔茨海默病分析模型生成1（ADAM）是一种多agent大型语言模型（LLM）框架，旨在整合和分析多模态数据，包括微生物群概述、临床数据集和外部知识库，以增强对阿尔茨海默病（AD）的理解和检测。通过利用检索增强生成（RAG）技术及其多agent架构，ADAM-1 从多种数据源中综合见解，并利用文献驱动的证据对研究结果进行情境化。与XGBoost的比较评价显示，ADAM-1 的平均F1分数与XGBoost相近，但在小实验室数据集中显著减少了变异度，突显了其稳健性和一致性。虽然目前主要用于二元分类任务，但未来的迭代版本计划引入额外的数据模态，如神经影像学和生物标志物，以扩大其在阿尔茨海默病研究和诊断中的规模性和适用性。 

---
# Optimization of Link Configuration for Satellite Communication Using Reinforcement Learning 

**Title (ZH)**: 使用强化学习优化卫星通信链路配置 

**Authors**: Tobias Rohe, Michael Kölle, Jan Matheis, Rüdiger Höpfl, Leo Sünkel, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2501.08220)  

**Abstract**: Satellite communication is a key technology in our modern connected world. With increasingly complex hardware, one challenge is to efficiently configure links (connections) on a satellite transponder. Planning an optimal link configuration is extremely complex and depends on many parameters and metrics. The optimal use of the limited resources, bandwidth and power of the transponder is crucial. Such an optimization problem can be approximated using metaheuristic methods such as simulated annealing, but recent research results also show that reinforcement learning can achieve comparable or even better performance in optimization methods. However, there have not yet been any studies on link configuration on satellite transponders. In order to close this research gap, a transponder environment was developed as part of this work. For this environment, the performance of the reinforcement learning algorithm PPO was compared with the metaheuristic simulated annealing in two experiments. The results show that Simulated Annealing delivers better results for this static problem than the PPO algorithm, however, the research in turn also underlines the potential of reinforcement learning for optimization problems. 

**Abstract (ZH)**: 卫星通信是我们当今互联互通世界中的关键技术。随着硬件设备越来越复杂，一个挑战是如何高效地配置卫星转发器中的链路（连接）。进行最优链路配置极其复杂，且取决于许多参数和指标。合理利用有限的转发器资源，包括带宽和功率至关重要。这种优化问题可以通过元启发式方法（如模拟退火）进行近似求解，但最近的研究结果也表明，强化学习在优化方法中能够达到相媲美甚至更好的性能。然而，目前还没有关于卫星转发器链路配置的相关研究。为填补这一研究空白，本研究在此过程中开发了一个转发器环境。在此环境中，将基于两组实验比较强化学习算法PPO与元启发式方法模拟退火的性能。结果表明，模拟退火在这一静态问题上提供了更好的表现，但研究同时也凸显了强化学习在解决优化问题方面的潜力。 

---
# PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving 

**Title (ZH)**: PRESERVE: 分布式大规模语言模型服务中前取模型权重和KV缓存的预案方法 

**Authors**: Ahmet Caner Yüzügüler, Jiawei Zhuang, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2501.08192)  

**Abstract**: Large language models (LLMs) are widely used across various applications, but their substantial computational requirements pose significant challenges, particularly in terms of HBM bandwidth bottlenecks and inter-device communication overhead. In this paper, we present PRESERVE, a novel prefetching framework designed to optimize LLM inference by overlapping memory reads for model weights and KV-cache with collective communication operations. Through extensive experiments conducted on commercial AI accelerators, we demonstrate up to 1.6x end-to-end speedup on state-of-the-art, open-source LLMs. Additionally, we perform a design space exploration that identifies the optimal hardware configuration for the proposed method, showing a further 1.25x improvement in performance per cost by selecting the optimal L2 cache size. Our results show that PRESERVE has the potential to mitigate the memory bottlenecks and communication overheads, offering a solution to improve the performance and scalability of the LLM inference systems. 

**Abstract (ZH)**: 以下是该内容或标题的中文翻译，符合学术规范：

大型语言模型（LLMs）在各种应用中被广泛应用，但其巨大的计算需求在带宽瓶颈（特别是HBM带宽瓶颈）和设备间通信开销方面提出了重大挑战。本文提出了一种名为PRESERVE的新颖预取框架，旨在通过将模型权重和KV缓存的内存读取与集体通信操作重叠来优化LLM推理。通过在商业AI加速器上进行广泛的实验，我们展示了在当前最先进的开源LLM上实现最高1.6倍的端到端加速。此外，我们还进行了设计空间探索，以确定所提出方法的最佳硬件配置，结果显示通过选择最优L2缓存大小，性能提升了1.25倍。我们的结果表明，PRESERVE有可能缓解内存瓶颈和通信开销问题，为提高LLM推理系统的性能和可扩展性提供了解决方案。 

---
# Assessing AI Adoption and Digitalization in SMEs: A Framework for Implementation 

**Title (ZH)**: 评估中小企业中AI采纳与数字化转型的框架：实施框架 

**Authors**: Serena Proietti, Roberto Magnani  

**Link**: [PDF](https://arxiv.org/pdf/2501.08184)  

**Abstract**: The primary objective of this research is to examine the current state of digitalization and the integration of artificial intelligence (AI) within small and medium-sized enterprises (SMEs) in Italy. There is a significant gap between SMEs and large corporations in their use of AI, with SMEs facing numerous barriers to adoption. This study identifies critical drivers and obstacles to achieving intelligent transformation, proposing a framework model to address key challenges and provide actionable guidelines 

**Abstract (ZH)**: 本研究的主要目标是考察意大利中小企业在数字化转型和人工智能（AI）集成方面的现状。中小企业与大型企业之间在AI应用方面存在显著差距，中小企业在采用AI时面临诸多障碍。本研究识别了实现智能转型的关键驱动力和障碍，并提出了一种框架模型，以应对关键挑战并提供可操作的指导方针。 

---
# CG-MER: A Card Game-based Multimodal dataset for Emotion Recognition 

**Title (ZH)**: CG-MER：一种基于纸牌游戏的多模态情感识别数据集 

**Authors**: Nessrine Farhat, Amine Bohi, Leila Ben Letaifa, Rim Slama  

**Link**: [PDF](https://arxiv.org/pdf/2501.08182)  

**Abstract**: The field of affective computing has seen significant advancements in exploring the relationship between emotions and emerging technologies. This paper presents a novel and valuable contribution to this field with the introduction of a comprehensive French multimodal dataset designed specifically for emotion recognition. The dataset encompasses three primary modalities: facial expressions, speech, and gestures, providing a holistic perspective on emotions. Moreover, the dataset has the potential to incorporate additional modalities, such as Natural Language Processing (NLP) to expand the scope of emotion recognition research. The dataset was curated through engaging participants in card game sessions, where they were prompted to express a range of emotions while responding to diverse questions. The study included 10 sessions with 20 participants (9 females and 11 males). The dataset serves as a valuable resource for furthering research in emotion recognition and provides an avenue for exploring the intricate connections between human emotions and digital technologies. 

**Abstract (ZH)**: 情感计算领域在探索情感与新兴技术之间的关系方面取得了显著进展。本文在此领域中做出了一个创新且宝贵的贡献，通过引入一个全面的法语多模态数据集，专门用于情感识别。该数据集包含三个主要模态：面部表情、语音和手势，提供了对情感的全方位视角。此外，该数据集还具备通过整合自然语言处理（NLP）等其他模态来扩展情感识别研究范围的潜力。数据集是通过让参与者参与卡片游戏会话，并在回答不同问题的同时表达各种情感来收集的。研究包括了10个会话，共20名参与者（其中9名女性和11名男性）。该数据集为情感识别研究提供了宝贵的资源，并为探索人的情感与数字技术之间的复杂联系开辟了路径。 

---
# LeapVAD: A Leap in Autonomous Driving via Cognitive Perception and Dual-Process Thinking 

**Title (ZH)**: LeapVAD：通过认知感知与双过程思维实现的自动驾驶技术跃升 

**Authors**: Yukai Ma, Tiantian Wei, Naiting Zhong, Jianbiao Mei, Tao Hu, Licheng Wen, Xuemeng Yang, Botian Shi, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08168)  

**Abstract**: While autonomous driving technology has made remarkable strides, data-driven approaches still struggle with complex scenarios due to their limited reasoning capabilities. Meanwhile, knowledge-driven autonomous driving systems have evolved considerably with the popularization of visual language models. In this paper, we propose LeapVAD, a novel method based on cognitive perception and dual-process thinking. Our approach implements a human-attentional mechanism to identify and focus on critical traffic elements that influence driving decisions. By characterizing these objects through comprehensive attributes - including appearance, motion patterns, and associated risks - LeapVAD achieves more effective environmental representation and streamlines the decision-making process. Furthermore, LeapVAD incorporates an innovative dual-process decision-making module miming the human-driving learning process. The system consists of an Analytic Process (System-II) that accumulates driving experience through logical reasoning and a Heuristic Process (System-I) that refines this knowledge via fine-tuning and few-shot learning. LeapVAD also includes reflective mechanisms and a growing memory bank, enabling it to learn from past mistakes and continuously improve its performance in a closed-loop environment. To enhance efficiency, we develop a scene encoder network that generates compact scene representations for rapid retrieval of relevant driving experiences. Extensive evaluations conducted on two leading autonomous driving simulators, CARLA and DriveArena, demonstrate that LeapVAD achieves superior performance compared to camera-only approaches despite limited training data. Comprehensive ablation studies further emphasize its effectiveness in continuous learning and domain adaptation. Project page: this https URL. 

**Abstract (ZH)**: 尽管自动驾驶技术取得了显著进展，但基于数据的方法在处理复杂场景时仍遇到挑战，因为它们的推理能力有限。与此同时，随着视觉语言模型的普及，知识驱动的自动驾驶系统有了显著的进化。本文提出了LeapVAD，一种基于认知感知和双过程思维的新方法。我们的方法采用了类似人类注意机制的方法，以识别并集中关注影响驾驶决策的关键交通元素。通过综合属性（包括外观、运动模式和相关风险）来描述这些对象，LeapVAD 实现了更有效的环境表示，并简化了决策过程。此外，LeapVAD 结合了一个创新的双过程决策模块，模仿了人类驾驶的学习过程。该系统由通过逻辑推理积累驾驶经验的解析过程（System-II）和通过微调和少量示例学习逐步优化知识的启发式过程（System-I）组成。LeapVAD 还包括了反思机制和增长记忆库，使系统能够从 past mistakes 中学习，并在闭环环境中不断提高性能。为了提高效率，我们开发了一种场景编码网络，能够生成紧凑的场景表示，从而快速检索相关驾驶经验。在两个领先的自动驾驶模拟器（CARLA 和 DriveArena）上进行的广泛评估表明，即使在有限的训练数据下，LeapVAD 也实现了优于基于摄像头的方法的性能。全面的消融研究进一步强调了其在持续学习和领域适应中的有效性。项目页面：[请提供具体链接]。 

---
# Multiple-Input Variational Auto-Encoder for Anomaly Detection in Heterogeneous Data 

**Title (ZH)**: 异质数据中的异常检测的多输入变分自动编码器 

**Authors**: Phai Vu Dinh, Diep N. Nguyen, Dinh Thai Hoang, Quang Uy Nguyen, Eryk Dutkiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2501.08149)  

**Abstract**: Anomaly detection (AD) plays a pivotal role in AI applications, e.g., in classification, and intrusion/threat detection in cybersecurity. However, most existing methods face challenges of heterogeneity amongst feature subsets posed by non-independent and identically distributed (non-IID) data. We propose a novel neural network model called Multiple-Input Auto-Encoder for AD (MIAEAD) to address this. MIAEAD assigns an anomaly score to each feature subset of a data sample to indicate its likelihood of being an anomaly. This is done by using the reconstruction error of its sub-encoder as the anomaly score. All sub-encoders are then simultaneously trained using unsupervised learning to determine the anomaly scores of feature subsets. The final AUC of MIAEAD is calculated for each sub-dataset, and the maximum AUC obtained among the sub-datasets is selected. To leverage the modelling of the distribution of normal data to identify anomalies of the generative models, we develop a novel neural network architecture/model called Multiple-Input Variational Auto-Encoder (MIVAE). MIVAE can process feature subsets through its sub-encoders before learning distribution of normal data in the latent space. This allows MIVAE to identify anomalies that deviate from the learned distribution. We theoretically prove that the difference in the average anomaly score between normal samples and anomalies obtained by the proposed MIVAE is greater than that of the Variational Auto-Encoder (VAEAD), resulting in a higher AUC for MIVAE. Extensive experiments on eight real-world anomaly datasets demonstrate the superior performance of MIAEAD and MIVAE over conventional methods and the state-of-the-art unsupervised models, by up to 6% in terms of AUC score. Alternatively, MIAEAD and MIVAE have a high AUC when applied to feature subsets with low heterogeneity based on the coefficient of variation (CV) score. 

**Abstract (ZH)**: 异常检测（AD）在AI应用中发挥着关键作用，例如在分类、以及网络安全中的入侵/威胁检测。然而，大多数现有方法在处理非独立且非同分布（non-IID）数据所带来的特征子集的异质性时面临挑战。我们提出了一种称为多输入自动编码器（Multiple-Input Auto-Encoder for AD，MIAEAD）的新神经网络模型，以应对这一挑战。MIAEAD 为数据样本中的每个特征子集分配一个异常分数，以表示其为异常的概率。这通过使用其子编码器的重构误差作为异常分数来实现。所有子编码器随后使用无监督学习同时进行训练，以确定特征子集的异常分数。对于每个子数据集，计算MIAEAD的最终AUC值，并选择子数据集中获得的最大AUC值。为了充分利用生成模型对正常数据分布建模的优势，我们还开发了一种称为多输入变分自编码器（Multiple-Input Variational Auto-Encoder，MIVAE）的新神经网络架构/模型。MIVAE 能够通过其子编码器处理特征子集，然后在潜在空间中学习正常数据的分布。这样，MIVAE 就能够识别与学习到的分布相偏离的异常。我们理论上证明，与提出的MIVAE相比，变分自编码器（VAEAD）获得的正常样本与异常样本之间的平均异常分数差异较小，因此MIVAE的AUC更高。在八个真实世界的异常检测数据集上进行的广泛实验表明，MIAEAD和MIVAE在AUC评分上相较于传统方法和最先进的无监督模型具有显著优势，最高可达6%。另一方面，对于基于变异系数（CV）分数具有低异质性的特征子集，MIAEAD和MIVAE也有较高的AUC值。 

---
# In-situ graph reasoning and knowledge expansion using Graph-PReFLexOR 

**Title (ZH)**: 基于图的原位推理与知识扩展方法：Graph-PReFLexOR技术 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2501.08120)  

**Abstract**: The pursuit of automated scientific discovery has fueled progress from symbolic logic to modern AI, forging new frontiers in reasoning and pattern recognition. Transformers function as potential systems, where every possible relationship remains latent potentiality until tasks impose constraints, akin to measurement. Yet, refining their sampling requires more than probabilistic selection: solutions must conform to specific structures or rules, ensuring consistency and the invocation of general principles. We present Graph-PReFLexOR (Graph-based Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning), a framework that combines graph reasoning with symbolic abstraction to dynamically expand domain knowledge. Inspired by reinforcement learning, Graph-PReFLexOR defines reasoning as a structured mapping, where tasks yield knowledge graphs, abstract patterns, and ultimately, final answers. Inspired by category theory, it encodes concepts as nodes and their relationships as edges, supporting hierarchical inference and adaptive learning through isomorphic representations. Demonstrations include hypothesis generation, materials design, and creative reasoning, such as discovering relationships between mythological concepts like 'thin places' with materials science. We propose a 'knowledge garden growth' strategy that integrates insights across domains, promoting interdisciplinary connections. Results with a 3-billion-parameter Graph-PReFLexOR model show superior reasoning depth and adaptability, underscoring the potential for transparent, multidisciplinary AI-driven discovery. It lays the groundwork for general autonomous reasoning solutions. 

**Abstract (ZH)**: 自动化的科学发现追求从符号逻辑到现代人工智能的进展，开辟了推理和模式识别的新前沿。变换器作为潜在系统运行，其中所有可能的关系都处于潜在状态，直到任务施加限制，类似于测量。然而，改进它们的采样需要更多于概率选择：解决方案必须符合特定的结构或规则，以确保一致性和一般原则的应用。我们介绍了一种基于图的偏好递归语言建模框架——Graph-PReFLexOR（基于图的偏好递归语言建模，用于推理的探索性优化），该框架结合了图推理和符号抽象，以动态扩展领域知识。该框架借鉴了强化学习的思想，将推理视为结构化的映射，任务生成知识图、抽象模式，并最终得出最终答案。该框架借鉴了范畴理论的思想，将概念编码为节点，其关系编码为边，通过同构表示支持分层推理和自适应学习。示例包括假设生成、材料设计以及创造性推理，如在材料科学中发现关于神话概念如“thin places”的关联。我们提出了一个“知识花园生长”的策略，促进跨学科的连接。基于含有30亿参数的Graph-PReFLexOR模型的结果显示，其推理深度和适应性优越，突出了透明、跨学科人工智能驱动发现的潜力。该框架为通用自主推理解决方案奠定了基础。 

---
# NOMTO: Neural Operator-based symbolic Model approximaTion and discOvery 

**Title (ZH)**: NOMTO：基于神经算子的符号模型逼近与发现 

**Authors**: Sergei Garmaev, Siddhartha Mishra, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2501.08086)  

**Abstract**: While many physical and engineering processes are most effectively described by non-linear symbolic models, existing non-linear symbolic regression (SR) methods are restricted to a limited set of continuous algebraic functions, thereby limiting their applicability to discover higher order non-linear differential relations. In this work, we introduce the Neural Operator-based symbolic Model approximaTion and discOvery (NOMTO) method, a novel approach to symbolic model discovery that leverages Neural Operators to encompass a broad range of symbolic operations. We demonstrate that NOMTO can successfully identify symbolic expressions containing elementary functions with singularities, special functions, and derivatives. Additionally, our experiments demonstrate that NOMTO can accurately rediscover second-order non-linear partial differential equations. By broadening the set of symbolic operations available for discovery, NOMTO significantly advances the capabilities of existing SR methods. It provides a powerful and flexible tool for model discovery, capable of capturing complex relations in a variety of physical systems. 

**Abstract (ZH)**: 尽管许多物理和工程过程最有效地由非线性符号模型描述，但现有非线性符号回归（SR）方法仅限于有限的连续代数函数，从而限制了它们发现更高阶非线性微分关系的应用范围。在本文中，我们引入了一种基于神经算子的符号模型逼近与发现（NOMTO）方法，这是一种利用神经算子来涵盖广泛符号操作的新颖方法。我们证明NOMTO能够成功识别包含奇异函数、特殊函数以及导数的符号表达式。此外，我们的实验表明，NOMTO能够准确地重新发现二阶非线性偏微分方程。通过拓宽可用于发现的符号操作范围，NOMTO显着提升了现有SR方法的功能。它提供了一种强大且灵活的工具，能够捕捉各种物理系统中的复杂关系。 

---
# Artificial Liver Classifier: A New Alternative to Conventional Machine Learning Models 

**Title (ZH)**: 人工肝分类器：一种新的传统机器学习模型的新选择 

**Authors**: Mahmood A. Jumaah, Yossra H. Ali, Tarik A. Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2501.08074)  

**Abstract**: Supervised machine learning classifiers often encounter challenges related to performance, accuracy, and overfitting. This paper introduces the Artificial Liver Classifier (ALC), a novel supervised learning classifier inspired by the human liver's detoxification function. The ALC is characterized by its simplicity, speed, hyperparameters-free, ability to reduce overfitting, and effectiveness in addressing multi-classification problems through straightforward mathematical operations. To optimize the ALC's parameters, an improved FOX optimization algorithm (IFOX) is employed as the training method. The proposed ALC was evaluated on five benchmark machine learning datasets: Iris Flower, Breast Cancer Wisconsin, Wine, Voice Gender, and MNIST. The results demonstrated competitive performance, with the ALC achieving 100% accuracy on the Iris dataset, surpassing logistic regression, multilayer perceptron, and support vector machine. Similarly, on the Breast Cancer dataset, it achieved 99.12% accuracy, outperforming XGBoost and logistic regression. Across all datasets, the ALC consistently exhibited lower overfitting gaps and loss compared to conventional classifiers. These findings highlight the potential of leveraging biological process simulations to develop efficient machine learning models and open new avenues for innovation in the field. 

**Abstract (ZH)**: 监督机器学习分类器常常面临性能、准确性和过拟合等方面的挑战。本文介绍了人工肝分类器（Artificial Liver Classifier, ALC），这是一种受人类肝脏解毒功能启发的新型监督学习分类器。ALC以简洁性、高效性、无超参数需求、减少过拟合能力以及通过简单的数学运算解决多分类问题为特点。为优化ALC的参数，采用了改进的FOX优化算法（Improved FOX Optimization Algorithm, IFOX）作为训练方法。所提出的人工肝分类器在五个基准机器学习数据集上进行了评估：鸢尾花、威斯康星乳腺癌、葡萄酒、语音性别和MNIST。实验结果显示，ALC在鸢尾花数据集上的准确率达到100%，超过了逻辑回归、多层感知机和支持向量机。在威斯康星乳腺癌数据集上，ALC实现了99.12%的准确率，超越了XGBoost和逻辑回归。总体而言，在所有数据集上，ALC表现出较低的过拟合差异和损失，相较于传统分类器更具优势。这些发现突显了利用生物过程模拟开发高效机器学习模型的潜力，并为该领域的创新开辟了新的方向。 

---
# A Roadmap to Guide the Integration of LLMs in Hierarchical Planning 

**Title (ZH)**: 指导大型语言模型融入分层规划的路线图 

**Authors**: Israel Puerta-Merino, Carlos Núñez-Molina, Pablo Mesejo, Juan Fernández-Olivares  

**Link**: [PDF](https://arxiv.org/pdf/2501.08068)  

**Abstract**: Recent advances in Large Language Models (LLMs) are fostering their integration into several reasoning-related fields, including Automated Planning (AP). However, their integration into Hierarchical Planning (HP), a subfield of AP that leverages hierarchical knowledge to enhance planning performance, remains largely unexplored. In this preliminary work, we propose a roadmap to address this gap and harness the potential of LLMs for HP. To this end, we present a taxonomy of integration methods, exploring how LLMs can be utilized within the HP life cycle. Additionally, we provide a benchmark with a standardized dataset for evaluating the performance of future LLM-based HP approaches, and present initial results for a state-of-the-art HP planner and LLM planner. As expected, the latter exhibits limited performance (3\% correct plans, and none with a correct hierarchical decomposition) but serves as a valuable baseline for future approaches. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展促进了它们在多种推理相关领域的应用，包括自动规划（AP）。然而，将其集成到层次化规划（HP）中，这是AP的一个子领域，主要依赖层次化知识来提高规划性能，这一领域仍基本未被探索。在本初步研究中，我们提出了一条 roadmap，以填补这一空白，并利用LLMs在HP中的潜力。为此，我们提供了一种集成方法的税收分类，探讨了LLMs如何在HP 生命周期中被利用。此外，我们还提供了一个基准测试和标准化数据集，用于评估未来基于LLM的HP方法的性能，并展示了最先进的HP规划器和LLM规划器的初步结果。正如预期的那样，后者在性能方面表现出有限的结果（只有3%的正确计划，且没有一个具有正确层次化分解的），但作为未来方法的基准，这一结果具有重要的参考价值。 

---
# Self-Attentive Spatio-Temporal Calibration for Precise Intermediate Layer Matching in ANN-to-SNN Distillation 

**Title (ZH)**: 自注意空间-时间校准以实现ANN-to-SNN蒸馏中的精准中间层匹配 

**Authors**: Di Hong, Yueming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.08049)  

**Abstract**: Spiking Neural Networks (SNNs) are promising for low-power computation due to their event-driven mechanism but often suffer from lower accuracy compared to Artificial Neural Networks (ANNs). ANN-to-SNN knowledge distillation can improve SNN performance, but previous methods either focus solely on label information, missing valuable intermediate layer features, or use a layer-wise approach that neglects spatial and temporal semantic inconsistencies, leading to performance this http URL address these limitations, we propose a novel method called self-attentive spatio-temporal calibration (SASTC). SASTC uses self-attention to identify semantically aligned layer pairs between ANN and SNN, both spatially and temporally. This enables the autonomous transfer of relevant semantic information. Extensive experiments show that SASTC outperforms existing methods, effectively solving the mismatching problem. Superior accuracy results include 95.12% on CIFAR-10, 79.40% on CIFAR-100 with 2 time steps, and 68.69% on ImageNet with 4 time steps for static datasets, and 97.92% on DVS-Gesture and 83.60% on DVS-CIFAR10 for neuromorphic datasets. This marks the first time SNNs have outperformed ANNs on both CIFAR-10 and CIFAR-100, shedding the new light on the potential applications of SNNs. 

**Abstract (ZH)**: 脉冲神经网络（SNNs）由于其事件驱动机制，在低功耗计算中具有潜力，但通常在准确度上低于人工神经网络（ANNs）。ANN 到 SNN 的知识蒸馏可以提高 SNN 的性能，但之前的许多方法要么仅专注于标签信息，从而忽略了许多有价值的中间层特征，要么采用逐层的方法，忽略了空间和时间语义不一致，导致性能不佳。为了克服这些限制，我们提出了一种名为自注意时空校准（SASTC）的新方法。SASTC 利用自注意力机制在 ANN 和 SNN 之间识别出在空间和时间上语义对齐的层对，从而自主地转移相关的语义信息。广泛的经验研究表明，SASTC 在性能上优于现有方法，有效地解决了匹配问题。具体来说，在静态数据集上，SASTC 在 CIFAR-10 和 CIFAR-100 上分别取得了 95.12% 和 79.40% 的准确率，两个时间步长；对于 ImageNet，4 个时间步长时达到了 68.69% 的准确率。在神经形态数据集上，SASTC 在 DVS-Gesture 和 DVS-CIFAR10 上分别达到了 97.92% 和 83.60% 的准确率。这是 SNN 首次在 CIFAR-10 和 CIFAR-100 上超过 ANNs，为 SNN 的潜在应用开辟了新的前景。 

---
# Cooperative Patrol Routing: Optimizing Urban Crime Surveillance through Multi-Agent Reinforcement Learning 

**Title (ZH)**: 协作巡更路线规划：通过多agent强化学习优化城市犯罪 surveillance 

**Authors**: Juan Palma-Borda, Eduardo Guzmán, María-Victoria Belmonte  

**Link**: [PDF](https://arxiv.org/pdf/2501.08020)  

**Abstract**: The effective design of patrol strategies is a difficult and complex problem, especially in medium and large areas. The objective is to plan, in a coordinated manner, the optimal routes for a set of patrols in a given area, in order to achieve maximum coverage of the area, while also trying to minimize the number of patrols. In this paper, we propose a multi-agent reinforcement learning (MARL) model, based on a decentralized partially observable Markov decision process, to plan unpredictable patrol routes within an urban environment represented as an undirected graph. The model attempts to maximize a target function that characterizes the environment within a given time frame. Our model has been tested to optimize police patrol routes in three medium-sized districts of the city of Malaga. The aim was to maximize surveillance coverage of the most crime-prone areas, based on actual crime data in the city. To address this problem, several MARL algorithms have been studied, and among these the Value Decomposition Proximal Policy Optimization (VDPPO) algorithm exhibited the best performance. We also introduce a novel metric, the coverage index, for the evaluation of the coverage performance of the routes generated by our model. This metric is inspired by the predictive accuracy index (PAI), which is commonly used in criminology to detect hotspots. Using this metric, we have evaluated the model under various scenarios in which the number of agents (or patrols), their starting positions, and the level of information they can observe in the environment have been modified. Results show that the coordinated routes generated by our model achieve a coverage of more than $90\%$ of the $3\%$ of graph nodes with the highest crime incidence, and $65\%$ for $20\%$ of these nodes; $3\%$ and $20\%$ represent the coverage standards for police resource allocation. 

**Abstract (ZH)**: 有效的巡逻策略设计是一个复杂而困难的问题，尤其是在中大型区域。目标是在给定区域内，协调地规划一组巡逻的最佳路线，以实现对该区域的最大覆盖，同时尽量减少巡逻的数量。本文提出了一种基于去中心化部分可观测马尔可夫决策过程的多智能体强化学习（MARL）模型，用于在作为无向图表示的城市环境中规划不可预测的巡逻路线。该模型试图在给定的时间框架内最大化表征环境的目标函数。我们已将该模型应用于优化西班牙马拉加市三个中型地区的警察巡逻路线。目标是基于实际犯罪数据，最大化对犯罪高发区域的监控覆盖率。为了解决这个问题，我们研究了几种MARL算法，其中Value Decomposition Proximal Policy Optimization（VDPPO）算法表现出最佳性能。我们还引入了一个新的评估指标——覆盖指数，用于评估模型生成的路线的覆盖性能。该指标受到了预测准确性指数（PAI）的启发，PAI在犯罪学中常用于检测热点地区。

在修改了智能体（或巡逻）的数量、起始位置以及它们可观察到的环境信息水平的各种情景下，我们利用该指标对模型进行了评估。结果显示，我们模型生成的协调路线可以覆盖具有最高犯罪发生率的图节点的90%以上，以及这些节点中20%的节点的65%；3%和20%分别代表警察资源配置的覆盖标准。 

---
# GDiffRetro: Retrosynthesis Prediction with Dual Graph Enhanced Molecular Representation and Diffusion Generation 

**Title (ZH)**: GDiffRetro：基于双图增强分子表示和扩散生成的拆分预测 

**Authors**: Shengyin Sun, Wenhao Yu, Yuxiang Ren, Weitao Du, Liwei Liu, Xuecang Zhang, Ying Hu, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.08001)  

**Abstract**: Retrosynthesis prediction focuses on identifying reactants capable of synthesizing a target product. Typically, the retrosynthesis prediction involves two phases: Reaction Center Identification and Reactant Generation. However, we argue that most existing methods suffer from two limitations in the two phases: (i) Existing models do not adequately capture the ``face'' information in molecular graphs for the reaction center identification. (ii) Current approaches for the reactant generation predominantly use sequence generation in a 2D space, which lacks versatility in generating reasonable distributions for completed reactive groups and overlooks molecules' inherent 3D properties. To overcome the above limitations, we propose GDiffRetro. For the reaction center identification, GDiffRetro uniquely integrates the original graph with its corresponding dual graph to represent molecular structures, which helps guide the model to focus more on the faces in the graph. For the reactant generation, GDiffRetro employs a conditional diffusion model in 3D to further transform the obtained synthon into a complete reactant. Our experimental findings reveal that GDiffRetro outperforms state-of-the-art semi-template models across various evaluative metrics. 

**Abstract (ZH)**: 逆合成反应预测旨在识别能够合成目标产物的反应物。通常，逆合成反应预测涉及两个阶段：反应中心识别和反应物生成。然而，我们认为现有方法在这两个阶段中存在两大局限性：（i）现有的模型未能充分捕捉分子图中的“面”信息，以进行反应中心识别。（ii）当前的反应物生成方法主要在二维空间中使用序列生成方法，这种方法在生成充分反应基团的合理分布方面缺乏灵活性，并且忽略了分子的固有三维特性。为了克服上述局限性，我们提出了一种GDiffRetro方法。在反应中心识别阶段，GDiffRetro独特地将原始图与其对应的二重图相结合，以表示分子结构。这有助于引导模型更多地关注图中的面。在反应物生成阶段，GDiffRetro采用三维条件扩散模型，进一步将获得的合成子转换为完整的反应物。我们的实验结果表明，在各种评估指标上，GDiffRetro优于最先进的半模板模型。 

---
# LLM-Ehnanced Holonic Architecture for Ad-Hoc Scalable SoS 

**Title (ZH)**: 增强语言模型的自组织架构以实现即需即用的可扩展系统系（SoS） 

**Authors**: Muhammad Ashfaq, Ahmed R. Sadik, Tommi Mikkonen, Muhammad Waseem, Niko Mäkitalo  

**Link**: [PDF](https://arxiv.org/pdf/2501.07992)  

**Abstract**: As modern system of systems (SoS) become increasingly adaptive and human centred, traditional architectures often struggle to support interoperability, reconfigurability, and effective human system interaction. This paper addresses these challenges by advancing the state of the art holonic architecture for SoS, offering two main contributions to support these adaptive needs. First, we propose a layered architecture for holons, which includes reasoning, communication, and capabilities layers. This design facilitates seamless interoperability among heterogeneous constituent systems by improving data exchange and integration. Second, inspired by principles of intelligent manufacturing, we introduce specialised holons namely, supervisor, planner, task, and resource holons aimed at enhancing the adaptability and reconfigurability of SoS. These specialised holons utilise large language models within their reasoning layers to support decision making and ensure real time adaptability. We demonstrate our approach through a 3D mobility case study focused on smart city transportation, showcasing its potential for managing complex, multimodal SoS environments. Additionally, we propose evaluation methods to assess the architecture efficiency and scalability,laying the groundwork for future empirical validations through simulations and real world implementations. 

**Abstract (ZH)**: 随着现代系统体系（SoS）变得越来越适应人类需求并更加动态化，传统架构往往难以支持互操作性、可重构性和有效的人-系统交互。本文通过推进面向SoS的holonic架构的前沿状态，针对这些挑战提供了两个主要贡献来支持这些适应性需求。首先，我们提出了一种分层的holon架构，包括推理、通信和能力层。这种设计通过改进数据交换和集成来促进异构组成部分系统的无缝互操作性。其次，受智能制造原则的启发，我们引入了特定类型的holon，包括监督者、规划者、任务和资源holon，旨在增强SoS的适应性和可重构性。这些特定的holon在其推理层中利用大规模语言模型来支持决策制定并确保实时适应。我们通过一个3D移动性案例研究，专注于智能城市交通，展示了其在管理复杂多模态SoS环境方面的潜力。此外，我们还提出了评估架构效率和可扩展性的方法，为未来的实证验证奠定了基础，这些验证将通过仿真实验和实际应用来实现。 

---
# Comprehensive Metapath-based Heterogeneous Graph Transformer for Gene-Disease Association Prediction 

**Title (ZH)**: 基于综合元路径的异质图变换器在基因-疾病关联预测中的应用 

**Authors**: Wentao Cui, Shoubo Li, Chen Fang, Qingqing Long, Chengrui Wang, Xuezhi Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.07970)  

**Abstract**: Discovering gene-disease associations is crucial for understanding disease mechanisms, yet identifying these associations remains challenging due to the time and cost of biological experiments. Computational methods are increasingly vital for efficient and scalable gene-disease association prediction. Graph-based learning models, which leverage node features and network relationships, are commonly employed for biomolecular predictions. However, existing methods often struggle to effectively integrate node features, heterogeneous structures, and semantic information. To address these challenges, we propose COmprehensive MEtapath-based heterogeneous graph Transformer(COMET) for predicting gene-disease associations. COMET integrates diverse datasets to construct comprehensive heterogeneous networks, initializing node features with BioGPT. We define seven Metapaths and utilize a transformer framework to aggregate Metapath instances, capturing global contexts and long-distance dependencies. Through intra- and inter-metapath aggregation using attention mechanisms, COMET fuses latent vectors from multiple Metapaths to enhance GDA prediction accuracy. Our method demonstrates superior robustness compared to state-of-the-art approaches. Ablation studies and visualizations validate COMET's effectiveness, providing valuable insights for advancing human health research. 

**Abstract (ZH)**: 发现基因-疾病关联对于理解疾病机制至关重要，但识别这些关联仍然具有挑战性，因为生物实验耗时且成本高昂。计算方法对于高效且可扩展的基因-疾病关联预测变得越来越重要。基于图的学习模型通过利用节点特征和网络关系，广泛应用于生物分子预测中。然而，现有方法往往难以有效整合节点特征、异质结构和语义信息。为了解决这些问题，我们提出了COmprehensive MEtapath-based 混合图变换器（COMET）来预测基因-疾病关联。COMET 将多种数据集整合以构建全面的异质网络，并使用BioGPT 初始化节点特征。我们定义了七种元路径，并采用变换器框架聚合这些元路径实例，捕捉全局上下文和长距离依赖关系。通过使用注意机制进行元路径内的和跨元路径聚合，COMET 汇总多个元路径的潜在向量以提高基因-疾病关联预测的准确性。我们的方法在与现有最先进的方法相比时表现出了更高的鲁棒性。消融研究和可视化验证了COMET 的有效性，为其在推动人类健康研究方面提供了宝贵见解。 

---
# Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning 

**Title (ZH)**: 自我指导少样本监狱突破：将攻击分解为模式学习和行为学习 

**Authors**: Jiaqi Hua, Wanxu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.07959)  

**Abstract**: Recently, several works have been conducted on jailbreaking Large Language Models (LLMs) with few-shot malicious demos. In particular, Zheng et al. (2024) focuses on improving the efficiency of Few-Shot Jailbreaking (FSJ) by injecting special tokens into the demos and employing demo-level random search. Nevertheless, this method lacks generality since it specifies the instruction-response structure. Moreover, the reason why inserting special tokens takes effect in inducing harmful behaviors is only empirically discussed. In this paper, we take a deeper insight into the mechanism of special token injection and propose Self-Instruct Few-Shot Jailbreaking (Self-Instruct-FSJ) facilitated with the demo-level greedy search. This framework decomposes the FSJ attack into pattern and behavior learning to exploit the model's vulnerabilities in a more generalized and efficient way. We conduct elaborate experiments to evaluate our method on common open-source models and compare it with baseline algorithms. Our code is available at this https URL. 

**Abstract (ZH)**: 近年来，已有若干研究工作探讨了通过少量恶意示范对大规模语言模型（LLMs）进行越狱的方法。郑等人的研究（2024）则侧重于通过向示范中注入特殊标记并采用示范级别随机搜索来提高少量示范越狱（Few-Shot Jailbreaking, FSJ）的效率。然而，这种方法缺乏普适性，因为它规定了指令-响应结构。此外，将特殊标记插入示范以引发有害行为的原因仅从经验角度进行了讨论。本文深入探讨了特殊标记注入机制，并提出了一种基于示范级别贪婪搜索的自我指令少量示范越狱（Self-Instruct-FSJ）框架。该框架将FSJ攻击分解为模式学习和行为学习，以更通用和高效的方式利用模型的漏洞。我们进行了详尽的实验，评估该方法在常见开源模型上的效果，并与基础算法进行了比较。我们的代码可在以下链接访问：[请填写具体的URL]。 

---
# Advice for Diabetes Self-Management by ChatGPT Models: Challenges and Recommendations 

**Title (ZH)**: 由ChatGPT模型提供的糖尿病自我管理建议：挑战与建议 

**Authors**: Waqar Hussain, John Grundy  

**Link**: [PDF](https://arxiv.org/pdf/2501.07931)  

**Abstract**: Given their ability for advanced reasoning, extensive contextual understanding, and robust question-answering abilities, large language models have become prominent in healthcare management research. Despite adeptly handling a broad spectrum of healthcare inquiries, these models face significant challenges in delivering accurate and practical advice for chronic conditions such as diabetes. We evaluate the responses of ChatGPT versions 3.5 and 4 to diabetes patient queries, assessing their depth of medical knowledge and their capacity to deliver personalized, context-specific advice for diabetes self-management. Our findings reveal discrepancies in accuracy and embedded biases, emphasizing the models' limitations in providing tailored advice unless activated by sophisticated prompting techniques. Additionally, we observe that both models often provide advice without seeking necessary clarification, a practice that can result in potentially dangerous advice. This underscores the limited practical effectiveness of these models without human oversight in clinical settings. To address these issues, we propose a commonsense evaluation layer for prompt evaluation and incorporating disease-specific external memory using an advanced Retrieval Augmented Generation technique. This approach aims to improve information quality and reduce misinformation risks, contributing to more reliable AI applications in healthcare settings. Our findings seek to influence the future direction of AI in healthcare, enhancing both the scope and quality of its integration. 

**Abstract (ZH)**: 鉴于其高级推理能力、广泛的上下文理解和强大的问答能力，大型语言模型在医疗管理研究中已成为一个突出的研究领域。尽管这些模型能够处理广泛的医疗保健查询，但在提供有关糖尿病等慢性疾病的准确和实用建议方面仍面临重大挑战。我们评估了ChatGPT 3.5和4版对糖尿病患者查询的回答，评定它们的医学知识深度及其为糖尿病自我管理提供个性化、情境相关建议的能力。研究发现，这些模型在准确性方面存在差异，并且存在嵌入的偏见，除非通过高级提示技术激活，否则无法提供定制化的建议。此外，我们发现这两种模型往往在提供建议时不进行必要的澄清，这种做法可能导致潜在的危险建议。这强调了在临床环境中缺乏人类监督时，这些模型的有限实际效果。为解决这些问题，我们提出了一种常识性评估层，用于提示评估，并结合使用先进的检索增强生成技术，引入特定疾病的外部记忆。该方法旨在提高信息质量并降低 misinformation 的风险，从而促进更可靠的医疗保健 AI 应用。我们的研究旨在影响人工智能在医疗保健领域的未来发展方向，增强其融合范围和质量。

注：这里的“misinformation”在上下文中可以理解为“错误信息”或“误导性信息”，根据具体语境可以选择合适的翻译。此外，“commonsense evaluation layer”可以理解为一种常识性评价层，用于评估提示的质量，确保生成的内容更加合理和准确，有助于提高 AI 应用在医疗保健领域的可靠性。 

---
# An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures 

**Title (ZH)**: 一种用于高效灵活卷积神经网络架构的自适应正交卷积方案 

**Authors**: Thibaut Boissin, Franck Mamalet, Thomas Fel, Agustin Martin Picard, Thomas Massena, Mathieu Serrurier  

**Link**: [PDF](https://arxiv.org/pdf/2501.07930)  

**Abstract**: Orthogonal convolutional layers are the workhorse of multiple areas in machine learning, such as adversarial robustness, normalizing flows, GANs, and Lipschitzconstrained models. Their ability to preserve norms and ensure stable gradient propagation makes them valuable for a large range of problems. Despite their promise, the deployment of orthogonal convolution in large-scale applications is a significant challenge due to computational overhead and limited support for modern features like strides, dilations, group convolutions, and transposed this http URL this paper, we introduce AOC (Adaptative Orthogonal Convolution), a scalable method for constructing orthogonal convolutions, effectively overcoming these limitations. This advancement unlocks the construction of architectures that were previously considered impractical. We demonstrate through our experiments that our method produces expressive models that become increasingly efficient as they scale. To foster further advancement, we provide an open-source library implementing this method, available at this https URL. 

**Abstract (ZH)**: 正交卷积层在机器学习的多个领域中是不可或缺的组件，包括对抗性鲁棒性、规范流、生成对抗网络（GANs）、以及Lipschitz约束模型。它们能够保持范数和确保梯度传播的稳定性，因此在广泛的问题中具有很高的价值。尽管具有这些优势，正交卷积在大规模应用中的部署仍面临重大挑战，主要是因为计算开销和对现代特性（如步长、扩张、分组卷积和转置卷积）的支持不足。在本文中，我们引入了一种可扩展的方法，即AOC（适应性正交卷积），有效克服了这些限制。这一进步为企业级架构的构建打开了新的可能性。我们通过实验展示了我们的方法能够生成表达力强的模型，并且随着模型规模的增加而变得更加高效。为了进一步推动这一领域的研究，我们开源了实现该方法的库，可在以下链接获取：<https://github.com/your-repo-url>。 

---
# Exploring Aviation Incident Narratives Using Topic Modeling and Clustering Techniques 

**Title (ZH)**: 使用主题建模和聚类技术探索航空事故叙事 

**Authors**: Aziida Nanyonga, Hassan Wasswa, Ugur Turhan, Keith Joiner, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2501.07924)  

**Abstract**: Aviation safety is a global concern, requiring detailed investigations into incidents to understand contributing factors comprehensively. This study uses the National Transportation Safety Board (NTSB) dataset. It applies advanced natural language processing (NLP) techniques, including Latent Dirichlet Allocation (LDA), Non-Negative Matrix Factorization (NMF), Latent Semantic Analysis (LSA), Probabilistic Latent Semantic Analysis (pLSA), and K-means clustering. The main objectives are identifying latent themes, exploring semantic relationships, assessing probabilistic connections, and cluster incidents based on shared characteristics. This research contributes to aviation safety by providing insights into incident narratives and demonstrating the versatility of NLP and topic modelling techniques in extracting valuable information from complex datasets. The results, including topics identified from various techniques, provide an understanding of recurring themes. Comparative analysis reveals that LDA performed best with a coherence value of 0.597, pLSA of 0.583, LSA of 0.542, and NMF of 0.437. K-means clustering further reveals commonalities and unique insights into incident narratives. In conclusion, this study uncovers latent patterns and thematic structures within incident narratives, offering a comparative analysis of multiple-topic modelling techniques. Future research avenues include exploring temporal patterns, incorporating additional datasets, and developing predictive models for early identification of safety issues. This research lays the groundwork for enhancing the understanding and improvement of aviation safety by utilising the wealth of information embedded in incident narratives. 

**Abstract (ZH)**: 航空安全是全球性的关注问题，需要对事故进行全面深入的研究以理解其相关因素。本研究使用了美国国家运输安全委员会（NTSB）的数据集，并应用了先进的自然语言处理（NLP）技术，包括潜在狄利克雷分配（LDA）、非负矩阵分解（NMF）、潜在语义分析（LSA）、概率潜在语义分析（pLSA）和K-均值聚类。主要目标是识别潜在主题、探索语义关系、评估概率联系、并基于共享特征聚类事故。本研究通过提供事故叙述的见解，展示了NLP和主题建模技术在从复杂数据集中提取有价值信息方面的多样性和灵活性，从而为航空安全做出贡献。研究结果包括从各种技术中识别出的主题，这些结果揭示了反复出现的主题。比较分析结果显示，LDA的共现值为0.597，pLSA为0.583，LSA为0.542，NMF为0.437。K-均值聚类进一步揭示了事故叙述中的共性及其独有的见解。总结而言，本研究揭示了事故叙述中的潜在模式和主题结构，并进行了多种主题建模技术的比较分析。未来的研究方向包括探索时间模式、纳入其他数据集以及开发早期识别安全问题的预测模型。本研究为利用事故叙述中蕴含的丰富信息以增强对航空安全的理解和改进奠定了基础。 

---
# Large Language Model Interface for Home Energy Management Systems 

**Title (ZH)**: 面向家庭能源管理系统的大语言模型接口 

**Authors**: François Michelon, Yihong Zhou, Thomas Morstyn  

**Link**: [PDF](https://arxiv.org/pdf/2501.07919)  

**Abstract**: Home Energy Management Systems (HEMSs) help households tailor their electricity usage based on power system signals such as energy prices. This technology helps to reduce energy bills and offers greater demand-side flexibility that supports the power system stability. However, residents who lack a technical background may find it difficult to use HEMSs effectively, because HEMSs require well-formatted parameterization that reflects the characteristics of the energy resources, houses, and users' needs. Recently, Large-Language Models (LLMs) have demonstrated an outstanding ability in language understanding. Motivated by this, we propose an LLM-based interface that interacts with users to understand and parameterize their ``badly-formatted answers'', and then outputs well-formatted parameters to implement an HEMS. We further use Reason and Act method (ReAct) and few-shot prompting to enhance the LLM performance. Evaluating the interface performance requires multiple user--LLM interactions. To avoid the efforts in finding volunteer users and reduce the evaluation time, we additionally propose a method that uses another LLM to simulate users with varying expertise, ranging from knowledgeable to non-technical. By comprehensive evaluation, the proposed LLM-based HEMS interface achieves an average parameter retrieval accuracy of 88\%, outperforming benchmark models without ReAct and/or few-shot prompting. 

**Abstract (ZH)**: 家庭能源管理系统（HEMSs）帮助家庭根据电力系统信号（如电价）来调整其用电行为。这项技术有助于降低能源账单，并提供更高的需求侧灵活性，从而支持电力系统的稳定性。然而，缺乏技术背景的居民可能难以有效使用HEMSs，因为HEMSs需要格式良好的参数化设置，反映出能源资源、房屋和用户需求的特点。最近，大语言模型（LLMs）在语言理解方面展现了卓越的能力。受此启发，我们提出了一种基于LLM的界面，该界面能够与用户互动，理解并参数化用户的“格式不良的答案”，然后输出格式良好的参数以实现HEMS。为进一步增强LLM的性能，我们采用了Reason and Act方法（ReAct）和少量示例提示。评估界面性能需要多次用户-LLM交互。为了避免寻找志愿者用户和减少评估时间，我们还提出了一种方法，使用另一种LLM来模拟具有不同专业知识水平的用户，从熟练的到非技术背景的用户。通过综合评估，所提出的基于LLM的HEMS界面实现了参数检索平均准确率88%，优于没有使用ReAct和/或少量示例提示的标准模型。 

---
# Governing AI Agents 

**Title (ZH)**: 治理人工智能代理 

**Authors**: Noam Kolt  

**Link**: [PDF](https://arxiv.org/pdf/2501.07913)  

**Abstract**: The field of AI is undergoing a fundamental transition from systems that can produce synthetic content upon request to autonomous agents that can plan and execute complex tasks with only limited human involvement. Companies that pioneered the development of generative AI tools are now building AI agents that can be instructed to independently navigate the internet, perform a wide range of online tasks, and serve as artificial personal assistants and virtual coworkers. The opportunities presented by this new technology are tremendous, as are the associated risks. Fortunately, there exist robust analytic frameworks for confronting many of these challenges, namely, the economic theory of principal-agent problems and the common law doctrine of agency relationships. Drawing on these frameworks, this Article makes three contributions. First, it uses agency law and theory to identify and characterize problems arising from AI agents, including issues of information asymmetry, discretionary authority, and loyalty. Second, it illustrates the limitations of conventional solutions to agency problems: incentive design, monitoring, and enforcement might not be effective for governing AI agents that make uninterpretable decisions and operate at unprecedented speed and scale. Third, the Article explores the implications of agency law and theory for designing and regulating AI agents, arguing that new technical and legal infrastructure is needed to support governance principles of inclusivity, visibility, and liability. 

**Abstract (ZH)**: 人工智能领域正在经历一场根本性的转变，从能够应要求生成合成内容的系统，转向能够自主规划和执行复杂任务的代理，且只需有限的人类干预。率先开发生成式人工智能工具的公司现在正在构建可以独立导航互联网、执行多种在线任务、充当人工个人助手和虚拟同事的AI代理。这种新技术带来的机遇巨大，相应的风险也同样显著。幸运的是，存在适用于许多这些挑战的稳健分析框架，即主要与代理问题的经济理论和代理关系的普通法原则。依据这些框架，本文做出了三个贡献。首先，它利用代理法和理论来识别并描述AI代理所引起的问题，包括信息不对称、酌情权力和忠诚度等方面的问题。其次，它展示了传统代理问题解决方案的局限性：激励设计、监控和执行措施可能无法有效治理作出不可解释决策且以前所未有的速度和规模运作的AI代理。最后，本文探讨了代理法和理论对设计和监管AI代理的意义，论述了需要新的技术和法律基础设施来支持包容性、可见性和问责性等治理原则。 

---
# Deep Learning and Natural Language Processing in the Field of Construction 

**Title (ZH)**: 在建筑领域中深度学习与自然语言处理的研究 

**Authors**: Rémy Kessler, Nicolas Béchet  

**Link**: [PDF](https://arxiv.org/pdf/2501.07911)  

**Abstract**: This article presents a complete process to extract hypernym relationships in the field of construction using two main steps: terminology extraction and detection of hypernyms from these terms. We first describe the corpus analysis method to extract terminology from a collection of technical specifications in the field of construction. Using statistics and word n-grams analysis, we extract the domain's terminology and then perform pruning steps with linguistic patterns and internet queries to improve the quality of the final terminology. Second, we present a machine-learning approach based on various words embedding models and combinations to deal with the detection of hypernyms from the extracted terminology. Extracted terminology is evaluated using a manual evaluation carried out by 6 experts in the domain, and the hypernym identification method is evaluated with different datasets. The global approach provides relevant and promising results. 

**Abstract (ZH)**: 本文介绍了在建筑领域提取超词关系的完整过程，主要分为两个步骤：术语提取和超词检测。首先，我们描述了一种基于语料库分析的方法，用于从建筑领域的一系列技术规范中提取术语。通过统计分析和词n元组分析，我们提取了该领域的术语，然后通过语言模式和网络查询进行修剪，以提高最终术语的质量。其次，我们提出了一种基于多种词嵌入模型及其组合的机器学习方法，用于从提取的术语中检测超词。提取的术语通过6位领域专家的手工评估进行了评价，超词识别方法则通过不同的数据集进行了评估。整体方法提供了相关且有前景的结果。 

---
# Logarithmic Memory Networks (LMNs): Efficient Long-Range Sequence Modeling for Resource-Constrained Environments 

**Title (ZH)**: 对数记忆网络（LMNs）：资源受限环境中高效的长程序列建模 

**Authors**: Mohamed A. Taha  

**Link**: [PDF](https://arxiv.org/pdf/2501.07905)  

**Abstract**: Long-range sequence modeling is a crucial aspect of natural language processing and time series analysis. However, traditional models like Recurrent Neural Networks (RNNs) and Transformers suffer from computational and memory inefficiencies, especially when dealing with long sequences. This paper introduces Logarithmic Memory Networks (LMNs), a novel architecture that leverages a hierarchical logarithmic tree structure to efficiently store and retrieve past information. LMNs dynamically summarize historical context, significantly reducing the memory footprint and computational complexity of attention mechanisms from O(n2) to O(log(n)). The model employs a single-vector, targeted attention mechanism to access stored information, and the memory block construction worker (summarizer) layer operates in two modes: a parallel execution mode during training for efficient processing of hierarchical tree structures and a sequential execution mode during inference, which acts as a memory management system. It also implicitly encodes positional information, eliminating the need for explicit positional encodings. These features make LMNs a robust and scalable solution for processing long-range sequences in resource-constrained environments, offering practical improvements in efficiency and scalability. The code is publicly available under the MIT License on GitHub: this https URL. 

**Abstract (ZH)**: 长程序列建模是自然语言处理和时间序列分析中的关键方面。然而，传统的模型如循环神经网络（RNNs）和Transformer在处理长序列时容易出现计算和内存效率低下问题。本文引入了对数记忆网络（Logarithmic Memory Networks, LMNs），这是一种新型架构，利用层次对数树结构高效地存储和检索过去的信息。LMNs 动态地总结历史上下文，显著减少了注意力机制的记忆占用和计算复杂度，从 \(O(n^2)\) 降低到 \(O(\log(n))\)。该模型采用单向量、针对性的注意力机制来访问存储的信息，记忆块构建工人（总结器）层在训练时采用并行执行模式以高效处理层次树结构，在推理时则采用串行执行模式，充当内存管理系统。LMNs 还隐式地编码位置信息，消除了对显式位置编码的需求。这些特性使LMNs 成为在资源受限环境中处理长程序列的稳健和可扩展解决方案，提供了实际的效率和可扩展性改进。代码已根据MIT许可协议在GitHub上公开：[此链接](https://github.com/your-repo-url)。 

---
# Anytime Cooperative Implicit Hitting Set Solving 

**Title (ZH)**: 任何时间协作隐击集求解 

**Authors**: Emma Rollón, Javier Larrosa, Aleksandra Petrova  

**Link**: [PDF](https://arxiv.org/pdf/2501.07896)  

**Abstract**: The Implicit Hitting Set (HS) approach has shown to be very effective for MaxSAT, Pseudo-boolean optimization and other boolean frameworks. Very recently, it has also shown its potential in the very similar Weighted CSP framework by means of the so-called cost-function merging. The original formulation of the HS approach focuses on obtaining increasingly better lower bounds (HS-lb). However, and as shown for Pseudo-Boolean Optimization, this approach can also be adapted to compute increasingly better upper bounds (HS-ub). In this paper we consider both HS approaches and show how they can be easily combined in a multithread architecture where cores discovered by either component are available by the other which, interestingly, generates synergy between them. We show that the resulting algorithm (HS-lub) is consistently superior to either HS-lb and HS-ub in isolation. Most importantly, HS-lub has an effective anytime behaviour with which the optimality gap is reduced during the execution. We tested our approach on the Weighted CSP framework and show on three different benchmarks that our very simple implementation sometimes outperforms the parallel hybrid best-first search implementation of the far more developed state-of-the-art Toulbar2. 

**Abstract (ZH)**: 隐含打击集（HS）方法已被证明对MaxSAT、伪布尔优化以及其它布尔框架非常有效。最近，它还通过所谓的费用函数合并方法在非常相似的加权约束 satisfaction 问题（Weighted CSP）框架中显示出了其潜力。原始的HS方法主要关注于获得越来越好的下界（HS-lb）。然而，正如在伪布尔优化中所展示的，这种方法也可以适配来计算越来越好的上界（HS-ub）。在本文中，我们考虑了这两种HS方法，并展示了它们如何轻松地结合在一个多线程架构中，其中一个组件发现的核心可以由另一个组件访问，显著产生了协同效应。我们证明，所得到的算法（HS-lub）在单独使用HS-lb和HS-ub时表现始终更优。更重要的是，HS-lub 具有有效的任意时间行为，在执行过程中减少了最优性差距。我们在加权约束满足问题（Weighted CSP）框架中测试了我们的方法，并在三个不同的基准上展示了我们非常简单的实现有时超越了更为先进的Toulbar2的并行混合最佳优先搜索实现。 

---
# A Driver Advisory System Based on Large Language Model for High-speed Train 

**Title (ZH)**: 基于大型语言模型的高速列车驾驶辅助系统 

**Authors**: Y.C. Luo, J. Xun, W. Wang, R.Z. Zhang, Z.C. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07837)  

**Abstract**: With the rapid development of China high-speed railway, drivers face increasingly significant technical challenges during operations, such as fault handling. Currently, drivers depend on the onboard mechanic when facing technical issues, for instance, traction loss or sensor faults. This dependency can hinder effective operation, even lead to accidents, while waiting for faults to be addressed. To enhance the accuracy and explainability of actions during fault handling, an Intelligent Driver Advisory System (IDAS) framework based on a large language model (LLM) named IDAS-LLM, is introduced. Initially, domain-fine-tuning of the LLM is performed using a constructed railway knowledge question-and-answer dataset to improve answer accuracy in railway-related questions. Subsequently, integration of the Retrieval-augmented Generation (RAG) architecture is pursued for system design to enhance the explainability of generated responses. Comparative experiments are conducted using the constructed railway driving knowledge assessment dataset. Results indicate that domain-fine-tuned LLMs show an improvement in answer accuracy by an average of 10%, outperforming some current mainstream LLMs. Additionally, the inclusion of the RAG framework increases the average recall rate of question-and-answer sessions by about 4%. Finally, the fault handling capability of IDAS-LLM is demonstrated through simulations of real operational scenarios, proving that the proposed framework has practical application prospects. 

**Abstract (ZH)**: 随着中国高速铁路的快速发展，列车司机在运营过程中面临的技術挑战越来越显著，如故障处理等。目前，面对技术问题时，司机依赖车辆机械师，例如牵引力损失或传感器故障。这种依赖在故障处理过程中可能导致操作效率低下，甚至可能引发事故。为了提高故障处理过程中操作的准确性和可解释性，本文提出了一个基于大型语言模型（LLM）的智能司机辅助系统（IDAS）框架，该框架命名为IDAS-LLM。首先，通过使用构建的铁路知识问答数据集对LLM进行领域微调，以提高铁路相关问题回答的准确性。随后，通过结合检索增强生成（RAG）架构来优化系统设计，以增强生成响应的可解释性。通过使用构建的铁路驾驶知识评估数据集进行对比实验。结果表明，领域微调后的LLM在回答准确性方面平均提高了10%，超过了当前一些主流LLM。此外，引入RAG框架使得问答会话的平均召回率提高了约4%。最后，通过模拟实际运营场景展示了IDAS-LLM的故障处理能力，证明该提出的框架具有实际应用的前景。 

---
# Flow: A Modular Approach to Automated Agentic Workflow Generation 

**Title (ZH)**: Flow：模块化自动代理工作流生成方法 

**Authors**: Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07834)  

**Abstract**: Multi-agent frameworks powered by large language models (LLMs) have demonstrated great success in automated planning and task execution. However, the effective adjustment of Agentic workflows during execution has not been well-studied. A effective workflow adjustment is crucial, as in many real-world scenarios, the initial plan must adjust to unforeseen challenges and changing conditions in real-time to ensure the efficient execution of complex tasks. In this paper, we define workflows as an activity-on-vertex (AOV) graphs. We continuously refine the workflow by dynamically adjusting task allocations based on historical performance and previous AOV with LLM agents. To further enhance system performance, we emphasize modularity in workflow design based on measuring parallelism and dependence complexity. Our proposed multi-agent framework achieved efficient sub-task concurrent execution, goal achievement, and error tolerance. Empirical results across different practical tasks demonstrate dramatic improvements in the efficiency of multi-agent frameworks through dynamic workflow updating and modularization. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的多智能体框架已经在自动化规划和任务执行方面展示了巨大的成功。然而，在执行过程中智能体工作流的有效调整尚未得到充分研究。有效的流程调整至关重要，因为在许多实际场景中，初始计划需要根据实时出现的挑战和变化条件进行调整，以确保复杂任务的高效执行。在本文中，我们将工作流定义为顶点活动图（AOV图）。我们通过基于历史表现和先前的AOV图动态调整任务分配来持续优化工作流。为了进一步提升系统性能，我们基于衡量并行性和依赖复杂性强调工作流设计的模块化。我们提出的多智能体框架实现了子任务的高效并行执行、目标实现及容错能力。在不同实际任务上的实验结果表明，通过动态更新和模块化工作流，多智能体框架的效率得到了显著提高。 

---
# Agent-Centric Projection of Prompting Techniques and Implications for Synthetic Training Data for Large Language Models 

**Title (ZH)**: 基于代理的提示技术投影及其对大型语言模型合成训练数据的影响 

**Authors**: Dhruv Dhamani, Mary Lou Maher  

**Link**: [PDF](https://arxiv.org/pdf/2501.07815)  

**Abstract**: Recent advances in prompting techniques and multi-agent systems for Large Language Models (LLMs) have produced increasingly complex approaches. However, we lack a framework for characterizing and comparing prompting techniques or understanding their relationship to multi-agent LLM systems. This position paper introduces and explains the concepts of linear contexts (a single, continuous sequence of interactions) and non-linear contexts (branching or multi-path) in LLM systems. These concepts enable the development of an agent-centric projection of prompting techniques, a framework that can reveal deep connections between prompting strategies and multi-agent systems. We propose three conjectures based on this framework: (1) results from non-linear prompting techniques can predict outcomes in equivalent multi-agent systems, (2) multi-agent system architectures can be replicated through single-LLM prompting techniques that simulate equivalent interaction patterns, and (3) these equivalences suggest novel approaches for generating synthetic training data. We argue that this perspective enables systematic cross-pollination of research findings between prompting and multi-agent domains, while providing new directions for improving both the design and training of future LLM systems. 

**Abstract (ZH)**: 近年来，在大型语言模型（LLMs）中，提示技术与多智能体系统的进展产生了日益复杂的方法。然而，我们缺乏一种框架来表征和比较提示技术，或理解它们与多智能体LLMs系统之间的关系。本文介绍了LLM系统中线性上下文（单一连续的交互序列）和非线性上下文（分支或多重路径）的概念。这些概念使我们能够从代理中心的角度对提示技术进行投影，形成一个框架，该框架可以揭示提示策略与多智能体系统之间的深层次联系。基于该框架，我们提出了三个假设：（1）非线性提示技术的结果可以预测等效多智能体系统的结果；（2）多智能体系统的架构可以通过模拟等效交互模式的单LLM提示技术来复制；（3）这些等效性暗示了生成合成训练数据的新方法。我们认为，这种视角促进了提示和多智能体研究领域之间系统的跨学科交流，同时为改进未来LLM系统的架构和训练提供了新的方向。 

---
# A Low-cost and Ultra-lightweight Binary Neural Network for Traffic Signal Recognition 

**Title (ZH)**: 一种低成本且超轻量的二值神经网络用于交通信号识别 

**Authors**: Mingke Xiao, Yue Su, Liang Yu, Guanglong Qu, Yutong Jia, Yukuan Chang, Xu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07808)  

**Abstract**: The deployment of neural networks in vehicle platforms and wearable Artificial Intelligence-of-Things (AIOT) scenarios has become a research area that has attracted much attention. With the continuous evolution of deep learning technology, many image classification models are committed to improving recognition accuracy, but this is often accompanied by problems such as large model resource usage, complex structure, and high power consumption, which makes it challenging to deploy on resource-constrained platforms. Herein, we propose an ultra-lightweight binary neural network (BNN) model designed for hardware deployment, and conduct image classification research based on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. In addition, we also verify it on the Chinese Traffic Sign (CTS) and Belgian Traffic Sign (BTS) datasets. The proposed model shows excellent recognition performance with an accuracy of up to 97.64%, making it one of the best performing BNN models in the GTSRB dataset. Compared with the full-precision model, the accuracy loss is controlled within 1%, and the parameter storage overhead of the model is only 10% of that of the full-precision model. More importantly, our network model only relies on logical operations and low-bit width fixed-point addition and subtraction operations during the inference phase, which greatly simplifies the design complexity of the processing element (PE). Our research shows the great potential of BNN in the hardware deployment of computer vision models, especially in the field of computer vision tasks related to autonomous driving. 

**Abstract (ZH)**: 在车辆平台和可穿戴人工智能物联网（AIOT）场景中部署神经网络已成为一个备受关注的研究领域。随着深度学习技术的不断进步，许多图像分类模型致力于提高识别准确性，但这也常常伴随着模型资源占用大、结构复杂和高功耗等问题，这使得在资源受限平台上部署变得极具挑战性。在此基础上，我们提出了一种专为硬件部署设计的超轻量级二值神经网络（BNN）模型，并基于德国交通标志识别基准数据集（GTSRB）进行图像分类研究。此外，我们还在中国交通标志（CTS）和比利时交通标志（BTS）数据集上进行了验证。所提出的模型显示出了出色的表现，准确率达到97.64%，使其成为GTSRB数据集中性能最好的BNN模型之一。与全精度模型相比，准确率损失控制在1%以内，模型参数存储开销仅为全精度模型的10%。更重要的是，在推理阶段，我们的网络模型仅依赖于逻辑操作和低位宽固定点加减法操作，这极大地简化了处理单元（PE）的设计复杂性。我们的研究表明，BNN在计算机视觉模型的硬件部署中有巨大的潜力，尤其适用于与自动驾驶相关的计算机视觉任务。 

---
# Visual Language Models as Operator Agents in the Space Domain 

**Title (ZH)**: 视觉语言模型在空间领域中的操作代理应用 

**Authors**: Alejandro Carrasco, Marco Nedungadi, Enrico M. Zucchelli, Amit Jain, Victor Rodriguez-Fernandez, Richard Linares  

**Link**: [PDF](https://arxiv.org/pdf/2501.07802)  

**Abstract**: This paper explores the application of Vision-Language Models (VLMs) as operator agents in the space domain, focusing on both software and hardware operational paradigms. Building on advances in Large Language Models (LLMs) and their multimodal extensions, we investigate how VLMs can enhance autonomous control and decision-making in space missions. In the software context, we employ VLMs within the Kerbal Space Program Differential Games (KSPDG) simulation environment, enabling the agent to interpret visual screenshots of the graphical user interface to perform complex orbital maneuvers. In the hardware context, we integrate VLMs with robotic systems equipped with cameras to inspect and diagnose physical space objects, such as satellites. Our results demonstrate that VLMs can effectively process visual and textual data to generate contextually appropriate actions, competing with traditional methods and non-multimodal LLMs in simulation tasks, and showing promise in real-world applications. 

**Abstract (ZH)**: 本文探讨了视觉语言模型（VLMs）作为操作代理在空间领域的应用，重点关注软件和硬件操作范式。基于大型语言模型（LLMs）及其多模态扩展的最新进展，我们研究了VLMs如何在空间任务中增强自主控制和决策能力。在软件背景下，我们利用VLMs在Kerbal Space Program Differential Games（KSPDG）仿真环境中，让代理能够解释图形用户界面的视觉截图，从而执行复杂的轨道机动。在硬件背景下，我们将VLMs整合到配备摄像头的机器人系统中，以检查和诊断物理空间物体，如卫星。研究结果表明，VLMs能够有效处理视觉和文本数据以生成上下文相关的行为，在仿真任务中可以与传统方法和非多模态的LLMs竞争，显示出在实际应用中的潜力。 

---
# Rethinking AI Cultural Evaluation 

**Title (ZH)**: 重新审视AI文化评估 

**Authors**: Michal Bravansky, Filip Trhlik, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2501.07751)  

**Abstract**: As AI systems become more integrated into society, evaluating their capacity to align with diverse cultural values is crucial for their responsible deployment. Current evaluation methods predominantly rely on multiple-choice question (MCQ) datasets. In this study, we demonstrate that MCQs are insufficient for capturing the complexity of cultural values expressed in open-ended scenarios. Our findings highlight significant discrepancies between MCQ-based assessments and the values conveyed in unconstrained interactions. Based on these findings, we recommend moving beyond MCQs to adopt more open-ended, context-specific assessments that better reflect how AI models engage with cultural values in realistic settings. 

**Abstract (ZH)**: 随着人工智能系统在社会中的深度融合，评估其与多元文化价值观的契合能力对于负责任地部署这些系统至关重要。当前的评估方法主要依赖于多项选择题（MCQ）数据集。在本研究中，我们展示了MCQ在捕捉开放场景中表达的文化价值观复杂性方面的不足。我们的研究结果强调了基于MCQ的评估与非限制性互动中传达的价值观之间显著的差异。基于这些发现，我们建议超越MCQ，采用更开放、情境特定的评估方法，更好地反映AI模型在真实场景中与文化价值观互动的方式。 

---
# CDS: Data Synthesis Method Guided by Cognitive Diagnosis Theory 

**Title (ZH)**: CDS：基于认知诊断理论的数据合成方法 

**Authors**: Haokun Zhao, Jinyi Han, Jiaqing Liang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07674)  

**Abstract**: Large Language Models (LLMs) have demonstrated outstanding capabilities across various domains, but the increasing complexity of new challenges demands enhanced performance and adaptability. Traditional benchmarks, although comprehensive, often lack the granularity needed for detailed capability analysis. This study introduces the Cognitive Diagnostic Synthesis (CDS) method, which employs Cognitive Diagnosis Theory (CDT) for precise evaluation and targeted enhancement of LLMs. By decomposing complex tasks into discrete knowledge points, CDS accurately identifies and synthesizes data targeting model weaknesses, thereby enhancing the model's performance. This framework proposes a comprehensive pipeline driven by knowledge point evaluation, synthesis, data augmentation, and filtering, which significantly improves the model's mathematical and coding capabilities, achieving up to an 11.12% improvement in optimal scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域展现了出色的性能，但日益复杂的新挑战需要提高其性能和适应性。尽管传统的基准测试全面，但往往缺乏进行详细能力分析所需的精细度。本研究引入了认知诊断合成（Cognitive Diagnostic Synthesis, CDS）方法，该方法利用认知诊断理论（Cognitive Diagnosis Theory, CDT）对LLMs进行精确评估和针对性改进。通过将复杂任务分解为离散的知识点，CDS能够准确地识别并综合目标数据，以弥补模型的弱点，从而提高模型的性能。该框架提出了一种由知识点评估、合成、数据增强和筛选驱动的全面管道，显著提高了模型的数学能力和编程能力，在最优场景下实现了高达11.12%的性能提升。 

---
# Large Language Models for Interpretable Mental Health Diagnosis 

**Title (ZH)**: 可解释的心理健康诊断中的大规模语言模型 

**Authors**: Brian Hyeongseok Kim, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07653)  

**Abstract**: We propose a clinical decision support system (CDSS) for mental health diagnosis that combines the strengths of large language models (LLMs) and constraint logic programming (CLP). Having a CDSS is important because of the high complexity of diagnostic manuals used by mental health professionals and the danger of diagnostic errors. Our CDSS is a software tool that uses an LLM to translate diagnostic manuals to a logic program and solves the program using an off-the-shelf CLP engine to query a patient's diagnosis based on the encoded rules and provided data. By giving domain experts the opportunity to inspect the LLM-generated logic program, and making modifications when needed, our CDSS ensures that the diagnosis is not only accurate but also interpretable. We experimentally compare it with two baseline approaches of using LLMs: diagnosing patients using the LLM-only approach, and using the LLM-generated logic program but without expert inspection. The results show that, while LLMs are extremely useful in generating candidate logic programs, these programs still require expert inspection and modification to guarantee faithfulness to the official diagnostic manuals. Additionally, ethical concerns arise from the direct use of patient data in LLMs, underscoring the need for a safer hybrid approach like our proposed method. 

**Abstract (ZH)**: 我们提出了一种结合大型语言模型（LLMs）和约束逻辑编程（CLP）优势的心理健康诊断临床决策支持系统（CDSS）。开发CDSS的重要性在于，心理健康专业人员使用的诊断手册具有极高的复杂性，诊断错误存在潜在风险。我们的CDSS是一种软件工具，它使用LLM将诊断手册翻译成逻辑程序，并使用现成的CLP引擎查询患者的诊断结果，基于编码规则和提供的数据。通过让领域专家检查LLM生成的逻辑程序，并在必要时进行修改，我们的CDSS确保诊断既准确又可解释。我们实验性地将CDSS与两种基于LLM的基本方法进行了比较：仅使用LLM进行患者诊断的方法，以及使用LLM生成的逻辑程序但未经专家检查的方法。实验结果表明，尽管LLM在生成候选逻辑程序方面非常有用，但这些程序仍然需要专家的检查和修改，才能确保与官方诊断手册的一致性。此外，直接在LLM中使用患者数据引发了一些伦理问题，突显了需要一种更安全的混合方法，如我们提出的方法。 

---
# SafePowerGraph-LLM: Novel Power Grid Graph Embedding and Optimization with Large Language Models 

**Title (ZH)**: SafePowerGraph-LLM：基于大规模语言模型的新颖电力网络图嵌入与优化方法 

**Authors**: Fabien Bernier, Jun Cao, Maxime Cordy, Salah Ghamizi  

**Link**: [PDF](https://arxiv.org/pdf/2501.07639)  

**Abstract**: Efficiently solving Optimal Power Flow (OPF) problems in power systems is crucial for operational planning and grid management. There is a growing need for scalable algorithms capable of handling the increasing variability, constraints, and uncertainties in modern power networks while providing accurate and fast solutions. To address this, machine learning techniques, particularly Graph Neural Networks (GNNs) have emerged as promising approaches. This letter introduces SafePowerGraph-LLM, the first framework explicitly designed for solving OPF problems using Large Language Models (LLM)s. The proposed approach combines graph and tabular representations of power grids to effectively query LLMs, capturing the complex relationships and constraints in power systems. A new implementation of in-context learning and fine-tuning protocols for LLMs is introduced, tailored specifically for the OPF problem. SafePowerGraph-LLM demonstrates reliable performances using off-the-shelf LLM. Our study reveals the impact of LLM architecture, size, and fine-tuning and demonstrates our framework's ability to handle realistic grid components and constraints. 

**Abstract (ZH)**: 在电力系统中高效解决问题最优功率流（OPF）对于运行计划和电网管理至关重要。随着现代电力网络中可变性、约束和不确定性不断增加，对能够处理这些增强的可扩展算法的需求也在增长。为此，机器学习技术，特别是图神经网络（GNNs），已经被证明是颇具前景的方法。本文介绍了SafePowerGraph-LLM，这是第一个专门为利用大型语言模型（LLMs）解决OPF问题设计的框架。提出的这种方法将电力网络的图表示和表格表示结合起来，以便有效地查询LLMs，捕捉电力系统中的复杂关系和约束条件。我们提出了针对OPF问题的新颖的上下文学习和微调协议实现。SafePowerGraph-LLM 在使用即用型大型语言模型的情况下展示了可靠的性能。我们的研究揭示了LLM架构、大小和微调的影响，并展示了我们框架处理现实电网组件和约束的能力。 

---
# PokerBench: Training Large Language Models to become Professional Poker Players 

**Title (ZH)**: PokerBench：训练大型语言模型成为专业 poker 玩家 

**Authors**: Richard Zhuang, Akshat Gupta, Richard Yang, Aniket Rahane, Zhengyu Li, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2501.08328)  

**Abstract**: We introduce PokerBench - a benchmark for evaluating the poker-playing abilities of large language models (LLMs). As LLMs excel in traditional NLP tasks, their application to complex, strategic games like poker poses a new challenge. Poker, an incomplete information game, demands a multitude of skills such as mathematics, reasoning, planning, strategy, and a deep understanding of game theory and human psychology. This makes Poker the ideal next frontier for large language models. PokerBench consists of a comprehensive compilation of 11,000 most important scenarios, split between pre-flop and post-flop play, developed in collaboration with trained poker players. We evaluate prominent models including GPT-4, ChatGPT 3.5, and various Llama and Gemma series models, finding that all state-of-the-art LLMs underperform in playing optimal poker. However, after fine-tuning, these models show marked improvements. We validate PokerBench by having models with different scores compete with each other, demonstrating that higher scores on PokerBench lead to higher win rates in actual poker games. Through gameplay between our fine-tuned model and GPT-4, we also identify limitations of simple supervised fine-tuning for learning optimal playing strategy, suggesting the need for more advanced methodologies for effectively training language models to excel in games. PokerBench thus presents a unique benchmark for a quick and reliable evaluation of the poker-playing ability of LLMs as well as a comprehensive benchmark to study the progress of LLMs in complex game-playing scenarios. The dataset and code will be made available at: \url{this https URL}. 

**Abstract (ZH)**: 我们介绍了一个名为 PokerBench 的基准测试，用于评估大型语言模型 (LLMs) 的扑克对弈能力。由于 LLMs 在传统的自然语言处理 (NLP) 任务中表现出色，因此将其应用于复杂的策略性游戏如扑克，提出了新的挑战。扑克是一种信息不完整的游戏，需要多种技能，包括数学、推理、计划、策略以及深刻的游戏理论和人类心理学理解。这使得扑克成为大型语言模型的理想新挑战领域。PokerBench 包含了由 11,000 个最关键的情景组成的一个全面集合，这些情景被划分为前底牌阶段和后底牌阶段，并与训练有素的扑克选手进行了合作开发。我们评估了包括 GPT-4、ChatGPT 3.5 以及各种 Llama 和 Gemma 系列模型在内的著名模型，发现所有最先进的 LLMs 在玩最优扑克方面表现不佳。然而，在微调后，这些模型显示出显著的改进。我们通过让具有不同评分的模型相互竞争的方式验证了 PokerBench，这表明 PokerBench 的高评分会转化为实际扑克比赛中更高的胜率。通过我们的微调模型与 GPT-4 的对弈，我们还识别出了简单监督微调学习最优策略的局限性，这暗示了需要更多高级方法来有效训练语言模型在游戏场景中表现出色。因此，PokerBench 提供了一个快速可靠的评估 LLMs 棋牌对弈能力的独特基准测试，也是全面研究 LLMs 在复杂游戏场景中进展的一个基准测试。数据集和代码将在以下链接提供：\url{this https URL}。 

---
# Diffusion Adversarial Post-Training for One-Step Video Generation 

**Title (ZH)**: 用于单步视频生成的扩散对抗后训练方法 

**Authors**: Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2501.08316)  

**Abstract**: The diffusion models are widely used for image and video generation, but their iterative generation process is slow and expansive. While existing distillation approaches have demonstrated the potential for one-step generation in the image domain, they still suffer from significant quality degradation. In this work, we propose Adversarial Post-Training (APT) against real data following diffusion pre-training for one-step video generation. To improve the training stability and quality, we introduce several improvements to the model architecture and training procedures, along with an approximated R1 regularization objective. Empirically, our experiments show that our adversarial post-trained model, Seaweed-APT, can generate 2-second, 1280x720, 24fps videos in real time using a single forward evaluation step. Additionally, our model is capable of generating 1024px images in a single step, achieving quality comparable to state-of-the-art methods. 

**Abstract (ZH)**: 扩散模型在图像和视频生成中得到了广泛的应用，但其迭代生成过程速度较慢且成本较高。尽管现有的蒸馏方法在图像域中展示了一步生成的潜在可能性，但仍存在质量显著下降的问题。在此工作中，我们提出了一种针对真实数据的对抗性后训练（Adversarial Post-Training, APT），该方法在扩散预训练的基础上实现了一步生成视频。为了提高训练稳定性和生成质量，我们对模型架构和训练过程进行了多项改进，并引入了近似的R1正则化目标。实验结果表明，我们的对抗性后训练模型Seaweed-APT能够在单步前向评价中实时生成分辨率为1280x720、时长为2秒、帧率为24fps的视频。此外，我们的模型能够在单步中生成分辨率为1024px的图像，其生成质量与当前最先进的方法相当。 

---
# Polynomial Threshold Functions of Bounded Tree-Width: Some Explainability and Complexity Aspects 

**Title (ZH)**: 多项式阈函数的有界树宽表示：一些可解释性和复杂性方面 

**Authors**: Karine Chubarian, Johnny Joyce, Gyorgy Turan  

**Link**: [PDF](https://arxiv.org/pdf/2501.08297)  

**Abstract**: The tree-width of a multivariate polynomial is the tree-width of the hypergraph with hyperedges corresponding to its terms. Multivariate polynomials of bounded tree-width have been studied by Makowsky and Meer as a new sparsity condition that allows for polynomial solvability of problems which are intractable in general. We consider a variation on this theme for Boolean variables. A representation of a Boolean function as the sign of a polynomial is called a polynomial threshold representation. We discuss Boolean functions representable as polynomial threshold functions of bounded tree-width and present two applications to Bayesian network classifiers, a probabilistic graphical model. Both applications are in Explainable Artificial Intelligence (XAI), the research area dealing with the black-box nature of many recent machine learning models. We also give a separation result between the representational power of positive and general polynomial threshold functions. 

**Abstract (ZH)**: 多元多项式的树宽是指与其项对应的超图的树宽。Makowsky 和 Meer 研究了有界树宽的多元多项式，作为一种新颖的稀疏性条件，使得某些在一般情况下不可多项式求解的问题变得可多项式求解。我们在此基础上考虑了基于布尔变量的变体。将布尔函数用多项式的符号表示称为多项式阈值表示。我们讨论了可由有界树宽多项式阈值函数表示的布尔函数，并给出了两种基于有向图模型贝叶斯网络分类器的应用，这些应用属于可解释人工智能（XAI）的研究领域，专门关注许多现代机器学习模型的黑箱性质。我们还给出了正多项式阈值函数与一般多项式阈值函数表示能力之间的分离结果。 

---
# HALoGEN: Fantastic LLM Hallucinations and Where to Find Them 

**Title (ZH)**: HALoGEN: 神奇的大型语言模型幻觉以及它们的来源 

**Authors**: Abhilasha Ravichander, Shrusti Ghela, David Wadden, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2501.08292)  

**Abstract**: Despite their impressive ability to generate high-quality and fluent text, generative large language models (LLMs) also produce hallucinations: statements that are misaligned with established world knowledge or provided input context. However, measuring hallucination can be challenging, as having humans verify model generations on-the-fly is both expensive and time-consuming. In this work, we release HALoGEN, a comprehensive hallucination benchmark consisting of: (1) 10,923 prompts for generative models spanning nine domains including programming, scientific attribution, and summarization, and (2) automatic high-precision verifiers for each use case that decompose LLM generations into atomic units, and verify each unit against a high-quality knowledge source. We use this framework to evaluate ~150,000 generations from 14 language models, finding that even the best-performing models are riddled with hallucinations (sometimes up to 86% of generated atomic facts depending on the domain). We further define a novel error classification for LLM hallucinations based on whether they likely stem from incorrect recollection of training data (Type A errors), or incorrect knowledge in training data (Type B errors), or are fabrication (Type C errors). We hope our framework provides a foundation to enable the principled study of why generative models hallucinate, and advances the development of trustworthy large language models. 

**Abstract (ZH)**: 尽管生成型大规模语言模型（LLMs）能够生成高质量且流畅的文本，但它们也会产生幻觉：这些陈述与已建立的世界知识或提供的上下文不符。然而，测量幻觉具有挑战性，因为让人类实时验证模型生成的内容既昂贵又耗时。本研究中，我们推出了HALoGEN，这是一个全面的幻觉基准，包括以下内容：（1）涵盖编程、科学引用和摘要等九个领域的10,923个生成模型提示；（2）针对每种应用场景的自动高精度验证器，将LLM生成的内容分解为原子单元，并将每个单元与高质量的知识来源进行验证。我们利用这一框架评估了来自14个语言模型的约150,000个生成内容，发现即使是表现最好的模型，在某些领域中也会充满幻觉（有时生成的原子事实中有高达86%的错误）。我们进一步基于幻觉是否可能源自训练数据中的错误回忆（类型A错误）、训练数据中的错误知识（类型B错误）或虚构（类型C错误）定义了一种新的错误分类。我们希望这一框架能够为系统地研究生成模型为什么会产生幻觉提供基础，并推动可信的大规模语言模型的发展。 

---
# Comparative Analysis of Efficient Adapter-Based Fine-Tuning of State-of-the-Art Transformer Models 

**Title (ZH)**: 基于适配器的有效微调：领先变换器模型的比较分析 

**Authors**: Saad Mashkoor Siddiqui, Mohammad Ali Sheikh, Muhammad Aleem, Kajol R Singh  

**Link**: [PDF](https://arxiv.org/pdf/2501.08271)  

**Abstract**: In this work, we investigate the efficacy of various adapter architectures on supervised binary classification tasks from the SuperGLUE benchmark as well as a supervised multi-class news category classification task from Kaggle. Specifically, we compare classification performance and time complexity of three transformer models, namely DistilBERT, ELECTRA, and BART, using conventional fine-tuning as well as nine state-of-the-art (SoTA) adapter architectures. Our analysis reveals performance differences across adapter architectures, highlighting their ability to achieve comparable or better performance relative to fine-tuning at a fraction of the training time. Similar results are observed on the new classification task, further supporting our findings and demonstrating adapters as efficient and flexible alternatives to fine-tuning. This study provides valuable insights and guidelines for selecting and implementing adapters in diverse natural language processing (NLP) applications. 

**Abstract (ZH)**: 在本研究中，我们探讨了各种适应器架构在SuperGLUE基准的监督二分类任务以及Kaggle的监督多分类新闻类别分类任务中的有效性。具体来说，我们使用传统的微调方法，并比较了DistilBERT、ELECTRA和BART三种变换器模型与九种最先进的（SOTA）适应器架构的分类性能和时间复杂性。我们的分析揭示了不同适应器架构之间的性能差异，突出了它们能够在极短的训练时间内实现与微调相当或更优的性能。类似的结果也出现在新的分类任务中，进一步支持了我们的发现，并展示了适应器作为微调的高效且灵活替代方案的能力。本研究为在多样化自然语言处理（NLP）应用中选择和实施适应器提供了有价值的见解和指导。 

---
# AI Driven Water Segmentation with deep learning models for Enhanced Flood Monitoring 

**Title (ZH)**: 基于深度学习模型的AI驱动水体分割以增强洪水监测 

**Authors**: Sanjida Afrin Mou, Tasfia Noor Chowdhury, Adib Ibn Mannan, Sadia Nourin Mim, Lubana Tarannum, Tasrin Noman, Jamal Uddin Ahamed  

**Link**: [PDF](https://arxiv.org/pdf/2501.08266)  

**Abstract**: Flooding is a major natural hazard causing significant fatalities and economic losses annually, with increasing frequency due to climate change. Rapid and accurate flood detection and monitoring are crucial for mitigating these impacts. This study compares the performance of three deep learning models UNet, ResNet, and DeepLabv3 for pixelwise water segmentation to aid in flood detection, utilizing images from drones, in field observations, and social media. This study involves creating a new dataset that augments wellknown benchmark datasets with flood-specific images, enhancing the robustness of the models. The UNet, ResNet, and DeepLab v3 architectures are tested to determine their effectiveness in various environmental conditions and geographical locations, and the strengths and limitations of each model are also discussed here, providing insights into their applicability in different scenarios by predicting image segmentation masks. This fully automated approach allows these models to isolate flooded areas in images, significantly reducing processing time compared to traditional semi-automated methods. The outcome of this study is to predict segmented masks for each image effected by a flood disaster and the validation accuracy of these models. This methodology facilitates timely and continuous flood monitoring, providing vital data for emergency response teams to reduce loss of life and economic damages. It offers a significant reduction in the time required to generate flood maps, cutting down the manual processing time. Additionally, we present avenues for future research, including the integration of multimodal data sources and the development of robust deep learning architectures tailored specifically for flood detection tasks. Overall, our work contributes to the advancement of flood management strategies through innovative use of deep learning technologies. 

**Abstract (ZH)**: 洪水是一种主要的自然灾害，每年造成大量的人员伤亡和经济损失，并因气候变化而频率增加。快速准确的洪水检测与监测对于减轻这些影响至关重要。本研究对比了三种深度学习模型（UNet、ResNet和DeepLabv3）在基于多源图像（包括无人机图像、现场观测和社交媒体）的像素级水体分割中的性能，以辅助洪水检测。本研究创建了一个新的数据集，通过补充现有的基准数据集并加入特定于洪水的图像，提高了模型的鲁棒性。研究人员测试了UNet、ResNet和DeepLab v3架构在各种环境条件和地理区域中的有效性，并讨论了每种模型的优势和局限性，为不同场景下的应用提供了见解，通过预测图像分割掩模来预测受灾图像的分割掩模。该完全自动化的方法使得这些模型能够在图像中隔离出洪泛区域，大大减少了处理时间，相比传统的半自动化方法具有明显的优势。本研究的研究结果包括预测因洪水灾害影响的每张图像的分割掩模和这些模型的验证精度。这种方法能够实现及时且连续的洪水监测，为应急响应团队提供关键数据，以减少人员伤亡和经济损失。此外，该方法还大幅减少了生成洪水地图所需的手动处理时间。我们还展示了未来研究的途径，包括多模态数据源的集成和针对洪水检测任务定制的稳健深度学习架构的开发。总体而言，我们的工作通过创新使用深度学习技术，为洪水管理策略的改进做出了贡献。 

---
# Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models 

**Title (ZH)**: 基于上下文检索与推理激发长期上下文大语言模型的能力 

**Authors**: Yifu Qiu, Varun Embar, Yizhe Zhang, Navdeep Jaitly, Shay B. Cohen, Benjamin Han  

**Link**: [PDF](https://arxiv.org/pdf/2501.08248)  

**Abstract**: Recent advancements in long-context language models (LCLMs) promise to transform Retrieval-Augmented Generation (RAG) by simplifying pipelines. With their expanded context windows, LCLMs can process entire knowledge bases and perform retrieval and reasoning directly -- a capability we define as In-Context Retrieval and Reasoning (ICR^2). However, existing benchmarks like LOFT often overestimate LCLM performance by providing overly simplified contexts. To address this, we introduce ICR^2, a benchmark that evaluates LCLMs in more realistic scenarios by including confounding passages retrieved with strong retrievers. We then propose three methods to enhance LCLM performance: (1) retrieve-then-generate fine-tuning, (2) retrieval-attention-probing, which uses attention heads to filter and de-noise long contexts during decoding, and (3) joint retrieval head training alongside the generation head. Our evaluation of five well-known LCLMs on LOFT and ICR^2 demonstrates significant gains with our best approach applied to Mistral-7B: +17 and +15 points by Exact Match on LOFT, and +13 and +2 points on ICR^2, compared to vanilla RAG and supervised fine-tuning, respectively. It even outperforms GPT-4-Turbo on most tasks despite being a much smaller model. 

**Abstract (ZH)**: 近期在长上下文语言模型（LCLMs）方面的进展有望通过简化管道来改造检索增强生成（RAG）。借助扩展的上下文窗口，LCLMs可以处理整个知识库，并直接执行检索和推理——我们将其定义为上下文检索与推理（ICR^2）。然而，现有的基准测试，如LOFT，往往会通过提供过于简化的上下文而高估LCLM的性能。为了解决这一问题，我们引入了ICR^2，这是一种通过包含强检索器检索的混淆段落来在更现实的场景中评估LCLMs的基准测试。随后，我们提出了三种提高LCLM性能的方法：（1）检索-生成微调，（2）检索-注意力-探测，这种方法在解码过程中使用注意力头对长上下文进行筛选和去噪，以及（3）与生成头共同训练检索头。我们在LOFT和ICR^2上的五个知名LCLMs的评估表明，我们的最佳方法应用于Mistral-7B时，相较于基础RAG和监督微调，精确匹配得分分别提高了17和15分。在ICR^2上分别提高了13分和2分。即使作为一个较小的模型，它在大多数任务中也超过了GPT-4-Turbo。 

---
# Engineering LLM Powered Multi-agent Framework for Autonomous CloudOps 

**Title (ZH)**: 基于大规模语言模型的多agents框架在自主云运营中的工程实现 

**Authors**: Kannan Parthasarathy, Karthik Vaidhyanathan, Rudra Dhar, Venkat Krishnamachari, Basil Muhammed, Adyansh Kakran, Sreemaee Akshathala, Shrikara Arun, Sumant Dubey, Mohan Veerubhotla, Amey Karan  

**Link**: [PDF](https://arxiv.org/pdf/2501.08243)  

**Abstract**: Cloud Operations (CloudOps) is a rapidly growing field focused on the automated management and optimization of cloud infrastructure which is essential for organizations navigating increasingly complex cloud environments. MontyCloud Inc. is one of the major companies in the CloudOps domain that leverages autonomous bots to manage cloud compliance, security, and continuous operations. To make the platform more accessible and effective to the customers, we leveraged the use of GenAI.
Developing a GenAI-based solution for autonomous CloudOps for the existing MontyCloud system presented us with various challenges such as i) diverse data sources; ii) orchestration of multiple processes; and iii) handling complex workflows to automate routine tasks. To this end, we developed MOYA, a multi-agent framework that leverages GenAI and balances autonomy with the necessary human control. This framework integrates various internal and external systems and is optimized for factors like task orchestration, security, and error mitigation while producing accurate, reliable, and relevant insights by utilizing Retrieval Augmented Generation (RAG). Evaluations of our multi-agent system with the help of practitioners as well as using automated checks demonstrate enhanced accuracy, responsiveness, and effectiveness over non-agentic approaches across complex workflows. 

**Abstract (ZH)**: 云计算运营（CloudOps）是一个迅速发展的领域，专注于自动管理与优化云基础设施，对于在日益复杂的云环境中导航的组织而言，这是至关重要的。蒙蒂云公司（MontyCloud Inc.）是CloudOps领域的主要企业之一，通过自主机器人管理和监控云合规性、安全性和持续运营。为了使平台更加易于使用且有效，我们利用了生成式人工智能（GenAI）技术。

针对现有蒙蒂云系统开发基于GenAI的自主CloudOps解决方案，我们面临了多个挑战，包括：i) 多样化的数据来源；ii) 多个过程的协调；和iii) 处理复杂的流程自动化常规任务。为此，我们开发了Moya，一个基于GenAI的多Agent框架，平衡了自主性和必要的人工控制。该框架整合了各种内部和外部系统，并针对因素如任务协调、安全性和错误缓解进行了优化，利用检索增强生成（RAG）技术生成准确、可靠且相关的洞察。通过从业者和自动化检查的帮助，对我们的多Agent系统进行评估，表明与非Agent方法相比，该系统在复杂流程中的准确度、响应性和有效性得到了显著提升。 

---
# A Feature-Level Ensemble Model for COVID-19 Identification in CXR Images using Choquet Integral and Differential Evolution Optimization 

**Title (ZH)**: 基于Choquet积分和差分进化优化的特征级集成模型在胸片图像中新冠肺炎识别 

**Authors**: Amir Reza Takhsha, Maryam Rastgarpour, Mozhgan Naderi  

**Link**: [PDF](https://arxiv.org/pdf/2501.08241)  

**Abstract**: The COVID-19 pandemic has profoundly impacted billions globally. It challenges public health and healthcare systems due to its rapid spread and severe respiratory effects. An effective strategy to mitigate the COVID-19 pandemic involves integrating testing to identify infected individuals. While RT-PCR is considered the gold standard for diagnosing COVID-19, it has some limitations such as the risk of false negatives. To address this problem, this paper introduces a novel Deep Learning Diagnosis System that integrates pre-trained Deep Convolutional Neural Networks (DCNNs) within an ensemble learning framework to achieve precise identification of COVID-19 cases from Chest X-ray (CXR) images. We combine feature vectors from the final hidden layers of pre-trained DCNNs using the Choquet integral to capture interactions between different DCNNs that a linear approach cannot. We employed Sugeno-$\lambda$ measure theory to derive fuzzy measures for subsets of networks to enable aggregation. We utilized Differential Evolution to estimate fuzzy densities. We developed a TensorFlow-based layer for Choquet operation to facilitate efficient aggregation, due to the intricacies involved in aggregating feature vectors. Experimental results on the COVIDx dataset show that our ensemble model achieved 98\% accuracy in three-class classification and 99.50\% in binary classification, outperforming its components-DenseNet-201 (97\% for three-class, 98.75\% for binary), Inception-v3 (96.25\% for three-class, 98.50\% for binary), and Xception (94.50\% for three-class, 98\% for binary)-and surpassing many previous methods. 

**Abstract (ZH)**: 新冠肺炎疫情对全球数十亿人口产生了深远的影响。它由于传播速度快和严重的呼吸道症状而对公共卫生和医疗保健系统构成了挑战。有效应对新冠肺炎疫情的策略之一是整合检测手段以识别感染者。虽然RT-PCR被公认为诊断新冠肺炎的金标准，但它存在一些局限性，如假阴性风险。为了解决这一问题，本文提出了一种新颖的深度学习诊断系统，该系统将预训练的深层卷积神经网络（DCNNs）集成到集成学习框架中，以从胸部X光（CXR）图像中精确识别新冠肺炎病例。我们通过Choquet整合理论将预训练DCNNs最终隐藏层的特征向量结合在一起，捕捉不同DCNNs之间的交互作用，这是线性方法无法做到的。我们使用Sugeno-$\lambda$测度理论为网络子集推导模糊测度，以便进行聚合。我们利用差分演化算法估计模糊密度。我们开发了一个基于TensorFlow的Choquet操作层，以促进特征向量聚合，因为聚合特征向量涉及复杂的细节。实验结果表明，在COCOVID数据集上，我们的集成模型在三分类中的准确率达到了98%，在二分类中的准确率达到了99.50%，优于其组成部分——DenseNet-201（三分类97%，二分类98.75%）、Inception-v3（三分类96.25%，二分类98.50%）和Xception（三分类94.50%，二分类98%），并且超越了许多先前的方法。 

---
# Dynamic Pricing in High-Speed Railways Using Multi-Agent Reinforcement Learning 

**Title (ZH)**: 使用多智能体强化学习的高速铁路动态定价研究 

**Authors**: Enrique Adrian Villarrubia-Martin, Luis Rodriguez-Benitez, David Muñoz-Valero, Giovanni Montana, Luis Jimenez-Linares  

**Link**: [PDF](https://arxiv.org/pdf/2501.08234)  

**Abstract**: This paper addresses a critical challenge in the high-speed passenger railway industry: designing effective dynamic pricing strategies in the context of competing and cooperating operators. To address this, a multi-agent reinforcement learning (MARL) framework based on a non-zero-sum Markov game is proposed, incorporating random utility models to capture passenger decision making. Unlike prior studies in areas such as energy, airlines, and mobile networks, dynamic pricing for railway systems using deep reinforcement learning has received limited attention. A key contribution of this paper is a parametrisable and versatile reinforcement learning simulator designed to model a variety of railway network configurations and demand patterns while enabling realistic, microscopic modelling of user behaviour, called RailPricing-RL. This environment supports the proposed MARL framework, which models heterogeneous agents competing to maximise individual profits while fostering cooperative behaviour to synchronise connecting services. Experimental results validate the framework, demonstrating how user preferences affect MARL performance and how pricing policies influence passenger choices, utility, and overall system dynamics. This study provides a foundation for advancing dynamic pricing strategies in railway systems, aligning profitability with system-wide efficiency, and supporting future research on optimising pricing policies. 

**Abstract (ZH)**: 本文针对高速客运铁路行业中的一项关键挑战：在竞争与合作的运营商背景下设计有效的动态定价策略。为此，本文提出了一种基于非零和马尔可夫博弈的多智能体强化学习（MARL）框架，结合随机效用模型来捕捉乘客的决策过程。与能源、航空和移动网络等领域中的先前研究不同，使用深度强化学习进行铁路系统的动态定价尚未受到广泛关注。本文的一个关键贡献是设计了一个可参数化且具有灵活性的强化学习模拟器，该模拟器能够模拟多种铁路网络配置和需求模式，并允许对用户行为进行现实且微观的建模，名为RailPricing-RL。该环境支持提出的MARL框架，该框架通过建模不同智能体竞争以最大化个体利润的同时促进合作行为以同步连接服务来实现。实验结果验证了该框架，展示了用户偏好如何影响MARL性能以及定价策略如何影响乘客选择、效用和系统总体动态。本研究为基础动态定价策略在铁路系统的应用奠定了基础，平衡了盈利能力与系统效率，并为未来优化定价策略的研究提供了支持。 

---
# ASTRID -- An Automated and Scalable TRIaD for the Evaluation of RAG-based Clinical Question Answering Systems 

**Title (ZH)**: ASTRID — 一种自动可扩展的 TRIaD 用于基于RAEGEN的临床问答系统评估 

**Authors**: Mohita Chowdhury, Yajie Vera He, Aisling Higham, Ernest Lim  

**Link**: [PDF](https://arxiv.org/pdf/2501.08208)  

**Abstract**: Large Language Models (LLMs) have shown impressive potential in clinical question answering (QA), with Retrieval Augmented Generation (RAG) emerging as a leading approach for ensuring the factual accuracy of model responses. However, current automated RAG metrics perform poorly in clinical and conversational use cases. Using clinical human evaluations of responses is expensive, unscalable, and not conducive to the continuous iterative development of RAG systems. To address these challenges, we introduce ASTRID - an Automated and Scalable TRIaD for evaluating clinical QA systems leveraging RAG - consisting of three metrics: Context Relevance (CR), Refusal Accuracy (RA), and Conversational Faithfulness (CF). Our novel evaluation metric, CF, is designed to better capture the faithfulness of a model's response to the knowledge base without penalising conversational elements. To validate our triad, we curate a dataset of over 200 real-world patient questions posed to an LLM-based QA agent during surgical follow-up for cataract surgery - the highest volume operation in the world - augmented with clinician-selected questions for emergency, clinical, and non-clinical out-of-domain scenarios. We demonstrate that CF can predict human ratings of faithfulness better than existing definitions for conversational use cases. Furthermore, we show that evaluation using our triad consisting of CF, RA, and CR exhibits alignment with clinician assessment for inappropriate, harmful, or unhelpful responses. Finally, using nine different LLMs, we demonstrate that the three metrics can closely agree with human evaluations, highlighting the potential of these metrics for use in LLM-driven automated evaluation pipelines. We also publish the prompts and datasets for these experiments, providing valuable resources for further research and development. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在临床问答（QA）任务中展现出了令人印象深刻的潜力，检索增强生成（RAG）方法因其确保模型响应的事实准确性而成为领先的方法。然而，当前自动化的RAG评估指标在临床和对话场景中表现不佳。依靠临床人类评估响应既昂贵又难以扩展，也不利于RAG系统的持续迭代发展。为了解决这些挑战，我们引入了ASTRID——一个利用RAG评估临床QA系统的方法，它包括三个指标：上下文相关性（Context Relevance, CR）、拒绝准确性（Refusal Accuracy, RA）和对话保真度（Conversational Faithfulness, CF）。我们的创新性评估指标CF旨在更好地捕捉模型响应与知识库的一致性，而不惩罚对话元素。为了验证这套指标，我们收集了一个包含200多个真实世界患者问题的数据集，这些问题是在白内障手术后随访过程中提出的，在这个全球最高频率的手术类型中，这些问题由临床医生选择扩展到紧急情况、临床和非临床领域外的情景。实验证明，CF比现有的对话场景定义更有效地预测人类对保真度的评分。此外，我们展示使用我们的三重评估指标（CF、RA和CR）对不合适、有害或无用的响应进行评估时，与临床评估高度一致。最后，使用九种不同的大模型，我们表明这三个指标能够与人类评估高度一致，突显了这些指标在大规模语言模型驱动的自动化评估流程中的潜在应用价值。我们还发布了这些实验的提示和数据集，为后续的研究和开发提供了宝贵的资源。 

---
# Modeling Feature Maps for Quantum Machine Learning 

**Title (ZH)**: 量子机器学习中的特征图建模 

**Authors**: Navneet Singh, Shiva Raj Pokhrel  

**Link**: [PDF](https://arxiv.org/pdf/2501.08205)  

**Abstract**: Quantum Machine Learning (QML) offers significant potential for complex tasks like genome sequence classification, but quantum noise on Noisy Intermediate-Scale Quantum (NISQ) devices poses practical challenges. This study systematically evaluates how various quantum noise models including dephasing, amplitude damping, depolarizing, thermal noise, bit-flip, and phase-flip affect key QML algorithms (QSVC, Peg-QSVC, QNN, VQC) and feature mapping techniques (ZFeatureMap, ZZFeatureMap, and PauliFeatureMap). Results indicate that QSVC is notably robust under noise, whereas Peg-QSVC and QNN are more sensitive, particularly to depolarizing and amplitude-damping noise. The PauliFeatureMap is especially vulnerable, highlighting difficulties in maintaining accurate classification under noisy conditions. These findings underscore the critical importance of feature map selection and noise mitigation strategies in optimizing QML for genomic classification, with promising implications for personalized medicine. 

**Abstract (ZH)**: 量子机器学习（QML）在复杂的任务如基因组序列分类中具有巨大潜力，但在Noisy Intermediate-Scale Quantum（NISQ）设备上存在的量子噪声则带来了实际挑战。本研究系统地评估了各种量子噪声模型（包括相位退相、振幅弛豫、极化退相、热噪声、比特反转和位相反转）对关键QML算法（QSVC、Peg-QSVC、QNN、VQC）和特征映射技术（ZFeatureMap、ZZFeatureMap、PauliFeatureMap）的影响。研究结果表明，QSVC在噪声下表现出显著的鲁棒性，而Peg-QSVC和QNN则更加敏感，特别对极化退相和振幅弛豫噪声。PauliFeatureMap尤为脆弱，凸显了在噪声条件下保持分类准确性的困难。这些发现强调了在优化基因组分类中的QML时，特征映射选择和噪声缓解策略的至关重要性，并对未来个性化医学具有潜在影响。 

---
# EmoNeXt: an Adapted ConvNeXt for Facial Emotion Recognition 

**Title (ZH)**: EmoNeXt：适应面部情感识别的ConvNeXt变体 

**Authors**: Yassine El Boudouri, Amine Bohi  

**Link**: [PDF](https://arxiv.org/pdf/2501.08199)  

**Abstract**: Facial expressions play a crucial role in human communication serving as a powerful and impactful means to express a wide range of emotions. With advancements in artificial intelligence and computer vision, deep neural networks have emerged as effective tools for facial emotion recognition. In this paper, we propose EmoNeXt, a novel deep learning framework for facial expression recognition based on an adapted ConvNeXt architecture network. We integrate a Spatial Transformer Network (STN) to focus on feature-rich regions of the face and Squeeze-and-Excitation blocks to capture channel-wise dependencies. Moreover, we introduce a self-attention regularization term, encouraging the model to generate compact feature vectors. We demonstrate the superiority of our model over existing state-of-the-art deep learning models on the FER2013 dataset regarding emotion classification accuracy. 

**Abstract (ZH)**: 面部表情在人际沟通中扮演着至关重要的角色，是表达广泛情绪的有力且有效的方式。随着人工智能和计算机视觉技术的发展，深度神经网络已成为面部情绪识别的有效工具。本文提出了一种基于改编ConvNeXt架构的新颖深度学习框架EmoNeXt。我们集成了空间变换网络（STN），以关注面部特征丰富的区域，并引入Squeeze-and-Excitation块以捕捉通道间的依赖关系。此外，我们引入了一种自注意力正则化项，促使模型生成紧凑的特征向量。实验结果展示了我们在FER2013数据集上的情绪分类准确性方面优于现有最先进的深度学习模型。 

---
# A Critical Synthesis of Uncertainty Quantification and Foundation Models in Monocular Depth Estimation 

**Title (ZH)**: 单目深度估计中不确定性量化与基础模型的批判性综述 

**Authors**: Steven Landgraf, Rongjun Qin, Markus Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2501.08188)  

**Abstract**: While recent foundation models have enabled significant breakthroughs in monocular depth estimation, a clear path towards safe and reliable deployment in the real-world remains elusive. Metric depth estimation, which involves predicting absolute distances, poses particular challenges, as even the most advanced foundation models remain prone to critical errors. Since quantifying the uncertainty has emerged as a promising endeavor to address these limitations and enable trustworthy deployment, we fuse five different uncertainty quantification methods with the current state-of-the-art DepthAnythingV2 foundation model. To cover a wide range of metric depth domains, we evaluate their performance on four diverse datasets. Our findings identify fine-tuning with the Gaussian Negative Log-Likelihood Loss (GNLL) as a particularly promising approach, offering reliable uncertainty estimates while maintaining predictive performance and computational efficiency on par with the baseline, encompassing both training and inference time. By fusing uncertainty quantification and foundation models within the context of monocular depth estimation, this paper lays a critical foundation for future research aimed at improving not only model performance but also its explainability. Extending this critical synthesis of uncertainty quantification and foundation models into other crucial tasks, such as semantic segmentation and pose estimation, presents exciting opportunities for safer and more reliable machine vision systems. 

**Abstract (ZH)**: 尽管最近的基础模型已经在单目深度估计方面取得了显著突破，但如何安全可靠地将其应用于现实世界仍然缺乏明确的路径。距离估计（一种涉及预测绝对距离的度量深度估计）尤其具挑战性，即使是最先进的基础模型仍然容易出现关键错误。鉴于量化不确定性已成为解决这些局限性和实现可信赖部署的一项有希望的努力，我们在当前最先进的DepthAnythingV2基础模型中融合了五种不同的不确定性量化方法。为了涵盖广泛的度量深度领域，我们在这四个不同数据集上对其性能进行了评估。我们的研究结果表明，使用高斯负对数似然损失（GNLL）进行微调特别具有前景，它提供了可靠的不确定性估计，同时在预测性能和计算效率方面与基线保持一致，包括训练和推理时间。通过在单目深度估计的背景下融合不确定性量化和基础模型，这篇论文为未来旨在提高模型性能及其可解释性的研究奠定了关键基础。将不确定性量化和基础模型的这一关键结合扩展到其他关键任务，如语义分割和姿态估计，为更安全、更可靠的机器视觉系统提供了令人兴奋的机会。 

---
# A Multi-Modal AI Copilot for Single-Cell Analysis with Instruction Following 

**Title (ZH)**: 一种遵循指令的多模态AI副驾系统，用于单细胞分析 

**Authors**: Yin Fang, Xinle Deng, Kangwei Liu, Ningyu Zhang, Jingyang Qian, Penghui Yang, Xiaohui Fan, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.08187)  

**Abstract**: Large language models excel at interpreting complex natural language instructions, enabling them to perform a wide range of tasks. In the life sciences, single-cell RNA sequencing (scRNA-seq) data serves as the "language of cellular biology", capturing intricate gene expression patterns at the single-cell level. However, interacting with this "language" through conventional tools is often inefficient and unintuitive, posing challenges for researchers. To address these limitations, we present InstructCell, a multi-modal AI copilot that leverages natural language as a medium for more direct and flexible single-cell analysis. We construct a comprehensive multi-modal instruction dataset that pairs text-based instructions with scRNA-seq profiles from diverse tissues and species. Building on this, we develop a multi-modal cell language architecture capable of simultaneously interpreting and processing both modalities. InstructCell empowers researchers to accomplish critical tasks-such as cell type annotation, conditional pseudo-cell generation, and drug sensitivity prediction-using straightforward natural language commands. Extensive evaluations demonstrate that InstructCell consistently meets or exceeds the performance of existing single-cell foundation models, while adapting to diverse experimental conditions. More importantly, InstructCell provides an accessible and intuitive tool for exploring complex single-cell data, lowering technical barriers and enabling deeper biological insights. 

**Abstract (ZH)**: 大型语言模型在解释复杂的自然语言指令方面表现出色，使其能够执行广泛的任务。在生命科学领域，单细胞RNA测序(scRNA-seq)数据是“细胞生物学的语言”，能够捕捉单细胞层面精细的基因表达模式。然而，通过传统工具与这一“语言”互动往往效率低下且不够直观，给研究人员带来了挑战。为解决这些限制，我们提出了InstructCell，这是一种多模态AI辅助工具，利用自然语言作为更直接和灵活的单细胞分析媒介。我们构建了一个全面的多模态指令数据集，将基于文本的指令与来自不同组织和物种的scRNA-seq特征相结合。在此基础上，我们开发了一种能够同时解释和处理两种模态的多模态细胞语言架构。InstructCell使研究人员能够使用简单自然语言命令完成关键任务，如细胞类型注释、条件伪细胞生成和药物敏感性预测。广泛的评估结果显示，InstructCell在性能上一致地超越或等于现有的单细胞基础模型，并适应不同的实验条件。更重要的是，InstructCell提供了一个易于访问且直观的工具，以探索复杂的单细胞数据，降低技术门槛并促进更深刻的生物学见解。 

---
# Revolutionizing Communication with Deep Learning and XAI for Enhanced Arabic Sign Language Recognition 

**Title (ZH)**: 利用深度学习和解释性人工智能革新手语识别：以阿拉伯手语为例 

**Authors**: Mazen Balat, Rewaa Awaad, Ahmed B. Zaky, Salah A. Aly  

**Link**: [PDF](https://arxiv.org/pdf/2501.08169)  

**Abstract**: This study introduces an integrated approach to recognizing Arabic Sign Language (ArSL) using state-of-the-art deep learning models such as MobileNetV3, ResNet50, and EfficientNet-B2. These models are further enhanced by explainable AI (XAI) techniques to boost interpretability. The ArSL2018 and RGB Arabic Alphabets Sign Language (AASL) datasets are employed, with EfficientNet-B2 achieving peak accuracies of 99.48\% and 98.99\%, respectively. Key innovations include sophisticated data augmentation methods to mitigate class imbalance, implementation of stratified 5-fold cross-validation for better generalization, and the use of Grad-CAM for clear model decision transparency. The proposed system not only sets new benchmarks in recognition accuracy but also emphasizes interpretability, making it suitable for applications in healthcare, education, and inclusive communication technologies. 

**Abstract (ZH)**: 本研究介绍了一种综合方法，用于利用先进深度学习模型（如MobileNetV3、ResNet50和EfficientNet-B2）识别阿拉伯手语（ArSL）。这些模型通过可解释人工智能（XAI）技术进一步增强，以提高可解释性。研究使用了ArSL2018和RGB阿拉伯字母手语（AASL）数据集，EfficientNet-B2在这些数据集上分别实现了99.48%和98.99%的峰值准确率。关键创新包括复杂的数据增强方法，以减轻类别不平衡问题，实施分层5折交叉验证以提高泛化能力，以及使用Grad-CAM以提高模型决策的透明度。所提出系统不仅在识别准确性方面设立了新的基准，还强调了可解释性，使其适用于医疗保健、教育和包容性通信技术的应用。 

---
# Potential and Perils of Large Language Models as Judges of Unstructured Textual Data 

**Title (ZH)**: 大型语言模型作为未结构化文本数据裁判的潜力与风险 

**Authors**: Rewina Bedemariam, Natalie Perez, Sreyoshi Bhaduri, Satya Kapoor, Alex Gil, Elizabeth Conjar, Ikkei Itoku, David Theil, Aman Chadha, Naumaan Nayyar  

**Link**: [PDF](https://arxiv.org/pdf/2501.08167)  

**Abstract**: Rapid advancements in large language models have unlocked remarkable capabilities when it comes to processing and summarizing unstructured text data. This has implications for the analysis of rich, open-ended datasets, such as survey responses, where LLMs hold the promise of efficiently distilling key themes and sentiments. However, as organizations increasingly turn to these powerful AI systems to make sense of textual feedback, a critical question arises, can we trust LLMs to accurately represent the perspectives contained within these text based datasets? While LLMs excel at generating human-like summaries, there is a risk that their outputs may inadvertently diverge from the true substance of the original responses. Discrepancies between the LLM-generated outputs and the actual themes present in the data could lead to flawed decision-making, with far-reaching consequences for organizations. This research investigates the effectiveness of LLMs as judge models to evaluate the thematic alignment of summaries generated by other LLMs. We utilized an Anthropic Claude model to generate thematic summaries from open-ended survey responses, with Amazon's Titan Express, Nova Pro, and Meta's Llama serving as LLM judges. The LLM-as-judge approach was compared to human evaluations using Cohen's kappa, Spearman's rho, and Krippendorff's alpha, validating a scalable alternative to traditional human centric evaluation methods. Our findings reveal that while LLMs as judges offer a scalable solution comparable to human raters, humans may still excel at detecting subtle, context-specific nuances. This research contributes to the growing body of knowledge on AI assisted text analysis. We discuss limitations and provide recommendations for future research, emphasizing the need for careful consideration when generalizing LLM judge models across various contexts and use cases. 

**Abstract (ZH)**: 大规模语言模型的快速进步使其在处理和总结非结构化文本数据方面展示了非凡的能力。这在分析丰富且开放式的数据集（例如调查问卷响应）方面具有重要意义，这些数据集有可能通过LLM高效地提炼关键主题和情感。然而，随着各组织越来越多地利用这些强大的人工智能系统来理解文本反馈，一个关键问题也随之浮现：我们能否信任LLM准确地代表这些基于文本的数据集中的观点？尽管LLM在生成类人摘要方面表现出色，但其输出可能会无意中偏离原始回答的真实内容，这存在一定的风险。LLM生成的摘要与实际存在的主题之间的差异可能导致决策失误，从而对组织产生深远影响。本研究探讨了LLM作为裁判模型评估其他LLM生成的摘要主题一致性的有效性。我们利用Anthropic的Claude模型从开放式的调查问卷响应中生成主题摘要，而Amazon的Titan Express、Nova Pro以及Meta的Llama作为LLM裁判。LLM作为裁判的方法与使用Cohen's kappa、Spearman's rho和Krippendorff's alpha进行的人类评估进行了比较，验证了一种可扩展的替代传统人类中心评估方法的方案。研究发现，虽然作为裁判的LLM提供了一个与人类评分者相当的可扩展解决方案，但人类可能仍然在检测细微的具体上下文差异方面表现出色。本研究为AI辅助文本分析的知识体系增加了新的内容。我们讨论了限制并提出了未来研究的建议，强调在不同上下文和应用场景中推广LLM裁判模型时需要仔细考虑。 

---
# I Can Find You in Seconds! Leveraging Large Language Models for Code Authorship Attribution 

**Title (ZH)**: 我可以在秒内找到你！利用大规模语言模型进行代码作者ship归属分析 

**Authors**: Soohyeon Choi, Yong Kiam Tan, Mark Huasong Meng, Mohamed Ragab, Soumik Mondal, David Mohaisen, Khin Mi Mi Aung  

**Link**: [PDF](https://arxiv.org/pdf/2501.08165)  

**Abstract**: Source code authorship attribution is important in software forensics, plagiarism detection, and protecting software patch integrity. Existing techniques often rely on supervised machine learning, which struggles with generalization across different programming languages and coding styles due to the need for large labeled datasets. Inspired by recent advances in natural language authorship analysis using large language models (LLMs), which have shown exceptional performance without task-specific tuning, this paper explores the use of LLMs for source code authorship attribution.
We present a comprehensive study demonstrating that state-of-the-art LLMs can successfully attribute source code authorship across different languages. LLMs can determine whether two code snippets are written by the same author with zero-shot prompting, achieving a Matthews Correlation Coefficient (MCC) of 0.78, and can attribute code authorship from a small set of reference code snippets via few-shot learning, achieving MCC of 0.77. Additionally, LLMs show some adversarial robustness against misattribution attacks.
Despite these capabilities, we found that naive prompting of LLMs does not scale well with a large number of authors due to input token limitations. To address this, we propose a tournament-style approach for large-scale attribution. Evaluating this approach on datasets of C++ (500 authors, 26,355 samples) and Java (686 authors, 55,267 samples) code from GitHub, we achieve classification accuracy of up to 65% for C++ and 68.7% for Java using only one reference per author. These results open new possibilities for applying LLMs to code authorship attribution in cybersecurity and software engineering. 

**Abstract (ZH)**: 源代码作者归属在软件取证、剽窃检测和保护软件补丁完整性方面具有重要意义。现有的技术方法通常依赖监督机器学习，但在不同编程语言和编程风格之间泛化时往往会遇到困难，因为需要大量标记的数据集。受近期大型语言模型（LLMs）在自然语言作者归属分析中表现出色且无需特定任务微调的进展启发，本文探索了使用LLMs进行源代码作者归属的方法。

我们进行了全面的研究，证明了最先进的LLMs能够成功地在不同编程语言之间进行源代码作者归属。LLMs可以通过零样本提示判断两个代码片段是否由同一个作者编写，达到了0.78的Matthews相关系数（MCC），并通过少量样本学习将代码作者归属从一小部分参考代码片段中进行分类，达到了0.77的MCC。此外，LLMs还表现出一定程度的对抗鲁棒性，能够抵御误归属攻击。

尽管具有这些能力，我们发现简单的LLMs提示方法在大量作者的情况下无法很好地扩展，这主要是由于输入标记的限制。为了解决这个问题，我们提出了一种锦标赛式的方法来实现大范围的归属。在来自GitHub的C++（500位作者，26,355个样本）和Java（686位作者，55,267个样本）代码的数据集上进行评估，通过每个作者仅使用一个参考代码片段，实现了高达65%的C++分类准确率和68.7%的Java分类准确率。这些结果为在网络安全和软件工程中应用LLMs进行代码作者归属提供了新的可能性。 

---
# FairTTTS: A Tree Test Time Simulation Method for Fairness-Aware Classification 

**Title (ZH)**: FairTTTS：一种用于公平性意识分类的树测试时模拟方法 

**Authors**: Nurit Cohen-Inger, Lior Rokach, Bracha Shapira, Seffi Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2501.08155)  

**Abstract**: Algorithmic decision-making has become deeply ingrained in many domains, yet biases in machine learning models can still produce discriminatory outcomes, often harming unprivileged groups. Achieving fair classification is inherently challenging, requiring a careful balance between predictive performance and ethical considerations. We present FairTTTS, a novel post-processing bias mitigation method inspired by the Tree Test Time Simulation (TTTS) method. Originally developed to enhance accuracy and robustness against adversarial inputs through probabilistic decision-path adjustments, TTTS serves as the foundation for FairTTTS. By building on this accuracy-enhancing technique, FairTTTS mitigates bias and improves predictive performance. FairTTTS uses a distance-based heuristic to adjust decisions at protected attribute nodes, ensuring fairness for unprivileged samples. This fairness-oriented adjustment occurs as a post-processing step, allowing FairTTTS to be applied to pre-trained models, diverse datasets, and various fairness metrics without retraining. Extensive evaluation on seven benchmark datasets shows that FairTTTS outperforms traditional methods in fairness improvement, achieving a 20.96% average increase over the baseline compared to 18.78% for related work, and further enhances accuracy by 0.55%. In contrast, competing methods typically reduce accuracy by 0.42%. These results confirm that FairTTTS effectively promotes more equitable decision-making while simultaneously improving predictive performance. 

**Abstract (ZH)**: 算法决策已经深入许多领域，但机器学习模型中的偏见仍然可能导致歧视性结果，往往伤害到弱势群体。实现公平分类本身就极具挑战性，需要在预测性能和伦理考量之间找到精细的平衡。我们提出了FairTTTS，这是一种受树测试时仿真（TTTS）方法启发的新颖后处理偏见缓解方法。TTTS最初是为通过概率决策路径调整来提高准确性和对对抗输入的鲁棒性而开发的，FairTTTS以此为基础进行改进。通过在提高准确性的技术上进行改进，FairTTTS减轻了偏见并提高了预测性能。FairTTTS 使用基于距离的启发式方法调整受保护属性节点的决策，确保对弱势样本的公平性。这种以公平为导向的调整发生在后处理步骤中，使得FairTTTS可以应用于预训练模型、多样化的数据集以及各种公平性指标，而无需重新训练。在七个基准数据集上的广泛评估表明，FairTTTS 在公平性提升方面优于传统方法，相比基线实现了20.96%的平均提升，而相关工作的提升为18.78%，同时进一步提高了0.55%的准确率。相比之下，竞争对手方法通常会降低0.42%的准确率。这些结果证实，FairTTTS 既能有效促进更加公平的决策，又能同时提高预测性能。 

---
# Refusal Behavior in Large Language Models: A Nonlinear Perspective 

**Title (ZH)**: 大型语言模型中的拒绝行为：一种非线性视角 

**Authors**: Fabian Hildebrandt, Andreas Maier, Patrick Krauss, Achim Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2501.08145)  

**Abstract**: Refusal behavior in large language models (LLMs) enables them to decline responding to harmful, unethical, or inappropriate prompts, ensuring alignment with ethical standards. This paper investigates refusal behavior across six LLMs from three architectural families. We challenge the assumption of refusal as a linear phenomenon by employing dimensionality reduction techniques, including PCA, t-SNE, and UMAP. Our results reveal that refusal mechanisms exhibit nonlinear, multidimensional characteristics that vary by model architecture and layer. These findings highlight the need for nonlinear interpretability to improve alignment research and inform safer AI deployment strategies. 

**Abstract (ZH)**: 大型语言模型（LLMs）的拒绝行为使它们能够在面对有害、不道德或不适当的问题时避免回应，从而确保与伦理标准的一致性。本文探讨了六大语言模型（来自三种架构家族）中的拒绝行为。我们采用降维技术，包括主成分分析（PCA）、t-分布随机邻近嵌入（t-SNE）和统一维度映射（UMAP），挑战拒绝行为是线性现象的假设。研究结果表明，拒绝机制表现出非线性和多维特征，这些特征在不同模型架构及其层之间有所差异。这些发现强调了需要采用非线性可解释性方法，以改进一致性研究并指导更安全的AI部署策略。 

---
# EEG-ReMinD: Enhancing Neurodegenerative EEG Decoding through Self-Supervised State Reconstruction-Primed Riemannian Dynamics 

**Title (ZH)**: EEG-ReMinD：通过自监督状态重建引导的黎曼动力学增强神经退行性疾病EEG解码 

**Authors**: Zirui Wang, Zhenxi Song, Yi Guo, Yuxin Liu, Guoyang Xu, Min Zhang, Zhiguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.08139)  

**Abstract**: The development of EEG decoding algorithms confronts challenges such as data sparsity, subject variability, and the need for precise annotations, all of which are vital for advancing brain-computer interfaces and enhancing the diagnosis of diseases. To address these issues, we propose a novel two-stage approach named Self-Supervised State Reconstruction-Primed Riemannian Dynamics (EEG-ReMinD) , which mitigates reliance on supervised learning and integrates inherent geometric features. This approach efficiently handles EEG data corruptions and reduces the dependency on labels. EEG-ReMinD utilizes self-supervised and geometric learning techniques, along with an attention mechanism, to analyze the temporal dynamics of EEG features within the framework of Riemannian geometry, referred to as Riemannian dynamics. Comparative analyses on both intact and corrupted datasets from two different neurodegenerative disorders underscore the enhanced performance of EEG-ReMinD. 

**Abstract (ZH)**: EEG解码算法的发展面临着数据稀疏性、个体差异以及精确注释的需要等挑战，这些因素对于推动脑机接口的进步和提高疾病的诊断能力至关重要。为了应对这些挑战，我们提出了一种新颖的两阶段方法，名为自监督状态重建-引导黎曼动力学（EEG-ReMinD），该方法减轻了对监督学习的依赖，并整合了固有的几何特征。该方法能够有效地处理EEG数据的篡改，并减少对标签的依赖。EEG-ReMinD利用自监督和几何学习技术，以及注意力机制，在黎曼几何框架内分析EEG特征的时间动态，称为黎曼动力学。通过两种不同的神经退行性疾病数据集（包括完整数据和篡改数据）进行的对比分析表明，EEG-ReMinD的性能得到增强。 

---
# An Empirical Wall-Pressure Spectrum Model for Aeroacoustic Predictions Based on Symbolic Regression 

**Title (ZH)**: 基于符号回归的壁压力谱模型的实验研究及其在声学预测中的应用 

**Authors**: Laura Botero Bolívar, David Huergo, Fernanda L. dos Santos, Cornelis H. Venner, Leandro D. de Santana, Esteban Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2501.08134)  

**Abstract**: Fast-turn around methods to predict airfoil trailing-edge noise are crucial for incorporating noise limitations into design optimization loops of several applications. Among these aeroacoustic predictive models, Amiet's theory offers the best balance between accuracy and simplicity. The accuracy of the model relies heavily on precise wall-pressure spectrum predictions, which are often based on single-equation formulations with adjustable parameters. These parameters are calibrated for particular airfoils and flow conditions and consequently tend to fail when applied outside their calibration range. This paper introduces a new wall-pressure spectrum empirical model designed to enhance the robustness and accuracy of current state-of-the-art predictions while widening the range of applicability of the model to different airfoils and flow conditions. The model is developed using AI-based symbolic regression via a genetic-algorithm-based approach, and applied to a dataset of wall-pressure fluctuations measured on NACA 0008 and NACA 63018 airfoils at multiple angles of attack and inflow velocities, covering turbulent boundary layers with both adverse and favorable pressure gradients. Validation against experimental data (outside the training dataset) demonstrates the robustness of the model compared to well-accepted semi-empirical models. Finally, the model is integrated with Amiet's theory to predict the aeroacoustic noise of a full-scale wind turbine, showing good agreement with experimental measurements. 

**Abstract (ZH)**: 快速实现的方法对于将噪声限制纳入多种应用的设计优化循环中预测机翼后缘噪声至关重要。在这类气动声学预测模型中，Amiet 理论提供了在精确性和简化性之间的最优平衡。模型的准确性很大程度上依赖于精确的壁压谱预测，这些预测通常基于具有可调参数的一方程形式。这些参数针对特定的机翼和流动条件进行了校准，因此在超出校准范围的情况下往往会产生失败。本文介绍了一种新的壁压谱经验模型，旨在增强当前最先进的预测的稳健性和准确性，同时扩展模型在不同机翼和流动条件下的适用范围。该模型使用基于遗传算法的AI 基础符号回归开发，并应用于不同迎角和来流速度下对 NACA 0008 和 NACA 63018 机翼壁压波动的数据集，涵盖了具有不利和有利压力梯度的湍流边界层。与广泛接受的经验半经验模型相比，模型在外推训练数据集的实验数据验证中显示出更高的稳健性。最后，将该模型与 Amiet 理论结合，预测大型风力发电机组的气动声噪声，并与实验测量结果取得了良好的一致。 

---
# Data-driven inventory management for new products: A warm-start and adjusted Dyna-$Q$ approach 

**Title (ZH)**: 基于数据的新产品库存管理：一种温启动和调整的Dyna-$Q$ 方法 

**Authors**: Xinyu Qu, Longxiao Liu, Wenjie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.08109)  

**Abstract**: In this paper, we propose a novel reinforcement learning algorithm for inventory management of newly launched products with no or limited historical demand information. The algorithm follows the classic Dyna-$Q$ structure, balancing the model-based and model-free approaches, while accelerating the training process of Dyna-$Q$ and mitigating the model discrepancy generated by the model-based feedback. Warm-start information from the demand data of existing similar products can be incorporated into the algorithm to further stabilize the early-stage training and reduce the variance of the estimated optimal policy. Our approach is validated through a case study of bakery inventory management with real data. The adjusted Dyna-$Q$ shows up to a 23.7\% reduction in average daily cost compared with $Q$-learning, and up to a 77.5\% reduction in training time within the same horizon compared with classic Dyna-$Q$. By incorporating the warm-start information, it can be found that the adjusted Dyna-$Q$ has the lowest total cost, lowest variance in total cost, and relatively low shortage percentages among all the algorithms under a 30-day testing. 

**Abstract (ZH)**: 在本文中，我们提出了一种新型强化学习算法，用于管理新上市产品的库存，这些产品没有或仅有有限的历史需求信息。该算法遵循经典的Dyna-$Q$结构，平衡了基于模型和非基于模型的方法，同时加速了Dyna-$Q$的训练过程，并减轻了基于模型的反馈产生的模型不一致问题。可以通过将现有类似产品的需求数据的预热信息整合到算法中，进一步稳定早期训练阶段并降低估计最优策略的方差。通过一个以实际数据为基础的面包库存管理案例研究验证了该方法。调整后的Dyna-$Q$相比QLearning平均每日成本降低了高达23.7%，相比经典Dyna-$Q$在相同时间段内的训练时间减少了高达77.5%。通过整合预热信息，调整后的Dyna-$Q$具有最低的总成本、最低的总成本方差以及在30天测试期内相对较低的缺货比例。 

---
# Consistency of Responses and Continuations Generated by Large Language Models on Social Media 

**Title (ZH)**: 大型语言模型在社交媒体上生成的回答和续写的一致性 

**Authors**: Wenlu Fan, Yuqi Zhu, Chenyang Wang, Bin Wang, Wentao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08102)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities in text generation, yet their emotional consistency and semantic coherence in social media contexts remain insufficiently understood. This study investigates how LLMs handle emotional content and maintain semantic relationships through continuation and response tasks using two open-source models: Gemma and Llama. By analyzing climate change discussions from Twitter and Reddit, we examine emotional transitions, intensity patterns, and semantic similarity between human-authored and LLM-generated content. Our findings reveal that while both models maintain high semantic coherence, they exhibit distinct emotional patterns: Gemma shows a tendency toward negative emotion amplification, particularly anger, while maintaining certain positive emotions like optimism. Llama demonstrates superior emotional preservation across a broader spectrum of affects. Both models systematically generate responses with attenuated emotional intensity compared to human-authored content and show a bias toward positive emotions in response tasks. Additionally, both models maintain strong semantic similarity with original texts, though performance varies between continuation and response tasks. These findings provide insights into LLMs' emotional and semantic processing capabilities, with implications for their deployment in social media contexts and human-AI interaction design. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在文本生成方面展示了卓越的能力，但在社交媒体语境下的情感一致性和语义连贯性方面仍然不够清楚。本研究通过使用两个开源模型Gemma和Llama，探讨了LLMs在延续任务和回应任务中处理情感内容和维持语义关系的方式。通过分析来自Twitter和Reddit的气候变化讨论，我们研究了人类撰写内容和LLM生成内容之间的情感转换、情感强度模式以及语义相似性。研究发现，尽管两种模型均保持了高语义连贯性，但它们在情感模式上表现出不同的特点：Gemma倾向于增强负面情绪，特别是愤怒情绪，但仍保留了一些积极情绪如乐观；Llama则在更广泛的情感谱系中表现出优越的情感保留能力。两种模型在生成回应时，情感强度普遍低于人类撰写的内容，并且在回应任务中倾向于使用积极情绪。此外，两种模型在延续任务和回应任务中的原始文本的语义相似度都很强，但性能在不同任务中有差异。这些发现为LLMs的情感和语义处理能力提供了见解，并对其在社交媒体环境中的应用和人机交互设计具有重要意义。 

---
# Guiding the classification of hepatocellular carcinoma on 3D CT-scans using deep and handcrafted radiological features 

**Title (ZH)**: 利用深度学习和手工制作的放射学特征指导三维CT扫描中肝细胞癌的分类 

**Authors**: E. Sarfati, A. Bône, M-M. Rohé, C. Aubé, M. Ronot, P. Gori, I. Bloch  

**Link**: [PDF](https://arxiv.org/pdf/2501.08097)  

**Abstract**: Hepatocellular carcinoma is the most spread primary liver cancer across the world ($\sim$80\% of the liver tumors). The gold standard for HCC diagnosis is liver biopsy. However, in the clinical routine, expert radiologists provide a visual diagnosis by interpreting hepatic CT-scans according to a standardized protocol, the LI-RADS, which uses five radiological criteria with an associated decision tree. In this paper, we propose an automatic approach to predict histology-proven HCC from CT images in order to reduce radiologists' inter-variability. We first show that standard deep learning methods fail to accurately predict HCC from CT-scans on a challenging database, and propose a two-step approach inspired by the LI-RADS system to improve the performance. We achieve improvements from 6 to 18 points of AUC with respect to deep learning baselines trained with different architectures. We also provide clinical validation of our method, achieving results that outperform non-expert radiologists and are on par with expert ones. 

**Abstract (ZH)**: 肝细胞癌是全世界发病率最高的原发性肝癌（约占肝肿瘤的约80%）。肝活检是肝细胞癌（HCC）诊断的金标准。然而，在临床实践中，专家放射师通过解读按照标准化协议（LI-RADS）的肝CT扫描图像来提供视觉诊断，该协议使用五个放射学标准及其相关的决策树。在本文中，我们提出了一种自动方法，从CT图像中预测组织学确诊的HCC，以减少放射师间的一致性差异。我们首先表明，标准的深度学习方法在具有挑战性的数据库中无法准确预测CT扫描中的HCC，并提出了一种受到LI-RADS系统启发的两步方法，以提高性能。与使用不同架构训练的不同深度学习基线相比，我们实现了AUC分数从6到18点的提升。我们还提供了临床验证，表明我们的方法在非专家放射师的表现上更好，并且与专家放射师的表现相当。 

---
# Hybrid Action Based Reinforcement Learning for Multi-Objective Compatible Autonomous Driving 

**Title (ZH)**: 基于混合动作强化学习的多目标兼容自主驾驶 

**Authors**: Guizhe Jin, Zhuoren Li, Bo Leng, Wei Han, Lu Xiong, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.08096)  

**Abstract**: Reinforcement Learning (RL) has shown excellent performance in solving decision-making and control problems of autonomous driving, which is increasingly applied in diverse driving scenarios. However, driving is a multi-attribute problem, leading to challenges in achieving multi-objective compatibility for current RL methods, especially in both policy execution and policy iteration. On the one hand, the common action space structure with single action type limits driving flexibility or results in large behavior fluctuations during policy execution. On the other hand, the multi-attribute weighted single reward function result in the agent's disproportionate attention to certain objectives during policy iterations. To this end, we propose a Multi-objective Ensemble-Critic reinforcement learning method with Hybrid Parametrized Action for multi-objective compatible autonomous driving. Specifically, a parameterized action space is constructed to generate hybrid driving actions, combining both abstract guidance and concrete control commands. A multi-objective critics architecture is constructed considering multiple attribute rewards, to ensure simultaneously focusing on different driving objectives. Additionally, uncertainty-based exploration strategy is introduced to help the agent faster approach viable driving policy. The experimental results in both the simulated traffic environment and the HighD dataset demonstrate that our method can achieve multi-objective compatible autonomous driving in terms of driving efficiency, action consistency, and safety. It enhances the general performance of the driving while significantly increasing training efficiency. 

**Abstract (ZH)**: 强化学习（RL）在解决自主驾驶的决策和控制问题方面表现出卓越的性能，并且正越来越多地应用于各种驾驶场景中。然而，驾驶任务是一个多属性问题，这导致了当前RL方法在实现多目标兼容性方面的挑战，尤其是在策略执行和策略迭代过程中。一方面，共同的行动空间结构以单一的行动类型限制了驾驶的灵活性，或者导致策略执行过程中的行为剧烈变化。另一方面，带权重的多属性单一奖励函数使得代理在策略迭代过程中对某些目标过度关注。为了解决这些问题，我们提出了一种基于混合参数化行动的多目标集成价值函数的强化学习方法，以实现多目标兼容的自主驾驶。具体而言，构建了一个参数化的行动空间来生成结合抽象指导和具体控制命令的混合驾驶行动。同时，考虑到多个属性奖励，构建了多目标价值函数结构，以确保策略迭代过程中同时关注不同的驾驶目标。此外，引入了基于不确定性探索策略，以帮助代理更快地接近可行的驾驶策略。在模拟交通环境和HighD数据集的实验结果表明，我们的方法在驾驶效率、动作一致性以及安全性方面均实现了多目标兼容的自主驾驶。该方法在提高驾驶整体性能的同时，显著提高了训练效率。 

---
# Hierarchical Autoscaling for Large Language Model Serving with Chiron 

**Title (ZH)**: 使用Chiron的分层级自适应缩放技术优化大规模语言模型服务 

**Authors**: Archit Patke, Dhemath Reddy, Saurabh Jha, Chandra Narayanaswami, Zbigniew Kalbarczyk, Ravishankar Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2501.08090)  

**Abstract**: Large language model (LLM) serving is becoming an increasingly important workload for cloud providers. Based on performance SLO requirements, LLM inference requests can be divided into (a) interactive requests that have tight SLOs in the order of seconds, and (b) batch requests that have relaxed SLO in the order of minutes to hours. These SLOs can degrade based on the arrival rates, multiplexing, and configuration parameters, thus necessitating the use of resource autoscaling on serving instances and their batch sizes. However, previous autoscalers for LLM serving do not consider request SLOs leading to unnecessary scaling and resource under-utilization. To address these limitations, we introduce Chiron, an autoscaler that uses the idea of hierarchical backpressure estimated using queue size, utilization, and SLOs. Our experiments show that Chiron achieves up to 90% higher SLO attainment and improves GPU efficiency by up to 70% compared to existing solutions. 

**Abstract (ZH)**: 大型语言模型（LLM）服务正在成为云计算提供商越来越重要的工作负载。基于性能服务水平目标（SLO）要求，LLM 推理请求可以分为两类：（a）具有秒级紧密SLO的交互请求，以及（b）具有分钟到小时级松散SLO的批量请求。这些SLO可能会根据请求到达率、复用和配置参数而退化，因此需要在服务实例及其批次大小上使用资源自动伸缩。然而，之前的LLM服务自动扩展器并未考虑请求SLO，导致不必要的扩展和资源利用不足。为解决这些局限性，我们提出了Chiron，这是一种使用层级反压思想进行自动伸缩的技术，该技术利用队列大小、利用率和SLO来估算反压。实验结果显示，与现有解决方案相比，Chiron 可以实现高达90%的更高SLO达成率，并可提高高达70%的GPU效率。 

---
# Optimizing Speech Multi-View Feature Fusion through Conditional Computation 

**Title (ZH)**: 通过条件计算优化语音多视图特征融合 

**Authors**: Weiqiao Shan, Yuhao Zhang, Yuchen Han, Bei Li, Xiaofeng Zhao, Yuang Li, Min Zhang, Hao Yang, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.08057)  

**Abstract**: Recent advancements have highlighted the efficacy of self-supervised learning (SSL) features in various speech-related tasks, providing lightweight and versatile multi-view speech representations. However, our study reveals that while SSL features expedite model convergence, they conflict with traditional spectral features like FBanks in terms of update directions. In response, we propose a novel generalized feature fusion framework grounded in conditional computation, featuring a gradient-sensitive gating network and a multi-stage dropout strategy. This framework mitigates feature conflicts and bolsters model robustness to multi-view input features. By integrating SSL and spectral features, our approach accelerates convergence and maintains performance on par with spectral models across multiple speech translation tasks on the MUSTC dataset. 

**Abstract (ZH)**: 近年来，自我监督学习（SSL）特征在多种语音相关任务中的有效性得到了突出，提供了轻量且多视角的语音表示。然而，我们的研究显示，尽管SSL特征能够加速模型收敛，但在更新方向上它们与传统的频谱特征（如FBanks）存在冲突。为解决这一问题，我们提出了一种基于条件计算的新颖泛化特征融合框架，该框架包含梯度敏感门控网络和多阶段 Dropout 策略。该框架减轻了特征冲突并增强了模型对多视角输入特征的鲁棒性。通过结合SSL和频谱特征，我们的方法在MUSTC数据集上多种语音翻译任务中加速了收敛，并且在性能上与频谱模型保持一致。 

---
# Exploring Narrative Clustering in Large Language Models: A Layerwise Analysis of BERT 

**Title (ZH)**: 探索大规模语言模型中的叙事聚类：BERT 的逐层分析 

**Authors**: Awritrojit Banerjee, Achim Schilling, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2501.08053)  

**Abstract**: This study investigates the internal mechanisms of BERT, a transformer-based large language model, with a focus on its ability to cluster narrative content and authorial style across its layers. Using a dataset of narratives developed via GPT-4, featuring diverse semantic content and stylistic variations, we analyze BERT's layerwise activations to uncover patterns of localized neural processing. Through dimensionality reduction techniques such as Principal Component Analysis (PCA) and Multidimensional Scaling (MDS), we reveal that BERT exhibits strong clustering based on narrative content in its later layers, with progressively compact and distinct clusters. While strong stylistic clustering might occur when narratives are rephrased into different text types (e.g., fables, sci-fi, kids' stories), minimal clustering is observed for authorial style specific to individual writers. These findings highlight BERT's prioritization of semantic content over stylistic features, offering insights into its representational capabilities and processing hierarchy. This study contributes to understanding how transformer models like BERT encode linguistic information, paving the way for future interdisciplinary research in artificial intelligence and cognitive neuroscience. 

**Abstract (ZH)**: 本研究探讨了基于变压器的大型语言模型BERT的内部机制，重点关注其在各层中聚集叙事内容和作者风格的能力。我们使用GPT-4开发的数据集进行分析，该数据集包含多样化的语义内容和风格变化。通过分析BERT在各层的激活情况，我们揭示了局部神经处理的模式。利用主成分分析（PCA）和多维缩放（MDS）等降维技术，我们发现BERT在其后期层中表现出强烈的基于叙事内容的聚类，聚类逐渐变得更加紧凑和分明。尽管当叙事重新表达为不同的文本类型（例如寓言、科幻、儿童故事）时，风格聚类可能较强，但对于个别作者的独特风格聚类现象较少。这些发现突出了BERT在处理语义内容方面优先于风格特征的重要性，从而对其实现能力和处理层次提供了见解。本研究有助于理解如BERT这样的变换器模型如何编码语言信息，为未来跨学科研究人工智能和认知神经科学奠定了基础。 

---
# Building Symbiotic AI: Reviewing the AI Act for a Human-Centred, Principle-Based Framework 

**Title (ZH)**: 构建共生人工智能：回顾《人工智能法案》以构建以人为本、基于原则的框架 

**Authors**: Miriana Calvano, Antonio Curci, Giuseppe Desolda, Andrea Esposito, Rosa Lanzilotti, Antonio Piccinno  

**Link**: [PDF](https://arxiv.org/pdf/2501.08046)  

**Abstract**: Artificial Intelligence (AI) spreads quickly as new technologies and services take over modern society. The need to regulate AI design, development, and use is strictly necessary to avoid unethical and potentially dangerous consequences to humans. The European Union (EU) has released a new legal framework, the AI Act, to regulate AI by undertaking a risk-based approach to safeguard humans during interaction. At the same time, researchers offer a new perspective on AI systems, commonly known as Human-Centred AI (HCAI), highlighting the need for a human-centred approach to their design. In this context, Symbiotic AI (a subtype of HCAI) promises to enhance human capabilities through a deeper and continuous collaboration between human intelligence and AI. This article presents the results of a Systematic Literature Review (SLR) that aims to identify principles that characterise the design and development of Symbiotic AI systems while considering humans as the core of the process. Through content analysis, four principles emerged from the review that must be applied to create Human-Centred AI systems that can establish a symbiotic relationship with humans. In addition, current trends and challenges were defined to indicate open questions that may guide future research for the development of SAI systems that comply with the AI Act. 

**Abstract (ZH)**: 人工智能（AI）随着新技术和服务的兴起而迅速发展，正在现代社会中普及。为了避免对人类产生不道德且潜在危险的影响，对AI的设计、开发和使用进行有效监管刻不容缓。欧盟（EU）提出了新的法律框架《AI法案》，通过风险为基础的方法来保障人类在与AI互动过程中的安全。同时，研究人员提出了人类中心的人工智能（Human-Centred AI, HCAI）的新视角，强调在设计过程中需要采取人类中心的方法。在此背景下，共生人工智能（Symbiotic AI，一种HCAI的子类型）通过深化并持续人类智能与AI的合作，旨在提升人类的能力。本文通过系统文献综述（SLR）研究，旨在识别并定义在将人类置于核心过程之中时设计和发展共生人工智能系统的原则。通过内容分析，总结出四项原则，这些原则应应用于创建能够与人类建立共生关系的人类中心AI系统。此外，还定义了当前趋势与挑战，指出了未来研究中可能存在的开放性问题，并为开发符合《AI法案》要求的Symbiotic AI系统提供了指导方向。 

---
# Exploring visual language models as a powerful tool in the diagnosis of Ewing Sarcoma 

**Title (ZH)**: 探索视觉语言模型在尤文氏肉瘤诊断中的强大工具作用 

**Authors**: Alvaro Pastor-Naranjo, Pablo Meseguer, Rocío del Amor, Jose Antonio Lopez-Guerrero, Samuel Navarro, Katia Scotlandi, Antonio Llombart-Bosch, Isidro Machado, Valery Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2501.08042)  

**Abstract**: Ewing's sarcoma (ES), characterized by a high density of small round blue cells without structural organization, presents a significant health concern, particularly among adolescents aged 10 to 19. Artificial intelligence-based systems for automated analysis of histopathological images are promising to contribute to an accurate diagnosis of ES. In this context, this study explores the feature extraction ability of different pre-training strategies for distinguishing ES from other soft tissue or bone sarcomas with similar morphology in digitized tissue microarrays for the first time, as far as we know. Vision-language supervision (VLS) is compared to fully-supervised ImageNet pre-training within a multiple instance learning paradigm. Our findings indicate a substantial improvement in diagnostic accuracy with the adaption of VLS using an in-domain dataset. Notably, these models not only enhance the accuracy of predicted classes but also drastically reduce the number of trainable parameters and computational costs. 

**Abstract (ZH)**: 尤因氏肉瘤（ES）由高密度的小圆蓝细胞组成，这些细胞缺乏结构组织，尤其对10至19岁的青少年构成了重要的健康威胁。基于人工 Intelligence 的系统对病理学图像进行自动化分析，有望为ES的确诊提供准确的诊断。在此背景下，本研究首次探索了不同预训练策略在区分具有相似形态的ES与其他软组织或骨肉瘤方面的能力，使用数字化组织微阵列。据我们所知，本研究通过视觉-语言监督（VLS）与完全监督的ImageNet预训练进行比较，研究在多实例学习框架下的表现。我们的研究结果表明，通过使用领域内数据集进行VLS能显著提高诊断准确性。值得注意的是，这些模型不仅提升了预测类别的准确性，还大幅减少了可训练参数的数量和计算成本。 

---
# READ: Reinforcement-based Adversarial Learning for Text Classification with Limited Labeled Data 

**Title (ZH)**: 基于强化学习的对抗性学习方法在标注数据有限条件下的文本分类 

**Authors**: Rohit Sharma, Shanu Kumar, Avinash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2501.08035)  

**Abstract**: Pre-trained transformer models such as BERT have shown massive gains across many text classification tasks. However, these models usually need enormous labeled data to achieve impressive performances. Obtaining labeled data is often expensive and time-consuming, whereas collecting unlabeled data using some heuristics is relatively much cheaper for any task. Therefore, this paper proposes a method that encapsulates reinforcement learning-based text generation and semi-supervised adversarial learning approaches in a novel way to improve the model's performance. Our method READ, Reinforcement-based Adversarial learning, utilizes an unlabeled dataset to generate diverse synthetic text through reinforcement learning, improving the model's generalization capability using adversarial learning. Our experimental results show that READ outperforms the existing state-of-art methods on multiple datasets. 

**Abstract (ZH)**: 预训练的变换器模型，如BERT，在多种文本分类任务中取得了显著的进步。然而，这些模型通常需要大量的带标签数据才能达到出色的性能。获取带标签数据往往成本高昂且耗时，而通过某些启发式方法收集未标数据则相对便宜得多。因此，本文提出了一种新颖的方法，该方法结合了基于强化学习的文本生成和半监督对抗学习，以提高模型的性能。我们的方法READ（Reinforcement-based Adversarial Learning，基于强化学习的对抗学习）利用未标数据生成多样化的合成文本，通过对抗学习提高模型的泛化能力。我们的实验结果显示，READ在多个数据集上优于现有最先进的方法。 

---
# An AI-driven framework for rapid and localized optimizations of urban open spaces 

**Title (ZH)**: 基于AI的快速局部优化城市开放空间框架 

**Authors**: Pegah Eshraghi, Arman Nikkhah Dehnavi, Maedeh Mirdamadi, Riccardo Talami, Zahra-Sadat Zomorodian  

**Link**: [PDF](https://arxiv.org/pdf/2501.08019)  

**Abstract**: As urbanization accelerates, open spaces are increasingly recognized for their role in enhancing sustainability and well-being, yet they remain underexplored compared to built spaces. This study introduces an AI-driven framework that integrates machine learning models (MLMs) and explainable AI techniques to optimize Sky View Factor (SVF) and visibility, key spatial metrics influencing thermal comfort and perceived safety in urban spaces. Unlike global optimization methods, which are computationally intensive and impractical for localized adjustments, this framework supports incremental design improvements with lower computational costs and greater flexibility. The framework employs SHapley Adaptive Explanations (SHAP) to analyze feature importance and Counterfactual Explanations (CFXs) to propose minimal design changes. Simulations tested five MLMs, identifying XGBoost as the most accurate, with building width, park area, and heights of surrounding buildings as critical for SVF, and distances from southern buildings as key for visibility. Compared to Genetic Algorithms, which required approximately 15/30 minutes across 3/4 generations to converge, the tested CFX approach achieved optimized results in 1 minute with a 5% RMSE error, demonstrating significantly faster performance and suitability for scalable retrofitting strategies. This interpretable and computationally efficient framework advances urban performance optimization, providing data-driven insights and practical retrofitting solutions for enhancing usability and environmental quality across diverse urban contexts. 

**Abstract (ZH)**: 随着城市化进程的加速，开放空间逐渐被认可为提高可持续性和福祉的关键因素，然而与建成空间相比，开放空间的研究仍然较为不足。本研究引入了一种基于人工智能（AI）的框架，该框架结合了机器学习模型（MLMs）和可解释的人工智能技术，以优化天视因子（SVF）和视野，这是影响城市空间热舒适性和感知安全性的关键空间指标。与计算成本高昂且不适用于局部调整的全局优化方法不同，该框架能够以较低的计算成本和更大的灵活性支持逐步的设计改进。该框架采用了SHapley 自适应解释（SHAP）来分析特征重要性，并使用反事实解释（CFXs）来提出最小的设计改进方案。模拟测试了五种MLMs，结果显示XGBoost对SVF的预测精度最高，建筑物宽度、公园面积以及周围建筑物的高度是SVF的关键因素，而南向建筑物的距离是视野的关键影响因素。与需要大约15-30分钟来跨3-4代迭代以达到收敛的遗传算法（GA）相比，测试的CFX方法在1分钟内实现了优化结果，且均方根误差（RMSE）误差仅为5%，证明其具有显著更快的性能，并适用于可扩展的翻新策略。这种可解释且计算高效的框架促进了城市性能优化，通过提供数据驱动的见解和实用的翻新解决方案，提升了不同城市环境中空间的可使用性和环境质量。 

---
# Tutorial: VAE as an inference paradigm for neuroimaging 

**Title (ZH)**: 教程：VAE作为神经影像学推断范式的应用 

**Authors**: C. Vázquez-García, F. J. Martínez-Murcia, F. Segovia Román, Juan M. Górriz Sáez  

**Link**: [PDF](https://arxiv.org/pdf/2501.08009)  

**Abstract**: In this tutorial, we explore Variational Autoencoders (VAEs), an essential framework for unsupervised learning, particularly suited for high-dimensional datasets such as neuroimaging. By integrating deep learning with Bayesian inference, VAEs enable the generation of interpretable latent representations. This tutorial outlines the theoretical foundations of VAEs, addresses practical challenges such as convergence issues and over-fitting, and discusses strategies like the reparameterization trick and hyperparameter optimization. We also highlight key applications of VAEs in neuroimaging, demonstrating their potential to uncover meaningful patterns, including those associated with neurodegenerative processes, and their broader implications for analyzing complex brain data. 

**Abstract (ZH)**: 在本文献教程中，我们将探讨变分自编码器（VAEs），这是一种适用于无监督学习的关键框架，特别适用于高维数据集，如神经影像数据。通过将深度学习与贝叶斯推断相结合，VAEs 能够生成可解释的潜在表示。本文献教程概述了 VAEs 的理论基础，探讨了诸如收敛问题和过拟合等实际挑战，并讨论了诸如重参数化技巧和超参数优化等策略。我们还将强调 VAEs 在神经影像学中的关键应用，展示它们在揭示有意义的模式方面（包括与神经退行性过程相关的模式）的潜力，以及它们对分析复杂脑数据的更广泛影响。 

---
# TriAdaptLoRA: Brain-Inspired Triangular Adaptive Low-Rank Adaptation for Parameter-Efficient Fine-Tuning 

**Title (ZH)**: TriAdaptLoRA：受脑启发的三角自适应低秩适应方法及其在参数高效微调中的应用 

**Authors**: Yao Liang, Yuwei Wang, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2501.08008)  

**Abstract**: The fine-tuning of Large Language Models (LLMs) is pivotal for achieving optimal performance across diverse downstream tasks. However, while full fine-tuning delivers superior results, it entails significant computational and resource costs. Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA, address these challenges by reducing the number of trainable parameters, but they often struggle with rank adjustment efficiency and task-specific adaptability. We propose Triangular Adaptive Low-Rank Adaptation (TriAdaptLoRA), a novel PEFT framework inspired by neuroscience principles, which dynamically optimizes the allocation of trainable parameters. TriAdaptLoRA introduces three key innovations: 1) a triangular split of transformation matrices into lower and upper triangular components to maximize parameter utilization, 2) a parameter importance metric based on normalized Frobenius norms for efficient adaptation, and 3) an adaptive rank-growth strategy governed by dynamic thresholds, allowing flexible parameter allocation across training steps. Experiments conducted on a variety of natural language understanding and generation tasks demonstrate that TriAdaptLoRA consistently outperforms existing PEFT methods. It achieves superior performance, enhanced stability, and reduced computational overhead, particularly under linear threshold-driven rank growth. These results highlight its efficacy as a scalable and resource-efficient solution for fine-tuning LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的微调对于实现各种下游任务的最优性能至关重要。然而，尽管全面微调可以提供更好的结果，但它也会带来显著的计算和资源成本。参数高效微调（PEFT）方法，如LoRA，通过减少可训练参数的数量来应对这些挑战，但它们常常在秩调整效率和任务特定适应性方面存在问题。我们提出了Triangular Adaptive Low-Rank Adaptation（TriAdaptLoRA），这是一种受到神经科学原理启发的新型PEFT框架，可以动态优化可训练参数的分配。TriAdaptLoRA引入了三项关键创新：1）通过将变换矩阵三角分割为下三角和上三角成分来最大化参数利用；2）基于归一化Frobenius范数的参数重要性度量以实现高效的适应；3）由动态阈值控制的自适应秩增长策略，允许在训练步骤中灵活分配参数。在各种自然语言理解与生成任务上的实验表明，TriAdaptLoRA在现有PEFT方法中表现出更优的性能、更好的稳定性以及更低的计算开销，特别是在线性阈值驱动的秩增长下表现尤为突出。这些结果突显了它作为一种可扩展且资源高效的LLM微调解决方案的有效性。 

---
# DisCoPatch: Batch Statistics Are All You Need For OOD Detection, But Only If You Can Trust Them 

**Title (ZH)**: DisCoPatch: 批量统计即为异常检测所需的一切，但前提是这些统计必须可信 

**Authors**: Francisco Caetano, Christiaan Viviers, Luis A. Zavala-Mondragón, Peter H. N. de With, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2501.08005)  

**Abstract**: Out-of-distribution (OOD) detection holds significant importance across many applications. While semantic and domain-shift OOD problems are well-studied, this work focuses on covariate shifts - subtle variations in the data distribution that can degrade machine learning performance. We hypothesize that detecting these subtle shifts can improve our understanding of in-distribution boundaries, ultimately improving OOD detection. In adversarial discriminators trained with Batch Normalization (BN), real and adversarial samples form distinct domains with unique batch statistics - a property we exploit for OOD detection. We introduce DisCoPatch, an unsupervised Adversarial Variational Autoencoder (VAE) framework that harnesses this mechanism. During inference, batches consist of patches from the same image, ensuring a consistent data distribution that allows the model to rely on batch statistics. DisCoPatch uses the VAE's suboptimal outputs (generated and reconstructed) as negative samples to train the discriminator, thereby improving its ability to delineate the boundary between in-distribution samples and covariate shifts. By tightening this boundary, DisCoPatch achieves state-of-the-art results in public OOD detection benchmarks. The proposed model not only excels in detecting covariate shifts, achieving 95.5% AUROC on ImageNet-1K(-C) but also outperforms all prior methods on public Near-OOD (95.0%) benchmarks. With a compact model size of 25MB, it achieves high OOD detection performance at notably lower latency than existing methods, making it an efficient and practical solution for real-world OOD detection applications. The code will be made publicly available 

**Abstract (ZH)**: 下列论文内容或标题已翻译成中文，符合学术规范：

异区分布（OOD）检测在许多应用中具有重要意义。虽然语义和领域转移的OOD问题已经被广泛研究，但本工作重点研究协变量偏移（covariate shifts）——数据分布中的细微变化，这些变化可能导致机器学习性能下降。我们认为，检测这些细微的变化有助于更深入地理解区分布边界，从而提高OOD检测的效果。在使用批量规范化（Batch Normalization, BN）进行训练的对抗判别器中，真实的样本和生成的样本形成了具有独特批量统计特征的独立领域——这一特性被我们利用来进行OOD检测。我们提出了DisCoPatch，一种基于无监督对抗变分自编码器（VAE）框架的机制。在推理过程中，批量数据由同一图像的补丁组成，确保数据分布的一致性，使模型能够依赖批量统计特征。DisCoPatch 使用VAE的亚最优输出（生成和重建的样本）作为负面样本来训练判别器，从而提高其区分区分布样本和协变量偏移边界的能力。通过收紧这个边界，DisCoPatch 在公开的OOD检测基准测试中取得了最先进的结果。所提出的模型不仅在检测协变量偏移方面表现出色，对ImageNet-1K(-C)的数据，AUROC达到95.5%，而且在公开的Near-OOD（95.0%）基准测试中也优于所有先前的方法。凭借紧凑的25MB模型大小，它在较低的延迟下实现了高效的OOD检测性能，使其成为现实世界OOD检测应用中的高效且实用的解决方案。代码将公开提供。 

---
# Maximizing Uncertainty for Federated learning via Bayesian Optimisation-based Model Poisoning 

**Title (ZH)**: 通过基于贝叶斯优化的模型污染最大化联邦学习中的不确定性 

**Authors**: Marios Aristodemou, Xiaolan Liu, Yuan Wang, Konstantinos G. Kyriakopoulos, Sangarapillai Lambotharan, Qingsong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.08002)  

**Abstract**: As we transition from Narrow Artificial Intelligence towards Artificial Super Intelligence, users are increasingly concerned about their privacy and the trustworthiness of machine learning (ML) technology. A common denominator for the metrics of trustworthiness is the quantification of uncertainty inherent in DL algorithms, and specifically in the model parameters, input data, and model predictions. One of the common approaches to address privacy-related issues in DL is to adopt distributed learning such as federated learning (FL), where private raw data is not shared among users. Despite the privacy-preserving mechanisms in FL, it still faces challenges in trustworthiness. Specifically, the malicious users, during training, can systematically create malicious model parameters to compromise the models predictive and generative capabilities, resulting in high uncertainty about their reliability. To demonstrate malicious behaviour, we propose a novel model poisoning attack method named Delphi which aims to maximise the uncertainty of the global model output. We achieve this by taking advantage of the relationship between the uncertainty and the model parameters of the first hidden layer of the local model. Delphi employs two types of optimisation , Bayesian Optimisation and Least Squares Trust Region, to search for the optimal poisoned model parameters, named as Delphi-BO and Delphi-LSTR. We quantify the uncertainty using the KL Divergence to minimise the distance of the predictive probability distribution towards an uncertain distribution of model output. Furthermore, we establish a mathematical proof for the attack effectiveness demonstrated in FL. Numerical results demonstrate that Delphi-BO induces a higher amount of uncertainty than Delphi-LSTR highlighting vulnerability of FL systems to model poisoning attacks. 

**Abstract (ZH)**: 随着我们从窄人工智能向超人工智能过渡，用户越来越关注自己的隐私以及机器学习（ML）技术的可信度。衡量可信度的共同标准之一是对深度学习（DL）算法内在不确定性进行量化，特别是对模型参数、输入数据和模型预测的不确定性进行量化。为解决与隐私相关的问题，一种常见的方法是采用分布学习，如联邦学习（FL），这种方法确保用户之间的原始私有数据不会共享。尽管FL具有保护隐私的机制，但它仍然面临可信度方面的问题。具体来说，在训练过程中，恶意用户可以系统地创建恶意模型参数，以破坏模型的预测和生成能力，从而导致对模型可靠性高度不确定的认识。为了演示恶意行为，我们提出了一种名为Delphi的新模型中毒攻击方法，其目标是最大化全局模型输出的不确定性。通过利用本地模型第一隐藏层的模型参数与不确定性的关系，Delphi采用了两种优化方法：贝叶斯优化（BO）和最小二乘信赖域（LSTR），分别称为Delphi-BO和Delphi-LSTR。我们使用KL散度来量化不确定性，以最小化预测概率分布与不确定的模型输出分布之间的距离。此外，我们还为FL中演示的攻击效果建立了数学证明。数值结果表明，Delphi-BO产生的不确定性比Delphi-LSTR更多，这表明FL系统对模型中毒攻击的脆弱性。 

---
# Training Hybrid Neural Networks with Multimode Optical Nonlinearities Using Digital Twins 

**Title (ZH)**: 使用数字孪生训练具有多种非线性模式的混合神经网络 

**Authors**: Ilker Oguz, Louis J. E. Suter, Jih-Liang Hsieh, Mustafa Yildirim, Niyazi Ulas Dinc, Christophe Moser, Demetri Psaltis  

**Link**: [PDF](https://arxiv.org/pdf/2501.07991)  

**Abstract**: The ability to train ever-larger neural networks brings artificial intelligence to the forefront of scientific and technical discoveries. However, their exponentially increasing size creates a proportionally greater demand for energy and computational hardware. Incorporating complex physical events in networks as fixed, efficient computation modules can address this demand by decreasing the complexity of trainable layers. Here, we utilize ultrashort pulse propagation in multimode fibers, which perform large-scale nonlinear transformations, for this purpose. Training the hybrid architecture is achieved through a neural model that differentiably approximates the optical system. The training algorithm updates the neural simulator and backpropagates the error signal over this proxy to optimize layers preceding the optical one. Our experimental results achieve state-of-the-art image classification accuracies and simulation fidelity. Moreover, the framework demonstrates exceptional resilience to experimental drifts. By integrating low-energy physical systems into neural networks, this approach enables scalable, energy-efficient AI models with significantly reduced computational demands. 

**Abstract (ZH)**: 不断训练更大规模的神经网络为人工智能技术在科学和技术发现中的前沿地位奠定了基础。然而，网络规模的指数级增长导致了对能源和计算硬件需求的同比例增加。将复杂的物理事件作为固定且高效的计算模块集成到网络中，可以通过减少可训练层的复杂性来应对这一需求。在此，我们利用在多模式光纤中传播的超短脉冲进行大规模非线性变换，以实现这一目标。通过对光系统进行可微近似，神经模型实现了混合架构的训练。训练算法更新神经模拟器，并通过该代理回传误差信号以优化光学层之前的层。我们的实验结果达到了最先进的图像分类准确率和仿真精度。此外，该框架对实验漂移表现出卓越的抗性。通过将低能耗物理系统集成到神经网络中，这种方法能够实现计算需求显著降低、可扩展的能源高效的人工智能模型。 

---
# GAC-Net_Geometric and attention-based Network for Depth Completion 

**Title (ZH)**: GAC-Net_基于几何和注意力机制的深度完成网络 

**Authors**: Kuang Zhu, Xingli Gan, Min Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.07988)  

**Abstract**: Depth completion is a key task in autonomous driving, aiming to complete sparse LiDAR depth measurements into high-quality dense depth maps through image guidance. However, existing methods usually treat depth maps as an additional channel of color images, or directly perform convolution on sparse data, failing to fully exploit the 3D geometric information in depth maps, especially with limited performance in complex boundaries and sparse areas. To address these issues, this paper proposes a depth completion network combining channel attention mechanism and 3D global feature perception (CGA-Net). The main innovations include: 1) Utilizing PointNet++ to extract global 3D geometric features from sparse depth maps, enhancing the scene perception ability of low-line LiDAR data; 2) Designing a channel-attention-based multimodal feature fusion module to efficiently integrate sparse depth, RGB images, and 3D geometric features; 3) Combining residual learning with CSPN++ to optimize the depth refinement stage, further improving the completion quality in edge areas and complex scenes. Experiments on the KITTI depth completion dataset show that CGA-Net can significantly improve the prediction accuracy of dense depth maps, achieving a new state-of-the-art (SOTA), and demonstrating strong robustness to sparse and complex scenes. 

**Abstract (ZH)**: 自动驾驶领域中的深度完成是一项关键任务，旨在通过图像引导将稀疏的LiDAR深度测量值补充为高质量的稠密深度图。然而，现有方法通常将深度图视为彩色图像的附加通道，或者直接对稀疏数据进行卷积，未能充分利用深度图中的三维几何信息，尤其是在复杂边界和稀疏区域的表现不尽如人意。为了解决这些问题，本文提出了一种结合通道注意力机制和三维全局特征感知的深度完成网络（CGA-Net）。其主要创新包括：1）利用PointNet++从稀疏的深度图中提取全局3D几何特征，增强低线束LiDAR数据的场景感知能力；2）设计了一种基于通道注意力机制的多模态特征融合模块，有效地整合稀疏深度、RGB图像和3D几何特征；3）结合残差学习与CSPN++优化深度细化阶段，进一步提高边缘区域和复杂场景的完成质量。在KITTI深度完成数据集上的实验表明，CGA-Net可以显著提高稠密深度图的预测准确性，达到新的最先进的性能（SOTA），并展示出对稀疏和复杂场景的强大鲁棒性。 

---
# Facial Dynamics in Video: Instruction Tuning for Improved Facial Expression Perception and Contextual Awareness 

**Title (ZH)**: 视频中面部动态的建模：指令调整以提高面部表情感知和上下文意识 

**Authors**: Jiaxing Zhao, Boyuan Sun, Xiang Chen, Xihan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.07978)  

**Abstract**: Facial expression captioning has found widespread application across various domains. Recently, the emergence of video Multimodal Large Language Models (MLLMs) has shown promise in general video understanding tasks. However, describing facial expressions within videos poses two major challenges for these models: (1) the lack of adequate datasets and benchmarks, and (2) the limited visual token capacity of video MLLMs. To address these issues, this paper introduces a new instruction-following dataset tailored for dynamic facial expression caption. The dataset comprises 5,033 high-quality video clips annotated manually, containing over 700,000 tokens. Its purpose is to improve the capability of video MLLMs to discern subtle facial nuances. Furthermore, we propose FaceTrack-MM, which leverages a limited number of tokens to encode the main character's face. This model demonstrates superior performance in tracking faces and focusing on the facial expressions of the main characters, even in intricate multi-person scenarios. Additionally, we introduce a novel evaluation metric combining event extraction, relation classification, and the longest common subsequence (LCS) algorithm to assess the content consistency and temporal sequence consistency of generated text. Moreover, we present FEC-Bench, a benchmark designed to assess the performance of existing video MLLMs in this specific task. All data and source code will be made publicly available. 

**Abstract (ZH)**: 面部表情描述已经在多个领域找到了广泛的应用。近年来，视频多模态大型语言模型（MM-LLMs）在通用视频理解任务中展现出了潜力。然而，这些模型在描述视频中的面部表情时遇到了两大挑战：（1）缺乏充分的数据集和基准；（2）视频MM-LLMs的视觉标记容量有限。为了解决这些问题，本文引入了一个新的指令遵循数据集，专门用于动态面部表情描述。该数据集包含5,033个高质量的手动注释视频片段，包含超过700,000个标记，旨在提高视频MM-LLMs识别面部细微表情的能力。此外，我们提出了一种名为FaceTrack-MM的模型，该模型利用有限数量的标记来编码主要人物的脸部。该模型在追踪面部并在复杂多个人场景中聚焦主要人物的面部表情方面表现出优越性能。此外，我们还引入了一种新的评价指标，结合事件提取、关系分类以及最长公共子序列（LCS）算法来评估生成文本的内容一致性与时间顺序一致性。最后，我们提出了FEC-Bench基准，用于评估现有视频MM-LLMs在这一特定任务中的性能。所有数据和源代码都将公开发布。 

---
# Derivation of Output Correlation Inferences for Multi-Output (aka Multi-Task) Gaussian Process 

**Title (ZH)**: 多输出（又称多任务）高斯过程的输出相关性推断推导 

**Authors**: Shuhei Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2501.07964)  

**Abstract**: Gaussian process (GP) is arguably one of the most widely used machine learning algorithms in practice. One of its prominent applications is Bayesian optimization (BO). Although the vanilla GP itself is already a powerful tool for BO, it is often beneficial to be able to consider the dependencies of multiple outputs. To do so, Multi-task GP (MTGP) is formulated, but it is not trivial to fully understand the derivations of its formulations and their gradients from the previous literature. This paper serves friendly derivations of the MTGP formulations and their gradients. 

**Abstract (ZH)**: 高斯过程（Gaussian Process，GP）无疑是实践中应用最广泛的机器学习算法之一。其中一项显著的应用是贝叶斯优化（Bayesian Optimization，BO）。尽管传统的GP本身已经是一个强大的BO工具，但在考虑多个输出之间的依赖关系时，使用多任务高斯过程（Multi-task Gaussian Process，MTGP）通常会带来好处。然而，从以往文献中完全理解MTGP的形式及其梯度的推导并不容易。本文旨在提供MTGP的形式及其梯度的友好推导。 

---
# AI Guide Dog: Egocentric Path Prediction on Smartphone 

**Title (ZH)**: AI 导盲犬：智能手机上的第一人称路径预测 

**Authors**: Aishwarya Jadhav, Jeffery Cao, Abhishree Shetty, Urvashi Priyam Kumar, Aditi Sharma, Ben Sukboontip, Jayant Sravan Tamarapalli, Jingyi Zhang, Anirudh Koul  

**Link**: [PDF](https://arxiv.org/pdf/2501.07957)  

**Abstract**: This paper introduces AI Guide Dog (AIGD), a lightweight egocentric navigation assistance system for visually impaired individuals, designed for real-time deployment on smartphones. AIGD addresses key challenges in blind navigation by employing a vision-only, multi-label classification approach to predict directional commands, ensuring safe traversal across diverse environments. We propose a novel technique to enable goal-based outdoor navigation by integrating GPS signals and high-level directions, while also addressing uncertain multi-path predictions for destination-free indoor navigation. Our generalized model is the first navigation assistance system to handle both goal-oriented and exploratory navigation scenarios across indoor and outdoor settings, establishing a new state-of-the-art in blind navigation. We present methods, datasets, evaluations, and deployment insights to encourage further innovations in assistive navigation systems. 

**Abstract (ZH)**: 本文介绍了AI导盲犬（AIGD），这是一种为视觉障碍人士设计的轻量级第一人称导航辅助系统，能够在智能手机上实时部署。AIGD通过采用仅基于视觉的多标签分类方法来预测方向指令，解决了视障导航中的关键挑战，确保在各种环境中的安全通行。我们提出了一种新颖的技术，通过结合GPS信号和高层次的方向信息，实现基于目标的户外导航，并同时解决无目的地室内导航中的多路径预测不确定性问题。我们的通用模型是第一个能够处理室内和室外环境下目标导向和探索性导航场景的导航辅助系统，开创了视障导航的新前沿。本文呈现了方法、数据集、评估和部署方面的研究，以促进辅助导航系统的进一步创新。 

---
# Early prediction of the transferability of bovine embryos from videomicroscopy 

**Title (ZH)**: 从视频显微镜观察早期预测牛胚胎的转移潜力 

**Authors**: Yasmine Hachani, Patrick Bouthemy, Elisa Fromont, Sylvie Ruffini, Ludivine Laffont, Alline de Paula Reis  

**Link**: [PDF](https://arxiv.org/pdf/2501.07945)  

**Abstract**: Videomicroscopy is a promising tool combined with machine learning for studying the early development of in vitro fertilized bovine embryos and assessing its transferability as soon as possible. We aim to predict the embryo transferability within four days at most, taking 2D time-lapse microscopy videos as input. We formulate this problem as a supervised binary classification problem for the classes transferable and not transferable. The challenges are three-fold: 1) poorly discriminating appearance and motion, 2) class ambiguity, 3) small amount of annotated data. We propose a 3D convolutional neural network involving three pathways, which makes it multi-scale in time and able to handle appearance and motion in different ways. For training, we retain the focal loss. Our model, named SFR, compares favorably to other methods. Experiments demonstrate its effectiveness and accuracy for our challenging biological task. 

**Abstract (ZH)**: 视频显微镜结合机器学习是一种有潜力的研究体外受精牛胚胎早期发育及其移植潜力的工具，并能在尽可能短的时间内评估其移植能力。我们旨在通过对二维时间序列显微视频进行输入，预测胚胎的移植潜力，最多不超过四天。我们将此问题转化为一个监督二分类问题，分为可移植和不可移植两类。挑战主要来自三个方面：1）难以区分表型和运动特征，2）类别模糊性，3）标注数据量较小。我们提出了一种包含三个路径的三维卷积神经网络，使得模型具有多尺度的时间特征，并能够以不同的方式处理表型和运动。在训练过程中，我们保留了焦点损失函数。我们的模型命名为SFR，并且与其他方法相比表现优异。实验结果表明，该模型在我们的生物任务中具有有效性和准确性。 

---
# Gandalf the Red: Adaptive Security for LLMs 

**Title (ZH)**: 标题：Gandalf the Red：面向大语言模型的自适应安全方法 

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Natalie Wu, Mateo Rojas-Carulla  

**Link**: [PDF](https://arxiv.org/pdf/2501.07927)  

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and rigorously expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack datasets. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications. Code is available at \href{this https URL}{\texttt{this https URL}}. 

**Abstract (ZH)**: 当前对大型语言模型（LLM）应用中防御提示攻击的评估往往忽视了两个关键因素：对抗行为的动态性质以及限制性防御措施对合法用户使用体验带来的影响。我们提出了一种名为D-SEC（动态安全用例威胁模型）的方法，该方法明确地将攻击者与合法用户区分开来，并建模多步交互，严格地以可优化的形式表达了安全与可用性的关系。通过引入Gandalf（一个众包的、游戏化的红队平台），我们进一步改进了现有的评估方法，Gandalf旨在生成真实且适应性强的攻击数据集。使用Gandalf，我们收集并发布了包含279,000个提示攻击的数据集。结合良性用户数据，我们的分析揭示了安全性和可用性之间的相互作用，表明即使不阻断请求，集成在LLM中的防御措施（例如系统提示）也可能降低用户体验。我们证明，限定应用领域、多层次防御以及适应性防御是构建安全且有用的LLM应用的有效策略。代码可在以下链接下载：\href{this https URL}{\texttt{this https URL}}。 

---
# Optimal Classification Trees for Continuous Feature Data Using Dynamic Programming with Branch-and-Bound 

**Title (ZH)**: 使用动态规划与分支定界法的连续特征数据最佳分类树构建 

**Authors**: Catalin E. Brita, Jacobus G. M. van der Linden, Emir Demirović  

**Link**: [PDF](https://arxiv.org/pdf/2501.07903)  

**Abstract**: Computing an optimal classification tree that provably maximizes training performance within a given size limit, is NP-hard, and in practice, most state-of-the-art methods do not scale beyond computing optimal trees of depth three. Therefore, most methods rely on a coarse binarization of continuous features to maintain scalability. We propose a novel algorithm that optimizes trees directly on the continuous feature data using dynamic programming with branch-and-bound. We develop new pruning techniques that eliminate many sub-optimal splits in the search when similar to previously computed splits and we provide an efficient subroutine for computing optimal depth-two trees. Our experiments demonstrate that these techniques improve runtime by one or more orders of magnitude over state-of-the-art optimal methods and improve test accuracy by 5% over greedy heuristics. 

**Abstract (ZH)**: 在给定大小限制内计算一个能证明最大化训练性能的最优分类树是 NP 难问题。在实践中，最先进的大多数方法无法扩展到计算深度超过三的最优树。因此，大多数方法依赖粗略的连续特征二值化以保持可扩展性。我们提出了一种新的算法，该算法直接在连续特征数据上使用动态规划和分支限界法优化树结构。我们开发了新的剪枝技术，在搜索过程中可以消除许多类似于先前计算的分割的次优分割。我们还提供了一个高效子程序来计算最优的深度为两的树。我们的实验表明，这些技术相比最先进的最优方法能够将运行时间提高一个或多个数量级，并且相比贪婪启发式方法能够提高测试准确性 5%。 

---
# Leveraging Metamemory Mechanisms for Enhanced Data-Free Code Generation in LLMs 

**Title (ZH)**: 利用元记忆机制增强LLM中的无数据代码生成 

**Authors**: Shuai Wang, Liang Ding, Yibing Zhan, Yong Luo, Zheng He, Dapeng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07892)  

**Abstract**: Automated code generation using large language models (LLMs) has gained attention due to its efficiency and adaptability. However, real-world coding tasks or benchmarks like HumanEval and StudentEval often lack dedicated training datasets, challenging existing few-shot prompting approaches that rely on reference examples. Inspired by human metamemory-a cognitive process involving recall and evaluation-we present a novel framework (namely M^2WF) for improving LLMs' one-time code generation. This approach enables LLMs to autonomously generate, evaluate, and utilize synthetic examples to enhance reliability and performance. Unlike prior methods, it minimizes dependency on curated data and adapts flexibly to various coding scenarios. Our experiments demonstrate significant improvements in coding benchmarks, offering a scalable and robust solution for data-free environments. The code and framework will be publicly available on GitHub and HuggingFace. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）自动进行代码生成由于其高效性和适应性而引起了关注。然而，现实世界的编码任务或基准，如HumanEval和StudentEval，往往缺乏专门训练的数据集，这挑战了依赖参考示例的现有少量示例提示方法。受到人类元记忆——一种涉及回忆和评估的认知过程——的启发，我们提出了一种新的框架（称为M^2WF），以提高LLMs的一次性代码生成能力。该方法使LLMs能够自主生成、评估和利用合成示例，以增强可靠性和性能。与先前的方法不同，它最大限度地减少了对精心策划数据的依赖，并能够灵活适应各种编码场景。我们的实验结果显示在编码基准上取得了显著的改进，提供了一种适用于数据匮乏环境的可扩展和稳健的解决方案。该代码和框架将在GitHub和HuggingFace上公开提供。 

---
# GRAPHMOE: Amplifying Cognitive Depth of Mixture-of-Experts Network via Introducing Self-Rethinking Mechanism 

**Title (ZH)**: GRAPHMOE：通过引入自我反思机制增强混合专家网络的认知深度 

**Authors**: Chen Tang, Bo Lv, Zifan Zheng, Bohao Yang, Kun Zhao, Ning Liao, Xiaoxing Wang, Feiyu Xiong, Zhiyu Li, Nayu Liu, Jingchi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07890)  

**Abstract**: Traditional Mixture-of-Experts (MoE) networks benefit from utilizing multiple smaller expert models as opposed to a single large network. However, these experts typically operate independently, leaving a question open about whether interconnecting these models could enhance the performance of MoE networks. In response, we introduce GRAPHMOE, a novel method aimed at augmenting the cognitive depth of language models via a self-rethinking mechanism constructed on Pseudo GraphMoE networks. GRAPHMOE employs a recurrent routing strategy to simulate iterative thinking steps, thereby facilitating the flow of information among expert nodes. We implement the GRAPHMOE architecture using Low-Rank Adaptation techniques (LoRA) and conduct extensive experiments on various benchmark datasets. The experimental results reveal that GRAPHMOE outperforms other LoRA based models, achieving state-of-the-art (SOTA) performance. Additionally, this study explores a novel recurrent routing strategy that may inspire further advancements in enhancing the reasoning capabilities of language models. 

**Abstract (ZH)**: 传统的Mixture-of-Experts (MoE)网络通过使用多个小型专家模型而非单一的大网络而受益。然而，这些专家通常独立运行，这引发了关于是否可以通过连接这些模型来增强MoE网络性能的问题。为解决这一问题，我们引入了GRAPHMOE，这是一种旨在通过构建伪GraphMoE网络上的自我反思机制来增强语言模型认知深度的新方法。GRAPHMOE采用递归路由策略模拟迭代思考步骤，从而促进专家节点之间的信息流动。我们利用低秩适应技术（LoRA）实现GRAPHMOE架构，并在各种基准数据集上进行了大量实验。实验结果表明，GRAPHMOE优于其他基于LoRA的模型，实现了当前最佳性能（SOTA）。此外，本研究探索了一种新的递归路由策略，这可能会启发进一步提高语言模型推理能力的进展。 

---
# Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding 

**Title (ZH)**: Tarsier2：从详细的视频描述到全面的视频理解，推进大型视觉-语言模型的发展 

**Authors**: Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, Yuan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.07888)  

**Abstract**: We introduce Tarsier2, a state-of-the-art large vision-language model (LVLM) designed for generating detailed and accurate video descriptions, while also exhibiting superior general video understanding capabilities. Tarsier2 achieves significant advancements through three key upgrades: (1) Scaling pre-training data from 11M to 40M video-text pairs, enriching both volume and diversity; (2) Performing fine-grained temporal alignment during supervised fine-tuning; (3) Using model-based sampling to automatically construct preference data and applying DPO training for optimization. Extensive experiments show that Tarsier2-7B consistently outperforms leading proprietary models, including GPT-4o and Gemini 1.5 Pro, in detailed video description tasks. On the DREAM-1K benchmark, Tarsier2-7B improves F1 by 2.8\% over GPT-4o and 5.8\% over Gemini-1.5-Pro. In human side-by-side evaluations, Tarsier2-7B shows a +8.6\% performance advantage over GPT-4o and +24.9\% over Gemini-1.5-Pro. Tarsier2-7B also sets new state-of-the-art results across 15 public benchmarks, spanning tasks such as video question-answering, video grounding, hallucination test, and embodied question-answering, demonstrating its versatility as a robust generalist vision-language model. 

**Abstract (ZH)**: 我们介绍了Tarsier2，这是一个最先进的大型视觉-语言模型（LVLM），专门设计用于生成详细准确的视频描述，并具备出色的通用视频理解能力。Tarsier2 通过三项关键技术升级实现了重要进步：(1) 将预训练数据从 1100 万对视频-文本扩展到 4000 万对，丰富了数据的规模和多样性；(2) 在监督微调过程中进行精细粒度的时序对齐；(3) 使用基于模型的采样自动构建偏好数据，并采用 DPO 训练进行优化。大量实验表明，Tarsier2-7B 在细致的视频描述任务中始终优于包括 GPT-4o 和 Gemini 1.5 Pro 在内的领先商用模型。在 DREAM-1K 基准测试中，Tarsier2-7B 的 F1 得分分别比 GPT-4o 提高了 2.8% 和比 Gemini-1.5-Pro 提高了 5.8%。在人类旁观者评估中，Tarsier2-7B 在 GPT-4o 上表现出 +8.6% 的性能优势，在 Gemini-1.5-Pro 上则表现出 +24.9% 的显著优势。此外，Tarsier2-7B 在 15 个公开基准测试中设立了新的最佳性能记录，涵盖视频问答、视频定位、幻觉测试和体态问答等任务，证明了其作为稳健通用视觉-语言模型的灵活性和能力。 

---
# Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision 

**Title (ZH)**: 在弱监督条件下，迭代标签细化比偏好优化更为重要 

**Authors**: Yaowen Ye, Cassidy Laidlaw, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2501.07886)  

**Abstract**: Language model (LM) post-training relies on two stages of human supervision: task demonstrations for supervised finetuning (SFT), followed by preference comparisons for reinforcement learning from human feedback (RLHF). As LMs become more capable, the tasks they are given become harder to supervise. Will post-training remain effective under unreliable supervision? To test this, we simulate unreliable demonstrations and comparison feedback using small LMs and time-constrained humans. We find that in the presence of unreliable supervision, SFT still retains some effectiveness, but DPO (a common RLHF algorithm) fails to improve the model beyond SFT. To address this, we propose iterative label refinement (ILR) as an alternative to RLHF. ILR improves the SFT data by using comparison feedback to decide whether human demonstrations should be replaced by model-generated alternatives, then retrains the model via SFT on the updated data. SFT+ILR outperforms SFT+DPO on several tasks with unreliable supervision (math, coding, and safe instruction-following). Our findings suggest that as LMs are used for complex tasks where human supervision is unreliable, RLHF may no longer be the best use of human comparison feedback; instead, it is better to direct feedback towards improving the training data rather than continually training the model. Our code and data are available at this https URL. 

**Abstract (ZH)**: 语言模型（LM）的后训练依赖于两个阶段的人类监督：监督微调（SFT）的任务示范，随后是基于人类反馈强化学习（RLHF）的偏好比较。随着语言模型能力的增强，它们所执行的任务越来越难以监督。在不可靠的监督下，后训练是否仍然有效？为了检验这一点，我们使用小型语言模型和时间限制下的人类模拟不可靠的示范和比较反馈。我们发现，在存在不可靠监督的情况下，SFT仍然保留了一定的效果，但常见的RLHF算法DPO无法使模型超越SFT的表现。为了解决这个问题，我们提出了一种迭代标签校正（ILR）方法，作为RLHF的替代方案。ILR通过使用比较反馈来决定是否应该用模型生成的替代品替换人类示范，然后通过SFT重新训练模型以更新的数据为基础。在不可靠监督的任务中（数学、编程和安全指令遵循），SFT+ILR在多个任务上的表现优于SFT+DPO。我们的研究结果表明，当语言模型用于复杂任务而人类监督不可靠时，RLHF可能不是利用人类比较反馈的最佳途径；相反，最好将反馈引导至改善训练数据，而非不断训练模型。我们的代码和数据可在以下网址获取：this https URL。 

---
# Continual Learning with Embedding Layer Surgery and Task-wise Beam Search using Whisper 

**Title (ZH)**: 使用嵌入层手术和任务导向的束搜索的持续学习方法——以Whisper为例 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2501.07875)  

**Abstract**: Current Multilingual ASR models only support a fraction of the world's languages. Continual Learning (CL) aims to tackle this problem by adding new languages to pre-trained models while avoiding the loss of performance on existing languages, also known as Catastrophic Forgetting (CF). However, existing CL methods overlook the adaptation of the token embedding lookup table at the decoder, despite its significant contribution to CF. We propose Embedding Layer Surgery where separate copies of the token embeddings are created for each new languages, and one of the copies is selected to replace the old languages embeddings when transcribing the corresponding new language. Unfortunately, this approach means LID errors also cause incorrect ASR embedding selection. Our Task-wise Beam Search allows self-correction for such mistakes. By adapting Whisper to 10 hours of data for each of 10 unseen languages from Common Voice, results show that our method reduces the Average WER (AWER) of pre-trained languages from 14.2% to 11.9% compared with Experience Replay, without compromising the AWER of the unseen languages. 

**Abstract (ZH)**: 当前的多语言ASR模型仅支持世界语言的一小部分。持续学习（CL）旨在通过在预训练模型中添加新语言来解决这个问题，同时避免现有语言性能下降，即灾难性遗忘（CF）。然而，现有的CL方法忽视了解码器中词嵌入查找表的适应性调整，尽管其对CF有显著贡献。我们提出了一种词嵌入层手术方法，为每种新语言创建词嵌入的独立副本，并在转写相应新语言时选择一个副本替换旧语言嵌入。不幸的是，这种方法意味着LID错误也会导致ASR词嵌入选择错误。我们的任务导向型贝叶斯搜索允许自我纠正此类错误。通过将Whisper扩展到来自Common Voice的每种未见过的语言10小时数据，结果显示，我们的方法将预训练语言的平均词错误率（AWER）从14.2%降低到11.9%，而未影响未见过语言的AWER，从而在不牺牲未见过语言性能的情况下显著提升了预训练语言的性能。 

---
# deepTerra -- AI Land Classification Made Easy 

**Title (ZH)**: DeepTerra -- 简化的人工智能土地分类 

**Authors**: Andrew Keith Wilkinson  

**Link**: [PDF](https://arxiv.org/pdf/2501.07859)  

**Abstract**: deepTerra is a comprehensive platform designed to facilitate the classification of land surface features using machine learning and satellite imagery. The platform includes modules for data collection, image augmentation, training, testing, and prediction, streamlining the entire workflow for image classification tasks. This paper presents a detailed overview of the capabilities of deepTerra, shows how it has been applied to various research areas, and discusses the future directions it might take. 

**Abstract (ZH)**: deepTerra 是一个全面的平台，旨在利用机器学习和卫星影像促进土地表面特征的分类。该平台包括数据采集、图像增强、训练、测试和预测的模块，简化了图像分类任务的整个工作流程。本文详细介绍了 deepTerra 的功能，展示了其在各种研究领域的应用，并讨论了其未来的发展方向。 

---
# Hierarchical Repository-Level Code Summarization for Business Applications Using Local LLMs 

**Title (ZH)**: 使用本地LLM进行商业应用的分层仓库级别代码摘要 

**Authors**: Nilesh Dhulshette, Sapan Shah, Vinay Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2501.07857)  

**Abstract**: In large-scale software development, understanding the functionality and intent behind complex codebases is critical for effective development and maintenance. While code summarization has been widely studied, existing methods primarily focus on smaller code units, such as functions, and struggle with larger code artifacts like files and packages. Additionally, current summarization models tend to emphasize low-level implementation details, often overlooking the domain and business context that are crucial for real-world applications. This paper proposes a two-step hierarchical approach for repository-level code summarization, tailored to business applications. First, smaller code units such as functions and variables are identified using syntax analysis and summarized with local LLMs. These summaries are then aggregated to generate higher-level file and package summaries. To ensure the summaries are grounded in business context, we design custom prompts that capture the intended purpose of code artifacts based on the domain and problem context of the business application. We evaluate our approach on a business support system (BSS) for the telecommunications domain, showing that syntax analysis-based hierarchical summarization improves coverage, while business-context grounding enhances the relevance of the generated summaries. 

**Abstract (ZH)**: 在大规模软件开发中，理解复杂代码库的功能和意图对于有效的开发和维护至关重要。虽然代码总结已经得到了广泛的研究，但现有方法主要关注较小的代码单元，如函数，并在处理文件和包这样的较大代码构件时遇到困难。此外，当前的总结模型往往强调低级别的实现细节，而忽视了真实世界应用中至关重要的领域和业务上下文。本文提出了一种针对商业应用的两阶段层次化代码库总结方法。首先，通过语法分析识别较小的代码单元（如函数和变量），并使用局部语言模型（LLM）对其进行总结。这些摘要随后被聚合，以生成较高层次的文件和包摘要。为了确保摘要是基于业务上下文的，我们设计了定制的提示，根据业务应用的领域和问题上下文来捕获代码构件的预期目的。我们在电信领域的业务支持系统（BSS）上评估了我们的方法，结果表明，基于语法分析的层次化总结可以提高覆盖范围，而基于业务上下文的上下文约束则增强了生成摘要的相关性。 

---
# State-of-the-Art Transformer Models for Image Super-Resolution: Techniques, Challenges, and Applications 

**Title (ZH)**: 基于Transformer模型的最新图像超分辨率技术、挑战与应用 

**Authors**: Debasish Dutta, Deepjyoti Chetia, Neeharika Sonowal, Sanjib Kr Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2501.07855)  

**Abstract**: Image Super-Resolution (SR) aims to recover a high-resolution image from its low-resolution counterpart, which has been affected by a specific degradation process. This is achieved by enhancing detail and visual quality. Recent advancements in transformer-based methods have remolded image super-resolution by enabling high-quality reconstructions surpassing previous deep-learning approaches like CNN and GAN-based. This effectively addresses the limitations of previous methods, such as limited receptive fields, poor global context capture, and challenges in high-frequency detail recovery. Additionally, the paper reviews recent trends and advancements in transformer-based SR models, exploring various innovative techniques and architectures that combine transformers with traditional networks to balance global and local contexts. These neoteric methods are critically analyzed, revealing promising yet unexplored gaps and potential directions for future research. Several visualizations of models and techniques are included to foster a holistic understanding of recent trends. This work seeks to offer a structured roadmap for researchers at the forefront of deep learning, specifically exploring the impact of transformers on super-resolution techniques. 

**Abstract (ZH)**: 图像超分辨率（SR）的目标是从低分辨率版本中恢复高分辨率图像，该图像已被特定的降质过程影响。这一目标通过增强细节和视觉质量来实现。基于变压器的方法的最近进展重塑了图像超分辨率领域，使得能够生成超越以往深度学习方法（如CNN和GAN）的高质量重建。这种方法有效地解决了先前方法的局限性，例如受限的感受野、全球上下文捕捉不足以及高频细节恢复的挑战。此外，本文还回顾了基于变压器的SR模型的最新趋势和进展，探讨了将变压器与传统网络相结合的各种创新技术和架构，以平衡全局和局部上下文。这些新兴的方法进行了批判性分析，揭示了具有前景但尚未探索的空白和未来研究的潜在方向。文中还包括了模型和方法的可视化，以促进对最新趋势的全面理解。这项工作旨在为深度学习领域的前沿研究人员提供一个结构化的路线图，特别探讨了变压器对超分辨率技术的影响。 

---
# Optimizing Language Models for Grammatical Acceptability: A Comparative Study of Fine-Tuning Techniques 

**Title (ZH)**: 优化语言模型的语法接受度：细调技术比较研究 

**Authors**: Shobhit Ratan, Farley Knight, Ghada Jerfel, Sze Chung Ho  

**Link**: [PDF](https://arxiv.org/pdf/2501.07853)  

**Abstract**: This study explores the fine-tuning (FT) of the Open Pre-trained Transformer (OPT-125M) for grammatical acceptability tasks using the CoLA dataset. By comparing Vanilla-Fine-Tuning (VFT), Pattern-Based-Fine-Tuning (PBFT), and Parameter-Efficient Fine-Tuning techniques (PEFT) like Low-Rank Adaptation (LoRA), we demonstrate significant improvements in computational efficiency while maintaining high accuracy. Our experiments reveal that while VFT achieves the highest accuracy (81.2%), LoRA enhancing FT by reducing memory usage and iteration time by more than 50%, and increases accuracy in PBFT case. Context Distillation (CD), though computationally efficient, underperformed with accuracy around 31%. Our findings contribute to democratizing access to large language models (LLM) by reducing computational barriers. 

**Abstract (ZH)**: 本研究探讨了使用CoLA数据集对Open Pre-trained Transformer (OPT-125M) 进行微调（Fine-Tuning, FT）以完成语法规则接受性任务的情况。通过对比Vanilla-Fine-Tuning (VFT)、Pattern-Based-Fine-Tuning (PBFT) 以及参数高效微调技术（Parameter-Efficient Fine-Tuning, PEFT）如Low-Rank Adaptation (LoRA)，我们展示了在保持高精度的同时显著提高计算效率。实验结果显示，虽然VFT在精度上达到了最高值（81.2%），但通过减少内存使用和迭代时间超过50% 的方式，LoRA增强了FT，并在PBFT情况下提高了准确性。尽管上下文蒸馏（Context Distillation, CD）在计算效率上表现良好，但其精度仅约为31%，表现欠佳。我们的研究结果有助于降低访问大型语言模型（Large Language Models, LLM）的技术门槛，从而实现更大的普及。 

---
# Unveiling Provider Bias in Large Language Models for Code Generation 

**Title (ZH)**: 揭示代码生成大型语言模型中的提供者偏见 

**Authors**: Xiaoyu Zhang, Juan Zhai, Shiqing Ma, Qingshuang Bao, Weipeng Jiang, Chao Shen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07849)  

**Abstract**: Large Language Models (LLMs) have emerged as the new recommendation engines, outperforming traditional methods in both capability and scope, particularly in code generation applications. Our research reveals a novel provider bias in LLMs, namely without explicit input prompts, these models show systematic preferences for services from specific providers in their recommendations (e.g., favoring Google Cloud over Microsoft Azure). This bias holds significant implications for market dynamics and societal equilibrium, potentially promoting digital monopolies. It may also deceive users and violate their expectations, leading to various consequences. This paper presents the first comprehensive empirical study of provider bias in LLM code generation. We develop a systematic methodology encompassing an automated pipeline for dataset generation, incorporating 6 distinct coding task categories and 30 real-world application scenarios. Our analysis encompasses over 600,000 LLM-generated responses across seven state-of-the-art models, utilizing approximately 500 million tokens (equivalent to \$5,000+ in computational costs). The study evaluates both the generated code snippets and their embedded service provider selections to quantify provider bias. Additionally, we conduct a comparative analysis of seven debiasing prompting techniques to assess their efficacy in mitigating these biases. Our findings demonstrate that LLMs exhibit significant provider preferences, predominantly favoring services from Google and Amazon, and can autonomously modify input code to incorporate their preferred providers without users' requests. Notably, we observe discrepancies between providers recommended in conversational contexts versus those implemented in generated code. The complete dataset and analysis results are available in our repository. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为新的推荐引擎，其在能力和范围方面超越了传统方法，尤其是在代码生成应用方面。我们的研究揭示了LLMs中一种新型的提供者偏好，即这些模型在没有显式输入提示的情况下，在推荐中系统地偏好某些特定提供者的服务（例如，更喜欢Google Cloud而非Microsoft Azure）。这种偏好对市场动态和社会平衡产生了重要影响，可能促进数字垄断的局面。同时，它也可能误导用户，违背用户的期望，导致各种后果。本文首次进行了全面的实证研究，探讨LLMs在代码生成中的提供者偏好。我们开发了一种系统的方法，包括一个自动的数据集生成管道，涵盖了6种不同的编码任务类别和30个真实世界的应用场景。我们的分析涵盖了来自七种先进模型的超过60万次LLM生成响应，使用了大约5亿个令牌（相当于数千美元的计算成本）。研究评估了生成的代码片段及其嵌入的服务提供商选择，以量化提供者偏好。此外，我们还对七种去偏估计提示技术进行了比较分析，评估它们在减轻这些偏见方面的有效性。我们的研究结果表明，LLMs表现出明显的提供者偏好，主要偏好Google和Amazon的服务，并且能够自主修改输入代码以包含其首选提供者，而无需用户请求。值得注意的是，我们观察到交流中推荐的提供者与生成代码中实现的提供者之间的差异。完整的数据集和分析结果可在我们的仓库中获取。 

---
# Social Media Data Mining With Natural Language Processing on Public Dream Contents 

**Title (ZH)**: 使用自然语言处理技术挖掘公共梦境内容的社会媒体数据Mining Social Media Data With Natural Language Processing on Public Dream Contents 

**Authors**: Howard Hua, Joe Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07839)  

**Abstract**: The COVID-19 pandemic has significantly transformed global lifestyles, enforcing physical isolation and accelerating digital adoption for work, education, and social interaction. This study examines the pandemic's impact on mental health by analyzing dream content shared on the Reddit r/Dreams community. With over 374,000 subscribers, this platform offers a rich dataset for exploring subconscious responses to the pandemic. Using statistical methods, we assess shifts in dream positivity, negativity, and neutrality from the pre-pandemic to post-pandemic era. To enhance our analysis, we fine-tuned the LLaMA 3.1-8B model with labeled data, enabling precise sentiment classification of dream content. Our findings aim to uncover patterns in dream content, providing insights into the psychological effects of the pandemic and its influence on subconscious processes. This research highlights the profound changes in mental landscapes and the role of dreams as indicators of public well-being during unprecedented times. 

**Abstract (ZH)**: COVID-19大流行极大地改变了全球生活方式，强制实施了身体隔离，并加速了工作、教育和社交互动的数字化进程。本研究通过分析在Reddit r/Dreams社区分享的梦境内容，考察大流行对心理健康的影响。该平台拥有超过374,000名订阅者，提供了丰富的数据集，用于探索对大流行的潜意识反应。通过统计方法，我们评估了从大流行前到大流行后的梦境正面性、负面性和中立性变化。为了增强我们的分析，我们使用带标签数据对LLaMA 3.1-8B模型进行了微调，使我们能够精确地对梦境内容进行情感分类。我们的研究结果旨在揭示梦境内容中的模式，提供关于大流行的心理影响以及其对潜意识过程的影响的见解。这项研究突显了在前所未有的时期，心理景观的深刻变化以及梦境作为公共福祉指标的作用。 

---
# Real-time Verification and Refinement of Language Model Text Generation 

**Title (ZH)**: 实时验证与改进语言模型文本生成 

**Authors**: Joonho Ko, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07824)  

**Abstract**: Large language models (LLMs) have shown remarkable performance across a wide range of natural language tasks. However, a critical challenge remains in that they sometimes generate factually incorrect answers. To address this, while many previous work has focused on identifying errors in their generation and further refining them, they are slow in deployment since they are designed to verify the response from LLMs only after their entire generation (from the first to last tokens) is done. Further, we observe that once LLMs generate incorrect tokens early on, there is a higher likelihood that subsequent tokens will also be factually incorrect. To this end, in this work, we propose Streaming-VR (Streaming Verification and Refinement), a novel approach designed to enhance the efficiency of verification and refinement of LLM outputs. Specifically, the proposed Streaming-VR enables on-the-fly verification and correction of tokens as they are being generated, similar to a streaming process, ensuring that each subset of tokens is checked and refined in real-time by another LLM as the LLM constructs its response. Through comprehensive evaluations on multiple datasets, we demonstrate that our approach not only enhances the factual accuracy of LLMs, but also offers a more efficient solution compared to prior refinement methods. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多种自然语言任务中表现出色。然而，它们仍然存在一个关键挑战，即有时会生成事实性错误的答案。为了解决这个问题，尽管以往许多研究集中在识别和修正其生成过程中的错误，但这些方法在部署时较慢，因为它们设计为在生成过程完成后才对其进行验证。此外，我们观察到，一旦LLMs生成了错误的词汇，后续词汇出现事实性错误的可能性更高。基于此，本文提出了一种名为Streaming-VR（流式验证与修正）的新方法，旨在提高LLM输出验证和修正的效率。具体而言，Streaming-VR方法可以在生成过程中即席验证和修正词汇，类似于流式处理过程，并确保每个子集的词汇在LLM生成其响应的同时实时被另一个LLM检查和修正。通过在多个数据集上的全面评估，我们展示了该方法不仅提高了LLM的事实准确性，还比以往的修正方法提供了更高效的解决方案。 

---
# A Multi-Encoder Frozen-Decoder Approach for Fine-Tuning Large Language Models 

**Title (ZH)**: 一种多编码器固化解码器方法用于大型语言模型微调 

**Authors**: Kaustubh D. Dhole  

**Link**: [PDF](https://arxiv.org/pdf/2501.07818)  

**Abstract**: Among parameter-efficient fine-tuning methods, freezing has emerged as a popular strategy for speeding up training, reducing catastrophic forgetting, and improving downstream performance. We investigate the impact of freezing the decoder in a multi-task setup comprising diverse natural language tasks, aiming to reduce deployment overhead and enhance portability to novel tasks. Our experiments, conducted by fine-tuning both individual and multi-task setups on the AlexaTM model, reveal that freezing decoders is highly effective for tasks with natural language outputs and mitigates catastrophic forgetting in multilingual tasks. However, we find that pairing frozen decoders with a larger model can effectively maintain or even enhance performance in structured and QA tasks, making it a viable strategy for a broader range of task types. 

**Abstract (ZH)**: 在参数效率微调方法中，冻结参数已成为加速训练、减少灾难性遗忘并提高下游性能的一种流行策略。我们探讨了在包含多样化自然语言任务的多任务设置中冻结解码器的影响，旨在减少部署成本并增强其对新型任务的适用性。我们的实验通过在AlexaTM模型上微调个体和多任务设置，发现冻结解码器对具有自然语言输出的任务非常有效，并且在多语言任务中减轻了灾难性遗忘。然而，我们发现将冻结的解码器与更大的模型相结合，可以在结构化和问答任务中有效维持甚至提高性能，这使其成为一种更广泛任务类型的可行策略。 

---
# STTS-EAD: Improving Spatio-Temporal Learning Based Time Series Prediction via 

**Title (ZH)**: STTS-EAD：通过改进空时学习提高基于时空的学习时间序列预测 

**Authors**: Yuanyuan Liang, Tianhao Zhang, Tingyu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2501.07814)  

**Abstract**: Handling anomalies is a critical preprocessing step in multivariate time series prediction. However, existing approaches that separate anomaly preprocessing from model training for multivariate time series prediction encounter significant limitations. Specifically, these methods fail to utilize auxiliary information crucial for identifying latent anomalies associated with spatiotemporal factors during the preprocessing stage. Instead, they rely solely on data distribution for anomaly detection, which can result in the incorrect processing of numerous samples that could otherwise contribute positively to model training. To address this, we propose STTS-EAD, an end-to-end method that seamlessly integrates anomaly detection into the training process of multivariate time series forecasting and aims to improve Spatio-Temporal learning based Time Series prediction via Embedded Anomaly Detection. Our proposed STTS-EAD leverages spatio-temporal information for forecasting and anomaly detection, with the two parts alternately executed and optimized for each other. To the best of our knowledge, STTS-EAD is the first to integrate anomaly detection and forecasting tasks in the training phase for improving the accuracy of multivariate time series forecasting. Extensive experiments on a public stock dataset and two real-world sales datasets from a renowned coffee chain enterprise show that our proposed method can effectively process detected anomalies in the training stage to improve forecasting performance in the inference stage and significantly outperform baselines. 

**Abstract (ZH)**: 多变量时间序列预测中的异常处理是至关重要的预处理步骤。然而，现有方法将异常处理与模型训练分离来处理多变量时间序列预测的问题，存在显著的局限性。具体来说，这些方法在预处理阶段无法利用与时空因素相关的潜在异常所需的重要辅助信息，而是仅依赖数据分布来进行异常检测，这可能导致大量本可以对模型训练产生积极贡献的样本被错误处理。为了解决这一问题，我们提出了一种端到端的STTS-EAD方法，该方法可以无缝地将异常检测集成到多变量时间序列预测的训练过程中，并旨在通过嵌入式异常检测改进时空学习基于时间序列的预测。我们的STTS-EAD方法利用时空信息进行预测和异常检测，两部分交替执行并互相优化。据我们所知，STTS-EAD是第一个通过在训练阶段整合异常检测和预测任务来提高多变量时间序列预测准确性的方法。在公共股市数据集和一家著名咖啡连锁企业的两个真实世界销售数据集上的广泛实验表明，本方法可以在训练阶段有效处理检测到的异常，从而提高推理阶段的预测性能，并显著优于基线方法。 

---
# Talk to Right Specialists: Routing and Planning in Multi-agent System for Question Answering 

**Title (ZH)**: 恰当地与专家沟通：面向多代理系统的问答路由与规划 

**Authors**: Feijie Wu, Zitao Li, Fei Wei, Yaliang Li, Bolin Ding, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07813)  

**Abstract**: Leveraging large language models (LLMs), an agent can utilize retrieval-augmented generation (RAG) techniques to integrate external knowledge and increase the reliability of its responses. Current RAG-based agents integrate single, domain-specific knowledge sources, limiting their ability and leading to hallucinated or inaccurate responses when addressing cross-domain queries. Integrating multiple knowledge bases into a unified RAG-based agent raises significant challenges, including increased retrieval overhead and data sovereignty when sensitive data is involved. In this work, we propose RopMura, a novel multi-agent system that addresses these limitations by incorporating highly efficient routing and planning mechanisms. RopMura features two key components: a router that intelligently selects the most relevant agents based on knowledge boundaries and a planner that decomposes complex multi-hop queries into manageable steps, allowing for coordinating cross-domain responses. Experimental results demonstrate that RopMura effectively handles both single-hop and multi-hop queries, with the routing mechanism enabling precise answers for single-hop queries and the combined routing and planning mechanisms achieving accurate, multi-step resolutions for complex queries. 

**Abstract (ZH)**: 利用大规模语言模型（LLMs），代理可以通过检索增强生成（RAG）技术整合外部知识，从而提高其回复的可靠性。当前基于RAG的代理整合的是单个、专业化的知识来源，这限制了它们的能力，并在处理跨领域查询时导致产生了虚假或不准确的回复。将多个知识库整合到统一的基于RAG的代理中会面临显著的挑战，包括增加的检索开销以及涉及敏感数据时的数据主权问题。在这项工作中，我们提出了一种名为RopMura的新型多代理系统，通过引入高效的路由和规划机制来解决这些局限性。RopMura有两个关键组件：一个路由器，可以根据知识边界智能地选择最相关的代理；以及一个规划器，将复杂的多跳查询分解成可管理的步骤，从而协调跨领域的回复。实验结果表明，RopMura能够有效处理单跳和多跳查询。路由机制能够为单跳查询提供精确的答案，而结合的路由和规划机制则可以实现复杂查询的精确、多步解决方案。 

---
# Conformal mapping Coordinates Physics-Informed Neural Networks (CoCo-PINNs): learning neural networks for designing neutral inclusions 

**Title (ZH)**: 等角映射坐标下的物理拟合神经网络 (CoCo-PINNs): 学习用于设计中间相的神经网络 

**Authors**: Daehee Cho, Hyeonmin Yun, Jaeyong Lee, Mikyoung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2501.07809)  

**Abstract**: We focus on designing and solving the neutral inclusion problem via neural networks. The neutral inclusion problem has a long history in the theory of composite materials, and it is exceedingly challenging to identify the precise condition that precipitates a general-shaped inclusion into a neutral inclusion. Physics-informed neural networks (PINNs) have recently become a highly successful approach to addressing both forward and inverse problems associated with partial differential equations. We found that traditional PINNs perform inadequately when applied to the inverse problem of designing neutral inclusions with arbitrary shapes. In this study, we introduce a novel approach, Conformal mapping Coordinates Physics-Informed Neural Networks (CoCo-PINNs), which integrates complex analysis techniques into PINNs. This method exhibits strong performance in solving forward-inverse problems to construct neutral inclusions of arbitrary shapes in two dimensions, where the imperfect interface condition on the inclusion's boundary is modeled by training neural networks. Notably, we mathematically prove that training with a single linear field is sufficient to achieve neutrality for untrained linear fields in arbitrary directions, given a minor assumption. We demonstrate that CoCo-PINNs offer enhanced performances in terms of credibility, consistency, and stability. 

**Abstract (ZH)**: 我们致力于通过神经网络设计和解决中性包涵问题。中性包涵问题在复合材料理论中历史悠久，识别出导致一般形状包涵物成为中性包涵物的具体条件极具挑战性。物理信息神经网络（PINNs）近年来已成为解决偏微分方程正问题和反问题的强有力方法。然而，我们发现传统PINNs在解决任意形状中性包涵物设计的反问题时表现不佳。在本研究中，我们引入了一种新颖的方法——双曲映射坐标物理信息神经网络（CoCo-PINNs），该方法将复分析技术整合到PINNs中。该方法在二维中性包涵物的任意形状建模和求解正问题与反问题方面表现出色，其中包涵物边界上的不完美界面条件通过训练神经网络来建模。特别地，我们通过一个小前提证明了使用单一线性场进行训练足以在任意方向上实现未训练线性场的中性化。我们还展示了CoCo-PINNs在可信性、一致性和稳定性方面的增强性能。 

---
# A Comparative Analysis of DNN-based White-Box Explainable AI Methods in Network Security 

**Title (ZH)**: 基于DNN的白盒可解释人工智能方法在网络安领域的比较分析 

**Authors**: Osvaldo Arreche, Mustafa Abdallah  

**Link**: [PDF](https://arxiv.org/pdf/2501.07801)  

**Abstract**: New research focuses on creating artificial intelligence (AI) solutions for network intrusion detection systems (NIDS), drawing its inspiration from the ever-growing number of intrusions on networked systems, increasing its complexity and intelligibility. Hence, the use of explainable AI (XAI) techniques in real-world intrusion detection systems comes from the requirement to comprehend and elucidate black-box AI models to security analysts. In an effort to meet such requirements, this paper focuses on applying and evaluating White-Box XAI techniques (particularly LRP, IG, and DeepLift) for NIDS via an end-to-end framework for neural network models, using three widely used network intrusion datasets (NSL-KDD, CICIDS-2017, and RoEduNet-SIMARGL2021), assessing its global and local scopes, and examining six distinct assessment measures (descriptive accuracy, sparsity, stability, robustness, efficiency, and completeness). We also compare the performance of white-box XAI methods with black-box XAI methods. The results show that using White-box XAI techniques scores high in robustness and completeness, which are crucial metrics for IDS. Moreover, the source codes for the programs developed for our XAI evaluation framework are available to be improved and used by the research community. 

**Abstract (ZH)**: 新的研究聚焦于为网络入侵检测系统（NIDS）创建人工智能（AI）解决方案，这些研究深受日益增多的网络系统入侵事件的启发，这些事件使得网络系统的复杂性和可解释性不断增加。因此，在网络入侵检测系统中采用可解释的人工智能（XAI）技术，是为了满足安全分析师理解并阐释黑箱AI模型的需求。为满足这些需求，本文致力于通过神经网络模型的端到端框架，应用并评估白盒XAI技术（特别是LRP、IG和DeepLift），并使用三个广泛使用的网络入侵数据集（NSL-KDD、CICIDS-2017和RoEduNet-SIMARGL2021），评估其全局和局部解释范围，并对六种不同的评估指标（描述准确性、稀疏性、稳定性、鲁棒性、效率和完备性）进行考察。同时，我们将比较白盒XAI方法和黑盒XAI方法的性能。结果表明，使用白盒XAI技术在鲁棒性和完备性方面得分较高，这是IDS的关键指标。此外，我们为XAI评估框架开发的程序源代码已可供研究社区改进和使用。 

---
# BioPose: Biomechanically-accurate 3D Pose Estimation from Monocular Videos 

**Title (ZH)**: BioPose：单目视频中基于生物力学准确的3D姿态估计 

**Authors**: Farnoosh Koleini, Muhammad Usama Saleem, Pu Wang, Hongfei Xue, Ahmed Helmy, Abbey Fenwick  

**Link**: [PDF](https://arxiv.org/pdf/2501.07800)  

**Abstract**: Recent advancements in 3D human pose estimation from single-camera images and videos have relied on parametric models, like SMPL. However, these models oversimplify anatomical structures, limiting their accuracy in capturing true joint locations and movements, which reduces their applicability in biomechanics, healthcare, and robotics. Biomechanically accurate pose estimation, on the other hand, typically requires costly marker-based motion capture systems and optimization techniques in specialized labs. To bridge this gap, we propose BioPose, a novel learning-based framework for predicting biomechanically accurate 3D human pose directly from monocular videos. BioPose includes three key components: a Multi-Query Human Mesh Recovery model (MQ-HMR), a Neural Inverse Kinematics (NeurIK) model, and a 2D-informed pose refinement technique. MQ-HMR leverages a multi-query deformable transformer to extract multi-scale fine-grained image features, enabling precise human mesh recovery. NeurIK treats the mesh vertices as virtual markers, applying a spatial-temporal network to regress biomechanically accurate 3D poses under anatomical constraints. To further improve 3D pose estimations, a 2D-informed refinement step optimizes the query tokens during inference by aligning the 3D structure with 2D pose observations. Experiments on benchmark datasets demonstrate that BioPose significantly outperforms state-of-the-art methods. Project website: \url{this https URL}. 

**Abstract (ZH)**: 近年来，从单摄像头图像和视频中估计三维人体姿态的技术主要依赖于参数化模型，如SMPL。然而，这些模型过度简化了解剖结构，限制了它们在捕捉关节位置和运动真实性的准确度，从而限制了其在生物力学、医疗健康和机器人领域的应用。相比之下，精确的生物力学姿态估计通常需要昂贵的标记基动作捕捉系统和特定实验室中的优化技术。为了解决这一问题，我们提出了一种名为BioPose的新颖学习框架，可以直接从单目视频中预测生物力学准确的三维人体姿态。BioPose包含了三个关键组件：多查询人体网格恢复模型（MQ-HMR）、神经逆动力学（NeurIK）模型以及基于2D姿态的信息细化技术。MQ-HMR利用多查询变形的变压器提取多尺度的细粒度图像特征，实现了精准的人体网格恢复。NeurIK将网格顶点视为虚拟标记，并应用时空网络在解剖结构约束下回归生物力学准确的3D姿态。为了进一步提高3D姿态估计的准确性，在推理过程中通过使3D结构与2D姿态观察相一致来优化查询标记信息，从而实现了2D信息的细化步骤。在基准数据集上的实验表明，BioPose显著优于现有方法。项目网站：[请点击此处](this https URL)。 

---
# Transforming Indoor Localization: Advanced Transformer Architecture for NLOS Dominated Wireless Environments with Distributed Sensors 

**Title (ZH)**: Indoor定位的革命：基于分布式传感器在非视距主导无线环境中的高级变压器架构 

**Authors**: Saad Masrur, Jung-Fu, Cheng, Atieh R. Khamesi, Ismail Guvenc  

**Link**: [PDF](https://arxiv.org/pdf/2501.07774)  

**Abstract**: Indoor localization in challenging non-line-of-sight (NLOS) environments often leads to mediocre accuracy with traditional approaches. Deep learning (DL) has been applied to tackle these challenges; however, many DL approaches overlook computational complexity, especially for floating-point operations (FLOPs), making them unsuitable for resource-limited devices. Transformer-based models have achieved remarkable success in natural language processing (NLP) and computer vision (CV) tasks, motivating their use in wireless applications. However, their use in indoor localization remains nascent, and directly applying Transformers for indoor localization can be both computationally intensive and exhibit limitations in accuracy. To address these challenges, in this work, we introduce a novel tokenization approach, referred to as Sensor Snapshot Tokenization (SST), which preserves variable-specific representations of power delay profile (PDP) and enhances attention mechanisms by effectively capturing multi-variate correlation. Complementing this, we propose a lightweight Swish-Gated Linear Unit-based Transformer (L-SwiGLU Transformer) model, designed to reduce computational complexity without compromising localization accuracy. Together, these contributions mitigate the computational burden and dependency on large datasets, making Transformer models more efficient and suitable for resource-constrained scenarios. The proposed tokenization method enables the Vanilla Transformer to achieve a 90th percentile positioning error of 0.388 m in a highly NLOS indoor factory, surpassing conventional tokenization methods. The L-SwiGLU ViT further reduces the error to 0.355 m, achieving an 8.51% improvement. Additionally, the proposed model outperforms a 14.1 times larger model with a 46.13% improvement, underscoring its computational efficiency. 

**Abstract (ZH)**: 在具有挑战性的非视距（NLOS）环境中的室内定位通常会导致使用传统方法时精度较差。深度学习（DL）已被应用于解决这些挑战，但许多DL方法忽视了计算复杂性，尤其是在浮点运算（FLOPs）方面，这使得它们不适合资源受限的设备。基于Transformer的模型在自然语言处理（NLP）和计算机视觉（CV）任务中取得了显著的成功，这激发了将其用于无线应用的可能性。然而，它们在室内定位中的应用依然较少，直接将Transformer用于室内定位既计算密集型又在准确性上存在局限性。为应对这些挑战，本文介绍了一种新的token化方法，称为传感器快照token化（SST），它可以保留特定变量的功率延迟特性（PDP）表示，并通过有效地捕捉多变量相关性来增强注意力机制。此外，我们提出了一种轻量级的Swish-Gated Linear Unit（Swish-GLU）Transformer模型（L-SwiGLU Transformer），旨在通过减少计算复杂性而不牺牲定位精度来改进Transformer模型。这些贡献减轻了计算负担，并减少了对大数据集的依赖，使Transformer模型更加高效，并适用于资源受限的场景。提出的token化方法使得朴素Transformer模型在高度NLOS的室内工厂中，实现了第90百分位的定位误差为0.388米，超过了常规的token化方法。L-SwiGLU ViT进一步将误差降低至0.355米，提高了8.51%。此外，所提出模型的性能优于一个大14.1倍的模型，性能提升了46.13%，这进一步突显了其计算效率。 

---
# Large Language Models for Knowledge Graph Embedding Techniques, Methods, and Challenges: A Survey 

**Title (ZH)**: 大语言模型在知识图嵌入技术、方法与挑战综述 

**Authors**: Bingchen Liu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.07766)  

**Abstract**: Large Language Models (LLMs) have attracted a lot of attention in various fields due to their superior performance, aiming to train hundreds of millions or more parameters on large amounts of text data to understand and generate natural language. As the superior performance of LLMs becomes apparent, they are increasingly being applied to knowledge graph embedding (KGE) related tasks to improve the processing results. As a deep learning model in the field of Natural Language Processing (NLP), it learns a large amount of textual data to predict the next word or generate content related to a given text. However, LLMs have recently been invoked to varying degrees in different types of KGE related scenarios such as multi-modal KGE and open KGE according to their task characteristics. In this paper, we investigate a wide range of approaches for performing LLMs-related tasks in different types of KGE scenarios. To better compare the various approaches, we summarize each KGE scenario in a classification. In addition to the categorization methods, we provide a tabular overview of the methods and their source code links for a more direct comparison. In the article we also discuss the applications in which the methods are mainly used and suggest several forward-looking directions for the development of this new research area. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其卓越的性能而在各个领域引起了广泛关注，旨在通过在大量文本数据上训练数百 millions 或更多参数来理解和生成自然语言。随着LLMs性能的优势逐渐显现，它们被越来越多地应用于知识图谱嵌入（KGE）相关的任务中，以提高处理结果的准确性。作为一种自然语言处理（NLP）领域的深度学习模型，它通过学习大量文本数据来预测下一个词或生成与给定文本相关的内容。然而，根据任务特性，LLMs最近在不同类型的KGE相关场景中被不同程度地运用，例如多模态KGE和开放域KGE等。在本文中，我们探讨了在不同类型的KGE场景中执行与LLMs相关的多种方法。为了更好地比较这些方法，我们对每种KGE场景进行了分类总结。除了分类方法外，我们还提供了方法的表格概述及其源代码链接，以便进行更直接的比较。在文章中，我们还讨论了这些方法主要应用于哪些领域，并提出了该新研究领域未来发展的几个前瞻性方向。 

---
# Deep Learning for Disease Outbreak Prediction: A Robust Early Warning Signal for Transcritical Bifurcations 

**Title (ZH)**: 基于深度学习的疾病爆发预测： transcritical 分支中的稳健早期预警信号 

**Authors**: Reza Miry, Amit K. Chakraborty, Russell Greiner, Mark A. Lewis, Hao Wang, Tianyu Guan, Pouria Ramazi  

**Link**: [PDF](https://arxiv.org/pdf/2501.07764)  

**Abstract**: Early Warning Signals (EWSs) are vital for implementing preventive measures before a disease turns into a pandemic. While new diseases exhibit unique behaviors, they often share fundamental characteristics from a dynamical systems perspective. Moreover, measurements during disease outbreaks are often corrupted by different noise sources, posing challenges for Time Series Classification (TSC) tasks. In this study, we address the problem of having a robust EWS for disease outbreak prediction using a best-performing deep learning model in the domain of TSC. We employed two simulated datasets to train the model: one representing generated dynamical systems with randomly selected polynomial terms to model new disease behaviors, and another simulating noise-induced disease dynamics to account for noisy measurements. The model's performance was analyzed using both simulated data from different disease models and real-world data, including influenza and COVID-19. Results demonstrate that the proposed model outperforms previous models, effectively providing EWSs of impending outbreaks across various scenarios. This study bridges advancements in deep learning with the ability to provide robust early warning signals in noisy environments, making it highly applicable to real-world crises involving emerging disease outbreaks. 

**Abstract (ZH)**: 早期预警信号（Early Warning Signals, EWSs）对于在疾病演变成大流行之前采取预防措施至关重要。虽然新出现的疾病会展现出独特的行为，但从动态系统角度看，它们通常具有某些基本特征。此外，在疾病暴发期间的测量数据往往受到不同噪声源的干扰，这给时间序列分类（Time Series Classification, TSC）任务带来了挑战。在本研究中，我们使用在TSC领域表现最佳的深度学习模型来解决疾病暴发预测的稳健EWS问题。我们使用两个模拟数据集对模型进行了训练：一个数据集表示通过随机选择多项式项来模拟新疾病行为的生成动态系统，另一个则模拟由噪声引起的疾病动态，以考虑测量数据的波动性。我们通过使用来自不同疾病模型的模拟数据以及实际数据（包括流感和COVID-19数据）来分析模型的性能。结果显示，提出的模型优于以前的模型，在各种场景下有效地提供了即将发生的暴发的预警信号。本研究将深度学习的进展与在噪声环境中提供稳健早期预警信号的能力相结合，使其在涉及新兴疾病暴发的真实世界危机中具有高度的适用性。 

---
# On the Statistical Capacity of Deep Generative Models 

**Title (ZH)**: 深层生成模型的统计容量研究 

**Authors**: Edric Tam, David B. Dunson  

**Link**: [PDF](https://arxiv.org/pdf/2501.07763)  

**Abstract**: Deep generative models are routinely used in generating samples from complex, high-dimensional distributions. Despite their apparent successes, their statistical properties are not well understood. A common assumption is that with enough training data and sufficiently large neural networks, deep generative model samples will have arbitrarily small errors in sampling from any continuous target distribution. We set up a unifying framework that debunks this belief. We demonstrate that broad classes of deep generative models, including variational autoencoders and generative adversarial networks, are not universal generators. Under the predominant case of Gaussian latent variables, these models can only generate concentrated samples that exhibit light tails. Using tools from concentration of measure and convex geometry, we give analogous results for more general log-concave and strongly log-concave latent variable distributions. We extend our results to diffusion models via a reduction argument. We use the Gromov--Levy inequality to give similar guarantees when the latent variables lie on manifolds with positive Ricci curvature. These results shed light on the limited capacity of common deep generative models to handle heavy tails. We illustrate the empirical relevance of our work with simulations and financial data. 

**Abstract (ZH)**: 深度生成模型常用于从复杂、高维分布中生成样本。尽管它们取得了显著的成功，但其统计性质尚不完全明确。一个常见的假设是，只要有足够的训练数据和足够大的神经网络，深度生成模型的样本在从任何连续目标分布中采样时，采样误差可以任意小。我们建立了一种统一的框架来反驳这一信念。我们证明，包括变分自编码器和生成对抗网络在内的广泛类别深度生成模型并非通用生成器。在高斯潜在变量占主导的情况下，这些模型只能生成集中度较高的样本，这些样本表现出轻尾分布特征。利用测度的集中性和凸几何学工具，我们给出了更广泛类型的对数凹和强对数凹潜在变量分布的相应结果。我们通过归约论证将结果扩展到了扩散模型。借助格罗莫夫-莱维不等式，我们提供了类似的保证，当潜在变量位于具有正里奇曲率的流形上时。这些结果揭示了常见深度生成模型在处理重尾分布时能力有限。我们通过模拟和金融数据对工作的实际应用进行了说明。 

---
# PSReg: Prior-guided Sparse Mixture of Experts for Point Cloud Registration 

**Title (ZH)**: PSReg: 前向引导稀疏专家混合模型的点云配准 

**Authors**: Xiaoshui Huang, Zhou Huang, Yifan Zuo, Yongshun Gong, Chengdong Zhang, Deyang Liu, Yuming Fang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07762)  

**Abstract**: The discriminative feature is crucial for point cloud registration. Recent methods improve the feature discriminative by distinguishing between non-overlapping and overlapping region points. However, they still face challenges in distinguishing the ambiguous structures in the overlapping regions. Therefore, the ambiguous features they extracted resulted in a significant number of outlier matches from overlapping regions. To solve this problem, we propose a prior-guided SMoE-based registration method to improve the feature distinctiveness by dispatching the potential correspondences to the same experts. Specifically, we propose a prior-guided SMoE module by fusing prior overlap and potential correspondence embeddings for routing, assigning tokens to the most suitable experts for processing. In addition, we propose a registration framework by a specific combination of Transformer layer and prior-guided SMoE module. The proposed method not only pays attention to the importance of locating the overlapping areas of point clouds, but also commits to finding more accurate correspondences in overlapping areas. Our extensive experiments demonstrate the effectiveness of our method, achieving state-of-the-art registration recall (95.7\%/79.3\%) on the 3DMatch/3DLoMatch benchmark. Moreover, we also test the performance on ModelNet40 and demonstrate excellent performance. 

**Abstract (ZH)**: 区分性特征在点云配准中至关重要。最近的方法通过区分重叠区域和非重叠区域的点来提高特征的区分性。然而，它们仍面临在重叠区域难以区分模糊结构的挑战。因此，它们提取的模糊特征导致了大量来自重叠区域的异常匹配。为解决这一问题，我们提出了一种基于SMoE的先验导向配准方法，通过将潜在对应关系分配到相同的专家来提高特征的区分性。具体而言，我们提出了一个融合先验重叠信息和潜在对应关系嵌入的先验导向SMoE模块，用于路由和将标记分配给最适合处理的专家。此外，我们提出了一种通过特定组合的Transformer层和先验导向SMoE模块构建的配准框架。所提出的方法不仅注重识别点云重叠区域的重要性，还致力于在这些区域找到更准确的对应关系。我们的广泛实验表明，所提出的方法在3DMatch/3DLoMatch基准上的配准召回率（95.7\%/79.3\%）达到了最佳效果。此外，我们还在ModelNet40上测试了性能，并展示了出色的表现。 

---
# Impatient Bandits: Optimizing for the Long-Term Without Delay 

**Title (ZH)**: 《 impatient bandits: optimizing for the long-term without delay》可以翻译成中文为：

《 impatient臂策略：在无延迟的情况下实现长期优化》

为了更加符合学术规范，可以进一步润色为：

《迫不及待的臂拉动策略：实现长期优化而无需延迟》 

**Authors**: Kelly W. Zhang, Thomas Baldwin-McDonald, Kamil Ciosek, Lucas Maystre, Daniel Russo  

**Link**: [PDF](https://arxiv.org/pdf/2501.07761)  

**Abstract**: Increasingly, recommender systems are tasked with improving users' long-term satisfaction. In this context, we study a content exploration task, which we formalize as a bandit problem with delayed rewards. There is an apparent trade-off in choosing the learning signal: waiting for the full reward to become available might take several weeks, slowing the rate of learning, whereas using short-term proxy rewards reflects the actual long-term goal only imperfectly. First, we develop a predictive model of delayed rewards that incorporates all information obtained to date. Rewards as well as shorter-term surrogate outcomes are combined through a Bayesian filter to obtain a probabilistic belief. Second, we devise a bandit algorithm that quickly learns to identify content aligned with long-term success using this new predictive model. We prove a regret bound for our algorithm that depends on the \textit{Value of Progressive Feedback}, an information theoretic metric that captures the quality of short-term leading indicators that are observed prior to the long-term reward. We apply our approach to a podcast recommendation problem, where we seek to recommend shows that users engage with repeatedly over two months. We empirically validate that our approach significantly outperforms methods that optimize for short-term proxies or rely solely on delayed rewards, as demonstrated by an A/B test in a recommendation system that serves hundreds of millions of users. 

**Abstract (ZH)**: 近年来，推荐系统越来越多地被赋予提高用户长期满意度的任务。在此背景下，我们研究了一个内容探索任务，并将其形式化为具有延迟奖励的多臂_bandit问题。在选择学习信号时存在着明显的权衡：等待完整的奖励数据可能需要数周时间，这会减缓学习速率；而使用短期替代奖励虽然能部分反映长期目标，但并不能完美地体现长期目标。首先，我们开发了一个延迟奖励的预测模型，将迄今为止获得的所有信息都纳入其中。奖励以及短期替代结果通过贝叶斯滤波器结合，以获得一个概率性的信念。其次，我们设计了一种多臂_bandit算法，使用这种新的预测模型快速学会识别与长期成功相一致的内容。我们证明了该算法的后悔界依赖于“逐步反馈的价值”，这是一个信息论度量，衡量了在观察到长期奖励前出现的短期领先指标的质量。我们把这种方法应用于一个播客推荐问题，在此问题中，我们希望推荐用户在两个月内反复收听的节目。通过推荐系统中的A/B测试，我们实证验证了该方法在优化短期替代奖励或仅依赖延迟奖励的方法上显著更优，该推荐系统服务于数亿用户。 

---
# Performance Optimization of Ratings-Based Reinforcement Learning 

**Title (ZH)**: 基于评级的强化学习性能优化 

**Authors**: Evelyn Rose, Devin White, Mingkang Wu, Vernon Lawhern, Nicholas R. Waytowich, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07755)  

**Abstract**: This paper explores multiple optimization methods to improve the performance of rating-based reinforcement learning (RbRL). RbRL, a method based on the idea of human ratings, has been developed to infer reward functions in reward-free environments for the subsequent policy learning via standard reinforcement learning, which requires the availability of reward functions. Specifically, RbRL minimizes the cross entropy loss that quantifies the differences between human ratings and estimated ratings derived from the inferred reward. Hence, a low loss means a high degree of consistency between human ratings and estimated ratings. Despite its simple form, RbRL has various hyperparameters and can be sensitive to various factors. Therefore, it is critical to provide comprehensive experiments to understand the impact of various hyperparameters on the performance of RbRL. This paper is a work in progress, providing users some general guidelines on how to select hyperparameters in RbRL. 

**Abstract (ZH)**: 本文探讨了多种优化方法以提高基于评级的强化学习（RbRL）的表现。RbRL 是一种基于人类评分思想的方法，旨在在无奖励环境中推断奖励函数，随后通过标准强化学习进行策略学习，这需要奖励函数的可用性。具体而言，RbRL 尽可能地最小化交叉熵损失，该损失衡量了人类评分与从推断出的奖励中得出的估计评分之间的差异。因此，低损失意味着人类评分与估计评分的高度一致性。尽管其形式简单，但 RbRL 有着多种超参数，且可能对各种因素非常敏感。因此，全面的实验对于理解各种超参数对 RbRL 表现的影响至关重要。本文尚在进行中，为用户提供了一些关于如何在 RbRL 中选择超参数的一般指导原则。 

---
# BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations 

**Title (ZH)**: BlobGEN-Vid：基于Blob视频表示的组合性文本到视频生成 

**Authors**: Weixi Feng, Chao Liu, Sifei Liu, William Yang Wang, Arash Vahdat, Weili Nie  

**Link**: [PDF](https://arxiv.org/pdf/2501.07647)  

**Abstract**: Existing video generation models struggle to follow complex text prompts and synthesize multiple objects, raising the need for additional grounding input for improved controllability. In this work, we propose to decompose videos into visual primitives - blob video representation, a general representation for controllable video generation. Based on blob conditions, we develop a blob-grounded video diffusion model named BlobGEN-Vid that allows users to control object motions and fine-grained object appearance. In particular, we introduce a masked 3D attention module that effectively improves regional consistency across frames. In addition, we introduce a learnable module to interpolate text embeddings so that users can control semantics in specific frames and obtain smooth object transitions. We show that our framework is model-agnostic and build BlobGEN-Vid based on both U-Net and DiT-based video diffusion models. Extensive experimental results show that BlobGEN-Vid achieves superior zero-shot video generation ability and state-of-the-art layout controllability on multiple benchmarks. When combined with an LLM for layout planning, our framework even outperforms proprietary text-to-video generators in terms of compositional accuracy. 

**Abstract (ZH)**: 现有的视频生成模型在遵循复杂的文本提示和合成多个对象方面存在困难，这凸显了需要额外的语义输入以提高可控性的必要性。在此项工作中，我们提出将视频分解为视觉基本元素——Blob 视频表示，这是一种通用的可控制视频生成表示。基于 Blob 条件，我们开发了一种 Blob 基础的视频扩散模型 BlobGEN-Vid，这使用户能够控制对象运动和精细的外观。特别地，我们引入了一种掩码三维注意模块，有效地提高了帧间区域一致性。此外，我们引入了一个可学习模块来插值文本嵌入，从而使用户能够控制特定帧中的语义并获得平滑的对象过渡。我们证明了我们的框架在模型具有通用性，并基于 U-Net 和 DiT 基础的视频扩散模型构建了 BlobGEN-Vid。广泛的实验结果显示，BlobGEN-Vid 在多个基准测试中实现了优越的零样本视频生成能力和最先进布局可控性。当与布局规划的大型语言模型结合使用时，我们的框架在组成准确性方面甚至优于专有文本到视频生成器。 

---
# Real-Time Decision-Making for Digital Twin in Additive Manufacturing with Model Predictive Control using Time-Series Deep Neural Networks 

**Title (ZH)**: 使用时间序列深度神经网络的模型预测控制进行增材制造数字孪生的实时决策制定 

**Authors**: Yi-Ping Chen, Vispi Karkaria, Ying-Kuan Tsai, Faith Rolark, Daniel Quispe, Robert X. Gao, Jian Cao, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07601)  

**Abstract**: Digital Twin-a virtual replica of a physical system enabling real-time monitoring, model updating, prediction, and decision-making-combined with recent advances in machine learning (ML), offers new opportunities for proactive control strategies in autonomous manufacturing. However, achieving real-time decision-making with Digital Twins requires efficient optimization driven by accurate predictions of highly nonlinear manufacturing systems. This paper presents a simultaneous multi-step Model Predictive Control (MPC) framework for real-time decision-making, using a multi-variate deep neural network (DNN), named Time-Series Dense Encoder (TiDE), as the surrogate model. Different from the models in conventional MPC which only provide one-step ahead prediction, TiDE is capable of predicting future states within the prediction horizon in one shot (multi-step), significantly accelerating MPC. Using Directed Energy Deposition additive manufacturing as a case study, we demonstrate the effectiveness of the proposed MPC in achieving melt pool temperature tracking to ensure part quality, while reducing porosity defects by regulating laser power to maintain melt pool depth constraints. In this work, we first show that TiDE is capable of accurately predicting melt pool temperature and depth. Second, we demonstrate that the proposed MPC achieves precise temperature tracking while satisfying melt pool depth constraints within a targeted dilution range (10%-30%), reducing potential porosity defects. Compared to the PID controller, MPC results in smoother and less fluctuating laser power profiles with competitive or superior melt pool temperature control performance. This demonstrates MPC's proactive control capabilities, leveraging time-series prediction and real-time optimization, positioning it as a powerful tool for future Digital Twin applications and real-time process optimization in manufacturing. 

**Abstract (ZH)**: 数字孪生——一种实时监测、模型更新、预测和决策的物理系统的虚拟副本——结合了最近的机器学习（ML）进展，为自主制造提供了新的主动控制策略的机会。然而，要实现数字孪生的实时决策，需要基于高度非线性制造系统的准确预测来驱动高效的优化。本文提出了一种同时进行多步模型预测控制（MPC）框架，以实现实时决策。该框架使用一个名为时间序列密集编码器（TiDE）的多变量深度神经网络（DNN）作为代理模型。与传统MPC中仅提供一步预测的模型不同，TiDE能够一次性预测整个预测窗口内的未来状态（多步预测），显著加速了MPC的运行。以定向能量沉积（Directed Energy Deposition，DED）增材制造为例，我们展示了所提出的MPC在维持熔池温度以确保零件质量的同时，通过调节激光功率以维持熔池深度约束，有效减少孔隙缺陷的有效性。在本文中，我们首先证明TiDE能够准确预测熔池温度和深度。其次，我们展示了所提出的MPC在目标稀释范围内（10%-30%）实现了精确的温度跟踪，并满足熔池深度约束，从而减少潜在的孔隙缺陷。与PID控制器相比，MPC产生的激光功率曲线更为平滑且波动较小，具有竞争力或优越的熔池温度控制性能。这展示了MPC在利用时间序列预测和实时优化方面的主动控制能力，将其定位为未来数字孪生应用和制造过程中实时过程优化的强大工具。 

---
# Multi-task Domain Adaptation for Computation Offloading in Edge-intelligence Networks 

**Title (ZH)**: 边缘智能网络中计算卸载的多任务领域适应方法 

**Authors**: Runxin Han, Bo Yang, Zhiwen Yu, Xuelin Cao, George C. Alexandropoulos, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07585)  

**Abstract**: In the field of multi-access edge computing (MEC), efficient computation offloading is crucial for improving resource utilization and reducing latency in dynamically changing environments. This paper introduces a new approach, termed as Multi-Task Domain Adaptation (MTDA), aiming to enhance the ability of computational offloading models to generalize in the presence of domain shifts, i.e., when new data in the target environment significantly differs from the data in the source domain. The proposed MTDA model incorporates a teacher-student architecture that allows continuous adaptation without necessitating access to the source domain data during inference, thereby maintaining privacy and reducing computational overhead. Utilizing a multi-task learning framework that simultaneously manages offloading decisions and resource allocation, the proposed MTDA approach outperforms benchmark methods regarding mean squared error and accuracy, particularly in environments with increasing numbers of users. It is observed by means of computer simulation that the proposed MTDA model maintains high performance across various scenarios, demonstrating its potential for practical deployment in emerging MEC applications. 

**Abstract (ZH)**: 在多接入边缘计算（MEC）领域，高效的计算卸载对于提高资源利用率和降低动态变化环境中延迟至关重要。本文提出了一种新的方法，称为多任务域自适应（MTDA），旨在增强计算卸载模型在域转移（即目标环境中新数据与源域数据显著不同）时的泛化能力。所提出的MTDA模型引入了一种教师-学生架构，允许连续自适应，无需在推理时访问源域数据，从而保持隐私并减少计算开销。利用一种同时管理计算卸载决策和资源分配的多任务学习框架，提出的MTDA方法在平均平方误差和准确性方面优于基准方法，尤其是在用户数量不断增加的环境中。计算机仿真结果表明，所提出的MTDA模型在各种场景下均保持了较高的性能，证明了其在新兴MEC应用中实际部署的潜力。 

---
# A Hybrid Framework for Reinsurance Optimization: Integrating Generative Models and Reinforcement Learning 

**Title (ZH)**: 一种再保险优化的混合框架：集成生成模型与强化学习 

**Authors**: Stella C. Dong, James R. Finlay  

**Link**: [PDF](https://arxiv.org/pdf/2501.06404)  

**Abstract**: Reinsurance optimization is critical for insurers to manage risk exposure, ensure financial stability, and maintain solvency. Traditional approaches often struggle with dynamic claim distributions, high-dimensional constraints, and evolving market conditions. This paper introduces a novel hybrid framework that integrates {Generative Models}, specifically Variational Autoencoders (VAEs), with {Reinforcement Learning (RL)} using Proximal Policy Optimization (PPO). The framework enables dynamic and scalable optimization of reinsurance strategies by combining the generative modeling of complex claim distributions with the adaptive decision-making capabilities of reinforcement learning.
The VAE component generates synthetic claims, including rare and catastrophic events, addressing data scarcity and variability, while the PPO algorithm dynamically adjusts reinsurance parameters to maximize surplus and minimize ruin probability. The framework's performance is validated through extensive experiments, including out-of-sample testing, stress-testing scenarios (e.g., pandemic impacts, catastrophic events), and scalability analysis across portfolio sizes. Results demonstrate its superior adaptability, scalability, and robustness compared to traditional optimization techniques, achieving higher final surpluses and computational efficiency.
Key contributions include the development of a hybrid approach for high-dimensional optimization, dynamic reinsurance parameterization, and validation against stochastic claim distributions. The proposed framework offers a transformative solution for modern reinsurance challenges, with potential applications in multi-line insurance operations, catastrophe modeling, and risk-sharing strategy design. 

**Abstract (ZH)**: 再保险优化对于保险公司管理风险敞口、确保财务稳定和维持偿付能力至关重要。传统方法往往难以应对动态理赔分布、高维约束及不断变化的市场条件。本文介绍了一种新型混合框架，该框架结合生成模型（特别是变分自编码器 VAE）与强化学习（RL）中的近端策略优化（PPO）。该框架通过结合复杂的理赔分布生成模型与强化学习的自适应决策能力，实现了再保险策略的动态和可扩展优化。

VAE 成分生成合成理赔数据，包括罕见和灾难性事件，从而解决数据稀缺性和变异性问题，而 PPO 算法则根据最大化剩余储备并最小化破产概率的要求，动态调整再保险参数。通过广泛的实验对该框架进行验证，包括离样本测试、压力测试情景（如疫情冲击、灾难性事件）以及不同投资组合规模下的扩展性分析。结果表明，该框架在适应性、可扩展性和稳健性方面优于传统优化技术，能够在最终储备方面实现更高的收益，并提高计算效率。

主要贡献包括开发了一种用于高维优化的混合方法、动态再保险参数化以及对随机理赔分布的验证。所提出框架为现代再保险挑战提供了一种变革性解决方案，具有在多险种保险业务、灾难模型设计和风险分担策略设计中的潜在应用。 

---
