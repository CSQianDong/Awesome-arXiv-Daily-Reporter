# Graph Neural Controlled Differential Equations For Collaborative Filtering 

**Title (ZH)**: 图神经控制微分方程在协同过滤中的应用 

**Authors**: Ke Xu, Weizhi Zhang, Zihe Song, Yuanjie Zhu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13908)  

**Abstract**: Graph Convolution Networks (GCNs) are widely considered state-of-the-art for recommendation systems. Several studies in the field of recommendation systems have attempted to apply collaborative filtering (CF) into the Neural ODE framework. These studies follow the same idea as LightGCN, which removes the weight matrix or with a discrete weight matrix. However, we argue that weight control is critical for neural ODE-based methods. The importance of weight in creating tailored graph convolution for each node is crucial, and employing a fixed/discrete weight means it cannot adjust over time within the ODE function. This rigidity in the graph convolution reduces its adaptability, consequently hindering the performance of recommendations. In this study, to create an optimal control for Neural ODE-based recommendation, we introduce a new method called Graph Neural Controlled Differential Equations for Collaborative Filtering (CDE-CF). Our method improves the performance of the Graph ODE-based method by incorporating weight control in a continuous manner. To evaluate our approach, we conducted experiments on various datasets. The results show that our method surpasses competing baselines, including GCNs-based models and state-of-the-art Graph ODE-based methods. 

**Abstract (ZH)**: 图卷积网络（GCNs）在推荐系统中广泛被认为是最先进的方法。在推荐系统领域，有多项研究试图将协同过滤（CF）应用到基于神经常微分方程（Neural ODE）的框架中。这些研究遵循与LightGCN相同的想法，即去除权重矩阵或使用离散的权重矩阵。然而，我们认为神经ODE方法中的权重控制至关重要。权重在为每个节点创建定制化的图卷积方面具有重要意义，而固定或离散的权重则无法在ODE函数中随时间调整。这种刚性限制了图卷积的适应性，从而影响推荐系统的性能。在本研究中，为了为基于神经ODE的推荐系统创建最优控制，我们引入了一种名为图神经控制常微分方程用于协同过滤（CDE-CF）的新方法。我们的方法通过以连续的方式引入权重控制，改进了基于图ODE的方法的性能。为了评估该方法，我们在多个数据集上进行了实验。实验结果表明，我们的方法超越了包括基于GCNs的模型和最先进的基于图ODE的方法在内的竞争对手。 

---
# Large Language Model driven Policy Exploration for Recommender Systems 

**Title (ZH)**: Large语言模型驱动的策略探索在推荐系统中的应用 

**Authors**: Jie Wang, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2501.13816)  

**Abstract**: Recent advancements in Recommender Systems (RS) have incorporated Reinforcement Learning (RL), framing the recommendation as a Markov Decision Process (MDP). However, offline RL policies trained on static user data are vulnerable to distribution shift when deployed in dynamic online environments. Additionally, excessive focus on exploiting short-term relevant items can hinder exploration, leading to suboptimal recommendations and negatively impacting long-term user gains. Online RL-based RS also face challenges in production deployment, due to the risks of exposing users to untrained or unstable policies. Large Language Models (LLMs) offer a promising solution to mimic user objectives and preferences for pre-training policies offline to enhance the initial recommendations in online settings. Effectively managing distribution shift and balancing exploration are crucial for improving RL-based RS, especially when leveraging LLM-based pre-training. To address these challenges, we propose an Interaction-Augmented Learned Policy (iALP) that utilizes user preferences distilled from an LLM. Our approach involves prompting the LLM with user states to extract item preferences, learning rewards based on feedback, and updating the RL policy using an actor-critic framework. Furthermore, to deploy iALP in an online scenario, we introduce an adaptive variant, A-iALP, that implements a simple fine-tuning strategy (A-iALP$_{ft}$), and an adaptive approach (A-iALP$_{ap}$) designed to mitigate issues with compromised policies and limited exploration. Experiments across three simulated environments demonstrate that A-iALP introduces substantial performance improvements 

**Abstract (ZH)**: 近年来，推荐系统（RS）的进步已经将强化学习（RL）纳入其中，将推荐问题框架化为马尔可夫决策过程（MDP）。然而，基于静止用户数据离线训练的RL策略在动态在线环境中部署时容易受到分布转移的影响。此外，过分注重利用短期内的相关项目可能会妨碍探索，导致推荐不理想，并对长期用户收益产生负面影响。基于在线RL的推荐系统在实际部署中也面临着挑战，因为这可能使用户接触到未训练或不稳定策略的风险。大型语言模型（LLMs）提供了一种有前景的解决方案，通过从LLM中提取用户偏好进行离线策略训练，以增强在线环境初期的推荐效果。有效地管理分布转移并平衡探索对于提高基于RL的推荐系统至关重要，特别是在利用基于LLM的预训练时。为了解决这些挑战，我们提出了一种增强交互的学习策略（iALP），它利用从LLM中提炼出的用户偏好。我们的方法包括使用LLM提示用户状态以提取项目偏好，基于反馈学习奖励，并使用演员-评论家框架更新RL策略。此外，为了在线环境中部署iALP，我们引入了一个适应性变体A-iALP，其中包括一种简单的微调策略（A-iALP$_{ft}$）和一种适应性方法（A-iALP$_{ap}$），该方法旨在缓解策略缺陷和有限探索所带来的问题。在三个模拟环境中进行的实验表明，A-iALP带来了显著的性能改进。 

---
# EICopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents 

**Title (ZH)**: EICopilot: 在大规模知识图谱中利用LLM驱动的代理进行搜索与探索企业信息 

**Authors**: Yuhui Yun, Huilong Ye, Xinru Li, Ruojia Li, Jingfeng Deng, Li Li, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.13746)  

**Abstract**: The paper introduces EICopilot, an novel agent-based solution enhancing search and exploration of enterprise registration data within extensive online knowledge graphs like those detailing legal entities, registered capital, and major shareholders. Traditional methods necessitate text-based queries and manual subgraph explorations, often resulting in time-consuming processes. EICopilot, deployed as a chatbot via Baidu Enterprise Search, improves this landscape by utilizing Large Language Models (LLMs) to interpret natural language queries. This solution automatically generates and executes Gremlin scripts, providing efficient summaries of complex enterprise relationships. Distinct feature a data pre-processing pipeline that compiles and annotates representative queries into a vector database of examples for In-context learning (ICL), a comprehensive reasoning pipeline combining Chain-of-Thought with ICL to enhance Gremlin script generation for knowledge graph search and exploration, and a novel query masking strategy that improves intent recognition for heightened script accuracy. Empirical evaluations demonstrate the superior performance of EICopilot, including speed and accuracy, over baseline methods, with the \emph{Full Mask} variant achieving a syntax error rate reduction to as low as 10.00% and an execution correctness of up to 82.14%. These components collectively contribute to superior querying capabilities and summarization of intricate datasets, positioning EICopilot as a groundbreaking tool in the exploration and exploitation of large-scale knowledge graphs for enterprise information search. 

**Abstract (ZH)**: 本文介绍了EICopilot，这是一种基于代理的新颖解决方案，用于增强在线知识图谱中的企业注册数据搜索与探索，这些知识图谱详细描述了法人实体、注册资金和主要股东等方面的信息。传统方法需要依赖文本查询并进行人工子图探索，经常导致耗时的过程。EICopilot 作为通过百度企业搜索部署的聊天机器人，通过利用大型语言模型（LLMs）来解释自然语言查询，从而改进了这一现状。该解决方案会自动生成并执行Gremlin脚本，提供对企业复杂关系的高效总结。其独特功能包括数据预处理管道，该管道将代表性查询编译并标注为向量数据库中的示例，用于情境学习（ICL）；结合Chain-of-Thought与ICL的全面推理管道，以增强Gremlin脚本生成能力，用于知识图谱搜索和探索；以及一种新的查询遮蔽策略，提高了意图识别的准确性，从而提高脚本准确性。实证评估表明，与基线方法相比，EICopilot 在速度和准确性方面表现出更优越的性能，其中“全遮蔽”变体的语法错误率降低至最低10.00%，执行正确率达到82.14%。这些组件共同提高了查询能力和复杂数据集的总结能力，将EICopilot 定位为探索和利用大规模知识图谱进行企业信息搜索的一种突破性工具。 

---
# AirTOWN: A Privacy-Preserving Mobile App for Real-time Pollution-Aware POI Suggestion 

**Title (ZH)**: AirTOWN：一种保护隐私的移动应用程序，用于实时污染感知兴趣点建议 

**Authors**: Giuseppe Fasano, Yashar Deldjoo, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2501.13608)  

**Abstract**: This demo paper presents \airtown, a privacy-preserving mobile application that provides real-time, pollution-aware recommendations for points of interest (POIs) in urban environments. By combining real-time Air Quality Index (AQI) data with user preferences, the proposed system aims to help users make health-conscious decisions about the locations they visit. The application utilizes collaborative filtering for personalized suggestions, and federated learning for privacy protection, and integrates AQI data from sensor networks in cities such as Bari, Italy, and Cork, UK. In areas with sparse sensor coverage, interpolation techniques approximate AQI values, ensuring broad applicability. This system offers a poromsing, health-oriented POI recommendation solution that adapts dynamically to current urban air quality conditions while safeguarding user privacy. 

**Abstract (ZH)**: 本文演示了一种名为\airtown的隐私保护移动应用，该应用能够在城市环境中为兴趣点（POIs）提供实时、污染感知的推荐服务。通过结合实时空气质量指数（AQI）数据和用户偏好，所提出的系统旨在帮助用户做出关于访问地点的健康决策。该应用利用协作过滤进行个性化建议，并利用联邦学习保护隐私，同时整合来自意大利巴里和英国科尔克等城市的传感器网络采集的AQI数据。在传感器覆盖稀疏的地区，通过插值技术来估算AQI值，确保该系统的广泛适用性。该系统提供了一种有前景的、以健康为导向的POI推荐解决方案，能够根据当前的城市空气质量条件动态调整，并同时保障用户隐私。 

---
# MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems 

**Title (ZH)**: MixRec：个体与集体混合增强推荐系统的数据增强 

**Authors**: Yi Zhang, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13579)  

**Abstract**: The core of the general recommender systems lies in learning high-quality embedding representations of users and items to investigate their positional relations in the feature space. Unfortunately, data sparsity caused by difficult-to-access interaction data severely limits the effectiveness of recommender systems. Faced with such a dilemma, various types of self-supervised learning methods have been introduced into recommender systems in an attempt to alleviate the data sparsity through distribution modeling or data augmentation. However, most data augmentation relies on elaborate manual design, which is not only not universal, but the bloated and redundant augmentation process may significantly slow down model training progress. To tackle these limitations, we propose a novel Dual Mixing-based Recommendation Framework (MixRec) to empower data augmentation as we wish. Specifically, we propose individual mixing and collective mixing, respectively. The former aims to provide a new positive sample that is unique to the target (user or item) and to make the pair-wise recommendation loss benefit from it, while the latter aims to portray a new sample that contains group properties in a batch. The two mentioned mixing mechanisms allow for data augmentation with only one parameter that does not need to be set multiple times and can be done in linear time complexity. Besides, we propose the dual-mixing contrastive learning to maximize the utilization of these new-constructed samples to enhance the consistency between pairs of positive samples. Experimental results on four real-world datasets demonstrate the effectiveness of MixRec in terms of recommendation performance, training efficiency, sparsity resistance, and usability. 

**Abstract (ZH)**: 通用推荐系统的核心在于学习高质量的用户和项目嵌入表示，以探究它们在特征空间中的位置关系。然而，由于交互数据难以获取导致的数据稀疏性严重限制了推荐系统的有效性。为了解决这一困境，各种类型的自监督学习方法被引入到推荐系统中，试图通过分布建模或数据增强来缓解数据稀疏性。然而，大多数数据增强依赖于复杂的手动设计，不仅缺乏普遍适用性，而且冗长和冗余的增强过程可能显著减慢模型的训练进度。为解决这些局限性，我们提出了一种新颖的基于双混合的推荐框架（MixRec），以按需增强数据。具体而言，我们提出了个体混合和集体混合。前者旨在为目标（用户或项目）提供一个新的唯一正样本，从而使成对推荐损失从中受益，而后者旨在在批量数据中描绘一个包含群体特性的新样本。上述两种混合机制允许使用单一参数进行数据增强，无需多次设置，并且可以在线性时间复杂度下完成。此外，我们提出了双混合对比学习，以最大化利用这些新构造的样本，增强正样本对之间的一致性。在四个真实世界数据集上的实验结果表明，MixRec 在推荐性能、训练效率、稀疏性抵抗性和易用性方面都具有有效性。 

---
# Federated Conformance Checking 

**Title (ZH)**: 联邦符合性检查 

**Authors**: Majid Rafiei, Mahsa Pourbafrani, Wil M.P. van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2501.13576)  

**Abstract**: Conformance checking is a crucial aspect of process mining, where the main objective is to compare the actual execution of a process, as recorded in an event log, with a reference process model, e.g., in the form of a Petri net or a BPMN. Conformance checking enables identifying deviations, anomalies, or non-compliance instances. It offers different perspectives on problems in processes, bottlenecks, or process instances that are not compliant with the model. Performing conformance checking in federated (inter-organizational) settings allows organizations to gain insights into the overall process execution and to identify compliance issues across organizational boundaries, which facilitates process improvement efforts among collaborating entities. In this paper, we propose a privacy-aware federated conformance-checking approach that allows for evaluating the correctness of overall cross-organizational process models, identifying miscommunications, and quantifying their costs. For evaluation, we design and simulate a supply chain process with three organizations engaged in purchase-to-pay, order-to-cash, and shipment processes. We generate synthetic event logs for each organization as well as the complete process, and we apply our approach to identify and evaluate the cost of pre-injected miscommunications. 

**Abstract (ZH)**: 符合学术规范的中文翻译如下：

合规性检查是过程挖掘中的关键方面，其主要目标是将事件日志中记录的实际过程执行情况与参考过程模型进行比较，例如Petri网或BPMN形式。合规性检查能够识别偏差、异常或不合规实例，提供关于过程问题、瓶颈或不符合模型的过程实例的不同视角。在联邦（跨组织）环境中进行合规性检查，使组织能够深入了解整个过程的执行情况，并识别跨组织边界的合规性问题，从而促进协作实体的过程改进努力。本文提出了一种隐私意识的联邦合规性检查方法，该方法能够评估跨组织整体过程模型的正确性，识别沟通失误，并量化其成本。为评估目的，我们设计并模拟了一个供应链过程，涉及三个组织参与采购到支付、订单到现金以及发货过程。我们为每个组织和整个过程生成了合成事件日志，并将我们的方法应用于识别和评估预先注入的沟通失误的成本。 

---
# Billion-scale Similarity Search Using a Hybrid Indexing Approach with Advanced Filtering 

**Title (ZH)**: 使用高级过滤与混合索引方法的大规模相似性搜索 

**Authors**: Simeon Emanuilov, Aleksandar Dimov  

**Link**: [PDF](https://arxiv.org/pdf/2501.13442)  

**Abstract**: This paper presents a novel approach for similarity search with complex filtering capabilities on billion-scale datasets, optimized for CPU inference. Our method extends the classical IVF-Flat index structure to integrate multi-dimensional filters. The proposed algorithm combines dense embeddings with discrete filtering attributes, enabling fast retrieval in high-dimensional spaces. Designed specifically for CPU-based systems, our disk-based approach offers a cost-effective solution for large-scale similarity search. We demonstrate the effectiveness of our method through a case study, showcasing its potential for various practical uses. 

**Abstract (ZH)**: 本文提出了一种针对十亿规模数据集的新型相似性搜索方法，该方法具备复杂的过滤能力，并且优化了基于CPU的推理性能。我们的方法将经典的IVF-Flat索引结构扩展，以集成多维过滤器。所提出的算法结合了密集嵌入和离散过滤属性，能够在高维空间中实现快速检索。该方法专为基于CPU的系统设计，通过磁盘基础的方法提供了一种成本有效的解决方案，适用于大规模相似性搜索。我们通过一个案例研究展示了该方法的有效性，展示了其在各种实际应用中的潜力。 

---
# Full-Stack Optimized Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation 

**Title (ZH)**: 全面优化的大规模语言模型用于推荐中的终生序列行为理解 

**Authors**: Rong Shan, Jiachen Zhu, Jianghao Lin, Chenxu Zhu, Bo Chen, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13344)  

**Abstract**: In this paper, we address the lifelong sequential behavior incomprehension problem in large language models (LLMs) for recommendation, where LLMs struggle to extract useful information from long user behavior sequences, even within their context limits. To tackle this, we propose ReLLaX (Retrieval-enhanced Large Language models Plus), a framework offering optimization across data, prompt, and parameter levels. At the data level, we introduce Semantic User Behavior Retrieval (SUBR) to reduce sequence heterogeneity, making it easier for LLMs to extract key information. For prompt-level enhancement, we employ Soft Prompt Augmentation (SPA) to inject collaborative knowledge, aligning item representations with recommendation tasks and improving LLMs's exploration of item relationships. Finally, at the parameter level, we propose Component Fully-interactive LoRA (CFLoRA), which enhances LoRA's expressiveness by enabling interactions between its components, allowing better capture of sequential information. Moreover, we present new perspectives to compare current LoRA-based LLM4Rec methods, i.e. from both a composite and a decomposed view. We theoretically demonstrate that the ways they employ LoRA for recommendation are degraded versions of our CFLoRA, with different constraints on atom component interactions. Extensive experiments on three public datasets demonstrate ReLLaX's superiority over existing baselines and its ability to mitigate lifelong sequential behavior incomprehension effectively. 

**Abstract (ZH)**: 在本文中，我们针对大型语言模型（LLMs）在推荐中的终身序列行为理解问题提出了解决方案，该问题在于LLMs难以从长时间用户行为序列中提取有用信息，即使是在其上下文限制内也是如此。为了解决这一问题，我们提出了ReLLaX（检索增强的大规模语言模型+），这是一个在数据、提示和参数层次上提供优化的框架。在数据层面上，我们引入了语义用户行为检索（SUBR），以减少序列的异质性，使LLMs更容易提取关键信息。在提示层面上，我们采用了软提示增强（SPA）来注入协作知识，使项目表示与推荐任务对齐，并改善LLMs在项目关系探索方面的表现。最后，在参数层面上，我们提出了组件全交互LoRA（CFLoRA），通过使LoRA的不同组件能够相互作用，增强了LoRA的可表达性，使其能够更好地捕捉序列信息。此外，我们从合成和分解的视角提出了新的方法比较视角，讨论了当前基于LoRA的LLM4Rec方法。理论上证明，他们使用的LoRA推荐方式实际上是CFLoRA的不同约束版本。在三个公开数据集上的大量实验表明，ReLLaX在对比现有的基线方法时具有明显的优势，并能够有效地缓解终身序列行为理解问题。 

---
# PCSI -- The Platform for Content-Structure Inference 

**Title (ZH)**: PCSI——内容结构推断平台 

**Authors**: Caleb Malchik, Joan Feigenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2501.13272)  

**Abstract**: The Platform for Content-Structure Inference (PCSI, pronounced "pixie") facilitates the sharing of information about the process of converting Web resources into structured content objects that conform to a predefined format. PCSI records encode methods for deriving structured content from classes of URLs, and report the results of applying particular methods to particular URLs. The methods are scripts written in Hex, a variant of Awk with facilities for traversing the HTML DOM. 

**Abstract (ZH)**: 内容结构推理平台（PCSI，发音为“pixie”）促进了将网络资源转换为符合预定义格式的结构化内容对象的信息共享。PCSI记录编码了从特定URL类中提取结构化内容的方法，并报告将特定方法应用于特定URL的结果。这些方法是以Hex编写的脚本，Hex是Awk的一种变体，具有遍历HTML DOM的功能。 

---
# Exploring GPT's Ability as a Judge in Music Understanding 

**Title (ZH)**: 探索GPT在音乐理解中的裁判能力 

**Authors**: Kun Fang, Ziyu Wang, Gus Xia, Ichiro Fujinaga  

**Link**: [PDF](https://arxiv.org/pdf/2501.13261)  

**Abstract**: Recent progress in text-based Large Language Models (LLMs) and their extended ability to process multi-modal sensory data have led us to explore their applicability in addressing music information retrieval (MIR) challenges. In this paper, we use a systematic prompt engineering approach for LLMs to solve MIR problems. We convert the music data to symbolic inputs and evaluate LLMs' ability in detecting annotation errors in three key MIR tasks: beat tracking, chord extraction, and key estimation. A concept augmentation method is proposed to evaluate LLMs' music reasoning consistency with the provided music concepts in the prompts. Our experiments tested the MIR capabilities of Generative Pre-trained Transformers (GPT). Results show that GPT has an error detection accuracy of 65.20%, 64.80%, and 59.72% in beat tracking, chord extraction, and key estimation tasks, respectively, all exceeding the random baseline. Moreover, we observe a positive correlation between GPT's error finding accuracy and the amount of concept information provided. The current findings based on symbolic music input provide a solid ground for future LLM-based MIR research. 

**Abstract (ZH)**: 近年来，基于文本的大规模语言模型（LLMs）的发展及其扩展处理多模态感官数据的能力，促使我们探索其在解决音乐信息检索（MIR）挑战方面的适用性。在本文中，我们采用系统性的提示工程方法来解决MIR问题。我们将音乐数据转换为符号输入，并评估LLMs在三个关键MIR任务（节奏跟踪、和弦提取和调性估计）中检测注释错误的能力。我们提出了一种概念增强方法，以评估LLMs在提示中提供的音乐概念下的音乐推理一致性。我们的实验测试了生成预训练变换器（GPT）的MIR能力。结果显示，GPT在节奏跟踪、和弦提取和调性估计任务中的错误检测准确率分别为65.20%、64.80%和59.72%，均超过了随机基线。此外，我们观察到GPT发现错误的准确率与提供给提示的概念信息量之间存在正相关关系。基于符号音乐输入的当前研究成果为未来的LLM驱动的MIR研究奠定了坚实基础。 

---
# RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering 

**Title (ZH)**: RAMQA：一种统一的检索增强多模态问答框架 

**Authors**: Yang Bai, Christan Earl Grant, Daisy Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13297)  

**Abstract**: Multi-modal retrieval-augmented Question Answering (MRAQA), integrating text and images, has gained significant attention in information retrieval (IR) and natural language processing (NLP). Traditional ranking methods rely on small encoder-based language models, which are incompatible with modern decoder-based generative large language models (LLMs) that have advanced various NLP tasks. To bridge this gap, we propose RAMQA, a unified framework combining learning-to-rank methods with generative permutation-enhanced ranking techniques. We first train a pointwise multi-modal ranker using LLaVA as the backbone. Then, we apply instruction tuning to train a LLaMA model for re-ranking the top-k documents using an innovative autoregressive multi-task learning approach. Our generative ranking model generates re-ranked document IDs and specific answers from document candidates in various permutations. Experiments on two MRAQA benchmarks, WebQA and MultiModalQA, show significant improvements over strong baselines, highlighting the effectiveness of our approach. Code and data are available at: this https URL 

**Abstract (ZH)**: 多模态检索增强的问答（MRAQA），结合文本和图像的信息检索（IR）和自然语言处理（NLP）领域引起了广泛关注。传统的排名方法依赖于小型的基于编码器的语言模型，这些模型与现代基于解码器的生成型大型语言模型（LLMs）不兼容，后者在各种NLP任务上取得了显著进展。为解决这一问题，我们提出了一种名为RAMQA的统一框架，该框架结合了学习排序方法与生成型排列增强排序技术。首先，我们使用LLaVA作为骨干模型训练一个点积多模态排名器。随后，采用指令调整方法，利用一种创新的自回归多任务学习策略训练一个LLaMA模型，用于对前k个文档进行重新排名。我们的生成型排名模型能够从文档候选集中生成重新排名的文档ID及其特定答案，在多种排列下生成。实验结果表明，RAMQA在两个MRAQA基准数据集WebQA和MultiModalQA上的表现显著优于强基线，突显了该方法的有效性。代码和数据可在以下链接获取：this https URL 

---
