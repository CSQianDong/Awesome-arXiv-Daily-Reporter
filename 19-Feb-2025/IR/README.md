# Learning More Effective Representations for Dense Retrieval through Deliberate Thinking Before Search 

**Title (ZH)**: 在搜索前进行精心思考以学习更有效表示方法的密集检索研究 

**Authors**: Yifan Ji, Zhipeng Xu, Zhenghao Liu, Yukun Yan, Shi Yu, Yishan Li, Zhiyuan Liu, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.12974)  

**Abstract**: Recent dense retrievers usually thrive on the emergency capabilities of Large Language Models (LLMs), using them to encode queries and documents into an embedding space for retrieval. These LLM-based dense retrievers have shown promising performance across various retrieval scenarios. However, relying on a single embedding to represent documents proves less effective in capturing different perspectives of documents for matching. In this paper, we propose Deliberate Thinking based Dense Retriever (DEBATER), which enhances these LLM-based retrievers by enabling them to learn more effective document representations through a step-by-step thinking process. DEBATER introduces the Chain-of-Deliberation mechanism to iteratively optimize document representations using a continuous chain of thought. To consolidate information from various thinking steps, DEBATER also incorporates the Self Distillation mechanism, which identifies the most informative thinking steps and integrates them into a unified text embedding. Experimental results show that DEBATER significantly outperforms existing methods across several retrieval benchmarks, demonstrating superior accuracy and robustness. All codes are available at this https URL. 

**Abstract (ZH)**: 近年来，密集检索器通常依赖大型语言模型（LLMs）的应急能力，利用这些模型将查询和文档编码到一个嵌入空间中进行检索。这些基于LLM的密集检索器在各种检索场景中表现出有希望的性能。然而，仅依靠单一嵌入来表示文档在捕捉文档的多角度方面效果不佳。在本文中，我们提出了一种精细思考为基础的密集检索器（DEBATER），通过逐步的思考过程增强这些基于LLM的检索器，使其学习更有效的文档表示。DEBATER引入了链式思考机制，通过连续的思想链条迭代优化文档表示。为了整合来自不同思考步骤的信息，DEBATER还引入了自我蒸馏机制，该机制识别出最具信息量的思考步骤，并将其整合进一个统一的文本嵌入中。实验结果表明，DEBATER在多个检索基准上显著优于现有方法，显示出更高的准确性和鲁棒性。所有代码已在以下网址提供：此 https URL。 

---
# Introducing Context Information in Lifelong Sequential Modeling using Temporal Convolutional Networks 

**Title (ZH)**: 使用时间卷积网络在终身序列建模中引入上下文信息 

**Authors**: Ting Guo, Zhaoyang Yang, Qinsong Zeng, Ming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12634)  

**Abstract**: The importance of lifelong sequential modeling (LSM) is growing in the realm of social media recommendation systems. A key component in this process is the attention module, which derives interest representations with respect to candidate items from the sequence. Typically, attention modules function in a point-wise fashion, concentrating only on the relevance of individual items in the sequence to the candidate item. However, the context information in the neighboring items that is useful for more accurately evaluating the significance of each item has not been taken into account. In this study, we introduce a novel network which employs the Temporal Convolutional Network (TCN) to generate context-aware representations for each item throughout the lifelong sequence. These improved representations are then utilized in the attention module to produce context-aware interest representations. Expanding on this TCN framework, we present a enhancement module which includes multiple TCN layers and their respective attention modules to capture interest representations across different context scopes. Additionally, we also incorporate a lightweight sub-network to create convolution filters based on users' basic profile features. These personalized filters are then applied in the TCN layers instead of the original global filters to produce more user-specific representations. We performed experiments on both a public dataset and a proprietary dataset. The findings indicate that the proposed network surpasses existing methods in terms of prediction accuracy and online performance metrics. 

**Abstract (ZH)**: 在社交媒体推荐系统领域，终生序列建模（LSM）的重要性日益凸显。在这个过程中，注意力模块是一个关键组件，它能够从序列中提取出与候选项目相关的兴趣表示。传统上，注意力模块通常以点积的方式运作，仅关注序列中各个项目与候选项目的相关性。然而，相邻项目的上下文信息对于更准确地评估每个项目的意义具有重要作用，这些信息尚未被充分考虑。在此研究中，我们引入了一种新颖的网络结构，通过时间卷积网络（TCN）生成终生序列中每个项目的上下文感知表示。这些改进后的表示随后被应用于注意力模块中，以生成上下文感知的兴趣表示。在此TCN框架的基础上，我们提出了一个增强模块，该模块包含多个TCN层及其相应的注意力模块，用于捕捉不同上下文范围内的兴趣表示。此外，我们还引入了一个轻量级子网络，基于用户的基本信息特征生成卷积滤波器。这些个性化的滤波器随后被应用于TCN层中，替代原来的全局滤波器，以生成更具体的用户表示。我们对公共数据集和 proprietary 数据集进行了实验。结果显示，所提出的网络在预测准确性和在线性能指标上均超过了现有方法。 

---
# G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation 

**Title (ZH)**: G-Refer：解释性推荐增强的图检索大语言模型 

**Authors**: Yuhan Li, Xinni Zhang, Linhao Luo, Heng Chang, Yuxiang Ren, Irwin King, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12586)  

**Abstract**: Explainable recommendation has demonstrated significant advantages in informing users about the logic behind recommendations, thereby increasing system transparency, effectiveness, and trustworthiness. To provide personalized and interpretable explanations, existing works often combine the generation capabilities of large language models (LLMs) with collaborative filtering (CF) information. CF information extracted from the user-item interaction graph captures the user behaviors and preferences, which is crucial for providing informative explanations. However, due to the complexity of graph structure, effectively extracting the CF information from graphs still remains a challenge. Moreover, existing methods often struggle with the integration of extracted CF information with LLMs due to its implicit representation and the modality gap between graph structures and natural language explanations. To address these challenges, we propose G-Refer, a framework using graph retrieval-augmented large language models (LLMs) for explainable recommendation. Specifically, we first employ a hybrid graph retrieval mechanism to retrieve explicit CF signals from both structural and semantic perspectives. The retrieved CF information is explicitly formulated as human-understandable text by the proposed graph translation and accounts for the explanations generated by LLMs. To bridge the modality gap, we introduce knowledge pruning and retrieval-augmented fine-tuning to enhance the ability of LLMs to process and utilize the retrieved CF information to generate explanations. Extensive experiments show that G-Refer achieves superior performance compared with existing methods in both explainability and stability. Codes and data are available at this https URL. 

**Abstract (ZH)**: 可解释推荐已经在告知用户推荐逻辑方面展现了显著优势，从而增加了系统的透明度、有效性和可信度。为了提供个性化且可解释的解释，现有研究通常结合大型语言模型（LLMs）的生成能力和基于协同过滤（CF）的信息。从用户-项交互图中提取的CF信息捕捉了用户行为和偏好，这是提供有用解释的关键。然而，由于图结构的复杂性，有效地从图中提取CF信息仍然是一项挑战。此外，现有方法在将提取的CF信息与LLMs集成时常常遇到困难，这主要是由于其隐式表示以及图结构与自然语言解释之间的模态差距。为了解决这些挑战，我们提出了一种基于图检索增强的大语言模型（LLMs）的框架——G-Refer。具体而言，我们首先采用一种混合图检索机制，从结构和语义两个视角检索显性的CF信号。通过提出的图翻译将检索到的CF信息明确定义为易于理解的文本，并用于支持LLMs生成解释。为了弥合模态差距，我们引入了知识精简和检索增强微调，以增强LLMs处理和利用检索到的CF信息生成解释的能力。大量的实验表明，G-Refer 在可解释性和稳定性方面均优于现有方法。相关代码和数据可以在此处获取：[提供链接]。 

---
# From Principles to Applications: A Comprehensive Survey of Discrete Tokenizers in Generation, Comprehension, Recommendation, and Information Retrieval 

**Title (ZH)**: 从原理到应用：离散分词器在生成、理解和推荐以及信息检索中的综述 

**Authors**: Jian Jia, Jingtong Gao, Ben Xue, Junhao Wang, Qingpeng Cai, Quan Chen, Xiangyu Zhao, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2502.12448)  

**Abstract**: Discrete tokenizers have emerged as indispensable components in modern machine learning systems, particularly within the context of autoregressive modeling and large language models (LLMs). These tokenizers serve as the critical interface that transforms raw, unstructured data from diverse modalities into discrete tokens, enabling LLMs to operate effectively across a wide range of tasks. Despite their central role in generation, comprehension, and recommendation systems, a comprehensive survey dedicated to discrete tokenizers remains conspicuously absent in the literature. This paper addresses this gap by providing a systematic review of the design principles, applications, and challenges of discrete tokenizers. We begin by dissecting the sub-modules of tokenizers and systematically demonstrate their internal mechanisms to provide a comprehensive understanding of their functionality and design. Building on this foundation, we synthesize state-of-the-art methods, categorizing them into multimodal generation and comprehension tasks, and semantic tokens for personalized recommendations. Furthermore, we critically analyze the limitations of existing tokenizers and outline promising directions for future research. By presenting a unified framework for understanding discrete tokenizers, this survey aims to guide researchers and practitioners in addressing open challenges and advancing the field, ultimately contributing to the development of more robust and versatile AI systems. 

**Abstract (ZH)**: 离散分词器已成为现代机器学习系统中的不可或缺组成部分，尤其是在自回归建模和大规模语言模型（LLMs）的背景下。这些分词器充当关键接口，将来自多种模态的原始未结构化数据转换为离散的标记，使LLMs能够有效执行各种任务。尽管分词器在生成、理解和推荐系统中发挥着核心作用，但关于离散分词器的全面综述仍然在文献中显得异常缺失。本文旨在填补这一空白，通过系统性地回顾离散分词器的设计原则、应用和挑战来提供这一综述。

我们首先解构分词器的子模块，并系统地展示其内部机制，以提供对其功能和设计的全面理解。在此基础上，我们综合分析了最新的方法，并将其分类为多模态生成和理解任务以及语义标记以实现个性化推荐。此外，我们对现有分词器的局限性进行了批判性分析，并概述了未来研究的潜在方向。通过提供一个统一的框架来理解离散分词器，本文旨在为研究人员和从业人员指导解决开放挑战并推进该领域的发展，最终促进更稳健和多功能AI系统的开发。 

---
# HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation 

**Title (ZH)**: HopRAG：基于逻辑感知的多跳推理检索增强生成 

**Authors**: Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12442)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often struggle with imperfect retrieval, as traditional retrievers focus on lexical or semantic similarity rather than logical relevance. To address this, we propose HopRAG, a novel RAG framework that augments retrieval with logical reasoning through graph-structured knowledge exploration. During indexing, HopRAG constructs a passage graph, with text chunks as vertices and logical connections established via LLM-generated pseudo-queries as edges. During retrieval, it employs a retrieve-reason-prune mechanism: starting with lexically or semantically similar passages, the system explores multi-hop neighbors guided by pseudo-queries and LLM reasoning to identify truly relevant ones. Extensive experiments demonstrate HopRAG's superiority, achieving 76.78\% higher answer accuracy and 65.07\% improved retrieval F1 score compared to conventional methods. The repository is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）系统常常难以应对不完美的检索问题，因为传统的检索器主要关注词汇或语义相似性，而忽视了逻辑相关性。为了解决这一问题，我们提出了HopRAG，一种新颖的RAG框架，通过图结构的知识探索引入逻辑推理来增强检索。在索引过程中，HopRAG 构造一个段落图，其中文本片段作为顶点，通过LLM生成的伪查询建立逻辑连接，作为边。在检索过程中，系统采用“检索-推理-修剪”的机制：从词汇或语义相似的段落开始，系统根据伪查询和LLM推理逐步探索多跳邻居，以识别真正相关的内容。广泛的实验结果证明了HopRAG 的优越性，其答案准确性比传统方法高76.78%，检索F1分数提高了65.07%。代码库可访问 [该链接]。 

---
# Solving the Cold Start Problem on One's Own as an End User via Preference Transfer 

**Title (ZH)**: 作为一个终端用户通过偏好转移自行解决冷启动问题 

**Authors**: Ryoma Sato  

**Link**: [PDF](https://arxiv.org/pdf/2502.12398)  

**Abstract**: We propose a new approach that enables end users to directly solve the cold start problem by themselves. The cold start problem is a common issue in recommender systems, and many methods have been proposed to address the problem on the service provider's side. However, when the service provider does not take action, users are left with poor recommendations and no means to improve their experience. We propose an algorithm, Pretender, that allows end users to proactively solve the cold start problem on their own. Pretender does not require any special support from the service provider and can be deployed independently by users. We formulate the problem as minimizing the distance between the source and target distributions and optimize item selection from the target service accordingly. Furthermore, we establish theoretical guarantees for Pretender based on a discrete quadrature problem. We conduct experiments on real-world datasets to demonstrate the effectiveness of Pretender. 

**Abstract (ZH)**: 我们提出了一种新的方法，使最终用户可以直接自己解决冷启动问题。冷启动问题是推荐系统中常见的问题，许多方法已被提出，旨在通过服务提供商的干预来解决该问题。然而，当服务提供商不采取行动时，用户只能收到质量较差的推荐，且缺乏改善其体验的方法。我们提出了一种名为 Pretender 的算法，允许最终用户主动解决冷启动问题。Pretender 不需要服务提供商的特殊支持，并可由用户独立部署。我们将问题形式化为最小化源分布和目标分布之间的距离，并相应地优化目标服务中的项目选择。此外，我们基于离散 quadrature 问题为 Pretender 建立了理论保证。我们在实际数据集上进行了实验，以证明 Pretender 的有效性。 

---
# REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark 

**Title (ZH)**: REAL-MM-RAG：一个现实世界多模态检索基准 

**Authors**: Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay, Leonid Karlinsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.12342)  

**Abstract**: Accurate multi-modal document retrieval is crucial for Retrieval-Augmented Generation (RAG), yet existing benchmarks do not fully capture real-world challenges with their current design. We introduce REAL-MM-RAG, an automatically generated benchmark designed to address four key properties essential for real-world retrieval: (i) multi-modal documents, (ii) enhanced difficulty, (iii) Realistic-RAG queries and (iv) accurate labeling. Additionally, we propose a multi-difficulty-level scheme based on query rephrasing to evaluate models' semantic understanding beyond keyword matching. Our benchmark reveals significant model weaknesses, particularly in handling table-heavy documents and robustness to query rephrasing. To mitigate these shortcomings, we curate a rephrased training set and introduce a new finance-focused, table-heavy dataset. Fine-tuning on these datasets enables models to achieve state-of-the-art retrieval performance on REAL-MM-RAG benchmark. Our work offers a better way to evaluate and improve retrieval in multi-modal RAG systems while also providing training data and models that address current limitations. 

**Abstract (ZH)**: 准确的多模态文档检索对于检索增强生成（RAG）至关重要，但现有的基准测试在当前设计中未能充分捕捉到实际应用中的挑战。我们引入了REAL-MM-RAG，这是一个自动生成的基准测试，旨在解决四个对于实际检索至关重要的关键属性：（i）多模态文档，（ii）增强的难度，（iii）现实场景下的RAG查询，（iv）准确的标注。此外，我们提出了基于查询重述的多难度层次方案，以评估模型超越关键词匹配的语义理解能力。我们的基准测试揭示了模型在应对表格密集型文档和查询重述下的鲁棒性方面的显著弱点。为了缓解这些不足，我们精心挑选了一个重述训练集，并引入了一个专注于金融且表格密集型的新数据集。在这些数据集上进行微调使模型在REAL-MM-RAG基准测试上实现了最先进的检索性能。我们的工作为评估和改进多模态RAG系统的检索提供了更好的方法，同时也提供了应对当前局限性的训练数据和模型。 

---
# Towards Text-Image Interleaved Retrieval 

**Title (ZH)**: 面向文本-图像交替检索的方向 

**Authors**: Xin Zhang, Ziqi Dai, Yongqi Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, Jun Yu, Wenjie Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12799)  

**Abstract**: Current multimodal information retrieval studies mainly focus on single-image inputs, which limits real-world applications involving multiple images and text-image interleaved content. In this work, we introduce the text-image interleaved retrieval (TIIR) task, where the query and document are interleaved text-image sequences, and the model is required to understand the semantics from the interleaved context for effective retrieval. We construct a TIIR benchmark based on naturally interleaved wikiHow tutorials, where a specific pipeline is designed to generate interleaved queries. To explore the task, we adapt several off-the-shelf retrievers and build a dense baseline by interleaved multimodal large language model (MLLM). We then propose a novel Matryoshka Multimodal Embedder (MME), which compresses the number of visual tokens at different granularity, to address the challenge of excessive visual tokens in MLLM-based TIIR models. Experiments demonstrate that simple adaption of existing models does not consistently yield effective results. Our MME achieves significant improvements over the baseline by substantially fewer visual tokens. We provide extensive analysis and will release the dataset and code to facilitate future research. 

**Abstract (ZH)**: 当前的多模态信息检索研究主要集中在单张图像的输入上，这限制了涉及多张图像和图文交错内容的实际应用。在本文中，我们介绍了图文交错检索（TIIR，Text-Image Interleaved Retrieval）任务，其中查询和文档是交错的文本图像序列，模型需要从交错的上下文中理解语义以进行有效的检索。我们基于自然交错的wikiHow教程构建了一个TIIR基准数据集，并设计了一个管线生成交错查询。为了探索该任务，我们适应了几种现成的信息检索器，并通过交错多模态大语言模型（MLLM）构建了一个稠密基线。然后，我们提出了一种新颖的Matryoshka多模态嵌入器（MME），该嵌入器在不同粒度上压缩视觉词的数量，以解决基于MLLM的TIIR模型中视觉词过多的问题。实验表明，简单的模型适应并不总能获得有效结果。我们的MME通过显著减少视觉词的数量，在基线之上取得了显著改进。我们进行了详细分析，并将发布数据集和代码，以促进未来的研究。 

---
