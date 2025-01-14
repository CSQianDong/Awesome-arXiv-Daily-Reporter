# Multimodal semantic retrieval for product search 

**Title (ZH)**: 多模态语义检索在产品搜索中的应用 

**Authors**: Dong Liu, Esther Lopez Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2501.07365)  

**Abstract**: Semantic retrieval (also known as dense retrieval) based on textual data has been extensively studied for both web search and product search application fields, where the relevance of a query and a potential target document is computed by their dense vector representation comparison. Product image is crucial for e-commence search interactions and is a key factor for customers at product explorations. But its impact for semantic retrieval has not been well studied yet. In this research, we build a multimodal representation for product items in e-commerece search in contrast to pure-text representation of products, and investigate the impact of such representations. The models are developed and evaluated on e-commerce datasets. We demonstrate that a multimodal representation scheme for a product can show improvement either on purchase recall or relevance accuracy in semantic retrieval. Additionally, we provide numerical analysis for exclusive matches retrieved by a multimodal semantic retrieval model versus a text-only semantic retrieval model, to demonstrate the validation of multimodal solutions. 

**Abstract (ZH)**: 基于文本数据的语义检索（也称为密集检索）在Web搜索和电子商务搜索等领域得到了广泛研究，其中查询和潜在目标文档的相关性通过它们的密集向量表示进行比较。产品图像对于电子商务搜索交互至关重要，是客户进行产品探索的重要因素。然而，产品图像对语义检索的影响尚未得到充分研究。在本研究中，我们构建了一种多模态表示方法，用于电子商务搜索中的产品项目，相比之下，传统的产品表示方法仅基于纯文本。我们研究了这种表示方法的影响，并在电子商务数据集上开发和评估了相应的模型。研究表明，产品中的多模态表示方案可以在语义检索中提高购买召回率或相关性准确性。此外，我们提供了唯一匹配项的数值分析，比较了多模态语义检索模型与仅基于文本的语义检索模型的表现，以验证多模态解决方案的有效性。 

---
# Dataset-Agnostic Recommender Systems 

**Title (ZH)**: 数据集无关推荐系统 

**Authors**: Tri Kurniawan Wijaya, Edoardo D'Amico, Xinyang Shao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07294)  

**Abstract**: [This is a position paper and does not contain any empirical or theoretical results] Recommender systems have become a cornerstone of personalized user experiences, yet their development typically involves significant manual intervention, including dataset-specific feature engineering, hyperparameter tuning, and configuration. To this end, we introduce a novel paradigm: Dataset-Agnostic Recommender Systems (DAReS) that aims to enable a single codebase to autonomously adapt to various datasets without the need for fine-tuning, for a given recommender system task. Central to this approach is the Dataset Description Language (DsDL), a structured format that provides metadata about the dataset's features and labels, and allow the system to understand dataset's characteristics, allowing it to autonomously manage processes like feature selection, missing values imputation, noise removal, and hyperparameter optimization. By reducing the need for domain-specific expertise and manual adjustments, DAReS offers a more efficient and scalable solution for building recommender systems across diverse application domains. It addresses critical challenges in the field, such as reusability, reproducibility, and accessibility for non-expert users or entry-level researchers. 

**Abstract (ZH)**: 这是一篇立场声明，不包含任何实证或理论结果。推荐系统已成为个性化用户体验的基础，但其开发通常需要大量的人工干预，包括特定数据集的特征工程、超参数调整和配置。为了解决这个问题，我们引入了一种新的范式：数据集无关的推荐系统（DAReS），旨在使单一代码库能够在无需微调的情况下自主适应各种数据集。该方法的核心是数据集描述语言（DsDL），这是一种结构化的格式，提供有关数据集特征和标签的元数据，使系统能够理解数据集的特性，从而自主管理特征选择、缺失值填充、噪声去除和超参数优化等过程。通过减少对特定领域专业知识和手动调整的需求，DAReS 提供了一种更高效、更具扩展性的解决方案，适用于不同应用领域的推荐系统构建。它解决了该领域中的关键挑战，如可重用性、可重现性以及非专家用户或初级研究人员的访问性。 

---
# Future-Conditioned Recommendations with Multi-Objective Controllable Decision Transformer 

**Title (ZH)**: 基于未来条件的多目标可控决策变换器推荐方法 

**Authors**: Chongming Gao, Kexin Huang, Ziang Fei, Jiaju Chen, Jiawei Chen, Jianshan Sun, Shuchang Liu, Qingpeng Cai, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07212)  

**Abstract**: Securing long-term success is the ultimate aim of recommender systems, demanding strategies capable of foreseeing and shaping the impact of decisions on future user satisfaction. Current recommendation strategies grapple with two significant hurdles. Firstly, the future impacts of recommendation decisions remain obscured, rendering it impractical to evaluate them through direct optimization of immediate metrics. Secondly, conflicts often emerge between multiple objectives, like enhancing accuracy versus exploring diverse recommendations. Existing strategies, trapped in a "training, evaluation, and retraining" loop, grow more labor-intensive as objectives evolve. To address these challenges, we introduce a future-conditioned strategy for multi-objective controllable recommendations, allowing for the direct specification of future objectives and empowering the model to generate item sequences that align with these goals autoregressively. We present the Multi-Objective Controllable Decision Transformer (MocDT), an offline Reinforcement Learning (RL) model capable of autonomously learning the mapping from multiple objectives to item sequences, leveraging extensive offline data. Consequently, it can produce recommendations tailored to any specified objectives during the inference stage. Our empirical findings emphasize the controllable recommendation strategy's ability to produce item sequences according to different objectives while maintaining performance that is competitive with current recommendation strategies across various objectives. 

**Abstract (ZH)**: 长期成功是推荐系统最终的目标，这要求具备预见和塑造决策对未来用户满意度影响的策略。当前的推荐策略面临着两大显著挑战。首先，推荐决策未来的影响难以预测，使通过直接优化即时指标对其进行评估变得不可行。其次，多种目标之间常常存在冲突，例如提高准确性与探索多样化推荐之间的矛盾。现有的策略在“训练、评估和重新训练”的循环中变得日益繁琐，随着目标的变化而增加更多的劳动投入。为应对这些挑战，我们提出了一种未来条件下的多目标可控推荐策略，允许直接指定未来的优化目标，并使模型能够自回归地生成符合这些目标的项目序列。我们提出了多目标可控决策变换器（MocDT），这是一种基于离线强化学习（Reinforcement Learning, RL）的模型，能够自主学习从多个目标到项目序列的映射，并利用大量的离线数据进行学习。因此，在推理阶段，它可以生成针对任何指定目标的推荐。我们的实证结果显示，这种可控推荐策略能够在不同目标下生成项目序列，同时在各种目标上的性能与当前推荐策略相当。 

---
# Intent-Interest Disentanglement and Item-Aware Intent Contrastive Learning for Sequential Recommendation 

**Title (ZH)**: 意图与兴趣解耦及项目感知意图对比学习在序列推荐中的应用 

**Authors**: Yijin Choi, Chiehyeon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2501.07096)  

**Abstract**: Recommender systems aim to provide personalized item recommendations by capturing user behaviors derived from their interaction history. Considering that user interactions naturally occur sequentially based on users' intents in mind, user behaviors can be interpreted as user intents. Therefore, intent-based sequential recommendations are actively studied recently to model user intents from historical interactions for a more precise user understanding beyond traditional studies that often overlook the underlying semantics behind user interactions. However, existing studies face three challenges: 1) the limited understanding of user behaviors by focusing solely on intents, 2) the lack of robustness in categorizing intents due to arbitrary fixed numbers of intent categories, and 3) the neglect of interacted items in modeling of user intents. To address these challenges, we propose Intent-Interest Disentanglement and Item-Aware Intent Contrastive Learning for Sequential Recommendation (IDCLRec). IDCLRec disentangles user behaviors into intents which are dynamic motivations and interests which are stable tastes of users for a comprehensive understanding of user behaviors. A causal cross-attention mechanism is used to identify consistent interests across interactions, while residual behaviors are modeled as intents by modeling their temporal dynamics through a similarity adjustment loss. In addition, without predefining the number of intent categories, an importance-weighted attention mechanism captures user-specific categorical intent considering the importance of intent for each interaction. Furthermore, we introduce item-aware contrastive learning which aligns intents that occurred the same interaction and aligns intent with item combinations occurred by the corresponding intent. Extensive experiments conducted on real-world datasets demonstrate the effectiveness of IDCLRec. 

**Abstract (ZH)**: 推荐系统旨在通过捕捉用户历史交互行为来提供个性化的物品推荐。考虑到用户交互自然按照用户的意图顺序发生，用户行为可以被解释为用户的意图。因此，基于意图的序列推荐最近被积极研究，以从历史交互中建模用户的意图，从而超越传统研究对用户意图背后语义的忽视。然而，现有的研究面临着三大挑战：1) 仅仅聚焦于意图会限制对用户行为的理解；2) 由于固定数量的意图类别导致意图分类的不稳定性；3) 在建模用户意图时忽略了已交互的物品。为解决这些挑战，我们提出了一种意图兴趣解耦和物品意识意图对比学习的序列推荐方法 (IDCLRec)。IDCLRec 将用户行为解耦为动态的意图和稳定的兴趣，以实现对用户行为的全面理解。因果交叉注意力机制用于识别一致的兴趣，而残余行为则通过相似性调整损失来建模其时间动态，从而作为意图。此外，IDCLRec 不预先定义意图类别的数量，而是通过重要性加权注意力机制捕捉特定于用户意图的分类，考虑每个交互中意图的重要性。进一步引入了物品意识对比学习，该方法将相同交互的发生意图对齐，并将意图与发生相应意图的物品组合对齐。在现实世界数据集上的广泛实验表明了 IDCLRec 的有效性。 

---
# Research on the Online Update Method for Retrieval-Augmented Generation (RAG) Model with Incremental Learning 

**Title (ZH)**: 基于增量学习的检索增强生成（RAG）模型在线更新方法研究 

**Authors**: Yuxin Fan, Yuxiang Wang, Lipeng Liu, Xirui Tang, Na Sun, Zidong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07063)  

**Abstract**: In the contemporary context of rapid advancements in information technology and the exponential growth of data volume, language models are confronted with significant challenges in effectively navigating the dynamic and ever-evolving information landscape to update and adapt to novel knowledge in real time. In this work, an online update method is proposed, which is based on the existing Retrieval Enhanced Generation (RAG) model with multiple innovation mechanisms. Firstly, the dynamic memory is used to capture the emerging data samples, and then gradually integrate them into the core model through a tunable knowledge distillation strategy. At the same time, hierarchical indexing and multi-layer gating mechanism are introduced into the retrieval module to ensure that the retrieved content is more targeted and accurate. Finally, a multi-stage network structure is established for different types of inputs in the generation stage, and cross-attention matching and screening are carried out on the intermediate representations of each stage to ensure the effective integration and iterative update of new and old knowledge. Experimental results show that the proposed method is better than the existing mainstream comparison models in terms of knowledge retention and inference accuracy. 

**Abstract (ZH)**: 在信息技术飞速发展和数据量指数级增长的当代背景下，语言模型面临着在动态且不断演变的信息环境中有效更新和适应新知识的显著挑战，尤其是在实时更新方面。本文提出了一种在线更新方法，该方法基于现有的检索增强生成（RAG）模型，并结合了多种创新机制。首先，动态内存被用于捕捉新兴数据样本，并通过可调的学习策略逐渐将这些样本整合到核心模型中。与此同时，引入了层次索引和多层门控机制，以确保检索内容更具针对性和准确性。最后，在生成阶段建立了多阶段网络结构，并在每阶段的中间表示上进行跨注意力匹配和筛选，以确保新旧知识的有效集成和迭代更新。实验结果表明，提出的更新方法在知识保留和推理准确度方面优于现有的主流比较模型。 

---
# Graph Contrastive Learning on Multi-label Classification for Recommendations 

**Title (ZH)**: 多标签分类中的图对比学习推荐方法 

**Authors**: Jiayang Wu, Wensheng Gan, Huashen Lu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06985)  

**Abstract**: In business analysis, providing effective recommendations is essential for enhancing company profits. The utilization of graph-based structures, such as bipartite graphs, has gained popularity for their ability to analyze complex data relationships. Link prediction is crucial for recommending specific items to users. Traditional methods in this area often involve identifying patterns in the graph structure or using representational techniques like graph neural networks (GNNs). However, these approaches encounter difficulties as the volume of data increases. To address these challenges, we propose a model called Graph Contrastive Learning for Multi-label Classification (MCGCL). MCGCL leverages contrastive learning to enhance recommendation effectiveness. The model incorporates two training stages: a main task and a subtask. The main task is holistic user-item graph learning to capture user-item relationships. The homogeneous user-user (item-item) subgraph is constructed to capture user-user and item-item relationships in the subtask. We assessed the performance using real-world datasets from Amazon Reviews in multi-label classification tasks. Comparative experiments with state-of-the-art methods confirm the effectiveness of MCGCL, highlighting its potential for improving recommendation systems. 

**Abstract (ZH)**: 在商业分析中，提供有效的推荐对于提升公司利润至关重要。借助基于图的结构，如二部图，来分析复杂的数据关系已经变得popular。链接预测对于为用户推荐特定项目至关重要。传统方法通常涉及识别图结构中的模式或使用图神经网络（GNNs）等表示技术。然而，随着数据量的增加，这些方法会遇到困难。为了解决这些问题，我们提出了一种名为多标签分类的图对比学习模型（MCGCL）。MCGCL利用对比学习来增强推荐的有效性。该模型包含两个训练阶段：主任务和子任务。主任务是概览用户-项目图学习，以捕捉用户-项目关系。在子任务中构建同构用户-用户（项目-项目）子图以捕捉用户-用户和项目-项目关系。我们使用来自Amazon评论的真实数据集在多标签分类任务中评估了性能。与当前最先进的方法进行的比较实验证实了MCGCL的有效性，突显了其在改进推荐系统方面的潜力。 

---
# Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation 

**Title (ZH)**: 基于重复偏差的超越准确性的指标优化以提高下一个购物篮推荐性能 

**Authors**: Yuanna Liu, Ming Li, Mohammad Aliannejadi, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2501.06362)  

**Abstract**: In next basket recommendation (NBR) a set of items is recommended to users based on their historical basket sequences. In many domains, the recommended baskets consist of both repeat items and explore items. Some state-of-the-art NBR methods are heavily biased to recommend repeat items so as to maximize utility. The evaluation and optimization of beyond-accuracy objectives for NBR, such as item fairness and diversity, has attracted increasing attention. How can such beyond-accuracy objectives be pursued in the presence of heavy repeat bias? We find that only optimizing diversity or item fairness without considering repeat bias may cause NBR algorithms to recommend more repeat items. To solve this problem, we propose a model-agnostic repeat-bias-aware optimization algorithm to post-process the recommended results obtained from NBR methods with the objective of mitigating repeat bias when optimizing diversity or item fairness. We consider multiple variations of our optimization algorithm to cater to multiple NBR methods. Experiments on three real-world grocery shopping datasets show that the proposed algorithms can effectively improve diversity and item fairness, and mitigate repeat bias at acceptable Recall loss. 

**Abstract (ZH)**: 在篮子推荐（Basket Recommendation, BRS）中，基于用户的 histórico 购物序列推荐一组物品。在许多领域中，推荐的篮子既包含重复项也包含探索项。一些先进的 BRS 方法严重偏向于推荐重复项，以最大化效用。对于 BRS 超越准确性的评价和优化目标，如物品公平性和多样性，的关注正在不断增加。如何在存在严重重复偏见的情况下实现这些超越准确性的目标？我们发现，仅优化多样性或物品公平性而不考虑重复偏见可能导致 BRS 算法推荐更多的重复项。为了解决这一问题，我们提出了一种模型无关的重复偏见感知优化算法，在优化多样性或物品公平性的同时，对 BRS 方法获得的推荐结果进行后处理，以减轻重复偏见。我们考虑了多种优化算法的变体，以适应多种不同的 BRS 方法。实验结果表明，提出的算法能够有效提高多样性和物品公平性，并在可接受的召回率损失下减轻重复偏见。 

---
# ListConRanker: A Contrastive Text Reranker with Listwise Encoding 

**Title (ZH)**: ListConRanker：一种基于列表编码的对比文本重排序器 

**Authors**: Junlong Liu, Yue Ma, Ruihui Zhao, Junhao Zheng, Qianli Ma, Yangyang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07111)  

**Abstract**: Reranker models aim to re-rank the passages based on the semantics similarity between the given query and passages, which have recently received more attention due to the wide application of the Retrieval-Augmented Generation. Most previous methods apply pointwise encoding, meaning that it can only encode the context of the query for each passage input into the model. However, for the reranker model, given a query, the comparison results between passages are even more important, which is called listwise encoding. Besides, previous models are trained using the cross-entropy loss function, which leads to issues of unsmooth gradient changes during training and low training efficiency. To address these issues, we propose a novel Listwise-encoded Contrastive text reRanker (ListConRanker). It can help the passage to be compared with other passages during the encoding process, and enhance the contrastive information between positive examples and between positive and negative examples. At the same time, we use the circle loss to train the model to increase the flexibility of gradients and solve the problem of training efficiency. Experimental results show that ListConRanker achieves state-of-the-art performance on the reranking benchmark of Chinese Massive Text Embedding Benchmark, including the cMedQA1.0, cMedQA2.0, MMarcoReranking, and T2Reranking datasets. 

**Abstract (ZH)**: 重排序模型旨在根据给定查询与段落之间的语义相似性重新排列段落，由于检索增强生成的广泛应用，这类模型近年来受到了更多关注。之前的大多数方法使用点wise编码，这意味着模型只能为每个段落输入查询的上下文进行编码。然而，对于重排序模型，给定一个查询时，段落之间的对比结果更为重要，这被称为列表wise编码。此外，之前的模型使用交叉熵损失函数进行训练，这会导致训练过程中梯度变化不平滑，训练效率低。为了解决这些问题，我们提出了一种新的列表wise编码对比文本重排序器(ListConRanker)。该模型在编码过程中帮助段落与其他段落进行对比，并增强正例之间的对比信息以及正例与负例之间的对比信息。同时，我们使用圆损失来训练模型，增加梯度的灵活性，并解决训练效率问题。实验结果表明，ListConRanker在中文大规模文本嵌入基准中的重排序基准上实现了最先进的性能，包括cMedQA1.0、cMedQA2.0、MMarcoReranking和T2Reranking数据集。 

---
# Dynamic Multimodal Fusion via Meta-Learning Towards Micro-Video Recommendation 

**Title (ZH)**: 面向微视频推荐的元学习驱动的动态多模态融合 

**Authors**: Han Liu, Yinwei Wei, Fan Liu, Wenjie Wang, Liqiang Nie, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2501.07110)  

**Abstract**: Multimodal information (e.g., visual, acoustic, and textual) has been widely used to enhance representation learning for micro-video recommendation. For integrating multimodal information into a joint representation of micro-video, multimodal fusion plays a vital role in the existing micro-video recommendation approaches. However, the static multimodal fusion used in previous studies is insufficient to model the various relationships among multimodal information of different micro-videos. In this paper, we develop a novel meta-learning-based multimodal fusion framework called Meta Multimodal Fusion (MetaMMF), which dynamically assigns parameters to the multimodal fusion function for each micro-video during its representation learning. Specifically, MetaMMF regards the multimodal fusion of each micro-video as an independent task. Based on the meta information extracted from the multimodal features of the input task, MetaMMF parameterizes a neural network as the item-specific fusion function via a meta learner. We perform extensive experiments on three benchmark datasets, demonstrating the significant improvements over several state-of-the-art multimodal recommendation models, like MMGCN, LATTICE, and InvRL. Furthermore, we lighten our model by adopting canonical polyadic decomposition to improve the training efficiency, and validate its effectiveness through experimental results. Codes are available at this https URL. 

**Abstract (ZH)**: 多模态信息（例如视觉、声学和文本）广泛用于增强微视频推荐中的表示学习。在现有的微视频推荐方法中，多模态融合对于将多模态信息整合到微视频的联合表示中起着至关重要的作用。然而，先前研究中使用的静态多模态融合不足以建模不同微视频之间多模态信息的各种关系。在本文中，我们提出了一种新颖的基于元学习的多模态融合框架，称为Meta多模态融合（MetaMMF），该框架在每个微视频的表示学习过程中动态为多模态融合函数分配参数。具体而言，MetaMMF 将每个微视频的多模态融合视为独立的任务。基于输入任务的多模态特征提取的元信息，MetaMMF 通过元学习器将一个神经网络参数化为特定于项目的融合函数。我们对三个基准数据集执行了大量实验，结果表明，MetaMMF 在多个最先进的多模态推荐模型（如MMGCN、LATTICE 和 InvRL）上显著提升了性能。此外，我们通过采用典范多项式分解来简化模型，以提高训练效率，并通过实验结果验证其有效性。代码可从以下链接获取：[此处替换为链接] 

---
# A Proposed Large Language Model-Based Smart Search for Archive System 

**Title (ZH)**: 一种基于大型语言模型的智能存档系统搜索方法 

**Authors**: Ha Dung Nguyen, Thi-Hoang Anh Nguyen, Thanh Binh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07024)  

**Abstract**: This study presents a novel framework for smart search in digital archival systems, leveraging the capabilities of Large Language Models (LLMs) to enhance information retrieval. By employing a Retrieval-Augmented Generation (RAG) approach, the framework enables the processing of natural language queries and transforming non-textual data into meaningful textual representations. The system integrates advanced metadata generation techniques, a hybrid retrieval mechanism, a router query engine, and robust response synthesis, the results proved search precision and relevance. We present the architecture and implementation of the system and evaluate its performance in four experiments concerning LLM efficiency, hybrid retrieval optimizations, multilingual query handling, and the impacts of individual components. Obtained results show significant improvements over conventional approaches and have demonstrated the potential of AI-powered systems to transform modern archival practices. 

**Abstract (ZH)**: 本研究提出了一种用于数字档案系统智能搜索的新框架，充分利用大型语言模型（LLMs）的能力来增强信息检索。通过采用检索增强生成（RAG）方法，该框架能够处理自然语言查询，并将非文本数据转换为有意义的文本表示。该系统集成了高级元数据生成技术、混合检索机制、路由器查询引擎以及强大的响应合成方法，结果显示了搜索精度和相关性的提升。我们展示了该系统的架构和实现，并在四个实验中评估了其性能，涉及大型语言模型效率、混合检索优化、多语言查询处理以及各组件的影响。所获得的结果表明，该方法显著优于传统方法，展示了基于人工智能的系统在转变现代档案实践方面的巨大潜力。 

---
# Patent Novelty Assessment Accelerating Innovation and Patent Prosecution 

**Title (ZH)**: 专利新颖性评估加速创新与专利审查 

**Authors**: Kapil Kashyap, Sean Fargose, Gandhar Dhonde, Aditya Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2501.06956)  

**Abstract**: In the rapidly evolving landscape of technological innovation, safeguarding intellectual property rights through patents is crucial for fostering progress and stimulating research and development investments. This report introduces a ground-breaking Patent Novelty Assessment and Claim Generation System, meticulously crafted to dissect the inventive aspects of intellectual property and simplify access to extensive patent claim data. Addressing a crucial gap in academic institutions, our system provides college students and researchers with an intuitive platform to navigate and grasp the intricacies of patent claims, particularly tailored for the nuances of Chinese patents. Unlike conventional analysis systems, our initiative harnesses a proprietary Chinese API to ensure unparalleled precision and relevance. The primary challenge lies in the complexity of accessing and comprehending diverse patent claims, inhibiting effective innovation upon existing ideas. Our solution aims to overcome these barriers by offering a bespoke approach that seamlessly retrieves comprehensive claim information, finely tuned to the specifics of the Chinese patent landscape. By equipping users with efficient access to comprehensive patent claim information, our transformative platform seeks to ignite informed exploration and innovation in the ever-evolving domain of intellectual property. Its envisioned impact transcends individual colleges, nurturing an environment conducive to research and development while deepening the understanding of patented concepts within the academic community. 

**Abstract (ZH)**: 在 technological innovation 领域快速发展的背景下，通过专利保护知识产权对于促进进步和刺激研究与开发投资至关重要。本报告介绍了一项开创性的专利新颖性评估和主张生成系统，该系统精心设计，旨在剖析知识产权的创新方面，并简化对广泛专利主张数据的访问。我们的系统填补了学术机构中的一个关键缺口，为大学生和研究人员提供了一个直观的平台，使他们能够导航并理解复杂的专利主张，特别是针对中国专利的细微之处进行了定制。与传统的分析系统不同，我们的倡议利用了一款专有的中文API，确保无与伦比的准确性和相关性。主要挑战在于访问和理解多样化的专利主张的复杂性，这抑制了对现有概念的有效创新。我们的解决方案通过提供一种量身定制的方法来克服这些障碍，该方法能够无缝获取详细的信息，并针对中国的专利格局进行精细调整。通过为用户提供高效访问全面专利主张信息的能力，我们的革新平台旨在激发对知识产权领域不断演变的领域的知情探索和创新。其预期影响超越了单一学院，创建了一种有利于研究与开发的环境，同时加深了学术界对专利概念的理解。 

---
# Causal Claims in Economics 

**Title (ZH)**: 经济中的因果断言 

**Authors**: Prashant Garg, Thiemo Fetzer  

**Link**: [PDF](https://arxiv.org/pdf/2501.06873)  

**Abstract**: We analyze over 44,000 NBER and CEPR working papers from 1980 to 2023 using a custom language model to construct knowledge graphs that map economic concepts and their relationships. We distinguish between general claims and those documented via causal inference methods (e.g., DiD, IV, RDD, RCTs). We document a substantial rise in the share of causal claims-from roughly 4% in 1990 to nearly 28% in 2020-reflecting the growing influence of the "credibility revolution." We find that causal narrative complexity (e.g., the depth of causal chains) strongly predicts both publication in top-5 journals and higher citation counts, whereas non-causal complexity tends to be uncorrelated or negatively associated with these outcomes. Novelty is also pivotal for top-5 publication, but only when grounded in credible causal methods: introducing genuinely new causal edges or paths markedly increases both the likelihood of acceptance at leading outlets and long-run citations, while non-causal novelty exhibits weak or even negative effects. Papers engaging with central, widely recognized concepts tend to attract more citations, highlighting a divergence between factors driving publication success and long-term academic impact. Finally, bridging underexplored concept pairs is rewarded primarily when grounded in causal methods, yet such gap filling exhibits no consistent link with future citations. Overall, our findings suggest that methodological rigor and causal innovation are key drivers of academic recognition, but sustained impact may require balancing novel contributions with conceptual integration into established economic discourse. 

**Abstract (ZH)**: 我们使用自定义语言模型分析了从1980年到2023年的超过44,000篇NBER和CEPR的工作论文，构建了知识图谱以映射经济概念及其关系。我们区分了普遍断言与通过因果推理方法（如差分对照法、工具变量法、断点回归法、随机对照试验等）验证的断言。我们记录了因果断言所占份额的显著上升——从1990年的约4%上升到2020年的近28%，这反映了“可信度革命”的日益影响。我们发现，因果叙事复杂性（如因果链的深度）强烈预测了顶级期刊的发表几率和较高的引用次数，而非因果复杂性则与此结果无显著相关性或呈负相关。新颖性对顶级期刊的发表也至关重要，但前提是基于可信的因果方法：引入真正的新型因果边或路径大大增加了被顶级出版物接受和长期引用的可能性，而非因果新颖性则效应较弱甚至为负效应。探讨中心、广为人知的概念的论文更容易获得引用，这突显了推动出版成功和长期学术影响的因素之间的差异。最后，基于因果方法填补未充分研究的概念对是主要的奖励，但这种填补与未来引用次数之间并无一致联系。总体而言，我们的研究结果表明，方法论严谨性和因果创新是学术认可的关键驱动因素，但持续影响力可能需要在新颖贡献与纳入现有经济讨论的概念整合之间取得平衡。 

---
# Unveiling Temporal Trends in 19th Century Literature: An Information Retrieval Approach 

**Title (ZH)**: 揭示19世纪文学中的时间趋势：一种信息检索方法 

**Authors**: Suchana Datta, Dwaipayan Roy, Derek Greene, Gerardine Meaney  

**Link**: [PDF](https://arxiv.org/pdf/2501.06833)  

**Abstract**: In English literature, the 19th century witnessed a significant transition in styles, themes, and genres. Consequently, the novels from this period display remarkable diversity. This paper explores these variations by examining the evolution of term usage in 19th century English novels through the lens of information retrieval. By applying a query expansion-based approach to a decade-segmented collection of fiction from the British Library, we examine how related terms vary over time. Our analysis employs multiple standard metrics including Kendall's tau, Jaccard similarity, and Jensen-Shannon divergence to assess overlaps and shifts in expanded query term sets. Our results indicate a significant degree of divergence in the related terms across decades as selected by the query expansion technique, suggesting substantial linguistic and conceptual changes throughout the 19th century novels. 

**Abstract (ZH)**: 在英语文学中，19世纪见证了文体、主题和体裁的重要转型。因此，该时期的 novels 展现了显著的多样性。本文通过信息检索的视角，探讨了19世纪英语小说中术语使用的变化。通过应用于大英图书馆分十年段的小说集合的基于查询扩展的方法，我们分析了相关术语随时间的变化。我们的分析使用了多种标准度量指标，包括肯德尔系数（Kendall's tau）、杰卡德相似度（Jaccard similarity）和 Jensen-Shannon 散度（Jensen-Shannon divergence），评估扩展查询术语集之间的重叠和变化。我们的研究表明，查询扩展技术选择的相关术语在不同十年间存在显著差异，这表明19世纪小说中的语言和概念发生了重大变化。 

---
# Large Language Models, Knowledge Graphs and Search Engines: A Crossroads for Answering Users' Questions 

**Title (ZH)**: 大型语言模型、知识图谱和搜索引擎：解答用户问题的交汇点 

**Authors**: Aidan Hogan, Xin Luna Dong, Denny Vrandečić, Gerhard Weikum  

**Link**: [PDF](https://arxiv.org/pdf/2501.06699)  

**Abstract**: Much has been discussed about how Large Language Models, Knowledge Graphs and Search Engines can be combined in a synergistic manner. A dimension largely absent from current academic discourse is the user perspective. In particular, there remain many open questions regarding how best to address the diverse information needs of users, incorporating varying facets and levels of difficulty. This paper introduces a taxonomy of user information needs, which guides us to study the pros, cons and possible synergies of Large Language Models, Knowledge Graphs and Search Engines. From this study, we derive a roadmap for future research. 

**Abstract (ZH)**: 关于大型语言模型、知识图谱和搜索引擎如何协同工作的讨论很多。目前学术讨论中一个被忽视的方面是从用户的角度出发。特别是在如何最好地满足用户多样的信息需求方面，仍然存在许多开放性问题，这些需求涉及不同的方面和难度等级。本文提出了用户信息需求的分类体系，这有助于我们研究大型语言模型、知识图谱和搜索引擎的优势、劣势及其可能的协同效应。通过这一研究，我们为未来的研究指明了道路。 

---
# Recommending the right academic programs: An interest mining approach using BERTopic 

**Title (ZH)**: 推荐合适的学术课程：一种基于BERTopic的兴趣挖掘方法 

**Authors**: Alessandro Hill, Kalen Goo, Puneet Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2501.06581)  

**Abstract**: Prospective students face the challenging task of selecting a university program that will shape their academic and professional careers. For decision-makers and support services, it is often time-consuming and extremely difficult to match personal interests with suitable programs due to the vast and complex catalogue information available. This paper presents the first information system that provides students with efficient recommendations based on both program content and personal preferences. BERTopic, a powerful topic modeling algorithm, is used that leverages text embedding techniques to generate topic representations. It enables us to mine interest topics from all course descriptions, representing the full body of knowledge taught at the institution. Underpinned by the student's individual choice of topics, a shortlist of the most relevant programs is computed through statistical backtracking in the knowledge map, a novel characterization of the program-course relationship. This approach can be applied to a wide range of educational settings, including professional and vocational training. A case study at a post-secondary school with 80 programs and over 5,000 courses shows that the system provides immediate and effective decision support. The presented interest topics are meaningful, leading to positive effects such as serendipity, personalization, and fairness, as revealed by a qualitative study involving 65 students. Over 98% of users indicated that the recommendations aligned with their interests, and about 94% stated they would use the tool in the future. Quantitative analysis shows the system can be configured to ensure fairness, achieving 98% program coverage while maintaining a personalization score of 0.77. These findings suggest that this real-time, user-centered, data-driven system could improve the program selection process. 

**Abstract (ZH)**: Prospective学生面临着选择塑造其学术和职业道路的大学项目的挑战性任务。对于决策者和支援服务而言，由于可供选择的信息量庞大且复杂，将个人兴趣与合适的项目匹配往往是一个耗时且极具挑战的任务。本文介绍了首个能够基于课程内容和个人偏好为学生提供高效推荐的信息系统。本系统使用一种强大的主题建模算法BERTopic，结合文本嵌入技术生成主题表示，从而从所有课程描述中挖掘兴趣主题，反映机构传授的全部知识范围。基于学生的个人兴趣主题选择，系统通过知识图谱中的统计追溯计算出相关性最高的项目列表，这是一种创新的课程-项目关系表征方式。该方法可以应用于各种教育场景，包括职业和职业教育。以一所设有80个项目和超过5,000门课程的高等学院为例，案例研究表明，该系统能够立即提供有效的决策支持。定性研究结果显示，所呈现实兴主题具有意义，带来诸如意外发现、个性化和公正性等积极影响，65名参与研究的学生对此表示肯定。超过98%的用户表示推荐结果与其兴趣相符，并且约94%的用户表示愿意在未来继续使用该工具。定量分析表明，该系统可以配置以确保公正性，能够在覆盖98%项目的同时维持0.77的个性化评分。这些发现表明，这种实时、以用户为中心、数据驱动的系统有可能改进项目选择过程。 

---
# Analyzing the Role of Context in Forecasting with Large Language Models 

**Title (ZH)**: 分析上下文在使用大规模语言模型进行预测中的作用 

**Authors**: Gerrit Mutschlechner, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2501.06496)  

**Abstract**: This study evaluates the forecasting performance of recent language models (LLMs) on binary forecasting questions. We first introduce a novel dataset of over 600 binary forecasting questions, augmented with related news articles and their concise question-related summaries. We then explore the impact of input prompts with varying level of context on forecasting performance. The results indicate that incorporating news articles significantly improves performance, while using few-shot examples leads to a decline in accuracy. We find that larger models consistently outperform smaller models, highlighting the potential of LLMs in enhancing automated forecasting. 

**Abstract (ZH)**: 本研究评估了近期语言模型（LLMs）在二元预测问题上的预测性能。我们首先介绍了一个包含超过600个二元预测问题的新颖数据集，该数据集还增加了相关的新闻文章及其简洁的问题相关摘要。随后，我们探讨了不同水平上下文输入提示对预测性能的影响。结果表明，融入新闻文章显著提高了性能，而使用少量示例则导致准确性下降。我们发现较大的模型始终优于较小的模型，这突显了LLMs在增强自动化预测方面的潜力。 

---
# Gender-Neutral Large Language Models for Medical Applications: Reducing Bias in PubMed Abstracts 

**Title (ZH)**: 面向医学应用的无性别偏向大型语言模型：减少PubMed摘要中的偏差 

**Authors**: Elizabeth Schaefer, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2501.06365)  

**Abstract**: This paper presents a pipeline for mitigating gender bias in large language models (LLMs) used in medical literature by neutralizing gendered occupational pronouns. A dataset of 379,000 PubMed abstracts from 1965-1980 was processed to identify and modify pronouns tied to professions. We developed a BERT-based model, ``Modern Occupational Bias Elimination with Refined Training,'' or ``MOBERT,'' trained on these neutralized abstracts, and compared its performance with ``1965Bert,'' trained on the original dataset. MOBERT achieved a 70\% inclusive replacement rate, while 1965Bert reached only 4\%. A further analysis of MOBERT revealed that pronoun replacement accuracy correlated with the frequency of occupational terms in the training data. We propose expanding the dataset and refining the pipeline to improve performance and ensure more equitable language modeling in medical applications. 

**Abstract (ZH)**: 本文提出了一种流水线方法，用于减轻医学文献中大型语言模型（LLM）中的性别偏见，该方法通过中性化职业代词来缓解性别偏见。我们处理了1965年至1980年间37.9万篇PubMed摘要，以识别并修改与职业相关的代词。我们开发了一个基于BERT的模型，名为“现代职业偏见消除与精细训练”（Modern Occupational Bias Elimination with Refined Training，简称MOBERT），该模型是在这些中性化摘要上训练的，并将其性能与基于原始数据集训练的“1965Bert”进行了比较。MOBERT实现了70%的包容性替换率，而1965Bert仅为4%。进一步分析MOBERT显示，代词替换的准确性与训练数据中职业术语的频率相关。我们建议扩大数据集并优化流水线，以提高性能并确保医学应用中更加公平的语言建模。 

---
# Environmental large language model Evaluation (ELLE) dataset: A Benchmark for Evaluating Generative AI applications in Eco-environment Domain 

**Title (ZH)**: 环境大型语言模型评估（ELLE）数据集：评估生成式AI在生态环境领域应用的基准 

**Authors**: Jing Guo, Nan Li, Ming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06277)  

**Abstract**: Generative AI holds significant potential for ecological and environmental applications such as monitoring, data analysis, education, and policy support. However, its effectiveness is limited by the lack of a unified evaluation framework. To address this, we present the Environmental Large Language model Evaluation (ELLE) question answer (QA) dataset, the first benchmark designed to assess large language models and their applications in ecological and environmental sciences. The ELLE dataset includes 1,130 question answer pairs across 16 environmental topics, categorized by domain, difficulty, and type. This comprehensive dataset standardizes performance assessments in these fields, enabling consistent and objective comparisons of generative AI performance. By providing a dedicated evaluation tool, ELLE dataset promotes the development and application of generative AI technologies for sustainable environmental outcomes. The dataset and code are available at this https URL and this https URL. 

**Abstract (ZH)**: 生成式人工智能在生态和环境应用方面具有显著潜力，可用于监测、数据分析、教育和政策支持等领域。然而，其有效性受限于缺乏统一的评估框架。为解决这一问题，我们提出了环境大型语言模型评估（ELLE）问答（QA）数据集，这是第一个旨在评估大型语言模型及其在生态和环境科学领域应用的基准。ELLE数据集包括涵盖16个环境主题的1,130个问答对，按照领域、难度和类型进行分类。该综合数据集规范了这些领域的性能评估，使生成式人工智能的性能评估能够进行一致和客观的比较。通过提供一个专用的评估工具，ELLE数据集促进了生成式人工智能技术在可持续环境结果方面的开发和应用。该数据集和代码可在以下网址获取：[该网址] 和 [该网址]。 

---
