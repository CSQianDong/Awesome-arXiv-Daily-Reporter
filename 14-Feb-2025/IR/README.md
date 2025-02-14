# FARM: Frequency-Aware Model for Cross-Domain Live-Streaming Recommendation 

**Title (ZH)**: FARM：频率感知的跨域直播推荐模型 

**Authors**: Xiaodong Li, Ruochen Yang, Shuang Wen, Shen Wang, Yueyang Liu, Guoquan Wang, Weisong Hu, Qiang Luo, Jiawei Sheng, Tingwen Liu, Jiangxia Cao, Shuang Yang, Zhaojie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09375)  

**Abstract**: Live-streaming services have attracted widespread popularity due to their real-time interactivity and entertainment value. Users can engage with live-streaming authors by participating in live chats, posting likes, or sending virtual gifts to convey their preferences and support. However, the live-streaming services faces serious data-sparsity problem, which can be attributed to the following two points: (1) User's valuable behaviors are usually sparse, e.g., like, comment and gift, which are easily overlooked by the model, making it difficult to describe user's personalized preference. (2) The main exposure content on our platform is short-video, which is 9 times higher than the exposed live-streaming, leading to the inability of live-streaming content to fully model user preference. To this end, we propose a Frequency-Aware Model for Cross-Domain Live-Streaming Recommendation, termed as FARM. Specifically, we first present the intra-domain frequency aware module to enable our model to perceive user's sparse yet valuable behaviors, i.e., high-frequency information, supported by the Discrete Fourier Transform (DFT). To transfer user preference across the short-video and live-streaming domains, we propose a novel preference align before fuse strategy, which consists of two parts: the cross-domain preference align module to align user preference in both domains with contrastive learning, and the cross-domain preference fuse module to further fuse user preference in both domains using a serious of tailor-designed attention mechanisms. Extensive offline experiments and online A/B testing on Kuaishou live-streaming services demonstrate the effectiveness and superiority of FARM. Our FARM has been deployed in online live-streaming services and currently serves hundreds of millions of users on Kuaishou. 

**Abstract (ZH)**: 直播服务由于其实时互动性和娱乐价值，已经赢得了广泛的 popularity。用户可以通过参与直播聊天、点赞或发送虚拟礼物等方式与直播作者互动，传达自己的偏好和支持。然而，直播服务面临严重的数据稀疏性问题，这可以归因于以下几个方面：（1）用户有价值的行为通常是稀疏的，例如点赞、评论和赠送虚拟礼物，这些行为容易被模型忽视，使得难以描述用户个性化偏好。（2）在我们平台上主要曝光的内容是短视频，其曝光量是直播内容的九倍，导致直播内容无法充分建模用户偏好。为解决这一问题，我们提出了一种跨域直播推荐的频率感知模型，称为FARM。具体而言，我们首先引入了域内频率感知模块，使我们的模型能够捕捉到用户稀疏但有价值的行为，即通过离散傅里叶变换（DFT）支持的高频信息。为实现短视频和直播之间的用户偏好转移，我们提出了一种新颖的偏好对齐再融合策略，该策略包括两个部分：跨域偏好对齐模块，通过对比学习实现两个领域用户的偏好对齐；以及跨域偏好融合模块，通过一系列量身定制的注意力机制进一步融合两个领域内的用户偏好。在快手直播服务上进行的广泛离线实验和在线A/B测试表明，FARM的有效性和优越性。当前，我们的FARM已经在在线直播服务中部署，并为快手的数亿用户提供服务。 

---
# Bridging Jensen Gap for Max-Min Group Fairness Optimization in Recommendation 

**Title (ZH)**: 桥接 Jensen 不等式以优化推荐中的极小极大分组公平性 

**Authors**: Chen Xu, Yuxin Li, Wenjie Wang, Liang Pang, Jun Xu, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2502.09319)  

**Abstract**: Group max-min fairness (MMF) is commonly used in fairness-aware recommender systems (RS) as an optimization objective, as it aims to protect marginalized item groups and ensures a fair competition platform. However, our theoretical analysis indicates that integrating MMF constraint violates the assumption of sample independence during optimization, causing the loss function to deviate from linear additivity. Such nonlinearity property introduces the Jensen gap between the model's convergence point and the optimal point if mini-batch sampling is applied. Both theoretical and empirical studies show that as the mini-batch size decreases and the group size increases, the Jensen gap will widen accordingly. Some methods using heuristic re-weighting or debiasing strategies have the potential to bridge the Jensen gap. However, they either lack theoretical guarantees or suffer from heavy computational costs. To overcome these limitations, we first theoretically demonstrate that the MMF-constrained objective can be essentially reformulated as a group-weighted optimization objective. Then we present an efficient and effective algorithm named FairDual, which utilizes a dual optimization technique to minimize the Jensen gap. Our theoretical analysis demonstrates that FairDual can achieve a sub-linear convergence rate to the globally optimal solution and the Jensen gap can be well bounded under a mini-batch sampling strategy with random shuffle. Extensive experiments conducted using six large-scale RS backbone models on three publicly available datasets demonstrate that FairDual outperforms all baselines in terms of both accuracy and fairness. Our data and codes are shared at this https URL. 

**Abstract (ZH)**: 群体最大最小公平性（MMF）在公平感知推荐系统（RS）中通常作为优化目标被使用，因为它旨在保护被边缘化的项目群体，并确保公平的竞争平台。然而，我们的理论分析表明，将MMF约束整合到优化过程中会违反样本独立性的假设，导致损失函数偏离线性可加性。这种非线性特性会在使用小批量采样时引入模型收敛点与最优点之间的詹森间隙。无论是理论研究还是实证研究都表明，随着小批量大小的减小和群体规模的增大，詹森间隙也会相应增大。一些使用启发式重新加权或反偏策略的方法有可能缩小詹森间隙，但它们要么缺乏理论保证，要么面临巨大的计算成本。为了克服这些局限性，我们首先从理论上证明，带有MMF约束的目标可以被重新表述为一个群体加权优化目标。随后，我们提出了一个名为FairDual的有效且高效的算法，该算法利用对偶优化技术来最小化詹森间隙。我们的理论分析表明，FairDual可以在随机洗牌的的小批量采样策略下，以亚线性收敛率收敛到全局最优解，并且詹森间隙可以得到良好控制。使用六个大规模推荐系统骨干模型在三个公开数据集上进行的大量实验表明，FairDual在准确性和公平性上都明显优于所有基线方法。我们的数据和代码已在此处 <https://some-url.com> 公开提供。 

---
# KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG 

**Title (ZH)**: KET-RAG：一种高效的成本优化图形-RAG多粒度索引框架 

**Authors**: Yiqian Huang, Shiqi Zhang, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.09304)  

**Abstract**: Graph-RAG constructs a knowledge graph from text chunks to improve retrieval in Large Language Model (LLM)-based question answering. It is particularly useful in domains such as biomedicine, law, and political science, where retrieval often requires multi-hop reasoning over proprietary documents. Some existing Graph-RAG systems construct KNN graphs based on text chunk relevance, but this coarse-grained approach fails to capture entity relationships within texts, leading to sub-par retrieval and generation quality. To address this, recent solutions leverage LLMs to extract entities and relationships from text chunks, constructing triplet-based knowledge graphs. However, this approach incurs significant indexing costs, especially for large document collections.
To ensure a good result accuracy while reducing the indexing cost, we propose KET-RAG, a multi-granular indexing framework. KET-RAG first identifies a small set of key text chunks and leverages an LLM to construct a knowledge graph skeleton. It then builds a text-keyword bipartite graph from all text chunks, serving as a lightweight alternative to a full knowledge graph. During retrieval, KET-RAG searches both structures: it follows the local search strategy of existing Graph-RAG systems on the skeleton while mimicking this search on the bipartite graph to improve retrieval quality. We evaluate eight solutions on two real-world datasets, demonstrating that KET-RAG outperforms all competitors in indexing cost, retrieval effectiveness, and generation quality. Notably, it achieves comparable or superior retrieval quality to Microsoft's Graph-RAG while reducing indexing costs by over an order of magnitude. Additionally, it improves the generation quality by up to 32.4% while lowering indexing costs by around 20%. 

**Abstract (ZH)**: Graph-RAG从文本片段构建知识图谱以提高基于大型语言模型（LLM）的问题回答中的检索性能。它特别适用于生物医学、法律和政治科学等领域，在这些领域中，检索往往需要在专有文档上进行多跳推理。现有的某些Graph-RAG系统基于文本片段的相关性构建最近邻（KNN）图，但这种方法粗粒度的处理无法捕捉文本内的实体关系，导致检索和生成质量不佳。为了解决这一问题，近期的解决方案利用大型语言模型从文本片段中提取实体和关系，构建三元组知识图谱。然而，这种方法会导致显著的索引成本，尤其是在大规模文档集合中。

为确保准确度的同时减少索引成本，我们提出了KET-RAG，一种多粒度索引框架。KET-RAG首先识别一小组关键文本片段，并利用大型语言模型（LLM）构建知识图谱框架。然后，它从所有文本片段构建一个文本-关键词二分图，作为完整知识图谱的轻量级替代。在检索过程中，KET-RAG同时搜索这两种结构：它在框架上遵循现有Graph-RAG系统的局部搜索策略，同时在二分图上模仿这种搜索以提高检索质量。我们在两个真实世界数据集上评估了八种解决方案，结果显示，KET-RAG在索引成本、检索效果和生成质量方面均优于其他竞争者。值得注意的是，与微软的Graph-RAG相比，KET-RAG在检索质量方面达到了相当或更好的水平，同时将其索引成本降低了十倍以上。此外，它在生成质量上提高了多达32.4%，而索引成本降低了约20%。 

---
# Use of Air Quality Sensor Network Data for Real-time Pollution-Aware POI Suggestion 

**Title (ZH)**: 使用空气质量传感器网络数据进行实时污染感知POI建议 

**Authors**: Giuseppe Fasano, Yashar Deldjoo, Tommaso di Noia, Bianca Lau, Sina Adham-Khiabani, Eric Morris, Xia Liu, Ganga Chinna Rao Devarapu, Liam O'Faolain  

**Link**: [PDF](https://arxiv.org/pdf/2502.09155)  

**Abstract**: This demo paper presents AirSense-R, a privacy-preserving mobile application that provides real-time, pollution-aware recommendations for points of interest (POIs) in urban environments. By combining real-time air quality monitoring data with user preferences, the proposed system aims to help users make health-conscious decisions about the locations they visit. The application utilizes collaborative filtering for personalized suggestions, and federated learning for privacy protection, and integrates air pollutant readings from AirSENCE sensor networks in cities such as Bari, Italy, and Cork, Ireland. Additionally, the AirSENCE prediction engine can be employed to detect anomaly readings and interpolate for air quality readings in areas with sparse sensor coverage. This system offers a promising, health-oriented POI recommendation solution that adapts dynamically to current urban air quality conditions while safeguarding user privacy. The code of AirTOWN and a demonstration video is made available at the following repo: this https URL. 

**Abstract (ZH)**: 以下是翻译的内容，符合学术规范：

本文介绍了一种保护隐私的移动应用程序——AirSense-R，该应用程序在城市环境中为兴趣点（POIs）提供实时、污染意识化的推荐。通过结合实时空气质量和用户偏好数据，所提出系统旨在帮助用户做出关于访问地点的健康意识化决策。应用程序利用协同过滤技术提供个性化建议，并采用联邦学习技术保护用户隐私。此外，该应用程序集成了意大利巴里、爱尔兰科克等城市的AirSENSE传感器网络的空气质量读数，并利用AirSENSE预测引擎检测异常读数并插值得到稀疏传感器覆盖区域的空气质量读数。该系统提供了一种具有前景的、面向健康的POI推荐解决方案，能够动态适应当前的城市空气质量状况，同时保障用户隐私。有关AirTOWN的源代码和演示视频可以在以下仓库中获取：this https URL。 

---
# Semantic Ads Retrieval at Walmart eCommerce with Language Models Progressively Trained on Multiple Knowledge Domains 

**Title (ZH)**: 在Walmart电子商务中基于语言模型逐步训练多知识领域进行语义广告检索 

**Authors**: Zhaodong Wang, Weizhi Du, Md Omar Faruk Rokon, Pooshpendu Adhikary, Yanbing Xue, Jiaxuan Xu, Jianghong Zhou, Kuang-chih Lee, Musen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.09089)  

**Abstract**: Sponsored search in e-commerce poses several unique and complex challenges. These challenges stem from factors such as the asymmetric language structure between search queries and product names, the inherent ambiguity in user search intent, and the vast volume of sparse and imbalanced search corpus data. The role of the retrieval component within a sponsored search system is pivotal, serving as the initial step that directly affects the subsequent ranking and bidding systems. In this paper, we present an end-to-end solution tailored to optimize the ads retrieval system on this http URL. Our approach is to pretrain the BERT-like classification model with product category information, enhancing the model's understanding of Walmart product semantics. Second, we design a two-tower Siamese Network structure for embedding structures to augment training efficiency. Third, we introduce a Human-in-the-loop Progressive Fusion Training method to ensure robust model performance. Our results demonstrate the effectiveness of this pipeline. It enhances the search relevance metric by up to 16% compared to a baseline DSSM-based model. Moreover, our large-scale online A/B testing demonstrates that our approach surpasses the ad revenue of the existing production model. 

**Abstract (ZH)**: 电子商务中的付费搜索面临着多种独特且复杂的挑战。这些挑战源于搜索查询和产品名称之间的不对称语言结构、用户搜索意图的固有模糊性，以及庞大且稀疏、不平衡的搜索语料库数据。付费搜索系统中的检索组件起着关键作用，它是直接影响后续排名和竞价系统的第一步。本文提出了一种端到端的整体解决方案，旨在优化此网址上的广告检索系统。我们的方法是使用产品类别信息对类似BERT的分类模型进行预训练，增强模型对沃尔玛产品语义的理解。其次，我们设计了一种双塔Siamese网络结构以增强训练效率。第三，我们引入了一种人机协作的渐进融合训练方法，以确保模型性能的稳定性。实验结果表明，该流水线的有效性，相比基于DSSM的基本模型，搜索相关性度量提高了高达16%。此外，我们在大规模在线A/B测试中也验证了我们的方法能够超越现有生产模型的广告收入。 

---
# Unleashing the Power of Large Language Model for Denoising Recommendation 

**Title (ZH)**: 释放大型语言模型在噪声推荐处理中的强大功能 

**Authors**: Shuyao Wang, Zhi Zheng, Yongduo Sui, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.09058)  

**Abstract**: Recommender systems are crucial for personalizing user experiences but often depend on implicit feedback data, which can be noisy and misleading. Existing denoising studies involve incorporating auxiliary information or learning strategies from interaction data. However, they struggle with the inherent limitations of external knowledge and interaction data, as well as the non-universality of certain predefined assumptions, hindering accurate noise identification. Recently, large language models (LLMs) have gained attention for their extensive world knowledge and reasoning abilities, yet their potential in enhancing denoising in recommendations remains underexplored. In this paper, we introduce LLaRD, a framework leveraging LLMs to improve denoising in recommender systems, thereby boosting overall recommendation performance. Specifically, LLaRD generates denoising-related knowledge by first enriching semantic insights from observational data via LLMs and inferring user-item preference knowledge. It then employs a novel Chain-of-Thought (CoT) technique over user-item interaction graphs to reveal relation knowledge for denoising. Finally, it applies the Information Bottleneck (IB) principle to align LLM-generated denoising knowledge with recommendation targets, filtering out noise and irrelevant LLM knowledge. Empirical results demonstrate LLaRD's effectiveness in enhancing denoising and recommendation accuracy. 

**Abstract (ZH)**: 推荐系统对于个性化用户体验至关重要，但往往依赖于隐式反馈数据，这些数据可能噪音较大且会误导。现有的去噪研究涉及通过辅助信息或交互数据的学习策略来整合信息，但这在面对外部知识和交互数据的固有局限性以及某些预定义假设的特异性时，仍存在困难，影响准确的噪声识别。近年来，大规模语言模型（LLM）因其广泛的领域知识和推理能力而受到关注，但其在推荐中的去噪增强潜力尚未得到充分探索。在本文中，我们提出了一种名为LLaRD的框架，利用大规模语言模型来提高推荐系统中的去噪性能，从而提高整体推荐性能。具体而言，LLaRD首先通过大规模语言模型丰富观测数据的语义洞察，并推断用户-项目偏好知识。然后，它使用一种新颖的事前推理（CoT）技术来揭示用户-项目交互图中的关系知识，以用于去噪。最后，它应用信息瓶颈（IB）原理来使大规模语言模型生成的去噪知识与推荐目标对齐，从而过滤掉噪声和无关的大规模语言模型知识。实验结果表明，LLaRD在提高去噪和推荐准确性方面具有有效性。 

---
# Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation 

**Title (ZH)**: 通过多视角图过滤利用成员-群组关系进行有效的群组推荐 

**Authors**: Chae-Hyun Kim, Yoon-Ryung Choi, Jin-Duk Park, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2502.09050)  

**Abstract**: Group recommendation aims at providing optimized recommendations tailored to diverse groups, enabling groups to enjoy appropriate items. On the other hand, most existing group recommendation methods are built upon deep neural network (DNN) architectures designed to capture the intricate relationships between member-level and group-level interactions. While these DNN-based approaches have proven their effectiveness, they require complex and expensive training procedures to incorporate group-level interactions in addition to member-level interactions. To overcome such limitations, we introduce Group-GF, a new approach for extremely fast recommendations of items to each group via multi-view graph filtering (GF) that offers a holistic view of complex member-group dynamics, without the need for costly model training. Specifically, in Group-GF, we first construct three item similarity graphs manifesting different viewpoints for GF. Then, we discover a distinct polynomial graph filter for each similarity graph and judiciously aggregate the three graph filters. Extensive experiments demonstrate the effectiveness of Group-GF in terms of significantly reducing runtime and achieving state-of-the-art recommendation accuracy. 

**Abstract (ZH)**: 群体推荐旨在提供定制化的优化推荐，以满足多样化群体的需求，使群体能够享受合适的项目。另一方面，现有的大多数群体推荐方法都是基于深度神经网络（DNN）架构构建的，这些架构设计用于捕捉成员级和群体级交互的复杂关系。虽然这些基于DNN的方法已经证明了其有效性，但在增加群体级交互的同时，它们还需要复杂的且昂贵的训练程序来捕捉成员级交互。为了克服这些限制，我们提出了一种名为Group-GF的新方法，通过多视图图过滤（GF）在极短时间内实现对每个群体的项目推荐，而不需要提供复杂的模型训练。具体来说，在Group-GF中，我们首先构建三个项目相似性图，以从不同视角进行GF。然后，我们为每个相似性图发现一个独特的多项式图滤波器，并有效地将这三个图滤波器进行整合。广泛的经验表明，Group-GF在显著降低运行时间和实现业界领先的推荐准确性方面具有有效性。 

---
# Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation 

**Title (ZH)**: 基于准则的图过滤：极快且准确的多准则推荐 

**Authors**: Jin-Duk Park, Jaemin Yoo, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2502.09046)  

**Abstract**: Multi-criteria (MC) recommender systems, which utilize MC rating information for recommendation, are increasingly widespread in various e-commerce domains. However, the MC recommendation using training-based collaborative filtering, requiring consideration of multiple ratings compared to single-criterion counterparts, often poses practical challenges in achieving state-of-the-art performance along with scalable model training. To solve this problem, we propose CA-GF, a training-free MC recommendation method, which is built upon criteria-aware graph filtering for efficient yet accurate MC recommendations. Specifically, first, we construct an item-item similarity graph using an MC user-expansion graph. Next, we design CA-GF composed of the following key components, including 1) criterion-specific graph filtering where the optimal filter for each criterion is found using various types of polynomial low-pass filters and 2) criteria preference-infused aggregation where the smoothed signals from each criterion are aggregated. We demonstrate that CA-GF is (a) efficient: providing the computational efficiency, offering the extremely fast runtime of less than 0.2 seconds even on the largest benchmark dataset, (b) accurate: outperforming benchmark MC recommendation methods, achieving substantial accuracy gains up to 24% compared to the best competitor, and (c) interpretable: providing interpretations for the contribution of each criterion to the model prediction based on visualizations. 

**Abstract (ZH)**: 基于多标准（MC）的推荐系统利用多标准评分信息进行推荐，在各种电子商务领域中的应用越来越广泛。然而，基于训练的数据驱动协同过滤的MC推荐方法，在相比单一标准方法需要考虑多种评分的情况下，往往会在实现前沿性能和可扩展模型训练方面遇到实际挑战。为了解决这个问题，我们提出了一种无需训练的方法CA-GF，该方法基于多标准感知图过滤技术，以高效准确地提供多标准推荐。具体来说，首先，我们通过构建多标准用户扩展图来构建项目-项目相似性图。接下来，设计CA-GF方法，其包含以下几个核心组件：（1）标准特定的图过滤，其中为每个标准找到最优滤波器，使用各种类型的多项式低通滤波器；（2）偏好聚合，在此步骤中，从每个标准平滑的信号得到聚合。我们证明了CA-GF方法在以下方面表现出色：（a）高效：提供计算效率，即使在最大的基准数据集上，仍能在不到0.2秒内提供运行时间；（b）准确：在基准多标准推荐方法中表现优异，与最佳竞争者相比，准确率提升高达24%；（c）可解释性：提供基于可视化技术对每个标准对模型预测贡献的解释。 

---
# A Contextual-Aware Position Encoding for Sequential Recommendation 

**Title (ZH)**: 一种基于情境的序列位置编码方法 

**Authors**: Jun Yuan, Guohao Cai, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.09027)  

**Abstract**: Sequential recommendation (SR), which encodes user activity to predict the next action, has emerged as a widely adopted strategy in developing commercial personalized recommendation systems. A critical component of modern SR models is the attention mechanism, which synthesizes users' historical activities. This mechanism is typically order-invariant and generally relies on position encoding (PE). Conventional SR models simply assign a learnable vector to each position, resulting in only modest gains compared to traditional recommendation models. Moreover, limited research has been conducted on position encoding tailored for sequential recommendation, leaving a significant gap in addressing its unique requirements. To bridge this gap, we propose a novel Contextual-Aware Position Encoding method for sequential recommendation, abbreviated as CAPE. To the best of our knowledge, CAPE is the first PE method specifically designed for sequential recommendation. Comprehensive experiments conducted on benchmark SR datasets demonstrate that CAPE consistently enhances multiple mainstream backbone models and achieves state-of-the-art performance, across small and large scale model size. Furthermore, we deployed CAPE in an industrial setting on a real-world commercial platform, clearly showcasing the effectiveness of our approach. Our source code is available at this https URL. 

**Abstract (ZH)**: 顺序推荐（SR），该方法通过编码用户活动来预测下一个行为，已成为开发商业个性化推荐系统的广泛采用策略。现代SR模型中的关键组成部分是注意机制，它综合了用户的 histórico 活动。这种机制通常是顺序无关的，通常依赖于位置编码（PE）。传统 SR 模型简单地为每个位置分配一个可学习的向量，相比于传统的推荐模型，这种做法仅能带来微小的改进。此外，关于适用于顺序推荐的位置编码的研究较少，这在满足其独特需求方面留下了一个显著的差距。为弥补此差距，我们提出了一种新颖的面向上下文的位置编码方法，称为 Contextual-Aware Position Encoding（CAPE）。据我们所知，CAPE 是第一个专门设计用于顺序推荐的位置编码方法。在基准 SR 数据集上进行的全面实验表明，CAPE 在不同规模的模型中都能显著提升多种主流的主干模型，并达到了最先进的性能。此外，我们在实际商业平台的工业环境中部署了 CAPE，清楚地展示了我们方法的有效性。我们的源代码可在以下网址获取：this [URL]。 

---
# Optimal Dataset Size for Recommender Systems: Evaluating Algorithms' Performance via Downsampling 

**Title (ZH)**: 推荐系统中 optimal 数据集大小的研究：通过下采样评估算法性能 

**Authors**: Ardalan Arabzadeh, Joeran Beel, Tobias Vente  

**Link**: [PDF](https://arxiv.org/pdf/2502.08845)  

**Abstract**: This thesis investigates dataset downsampling as a strategy to optimize energy efficiency in recommender systems while maintaining competitive performance. With increasing dataset sizes posing computational and environmental challenges, this study explores the trade-offs between energy efficiency and recommendation quality in Green Recommender Systems, which aim to reduce environmental impact. By applying two downsampling approaches to seven datasets, 12 algorithms, and two levels of core pruning, the research demonstrates significant reductions in runtime and carbon emissions. For example, a 30% downsampling portion can reduce runtime by 52% compared to the full dataset, leading to a carbon emission reduction of up to 51.02 KgCO2e during the training of a single algorithm on a single dataset. The analysis reveals that algorithm performance under different downsampling portions depends on factors like dataset characteristics, algorithm complexity, and the specific downsampling configuration (scenario dependent). Some algorithms, which showed lower nDCG@10 scores compared to higher-performing ones, exhibited lower sensitivity to the amount of training data, offering greater potential for efficiency in lower downsampling portions. On average, these algorithms retained 81% of full-size performance using only 50% of the training set. In certain downsampling configurations, where more users were progressively included while keeping the test set size fixed, they even showed higher nDCG@10 scores than when using the full dataset. These findings highlight the feasibility of balancing sustainability and effectiveness, providing insights for designing energy-efficient recommender systems and promoting sustainable AI practices. 

**Abstract (ZH)**: 本论文探讨了数据集下采样作为优化推荐系统能效的一种策略，同时保持竞争力。随着数据集规模的增加带来的计算和环境挑战，本研究探索了绿色推荐系统中的能效与推荐质量之间的权衡，旨在减少环境影响。通过将两种下采样方法应用于七个数据集、十二种算法以及两种核心剪枝级别，研究证明在运行时间和碳排放方面实现了显著减少。例如，与完整数据集相比，30%的数据下采样可将运行时间减少52%，并在单个数据集和单一算法的训练过程中减少高达51.02公斤当量二氧化碳的碳排放。分析显示，不同下采样比例下的算法性能依赖于数据集特性、算法复杂性以及具体的下采样配置（场景依赖）。一些性能较低的算法（nDCG@10分数较低），相较于高性能算法，其对训练数据量的变化表现出了较低的敏感性，从而在较低的下采样比例下具有更大的能效提升潜力。平均而言，这些算法仅使用训练集的50%便能保留81%的全尺寸性能。在某些下采样配置中，当逐渐增加用户数量并保持测试集大小不变时，它们在nDCG@10分数上甚至超过了使用完整数据集的情况。这些发现强调了在可持续性和有效性之间实现平衡的可能性，并为设计能效推荐系统以及促进可持续人工智能实践提供了有价值的见解。 

---
# Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation 

**Title (ZH)**: 任意模态提问：多模态检索增强生成综述

这个标题翻译成中文采用了较为严谨和学术化的表达方式。其中，“Ask in Any Modality”翻译为“任意模态提问”，“Multimodal Retrieval-Augmented Generation”翻译为“多模态检索增强生成”，“Comprehensive Survey”则翻译为“综述”，整体符合学术论文标题的规范和要求。 

**Authors**: Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2502.08826)  

**Abstract**: Large Language Models (LLMs) struggle with hallucinations and outdated knowledge due to their reliance on static training data. Retrieval-Augmented Generation (RAG) mitigates these issues by integrating external dynamic information enhancing factual and updated grounding. Recent advances in multimodal learning have led to the development of Multimodal RAG, incorporating multiple modalities such as text, images, audio, and video to enhance the generated outputs. However, cross-modal alignment and reasoning introduce unique challenges to Multimodal RAG, distinguishing it from traditional unimodal RAG. This survey offers a structured and comprehensive analysis of Multimodal RAG systems, covering datasets, metrics, benchmarks, evaluation, methodologies, and innovations in retrieval, fusion, augmentation, and generation. We precisely review training strategies, robustness enhancements, and loss functions, while also exploring the diverse Multimodal RAG scenarios. Furthermore, we discuss open challenges and future research directions to support advancements in this evolving field. This survey lays the foundation for developing more capable and reliable AI systems that effectively leverage multimodal dynamic external knowledge bases. Resources are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）因依赖静态训练数据而在幻觉和过时知识方面存在问题。检索增强生成（RAG）通过集成外部动态信息来缓解这些问题，从而增强事实和更新的真实性。近年来，多模态学习的进步导致了多模态RAG的开发，它结合了多种模态（如文本、图像、音频和视频），以增强生成输出。然而，跨模态对齐和推理引入了多模态RAG的独特挑战，使其区别于传统的单模态RAG。本文综述提供了一个结构化和全面的多模态RAG系统分析，涵盖了数据集、评估指标、基准测试、评估方法、检索、融合、增强和生成的方法论及创新。我们详细回顾了训练策略、鲁棒性增强和损失函数，并探讨了多种多模态RAG场景。此外，我们讨论了开放挑战和未来研究方向，以支持这一领域的发展。本文奠定了开发更强大和可靠的AI系统的基础，这些系统可以有效地利用多模态动态外部知识库。相关资源可在以下链接获取：[这里插入链接] 

---
