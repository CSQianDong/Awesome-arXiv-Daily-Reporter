# Contrastive Representation for Interactive Recommendation 

**Title (ZH)**: 交互式推荐的对比表示方法 

**Authors**: Jingyu Li, Zhiyong Feng, Dongxiao He, Hongqi Chen, Qinghang Gao, Guoli Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18396)  

**Abstract**: Interactive Recommendation (IR) has gained significant attention recently for its capability to quickly capture dynamic interest and optimize both short and long term objectives. IR agents are typically implemented through Deep Reinforcement Learning (DRL), because DRL is inherently compatible with the dynamic nature of IR. However, DRL is currently not perfect for IR. Due to the large action space and sample inefficiency problem, training DRL recommender agents is challenging. The key point is that useful features cannot be extracted as high-quality representations for the recommender agent to optimize its policy. To tackle this problem, we propose Contrastive Representation for Interactive Recommendation (CRIR). CRIR efficiently extracts latent, high-level preference ranking features from explicit interaction, and leverages the features to enhance users' representation. Specifically, the CRIR provides representation through one representation network, and refines it through our proposed Preference Ranking Contrastive Learning (PRCL). The key insight of PRCL is that it can perform contrastive learning without relying on computations involving high-level representations or large potential action sets. Furthermore, we also propose a data exploiting mechanism and an agent training mechanism to better adapt CRIR to the DRL backbone. Extensive experiments have been carried out to show our method's superior improvement on the sample efficiency while training an DRL-based IR agent. 

**Abstract (ZH)**: 交互推荐（IR）近年来因其能够迅速捕捉动态兴趣并优化短期和长期目标而获得了广泛关注。IR代理通常通过深度强化学习（DRL）实现，因为DRL本身与IR的动态特性相兼容。然而，DRL当前并不完全适用于IR。由于动作空间庞大和样本效率问题，训练DRL推荐代理是一项挑战性任务。关键在于难以提取出有用特征，使其以高质量表示形式优化其策略。为了应对这一问题，我们提出了对比表示法用于交互推荐（CRIR）。CRIR能够高效地从显式的交互中提取潜在的高层次偏好排序特征，并利用这些特征增强用户的表示。具体来说，CRIR通过一个表示网络提供表示，并通过我们提出的偏好排序对比学习（PRCL）对该表示进行细化。PRCL的关键洞察是，它可以在不依赖于涉及高层表示或大规模潜在动作集的计算的情况下执行对比学习。此外，我们还提出了数据利用机制和代理训练机制，以更好地使CRIR适应DRL框架。我们进行了大量的实验，展示了在训练基于DRL的IR代理方面，我们的方法在样本效率方面的显著改进。 

---
# RaSeRec: Retrieval-Augmented Sequential Recommendation 

**Title (ZH)**: RaSeRec：检索增强的序列推荐 

**Authors**: Xinping Zhao, Baotian Hu, Yan Zhong, Shouzheng Huang, Zihao Zheng, Meng Wang, Haofen Wang, Min zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18378)  

**Abstract**: Although prevailing supervised and self-supervised learning (SSL)-augmented sequential recommendation (SeRec) models have achieved improved performance with powerful neural network architectures, we argue that they still suffer from two limitations: (1) Preference Drift, where models trained on past data can hardly accommodate evolving user preference; and (2) Implicit Memory, where head patterns dominate parametric learning, making it harder to recall long tails. In this work, we explore retrieval augmentation in SeRec, to address these limitations. To this end, we propose a Retrieval-Augmented Sequential Recommendation framework, named RaSeRec, the main idea of which is to maintain a dynamic memory bank to accommodate preference drifts and retrieve relevant memories to augment user modeling explicitly. It consists of two stages: (i) collaborative-based pre-training, which learns to recommend and retrieve; (ii) retrieval-augmented fine-tuning, which learns to leverage retrieved memories. Extensive experiments on three datasets fully demonstrate the superiority and effectiveness of RaSeRec. 

**Abstract (ZH)**: 尽管现有的监督学习和自我监督学习（SSL）增强的序列推荐（SeRec）模型通过强大的神经网络架构已经取得了改进的性能，我们仍然认为它们存在两个局限性：（1）偏好漂移，即在过去的数据上训练的模型难以适应用户偏好的演变；（2）隐含记忆，其中头部模式主导了参数学习，使得难以召回长尾项。在本文中，我们探讨了在SeRec中引入检索增强的方法，以解决这些局限性。为此，我们提出了一种检索增强序列推荐框架，命名为RaSeRec，其主要思想是维护一个动态的记忆库以适应偏好漂移，并检索相关记忆以显式增强用户建模。该框架包含两个阶段：（i）基于合作的预训练，该阶段学习推荐和检索；（ii）检索增强的微调，该阶段学习利用检索的记忆。在三个数据集上的广泛实验充分展示了RaSeRec的优势和有效性。 

---
# An Automatic Graph Construction Framework based on Large Language Models for Recommendation 

**Title (ZH)**: 基于大型语言模型的自动图构建推荐框架 

**Authors**: Rong Shan, Jianghao Lin, Chenxu Zhu, Bo Chen, Menghui Zhu, Kangning Zhang, Jieming Zhu, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18241)  

**Abstract**: Graph neural networks (GNNs) have emerged as state-of-the-art methods to learn from graph-structured data for recommendation. However, most existing GNN-based recommendation methods focus on the optimization of model structures and learning strategies based on pre-defined graphs, neglecting the importance of the graph construction stage. Earlier works for graph construction usually rely on speciffic rules or crowdsourcing, which are either too simplistic or too labor-intensive. Recent works start to utilize large language models (LLMs) to automate the graph construction, in view of their abundant open-world knowledge and remarkable reasoning capabilities. Nevertheless, they generally suffer from two limitations: (1) invisibility of global view (e.g., overlooking contextual information) and (2) construction inefficiency. To this end, we introduce AutoGraph, an automatic graph construction framework based on LLMs for recommendation. Specifically, we first use LLMs to infer the user preference and item knowledge, which is encoded as semantic vectors. Next, we employ vector quantization to extract the latent factors from the semantic vectors. The latent factors are then incorporated as extra nodes to link the user/item nodes, resulting in a graph with in-depth global-view semantics. We further design metapath-based message aggregation to effectively aggregate the semantic and collaborative information. The framework is model-agnostic and compatible with different backbone models. Extensive experiments on three real-world datasets demonstrate the efficacy and efffciency of AutoGraph compared to existing baseline methods. We have deployed AutoGraph in Huawei advertising platform, and gain a 2.69% improvement on RPM and a 7.31% improvement on eCPM in the online A/B test. Currently AutoGraph has been used as the main trafffc model, serving hundreds of millions of people. 

**Abstract (ZH)**: 图神经网络（GNNs）已成为从图结构数据中进行推荐的最先进的方法。然而，现有的大多数基于GNN的推荐方法主要集中在模型结构和基于预定义图的学习策略的优化上，忽略了图构建阶段的重要性。早期的图构建工作通常依赖于特定规则或众包，这些方法要么过于简单，要么过于费时。最近的工作开始利用大语言模型（LLMs）来自动化图构建，利用它们丰富的开放世界知识和卓越的推理能力。尽管如此，它们通常存在两个局限性：（1）全局视图的不可见性（例如，忽视上下文信息）和（2）构建效率低下。为此，我们提出了AutoGraph，这是一种基于LLMs的推荐自动图构建框架。具体来说，我们首先使用LLMs推断用户偏好和项目知识，并将这些知识编码为语义向量。然后，我们使用向量量化来从语义向量中提取潜在因子。接着，将这些潜在因子作为额外节点加入到用户/项目节点中，构建出具有深入全局视图语义的图。我们还设计了基于元路径的消息聚合来有效聚合语义和协同信息。该框架具有模型无关性，并且与不同的骨干模型兼容。在三个真实世界数据集上的广泛实验表明，AutoGraph在与现有基线方法相比在有效性和效率上都表现出色。我们已在华为广告平台部署了AutoGraph，通过在线A/B测试，在RPM上获得了2.69%的提升，在eCPM上获得了7.31%的提升。目前，AutoGraph已成为主要流量模型，服务于数亿用户。 

---
# Efficient Long Context Language Model Retrieval with Compression 

**Title (ZH)**: 高效的长上下文语言模型检索与压缩 

**Authors**: Minju Seo, Jinheon Baek, Seongyun Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18232)  

**Abstract**: Long Context Language Models (LCLMs) have emerged as a new paradigm to perform Information Retrieval (IR), which enables the direct ingestion and retrieval of information by processing an entire corpus in their single context, showcasing the potential to surpass traditional sparse and dense retrieval methods. However, processing a large number of passages within in-context for retrieval is computationally expensive, and handling their representations during inference further exacerbates the processing time; thus, we aim to make LCLM retrieval more efficient and potentially more effective with passage compression. Specifically, we propose a new compression approach tailored for LCLM retrieval, which is trained to maximize the retrieval performance while minimizing the length of the compressed passages. To accomplish this, we generate the synthetic data, where compressed passages are automatically created and labeled as chosen or rejected according to their retrieval success for a given query, and we train the proposed Compression model for Long context Retrieval (CoLoR) with this data via preference optimization while adding the length regularization loss on top of it to enforce brevity. Through extensive experiments on 9 datasets, we show that CoLoR improves the retrieval performance by 6% while compressing the in-context size by a factor of 1.91. 

**Abstract (ZH)**: 长上下文语言模型（LCLMs）已成为一种新的信息检索（IR）范式，能够通过一次性处理整个语料库直接进行信息的摄入和检索，展示了超越传统稀疏和密集检索方法的潜力。然而，在上下文中处理大量段落进行检索会消耗大量的计算资源，而在推理过程中处理其表示进一步加剧了处理时间；因此，我们旨在通过段落压缩来提高LCLM检索的效率并可能使其更加有效。具体来说，我们提出了一种新的针对LCLM检索的压缩方法，该方法在最大限度地提高检索性能的同时，尽量缩短压缩段落的长度。为了实现这一点，我们生成了合成数据，在其中自动创建并标注压缩段落，这些段落根据给定查询的检索成功率被标记为选择或拒绝。然后，我们使用这些数据通过偏好优化训练提出的长上下文检索压缩模型（CoLoR），并在其中加入了长度正则化损失，以促进简洁性。通过在9个数据集上的广泛实验，我们表明，CoLoR在压缩上下文大小1.91倍的情况下，将检索性能提高了6%。 

---
# Molar: Multimodal LLMs with Collaborative Filtering Alignment for Enhanced Sequential Recommendation 

**Title (ZH)**: Molar：基于协作过滤对齐的多模态大语言模型以增强序贯推荐 

**Authors**: Yucong Luo, Qitao Qin, Hao Zhang, Mingyue Cheng, Ruiran Yan, Kefan Wang, Jie Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18176)  

**Abstract**: Sequential recommendation (SR) systems have evolved significantly over the past decade, transitioning from traditional collaborative filtering to deep learning approaches and, more recently, to large language models (LLMs). While the adoption of LLMs has driven substantial advancements, these models inherently lack collaborative filtering information, relying primarily on textual content data neglecting other modalities and thus failing to achieve optimal recommendation performance. To address this limitation, we propose Molar, a Multimodal large language sequential recommendation framework that integrates multiple content modalities with ID information to capture collaborative signals effectively. Molar employs an MLLM to generate unified item representations from both textual and non-textual data, facilitating comprehensive multimodal modeling and enriching item embeddings. Additionally, it incorporates collaborative filtering signals through a post-alignment mechanism, which aligns user representations from content-based and ID-based models, ensuring precise personalization and robust performance. By seamlessly combining multimodal content with collaborative filtering insights, Molar captures both user interests and contextual semantics, leading to superior recommendation accuracy. Extensive experiments validate that Molar significantly outperforms traditional and LLM-based baselines, highlighting its strength in utilizing multimodal data and collaborative signals for sequential recommendation tasks. The source code is available at this https URL. 

**Abstract (ZH)**: 在过去的十年中，序列推荐（SR）系统经历了显著的发展，从传统的协同过滤方法过渡到深度学习方法，并且最近转向了大型语言模型（LLMs）。虽然LLMs的应用带动了重大进步，但这些模型本质上缺乏协同过滤信息，主要依赖文本内容数据而忽视了其他类型的数据，因此未能实现最佳的推荐性能。为了解决这一局限，我们提出了Molar，这是一种多模态大型语言序列推荐框架，能够整合多种内容模态和ID信息，以有效捕获协作信号。Molar利用一个MLLM从文本和非文本数据生成统一的项目表示，促进全面的多模态建模并丰富项目嵌入。此外，它通过后对齐机制将基于内容和基于ID的模型用户表示对齐，以确保精确的个性化和稳健的性能。通过无缝结合多模态内容和协同过滤洞察，Molar能够捕捉用户兴趣和上下文语义，从而获得更优的推荐准确性。大量的实验验证了Molar显著优于传统的和基于LLM的方法，突显了其利用多模态数据和协作信号进行序列推荐任务的优势。源代码可在以下链接获取：this https URL。 

---
# Unlocking the Hidden Treasures: Enhancing Recommendations with Unlabeled Data 

**Title (ZH)**: 解锁隐藏的 treasures：利用未标记数据增强推荐系统 

**Authors**: Yuhan Zhao, Rui Chen, Qilong Han, Hongtao Song, Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.18170)  

**Abstract**: Collaborative filtering (CF) stands as a cornerstone in recommender systems, yet effectively leveraging the massive unlabeled data presents a significant challenge. Current research focuses on addressing the challenge of unlabeled data by extracting a subset that closely approximates negative samples. Regrettably, the remaining data are overlooked, failing to fully integrate this valuable information into the construction of user preferences. To address this gap, we introduce a novel positive-neutral-negative (PNN) learning paradigm. PNN introduces a neutral class, encompassing intricate items that are challenging to categorize directly as positive or negative samples. By training a model based on this triple-wise partial ranking, PNN offers a promising solution to learning complex user preferences. Through theoretical analysis, we connect PNN to one-way partial AUC (OPAUC) to validate its efficacy. Implementing the PNN paradigm is, however, technically challenging because: (1) it is difficult to classify unlabeled data into neutral or negative in the absence of supervised signals; (2) there does not exist any loss function that can handle set-level triple-wise ranking relationships. To address these challenges, we propose a semi-supervised learning method coupled with a user-aware attention model for knowledge acquisition and classification refinement. Additionally, a novel loss function with a two-step centroid ranking approach enables handling set-level rankings. Extensive experiments on four real-world datasets demonstrate that, when combined with PNN, a wide range of representative CF models can consistently and significantly boost their performance. Even with a simple matrix factorization, PNN can achieve comparable performance to sophisticated graph neutral networks. 

**Abstract (ZH)**: 协同过滤（CF）是推荐系统中的基石，然而有效地利用大量未标记数据仍然是一个重大挑战。当前的研究主要集中在通过提取一个接近负样本的子集来应对未标记数据的挑战。遗憾的是，剩余的数据被忽视了，未能充分利用这些有价值的信息来构建用户偏好。为解决这一问题，我们提出了新的正中负三元学习范式（PNN）。PNN引入了中性类，涵盖了难以直接归类为正或负样本的复杂物品。通过基于这种三元部分排名训练模型，PNN 提供了一种学习复杂用户偏好的有前途的方法。通过理论分析，我们将 PNN 连接到单向部分AUC（OPAUC）以验证其效果。

然而，实施PNN范式存在技术上挑战：（1）在缺乏监督信号的情况下，难以将未标记数据分类为中性或负样本；（2）不存在可以处理集合层次三元部分排名关系的损失函数。为解决这些挑战，我们提出了一种半监督学习方法，并结合了用户感知注意力模型来进行知识获取和分类细化。此外，引入了一种新的损失函数，采用两步质心排名方法，以处理集合层次的排名。在四个真实数据集上的广泛实验表明，当与PNN结合使用时，各种代表性CF模型可以一致且显著地提升其性能。即使使用简单的矩阵分解，PNN也能达到复杂图中性网络相当的性能。 

---
# From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking 

**Title (ZH)**: 从成对到排名：迈向理想的协作过滤伪排名阶梯模型 

**Authors**: Yuhan Zhao, Rui Chen, Li Chen, Shuang Zhang, Qilong Han, Hongtao Song  

**Link**: [PDF](https://arxiv.org/pdf/2412.18168)  

**Abstract**: Intuitively, an ideal collaborative filtering (CF) model should learn from users' full rankings over all items to make optimal top-K recommendations. Due to the absence of such full rankings in practice, most CF models rely on pairwise loss functions to approximate full rankings, resulting in an immense performance gap. In this paper, we provide a novel analysis using the multiple ordinal classification concept to reveal the inevitable gap between a pairwise approximation and the ideal case. However, bridging the gap in practice encounters two formidable challenges: (1) none of the real-world datasets contains full ranking information; (2) there does not exist a loss function that is capable of consuming ranking information. To overcome these challenges, we propose a pseudo-ranking paradigm (PRP) that addresses the lack of ranking information by introducing pseudo-rankings supervised by an original noise injection mechanism. Additionally, we put forward a new ranking loss function designed to handle ranking information effectively. To ensure our method's robustness against potential inaccuracies in pseudo-rankings, we equip the ranking loss function with a gradient-based confidence mechanism to detect and mitigate abnormal gradients. Extensive experiments on four real-world datasets demonstrate that PRP significantly outperforms state-of-the-art methods. 

**Abstract (ZH)**: 直观地讲，理想的协同过滤（CF）模型应该通过学习用户对所有项目的完整排名来做出最优的Top-K推荐。然而，在实践中无法获得这样的完整排名，因此大多数CF模型依赖于成对损失函数来近似完整排名，导致巨大的性能差距。在本文中，我们通过多重序分类的概念提供了一种新的分析方法，揭示了成对近似与理想情况之间的不可避免的差距。然而，在实际中弥合这一差距面临两大挑战：（1）现实世界的数据集并未包含完整的排名信息；（2）不存在能够利用排名信息的损失函数。为克服这些挑战，我们提出了一种伪排名范式（PRP），通过引入受原始噪声注入机制监督的伪排名来解决缺乏排名信息的问题。此外，我们还提出了一种新的排名损失函数，以有效处理排名信息。为了确保该方法对伪排名潜在不准确性的鲁棒性，我们在排名损失函数中加入了基于梯度的信心机制，以检测和缓解异常梯度。在四个现实世界数据集上的广泛实验表明，PRP 显著优于现有最先进的方法。 

---
# BRIDGE: Bundle Recommendation via Instruction-Driven Generation 

**Title (ZH)**: BRIDGE：基于指令驱动生成的束推荐方法 

**Authors**: Tuan-Nghia Bui, Huy-Son Nguyen, Cam-Van Nguyen Thi, Hoang-Quynh Le, Duc-Trong Le  

**Link**: [PDF](https://arxiv.org/pdf/2412.18092)  

**Abstract**: Bundle recommendation aims to suggest a set of interconnected items to users. However, diverse interaction types and sparse interaction matrices often pose challenges for previous approaches in accurately predicting user-bundle adoptions. Inspired by the distant supervision strategy and generative paradigm, we propose BRIDGE, a novel framework for bundle recommendation. It consists of two main components namely the correlation-based item clustering and the pseudo bundle generation modules. Inspired by the distant supervision approach, the former is to generate more auxiliary information, e.g., instructive item clusters, for training without using external data. This information is subsequently aggregated with collaborative signals from user historical interactions to create pseudo `ideal' bundles. This capability allows BRIDGE to explore all aspects of bundles, rather than being limited to existing real-world bundles. It effectively bridging the gap between user imagination and predefined bundles, hence improving the bundle recommendation performance. Experimental results validate the superiority of our models over state-of-the-art ranking-based methods across five benchmark datasets. 

**Abstract (ZH)**: 组合推荐的目标是向用户推荐一组相互关联的项目。然而，多样化的交互类型和稀疏的交互矩阵往往给以往方法准确预测用户的组合采用带来了挑战。受远程监督策略和生成范式的启发，我们提出了一种新颖的组合推荐框架——BRIDGE。该框架包含两个主要模块，即基于关联的项目聚类模块和伪组合生成模块。受远程监督方法的启发，前者旨在生成更多的辅助信息，例如指导性的项目聚类，在不使用外部数据的情况下用于训练。这些信息随后与用户的历史交互的协同信号结合，生成伪“理想”组合。这种能力使BRIDGE能够探索所有可能的组合方面，而不仅仅是局限于现有的实际组合。它有效地弥合了用户想象与预定义组合之间的差距，从而提高了组合推荐的性能。实验结果表明，我们的模型在五个基准数据集中优于最先进的基于排名的方法。 

---
# Prompt Tuning for Item Cold-start Recommendation 

**Title (ZH)**: 为了解决冷启动项目的推荐问题，本文提出了提示调整方法（Prompt Tuning）进行项目冷启动推荐。 

**Authors**: Yuezihan Jiang, Gaode Chen, Wenhan Zhang, Jingchi Wang, Yinjie Jiang, Qi Zhang, Jingjian Lin, Peng Jiang, Kaigui Bian  

**Link**: [PDF](https://arxiv.org/pdf/2412.18082)  

**Abstract**: The item cold-start problem is crucial for online recommender systems, as the success of the cold-start phase determines whether items can transition into popular ones. Prompt learning, a powerful technique used in natural language processing (NLP) to address zero- or few-shot problems, has been adapted for recommender systems to tackle similar challenges. However, existing methods typically rely on content-based properties or text descriptions for prompting, which we argue may be suboptimal for cold-start recommendations due to 1) semantic gaps with recommender tasks, 2) model bias caused by warm-up items contribute most of the positive feedback to the model, which is the core of the cold-start problem that hinders the recommender quality on cold-start items. We propose to leverage high-value positive feedback, termed pinnacle feedback as prompt information, to simultaneously resolve the above two problems. We experimentally prove that compared to the content description proposed in existing works, the positive feedback is more suitable to serve as prompt information by bridging the semantic gaps. Besides, we propose item-wise personalized prompt networks to encode pinnaclce feedback to relieve the model bias by the positive feedback dominance problem. Extensive experiments on four real-world datasets demonstrate the superiority of our model over state-of-the-art methods. Moreover, PROMO has been successfully deployed on a popular short-video sharing platform, a billion-user scale commercial short-video application, achieving remarkable performance gains across various commercial metrics within cold-start scenarios 

**Abstract (ZH)**: 在线推荐系统的冷启动问题是至关重要的，因为冷启动阶段的成功与否决定了物品能否顺利转变为流行物品。提示学习（Prompt Learning）作为一种在自然语言处理（NLP）领域有效解决零样本或少样本问题的技术，已被应用于推荐系统以应对类似挑战。然而，现有方法通常依赖于基于内容的属性或文本描述来生成提示，这我们认为对于冷启动推荐可能并非最优解，原因在于：1) 与推荐任务之间存在的语义差距，2) 模型偏差，在预热阶段物品提供的正面反馈构成了模型的主要正反馈，这正是冷启动问题的核心，阻碍了冷启动物品的推荐质量。我们提出利用高价值的正面反馈作为提示信息，称之为顶点反馈，以同时解决上述两个问题。实验表明，与现有工作中提出的基于内容描述相比，正面反馈更能通过减少语义差距来作为提示信息。此外，我们提出了一种物品级别个性化的提示网络，将顶点反馈编码以缓解正面反馈主导问题导致的模型偏差。在四个真实世界数据集上的实验结果证明了我们模型相比现有领先方法的优越性。此外，PROMO 已成功部署在一个亿级用户规模的短视频应用平台上，在冷启动场景中实现了多种商业指标上的显著性能提升。 

---
# Time-Probability Dependent Knowledge Extraction in IoT-enabled Smart Building 

**Title (ZH)**: 基于物联网的智能建筑中时间-概率依赖的知识提取 

**Authors**: Hangli Ge, Hirotsugu Seike, Noboru Koshizuka  

**Link**: [PDF](https://arxiv.org/pdf/2412.18042)  

**Abstract**: Smart buildings incorporate various emerging Internet of Things (IoT) applications for comprehensive management of energy efficiency, human comfort, automation, and security. However, the development of a knowledge extraction framework is fundamental. Currently, there is a lack of a unified and practical framework for modeling heterogeneous sensor data within buildings. In this paper, we propose a practical inference framework for extracting status-to-event knowledge within smart building. Our proposal includes IoT-based API integration, ontology model design, and time probability dependent knowledge extraction methods. The Building Topology Ontology (BOT) was leveraged to construct spatial relations among sensors and spaces within the building. We utilized Apache Jena Fuseki's SPARQL server for storing and querying the RDF triple data. Two types of knowledge could be extracted: timestamp-based probability for abnormal event detection and time interval-based probability for conjunction of multiple events. We conducted experiments (over a 78-day period) in a real smart building environment. The data of light and elevator states has been collected for evaluation. The evaluation revealed several inferred events, such as room occupancy, elevator trajectory tracking, and the conjunction of both events. The numerical values of detected event counts and probability demonstrate the potential for automatic control in the smart building. 

**Abstract (ZH)**: 智能建筑融合了各种新兴的物联网（IoT）应用，以实现全方位的能耗管理、人体舒适度调控、自动化和安全控制。然而，构建知识提取框架是至关重要的。目前，缺乏一个统一且实用的框架来建模建筑物内的异构传感器数据。在本文中，我们提出了一种实用的推理框架，用于在智能建筑中提取状态到事件的知识。我们的提案包括基于IoT的应用程序接口（API）集成、本体模型设计以及时间概率相关的知识提取方法。我们利用Building Topology Ontology (BOT) 构建了建筑物内部传感器和空间之间的空间关系。我们使用Apache Jena Fuseki的SPARQL服务器来存储和查询RDF三元组数据。我们可以提取两种类型的知识：基于时间戳的概率，用于异常事件检测；基于时间间隔的概率，用于多个事件的联合。我们在一个真实的智能建筑环境中进行了实验（长达78天），收集了灯光和电梯状态的数据以进行评估。评估结果显示了一些推断出的事件，如房间占用情况、电梯轨迹跟踪及其联合事件。检测到的事件数量和概率的数值显示了智能建筑中自动控制的潜在可能性。 

---
# WavePulse: Real-time Content Analytics of Radio Livestreams 

**Title (ZH)**: WavePulse：实时广播现场内容分析 

**Authors**: Govind Mittal, Sarthak Gupta, Shruti Wagle, Chirag Chopra, Anthony J DeMattee, Nasir Memon, Mustaque Ahamad, Chinmay Hegde  

**Link**: [PDF](https://arxiv.org/pdf/2412.17998)  

**Abstract**: Radio remains a pervasive medium for mass information dissemination, with AM/FM stations reaching more Americans than either smartphone-based social networking or live television. Increasingly, radio broadcasts are also streamed online and accessed over the Internet. We present WavePulse, a framework that records, documents, and analyzes radio content in real-time. While our framework is generally applicable, we showcase the efficacy of WavePulse in a collaborative project with a team of political scientists focusing on the 2024 Presidential Elections. We use WavePulse to monitor livestreams of 396 news radio stations over a period of three months, processing close to 500,000 hours of audio streams. These streams were converted into time-stamped, diarized transcripts and analyzed to track answer key political science questions at both the national and state levels. Our analysis revealed how local issues interacted with national trends, providing insights into information flow. Our results demonstrate WavePulse's efficacy in capturing and analyzing content from radio livestreams sourced from the Web. Code and dataset can be accessed at \url{this https URL}. 

**Abstract (ZH)**: 广播仍然是一个普遍的信息传播媒介，AM/FM电台比基于智能手机的社交网络或直播电视更能触及更多的美国人。随着广播节目的发展，越来越多的广播内容也通过互联网进行实时流媒体，并通过网络访问。我们提出了WavePulse框架，该框架可以实时记录、文档化和分析广播内容。虽然WavePulse框架具有通用性，但我们在此与一支由政治学者组成的团队合作，重点关注2024年总统选举项目中WavePulse的有效性。我们使用WavePulse来监控为期三个月内396家新闻广播电台的直播内容，处理了接近50万小时的音频流。这些音频流被转换成带时间戳的、分段的转录文本，并进行分析，以追踪国家和州层面的关键政治学问题。我们的分析揭示了地方问题与国家趋势之间的互动，提供了关于信息流的见解。结果显示，WavePulse在从互联网获取的广播直播内容捕获和分析方面具有有效性。相关代码和数据集可在 \url{此网址} 获取。 

---
# GeAR: Graph-enhanced Agent for Retrieval-augmented Generation 

**Title (ZH)**: GeAR：图增强代理用于检索增强生成 

**Authors**: Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Piramanayagam, Damien Graux, Dandan Tu, Zeren Jiang, Ruofei Lai, Yang Ren, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18431)  

**Abstract**: Retrieval-augmented generation systems rely on effective document retrieval capabilities. By design, conventional sparse or dense retrievers face challenges in multi-hop retrieval scenarios. In this paper, we present GeAR, which advances RAG performance through two key innovations: (i) graph expansion, which enhances any conventional base retriever, such as BM25, and (ii) an agent framework that incorporates graph expansion. Our evaluation demonstrates GeAR's superior retrieval performance on three multi-hop question answering datasets. Additionally, our system achieves state-of-the-art results with improvements exceeding 10% on the challenging MuSiQue dataset, while requiring fewer tokens and iterations compared to other multi-step retrieval systems. 

**Abstract (ZH)**: 检索增强生成系统依赖于有效的文档检索能力。从设计上看，传统的稀疏或密集检索器在多跳检索场景中面临挑战。本文中，我们提出了一种名为GeAR的新系统，通过两项关键创新来提升RAG（Retrieval-Augmented Generation）的效果：（i）图扩展技术，可以增强任何传统的基线检索器，如BM25；（ii）包含图扩展技术的代理框架。我们的评估结果显示，GeAR在三个多跳问答数据集上表现出优越的检索性能。此外，我们的系统在具有挑战性的MuSiQue数据集上达到了最先进的性能，改进幅度超过10%，所需的token数量和迭代次数也更少，优于其他多步检索系统。 

---
# Bidirectional Topic Matching: Quantifying Thematic Overlap Between Corpora Through Topic Modelling 

**Title (ZH)**: 双向主题匹配：通过主题建模衡量语料库之间主题重叠程度 

**Authors**: Raven Adam, Marie Lisa Kogler  

**Link**: [PDF](https://arxiv.org/pdf/2412.18376)  

**Abstract**: This study introduces Bidirectional Topic Matching (BTM), a novel method for cross-corpus topic modeling that quantifies thematic overlap and divergence between corpora. BTM is a flexible framework that can incorporate various topic modeling approaches, including BERTopic, Top2Vec, and Latent Dirichlet Allocation (LDA). BTM employs a dual-model approach, training separate topic models for each corpus and applying them reciprocally to enable comprehensive cross-corpus comparisons. This methodology facilitates the identification of shared themes and unique topics, providing nuanced insights into thematic relationships. Validation against cosine similarity-based methods demonstrates the robustness of BTM, with strong agreement metrics and distinct advantages in handling outlier topics. A case study on climate news articles showcases BTM's utility, revealing significant thematic overlaps and distinctions between corpora focused on climate change and climate action. BTM's flexibility and precision make it a valuable tool for diverse applications, from political discourse analysis to interdisciplinary studies. By integrating shared and unique topic analyses, BTM offers a comprehensive framework for exploring thematic relationships, with potential extensions to multilingual and dynamic datasets. This work highlights BTM's methodological contributions and its capacity to advance discourse analysis across various domains. 

**Abstract (ZH)**: 本文介绍了双向主题匹配（BTM）方法，这是一种新型的跨语料库主题建模方法，用于定量分析语料库之间的主题重叠与差异。BTM 是一种灵活的框架，可以结合各种主题建模方法，如 BERTopic、Top2Vec 和潜在狄利克雷分配（LDA）。BTM 采用双模型方法，为每个语料库分别训练主题模型，并将它们相互应用于实现全面的跨语料库比较。这一方法有助于识别共享主题和独特话题，提供了对主题关系的细致洞察。与基于余弦相似度的方法相比，BTM 显示出了稳健性，具有强对比度的度量标准，并且在处理异常话题方面具有明显优势。通过对气候新闻文章的案例研究展示了 BTM 的应用价值，揭示了聚焦气候变化和气候行动的语料库之间的显著主题重叠和区别。BTM 的灵活性和精度使其成为多种应用的有效工具，从政治话语分析到跨学科研究。通过综合分析共享和独特的话题，BTM 提供了一个探索主题关系的全面框架，并有可能扩展到多语言和动态数据集。本文突出了 BTM 的方法论贡献及其在各个领域促进话语分析的能力。 

---
# Joint Knowledge Editing for Information Enrichment and Probability Promotion 

**Title (ZH)**: 知识联合编辑以实现信息丰富化和概率提升 

**Authors**: Wenhang Shi, Yiren Chen, Shuqing Bian, Xinyi Zhang, Zhe Zhao, Pengfei Hu, Wei Lu, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2412.17872)  

**Abstract**: Knowledge stored in large language models requires timely updates to reflect the dynamic nature of real-world information. To update the knowledge, most knowledge editing methods focus on the low layers, since recent probes into the knowledge recall process reveal that the answer information is enriched in low layers. However, these probes only and could only reveal critical recall stages for the original answers, while the goal of editing is to rectify model's prediction for the target answers. This inconsistency indicates that both the probe approaches and the associated editing methods are deficient. To mitigate the inconsistency and identify critical editing regions, we propose a contrast-based probe approach, and locate two crucial stages where the model behavior diverges between the original and target answers: Information Enrichment in low layers and Probability Promotion in high layers. Building upon the insights, we develop the Joint knowledge Editing for information Enrichment and probability Promotion (JEEP) method, which jointly edits both the low and high layers to modify the two critical recall stages. Considering the mutual interference and growing forgetting due to dual modifications, JEEP is designed to ensure that updates to distinct regions share the same objectives and are complementary. We rigorously evaluate JEEP by editing up to thousands of facts on various models, i.e., GPT-J (6B) and LLaMA (7B), and addressing diverse editing objectives, i.e., adding factual and counterfactual knowledge. In all tested scenarios, JEEP achieves best performances, validating the effectiveness of the revealings of our probe approach and the designs of our editing method. Our code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型中存储的知识需要及时更新，以反映现实世界信息的动态性质。为了更新知识，大多数知识编辑方法集中于低层，因为最近对知识回忆过程的探查表明，答案信息主要集中在低层。然而，这些探查仅揭示了原始答案的关键回忆阶段，而编辑的目标是纠正模型对目标答案的预测。这种不一致表明，探查方法和相关的编辑方法都存在不足。为了解决这种不一致并识别关键编辑区域，我们提出了一种基于对比的探针方法，并定位了两种关键阶段，即模型行为在原始答案和目标答案之间的分歧：低层信息增强和高层概率促进。基于这些洞察，我们开发了Joint Knowledge Editing for Information Enrichment and Probability Promotion (JEEP) 方法，该方法共同编辑低层和高层，以修改这两种关键回忆阶段。考虑到双重修改导致的相互干扰和遗忘累积增加，JEEP 设计为确保不同区域的更新共享相同的目标并彼此补充。我们通过在各种模型（如 GPT-J (6B) 和 LLaMA (7B)）上编辑多达数千条事实，以及解决各种编辑目标（如添加事实性知识和反事实知识），严格评估了 JEEP。在所有测试的场景中，JEEP 实现了最佳性能，验证了探针方法揭示的有效性和编辑方法设计的有效性。我们的代码和数据可在以下链接获取：[此处提供链接]。 

---
# Leveraging Memory Retrieval to Enhance LLM-based Generative Recommendation 

**Title (ZH)**: 利用记忆检索增强基于大规模语言模型的生成性推荐 

**Authors**: Chengbing Wang, Yang Zhang, Fengbin Zhu, Jizhi Zhang, Tianhao Shi, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17593)  

**Abstract**: Leveraging Large Language Models (LLMs) to harness user-item interaction histories for item generation has emerged as a promising paradigm in generative recommendation. However, the limited context window of LLMs often restricts them to focusing on recent user interactions only, leading to the neglect of long-term interests involved in the longer histories. To address this challenge, we propose a novel Automatic Memory-Retrieval framework (AutoMR), which is capable of storing long-term interests in the memory and extracting relevant information from it for next-item generation within LLMs. Extensive experimental results on two real-world datasets demonstrate the effectiveness of our proposed AutoMR framework in utilizing long-term interests for generative recommendation. 

**Abstract (ZH)**: 利用大型语言模型（LLMs）挖掘用户-项目互动历史以生成项目，在生成型推荐领域被认为是一种有前途的范式。然而，LLMs 较小的上下文窗口限制了它们仅关注最近的用户互动，而忽视了较长历史中涉及的长期兴趣。为了解决这一挑战，我们提出了一种新的自动记忆检索框架（AutoMR），该框架能在记忆中存储长期兴趣，并从记忆中提取相关信息用于后续项目的生成。在两个实际数据集上的广泛实验结果证明了我们所提出的 AutoMR 框架在利用长期兴趣进行生成型推荐方面的有效性。 

---
# CiteBART: Learning to Generate Citations for Local Citation Recommendation 

**Title (ZH)**: CiteBART：学习生成局部引文推荐中的引用 

**Authors**: Ege Yiğit Çelik, Selma Tekir  

**Link**: [PDF](https://arxiv.org/pdf/2412.17534)  

**Abstract**: Citations are essential building blocks in scientific writing. The scientific community is longing for support in their generation. Citation generation involves two complementary subtasks: Determining the citation worthiness of a context and, if it's worth it, proposing the best candidate papers for the citation placeholder. The latter subtask is called local citation recommendation (LCR). This paper proposes CiteBART, a custom BART pre-training based on citation token masking to generate citations to achieve LCR. In the base scheme, we mask the citation token in the local citation context to make the citation prediction. In the global one, we concatenate the citing paper's title and abstract to the local citation context to learn to reconstruct the citation token. CiteBART outperforms state-of-the-art approaches on the citation recommendation benchmarks except for the smallest FullTextPeerRead dataset. The effect is significant in the larger benchmarks, e.g., Refseer and ArXiv. We present a qualitative analysis and an ablation study to provide insights into the workings of CiteBART. Our analyses confirm that its generative nature brings about a zero-shot capability. 

**Abstract (ZH)**: 引文是科学研究写作中的重要构建模块，科学界迫切需要对其进行支持。引文生成涉及两个互补的子任务：确定上下文是否有引文价值，如果有必要，提出最适合的候选论文供引文使用。后者子任务被称为局部引文推荐（Local Citation Recommendation, LCR）。本文提出了一种名为CiteBART的方法，该方法基于引文标记掩蔽的BART预训练，以实现LCR。在基线方案中，我们对局部引文上下文中的引文标记进行掩蔽以预测引文；在全局方案中，我们将引用论文的标题和摘要与局部引文上下文连接起来，以学习重建引文标记。CiteBART在引文推荐基准测试中（除全文本同行评审数据集外）均优于现有方法，尤其在较大的基准测试如Refseer和ArXiv上效果显著。我们进行了定性分析和消融研究，以提供对CiteBART工作原理的见解。我们的分析证实，其生成性特性带来了零样本能力。 

---
# Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark 

**Title (ZH)**: 情景导向推荐：多情景推荐基准测试 

**Authors**: Xiaopeng Li, Jingtong Gao, Pengyue Jia, Yichao Wang, Wanyu Wang, Yejing Wang, Yuhao Wang, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17374)  

**Abstract**: Multi Scenario Recommendation (MSR) tasks, referring to building a unified model to enhance performance across all recommendation scenarios, have recently gained much attention. However, current research in MSR faces two significant challenges that hinder the field's development: the absence of uniform procedures for multi-scenario dataset processing, thus hindering fair comparisons, and most models being closed-sourced, which complicates comparisons with current SOTA models. Consequently, we introduce our benchmark, \textbf{Scenario-Wise Rec}, which comprises 6 public datasets and 12 benchmark models, along with a training and evaluation pipeline. Additionally, we validated the benchmark using an industrial advertising dataset, reinforcing its reliability and applicability in real-world scenarios. We aim for this benchmark to offer researchers valuable insights from prior work, enabling the development of novel models based on our benchmark and thereby fostering a collaborative research ecosystem in MSR. Our source code is also publicly available. 

**Abstract (ZH)**: 以下是将给定内容翻译成中文，并确保符合学术规范后的版本：

多场景推荐（Multi-Scenario Recommendation, MSR）任务是指构建一个统一模型以提高所有推荐场景下的性能，近年来受到广泛关注。然而，当前MSR领域的研究面临两个重要的挑战，这些挑战阻碍了该领域的发展：缺乏统一的多场景数据集处理程序，导致无法进行公平比较；大多数模型为闭源状态，增加了与当前最佳模型（State-of-the-Art, SOTA）进行比较的难度。因此，我们引入了一个基准系统——**情景感知推荐（Scenario-Wise Rec）**，它包括6个公开数据集和12个基准模型，并提供了一组训练和评估管道。此外，我们使用一个工业广告数据集验证了这一基准系统的可靠性和实际应用性。我们的目标是期望这一基准系统能够为研究人员提供宝贵的经验参考，从而基于此基准系统开发新的模型，促进MSR领域的合作研究生态系统。我们的源代码也已公开。 

---
# Efficient fine-tuning methodology of text embedding models for information retrieval: contrastive learning penalty (clp) 

**Title (ZH)**: 高效文本嵌入模型微调方法：对比学习惩罚（CLP） 

**Authors**: Jeongsu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17364)  

**Abstract**: Text embedding models play a crucial role in natural language processing, particularly in information retrieval, and their importance is further highlighted with the recent utilization of RAG (Retrieval- Augmented Generation). This study presents an efficient fine-tuning methodology encompassing data selection, loss function, and model architecture to enhance the information retrieval performance of pre-trained text embedding models. In particular, this study proposes a novel Contrastive Learning Penalty function that overcomes the limitations of existing Contrastive Learning. The proposed methodology achieves significant performance improvements over existing methods in document retrieval tasks. This study is expected to contribute to improving the performance of information retrieval systems through fine-tuning of text embedding models. The code for this study can be found at this https URL, and the best-performing model can be found at this https URL. 

**Abstract (ZH)**: 文本嵌入模型在自然语言处理中扮演着关键角色，尤其是在信息检索方面，随着RAG（检索增强生成）的近期应用，其重要性进一步凸显。本研究提出了一种高效微调方法，涵盖数据选择、损失函数和模型结构，以增强预训练文本嵌入模型的信息检索性能。特别是在此研究中，提出了一种新颖的对比学习惩罚函数，克服了现有对比学习的局限性。所提出的方法在文档检索任务中的性能显著优于现有方法。该研究预计可以通过文本嵌入模型的微调来提高信息检索系统的性能。有关此研究的代码可以在此找到：[此链接](此链接)，性能最佳的模型可以在此找到：[此链接](此链接)。 

---
# Popularity Estimation and New Bundle Generation using Content and Context based Embeddings 

**Title (ZH)**: 基于内容和上下文嵌入的流行度估计与新型捆绑包生成 

**Authors**: Ashutosh Nayak, Prajwal NJ, Sameeksha Keshav, Kavitha S.N., Roja Reddy, Rajasekhara Reddy Duvvuru Muni  

**Link**: [PDF](https://arxiv.org/pdf/2412.17310)  

**Abstract**: Recommender systems create enormous value for businesses and their consumers. They increase revenue for businesses while improving the consumer experience by recommending relevant products amidst huge product base. Product bundling is an exciting development in the field of product recommendations. It aims at generating new bundles and recommending exciting and relevant bundles to their consumers. Unlike traditional recommender systems that recommend single items to consumers, product bundling aims at targeting a bundle, or a set of items, to the consumers. While bundle recommendation has attracted significant research interest recently, extant literature on bundle generation is scarce. Moreover, metrics to identify if a bundle is popular or not is not well studied. In this work, we aim to fulfill this gap by introducing new bundle popularity metrics based on sales, consumer experience and item diversity in a bundle. We use these metrics in the methodology proposed in this paper to generate new bundles for mobile games using content aware and context aware embeddings. We use opensource Steam Games dataset for our analysis. Our experiments indicate that we can generate new bundles that can outperform the existing bundles on the popularity metrics by 32% - 44%. Our experiments are computationally efficient and the proposed methodology is generic that can be extended to other bundling problems e.g. product bundling, music bundling. 

**Abstract (ZH)**: 推荐系统为企业及其消费者创造了巨大的价值。它们通过在庞大的产品库中推荐相关产品，从而增加企业的收入并改善消费者的体验。产品捆绑是产品推荐领域的一项令人兴奋的发展。其目标是生成新的捆绑包，并向消费者推荐令人兴奋且相关的产品捆绑包。与传统的推荐系统仅向消费者推荐单一项目不同，产品捆绑的目标是将一个产品集合或一组项目推荐给消费者。尽管捆绑推荐近年来引起了广泛关注，但现有文献关于捆绑生成的研究较少。此外，用于确定捆绑是否受欢迎的指标也未得到充分研究。在这项工作中，我们旨在通过引入基于销售额、消费者体验和捆绑内项目多样性的新捆绑受欢迎度指标来填补这一空白。我们使用这些指标，结合本文提出的方法，通过内容感知和情境感知嵌入生成新的移动游戏捆绑包。我们的分析基于开源的Steam Games数据集。我们的实验表明，我们可以生成的新捆绑包在受欢迎度指标上的表现可比现有捆绑包高出32%至44%。我们的实验在计算效率上具有优势，并且提出的方法具有通用性，可以应用于其他捆绑问题，例如产品捆绑和音乐捆绑。 

---
# SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval 

**Title (ZH)**: SyNeg: 由大模型驱动的合成否定样本用于密集检索 

**Authors**: Xiaopeng Li, Xiangyang Li, Hao Zhang, Zhaocheng Du, Pengyue Jia, Yichao Wang, Xiangyu Zhao, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17250)  

**Abstract**: The performance of Dense retrieval (DR) is significantly influenced by the quality of negative sampling. Traditional DR methods primarily depend on naive negative sampling techniques or on mining hard negatives through external retriever and meticulously crafted strategies. However, naive negative sampling often fails to adequately capture the accurate boundaries between positive and negative samples, whereas existing hard negative sampling methods are prone to false negatives, resulting in performance degradation and training instability. Recent advancements in large language models (LLMs) offer an innovative solution to these challenges by generating contextually rich and diverse negative samples. In this work, we present a framework that harnesses LLMs to synthesize high-quality hard negative samples. We first devise a \textit{multi-attribute self-reflection prompting strategy} to direct LLMs in hard negative sample generation. Then, we implement a \textit{hybrid sampling strategy} that integrates these synthetic negatives with traditionally retrieved negatives, thereby stabilizing the training process and improving retrieval performance. Extensive experiments on five benchmark datasets demonstrate the efficacy of our approach, and code is also publicly available. 

**Abstract (ZH)**: 密集检索(DR)的性能显著受到负样本质量的影响。传统DR方法主要依赖于简单的负样本采样技术或通过外部检索器和精心设计的方法挖掘难以负样本。然而，简单的负样本采样常常无法充分捕捉正负样本之间的准确界限，而现有的难以负样本采样方法则容易产生假阴性，导致性能下降和训练不稳定。近年来，大型语言模型(LLMs)的进步为解决这些问题提供了新的解决方案，通过生成丰富且多样化的负样本。在本工作中，我们提出了一种利用LLMs生成高质量难以负样本的框架。我们首先设计了一种**多属性自我反思提示策略**，以指导LLMs进行难以负样本的生成。然后，我们实施了一种**混合采样策略**，将这些合成的负样本与传统检索到的负样本结合，从而稳定训练过程并提高检索性能。在五个基准数据集上的大量实验中证明了我们方法的有效性，并且相关代码也已公开发布。 

---
# GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems 

**Title (ZH)**: GraphHash：图聚类在推荐系统中实现参数效率 

**Authors**: Xinyi Wu, Donald Loveland, Runjin Chen, Yozen Liu, Xin Chen, Leonardo Neves, Ali Jadbabaie, Clark Mingxuan Ju, Neil Shah, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.17245)  

**Abstract**: Deep recommender systems rely heavily on large embedding tables to handle high-cardinality categorical features such as user/item identifiers, and face significant memory constraints at scale. To tackle this challenge, hashing techniques are often employed to map multiple entities to the same embedding and thus reduce the size of the embedding tables. Concurrently, graph-based collaborative signals have emerged as powerful tools in recommender systems, yet their potential for optimizing embedding table reduction remains unexplored. This paper introduces GraphHash, the first graph-based approach that leverages modularity-based bipartite graph clustering on user-item interaction graphs to reduce embedding table sizes. We demonstrate that the modularity objective has a theoretical connection to message-passing, which provides a foundation for our method. By employing fast clustering algorithms, GraphHash serves as a computationally efficient proxy for message-passing during preprocessing and a plug-and-play graph-based alternative to traditional ID hashing. Extensive experiments show that GraphHash substantially outperforms diverse hashing baselines on both retrieval and click-through-rate prediction tasks. In particular, GraphHash achieves on average a 101.52% improvement in recall when reducing the embedding table size by more than 75%, highlighting the value of graph-based collaborative information for model reduction. 

**Abstract (ZH)**: 深度推荐系统高度依赖于大型嵌入表来处理高基数的分类特征，如用户/项目标识符，并在大规模应用中面临显著的内存约束。为了解决这一挑战，常常用哈希技术将多个实体映射到同一嵌入，从而减小嵌入表的大小。同时，基于图的协作信号已成为推荐系统中强大的工具，但它们在优化嵌入表缩减方面的潜力尚未被充分探索。本文提出了GraphHash，这是第一个利用基于模块性且基于二分图聚类的方法在用户-项目交互图上缩减嵌入表大小的图基方法。我们证明了模块性目标与消息传递之间存在理论联系，这为我们的方法奠定了基础。通过使用快速聚类算法，GraphHash在预处理期间作为消息传递的计算高效代理，并且是传统ID哈希的即插即用图基替代方案。广泛的实验证明，GraphHash在检索和点击率预测任务中显著优于多种哈希基线。特别是，当嵌入表尺寸减少超过75%时，GraphHash平均实现了101.52%的召回率提升，突显了基于图的协作信息在模型缩减中的价值。 

---
# LLM-based relevance assessment still can't replace human relevance assessment 

**Title (ZH)**: 基于LLM的相关性评估仍无法替代人工相关性评估 

**Authors**: Charles L. A. Clarke, Laura Dietz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17156)  

**Abstract**: The use of large language models (LLMs) for relevance assessment in information retrieval has gained significant attention, with recent studies suggesting that LLM-based judgments provide comparable evaluations to human judgments. Notably, based on TREC 2024 data, Upadhyay et al. make a bold claim that LLM-based relevance assessments, such as those generated by the UMBRELA system, can fully replace traditional human relevance assessments in TREC-style evaluations. This paper critically examines this claim, highlighting practical and theoretical limitations that undermine the validity of this conclusion. First, we question whether the evidence provided by Upadhyay et al. really supports their claim, particularly if a test collection is used asa benchmark for future improvements. Second, through a submission deliberately intended to do so, we demonstrate the ease with which automatic evaluation metrics can be subverted, showing that systems designed to exploit these evaluations can achieve artificially high scores. Theoretical challenges -- such as the inherent narcissism of LLMs, the risk of overfitting to LLM-based metrics, and the potential degradation of future LLM performance -- must be addressed before LLM-based relevance assessments can be considered a viable replacement for human judgments. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在信息检索中的相关性评估应用已经引起广泛关注，最近的研究表明，基于LLM的判断可以提供与人类判断相媲美的评估结果。基于TREC 2024数据，Upadhyay等人提出了一个大胆的论点，即基于LLM的相关性评估，如UMBRELA系统生成的评估，完全可以在TREC风格的评估中取代传统的人类相关性评估。本文对此论点进行了批判性评估，指出了实际和理论上的局限性，这些局限性削弱了这一结论的有效性。首先，我们质疑Upadhyay等人提供的证据是否真正支持他们的论点，尤其是在未来的改进中使用测试集合作为基准的情况下。其次，通过一个特意设计用于此目的的提交，我们证明了自动评估指标可以被轻易破坏，展示了系统如何通过利用这些评估实现虚假高的评分。在大规模语言模型（LLMs）相关性评估被认为是一个可行的人类判断替代方案之前，必须解决诸如LLMs的固有自我中心倾向、过度拟合LLM指标的风险以及未来LLM性能可能降级等理论挑战。 

---
# Iterative NLP Query Refinement for Enhancing Domain-Specific Information Retrieval: A Case Study in Career Services 

**Title (ZH)**: 迭代自然语言处理查询细化以增强领域特定信息检索：职业服务中的案例研究 

**Authors**: Elham Peimani, Gurpreet Singh, Nisarg Mahyavanshi, Aman Arora, Awais Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2412.17075)  

**Abstract**: Retrieving semantically relevant documents in niche domains poses significant challenges for traditional TF-IDF-based systems, often resulting in low similarity scores and suboptimal retrieval performance. This paper addresses these challenges by introducing an iterative and semi-automated query refinement methodology tailored to Humber College's career services webpages. Initially, generic queries related to interview preparation yield low top-document similarities (approximately 0.2--0.3). To enhance retrieval effectiveness, we implement a two-fold approach: first, domain-aware query refinement by incorporating specialized terms such as resources-online-learning, student-online-services, and career-advising; second, the integration of structured educational descriptors like "online resume and interview improvement tools." Additionally, we automate the extraction of domain-specific keywords from top-ranked documents to suggest relevant terms for query expansion. Through experiments conducted on five baseline queries, our semi-automated iterative refinement process elevates the average top similarity score from approximately 0.18 to 0.42, marking a substantial improvement in retrieval performance. The implementation details, including reproducible code and experimental setups, are made available in our GitHub repositories \url{this https URL} and \url{this https URL}. We also discuss the limitations of our approach and propose future directions, including the integration of advanced neural retrieval models. 

**Abstract (ZH)**: 传统基于TF-IDF的系统在处理特定领域的语义相关文档检索时面临显著挑战，通常导致相似度分数较低且检索性能不佳。本文通过引入一种迭代且半自动化的查询细化方法，针对Humber学院职业服务网页进行改进，以应对这些挑战。最初，与面试准备相关的通用查询导致顶级文档相似度较低（大约为0.2-0.3）。为了提高检索效果，我们采用了两步方法：首先，通过引入专业术语（如resources-online-learning、student-online-services和career-advising）来进行领域意识查询细化，其次，集成结构化教育描述词（如在线简历和面试改进工具）。此外，我们自动化地从顶级文档中提取领域特定关键词，以建议查询扩展的相关术语。通过在五个基线查询上进行实验，我们的半自动化迭代细化过程将平均顶级相似度分数从大约0.18提升至0.42，显著提升了检索性能。我们已经在GitHub仓库中提供了实现细节，包括可复现的代码和实验设置，相关链接分别为：\url{此链接}和\url{此链接}。我们还讨论了该方法的局限性，并提出了未来研究方向，包括集成先进的神经检索模型。 

---
# LLM-Powered User Simulator for Recommender System 

**Title (ZH)**: LLM驱动的用户模拟器在推荐系统中的应用 

**Authors**: Zijian Zhang, Shuchang Liu, Ziru Liu, Rui Zhong, Qingpeng Cai, Xiangyu Zhao, Chunxu Zhang, Qidong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16984)  

**Abstract**: User simulators can rapidly generate a large volume of timely user behavior data, providing a testing platform for reinforcement learning-based recommender systems, thus accelerating their iteration and optimization. However, prevalent user simulators generally suffer from significant limitations, including the opacity of user preference modeling and the incapability of evaluating simulation accuracy. In this paper, we introduce an LLM-powered user simulator to simulate user engagement with items in an explicit manner, thereby enhancing the efficiency and effectiveness of reinforcement learning-based recommender systems training. Specifically, we identify the explicit logic of user preferences, leverage LLMs to analyze item characteristics and distill user sentiments, and design a logical model to imitate real human engagement. By integrating a statistical model, we further enhance the reliability of the simulation, proposing an ensemble model that synergizes logical and statistical insights for user interaction simulations. Capitalizing on the extensive knowledge and semantic generation capabilities of LLMs, our user simulator faithfully emulates user behaviors and preferences, yielding high-fidelity training data that enrich the training of recommendation algorithms. We establish quantifying and qualifying experiments on five datasets to validate the simulator's effectiveness and stability across various recommendation scenarios. 

**Abstract (ZH)**: 用户模拟器可以快速生成大量及时的用户行为数据，为基于强化学习的推荐系统提供测试平台，从而加速其迭代和优化。然而，目前广泛使用的用户模拟器通常存在显著的局限性，包括用户偏好建模的不透明性和评估模拟准确性的能力不足。在本文中，我们介绍了一个以LLM（大型语言模型）为驱动的用户模拟器，以显式方式模拟用户与项目之间的互动，从而提高基于强化学习的推荐系统训练的效率和有效性。具体来说，我们识别了用户偏好的显式逻辑，利用LLM分析项目特征和萃取用户情感，并设计了一个逻辑模型来模仿真实的人类互动。通过集成统计模型，我们进一步增强了模拟的可靠性，提出了一种结合逻辑和统计洞察的集成模型，用于用户互动模拟。借助LLM广泛的知识和语义生成能力，我们的用户模拟器能够忠实模拟用户行为和偏好，生成高保真度的训练数据，丰富推荐算法的训练。我们在五个数据集上建立了量化和定性的实验，以验证模拟器在各种推荐场景中的有效性和稳定性。 

---
# Multifaceted User Modeling in Recommendation: A Federated Foundation Models Approach 

**Title (ZH)**: 基于联邦基础模型的多维度用户建模推荐方法 

**Authors**: Chunxu Zhang, Guodong Long, Hongkuan Guo, Zhaojie Liu, Guorui Zhou, Zijian Zhang, Yang Liu, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16969)  

**Abstract**: Multifaceted user modeling aims to uncover fine-grained patterns and learn representations from user data, revealing their diverse interests and characteristics, such as profile, preference, and personality. Recent studies on foundation model-based recommendation have emphasized the Transformer architecture's remarkable ability to capture complex, non-linear user-item interaction relationships. This paper aims to advance foundation model-based recommendersystems by introducing enhancements to multifaceted user modeling capabilities. We propose a novel Transformer layer designed specifically for recommendation, using the self-attention mechanism to capture sequential user-item interaction patterns. Specifically, we design a group gating network to identify user groups, enabling hierarchical discovery across different layers, thereby capturing the multifaceted nature of user interests through multiple Transformer layers. Furthermore, to broaden the data scope and further enhance multifaceted user modeling, we extend the framework to a federated setting, enabling the use of private datasets while ensuring privacy. Experimental validations on benchmark datasets demonstrate the superior performance of our proposed method. Code is available. 

**Abstract (ZH)**: 多维度用户建模旨在从用户数据中发现细粒度模式并学习用户表示，揭示其多样化的兴趣和特征，如个人资料、偏好和性格。基于基础模型的推荐系统研究最近强调了Transformer架构在捕捉复杂非线性用户-项目互动关系方面的能力。本文旨在通过增强多维度用户建模能力来推动基于基础模型的推荐系统的发展。我们提出了一种用于推荐的新颖Transformer层，利用自注意力机制捕捉用户的序列性项目互动模式。具体而言，我们设计了一种组门控网络来识别用户群体，使得多层次的层次发现成为可能，从而通过多层Transformer捕捉用户的复杂兴趣。此外，为了扩大数据范围并进一步增强多维度用户建模，我们将框架扩展到联邦学习设置中，从而能够在保障隐私的前提下使用私有数据集。在基准数据集上的实验验证展示了我们所提出方法的优越性能。代码已开源。 

---
