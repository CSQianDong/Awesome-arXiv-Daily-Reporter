# Contrastive Representation for Interactive Recommendation 

**Title (ZH)**: 交互推荐的对比表示方法 

**Authors**: Jingyu Li, Zhiyong Feng, Dongxiao He, Hongqi Chen, Qinghang Gao, Guoli Wu  

**Link**: [PDF](https://arxiv.org/pdf/2412.18396)  

**Abstract**: Interactive Recommendation (IR) has gained significant attention recently for its capability to quickly capture dynamic interest and optimize both short and long term objectives. IR agents are typically implemented through Deep Reinforcement Learning (DRL), because DRL is inherently compatible with the dynamic nature of IR. However, DRL is currently not perfect for IR. Due to the large action space and sample inefficiency problem, training DRL recommender agents is challenging. The key point is that useful features cannot be extracted as high-quality representations for the recommender agent to optimize its policy. To tackle this problem, we propose Contrastive Representation for Interactive Recommendation (CRIR). CRIR efficiently extracts latent, high-level preference ranking features from explicit interaction, and leverages the features to enhance users' representation. Specifically, the CRIR provides representation through one representation network, and refines it through our proposed Preference Ranking Contrastive Learning (PRCL). The key insight of PRCL is that it can perform contrastive learning without relying on computations involving high-level representations or large potential action sets. Furthermore, we also propose a data exploiting mechanism and an agent training mechanism to better adapt CRIR to the DRL backbone. Extensive experiments have been carried out to show our method's superior improvement on the sample efficiency while training an DRL-based IR agent. 

**Abstract (ZH)**: 交互推荐（IR）最近受到了广泛关注，因为其能够迅速捕捉动态兴趣，并优化短期和长期目标。IR代理通常通过深度强化学习（DRL）实现，因为DRL天生与IR的动态性质相兼容。然而，DRL目前尚未完全适用于IR。由于动作空间庞大和样本效率问题，训练DRL推荐代理极具挑战性。关键问题是，有用的特征无法被提取为高质量的表示，以优化推荐代理的策略。为解决这一问题，我们提出了对比表示的交互推荐（CRIR）。CRIR有效从显式交互中抽取潜在的高度偏好排序特征，并利用这些特征增强用户的表示。具体而言，CRIR通过一个表示网络提供表示，并通过我们提出的偏好排序对比学习（PRCL）进行细化。PRCL的关键洞察是，它可以在不依赖于涉及高级表示或大规模潜在动作集的计算的情况下执行对比学习。此外，我们还提出了一种数据利用机制和代理训练机制，以更好地使CRIR适应DRL框架。我们进行了大量实验，证明了CRIR在训练基于DRL的IR代理时在样本效率方面具有显著的改进优势。 

---
# RaSeRec: Retrieval-Augmented Sequential Recommendation 

**Title (ZH)**: RaSeRec：检索增强的序列推荐 

**Authors**: Xinping Zhao, Baotian Hu, Yan Zhong, Shouzheng Huang, Zihao Zheng, Meng Wang, Haofen Wang, Min zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18378)  

**Abstract**: Although prevailing supervised and self-supervised learning (SSL)-augmented sequential recommendation (SeRec) models have achieved improved performance with powerful neural network architectures, we argue that they still suffer from two limitations: (1) Preference Drift, where models trained on past data can hardly accommodate evolving user preference; and (2) Implicit Memory, where head patterns dominate parametric learning, making it harder to recall long tails. In this work, we explore retrieval augmentation in SeRec, to address these limitations. To this end, we propose a Retrieval-Augmented Sequential Recommendation framework, named RaSeRec, the main idea of which is to maintain a dynamic memory bank to accommodate preference drifts and retrieve relevant memories to augment user modeling explicitly. It consists of two stages: (i) collaborative-based pre-training, which learns to recommend and retrieve; (ii) retrieval-augmented fine-tuning, which learns to leverage retrieved memories. Extensive experiments on three datasets fully demonstrate the superiority and effectiveness of RaSeRec. 

**Abstract (ZH)**: 尽管现有的监督学习和自监督学习（SSL）增强的序列推荐（SeRec）模型通过强大的神经网络架构实现了性能提升，但我们认为它们仍然存在两个局限性：（1）偏好漂移，即在过去的数据上训练的模型难以适应用户偏好的变化；（2）隐式记忆，其中头部模式主导参数学习，使得难以回忆长尾项。在此项工作中，我们探讨了在SeRec中引入检索增强技术，以解决这些局限性。为此，我们提出了一种检索增强序列推荐框架，命名为RaSeRec，其主要思想是维护一个动态记忆库以适应偏好漂移，并检索相关记忆以显式增强用户建模。该框架由两个阶段组成：（i）基于协作的预训练，它学习推荐和检索；（ii）检索增强微调，它学习利用检索到的记忆。在三个数据集上的广泛实验充分证明了RaSeRec的优越性和有效性。 

---
# An Automatic Graph Construction Framework based on Large Language Models for Recommendation 

**Title (ZH)**: 基于大型语言模型的自动图构建推荐框架 

**Authors**: Rong Shan, Jianghao Lin, Chenxu Zhu, Bo Chen, Menghui Zhu, Kangning Zhang, Jieming Zhu, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18241)  

**Abstract**: Graph neural networks (GNNs) have emerged as state-of-the-art methods to learn from graph-structured data for recommendation. However, most existing GNN-based recommendation methods focus on the optimization of model structures and learning strategies based on pre-defined graphs, neglecting the importance of the graph construction stage. Earlier works for graph construction usually rely on speciffic rules or crowdsourcing, which are either too simplistic or too labor-intensive. Recent works start to utilize large language models (LLMs) to automate the graph construction, in view of their abundant open-world knowledge and remarkable reasoning capabilities. Nevertheless, they generally suffer from two limitations: (1) invisibility of global view (e.g., overlooking contextual information) and (2) construction inefficiency. To this end, we introduce AutoGraph, an automatic graph construction framework based on LLMs for recommendation. Specifically, we first use LLMs to infer the user preference and item knowledge, which is encoded as semantic vectors. Next, we employ vector quantization to extract the latent factors from the semantic vectors. The latent factors are then incorporated as extra nodes to link the user/item nodes, resulting in a graph with in-depth global-view semantics. We further design metapath-based message aggregation to effectively aggregate the semantic and collaborative information. The framework is model-agnostic and compatible with different backbone models. Extensive experiments on three real-world datasets demonstrate the efficacy and efffciency of AutoGraph compared to existing baseline methods. We have deployed AutoGraph in Huawei advertising platform, and gain a 2.69% improvement on RPM and a 7.31% improvement on eCPM in the online A/B test. Currently AutoGraph has been used as the main trafffc model, serving hundreds of millions of people. 

**Abstract (ZH)**: 图神经网络（GNNs）已经成了从图结构数据中进行推荐的最先进方法。然而，现有大多数基于GNN的推荐方法集中于模型结构和基于预定义图的学習策略的优化，忽视了图构建阶段的重要性。早期的图构建工作通常依赖于特定规则或众包，这些方法要么过于简单，要么过于耗时。最近的研究开始利用大型语言模型（LLMs）来自动化图的构建，鉴于它们丰富的开放式知识和出色的推理能力。然而，这些方法通常存在两个局限性：（1）全局视图的不可见性（例如，忽略了上下文信息）（2）构建效率低下。为此，我们引入了AutoGraph，这是一种基于LLMs的推荐图自动生成框架。具体来说，我们首先利用LLMs推断用户偏好和项目知识，并将这些知识编码为语义向量。接下来，利用向量量化从语义向量中提取隐因子，并将这些隐因子作为额外节点与用户/项目节点相连，从而使生成的图具有深层次的全局视图语义。我们进一步设计了基于元路径的消息聚合，以有效地整合语义和协同信息。该框架具有模型无关性，并兼容不同的骨干模型。在三个实际数据集上的广泛实验表明，AutoGraph相较于现有基准方法具有更高的有效性和效率。我们已在华为广告平台部署了AutoGraph，并在在线A/B测试中获得了2.69%的RPM提升和7.31%的eCPM提升。目前AutoGraph已成为主要流量模型，服务着数亿用户。 

---
# Efficient Long Context Language Model Retrieval with Compression 

**Title (ZH)**: 高效的长上下文语言模型检索方法及其压缩技术 

**Authors**: Minju Seo, Jinheon Baek, Seongyun Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18232)  

**Abstract**: Long Context Language Models (LCLMs) have emerged as a new paradigm to perform Information Retrieval (IR), which enables the direct ingestion and retrieval of information by processing an entire corpus in their single context, showcasing the potential to surpass traditional sparse and dense retrieval methods. However, processing a large number of passages within in-context for retrieval is computationally expensive, and handling their representations during inference further exacerbates the processing time; thus, we aim to make LCLM retrieval more efficient and potentially more effective with passage compression. Specifically, we propose a new compression approach tailored for LCLM retrieval, which is trained to maximize the retrieval performance while minimizing the length of the compressed passages. To accomplish this, we generate the synthetic data, where compressed passages are automatically created and labeled as chosen or rejected according to their retrieval success for a given query, and we train the proposed Compression model for Long context Retrieval (CoLoR) with this data via preference optimization while adding the length regularization loss on top of it to enforce brevity. Through extensive experiments on 9 datasets, we show that CoLoR improves the retrieval performance by 6% while compressing the in-context size by a factor of 1.91. 

**Abstract (ZH)**: 长上下文语言模型（LCLMs）已经出现并成为信息检索（IR）的一种新范式，使得通过一次性处理整个语料库来直接摄入和检索信息成为可能，展现出了超越传统稀疏检索和密集检索方法的潜在优势。然而，在上下文中处理大量段落进行检索会带来高昂的计算成本，而在此过程中处理其表示将进一步增加处理时间；因此，我们旨在通过段落压缩使LCLM检索更加高效并可能更加有效。具体而言，我们提出了一种针对LCLM检索的新型压缩方法，该方法在最大化检索性能的同时尽量减少压缩段落的长度。为了实现这一目标，我们生成了合成数据，在这些数据中，压缩段落是自动创建并根据给定查询的检索成功情况被标注为选择或拒绝，我们通过偏好优化训练提出的长上下文检索压缩模型（CoLoR），并在其上增加了长度正则化损失以促进简洁性。通过对9个数据集的广泛实验，结果显示，CoLoR在压缩上下文规模1.91倍的同时，检索性能提高了6%。 

---
# Molar: Multimodal LLMs with Collaborative Filtering Alignment for Enhanced Sequential Recommendation 

**Title (ZH)**: Molar：具有协作过滤对齐的多模态语言模型以增强序列推荐 

**Authors**: Yucong Luo, Qitao Qin, Hao Zhang, Mingyue Cheng, Ruiran Yan, Kefan Wang, Jie Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2412.18176)  

**Abstract**: Sequential recommendation (SR) systems have evolved significantly over the past decade, transitioning from traditional collaborative filtering to deep learning approaches and, more recently, to large language models (LLMs). While the adoption of LLMs has driven substantial advancements, these models inherently lack collaborative filtering information, relying primarily on textual content data neglecting other modalities and thus failing to achieve optimal recommendation performance. To address this limitation, we propose Molar, a Multimodal large language sequential recommendation framework that integrates multiple content modalities with ID information to capture collaborative signals effectively. Molar employs an MLLM to generate unified item representations from both textual and non-textual data, facilitating comprehensive multimodal modeling and enriching item embeddings. Additionally, it incorporates collaborative filtering signals through a post-alignment mechanism, which aligns user representations from content-based and ID-based models, ensuring precise personalization and robust performance. By seamlessly combining multimodal content with collaborative filtering insights, Molar captures both user interests and contextual semantics, leading to superior recommendation accuracy. Extensive experiments validate that Molar significantly outperforms traditional and LLM-based baselines, highlighting its strength in utilizing multimodal data and collaborative signals for sequential recommendation tasks. The source code is available at this https URL. 

**Abstract (ZH)**: 在过去的十年里，序列推荐（SR）系统经历了显著的发展，从传统的协同过滤方法过渡到深度学习方法，最近则转向了大型语言模型（LLMs）。尽管LLMs的应用推动了显著的进步，但这些模型本身缺乏协同过滤的信息，主要依赖于文本内容数据，忽视了其他模态的数据，因而未能实现最佳的推荐性能。为了解决这一限制，我们提出了Molar——一个多模态大型语言序列推荐框架，该框架整合了多种内容模态和ID信息，以有效捕捉协同信号。Molar采用多模态大型语言模型（MLLM）从文本和非文本数据中生成统一的项目表示，促进全面的多模态建模并丰富项目嵌入。此外，Molar通过后对齐机制整合了协同过滤信号，这种机制对基于内容和基于ID的用户表示进行对齐，以确保精确的个性化和稳健的性能。通过无缝结合多模态内容与协同过滤洞察，Molar能够同时捕获用户兴趣和上下文语义，从而提高推荐精度。大量实验验证了Molar在传统方法和LLM基线模型中显著优越的表现，突显了其利用多模态数据和协同信号进行序列推荐任务的优势。相关源代码可在以下链接获取：[请提供具体链接]。 

---
# Unlocking the Hidden Treasures: Enhancing Recommendations with Unlabeled Data 

**Title (ZH)**: 解锁隐藏的宝藏：利用未标label数据提升推荐系统性能 

**Authors**: Yuhan Zhao, Rui Chen, Qilong Han, Hongtao Song, Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.18170)  

**Abstract**: Collaborative filtering (CF) stands as a cornerstone in recommender systems, yet effectively leveraging the massive unlabeled data presents a significant challenge. Current research focuses on addressing the challenge of unlabeled data by extracting a subset that closely approximates negative samples. Regrettably, the remaining data are overlooked, failing to fully integrate this valuable information into the construction of user preferences. To address this gap, we introduce a novel positive-neutral-negative (PNN) learning paradigm. PNN introduces a neutral class, encompassing intricate items that are challenging to categorize directly as positive or negative samples. By training a model based on this triple-wise partial ranking, PNN offers a promising solution to learning complex user preferences. Through theoretical analysis, we connect PNN to one-way partial AUC (OPAUC) to validate its efficacy. Implementing the PNN paradigm is, however, technically challenging because: (1) it is difficult to classify unlabeled data into neutral or negative in the absence of supervised signals; (2) there does not exist any loss function that can handle set-level triple-wise ranking relationships. To address these challenges, we propose a semi-supervised learning method coupled with a user-aware attention model for knowledge acquisition and classification refinement. Additionally, a novel loss function with a two-step centroid ranking approach enables handling set-level rankings. Extensive experiments on four real-world datasets demonstrate that, when combined with PNN, a wide range of representative CF models can consistently and significantly boost their performance. Even with a simple matrix factorization, PNN can achieve comparable performance to sophisticated graph neutral networks. 

**Abstract (ZH)**: 协同过滤（CF）是推荐系统中的基石，然而有效地利用大量的未标注数据是一个显著的挑战。当前的研究集中在通过提取一个子集来近似负样本来应对未标注数据的挑战，但遗憾的是，其余数据未被重视，未能充分整合这些有价值的信息以构建用户偏好。为了解决这一问题，我们提出了一种新的正样本-中性样本-负样本（Positive-Neutral-Negative，PNN）学习范式。PNN引入了一个中性类，包括难以直接归类为正样本或负样本的复杂项目。通过基于这种部分三元排序训练模型，PNN提供了一种学习复杂用户偏好的有前景的解决方案。通过理论分析，我们将PNN与单向部分AUC（One-Way Partial AUC，OPAUC）连接起来，以验证其有效性。然而，实现PNN范式在技术上具有挑战性，因为：（1）在缺乏监督信号的情况下，难以将未标注数据分类为中性或负样本；（2）不存在能够处理集合级别三元排序关系的损失函数。为解决这些问题，我们提出了一种半监督学习方法，并结合用户感知注意模型进行知识获取和分类精炼。此外，我们提出了一种新型损失函数，采用两步聚类排序方法来处理集合级别的排名关系。在四个真实世界的数据集上的广泛实验表明，当与PNN结合使用时，多种代表性CF模型可以一致且显著地提高性能。即使使用简单的矩阵分解方法，PNN也能达到与复杂图形中性网络相当的性能。 

---
# From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking 

**Title (ZH)**: 从成对分析到排名：通过伪排名攀登理想的协作过滤阶梯 

**Authors**: Yuhan Zhao, Rui Chen, Li Chen, Shuang Zhang, Qilong Han, Hongtao Song  

**Link**: [PDF](https://arxiv.org/pdf/2412.18168)  

**Abstract**: Intuitively, an ideal collaborative filtering (CF) model should learn from users' full rankings over all items to make optimal top-K recommendations. Due to the absence of such full rankings in practice, most CF models rely on pairwise loss functions to approximate full rankings, resulting in an immense performance gap. In this paper, we provide a novel analysis using the multiple ordinal classification concept to reveal the inevitable gap between a pairwise approximation and the ideal case. However, bridging the gap in practice encounters two formidable challenges: (1) none of the real-world datasets contains full ranking information; (2) there does not exist a loss function that is capable of consuming ranking information. To overcome these challenges, we propose a pseudo-ranking paradigm (PRP) that addresses the lack of ranking information by introducing pseudo-rankings supervised by an original noise injection mechanism. Additionally, we put forward a new ranking loss function designed to handle ranking information effectively. To ensure our method's robustness against potential inaccuracies in pseudo-rankings, we equip the ranking loss function with a gradient-based confidence mechanism to detect and mitigate abnormal gradients. Extensive experiments on four real-world datasets demonstrate that PRP significantly outperforms state-of-the-art methods. 

**Abstract (ZH)**: 直观上，一个理想的协同过滤（CF）模型应该从用户对所有项目的完整排名中学习，以生成最佳的Top-K推荐。由于实践中难以获取这种完整的排名信息，大多数CF模型依赖于对偶损失函数来近似完整的排名，导致了巨大的性能差距。本文通过使用多重序数分类的概念提供了一种新的分析方法，揭示了对偶近似与理想情况之间的不可避免差距。然而，在实践中弥合这一差距面临两大挑战：（1）几乎所有现实世界的数据集都不包含完整的排名信息；（2）不存在能够消耗排名信息的损失函数。为克服这些挑战，我们提出了一种伪排名 paradigm（PRP），通过引入由原始噪声注入机制监督的伪排名来解决排名信息的缺乏问题。此外，我们还提出了一种新的排名损失函数，以有效处理排名信息。为了确保方法在伪排名可能存在的潜在不准确性方面的鲁棒性，我们在损失函数中加入了基于梯度的置信机制以检测并缓解异常梯度。在四个真实世界数据集上的广泛实验表明，PRP方法显著优于现有最先进的方法。 

---
# BRIDGE: Bundle Recommendation via Instruction-Driven Generation 

**Title (ZH)**: BRIDGE：基于指令驱动生成的捆绑推荐系统 

**Authors**: Tuan-Nghia Bui, Huy-Son Nguyen, Cam-Van Nguyen Thi, Hoang-Quynh Le, Duc-Trong Le  

**Link**: [PDF](https://arxiv.org/pdf/2412.18092)  

**Abstract**: Bundle recommendation aims to suggest a set of interconnected items to users. However, diverse interaction types and sparse interaction matrices often pose challenges for previous approaches in accurately predicting user-bundle adoptions. Inspired by the distant supervision strategy and generative paradigm, we propose BRIDGE, a novel framework for bundle recommendation. It consists of two main components namely the correlation-based item clustering and the pseudo bundle generation modules. Inspired by the distant supervision approach, the former is to generate more auxiliary information, e.g., instructive item clusters, for training without using external data. This information is subsequently aggregated with collaborative signals from user historical interactions to create pseudo `ideal' bundles. This capability allows BRIDGE to explore all aspects of bundles, rather than being limited to existing real-world bundles. It effectively bridging the gap between user imagination and predefined bundles, hence improving the bundle recommendation performance. Experimental results validate the superiority of our models over state-of-the-art ranking-based methods across five benchmark datasets. 

**Abstract (ZH)**: 捆绑推荐旨在向用户推荐一系列相互连接的项目。然而，多样的交互类型和稀疏的交互矩阵常常为之前的研究方法准确预测用户对捆绑组合的采用带来了挑战。受远程监督策略和生成范式的启发，我们提出了一种名为BRIDGE的新框架，用于捆绑推荐。该框架包含两个主要组件：基于相关性的物品聚类模块和伪捆绑生成模块。受远程监督方法的启发，前者的目的是生成更多的辅助信息，例如有指导意义的物品聚类，用于训练而无需使用外部数据。这些信息随后与用户历史交互中的协同信号相结合，以创建伪“理想”捆绑。这一能力使BRIDGE能够探索所有捆绑方面的表现，而不是局限于现有的现实世界捆绑。它有效地弥合了用户的想象与预定义捆绑之间的差距，从而提高了捆绑推荐的效果。实验结果验证了与五个基准数据集上的最新排序方法相比，我们的模型具有优越性。 

---
# Prompt Tuning for Item Cold-start Recommendation 

**Title (ZH)**: 物品冷启动推荐的提示调整 

**Authors**: Yuezihan Jiang, Gaode Chen, Wenhan Zhang, Jingchi Wang, Yinjie Jiang, Qi Zhang, Jingjian Lin, Peng Jiang, Kaigui Bian  

**Link**: [PDF](https://arxiv.org/pdf/2412.18082)  

**Abstract**: The item cold-start problem is crucial for online recommender systems, as the success of the cold-start phase determines whether items can transition into popular ones. Prompt learning, a powerful technique used in natural language processing (NLP) to address zero- or few-shot problems, has been adapted for recommender systems to tackle similar challenges. However, existing methods typically rely on content-based properties or text descriptions for prompting, which we argue may be suboptimal for cold-start recommendations due to 1) semantic gaps with recommender tasks, 2) model bias caused by warm-up items contribute most of the positive feedback to the model, which is the core of the cold-start problem that hinders the recommender quality on cold-start items. We propose to leverage high-value positive feedback, termed pinnacle feedback as prompt information, to simultaneously resolve the above two problems. We experimentally prove that compared to the content description proposed in existing works, the positive feedback is more suitable to serve as prompt information by bridging the semantic gaps. Besides, we propose item-wise personalized prompt networks to encode pinnaclce feedback to relieve the model bias by the positive feedback dominance problem. Extensive experiments on four real-world datasets demonstrate the superiority of our model over state-of-the-art methods. Moreover, PROMO has been successfully deployed on a popular short-video sharing platform, a billion-user scale commercial short-video application, achieving remarkable performance gains across various commercial metrics within cold-start scenarios 

**Abstract (ZH)**: 冷启动问题是在线推荐系统中的关键问题，因为冷启动阶段的成功与否决定了物品能否过渡为受欢迎的物品。提示学习是一种在自然语言处理（NLP）中用于解决零样本或少样本问题的强大技术，已经适应于推荐系统以应对类似的挑战。然而，现有的方法通常依赖于基于内容的属性或文本描述作为提示，我们认为这对冷启动推荐来说可能不太理想。这主要是由于以下两个理由：1）语义差距与推荐任务之间存在偏差，2）模型偏差导致预热物品提供的正反馈占主要部分，这是冷启动问题的核心，阻碍了推荐系统的质量提升。我们提出利用高价值的正反馈（称为顶点反馈）作为提示信息，同时解决上述两个问题。实验结果证明，相比于现有工作中提出的基于内容的描述，正反馈更加适合作为提示信息，能够更好地弥补语义差距。此外，我们提出了基于物品个性化提示网络来编码顶点反馈，以缓解正反馈主导问题所带来的模型偏差。在四个实际数据集上的广泛实验表明，与最先进的方法相比，我们的模型具有明显的优势。此外，PROMO已在一款广泛用户群体（亿级用户）的流行短视频共享平台上成功部署，并在冷启动场景下实现了各种商业指标上的显著性能提升。 

---
# Time-Probability Dependent Knowledge Extraction in IoT-enabled Smart Building 

**Title (ZH)**: 物联网赋能智能建筑中的时间-概率依赖知识提取 

**Authors**: Hangli Ge, Hirotsugu Seike, Noboru Koshizuka  

**Link**: [PDF](https://arxiv.org/pdf/2412.18042)  

**Abstract**: Smart buildings incorporate various emerging Internet of Things (IoT) applications for comprehensive management of energy efficiency, human comfort, automation, and security. However, the development of a knowledge extraction framework is fundamental. Currently, there is a lack of a unified and practical framework for modeling heterogeneous sensor data within buildings. In this paper, we propose a practical inference framework for extracting status-to-event knowledge within smart building. Our proposal includes IoT-based API integration, ontology model design, and time probability dependent knowledge extraction methods. The Building Topology Ontology (BOT) was leveraged to construct spatial relations among sensors and spaces within the building. We utilized Apache Jena Fuseki's SPARQL server for storing and querying the RDF triple data. Two types of knowledge could be extracted: timestamp-based probability for abnormal event detection and time interval-based probability for conjunction of multiple events. We conducted experiments (over a 78-day period) in a real smart building environment. The data of light and elevator states has been collected for evaluation. The evaluation revealed several inferred events, such as room occupancy, elevator trajectory tracking, and the conjunction of both events. The numerical values of detected event counts and probability demonstrate the potential for automatic control in the smart building. 

**Abstract (ZH)**: 智能建筑整合了各种新兴的物联网（IoT）应用，以实现能源效率、人体舒适度、自动化和安全性的全面管理。然而，构建知识提取框架是基础性的。目前，缺乏一种统一且实用的方法来建模建筑物内的异构传感器数据。本文提出了一种实用的推理框架，用于在智能建筑中提取状态到事件的知识。我们的提案包括基于物联网的API集成、本体模型设计以及时间概率依赖的知识提取方法。我们利用Building Topology Ontology (BOT)构建了建筑物内部传感器和空间之间的空间关系。我们使用Apache Jena Fuseki的SPARQL服务器来存储和查询RDF三元组数据。可以提取两种类型的知识：基于时间戳的概率用于异常事件检测，时间间隔概率用于多个事件的联合。我们在一个实际的智能建筑环境中进行了实验（持续78天），收集了灯光和电梯状态的数据以进行评估。评估结果显示了诸如房间占用率、电梯轨迹跟踪以及这两者联合事件的推断情况。检测到的事件数量和概率的数值表明智能建筑中自动控制的潜在可能性。 

---
# WavePulse: Real-time Content Analytics of Radio Livestreams 

**Title (ZH)**: WavePulse：实时广播现场内容分析 

**Authors**: Govind Mittal, Sarthak Gupta, Shruti Wagle, Chirag Chopra, Anthony J DeMattee, Nasir Memon, Mustaque Ahamad, Chinmay Hegde  

**Link**: [PDF](https://arxiv.org/pdf/2412.17998)  

**Abstract**: Radio remains a pervasive medium for mass information dissemination, with AM/FM stations reaching more Americans than either smartphone-based social networking or live television. Increasingly, radio broadcasts are also streamed online and accessed over the Internet. We present WavePulse, a framework that records, documents, and analyzes radio content in real-time. While our framework is generally applicable, we showcase the efficacy of WavePulse in a collaborative project with a team of political scientists focusing on the 2024 Presidential Elections. We use WavePulse to monitor livestreams of 396 news radio stations over a period of three months, processing close to 500,000 hours of audio streams. These streams were converted into time-stamped, diarized transcripts and analyzed to track answer key political science questions at both the national and state levels. Our analysis revealed how local issues interacted with national trends, providing insights into information flow. Our results demonstrate WavePulse's efficacy in capturing and analyzing content from radio livestreams sourced from the Web. Code and dataset can be accessed at \url{this https URL}. 

**Abstract (ZH)**: 无线电仍然是大规模信息传播的一个普遍媒介，AM/FM广播电台触及的美国民众数量超过了基于智能手机的社交网络或直播电视。越来越多地，广播内容也被流媒体化并可通过互联网访问。我们提出了WavePulse框架，该框架能够实时记录、整理和分析广播内容。虽然我们的框架具有广泛的适用性，但我们在此呈现了WavePulse在一项与政治科学家团队合作的项目中的效果，该项目关注2024年的美国总统选举。我们使用WavePulse监测了为期三个月内396家新闻广播电台的实时流媒体，处理了近50万小时的音频流。这些音频流被转换为带时间戳的、分一时段的转录文本，并进行了分析，以追踪国家和州层面的关键政治学问题。我们的分析揭示了地方问题与国家趋势之间的相互作用，提供了关于信息流动的见解。我们的结果证明了WavePulse在抓取和分析来自网络的广播实时流内容方面的有效性。相关代码和数据集可在此链接访问：\url{this https URL}。 

---
# GeAR: Graph-enhanced Agent for Retrieval-augmented Generation 

**Title (ZH)**: GeAR：基于图的代理增强检索增强生成

这个翻译符合学术规范，同时保持了原文的意思和结构。在这里，“Graph-enhanced”被翻译为“基于图的”，“agent”翻译为“代理”，“retrieval-augmented generation”翻译为“检索增强生成”，以确保术语的专业性和准确性。 

**Authors**: Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Piramanayagam, Damien Graux, Dandan Tu, Zeren Jiang, Ruofei Lai, Yang Ren, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18431)  

**Abstract**: Retrieval-augmented generation systems rely on effective document retrieval capabilities. By design, conventional sparse or dense retrievers face challenges in multi-hop retrieval scenarios. In this paper, we present GeAR, which advances RAG performance through two key innovations: (i) graph expansion, which enhances any conventional base retriever, such as BM25, and (ii) an agent framework that incorporates graph expansion. Our evaluation demonstrates GeAR's superior retrieval performance on three multi-hop question answering datasets. Additionally, our system achieves state-of-the-art results with improvements exceeding 10% on the challenging MuSiQue dataset, while requiring fewer tokens and iterations compared to other multi-step retrieval systems. 

**Abstract (ZH)**: 检索增强生成系统依赖于有效的文档检索能力。从设计上讲，传统的稀疏或密集检索器在多跳检索场景中面临挑战。本文中，我们提出了GeAR，通过两项关键创新来提升RAG（检索增强生成）的表现：(i) 图扩展，该方法可以增强任何传统的基线检索器，例如BM25；(ii) 一个代理框架，该框架结合了图扩展。我们的评估结果显示，GeAR在三个多跳问答数据集上的检索性能优于其他方法。此外，在具有挑战性的MuSiQue数据集上，我们的系统取得了当前最佳结果，相比其他多步检索系统，所需token数量和迭代次数更少，性能提升超过10%。 

---
# Bidirectional Topic Matching: Quantifying Thematic Overlap Between Corpora Through Topic Modelling 

**Title (ZH)**: 双向主题匹配：通过主题建模量化语料库之间的主题重叠 

**Authors**: Raven Adam, Marie Lisa Kogler  

**Link**: [PDF](https://arxiv.org/pdf/2412.18376)  

**Abstract**: This study introduces Bidirectional Topic Matching (BTM), a novel method for cross-corpus topic modeling that quantifies thematic overlap and divergence between corpora. BTM is a flexible framework that can incorporate various topic modeling approaches, including BERTopic, Top2Vec, and Latent Dirichlet Allocation (LDA). BTM employs a dual-model approach, training separate topic models for each corpus and applying them reciprocally to enable comprehensive cross-corpus comparisons. This methodology facilitates the identification of shared themes and unique topics, providing nuanced insights into thematic relationships. Validation against cosine similarity-based methods demonstrates the robustness of BTM, with strong agreement metrics and distinct advantages in handling outlier topics. A case study on climate news articles showcases BTM's utility, revealing significant thematic overlaps and distinctions between corpora focused on climate change and climate action. BTM's flexibility and precision make it a valuable tool for diverse applications, from political discourse analysis to interdisciplinary studies. By integrating shared and unique topic analyses, BTM offers a comprehensive framework for exploring thematic relationships, with potential extensions to multilingual and dynamic datasets. This work highlights BTM's methodological contributions and its capacity to advance discourse analysis across various domains. 

**Abstract (ZH)**: 本文介绍了双向主题匹配（BTM）方法，这是一种新的跨语料库主题建模方法，用于量化不同语料库之间的主题重叠和差异。BTM 是一个灵活的框架，可以结合各种主题建模方法，包括 BERTopic、Top2Vec 和隐狄利克雷分配（LDA）。BTM 采用双模型方法，分别为每个语料库训练独立的主题模型，并通过相互应用进行交叉比较，从而实现全面的跨语料库对比。这种方法有助于识别共享主题和独特的主题，提供了关于主题关系的细致洞察。与余弦相似度方法相比，BTM 的验证结果显示其具有较高的稳健性和处理异常主题的优势。在气候新闻文章的案例研究中展示了 BTM 的实用性，揭示了关注气候变化和气候行动的不同语料库之间显著的主题重叠和区别。BTM 的灵活性和精确性使其成为政治话语分析和跨学科研究等多种应用的宝贵工具。通过结合共享主题和独特主题的分析，BTM 提供了一个全面的主题关系探索框架，且有可能扩展到多语言和动态数据集。本文突出了 BTM 的方法论贡献及其在各个领域的话语分析中推进研究的能力。 

---
# Joint Knowledge Editing for Information Enrichment and Probability Promotion 

**Title (ZH)**: 联合知识编辑以实现信息丰富和概率提升 

**Authors**: Wenhang Shi, Yiren Chen, Shuqing Bian, Xinyi Zhang, Zhe Zhao, Pengfei Hu, Wei Lu, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2412.17872)  

**Abstract**: Knowledge stored in large language models requires timely updates to reflect the dynamic nature of real-world information. To update the knowledge, most knowledge editing methods focus on the low layers, since recent probes into the knowledge recall process reveal that the answer information is enriched in low layers. However, these probes only and could only reveal critical recall stages for the original answers, while the goal of editing is to rectify model's prediction for the target answers. This inconsistency indicates that both the probe approaches and the associated editing methods are deficient. To mitigate the inconsistency and identify critical editing regions, we propose a contrast-based probe approach, and locate two crucial stages where the model behavior diverges between the original and target answers: Information Enrichment in low layers and Probability Promotion in high layers. Building upon the insights, we develop the Joint knowledge Editing for information Enrichment and probability Promotion (JEEP) method, which jointly edits both the low and high layers to modify the two critical recall stages. Considering the mutual interference and growing forgetting due to dual modifications, JEEP is designed to ensure that updates to distinct regions share the same objectives and are complementary. We rigorously evaluate JEEP by editing up to thousands of facts on various models, i.e., GPT-J (6B) and LLaMA (7B), and addressing diverse editing objectives, i.e., adding factual and counterfactual knowledge. In all tested scenarios, JEEP achieves best performances, validating the effectiveness of the revealings of our probe approach and the designs of our editing method. Our code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型中存储的知识需要及时更新，以反映现实世界信息的动态性。为了更新知识，大多数知识编辑方法主要集中在低层，因为近期对知识检索过程的探查显示答案信息较多存在于低层。然而，这些探查只能揭示出原始答案的关键检索阶段，而编辑的目标是纠正模型对目标答案的预测。这种不一致性表明，目前的探查方法和相应的编辑方法都存在缺陷。为缓解这一不一致性并识别关键的编辑区域，我们提出了一种对比探查方法，并确定了两个关键阶段，模型在这些阶段的行为在原始答案和目标答案之间存在差异：低层中的信息丰富化和高层中的概率增强。基于这些洞见，我们开发了一种联合信息丰富化和概率增强的知识编辑方法（Joint knowledge Editing for information Enrichment and probability Promotion，简称JEEP），该方法同时编辑低层和高层，以修改这两种关键检索阶段。考虑到双重修改导致的相互干扰和不断增长的遗忘，JEEP设计确保不同区域的更新共享相同的目标并相互补充。我们通过在各种模型（如GPT-J（6B）和LLaMA（7B））上编辑多达数千个事实，并解决多种编辑目标（如增加事实性知识和反事实知识），严格评估了JEEP。在所有测试场景中，JEEP均表现最好，验证了我们探查方法揭示的有效性和我们编辑方法的设计。我们的代码和数据可在以下链接获取：[此链接](https://github.com/your-repo-url)。 

---
