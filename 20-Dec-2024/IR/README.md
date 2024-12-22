# Nano-ESG: Extracting Corporate Sustainability Information from News Articles 

**Title (ZH)**: Nano-ESG：从新闻文章中提取企业可持续性信息 

**Authors**: Fabian Billert, Stefan Conrad  

**Link**: [PDF](https://arxiv.org/pdf/2412.15093)  

**Abstract**: Determining the sustainability impact of companies is a highly complex subject which has garnered more and more attention over the past few years. Today, investors largely rely on sustainability-ratings from established rating-providers in order to analyze how responsibly a company acts. However, those ratings have recently been criticized for being hard to understand and nearly impossible to reproduce.
An independent way to find out about the sustainability practices of companies lies in the rich landscape of news article data. In this paper, we explore a different approach to identify key opportunities and challenges of companies in the sustainability domain. We present a novel dataset of more than 840,000 news articles which were gathered for major German companies between January 2023 and September 2024. By applying a mixture of Natural Language Processing techniques, we first identify relevant articles, before summarizing them and extracting their sustainability-related sentiment and aspect using Large Language Models (LLMs). Furthermore, we conduct an evaluation of the obtained data and determine that the LLM-produced answers are accurate. We release both datasets at this https URL. 

**Abstract (ZH)**: 确定企业的可持续性影响是一个高度复杂的主题，近年来引起了越来越多的关注。如今，投资者主要依赖于评级机构提供的可持续性评级，以分析公司在多大程度上负责任地行动。然而，这些评级最近因难以理解且几乎不可能重现而受到批评。

在企业的可持续性实践方面，新闻文章数据丰富的领域提供了一种独立的方法。本文探讨了一种不同的方法，以识别可持续性领域中公司面临的关键机会和挑战。我们呈现了一个包含超过840,000篇新闻文章的新数据集，这些文章是为2023年1月至2024年9月期间的主要德国公司收集的。通过结合自然语言处理技术，我们首先识别出相关文章，然后对其进行总结，并使用大型语言模型（LLM）提取其可持续性相关的观点和方面。此外，我们对获得的数据进行了评估，并确定LLM生成的答案是准确的。我们在此处发布两个数据集：[请插入网址]。 

---
# DisCo: Graph-Based Disentangled Contrastive Learning for Cold-Start Cross-Domain Recommendation 

**Title (ZH)**: DisCo：基于图的解纠缠对比学习在冷启动跨域推荐中的应用 

**Authors**: Hourun Li, Yifan Wang, Zhiping Xiao, Jia Yang, Changling Zhou, Ming Zhang, Wei Ju  

**Link**: [PDF](https://arxiv.org/pdf/2412.15005)  

**Abstract**: Recommender systems are widely used in various real-world applications, but they often encounter the persistent challenge of the user cold-start problem. Cross-domain recommendation (CDR), which leverages user interactions from one domain to improve prediction performance in another, has emerged as a promising solution. However, users with similar preferences in the source domain may exhibit different interests in the target domain. Therefore, directly transferring embeddings may introduce irrelevant source-domain collaborative information. In this paper, we propose a novel graph-based disentangled contrastive learning framework to capture fine-grained user intent and filter out irrelevant collaborative information, thereby avoiding negative transfer. Specifically, for each domain, we use a multi-channel graph encoder to capture diverse user intents. We then construct the affinity graph in the embedding space and perform multi-step random walks to capture high-order user similarity relationships. Treating one domain as the target, we propose a disentangled intent-wise contrastive learning approach, guided by user similarity, to refine the bridging of user intents across domains. Extensive experiments on four benchmark CDR datasets demonstrate that DisCo consistently outperforms existing state-of-the-art baselines, thereby validating the effectiveness of both DisCo and its components. 

**Abstract (ZH)**: 推荐系统被广泛应用于各种实际应用场景中，但它们常常面临用户冷启动问题这一持续性的挑战。跨领域推荐（CDR），通过利用一个领域的用户交互来改进另一个领域的预测性能，已经成为了颇有前景的解决方案。然而，在源领域中具有相似偏好的用户在目标领域中可能会表现出不同的兴趣。因此，直接转移嵌入向量可能会引入与目标领域无关的源领域合作信息。在本文中，我们提出了一种新的基于图的去纠缠对比学习框架，以捕捉细微的用户意图并过滤掉无关的合作信息，从而避免负迁移。具体而言，对于每个领域，我们使用多通道图编码器来捕捉多种用户意图。然后在嵌入空间中构建亲和图，并通过多步随机游走来捕捉高级的用户相似关系。将一个领域视为目标领域，我们提出了一种根据用户相似性指导的去纠缠意图级对比学习方法，以细化跨领域的用户意图连接。在四个基准CDR数据集上的广泛实验表明，DisCo始终优于现有的最先进的基线方法，从而验证了DisCo及其组件的有效性。 

---
# Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation 

**Title (ZH)**: 基于频谱的模态表示融合图卷积网络在多模态推荐中的应用 

**Authors**: Rongqing Kenneth Ong, Andy W. H. Khong  

**Link**: [PDF](https://arxiv.org/pdf/2412.14978)  

**Abstract**: Incorporating multi-modal features as side information has recently become a trend in recommender systems. To elucidate user-item preferences, recent studies focus on fusing modalities via concatenation, element-wise sum, or attention mechanisms. Despite having notable success, existing approaches do not account for the modality-specific noise encapsulated within each modality. As a result, direct fusion of modalities will lead to the amplification of cross-modality noise. Moreover, the variation of noise that is unique within each modality results in noise alleviation and fusion being more challenging. In this work, we propose a new Spectrum-based Modality Representation (SMORE) fusion graph recommender that aims to capture both uni-modal and fusion preferences while simultaneously suppressing modality noise. Specifically, SMORE projects the multi-modal features into the frequency domain and leverages the spectral space for fusion. To reduce dynamic contamination that is unique to each modality, we introduce a filter to attenuate and suppress the modality noise adaptively while capturing the universal modality patterns effectively. Furthermore, we explore the item latent structures by designing a new multi-modal graph learning module to capture associative semantic correlations and universal fusion patterns among similar items. Finally, we formulate a new modality-aware preference module, which infuses behavioral features and balances the uni- and multi-modal features for precise preference modeling. This empowers SMORE with the ability to infer both user modality-specific and fusion preferences more accurately. Experiments on three real-world datasets show the efficacy of our proposed model. The source code for this work has been made publicly available at this https URL. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

近年来，将多模态特征作为辅助信息已在推荐系统中成为一种趋势。为了阐明用户-项目偏好，最近的研究主要关注通过串联、元素级求和或注意力机制来融合模态。尽管这些方法取得了显著的成功，但现有方法并未考虑封装在各模态内的模态特定噪声。因此，直接融合模态会导致跨模态噪声的放大。此外，每种模态中独特的噪声变异使得噪声缓解和融合更为复杂。在本工作中，我们提出了一个名为基于频谱的模态表示（SMORE）融合图推荐系统，旨在同时捕捉单模态和融合偏好，同时抑制模态噪声。具体而言，SMORE 将多模态特征投影到频域，并利用频谱空间进行融合。为了减少每种模态特有的动态污染，我们引入了一种滤波器，以便适应性地衰减和抑制模态噪声，同时有效捕捉通用模态模式。此外，我们通过设计一个新型的多模态图学习模块来探索项目的潜在结构，以捕捉相似项目间的关联语义关联和通用融合模式。最后，我们提出了一个模态感知偏好模块，该模块融合行为特征并平衡单模态和多模态特征，以实现精确的偏好建模。这赋予了SMORE更准确地推断用户模态特定偏好和融合偏好的能力。在三个真实世界的数据集上的实验表明，我们提出的模型具有有效性。此工作的源代码已在以下网址公开：[相关链接]。 

---
# ECLIPSE: Contrastive Dimension Importance Estimation with Pseudo-Irrelevance Feedback for Dense Retrieval 

**Title (ZH)**: ECLIPSE：基于伪不可关联反馈的对比维数重要性估计dense检索方法 

**Authors**: Giulio D'Erasmo, Giovanni Trappolini, Nicola Tonellotto, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2412.14967)  

**Abstract**: Recent advances in Information Retrieval have leveraged high-dimensional embedding spaces to improve the retrieval of relevant documents. Moreover, the Manifold Clustering Hypothesis suggests that despite these high-dimensional representations, documents relevant to a query reside on a lower-dimensional, query-dependent manifold. While this hypothesis has inspired new retrieval methods, existing approaches still face challenges in effectively separating non-relevant information from relevant signals. We propose a novel methodology that addresses these limitations by leveraging information from both relevant and non-relevant documents. Our method, ECLIPSE, computes a centroid based on irrelevant documents as a reference to estimate noisy dimensions present in relevant ones, enhancing retrieval performance. Extensive experiments on three in-domain and one out-of-domain benchmarks demonstrate an average improvement of up to 19.50% (resp. 22.35%) in mAP(AP) and 11.42% (resp. 13.10%) in nDCG@10 w.r.t. the DIME-based baseline (resp. the baseline using all dimensions). Our results pave the way for more robust, pseudo-irrelevance-based retrieval systems in future IR research. 

**Abstract (ZH)**: 近年来，信息检索领域的发展已经利用高维嵌入空间来提高相关文档的检索效果。此外，流形聚类假设表明，尽管存在这些高维表示，但与查询相关的文档仍然存在于一个低维度的、查询依赖的流形上。虽然这一假设启发了新的检索方法，但现有的方法仍然面临有效分离无关信息与相关信号的挑战。我们提出了一种新的方法来解决这些限制，该方法利用了相关和无关文档中的信息。我们的方法ECLIPSE基于无关文档计算一个质心，作为参考来估计相关文档中存在的噪 authService相关维度，从而提高检索性能。在三个领域内和一个领域外基准上的大量实验表明，我们的方法在mAP（平均精度）上平均提高了19.50%（相当于DIME基线的22.35%），在nDCG@10（在第10位的归一化折扣累积_gain）上平均提高了11.42%（相当于包含所有维度的基线的13.10%）。我们的研究结果为未来信息检索研究中的更稳健、基于伪无关信息的检索系统铺平了道路。 

---
# Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models 

**Title (ZH)**: 滑动窗口并非终点：探究长上下文大语言模型的全面排序能力 

**Authors**: Wenhan Liu, Xinyu Ma, Yutao Zhu, Ziliang Zhao, Shuaiqiang Wang, Dawei Yin, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14574)  

**Abstract**: Large Language Models (LLMs) have shown exciting performance in listwise passage ranking. Due to the limited input length, existing methods often adopt the sliding window strategy. Such a strategy, though effective, is inefficient as it involves repetitive and serialized processing, which usually re-evaluates relevant passages multiple times. As a result, it incurs redundant API costs, which are proportional to the number of inference tokens. The development of long-context LLMs enables the full ranking of all passages within a single inference, avoiding redundant API costs. In this paper, we conduct a comprehensive study of long-context LLMs for ranking tasks in terms of efficiency and effectiveness. Surprisingly, our experiments reveal that full ranking with long-context LLMs can deliver superior performance in the supervised fine-tuning setting with a huge efficiency improvement. Furthermore, we identify two limitations of fine-tuning the full ranking model based on existing methods: (1) sliding window strategy fails to produce a full ranking list as a training label, and (2) the language modeling loss cannot emphasize top-ranked passage IDs in the label. To alleviate these issues, we propose a new complete listwise label construction approach and a novel importance-aware learning objective for full ranking. Experiments show the superior performance of our method over baselines. Our codes are available at \url{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在列表式段落排序任务中展现了令人兴奋的表现。由于输入长度有限，现有方法通常采用滑动窗口策略。尽管这种策略有效，但它效率较低，因为其涉及重复且串联的处理过程，通常会对相关段落进行多次重新评估。因此，它会导致冗余的API费用，这些费用与推理词的数量成比例。长上下文LLMs的发展使得在单次推理中对所有段落进行全面排序成为可能，从而避免了冗余的API费用。在这篇论文中，我们从效率和效果两个方面对长上下文LLMs进行了全面研究。令人惊讶的是，我们的实验表明，在监督微调设置中，使用长上下文LLMs进行全面排序可以实现显著的性能提升。此外，我们发现基于现有方法进行全面排序模型微调的两个局限性：（1）滑动窗口策略无法生成一个完整的排序列表作为训练标签；（2）语言建模损失不能强调标签中排名第一的段落ID。为了解决这些问题，我们提出了一种新的完整列表标签构建方法和一种新的重要性感知学习目标，以实现全面排序。实验结果显示，我们的方法优于基线方法。我们的代码可在 \url{此链接} 获取。 

---
# HEC-GCN: Hypergraph Enhanced Cascading Graph Convolution Network for Multi-Behavior Recommendation 

**Title (ZH)**: HEC-GCN：基于超图增强级联图卷积网络的多行为推荐 

**Authors**: Yabo Yin, Xiaofei Zhu, Wenshan Wang, Yihao Zhang, Pengfei Wang, Yixing Fan, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2412.14476)  

**Abstract**: Multi-behavior recommendation (MBR) has garnered growing attention recently due to its ability to mitigate the sparsity issue by inferring user preferences from various auxiliary behaviors to improve predictions for the target behavior. Although existing research on MBR has yielded impressive results, they still face two major limitations. First, previous methods mainly focus on modeling fine-grained interaction information between users and items under each behavior, which may suffer from sparsity issue. Second, existing models usually concentrate on exploiting dependencies between two consecutive behaviors, leaving intra- and inter-behavior consistency largely unexplored. To the end, we propose a novel approach named Hypergraph Enhanced Cascading Graph Convolution Network for multi-behavior recommendation (HEC-GCN). To be specific, we first explore both fine- and coarse-grained correlations among users or items of each behavior by simultaneously modeling the behavior-specific interaction graph and its corresponding hypergraph in a cascaded manner. Then, we propose a behavior consistency-guided alignment strategy that ensures consistent representations between the interaction graph and its associated hypergraph for each behavior, while also maintaining representation consistency across different behaviors. Extensive experiments and analyses on three public benchmark datasets demonstrate that our proposed approach is consistently superior to previous state-of-the-art methods due to its capability to effectively attenuate the sparsity issue as well as preserve both intra- and inter-behavior consistencies. The code is available at this https URL. 

**Abstract (ZH)**: 多行为推荐（Multi-behavior Recommendation, MBR）最近引起了广泛关注，因为它能够通过从多种辅助行为中推断用户偏好来缓解目标行为预测中的稀疏性问题。尽管现有的MBR研究取得了显著成果，但仍然存在两个主要局限性。首先，现有方法主要集中在建模每种行为下用户与项目之间的精细交互信息，这可能会受到稀疏性问题的影响。其次，现有模型通常关注连续两种行为之间的依赖关系，而对行为内部和跨行为的一致性探索不足。为此，我们提出了一种新颖的方法——基于超图增强级联图卷积网络（Hypergraph Enhanced Cascading Graph Convolution Network, HEC-GCN）——来解决多行为推荐问题。

具体而言，我们首先通过同时建模特定行为的交互图及其相应的超图来探索每种行为内用户或项目之间的精细和粗糙关联，在级联方式下进行建模。然后，我们提出了一种行为一致性指导下的对齐策略，该策略确保每种行为的交互图与其相关超图之间的表示具有一致性，同时在不同行为之间保持表示一致性。在三个公开基准数据集上的广泛实验和分析表明，我们提出的这种方法由于能够有效缓解稀疏性问题并保留行为内部和跨行为一致性，在所有现有先进方法中表现始终更优。代码在以下链接中提供：[链接]。 

---
# VISA: Retrieval Augmented Generation with Visual Source Attribution 

**Title (ZH)**: VISA：带有视觉来源归因的检索增强生成 

**Authors**: Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Wenhu Chen, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14457)  

**Abstract**: Generation with source attribution is important for enhancing the verifiability of retrieval-augmented generation (RAG) systems. However, existing approaches in RAG primarily link generated content to document-level references, making it challenging for users to locate evidence among multiple content-rich retrieved documents. To address this challenge, we propose Retrieval-Augmented Generation with Visual Source Attribution (VISA), a novel approach that combines answer generation with visual source attribution. Leveraging large vision-language models (VLMs), VISA identifies the evidence and highlights the exact regions that support the generated answers with bounding boxes in the retrieved document screenshots. To evaluate its effectiveness, we curated two datasets: Wiki-VISA, based on crawled Wikipedia webpage screenshots, and Paper-VISA, derived from PubLayNet and tailored to the medical domain. Experimental results demonstrate the effectiveness of VISA for visual source attribution on documents' original look, as well as highlighting the challenges for improvement. Code, data, and model checkpoints will be released. 

**Abstract (ZH)**: 来源归属性推理增强生成对于提高检索增强生成（RAG）系统的可验证性至关重要。然而，现有的RAG方法主要将生成的内容与文档级别的引用相关联，使得用户在多个内容丰富的检索文档中查找证据变得困难。为了解决这一挑战，我们提出了视觉来源归属性检索增强生成（Visual Source Attribution-VISA），这是一种结合了答案生成和视觉来源归属性的方法。利用大规模的视觉-语言模型（VLMs），VISA能够识别支持生成答案的证据，并在检索到的文档截图中通过边界框突出显示具体的地区。为了评估其有效性，我们构建了两个数据集：基于爬取的维基百科网页截图的Wiki-VISA，以及基于PubLayNet并针对医学领域的Paper-VISA。实验结果表明，VISA在文档原始外观的视觉来源归属性推理方面具有有效性，并指出了改进面临的挑战。代码、数据和模型检查点将被公开。 

---
# Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems 

**Title (ZH)**: 较长的提示是否总是更好？大型语言模型在推荐系统中选择提示的研究 

**Authors**: Genki Kusano, Kosuke Akimoto, Kunihiro Takeoka  

**Link**: [PDF](https://arxiv.org/pdf/2412.14454)  

**Abstract**: In large language models (LLM)-based recommendation systems (LLM-RSs), accurately predicting user preferences by leveraging the general knowledge of LLMs is possible without requiring extensive training data. By converting recommendation tasks into natural language inputs called prompts, LLM-RSs can efficiently solve issues that have been difficult to address due to data scarcity but are crucial in applications such as cold-start and cross-domain problems. However, when applying this in practice, selecting the prompt that matches tasks and data is essential. Although numerous prompts have been proposed in LLM-RSs and representing the target user in prompts significantly impacts recommendation accuracy, there are still no clear guidelines for selecting specific prompts.
In this paper, we categorize and analyze prompts from previous research to establish practical prompt selection guidelines. Through 450 experiments with 90 prompts and five real-world datasets, we examined the relationship between prompts and dataset characteristics in recommendation accuracy. We found that no single prompt consistently outperforms others; thus, selecting prompts on the basis of dataset characteristics is crucial. Here, we propose a prompt selection method that achieves higher accuracy with minimal validation data. Because increasing the number of prompts to explore raises costs, we also introduce a cost-efficient strategy using high-performance and cost-efficient LLMs, significantly reducing exploration costs while maintaining high prediction accuracy. Our work offers valuable insights into the prompt selection, advancing accurate and efficient LLM-RSs. 

**Abstract (ZH)**: 在基于大规模语言模型（LLM）的推荐系统（LLM-RSs）中，通过利用LLM的普遍知识，无需大量训练数据即可准确预测用户偏好。通过将推荐任务转化为自然语言输入（称为提示），LLM-RSs可以高效地解决由于数据稀缺而难以处理但在冷启动和跨域问题等应用中至关重要的问题。然而，在实际应用中，选择与任务和数据匹配的提示至关重要。尽管在LLM-RSs中已经提出了许多提示，且提示中的目标用户的表示对推荐准确性产生了显著影响，但仍缺乏明确的提示选择指南。

本文对先前研究中的提示进行分类和分析，以确立实用的提示选择指南。通过使用90个提示和五个真实世界数据集进行450次实验，我们考察了提示与数据集特性之间的关系在推荐准确性方面的影响。我们发现没有一种提示能够始终优于其他提示，因此，基于数据集特性选择提示至关重要。在此基础上，我们提出了一种提示选择方法，仅需少量验证数据即可实现更高的准确性。由于增加提示数量以进行探索会增加成本，我们还介绍了一种成本有效的策略，使用高性能且成本效益高的LLM，显著降低了探索成本同时保持了高预测准确性。我们的工作为提示选择提供了宝贵的见解，推动了准确高效的LLM-RSs的发展。 

---
# ChainRank-DPO: Chain Rank Direct Preference Optimization for LLM Rankers 

**Title (ZH)**: ChainRank-DPO：链式排名直接偏好优化方法用于LLM排名器 

**Authors**: Haowei Liu, Xuyang Wu, Guohao Sun, Zhiqiang Tao, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14405)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable effectiveness in text reranking through works like RankGPT, leveraging their human-like reasoning about relevance. However, supervised fine-tuning for ranking often diminishes these models' general-purpose capabilities, including the crucial reasoning abilities that make them valuable for ranking. We introduce a novel approach integrating Chain-of-Thought prompting with an SFT-DPO (Supervised Fine-Tuning followed by Direct Preference Optimization) pipeline to preserve these capabilities while improving ranking performance. Our experiments on TREC 2019 and 2020 Deep Learning datasets show that our approach outperforms the state-of-the-art RankZephyr while maintaining strong performance on the Massive Multitask Language Understanding (MMLU) benchmark, demonstrating effective preservation of general-purpose capabilities through thoughtful fine-tuning strategies. Our code and data will be publicly released upon the acceptance of the paper. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已在通过RankGPT等作品中的文本重排序任务中展示了令人瞩目的有效性，这主要得益于它们在相关性上的类似人类的推理能力。然而，监督微调排序通常会削弱这些模型的通用能力，包括使它们在排序任务中具有价值的关键推理能力。我们提出了一种新的方法，将链式思维提示与SFT-DPO（监督微调后直接偏好优化）管道相结合，以保留这些能力并提高排序性能。我们在TREC 2019和2020年深度学习数据集上的实验表明，我们的方法在保持强劲性能的同时超越了最先进的RankZephyr方法，证明了通过谨慎的微调策略有效保持了通用能力。我们的代码和数据将在论文被接受后公开发布。 

---
# Embedding Cultural Diversity in Prototype-based Recommender Systems 

**Title (ZH)**: 在原型推荐系统中嵌入文化多样性 

**Authors**: Armin Moradi, Nicola Neophytou, Florian Carichon, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2412.14329)  

**Abstract**: Popularity bias in recommender systems can increase cultural overrepresentation by favoring norms from dominant cultures and marginalizing underrepresented groups. This issue is critical for platforms offering cultural products, as they influence consumption patterns and human perceptions. In this work, we address popularity bias by identifying demographic biases within prototype-based matrix factorization methods. Using the country of origin as a proxy for cultural identity, we link this demographic attribute to popularity bias by refining the embedding space learning process. First, we propose filtering out irrelevant prototypes to improve representativity. Second, we introduce a regularization technique to enforce a uniform distribution of prototypes within the embedding space. Across four datasets, our results demonstrate a 27\% reduction in the average rank of long-tail items and a 2\% reduction in the average rank of items from underrepresented countries. Additionally, our model achieves a 2\% improvement in HitRatio@10 compared to the state-of-the-art, highlighting that fairness is enhanced without compromising recommendation quality. Moreover, the distribution of prototypes leads to more inclusive explanations by better aligning items with diverse prototypes. 

**Abstract (ZH)**: 推荐系统中的流行度偏见可能会通过偏好主导文化规范而增加文化过度代表的现象，并边缘化未被代表的群体。这对于提供文化产品的平台来说是一个关键问题，因为这些平台会影响消费模式和人类的认知。在本研究中，我们通过识别基于原型的矩阵分解方法中的人口统计学偏见来解决流行度偏见问题。使用来源国家作为文化身份的代理，我们通过精炼嵌入空间的学习过程将这一人口统计学属性与流行度偏见联系起来。首先，我们提出过滤掉无关的原型以提高代表性。其次，我们引入了一种正则化技术，以确保嵌入空间中原型的均匀分布。在四个数据集上，我们的结果表明，长尾项目的平均排名降低了27%，未被充分代表的国家的项目的平均排名降低了2%。此外，与当前最先进的技术相比，我们的模型在HitRatio@10上提高了2%的表现，这表明公平性得到了提升，而不会牺牲推荐的质量。此外，原型的分布导致了更具包容性的解释，因为它更好地将项目与多样性的原型对齐。 

---
# SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation 

**Title (ZH)**: SAFERec：结合自我注意力和频率增强的篮子推荐模型 

**Authors**: Oleg Lashinin, Denis Krasilnikov, Aleksandr Milogradskii, Marina Ananyeva  

**Link**: [PDF](https://arxiv.org/pdf/2412.14302)  

**Abstract**: Transformer-based approaches such as BERT4Rec and SASRec demonstrate strong performance in Next Item Recommendation (NIR) tasks. However, applying these architectures to Next-Basket Recommendation (NBR) tasks, which often involve highly repetitive interactions, is challenging due to the vast number of possible item combinations in a basket. Moreover, frequency-based methods such as TIFU-KNN and UP-CF still demonstrate strong performance in NBR tasks, frequently outperforming deep-learning approaches. This paper introduces SAFERec, a novel algorithm for NBR that enhances transformer-based architectures from NIR by incorporating item frequency information, consequently improving their applicability to NBR tasks. Extensive experiments on multiple datasets show that SAFERec outperforms all other baselines, specifically achieving an 8\% improvement in Recall@10. 

**Abstract (ZH)**: 基于Transformer的方法，如BERT4Rec和SASRec，在Next Item Recommendation (NIR)任务中表现出强大的性能。然而，将这些架构应用于Next-Basket Recommendation (NBR)任务中是具有挑战性的，因为篮子中可能包含大量的项目组合，且频繁出现高度重复的交互。此外，基于频率的方法，如TIFU-KNN和UP-CF，在NBR任务中仍然表现出强大的性能，经常优于深度学习方法。本文介绍了一种新的算法SAFERec，该算法通过结合项目频率信息来增强NIR中的Transformer架构，从而提高了其在NBR任务中的适用性。在多个数据集的广泛实验中，SAFERec在所有基线方法中表现出色，特别是在Recall@10方面实现了8%的提升。 

---
# Progressive Multimodal Reasoning via Active Retrieval 

**Title (ZH)**: 基于主动检索的逐步多模态推理 

**Authors**: Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14835)  

**Abstract**: Multi-step multimodal reasoning tasks pose significant challenges for multimodal large language models (MLLMs), and finding effective ways to enhance their performance in such scenarios remains an unresolved issue. In this paper, we propose AR-MCTS, a universal framework designed to progressively improve the reasoning capabilities of MLLMs through Active Retrieval (AR) and Monte Carlo Tree Search (MCTS). Our approach begins with the development of a unified retrieval module that retrieves key supporting insights for solving complex reasoning problems from a hybrid-modal retrieval corpus. To bridge the gap in automated multimodal reasoning verification, we employ the MCTS algorithm combined with an active retrieval mechanism, which enables the automatic generation of step-wise annotations. This strategy dynamically retrieves key insights for each reasoning step, moving beyond traditional beam search sampling to improve the diversity and reliability of the reasoning space. Additionally, we introduce a process reward model that aligns progressively to support the automatic verification of multimodal reasoning tasks. Experimental results across three complex multimodal reasoning benchmarks confirm the effectiveness of the AR-MCTS framework in enhancing the performance of various multimodal models. Further analysis demonstrates that AR-MCTS can optimize sampling diversity and accuracy, yielding reliable multimodal reasoning. 

**Abstract (ZH)**: 多步骤多模态推理任务对多模态大型语言模型（MLLMs）提出了重大挑战，如何在这些情境中有效提升其性能仍是一个未解决的问题。本文提出了一种名为AR-MCTS的通用框架，该框架旨在通过主动检索（AR）和蒙特卡罗树搜索（MCTS）逐步提升MLLMs的推理能力。我们的方法从开发一个统一的检索模块开始，该模块可以从混合模态检索语料库中检索解决复杂推理问题所需的关键支持见解。为了在自动化多模态推理验证中缩小差距，我们结合了MCTS算法和主动检索机制，以实现逐步生成步骤注解。这种策略动态地为每一推理步骤检索关键见解，超越了传统的束搜索采样方法，以提高推理空间的多样性和可靠性。此外，我们还引入了一个过程奖励模型，以逐步支持多模态推理任务的自动验证。在三个复杂多模态推理基准上的实验结果证实了AR-MCTS框架在提升各种多模态模型性能方面的有效性。进一步的分析表明，AR-MCTS可以优化采样多样性和准确性，从而实现可靠多模态推理。 

---
# Efficient Self-Supervised Video Hashing with Selective State Spaces 

**Title (ZH)**: 高效的自监督视频哈希方法：选择性状态空间机制 

**Authors**: Jinpeng Wang, Niu Lian, Jun Li, Yuting Wang, Yan Feng, Bin Chen, Yongbing Zhang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2412.14518)  

**Abstract**: Self-supervised video hashing (SSVH) is a practical task in video indexing and retrieval. Although Transformers are predominant in SSVH for their impressive temporal modeling capabilities, they often suffer from computational and memory inefficiencies. Drawing inspiration from Mamba, an advanced state-space model, we explore its potential in SSVH to achieve a better balance between efficacy and efficiency. We introduce S5VH, a Mamba-based video hashing model with an improved self-supervised learning paradigm. Specifically, we design bidirectional Mamba layers for both the encoder and decoder, which are effective and efficient in capturing temporal relationships thanks to the data-dependent selective scanning mechanism with linear complexity. In our learning strategy, we transform global semantics in the feature space into semantically consistent and discriminative hash centers, followed by a center alignment loss as a global learning signal. Our self-local-global (SLG) paradigm significantly improves learning efficiency, leading to faster and better convergence. Extensive experiments demonstrate S5VH's improvements over state-of-the-art methods, superior transferability, and scalable advantages in inference efficiency. Code is available at this https URL. 

**Abstract (ZH)**: 自监督视频哈希（SSVH）是视频索引和检索中的一个实际任务。尽管Transformer在SSVH中占据主导地位，得益于其出色的时序建模能力，但它们经常面临计算和内存效率低下的问题。受到Mamba这一高级状态空间模型的启发，我们探讨了Mamba在SSVH中的潜力，以实现效用和效率之间的更好平衡。我们提出了S5VH，这是一种基于Mamba的视频哈希模型，采用了改进的自监督学习范式。具体而言，我们在编码器和解码器中设计了双向Mamba层，这些层通过依赖于数据的选择性扫描机制，具有线性复杂度，从而能够有效地捕捉时序关系。在我们的学习策略中，我们将特征空间中的全局语义转换为语义上一致且区分性强的哈希中心，并通过中心对齐损失作为全局学习信号。我们自局部-全局（SLG）范式显著提高了学习效率，从而实现了更快更好地收敛。广泛的实验结果证明了S5VH在性能和效率上的优越性，具有更好的迁移性和可扩展性。代码可在以下链接中获取：this https URL。 

---
# Moving Beyond LDA: A Comparison of Unsupervised Topic Modelling Techniques for Qualitative Data Analysis of Online Communities 

**Title (ZH)**: 超越LDA：无监督主题建模技术在在线社区定性数据分析中的比较 

**Authors**: Amandeep Kaur, James R. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2412.14486)  

**Abstract**: Social media constitutes a rich and influential source of information for qualitative researchers. Although computational techniques like topic modelling assist with managing the volume and diversity of social media content, qualitative researcher's lack of programming expertise creates a significant barrier to their adoption. In this paper we explore how BERTopic, an advanced Large Language Model (LLM)-based topic modelling technique, can support qualitative data analysis of social media. We conducted interviews and hands-on evaluations in which qualitative researchers compared topics from three modelling techniques: LDA, NMF, and BERTopic. BERTopic was favoured by 8 of 12 participants for its ability to provide detailed, coherent clusters for deeper understanding and actionable insights. Participants also prioritised topic relevance, logical organisation, and the capacity to reveal unexpected relationships within the data. Our findings underscore the potential of LLM-based techniques for supporting qualitative analysis. 

**Abstract (ZH)**: 社会媒体构成了丰富且有影响力的定性研究信息来源。尽管像主题建模这样的计算技术有助于处理社会媒体内容的庞大和多样性，但定性研究者的编程技能缺乏却构成了其采用的重要障碍。本文探讨了如何通过基于大型语言模型（LLM）的主题建模技术BERTopic来支持社会媒体的定性数据分析。我们进行了访谈和实际操作评估，让定性研究人员将三种建模技术——LDA、NMF 和 BERTopic——产生的主题进行了比较。共有12名参与者中有8人青睐BERTopic，因为它能够提供详细、连贯的主题簇以加深理解并提取实质性见解。参与者还强调了主题的相关性、逻辑组织以及揭示数据中意想不到的关系的能力。我们的研究结果表明，基于大型语言模型的技术有可能支持定性分析。 

---
# State Space Models are Strong Text Rerankers 

**Title (ZH)**: 状态空间模型是强大的文本重排序器 

**Authors**: Zhichao Xu, Jinghua Yan, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2412.14354)  

**Abstract**: Transformers dominate NLP and IR; but their inference inefficiencies and challenges in extrapolating to longer contexts have sparked interest in alternative model architectures. Among these, state space models (SSMs) like Mamba offer promising advantages, particularly $O(1)$ time complexity in inference. Despite their potential, SSMs' effectiveness at text reranking -- a task requiring fine-grained query-document interaction and long-context understanding -- remains underexplored.
This study benchmarks SSM-based architectures (specifically, Mamba-1 and Mamba-2) against transformer-based models across various scales, architectures, and pre-training objectives, focusing on performance and efficiency in text reranking tasks. We find that (1) Mamba architectures achieve competitive text ranking performance, comparable to transformer-based models of similar size; (2) they are less efficient in training and inference compared to transformers with flash attention; and (3) Mamba-2 outperforms Mamba-1 in both performance and efficiency. These results underscore the potential of state space models as a transformer alternative and highlight areas for improvement in future IR applications. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文版本：

变换器在自然语言处理（NLP）和信息检索（IR）中占据主导地位，但它们在推理效率方面的不足以及在处理较长上下文时的外推挑战引起了对替代模型架构的兴趣。在这些替代模型中，状态空间模型（SSMs）如Mamba展现出潜在的优势，尤其是在推理过程中提供恒定时间复杂度（$O(1)$）。尽管存在这些潜力，但状态空间模型在文本重排序任务中的效果尚未得到充分探索，而文本重排序任务需要精细的查询-文档交互和长上下文理解。

本研究将基于状态空间模型的架构（具体为Mamba-1和Mamba-2）与基于变换器的模型在不同规模、架构和预训练目标下进行基准测试，重点在于文本重排序任务中的性能和效率。研究结果表明：（1）Mamba架构在文本排序性能上达到与相似规模的变换器基模型相当的水平；（2）Mamba与具有闪存注意机制的变换器相比，在训练和推理方面效率较低；（3）Mamba-2在性能和效率上均优于Mamba-1。这些结果强调了状态空间模型作为变换器替代品的潜力，并指出了未来信息检索应用中需要改进的领域。 

---
# Transversal PACS Browser API: Addressing Interoperability Challenges in Medical Imaging Systems 

**Title (ZH)**: 横截面PACS浏览API：解决医疗成像系统中的互操作性挑战 

**Authors**: Diogo Lameira, Filipa Ferraz  

**Link**: [PDF](https://arxiv.org/pdf/2412.14229)  

**Abstract**: Advances in imaging technologies have revolutionised the medical imaging and healthcare sectors, leading to the widespread adoption of PACS for the storage, retrieval, and communication of medical images. Although these systems have improved operational efficiency, significant challenges remain in effectively retrieving DICOM images, which are essential for diagnosis and overall patient care. Moreover, issues such as fragmented systems, interoperability barriers, and complex user interfaces can often prevent healthcare professionals from efficiently accessing medical images. Addressing these challenges, the Transversal PACS Browser API is a robust and user-friendly solution designed to enhance the process of querying and retrieving DICOM images. It offers advanced filtering capabilities through a variety of filter options as well as a custom field search, that allows users to easily navigate through large medical image collections with ease. Additionally, the application provides a unified interface for querying and retrieving from multiple PACS stations, addressing the challenges of fragmentation and complexity associated with accessing medical images. Other key features include the ability to pre-view images directly within the application. All of this contributes to the transversal nature of the API, serving not only healthcare providers, but anyone who relies on efficient access to these resources. To validate the performance and usability of the application, comprehensive testing was carried out with stakeholders of the field, the results of which showed general satisfaction, highlighting the API's clean design, ease of use, and effective search capabilities of the API, as well as the usefulness of previewing images within the application. 

**Abstract (ZH)**: 影像技术的进步已彻底改变了医学影像和医疗保健领域，促使普及了 PACS（病人存取控制系统）系统，该系统用于存储、检索和传达医学影像。尽管这些系统在提升运营效率方面取得了显著进展，但在有效检索 DICOM 图像方面仍面临重大挑战，而这些图像对于诊断和整体患者护理至关重要。此外，诸如系统碎片化、互操作性障碍及复杂的用户界面等问题常常妨碍医疗专业人员高效访问医学影像。

为应对这些挑战，跨部门 PACS 浏览器 API 提供了一个强大且用户友好的解决方案，旨在增强 DICOM 图像的查询和检索过程。它通过多种过滤选项和自定义字段搜索提供了高级筛选能力，使用户能够轻松浏览大型医学影像集合。此外，该应用提供了统一的界面以查询和从多个 PACS 站点检索医学影像，解决了访问医学影像过程中碎片化和复杂性的问题。其他关键功能包括直接在应用中预览图像的能力。所有这些都使得该 API 具有横向特性，不仅适用于医疗提供者，也适用于依赖高效访问这些资源的任何人。

为了验证该应用的性能和易用性，对相关领域的利益相关者进行了全面测试，结果显示普遍满意。这些结果强调了 API 的简洁设计、易于使用以及 API 的有效搜索能力，并突显了在应用中预览图像的功能的重要性。

综上所述，跨部门 PACS 浏览器 API 是一个旨在提升 DICOM 图像查询和检索过程的解决方案，通过多种高级筛选选项和预览图像功能，它为用户提供了方便且高效的访问途径。 

---
# Whom do Explanations Serve? A Systematic Literature Survey of User Characteristics in Explainable Recommender Systems Evaluation 

**Title (ZH)**: 解释性推荐系统评价中用户的特性研究：一项系统文献综述 

**Authors**: Kathrin Wardatzky, Oana Inel, Luca Rossetto, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2412.14193)  

**Abstract**: Adding explanations to recommender systems is said to have multiple benefits, such as increasing user trust or system transparency. Previous work from other application areas suggests that specific user characteristics impact the users' perception of the explanation. However, we rarely find this type of evaluation for recommender systems explanations. This paper addresses this gap by surveying 124 papers in which recommender systems explanations were evaluated in user studies. We analyzed their participant descriptions and study results where the impact of user characteristics on the explanation effects was measured. Our findings suggest that the results from the surveyed studies predominantly cover specific users who do not necessarily represent the users of recommender systems in the evaluation domain. This may seriously hamper the generalizability of any insights we may gain from current studies on explanations in recommender systems. We further find inconsistencies in the data reporting, which impacts the reproducibility of the reported results. Hence, we recommend actions to move toward a more inclusive and reproducible evaluation. 

**Abstract (ZH)**: 将推荐系统中添加解释的说法认为具有多方面的好处，比如增加用户信任或提高系统的透明度。从前人在其他应用领域的研究中可以看出，特定的用户特征对用户对解释的感知产生了影响。然而，我们在推荐系统解释方面很少看到这种类型的影响评估。本文通过调研了124篇在用户研究中评估推荐系统解释的论文，填补了这一空白。我们分析了这些研究中关于参与者的描述以及关于用户特征对解释效果影响的研究结果。我们的研究发现表明，被调研的研究结果大多集中在特定的用户群体上，这些用户群体并不一定代表了评价领域中推荐系统的实际用户。这可能严重影响我们从当前关于推荐系统解释的研究中获得的任何洞察的普适性。此外，我们还发现了报告数据的一致性问题，这影响了所报告结果的可重复性。因此，我们建议采取行动以实现更加包容性和可重复性的评估。 

---
# Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics with Large Language Models 

**Title (ZH)**: 基于大型语言模型的数据分析中多步洞察综合的高级推理与转换引擎 

**Authors**: Atin Sakkeer Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2412.14146)  

**Abstract**: This paper presents the Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics (ARTEMIS-DA), a novel framework designed to augment Large Language Models (LLMs) for solving complex, multi-step data analytics tasks. ARTEMIS-DA integrates three core components: the Planner, which dissects complex user queries into structured, sequential instructions encompassing data preprocessing, transformation, predictive modeling, and visualization; the Coder, which dynamically generates and executes Python code to implement these instructions; and the Grapher, which interprets generated visualizations to derive actionable insights. By orchestrating the collaboration between these components, ARTEMIS-DA effectively manages sophisticated analytical workflows involving advanced reasoning, multi-step transformations, and synthesis across diverse data modalities. The framework achieves state-of-the-art (SOTA) performance on benchmarks such as WikiTableQuestions and TabFact, demonstrating its ability to tackle intricate analytical tasks with precision and adaptability. By combining the reasoning capabilities of LLMs with automated code generation and execution and visual analysis, ARTEMIS-DA offers a robust, scalable solution for multi-step insight synthesis, addressing a wide range of challenges in data analytics. 

**Abstract (ZH)**: 本文介绍了高级推理与转换引擎（ARTEMIS-DA，Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics），这是一种新型框架，旨在增强大型语言模型（LLMs），以解决复杂的多步骤数据解析任务。ARTEMIS-DA 融合了三个核心组件：规划器（Planner），它将复杂的用户查询分解成结构化、顺序的指令，涵盖数据预处理、转换、预测建模和可视化；编码器（Coder），它动态生成并执行 Python 代码以实现这些指令；以及绘图器（Grapher），它解释生成的可视化结果以推导出可操作的洞察。通过协调这些组件之间的协作，ARTEMIS-DA 有效地管理了涉及高级推理、多步骤转换和跨多种数据模态综合的复杂分析工作流。该框架在 WikiTableQuestions 和 TabFact 等基准测试中达到了最先进的（SOTA）性能，展示了其在精确性和适应性方面处理复杂分析任务的能力。通过结合大型语言模型的推理能力、自动化代码生成和执行以及可视化分析，ARTEMIS-DA 提供了一种稳健且可扩展的多步骤洞察综合解决方案，能够应对数据解析的广泛挑战。 

---
# A Cognitive Ideation Support Framework using IBM Watson Services 

**Title (ZH)**: 使用IBM Watson服务的认知创意支持框架 

**Authors**: Samaa Elnagar, Kweku-Muata Osei-Bryson  

**Link**: [PDF](https://arxiv.org/pdf/2412.14025)  

**Abstract**: Ideas generation is a core activity for innovation in organizations. The creativity of the generated ideas depends not only on the knowledge retrieved from the organizations' knowledge bases, but also on the external knowledge retrieved from other resources. Unfortunately, organizations often cannot efficiently utilize the knowledge in the knowledge bases due to the limited abilities of the search and retrieval mechanisms especially when dealing with unstructured data. In this paper, we present a new cognitive support framework for ideation that uses the IBM Watson DeepQA services. IBM Watson is a Question Answering system which mimics human cognitive abilities to retrieve and rank information. The proposed framework is based on the Search for Ideas in the Associative Memory (SIAM) model to help organizations develop creative ideas through discovering new relationships between retrieved data. To evaluate the effectiveness of the proposed system, the generated ideas generated are selected and assessed using a set of established creativity criteria. 

**Abstract (ZH)**: 创新是组织的核心活动之一。生成的创意的质量不仅取决于从组织的知识库中检索的知识，还取决于从其他资源中检索的外部知识。不幸的是，由于搜索和检索机制能力有限，特别是在处理非结构化数据时，组织往往难以有效利用知识库中的知识。本文提出了一种新的认知支持框架，用于创意生成，该框架利用了IBM Watson DeepQA服务。IBM Watson是一个问答系统，模仿人类的认知能力，检索和排序信息。所提出的框架基于关联记忆中创意搜索（SIAM）模型，通过发现检索数据之间的新关系帮助组织发展创造性想法。为了评估所提出系统的有效性，生成的创意被选择并使用一套既定的创造性标准进行评估。 

---
# CRM: Retrieval Model with Controllable Condition 

**Title (ZH)**: CRM：可控条件下的检索模型 

**Authors**: Chi Liu, Jiangxia Cao, Rui Huang, Kuo Cai, Weifeng Ding, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13844)  

**Abstract**: Recommendation systems (RecSys) are designed to connect users with relevant items from a vast pool of candidates while aligning with the business goals of the platform. A typical industrial RecSys is composed of two main stages, retrieval and ranking: (1) the retrieval stage aims at searching hundreds of item candidates satisfied user interests; (2) based on the retrieved items, the ranking stage aims at selecting the best dozen items by multiple targets estimation for each item candidate, including classification and regression targets. Compared with ranking model, the retrieval model absence of item candidate information during inference, therefore retrieval models are often trained by classification target only (e.g., click-through rate), but failed to incorporate regression target (e.g., the expected watch-time), which limit the effectiveness of retrieval. In this paper, we propose the Controllable Retrieval Model (CRM), which integrates regression information as conditional features into the two-tower retrieval paradigm. This modification enables the retrieval stage could fulfill the target gap with ranking model, enhancing the retrieval model ability to search item candidates satisfied the user interests and condition effectively. We validate the effectiveness of CRM through real-world A/B testing and demonstrate its successful deployment in Kuaishou short-video recommendation system, which serves over 400 million users. 

**Abstract (ZH)**: 推荐系统（RecSys）旨在将用户与海量候选项中相关的内容相连接，同时符合平台的商业目标。典型的工业推荐系统通常包含两个主要阶段：检索和排名：（1）检索阶段旨在搜索满足用户兴趣的数百个候选项；（2）基于检索出的项，排名阶段旨在通过每个候选项的多目标估计，选择最佳的几个项，包括分类和回归目标。与排名模型相比，检索模型在推理过程中缺乏候选项的信息，因此检索模型通常仅通过分类目标（例如点击率）进行训练，而未能纳入回归目标（例如预期观看时间），这限制了检索模型的效果。在本文中，我们提出了一种可控的检索模型（Controllable Retrieval Model, CRM），它将回归信息作为条件特征整合到双塔检索框架中。这一修改使检索阶段能够弥补与排名模型的目标差距，增强检索模型在搜索满足用户兴趣和条件的候选项方面的能力。我们通过实际的A/B测试验证了CRM的有效性，并展示了其在快手短视频推荐系统中的成功部署，该系统服务于超过4亿用户。 

---
# Maybe you are looking for CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval 

**Title (ZH)**: 也许您正在寻找CroQS：跨模态查询建议用于文本到图像检索 

**Authors**: Giacomo Pacini, Fabio Carrara, Nicola Messina, Nicola Tonellotto, Giuseppe Amato, Fabrizio Falchi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13834)  

**Abstract**: Query suggestion, a technique widely adopted in information retrieval, enhances system interactivity and the browsing experience of document collections. In cross-modal retrieval, many works have focused on retrieving relevant items from natural language queries, while few have explored query suggestion solutions. In this work, we address query suggestion in cross-modal retrieval, introducing a novel task that focuses on suggesting minimal textual modifications needed to explore visually consistent subsets of the collection, following the premise of ''Maybe you are looking for''. To facilitate the evaluation and development of methods, we present a tailored benchmark named CroQS. This dataset comprises initial queries, grouped result sets, and human-defined suggested queries for each group. We establish dedicated metrics to rigorously evaluate the performance of various methods on this task, measuring representativeness, cluster specificity, and similarity of the suggested queries to the original ones. Baseline methods from related fields, such as image captioning and content summarization, are adapted for this task to provide reference performance scores. Although relatively far from human performance, our experiments reveal that both LLM-based and captioning-based methods achieve competitive results on CroQS, improving the recall on cluster specificity by more than 115% and representativeness mAP by more than 52% with respect to the initial query. The dataset, the implementation of the baseline methods and the notebooks containing our experiments are available here: this https URL 

**Abstract (ZH)**: 查询建议是一种广泛应用于信息检索技术，它增强了系统的互动性和文档集合的浏览体验。在跨模态检索中，许多研究集中在从自然语言查询中检索相关项，而对于查询建议的研究却较少。在这项工作中，我们针对跨模态检索中的查询建议问题提出了一种新的任务，该任务关注建议最少的文字修改以探索与视觉一致的集合子集，遵循“或许您在寻找”的前提。为促进对该任务的方法的评估和开发，我们提出了一种定制化的基准数据集名称为CroQS。该数据集包括初始查询、分组结果集以及每组的人工建议查询。我们制定了专门的指标来严格评估各种方法在该任务上的性能，测量建议查询的代表性、簇特异性以及与原始查询的相似度。来自相关领域的基线方法，如图像描述和内容摘要，被调整用于该任务，提供了参考性能分数。尽管与人类性能相比仍有较大差距，但我们的实验表明，基于大型语言模型的方法和基于图像描述的方法在CroQS上取得非常竞争力的结果，与初始查询相比，簇特异性召回率提高超过115%，代表性mAP提高超过52%。该数据集、基线方法的实现以及包含我们实验的笔记本已在此处提供：this https URL 

---
# Heterogeneous Graph Collaborative Filtering 

**Title (ZH)**: 异质图协作过滤 

**Authors**: Lianghao Xia, Meiyan Xie, Yong Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13825)  

**Abstract**: For modern recommender systems, the use of low-dimensional latent representations to embed users and items based on their observed interactions has become commonplace. However, many existing recommendation models are primarily designed for coarse-grained and homogeneous interactions, which limits their effectiveness in two critical dimensions. Firstly, these models fail to leverage the relational dependencies that exist across different types of user behaviors, such as page views, collects, comments, and purchases. Secondly, they struggle to capture the fine-grained latent factors that drive user interaction patterns. To address these limitations, we present a heterogeneous graph collaborative filtering model MixRec that excels at disentangling users' multi-behavior interaction patterns and uncovering the latent intent factors behind each behavior. Our model achieves this by incorporating intent disentanglement and multi-behavior modeling, facilitated by a parameterized heterogeneous hypergraph architecture. Furthermore, we introduce a novel contrastive learning paradigm that adaptively explores the advantages of self-supervised data augmentation, thereby enhancing the model's resilience against data sparsity and expressiveness with relation heterogeneity. To validate the efficacy of MixRec, we conducted extensive experiments on three public datasets. The results clearly demonstrate its superior performance, significantly outperforming various state-of-the-art baselines. Our model is open-sourced and available at: this https URL. 

**Abstract (ZH)**: 对于现代推荐系统而言，基于用户观察到的交互行为使用低维潜在表示嵌入用户和项目已成为常见做法。然而，许多现有的推荐模型主要针对粗粒度且同质化的交互设计，这在两个关键维度上限制了它们的效果。首先，这些模型未能利用不同类型用户行为之间的关系依赖性，例如页面浏览、收藏、评论和购买行为。其次，它们难以捕捉驱动用户交互模式的细粒度潜在因素。为了解决这些限制，我们提出了一种异构图协作过滤模型MixRec，该模型能够出色地分离用户多行为的交互模式，并揭示每种行为背后的潜在意图因素。我们的模型通过引入意图分离和多行为建模，结合参数化异构超图架构实现这一目标。此外，我们引入了一种新颖的对比学习范式，该范式能够自适应地探索自监督数据增强的优势，从而增强模型在数据稀疏性和关系异质性下的鲁棒性和表达能力。为了验证MixRec的有效性，我们在三个公开数据集上进行了广泛的实验。结果显示，该模型的性能明显优于多种最新的基线模型。我们的模型已开源，并可在以下链接获取：this https URL。 

---
# Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization 

**Title (ZH)**: 语义趋同：通过两阶段对齐和行为语义令牌化谐调推荐系统 

**Authors**: Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13771)  

**Abstract**: Large language models (LLMs), endowed with exceptional reasoning capabilities, are adept at discerning profound user interests from historical behaviors, thereby presenting a promising avenue for the advancement of recommendation systems. However, a notable discrepancy persists between the sparse collaborative semantics typically found in recommendation systems and the dense token representations within LLMs. In our study, we propose a novel framework that harmoniously merges traditional recommendation models with the prowess of LLMs. We initiate this integration by transforming ItemIDs into sequences that align semantically with the LLMs space, through the proposed Alignment Tokenization module. Additionally, we design a series of specialized supervised learning tasks aimed at aligning collaborative signals with the subtleties of natural language semantics. To ensure practical applicability, we optimize online inference by pre-caching the top-K results for each user, reducing latency and improving effciency. Extensive experimental evidence indicates that our model markedly improves recall metrics and displays remarkable scalability of recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其出色的推理能力，能够从用户的历史行为中识别出深层次的兴趣，从而为推荐系统的发展提供了新的可能性。然而，推荐系统中常见的稀疏协作语义与LLMs中密集的词表示之间存在显著的差距。在我们的研究中，提出了一种新的框架，将传统的推荐模型与LLMs的能力和谐地结合起来。我们通过提出的对齐分词模块将ItemIDs转换为与LLMs空间相匹配的语义序列，从而开启这一集成过程。此外，我们设计了一系列专门的监督学习任务，旨在将协作信号与自然语言语义的细微差别对齐。为了确保实际应用的可行性，我们通过预先缓存每个用户的Top-K结果来优化在线推理过程，从而降低延迟并提高效率。大量的实验证据表明，我们的模型在召回率指标上取得了显著的提升，并展示了推荐系统令人瞩目的扩展性。 

---
# Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models 

**Title (ZH)**: 使用大规模语言模型弥合知识aware推荐中的用户侧知识差距 

**Authors**: Zheng Hu, Zhe Li, Ziyun Jiao, Satoshi Nakagawa, Jiawen Deng, Shimin Cai, Tao Zhou, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.13544)  

**Abstract**: In recent years, knowledge graphs have been integrated into recommender systems as item-side auxiliary information, enhancing recommendation accuracy. However, constructing and integrating structural user-side knowledge remains a significant challenge due to the improper granularity and inherent scarcity of user-side features. Recent advancements in Large Language Models (LLMs) offer the potential to bridge this gap by leveraging their human behavior understanding and extensive real-world knowledge. Nevertheless, integrating LLM-generated information into recommender systems presents challenges, including the risk of noisy information and the need for additional knowledge transfer. In this paper, we propose an LLM-based user-side knowledge inference method alongside a carefully designed recommendation framework to address these challenges. Our approach employs LLMs to infer user interests based on historical behaviors, integrating this user-side information with item-side and collaborative data to construct a hybrid structure: the Collaborative Interest Knowledge Graph (CIKG). Furthermore, we propose a CIKG-based recommendation framework that includes a user interest reconstruction module and a cross-domain contrastive learning module to mitigate potential noise and facilitate knowledge transfer. We conduct extensive experiments on three real-world datasets to validate the effectiveness of our method. Our approach achieves state-of-the-art performance compared to competitive baselines, particularly for users with sparse interactions. 

**Abstract (ZH)**: 近年来，知识图谱被整合到推荐系统中，用作项目侧面的辅助信息，以提高推荐准确性。然而，由于用户侧面特征的不适当粒度和固有的稀缺性，构建和整合结构化的用户侧面知识仍是重大挑战。大型语言模型（LLMs）的最近进展为解决这一问题提供了潜力，它们能够利用对人类行为的理解以及广泛的实际世界知识。然而，将LLM生成的信息集成到推荐系统中也面临着挑战，包括噪声信息的风险和额外知识迁移的需求。本文提出了一种基于LLM的用户侧面知识推理方法，并设计了一个精心构建的推荐框架以应对这些挑战。我们的方法利用LLM根据历史行为推断用户兴趣，将用户侧面信息与项目侧面信息和协作数据相结合，构建了一种混合结构：协作兴趣知识图谱（CIKG）。此外，我们提出了一种基于CIKG的推荐框架，包括用户兴趣重构模块和跨领域对比学习模块，以减轻潜在的噪声并促进知识迁移。我们在三个真实世界的数据集上进行了广泛的实验证明了我们方法的有效性。对于稀疏交互的用户，我们的方法在与竞争基线相比时达到了最先进的性能。 

---
# Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future 

**Title (ZH)**: 大型语言模型增强的推荐系统：分类、趋势、应用与未来 

**Authors**: Qidong Liu, Xiangyu Zhao, Yuhao Wang, Yejing Wang, Zijian Zhang, Yuqi Sun, Xiang Li, Maolin Wang, Pengyue Jia, Chong Chen, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.13432)  

**Abstract**: Large Language Model (LLM) has transformative potential in various domains, including recommender systems (RS). There have been a handful of research that focuses on empowering the RS by LLM. However, previous efforts mainly focus on LLM as RS, which may face the challenge of intolerant inference costs by LLM. Recently, the integration of LLM into RS, known as LLM-Enhanced Recommender Systems (LLMERS), has garnered significant interest due to its potential to address latency and memory constraints in real-world applications. This paper presents a comprehensive survey of the latest research efforts aimed at leveraging LLM to enhance RS capabilities. We identify a critical shift in the field with the move towards incorporating LLM into the online system, notably by avoiding their use during inference. Our survey categorizes the existing LLMERS approaches into three primary types based on the component of the RS model being augmented: Knowledge Enhancement, Interaction Enhancement, and Model Enhancement. We provide an in-depth analysis of each category, discussing the methodologies, challenges, and contributions of recent studies. Furthermore, we highlight several promising research directions that could further advance the field of LLMERS. 

**Abstract (ZH)**: 大型语言模型（LLM）在各个领域具有变革性的潜力，包括推荐系统（RS）。已有少数研究专注于通过LLM增强RS。然而，以往的努力主要集中在将LLM作为RS，这可能会面临LLM无法承受的推理成本挑战。最近，将LLM集成到RS中的做法，被称为LLM增强推荐系统（LLMERS），由于其在解决实际应用中的延迟和内存约束方面的潜力，引起了广泛关注。本文对旨在利用LLM增强RS能力的最新研究进行了全面综述。我们发现领域内存在一个关键的转变，即转向将LLM纳入在线系统，特别是在推理过程中避免使用LLM。我们的综述将现有的LLMERS方法按RS模型增强的组件划分为三种主要类型：知识增强、交互增强和模型增强。我们对每种类型进行了深入分析，讨论了最近研究的方法、挑战和贡献。此外，我们还强调了一些有前景的研究方向，这些方向有助于进一步推动LLMERS领域的发展。 

---
