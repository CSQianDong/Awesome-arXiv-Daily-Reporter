# Nano-ESG: Extracting Corporate Sustainability Information from News Articles 

**Title (ZH)**: 纳米ESG：从新闻文章中提取企业可持续性信息 

**Authors**: Fabian Billert, Stefan Conrad  

**Link**: [PDF](https://arxiv.org/pdf/2412.15093)  

**Abstract**: Determining the sustainability impact of companies is a highly complex subject which has garnered more and more attention over the past few years. Today, investors largely rely on sustainability-ratings from established rating-providers in order to analyze how responsibly a company acts. However, those ratings have recently been criticized for being hard to understand and nearly impossible to reproduce.
An independent way to find out about the sustainability practices of companies lies in the rich landscape of news article data. In this paper, we explore a different approach to identify key opportunities and challenges of companies in the sustainability domain. We present a novel dataset of more than 840,000 news articles which were gathered for major German companies between January 2023 and September 2024. By applying a mixture of Natural Language Processing techniques, we first identify relevant articles, before summarizing them and extracting their sustainability-related sentiment and aspect using Large Language Models (LLMs). Furthermore, we conduct an evaluation of the obtained data and determine that the LLM-produced answers are accurate. We release both datasets at this https URL. 

**Abstract (ZH)**: 确定公司可持续性影响是一项高度复杂的研究课题，近年来越来越受到关注。如今，投资者越来越多地依赖于可持续性评级机构提供的评级数据，以分析公司的行为是否负责任。然而，这些评级最近受到了批评，被认为是难以理解且难以复现的。

从丰富的新闻文章数据中了解公司的可持续性实践提供了一种独立的方法。在本文中，我们探讨了一种不同的方法来识别公司在可持续性领域的关键机遇与挑战。我们汇集了一个包含超过840,000篇新闻文章的新数据集，这些文章是为2023年1月至2024年9月期间的主要德国公司收集的。通过应用多种自然语言处理技术，我们首先识别出相关文章，然后对其进行总结，并使用大规模语言模型（LLMs）提取其可持续性相关的观点和方面。此外，我们对该数据进行了评估，并确定LLM生成的答案是准确的。我们将这两个数据集公开发布在如下链接：[在此处插入链接]。 

---
# DisCo: Graph-Based Disentangled Contrastive Learning for Cold-Start Cross-Domain Recommendation 

**Title (ZH)**: DisCo：基于图的解耦对比学习在冷启动跨域推荐中的应用 

**Authors**: Hourun Li, Yifan Wang, Zhiping Xiao, Jia Yang, Changling Zhou, Ming Zhang, Wei Ju  

**Link**: [PDF](https://arxiv.org/pdf/2412.15005)  

**Abstract**: Recommender systems are widely used in various real-world applications, but they often encounter the persistent challenge of the user cold-start problem. Cross-domain recommendation (CDR), which leverages user interactions from one domain to improve prediction performance in another, has emerged as a promising solution. However, users with similar preferences in the source domain may exhibit different interests in the target domain. Therefore, directly transferring embeddings may introduce irrelevant source-domain collaborative information. In this paper, we propose a novel graph-based disentangled contrastive learning framework to capture fine-grained user intent and filter out irrelevant collaborative information, thereby avoiding negative transfer. Specifically, for each domain, we use a multi-channel graph encoder to capture diverse user intents. We then construct the affinity graph in the embedding space and perform multi-step random walks to capture high-order user similarity relationships. Treating one domain as the target, we propose a disentangled intent-wise contrastive learning approach, guided by user similarity, to refine the bridging of user intents across domains. Extensive experiments on four benchmark CDR datasets demonstrate that DisCo consistently outperforms existing state-of-the-art baselines, thereby validating the effectiveness of both DisCo and its components. 

**Abstract (ZH)**: 推荐系统在各种实际应用中得到了广泛应用，但它们经常面临用户冷启动问题的持续挑战。跨域推荐（CDR），即利用一个领域内的用户交互来提高另一个领域内预测性能，已被证明是一种有前景的解决方案。然而，源领域中具有相似偏好用户的兴趣在目标领域中可能截然不同。因此，直接传输嵌入可能会引入无关的源领域协作信息。在本文中，我们提出了一种新颖的基于图的解耦对比学习框架，以捕捉细微的用户意图并过滤掉无关的协作信息，从而避免负迁移。具体来说，对于每个领域，我们采用多通道图编码器来捕捉多样化的用户意图。然后在嵌入空间中构建亲和图，并执行多步随机游走以捕捉高级别的用户相似关系。将一个领域视为目标域，我们提出了一种基于用户相似性的解耦意图对比学习方法，以细粒度地调整跨领域的用户意图桥梁。在四个基准CDR数据集上的广泛实验表明，DisCo始终优于现有最先进的基线模型，从而验证了DisCo及其组成部分的有效性。 

---
# Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation 

**Title (ZH)**: 基于频谱的模态表示融合图卷积网络在多模态推荐中的应用 

**Authors**: Rongqing Kenneth Ong, Andy W. H. Khong  

**Link**: [PDF](https://arxiv.org/pdf/2412.14978)  

**Abstract**: Incorporating multi-modal features as side information has recently become a trend in recommender systems. To elucidate user-item preferences, recent studies focus on fusing modalities via concatenation, element-wise sum, or attention mechanisms. Despite having notable success, existing approaches do not account for the modality-specific noise encapsulated within each modality. As a result, direct fusion of modalities will lead to the amplification of cross-modality noise. Moreover, the variation of noise that is unique within each modality results in noise alleviation and fusion being more challenging. In this work, we propose a new Spectrum-based Modality Representation (SMORE) fusion graph recommender that aims to capture both uni-modal and fusion preferences while simultaneously suppressing modality noise. Specifically, SMORE projects the multi-modal features into the frequency domain and leverages the spectral space for fusion. To reduce dynamic contamination that is unique to each modality, we introduce a filter to attenuate and suppress the modality noise adaptively while capturing the universal modality patterns effectively. Furthermore, we explore the item latent structures by designing a new multi-modal graph learning module to capture associative semantic correlations and universal fusion patterns among similar items. Finally, we formulate a new modality-aware preference module, which infuses behavioral features and balances the uni- and multi-modal features for precise preference modeling. This empowers SMORE with the ability to infer both user modality-specific and fusion preferences more accurately. Experiments on three real-world datasets show the efficacy of our proposed model. The source code for this work has been made publicly available at this https URL. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

近年来，将多模态特征作为辅助信息在推荐系统中变得越来越流行。为阐明用户-项目的偏好，最近的研究主要集中在通过串联、元素级求和或注意力机制融合模态。尽管这种方法已经取得显著的成功，但现有的方法并未考虑每个模态中包含的特定噪声。因此，直接融合模态会导致跨模态噪声的放大。此外，每个模态内部特有的噪声变化使得噪声抑制和融合变得更加困难。在本项工作中，我们提出了一种新的基于频谱的模态表示（Spectrum-based Modality Representation, SMORE）融合图推荐模型，旨在捕捉单模态和融合偏好同时抑制模态噪声。具体而言，SMORE将多模态特征投影到频域，并利用频谱空间进行融合。为了减少每个模态特有的动态污染，我们引入了一个滤波器来适应性地衰减和抑制模态噪声，同时有选择地捕获通用模态模式。此外，我们通过设计一种新的多模态图学习模块，探索项目的潜在结构，以捕获类似项目间的关联语义联系和通用融合模式。最后，我们提出了一个模态感知偏好模块，该模块融合行为特征并平衡单模态和多模态特征，以进行精确的偏好建模。这赋予了SMORE更准确地推断用户模态特定和融合偏好的能力。在三个真实数据集上的实验表明了我们所提出模型的有效性。该项目的源代码已在此处公开：https URL。 

---
# ECLIPSE: Contrastive Dimension Importance Estimation with Pseudo-Irrelevance Feedback for Dense Retrieval 

**Title (ZH)**: ECLIPSE：基于伪无关反馈的对比维度重要性估计在密集检索中的应用 

**Authors**: Giulio D'Erasmo, Giovanni Trappolini, Nicola Tonellotto, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2412.14967)  

**Abstract**: Recent advances in Information Retrieval have leveraged high-dimensional embedding spaces to improve the retrieval of relevant documents. Moreover, the Manifold Clustering Hypothesis suggests that despite these high-dimensional representations, documents relevant to a query reside on a lower-dimensional, query-dependent manifold. While this hypothesis has inspired new retrieval methods, existing approaches still face challenges in effectively separating non-relevant information from relevant signals. We propose a novel methodology that addresses these limitations by leveraging information from both relevant and non-relevant documents. Our method, ECLIPSE, computes a centroid based on irrelevant documents as a reference to estimate noisy dimensions present in relevant ones, enhancing retrieval performance. Extensive experiments on three in-domain and one out-of-domain benchmarks demonstrate an average improvement of up to 19.50% (resp. 22.35%) in mAP(AP) and 11.42% (resp. 13.10%) in nDCG@10 w.r.t. the DIME-based baseline (resp. the baseline using all dimensions). Our results pave the way for more robust, pseudo-irrelevance-based retrieval systems in future IR research. 

**Abstract (ZH)**: 近年来，信息检索领域的最新进展利用高维嵌入空间来提高相关文档的检索效果。此外，流形聚类假设表明，在高维表示下，与查询相关的文档仍位于一个由查询决定的较低维度的流形上。尽管这一假设激发了新的检索方法，但现有方法在有效地分离不相关信息并突出相关信号方面仍面临挑战。我们提出了一种新的方法，通过利用相关和不相关信息之间的信息来解决这些限制。我们的方法ECLIPSE基于无关文档计算一个质心，用作参考来估计相关文档中存在的噪声维度，从而提高检索性能。在三个领域内和一个领域外基准上的大量实验表明，与基于DIME的基线（分别）相比，ECLIPSE在mAP（AP）方面的平均改进幅度为19.50%（分别）和nDCG@10方面的改进幅度为11.42%（分别）。我们的结果为未来的信息检索研究铺平了基于伪无关信息的更稳健检索系统的道路。 

---
# Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models 

**Title (ZH)**: 滑动窗口并非终点：探索长上下文大型语言模型的全面排序能力 

**Authors**: Wenhan Liu, Xinyu Ma, Yutao Zhu, Ziliang Zhao, Shuaiqiang Wang, Dawei Yin, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14574)  

**Abstract**: Large Language Models (LLMs) have shown exciting performance in listwise passage ranking. Due to the limited input length, existing methods often adopt the sliding window strategy. Such a strategy, though effective, is inefficient as it involves repetitive and serialized processing, which usually re-evaluates relevant passages multiple times. As a result, it incurs redundant API costs, which are proportional to the number of inference tokens. The development of long-context LLMs enables the full ranking of all passages within a single inference, avoiding redundant API costs. In this paper, we conduct a comprehensive study of long-context LLMs for ranking tasks in terms of efficiency and effectiveness. Surprisingly, our experiments reveal that full ranking with long-context LLMs can deliver superior performance in the supervised fine-tuning setting with a huge efficiency improvement. Furthermore, we identify two limitations of fine-tuning the full ranking model based on existing methods: (1) sliding window strategy fails to produce a full ranking list as a training label, and (2) the language modeling loss cannot emphasize top-ranked passage IDs in the label. To alleviate these issues, we propose a new complete listwise label construction approach and a novel importance-aware learning objective for full ranking. Experiments show the superior performance of our method over baselines. Our codes are available at \url{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在列表式段落排序任务中表现出了令人兴奋的成绩。由于输入长度有限，现有方法通常采用滑动窗口策略。尽管该策略有效，但它效率低下，因为涉及重复的序列化处理，通常会多次评估相关的段落。结果，这导致了冗余的API成本，这些成本与推理 token 的数量成正比。长语境LLMs的发展使得可以在单次推理中完整排序所有段落，从而避免了冗余的API成本。在这篇论文中，我们对长语境LLMs在效率和有效性方面对排序任务进行了全面研究。令人惊讶的是，我们的实验表明，在监督微调设置中，使用长语境LLMs进行全面排序可以实现优越的性能，并显著提高效率。此外，我们识别出基于现有方法微调全面排序模型的两个限制：(1) 滑动窗口策略无法生成完整的排序列表作为训练标签，(2) 语言模型损失不能在标签中强调排名靠前的段落ID。为了解决这些问题，我们提出了一种新的完整列表式标签构建方法和一种新颖的重要性感知学习目标，用于全面排序。实验结果显示，我们的方法在基线方法上表现出优越的性能。我们的代码已发布在 \url{此链接}。 

---
# HEC-GCN: Hypergraph Enhanced Cascading Graph Convolution Network for Multi-Behavior Recommendation 

**Title (ZH)**: .HEC-GCN：基于超图增强的级联图卷积网络多行为推荐 

**Authors**: Yabo Yin, Xiaofei Zhu, Wenshan Wang, Yihao Zhang, Pengfei Wang, Yixing Fan, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2412.14476)  

**Abstract**: Multi-behavior recommendation (MBR) has garnered growing attention recently due to its ability to mitigate the sparsity issue by inferring user preferences from various auxiliary behaviors to improve predictions for the target behavior. Although existing research on MBR has yielded impressive results, they still face two major limitations. First, previous methods mainly focus on modeling fine-grained interaction information between users and items under each behavior, which may suffer from sparsity issue. Second, existing models usually concentrate on exploiting dependencies between two consecutive behaviors, leaving intra- and inter-behavior consistency largely unexplored. To the end, we propose a novel approach named Hypergraph Enhanced Cascading Graph Convolution Network for multi-behavior recommendation (HEC-GCN). To be specific, we first explore both fine- and coarse-grained correlations among users or items of each behavior by simultaneously modeling the behavior-specific interaction graph and its corresponding hypergraph in a cascaded manner. Then, we propose a behavior consistency-guided alignment strategy that ensures consistent representations between the interaction graph and its associated hypergraph for each behavior, while also maintaining representation consistency across different behaviors. Extensive experiments and analyses on three public benchmark datasets demonstrate that our proposed approach is consistently superior to previous state-of-the-art methods due to its capability to effectively attenuate the sparsity issue as well as preserve both intra- and inter-behavior consistencies. The code is available at this https URL. 

**Abstract (ZH)**: 多行为推荐（MBR）近年来因其能够通过从各种辅助行为推断用户偏好来减轻目标行为预测中的稀疏性问题而受到了越来越多的关注。尽管现有的MBR研究取得了令人印象深刻的成果，但仍面临两大主要限制。首先，现有方法主要关注在每种行为下用户与物品之间的细粒度交互信息建模，这可能导致稀疏性问题。其次，现有模型通常侧重于利用两种连续行为之间的依赖关系，而忽略内部和跨行为一致性。为解决这些问题，我们提出了一种名为Hypergraph Enhanced Cascading Graph Convolution Network for Multi-behavior Recommendation（HEC-GCN）的新颖方法。具体而言，我们首先通过同时建模行为特定交互图及其对应的超图来逐步探索每种行为中用户或物品间的细粒度和粗粒度相关性。然后，我们提出了一种行为一致性引导的对齐策略，确保交互图及其相关超图之间的统一表示，同时在不同行为之间保持表示一致性。在三个公共基准数据集上进行的广泛实验和分析表明，我们的方法由于能够有效减轻稀疏性问题并保持内部和跨行为一致性，从而在所有方面都优于现有的领先方法。代码可在以下网址获取：[此链接]。 

---
# VISA: Retrieval Augmented Generation with Visual Source Attribution 

**Title (ZH)**: VISA：带有视觉来源归因的检索增强生成 

**Authors**: Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Wenhu Chen, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14457)  

**Abstract**: Generation with source attribution is important for enhancing the verifiability of retrieval-augmented generation (RAG) systems. However, existing approaches in RAG primarily link generated content to document-level references, making it challenging for users to locate evidence among multiple content-rich retrieved documents. To address this challenge, we propose Retrieval-Augmented Generation with Visual Source Attribution (VISA), a novel approach that combines answer generation with visual source attribution. Leveraging large vision-language models (VLMs), VISA identifies the evidence and highlights the exact regions that support the generated answers with bounding boxes in the retrieved document screenshots. To evaluate its effectiveness, we curated two datasets: Wiki-VISA, based on crawled Wikipedia webpage screenshots, and Paper-VISA, derived from PubLayNet and tailored to the medical domain. Experimental results demonstrate the effectiveness of VISA for visual source attribution on documents' original look, as well as highlighting the challenges for improvement. Code, data, and model checkpoints will be released. 

**Abstract (ZH)**: 生成并配以来源归属性的方法对于增强检索增强生成（RAG）系统的可验证性至关重要。然而，现有的RAG方法主要将生成的内容与文档级别的引用关联起来，使得用户在多个内容丰富的检索文档中难以定位证据。为了解决这一挑战，我们提出了结合回答生成与视觉来源归属性的检索增强生成方法（Retrieval-Augmented Generation with Visual Source Attribution, VISA）。VISA 利用大规模的视觉语言模型（VLMs），通过在检索文档的屏幕截图中使用边界框突出显示支持生成答案的具体证据和区域，实现了视觉来源归属性。为了评估其有效性，我们精心制作了两个数据集：基于爬取的Wikipedia网页屏幕截图构建的Wiki-VISA，以及基于PubLayNet并针对医学领域定制的Paper-VISA。实验结果表明，VISA 在保留文档原始外观的同时，能够有效地进行视觉来源归属性，同时也指出了改进的挑战。我们将发布代码、数据和模型检查点。 

---
# Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems 

**Title (ZH)**: 较长的提示是否总是更好的选择？大型语言模型在推荐系统中的prompt选择研究 

**Authors**: Genki Kusano, Kosuke Akimoto, Kunihiro Takeoka  

**Link**: [PDF](https://arxiv.org/pdf/2412.14454)  

**Abstract**: In large language models (LLM)-based recommendation systems (LLM-RSs), accurately predicting user preferences by leveraging the general knowledge of LLMs is possible without requiring extensive training data. By converting recommendation tasks into natural language inputs called prompts, LLM-RSs can efficiently solve issues that have been difficult to address due to data scarcity but are crucial in applications such as cold-start and cross-domain problems. However, when applying this in practice, selecting the prompt that matches tasks and data is essential. Although numerous prompts have been proposed in LLM-RSs and representing the target user in prompts significantly impacts recommendation accuracy, there are still no clear guidelines for selecting specific prompts.
In this paper, we categorize and analyze prompts from previous research to establish practical prompt selection guidelines. Through 450 experiments with 90 prompts and five real-world datasets, we examined the relationship between prompts and dataset characteristics in recommendation accuracy. We found that no single prompt consistently outperforms others; thus, selecting prompts on the basis of dataset characteristics is crucial. Here, we propose a prompt selection method that achieves higher accuracy with minimal validation data. Because increasing the number of prompts to explore raises costs, we also introduce a cost-efficient strategy using high-performance and cost-efficient LLMs, significantly reducing exploration costs while maintaining high prediction accuracy. Our work offers valuable insights into the prompt selection, advancing accurate and efficient LLM-RSs. 

**Abstract (ZH)**: 在基于大语言模型（LLM）的推荐系统（LLM-RS）中，通过利用LLM的普遍知识来准确预测用户偏好是可能的，无需大量训练数据。通过将推荐任务转化为称为提示的自然语言输入，LLM-RS可以有效地解决由于数据稀缺但对实际应用至关重要的问题，如冷启动和跨域问题。然而，在实践中应用这种方法时，选择与任务和数据匹配的提示至关重要。尽管在LLM-RS中提出了大量的提示，且在提示中代表目标用户显著影响推荐准确度，但仍然缺乏选择具体提示的明确指导方针。

本文对之前研究中的提示进行分类和分析，以建立实用的提示选择指南。通过使用90种提示和五个实际数据集进行450次实验，我们探究了提示与数据集特征之间的关系对推荐准确度的影响。我们发现没有一种提示在所有情况下都表现最优；因此，根据数据集特征选择提示至关重要。在此基础上，我们提出了一种新的提示选择方法，该方法能够在少量验证数据的情况下实现更高的准确度。由于增加提示的数量以进行探索会增加成本，我们还引入了一种高效的成本策略，利用高性能且成本效益高的LLM，大幅降低探索成本，同时保持高预测准确度。本研究为提示选择提供了有价值的见解，促进了准确高效的LLM-RS的发展。 

---
# ChainRank-DPO: Chain Rank Direct Preference Optimization for LLM Rankers 

**Title (ZH)**: ChainRank-DPO：链式排序直接偏好优化算法 vatandaşlar ve onların hakları konusunda daha derin bir bilgi edinme ve tartışmacompanions and their rights concerning chainrank-dpo: 链式排序直接偏好优化算法中的同伴及其权利探讨

这里的翻译需要稍微解释一下：

1. "ChainRank-DPO" 是一个具体的算法名称，保持不变。

2. "companions" 在这里翻译为“同伴”，与原文内容相对应。

3. "and their rights concerning chainrank-dpo" 翻译为“及其在链式排序直接偏好优化算法中的权利”，以确保语义准确。

若原文意图是指某种与CHAINRANK-DPO算法相关的权利问题或伦理考量，可进一步调整为：

"ChainRank-DPO中的权利问题：同伴及其在该算法中的权益"

这样更符合学术论文标题的要求。 

**Authors**: Haowei Liu, Xuyang Wu, Guohao Sun, Zhiqiang Tao, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14405)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable effectiveness in text reranking through works like RankGPT, leveraging their human-like reasoning about relevance. However, supervised fine-tuning for ranking often diminishes these models' general-purpose capabilities, including the crucial reasoning abilities that make them valuable for ranking. We introduce a novel approach integrating Chain-of-Thought prompting with an SFT-DPO (Supervised Fine-Tuning followed by Direct Preference Optimization) pipeline to preserve these capabilities while improving ranking performance. Our experiments on TREC 2019 and 2020 Deep Learning datasets show that our approach outperforms the state-of-the-art RankZephyr while maintaining strong performance on the Massive Multitask Language Understanding (MMLU) benchmark, demonstrating effective preservation of general-purpose capabilities through thoughtful fine-tuning strategies. Our code and data will be publicly released upon the acceptance of the paper. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通过如RankGPT等研究工作中的文本重排名任务中展现出了显著的效果，得益于它们在相关性方面的类人推理能力。然而，针对排序任务的监督微调往往会削弱这些模型的一般化能力，包括使它们在排序任务中具有价值的重要推理能力。我们提出了一种新的方法，将Chain-of-Thought提示与SFT-DPO（监督微调后直接偏好优化）管道相结合，以在保持这些能力的同时提升排序性能。我们的实验在TREC 2019和2020年深度学习数据集上表明，我们的方法在性能上优于最新的RankZephyr，同时在大规模多任务语言理解（MMLU）基准测试中保持了强大的性能，证明了通过精心设计的微调策略有效地保留了这些一般化能力。论文被接受后，我们将公开发布我们的代码和数据。 

---
# Embedding Cultural Diversity in Prototype-based Recommender Systems 

**Title (ZH)**: 将文化多样性融入原型推荐系统中 

**Authors**: Armin Moradi, Nicola Neophytou, Florian Carichon, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2412.14329)  

**Abstract**: Popularity bias in recommender systems can increase cultural overrepresentation by favoring norms from dominant cultures and marginalizing underrepresented groups. This issue is critical for platforms offering cultural products, as they influence consumption patterns and human perceptions. In this work, we address popularity bias by identifying demographic biases within prototype-based matrix factorization methods. Using the country of origin as a proxy for cultural identity, we link this demographic attribute to popularity bias by refining the embedding space learning process. First, we propose filtering out irrelevant prototypes to improve representativity. Second, we introduce a regularization technique to enforce a uniform distribution of prototypes within the embedding space. Across four datasets, our results demonstrate a 27\% reduction in the average rank of long-tail items and a 2\% reduction in the average rank of items from underrepresented countries. Additionally, our model achieves a 2\% improvement in HitRatio@10 compared to the state-of-the-art, highlighting that fairness is enhanced without compromising recommendation quality. Moreover, the distribution of prototypes leads to more inclusive explanations by better aligning items with diverse prototypes. 

**Abstract (ZH)**: 推荐系统中的流行度偏差可能会通过偏好占主导地位文化的规范并边缘化被代表不足的群体，从而加剧文化过度代表。这一问题对于提供文化产品的平台来说至关重要，因为这些平台会影响消费模式和人类的认知。在本研究中，我们通过识别基于原型矩阵分解方法中的人口统计学偏差来解决流行度偏差问题。我们以原产地作为文化身份的代理指标，通过细化嵌入空间的学习过程将人口统计学属性与流行度偏差联系起来。首先，我们提出过滤掉无关的原型以提高代表性。其次，我们引入了一种正则化技术以在嵌入空间内强制实现原型分布的一致性。在四个数据集上，我们的结果表明平均排名尾部项目的排名降低了27%，而来自被代表不足国家的项目的平均排名降低了2%。此外，与最先进的模型相比，我们的模型在HitRatio@10上提高了2%，表明公平性得以提升而不牺牲推荐质量。而且，原型分布有助于提供更加包容性解释，因为它更好地将项目与多种原型对齐。 

---
# SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation 

**Title (ZH)**: SAFERec：自我注意与频率增强模型在下一个购物车推荐中的应用 

**Authors**: Oleg Lashinin, Denis Krasilnikov, Aleksandr Milogradskii, Marina Ananyeva  

**Link**: [PDF](https://arxiv.org/pdf/2412.14302)  

**Abstract**: Transformer-based approaches such as BERT4Rec and SASRec demonstrate strong performance in Next Item Recommendation (NIR) tasks. However, applying these architectures to Next-Basket Recommendation (NBR) tasks, which often involve highly repetitive interactions, is challenging due to the vast number of possible item combinations in a basket. Moreover, frequency-based methods such as TIFU-KNN and UP-CF still demonstrate strong performance in NBR tasks, frequently outperforming deep-learning approaches. This paper introduces SAFERec, a novel algorithm for NBR that enhances transformer-based architectures from NIR by incorporating item frequency information, consequently improving their applicability to NBR tasks. Extensive experiments on multiple datasets show that SAFERec outperforms all other baselines, specifically achieving an 8\% improvement in Recall@10. 

**Abstract (ZH)**: 基于Transformer的方法，如BERT4Rec和SASRec，在Next Item Recommendation (NIR)任务中表现出很强的性能。然而，将这些架构应用于Next-Basket Recommendation (NBR)任务中是具有挑战性的，因为篮子中可能包含大量的项目组合，导致高度重复的交互。此外，基于频率的方法，如TIFU-KNN和UP-CF，在NBR任务中仍然表现出很强的性能，经常优于深度学习方法。本文提出了一种名为SAFERec的新算法，通过结合项目频率信息来增强NIR中的Transformer架构，从而改善它们在NBR任务中的适用性。在多个数据集上的广泛实验表明，SAFERec在所有基准方法中表现出优越性，特别是在Recall@10方面提高了8%。 

---
# Progressive Multimodal Reasoning via Active Retrieval 

**Title (ZH)**: 基于主动检索的 progressive 多模态推理 

**Authors**: Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14835)  

**Abstract**: Multi-step multimodal reasoning tasks pose significant challenges for multimodal large language models (MLLMs), and finding effective ways to enhance their performance in such scenarios remains an unresolved issue. In this paper, we propose AR-MCTS, a universal framework designed to progressively improve the reasoning capabilities of MLLMs through Active Retrieval (AR) and Monte Carlo Tree Search (MCTS). Our approach begins with the development of a unified retrieval module that retrieves key supporting insights for solving complex reasoning problems from a hybrid-modal retrieval corpus. To bridge the gap in automated multimodal reasoning verification, we employ the MCTS algorithm combined with an active retrieval mechanism, which enables the automatic generation of step-wise annotations. This strategy dynamically retrieves key insights for each reasoning step, moving beyond traditional beam search sampling to improve the diversity and reliability of the reasoning space. Additionally, we introduce a process reward model that aligns progressively to support the automatic verification of multimodal reasoning tasks. Experimental results across three complex multimodal reasoning benchmarks confirm the effectiveness of the AR-MCTS framework in enhancing the performance of various multimodal models. Further analysis demonstrates that AR-MCTS can optimize sampling diversity and accuracy, yielding reliable multimodal reasoning. 

**Abstract (ZH)**: 多步多模态推理任务对多模态大规模语言模型（MLLMs）构成了显著挑战，如何在这种场景下有效提升其性能仍然是一个未解决的问题。本文提出了一种名为AR-MCTS的通用框架，旨在通过主动检索（AR）和蒙特卡罗树搜索（MCTS）逐步提高MLLMs的推理能力。本文方法首先开发了一个统一的检索模块，从混合模态检索库中获取解决复杂推理问题的关键支持见解。为克服自动化多模态推理验证中的差距，我们结合使用了MCTS算法和主动检索机制，这使得自动生成逐步注解成为可能。该策略动态地为每个推理步骤检索关键见解，超越了传统的束搜索采样方法，从而改善了推理空间的多样性和可靠性。此外，我们引入了一个过程奖励模型，以逐步支持多模态推理任务的自动验证。在三个复杂多模态推理基准上的实验结果证实了AR-MCTS框架在提高各种多模态模型性能方面的有效性。进一步的分析表明，AR-MCTS可以优化采样多样性和准确性，从而实现可靠的多模态推理。 

---
# Efficient Self-Supervised Video Hashing with Selective State Spaces 

**Title (ZH)**: 高效的自监督视频哈希方法：选择性状态空间方法 

**Authors**: Jinpeng Wang, Niu Lian, Jun Li, Yuting Wang, Yan Feng, Bin Chen, Yongbing Zhang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2412.14518)  

**Abstract**: Self-supervised video hashing (SSVH) is a practical task in video indexing and retrieval. Although Transformers are predominant in SSVH for their impressive temporal modeling capabilities, they often suffer from computational and memory inefficiencies. Drawing inspiration from Mamba, an advanced state-space model, we explore its potential in SSVH to achieve a better balance between efficacy and efficiency. We introduce S5VH, a Mamba-based video hashing model with an improved self-supervised learning paradigm. Specifically, we design bidirectional Mamba layers for both the encoder and decoder, which are effective and efficient in capturing temporal relationships thanks to the data-dependent selective scanning mechanism with linear complexity. In our learning strategy, we transform global semantics in the feature space into semantically consistent and discriminative hash centers, followed by a center alignment loss as a global learning signal. Our self-local-global (SLG) paradigm significantly improves learning efficiency, leading to faster and better convergence. Extensive experiments demonstrate S5VH's improvements over state-of-the-art methods, superior transferability, and scalable advantages in inference efficiency. Code is available at this https URL. 

**Abstract (ZH)**: 自监督视频哈希（SSVH）是视频索引和检索中的一个实际任务。尽管 Transformer 在 SSVH 中因其出色的时间建模能力而占据主导地位，但它们往往因计算和内存效率低下而受到限制。受 Mamba（一种先进的状态空间模型）的启发，我们探索其在 SSVH 中的潜力，以实现效率和效果之间的更好平衡。我们引入了 S5VH，这是一种基于 Mamba 的视频哈希模型，具有改进的自监督学习范式。具体而言，我们在编码器和解码器中设计了双向的 Mamba 层，这些层通过数据依赖的选择性扫描机制捕获时间关系时既有效又高效，具有线性复杂度。在我们的学习策略中，我们将特征空间中的全局语义转换为语义一致且区分性较强的哈希中心，随后通过中心对齐损失作为全局学习信号。我们的自局部-全局（SLG）范式显著提高了学习效率，导致更快且更好的收敛。大量实验表明，S5VH 在性能上优于现有方法，具有更优的迁移能力和可扩展的优势。推断效率上的优势。相关代码可在以下链接获取：this https URL。 

---
# Moving Beyond LDA: A Comparison of Unsupervised Topic Modelling Techniques for Qualitative Data Analysis of Online Communities 

**Title (ZH)**: 超越LDA：无监督主题建模技术在在线社区定性数据分析中的比较 

**Authors**: Amandeep Kaur, James R. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2412.14486)  

**Abstract**: Social media constitutes a rich and influential source of information for qualitative researchers. Although computational techniques like topic modelling assist with managing the volume and diversity of social media content, qualitative researcher's lack of programming expertise creates a significant barrier to their adoption. In this paper we explore how BERTopic, an advanced Large Language Model (LLM)-based topic modelling technique, can support qualitative data analysis of social media. We conducted interviews and hands-on evaluations in which qualitative researchers compared topics from three modelling techniques: LDA, NMF, and BERTopic. BERTopic was favoured by 8 of 12 participants for its ability to provide detailed, coherent clusters for deeper understanding and actionable insights. Participants also prioritised topic relevance, logical organisation, and the capacity to reveal unexpected relationships within the data. Our findings underscore the potential of LLM-based techniques for supporting qualitative analysis. 

**Abstract (ZH)**: 社交媒体是一种丰富且有影响力的定性研究信息来源。尽管计算技术，如主题建模，有助于管理社交媒体内容的规模和多样性，但缺乏编程技能的定性研究人员在使用这些技术方面面临着重大障碍。本文探讨了如何利用基于大型语言模型（LLM）的主题建模技术BERTopic 支持社交媒体的定性数据分析。我们进行了访谈和实际操作评估，让定性研究人员将三种建模技术（LDA、NMF 和 BERTopic）生成的主题进行了比较。结果显示，12 名参与者中有 8 人更偏好 BERTopic，因为它能够提供详细、连贯的主题聚类，从而有助于深入理解和获取行动性见解。参与者还优先考虑了主题的相关性、逻辑组织以及揭示数据中未预见关系的能力。我们的研究结果突显了基于大型语言模型的方法在支持定性分析方面的潜在价值。 

---
# State Space Models are Strong Text Rerankers 

**Title (ZH)**: 状态空间模型是强大的文本重排序器 

**Authors**: Zhichao Xu, Jinghua Yan, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2412.14354)  

**Abstract**: Transformers dominate NLP and IR; but their inference inefficiencies and challenges in extrapolating to longer contexts have sparked interest in alternative model architectures. Among these, state space models (SSMs) like Mamba offer promising advantages, particularly $O(1)$ time complexity in inference. Despite their potential, SSMs' effectiveness at text reranking -- a task requiring fine-grained query-document interaction and long-context understanding -- remains underexplored.
This study benchmarks SSM-based architectures (specifically, Mamba-1 and Mamba-2) against transformer-based models across various scales, architectures, and pre-training objectives, focusing on performance and efficiency in text reranking tasks. We find that (1) Mamba architectures achieve competitive text ranking performance, comparable to transformer-based models of similar size; (2) they are less efficient in training and inference compared to transformers with flash attention; and (3) Mamba-2 outperforms Mamba-1 in both performance and efficiency. These results underscore the potential of state space models as a transformer alternative and highlight areas for improvement in future IR applications. 

**Abstract (ZH)**: 转换器在自然语言处理（NLP）和信息检索（IR）中占据主导地位；但它们在推理时的低效率以及在处理较长上下文时的外推挑战，引发了对替代模型架构的兴趣。在这之中，状态空间模型（SSMs）如Mamba展现出有希望的优势，尤其是在推理时具有$O(1)$的时间复杂性。尽管如此，SSMs在文本重排任务中的有效性仍然未被充分探索——该任务需要精细的查询-文档交互和长上下文理解。

本研究将基于SSMs的架构（特别是Mamba-1和Mamba-2）与基于转换器的模型进行基准测试，考虑到不同规模、架构和预训练目标，重点关注文本重排任务中的性能和效率。研究结果表明：

1. Mamba架构在文本排名性能上达到了与相似规模的转换器模型相竞争的水平；
2. 在训练和推理效率方面，Mamba不如采用闪存注意机制的转换器模型；
3. Mamba-2在性能和效率上均超过了Mamba-1。

这些结果强调了状态空间模型作为转换器替代方案的潜在价值，并指出了未来信息检索应用中需要改进的领域。 

---
# Transversal PACS Browser API: Addressing Interoperability Challenges in Medical Imaging Systems 

**Title (ZH)**: 横向PACS浏览器API：解决医学成像系统中的互操作性挑战 

**Authors**: Diogo Lameira, Filipa Ferraz  

**Link**: [PDF](https://arxiv.org/pdf/2412.14229)  

**Abstract**: Advances in imaging technologies have revolutionised the medical imaging and healthcare sectors, leading to the widespread adoption of PACS for the storage, retrieval, and communication of medical images. Although these systems have improved operational efficiency, significant challenges remain in effectively retrieving DICOM images, which are essential for diagnosis and overall patient care. Moreover, issues such as fragmented systems, interoperability barriers, and complex user interfaces can often prevent healthcare professionals from efficiently accessing medical images. Addressing these challenges, the Transversal PACS Browser API is a robust and user-friendly solution designed to enhance the process of querying and retrieving DICOM images. It offers advanced filtering capabilities through a variety of filter options as well as a custom field search, that allows users to easily navigate through large medical image collections with ease. Additionally, the application provides a unified interface for querying and retrieving from multiple PACS stations, addressing the challenges of fragmentation and complexity associated with accessing medical images. Other key features include the ability to pre-view images directly within the application. All of this contributes to the transversal nature of the API, serving not only healthcare providers, but anyone who relies on efficient access to these resources. To validate the performance and usability of the application, comprehensive testing was carried out with stakeholders of the field, the results of which showed general satisfaction, highlighting the API's clean design, ease of use, and effective search capabilities of the API, as well as the usefulness of previewing images within the application. 

**Abstract (ZH)**: 影像技术的进步已经彻底改变了医学影像和医疗保健领域，导致广泛采用了PACS系统，用于存储、检索和传输医疗图像。尽管这些系统提高了操作效率，但在有效检索DICOM图像方面仍存在重大挑战，而这些图像对于诊断和整体患者护理至关重要。此外，诸如系统碎片化、互操作性障碍和复杂用户界面等问题常常阻碍医护人员高效访问医疗图像。为应对这些挑战，跨系统PACS浏览器API提供了一种强大且用户友好的解决方案，旨在增强查询和检索DICOM图像的过程。该API通过多种过滤选项和自定义字段搜索提供了高级过滤能力，使用户能够轻松地在大型医疗图像集合中导航。此外，该应用程序还提供了一个统一界面，用于从多个PACS工作站查询和检索数据，解决了访问医疗图像时碎片化和复杂性的难题。其他关键功能包括可以直接在应用程序中预览图像的能力。所有这些都增强了API的横向性质，使其不仅适用于医疗服务提供者，还适用于依赖于这些资源高效访问的任何人。为了验证应用程序的性能和易用性，对其进行了全面测试，测试对象包括该领域的利益相关者。测试结果表明，总体上获得了满意评价，强调了API简洁的设计、易用性和有效查询功能，以及在应用程序中预览图像的实用性。 

---
# Whom do Explanations Serve? A Systematic Literature Survey of User Characteristics in Explainable Recommender Systems Evaluation 

**Title (ZH)**: 《解释型推荐系统评估中用户特征的作用对象是谁？——一项系统的文献综述》

这个标题翻译成中文后，既保留了原意，又符合学术论文标题的规范。 

**Authors**: Kathrin Wardatzky, Oana Inel, Luca Rossetto, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2412.14193)  

**Abstract**: Adding explanations to recommender systems is said to have multiple benefits, such as increasing user trust or system transparency. Previous work from other application areas suggests that specific user characteristics impact the users' perception of the explanation. However, we rarely find this type of evaluation for recommender systems explanations. This paper addresses this gap by surveying 124 papers in which recommender systems explanations were evaluated in user studies. We analyzed their participant descriptions and study results where the impact of user characteristics on the explanation effects was measured. Our findings suggest that the results from the surveyed studies predominantly cover specific users who do not necessarily represent the users of recommender systems in the evaluation domain. This may seriously hamper the generalizability of any insights we may gain from current studies on explanations in recommender systems. We further find inconsistencies in the data reporting, which impacts the reproducibility of the reported results. Hence, we recommend actions to move toward a more inclusive and reproducible evaluation. 

**Abstract (ZH)**: 将推荐系统添加解释被认为具有多方面的好处，例如增加用户信任或提高系统透明度。其他应用领域中的先前研究表明，特定用户特征会影响用户对解释的感知。然而，我们很少在推荐系统的解释评估中发现这一类型的研究。本文通过调查124篇评估推荐系统解释的用户研究论文来填补这一空白。我们分析了这些论文中关于参与者描述和实验结果的记录，其中实验结果衡量了用户特征对解释效果的影响。我们的研究发现，被调查研究的主要结果大多针对特定的用户群体，这些用户并不一定代表评价领域内的推荐系统用户。这可能会严重影响我们从当前关于推荐系统解释的研究中获得的任何认知的普遍适用性。进一步地，我们发现数据报告中存在不一致之处，这影响了所报告结果的可再现性。因此，我们建议采取措施以实现更加包容性和可再现性的评估。 

---
# Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics with Large Language Models 

**Title (ZH)**: 使用大型语言模型在数据-analytics中进行多步洞察综合的先进推理与转换引擎 

**Authors**: Atin Sakkeer Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2412.14146)  

**Abstract**: This paper presents the Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics (ARTEMIS-DA), a novel framework designed to augment Large Language Models (LLMs) for solving complex, multi-step data analytics tasks. ARTEMIS-DA integrates three core components: the Planner, which dissects complex user queries into structured, sequential instructions encompassing data preprocessing, transformation, predictive modeling, and visualization; the Coder, which dynamically generates and executes Python code to implement these instructions; and the Grapher, which interprets generated visualizations to derive actionable insights. By orchestrating the collaboration between these components, ARTEMIS-DA effectively manages sophisticated analytical workflows involving advanced reasoning, multi-step transformations, and synthesis across diverse data modalities. The framework achieves state-of-the-art (SOTA) performance on benchmarks such as WikiTableQuestions and TabFact, demonstrating its ability to tackle intricate analytical tasks with precision and adaptability. By combining the reasoning capabilities of LLMs with automated code generation and execution and visual analysis, ARTEMIS-DA offers a robust, scalable solution for multi-step insight synthesis, addressing a wide range of challenges in data analytics. 

**Abstract (ZH)**: 本文介绍了Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics（ARTEMIS-DA），这是一种新颖的框架，旨在增强大型语言模型（LLMs）以解决复杂的多步数据分析任务。ARTEMIS-DA 集成了三个核心组件：规划器（Planner），它将复杂的用户查询分解为结构化、序列化的指令，涵盖数据预处理、变换、预测建模和可视化；编码器（Coder），它动态生成并执行 Python 代码以实现这些指令；以及图形生成器（Grapher），它解释生成的可视化以得出 actionable 的洞见。通过协调这些组件之间的合作，ARTEMIS-DA 有效管理了涉及高级推理、多步骤变换和跨多种数据模态综合的复杂分析工作流。该框架在 WikiTableQuestions 和 TabFact 等基准测试中实现了最先进的（SOTA）性能，展示了其解决复杂分析任务的精确性和适应性。通过结合 LLM 的推理能力、自动化代码生成与执行以及可视化分析，ARTEMIS-DA 提供了一种稳健且可扩展的解决方案，用于多步洞见综合，解决了数据分析中广泛存在的挑战。 

---
# A Cognitive Ideation Support Framework using IBM Watson Services 

**Title (ZH)**: 使用IBM Watson服务的认知构想支持框架 

**Authors**: Samaa Elnagar, Kweku-Muata Osei-Bryson  

**Link**: [PDF](https://arxiv.org/pdf/2412.14025)  

**Abstract**: Ideas generation is a core activity for innovation in organizations. The creativity of the generated ideas depends not only on the knowledge retrieved from the organizations' knowledge bases, but also on the external knowledge retrieved from other resources. Unfortunately, organizations often cannot efficiently utilize the knowledge in the knowledge bases due to the limited abilities of the search and retrieval mechanisms especially when dealing with unstructured data. In this paper, we present a new cognitive support framework for ideation that uses the IBM Watson DeepQA services. IBM Watson is a Question Answering system which mimics human cognitive abilities to retrieve and rank information. The proposed framework is based on the Search for Ideas in the Associative Memory (SIAM) model to help organizations develop creative ideas through discovering new relationships between retrieved data. To evaluate the effectiveness of the proposed system, the generated ideas generated are selected and assessed using a set of established creativity criteria. 

**Abstract (ZH)**: 创新是组织中的核心活动之一。生成的创意不仅依赖于从组织知识库中检索的知识，还依赖于从其他资源中检索的外部知识。不幸的是，组织往往因搜索和检索机制能力有限，尤其是处理非结构化数据时，难以有效利用知识库中的知识。本文提出了一种基于IBM Watson DeepQA服务的认知支持框架，用于创意生成。IBM Watson是一个问答系统，通过模拟人类的认知能力来检索和排序信息。本文提出的方法基于联想记忆中创意发现模型（SIAM），旨在帮助组织通过发现已检索数据之间的新关系来发展创意。为了评估该系统的有效性，生成的创意被选择并根据一套公认的创造性标准进行评估。 

---
# CRM: Retrieval Model with Controllable Condition 

**Title (ZH)**: CRM：可控条件检索模型 

**Authors**: Chi Liu, Jiangxia Cao, Rui Huang, Kuo Cai, Weifeng Ding, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13844)  

**Abstract**: Recommendation systems (RecSys) are designed to connect users with relevant items from a vast pool of candidates while aligning with the business goals of the platform. A typical industrial RecSys is composed of two main stages, retrieval and ranking: (1) the retrieval stage aims at searching hundreds of item candidates satisfied user interests; (2) based on the retrieved items, the ranking stage aims at selecting the best dozen items by multiple targets estimation for each item candidate, including classification and regression targets. Compared with ranking model, the retrieval model absence of item candidate information during inference, therefore retrieval models are often trained by classification target only (e.g., click-through rate), but failed to incorporate regression target (e.g., the expected watch-time), which limit the effectiveness of retrieval. In this paper, we propose the Controllable Retrieval Model (CRM), which integrates regression information as conditional features into the two-tower retrieval paradigm. This modification enables the retrieval stage could fulfill the target gap with ranking model, enhancing the retrieval model ability to search item candidates satisfied the user interests and condition effectively. We validate the effectiveness of CRM through real-world A/B testing and demonstrate its successful deployment in Kuaishou short-video recommendation system, which serves over 400 million users. 

**Abstract (ZH)**: 推荐系统（RecSys）旨在将用户与海量候选物品中相关的物品进行连接，并与推荐平台的商业目标相一致。一个典型的工业推荐系统通常由两个主要阶段组成：检索和排名：（1）检索阶段旨在搜索满足用户兴趣的数百个候选物品；（2）基于检索出的物品，排名阶段旨在通过多个目标估计每个候选物品的最佳前十几项，包括分类和回归目标。与排名模型相比，检索模型在推断阶段并不包含候选物品的信息，因此检索模型通常仅通过分类目标（例如点击率）进行训练，而未能结合回归目标（例如预计观看时间），这限制了检索模型的有效性。本文中，我们提出了一种可控检索模型（Controllable Retrieval Model，CRM），该模型将回归信息作为条件特征整合到两塔检索框架中。这种修改使得检索阶段能够弥补与排名模型之间的目标差距，增强检索模型在搜索满足用户兴趣和条件的候选物品方面的能力。我们通过实际的A/B测试验证了CRM的有效性，并展示了其在字节跳动短视频推荐系统中的成功部署，该系统服务于超过4亿用户。 

---
# Maybe you are looking for CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval 

**Title (ZH)**: 也许您正在寻找CroQS：跨模态查询建议用于文本到图像检索 

**Authors**: Giacomo Pacini, Fabio Carrara, Nicola Messina, Nicola Tonellotto, Giuseppe Amato, Fabrizio Falchi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13834)  

**Abstract**: Query suggestion, a technique widely adopted in information retrieval, enhances system interactivity and the browsing experience of document collections. In cross-modal retrieval, many works have focused on retrieving relevant items from natural language queries, while few have explored query suggestion solutions. In this work, we address query suggestion in cross-modal retrieval, introducing a novel task that focuses on suggesting minimal textual modifications needed to explore visually consistent subsets of the collection, following the premise of ''Maybe you are looking for''. To facilitate the evaluation and development of methods, we present a tailored benchmark named CroQS. This dataset comprises initial queries, grouped result sets, and human-defined suggested queries for each group. We establish dedicated metrics to rigorously evaluate the performance of various methods on this task, measuring representativeness, cluster specificity, and similarity of the suggested queries to the original ones. Baseline methods from related fields, such as image captioning and content summarization, are adapted for this task to provide reference performance scores. Although relatively far from human performance, our experiments reveal that both LLM-based and captioning-based methods achieve competitive results on CroQS, improving the recall on cluster specificity by more than 115% and representativeness mAP by more than 52% with respect to the initial query. The dataset, the implementation of the baseline methods and the notebooks containing our experiments are available here: this https URL 

**Abstract (ZH)**: 查询建议是一种广泛应用于信息检索的技术，能够增强系统的互动性和文档集合的浏览体验。在跨模态检索中，许多工作都集中在从自然语言查询中检索相关项上，而鲜有研究关注查询建议解决方案。在本次研究中，我们针对跨模态检索中的查询建议问题，引入了一个新的任务，旨在根据“也许您在寻找”的前提，建议最小限度的文本修改，以探索集合中的视觉一致子集。为了促进该任务的评估和发展，我们提出了一个定制基准CroQS。该数据集包含初始查询、分组结果集以及每个分组的人工定义建议查询。我们为该任务建立了专门的评估指标，以严格评估各种方法的性能，测量建议查询的代表性、聚类特异性以及与原始查询的相似度。来自相关领域的基线方法（如图像字幕生成和内容摘要）进行了适应性调整，以提供参考性能分数。尽管与人类性能相比仍有差距，但我们的实验结果显示，基于大语言模型（LLM）的方法和基于字幕的方法都在CroQS上取得了竞争力的结果，将聚类特异性的召回率提高了超过115%，代表性mAP提高了超过52%。相较于初始查询。这一数据集、基线方法的实现以及包含我们实验的笔记本都可以在这里访问：[请点击这里](this https URL)。 

---
# Heterogeneous Graph Collaborative Filtering 

**Title (ZH)**: 异质图协作过滤 

**Authors**: Lianghao Xia, Meiyan Xie, Yong Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13825)  

**Abstract**: For modern recommender systems, the use of low-dimensional latent representations to embed users and items based on their observed interactions has become commonplace. However, many existing recommendation models are primarily designed for coarse-grained and homogeneous interactions, which limits their effectiveness in two critical dimensions. Firstly, these models fail to leverage the relational dependencies that exist across different types of user behaviors, such as page views, collects, comments, and purchases. Secondly, they struggle to capture the fine-grained latent factors that drive user interaction patterns. To address these limitations, we present a heterogeneous graph collaborative filtering model MixRec that excels at disentangling users' multi-behavior interaction patterns and uncovering the latent intent factors behind each behavior. Our model achieves this by incorporating intent disentanglement and multi-behavior modeling, facilitated by a parameterized heterogeneous hypergraph architecture. Furthermore, we introduce a novel contrastive learning paradigm that adaptively explores the advantages of self-supervised data augmentation, thereby enhancing the model's resilience against data sparsity and expressiveness with relation heterogeneity. To validate the efficacy of MixRec, we conducted extensive experiments on three public datasets. The results clearly demonstrate its superior performance, significantly outperforming various state-of-the-art baselines. Our model is open-sourced and available at: this https URL. 

**Abstract (ZH)**: 对于现代推荐系统而言，基于用户观察到的交互行为使用低维度潜在表示来嵌入用户和物品已经成为常态。然而，现有的许多推荐模型主要是为粗粒度和同质的交互行为设计的，这在两个关键维度上限制了它们的效果。首先，这些模型无法充分利用不同类型的用户行为之间的关系依赖性，如页面浏览、收藏、评论和购买行为。其次，它们难以捕捉驱动用户交互模式的细粒度潜在因子。为了解决这些局限性，我们提出了一种异构图协同过滤模型MixRec，该模型能够有效地将用户的多行为交互模式解耦，并揭示每种行为背后的潜在意图因子。我们的模型通过一个参数化的异构超图架构来进行意图解耦和多行为建模。此外，我们引入了一种新颖的对比学习范式，该范式能够自适应地探索自监督数据增强的优势，从而增强模型对数据稀疏性和关系异质性的鲁棒性和表达能力。为了验证MixRec的有效性，我们在三个公开数据集上进行了广泛的实验。实验结果清楚地表明，MixRec表现出色，显著优于多种最新的基线模型。我们的模型已开源，并可在以下链接获取：[this https URL]。 

---
# Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization 

**Title (ZH)**: 语义收敛：通过两阶段对齐和行为语义词素化 harmonize 推荐系统 

**Authors**: Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13771)  

**Abstract**: Large language models (LLMs), endowed with exceptional reasoning capabilities, are adept at discerning profound user interests from historical behaviors, thereby presenting a promising avenue for the advancement of recommendation systems. However, a notable discrepancy persists between the sparse collaborative semantics typically found in recommendation systems and the dense token representations within LLMs. In our study, we propose a novel framework that harmoniously merges traditional recommendation models with the prowess of LLMs. We initiate this integration by transforming ItemIDs into sequences that align semantically with the LLMs space, through the proposed Alignment Tokenization module. Additionally, we design a series of specialized supervised learning tasks aimed at aligning collaborative signals with the subtleties of natural language semantics. To ensure practical applicability, we optimize online inference by pre-caching the top-K results for each user, reducing latency and improving effciency. Extensive experimental evidence indicates that our model markedly improves recall metrics and displays remarkable scalability of recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）凭借出色的推理能力，能够从用户的历史行为中识别出深层次的兴趣，从而为推荐系统的进步提供了新的机遇。然而，推荐系统中常见的稀疏协作语义与LLMs中密集的令牌表示之间存在显著的差距。在本研究中，我们提出了一种新型框架，旨在传统推荐模型与LLMs能力之间实现和谐的融合。我们通过引入“对齐分词”模块，将ItemIDs转化为与LLMs空间相匹配的序列，从而启动这一整合过程。此外，我们设计了一系列专门的监督学习任务，旨在使协作信号与自然语言语义的细微差别对齐。为了确保其实用性，我们通过预缓存每个用户的Top-K结果来优化在线推理过程，从而降低延迟并提高效率。广泛的实验证据表明，我们的模型显著提高了召回率指标，并展示了推荐系统的优异扩展性。 

---
# Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models 

**Title (ZH)**: 用大型语言模型弥合知识感知推荐中的用户侧知识差距 

**Authors**: Zheng Hu, Zhe Li, Ziyun Jiao, Satoshi Nakagawa, Jiawen Deng, Shimin Cai, Tao Zhou, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.13544)  

**Abstract**: In recent years, knowledge graphs have been integrated into recommender systems as item-side auxiliary information, enhancing recommendation accuracy. However, constructing and integrating structural user-side knowledge remains a significant challenge due to the improper granularity and inherent scarcity of user-side features. Recent advancements in Large Language Models (LLMs) offer the potential to bridge this gap by leveraging their human behavior understanding and extensive real-world knowledge. Nevertheless, integrating LLM-generated information into recommender systems presents challenges, including the risk of noisy information and the need for additional knowledge transfer. In this paper, we propose an LLM-based user-side knowledge inference method alongside a carefully designed recommendation framework to address these challenges. Our approach employs LLMs to infer user interests based on historical behaviors, integrating this user-side information with item-side and collaborative data to construct a hybrid structure: the Collaborative Interest Knowledge Graph (CIKG). Furthermore, we propose a CIKG-based recommendation framework that includes a user interest reconstruction module and a cross-domain contrastive learning module to mitigate potential noise and facilitate knowledge transfer. We conduct extensive experiments on three real-world datasets to validate the effectiveness of our method. Our approach achieves state-of-the-art performance compared to competitive baselines, particularly for users with sparse interactions. 

**Abstract (ZH)**: 近年来，知识图谱已作为项目侧辅助信息集成到推荐系统中，提高了推荐的准确性。然而，由于用户侧特征的不适当粒度和固有的稀缺性，构建和整合结构性用户侧知识仍然是一个重大挑战。大型语言模型（LLMs）的最新进展为解决这一问题提供了潜力，通过利用它们对人类行为的理解和广泛的实际知识。然而，将LLM生成的信息集成到推荐系统中也面临挑战，包括噪声信息的风险和额外的知识迁移需求。在本文中，我们提出了一种基于LLM的用户侧知识推断方法以及一个精心设计的推荐框架，以应对这些挑战。我们的方法利用LLM根据历史行为推断用户兴趣，并将这种用户侧信息与项目侧和协同数据相结合，构建一种混合结构：协作兴趣知识图谱（CIKG）。此外，我们提出了一种基于CIKG的推荐框架，包括用户兴趣重建模块和跨域对比学习模块，以减轻潜在的噪声风险并促进知识迁移。我们在三个真实世界数据集上进行了广泛实验，以验证我们方法的有效性。我们的方法在用户交互稀疏的情况下特别优于竞争基线，取得了最先进的性能。 

---
# Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future 

**Title (ZH)**: 增强型大型语言模型推荐系统：分类、趋势、应用与未来 

**Authors**: Qidong Liu, Xiangyu Zhao, Yuhao Wang, Yejing Wang, Zijian Zhang, Yuqi Sun, Xiang Li, Maolin Wang, Pengyue Jia, Chong Chen, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.13432)  

**Abstract**: Large Language Model (LLM) has transformative potential in various domains, including recommender systems (RS). There have been a handful of research that focuses on empowering the RS by LLM. However, previous efforts mainly focus on LLM as RS, which may face the challenge of intolerant inference costs by LLM. Recently, the integration of LLM into RS, known as LLM-Enhanced Recommender Systems (LLMERS), has garnered significant interest due to its potential to address latency and memory constraints in real-world applications. This paper presents a comprehensive survey of the latest research efforts aimed at leveraging LLM to enhance RS capabilities. We identify a critical shift in the field with the move towards incorporating LLM into the online system, notably by avoiding their use during inference. Our survey categorizes the existing LLMERS approaches into three primary types based on the component of the RS model being augmented: Knowledge Enhancement, Interaction Enhancement, and Model Enhancement. We provide an in-depth analysis of each category, discussing the methodologies, challenges, and contributions of recent studies. Furthermore, we highlight several promising research directions that could further advance the field of LLMERS. 

**Abstract (ZH)**: 大型语言模型（LLM）在多个领域具有变革性的潜力，包括推荐系统（RS）。已有少量研究关注通过LLM来增强RS。然而，之前的大部分努力主要集中在将LLM作为RS本身，这可能会面临LLM不可容忍的推理成本挑战。最近，将LLM集成到RS中，被称为LLM增强推荐系统（LLMERS），因其在解决实际应用中延迟和内存限制方面的潜力而引起了广泛关注。本文对最新的研究努力进行了全面综述，旨在利用LLM来增强RS能力。我们识别出领域内的一项关键转变，即转向将LLM集成到在线系统中，特别是在推理阶段避免使用LLM。我们按照被增强的RS模型组件将现有的LLMERS方法分类为三类：知识增强、交互增强和模型增强。我们对每类进行了深入分析，讨论了最近研究的方法、挑战和贡献。此外，我们还指出了几种有前途的研究方向，这些方向有望进一步推动LLMERS领域的进展。 

---
# Nano-ESG: Extracting Corporate Sustainability Information from News Articles 

**Title (ZH)**: 纳米ESG：从新闻文章中提取企业可持续性信息 

**Authors**: Fabian Billert, Stefan Conrad  

**Link**: [PDF](https://arxiv.org/pdf/2412.15093)  

**Abstract**: Determining the sustainability impact of companies is a highly complex subject which has garnered more and more attention over the past few years. Today, investors largely rely on sustainability-ratings from established rating-providers in order to analyze how responsibly a company acts. However, those ratings have recently been criticized for being hard to understand and nearly impossible to reproduce.
An independent way to find out about the sustainability practices of companies lies in the rich landscape of news article data. In this paper, we explore a different approach to identify key opportunities and challenges of companies in the sustainability domain. We present a novel dataset of more than 840,000 news articles which were gathered for major German companies between January 2023 and September 2024. By applying a mixture of Natural Language Processing techniques, we first identify relevant articles, before summarizing them and extracting their sustainability-related sentiment and aspect using Large Language Models (LLMs). Furthermore, we conduct an evaluation of the obtained data and determine that the LLM-produced answers are accurate. We release both datasets at this https URL. 

**Abstract (ZH)**: 确定公司的可持续性影响是一个高度复杂的问题，近年来引起了越来越多的关注。目前，投资者大量依赖于已成立的评级机构提供的可持续性评级，以分析公司的行为是否负责任。然而，这些评级最近因难以理解且几乎无法复制而受到了批评。

寻找公司可持续性实践的独立途径在于丰富的新闻文章数据景观。在这篇论文中，我们探索了一种不同的方法，以识别公司在可持续性领域的关键机遇和挑战。我们提供了一个包含超过840,000篇新闻文章的新数据集，这些文章是为德国主要公司于2023年1月至2024年9月期间收集的。通过结合多种自然语言处理技术，我们首先识别出相关文章，然后对其进行总结，并使用大型语言模型（LLM）提取其与可持续性相关的观点和方面。此外，我们对获得的数据进行了评估，并确定LLM生成的回答是准确的。我们在此 https URL 公布这两个数据集。 

---
# DisCo: Graph-Based Disentangled Contrastive Learning for Cold-Start Cross-Domain Recommendation 

**Title (ZH)**: DisCo：基于图的去纠缠对比学习在冷启动跨域推荐中的应用 

**Authors**: Hourun Li, Yifan Wang, Zhiping Xiao, Jia Yang, Changling Zhou, Ming Zhang, Wei Ju  

**Link**: [PDF](https://arxiv.org/pdf/2412.15005)  

**Abstract**: Recommender systems are widely used in various real-world applications, but they often encounter the persistent challenge of the user cold-start problem. Cross-domain recommendation (CDR), which leverages user interactions from one domain to improve prediction performance in another, has emerged as a promising solution. However, users with similar preferences in the source domain may exhibit different interests in the target domain. Therefore, directly transferring embeddings may introduce irrelevant source-domain collaborative information. In this paper, we propose a novel graph-based disentangled contrastive learning framework to capture fine-grained user intent and filter out irrelevant collaborative information, thereby avoiding negative transfer. Specifically, for each domain, we use a multi-channel graph encoder to capture diverse user intents. We then construct the affinity graph in the embedding space and perform multi-step random walks to capture high-order user similarity relationships. Treating one domain as the target, we propose a disentangled intent-wise contrastive learning approach, guided by user similarity, to refine the bridging of user intents across domains. Extensive experiments on four benchmark CDR datasets demonstrate that DisCo consistently outperforms existing state-of-the-art baselines, thereby validating the effectiveness of both DisCo and its components. 

**Abstract (ZH)**: 推荐系统在各种实际应用中得到了广泛的应用，但它们经常面临用户冷启动问题这一持久挑战。多域推荐（Cross-domain Recommendation, CDR）通过利用一个领域中的用户交互来改善另一个领域中的预测性能，已成为一种有前景的解决方案。然而，在源领域具有相似偏好的用户在其目标领域中可能表现出不同的兴趣。因此，直接转移嵌入信息可能会引入无关的源领域协作信息。在本文中，我们提出了一种新颖的基于图的解耦对比学习框架，以捕捉用户细粒度的意图并过滤掉无关的协作信息，从而避免负迁移。具体地，对于每个领域，我们使用多通道图编码器来捕捉不同用户意图。然后在嵌入空间中构建亲和图，并通过多步随机游走来捕捉高阶的用户相似关系。将一个领域视为目标领域，我们提出了一种基于用户相似性的解耦意图对比学习方法，以在领域间细化用户意图的联系。在四个基准CDR数据集上的广泛实验表明，DisCo始终优于现有的最佳基线方法，从而证明了DisCo及其组件的有效性。 

---
# Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation 

**Title (ZH)**: 基于频谱的模态表示融合图卷积网络在多模态推荐中的应用 

**Authors**: Rongqing Kenneth Ong, Andy W. H. Khong  

**Link**: [PDF](https://arxiv.org/pdf/2412.14978)  

**Abstract**: Incorporating multi-modal features as side information has recently become a trend in recommender systems. To elucidate user-item preferences, recent studies focus on fusing modalities via concatenation, element-wise sum, or attention mechanisms. Despite having notable success, existing approaches do not account for the modality-specific noise encapsulated within each modality. As a result, direct fusion of modalities will lead to the amplification of cross-modality noise. Moreover, the variation of noise that is unique within each modality results in noise alleviation and fusion being more challenging. In this work, we propose a new Spectrum-based Modality Representation (SMORE) fusion graph recommender that aims to capture both uni-modal and fusion preferences while simultaneously suppressing modality noise. Specifically, SMORE projects the multi-modal features into the frequency domain and leverages the spectral space for fusion. To reduce dynamic contamination that is unique to each modality, we introduce a filter to attenuate and suppress the modality noise adaptively while capturing the universal modality patterns effectively. Furthermore, we explore the item latent structures by designing a new multi-modal graph learning module to capture associative semantic correlations and universal fusion patterns among similar items. Finally, we formulate a new modality-aware preference module, which infuses behavioral features and balances the uni- and multi-modal features for precise preference modeling. This empowers SMORE with the ability to infer both user modality-specific and fusion preferences more accurately. Experiments on three real-world datasets show the efficacy of our proposed model. The source code for this work has been made publicly available at this https URL. 

**Abstract (ZH)**: 将多模态特征作为侧信息纳入推荐系统中已成为近年来的趋势。为了阐明用户对项目的偏好，近期的研究集中在通过拼接、元素级求和或注意力机制融合模态信息。尽管这些方法在实践中取得了显著成功，但现有的方法并未考虑到每个模态中封装的特定噪声。因此，直接融合模态会加剧跨模态噪声。此外，每个模态特有的噪声变化进一步增加了噪声抑制和融合的难度。在本研究中，我们提出了一种新的基于光谱的模态表示融合图推荐系统（SMORE），旨在同时捕捉单模态和融合偏好，同时抑制模态噪声。具体而言，SMORE将多模态特征投影到频域，并利用频谱空间进行融合。为了减少每个模态特有的动态污染，我们引入了一个滤波器，以适应性地抑制和降低模态噪声，同时有效地捕捉通用模态模式。此外，我们通过设计一个多模态图学习模块来探索项目潜在结构，该模块能够捕捉相似项目之间的关联语义关联和通用融合模式。最后，我们提出了一个新的模态感知偏好模块，该模块融合行为特征，平衡单模态和多模态特征，从而实现精准的偏好建模。这一模块赋予了SMORE更准确地推断用户模态特定偏好和融合偏好能力。在三个真实世界数据集上的实验显示了我们提出模型的有效性。关于此工作的源代码已公开，网址为 <https://github.com/your-repo-url/smore>。 

---
# ECLIPSE: Contrastive Dimension Importance Estimation with Pseudo-Irrelevance Feedback for Dense Retrieval 

**Title (ZH)**: ECLIPSE：基于伪不可忽略反馈的对比维度重要性估计在密集检索中的应用 

**Authors**: Giulio D'Erasmo, Giovanni Trappolini, Nicola Tonellotto, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2412.14967)  

**Abstract**: Recent advances in Information Retrieval have leveraged high-dimensional embedding spaces to improve the retrieval of relevant documents. Moreover, the Manifold Clustering Hypothesis suggests that despite these high-dimensional representations, documents relevant to a query reside on a lower-dimensional, query-dependent manifold. While this hypothesis has inspired new retrieval methods, existing approaches still face challenges in effectively separating non-relevant information from relevant signals. We propose a novel methodology that addresses these limitations by leveraging information from both relevant and non-relevant documents. Our method, ECLIPSE, computes a centroid based on irrelevant documents as a reference to estimate noisy dimensions present in relevant ones, enhancing retrieval performance. Extensive experiments on three in-domain and one out-of-domain benchmarks demonstrate an average improvement of up to 19.50% (resp. 22.35%) in mAP(AP) and 11.42% (resp. 13.10%) in nDCG@10 w.r.t. the DIME-based baseline (resp. the baseline using all dimensions). Our results pave the way for more robust, pseudo-irrelevance-based retrieval systems in future IR research. 

**Abstract (ZH)**: 近年来，信息检索领域的最新进展利用高维嵌入空间来提高相关文档的检索效果。此外，流形聚类假说表明，尽管存在这些高维表示，与查询相关的文档仍然位于一个与查询有关的低维流形上。虽然这一假说激励了新的检索方法，但现有的方法仍然面临着从非相关信息中有效分离相关信号的挑战。我们提出了一种新的方法来解决这些限制，该方法通过利用相关和非相关文档的信息协同工作。我们的方法ECLIPSE基于非相关文档计算一个质心，作为参考来估计相关文档中存在的噪声维度，从而提高检索性能。在三个领域内和一个领域外的基准测试中的大量实验结果显示，与基于DIME（Dimensionality Invariant Method）的基线相比，ECLIPSE在mAP（平均精度）上平均提高19.50%（相对提升了22.35%），在nDCG@10（在前10位的相关度增益）上平均提高11.42%（相对提升了13.10%）。我们的结果为未来信息检索研究中的更稳健的、基于伪无关信息的检索系统开拓了新的途径。 

---
# Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models 

**Title (ZH)**: 滑动窗口并非终点：探索长上下文大型语言模型的全面排名 

**Authors**: Wenhan Liu, Xinyu Ma, Yutao Zhu, Ziliang Zhao, Shuaiqiang Wang, Dawei Yin, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14574)  

**Abstract**: Large Language Models (LLMs) have shown exciting performance in listwise passage ranking. Due to the limited input length, existing methods often adopt the sliding window strategy. Such a strategy, though effective, is inefficient as it involves repetitive and serialized processing, which usually re-evaluates relevant passages multiple times. As a result, it incurs redundant API costs, which are proportional to the number of inference tokens. The development of long-context LLMs enables the full ranking of all passages within a single inference, avoiding redundant API costs. In this paper, we conduct a comprehensive study of long-context LLMs for ranking tasks in terms of efficiency and effectiveness. Surprisingly, our experiments reveal that full ranking with long-context LLMs can deliver superior performance in the supervised fine-tuning setting with a huge efficiency improvement. Furthermore, we identify two limitations of fine-tuning the full ranking model based on existing methods: (1) sliding window strategy fails to produce a full ranking list as a training label, and (2) the language modeling loss cannot emphasize top-ranked passage IDs in the label. To alleviate these issues, we propose a new complete listwise label construction approach and a novel importance-aware learning objective for full ranking. Experiments show the superior performance of our method over baselines. Our codes are available at \url{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在列表式段落排序任务中展现了令人兴奋的性能。由于输入长度有限，现有方法往往采用滑动窗口策略。尽管该策略有效，但其效率较低，因为它涉及重复且串行的处理，通常需要多次重新评估相关段落。这导致了冗余的API开销，开销与推理token的数量成正比。长语境LLMs的发展使得能够在单次推理中对所有段落进行完整排序，从而避免了冗余的API开销。在本文中，我们对长语境LLMs在排序任务中的效率和有效性进行了全面研究。令人惊讶的是，我们的实验表明，在监督微调设置中，使用长语境LLMs进行完整排序可以实现优于现有方法的性能，并且具有显著的效率改进。此外，我们发现基于现有方法微调完整排序模型的两个局限性：（1）滑动窗口策略无法产生完整排序列表作为训练标签；（2）语言模型损失不能强调标签中顶级排序段落ID的重要性。为缓解这些问题，我们提出了一种新的完整列表式标签构建方法和一种新的重要性感知学习目标，用于完整排序。实验结果表明，我们的方法优于基线方法。我们的代码可在 \url{此处提供URL} 获取。 

---
# HEC-GCN: Hypergraph Enhanced Cascading Graph Convolution Network for Multi-Behavior Recommendation 

**Title (ZH)**: HEC-GCN：超图增强级联图卷积网络在多行为推荐中的应用 

**Authors**: Yabo Yin, Xiaofei Zhu, Wenshan Wang, Yihao Zhang, Pengfei Wang, Yixing Fan, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2412.14476)  

**Abstract**: Multi-behavior recommendation (MBR) has garnered growing attention recently due to its ability to mitigate the sparsity issue by inferring user preferences from various auxiliary behaviors to improve predictions for the target behavior. Although existing research on MBR has yielded impressive results, they still face two major limitations. First, previous methods mainly focus on modeling fine-grained interaction information between users and items under each behavior, which may suffer from sparsity issue. Second, existing models usually concentrate on exploiting dependencies between two consecutive behaviors, leaving intra- and inter-behavior consistency largely unexplored. To the end, we propose a novel approach named Hypergraph Enhanced Cascading Graph Convolution Network for multi-behavior recommendation (HEC-GCN). To be specific, we first explore both fine- and coarse-grained correlations among users or items of each behavior by simultaneously modeling the behavior-specific interaction graph and its corresponding hypergraph in a cascaded manner. Then, we propose a behavior consistency-guided alignment strategy that ensures consistent representations between the interaction graph and its associated hypergraph for each behavior, while also maintaining representation consistency across different behaviors. Extensive experiments and analyses on three public benchmark datasets demonstrate that our proposed approach is consistently superior to previous state-of-the-art methods due to its capability to effectively attenuate the sparsity issue as well as preserve both intra- and inter-behavior consistencies. The code is available at this https URL. 

**Abstract (ZH)**: 多行为推荐（Multi-behavior Recommendation, MBR）近年来引起了越来越多的关注，由于其能够通过从各种辅助行为中推断用户偏好来减轻目标行为预测中的稀疏性问题，从而为其提供了改进的空间。尽管现有对MBR的研究已经取得了令人印象深刻的成果，但它们仍然面临着两个主要局限性。首先，现有方法主要侧重于在每种行为下建模用户与项目的精细交互信息，这可能会导致稀疏性问题。其次，现有的模型通常专注于探索两种连续行为之间的依赖关系，而忽略了内在行为一致性和跨行为一致性的探索。有鉴于此，我们提出了一种名为Hypergraph Enhanced Cascading Graph Convolution Network for multi-behavior recommendation（HEC-GCN）的新颖方法。具体而言，我们首先通过同时建模每种行为的特异性交互图及其相应的超图，以递归的方式探索用户或项目之间的细粒度和粗粒度相关性。然后，我们提出了一种行为一致性导向的对齐策略，以确保每种行为的交互图与其相关超图之间的一致表示，同时保持不同行为之间的表示一致性。在三个公开基准数据集上的广泛实验和分析证明，我们提出的该方法由于其能够有效缓解稀疏性问题，并且保持了内在和跨行为一致性方面，相对于以前最先进的方法具有明显的优越性。代码已托管于此 [https://.../]。 

---
# VISA: Retrieval Augmented Generation with Visual Source Attribution 

**Title (ZH)**: VISA：带有视觉来源归因的检索增强生成 

**Authors**: Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Wenhu Chen, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14457)  

**Abstract**: Generation with source attribution is important for enhancing the verifiability of retrieval-augmented generation (RAG) systems. However, existing approaches in RAG primarily link generated content to document-level references, making it challenging for users to locate evidence among multiple content-rich retrieved documents. To address this challenge, we propose Retrieval-Augmented Generation with Visual Source Attribution (VISA), a novel approach that combines answer generation with visual source attribution. Leveraging large vision-language models (VLMs), VISA identifies the evidence and highlights the exact regions that support the generated answers with bounding boxes in the retrieved document screenshots. To evaluate its effectiveness, we curated two datasets: Wiki-VISA, based on crawled Wikipedia webpage screenshots, and Paper-VISA, derived from PubLayNet and tailored to the medical domain. Experimental results demonstrate the effectiveness of VISA for visual source attribution on documents' original look, as well as highlighting the challenges for improvement. Code, data, and model checkpoints will be released. 

**Abstract (ZH)**: 生成并提供来源归属对于提升检索增强生成（RAG）系统的可验证性至关重要。然而，现有RAG方法主要将生成内容与文档级别的引用关联起来，使得用户在多个内容丰富的检索文档中难以定位证据。为应对这一挑战，我们提出了一种新型方法——检索增强生成与可视来源归属（VISA），该方法结合了答案生成与可视来源归属。利用大规模的视觉-语言模型（VLMs），VISA 能够识别支持生成答案的证据，并通过在检索文档截图中使用边界框突出显示具体的支撑区域。为了评估其有效性，我们构建了两个数据集：Wiki-VISA，基于抓取的维基百科网页截图；以及Paper-VISA，基于PubLayNet，并针对医学领域进行了定制。实验结果表明，VISA 在保持文档原始外观的同时，能够有效地进行可视化来源归属，并揭示存在的改进挑战。我们还将发布相关代码、数据和模型检查点。 

---
# Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems 

**Title (ZH)**: 更长的提示是否一定更好？大型语言模型在推荐系统中的提示选择研究 

**Authors**: Genki Kusano, Kosuke Akimoto, Kunihiro Takeoka  

**Link**: [PDF](https://arxiv.org/pdf/2412.14454)  

**Abstract**: In large language models (LLM)-based recommendation systems (LLM-RSs), accurately predicting user preferences by leveraging the general knowledge of LLMs is possible without requiring extensive training data. By converting recommendation tasks into natural language inputs called prompts, LLM-RSs can efficiently solve issues that have been difficult to address due to data scarcity but are crucial in applications such as cold-start and cross-domain problems. However, when applying this in practice, selecting the prompt that matches tasks and data is essential. Although numerous prompts have been proposed in LLM-RSs and representing the target user in prompts significantly impacts recommendation accuracy, there are still no clear guidelines for selecting specific prompts.
In this paper, we categorize and analyze prompts from previous research to establish practical prompt selection guidelines. Through 450 experiments with 90 prompts and five real-world datasets, we examined the relationship between prompts and dataset characteristics in recommendation accuracy. We found that no single prompt consistently outperforms others; thus, selecting prompts on the basis of dataset characteristics is crucial. Here, we propose a prompt selection method that achieves higher accuracy with minimal validation data. Because increasing the number of prompts to explore raises costs, we also introduce a cost-efficient strategy using high-performance and cost-efficient LLMs, significantly reducing exploration costs while maintaining high prediction accuracy. Our work offers valuable insights into the prompt selection, advancing accurate and efficient LLM-RSs. 

**Abstract (ZH)**: 在基于大型语言模型（LLM）的推荐系统（LLM-RSs）中，利用LLM的通用知识来准确预测用户偏好，无需大量训练数据即可实现。通过将推荐任务转化为称为提示（prompts）的自然语言输入，LLM-RSs可以高效地解决由于数据稀缺但对实际应用至关重要的问题，如冷启动问题和跨域问题等。然而，在实践中应用这一方法时，选择与任务和数据相匹配的提示至关重要。尽管在LLM-RSs中已经提出了众多提示，并且提示中的目标用户表示方式显著影响推荐准确性，但仍没有明确的指南来选择特定的提示。

在此论文中，我们对来自先前研究的提示进行分类与分析，以建立实用的提示选择指南。通过使用90个提示和五个真实世界数据集进行450次实验，我们研究了提示与数据集特征之间的关系在推荐准确性中的影响。我们的研究表明，没有一种提示在所有情况下都能优越于其他提示，因此根据数据集特征选择提示至关重要。在此基础上，我们提出了一种利用最少验证数据提高准确性的提示选择方法。由于增加提示数量来探索会增加成本，我们还引入了一种高效的策略，使用高性能且成本效益高的LLM，从而显著降低了探索成本，同时保持了高预测准确性。我们的研究为提示选择提供了宝贵的见解，并推进了准确高效的LLM-RSs的发展。 

---
# ChainRank-DPO: Chain Rank Direct Preference Optimization for LLM Rankers 

**Title (ZH)**: ChainRank-DPO：面向生成式预训练模型排序器的链式排名直接偏好优化 

**Authors**: Haowei Liu, Xuyang Wu, Guohao Sun, Zhiqiang Tao, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14405)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable effectiveness in text reranking through works like RankGPT, leveraging their human-like reasoning about relevance. However, supervised fine-tuning for ranking often diminishes these models' general-purpose capabilities, including the crucial reasoning abilities that make them valuable for ranking. We introduce a novel approach integrating Chain-of-Thought prompting with an SFT-DPO (Supervised Fine-Tuning followed by Direct Preference Optimization) pipeline to preserve these capabilities while improving ranking performance. Our experiments on TREC 2019 and 2020 Deep Learning datasets show that our approach outperforms the state-of-the-art RankZephyr while maintaining strong performance on the Massive Multitask Language Understanding (MMLU) benchmark, demonstrating effective preservation of general-purpose capabilities through thoughtful fine-tuning strategies. Our code and data will be publicly released upon the acceptance of the paper. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过如RankGPT等研究工作在文本重排序方面展现出了非凡的效果，得益于它们在相关性方面具有的类人类推理能力。然而，用于排名的监督微调往往会削弱这些模型的一般用途能力，包括使它们对排名有价值的关键推理能力。我们提出了一种新的方法，通过将思维链提示与SFT-DPO（监督微调后直接偏好优化）管道相结合，以保留这些能力并提高排名性能。我们在TREC 2019和2020深度学习数据集上的实验表明，我们的方法在最先进的RankZephyr之上表现更优，同时在大规模多任务语言理解（MMLU）基准测试中保持了良好的性能，证明了通过精心设计的微调策略有效地保留了一般用途能力。我们在论文被接受后会公开我们的代码和数据。 

---
# Embedding Cultural Diversity in Prototype-based Recommender Systems 

**Title (ZH)**: 将文化多样性嵌入原型推荐系统中 

**Authors**: Armin Moradi, Nicola Neophytou, Florian Carichon, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2412.14329)  

**Abstract**: Popularity bias in recommender systems can increase cultural overrepresentation by favoring norms from dominant cultures and marginalizing underrepresented groups. This issue is critical for platforms offering cultural products, as they influence consumption patterns and human perceptions. In this work, we address popularity bias by identifying demographic biases within prototype-based matrix factorization methods. Using the country of origin as a proxy for cultural identity, we link this demographic attribute to popularity bias by refining the embedding space learning process. First, we propose filtering out irrelevant prototypes to improve representativity. Second, we introduce a regularization technique to enforce a uniform distribution of prototypes within the embedding space. Across four datasets, our results demonstrate a 27\% reduction in the average rank of long-tail items and a 2\% reduction in the average rank of items from underrepresented countries. Additionally, our model achieves a 2\% improvement in HitRatio@10 compared to the state-of-the-art, highlighting that fairness is enhanced without compromising recommendation quality. Moreover, the distribution of prototypes leads to more inclusive explanations by better aligning items with diverse prototypes. 

**Abstract (ZH)**: 推荐系统中的流行度偏差可能会加剧文化过度代表，因为它倾向于支持主导文化中的规范，而边缘化未被充分代表的群体。这一问题对于提供文化产品的平台尤其关键，因为它们会影响消费模式和人类的认知。在本研究中，我们通过识别基于原型的矩阵分解方法中的人口统计偏差来解决流行度偏差问题。使用起源国家作为文化身份的代理，我们通过精炼嵌入空间的学习过程将该人口统计属性与流行度偏差联系起来。首先，我们建议过滤掉无关的原型以提高代表性。其次，我们引入了一种正则化技术，以在嵌入空间中强制实现原型的均匀分布。在四个数据集中，我们的结果显示，尾部较长项的平均排名降低了27%，来自未充分代表国家的项目平均排名降低了2%。此外，与当前最先进的方法相比，我们的模型在HitRatio@10方面提高了2%，这表明公平性得到了提升而不影响推荐质量。同时，原型的分布导致了更加包容的解释，因为这些原型更好地与多样化的项目对齐。 

---
# SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation 

**Title (ZH)**: SAFERec：结合自注意力机制和频率增强的next-basket推荐模型 

**Authors**: Oleg Lashinin, Denis Krasilnikov, Aleksandr Milogradskii, Marina Ananyeva  

**Link**: [PDF](https://arxiv.org/pdf/2412.14302)  

**Abstract**: Transformer-based approaches such as BERT4Rec and SASRec demonstrate strong performance in Next Item Recommendation (NIR) tasks. However, applying these architectures to Next-Basket Recommendation (NBR) tasks, which often involve highly repetitive interactions, is challenging due to the vast number of possible item combinations in a basket. Moreover, frequency-based methods such as TIFU-KNN and UP-CF still demonstrate strong performance in NBR tasks, frequently outperforming deep-learning approaches. This paper introduces SAFERec, a novel algorithm for NBR that enhances transformer-based architectures from NIR by incorporating item frequency information, consequently improving their applicability to NBR tasks. Extensive experiments on multiple datasets show that SAFERec outperforms all other baselines, specifically achieving an 8\% improvement in Recall@10. 

**Abstract (ZH)**: 基于Transformer的方法，如BERT4Rec和SASRec，在Next Item Recommendation (NIR) 任务中表现出强大的性能。然而，将这些架构应用于Next-Basket Recommendation (NBR) 任务时，往往会遇到挑战，因为篮子中可能包含大量的项目组合，导致重复交互频繁出现。此外，基于频率的方法，如TIFU-KNN和UP-CF，在NBR 任务中仍然表现出强大的性能，经常超越深度学习方法。本文介绍了一种名为SAFERec的新算法，该算法通过结合项目频率信息来增强NIR中基于Transformer的架构，从而提高其在NBR 任务中的适用性。在多个数据集上的广泛实验表明，SAFERec 在所有基准模型中表现最佳，特别是在Recall@10上提高了8%。 

---
# Progressive Multimodal Reasoning via Active Retrieval 

**Title (ZH)**: 渐进多模态推理通过主动检索实现 

**Authors**: Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14835)  

**Abstract**: Multi-step multimodal reasoning tasks pose significant challenges for multimodal large language models (MLLMs), and finding effective ways to enhance their performance in such scenarios remains an unresolved issue. In this paper, we propose AR-MCTS, a universal framework designed to progressively improve the reasoning capabilities of MLLMs through Active Retrieval (AR) and Monte Carlo Tree Search (MCTS). Our approach begins with the development of a unified retrieval module that retrieves key supporting insights for solving complex reasoning problems from a hybrid-modal retrieval corpus. To bridge the gap in automated multimodal reasoning verification, we employ the MCTS algorithm combined with an active retrieval mechanism, which enables the automatic generation of step-wise annotations. This strategy dynamically retrieves key insights for each reasoning step, moving beyond traditional beam search sampling to improve the diversity and reliability of the reasoning space. Additionally, we introduce a process reward model that aligns progressively to support the automatic verification of multimodal reasoning tasks. Experimental results across three complex multimodal reasoning benchmarks confirm the effectiveness of the AR-MCTS framework in enhancing the performance of various multimodal models. Further analysis demonstrates that AR-MCTS can optimize sampling diversity and accuracy, yielding reliable multimodal reasoning. 

**Abstract (ZH)**: 多步多模态推理任务对多模态大型语言模型（MLLMs）构成了显著挑战，如何在这种场景下有效提升其性能至今仍是未解决的问题。本文提出了一种名为AR-MCTS的通用框架，旨在通过积极检索（Active Retrieval, AR）和蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）逐步提高MLLMs的推理能力。我们的方法首先开发了一个统一的检索模块，该模块可以从混合模态检索语料库中检索关键支持性见解，以解决复杂的推理问题。为弥合自动多模态推理验证的差距，我们采用了结合了积极检索机制的MCTS算法，从而实现了逐步生成逐步标注的能力。这一策略能够动态地为每个推理步骤检索关键见解，超越了传统的束搜索采样方法，以提高推理空间的多样性和可靠性。此外，我们还引入了一种过程奖励模型，以逐步对齐支持多模态推理任务的自动验证。在三个复杂多模态推理基准上进行的实验结果证明了AR-MCTS框架在提高各类多模态模型性能方面的有效性。进一步的分析表明，AR-MCTS能够优化采样的多样性和准确性，产生可靠得多模态推理结果。 

---
# Efficient Self-Supervised Video Hashing with Selective State Spaces 

**Title (ZH)**: 高效的选择性状态空间自监督视频哈希方法 

**Authors**: Jinpeng Wang, Niu Lian, Jun Li, Yuting Wang, Yan Feng, Bin Chen, Yongbing Zhang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2412.14518)  

**Abstract**: Self-supervised video hashing (SSVH) is a practical task in video indexing and retrieval. Although Transformers are predominant in SSVH for their impressive temporal modeling capabilities, they often suffer from computational and memory inefficiencies. Drawing inspiration from Mamba, an advanced state-space model, we explore its potential in SSVH to achieve a better balance between efficacy and efficiency. We introduce S5VH, a Mamba-based video hashing model with an improved self-supervised learning paradigm. Specifically, we design bidirectional Mamba layers for both the encoder and decoder, which are effective and efficient in capturing temporal relationships thanks to the data-dependent selective scanning mechanism with linear complexity. In our learning strategy, we transform global semantics in the feature space into semantically consistent and discriminative hash centers, followed by a center alignment loss as a global learning signal. Our self-local-global (SLG) paradigm significantly improves learning efficiency, leading to faster and better convergence. Extensive experiments demonstrate S5VH's improvements over state-of-the-art methods, superior transferability, and scalable advantages in inference efficiency. Code is available at this https URL. 

**Abstract (ZH)**: 自我监督视频哈希（SSVH）是视频索引和检索中的一个实际任务。尽管 Transformers 由于其出色的时序建模能力在 SSVH 中占主导地位，但它们经常面临计算和内存效率低下的问题。受 Mamba（一种高级状态空间模型）的启发，我们研究其在 SSVH 中的应用，以实现效用和效率之间的更好平衡。我们提出了 S5VH，一种基于 Mamba 的视频哈希模型，引入了改进的自我监督学习范式。具体而言，我们为编码器和解码器设计了双向 Mamba 层，这些层通过与线性复杂度相关的数据依赖选择性扫描机制有效地捕捉时序关系。在我们的学习策略中，我们将特征空间中的全局语义转化为语义一致且具有区分性的哈希中心，随后应用中心对齐损失作为全局学习信号。我们的局部-全局（SLG）范式显著提高了学习效率，导致更快更好的收敛。广泛的经验表明，S5VH 在最先进的方法上具有改进、更强的迁移能力和可扩展的优势。推理效率上的优势已在实验中得到验证。代码可在以下网址获取：[提供网址] 

---
# Moving Beyond LDA: A Comparison of Unsupervised Topic Modelling Techniques for Qualitative Data Analysis of Online Communities 

**Title (ZH)**: 超越LDA：无监督主题建模技术在在线社区定性数据分析中的比较 

**Authors**: Amandeep Kaur, James R. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2412.14486)  

**Abstract**: Social media constitutes a rich and influential source of information for qualitative researchers. Although computational techniques like topic modelling assist with managing the volume and diversity of social media content, qualitative researcher's lack of programming expertise creates a significant barrier to their adoption. In this paper we explore how BERTopic, an advanced Large Language Model (LLM)-based topic modelling technique, can support qualitative data analysis of social media. We conducted interviews and hands-on evaluations in which qualitative researchers compared topics from three modelling techniques: LDA, NMF, and BERTopic. BERTopic was favoured by 8 of 12 participants for its ability to provide detailed, coherent clusters for deeper understanding and actionable insights. Participants also prioritised topic relevance, logical organisation, and the capacity to reveal unexpected relationships within the data. Our findings underscore the potential of LLM-based techniques for supporting qualitative analysis. 

**Abstract (ZH)**: 社交媒体是一个丰富且具有影响力的高质量信息来源，尤其对定性研究人员而言。尽管计算技术如主题建模有助于处理社交媒体内容的巨大体量和多样性，但定性研究人员缺乏编程技能这一问题成为其采用这些技术的显著障碍。本文探讨了如何利用基于大型语言模型（LLM）的高级主题建模技术BERTopic支持社交媒体的定性数据分析。我们在研究中通过采访和动手实践评估，让12位定性研究人员比较了三种建模技术（LDA、NMF和BERTopic）生成的主题。8名参与者更偏好BERTopic，因为它能够提供详细、连贯的簇集，有助于更深入的理解和行动性的洞察。研究人员还强调了主题相关性、逻辑组织以及揭示数据中意外关系的能力。我们的研究结果强调了基于LLM的技术在支持定性分析方面的潜力。 

---
# State Space Models are Strong Text Rerankers 

**Title (ZH)**: 状态空间模型是强大的文本重排序器 

**Authors**: Zhichao Xu, Jinghua Yan, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2412.14354)  

**Abstract**: Transformers dominate NLP and IR; but their inference inefficiencies and challenges in extrapolating to longer contexts have sparked interest in alternative model architectures. Among these, state space models (SSMs) like Mamba offer promising advantages, particularly $O(1)$ time complexity in inference. Despite their potential, SSMs' effectiveness at text reranking -- a task requiring fine-grained query-document interaction and long-context understanding -- remains underexplored.
This study benchmarks SSM-based architectures (specifically, Mamba-1 and Mamba-2) against transformer-based models across various scales, architectures, and pre-training objectives, focusing on performance and efficiency in text reranking tasks. We find that (1) Mamba architectures achieve competitive text ranking performance, comparable to transformer-based models of similar size; (2) they are less efficient in training and inference compared to transformers with flash attention; and (3) Mamba-2 outperforms Mamba-1 in both performance and efficiency. These results underscore the potential of state space models as a transformer alternative and highlight areas for improvement in future IR applications. 

**Abstract (ZH)**: 变压器在自然语言处理（NLP）和信息检索（IR）领域占据主导地位，但它们在推理上的低效性和难以扩展到较长语境的问题引发了对替代模型架构的兴趣。在这些替代模型中，状态空间模型（SSMs）如Mamba表现出令人期待的优势，尤其是在推理上的恒定时间复杂度$O(1)$。尽管如此，SSMs在文本重排序任务中的有效性仍然未得到充分探索，而文本重排序任务要求进行精细的查询文档交互和长语境的理解。

本研究在各种规模、架构和预训练目标下，对比了基于SSM的架构（具体来说是Mamba-1和Mamba-2）与基于变压器的模型，重点关注文本重排序任务中的性能和效率。我们发现：
1. Mamba架构在文本排序性能上达到了与相似大小的基于变压器的模型相当的水平；
2. 在训练和推理效率方面，Mamba架构相对于使用闪存注意力的变压器来说效率较低；
3. Mamba-2在性能和效率上都优于Mamba-1。

这些结果凸显了状态空间模型作为变压器替代品的潜力，并指出了未来信息检索应用中需要改进的领域。 

---
# Transversal PACS Browser API: Addressing Interoperability Challenges in Medical Imaging Systems 

**Title (ZH)**: 横向PACS浏览器API：解决医疗成像系统互操作性挑战 

**Authors**: Diogo Lameira, Filipa Ferraz  

**Link**: [PDF](https://arxiv.org/pdf/2412.14229)  

**Abstract**: Advances in imaging technologies have revolutionised the medical imaging and healthcare sectors, leading to the widespread adoption of PACS for the storage, retrieval, and communication of medical images. Although these systems have improved operational efficiency, significant challenges remain in effectively retrieving DICOM images, which are essential for diagnosis and overall patient care. Moreover, issues such as fragmented systems, interoperability barriers, and complex user interfaces can often prevent healthcare professionals from efficiently accessing medical images. Addressing these challenges, the Transversal PACS Browser API is a robust and user-friendly solution designed to enhance the process of querying and retrieving DICOM images. It offers advanced filtering capabilities through a variety of filter options as well as a custom field search, that allows users to easily navigate through large medical image collections with ease. Additionally, the application provides a unified interface for querying and retrieving from multiple PACS stations, addressing the challenges of fragmentation and complexity associated with accessing medical images. Other key features include the ability to pre-view images directly within the application. All of this contributes to the transversal nature of the API, serving not only healthcare providers, but anyone who relies on efficient access to these resources. To validate the performance and usability of the application, comprehensive testing was carried out with stakeholders of the field, the results of which showed general satisfaction, highlighting the API's clean design, ease of use, and effective search capabilities of the API, as well as the usefulness of previewing images within the application. 

**Abstract (ZH)**: 影像技术的进步已经革命性地改变了医学影像和医疗保健领域，导致PACS（picture archiving and communication system）系统的广泛应用，用于存储、检索和传输医学影像。尽管这些系统提高了操作效率，但在有效检索DICOM影像方面仍然面临着重大挑战，这些影像对于诊断和整体患者护理至关重要。此外，系统碎片化问题、互操作性障碍以及复杂的用户界面常常导致医疗专业人员无法高效访问医学影像。针对这些挑战，Transversal PACS浏览器API提供了一种稳健且用户友好的解决方案，旨在优化和简化DICOM图像的查询和检索过程。该API通过多种过滤选项和自定义字段搜索提供高级过滤功能，使用户能够轻松导航庞大的医学影像集合。此外，该应用还提供了一个统一界面，用于从多个PACS站点查询和检索影像，以解决访问医学影像时碎片化和复杂性带来的挑战。其他关键功能包括预览图像的能力，可以直接在应用程序中查看。所有这些功能共同构成了API的横向特性，不仅服务于医疗提供者，还服务于依赖于高效访问这些资源的任何人。为了验证应用程序的性能和易用性，对领域内的利益相关者进行了全面测试，测试结果显示总体满意度，强调了API简洁的设计、易于使用的特性，以及API的有效搜索功能，以及在应用程序中预览图像的实用性。 

---
# Whom do Explanations Serve? A Systematic Literature Survey of User Characteristics in Explainable Recommender Systems Evaluation 

**Title (ZH)**: 《解释性推荐系统评估中用户的特性服务于谁？一项系统文献综述》

这个标题翻译成中文既符合学术规范，又能准确传达原文的意思。 

**Authors**: Kathrin Wardatzky, Oana Inel, Luca Rossetto, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2412.14193)  

**Abstract**: Adding explanations to recommender systems is said to have multiple benefits, such as increasing user trust or system transparency. Previous work from other application areas suggests that specific user characteristics impact the users' perception of the explanation. However, we rarely find this type of evaluation for recommender systems explanations. This paper addresses this gap by surveying 124 papers in which recommender systems explanations were evaluated in user studies. We analyzed their participant descriptions and study results where the impact of user characteristics on the explanation effects was measured. Our findings suggest that the results from the surveyed studies predominantly cover specific users who do not necessarily represent the users of recommender systems in the evaluation domain. This may seriously hamper the generalizability of any insights we may gain from current studies on explanations in recommender systems. We further find inconsistencies in the data reporting, which impacts the reproducibility of the reported results. Hence, we recommend actions to move toward a more inclusive and reproducible evaluation. 

**Abstract (ZH)**: 在推荐系统中添加解释被认为具有多重益处，例如增加用户信任或提高系统的透明度。来自其他应用场景的先前研究提示，特定的用户特征会影响用户对解释的感知。然而，我们很少在推荐系统解释的研究中找到这种类型的评估。本文通过回顾124篇论文来填补这一空白，这些论文通过用户研究评估了推荐系统中的解释效果。我们分析了其中关于用户特征如何影响解释效果的研究参与者描述和研究结果。我们的研究结果表明，被调查的研究大多侧重于特定的用户群体，这些用户未必代表推荐系统在评估领域中的实际用户。这可能严重影响我们从当前关于推荐系统解释的研究中获得的见解的普适性。此外，我们还发现了数据报告中的不一致性，这影响了研究结果的可再现性。因此，我们建议采取措施以实现更包容性和可再现性的评价方法。 

---
# Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics with Large Language Models 

**Title (ZH)**: 使用大型语言模型进行数据科学中多步洞察综合的高级推理与转换引擎 

**Authors**: Atin Sakkeer Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2412.14146)  

**Abstract**: This paper presents the Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics (ARTEMIS-DA), a novel framework designed to augment Large Language Models (LLMs) for solving complex, multi-step data analytics tasks. ARTEMIS-DA integrates three core components: the Planner, which dissects complex user queries into structured, sequential instructions encompassing data preprocessing, transformation, predictive modeling, and visualization; the Coder, which dynamically generates and executes Python code to implement these instructions; and the Grapher, which interprets generated visualizations to derive actionable insights. By orchestrating the collaboration between these components, ARTEMIS-DA effectively manages sophisticated analytical workflows involving advanced reasoning, multi-step transformations, and synthesis across diverse data modalities. The framework achieves state-of-the-art (SOTA) performance on benchmarks such as WikiTableQuestions and TabFact, demonstrating its ability to tackle intricate analytical tasks with precision and adaptability. By combining the reasoning capabilities of LLMs with automated code generation and execution and visual analysis, ARTEMIS-DA offers a robust, scalable solution for multi-step insight synthesis, addressing a wide range of challenges in data analytics. 

**Abstract (ZH)**: 本文提出了Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics（ARTEMIS-DA）框架，这是一种旨在增强大规模语言模型（LLMs）以解决复杂多步骤数据分析师任务的新颖框架。ARTEMIS-DA 集成了三个核心组件：计划器，它将复杂的用户查询分解为包括数据预处理、转换、预测建模和可视化在内的结构化、顺序化指令；编码器，它动态生成并执行Python代码以实现这些指令；以及图形器，它解释生成的可视化并从中提取可操作的见解。通过协调这些组件之间的协作，ARTEMIS-DA 有效地管理了涉及高级推理、多步骤转换和跨不同数据模态综合的复杂分析工作流。该框架在WikiTableQuestions和TabFact等基准测试中达到了最先进的（SOTA）性能，展示了其对复杂分析任务的精度和适应性。通过结合LLM的推理能力、自动化代码生成与执行以及可视化分析，ARTEMIS-DA 提供了一种稳健且可扩展的解决方案，以实现多步骤洞察综合，并应对数据分析师中的各种挑战。 

---
# A Cognitive Ideation Support Framework using IBM Watson Services 

**Title (ZH)**: 使用IBM Watson服务的认知构想支持框架 

**Authors**: Samaa Elnagar, Kweku-Muata Osei-Bryson  

**Link**: [PDF](https://arxiv.org/pdf/2412.14025)  

**Abstract**: Ideas generation is a core activity for innovation in organizations. The creativity of the generated ideas depends not only on the knowledge retrieved from the organizations' knowledge bases, but also on the external knowledge retrieved from other resources. Unfortunately, organizations often cannot efficiently utilize the knowledge in the knowledge bases due to the limited abilities of the search and retrieval mechanisms especially when dealing with unstructured data. In this paper, we present a new cognitive support framework for ideation that uses the IBM Watson DeepQA services. IBM Watson is a Question Answering system which mimics human cognitive abilities to retrieve and rank information. The proposed framework is based on the Search for Ideas in the Associative Memory (SIAM) model to help organizations develop creative ideas through discovering new relationships between retrieved data. To evaluate the effectiveness of the proposed system, the generated ideas generated are selected and assessed using a set of established creativity criteria. 

**Abstract (ZH)**: 创新是组织中的一项核心活动。生成的想法的创造性不仅依赖于从组织的知识库中检索到的知识，还依赖于从其他资源中检索到的外部知识。不幸的是，由于搜索和检索机制能力有限，尤其是在处理非结构化数据时，组织往往无法有效利用知识库中的知识。本文提出了一种新的认知支持框架，利用IBM Watson DeepQA服务来促进创意生成。IBM Watson是一款具有问答功能的系统，模拟人类的认知能力，用于检索和排序信息。所提出的框架基于联想记忆中的想法搜索（SIAM）模型，通过发现检索到的数据之间的新关系来帮助组织生成具有创造性的想法。为了评价该系统的效果，生成的想法被选择并使用一系列已确立的创造性标准进行评估。 

---
# CRM: Retrieval Model with Controllable Condition 

**Title (ZH)**: CRM：可控条件检索模型 

**Authors**: Chi Liu, Jiangxia Cao, Rui Huang, Kuo Cai, Weifeng Ding, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13844)  

**Abstract**: Recommendation systems (RecSys) are designed to connect users with relevant items from a vast pool of candidates while aligning with the business goals of the platform. A typical industrial RecSys is composed of two main stages, retrieval and ranking: (1) the retrieval stage aims at searching hundreds of item candidates satisfied user interests; (2) based on the retrieved items, the ranking stage aims at selecting the best dozen items by multiple targets estimation for each item candidate, including classification and regression targets. Compared with ranking model, the retrieval model absence of item candidate information during inference, therefore retrieval models are often trained by classification target only (e.g., click-through rate), but failed to incorporate regression target (e.g., the expected watch-time), which limit the effectiveness of retrieval. In this paper, we propose the Controllable Retrieval Model (CRM), which integrates regression information as conditional features into the two-tower retrieval paradigm. This modification enables the retrieval stage could fulfill the target gap with ranking model, enhancing the retrieval model ability to search item candidates satisfied the user interests and condition effectively. We validate the effectiveness of CRM through real-world A/B testing and demonstrate its successful deployment in Kuaishou short-video recommendation system, which serves over 400 million users. 

**Abstract (ZH)**: 推荐系统（RecSys）旨在将用户与大量候选项目中的相关项目连接起来，并与平台的业务目标相一致。一个典型的工业界推荐系统通常由两个主要阶段组成：检索和排序：（1）检索阶段旨在搜索满足用户兴趣的数百个候选项目；（2）基于检索出的项目，排序阶段旨在通过为每个候选项目进行多目标估计来选择最理想的十几个项目，包括分类和回归目标。与排序模型相比，检索模型在推理过程中缺乏候选项目的信息，因此检索模型往往仅通过分类目标（如点击率）进行训练，但未能纳入回归目标（如预期观看时长），这限制了检索的效果。在本文中，我们提出了可控检索模型（Controllable Retrieval Model, CRM），该模型将回归信息作为条件特征集成到两塔式检索框架中。这一修改使得检索阶段可以弥补与排序模型之间的目标差距，提升检索模型在搜索满足用户兴趣和条件的候选项目方面的能力。我们通过实际的A/B测试验证了CRM的有效性，并展示了其在快手短视频推荐系统中的成功部署，该系统服务于超过4亿用户。 

---
# Maybe you are looking for CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval 

**Title (ZH)**: 也许您正在寻找CroQS：跨模态查询建议用于文本到图像检索 

**Authors**: Giacomo Pacini, Fabio Carrara, Nicola Messina, Nicola Tonellotto, Giuseppe Amato, Fabrizio Falchi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13834)  

**Abstract**: Query suggestion, a technique widely adopted in information retrieval, enhances system interactivity and the browsing experience of document collections. In cross-modal retrieval, many works have focused on retrieving relevant items from natural language queries, while few have explored query suggestion solutions. In this work, we address query suggestion in cross-modal retrieval, introducing a novel task that focuses on suggesting minimal textual modifications needed to explore visually consistent subsets of the collection, following the premise of ''Maybe you are looking for''. To facilitate the evaluation and development of methods, we present a tailored benchmark named CroQS. This dataset comprises initial queries, grouped result sets, and human-defined suggested queries for each group. We establish dedicated metrics to rigorously evaluate the performance of various methods on this task, measuring representativeness, cluster specificity, and similarity of the suggested queries to the original ones. Baseline methods from related fields, such as image captioning and content summarization, are adapted for this task to provide reference performance scores. Although relatively far from human performance, our experiments reveal that both LLM-based and captioning-based methods achieve competitive results on CroQS, improving the recall on cluster specificity by more than 115% and representativeness mAP by more than 52% with respect to the initial query. The dataset, the implementation of the baseline methods and the notebooks containing our experiments are available here: this https URL 

**Abstract (ZH)**: 查询建议是一种广泛应用于信息检索的技术，可以增强系统的互动性和文档集合的浏览体验。在跨模态检索中，许多研究工作集中在从自然语言查询中检索相关项目，而很少探索查询建议解决方案。本研究中，我们针对跨模态检索中的查询建议问题，引入了一个新的任务，该任务旨在建议最小的文本修改，以便探索视觉上一致的集合子集，遵循“或许您正在寻找”的假设。为了促进该任务方法的评估和发展，我们提供了一个定制基准——CroQS。该数据集包括初始查询、分组结果集以及每个组的人工定义的建议查询。为了严格评估各种方法的性能，我们制定了专门的指标来衡量建议查询的代表性、聚类的特异性以及与原始查询的相似性。我们还针对该任务调整了来自相关领域的基础方法，如图像标题生成和内容摘要，以提供参考性能分数。尽管相比之下，这些方法的人类性能仍有一定差距，但在CroQS任务上，基于语言模型的方法和基于图像标题的方法均表现出竞争力，将聚类特异性的召回率提高了115%以上，代表性mAP提高了52%以上，相对于初始查询。数据集、基础方法的实现以及包含我们实验的笔记本均在此处提供：[此链接](this https URL)。 

---
# Heterogeneous Graph Collaborative Filtering 

**Title (ZH)**: 异构图协同过滤 

**Authors**: Lianghao Xia, Meiyan Xie, Yong Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13825)  

**Abstract**: For modern recommender systems, the use of low-dimensional latent representations to embed users and items based on their observed interactions has become commonplace. However, many existing recommendation models are primarily designed for coarse-grained and homogeneous interactions, which limits their effectiveness in two critical dimensions. Firstly, these models fail to leverage the relational dependencies that exist across different types of user behaviors, such as page views, collects, comments, and purchases. Secondly, they struggle to capture the fine-grained latent factors that drive user interaction patterns. To address these limitations, we present a heterogeneous graph collaborative filtering model MixRec that excels at disentangling users' multi-behavior interaction patterns and uncovering the latent intent factors behind each behavior. Our model achieves this by incorporating intent disentanglement and multi-behavior modeling, facilitated by a parameterized heterogeneous hypergraph architecture. Furthermore, we introduce a novel contrastive learning paradigm that adaptively explores the advantages of self-supervised data augmentation, thereby enhancing the model's resilience against data sparsity and expressiveness with relation heterogeneity. To validate the efficacy of MixRec, we conducted extensive experiments on three public datasets. The results clearly demonstrate its superior performance, significantly outperforming various state-of-the-art baselines. Our model is open-sourced and available at: this https URL. 

**Abstract (ZH)**: 对于现代推荐系统而言，利用低维度的潜在表示方法，根据用户和项目的观察到的交互来嵌入用户和项目已经成为常见做法。然而，许多现有的推荐模型主要是为粗粒度和同质化交互设计的，这在两个关键维度上限制了它们的有效性。首先，这些模型未能利用不同类型用户行为之间的关系依赖性，例如页面浏览、收藏、评论和购买。其次，它们难以捕捉到驱动用户交互模式的细微潜在因素。为了克服这些局限，我们提出了一种异质图协作过滤模型MixRec，该模型擅长分离用户的多行为交互模式，并揭示每种行为背后的潜在意图因素。我们的模型通过结合意图分离和多行为建模，利用参数化异质超图架构来实现这一点。此外，我们引入了一种新颖的对比学习范式，该范式能够自适应地探索自我监督数据增强的优势，从而提高模型在数据稀疏性和关系异质性下的鲁棒性和表达能力。为了验证MixRec的有效性，我们在三个公开数据集上进行了广泛的实验。结果显示，MixRec的表现明显优于各种先进的基线模型。我们的模型已经开源，并可在以下链接获取：this https URL。 

---
# Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization 

**Title (ZH)**: 语义收敛：通过两阶段对齐和行为语义词元化 harmonize 推荐系统 

**Authors**: Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13771)  

**Abstract**: Large language models (LLMs), endowed with exceptional reasoning capabilities, are adept at discerning profound user interests from historical behaviors, thereby presenting a promising avenue for the advancement of recommendation systems. However, a notable discrepancy persists between the sparse collaborative semantics typically found in recommendation systems and the dense token representations within LLMs. In our study, we propose a novel framework that harmoniously merges traditional recommendation models with the prowess of LLMs. We initiate this integration by transforming ItemIDs into sequences that align semantically with the LLMs space, through the proposed Alignment Tokenization module. Additionally, we design a series of specialized supervised learning tasks aimed at aligning collaborative signals with the subtleties of natural language semantics. To ensure practical applicability, we optimize online inference by pre-caching the top-K results for each user, reducing latency and improving effciency. Extensive experimental evidence indicates that our model markedly improves recall metrics and displays remarkable scalability of recommendation systems. 

**Abstract (ZH)**: larg的语言模型（LLMs）因其出众的推理能力，能够从用户的历史行为中洞察深层次的用户兴趣，从而为推荐系统的发展提供了新的前景。然而，推荐系统中常见的稀疏协作语义与LLMs中密集的词表示之间仍存在明显的差异。在我们的研究中，我们提出了一种新的框架，以和谐的方式将传统的推荐模型与LLMs的优势相结合。我们通过提出的对齐分词模块将ItemIDs转换为与LLMs空间相匹配的语义序列，从而启动这一整合过程。此外，我们设计了一系列专门的监督学习任务，旨在将协作信号与自然语言语义的细微差别对齐。为了确保其实用性，我们通过为每个用户预缓存前K个结果来优化在线推理，从而降低延迟并提高效率。大量的实验证据表明，我们的模型显著提高了召回率指标，并展示了推荐系统出色的可扩展性。 

---
# Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models 

**Title (ZH)**: 使用大型语言模型弥合知识aware推荐中的用户侧知识缺口 

**Authors**: Zheng Hu, Zhe Li, Ziyun Jiao, Satoshi Nakagawa, Jiawen Deng, Shimin Cai, Tao Zhou, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.13544)  

**Abstract**: In recent years, knowledge graphs have been integrated into recommender systems as item-side auxiliary information, enhancing recommendation accuracy. However, constructing and integrating structural user-side knowledge remains a significant challenge due to the improper granularity and inherent scarcity of user-side features. Recent advancements in Large Language Models (LLMs) offer the potential to bridge this gap by leveraging their human behavior understanding and extensive real-world knowledge. Nevertheless, integrating LLM-generated information into recommender systems presents challenges, including the risk of noisy information and the need for additional knowledge transfer. In this paper, we propose an LLM-based user-side knowledge inference method alongside a carefully designed recommendation framework to address these challenges. Our approach employs LLMs to infer user interests based on historical behaviors, integrating this user-side information with item-side and collaborative data to construct a hybrid structure: the Collaborative Interest Knowledge Graph (CIKG). Furthermore, we propose a CIKG-based recommendation framework that includes a user interest reconstruction module and a cross-domain contrastive learning module to mitigate potential noise and facilitate knowledge transfer. We conduct extensive experiments on three real-world datasets to validate the effectiveness of our method. Our approach achieves state-of-the-art performance compared to competitive baselines, particularly for users with sparse interactions. 

**Abstract (ZH)**: 近年来，知识图谱已被集成到推荐系统中作为项目侧辅助信息，以提高推荐准确性。然而，由于用户侧特征的不当粒度和固有的稀缺性，构建和整合结构化用户侧知识仍是一项重大挑战。大型语言模型（LLMs）的最新进展提供了通过利用其对人类行为的理解和广泛的实际世界知识来弥合这一差距的潜力。然而，将LLM生成的信息集成到推荐系统中也面临着挑战，包括噪声信息的风险以及额外知识迁移的需求。本文提出了一种基于LLM的用户侧知识推理方法以及一个精心设计的推荐框架，以应对这些挑战。我们的方法使用LLM根据历史行为推断用户兴趣，并将这种用户侧信息与项目侧和协作数据相结合，构建一个混合结构：协作兴趣知识图谱（CIKG）。此外，我们提出了一种基于CIKG的推荐框架，其中包括一个用户兴趣重构模块和一个跨域对比学习模块，以减轻潜在的噪声并促进知识迁移。在三个真实世界数据集上进行了广泛的实验以验证方法的有效性。与竞争基线相比，我们的方法在用户互动稀少的情况下达到了最先进的性能。 

---
# Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future 

**Title (ZH)**: 大型语言模型增强的推荐系统：分类、趋势、应用与未来 

**Authors**: Qidong Liu, Xiangyu Zhao, Yuhao Wang, Yejing Wang, Zijian Zhang, Yuqi Sun, Xiang Li, Maolin Wang, Pengyue Jia, Chong Chen, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.13432)  

**Abstract**: Large Language Model (LLM) has transformative potential in various domains, including recommender systems (RS). There have been a handful of research that focuses on empowering the RS by LLM. However, previous efforts mainly focus on LLM as RS, which may face the challenge of intolerant inference costs by LLM. Recently, the integration of LLM into RS, known as LLM-Enhanced Recommender Systems (LLMERS), has garnered significant interest due to its potential to address latency and memory constraints in real-world applications. This paper presents a comprehensive survey of the latest research efforts aimed at leveraging LLM to enhance RS capabilities. We identify a critical shift in the field with the move towards incorporating LLM into the online system, notably by avoiding their use during inference. Our survey categorizes the existing LLMERS approaches into three primary types based on the component of the RS model being augmented: Knowledge Enhancement, Interaction Enhancement, and Model Enhancement. We provide an in-depth analysis of each category, discussing the methodologies, challenges, and contributions of recent studies. Furthermore, we highlight several promising research directions that could further advance the field of LLMERS. 

**Abstract (ZH)**: 大规模语言模型（LLM）在各个领域具有变革性的潜力，包括推荐系统（RS）。已有少量研究关注通过LLM增强RS。然而，先前的努力主要集中在将LLM作为RS方面，这可能面临LLM承受不了的推理成本问题。最近，将LLM集成到RS中的做法，称为LLM增强推荐系统（LLMERS），因其在解决实际应用中的延迟和内存限制方面的潜力而引起了广泛关注。本文对最新研究工作进行了全面综述，旨在利用LLM增强RS能力。我们识别出了一个关键转变，即向将LLM集成到在线系统中迈进，特别避免在推理过程中使用LLM。我们按照增强RS模型组件的方式将现有的LLMERS方法分为三大类：知识增强、交互增强和模型增强。我们对每种类别进行了深入分析，讨论了近期研究的方法、挑战和贡献。此外，我们还指出了几个有前景的研究方向，这些方向有望进一步推动LLMERS领域的发展。 

---
# Nano-ESG: Extracting Corporate Sustainability Information from News Articles 

**Title (ZH)**: 纳米ESG：从新闻文章中提取企业可持续性信息 

**Authors**: Fabian Billert, Stefan Conrad  

**Link**: [PDF](https://arxiv.org/pdf/2412.15093)  

**Abstract**: Determining the sustainability impact of companies is a highly complex subject which has garnered more and more attention over the past few years. Today, investors largely rely on sustainability-ratings from established rating-providers in order to analyze how responsibly a company acts. However, those ratings have recently been criticized for being hard to understand and nearly impossible to reproduce.
An independent way to find out about the sustainability practices of companies lies in the rich landscape of news article data. In this paper, we explore a different approach to identify key opportunities and challenges of companies in the sustainability domain. We present a novel dataset of more than 840,000 news articles which were gathered for major German companies between January 2023 and September 2024. By applying a mixture of Natural Language Processing techniques, we first identify relevant articles, before summarizing them and extracting their sustainability-related sentiment and aspect using Large Language Models (LLMs). Furthermore, we conduct an evaluation of the obtained data and determine that the LLM-produced answers are accurate. We release both datasets at this https URL. 

**Abstract (ZH)**: 确定公司的可持续性影响是一个高度复杂的研究课题，近年来引起了越来越多的关注。如今，投资者大量依赖于知名评级机构提供的可持续性评级，以分析公司的负责任行为。然而，这些评级最近受到了批评，因为它们难以理解且几乎不可能复制。

一种独立于这些评级的方法是利用新闻文章数据丰富的生态环境来了解公司的可持续性实践情况。在本文中，我们探讨了一种不同的方法来识别公司在可持续性领域的关键机遇和挑战。我们提供了一个包含超过840,000篇新闻文章的新数据集，这些文章是为2023年1月至2024年9月期间的主要德国公司收集的。通过应用多种自然语言处理技术，我们首先识别出相关文章，然后对其进行总结，并使用大型语言模型（LLMs）提取其与可持续性相关的观点和方面。此外，我们对获取的数据进行了评估，并确定LLM生成的答案是准确的。我们在此在线链接中发布了这两个数据集：[提供链接]。 

---
# DisCo: Graph-Based Disentangled Contrastive Learning for Cold-Start Cross-Domain Recommendation 

**Title (ZH)**: DisCo：基于图的去纠缠对比学习在冷启动跨域推荐中的应用 

**Authors**: Hourun Li, Yifan Wang, Zhiping Xiao, Jia Yang, Changling Zhou, Ming Zhang, Wei Ju  

**Link**: [PDF](https://arxiv.org/pdf/2412.15005)  

**Abstract**: Recommender systems are widely used in various real-world applications, but they often encounter the persistent challenge of the user cold-start problem. Cross-domain recommendation (CDR), which leverages user interactions from one domain to improve prediction performance in another, has emerged as a promising solution. However, users with similar preferences in the source domain may exhibit different interests in the target domain. Therefore, directly transferring embeddings may introduce irrelevant source-domain collaborative information. In this paper, we propose a novel graph-based disentangled contrastive learning framework to capture fine-grained user intent and filter out irrelevant collaborative information, thereby avoiding negative transfer. Specifically, for each domain, we use a multi-channel graph encoder to capture diverse user intents. We then construct the affinity graph in the embedding space and perform multi-step random walks to capture high-order user similarity relationships. Treating one domain as the target, we propose a disentangled intent-wise contrastive learning approach, guided by user similarity, to refine the bridging of user intents across domains. Extensive experiments on four benchmark CDR datasets demonstrate that DisCo consistently outperforms existing state-of-the-art baselines, thereby validating the effectiveness of both DisCo and its components. 

**Abstract (ZH)**: 推荐系统在各种实际应用中被广泛应用，但它们经常遇到持久存在的用户冷启动问题。多域推荐（Cross-Domain Recommendation, CDR）通过利用一个领域中的用户交互来提高另一个领域中的预测性能，已经作为一种有潜力的解决方案而出现。然而，在源领域中具有相似偏好的用户在目标领域中可能表现出不同的兴趣。因此，直接进行嵌入传递可能会引入与目标领域无关的协作信息。在本文中，我们提出了一个新颖的基于图的解耦对比学习框架，该框架能够捕捉细微的用户意图，并过滤掉无关的协作信息，从而避免负迁移。具体来说，对于每个领域，我们使用多通道图编码器来捕捉多样化的用户意图。然后我们在嵌入空间中构建亲和图，并执行多步随机漫步以捕捉高阶的用户相似关系。我们将一个领域视为目标领域，提出了基于用户相似性的意图解耦对比学习方法，通过这些方法来精细化用户意图在不同领域之间的转移。在四个基准CDR数据集上的广泛实验表明，DisCo在所有情况下均优于现有的先进基线方法，这验证了DisCo和其各组成部分的有效性。 

---
# Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation 

**Title (ZH)**: 基于频谱的模态表示融合图卷积网络在多模态推荐中的应用 

**Authors**: Rongqing Kenneth Ong, Andy W. H. Khong  

**Link**: [PDF](https://arxiv.org/pdf/2412.14978)  

**Abstract**: Incorporating multi-modal features as side information has recently become a trend in recommender systems. To elucidate user-item preferences, recent studies focus on fusing modalities via concatenation, element-wise sum, or attention mechanisms. Despite having notable success, existing approaches do not account for the modality-specific noise encapsulated within each modality. As a result, direct fusion of modalities will lead to the amplification of cross-modality noise. Moreover, the variation of noise that is unique within each modality results in noise alleviation and fusion being more challenging. In this work, we propose a new Spectrum-based Modality Representation (SMORE) fusion graph recommender that aims to capture both uni-modal and fusion preferences while simultaneously suppressing modality noise. Specifically, SMORE projects the multi-modal features into the frequency domain and leverages the spectral space for fusion. To reduce dynamic contamination that is unique to each modality, we introduce a filter to attenuate and suppress the modality noise adaptively while capturing the universal modality patterns effectively. Furthermore, we explore the item latent structures by designing a new multi-modal graph learning module to capture associative semantic correlations and universal fusion patterns among similar items. Finally, we formulate a new modality-aware preference module, which infuses behavioral features and balances the uni- and multi-modal features for precise preference modeling. This empowers SMORE with the ability to infer both user modality-specific and fusion preferences more accurately. Experiments on three real-world datasets show the efficacy of our proposed model. The source code for this work has been made publicly available at this https URL. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，并符合学术规范：

将多模态特征作为辅助信息在推荐系统中最近已成为一种趋势。为了阐明用户和项目的偏好，最近的研究主要集中在通过连接、逐元素求和或注意力机制来融合模态。尽管这些方法在某些方面取得了显著成功，但现有的方法并未考虑到每个模态内固有的模态特定噪声。因此，直接融合模态会放大跨模态噪声。此外，由于每个模态内特有的噪声变异，噪声减弱和融合变得更加具有挑战性。在这项工作中，我们提出了一种新的基于频谱的模态表示（SMORE）融合图推荐系统，旨在同时捕捉单模态和融合偏好，同时抑制模态噪声。具体而言，SMORE 将多模态特征投影到频率域，并利用频谱空间进行融合。为了减少每个模态特有的动态污染，我们引入了滤波器，以适应性地削弱并抑制模态噪声，同时有效地捕捉通用的模态模式。此外，我们通过设计一个新的多模态图学习模块来探索项目潜隐结构，以捕获相似项目之间的关联语义关联和通用融合模式。最后，我们提出了一个新的模态感知偏好模块，在此基础上将行为特征注入系统，并平衡单模态和多模态特征，以实现精确的偏好建模。这赋予了SMORE更准确地推断用户模态特定偏好和融合偏好能力。在三个真实数据集上的实验表明了我们提出模型的有效性。该工作的源代码已在此处公开（请提供链接）。 

---
# ECLIPSE: Contrastive Dimension Importance Estimation with Pseudo-Irrelevance Feedback for Dense Retrieval 

**Title (ZH)**: ECLIPSE：基于伪无关反馈的对比维度重要性估计算法在密集检索中的应用 

**Authors**: Giulio D'Erasmo, Giovanni Trappolini, Nicola Tonellotto, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2412.14967)  

**Abstract**: Recent advances in Information Retrieval have leveraged high-dimensional embedding spaces to improve the retrieval of relevant documents. Moreover, the Manifold Clustering Hypothesis suggests that despite these high-dimensional representations, documents relevant to a query reside on a lower-dimensional, query-dependent manifold. While this hypothesis has inspired new retrieval methods, existing approaches still face challenges in effectively separating non-relevant information from relevant signals. We propose a novel methodology that addresses these limitations by leveraging information from both relevant and non-relevant documents. Our method, ECLIPSE, computes a centroid based on irrelevant documents as a reference to estimate noisy dimensions present in relevant ones, enhancing retrieval performance. Extensive experiments on three in-domain and one out-of-domain benchmarks demonstrate an average improvement of up to 19.50% (resp. 22.35%) in mAP(AP) and 11.42% (resp. 13.10%) in nDCG@10 w.r.t. the DIME-based baseline (resp. the baseline using all dimensions). Our results pave the way for more robust, pseudo-irrelevance-based retrieval systems in future IR research. 

**Abstract (ZH)**: 近期在信息检索领域的进展利用高维嵌入空间提高了相关文档的检索效果。此外，流形聚类假设表明，尽管存在这些高维表示，与查询相关的文档实际上位于查询依赖的低维流形上。虽然这一假设启发了新的检索方法，但现有方法仍面临有效分离非相关信息与相关信号的挑战。为此，我们提出了一种新的方法，通过利用相关和非相关文档的信息来解决这些限制。我们的方法ECLIPSE通过将非相关文档作为参考来计算中心点，以估计相关文档中存在的噪声维度，从而提升检索性能。在三项领域内和一项领域外基准上的广泛实验表明，与基于DIME的方法（分别为22.35%）相比，ECLIPSE在mAP(AP)上的平均改进达到了19.50%，在nDCG@10上的平均改进达到了13.10%（分别为11.42%）。我们的结果为未来信息检索研究中更稳健、基于伪无关的检索系统铺平了道路。 

---
# Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models 

**Title (ZH)**: 滑动窗口不是终点：探索长语境大型语言模型的全程排名 

**Authors**: Wenhan Liu, Xinyu Ma, Yutao Zhu, Ziliang Zhao, Shuaiqiang Wang, Dawei Yin, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2412.14574)  

**Abstract**: Large Language Models (LLMs) have shown exciting performance in listwise passage ranking. Due to the limited input length, existing methods often adopt the sliding window strategy. Such a strategy, though effective, is inefficient as it involves repetitive and serialized processing, which usually re-evaluates relevant passages multiple times. As a result, it incurs redundant API costs, which are proportional to the number of inference tokens. The development of long-context LLMs enables the full ranking of all passages within a single inference, avoiding redundant API costs. In this paper, we conduct a comprehensive study of long-context LLMs for ranking tasks in terms of efficiency and effectiveness. Surprisingly, our experiments reveal that full ranking with long-context LLMs can deliver superior performance in the supervised fine-tuning setting with a huge efficiency improvement. Furthermore, we identify two limitations of fine-tuning the full ranking model based on existing methods: (1) sliding window strategy fails to produce a full ranking list as a training label, and (2) the language modeling loss cannot emphasize top-ranked passage IDs in the label. To alleviate these issues, we propose a new complete listwise label construction approach and a novel importance-aware learning objective for full ranking. Experiments show the superior performance of our method over baselines. Our codes are available at \url{this https URL}. 

**Abstract (ZH)**: 以下是论文内容或标题的中文翻译，符合学术规范：

大规模语言模型（LLMs）在列表式段落排名任务中展现了令人兴奋的性能。由于输入长度有限，现有方法常采用滑动窗口策略。虽然这种策略的有效性已得到证实，但由于其涉及重复且串行的处理过程，通常需要多次重新评估相关段落，导致计算开销的累积，这些开销与推理 token 数量成比例。长语境 LLM 的发展使得能够在单次推理中全面排名所有段落，从而避免重复 API 成本。在本文中，我们对长语境 LLM 在排名任务中的效率和有效性进行了全面研究。令人惊讶的是，我们的实验表明，在监督微调设置中，使用长语境 LLM 进行全面排名可以显著提高性能，并提高效率。此外，我们还指出了基于现有方法微调全面排名模型的两个局限性：（1）滑动窗口策略无法生成完整的排名列表作为训练标签；（2）语言建模损失不能强化标签中排名靠前的段落 ID。为了解决这些问题，我们提出了一种新的完整列表式标签构建方法和一种新的关注重要性的学习目标，以全面排名。实验结果表明，我们的方法在基准方法上具有优越的性能。我们的代码可在 \url{https://…} 获取。 

---
# HEC-GCN: Hypergraph Enhanced Cascading Graph Convolution Network for Multi-Behavior Recommendation 

**Title (ZH)**: HEC-GCN：增强超图的级联图卷积网络在多行为推荐中的应用 

**Authors**: Yabo Yin, Xiaofei Zhu, Wenshan Wang, Yihao Zhang, Pengfei Wang, Yixing Fan, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2412.14476)  

**Abstract**: Multi-behavior recommendation (MBR) has garnered growing attention recently due to its ability to mitigate the sparsity issue by inferring user preferences from various auxiliary behaviors to improve predictions for the target behavior. Although existing research on MBR has yielded impressive results, they still face two major limitations. First, previous methods mainly focus on modeling fine-grained interaction information between users and items under each behavior, which may suffer from sparsity issue. Second, existing models usually concentrate on exploiting dependencies between two consecutive behaviors, leaving intra- and inter-behavior consistency largely unexplored. To the end, we propose a novel approach named Hypergraph Enhanced Cascading Graph Convolution Network for multi-behavior recommendation (HEC-GCN). To be specific, we first explore both fine- and coarse-grained correlations among users or items of each behavior by simultaneously modeling the behavior-specific interaction graph and its corresponding hypergraph in a cascaded manner. Then, we propose a behavior consistency-guided alignment strategy that ensures consistent representations between the interaction graph and its associated hypergraph for each behavior, while also maintaining representation consistency across different behaviors. Extensive experiments and analyses on three public benchmark datasets demonstrate that our proposed approach is consistently superior to previous state-of-the-art methods due to its capability to effectively attenuate the sparsity issue as well as preserve both intra- and inter-behavior consistencies. The code is available at this https URL. 

**Abstract (ZH)**: 多行为推荐（MBR）最近引起了广泛关注，因为它能够通过从各种辅助行为中推断用户偏好来缓解目标行为预测中的稀疏性问题。尽管现有的MBR研究取得了显著成果，但仍存在两大局限性。首先，以往的方法主要侧重于在每种行为下建模用户与物品之间的精细交互信息，这可能会导致稀疏性问题。其次，现有的模型通常专注于挖掘两种连续行为之间的依赖关系，而忽略了行为内的和行为间的连贯性。为了解决这些问题，我们提出了一种名为Hypergraph Enhanced Cascading Graph Convolution Network for多行为推荐（HEC-GCN）的新方法。具体而言，我们首先通过依次建模每种行为的特性交互图及其相应的超图来探索用户或物品在每种行为中的细粒度和粗粒度相关性。然后，我们提出了一种行为一致性引导的对齐策略，该策略确保了每种行为下的交互图与其相关超图的一致性表示，并且在不同行为之间也保持了一致性表示。在三个公开基准数据集上的广泛实验和分析表明，我们的方法在缓解稀疏性问题和保持行为内的和行为间的连贯性方面，优于之前的先进方法。代码可以在以下链接获取：this https URL。 

---
# VISA: Retrieval Augmented Generation with Visual Source Attribution 

**Title (ZH)**: VISA：具有视觉来源归因的检索增强生成 

**Authors**: Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Wenhu Chen, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14457)  

**Abstract**: Generation with source attribution is important for enhancing the verifiability of retrieval-augmented generation (RAG) systems. However, existing approaches in RAG primarily link generated content to document-level references, making it challenging for users to locate evidence among multiple content-rich retrieved documents. To address this challenge, we propose Retrieval-Augmented Generation with Visual Source Attribution (VISA), a novel approach that combines answer generation with visual source attribution. Leveraging large vision-language models (VLMs), VISA identifies the evidence and highlights the exact regions that support the generated answers with bounding boxes in the retrieved document screenshots. To evaluate its effectiveness, we curated two datasets: Wiki-VISA, based on crawled Wikipedia webpage screenshots, and Paper-VISA, derived from PubLayNet and tailored to the medical domain. Experimental results demonstrate the effectiveness of VISA for visual source attribution on documents' original look, as well as highlighting the challenges for improvement. Code, data, and model checkpoints will be released. 

**Abstract (ZH)**: 源文追溯的生成对于增强检索增强生成（RAG）系统的可验证性非常重要。然而，现有的RAG方法主要将生成的内容与文档级别的引用关联起来，这使得用户在多个内容丰富的检索文档中定位证据变得颇具挑战性。为解决这一问题，我们提出了一种名为视觉源追溯的检索增强生成（VISA）的新颖方法，该方法结合了答案生成与视觉源追溯。借助大型视觉语言模型（VLMs），VISA能够在检索文档截图中标注支撑生成答案的具体证据并用边界框突出显示这些区域。为评估其有效性，我们构建了两个数据集：基于爬取的维基百科网页截图的Wiki-VISA和基于PubLayNet构建并针对医疗领域进行调整的Paper-VISA。实验结果表明，VISA在文档原始布局的视觉源追溯方面有效性很高，同时也指出了改进的挑战。我们将公布代码、数据和模型检查点。 

---
# Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems 

**Title (ZH)**: 更长的提示是否总是更好？大型语言模型在推荐系统中的提示选择研究 

**Authors**: Genki Kusano, Kosuke Akimoto, Kunihiro Takeoka  

**Link**: [PDF](https://arxiv.org/pdf/2412.14454)  

**Abstract**: In large language models (LLM)-based recommendation systems (LLM-RSs), accurately predicting user preferences by leveraging the general knowledge of LLMs is possible without requiring extensive training data. By converting recommendation tasks into natural language inputs called prompts, LLM-RSs can efficiently solve issues that have been difficult to address due to data scarcity but are crucial in applications such as cold-start and cross-domain problems. However, when applying this in practice, selecting the prompt that matches tasks and data is essential. Although numerous prompts have been proposed in LLM-RSs and representing the target user in prompts significantly impacts recommendation accuracy, there are still no clear guidelines for selecting specific prompts.
In this paper, we categorize and analyze prompts from previous research to establish practical prompt selection guidelines. Through 450 experiments with 90 prompts and five real-world datasets, we examined the relationship between prompts and dataset characteristics in recommendation accuracy. We found that no single prompt consistently outperforms others; thus, selecting prompts on the basis of dataset characteristics is crucial. Here, we propose a prompt selection method that achieves higher accuracy with minimal validation data. Because increasing the number of prompts to explore raises costs, we also introduce a cost-efficient strategy using high-performance and cost-efficient LLMs, significantly reducing exploration costs while maintaining high prediction accuracy. Our work offers valuable insights into the prompt selection, advancing accurate and efficient LLM-RSs. 

**Abstract (ZH)**: 在基于大规模语言模型（LLM）的推荐系统（LLM-RS）中，通过利用LLM的通用知识，无需大量训练数据即可准确预测用户偏好。通过将推荐任务转换为称为提示的自然语言输入，LLM-RS可以高效解决由于数据稀缺而在冷启动和跨域问题等实际应用中难以解决的问题。然而，在实际应用中，选择与任务和数据相匹配的提示至关重要。尽管在LLM-RS中已经提出了多种提示，并且提示中对目标用户的表示显著影响推荐精度，但仍没有明确的指南来选择特定的提示。

本文将前人的提示进行分类和分析，以制定实际的提示选择指南。通过450次实验，使用90种提示和五个真实世界的数据集，我们考察了提示和数据集特性与推荐精度之间的关系。我们发现，并没有单一的提示在所有情况下都能最好；因此，根据数据集特性选择提示是至关重要的。在此基础上，我们提出了一种在验证数据量极少的情况下能实现更高精度的提示选择方法。由于增加提示的数量以探索会提高成本，我们还提出了使用高性能且成本效益高的LLM的成本效益策略，显著降低了探索成本的同时保持了高预测精度。我们的工作为提示选择提供了宝贵的见解，促进了准确和高效的LLM-RS的发展。 

---
# ChainRank-DPO: Chain Rank Direct Preference Optimization for LLM Rankers 

**Title (ZH)**: 链路排名-DPO：链路排名直接偏好优化 

**Authors**: Haowei Liu, Xuyang Wu, Guohao Sun, Zhiqiang Tao, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2412.14405)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable effectiveness in text reranking through works like RankGPT, leveraging their human-like reasoning about relevance. However, supervised fine-tuning for ranking often diminishes these models' general-purpose capabilities, including the crucial reasoning abilities that make them valuable for ranking. We introduce a novel approach integrating Chain-of-Thought prompting with an SFT-DPO (Supervised Fine-Tuning followed by Direct Preference Optimization) pipeline to preserve these capabilities while improving ranking performance. Our experiments on TREC 2019 and 2020 Deep Learning datasets show that our approach outperforms the state-of-the-art RankZephyr while maintaining strong performance on the Massive Multitask Language Understanding (MMLU) benchmark, demonstrating effective preservation of general-purpose capabilities through thoughtful fine-tuning strategies. Our code and data will be publicly released upon the acceptance of the paper. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过如RankGPT等研究工作在文本重排序方面表现出了显著的效果，并借助其类似人类的关联性推理。然而，排名相关的监督微调往往会削弱这些模型的通用能力，包括对它们作为排名工具至关重要的推理能力。为此，我们提出了一种新颖的方法，将链式思维提示（Chain-of-Thought prompting）与一个SFT-DPO（监督微调后直接偏好优化）管道相结合，以在保持这些能力的同时提高排名性能。我们在TREC 2019和2020年的深度学习数据集上的实验表明，我们的方法在最先进的RankZephyr上表现更优，并且在大规模多任务语言理解（MMLU）基准测试中保持了较强的性能，这证明了通过仔细的微调策略有效保留了通用能力。论文被接受后，我们将公开发布我们的代码和数据。 

---
# Embedding Cultural Diversity in Prototype-based Recommender Systems 

**Title (ZH)**: 将文化多样性嵌入原型推荐系统中 

**Authors**: Armin Moradi, Nicola Neophytou, Florian Carichon, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2412.14329)  

**Abstract**: Popularity bias in recommender systems can increase cultural overrepresentation by favoring norms from dominant cultures and marginalizing underrepresented groups. This issue is critical for platforms offering cultural products, as they influence consumption patterns and human perceptions. In this work, we address popularity bias by identifying demographic biases within prototype-based matrix factorization methods. Using the country of origin as a proxy for cultural identity, we link this demographic attribute to popularity bias by refining the embedding space learning process. First, we propose filtering out irrelevant prototypes to improve representativity. Second, we introduce a regularization technique to enforce a uniform distribution of prototypes within the embedding space. Across four datasets, our results demonstrate a 27\% reduction in the average rank of long-tail items and a 2\% reduction in the average rank of items from underrepresented countries. Additionally, our model achieves a 2\% improvement in HitRatio@10 compared to the state-of-the-art, highlighting that fairness is enhanced without compromising recommendation quality. Moreover, the distribution of prototypes leads to more inclusive explanations by better aligning items with diverse prototypes. 

**Abstract (ZH)**: 推荐系统中的流行度偏差会通过偏向主流文化规范而增加文化过度代表，同时边缘化代表性不足的群体。这对于提供文化产品的平台来说是一个关键问题，因为这些平台会影响消费模式和人类认知。在本研究中，我们通过识别基于原型的矩阵分解方法中的种族偏差来应对流行度偏差。以国家为来源作为文化身份的代理，我们通过细化嵌入空间的学习过程将这一种族属性与流行度偏差联系起来。首先，我们建议过滤掉与任务无关的原型以提高代表性。其次，我们引入一种正则化技术来确保嵌入空间内原型的均匀分布。在四个数据集中，我们的结果显示出长尾项目的平均排名降低了27%，而来自代表性不足国家的项目的平均排名降低了2%。此外，与当前最先进的方法相比，我们的模型在HitRatio@10方面提高了2%，这表明在不牺牲推荐质量的情况下提升了公平性。此外，原型的分布导致了更具包容性的解释，因为它们更好地与多样性的项目对齐。 

---
# SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation 

**Title (ZH)**: SAFERec：自我注意力和频率增强模型的下一购物篮推荐算法 

**Authors**: Oleg Lashinin, Denis Krasilnikov, Aleksandr Milogradskii, Marina Ananyeva  

**Link**: [PDF](https://arxiv.org/pdf/2412.14302)  

**Abstract**: Transformer-based approaches such as BERT4Rec and SASRec demonstrate strong performance in Next Item Recommendation (NIR) tasks. However, applying these architectures to Next-Basket Recommendation (NBR) tasks, which often involve highly repetitive interactions, is challenging due to the vast number of possible item combinations in a basket. Moreover, frequency-based methods such as TIFU-KNN and UP-CF still demonstrate strong performance in NBR tasks, frequently outperforming deep-learning approaches. This paper introduces SAFERec, a novel algorithm for NBR that enhances transformer-based architectures from NIR by incorporating item frequency information, consequently improving their applicability to NBR tasks. Extensive experiments on multiple datasets show that SAFERec outperforms all other baselines, specifically achieving an 8\% improvement in Recall@10. 

**Abstract (ZH)**: 基于Transformer的方法，如BERT4Rec和SASRec，在Next Item Recommendation (NIR)任务中表现出色。然而，将这些架构应用到Next-Basket Recommendation (NBR)任务中时，由于篮子中可能的项目组合数量庞大，且NBR任务经常涉及高度重复的交互，因此面临挑战。此外，基于频率的方法，如TIFU-KNN和UP-CF在NBR任务中仍然表现出色，经常优于深度学习方法。本文提出了一种新的算法SAFERec，该算法通过整合项目频率信息增强了NIR中的Transformer架构，从而使其更适用于NBR任务。在多种数据集上的广泛实验表明，SAFERec在所有基准方法中表现最佳，特别是在Recall@10上提高了8%。 

---
# Progressive Multimodal Reasoning via Active Retrieval 

**Title (ZH)**: 逐步多模态推理通过主动检索 

**Authors**: Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2412.14835)  

**Abstract**: Multi-step multimodal reasoning tasks pose significant challenges for multimodal large language models (MLLMs), and finding effective ways to enhance their performance in such scenarios remains an unresolved issue. In this paper, we propose AR-MCTS, a universal framework designed to progressively improve the reasoning capabilities of MLLMs through Active Retrieval (AR) and Monte Carlo Tree Search (MCTS). Our approach begins with the development of a unified retrieval module that retrieves key supporting insights for solving complex reasoning problems from a hybrid-modal retrieval corpus. To bridge the gap in automated multimodal reasoning verification, we employ the MCTS algorithm combined with an active retrieval mechanism, which enables the automatic generation of step-wise annotations. This strategy dynamically retrieves key insights for each reasoning step, moving beyond traditional beam search sampling to improve the diversity and reliability of the reasoning space. Additionally, we introduce a process reward model that aligns progressively to support the automatic verification of multimodal reasoning tasks. Experimental results across three complex multimodal reasoning benchmarks confirm the effectiveness of the AR-MCTS framework in enhancing the performance of various multimodal models. Further analysis demonstrates that AR-MCTS can optimize sampling diversity and accuracy, yielding reliable multimodal reasoning. 

**Abstract (ZH)**: 多步多模态推理任务对多模态大型语言模型（MLLMs）构成了显著挑战，如何在这些场景中有效提升其性能仍然是未解决的问题。本文提出了一种名为AR-MCTS的通用框架，旨在通过主动检索（AR）和蒙特卡洛树搜索（MCTS）逐步提高MLLMs的推理能力。该方法首先开发了一个统一的检索模块，从混合模态检索语料库中获取解决复杂推理问题的关键支持见解。为解决自动化多模态推理验证中的缺口，我们采用了结合主动检索机制的蒙特卡洛树搜索算法，这能够自动生成逐步注释。该策略动态地为每个推理步骤检索关键见解，超越传统的束搜索采样，以提高推理空间的多样性和可靠性。此外，我们引入了一个过程奖励模型，旨在逐步支持多模态推理任务的自动验证。在三个复杂的多模态推理基准测试中的实验结果证实了AR-MCTS框架在提升各种多模态模型性能方面的有效性。进一步的分析表明，AR-MCTS可以优化采样多样性和准确性，从而实现可靠的多模态推理。 

---
# Efficient Self-Supervised Video Hashing with Selective State Spaces 

**Title (ZH)**: 高效的选择性状态空间自我监督视频哈希 

**Authors**: Jinpeng Wang, Niu Lian, Jun Li, Yuting Wang, Yan Feng, Bin Chen, Yongbing Zhang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2412.14518)  

**Abstract**: Self-supervised video hashing (SSVH) is a practical task in video indexing and retrieval. Although Transformers are predominant in SSVH for their impressive temporal modeling capabilities, they often suffer from computational and memory inefficiencies. Drawing inspiration from Mamba, an advanced state-space model, we explore its potential in SSVH to achieve a better balance between efficacy and efficiency. We introduce S5VH, a Mamba-based video hashing model with an improved self-supervised learning paradigm. Specifically, we design bidirectional Mamba layers for both the encoder and decoder, which are effective and efficient in capturing temporal relationships thanks to the data-dependent selective scanning mechanism with linear complexity. In our learning strategy, we transform global semantics in the feature space into semantically consistent and discriminative hash centers, followed by a center alignment loss as a global learning signal. Our self-local-global (SLG) paradigm significantly improves learning efficiency, leading to faster and better convergence. Extensive experiments demonstrate S5VH's improvements over state-of-the-art methods, superior transferability, and scalable advantages in inference efficiency. Code is available at this https URL. 

**Abstract (ZH)**: 自监督视频哈希（SSVH）是视频索引和检索中的一项实用任务。虽然变压器由于其出色的时序建模能力在SSVH中占据主导地位，但它们经常面临计算和内存效率低的问题。受到Mamba（一种先进的状态空间模型）的启发，我们探索了将其应用于SSVH，以实现高效性和效用之间的更好平衡。我们提出了S5VH，一种基于Mamba的视频哈希模型，具有改进的自监督学习范式。具体而言，我们设计了双向的Mamba层，无论是编码器还是解码器，都能通过依赖于数据的选择性扫描机制捕获时序关系，具有线性复杂度，这使得捕获时序关系既有效又高效。在我们的学习策略中，我们将特征空间中的全局语义转换为语义一致且具有判别力的哈希中心，随后通过中心对齐损失作为全局学习信号。我们的局部-全局（SLG）范式显著提高了学习效率，导致更快更好的收敛。广泛的经验表明，S5VH相较于最先进的方法具有改进效果、更好的可迁移性和可伸缩优势的推理效率。代码可从以下链接获取：请替换为实际链接。 

---
# Moving Beyond LDA: A Comparison of Unsupervised Topic Modelling Techniques for Qualitative Data Analysis of Online Communities 

**Title (ZH)**: 超越LDA：无监督主题建模技术在在线社区定性数据分析中的比较 

**Authors**: Amandeep Kaur, James R. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2412.14486)  

**Abstract**: Social media constitutes a rich and influential source of information for qualitative researchers. Although computational techniques like topic modelling assist with managing the volume and diversity of social media content, qualitative researcher's lack of programming expertise creates a significant barrier to their adoption. In this paper we explore how BERTopic, an advanced Large Language Model (LLM)-based topic modelling technique, can support qualitative data analysis of social media. We conducted interviews and hands-on evaluations in which qualitative researchers compared topics from three modelling techniques: LDA, NMF, and BERTopic. BERTopic was favoured by 8 of 12 participants for its ability to provide detailed, coherent clusters for deeper understanding and actionable insights. Participants also prioritised topic relevance, logical organisation, and the capacity to reveal unexpected relationships within the data. Our findings underscore the potential of LLM-based techniques for supporting qualitative analysis. 

**Abstract (ZH)**: 社交媒体是一种丰富且有影响力的定性研究信息来源。虽然如主题建模等计算技术有助于处理社交媒体内容的数量和多样性，但由于定性研究者缺乏编程技能，这成为他们采用这些技术的一个重大障碍。本文探索了基于大型语言模型（LLM）的主题建模技术BERTopic如何支持社交媒体的定性数据分析。我们通过采访和实际操作评估，让定性研究者将三种建模技术——LDA、NMF 和 BERTopic——产生的主题进行了对比。结果显示，有12名参与者中有8人偏好BERTopic，因其能够提供详细的、逻辑连贯的主题群集，以实现更深入的理解和可操作的见解。参与者还优先考虑主题的相关性、逻辑组织以及揭示数据中未预期关系的能力。我们的研究结果强调了基于大型语言模型技术在支持定性分析方面的潜力。 

---
# State Space Models are Strong Text Rerankers 

**Title (ZH)**: 状态空间模型是强大的文本重排序器 

**Authors**: Zhichao Xu, Jinghua Yan, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2412.14354)  

**Abstract**: Transformers dominate NLP and IR; but their inference inefficiencies and challenges in extrapolating to longer contexts have sparked interest in alternative model architectures. Among these, state space models (SSMs) like Mamba offer promising advantages, particularly $O(1)$ time complexity in inference. Despite their potential, SSMs' effectiveness at text reranking -- a task requiring fine-grained query-document interaction and long-context understanding -- remains underexplored.
This study benchmarks SSM-based architectures (specifically, Mamba-1 and Mamba-2) against transformer-based models across various scales, architectures, and pre-training objectives, focusing on performance and efficiency in text reranking tasks. We find that (1) Mamba architectures achieve competitive text ranking performance, comparable to transformer-based models of similar size; (2) they are less efficient in training and inference compared to transformers with flash attention; and (3) Mamba-2 outperforms Mamba-1 in both performance and efficiency. These results underscore the potential of state space models as a transformer alternative and highlight areas for improvement in future IR applications. 

**Abstract (ZH)**: _transformers 在自然语言处理 (NLP) 和信息检索 (IR) 领域占据主导地位；但由于其推理效率低下和在处理更长上下文时的外推挑战，人们对其替代模型架构产生了兴趣。在这之中，状态空间模型 (SSMs) 如 Mamba 提供了潜在的优势，特别是推理的时间复杂度为 \(O(1)\)。尽管 SSMs 具有潜在优势，但其在文本重排序任务中的有效性——这一任务需要精细的查询文档交互和长上下文理解——仍被关注不足。

本研究通过在不同规模、架构和预训练目标下将基于 SSM 的架构（具体而言是 Mamba-1 和 Mamba-2）与基于 transformers 的模型进行基准测试，重点评估了在文本重排序任务中的性能和效率。我们发现：(1) Mamba 架构在文本排名性能上达到与相似大小的 transformers 型模型竞争的水平；(2) 在训练和推理效率方面，它们不如使用闪存注意力机制的 transformers；(3) Mamba-2 在性能和效率上都优于 Mamba-1。这些结果表明了状态空间模型作为 transformers 替代品的潜力，并指出了未来信息检索应用中需要改进的领域。_

（注：在翻译过程中，对原文中的技术细节进行了适当的调整，以适应中文表达习惯，同时保持了原文的科学严谨性。） 

---
# Transversal PACS Browser API: Addressing Interoperability Challenges in Medical Imaging Systems 

**Title (ZH)**: 横截面PACS浏览器API：解决医学成像系统中的互操作性挑战 

**Authors**: Diogo Lameira, Filipa Ferraz  

**Link**: [PDF](https://arxiv.org/pdf/2412.14229)  

**Abstract**: Advances in imaging technologies have revolutionised the medical imaging and healthcare sectors, leading to the widespread adoption of PACS for the storage, retrieval, and communication of medical images. Although these systems have improved operational efficiency, significant challenges remain in effectively retrieving DICOM images, which are essential for diagnosis and overall patient care. Moreover, issues such as fragmented systems, interoperability barriers, and complex user interfaces can often prevent healthcare professionals from efficiently accessing medical images. Addressing these challenges, the Transversal PACS Browser API is a robust and user-friendly solution designed to enhance the process of querying and retrieving DICOM images. It offers advanced filtering capabilities through a variety of filter options as well as a custom field search, that allows users to easily navigate through large medical image collections with ease. Additionally, the application provides a unified interface for querying and retrieving from multiple PACS stations, addressing the challenges of fragmentation and complexity associated with accessing medical images. Other key features include the ability to pre-view images directly within the application. All of this contributes to the transversal nature of the API, serving not only healthcare providers, but anyone who relies on efficient access to these resources. To validate the performance and usability of the application, comprehensive testing was carried out with stakeholders of the field, the results of which showed general satisfaction, highlighting the API's clean design, ease of use, and effective search capabilities of the API, as well as the usefulness of previewing images within the application. 

**Abstract (ZH)**: 成像技术的进步彻底改变了医学成像和医疗保健领域，推动了PACS（Picture archiving and communication system，影像归档和通讯系统）的广泛应用。PACS系统用于存储、检索和传输医学影像，虽然提高了操作效率，但仍存在一些有效检索DICOM（Digital Imaging and Communications in Medicine，医学数字成像与通信）影像的重要挑战。这些问题包括系统碎片化、互联互通障碍以及复杂的人机接口，往往妨碍了医疗专业人员高效访问影像数据。为了应对这些挑战，Transversal PACS浏览器API提供了一个强大且用户友好的解决方案，旨在增强DICOM影像的查询和检索过程。该API提供多种过滤选项和定制字段搜索功能，使用户能够轻松导航庞大的医学影像集合。此外，该应用程序提供了一个统一界面，用于从多个PACS工作站查询和检索影像数据，解决了访问影像数据时存在的碎片化和复杂性问题。其他关键功能包括直接在应用程序中预览图像。所有这些特性都体现了该API的横向性质，不仅服务于医疗提供者，也满足依赖于高效访问这些资源的所有人员的需求。为了验证应用程序的性能和用户体验，对该领域的关键利益相关者进行了全面测试，测试结果表明，大多数用户对API的设计简洁、使用简便以及有效的搜索功能给予了高度评价，并强调了在应用程序中预览影像的实用性。 

---
# Whom do Explanations Serve? A Systematic Literature Survey of User Characteristics in Explainable Recommender Systems Evaluation 

**Title (ZH)**: 解释性推荐系统评估中的用户特征：一项关于解释目的的系统文献综述

这个标题翻译成中文既保留了原文的意思，又符合学术规范。如果您有更具体的翻译需求或想要进一步的修改，请告诉我。 

**Authors**: Kathrin Wardatzky, Oana Inel, Luca Rossetto, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2412.14193)  

**Abstract**: Adding explanations to recommender systems is said to have multiple benefits, such as increasing user trust or system transparency. Previous work from other application areas suggests that specific user characteristics impact the users' perception of the explanation. However, we rarely find this type of evaluation for recommender systems explanations. This paper addresses this gap by surveying 124 papers in which recommender systems explanations were evaluated in user studies. We analyzed their participant descriptions and study results where the impact of user characteristics on the explanation effects was measured. Our findings suggest that the results from the surveyed studies predominantly cover specific users who do not necessarily represent the users of recommender systems in the evaluation domain. This may seriously hamper the generalizability of any insights we may gain from current studies on explanations in recommender systems. We further find inconsistencies in the data reporting, which impacts the reproducibility of the reported results. Hence, we recommend actions to move toward a more inclusive and reproducible evaluation. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

将解释添加到推荐系统中被认为可以带来多种好处，例如增加用户的信任度或提高系统的透明度。来自其他应用领域的先前工作表明，特定的用户特征会影响用户对解释的认知。然而，我们很少在推荐系统解释的评估中看到这种类型的评价。本文通过调查了124篇在用户研究中评估推荐系统解释的论文，来填补这一空白。我们分析了这些论文中的参与者描述和研究结果，其中测量了用户特征对解释效果的影响。我们的研究发现表明，被调查的研究结果大多仅涵盖特定用户群体，这些用户不一定能够代表推荐系统评估领域中的用户群体。这可能严重妨碍我们从当前关于推荐系统解释的研究中获得的洞察力的普遍性。此外，我们还发现报告数据的一致性问题，影响了报告结果的可再现性。因此，我们建议采取措施，以实现更具包容性和可再现性的评估。 

---
# Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics with Large Language Models 

**Title (ZH)**: 基于大型语言模型的数据analytics中多步洞察合成的高级推理与变换引擎 

**Authors**: Atin Sakkeer Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2412.14146)  

**Abstract**: This paper presents the Advanced Reasoning and Transformation Engine for Multi-Step Insight Synthesis in Data Analytics (ARTEMIS-DA), a novel framework designed to augment Large Language Models (LLMs) for solving complex, multi-step data analytics tasks. ARTEMIS-DA integrates three core components: the Planner, which dissects complex user queries into structured, sequential instructions encompassing data preprocessing, transformation, predictive modeling, and visualization; the Coder, which dynamically generates and executes Python code to implement these instructions; and the Grapher, which interprets generated visualizations to derive actionable insights. By orchestrating the collaboration between these components, ARTEMIS-DA effectively manages sophisticated analytical workflows involving advanced reasoning, multi-step transformations, and synthesis across diverse data modalities. The framework achieves state-of-the-art (SOTA) performance on benchmarks such as WikiTableQuestions and TabFact, demonstrating its ability to tackle intricate analytical tasks with precision and adaptability. By combining the reasoning capabilities of LLMs with automated code generation and execution and visual analysis, ARTEMIS-DA offers a robust, scalable solution for multi-step insight synthesis, addressing a wide range of challenges in data analytics. 

**Abstract (ZH)**: 本文介绍了先进推理与转换引擎多步骤洞察综合在数据分析中的应用（ARTEMIS-DA），这是一种新型框架，旨在增强大型语言模型（LLMs），以解决复杂的多步骤数据分析任务。ARTEMIS-DA 集成了三个核心组件：规划器，它将复杂的用户查询分解为结构化、序列化的指令，涵盖数据预处理、转换、预测建模和可视化；编码器，它动态生成并执行Python代码以实现这些指令；以及图解器，它解释生成的可视化内容以推导出可行的洞察。通过协调这些组件之间的合作，ARTEMIS-DA 有效地管理了涉及高级推理、多步骤转换和跨多种数据模态综合的复杂分析工作流。该框架在WikiTableQuestions和TabFact等基准测试中达到了最先进的性能（SOTA），展示了它在精确性和适应性方面的能力，以应对复杂的分析任务。通过将LLMs的推理能力与自动化代码生成和执行以及可视化分析相结合，ARTEMIS-DA 提供了一种稳健、可扩展的解决方案，用于多步骤洞察综合，解决了数据分析中的广泛挑战。 

---
# A Cognitive Ideation Support Framework using IBM Watson Services 

**Title (ZH)**: 使用IBM Watson服务的认知构思支持框架 

**Authors**: Samaa Elnagar, Kweku-Muata Osei-Bryson  

**Link**: [PDF](https://arxiv.org/pdf/2412.14025)  

**Abstract**: Ideas generation is a core activity for innovation in organizations. The creativity of the generated ideas depends not only on the knowledge retrieved from the organizations' knowledge bases, but also on the external knowledge retrieved from other resources. Unfortunately, organizations often cannot efficiently utilize the knowledge in the knowledge bases due to the limited abilities of the search and retrieval mechanisms especially when dealing with unstructured data. In this paper, we present a new cognitive support framework for ideation that uses the IBM Watson DeepQA services. IBM Watson is a Question Answering system which mimics human cognitive abilities to retrieve and rank information. The proposed framework is based on the Search for Ideas in the Associative Memory (SIAM) model to help organizations develop creative ideas through discovering new relationships between retrieved data. To evaluate the effectiveness of the proposed system, the generated ideas generated are selected and assessed using a set of established creativity criteria. 

**Abstract (ZH)**: 创新是组织的核心活动之一。生成的想法的创造性不仅取决于从组织的知识库中检索的知识，还取决于从其他资源中检索的外部知识。不幸的是，由于搜索和检索机制的限制，尤其是在处理非结构化数据时，组织往往难以有效地利用知识库中的知识。本文提出了一种新的认知支持框架，用于创意生成，并利用IBM Watson DeepQA服务。IBM Watson是一种问答系统，模仿人类认知能力来检索和排名信息。所提出的框架基于联想记忆中的想法搜索（SIAM）模型，帮助组织通过发现检索数据之间的新关系来开发创意。为了评估所提出系统的有效性，生成的想法将根据一套已建立的创造性标准进行选择和评估。 

---
# CRM: Retrieval Model with Controllable Condition 

**Title (ZH)**: CRM：可控条件检索模型 

**Authors**: Chi Liu, Jiangxia Cao, Rui Huang, Kuo Cai, Weifeng Ding, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13844)  

**Abstract**: Recommendation systems (RecSys) are designed to connect users with relevant items from a vast pool of candidates while aligning with the business goals of the platform. A typical industrial RecSys is composed of two main stages, retrieval and ranking: (1) the retrieval stage aims at searching hundreds of item candidates satisfied user interests; (2) based on the retrieved items, the ranking stage aims at selecting the best dozen items by multiple targets estimation for each item candidate, including classification and regression targets. Compared with ranking model, the retrieval model absence of item candidate information during inference, therefore retrieval models are often trained by classification target only (e.g., click-through rate), but failed to incorporate regression target (e.g., the expected watch-time), which limit the effectiveness of retrieval. In this paper, we propose the Controllable Retrieval Model (CRM), which integrates regression information as conditional features into the two-tower retrieval paradigm. This modification enables the retrieval stage could fulfill the target gap with ranking model, enhancing the retrieval model ability to search item candidates satisfied the user interests and condition effectively. We validate the effectiveness of CRM through real-world A/B testing and demonstrate its successful deployment in Kuaishou short-video recommendation system, which serves over 400 million users. 

**Abstract (ZH)**: 推荐系统（RecSys）旨在将用户与广泛候选项目中相关项目连接起来，同时实现平台的业务目标。一个典型的工业推荐系统通常包括两个主要阶段：检索和排序：(1) 检索阶段旨在搜索满足用户兴趣的数百个候选项目；(2) 基于检索出的项目，排序阶段旨在通过每个候选项目的多目标估计来选择最佳的十几个项目，包括分类目标和回归目标。与排序模型相比，检索模型在推理过程中缺乏候选项目的相关信息，因此检索模型通常仅通过分类目标（例如点击率）进行训练，而无法纳入回归目标（例如预期观看时间），从而限制了检索模型的效果。本文提出了可控制检索模型（CRM），该模型将回归信息作为条件特征集成到两塔检索框架中。这种修改使检索阶段能够弥补与排序模型的目标差距，增强检索模型搜索满足用户兴趣和条件的候选项目的能效。我们通过实际的A/B测试验证了CRM的有效性，并展示了其在快手短视频推荐系统中的成功部署，该系统服务于超过4亿用户。 

---
# Maybe you are looking for CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval 

**Title (ZH)**: 也许您正在寻找跨模态查询建议技术：用于文本到图像检索（CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval） 

**Authors**: Giacomo Pacini, Fabio Carrara, Nicola Messina, Nicola Tonellotto, Giuseppe Amato, Fabrizio Falchi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13834)  

**Abstract**: Query suggestion, a technique widely adopted in information retrieval, enhances system interactivity and the browsing experience of document collections. In cross-modal retrieval, many works have focused on retrieving relevant items from natural language queries, while few have explored query suggestion solutions. In this work, we address query suggestion in cross-modal retrieval, introducing a novel task that focuses on suggesting minimal textual modifications needed to explore visually consistent subsets of the collection, following the premise of ''Maybe you are looking for''. To facilitate the evaluation and development of methods, we present a tailored benchmark named CroQS. This dataset comprises initial queries, grouped result sets, and human-defined suggested queries for each group. We establish dedicated metrics to rigorously evaluate the performance of various methods on this task, measuring representativeness, cluster specificity, and similarity of the suggested queries to the original ones. Baseline methods from related fields, such as image captioning and content summarization, are adapted for this task to provide reference performance scores. Although relatively far from human performance, our experiments reveal that both LLM-based and captioning-based methods achieve competitive results on CroQS, improving the recall on cluster specificity by more than 115% and representativeness mAP by more than 52% with respect to the initial query. The dataset, the implementation of the baseline methods and the notebooks containing our experiments are available here: this https URL 

**Abstract (ZH)**: 查询建议是一种在信息检索中广泛应用的技术，可以增强系统的互动性和文档集合的浏览体验。在跨模态检索中，许多研究侧重于从自然语言查询中检索相关项，但对于查询建议的研究却较少。本文旨在解决跨模态检索中的查询建议问题，提出了一项新的任务，该任务关注在“也许您正在寻找”的前提下，建议最小的文本修改，以探索集合中视觉上一致的子集。为了促进该任务的评估和方法的发展，我们提出了一套专门的基准测试，称为CroQS。该数据集包括初始查询、分组结果集以及每个组的人工定义的建议查询。我们制定了专用的评估指标，以严格评估不同方法在该任务上的表现，测量结果的代表性、簇的特异性以及建议查询与原始查询的相似度。来自相关领域的基线方法，如图像描述和内容总结，也被适应于该任务，提供了参考性能分数。尽管与人类性能相去甚远，但我们的实验显示，基于LLM的方法和基于图像描述的方法在CroQS上都能取得可竞争的结果，在簇的特异性召回率上提高了115%以上，在代表性mAP上提高了52%以上，相对于初始查询。该数据集、基线方法的实现以及包含我们实验的笔记本文件，均可在以下链接获得: [这里](this https URL) 

---
# Heterogeneous Graph Collaborative Filtering 

**Title (ZH)**: 异质图协作过滤 

**Authors**: Lianghao Xia, Meiyan Xie, Yong Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13825)  

**Abstract**: For modern recommender systems, the use of low-dimensional latent representations to embed users and items based on their observed interactions has become commonplace. However, many existing recommendation models are primarily designed for coarse-grained and homogeneous interactions, which limits their effectiveness in two critical dimensions. Firstly, these models fail to leverage the relational dependencies that exist across different types of user behaviors, such as page views, collects, comments, and purchases. Secondly, they struggle to capture the fine-grained latent factors that drive user interaction patterns. To address these limitations, we present a heterogeneous graph collaborative filtering model MixRec that excels at disentangling users' multi-behavior interaction patterns and uncovering the latent intent factors behind each behavior. Our model achieves this by incorporating intent disentanglement and multi-behavior modeling, facilitated by a parameterized heterogeneous hypergraph architecture. Furthermore, we introduce a novel contrastive learning paradigm that adaptively explores the advantages of self-supervised data augmentation, thereby enhancing the model's resilience against data sparsity and expressiveness with relation heterogeneity. To validate the efficacy of MixRec, we conducted extensive experiments on three public datasets. The results clearly demonstrate its superior performance, significantly outperforming various state-of-the-art baselines. Our model is open-sourced and available at: this https URL. 

**Abstract (ZH)**: 对于现代推荐系统而言，在基于用户观察到的交互使用低维潜在表示嵌入用户和项目逐渐变得常见。然而，许多现有的推荐模型主要针对粗粒度和同质的交互进行设计，这在两个关键维度上限制了它们的有效性。首先，这些模型未能充分利用不同类型用户行为（如浏览、收藏、评论和购买）之间的关系依赖性。其次，它们难以捕捉驱动用户交互模式的细粒度潜在因素。为了解决这些限制，我们提出了一种异质图协同过滤模型MixRec，该模型擅长分离用户的多行为交互模式，并揭示每种行为背后的潜在意图因素。我们的模型通过采用参数化的异质超图架构，结合意图分离和多行为建模实现了这一点。此外，我们引入了一种新颖的对比学习范式，该范式能够自适应地探索自我监督数据增强的优势，从而增强模型对数据稀疏性和关系异质性的鲁棒性和表达能力。为了验证MixRec的有效性，我们在三个公开数据集上进行了广泛实验。结果明显表明，MixRec在性能上超过了各种最先进的基线模型。我们的模型已开源，并可在以下链接获取：this https URL。 

---
# Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization 

**Title (ZH)**: 语义趋同：通过两级对齐和行为语义词元化 harmonize 推荐系统 

**Authors**: Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13771)  

**Abstract**: Large language models (LLMs), endowed with exceptional reasoning capabilities, are adept at discerning profound user interests from historical behaviors, thereby presenting a promising avenue for the advancement of recommendation systems. However, a notable discrepancy persists between the sparse collaborative semantics typically found in recommendation systems and the dense token representations within LLMs. In our study, we propose a novel framework that harmoniously merges traditional recommendation models with the prowess of LLMs. We initiate this integration by transforming ItemIDs into sequences that align semantically with the LLMs space, through the proposed Alignment Tokenization module. Additionally, we design a series of specialized supervised learning tasks aimed at aligning collaborative signals with the subtleties of natural language semantics. To ensure practical applicability, we optimize online inference by pre-caching the top-K results for each user, reducing latency and improving effciency. Extensive experimental evidence indicates that our model markedly improves recall metrics and displays remarkable scalability of recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其卓越的推理能力，能够从用户的历史行为中识别出深层次的兴趣偏好，为推荐系统的进步提供了广阔的可能性。然而，推荐系统中常见的稀疏协作语义与LLMs中的密集词嵌入之间存在显著的差距。在本研究中，我们提出了一种新颖的框架，将传统的推荐模型与LLMs的优势融合在一起。我们通过提议的对齐分词模块，将ItemIDs转换为与LLMs空间语义对齐的序列，以启动这一整合过程。此外，我们设计了一系列专门的监督学习任务，旨在将协作信号与自然语言语义的细微差别对齐。为了确保其实用性，我们通过为每个用户预先缓存前K个结果来优化在线推理，从而降低延迟并提高效率。广泛的实验结果表明，我们的模型在召回率指标上有了显著的提升，并展示了推荐系统出色的可扩展性。 

---
# Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models 

**Title (ZH)**: 使用大规模语言模型弥补知识感知推荐中的用户侧知识缺口 

**Authors**: Zheng Hu, Zhe Li, Ziyun Jiao, Satoshi Nakagawa, Jiawen Deng, Shimin Cai, Tao Zhou, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.13544)  

**Abstract**: In recent years, knowledge graphs have been integrated into recommender systems as item-side auxiliary information, enhancing recommendation accuracy. However, constructing and integrating structural user-side knowledge remains a significant challenge due to the improper granularity and inherent scarcity of user-side features. Recent advancements in Large Language Models (LLMs) offer the potential to bridge this gap by leveraging their human behavior understanding and extensive real-world knowledge. Nevertheless, integrating LLM-generated information into recommender systems presents challenges, including the risk of noisy information and the need for additional knowledge transfer. In this paper, we propose an LLM-based user-side knowledge inference method alongside a carefully designed recommendation framework to address these challenges. Our approach employs LLMs to infer user interests based on historical behaviors, integrating this user-side information with item-side and collaborative data to construct a hybrid structure: the Collaborative Interest Knowledge Graph (CIKG). Furthermore, we propose a CIKG-based recommendation framework that includes a user interest reconstruction module and a cross-domain contrastive learning module to mitigate potential noise and facilitate knowledge transfer. We conduct extensive experiments on three real-world datasets to validate the effectiveness of our method. Our approach achieves state-of-the-art performance compared to competitive baselines, particularly for users with sparse interactions. 

**Abstract (ZH)**: 近年来，知识图谱已被整合到推荐系统中，作为物品方面的辅助信息，以提升推荐的准确性。然而，构建和整合用户方面的结构性知识仍然是一项重大挑战，主要是由于用户特征的不适当粒度和固有的稀疏性。大型语言模型（LLMs）的最新进展提供了一种可能的解决方案，通过利用其对人类行为的理解和广泛的实际知识来弥合这一差距。然而，将LLM生成的信息整合到推荐系统中也面临挑战，包括噪声信息的风险和额外知识转移的需求。在本文中，我们提出了一种基于LLM的用户方面知识推断方法以及一个精心设计的推荐框架，以应对这些挑战。我们的方法利用LLM根据历史行为推断用户兴趣，将这种用户方面的信息与物品方面的信息和协作数据相结合，构造出一种混合结构：协作兴趣知识图谱（CIKG）。此外，我们提出了一种基于CIKG的推荐框架，其中包括用户兴趣重构模块和跨域对比学习模块，以减轻潜在的噪声并促进知识转移。我们通过对三个真实世界数据集进行广泛实验，验证了我们方法的有效性。我们的方法在与竞争基线模型相比时达到了最先进的性能，特别是在交互稀疏的用户方面表现出色。 

---
# Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future 

**Title (ZH)**: 大型语言模型增强的推荐系统：分类、趋势、应用和未来 

**Authors**: Qidong Liu, Xiangyu Zhao, Yuhao Wang, Yejing Wang, Zijian Zhang, Yuqi Sun, Xiang Li, Maolin Wang, Pengyue Jia, Chong Chen, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.13432)  

**Abstract**: Large Language Model (LLM) has transformative potential in various domains, including recommender systems (RS). There have been a handful of research that focuses on empowering the RS by LLM. However, previous efforts mainly focus on LLM as RS, which may face the challenge of intolerant inference costs by LLM. Recently, the integration of LLM into RS, known as LLM-Enhanced Recommender Systems (LLMERS), has garnered significant interest due to its potential to address latency and memory constraints in real-world applications. This paper presents a comprehensive survey of the latest research efforts aimed at leveraging LLM to enhance RS capabilities. We identify a critical shift in the field with the move towards incorporating LLM into the online system, notably by avoiding their use during inference. Our survey categorizes the existing LLMERS approaches into three primary types based on the component of the RS model being augmented: Knowledge Enhancement, Interaction Enhancement, and Model Enhancement. We provide an in-depth analysis of each category, discussing the methodologies, challenges, and contributions of recent studies. Furthermore, we highlight several promising research directions that could further advance the field of LLMERS. 

**Abstract (ZH)**: 大型语言模型（LLM）在多个领域具有变革性的潜力，包括推荐系统（RS）。已有少量研究关注通过LLM增强RS的能力。然而，之前的大部分努力主要集中在将LLM作为RS的一部分，这可能会面临LLM推断成本难以承受的挑战。最近，将LLM集成到RS中，称为LLM增强推荐系统（LLMERS），因其能够解决实际应用中的延迟和内存约束问题而引起了显著的兴趣。本文对最新的研究努力进行了全面综述，旨在利用LLM增强RS的能力。我们识别出领域内的一个关键转变，即转向将LLM集成到在线系统中，尤其是在推断过程中避免使用LLM。我们根据增强RS模型的不同组件将现有的LLMERS方法分为三大类：知识增强、交互增强和模型增强。我们对每种类别进行了深入分析，讨论了近期研究的方法、挑战和贡献。此外，我们还强调了几条有前景的研究方向，这些方向有望进一步推动LLMERS领域的发展。 

---
