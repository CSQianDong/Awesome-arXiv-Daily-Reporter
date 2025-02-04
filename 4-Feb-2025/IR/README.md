# Query Brand Entity Linking in E-Commerce Search 

**Title (ZH)**: 电子商务搜索中的查询品牌实体链接 

**Authors**: Dong Liu, Sreyashi Nag  

**Link**: [PDF](https://arxiv.org/pdf/2502.01555)  

**Abstract**: In this work, we address the brand entity linking problem for e-commerce search queries. The entity linking task is done by either i)a two-stage process consisting of entity mention detection followed by entity disambiguation or ii) an end-to-end linking approaches that directly fetch the target entity given the input text. The task presents unique challenges: queries are extremely short (averaging 2.4 words), lack natural language structure, and must handle a massive space of unique brands. We present a two-stage approach combining named-entity recognition with matching, and a novel end-to-end solution using extreme multi-class classification. We validate our solutions by both offline benchmarks and the impact of online A/B test. 

**Abstract (ZH)**: 在本文中，我们针对电子商务搜索查询中的品牌实体链接问题进行了研究。实体链接任务可以通过以下两种方式之一完成：i) 一个两阶段过程，包括实体提学术认检测后跟实体消歧；或者ii) 直接从输入文本中获取目标实体的端到端链接方法。该任务面临着独特的挑战：查询极为简短（平均仅2.4个单词），缺乏自然语言结构，并且必须处理大量独特的品牌数量。我们提出了一种结合命名实体识别与匹配的两阶段方法，并提出了一种新颖的端到端解决方案，利用极端多分类方法。我们通过离线基准测试和在线A/B测试的直接影响来验证我们的解决方案。 

---
# VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos 

**Title (ZH)**: VideoRAG：极长上下文视频增强生成 

**Authors**: Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01549)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated remarkable success in enhancing Large Language Models (LLMs) through external knowledge integration, yet its application has primarily focused on textual content, leaving the rich domain of multi-modal video knowledge predominantly unexplored. This paper introduces VideoRAG, the first retrieval-augmented generation framework specifically designed for processing and understanding extremely long-context videos. Our core innovation lies in its dual-channel architecture that seamlessly integrates (i) graph-based textual knowledge grounding for capturing cross-video semantic relationships, and (ii) multi-modal context encoding for efficiently preserving visual features. This novel design empowers VideoRAG to process unlimited-length videos by constructing precise knowledge graphs that span multiple videos while maintaining semantic dependencies through specialized multi-modal retrieval paradigms. Through comprehensive empirical evaluation on our proposed LongerVideos benchmark-comprising over 160 videos totaling 134+ hours across lecture, documentary, and entertainment categories-VideoRAG demonstrates substantial performance compared to existing RAG alternatives and long video understanding methods. The source code of VideoRAG implementation and the benchmark dataset are openly available at: this https URL. 

**Abstract (ZH)**: 以下是符合学术规范的翻译：

检索增强生成（RAG）在通过外部知识整合增强大型语言模型（LLMs）方面已经取得了显著的成功，其应用领域主要集中在文本内容上，而多模态视频知识的丰富领域仍然未得到充分探索。本文提出了一种名为VideoRAG的新框架，这是第一个专门针对处理和理解极长视频的检索增强生成框架。我们的核心创新在于该框架采用了一种双通道架构，该架构能够无缝地将（i）基于图的文本知识关联应用于捕捉视频间的跨层语义关系，以及（ii）多模态上下文编码应用于高效地保留视觉特征相结合。这种新颖的设计使VideoRAG能够通过构建跨视频的精确知识图谱来处理无限长度的视频，同时通过专门的多模态检索范式保持语义依赖性。

通过在我们提出的LongerVideos基准上进行全面的经验性评估（该基准包含超过160个视频，总计134+小时，涵盖了讲座、纪录片和娱乐等多个类别），VideoRAG与现有的RAG替代方法和长视频理解方法相比，展现了显著的性能优势。VideoRAG的实现代码和基准数据集均可从以下链接公开获取：this https URL。 

---
# Augmented Knowledge Graph Querying leveraging LLMs 

**Title (ZH)**: 利用大语言模型增强知识图谱查询 

**Authors**: Marco Arazzi, Davide Ligari, Serena Nicolazzo, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2502.01298)  

**Abstract**: Adopting Knowledge Graphs (KGs) as a structured, semantic-oriented, data representation model has significantly improved data integration, reasoning, and querying capabilities across different domains. This is especially true in modern scenarios such as Industry 5.0, in which the integration of data produced by humans, smart devices, and production processes plays a crucial role. However, the management, retrieval, and visualization of data from a KG using formal query languages can be difficult for non-expert users due to their technical complexity, thus limiting their usage inside industrial environments. For this reason, we introduce SparqLLM, a framework that utilizes a Retrieval-Augmented Generation (RAG) solution, to enhance the querying of Knowledge Graphs (KGs). SparqLLM executes the Extract, Transform, and Load (ETL) pipeline to construct KGs from raw data. It also features a natural language interface powered by Large Language Models (LLMs) to enable automatic SPARQL query generation. By integrating template-based methods as retrieved-context for the LLM, SparqLLM enhances query reliability and reduces semantic errors, ensuring more accurate and efficient KG interactions. Moreover, to improve usability, the system incorporates a dynamic visualization dashboard that adapts to the structure of the retrieved data, presenting the query results in an intuitive format. Rigorous experimental evaluations demonstrate that SparqLLM achieves high query accuracy, improved robustness, and user-friendly interaction with KGs, establishing it as a scalable solution to access semantic data. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

采用知识图谱（KGs）作为结构化且语义导向的数据表示模型，极大地提升了不同领域间的数据集成、推理和查询能力。特别是在第五代工业（Industry 5.0）等现代场景中，人类、智能设备和生产过程所产生的数据的整合起着至关重要的作用。然而，由于使用形式化的查询语言管理、检索和可视化知识图谱中的数据对于非专家用户来说可能会非常复杂，从而限制了其在工业环境中的应用。因此，我们提出了SparqLLM框架，该框架利用检索增强生成（RAG）解决方案来增强对知识图谱的查询能力。SparqLLM执行抽取、转换和加载（ETL）管道，从原始数据中构建知识图谱。此外，该框架还配备了一个由大型语言模型（LLMs）驱动的自然语言界面，能够实现自动SPARQL查询生成。通过将基于模板的方法作为检索上下文整合到LLM中，SparqLLM提高了查询可靠性，减少了语义错误，确保了更准确和高效的知识图谱交互。为了提高易用性，系统还集成了一个动态可视化仪表板，根据检索数据的结构进行调整，以直观的格式展示查询结果。严格的实证研究表明，SparqLLM实现了高度准确的查询、增强的鲁棒性和用户友好的知识图谱交互，使其成为访问语义数据的可扩展解决方案。 

---
# GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation 

**Title (ZH)**: GFM-RAG：图基础模型增强生成中的检索技术 

**Authors**: Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.01113)  

**Abstract**: Retrieval-augmented generation (RAG) has proven effective in integrating knowledge into large language models (LLMs). However, conventional RAGs struggle to capture complex relationships between pieces of knowledge, limiting their performance in intricate reasoning that requires integrating knowledge from multiple sources. Recently, graph-enhanced retrieval augmented generation (GraphRAG) builds graph structure to explicitly model these relationships, enabling more effective and efficient retrievers. Nevertheless, its performance is still hindered by the noise and incompleteness within the graph structure. To address this, we introduce GFM-RAG, a novel graph foundation model (GFM) for retrieval augmented generation. GFM-RAG is powered by an innovative graph neural network that reasons over graph structure to capture complex query-knowledge relationships. The GFM with 8M parameters undergoes a two-stage training process on large-scale datasets, comprising 60 knowledge graphs with over 14M triples and 700k documents. This results in impressive performance and generalizability for GFM-RAG, making it the first graph foundation model applicable to unseen datasets for retrieval without any fine-tuning required. Extensive experiments on three multi-hop QA datasets and seven domain-specific RAG datasets demonstrate that GFM-RAG achieves state-of-the-art performance while maintaining efficiency and alignment with neural scaling laws, highlighting its potential for further improvement. 

**Abstract (ZH)**: 检索增强生成（RAG）已被证明在将知识整合到大型语言模型（LLMs）中是有效的。然而，传统的RAG在捕获知识片段之间的复杂关系方面存在困难，限制了它们在需要从多个来源整合知识的复杂推理中的表现。最近，图增强检索增强生成（GraphRAG）通过构建图结构来明确建模这些关系，从而使得检索器更加有效和高效。然而，其性能仍然受到图结构内部噪声和不完整性的限制。为解决这个问题，我们引入了GFM-RAG，这是一种新型的图基础模型（GFM），用于检索增强生成。GFM-RAG依托一种创新的图神经网络，该网络可以在图结构上进行推理以捕捉复杂查询-知识关系。具有800万参数的GFM在大规模数据集上进行了两阶段训练，这些数据集包括60个知识图谱，包含超过1400万三元组和70万文档。这使得GFM-RAG在性能和泛化能力方面表现出色，并且它是首个在无需微调即可应用于未见过的数据集的图基础模型，用于检索。在三个多跳问答数据集和七个领域特定的RAG数据集上的广泛实验表明，GFM-RAG在保持高效的同时取得了领先的表现，并且与神经扩展定律保持一致，突显了其进一步改进的潜力。 

---
# RankFlow: A Multi-Role Collaborative Reranking Workflow Utilizing Large Language Models 

**Title (ZH)**: RankFlow：一种利用大规模语言模型的多角色协作重排工作流 

**Authors**: Can Jin, Hongwu Peng, Anxiang Zhang, Nuo Chen, Jiahui Zhao, Xi Xie, Kuangzheng Li, Shuya Feng, Kai Zhong, Caiwen Ding, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2502.00709)  

**Abstract**: In an Information Retrieval (IR) system, reranking plays a critical role by sorting candidate passages according to their relevance to a specific query. This process demands a nuanced understanding of the variations among passages linked to the query. In this work, we introduce RankFlow, a multi-role reranking workflow that leverages the capabilities of Large Language Models (LLMs) and role specializations to improve reranking performance. RankFlow enlists LLMs to fulfill four distinct roles: the query Rewriter, the pseudo Answerer, the passage Summarizer, and the Reranker. This orchestrated approach enables RankFlow to: (1) accurately interpret queries, (2) draw upon LLMs' extensive pre-existing knowledge, (3) distill passages into concise versions, and (4) assess passages in a comprehensive manner, resulting in notably better reranking results. Our experimental results reveal that RankFlow outperforms existing leading approaches on widely recognized IR benchmarks, such as TREC-DL, BEIR, and NovelEval. Additionally, we investigate the individual contributions of each role in RankFlow. Code is available at this https URL. 

**Abstract (ZH)**: 在信息检索（IR）系统中，重排序通过根据候选段落与特定查询的相关性对其进行排序，在这一过程中扮演着关键角色。这一过程需要对查询相关段落之间细微差异的深刻理解。在这项工作中，我们引入了RankFlow，这是一种利用大型语言模型（LLMs）和角色专业化能力的多角色重排序工作流，旨在提高重排序性能。RankFlow 固定了 LLMs 扮演四个不同的角色：查询重写者、伪答案者、段落摘要者和重排序器。这种协调的方法使得RankFlow能够：（1）准确解释查询，（2）利用LLMs广泛的现有知识，（3）将段落提炼为简洁版本，以及（4）全面评估段落，从而显著提高重排序结果。我们的实验结果表明，RankFlow 在广泛认可的IR基准测试（如TREC-DL、BEIR和NovelEval）中优于现有的领先方法。此外，我们还研究了 RankFlow 中每个角色的独立贡献。源代码可在以下链接获取：[此处替换为实际链接]。 

---
# Retracted Citations and Self-citations in Retracted Publications: A Comparative Study of Plagiarism and Fake Peer Review 

**Title (ZH)**: 被撤回出版物中的撤回引用和自引： regarding plagiarism and fake peer review 的比较研究 

**Authors**: Kiran Sharmaa, Parul Khurana  

**Link**: [PDF](https://arxiv.org/pdf/2502.00673)  

**Abstract**: Retracted citations remain a significant concern in academia as they perpetuate misinformation and compromise the integrity of scientific literature despite their invalidation. To analyze the impact of retracted citations, we focused on two retraction categories: plagiarism and fake peer review. The data set was sourced from Scopus and the reasons for the retraction were mapped using the Retraction Watch database. The retraction trend shows a steady average growth in plagiarism cases of 1.2 times, while the fake peer review exhibits a fluctuating pattern with an average growth of 5.5 times. Although fewer papers are retracted in the plagiarism category compared to fake peer reviews, plagiarism-related papers receive 2.5 times more citations. Furthermore, the total number of retracted citations for plagiarized papers is 1.8 times higher than that for fake peer review papers. Within the plagiarism category, 46% of the retracted citations are due to plagiarism, while 53.6% of the retracted citations in the fake peer review category are attributed to the fake peer review. The results also suggest that fake peer review cases are identified and retracted more rapidly than plagiarism cases. Finally, self-citations constitute a small percentage of citations to retracted papers but are notably higher among citations that are later retracted in both the categories. 

**Abstract (ZH)**: 撤回引用在学术界仍然是一个重大问题，尽管它们被无效化，但仍会传播错误信息并损害科学文献的完整性。为了分析撤回引用的影响，我们专注于两类撤回：抄袭和虚假同行评议。数据来源于Scopus，撤回原因利用Retraction Watch数据库进行映射。撤回趋势显示，抄袭案件的平均增长率为1.2倍，而虚假同行评议则表现出波动模式，平均增长率为5.5倍。尽管抄袭案件中被撤回的论文数量少于虚假同行评议，但与抄袭相关的论文获得了2.5倍更多的引用次数。此外，抄袭论文的总撤回引用次数是虚假同行评议论文的1.8倍。在抄袭类别中，46%的撤回引用是由于抄袭造成的，而在虚假同行评议类别中，53.6%的撤回引用归因于虚假同行评审。研究结果还表明，与抄袭案件相比，虚假同行评议案件被更快地识别和撤回。最后，自我引用在撤回论文的引用中占很小的百分比，但在两类中后续被撤回的引用中明显更高。 

---
# Personalized Denoising Implicit Feedback for Robust Recommender System 

**Title (ZH)**: 个性化去噪隐式反馈以构建健壮的推荐系统 

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00348)  

**Abstract**: While implicit feedback is foundational to modern recommender systems, factors such as human error, uncertainty, and ambiguity in user behavior inevitably introduce significant noise into this feedback, adversely affecting the accuracy and robustness of recommendations. To address this issue, existing methods typically aim to reduce the training weight of noisy feedback or discard it entirely, based on the observation that noisy interactions often exhibit higher losses in the overall loss distribution. However, we identify two key issues: (1) there is a significant overlap between normal and noisy interactions in the overall loss distribution, and (2) this overlap becomes even more pronounced when transitioning from pointwise loss functions (e.g., BCE loss) to pairwise loss functions (e.g., BPR loss). This overlap leads traditional methods to misclassify noisy interactions as normal, and vice versa. To tackle these challenges, we further investigate the loss overlap and find that for a given user, there is a clear distinction between normal and noisy interactions in the user's personal loss distribution. Based on this insight, we propose a resampling strategy to Denoise using the user's Personal Loss distribution, named PLD, which reduces the probability of noisy interactions being optimized. Specifically, during each optimization iteration, we create a candidate item pool for each user and resample the items from this pool based on the user's personal loss distribution, prioritizing normal interactions. Additionally, we conduct a theoretical analysis to validate PLD's effectiveness and suggest ways to further enhance its performance. Extensive experiments conducted on three datasets with varying noise ratios demonstrate PLD's efficacy and robustness. 

**Abstract (ZH)**: 尽管隐式反馈是现代推荐系统的基础，但用户行为中的人为错误、不确定性和模糊性等因素不可避免地会在反馈中引入大量噪声，从而影响推荐的准确性和鲁棒性。为解决这一问题，现有方法通常旨在减少噪声反馈的训练权重或完全丢弃这些反馈，基于观察到噪声交互在整体损失分布中通常具有更高的损失值。然而，我们发现了两个关键问题：(1) 在整体损失分布中，正常交互和噪声交互之间存在显著重叠；(2) 在从点wise损失函数（例如：BCE损失）过渡到pairwise损失函数（例如：BPR损失）时，这种重叠变得更为明显。这种重叠导致传统方法错误地将噪声交互分类为正常交互，反之亦然。为应对这些挑战，我们进一步研究了损失重叠情况，并发现对于每个用户，在用户个人损失分布中，正常交互和噪声交互之间存在明显区别。基于这一洞察，我们提出了一种基于用户个人损失分布去噪的重采样策略，称为PLD（Personal Loss Distribution），该策略降低了噪声交互被优化的概率。具体而言，在每次优化迭代中，我们为每个用户创建一个候选项目池，并根据用户个人损失分布重采样该项目池中的项目，优先选择正常交互。此外，我们还进行了理论分析以验证PLD的有效性，并提出了进一步提高其性能的方法。在三个不同噪声比例的数据集上进行的广泛实验表明，PLD在去噪方面具有高效性和鲁棒性。 

---
# MIM: Multi-modal Content Interest Modeling Paradigm for User Behavior Modeling 

**Title (ZH)**: MIM：多模态内容兴趣建模范式用于用户行为建模 

**Authors**: Bencheng Yan, Si Chen, Shichang Jia, Jianyu Liu, Yueran Liu, Chenghan Fu, Wanxian Guan, Hui Zhao, Xiang Zhang, Kai Zhang, Wenbo Su, Pengjie Wang, Jian Xu, Bo Zheng, Baolin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00321)  

**Abstract**: Click-Through Rate (CTR) prediction is a crucial task in recommendation systems, online searches, and advertising platforms, where accurately capturing users' real interests in content is essential for performance. However, existing methods heavily rely on ID embeddings, which fail to reflect users' true preferences for content such as images and titles. This limitation becomes particularly evident in cold-start and long-tail scenarios, where traditional approaches struggle to deliver effective results. To address these challenges, we propose a novel Multi-modal Content Interest Modeling paradigm (MIM), which consists of three key stages: Pre-training, Content-Interest-Aware Supervised Fine-Tuning (C-SFT), and Content-Interest-Aware UBM (CiUBM). The pre-training stage adapts foundational models to domain-specific data, enabling the extraction of high-quality multi-modal embeddings. The C-SFT stage bridges the semantic gap between content and user interests by leveraging user behavior signals to guide the alignment of embeddings with user preferences. Finally, the CiUBM stage integrates multi-modal embeddings and ID-based collaborative filtering signals into a unified framework. Comprehensive offline experiments and online A/B tests conducted on the Taobao, one of the world's largest e-commerce platforms, demonstrated the effectiveness and efficiency of MIM method. The method has been successfully deployed online, achieving a significant increase of +14.14% in CTR and +4.12% in RPM, showcasing its industrial applicability and substantial impact on platform performance. To promote further research, we have publicly released the code and dataset at this https URL. 

**Abstract (ZH)**: 点击率（CTR）预测是推荐系统、在线搜索和广告平台中的关键任务，准确捕捉用户对内容的真实兴趣对于提高性能至关重要。然而，现有的方法过度依赖于ID嵌入，这无法反映用户对诸如图像和标题等内容的真实偏好。这种限制在冷启动和长尾场景中尤为明显，传统方法在这种情况下难以提供有效的结果。为了解决这些挑战，我们提出了一种新型的多模态内容兴趣建模框架（MIM），它包含三个关键阶段：预训练、内容-兴趣感知监督微调（C-SFT）以及内容-兴趣感知统一模型（CiUBM）。预训练阶段将基础模型适应特定领域的数据，使其能够提取高质量的多模态嵌入。C-SFT阶段通过利用用户行为信号来弥合内容与用户兴趣之间的语义差距，从而引导嵌入与用户偏好的对齐。最后，CiUBM阶段将多模态嵌入和基于ID的合作过滤信号整合到一个统一框架中。在淘宝，全球最大的电商平台之一，进行的全面离线实验和在线A/B测试表明了MIM方法的有效性和效率。该方法已成功部署在线，实现了点击率CTR提升14.14%，收益每千次展示RPM提升4.12%，展示了其工业应用前景及其在平台性能上的显著影响。为了促进进一步研究，我们已在以下网址公开发布了代码和数据集：[此链接](this https URL)。 

---
# Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search 

**Title (ZH)**: 广告中介偏见：优化关键词推荐相关性以匹配搜索需求 

**Authors**: Soumik Dey, Wei Zhang, Hansi Wu, Bingfeng Dong, Binbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.00131)  

**Abstract**: E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). Keyphrases must be pertinent to items; otherwise, it can result in seller dissatisfaction and poor targeting -- towards that end relevance filters are employed. In this work, we describe the shortcomings of training relevance filter models on biased click/sales signals. We re-conceptualize advertiser keyphrase relevance as interaction between two dynamical systems -- Advertising which produces the keyphrases and Search which acts as a middleman to reach buyers. We discuss the bias of search relevance systems (middleman bias) and the need to align advertiser keyphrases with search relevance signals. We also compare the performance of cross encoders and bi-encoders in modeling this alignment and the scalability of such a solution for sellers at eBay. 

**Abstract (ZH)**: 电子商务卖家根据其库存推荐相关关键词，用于广告宣传以提升买家参与度（点击量/销售额）。关键词必须与商品相关，否则可能导致卖家不满和目标不精准——为此，我们使用相关性过滤器。然而，在本文中，我们探讨了基于有偏见的点击/销售信号训练相关性过滤器模型的局限性。我们将广告商的关键词相关性重新概念化为两个动态系统之间的交互——广告系统生成关键词，搜索引擎作为中间人连接到买家。我们讨论了搜索相关性系统的偏见（中间人偏见），并强调了将广告商的关键词与搜索相关性信号对齐的必要性。我们还比较了交叉编码器和双编码器在这方面的性能，并讨论了这种解决方案在eBay等销售商中的可扩展性。 

---
# Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models 

**Title (ZH)**: Topic-FlipRAG：面向主题的对抗性意见操纵攻击以检索增强生成模型为目标 

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01386)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research. 

**Abstract (ZH)**: 基于大规模语言模型（LLMs）的检索增强生成（RAG）系统已成为问答和内容生成等任务的重要工具。然而，它们对公众意见和社会信息传播的影响日益增大，使得安全研究对其潜在漏洞的关注愈加迫切。先前的研究主要针对事实或单一查询的操控进行了攻击。在本文中，我们探讨了一个更为实际的场景：面向主题的敌对意见操纵攻击，针对RAG模型，其中LLMs需综合多种视角进行推理，使其特别容易受到系统性知识投毒的影响。具体而言，我们提出了一种名为Topic-FlipRAG的两阶段操纵攻击框架，该框架战略性地构建敌对扰动以影响相关查询下的意见。此方法结合了传统的对抗排序攻击技术，并利用LLMs广泛且内部的相关知识和推理能力来执行语义级扰动。实验结果显示，所提出的攻击能够有效改变模型特定主题输出的意见，显著影响用户信息感知。当前的防御方法无法有效抵御此类攻击，突显了提升RAG系统安全防护的必要性，并为大规模语言模型（LLMs）安全研究提供了关键见解。 

---
# PSSD: Making Large Language Models Self-denial via Human Psyche Structure 

**Title (ZH)**: PSSD：通过人类心理结构使大型语言模型自我质疑 

**Authors**: Jinzhi Liao, Zenghua Liao, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.01344)  

**Abstract**: The enhance of accuracy in reasoning results of LLMs arouses the community's interests, wherein pioneering studies investigate post-hoc strategies to rectify potential mistakes. Despite extensive efforts, they are all stuck in a state of resource competition demanding significant time and computing expenses. The cause of the situation lies in the failure of identifying the fundamental feature of the solutions in this line, coined as the self-denial of LLMs. In other words, LLMs should confidently determine the potential existence of mistakes and carefully execute the targeted correction. As the whole procedure conducts within LLMs, supporting and persuasive references are hard to acquire, while the absence of specific steps towards refining hidden mistakes persists even when errors are acknowledged. In response to the challenges, we present PSSD, which refers to and implements the human psyche structure such that three distinct and interconnected roles contribute to human reasoning. Specifically, PSSD leverages the recent multi-agent paradigm, and is further enhanced with three innovatively conceived roles: (1) the intuition-based id role that provides initial attempts based on benign LLMs; (2) the rule-driven superego role that summarizes rules to regulate the above attempts, and returns specific key points as guidance; and (3) the script-centric ego role that absorbs all procedural information to generate executable script for the final answer prediction. Extensive experiments demonstrate that the proposed design not only better enhance reasoning capabilities, but also seamlessly integrate with current models, leading to superior performance. 

**Abstract (ZH)**: 增强大型语言模型推理结果的准确性引起了学术界的兴趣，其中先驱研究探讨了后验策略以纠正潜在错误。尽管付出了大量努力，这些研究仍然深陷资源竞争的困境，耗费大量时间和计算资源。这种情况的原因在于未能识别这一系列解决方案的基本特征，这种特征被称作大型语言模型的自我否定。换句话说，大型语言模型应该自信地确定潜在错误的存在，并仔细执行针对性的修正。由于整个过程都在大型语言模型内部进行，支持性和说服性的参考文献难以获得，即使承认错误存在，具体步骤用于改进隐藏的错误仍然缺失。为应对这些挑战，我们提出了一种名为PSSD的设计，该设计参考并实现了人类心理结构，其中三个相互独立且相互关联的角色共同促进人类推理。具体而言，PSSD利用了最近的多代理范式，并进一步增加了三种创新构想的角色：（1）基于直觉的自我角色，基于良性大型语言模型提供初步尝试；（2）规则驱动的超我角色，总结规则以调节上述尝试，并返回具体关键点作为指导；（3）剧本为中心的本我角色，吸收所有程序信息以生成最终答案预测所需的可执行脚本。大量实验表明，所提出的设计不仅能更好地增强推理能力，还能无缝集成当前模型，从而实现卓越的性能。 

---
# DeepRAG: Thinking to Retrieval Step by Step for Large Language Models 

**Title (ZH)**: DeepRAG:逐步思考以进行大规模语言模型的检索 

**Authors**: Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.01142)  

**Abstract**: Large Language Models (LLMs) have shown remarkable potential in reasoning while they still suffer from severe factual hallucinations due to timeliness, accuracy, and coverage of parametric knowledge. Meanwhile, integrating reasoning with retrieval-augmented generation (RAG) remains challenging due to ineffective task decomposition and redundant retrieval, which can introduce noise and degrade response quality. In this paper, we propose DeepRAG, a framework that models retrieval-augmented reasoning as a Markov Decision Process (MDP), enabling strategic and adaptive retrieval. By iteratively decomposing queries, DeepRAG dynamically determines whether to retrieve external knowledge or rely on parametric reasoning at each step. Experiments show that DeepRAG improves retrieval efficiency while improving answer accuracy by 21.99%, demonstrating its effectiveness in optimizing retrieval-augmented reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理方面展现了显著的潜力，但由于参数知识在时效性、准确性和覆盖率方面的局限性，它们仍然遭受严重的事实性幻觉。同时，将推理与检索增强生成（RAG）集成仍然具有挑战性，这主要是由于任务分解无效和检索冗余，这些因素可能会引入噪声并降低响应质量。本文提出了一种名为DeepRAG的框架，该框架将检索增强推理建模为马尔可夫决策过程（MDP），从而实现策略性和自适应的检索。通过迭代分解查询，DeepRAG在每一步动态决定是检索外部知识还是依赖参数推理。实验结果表明，DeepRAG在提高检索效率的同时，准确度提升了21.99%，证明了其在优化检索增强推理方面的有效性。 

---
# HintEval: A Comprehensive Framework for Hint Generation and Evaluation for Questions 

**Title (ZH)**: 指示生成与评估框架：HintEval 

**Authors**: Jamshid Mozafari, Bhawna Piryani, Abdelrahman Abdallah, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2502.00857)  

**Abstract**: Large Language Models (LLMs) are transforming how people find information, and many users turn nowadays to chatbots to obtain answers to their questions. Despite the instant access to abundant information that LLMs offer, it is still important to promote critical thinking and problem-solving skills. Automatic hint generation is a new task that aims to support humans in answering questions by themselves by creating hints that guide users toward answers without directly revealing them. In this context, hint evaluation focuses on measuring the quality of hints, helping to improve the hint generation approaches. However, resources for hint research are currently spanning different formats and datasets, while the evaluation tools are missing or incompatible, making it hard for researchers to compare and test their models. To overcome these challenges, we introduce HintEval, a Python library that makes it easy to access diverse datasets and provides multiple approaches to generate and evaluate hints. HintEval aggregates the scattered resources into a single toolkit that supports a range of research goals and enables a clear, multi-faceted, and reliable evaluation. The proposed library also includes detailed online documentation, helping users quickly explore its features and get started. By reducing barriers to entry and encouraging consistent evaluation practices, HintEval offers a major step forward for facilitating hint generation and analysis research within the NLP/IR community. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在改变人们获取信息的方式，如今许多用户转向聊天机器人以获取问题的答案。尽管LLMs提供了即时访问大量信息的能力，但促进批判性思维和问题解决能力仍然至关重要。自动提示生成是一项新兴任务，旨在通过创建导向性提示来支持用户独立回答问题，这些提示引导用户找到答案而不直接透露答案。在此背景下，提示评价关注于衡量提示的质量，帮助改进提示生成方法。然而，当前用于提示研究的资源分布在不同的格式和数据集中，而缺乏一致性评价工具或不兼容，使得研究人员难以比较和测试模型。为克服这些挑战，我们引入了HintEval，这是一个Python库，使用户能够轻松访问多样化的数据集，并提供了生成和评估提示的多种方法。HintEval将分散的资源整合为一个工具包，支持多种研究目标，并能提供清晰、多维度和可靠的评估。所提议的库还包含详细的在线文档，帮助用户快速探索其功能并开始使用。通过降低入门门槛并促进一致性的评估实践，HintEval为自然语言处理/信息检索（NLP/IR）社区内的提示生成和分析研究提供了重要进步。 

---
# On Overlap Ratio in Defocused Electron Ptychography 

**Title (ZH)**: 被告焦电子 Ptychography 中的重叠比研究 

**Authors**: Amirafshar Moshtaghpour, Angus I. Kirkland  

**Link**: [PDF](https://arxiv.org/pdf/2502.00762)  

**Abstract**: Four-dimensional Scanning Transmission Electron Microscopy (4D STEM) with data acquired using a defocused electron probe is a promising tool for characterising complex biological specimens and materials through a phase retrieval process known as Electron Ptychography (EP). The efficacy of 4D STEM acquisition and the resulting quality of EP reconstruction depends on the overlap ratio of adjacent illuminated areas. This paper demonstrates how the overlap ratio impacts the data redundancy and the quality of the EP reconstruction. We define two quantities as a function of the overlap ratio that are independent of both the object and the EP algorithm. Subsequently, we evaluate an EP algorithm for varying overlap ratios using simulated 4D STEM datasets. Notably, a 40% or greater overlap ratio yields stable, high-quality reconstructions. 

**Abstract (ZH)**: 使用发散电子束获取数据的四维扫描透射电子显微镜（4D STEM）是一种通过电子 Ptychography（EP）重建过程来表征复杂生物样本和材料的有前景的工具。4D STEM 数据采集的有效性和 EP 重建结果的质量取决于相邻照射区域的重叠比率。本文展示了重叠比率如何影响数据冗余度和 EP 重建质量。我们定义了两个量，它们与对象和 EP 算法无关。随后，我们使用模拟的 4D STEM 数据集评估了不同重叠比率下的 EP 算法表现。值得注意的是，重叠比率在 40% 或更高时可以实现稳定且高质量的重建。 

---
# Zero-Shot Warning Generation for Misinformative Multimodal Content 

**Title (ZH)**: 零样本误导性多模态内容预警生成 

**Authors**: Giovanni Pio Delvecchio, Huy Hong Nguyen, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2502.00752)  

**Abstract**: The widespread prevalence of misinformation poses significant societal concerns. Out-of-context misinformation, where authentic images are paired with false text, is particularly deceptive and easily misleads audiences. Most existing detection methods primarily evaluate image-text consistency but often lack sufficient explanations, which are essential for effectively debunking misinformation. We present a model that detects multimodal misinformation through cross-modality consistency checks, requiring minimal training time. Additionally, we propose a lightweight model that achieves competitive performance using only one-third of the parameters. We also introduce a dual-purpose zero-shot learning task for generating contextualized warnings, enabling automated debunking and enhancing user comprehension. Qualitative and human evaluations of the generated warnings highlight both the potential and limitations of our approach. 

**Abstract (ZH)**: 广泛存在的错误信息对社会构成了重大关切。脱离语境的错误信息，即真实图片配以虚假文字，尤其具有欺骗性，并容易误导受众。现有大多数检测方法主要评估图像与文字的一致性，但往往缺乏足够的解释，而这些解释对于有效地揭示错误信息至关重要。我们提出了一种通过跨模态一致性检查来检测多模态错误信息的模型，该模型所需训练时间minimal。此外，我们提出了一种轻量级模型，仅使用参数总量的三分之一便实现了竞争力的性能。我们还引入了一项双重目的的零样本学习任务，用于生成上下文化的警告，以实现自动化揭示错误信息并增强用户理解。生成的警告在定性和人类评估中均突显出该方法的潜力和局限性。 

---
# Predictive modeling and anomaly detection in large-scale web portals through the CAWAL framework 

**Title (ZH)**: 通过CAWAL框架进行大型网络门户的预测建模与异常检测 

**Authors**: Ozkan Canay, Umit Kocabicak  

**Link**: [PDF](https://arxiv.org/pdf/2502.00413)  

**Abstract**: This study presents an approach that uses session and page view data collected through the CAWAL framework, enriched through specialized processes, for advanced predictive modeling and anomaly detection in web usage mining (WUM) applications. Traditional WUM methods often rely on web server logs, which limit data diversity and quality. Integrating application logs with web analytics, the CAWAL framework creates comprehensive session and page view datasets, providing a more detailed view of user interactions and effectively addressing these limitations. This integration enhances data diversity and quality while eliminating the preprocessing stage required in conventional WUM, leading to greater process efficiency. The enriched datasets, created by cross-integrating session and page view data, were applied to advanced machine learning models, such as Gradient Boosting and Random Forest, which are known for their effectiveness in capturing complex patterns and modeling non-linear relationships. These models achieved over 92% accuracy in predicting user behavior and significantly improved anomaly detection capabilities. The results show that this approach offers detailed insights into user behavior and system performance metrics, making it a reliable solution for improving large-scale web portals' efficiency, reliability, and scalability. 

**Abstract (ZH)**: 本研究提出了一种方法，该方法利用通过CAWAL框架收集的会话和页面视图数据，并通过专门的处理过程进行丰富，用于网络使用挖掘（WUM）应用中的高级预测建模和异常检测。传统WUM方法往往依赖于网站服务器日志，这限制了数据的多样性和质量。通过将应用程序日志与网络分析集成，CAWAL框架创建了全面的会话和页面视图数据集，从而提供了更详尽的用户交互视图，并有效解决了这些限制。这种集成增强了数据的多样性和质量，同时消除了传统WUM所需的预处理阶段，从而提高了处理效率。通过跨集成会话和页面视图数据创建的丰富数据集应用于先进的机器学习模型，如梯度提升和随机森林，这些模型因其在捕捉复杂模式和建模非线性关系方面的有效性而闻名。这些模型在预测用户行为方面达到了超过92%的准确率，并显著提高了异常检测能力。研究结果表明，该方法提供了对用户行为和系统性能指标的详细见解，使其成为提高大规模网络门户网站效率、可靠性和可扩展性的可靠解决方案。 

---
# MODS: Moderating a Mixture of Document Speakers to Summarize Debatable Queries in Document Collections 

**Title (ZH)**: MODS：调节文档演讲者混合以总结存有争议的查询在文档集合中的内容 

**Authors**: Nishant Balepur, Alexa Siu, Nedim Lipka, Franck Dernoncourt, Tong Sun, Jordan Boyd-Graber, Puneet Mathur  

**Link**: [PDF](https://arxiv.org/pdf/2502.00322)  

**Abstract**: Query-focused summarization (QFS) gives a summary of documents to answer a query. Past QFS work assumes queries have one answer, ignoring debatable ones (Is law school worth it?). We introduce Debatable QFS (DQFS), a task to create summaries that answer debatable queries via documents with opposing perspectives; summaries must comprehensively cover all sources and balance perspectives, favoring no side. These goals elude LLM QFS systems, which: 1) lack structured content plans, failing to guide LLMs to write balanced summaries, and 2) use the same query to retrieve contexts across documents, failing to cover all perspectives specific to each document's content. To overcome this, we design MODS, a multi-LLM framework mirroring human panel discussions. MODS treats documents as individual Speaker LLMs and has a Moderator LLM that picks speakers to respond to tailored queries for planned topics. Speakers use tailored queries to retrieve relevant contexts from their documents and supply perspectives, which are tracked in a rich outline, yielding a content plan to guide the final summary. Experiments on ConflictingQA with controversial web queries and DebateQFS, our new dataset of debate queries from Debatepedia, show MODS beats SOTA by 38-59% in topic paragraph coverage and balance, based on new citation metrics. Users also find MODS's summaries to be readable and more balanced. 

**Abstract (ZH)**: 查询导向总结（QFS）为回答查询提供文档摘要。过去的QFS工作假设查询有一个明确的答案，忽略了有争议的问题（例如，“法学院值得吗？”）。我们引入了有争议的QFS（DQFS），这是一个通过包含对立观点的文档来创建摘要的任务，以回答有争议的查询；摘要必须全面涵盖各种资源，并平衡观点，不偏袒任何一方。这些目标超出了当前的大型语言模型（LLM）QFS系统的范围，因为：1) 它们缺乏结构化的内容计划，无法引导LLM撰写平衡的摘要，2) 它们使用相同的查询从各个文档中检索背景信息，未能覆盖与每个文档内容特定相关的所有观点。为了解决这些问题，我们设计了MODS，这是一种多LLM框架，模仿人类的小组讨论。MODS将文档视为个体演讲者LLM，并配备了一个主持人LLM，主持人会选择合适的演讲者来针对特定查询回应预定的主题。演讲者使用定制的查询从其文档中检索相关信息并提供观点，这些观点在丰富的提纲中被跟踪，从而形成一项内容计划来指导最终的摘要。通过在具有争议性网络查询的ConflictingQA数据集和我们的新数据集DebateQFS（来自Debatepedia的辩论查询）上的实验表明，MODS在主题段落覆盖和平衡方面优于当前最佳技术（SOTA）38-59%，根据新的引用度量标准。用户也发现MODS生成的摘要更易读且更平衡。 

---
# Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation 

**Title (ZH)**: 《谜题来了！基于检索增强生成的隐蔽性成员推理》

这个翻译符合学术规范，并且准确地传达了原文的意思。其中，“Stealthy Membership Inference”被翻译为“隐蔽性成员推理”，强调了该方法的隐蔽性和欺骗性特点。“Retrieval-Augmented Generation”则翻译为“基于检索增强生成”，准确反映了该方法利用检索机制增强生成过程的特点。 

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2502.00306)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference. 

**Abstract (ZH)**: 检索增强生成（RAG）使大型语言模型（LLMs）能够在不修改模型参数的情况下利用外部知识库生成基于事实的响应。虽然缺乏权重调整可以防止通过模型参数泄漏信息，但这也引入了模型上下文中的推断对手利用检索到的文档的风险。现有的成员身份推断和数据提取方法通常依赖于脱狱或精心构造的不自然查询，这些方法可以被常见的RAG系统中常用的查询重写技术轻易检测或阻止。在本工作中，我们提出了一种名为查询攻击（Interrogation Attack, IA）的技术，旨在对RAG数据存储中的文档进行成员身份推断。通过构造只能在目标文档存在的情况下才能回答的自然文本查询，我们的方法仅使用30个查询即可成功进行推断，并且保持隐蔽性；现有的简单检测器比攻击产生的提示识别出的对手提示多约76倍。在多种RAG配置中，我们观察到在1%假阳性率下的查全率（TPR@1%FPR）提高了2倍，同时每文档推理的成本低于0.02美元。 

---
# DEUCE: Dual-diversity Enhancement and Uncertainty-awareness for Cold-start Active Learning 

**Title (ZH)**: DEUCE：冷启动主动学习中的双重多样性增强与不确定性感知 

**Authors**: Jiaxin Guo, C. L. Philip Chen, Shuzhen Li, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00305)  

**Abstract**: Cold-start active learning (CSAL) selects valuable instances from an unlabeled dataset for manual annotation. It provides high-quality data at a low annotation cost for label-scarce text classification. However, existing CSAL methods overlook weak classes and hard representative examples, resulting in biased learning. To address these issues, this paper proposes a novel dual-diversity enhancing and uncertainty-aware (DEUCE) framework for CSAL. Specifically, DEUCE leverages a pretrained language model (PLM) to efficiently extract textual representations, class predictions, and predictive uncertainty. Then, it constructs a Dual-Neighbor Graph (DNG) to combine information on both textual diversity and class diversity, ensuring a balanced data distribution. It further propagates uncertainty information via density-based clustering to select hard representative instances. DEUCE performs well in selecting class-balanced and hard representative data by dual-diversity and informativeness. Experiments on six NLP datasets demonstrate the superiority and efficiency of DEUCE. 

**Abstract (ZH)**: 冷启动主动学习（CSAL）可以从未标记的数据集中选择有价值的实例进行手动注释，以在标签稀缺的文本分类中提供高质量的数据，同时降低成本。然而，现有的CSAL方法忽视了弱类和具有代表性的困难示例，导致学习偏差。为解决这些问题，本文提出了一种新颖的双重多样性增强和不确定性感知（DEUCE）框架用于CSAL。具体而言，DEUCE 利用预训练语言模型（PLM）高效地提取文本表示、类别预测和预测不确定性。然后，它构建了一个双重邻接图（DNG），以结合文本多样性和类别多样性中的信息，确保数据分布的均衡。此外，DEUCE 通过基于密度的聚类传播不确定性信息，选择具有代表性的困难实例。DEUCE 通过双重多样性和信息性在选择类别平衡和具有代表性的困难数据方面表现出色。实验在六个NLP数据集上的结果显示了DEUCE的优势和效率。 

---
# Towards Recommender Systems LLMs Playground (RecSysLLMsP): Exploring Polarization and Engagement in Simulated Social Networks 

**Title (ZH)**: 向导型企业语言模型（RecSysLLMsP）：探索模拟社会网络中的极化与互动 

**Authors**: Ljubisa Bojic, Zorica Dodevska, Yashar Deldjoo, Nenad Pantelic  

**Link**: [PDF](https://arxiv.org/pdf/2502.00055)  

**Abstract**: Given the exponential advancement in AI technologies and the potential escalation of harmful effects from recommendation systems, it is crucial to simulate and evaluate these effects early on. Doing so can help prevent possible damage to both societies and technology companies. This paper introduces the Recommender Systems LLMs Playground (RecSysLLMsP), a novel simulation framework leveraging Large Language Models (LLMs) to explore the impacts of different content recommendation setups on user engagement and polarization in social networks. By creating diverse AI agents (AgentPrompts) with descriptive, static, and dynamic attributes, we assess their autonomous behaviour across three scenarios: Plurality, Balanced, and Similarity. Our findings reveal that the Similarity Scenario, which aligns content with user preferences, maximizes engagement while potentially fostering echo chambers. Conversely, the Plurality Scenario promotes diverse interactions but produces mixed engagement results. Our study emphasizes the need for a careful balance in recommender system designs to enhance user satisfaction while mitigating societal polarization. It underscores the unique value and challenges of incorporating LLMs into simulation environments. The benefits of RecSysLLMsP lie in its potential to calculate polarization effects, which is crucial for assessing societal impacts and determining user engagement levels with diverse recommender system setups. This advantage is essential for developing and maintaining a successful business model for social media companies. However, the study's limitations revolve around accurately emulating reality. Future efforts should validate the similarity in behaviour between real humans and AgentPrompts and establish metrics for measuring polarization scores. 

**Abstract (ZH)**: 随着人工智能技术的指数级进步以及推荐系统可能引发的负面效应加剧，及早模拟和评估这些效应显得尤为重要。这不仅有助于预防可能对社会和科技公司造成的损害，还能为推荐系统的改进提供有效指导。本文介绍了“推荐系统大语言模型游乐场”（RecSysLLMsP），这是一种利用大语言模型（LLMs）探索不同内容推荐设置对社交网络用户参与度和极化影响的新型模拟框架。通过创建具有描述性、静态和动态属性的多样化AI代理（AgentPrompts），我们在三种情景下评估它们的自主行为：多元（Plurality）、平衡（Balanced）和相似（Similarity）。

研究发现，当内容与用户偏好一致时，相似情景（Similarity Scenario）能够最大化用户的参与度，但可能会促进回音室效应。相比之下，多元情景（Plurality Scenario）则促进多样化的交互，但用户参与效果参差不齐。本研究强调设计推荐系统时需要保持谨慎的平衡，旨在提升用户体验的同时减少社会极化的风险。它突显了将大语言模型集成到模拟环境中的独特价值和挑战。

RecSysLLMsP 的优势在于能计算出极化效应，这对于评估社会影响并确定不同的推荐系统设置下用户的参与程度至关重要，对于社交媒体公司的业务模式发展和维持具有重要价值。然而，研究的局限性主要集中在准确模拟现实挑战上。未来的研究应验证现实人类行为和AgentPrompts行为之间的相似性，并建立衡量极化分数的指标。 

---
# Querying Databases with Function Calling 

**Title (ZH)**: 用函数调用查询数据库 

**Authors**: Connor Shorten, Charles Pierse, Thomas Benjamin Smith, Karel D'Oosterlinck, Tuana Celik, Erika Cardenas, Leonie Monigatti, Mohd Shukri Hasan, Edward Schmuhl, Daniel Williams, Aravind Kesiraju, Bob van Luijt  

**Link**: [PDF](https://arxiv.org/pdf/2502.00032)  

**Abstract**: The capabilities of Large Language Models (LLMs) are rapidly accelerating largely thanks to their integration with external tools. Querying databases is among the most effective of these integrations, enabling LLMs to access private or continually updating data. While Function Calling is the most common method for interfacing external tools to LLMs, its application to database querying as a tool has been underexplored. We propose a tool definition for database querying that unifies accessing data with search queries, filters, or a combination both, as well as transforming results with aggregation and groupby operators. To evaluate its effectiveness, we conduct a study with 8 LLMs spanning 5 model families. We present a novel pipeline adapting the Gorilla LLM framework to create synthetic database schemas and queries. We primarily evaluate the models with the Exact Match of predicted and ground truth query APIs. Among the models tested, Claude 3.5 Sonnet achieves the highest performance with an Exact Match score of 74.3%, followed by GPT-4o mini at 73.7%, and GPT-4o at 71.8%. We further breakdown these results per API component utilized and across synthetic use cases. We find that LLMs are highly effective at utilizing operators on boolean properties, but struggle with text property filters. Across use cases we find robust results with the higher performing models such as GPT-4o, but significant performance variance across use cases from lower performing models. We additionally conduct ablation studies exploring the impact of parallel tool calling, adding a rationale as an argument of the tool call, using a separate tool per database collection, and tool calling with structured outputs. Our findings demonstrate the effectiveness of enabling LLMs to query databases with Function Calling. We have open-sourced our experimental code and results at this http URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的能力正在迅速加速，这主要得益于它们与外部工具的整合。查询数据库是这些整合中最有效的形式之一，使LLMs能够访问私有数据或实时更新的数据。尽管函数调用是将外部工具与LLMs接口的最常用方法，但将其应用于数据库查询工具的潜力尚未被充分探索。本文提出了一种数据库查询工具定义，该定义综合了使用搜索查询、过滤条件或两者的组合访问数据的方法，以及使用聚合和分组操作变换结果。为了评估其有效性，我们在涵盖5个模型家族的8个LLM上进行了研究。我们介绍了一种新颖的流水线，将Gorilla LLM框架适应用于生成合成数据库模式和查询。我们主要通过预测查询API与真实查询API的精确匹配来评估模型。在测试的模型中，Claude 3.5 Sonnet的精确匹配得分为74.3%，其次是GPT-4o mini的73.7%和GPT-4o的71.8%。我们进一步按利用的API组件和合成用例类别分拆这些结果。我们发现，LLMs在利用布尔属性上的操作方面非常有效，但在处理文本属性过滤方面存在困难。在各种用例中，我们发现表现较好的模型如GPT-4o具有稳健的结果，但表现较差的模型在不同用例中的性能差异较大。我们还进行了消融研究，探讨了并行工具调用、在工具调用中添加理由、为每个数据库集合使用一个单独的工具以及带有结构化输出的工具调用的影响。我们的研究结果表明，启用LLMs使用函数调用查询数据库是有效的。我们已将实验代码和结果开源在该网址：<网址>。 

---
