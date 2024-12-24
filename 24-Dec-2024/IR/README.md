# Leveraging Memory Retrieval to Enhance LLM-based Generative Recommendation 

**Title (ZH)**: 利用记忆检索增强基于大规模语言模型的生成性推荐 

**Authors**: Chengbing Wang, Yang Zhang, Fengbin Zhu, Jizhi Zhang, Tianhao Shi, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17593)  

**Abstract**: Leveraging Large Language Models (LLMs) to harness user-item interaction histories for item generation has emerged as a promising paradigm in generative recommendation. However, the limited context window of LLMs often restricts them to focusing on recent user interactions only, leading to the neglect of long-term interests involved in the longer histories. To address this challenge, we propose a novel Automatic Memory-Retrieval framework (AutoMR), which is capable of storing long-term interests in the memory and extracting relevant information from it for next-item generation within LLMs. Extensive experimental results on two real-world datasets demonstrate the effectiveness of our proposed AutoMR framework in utilizing long-term interests for generative recommendation. 

**Abstract (ZH)**: 利用大规模语言模型（LLMs）捕获用户-项目交互历史以生成项目，在生成推荐中展现出一种有前景的范式。然而，LLMs 的有限上下文窗口经常限制它们仅关注最近的用户交互，而忽略了长时间历史中涉及的长期兴趣。为解决这一挑战，我们提出了一种新的自动记忆检索框架（AutoMR），该框架能够存储长期兴趣并在LLMs中从记忆中提取相关信息用于生成下一个项目。在两个真实世界数据集上的 extensive 实验结果表明，我们的 AutoMR 框架在利用长期兴趣进行生成推荐方面具有有效性。 

---
# CiteBART: Learning to Generate Citations for Local Citation Recommendation 

**Title (ZH)**: CiteBART：学习为局部引文推荐生成引文 

**Authors**: Ege Yiğit Çelik, Selma Tekir  

**Link**: [PDF](https://arxiv.org/pdf/2412.17534)  

**Abstract**: Citations are essential building blocks in scientific writing. The scientific community is longing for support in their generation. Citation generation involves two complementary subtasks: Determining the citation worthiness of a context and, if it's worth it, proposing the best candidate papers for the citation placeholder. The latter subtask is called local citation recommendation (LCR). This paper proposes CiteBART, a custom BART pre-training based on citation token masking to generate citations to achieve LCR. In the base scheme, we mask the citation token in the local citation context to make the citation prediction. In the global one, we concatenate the citing paper's title and abstract to the local citation context to learn to reconstruct the citation token. CiteBART outperforms state-of-the-art approaches on the citation recommendation benchmarks except for the smallest FullTextPeerRead dataset. The effect is significant in the larger benchmarks, e.g., Refseer and ArXiv. We present a qualitative analysis and an ablation study to provide insights into the workings of CiteBART. Our analyses confirm that its generative nature brings about a zero-shot capability. 

**Abstract (ZH)**: 引文是科学写作的基本组成部分。科学界渴望建立一套支持引文生成的方法。引文生成涉及两个互补的子任务：确定某一文字段落的引文价值，并在有必要时提出最适合的候选论文作为引文填充项。后者被称为局部引文推荐（Local Citation Recommendation, LCR）。本文提出了一种名为CiteBART的方法，该方法基于引文标记遮蔽的自定义BART预训练模型，以实现LCR。在基线方案中，通过遮蔽局部引文上下文中的引文标记来预测引文；而在全局方案中，将引用论文的标题和摘要与局部引文上下文连接起来，以学习重建引文标记。CiteBART在引文推荐基准测试中（除了最小的FullTextPeerRead数据集外）优于现有方法。特别是在Refseer和ArXiv等更大的基准测试中，效果尤为显著。我们进行了定性分析和消融研究，以提供对CiteBART工作原理的见解。我们的分析证实，其生成特性赋予了它零样本能力。 

---
# Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark 

**Title (ZH)**: 场景导向的推荐：一种多场景推荐基准 

**Authors**: Xiaopeng Li, Jingtong Gao, Pengyue Jia, Yichao Wang, Wanyu Wang, Yejing Wang, Yuhao Wang, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17374)  

**Abstract**: Multi Scenario Recommendation (MSR) tasks, referring to building a unified model to enhance performance across all recommendation scenarios, have recently gained much attention. However, current research in MSR faces two significant challenges that hinder the field's development: the absence of uniform procedures for multi-scenario dataset processing, thus hindering fair comparisons, and most models being closed-sourced, which complicates comparisons with current SOTA models. Consequently, we introduce our benchmark, \textbf{Scenario-Wise Rec}, which comprises 6 public datasets and 12 benchmark models, along with a training and evaluation pipeline. Additionally, we validated the benchmark using an industrial advertising dataset, reinforcing its reliability and applicability in real-world scenarios. We aim for this benchmark to offer researchers valuable insights from prior work, enabling the development of novel models based on our benchmark and thereby fostering a collaborative research ecosystem in MSR. Our source code is also publicly available. 

**Abstract (ZH)**: 多场景推荐（Multi-Scenario Recommendation, MSR）任务指的是构建一个统一模型以提升在所有推荐场景中的性能，近年来引起了广泛关注。然而，当前在MSR领域的研究面临着两个显著的挑战，这些挑战阻碍了该领域的发展：一是缺乏统一的多场景数据集处理流程，从而妨碍了公平的比较；二是大多数模型是闭源的，这使与其他现有最先进的模型（SOTA）进行比较变得复杂。因此，我们提出了一个基准测试系统——**场景导向型推荐**（Scenario-Wise Rec），该系统包括6个公开数据集和12个基准模型，并提供了一个训练和评估流程。此外，我们还使用工业广告数据集对基准进行了验证，增强了其在实际应用场景中的可靠性和适用性。我们希望这个基准能够为研究人员提供宝贵的先前研究洞察，从而基于我们的基准开发新的模型，进而促进MSR领域的协作研究生态。我们的源代码也是公开的。 

---
# Efficient fine-tuning methodology of text embedding models for information retrieval: contrastive learning penalty (clp) 

**Title (ZH)**: 基于对比学习惩罚（CLP）的文本嵌入模型高效微调方法：面向信息检索的应用 

**Authors**: Jeongsu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17364)  

**Abstract**: Text embedding models play a crucial role in natural language processing, particularly in information retrieval, and their importance is further highlighted with the recent utilization of RAG (Retrieval- Augmented Generation). This study presents an efficient fine-tuning methodology encompassing data selection, loss function, and model architecture to enhance the information retrieval performance of pre-trained text embedding models. In particular, this study proposes a novel Contrastive Learning Penalty function that overcomes the limitations of existing Contrastive Learning. The proposed methodology achieves significant performance improvements over existing methods in document retrieval tasks. This study is expected to contribute to improving the performance of information retrieval systems through fine-tuning of text embedding models. The code for this study can be found at this https URL, and the best-performing model can be found at this https URL. 

**Abstract (ZH)**: 文本嵌入模型在自然语言处理中扮演着关键角色，特别是在信息检索方面，其重要性随着最近RAG（检索增强生成）的利用而进一步凸显。本研究提出了一种高效的微调方法，涵盖了数据选择、损失函数和模型架构等方面，以提升预训练文本嵌入模型的信息检索性能。特别是，本研究提出了一种新颖的对比学习惩罚函数，克服了现有对比学习的局限性。所提出的方法在文档检索任务中取得了显著的性能提升。本研究预计将通过文本嵌入模型的微调来提升信息检索系统的性能。本文中的代码可以从以下链接获取：[此处插入链接]，而表现最佳的模型可以从以下链接获取：[此处插入链接]。 

---
# Popularity Estimation and New Bundle Generation using Content and Context based Embeddings 

**Title (ZH)**: 基于内容和上下文嵌入的流行度估计与新捆绑包生成 

**Authors**: Ashutosh Nayak, Prajwal NJ, Sameeksha Keshav, Kavitha S.N., Roja Reddy, Rajasekhara Reddy Duvvuru Muni  

**Link**: [PDF](https://arxiv.org/pdf/2412.17310)  

**Abstract**: Recommender systems create enormous value for businesses and their consumers. They increase revenue for businesses while improving the consumer experience by recommending relevant products amidst huge product base. Product bundling is an exciting development in the field of product recommendations. It aims at generating new bundles and recommending exciting and relevant bundles to their consumers. Unlike traditional recommender systems that recommend single items to consumers, product bundling aims at targeting a bundle, or a set of items, to the consumers. While bundle recommendation has attracted significant research interest recently, extant literature on bundle generation is scarce. Moreover, metrics to identify if a bundle is popular or not is not well studied. In this work, we aim to fulfill this gap by introducing new bundle popularity metrics based on sales, consumer experience and item diversity in a bundle. We use these metrics in the methodology proposed in this paper to generate new bundles for mobile games using content aware and context aware embeddings. We use opensource Steam Games dataset for our analysis. Our experiments indicate that we can generate new bundles that can outperform the existing bundles on the popularity metrics by 32% - 44%. Our experiments are computationally efficient and the proposed methodology is generic that can be extended to other bundling problems e.g. product bundling, music bundling. 

**Abstract (ZH)**: 推荐系统为企业和消费者创造了巨大的价值。它们通过在庞大的产品库中推荐相关产品，从而增加企业的收入，同时提升消费者的体验。产品捆绑是产品推荐领域的一项令人兴奋的发展。其目的是生成新的捆绑包，并向消费者推荐新颖且相关的产品捆绑包。与传统的推荐系统仅向消费者推荐单一项目不同，产品捆绑旨在为目标消费者推荐一组项目。虽然捆绑推荐近年来吸引了大量的研究兴趣，但现有关于捆绑生成的研究文献相对较少。此外，用于识别捆绑是否受欢迎的标准和指标也研究不足。在这项工作中，我们旨在通过引入基于销售、消费者体验和捆绑包内项目多样性的新捆绑包流行度指标来填补这一空白。我们使用这些指标在本文提出的框架中生成用于移动游戏的新捆绑包，利用内容感知和上下文感知的嵌入。我们使用开源的Steam游戏数据集进行分析。我们的实验表明，我们生成的新捆绑包在流行度指标上可以比现有捆绑包高出32%至44%。我们的实验计算效率高，提出的框架具有通用性，可以应用于其他捆绑问题，例如产品捆绑和音乐捆绑。 

---
# SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval 

**Title (ZH)**: SyNeg：由大型语言模型驱动的合成硬负例在密集检索中的应用 

**Authors**: Xiaopeng Li, Xiangyang Li, Hao Zhang, Zhaocheng Du, Pengyue Jia, Yichao Wang, Xiangyu Zhao, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17250)  

**Abstract**: The performance of Dense retrieval (DR) is significantly influenced by the quality of negative sampling. Traditional DR methods primarily depend on naive negative sampling techniques or on mining hard negatives through external retriever and meticulously crafted strategies. However, naive negative sampling often fails to adequately capture the accurate boundaries between positive and negative samples, whereas existing hard negative sampling methods are prone to false negatives, resulting in performance degradation and training instability. Recent advancements in large language models (LLMs) offer an innovative solution to these challenges by generating contextually rich and diverse negative samples. In this work, we present a framework that harnesses LLMs to synthesize high-quality hard negative samples. We first devise a \textit{multi-attribute self-reflection prompting strategy} to direct LLMs in hard negative sample generation. Then, we implement a \textit{hybrid sampling strategy} that integrates these synthetic negatives with traditionally retrieved negatives, thereby stabilizing the training process and improving retrieval performance. Extensive experiments on five benchmark datasets demonstrate the efficacy of our approach, and code is also publicly available. 

**Abstract (ZH)**: 密集检索（DR）的性能显著受负样本采样质量的影响。传统DR方法主要依赖于朴素的负样本采样技术或通过外部检索器挖掘难以区分的负样本，并采用精心设计的策略。然而，朴素的负样本采样常常未能充分捕捉正负样本之间的准确边界，而现有的难以区分的负样本采样方法则容易产生假的负样本，导致性能下降和训练不稳定。近年来，大型语言模型（LLMs）的发展为解决这些挑战提供了创新的解决方案，通过生成上下文丰富且多样化的负样本。本文提出了一种框架，利用LLMs合成高质量的难以区分的负样本。首先，我们设计了一种\textit{多属性自我反思提示策略}，以指导LLMs生成难以区分的负样本。然后，我们采用了\textit{混合采样策略}，将这些合成的负样本与传统检索得到的负样本结合，从而稳定训练过程并提高检索性能。在五个基准数据集上的大量实验表明了我们方法的有效性，并且相关代码已公开。 

---
# GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems 

**Title (ZH)**: GraphHash：图聚类实现推荐系统中的参数效率 

**Authors**: Xinyi Wu, Donald Loveland, Runjin Chen, Yozen Liu, Xin Chen, Leonardo Neves, Ali Jadbabaie, Clark Mingxuan Ju, Neil Shah, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.17245)  

**Abstract**: Deep recommender systems rely heavily on large embedding tables to handle high-cardinality categorical features such as user/item identifiers, and face significant memory constraints at scale. To tackle this challenge, hashing techniques are often employed to map multiple entities to the same embedding and thus reduce the size of the embedding tables. Concurrently, graph-based collaborative signals have emerged as powerful tools in recommender systems, yet their potential for optimizing embedding table reduction remains unexplored. This paper introduces GraphHash, the first graph-based approach that leverages modularity-based bipartite graph clustering on user-item interaction graphs to reduce embedding table sizes. We demonstrate that the modularity objective has a theoretical connection to message-passing, which provides a foundation for our method. By employing fast clustering algorithms, GraphHash serves as a computationally efficient proxy for message-passing during preprocessing and a plug-and-play graph-based alternative to traditional ID hashing. Extensive experiments show that GraphHash substantially outperforms diverse hashing baselines on both retrieval and click-through-rate prediction tasks. In particular, GraphHash achieves on average a 101.52% improvement in recall when reducing the embedding table size by more than 75%, highlighting the value of graph-based collaborative information for model reduction. 

**Abstract (ZH)**: 深度推荐系统高度依赖于大型嵌入表来处理高基数的分类特征，如用户/项标识符，并在规模上面临显著的内存限制。为了解决这一挑战，经常采用散列技术将多个实体映射到相同的嵌入，从而减小嵌入表的大小。与此同时，基于图的协作信号已成为推荐系统中强有力的工具，但它们在优化嵌入表压缩方面的潜力尚未被充分探索。本文提出了GraphHash，这是第一个利用基于模块性的二分图簇聚类方法来减少嵌入表大小的基于图的方法。我们表明，模块性目标与消息传递之间存在理论联系，为我们的方法奠定了基础。通过采用快速聚类算法，GraphHash 在预处理过程中为消息传递提供了一个高效的代理，并为传统的ID散列提供了一种灵活的基于图的替代方案。大量实验表明，GraphHash 在检索和点击率预测任务上都显著优于各种基础散列方法。特别是，当嵌入表大小减少超过75%时，GraphHash 的召回率平均提高101.52%，这突显了基于图的协作信息在模型压缩方面的价值。 

---
# LLM-based relevance assessment still can't replace human relevance assessment 

**Title (ZH)**: 基于大语言模型的相关性评估仍然无法替代人工相关性评估 

**Authors**: Charles L. A. Clarke, Laura Dietz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17156)  

**Abstract**: The use of large language models (LLMs) for relevance assessment in information retrieval has gained significant attention, with recent studies suggesting that LLM-based judgments provide comparable evaluations to human judgments. Notably, based on TREC 2024 data, Upadhyay et al. make a bold claim that LLM-based relevance assessments, such as those generated by the UMBRELA system, can fully replace traditional human relevance assessments in TREC-style evaluations. This paper critically examines this claim, highlighting practical and theoretical limitations that undermine the validity of this conclusion. First, we question whether the evidence provided by Upadhyay et al. really supports their claim, particularly if a test collection is used asa benchmark for future improvements. Second, through a submission deliberately intended to do so, we demonstrate the ease with which automatic evaluation metrics can be subverted, showing that systems designed to exploit these evaluations can achieve artificially high scores. Theoretical challenges -- such as the inherent narcissism of LLMs, the risk of overfitting to LLM-based metrics, and the potential degradation of future LLM performance -- must be addressed before LLM-based relevance assessments can be considered a viable replacement for human judgments. 

**Abstract (ZH)**: 大语言模型（LLMs）在信息检索中的相关性评估应用得到了广泛关注，近期的研究表明，基于LLM的判断与人类判断提供了一致的评估结果。值得注意的是，Upadhyay等人基于TREC 2024数据提出，基于LLM的相关性评估，如UMBRELA系统生成的评估，可以在TREC风格的评估中完全替代传统的手工相关性评估。本文对该主张进行了批判性探讨，指出了支撑这一结论的实践和理论限制。首先，我们质疑Upadhyay等人提供的证据是否真正支持其主张，特别是如果一个测试集合被用作未来改进的标准。其次，通过一个故意设计用于展示这一点的提交，我们证明自动评估指标可以被轻松操控，表明那些专门利用这些评估的系统可以达到虚假的高分数。理论上的挑战——如LLMs的固有自恋倾向、过度拟合到基于LLM的指标的风险以及未来LLM性能可能的退化——必须得到解决，才能使基于LLM的相关性评估被视为手工判断的有效替代物。 

---
# Iterative NLP Query Refinement for Enhancing Domain-Specific Information Retrieval: A Case Study in Career Services 

**Title (ZH)**: 基于迭代自然语言处理查询修正的领域特定信息检索增强：以职业服务为例的研究 

**Authors**: Elham Peimani, Gurpreet Singh, Nisarg Mahyavanshi, Aman Arora, Awais Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2412.17075)  

**Abstract**: Retrieving semantically relevant documents in niche domains poses significant challenges for traditional TF-IDF-based systems, often resulting in low similarity scores and suboptimal retrieval performance. This paper addresses these challenges by introducing an iterative and semi-automated query refinement methodology tailored to Humber College's career services webpages. Initially, generic queries related to interview preparation yield low top-document similarities (approximately 0.2--0.3). To enhance retrieval effectiveness, we implement a two-fold approach: first, domain-aware query refinement by incorporating specialized terms such as resources-online-learning, student-online-services, and career-advising; second, the integration of structured educational descriptors like "online resume and interview improvement tools." Additionally, we automate the extraction of domain-specific keywords from top-ranked documents to suggest relevant terms for query expansion. Through experiments conducted on five baseline queries, our semi-automated iterative refinement process elevates the average top similarity score from approximately 0.18 to 0.42, marking a substantial improvement in retrieval performance. The implementation details, including reproducible code and experimental setups, are made available in our GitHub repositories \url{this https URL} and \url{this https URL}. We also discuss the limitations of our approach and propose future directions, including the integration of advanced neural retrieval models. 

**Abstract (ZH)**: 传统的基于TF-IDF的系统在检索专业领域内的语义相关文档时面临着显著挑战，常常导致相似度分数较低和检索性能不佳。本文通过引入一种针对哈伯默学院职业服务网页的迭代且半自动化的查询优化方法，来应对这些挑战。最初，与面试准备相关的通用查询在顶级文档相似度方面表现不佳（大约为0.2-0.3）。为了提高检索效果，我们采用了两步方法：首先，通过引入专业术语（如resources-online-learning、student-online-services、career-advising）进行领域意识下的查询优化；其次，将结构化的教育描述符（如“在线简历和面试改进工具”）整合进来。此外，我们自动提取顶级文档中的领域特定关键词，以建议查询扩展的相关术语。通过针对五个基础查询进行的实验，我们的半自动迭代优化过程将平均顶级相似度分数从大约0.18提高到0.42，显著提升了检索性能。实验的详细信息，包括可复现的代码和实验设置，已在我们的GitHub仓库中公开，网址分别为this https URL和this https URL。我们还讨论了该方法的局限性，并提出了未来工作方向，包括整合先进的神经检索模型。 

---
# LLM-Powered User Simulator for Recommender System 

**Title (ZH)**: 基于LLM的用户模拟器在推荐系统中的应用 

**Authors**: Zijian Zhang, Shuchang Liu, Ziru Liu, Rui Zhong, Qingpeng Cai, Xiangyu Zhao, Chunxu Zhang, Qidong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16984)  

**Abstract**: User simulators can rapidly generate a large volume of timely user behavior data, providing a testing platform for reinforcement learning-based recommender systems, thus accelerating their iteration and optimization. However, prevalent user simulators generally suffer from significant limitations, including the opacity of user preference modeling and the incapability of evaluating simulation accuracy. In this paper, we introduce an LLM-powered user simulator to simulate user engagement with items in an explicit manner, thereby enhancing the efficiency and effectiveness of reinforcement learning-based recommender systems training. Specifically, we identify the explicit logic of user preferences, leverage LLMs to analyze item characteristics and distill user sentiments, and design a logical model to imitate real human engagement. By integrating a statistical model, we further enhance the reliability of the simulation, proposing an ensemble model that synergizes logical and statistical insights for user interaction simulations. Capitalizing on the extensive knowledge and semantic generation capabilities of LLMs, our user simulator faithfully emulates user behaviors and preferences, yielding high-fidelity training data that enrich the training of recommendation algorithms. We establish quantifying and qualifying experiments on five datasets to validate the simulator's effectiveness and stability across various recommendation scenarios. 

**Abstract (ZH)**: 用户模拟器可以快速生成大量及时的用户行为数据，为基于强化学习的推荐系统提供测试平台，从而加速其迭代和优化。然而，现有的用户模拟器通常面临一些重大限制，包括用户偏好建模的透明度不足和对模拟准确性的评估能力有限。在本文中，我们引入了一个基于大语言模型（LLM）的用户模拟器，以显式的方式模拟用户的项目互动，从而提高基于强化学习的推荐系统训练的效率和效果。具体而言，我们识别了用户的显式偏好逻辑，利用大语言模型分析项目特征并提炼用户情绪，并设计了一个逻辑模型来模仿真实的人类互动。通过整合统计模型，我们进一步增强了模拟的可靠性，提出了一种综合逻辑和统计洞见的集成模型，用于用户互动的模拟。凭借大语言模型的广泛知识和语义生成能力，我们的用户模拟器忠实模拟用户行为和偏好，提供高质量的训练数据以丰富推荐算法的训练。我们通过对五个数据集进行量化和定性的实验，验证了模拟器在各种推荐场景中的有效性与稳定性。 

---
# Multifaceted User Modeling in Recommendation: A Federated Foundation Models Approach 

**Title (ZH)**: 多方面用户建模在推荐中的应用：联邦基础模型方法 

**Authors**: Chunxu Zhang, Guodong Long, Hongkuan Guo, Zhaojie Liu, Guorui Zhou, Zijian Zhang, Yang Liu, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16969)  

**Abstract**: Multifaceted user modeling aims to uncover fine-grained patterns and learn representations from user data, revealing their diverse interests and characteristics, such as profile, preference, and personality. Recent studies on foundation model-based recommendation have emphasized the Transformer architecture's remarkable ability to capture complex, non-linear user-item interaction relationships. This paper aims to advance foundation model-based recommendersystems by introducing enhancements to multifaceted user modeling capabilities. We propose a novel Transformer layer designed specifically for recommendation, using the self-attention mechanism to capture sequential user-item interaction patterns. Specifically, we design a group gating network to identify user groups, enabling hierarchical discovery across different layers, thereby capturing the multifaceted nature of user interests through multiple Transformer layers. Furthermore, to broaden the data scope and further enhance multifaceted user modeling, we extend the framework to a federated setting, enabling the use of private datasets while ensuring privacy. Experimental validations on benchmark datasets demonstrate the superior performance of our proposed method. Code is available. 

**Abstract (ZH)**: 多方面用户建模旨在从用户数据中揭示精细的模式并学习其表示，以揭示多样化的兴趣和特征，如用户画像、偏好和个人特质。基于基础模型的推荐系统研究最近强调了Transformer架构在捕捉复杂非线性用户-项目交互关系方面的显著能力。本文旨在通过增强多方面用户建模能力来推进基于基础模型的推荐系统。我们提出了一种专门为推荐设计的新颖Transformer层，利用自注意力机制捕捉序列化的用户-项目交互模式。具体来说，我们设计了一个分组门控网络来识别用户群体，从而在不同层次上实现分层发现，通过多个Transformer层捕捉用户兴趣的多面性。为进一步拓宽数据范围并增强多方面用户建模，我们还将框架扩展到联邦学习设置中，能够在确保隐私的同时使用私有数据集。基准数据集上的实验验证表明，我们提出的方法具有优越的性能。代码已开源。 

---
# Towards a Unified Paradigm: Integrating Recommendation Systems as a New Language in Large Models 

**Title (ZH)**: 向着统一范式的迈进：将推荐系统融入大型模型作为新型语言 

**Authors**: Kai Zheng, Qingfeng Sun, Can Xu, Peng Yu, Qingwei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2412.16933)  

**Abstract**: This paper explores the use of Large Language Models (LLMs) for sequential recommendation, which predicts users' future interactions based on their past behavior. We introduce a new concept, "Integrating Recommendation Systems as a New Language in Large Models" (RSLLM), which combines the strengths of traditional recommenders and LLMs. RSLLM uses a unique prompting method that combines ID-based item embeddings from conventional recommendation models with textual item features. It treats users' sequential behaviors as a distinct language and aligns the ID embeddings with the LLM's input space using a projector. We also propose a two-stage LLM fine-tuning framework that refines a pretrained LLM using a combination of two contrastive losses and a language modeling loss. The LLM is first fine-tuned using text-only prompts, followed by target domain fine-tuning with unified prompts. This trains the model to incorporate behavioral knowledge from the traditional sequential recommender into the LLM. Our empirical results validate the effectiveness of our proposed framework. 

**Abstract (ZH)**: 本文探讨了使用大型语言模型（LLM）进行序列推荐的方法，该方法基于用户的历史行为预测其未来的交互。我们提出了一种新的概念，“将推荐系统作为一种新的大型模型语言”（RSLLM，即Recommendation Systems as a Language in Large Models），该概念结合了传统推荐系统和LLM的优点。RSLLM采用了一种独特的提示方法，将传统推荐模型中的基于ID的项目嵌入与文本项目特征相結合。用户的行为序列被视作一种独特的语言，并通过投影器将ID嵌入与LLM的输入空间对齐。此外，我们还提出了一种双阶段的LLM微调框架，该框架使用两种对比损失和语言模型损失的组合对预训练的LLM进行微调。首先使用文本提示对LLM进行微调，随后使用统一的提示对特定领域进行微调。这一过程训练模型将传统序列推荐器中的行为知识整合到LLM中。我们的实验证明了所提出框架的有效性。 

---
# Enhancing Supply Chain Transparency in Emerging Economies Using Online Contents and LLMs 

**Title (ZH)**: 利用在线内容和大型语言模型增强新兴经济体的供应链透明度 

**Authors**: Bohan Jin, Qianyou Sun, Lihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16922)  

**Abstract**: In the current global economy, supply chain transparency plays a pivotal role in ensuring this security by enabling companies to monitor supplier performance and fostering accountability and responsibility. Despite the advancements in supply chain relationship datasets like Bloomberg and FactSet, supply chain transparency remains a significant challenge in emerging economies due to issues such as information asymmetry and institutional gaps in regulation. This study proposes a novel approach to enhance supply chain transparency in emerging economies by leveraging online content and large language models (LLMs). We develop a Supply Chain Knowledge Graph Mining System that integrates advanced LLMs with web crawler technology to automatically collect and analyze supply chain information. The system's effectiveness is validated through a case study focusing on the semiconductor supply chain, a domain that has recently gained significant attention due to supply chain risks. Our results demonstrate that the proposed system provides greater applicability for emerging economies, such as mainland China, complementing the data gaps in existing datasets. However, challenges including the accurate estimation of monetary and material flows, the handling of time series data, synonyms disambiguation, and mitigating biases from online contents still remains. Future research should focus on addressing these issues to further enhance the system's capabilities and broaden its application to other emerging economies and industries. 

**Abstract (ZH)**: 在全球经济中，供应链透明度在确保安全方面发挥着关键作用，它使企业能够监控供应商表现并促进问责制和责任感。尽管 Bloomberg 和 FactSet 等供应链关系数据集取得了进步，但由于信息不对称和监管机制的缺口等问题，供应链透明度在新兴经济体中仍是一个重大挑战。本研究提出了一种利用在线内容和大规模语言模型（LLM）的新方法，以增强新兴经济体中的供应链透明度。我们开发了一种供应链知识图谱分析系统，该系统整合了先进的 LLM 和网络爬虫技术，以自动收集和分析供应链信息。通过专注于半导体供应链的案例研究来验证该系统的效果，半导体供应链因供应链风险而近期引起了广泛关注。研究结果表明，所提出的系统对如中国大陆等新兴经济体具有更广泛的适用性，可以补充现有数据集的数据缺口。然而，包括货币和物资流动的准确估算、时间序列数据处理、同义词消歧和减轻在线内容偏见等问题仍然存在挑战。未来的研究应关注这些问题，以进一步增强系统的功能，将其应用扩展到其他新兴经济体和行业。 

---
# Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks 

**Title (ZH)**: 面向更稳健的检索增强生成：评估对抗中毒攻击下的RAG性能 

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2412.16708)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into retrieval databases can mislead the model into generating factually incorrect outputs. In this paper, we investigate both the retrieval and the generation components of RAG systems to understand how to enhance their robustness against such attacks. From the retrieval perspective, we analyze why and how the adversarial contexts are retrieved and assess how the quality of the retrieved passages impacts downstream generation. From a generation perspective, we evaluate whether LLMs' advanced critical thinking and internal knowledge capabilities can be leveraged to mitigate the impact of adversarial contexts, i.e., using skeptical prompting as a self-defense mechanism. Our experiments and findings provide actionable insights into designing safer and more resilient retrieval-augmented frameworks, paving the way for their reliable deployment in real-world applications. 

**Abstract (ZH)**: 检索增强生成（RAG）系统作为一种减轻大规模语言模型（LLM）幻觉并提高其在知识密集型领域性能的有前景的解决方案而崭露头角。然而，这些系统容易受到对抗性污染攻击的影响，在这些攻击中，恶意段落被注入检索数据库，可能导致模型生成事实不正确的输出。在本文中，我们研究了RAG系统的检索和生成组件，以了解如何增强其对这些攻击的鲁棒性。从检索的角度来看，我们分析了为什么以及如何检索到对抗性上下文，并评估了检索到段落质量对后续生成的影响。从生成的角度来看，我们评估了是否可以利用LLM的高级批判性思维和内部知识能力来减轻对抗性上下文的影响，即使用怀疑性提示作为一种自我防御机制。我们的实验和发现为设计更安全、更稳健的检索增强框架提供了可操作的见解，铺平了其实用部署在实际应用中的道路。 

---
# AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles 

**Title (ZH)**: 阿尔茨海默病RAG：面向PubMed文章的多模态检索增强生成 

**Authors**: Aritra Kumar Lahiri, Qinmin Vivian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16701)  

**Abstract**: Recent advancements in generative AI have flourished the development of highly adept Large Language Models (LLMs) that integrate diverse data types to empower decision-making. Among these, Multimodal Retrieval-Augmented Generation (RAG) applications are promising for their capability to combine the strengths of information retrieval and generative models, enhancing their utility across various domains, including biomedical research. This paper introduces AlzheimerRAG, a Multimodal RAG pipeline tool for biomedical research use cases, primarily focusing on Alzheimer's disease from PubMed articles. Our pipeline incorporates multimodal fusion techniques to integrate textual and visual data processing by efficiently indexing and accessing vast amounts of biomedical literature. Preliminary experimental results against benchmarks, such as BioASQ and PubMedQA, have returned improved results in information retrieval and synthesis of domain-specific information. We also demonstrate a case study with our RAG pipeline across different Alzheimer's clinical scenarios. We infer that AlzheimerRAG can generate responses with accuracy non-inferior to humans and with low rates of hallucination. Overall, a reduction in cognitive task load is observed, which allows researchers to gain multimodal insights, improving understanding and treatment of Alzheimer's disease. 

**Abstract (ZH)**: 近年来生成型人工智能的快速发展催生了高度专业的大型语言模型（LLMs），这些模型能够整合多种数据类型以助力决策。其中，多模态检索增强生成（Multimodal Retrieval-Augmented Generation, RAG）的应用尤为有前景，因为它们能够结合信息检索和生成模型的优势，从而在生物医学研究等多个领域提升其应用价值。本文介绍了一种名为AlzheimerRAG的多模态RAG管道工具，主要针对PubMed文章中的阿尔茨海默病研究案例。我们的管道工具采用多模态融合技术，通过高效地索引和访问大量的生物医学文献，整合文本和视觉数据处理。初步实验结果表明，与BioASQ和PubMedQA等基准进行对比时，信息检索和领域特定信息综合方面取得了改进的结果。我们还展示了RAG管道在不同阿尔茨海默病临床场景中的案例研究。通过这些研究，我们推断AlzheimerRAG可以生成与人类具有相当准确度且幻觉率较低的回答。总体而言，认知任务负担的减少使得研究人员能够获得多模态洞察，从而提高对阿尔茨海默病的理解和治疗方法的研发。 

---
# Large Language Model Can Be a Foundation for Hidden Rationale-Based Retrieval 

**Title (ZH)**: 大型语言模型可以作为基于隐含推理的检索的基础 

**Authors**: Luo Ji, Feixiang Guo, Teng Chen, Qingqing Gu, Xiaoyu Wang, Ningyuan Xi, Yihong Wang, Peng Yu, Yue Zhao, Hongyang Lei, Zhonglin Jiang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16615)  

**Abstract**: Despite the recent advancement in Retrieval-Augmented Generation (RAG) systems, most retrieval methodologies are often developed for factual retrieval, which assumes query and positive documents are semantically similar. In this paper, we instead propose and study a more challenging type of retrieval task, called hidden rationale retrieval, in which query and document are not similar but can be inferred by reasoning chains, logic relationships, or empirical experiences. To address such problems, an instruction-tuned Large language model (LLM) with a cross-encoder architecture could be a reasonable choice. To further strengthen pioneering LLM-based retrievers, we design a special instruction that transforms the retrieval task into a generative task by prompting LLM to answer a binary-choice question. The model can be fine-tuned with direct preference optimization (DPO). The framework is also optimized for computational efficiency with no performance degradation. We name this retrieval framework by RaHoRe and verify its zero-shot and fine-tuned performance superiority on Emotional Support Conversation (ESC), compared with previous retrieval works. Our study suggests the potential to employ LLM as a foundation for a wider scope of retrieval tasks. Our codes, models, and datasets are available on this https URL. 

**Abstract (ZH)**: 尽管近年来检索增强生成（RAG）系统取得了进展，大多数检索方法仍侧重于事实检索，假设查询和正文档在语义上相似。在本文中，我们提出并研究了一种更为具有挑战性的检索任务类型，称为隐藏理性检索，其中查询和文档不相似，但可以通过推理链、逻辑关系或经验推断出来。为了解决这些问题，带有交叉编码器架构的指令调节大规模语言模型（LLM）是一种合理的选择。为了进一步加强基于LLM的检索器，我们设计了一种特别的指令，通过提示LLM回答一个二元选择问题，将检索任务转化为生成任务。该模型可以使用直接偏好优化（DPO）进行微调。该框架在计算效率上进行了优化，且不会降低性能。我们将这种检索框架命名为RaHoRe，并在情感支持对话（ESC）任务上验证了其零样本和微调后的性能优越性，与之前的检索工作相比具有优势。我们的研究表明，可以将LLM作为更广泛的检索任务的基础。我们的代码、模型和数据集可在以下链接获得：https://... 

---
# Improving FIM Code Completions via Context & Curriculum Based Learning 

**Title (ZH)**: 基于上下文和课程学习提高FIM代码补全效果 

**Authors**: Hitesh Sagtani, Rishabh Mehrotra, Beyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16589)  

**Abstract**: Fill-in-the-Middle (FIM) models play a vital role in code completion tasks, leveraging both prefix and suffix context to provide more accurate and contextually relevant suggestions. This paper presents approaches to improve FIM code completion while addressing the challenge of maintaining low latency for real-time coding assistance. We enhance FIM code completion by incorporating context and curriculum examples in the training process. We identify patterns where completion suggestions fail more frequently, revealing complexities that smaller language models struggle with. To address these challenges, we develop a curriculum dataset by extracting hard-to-complete patterns from code repositories and generate context examples using semantic and static analysis tools (e.g. TSC compiler). We fine-tune various sized models, including StarCoder and DeepSeek, on this enhanced dataset. Our evaluation encompasses three key dimensions: the Santa Coder FIM task, the Amazon CCEval benchmark, and a new Multi-Line Infilling evaluation benchmark derived from SWE-bench. Comprehensive ablation studies across multiple model sizes reveal that while all fine-tuned models show improvements, the performance gains are more pronounced for smaller parameter models and incorporating difficult-to-complete examples, as part of curriculum learning, improves the code completion performance. This finding is particularly significant given the latency constraints of code completion tasks. While larger models like GPT and Claude perform well in multi-line completions but are prohibitively challenging to use given high latency, and our fine-tuned models achieve a balance between performance and latency. Finally, we validate our approach through online A/B testing, demonstrating tangible improvements in Completion Acceptance Rate (CAR) and Completion Persistence Rate (CPR), with zero latency impact. 

**Abstract (ZH)**: 填中（Fill-in-the-Middle，FIM）模型在代码补全任务中发挥着至关重要的作用，它们利用前缀和后缀语境来提供更准确和上下文相关性更强的建议。本文介绍了提高FIM代码补全性能的方法，同时解决了实时编码辅助中保持低延迟的挑战。我们通过在训练过程中引入语境和递增学习示例来增强FIM代码补全。我们识别出补全建议失败更频繁的模式，揭示了小型语言模型难以处理的复杂性。为了解决这些挑战，我们开发了一个递增学习数据集，通过从代码仓库中提取难以补全的模式，并使用语义和静态分析工具（例如TSC编译器）生成语境示例。我们分别在StarCoder和DeepSeek等不同规模模型上对这个增强的数据集进行了微调。我们的评估涵盖了三个关键维度：Santa Coder FIM任务、Amazon CCEval基准及来自SWE-bench的新多行填充评估基准。横跨多个模型规模的详细消融研究发现，虽然所有微调后的模型都显示出改进，但参数较少的模型的性能增益更为显著，将难以补全的示例作为递增学习的一部分进行微调，能够显著提高代码补全性能。这一发现特别重要，因为代码补全任务面临延迟限制。虽然大型模型如GPT和Claude在多行补全方面表现良好，但由于高延迟，它们的使用非常具有挑战性，而我们的微调模型则能够在性能和延迟之间达到平衡。最后，我们通过在线A/B测试验证了该方法，结果表明在无延迟影响的情况下，代码补全接受率（CAR）和代码补全持续率（CPR）均得到了实际的提升。 

---
# EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models 

**Title (ZH)**: EMPRA：针对神经排序模型的扰动嵌入排名攻击 

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2412.16382)  

**Abstract**: Recent research has shown that neural information retrieval techniques may be susceptible to adversarial attacks. Adversarial attacks seek to manipulate the ranking of documents, with the intention of exposing users to targeted content. In this paper, we introduce the Embedding Perturbation Rank Attack (EMPRA) method, a novel approach designed to perform adversarial attacks on black-box Neural Ranking Models (NRMs). EMPRA manipulates sentence-level embeddings, guiding them towards pertinent context related to the query while preserving semantic integrity. This process generates adversarial texts that seamlessly integrate with the original content and remain imperceptible to humans. Our extensive evaluation conducted on the widely-used MS MARCO V1 passage collection demonstrate the effectiveness of EMPRA against a wide range of state-of-the-art baselines in promoting a specific set of target documents within a given ranked results. Specifically, EMPRA successfully achieves a re-ranking of almost 96% of target documents originally ranked between 51-100 to rank within the top 10. Furthermore, EMPRA does not depend on surrogate models for adversarial text generation, enhancing its robustness against different NRMs in realistic settings. 

**Abstract (ZH)**: 近年来的研究表明，神经信息检索技术可能容易受到对抗攻击的影响。对抗攻击旨在操控文档的排名，以使用户暴露于特定内容。本文我们介绍了一种新颖的方法——嵌入扰动排名攻击（Embedding Perturbation Rank Attack, EMPRA），该方法旨在对黑盒神经排名模型（Neural Ranking Models, NRMs）进行对抗攻击。EMPRA通过操纵句子级别的嵌入，使其导向与查询相关的关键语境，同时保持语义完整性。这一过程生成的对抗文本能够无缝融入原始内容，且对人类不可感知。我们在广泛使用于评估的MS MARCO V1段落集合上进行了全面的评估，表明EMPRA在促进特定目标文档排名方面对多种最新的基线算法具有显著效果。具体而言，EMPRA成功地重新排列了原本排名在51-100位的几乎所有目标文档，使其排名进入前10位。此外，EMPRA不依赖于代理模型来进行对抗文本生成，这增强了其在实际应用场景中对抗不同NRMs的鲁棒性。 

---
# Minimum Weighted Feedback Arc Sets for Ranking from Pairwise Comparisons 

**Title (ZH)**: 最小加权反馈弧集：基于成对比较的排名问题 

**Authors**: Soroush Vahidi, Ioannis Koutis  

**Link**: [PDF](https://arxiv.org/pdf/2412.16181)  

**Abstract**: The Minimum Weighted Feedback Arc Set (MWFAS) problem is fundamentally connected to the Ranking Problem -- the task of deriving global rankings from pairwise comparisons. Recent work [He et al. ICML2022] has advanced the state-of-the-art for the Ranking Problem using learning-based methods, improving upon multiple previous approaches. However, the connection to MWFAS remains underexplored. This paper investigates this relationship and presents efficient combinatorial algorithms for solving MWFAS, thus addressing the Ranking Problem. Our experimental results demonstrate that these simple, learning-free algorithms not only significantly outperform learning-based methods in terms of speed but also generally achieve superior ranking accuracy. 

**Abstract (ZH)**: 最小子图反馈弧集（Minimum Weighted Feedback Arc Set，MWFAS）问题与排名问题（Ranking Problem）从根本上说具有密切联系，即从成对比较中推导出全局排名的任务。最近的工作 [He等，ICML2022] 使用基于学习的方法推进了排名问题的状态前沿，改善了多个先前的方法。然而，MWFAS 与排名问题之间的联系仍然没有得到充分探索。本文研究了这种关系，并提出了求解 MWFAS 的高效组合算法，从而解决了排名问题。我们的实验结果表明，这些简单且不依赖学习的算法不仅在速度上远超基于学习的方法，而且通常能达到更高的排名准确性。 

---
# RAGONITE: Iterative Retrieval on Induced Databases and Verbalized RDF for Conversational QA over KGs with RAG 

**Title (ZH)**: RAGONITE：基于诱导数据库和口头化RDF的RAG对知识图谱进行对话式QA的迭代检索方法 

**Authors**: Rishiraj Saha Roy, Chris Hinze, Joel Schlotthauer, Farzad Naderi, Viktor Hangya, Andreas Foltyn, Luzian Hahn, Fabian Kuech  

**Link**: [PDF](https://arxiv.org/pdf/2412.17690)  

**Abstract**: Conversational question answering (ConvQA) is a convenient means of searching over RDF knowledge graphs (KGs), where a prevalent approach is to translate natural language questions to SPARQL queries. However, SPARQL has certain shortcomings: (i) it is brittle for complex intents and conversational questions, and (ii) it is not suitable for more abstract needs. Instead, we propose a novel two-pronged system where we fuse: (i) SQL-query results over a database automatically derived from the KG, and (ii) text-search results over verbalizations of KG facts. Our pipeline supports iterative retrieval: when the results of any branch are found to be unsatisfactory, the system can automatically opt for further rounds. We put everything together in a retrieval augmented generation (RAG) setup, where an LLM generates a coherent response from accumulated search results. We demonstrate the superiority of our proposed system over several baselines on a knowledge graph of BMW automobiles. 

**Abstract (ZH)**: 对话式问答（ConvQA）是搜索RDF知识图谱（KGs）的一种便捷方法，其中一种流行的途径是将自然语言问题转换为SPARQL查询。然而，SPARQL存在一定的局限性：（i）其在处理复杂意图和对话式问题时较为脆弱，（ii）不适用于更抽象的需求。因此，我们提出了一种新颖的两阶段系统，该系统结合了以下两个方面：（i）来自知识图谱自动推导出的数据库的SQL查询结果，和（ii）基于知识图谱事实表达的文本搜索结果。我们的流水线支持迭代检索：当任何分支的结果不满意时，系统可以自动选择进一步的检索轮次。我们将这一切整合到检索增强生成（RAG）框架中，在该框架中，预训练语言模型（LLM）从累积的搜索结果中生成连贯的回复。我们通过在宝马汽车知识图谱上与多个基线系统的对比实验证明了我们提出的系统具有优越性。 

---
# Comparative Analysis of Document-Level Embedding Methods for Similarity Scoring on Shakespeare Sonnets and Taylor Swift Lyrics 

**Title (ZH)**: 莎士比亚十四行诗与泰勒·斯威夫特歌词基于文档级嵌入方法的相似性评分比较分析 

**Authors**: Klara Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2412.17552)  

**Abstract**: This study evaluates the performance of TF-IDF weighting, averaged Word2Vec embeddings, and BERT embeddings for document similarity scoring across two contrasting textual domains. By analysing cosine similarity scores, the methods' strengths and limitations are highlighted. The findings underscore TF-IDF's reliance on lexical overlap and Word2Vec's superior semantic generalisation, particularly in cross-domain comparisons. BERT demonstrates lower performance in challenging domains, likely due to insufficient domainspecific fine-tuning. 

**Abstract (ZH)**: 本研究评估了TF-IDF加权、平均Word2Vec嵌入和BERT嵌入在两个对比文本领域中文档相似性评分的表现。通过分析余弦相似度得分，展示了这些方法的优势和局限性。研究结果强调了TF-IDF对词汇重叠的依赖性以及Word2Vec在跨领域比较中的优越语义泛化能力。BERT在具有挑战性的领域中的表现较低，这可能归因于其不足的专业领域微调。 

---
# LegalAgentBench: Evaluating LLM Agents in Legal Domain 

**Title (ZH)**: LegalAgentBench：评估法律领域中的LLM代理 

**Authors**: Haitao Li, Junjie Chen, Jingli Yang, Qingyao Ai, Wei Jia, Youfeng Liu, Kai Lin, Yueyue Wu, Guozhi Yuan, Yiran Hu, Wuyue Wang, Yiqun Liu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17259)  

**Abstract**: With the increasing intelligence and autonomy of LLM agents, their potential applications in the legal domain are becoming increasingly apparent. However, existing general-domain benchmarks cannot fully capture the complexity and subtle nuances of real-world judicial cognition and decision-making. Therefore, we propose LegalAgentBench, a comprehensive benchmark specifically designed to evaluate LLM Agents in the Chinese legal domain. LegalAgentBench includes 17 corpora from real-world legal scenarios and provides 37 tools for interacting with external knowledge. We designed a scalable task construction framework and carefully annotated 300 tasks. These tasks span various types, including multi-hop reasoning and writing, and range across different difficulty levels, effectively reflecting the complexity of real-world legal scenarios. Moreover, beyond evaluating final success, LegalAgentBench incorporates keyword analysis during intermediate processes to calculate progress rates, enabling more fine-grained evaluation. We evaluated eight popular LLMs, highlighting the strengths, limitations, and potential areas for improvement of existing models and methods. LegalAgentBench sets a new benchmark for the practical application of LLMs in the legal domain, with its code and data available at \url{this https URL}. 

**Abstract (ZH)**: 随着大型语言模型（LLM）代理的智能化和自主性不断提高，它们在法律领域的潜在应用越来越明显。然而，现有的通用领域基准并不能充分捕捉现实世界司法认知和决策的复杂性和细微差别。因此，我们提出了LegalAgentBench，这是一种专门针对中国法律领域设计的全面基准，用于评估LLM代理的表现。LegalAgentBench包含了17个来自实际法律场景的数据集，并提供了37种与外部知识交互的工具。我们设计了一个可扩展的任务构建框架，并仔细标注了300个任务。这些任务涵盖了多种类型，包括多跳推理和写作等任务，并且难度跨度较大，有效反映了实际法律场景的复杂性。此外，LegalAgentBench不仅评估最终成功，还在中间过程中融入关键词分析，计算进步率，从而提供更细致的评估。我们评估了八种流行的LLM，突显了现有模型和方法的优势、局限性以及改进空间。LegalAgentBench为LLM在法律领域的实际应用设立了新的基准，其代码和数据可在网页 \url{this https URL} 获取。 

---
# Unity is Strength: Unifying Convolutional and Transformeral Features for Better Person Re-Identification 

**Title (ZH)**: 团结即力量：统一卷积和Transformer特征以提高人员再识别性能 

**Authors**: Yuhao Wang, Pingping Zhang, Xuehu Liu, Zhengzheng Tu, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17239)  

**Abstract**: Person Re-identification (ReID) aims to retrieve the specific person across non-overlapping cameras, which greatly helps intelligent transportation systems. As we all know, Convolutional Neural Networks (CNNs) and Transformers have the unique strengths to extract local and global features, respectively. Considering this fact, we focus on the mutual fusion between them to learn more comprehensive representations for persons. In particular, we utilize the complementary integration of deep features from different model structures. We propose a novel fusion framework called FusionReID to unify the strengths of CNNs and Transformers for image-based person ReID. More specifically, we first deploy a Dual-branch Feature Extraction (DFE) to extract features through CNNs and Transformers from a single image. Moreover, we design a novel Dual-attention Mutual Fusion (DMF) to achieve sufficient feature fusions. The DMF comprises Local Refinement Units (LRU) and Heterogenous Transmission Modules (HTM). LRU utilizes depth-separable convolutions to align deep features in channel dimensions and spatial sizes. HTM consists of a Shared Encoding Unit (SEU) and two Mutual Fusion Units (MFU). Through the continuous stacking of HTM, deep features after LRU are repeatedly utilized to generate more discriminative features. Extensive experiments on three public ReID benchmarks demonstrate that our method can attain superior performances than most state-of-the-arts. The source code is available at this https URL. 

**Abstract (ZH)**: 人体重新识别（ReID）的目标是在非重叠摄像机之间检索特定的人，这在智能交通系统中起到了极大的帮助作用。众所周知，卷积神经网络（CNNs）和变换器（Transformers）分别在提取局部和全局特征方面具有独特优势。考虑到这一点，我们关注它们之间的相互融合，以学习更全面的人体表示。尤其地，我们利用了来自不同模型结构的深层特征的互补整合。我们提出了一种新颖的融合框架——FusionReID，以结合CNNs和Transformers的优势，用于基于图像的人体ReID。具体而言，我们首先部署了双重分支特征提取（DFE），从单张图像中通过CNNs和Transformers提取特征。此外，我们设计了一种新颖的双重注意力互融（DMF）以实现充分的特征融合。DMF 包括局部完善单元（LRU）和异构传输模块（HTM）。LRU 利用深度可分离卷积来在通道维度和空间大小上对齐深层特征。HTM 包括共享编码单元（SEU）和两个互融单元（MFU）。通过HTM的连续堆叠，LRU之后的深层特征被反复利用以生成更具判别性的特征。在三个公开的ReID基准上的广泛实验表明，我们的方法在大多数最新的方法中表现优异。源代码可在以下链接获取：[此处链接]。 

---
# Enhancing Item Tokenization for Generative Recommendation through Self-Improvement 

**Title (ZH)**: 通过自我提升提高项目token化在生成性推荐中的效果 

**Authors**: Runjin Chen, Mingxuan Ju, Ngoc Bui, Dimosthenis Antypas, Stanley Cai, Xiaopeng Wu, Leonardo Neves, Zhangyang Wang, Neil Shah, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.17171)  

**Abstract**: Generative recommendation systems, driven by large language models (LLMs), present an innovative approach to predicting user preferences by modeling items as token sequences and generating recommendations in a generative manner. A critical challenge in this approach is the effective tokenization of items, ensuring that they are represented in a form compatible with LLMs. Current item tokenization methods include using text descriptions, numerical strings, or sequences of discrete tokens. While text-based representations integrate seamlessly with LLM tokenization, they are often too lengthy, leading to inefficiencies and complicating accurate generation. Numerical strings, while concise, lack semantic depth and fail to capture meaningful item relationships. Tokenizing items as sequences of newly defined tokens has gained traction, but it often requires external models or algorithms for token assignment. These external processes may not align with the LLM's internal pretrained tokenization schema, leading to inconsistencies and reduced model performance. To address these limitations, we propose a self-improving item tokenization method that allows the LLM to refine its own item tokenizations during training process. Our approach starts with item tokenizations generated by any external model and periodically adjusts these tokenizations based on the LLM's learned patterns. Such alignment process ensures consistency between the tokenization and the LLM's internal understanding of the items, leading to more accurate recommendations. Furthermore, our method is simple to implement and can be integrated as a plug-and-play enhancement into existing generative recommendation systems. Experimental results on multiple datasets and using various initial tokenization strategies demonstrate the effectiveness of our method, with an average improvement of 8\% in recommendation performance. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的生成型推荐系统通过将物品表示为标记序列并以生成的方式生成推荐，提供了一种创新的方法来预测用户偏好。这种方法的关键挑战是如何有效地对物品进行标记化，确保其以与LLMs兼容的形式表示。当前的物品标记化方法包括使用文本描述、数值字符串或离散标记序列。基于文本的表示与LLMs的标记化无缝集成，但通常太冗长，导致效率低下且使准确生成变得复杂。数值字符串虽然简洁，但缺乏语义深度，无法捕捉有意义的物品关系。将物品标记化为新定义的标记序列的方法得到了越来越多的关注，但通常需要外部模型或算法来分配标记。这些外部过程可能与LLMs的内部预训练标记化方案不一致，导致不一致性和模型性能下降。为了解决这些局限性，我们提出了一种自我提升的物品标记化方法，允许LLMs在训练过程中改进其自身的物品标记化。我们的方法从使用任何外部模型生成的物品标记开始，并根据LLMs学到的模式定期调整这些标记。这一对齐过程确保标记化与LLMs对物品内部理解的一致性，从而提高推荐准确性。此外，我们的方法易于实现，并可以作为即插即用功能集成到现有的生成型推荐系统中。在多项数据集上进行的实验和使用各种初始标记化策略的实验结果表明，我们的方法非常有效，平均推荐性能提高了8%。 

---
# GME: Improving Universal Multimodal Retrieval by Multimodal LLMs 

**Title (ZH)**: GME：通过多模态大语言模型提高通用多模态检索性能 

**Authors**: Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16855)  

**Abstract**: Universal Multimodal Retrieval (UMR) aims to enable search across various modalities using a unified model, where queries and candidates can consist of pure text, images, or a combination of both. Previous work has attempted to adopt multimodal large language models (MLLMs) to realize UMR using only text data. However, our preliminary experiments demonstrate that more diverse multimodal training data can further unlock the potential of MLLMs. Despite its effectiveness, the existing multimodal training data is highly imbalanced in terms of modality, which motivates us to develop a training data synthesis pipeline and construct a large-scale, high-quality fused-modal training dataset. Based on the synthetic training data, we develop the General Multimodal Embedder (GME), an MLLM-based dense retriever designed for UMR. Furthermore, we construct a comprehensive UMR Benchmark (UMRB) to evaluate the effectiveness of our approach. Experimental results show that our method achieves state-of-the-art performance among existing UMR methods. Last, we provide in-depth analyses of model scaling, training strategies, and perform ablation studies on both the model and synthetic data. 

**Abstract (ZH)**: 通用多模态检索（UMR）旨在使用统一模型在各种模态之间进行搜索，其中查询和候选项可以是纯文本、图像，或两者结合。以往的工作试图采用多模态大型语言模型（MLLM）仅使用文本数据来实现UMR。然而，我们的初步实验表明，更多的多样化多模态训练数据可以进一步发挥MLLM的潜力。尽管多模态训练数据的有效性已经得到验证，但现有的多模态训练数据在模态方面高度不平衡，这促使我们开发了一个训练数据合成管道，并构建了一个大规模、高质量的融合模态训练数据集。基于合成训练数据，我们开发了通用多模态嵌入器（GME），一个基于MLLM的密集检索器，旨在实现UMR。此外，我们构建了一个综合的UMR基准（UMRB），以评估我们方法的有效性。实验结果表明，我们的方法在现有的UMR方法中取得了最先进的性能。最后，我们对模型缩放、训练策略进行了深入分析，并在模型和合成数据上进行了消融研究。 

---
# DragonVerseQA: Open-Domain Long-Form Context-Aware Question-Answering 

**Title (ZH)**: DragonVerseQA：开放领域长文本上下文感知问答 

**Authors**: Aritra Kumar Lahiri, Qinmin Vivian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16694)  

**Abstract**: This paper proposes a novel approach to develop an open-domain and long-form Over-The-Top (OTT) Question-Answering (QA) dataset, DragonVerseQA, specifically oriented to the fantasy universe of "House of the Dragon" and "Game Of Thrones" TV series. Most existing QA datasets focus on short, fact-based answers sourced almost solely from Wikipedia articles, devoid of depth and contextual richness for sophisticated narrative understanding. We curate a dataset that combines full episode summaries sourced from HBO and fandom wiki websites, user reviews from sources like IMDb and Rotten Tomatoes, and high-quality, open-domain, legally admissible sources, and structured data from repositories like WikiData into one dataset. The dataset provides a multi-dimensional context, reflecting complex character dynamics and plot developments from these varied sources. That means, on equal footing, only after heavy data preprocessing and filtering methods will meaningful, non-spam unbiased reviews be available in this enriched dataset. The comprehensive insights are given through the long-form answers generated from this enriched context. This is what makes this valuable dataset for improving conversational AI, narrative analysis, sentiment analysis, summarization techniques, and relation extraction.
A comparative analysis with state-of-the-art QA datasets such as SQuAD 2.0, TriviaQA, and Natural Questions brings to light the unique advantages of our dataset in terms of contextual complexity and answer length. Detailed reviews add layers to audience sentiment and narrative interpretation, raising the bar for domain-specific QA with a new quality benchmark. Our work also allows a deeper understanding of entertainment-industry content and opens the door to more knowledgeable and creative AI-driven interactions within digital media environments. 

**Abstract (ZH)**: 本文提出了一种新颖的方法，以开发一个面向开放式和长篇对话型Over-The-Top (OTT) 问答（Question-Answering, QA）数据集——DragonVerseQA，该数据集特别适用于电视剧《龙之家族》和《权力的游戏》的幻想宇宙。现有的大多数QA数据集主要关注简短的、基于事实的答案，这些答案几乎源自维基百科文章，缺乏深度和情境丰富性，不利于复杂叙事的理解。我们精心策划了一个数据集，该数据集结合了来自HBO和粉丝维基网站的完整剧情概要，来自IMDb和烂番茄等网站的用户评论，以及高质量、开放域、合法可使用的来源，和来自WikiData等数据仓库的结构化数据。数据集提供了多维度的背景信息，体现了这些不同来源中的复杂角色动态和情节发展。这意味着，只有经过大量数据预处理和过滤方法之后，才会有有意义的、非垃圾的、公正的评论被纳入这个丰富化的数据集中。通过该丰富化的背景信息生成的长篇回答提供了全面的洞察力。这正是这个宝贵数据集用于提高对话式人工智能、叙事分析、情感分析、总结技术和关系抽取价值所在。

与SQuAD 2.0、TriviaQA和Natural Questions等最先进的QA数据集进行比较分析，突显了我们在上下文复杂性和答案长度方面所具备的独特优势。详细的评论增加了观众情感和叙事解释的层次性，为特定领域QA的高质量标准设定了新的门槛。我们的研究工作还加深了对娱乐行业内容的理解，并为在数字媒体环境中实现更富有知识性和创造力的人工智能交互打开了大门。 

---
# STKDRec: Spatial-Temporal Knowledge Distillation for Takeaway Recommendation 

**Title (ZH)**: STKDRec: 空间-时间知识蒸馏在外卖推荐中的应用 

**Authors**: Shuyuan Zhao, Wei Chen, Boyan Shi, Liyong Zhou, Shuohao Lin, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2412.16502)  

**Abstract**: The takeaway recommendation system is designed to recommend users' future takeaway purchases based on their historical purchase behaviors, thereby improving user satisfaction and increasing merchant sales. Existing methods focus on incorporating auxiliary information or leveraging knowledge graphs to alleviate the sparsity issue of user purchase sequence data. However, two main challenges limit the performance of these approaches: (1) how to capture dynamic user preferences on complex geospatial information and (2) how to efficiently integrate spatial-temporal knowledge from graphs and sequence data with low calculation costs. In this paper, we propose a novel spatial-temporal knowledge distillation for takeaway recommendation model (STKDRec) based on the two-stage training process. Specifically, during the first pre-training stage, a spatial-temporal knowledge graph (STKG) encoder is pre-trained to extract the high-order spatial-temporal and collaborative associations within the STKG. During the second STKD stage, a spatial-temporal Transformer is employed to comprehensively model dynamic user preferences on various types of fine-grained geospatial information from a sequence perspective. Furthermore, the STKD strategy is introduced to adaptively fuse the rich spatial-temporal knowledge from the pre-trained STKG encoder and the spatial-temporal transformer while reducing the cost of model training. Extensive experiments on three real-world datasets show that our STKDRec significantly outperforms the state-of-the-art baselines. Our code is available at:this https URL. 

**Abstract (ZH)**: 外卖推荐系统旨在根据用户的 histórico 购买行为，推荐其未来可能的外卖购买，从而提升用户满意度并增加商家销售额。现有方法主要集中在引入辅助信息或利用知识图谱来缓解用户购买序列数据的稀疏性问题。然而，这两种主要挑战限制了这些方法的效果：(1) 如何捕捉复杂地理空间信息下的动态用户偏好；(2) 如何以低成本高效地结合图结构和序列数据中的时空知识。在本文中，我们提出了一种基于两阶段训练过程的新时空知识蒸馏外卖推荐模型（STKDRec）。具体来说，在第一阶段预训练过程中，我们预先训练了一个时空知识图谱（STKG）编码器以提取STKG中的高阶时空和协作关联。在第二阶段STKD过程中，我们使用时空Transformer从序列角度全面建模用户在多种细粒度地理空间信息下的动态偏好。此外，我们引入了STKD策略以适应性地融合预训练的STKG编码器和时空Transformer中的丰富时空知识，同时减少模型训练的成本。在三个真实世界的数据集上的广泛实验表明，我们的STKDRec显著优于现有最先进的基线方法。我们的代码已发布于[this https URL]。 

---
# THeGCN: Temporal Heterophilic Graph Convolutional Network 

**Title (ZH)**: THeGCN：时空异质图卷积网络

解释：
- THeGCN: 这是原文的缩写，保持不变。
- Temporal: 时光的，此处指的是时间维度的信息。
- Heterophilic: 异质性的，即网络中的节点之间具有不同的性质或连接模式。
- Graph Convolutional Network: 图卷积网络，是一种在图结构数据上进行机器学习的技术。

翻译时，考虑到学术规范和术语的准确性和广泛认可性，将原文翻译为“时空异质图卷积网络”，确保既传达了原文的意思，又符合中文学术表达的习惯。 

**Authors**: Yuchen Yan, Yuzhong Chen, Huiyuan Chen, Xiaoting Li, Zhe Xu, Zhichen Zeng, Zhining Liu, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2412.16435)  

**Abstract**: Graph Neural Networks (GNNs) have exhibited remarkable efficacy in diverse graph learning tasks, particularly on static homophilic graphs. Recent attention has pivoted towards more intricate structures, encompassing (1) static heterophilic graphs encountering the edge heterophily issue in the spatial domain and (2) event-based continuous graphs in the temporal domain. State-of-the-art (SOTA) has been concurrently addressing these two lines of work but tends to overlook the presence of heterophily in the temporal domain, constituting the temporal heterophily issue. Furthermore, we highlight that the edge heterophily issue and the temporal heterophily issue often co-exist in event-based continuous graphs, giving rise to the temporal edge heterophily challenge. To tackle this challenge, this paper first introduces the temporal edge heterophily measurement. Subsequently, we propose the Temporal Heterophilic Graph Convolutional Network (THeGCN), an innovative model that incorporates the low/high-pass graph signal filtering technique to accurately capture both edge (spatial) heterophily and temporal heterophily. Specifically, the THeGCN model consists of two key components: a sampler and an aggregator. The sampler selects events relevant to a node at a given moment. Then, the aggregator executes message-passing, encoding temporal information, node attributes, and edge attributes into node embeddings. Extensive experiments conducted on 5 real-world datasets validate the efficacy of THeGCN. 

**Abstract (ZH)**: 图神经网络（GNNs）在各种图学习任务中展现了卓越的效果，尤其是在静态同质图上。最近的研究兴趣转向了更为复杂的结构，包括（1）遇到空间领域边异质性问题的静态异质图；（2）时间领域上的事件驱动连续图。当前最先进的方法（SOTA）同时研究了这两类问题，但在解决时间领域中的异质性问题方面往往不够重视，构成了时间异质性问题。此外，我们强调，在事件驱动连续图中，边异质性问题和时间异质性问题通常同时存在，从而产生了时间边异质性挑战。为了应对这一挑战，本文首先引入了时间边异质性测量。紧接着，我们提出了时间异质图卷积网络（THeGCN），这是一种结合低/高频图信号滤波技术的创新模型，能够准确捕捉边（空间）异质性和时间异质性。具体来说，THeGCN模型由两个关键组件组成：采样器和聚合器。采样器在给定时刻选择与节点相关的重要事件。然后，聚合器执行消息传递，将时间信息、节点属性和边属性编码到节点嵌入中。在使用5个真实数据集进行的广泛实验中，THeGCN的有效性得到了验证。 

---
# HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases 

**Title (ZH)**: HybGRAG：基于文本和关系知识库的混合检索增强生成方法 

**Authors**: Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N. Ioannidis, Huzefa Rangwala, Christos Faloutsos  

**Link**: [PDF](https://arxiv.org/pdf/2412.16311)  

**Abstract**: Given a semi-structured knowledge base (SKB), where text documents are interconnected by relations, how can we effectively retrieve relevant information to answer user questions? Retrieval-Augmented Generation (RAG) retrieves documents to assist large language models (LLMs) in question answering; while Graph RAG (GRAG) uses structured knowledge bases as its knowledge source. However, many questions require both textual and relational information from SKB - referred to as "hybrid" questions - which complicates the retrieval process and underscores the need for a hybrid retrieval method that leverages both information. In this paper, through our empirical analysis, we identify key insights that show why existing methods may struggle with hybrid question answering (HQA) over SKB. Based on these insights, we propose HybGRAG for HQA consisting of a retriever bank and a critic module, with the following advantages: (1) Agentic, it automatically refines the output by incorporating feedback from the critic module, (2) Adaptive, it solves hybrid questions requiring both textual and relational information with the retriever bank, (3) Interpretable, it justifies decision making with intuitive refinement path, and (4) Effective, it surpasses all baselines on HQA benchmarks. In experiments on the STaRK benchmark, HybGRAG achieves significant performance gains, with an average relative improvement in Hit@1 of 51%. 

**Abstract (ZH)**: 给定一个半结构化知识库（SKB），其中文本文档通过关系相互连接，我们如何有效地检索相关信息来回答用户问题？Retrieval-Augmented Generation (RAG) 通过检索文档来辅助大语言模型（LLMs）进行问答；而 Graph RAG (GRAG) 则使用结构化的知识库作为其知识来源。然而，许多问题需要从 SKB 中同时获取文本和关系信息，这类问题被称为“混合”问题，这增加了检索的复杂性，也突显了同时利用这两种信息的混合检索方法的重要性。本文通过对实证分析，我们发现了一些关键见解，这些见解揭示了现有方法在处理 SKB 上的混合问题解答（HQA）时可能遇到的挑战。基于这些见解，我们提出了 HybGRAG 方法以解决混合问题，该方法包括检索机构和批评模块，并具备以下优势：（1）主体作用，它能够自动改进输出，同时整合来自批评模块的反馈；（2）自适应，它能够利用检索机构解决既需要文本信息又需要关系信息的混合问题；（3）可解释性，它能够通过直观的改进路径解释决策过程；（4）有效，它在混合问题问答基准上超越了所有基线方法。在 STaRK 基准实验中，HybGRAG 实现了显著的性能提升，平均改进率达到 51% 的 Hit@1。 

---
# Learned Compression of Nonlinear Time Series With Random Access 

**Title (ZH)**: 具有随机访问功能的非线性时间序列的learned压缩 

**Authors**: Andrea Guerra, Giorgio Vinciguerra, Antonio Boffa, Paolo Ferragina  

**Link**: [PDF](https://arxiv.org/pdf/2412.16266)  

**Abstract**: Time series play a crucial role in many fields, including finance, healthcare, industry, and environmental monitoring. The storage and retrieval of time series can be challenging due to their unstoppable growth. In fact, these applications often sacrifice precious historical data to make room for new data.
General-purpose compressors can mitigate this problem with their good compression ratios, but they lack efficient random access on compressed data, thus preventing real-time analyses. Ad-hoc streaming solutions, instead, typically optimise only for compression and decompression speed, while giving up compression effectiveness and random access functionality. Furthermore, all these methods lack awareness of certain special regularities of time series, whose trends over time can often be described by some linear and nonlinear functions.
To address these issues, we introduce NeaTS, a randomly-accessible compression scheme that approximates the time series with a sequence of nonlinear functions of different kinds and shapes, carefully selected and placed by a partitioning algorithm to minimise the space. The approximation residuals are bounded, which allows storing them in little space and thus recovering the original data losslessly, or simply discarding them to obtain a lossy time series representation with maximum error guarantees.
Our experiments show that NeaTS improves the compression ratio of the state-of-the-art lossy compressors that use linear or nonlinear functions (or both) by up to 14%. Compared to lossless compressors, NeaTS emerges as the only approach to date providing, simultaneously, compression ratios close to or better than the best existing compressors, a much faster decompression speed, and orders of magnitude more efficient random access, thus enabling the storage and real-time analysis of massive and ever-growing amounts of (historical) time series data. 

**Abstract (ZH)**: 时间序列在金融、医疗、工业和环境监测等多个领域都起着至关重要的作用。由于时间序列数据的不断增长，其存储和检索变得具有挑战性。实际上，这些应用往往需要牺牲宝贵的 Historical 数据以腾出空间来存储新数据。

通用压缩器可以通过提供良好的压缩比来缓解这个问题，但它们缺乏对压缩数据的高效随机访问功能，从而阻碍了实时分析。而针对流数据的定制解决方案通常仅优化压缩和解压缩速度，牺牲了压缩效果和随机访问功能。此外，所有这些方法都无法充分利用时间序列数据中的某些特殊规律，这些规律通常可以用线性和非线性函数来描述。

为了解决这些问题，我们引入了 NeaTS，这是一种支持随机访问的压缩方案，它用不同类型和形状的非线性函数序列来近似时间序列。通过一个分划分算法，这些函数被精心选择和定位以最小化所需空间。近似残差被限制在一定范围内，这使得可以以很小的空间存储它们，并且可以无损地恢复原始数据，或者简单地丢弃它们以获得具有最大误差保证的时间序列表示。

我们的实验表明，NeaTS 可以将线性或非线性（或两者结合）的最先进的压缩器的压缩比提高多达 14%。与无损压缩相比，NeaTS 是迄今为止唯一一种同时提供接近或优于现有最优压缩器的压缩比、解压缩速度显著加快、随机访问效率提高了几个数量级的方法，从而能够存储和实时分析大量的不断增长的时间序列数据。 

---
# Zero-Shot Image Moderation in Google Ads with LLM-Assisted Textual Descriptions and Cross-modal Co-embeddings 

**Title (ZH)**: 在Google广告中使用LLM辅助文本描述和跨模态共嵌入的零样本图像审核 

**Authors**: Enming Luo, Wei Qiao, Katie Warren, Jingxiang Li, Eric Xiao, Krishna Viswanathan, Yuan Wang, Yintao Liu, Jimin Li, Ariel Fuxman  

**Link**: [PDF](https://arxiv.org/pdf/2412.16215)  

**Abstract**: We present a scalable and agile approach for ads image content moderation at Google, addressing the challenges of moderating massive volumes of ads with diverse content and evolving policies. The proposed method utilizes human-curated textual descriptions and cross-modal text-image co-embeddings to enable zero-shot classification of policy violating ads images, bypassing the need for extensive supervised training data and human labeling. By leveraging large language models (LLMs) and user expertise, the system generates and refines a comprehensive set of textual descriptions representing policy guidelines. During inference, co-embedding similarity between incoming images and the textual descriptions serves as a reliable signal for policy violation detection, enabling efficient and adaptable ads content moderation. Evaluation results demonstrate the efficacy of this framework in significantly boosting the detection of policy violating content. 

**Abstract (ZH)**: 我们提出了一种可扩展且灵活的方法，用于在谷歌上进行广告图像内容审核，以应对审核大量多样内容和不断变化的政策所带来的挑战。该方法利用了人类策展的文本描述和跨模态文本-图像共嵌入来实现对政策违规广告图像的零样本分类，从而避免了需要大量监督训练数据和人类标注。通过利用大规模语言模型（LLMs）和用户专业知识，系统生成并完善了一套全面的文本描述，代表了政策指导方针。在推理过程中，incoming图像与文本描述之间的共嵌入相似性成为可靠的行为策略违规检测信号，从而实现高效且灵活的广告内容审核。评估结果表明，该框架在显著提升政策违规内容检测能力方面具有有效性。 

---
