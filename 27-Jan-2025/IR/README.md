# Knowledge Graphs Construction from Criminal Court Appeals: Insights from the French Cassation Court 

**Title (ZH)**: 从法国最高法院刑事上诉案件构建知识图谱：见解与洞察 

**Authors**: Alexander V. Belikov, Sacha Raoult  

**Link**: [PDF](https://arxiv.org/pdf/2501.14579)  

**Abstract**: Despite growing interest, accurately and reliably representing unstructured data, such as court decisions, in a structured form, remains a challenge. Recent advancements in generative AI applied to language modeling enabled the transformation of text into knowledge graphs, unlocking new opportunities for analysis and modeling. This paper presents a framework for constructing knowledge graphs from appeals to the French Cassation Court. The framework includes a domain-specific ontology and a derived dataset, offering a foundation for structured legal data representation and analysis. 

**Abstract (ZH)**: 尽管人们对表示非结构化数据（如法院判决）的兴趣不断增加，但将其准确且可靠地转换为结构化形式仍然是一大挑战。最近应用于语言模型的生成式AI的发展使文本能够转化为知识图谱，从而开拓了分析和建模的新途径。本文提出了一种从法国最高法院上诉案件构建知识图谱的框架。该框架包括一个领域特定的本体论和一个衍生数据集，为法律数据的结构化表示和分析提供了基础。 

---
# On Correlating Factors for Domain Adaptation Performance 

**Title (ZH)**: 《影响领域适应性能的相关因素研究》 

**Authors**: Goksenin Yuksel, Jaap Kamps  

**Link**: [PDF](https://arxiv.org/pdf/2501.14466)  

**Abstract**: Dense retrievers have demonstrated significant potential for neural information retrieval; however, they lack robustness to domain shifts, limiting their efficacy in zero-shot settings across diverse domains. In this paper, we set out to analyze the possible factors that lead to successful domain adaptation of dense retrievers. We include domain similarity proxies between generated queries to test and source domains. Furthermore, we conduct a case study comparing two powerful domain adaptation techniques. We find that generated query type distribution is an important factor, and generating queries that share a similar domain to the test documents improves the performance of domain adaptation methods. This study further emphasizes the importance of domain-tailored generated queries. 

**Abstract (ZH)**: 密度检索器在神经信息检索中展示了显著的潜力；然而，它们在领域转移方面缺乏鲁棒性，限制了它们在各种领域中的零样本设置中的有效性。本文旨在分析导致密度检索器成功领域适应的可能因素。我们将领域相似性代理引入生成查询与测试集和源集之间，以进行测试。此外，我们还对比了两种强大的领域适应技术。研究发现，生成查询类型分布是一个重要因素，生成与测试文档领域相似的查询可以提高领域适应方法的有效性。该研究进一步突出了领域特定生成查询的重要性。 

---
# Interpretability Analysis of Domain Adapted Dense Retrievers 

**Title (ZH)**: 领域适应密集检索模型的可解释性分析 

**Authors**: Goksenin Yuksel, Jaap Kamps  

**Link**: [PDF](https://arxiv.org/pdf/2501.14459)  

**Abstract**: Dense retrievers have demonstrated significant potential for neural information retrieval; however, they exhibit a lack of robustness to domain shifts, thereby limiting their efficacy in zero-shot settings across diverse domains. Previous research has investigated unsupervised domain adaptation techniques to adapt dense retrievers to target domains. However, these studies have not focused on explainability analysis to understand how such adaptations alter the model's behavior. In this paper, we propose utilizing the integrated gradients framework to develop an interpretability method that provides both instance-based and ranking-based explanations for dense retrievers. To generate these explanations, we introduce a novel baseline that reveals both query and document attributions. This method is used to analyze the effects of domain adaptation on input attributions for query and document tokens across two datasets: the financial question answering dataset (FIQA) and the biomedical information retrieval dataset (TREC-COVID). Our visualizations reveal that domain-adapted models focus more on in-domain terminology compared to non-adapted models, exemplified by terms such as "hedge," "gold," "corona," and "disease." This research addresses how unsupervised domain adaptation techniques influence the behavior of dense retrievers when adapted to new domains. Additionally, we demonstrate that integrated gradients are a viable choice for explaining and analyzing the internal mechanisms of these opaque neural models. 

**Abstract (ZH)**: 密集检索器在神经信息检索中展现了显著的潜力；然而，它们对于领域转换的鲁棒性较差，这限制了它们在跨不同领域的零样本设置中的有效性。之前的研究所调查了无监督领域适应技术，以使密集检索器适应目标领域。然而，这些研究并未着重于解释性分析，以了解这些适应如何改变模型的行为。本文提出使用集成梯度框架来开发一种解释性方法，该方法为密集检索器提供基于实例和基于排序的解释。为了生成这些解释，我们引入了一种新颖的基线，该基线揭示了查询和文档的归因。这种方法被用于分析查询和文档词素在两种数据集上的输入归因效果：金融问答数据集（FIQA）和医学信息检索数据集（TREC-COVID）。我们的可视化结果表明，经过领域适应的模型比未适应的模型更关注领域内术语，例如“hedging”、“gold”、“corona”和“disease”等术语。本研究探讨了无监督领域适应技术如何影响密集检索器在新领域中的行为。此外，我们证明集成梯度是解释和分析这些不透明神经模型内部机制的一种可行选择。 

---
# Remining Hard Negatives for Generative Pseudo Labeled Domain Adaptation 

**Title (ZH)**: 利用剩余难负例进行生成式伪标记领域适应 

**Authors**: Goksenin Yuksel, David Rau, Jaap Kamps  

**Link**: [PDF](https://arxiv.org/pdf/2501.14434)  

**Abstract**: Dense retrievers have demonstrated significant potential for neural information retrieval; however, they exhibit a lack of robustness to domain shifts, thereby limiting their efficacy in zero-shot settings across diverse domains. A state-of-the-art domain adaptation technique is Generative Pseudo Labeling (GPL). GPL uses synthetic query generation and initially mined hard negatives to distill knowledge from cross-encoder to dense retrievers in the target domain. In this paper, we analyze the documents retrieved by the domain-adapted model and discover that these are more relevant to the target queries than those of the non-domain-adapted model. We then propose refreshing the hard-negative index during the knowledge distillation phase to mine better hard negatives. Our remining R-GPL approach boosts ranking performance in 13/14 BEIR datasets and 9/12 LoTTe datasets. Our contributions are (i) analyzing hard negatives returned by domain-adapted and non-domain-adapted models and (ii) applying the GPL training with and without hard-negative re-mining in LoTTE and BEIR datasets. 

**Abstract (ZH)**: 密集检索器在神经信息检索方面展现了显著的潜力；然而，它们在处理跨领域变化时表现出了鲁棒性不足的问题，从而限制了它们在不同领域零样本设置中的有效性。一种最先进的领域适应技术是生成式伪标签（Generative Pseudo Labeling, GPL）。GPL 通过合成查询生成和最初挖掘的硬负例，从跨编码器的知识中提炼出信息，传递给目标领域的密集检索器。在本文中，我们分析了领域适应模型检索的文档，并发现这些文档比未进行领域适应模型检索的相关性更高。我们随后提出，在知识提炼阶段刷新硬负例索引以挖掘更好的硬负例。我们的重挖掘 R-GPL 方法在 13/14 个 BEIR 数据集和 9/12 个 LoTTe 数据集上增强了排序性能。我们的贡献包括：(i) 分析领域适应模型和未进行领域适应模型返回的硬负例；(ii) 在 LoTTE 和 BEIR 数据集上应用包含和不包含硬负例重挖掘的 GPL 训练方法。 

---
# Handling Heterophily in Recommender Systems with Wavelet Hypergraph Diffusion 

**Title (ZH)**: 使用小波超图扩散处理推荐系统中的异质性 

**Authors**: Darnbi Sakong, Thanh Tam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14399)  

**Abstract**: Recommender systems are pivotal in delivering personalised user experiences across various domains. However, capturing the heterophily patterns and the multi-dimensional nature of user-item interactions poses significant challenges. To address this, we introduce FWHDNN (Fusion-based Wavelet Hypergraph Diffusion Neural Networks), an innovative framework aimed at advancing representation learning in hypergraph-based recommendation tasks. The model incorporates three key components: (1) a cross-difference relation encoder leveraging heterophily-aware hypergraph diffusion to adapt message-passing for diverse class labels, (2) a multi-level cluster-wise encoder employing wavelet transform-based hypergraph neural network layers to capture multi-scale topological relationships, and (3) an integrated multi-modal fusion mechanism that combines structural and textual information through intermediate and late-fusion strategies. Extensive experiments on real-world datasets demonstrate that FWHDNN surpasses state-of-the-art methods in accuracy, robustness, and scalability in capturing high-order interconnections between users and items. 

**Abstract (ZH)**: 推荐系统在各个领域提供个性化用户体验方面起着关键作用。然而，捕捉用户项交互的异质性和多维度性质构成了重大挑战。为了解决这一问题，我们提出了一种名为FWHDNN（基于融合的波动超图扩散神经网络）的创新框架，该框架旨在推进基于超图的推荐任务中的表示学习。该模型包含三个关键组件：（1）一个多差异关系编码器，利用异质性感知的超图扩散来适应消息传递，以适应不同的类别标签；（2）一个多尺度聚类编码器，采用基于小波变换的超图神经网络层来捕捉多尺度拓扑关系；（3）一种集成的多模态融合机制，通过中间融合和晚期融合策略结合结构和文本信息。在实际数据集上的广泛实验结果表明，FWHDNN在捕捉用户和项之间的高阶相互作用方面超越了最先进的方法，在准确度、鲁棒性和可扩展性方面表现更佳。 

---
# Chain-of-Retrieval Augmented Generation 

**Title (ZH)**: 链式检索增强生成 

**Authors**: Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.14342)  

**Abstract**: This paper introduces an approach for training o1-like RAG models that retrieve and reason over relevant information step by step before generating the final answer. Conventional RAG methods usually perform a single retrieval step before the generation process, which limits their effectiveness in addressing complex queries due to imperfect retrieval results. In contrast, our proposed method, CoRAG (Chain-of-Retrieval Augmented Generation), allows the model to dynamically reformulate the query based on the evolving state. To train CoRAG effectively, we utilize rejection sampling to automatically generate intermediate retrieval chains, thereby augmenting existing RAG datasets that only provide the correct final answer. At test time, we propose various decoding strategies to scale the model's test-time compute by controlling the length and number of sampled retrieval chains. Experimental results across multiple benchmarks validate the efficacy of CoRAG, particularly in multi-hop question answering tasks, where we observe more than 10 points improvement in EM score compared to strong baselines. On the KILT benchmark, CoRAG establishes a new state-of-the-art performance across a diverse range of knowledge-intensive tasks. Furthermore, we offer comprehensive analyses to understand the scaling behavior of CoRAG, laying the groundwork for future research aimed at developing factual and grounded foundation models. 

**Abstract (ZH)**: 本文介绍了一种训练o1-like RAG模型的方法，该方法在生成最终答案之前，逐步检索和推理相关信息。传统的RAG方法通常在生成过程前只执行一次检索步骤，这限制了它们在处理复杂查询时的有效性，尤其是由于检索结果不完美的原因。相比之下，我们提出的方法CoRAG（链式检索增强生成）允许模型根据最新的检索状态动态重新表述查询。为了有效训练CoRAG，我们利用拒绝采样自动生成中间的检索链，从而增强仅提供正确最终答案的现有RAG数据集。在测试阶段，我们提出多种解码策略，通过控制采样检索链的长度和数量来扩展模型的测试计算能力。跨多个基准实验结果验证了CoRAG的有效性，特别是在多跳问答任务中，我们观察到与强基线相比EM分数提高了超过10个百分点。在KILT基准上，CoRAG在多种知识密集型任务中建立了新的最佳性能。此外，我们提供了关于CoRAG扩展行为的全面分析，为未来旨在开发事实性和扎根基础模型的研究奠定了基础。 

---
# Multi-stage Large Language Model Pipelines Can Outperform GPT-4o in Relevance Assessment 

**Title (ZH)**: 多阶段大型语言模型管道可以在相关性评估中超越GPT-4o 

**Authors**: Julian A. Schnabel, Johanne R. Trippas, Falk Scholer, Danula Hettiachchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.14296)  

**Abstract**: The effectiveness of search systems is evaluated using relevance labels that indicate the usefulness of documents for specific queries and users. While obtaining these relevance labels from real users is ideal, scaling such data collection is challenging. Consequently, third-party annotators are employed, but their inconsistent accuracy demands costly auditing, training, and monitoring. We propose an LLM-based modular classification pipeline that divides the relevance assessment task into multiple stages, each utilising different prompts and models of varying sizes and capabilities. Applied to TREC Deep Learning (TREC-DL), one of our approaches showed an 18.4% Krippendorff's $\alpha$ accuracy increase over OpenAI's GPT-4o mini while maintaining a cost of about 0.2 USD per million input tokens, offering a more efficient and scalable solution for relevance assessment. This approach beats the baseline performance of GPT-4o (5 USD). With a pipeline approach, even the accuracy of the GPT-4o flagship model, measured in $\alpha$, could be improved by 9.7%. 

**Abstract (ZH)**: 基于搜索系统的有效性通过相关性标签进行评估，这些标签表明特定查询和用户对于文档的有用性。尽管从真实用户处获取这些相关性标签是理想的，但大规模收集此类数据具有挑战性。因此，通常会雇佣第三方标注者，但他们的不一致准确性需要耗费大量成本进行审核、培训和监控。我们提出了一种基于大语言模型（LLM）的模块化分类流水线，将相关性评估任务划分为多个阶段，每个阶段使用不同类型的提示和具有不同大小与能力的模型。应用于TREC深度学习（TREC-DL）任务中，我们的一种方法在保持每百万输入标记成本约为0.2美元的情况下，Krippendorff’s $\alpha$精度提高了18.4%，提供了一种更为高效和可扩展的相关性评估解决方案。这种方法优于GPT-4o基线模型（5美元）。使用流水线方法，即使是GPT-4o旗舰模型的$\alpha$精度也得以提高9.7%。 

---
# Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation 

**Title (ZH)**: 面向多模态序列推荐的分层时敏专家混合模型 

**Authors**: Shengzhe Zhang, Liyi Chen, Dazhong Shen, Chao Wang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.14269)  

**Abstract**: Multi-modal sequential recommendation (SR) leverages multi-modal data to learn more comprehensive item features and user preferences than traditional SR methods, which has become a critical topic in both academia and industry. Existing methods typically focus on enhancing multi-modal information utility through adaptive modality fusion to capture the evolving of user preference from user-item interaction sequences. However, most of them overlook the interference caused by redundant interest-irrelevant information contained in rich multi-modal data. Additionally, they primarily rely on implicit temporal information based solely on chronological ordering, neglecting explicit temporal signals that could more effectively represent dynamic user interest over time. To address these limitations, we propose a Hierarchical time-aware Mixture of experts for multi-modal Sequential Recommendation (HM4SR) with a two-level Mixture of Experts (MoE) and a multi-task learning strategy. Specifically, the first MoE, named Interactive MoE, extracts essential user interest-related information from the multi-modal data of each item. Then, the second MoE, termed Temporal MoE, captures user dynamic interests by introducing explicit temporal embeddings from timestamps in modality encoding. To further address data sparsity, we propose three auxiliary supervision tasks: sequence-level category prediction (CP) for item feature understanding, contrastive learning on ID (IDCL) to align sequence context with user interests, and placeholder contrastive learning (PCL) to integrate temporal information with modalities for dynamic interest modeling. Extensive experiments on four public datasets verify the effectiveness of HM4SR compared to several state-of-the-art approaches. 

**Abstract (ZH)**: 多模态序列推荐（SR）利用多模态数据来学习比传统SR方法更为全面的项目特征和用户偏好，已成为学术界和工业界的关键研究课题。现有方法通常侧重于通过自适应模态融合增强多模态信息效用，以捕捉用户偏好随用户-项目交互序列的演变。然而，大多数方法忽视了丰富多模态数据中包含的冗余兴趣无关信息所带来的干扰。此外，它们主要依赖于基于时间顺序的隐式时间信息，而忽略了能更有效地表现用户随时间动态兴趣的显式时间信号。为解决这些局限性，我们提出了一种带有两层专家混合（MoE）和多任务学习策略的层级时间感知专家混合多模态序列推荐（HM4SR）。具体而言，第一层专家混合，称为交互式MoE，从每个项目的多模态数据中提取出与用户兴趣相关的重要信息。然后，第二层专家混合，称为时间MoE，在模态编码中引入时间戳的显式时间嵌入以捕捉用户动态兴趣。为了进一步解决数据稀疏性问题，我们提出了三种辅助监督任务：序列级别类别预测（CP）以理解项目特征、基于ID的对比学习（IDCL）以对齐序列上下文与用户兴趣，以及占位符对比学习（PCL）以将时间信息与模态结合以建模动态兴趣。在四个公开数据集上的广泛实验验证了HM4SR相较于多种最先进的方法的有效性。 

---
# Pre-train and Fine-tune: Recommenders as Large Models 

**Title (ZH)**: 预训练与 fine-tuning：将推荐系统作为大规模模型 

**Authors**: Zhenhao Jiang, Chenghao Chen, Hao Feng, Yu Yang, Jin Liu, Jie Zhang, Jia Jia, Ning Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14268)  

**Abstract**: In reality, users have different interests in different periods, regions, scenes, etc. Such changes in interest are so drastic that they are difficult to be captured by recommenders. Existing multi-domain learning can alleviate this problem. However, the structure of the industrial recommendation system is complex, the amount of data is huge, and the training cost is extremely high, so it is difficult to modify the structure of the industrial recommender and re-train it. To fill this gap, we consider recommenders as large pre-trained models and fine-tune them. We first propose the theory of the information bottleneck for fine-tuning and present an explanation for the fine-tuning technique in recommenders. To tailor for recommendation, we design an information-aware adaptive kernel (IAK) technique to fine-tune the pre-trained recommender. Specifically, we define fine-tuning as two phases: knowledge compression and knowledge matching and let the training stage of IAK explicitly approximate these two phases. Our proposed approach designed from the essence of fine-tuning is well interpretable. Extensive online and offline experiments show the superiority of our proposed method. Besides, we also share unique and important lessons we learned when deploying the method in a large-scale online platform. We also present the potential issues of fine-tuning techniques in recommendation systems and the corresponding solutions. The recommender with IAK technique has been deployed on the homepage of a billion-scale online food platform for several months and has yielded considerable profits in our business. 

**Abstract (ZH)**: 实际上，用户在不同时间段、不同地区和不同场景中具有不同的兴趣。这种兴趣变化之剧烈，使得现有的推荐系统难以捕捉到这些变化。现有的多域学习可以在一定程度上缓解这一问题。然而，工业推荐系统的结构复杂，数据量巨大，训练成本极高，因此很难修改工业推荐系统的结构并重新训练。为了解决这个问题，我们将推荐系统视为大型预训练模型，并进行微调。首先，我们提出了信息瓶颈的微调理论，并给出了推荐系统中微调技术的解释。为了适应推荐需求，我们设计了一种信息感知自适应核（IAK）技术来进行预训练推荐器的微调。具体而言，我们将微调分为两个阶段：知识压缩和知识匹配，并使IAK的训练阶段明确地逼近这两个阶段。我们提出的方法从微调的本质出发，易于解释。广泛的在线和离线实验表明了我们提出方法的优势。此外，我们还分享了在大规模在线平台上部署该方法时获得的独特且重要的经验教训。我们还探讨了推荐系统中微调技术的潜在问题及其相应的解决方案。使用IAK技术的推荐系统已在一家十亿级在线食品平台的首页部署了几个月，并为我们带来了显著的收益。 

---
# Do LLMs Provide Consistent Answers to Health-Related Questions across Languages? 

**Title (ZH)**: 大型语言模型在不同语言中对健康相关问题的回答是否一致？ 

**Authors**: Ipek Baris Schlicht, Zhixue Zhao, Burcu Sayin, Lucie Flek, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2501.14719)  

**Abstract**: Equitable access to reliable health information is vital for public health, but the quality of online health resources varies by language, raising concerns about inconsistencies in Large Language Models (LLMs) for healthcare. In this study, we examine the consistency of responses provided by LLMs to health-related questions across English, German, Turkish, and Chinese. We largely expand the HealthFC dataset by categorizing health-related questions by disease type and broadening its multilingual scope with Turkish and Chinese translations. We reveal significant inconsistencies in responses that could spread healthcare misinformation. Our main contributions are 1) a multilingual health-related inquiry dataset with meta-information on disease categories, and 2) a novel prompt-based evaluation workflow that enables sub-dimensional comparisons between two languages through parsing. Our findings highlight key challenges in deploying LLM-based tools in multilingual contexts and emphasize the need for improved cross-lingual alignment to ensure accurate and equitable healthcare information. 

**Abstract (ZH)**: 获取可靠的健康信息的机会公平是公共卫生的关键，但在线健康资源的质量因语言而异，这引起了对大型语言模型（LLMs）在医疗保健领域一致性方面的担忧。在本研究中，我们考察了LLMs对英、德、土耳其和中文健康相关问题的回答一致性。我们大幅扩展了HealthFC数据集，通过按疾病类型分类健康相关问题，并进一步扩展其多语言范围，加入了土耳其语和汉语的翻译。我们揭示了回答中存在的显著差异，这些差异可能会传播医疗误导信息。我们主要的贡献包括1) 一个包含疾病类别元信息的多语言健康相关查询数据集，以及2) 一种新颖的基于提示的评估工作流，可以通过解析来实现两种语言在次维度上的比较。研究结果突显了在多语言环境下部署基于LLM的工具所面临的关键挑战，并强调了为了确保准确性和公平性，改进跨语言对齐的必要性。 

---
# CAMEO: Autocorrelation-Preserving Line Simplification for Lossy Time Series Compression 

**Title (ZH)**: CAMEO: 保留自相关性的折线简化方法用于失真时间序列压缩 

**Authors**: Carlos Enrique Muñiz-Cuza, Matthias Boehm, Torben Bach Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14432)  

**Abstract**: Time series data from a variety of sensors and IoT devices need effective compression to reduce storage and I/O bandwidth requirements. While most time series databases and systems rely on lossless compression, lossy techniques offer even greater space-saving with a small loss in precision. However, the unknown impact on downstream analytics applications requires a semi-manual trial-and-error exploration. We initiate work on lossy compression that provides guarantees on complex statistical features (which are strongly correlated with the accuracy of the downstream analytics). Specifically, we propose a new lossy compression method that provides guarantees on the autocorrelation and partial-autocorrelation functions (ACF/PACF) of a time series. Our method leverages line simplification techniques as well as incremental maintenance of aggregates, blocking, and parallelization strategies for effective and efficient compression. The results show that our method improves compression ratios by 2x on average and up to 54x on selected datasets, compared to previous lossy and lossless compression methods. Moreover, we maintain -- and sometimes even improve -- the forecasting accuracy by preserving the autocorrelation properties of the time series. Our framework is extensible to multivariate time series and other statistical features of the time series. 

**Abstract (ZH)**: 来自各种传感器和物联网设备的时间序列数据需要有效的压缩以减少存储和I/O带宽需求。虽然大多数时间序列数据库和系统依赖于无损压缩，但无损技术在保证精度的同时提供了更高的空间节省。然而，这些技术对下游分析应用的影响尚不清楚，需要进行半自动的试错探索。我们开始了在提供对复杂统计特征保证（这些特征与下游分析的准确性密切相关）的无损压缩研究。具体而言，我们提出了一种新的无损压缩方法，该方法能提供时间序列自相关函数（ACF）和偏自相关函数（PACF）的保证。该方法利用了线简化技术，并采用增量聚合维护、分块和并行化策略，以实现高效压缩。实验结果表明，与之前的无损和有损压缩方法相比，我们的方法平均压缩比提高了2倍，最高可提高54倍。同时，通过保留时间序列的自相关特性，我们维护甚至提高了预测准确性。我们的框架还可以扩展到多变量时间序列以及其他时间序列的统计特征。 

---
# Revisiting Applicable and Comprehensive Knowledge Tracing in Large-Scale Data 

**Title (ZH)**: 重新审视大规模数据中的适用性和综合性知识追踪 

**Authors**: Yiyun Zhou, Wenkang Han, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14256)  

**Abstract**: Knowledge Tracing (KT) is a fundamental component of Intelligent Tutoring Systems (ITS), enabling the modeling of students' knowledge states to predict future performance. The introduction of Deep Knowledge Tracing (DKT), the first deep learning-based KT (DLKT) model, has brought significant advantages in terms of applicability and comprehensiveness. However, recent DLKT models, such as Attentive Knowledge Tracing (AKT), have often prioritized predictive performance at the expense of these benefits. While deep sequential models like DKT have shown potential, they face challenges related to parallel computing, storage decision modification, and limited storage capacity. To address these limitations, we propose DKT2, a novel KT model that leverages the recently developed xLSTM architecture. DKT2 enhances input representation using the Rasch model and incorporates Item Response Theory (IRT) for interpretability, allowing for the decomposition of learned knowledge into familiar and unfamiliar knowledge. By integrating this knowledge with predicted questions, DKT2 generates comprehensive knowledge states. Extensive experiments conducted across three large-scale datasets demonstrate that DKT2 consistently outperforms 17 baseline models in various prediction tasks, underscoring its potential for real-world educational applications. This work bridges the gap between theoretical advancements and practical implementation in this http URL code and datasets will be available at this https URL. 

**Abstract (ZH)**: 知识追踪（KT）是智能辅导系统（ITS）的基本组成部分，能够在预测学生未来表现的同时，建模学生当前的知识状态。基于深度学习的知识追踪（DLKT，Deep Learning-based Knowledge Tracing）模型的引入，极大地改善了应用性和全面性。然而，近期的DLKT模型，如注意机制知识追踪（AKT，Attentive Knowledge Tracing），往往在预测性能上取得成效，却牺牲了这些优势。尽管深层序列模型如DKT展示了潜力，但它们在并行计算、存储决策修改和有限的存储容量等方面面临挑战。为解决这些局限，我们提出DKT2，这是一种新型的知识追踪模型，利用了最近发展起来的xLSTM架构。DKT2 通过使用Rasch模型增强输入表示，并结合项目反应理论（IRT）以提高可解释性，从而将学习到的知识分解为熟悉的和不熟悉的知识。通过将这些知识与预测出的问题相结合，DKT2 生成了全面的知识状态。在三个大规模数据集上进行的广泛实验表明，DKT2 在各种预测任务中均优于17个基线模型，表明其在实际教育应用中的潜力。本文填补了理论进步与实际应用之间的差距，在此http URL 公开了相关代码和数据集。 

---
# MedSlice: Fine-Tuned Large Language Models for Secure Clinical Note Sectioning 

**Title (ZH)**: MedSlice: 细调的大语言模型在临床病历段落划分中的安全应用 

**Authors**: Joshua Davis, Thomas Sounack, Kate Sciacca, Jessie M Brain, Brigitte N Durieux, Nicole D Agaronnik, Charlotta Lindvall  

**Link**: [PDF](https://arxiv.org/pdf/2501.14105)  

**Abstract**: Extracting sections from clinical notes is crucial for downstream analysis but is challenging due to variability in formatting and labor-intensive nature of manual sectioning. While proprietary large language models (LLMs) have shown promise, privacy concerns limit their accessibility. This study develops a pipeline for automated note sectioning using open-source LLMs, focusing on three sections: History of Present Illness, Interval History, and Assessment and Plan. We fine-tuned three open-source LLMs to extract sections using a curated dataset of 487 progress notes, comparing results relative to proprietary models (GPT-4o, GPT-4o mini). Internal and external validity were assessed via precision, recall and F1 score. Fine-tuned Llama 3.1 8B outperformed GPT-4o (F1=0.92). On the external validity test set, performance remained high (F1= 0.85). Fine-tuned open-source LLMs can surpass proprietary models in clinical note sectioning, offering advantages in cost, performance, and accessibility. 

**Abstract (ZH)**: 从临床笔记中提取段落对于下游分析至关重要，但由于格式变化多样性和手工段落提取的劳动密集性，这一过程具有挑战性。尽管专有大型语言模型（LLMs）展现出潜在的应用前景，但由于隐私问题，它们的可访问性受到限制。本研究开发了一种使用开源LLMs的自动化笔记段落提取管道，重点关注三个段落：现病史、间诊记录和评估与计划。我们利用一个包含487份病程记录的定制数据集，对三种开源LLMs进行了微调，并将其结果与专有模型（GPT-4o、GPT-4o mini）进行了比较。通过精确度、召回率和F1分数进行了内部和外部有效性的评估。微调后的Llama 3.1 8B在F1分数方面优于GPT-4o（F1=0.92）。在外部分析测试集上，性能依然保持较高水平（F1=0.85）。研究表明，微调后的开源LLMs在临床笔记段落提取方面可以超越专有模型，提供成本效益、高性能和易于访问的优势。 

---
# CAPRAG: A Large Language Model Solution for Customer Service and Automatic Reporting using Vector and Graph Retrieval-Augmented Generation 

**Title (ZH)**: CAPRAG：一种基于向量和图检索增强生成的大型语言模型解决方案，用于客户服务和自动报告 

**Authors**: Hamza Landolsi, Kais Letaief, Nizar Taghouti, Ines Abdeljaoued-Tej  

**Link**: [PDF](https://arxiv.org/pdf/2501.13993)  

**Abstract**: The introduction of new features and services in the banking sector often overwhelms customers, creating an opportunity for banks to enhance user experience through financial chatbots powered by large language models (LLMs). We initiated an AI agent designed to provide customers with relevant information about banking services and insights from annual reports. We proposed a hybrid Customer Analysis Pipeline Retrieval-Augmented Generation (CAPRAG) that effectively addresses both relationship-based and contextual queries, thereby improving customer engagement in the digital banking landscape. To implement this, we developed a processing pipeline to refine text data, which we utilized in two main frameworks: Vector RAG and Graph RAG. This dual approach enables us to populate both vector and graph databases with processed data for efficient retrieval. The Cypher query component is employed to effectively query the graph database. When a user submits a query, it is first expanded by a query expansion module before being routed to construct a final query from the hybrid Knowledge Base (KB). This final query is then sent to an open-source LLM for response generation. Overall, our innovative, designed to international banks, serves bank's customers in an increasingly complex digital environment, enhancing clarity and accessibility of information. 

**Abstract (ZH)**: 银行领域中引入新的功能和服务往往会让客户感到不知所措，这为银行通过大规模语言模型（LLMs）驱动的金融聊天机器人提升用户体验提供了机会。我们启动了一个AI代理，旨在为客户提供有关银行服务的相关信息和年报洞察。我们提出了一种混合客户分析管道检索增强生成（CAPRAG）方法，能够有效应对关系型和上下文型查询，从而在数字化银行环境中提升客户参与度。为实现这一目标，我们开发了一个处理管道来细化文本数据，并将其应用于两个主要框架：向量检索增强生成（Vector RAG）和图检索增强生成（Graph RAG）。这种双管齐下的方法使我们能够将处理后的数据填充到向量和图数据库中，以便高效检索。我们使用Cypher查询组件来有效地查询图数据库。当用户提交查询时，它首先通过查询扩展模块进行扩展，然后路由到混合知识库（KB）构建最终查询。该最终查询随后发送给开源的大规模语言模型以生成响应。总体而言，我们设计的这一创新方案致力于为国际银行及其客户提供更加复杂数字化环境中的清晰和易于获取的信息，从而提升他们的用户体验。 

---
# Assisting Mathematical Formalization with A Learning-based Premise Retriever 

**Title (ZH)**: 使用基于学习的前提检索辅助数学形式化 

**Authors**: Yicheng Tao, Haotian Liu, Shanwen Wang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13959)  

**Abstract**: Premise selection is a crucial yet challenging step in mathematical formalization, especially for users with limited experience. Due to the lack of available formalization projects, existing approaches that leverage language models often suffer from data scarcity. In this work, we introduce an innovative method for training a premise retriever to support the formalization of mathematics. Our approach employs a BERT model to embed proof states and premises into a shared latent space. The retrieval model is trained within a contrastive learning framework and incorporates a domain-specific tokenizer along with a fine-grained similarity computation method. Experimental results show that our model is highly competitive compared to existing baselines, achieving strong performance while requiring fewer computational resources. Performance is further enhanced through the integration of a re-ranking module. To streamline the formalization process, we will release a search engine that enables users to query Mathlib theorems directly using proof states, significantly improving accessibility and efficiency. Codes are available at this https URL. 

**Abstract (ZH)**: 前提选择是数学形式化过程中一个关键但具有挑战性的步骤，尤其是对于经验有限的用户而言。由于可用的形式化项目有限，现有的利用语言模型的方法往往面临数据稀缺的问题。在本文中，我们提出了一种创新的方法，用于训练一个前提检索器以支持数学的形式化。我们的方法采用BERT模型将证明状态和前提嵌入到共享的潜在空间中。检索模型在对比学习框架下进行训练，并结合领域特定的分词器及细粒度的相似度计算方法。实验结果显示，我们的模型在与现有基线方法的对比中表现出很高的竞争力，能够在较少的计算资源要求下达到强大的性能。通过引入重排模块，性能进一步提升。为了简化形式化过程，我们还将发布一个搜索引擎，使用户能够直接使用证明状态查询Mathlib定理，从而显著提高其可访问性和效率。代码可在此处访问：[此网址]。 

---
# A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models 

**Title (ZH)**: 图检索增强生成在定制化大规模语言模型中的调研 

**Authors**: Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13958)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in a wide range of tasks, yet their application to specialized domains remains challenging due to the need for deep expertise. Retrieval-augmented generation (RAG) has emerged as a promising solution to customize LLMs for professional fields by seamlessly integrating external knowledge bases, enabling real-time access to domain-specific expertise during inference. Despite its potential, traditional RAG systems, based on flat text retrieval, face three critical challenges: (i) complex query understanding in professional contexts, (ii) difficulties in knowledge integration across distributed sources, and (iii) system efficiency bottlenecks at scale. This survey presents a systematic analysis of Graph-based Retrieval-Augmented Generation (GraphRAG), a new paradigm that revolutionizes domain-specific LLM applications. GraphRAG addresses traditional RAG limitations through three key innovations: (i) graph-structured knowledge representation that explicitly captures entity relationships and domain hierarchies, (ii) efficient graph-based retrieval techniques that enable context-preserving knowledge retrieval with multihop reasoning ability, and (iii) structure-aware knowledge integration algorithms that leverage retrieved knowledge for accurate and logical coherent generation of LLMs. In this survey, we systematically analyze the technical foundations of GraphRAG and examine current implementations across various professional domains, identifying key technical challenges and promising research directions. All the related resources of GraphRAG, including research papers, open-source data, and projects, are collected for the community in \textcolor{blue}{\url{this https URL}}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在一系列任务中展示了卓越的能力，但在应用于专业领域时仍然面临挑战，因为需要深厚的专业知识。检索增强生成（RAG）作为一种有前途的解决方案，通过无缝集成外部知识库，使在推理过程中能够实时访问特定领域的专业知识。尽管具有潜力，但传统的基于扁平文本检索的RAG系统面临着三个关键挑战：（i）专业环境下的复杂查询理解，（ii）跨分布式源的知识整合困难，以及（iii）在大规模系统中的效率瓶颈。本文综述了基于图的检索增强生成（GraphRAG），这是一种革新特定领域LLM应用的新范式。GraphRAG通过三个关键创新解决了传统RAG的局限性：（i）结构化的图式知识表示，明确捕捉实体关系和领域层次结构，（ii）高效的图式检索技术，能够进行上下文保存的知识检索并具备多跳推理能力，以及（iii）结构感知的知识整合算法，利用检索到的知识进行准确且逻辑连贯的LLM生成。在本文综述中，我们系统分析了GraphRAG的技术基础，并考察了其在各个专业领域的当前实现，识别了关键的技术挑战和有前途的研究方向。所有与GraphRAG相关的资源，包括研究论文、开源数据和项目，都被收集在\textcolor{blue}{\url{这里}}为社区提供。 

---
# Zep: A Temporal Knowledge Graph Architecture for Agent Memory 

**Title (ZH)**: ZEP：一种用于代理记忆的时间知识图架构

这个翻译符合学术规范，保留了原文的核心意思，并且采用了更加正式和学术化的表达方式。 

**Authors**: Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef  

**Link**: [PDF](https://arxiv.org/pdf/2501.13956)  

**Abstract**: We introduce Zep, a novel memory layer service for AI agents that outperforms the current state-of-the-art system, MemGPT, in the Deep Memory Retrieval (DMR) benchmark. Additionally, Zep excels in more comprehensive and challenging evaluations than DMR that better reflect real-world enterprise use cases. While existing retrieval-augmented generation (RAG) frameworks for large language model (LLM)-based agents are limited to static document retrieval, enterprise applications demand dynamic knowledge integration from diverse sources including ongoing conversations and business data. Zep addresses this fundamental limitation through its core component Graphiti -- a temporally-aware knowledge graph engine that dynamically synthesizes both unstructured conversational data and structured business data while maintaining historical relationships. In the DMR benchmark, which the MemGPT team established as their primary evaluation metric, Zep demonstrates superior performance (94.8% vs 93.4%). Beyond DMR, Zep's capabilities are further validated through the more challenging LongMemEval benchmark, which better reflects enterprise use cases through complex temporal reasoning tasks. In this evaluation, Zep achieves substantial results with accuracy improvements of up to 18.5% while simultaneously reducing response latency by 90% compared to baseline implementations. These results are particularly pronounced in enterprise-critical tasks such as cross-session information synthesis and long-term context maintenance, demonstrating Zep's effectiveness for deployment in real-world applications. 

**Abstract (ZH)**: 我们引入了Zep，这是一种新的记忆层服务，它在深度记忆检索（DMR）基准测试中超越了当前最先进的系统MemGPT。此外，Zep在比DMR更为全面和具有挑战性的评估中表现出色，这些评估更好地反映了实际的企业应用场景。现有的基于大型语言模型（LLM）的检索增强生成（RAG）框架仅限于静态文档检索，而企业应用则需要动态集成来自各种来源（包括正在进行的对话和业务数据）的知识。Zep通过其核心组件Graphiti——一个具有时间感知的知识图谱引擎来解决这一根本性的局限性。Graphiti能够动态地综合未结构化对话数据和结构化业务数据，并保留历史关系。在MemGPT团队将其确立为主要评估指标的DMR基准中，Zep展示了更优的性能（94.8% vs 93.4%）。除了DMR之外，Zep的能力还通过更具挑战性的LongMemEval基准得到了验证，该基准通过复杂的时序推理任务更好地反映了企业的应用案例。在这一评估中，Zep实现了显著的结果，准确率提高了高达18.5%，同时将响应延迟降低了90%相比于基线实施。这些结果在诸如会话间信息综合和长期上下文保存等企业关键任务中尤为显著，展示了Zep在实际应用部署中的有效性。 

---
# Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents 

**Title (ZH)**: Chat3GPP：一个用于3GPP文档的开源检索增强生成框架 

**Authors**: Long Huang, Ming Zhao, Limin Xiao, Xiujun Zhang, Jungang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13954)  

**Abstract**: The 3rd Generation Partnership Project (3GPP) documents is key standards in global telecommunications, while posing significant challenges for engineers and researchers in the telecommunications field due to the large volume and complexity of their contents as well as the frequent updates. Large language models (LLMs) have shown promise in natural language processing tasks, but their general-purpose nature limits their effectiveness in specific domains like telecommunications. To address this, we propose Chat3GPP, an open-source retrieval-augmented generation (RAG) framework tailored for 3GPP specifications. By combining chunking strategies, hybrid retrieval and efficient indexing methods, Chat3GPP can efficiently retrieve relevant information and generate accurate responses to user queries without requiring domain-specific fine-tuning, which is both flexible and scalable, offering significant potential for adapting to other technical standards beyond 3GPP. We evaluate Chat3GPP on two telecom-specific datasets and demonstrate its superior performance compared to existing methods, showcasing its potential for downstream tasks like protocol generation and code automation. 

**Abstract (ZH)**: 3GPP文档是全球电信领域的关键标准，但由于其内容庞大且复杂，以及频繁的更新，给电信领域的工程师和研究人员带来了巨大挑战。大型语言模型（LLMs）在自然语言处理任务中展现了巨大潜力，但其通用性限制了其在特定领域（如电信领域）的有效性。为了解决这一问题，我们提出了一种针对3GPP规范的开源检索增强生成（RAG）框架——Chat3GPP。通过结合分块策略、混合检索和高效的索引方法，Chat3GPP能够高效地检索相关信息，并生成准确的用户查询响应，而无需进行特定领域的微调。这使得Chat3GPP具有灵活性和可扩展性，为适应其他技术标准提供了巨大潜力。我们在两个电信特定数据集上评估了Chat3GPP，并展示了其在现有方法中的优越性能，彰显了其在协议生成和代码自动化等下游任务中的潜在应用价值。 

---
