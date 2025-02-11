# The 2021 Tokyo Olympics Multilingual News Article Dataset 

**Title (ZH)**: 2021年东京奥运会多语言新闻文章数据集 

**Authors**: Erik Novak, Erik Calcina, Dunja Mladenić, Marko Grobelnik  

**Link**: [PDF](https://arxiv.org/pdf/2502.06648)  

**Abstract**: In this paper, we introduce a dataset of multilingual news articles covering the 2021 Tokyo Olympics. A total of 10,940 news articles were gathered from 1,918 different publishers, covering 1,350 sub-events of the 2021 Olympics, and published between July 1, 2021, and August 14, 2021. These articles are written in nine languages from different language families and in different scripts. To create the dataset, the raw news articles were first retrieved via a service that collects and analyzes news articles. Then, the articles were grouped using an online clustering algorithm, with each group containing articles reporting on the same sub-event. Finally, the groups were manually annotated and evaluated. The development of this dataset aims to provide a resource for evaluating the performance of multilingual news clustering algorithms, for which limited datasets are available. It can also be used to analyze the dynamics and events of the 2021 Tokyo Olympics from different perspectives. The dataset is available in CSV format and can be accessed from the this http URL repository. 

**Abstract (ZH)**: 在这篇论文中，我们介绍了涵盖2021年东京奥运会的多语言新闻文章数据集。我们从1,918家不同的出版商收集了共计10,940篇新闻文章，涵盖了2021年奥运会的1,350个子项目，发布日期从2021年7月1日到2021年8月14日。这些文章使用了九种不同语系和不同文字体系的九种语言撰写。为了构建该数据集，首先通过一个收集和分析新闻文章的服务获取了原始新闻文章，然后使用在线聚类算法将文章分组，每个组包含报道同一子项目的文章。最后，对这些组进行了人工标注和评估。构建此数据集的目的是提供一个用于评估多语言新闻聚类算法性能的资源，现有可用的数据集较少。它还可以用于从不同角度分析2021年东京奥运会的动态和事件。该数据集以CSV格式提供，并可通过以下链接访问：[请将此httpURL替换为实际的URL链接]。 

---
# LiveForesighter: Generating Future Information for Live-Streaming Recommendations at Kuaishou 

**Title (ZH)**: 快手的LiveForesighter：为直播推荐生成未来信息 

**Authors**: Yucheng Lu, Jiangxia Cao, Xu Kuan, Wei Cheng, Wei Jiang, Jiaming Zhang, Yang Shuang, Liu Zhaojie, Liyin Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06557)  

**Abstract**: Live-streaming, as a new-generation media to connect users and authors, has attracted a lot of attention and experienced rapid growth in recent years. Compared with the content-static short-video recommendation, the live-streaming recommendation faces more challenges in giving our users a satisfactory experience: (1) Live-streaming content is dynamically ever-changing along time. (2) valuable behaviors (e.g., send digital-gift, buy products) always require users to watch for a long-time (>10 min). Combining the two attributes, here raising a challenging question for live-streaming recommendation: How to discover the live-streamings that the content user is interested in at the current moment, and further a period in the future? 

**Abstract (ZH)**: 直播作为一种新型媒体，能够连接用户和创作者，近年来引起了广泛关注并经历了快速增长。与内容静态的短视频推荐相比，直播推荐面临更多的挑战，以给用户带来满意的体验：（1）直播内容随着时间动态变化；（2）有价值的用户行为（例如赠送虚拟礼物、购买产品）通常需要用户长时间观看（>10分钟）。结合这两种属性，这里提出一个对于直播推荐具有挑战性的问题：如何发现用户当前和未来一段时间内感兴趣的直播内容？ 

---
# Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation 

**Title (ZH)**: 生成推荐中的逐步协作与语义知识融合 

**Authors**: Longtao Xiao, Haozhao Wang, Cheng Wang, Linfei Ji, Yifan Wang, Jieming Zhu, Zhenhua Dong, Rui Zhang, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.06269)  

**Abstract**: With the recent surge in interest surrounding generative paradigms, generative recommendation has increasingly attracted the attention of researchers in the recommendation community. This paradigm generally consists of two stages. In the first stage, pretrained semantic embeddings or collaborative ID embeddings are quantized to create item codes, aiming to capture and preserve rich semantic or collaborative knowledge within these codes. The second stage involves utilizing these discrete codes to perform an autoregressive sequence generation task. Existing methods often either overlook collaborative or semantic knowledge, or combine the two roughly. In this paper, we observe that naively concatenating representations from semantic and collaborative modality leads to a semantic domination issue, where the resulting representation is overly influenced by semantic information, effectively overshadowing the collaborative representation. Consequently, downstream recommendation tasks fail to fully exploit the knowledge from both modalities, resulting in suboptimal performance. To address this, we propose a progressive collaborative and semantic knowledge fusion model for generative recommendation, named PRORec, which integrates semantic and collaborative knowledge with a unified code through a two-stage framework. Specifically, in the first stage, we propose a cross-modality knowledge alignment task, which integrates semantic knowledge into collaborative embeddings, enhancing their representational capability. In the second stage, we propose an in-modality knowledge distillation task, designed to effectively capture and integrate knowledge from both semantic and collaborative modalities. Extensive experiments on three widely used benchmarks validate the effectiveness of our approach, demonstrating its superiority compared to existing methods. 

**Abstract (ZH)**: 近年来，生成范式的兴趣激增，生成推荐逐渐引起了推荐领域研究人员的广泛关注。这种范式通常由两个阶段构成。第一阶段，使用预训练的语义嵌入或协作ID嵌入构建项目代码，目的是捕捉和保留这些代码中的丰富语义或协作知识。第二阶段，则是利用这些离散代码执行自回归序列生成任务。现有方法往往要么忽略了协作知识，要么草率地将语义和协作知识结合起来。本文观察到，简单地将语义和协作模态的表示进行拼接会导致语义主导问题，即生成的表示过度依赖于语义信息，有效地遮蔽了协作表示。因此，下游推荐任务无法充分利用两种模态的知识，导致性能次优。为了解决这个问题，我们提出了一种渐进式的协作与语义知识融合模型，命名为PRORec，该模型通过统一代码以两阶段框架将语义和协作知识融合起来。具体而言，在第一阶段，我们提出了一种跨模态知识对齐任务，将语义知识整合到协作嵌入中，增强其表示能力。在第二阶段，我们提出了一种模内知识蒸馏任务，旨在有效捕捉和整合来自语义和协作模态的知识。在三个广泛使用的基准上的大量实验验证了我们方法的有效性，证明了其优于现有方法的优越性。 

---
# Evaluating Entity Retrieval in Electronic Health Records: a Semantic Gap Perspective 

**Title (ZH)**: 从语义差距的角度评价电子健康记录中的实体检索 

**Authors**: Zhengyun Zhao, Hongyi Yuan, Jingjing Liu, Haichao Chen, Huaiyuan Ying, Songchi Zhou, Sheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06252)  

**Abstract**: Entity retrieval plays a crucial role in the utilization of Electronic Health Records (EHRs) and is applied across a wide range of clinical practices. However, a comprehensive evaluation of this task is lacking due to the absence of a public benchmark. In this paper, we propose the development and release of a novel benchmark for evaluating entity retrieval in EHRs, with a particular focus on the semantic gap issue. Using discharge summaries from the MIMIC-III dataset, we incorporate ICD codes and prescription labels associated with the notes as queries, and annotate relevance judgments using GPT-4. In total, we use 1,000 patient notes, generate 1,246 queries, and provide over 77,000 relevance annotations. To offer the first assessment of the semantic gap, we introduce a novel classification system for relevance matches. Leveraging GPT-4, we categorize each relevant pair into one of five categories: string, synonym, abbreviation, hyponym, and implication. Using the proposed benchmark, we evaluate several retrieval methods, including BM25, query expansion, and state-of-the-art dense retrievers. Our findings show that BM25 provides a strong baseline but struggles with semantic matches. Query expansion significantly improves performance, though it slightly reduces string match capabilities. Dense retrievers outperform traditional methods, particularly for semantic matches, and general-domain dense retrievers often surpass those trained specifically in the biomedical domain. 

**Abstract (ZH)**: 电子健康记录（EHRs）的实体检索在利用EHRs中扮演着关键角色，并被广泛应用于各种临床实践中。然而，由于缺乏公开的基准测试，这项任务的全面评估一直缺失。本文提出并发布了一个新的基准测试，用于评估EHR中的实体检索，特别关注语义差距问题。利用MIMIC-III数据集中的出院总结，我们将与笔记相关的ICD编码和处方标签作为查询，并使用GPT-4标注相关性判断。总共使用了1000个患者笔记，生成了1,246个查询，并提供了超过77,000个相关性标注。为了首次评估语义差距，我们引入了一个新的相关性匹配分类系统。利用GPT-4，我们将每个相关对分类为五类之一：字符串、同义词、缩写、下位词和蕴含。使用提出的基准测试，我们评估了几种检索方法，包括BM25、查询扩展以及最先进的稠密检索器。我们的研究结果表明，BM25提供了强大的基准，但在处理语义匹配时存在困难。查询扩展显著提高了性能，尽管这在一定程度上减少了字符串匹配的能力。稠密检索器在语义匹配方面优于传统方法，尤其是对于语义匹配任务，通用领域稠密检索器常常优于专门在生物医学领域训练的检索器。 

---
# RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning 

**Title (ZH)**: RALLRec：通过表示学习改进检索增强大型语言模型推荐 

**Authors**: Jian Xu, Sichun Luo, Xiangyu Chen, Haoming Huang, Hanxu Hou, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.06101)  

**Abstract**: Large Language Models (LLMs) have been integrated into recommendation systems to enhance user behavior comprehension. The Retrieval Augmented Generation (RAG) technique is further incorporated into these systems to retrieve more relevant items and improve system performance. However, existing RAG methods rely primarily on textual semantics and often fail to incorporate the most relevant items, limiting the effectiveness of the systems.
In this paper, we propose Representation learning for retrieval-Augmented Large Language model Recommendation (RALLRec). Specifically, we enhance textual semantics by prompting LLMs to generate more detailed item descriptions, followed by joint representation learning of textual and collaborative semantics, which are extracted by the LLM and recommendation models, respectively. Considering the potential time-varying characteristics of user interest, a simple yet effective reranking method is further introduced to capture the dynamics of user preference. We conducted extensive experiments on three real-world datasets, and the evaluation results validated the effectiveness of our method. Code is made public at this https URL. 

**Abstract (ZH)**: 以下是您的论文内容或标题的中文翻译，符合学术规范：

大型语言模型（LLMs）已整合到推荐系统中，以增强对用户行为的理解。检索增强生成（RAG）技术进一步融入这些系统中，以检索更多相关项并提高系统性能。然而，现有的RAG方法主要依赖文本语义，往往会忽略最相关的项，限制了系统的有效性。

在本文中，我们提出了检索增强大型语言模型推荐（RALLRec）的表示学习方法。具体而言，通过提示LLMs生成更详细的物品描述来增强文本语义，随后进行文本和合作语义的联合表示学习，这些语义分别由LLMs和推荐模型提取。考虑到用户兴趣的潜在时间变化特性，我们引入了一种简单有效的再排名方法，以捕捉用户偏好的动态变化。我们在三个真实世界的数据集上进行了广泛的实验，评估结果验证了我们方法的有效性。代码已在此处公开：[请在此插入网址]。 

---
# NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems 

**Title (ZH)**: NLGR：利用邻居列表进行生成性重新 Ranking 在个性化推荐系统中的应用 

**Authors**: Shuli Wang, Xue Wei, Senjie Kou, Chi Wang, Wenshuai Chen, Qi Tang, Yinhua Zhu, Xiong Xiao, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06097)  

**Abstract**: Reranking plays a crucial role in modern multi-stage recommender systems by rearranging the initial ranking list. Due to the inherent challenges of combinatorial search spaces, some current research adopts an evaluator-generator paradigm, with a generator generating feasible sequences and an evaluator selecting the best sequence based on the estimated list utility. However, these methods still face two issues. Firstly, due to the goal inconsistency problem between the evaluator and generator, the generator tends to fit the local optimal solution of exposure distribution rather than combinatorial space optimization. Secondly, the strategy of generating target items one by one is difficult to achieve optimality because it ignores the information of subsequent items.
To address these issues, we propose a utilizing Neighbor Lists model for Generative Reranking (NLGR), which aims to improve the performance of the generator in the combinatorial space. NLGR follows the evaluator-generator paradigm and improves the generator's training and generating methods. Specifically, we use neighbor lists in combination space to enhance the training process, making the generator perceive the relative scores and find the optimization direction. Furthermore, we propose a novel sampling-based non-autoregressive generation method, which allows the generator to jump flexibly from the current list to any neighbor list. Extensive experiments on public and industrial datasets validate NLGR's effectiveness and we have successfully deployed NLGR on the Meituan food delivery platform. 

**Abstract (ZH)**: 重新排序在现代多阶段推荐系统中扮演着至关重要的角色，它通过重新排列初始排名列表来提升推荐效果。由于排列组合搜索空间中的固有挑战，当前一些研究采用评估器-生成器范式，生成器生成可行序列，评估器基于估计的列表效益选择最佳序列。然而，这些方法仍然面临两个问题。首先，由于评估器和生成器目标不一致，生成器往往适应曝光分布的局部最优解，而不是组合空间的优化。其次，一项项生成目标项目的方法难以实现最优解，因为它忽视了后续项目的相关信息。

为了解决这些问题，我们提出了一种用于生成重排序的利用邻列表模型（Neighbor Lists for Generative Reranking, NLGR），旨在提高生成器在组合空间中的性能。NLGR 遵循评估器-生成器范式，并改进了生成器的训练和生成方法。具体而言，我们在组合空间中使用邻列表来增强训练过程，使生成器能够感知相对得分并找到优化方向。此外，我们提出了一种基于采样的非自回归生成方法，允许生成器从当前列表灵活跳转至任何邻列表。在公开和工业数据集上的广泛实验验证了 NLGR 的有效性，并且我们已经在美团外卖平台上成功部署了 NLGR。 

---
# FactIR: A Real-World Zero-shot Open-Domain Retrieval Benchmark for Fact-Checking 

**Title (ZH)**: FactIR：一种用于事实核查的现实世界零样本跨域检索基准 

**Authors**: Venktesh V, Vinay Setty  

**Link**: [PDF](https://arxiv.org/pdf/2502.06006)  

**Abstract**: The field of automated fact-checking increasingly depends on retrieving web-based evidence to determine the veracity of claims in real-world scenarios. A significant challenge in this process is not only retrieving relevant information, but also identifying evidence that can both support and refute complex claims. Traditional retrieval methods may return documents that directly address claims or lean toward supporting them, but often struggle with more complex claims requiring indirect reasoning. While some existing benchmarks and methods target retrieval for fact-checking, a comprehensive real-world open-domain benchmark has been lacking. In this paper, we present a real-world retrieval benchmark FactIR, derived from Factiverse production logs, enhanced with human annotations. We rigorously evaluate state-of-the-art retrieval models in a zero-shot setup on FactIR and offer insights for developing practical retrieval systems for fact-checking. Code and data are available at this https URL. 

**Abstract (ZH)**: 自动事实核查领域越来越多地依赖于获取网络证据来确定实际场景中声明的真实性。这一过程中的一个重要挑战不仅是检索相关信息，还需要识别既能支持又能反驳复杂声明的证据。传统检索方法可能会返回直接涉及声明或偏向支持声明的文档，但在处理需要间接推理的复杂声明时常常力不从心。尽管已有的一些基准和方法针对事实核查中的检索进行了优化，但全面的现实世界开放域基准仍然欠缺。在本文中，我们提出了一种基于Factiverse生产日志并经过人工标注增强的事实核查检索基准——FactIR。我们以零样本设置对前沿的检索模型进行了严格的评估，并提供了开发实际应用的检索系统以用于事实核查的研究洞见。相关的代码和数据可以在以下链接获取：[this https URL](this https URL)。 

---
# Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education 

**Title (ZH)**: Uni-Retrieval：STEM教育的多风格检索框架 

**Authors**: Yanhao Jia, Xinyi Wu, Hao Li, Qinglin Zhang, Yuxiao Hu, Shuai Zhao, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.05863)  

**Abstract**: In AI-facilitated teaching, leveraging various query styles to interpret abstract text descriptions is crucial for ensuring high-quality teaching. However, current retrieval models primarily focus on natural text-image retrieval, making them insufficiently tailored to educational scenarios due to the ambiguities in the retrieval process. In this paper, we propose a diverse expression retrieval task tailored to educational scenarios, supporting retrieval based on multiple query styles and expressions. We introduce the STEM Education Retrieval Dataset (SER), which contains over 24,000 query pairs of different styles, and the Uni-Retrieval, an efficient and style-diversified retrieval vision-language model based on prompt tuning. Uni-Retrieval extracts query style features as prototypes and builds a continuously updated Prompt Bank containing prompt tokens for diverse queries. This bank can updated during test time to represent domain-specific knowledge for different subject retrieval scenarios. Our framework demonstrates scalability and robustness by dynamically retrieving prompt tokens based on prototype similarity, effectively facilitating learning for unknown queries. Experimental results indicate that Uni-Retrieval outperforms existing retrieval models in most retrieval tasks. This advancement provides a scalable and precise solution for diverse educational needs. 

**Abstract (ZH)**: 在AI辅助教学中，利用各种查询风格解释抽象的文字描述对于确保高质量教学至关重要。然而，当前的检索模型主要集中在自然文本图像检索上，这让它们在教育场景中显得不够贴合，尤其是由于检索过程中的模糊性。在这篇论文中，我们提出了一种针对教育场景进行多元表达检索的任务，支持基于多种查询风格和表达的检索。我们介绍了STEM教育检索数据集（SER），其中包含了超过24,000对不同风格的查询配对，以及基于提示调优的Uni-Retrieval高效且风格多样化的检索视觉语言模型。Uni-Retrieval提取查询风格特征作为原型，并构建了一个不断更新的提示银行，包含多种查询的提示标记。该银行可以在测试时根据原型相似性进行更新，以表示不同学科检索场景中的特定领域知识。我们的框架通过动态检索基于原型相似性的提示标记，有效地促进了对未知查询的学习。实验结果表明，Uni-Retrieval在大多数检索任务中优于现有检索模型。这一进展为满足多样化的教育需求提供了可扩展且精准的解决方案。 

---
# HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads 

**Title (ZH)**: HCMRM：一种高一致性的多模态相关性模型用于搜索广告 

**Authors**: Guobing Gan, Kaiming Gao, Li Wang, Shen Jiang, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05822)  

**Abstract**: Search advertising is essential for merchants to reach the target users on short video platforms. Short video ads aligned with user search intents are displayed through relevance matching and bid ranking mechanisms. This paper focuses on improving query-to-video relevance matching to enhance the effectiveness of ranking in ad systems. Recent vision-language pre-training models have demonstrated promise in various multimodal tasks. However, their contribution to downstream query-video relevance tasks is limited, as the alignment between the pair of visual signals and text differs from the modeling of the triplet of the query, visual signals, and video text. In addition, our previous relevance model provides limited ranking capabilities, largely due to the discrepancy between the binary cross-entropy fine-tuning objective and the ranking objective. To address these limitations, we design a high-consistency multimodal relevance model (HCMRM). It utilizes a simple yet effective method to enhance the consistency between pre-training and relevance tasks. Specifically, during the pre-training phase, along with aligning visual signals and video text, several keywords are extracted from the video text as pseudo-queries to perform the triplet relevance modeling. For the fine-tuning phase, we introduce a hierarchical softmax loss, which enables the model to learn the order within labels while maximizing the distinction between positive and negative samples. This promotes the fusion ranking of relevance and bidding in the subsequent ranking stage. The proposed method has been deployed in the Kuaishou search advertising system for over a year, contributing to a 6.1% reduction in the proportion of irrelevant ads and a 1.4% increase in ad revenue. 

**Abstract (ZH)**: 短视频广告对于商家在短视频平台上触及目标用户至关重要。通过相关性匹配和出价排名机制，与用户搜索意图相匹配的短视频广告会在平台上展示。本文聚焦于提升查询与视频的相关性匹配，以增强广告系统的排名效果。近期的视觉-语言预训练模型已显示出在多种多模态任务上的潜力，但它们对下游查询与视频相关性任务的贡献有限，因为视觉信号对齐与文本之间的对齐和查询、视觉信号与视频文本三元组建模之间存在差异。此外，我们之前的相关性模型在排序能力方面相对有限，主要原因在于二元交叉熵微调目标与排序目标之间的差距。为了解决这些问题，我们设计了一种高一致性多模态相关性模型（HCMRM）。该模型利用简单而有效的方法，增强了预训练与相关性建模任务之间的一致性。具体而言，在预训练阶段，除了对齐视觉信号和视频文本外，还从视频文本中提取多个关键词作为伪查询，以进行三元组相关性建模。在微调阶段，我们引入了一种层次softmax损失，使模型能够在最大化正负样本区分的同时学习标签内的顺序。这促进了后续排名阶段相关性和出价的融合排序。所提出的该方法已在抖音搜索广告系统中部署超过一年，成功将无关广告的比例减少了6.1%，广告收入增长了1.4%。 

---
# FlashCheck: Exploration of Efficient Evidence Retrieval for Fast Fact-Checking 

**Title (ZH)**: FlashCheck：快速事实核查中高效证据检索的探索 

**Authors**: Kevin Nanekhan, Venktesh V, Erik Martin, Henrik Vatndal, Vinay Setty, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2502.05803)  

**Abstract**: The advances in digital tools have led to the rampant spread of misinformation. While fact-checking aims to combat this, manual fact-checking is cumbersome and not scalable. It is essential for automated fact-checking to be efficient for aiding in combating misinformation in real-time and at the source. Fact-checking pipelines primarily comprise a knowledge retrieval component which extracts relevant knowledge to fact-check a claim from large knowledge sources like Wikipedia and a verification component. The existing works primarily focus on the fact-verification part rather than evidence retrieval from large data collections, which often face scalability issues for practical applications such as live fact-checking. In this study, we address this gap by exploring various methods for indexing a succinct set of factual statements from large collections like Wikipedia to enhance the retrieval phase of the fact-checking pipeline. We also explore the impact of vector quantization to further improve the efficiency of pipelines that employ dense retrieval approaches for first-stage retrieval. We study the efficiency and effectiveness of the approaches on fact-checking datasets such as HoVer and WiCE, leveraging Wikipedia as the knowledge source. We also evaluate the real-world utility of the efficient retrieval approaches by fact-checking 2024 presidential debate and also open source the collection of claims with corresponding labels identified in the debate. Through a combination of indexed facts together with Dense retrieval and Index compression, we achieve up to a 10.0x speedup on CPUs and more than a 20.0x speedup on GPUs compared to the classical fact-checking pipelines over large collections. 

**Abstract (ZH)**: 数字工具的进步导致了虚假信息的猖獗传播。虽然事实核查旨在对抗这一现象，但手动事实核查既繁琐又不具有可扩展性。因此，自动事实核查对于实时并在源头上反击虚假信息至关重要。事实核查流程主要由知识检索组件和验证组件组成，其中知识检索组件从维基百科等大型知识库中提取用于核查声明的相关知识，验证组件则进行验证。现有的研究主要集中在事实验证部分，而较少关注从大规模数据集合中检索证据，这在实际应用如现场事实核查中往往面临可扩展性问题。本研究旨在填补这一空白，通过探索多种方法对大型数据集合（如维基百科）进行索引，以增强事实核查流程中的检索阶段。我们还研究了向量量化对采用密集检索方法的第一阶段检索流程效率的进一步提升作用。我们通过使用维基百科作为知识源，在HoVer和WiCE等事实核查数据集上研究了这些方法的有效性和效率。此外，我们通过对2024年总统辩论进行事实核查，评估了高效检索方法的实际应用价值，并开源了辩论中识别出的声明及其相应的标签。通过结合索引事实、密集检索以及索引压缩，我们在大规模数据集上实现了高达10倍的CPU加速以及超过20倍的GPU加速，与经典的事实核查流程相比。 

---
# Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art 

**Title (ZH)**: 基于图的向量搜索：对最新技术的实验评估 

**Authors**: Ilias Azizi, Karima Echihabi, Themis Palpanas  

**Link**: [PDF](https://arxiv.org/pdf/2502.05575)  

**Abstract**: Vector data is prevalent across business and scientific applications, and its popularity is growing with the proliferation of learned embeddings. Vector data collections often reach billions of vectors with thousands of dimensions, thus, increasing the complexity of their analysis. Vector search is the backbone of many critical analytical tasks, and graph-based methods have become the best choice for analytical tasks that do not require guarantees on the quality of the answers. We briefly survey in-memory graph-based vector search, outline the chronology of the different methods and classify them according to five main design paradigms: seed selection, incremental insertion, neighborhood propagation, neighborhood diversification, and divide-and-conquer. We conduct an exhaustive experimental evaluation of twelve state-of-the-art methods on seven real data collections, with sizes up to 1 billion vectors. We share key insights about the strengths and limitations of these methods; e.g., the best approaches are typically based on incremental insertion and neighborhood diversification, and the choice of the base graph can hurt scalability. Finally, we discuss open research directions, such as the importance of devising more sophisticated data-adaptive seed selection and diversification strategies. 

**Abstract (ZH)**: 向量数据在商业和科学应用中广泛存在，随着学习嵌入的普及，其受欢迎程度正在增长。向量数据集合通常包含数亿个向量，具有数千个维度，从而增加了其分析的复杂性。向量搜索是许多关键分析任务的核心部分，基于图的方法已成为不需要答案质量保证的分析任务的最佳选择。我们简要介绍了基于内存的向量搜索的图方法，并概述了不同方法的发展历程，根据五个主要的设计范式对其进行分类：种子选择、增量插入、邻域传播、邻域多样化和分而治之。我们对七个实际数据集合中的十二种最先进的方法进行了详尽的实验评估，这些数据集的最大规模可达1亿个向量。我们分享了这些方法的关键见解；例如，最好的方法通常基于增量插入和邻域多样化，而基图的选择可能会损害可扩展性。最后，我们讨论了开放的研究方向，例如更复杂的数据自适应种子选择和多样化策略的重要性。 

---
# Diffusion Model for Interest Refinement in Multi-Interest Recommendation 

**Title (ZH)**: 多兴趣推荐中的兴趣细化扩散模型 

**Authors**: Yankun Le, Haoran Li, Baoyuan Ou, Yinjie Qing, Zhixuan Yang, Ruilong Su, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05561)  

**Abstract**: Multi-interest candidate matching plays a pivotal role in personalized recommender systems, as it captures diverse user interests from their historical behaviors. Most existing methods utilize attention mechanisms to generate interest representations by aggregating historical item embeddings. However, these methods only capture overall item-level relevance, leading to coarse-grained interest representations that include irrelevant information. To address this issue, we propose the Diffusion Multi-Interest model (DMI), a novel framework for refining user interest representations at the dimension level. Specifically, DMI first introduces controllable noise into coarse-grained interest representations at the dimensional level. Then, in the iterative reconstruction process, DMI combines a cross-attention mechanism and an item pruning strategy to reconstruct the personalized interest vectors with the guidance of tailored collaborative information. Extensive experiments demonstrate the effectiveness of DMI, surpassing state-of-the-art methods on offline evaluations and an online A/B test. Successfully deployed in the real-world recommender system, DMI effectively enhances user satisfaction and system performance at scale, serving the major traffic of hundreds of millions of daily active users. \footnote{The code will be released for reproducibility once the paper is accepted.} 

**Abstract (ZH)**: 多兴趣候选匹配在个性化推荐系统中起着关键作用，因为它可以从用户历史行为中捕获多样的用户兴趣。目前大多数方法利用注意力机制通过聚合历史物品嵌入来生成兴趣表示。然而，这些方法仅捕获粗粒度的物品级别相关性，导致包含无关信息的兴趣表示。为解决这一问题，我们提出了一种新的框架——扩散多兴趣模型（Diffusion Multi-Interest, DMI），以在维度级别细化用户兴趣表示。具体而言，DMI 首先在维度级别引入可控的噪声，然后在迭代重构过程中，结合交叉注意力机制和物品剪枝策略，在定制的协作信息指导下重构个性化的兴趣向量。大量实验表明，DMI 在离线评估和在线 A/B 测试中均优于最先进的方法。该模型成功部署于实际的推荐系统中，显著提升了用户的满意度和系统的性能，服务于数亿活跃用户的大量流量。\[注\]：论文被接受后，代码将开源以确保可再现性。 

---
# Large Memory Network for Recommendation 

**Title (ZH)**: 大型内存网络推荐系统 

**Authors**: Hui Lu, Zheng Chai, Yuchao Zheng, Zhe Chen, Deping Xie, Peng Xu, Xun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.05558)  

**Abstract**: Modeling user behavior sequences in recommender systems is essential for understanding user preferences over time, enabling personalized and accurate recommendations for improving user retention and enhancing business values. Despite its significance, there are two challenges for current sequential modeling approaches. From the spatial dimension, it is difficult to mutually perceive similar users' interests for a generalized intention understanding; from the temporal dimension, current methods are generally prone to forgetting long-term interests due to the fixed-length input sequence. In this paper, we present Large Memory Network (LMN), providing a novel idea by compressing and storing user history behavior information in a large-scale memory block. With the elaborated online deployment strategy, the memory block can be easily scaled up to million-scale in the industry. Extensive offline comparison experiments, memory scaling up experiments, and online A/B test on Douyin E-Commerce Search (ECS) are performed, validating the superior performance of LMN. Currently, LMN has been fully deployed in Douyin ECS, serving millions of users each day. 

**Abstract (ZH)**: 在推荐系统中建模用户行为序列对于理解用户随着时间变化的偏好至关重要，这有助于提供个性化的、准确的推荐，从而提高用户留存率并增强业务价值。尽管如此，当前的序列建模方法在应对两个挑战方面仍存在不足。从空间维度来看，很难互相感知相似用户的兴趣，以实现更广泛的意图理解；从时间维度来看，由于固定长度输入序列，当前方法通常容易忘记用户的长期兴趣。本文中，我们提出了大型记忆网络（Large Memory Network, LMN），提供了一种新颖的思想，通过压缩和存储用户的历史行为信息在大规模记忆块中。借助详细的设计在线部署策略，该记忆块可以在工业环境中轻松扩展到数百万规模。我们进行了广泛的离线比较实验、记忆规模扩展实验以及抖音电子商务搜索（ECS）的在线A/B测试，验证了LMN的卓越性能。目前，LMN已在抖音ECS中全面部署，每日为数百万用户提供服务。 

---
# Adaptive Domain Scaling for Personalized Sequential Modeling in Recommenders 

**Title (ZH)**: 适应性领域缩放以实现推荐系统中的个性化序列建模 

**Authors**: Zheng Chai, Hui Lu, Di Chen, Qin Ren, Xun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.05523)  

**Abstract**: Users generally exhibit complex behavioral patterns and diverse intentions in multiple business scenarios of super applications like Douyin, presenting great challenges to current industrial multi-domain recommenders. To mitigate the discrepancies across diverse domains, researches and industrial practices generally emphasize sophisticated network structures to accomodate diverse data distributions, while neglecting the inherent understanding of user behavioral sequence from the multi-domain perspective. In this paper, we present Adaptive Domain Scaling (ADS) model, which comprehensively enhances the personalization capability in target-aware sequence modeling across multiple domains. Specifically, ADS comprises of two major modules, including personalized sequence representation generation (PSRG) and personalized candidate representation generation (PCRG). The modules contribute to the tailored multi-domain learning by dynamically learning both the user behavioral sequence item representation and the candidate target item representation under different domains, facilitating adaptive user intention understanding. Experiments are performed on both a public dataset and two billion-scaled industrial datasets, and the extensive results verify the high effectiveness and compatibility of ADS. Besides, we conduct online experiments on two influential business scenarios including Douyin Advertisement Platform and Douyin E-commerce Service Platform, both of which show substantial business improvements. Currently, ADS has been fully deployed in many recommendation services at ByteDance, serving billions of users. 

**Abstract (ZH)**: 在诸如抖音这样的超级应用程序的多个商业场景中，用户通常表现出复杂的行为模式和多样的意图，这给当前工业多域推荐系统带来了极大的挑战。为了缓解不同域之间的差异，研究和工业实践通常强调使用复杂的网络结构来适应各种数据分布，而忽视了从多域视角理解用户行为序列的固有理解。在本文中，我们提出了一种自适应域扩展（ADS）模型，该模型在多域目标感知序列建模中全面提升个性化能力。具体来说，ADS 包括两个主要模块：个性化序列表示生成（PSRG）和个人化候选表示生成（PCRG）。这些模块通过在不同域下动态学习用户行为序列项表示和候选目标项表示，实现定制化的多域学习，从而提升适应性用户意图理解。我们在一个公开数据集和两个亿级规模的工业数据集上进行了实验，广泛的结果验证了ADS的高度有效性和兼容性。此外，我们在抖音广告平台和抖音电商服务平台等两个重要商业场景中进行了在线实验，这两项实验都显示出显著的商业提升。目前，ADS 已经全面部署在字节跳动的各种推荐服务中，服务于数亿用户。 

---
# Hypencoder: Hypernetworks for Information Retrieval 

**Title (ZH)**: Hypencoder：用于信息检索的超网络 

**Authors**: Julian Killingback, Hansi Zeng, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2502.05364)  

**Abstract**: The vast majority of retrieval models depend on vector inner products to produce a relevance score between a query and a document. This naturally limits the expressiveness of the relevance score that can be employed. We propose a new paradigm, instead of producing a vector to represent the query we produce a small neural network which acts as a learned relevance function. This small neural network takes in a representation of the document, in this paper we use a single vector, and produces a scalar relevance score. To produce the little neural network we use a hypernetwork, a network that produce the weights of other networks, as our query encoder or as we call it a Hypencoder. Experiments on in-domain search tasks show that Hypencoder is able to significantly outperform strong dense retrieval models and has higher metrics then reranking models and models an order of magnitude larger. Hypencoder is also shown to generalize well to out-of-domain search tasks. To assess the extent of Hypencoder's capabilities, we evaluate on a set of hard retrieval tasks including tip-of-the-tongue retrieval and instruction-following retrieval tasks and find that the performance gap widens substantially compared to standard retrieval tasks. Furthermore, to demonstrate the practicality of our method we implement an approximate search algorithm and show that our model is able to search 8.8M documents in under 60ms. 

**Abstract (ZH)**: 大多数检索模型依赖向量内积来生成查询与文档之间的相关性评分。这自然地限制了可以使用的相关性评分的表达能力。我们提出了一种新的范式，而不是生成一个向量来表示查询，我们生成一个小型的神经网络，作为学习的相关函数。这个小型神经网络接收文档的表示，本文中我们使用单个向量，并生成一个标量相关性评分。为了生成这个小型神经网络，我们使用了一种超网络（hypernetwork），即产生其他网络权重的网络，作为查询编码器或我们称之为Hypencoder。在领域相关检索任务上的实验表明，Hypencoder能够显著优于强大的密集检索模型，并且在重新排序模型和大一个数量级的模型方面具有更高的指标。此外，Hypencoder也被证明在跨领域检索任务上具有很好的泛化能力。为了评估Hypencoder的能力，我们在一系列困难的检索任务上进行了评估，包括“舌尖上的词汇”检索和指令跟随检索任务，发现与标准检索任务相比，性能差距显著扩大。此外，为了证明我们方法的实用性，我们实现了近似搜索算法，并展示了我们的模型能够在60毫秒内搜索超过8.8百万份文档。 

---
# RSAttAE: An Information-Aware Attention-based Autoencoder Recommender System 

**Title (ZH)**: RSAttAE：一种信息感知注意力自编码推荐系统 

**Authors**: Amirhossein Dadashzadeh Taromi, Sina Heydari, Mohsen Hooshmand, Majid Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2502.06705)  

**Abstract**: Recommender systems play a crucial role in modern life, including information retrieval, the pharmaceutical industry, retail, and entertainment. The entertainment sector, in particular, attracts significant attention and generates substantial profits. This work proposes a new method for predicting unknown user-movie ratings to enhance customer satisfaction. To achieve this, we utilize the MovieLens 100K dataset. Our approach introduces an attention-based autoencoder to create meaningful representations and the XGBoost method for rating predictions. The results demonstrate that our proposal outperforms most of the existing state-of-the-art methods. Availability: this http URL 

**Abstract (ZH)**: 推荐系统在现代生活中扮演着至关重要的角色，包括信息检索、制药行业、零售业和娱乐业。特别是在娱乐领域，它吸引了大量关注并产生了显著的经济效益。本研究提出了一种新的方法，用于预测未知用户对电影的评分，以增强顾客满意度。为实现这一目标，我们使用了MovieLens 100K数据集。我们提出的方法引入了基于注意力机制的自编码器来生成有意义的表示，并结合了XGBoost方法进行评分预测。结果表明，我们的方法在大多数现有的先进方法中表现更优。可用性：[此处填写链接] 

---
# FunduSAM: A Specialized Deep Learning Model for Enhanced Optic Disc and Cup Segmentation in Fundus Images 

**Title (ZH)**: FunduSAM：一种专门用于眼底图像视盘和杯状凹陷增强分割的深度学习模型 

**Authors**: Jinchen Yu, Yongwei Nie, Fei Qi, Wenxiong Liao, Hongmin Cai  

**Link**: [PDF](https://arxiv.org/pdf/2502.06220)  

**Abstract**: The Segment Anything Model (SAM) has gained popularity as a versatile image segmentation method, thanks to its strong generalization capabilities across various domains. However, when applied to optic disc (OD) and optic cup (OC) segmentation tasks, SAM encounters challenges due to the complex structures, low contrast, and blurred boundaries typical of fundus images, leading to suboptimal performance. To overcome these challenges, we introduce a novel model, FunduSAM, which incorporates several Adapters into SAM to create a deep network specifically designed for OD and OC segmentation. The FunduSAM utilizes Adapter into each transformer block after encoder for parameter fine-tuning (PEFT). It enhances SAM's feature extraction capabilities by designing a Convolutional Block Attention Module (CBAM), addressing issues related to blurred boundaries and low contrast. Given the unique requirements of OD and OC segmentation, polar transformation is used to convert the original fundus OD images into a format better suited for training and evaluating FunduSAM. A joint loss is used to achieve structure preservation between the OD and OC, while accurate segmentation. Extensive experiments on the REFUGE dataset, comprising 1,200 fundus images, demonstrate the superior performance of FunduSAM compared to five mainstream approaches. 

**Abstract (ZH)**: 段落的任何一部分都翻译成中文如下：

段落标题翻译：
《任何段落的具体标题或研究内容》（FunduSAM：一种专为视盘（OD）和视杯（OC）分割设计的模型）

段落正文翻译：
段Anything 模型（SAM）因其在各种领域中表现出的强大泛化能力而广受欢迎。然而，当应用于视盘（OD）和视杯（OC）分割任务时，由于底片图像中复杂结构、低对比度和模糊边界的特征，SAM会遇到挑战，导致其性能不尽理想。为了解决这些问题，我们提出了一种名为 FunduSAM 的新型模型，它通过将若干 Adapters 集成到 SAM 中，创建了一个专门用于 OD 和 OC 分割的深度网络。FunduSAM 在每个编码器后的变压器块中引入了 Adapters，以实现参数微调（PEFT）。通过设计卷积块注意力模块（CBAM），它增强了 SAM 的特征提取能力，解决了边缘模糊和低对比度的问题。鉴于 OD 和 OC 分割的特殊需求，我们使用极坐标变换将原始的底片视盘图像转换为更适合训练和评估 FunduSAM 的格式。同时，采用联合损失函数来实现 OD 和 OC 结构的保留与精确分割。在 REFUGE 数据集中进行的大量实验（该数据集包含 1,200 张底片图像）表明，与五种主流方法相比，FunduSAM 的性能更为优越。 

---
# Optimizing Knowledge Integration in Retrieval-Augmented Generation with Self-Selection 

**Title (ZH)**: 在检索增强生成中通过自我选择优化知识集成 

**Authors**: Yan Weng, Fengbin Zhu, Tong Ye, Haoyan Liu, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2502.06148)  

**Abstract**: Retrieval-Augmented Generation (RAG), which integrates external knowledge into Large Language Models (LLMs), has proven effective in enabling LLMs to produce more accurate and reliable responses. However, it remains a significant challenge how to effectively integrate external retrieved knowledge with internal parametric knowledge in LLMs. In this work, we propose a novel Self-Selection RAG framework, where the LLM is made to select from pairwise responses generated with internal parametric knowledge solely and with external retrieved knowledge together to achieve enhanced accuracy. To this end, we devise a Self-Selection-RGP method to enhance the capabilities of the LLM in both generating and selecting the correct answer, by training the LLM with Direct Preference Optimization (DPO) over a curated Retrieval Generation Preference (RGP) dataset. Experimental results with two open-source LLMs (i.e., Llama2-13B-Chat and Mistral-7B) well demonstrate the superiority of our approach over other baseline methods on Natural Questions (NQ) and TrivialQA datasets. 

**Abstract (ZH)**: 检索增强生成（RAG），即将外部知识整合到大型语言模型（LLM）中，已被证明能够有效提高LLM生成更准确和可靠响应的能力。然而，如何有效地将外部检索到的知识与LLM内部参数化的知识进行整合仍是一个重大挑战。在此项工作中，我们提出了一种新的自选型RAG框架，在该框架中，LLM被设计为从仅使用内部参数化知识生成的回答和结合外部检索知识生成的回答中选择，从而实现增强的准确性。为此，我们设计了一种自选型RGP方法，通过使用直接偏好优化（DPO）在经过精心策划的检索生成偏好（RGP）数据集上训练LLM，以增强LLM在生成和选择正确答案方面的能力。使用两个开源LLM（即Llama2-13B-Chat和Mistral-7B）进行的实验结果表明，与基准方法相比，我们的方法在自然问题（NQ）和TrivialQA数据集上具有明显优势。 

---
# Benchmarking Prompt Sensitivity in Large Language Models 

**Title (ZH)**: 大型语言模型中提示敏感性的基准测试 

**Authors**: Amirhossein Razavi, Mina Soltangheis, Negar Arabzadeh, Sara Salamat, Morteza Zihayat, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2502.06065)  

**Abstract**: Large language Models (LLMs) are highly sensitive to variations in prompt formulation, which can significantly impact their ability to generate accurate responses. In this paper, we introduce a new task, Prompt Sensitivity Prediction, and a dataset PromptSET designed to investigate the effects of slight prompt variations on LLM performance. Using TriviaQA and HotpotQA datasets as the foundation of our work, we generate prompt variations and evaluate their effectiveness across multiple LLMs. We benchmark the prompt sensitivity prediction task employing state-of-the-art methods from related tasks, including LLM-based self-evaluation, text classification, and query performance prediction techniques. Our findings reveal that existing methods struggle to effectively address prompt sensitivity prediction, underscoring the need to understand how information needs should be phrased for accurate LLM responses. 

**Abstract (ZH)**: 以下是将原文翻译成中文的版本，符合学术规范：

大型语言模型（LLMs）对提示（Prompt）制定的微小变化极为敏感，这些变化可能显著影响其生成准确响应的能力。本文介绍了一个新的任务，即提示敏感性预测（Prompt Sensitivity Prediction），以及一个名为PromptSET的数据集，旨在研究微小提示变化对LLM性能的影响。基于TriviaQA和HotpotQA数据集，我们生成了提示变化并评估了这些变化在多个LLM上的有效性。我们使用相关任务中的先进方法，包括基于LLM的自我评估、文本分类和查询性能预测技术，来基准测试提示敏感性预测任务。我们的研究发现现有的方法难以有效解决提示敏感性预测问题，强调了理解如何以明确方式表述信息需求的重要性，以获得准确的LLM响应。 

---
# Multi-Branch Collaborative Learning Network for Video Quality Assessment in Industrial Video Search 

**Title (ZH)**: 工业视频搜索中基于多分支协作学习网络的视频质量评估 

**Authors**: Hengzhu Tang, Zefeng Zhang, Zhiping Li, Zhenyu Zhang, Xing Wu, Li Gao, Suqi Cheng, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.05924)  

**Abstract**: Video Quality Assessment (VQA) is vital for large-scale video retrieval systems, aimed at identifying quality issues to prioritize high-quality videos. In industrial systems, low-quality video characteristics fall into four categories: visual-related issues like mosaics and black boxes, textual issues from video titles and OCR content, and semantic issues like frame incoherence and frame-text mismatch from AI-generated videos. Despite their prevalence in industrial settings, these low-quality videos have been largely overlooked in academic research, posing a challenge for accurate identification. To address this, we introduce the Multi-Branch Collaborative Network (MBCN) tailored for industrial video retrieval systems. MBCN features four branches, each designed to tackle one of the aforementioned quality issues. After each branch independently scores videos, we aggregate these scores using a weighted approach and a squeeze-and-excitation mechanism to dynamically address quality issues across different scenarios. We implement point-wise and pair-wise optimization objectives to ensure score stability and reasonableness. Extensive offline and online experiments on a world-level video search engine demonstrate MBCN's effectiveness in identifying video quality issues, significantly enhancing the retrieval system's ranking performance. Detailed experimental analyses confirm the positive contribution of all four evaluation branches. Furthermore, MBCN significantly improves recognition accuracy for low-quality AI-generated videos compared to the baseline. 

**Abstract (ZH)**: 视频质量评估（VQA）对于大规模视频检索系统至关重要，旨在识别质量问题以优先处理高质量视频。在工业系统中，低质量视频特征可以归入四类：视觉相关问题，如马赛克和黑框；视频标题和OCR内容中的文本问题；以及人工智能生成视频中的语义问题，如帧不一致和帧-文本不匹配。尽管这些低质量视频在工业环境中极为常见，但它们在学术研究中却很少被关注，这给准确识别带来了挑战。为了解决这一问题，我们引入了一种针对工业视频检索系统的多分支协作网络（MBCN）。MBCN 包含四个分支，每个分支专门针对上述的一种质量问题。在每个分支独立对视频进行打分后，我们通过加权方法和挤压-注意力机制来综合这些分数，以动态地解决不同场景中的质量问题。我们实施了点间和对间的优化目标，以确保评分的稳定性和合理性。在世界级视频搜索引擎上进行的大量离线和在线实验表明，MBCN 在识别视频质量问题方面非常有效，显著提高了检索系统的排名性能。详细的实验分析表明，所有四个评估分支都对整体性能产生了积极的贡献。此外，MBCN 在识别低质量AI生成视频方面的准确性显著高于基线方法。 

---
# LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification 

**Title (ZH)**: LegalSeg：通过修辞角色分类解锁印度法律判决的结构 

**Authors**: Shubham Kumar Nigam, Tanmay Dubey, Govind Sharma, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2502.05836)  

**Abstract**: In this paper, we address the task of semantic segmentation of legal documents through rhetorical role classification, with a focus on Indian legal judgments. We introduce LegalSeg, the largest annotated dataset for this task, comprising over 7,000 documents and 1.4 million sentences, labeled with 7 rhetorical roles. To benchmark performance, we evaluate multiple state-of-the-art models, including Hierarchical BiLSTM-CRF, TransformerOverInLegalBERT (ToInLegalBERT), Graph Neural Networks (GNNs), and Role-Aware Transformers, alongside an exploratory RhetoricLLaMA, an instruction-tuned large language model. Our results demonstrate that models incorporating broader context, structural relationships, and sequential sentence information outperform those relying solely on sentence-level features. Additionally, we conducted experiments using surrounding context and predicted or actual labels of neighboring sentences to assess their impact on classification accuracy. Despite these advancements, challenges persist in distinguishing between closely related roles and addressing class imbalance. Our work underscores the potential of advanced techniques for improving legal document understanding and sets a strong foundation for future research in legal NLP. 

**Abstract (ZH)**: 在这篇论文中，我们通过修辞角色分类任务探讨了法律文件的语义分割问题，重点是我国法律判决书。我们引入了LegalSeg，这是迄今为止针对该任务的最大规模标注数据集，包含超过7,000份文档和140万句话，标注了7种修辞角色。为了评估性能，我们测试了多种最新的模型，包括层次BiLSTM-CRF、TransformerOverInLegalBERT（ToInLegalBERT）、图神经网络（GNNs）和感知修辞角色的Transformer模型，以及一个探索性的RhetoricLLaMA，即经过指令调优的大语言模型。我们的结果表明，能够利用更广泛的上下文、结构关系和序列句子信息的模型优于仅仅依赖句内特征的模型。此外，我们还使用了相邻句子的上下文和预测或实际标签进行实验，以评估其对分类准确性的影响。尽管取得了这些进展，但在区分密切相关的角色和解决类别不平衡问题方面仍面临挑战。我们的工作强调了高级技术在提高法律文件理解方面的潜力，并为未来法律自然语言处理研究奠定了坚实的基础。 

---
# A Tutorial On Intersectionality in Fair Rankings 

**Title (ZH)**: 《关于公平排名中多重交集性的教程》 

**Authors**: Chiara Criscuolo, Davide Martinenghi, Giuseppe Piccirillo  

**Link**: [PDF](https://arxiv.org/pdf/2502.05333)  

**Abstract**: We address the critical issue of biased algorithms and unfair rankings, which have permeated various sectors, including search engines, recommendation systems, and workforce management. These biases can lead to discriminatory outcomes in a data-driven world, especially against marginalized and underrepresented groups. Efforts towards responsible data science and responsible artificial intelligence aim to mitigate these biases and promote fairness, diversity, and transparency. However, most fairness-aware ranking methods singularly focus on protected attributes such as race, gender, or socio-economic status, neglecting the intersectionality of these attributes, i.e., the interplay between multiple social identities. Understanding intersectionality is crucial to ensure that existing inequalities are not preserved by fair rankings. We offer a description of the main ways to incorporate intersectionality in fair ranking systems through practical examples and provide a comparative overview of existing literature and a synoptic table summarizing the various methodologies. Our analysis highlights the need for intersectionality to attain fairness, while also emphasizing that fairness, alone, does not necessarily imply intersectionality. 

**Abstract (ZH)**: 我们探讨了一个关键问题，即偏向性算法和不公平排名已经渗透到包括搜索引擎、推荐系统和劳动力管理在内的各个领域。这些偏向性可能导致在数据驱动的世界中产生歧视性结果，尤其是针对边缘化和代表性不足的群体。负责任的数据科学和负责任的人工智能努力旨在减轻这些偏见，促进公平、多样性和透明度。然而，大多数面向公平性的排名方法主要关注种族、性别或社会经济地位等保护属性，而忽视了这些属性的交叉性，即多种社会身份之间的相互作用。理解交叉性对于确保现有的不平等不会被公平排名所保留至关重要。我们通过实用示例描述了将交叉性纳入公平排名系统的几种主要方法，并提供了现有文献的比较综述和概述性表格，总结了各种方法论。我们的分析强调了实现公平性所需的交叉性，同时也强调了公平性本身并不一定意味着交叉性。 

---
# Efficient Knowledge Feeding to Language Models: A Novel Integrated Encoder-Decoder Architecture 

**Title (ZH)**: 高效的知识喂入语言模型：一种新颖的集成编码-解码架构 

**Authors**: S Santosh Kumar, Rishi Gottimukkala, Supriya Devidutta, Karthikeyan S  

**Link**: [PDF](https://arxiv.org/pdf/2502.05233)  

**Abstract**: This paper introduces a novel approach to efficiently feeding knowledge to language models (LLMs) during prediction by integrating retrieval and generation processes within a unified framework. While the Retrieval-Augmented Generation (RAG) model addresses gaps in LLMs' training data and knowledge limits, it is hindered by token limit restrictions and dependency on the retrieval system's accuracy. Our proposed architecture incorporates in-context vectors (ICV) to overcome these challenges. ICV recasts in-context learning by using latent embeddings of LLMs to create a vector that captures essential task information. This vector is then used to shift the latent states of the LLM, enhancing the generation process without adding demonstration examples to the prompt. ICV directly integrates information into the model, enabling it to process this information more effectively. Our extensive experimental evaluation demonstrates that ICV outperforms standard in-context learning and fine-tuning across question-answering, information retrieval, and other tasks. This approach mitigates the limitations of current RAG models and offers a more robust solution for handling extensive and diverse datasets. Despite leveraging a fraction of the parameters, our ICV-enhanced model achieves competitive performance against models like LLaMA-3, Gemma, and Phi-3, significantly reducing computational costs and memory requirements. ICV reduces prompt length, is easy to control, surpasses token limitations, and is computationally efficient compared to fine-tuning. 

**Abstract (ZH)**: 本文提出了一种新颖的方法，通过将检索和生成过程统一在一个框架中，以提高语言模型（LLM）在预测过程中高效地获取知识的方式。虽然检索增强生成（RAG）模型能够解决LLM训练数据和知识限制的问题，但它受限于标记数量的限制，并且依赖于检索系统的准确性。我们提出的架构引入了上下文向量（ICV），以克服这些挑战。ICV 通过使用LLM的潜在嵌入来重构上下文学习，创建一个能够捕捉任务关键信息的向量。然后，该向量被用于调整LLM的潜在状态，从而增强生成过程，而无需在提示中增加演示示例。ICV 直接将信息整合到模型中，使模型能够更有效地处理这些信息。我们的广泛实验评估表明，ICV 在问答、信息检索及其他任务上都优于标准的上下文学习和微调方法，从而缓解了当前RAG模型的局限性，并提供了一个更为稳健的解决方案，以处理大量和多样化的数据集。尽管利用了模型参数的一小部分，我们的ICV增强模型在性能上与LLaMA-3、Gemma和Phi-3等模型相当，显著降低了计算成本和内存需求。ICV 减少了提示长度，易于控制，超过了标记数量的限制，并且相比于微调，在计算效率上更具优势。 

---
