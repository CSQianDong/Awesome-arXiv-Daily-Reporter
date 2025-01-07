# LightGNN: Simple Graph Neural Network for Recommendation 

**Title (ZH)**: LightGNN：简单的图神经网络推荐模型 

**Authors**: Guoxuan Chen, Lianghao Xia, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.03228)  

**Abstract**: Graph neural networks (GNNs) have demonstrated superior performance in collaborative recommendation through their ability to conduct high-order representation smoothing, effectively capturing structural information within users' interaction patterns. However, existing GNN paradigms face significant challenges in scalability and robustness when handling large-scale, noisy, and real-world datasets. To address these challenges, we present LightGNN, a lightweight and distillation-based GNN pruning framework designed to substantially reduce model complexity while preserving essential collaboration modeling capabilities. Our LightGNN framework introduces a computationally efficient pruning module that adaptively identifies and removes redundant edges and embedding entries for model compression. The framework is guided by a resource-friendly hierarchical knowledge distillation objective, whose intermediate layer augments the observed graph to maintain performance, particularly in high-rate compression scenarios. Extensive experiments on public datasets demonstrate LightGNN's effectiveness, significantly improving both computational efficiency and recommendation accuracy. Notably, LightGNN achieves an 80% reduction in edge count and 90% reduction in embedding entries while maintaining performance comparable to more complex state-of-the-art baselines. The implementation of our LightGNN framework is available at the github repository: this https URL. 

**Abstract (ZH)**: 图神经网络（GNNs）在通过其高阶表示平滑能力在协同推荐中展示了优越的性能，能够有效地捕捉用户交互模式中的结构信息。然而，现有的GNN范式在处理大规模、嘈杂、真实世界数据集时面临显著的可扩展性和稳健性挑战。为了解决这些挑战，我们提出了LightGNN，这是一种轻量级且基于蒸馏的GNN剪枝框架，旨在大幅减少模型复杂度同时保留核心的协同建模能力。我们的LightGNN框架引入了一个计算高效的学习剪枝模块，该模块能够自适应地识别并移除冗余边和嵌入项，从而实现模型压缩。该框架由一种资源友好的分级知识蒸馏目标引导，其中间层通过增强观测到的图来保持性能，特别是在高压缩率场景中。在公共数据集上的广泛实验表明，LightGNN的有效性显著提高了计算效率和推荐精度。值得注意的是，LightGNN在保持与更复杂的状态-of-the-art基准相当的性能的同时，实现了边数80%的减少和嵌入项90%的减少。我们的LightGNN框架的实现可在GitHub仓库中获得：[此链接].

注：由于Markdown的限制，您可能需要将上述代码块中的链接替换为实际的GitHub链接。 

---
# Personalized Fashion Recommendation with Image Attributes and Aesthetics Assessment 

**Title (ZH)**: 基于图像属性和美学评估的个性化服装推荐 

**Authors**: Chongxian Chen, Fan Mo, Xin Fan, Hayato Yamana  

**Link**: [PDF](https://arxiv.org/pdf/2501.03085)  

**Abstract**: Personalized fashion recommendation is a difficult task because 1) the decisions are highly correlated with users' aesthetic appetite, which previous work frequently overlooks, and 2) many new items are constantly rolling out that cause strict cold-start problems in the popular identity (ID)-based recommendation methods. These new items are critical to recommend because of trend-driven consumerism. In this work, we aim to provide more accurate personalized fashion recommendations and solve the cold-start problem by converting available information, especially images, into two attribute graphs focusing on optimized image utilization and noise-reducing user modeling. Compared with previous methods that separate image and text as two components, the proposed method combines image and text information to create a richer attributes graph. Capitalizing on the advancement of large language and vision models, we experiment with extracting fine-grained attributes efficiently and as desired using two different prompts. Preliminary experiments on the IQON3000 dataset have shown that the proposed method achieves competitive accuracy compared with baselines. 

**Abstract (ZH)**: 个性化时装推荐是一项具有挑战性的工作，因为1）用户的审美偏好与决策高度相关，而 previous work 时常忽视这一点；2）不断有大量新商品推出，这导致基于用户ID的传统推荐方法在冷启动（cold-start）问题上表现严苛。由于受到趋势驱动的消费行为的影响，这些新商品至关重要。为此，我们旨在通过将可用信息，尤其是图像信息，转换为两个属性图来提供更准确的个性化时尚推荐，并解决冷启动问题。这两大属性图着重于优化图像利用和减少用户建模中的噪声。与以往将图像和文本分离开来处理的方法不同，我们提出的方法将图像和文本信息结合起来，形成一个更为丰富的属性图。借助大型语言模型和视觉模型的进步，我们尝试使用两种不同的提示高效、灵活地提取细粒度的属性。在使用IQON3000数据集进行的初步实验中，我们提出的方法在准确度上与基线方法相当。 

---
# OpenTable data with multi-criteria ratings 

**Title (ZH)**: 带有多个评价标准的OpenTable数据 

**Authors**: Yong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.03072)  

**Abstract**: With the development of recommender systems (RSs), several promising systems have emerged, such as context-aware RS, multi-criteria RS, and group RS. Multi-criteria recommender systems (MCRSs) are designed to provide personalized recommendations by considering user preferences in multiple attributes or criteria simultaneously. Unlike traditional RSs that typically focus on a single rating, these systems help users make more informed decisions by considering their diverse preferences and needs across various dimensions. In this article, we release the OpenTable data set which was crawled from this http URL. The data set can be considered as a benchmark data set for multi-criteria recommendations. 

**Abstract (ZH)**: 随着推荐系统（RS）的发展，已经涌现出多种有前景的系统，如情境感知推荐系统、多准则推荐系统和群组推荐系统。多准则推荐系统（MCRS）旨在通过同时考虑用户的多种属性或多个准则是来提供个性化推荐。与传统推荐系统通常仅侧重单一评分不同，这类系统有助于用户在多个维度上考虑其多样化的偏好和需求，从而做出更加明智的决策。在本文中，我们发布了一个从该网址爬取的OpenTable数据集，该数据集可被视为多准则推荐的基准数据集。 

---
# FlipedRAG: Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models 

**Title (ZH)**: FlipedRAG：针对大型语言模型检索增强生成的黑箱意见操纵攻击 

**Authors**: Zhuo Chen, Yuyang Gong, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu, Jiawei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.02968)  

**Abstract**: Retrieval-Augmented Generation (RAG) addresses hallucination and real-time constraints by dynamically retrieving relevant information from a knowledge database to supplement the LLMs' input. When presented with a query, RAG selects the most semantically similar texts from its knowledge bases and uses them as context for the LLMs to generate more accurate responses. RAG also creates a new attack surface, especially since RAG databases are frequently sourced from public domains. While existing studies have predominantly focused on optimizing RAG's performance and efficiency, emerging research has begun addressing the security concerns associated with RAG. However, these works have some limitations, typically focusing on either white-box methodologies or heuristic-based black-box attacks. Furthermore, prior research has mainly targeted simple factoid question answering, which is neither practically challenging nor resistant to correction. In this paper, we unveil a more realistic and threatening scenario: opinion manipulation for controversial topics against RAG. Particularly, we propose a novel RAG black-box attack method, termed FlipedRAG, which is transfer-based. By leveraging instruction engineering, we obtain partial retrieval model outputs from black-box RAG system, facilitating the training of surrogate models to enhance the effectiveness of opinion manipulation attack. Extensive experimental results confirms that our approach significantly enhances the average success rate of opinion manipulation by 16.7%. It achieves an average of a 50% directional change in the opinion polarity of RAG responses across four themes. Additionally, it induces a 20% shift in user cognition. Furthermore, we discuss the efficacy of potential defense mechanisms and conclude that they are insufficient in mitigating this type of attack, highlighting the urgent need to develop novel defensive strategies. 

**Abstract (ZH)**: 检索增强生成（RAG）通过从知识数据库动态检索相关信息来补充语言模型（LLM）的输入，从而解决了幻觉和实时性限制的问题。在接收到查询时，RAG 会从其知识库中选择最具语义相似性的文本，并将其作为上下文来生成更准确的响应。同时，RAG 也创建了一个新的攻击面，特别是在 RAG 数据库经常来源于公开领域的情况下。尽管现有的研究主要集中在优化 RAG 的性能和效率上，但新兴的研究已经开始关注与 RAG 相关的安全问题。然而，这些工作存在一定的局限性，通常侧重于白盒方法或启发式黑盒攻击。此外，先前的研究主要针对简单的事实性问题回答，这些问题既不实际困难，也不具有改正性。在本文中，我们揭示了一个更现实且更具威胁性的场景：针对 RAG 的观点操纵，特别是出于争议性话题的考虑。我们提出了一种新颖的基于转移的学习 RAG 黑盒攻击方法，称为 FlipedRAG。通过利用指令工程，我们从黑盒 RAG 系统中获得了部分检索模型输出，这有助于培训替代模型以增强观点操纵攻击的有效性。大量实验证明，我们的方法显著提高了观点操纵的平均成功率 16.7%。它在四个主题的响应中实现了平均 50% 的观点极性方向性变化，并且引发了 20% 的用户认知偏移。此外，我们讨论了潜在防御机制的有效性，并得出结论认为这些防御机制不足以缓解这种攻击，突显了开发新型防御策略的紧迫需求。 

---
# Foundations of GenIR 

**Title (ZH)**: “GenIR的理论基础”

在这个翻译中，“Foundations”被译为“理论基础”，这是学术论文中常用的表达方式，能够准确地传达出原文的意思。如果需要更具体的背景或定义，可以进一步提供相关信息以便进行更精确的翻译。 

**Authors**: Qingyao Ai, Jingtao Zhan, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.02842)  

**Abstract**: The chapter discusses the foundational impact of modern generative AI models on information access (IA) systems. In contrast to traditional AI, the large-scale training and superior data modeling of generative AI models enable them to produce high-quality, human-like responses, which brings brand new opportunities for the development of IA paradigms. In this chapter, we identify and introduce two of them in details, i.e., information generation and information synthesis. Information generation allows AI to create tailored content addressing user needs directly, enhancing user experience with immediate, relevant outputs. Information synthesis leverages the ability of generative AI to integrate and reorganize existing information, providing grounded responses and mitigating issues like model hallucination, which is particularly valuable in scenarios requiring precision and external knowledge. This chapter delves into the foundational aspects of generative models, including architecture, scaling, and training, and discusses their applications in multi-modal scenarios. Additionally, it examines the retrieval-augmented generation paradigm and other methods for corpus modeling and understanding, demonstrating how generative AI can enhance information access systems. It also summarizes potential challenges and fruitful directions for future studies. 

**Abstract (ZH)**: 本章讨论了现代生成型AI模型对信息访问（Information Access, IA）系统的基础性影响。与传统AI相比，生成型AI模型的大规模训练和卓越的数据建模能力使其能够生产出高质量、类人的响应，从而为IA范式的创新发展带来了全新的机遇。在本章中，我们将详细阐述并介绍其中的两种范式，即信息生成（Information Generation）和信息合成（Information Synthesis）。信息生成使AI能够直接根据用户需求生成定制化的內容，提升用户体验，提供即时的相关输出。信息合成则利用生成型AI整合和重组现有信息的能力，提供基于实际的数据响应，并减轻诸如模型幻觉等问题，特别是在需要精确性和外部知识的情景中尤为重要。本章深入探讨了生成模型的基础方面，包括架构、扩展和训练，并讨论了其在多模态场景中的应用。此外，本章还探讨了检索增强生成范式以及其他用于语料库建模和理解的方法，展示了生成型AI如何增强信息访问系统。最后，本章还总结了未来研究中可能面临的挑战和富有成效的研究方向。 

---
# Improving GenIR Systems Based on User Feedback 

**Title (ZH)**: 基于用户反馈改善生成式信息检索系统 

**Authors**: Qingyao Ai, Zhicheng Dou, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02838)  

**Abstract**: In this chapter, we discuss how to improve the GenIR systems based on user feedback. Before describing the approaches, it is necessary to be aware that the concept of "user" has been extended in the interactions with the GenIR systems. Different types of feedback information and strategies are also provided. Then the alignment techniques are highlighted in terms of objectives and methods. Following this, various ways of learning from user feedback in GenIR are presented, including continual learning, learning and ranking in the conversational context, and prompt learning. Through this comprehensive exploration, it becomes evident that innovative techniques are being proposed beyond traditional methods of utilizing user feedback, and contribute significantly to the evolution of GenIR in the new era. We also summarize some challenging topics and future directions that require further investigation. 

**Abstract (ZH)**: 在本章中，我们讨论了基于用户反馈提升生成式交互系统（GenIR）的方法。在描述这些方法之前，需要意识到“用户”这一概念在与GenIR系统的交互中已被拓展。此外，还提供了不同类型的反馈信息和策略。然后，我们强调了在目标和方法方面对对齐技术的重要性。随后，我们介绍了从用户反馈中在GenIR中学习的不同方式，包括持续学习、在对话上下文中的学习与排名，以及提示学习。通过这一全面的探索，可以看出，正在提出超越传统方法的创新技术，这些技术对GenIR在新时代的发展具有重要意义。我们还总结了一些有待进一步研究的挑战性主题和未来方向。 

---
# GeAR: Generation Augmented Retrieval 

**Title (ZH)**: GeAR：生成增强的检索 

**Authors**: Haoyu Liu, Shaohan Huang, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Weiwei Deng, Feng Sun, Furu Wei, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02772)  

**Abstract**: Document retrieval techniques form the foundation for the development of large-scale information systems. The prevailing methodology is to construct a bi-encoder and compute the semantic similarity. However, such scalar similarity is difficult to reflect enough information and impedes our comprehension of the retrieval results. In addition, this computational process mainly emphasizes the global semantics and ignores the fine-grained semantic relationship between the query and the complex text in the document. In this paper, we propose a new method called $\textbf{Ge}$neration $\textbf{A}$ugmented $\textbf{R}$etrieval ($\textbf{GeAR}$) that incorporates well-designed fusion and decoding modules. This enables GeAR to generate the relevant text from documents based on the fused representation of the query and the document, thus learning to "focus on" the fine-grained information. Also when used as a retriever, GeAR does not add any computational burden over bi-encoders. To support the training of the new framework, we have introduced a pipeline to efficiently synthesize high-quality data by utilizing large language models. GeAR exhibits competitive retrieval and localization performance across diverse scenarios and datasets. Moreover, the qualitative analysis and the results generated by GeAR provide novel insights into the interpretation of retrieval results. The code, data, and models will be released after completing technical review to facilitate future research. 

**Abstract (ZH)**: 文档检索技术是大规模信息系统开发的基础。当前主流的方法是构建双编码器并计算语义相似度。然而，这种标量相似度难以充分反映信息，并阻碍了我们对检索结果的理解。此外，这一计算过程主要关注全局语义，而忽略了查询与文档中复杂文本之间的细粒度语义关系。在本文中，我们提出了一种名为**Ge**neration **A**ugmented **R**etrieval（简称**GeAR**）的新方法，该方法结合了精心设计的融合和解码模块。这使得GeAR能够在融合查询和文档表示的基础上生成相关的文本，从而学会“聚焦”于细粒度的信息。当作为检索器使用时，GeAR也不会增加任何额外的计算负担。为了支持新框架的训练，我们引入了一个管道，通过利用大型语言模型高效地合成高质量数据。GeAR在多种场景和数据集上展示了具有竞争力的检索和定位性能。此外，GeAR 的定性分析及其生成的结果为检索结果的解释提供了新的见解。在完成技术审查后，我们将发布代码、数据和模型，以促进未来的研究。 

---
# Tree-based RAG-Agent Recommendation System: A Case Study in Medical Test Data 

**Title (ZH)**: 基于树状结构的RAG代理推荐系统：一项关于医学测试数据的案例研究 

**Authors**: Yahe Yang, Chengyue Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02727)  

**Abstract**: We present HiRMed (Hierarchical RAG-enhanced Medical Test Recommendation), a novel tree-structured recommendation system that leverages Retrieval-Augmented Generation (RAG) for intelligent medical test recommendations. Unlike traditional vector similarity-based approaches, our system performs medical reasoning at each tree node through a specialized RAG process. Starting from the root node with initial symptoms, the system conducts step-wise medical analysis to identify potential underlying conditions and their corresponding diagnostic requirements. At each level, instead of simple matching, our RAG-enhanced nodes analyze retrieved medical knowledge to understand symptom-disease relationships and determine the most appropriate diagnostic path. The system dynamically adjusts its recommendation strategy based on medical reasoning results, considering factors such as urgency levels and diagnostic uncertainty. Experimental results demonstrate that our approach achieves superior performance in terms of coverage rate, accuracy, and miss rate compared to conventional retrieval-based methods. This work represents a significant advance in medical test recommendation by introducing medical reasoning capabilities into the traditional tree-based retrieval structure. 

**Abstract (ZH)**: 我们提出了一种名为HiRMed（层次化RAG增强医疗测试推荐）的新型树状推荐系统，该系统利用检索增强生成（RAG）技术实现智能化的医疗测试推荐。与传统的基于向量相似度的方法不同，该系统在每个树节点处通过专门的RAG过程进行医疗推理。从根节点的初始症状开始，系统进行逐步的医疗分析，以识别潜在的下层状况及其相应的诊断需求。在每一层中，而不是简单的匹配，我们的RAG增强节点分析检索到的医学知识，理解症状-疾病关系，并确定最合适的诊断路径。系统根据医疗推理结果动态调整其推荐策略，考虑诸如紧急程度和诊断不确定性等因素。实验结果表明，与传统的基于检索的方法相比，我们的方法在覆盖率、准确性和漏检率方面表现出更优的性能。这项工作通过将医疗推理能力引入传统的基于树的检索结构中，显著推进了医疗测试推荐领域的发展。 

---
# Quantum Cognition-Inspired EEG-based Recommendation via Graph Neural Networks 

**Title (ZH)**: 基于图神经网络的启发于量子认知的脑电波推荐方法 

**Authors**: Jinkun Han, Wei Li, Yingshu Li, Zhipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2501.02671)  

**Abstract**: Current recommendation systems recommend goods by considering users' historical behaviors, social relations, ratings, and other multi-modals. Although outdated user information presents the trends of a user's interests, no recommendation system can know the users' real-time thoughts indeed. With the development of brain-computer interfaces, it is time to explore next-generation recommenders that show users' real-time thoughts without delay. Electroencephalography (EEG) is a promising method of collecting brain signals because of its convenience and mobility. Currently, there is only few research on EEG-based recommendations due to the complexity of learning human brain activity. To explore the utility of EEG-based recommendation, we propose a novel neural network model, QUARK, combining Quantum Cognition Theory and Graph Convolutional Networks for accurate item recommendations. Compared with the state-of-the-art recommendation models, the superiority of QUARK is confirmed via extensive experiments. 

**Abstract (ZH)**: 当前的推荐系统通过考虑用户的 histórico 行为、社会关系、评分及其他多模数据来推荐商品。尽管过时的用户信息可以反映出用户兴趣的趋势，但实际上，没有推荐系统能够切实了解用户的实时想法。随着脑-机接口技术的发展，是时候探索能够展现用户实时想法的下一代推荐系统了。脑电图（EEG）是一种收集脑电信号的有前途的方法，因为它既方便又便携。目前，由于学习人类脑活动的复杂性，基于 EEG 的推荐研究还很少见。为探讨基于 EEG 的推荐系统的实用价值，我们提出了一种结合量子认知理论和图卷积网络的新型神经网络模型 QUARK，以实现精确的商品推荐。通过广泛的实验，QUARK 的优越性得到了验证，相比于现有的先进推荐模型，QUARK 显示出更佳的表现。 

---
# Multi-Aggregator Time-Warping Heterogeneous Graph Neural Network for Personalized Micro-Video Recommendation 

**Title (ZH)**: 面向个性化微视频推荐的多聚合器时间扭曲异构图神经网络 

**Authors**: Jinkun Han, Wei Li, Xhipeng Cai, Yingshu Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.02666)  

**Abstract**: Micro-video recommendation is attracting global attention and becoming a popular daily service for people of all ages. Recently, Graph Neural Networks-based micro-video recommendation has displayed performance improvement for many kinds of recommendation tasks. However, the existing works fail to fully consider the characteristics of micro-videos, such as the high timeliness of news nature micro-video recommendation and sequential interactions of frequently changed interests. In this paper, a novel Multi-aggregator Time-warping Heterogeneous Graph Neural Network (MTHGNN) is proposed for personalized news nature micro-video recommendation based on sequential sessions, where characteristics of micro-videos are comprehensively studied, users' preference is mined via multi-aggregator, the temporal and dynamic changes of users' preference are captured, and timeliness is considered. Through the comparison with the state-of-the-arts, the experimental results validate the superiority of our MTHGNN model. 

**Abstract (ZH)**: 近年来，微视频推荐引起了全球的关注，并逐渐成为各个年龄段人们的日常生活服务。最近，基于图神经网络的微视频推荐在多种推荐任务中显示出了性能的提升。然而，现有的工作在处理微视频特点时未能充分考虑，例如新闻性质的微视频推荐的新鲜度及其兴趣频率变化下的序列交互。为了解决这些问题，本文提出了一种基于序列会话的新型多聚合时间扭曲异构图神经网络（MTHGNN），以实现个性化新闻性质微视频推荐。该模型全面研究了微视频的特点，通过多聚合器挖掘用户的偏好，捕捉用户的偏好随时间和动态变化，并考虑了新鲜度。通过与现有先进模型的对比实验，验证了MTHGNN模型的优越性。 

---
# Interactive Information Need Prediction with Intent and Context 

**Title (ZH)**: 基于意图和上下文的交互式信息需求预测 

**Authors**: Kevin Ros, Dhyey Pandya, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2501.02635)  

**Abstract**: The ability to predict a user's information need would have wide-ranging implications, from saving time and effort to mitigating vocabulary gaps. We study how to interactively predict a user's information need by letting them select a pre-search context (e.g., a paragraph, sentence, or singe word) and specify an optional partial search intent (e.g., "how", "why", "applications", etc.). We examine how various generative language models can explicitly make this prediction by generating a question as well as how retrieval models can implicitly make this prediction by retrieving an answer. We find that this prediction process is possible in many cases and that user-provided partial search intent can help mitigate large pre-search contexts. We conclude that this framework is promising and suitable for real-world applications. 

**Abstract (ZH)**: 预测用户的信息需求具有广泛的影响，从节省时间和精力到减轻词汇差距。我们研究了如何通过让用户选择一个预搜索上下文（例如，一段文字、一句话或单个词）并指定一种可选的部分搜索意图（例如，“如何”、“为什么”、“应用”等）来互动地预测用户的后续信息需求。我们探讨了各种生成型语言模型如何明确地通过生成一个问题来做出这一预测，以及检索模型如何通过检索答案隐式地做出这一预测。我们发现，在许多情况下这种预测是可行的，并且用户提供的部分搜索意图可以帮助缓解较大的预搜索上下文问题。我们得出结论，这种框架具有前景，并且适用于实际应用场景。 

---
# Citation Structural Diversity: A Novel and Concise Metric Combining Structure and Semantics for Literature Evaluation 

**Title (ZH)**: 引文结构多样性：一种结合结构与语义的新颖简明评价指标 

**Authors**: Mingyue Kong, Yinglong Zhang, Likun Sheng, Kaifeng Hong  

**Link**: [PDF](https://arxiv.org/pdf/2501.02429)  

**Abstract**: As academic research becomes increasingly diverse, traditional literature evaluation methods face significant limitations,particularly in capturing the complexity of academic dissemination and the multidimensional impacts of literature. To address these challenges, this paper introduces a novel literature evaluation model of citation structural diversity, with a focus on assessing its feasibility as an evaluation metric. By refining citation network and incorporating both ciation structural features and semantic information, the study examines the influence of the proposed model of citation structural diversity on citation volume and long-term academic impact. The findings reveal that literature with higher citation structural diversity demonstrates notable advantages in both citation frequency and sustained academic influence. Through data grouping and a decade-long citation trend analysis, the potential application of this model in literature evaluation is further validated. This research offers a fresh perspective on optimizing literature evaluation methods and emphasizes the distinct advantages of citation structural diversity in measuring interdisciplinarity. 

**Abstract (ZH)**: 随着学术研究日益多元化，传统文献评价方法面临着显著的限制，特别是无法充分捕捉学术传播的复杂性以及文献的多维度影响。为应对这些挑战，本文介绍了一种新的文献评价模型——引用结构多样性，并重点关注其作为评价指标的可行性。通过精炼引用网络并融合引用结构特征和语义信息，研究探讨了所提引用结构多样性模型对引用量和长期学术影响的影响力。研究发现，具有更高引用结构多样性的文献在引用频率和持久的学术影响方面表现出明显的优势。通过对数据进行分组和长达十年的引用趋势分析，进一步验证了该模型在文献评价中的潜在应用。该研究为优化文献评价方法提供了新视角，并强调了引用结构多样性在衡量跨学科性方面的独特优势。 

---
# GenTREC: The First Test Collection Generated by Large Language Models for Evaluating Information Retrieval Systems 

**Title (ZH)**: GenTREC: 首个多语言模型生成的测试集合，用于评估信息检索系统 

**Authors**: Mehmet Deniz Türkmen, Mucahid Kutlu, Bahadir Altun, Gokalp Cosgun  

**Link**: [PDF](https://arxiv.org/pdf/2501.02408)  

**Abstract**: Building test collections for Information Retrieval evaluation has traditionally been a resource-intensive and time-consuming task, primarily due to the dependence on manual relevance judgments. While various cost-effective strategies have been explored, the development of such collections remains a significant challenge. In this paper, we present GenTREC , the first test collection constructed entirely from documents generated by a Large Language Model (LLM), eliminating the need for manual relevance judgments. Our approach is based on the assumption that documents generated by an LLM are inherently relevant to the prompts used for their generation. Based on this heuristic, we utilized existing TREC search topics to generate documents. We consider a document relevant only to the prompt that generated it, while other document-topic pairs are treated as non-relevant. To introduce realistic retrieval challenges, we also generated non-relevant documents, ensuring that IR systems are tested against a diverse and robust set of materials. The resulting GenTREC collection comprises 96,196 documents, 300 topics, and 18,964 relevance "judgments". We conducted extensive experiments to evaluate GenTREC in terms of document quality, relevance judgment accuracy, and evaluation reliability. Notably, our findings indicate that the ranking of IR systems using GenTREC is compatible with the evaluations conducted using traditional TREC test collections, particularly for P@100, MAP, and RPrec metrics. Overall, our results show that our proposed approach offers a promising, low-cost alternative for IR evaluation, significantly reducing the burden of building and maintaining future IR evaluation resources. 

**Abstract (ZH)**: 构建信息检索评估所需的数据集历来是一个资源密集型和耗时的任务，主要原因是依赖于人工相关性判断。尽管已经探索了各种成本效益较高的策略，但开发此类数据集仍然是一项重大挑战。本文中，我们介绍了GenTREC，这是首个完全基于大型语言模型（LLM）生成文档构建的测试数据集，从而消除了人工相关性判断的需要。我们的方法基于假设，由LLM生成的文档与生成它们的提示之间具有内在的相关性。基于这一启发式方法，我们使用现有TREC检索主题生成文档。我们只将文档视为与生成它的提示相关的，而其他文档-主题配对则被视为不相关的。为了引入现实的检索挑战，我们还生成了不相关的文档，确保IR系统是在多种多样的且稳健的材料上进行测试的。生成的GenTREC数据集包含96,196份文档、300个主题和18,964份相关性“判断”。我们进行了广泛的实验，从文档质量、相关性判断准确性以及评估可靠性等方面评估GenTREC。值得注意的是，我们的发现表明，使用GenTREC进行IR系统的排名与使用传统TREC测试数据集进行的评估结果是兼容的，特别是在P@100、MAP和RPrec指标上。总体而言，我们的结果表明，我们提出的方法为IR评估提供了一种有前景的低成本替代方案，显著降低了未来构建和维护IR评估资源的负担。 

---
# Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation 

**Title (ZH)**: 基于知识图谱检索增强的生成推荐模型 

**Authors**: Shijie Wang, Wenqi Fan, Yue Feng, Xinyu Ma, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2501.02226)  

**Abstract**: Recommender systems have become increasingly vital in our daily lives, helping to alleviate the problem of information overload across various user-oriented online services. The emergence of Large Language Models (LLMs) has yielded remarkable achievements, demonstrating their potential for the development of next-generation recommender systems. Despite these advancements, LLM-based recommender systems face inherent limitations stemming from their LLM backbones, particularly issues of hallucinations and the lack of up-to-date and domain-specific knowledge. Recently, Retrieval-Augmented Generation (RAG) has garnered significant attention for addressing these limitations by leveraging external knowledge sources to enhance the understanding and generation of LLMs. However, vanilla RAG methods often introduce noise and neglect structural relationships in knowledge, limiting their effectiveness in LLM-based recommendations. To address these limitations, we propose to retrieve high-quality and up-to-date structure information from the knowledge graph (KG) to augment recommendations. Specifically, our approach develops a retrieval-augmented framework, termed K-RagRec, that facilitates the recommendation generation process by incorporating structure information from the external KG. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed method. 

**Abstract (ZH)**: 推荐系统在我们的日常生活中变得愈发重要，它们帮助各类用户面向在线服务缓解信息过载问题。大型语言模型（LLMs）的出现取得了显著成就，显示出其在新一代推荐系统开发中的潜力。然而，尽管取得了这些进展，LLM 基于的推荐系统仍然面临来自其LLM基础模型固有的限制，特别是幻觉问题以及缺乏最新的和专门领域的知识。最近，检索增强生成（RAG）方法引起了广泛关注，通过利用外部知识源来增强LLM的理解和生成能力，以解决这些限制。然而，传统的RAG方法往往引入噪声并忽略知识中的结构性关系，限制了它们在基于LLM的推荐中的有效性。为了解决这些限制，我们提出了一种方法，从知识图谱（KG）中检索高质量和最新的结构信息来增强推荐。具体而言，我们的方法开发了一种检索增强框架，称为K-RagRec，通过结合外部KG中的结构信息来促进推荐生成过程。进行了广泛的实验以展示我们提出方法的有效性。 

---
# The Application of Large Language Models in Recommendation Systems 

**Title (ZH)**: 大型语言模型在推荐系统中的应用 

**Authors**: Peiyang Yu, Zeqiu Xu, Jiani Wang, Xiaochuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.02178)  

**Abstract**: The integration of Large Language Models into recommendation frameworks presents key advantages for personalization and adaptability of experiences to the users. Classic methods of recommendations, such as collaborative filtering and content-based filtering, are seriously limited in the solution of cold-start problems, sparsity of data, and lack of diversity in information considered. LLMs, of which GPT-4 is a good example, have emerged as powerful tools that enable recommendation frameworks to tap into unstructured data sources such as user reviews, social interactions, and text-based content. By analyzing these data sources, LLMs improve the accuracy and relevance of recommendations, thereby overcoming some of the limitations of traditional approaches. This work discusses applications of LLMs in recommendation systems, especially in electronic commerce, social media platforms, streaming services, and educational technologies. This showcases how LLMs enrich recommendation diversity, user engagement, and the system's adaptability; yet it also looks into the challenges connected to their technical implementation. This can also be presented as a study that shows the potential of LLMs for changing user experiences and making innovation possible in industries. 

**Abstract (ZH)**: 将大型语言模型（LLM）集成到推荐框架中，为个性化和用户体验的适应性带来了关键优势。传统的推荐方法，如协同过滤和基于内容的过滤，严重受限于冷启动问题、数据稀疏性和信息多样性不足。像GPT-4这样的LLM已经成为强大的工具，使推荐框架能够利用结构化数据源，如用户评论、社交互动和文本内容。通过对这些数据源的分析，LLM提高了推荐的准确性和相关性，从而克服了传统方法的一些局限性。本文讨论了LLM在推荐系统中的应用，特别是在电子商务、社交媒体平台、流媒体服务和教育技术领域。这展示了LLM如何丰富推荐的多样性、增强用户的参与度和系统的适应性；同时也探讨了其技术实现所面临的挑战。这项工作还可以被视为一项研究，展示了LLM在改变用户体验和推动行业创新方面的潜力。 

---
# The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit 

**Title (ZH)**: 效率与准确性的权衡：使用多头早期退出优化增强型LLM推荐系统 

**Authors**: Huixue Zhou, Hengrui Gu, Xi Liu, Kaixiong Zhou, Mingfu Liang, Yongkang Xiao, Srinivas Govindan, Piyush Chawla, Jiyan Yang, Xiangfei Meng, Huayu Li, Buyun Zhang, Liang Luo, Wen-Yen Chen, Yiping Han, Bo Long, Rui Zhang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.02173)  

**Abstract**: The deployment of Large Language Models (LLMs) in recommender systems for predicting Click-Through Rates (CTR) necessitates a delicate balance between computational efficiency and predictive accuracy. This paper presents an optimization framework that combines Retrieval-Augmented Generation (RAG) with an innovative multi-head early exit architecture to concurrently enhance both aspects. By integrating Graph Convolutional Networks (GCNs) as efficient retrieval mechanisms, we are able to significantly reduce data retrieval times while maintaining high model performance. The early exit strategy employed allows for dynamic termination of model inference, utilizing real-time predictive confidence assessments across multiple heads. This not only quickens the responsiveness of LLMs but also upholds or improves their accuracy, making it ideal for real-time application scenarios. Our experiments demonstrate how this architecture effectively decreases computation time without sacrificing the accuracy needed for reliable recommendation delivery, establishing a new standard for efficient, real-time LLM deployment in commercial systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推荐系统中的部署对于预测点击率（CTR）而言，需要在计算效率和预测准确性之间保持微妙的平衡。本文提出了一种优化框架，结合了检索增强生成（RAG）与一种创新的多头早期退出架构，以同时提升这两个方面。通过将图卷积网络（GCNs）作为高效的检索机制进行整合，我们能够在降低数据检索时间的同时，保持高模型性能。所采用的早期退出策略允许模型推理的动态终止，通过多个头的实时预测置信度评估来进行。这不仅能加快大语言模型的响应速度，而且还能够维持或提升其准确性，使其适用于实时应用场景。我们的实验表明，这种架构能够在不牺牲可靠推荐所需准确性的前提下有效减少计算时间，从而为商业系统中高效、实时的大语言模型部署设立了新标准。 

---
# Graph-based Retrieval Augmented Generation for Dynamic Few-shot Text Classification 

**Title (ZH)**: 基于图的检索增强生成方法在动态少样本文本分类中的应用 

**Authors**: Yubo Wang, Haoyang Li, Fei Teng, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.02844)  

**Abstract**: Text classification is a fundamental task in natural language processing, pivotal to various applications such as query optimization, data integration, and schema matching. While neural network-based models, such as CNN and BERT, have demonstrated remarkable performance in text classification, their effectiveness heavily relies on abundant labeled training data. This dependency makes these models less effective in dynamic few-shot text classification, where labeled data is scarce, and target labels frequently evolve based on application needs. Recently, large language models (LLMs) have shown promise due to their extensive pretraining and contextual understanding. Current approaches provide LLMs with text inputs, candidate labels, and additional side information (e.g., descriptions) to predict text labels. However, their effectiveness is hindered by the increased input size and the noise introduced through side information processing. To address these limitations, we propose a graph-based online retrieval-augmented generation framework, namely GORAG, for dynamic few-shot text classification. GORAG constructs and maintains an adaptive information graph by extracting side information across all target texts, rather than treating each input independently. It employs a weighted edge mechanism to prioritize the importance and reliability of extracted information and dynamically retrieves relevant context using a minimum-cost spanning tree tailored for each text input. Empirical evaluations demonstrate that GORAG outperforms existing approaches by providing more comprehensive and accurate contextual information. 

**Abstract (ZH)**: 文本分类是自然语言处理中的一个基本任务，对于各种应用如查询优化、数据集成和模式匹配至关重要。虽然基于神经网络的模型，如CNN和BERT，在文本分类方面表现出色，但它们的有效性高度依赖于充足的标注训练数据。这种依赖性使其在动态少量标注的文本分类任务中效果较差，在这种任务中，标注数据稀缺，目标标签会频繁根据应用需求而变化。最近，大型语言模型（LLMs）由于其广泛的预训练和语境理解能力，显示出了潜力。现有方法通过提供文本输入、候选标签以及额外的辅助信息（如描述）来预测文本标签。然而，这些方法的有效性受到了输入尺寸增加以及辅助信息处理过程中引入噪声的阻碍。为了解决这些问题，我们提出了一种基于图的在线检索增强生成框架，即GORAG，用于动态少量标注的文本分类。GORAG通过跨所有目标文本提取辅助信息，而不是独立处理每个输入，构建并维护一个自适应信息图。它采用加权边机制以优先处理提取信息的重要性与可靠性，并动态检索针对每个文本输入定制的最小子成本生成树来获取相关的上下文。实证研究表明，GORAG相较于现有方法能够提供更加全面和准确的上下文信息。 

---
# Integrating Language-Image Prior into EEG Decoding for Cross-Task Zero-Calibration RSVP-BCI 

**Title (ZH)**: 将语言-图像先验融入事件相关电位脑机接口的跨任务零校准解码中 

**Authors**: Xujin Li, Wei Wei, Shuang Qiu, Xinyi Zhang, Fu Li, Huiguang He  

**Link**: [PDF](https://arxiv.org/pdf/2501.02841)  

**Abstract**: Rapid Serial Visual Presentation (RSVP)-based Brain-Computer Interface (BCI) is an effective technology used for information detection by detecting Event-Related Potentials (ERPs). The current RSVP decoding methods can perform well in decoding EEG signals within a single RSVP task, but their decoding performance significantly decreases when directly applied to different RSVP tasks without calibration data from the new tasks. This limits the rapid and efficient deployment of RSVP-BCI systems for detecting different categories of targets in various scenarios. To overcome this limitation, this study aims to enhance the cross-task zero-calibration RSVP decoding performance. First, we design three distinct RSVP tasks for target image retrieval and build an open-source dataset containing EEG signals and corresponding stimulus images. Then we propose an EEG with Language-Image Prior fusion Transformer (ELIPformer) for cross-task zero-calibration RSVP decoding. Specifically, we propose a prompt encoder based on the language-image pre-trained model to extract language-image features from task-specific prompts and stimulus images as prior knowledge for enhancing EEG decoding. A cross bidirectional attention mechanism is also adopted to facilitate the effective feature fusion and alignment between the EEG and language-image features. Extensive experiments demonstrate that the proposed model achieves superior performance in cross-task zero-calibration RSVP decoding, which promotes the RSVP-BCI system from research to practical application. 

**Abstract (ZH)**: 基于Rapid Serial Visual Presentation (RSVP)的脑-计算机接口（BCI）是一种通过检测事件相关电位（ERPs）来检测信息的有效技术。当前的RSVP解码方法在单一RSVP任务中能够很好地解码EEG信号，但在直接应用于没有来自新任务校准数据的不同RSVP任务时，其解码性能显著下降。这限制了RSVP-BCI系统在各种场景中快速高效地检测不同类别目标的能力。为克服这一局限性，本研究旨在提高跨任务零校准RSVP解码性能。首先，我们设计了三个不同的RSVP任务用于目标图像检索，并构建了一个包含EEG信号及其对应刺激图像的开源数据集。然后，我们提出了一种融合语言-图像先验的Transformer（ELIPformer）用于跨任务零校准RSVP解码。具体来说，我们提出了基于语言-图像预训练模型的提示编码器，用于从任务特定提示和刺激图像中提取语言-图像特征作为增强EEG解码的先验知识。我们还采用了一种交叉双向注意机制，以促进EEG和语言-图像特征之间的有效特征融合和对齐。广泛的实验表明，所提出的方法在跨任务零校准RSVP解码中表现 superiority，这促进了RSVP-BCI系统从研究到实际应用的转化。 

---
# Forward Once for All: Structural Parameterized Adaptation for Efficient Cloud-coordinated On-device Recommendation 

**Title (ZH)**: 一次性前推：面向设备端推荐的结构参数化适应性优化方法 

**Authors**: Kairui Fu, Zheqi Lv, Shengyu Zhang, Fan Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02837)  

**Abstract**: In cloud-centric recommender system, regular data exchanges between user devices and cloud could potentially elevate bandwidth demands and privacy risks. On-device recommendation emerges as a viable solution by performing reranking locally to alleviate these concerns. Existing methods primarily focus on developing local adaptive parameters, while potentially neglecting the critical role of tailor-made model architecture. Insights from broader research domains suggest that varying data distributions might favor distinct architectures for better fitting. In addition, imposing a uniform model structure across heterogeneous devices may result in risking inefficacy on less capable devices or sub-optimal performance on those with sufficient capabilities. In response to these gaps, our paper introduces Forward-OFA, a novel approach for the dynamic construction of device-specific networks (both structure and parameters). Forward-OFA employs a structure controller to selectively determine whether each block needs to be assembled for a given device. However, during the training of the structure controller, these assembled heterogeneous structures are jointly optimized, where the co-adaption among blocks might encounter gradient conflicts. To mitigate this, Forward-OFA is designed to establish a structure-guided mapping of real-time behaviors to the parameters of assembled networks. Structure-related parameters and parallel components within the mapper prevent each part from receiving heterogeneous gradients from others, thus bypassing the gradient conflicts for coupled optimization. Besides, direct mapping enables Forward-OFA to achieve adaptation through only one forward pass, allowing for swift adaptation to changing interests and eliminating the requirement for on-device backpropagation. Experiments on real-world datasets demonstrate the effectiveness and efficiency of Forward-OFA. 

**Abstract (ZH)**: 在以云为中心的推荐系统中，用户设备与云之间的常规数据交换可能会增加带宽需求和隐私风险。本地推荐作为一种可行的解决方案，通过在本地重新排序来减轻这些担忧。现有方法主要侧重于开发本地适应性参数，而可能忽视了特定结构模型的重要作用。更广泛的研究领域揭示出，不同的数据分布可能有利于适合不同架构的模型。此外，在异构设备上采用统一的模型结构可能会在能力较弱的设备上导致无效性，或者在具有足够能力的设备上导致次优性能。为应对这些不足，我们提出了Forward-OFA，这是一种新型的用于动态构建设备特定网络（包括结构和参数）的方法。Forward-OFA 使用结构控制器来选择性地确定每个模块是否需要组装给定设备。然而，在训练结构控制器期间，这些组装的异构结构会在联合优化中优化，模块之间的共适应可能会出现梯度冲突。为了缓解这一问题，Forward-OFA 设计了一个结构引导的实时行为到组装网络参数的映射。与映射相关的结构参数和并行组件防止每个部分从其他部分接收异构梯度，从而避免了耦合优化中的梯度冲突。此外，直接映射使Forward-OFA 只需一次前向传播即可实现适应，从而迅速适应不断变化的兴趣，并消除了设备上回传的需求。在现实世界数据集上的实验表明，Forward-OFA 具有有效性和高效率。 

---
# Can Impressions of Music be Extracted from Thumbnail Images? 

**Title (ZH)**: 音乐的印象能否从缩略图中提取？ 

**Authors**: Takashi Harada, Takehiro Motomitsu, Katsuhiko Hayashi, Yusuke Sakai, Hidetaka Kamigaito  

**Link**: [PDF](https://arxiv.org/pdf/2501.02511)  

**Abstract**: In recent years, there has been a notable increase in research on machine learning models for music retrieval and generation systems that are capable of taking natural language sentences as inputs. However, there is a scarcity of large-scale publicly available datasets, consisting of music data and their corresponding natural language descriptions known as music captions. In particular, non-musical information such as suitable situations for listening to a track and the emotions elicited upon listening is crucial for describing music. This type of information is underrepresented in existing music caption datasets due to the challenges associated with extracting it directly from music data. To address this issue, we propose a method for generating music caption data that incorporates non-musical aspects inferred from music thumbnail images, and validated the effectiveness of our approach through human evaluations. Additionally, we created a dataset with approximately 360,000 captions containing non-musical aspects. Leveraging this dataset, we trained a music retrieval model and demonstrated its effectiveness in music retrieval tasks through evaluation. 

**Abstract (ZH)**: 近年来，针对能够以自然语言句子为输入的音乐检索和生成系统的研究中，机器学习模型的研究取得了显著进展。然而，现有的大规模公开可用数据集中，包含音乐数据及其对应自然语言描述（即音乐描述）的数据集仍然十分稀缺。特别是适合听某一曲目的场合信息以及听该曲目时所引发的情感等非音乐信息对于描述音乐至关重要。现有的音乐描述数据集中这种信息的代表性不足，主要是由于直接从音乐数据中提取这些信息存在挑战。为解决这一问题，我们提出了一种结合从音乐缩略图中推断的非音乐方面信息生成音乐描述数据的方法，并通过人工评估验证了该方法的有效性。此外，我们构建了一个包含约36万条涵盖非音乐方面的描述数据集。利用这个数据集，我们训练了一个音乐检索模型，并通过评估展示了其在音乐检索任务中的有效性。 

---
# DiffGraph: Heterogeneous Graph Diffusion Model 

**Title (ZH)**: DiffGraph：异构图扩散模型 

**Authors**: Zongwei Li, Lianghao Xia, Hua Hua, Shijie Zhang, Shuangyang Wang, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02313)  

**Abstract**: Recent advances in Graph Neural Networks (GNNs) have revolutionized graph-structured data modeling, yet traditional GNNs struggle with complex heterogeneous structures prevalent in real-world scenarios. Despite progress in handling heterogeneous interactions, two fundamental challenges persist: noisy data significantly compromising embedding quality and learning performance, and existing methods' inability to capture intricate semantic transitions among heterogeneous relations, which impacts downstream predictions. To address these fundamental issues, we present the Heterogeneous Graph Diffusion Model (DiffGraph), a pioneering framework that introduces an innovative cross-view denoising strategy. This advanced approach transforms auxiliary heterogeneous data into target semantic spaces, enabling precise distillation of task-relevant information. At its core, DiffGraph features a sophisticated latent heterogeneous graph diffusion mechanism, implementing a novel forward and backward diffusion process for superior noise management. This methodology achieves simultaneous heterogeneous graph denoising and cross-type transition, while significantly simplifying graph generation through its latent-space diffusion capabilities. Through rigorous experimental validation on both public and industrial datasets, we demonstrate that DiffGraph consistently surpasses existing methods in link prediction and node classification tasks, establishing new benchmarks for robustness and efficiency in heterogeneous graph processing. The model implementation is publicly available at: this https URL. 

**Abstract (ZH)**: 近年来，图神经网络（GNNs）在图结构数据建模方面取得了革命性的进展，然而传统的GNNs在处理现实场景中普遍存在的复杂异质结构方面存在困难。尽管在处理异质交互方面取得了进步，但仍然存在着两个根本性的挑战：噪声数据严重损害了嵌入质量和学习性能，以及现有方法无法捕捉异质关系之间的细腻语义转换，这影响了下游预测。为了解决这些根本问题，我们提出了一种名为异质图扩散模型（DiffGraph）的开创性框架，引入了一种创新的跨视图去噪策略。这种高级方法将辅助的异质数据转换为目标语义空间，从而能够精确地提取任务相关的信息。DiffGraph的核心特点是复杂的潜在异质图扩散机制，通过新颖的正向和反向扩散过程实现了卓越的去噪管理。该方法在潜在空间扩散能力的简化下同时实现了异质图去噪和跨类型转换，从而显著简化了图生成。通过在公共和工业数据集上的严格实验验证，我们证明了DiffGraph在链接预测和节点分类任务中均优于现有方法，为异质图处理的鲁棒性和效率设定了新的基准。该模型实施已在以下网址公开提供：this https URL。 

---
