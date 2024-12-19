# A Cognitive Ideation Support Framework using IBM Watson Services 

**Title (ZH)**: 使用 IBM Watson 服务的认知创新支持框架 

**Authors**: Samaa Elnagar, Kweku-Muata Osei-Bryson  

**Link**: [PDF](https://arxiv.org/pdf/2412.14025)  

**Abstract**: Ideas generation is a core activity for innovation in organizations. The creativity of the generated ideas depends not only on the knowledge retrieved from the organizations' knowledge bases, but also on the external knowledge retrieved from other resources. Unfortunately, organizations often cannot efficiently utilize the knowledge in the knowledge bases due to the limited abilities of the search and retrieval mechanisms especially when dealing with unstructured data. In this paper, we present a new cognitive support framework for ideation that uses the IBM Watson DeepQA services. IBM Watson is a Question Answering system which mimics human cognitive abilities to retrieve and rank information. The proposed framework is based on the Search for Ideas in the Associative Memory (SIAM) model to help organizations develop creative ideas through discovering new relationships between retrieved data. To evaluate the effectiveness of the proposed system, the generated ideas generated are selected and assessed using a set of established creativity criteria. 

**Abstract (ZH)**: 创新是组织核心活动之一，生成的创意的质量不仅依赖于从组织知识库中检索的知识，还依赖于从其他资源中检索的外部知识。遗憾的是，由于搜索和检索机制能力有限，特别是在处理非结构化数据时，组织往往无法有效利用知识库中的知识。本文提出了一种新的认知支持框架，用于创意生成，并利用IBM Watson DeepQA服务。IBM Watson是一种问题回答系统，模仿人类认知能力来检索和排名信息。所提出的框架基于联想记忆中创意搜索（SIAM）模型，通过发现检索数据之间的新关系帮助组织发展创意。为了评估所提出系统的有效性，生成的创意被选择并使用一套已建立的创造力标准进行评估。 

---
# CRM: Retrieval Model with Controllable Condition 

**Title (ZH)**: CRM：可控条件的检索模型 

**Authors**: Chi Liu, Jiangxia Cao, Rui Huang, Kuo Cai, Weifeng Ding, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.13844)  

**Abstract**: Recommendation systems (RecSys) are designed to connect users with relevant items from a vast pool of candidates while aligning with the business goals of the platform. A typical industrial RecSys is composed of two main stages, retrieval and ranking: (1) the retrieval stage aims at searching hundreds of item candidates satisfied user interests; (2) based on the retrieved items, the ranking stage aims at selecting the best dozen items by multiple targets estimation for each item candidate, including classification and regression targets. Compared with ranking model, the retrieval model absence of item candidate information during inference, therefore retrieval models are often trained by classification target only (e.g., click-through rate), but failed to incorporate regression target (e.g., the expected watch-time), which limit the effectiveness of retrieval. In this paper, we propose the Controllable Retrieval Model (CRM), which integrates regression information as conditional features into the two-tower retrieval paradigm. This modification enables the retrieval stage could fulfill the target gap with ranking model, enhancing the retrieval model ability to search item candidates satisfied the user interests and condition effectively. We validate the effectiveness of CRM through real-world A/B testing and demonstrate its successful deployment in Kuaishou short-video recommendation system, which serves over 400 million users. 

**Abstract (ZH)**: 推荐系统（RecSys）旨在将用户与大量候选项中的相关项目连接起来，并与平台的业务目标相一致。典型的工业推荐系统通常由两个主要阶段组成：检索和排序：（1）检索阶段旨在搜索满足用户兴趣的大约几百个候选项；（2）基于检索出的项，排序阶段旨在通过多项目标估计选择出最佳十几个候选项，包括分类和回归目标。与排序模型相比，检索模型在推理过程中缺乏候选项的信息，因此检索模型通常仅通过分类目标（例如点击率）进行训练，而未能纳入回归目标（例如预期观看时间），这限制了检索模型的效果。在本文中，我们提出了一种可控检索模型（CRM），该模型将回归信息整合为条件特征，融入双塔检索架构中。这一修改使检索阶段能够弥补与排序模型之间的目标差距，增强检索模型的能力，使其能够有效地搜索满足用户兴趣和条件的候选项。我们通过实际的A/B测试验证了CRM的有效性，并展示了其在快手短视频推荐系统中的成功部署，该系统服务于超过4亿用户。 

---
# Maybe you are looking for CroQS: Cross-modal Query Suggestion for Text-to-Image Retrieval 

**Title (ZH)**: 也许您正在寻找CroQS：跨模态查询建议在文本到图像检索中的应用 

**Authors**: Giacomo Pacini, Fabio Carrara, Nicola Messina, Nicola Tonellotto, Giuseppe Amato, Fabrizio Falchi  

**Link**: [PDF](https://arxiv.org/pdf/2412.13834)  

**Abstract**: Query suggestion, a technique widely adopted in information retrieval, enhances system interactivity and the browsing experience of document collections. In cross-modal retrieval, many works have focused on retrieving relevant items from natural language queries, while few have explored query suggestion solutions. In this work, we address query suggestion in cross-modal retrieval, introducing a novel task that focuses on suggesting minimal textual modifications needed to explore visually consistent subsets of the collection, following the premise of ''Maybe you are looking for''. To facilitate the evaluation and development of methods, we present a tailored benchmark named CroQS. This dataset comprises initial queries, grouped result sets, and human-defined suggested queries for each group. We establish dedicated metrics to rigorously evaluate the performance of various methods on this task, measuring representativeness, cluster specificity, and similarity of the suggested queries to the original ones. Baseline methods from related fields, such as image captioning and content summarization, are adapted for this task to provide reference performance scores. Although relatively far from human performance, our experiments reveal that both LLM-based and captioning-based methods achieve competitive results on CroQS, improving the recall on cluster specificity by more than 115% and representativeness mAP by more than 52% with respect to the initial query. The dataset, the implementation of the baseline methods and the notebooks containing our experiments are available here: this https URL 

**Abstract (ZH)**: 查询建议是一种广泛应用于信息检索的技术，能够增强系统的互动性和文档集合的浏览体验。在跨模态检索中，许多研究集中在从自然语言查询中检索相关项上，而很少有人关注查询建议解决方案。在本项研究中，我们针对跨模态检索中的查询建议问题，提出了一个新任务，该任务关注于建议针对集合中视觉一致子集所需的最小文本修改，并遵循“也许您寻找的是...”这一前提。为了促进评估和方法的发展，我们提供了一个定制的基准数据集，称为CroQS。该数据集包含初始查询、分组结果集以及为每个分组定义的人工建议查询。我们为该任务建立了专门的评估指标，以严格评估各种方法的表现，这些指标包括建议查询的代表性、聚类特异性和建议查询与原始查询的相似度。我们还对相关领域的基线方法，如图像描述和内容总结，进行了适应性改进，以提供参考性能分数。尽管相较于人类性能仍有一定差距，但实验结果显示，基于大语言模型的方法和基于图像描述的方法在CroQS任务上取得了可比较的结果。相较于初始查询，这两种方法分别在聚类特异性召回率和代表性平均精度方面提高了115%以上和52%以上。我们的数据集、基线方法的实现以及包含实验的笔记本代码均在此处提供：this https URL 

---
# Heterogeneous Graph Collaborative Filtering 

**Title (ZH)**: 异质图协作过滤 

**Authors**: Lianghao Xia, Meiyan Xie, Yong Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.13825)  

**Abstract**: For modern recommender systems, the use of low-dimensional latent representations to embed users and items based on their observed interactions has become commonplace. However, many existing recommendation models are primarily designed for coarse-grained and homogeneous interactions, which limits their effectiveness in two critical dimensions. Firstly, these models fail to leverage the relational dependencies that exist across different types of user behaviors, such as page views, collects, comments, and purchases. Secondly, they struggle to capture the fine-grained latent factors that drive user interaction patterns. To address these limitations, we present a heterogeneous graph collaborative filtering model MixRec that excels at disentangling users' multi-behavior interaction patterns and uncovering the latent intent factors behind each behavior. Our model achieves this by incorporating intent disentanglement and multi-behavior modeling, facilitated by a parameterized heterogeneous hypergraph architecture. Furthermore, we introduce a novel contrastive learning paradigm that adaptively explores the advantages of self-supervised data augmentation, thereby enhancing the model's resilience against data sparsity and expressiveness with relation heterogeneity. To validate the efficacy of MixRec, we conducted extensive experiments on three public datasets. The results clearly demonstrate its superior performance, significantly outperforming various state-of-the-art baselines. Our model is open-sourced and available at: this https URL. 

**Abstract (ZH)**: 对于现代推荐系统而言，通过低维度的潜在表示将用户和项目根据其已观察到的交互进行嵌入已经成为一种常见做法。然而，现有的许多推荐模型主要针对粗粒度和同质的交互进行设计，这限制了它们在两个关键维度上的有效性。首先，这些模型未能利用不同类型用户行为之间的关系依赖，如页面浏览、收藏、评论和购买。其次，它们难以捕捉驱动用户交互模式的细微潜在因素。为了解决这些限制，我们提出了一种异构图协同过滤模型 MixRec，该模型能够出色地解析用户多行为交互模式，并揭示每个行为背后的潜在意图因素。我们的模型通过一个参数化异构超图架构结合意图分离和多行为建模实现这一点。此外，我们提出了一种新颖的对比学习范式，该范式能够适配地探索自我监督数据增强的优势，从而增强模型对数据稀疏性和关系异质性的鲁棒性和表达能力。为了验证 MixRec 的有效性，我们在三个公开数据集上进行了广泛的实验。结果清楚地表明其优越的表现，显著优于各种最先进的基线模型。我们的模型是开源的，可以在以下链接获取：[请在此处提供链接]。 

---
# Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization 

**Title (ZH)**: 语义融合：通过两阶段对齐和行为语义标记化协调推荐系统 

**Authors**: Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13771)  

**Abstract**: Large language models (LLMs), endowed with exceptional reasoning capabilities, are adept at discerning profound user interests from historical behaviors, thereby presenting a promising avenue for the advancement of recommendation systems. However, a notable discrepancy persists between the sparse collaborative semantics typically found in recommendation systems and the dense token representations within LLMs. In our study, we propose a novel framework that harmoniously merges traditional recommendation models with the prowess of LLMs. We initiate this integration by transforming ItemIDs into sequences that align semantically with the LLMs space, through the proposed Alignment Tokenization module. Additionally, we design a series of specialized supervised learning tasks aimed at aligning collaborative signals with the subtleties of natural language semantics. To ensure practical applicability, we optimize online inference by pre-caching the top-K results for each user, reducing latency and improving effciency. Extensive experimental evidence indicates that our model markedly improves recall metrics and displays remarkable scalability of recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）具备卓越的推理能力，能够从用户的历史行为中洞悉深刻的兴趣，从而为推荐系统的进步提供了广阔前景。然而，推荐系统中通常存在的稀疏协作语义与LLMs所具有的密集词表示之间存在明显的不匹配。在本研究中，我们提出了一种新的框架，巧妙地将传统推荐模型与LLMs的优势相结合。我们通过引入提出的Alignment Tokenization模块，将ItemIDs转换为与LLMs空间相匹配的语义序列，以启动这种整合。此外，我们设计了一系列专门的监督学习任务，旨在将协作信号与自然语言语义的细微差别对齐。为了确保其实用性，我们通过预缓存每个用户的前K个结果来优化在线推理过程，从而降低延迟并提高效率。大量实验结果表明，我们的模型显著提高了召回度指标，并展示了推荐系统出色的扩展性。 

---
# Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models 

**Title (ZH)**: 使用大规模语言模型跨越用户侧知识差距的知识感知推荐方法 

**Authors**: Zheng Hu, Zhe Li, Ziyun Jiao, Satoshi Nakagawa, Jiawen Deng, Shimin Cai, Tao Zhou, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2412.13544)  

**Abstract**: In recent years, knowledge graphs have been integrated into recommender systems as item-side auxiliary information, enhancing recommendation accuracy. However, constructing and integrating structural user-side knowledge remains a significant challenge due to the improper granularity and inherent scarcity of user-side features. Recent advancements in Large Language Models (LLMs) offer the potential to bridge this gap by leveraging their human behavior understanding and extensive real-world knowledge. Nevertheless, integrating LLM-generated information into recommender systems presents challenges, including the risk of noisy information and the need for additional knowledge transfer. In this paper, we propose an LLM-based user-side knowledge inference method alongside a carefully designed recommendation framework to address these challenges. Our approach employs LLMs to infer user interests based on historical behaviors, integrating this user-side information with item-side and collaborative data to construct a hybrid structure: the Collaborative Interest Knowledge Graph (CIKG). Furthermore, we propose a CIKG-based recommendation framework that includes a user interest reconstruction module and a cross-domain contrastive learning module to mitigate potential noise and facilitate knowledge transfer. We conduct extensive experiments on three real-world datasets to validate the effectiveness of our method. Our approach achieves state-of-the-art performance compared to competitive baselines, particularly for users with sparse interactions. 

**Abstract (ZH)**: 近年来，知识图谱已成为推荐系统中的项目侧面辅助信息，提升了推荐准确度。然而，由于用户侧特征的不合适粒度和固有的稀缺性，构建和整合用户侧结构信息仍然是一项重大挑战。近年来，大型语言模型（LLMs）的进步为解决这一问题提供了可能，通过利用其对人类行为的理解和广泛的实际世界知识。然而，将LLM生成的信息整合到推荐系统中也面临着挑战，包括噪声信息的风险以及额外知识转移的需要。在本文中，我们提出了一种基于LLM的用户侧知识推理方法以及一个精心设计的推荐框架来解决这些挑战。我们的方法利用LLM根据历史行为推断用户兴趣，并将这种用户侧信息与项目侧和协同信息集成以构建一种混合结构：协作兴趣知识图谱（CIKG）。此外，我们提出了一种基于CIKG的推荐框架，包括一个用户兴趣重构模块和一个跨领域对比学习模块，以减轻潜在的噪声并促进知识转移。我们在三个真实世界数据集上进行了广泛的实验以验证我们方法的有效性。我们的方法在与竞争基准相比时达到了最先进的性能，特别是在稀疏交互的用户方面。 

---
# Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future 

**Title (ZH)**: 大型语言模型增强的推荐系统：分类、趋势、应用和未来 

**Authors**: Qidong Liu, Xiangyu Zhao, Yuhao Wang, Yejing Wang, Zijian Zhang, Yuqi Sun, Xiang Li, Maolin Wang, Pengyue Jia, Chong Chen, Wei Huang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.13432)  

**Abstract**: Large Language Model (LLM) has transformative potential in various domains, including recommender systems (RS). There have been a handful of research that focuses on empowering the RS by LLM. However, previous efforts mainly focus on LLM as RS, which may face the challenge of intolerant inference costs by LLM. Recently, the integration of LLM into RS, known as LLM-Enhanced Recommender Systems (LLMERS), has garnered significant interest due to its potential to address latency and memory constraints in real-world applications. This paper presents a comprehensive survey of the latest research efforts aimed at leveraging LLM to enhance RS capabilities. We identify a critical shift in the field with the move towards incorporating LLM into the online system, notably by avoiding their use during inference. Our survey categorizes the existing LLMERS approaches into three primary types based on the component of the RS model being augmented: Knowledge Enhancement, Interaction Enhancement, and Model Enhancement. We provide an in-depth analysis of each category, discussing the methodologies, challenges, and contributions of recent studies. Furthermore, we highlight several promising research directions that could further advance the field of LLMERS. 

**Abstract (ZH)**: 大语言模型（LLM）在多个领域具有变革性潜力，包括推荐系统（RS）。已有少数研究专注于通过LLM增强RS。然而，先前的努力主要侧重于将LLM作为RS，这可能会面临LLM容忍性较低的推理成本挑战。最近，将LLM集成到RS中的做法，即LLM增强推荐系统（LLMERS），因有望解决实际应用中的延迟和内存限制问题而引起了广泛关注。本文对旨在利用LLM增强RS能力的最新研究进行了一篇全面综述。我们指出，随着研究领域转向将LLM集成到在线系统中，特别是避免在推理过程中使用LLM，出现了关键性的转变。我们的综述将现有的LLMERS方法划分为三大类，根据增强RS模型的组件分别为知识增强、交互增强和模型增强。我们对每一类进行了深入分析，讨论了近期研究的方法、挑战和贡献。此外，我们还指出了几条有前景的研究方向，这些方向可能会进一步推动LLMERS领域的发展。 

---
# Lightweight yet Fine-grained: A Graph Capsule Convolutional Network with Subspace Alignment for Shared-account Sequential Recommendation 

**Title (ZH)**: 轻量且精细：一种基于子空间对齐的图胶囊卷积网络在共享账户序列推荐中的应用 

**Authors**: Jinyu Zhang, Zhongying Zhao, Chao Li, Yanwei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13408)  

**Abstract**: Shared-account Sequential Recommendation (SSR) aims to provide personalized recommendations for accounts shared by multiple users with varying sequential preferences. Previous studies on SSR struggle to capture the fine-grained associations between interactions and different latent users within the shared account's hybrid sequences. Moreover, most existing SSR methods (e.g., RNN-based or GCN-based methods) have quadratic computational complexities, hindering the deployment of SSRs on resource-constrained devices. To this end, we propose a Lightweight Graph Capsule Convolutional Network with subspace alignment for shared-account sequential recommendation, named LightGC$^2$N. Specifically, we devise a lightweight graph capsule convolutional network. It facilitates the fine-grained matching between interactions and latent users by attentively propagating messages on the capsule graphs. Besides, we present an efficient subspace alignment method. This method refines the sequence representations and then aligns them with the finely clustered preferences of latent users. The experimental results on four real-world datasets indicate that LightGC$^2$N outperforms nine state-of-the-art methods in accuracy and efficiency. 

**Abstract (ZH)**: 共享账户顺序推荐（SSR）旨在为多个用户共用且具有不同顺序偏好的账户提供个性化推荐。以往关于SSR的研究难以捕捉共享账户混合序列中交互与不同潜在用户之间的细粒度关联。此外，大多数现有的SSR方法（例如基于RNN的方法或基于GCN的方法）具有二次的计算复杂度，这阻碍了在资源受限的设备上部署SSR。为了解决这一问题，我们提出了一种轻量级图胶囊卷积网络配以子空间对齐方法，以实现共享账户的顺序推荐，该方法名为LightGC$^2$N。具体而言，我们设计了一种轻量级图胶囊卷积网络，通过注意力传播消息在胶囊图上，实现交互与潜在用户之间的细粒度匹配。此外，我们还提出了一种高效子空间对齐方法，该方法首先细化序列表示，然后将它们与潜在用户的精细聚类偏好对齐。在四个真实数据集上的实验证明，LightGC$^2$N在准确性和效率上均优于九种最先进的方法。 

---
# JudgeBlender: Ensembling Judgments for Automatic Relevance Assessment 

**Title (ZH)**: JudgeBlender：组合判决以实现自动相关性评估 

**Authors**: Hossein A. Rahmani, Emine Yilmaz, Nick Craswell, Bhaskar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2412.13268)  

**Abstract**: The effective training and evaluation of retrieval systems require a substantial amount of relevance judgments, which are traditionally collected from human assessors -- a process that is both costly and time-consuming. Large Language Models (LLMs) have shown promise in generating relevance labels for search tasks, offering a potential alternative to manual assessments. Current approaches often rely on a single LLM, such as GPT-4, which, despite being effective, are expensive and prone to intra-model biases that can favour systems leveraging similar models. In this work, we introduce JudgeBlender, a framework that employs smaller, open-source models to provide relevance judgments by combining evaluations across multiple LLMs (LLMBlender) or multiple prompts (PromptBlender). By leveraging the LLMJudge benchmark [18], we compare JudgeBlender with state-of-the-art methods and the top performers in the LLMJudge challenge. Our results show that JudgeBlender achieves competitive performance, demonstrating that very large models are often unnecessary for reliable relevance assessments. 

**Abstract (ZH)**: 有效的检索系统训练和评估需要大量的相关性判断，这一过程传统上依赖于人类评估者来完成，该过程既耗时又昂贵。大型语言模型（LLMs）已经在生成搜索任务的相关标签方面显示出潜力，这为替代手动评估提供了可能。当前的方法通常依赖单一的LLM，如GPT-4，尽管这种做法有效，但成本较高，并且容易受到模型内部偏差的影响，这些偏差可能倾向于支持使用类似模型的系统。在此项工作中，我们引入了JudgeBlender框架，该框架利用较小的开源模型通过结合多个LLM（LLMBlender）或多个提示（PromptBlender）的评估来提供相关性判断。利用LLMJudge基准测试[18]，我们对比了JudgeBlender与最先进的方法以及LLMJudge挑战中的顶尖表现者。实验结果表明，JudgeBlender达到了具有竞争力的性能，这表明对于可靠的相关性评估，通常不需要使用非常大的模型。 

---
# Adaptive Two-Phase Finetuning LLMs for Japanese Legal Text Retrieval 

**Title (ZH)**: 适应性两阶段微调大规模语言模型以用于日语法律文本检索 

**Authors**: Quang Hoang Trung, Nguyen Van Hoang Phuc, Le Trung Hoang, Quang Huu Hieu, Vo Nguyen Le Duy  

**Link**: [PDF](https://arxiv.org/pdf/2412.13205)  

**Abstract**: Text Retrieval (TR) involves finding and retrieving text-based content relevant to a user's query from a large repository, with applications in real-world scenarios such as legal document retrieval. While most existing studies focus on English, limited work addresses Japanese contexts. In this paper, we introduce a new dataset specifically designed for Japanese legal contexts and propose a novel two-phase pipeline tailored to this domain.
In the first phase, the model learns a broad understanding of global contexts, enhancing its generalization and adaptability to diverse queries. In the second phase, the model is fine-tuned to address complex queries specific to legal scenarios. Extensive experiments are conducted to demonstrate the superior performance of our method, which outperforms existing baselines.
Furthermore, our pipeline proves effective in English contexts, surpassing comparable baselines on the MS MARCO dataset. We have made our code publicly available on GitHub, and the model checkpoints are accessible via HuggingFace. 

**Abstract (ZH)**: 文本检索（Text Retrieval，TR）涉及从大型库中找到并检索与用户查询相关的基于文本的内容，其应用范围包括如法律文件检索在内的实际场景。尽管大部分现有研究主要关注英语，但对于日语环境的研究则相对有限。在本文中，我们介绍了一个专门为日本法律情境设计的新数据集，并提出了一种针对该领域的新型两阶段管道。

在第一阶段，模型学习广泛的全局上下文，提高其对各种查询的泛化能力和适应性。在第二阶段，模型针对具体的法律场景中的复杂查询进行微调。通过广泛实验，证明了我们方法的优越性能，其在现有基准之上表现更佳。

此外，我们的管道在英语情境中也证明了其有效性，在MS MARCO数据集上超过了一些可比的基准。我们已在GitHub上公开了我们的代码，并通过HuggingFace提供了模型检查点。 

---
# Adversarial Hubness in Multi-Modal Retrieval 

**Title (ZH)**: 多模态检索中的对抗性同邻近现象 

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov  

**Link**: [PDF](https://arxiv.org/pdf/2412.14113)  

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries. In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts. We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system based on a tutorial from Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries). We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts. 

**Abstract (ZH)**: 高维向量空间中的“中心点现象”是指自然分布中的单个点与许多其他点异常接近的现象。这是一个在信息检索中广为人知的问题，它会导致一些项目意外（且错误地）对许多查询显得相关。在这篇论文中，我们研究攻击者如何利用中心点现象将任何多模态检索系统中的图像或音频输入转化为对抗性中心点。对抗性中心点可以被用来注入通用的对抗性内容（例如垃圾信息），这些内容会在对成千上万不同的查询响应中被检索出来，同时也可用于针对特定攻击者选定概念的查询目标攻击。我们提出了创造对抗性中心点的方法，并对基准多模态检索数据集和一个基于Pinecone（一个流行的向量数据库）教程的基于图像到图像的检索系统进行了评估。例如，在文本描述到图像检索中，单个对抗性中心点被检索为超过21,000个测试查询中的最相关的图像（相比之下，最常见的自然中心点仅在102个查询中成为最相关的）。我们也研究了减轻自然中心点现象的技术是否能有效防御对抗性中心点，结果表明，它们对针对特定概念查询的目标攻击并不有效。 

---
# RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment 

**Title (ZH)**: RAG-RewardBench：用于偏好对齐的检索增强生成中奖励模型的基准测试 

**Authors**: Zhuoran Jin, Hongbang Yuan, Tianyi Men, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13746)  

**Abstract**: Despite the significant progress made by existing retrieval augmented language models (RALMs) in providing trustworthy responses and grounding in reliable sources, they often overlook effective alignment with human preferences. In the alignment process, reward models (RMs) act as a crucial proxy for human values to guide optimization. However, it remains unclear how to evaluate and select a reliable RM for preference alignment in RALMs. To this end, we propose RAG-RewardBench, the first benchmark for evaluating RMs in RAG settings. First, we design four crucial and challenging RAG-specific scenarios to assess RMs, including multi-hop reasoning, fine-grained citation, appropriate abstain, and conflict robustness. Then, we incorporate 18 RAG subsets, six retrievers, and 24 RALMs to increase the diversity of data sources. Finally, we adopt an LLM-as-a-judge approach to improve preference annotation efficiency and effectiveness, exhibiting a strong correlation with human annotations. Based on the RAG-RewardBench, we conduct a comprehensive evaluation of 45 RMs and uncover their limitations in RAG scenarios. Additionally, we also reveal that existing trained RALMs show almost no improvement in preference alignment, highlighting the need for a shift towards preference-aligned this http URL release our benchmark and code publicly at this https URL for future work. 

**Abstract (ZH)**: 尽管现有的检索增强语言模型（RALMs）在提供可信回答和可靠信息源依据方面取得了显著进展，但它们往往忽视了与人类偏好有效对齐的重要性。在对齐过程中，奖励模型（RMs）作为人类价值观的关键代理，对优化过程起到指导作用。然而，如何评估和选择适用于RALMs的可靠RMs来实现偏好对齐仍不清晰。为了解决这一问题，我们提出了RAG-RewardBench，这是首个用于评估RAG环境下RMs的基准。首先，我们设计了四个关键且具有挑战性的RAG特定场景来评估RMs，包括多跳推理、精细引用、适当回避以及冲突稳健性。其次，我们整合了18个RAG子集、六种检索器和24种RALMs，以增加数据源多样性。最后，我们采用LLM作为裁判的方法提高了偏好注释的效率和有效性，该方法与人类注释具有较强的关联性。基于RAG-RewardBench，我们对45种RMs进行了全面评估，并揭示了它们在RAG场景中的局限性。此外，我们还发现现有的训练过的RALMs在偏好对齐方面几乎没有改进，突显了转向偏好对齐的必要性。我们将在https://github.com/your-repo/reward-bench-public提供我们的基准和代码，供未来研究使用。 

---
# Reverse Region-to-Entity Annotation for Pixel-Level Visual Entity Linking 

**Title (ZH)**: 像素级视觉实体链接中的反向区域到实体标注 

**Authors**: Zhengfei Xu, Sijia Zhao, Yanchao Hao, Xiaolong Liu, Lili Li, Yuyang Yin, Bo Li, Xi Chen, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13614)  

**Abstract**: Visual Entity Linking (VEL) is a crucial task for achieving fine-grained visual understanding, matching objects within images (visual mentions) to entities in a knowledge base. Previous VEL tasks rely on textual inputs, but writing queries for complex scenes can be challenging. Visual inputs like clicks or bounding boxes offer a more convenient alternative. Therefore, we propose a new task, Pixel-Level Visual Entity Linking (PL-VEL), which uses pixel masks from visual inputs to refer to objects, supplementing reference methods for VEL. To facilitate research on this task, we have constructed the MaskOVEN-Wiki dataset through an entirely automatic reverse region-entity annotation framework. This dataset contains over 5 million annotations aligning pixel-level regions with entity-level labels, which will advance visual understanding towards fine-grained. Moreover, as pixel masks correspond to semantic regions in an image, we enhance previous patch-interacted attention with region-interacted attention by a visual semantic tokenization approach. Manual evaluation results indicate that the reverse annotation framework achieved a 94.8% annotation success rate. Experimental results show that models trained on this dataset improved accuracy by 18 points compared to zero-shot models. Additionally, the semantic tokenization method achieved a 5-point accuracy improvement over the trained baseline. 

**Abstract (ZH)**: 视觉实体链接（VEL）是实现细粒度视觉理解的关键任务，它涉及将图像中的对象（视觉提及）与知识库中的实体相匹配。之前的VEL任务依赖于文本输入，但为复杂场景编写查询可能具有挑战性。通过点击或边界框等视觉输入可以提供一种更方便的选择。因此，我们提出了一个新的任务——像素级视觉实体链接（PL-VEL），该任务利用视觉输入的像素掩码来引用对象，补充了VL任务中的引用方法。为了促进该任务的研究，我们通过一个完全自动的反向区域-实体注释框架构建了MaskOVEN-Wiki数据集。该数据集包含超过500万条注释，将像素级区域与实体级标签对齐，从而推动视觉理解向细粒度方向发展。此外，由于像素掩码对应于图像中的语义区域，我们通过视觉语义分词方法增强了之前的局部交互注意力机制，引入了区域交互注意力机制。人工评估结果显示，反向注释框架的成功率为94.8%。实验结果表明，基于该数据集训练的模型在准确率上比零样本模型提高了18个百分点。此外，语义分词方法在训练基线上的准确率实现了5个百分点的提升。 

---
# Information-Theoretic Generative Clustering of Documents 

**Title (ZH)**: 信息论生成聚类文档方法 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2412.13534)  

**Abstract**: We present {\em generative clustering} (GC) for clustering a set of documents, $\mathrm{X}$, by using texts $\mathrm{Y}$ generated by large language models (LLMs) instead of by clustering the original documents $\mathrm{X}$. Because LLMs provide probability distributions, the similarity between two documents can be rigorously defined in an information-theoretic manner by the KL divergence. We also propose a natural, novel clustering algorithm by using importance sampling. We show that GC achieves the state-of-the-art performance, outperforming any previous clustering method often by a large margin. Furthermore, we show an application to generative document retrieval in which documents are indexed via hierarchical clustering and our method improves the retrieval accuracy. 

**Abstract (ZH)**: 我们提出了一种生成聚类（Generative Clustering, GC）的方法，通过使用大型语言模型（LLMs）生成的文本集合$\mathrm{Y}$进行文档聚类，而不是直接对原始文档集合$\mathrm{X}$进行聚类。由于大型语言模型提供了概率分布，两份文档之间的相似度可以以信息论的方式通过KL散度严格定义。此外，我们还提出了一种基于重要性采样的新颖聚类算法。实验结果显示，GC方法达到最先进的性能，通常在性能上显著超越了之前的任何聚类方法。此外，我们展示了生成式文档检索的应用，其中文档通过层次聚类进行索引，我们的方法可以提高检索准确性。 

---
# AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark 

**Title (ZH)**: AIR-Bench：自动化异构信息检索基准测试 

**Authors**: Jianlyu Chen, Nan Wang, Chaofan Li, Bo Wang, Shitao Xiao, Han Xiao, Hao Liao, Defu Lian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13102)  

**Abstract**: Evaluation plays a crucial role in the advancement of information retrieval (IR) models. However, current benchmarks, which are based on predefined domains and human-labeled data, face limitations in addressing evaluation needs for emerging domains both cost-effectively and efficiently. To address this challenge, we propose the Automated Heterogeneous Information Retrieval Benchmark (AIR-Bench). AIR-Bench is distinguished by three key features: 1) Automated. The testing data in AIR-Bench is automatically generated by large language models (LLMs) without human intervention. 2) Heterogeneous. The testing data in AIR-Bench is generated with respect to diverse tasks, domains and languages. 3) Dynamic. The domains and languages covered by AIR-Bench are constantly augmented to provide an increasingly comprehensive evaluation benchmark for community developers. We develop a reliable and robust data generation pipeline to automatically create diverse and high-quality evaluation datasets based on real-world corpora. Our findings demonstrate that the generated testing data in AIR-Bench aligns well with human-labeled testing data, making AIR-Bench a dependable benchmark for evaluating IR models. The resources in AIR-Bench are publicly available at this https URL. 

**Abstract (ZH)**: 评价在信息检索（IR）模型的发展中扮演着至关重要的角色。然而，当前的基准测试大多基于预定义领域和人工标注的数据，这在解决新兴领域评价需求时存在成本效益和效率上的局限性。为应对这一挑战，我们提出了自动异构信息检索基准（AIR-Bench）。AIR-Bench 有三个关键特点：1）自动化。AIR-Bench 的测试数据是由大型语言模型（LLMs）自动生成的，无需人工干预。2）异构性。AIR-Bench 的测试数据涵盖多种任务、领域和语言。3）动态性。AIR-Bench 所涵盖的领域和语言不断扩展，以提供一个越来越全面的评价基准，供社区开发者使用。我们开发了一条可靠且稳健的数据生成管道，基于现实世界的语料库自动创建多样性和高质量的评价数据集。我们的研究结果表明，AIR-Bench 生成的测试数据与人工标注的测试数据高度一致，使 AIR-Bench 成为评价 IR 模型的可靠基准。相关的资源可以在以下网址获取：https://example.com（请注意替换为实际网址）。 

---
# A Survey on Recommendation Unlearning: Fundamentals, Taxonomy, Evaluation, and Open Questions 

**Title (ZH)**: 推荐学习取消研究综述：基础、分类、评估及开放问题 

**Authors**: Yuyuan Li, Xiaohua Feng, Chaochao Chen, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12836)  

**Abstract**: Recommender systems have become increasingly influential in shaping user behavior and decision-making, highlighting their growing impact in various domains. Meanwhile, the widespread adoption of machine learning models in recommender systems has raised significant concerns regarding user privacy and security. As compliance with privacy regulations becomes more critical, there is a pressing need to address the issue of recommendation unlearning, i.e., eliminating the memory of specific training data from the learned recommendation models. Despite its importance, traditional machine unlearning methods are ill-suited for recommendation unlearning due to the unique challenges posed by collaborative interactions and model parameters. This survey offers a comprehensive review of the latest advancements in recommendation unlearning, exploring the design principles, challenges, and methodologies associated with this emerging field. We provide a unified taxonomy that categorizes different recommendation unlearning approaches, followed by a summary of widely used benchmarks and metrics for evaluation. By reviewing the current state of research, this survey aims to guide the development of more efficient, scalable, and robust recommendation unlearning techniques. Furthermore, we identify open research questions in this field, which could pave the way for future innovations not only in recommendation unlearning but also in a broader range of unlearning tasks across different machine learning applications. 

**Abstract (ZH)**: 推荐系统在塑造用户行为和决策方面的作用越来越重要，这突显了它们在各个领域影响力的不断增强。与此同时，推荐系统中广泛采用机器学习模型引发了对用户隐私和安全的重大关注。随着遵守隐私法规变得愈加重要，消除特定训练数据的推荐记忆的需求迫在眉睫。尽管这一点很重要，但传统的机器遗忘方法由于协作交互和模型参数的独特挑战，并不适用于推荐系统的遗忘。本文综述了推荐系统遗忘方面的最新进展，探讨了这一新兴领域的设计原则、挑战和方法论。我们提供了一个统一的分类体系，将不同的推荐系统遗忘方法进行归类，并总结了常用的标准和评估指标。通过对当前研究的综述，本文旨在指导更高效、更具扩展性和鲁棒性的推荐系统遗忘技术的发展。此外，我们还指出了该领域中的开放研究问题，这可能为推荐系统遗忘以及其他不同机器学习应用中的遗忘任务的未来创新奠定基础。 

---
# RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service 

**Title (ZH)**: RemoteRAG：一种隐私保护的大语言模型云检索增强服务 

**Authors**: Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan, Yunhao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2412.12775)  

**Abstract**: Retrieval-augmented generation (RAG) improves the service quality of large language models by retrieving relevant documents from credible literature and integrating them into the context of the user query. Recently, the rise of the cloud RAG service has made it possible for users to query relevant documents conveniently. However, directly sending queries to the cloud brings potential privacy leakage. In this paper, we are the first to formally define the privacy-preserving cloud RAG service to protect the user query and propose RemoteRAG as a solution regarding privacy, efficiency, and accuracy. For privacy, we introduce $(n,\epsilon)$-DistanceDP to characterize privacy leakage of the user query and the leakage inferred from relevant documents. For efficiency, we limit the search range from the total documents to a small number of selected documents related to a perturbed embedding generated from $(n,\epsilon)$-DistanceDP, so that computation and communication costs required for privacy protection significantly decrease. For accuracy, we ensure that the small range includes target documents related to the user query with detailed theoretical analysis. Experimental results also demonstrate that RemoteRAG can resist existing embedding inversion attack methods while achieving no loss in retrieval under various settings. Moreover, RemoteRAG is efficient, incurring only $0.67$ seconds and $46.66$KB of data transmission ($2.72$ hours and $1.43$ GB with the non-optimized privacy-preserving scheme) when retrieving from a total of $10^6$ documents. 

**Abstract (ZH)**: 检索增强生成（RAG）通过从可靠文献中检索相关文档并将其整合到用户的查询语境中，提高了大型语言模型的服务质量。最近，云RAG服务的兴起使得用户可以方便地查询相关文档。然而，直接将查询发送到云中会带来潜在的隐私泄露风险。本文首次正式定义了保护用户查询隐私的云RAG服务，并提出RemoteRAG作为在隐私、效率和准确性方面的解决方案。为了保护隐私，我们引入了基于$(n,\epsilon)$-DistanceDP的隐私泄露表征，以量化用户查询及其从相关文档推断出的隐私泄露。为了提高效率，我们将搜索范围限制在从总文档中筛选出的一小部分与$(n,\epsilon)$-DistanceDP生成的扰动嵌入相关的文档，从而显著减少了保护隐私所需的计算和通信成本。为了保证准确性，我们通过详细的理论分析确保该小范围内包含与用户查询相关的目标文档。实验结果还表明，RemoteRAG可以在多种设置下抵抗现存的嵌入反转攻击方法，同时在检索方面保持无损失。此外，RemoteRAG在从总共$10^6$份文档中检索时仅需要0.67秒和46.66KB的数据传输（未经优化的隐私保护方案需要2.72小时和1.43GB的数据传输）。 

---
# A Survey on Sequential Recommendation 

**Title (ZH)**: 序贯推荐综述 

**Authors**: Liwei Pan, Weike Pan, Meiyan Wei, Hongzhi Yin, Zhong Ming  

**Link**: [PDF](https://arxiv.org/pdf/2412.12770)  

**Abstract**: Different from most conventional recommendation problems, sequential recommendation focuses on learning users' preferences by exploiting the internal order and dependency among the interacted items, which has received significant attention from both researchers and practitioners. In recent years, we have witnessed great progress and achievements in this field, necessitating a new survey. In this survey, we study the SR problem from a new perspective (i.e., the construction of an item's properties), and summarize the most recent techniques used in sequential recommendation such as pure ID-based SR, SR with side information, multi-modal SR, generative SR, LLM-powered SR, ultra-long SR and data-augmented SR. Moreover, we introduce some frontier research topics in sequential recommendation, e.g., open-domain SR, data-centric SR, could-edge collaborative SR, continuous SR, SR for good, and explainable SR. We believe that our survey could be served as a valuable roadmap for readers in this field. 

**Abstract (ZH)**: 与大多数传统推荐问题不同，序列推荐关注通过利用用户交互项之间的内部顺序和依赖关系来学习用户偏好，这一领域得到了研究者和实践者的广泛关注。近年来，该领域取得了显著的进步和成果，因此迫切需要进行新的综述。在本次综述中，我们从一个新的角度（即项目属性的构建）研究序列推荐问题，并总结了目前应用于序列推荐的最新技术，例如基于纯ID的序列推荐、包含侧信息的序列推荐、多模态序列推荐、生成式序列推荐、基于大语言模型的序列推荐、超长序列推荐以及数据增强的序列推荐。此外，我们介绍了序列推荐领域的前沿研究主题，例如开放领域序列推荐、数据驱动的序列推荐、云边协同的序列推荐、连续序列推荐、有益的序列推荐以及可解释的序列推荐。我们相信，本次综述能够为该领域的读者提供有价值的指引。 

---
# Token-Level Graphs for Short Text Classification 

**Title (ZH)**: 短文本分类中的令牌级图方法 

**Authors**: Gregor Donabauer, Udo Kruschwitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.12754)  

**Abstract**: The classification of short texts is a common subtask in Information Retrieval (IR). Recent advances in graph machine learning have led to interest in graph-based approaches for low resource scenarios, showing promise in such settings. However, existing methods face limitations such as not accounting for different meanings of the same words or constraints from transductive approaches. We propose an approach which constructs text graphs entirely based on tokens obtained through pre-trained language models (PLMs). By applying a PLM to tokenize and embed the texts when creating the graph(-nodes), our method captures contextual and semantic information, overcomes vocabulary constraints, and allows for context-dependent word meanings. Our approach also makes classification more efficient with reduced parameters compared to classical PLM fine-tuning, resulting in more robust training with few samples. Experimental results demonstrate how our method consistently achieves higher scores or on-par performance with existing methods, presenting an advancement in graph-based text classification techniques. To support reproducibility of our work we make all implementations publicly available to the community\footnote{\url{this https URL}}. 

**Abstract (ZH)**: 短文本分类是信息检索（IR）中的一个常见子任务。近年来，图机器学习的进展激发了在资源有限场景中使用基于图的方法的兴趣，这些方法在这样的环境中显示出潜力。然而，现有方法存在一些局限性，例如没有考虑到同义词的不同含义或限制性归纳方法的约束。我们提出了一种方法，该方法完全基于预训练语言模型（PLMs）获取的令牌构建文本图。在创建图（节点）时，通过应用PLM对文本进行分词并嵌入，我们的方法能够捕捉上下文和语义信息，克服词汇量限制，并且能够在不同上下文中灵活解释词义。相比经典的PLM微调方法，我们的方法在参数减少的情况下提高了分类效率，从而在少量样本下实现更稳健的训练。实验结果表明，我们的方法在性能上通常优于或与现有方法持平，这在图基于的文本分类技术方面构成了进展。为了支持研究的可复现性，我们已将所有实现公开给社区（\url{此处应填写具体的URL链接}）。 

---
# Boosting LLM-based Relevance Modeling with Distribution-Aware Robust Learning 

**Title (ZH)**: 基于分布意识稳健学习的LLM驱动相关性建模增强 

**Authors**: Hong Liu, Saisai Gong, Yixin Ji, Kaixin Wu, Jia Xu, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2412.12504)  

**Abstract**: With the rapid advancement of pre-trained large language models (LLMs), recent endeavors have leveraged the capabilities of LLMs in relevance modeling, resulting in enhanced performance. This is usually done through the process of fine-tuning LLMs on specifically annotated datasets to determine the relevance between queries and items. However, there are two limitations when LLMs are naively employed for relevance modeling through fine-tuning and inference. First, it is not inherently efficient for performing nuanced tasks beyond simple yes or no answers, such as assessing search relevance. It may therefore tend to be overconfident and struggle to distinguish fine-grained degrees of relevance (e.g., strong relevance, weak relevance, irrelevance) used in search engines. Second, it exhibits significant performance degradation when confronted with data distribution shift in real-world scenarios. In this paper, we propose a novel Distribution-Aware Robust Learning framework (DaRL) for relevance modeling in Alipay Search. Specifically, we design an effective loss function to enhance the discriminability of LLM-based relevance modeling across various fine-grained degrees of query-item relevance. To improve the generalizability of LLM-based relevance modeling, we first propose the Distribution-Aware Sample Augmentation (DASA) module. This module utilizes out-of-distribution (OOD) detection techniques to actively select appropriate samples that are not well covered by the original training set for model fine-tuning. Furthermore, we adopt a multi-stage fine-tuning strategy to simultaneously improve in-distribution (ID) and OOD performance, bridging the performance gap between them. DaRL has been deployed online to serve the Alipay's insurance product search... 

**Abstract (ZH)**: 随着预训练大型语言模型（LLMs）的迅速发展，最近的研究已经利用了LLMs在相关性建模方面的能力，从而提升了模型的性能。这通常是通过在特定注释数据集上微调LLMs来确定查询与项目之间的相关性来实现的。然而，当LLMs通过微调和推理来简单地进行相关性建模时，存在两个主要限制。首先，LLMs在执行简单的是或否之外的复杂任务时并不高效，例如评估搜索相关性。因此，它们可能过于自信并难以区分搜索引擎中使用的细微相关性程度（如强相关、弱相关和不相关）。其次，它们在现实世界场景中遇到数据分布变化时会表现出显著的性能下降。在这篇论文中，我们提出了一种新型的“aware分布鲁棒学习”框架（DaRL）用于支付宝搜索的相关性建模。具体来说，我们设计了一个有效的损失函数，以增强基于LLMs的相关性建模在各种细微的相关性程度中的区辨力。为了提高基于LLMs的相关性建模的泛化能力，我们首先提出了一种“aware分布样本增强”（DASA）模块。该模块利用出了领域分布（OOD）检测技术，主动选择原来训练集中覆盖不足的适当样本进行模型微调。此外，我们采用了多阶段微调策略，同时提升领域内（ID）和领域外（OOD）性能，缩小两者之间的性能差距。DaRL已经在支付宝保险产品搜索中上线部署…… 

---
# LLM is Knowledge Graph Reasoner: LLM's Intuition-aware Knowledge Graph Reasoning for Cold-start Sequential Recommendation 

**Title (ZH)**: 大语言模型是知识图谱推理器：大语言模型在冷启动序列推荐中的直觉驱动知识图谱推理 

**Authors**: Keigo Sakurai, Ren Togo, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2412.12464)  

**Abstract**: Knowledge Graphs (KGs) represent relationships between entities in a graph structure and have been widely studied as promising tools for realizing recommendations that consider the accurate content information of items. However, traditional KG-based recommendation methods face fundamental challenges: insufficient consideration of temporal information and poor performance in cold-start scenarios. On the other hand, Large Language Models (LLMs) can be considered databases with a wealth of knowledge learned from the web data, and they have recently gained attention due to their potential application as recommendation systems. Although approaches that treat LLMs as recommendation systems can leverage LLMs' high recommendation literacy, their input token limitations make it impractical to consider the entire recommendation domain dataset and result in scalability issues. To address these challenges, we propose a LLM's Intuition-aware Knowledge graph Reasoning model (LIKR). Our main idea is to treat LLMs as reasoners that output intuitive exploration strategies for KGs. To integrate the knowledge of LLMs and KGs, we trained a recommendation agent through reinforcement learning using a reward function that integrates different recommendation strategies, including LLM's intuition and KG embeddings. By incorporating temporal awareness through prompt engineering and generating textual representations of user preferences from limited interactions, LIKR can improve recommendation performance in cold-start scenarios. Furthermore, LIKR can avoid scalability issues by using KGs to represent recommendation domain datasets and limiting the LLM's output to KG exploration strategies. Experiments on real-world datasets demonstrate that our model outperforms state-of-the-art recommendation methods in cold-start sequential recommendation scenarios. 

**Abstract (ZH)**: 知识图谱（KGs）以图结构形式表示实体之间的关系，并已被广泛研究作为实现考虑项目准确内容信息的推荐的有潜力工具。然而，传统的基于知识图谱的推荐方法面临根本性的挑战：时间信息考虑不足以及在冷启动场景中的表现较差。另一方面，大型语言模型（LLMs）可以被视为从网络数据中学习了大量知识的数据库，并且由于其作为推荐系统的潜在应用而最近受到了关注。尽管将LLMs视为推荐系统的做法可以利用LLMs的高推荐素养，但由于其输入令牌的限制，考虑整个推荐领域数据集变得不切实际，导致可扩展性问题。为了解决这些挑战，我们提出了一种LLM的直觉驱动的知识图谱推理模型（LIKR）。我们的主要思想是将LLMs视为推理器，输出对知识图谱进行直观探索的战略。为了整合LLMs和KGs的知识，我们通过强化学习训练了一个推荐代理，并使用结合了不同推荐策略的奖励函数，包括LLM的直觉和KG嵌入。通过提示工程技术增强时间意识，并从有限的交互中生成用户的偏好文本表示，LIKR可以在冷启动场景中提升推荐性能。此外，LIKR可以通过使用KG表示推荐领域数据集并将LLM的输出限制为KG探索策略来避免可扩展性问题。实验表明，我们的模型在冷启动序列推荐场景中优于最先进的推荐方法。 

---
# Searching Personal Collections 

**Title (ZH)**: 搜索个人收藏 

**Authors**: Michael Bendersky, Donald Metzler, Marc Najork, Xuanhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.12330)  

**Abstract**: This article describes the history of information retrieval on personal document collections. 

**Abstract (ZH)**: 本文描述了个人文档集合中信息检索的历史。 

---
# Enhancing the conformal predictability of context-aware recommendation systems by using Deep Autoencoders 

**Title (ZH)**: 使用深层自编码器增强基于上下文的认知推荐系统的拟合可预测性 

**Authors**: Saloua Zammali, Siddhant Dutta, Sadok Ben Yahia  

**Link**: [PDF](https://arxiv.org/pdf/2412.12110)  

**Abstract**: In the field of Recommender Systems (RS), neural collaborative filtering represents a significant milestone by combining matrix factorization and deep neural networks to achieve promising results. Traditional methods like matrix factorization often rely on linear models, limiting their capability to capture complex interactions between users, items, and contexts. This limitation becomes particularly evident with high-dimensional datasets due to their inability to capture relationships among users, items, and contextual factors. Unsupervised learning and dimension reduction tasks utilize autoencoders, neural network-based models renowned for their capacity to encode and decode data. Autoencoders learn latent representations of inputs, reducing dataset size while capturing complex patterns and features. In this paper, we introduce a framework that combines neural contextual matrix factorization with autoencoders to predict user ratings for items. We provide a comprehensive overview of the framework's design and implementation. To evaluate its performance, we conduct experiments on various real-world datasets and compare the results against state-of-the-art approaches. We also extend the concept of conformal prediction to prediction rating and introduce a Conformal Prediction Rating (CPR). For RS, we define the nonconformity score, a key concept of conformal prediction, and demonstrate that it satisfies the exchangeability property. 

**Abstract (ZH)**: 在推荐系统（RS）领域，神经协作过滤标志着一个重要里程碑，通过结合矩阵分解和深度神经网络以实现有希望的结果。传统的矩阵分解等方法通常依赖于线性模型，限制了它们捕获用户、项目和上下文之间复杂交互的能力。在高维数据集上，这种限制尤为明显，因为它们无法捕获用户、项目和上下文因素之间的关系。无监督学习和降维任务中使用的自动编码器是一种基于神经网络的模型，因其编码和解码数据的能力而闻名。自动编码器学习输入的潜在表示，从而减少数据集的大小并捕捉到复杂的模式和特征。在本文中，我们提出了一种结合神经上下文矩阵分解与自动编码器的框架，以预测用户对项目的评分。我们提供了该框架的设计和实现的全面概述。为了评估其性能，我们在多个实际数据集上进行了实验，并将结果与最新的方法进行了比较。我们还扩展了集合理论中的可校准预测到评分预测，并介绍了可校准预测评分（CPR）。对于推荐系统，我们定义了非一致性评分这一可校准预测的关键概念，并证明它满足交换性性质。 

---
# Re-calibrating methodologies in social media research: Challenge the visual, work with Speech 

**Title (ZH)**: 在社交媒体研究中重新校准方法：质疑视觉呈现，重视语音数据 

**Authors**: Hongrui Jin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13170)  

**Abstract**: This article methodologically reflects on how social media scholars can effectively engage with speech-based data in their analyses. While contemporary media studies have embraced textual, visual, and relational data, the aural dimension remained comparatively under-explored. Building on the notion of secondary orality and rejection towards purely visual culture, the paper argues that considering voice and speech at scale enriches our understanding of multimodal digital content. The paper presents the TikTok Subtitles Toolkit that offers accessible speech processing readily compatible with existing workflows. In doing so, it opens new avenues for large-scale inquiries that blend quantitative insights with qualitative precision. Two illustrative cases highlight both opportunities and limitations of speech research: while genres like #storytime on TikTok benefit from the exploration of spoken narratives, nonverbal or music-driven content may not yield significant insights using speech data. The article encourages researchers to integrate aural exploration thoughtfully to complement existing methods, rather than replacing them. I conclude that the expansion of our methodological repertoire enables richer interpretations of platformised content, and our capacity to unpack digital cultures as they become increasingly multimodal. 

**Abstract (ZH)**: 本文从方法论的角度探讨了社交媒体学者如何有效地在其分析中利用基于言语的数据。虽然当代理论媒体研究已经接纳了文本、视觉和关系数据，但听觉维度相对而言被探索较少。基于次级口语的概念和对纯粹视觉文化的排斥，本文主张在大规模范围内考虑声音和言语能够丰富我们对多模态数字内容的理解。本文介绍了抖音字幕工具包，该工具包提供了易于访问的言语处理方式，并与现有的工作流程无缝兼容。通过这种方式，它为融合定量洞察与定性精确性的大规模研究开启了新的途径。两个案例揭示了言语研究的机遇与局限性：尽管如抖音上的#storytime等类别的内容可以从言语叙述的探索中获益，但非言语或音乐驱动的内容可能无法从言语数据中提取出显著的洞见。本文鼓励研究者将听觉探索融入现有方法以补充现有方法，而非取代它们。我得出结论认为，扩展方法论方法库能够使我们对平台化内容进行更丰富的解释，并增强我们拆解越来越具备多模态特性的数字文化的能力。 

---
# C-FedRAG: A Confidential Federated Retrieval-Augmented Generation System 

**Title (ZH)**: C-FedRAG：一种保密 Federated Retrieval-Augmented Generation 系统

解释：
- "C" 在这个上下文中代表 "Confidential"，即保密的。
- "FedRAG" 保持不变，因为它是一个特定的系统命名。
- "Federated" 直接翻译为“联邦的”或“协作的”，在技术领域常译为“联邦学习”或"Federated"。
- "Retrieval-Augmented Generation" 可以翻译为“检索增强生成”，这个术语在自然语言处理领域较为常见。

这样的翻译既符合学术规范，又能准确传达原文的意思。 

**Authors**: Parker Addison, Minh-Tuan H. Nguyen, Tomislav Medan, Mohammad T. Manzari, Brendan McElrone, Laksh Lalwani, Aboli More, Smita Sharma, Holger R. Roth, Isaac Yang, Chester Chen, Daguang Xu, Yan Cheng, Andrew Feng, Ziyue Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13163)  

**Abstract**: Organizations seeking to utilize Large Language Models (LLMs) for knowledge querying and analysis often encounter challenges in maintaining an LLM fine-tuned on targeted, up-to-date information that keeps answers relevant and grounded. Retrieval Augmented Generation (RAG) has quickly become a feasible solution for organizations looking to overcome the challenges of maintaining proprietary models and to help reduce LLM hallucinations in their query responses. However, RAG comes with its own issues regarding scaling data pipelines across tiered-access and disparate data sources. In many scenarios, it is necessary to query beyond a single data silo to provide richer and more relevant context for an LLM. Analyzing data sources within and across organizational trust boundaries is often limited by complex data-sharing policies that prohibit centralized data storage, therefore, inhibit the fast and effective setup and scaling of RAG solutions. In this paper, we introduce Confidential Computing (CC) techniques as a solution for secure Federated Retrieval Augmented Generation (FedRAG). Our proposed Confidential FedRAG system (C-FedRAG) enables secure connection and scaling of a RAG workflows across a decentralized network of data providers by ensuring context confidentiality. We also demonstrate how to implement a C-FedRAG system using the NVIDIA FLARE SDK and assess its performance using the MedRAG toolkit and MIRAGE benchmarking dataset. 

**Abstract (ZH)**: 组织利用大型语言模型（LLMs）进行知识查询和分析时，常常会面临保持模型针对特定、最新信息进行微调的挑战，以确保答案的相关性和真实性。检索增强生成（RAG）迅速成为了解决组织维护专有模型及减少LLM查询响应中妄想现象的一种可行方案。然而，RAG 在跨层级访问和异构数据源扩展数据管道方面也带来了一些问题。在许多情况下，为了提供更丰富和相关的情境，需要超越单一数据孤岛进行查询。然而，分析组织信任边界内外的数据源时常受到复杂数据共享政策的限制，这些政策通常禁止集中式数据存储，从而妨碍了RAG解决方案的快速和有效部署与扩展。在本文中，我们介绍了一种使用保密计算（CC）技术的安全联邦检索增强生成（FedRAG）方案。我们的提议的保密联邦检索增强生成系统（C-FedRAG）能够在分布式数据提供者网络中确保上下文的保密性，从而实现RAG工作流程的安全连接和扩展。我们还展示了如何使用NVIDIA FLARE SDK实现C-FedRAG系统，并使用MedRAG工具包和MIRAGE基准测试数据集评估其性能。 

---
