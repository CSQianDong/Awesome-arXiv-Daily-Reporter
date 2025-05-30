# SampleLLM: Optimizing Tabular Data Synthesis in Recommendations 

**Title (ZH)**: SampleLLM：优化推荐系统中表格数据合成 

**Authors**: Jingtong Gao, Zhaocheng Du, Xiaopeng Li, Xiangyu Zhao, Yichao Wang, Xiangyang Li, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16125)  

**Abstract**: Tabular data synthesis is crucial in machine learning, yet existing general methods-primarily based on statistical or deep learning models-are highly data-dependent and often fall short in recommender systems. This limitation arises from their difficulty in capturing complex distributions and understanding feature relationships from sparse and limited data, along with their inability to grasp semantic feature relations. Recently, Large Language Models (LLMs) have shown potential in generating synthetic data samples through few-shot learning and semantic understanding. However, they often suffer from inconsistent distribution and lack of diversity due to their inherent distribution disparity with the target dataset. To address these challenges and enhance tabular data synthesis for recommendation tasks, we propose a novel two-stage framework named SampleLLM to improve the quality of LLM-based tabular data synthesis for recommendations by ensuring better distribution alignment. In the first stage, SampleLLM employs LLMs with Chain-of-Thought prompts and diverse exemplars to generate data that closely aligns with the target dataset distribution, even when input samples are limited. The second stage uses an advanced feature attribution-based importance sampling method to refine feature relationships within the synthesized data, reducing any distribution biases introduced by the LLM. Experimental results on three recommendation datasets, two general datasets, and online deployment illustrate that SampleLLM significantly surpasses existing methods for recommendation tasks and holds promise for a broader range of tabular data scenarios. 

**Abstract (ZH)**: 表格数据合成在机器学习中至关重要，然而现有的通用方法——主要是基于统计或深度学习模型的方法——高度依赖于特定数据集，并且在推荐系统中往往效果不佳。这一局限性源于它们在捕捉复杂分布和理解稀疏有限数据中的特征关系方面存在困难，以及无法把握语义特征关系。最近，大型语言模型（LLMs）在通过少样本学习和语义理解生成合成数据样本方面显示出潜力。然而，由于与目标数据集固有的分布差异，它们往往会导致分布不一致和多样性不足。为了应对这些挑战并提高推荐任务中的表格数据合成质量，我们提出了一种名为SampleLLM的新两阶段框架，通过确保更好的分布对齐来增强基于LLM的表格数据合成质量。在第一阶段，SampleLLM利用带有链式思维提示和多样示例的LLMs生成与目标数据集分布紧密对齐的数据，即使输入样本有限。第二阶段使用高级特征归因为基础的重要性采样方法进一步细化合成数据中的特征关系，减少LLM引入的任何分布偏差。在三个推荐数据集、两个通用数据集以及在线部署实验中，SampleLLM的表现显著优于现有方法，并且为更广泛的表格数据场景提供了可能性。 

---
# Survey: Understand the challenges of MachineLearning Experts using Named EntityRecognition Tools 

**Title (ZH)**: 调查研究：了解使用命名实体识别工具的机器学习专家所面临的挑战 

**Authors**: Florian Freund, Philippe Tamla, Matthias Hemmje  

**Link**: [PDF](https://arxiv.org/pdf/2501.16112)  

**Abstract**: This paper presents a survey based on Kasunic's survey research methodology to identify the criteria used by Machine Learning (ML) experts to evaluate Named Entity Recognition (NER) tools and frameworks. Comparison and selection of NER tools and frameworks is a critical step in leveraging NER for Information Retrieval to support the development of Clinical Practice Guidelines. In addition, this study examines the main challenges faced by ML experts when choosing suitable NER tools and frameworks. Using Nunamaker's methodology, the article begins with an introduction to the topic, contextualizes the research, reviews the state-of-the-art in science and technology, and identifies challenges for an expert survey on NER tools and frameworks. This is followed by a description of the survey's design and implementation. The paper concludes with an evaluation of the survey results and the insights gained, ending with a summary and conclusions. 

**Abstract (ZH)**: 本文基于Kasunic的调查研究方法，对机器学习（ML）专家用于评估命名实体识别（NER）工具和框架的标准进行了调查。选择合适的NER工具和框架是利用NER支持临床实践指南开发中的关键步骤。此外，本研究还探讨了ML专家在选择适合的NER工具和框架时面临的挑战。本文采用Nunamaker的调查研究方法，首先对研究主题进行了介绍，对科学和技术的现状进行了回顾，并指出了针对NER工具和框架专家调查的研究挑战。随后详细描述了调查的设计与实施过程。最后，本文通过对调查结果的评估和获得的见解进行了总结，并提出了结论。 

---
# Options-Aware Dense Retrieval for Multiple-Choice query Answering 

**Title (ZH)**: 基于选项意识的密集检索方法用于多项选择查询作答 

**Authors**: Manish Singh, Manish Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2501.16111)  

**Abstract**: Long-context multiple-choice question answering tasks require robust reasoning over extensive text sources. Since most of the pre-trained transformer models are restricted to processing only a few hundred words at a time, successful completion of such tasks often relies on the identification of evidence spans, such as sentences, that provide supporting evidence for selecting the correct answer. Prior research in this domain has predominantly utilized pre-trained dense retrieval models, given the absence of supervision to fine-tune the retrieval process. This paper proposes a novel method called Options Aware Dense Retrieval (OADR) to address these challenges. ORDA uses an innovative approach to fine-tuning retrieval by leveraging query-options embeddings, which aim to mimic the embeddings of the oracle query (i.e., the query paired with the correct answer) for enhanced identification of supporting evidence. Through experiments conducted on the QuALITY benchmark dataset, we demonstrate that our proposed model surpasses existing baselines in terms of performance and accuracy. 

**Abstract (ZH)**: 长上下文多选题回答任务需要在广泛的文本来源中进行 robust 的推理。由于大多数预训练的变换器模型只能一次处理几百个词，因此成功完成此类任务往往依赖于识别能够支持选择正确答案的证据片段，例如句子。在此领域之前的研究所使用的主要是预训练密集检索模型，由于缺乏对检索过程进行微调的监督。本文提出了一种名为 Options Aware Dense Retrieval (OADR) 的新方法，以解决这些挑战。ORDA 通过利用查询-选项嵌入来进行检索微调，旨在模仿 oracle 查询的嵌入（即与正确答案配对的查询），从而增强支持证据的识别能力。通过在 QuALITY 基准数据集上进行的实验表明，我们提出的模型在性能和准确性方面均超过了现有基准。 

---
# Understanding Long Videos via LLM-Powered Entity Relation Graphs 

**Title (ZH)**: 通过基于LLM的实体关系图理解长视频 

**Authors**: Meng Chu, Yicong Li, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2501.15953)  

**Abstract**: The analysis of extended video content poses unique challenges in artificial intelligence, particularly when dealing with the complexity of tracking and understanding visual elements across time. Current methodologies that process video frames sequentially struggle to maintain coherent tracking of objects, especially when these objects temporarily vanish and later reappear in the footage. A critical limitation of these approaches is their inability to effectively identify crucial moments in the video, largely due to their limited grasp of temporal relationships. To overcome these obstacles, we present GraphVideoAgent, a cutting-edge system that leverages the power of graph-based object tracking in conjunction with large language model capabilities. At its core, our framework employs a dynamic graph structure that maps and monitors the evolving relationships between visual entities throughout the video sequence. This innovative approach enables more nuanced understanding of how objects interact and transform over time, facilitating improved frame selection through comprehensive contextual awareness. Our approach demonstrates remarkable effectiveness when tested against industry benchmarks. In evaluations on the EgoSchema dataset, GraphVideoAgent achieved a 2.2 improvement over existing methods while requiring analysis of only 8.2 frames on average. Similarly, testing on the NExT-QA benchmark yielded a 2.0 performance increase with an average frame requirement of 8.1. These results underscore the efficiency of our graph-guided methodology in enhancing both accuracy and computational performance in long-form video understanding tasks. 

**Abstract (ZH)**: 将上述论文内容或标题翻译成中文，同时符合学术规范如下：

分析扩展视频内容在人工智能领域中提出了独特的挑战，特别是在处理时间上视觉元素的复杂追踪和理解方面。当前逐帧处理视频的方法在保持对象连贯追踪方面存在困难，尤其是当这些对象暂时消失并在后续重新出现在视频中时。这些方法的主要局限性在于它们在识别视频中的关键时刻能力有限，主要是因为它们对时间关系的把握有限。为克服这些挑战，我们提出了一种名为GraphVideoAgent的先进系统，该系统结合了基于图的对象追踪能力和大型语言模型的能力。该框架的核心在于利用动态图结构，该结构在整个视频序列中映射和监控视觉实体之间的动态关系。这一创新方法能够更深入地理解对象如何随时间相互作用和变化，从而通过全面的上下文感知来优化帧的选择。在行业基准测试中，我们的方法表现出显著的效果。根据EgoSchema数据集的评估，GraphVideoAgent在效果上比现有方法提高了2.2%，并且平均只需要分析8.2帧。在NExT-QA基准测试中，GraphVideoAgent同样实现了2.0%的性能提升，平均帧需求仅为8.1帧。这些结果突显了我们在长视频理解任务中通过基于图的方法提高准确性和计算性能方面的高效性。 

---
# Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System 

**Title (ZH)**: 长期兴趣计时器：流式推荐系统中的细粒度时间感知 

**Authors**: Yongchun Zhu, Guanyu Jiang, Jingwu Chen, Feng Zhang, Xiao Yang, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15817)  

**Abstract**: User interests manifest a dynamic pattern within the course of a day, e.g., a user usually favors soft music at 8 a.m. but may turn to ambient music at 10 p.m. To model dynamic interests in a day, hour embedding is widely used in traditional daily-trained industrial recommendation systems. However, its discreteness can cause periodical online patterns and instability in recent streaming recommendation systems. Recently, Interest Clock has achieved remarkable performance in streaming recommendation systems. Nevertheless, it models users' dynamic interests in a coarse-grained manner, merely encoding users' discrete interests of 24 hours from short-term behaviors. In this paper, we propose a fine-grained method for perceiving time information for streaming recommendation systems, named Long-term Interest Clock (LIC). The key idea of LIC is adaptively calculating current user interests by taking into consideration the relevance of long-term behaviors around current time (e.g., 8 a.m.) given a candidate item. LIC consists of two modules: (1) Clock-GSU retrieves a sub-sequence by searching through long-term behaviors, using query information from a candidate item and current time, (2) Clock-ESU employs a time-gap-aware attention mechanism to aggregate sub-sequence with the candidate item. With Clock-GSU and Clock-ESU, LIC is capable of capturing users' dynamic fine-grained interests from long-term behaviors. We conduct online A/B tests, obtaining +0.122% improvements on user active days. Besides, the extended offline experiments show improvements as well. Long-term Interest Clock has been integrated into Douyin Music App's recommendation system. 

**Abstract (ZH)**: 用户的兴趣在一天中表现出动态模式，例如，用户通常在早上8点偏好轻柔音乐，但在晚上10点可能会转向环境音乐。为了模型这种日间的动态兴趣，传统的一日训练工业推荐系统中广泛使用了小时嵌入。然而，其离散性会导致近期流式推荐系统中的周期性在线模式和不稳定性。最近，Interest Clock在流式推荐系统中取得了显著的性能。尽管如此，它以粗粒度的方式建模用户的动态兴趣，仅通过短期行为编码了24小时的离散兴趣。在本文中，我们提出了一种新的细粒度方法，即细粒度长时兴趣钟（Long-term Interest Clock, LIC），用于感知流式推荐系统中的时间信息。LIC的核心思想是在给定候选项目的前提下，结合当前时间周围的长期行为相关性，动态计算当前用户的兴趣。LIC包含两个模块：(1) Clock-GSU 通过查询候选项目信息和当前时间，检索一个子序列；(2) Clock-ESU 使用时间间隔感知的注意力机制将候选项目与子序列聚合。借助Clock-GSU和Clock-ESU，LIC能够从长期行为中捕捉用户的动态细粒度兴趣。我们在在线A/B测试中获得了用户活跃天数提高0.122%的结果。此外，扩展的离线实验也显示了改进。细粒度长时兴趣钟已被集成到抖音音乐应用的推荐系统中。 

---
# AdaF^2M^2: Comprehensive Learning and Responsive Leveraging Features in Recommendation System 

**Title (ZH)**: AdaF^2M^2：综合学习与响应性利用特征在推荐系统中的应用 

**Authors**: Yongchun Zhu, Jingwu Chen, Ling Chen, Yitan Li, Feng Zhang, Xiao Yang, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15816)  

**Abstract**: Feature modeling, which involves feature representation learning and leveraging, plays an essential role in industrial recommendation systems. However, the data distribution in real-world applications usually follows a highly skewed long-tail pattern due to the popularity bias, which easily leads to over-reliance on ID-based features, such as user/item IDs and ID sequences of interactions. Such over-reliance makes it hard for models to learn features comprehensively, especially for those non-ID meta features, e.g., user/item characteristics. Further, it limits the feature leveraging ability in models, getting less generalized and more susceptible to data noise. Previous studies on feature modeling focus on feature extraction and interaction, hardly noticing the problems brought about by the long-tail data distribution. To achieve better feature representation learning and leveraging on real-world data, we propose a model-agnostic framework AdaF^2M^2, short for Adaptive Feature Modeling with Feature Mask. The feature-mask mechanism helps comprehensive feature learning via multi-forward training with augmented samples, while the adapter applies adaptive weights on features responsive to different user/item states. By arming base models with AdaF^2M^2, we conduct online A/B tests on multiple recommendation scenarios, obtaining +1.37% and +1.89% cumulative improvements on user active days and app duration respectively. Besides, the extended offline experiments on different models show improvements as well. AdaF$^2$M$^2$ has been widely deployed on both retrieval and ranking tasks in multiple applications of Douyin Group, indicating its superior effectiveness and universality. 

**Abstract (ZH)**: 特征建模涉及特征表示学习和利用，在工业推荐系统中发挥着重要作用。然而，在现实世界应用中，由于流行度偏差，数据分布通常呈现出高度偏斜的长尾模式。这种偏斜容易导致模型过分依赖基于ID的特征，如用户/项目ID及交互序列。这种过分依赖使得模型难以全面学习特征，特别是一些非ID元特征，例如用户/项目特性。进一步，这也限制了模型在特征利用方面的能力，使其更不易泛化，并对数据噪声更为敏感。以往关于特征建模的研究主要集中在特征提取和交互上，很少注意到长尾数据分布带来的问题。为了在实际数据上实现更好的特征表示学习和利用，我们提出了一种模型通用框架AdaF^2M^2（Adaptive Feature Modeling with Feature Mask）。特征掩码机制通过增强样本的多前向训练来促进全面特征学习，而适配器则在对不同用户/项目状态做出响应时对特征施加自适应权重。通过为基模型配备AdaF^2M^2，我们在多个推荐场景中进行了在线A/B测试，分别在用户活跃天数和应用持续时间上取得了+1.37%和+1.89%的累积改进。此外，不同模型的扩展离线实验也显示出改进效果。AdaF^2M^2已在抖音集团多个应用中的检索和排序任务中广泛部署，表明其优越的有效性和通用性。 

---
# Unveiling the Potential of Multimodal Retrieval Augmented Generation with Planning 

**Title (ZH)**: 探索规划增强的多模态检索生成的潜力 

**Authors**: Xiaohan Yu, Zhihan Yang, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15470)  

**Abstract**: Multimodal Retrieval Augmented Generation (MRAG) systems, while promising for enhancing Multimodal Large Language Models (MLLMs), often rely on rigid, single-step retrieval methods. This limitation hinders their ability to effectively address real-world scenarios that demand adaptive information acquisition and query refinement. To overcome this, we introduce the novel task of Multimodal Retrieval Augmented Generation Planning (MRAG Planning), focusing on optimizing MLLM performance while minimizing computational overhead. We present CogPlanner, a versatile framework inspired by human cognitive processes. CogPlanner iteratively refines queries and selects retrieval strategies, enabling both parallel and sequential modeling approaches. To rigorously evaluate MRAG Planning, we introduce CogBench, a new benchmark specifically designed for this task. CogBench facilitates the integration of lightweight CogPlanner with resource-efficient MLLMs. Our experimental findings demonstrate that CogPlanner surpasses existing MRAG baselines, achieving significant improvements in both accuracy and efficiency with minimal computational overhead. 

**Abstract (ZH)**: 多模态检索增强生成（MRAG）系统虽然在增强多模态大型语言模型（MLLMs）方面潜力巨大，但往往依赖于僵化的单一步骤检索方法。这一局限性限制了其在需要适应性信息获取和查询优化的实际场景中的能力。为解决这一问题，我们提出了一个新的任务——多模态检索增强生成规划（MRAG Planning），旨在优化MLLM性能的同时减小计算开销。我们提出了CogPlanner这一多功能框架，灵感源自人类的认知过程。CogPlanner通过迭代优化查询和选择检索策略，支持并行和序列模型方法。为了严格评估MRAG Planning，我们引入了CogBench，这是一种专门为此任务设计的新基准。CogBench 使轻量级的CogPlanner与资源高效的MLLMs的集成变得容易。我们的实验结果表明，CogPlanner超越了现有的MRAG基准，实现了在准确性和效率方面的显著提升，且计算开销最小。 

---
# An Aspect Performance-aware Hypergraph Neural Network for Review-based Recommendation 

**Title (ZH)**: 基于评论的推荐中面向方面性能的超图神经网络 

**Authors**: Junrui Liu, Tong Li, Di Wu, Zifang Tang, Yuan Fang, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15429)  

**Abstract**: Online reviews allow consumers to provide detailed feedback on various aspects of items. Existing methods utilize these aspects to model users' fine-grained preferences for specific item features through graph neural networks. We argue that the performance of items on different aspects is important for making precise recommendations, which has not been taken into account by existing approaches, due to lack of data. In this paper, we propose an aspect performance-aware hypergraph neural network (APH) for the review-based recommendation, which learns the performance of items from the conflicting sentiment polarity of user reviews. Specifically, APH comprehensively models the relationships among users, items, aspects, and sentiment polarity by systematically constructing an aspect hypergraph based on user reviews. In addition, APH aggregates aspects representing users and items by employing an aspect performance-aware hypergraph aggregation method. It aggregates the sentiment polarities from multiple users by jointly considering user preferences and the semantics of their sentiments, determining the weights of sentiment polarities to infer the performance of items on various aspects. Such performances are then used as weights to aggregate neighboring aspects. Experiments on six real-world datasets demonstrate that APH improves MSE, Precision@5, and Recall@5 by an average of 2.30%, 4.89%, and 1.60% over the best baseline. The source code and data are available at this https URL. 

**Abstract (ZH)**: 在线评价使消费者能够对商品的各个方面提供详细的反馈。现有方法通过图神经网络利用这些方面来建模用户对商品特定特征的精细偏好。然而，现有方法由于数据不足，未能充分考虑到不同方面商品表现的重要性，这在进行精确推荐时是关键因素。本文提出了一种面向评价的性能感知超图神经网络（APH），该网络通过用户评价中的矛盾情感极性来学习商品的表现。具体而言，APH 利用基于用户评价系统的系统方法构建一个超图，全面建模用户、商品、方面和情感极性的关系。此外，APH 通过一种面向性能的超图聚合方法聚合代表用户和商品的方面。APH 同时考虑用户偏好和他们情感的意义，联合确定情感极性的权重以推断商品在各个方面上的表现。然后，使用这些表现作为权重来聚合邻居方面。在六个真实数据集上的实验表明，相较于最佳基线模型，APH 将均方误差（MSE）、 Precision@5 和 Recall@5 提高了平均分别为 2.30%、4.89% 和 1.60%。源代码和数据可在以下链接获取：this https URL。 

---
# An Empirically-parametrized Spatio-Temporal Extended-SIR Model for Combined Dilution and Vaccination Mitigation for Rabies Outbreaks in Wild Jackals 

**Title (ZH)**: 基于实证参数化的时空扩展SIR模型：结合稀释和疫苗接种控制野狼狂犬病暴发的研究 

**Authors**: Teddy Lazebnik, Yehuda Samuel, Jonathan Tichon, Roi Lapid, Roni King, Tomer Nissimian, Orr Spiegel  

**Link**: [PDF](https://arxiv.org/pdf/2501.15425)  

**Abstract**: The transmission of zoonotic diseases between animals and humans poses an increasing threat. Rabies is a prominent example with various instances globally, facilitated by a surplus of meso-predators (commonly, facultative synanthropic species e.g., golden jackals [Canis aureus, hereafter jackals]) thanks to the abundance of anthropogenic resources leading to dense populations close to human establishments. To mitigate rabies outbreaks and prevent human infections, authorities target the jackal which is the main rabies vector in many regions, through the dissemination of oral vaccines in known jackals' activity centers, as well as opportunistic culling to reduce population density. Because dilution (i.e., culling) is not selective towards sick or un-vaccinated individuals, these two complementary epizootic intervention policies (EIPs) can interfere with each other. Nonetheless, there is only limited examination of the interactive effectiveness of these EIPs and their potential influence on rabies epizootic spread dynamics, highlighting the need to understand these measures and the spread of rabies in wild jackals. In this study, we introduce a novel spatio-temporal extended-SIR (susceptible-infected-recovered) model with a graph-based spatial framework for evaluating mitigation efficiency. We implement the model in a case study using a jackal population in northern Israel, and using spatial and movement data collected by Advanced Tracking and Localization of Animals in real-life Systems (ATLAS) telemetry. An agent-based simulation approach allows us to explore various biologically-realistic scenarios, and assess the impact of different EIPs configurations. Our model suggests that under biologically-realistic underlying assumptions and scenarios, the effectiveness of both EIPs is not influenced much by the jackal population size but is sensitive to their dispersal between activity centers. 

**Abstract (ZH)**: 动物与人类之间人畜共通疾病的传播构成了日益增大的威胁。狂犬病是全球各地的一个突出例子，这得益于次级食肉动物（通常是半适应性人类共存物种，例如金丝雀犬[Canis aureus，以下简称金丝雀犬]）的数量增多，这些次级食肉动物得益于人类造成的资源丰富，形成了与人类居住地邻近的密集种群。为了遏制狂犬病暴发并防止人类感染，当局通过在已知金丝雀犬活动中心散发口服疫苗，以及机会性猎杀来降低种群密度，以针对金丝雀犬作为许多地区的主要狂犬病传播媒介。然而，由于稀释作用（即猎杀）并不针对生病或未接种疫苗的个体，这两种互补的流行病干预措施（EIPs）可能会互相干扰。尽管如此，对这两种EIPs的交互有效性和对狂犬病传播动力学的潜在影响的研究仍然有限，这突显了了解这些措施和狂犬病在野生金丝雀犬中传播的重要性。在本研究中，我们引入了一种基于图的空间扩展SIR（易感-感染-恢复）模型，以评估缓解措施的效率。我们使用以色列北部的金丝雀犬种群作为案例研究，并应用Advanced Tracking and Localization of Animals in real-life Systems（ATLAS）遥测系统收集的空间和运动数据来实施此模型。通过基于代理的仿真方法，我们可以探索多种生物现实情境，并评估不同EIPs配置的影响。我们的模型表明，在生物现实的基本假设和情境下，这两种EIPs的有效性不会受到金丝雀犬种群规模的影响，而是对它们在活动中心之间的扩散敏感。 

---
# Zero-Shot Interactive Text-to-Image Retrieval via Diffusion-Augmented Representations 

**Title (ZH)**: 基于扩散增强表示的零样本交互式文本到图像检索 

**Authors**: Zijun Long, Kangheng Liang, Gerardo Aragon-Camarasa, Richard Mccreadie, Paul Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2501.15379)  

**Abstract**: Interactive Text-to-Image Retrieval (I-TIR) has emerged as a transformative user-interactive tool for applications in domains such as e-commerce and education. Yet, current methodologies predominantly depend on finetuned Multimodal Large Language Models (MLLMs), which face two critical limitations: (1) Finetuning imposes prohibitive computational overhead and long-term maintenance costs. (2) Finetuning narrows the pretrained knowledge distribution of MLLMs, reducing their adaptability to novel scenarios. These issues are exacerbated by the inherently dynamic nature of real-world I-TIR systems, where queries and image databases evolve in complexity and diversity, often deviating from static training distributions. To overcome these constraints, we propose Diffusion Augmented Retrieval (DAR), a paradigm-shifting framework that bypasses MLLM finetuning entirely. DAR synergizes Large Language Model (LLM)-guided query refinement with Diffusion Model (DM)-based visual synthesis to create contextually enriched intermediate representations. This dual-modality approach deciphers nuanced user intent more holistically, enabling precise alignment between textual queries and visually relevant images. Rigorous evaluations across four benchmarks reveal DAR's dual strengths: (1) Matches state-of-the-art finetuned I-TIR models on straightforward queries without task-specific training. (2) Scalable Generalization: Surpasses finetuned baselines by 7.61% in Hits@10 (top-10 accuracy) under multi-turn conversational complexity, demonstrating robustness to intricate, distributionally shifted interactions. By eliminating finetuning dependencies and leveraging generative-augmented representations, DAR establishes a new trajectory for efficient, adaptive, and scalable cross-modal retrieval systems. 

**Abstract (ZH)**: 交互式文本到图像检索（I-TIR）已成为电子商务和教育领域等应用中的一种变革性用户互动工具。然而，现有方法主要依赖于微调的多模态大型语言模型（MLLMs），这些方法面临两个关键限制：（1）微调会带来巨大的计算开销和长期维护成本；（2）微调会限制MLLMs的预训练知识分布，降低其对新颖场景的适应性。这些问题在实际世界的I-TIR系统中尤为突出，因为查询和图像数据库在复杂性和多样性方面不断变化，往往与静态训练分布相偏离。为克服这些限制，我们提出了一种名为扩散增强检索（DAR）的范式革新框架，完全绕过了MLLM的微调。DAR 结合了大型语言模型（LLM）引导的查询细化和基于扩散模型（DM）的视觉合成，以创建上下文增强的中间表示。这种双模态方法更全面地了解用户意图，使得文本查询与相关图像之间实现精确对齐。跨四个基准的严格评估显示，DAR 的双重优势在于：（1）对于简单的查询，DAR 能够与最先进的微调I-TIR模型媲美，无需特定任务的训练；（2）可扩展的泛化：在多轮对话复杂性下，DAR 使Hits@10（前10准确率）提高了7.61%，展示了其对复杂、分布变化交互的鲁棒性。通过消除对微调的依赖并利用生成增强的表示形式，DAR 建立了一条高效、适应性强且可扩展的跨模态检索系统的全新路径。 

---
# Generating Negative Samples for Multi-Modal Recommendation 

**Title (ZH)**: 多模态推荐中的负样本生成 

**Authors**: Yanbiao Ji, Yue Ding, Dan Luo, Chang Liu, Jing Tong, Shaokai Wi, Hongtao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15183)  

**Abstract**: Multi-modal recommender systems (MMRS) have gained significant attention due to their ability to leverage information from various modalities to enhance recommendation quality. However, existing negative sampling techniques often struggle to effectively utilize the multi-modal data, leading to suboptimal performance. In this paper, we identify two key challenges in negative sampling for MMRS: (1) producing cohesive negative samples contrasting with positive samples and (2) maintaining a balanced influence across different modalities. To address these challenges, we propose NegGen, a novel framework that utilizes multi-modal large language models (MLLMs) to generate balanced and contrastive negative samples. We design three different prompt templates to enable NegGen to analyze and manipulate item attributes across multiple modalities, and then generate negative samples that introduce better supervision signals and ensure modality balance. Furthermore, NegGen employs a causal learning module to disentangle the effect of intervened key features and irrelevant item attributes, enabling fine-grained learning of user preferences. Extensive experiments on real-world datasets demonstrate the superior performance of NegGen compared to state-of-the-art methods in both negative sampling and multi-modal recommendation. 

**Abstract (ZH)**: 多模态推荐系统（MMRS）由于能够利用多种模态的信息来提高推荐质量而引起了广泛关注。然而，现有的负样本生成技术往往难以有效利用多模态数据，导致性能不佳。本文识别了多模态推荐系统中负样本生成的两个关键挑战：（1）生成与正样本形成对比的连贯负样本，以及（2）在不同模态间维持平衡的影响。为了解决这些问题，我们提出了一种名为NegGen的新框架，该框架利用多模态大型语言模型（MLLMs）生成平衡且对比式的负样本。我们设计了三种不同的提示模板，以使NegGen能够跨多个模态分析和操纵项目属性，并进一步生成能够引入更多监督信号并确保模态平衡的负样本。此外，NegGen采用因果学习模块来分离干预关键特征和无关项目属性的作用，从而实现细粒度的用户偏好学习。在实际数据集上进行的广泛实验表明，NegGen在负样本生成和多模态推荐方面均优于现有最先进的方法。 

---
# Technology Mapping with Large Language Models 

**Title (ZH)**: 使用大型语言模型进行技术映射 

**Authors**: Minh Hieu Nguyen, Hien Thu Pham, Hiep Minh Ha, Ngoc Quang Hung Le, Jun Jo  

**Link**: [PDF](https://arxiv.org/pdf/2501.15120)  

**Abstract**: In today's fast-evolving business landscape, having insight into the technology stacks that organizations use is crucial for forging partnerships, uncovering market openings, and informing strategic choices. However, conventional technology mapping, which typically hinges on keyword searches, struggles with the sheer scale and variety of data available, often failing to capture nascent technologies. To overcome these hurdles, we present STARS (Semantic Technology and Retrieval System), a novel framework that harnesses Large Language Models (LLMs) and Sentence-BERT to pinpoint relevant technologies within unstructured content, build comprehensive company profiles, and rank each firm's technologies according to their operational importance. By integrating entity extraction with Chain-of-Thought prompting and employing semantic ranking, STARS provides a precise method for mapping corporate technology portfolios. Experimental results show that STARS markedly boosts retrieval accuracy, offering a versatile and high-performance solution for cross-industry technology mapping. 

**Abstract (ZH)**: 在当今快速演变的商业环境中，了解组织所使用的技术栈对于建立合作伙伴关系、发现市场机遇和指导战略决策至关重要。然而，传统的技术映射方法通常依赖关键词搜索，难以处理大量和多样化的数据，常常无法捕捉新兴技术。为克服这些挑战，我们提出了STARS（语义技术和检索系统）这一新型框架，该框架利用大型语言模型（LLMs）和Sentence-BERT来识别非结构化内容中的相关技术，构建全面的公司概况，并根据运营重要性对每家公司的技术进行排名。通过集成实体提取与链式推理提示，并采用语义排名，STARS提供了一种精确的技术组合映射方法。实验结果显示，STARS显著提高了检索精度，为跨行业的技术映射提供了灵活且高性能的解决方案。 

---
# ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation 

**Title (ZH)**: ABXI：任务导向跨领域序列推荐中的不变兴趣适配 

**Authors**: Qingtian Bian, Marcus Vinícius de Carvalho, Tieying Li, Jiaxing Xu, Hui Fang, Yiping Ke  

**Link**: [PDF](https://arxiv.org/pdf/2501.15118)  

**Abstract**: Cross-Domain Sequential Recommendation (CDSR) has recently gained attention for countering data sparsity by transferring knowledge across domains. A common approach merges domain-specific sequences into cross-domain sequences, serving as bridges to connect domains. One key challenge is to correctly extract the shared knowledge among these sequences and appropriately transfer it. Most existing works directly transfer unfiltered cross-domain knowledge rather than extracting domain-invariant components and adaptively integrating them into domain-specific modelings. Another challenge lies in aligning the domain-specific and cross-domain sequences. Existing methods align these sequences based on timestamps, but this approach can cause prediction mismatches when the current tokens and their targets belong to different domains. In such cases, the domain-specific knowledge carried by the current tokens may degrade performance. To address these challenges, we propose the A-B-Cross-to-Invariant Learning Recommender (ABXI). Specifically, leveraging LoRA's effectiveness for efficient adaptation, ABXI incorporates two types of LoRAs to facilitate knowledge adaptation. First, all sequences are processed through a shared encoder that employs a domain LoRA for each sequence, thereby preserving unique domain characteristics. Next, we introduce an invariant projector that extracts domain-invariant interests from cross-domain representations, utilizing an invariant LoRA to adapt these interests into modeling each specific domain. Besides, to avoid prediction mismatches, all domain-specific sequences are aligned to match the domains of the cross-domain ground truths. Experimental results on three datasets demonstrate that our approach outperforms other CDSR counterparts by a large margin. The codes are available in \url{this https URL}. 

**Abstract (ZH)**: 跨域序列推荐（Cross-Domain Sequential Recommendation, CDSR）近年来因其在应对数据稀疏性方面可以通过领域间的知识转移而引起了广泛关注。一种常见的方法是将领域特定的序列合并为跨领域序列，以充当连接不同领域的桥梁。一个关键挑战是如何正确地提取这些序列中的共享知识，并适当地进行转移。大多数现有工作直接转移未经过滤的跨领域知识，而没有提取出不变的领域特征组件并将其适当地整合进领域特定的建模中。另一个挑战在于对齐领域特定序列和跨领域序列。现有方法通常是基于时间戳对这些序列进行对齐，但这种方法在当前令牌及其目标属于不同领域时会导致预测不匹配。在这种情况下，当前令牌携带的领域特定知识可能会降低性能。为了应对这些挑战，我们提出了“A-to-B-不变学习推荐系统”（ABXI）。

具体而言，通过利用LoRA在高效适应性方面的有效性，ABXI结合了两种类型的LoRA来促进知识适配。首先，所有序列都通过一个共享编码器进行处理，每个序列都使用一个领域特定的LoRA，从而保持各自的领域特征。其次，我们引入了一个不变投影器，从跨领域的表示中提取出领域不变的兴趣，使用一个不变的LoRA将这些兴趣适配到每个特定领域中进行建模。除此之外，为了避免预测不匹配，所有领域特定序列都被对齐以匹配跨领域真实值的领域。

实验结果表明，我们的方法在三个数据集上的表现远超其他CDSR方法。代码可在 \url{此网址} 获取。 

---
# PatchRec: Multi-Grained Patching for Efficient LLM-based Sequential Recommendation 

**Title (ZH)**: PatchRec：高效的基于大规模语言模型的序列推荐多粒度补丁方法 

**Authors**: Jiayi Liao, Ruobing Xie, Sihang Li, Xiang Wang, Xingwu Sun, Zhanhui Kang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2501.15087)  

**Abstract**: Large Language Models for sequential recommendation (LLM4SR), which transform user-item interactions into language modeling, have shown promising results. However, due to the limitations of context window size and the computational costs associated with Large Language Models (LLMs), current approaches primarily truncate user history by only considering the textual information of items from the most recent interactions in the input prompt. This truncation fails to fully capture the long-term behavioral patterns of users. To address this, we propose a multi-grained patching framework -- PatchRec. It compresses the textual tokens of an item title into a compact item patch, and further compresses multiple item patches into a denser session patch, with earlier interactions being compressed to a greater degree. The framework consists of two stages: (1) Patch Pre-training, which familiarizes LLMs with item-level compression patterns, and (2) Patch Fine-tuning, which teaches LLMs to model sequences at multiple granularities. Through this simple yet effective approach, empirical results demonstrate that PatchRec outperforms existing methods, achieving significant performance gains with fewer tokens fed to the LLM. Specifically, PatchRec shows up to a 32% improvement in HR@20 on the Goodreads dataset over uncompressed baseline, while using only 7% of the tokens. This multi-grained sequence modeling paradigm, with an adjustable compression ratio, enables LLMs to be efficiently deployed in real-world recommendation systems that handle extremely long user behavior sequences. 

**Abstract (ZH)**: 大型语言模型在序列推荐中的应用（LLM4SR），这些模型将用户-项交互转化为语言模型，显示出了令人鼓舞的结果。然而，由于上下文窗口大小的限制以及大型语言模型（LLMs）相关的计算成本，当前的方法主要通过仅在输入提示中考虑最近交互的项的文本信息来进行用户历史截断。这种截断方法未能充分捕捉用户的长期行为模式。为了解决这个问题，我们提出了一种多粒度补丁框架——PatchRec。该框架将项标题的文本标记压缩为紧凑的项补丁，并进一步将多个项补丁压缩为更密集的会话补丁，更早的交互被压缩得更多。该框架包括两个阶段：（1）补丁预训练，使LLMs熟悉项级别压缩模式，以及（2）补丁微调，使LLMs能够以多种粒度建模序列。通过这种简单而有效的方法，实验结果表明，PatchRec优于现有方法，在Goodreads数据集上，使用仅7%的令牌，取得了显著性能提升，特别是在HR@20方面，与未压缩的基线相比，提高了32%。这种可调节压缩比的多粒度序列建模范式，使LLMs能够高效地部署在处理极其长的用户行为序列的实际推荐系统中。 

---
# CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs 

**Title (ZH)**: CG-RAG：引用图检索增强的LLM研究问题回答 

**Authors**: Yuntong Hu, Zhihan Lei, Zhongjie Dai, Allen Zhang, Abhinav Angirekula, Zheng Zhang, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15067)  

**Abstract**: Research question answering requires accurate retrieval and contextual understanding of scientific literature. However, current Retrieval-Augmented Generation (RAG) methods often struggle to balance complex document relationships with precise information retrieval. In this paper, we introduce Contextualized Graph Retrieval-Augmented Generation (CG-RAG), a novel framework that integrates sparse and dense retrieval signals within graph structures to enhance retrieval efficiency and subsequently improve generation quality for research question answering. First, we propose a contextual graph representation for citation graphs, effectively capturing both explicit and implicit connections within and across documents. Next, we introduce Lexical-Semantic Graph Retrieval (LeSeGR), which seamlessly integrates sparse and dense retrieval signals with graph encoding. It bridges the gap between lexical precision and semantic understanding in citation graph retrieval, demonstrating generalizability to existing graph retrieval and hybrid retrieval methods. Finally, we present a context-aware generation strategy that utilizes the retrieved graph-structured information to generate precise and contextually enriched responses using large language models (LLMs). Extensive experiments on research question answering benchmarks across multiple domains demonstrate that our CG-RAG framework significantly outperforms RAG methods combined with various state-of-the-art retrieval approaches, delivering superior retrieval accuracy and generation quality. 

**Abstract (ZH)**: 以下是根据学术规范翻译的内容：

研究问题回答需要精确检索和理解科学文献的上下文。然而，当前的检索增强生成（RAG）方法往往难以平衡复杂文档关系与精确信息检索之间的关系。本文介绍了一种名为情境化图检索增强生成（CG-RAG）的新框架，该框架通过在图结构中整合稀疏和密集的检索信号来提高检索效率，从而改善研究问题回答的生成质量。首先，我们提出了一种文献引用图的情境化图表示方法，有效地捕获文档内部及跨文档的显式和隐式联系。其次，我们引入了基于词性和语义图检索（LeSeGR）的方法，该方法无缝地将稀疏和密集的检索信号与图编码相结合。LeSeGR在文献引用图检索中填补了词性和语义理解之间的差距，展示了对该领域现有图检索和混合检索方法的通用性。最后，我们提出了一个上下文感知的生成策略，该策略利用检索到的图结构信息和大规模语言模型（LLM）生成精确且上下文丰富的响应。在多个领域的研究问题回答基准测试中的广泛实验表明，我们的CG-RAG框架显著优于结合了各种先进检索方法的RAG方法，提供了更高的检索准确性和生成质量。 

---
# Search results diversification in competitive search 

**Title (ZH)**: 竞争性搜索中的检索结果多样化 

**Authors**: Tommy Mordo, Itamar Reinman, Moshe Tennenholtz, Oren Kurland  

**Link**: [PDF](https://arxiv.org/pdf/2501.14922)  

**Abstract**: In Web retrieval, there are many cases of competition between authors of Web documents: their incentive is to have their documents highly ranked for queries of interest. As such, the Web is a prominent example of a competitive search setting. Past work on competitive search focused on ranking functions based solely on relevance estimation. We study ranking functions that integrate a results-diversification aspect. We show that the competitive search setting with diversity-based ranking has an equilibrium. Furthermore, we theoretically and empirically show that the phenomenon of authors mimicking content in documents highly ranked in the past, which was demonstrated in previous work, is mitigated when search results diversification is applied. 

**Abstract (ZH)**: 在网页检索中，存在着网页文档作者之间的竞争：他们希望通过优化其文档，使其在感兴趣的查询中获得较高的排名。因此，互联网是一个典型的竞争性搜索环境。过去对竞争性搜索的研究主要集中在基于相关性估计的排名函数上。我们研究了结合结果多样化方面的排名函数。我们展示了基于多样化排名的竞争性搜索设置存在均衡。此外，我们通过理论和实证研究展示了先前研究中所证明的一种现象——作者模仿过去排名较高的网页文档中的内容——当应用检索结果多样化时会得到缓解。 

---
# Multi-Modality Transformer for E-Commerce: Inferring User Purchase Intention to Bridge the Query-Product Gap 

**Title (ZH)**: 面向电子商务的多模态变压器：推断用户购买意图以缩短查询与产品之间的差距 

**Authors**: Srivatsa Mallapragada, Ying Xie, Varsha Rani Chawan, Zeyad Hailat, Yuanbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14826)  

**Abstract**: E-commerce click-stream data and product catalogs offer critical user behavior insights and product knowledge. This paper propose a multi-modal transformer termed as PINCER, that leverages the above data sources to transform initial user queries into pseudo-product representations. By tapping into these external data sources, our model can infer users' potential purchase intent from their limited queries and capture query relevant product features. We demonstrate our model's superior performance over state-of-the-art alternatives on e-commerce online retrieval in both controlled and real-world experiments. Our ablation studies confirm that the proposed transformer architecture and integrated learning strategies enable the mining of key data sources to infer purchase intent, extract product features, and enhance the transformation pipeline from queries to more accurate pseudo-product representations. 

**Abstract (ZH)**: 电子商务点击流数据和产品目录提供了关键的用户行为见解和产品知识。本文提出了一种名为PINCE的多模态变压器，该模型利用上述数据源将初始用户查询转化为伪产品表示。通过利用这些外部数据源，我们的模型可以从用户有限的查询中推断出用户的潜在购买意图，并捕获查询相关的商品特征。我们在受控和真实世界实验中展示了该模型在电子商务在线检索方面的优越性能，与最先进的替代方案相比，其表现更为出色。我们的消融研究证实，提出的变压器架构和集成学习策略能够挖掘关键数据源以推断购买意图、提取产品特征，并增强从查询到更准确伪产品表示的转换管道。 

---
# Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries 

**Title (ZH)**: 将这些声音分离出来，以及周围的所有声音成分：基于超椭球查询的音乐源分离 

**Authors**: Karn N. Watcharasupat, Alexander Lerch  

**Link**: [PDF](https://arxiv.org/pdf/2501.16171)  

**Abstract**: Music source separation is an audio-to-audio retrieval task of extracting one or more constituent components, or composites thereof, from a musical audio mixture. Each of these constituent components is often referred to as a "stem" in literature. Historically, music source separation has been dominated by a stem-based paradigm, leading to most state-of-the-art systems being either a collection of single-stem extraction models, or a tightly coupled system with a fixed, difficult-to-modify, set of supported stems. Combined with the limited data availability, advances in music source separation have thus been mostly limited to the "VDBO" set of stems: \textit{vocals}, \textit{drum}, \textit{bass}, and the catch-all \textit{others}. Recent work in music source separation has begun to challenge the fixed-stem paradigm, moving towards models able to extract any musical sound as long as this target type of sound could be specified to the model as an additional query input. We generalize this idea to a \textit{query-by-region} source separation system, specifying the target based on the query regardless of how many sound sources or which sound classes are contained within it. To do so, we propose the use of hyperellipsoidal regions as queries to allow for an intuitive yet easily parametrizable approach to specifying both the target (location) as well as its spread. Evaluation of the proposed system on the MoisesDB dataset demonstrated state-of-the-art performance of the proposed system both in terms of signal-to-noise ratios and retrieval metrics. 

**Abstract (ZH)**: 音乐源分离是一种从音乐混合音频中提取一个或多个构成成分或其组合的音频到音频检索任务。这些构成成分在文献中通常被称为“声轨”。历史上，音乐源分离主要受“声轨”为中心的范式支配，导致最先进的系统要么是一系列单一声轨提取模型的集合，要么是一个紧密耦合的系统，具有固定且难以修改的支持声轨集。结合可用数据的限制，音乐源分离的主要进展仅限于“VDBO”声轨集：即声乐（vocals）、打击乐（drum）、低音（bass）和其余类别（others）。最近的音乐源分离研究已经开始挑战固定的声轨范式，转向能够提取任何音乐声音的模型，只要目标声音类型可以作为额外查询输入提供给模型。我们将这一想法推广到基于区域的查询源分离系统，根据查询指定目标，而不考虑其中包含多少声音源或属于何种声音类别。为了实现这一点，我们提出使用超椭球体区域作为查询，以提供直观且易于参数化的指定目标（位置）及其扩散范围的方法。在MoisesDB数据集上的评估表明，所提出系统的信号与噪声比以及检索指标均达到了最先进的性能。 

---
# PISCO: Pretty Simple Compression for Retrieval-Augmented Generation 

**Title (ZH)**: PISCO：简单的检索增强生成压缩方法 

**Authors**: Maxime Louis, Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2501.16075)  

**Abstract**: Retrieval-Augmented Generation (RAG) pipelines enhance Large Language Models (LLMs) by retrieving relevant documents, but they face scalability issues due to high inference costs and limited context size. Document compression is a practical solution, but current soft compression methods suffer from accuracy losses and require extensive pretraining. In this paper, we introduce PISCO, a novel method that achieves a 16x compression rate with minimal accuracy loss (0-3%) across diverse RAG-based question-answering (QA) tasks. Unlike existing approaches, PISCO requires no pretraining or annotated data, relying solely on sequence-level knowledge distillation from document-based questions. With the ability to fine-tune a 7-10B LLM in 48 hours on a single A100 GPU, PISCO offers a highly efficient and scalable solution. We present comprehensive experiments showing that PISCO outperforms existing compression models by 8% in accuracy. 

**Abstract (ZH)**: 检索增强生成（RAG）管道通过检索相关文档来增强大型语言模型（LLMs），但由于高推理成本和有限的上下文大小，它们面临着规模上的挑战。文档压缩是一种实际的解决方案，但当前的软压缩方法会导致准确度损失，并且需要大量的预训练。在本文中，我们介绍了一种名为PISCO的新方法，该方法在各种基于RAG的问答（QA）任务中实现了16倍的压缩率，并且准确度损失极小（0-3%）。与现有方法不同，PISCO不需要预训练或标注数据，仅依赖于基于文档的问题的序列级知识蒸馏。通过在单个A100 GPU上48小时内微调一个7-10B的LLM，PISCO提供了一个高效且可扩展的解决方案。我们进行了全面的实验，结果显示PISCO在准确度方面比现有压缩模型高出8%。 

---
# Parametric Retrieval Augmented Generation 

**Title (ZH)**: 参数驱动的检索增强生成 

**Authors**: Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15915)  

**Abstract**: Retrieval-augmented generation (RAG) techniques have emerged as a promising solution to enhance the reliability of large language models (LLMs) by addressing issues like hallucinations, outdated knowledge, and domain adaptation. In particular, existing RAG methods append relevant documents retrieved from external corpus or databases to the input of LLMs to guide their generation process, which we refer to as the in-context knowledge injection method. While this approach is simple and often effective, it has inherent limitations. Firstly, increasing the context length and number of relevant documents can lead to higher computational overhead and degraded performance, especially in complex reasoning tasks. More importantly, in-context knowledge injection operates primarily at the input level, but LLMs store their internal knowledge in their parameters. This gap fundamentally limits the capacity of in-context methods. To this end, we introduce Parametric retrieval-augmented generation (Parametric RAG), a new RAG paradigm that integrates external knowledge directly into the parameters of feed-forward networks (FFN) of an LLM through document parameterization. This approach not only saves online computational costs by eliminating the need to inject multiple documents into the LLMs' input context, but also deepens the integration of external knowledge into the parametric knowledge space of the LLM. Experimental results demonstrate that Parametric RAG substantially enhances both the effectiveness and efficiency of knowledge augmentation in LLMs. Also, it can be combined with in-context RAG methods to achieve even better performance.
We have open-sourced all the code, data, and models in the following anonymized GitHub link: this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）技术已成为提高大型语言模型（LLMs）可靠性的有前途的解决方案，通过解决幻觉、过时的知识和领域适应性等问题。特别是，现有的RAG方法在LLMs的输入中附加从外部语料库或数据库检索的相关文档，以引导其生成过程，我们称之为上下文内知识注入方法。虽然这种方法简单且往往有效，但它具有固有局限性。首先，增加上下文长度和相关文档的数量会导致更高的计算开销和性能下降，尤其是在复杂的推理任务中。更重要的是，上下文内知识注入主要在输入级别进行，而LLMs在其参数中存储内部知识。这一差距从根本上限制了上下文内方法的容量。为了解决这些问题，我们提出了一种新的RAG范式——参数化检索增强生成（Parametric RAG），该范式直接将外部知识融入LLMs的前向网络（FFN）参数中，通过文档参数化实现。这种方法不仅通过消除向LLMs输入上下文注入多个文档的需要来节省在线计算成本，而且加深了外部知识与LLMs参数化知识空间的集成。实验证明，Parametric RAG在LLMs中显著增强了知识增强的有效性和效率。此外，它可以与上下文内RAG方法结合使用，以实现更好的性能。

此外，我们已经在以下匿名GitHub链接中开源了所有代码、数据和模型：this https URL 

---
# LemmaHead: RAG Assisted Proof Generation Using Large Language Models 

**Title (ZH)**: LemmaHead：使用大型语言模型的RAG辅助证明生成 

**Authors**: Tianbo Yang, Mingqi Yang, Hongyi Zhao, Tianshuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15797)  

**Abstract**: Developing the logic necessary to solve mathematical problems or write mathematical proofs is one of the more difficult objectives for large language models (LLMS). Currently, the most popular methods in literature consists of fine-tuning the model on written mathematical content such as academic publications and textbooks, so that the model can learn to emulate the style of mathematical writing. In this project, we explore the effectiveness of using retrieval augmented generation (RAG) to address gaps in the mathematical reasoning of LLMs. We develop LemmaHead, a RAG knowledge base that supplements queries to the model with relevant mathematical context, with particular focus on context from published textbooks. To measure our model's performance in mathematical reasoning, our testing paradigm focuses on the task of automated theorem proving via generating proofs to a given mathematical claim in the Lean formal language. 

**Abstract (ZH)**: 构建解决数学问题或编写数学证明所需的逻辑是大型语言模型（LLMs）面临的一项更为困难的目标。目前文献中最流行的方法之一是通过微调模型来学习数学内容的写作风格，例如学术出版物和教科书，从而使模型能够模仿数学写作的风格。在本项目中，我们探讨了使用检索增强生成（RAG）来弥补LLMs在数学推理方面的不足。我们开发了LemmaHead，这是一种RAG知识库，能够在查询中补充相关数学背景信息，特别强调来自已出版教科书的内容。为了衡量模型在数学推理方面的性能，我们的测试框架集中在自动化定理证明任务上，即使用Lean形式语言生成给定数学命题的证明。 

---
# SCP-116K: A High-Quality Problem-Solution Dataset and a Generalized Pipeline for Automated Extraction in the Higher Education Science Domain 

**Title (ZH)**: SCP-116K：高质量问题-解决方案数据集及高等教育科学领域自动化提取的通用管道 

**Authors**: Dakuan Lu, Xiaoyu Tan, Rui Xu, Tianchu Yao, Chao Qu, Wei Chu, Yinghui Xu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2501.15587)  

**Abstract**: Recent breakthroughs in large language models (LLMs) exemplified by the impressive mathematical and scientific reasoning capabilities of the o1 model have spotlighted the critical importance of high-quality training data in advancing LLM performance across STEM disciplines. While the mathematics community has benefited from a growing body of curated datasets, the scientific domain at the higher education level has long suffered from a scarcity of comparable resources. To address this gap, we present SCP-116K, a new large-scale dataset of 116,756 high-quality problem-solution pairs, automatically extracted from heterogeneous sources using a streamlined and highly generalizable pipeline. Our approach involves stringent filtering to ensure the scientific rigor and educational level of the extracted materials, while maintaining adaptability for future expansions or domain transfers. By openly releasing both the dataset and the extraction pipeline, we seek to foster research on scientific reasoning, enable comprehensive performance evaluations of new LLMs, and lower the barrier to replicating the successes of advanced models like o1 in the broader science community. We believe SCP-116K will serve as a critical resource, catalyzing progress in high-level scientific reasoning tasks and promoting further innovations in LLM development. The dataset and code are publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新突破，如o1模型在数学和科学推理方面表现出的强大能力，突显了高质量训练数据对推动LLM在STEM学科中的性能提升的重要性。尽管数学界受益于日益增长的经过整理的数据集，高等教育领域的科学领域长期缺乏类似资源。为弥补这一差距，我们提出了SCP-116K，这是一个包含116,756个高质量问题-解决方案配对的新大规模数据集合，这些配对是通过简化且高度通用的管道从异构来源自动提取的。我们的方法涉及严格的筛选步骤，以确保提取材料的科学严谨性及其教育水平，同时保持对未来扩展或领域转移的适应性。通过公开发布该数据集和提取管道，我们期望促进科学推理方面的研究，使新型LLM的全面性能评估成为可能，并降低在更广泛的科学界重复高阶模型如o1的成功门槛。我们相信SCP-116K将成为一个关键资源，推动高级科学推理任务的进步，并促进LLM开发中的进一步创新。该数据集和代码在此处公开获取：[请在此处插入URL]。 

---
# Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning 

**Title (ZH)**: 通过多智能体强化学习提高检索增强生成 

**Authors**: Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming Yang, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15228)  

**Abstract**: Retrieval-augmented generation (RAG) is extensively utilized to incorporate external, current knowledge into large language models, thereby minimizing hallucinations. A standard RAG pipeline may comprise several components, such as query rewriting, document retrieval, document filtering, and answer generation. However, these components are typically optimized separately through supervised fine-tuning, which can lead to misalignments between the objectives of individual modules and the overarching aim of generating accurate answers in question-answering (QA) tasks. Although recent efforts have explored reinforcement learning (RL) to optimize specific RAG components, these approaches often focus on overly simplistic pipelines with only two components or do not adequately address the complex interdependencies and collaborative interactions among the modules. To overcome these challenges, we propose treating the RAG pipeline as a multi-agent cooperative task, with each component regarded as an RL agent. Specifically, we present MMOA-RAG, a Multi-Module joint Optimization Algorithm for RAG, which employs multi-agent reinforcement learning to harmonize all agents' goals towards a unified reward, such as the F1 score of the final answer. Experiments conducted on various QA datasets demonstrate that MMOA-RAG improves the overall pipeline performance and outperforms existing baselines. Furthermore, comprehensive ablation studies validate the contributions of individual components and the adaptability of MMOA-RAG across different RAG components and datasets. The code of MMOA-RAG is on this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）广泛应用于将外部的最新知识融入大型语言模型中，从而减少模型的虚构性。一个标准的RAG流水线可能包括多个组件，如查询重写、文档检索、文档过滤和答案生成。然而，这些组件通常通过有监督的微调分别进行优化，这可能导致各个模块的优化目标与整体生成准确答案的目标产生不一致。虽然最近的研究尝试使用强化学习（RL）来优化特定的RAG组件，但这些方法往往集中在具有两个组件的过于简化的流水线上，或者未能充分解决模块之间的复杂相互依赖和协作交互问题。为了克服这些挑战，我们提出将RAG流水线视为一个多智能体协作任务，将每个组件视为一个RL代理。具体来说，我们提出了MMOA-RAG，即一种多模块联合优化算法，采用多智能体强化学习来使所有代理的目标统一到一个共同的奖励，如最终答案的F1分数。在各种问答数据集上的实验表明，MMOA-RAG提高了整体流水线性能并优于现有基线。此外，全面的消融研究验证了各组件的贡献以及MMOA-RAG在不同RAG组件和数据集上的适应性。MMOA-RAG的代码可以通过以下链接访问：https://github.com/your-repository-name/MMOA-RAG。 

---
# MDEval: Evaluating and Enhancing Markdown Awareness in Large Language Models 

**Title (ZH)**: MDEval: 评估和提升大规模语言模型中Markdown意识 

**Authors**: Zhongpu Chen, Yinfeng Liu, Long Shi, Zhi-Jie Wang, Xingyan Chen, Yu Zhao, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2501.15000)  

**Abstract**: Large language models (LLMs) are expected to offer structured Markdown responses for the sake of readability in web chatbots (e.g., ChatGPT). Although there are a myriad of metrics to evaluate LLMs, they fail to evaluate the readability from the view of output content structure. To this end, we focus on an overlooked yet important metric -- Markdown Awareness, which directly impacts the readability and structure of the content generated by these language models. In this paper, we introduce MDEval, a comprehensive benchmark to assess Markdown Awareness for LLMs, by constructing a dataset with 20K instances covering 10 subjects in English and Chinese. Unlike traditional model-based evaluations, MDEval provides excellent interpretability by combining model-based generation tasks and statistical methods. Our results demonstrate that MDEval achieves a Spearman correlation of 0.791 and an accuracy of 84.1% with human, outperforming existing methods by a large margin. Extensive experimental results also show that through fine-tuning over our proposed dataset, less performant open-source models are able to achieve comparable performance to GPT-4o in terms of Markdown Awareness. To ensure reproducibility and transparency, MDEval is open sourced at this https URL. 

**Abstract (ZH)**: 以下是将该论文内容翻译成中文，符合学术规范的版本：

大型语言模型（LLMs）预计会提供结构化的 Markdown 回应，以提高网页聊天机器人（例如 ChatGPT）的可读性。尽管有许多指标可以评估 LLM，但它们在评估输出内容结构的可读性方面存在局限性。因此，我们关注一个被忽视但很重要的指标——Markdown 意识，这一指标直接影响由这些语言模型生成的内容的可读性和结构。在本文中，我们引入了 MDEval，这是一种全面的基准测试，用于评估 LLM 的 Markdown 意识。我们构建了一个包含 20,000 个实例的数据集，这些实例覆盖了英语和汉语中的 10 个主题。与传统的模型评估不同，MDEval 通过结合基于模型的生成任务和统计方法提供了出色的可解释性。我们的结果表明，MDEval 在与人类进行比较时，实现了皮尔森相关系数 0.791 和准确率 84.1%，远优于现有方法。大量的实验结果还表明，通过对我们提出的数据集进行微调，性能较为弱的开源模型能够达到与 GPT-4o 相似的 Markdown 意识水平。为确保再现性和透明度，MDEval 已在以下链接公开：[在此处插入链接]。

请注意，翻译过程中保持了原文的结构和术语，以符合学术写作的规范。同时，确保关键指标和方法术语（如 Markdown 意识、MDEval 等）没有被错误翻译或误解。 

---
# ExPerT: Effective and Explainable Evaluation of Personalized Long-Form Text Generation 

**Title (ZH)**: ExPerT: 有效且可解释的个性化长文本生成评估方法 

**Authors**: Alireza Salemi, Julian Killingback, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2501.14956)  

**Abstract**: Evaluating personalized text generated by large language models (LLMs) is challenging, as only the LLM user, i.e., prompt author, can reliably assess the output, but re-engaging the same individuals across studies is infeasible. This paper addresses the challenge of evaluating personalized text generation by introducing ExPerT, an explainable reference-based evaluation framework. ExPerT leverages an LLM to extract atomic aspects and their evidence from the generated and reference texts, match the aspects, and evaluate their alignment based on content and writing style -- two key attributes in personalized text generation. Additionally, ExPerT generates detailed, fine-grained explanations for every step of the evaluation process, enhancing transparency and interpretability. Our experiments demonstrate that ExPerT achieves a 7.2% relative improvement in alignment with human judgments compared to the state-of-the-art text generation evaluation methods. Furthermore, human evaluators rated the usability of ExPerT's explanations at 4.7 out of 5, highlighting its effectiveness in making evaluation decisions more interpretable. 

**Abstract (ZH)**: 评估由大型语言模型（LLMs）生成的个性化文本具有挑战性，因为只有模型的使用者，即提示作者，才能可靠地评估输出，但在不同研究中重新获得同一评估人员是不现实的。本文通过引入ExPerT（可解释的基于参考的评估框架）来应对个性化文本生成的评估挑战。ExPerT利用LLM从生成文本和参考文本中提取原子方面及其证据，匹配这些方面，并基于内容和写作风格对它们的对齐程度进行评估——这是个性化文本生成的两个关键属性。此外，ExPerT为评估过程中每一阶段生成了详细的细粒度解释，增强了透明度和可解释性。我们的实验表明，ExPerT在与最先进的文本生成评估方法进行比较时，在与人类判断的对齐度上取得了7.2%的相对改进。另外，人类评价者对ExPerT解释的可用性打分为4.7分（满分5分），突显了该方法在使评价决策更具可解释性方面的有效性。 

---
# Leveraging Social Media Data and Artificial Intelligence for Improving Earthquake Response Efforts 

**Title (ZH)**: 利用社交媒体数据和人工智能优化地震响应努力 

**Authors**: Kalin Kopanov, Velizar Varbanov, Tatiana Atanasova  

**Link**: [PDF](https://arxiv.org/pdf/2501.14767)  

**Abstract**: The integration of social media and artificial intelligence (AI) into disaster management, particularly for earthquake response, represents a profound evolution in emergency management practices. In the digital age, real-time information sharing has reached unprecedented levels, with social media platforms emerging as crucial communication channels during crises. This shift has transformed traditional, centralized emergency services into more decentralized, participatory models of disaster situational awareness. Our study includes an experimental analysis of 8,900 social media interactions, including 2,920 posts and 5,980 replies on X (formerly Twitter), following a magnitude 5.1 earthquake in Oklahoma on February 2, 2024. The analysis covers data from the immediate aftermath and extends over the following seven days, illustrating the critical role of digital platforms in modern disaster response. The results demonstrate that social media platforms can be effectively used as real-time situational awareness tools, delivering critical information to society and authorities during emergencies. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

将社交媒体与人工智能（AI）集成到灾害管理中，特别是在地震响应中的应用，代表了应急管理实践的重大演变。在数字时代，信息分享达到了前所未有的水平，社交媒体平台已成为危机期间重要的沟通渠道。这一转变将传统的集中式应急服务转变为更加分散和参与式的灾害情境认知模式。本研究包括一项针对2024年2月2日美国俄克拉何马州发生5.1级地震后社交媒体互动的实验分析，涵盖了8900次社交媒体互动，其中包含2920个帖子和5980条回复，全部来自X（原Twitter）平台。分析内容覆盖了地震发生后的立即反应阶段以及随后的七天，展示了数字平台在现代灾害响应中的关键作用。研究结果表明，社交媒体平台可以作为实时情境感知工具有效使用，为社会和当局在紧急情况下提供关键信息。 

---
