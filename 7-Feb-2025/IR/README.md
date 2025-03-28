# Counterfactual Query Rewriting to Use Historical Relevance Feedback 

**Title (ZH)**: 将反事实查询重写以利用历史相关反馈 

**Authors**: Jüri Keller, Maik Fröbe, Gijs Hendriksen, Daria Alexander, Martin Potthast, Matthias Hagen, Philipp Schaer  

**Link**: [PDF](https://arxiv.org/pdf/2502.03891)  

**Abstract**: When a retrieval system receives a query it has encountered before, previous relevance feedback, such as clicks or explicit judgments can help to improve retrieval results. However, the content of a previously relevant document may have changed, or the document might not be available anymore. Despite this evolved corpus, we counterfactually use these previously relevant documents as relevance signals. In this paper we proposed approaches to rewrite user queries and compare them against a system that directly uses the previous qrels for the ranking. We expand queries with terms extracted from the previously relevant documents or derive so-called keyqueries that rank the previously relevant documents to the top of the current corpus. Our evaluation in the CLEF LongEval scenario shows that rewriting queries with historical relevance feedback improves the retrieval effectiveness and even outperforms computationally expensive transformer-based approaches. 

**Abstract (ZH)**: 当检索系统接收到之前遇到过的查询时，之前的相关性反馈（如点击或显式判断）可以帮助提高检索结果。然而，之前相关文档的内容可能已经发生变化，或者该文档可能已经不可用。尽管如此，我们出于反事实的角度将其之前的相关文档作为相关性信号加以使用。本文提出了通过重写用户查询并将其与直接使用之前的相关性反馈（qrels）进行排名的系统进行对比的方法。我们通过将之前相关文档中的术语加入查询中，或者推导出所谓的关键查询（ranking the previously relevant documents to the top of the current corpus），来扩展查询。在CLEF LongEval场景下的评估表明，使用历史相关性反馈重写查询可以提高检索效果，并且甚至超过了计算复杂度较高的基于转换器的方法。 

---
# Boosting Knowledge Graph-based Recommendations through Confidence-Aware Augmentation with Large Language Models 

**Title (ZH)**: 通过大型语言模型aware置信度增强的知识图谱推荐提升 

**Authors**: Rui Cai, Chao Wang, Qianyi Cai, Dazhong Shen, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.03715)  

**Abstract**: Knowledge Graph-based recommendations have gained significant attention due to their ability to leverage rich semantic relationships. However, constructing and maintaining Knowledge Graphs (KGs) is resource-intensive, and the accuracy of KGs can suffer from noisy, outdated, or irrelevant triplets. Recent advancements in Large Language Models (LLMs) offer a promising way to improve the quality and relevance of KGs for recommendation tasks. Despite this, integrating LLMs into KG-based systems presents challenges, such as efficiently augmenting KGs, addressing hallucinations, and developing effective joint learning methods. In this paper, we propose the Confidence-aware KG-based Recommendation Framework with LLM Augmentation (CKG-LLMA), a novel framework that combines KGs and LLMs for recommendation task. The framework includes: (1) an LLM-based subgraph augmenter for enriching KGs with high-quality information, (2) a confidence-aware message propagation mechanism to filter noisy triplets, and (3) a dual-view contrastive learning method to integrate user-item interactions and KG data. Additionally, we employ a confidence-aware explanation generation process to guide LLMs in producing realistic explanations for recommendations. Finally, extensive experiments demonstrate the effectiveness of CKG-LLMA across multiple public datasets. 

**Abstract (ZH)**: 基于知识图谱的推荐方法由于能够利用丰富的语义关系而引起了广泛关注。然而，构建和维护知识图谱（KGs）耗费资源，且KGs的准确性可能会受到嘈杂、过时或无关的三元组的影响。近年来，大型语言模型（LLMs）的进步为提高KGs的质量和相关性提供了有前景的方式。尽管如此，将LLMs整合到基于KG的系统中仍面临一些挑战，例如高效地扩充KGs、解决幻觉问题以及开发有效的联合学习方法。在本文中，我们提出了基于知识图谱的认知增强推荐框架与LLM增强（CKG-LLMA），这是一种结合KGs和LLMs的创新框架，用于推荐任务。该框架包括：（1）基于LLM的子图扩充器，用于丰富KGs中的高质量信息；（2）认知增强的消息传播机制，用于过滤噪音三元组；（3）双视角对比学习方法，用于整合用户-物品交互和KG数据。此外，我们还采用认知增强的解释生成过程来引导LLMs生成与推荐相关的现实解释。最后，广泛的实验表明，CKG-LLMA在多个公开数据集上的有效性。 

---
# Contrastive Learning for Cold Start Recommendation with Adaptive Feature Fusion 

**Title (ZH)**: 冷启动推荐中的自适应特征融合对比学习 

**Authors**: Jiacheng Hu, Tai An, Zidong Yu, Junliang Du, Yuanshuai Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.03664)  

**Abstract**: This paper proposes a cold start recommendation model that integrates contrastive learning, aiming to solve the problem of performance degradation of recommendation systems in cold start scenarios due to the scarcity of user and item interaction data. The model dynamically adjusts the weights of key features through an adaptive feature selection module and effectively integrates user attributes, item meta-information, and contextual features by combining a multimodal feature fusion mechanism, thereby improving recommendation performance. In addition, the model introduces a contrastive learning mechanism to enhance the robustness and generalization ability of feature representation by constructing positive and negative sample pairs. Experiments are conducted on the MovieLens-1M dataset. The results show that the proposed model significantly outperforms mainstream recommendation methods such as Matrix Factorization, LightGBM, DeepFM, and AutoRec in terms of HR, NDCG, MRR, and Recall, especially in cold start scenarios. Ablation experiments further verify the key role of each module in improving model performance, and the learning rate sensitivity analysis shows that a moderate learning rate is crucial to the optimization effect of the model. This study not only provides a new solution to the cold start problem but also provides an important reference for the application of contrastive learning in recommendation systems. In the future, this model is expected to play a role in a wider range of scenarios, such as real-time recommendation and cross-domain recommendation. 

**Abstract (ZH)**: 本文提出了一种结合对比学习的冷启动推荐模型，旨在解决由于用户和项目交互数据稀缺而在冷启动场景中推荐系统的性能下降问题。该模型通过自适应特征选择模块动态调整关键特征的权重，并通过多模态特征融合机制有效整合用户属性、项目元信息和上下文特征，从而提高推荐性能。此外，该模型引入了对比学习机制，通过构建正负样本对来增强特征表示的鲁棒性和泛化能力。实验在MovieLens-1M数据集上进行。结果显示，提出的模型在HR、NDCG、MRR和召回率等指标上显著优于主流推荐方法（如矩阵分解、LightGBM、DeepFM和AutoRec），特别是在冷启动场景中表现尤为突出。消融实验进一步验证了每个模块在提高模型性能中的关键作用，学习率敏感性分析表明，适度的学习率对模型优化效果至关重要。本研究不仅为解决冷启动问题提供了新的解决方案，还为对比学习在推荐系统中的应用提供了重要的参考。未来，该模型有望在更多的场景中发挥作用，如实时推荐和跨域推荐。 

---
# Digital Gatekeeping: An Audit of Search Engine Results shows tailoring of queries on the Israel-Palestine Conflict 

**Title (ZH)**: 数字化把关：对搜索引擎结果的审查显示对巴以冲突查询的定制化 

**Authors**: Íris Damião, José M. Reis, Paulo Almeida, Nuno Santos, Joana Gonçalves-Sá  

**Link**: [PDF](https://arxiv.org/pdf/2502.04266)  

**Abstract**: Search engines, often viewed as reliable gateways to information, tailor search results using customization algorithms based on user preferences, location, and more. While this can be useful for routine queries, it raises concerns when the topics are sensitive or contentious, possibly limiting exposure to diverse viewpoints and increasing polarization.
To examine the extent of this tailoring, we focused on the Israel-Palestine conflict and developed a privacy-protecting tool to audit the behavior of three search engines: DuckDuckGo, Google and Yahoo. Our study focused on two main questions: (1) How do search results for the same query about the conflict vary among different users? and (2) Are these results influenced by the user's location and browsing history?
Our findings revealed significant customization based on location and browsing preferences, unlike previous studies that found only mild personalization for general topics. Moreover, queries related to the conflict were more customized than unrelated queries, and the results were not neutral concerning the conflict's portrayal. 

**Abstract (ZH)**: 搜索引擎通常被视作可靠的信息入口，它们通过定制算法根据用户的偏好、位置等因素调整搜索结果。虽然这在常规查询中可能很有帮助，但在敏感或争议性话题上使用这种个性化方法时，可能会限制用户接触不同观点的机会，从而加剧观点分歧。
为了研究这种个性化现象的程度，我们聚焦于以色列-巴勒斯坦冲突这一热点问题，并开发了一个保护隐私的工具，用于审计三家搜索引擎（DuckDuckGo、Google和Yahoo）的行为。我们的研究主要关注两个问题：（1）不同用户对于同一冲突查询的搜索结果显示有何不同？（2）这些结果是否受到用户地理位置和浏览历史的影响？
研究发现，搜索结果在位置和浏览偏好方面存在明显的个性化差异，这与之前对于普通话题仅发现轻微个人化的研究结果不同。此外，与冲突相关的查询比不相关的查询更加个性化，而搜索结果在呈现冲突方面并非中立。 

---
# MRAMG-Bench: A BeyondText Benchmark for Multimodal Retrieval-Augmented Multimodal Generation 

**Title (ZH)**: MRAMG-Bench：一个超越文本的多模态检索增强多模态生成基准测试 

**Authors**: Qinhan Yu, Zhiyou Xiao, Binghui Li, Zhengren Wang, Chong Chen, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04176)  

**Abstract**: Recent advancements in Retrieval-Augmented Generation (RAG) have shown remarkable performance in enhancing response accuracy and relevance by integrating external knowledge into generative models. However, existing RAG methods primarily focus on providing text-only answers, even in multimodal retrieval-augmented generation scenarios. In this work, we introduce the Multimodal Retrieval-Augmented Multimodal Generation (MRAMG) task, which aims to generate answers that combine both text and images, fully leveraging the multimodal data within a corpus. Despite the importance of this task, there is a notable absence of a comprehensive benchmark to effectively evaluate MRAMG performance. To bridge this gap, we introduce the MRAMG-Bench, a carefully curated, human-annotated dataset comprising 4,346 documents, 14,190 images, and 4,800 QA pairs, sourced from three categories: Web Data, Academic Papers, and Lifestyle. The dataset incorporates diverse difficulty levels and complex multi-image scenarios, providing a robust foundation for evaluating multimodal generation tasks. To facilitate rigorous evaluation, our MRAMG-Bench incorporates a comprehensive suite of both statistical and LLM-based metrics, enabling a thorough analysis of the performance of popular generative models in the MRAMG task. Besides, we propose an efficient multimodal answer generation framework that leverages both LLMs and MLLMs to generate multimodal responses. Our datasets are available at: this https URL. 

**Abstract (ZH)**: 最近在检索增强生成（RAG）领域的进展显著提升了生成模型通过整合外部知识以提高响应准确性和相关性的能力。然而，现有的RAG方法主要集中在提供纯文本答案，即使在多模态检索增强生成的场景中也是如此。本文中，我们引入了多模态检索增强多模态生成（MRAMG）任务，旨在生成结合文本和图像的答案，充分挖掘语料库中的多模态数据。尽管该任务的重要性不言而喻，但仍缺乏一个全面的基准来有效评估MRAMG的性能。为解决这一问题，我们引入了MRAMG-Bench数据集，这是一个精心策划、由人工标注的包含4,346份文档、14,190张图片和4,800个问答对的数据集，来源自三个类别：网络数据、学术论文和生活方式。该数据集涵盖了多种难度级别和复杂的多图像场景，为评估多模态生成任务提供了坚实的基础。为了方便严格的评估，我们的MRAMG-Bench引入了一系列全面的统计和LLM基评估指标，能够全面分析流行生成模型在MRAMG任务中的性能。此外，我们还提出了一种高效的多模态答案生成框架，充分利用LLM和MLLM生成多模态响应。我们的数据集可在以下链接获取：[这里](this https URL)。 

---
# LLM Alignment as Retriever Optimization: An Information Retrieval Perspective 

**Title (ZH)**: LLM对齐作为检索器优化：从信息检索视角看待 

**Authors**: Bowen Jin, Jinsung Yoon, Zhen Qin, Ziqi Wang, Wei Xiong, Yu Meng, Jiawei Han, Sercan O. Arik  

**Link**: [PDF](https://arxiv.org/pdf/2502.03699)  

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence with capabilities in reasoning, coding, and communication, driving innovation across industries. Their true potential depends on effective alignment to ensure correct, trustworthy and ethical behavior, addressing challenges like misinformation, hallucinations, bias and misuse. While existing Reinforcement Learning (RL)-based alignment methods are notoriously complex, direct optimization approaches offer a simpler alternative. In this work, we introduce a novel direct optimization approach for LLM alignment by drawing on established Information Retrieval (IR) principles. We present a systematic framework that bridges LLM alignment and IR methodologies, mapping LLM generation and reward models to IR's retriever-reranker paradigm. Building on this foundation, we propose LLM Alignment as Retriever Preference Optimization (LarPO), a new alignment method that enhances overall alignment quality. Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 % averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work opens new avenues for advancing LLM alignment by integrating IR foundations, offering a promising direction for future research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过其推理、编程和通信能力革新了人工智能领域，并正在各行各业中推动创新。其真正潜力取决于有效的对齐，以确保正确的、可信的和符合伦理的行为，解决诸如假信息、幻觉、偏见和滥用等问题。虽然现有的基于强化学习（RL）的对齐方法非常复杂，直接优化的方法提供了一种更简单的选择。在本文中，我们通过借鉴成熟的检索原理（IR），介绍了一种新的直接优化方法，用于LLM对齐。我们提出了一种系统框架，将LLM对齐和检索方法论联系起来，将LLM生成和奖励模型映射到检索器-重排序器范式。基于这一基础，我们提出了LLM对齐作为一种检索偏好优化（LarPO）的新方法，以提高整体对齐质量。广泛的实验验证了LarPO的有效性，分别在AlpacaEval2和MixEval-Hard上平均提高了38.9%和13.7%。我们的工作通过整合检索基础，为LLM对齐开辟了新的途径，并为未来的研究提供了富有前景的方向。 

---
# Can Cross Encoders Produce Useful Sentence Embeddings? 

**Title (ZH)**: 跨编码器能否生成有用的主题词嵌入？ 

**Authors**: Haritha Ananthakrishnan, Julian Dolby, Harsha Kokel, Horst Samulowitz, Kavitha Srinivas  

**Link**: [PDF](https://arxiv.org/pdf/2502.03552)  

**Abstract**: Cross encoders (CEs) are trained with sentence pairs to detect relatedness. As CEs require sentence pairs at inference, the prevailing view is that they can only be used as re-rankers in information retrieval pipelines. Dual encoders (DEs) are instead used to embed sentences, where sentence pairs are encoded by two separate encoders with shared weights at training, and a loss function that ensures the pair's embeddings lie close in vector space if the sentences are related. DEs however, require much larger datasets to train, and are less accurate than CEs. We report a curious finding that embeddings from earlier layers of CEs can in fact be used within an information retrieval pipeline. We show how to exploit CEs to distill a lighter-weight DE, with a 5.15x speedup in inference time. 

**Abstract (ZH)**: 交叉编码器（CEs）通过训练句子对来检测相关性。由于CEs在推理过程中需要句子对，因此普遍认为它们只能作为信息检索管道中的重新排序器。双编码器（DEs）则用于嵌入句子，两个编码器在训练过程中共享权重，并且使用一个损失函数来确保相关句子的嵌入在向量空间中靠近。然而，DEs需要更大的数据集进行训练，并且通常不如CEs准确。我们报告了一个有趣的发现，即CEs早期层的嵌入实际上可以在信息检索管道中使用。我们展示了如何利用CEs来提炼一个较轻量级的DE，并实现了5.15倍的推理时间加速。 

---
