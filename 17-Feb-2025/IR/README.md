# A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems 

**Title (ZH)**: 面向在线推荐系统的混合跨阶段协调预排序模型 

**Authors**: Binglei Zhao, Houying Qi, Guang Xu, Mian Ma, Xiwei Zhao, Feng Mei, Sulong Xu, Jinghe Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10284)  

**Abstract**: Large-scale recommendation systems often adopt cascading architecture consisting of retrieval, pre-ranking, ranking, and re-ranking stages. With strict latency requirements, pre-ranking utilizes lightweight models to perform a preliminary selection from massive retrieved candidates. However, recent works focus solely on improving consistency with ranking, relying exclusively on downstream stages. Since downstream input is derived from the pre-ranking output, they will exacerbate the sample selection bias (SSB) issue and Matthew effect, leading to sub-optimal results. To address the limitation, we propose a novel Hybrid Cross-Stage Coordination Pre-ranking model (HCCP) to integrate information from upstream (retrieval) and downstream (ranking, re-ranking) stages. Specifically, cross-stage coordination refers to the pre-ranking's adaptability to the entire stream and the role of serving as a more effective bridge between upstream and downstream. HCCP consists of Hybrid Sample Construction and Hybrid Objective Optimization. Hybrid sample construction captures multi-level unexposed data from the entire stream and rearranges them to become the optimal guiding "ground truth" for pre-ranking learning. Hybrid objective optimization contains the joint optimization of consistency and long-tail precision through our proposed Margin InfoNCE loss. It is specifically designed to learn from such hybrid unexposed samples, improving the overall performance and mitigating the SSB issue. The appendix describes a proof of the efficacy of the proposed loss in selecting potential positives. Extensive offline and online experiments indicate that HCCP outperforms SOTA methods by improving cross-stage coordination. It contributes up to 14.9% UCVR and 1.3% UCTR in the JD E-commerce recommendation system. Concerning code privacy, we provide a pseudocode for reference. 

**Abstract (ZH)**: 大规模推荐系统通常采用递进架构，包括检索、预排序、排序和再排序阶段。由于存在严格的延迟要求，预排序阶段使用轻量级模型对大规模检索候选项进行初步筛选。然而，现有的研究工作主要集中在提升与排序阶段的一致性上，仅仅依赖下游阶段。由于下游输入源自预排序输出，这将加剧样本选择偏差（SSB）问题和马太效应，导致次优结果。为了解决这一局限性，我们提出了一种新颖的混合跨阶段协调预排序模型（HCCP），以整合上游（检索）和下游（排序、再排序）阶段的信息。具体而言，跨阶段协调指的是预排序在其整个流中的适应性和作为上游与下游更有效桥梁的角色。HCCP包含混合样本构建和混合目标优化。混合样本构建利用整个流中的多级未曝光数据，并重新安排这些数据以成为预排序学习的最佳引导“地面真值”。混合目标优化包括通过我们提出的Margin InfoNCE损失，对一致性和长尾精度的联合优化。该损失设计用于从这些混合未曝光样本中学习，从而提升整体性能并减轻SSB问题。附录中描述了提出损失有效性的证明。广泛离线和在线实验表明，HCCP通过提升跨阶段协调性能，优于现有最佳方法。在京东电商推荐系统中，HCCP贡献了高达14.9%的UCVR和1.3%的UCTR的改进。关于代码隐私，我们提供了参考用的伪代码。 

---
# SessionRec: Next Session Prediction Paradigm For Generative Sequential Recommendation 

**Title (ZH)**: SessionRec：生成式序列推荐中的下一个会话预测范式 

**Authors**: Lei Huang, Hao Guo, Linzhi Peng, Long Zhang, Xiaoteng Wang, Daoyuan Wang, Shichao Wang, Jinpeng Wang, Lei Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10157)  

**Abstract**: We introduce SessionRec, a novel next-session prediction paradigm (NSPP) for generative sequential recommendation, addressing the fundamental misalignment between conventional next-item prediction paradigm (NIPP) and real-world recommendation scenarios. Unlike NIPP's item-level autoregressive generation that contradicts actual session-based user interactions, our framework introduces a session-aware representation learning through hierarchical sequence aggregation (intra/inter-session), reducing attention computation complexity while enabling implicit modeling of massive negative interactions, and a session-based prediction objective that better captures users' diverse interests through multi-item recommendation in next sessions. Moreover, we found that incorporating a rank loss for items within the session under the next session prediction paradigm can significantly improve the ranking effectiveness of generative sequence recommendation models. We also verified that SessionRec exhibits clear power-law scaling laws similar to those observed in LLMs. Extensive experiments conducted on public datasets and online A/B test in Meituan App demonstrate the effectiveness of SessionRec. The proposed paradigm establishes new foundations for developing industrial-scale generative recommendation systems through its model-agnostic architecture and computational efficiency. 

**Abstract (ZH)**: 我们引入了SessionRec，这是一种针对生成性序列推荐的新会话预测范式（NSPP），解决了传统项目级自回归生成范式（NIPP）与现实世界推荐场景之间的基本不匹配问题。与NIPP基于项目级别的自回归生成方式相矛盾，我们的框架通过层次序列聚合（会内/会间）引入了会话感知的表示学习，从而降低注意力计算复杂性，同时能够隐式建模大量负面交互。此外，我们的框架通过在下一个会话中进行多项推荐以更好地捕捉用户多样化的兴趣，实现了基于会话的预测目标。进一步地，我们发现，在未来的会话预测范式中引入会话内部项目的排名损失可以显著提高生成序列推荐模型的排名效果。此外，我们验证了SessionRec呈现出与大规模语言模型（LLMs）类似的幂律缩放规律。通过公共数据集和美团应用上的在线A/B测试，我们证明了SessionRec的有效性。提出的范式通过其模型无关的架构和计算效率，为开发工业规模的生成推荐系统奠定了新的基础。 

---
# Semantica: Decentralized Search using a LLM-Guided Semantic Tree Overlay 

**Title (ZH)**: Semantica：由LLM引导的语义树overlay的去中心化搜索 

**Authors**: Petru Neague, Quinten Stokkink, Naman Goel, Johan Pouwelse  

**Link**: [PDF](https://arxiv.org/pdf/2502.10151)  

**Abstract**: Centralized search engines are key for the Internet, but lead to undesirable concentration of power. Decentralized alternatives fail to offer equal document retrieval accuracy and speed. Nevertheless, Semantic Overlay Networks can come close to the performance of centralized solutions when the semantics of documents are properly captured. This work uses embeddings from Large Language Models to capture semantics and fulfill the promise of Semantic Overlay Networks. Our proposed algorithm, called Semantica, constructs a prefix tree (trie) utilizing document embeddings calculated by a language model. Users connect to each other based on the embeddings of their documents, ensuring that semantically similar users are directly linked. Thereby, this construction makes it more likely for user searches to be answered by the users that they are directly connected to, or by the users they are close to in the network connection graph. The implementation of our algorithm also accommodates the semantic diversity of individual users by spawning "clone" user identifiers in the tree. Our experiments use emulation with a real-world workload to show Semantica's ability to identify and connect to similar users quickly. Semantica finds up to ten times more semantically similar users than current state-of-the-art approaches. At the same time, Semantica can retrieve more than two times the number of relevant documents given the same network load. We also make our code publicly available to facilitate further research in the area. 

**Abstract (ZH)**: 集中式搜索引擎是互联网的关键组成部分，但会导致权力过度集中。去中心化的替代方案在文档检索的准确性和速度方面无法与集中式解决方案相媲美。然而，语义Overlay网络在文档语义得到适当捕捉时，可以接近集中式解决方案的性能。本研究利用大规模语言模型的嵌入来捕捉语义，并实现语义Overlay网络的承诺。我们提出的算法称为Semantica，利用语言模型计算出来的文档嵌入构建前缀树（Trie）。用户基于其文档的嵌入相互连接，确保语义相似的用户能够直接相连。这种构造使得用户的搜索查询更有可能由他们直接连接的用户或在网络连接图中接近的用户来回答。此外，我们的算法通过生成“克隆”用户标识符来适应用户的语义多样性，从而使这些标识符在树中出现。我们的实验通过使用真实负载的模拟来展示Semantica快速识别和连接相似用户的的能力。Semantica在识别语义相似用户方面比当前最先进的方法多出10倍。同时，在相同的网络负载下，Semantica可以检索到超过两倍的相关文档数量。我们还将我们的代码公开，以促进该领域的进一步研究。 

---
# A Survey on LLM-powered Agents for Recommender Systems 

**Title (ZH)**: 基于大规模语言模型的推荐系统代理综述 

**Authors**: Qiyao Peng, Hongtao Liu, Hua Huang, Qing Yang, Minglai Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10050)  

**Abstract**: Recommender systems are essential components of many online platforms, yet traditional approaches still struggle with understanding complex user preferences and providing explainable recommendations. The emergence of Large Language Model (LLM)-powered agents offers a promising approach by enabling natural language interactions and interpretable reasoning, potentially transforming research in recommender systems. This survey provides a systematic review of the emerging applications of LLM-powered agents in recommender systems. We identify and analyze three key paradigms in current research: (1) Recommender-oriented approaches, which leverage intelligent agents to enhance the fundamental recommendation mechanisms; (2) Interaction-oriented approaches, which facilitate dynamic user engagement through natural dialogue and interpretable suggestions; and (3) Simulation-oriented approaches, which employ multi-agent frameworks to model complex user-item interactions and system dynamics. Beyond paradigm categorization, we analyze the architectural foundations of LLM-powered recommendation agents, examining their essential components: profile construction, memory management, strategic planning, and action execution. Our investigation extends to a comprehensive analysis of benchmark datasets and evaluation frameworks in this domain. This systematic examination not only illuminates the current state of LLM-powered agent recommender systems but also charts critical challenges and promising research directions in this transformative field. 

**Abstract (ZH)**: 推荐系统是许多在线平台的关键组成部分，尽管传统的做法仍然难以理解复杂的用户偏好并提供可解释的推荐。通过使用大型语言模型（LLM）驱动的代理，可以实现自然语言交互和可解释的推理，从而为推荐系统研究带来潜在的变革。本文综述了LLM驱动代理在推荐系统中的新兴应用。我们总结了当前研究中的三大主要范式：（1）以推荐为导向的方法，利用智能代理来增强基本的推荐机制；（2）以交互为导向的方法，通过自然对话和可解释的建议促进动态用户参与；以及（3）以仿真为导向的方法，利用多智能体框架来建模复杂的用户-项目交互和系统动力学。除了范式分类，我们还分析了LLM驱动推荐代理的架构基础，审视了其主要组成部分：用户档案构建、内存管理、战略规划和行动执行。我们进一步对这一领域的基准数据集和评估框架进行了全面分析。通过这种系统的检查，不仅揭示了LLM驱动代理推荐系统当前的状态，还指出了这一变革领域中关键挑战和有前途的研究方向。 

---
# ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation 

**Title (ZH)**: ArchRAG：带属性社区的层次检索增强生成 

**Authors**: Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, Yuchi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.09891)  

**Abstract**: Retrieval-Augmented Generation (RAG) has proven effective in integrating external knowledge into large language models (LLMs) for question-answer (QA) tasks. The state-of-the-art RAG approaches often use the graph data as the external data since they capture the rich semantic information and link relationships between entities. However, existing graph-based RAG approaches cannot accurately identify the relevant information from the graph and also consume large numbers of tokens in the online retrieval process. To address these issues, we introduce a novel graph-based RAG approach, called Attributed Community-based Hierarchical RAG (ArchRAG), by augmenting the question using attributed communities, and also introducing a novel LLM-based hierarchical clustering method. To retrieve the most relevant information from the graph for the question, we build a novel hierarchical index structure for the attributed communities and develop an effective online retrieval method. Experimental results demonstrate that ArchRAG outperforms existing methods in terms of both accuracy and token cost. 

**Abstract (ZH)**: 检索增强生成（RAG）在将外部知识集成到大型语言模型（LLMs）中以完成问题回答（QA）任务方面已被证明是有效的。最先进的RAG方法通常使用图数据作为外部数据，因为它们捕获了丰富的语义信息并链接了实体之间的关系。然而，现有的基于图的RAG方法无法准确识别图中的相关信息，并且在在线检索过程中消耗了大量的令牌。为了解决这些问题，我们提出了一种基于图的新型RAG方法，称为具属性社区的分层RAG（ArchRAG），该方法通过使用具属性社区来增强问题，并引入了一种新的基于LLM的分层聚类方法。为了从图中为问题检索最相关的信息，我们构建了一种新型的分层索引结构来为属性社区构建索引，并开发了一种有效的在线检索方法。实验结果表明，ArchRAG在准确性和令牌成本方面均优于现有方法。 

---
# An Efficient Large Recommendation Model: Towards a Resource-Optimal Scaling Law 

**Title (ZH)**: 一种高效的大规模推荐模型：向着资源优化的扩展律 

**Authors**: Songpei Xu, Shijia Wang, Da Guo, Xianwen Guo, Qiang Xiao, Fangjian Li, Chuanjiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.09888)  

**Abstract**: The pursuit of scaling up recommendation models confronts intrinsic tensions between expanding model capacity and preserving computational tractability. While prior studies have explored scaling laws for recommendation systems, their resource-intensive paradigms -- often requiring tens of thousands of A100 GPU hours -- remain impractical for most industrial applications. This work addresses a critical gap: achieving sustainable model scaling under strict computational budgets. We propose Climber, a resource-efficient recommendation framework comprising two synergistic components: the ASTRO model architecture for algorithmic innovation and the TURBO acceleration framework for engineering optimization. ASTRO (Adaptive Scalable Transformer for RecOmmendation) adopts two core innovations: (1) multi-scale sequence partitioning that reduces attention complexity from O(n^2d) to O(n^2d/Nb) via hierarchical blocks, enabling more efficient scaling with sequence length; (2) dynamic temperature modulation that adaptively adjusts attention scores for multimodal distributions arising from inherent multi-scenario and multi-behavior interactions. Complemented by TURBO (Two-stage Unified Ranking with Batched Output), a co-designed acceleration framework integrating gradient-aware feature compression and memory-efficient Key-Value caching, Climber achieves 5.15x throughput gains without performance degradation. Comprehensive offline experiments on multiple datasets validate that Climber exhibits a more ideal scaling curve. To our knowledge, this is the first publicly documented framework where controlled model scaling drives continuous online metric growth (12.19% overall lift) without prohibitive resource costs. Climber has been successfully deployed on Netease Cloud Music, one of China's largest music streaming platforms, serving tens of millions of users daily. 

**Abstract (ZH)**: 将推荐模型的扩展规模追求与扩大模型容量和保持计算可处理性之间的固有紧张关系相平衡是一项挑战。尽管先前的研究已经探索了推荐系统的标度定律，但其耗资源的范式——通常需要数万小时的A100 GPU时间——仍不适用于大多数工业应用。本研究填补了一个关键缺口：在严格计算预算下实现可持续的模型扩展。我们提出了一种资源高效的推荐框架——Climber，该框架包含两个协同的组件：ASTRO模型架构用于算法创新，以及TURBO加速框架用于工程优化。ASTRO（自适应可扩展变换器推荐模型）引入了两项核心创新：（1）多尺度序列划分，通过层次化块将注意力复杂度从O(n^2d)降低到O(n^2d/Nb)，从而在序列长度增加时实现更高效的扩展；（2）动态温度调节，能够根据固有的多场景和多行为交互产生的多模态分布自适应调整注意力得分。通过与TURBO（两阶段联合排名与批量输出）的配合，这是一种集成了梯度感知特征压缩和内存高效键值缓存的协同设计加速框架，Climber实现了5.15倍的吞吐量提升，而不会牺牲性能。在多个数据集上的全面离线实验验证，Climber展现了更理想化的扩展曲线。据我们所知，这是首个公开记录的框架，通过可控的模型扩展来推动持续的在线度量增长（整体提升12.19%），而不会带来高昂的资源成本。Climber已成功部署在中国最大的音乐流媒体平台之一的网易云音乐上，每天服务于数千万用户。 

---
# Data and Decision Traceability for the Welder's Arc 

**Title (ZH)**: 焊工电弧的数据与决策可追溯性 

**Authors**: Yasir Latif, Latha Pratti, Samya Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.09827)  

**Abstract**: Space Protocol is applying the principles derived from MITRE and NIST's Supply Chain Traceability: Manufacturing Meta-Framework (NIST IR 8536) to a complex multi party system to achieve introspection, auditing, and replay of data and decisions that ultimately lead to a end decision. The core goal of decision traceability is to ensure transparency, accountability, and integrity within the WA system. This is accomplished by providing a clear, auditable path from the system's inputs all the way to the final decision. This traceability enables the system to track the various algorithms and data flows that have influenced a particular outcome. 

**Abstract (ZH)**: 空间协议正在将MITRE和NIST的供应链可追溯性制造元框架（NIST IR 8536）的原则应用于一个复杂的多方系统中，以实现数据和决策的内部审视、审计和重演，最终达成终决决定。决策可追溯的核心目标是在WA系统中确保透明度、问责制和完整性。这一目标是通过提供从系统输入到最终决定的清晰、可审计路径来实现的。这种可追溯性使系统能够追踪影响特定结果的各种算法和数据流。 

---
# A Survey on LLM-based News Recommender Systems 

**Title (ZH)**: 基于大语言模型的新闻推荐系统综述 

**Authors**: Rongyao Wang, Veronica Liesaputra, Zhiyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09797)  

**Abstract**: News recommender systems play a critical role in mitigating the information overload problem. In recent years, due to the successful applications of large language model technologies, researchers have utilized Discriminative Large Language Models (DLLMs) or Generative Large Language Models (GLLMs) to improve the performance of news recommender systems. Although several recent surveys review significant challenges for deep learning-based news recommender systems, such as fairness, privacy-preserving, and responsibility, there is a lack of a systematic survey on Large Language Model (LLM)-based news recommender systems. In order to review different core methodologies and explore potential issues systematically, we categorize DLLM-based and GLLM-based news recommender systems under the umbrella of LLM-based news recommender systems. In this survey, we first overview the development of deep learning-based news recommender systems. Then, we review LLM-based news recommender systems based on three aspects: news-oriented modeling, user-oriented modeling, and prediction-oriented modeling. Next, we examine the challenges from various perspectives, including datasets, benchmarking tools, and methodologies. Furthermore, we conduct extensive experiments to analyze how large language model technologies affect the performance of different news recommender systems. Finally, we comprehensively explore the future directions for LLM-based news recommendations in the era of LLMs. 

**Abstract (ZH)**: 新闻推荐系统在缓解信息过载问题中发挥着关键作用。近年来，由于大语言模型技术的成功应用，研究人员利用判别大语言模型（DLLMs）或生成大语言模型（GLLMs）来提高新闻推荐系统的性能。虽然最近有一些调查回顾了基于深度学习的新闻推荐系统面临的重大挑战，如公平性、隐私保护和责任感，但对于基于大语言模型（LLMs）的新闻推荐系统缺乏系统性的回顾。为了系统地回顾不同核心方法并探索潜在问题，我们将判别大语言模型和生成大语言模型的新闻推荐系统归类为基于大语言模型的新闻推荐系统。在此调查中，我们首先概述基于深度学习的新闻推荐系统的研发情况。然后，我们根据三个方面对基于大语言模型的新闻推荐系统进行回顾：新闻导向建模、用户导向建模和预测导向建模。接下来，我们从多个角度分析挑战，包括数据集、基准测试工具和方法论。此外，我们进行了大量实验，分析大语言模型技术如何影响不同新闻推荐系统的性能。最后，我们全面探讨了大语言模型时代基于大语言模型的新闻推荐系统的未来发展方向。 

---
# ProReco: A Process Discovery Recommender System 

**Title (ZH)**: ProReco：一种流程发现推荐系统 

**Authors**: Tsung-Hao Huang, Tarek Junied, Marco Pegoraro, Wil M. P. van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2502.10230)  

**Abstract**: Process discovery aims to automatically derive process models from historical execution data (event logs). While various process discovery algorithms have been proposed in the last 25 years, there is no consensus on a dominating discovery algorithm. Selecting the most suitable discovery algorithm remains a challenge due to competing quality measures and diverse user requirements. Manually selecting the most suitable process discovery algorithm from a range of options for a given event log is a time-consuming and error-prone task. This paper introduces ProReco, a Process discovery Recommender system designed to recommend the most appropriate algorithm based on user preferences and event log characteristics. ProReco incorporates state-of-the-art discovery algorithms, extends the feature pools from previous work, and utilizes eXplainable AI (XAI) techniques to provide explanations for its recommendations. 

**Abstract (ZH)**: 过程发现旨在从历史执行数据（事件日志）中自动推导出过程模型。尽管在过去25年间提出了多种过程发现算法，但尚未形成主导性的发现算法。由于存在竞争性的质量度量和多样化的用户需求，选择最适合的过程发现算法仍然是一项挑战。针对特定事件日志手动选择最适合的过程发现算法是一个耗时且易出错的任务。本文介绍了一种名为ProReco的过程发现推荐系统，该系统可以根据用户偏好和事件日志特性推荐最合适的算法。ProReco整合了最新的发现算法，扩展了之前工作的特征池，并利用可解释的人工智能（XAI）技术来为其推荐提供解释。 

---
# KGGen: Extracting Knowledge Graphs from Plain Text with Language Models 

**Title (ZH)**: KGGen：使用语言模型从原始文本中提取知识图谱 

**Authors**: Belinda Mo, Kyssen Yu, Joshua Kazdan, Proud Mpala, Lisa Yu, Chris Cundy, Charilaos Kanatsoulis, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2502.09956)  

**Abstract**: Recent interest in building foundation models for KGs has highlighted a fundamental challenge: knowledge-graph data is relatively scarce. The best-known KGs are primarily human-labeled, created by pattern-matching, or extracted using early NLP techniques. While human-generated KGs are in short supply, automatically extracted KGs are of questionable quality. We present a solution to this data scarcity problem in the form of a text-to-KG generator (KGGen), a package that uses language models to create high-quality graphs from plaintext. Unlike other KG extractors, KGGen clusters related entities to reduce sparsity in extracted KGs. KGGen is available as a Python library (\texttt{pip install kg-gen}), making it accessible to everyone. Along with KGGen, we release the first benchmark, Measure of of Information in Nodes and Edges (MINE), that tests an extractor's ability to produce a useful KG from plain text. We benchmark our new tool against existing extractors and demonstrate far superior performance. 

**Abstract (ZH)**: 近年来，构建知识图谱（Knowledge Graphs，简称KGs）的基础模型引起了广泛关注，这突显出一个基本挑战：知识图谱数据相对稀缺。目前最出名的知识图谱主要由人类标注、模式匹配或早期自然语言处理（NLP）技术提取而成。虽然人工生成的知识图谱数量有限，但自动提取的知识图谱质量参差不齐。为了解决这一数据稀缺问题，我们提出了一种文本生成知识图谱（KGGen）的方法，这是一种使用语言模型从纯文本生成高质量图谱的工具包。与其它提取器不同，KGGen通过聚类相关实体来减少提取知识图谱的稀疏性。KGGen作为一种Python库（`pip install kg-gen`）可供所有人使用。除了KGGen，我们还发布了第一个基准测试——节点和边的信息量衡量标准（MINE），以测试提取器从纯文本生成有用知识图谱的能力。我们将新工具与现有工具进行基准测试，并展示了显著更优的性能。 

---
# Prioritized Ranking Experimental Design Using Recommender Systems in Two-Sided Platforms 

**Title (ZH)**: 在双边平台上使用推荐系统进行优先级排序实验设计 

**Authors**: Mahyar Habibi, Zahra Khanalizadeh, Negar Ziaeian  

**Link**: [PDF](https://arxiv.org/pdf/2502.09806)  

**Abstract**: Interdependencies between units in online two-sided marketplaces complicate estimating causal effects in experimental settings. We propose a novel experimental design to mitigate the interference bias in estimating the total average treatment effect (TATE) of item-side interventions in online two-sided marketplaces. Our Two-Sided Prioritized Ranking (TSPR) design uses the recommender system as an instrument for experimentation. TSPR strategically prioritizes items based on their treatment status in the listings displayed to users. We designed TSPR to provide users with a coherent platform experience by ensuring access to all items and a consistent realization of their treatment by all users. We evaluate our experimental design through simulations using a search impression dataset from an online travel agency. Our methodology closely estimates the true simulated TATE, while a baseline item-side estimator significantly overestimates TATE. 

**Abstract (ZH)**: 在线双边市场中各单位之间的相互依赖性使得在实验设置中估计因果效应变得复杂。我们提出了一种新的实验设计，以减轻在估计在线双边市场中物品侧干预的总平均治疗效果（TATE）时的交互偏差。我们的双侧优先排名（TSPR）设计利用推荐系统作为实验工具。TSPR会根据展示给用户的列表中的治疗状态战略地优先考虑物品。我们设计TSPR确保用户能够获得所有物品，并且所有用户能够以一致的方式体验它们的治疗效果。我们通过使用在线旅游代理机构的搜索曝光数据集进行模拟来评估我们的实验设计。我们的方法密切估计了真实的模拟TATE，而基准物品侧估计器则显著高估了TATE。 

---
