# Rank1: Test-Time Compute for Reranking in Information Retrieval 

**Title (ZH)**: 排名第一：检索时计算在信息检索中的重新排序计算 

**Authors**: Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2502.18418)  

**Abstract**: We introduce Rank1, the first reranking model trained to take advantage of test-time compute. Rank1 demonstrates the applicability within retrieval of using a reasoning language model (i.e. OpenAI's o1, Deepseek's R1, etc.) for distillation in order to rapidly improve the performance of a smaller model. We gather and open-source a dataset of more than 600,000 examples of R1 reasoning traces from queries and passages in MS MARCO. Models trained on this dataset show: (1) state-of-the-art performance on advanced reasoning and instruction following datasets; (2) work remarkably well out of distribution due to the ability to respond to user-input prompts; and (3) have explainable reasoning chains that can be given to users or RAG-based systems. Further, we demonstrate that quantized versions of these models retain strong performance while using less compute/memory. Overall, Rank1 shows that test-time compute allows for a fundamentally new type of explainable and performant reranker model for search. 

**Abstract (ZH)**: 我们引入了Rank1，这是第一个专门在测试时计算资源的帮助下进行训练的重排序模型。Rank1展示了在检索中使用推理语言模型（例如OpenAI的o1、Deepseek的R1等）进行知识蒸馏的可能性，以迅速提升小型模型的性能。我们收集并开源了一个包含超过600,000个R1推理跟踪实例的数据集，这些实例源自MS MARCO中的查询和段落。在这些数据集上训练的模型表现出以下特点：（1）在高级推理和指令跟随数据集上达到最先进的性能；（2）由于能够响应用户输入的提示，其出分布泛化能力非常出色；（3）具有可解释的推理链路，可以提供给用户或基于检索增强生成信息（RAG）的系统。此外，我们还展示了这些模型的量化版本在使用较少计算/内存资源的情况下仍然保持了强大的性能。总体而言，Rank1表明测试时计算资源为搜索任务提供了一种全新的可解释性和高性能的重排序模型。 

---
# A Unified Bayesian Perspective for Conventional and Robust Adaptive Filters 

**Title (ZH)**: 一种统一的贝叶斯视角下的传统和鲁棒自适应滤波器研究 

**Authors**: Leszek Szczecinski, Jacob Benesty, Eduardo Vinicius Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2502.18325)  

**Abstract**: In this work, we present a new perspective on the origin and interpretation of adaptive filters. By applying Bayesian principles of recursive inference from the state-space model and using a series of simplifications regarding the structure of the solution, we can present, in a unified framework, derivations of many adaptive filters which depend on the probabilistic model of the observational noise. In particular, under a Gaussian model, we obtain solutions well-known in the literature (such as LMS, NLMS, or Kalman filter), while using non-Gaussian noise, we obtain new families of adaptive filter. Notably, under assumption of Laplacian noise, we obtain a family of robust filters of which the signed-error algorithm is a well-known member, while other algorithms, derived effortlessly in the proposed framework, are entirely new. Numerical examples are shown to illustrate the properties and provide a better insight into the performance of the derived adaptive filters. 

**Abstract (ZH)**: 在这项工作中，我们提出了适应性滤波器起源及其解释的一个新的视角。通过在状态空间模型中应用递归推理的贝叶斯原则，并对解的结构进行一系列简化，我们可以在统一的框架下推导出多种依赖于观测噪声概率模型的适应性滤波器。特别是在高斯模型下，我们得到了文献中已知的多种解决方案（如LMS、NLMS或Kalman滤波器），而在使用非高斯噪声的情况下，我们则获得了新的适应性滤波器家族。值得注意的是，在拉普拉斯噪声假设下，我们获得了一族鲁棒滤波器，其中符号误差算法是一个已知成员，而其它在提出框架下轻松获得的算法则完全是新的。数值示例被用来说明这些适应性滤波器的性质，从而更好地理解其性能。 

---
# Neural Network Graph Similarity Computation Based on Graph Fusion 

**Title (ZH)**: 基于图融合的神经网络图相似性计算 

**Authors**: Zenghui Chang, Yiqiao Zhang, Hong Cai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18291)  

**Abstract**: Graph similarity learning, crucial for tasks such as graph classification and similarity search, focuses on measuring the similarity between two graph-structured entities. The core challenge in this field is effectively managing the interactions between graphs. Traditional methods often entail separate, redundant computations for each graph pair, leading to unnecessary complexity. This paper revolutionizes the approach by introducing a parallel graph interaction method called graph fusion. By merging the node sequences of graph pairs into a single large graph, our method leverages a global attention mechanism to facilitate interaction computations and to harvest cross-graph insights. We further assess the similarity between graph pairs at two distinct levels-graph-level and node-level-introducing two innovative, yet straightforward, similarity computation algorithms. Extensive testing across five public datasets shows that our model not only outperforms leading baseline models in graph-to-graph classification and regression tasks but also sets a new benchmark for performance and efficiency. The code for this paper is open-source and available at this https URL 

**Abstract (ZH)**: 图相似性学习对于图分类和相似性搜索等任务至关重要，其核心在于衡量两个图结构实体之间的相似性。该领域的核心挑战是如何有效地管理图之间的交互。传统方法通常需要为每对图进行单独且冗余的计算，导致不必要的复杂性。本文通过引入一种并行图交互方法——图融合，革新了这一方法。通过将每对图的节点序列合并成一个大图，我们的方法利用全局注意力机制来促进交互计算，并获得跨图洞察。此外，我们还在图级和节点级分别评估图对之间的相似性，提出了两个创新且简单有效的相似性计算算法。在五个公开数据集上进行的广泛测试表明，我们的模型不仅在图到图的分类和回归任务中表现出优于现有基线模型的性能，还为性能和效率设定了新标准。本文的代码已开源，可以通过以下链接获取：[此链接] 

---
# HyperG: Hypergraph-Enhanced LLMs for Structured Knowledge 

**Title (ZH)**: HyperG：基于超图增强的大语言模型处理结构化知识 

**Authors**: Sirui Huang, Hanqian Li, Yanggan Gu, Xuming Hu, Qing Li, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18125)  

**Abstract**: Given that substantial amounts of domain-specific knowledge are stored in structured formats, such as web data organized through HTML, Large Language Models (LLMs) are expected to fully comprehend this structured information to broaden their applications in various real-world downstream tasks. Current approaches for applying LLMs to structured data fall into two main categories: serialization-based and operation-based methods. Both approaches, whether relying on serialization or using SQL-like operations as an intermediary, encounter difficulties in fully capturing structural relationships and effectively handling sparse data. To address these unique characteristics of structured data, we propose HyperG, a hypergraph-based generation framework aimed at enhancing LLMs' ability to process structured knowledge. Specifically, HyperG first augment sparse data with contextual information, leveraging the generative power of LLMs, and incorporate a prompt-attentive hypergraph learning (PHL) network to encode both the augmented information and the intricate structural relationships within the data. To validate the effectiveness and generalization of HyperG, we conduct extensive experiments across two different downstream tasks requiring structured knowledge. 

**Abstract (ZH)**: 鉴于大量特定领域的知识以结构化格式存储，例如通过HTML组织的Web数据，大型语言模型（LLMs）预计将全面理解这些结构化信息，从而在各种现实世界的下游任务中得到更广泛的应用。目前将LLMs应用到结构化数据中的方法主要分为两类：序列化方法和操作方法。不论是依赖于序列化方法还是通过SQL样式的操作作为中介，这两种方法都难以全面捕捉结构关系并有效地处理稀疏数据。为应对结构化数据的独特特征，我们提出了一种基于超图的生成框架HyperG，旨在增强LLMs处理结构化知识的能力。具体而言，HyperG首先通过上下文信息增强稀疏数据，利用LLMs的生成能力，并结合一个提示感知超图学习（PHL）网络来编码增强的信息以及数据中的复杂结构关系。为了验证HyperG的有效性和泛化能力，我们在两个不同的需要结构化知识的下游任务中进行了广泛的实验。 

---
# Tip of the Tongue Query Elicitation for Simulated Evaluation 

**Title (ZH)**: 舌尖上的查询诱致方法在模拟评估中的应用 

**Authors**: Yifan He, To Eun Kim, Fernando Diaz, Jaime Arguello, Bhaskar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2502.17776)  

**Abstract**: Tip-of-the-tongue (TOT) search occurs when a user struggles to recall a specific identifier, such as a document title. While common, existing search systems often fail to effectively support TOT scenarios. Research on TOT retrieval is further constrained by the challenge of collecting queries, as current approaches rely heavily on community question-answering (CQA) websites, leading to labor-intensive evaluation and domain bias. To overcome these limitations, we introduce two methods for eliciting TOT queries - leveraging large language models (LLMs) and human participants - to facilitate simulated evaluations of TOT retrieval systems. Our LLM-based TOT user simulator generates synthetic TOT queries at scale, achieving high correlations with how CQA-based TOT queries rank TOT retrieval systems when tested in the Movie domain. Additionally, these synthetic queries exhibit high linguistic similarity to CQA-derived queries. For human-elicited queries, we developed an interface that uses visual stimuli to place participants in a TOT state, enabling the collection of natural queries. In the Movie domain, system rank correlation and linguistic similarity analyses confirm that human-elicited queries are both effective and closely resemble CQA-based queries. These approaches reduce reliance on CQA-based data collection while expanding coverage to underrepresented domains, such as Landmark and Person. LLM-elicited queries for the Movie, Landmark, and Person domains have been released as test queries in the TREC 2024 TOT track, with human-elicited queries scheduled for inclusion in the TREC 2025 TOT track. Additionally, we provide source code for synthetic query generation and the human query collection interface, along with curated visual stimuli used for eliciting TOT queries. 

**Abstract (ZH)**: 舌尖效应（Tip-of-the-tongue, TOT）搜索发生在用户努力回忆特定标识符，如文档标题时。虽然TOT搜索是常见的，但现有的搜索系统往往难以有效支持TOT场景。由于当前收集查询的方法主要依赖社区问答（CQA）网站，这限制了研究并导致了劳动密集型的评估和领域偏差。为克服这些限制，我们提出了两种方法来 eliciting TOT 查询——利用大规模语言模型（LLMs）和人类参与者——以促进TOT检索系统的模拟评估。我们基于LLM的TOT用户模拟器大规模生成合成的TOT查询，当在电影领域测试时，这些合成查询与基于CQA的TOT查询对TOT检索系统进行排名的相关性非常高。此外，这些合成查询在语言学上与CQA衍生查询高度相似。对于人类引发的查询，我们开发了一个界面，使用视觉刺激来使参与者处于TOT状态，从而收集自然查询。在电影领域，系统排名相关性和语言学相似性分析证实，人类引发的查询既有效又与CQA衍生查询高度相似。这些方法减少了对基于CQA的数据收集的依赖，同时扩大了覆盖范围，包括较少被代表的领域，如地标（Landmark）和人物（Person）。电影、地标和人物领域的LLM引发的查询已作为测试查询发布，人类引发的查询将在TREC 2025 TOT 任务中增加。此外，我们还提供了生成合成查询和人类查询收集界面的源代码，并提供了用于引发TOT查询的精选视觉刺激。 

---
# External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation 

**Title (ZH)**: 外部大型基础模型：如何高效处理万亿参数的在线广告推荐 

**Authors**: Mingfu Liang, Xi Liu, Rong Jin, Boyang Liu, Qiuling Suo, Qinghai Zhou, Song Zhou, Laming Chen, Hua Zheng, Zhiyuan Li, Shali Jiang, Jiyan Yang, Xiaozhen Xia, Fan Yang, Yasmine Badr, Ellie Wen, Shuyu Xu, Hansey Chen, Zhengyu Zhang, Jade Nie, Chunzhi Yang, Zhichen Zeng, Weilin Zhang, Xingliang Huang, Qianru Li, Shiquan Wang, Evelyn Lyu, Wenjing Lu, Rui Zhang, Wenjun Wang, Jason Rudy, Mengyue Hang, Kai Wang, Yinbin Ma, Shuaiwen Wang, Sihan Zeng, Tongyi Tang, Xiaohan Wei, Longhao Jin, Jamey Zhang, Marcus Chen, Jiayi Zhang, Angie Huang, Chi Zhang, Zhengli Zhao, Jared Yang, Qiang Jin, Xian Chen, Amit Anand Amlesahwaram, Lexi Song, Liang Luo, Yuchen Hao, Nan Xiao, Yavuz Yetim, Luoshang Pan, Gaoxiang Liu, Yuxi Hu, Yuzhen Huang, Jackie Xu, Rich Zhu, Xin Zhang, Yiqun Liu, Hang Yin, Yuxin Chen, Buyun Zhang, Xiaoyi Liu, Sylvia Wang, Wenguang Mao, Zhijing Li, Qin Huang, Chonglin Sun, Shupin Mao, Jingzheng Qin, Peggy Yao, Jae-Woo Choi, Bin Gao, Ernest Wang, Lei Zhang, Wen-Yen Chen, Ted Lee, Jay Zha, Yi Meng, Alex Gong, Edison Gao, Alireza Vahdatpour, Yiping Han, Yantao Yao, Toshinari Kureha, Shuo Chang, Musharaf Sultan, John Bocharov, Sagar Chordia, Xiaorui Gan, Peng Sun, Rocky Liu, Bo Long, Wenlin Chen, Santanu Kolay, Huayu Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17494)  

**Abstract**: Ads recommendation is a prominent service of online advertising systems and has been actively studied. Recent studies indicate that scaling-up and advanced design of the recommendation model can bring significant performance improvement. However, with a larger model scale, such prior studies have a significantly increasing gap from industry as they often neglect two fundamental challenges in industrial-scale applications. First, training and inference budgets are restricted for the model to be served, exceeding which may incur latency and impair user experience. Second, large-volume data arrive in a streaming mode with data distributions dynamically shifting, as new users/ads join and existing users/ads leave the system. We propose the External Large Foundation Model (ExFM) framework to address the overlooked challenges. Specifically, we develop external distillation and a data augmentation system (DAS) to control the computational cost of training/inference while maintaining high performance. We design the teacher in a way like a foundation model (FM) that can serve multiple students as vertical models (VMs) to amortize its building cost. We propose Auxiliary Head and Student Adapter to mitigate the data distribution gap between FM and VMs caused by the streaming data issue. Comprehensive experiments on internal industrial-scale applications and public datasets demonstrate significant performance gain by ExFM. 

**Abstract (ZH)**: 在线广告系统中的广告推荐是一项重要的服务，近年来受到了广泛关注和积极研究。近期的研究表明，通过扩大推荐模型的规模和优化设计，可以显著提高推荐性能。然而，随着模型规模的增大，这类先前的研究与工业应用之间存在显著的差距，主要原因是它们往往忽视了大规模工业应用中的两个基本挑战。首先，模型的训练和推理预算是受限的，超出预算可能导致延迟，从而影响用户体验。其次，大量的数据以流式方式到达，数据分布随着新用户和新广告的加入以及旧用户和旧广告的退出而动态变化。我们提出了外部大型基础模型（ExFM）框架来解决这些被忽视的挑战。具体而言，我们开发了外部蒸馏和数据增强系统（DAS），以在控制训练和推理的计算成本的同时保持高性能。我们设计了教师模型，该模型类似于基础模型（FM），可以支持多个学生模型（VMs）以摊销其构建成本。我们提出了辅助头和学生适配器来缓解由流式数据问题引起的FM和VMs之间数据分布差距。在内部大规模工业应用和公开数据集上的全面实验表明，ExFM能显著提高性能。 

---
# DRAMA: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers 

**Title (ZH)**: DRAMA: 大型语言模型向小型密集检索模型的多样化增强 

**Authors**: Xueguang Ma, Xi Victoria Lin, Barlas Oguz, Jimmy Lin, Wen-tau Yih, Xilun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18460)  

**Abstract**: Large language models (LLMs) have demonstrated strong effectiveness and robustness while fine-tuned as dense retrievers. However, their large parameter size brings significant inference time computational challenges, including high encoding costs for large-scale corpora and increased query latency, limiting their practical deployment. While smaller retrievers offer better efficiency, they often fail to generalize effectively with limited supervised fine-tuning data. In this work, we introduce DRAMA, a training framework that leverages LLMs to train smaller generalizable dense retrievers. In particular, we adopt pruned LLMs as the backbone and train on diverse LLM-augmented data in a single-stage contrastive learning setup. Experiments show that DRAMA offers better multilingual and long-context capabilities than traditional encoder-based retrievers, and achieves strong performance across multiple tasks and languages. These highlight the potential of connecting the training of smaller retrievers with the growing advancements in LLMs, bridging the gap between efficiency and generalization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在微调为密集检索器后显示出强大的有效性和鲁棒性。然而，它们庞大的参数规模带来了显著的推理时间计算挑战，包括大规模语料库的高编码成本和查询延迟的增加，从而限制了它们的实用部署。尽管较小的检索器更有效，但它们往往由于有限的监督微调数据而难以有效泛化。在本项工作中，我们引入了DRAMA，这是一种利用LLMs来训练更小的泛化密集检索器的训练框架。特别地，我们采用剪枝后的LLMs作为骨干，并在单阶段对比学习设置中对多样化增强的LLM数据进行训练。实验表明，DRAMA在多语言和长上下文能力方面优于传统的基于编码器的检索器，并在多个任务和语言上取得了强大的性能。这些结果凸显了连接较小检索器的训练与不断发展的LLMs的潜力，从而弥合了效率与泛化之间的差距。 

---
# How Vital is the Jurisprudential Relevance: Law Article Intervened Legal Case Retrieval and Matching 

**Title (ZH)**: 法律判例的相关性至关重要：法律文章干预下的案例检索与匹配 

**Authors**: Nuo Xu, Pinghui Wang, Zi Liang, Junzhou Zhao, Xiaohong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18292)  

**Abstract**: Legal case retrieval (LCR) aims to automatically scour for comparable legal cases based on a given query, which is crucial for offering relevant precedents to support the judgment in intelligent legal systems. Due to similar goals, it is often associated with a similar case matching (LCM) task. To address them, a daunting challenge is assessing the uniquely defined legal-rational similarity within the judicial domain, which distinctly deviates from the semantic similarities in general text retrieval. Past works either tagged domain-specific factors or incorporated reference laws to capture legal-rational information. However, their heavy reliance on expert or unrealistic assumptions restricts their practical applicability in real-world scenarios. In this paper, we propose an end-to-end model named LCM-LAI to solve the above challenges. Through meticulous theoretical analysis, LCM-LAI employs a dependent multi-task learning framework to capture legal-rational information within legal cases by a law article prediction (LAP) sub-task, without any additional assumptions in inference. Besides, LCM-LAI proposes an article-aware attention mechanism to evaluate the legal-rational similarity between across-case sentences based on law distribution, which is more effective than conventional semantic similarity. Weperform a series of exhaustive experiments including two different tasks involving four real-world datasets. Results demonstrate that LCM-LAI achieves state-of-the-art performance. 

**Abstract (ZH)**: 法律案例检索（LCR）旨在基于给定的查询自动搜索相似的法律案例，这对于在智能法律系统中提供相关的先例以支持判决至关重要。由于相似的目标，它通常与类似案例匹配任务（LCM）相关联。为了解决这些问题，一个重大挑战在于评估司法领域中被独特定义的法律理性相似性，这与常规文本检索中的语义相似性截然不同。以往的工作要么标记领域特定因素，要么引入参考法律来捕捉法律理性的信息。然而，它们对专家知识的依赖或不切实际的假设限制了其在实际应用场景中的实用性。在本文中，我们提出了一种名为LCM-LAI的端到端模型来解决上述挑战。通过细致的理论分析，LCM-LAI采用依赖多任务学习框架，在法律案例中通过法律条文预测（LAP）子任务来捕获法律理性的信息，无需额外假设。此外，LCM-LAI提出了基于法律分布的文档意识注意力机制，以评估案例间句子的法律理性相似性，其效果优于传统的语义相似性。我们进行了包括两个不同任务的详尽实验，涉及四个真实世界的数据集。结果表明，LCM-LAI取得了最先进的性能。 

---
# LevelRAG: Enhancing Retrieval-Augmented Generation with Multi-hop Logic Planning over Rewriting Augmented Searchers 

**Title (ZH)**: LevelRAG：通过重写增强的检索增强生成中多跳逻辑规划的应用 

**Authors**: Zhuocheng Zhang, Yang Feng, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18139)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a crucial method for mitigating hallucinations in Large Language Models (LLMs) and integrating external knowledge into their responses. Existing RAG methods typically employ query rewriting to clarify the user intent and manage multi-hop logic, while using hybrid retrieval to expand search scope. However, the tight coupling of query rewriting to the dense retriever limits its compatibility with hybrid retrieval, impeding further RAG performance improvements. To address this challenge, we introduce a high-level searcher that decomposes complex queries into atomic queries, independent of any retriever-specific optimizations. Additionally, to harness the strengths of sparse retrievers for precise keyword retrieval, we have developed a new sparse searcher that employs Lucene syntax to enhance retrieval this http URL web and dense searchers, these components seamlessly collaborate within our proposed method, \textbf{LevelRAG}. In LevelRAG, the high-level searcher orchestrates the retrieval logic, while the low-level searchers (sparse, web, and dense) refine the queries for optimal retrieval. This approach enhances both the completeness and accuracy of the retrieval process, overcoming challenges associated with current query rewriting techniques in hybrid retrieval scenarios. Empirical experiments conducted on five datasets, encompassing both single-hop and multi-hop question answering tasks, demonstrate the superior performance of LevelRAG compared to existing RAG methods. Notably, LevelRAG outperforms the state-of-the-art proprietary model, GPT4o, underscoring its effectiveness and potential impact on the RAG field. 

**Abstract (ZH)**: 检索增强生成（RAG）是减轻大型语言模型（LLMs）幻觉现象和将外部知识整合到其回应中的关键方法。现有RAG方法通常通过查询重新写入来澄清用户意图并管理多跳逻辑，同时使用混合检索来扩大搜索范围。然而，查询重新写入与密集检索器的紧密耦合限制了其与混合检索的兼容性，阻碍了RAG性能的进一步提升。为了解决这一挑战，我们引入了一个高级搜索器，将复杂查询分解为原子查询，不依赖于任何特定检索器的优化。此外，为了利用稀疏检索器的优势进行精准关键词检索，我们开发了一种新的稀疏搜索器，利用Lucene语法增强检索。这些组件在我们提出的方法——**LevelRAG** 中无缝协作。在LevelRAG中，高级搜索器统筹检索逻辑，而低级搜索器（稀疏、网页和密集）则对查询进行优化以实现最佳检索。这种方法提高了检索过程的完整性和准确性，克服了当前混合检索场景中查询重新写入技术面临的挑战。我们在五个数据集上进行了实证实验，涵盖了单跳和多跳问答任务，结果表明LevelRAG在与现有RAG方法的比较中表现出更优的性能。特别地，LevelRAG在性能上超越了当前最先进的商业化模型GPT4o，这进一步证明了其有效性和对RAG领域的潜在影响。 

---
# AfroXLMR-Comet: Multilingual Knowledge Distillation with Attention Matching for Low-Resource languages 

**Title (ZH)**: AfroXLMR-Comet: 基于注意力匹配的多语言知识蒸馏方法，用于低资源语言 

**Authors**: Joshua Sakthivel Raju, Sanjay S, Jaskaran Singh Walia, Srinivas Raghav, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2502.18020)  

**Abstract**: Language model compression through knowledge distillation has emerged as a promising approach for deploying large language models in resource-constrained environments. However, existing methods often struggle to maintain performance when distilling multilingual models, especially for low-resource languages. In this paper, we present a novel hybrid distillation approach that combines traditional knowledge distillation with a simplified attention matching mechanism, specifically designed for multilingual contexts. Our method introduces an extremely compact student model architecture, significantly smaller than conventional multilingual models. We evaluate our approach on five African languages: Kinyarwanda, Swahili, Hausa, Igbo, and Yoruba. The distilled student model; AfroXLMR-Comet successfully captures both the output distribution and internal attention patterns of a larger teacher model (AfroXLMR-Large) while reducing the model size by over 85%. Experimental results demonstrate that our hybrid approach achieves competitive performance compared to the teacher model, maintaining an accuracy within 85% of the original model's performance while requiring substantially fewer computational resources. Our work provides a practical framework for deploying efficient multilingual models in resource-constrained environments, particularly benefiting applications involving African languages. 

**Abstract (ZH)**: 通过对知识蒸馏的压缩语言模型已经在资源受限环境中部署大型语言模型方面展现出了前景。然而，现有方法在蒸馏多语言模型时往往难以保持性能，尤其是在低资源语言方面。在本文中，我们提出了一种新颖的混合蒸馏方法，该方法结合了传统的知识蒸馏和简化的注意力匹配机制，专门针对多语言上下文。我们的方法引入了一个极其紧凑的学生模型架构，其大小远小于传统的多语言模型。我们在五种非洲语言（基卢瓦语、斯瓦希里语、豪萨语、伊博语和约鲁巴语）上评估了我们的方法。蒸馏后的学生模型AfroXLMR-Comet成功捕获了更大教师模型（AfroXLMR-Large）的输出分布和内部注意力模式，同时将模型大小减少了85%以上。实验结果显示，我们的混合方法在性能上与教师模型相当，保持了与原始模型85%以上的准确率，同时所需计算资源大幅减少。我们的工作提供了一种实用的框架，在资源受限环境中部署高效的多语言模型，特别有利于涉及非洲语言的应用。 

---
# ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents 

**Title (ZH)**: ViDoRAG：通过动态迭代推理代理增强的视觉文档检索生成方法 

**Authors**: Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18017)  

**Abstract**: Understanding information from visually rich documents remains a significant challenge for traditional Retrieval-Augmented Generation (RAG) methods. Existing benchmarks predominantly focus on image-based question answering (QA), overlooking the fundamental challenges of efficient retrieval, comprehension, and reasoning within dense visual documents. To bridge this gap, we introduce ViDoSeek, a novel dataset designed to evaluate RAG performance on visually rich documents requiring complex reasoning. Based on it, we identify key limitations in current RAG approaches: (i) purely visual retrieval methods struggle to effectively integrate both textual and visual features, and (ii) previous approaches often allocate insufficient reasoning tokens, limiting their effectiveness. To address these challenges, we propose ViDoRAG, a novel multi-agent RAG framework tailored for complex reasoning across visual documents. ViDoRAG employs a Gaussian Mixture Model (GMM)-based hybrid strategy to effectively handle multi-modal retrieval. To further elicit the model's reasoning capabilities, we introduce an iterative agent workflow incorporating exploration, summarization, and reflection, providing a framework for investigating test-time scaling in RAG domains. Extensive experiments on ViDoSeek validate the effectiveness and generalization of our approach. Notably, ViDoRAG outperforms existing methods by over 10% on the competitive ViDoSeek benchmark. 

**Abstract (ZH)**: 传统的检索增强生成（RAG）方法在理解视觉丰富的文档方面仍然面临重大挑战。现有的基准主要集中在基于图像的问答（QA），忽视了密集视觉文档中高效检索、理解和推理的基本挑战。为解决这一问题，我们提出了ViDoSeek，一个新的数据集，旨在评估RAG方法在需要复杂推理的视觉丰富文档中的性能。基于此，我们识别了当前RAG方法的关键局限性：（i）纯粹基于视觉的检索方法难以有效地整合文本和视觉特征，（ii）之前的许多方法在推理方面的效率受到限制，因为它们分配的推理令牌往往不足。为了应对这些挑战，我们提出了ViDoRAG，一个专门为视觉文档中的复杂推理设计的新型多智能体RAG框架。ViDoRAG采用基于高斯混合模型（GMM）的混合策略来有效处理多模态检索。为了进一步激发模型的推理能力，我们引入了一种迭代的智能体工作流，包含探索、总结和反思，为RAG领域中测试时的扩展提供一个框架。我们在ViDoSeek上的广泛实验验证了我们的方法的有效性和泛化能力。值得注意的是，ViDoRAG在竞争性的ViDoSeek基准上比现有方法高出10%以上。 

---
# MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification 

**Title (ZH)**: MAGE：多头注意力引导嵌入在资源贫乏的 sentiment 分类中的应用 

**Authors**: Varun Vashisht, Samar Singh, Mihir Konduskar, Jaskaran Singh Walia, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2502.17987)  

**Abstract**: Due to the lack of quality data for low-resource Bantu languages, significant challenges are presented in text classification and other practical implementations. In this paper, we introduce an advanced model combining Language-Independent Data Augmentation (LiDA) with Multi-Head Attention based weighted embeddings to selectively enhance critical data points and improve text classification performance. This integration allows us to create robust data augmentation strategies that are effective across various linguistic contexts, ensuring that our model can handle the unique syntactic and semantic features of Bantu languages. This approach not only addresses the data scarcity issue but also sets a foundation for future research in low-resource language processing and classification tasks. 

**Abstract (ZH)**: 由于低资源班图语缺乏高质量数据，文本分类及其他实际应用面临显著挑战。本文介绍了一种结合语言无关数据增强（LiDA）与多头注意力加权嵌入的先进模型，该模型能够选择性地增强关键数据点，从而提高文本分类性能。这种整合使得我们可以创建出跨各种语言环境都有效的健壮数据增强策略，确保模型能够处理班图语言的独特句法和语义特征。该方法不仅解决了数据稀缺的问题，也为未来低资源语言处理和分类任务的研究奠定了基础。 

---
# On Synthetic Data Strategies for Domain-Specific Generative Retrieval 

**Title (ZH)**: 面向特定领域生成检索的合成数据策略 

**Authors**: Haoyang Wen, Jiang Guo, Yi Zhang, Jiarong Jiang, Zhiguo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17957)  

**Abstract**: This paper investigates synthetic data generation strategies in developing generative retrieval models for domain-specific corpora, thereby addressing the scalability challenges inherent in manually annotating in-domain queries. We study the data strategies for a two-stage training framework: in the first stage, which focuses on learning to decode document identifiers from queries, we investigate LLM-generated queries across multiple granularity (e.g. chunks, sentences) and domain-relevant search constraints that can better capture nuanced relevancy signals. In the second stage, which aims to refine document ranking through preference learning, we explore the strategies for mining hard negatives based on the initial model's predictions. Experiments on public datasets over diverse domains demonstrate the effectiveness of our synthetic data generation and hard negative sampling approach. 

**Abstract (ZH)**: 本文探讨了在开发针对特定领域的生成检索模型时的合成数据生成策略，以应对手动标注领域内查询所固有的可扩展性挑战。我们研究了两级训练框架中的数据策略：在第一阶段，重点学习从查询中解码文档标识符，我们研究了语言模型（LLM）生成的查询在多种粒度（例如段落、句子）和与领域相关搜索约束下的表现，以更好地捕捉细微的相关性信号。在第二阶段，目标是通过偏好学习精化文档排名，我们探讨了基于初始模型预测结果进行困难负样本挖掘的策略。在不同领域的公开数据集上进行的实验表明，我们的合成数据生成和困难负样本采样方法的有效性。 

---
# Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness 

**Title (ZH)**: 揭露推荐系统中的性别偏见并增强类别感知公平性 

**Authors**: Tahsin Alamgir Kheya, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2502.17921)  

**Abstract**: Recommendation systems are now an integral part of our daily lives. We rely on them for tasks such as discovering new movies, finding friends on social media, and connecting job seekers with relevant opportunities. Given their vital role, we must ensure these recommendations are free from societal stereotypes. Therefore, evaluating and addressing such biases in recommendation systems is crucial. Previous work evaluating the fairness of recommended items fails to capture certain nuances as they mainly focus on comparing performance metrics for different sensitive groups. In this paper, we introduce a set of comprehensive metrics for quantifying gender bias in recommendations. Specifically, we show the importance of evaluating fairness on a more granular level, which can be achieved using our metrics to capture gender bias using categories of recommended items like genres for movies. Furthermore, we show that employing a category-aware fairness metric as a regularization term along with the main recommendation loss during training can help effectively minimize bias in the models' output. We experiment on three real-world datasets, using five baseline models alongside two popular fairness-aware models, to show the effectiveness of our metrics in evaluating gender bias. Our metrics help provide an enhanced insight into bias in recommended items compared to previous metrics. Additionally, our results demonstrate how incorporating our regularization term significantly improves the fairness in recommendations for different categories without substantial degradation in overall recommendation performance. 

**Abstract (ZH)**: 推荐系统如今已成为我们日常生活中不可或缺的一部分。我们依赖它们来发现新电影、在社交媒体上寻找朋友以及将求职者与相关机会联系起来。鉴于它们的重要性，我们必须确保这些推荐不含有社会偏见。因此，评估和解决推荐系统中的这种偏见至关重要。以往关于推荐项目公平性的评估工作未能全面捕捉到某些细微之处，因为它们主要关注不同敏感群体的性能指标对比。在本文中，我们引入了一套全面的指标来量化推荐中的性别偏见。具体而言，我们展示了在更为细腻的层面上评估公平性的必要性，这可以通过我们的指标来实现，这些指标可以利用推荐项目的类别，例如电影的类型来捕捉性别偏见。此外，我们展示了在训练过程中使用类别感知的公平性指标作为正则化项，同时与主要的推荐损失一起使用，可以有效减少模型输出中的偏见。我们在三个真实世界的数据集上进行了实验，使用了五个基线模型以及两种流行的公平性意识模型，以证明我们提出的指标在评估性别偏见方面的有效性。我们的指标相比之前的方法能更深入地揭示推荐项目的偏见。此外，我们的结果还表明，引入我们的正则化项可以显著提高不同类别中推荐的公平性，同时不会大幅降低总体推荐性能。 

---
# The GigaMIDI Dataset with Features for Expressive Music Performance Detection 

**Title (ZH)**: gigamidi数据集及其用于表达性音乐表演检测的特征 

**Authors**: Keon Ju Maverick Lee, Jeff Ens, Sara Adkins, Pedro Sarmento, Mathieu Barthet, Philippe Pasquier  

**Link**: [PDF](https://arxiv.org/pdf/2502.17726)  

**Abstract**: The Musical Instrument Digital Interface (MIDI), introduced in 1983, revolutionized music production by allowing computers and instruments to communicate efficiently. MIDI files encode musical instructions compactly, facilitating convenient music sharing. They benefit Music Information Retrieval (MIR), aiding in research on music understanding, computational musicology, and generative music. The GigaMIDI dataset contains over 1.4 million unique MIDI files, encompassing 1.8 billion MIDI note events and over 5.3 million MIDI tracks. GigaMIDI is currently the largest collection of symbolic music in MIDI format available for research purposes under fair dealing. Distinguishing between non-expressive and expressive MIDI tracks is challenging, as MIDI files do not inherently make this distinction. To address this issue, we introduce a set of innovative heuristics for detecting expressive music performance. These include the Distinctive Note Velocity Ratio (DNVR) heuristic, which analyzes MIDI note velocity; the Distinctive Note Onset Deviation Ratio (DNODR) heuristic, which examines deviations in note onset times; and the Note Onset Median Metric Level (NOMML) heuristic, which evaluates onset positions relative to metric levels. Our evaluation demonstrates these heuristics effectively differentiate between non-expressive and expressive MIDI tracks. Furthermore, after evaluation, we create the most substantial expressive MIDI dataset, employing our heuristic, NOMML. This curated iteration of GigaMIDI encompasses expressively-performed instrument tracks detected by NOMML, containing all General MIDI instruments, constituting 31% of the GigaMIDI dataset, totalling 1,655,649 tracks. 

**Abstract (ZH)**: 1983年引入的Musical Instrument Digital Interface (MIDI) 革新了音乐制作，使其能够使计算机和乐器高效沟通。MIDI文件通过紧凑编码音乐指令，方便了音乐分享。这些文件对音乐信息检索（MIR）研究大有裨益，有助于音乐理解、计算音乐学和生成音乐的研究。GigaMIDI数据集包含超过140万份独特的MIDI文件，共有18亿个MIDI音乐事件以及超过530万条MIDI音轨。GigaMIDI目前是可用的、根据合理使用原则进行研究的最大规模MIDI格式符号音乐数据集。

区分非表现性和表现性MIDI音轨具有挑战性，因为MIDI文件本身并没有这种区分。为解决这一问题，我们引入了一套创新的启发式方法来检测表现性音乐表演。这些方法包括独特的音符速度比（Distinctive Note Velocity Ratio, DNVR）启发式，用于分析MIDI音符速度；独特的音符起始时间偏差比（Distinctive Note Onset Deviation Ratio, DNODR）启发式，用于检查音符起始时间的偏差；以及基于音符起始位置和节奏级水平（Note Onset Median Metric Level, NOMML）评价启发式，用于评估音符起始位置相对于节奏级水平的情况。我们的评估显示，这些启发式方法有效地区分了非表现性和表现性MIDI音轨。此外，在评估之后，我们利用NOMML启发式创建了规模最大的表现性MIDI数据集。经过挑选的GigaMIDI版本包括了由NOMML检测到的表现性表演乐器音轨，包含所有通用MIDI乐器，占GigaMIDI数据集的31%，总共1,655,649条音轨。 

---
# Data Voids and Warning Banners on Google Search 

**Title (ZH)**: Google搜索中的数据空白与警告标志 

**Authors**: Ronald E. Robertson, Evan M. Williams, Kathleen M. Carley, David Thiel  

**Link**: [PDF](https://arxiv.org/pdf/2502.17542)  

**Abstract**: The content moderation systems used by social media sites are a topic of widespread interest and research, but less is known about the use of similar systems by web search engines. For example, Google Search attempts to help its users navigate three distinct types of data voids--when the available search results are deemed low-quality, low-relevance, or rapidly-changing--by placing one of three corresponding warning banners at the top of the search page. Here we collected 1.4M unique search queries shared on social media to surface Google's warning banners, examine when and why those banners were applied, and train deep learning models to identify data voids beyond Google's classifications. Across three data collection waves (Oct 2023, Mar 2024, Sept 2024), we found that Google returned a warning banner for about 1% of our search queries, with substantial churn in the set of queries that received a banner across waves. The low-quality banners, which warn users that their results "may not have reliable information on this topic," were especially rare, and their presence was associated with low-quality domains in the search results and conspiracy-related keywords in the search query. Low-quality banner presence was also inconsistent over short time spans, even when returning highly similar search results. In August 2024, low-quality banners stopped appearing on the SERPs we collected, but average search result quality remained largely unchanged, suggesting they may have been discontinued by Google. Using our deep learning models to analyze both queries and search results in context, we identify 29 to 58 times more low-quality data voids than there were low-quality banners, and find a similar number after the banners had disappeared. Our findings point to the need for greater transparency on search engines' content moderation practices, especially around important events like elections. 

**Abstract (ZH)**: 社交媒体平台所使用的内容审核系统是广泛研究的主题，但人们对网络搜索引擎中类似系统的使用了解较为有限。例如，Google搜索通过在搜索页面顶部放置三种相应的警示横幅，旨在帮助用户应对三种不同类型的数据空洞——当搜索结果被认为是质量低、相关性差或快速变化时。在这里，我们收集了140万条在社交媒体上共享的独特搜索查询，以揭示Google的警示横幅、分析这些横幅何时及为何被应用，并训练深度学习模型以识别超出Google分类的数据空洞。在三个数据收集阶段（2023年10月、2024年3月、2024年9月），我们发现Google大约为1%的搜索查询返回了警示横幅，但接受横幅的查询集在不同阶段存在显著变化。质量低劣的横幅（警告用户其结果“可能在该主题上没有可靠的信息”）尤其罕见，其存在与搜索结果中的低质量网站以及搜索查询中的阴谋论相关关键词有关。即使返回高度相似的搜索结果，质量低劣的横幅的存在也不一致。2024年8月，收集到的SERP（搜索结果页面）上不再出现质量低劣的横幅，但平均搜索结果质量保持相对稳定，这表明Google可能已停止使用这些横幅。通过分析情境下查询和搜索结果，我们利用深度学习模型识别出比质量低劣横幅多29到58倍的质量低劣数据空洞，在横幅消失后，我们仍然发现了类似数量的数据空洞。我们的研究结果指出了在重要事件（如选举）期间，搜索引擎的内容审核实践需要更大的透明度。 

---
