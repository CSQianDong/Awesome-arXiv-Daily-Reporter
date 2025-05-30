# TrustRAG: An Information Assistant with Retrieval Augmented Generation 

**Title (ZH)**: TrustRAG：一种检索增强生成的信息助手 

**Authors**: Yixing Fan, Qiang Yan, Wenshan Wang, Jiafeng Guo, Ruqing Zhang, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13719)  

**Abstract**: \Ac{RAG} has emerged as a crucial technique for enhancing large models with real-time and domain-specific knowledge. While numerous improvements and open-source tools have been proposed to refine the \ac{RAG} framework for accuracy, relatively little attention has been given to improving the trustworthiness of generated results. To address this gap, we introduce TrustRAG, a novel framework that enhances \ac{RAG} from three perspectives: indexing, retrieval, and generation. Specifically, in the indexing stage, we propose a semantic-enhanced chunking strategy that incorporates hierarchical indexing to supplement each chunk with contextual information, ensuring semantic completeness. In the retrieval stage, we introduce a utility-based filtering mechanism to identify high-quality information, supporting answer generation while reducing input length. In the generation stage, we propose fine-grained citation enhancement, which detects opinion-bearing sentences in responses and infers citation relationships at the sentence-level, thereby improving citation accuracy. We open-source the TrustRAG framework and provide a demonstration studio designed for excerpt-based question answering tasks \footnote{this https URL}. Based on these, we aim to help researchers: 1) systematically enhancing the trustworthiness of \ac{RAG} systems and (2) developing their own \ac{RAG} systems with more reliable outputs. 

**Abstract (ZH)**: \Ac{RAG} 已成为增强大型模型实时和领域特定知识的关键技术。尽管已经提出了许多改进和开源工具来提高 \ac{RAG} 框架的准确度，但生成结果的信任度改进则相对较少受到关注。为弥补这一不足，我们引入了 TrustRAG，这是一种从三个角度增强 \ac{RAG} 的新型框架：索引、检索和生成。具体而言，在索引阶段，我们提出了一种语义增强的分块策略，结合了层次索引，以补充每个分块的上下文信息，确保语义完整；在检索阶段，我们引入了一种基于效用的过滤机制，以识别高质量的信息，支持答案生成并减少输入长度；在生成阶段，我们提出了细粒度的引文增强策略，检测响应中的主观陈述句子，并在句级推断引文关系，从而提高引文准确性。我们开源了 TrustRAG 框架，并提供了一个用于节选问答任务的演示工作室\footnote{this https URL}。基于这些改进，我们希望帮助研究人员：1）系统地增强 \ac{RAG} 系统的信任度；2）开发具有更可靠输出的自定义 \ac{RAG} 系统。 

---
# SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering? 

**Title (ZH)**: SearchRAG：搜索引擎对基于LLM的医疗问答有帮助吗？ 

**Authors**: Yucheng Shi, Tianze Yang, Canyu Chen, Quanzheng Li, Tianming Liu, Xiang Li, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13233)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in general domains but often struggle with tasks requiring specialized knowledge. Conventional Retrieval-Augmented Generation (RAG) techniques typically retrieve external information from static knowledge bases, which can be outdated or incomplete, missing fine-grained clinical details essential for accurate medical question answering. In this work, we propose SearchRAG, a novel framework that overcomes these limitations by leveraging real-time search engines. Our method employs synthetic query generation to convert complex medical questions into search-engine-friendly queries and utilizes uncertainty-based knowledge selection to filter and incorporate the most relevant and informative medical knowledge into the LLM's input. Experimental results demonstrate that our method significantly improves response accuracy in medical question answering tasks, particularly for complex questions requiring detailed and up-to-date knowledge. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在通用领域展现了出色的性能，但在需要专业知识的任务上往往表现不佳。传统的检索增强生成（RAG）技术通常从静态知识库中检索外部信息，这些知识库可能过时或不完整，缺乏准确回答医学问题所必需的细粒度临床细节。本研究中，我们提出了一种名为SearchRAG的新框架，该框架通过利用实时搜索引擎来克服这些局限性。该方法通过合成查询生成将复杂的医学问题转换为搜索引擎友好的查询，并通过基于不确定性的知识选择来筛选和整合与LLM输入最相关的、最有信息价值的医学知识。实验结果表明，该方法显著提高了在医学问答任务中的响应准确性，尤其是在需要详细和最新知识的复杂问题上。 

---
# RAG-Gym: Optimizing Reasoning and Search Agents with Process Supervision 

**Title (ZH)**: RAG-Gym：通过过程监督优化推理和搜索代理 

**Authors**: Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang, Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing Song, Dengyu Wang, Minjia Zhang, Zhiyong Lu, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13957)  

**Abstract**: Retrieval-augmented generation (RAG) has shown great potential for knowledge-intensive tasks, but its traditional architectures rely on static retrieval, limiting their effectiveness for complex questions that require sequential information-seeking. While agentic reasoning and search offer a more adaptive approach, most existing methods depend heavily on prompt engineering. In this work, we introduce RAG-Gym, a unified optimization framework that enhances information-seeking agents through fine-grained process supervision at each search step. We also propose ReSearch, a novel agent architecture that synergizes answer reasoning and search query generation within the RAG-Gym framework. Experiments on four challenging datasets show that RAG-Gym improves performance by up to 25.6\% across various agent architectures, with ReSearch consistently outperforming existing baselines. Further analysis highlights the effectiveness of advanced LLMs as process reward judges and the transferability of trained reward models as verifiers for different LLMs. Additionally, we examine the scaling properties of training and inference in agentic RAG. The project homepage is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）在知识密集型任务中展现出了巨大的潜力，但其传统的架构依赖于静态检索，这限制了其在需要顺序信息搜索的复杂问题上的有效性。虽然主动推理和搜索提供了更具适应性的方法，但大多数现有方法仍然高度依赖于提示工程。在本工作中，我们引入了RAG-Gym，这是一种统一的优化框架，通过在每个检索步骤中提供细致的过程监督来增强信息查询代理。我们还提出了ReSearch，这是一种新的代理架构，在RAG-Gym框架中结合了答案推理和搜索查询生成。在四个具有挑战性的数据集中进行的实验表明，RAG-Gym在各种代理架构中提高了高达25.6%的性能，而ReSearch在所有基线中始终保持优异表现。进一步的分析强调了高级语言模型作为过程奖励裁判的有效性，并展示了训练好的奖励模型在不同语言模型中作为验证者的可迁移性。此外，我们还研究了主动RAG的训练和推理的扩展性能。该项目的主页可通过以下链接访问：[这是一个网址]。 

---
# DH-RAG: A Dynamic Historical Context-Powered Retrieval-Augmented Generation Method for Multi-Turn Dialogue 

**Title (ZH)**: DH-RAG：一种基于动态历史语境的检索增强生成方法用于多轮对话 

**Authors**: Feiyuan Zhang, Dezhi Zhu, James Ming, Yilun Jin, Di Chai, Liu Yang, Han Tian, Zhaoxin Fan, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13847)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have shown substantial benefits in applications such as question answering and multi-turn dialogue \citep{lewis2020retrieval}. However, traditional RAG methods, while leveraging static knowledge bases, often overlook the potential of dynamic historical information in ongoing conversations. To bridge this gap, we introduce DH-RAG, a Dynamic Historical Context-Powered Retrieval-Augmented Generation Method for Multi-Turn Dialogue. DH-RAG is inspired by human cognitive processes that utilize both long-term memory and immediate historical context in conversational responses \citep{stafford1987conversational}. DH-RAG is structured around two principal components: a History-Learning based Query Reconstruction Module, designed to generate effective queries by synthesizing current and prior interactions, and a Dynamic History Information Updating Module, which continually refreshes historical context throughout the dialogue. The center of DH-RAG is a Dynamic Historical Information database, which is further refined by three strategies within the Query Reconstruction Module: Historical Query Clustering, Hierarchical Matching, and Chain of Thought Tracking. Experimental evaluations show that DH-RAG significantly surpasses conventional models on several benchmarks, enhancing response relevance, coherence, and dialogue quality. 

**Abstract (ZH)**: 检索增强生成（RAG）系统在问答和多轮对话等多种应用中展现了显著的优势 \citep{lewis2020retrieval}。然而，传统的RAG方法虽然利用了静态的知识库，但在处理正在进行的对话时往往忽视了动态历史信息的潜力。为弥补这一差距，我们提出了一种名为DH-RAG的方法，该方法是基于动态历史上下文的检索增强生成方法，适用于多轮对话。DH-RAG受到人类认知过程的启发，该过程在对话回应中综合利用了长期记忆和即时历史信息 \citep{stafford1987conversational}。DH-RAG结构上由两个主要部分组成：基于历史学习的查询重构模块和动态历史信息更新模块。基于历史学习的查询重构模块旨在通过综合当前和先前的交互生成有效的查询，而动态历史信息更新模块则在整个对话过程中不断更新历史上下文。DH-RAG的核心是一个动态历史信息数据库，该数据库在查询重构模块的三个策略的细化下进一步优化：历史查询聚类、层次匹配和思路跟踪。实验评估显示，DH-RAG在多个基准测试中显著优于传统模型，提高了回应的相关性、连贯性和对话质量。 

---
# Are Large Language Models In-Context Graph Learners? 

**Title (ZH)**: 大型语言模型是上下文图学习者吗？ 

**Authors**: Jintang Li, Ruofan Wu, Yuchang Zhu, Huizhe Zhang, Liang Chen, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13562)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable in-context reasoning capabilities across a wide range of tasks, particularly with unstructured inputs such as language or images. However, LLMs struggle to handle structured data, such as graphs, due to their lack of understanding of non-Euclidean structures. As a result, without additional fine-tuning, their performance significantly lags behind that of graph neural networks (GNNs) in graph learning tasks. In this paper, we show that learning on graph data can be conceptualized as a retrieval-augmented generation (RAG) process, where specific instances (e.g., nodes or edges) act as queries, and the graph itself serves as the retrieved context. Building on this insight, we propose a series of RAG frameworks to enhance the in-context learning capabilities of LLMs for graph learning tasks. Comprehensive evaluations demonstrate that our proposed RAG frameworks significantly improve LLM performance on graph-based tasks, particularly in scenarios where a pretrained LLM must be used without modification or accessed via an API. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展示了令人印象深刻的上下文推理能力，尤其是在处理未结构化的输入（如语言或图像）方面。然而，LLMs 在处理结构化数据（如图）方面表现不佳，这主要是由于它们无法理解非欧几里得结构。因此，在不需要额外微调的情况下，它们在图学习任务中的表现远远落后于图神经网络（GNNs）。本文中，我们表明在图数据上进行学习可以被视为一种检索增强生成（RAG）过程，其中特定实例（例如节点或边）作为查询，而图本身则作为检索的上下文。基于这一见解，我们提出了一系列 RAG 框架，以增强 LLMs 在图学习任务中的上下文学习能力。全面的评估结果表明，我们提出的 RAG 框架显著提升了 LLMs 在基于图的任务中的性能，特别是在需要使用未修改的预训练 LLM 或通过 API 访问的情景中。 

---
# RGAR: Recurrence Generation-augmented Retrieval for Factual-aware Medical Question Answering 

**Title (ZH)**: RGAR：基于循环生成增强检索的医学事实导向问答方法 

**Authors**: Sichu Liang, Linhai Zhang, Hongyu Zhu, Wenwen Wang, Yulan He, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.13361)  

**Abstract**: Medical question answering requires extensive access to specialized conceptual knowledge. The current paradigm, Retrieval-Augmented Generation (RAG), acquires expertise medical knowledge through large-scale corpus retrieval and uses this knowledge to guide a general-purpose large language model (LLM) for generating answers. However, existing retrieval approaches often overlook the importance of factual knowledge, which limits the relevance of retrieved conceptual knowledge and restricts its applicability in real-world scenarios, such as clinical decision-making based on Electronic Health Records (EHRs). This paper introduces RGAR, a recurrence generation-augmented retrieval framework that retrieves both relevant factual and conceptual knowledge from dual sources (i.e., EHRs and the corpus), allowing them to interact and refine each another. Through extensive evaluation across three factual-aware medical question answering benchmarks, RGAR establishes a new state-of-the-art performance among medical RAG systems. Notably, the Llama-3.1-8B-Instruct model with RGAR surpasses the considerably larger, RAG-enhanced GPT-3.5. Our findings demonstrate the benefit of extracting factual knowledge for retrieval, which consistently yields improved generation quality. 

**Abstract (ZH)**: 医疗问答需要广泛访问专业概念知识。当前的范式，检索增强生成（RAG），通过大规模语料库检索获得专业知识，并利用这些知识来引导通用的大规模语言模型（LLM）生成答案。然而，现有的检索方法往往忽视了事实性知识的重要性，这限制了检索到的概念性知识的相关性，从而限制了其在实际场景中的应用，例如基于电子健康记录（EHR）的临床决策。本文介绍了RGAR，这是一种循环生成增强的检索框架，可以从双重来源（即EHR和语料库）检索相关事实性与概念性知识，并使其相互作用和相互完善。通过在三个事实性意识医疗问答基准上进行广泛的评估，RGAR在医疗RAG系统中确立了新的最先进的性能。值得注意的是，配备RGAR的Llama-3.1-8B-Instruct模型超越了更大且经过RAG增强的GPT-3.5模型。我们的研究结果表明，提取事实性知识对于检索的好处，这在一致性上提高了生成质量。 

---
