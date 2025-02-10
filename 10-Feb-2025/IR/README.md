# Holistically Guided Monte Carlo Tree Search for Intricate Information Seeking 

**Title (ZH)**: 全面引导的蒙特卡洛树搜索方法用于复杂的信息检索 

**Authors**: Ruiyang Ren, Yuhao Wang, Junyi Li, Jinhao Jiang, Wayne Xin Zhao, Wenjie Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2502.04751)  

**Abstract**: In the era of vast digital information, the sheer volume and heterogeneity of available information present significant challenges for intricate information seeking. Users frequently face multistep web search tasks that involve navigating vast and varied data sources. This complexity demands every step remains comprehensive, accurate, and relevant. However, traditional search methods often struggle to balance the need for localized precision with the broader context required for holistic understanding, leaving critical facets of intricate queries underexplored. In this paper, we introduce an LLM-based search assistant that adopts a new information seeking paradigm with holistically guided Monte Carlo tree search (HG-MCTS). We reformulate the task as a progressive information collection process with a knowledge memory and unite an adaptive checklist with multi-perspective reward modeling in MCTS. The adaptive checklist provides explicit sub-goals to guide the MCTS process toward comprehensive coverage of complex user queries. Simultaneously, our multi-perspective reward modeling offers both exploration and retrieval rewards, along with progress feedback that tracks completed and remaining sub-goals, refining the checklist as the tree search progresses. By striking a balance between localized tree expansion and global guidance, HG-MCTS reduces redundancy in search paths and ensures that all crucial aspects of an intricate query are properly addressed. Extensive experiments on real-world intricate information seeking tasks demonstrate that HG-MCTS acquires thorough knowledge collections and delivers more accurate final responses compared with existing baselines. 

**Abstract (ZH)**: 在大数据信息时代，大量且异构的信息资源给复杂的检索工作带来了重大挑战。用户经常面临多步骤的网络搜索任务，需要在海量且多样化的数据源中进行导航。这种复杂性要求每一个搜索步骤都必须全面、准确且相关。然而，传统的搜索方法常常难以平衡局部精确性与整体理解所需的广泛背景之间的需求，导致复杂的查询中关键方面被忽视。在本文中，我们提出了一种基于大规模语言模型（LLM）的搜索助手，采用了一种新的以整体引导的蒙特卡洛树搜索（HG-MCTS）为特征的信息检索范式。我们将任务重新构想为一个渐进的信息收集过程，并结合知识记忆、自适应检查列表和多视角奖励建模来统一在MCTS中的应用。自适应检查列表提供明确的子目标来引导MCTS过程，以全面覆盖复杂的用户查询。同时，我们的多视角奖励建模提供了探索和检索奖励，并提供进度反馈以跟踪已完成和剩余的子目标，随着树搜索的进行逐步优化检查列表。通过平衡局部树扩展与全局指导，HG-MCTS减少了搜索路径的冗余，并确保所有关键方面都能得到妥善处理。在实际复杂的检索任务上的广泛实验表明，HG-MCTS能够获取更全面的知识集，并提供比现有基线更准确的最终响应。 

---
# Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy 

**Title (ZH)**: 通过优先考虑主题相关性和事实准确性以增强健康信息检索的RAG方法 

**Authors**: Rishabh Uapadhyay, Marco Viviani  

**Link**: [PDF](https://arxiv.org/pdf/2502.04666)  

**Abstract**: The exponential surge in online health information, coupled with its increasing use by non-experts, highlights the pressing need for advanced Health Information Retrieval models that consider not only topical relevance but also the factual accuracy of the retrieved information, given the potential risks associated with health misinformation. To this aim, this paper introduces a solution driven by Retrieval-Augmented Generation (RAG), which leverages the capabilities of generative Large Language Models (LLMs) to enhance the retrieval of health-related documents grounded in scientific evidence. In particular, we propose a three-stage model: in the first stage, the user's query is employed to retrieve topically relevant passages with associated references from a knowledge base constituted by scientific literature. In the second stage, these passages, alongside the initial query, are processed by LLMs to generate a contextually relevant rich text (GenText). In the last stage, the documents to be retrieved are evaluated and ranked both from the point of view of topical relevance and factual accuracy by means of their comparison with GenText, either through stance detection or semantic similarity. In addition to calculating factual accuracy, GenText can offer a layer of explainability for it, aiding users in understanding the reasoning behind the retrieval. Experimental evaluation of our model on benchmark datasets and against baseline models demonstrates its effectiveness in enhancing the retrieval of both topically relevant and factually accurate health information, thus presenting a significant step forward in the health misinformation mitigation problem. 

**Abstract (ZH)**: 互联网健康信息的指数级增长，以及其被非专业人士广泛应用的趋势，突显了迫切需要发展高级的健康信息检索模型的重要性。这些模型不仅要考虑检索信息的相关性，还要确保信息的准确性，因为健康错误信息的风险隐患不容忽视。为了应对这一挑战，本文提出了一种基于检索增强生成（RAG）方法的解决方案，利用生成型大规模语言模型（LLMs）的能力来提升基于科学研究证据的健康相关文档的检索质量。具体而言，我们提出了一种三阶段模型：首先，利用用户的查询从科学文献构成的知识库中检索相关段落及其引用文献；其次，这些段落与初始查询一起，通过LLMs生成上下文相关的内容丰富文本（GenText）；最后，通过与GenText的比较，无论是通过立场检测还是语义相似性，评估并排序待检索的文档，从相关性和准确性的角度进行综合考量。除了计算事实准确性外，GenText还可以提供一层解释性，帮助用户理解检索背后的理由。我们模型在基准数据集上的实验评估和与基线模型的对比表明，它在增强Topically Relevant和Factually Accurate健康信息检索方面具有明显优势，从而在健康错误信息防控问题上迈出了重要一步。 

---
# Cross-Encoder Rediscovers a Semantic Variant of BM25 

**Title (ZH)**: 交叉编码器重新发现BM25的语义变体 

**Authors**: Meng Lu, Catherine Chen, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2502.04645)  

**Abstract**: Neural Ranking Models (NRMs) have rapidly advanced state-of-the-art performance on information retrieval tasks. In this work, we investigate a Cross-Encoder variant of MiniLM to determine which relevance features it computes and where they are stored. We find that it employs a semantic variant of the traditional BM25 in an interpretable manner, featuring localized components: (1) Transformer attention heads that compute soft term frequency while controlling for term saturation and document length effects, and (2) a low-rank component of its embedding matrix that encodes inverse document frequency information for the vocabulary. This suggests that the Cross-Encoder uses the same fundamental mechanisms as BM25, but further leverages their capacity to capture semantics for improved retrieval performance. The granular understanding lays the groundwork for model editing to enhance model transparency, addressing safety concerns, and improving scalability in training and real-world applications. 

**Abstract (ZH)**: 神经排序模型（Neural Ranking Models, NRMs）在信息检索任务上的性能已迅速超越了现有最佳水平。本研究中，我们探讨了MiniLM的交叉编码器变体，以确定该模型计算哪些相关性特征以及这些特征的具体存储位置。我们发现，它以可解释的方式使用了一种语义变体的传统BM25方法，包括局部组件：（1）transformer自注意力头，用于计算软词频的同时控制词频饱和度和文档长度的影响，以及（2）其嵌入矩阵的一个低秩分量，用于编码词汇的逆文档频率信息。这表明，交叉编码器使用了与BM25相同的基本机制，但进一步利用了其捕捉语义的能力，以提高检索性能。深度理解这些机制为模型的编辑、提升模型透明度、解决安全问题以及改善训练和实际应用中的可扩展性奠定了基础。 

---
# MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot 

**Title (ZH)**: 医卫伴侣增强版：基于知识图谱启发推理的检索增强生成技术（MedRAG） 

**Authors**: Xuejiao Zhao, Siyan Liu, Su-Yin Yang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2502.04413)  

**Abstract**: Retrieval-augmented generation (RAG) is a well-suited technique for retrieving privacy-sensitive Electronic Health Records (EHR). It can serve as a key module of the healthcare copilot, helping reduce misdiagnosis for healthcare practitioners and patients. However, the diagnostic accuracy and specificity of existing heuristic-based RAG models used in the medical domain are inadequate, particularly for diseases with similar manifestations. This paper proposes MedRAG, a RAG model enhanced by knowledge graph (KG)-elicited reasoning for the medical domain that retrieves diagnosis and treatment recommendations based on manifestations. MedRAG systematically constructs a comprehensive four-tier hierarchical diagnostic KG encompassing critical diagnostic differences of various diseases. These differences are dynamically integrated with similar EHRs retrieved from an EHR database, and reasoned within a large language model. This process enables more accurate and specific decision support, while also proactively providing follow-up questions to enhance personalized medical decision-making. MedRAG is evaluated on both a public dataset DDXPlus and a private chronic pain diagnostic dataset (CPDD) collected from Tan Tock Seng Hospital, and its performance is compared against various existing RAG methods. Experimental results show that, leveraging the information integration and relational abilities of the KG, our MedRAG provides more specific diagnostic insights and outperforms state-of-the-art models in reducing misdiagnosis rates. Our code will be available at this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）是一种适用于检索敏感电子健康记录（EHR）的技术，非常适合医疗领域的应用。它可以作为医疗copilot的关键模块，帮助减少医疗工作者和患者的误诊。然而，现有基于启发式的医疗领域的RAG模型在疾病表现相似的情况下诊断准确性和特异性不足。本文提出了一种名为MedRAG的模型，该模型通过知识图谱（KG）驱动的推理增强了RAG技术，用于基于症状检索诊断和治疗建议。MedRAG系统性地构建了一个包含各种疾病关键诊断差异的多层次诊断KG。这些差异与从EHR数据库中检索到的相似EHR动态集成，并在大型语言模型中进行推理。这一过程不仅使诊断决策支持更为准确和具体，还能主动提供后续问题以增强个性化的医疗决策。MedRAG在公共数据集DDXPlus和Tan Tock Seng医院收集的慢性疼痛诊断私有数据集（CPDD）上进行了评估，并将其性能与多种现有RAG方法进行了比较。实验结果表明，通过利用知识图谱的信息整合和关系处理能力，MedRAG提供了更具体的诊断见解，并在降低误诊率方面优于现有最先进的模型。我们的代码将在以下网址提供：[提供链接的网址] 

---
