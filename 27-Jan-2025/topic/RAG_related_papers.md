# Chain-of-Retrieval Augmented Generation 

**Title (ZH)**: 链式检索增强生成 

**Authors**: Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.14342)  

**Abstract**: This paper introduces an approach for training o1-like RAG models that retrieve and reason over relevant information step by step before generating the final answer. Conventional RAG methods usually perform a single retrieval step before the generation process, which limits their effectiveness in addressing complex queries due to imperfect retrieval results. In contrast, our proposed method, CoRAG (Chain-of-Retrieval Augmented Generation), allows the model to dynamically reformulate the query based on the evolving state. To train CoRAG effectively, we utilize rejection sampling to automatically generate intermediate retrieval chains, thereby augmenting existing RAG datasets that only provide the correct final answer. At test time, we propose various decoding strategies to scale the model's test-time compute by controlling the length and number of sampled retrieval chains. Experimental results across multiple benchmarks validate the efficacy of CoRAG, particularly in multi-hop question answering tasks, where we observe more than 10 points improvement in EM score compared to strong baselines. On the KILT benchmark, CoRAG establishes a new state-of-the-art performance across a diverse range of knowledge-intensive tasks. Furthermore, we offer comprehensive analyses to understand the scaling behavior of CoRAG, laying the groundwork for future research aimed at developing factual and grounded foundation models. 

**Abstract (ZH)**: 本文介绍了一种训练O1-like RAG模型的方法，该方法在生成最终答案之前，逐步检索和推理相关的信息。传统的RAG方法通常在生成过程之前只进行一次检索步骤，这限制了它们在处理复杂查询时的有效性，尤其是在检索结果不够完善的情况下。相比之下，我们提出的方法CoRAG（Chain-of-Retrieval Augmented Generation）允许模型根据检索状态的演变动态重新构建查询。为了有效地训练CoRAG，我们利用拒绝采样自动生成中间检索链，从而增强现有仅提供正确最终答案的RAG数据集。在测试阶段，我们提出了多种解码策略，通过控制检索链的长度和数量来扩展模型的测试计算能力。在多个基准测试中的实验结果验证了CoRAG的有效性，特别是在多跳问答任务中，观察到EM分数相对于强基线模型有超过10点的改进。在KILT基准测试中，CoRAG在各种知识密集型任务中取得了新的最佳性能。此外，我们还提供了全面的分析，以理解CoRAG的扩展行为，为未来旨在开发事实性和实际 grounding 的基础模型的研究奠定了基础。 

---
# CAPRAG: A Large Language Model Solution for Customer Service and Automatic Reporting using Vector and Graph Retrieval-Augmented Generation 

**Title (ZH)**: CAPRAG：一种基于向量和图检索增强生成的大型语言模型解决方案，用于客户服务和自动报告 

**Authors**: Hamza Landolsi, Kais Letaief, Nizar Taghouti, Ines Abdeljaoued-Tej  

**Link**: [PDF](https://arxiv.org/pdf/2501.13993)  

**Abstract**: The introduction of new features and services in the banking sector often overwhelms customers, creating an opportunity for banks to enhance user experience through financial chatbots powered by large language models (LLMs). We initiated an AI agent designed to provide customers with relevant information about banking services and insights from annual reports. We proposed a hybrid Customer Analysis Pipeline Retrieval-Augmented Generation (CAPRAG) that effectively addresses both relationship-based and contextual queries, thereby improving customer engagement in the digital banking landscape. To implement this, we developed a processing pipeline to refine text data, which we utilized in two main frameworks: Vector RAG and Graph RAG. This dual approach enables us to populate both vector and graph databases with processed data for efficient retrieval. The Cypher query component is employed to effectively query the graph database. When a user submits a query, it is first expanded by a query expansion module before being routed to construct a final query from the hybrid Knowledge Base (KB). This final query is then sent to an open-source LLM for response generation. Overall, our innovative, designed to international banks, serves bank's customers in an increasingly complex digital environment, enhancing clarity and accessibility of information. 

**Abstract (ZH)**: 银行部门引入新功能和服务往往会让客户感到不知所措，为银行通过大型语言模型（LLMs）驱动的金融聊天机器人提升用户体验提供了机会。我们启动了一个AI代理，旨在为客户提供与银行服务相关的信息以及年度报告的见解。我们提出了一种混合客户分析管道检索-增强生成（CAPRAG）方法，该方法有效地处理了基于关系和上下文的查询，从而在数字银行领域中提高了客户服务的参与度。为了实现这一目标，我们开发了一个处理管道来精炼文本数据，并将其应用于两个主要框架：向量RAG和图RAG。这种双重方法使我们能够将处理后的数据填充到向量和图数据库中，以实现高效的检索。我们使用Cypher查询组件来有效地查询图数据库。当用户提交查询时，该查询首先由查询扩展模块扩展，然后路由到构建最终查询的混合知识库（KB）。该最终查询随后发送给开源LLM以生成回应。总体而言，我们创新的解决方案旨在为国际银行的客户提供一个日益复杂的数字环境中的帮助，从而提高信息的清晰度和可访问性。 

---
# A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models 

**Title (ZH)**: 面向定制化大规模语言模型的图检索增强生成综述 

**Authors**: Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13958)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in a wide range of tasks, yet their application to specialized domains remains challenging due to the need for deep expertise. Retrieval-augmented generation (RAG) has emerged as a promising solution to customize LLMs for professional fields by seamlessly integrating external knowledge bases, enabling real-time access to domain-specific expertise during inference. Despite its potential, traditional RAG systems, based on flat text retrieval, face three critical challenges: (i) complex query understanding in professional contexts, (ii) difficulties in knowledge integration across distributed sources, and (iii) system efficiency bottlenecks at scale. This survey presents a systematic analysis of Graph-based Retrieval-Augmented Generation (GraphRAG), a new paradigm that revolutionizes domain-specific LLM applications. GraphRAG addresses traditional RAG limitations through three key innovations: (i) graph-structured knowledge representation that explicitly captures entity relationships and domain hierarchies, (ii) efficient graph-based retrieval techniques that enable context-preserving knowledge retrieval with multihop reasoning ability, and (iii) structure-aware knowledge integration algorithms that leverage retrieved knowledge for accurate and logical coherent generation of LLMs. In this survey, we systematically analyze the technical foundations of GraphRAG and examine current implementations across various professional domains, identifying key technical challenges and promising research directions. All the related resources of GraphRAG, including research papers, open-source data, and projects, are collected for the community in \textcolor{blue}{\url{this https URL}}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展现了惊人的能力，然而将其应用于专业化领域仍然具有挑战性，因为需要深厚的专业知识。检索增强生成（RAG）作为一种前景广阔的方法，通过无缝集成外部知识库，实现在推断过程中对领域特定专业知识的实时访问，从而为专业领域量身定制LLMs。尽管具有潜力，传统RAG系统基于扁平文本检索，面临三个关键挑战：（i）专业语境下的复杂查询理解，（ii）跨分布式来源的知识整合难题，以及（iii）系统在大规模应用中的效率瓶颈。本文综述了图检索增强生成（GraphRAG）这一新范式，该范式通过三种关键创新解决了传统RAG的局限性：（i）以图结构表示知识并明确捕捉实体关系和领域层次结构，（ii）高效的基于图的检索技术，能够支持上下文保持的知识检索和多跳推理能力，以及（iii）结构感知的知识整合算法，利用检索到的知识为LLMs生成准确且合乎逻辑的内容。在本文综述中，我们系统地分析了GraphRAG的技术基础，并考察了其在各种专业领域的现有实现，识别出关键的技术挑战和有前景的研究方向。所有与GraphRAG相关的资源，包括研究论文、开源数据和项目，已收集于此处 \textcolor{blue}{\[https://example.com\]}，供社区参考。 

---
# Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents 

**Title (ZH)**: Chat3GPP：一个开源检索增强生成框架用于3GPP文档 

**Authors**: Long Huang, Ming Zhao, Limin Xiao, Xiujun Zhang, Jungang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13954)  

**Abstract**: The 3rd Generation Partnership Project (3GPP) documents is key standards in global telecommunications, while posing significant challenges for engineers and researchers in the telecommunications field due to the large volume and complexity of their contents as well as the frequent updates. Large language models (LLMs) have shown promise in natural language processing tasks, but their general-purpose nature limits their effectiveness in specific domains like telecommunications. To address this, we propose Chat3GPP, an open-source retrieval-augmented generation (RAG) framework tailored for 3GPP specifications. By combining chunking strategies, hybrid retrieval and efficient indexing methods, Chat3GPP can efficiently retrieve relevant information and generate accurate responses to user queries without requiring domain-specific fine-tuning, which is both flexible and scalable, offering significant potential for adapting to other technical standards beyond 3GPP. We evaluate Chat3GPP on two telecom-specific datasets and demonstrate its superior performance compared to existing methods, showcasing its potential for downstream tasks like protocol generation and code automation. 

**Abstract (ZH)**: 第三代合作伙伴项目（3GPP）文档是全球电信领域的重要标准，但由于其内容巨大且复杂，以及频繁的更新，给电信领域的工程师和研究人员带来了显著的挑战。大规模语言模型（LLMs）在自然语言处理任务中展现出了巨大潜力，但其通用性质限制了其在特定领域如电信中的有效性。为了解决这一问题，我们提出了一种名为Chat3GPP的开源检索增强生成（RAG）框架，专门针对3GPP规范。通过结合分块策略、混合检索和高效的索引方法，Chat3GPP能够高效地检索相关信息，并生成准确的用户查询回答，而无需特定领域的微调。这种框架既灵活又可扩展，为适应其他技术标准提供了显著的潜力。我们对Chat3GPP进行了两种电信特定数据集的评估，并展示了其在现有方法中的优越性能，证明了其在协议生成和代码自动化等下游任务中的潜在应用价值。 

---
# RELexED: Retrieval-Enhanced Legal Summarization with Exemplar Diversity 

**Title (ZH)**: RELexED：基于范例多样性增强的检索增强法律摘要生成 

**Authors**: T.Y.S.S. Santosh, Chen Jia, Patrick Goroncy, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2501.14113)  

**Abstract**: This paper addresses the task of legal summarization, which involves distilling complex legal documents into concise, coherent summaries. Current approaches often struggle with content theme deviation and inconsistent writing styles due to their reliance solely on source documents. We propose RELexED, a retrieval-augmented framework that utilizes exemplar summaries along with the source document to guide the model. RELexED employs a two-stage exemplar selection strategy, leveraging a determinantal point process to balance the trade-off between similarity of exemplars to the query and diversity among exemplars, with scores computed via influence functions. Experimental results on two legal summarization datasets demonstrate that RELexED significantly outperforms models that do not utilize exemplars and those that rely solely on similarity-based exemplar selection. 

**Abstract (ZH)**: 本文探讨了法律总结的任务，即将复杂的法律文件提炼为简洁且连贯的摘要。现有的方法往往因为仅依赖原始文档而面临内容主题偏移和不一致写作风格的问题。我们提出了一种名为RELexED的检索增强框架，该框架利用示例摘要和原始文档来指导模型。RELexED采用两阶段的示例选择策略，利用确定性点过程在示例与查询之间的相似性和示例之间的多样性之间进行权衡，并通过影响函数计算得分。在两个法律总结数据集上的实验结果表明，RELexED在不使用示例和仅依赖基于相似性的示例选择的模型中表现显著更好。 

---
# GraphRAG under Fire 

**Title (ZH)**: 《GraphRAG受考验》

这个翻译在保持原意的同时，尽量符合学术规范的表达方式。如果需要更加具体的上下文或是有不同的表达偏好，请提供更多信息。 

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14050)  

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their reasoning. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: compared to conventional RAG, GraphRAG's graph-based indexing and retrieval enhance resilience against simple poisoning attacks; meanwhile, the same features also create new attack surfaces. We present GRAGPoison, a novel attack that exploits shared relations in the knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GRAGPoison employs three key strategies: i) relation injection to introduce false knowledge, ii) relation enhancement to amplify poisoning influence, and iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GRAGPoison substantially outperforms existing attacks in terms of effectiveness (up to 98% success rate) and scalability (using less than 68% poisoning text). We also explore potential defensive measures and their limitations, identifying promising directions for future research. 

**Abstract (ZH)**: GraphRAG通过将外部知识结构化为多尺度知识图谱，增强了检索增强生成（RAG）的能力，使语言模型在其推理过程中能够整合广泛的语境和细微的细节。尽管GraphRAG在多个领域已显示出成功，但其安全性影响仍很大程度上未被探索。为解决这一问题，本研究考察了GraphRAG对抗中毒攻击的脆弱性，并揭示了一个有趣的安全悖论：与传统的RAG相比，GraphRAG基于图的索引和检索增强了对简单中毒攻击的抵抗力；同时，相同的特征也创设了新的攻击途径。我们提出了一种名为GRAGPoison的新颖攻击方法，利用知识图谱中的共享关系来构建能够同时破坏多个查询的中毒文本。GRAGPoison采用三种关键策略：（i）关系注入以引入虚假知识，（ii）关系增强以放大中毒影响，（iii）叙事生成以在连贯文本中嵌入恶意内容。跨多个数据集和模型的实证评估显示，GRAGPoison在有效性（高达98%的成功率）和可扩展性（使用不到68%的中毒文本）方面远超现有攻击。我们还探索了潜在的防御措施及其局限性，并指出了未来研究的积极方向。 

---
