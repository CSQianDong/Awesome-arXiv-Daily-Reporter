# HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation 

**Title (ZH)**: HopRAG：基于逻辑感知的多跳推理检索增强生成 

**Authors**: Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12442)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often struggle with imperfect retrieval, as traditional retrievers focus on lexical or semantic similarity rather than logical relevance. To address this, we propose HopRAG, a novel RAG framework that augments retrieval with logical reasoning through graph-structured knowledge exploration. During indexing, HopRAG constructs a passage graph, with text chunks as vertices and logical connections established via LLM-generated pseudo-queries as edges. During retrieval, it employs a retrieve-reason-prune mechanism: starting with lexically or semantically similar passages, the system explores multi-hop neighbors guided by pseudo-queries and LLM reasoning to identify truly relevant ones. Extensive experiments demonstrate HopRAG's superiority, achieving 76.78\% higher answer accuracy and 65.07\% improved retrieval F1 score compared to conventional methods. The repository is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）系统常常难以应对不完美的检索问题，因为传统的检索器主要关注词汇或语义相似性，而忽视了逻辑相关性。为了解决这一问题，我们提出了HopRAG，一种新颖的RAG框架，通过图结构的知识探索引入逻辑推理来增强检索。在索引过程中，HopRAG 构造一个段落图，其中文本片段作为顶点，通过LLM生成的伪查询建立逻辑连接，作为边。在检索过程中，系统采用“检索-推理-修剪”的机制：从词汇或语义相似的段落开始，系统根据伪查询和LLM推理逐步探索多跳邻居，以识别真正相关的内容。广泛的实验结果证明了HopRAG 的优越性，其答案准确性比传统方法高76.78%，检索F1分数提高了65.07%。代码库可访问 [该链接]。 

---
# REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark 

**Title (ZH)**: REAL-MM-RAG：一个现实世界多模态检索基准 

**Authors**: Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay, Leonid Karlinsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.12342)  

**Abstract**: Accurate multi-modal document retrieval is crucial for Retrieval-Augmented Generation (RAG), yet existing benchmarks do not fully capture real-world challenges with their current design. We introduce REAL-MM-RAG, an automatically generated benchmark designed to address four key properties essential for real-world retrieval: (i) multi-modal documents, (ii) enhanced difficulty, (iii) Realistic-RAG queries and (iv) accurate labeling. Additionally, we propose a multi-difficulty-level scheme based on query rephrasing to evaluate models' semantic understanding beyond keyword matching. Our benchmark reveals significant model weaknesses, particularly in handling table-heavy documents and robustness to query rephrasing. To mitigate these shortcomings, we curate a rephrased training set and introduce a new finance-focused, table-heavy dataset. Fine-tuning on these datasets enables models to achieve state-of-the-art retrieval performance on REAL-MM-RAG benchmark. Our work offers a better way to evaluate and improve retrieval in multi-modal RAG systems while also providing training data and models that address current limitations. 

**Abstract (ZH)**: 准确的多模态文档检索对于检索增强生成（RAG）至关重要，但现有的基准测试在当前设计中未能充分捕捉到实际应用中的挑战。我们引入了REAL-MM-RAG，这是一个自动生成的基准测试，旨在解决四个对于实际检索至关重要的关键属性：（i）多模态文档，（ii）增强的难度，（iii）现实场景下的RAG查询，（iv）准确的标注。此外，我们提出了基于查询重述的多难度层次方案，以评估模型超越关键词匹配的语义理解能力。我们的基准测试揭示了模型在应对表格密集型文档和查询重述下的鲁棒性方面的显著弱点。为了缓解这些不足，我们精心挑选了一个重述训练集，并引入了一个专注于金融且表格密集型的新数据集。在这些数据集上进行微调使模型在REAL-MM-RAG基准测试上实现了最先进的检索性能。我们的工作为评估和改进多模态RAG系统的检索提供了更好的方法，同时也提供了应对当前局限性的训练数据和模型。 

---
# Enhancing Frame Detection with Retrieval Augmented Generation 

**Title (ZH)**: 增强框架检测的检索增强生成方法 

**Authors**: Papa Abdou Karim Karou Diallo, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2502.12210)  

**Abstract**: Recent advancements in Natural Language Processing have significantly improved the extraction of structured semantic representations from unstructured text, especially through Frame Semantic Role Labeling (FSRL). Despite this progress, the potential of Retrieval-Augmented Generation (RAG) models for frame detection remains under-explored. In this paper, we present the first RAG-based approach for frame detection called RCIF (Retrieve Candidates and Identify Frames). RCIF is also the first approach to operate without the need for explicit target span and comprises three main stages: (1) generation of frame embeddings from various representations ; (2) retrieval of candidate frames given an input text; and (3) identification of the most suitable frames. We conducted extensive experiments across multiple configurations, including zero-shot, few-shot, and fine-tuning settings. Our results show that our retrieval component significantly reduces the complexity of the task by narrowing the search space thus allowing the frame identifier to refine and complete the set of candidates. Our approach achieves state-of-the-art performance on FrameNet 1.5 and 1.7, demonstrating its robustness in scenarios where only raw text is provided. Furthermore, we leverage the structured representation obtained through this method as a proxy to enhance generalization across lexical variations in the task of translating natural language questions into SPARQL queries. 

**Abstract (ZH)**: 近年来，自然语言处理（NLP）的最新进展显著提高了从非结构化文本中提取结构化语义表示的能力，尤其是通过框架语义角色标注（Frame Semantic Role Labeling, FSRL）。尽管取得了这些进展，Retrieval-Augmented Generation（RAG）模型在框架检测方面的潜力仍被低估。本文介绍了首个基于RAG的方法，即RCIF（Retrieve Candidates and Identify Frames），这是首个无需显式目标片段的方法，包含三个主要阶段：（1）从多种表示生成框架嵌入；（2）根据输入文本检索候选框架；以及（3）识别最合适的框架。我们进行了广泛的实验，包括零样本、少量样本和微调设置。实验结果表明，我们的检索组件通过缩小搜索空间显著降低了任务的复杂性，从而允许框架识别器对候选框架进行细化和补充。我们的方法在FrameNet 1.5和1.7上达到了最先进的性能，证明了在仅提供原始文本的情况下其稳健性。此外，我们利用此方法获得的结构化表示作为代理，增强在将自然语言问题翻译为SPARQL查询任务中跨词汇变体的一般化能力。 

---
