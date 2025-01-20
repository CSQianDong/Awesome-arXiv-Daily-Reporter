# AirRAG: Activating Intrinsic Reasoning for Retrieval Augmented Generation via Tree-based Search 

**Title (ZH)**: AirRAG：通过树状搜索激活内在推理以增强检索生成

这个翻译符合学术规范，保留了原文的核心概念和结构。希望这对你有帮助！如果有更具体的内容需要翻译或进一步的帮助，请告诉我。 

**Authors**: Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Jingyi Song, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10053)  

**Abstract**: Leveraging the autonomous decision-making capabilities of large language models (LLMs) demonstrates superior performance in reasoning tasks. Despite the successes of iterative or recursive retrieval-augmented generation (RAG), they often are trapped in a single solution space when confronted with complex tasks. In this paper, we propose a novel thinking pattern in RAG which integrates system analysis with efficient reasoning actions, significantly activating intrinsic reasoning capabilities and expanding the solution space of specific tasks via Monte Carlo Tree Search (MCTS), dubbed AirRAG. Specifically, our approach designs five fundamental reasoning actions that are expanded to a wide tree-based reasoning spaces using MCTS. The extension also uses self-consistency verification to explore potential reasoning paths and implement inference scaling. In addition, computationally optimal strategies are used to apply more inference computation to key actions to achieve further performance improvements. Experimental results demonstrate the effectiveness of AirRAG through considerable performance gains over complex QA datasets. Furthermore, AirRAG is flexible and lightweight, making it easy to integrate with other advanced technologies. 

**Abstract (ZH)**: 利用大型语言模型（LLMs）的自主决策能力在推理任务中展现了优异的表现。尽管迭代或递归检索增强生成（RAG）方法取得了成功，但在面对复杂任务时，它们往往受限于单一的解决方案空间。本文提出了一种名为AirRAG的新颖的RAG推理模式，该方法将系统分析与高效的推理动作相结合，通过蒙特卡洛树搜索（MCTS）显著激活内在的推理能力，并通过这种方法扩展特定任务的解决方案空间。具体而言，我们的方法设计了五个基本的推理动作，并通过MCTS扩展到广泛的树结构推理空间。扩展还采用了自我一致性验证来探索潜在的推理路径并实现推理扩展。此外，使用计算上优化的策略将更多的推理计算应用到关键动作上，以实现进一步的性能提升。实验结果显示，与复杂的QA数据集相比，AirRAG在性能提升方面具有明显的效果。此外，AirRAG具有高度的灵活性和轻量级特性，使其易于与其他先进技术集成。 

---
# Conversational Text Extraction with Large Language Models Using Retrieval-Augmented Systems 

**Title (ZH)**: 使用检索增强系统的大语言模型对话文本提取 

**Authors**: Soham Roy, Mitul Goswami, Nisharg Nargund, Suneeta Mohanty, Prasant Kumar Pattnaik  

**Link**: [PDF](https://arxiv.org/pdf/2501.09801)  

**Abstract**: This study introduces a system leveraging Large Language Models (LLMs) to extract text and enhance user interaction with PDF documents via a conversational interface. Utilizing Retrieval-Augmented Generation (RAG), the system provides informative responses to user inquiries while highlighting relevant passages within the PDF. Upon user upload, the system processes the PDF, employing sentence embeddings to create a document-specific vector store. This vector store enables efficient retrieval of pertinent sections in response to user queries. The LLM then engages in a conversational exchange, using the retrieved information to extract text and generate comprehensive, contextually aware answers. While our approach demonstrates competitive ROUGE values compared to existing state-of-the-art techniques for text extraction and summarization, we acknowledge that further qualitative evaluation is necessary to fully assess its effectiveness in real-world applications. The proposed system gives competitive ROUGE values as compared to existing state-of-the-art techniques for text extraction and summarization, thus offering a valuable tool for researchers, students, and anyone seeking to efficiently extract knowledge and gain insights from documents through an intuitive question-answering interface. 

**Abstract (ZH)**: 本研究介绍了一种利用大型语言模型（LLMs）的系统，通过对话界面提取文本并增强用户与PDF文档的交互。该系统利用检索增强生成（RAG）技术，能够对用户的查询提供信息性的回应，并突出显示PDF中的相关段落。用户上传PDF文档后，系统对其进行处理，使用句子嵌入创建特定文档的向量库。该向量库能够在接收到用户查询时高效地检索相关部分。随后，大型语言模型参与对话交流，利用检索到的信息提取文本并生成全面且上下文相关的答案。尽管我们的方法在文本抽取与总结方面与现有的先进技术相比展现了竞争力的ROUGE值，但我们认识到还需要进一步的定性评估来全面评估其在实际应用中的有效性。所提出的系统在文本抽取与总结方面与现有的先进技术相比展示了竞争性的ROUGE值，从而为研究人员、学生以及希望通过直观的问答接口高效地从文档中提取知识和获得洞察的人提供了一个有 value 的工具。 

---
# Passage Segmentation of Documents for Extractive Question Answering 

**Title (ZH)**: 文档中的段落分割用于提取式问答 

**Authors**: Zuhong Liu, Charles-Elie Simon, Fabien Caspani  

**Link**: [PDF](https://arxiv.org/pdf/2501.09940)  

**Abstract**: Retrieval-Augmented Generation (RAG) has proven effective in open-domain question answering. However, the chunking process, which is essential to this pipeline, often receives insufficient attention relative to retrieval and synthesis components. This study emphasizes the critical role of chunking in improving the performance of both dense passage retrieval and the end-to-end RAG pipeline. We then introduce the Logits-Guided Multi-Granular Chunker (LGMGC), a novel framework that splits long documents into contextualized, self-contained chunks of varied granularity. Our experimental results, evaluated on two benchmark datasets, demonstrate that LGMGC not only improves the retrieval step but also outperforms existing chunking methods when integrated into a RAG pipeline. 

**Abstract (ZH)**: 检索增强生成（RAG）在开放式领域问答中已被证明是有效的。然而，这一管道中的关键步骤——分块过程——往往相较于检索和合成组件而言得到了较少的关注。本研究强调了分块在提高密集段落检索性能以及端到端RAG管道性能中的关键作用。随后，我们引入了一种新的框架——Logits-Guided多粒度分块器（LGMGC），该框架能够将长文档拆分成自包含的、上下文化的不同粒度的片段。我们在两个基准数据集上的实验结果表明，LGMGC不仅改善了检索步骤的表现，而且在其集成到RAG管道中时，其性能也优于现有的分块方法。 

---
# Multi-stage Training of Bilingual Islamic LLM for Neural Passage Retrieval 

**Title (ZH)**: 面向神经段落检索的双语伊斯兰大语言模型的多阶段训练方法 

**Authors**: Vera Pavlova  

**Link**: [PDF](https://arxiv.org/pdf/2501.10175)  

**Abstract**: This study examines the use of Natural Language Processing (NLP) technology within the Islamic domain, focusing on developing an Islamic neural retrieval model. By leveraging the robust XLM-R model, the research employs a language reduction technique to create a lightweight bilingual large language model (LLM). Our approach for domain adaptation addresses the unique challenges faced in the Islamic domain, where substantial in-domain corpora exist only in Arabic while limited in other languages, including English.
The work utilizes a multi-stage training process for retrieval models, incorporating large retrieval datasets, such as MS MARCO, and smaller, in-domain datasets to improve retrieval performance. Additionally, we have curated an in-domain retrieval dataset in English by employing data augmentation techniques and involving a reliable Islamic source. This approach enhances the domain-specific dataset for retrieval, leading to further performance gains.
The findings suggest that combining domain adaptation and a multi-stage training method for the bilingual Islamic neural retrieval model enables it to outperform monolingual models on downstream retrieval tasks. 

**Abstract (ZH)**: 本研究探讨了自然语言处理（NLP）技术在伊斯兰领域的应用，重点关注开发一种伊斯兰神经检索模型。通过利用强大的XLM-R模型，研究采用语言缩减技术创建了轻量级的双语大型语言模型（LLM）。我们在领域适应方面的方法应对了伊斯兰领域特有的挑战，其中大量领域内语料库仅存在于阿拉伯语中，而在其他语言中，尤其是英语中则极为有限。

研究利用了多阶段训练方法来改进检索模型，结合了大型检索数据集（如MS MARCO）以及更小的领域内数据集，以提高检索性能。此外，我们通过采用数据增强技术并利用可靠的伊斯兰来源数据，建立了英语领域的领域内检索数据集。这种方法增强了领域特定的检索数据集，从而进一步提高了性能。

研究结果表明，结合领域适应和多阶段训练方法能够使双语伊斯兰神经检索模型在下游检索任务中优于单一语言模型。 

---
# FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs 

**Title (ZH)**: FRAG：一种基于知识图谱的灵活模块化检索增强生成框架 

**Authors**: Zengyi Gao, Yukun Cao, Hairu Wang, Ao Ke, Yuan Feng, Xike Xie, S Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.09957)  

**Abstract**: To mitigate the hallucination and knowledge deficiency in large language models (LLMs), Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) has shown promising potential by utilizing KGs as external resource to enhance LLMs this http URL, existing KG-RAG approaches struggle with a trade-off between flexibility and retrieval this http URL methods prioritize flexibility by avoiding the use of KG-fine-tuned models during retrieval, leading to fixed retrieval strategies and suboptimal retrieval this http URL, coupled methods embed KG information within models to improve retrieval quality, but at the expense of this http URL this paper, we propose a novel flexible modular KG-RAG framework, termed FRAG, which synergizes the advantages of both this http URL estimates the hop range of reasoning paths based solely on the query and classify it as either simple or this http URL match the complexity of the query, tailored pipelines are applied to ensure efficient and accurate reasoning path retrieval, thus fostering the final reasoning this http URL using the query text instead of the KG to infer the structural information of reasoning paths and employing adaptable retrieval strategies, FRAG improves retrieval quality while maintaining this http URL, FRAG does not require extra LLMs fine-tuning or calls, significantly boosting efficiency and conserving this http URL experiments show that FRAG achieves state-of-the-art performance with high efficiency and low resource consumption. 

**Abstract (ZH)**: 为了缓解大规模语言模型（LLMs）中的幻觉和知识缺陷，通过利用知识图谱（KGs）作为外部资源来增强LLMs的检索增强生成（RAG）方法显示出有前景的潜力（this http URL），现有的KG-RAG方法在灵活性与检索之间面临权衡（this http URL）。一种方法通过避免在检索期间使用KG微调模型来优先考虑灵活性，导致固定检索策略和次优的检索效果（this http URL）。另一种方法将KG信息嵌入模型中以提高检索质量，但会牺牲（this http URL）。本文提出了一种新的灵活模块化的 KG-RAG 框架，称为FRAG，该框架结合了上述两种方法的优点（this http URL）。基于查询估算推理路径的合理跨度，并将其分类为简单或复杂，根据查询的复杂度调整合适的流水线，以确保高效准确的推理路径检索，从而促进最终的推理过程（this http URL）。使用查询文本而不是KG来推断推理路径的结构信息，并采用可调适的检索策略，FRAG 提高了检索质量，同时保持了灵活性（this http URL）。FRAG 不需要额外的LLMs微调或调用，显著提高了效率并节省了资源。实验结果表明，FRAG 在高效率和低资源消耗的情况下达到了最先进的性能（this http URL）。 

---
