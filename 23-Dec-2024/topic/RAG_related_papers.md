# Towards Interpretable Radiology Report Generation via Concept Bottlenecks using a Multi-Agentic RAG 

**Title (ZH)**: 通过多智能体 Retrieval-Augmented Generation (RAG) 中的概念瓶颈实现可解释的放射学报告生成 

**Authors**: Hasan Md Tusfiqur Alam, Devansh Srivastav, Md Abdul Kadir, Daniel Sonntag  

**Link**: [PDF](https://arxiv.org/pdf/2412.16086)  

**Abstract**: Deep learning has advanced medical image classification, but interpretability challenges hinder its clinical adoption. This study enhances interpretability in Chest X-ray (CXR) classification by using concept bottleneck models (CBMs) and a multi-agent Retrieval-Augmented Generation (RAG) system for report generation. By modeling relationships between visual features and clinical concepts, we create interpretable concept vectors that guide a multi-agent RAG system to generate radiology reports, enhancing clinical relevance, explainability, and transparency. Evaluation of the generated reports using an LLM-as-a-judge confirmed the interpretability and clinical utility of our model's outputs. On the COVID-QU dataset, our model achieved 81% classification accuracy and demonstrated robust report generation performance, with five key metrics ranging between 84% and 90%. This interpretable multi-agent framework bridges the gap between high-performance AI and the explainability required for reliable AI-driven CXR analysis in clinical settings. 

**Abstract (ZH)**: 深度学习技术的进步促进了医学图像分类，但其可解释性不足限制了其临床应用。本研究通过使用概念瓶颈模型（CBMs）和多智能体检索增强生成（RAG）系统，增强了胸部X光（CXR）分类的可解释性。通过建模视觉特征与临床概念之间的关系，我们创建了可解释的概念向量，这些向量指导多智能体RAG系统生成放射学报告，从而增强了临床相关性、可解释性和透明度。利用语言模型作为评委进行生成报告的评估，证实了我们模型输出的可解释性和临床实用性。在COVID-QU数据集中，我们的模型实现了81%的分类准确率，并展示了稳健的报告生成性能，五个关键指标的范围在84%到90%之间。这种可解释的多智能体框架填补了高性能AI与临床环境中可靠AI驱动CXR分析所需可解释性之间的空白。 

---
# On the Suitability of pre-trained foundational LLMs for Analysis in German Legal Education 

**Title (ZH)**: 预训练基础大语言模型在德国法律教育分析中的适用性研究 

**Authors**: Lorenz Wendlinger, Christian Braun, Abdullah Al Zubaer, Simon Alexander Nonn, Sarah Großkopf, Christofer Fellicious, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2412.15902)  

**Abstract**: We show that current open-source foundational LLMs possess instruction capability and German legal background knowledge that is sufficient for some legal analysis in an educational context. However, model capability breaks down in very specific tasks, such as the classification of "Gutachtenstil" appraisal style components, or with complex contexts, such as complete legal opinions. Even with extended context and effective prompting strategies, they cannot match the Bag-of-Words baseline. To combat this, we introduce a Retrieval Augmented Generation based prompt example selection method that substantially improves predictions in high data availability scenarios. We further evaluate the performance of pre-trained LLMs on two standard tasks for argument mining and automated essay scoring and find it to be more adequate. Throughout, pre-trained LLMs improve upon the baseline in scenarios with little or no labeled data with Chain-of-Thought prompting further helping in the zero-shot case. 

**Abstract (ZH)**: 我们表明，当前开源的基础型大语言模型具有一定的指令能力和德法律制背景知识，足以在教育情境下进行部分法律分析。然而，模型的能力在某些特定任务中会失效，例如“Gutachtenstil”评估风格成分的分类，或者在复杂的语境中，如完整的法律意见。即使提供扩展的语境和有效的提示策略，它们也无法匹配基于词汇的基线模型。为解决这一问题，我们提出了一种检索增强生成（RAG）基于提示示例选择的方法，该方法在数据充足的情况下显著提高了预测效果。此外，我们进一步评估了预训练大语言模型在两个标准任务中的表现，即论据挖掘和自动作文评分，并发现它们更为合适。在整个过程中，预训练的大语言模型在标注数据较少或没有的情况下表现更好，链式思考（Chain-of-Thought）提示在零样本情况下也有助于提高性能。 

---
# XRAG: eXamining the Core -- Benchmarking Foundational Components in Advanced Retrieval-Augmented Generation 

**Title (ZH)**: XRAG: 探索核心——高级检索增强生成的基础组件评估 

**Authors**: Qianren Mao, Yangyifei Luo, Jinlong Zhang, Hanwen Hao, Zhilong Cao, Xiaolong Wang, Xiao Guan, Zhenting Huang, Weifeng Jiang, Shuyu Guo, Zhentao Han, Qili Zhang, Siyuan Tao, Yujie Liu, Junnan Liu, Zhixing Tan, Jie Sun, Bo Li, Xudong Liu, Richong Zhang, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.15529)  

**Abstract**: Retrieval-augmented generation (RAG) synergizes the retrieval of pertinent data with the generative capabilities of Large Language Models (LLMs), ensuring that the generated output is not only contextually relevant but also accurate and this http URL introduce XRAG, an open-source, modular codebase that facilitates exhaustive evaluation of the performance of foundational components of advanced RAG modules. These components are systematically categorized into four core phases: pre-retrieval, retrieval, post-retrieval, and generation. We systematically analyse them across reconfigured datasets, providing a comprehensive benchmark for their effectiveness. Given the escalating complexity of RAG systems, we underscore the necessity of identifying potential failure points of RAG modules. We formulate a suite of experimental methodologies and diagnostic testing protocols to dissect the failure points inherent in the engineering of RAG modules. Subsequently, we proffer bespoke solutions that are designed to augment the validation processes and bolster the overall performance of these modules. Our work thoroughly evaluates the performance of core advanced components in RAG systems, providing insights into optimizations for prevalent failure points. 

**Abstract (ZH)**: 检索增强生成（RAG）将相关数据的检索能力与大型语言模型（LLMs）的生成能力相结合，确保生成的输出不仅具有上下文相关性，还能保持准确性和可靠性。本文介绍了XRAG，一个开源、模块化的代码库，旨在进行全面评估先进RAG模块基础组件的性能。这些组件被系统地归类为四个核心阶段：预检索、检索、后检索和生成。我们通过对重新配置的数据集进行全面分析，提供了这些组件有效性的综合基准。随着RAG系统的复杂性日益增加，我们强调识别RAG模块潜在故障点的重要性。我们制定了实验方法和诊断测试协议，旨在剖析工程RAG模块时固有的故障点。随后，我们提出了一些定制化的解决方案，旨在增强这些模块的验证过程，从而提高它们的整体性能。我们的工作全面评估了RAG系统中核心先进组件的性能，提供了针对常见故障点优化的见解。 

---
# SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation 

**Title (ZH)**: SimGRAG：利用相似子图进行知识图驱动的检索增强生成 

**Authors**: Yuzheng Cai, Zhenyue Guo, Yiwen Pei, Wanrui Bian, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.15272)  

**Abstract**: Recent advancements in large language models (LLMs) have shown impressive versatility across various tasks. To eliminate its hallucinations, retrieval-augmented generation (RAG) has emerged as a powerful approach, leveraging external knowledge sources like knowledge graphs (KGs). In this paper, we study the task of KG-driven RAG and propose a novel Similar Graph Enhanced Retrieval-Augmented Generation (SimGRAG) method. It effectively addresses the challenge of aligning query texts and KG structures through a two-stage process: (1) query-to-pattern, which uses an LLM to transform queries into a desired graph pattern, and (2) pattern-to-subgraph, which quantifies the alignment between the pattern and candidate subgraphs using a graph semantic distance (GSD) metric. We also develop an optimized retrieval algorithm that efficiently identifies the top-$k$ subgraphs within 1-second latency on a 10-million-scale KG. Extensive experiments show that SimGRAG outperforms state-of-the-art KG-driven RAG methods in both question answering and fact verification, offering superior plug-and-play usability and scalability. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在各种任务中展现了令人印象深刻的灵活性。为了消除其幻觉，检索增强生成（RAG）方法已经作为一种有力的手段出现了，它利用了知识图谱（KGs）等外部知识源。本文研究了基于KG的RAG任务，并提出了一种新颖的相似图增强检索增强生成（SimGRAG）方法。该方法通过一个两阶段过程有效解决了查询文本与KG结构对齐的挑战：（1）查询到模式，使用LLM将查询转换为所需的图模式；（2）模式到子图，使用图语义距离（GSD）度量来量化模式与候选子图之间的对齐程度。此外，我们还开发了一种优化的检索算法，该算法能在1秒钟的延迟内高效地在规模为1000万的知识图谱中识别出前k个子图。广泛的实验表明，SimGRAG在问题回答和事实验证方面均优于现有的基于KG的RAG方法，提供了卓越的即插即用可行性和可扩展性。 

---
# Accelerating Retrieval-Augmented Generation 

**Title (ZH)**: 加速检索增强生成 

**Authors**: Derrick Quinn, Mohammad Nouri, Neel Patel, John Salihu, Alireza Salemi, Sukhan Lee, Hamed Zamani, Mohammad Alian  

**Link**: [PDF](https://arxiv.org/pdf/2412.15246)  

**Abstract**: An evolving solution to address hallucination and enhance accuracy in large language models (LLMs) is Retrieval-Augmented Generation (RAG), which involves augmenting LLMs with information retrieved from an external knowledge source, such as the web. This paper profiles several RAG execution pipelines and demystifies the complex interplay between their retrieval and generation phases. We demonstrate that while exact retrieval schemes are expensive, they can reduce inference time compared to approximate retrieval variants because an exact retrieval model can send a smaller but more accurate list of documents to the generative model while maintaining the same end-to-end accuracy. This observation motivates the acceleration of the exact nearest neighbor search for RAG.
In this work, we design Intelligent Knowledge Store (IKS), a type-2 CXL device that implements a scale-out near-memory acceleration architecture with a novel cache-coherent interface between the host CPU and near-memory accelerators. IKS offers 13.4-27.9x faster exact nearest neighbor search over a 512GB vector database compared with executing the search on Intel Sapphire Rapids CPUs. This higher search performance translates to 1.7-26.3x lower end-to-end inference time for representative RAG applications. IKS is inherently a memory expander; its internal DRAM can be disaggregated and used for other applications running on the server to prevent DRAM, which is the most expensive component in today's servers, from being stranded. 

**Abstract (ZH)**: 解决大型语言模型（LLMs）幻觉问题并提高其准确性的不断演化的解决方案是检索增强生成（RAG），这种技术通过从外部知识来源（例如网络）检索信息来增强LLMs。本文概述了几种RAG执行管道，并阐明了其检索和生成阶段之间的复杂交互关系。我们展示了虽然精确检索方案成本较高，但与近似检索变体相比，它们可以在保持相同端到端准确性的前提下减少推理时间，因为精确检索模型可以向生成模型发送更小但更准确的文档列表。这一观察结果促使我们加速RAG中的精确最近邻搜索。

在本文中，我们设计了一种名为智能知识存储（IKS）的CXL类型2设备，它实现了扩展的近内存加速架构，并在主机CPU和近内存加速器之间采用了一种新颖的缓存一致性接口。相比于在Intel Sapphire Rapids CPU上执行搜索，IKS在512GB向量数据库上的精确最近邻搜索速度提高了13.4到27.9倍。这种更高的搜索性能转化为代表性的RAG应用程序中1.7到26.3倍的端到端推理时间降低。IKS本质上是一种内存扩展器；其内部DRAM可以分离并用于服务器上的其他应用程序，从而防止当今服务器中最昂贵的组件——DRAM——被闲置。 

---
# OG-RAG: Ontology-Grounded Retrieval-Augmented Generation For Large Language Models 

**Title (ZH)**: OG-RAG：面向本体检索增强生成方法用于大型语言模型 

**Authors**: Kartik Sharma, Peeyush Kumar, Yunqing Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.15235)  

**Abstract**: This paper presents OG-RAG, an Ontology-Grounded Retrieval Augmented Generation method designed to enhance LLM-generated responses by anchoring retrieval processes in domain-specific ontologies. While LLMs are widely used for tasks like question answering and search, they struggle to adapt to specialized knowledge, such as industrial workflows or knowledge work, without expensive fine-tuning or sub-optimal retrieval methods. Existing retrieval-augmented models, such as RAG, offer improvements but fail to account for structured domain knowledge, leading to suboptimal context generation. Ontologies, which conceptually organize domain knowledge by defining entities and their interrelationships, offer a structured representation to address this gap. OG-RAG constructs a hypergraph representation of domain documents, where each hyperedge encapsulates clusters of factual knowledge grounded using domain-specific ontology. An optimization algorithm then retrieves the minimal set of hyperedges that constructs a precise, conceptually grounded context for the LLM. This method enables efficient retrieval while preserving the complex relationships between entities. OG-RAG applies to domains where fact-based reasoning is essential, particularly in tasks that require workflows or decision-making steps to follow predefined rules and procedures. These include industrial workflows in healthcare, legal, and agricultural sectors, as well as knowledge-driven tasks such as news journalism, investigative research, consulting and more. Our evaluations demonstrate that OG-RAG increases the recall of accurate facts by 55% and improves response correctness by 40% across four different LLMs. Additionally, OG-RAG enables 30% faster attribution of responses to context and boosts fact-based reasoning accuracy by 27% compared to baseline methods. 

**Abstract (ZH)**: 本文提出了一种基于本体的检索增强生成方法（OG-RAG），旨在通过将检索过程锚定在领域特定本体上，从而增强LLM生成的响应。虽然LLM广泛应用于问答和搜索等任务，但在处理如工业流程或知识工作等专业领域知识时，它们往往无法通过昂贵的微调或不理想的检索方法进行有效适应。现有的检索增强模型，如RAG，虽然提供了改进，但未能考虑到结构化的领域知识，导致上下文生成效果欠佳。本体从概念上组织领域知识，通过定义实体及其相互关系来提供一种结构化的表示方法，以此填补这一空白。OG-RAG 构建了一个领域文档的超图表示，其中每个超边封装了通过领域特定本体进行事实性知识归因的实体集群。随后优化算法检索出能够构成精确、概念性上下文的最小超边集，以供LLM使用。这种方法能够在保留实体间复杂关系的同时实现高效的检索。OG-RAG 适用于基于事实的推理至关重要的领域，特别是在需要遵循预定义规则和程序的工作流或决策任务中尤为重要。这些领域包括医疗、法律和农业行业的工业工作流程，以及诸如新闻采编、调查研究、咨询等知识驱动的任务。我们的评估结果显示，OG-RAG 将准确事实的召回率提高了55%，并将响应的正确性提高了40%。此外，OG-RAG 使得对上下文的响应归属速度提高了30%，并且相比基准方法提高了27%的事实性推理准确性。 

---
# SKETCH: Structured Knowledge Enhanced Text Comprehension for Holistic Retrieval 

**Title (ZH)**: SKETCH: 结构化知识增强的文本理解方法用于全面检索 

**Authors**: Aakash Mahalingam, Vinesh Kumar Gande, Aman Chadha, Vinija Jain, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2412.15443)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have become pivotal in leveraging vast corpora to generate informed and contextually relevant responses, notably reducing hallucinations in Large Language Models. Despite significant advancements, these systems struggle to efficiently process and retrieve information from large datasets while maintaining a comprehensive understanding of the context. This paper introduces SKETCH, a novel methodology that enhances the RAG retrieval process by integrating semantic text retrieval with knowledge graphs, thereby merging structured and unstructured data for a more holistic comprehension. SKETCH, demonstrates substantial improvements in retrieval performance and maintains superior context integrity compared to traditional methods. Evaluated across four diverse datasets: QuALITY, QASPER, NarrativeQA, and Italian Cuisine-SKETCH consistently outperforms baseline approaches on key RAGAS metrics such as answer_relevancy, faithfulness, context_precision and context_recall. Notably, on the Italian Cuisine dataset, SKETCH achieved an answer relevancy of 0.94 and a context precision of 0.99, representing the highest performance across all evaluated metrics. These results highlight SKETCH's capability in delivering more accurate and contextually relevant responses, setting new benchmarks for future retrieval systems. 

**Abstract (ZH)**: 检索增强生成（RAG）系统已经被证明是利用大量语料库生成有信息量且上下文相关回答的关键工具，显著减少了大型语言模型中的幻觉现象。尽管取得了显著的进展，这些系统在高效处理和从大型数据集中检索信息的同时保持全面的上下文理解方面仍存在挑战。本文提出了一种名为SKETCH的新方法，通过将语义文本检索与知识图谱结合，增强了RAG的检索过程，从而将结构化数据和非结构化数据结合起来，提供了更全面的理解。SKETCH在检索性能上表现出显著提升，并在保持上下文完整性方面优于传统方法。SKETCH在QuALITY、QASPER、NarrativeQA和意大利菜谱这四个不同数据集上的表现均超过了基准方法，在关键的RAGAS指标（如答案相关性、忠实性、上下文精确度和上下文召回率）上均表现出色。特别是在意大利菜谱数据集上，SKETCH达到了0.94的答案相关性和0.99的上下文精确度，这些指标在整个评估中均达到了最高性能。这些结果突显了SKETCH在提供更准确且上下文相关的回答方面的能力，为未来的检索系统设定了新的基准。 

---
# A MapReduce Approach to Effectively Utilize Long Context Information in Retrieval Augmented Language Models 

**Title (ZH)**: 一种利用检索增强语言模型中长上下文信息的有效MapReduce方法 

**Authors**: Gongbo Zhang, Zihan Xu, Qiao Jin, Fangyi Chen, Yilu Fang, Yi Liu, Justin F. Rousseau, Ziyang Xu, Zhiyong Lu, Chunhua Weng, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2412.15271)  

**Abstract**: While holding great promise for improving and facilitating healthcare, large language models (LLMs) struggle to produce up-to-date responses on evolving topics due to outdated knowledge or hallucination. Retrieval-augmented generation (RAG) is a pivotal innovation that improves the accuracy and relevance of LLM responses by integrating LLMs with a search engine and external sources of knowledge. However, the quality of RAG responses can be largely impacted by the rank and density of key information in the retrieval results, such as the "lost-in-the-middle" problem. In this work, we aim to improve the robustness and reliability of the RAG workflow in the medical domain. Specifically, we propose a map-reduce strategy, BriefContext, to combat the "lost-in-the-middle" issue without modifying the model weights. We demonstrated the advantage of the workflow with various LLM backbones and on multiple QA datasets. This method promises to improve the safety and reliability of LLMs deployed in healthcare domains. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在提高和促进医疗服务方面展现出巨大的潜力，它们在处理快速演化的主题时，由于知识陈旧或幻觉（hallucination）而难以提供及时准确的回答。检索增强生成（RAG，Retrieval-Augmented Generation）作为一种关键的创新技术，通过将LLMs与搜索引擎和外部知识源相结合，提高了其回答的准确性和相关性。然而，RAG的回答质量很大程度上受到检索结果中关键信息排名和密度的影响，例如“中间信息丢失”问题。在这项工作中，我们旨在提高医疗领域的RAG工作流的稳健性和可靠性。具体而言，我们提出了一种映射-减少策略（BriefContext），通过这种策略可以在不修改模型权重的情况下应对“中间信息丢失”问题。我们通过多种LLM基础模型和多个QA数据集展示了该方法的优势。该方法有望提高部署在医疗领域的LLMs的安全性和可靠性。 

---
# DisEmbed: Transforming Disease Understanding through Embeddings 

**Title (ZH)**: DisEmbed：通过嵌入变换疾病理解 

**Authors**: Salman Faroz  

**Link**: [PDF](https://arxiv.org/pdf/2412.15258)  

**Abstract**: The medical domain is vast and diverse, with many existing embedding models focused on general healthcare applications. However, these models often struggle to capture a deep understanding of diseases due to their broad generalization across the entire medical field. To address this gap, I present DisEmbed, a disease-focused embedding model. DisEmbed is trained on a synthetic dataset specifically curated to include disease descriptions, symptoms, and disease-related Q\&A pairs, making it uniquely suited for disease-related tasks. For evaluation, I benchmarked DisEmbed against existing medical models using disease-specific datasets and the triplet evaluation method. My results demonstrate that DisEmbed outperforms other models, particularly in identifying disease-related contexts and distinguishing between similar diseases. This makes DisEmbed highly valuable for disease-specific use cases, including retrieval-augmented generation (RAG) tasks, where its performance is particularly robust. 

**Abstract (ZH)**: 医学领域庞大而多样，尽管已有许多嵌入模型专注于一般医疗应用，但这些模型在捕捉疾病深层次的理解方面往往存在困难，原因在于它们在整個医学领域中的广泛泛化。为了解决这一问题，我提出了DisEmbed，一种疾病导向的嵌入模型。DisEmbed基于一个专门设计的合成数据集进行训练，该数据集包含了疾病描述、症状以及疾病相关的问答对，使其特别适用于疾病相关的任务。为了评估DisEmbed的表现，我使用了疾病专门化的数据集和三元组评估方法，将DisEmbed与现有的医疗模型进行了基准测试。结果显示，DisEmbed在识别疾病相关上下文和区分相似疾病方面表现尤为出色，使其在疾病专门的应用场景中具有极高的价值，特别是在增强检索生成（RAG）任务中，其性能尤为稳健。 

---
# A Retrieval-Augmented Generation Framework for Academic Literature Navigation in Data Science 

**Title (ZH)**: 数据科学中学术文献导航的检索增强生成框架 

**Authors**: Ahmet Yasin Aytar, Kemal Kilic, Kamer Kaya  

**Link**: [PDF](https://arxiv.org/pdf/2412.15404)  

**Abstract**: In the rapidly evolving field of data science, efficiently navigating the expansive body of academic literature is crucial for informed decision-making and innovation. This paper presents an enhanced Retrieval-Augmented Generation (RAG) application, an artificial intelligence (AI)-based system designed to assist data scientists in accessing precise and contextually relevant academic resources. The AI-powered application integrates advanced techniques, including the GeneRation Of BIbliographic Data (GROBID) technique for extracting bibliographic information, fine-tuned embedding models, semantic chunking, and an abstract-first retrieval method, to significantly improve the relevance and accuracy of the retrieved information. This implementation of AI specifically addresses the challenge of academic literature navigation. A comprehensive evaluation using the Retrieval-Augmented Generation Assessment System (RAGAS) framework demonstrates substantial improvements in key metrics, particularly Context Relevance, underscoring the system's effectiveness in reducing information overload and enhancing decision-making processes. Our findings highlight the potential of this enhanced Retrieval-Augmented Generation system to transform academic exploration within data science, ultimately advancing the workflow of research and innovation in the field. 

**Abstract (ZH)**: 在数据科学这一快速发展的领域中，有效地导航海量的学术文献对于做出明智的决策和创新至关重要。本文介绍了一种增强型检索增强生成（RAG）应用，这是一种基于人工智能（AI）的系统，旨在帮助数据科学家访问精确且上下文相关的学术资源。该AI驱动的应用程序集成了先进的技术，包括用于提取参考文献信息的GeneRation Of BIbliographic Data (GROBID) 技术、微调嵌入模型、语义切块，以及摘录优先检索方法，以显著提高检索信息的相关性和准确性。该AI 实现特别解决了学术文献导航的挑战。使用检索增强生成评估系统（RAGAS）框架进行的全面评估显示，在关键指标上取得了显著改进，特别是在上下文相关性方面，突显了该系统在减少信息过载和增强决策过程方面的有效性。我们的研究结果强调了这种增强型检索增强生成系统的潜在能力，以重塑数据科学中的学术探索，并最终推动该领域研究与创新的工作流程。 

---
