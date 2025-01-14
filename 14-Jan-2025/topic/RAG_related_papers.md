# A Proposed Large Language Model-Based Smart Search for Archive System 

**Title (ZH)**: 一种基于大型语言模型的智能存档系统搜索方法 

**Authors**: Ha Dung Nguyen, Thi-Hoang Anh Nguyen, Thanh Binh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07024)  

**Abstract**: This study presents a novel framework for smart search in digital archival systems, leveraging the capabilities of Large Language Models (LLMs) to enhance information retrieval. By employing a Retrieval-Augmented Generation (RAG) approach, the framework enables the processing of natural language queries and transforming non-textual data into meaningful textual representations. The system integrates advanced metadata generation techniques, a hybrid retrieval mechanism, a router query engine, and robust response synthesis, the results proved search precision and relevance. We present the architecture and implementation of the system and evaluate its performance in four experiments concerning LLM efficiency, hybrid retrieval optimizations, multilingual query handling, and the impacts of individual components. Obtained results show significant improvements over conventional approaches and have demonstrated the potential of AI-powered systems to transform modern archival practices. 

**Abstract (ZH)**: 本研究提出了一种用于数字档案系统智能搜索的新框架，充分利用大型语言模型（LLMs）的能力来增强信息检索。通过采用检索增强生成（RAG）方法，该框架能够处理自然语言查询，并将非文本数据转换为有意义的文本表示。该系统集成了高级元数据生成技术、混合检索机制、路由器查询引擎以及强大的响应合成方法，结果显示了搜索精度和相关性的提升。我们展示了该系统的架构和实现，并在四个实验中评估了其性能，涉及大型语言模型效率、混合检索优化、多语言查询处理以及各组件的影响。所获得的结果表明，该方法显著优于传统方法，展示了基于人工智能的系统在转变现代档案实践方面的巨大潜力。 

---
# MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation 

**Title (ZH)**: MiniRAG：趋向极简的检索增强生成 

**Authors**: Tianyu Fan, Jingyuan Wang, Xubin Ren, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06713)  

**Abstract**: The growing demand for efficient and lightweight Retrieval-Augmented Generation (RAG) systems has highlighted significant challenges when deploying Small Language Models (SLMs) in existing RAG frameworks. Current approaches face severe performance degradation due to SLMs' limited semantic understanding and text processing capabilities, creating barriers for widespread adoption in resource-constrained scenarios. To address these fundamental limitations, we present MiniRAG, a novel RAG system designed for extreme simplicity and efficiency. MiniRAG introduces two key technical innovations: (1) a semantic-aware heterogeneous graph indexing mechanism that combines text chunks and named entities in a unified structure, reducing reliance on complex semantic understanding, and (2) a lightweight topology-enhanced retrieval approach that leverages graph structures for efficient knowledge discovery without requiring advanced language capabilities. Our extensive experiments demonstrate that MiniRAG achieves comparable performance to LLM-based methods even when using SLMs while requiring only 25\% of the storage space. Additionally, we contribute a comprehensive benchmark dataset for evaluating lightweight RAG systems under realistic on-device scenarios with complex queries. We fully open-source our implementation and datasets at: this https URL. 

**Abstract (ZH)**: 随着对高效且轻量级检索增强生成（RAG）系统的日益需求，已经突显出在现有RAG框架中部署小型语言模型（SLM）时面临的重大挑战。当前的方法因SLM有限的语义理解和文本处理能力而遭受严重的性能下降，这在资源受限的场景中构成了广泛采用的障碍。为了解决这些根本限制，我们提出了一种新型的RAG系统——MiniRAG，旨在实现极端的简单性和效率。MiniRAG引入了两项关键技术创新：（1）一种语义感知的异构图索引机制，将文本片段和命名实体统一在一个结构中，减少对复杂语义理解的依赖；（2）一种基于图结构的轻量级拓扑增强检索方法，利用图结构进行高效的知识发现，而不需要先进的语言能力。我们广泛而深入的实验表明，即使使用SLM，MiniRAG也能实现与基于大规模语言模型（LLM）的方法相当的性能，同时只需要25%的存储空间。此外，我们还贡献了一个全面的基准数据集，用于在复杂查询下对轻量级RAG系统进行现实设备场景的评估。我们已完全开源我们的实现和数据集：请访问这个链接：this https URL。 

---
# Enhancing Retrieval-Augmented Generation: A Study of Best Practices 

**Title (ZH)**: 增强检索增强生成：最佳实践研究 

**Authors**: Siran Li, Linus Stenzel, Carsten Eickhoff, Seyed Ali Bahrainian  

**Link**: [PDF](https://arxiv.org/pdf/2501.07391)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have recently shown remarkable advancements by integrating retrieval mechanisms into language models, enhancing their ability to produce more accurate and contextually relevant responses. However, the influence of various components and configurations within RAG systems remains underexplored. A comprehensive understanding of these elements is essential for tailoring RAG systems to complex retrieval tasks and ensuring optimal performance across diverse applications. In this paper, we develop several advanced RAG system designs that incorporate query expansion, various novel retrieval strategies, and a novel Contrastive In-Context Learning RAG. Our study systematically investigates key factors, including language model size, prompt design, document chunk size, knowledge base size, retrieval stride, query expansion techniques, Contrastive In-Context Learning knowledge bases, multilingual knowledge bases, and Focus Mode retrieving relevant context at sentence-level. Through extensive experimentation, we provide a detailed analysis of how these factors influence response quality. Our findings offer actionable insights for developing RAG systems, striking a balance between contextual richness and retrieval-generation efficiency, thereby paving the way for more adaptable and high-performing RAG frameworks in diverse real-world scenarios. Our code and implementation details are publicly available. 

**Abstract (ZH)**: 检索增强生成（RAG）系统通过将检索机制融入语言模型，最近取得了显著进展，增强了其生成更加准确和语境相关响应的能力。然而，RAG系统内部各种组件和配置的影响仍未被充分探索。对这些元素的全面理解对于根据复杂检索任务定制RAG系统并确保在各种应用场景下实现最佳性能至关重要。在本文中，我们开发了几个先进的RAG系统设计，整合了查询扩展、各种新颖的检索策略以及一种新颖的对比上下文学习RAG。我们通过系统的实验研究了关键因素，包括语言模型的规模、提示设计、文档片段大小、知识库规模、检索步长、查询扩展技术、对比上下文学习知识库、多语言知识库以及聚焦模式在句子级别检索相关上下文。通过广泛的实验，我们详细分析了这些因素如何影响响应质量。我们的发现为开发RAG系统提供了可操作的见解，平衡了上下文丰富性和检索生成效率，从而为不同真实世界场景下的更灵活和高性能RAG框架奠定了基础。我们的代码和实现细节是公开的。 

---
# First Token Probability Guided RAG for Telecom Question Answering 

**Title (ZH)**: 基于首个标记概率的RAG在电信领域问答中的应用 

**Authors**: Tingwei Chen, Jiayi Chen, Zijian Zhao, Haolong Chen, Liang Zhang, Guangxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06468)  

**Abstract**: Large Language Models (LLMs) have garnered significant attention for their impressive general-purpose capabilities. For applications requiring intricate domain knowledge, Retrieval-Augmented Generation (RAG) has shown a distinct advantage in incorporating domain-specific information into LLMs. However, existing RAG research has not fully addressed the challenges of Multiple Choice Question Answering (MCQA) in telecommunications, particularly in terms of retrieval quality and mitigating hallucinations. To tackle these challenges, we propose a novel first token probability guided RAG framework. This framework leverages confidence scores to optimize key hyperparameters, such as chunk number and chunk window size, while dynamically adjusting the context. Our method starts by retrieving the most relevant chunks and generates a single token as the potential answer. The probabilities of all options are then normalized to serve as confidence scores, which guide the dynamic adjustment of the context. By iteratively optimizing the hyperparameters based on these confidence scores, we can continuously improve RAG performance. We conducted experiments to validate the effectiveness of our framework, demonstrating its potential to enhance accuracy in domain-specific MCQA tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其广泛的应用能力而引起了广泛关注。在需要复杂领域知识的应用中，检索增强生成（RAG）展示了将领域特定信息整合到LLMs中的独特优势。然而，现有的RAG研究尚未全面解决电信领域的多项选择题作答（MCQA）难题，尤其是在检索质量和缓解幻觉方面。为了解决这些问题，我们提出了一种新颖的一开始基于第一令牌概率的RAG框架。该框架利用置信度分数优化关键超参数，如片段数量和窗口大小，并动态调整上下文。我们的方法首先检索最相关的片段，生成一个潜在答案的单个令牌。然后将所有选项的概率正则化，作为置信度分数，以指导上下文的动态调整。通过迭代基于这些置信度分数优化超参数，可以不断提升RAG性能。我们进行了实验以验证该框架的有效性，展示了其在特定领域MCQA任务中提高准确性的潜力。 

---
# Research on the Online Update Method for Retrieval-Augmented Generation (RAG) Model with Incremental Learning 

**Title (ZH)**: 增量学习下检索增强生成（RAG）模型的在线更新方法研究 

**Authors**: Yuxin Fan, Yuxiang Wang, Lipeng Liu, Xirui Tang, Na Sun, Zidong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07063)  

**Abstract**: In the contemporary context of rapid advancements in information technology and the exponential growth of data volume, language models are confronted with significant challenges in effectively navigating the dynamic and ever-evolving information landscape to update and adapt to novel knowledge in real time. In this work, an online update method is proposed, which is based on the existing Retrieval Enhanced Generation (RAG) model with multiple innovation mechanisms. Firstly, the dynamic memory is used to capture the emerging data samples, and then gradually integrate them into the core model through a tunable knowledge distillation strategy. At the same time, hierarchical indexing and multi-layer gating mechanism are introduced into the retrieval module to ensure that the retrieved content is more targeted and accurate. Finally, a multi-stage network structure is established for different types of inputs in the generation stage, and cross-attention matching and screening are carried out on the intermediate representations of each stage to ensure the effective integration and iterative update of new and old knowledge. Experimental results show that the proposed method is better than the existing mainstream comparison models in terms of knowledge retention and inference accuracy. 

**Abstract (ZH)**: 在信息技术飞速发展和数据量呈指数增长的当代背景下，语言模型面临着在动态且不断演变的信息环境中高效更新和适应新知识的显著挑战，尤其是在实时更新方面。本文提出了一种在线更新方法，基于现有的检索增强生成（RAG）模型并结合了多种创新机制。首先，动态内存用于捕捉新兴数据样本，并通过可调的知识蒸馏策略逐步将其整合到核心模型中。同时，通过引入层次索引和多层门控机制来优化检索模块，以确保检索内容更加精准和目标化。最后，在生成阶段构建了多阶段网络结构，并对每个阶段的中间表示进行了交叉注意匹配和筛选，以确保新旧知识的有效整合和迭代更新。实验结果表明，所提出的方法在知识保留和推理准确性方面优于现有的主流对比模型。 

---
# Parallel Key-Value Cache Fusion for Position Invariant RAG 

**Title (ZH)**: 位置不变的RAG中的并行键值缓存融合 

**Authors**: Philhoon Oh, Jinwoo Shin, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2501.07523)  

**Abstract**: Recent advancements in Large Language Models (LLMs) underscore the necessity of Retrieval Augmented Generation (RAG) to leverage external information. However, LLMs are sensitive to the position of relevant information within contexts and tend to generate incorrect responses when such information is placed in the middle, known as `Lost in the Middle' phenomenon. In this paper, we introduce a framework that generates consistent outputs for decoder-only models, irrespective of the input context order. Experimental results for three open domain question answering tasks demonstrate position invariance, where the model is not sensitive to input context order, and superior robustness to irrelevent passages compared to prevailing approaches for RAG pipelines. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展突显了检索增强生成（RAG）方法在利用外部信息方面的重要性。然而，LLMs 对于上下文中相关信息的位置非常敏感，当相关信息位于中间位置时，模型容易生成错误的响应，这种现象被称为“中间迷失”现象。本文提出了一种框架，能够在任何输入上下文顺序下生成一致的输出。实验结果表明，该模型对输入上下文顺序不敏感，并且与现有的RAG管道方法相比，在处理无关段落时具有更好的鲁棒性。具体来说，我们在三个开放域的问答任务上进行了实验，验证了该模型在位置不变性方面的优越表现。 

---
