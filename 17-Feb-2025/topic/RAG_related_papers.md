# ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation 

**Title (ZH)**: ArchRAG：具属性的社区导向层次检索增强生成 

**Authors**: Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, Yuchi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.09891)  

**Abstract**: Retrieval-Augmented Generation (RAG) has proven effective in integrating external knowledge into large language models (LLMs) for question-answer (QA) tasks. The state-of-the-art RAG approaches often use the graph data as the external data since they capture the rich semantic information and link relationships between entities. However, existing graph-based RAG approaches cannot accurately identify the relevant information from the graph and also consume large numbers of tokens in the online retrieval process. To address these issues, we introduce a novel graph-based RAG approach, called Attributed Community-based Hierarchical RAG (ArchRAG), by augmenting the question using attributed communities, and also introducing a novel LLM-based hierarchical clustering method. To retrieve the most relevant information from the graph for the question, we build a novel hierarchical index structure for the attributed communities and develop an effective online retrieval method. Experimental results demonstrate that ArchRAG outperforms existing methods in terms of both accuracy and token cost. 

**Abstract (ZH)**: 检索增强生成（RAG）方法已被证明在将外部知识集成到大型语言模型（LLMs）中以进行问答（QA）任务时是有效的。最先进的一些RAG方法通常使用图数据作为外部数据，因为图数据能够捕捉丰富的语义信息并反映实体之间的关系。然而，现有的基于图的RAG方法在从图中准确识别相关信息方面存在困难，并且在在线检索过程中消耗了大量的 tokens。为了解决这些问题，我们提出了一种新颖的基于图的RAG方法，称为属性社区层次RAG（ArchRAG），该方法通过使用属性社区增强问题，并引入了一种新颖的基于大型语言模型的层次聚类方法。为了从图中为问题检索最相关的信息，我们构建了一种新颖的层次索引结构来索引属性社区，并开发了一种有效的在线检索方法。实验结果表明，ArchRAG 在准确性和 tokens 成本方面都优于现有方法。 

---
# AutoS$^2$earch: Unlocking the Reasoning Potential of Large Models for Web-based Source Search 

**Title (ZH)**: AutoS$^2$earch：解锁大型模型在基于网页的源搜索中的推理潜力 

**Authors**: Zhengqiu Zhu, Yatai Ji, Jiaheng Huang, Yong Zhao, Sihang Qiu, Rusheng Ju  

**Link**: [PDF](https://arxiv.org/pdf/2502.09913)  

**Abstract**: Web-based management systems have been widely used in risk control and industrial safety. However, effectively integrating source search capabilities into these systems, to enable decision-makers to locate and address the hazard (e.g., gas leak detection) remains a challenge. While prior efforts have explored using web crowdsourcing and AI algorithms for source search decision support, these approaches suffer from overheads in recruiting human participants and slow response times in time-sensitive situations. To address this, we introduce AutoS$^2$earch, a novel framework leveraging large models for zero-shot source search in web applications. AutoS$^2$earch operates on a simplified visual environment projected through a web-based display, utilizing a chain-of-thought prompt designed to emulate human reasoning. The multi-modal large language model (MLLMs) dynamically converts visual observations into language descriptions, enabling the LLM to perform linguistic reasoning on four directional choices. Extensive experiments demonstrate that AutoS$^2$earch achieves performance nearly equivalent to human-AI collaborative source search while eliminating dependency on crowdsourced labor. Our work offers valuable insights in using web engineering to design such autonomous systems in other industrial applications. 

**Abstract (ZH)**: 基于Web的管理系统在风险控制和工业安全领域得到了广泛应用。然而，有效地将源搜索能力整合进这些系统，以使决策者能够定位并解决安全隐患（例如，气体泄漏检测）仍然具有挑战性。虽然先前的努力探索了使用网络众包和AI算法来支持源搜索决策，但这些方法在招募人类参与者方面存在负担，并且在时间敏感的情况下响应速度较慢。为解决这一问题，我们提出了一种名为AutoS$^2$earch的新颖框架，该框架利用大规模模型在Web应用程序中进行零样本源搜索。AutoS$^2$earch在基于Web的显示中应用简化视觉环境，利用一个链式思考提示来模仿人类推理。多模态大规模语言模型（Multi-Modal Large Language Models, MLLMs）动态地将视觉观察转化为语言描述，从而允许LLM在四个方向选择上进行语言推理。大量的实验表明，AutoS$^2$earch在性能上几乎与人类-AI协作的源搜索相当，并且消除了对众包劳动力的依赖。我们的研究为利用Web工程设计此类自主系统提供了宝贵见解，在其他工业应用中也具有重要意义。 

---
# LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs - No Silver Bullet for LC or RAG Routing 

**Title (ZH)**: LaRA: 检验检索增强生成和长上下文语言模型 - 不存在适用于长上下文或检索增强生成路由的万能解决方案 

**Authors**: Kuan Li, Liwen Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Shuai Wang, Minhao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.09977)  

**Abstract**: Effectively incorporating external knowledge into Large Language Models (LLMs) is crucial for enhancing their capabilities and addressing real-world needs. Retrieval-Augmented Generation (RAG) offers an effective method for achieving this by retrieving the most relevant fragments into LLMs. However, the advancements in context window size for LLMs offer an alternative approach, raising the question of whether RAG remains necessary for effectively handling external knowledge. Several existing studies provide inconclusive comparisons between RAG and long-context (LC) LLMs, largely due to limitations in the benchmark designs. In this paper, we present LaRA, a novel benchmark specifically designed to rigorously compare RAG and LC LLMs. LaRA encompasses 2,326 test cases across four practical QA task categories and three types of naturally occurring long texts. Through systematic evaluation of seven open-source and four proprietary LLMs, we find that the optimal choice between RAG and LC depends on a complex interplay of factors, including the model's parameter size, long-text capabilities, context length, task type, and the characteristics of the retrieved chunks. Our findings provide actionable guidelines for practitioners to effectively leverage both RAG and LC approaches in developing and deploying LLM applications. Our code and dataset is provided at: \href{this https URL}{\textbf{this https URL}}. 

**Abstract (ZH)**: 有效地将外部知识融入大型语言模型（LLMs）对于增强其能力和应对实际需求至关重要。检索增强生成（RAG）提供了一种有效的方法，通过检索最相关的片段来增强LLMs。然而，LLMs的上下文窗口大小的进展提供了一种替代方法，这引发了是否RAG对于有效处理外部知识仍然必要的疑问。现有的一些研究对RAG和长上下文（LC）LLMs之间的比较结果并不明确，这主要是由于基准设计的局限性。在本文中，我们提出了LaRA，这是一种新的基准工具，专门设计用于严格比较RAG和LC LLMs。LaRA涵盖了四大类实际问答任务的2,326个测试案例，以及三种类型的自然生成的长文本。通过系统评估七种开源和四种专有LLM，我们发现，RAG和LC之间的最优选择取决于一系列复杂因素的相互作用，包括模型的参数量、长文本处理能力、上下文长度、任务类型以及检索片段的特性。我们的研究结果为实践者提供了关于如何有效地在开发和部署LLM应用程序时利用RAG和LC方法的实用指南。我们的代码和数据集可在此处获取：\href{this https URL}{此链接}。 

---
# AI-VERDE: A Gateway for Egalitarian Access to Large Language Model-Based Resources For Educational Institutions 

**Title (ZH)**: AI-VERDE：一种面向教育机构的大语言模型资源平等访问网关 

**Authors**: Paul Mithun, Enrique Noriega-Atala, Nirav Merchant, Edwin Skidmore  

**Link**: [PDF](https://arxiv.org/pdf/2502.09651)  

**Abstract**: We present AI-VERDE, a unified LLM-as-a-platform service designed to facilitate seamless integration of commercial, cloud-hosted, and on-premise open LLMs in academic settings. AI-VERDE streamlines access management for instructional and research groups by providing features such as robust access control, privacy-preserving mechanisms, native Retrieval-Augmented Generation (RAG) support, budget management for third-party LLM services, and both a conversational web interface and API access. In a pilot deployment at a large public university, AI-VERDE demonstrated significant engagement across diverse educational and research groups, enabling activities that would typically require substantial budgets for commercial LLM services with limited user and team management capabilities. To the best of our knowledge, AI-Verde is the first platform to address both academic and research needs for LLMs within an higher education institutional framework. 

**Abstract (ZH)**: 我们将介绍AI-VERDE，这是一种统一的基于大语言模型（LLM）的平台服务，旨在促进商业、云托管以及本地开放LLM在学术环境中的无缝集成。AI-VERDE通过提供诸如强大的访问控制、隐私保护机制、原生的检索增强生成（RAG）支持、第三方LLM服务的预算管理以及会话式的Web界面和API访问等功能，简化了教学和研究团队的访问管理。在一所大型公立大学的试点部署中，AI-VERDE展示了在各类教育和研究团体中显著的参与度，使得原本需要大量预算的商业LLM服务下的活动变得可行，并且具有有限用户和团队管理功能。据我们所知，AI-VERDE是首款在高等教育机构框架内同时满足学术研究和研究需求的平台。 

---
