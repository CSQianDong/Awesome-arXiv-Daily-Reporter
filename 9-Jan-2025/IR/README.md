# Knowledge Retrieval Based on Generative AI 

**Title (ZH)**: 基于生成式人工智能的知识检索 

**Authors**: Te-Lun Yang, Jyi-Shane Liu, Yuen-Hsien Tseng, Jyh-Shing Roger Jang  

**Link**: [PDF](https://arxiv.org/pdf/2501.04635)  

**Abstract**: This study develops a question-answering system based on Retrieval-Augmented Generation (RAG) using Chinese Wikipedia and Lawbank as retrieval sources. Using TTQA and TMMLU+ as evaluation datasets, the system employs BGE-M3 for dense vector retrieval to obtain highly relevant search results and BGE-reranker to reorder these results based on query relevance. The most pertinent retrieval outcomes serve as reference knowledge for a Large Language Model (LLM), enhancing its ability to answer questions and establishing a knowledge retrieval system grounded in generative AI.
The system's effectiveness is assessed through a two-stage evaluation: automatic and assisted performance evaluations. The automatic evaluation calculates accuracy by comparing the model's auto-generated labels with ground truth answers, measuring performance under standardized conditions without human intervention. The assisted performance evaluation involves 20 finance-related multiple-choice questions answered by 20 participants without financial backgrounds. Initially, participants answer independently. Later, they receive system-generated reference information to assist in answering, examining whether the system improves accuracy when assistance is provided.
The main contributions of this research are: (1) Enhanced LLM Capability: By integrating BGE-M3 and BGE-reranker, the system retrieves and reorders highly relevant results, reduces hallucinations, and dynamically accesses authorized or public knowledge sources. (2) Improved Data Privacy: A customized RAG architecture enables local operation of the LLM, eliminating the need to send private data to external servers. This approach enhances data security, reduces reliance on commercial services, lowers operational costs, and mitigates privacy risks. 

**Abstract (ZH)**: 本研究基于检索增强生成（RAG）方法，开发了一个基于中文维基百科和Lawbank的问答系统。该系统使用TTQA和TMMLU+作为评估数据集，采用BGE-M3进行密集向量检索，以获得高度相关的结果，同时使用BGE-reranker根据查询相关性对这些结果进行重新排序。最相关的检索结果作为大型语言模型（LLM）的知识参考，提升其回答问题的能力，并构建了基于生成AI的知识检索系统。

该系统的有效性通过两阶段评估进行检验：自动评估和辅助评估。自动评估通过比较模型自动生成的标签与真实答案，计算准确率，在标准化条件下无需人工干预进行性能评估。辅助评估包括20道与金融相关的选择题，由20位不具备金融背景的参与者回答。最初，参与者独立作答；随后，他们收到系统生成的参考信息来辅助答题，以检验系统在提供辅助信息时能否提高准确性。

本研究的主要贡献包括：（1）增强的LLM能力：通过整合BGE-M3和BGE-reranker，系统能够检索和重新排序高度相关的结果，减少幻觉现象，并动态访问授权或公开的知识来源。（2）改进的数据隐私：定制的RAG架构使LLM可以本地运行，无需将私人数据发送到外部服务器。这种方法提高了数据安全性，减少了对商业化服务的依赖，降低了运营成本，并减少了隐私风险。 

---
# Evaluating Interval-based Tokenization for Pitch Representation in Symbolic Music Analysis 

**Title (ZH)**: 基于区间划分的令牌化方法在符号音乐分析中的音高表示评估 

**Authors**: Dinh-Viet-Toan Le, Louis Bigo, Mikaela Keller  

**Link**: [PDF](https://arxiv.org/pdf/2501.04630)  

**Abstract**: Symbolic music analysis tasks are often performed by models originally developed for Natural Language Processing, such as Transformers. Such models require the input data to be represented as sequences, which is achieved through a process of tokenization. Tokenization strategies for symbolic music often rely on absolute MIDI values to represent pitch information. However, music research largely promotes the benefit of higher-level representations such as melodic contour and harmonic relations for which pitch intervals turn out to be more expressive than absolute pitches. In this work, we introduce a general framework for building interval-based tokenizations. By evaluating these tokenizations on three music analysis tasks, we show that such interval-based tokenizations improve model performances and facilitate their explainability. 

**Abstract (ZH)**: 符号音乐分析任务通常由原本用于自然语言处理（NLP）的模型，如变换器（Transformers）来执行。这些模型需要输入数据以序列形式表示，这一过程通过分词化实现。符号音乐的分词化策略通常依赖于绝对MIDI值来表示音高信息。然而，音乐研究普遍认为，如旋律轮廓和和声关系这类更高层次的表示更有优势，这些表示中音程比绝对音高更为表达力强。在本文中，我们提出了一种基于音程的分词化普遍框架。通过在三个音乐分析任务上评估这些分词化方法，我们展示了基于音程的分词化能够提高模型性能并增强模型的可解释性。 

---
# A Closer Look on Gender Stereotypes in Movie Recommender Systems and Their Implications with Privacy 

**Title (ZH)**: 对电影推荐系统中性别刻板印象的进一步探究及其对隐私的影响 

**Authors**: Falguni Roy, Yiduo Shen, Na Zhao, Xiaofeng Ding, Md. Omar Faruk  

**Link**: [PDF](https://arxiv.org/pdf/2501.04420)  

**Abstract**: The movie recommender system typically leverages user feedback to provide personalized recommendations that align with user preferences and increase business revenue. This study investigates the impact of gender stereotypes on such systems through a specific attack scenario. In this scenario, an attacker determines users' gender, a private attribute, by exploiting gender stereotypes about movie preferences and analyzing users' feedback data, which is either publicly available or observed within the system. The study consists of two phases. In the first phase, a user study involving 630 participants identified gender stereotypes associated with movie genres, which often influence viewing choices. In the second phase, four inference algorithms were applied to detect gender stereotypes by combining the findings from the first phase with users' feedback data. Results showed that these algorithms performed more effectively than relying solely on feedback data for gender inference. Additionally, we quantified the extent of gender stereotypes to evaluate their broader impact on digital computational science. The latter part of the study utilized two major movie recommender datasets: MovieLens 1M and Yahoo!Movie. Detailed experimental information is available on our GitHub repository: this https URL 

**Abstract (ZH)**: 电影推荐系统通常通过利用用户反馈来提供个性化推荐，以满足用户偏好并增加商业收益。本研究通过特定的攻击场景，探讨性别刻板印象对这类系统的影响。在该攻击场景中，攻击者通过利用关于电影偏好的性别刻板印象以及分析用户反馈数据（这些数据可能是公开的，也可能是在系统内部观察到的），来确定用户的性别，这是一个私有属性。本研究分为两个阶段。在第一阶段，涉及630名参与者的用户研究确定了与电影类型相关的性别刻板印象，这些刻板印象通常会影响观影选择。在第二阶段，四种推理算法被应用于通过结合第一阶段的发现与用户反馈数据来检测性别刻板印象。结果表明，这些算法在性别推断方面比仅依赖反馈数据更为有效。此外，我们还量化了性别刻板印象的程度，以评估它们对数字计算科学的更广泛影响。研究的后一部分使用了两个主要的电影推荐数据集：MovieLens 1M和Yahoo！Movie。详细的实验信息可在我们的GitHub仓库中找到：[这里](this https URL) 

---
# An innovative data collection method to eliminate the preprocessing phase in web usage mining 

**Title (ZH)**: 一种创新的数据采集方法，用于在网络使用挖掘中消除预处理阶段 

**Authors**: Ozkan Canay, Umit Kocabicak  

**Link**: [PDF](https://arxiv.org/pdf/2501.04364)  

**Abstract**: The underlying data source for web usage mining (WUM) is commonly thought to be server logs. However, access log files ensure quite limited data about the clients. Identifying sessions from this messy data takes a considerable effort, and operations performed for this purpose do not always yield excellent results. Also, this data cannot be used for web analytics efficiently. This study proposes an innovative method for user tracking, session management, and collecting web usage data. The method is mainly based on a new approach for using collected data for web analytics extraction as the data source in web usage mining. An application-based API has been developed with a different strategy from conventional client-side methods to obtain and process log data. The log data has been successfully gathered by integrating the technique into an enterprise web application. The results reveal that the homogeneous structured data collected and stored with this method is more convenient to browse, filter, and process than web server logs. This data stored on a relational database can be used effortlessly as a reliable data source for high-performance web usage mining activity, real-time web analytics, or a functional recommendation system. 

**Abstract (ZH)**: 网络使用挖掘（Web Usage Mining, WUM）通常被认为依赖于服务器日志作为底层数据源。然而，访问日志文件仅提供关于客户端的有限信息。从这些杂乱的数据中识别会话需要相当大的努力，而为此目的进行的操作并不总是能够获得理想的结果。此外，这种数据也不能高效地用于网站分析。本研究提出了一种创新的方法，用于用户追踪、会话管理以及收集网络使用数据。该方法主要基于一种新的方法，即利用收集的数据作为网络使用挖掘的数据源，以提取网络分析信息。本研究开发了一种基于应用的API，采用了一种不同于传统客户端方法的不同策略来获取和处理日志数据。通过将该技术集成到企业网络应用程序中，成功收集了日志数据。研究结果表明，使用此方法收集和存储的同质结构化数据比服务器日志更容易浏览、过滤和处理。存储在关系数据库中的这些数据可以轻松地作为高性能网络使用挖掘活动、实时网站分析或功能推荐系统的可靠数据源。 

---
# Advancing Similarity Search with GenAI: A Retrieval Augmented Generation Approach 

**Title (ZH)**: 利用生成式人工智能推进相似性搜索：一种检索增强生成方法 

**Authors**: Jean Bertin  

**Link**: [PDF](https://arxiv.org/pdf/2501.04006)  

**Abstract**: This article introduces an innovative Retrieval Augmented Generation approach to similarity search. The proposed method uses a generative model to capture nuanced semantic information and retrieve similarity scores based on advanced context understanding. The study focuses on the BIOSSES dataset containing 100 pairs of sentences extracted from the biomedical domain, and introduces similarity search correlation results that outperform those previously attained on this dataset. Through an in-depth analysis of the model sensitivity, the research identifies optimal conditions leading to the highest similarity search accuracy: the results reveals high Pearson correlation scores, reaching specifically 0.905 at a temperature of 0.5 and a sample size of 20 examples provided in the prompt. The findings underscore the potential of generative models for semantic information retrieval and emphasize a promising research direction to similarity search. 

**Abstract (ZH)**: 本文介绍了用于相似性搜索的一种创新性检索增强生成（Retrieval Augmented Generation, RAG）方法。所提出的方案利用生成模型捕获微妙的语义信息，并基于先进的上下文理解来检索相似性得分。研究重点在于BIOSSES数据集，该数据集包含100个医学领域句子配对，展示了优于该数据集此前成果的相似性搜索相关性结果。通过深入分析模型的敏感性，研究确定了导致最高相似性搜索准确率的最优条件：结果揭示出高皮尔逊相关系数，特别是在温度为0.5、提示中提供20个示例的情况下，具体相关系数达到了0.905。研究结果突显了生成模型在语义信息检索方面的潜力，并强调了为相似性搜索探索有前景的研究方向的重要性。 

---
# Re-ranking the Context for Multimodal Retrieval Augmented Generation 

**Title (ZH)**: 多模态检索增强生成中重构上下文的重新排名 

**Authors**: Matin Mortaheb, Mohammad A. Amir Khojastepour, Srimat T. Chakradhar, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2501.04695)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge to generate a response within a context with improved accuracy and reduced hallucinations. However, multi-modal RAG systems face unique challenges: (i) the retrieval process may select irrelevant entries to user query (e.g., images, documents), and (ii) vision-language models or multi-modal language models like GPT-4o may hallucinate when processing these entries to generate RAG output. In this paper, we aim to address the first challenge, i.e, improving the selection of relevant context from the knowledge-base in retrieval phase of the multi-modal RAG. Specifically, we leverage the relevancy score (RS) measure designed in our previous work for evaluating the RAG performance to select more relevant entries in retrieval process. The retrieval based on embeddings, say CLIP-based embedding, and cosine similarity usually perform poorly particularly for multi-modal data. We show that by using a more advanced relevancy measure, one can enhance the retrieval process by selecting more relevant pieces from the knowledge-base and eliminate the irrelevant pieces from the context by adaptively selecting up-to-$k$ entries instead of fixed number of entries. Our evaluation using COCO dataset demonstrates significant enhancement in selecting relevant context and accuracy of the generated response. 

**Abstract (ZH)**: 检索增强生成（RAG）通过融入外部知识，增强了大型语言模型（LLMs）在上下文中生成响应的准确性和降低了虚构现象。然而，多模态RAG系统面临独特的挑战：（i）检索过程可能会选择与用户查询无关的条目（例如，图像、文档），以及（ii）当使用如GPT-4o等视觉语言模型或多模态语言模型处理这些条目生成RAG输出时，可能会产生虚构现象。本文旨在解决第一个挑战，即在多模态RAG的检索阶段，改进从知识库中选择相关上下文的方式。具体来说，我们利用我们在先前工作中设计的相关性分数（RS）衡量方法来评估RAG性能，从而在检索过程中选择更多相关内容。基于嵌入（如CLIP嵌入）和余弦相似性的检索通常在多模态数据上表现不佳。我们表明，通过使用更高级的相关性衡量方法，可以在检索过程中选择更多相关内容，并通过适配性地选择最多$k$个条目而不是固定数量的条目，来消除上下文中的无关条目。使用COCO数据集的评估结果表明，这种改进方法显著提高了选择相关上下文和生成响应准确性。 

---
# Multi-task retriever fine-tuning for domain-specific and efficient RAG 

**Title (ZH)**: 针对特定领域并实现高效RAG的多任务检索器微调 

**Authors**: Patrice Béchard, Orlando Marquez Ayala  

**Link**: [PDF](https://arxiv.org/pdf/2501.04652)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become ubiquitous when deploying Large Language Models (LLMs), as it can address typical limitations such as generating hallucinated or outdated information. However, when building real-world RAG applications, practical issues arise. First, the retrieved information is generally domain-specific. Since it is computationally expensive to fine-tune LLMs, it is more feasible to fine-tune the retriever to improve the quality of the data included in the LLM input. Second, as more applications are deployed in the same real-world system, one cannot afford to deploy separate retrievers. Moreover, these RAG applications normally retrieve different kinds of data. Our solution is to instruction fine-tune a small retriever encoder on a variety of domain-specific tasks to allow us to deploy one encoder that can serve many use cases, thereby achieving low-cost, scalability, and speed. We show how this encoder generalizes to out-of-domain settings as well as to an unseen retrieval task on real-world enterprise use cases. 

**Abstract (ZH)**: 检索增强生成（RAG）在部署大型语言模型（LLMs）时变得无处不在，因为它可以解决生成虚构或过时信息等典型限制。然而，在构建实际应用的RAG系统时，一些实际问题随之出现。首先，检索到的信息通常是特定领域的。由于对LLMs进行微调计算成本较高，更实际的做法是微调检索器以提高输入LLMs中的数据质量。其次，随着在同一个实际系统中部署越来越多的应用，无法部署分离的检索器。此外，这些RAG应用通常检索不同类型的数据库信息。我们的解决方案是针对多种特定领域的任务对一个小规模的检索器编码器进行指令微调，以便我们可以部署一个编码器服务于多种应用场景，从而实现低成本、可扩展性和高速度。我们展示了在实际企业应用中，该编码器如何在领域外场景以及新的检索任务中泛化能力。 

---
# User Simulation in the Era of Generative AI: User Modeling, Synthetic Data Generation, and System Evaluation 

**Title (ZH)**: 生成式AI时代中的用户模拟：用户建模、合成数据生成与系统评估 

**Authors**: Krisztian Balog, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2501.04410)  

**Abstract**: User simulation is an emerging interdisciplinary topic with multiple critical applications in the era of Generative AI. It involves creating an intelligent agent that mimics the actions of a human user interacting with an AI system, enabling researchers to model and analyze user behaviour, generate synthetic data for training, and evaluate interactive AI systems in a controlled and reproducible manner. User simulation has profound implications for diverse fields and plays a vital role in the pursuit of Artificial General Intelligence. This paper provides an overview of user simulation, highlighting its key applications, connections to various disciplines, and outlining future research directions to advance this increasingly important technology. 

**Abstract (ZH)**: 用户模拟是生成式人工智能时代的一个跨学科新兴话题，具有多种关键应用。它涉及创建一个模仿人类用户与人工智能系统交互行为的智能代理，从而使研究人员能够模拟和分析用户行为、生成用于训练的合成数据，并以可控和可重复的方式评估交互式人工智能系统。用户模拟对多个领域具有深远影响，并在追求通用人工智能的进程中扮演着至关重要的角色。本文对用户模拟进行了概述，强调其关键应用、与各学科的联系，并概述了未来的研究方向，以推进这一日益重要的技术。 

---
# Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation 

**Title (ZH)**: 增强推理的自我训练方法用于长格式个性化文本生成 

**Authors**: Alireza Salemi, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Weize Kong, Tao Chen, Zhuowan Li, Michael Bendersky, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2501.04167)  

**Abstract**: Personalized text generation requires a unique ability of large language models (LLMs) to learn from context that they often do not encounter during their standard training. One way to encourage LLMs to better use personalized context for generating outputs that better align with the user's expectations is to instruct them to reason over the user's past preferences, background knowledge, or writing style. To achieve this, we propose Reasoning-Enhanced Self-Training for Personalized Text Generation (REST-PG), a framework that trains LLMs to reason over personal data during response generation. REST-PG first generates reasoning paths to train the LLM's reasoning abilities and then employs Expectation-Maximization Reinforced Self-Training to iteratively train the LLM based on its own high-reward outputs. We evaluate REST-PG on the LongLaMP benchmark, consisting of four diverse personalized long-form text generation tasks. Our experiments demonstrate that REST-PG achieves significant improvements over state-of-the-art baselines, with an average relative performance gain of 14.5% on the benchmark. 

**Abstract (ZH)**: 个性化文本生成需要大规模语言模型（LLMs）具备一种独特的能力，即从它们在标准训练过程中不太可能遇到的情境中学习。为了鼓励LLMs更好地利用个性化的上下文来生成更符合用户预期的输出，可以通过指示它们根据用户的过去偏好、背景知识或写作风格进行推理。为了实现这一点，我们提出了一种增强推理的自我训练框架（Reasoning-Enhanced Self-Training for Personalized Text Generation，简称REST-PG），该框架在响应生成过程中训练LLMs进行推理。REST-PG首先生成推理路径，以提高LLMs的推理能力，然后通过利用期望最大化强化自我训练的方法，基于模型的高奖励输出迭代训练LLMs。我们使用包含四个多样化的个性化长文本生成任务的LongLaMP基准测试REST-PG。实验结果显示，REST-PG在基准测试中显著优于当前最先进的基线模型，取得了平均14.5%的相对性能提升。 

---
# KGIF: Optimizing Relation-Aware Recommendations with Knowledge Graph Information Fusion 

**Title (ZH)**: KGIF：利用知识图谱信息融合优化关系感知推荐系统 

**Authors**: Dong Hyun Jeon, Wenbo Sun, Houbing Herbert Song, Dongfang Liu, Velasquez Alvaro, Yixin Chloe Xie, Shuteng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2501.04161)  

**Abstract**: While deep-learning-enabled recommender systems demonstrate strong performance benchmarks, many struggle to adapt effectively in real-world environments due to limited use of user-item relationship data and insufficient transparency in recommendation generation. Traditional collaborative filtering approaches fail to integrate multifaceted item attributes, and although Factorization Machines account for item-specific details, they overlook broader relational patterns. Collaborative knowledge graph-based models have progressed by embedding user-item interactions with item-attribute relationships, offering a holistic perspective on interconnected entities. However, these models frequently aggregate attribute and interaction data in an implicit manner, leaving valuable relational nuances underutilized.
This study introduces the Knowledge Graph Attention Network with Information Fusion (KGIF), a specialized framework designed to merge entity and relation embeddings explicitly through a tailored self-attention mechanism. The KGIF framework integrates reparameterization via dynamic projection vectors, enabling embeddings to adaptively represent intricate relationships within knowledge graphs. This explicit fusion enhances the interplay between user-item interactions and item-attribute relationships, providing a nuanced balance between user-centric and item-centric representations. An attentive propagation mechanism further optimizes knowledge graph embeddings, capturing multi-layered interaction patterns. The contributions of this work include an innovative method for explicit information fusion, improved robustness for sparse knowledge graphs, and the ability to generate explainable recommendations through interpretable path visualization. 

**Abstract (ZH)**: 虽然深度学习驱动的推荐系统在性能指标上表现出色，但许多系统在实际应用环境中难以有效适应，主要原因是对用户-项目关系数据的使用有限，以及推荐生成过程中的透明度不足。传统的协作过滤方法无法整合项目的多维度属性，而因子分解机虽然考虑了项目特有的细节，但忽略了更广泛的关联模式。基于协作知识图谱的模型通过嵌入用户-项目交互与项目属性之间的关系，提供了一种全面的实体相互作用视角。然而，这些模型经常以隐式方式汇总属性和交互数据，导致众多有价值的关联细微差别被浪费。

本研究引入了一种专门的框架——知识图注意力网络与信息融合（KGIF），该框架通过定制化的自注意力机制显式地将实体和关系嵌入进行融合。KGIF框架通过动态投影向量实现重新参数化，使嵌入能够自适应地表示知识图谱中的复杂关系。这种显式的融合增强了用户-项目交互与项目属性关系之间的交互作用，提供了用户中心和项目中心表示之间的精致平衡。进一步优化的知识图嵌入的注意力传播机制捕捉到了多层次的交互模式。本文的主要贡献包括一种创新的信息融合方法、对稀疏知识图谱的改进鲁棒性，以及通过可解释的路径可视化生成可解释的推荐。

这种翻译保留了原文的学术规范和专业术语，并确保了内容的准确传达。 

---
# A Generative AI-driven Metadata Modelling Approach 

**Title (ZH)**: 基于生成人工智能的元数据建模方法 

**Authors**: Mayukh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.04008)  

**Abstract**: Since decades, the modelling of metadata has been core to the functioning of any academic library. Its importance has only enhanced with the increasing pervasiveness of Generative Artificial Intelligence (AI)-driven information activities and services which constitute a library's outreach. However, with the rising importance of metadata, there arose several outstanding problems with the process of designing a library metadata model impacting its reusability, crosswalk and interoperability with other metadata models. This paper posits that the above problems stem from an underlying thesis that there should only be a few core metadata models which would be necessary and sufficient for any information service using them, irrespective of the heterogeneity of intra-domain or inter-domain settings. To that end, this paper advances a contrary view of the above thesis and substantiates its argument in three key steps. First, it introduces a novel way of thinking about a library metadata model as an ontology-driven composition of five functionally interlinked representation levels from perception to its intensional definition via properties. Second, it introduces the representational manifoldness implicit in each of the five levels which cumulatively contributes to a conceptually entangled library metadata model. Finally, and most importantly, it proposes a Generative AI-driven Human-Large Language Model (LLM) collaboration based metadata modelling approach to disentangle the entanglement inherent in each representation level leading to the generation of a conceptually disentangled metadata model. Throughout the paper, the arguments are exemplified by motivating scenarios and examples from representative libraries handling cancer information. 

**Abstract (ZH)**: 自几十年前起，元数据建模一直是任何学术图书馆运作的核心。随着生成性人工智能（AI）驱动的信息活动和服务的不断增加，这些活动和服务构成了图书馆的 outreach，元数据的重要性也得到了进一步提升。然而，随着元数据重要性的提升，设计图书馆元数据模型时出现了若干影响其重用性、元数据转换和与其他元数据模型的互操作性的问题。本文认为，这些问题源于一个核心论点，即仅需少数几个核心元数据模型就足以满足任何使用这些模型的信息服务的需求，无论是在领域内还是跨领域环境中存在何种异质性。为了应对这一情况，本文提出了一个对立的观点，并通过三个关键步骤来阐述这一观点。首先，本文引入了一种新的思维方式，即将图书馆元数据模型视为由感知到内涵定义的五个功能相连的表示层次构成的本体驱动组合。其次，本文提出在每个五个层次中都潜在存在着表示多样性，这些多样性累积起来构成了一个概念上交织的图书馆元数据模型。最后，也是最重要的是，本文提出了一种生成性AI驱动的人工智能大语言模型（LLM）协作的元数据建模方法，以拆解每个表示层次中固有的交织性，从而生成一个概念上拆解的元数据模型。在整个论文中，通过代表图书馆处理癌症信息的相关情境和例子展示了这些论点。 

---
