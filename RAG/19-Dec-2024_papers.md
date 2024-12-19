# Enhancing Rhetorical Figure Annotation: An Ontology-Based Web Application with RAG Integration 

**Title (ZH)**: 基于本体的增强修辞 figura 注释：结合RAG的网络应用 

**Authors**: Ramona Kühn, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2412.13799)  

**Abstract**: Rhetorical figures play an important role in our communication. They are used to convey subtle, implicit meaning, or to emphasize statements. We notice them in hate speech, fake news, and propaganda. By improving the systems for computational detection of rhetorical figures, we can also improve tasks such as hate speech and fake news detection, sentiment analysis, opinion mining, or argument mining. Unfortunately, there is a lack of annotated data, as well as qualified annotators that would help us build large corpora to train machine learning models for the detection of rhetorical figures. The situation is particularly difficult in languages other than English, and for rhetorical figures other than metaphor, sarcasm, and irony. To overcome this issue, we develop a web application called "Find your Figure" that facilitates the identification and annotation of German rhetorical figures. The application is based on the German Rhetorical ontology GRhOOT which we have specially adapted for this purpose. In addition, we improve the user experience with Retrieval Augmented Generation (RAG). In this paper, we present the restructuring of the ontology, the development of the web application, and the built-in RAG pipeline. We also identify the optimal RAG settings for our application. Our approach is one of the first to practically use rhetorical ontologies in combination with RAG and shows promising results. 

**Abstract (ZH)**: 修辞手法在我们的沟通中发挥着重要作用。它们用于传达微妙的暗示意义，或强调陈述。我们在仇恨言论、假新闻和宣传中都能注意到它们的存在。通过改进修辞手法计算检测系统，我们也可以提升仇恨言论和假新闻检测、情感分析、观点挖掘或论点挖掘等任务。不幸的是，标注数据和合格的标注者仍然缺乏，这阻碍了我们构建用于修辞手法检测的大型语料库以训练机器学习模型。在这种情况下，情况尤其困难，尤其是在非英语语言中，以及对于除了隐喻、讽刺和幽默之外的其他修辞手法。

为了解决这一问题，我们开发了一个名为“Find your Figure”的网络应用，旨在简化德语修辞手法的识别和标注过程。该应用基于专门为此改编的德语修辞本体GRhOOT。此外，我们通过检索增强生成（RAG）技术改进了用户体验。在本文中，我们介绍了本体的重构、网络应用的开发以及内置的RAG管道。我们还确定了最适合我们应用的RAG设置。我们的方法是第一个实际利用本体与RAG相结合的方法之一，并显示出令人鼓舞的结果。 

---
# RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment 

**Title (ZH)**: RAG-RewardBench：检索增强生成中的奖励模型基准测试，用于偏好对齐 

**Authors**: Zhuoran Jin, Hongbang Yuan, Tianyi Men, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13746)  

**Abstract**: Despite the significant progress made by existing retrieval augmented language models (RALMs) in providing trustworthy responses and grounding in reliable sources, they often overlook effective alignment with human preferences. In the alignment process, reward models (RMs) act as a crucial proxy for human values to guide optimization. However, it remains unclear how to evaluate and select a reliable RM for preference alignment in RALMs. To this end, we propose RAG-RewardBench, the first benchmark for evaluating RMs in RAG settings. First, we design four crucial and challenging RAG-specific scenarios to assess RMs, including multi-hop reasoning, fine-grained citation, appropriate abstain, and conflict robustness. Then, we incorporate 18 RAG subsets, six retrievers, and 24 RALMs to increase the diversity of data sources. Finally, we adopt an LLM-as-a-judge approach to improve preference annotation efficiency and effectiveness, exhibiting a strong correlation with human annotations. Based on the RAG-RewardBench, we conduct a comprehensive evaluation of 45 RMs and uncover their limitations in RAG scenarios. Additionally, we also reveal that existing trained RALMs show almost no improvement in preference alignment, highlighting the need for a shift towards preference-aligned this http URL release our benchmark and code publicly at this https URL for future work. 

**Abstract (ZH)**: 尽管现有的检索增强语言模型（RALMs）在提供可靠响应和基于可靠来源的基础上取得了显著进展，但在有效与人类偏好对齐方面，它们常常有所忽视。在对齐过程中，奖励模型（RMs）作为一种关键代理，用于指引优化过程，反映人类价值观。然而，如何评价和选择适用于RALMs的可靠RM以实现偏好对齐仍然是一个未解之谜。为此，我们提出RAG-RewardBench，这是首个在检索增强（RAG）环境中评估RM基准。首先，我们设计了四个关键且具有挑战性的RAG特定场景，以评估RM，包括多跳推理、精细引文、适当地保持中立以及冲突稳健性。其次，我们整合了18个RAG子集、六种检索器和24种RALMs，以增加数据源的多样性。最后，我们采用LLM作为裁判的方法，以提高偏好注解的效率和有效性，表现出与人类注解的强烈相关性。基于RAG-RewardBench，我们对45种RM进行了全面评估，并发现了它们在RAG场景下的局限性。此外，我们还揭示了现有的训练过的RALMs在偏好对齐方面几乎没有改进，突显了向偏好对齐转变的必要性。我们将在https://your-public-url发布这个基准和代码，以便未来的研究工作使用。 

---
# Federated Learning and RAG Integration: A Scalable Approach for Medical Large Language Models 

**Title (ZH)**: 联邦学习与RAG集成：一种适用于医疗大型语言模型的可扩展方法 

**Authors**: Jincheol Jung, Hongju Jeong, Eui-Nam Huh  

**Link**: [PDF](https://arxiv.org/pdf/2412.13720)  

**Abstract**: This study analyzes the performance of domain-specific Large Language Models (LLMs) for the medical field by integrating Retrieval-Augmented Generation (RAG) systems within a federated learning framework. Leveraging the inherent advantages of federated learning, such as preserving data privacy and enabling distributed computation, this research explores the integration of RAG systems with models trained under varying client configurations to optimize performance. Experimental results demonstrate that the federated learning-based models integrated with RAG systems consistently outperform their non-integrated counterparts across all evaluation metrics. This study highlights the potential of combining federated learning and RAG systems for developing domain-specific LLMs in the medical field, providing a scalable and privacy-preserving solution for enhancing text generation capabilities. 

**Abstract (ZH)**: 本文通过在联邦学习框架中整合检索增强生成（RAG）系统，分析了针对医疗领域的专用大型语言模型（LLMs）的性能。利用联邦学习的固有优势，如保护数据隐私和实现分布式计算，本研究探讨了在不同客户端配置下将RAG系统与训练模型进行集成，以优化性能的可能性。实验结果表明，基于联邦学习并与RAG系统集成的模型在所有评估指标上均优于未集成的模型。本文强调了将联邦学习与RAG系统相结合，开发医疗领域的专用LLMs的潜力，提供了一种可扩展且保护隐私的解决方案，以增强文本生成能力。 

---
# RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service 

**Title (ZH)**: RemoteRAG：一种隐私 preservation 的云大语言模型检索即服务（RAG）服务 

**Authors**: Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan, Yunhao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2412.12775)  

**Abstract**: Retrieval-augmented generation (RAG) improves the service quality of large language models by retrieving relevant documents from credible literature and integrating them into the context of the user query. Recently, the rise of the cloud RAG service has made it possible for users to query relevant documents conveniently. However, directly sending queries to the cloud brings potential privacy leakage. In this paper, we are the first to formally define the privacy-preserving cloud RAG service to protect the user query and propose RemoteRAG as a solution regarding privacy, efficiency, and accuracy. For privacy, we introduce $(n,\epsilon)$-DistanceDP to characterize privacy leakage of the user query and the leakage inferred from relevant documents. For efficiency, we limit the search range from the total documents to a small number of selected documents related to a perturbed embedding generated from $(n,\epsilon)$-DistanceDP, so that computation and communication costs required for privacy protection significantly decrease. For accuracy, we ensure that the small range includes target documents related to the user query with detailed theoretical analysis. Experimental results also demonstrate that RemoteRAG can resist existing embedding inversion attack methods while achieving no loss in retrieval under various settings. Moreover, RemoteRAG is efficient, incurring only $0.67$ seconds and $46.66$KB of data transmission ($2.72$ hours and $1.43$ GB with the non-optimized privacy-preserving scheme) when retrieving from a total of $10^6$ documents. 

**Abstract (ZH)**: 检索增强生成（RAG）通过从可信文献中检索相关文档并将其集成到用户的查询上下文中，从而提高大型语言模型的服务质量。最近，云RAG服务的兴起使得用户能够方便地查询相关文档。然而，直接将查询发送到云中会带来潜在的隐私泄露风险。本文首次正式定义了保护用户查询隐私的云RAG服务，并提出RemoteRAG作为在隐私、效率和准确性方面的一揽子解决方案。为了保护隐私，我们引入了$(n, \epsilon)$-DistanceDP来表征用户查询及其从相关文档推理出的隐私泄露。为了提高效率，我们将搜索范围限定在从$(n, \epsilon)$-DistanceDP生成的扰动嵌入中相关的少量文档上，从而显著降低用于隐私保护的计算和通信成本。为了保证准确性，我们通过详细的理论分析确保该小范围包括与用户查询相关的目标文档。实验结果还表明，RemoteRAG能够在各种设置下抵御现有的嵌入反向攻击方法，并且在检索损失方面的表现与现有方法相当。此外，当从总共$10^6$份文档中检索时，RemoteRAG表现出高效性，仅需0.67秒和46.66KB的数据传输量（在未优化的隐私保护方案下分别为2.72小时和1.43GB）。 

---
# C-FedRAG: A Confidential Federated Retrieval-Augmented Generation System 

**Title (ZH)**: C-FedRAG：一种保密联邦检索增强生成系统 

**Authors**: Parker Addison, Minh-Tuan H. Nguyen, Tomislav Medan, Mohammad T. Manzari, Brendan McElrone, Laksh Lalwani, Aboli More, Smita Sharma, Holger R. Roth, Isaac Yang, Chester Chen, Daguang Xu, Yan Cheng, Andrew Feng, Ziyue Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13163)  

**Abstract**: Organizations seeking to utilize Large Language Models (LLMs) for knowledge querying and analysis often encounter challenges in maintaining an LLM fine-tuned on targeted, up-to-date information that keeps answers relevant and grounded. Retrieval Augmented Generation (RAG) has quickly become a feasible solution for organizations looking to overcome the challenges of maintaining proprietary models and to help reduce LLM hallucinations in their query responses. However, RAG comes with its own issues regarding scaling data pipelines across tiered-access and disparate data sources. In many scenarios, it is necessary to query beyond a single data silo to provide richer and more relevant context for an LLM. Analyzing data sources within and across organizational trust boundaries is often limited by complex data-sharing policies that prohibit centralized data storage, therefore, inhibit the fast and effective setup and scaling of RAG solutions. In this paper, we introduce Confidential Computing (CC) techniques as a solution for secure Federated Retrieval Augmented Generation (FedRAG). Our proposed Confidential FedRAG system (C-FedRAG) enables secure connection and scaling of a RAG workflows across a decentralized network of data providers by ensuring context confidentiality. We also demonstrate how to implement a C-FedRAG system using the NVIDIA FLARE SDK and assess its performance using the MedRAG toolkit and MIRAGE benchmarking dataset. 

**Abstract (ZH)**: 组织希望利用大规模语言模型（LLMs）进行知识查询和分析时，常常面临维护一个针对特定且最新信息进行微调的LLM的挑战，这会导致答案的相关性和现实性受损。检索增强生成（RAG）迅速成为克服维护专有模型挑战和降低LLM在查询响应中产生幻觉可能性的可行解决方案。然而，RAG在大规模数据管道跨分层访问和不同数据源扩展时存在自身的问题。在许多情况下，为了提供更丰富和相关的情境，需要跨多个数据孤岛进行查询。分析组织信任边界内的数据源和跨边界的数据源常常受限于复杂的数据共享政策，这些政策禁止集中存储数据，从而阻碍了RAG解决方案的快速有效设置与扩展。在本文中，我们介绍保密计算（CC）技术作为一种安全联邦检索增强生成（FedRAG）的解决方案。我们提出的保密联邦RAG系统（C-FedRAG）通过确保上下文保密性，在分散的数据提供者网络中安全连接并扩展RAG工作流程。我们还展示了如何使用NVIDIA FLARE SDK实现C-FedRAG系统，并使用MedRAG工具包和MIRAGE基准测试数据集评估其性能。 

---
