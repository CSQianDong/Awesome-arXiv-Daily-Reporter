# Enhancing Rhetorical Figure Annotation: An Ontology-Based Web Application with RAG Integration 

**Title (ZH)**: 基于本体的集成RAG技术的修辞手法标注增强：一个网络应用系统 

**Authors**: Ramona Kühn, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2412.13799)  

**Abstract**: Rhetorical figures play an important role in our communication. They are used to convey subtle, implicit meaning, or to emphasize statements. We notice them in hate speech, fake news, and propaganda. By improving the systems for computational detection of rhetorical figures, we can also improve tasks such as hate speech and fake news detection, sentiment analysis, opinion mining, or argument mining. Unfortunately, there is a lack of annotated data, as well as qualified annotators that would help us build large corpora to train machine learning models for the detection of rhetorical figures. The situation is particularly difficult in languages other than English, and for rhetorical figures other than metaphor, sarcasm, and irony. To overcome this issue, we develop a web application called "Find your Figure" that facilitates the identification and annotation of German rhetorical figures. The application is based on the German Rhetorical ontology GRhOOT which we have specially adapted for this purpose. In addition, we improve the user experience with Retrieval Augmented Generation (RAG). In this paper, we present the restructuring of the ontology, the development of the web application, and the built-in RAG pipeline. We also identify the optimal RAG settings for our application. Our approach is one of the first to practically use rhetorical ontologies in combination with RAG and shows promising results. 

**Abstract (ZH)**: 修辞手法在我们的交流中发挥着重要作用。它们用于传达微妙的隐含意义，或强调陈述。我们在仇恨言论、假新闻和宣传中注意到它们。通过改进计算检测修辞手法的系统，我们也可以提高仇恨言论和假新闻检测、情感分析、意见挖掘或论点挖掘等任务的性能。不幸的是，目前缺乏标注数据，以及合格的标注人员来帮助我们构建大型语料库进行机器学习模型训练以检测修辞手法。这种情况在除了英语之外的语言中尤其困难，对于除了比喻、讽刺和 irony 之外的其他修辞手法，情况更加复杂。为了解决这一问题，我们开发了一个名为“Find your Figure”的网络应用程序，以简化德语修辞手法的识别和标注。该应用程序基于我们特别为这一目的改编的德语修辞本体论GRhOOT。此外，我们通过检索增强生成（RAG）提升了用户体验。在本文中，我们介绍了本体论的重新构建、网络应用程序的开发以及内置的RAG管道。我们还确定了适用于我们应用程序的最佳RAG设置。我们的方法是第一个在实际中将修辞本体论与RAG相结合的方法，并且显示了有希望的结果。 

---
# RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment 

**Title (ZH)**: RAG-RewardBench：用于偏好对齐的检索增强生成中奖励模型的基准测试 

**Authors**: Zhuoran Jin, Hongbang Yuan, Tianyi Men, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.13746)  

**Abstract**: Despite the significant progress made by existing retrieval augmented language models (RALMs) in providing trustworthy responses and grounding in reliable sources, they often overlook effective alignment with human preferences. In the alignment process, reward models (RMs) act as a crucial proxy for human values to guide optimization. However, it remains unclear how to evaluate and select a reliable RM for preference alignment in RALMs. To this end, we propose RAG-RewardBench, the first benchmark for evaluating RMs in RAG settings. First, we design four crucial and challenging RAG-specific scenarios to assess RMs, including multi-hop reasoning, fine-grained citation, appropriate abstain, and conflict robustness. Then, we incorporate 18 RAG subsets, six retrievers, and 24 RALMs to increase the diversity of data sources. Finally, we adopt an LLM-as-a-judge approach to improve preference annotation efficiency and effectiveness, exhibiting a strong correlation with human annotations. Based on the RAG-RewardBench, we conduct a comprehensive evaluation of 45 RMs and uncover their limitations in RAG scenarios. Additionally, we also reveal that existing trained RALMs show almost no improvement in preference alignment, highlighting the need for a shift towards preference-aligned this http URL release our benchmark and code publicly at this https URL for future work. 

**Abstract (ZH)**: 尽管现有的检索增强语言模型（RALMs）在提供可信回答和可靠信息源依据方面取得了显著进展，但它们往往忽视了与人类偏好有效对齐的重要性。在对齐过程中，奖励模型（RMs）作为人类价值观的关键代理，对优化过程起到指导作用。然而，如何评估和选择适用于RALMs的可靠RMs来实现偏好对齐仍不清晰。为了解决这一问题，我们提出了RAG-RewardBench，这是首个用于评估RAG环境下RMs的基准。首先，我们设计了四个关键且具有挑战性的RAG特定场景来评估RMs，包括多跳推理、精细引用、适当回避以及冲突稳健性。其次，我们整合了18个RAG子集、六种检索器和24种RALMs，以增加数据源多样性。最后，我们采用LLM作为裁判的方法提高了偏好注释的效率和有效性，该方法与人类注释具有较强的关联性。基于RAG-RewardBench，我们对45种RMs进行了全面评估，并揭示了它们在RAG场景中的局限性。此外，我们还发现现有的训练过的RALMs在偏好对齐方面几乎没有改进，突显了转向偏好对齐的必要性。我们将在https://github.com/your-repo/reward-bench-public提供我们的基准和代码，供未来研究使用。 

---
# Federated Learning and RAG Integration: A Scalable Approach for Medical Large Language Models 

**Title (ZH)**: 联邦学习与RAG集成：一种适用于医疗大规模语言模型的扩展方法 

**Authors**: Jincheol Jung, Hongju Jeong, Eui-Nam Huh  

**Link**: [PDF](https://arxiv.org/pdf/2412.13720)  

**Abstract**: This study analyzes the performance of domain-specific Large Language Models (LLMs) for the medical field by integrating Retrieval-Augmented Generation (RAG) systems within a federated learning framework. Leveraging the inherent advantages of federated learning, such as preserving data privacy and enabling distributed computation, this research explores the integration of RAG systems with models trained under varying client configurations to optimize performance. Experimental results demonstrate that the federated learning-based models integrated with RAG systems consistently outperform their non-integrated counterparts across all evaluation metrics. This study highlights the potential of combining federated learning and RAG systems for developing domain-specific LLMs in the medical field, providing a scalable and privacy-preserving solution for enhancing text generation capabilities. 

**Abstract (ZH)**: 本研究通过在联邦学习框架中整合检索增强生成（RAG）系统，分析了专门领域的大语言模型（LLMs）在医疗领域的性能。依托联邦学习固有的优势，如保护数据隐私和实现分布式计算，本研究探讨了在不同客户端配置下将RAG系统与模型进行整合以优化性能的可能性。实验结果表明，基于联邦学习并与RAG系统整合的模型在所有评估指标上均优于未整合的模型。本研究强调了将联邦学习和RAG系统结合用于开发医疗领域的专门领域大语言模型的潜力，提供了一个可扩展且保护隐私的解决方案，以提升文本生成能力。 

---
# Language verY Rare for All 

**Title (ZH)**: “语言非常稀缺对于所有人” 

**Authors**: Ibrahim Merad, Amos Wolf, Ziad Mazzawi, Yannick Léo  

**Link**: [PDF](https://arxiv.org/pdf/2412.13924)  

**Abstract**: In the quest to overcome language barriers, encoder-decoder models like NLLB have expanded machine translation to rare languages, with some models (e.g., NLLB 1.3B) even trainable on a single GPU. While general-purpose LLMs perform well in translation, open LLMs prove highly competitive when fine-tuned for specific tasks involving unknown corpora. We introduce LYRA (Language verY Rare for All), a novel approach that combines open LLM fine-tuning, retrieval-augmented generation (RAG), and transfer learning from related high-resource languages. This study is exclusively focused on single-GPU training to facilitate ease of adoption. Our study focuses on two-way translation between French and Monégasque, a rare language unsupported by existing translation tools due to limited corpus availability. Our results demonstrate LYRA's effectiveness, frequently surpassing and consistently matching state-of-the-art encoder-decoder models in rare language translation. 

**Abstract (ZH)**: 为了克服语言障碍，如NLLB这样的编码器-解码器模型已经扩展了机器翻译到稀有语言，有些模型（例如NLLB 1.3B）甚至可以在单块GPU上进行训练。虽然通用的大型语言模型在翻译方面表现良好，但在特定任务的微调中，开源的大型语言模型在处理未知语料库时表现出极高的竞争力。我们提出了LYRA（Language verY Rare for All）这一创新方法，结合了开放的大型语言模型微调、检索增强生成（RAG）以及从相关高资源语言转移学习。本研究专注于单块GPU训练，以促进其易于采用。我们的研究集中在法语和摩纳哥语之间的双向翻译上，这种稀有语言由于语料库的有限可用性而无法被现有的翻译工具支持。我们的研究结果证明了LYRA的有效性，在稀有语言翻译方面频繁超越甚至一致性地匹配最先进的编码器-解码器模型。 

---
# RemoteRAG: A Privacy-Preserving LLM Cloud RAG Service 

**Title (ZH)**: RemoteRAG：一种隐私保护的大语言模型云检索增强服务 

**Authors**: Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan, Yunhao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2412.12775)  

**Abstract**: Retrieval-augmented generation (RAG) improves the service quality of large language models by retrieving relevant documents from credible literature and integrating them into the context of the user query. Recently, the rise of the cloud RAG service has made it possible for users to query relevant documents conveniently. However, directly sending queries to the cloud brings potential privacy leakage. In this paper, we are the first to formally define the privacy-preserving cloud RAG service to protect the user query and propose RemoteRAG as a solution regarding privacy, efficiency, and accuracy. For privacy, we introduce $(n,\epsilon)$-DistanceDP to characterize privacy leakage of the user query and the leakage inferred from relevant documents. For efficiency, we limit the search range from the total documents to a small number of selected documents related to a perturbed embedding generated from $(n,\epsilon)$-DistanceDP, so that computation and communication costs required for privacy protection significantly decrease. For accuracy, we ensure that the small range includes target documents related to the user query with detailed theoretical analysis. Experimental results also demonstrate that RemoteRAG can resist existing embedding inversion attack methods while achieving no loss in retrieval under various settings. Moreover, RemoteRAG is efficient, incurring only $0.67$ seconds and $46.66$KB of data transmission ($2.72$ hours and $1.43$ GB with the non-optimized privacy-preserving scheme) when retrieving from a total of $10^6$ documents. 

**Abstract (ZH)**: 检索增强生成（RAG）通过从可靠文献中检索相关文档并将其整合到用户的查询语境中，提高了大型语言模型的服务质量。最近，云RAG服务的兴起使得用户可以方便地查询相关文档。然而，直接将查询发送到云中会带来潜在的隐私泄露风险。本文首次正式定义了保护用户查询隐私的云RAG服务，并提出RemoteRAG作为在隐私、效率和准确性方面的解决方案。为了保护隐私，我们引入了基于$(n,\epsilon)$-DistanceDP的隐私泄露表征，以量化用户查询及其从相关文档推断出的隐私泄露。为了提高效率，我们将搜索范围限制在从总文档中筛选出的一小部分与$(n,\epsilon)$-DistanceDP生成的扰动嵌入相关的文档，从而显著减少了保护隐私所需的计算和通信成本。为了保证准确性，我们通过详细的理论分析确保该小范围内包含与用户查询相关的目标文档。实验结果还表明，RemoteRAG可以在多种设置下抵抗现存的嵌入反转攻击方法，同时在检索方面保持无损失。此外，RemoteRAG在从总共$10^6$份文档中检索时仅需要0.67秒和46.66KB的数据传输（未经优化的隐私保护方案需要2.72小时和1.43GB的数据传输）。 

---
# C-FedRAG: A Confidential Federated Retrieval-Augmented Generation System 

**Title (ZH)**: C-FedRAG：一种保密 Federated Retrieval-Augmented Generation 系统

解释：
- "C" 在这个上下文中代表 "Confidential"，即保密的。
- "FedRAG" 保持不变，因为它是一个特定的系统命名。
- "Federated" 直接翻译为“联邦的”或“协作的”，在技术领域常译为“联邦学习”或"Federated"。
- "Retrieval-Augmented Generation" 可以翻译为“检索增强生成”，这个术语在自然语言处理领域较为常见。

这样的翻译既符合学术规范，又能准确传达原文的意思。 

**Authors**: Parker Addison, Minh-Tuan H. Nguyen, Tomislav Medan, Mohammad T. Manzari, Brendan McElrone, Laksh Lalwani, Aboli More, Smita Sharma, Holger R. Roth, Isaac Yang, Chester Chen, Daguang Xu, Yan Cheng, Andrew Feng, Ziyue Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.13163)  

**Abstract**: Organizations seeking to utilize Large Language Models (LLMs) for knowledge querying and analysis often encounter challenges in maintaining an LLM fine-tuned on targeted, up-to-date information that keeps answers relevant and grounded. Retrieval Augmented Generation (RAG) has quickly become a feasible solution for organizations looking to overcome the challenges of maintaining proprietary models and to help reduce LLM hallucinations in their query responses. However, RAG comes with its own issues regarding scaling data pipelines across tiered-access and disparate data sources. In many scenarios, it is necessary to query beyond a single data silo to provide richer and more relevant context for an LLM. Analyzing data sources within and across organizational trust boundaries is often limited by complex data-sharing policies that prohibit centralized data storage, therefore, inhibit the fast and effective setup and scaling of RAG solutions. In this paper, we introduce Confidential Computing (CC) techniques as a solution for secure Federated Retrieval Augmented Generation (FedRAG). Our proposed Confidential FedRAG system (C-FedRAG) enables secure connection and scaling of a RAG workflows across a decentralized network of data providers by ensuring context confidentiality. We also demonstrate how to implement a C-FedRAG system using the NVIDIA FLARE SDK and assess its performance using the MedRAG toolkit and MIRAGE benchmarking dataset. 

**Abstract (ZH)**: 组织利用大型语言模型（LLMs）进行知识查询和分析时，常常会面临保持模型针对特定、最新信息进行微调的挑战，以确保答案的相关性和真实性。检索增强生成（RAG）迅速成为了解决组织维护专有模型及减少LLM查询响应中妄想现象的一种可行方案。然而，RAG 在跨层级访问和异构数据源扩展数据管道方面也带来了一些问题。在许多情况下，为了提供更丰富和相关的情境，需要超越单一数据孤岛进行查询。然而，分析组织信任边界内外的数据源时常受到复杂数据共享政策的限制，这些政策通常禁止集中式数据存储，从而妨碍了RAG解决方案的快速和有效部署与扩展。在本文中，我们介绍了一种使用保密计算（CC）技术的安全联邦检索增强生成（FedRAG）方案。我们的提议的保密联邦检索增强生成系统（C-FedRAG）能够在分布式数据提供者网络中确保上下文的保密性，从而实现RAG工作流程的安全连接和扩展。我们还展示了如何使用NVIDIA FLARE SDK实现C-FedRAG系统，并使用MedRAG工具包和MIRAGE基准测试数据集评估其性能。 

---
