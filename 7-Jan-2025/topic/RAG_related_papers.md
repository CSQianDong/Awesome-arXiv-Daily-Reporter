# QuIM-RAG: Advancing Retrieval-Augmented Generation with Inverted Question Matching for Enhanced QA Performance 

**Title (ZH)**: QuIM-RAG：通过反转问题匹配提升检索增强生成的问答性能 

**Authors**: Binita Saha, Utsha Saha, Muhammad Zubair Malik  

**Link**: [PDF](https://arxiv.org/pdf/2501.02702)  

**Abstract**: This work presents a novel architecture for building Retrieval-Augmented Generation (RAG) systems to improve Question Answering (QA) tasks from a target corpus. Large Language Models (LLMs) have revolutionized the analyzing and generation of human-like text. These models rely on pre-trained data and lack real-time updates unless integrated with live data tools. RAG enhances LLMs by integrating online resources and databases to generate contextually appropriate responses. However, traditional RAG still encounters challenges like information dilution and hallucinations when handling vast amounts of data. Our approach addresses these challenges by converting corpora into a domain-specific dataset and RAG architecture is constructed to generate responses from the target document. We introduce QuIM-RAG (Question-to-question Inverted Index Matching), a novel approach for the retrieval mechanism in our system. This strategy generates potential questions from document chunks and matches these with user queries to identify the most relevant text chunks for generating accurate answers. We have implemented our RAG system on top of the open-source Meta-LLaMA3-8B-instruct model by Meta Inc. that is available on Hugging Face. We constructed a custom corpus of 500+ pages from a high-traffic website accessed thousands of times daily for answering complex questions, along with manually prepared ground truth QA for evaluation. We compared our approach with traditional RAG models using BERT-Score and RAGAS, state-of-the-art metrics for evaluating LLM applications. Our evaluation demonstrates that our approach outperforms traditional RAG architectures on both metrics. 

**Abstract (ZH)**: 本文提出了一个新的架构，用于构建检索增强生成（RAG）系统，以提高目标语料库中的问答（QA）任务性能。大型语言模型（LLMs）已经彻底改变了人类似文本的分析和生成。这些模型依赖于预训练数据，除非与实时数据工具集成，否则无法实现在线更新。RAG通过整合在线资源和数据库来增强LLMs，从而生成上下文相关性更强的响应。然而，传统的RAG在处理大量数据时仍面临信息稀释和幻觉等挑战。我们通过将语料库转化为领域特定的数据集来解决这些挑战，并构建了RAG架构以从目标文档中生成响应。我们引入了一种名为QuIM-RAG（问题到问题倒排索引匹配）的新方法，用于我们的系统中的检索机制。该策略从文档片段中生成潜在问题，并将这些问题与用户查询匹配，以识别生成准确答案所需的最相关文本片段。我们以Meta Inc.提供的开源Meta-LLaMA3-8B-instruct模型为基础，该模型可在Hugging Face上获得，并构建了一个包含500多页的自定义语料库，用于回答复杂问题，同时准备了人工整理的真实QA数据以进行评估。我们使用BERT-Score和RAGAS等最先进的指标，与传统的RAG模型进行了比较评估。我们的评估结果表明，我们的方法在两个指标上均优于传统的RAG架构。 

---
# Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications 

**Title (ZH)**: 面向全方位RAG：大型语言模型在医疗应用中的全面检索增强生成 

**Authors**: Zhe Chen, Yusheng Liao, Shuyang Jiang, Pingjie Wang, Yiqiu Guo, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02460)  

**Abstract**: Large language models (LLMs) hold promise for addressing healthcare challenges but often generate hallucinations due to limited integration of medical knowledge. Incorporating external medical knowledge is therefore critical, especially considering the breadth and complexity of medical content, which necessitates effective multi-source knowledge acquisition. We address this challenge by framing it as a source planning problem, where the task is to formulate context-appropriate queries tailored to the attributes of diverse knowledge sources. Existing approaches either overlook source planning or fail to achieve it effectively due to misalignment between the model's expectation of the sources and their actual content. To bridge this gap, we present MedOmniKB, a comprehensive repository comprising multigenre and multi-structured medical knowledge sources. Leveraging these sources, we propose the Source Planning Optimisation (SPO) method, which enhances multi-source utilisation through explicit planning optimisation. Our approach involves enabling an expert model to explore and evaluate potential plans while training a smaller model to learn source alignment using positive and negative planning samples. Experimental results demonstrate that our method substantially improves multi-source planning performance, enabling the optimised small model to achieve state-of-the-art results in leveraging diverse medical knowledge sources. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解决医疗挑战方面具有潜力，但往往会产生幻觉，这是因为医学知识的整合有限。因此，集成外部医学知识至关重要，尤其是在面对广度和复杂性都很高的医疗内容时，这需要有效的多源知识获取。我们通过将这一挑战重新定义为一个来源规划问题来应对这一挑战，其中任务是制定适合上下文的查询，针对多种知识源的属性进行定制。现有的方法要么忽视了来源规划，要么未能有效实现这一目标，原因在于模型对源的期望与其实际内容之间的不一致。为了弥合这一差距，我们提出了MedOmniKB这一综合资源库，其中包括多体裁和多结构的医学知识源。借助这些资源，我们提出了来源规划优化（SPO）方法，该方法通过明确的规划优化来增强多源利用。我们的方法包括让专家模型探索和评估潜在计划的可能性，同时训练一个小模型来学习源的对齐，使用正反规划样本作为训练材料。实验结果表明，我们的方法显著提高了多源规划性能，使优化的小模型能够在利用多种医学知识源方面达到最先进的效果。 

---
# Personalized Graph-Based Retrieval for Large Language Models 

**Title (ZH)**: 基于图的个性化检索方法在大型语言模型中的应用 

**Authors**: Steven Au, Cameron J. Dimacali, Ojasmitha Pedirappagari, Namyong Park, Franck Dernoncourt, Yu Wang, Nikos Kanakaris, Hanieh Deilamsalehy, Ryan A. Rossi, Nesreen K. Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2501.02157)  

**Abstract**: As large language models (LLMs) evolve, their ability to deliver personalized and context-aware responses offers transformative potential for improving user experiences. Existing personalization approaches, however, often rely solely on user history to augment the prompt, limiting their effectiveness in generating tailored outputs, especially in cold-start scenarios with sparse data. To address these limitations, we propose Personalized Graph-based Retrieval-Augmented Generation (PGraphRAG), a framework that leverages user-centric knowledge graphs to enrich personalization. By directly integrating structured user knowledge into the retrieval process and augmenting prompts with user-relevant context, PGraphRAG enhances contextual understanding and output quality. We also introduce the Personalized Graph-based Benchmark for Text Generation, designed to evaluate personalized text generation tasks in real-world settings where user history is sparse or unavailable. Experimental results show that PGraphRAG significantly outperforms state-of-the-art personalization methods across diverse tasks, demonstrating the unique advantages of graph-based retrieval for personalization. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的发展，它们提供个性化的、情境感知的响应能力具有改变用户体验的潜力。然而，现有的个性化方法通常仅依赖用户的使用历史来增强提示，这在数据稀疏或缺失的冷启动场景中限制了其生成个性化输出的效果。为了解决这些限制，我们提出了一种基于个性化图的检索增强生成（PGraphRAG）框架，该框架利用用户为中心的知识图谱来增强个性化。通过直接将结构化的用户知识整合到检索过程中，并使用用户相关背景信息增强提示，PGraphRAG 提升了语境理解和输出质量。我们还引入了基于个性化图的文本生成基准测试，旨在评估在用户历史稀疏或不可用的现实环境中个性化文本生成任务的表现。实验结果表明，PGraphRAG 在多种任务中显著优于现有的个性化方法，证明了基于图的检索方法在个性化中的独特优势。 

---
# FlipedRAG: Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models 

**Title (ZH)**: FlipedRAG：针对大型语言模型检索增强生成的黑箱意见操纵攻击 

**Authors**: Zhuo Chen, Yuyang Gong, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu, Jiawei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.02968)  

**Abstract**: Retrieval-Augmented Generation (RAG) addresses hallucination and real-time constraints by dynamically retrieving relevant information from a knowledge database to supplement the LLMs' input. When presented with a query, RAG selects the most semantically similar texts from its knowledge bases and uses them as context for the LLMs to generate more accurate responses. RAG also creates a new attack surface, especially since RAG databases are frequently sourced from public domains. While existing studies have predominantly focused on optimizing RAG's performance and efficiency, emerging research has begun addressing the security concerns associated with RAG. However, these works have some limitations, typically focusing on either white-box methodologies or heuristic-based black-box attacks. Furthermore, prior research has mainly targeted simple factoid question answering, which is neither practically challenging nor resistant to correction. In this paper, we unveil a more realistic and threatening scenario: opinion manipulation for controversial topics against RAG. Particularly, we propose a novel RAG black-box attack method, termed FlipedRAG, which is transfer-based. By leveraging instruction engineering, we obtain partial retrieval model outputs from black-box RAG system, facilitating the training of surrogate models to enhance the effectiveness of opinion manipulation attack. Extensive experimental results confirms that our approach significantly enhances the average success rate of opinion manipulation by 16.7%. It achieves an average of a 50% directional change in the opinion polarity of RAG responses across four themes. Additionally, it induces a 20% shift in user cognition. Furthermore, we discuss the efficacy of potential defense mechanisms and conclude that they are insufficient in mitigating this type of attack, highlighting the urgent need to develop novel defensive strategies. 

**Abstract (ZH)**: 检索增强生成（RAG）通过从知识数据库动态检索相关信息来补充语言模型（LLM）的输入，从而解决了幻觉和实时性限制的问题。在接收到查询时，RAG 会从其知识库中选择最具语义相似性的文本，并将其作为上下文来生成更准确的响应。同时，RAG 也创建了一个新的攻击面，特别是在 RAG 数据库经常来源于公开领域的情况下。尽管现有的研究主要集中在优化 RAG 的性能和效率上，但新兴的研究已经开始关注与 RAG 相关的安全问题。然而，这些工作存在一定的局限性，通常侧重于白盒方法或启发式黑盒攻击。此外，先前的研究主要针对简单的事实性问题回答，这些问题既不实际困难，也不具有改正性。在本文中，我们揭示了一个更现实且更具威胁性的场景：针对 RAG 的观点操纵，特别是出于争议性话题的考虑。我们提出了一种新颖的基于转移的学习 RAG 黑盒攻击方法，称为 FlipedRAG。通过利用指令工程，我们从黑盒 RAG 系统中获得了部分检索模型输出，这有助于培训替代模型以增强观点操纵攻击的有效性。大量实验证明，我们的方法显著提高了观点操纵的平均成功率 16.7%。它在四个主题的响应中实现了平均 50% 的观点极性方向性变化，并且引发了 20% 的用户认知偏移。此外，我们讨论了潜在防御机制的有效性，并得出结论认为这些防御机制不足以缓解这种攻击，突显了开发新型防御策略的紧迫需求。 

---
# Tree-based RAG-Agent Recommendation System: A Case Study in Medical Test Data 

**Title (ZH)**: 基于树结构的RAG代理推荐系统：一项关于医学测试数据的研究案例 

**Authors**: Yahe Yang, Chengyue Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.02727)  

**Abstract**: We present HiRMed (Hierarchical RAG-enhanced Medical Test Recommendation), a novel tree-structured recommendation system that leverages Retrieval-Augmented Generation (RAG) for intelligent medical test recommendations. Unlike traditional vector similarity-based approaches, our system performs medical reasoning at each tree node through a specialized RAG process. Starting from the root node with initial symptoms, the system conducts step-wise medical analysis to identify potential underlying conditions and their corresponding diagnostic requirements. At each level, instead of simple matching, our RAG-enhanced nodes analyze retrieved medical knowledge to understand symptom-disease relationships and determine the most appropriate diagnostic path. The system dynamically adjusts its recommendation strategy based on medical reasoning results, considering factors such as urgency levels and diagnostic uncertainty. Experimental results demonstrate that our approach achieves superior performance in terms of coverage rate, accuracy, and miss rate compared to conventional retrieval-based methods. This work represents a significant advance in medical test recommendation by introducing medical reasoning capabilities into the traditional tree-based retrieval structure. 

**Abstract (ZH)**: 我们提出了一种名为 HiRMed（层次增强检索生成医疗测试推荐）的创新树结构推荐系统，该系统利用检索增强生成（RAG）技术实现智能化的医疗测试推荐。与传统的基于向量相似性的方法不同，我们的系统在每一棵树节点处通过专门的 RAG 过程进行医疗推理。从根节点出发，初始症状开始，系统逐步进行医疗分析，以识别潜在的病因及其对应的诊断需求。在每一层中，相较于简单的匹配，带有 RAG 增强的节点分析检索到的医学知识，了解症状-疾病关系，并确定最合适的诊断路径。系统根据医疗推理的结果动态调整其推荐策略，综合考虑急迫程度和诊断不确定性等因素。实验结果表明，与传统的基于检索的方法相比，我们的方法在覆盖率、准确性和漏诊率等方面表现更优。这项工作代表了在传统的基于树的检索结构中引入医疗推理能力的重要进展，在医疗测试推荐方面取得了显著进步。 

---
# Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation 

**Title (ZH)**: 基于知识图谱检索增强的生成推荐模型 

**Authors**: Shijie Wang, Wenqi Fan, Yue Feng, Xinyu Ma, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2501.02226)  

**Abstract**: Recommender systems have become increasingly vital in our daily lives, helping to alleviate the problem of information overload across various user-oriented online services. The emergence of Large Language Models (LLMs) has yielded remarkable achievements, demonstrating their potential for the development of next-generation recommender systems. Despite these advancements, LLM-based recommender systems face inherent limitations stemming from their LLM backbones, particularly issues of hallucinations and the lack of up-to-date and domain-specific knowledge. Recently, Retrieval-Augmented Generation (RAG) has garnered significant attention for addressing these limitations by leveraging external knowledge sources to enhance the understanding and generation of LLMs. However, vanilla RAG methods often introduce noise and neglect structural relationships in knowledge, limiting their effectiveness in LLM-based recommendations. To address these limitations, we propose to retrieve high-quality and up-to-date structure information from the knowledge graph (KG) to augment recommendations. Specifically, our approach develops a retrieval-augmented framework, termed K-RagRec, that facilitates the recommendation generation process by incorporating structure information from the external KG. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed method. 

**Abstract (ZH)**: 推荐系统在我们的日常生活中变得愈发重要，它们帮助各类用户面向在线服务缓解信息过载问题。大型语言模型（LLMs）的出现取得了显著成就，显示出其在新一代推荐系统开发中的潜力。然而，尽管取得了这些进展，LLM 基于的推荐系统仍然面临来自其LLM基础模型固有的限制，特别是幻觉问题以及缺乏最新的和专门领域的知识。最近，检索增强生成（RAG）方法引起了广泛关注，通过利用外部知识源来增强LLM的理解和生成能力，以解决这些限制。然而，传统的RAG方法往往引入噪声并忽略知识中的结构性关系，限制了它们在基于LLM的推荐中的有效性。为了解决这些限制，我们提出了一种方法，从知识图谱（KG）中检索高质量和最新的结构信息来增强推荐。具体而言，我们的方法开发了一种检索增强框架，称为K-RagRec，通过结合外部KG中的结构信息来促进推荐生成过程。进行了广泛的实验以展示我们提出方法的有效性。 

---
# The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit 

**Title (ZH)**: 效率与准确性的权衡：使用多头早期退出优化增强型LLM推荐系统 

**Authors**: Huixue Zhou, Hengrui Gu, Xi Liu, Kaixiong Zhou, Mingfu Liang, Yongkang Xiao, Srinivas Govindan, Piyush Chawla, Jiyan Yang, Xiangfei Meng, Huayu Li, Buyun Zhang, Liang Luo, Wen-Yen Chen, Yiping Han, Bo Long, Rui Zhang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.02173)  

**Abstract**: The deployment of Large Language Models (LLMs) in recommender systems for predicting Click-Through Rates (CTR) necessitates a delicate balance between computational efficiency and predictive accuracy. This paper presents an optimization framework that combines Retrieval-Augmented Generation (RAG) with an innovative multi-head early exit architecture to concurrently enhance both aspects. By integrating Graph Convolutional Networks (GCNs) as efficient retrieval mechanisms, we are able to significantly reduce data retrieval times while maintaining high model performance. The early exit strategy employed allows for dynamic termination of model inference, utilizing real-time predictive confidence assessments across multiple heads. This not only quickens the responsiveness of LLMs but also upholds or improves their accuracy, making it ideal for real-time application scenarios. Our experiments demonstrate how this architecture effectively decreases computation time without sacrificing the accuracy needed for reliable recommendation delivery, establishing a new standard for efficient, real-time LLM deployment in commercial systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推荐系统中的部署对于预测点击率（CTR）而言，需要在计算效率和预测准确性之间保持微妙的平衡。本文提出了一种优化框架，结合了检索增强生成（RAG）与一种创新的多头早期退出架构，以同时提升这两个方面。通过将图卷积网络（GCNs）作为高效的检索机制进行整合，我们能够在降低数据检索时间的同时，保持高模型性能。所采用的早期退出策略允许模型推理的动态终止，通过多个头的实时预测置信度评估来进行。这不仅能加快大语言模型的响应速度，而且还能够维持或提升其准确性，使其适用于实时应用场景。我们的实验表明，这种架构能够在不牺牲可靠推荐所需准确性的前提下有效减少计算时间，从而为商业系统中高效、实时的大语言模型部署设立了新标准。 

---
# Graph-based Retrieval Augmented Generation for Dynamic Few-shot Text Classification 

**Title (ZH)**: 基于图的检索增强生成方法在动态少样本文本分类中的应用 

**Authors**: Yubo Wang, Haoyang Li, Fei Teng, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.02844)  

**Abstract**: Text classification is a fundamental task in natural language processing, pivotal to various applications such as query optimization, data integration, and schema matching. While neural network-based models, such as CNN and BERT, have demonstrated remarkable performance in text classification, their effectiveness heavily relies on abundant labeled training data. This dependency makes these models less effective in dynamic few-shot text classification, where labeled data is scarce, and target labels frequently evolve based on application needs. Recently, large language models (LLMs) have shown promise due to their extensive pretraining and contextual understanding. Current approaches provide LLMs with text inputs, candidate labels, and additional side information (e.g., descriptions) to predict text labels. However, their effectiveness is hindered by the increased input size and the noise introduced through side information processing. To address these limitations, we propose a graph-based online retrieval-augmented generation framework, namely GORAG, for dynamic few-shot text classification. GORAG constructs and maintains an adaptive information graph by extracting side information across all target texts, rather than treating each input independently. It employs a weighted edge mechanism to prioritize the importance and reliability of extracted information and dynamically retrieves relevant context using a minimum-cost spanning tree tailored for each text input. Empirical evaluations demonstrate that GORAG outperforms existing approaches by providing more comprehensive and accurate contextual information. 

**Abstract (ZH)**: 文本分类是自然语言处理中的一个基本任务，对于各种应用如查询优化、数据集成和模式匹配至关重要。虽然基于神经网络的模型，如CNN和BERT，在文本分类方面表现出色，但它们的有效性高度依赖于充足的标注训练数据。这种依赖性使其在动态少量标注的文本分类任务中效果较差，在这种任务中，标注数据稀缺，目标标签会频繁根据应用需求而变化。最近，大型语言模型（LLMs）由于其广泛的预训练和语境理解能力，显示出了潜力。现有方法通过提供文本输入、候选标签以及额外的辅助信息（如描述）来预测文本标签。然而，这些方法的有效性受到了输入尺寸增加以及辅助信息处理过程中引入噪声的阻碍。为了解决这些问题，我们提出了一种基于图的在线检索增强生成框架，即GORAG，用于动态少量标注的文本分类。GORAG通过跨所有目标文本提取辅助信息，而不是独立处理每个输入，构建并维护一个自适应信息图。它采用加权边机制以优先处理提取信息的重要性与可靠性，并动态检索针对每个文本输入定制的最小子成本生成树来获取相关的上下文。实证研究表明，GORAG相较于现有方法能够提供更加全面和准确的上下文信息。 

---
