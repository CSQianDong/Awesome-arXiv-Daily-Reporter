# Explainable Lane Change Prediction for Near-Crash Scenarios Using Knowledge Graph Embeddings and Retrieval Augmented Generation 

**Title (ZH)**: 使用知识图嵌入和检索增强生成进行可解释的变道预测以应对近危机场景 

**Authors**: M. Manzour, A. Ballardini, R. Izquierdo, M. Á. Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2501.11560)  

**Abstract**: Lane-changing maneuvers, particularly those executed abruptly or in risky situations, are a significant cause of road traffic accidents. However, current research mainly focuses on predicting safe lane changes. Furthermore, existing accident datasets are often based on images only and lack comprehensive sensory data. In this work, we focus on predicting risky lane changes using the CRASH dataset (our own collected dataset specifically for risky lane changes), and safe lane changes (using the HighD dataset). Then, we leverage KG and Bayesian inference to predict these maneuvers using linguistic contextual information, enhancing the model's interpretability and transparency. The model achieved a 91.5% f1-score with anticipation time extending to four seconds for risky lane changes, and a 90.0% f1-score for predicting safe lane changes with the same anticipation time. We validate our model by integrating it into a vehicle within the CARLA simulator in scenarios that involve risky lane changes. The model managed to anticipate sudden lane changes, thus providing automated vehicles with further time to plan and execute appropriate safe reactions. Finally, to enhance the explainability of our model, we utilize RAG to provide clear and natural language explanations for the given prediction. 

**Abstract (ZH)**: 换道行为，尤其是那些执行迅速或在风险较高的情况下进行的换道行为，是导致道路交通事故的重要原因之一。然而，当前的研究主要集中在预测安全的换道行为上。此外，现有的事故数据集往往仅基于图像，缺乏全面的感官数据。在本研究中，我们专注于使用CRASH数据集（我们自己收集的专门针对风险换道行为的数据集）和HighD数据集来预测风险换道行为和安全换道行为。然后，我们利用知识图谱（KG）和贝叶斯推理来利用语义上下文信息预测这些行为，这增强了模型的可解释性和透明度。该模型在预测风险换道行为时，当预见时间为四秒时，获得了91.5%的F1分数；在预测安全换道行为时，相同预见时间下的F1分数达到90.0%。我们通过将该模型集成到CARLA模拟器中的车辆中，针对涉及风险换道的场景进行了验证。模型能够预测突然的换道，从而为自动驾驶车辆提供了额外的时间来规划并执行适当的安全反应。最后，为了增强我们模型的可解释性，我们利用RAG（ rouge-adaptive generation）机制提供清晰且自然语言形式的解释，以说明给定的预测结果。 

---
# Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home 

**Title (ZH)**: 没有自我知识的自适应检索？让不确定性回归 

**Authors**: Viktor Moskvoretskii, Maria Lysyuk, Mikhail Salnikov, Nikolay Ivanov, Sergey Pletenev, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Irina Nikishina, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2501.12835)  

**Abstract**: Retrieval Augmented Generation (RAG) improves correctness of Question Answering (QA) and addresses hallucinations in Large Language Models (LLMs), yet greatly increase computational costs. Besides, RAG is not always needed as may introduce irrelevant information. Recent adaptive retrieval methods integrate LLMs' intrinsic knowledge with external information appealing to LLM self-knowledge, but they often neglect efficiency evaluations and comparisons with uncertainty estimation techniques. We bridge this gap by conducting a comprehensive analysis of 35 adaptive retrieval methods, including 8 recent approaches and 27 uncertainty estimation techniques, across 6 datasets using 10 metrics for QA performance, self-knowledge, and efficiency. Our findings show that uncertainty estimation techniques often outperform complex pipelines in terms of efficiency and self-knowledge, while maintaining comparable QA performance. 

**Abstract (ZH)**: 检索增强生成（RAG）通过提高问答（QA）的准确性并解决大规模语言模型（LLMs）中的幻觉问题，但也大大增加了计算成本。此外，RAG 并非总有必要，因为它可能会引入无关信息。最近的自适应检索方法结合了LLMs 的内在知识与外部信息，以适应LLMs 的自知之明，但这些方法往往忽视了效率评估，并未与不确定性估计技术进行比较。我们通过基于6个数据集的综合分析来弥补这一差距，该分析涵盖了10种评估指标，对35种自适应检索方法进行了评估，其中包括8种最近的方法和技术，以及27种不确定性估计技术。我们的研究结果表明，不确定性估计技术在效率和自知之明方面通常优于复杂的工作流程，同时保持与QA性能相当的水平。 

---
# Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana 

**Title (ZH)**: 使用DataMorgana生成多样化问答基准以评估RAG系统的性能 

**Authors**: Simone Filice, Guy Horowitz, David Carmel, Zohar Karnin, Liane Lewin-Eytan, Yoelle Maarek  

**Link**: [PDF](https://arxiv.org/pdf/2501.12789)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) systems, especially in domain-specific contexts, requires benchmarks that address the distinctive requirements of the applicative scenario. Since real data can be hard to obtain, a common strategy is to use LLM-based methods to generate synthetic data. Existing solutions are general purpose: given a document, they generate a question to build a Q&A pair. However, although the generated questions can be individually good, they are typically not diverse enough to reasonably cover the different ways real end-users can interact with the RAG system. We introduce here DataMorgana, a tool for generating highly customizable and diverse synthetic Q&A benchmarks tailored to RAG applications. DataMorgana enables detailed configurations of user and question categories and provides control over their distribution within the benchmark. It uses a lightweight two-stage process, ensuring efficiency and fast iterations, while generating benchmarks that reflect the expected traffic. We conduct a thorough line of experiments, showing quantitatively and qualitatively that DataMorgana surpasses existing tools and approaches in producing lexically, syntactically, and semantically diverse question sets across domain-specific and general-knowledge corpora. DataMorgana will be made available to selected teams in the research community, as first beta testers, in the context of the upcoming SIGIR'2025 LiveRAG challenge to be announced in early February 2025. 

**Abstract (ZH)**: 评估检索增强生成（RAG）系统，尤其是在特定领域的背景下，需要使用能够应对应用场景独特要求的基准测试。由于真实数据可能难以获取，一个常用策略是使用基于大规模语言模型（LLM）的方法生成合成数据。现有的解决方案通常是通用的：给定一篇文档，它们生成一个问题来构建问答对。然而，尽管生成的问题单个来看可能是好的，但通常它们不够多样化，无法合理覆盖真实最终用户与RAG系统交互的各种方式。我们在此介绍DataMorgana，这是一种用于生成高度自定义和多样化合成问答基准的工具，专门针对RAG应用。DataMorgana允许详细配置用户和问题类别，并提供对基准中它们分布的控制。它使用了一个轻量级的两阶段过程，确保高效和快速迭代，同时生成反映预计流量的基准。我们进行了详尽的实验，定量和定性地展示了DataMorgana在生成跨特定领域和通用知识数据库的词汇、语法和语义多样化问题集方面超越了现有工具和方法。DataMorgana将在2025年初春SIGIR'2025 LiveRAG挑战赛中宣布，作为即将公开的第一个测试版本，提供给研究社区的部分团队进行测试。 

---
