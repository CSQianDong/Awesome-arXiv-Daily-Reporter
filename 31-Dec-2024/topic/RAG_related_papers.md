# Understanding the Impact of Confidence in Retrieval Augmented Generation: A Case Study in the Medical Domain 

**Title (ZH)**: 理解检索增强生成中信心的影响：医疗领域案例研究 

**Authors**: Shintaro Ozaki, Yuta Kato, Siyuan Feng, Masayo Tomita, Kazuki Hayashi, Ryoma Obara, Masafumi Oyamada, Katsuhiko Hayashi, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2412.20309)  

**Abstract**: Retrieval Augmented Generation (RAG) complements the knowledge of Large Language Models (LLMs) by leveraging external information to enhance response accuracy for queries. This approach is widely applied in several fields by taking its advantage of injecting the most up-to-date information, and researchers are focusing on understanding and improving this aspect to unlock the full potential of RAG in such high-stakes applications. However, despite the potential of RAG to address these needs, the mechanisms behind the confidence levels of its outputs remain underexplored, although the confidence of information is very critical in some domains, such as finance, healthcare, and medicine. Our study focuses the impact of RAG on confidence within the medical domain under various configurations and models. We evaluate confidence by treating the model's predicted probability as its output and calculating Expected Calibration Error (ECE) and Adaptive Calibration Error (ACE) scores based on the probabilities and accuracy. In addition, we analyze whether the order of retrieved documents within prompts calibrates the confidence. Our findings reveal large variation in confidence and accuracy depending on the model, settings, and the format of input prompts. These results underscore the necessity of optimizing configurations based on the specific model and conditions. 

**Abstract (ZH)**: 检索增强生成（RAG）通过利用外部信息来补充大型语言模型（LLMs）的知识，从而提高查询响应的准确性。该方法在多个领域得到了广泛应用，利用其注入最新信息的优势，并且研究人员正致力于理解和改进这一方面，以充分发挥RAG在高风险应用中的潜力。然而，尽管RAG有潜力解决这些需求，其输出的置信水平背后的机制仍被广泛忽视，尤其是在金融、医疗保健和医学等领域，信息的置信度至关重要。我们的研究集中在不同配置和模型下，RAG在医疗领域的置信度影响。我们通过将模型预测的概率视为输出，并基于这些概率和准确性计算期望校准误差（ECE）和自适应校准误差（ACE）得分来进行置信度评估。此外，我们还分析了提示中检索文档的顺序是否校准了置信度。研究结果表明，置信度和准确性在不同的模型、设置以及输入提示的格式上存在显著差异。这些结果强调了根据特定模型和条件优化配置的必要性。 

---
# A Comprehensive Framework for Reliable Legal AI: Combining Specialized Expert Systems and Adaptive Refinement 

**Title (ZH)**: 可靠法律AI的综合框架：结合专门专家系统和自适应精炼 

**Authors**: Sidra Nasir, Qamar Abbas, Samita Bai, Rizwan Ahmed Khan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20468)  

**Abstract**: This article discusses the evolving role of artificial intelligence (AI) in the legal profession, focusing on its potential to streamline tasks such as document review, research, and contract drafting. However, challenges persist, particularly the occurrence of "hallucinations" in AI models, where they generate inaccurate or misleading information, undermining their reliability in legal contexts. To address this, the article proposes a novel framework combining a mixture of expert systems with a knowledge-based architecture to improve the precision and contextual relevance of AI-driven legal services. This framework utilizes specialized modules, each focusing on specific legal areas, and incorporates structured operational guidelines to enhance decision-making. Additionally, it leverages advanced AI techniques like Retrieval-Augmented Generation (RAG), Knowledge Graphs (KG), and Reinforcement Learning from Human Feedback (RLHF) to improve the system's accuracy. The proposed approach demonstrates significant improvements over existing AI models, showcasing enhanced performance in legal tasks and offering a scalable solution to provide more accessible and affordable legal services. The article also outlines the methodology, system architecture, and promising directions for future research in AI applications for the legal sector. 

**Abstract (ZH)**: 本文探讨了人工智能（AI）在法律职业中的不断演变的作用，着重讨论了其在文书审查、研究和合同起草等任务上的潜在效能。然而，挑战仍然存在，尤其是在AI模型中出现的“幻觉”现象，这些模型生成的信息不准确或误导，从而削弱了它们在法律环境中的可靠性。为了解决这一问题，本文提出了一种新颖的框架，该框架将专家系统与基于知识的架构相结合，以提高AI驱动的法律服务的精确度和上下文相关性。该框架利用专门模块，每个模块专注于特定的法律领域，并结合结构化操作指南以增强决策过程。此外，该框架利用了先进的AI技术，如检索增强生成（RAG）、知识图谱（KG）和基于人类反馈的强化学习（RLHF），以提高系统的准确性。所提出的方法在现有AI模型上表现出显著改进，在法律任务中的性能得到增强，并提供了一种可扩展的解决方案，以提供更普及和经济实惠的法律服务。本文还介绍了研究方法、系统架构以及未来研究中在法律领域应用AI的有希望的方向。 

---
# Plancraft: an evaluation dataset for planning with LLM agents 

**Title (ZH)**: PlanCraft：用于评估基于LLM代理的规划数据集 

**Authors**: Gautier Dagan, Frank Keller, Alex Lascarides  

**Link**: [PDF](https://arxiv.org/pdf/2412.21033)  

**Abstract**: We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as an oracle planner and oracle RAG information extractor, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and strategies on our task and compare their performance to a handcrafted planner. We find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and we offer suggestions on how to improve their capabilities. 

**Abstract (ZH)**: 我们介绍了Plancraft，这是一个针对大规模语言模型（LLM）代理的跨模态评估数据集。Plancraft 支持文本-only 和跨模态两种界面，基于《我的世界》（Minecraft）的制作用户界面（GUI）。我们包含了《我的世界》维基，用于评估工具使用和检索增强生成（RAG），同时也提供了一个 oracle 计划者和 oracle RAG 信息提取器，以消除现代代理架构中不同组件的影响。为了评估决策能力，Plancraft 还包括了一部分故意无法解决的示例，这些示例为代理提供了真实的挑战，不仅需要代理完成任务，还需要代理判断这些任务是否可解。我们对开源和封闭源代码的 LLM 和策略进行了基准测试，并将它们的表现与手工设计的计划者进行了比较。我们发现，LLM 和视觉-语言模型在处理 Plancraft 引入的规划问题时表现不佳，并提出了提高其能力的建议。 

---
