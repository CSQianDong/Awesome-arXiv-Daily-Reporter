# Enhancing Visual Inspection Capability of Multi-Modal Large Language Models on Medical Time Series with Supportive Conformalized and Interpretable Small Specialized Models 

**Title (ZH)**: 增强多模态大型语言模型在医疗时间序列视觉检查能力的支持性同构造化和可解释小型专业化模型辅助方法 

**Authors**: Huayu Li, Xiwen Chen, Ci Zhang, Stuart F. Quan, William D.S. Killgore, Shu-Fen Wung, Chen X. Chen, Geng Yuan, Jin Lu, Ao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.16215)  

**Abstract**: Large language models (LLMs) exhibit remarkable capabilities in visual inspection of medical time-series data, achieving proficiency comparable to human clinicians. However, their broad scope limits domain-specific precision, and proprietary weights hinder fine-tuning for specialized datasets. In contrast, small specialized models (SSMs) excel in targeted tasks but lack the contextual reasoning required for complex clinical decision-making. To address these challenges, we propose ConMIL (Conformalized Multiple Instance Learning), a decision-support SSM that integrates seamlessly with LLMs. By using Multiple Instance Learning (MIL) to identify clinically significant signal segments and conformal prediction for calibrated set-valued outputs, ConMIL enhances LLMs' interpretative capabilities for medical time-series analysis. Experimental results demonstrate that ConMIL significantly improves the performance of state-of-the-art LLMs, such as ChatGPT4.0 and Qwen2-VL-7B. Specifically, \ConMIL{}-supported Qwen2-VL-7B achieves 94.92% and 96.82% precision for confident samples in arrhythmia detection and sleep staging, compared to standalone LLM accuracy of 46.13% and 13.16%. These findings highlight the potential of ConMIL to bridge task-specific precision and broader contextual reasoning, enabling more reliable and interpretable AI-driven clinical decision support. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医学时间序列数据的视觉检查方面表现出显著的能力，其专业水平接近于人类临床医生。然而，其广泛的适用范围限制了其在特定领域的精密度，而专有的模型权重则阻碍了对专门数据集的微调。相比之下，小型专门模型（SSMs）在执行特定任务方面表现出色，但缺乏进行复杂临床决策所需的背景推理能力。为解决这些挑战，我们提出了一种决策支持的小型专门模型（ConMIL，Conformalized Multiple Instance Learning），该模型能够与LLMs无缝集成。通过使用多重实例学习（MIL）识别具有临床意义的信号片段，并使用校准型集合值输出的卷积预测，ConMIL增强了LLMs对医学时间序列分析的解释能力。实验结果表明，ConMIL显著提高了当前先进的LLMs（如ChatGPT4.0和Qwen2-VL-7B）的表现。具体而言，ConMIL支持的Qwen2-VL-7B在心律失常检测和睡眠阶段分类中的精确度分别达到了94.92%和96.82%，而单独的LLMs在这两项任务中的精度分别为46.13%和13.16%。这些发现突显了ConMIL在任务特定精度和更广泛背景推理之间架起桥梁的潜力，从而能够提供更可靠和可解释的AI驱动的临床决策支持。 

---
# From Informal to Formal -- Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs 

**Title (ZH)**: 从非形式化到形式化——集成并评估大语言模型在自然语言需求到可验证的形式证明中的应用 

**Authors**: Jialun Cao, Yaojie Lu, Meiziniu Li, Haoyang Ma, Haokun Li, Mengda He, Cheng Wen, Le Sun, Hongyu Zhang, Shengchao Qin, Shing-Chi Cheung, Cong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2501.16207)  

**Abstract**: The research in AI-based formal mathematical reasoning has shown an unstoppable growth trend. These studies have excelled in mathematical competitions like IMO, showing significant progress. However, these studies intertwined multiple skills simultaneously, i.e., problem-solving, reasoning, and writing formal specifications, making it hard to precisely identify the LLMs' strengths and weaknesses in each task. This paper focuses on formal verification, an immediate application scenario of formal reasoning, and decomposes it into six sub-tasks. We constructed 18k high-quality instruction-response pairs across five mainstream formal specification languages (Coq, Lean4, Dafny, ACSL, and TLA+) in six formal-verification-related tasks by distilling GPT-4o. They are split into a 14k+ fine-tuning dataset FM-alpaca and a 4k benchmark FM-Bench. We found that LLMs are good at writing proof segments when given either the code, or the detailed description of proof steps. Also, the fine-tuning brought about a nearly threefold improvement at most. Interestingly, we observed that fine-tuning with formal data also enhances mathematics, reasoning, and coding abilities. We hope our findings inspire further research. Fine-tuned models are released to facilitate subsequent studies 

**Abstract (ZH)**: 基于AI的正式数学推理研究显示出了不可阻挡的发展趋势。这些研究在数学竞赛中，如国际数学奥林匹克竞赛（IMO），取得了显著的进步。然而，这些研究同时涉及多种技能，包括问题解决、推理和编写形式化规范，使得难以精确识别大模型（LLMs）在每个任务中的优势和不足。本文重点关注形式化验证，这是形式化推理的直接应用场景，并将其分解为六个子任务。我们通过精炼GPT-4o构建了涵盖五个主流形式化规范语言（Coq、Lean4、Dafny、ACSL和TLA+）的18,000个高质量指令-响应对，分布在六个形式化验证相关任务中。这些数据被划分为14,000个以上用于微调的FM-alpaca数据集和4,000个基准测试数据集FM-Bench。我们发现，当给定代码或详细推理步骤的描述时，大模型在编写证明片段方面表现出色。此外，微调带来了大约三倍的进步。有趣的是，我们发现使用形式化数据进行微调也能提升数学、推理和编程能力。我们希望我们的发现能激发进一步的研究，并发布了微调后的模型以促进后续的研究。 

---
# Are Transformers Able to Reason by Connecting Separated Knowledge in Training Data? 

**Title (ZH)**: Transformer模型能够在训练数据中将分离的知识进行关联并进行推理吗？ 

**Authors**: Yutong Yin, Zhaoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15857)  

**Abstract**: Humans exhibit remarkable compositional reasoning by integrating knowledge from various sources. For example, if someone learns ( B = f(A) ) from one source and ( C = g(B) ) from another, they can deduce ( C=g(B)=g(f(A)) ) even without encountering ( ABC ) together, showcasing the generalization ability of human intelligence. In this paper, we introduce a synthetic learning task, "FTCT" (Fragmented at Training, Chained at Testing), to validate the potential of Transformers in replicating this skill and interpret its inner mechanism. In the training phase, data consist of separated knowledge fragments from an overall causal graph. During testing, Transformers must infer complete causal graph traces by integrating these fragments. Our findings demonstrate that few-shot Chain-of-Thought prompting enables Transformers to perform compositional reasoning on FTCT by revealing correct combinations of fragments, even if such combinations were absent in the training data. Furthermore, the emergence of compositional reasoning ability is strongly correlated with the model complexity and training-testing data similarity. We propose, both theoretically and empirically, that Transformers learn an underlying generalizable program from training, enabling effective compositional reasoning during testing. 

**Abstract (ZH)**: 人类展示了通过整合来自各种来源的知识进行组合推理的非凡能力。例如，如果某人从一个来源得知(B = f(A))，从另一个来源得知(C = g(B))，他们甚至可以在未同时遇到(ABC)的情况下，推导出(C = g(B) = g(f(A)))，这彰显了人类智能的泛化能力。在这项研究中，我们引入了一个合成学习任务，称为“FTCT”（Fragmented at Training, Chained at Testing，训练时碎片化，测试时串联），以验证Transformer在复制这种能力的潜力，并解释其内部机制。在训练阶段，数据由整体因果图中的分离知识碎片组成。在测试阶段，Transformer必须通过整合这些片段推断完整的因果图路径。我们的研究发现，少量示例的链式推理提示能够使Transformer在FTCT任务中进行组合推理，通过揭示正确的片段组合，即使在训练数据中没有出现这些组合。此外，组合推理能力的出现与模型复杂度以及训练-测试数据相似性密切相关。我们从理论上和实验上提出，Transformer通过训练学习到一种基础可泛化的程序，在测试阶段能够有效进行组合推理。 

---
# Harnessing Diverse Perspectives: A Multi-Agent Framework for Enhanced Error Detection in Knowledge Graphs 

**Title (ZH)**: 利用多元视角：一种增强知识图谱错误检测的多代理框架 

**Authors**: Yu Li, Yi Huang, Guilin Qi, Junlan Feng, Nan Hu, Songlin Zhai, Haohan Xue, Yongrui Chen, Ruoyan Shen, Tongtong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15791)  

**Abstract**: Knowledge graphs are widely used in industrial applications, making error detection crucial for ensuring the reliability of downstream applications. Existing error detection methods often fail to effectively leverage fine-grained subgraph information and rely solely on fixed graph structures, while also lacking transparency in their decision-making processes, which results in suboptimal detection performance. In this paper, we propose a novel Multi-Agent framework for Knowledge Graph Error Detection (MAKGED) that utilizes multiple large language models (LLMs) in a collaborative setting. By concatenating fine-grained, bidirectional subgraph embeddings with LLM-based query embeddings during training, our framework integrates these representations to produce four specialized agents. These agents utilize subgraph information from different dimensions to engage in multi-round discussions, thereby improving error detection accuracy and ensuring a transparent decision-making process. Extensive experiments on FB15K and WN18RR demonstrate that MAKGED outperforms state-of-the-art methods, enhancing the accuracy and robustness of KG evaluation. For specific industrial scenarios, our framework can facilitate the training of specialized agents using domain-specific knowledge graphs for error detection, which highlights the potential industrial application value of our framework. Our code and datasets are available at this https URL. 

**Abstract (ZH)**: 知识图谱在工业应用中广泛应用，因此错误检测对于确保下游应用的可靠性至关重要。现有错误检测方法往往未能有效利用细粒度的子图信息，而是依赖于固定的图结构，同时在决策过程中缺乏透明性，导致检测性能不佳。本文提出了一种名为多代理框架的知识图谱错误检测（MAKGED）方法，该方法利用多个大型语言模型（LLMs）在协作环境中进行工作。通过在训练过程中将细粒度的双向子图嵌入与基于LLM的查询嵌入进行连接，该框架将这些表示整合成四种专门的代理。这些代理利用来自不同维度的子图信息进行多轮讨论，从而提高错误检测精度并确保决策过程的透明性。在FB15K和WN18RR上的 extensive 实验表明，MAKGED 在错误检测精度和鲁棒性方面优于最先进的方法。对于特定的工业场景，该框架可以根据专业知识图谱训练专门的代理进行错误检测，突显了该框架的潜在工业应用价值。我们的代码和数据集可以在此处访问：<此网址>。 

---
# LLM-powered Multi-agent Framework for Goal-oriented Learning in Intelligent Tutoring System 

**Title (ZH)**: 基于大型语言模型的多智能体框架：面向目标学习的智能辅导系统 

**Authors**: Tianfu Wang, Yi Zhan, Jianxun Lian, Zhengyu Hu, Nicholas Jing Yuan, Qi Zhang, Xing Xie, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.15749)  

**Abstract**: Intelligent Tutoring Systems (ITSs) have revolutionized education by offering personalized learning experiences. However, as goal-oriented learning, which emphasizes efficiently achieving specific objectives, becomes increasingly important in professional contexts, existing ITSs often struggle to deliver this type of targeted learning experience. In this paper, we propose GenMentor, an LLM-powered multi-agent framework designed to deliver goal-oriented, personalized learning within ITS. GenMentor begins by accurately mapping learners' goals to required skills using a fine-tuned LLM trained on a custom goal-to-skill dataset. After identifying the skill gap, it schedules an efficient learning path using an evolving optimization approach, driven by a comprehensive and dynamic profile of learners' multifaceted status. Additionally, GenMentor tailors learning content with an exploration-drafting-integration mechanism to align with individual learner needs. Extensive automated and human evaluations demonstrate GenMentor's effectiveness in learning guidance and content quality. Furthermore, we have deployed it in practice and also implemented it as an application. Practical human study with professional learners further highlights its effectiveness in goal alignment and resource targeting, leading to enhanced personalization. Supplementary resources are available at this https URL. 

**Abstract (ZH)**: 智能辅导系统（ITSs）通过提供个性化的学习体验，彻底改变了教育领域。然而，随着目标导向学习——强调高效达成具体目标——在专业环境中的重要性日益提高，现有的ITSs往往难以提供这种针对性的学习体验。本文提出了一种名为GenMentor的框架，该框架由大语言模型（LLM）驱动，旨在为ITS提供目标导向的个性化学习体验。GenMentor首先通过微调在自定义目标到技能数据集上训练的LLM，准确地将学习者的目标与所需的技能相匹配。在识别出技能缺口后，它利用一种不断演化的优化方法，根据学习者多维度状态的全面且动态的概况，规划一条高效的学习路径。此外，GenMentor采用了探索-草拟-整合机制来定制学习内容，以满足个体学习者的需求。广泛的自动化和人工评估表明，GenMentor在学习指导和内容质量方面具有显著效果。此外，我们已在实践中部署了GenMentor，并将其作为应用程序进行实施。对专业学习者的实际人类研究进一步突显了其在目标对齐和资源分配方面的有效性，从而增强了个性化水平。有关补充资源，请参阅此链接：[提供的链接]。 

---
# Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning 

**Title (ZH)**: 重新思考外部慢思考：从雪球错误到正确推理的概率 

**Authors**: Zeyu Gan, Yun Liao, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15602)  

**Abstract**: Test-time scaling, which is also often referred to as \textit{slow-thinking}, has been demonstrated to enhance multi-step reasoning in large language models (LLMs). However, despite its widespread utilization, the mechanisms underlying slow-thinking methods remain poorly understood. This paper explores the mechanisms of external slow-thinking from a theoretical standpoint. We begin by examining the snowball error effect within the LLM reasoning process and connect it to the likelihood of correct reasoning using information theory. Building on this, we show that external slow-thinking methods can be interpreted as strategies to mitigate the error probability. We further provide a comparative analysis of popular external slow-thinking approaches, ranging from simple to complex, highlighting their differences and interrelationships. Our findings suggest that the efficacy of these methods is not primarily determined by the specific framework employed, and that expanding the search scope or the model's internal reasoning capacity may yield more sustained improvements in the long term. We open-source our code at \url{this https URL}. 

**Abstract (ZH)**: 测试时缩放，也经常被称为“慢思考”，已被证明能够增强大型语言模型（LLMs）的多步推理能力。然而，尽管其广泛应用，慢思考方法背后的机制仍然知之甚少。本文从理论角度探讨了外部慢思考的机制。我们首先考察了LLM推理过程中的雪球错误效应，并用信息论将其与正确推理的可能性联系起来。在此基础上，我们表明外部慢思考方法可以被视为降低错误概率的策略。我们进一步对多种流行的外部慢思考方法进行了比较分析，涵盖从简单到复杂的各种方法，突显其差异和相互关系。我们的研究成果表明，这些方法的有效性主要不取决于所采用的具体框架，而是通过扩大搜索范围或增强模型内部推理能力，可以实现更具持续性的改进。我们已将代码开源，地址为：[此链接](this https URL)。 

---
# Causal Graphs Meet Thoughts: Enhancing Complex Reasoning in Graph-Augmented LLMs 

**Title (ZH)**: 因果图结合思维：增强图增强型大语言模型的复杂推理能力 

**Authors**: Hang Luo, Jian Zhang, Chujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14892)  

**Abstract**: In knowledge-intensive tasks, especially in high-stakes domains like medicine and law, it is critical not only to retrieve relevant information but also to provide causal reasoning and explainability. Large language models (LLMs) have achieved remarkable performance in natural language understanding and generation tasks. However, they often suffer from limitations such as difficulty in incorporating new knowledge, generating hallucinations, and explaining their reasoning process. To address these challenges, integrating knowledge graphs with Graph Retrieval-Augmented Generation (Graph RAG) has emerged as an effective solution. Traditional Graph RAG methods often rely on simple graph traversal or semantic similarity, which do not capture causal relationships or align well with the model's internal reasoning steps. This paper proposes a novel pipeline that filters large knowledge graphs to emphasize cause-effect edges, aligns the retrieval process with the model's chain-of-thought (CoT), and enhances reasoning through multi-stage path improvements. Experiments on medical question-answering tasks show consistent gains, with up to a 10\% absolute improvement across multiple large language models (LLMs). This approach demonstrates the value of combining causal reasoning with stepwise retrieval, leading to more interpretable and logically grounded solutions for complex queries. 

**Abstract (ZH)**: 在知识密集型任务中，特别是在医学和法律等高风险领域，不仅需要检索相关的信息，还需要提供因果推理和可解释性。大型语言模型（LLMs）在自然语言理解与生成任务中取得了显著的成果。然而，它们往往存在难以融入新知识、生成虚假信息以及解释推理过程等局限性。为了解决这些挑战，将知识图谱与图检索增强生成（Graph RAG）相结合已成为有效的方法。传统的Graph RAG方法通常依赖于简单的图遍历或语义相似度，无法捕捉因果关系或与模型的内部推理步骤很好地对齐。本文提出了一种新的管道，该管道通过对大规模知识图谱进行过滤以强调因果关系边，将检索过程与模型的链式思维（CoT）对齐，并通过多阶段路径改进来增强推理。在医学问答任务上的实验结果表明，这种方法在多个大型语言模型（LLMs）上表现出了一致的改进，绝对改进幅度最高可达10%。该方法证明了将因果推理与逐步检索相结合的价值，从而使复杂查询的结果更易解释且逻辑基础更牢固。 

---
# Evaluating The Performance of Using Large Language Models to Automate Summarization of CT Simulation Orders in Radiation Oncology 

**Title (ZH)**: 评估使用大型语言模型自动化放射肿瘤学CT模拟订单总结的效果 

**Authors**: Meiyun Cao, Shaw Hu, Jason Sharp, Edward Clouser, Jason Holmes, Linda L. Lam, Xiaoning Ding, Diego Santos Toesca, Wendy S. Lindholm, Samir H. Patel, Sujay A. Vora, Peilong Wang, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.16309)  

**Abstract**: Purpose: This study aims to use a large language model (LLM) to automate the generation of summaries from the CT simulation orders and evaluate its performance.
Materials and Methods: A total of 607 CT simulation orders for patients were collected from the Aria database at our institution. A locally hosted Llama 3.1 405B model, accessed via the Application Programming Interface (API) service, was used to extract keywords from the CT simulation orders and generate summaries. The downloaded CT simulation orders were categorized into seven groups based on treatment modalities and disease sites. For each group, a customized instruction prompt was developed collaboratively with therapists to guide the Llama 3.1 405B model in generating summaries. The ground truth for the corresponding summaries was manually derived by carefully reviewing each CT simulation order and subsequently verified by therapists. The accuracy of the LLM-generated summaries was evaluated by therapists using the verified ground truth as a reference.
Results: About 98% of the LLM-generated summaries aligned with the manually generated ground truth in terms of accuracy. Our evaluations showed an improved consistency in format and enhanced readability of the LLM-generated summaries compared to the corresponding therapists-generated summaries. This automated approach demonstrated a consistent performance across all groups, regardless of modality or disease site.
Conclusions: This study demonstrated the high precision and consistency of the Llama 3.1 405B model in extracting keywords and summarizing CT simulation orders, suggesting that LLMs have great potential to help with this task, reduce the workload of therapists and improve workflow efficiency. 

**Abstract (ZH)**: 目的：本研究旨在利用大型语言模型（LLM）自动化生成CT模拟订单的摘要，并评估其性能。
材料与方法：从我们机构的Aria数据库中收集了共607份患者的CT模拟订单。通过应用程序编程接口（API）服务访问我们本地托管的Llama 3.1 405B模型，用于从CT模拟订单中提取关键词并生成摘要。下载的CT模拟订单根据治疗方式和疾病部位分类为七个组别。对于每个组别，研究人员与治疗师合作开发了定制的指令提示，以指导Llama 3.1 405B模型生成摘要。每个组别的摘要地面真相通过仔细审查每个CT模拟订单并由治疗师验证后人为构建。治疗师使用验证过的地面真相作为参考，评估LLM生成的摘要的准确性。

结果：LLM生成的摘要与手工生成的地面真相在准确性方面的一致性约为98%。评估结果显示，与相应治疗师生成的摘要相比，LLM生成的摘要在格式上更具一致性且可读性更高。该自动化方法在所有组别中均表现出一致的性能，不受治疗方式或疾病部位的影响。

结论：本研究表明，Llama 3.1 405B模型在提取关键词和总结CT模拟订单方面具有高精度和一致性，表明LLM有巨大潜力帮助完成此任务，减轻治疗师的工作负担并提高工作效率。 

---
# Large Models in Dialogue for Active Perception and Anomaly Detection 

**Title (ZH)**: 大型模型在对话中的主动感知与异常检测 

**Authors**: Tzoulio Chamiti, Nikolaos Passalis, Anastasios Tefas  

**Link**: [PDF](https://arxiv.org/pdf/2501.16300)  

**Abstract**: Autonomous aerial monitoring is an important task aimed at gathering information from areas that may not be easily accessible by humans. At the same time, this task often requires recognizing anomalies from a significant distance or not previously encountered in the past. In this paper, we propose a novel framework that leverages the advanced capabilities provided by Large Language Models (LLMs) to actively collect information and perform anomaly detection in novel scenes. To this end, we propose an LLM based model dialogue approach, in which two deep learning models engage in a dialogue to actively control a drone to increase perception and anomaly detection accuracy. We conduct our experiments in a high fidelity simulation environment where an LLM is provided with a predetermined set of natural language movement commands mapped into executable code functions. Additionally, we deploy a multimodal Visual Question Answering (VQA) model charged with the task of visual question answering and captioning. By engaging the two models in conversation, the LLM asks exploratory questions while simultaneously flying a drone into different parts of the scene, providing a novel way to implement active perception. By leveraging LLMs reasoning ability, we output an improved detailed description of the scene going beyond existing static perception approaches. In addition to information gathering, our approach is utilized for anomaly detection and our results demonstrate the proposed methods effectiveness in informing and alerting about potential hazards. 

**Abstract (ZH)**: 自主导航监测是一项重要的任务，旨在从人类难以到达的区域收集信息。同时，这项任务往往需要在远距离或以前未遇到的场景中识别异常。在本文中，我们提出了一种新的框架，利用大型语言模型（LLMs）的高级能力，主动收集信息并进行异常检测。为此，我们提出了一种基于LLM的模型对话方法，其中两个深度学习模型进行对话，主动控制无人机以提高感知和异常检测的准确性。我们在一个高保真仿真环境中进行了实验，其中LLM被提供了一组预先确定的自然语言移动指令，这些指令映射为可执行代码功能。此外，我们部署了一个多模态视觉问答（VQA）模型，负责视觉问答和图像字幕生成的任务。通过让两个模型进行对话，LLM在飞行无人机探索不同场景部分时提出探索性问题，从而实现了一种新颖的主动感知方式。通过利用LLM的推理能力，我们输出了超越现有静态感知方法的改进详细场景描述。除了信息收集，我们的方法还用于异常检测，实验结果证明了所提出方法在告知和预警潜在危险方面的效果。 

---
# Language-Based Bayesian Optimization Research Assistant (BORA) 

**Title (ZH)**: 语言驱动的贝叶斯优化研究助手（BORA） 

**Authors**: Abdoulatif Cissé, Xenophon Evangelopoulos, Vladimir V. Gusev, Andrew I. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2501.16224)  

**Abstract**: Many important scientific problems involve multivariate optimization coupled with slow and laborious experimental measurements. These complex, high-dimensional searches can be defined by non-convex optimization landscapes that resemble needle-in-a-haystack surfaces, leading to entrapment in local minima. Contextualizing optimizers with human domain knowledge is a powerful approach to guide searches to localized fruitful regions. However, this approach is susceptible to human confirmation bias and it is also challenging for domain experts to keep track of the rapidly expanding scientific literature. Here, we propose the use of Large Language Models (LLMs) for contextualizing Bayesian optimization (BO) via a hybrid optimization framework that intelligently and economically blends stochastic inference with domain knowledge-based insights from the LLM, which is used to suggest new, better-performing areas of the search space for exploration. Our method fosters user engagement by offering real-time commentary on the optimization progress, explaining the reasoning behind the search strategies. We validate the effectiveness of our approach on synthetic benchmarks with up to 15 independent variables and demonstrate the ability of LLMs to reason in four real-world experimental tasks where context-aware suggestions boost optimization performance substantially. 

**Abstract (ZH)**: 许多重要的科学问题涉及多变量优化与缓慢且繁琐的实验测量相结合。这类复杂且高维度的搜索可以由非凸优化景观定义，其类似于针尖在干草堆上的表面，这可能导致陷入局部极小值。利用人类领域知识来上下文化优化器是一种强大的方法，可以引导搜索到局部具有成果的区域。然而，这种方法容易受到人类确认偏见的影响，并且对于领域专家来说，跟踪迅速扩展的科学文献也极具挑战性。在此，我们提出使用大型语言模型（LLMs）通过将随机推理与来自LLM的基于领域知识的见解智能而经济地结合的混合优化框架，来上下文化贝叶斯优化（BO）。LLM用于建议搜索空间中新的、表现更好的区域以进行探索。我们的方法通过提供实时的优化进展评论来促进用户的参与，并解释搜索策略背后的推理。我们在多达15个独立变量的合成基准上验证了我们方法的有效性，并在四个真实世界的实验任务中展示了LLM能够基于上下文提供建议以大大提升优化性能的能力。 

---
# Raiders of the Lost Dependency: Fixing Dependency Conflicts in Python using LLMs 

**Title (ZH)**: 《寻回缺失的依赖：使用大语言模型解决Python中的依赖冲突》 

**Authors**: Antony Bartlett, Cynthia Liem, Annibale Panichella  

**Link**: [PDF](https://arxiv.org/pdf/2501.16191)  

**Abstract**: Fixing Python dependency issues is a tedious and error-prone task for developers, who must manually identify and resolve environment dependencies and version constraints of third-party modules and Python interpreters. Researchers have attempted to automate this process by relying on large knowledge graphs and database lookup tables. However, these traditional approaches face limitations due to the variety of dependency error types, large sets of possible module versions, and conflicts among transitive dependencies. This study explores the potential of using large language models (LLMs) to automatically fix dependency issues in Python programs. We introduce PLLM (pronounced "plum"), a novel technique that employs retrieval-augmented generation (RAG) to help an LLM infer Python versions and required modules for a given Python file. PLLM builds a testing environment that iteratively (1) prompts the LLM for module combinations, (2) tests the suggested changes, and (3) provides feedback (error messages) to the LLM to refine the fix. This feedback cycle leverages natural language processing (NLP) to intelligently parse and interpret build error messages. We benchmark PLLM on the Gistable HG2.9K dataset, a collection of challenging single-file Python gists. We compare PLLM against two state-of-the-art automatic dependency inference approaches, namely PyEGo and ReadPyE, w.r.t. the ability to resolve dependency issues. Our results indicate that PLLM can fix more dependency issues than the two baselines, with +218 (+15.97%) more fixes over ReadPyE and +281 (+21.58%) over PyEGo. Our deeper analyses suggest that PLLM is particularly beneficial for projects with many dependencies and for specific third-party numerical and machine-learning modules. Our findings demonstrate the potential of LLM-based approaches to iteratively resolve Python dependency issues. 

**Abstract (ZH)**: 修复 Python 依赖问题对开发者来说是一个繁琐且容易出错的任务，他们需要手动识别和解决第三方模块和 Python 解释器的环境依赖及其版本限制。研究人员尝试通过依赖大型知识图谱和数据库查找表来自动化这一过程。然而，传统方法由于依赖错误类型多样、可能的模块版本众多以及传递依赖之间的冲突，面临着局限性。本研究探讨了使用大规模语言模型（LLMs）自动修复 Python 程序依赖问题的潜力。我们提出了一种名为 PLLM（发音为“plum”）的新技术，该技术利用检索增强生成（RAG）帮助 LLM 推断给定 Python 文件所需的 Python 版本和模块组合。PLLM 构建了一个测试环境，该环境通过以下步骤迭代工作：（1）提示 LLM 提出模块组合，（2）测试建议的更改，（3）向 LLM 提供反馈（错误消息），以便进一步优化修复。这一反馈循环利用自然语言处理（NLP）智能解析和解释构建错误消息。我们使用包含具有挑战性的单文件 Python gists 的 Gistable HG2.9K 数据集对 PLLM 进行基准测试，并将 PLLM 与两种最新的自动依赖推理方法 PyEGo 和 ReadPyE 进行比较，比较它们解决依赖问题的能力。结果表明，PLLM 比两种基准方法能够修复更多的依赖问题，相对于 ReadPyE 多修复 +218（+15.97%）个问题，相对于 PyEGo 多修复 +281（+21.58%）个问题。我们更深入的分析表明，PLLM 特别适用于具有众多依赖项的项目，以及特定的第三方数值和机器学习模块。我们的研究结果表明，基于 LLM 的方法有潜力逐步解决 Python 依赖问题。 

---
# PRISMe: A Novel LLM-Powered Tool for Interactive Privacy Policy Assessment 

**Title (ZH)**: PRISMe：一种新型的基于大语言模型的互动隐私政策评估工具 

**Authors**: Vincent Freiberger, Arthur Fleig, Erik Buchmann  

**Link**: [PDF](https://arxiv.org/pdf/2501.16033)  

**Abstract**: Protecting online privacy requires users to engage with and comprehend website privacy policies, but many policies are difficult and tedious to read. We present PRISMe (Privacy Risk Information Scanner for Me), a novel Large Language Model (LLM)-driven privacy policy assessment tool, which helps users to understand the essence of a lengthy, complex privacy policy while browsing. The tool, a browser extension, integrates a dashboard and an LLM chat. One major contribution is the first rigorous evaluation of such a tool. In a mixed-methods user study (N=22), we evaluate PRISMe's efficiency, usability, understandability of the provided information, and impacts on awareness. While our tool improves privacy awareness by providing a comprehensible quick overview and a quality chat for in-depth discussion, users note issues with consistency and building trust in the tool. From our insights, we derive important design implications to guide future policy analysis tools. 

**Abstract (ZH)**: 保护在线隐私要求用户参与并理解网站隐私政策，但许多政策的内容冗长且难以阅读。我们提出了PRISMe（Privacy Risk Information Scanner for Me），这是一种基于大型语言模型（LLM）的新型隐私政策评估工具，帮助用户在浏览过程中理解复杂隐私政策的实质。该工具是一个浏览器扩展程序，集成了仪表板和LLM聊天功能。一个主要贡献是我们首次进行了此类工具的严格评估。在一项混合方法用户研究（n=22）中，我们评估了PRISMe的效率、易用性、提供信息的可理解性以及对隐私意识的影响。虽然我们的工具通过提供易于理解的快速概览和高质量的聊天对话来提高隐私意识，但用户指出工具的一致性和建立信任方面存在一些问题。从我们的研究中，我们得出了重要的设计启示，以指导未来政策分析工具的发展。 

---
# Large Language Models to Diffusion Finetuning 

**Title (ZH)**: 大型语言模型应用于扩散微调 

**Authors**: Edoardo Cetin, Tianyu Zhao, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15781)  

**Abstract**: We propose a new finetuning method to provide pre-trained large language models (LMs) the ability to scale test-time compute through the diffusion framework. By increasing the number of diffusion steps, we show our finetuned models achieve monotonically increasing accuracy, directly translating to improved performance across downstream tasks. Furthermore, our finetuned models can expertly answer questions on specific topics by integrating powerful guidance techniques, and autonomously determine the compute required for a given problem by leveraging adaptive ODE solvers. Our method is universally applicable to any foundation model pre-trained with a cross-entropy loss and does not modify any of its original weights, fully preserving its strong single-step generation capabilities. We show our method is more effective and fully compatible with traditional finetuning approaches, introducing an orthogonal new direction to unify the strengths of the autoregressive and diffusion frameworks. 

**Abstract (ZH)**: 我们提出了一种新的微调方法，通过扩散框架赋予预训练的大语言模型（LMs）在测试时扩展计算量的能力。通过增加扩散步骤的数量，我们展示了微调后的模型在准确性上呈现单调递增的趋势，直接转化为下游任务上的性能提升。此外，我们的微调模型能够通过整合强大的指导技术，专家般地回答特定主题的问题，并通过利用自适应ODE求解器自主确定给定问题所需的计算量。该方法适用于任何基于交叉熵损失预训练的基础模型，并未修改其任何原始权重，完全保留了其强大的单步生成能力。我们证明该方法比传统微调方法更为有效，并且完全兼容传统微调方法，引入了一种新的正交方向，统一了自回归框架和扩散框架的优点。 

---
# IndicMMLU-Pro: Benchmarking the Indic Large Language Models 

**Title (ZH)**: IndicMMLU-Pro：评估印度语系大型语言模型 

**Authors**: Sankalp KJ, Ashutosh Kumar, Laxmaan Balaji, Nikunj Kotecha, Vinija Jain, Aman Chadha, Sreyoshi Bhaduri  

**Link**: [PDF](https://arxiv.org/pdf/2501.15747)  

**Abstract**: Known by more than 1.5 billion people in the Indian subcontinent, Indic languages present unique challenges and opportunities for natural language processing (NLP) research due to their rich cultural heritage, linguistic diversity, and complex structures. IndicMMLU-Pro is a comprehensive benchmark designed to evaluate Large Language Models (LLMs) across Indic languages, building upon the MMLU Pro (Massive Multitask Language Understanding) framework. Covering major languages such as Hindi, Bengali, Gujarati, Marathi, Kannada, Punjabi, Tamil, Telugu, and Urdu, our benchmark addresses the unique challenges and opportunities presented by the linguistic diversity of the Indian subcontinent. This benchmark encompasses a wide range of tasks in language comprehension, reasoning, and generation, meticulously crafted to capture the intricacies of Indian languages. IndicMMLU-Pro provides a standardized evaluation framework to push the research boundaries in Indic language AI, facilitating the development of more accurate, efficient, and culturally sensitive models. This paper outlines the benchmarks' design principles, task taxonomy, data collection methodology, and presents baseline results from state-of-the-art multilingual models. 

**Abstract (ZH)**: 印度次大陆的15亿多人口所熟知的印度语言在自然语言处理（NLP）研究中因其丰富的文化遗产、语言多样性和复杂结构而面临着独特的挑战和机遇。IndicMMLU-Pro 是一个全面的基准测试，旨在评估大型语言模型（LLMs）在印度语言中的表现，该基准测试是在MMLU Pro（大规模多任务语言理解）框架的基础上构建的。涵盖哈林语、孟加拉语、古吉拉特语、马拉地语、卡纳达语、旁遮普语、泰米尔语、泰卢固语和乌尔都语等主要语言，该基准测试针对印度次大陆语言多样性的独特挑战和机遇。这一基准测试涵盖了语言理解、推理和生成等广泛的任务，精心设计以捕捉印度语言的复杂性。IndicMMLU-Pro 提供了一个标准化的评估框架，以推动印度语言AI的研究边界，促进更准确、高效和文化敏感模型的发展。本文概述了基准测试的设计原则、任务分类学、数据收集方法，并展示了最先进的多语言模型的基线结果。 

---
# Gensors: Authoring Personalized Visual Sensors with Multimodal Foundation Models and Reasoning 

**Title (ZH)**: Gensors：使用多模态基础模型和推理构建个性化的视觉传感器 

**Authors**: Michael Xieyang Liu, Savvas Petridis, Vivian Tsai, Alexander J. Fiannaca, Alex Olwal, Michael Terry, Carrie J. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2501.15727)  

**Abstract**: Multimodal large language models (MLLMs), with their expansive world knowledge and reasoning capabilities, present a unique opportunity for end-users to create personalized AI sensors capable of reasoning about complex situations. A user could describe a desired sensing task in natural language (e.g., "alert if my toddler is getting into mischief"), with the MLLM analyzing the camera feed and responding within seconds. In a formative study, we found that users saw substantial value in defining their own sensors, yet struggled to articulate their unique personal requirements and debug the sensors through prompting alone. To address these challenges, we developed Gensors, a system that empowers users to define customized sensors supported by the reasoning capabilities of MLLMs. Gensors 1) assists users in eliciting requirements through both automatically-generated and manually created sensor criteria, 2) facilitates debugging by allowing users to isolate and test individual criteria in parallel, 3) suggests additional criteria based on user-provided images, and 4) proposes test cases to help users "stress test" sensors on potentially unforeseen scenarios. In a user study, participants reported significantly greater sense of control, understanding, and ease of communication when defining sensors using Gensors. Beyond addressing model limitations, Gensors supported users in debugging, eliciting requirements, and expressing unique personal requirements to the sensor through criteria-based reasoning; it also helped uncover users' "blind spots" by exposing overlooked criteria and revealing unanticipated failure modes. Finally, we discuss how unique characteristics of MLLMs--such as hallucinations and inconsistent responses--can impact the sensor-creation process. These findings contribute to the design of future intelligent sensing systems that are intuitive and customizable by everyday users. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）凭借其广泛的世界知识和推理能力，为终端用户创造个性化的AI传感器提供了独特的机会，这些传感器能够对复杂情况进行推理。用户可以用自然语言描述一个期望的感知任务（例如，“如果我的幼儿在搞破坏，请报警”），MLLMs 分析摄像头视频并在几秒钟内做出响应。在一项形成性研究中，我们发现用户在定义自己的传感器方面看到了巨大的价值，但他们在表达独特的个人需求及仅通过提示进行调试方面遇到困难。为了解决这些问题，我们开发了Gensors系统，它赋予用户通过MLLMs 的推理能力定义自定义传感器的能力。Gensors 1) 通过自动生成和手动创建传感器标准帮助用户提取需求，2) 通过允许用户并行隔离和测试各个标准的方式支持调试，3) 根据用户提供的图片建议额外的标准，4) 提出测试案例帮助用户对传感器进行“压力测试”，以应对可能不可预见的情况。在一项用户研究中，参与者在使用Gensors定义传感器时报告了更大的控制感、理解和沟通便利性。除了解决模型限制外，Gensors还支持用户通过基于标准的推理调试、提取需求并表达独特的个人需求，同时通过暴露被忽视的标准并揭示不可预见的故障模式，帮助用户发现“盲点”。最后，我们讨论了MLLMs的独特特征（如幻觉和不一致的响应）如何影响传感器创建过程。这些发现为未来直观且可定制的智能感知系统的设计做出了贡献。 

---
# Advancing Generative Artificial Intelligence and Large Language Models for Demand Side Management with Electric Vehicles 

**Title (ZH)**: 推动生成式人工智能和大型语言模型在电动车辆需求侧管理中的应用与发展 

**Authors**: Hanwen Zhang, Ruichen Zhang, Wei Zhang, Dusit Niyato, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15544)  

**Abstract**: Generative artificial intelligence, particularly through large language models (LLMs), is poised to transform energy optimization and demand side management (DSM) within microgrids. This paper explores the integration of LLMs into energy management, emphasizing their roles in automating the optimization of DSM strategies with electric vehicles. We investigate challenges and solutions associated with DSM and explore the new opportunities presented by leveraging LLMs. Then, We propose an innovative solution that enhances LLMs with retrieval-augmented generation for automatic problem formulation, code generation, and customizing optimization. We present a case study to demonstrate the effectiveness of our proposed solution in charging scheduling and optimization for electric vehicles, highlighting our solution's significant advancements in energy efficiency and user adaptability. This work underscores the potential of LLMs for energy optimization and fosters a new era of intelligent DSM solutions. 

**Abstract (ZH)**: 生成式人工智能，尤其是在大型语言模型（LLMs）的帮助下，有望重塑微网中的能源优化和需求侧管理（DSM）。本文探讨了将LLMs集成到能源管理系统中的方法，强调了它们在自动化电动汽车DSM策略优化方面的作用。我们研究了DSM面临的挑战及其解决方案，并探索了利用LLMs所带来的新机遇。接下来，我们提出了一种创新解决方案，通过检索增强生成技术增强LLMs，以实现自动问题表述、代码生成和定制优化。我们通过一个案例研究展示了该解决方案在电动汽车充电调度和优化方面的有效性，突显了该解决方案在提高能源效率和用户适应性方面的显著进步。本文强调了LLMs在能源优化领域的潜力，并促进了新一代智能DSM解决方案的发展。 

---
# Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values? 

**Title (ZH)**: 注意价值与行动之间的差距：大语言模型的行动是否与其价值观一致？ 

**Authors**: Hua Shen, Nicholas Clark, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2501.15463)  

**Abstract**: Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps. 

**Abstract (ZH)**: 现有的研究主要通过评估大型语言模型（LLM）对特定价值观的倾向来评价其价值。然而，“价值-行动差距”现象揭示了个体在现实世界情境中所声明的价值与其行为之间的差异，这一现象根植于环境和社会心理学领域。在多大程度上，LLM在受到其声明的价值影响后所表现出的行为与其声明的价值之间存在类似的差距？本研究引入了ValueActionLens，这是一种评估LLM声明的价值与其价值驱动行为之间对齐程度的评价框架。该框架包括生成包含14,800个跨十二种文化与十一种社会议题的价值驱动行为的数据集，并设立两个任务来评估LLM声明的价值倾向与其价值驱动行为在三种不同对齐度量标准下的对齐程度。广泛的实验表明，LLM声明的价值与行为之间的对齐程度并不理想，且在不同场景和模型中存在显著差异。对对齐不当结果的分析揭示了某些价值-行动差距可能带来的潜在危害。为预测价值-行动差距，我们还发现利用推理解释可以提高预测性能。这些发现强调了仅依赖LLM声明的价值来预测其行为的风险，并强调了对LLM的价值及其价值-行动差距进行情境意识评估的重要性。 

---
# Token Democracy: The Architectural Limits of Alignment in Transformer-Based Language Models 

**Title (ZH)**: 代币民主：基于变压器的语言模型对齐的架构限制 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2501.15446)  

**Abstract**: Modern language models paradoxically combine unprecedented capability with persistent vulnerability in that they can draft poetry yet cannot reliably refuse harmful requests. We reveal this fragility stems not from inadequate training, but from a fundamental architectural limitation: transformers process all tokens as equals. Transformers operate as computational democracies, granting equal voice to all tokens. This is a design tragically unsuited for AGI, where we cannot risk adversarial "candidates" hijacking the system. Through formal analysis, we demonstrate that safety instructions fundamentally lack privileged status in transformer architectures, that they compete with adversarial inputs in the same computational arena, making robust alignment through prompting or fine-tuning inherently limited. This "token democracy" explains why jailbreaks bypass even extensively safety-trained models and why positional shifts erode prompt effectiveness. Our work systematizes practitioners' tacit knowledge into an architectural critique, showing current alignment approaches create mere preferences, not constraints. 

**Abstract (ZH)**: 现代语言模型在前所未有的能力和持续的脆弱性之间表现出一种悖论。它们能够创作诗歌，但却不能可靠地拒绝有害请求。我们揭示这种脆弱性并非源于训练不足，而是源自根本的架构限制：变换器将所有标记视为平等。变换器作为一种计算民主体制，赋予所有标记平等的话语权。这一设计对AGI来说是灾难性的，因为我们无法在其中冒险让对抗性“候选人”劫持系统。通过形式分析，我们证明了安全指令在变换器架构中并未享有特权地位，它们与其他对抗性输入在相同的计算领域中竞争，使得通过提示或微调实现稳健对齐本质上是有限的。这种“标记民主”解释了为什么突破（jailbreaks）能够绕过甚至最广泛训练的安全模型，以及位置偏移会削弱提示效果。我们的研究将实践者的隐性知识系统化，揭示当前的对齐方法仅创造了偏好而非约束。 

---
# Semantic Layered Embedding Diffusion in Large Language Models for Multi-Contextual Consistency 

**Title (ZH)**: 大型语言模型中多上下文一致性中的语义分层嵌入扩散 

**Authors**: Irin Kabakum, Thomas Montgomery, Daniel Ravenwood, Genevieve Harrington  

**Link**: [PDF](https://arxiv.org/pdf/2501.15405)  

**Abstract**: The Semantic Layered Embedding Diffusion (SLED) mechanism redefines the representation of hierarchical semantics within transformer-based architectures, enabling enhanced contextual consistency across a wide array of linguistic tasks. By introducing a multi-layered diffusion process grounded in spectral analysis, it achieves a complex balance between global and local semantic coherence. Experimental results demonstrate significant improvements in perplexity and BLEU scores, emphasizing the mechanism's ability to adapt effectively across diverse domains, including multilingual and cross-domain text generation. A rigorous mathematical framework underpins the embedding diffusion process, incorporating weighted adjacency matrices, kernel-based refinements, and dynamic layer-wise normalization. Error distribution analysis reveals that SLED addresses challenges in semantic alignment and coherence, outperforming baseline approaches across varied benchmarks. Scalability studies illustrate that its performance gains are maintained consistently across different model sizes, reflecting a practical balance between computational efficiency and linguistic precision. The implementation also achieves energy efficiency, reducing resource consumption during training and inference phases without compromising accuracy. Qualitative case studies further validate its adaptability to extended narratives and context-intensive scenarios, highlighting the mechanism's potential for real-world applications. SLED offers a different perspective on embedding design and its implications for advancing language modeling. 

**Abstract (ZH)**: SLED（语义分层嵌入扩散）机制重新定义了基于变换器架构中的层级语义表示，从而在广泛的语言任务中增强了上下文一致性。通过基于谱分析引入多层扩散过程，它在全局和局部语义一致性之间实现了复杂的平衡。实验结果表明，在困惑度和BLEU评分上取得了显著改进，强调了该机制在多种不同领域的有效适应能力，包括多语言和跨域文本生成。该机制嵌入扩散过程的数学框架严谨，结合了加权邻接矩阵、核基础改进和动态层归一化。误差分布分析表明，SLED在解决语义对齐和连贯性的挑战方面优于基础方法，在多种基准测试中表现更佳。扩展性研究表明，其性能增益在不同模型规模下保持一致，反映了在计算效率和语言精度之间的实用平衡。同时，实现也实现了能效优化，在训练和推理阶段降低资源消耗而不影响准确性。定性案例研究进一步验证了其对扩展叙事和情境密集场景的适应性，突显了该机制在实际应用中的潜力。SLED为嵌入设计及其对推进语言建模的影响提供了新的视角。 

---
# Large Language Models as Theory of Mind Aware Generative Agents with Counterfactual Reflection 

**Title (ZH)**: 大型语言模型作为具备反事实反思能力的理论理解生成代理 

**Authors**: Bo Yang, Jiaxian Guo, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2501.15355)  

**Abstract**: Recent studies have increasingly demonstrated that large language models (LLMs) possess significant theory of mind (ToM) capabilities, showing the potential for simulating the tracking of mental states in generative agents. In this study, we propose a novel paradigm called ToM-agent, designed to empower LLMs-based generative agents to simulate ToM in open-domain conversational interactions. ToM-agent disentangles the confidence from mental states, facilitating the emulation of an agent's perception of its counterpart's mental states, such as beliefs, desires, and intentions (BDIs). Using past conversation history and verbal reflections, ToM-Agent can dynamically adjust counterparts' inferred BDIs, along with related confidence levels. We further put forth a counterfactual intervention method that reflects on the gap between the predicted responses of counterparts and their real utterances, thereby enhancing the efficiency of reflection. Leveraging empathetic and persuasion dialogue datasets, we assess the advantages of implementing the ToM-agent with downstream tasks, as well as its performance in both the first-order and the \textit{second-order} ToM. Our findings indicate that the ToM-agent can grasp the underlying reasons for their counterpart's behaviors beyond mere semantic-emotional supporting or decision-making based on common sense, providing new insights for studying large-scale LLMs-based simulation of human social behaviors. 

**Abstract (ZH)**: 近年来，大量研究表明大型语言模型（LLMs）具备显著的理论心智（ToM）能力，显示出在生成代理中模拟追踪心智状态的潜在可能性。在此研究中，我们提出了一种名为ToM-agent的新范式，旨在赋予基于LLMs的生成代理模拟ToM的能力，特别是在开放式领域对话交互中的应用。ToM-agent将信心与心智状态分离，促进代理对其对应方的心智状态（如信念、欲望和意图，BDIs）感知的模拟。利用过去的对话历史和言语反思，ToM-Agent可以动态调整对对应方的推断BDIs及其相关信心水平进行调整。我们还提出了一个假设干预方法，通过反映预测响应与实际言辞之间的差距，从而增强反思效率。借助同理心和说服性对话数据集，我们评估了实施ToM-agent在下游任务中的优势，以及其在一级和二级ToM中的性能表现。我们的研究表明，ToM-agent能够把握对应方行为背后的根本原因，而不仅仅依赖于语义情感支撑或基于常识的决策制定，为研究大规模LLMs基于的心智模型模拟人类社会行为提供了新的视角。 

---
# Option-ID Based Elimination For Multiple Choice Questions 

**Title (ZH)**: 基于Option-ID的多项选择题消除方法 

**Authors**: Zhenhao Zhu, Bulou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15175)  

**Abstract**: Multiple choice questions (MCQs) are a common and important task for evaluating large language models (LLMs). Based on common strategies humans use when answering MCQs, the process of elimination has been proposed as an effective problem-solving method. Existing methods to the process of elimination generally fall into two categories: one involves having the model directly select the incorrect answer, while the other involves scoring the options. However, both methods incur high computational costs and often perform worse than methods that answer based on option ID. To address this issue, this paper proposes a process of elimination based on option ID. We select 10 LLMs and conduct zero-shot experiments on 7 different datasets. The experimental results demonstrate that our method significantly improves the model's performance. Further analysis reveals that the sequential elimination strategy can effectively enhance the model's reasoning ability. Additionally, we find that sequential elimination is also applicable to few-shot settings and can be combined with debias methods to further improve model performance. 

**Abstract (ZH)**: 多项选择题（MCQs）是评估大型语言模型（LLMs）的一种常见且重要的任务。基于人类在回答MCQs时通常使用的方法，逐项排除策略被提出作为一种有效的问题解决方法。现有的逐项排除方法大体可分为两类：一类是让模型直接选择错误答案，另一类是评分选项。然而，这两种方法都带来了较高的计算成本，并且通常不如基于选项ID作答的方法表现更好。为了解决这一问题，本文提出了一种基于选项ID的逐项排除策略。我们选择了10个LLM，并在7个不同的数据集上进行了零样本实验。实验结果表明，我们的方法显著提高了模型的性能。进一步分析表明，序列排除策略可以有效增强模型的推理能力。此外，我们发现序列排除策略也适用于少样本设置，并且可以与去偏见方法相结合，进一步提高模型性能。 

---
# Can Large Language Models Be Trusted as Black-Box Evolutionary Optimizers for Combinatorial Problems? 

**Title (ZH)**: 大型语言模型可以作为组合问题的黑盒进化优化器被信任吗？ 

**Authors**: Jie Zhao, Tao Wen, Kang Hao Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2501.15081)  

**Abstract**: Evolutionary computation excels in complex optimization but demands deep domain knowledge, restricting its accessibility. Large Language Models (LLMs) offer a game-changing solution with their extensive knowledge and could democratize the optimization paradigm. Although LLMs possess significant capabilities, they may not be universally effective, particularly since evolutionary optimization encompasses multiple stages. It is therefore imperative to evaluate the suitability of LLMs as evolutionary optimizer (EVO). Thus, we establish a series of rigid standards to thoroughly examine the fidelity of LLM-based EVO output in different stages of evolutionary optimization and then introduce a robust error-correction mechanism to mitigate the output uncertainty. Furthermore, we explore a cost-efficient method that directly operates on entire populations with excellent effectiveness in contrast to individual-level optimization. Through extensive experiments, we rigorously validate the performance of LLMs as operators targeted for combinatorial problems. Our findings provide critical insights and valuable observations, advancing the understanding and application of LLM-based optimization. 

**Abstract (ZH)**: 进化计算在复杂优化方面表现出色，但需深厚的领域知识，限制了其可访问性。大型语言模型（LLMs）凭借其广泛的知识提供了变革性的解决方案，有可能普及优化范式。尽管LLMs具备显著的能力，但在进化优化的多个阶段中，它们可能并不普遍有效。因此，有必要评估LLMs作为进化优化器（EVO）的适用性。为此，我们制定了一系列严格的评价标准，全面考察基于LLM的EVO在进化优化不同阶段的输出准确度，并引入了稳健的错误校正机制以减轻输出的不确定性。此外，我们探索了一种成本效益高的方法，可以直接在种群层面进行操作，与基于个体的优化相比效果显著。通过广泛的实验，我们严格验证了LLMs作为针对组合问题的操作符的性能。我们的研究结果提供了宝贵的见解和观察，推动了基于LLM的优化的理解和应用。 

---
# PatentLMM: Large Multimodal Model for Generating Descriptions for Patent Figures 

**Title (ZH)**: PatentLMM：大型多模态模型，用于生成专利图表的描述 

**Authors**: Shreya Shukla, Nakul Sharma, Manish Gupta, Anand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2501.15074)  

**Abstract**: Writing comprehensive and accurate descriptions of technical drawings in patent documents is crucial to effective knowledge sharing and enabling the replication and protection of intellectual property. However, automation of this task has been largely overlooked by the research community. To this end, we introduce PatentDesc-355K, a novel large-scale dataset containing ~355K patent figures along with their brief and detailed textual descriptions extracted from more than 60K US patent documents. In addition, we propose PatentLMM - a novel multimodal large language model specifically tailored to generate high-quality descriptions of patent figures. Our proposed PatentLMM comprises two key components: (i) PatentMME, a specialized multimodal vision encoder that captures the unique structural elements of patent figures, and (ii) PatentLLaMA, a domain-adapted version of LLaMA fine-tuned on a large collection of patents. Extensive experiments demonstrate that training a vision encoder specifically designed for patent figures significantly boosts the performance, generating coherent descriptions compared to fine-tuning similar-sized off-the-shelf multimodal models. PatentDesc-355K and PatentLMM pave the way for automating the understanding of patent figures, enabling efficient knowledge sharing and faster drafting of patent documents. We make the code and data publicly available. 

**Abstract (ZH)**: 在专利文件中对技术图纸进行全面准确的描述对于有效的知识共享、专利的复制和知识产权保护至关重要。然而，这一任务的自动化在研究界尚未受到足够的关注。为此，我们引入了包含约35.5万个专利图形及其从超过6万份美国专利文件中提取的简短和详细文本描述的新颖大规模数据集——PatentDesc-355K。此外，我们提出了一种专门针对专利图形生成高质量描述的新型多模态大语言模型——PatentLMM。我们的PatentLMM包含两大关键组成部分：(i) PatentMME，一种专门针对专利图形的多模态视觉编码器，能够捕捉专利图形的独特结构元素；(ii) PatentLLaMA，一个经过大规模专利数据微调的LLaMA领域适配版本。大量实验表明，针对专利图形设计的视觉编码器显著提升了性能，生成的描述更具连贯性，相较于微调大小相近的现成多模态模型。PatentDesc-355K和PatentLMM为自动化理解和处理专利图形铺平了道路，促进了高效的知识共享并加快了专利文件的起草。我们公开了代码和数据。 

---
# An Attempt to Unraveling Token Prediction Refinement and Identifying Essential Layers of Large Language Models 

**Title (ZH)**: 尝试解析.token预测精炼并识别大规模语言模型中的核心层 

**Authors**: Jaturong Kongmanee  

**Link**: [PDF](https://arxiv.org/pdf/2501.15054)  

**Abstract**: This research aims to unravel how large language models (LLMs) iteratively refine token predictions (or, in a general sense, vector predictions). We utilized a logit lens technique to analyze the model's token predictions derived from intermediate representations. Specifically, we focused on how LLMs access and use information from input contexts, and how positioning of relevant information affects the model's token prediction refinement process. Our findings for multi-document question answering task, by varying input context lengths (the number of documents), using GPT-2, revealed that the number of layers between the first layer that the model predicted next tokens correctly and the later layers that the model finalized its correct predictions, as a function of the position of relevant information (i.e., placing the relevant one at the beginning, middle, or end of the input context), has a nearly inverted U shape. We found that the gap between these two layers, on average, diminishes when relevant information is positioned at the beginning or end of the input context, suggesting that the model requires more refinements when processing longer contexts with relevant information situated in the middle, and highlighting which layers are essential for determining the correct output. Our analysis provides insights about how token predictions are distributed across different conditions, and establishes important connections to existing hypotheses and previous findings in AI safety research and development. 

**Abstract (ZH)**: 本研究旨在探究大型语言模型（LLMs）如何迭代优化词元预测（或更广泛地说，向量预测）。我们使用了逻辑斯蒂视窗技术来分析模型从中间表示中生成的词元预测。具体而言，我们重点关注LLMs如何访问并利用输入上下文中的信息，以及相关信息的位置如何影响模型词元预测的优化过程。通过对多文档问答任务的研究，我们使用GPT-2模型，随着输入上下文长度（文档数量）的变化，发现从模型首次准确预测下一个词元的层到模型最终确定正确预测的层之间的层数，作为相关信息位置的函数（即放在输入上下文的起始、中间或结尾），呈现出几乎倒U形的变化趋势。我们发现，当相关信息位于输入上下文的起始或结尾时，这两个层之间的差异通常会减小，表明当处理包含相关信息位于中间的较长上下文时，模型需要更多的优化，并突出哪些层对于确定正确输出至关重要。我们的分析提供了关于在不同条件下词元预测分布的见解，并与现有假设以及人工智能安全研究和开发中的先前发现建立了重要的联系。 

---
# LLM4DistReconfig: A Fine-tuned Large Language Model for Power Distribution Network Reconfiguration 

**Title (ZH)**: LLM4DistReconfig：一个用于电力配电网重构的微调大型语言模型 

**Authors**: Panayiotis Christou, Md. Zahidul Islam, Yuzhang Lin, Jingwei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.14960)  

**Abstract**: Power distribution networks are evolving due to the integration of DERs and increased customer participation. To maintain optimal operation, minimize losses, and meet varying load demands, frequent network reconfiguration is necessary. Traditionally, the reconfiguration task relies on optimization software and expert operators, but as systems grow more complex, faster and more adaptive solutions are required without expert intervention. Data-driven reconfiguration is gaining traction for its accuracy, speed, and robustness against incomplete network data. LLMs, with their ability to capture complex patterns, offer a promising approach for efficient and responsive network reconfiguration in evolving complex power networks.
In this work, we introduce LLM4DistReconfig, a deep learning-based approach utilizing a fine-tuned LLM to solve the distribution network reconfiguration problem. By carefully crafting prompts and designing a custom loss function, we train the LLM with inputs representing network parameters such as buses, available lines, open lines, node voltages, and system loss. The model then predicts optimal reconfigurations by outputting updated network configurations that minimize system loss while meeting operational constraints. Our approach significantly reduces inference time compared to classical algorithms, allowing for near real-time optimal reconfiguration after training. Experimental results show that our method generates optimal configurations minimizing system loss for five individual and a combined test dataset. It also produces minimal invalid edges, no cycles, or subgraphs across all datasets, fulfilling domain-specific needs. Additionally, the generated responses contain less than 5% improper outputs on seen networks and satisfactory results on unseen networks, demonstrating its effectiveness and reliability for the reconfiguration task. 

**Abstract (ZH)**: 分布式网络由于分布式能源（DERs）的整合和客户参与度的增加而不断发展。为了保持最优运行状态、减少损耗并满足不断变化的负荷需求，频繁的网络重构是必要的。传统上，重构任务依赖于优化软件和专家操作员，但随着系统复杂性的增加，需要更加快速和适应性强的解决方案，无需专家干预。基于数据驱动的重构因其准确性、速度以及对不完整网络数据的稳健性而受到青睐。大规模语言模型（LLMs），由于其捕捉复杂模式的能力，为复杂电力网络中高效和及时的网络重构提供了一种有前途的方法。

在本工作中，我们提出了LLM4DistReconfig，一种基于深度学习的方法，利用微调后的LLM来解决分布网络重构问题。通过精心设计提示并设计自定义损失函数，我们训练LLM以网络参数如母线、可用线路、开放线路、节点电压和系统损耗等作为输入。该模型通过输出更新的网络配置来预测最优的重构方案，这些配置在满足操作约束的同时最小化系统损耗。与传统算法相比，我们的方法显著减少了推理时间，使得经过训练后能够实现接近实时的最优重构。实验结果表明，我们的方法能够针对单个测试集和组合测试集生成最优配置，同时最小化系统损耗。此外，所有数据集生成的重构中没有出现无效边缘、自循环或子图，满足了特定领域的需要。此外，生成的响应在已见过的网络中不含有超过5%的不当输出，在未见过的网络中也取得了令人满意的结果，这证明了其在重构任务中的有效性和可靠性。 

---
# ExPerT: Effective and Explainable Evaluation of Personalized Long-Form Text Generation 

**Title (ZH)**: ExPerT: 有效的可解释个性化长文本生成评估 

**Authors**: Alireza Salemi, Julian Killingback, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2501.14956)  

**Abstract**: Evaluating personalized text generated by large language models (LLMs) is challenging, as only the LLM user, i.e., prompt author, can reliably assess the output, but re-engaging the same individuals across studies is infeasible. This paper addresses the challenge of evaluating personalized text generation by introducing ExPerT, an explainable reference-based evaluation framework. ExPerT leverages an LLM to extract atomic aspects and their evidence from the generated and reference texts, match the aspects, and evaluate their alignment based on content and writing style -- two key attributes in personalized text generation. Additionally, ExPerT generates detailed, fine-grained explanations for every step of the evaluation process, enhancing transparency and interpretability. Our experiments demonstrate that ExPerT achieves a 7.2% relative improvement in alignment with human judgments compared to the state-of-the-art text generation evaluation methods. Furthermore, human evaluators rated the usability of ExPerT's explanations at 4.7 out of 5, highlighting its effectiveness in making evaluation decisions more interpretable. 

**Abstract (ZH)**: 评估由大规模语言模型（LLMs）生成的个性化文本具有挑战性，因为只有LLM的用户，即提示作者，才能可靠地评估输出，但在不同研究中重新 Engagement 同一个体是不可行的。本文通过引入ExPerT，一种基于解释的参考框架评估方法，来应对个性化文本生成的评估挑战。ExPerT 利用LLM从生成文本和参考文本中提取基本方面及其证据，匹配这些方面，并基于内容和写作风格评估它们的一致性——这是个性化文本生成的两个关键属性。此外，ExPerT 为评估过程中的每个步骤生成详细的细粒度解释，从而提高透明度和可解释性。我们的实验表明，与当前最先进的文本生成评估方法相比，ExPerT 在一致性方面表现出 7.2% 的相对改进。此外，人类评估者对ExPerT解释的使用评价为4.7分（满分5分），这突显了ExPerT在使评估决策更易于解释方面的有效性。 

---
# CASE-Bench: Context-Aware Safety Evaluation Benchmark for Large Language Models 

**Title (ZH)**: CASE-Bench：面向大型语言模型的上下文感知安全性评估基准 

**Authors**: Guangzhi Sun, Xiao Zhan, Shutong Feng, Philip C. Woodland, Jose Such  

**Link**: [PDF](https://arxiv.org/pdf/2501.14940)  

**Abstract**: Aligning large language models (LLMs) with human values is essential for their safe deployment and widespread adoption. Current LLM safety benchmarks often focus solely on the refusal of individual problematic queries, which overlooks the importance of the context where the query occurs and may cause undesired refusal of queries under safe contexts that diminish user experience. Addressing this gap, we introduce CASE-Bench, a Context-Aware Safety Evaluation Benchmark that integrates context into safety assessments of LLMs. CASE-Bench assigns distinct, formally described contexts to categorized queries based on Contextual Integrity theory. Additionally, in contrast to previous studies which mainly rely on majority voting from just a few annotators, we recruited a sufficient number of annotators necessary to ensure the detection of statistically significant differences among the experimental conditions based on power analysis. Our extensive analysis using CASE-Bench on various open-source and commercial LLMs reveals a substantial and significant influence of context on human judgments (p<0.0001 from a z-test), underscoring the necessity of context in safety evaluations. We also identify notable mismatches between human judgments and LLM responses, particularly in commercial models within safe contexts. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与人类价值相一致是确保其安全部署和广泛应用的关键。当前的LLM安全评估基准往往仅关注对个别问题的拒绝，而忽视了查询发生的上下文的重要性，可能导致在安全上下文中不必要地拒绝查询，从而损害用户体验。为解决这一问题，我们引入了CASE-Bench，这是一种基于上下文的安全评估基准，将上下文整合进对LLMs的安全评估中。CASE-Bench 根据情境完备性理论为分类后的查询分配不同的、形式化描述的上下文。此外，与以往主要依赖少数标注者投票的研究不同，我们通过功效分析招募了足够数量的标注者，以确保在实验条件下检测到统计显著差异。通过使用CASE-Bench 对各种开源和商用LLMs进行广泛分析，我们发现上下文对人类判断产生了显著影响（p<0.0001来自Z检验），强调了在安全评估中考虑上下文的重要性。我们还发现，在安全上下文中，人类判断与LLM响应之间存在明显的不一致性，尤其是在商用模型中更为明显。 

---
# Explaining Categorical Feature Interactions Using Graph Covariance and LLMs 

**Title (ZH)**: 使用图形协方差和大规模语言模型解释分类特征交互 

**Authors**: Cencheng Shen, Darren Edge, Jonathan Larson, Carey E. Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2501.14932)  

**Abstract**: Modern datasets often consist of numerous samples with abundant features and associated timestamps. Analyzing such datasets to uncover underlying events typically requires complex statistical methods and substantial domain expertise. A notable example, and the primary data focus of this paper, is the global synthetic dataset from the Counter Trafficking Data Collaborative (CTDC) -- a global hub of human trafficking data containing over 200,000 anonymized records spanning from 2002 to 2022, with numerous categorical features for each record. In this paper, we propose a fast and scalable method for analyzing and extracting significant categorical feature interactions, and querying large language models (LLMs) to generate data-driven insights that explain these interactions. Our approach begins with a binarization step for categorical features using one-hot encoding, followed by the computation of graph covariance at each time. This graph covariance quantifies temporal changes in dependence structures within categorical data and is established as a consistent dependence measure under the Bernoulli distribution. We use this measure to identify significant feature pairs, such as those with the most frequent trends over time or those exhibiting sudden spikes in dependence at specific moments. These extracted feature pairs, along with their timestamps, are subsequently passed to an LLM tasked with generating potential explanations of the underlying events driving these dependence changes. The effectiveness of our method is demonstrated through extensive simulations, and its application to the CTDC dataset reveals meaningful feature pairs and potential data stories underlying the observed feature interactions. 

**Abstract (ZH)**: 现代数据集通常包含大量样本和丰富的特征以及相关的时戳。分析这些数据集以发现潜在事件通常需要复杂统计方法以及大量领域的专业知识。一个典型例子，也是本文的主要数据关注点，是来自反人口贩卖数据协作组织（CTDC）的全球合成数据集——这是一个全球的人口贩卖数据枢纽，包含了从2002年到2022年超过20万条匿名记录，每条记录还包含了多个分类特征。本文提出了一种快速和可扩展的方法，用于分析和提取重要的分类特征相互作用，并利用大型语言模型（LLMs）生成数据驱动的见解，以解释这些相互作用。我们的方法首先使用一位编码对分类特征进行二值化处理，然后在每个时间点上计算图协方差。这种图协方差量化了分类数据中依赖结构的时态变化，并在伯努利分布下被证明是一致的依赖性度量。我们使用这一度量来识别显著的特征对，例如随着时间最频繁的趋势对或在特定时刻突然表现出依赖性突增的趋势对。这些提取的特征对及其时间戳随后被传递给负责生成这些依赖性变化背后事件潜在解释的LLM。我们通过广泛的模拟验证了该方法的有效性，并将其应用于CTDC数据集，揭示了有意义的特征对和潜在的数据故事，这些故事解释了观察到的特征相互作用背后的原因。 

---
# JustLogic: A Comprehensive Benchmark for Evaluating Deductive Reasoning in Large Language Models 

**Title (ZH)**: JustLogic：评估大型语言模型演绎推理能力的综合性基准测试 

**Authors**: Michael K. Chen, Xikun Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14851)  

**Abstract**: Logical reasoning is a critical component of Large Language Models (LLMs), and substantial research efforts in recent years have aimed to enhance their deductive reasoning capabilities. However, existing deductive reasoning benchmarks, which are crucial for evaluating and advancing LLMs, are inadequate due to their lack of task complexity, presence of prior knowledge as a confounder, and superficial error analysis. To address these deficiencies, we introduce JustLogic, a synthetically generated deductive reasoning benchmark designed for rigorous evaluation of LLMs. JustLogic is (i) highly complex, capable of generating a diverse range of linguistic patterns, vocabulary, and argument structures; (ii) prior knowledge independent, eliminating the advantage of models possessing prior knowledge and ensuring that only deductive reasoning is used to answer questions; and (iii) capable of in-depth error analysis on the heterogeneous effects of reasoning depth and argument form on model accuracy. Our experimental results on JustLogic reveal that most state-of-the-art (SOTA) LLMs perform significantly worse than the human average, demonstrating substantial room for model improvement. All code and data are available at this https URL 

**Abstract (ZH)**: 逻辑推理是大型语言模型（LLMs）的一个关键组成部分，近年来，大量研究致力于提升其演绎推理能力。然而，现有的演绎推理基准在评估和促进LLMs方面存在不足，因为这些基准的任务复杂性较低、包含了先验知识的干扰，并且浅显的错误分析。为解决这些问题，我们提出JustLogic，这是一种专门为严格评估LLMs设计的合成演绎推理基准。JustLogic具有以下特点：（i）高度复杂，能够生成多种多样的语言模式、词汇和论证结构；（ii）不依赖于先验知识，去除模型依赖先验知识的优势，确保仅使用演绎推理来回答问题；（iii）能够进行深入的错误分析，探讨推理深度和论证形式对模型准确率的异质性影响。我们在JustLogic上的实验结果显示，大多数最新（SOTA）的LLMs的表现明显低于人类平均水平，表明模型仍有很大的改进空间。所有代码和数据可在以下链接获取：https://... 

---
# DeServe: Towards Affordable Offline LLM Inference via Decentralization 

**Title (ZH)**: DeServe：通过去中心化实现可负担的离线大规模语言模型推理 

**Authors**: Linyu Wu, Xiaoyuan Liu, Tianneng Shi, Zhe Ye, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2501.14784)  

**Abstract**: The rapid growth of generative AI and its integration into everyday workflows have significantly increased the demand for large language model (LLM) inference services. While proprietary models remain popular, recent advancements in open-source LLMs have positioned them as strong contenders. However, deploying these models is often constrained by the high costs and limited availability of GPU resources. In response, this paper presents the design of a decentralized offline serving system for LLM inference. Utilizing idle GPU resources, our proposed system, DeServe, decentralizes access to LLMs at a lower cost. DeServe specifically addresses key challenges in optimizing serving throughput in high-latency network environments. Experiments demonstrate that DeServe achieves a 6.7x-12.6x improvement in throughput over existing serving system baselines in such conditions. 

**Abstract (ZH)**: 生成型AI的快速发展及其融入日常工作流中，显著增加了对大规模语言模型（LLM）推理服务的需求。虽然专有模型仍很流行，但最近开源LLM的进步使它们成为了强有力的竞争对手。然而，部署这些模型往往受限于GPU资源的高成本和有限可用性。针对这一问题，本文提出了一种去中心化的离线服务系统设计，用于LLM推理。利用闲置的GPU资源，我们提出的系统DeServe以更低的成本实现了对LLM的去中心化访问。DeServe特别解决了在高延迟网络环境中优化服务吞吐量的关键挑战。实验表明，在这些条件下，DeServe在吞吐量方面相对于现有的服务系统基线实现了6.7至12.6倍的提升。 

---
# DropMicroFluidAgents (DMFAs): Autonomous Droplet Microfluidic Research Framework Through Large Language Model Agents 

**Title (ZH)**: DropMicroFluidAgents (DMFAs): 通过大型语言模型代理实现的自主液滴微流控研究框架 

**Authors**: Dinh-Nguyen Nguyen, Raymond Kai-Yu Tong, Ngoc-Duy Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2501.14772)  

**Abstract**: Applying Large language models (LLMs) within specific domains requires substantial adaptation to account for the unique terminologies, nuances, and context-specific challenges inherent to those areas. Here, we introduce DropMicroFluidAgents (DMFAs), an advanced language-driven framework leveraging state-of-the-art pre-trained LLMs. DMFAs employs LLM agents to perform two key functions: (1) delivering focused guidance, answers, and suggestions specific to droplet microfluidics and (2) generating machine learning models to optimise and automate the design of droplet microfluidic devices, including the creation of code-based computer-aided design (CAD) scripts to enable rapid and precise design execution. Experimental evaluations demonstrated that the integration of DMFAs with the LLAMA3.1 model yielded the highest accuracy of 76.15%, underscoring the significant performance enhancement provided by agent integration. This effect was particularly pronounced when DMFAs were paired with the GEMMA2 model, resulting in a 34.47% improvement in accuracy compared to the standalone GEMMA2 configuration. This study demonstrates the effective use of LLM agents in droplet microfluidics research as powerful tools for automating workflows, synthesising knowledge, optimising designs, and interacting with external systems. These capabilities enable their application across education and industrial support, driving greater efficiency in scientific discovery and innovation. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，符合学术规范：

在特定领域应用大型语言模型（LLMs）需要进行大量的适应，以应对这些领域内独特的术语、细微差别和情境特定的挑战。本文介绍了DropMicroFluidAgents（DMFAs），这是一种利用最新预训练LLMs的先进语言驱动框架。DMFAs 通过LLMs代理执行两个关键功能：（1）提供针对微液滴微流控领域的集中指导、回答和建议；（2）生成机器学习模型以优化和自动化微液滴微流控装置的设计，包括生成基于代码的计算机辅助设计（CAD）脚本，以实现快速和精确的设计执行。实验评估表明，将DMFAs 与LLAMA3.1模型结合使用，能够获得最高的准确率76.15%，突显了代理集成提供的显著性能提升。当DMFAs 与GEMMA2模型配对时，其准确率提高了34.47%，超过了仅使用GEMMA2配置的情况。研究证明了在微液滴微流控研究中有效使用LLMs代理作为自动化工序的强大工具，能够整合知识、优化设计，并与外部系统交互。这些能力使得它们能够在教育和工业支持中得到应用，促进科学研究和创新效率的提升。 

---
# EvalSVA: Multi-Agent Evaluators for Next-Gen Software Vulnerability Assessment 

**Title (ZH)**: EvalSVA：面向下一代软件漏洞评估的多agent评估器 

**Authors**: Xin-Cheng Wen, Jiaxin Ye, Cuiyun Gao, Lianwei Wu, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14737)  

**Abstract**: Software Vulnerability (SV) assessment is a crucial process of determining different aspects of SVs (e.g., attack vectors and scope) for developers to effectively prioritize efforts in vulnerability mitigation. It presents a challenging and laborious process due to the complexity of SVs and the scarcity of labeled data. To mitigate the above challenges, we introduce EvalSVA, a multi-agent evaluators team to autonomously deliberate and evaluate various aspects of SV assessment. Specifically, we propose a multi-agent-based framework to simulate vulnerability assessment strategies in real-world scenarios, which employs multiple Large Language Models (LLMs) into an integrated group to enhance the effectiveness of SV assessment in the limited data. We also design diverse communication strategies to autonomously discuss and assess different aspects of SV. Furthermore, we construct a multi-lingual SV assessment dataset based on the new standard of CVSS, comprising 699, 888, and 1,310 vulnerability-related commits in C++, Python, and Java, respectively. Our experimental results demonstrate that EvalSVA averagely outperforms the 44.12\% accuracy and 43.29\% F1 for SV assessment compared with the previous methods. It shows that EvalSVA offers a human-like process and generates both reason and answer for SV assessment. EvalSVA can also aid human experts in SV assessment, which provides more explanation and details for SV assessment. 

**Abstract (ZH)**: 软件漏洞（SV）评估是确定不同方面漏洞（例如攻击向量和影响范围）的重要过程，旨在帮助开发者有效优先考虑漏洞缓解工作。这一过程由于漏洞的复杂性和标注数据的稀缺性而变得具有挑战性和繁琐。为了应对上述挑战，我们引入了EvalSVA，这是一个自主讨论和评估漏洞评估各种方面的一组多智能体评价者团队。具体而言，我们提出了一种基于多智能体的框架，在实际场景中模拟漏洞评估策略，该框架通过将多个大型语言模型（LLMs）整合到一个小组中，增强了在数据有限情况下的漏洞评估效果。我们还设计了多种通信策略，以自主讨论和评估不同方面漏洞。此外，我们根据新的CVSS标准构建了一个多语言漏洞评估数据集，其中包含C++、Python和Java语言中分别共计699,888个和1,310个漏洞相关提交记录。实验结果显示，EvalSVA相比之前的评估方法，在漏洞评估准确性和F1分数上平均高出44.12%和43.29%。这表明EvalSVA提供了类似人类的过程，并为漏洞评估生成了合理性和答案。同时，EvalSVA也可以帮助人类专家进行漏洞评估，提供更详细的解释和信息。 

---
# From Critique to Clarity: A Pathway to Faithful and Personalized Code Explanations with Large Language Models 

**Title (ZH)**: 从批判到清晰：一条通往忠实且个性化代码解释的道路——大型语言模型的应用 

**Authors**: Zexing Xu, Zhuang Luo, Yichuan Li, Kyumin Lee, S. Rasoul Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2501.14731)  

**Abstract**: In the realm of software development, providing accurate and personalized code explanations is crucial for both technical professionals and business stakeholders. Technical professionals benefit from enhanced understanding and improved problem-solving skills, while business stakeholders gain insights into project alignments and transparency. Despite the potential, generating such explanations is often time-consuming and challenging. This paper presents an innovative approach that leverages the advanced capabilities of large language models (LLMs) to generate faithful and personalized code explanations. Our methodology integrates prompt enhancement, self-correction mechanisms, personalized content customization, and interaction with external tools, facilitated by collaboration among multiple LLM agents. We evaluate our approach using both automatic and human assessments, demonstrating that our method not only produces accurate explanations but also tailors them to individual user preferences. Our findings suggest that this approach significantly improves the quality and relevance of code explanations, offering a valuable tool for developers and stakeholders alike. 

**Abstract (ZH)**: 在软件开发领域，提供准确且个性化的代码解释对于技术人员和商业利益相关者都至关重要。技术人员可以从增强的理解力和改善的问题解决能力中受益，而商业利益相关者则可以借此获得项目对齐和透明度的洞见。尽管具有这种潜力，生成这样的解释往往是耗时且具有挑战性的。本文提出了一种创新的方法，利用大型语言模型（LLMs）的高级功能来生成忠实且个性化的代码解释。我们的方法结合了提示增强、自我校正机制、个性化内容定制以及与外部工具的交互，通过多个LLM代理的合作来实现。我们通过自动评估和人工评估两种方式对我们的方法进行了评估，结果显示，我们的方法不仅能够生成准确的解释，还能针对个体用户的偏好进行定制。我们的研究结果表明，这种方法显著提高了代码解释的质量和相关性，为开发人员和利益相关者提供了有价值的工具。 

---
# SampleLLM: Optimizing Tabular Data Synthesis in Recommendations 

**Title (ZH)**: SampleLLM：优化推荐系统中表格数据合成 

**Authors**: Jingtong Gao, Zhaocheng Du, Xiaopeng Li, Xiangyu Zhao, Yichao Wang, Xiangyang Li, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.16125)  

**Abstract**: Tabular data synthesis is crucial in machine learning, yet existing general methods-primarily based on statistical or deep learning models-are highly data-dependent and often fall short in recommender systems. This limitation arises from their difficulty in capturing complex distributions and understanding feature relationships from sparse and limited data, along with their inability to grasp semantic feature relations. Recently, Large Language Models (LLMs) have shown potential in generating synthetic data samples through few-shot learning and semantic understanding. However, they often suffer from inconsistent distribution and lack of diversity due to their inherent distribution disparity with the target dataset. To address these challenges and enhance tabular data synthesis for recommendation tasks, we propose a novel two-stage framework named SampleLLM to improve the quality of LLM-based tabular data synthesis for recommendations by ensuring better distribution alignment. In the first stage, SampleLLM employs LLMs with Chain-of-Thought prompts and diverse exemplars to generate data that closely aligns with the target dataset distribution, even when input samples are limited. The second stage uses an advanced feature attribution-based importance sampling method to refine feature relationships within the synthesized data, reducing any distribution biases introduced by the LLM. Experimental results on three recommendation datasets, two general datasets, and online deployment illustrate that SampleLLM significantly surpasses existing methods for recommendation tasks and holds promise for a broader range of tabular data scenarios. 

**Abstract (ZH)**: 表格数据合成在机器学习中至关重要，然而现有的通用方法——主要是基于统计或深度学习模型的方法——高度依赖于特定数据集，并且在推荐系统中往往效果不佳。这一局限性源于它们在捕捉复杂分布和理解稀疏有限数据中的特征关系方面存在困难，以及无法把握语义特征关系。最近，大型语言模型（LLMs）在通过少样本学习和语义理解生成合成数据样本方面显示出潜力。然而，由于与目标数据集固有的分布差异，它们往往会导致分布不一致和多样性不足。为了应对这些挑战并提高推荐任务中的表格数据合成质量，我们提出了一种名为SampleLLM的新两阶段框架，通过确保更好的分布对齐来增强基于LLM的表格数据合成质量。在第一阶段，SampleLLM利用带有链式思维提示和多样示例的LLMs生成与目标数据集分布紧密对齐的数据，即使输入样本有限。第二阶段使用高级特征归因为基础的重要性采样方法进一步细化合成数据中的特征关系，减少LLM引入的任何分布偏差。在三个推荐数据集、两个通用数据集以及在线部署实验中，SampleLLM的表现显著优于现有方法，并且为更广泛的表格数据场景提供了可能性。 

---
# Understanding Long Videos via LLM-Powered Entity Relation Graphs 

**Title (ZH)**: 通过基于LLM的实体关系图理解长视频 

**Authors**: Meng Chu, Yicong Li, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2501.15953)  

**Abstract**: The analysis of extended video content poses unique challenges in artificial intelligence, particularly when dealing with the complexity of tracking and understanding visual elements across time. Current methodologies that process video frames sequentially struggle to maintain coherent tracking of objects, especially when these objects temporarily vanish and later reappear in the footage. A critical limitation of these approaches is their inability to effectively identify crucial moments in the video, largely due to their limited grasp of temporal relationships. To overcome these obstacles, we present GraphVideoAgent, a cutting-edge system that leverages the power of graph-based object tracking in conjunction with large language model capabilities. At its core, our framework employs a dynamic graph structure that maps and monitors the evolving relationships between visual entities throughout the video sequence. This innovative approach enables more nuanced understanding of how objects interact and transform over time, facilitating improved frame selection through comprehensive contextual awareness. Our approach demonstrates remarkable effectiveness when tested against industry benchmarks. In evaluations on the EgoSchema dataset, GraphVideoAgent achieved a 2.2 improvement over existing methods while requiring analysis of only 8.2 frames on average. Similarly, testing on the NExT-QA benchmark yielded a 2.0 performance increase with an average frame requirement of 8.1. These results underscore the efficiency of our graph-guided methodology in enhancing both accuracy and computational performance in long-form video understanding tasks. 

**Abstract (ZH)**: 将上述论文内容或标题翻译成中文，同时符合学术规范如下：

分析扩展视频内容在人工智能领域中提出了独特的挑战，特别是在处理时间上视觉元素的复杂追踪和理解方面。当前逐帧处理视频的方法在保持对象连贯追踪方面存在困难，尤其是当这些对象暂时消失并在后续重新出现在视频中时。这些方法的主要局限性在于它们在识别视频中的关键时刻能力有限，主要是因为它们对时间关系的把握有限。为克服这些挑战，我们提出了一种名为GraphVideoAgent的先进系统，该系统结合了基于图的对象追踪能力和大型语言模型的能力。该框架的核心在于利用动态图结构，该结构在整个视频序列中映射和监控视觉实体之间的动态关系。这一创新方法能够更深入地理解对象如何随时间相互作用和变化，从而通过全面的上下文感知来优化帧的选择。在行业基准测试中，我们的方法表现出显著的效果。根据EgoSchema数据集的评估，GraphVideoAgent在效果上比现有方法提高了2.2%，并且平均只需要分析8.2帧。在NExT-QA基准测试中，GraphVideoAgent同样实现了2.0%的性能提升，平均帧需求仅为8.1帧。这些结果突显了我们在长视频理解任务中通过基于图的方法提高准确性和计算性能方面的高效性。 

---
# Technology Mapping with Large Language Models 

**Title (ZH)**: 使用大型语言模型进行技术映射 

**Authors**: Minh Hieu Nguyen, Hien Thu Pham, Hiep Minh Ha, Ngoc Quang Hung Le, Jun Jo  

**Link**: [PDF](https://arxiv.org/pdf/2501.15120)  

**Abstract**: In today's fast-evolving business landscape, having insight into the technology stacks that organizations use is crucial for forging partnerships, uncovering market openings, and informing strategic choices. However, conventional technology mapping, which typically hinges on keyword searches, struggles with the sheer scale and variety of data available, often failing to capture nascent technologies. To overcome these hurdles, we present STARS (Semantic Technology and Retrieval System), a novel framework that harnesses Large Language Models (LLMs) and Sentence-BERT to pinpoint relevant technologies within unstructured content, build comprehensive company profiles, and rank each firm's technologies according to their operational importance. By integrating entity extraction with Chain-of-Thought prompting and employing semantic ranking, STARS provides a precise method for mapping corporate technology portfolios. Experimental results show that STARS markedly boosts retrieval accuracy, offering a versatile and high-performance solution for cross-industry technology mapping. 

**Abstract (ZH)**: 在当今快速演变的商业环境中，了解组织所使用的技术栈对于建立合作伙伴关系、发现市场机遇和指导战略决策至关重要。然而，传统的技术映射方法通常依赖关键词搜索，难以处理大量和多样化的数据，常常无法捕捉新兴技术。为克服这些挑战，我们提出了STARS（语义技术和检索系统）这一新型框架，该框架利用大型语言模型（LLMs）和Sentence-BERT来识别非结构化内容中的相关技术，构建全面的公司概况，并根据运营重要性对每家公司的技术进行排名。通过集成实体提取与链式推理提示，并采用语义排名，STARS提供了一种精确的技术组合映射方法。实验结果显示，STARS显著提高了检索精度，为跨行业的技术映射提供了灵活且高性能的解决方案。 

---
# PatchRec: Multi-Grained Patching for Efficient LLM-based Sequential Recommendation 

**Title (ZH)**: PatchRec：高效的基于大规模语言模型的序列推荐多粒度补丁方法 

**Authors**: Jiayi Liao, Ruobing Xie, Sihang Li, Xiang Wang, Xingwu Sun, Zhanhui Kang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2501.15087)  

**Abstract**: Large Language Models for sequential recommendation (LLM4SR), which transform user-item interactions into language modeling, have shown promising results. However, due to the limitations of context window size and the computational costs associated with Large Language Models (LLMs), current approaches primarily truncate user history by only considering the textual information of items from the most recent interactions in the input prompt. This truncation fails to fully capture the long-term behavioral patterns of users. To address this, we propose a multi-grained patching framework -- PatchRec. It compresses the textual tokens of an item title into a compact item patch, and further compresses multiple item patches into a denser session patch, with earlier interactions being compressed to a greater degree. The framework consists of two stages: (1) Patch Pre-training, which familiarizes LLMs with item-level compression patterns, and (2) Patch Fine-tuning, which teaches LLMs to model sequences at multiple granularities. Through this simple yet effective approach, empirical results demonstrate that PatchRec outperforms existing methods, achieving significant performance gains with fewer tokens fed to the LLM. Specifically, PatchRec shows up to a 32% improvement in HR@20 on the Goodreads dataset over uncompressed baseline, while using only 7% of the tokens. This multi-grained sequence modeling paradigm, with an adjustable compression ratio, enables LLMs to be efficiently deployed in real-world recommendation systems that handle extremely long user behavior sequences. 

**Abstract (ZH)**: 大型语言模型在序列推荐中的应用（LLM4SR），这些模型将用户-项交互转化为语言模型，显示出了令人鼓舞的结果。然而，由于上下文窗口大小的限制以及大型语言模型（LLMs）相关的计算成本，当前的方法主要通过仅在输入提示中考虑最近交互的项的文本信息来进行用户历史截断。这种截断方法未能充分捕捉用户的长期行为模式。为了解决这个问题，我们提出了一种多粒度补丁框架——PatchRec。该框架将项标题的文本标记压缩为紧凑的项补丁，并进一步将多个项补丁压缩为更密集的会话补丁，更早的交互被压缩得更多。该框架包括两个阶段：（1）补丁预训练，使LLMs熟悉项级别压缩模式，以及（2）补丁微调，使LLMs能够以多种粒度建模序列。通过这种简单而有效的方法，实验结果表明，PatchRec优于现有方法，在Goodreads数据集上，使用仅7%的令牌，取得了显著性能提升，特别是在HR@20方面，与未压缩的基线相比，提高了32%。这种可调节压缩比的多粒度序列建模范式，使LLMs能够高效地部署在处理极其长的用户行为序列的实际推荐系统中。 

---
# CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs 

**Title (ZH)**: CG-RAG：引用图检索增强的LLM研究问题回答 

**Authors**: Yuntong Hu, Zhihan Lei, Zhongjie Dai, Allen Zhang, Abhinav Angirekula, Zheng Zhang, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15067)  

**Abstract**: Research question answering requires accurate retrieval and contextual understanding of scientific literature. However, current Retrieval-Augmented Generation (RAG) methods often struggle to balance complex document relationships with precise information retrieval. In this paper, we introduce Contextualized Graph Retrieval-Augmented Generation (CG-RAG), a novel framework that integrates sparse and dense retrieval signals within graph structures to enhance retrieval efficiency and subsequently improve generation quality for research question answering. First, we propose a contextual graph representation for citation graphs, effectively capturing both explicit and implicit connections within and across documents. Next, we introduce Lexical-Semantic Graph Retrieval (LeSeGR), which seamlessly integrates sparse and dense retrieval signals with graph encoding. It bridges the gap between lexical precision and semantic understanding in citation graph retrieval, demonstrating generalizability to existing graph retrieval and hybrid retrieval methods. Finally, we present a context-aware generation strategy that utilizes the retrieved graph-structured information to generate precise and contextually enriched responses using large language models (LLMs). Extensive experiments on research question answering benchmarks across multiple domains demonstrate that our CG-RAG framework significantly outperforms RAG methods combined with various state-of-the-art retrieval approaches, delivering superior retrieval accuracy and generation quality. 

**Abstract (ZH)**: 以下是根据学术规范翻译的内容：

研究问题回答需要精确检索和理解科学文献的上下文。然而，当前的检索增强生成（RAG）方法往往难以平衡复杂文档关系与精确信息检索之间的关系。本文介绍了一种名为情境化图检索增强生成（CG-RAG）的新框架，该框架通过在图结构中整合稀疏和密集的检索信号来提高检索效率，从而改善研究问题回答的生成质量。首先，我们提出了一种文献引用图的情境化图表示方法，有效地捕获文档内部及跨文档的显式和隐式联系。其次，我们引入了基于词性和语义图检索（LeSeGR）的方法，该方法无缝地将稀疏和密集的检索信号与图编码相结合。LeSeGR在文献引用图检索中填补了词性和语义理解之间的差距，展示了对该领域现有图检索和混合检索方法的通用性。最后，我们提出了一个上下文感知的生成策略，该策略利用检索到的图结构信息和大规模语言模型（LLM）生成精确且上下文丰富的响应。在多个领域的研究问题回答基准测试中的广泛实验表明，我们的CG-RAG框架显著优于结合了各种先进检索方法的RAG方法，提供了更高的检索准确性和生成质量。 

---
# Parametric Retrieval Augmented Generation 

**Title (ZH)**: 参数化检索增强生成 

**Authors**: Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15915)  

**Abstract**: Retrieval-augmented generation (RAG) techniques have emerged as a promising solution to enhance the reliability of large language models (LLMs) by addressing issues like hallucinations, outdated knowledge, and domain adaptation. In particular, existing RAG methods append relevant documents retrieved from external corpus or databases to the input of LLMs to guide their generation process, which we refer to as the in-context knowledge injection method. While this approach is simple and often effective, it has inherent limitations. Firstly, increasing the context length and number of relevant documents can lead to higher computational overhead and degraded performance, especially in complex reasoning tasks. More importantly, in-context knowledge injection operates primarily at the input level, but LLMs store their internal knowledge in their parameters. This gap fundamentally limits the capacity of in-context methods. To this end, we introduce Parametric retrieval-augmented generation (Parametric RAG), a new RAG paradigm that integrates external knowledge directly into the parameters of feed-forward networks (FFN) of an LLM through document parameterization. This approach not only saves online computational costs by eliminating the need to inject multiple documents into the LLMs' input context, but also deepens the integration of external knowledge into the parametric knowledge space of the LLM. Experimental results demonstrate that Parametric RAG substantially enhances both the effectiveness and efficiency of knowledge augmentation in LLMs. Also, it can be combined with in-context RAG methods to achieve even better performance.
We have open-sourced all the code, data, and models in the following anonymized GitHub link: this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）技术已经成为了提升大型语言模型（LLMs）可靠性的有前途的解决方案，尤其是在解决幻觉、过时的知识和领域适应性等问题方面表现出色。具体而言，现有的RAG方法通过将从外部语料库或数据库中检索到的相关文档附加到LLM的输入中，以指导生成过程，这种方法我们称之为上下文内知识注入方法。尽管这种方法简单且通常有效，但它具有内在局限性。首先，增加上下文长度和相关文档的数量会带来更高的计算开销和性能下降，尤其是在复杂的推理任务中。更重要的是，上下文内知识注入主要在输入级别进行，但LLM将其内部知识存储在参数中。这一差距从根本上限制了上下文内方法的能力。为了解决这些问题，我们引入了参数化检索增强生成（Parametric RAG），这是一种新的RAG范式，通过文档参数化将外部知识直接整合到LLM前馈网络（FFN）的参数中。这种方法不仅可以通过消除向LLM输入上下文注入多个文档的需求来节省在线计算成本，还能更深层次地将外部知识整合到LLM的参数化知识空间中。实验结果表明，Parametric RAG显著提升了LLM中知识增强的有效性和效率。此外，它还可以与上下文内RAG方法结合使用，实现更佳的性能。

我们已在去标识化的GitHub链接中开源了所有代码、数据和模型：[这个链接](this https URL) 

---
# LemmaHead: RAG Assisted Proof Generation Using Large Language Models 

**Title (ZH)**: LemmaHead：使用大语言模型的RAG辅助定理证明 

**Authors**: Tianbo Yang, Mingqi Yang, Hongyi Zhao, Tianshuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15797)  

**Abstract**: Developing the logic necessary to solve mathematical problems or write mathematical proofs is one of the more difficult objectives for large language models (LLMS). Currently, the most popular methods in literature consists of fine-tuning the model on written mathematical content such as academic publications and textbooks, so that the model can learn to emulate the style of mathematical writing. In this project, we explore the effectiveness of using retrieval augmented generation (RAG) to address gaps in the mathematical reasoning of LLMs. We develop LemmaHead, a RAG knowledge base that supplements queries to the model with relevant mathematical context, with particular focus on context from published textbooks. To measure our model's performance in mathematical reasoning, our testing paradigm focuses on the task of automated theorem proving via generating proofs to a given mathematical claim in the Lean formal language. 

**Abstract (ZH)**: 开发解决数学问题或撰写数学证明所需的逻辑是大型语言模型（LLMs）面临的更加困难的目标之一。目前，文献中最流行的 方法是通过针对学术出版物和教科书中的数学内容进行微调，使模型能够学习模仿数学写作的风格。在这个项目中，我们探索使用检索增强生成（RAG）来弥补LLMs在数学推理方面的不足。我们开发了LemmaHead，这是一种RAG知识库，通过向模型提供相关数学上下文（特别是来自已出版教科书的上下文）来补充查询。为了衡量模型在数学推理方面的表现，我们测试范式关注的是通过生成给定数学命题在Lean形式语言中的证明来进行自动定理证明的任务。 

---
# RAPID: Retrieval-Augmented Parallel Inference Drafting for Text-Based Video Event Retrieval 

**Title (ZH)**: RAPID：基于检索增强并行推理的文本视频事件检索草稿生成方法 

**Authors**: Long Nguyen, Huy Nguyen, Bao Khuu, Huy Luu, Huy Le, Tuan Nguyen, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16303)  

**Abstract**: Retrieving events from videos using text queries has become increasingly challenging due to the rapid growth of multimedia content. Existing methods for text-based video event retrieval often focus heavily on object-level descriptions, overlooking the crucial role of contextual information. This limitation is especially apparent when queries lack sufficient context, such as missing location details or ambiguous background elements. To address these challenges, we propose a novel system called RAPID (Retrieval-Augmented Parallel Inference Drafting), which leverages advancements in Large Language Models (LLMs) and prompt-based learning to semantically correct and enrich user queries with relevant contextual information. These enriched queries are then processed through parallel retrieval, followed by an evaluation step to select the most relevant results based on their alignment with the original query. Through extensive experiments on our custom-developed dataset, we demonstrate that RAPID significantly outperforms traditional retrieval methods, particularly for contextually incomplete queries. Our system was validated for both speed and accuracy through participation in the Ho Chi Minh City AI Challenge 2024, where it successfully retrieved events from over 300 hours of video. Further evaluation comparing RAPID with the baseline proposed by the competition organizers demonstrated its superior effectiveness, highlighting the strength and robustness of our approach. 

**Abstract (ZH)**: 基于文本查询从视频中检索事件变得越来越具有挑战性，这主要是由于多媒体内容的迅速增长。现有的基于文本的视频事件检索方法往往侧重于对象级别的描述，而忽视了上下文信息的重要作用。特别是在查询缺乏足够的上下文信息时，这一局限尤为明显，例如缺失地理位置详情或背景元素模糊不清。为了解决这些挑战，我们提出了一种名为RAPID（检索增强并行推理起草）的新系统，该系统利用了大型语言模型（LLMs）和提示式学习技术，对用户查询进行语义纠正和丰富，添加相关的上下文信息。这些经过丰富化的查询随后通过并行检索处理，在评估步骤中根据与原始查询的一致性选择最相关的结果。通过在我们自定义开发的数据集上进行大量实验，我们证明了RAPID显著优于传统的检索方法，特别是在上下文不完整查询的情况下。我们的系统通过参加2024胡志明市人工智能挑战赛进行了速度和准确性的验证，在此次挑战赛中成功检索了超过300小时的视频事件。进一步的评估表明，RAPID在与竞赛组织者提出的基线方法进行比较时表现更为优越，突显了我们方法的强度和鲁棒性。 

---
# AdaCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Chain-of-Thought 

**Title (ZH)**: AdaCoT：重新思考适应性思维链在跨语言事实推理中的作用 

**Authors**: Xin Huang, Tarun Kumar Vangani, Zhengyuan Liu, Bowei Zou, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2501.16154)  

**Abstract**: Large language models (LLMs) have shown impressive multilingual capabilities through pretraining on diverse corpora. While these models show strong reasoning abilities, their performance varies significantly across languages due to uneven training data distribution. Existing approaches using machine translation, and extensive multilingual pretraining and cross-lingual tuning face scalability challenges and often fail to capture nuanced reasoning processes across languages. In this paper, we introduce AdaCoT (Adaptive Chain-of-Thought), a framework that enhances multilingual reasoning by dynamically routing thought processes through intermediary "thinking languages" before generating target-language responses. AdaCoT leverages a language-agnostic core and incorporates an adaptive, reward-based mechanism for selecting optimal reasoning pathways without requiring additional pretraining. Our comprehensive evaluation across multiple benchmarks demonstrates substantial improvements in both factual reasoning quality and cross-lingual consistency, with particularly strong performance gains in low-resource language settings. The results suggest that adaptive reasoning paths can effectively bridge the performance gap between high and low-resource languages while maintaining cultural and linguistic nuances. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过在多种语料上进行预训练展示了令人印象深刻的多语言能力。虽然这些模型在推理能力方面表现出色，但由于训练数据分布不均，它们在不同语言上的表现差异显著。现有的使用机器翻译、广泛进行多语言预训练和跨语言微调的方法面临着可扩展性的挑战，并且通常不能捕捉到跨语言的细微推理过程。在本文中，我们提出了AdaCoT（自适应推理链），这是一种通过动态路由思考过程通过中间的“思考语言”来增强多语言推理的框架，从而在生成目标语言回应之前进行推理。AdaCoT 利用了一个语言无关的核心，并引入了一种适应性的、基于奖励的机制，用于选择最优的推理路径，而无需额外的预训练。我们在多个基准测试中的全面评估显示，在事实推理质量和跨语言一致性方面取得了显著的改进，特别是在资源匮乏的语言环境中表现尤为突出。结果表明，适应性的推理路径可以有效地缩小高资源和低资源语言之间的性能差距，同时保留文化与语言的细微差异。 

---
# Integration of LLM Quality Assurance into an NLG System 

**Title (ZH)**: 将LLM质量保障集成到自然语言生成系统中 

**Authors**: Ching-Yi Chen, Johanna Heininger, Adela Schneider, Christian Eckard, Andreas Madsack, Robert Weißgraeber  

**Link**: [PDF](https://arxiv.org/pdf/2501.16078)  

**Abstract**: In this paper, we present a system that uses a Large Language Model (LLM) to perform grammar and spelling correction as a component of Quality Assurance (QA) for texts generated by NLG systems, which is important for text production in real-world scenarios. Evaluating the results of the system on work-in-progress sports news texts in three languages, we show that it is able to deliver acceptable corrections. 

**Abstract (ZH)**: 在本文中，我们介绍了一个系统，该系统利用大型语言模型（LLM）作为自然语言生成（NLG）系统生成文本的质量保证（QA）组成部分，以执行语法和拼写纠错。这对于实际场景中的文本生成非常重要。我们通过对三种语言的工作中进行中的体育新闻文本进行评估，结果显示该系统能够提供可接受的纠错效果。 

---
# MADP: Multi-Agent Deductive Planning for Enhanced Cognitive-Behavioral Mental Health Question Answer 

**Title (ZH)**: MADP：增强认知行为心理健康问答的多智能体演绎规划 

**Authors**: Qi Chen, Dexi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15826)  

**Abstract**: The Mental Health Question Answer (MHQA) task requires the seeker and supporter to complete the support process in one-turn dialogue. Given the richness of help-seeker posts, supporters must thoroughly understand the content and provide logical, comprehensive, and well-structured responses. Previous works in MHQA mostly focus on single-agent approaches based on the cognitive element of Cognitive Behavioral Therapy (CBT), but they overlook the interactions among various CBT elements, such as emotion and cognition. This limitation hinders the models' ability to thoroughly understand the distress of help-seekers. To address this, we propose a framework named Multi-Agent Deductive Planning (MADP), which is based on the interactions between the various psychological elements of CBT. This method guides Large Language Models (LLMs) to achieve a deeper understanding of the seeker's context and provide more personalized assistance based on individual circumstances. Furthermore, we construct a new dataset based on the MADP framework and use it to fine-tune LLMs, resulting in a specialized model named MADP-LLM. We conduct extensive experiments, including comparisons with multiple LLMs, human evaluations, and automatic evaluations, to validate the effectiveness of the MADP framework and MADP-LLM. 

**Abstract (ZH)**: 认知行为疗法（CBT）要素下的心理卫生问答（MHQA）任务要求求助者和支持者在一次对话中完成支持过程。鉴于求助者的帖子内容丰富，支持者必须全面理解内容并提供逻辑性强、系统全面且结构良好的回复。以前的MHQA研究主要集中在基于CBT认知要素的单智能体方法上，但它们忽略了CBT中各种要素之间的交互，如情绪和认知。这一限制妨碍了模型全面理解求助者的困扰能力。为解决这一问题，我们提出了一种名为多智能体演绎规划（MADP）的框架，该框架基于CBT的各种心理要素之间的交互。该方法引导大规模语言模型（LLMs）更深入地理解求助者的背景，并根据个人情况提供更加个性化的帮助。此外，我们基于MADP框架构建了一个新的数据集，并使用该数据集对LLMs进行微调，从而创建了一种专门的模型MADP-LLM。我们进行了广泛实证研究，包括与多种LLMs的对比、人工评估和自动评估，以验证MADP框架和MADP-LLM的有效性。 

---
# Adapting Biomedical Abstracts into Plain language using Large Language Models 

**Title (ZH)**: 使用大型语言模型将生物医学摘要转换为通俗语言 

**Authors**: Haritha Gangavarapu, Giridhar Kaushik Ramachandran, Kevin Lybarger, Meliha Yetisgen, Özlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2501.15700)  

**Abstract**: A vast amount of medical knowledge is available for public use through online health forums, and question-answering platforms on social media. The majority of the population in the United States doesn't have the right amount of health literacy to make the best use of that information. Health literacy means the ability to obtain and comprehend the basic health information to make appropriate health decisions. To build the bridge between this gap, organizations advocate adapting this medical knowledge into plain language. Building robust systems to automate the adaptations helps both medical and non-medical professionals best leverage the available information online. The goal of the Plain Language Adaptation of Biomedical Abstracts (PLABA) track is to adapt the biomedical abstracts in English language extracted from PubMed based on the questions asked in MedlinePlus for the general public using plain language at the sentence level. As part of this track, we leveraged the best open-source Large Language Models suitable and fine-tuned for dialog use cases. We compare and present the results for all of our systems and our ranking among the other participants' submissions. Our top performing GPT-4 based model ranked first in the avg. simplicity measure and 3rd on the avg. accuracy measure. 

**Abstract (ZH)**: 通过在线健康论坛和社交媒体上的问答平台，大量医疗知识可供公众使用。然而，美国大多数人口缺乏足够的健康素养，无法充分利用这些信息。健康素养是指获取和理解基本健康信息以作出适当健康决策的能力。为了弥合这一差距，组织倡导将这些医学知识转化为易于理解的语言。构建能够自动化这一转换的 robust 系统，有助于医学和非医学专业人士更好地利用互联网上的可用信息。平易近人语言适应生物医学摘要（PLABA）赛道的目标是基于美国医学指南中的问题，将从 PubMed 提取的英文生物医学摘要转化为平易近人语言，从句层面进行调整。作为该赛道的一部分，我们利用了适合对话应用的最佳开源大型语言模型，并对其进行了微调。我们比较了所有系统的结果，并展示了我们在其他参赛者提交中的排名。基于 GPT-4 的顶级模型在平均简洁性指标中排名第一，在平均准确度指标中排名第三。 

---
# TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs 

**Title (ZH)**: TensorLLM：通过增强推理和压缩的多头注意力张量表示在大规模语言模型中的应用 

**Authors**: Yuxuan Gu, Wuyang Zhou, Giorgos Iacovides, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2501.15674)  

**Abstract**: The reasoning abilities of Large Language Models (LLMs) can be improved by structurally denoising their weights, yet existing techniques primarily focus on denoising the feed-forward network (FFN) of the transformer block, and can not efficiently utilise the Multi-head Attention (MHA) block, which is the core of transformer architectures. To address this issue, we propose a novel intuitive framework that, at its very core, performs MHA compression through a multi-head tensorisation process and the Tucker decomposition. This enables both higher-dimensional structured denoising and compression of the MHA weights, by enforcing a shared higher-dimensional subspace across the weights of the multiple attention heads. We demonstrate that this approach consistently enhances the reasoning capabilities of LLMs across multiple benchmark datasets, and for both encoder-only and decoder-only architectures, while achieving compression rates of up to $\sim 250$ times in the MHA weights, all without requiring any additional data, training, or fine-tuning. Furthermore, we show that the proposed method can be seamlessly combined with existing FFN-only-based denoising techniques to achieve further improvements in LLM reasoning performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理能力可以通过结构化去除其权重中的噪声来提高，但现有的技术主要集中在去除变压器块中的前向网络（FFN）噪声，而无法有效地利用Multi-head Attention（MHA）块，这是变压器架构的核心部分。为了解决这一问题，我们提出了一种新颖直观的框架，该框架的核心是通过多头张量化过程和Tucker分解来执行MHA压缩。这种方法通过在多个注意头的权重中强制共享更高维度的子空间，实现了更高维度的结构化去除噪声和MHA权重的压缩。我们证明了这种做法在多个基准数据集上一致地增强了LLMs的推理能力，无论是在仅编码器架构还是仅解码器架构中，同时在MHA权重的压缩率最高可达约250倍，而无需添加额外的数据、训练或微调。此外，我们展示了所提出的方法可以无缝结合现有的仅基于FFN的去噪技术，进一步提高LLMs的推理性能。 

---
# Error Classification of Large Language Models on Math Word Problems: A Dynamically Adaptive Framework 

**Title (ZH)**: 大型语言模型在数学文字题上的错误分类：一种动态适应性框架 

**Authors**: Yuhong Sun, Zhangyue Yin, Xuanjing Huang, Xipeng Qiu, Hui Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15581)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. Math Word Problems (MWPs) serve as a crucial benchmark for evaluating LLMs' reasoning abilities. While most research primarily focuses on improving accuracy, it often neglects understanding and addressing the underlying patterns of errors. Current error classification methods rely on static and predefined categories, which limit their ability to capture the full spectrum of error patterns in mathematical reasoning. To enable systematic error analysis, we collect error samples from 15 different LLMs of varying sizes across four distinct MWP datasets using multiple sampling strategies. Based on this extensive collection, we introduce MWPES-300K, a comprehensive dataset containing 304,865 error samples that cover diverse error patterns and reasoning paths. To reduce human bias and enable fine-grained analysis of error patterns, we propose a novel framework for automated dynamic error classification in mathematical reasoning. Experimental results demonstrate that dataset characteristics significantly shape error patterns, which evolve from basic to complex manifestations as model capabilities increase. With deeper insights into error patterns, we propose error-aware prompting that incorporates common error patterns as explicit guidance, leading to significant improvements in mathematical reasoning performance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各个领域展现了出色的性能。数学应用题（MWPs）是评估LLMs推理能力的重要基准。尽管大多数研究主要集中在提高准确性上，但往往会忽略对错误背后模式的理解和解决。当前的错误分类方法依赖于静态和预定义的类别，这限制了它们捕捉数学推理中错误模式全面性的能力。为了实现系统性的错误分析，我们通过多种采样策略从四个不同的MWP数据集中收集了15个不同规模LLMs的错误样本。基于这一广泛的收集，我们引入了MWPES-300K，这是一个包含304,865个错误样本的综合数据集，涵盖了多样化的错误模式和推理路径。为了减少人为偏见并实现精细的错误模式分析，我们提出了一种新的自动动态错误分类框架，专门用于数学推理。实验结果表明，数据集的特性显著影响了错误模式，随着模型能力的增强，错误模式从基础的逐步演变为复杂的形态。通过深入了解错误模式，我们提出了一种错误感知的提示方法，结合常见错误模式作为显式指导，这在数学推理性能上取得了显著提升。 

---
# Instruction Tuning for Story Understanding and Generation with Weak Supervision 

**Title (ZH)**: 带有弱监督的故事情节理解与生成指令调优 

**Authors**: Yangshu Yuan, Heng Chen, Christian Ng  

**Link**: [PDF](https://arxiv.org/pdf/2501.15574)  

**Abstract**: Story understanding and generation have long been a challenging task in natural language processing (NLP), especially when dealing with various levels of instruction specificity. In this paper, we propose a novel approach called "Weak to Strong Instruction Tuning" for improving story generation by tuning models with instructions of varying clarity. We explore the potential of large language models (LLMs) to adapt to different types of instructions, weak and strong, and show that our method significantly enhances performance in story comprehension and generation. By leveraging the strength of instruction tuning, we train models to understand the nuances of story plots, characters, and themes while generating coherent and engaging narratives. Through extensive experiments on several benchmark datasets and comparison with state-of-the-art baselines, we demonstrate that our method outperforms existing techniques, yielding substantial improvements in both automatic evaluation metrics and human evaluations. Our work shows that adaptive instruction tuning can be a powerful tool in refining generative models for complex narrative tasks. 

**Abstract (ZH)**: 故事理解与生成一直是自然语言处理（NLP）领域的一项具有挑战性的任务，尤其是在处理不同层次的指令明确性时。本文提出了一种名为“从弱到强指令微调”的新方法，通过使用不同程度清晰度的指令来提高故事生成的效果。我们探索了大型语言模型（LLMs）适应不同类型指令（弱指令和强指令）的潜力，并表明我们的方法在故事理解和生成方面显著提升了性能。通过利用指令微调的优势，我们训练模型理解故事剧情、角色和主题的细微差别，生成连贯且吸引人的叙述。通过在多个基准数据集上进行广泛的实验，并与最新基准技术进行比较，我们证明了我们的方法优于现有技术，不仅在自动评价指标方面，还在人类评价方面都取得了显著的改进。我们的研究表明，适应性指令微调可以成为改进复杂叙事任务生成模型的强大工具。 

---
# OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas 

**Title (ZH)**: 开放角色：使用大规模合成人设训练可定制的角色扮演大语言模型 

**Authors**: Xiaoyang Wang, Hongming Zhang, Tao Ge, Wenhao Yu, Dian Yu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15427)  

**Abstract**: Customizable role-playing in large language models (LLMs), also known as character generalization, is gaining increasing attention for its versatility and cost-efficiency in developing and deploying role-playing dialogue agents. This study explores a large-scale data synthesis approach to equip LLMs with character generalization capabilities. We begin by synthesizing large-scale character profiles using personas from Persona Hub and then explore two strategies: response rewriting and response generation, to create character-aligned instructional responses. To validate the effectiveness of our synthetic instruction tuning data for character generalization, we perform supervised fine-tuning (SFT) using the LLaMA-3 8B model. Our best-performing model strengthens the original LLaMA-3 8B Instruct model and achieves performance comparable to GPT-4o models on role-playing dialogue. We release our synthetic characters and instruction-tuning dialogues to support public research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）中的可定制角色扮演，也被称为角色泛化，因其在开发和部署角色对话代理方面的灵活性和成本效益而越来越受到关注。本研究探讨了大规模数据合成方法以增强LLMs的角色泛化能力。我们首先使用Persona Hub中的persona合成大规模角色档案，然后探索了两种策略：响应重写和响应生成，以创建与角色一致的指令性回应。为了验证我们合成的指令调优数据在角色泛化方面的有效性，我们使用LLaMA-3 8B模型进行了监督微调（SFT）。我们性能最佳的模型增强了原始的LLaMA-3 8B指令模型，并在角色对话方面达到了与GPT-4o模型相当的性能。我们将合成的角色和指令调优对话发布出来，以支持公开研究。 

---
# Prompting ChatGPT for Chinese Learning as L2: A CEFR and EBCL Level Study 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

"使用提示引导ChatGPT进行汉语作为二外的学习：基于CEFR和EBCL的水平研究"

解释：
- "Prompting ChatGPT for Chinese Learning as L2" 翻译为“使用提示引导ChatGPT进行汉语作为二外的学习”。
- "A CEFR and EBCL Level Study" 翻译为“基于CEFR和EBCL的水平研究”。

CEFR代表Common European Framework of Reference for Languages（共同欧洲语言参考框架），EBCL代表European Business Language Test（欧洲商务语言测试），这些都是国际上用来评估语言水平的标准。 

**Authors**: Miao Lin-Zucker, Joël Bellasen, Jean-Daniel Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2501.15247)  

**Abstract**: The use of chatbots in language learning has evolved significantly since the 1960s, becoming more sophisticated platforms as generative AI emerged. These tools now simulate natural conversations, adapting to individual learners' needs, including those studying Chinese. Our study explores how learners can use specific prompts to engage Large Language Models (LLM) as personalized chatbots, aiming to target their language level based on the Common European Framework of Reference for Languages (CEFR) and the European Benchmarking Chinese Language (EBCL) project. Focusing on A1, A1+ and A2 levels, we examine the teaching of Chinese, which presents unique challenges due to its logographic writing system. Our goal is to develop prompts that integrate oral and written skills, using high-frequency character lists and controlling oral lexical productions. These tools, powered by generative AI, aim to enhance language practice by crossing lexical and sinographic recurrence. While generative AI shows potential as a personalized tutor, further evaluation is needed to assess its effectiveness. We conducted a systematic series of experiments using ChatGPT models to evaluate their adherence to constraints specified in the prompts. The results indicate that incorporating level A1 and A1+ characters, along with the associated reference list, significantly enhances compliance with the EBCL character set. Properly prompted, LLMs can increase exposure to the target language and offer interactive exchanges to develop language skills. 

**Abstract (ZH)**: 自20世纪60年代以来，聊天机器人的使用在语言学习中取得了显著进展，随着生成式AI的出现，这些工具变得更为复杂和完善。这些工具现在可以模拟自然对话，能够根据个人学习者的需求进行调整，包括那些正在学习汉语的人。本研究探讨了学习者如何使用特定提示来与大型语言模型（LLM）进行个性化对话，旨在根据《共同欧洲框架reference for Languages（CEFR）》和《欧洲汉语水平评价项目（EBCL）》的要求将语言水平定位。我们专注于A1、A1+和A2级别，研究汉语的教学，而汉语因其表意文字系统存在独特的挑战。我们的目标是开发能够融合口语和书面技能的提示，利用高频汉字列表并控制口语词汇产出。这些工具依靠生成式AI，旨在通过词汇和汉字的重复来增强语言练习。虽然生成式AI作为个性化辅导显示出潜力，但还需进一步评估其有效性。我们使用ChatGPT模型进行了系统性的实验，以评估其在提示规定约束条件下的遵守情况。结果表明，结合A1和A1+级别的汉字及其相应的参考列表，显著增强了对EBCL汉字集的遵守程度。适当提示，大型语言模型可以增加目标语言的接触机会，并提供互动交流以发展语言技能。 

---
# Speech Translation Refinement using Large Language Models 

**Title (ZH)**: 使用大语言模型进行语音翻译精炼 

**Authors**: Huaixia Dou, Xinyu Tian, Xinglin Lyu, Jie Zhu, Junhui Li, Lifan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.15090)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated their remarkable capabilities across various language tasks. Inspired by the success of text-to-text translation refinement, this paper investigates how LLMs can improve the performance of speech translation by introducing a joint refinement process. Through the joint refinement of speech translation (ST) and automatic speech recognition (ASR) transcription via LLMs, the performance of the ST model is significantly improved in both training-free in-context learning and parameter-efficient fine-tuning scenarios. Additionally, we explore the effect of document-level context on refinement under the context-aware fine-tuning scenario. Experimental results on the MuST-C and CoVoST 2 datasets, which include seven translation tasks, demonstrate the effectiveness of the proposed approach using several popular LLMs including GPT-3.5-turbo, LLaMA3-8B, and Mistral-12B. Further analysis further suggests that jointly refining both transcription and translation yields better performance compared to refining translation alone. Meanwhile, incorporating document-level context significantly enhances refinement performance. We release our code and datasets on GitHub. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在各种语言任务中的出色表现已经得到了验证。受文本到文本转换 refinement 成功的启发，本论文研究了通过引入联合 refinement 过程，LLMs 如何提升语音翻译（ST）的性能。通过利用 LLMs 对语音翻译和自动语音识别（ASR）转写进行联合 refinement，ST 模型在无训练情况下的上下文学习和参数高效微调场景中都表现出显著的性能提升。此外，我们还探讨了在上下文感知微调场景中文档级上下文对 refinement 的影响。在包含七个翻译任务的 MuST-C 和 CoVoST 2 数据集上的实验结果表明，使用包括 GPT-3.5-turbo、LLaMA-3-8B 和 Mistral-12B 等几种流行的 LLMs，所提出的方法具有有效性。进一步的分析还表明，同时对转写和翻译进行 joint refinement 比仅对翻译进行 refinement 能够获得更好的性能。同时，在 refinement 过程中引入文档级上下文能够显著提高性能。我们已在 GitHub 上公开了我们的代码和数据集。 

---
# LongReason: A Synthetic Long-Context Reasoning Benchmark via Context Expansion 

**Title (ZH)**: 长推理：通过上下文扩展生成的长语境推理基准 

**Authors**: Zhan Ling, Kang Liu, Kai Yan, Yifan Yang, Weijian Lin, Ting-Han Fan, Lingfeng Shen, Zhengyin Du, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.15089)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable progress in understanding long-context inputs. However, benchmarks for evaluating the long-context reasoning abilities of LLMs fall behind the pace. Existing benchmarks often focus on a narrow range of tasks or those that do not demand complex reasoning. To address this gap and enable a more comprehensive evaluation of the long-context reasoning capabilities of current LLMs, we propose a new synthetic benchmark, LongReason, which is constructed by synthesizing long-context reasoning questions from a varied set of short-context reasoning questions through context expansion. LongReason consists of 794 multiple-choice reasoning questions with diverse reasoning patterns across three task categories: reading comprehension, logical inference, and mathematical word problems. We evaluate 21 LLMs on LongReason, revealing that most models experience significant performance drops as context length increases. Our further analysis shows that even state-of-the-art LLMs still have significant room for improvement in providing robust reasoning across different tasks. We will open-source LongReason to support the comprehensive evaluation of LLMs' long-context reasoning capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在理解长文本上下文方面取得了显著进步。然而，评估LLMs长文本推理能力的基准测试却落在了后面。现有的基准测试往往聚焦于狭窄的任务范围或那些不需要复杂推理的任务。为了填补这一空白并使对当前LLMs长文本推理能力的评估更加全面，我们提出了一种新的合成基准测试——LongReason，该基准测试通过扩展上下文将各种短文本推理问题综合为长文本推理问题。LongReason 包括 794 个多选推理问题，这些问题在三个任务类别（阅读理解、逻辑推理和数学应用题）中具有多样化的推理模式。我们评估了 21 种不同的LLMs在LongReason上的表现，结果显示大多数模型在其推理能力随着上下文长度增加时出现显著下降。进一步分析表明，即使是最先进的LLMs在跨不同任务提供稳健推理方面仍有显著改进的空间。我们将开源LongReason，以支持对LLMs长文本推理能力的全面评估。 

---
# Zero-Shot Decision Tree Construction via Large Language Models 

**Title (ZH)**: 基于大型语言模型的零样本决策树构建 

**Authors**: Lucas Carrasco, Felipe Urrutia, Andrés Abeliuk  

**Link**: [PDF](https://arxiv.org/pdf/2501.16247)  

**Abstract**: This paper introduces a novel algorithm for constructing decision trees using large language models (LLMs) in a zero-shot manner based on Classification and Regression Trees (CART) principles. Traditional decision tree induction methods rely heavily on labeled data to recursively partition data using criteria such as information gain or the Gini index. In contrast, we propose a method that uses the pre-trained knowledge embedded in LLMs to build decision trees without requiring training data. Our approach leverages LLMs to perform operations essential for decision tree construction, including attribute discretization, probability calculation, and Gini index computation based on the probabilities. We show that these zero-shot decision trees can outperform baseline zero-shot methods and achieve competitive performance compared to supervised data-driven decision trees on tabular datasets. The decision trees constructed via this method provide transparent and interpretable models, addressing data scarcity while preserving interpretability. This work establishes a new baseline in low-data machine learning, offering a principled, knowledge-driven alternative to data-driven tree construction. 

**Abstract (ZH)**: 本文介绍了一种基于分类和回归树（CART）原则，使用大型语言模型（LLMs）以零样本方式构造决策树的新算法。传统的决策树归纳方法高度依赖标记数据，通过诸如信息增益或基尼指数等标准递归地对数据进行分割。与之不同，我们提出了一种方法，利用预训练在LLMs中嵌入的知识来构建决策树，而不需要训练数据。我们的方法通过利用LLMs执行决策树构建所需的基本操作，如属性离散化、概率计算和基于概率的基尼指数计算，来实现这一点。我们展示了这些零样本决策树不仅可以超越基线的零样本方法，还能在表格数据集上与监督驱动的决策树竞争。通过这种方法构建的决策树提供了透明且可解释的模型，在缓解数据稀缺性的同时保持了可解释性。这项工作在低数据机器学习领域建立了一个新的基准，提供了一种以知识为导向、而非数据驱动的方式构建决策树的方法。 

---
# Phase Transitions in Large Language Models and the $O(N)$ Model 

**Title (ZH)**: 大型语言模型中的相变现象与$O(N)$模型 

**Authors**: Youran Sun, Babak Haghighat  

**Link**: [PDF](https://arxiv.org/pdf/2501.16241)  

**Abstract**: Large language models (LLMs) exhibit unprecedentedly rich scaling behaviors. In physics, scaling behavior is closely related to phase transitions, critical phenomena, and field theory. To investigate the phase transition phenomena in LLMs, we reformulated the Transformer architecture as an $O(N)$ model. Our study reveals two distinct phase transitions corresponding to the temperature used in text generation and the model's parameter size, respectively. The first phase transition enables us to estimate the internal dimension of the model, while the second phase transition is of \textit{higher-depth} and signals the emergence of new capabilities. As an application, the energy of the $O(N)$ model can be used to evaluate whether an LLM's parameters are sufficient to learn the training data. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展现出前所未有的丰富标度行为。在物理学中，标度行为与相变、临界现象和场论密切相关。为了研究LLMs中的相变现象，我们将Transformer架构重新表述为$O(N)$模型。我们的研究揭示了与文本生成中使用的温度和模型参数量对应的两种不同的相变现象。第一个相变使我们能够估算模型的内部维度，而第二个相变是更高层次的，并标志着新能力的涌现。作为应用，$O(N)$模型的能量可以用于评估LLM的参数是否足以学习训练数据。 

---
# ToMoE: Converting Dense Large Language Models to Mixture-of-Experts through Dynamic Structural Pruning 

**Title (ZH)**: ToMoE: 将稠密大型语言模型转换为混合专家模型的动态结构剪枝方法 

**Authors**: Shangqian Gao, Ting Hua, Reza Shirkavand, Chi-Heng Lin, Zhen Tang, Zhengao Li, Longge Yuan, Fangyi Li, Zeyu Zhang, Alireza Ganjdanesh, Lou Qian, Xu Jie, Yen-Chang Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15316)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable abilities in tackling a wide range of complex tasks. However, their huge computational and memory costs raise significant challenges in deploying these models on resource-constrained devices or efficiently serving them. Prior approaches have attempted to alleviate these problems by permanently removing less important model structures, yet these methods often result in substantial performance degradation due to the permanent deletion of model parameters. In this work, we tried to mitigate this issue by reducing the number of active parameters without permanently removing them. Specifically, we introduce a differentiable dynamic pruning method that pushes dense models to maintain a fixed number of active parameters by converting their MLP layers into a Mixture of Experts (MoE) architecture. Our method, even without fine-tuning, consistently outperforms previous structural pruning techniques across diverse model families, including Phi-2, LLaMA-2, LLaMA-3, and Qwen-2.5. 

**Abstract (ZH)**: 大语言模型（LLMs）在应对各种复杂的任务方面展现了显著的能力。然而，这些模型巨大的计算和内存成本在将其部署在资源受限的设备上或高效地服务于这些模型时提出了重大挑战。先前的方法试图通过永久移除不重要的模型结构来缓解这些问题，但这些方法常常由于永久删除模型参数而导致性能显著下降。在本工作中，我们尝试通过减少活跃参数的数量而无需永久移除参数来缓解这一问题。具体而言，我们提出了一种可微分的动态剪枝方法，将密集模型中的MLP层转换为专家混合（Mixture of Experts，MoE）架构，以保持固定的活跃参数数量。即使在无需微调的情况下，我们的方法在包括Phi-2、LLaMA-2、LLaMA-3和Qwen-2.5等不同模型家族的多种应用中，也持续优于先前的结构剪枝技术。 

---
