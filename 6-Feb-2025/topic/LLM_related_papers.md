# BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving 

**Title (ZH)**: BFS-Prover：基于大规模语言模型的自动定理证明的可扩展最佳优先树搜索方法 

**Authors**: Ran Xin, Chenguang Xi, Jie Yang, Feng Chen, Hang Wu, Xia Xiao, Yifan Sun, Shen Zheng, Kai Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.03438)  

**Abstract**: Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating proof search spaces. While the existing approaches primarily rely on value functions and Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Search (BFS) remains underexplored. This paper investigates whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present \texttt{BFS-Prover}, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. \texttt{BFS-Prover} achieves a score of $71.31$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进步激发了对使用Lean4进行自动定理证明的兴趣，而有效的树搜索方法对于导航证明搜索空间至关重要。尽管现有的方法主要依赖于价值函数和蒙特卡洛树搜索（MCTS），但像最佳优先搜索（BFS）这样简单的方法的潜力仍未得到充分探索。本文探讨了BFS是否能在大规模定理证明任务中达到竞争性的性能。我们提出了一种可扩展的专家迭代框架\texttt{BFS-Prover}，并包含三项关键技术创新。首先，在每次专家迭代中实施战略数据过滤，排除可以通过束搜索节点扩展解决的问题，重点关注更难的案例。其次，通过直接偏好优化（DPO）改进BFS的样本效率，该方法应用于自动注释有编译器错误反馈的状态-策略对，以细化LLM的策略，使其优先考虑有成效的扩展。第三，我们采用长度规范化在BFS中，以促进对更深层证明路径的探索。在MiniF2F测试集上，\texttt{BFS-Prover} 达到了71.31的得分，从而挑战了复杂树搜索方法的必要性，证明了在适当规模化的条件下，BFS可以达到竞争性的性能。 

---
# SymAgent: A Neural-Symbolic Self-Learning Agent Framework for Complex Reasoning over Knowledge Graphs 

**Title (ZH)**: SymAgent：一种用于知识图谱复杂推理的神经符号自我学习代理框架 

**Authors**: Ben Liu, Jihai Zhang, Fangquan Lin, Cheng Yang, Min Peng, Wotao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.03283)  

**Abstract**: Recent advancements have highlighted that Large Language Models (LLMs) are prone to hallucinations when solving complex reasoning problems, leading to erroneous results. To tackle this issue, researchers incorporate Knowledge Graphs (KGs) to improve the reasoning ability of LLMs. However, existing methods face two limitations: 1) they typically assume that all answers to the questions are contained in KGs, neglecting the incompleteness issue of KGs, and 2) they treat the KG as a static repository and overlook the implicit logical reasoning structures inherent in KGs. In this paper, we introduce SymAgent, an innovative neural-symbolic agent framework that achieves collaborative augmentation between KGs and LLMs. We conceptualize KGs as dynamic environments and transform complex reasoning tasks into a multi-step interactive process, enabling KGs to participate deeply in the reasoning process. SymAgent consists of two modules: Agent-Planner and Agent-Executor. The Agent-Planner leverages LLM's inductive reasoning capability to extract symbolic rules from KGs, guiding efficient question decomposition. The Agent-Executor autonomously invokes predefined action tools to integrate information from KGs and external documents, addressing the issues of KG incompleteness. Furthermore, we design a self-learning framework comprising online exploration and offline iterative policy updating phases, enabling the agent to automatically synthesize reasoning trajectories and improve performance. Experimental results demonstrate that SymAgent with weak LLM backbones (i.e., 7B series) yields better or comparable performance compared to various strong baselines. Further analysis reveals that our agent can identify missing triples, facilitating automatic KG updates. 

**Abstract (ZH)**: 近年来的研究表明，大型语言模型（LLMs）在解决复杂推理问题时容易产生幻觉（hallucination），导致错误的结果。为了解决这一问题，研究人员通过引入知识图（KGs）来提高LLMs的推理能力。然而，现有方法存在两个局限性：1）它们通常假设所有答案都包含在KGs中，忽视了KGs的不完整性问题；2）它们将KG视为静态资源，并忽略了KG中固有的隐含逻辑推理结构。在此论文中，我们提出了SymAgent，这是一种创新的神经-符号代理框架，实现了KGs和LLMs之间的协作增强。我们将KGs视为动态环境，并将复杂的推理任务转化为多步交互过程，使KGs能够深度参与推理过程。SymAgent由两个模块组成：Agent-Planner和Agent-Executor。Agent-Planner利用LLMs的归纳推理能力从KGs中提取符号规则，指导有效的问题分解。Agent-Executor自主调用预定义的动作工具，整合KGs和外部文档中的信息，解决KG不完整的问题。此外，我们设计了一个自我学习框架，包括在线探索和离线迭代策略更新阶段，使代理能够自动综合推理轨迹并提高性能。实验结果表明，使用较弱的LLM底座（如7B系列）的SymAgent相较于各种强大的基线具有更好的或相当的性能。进一步分析表明，我们的代理能够识别缺失的三元组，从而促进自动更新KG。 

---
# SensorChat: Answering Qualitative and Quantitative Questions during Long-Term Multimodal Sensor Interactions 

**Title (ZH)**: SensorChat: 在长期多模态传感器交互中回答定性与定量问题 

**Authors**: Xiaofan Yu, Lanxiang Hu, Benjamin Reichman, Dylan Chu, Rushil Chandrupatla, Xiyuan Zhang, Larry Heck, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2502.02883)  

**Abstract**: Natural language interaction with sensing systems is crucial for enabling all users to comprehend sensor data and its impact on their everyday lives. However, existing systems, which typically operate in a Question Answering (QA) manner, are significantly limited in terms of the duration and complexity of sensor data they can handle. In this work, we introduce SensorChat, the first end-to-end QA system designed for long-term sensor monitoring with multimodal and high-dimensional data including time series. SensorChat effectively answers both qualitative (requiring high-level reasoning) and quantitative (requiring accurate responses derived from sensor data) questions in real-world scenarios. To achieve this, SensorChat uses an innovative three-stage pipeline that includes question decomposition, sensor data query, and answer assembly. The first and third stages leverage Large Language Models (LLMs) for intuitive human interactions and to guide the sensor data query process. Unlike existing multimodal LLMs, SensorChat incorporates an explicit query stage to precisely extract factual information from long-duration sensor data. We implement SensorChat and demonstrate its capability for real-time interactions on a cloud server while also being able to run entirely on edge platforms after quantization. Comprehensive QA evaluations show that SensorChat achieves up to 26% higher answer accuracy than state-of-the-art systems on quantitative questions. Additionally, a user study with eight volunteers highlights SensorChat's effectiveness in handling qualitative and open-ended questions. 

**Abstract (ZH)**: 自然语言与传感系统的交互对于使所有用户能够理解传感器数据及其对日常生活的影响至关重要。然而，现有的系统通常以问答（QA）的方式运行，它们在处理传感器数据的时间长度和复杂性方面存在显著限制。在此项工作中，我们引入了SensorChat，这是一种专门为长时间传感器监测设计的端到端问答系统，能够处理包括时间序列在内的多模态和高维数据。SensorChat能够有效地回答质性问题（需要高层次的推理）和量化问题（需要从传感器数据中得出精确的回答），这些问题在实际场景中频繁出现。为了实现这一点，SensorChat采用了一个创新性的三阶段管道，包括问题分解、传感器数据查询和答案组装。前两阶段使用大语言模型（LLMs）来实现直观的人机交互，并指导传感器数据查询过程。与现有的多模态大语言模型不同，SensorChat引入了一个显式查询阶段，能够精确提取长时间传感器数据中的事实信息。我们实现了SensorChat，并在云服务器上展示了其实时交互的能力，同时在量化后也可以在边缘平台上完全运行。全面的问答评估表明，在量化问题上，SensorChat的回答准确性比最先进的系统高出26%。此外，一项涉及八名志愿者的用户研究证实了SensorChat在处理质性和开放性问题方面的有效性。 

---
# A Schema-Guided Reason-while-Retrieve framework for Reasoning on Scene Graphs with Large-Language-Models (LLMs) 

**Title (ZH)**: 基于结构引导的在检索中推理框架：使用大规模语言模型（LLMs）对场景图进行推理 

**Authors**: Yiye Chen, Harpreet Sawhney, Nicholas Gydé, Yanan Jian, Jack Saunders, Patricio Vela, Ben Lundell  

**Link**: [PDF](https://arxiv.org/pdf/2502.03450)  

**Abstract**: Scene graphs have emerged as a structured and serializable environment representation for grounded spatial reasoning with Large Language Models (LLMs). In this work, we propose SG-RwR, a Schema-Guided Retrieve-while-Reason framework for reasoning and planning with scene graphs. Our approach employs two cooperative, code-writing LLM agents: a (1) Reasoner for task planning and information queries generation, and a (2) Retriever for extracting corresponding graph information following the queries. Two agents collaborate iteratively, enabling sequential reasoning and adaptive attention to graph information. Unlike prior works, both agents are prompted only with the scene graph schema rather than the full graph data, which reduces the hallucination by limiting input tokens, and drives the Reasoner to generate reasoning trace this http URL the trace, the Retriever programmatically query the scene graph data based on the schema understanding, allowing dynamic and global attention on the graph that enhances alignment between reasoning and retrieval. Through experiments in multiple simulation environments, we show that our framework surpasses existing LLM-based approaches in numerical Q\&A and planning tasks, and can benefit from task-level few-shot examples, even in the absence of agent-level demonstrations. Project code will be released. 

**Abstract (ZH)**: 场景图已作为Large Language Models (LLMs) 进行有grounding的空间推理时的一种结构化和可序列化环境表示而崭露头角。本文中，我们提出了SG-RwR，这是一种基于Schema-Guided Retrieve-while-Reason框架，用于利用场景图进行推理和规划。我们的方法使用了两个协作的代码编写LLM代理：一个（1）推理器，用于任务规划和信息查询生成；另一个（2）检索器，根据查询提取相应的图信息。两个代理通过迭代协作，实现了顺序推理和对图信息的适应性关注。与以往工作不同，这两个代理仅被提示场景图的模式而非完整的图数据，这通过限制输入token来减少妄想，并促使推理器生成推理轨迹。基于轨迹，检索器可以根据模式理解程序化地查询场景图数据，从而在图上实现动态和全局关注，提高推理和检索之间的对齐。通过在多个模拟环境中进行实验，我们展示了本框架在数值问答和规划任务中优于现有的基于LLM的方法，并且可以在缺乏代理级示范的情况下从任务级的少量示例中受益。项目代码将开源。 

---
# LIMO: Less is More for Reasoning 

**Title (ZH)**: LIMO：更少即是更多，精简促进推理 

**Authors**: Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.03387)  

**Abstract**: We present a fundamental discovery that challenges our understanding of how complex reasoning emerges in large language models. While conventional wisdom suggests that sophisticated reasoning tasks demand extensive training data (>100,000 examples), we demonstrate that complex mathematical reasoning abilities can be effectively elicited with surprisingly few examples. Through comprehensive experiments, our proposed model LIMO demonstrates unprecedented performance in mathematical reasoning. With merely 817 curated training samples, LIMO achieves 57.1% accuracy on AIME and 94.8% on MATH, improving from previous SFT-based models' 6.5% and 59.2% respectively, while only using 1% of the training data required by previous approaches. LIMO demonstrates exceptional out-of-distribution generalization, achieving 40.5% absolute improvement across 10 diverse benchmarks, outperforming models trained on 100x more data, challenging the notion that SFT leads to memorization rather than generalization. Based on these results, we propose the Less-Is-More Reasoning Hypothesis (LIMO Hypothesis): In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning capabilities can emerge through minimal but precisely orchestrated demonstrations of cognitive processes. This hypothesis posits that the elicitation threshold for complex reasoning is determined by two key factors: (1) the completeness of the model's encoded knowledge foundation during pre-training, and (2) the effectiveness of post-training examples as "cognitive templates" that show the model how to utilize its knowledge base to solve complex reasoning tasks. To facilitate reproducibility and future research in data-efficient reasoning, we release LIMO as a comprehensive open-source suite at this https URL. 

**Abstract (ZH)**: 我们提出了一个根本性的发现，挑战了我们对大型语言模型中复杂推理是如何产生的理解。尽管传统观点认为复杂的推理任务需要大量的训练数据（>100,000 个示例），我们证明了复杂的数学推理能力可以用出人意料的少量示例有效激发。通过全面的实验，我们提出的模型 LIMO 在数学推理方面取得了前所未有的性能。仅使用 817 个精挑细选的训练样本，LIMO 在 AIME 上达到了 57.1% 的准确率，在 MATH 上达到了 94.8% 的准确率，分别比之前的 SFT 基准模型提高了 50.6% 和 35.6%，同时只使用了之前方法所需训练数据的 1%。LIMO 展示了出色的泛化能力，在 10 个不同的基准测试中取得了 40.5% 的绝对提升，超越了在 100 倍更多数据上训练的模型，挑战了 SFT 导致记忆而非泛化的观点。基于这些结果，我们提出了“少即是多推理假设”（LIMO 假说）：在基础模型中，如果在其预训练期间全面编码了领域知识，那么通过少量但精准调节的认知过程演示，复杂的推理能力可以得以产生。该假设表明，复杂推理的激发阈值由两个关键因素决定：（1）模型在预训练期间编码的知识基础的完整性；（2）后训练示例作为“认知模板”的有效性，展示了模型如何利用其知识库解决复杂的推理任务。为了促进高效推理的可再现性和未来研究，我们在此 https://github.com/阿里云Qwen/LIMO 开放源代码 LIMO 作为全面的开源套件。

注：上述翻译基于您提供的英文文本进行，其中"this https URL"部分保持英文形式，这是标准的做法。如果您需要提供具体的链接内容，请告知。 

---
# MeDiSumQA: Patient-Oriented Question-Answer Generation from Discharge Letters 

**Title (ZH)**: MeDiSumQA：从出院病历中生成患者导向的问题与答案摘要 

**Authors**: Amin Dada, Osman Alperen Koras, Marie Bauer, Amanda Butler, Kaleb E. Smith, Jens Kleesiek, Julian Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2502.03298)  

**Abstract**: While increasing patients' access to medical documents improves medical care, this benefit is limited by varying health literacy levels and complex medical terminology. Large language models (LLMs) offer solutions by simplifying medical information. However, evaluating LLMs for safe and patient-friendly text generation is difficult due to the lack of standardized evaluation resources. To fill this gap, we developed MeDiSumQA. MeDiSumQA is a dataset created from MIMIC-IV discharge summaries through an automated pipeline combining LLM-based question-answer generation with manual quality checks. We use this dataset to evaluate various LLMs on patient-oriented question-answering. Our findings reveal that general-purpose LLMs frequently surpass biomedical-adapted models, while automated metrics correlate with human judgment. By releasing MeDiSumQA on PhysioNet, we aim to advance the development of LLMs to enhance patient understanding and ultimately improve care outcomes. 

**Abstract (ZH)**: 尽管增加患者对医疗文件的访问能够改善医疗服务，这一好处受到不同健康素养水平和复杂医学术语的限制。大规模语言模型（LLMs）通过简化医疗信息提供了解决方案。然而，由于缺乏标准化的评估资源，对LLMs进行安全性和患者友好的文本生成评估颇具挑战性。为填补这一空白，我们开发了MeDiSumQA。MeDiSumQA是由MIMIC-IV出院总结通过一个自动化的管道生成的数据集，该管道结合了基于LLM的问题-答案生成和人工质量检查。我们使用该数据集评估各种LLMs在以患者为中心的问题回答方面的性能。我们的研究发现，通用语言模型通常优于医学适应型模型，而自动评估指标与人类判断相关。通过在PhysioNet上公开MeDiSumQA，我们旨在促进LLMs的发展，以增强患者的理解并最终改善护理结果。 

---
# Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning 

**Title (ZH)**: Token Assorted：结合潜在词元和文本词元以提高语言模型推理能力 

**Authors**: DiJia Su, Hanlin Zhu, Yingchen Xu, Jiantao Jiao, Yuandong Tian, Qinqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.03275)  

**Abstract**: Large Language Models (LLMs) excel at reasoning and planning when trained on chainof-thought (CoT) data, where the step-by-step thought process is explicitly outlined by text tokens. However, this results in lengthy inputs where many words support textual coherence rather than core reasoning information, and processing these inputs consumes substantial computation resources. In this work, we propose a hybrid representation of the reasoning process, where we partially abstract away the initial reasoning steps using latent discrete tokens generated by VQ-VAE, significantly reducing the length of reasoning traces. We explore the use of latent trace abstractions in two scenarios: 1) training the model from scratch for the Keys-Finding Maze problem, 2) fine-tuning LLMs on this hybrid data with an extended vocabulary including unseen latent tokens, for both logical and mathematical reasoning problems. To facilitate effective learning, we introduce a simple training procedure that randomly mixes latent and text tokens, which enables fast adaptation to new latent tokens. Our approach consistently outperforms the baselines methods in various benchmarks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在训练于链式思考（CoT）数据集上时，能够很好地进行推理和规划，其中推理过程通过文本标记明确列出。然而，这导致输入变得相当长，其中许多单词增加了文本连贯性而非核心推理信息，处理这些输入消耗了大量的计算资源。在本研究中，我们提出了一种混合表示推理过程的方法，通过部分使用由VQ-VAE生成的潜在离散标记来抽象初始推理步骤，显著减少了推理追踪的长度。我们探讨了潜在追踪抽象在两种场景中的应用：1) 从头开始训练模型解决“寻找钥匙迷宫”问题；2) 在这种混合数据上微调LLMs，包括扩展词汇表，包含看不见的潜在标记，以解决逻辑和数学推理问题。为促进有效的学习，我们引入了一种简单的训练程序，随机混合潜在标记和文本标记，这使得模型能够快速适应新的潜在标记。我们的方法在各种基准测试中始终优于基准方法。 

---
# Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models 

**Title (ZH)**: 通过词-token层面跨层熵改进大型语言模型的解码事实性 

**Authors**: Jialiang Wu, Yi Shen, Sijia Liu, Yi Tang, Sen Song, Xiaoyi Wang, Longjun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2502.03199)  

**Abstract**: Despite their impressive capacities, Large language models (LLMs) often struggle with the hallucination issue of generating inaccurate or fabricated content even when they possess correct knowledge. In this paper, we extend the exploration of the correlation between hidden-state prediction changes and output factuality into a deeper, token-wise level. Based on the insights , we propose cross-layer Entropy eNhanced Decoding (END), a decoding method that mitigates hallucinations without requiring extra training. END leverages inner probability changes across layers to individually quantify the factual knowledge required for each candidate token, and adjusts the final predicting distribution to prioritize tokens with higher factuality. Experiments on both hallucination and QA benchmarks demonstrate that END significantly enhances the truthfulness and informativeness of generated content while maintaining robust QA accuracy. Moreover, our work provides a deeper perspective on understanding the correlations between inherent knowledge and output factuality. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具有出色的能力，但在拥有正确知识的情况下，它们往往难以避免生成不准确或虚构内容的幻觉问题。在本文中，我们进一步将隐藏状态预测变化与输出事实性的相关性探索扩展到更深层次的、以令牌为基础的层面。基于这些洞察，我们提出了一种名为跨层熵增强解码（END）的解码方法，该方法能够在不需要额外训练的情况下减轻幻觉问题。END 利用跨层内概率变化，分别量化每个候选令牌所需的事实知识，并调整最终的预测分布，以优先考虑具有更高事实性的令牌。在幻觉和问答基准测试中的实验表明，END 显著提高了生成内容的真实性与信息量，同时保持了稳健的问答准确性。此外，我们的工作为理解固有知识与输出事实性之间的关系提供了更深入的视角。 

---
# Scalable In-Context Learning on Tabular Data via Retrieval-Augmented Large Language Models 

**Title (ZH)**: 通过检索增强大型语言模型实现表格数据的大规模上下文学习 

**Authors**: Xumeng Wen, Shun Zheng, Zhen Xu, Yiming Sun, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2502.03147)  

**Abstract**: Recent studies have shown that large language models (LLMs), when customized with post-training on tabular data, can acquire general tabular in-context learning (TabICL) capabilities. These models are able to transfer effectively across diverse data schemas and different task domains. However, existing LLM-based TabICL approaches are constrained to few-shot scenarios due to the sequence length limitations of LLMs, as tabular instances represented in plain text consume substantial tokens. To address this limitation and enable scalable TabICL for any data size, we propose retrieval-augmented LLMs tailored to tabular data. Our approach incorporates a customized retrieval module, combined with retrieval-guided instruction-tuning for LLMs. This enables LLMs to effectively leverage larger datasets, achieving significantly improved performance across 69 widely recognized datasets and demonstrating promising scaling behavior. Extensive comparisons with state-of-the-art tabular models reveal that, while LLM-based TabICL still lags behind well-tuned numeric models in overall performance, it uncovers powerful algorithms under limited contexts, enhances ensemble diversity, and excels on specific datasets. These unique properties underscore the potential of language as a universal and accessible interface for scalable tabular data learning. 

**Abstract (ZH)**: 近年来的研究表明，通过在表格数据上进行后训练，大型语言模型（LLMs）能够获得一般的表格上下文学习（TabICL）能力。这些模型能够有效地跨越多种数据模式和不同的任务领域进行迁移。然而，现有的基于LLM的TabICL方法由于序列长度的限制，大多局限于少样本场景，因为以文本形式表示的表格实例会消耗大量的令牌。为了解决这一限制，并能够对任意大小的数据进行扩展的TabICL，我们提出了一种针对表格数据的检索增强LLMs方法。该方法结合了一个定制的检索模块，并通过检索指导的指令微调来对LLMs进行优化。这使得LLMs能够充分利用更大规模的数据集，实现了在69个广泛认可的基准数据集上的显著性能提升，并表现出良好的扩展行为。与最先进的表格模型的广泛比较表明，虽然基于LLM的TabICL在整体性能上仍落后于仔细调优的数值模型，但它在有限上下文中揭示了强大的算法，增强了模型多样性，并在特定数据集上表现出色。这些独特的特性突显了语言作为面向大规模表格数据学习的通用且可访问界面的潜力。 

---
# MedBioLM: Optimizing Medical and Biological QA with Fine-Tuned Large Language Models and Retrieval-Augmented Generation 

**Title (ZH)**: MedBioLM：通过微调大规模语言模型和检索增强生成技术优化医学和生物学问答 

**Authors**: Seonok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.03004)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across natural language processing tasks. However, their application to specialized domains such as medicine and biology requires further optimization to ensure factual accuracy, reliability, and contextual depth. We introduce MedBioLM, a domain-adapted biomedical question-answering model designed to enhance both short-form and long-form queries. By integrating fine-tuning and retrieval-augmented generation (RAG), MedBioLM dynamically incorporates domain-specific knowledge, improving reasoning abilities and factual accuracy. To evaluate its effectiveness, we fine-tuned the model on diverse biomedical QA datasets, covering structured multiple-choice assessments and complex clinical reasoning tasks. Fine-tuning significantly improves accuracy on benchmark datasets, while RAG enhances factual consistency. These results highlight the potential of domain-optimized LLMs in advancing biomedical research, medical education, and clinical decision support. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理任务中展现出了令人印象深刻的性能。然而，将其应用于医学和生物学等专门领域需要进一步优化，以确保事实的准确性、可靠性和情境深度。我们介绍了MedBioLM，这是一种专门设计的生物医学问答模型，旨在提高短形式和长形式查询的能力。通过集成微调和检索增强生成（RAG）技术，MedBioLM动态地融入了领域特定的知识，从而提升了推理能力和事实准确性。为了评估其有效性，我们在多种生物医学问答数据集上进行了微调，涵盖了结构化的多项选择评估和复杂的临床推理任务。微调在基准数据集上的准确率显著提高，而RAG则增强了事实的一致性。这些结果突显了优化领域的大规模语言模型在促进生物医学研究、医学教育和临床决策支持方面的潜力。 

---
# Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons 

**Title (ZH)**: 训练作为裁判的大型语言模型：流程、见解与实践经验 

**Authors**: Renjun Hu, Yi Cheng, Libin Meng, Jiaxin Xia, Yi Zong, Xing Shi, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.02988)  

**Abstract**: The rapid advancement of large language models (LLMs) has opened new possibilities for their adoption as evaluative judges. This paper introduces Themis, a fine-tuned LLM judge that delivers sophisticated context-aware evaluations. We provide a comprehensive overview of the development pipeline for Themis, highlighting its scenario-dependent evaluation prompts and two novel methods for controlled instruction generation. These designs enable Themis to effectively distill evaluative skills from teacher models, while retaining flexibility for continuous development. We introduce two human-labeled benchmarks for meta-evaluation, demonstrating that Themis can achieve high alignment with human preferences in an economical manner. Additionally, we explore insights into the LLM-as-a-judge paradigm, revealing nuances in performance and the varied effects of reference answers. Notably, we observe that pure knowledge distillation from strong LLMs, though common, does not guarantee performance improvement through scaling. We propose a mitigation strategy based on instruction-following difficulty. Furthermore, we provide practical guidelines covering data balancing, prompt customization, multi-objective training, and metric aggregation. We aim for our method and findings, along with the fine-tuning data, benchmarks, and model checkpoints, to support future research and development in this area. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展为将其作为评估法官的应用开启了新的可能性。本文介绍了Themis，这是一种细调的LLM法官，能够提供复杂的上下文感知评估。我们详细介绍了Themis的开发流程，强调了其场景依赖的评估提示，并介绍了两种新的控制指令生成方法。这些设计使得Themis能够有效地从教师模型中提炼评估技能，同时保留持续开发的灵活性。我们介绍了两个元评估的人工标注基准，展示了Themis能够在经济有效的方式下实现对人类偏好的高度一致。此外，我们探讨了LLM作为法官的范式，揭示了其性能中的复杂性以及参考答案的多样效用。值得注意的是，我们观察到，尽管从强大的LLM中提取纯粹的知识是一种常见做法，但通过扩展并不能保证性能的提升。我们提出了基于指令跟随难度的缓解策略。此外，我们还提供了关于数据平衡、提示定制、多目标训练和指标聚合的实用指南。我们希望我们的方法和发现，包括细调数据、基准和模型检查点，能够支持该领域未来的研究和发展。 

---
# LLM-KT: Aligning Large Language Models with Knowledge Tracing using a Plug-and-Play Instruction 

**Title (ZH)**: LLM-KT：通过可插拔指令对大型语言模型与知识追踪进行对齐 

**Authors**: Ziwei Wang, Jie Zhou, Qin Chen, Min Zhang, Bo Jiang, Aimin Zhou, Qinchun Bai, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2502.02945)  

**Abstract**: The knowledge tracing (KT) problem is an extremely important topic in personalized education, which aims to predict whether students can correctly answer the next question based on their past question-answer records. Prior work on this task mainly focused on learning the sequence of behaviors based on the IDs or textual information. However, these studies usually fail to capture students' sufficient behavioral patterns without reasoning with rich world knowledge about questions. In this paper, we propose a large language models (LLMs)-based framework for KT, named \texttt{\textbf{LLM-KT}}, to integrate the strengths of LLMs and traditional sequence interaction models. For task-level alignment, we design Plug-and-Play instruction to align LLMs with KT, leveraging LLMs' rich knowledge and powerful reasoning capacity. For modality-level alignment, we design the plug-in context and sequence to integrate multiple modalities learned by traditional methods. To capture the long context of history records, we present a plug-in context to flexibly insert the compressed context embedding into LLMs using question-specific and concept-specific tokens. Furthermore, we introduce a plug-in sequence to enhance LLMs with sequence interaction behavior representation learned by traditional sequence models using a sequence adapter. Extensive experiments show that \texttt{\textbf{LLM-KT}} obtains state-of-the-art performance on four typical datasets by comparing it with approximately 20 strong baselines. 

**Abstract (ZH)**: 知识追踪（KT）问题是个性化教育中的一个极其重要的研究主题，旨在基于学生以往的问题答题记录预测他们是否能正确回答下一个问题。在此之前，对该任务的研究主要集中在基于问题ID或文本信息学习行为序列上。然而，这些研究通常未能捕捉到学生的行为模式，尤其是在缺乏关于问题的丰富背景知识的情况下。本文提出了一种基于大语言模型（LLMs）的知识追踪框架，命名为\texttt{\textbf{LLM-KT}}，以整合LLMs和传统序列交互模型的优势。为了任务级别对齐，我们设计了一种可插拔指令，利用LLMs丰富的知识和强大的推理能力将LLMs与知识追踪进行对齐。为了模态级别对齐，我们设计了插件上下文和序列以整合传统方法学习的多种模态信息。为了捕捉历史记录中的长上下文，我们提出了一个插件上下文，利用问题特定和概念特定的标记将压缩的上下文嵌入灵活地插入到LLMs中。此外，我们引入了一个插件序列，通过序列适配器增强了LLMs，使其具备传统序列模型学习到的序列交互行为表示。广泛的实验表明，\texttt{\textbf{LLM-KT}}在四个典型数据集上超过了约20种强大的基线方法，获得了最先进的性能。 

---
# Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning 

**Title (ZH)**: 位置：多模态大型语言模型可以显著推进科学推理 

**Authors**: Yibo Yan, Shen Wang, Jiahao Huo, Jingheng Ye, Zhendong Chu, Xuming Hu, Philip S. Yu, Carla Gomes, Bart Selman, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.02871)  

**Abstract**: Scientific reasoning, the process through which humans apply logic, evidence, and critical thinking to explore and interpret scientific phenomena, is essential in advancing knowledge reasoning across diverse fields. However, despite significant progress, current scientific reasoning models still struggle with generalization across domains and often fall short of multimodal perception. Multimodal Large Language Models (MLLMs), which integrate text, images, and other modalities, present an exciting opportunity to overcome these limitations and enhance scientific reasoning. Therefore, this position paper argues that MLLMs can significantly advance scientific reasoning across disciplines such as mathematics, physics, chemistry, and biology. First, we propose a four-stage research roadmap of scientific reasoning capabilities, and highlight the current state of MLLM applications in scientific reasoning, noting their ability to integrate and reason over diverse data types. Second, we summarize the key challenges that remain obstacles to achieving MLLM's full potential. To address these challenges, we propose actionable insights and suggestions for the future. Overall, our work offers a novel perspective on MLLM integration with scientific reasoning, providing the LLM community with a valuable vision for achieving Artificial General Intelligence (AGI). 

**Abstract (ZH)**: 科学推理是人类运用逻辑、证据和批判性思维探索和解释科学现象的过程，对于跨学科知识推理的发展至关重要。尽管已取得显著进展，现有科学推理模型仍然在领域间的泛化以及多模态感知方面存在局限。多模态大型语言模型（MLLMs），通过集成文本、图像和其他模态信息，为克服这些局限和提升科学推理提供了令人兴奋的机遇。因此，本文立场认为MLLMs可以在数学、物理学、化学和生物学等学科中显著推进科学推理。首先，我们提出了科学推理能力的四阶段研究路线图，并强调了当前MLLM在科学推理中的应用状态，指出其在整合和处理多种数据类型方面的优势。其次，我们总结了仍然阻碍MLLM充分发挥潜力的关键挑战，并提出了解决这些挑战的具体建议。总体而言，我们的工作为MLLM与科学推理集成提供了一个新颖视角，为大语言模型（LLM）社区提供了实现通用人工智能（AGI）的宝贵愿景。 

---
# A Systematic Approach for Assessing Large Language Models' Test Case Generation Capability 

**Title (ZH)**: 一种系统性方法评估大型语言模型的测试案例生成能力 

**Authors**: Hung-Fu Chang, Mohammad Shokrolah Shirazi  

**Link**: [PDF](https://arxiv.org/pdf/2502.02866)  

**Abstract**: Software testing ensures the quality and reliability of software products, but manual test case creation is labor-intensive. With the rise of large language models (LLMs), there is growing interest in unit test creation with LLMs. However, effective assessment of LLM-generated test cases is limited by the lack of standardized benchmarks that comprehensively cover diverse programming scenarios. To address the assessment of LLM's test case generation ability and lacking dataset for evaluation, we propose the Generated Benchmark from Control-Flow Structure and Variable Usage Composition (GBCV) approach, which systematically generates programs used for evaluating LLMs' test generation capabilities. By leveraging basic control-flow structures and variable usage, GBCV provides a flexible framework to create a spectrum of programs ranging from simple to complex. Because GPT-4o and GPT-3-Turbo are publicly accessible models, to present real-world regular user's use case, we use GBCV to assess LLM performance on them. Our findings indicate that GPT-4o performs better on complex program structures, while all models effectively detect boundary values in simple conditions but face challenges with arithmetic computations. This study highlights the strengths and limitations of LLMs in test generation, provides a benchmark framework, and suggests directions for future improvement. 

**Abstract (ZH)**: 软件测试确保软件产品的质量和可靠性，但手工创建测试用例耗费大量人力。随着大型语言模型（LLMs）的发展，使用LLMs生成单元测试变得越来越受欢迎。然而，有效评估LLMs生成的测试用例的能力受到缺乏全面覆盖多样编程场景的标准基准的限制。为了解决LLMs测试生成能力的评估问题及缺乏评估数据集，我们提出了基于控制流结构和变量使用组成生成基准（GBCV）的方法，该方法系统地生成用于评估LLMs测试生成能力的程序。通过利用基本的控制流结构和变量使用，GBCV提供了一个灵活的框架，可以创建从简单到复杂的程序谱系。由于GPT-4o和GPT-3-Turbo是可公开访问的模型，为了展示真实的用户使用案例，我们使用GBCV评估了这些模型的性能。我们的研究结果表明，GPT-4o在复杂的程序结构上表现更好，而所有模型在简单的边界条件中都能有效检测边界值，但在算术计算方面存在挑战。本研究突显了LLMs在测试生成中的优势与局限，提供了基准框架，并提出了未来改进的方向。 

---
# Mol-LLM: Generalist Molecular LLM with Improved Graph Utilization 

**Title (ZH)**: Mol-LLM：擅长图利用的通用分子大型语言模型 

**Authors**: Chanhui Lee, Yuheon Song, YongJun Jeong, Hanbum Ko, Rodrigo Hormazabal, Sehui Han, Kyunghoon Bae, Sungbin Lim, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.02810)  

**Abstract**: Recent advances in Large Language Models (LLMs) have motivated the development of general LLMs for molecular tasks. While several studies have demonstrated that fine-tuned LLMs can achieve impressive benchmark performances, they are far from genuine generalist molecular LLMs due to a lack of fundamental understanding of molecular structure. Specifically, when given molecular task instructions, LLMs trained with naive next-token prediction training assign similar likelihood scores to both original and negatively corrupted molecules, revealing their lack of molecular structure understanding that is crucial for reliable and general molecular LLMs. To overcome this limitation and obtain a true generalist molecular LLM, we introduce a novel multi-modal training method based on a thorough multi-modal instruction tuning as well as a molecular structure preference optimization between chosen and rejected graphs. On various molecular benchmarks, the proposed generalist molecular LLM, called Mol-LLM, achieves state-of-the-art performances among generalist LLMs on most tasks, at the same time, surpassing or comparable to state-of-the-art specialist LLMs. Moreover, Mol-LLM also shows superior generalization performances in reaction prediction tasks, demonstrating the effect of the molecular structure understanding for generalization perspective. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的发展促进了适用于分子任务的通用LLMs的研究。尽管已有研究表明微调后的LLMs可以在基准测试中取得令人印象深刻的性能，但它们远不能成为真正的通用分子LLMs，因为缺乏对分子结构的基本理解。具体而言，当给定分子任务指令时，通过简单的下一个令牌预测训练的LLMs会对原始分子和负向破坏的分子分配相似的概率分数，这暴露出它们缺乏对可靠且通用的分子LLMs至关重要的分子结构理解。为克服这一局限性并获得真正的通用分子LLM，我们引入了基于全面的多模态指令调优以及选择和拒绝图之间的分子结构偏好优化的新型多模态训练方法。在多种分子基准测试中，所提出的通用分子LLM（称为Mol-LLM）在大多数任务中取得了最先进的性能，同时超越或与最先进的专门分子LLMs相当。此外，Mol-LLM还在反应预测任务中表现出优越的泛化性能，这表明了从泛化角度来看，对分子结构理解的影响。 

---
# Classroom Simulacra: Building Contextual Student Generative Agents in Online Education for Learning Behavioral Simulation 

**Title (ZH)**: 教室仿真：构建在线教育中用于学习行为模拟的学生生成代理节点 

**Authors**: Songlin Xu, Hao-Ning Wen, Hongyi Pan, Dallas Dominguez, Dongyin Hu, Xinyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02780)  

**Abstract**: Student simulation supports educators to improve teaching by interacting with virtual students. However, most existing approaches ignore the modulation effects of course materials because of two challenges: the lack of datasets with granularly annotated course materials, and the limitation of existing simulation models in processing extremely long textual data. To solve the challenges, we first run a 6-week education workshop from N = 60 students to collect fine-grained data using a custom built online education system, which logs students' learning behaviors as they interact with lecture materials over time. Second, we propose a transferable iterative reflection (TIR) module that augments both prompting-based and finetuning-based large language models (LLMs) for simulating learning behaviors. Our comprehensive experiments show that TIR enables the LLMs to perform more accurate student simulation than classical deep learning models, even with limited demonstration data. Our TIR approach better captures the granular dynamism of learning performance and inter-student correlations in classrooms, paving the way towards a ''digital twin'' for online education. 

**Abstract (ZH)**: 学生模拟支持教育工作者通过与虚拟学生互动来改进教学。然而，由于两个挑战的存在，大多数现有的方法忽视了课程材料的调节作用：缺乏细粒度标注的课程材料数据集，以及现有模拟模型在处理极其长的文本数据方面的局限性。为了克服这些挑战，我们首先从60名学生那里运行了一个为期6周的教育工作坊，使用自建的在线教育系统收集细粒度数据，该系统记录了学生在时间上与讲义材料互动时的学习行为。其次，我们提出了一种可迁移的迭代反思（TIR）模块，该模块增强了基于提示和基于微调的大规模语言模型（LLMs），用于模拟学习行为。我们的全面实验表明，TIR使LLMs在有限的示范数据下，比传统的深度学习模型更能够进行准确的学生模拟。我们的TIR方法更好地捕捉了课堂中学习表现和学生间相关性的细粒度动态性，为在线教育实现“数字孪生”创造了可能性。 

---
# e-SimFT: Alignment of Generative Models with Simulation Feedback for Pareto-Front Design Exploration 

**Title (ZH)**: e-SimFT：基于仿真反馈的生成模型对齐方法，用于帕累托前沿设计探索 

**Authors**: Hyunmin Cheong, Mohammadmehdi Ataei, Amir Hosein Khasahmadi, Pradeep Kumar Jayaraman  

**Link**: [PDF](https://arxiv.org/pdf/2502.02628)  

**Abstract**: Deep generative models have recently shown success in solving complex engineering design problems where models predict solutions that address the design requirements specified as input. However, there remains a challenge in aligning such models for effective design exploration. For many design problems, finding a solution that meets all the requirements is infeasible. In such a case, engineers prefer to obtain a set of Pareto optimal solutions with respect to those requirements, but uniform sampling of generative models may not yield a useful Pareto front. To address this gap, we introduce a new framework for Pareto-front design exploration with simulation fine-tuned generative models. First, the framework adopts preference alignment methods developed for Large Language Models (LLMs) and showcases the first application in fine-tuning a generative model for engineering design. The important distinction here is that we use a simulator instead of humans to provide accurate and scalable feedback. Next, we propose epsilon-sampling, inspired by the epsilon-constraint method used for Pareto-front generation with classical optimization algorithms, to construct a high-quality Pareto front with the fine-tuned models. Our framework, named e-SimFT, is shown to produce better-quality Pareto fronts than existing multi-objective alignment methods. 

**Abstract (ZH)**: 深度生成模型近年来在解决复杂的工程设计问题上取得了成功，模型可以根据输入的设计要求预测出相应的解决方案。然而，在将这些模型用于有效的设计探索时仍然存在挑战。对于许多设计问题，找到满足所有要求的解决方案可能不可行。在这种情况下，工程师更倾向于获得一组与这些要求相关的帕累托最优解，但均匀采样生成模型可能无法生成有用的帕累托前沿。为了解决这一问题，我们提出了一种新的框架，用于使用经过模拟调优的生成模型进行帕累托前沿设计探索。首先，该框架采用为大型语言模型（LLMs）开发的偏好对齐方法，并展示了首个将其应用于调优生成模型进行工程设计的应用实例。关键区别在于，我们使用模拟器而不是人类来提供准确且可扩展的反馈。其次，我们提出了一种基于ε-约束方法的ε采样方法，该方法用于经典优化算法生成帕累托前沿，以通过调优后的模型构建高质量的帕累托前沿。我们所提出的框架名为e-SimFT，已被证明可以产生比现有多目标对齐方法更好的帕累托前沿。 

---
# Think or Step-by-Step? UnZIPping the Black Box in Zero-Shot Prompts 

**Title (ZH)**: 思考还是逐步推理？解开零样本提示黑箱之谜 

**Authors**: Nikta Gohari Sadr, Sangmitra Madhusudan, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2502.03418)  

**Abstract**: Zero-shot prompting techniques have significantly improved the performance of Large Language Models (LLMs). However, we lack a clear understanding of why zero-shot prompts are so effective. For example, in the prompt "Let's think step-by-step," is "think" or "step-by-step" more crucial to its success? Existing interpretability methods, such as gradient-based and attention-based approaches, are computationally intensive and restricted to open-source models. We introduce the ZIP score (Zero-shot Importance of Perturbation score), a versatile metric applicable to both open and closed-source models, based on systematic input word perturbations. Our experiments across four recent LLMs, seven widely-used prompts, and several tasks, reveal interesting patterns in word importance. For instance, while both 'step-by-step' and 'think' show high ZIP scores, which one is more influential depends on the model and task. We validate our method using controlled experiments and compare our results with human judgments, finding that proprietary models align more closely with human intuition regarding word significance. These findings enhance our understanding of LLM behavior and contribute to developing more effective zero-shot prompts and improved model analysis. 

**Abstract (ZH)**: 零样本提示技术显著提高了大型语言模型（LLMs）的性能。然而，我们对零样本提示为何如此有效缺乏清晰的理解。例如，在提示“让我们一步一步地思考”中，“思考”还是“一步一步地”更为关键？现有的可解释性方法，如梯度基和注意力基方法，计算量大且仅限于开源模型。我们引入了ZIP分数（Zero-shot Importance of Perturbation分数），这是一种适用于开源和闭源模型的通用度量标准，基于系统的输入词扰动。我们的实验跨越了四个最新的LLMs、七个广泛使用的提示以及多个任务，揭示了一些有趣的重要词模式。例如，虽然“step-by-step”和“think”都显示出高ZIP分数，但哪一个更具影响力取决于模型和任务。我们使用受控实验验证了我们的方法，并将我们的结果与人类判断进行了比较，发现专有模型在词汇重要性方面与人类直觉更为一致。这些发现增强了我们对LLM行为的理解，并为开发更有效的零样本提示和改进模型分析做出了贡献。 

---
# Demystifying Long Chain-of-Thought Reasoning in LLMs 

**Title (ZH)**: 揭开大型语言模型中长链条推理的奥秘 

**Authors**: Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, Xiang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.03373)  

**Abstract**: Scaling inference compute enhances reasoning in large language models (LLMs), with long chains-of-thought (CoTs) enabling strategies like backtracking and error correction. Reinforcement learning (RL) has emerged as a crucial method for developing these capabilities, yet the conditions under which long CoTs emerge remain unclear, and RL training requires careful design choices. In this study, we systematically investigate the mechanics of long CoT reasoning, identifying the key factors that enable models to generate long CoT trajectories. Through extensive supervised fine-tuning (SFT) and RL experiments, we present four main findings: (1) While SFT is not strictly necessary, it simplifies training and improves efficiency; (2) Reasoning capabilities tend to emerge with increased training compute, but their development is not guaranteed, making reward shaping crucial for stabilizing CoT length growth; (3) Scaling verifiable reward signals is critical for RL. We find that leveraging noisy, web-extracted solutions with filtering mechanisms shows strong potential, particularly for out-of-distribution (OOD) tasks such as STEM reasoning; and (4) Core abilities like error correction are inherently present in base models, but incentivizing these skills effectively for complex tasks via RL demands significant compute, and measuring their emergence requires a nuanced approach. These insights provide practical guidance for optimizing training strategies to enhance long CoT reasoning in LLMs. Our code is available at: this https URL. 

**Abstract (ZH)**: 扩展推理计算可以增强大型语言模型（LLMs）的推理能力，而具有长链思考（长CoT）的策略可以支持回溯和错误纠正等方法。强化学习（RL）已成为开发这些能力的关键方法，然而长CoT如何出现的具体条件仍然不清楚，且RL训练需要精心的设计选择。本研究系统地探讨了长CoT推理的机制，确定了使模型能够生成长CoT轨迹的关键因素。通过广泛的监督微调（SFT）和RL实验，我们提出了四项主要发现：（1）虽然SFT不是严格必要的，但它简化了训练并提高了效率；（2）推理能力倾向于随着训练计算量的增加而出现，但其发展并非必然，因此合理的奖赏塑造对于稳定CoT长度增长至关重要；（3）扩展可验证的奖赏信号对RL至关重要。我们发现利用过滤机制下的噪声、从网络提取的解决方案显示出强大的潜力，尤其是在STEM推理等分布外（OOD）任务中；（4）核心能力如错误纠正在基础模型中固有存在，但通过RL有效地激励这些技能以应对复杂任务需要大量计算资源，而且衡量其出现的方式也需精细化。

这些见解为优化训练策略以增强LLMs中的长CoT推理提供了实用指导。我们的代码可在以下链接获取：this https URL。 

---
# Minerva: A Programmable Memory Test Benchmark for Language Models 

**Title (ZH)**: Minerva：一种用于语言模型的可编程内存测试基准 

**Authors**: Menglin Xia, Victor Ruehle, Saravan Rajmohan, Reza Shokri  

**Link**: [PDF](https://arxiv.org/pdf/2502.03358)  

**Abstract**: How effectively can LLM-based AI assistants utilize their memory (context) to perform various tasks? Traditional data benchmarks, which are often manually crafted, suffer from several limitations: they are static, susceptible to overfitting, difficult to interpret, and lack actionable insights--failing to pinpoint the specific capabilities a model lacks when it does not pass a test. In this paper, we present a framework for automatically generating a comprehensive set of tests to evaluate models' abilities to use their memory effectively. Our framework extends the range of capability tests beyond the commonly explored (passkey, key-value, needle in the haystack) search, a dominant focus in the literature. Specifically, we evaluate models on atomic tasks such as searching, recalling, editing, matching, comparing information in context memory, and performing basic operations when inputs are structured into distinct blocks, simulating real-world data. Additionally, we design composite tests to investigate the models' ability to maintain state while operating on memory. Our benchmark enables an interpretable, detailed assessment of memory capabilities of LLMs. 

**Abstract (ZH)**: 基于LLM的AI助手能够有效地利用其记忆（上下文）来执行各种任务吗？传统的数据基准通常是由人工构建的，存在若干局限性：它们是静态的、容易过拟合、难以解释，并且缺乏可操作的洞察——无法准确指出一个模型在测试中未通过时缺少的具体能力。在本文中，我们提出了一种自动生成全面测试集的框架，用于评估模型有效利用其记忆的能力。我们的框架将能力测试的范围扩展到超越文献中通常探索的（如密钥、键值对、大海捞针）搜索任务。具体而言，我们评估模型在原子任务（如搜索、回忆、编辑、根据上下文记忆匹配信息、以及结构化输入时执行基本操作）上的表现，模拟实际数据。此外，我们设计了复合测试来研究模型在操作记忆时维持状态的能力。我们的基准测试能够实现对LLM记忆能力的可解释和详细的评估。 

---
# Out-of-Distribution Detection using Synthetic Data Generation 

**Title (ZH)**: 使用合成数据生成进行域外检测 

**Authors**: Momin Abbas, Muneeza Azmat, Raya Horesh, Mikhail Yurochkin  

**Link**: [PDF](https://arxiv.org/pdf/2502.03323)  

**Abstract**: Distinguishing in- and out-of-distribution (OOD) inputs is crucial for reliable deployment of classification systems. However, OOD data is typically unavailable or difficult to collect, posing a significant challenge for accurate OOD detection. In this work, we present a method that harnesses the generative capabilities of Large Language Models (LLMs) to create high-quality synthetic OOD proxies, eliminating the dependency on any external OOD data source. We study the efficacy of our method on classical text classification tasks such as toxicity detection and sentiment classification as well as classification tasks arising in LLM development and deployment, such as training a reward model for RLHF and detecting misaligned generations. Extensive experiments on nine InD-OOD dataset pairs and various model sizes show that our approach dramatically lowers false positive rates (achieving a perfect zero in some cases) while maintaining high accuracy on in-distribution tasks, outperforming baseline methods by a significant margin. 

**Abstract (ZH)**: 可靠部署分类系统的关键在于区分分布内（In-Distribution, InD）和分布外（Out-of-Distribution, OOD）输入。然而，OOD数据通常难以获取或难以收集，这为准确的OOD检测带来了重大挑战。在本文中，我们提出了一种方法，利用大型语言模型（LLM）的生成能力创建高质量的合成OOD代理，从而消除对外部OOD数据源的依赖。我们在传统的文本分类任务（如毒性检测和情感分类）以及LLM开发和部署中产生的分类任务（如为RLHF训练奖励模型和检测对齐偏差生成）上研究了该方法的有效性。在九对InD-OOD数据集和各种模型规模的广泛实验中，我们的方法显著降低了假阳性率（在某些情况下实现完美的零假阳性），同时在分布内任务上保持了高精度，超过了基线方法的显著幅度。 

---
# Teaching Large Language Models Number-Focused Headline Generation With Key Element Rationales 

**Title (ZH)**: 用关键元素理由指导大规模语言模型生成以数字为重点的新闻标题 

**Authors**: Zhen Qian, Xiuzhen Zhang, Xiaofei Xu, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2502.03129)  

**Abstract**: Number-focused headline generation is a summarization task requiring both high textual quality and precise numerical accuracy, which poses a unique challenge for Large Language Models (LLMs). Existing studies in the literature focus only on either textual quality or numerical reasoning and thus are inadequate to address this challenge. In this paper, we propose a novel chain-of-thought framework for using rationales comprising key elements of the Topic, Entities, and Numerical reasoning (TEN) in news articles to enhance the capability for LLMs to generate topic-aligned high-quality texts with precise numerical accuracy. Specifically, a teacher LLM is employed to generate TEN rationales as supervision data, which are then used to teach and fine-tune a student LLM. Our approach teaches the student LLM automatic generation of rationales with enhanced capability for numerical reasoning and topic-aligned numerical headline generation. Experiments show that our approach achieves superior performance in both textual quality and numerical accuracy. 

**Abstract (ZH)**: 以下是对给定内容的中文翻译，符合学术规范：

基于数字的标题生成是一项要求极高文本质量和精确数字准确性的摘要任务，这为大型语言模型（LLMs）带来了独特的挑战。现有文献中的研究仅侧重于文本质量或数值推理中的一个方面，因此无法有效应对这一挑战。本文提出了一种新颖的链式思考框架，该框架利用新闻文章中与主题（Topic）、实体（Entities）和数值推理（Numerical reasoning）相关的理由（TEN），增强大型语言模型生成与主题一致、高质量且精确数值的文本的能力。具体而言，使用具有较强数值推理能力的教师语言模型生成TEN理由作为监督数据，然后利用这些数据训练和微调学生模型。我们的方法能够使学生模型自动生成理由，并增强其数值推理能力和主题一致的数值标题生成能力。实验结果表明，我们的方法在文本质量和数值准确性方面均取得了优越表现。 

---
# IAO Prompting: Making Knowledge Flow Explicit in LLMs through Structured Reasoning Templates 

**Title (ZH)**: IAO提示：通过结构化推理模板使知识流动在大型语言模型中变得明确 

**Authors**: Aissatou Diallo, Antonis Bikakis, Luke Dickens, Anthony Hunter, Rob Miller  

**Link**: [PDF](https://arxiv.org/pdf/2502.03080)  

**Abstract**: While Large Language Models (LLMs) demonstrate impressive reasoning capabilities, understanding and validating their knowledge utilization remains challenging. Chain-of-thought (CoT) prompting partially addresses this by revealing intermediate reasoning steps, but the knowledge flow and application remain implicit. We introduce IAO (Input-Action-Output) prompting, a structured template-based method that explicitly models how LLMs access and apply their knowledge during complex reasoning tasks. IAO decomposes problems into sequential steps, each clearly identifying the input knowledge being used, the action being performed, and the resulting output. This structured decomposition enables us to trace knowledge flow, verify factual consistency, and identify potential knowledge gaps or misapplications. Through experiments across diverse reasoning tasks, we demonstrate that IAO not only improves zero-shot performance but also provides transparency in how LLMs leverage their stored knowledge. Human evaluation confirms that this structured approach enhances our ability to verify knowledge utilization and detect potential hallucinations or reasoning errors. Our findings provide insights into both knowledge representation within LLMs and methods for more reliable knowledge application. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）展示了令人印象深刻的推理能力，但理解和验证其知识利用仍然具有挑战性。思维链（CoT）提示部分解决了这一问题，通过揭示中间的推理步骤来展示模型的思维过程，但知识流和应用仍然较为隐含。我们引入了一种名为IAO（输入-动作-输出）提示的方法，这是一种结构化模板方法，明确地建模了LLMs在复杂推理任务中是如何获取和应用知识的。IAO将问题分解为一系列顺序步骤，每个步骤都明确标识出所使用的输入知识、所执行的动作以及由此产生的结果。这种结构化分解使我们能够追踪知识流、验证事实一致性，并识别潜在的知识空白或误用。通过跨各种推理任务的实验，我们证明IAO不仅提高了零样本性能，还提升了我们理解LLMs如何利用其存储知识的透明度。人类评估证实，这种结构化方法增强了我们验证知识利用能力和检测潜在幻觉或推理错误的能力。我们的研究结果为了解LLMs中的知识表示以及更可靠的知识应用方法提供了见解。 

---
# Position: Editing Large Language Models Poses Serious Safety Risks 

**Title (ZH)**: 位置：编辑大型语言模型存在严重的安全风险 

**Authors**: Paul Youssef, Zhixue Zhao, Daniel Braun, Jörg Schlötterer, Christin Seifert  

**Link**: [PDF](https://arxiv.org/pdf/2502.02958)  

**Abstract**: Large Language Models (LLMs) contain large amounts of facts about the world. These facts can become outdated over time, which has led to the development of knowledge editing methods (KEs) that can change specific facts in LLMs with limited side effects. This position paper argues that editing LLMs poses serious safety risks that have been largely overlooked. First, we note the fact that KEs are widely available, computationally inexpensive, highly performant, and stealthy makes them an attractive tool for malicious actors. Second, we discuss malicious use cases of KEs, showing how KEs can be easily adapted for a variety of malicious purposes. Third, we highlight vulnerabilities in the AI ecosystem that allow unrestricted uploading and downloading of updated models without verification. Fourth, we argue that a lack of social and institutional awareness exacerbates this risk, and discuss the implications for different stakeholders. We call on the community to (i) research tamper-resistant models and countermeasures against malicious model editing, and (ii) actively engage in securing the AI ecosystem. 

**Abstract (ZH)**: 大型语言模型（LLMs）包含了大量关于世界的事实。这些事实可能会随着时间的推移变得过时，这导致了知识编辑方法（KEs）的发展，这些方法可以在LLMs中改变特定的事实，并具有有限的副作用。本文观点认为，编辑LLMs带来了严重安全隐患，这一问题并未得到充分关注。首先，我们指出KEs广泛可用、计算成本低廉、性能强大且隐蔽的特点，使它们成为恶意行为者青睐的工具。第二，我们讨论了KEs的恶意使用案例，展示了如何轻松将其改编用于多种恶意目的。第三，我们突出了AI生态系统中的漏洞，这些漏洞允许未经验证就上传和下载更新后的模型。第四，我们指出缺乏社会和机构意识进一步加剧了这一风险，并讨论了不同利益相关者的潜在影响。我们呼吁社区（i）研究防篡改模型和对抗恶意模型编辑的对策，并（ii）积极参与保护AI生态系统的安全。 

---
# Lowering the Barrier of Machine Learning: Achieving Zero Manual Labeling in Review Classification Using LLMs 

**Title (ZH)**: 降低机器学习的门槛：利用大语言模型实现评论分类中的零人工标注 

**Authors**: Yejian Zhang, Shingo Takada  

**Link**: [PDF](https://arxiv.org/pdf/2502.02893)  

**Abstract**: With the internet's evolution, consumers increasingly rely on online reviews for service or product choices, necessitating that businesses analyze extensive customer feedback to enhance their offerings. While machine learning-based sentiment classification shows promise in this realm, its technical complexity often bars small businesses and individuals from leveraging such advancements, which may end up making the competitive gap between small and large businesses even bigger in terms of improving customer satisfaction. This paper introduces an approach that integrates large language models (LLMs), specifically Generative Pre-trained Transformer (GPT) and Bidirectional Encoder Representations from Transformers (BERT)-based models, making it accessible to a wider audience. Our experiments across various datasets confirm that our approach retains high classification accuracy without the need for manual labeling, expert knowledge in tuning and data annotation, or substantial computational power. By significantly lowering the barriers to applying sentiment classification techniques, our methodology enhances competitiveness and paves the way for making machine learning technology accessible to a broader audience. 

**Abstract (ZH)**: 随着互联网的发展，消费者越来越多地依赖在线评价来选择服务或产品，这促使企业需要分析大量的客户反馈以提升其产品和服务。虽然基于机器学习的情绪分类在这一领域展现出巨大的潜力，但由于技术复杂性，小型企业和个人往往无法利用这些进展。这可能会加剧小型企业和大型企业在提升客户满意度方面的竞争差距。本文提出了一种方法，该方法整合了大型语言模型（LLMs），具体包括生成型预训练Transformer（GPT）和双向编码表示Transformer（BERT）模型，使其能够更加广泛地应用。我们对多个数据集的实验结果表明，该方法在保持高分类准确性的前提下，无需人工标注、专家调优和数据注释，也无需大量计算资源。通过显著降低应用情绪分类技术的门槛，我们的方法提高了竞争力，并为使机器学习技术更广泛地应用铺平了道路。 

---
# CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration 

**Title (ZH)**: CAMI：一种通过状态推断和主题探索支持动机访谈的咨询代理 

**Authors**: Yizhe Yang, Palakorn Achananuparp, Heyan Huang, Jing Jiang, Kit Phey Leng, Nicholas Gabriel Lim, Cameron Tan Shi Ern, Ee-peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.02807)  

**Abstract**: Conversational counselor agents have become essential tools for addressing the rising demand for scalable and accessible mental health support. This paper introduces CAMI, a novel automated counselor agent grounded in Motivational Interviewing (MI) -- a client-centered counseling approach designed to address ambivalence and facilitate behavior change. CAMI employs a novel STAR framework, consisting of client's state inference, motivation topic exploration, and response generation modules, leveraging large language models (LLMs). These components work together to evoke change talk, aligning with MI principles and improving counseling outcomes for clients from diverse backgrounds. We evaluate CAMI's performance through both automated and manual evaluations, utilizing simulated clients to assess MI skill competency, client's state inference accuracy, topic exploration proficiency, and overall counseling success. Results show that CAMI not only outperforms several state-of-the-art methods but also shows more realistic counselor-like behavior. Additionally, our ablation study underscores the critical roles of state inference and topic exploration in achieving this performance. 

**Abstract (ZH)**: 会话咨询代理已成为应对不断增长的可扩展和易获取心理健康支持需求的重要工具。本文介绍了CAMI，这是一种基于动机访谈（MI）的新型自动化咨询代理——动机访谈是一种以客户为中心的咨询方法，旨在解决客户的犹疑和促进行为改变。CAMI采用了一种创新的STAR框架，包括客户端状态推理、动机话题探索和响应生成模块，利用大规模语言模型（LLM）。这些组件共同作用以诱发改变对话，符合动机访谈的原则，并改善来自不同背景客户的咨询服务效果。我们通过自动评估和人工评估来评估CAMI的性能，利用模拟客户评估其动机访谈技能、客户状态推理准确性、话题探索能力以及整体咨询成效。研究结果显示，CAMI不仅优于多种最先进的方法，还表现出更接近人类咨询师的行为。此外，我们的消融研究进一步强调了状态推理和话题探索在实现这一性能中的关键作用。 

---
# Transformers Boost the Performance of Decision Trees on Tabular Data across Sample Sizes 

**Title (ZH)**: Transformer模型在不同样本规模下提升了表格数据上决策树的性能 

**Authors**: Mayuka Jayawardhana, Renbo Tu, Samuel Dooley, Valeriia Cherepanova, Andrew Gordon Wilson, Frank Hutter, Colin White, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2502.02672)  

**Abstract**: Large language models (LLMs) perform remarkably well on tabular datasets in zero- and few-shot settings, since they can extract meaning from natural language column headers that describe features and labels. Similarly, TabPFN, a recent non-LLM transformer pretrained on numerous tables for in-context learning, has demonstrated excellent performance for dataset sizes up to a thousand samples. In contrast, gradient-boosted decision trees (GBDTs) are typically trained from scratch on each dataset without benefiting from pretraining data and must learn the relationships between columns from their entries alone since they lack natural language understanding. LLMs and TabPFN excel on small tabular datasets where a strong prior is essential, yet they are not competitive with GBDTs on medium or large datasets, since their context lengths are limited. In this paper, we propose a simple and lightweight approach for fusing large language models and TabPFN with gradient-boosted decision trees, which allows scalable GBDTs to benefit from the natural language capabilities and pretraining of transformers. We name our fusion methods LLM-Boost and PFN-Boost, respectively. While matching or surpassing the performance of the transformer at sufficiently small dataset sizes and GBDTs at sufficiently large sizes, LLM-Boost and PFN-Boost outperform both standalone components on a wide range of dataset sizes in between. We demonstrate state-of-the-art performance against numerous baselines and ensembling algorithms. We find that PFN-Boost achieves the best average performance among all methods we test for all but very small dataset sizes. We release our code at this http URL . 

**Abstract (ZH)**: 以下是经过学术规范翻译后的论文内容或标题：

大规模语言模型（LLMs）在零样本和少样本设置中对表格数据集表现出色，因为它们可以从描述特征和标签的自然语言列头中提取意义。类似地，TabPFN 是一种近期的非 LLM 转换器，在预训练了大量的表格数据后，对于样本量多达一千个的数据集展现了优秀的性能。相比之下，梯度提升决策树（GBDTs）通常需要从头开始训练每个数据集，不能从预训练数据中受益，并且由于缺乏自然语言理解能力，只能通过学习表格项之间的关系来学习列之间的关系。LLMs 和 TabPFN 在小的表格数据集上表现出色，这些数据集需要强烈先验知识，但当数据集规模中等或较大时，它们在与 GBDTs 的竞争中并不具备优势，因为它们的上下文长度是有限的。在本文中，我们提出了一种简单而轻量级的方法，用于将大型语言模型、TabPFN 与梯度提升决策树融合，从而使可扩展的 GBDTs 能够利用转换器的自然语言能力和预训练。我们分别将这两种融合方法命名为 LLM-Boost 和 PFN-Boost。在足够小的数据集规模下，LLM-Boost 和 PFN-Boost 的性能与转换器相当或超过转换器；在足够大的数据集规模下，其性能与 GBDTs 相当或超过 GBDTs。在各种规模的数据集上，LLM-Boost 和 PFN-Boost 在大多数情况下均优于各自的独立组件。我们与多个基准和集成算法进行了对比实验，并展示了最先进的性能表现。我们发现，在除非常小的数据集规模之外的所有测试方法中，PFN-Boost 在所有方法中表现最佳。我们已将我们的代码发布在以下网址：[链接]。 

---
# Do Large Language Model Benchmarks Test Reliability? 

**Title (ZH)**: 大型语言模型基准测试能否衡量可靠性？ 

**Authors**: Joshua Vendrow, Edward Vendrow, Sara Beery, Aleksander Madry  

**Link**: [PDF](https://arxiv.org/pdf/2502.03461)  

**Abstract**: When deploying large language models (LLMs), it is important to ensure that these models are not only capable, but also reliable. Many benchmarks have been created to track LLMs' growing capabilities, however there has been no similar focus on measuring their reliability. To understand the potential ramifications of this gap, we investigate how well current benchmarks quantify model reliability. We find that pervasive label errors can compromise these evaluations, obscuring lingering model failures and hiding unreliable behavior.
Motivated by this gap in the evaluation of reliability, we then propose the concept of so-called platinum benchmarks, i.e., benchmarks carefully curated to minimize label errors and ambiguity. As a first attempt at constructing such benchmarks, we revise examples from fifteen existing popular benchmarks. We evaluate a wide range of models on these platinum benchmarks and find that, indeed, frontier LLMs still exhibit failures on simple tasks such as elementary-level math word problems. Analyzing these failures further reveals previously unidentified patterns of problems on which frontier models consistently struggle. We provide code at this https URL 

**Abstract (ZH)**: 在部署大型语言模型（LLMs）时，确保这些模型不仅功能强大，而且可靠也非常关键。虽然已经创建了许多基准来跟踪LLMs的能力增长，但迄今为止没有类似的焦点放在衡量其可靠性的方面。为了理解这一差距的潜在影响，我们调查了当前基准在衡量模型可靠性方面的有效性。我们发现，普遍的标签错误可能会破坏这些评估，使得持续存在的模型失败和不可靠行为变得隐匿。

鉴于在可靠性评估方面存在的这一差距，我们随后提出了所谓的铂金基准的概念，即精心筛选并编程的基准，旨在最小化标签错误和模糊性。我们首先尝试构建此类基准，对十五个现有流行基准中的一些示例进行了修订。我们在这些铂金基准上评估了多种模型，并发现前沿的LLMs仍然在诸如基础数学文字问题等简单任务上表现出失败。进一步分析这些失败揭示了前沿模型在某些问题上一贯存在的未识别困难模式。代码可在以下链接获得：[提供代码的链接] 

---
# Leveraging the true depth of LLMs 

**Title (ZH)**: 利用大型语言模型的真正深度 

**Authors**: Ramón Calvo González, Daniele Paliotta, Matteo Pagliardini, Martin Jaggi, François Fleuret  

**Link**: [PDF](https://arxiv.org/pdf/2502.02790)  

**Abstract**: Large Language Models demonstrate remarkable capabilities at the cost of high compute requirements. While recent research has shown that intermediate layers can be removed or have their order shuffled without impacting performance significantly, these findings have not been employed to reduce the computational cost of inference. We investigate several potential ways to reduce the depth of pre-trained LLMs without significantly affecting performance. Leveraging our insights, we present a novel approach that exploits this decoupling between layers by grouping some of them into pairs that can be evaluated in parallel.
This modification of the computational graph -- through better parallelism -- results in an average improvement of around 1.20x on the number of tokens generated per second, without re-training nor fine-tuning, while retaining 95%-99% of the original accuracy. Empirical evaluation demonstrates that this approach significantly improves serving efficiency while maintaining model performance, offering a practical improvement for large-scale LLM deployment. 

**Abstract (ZH)**: 大规模语言模型在高计算需求的代价下展示了显著的能力。虽然最近的研究表明中间层可以被移除或重新排序而不显著影响性能，但这些发现尚未被应用于降低推理的计算成本。我们探讨了几种可能的方法，在不显著影响性能的前提下减少预训练语言模型的深度。利用我们的见解，我们提出了一种新颖的方法，通过将一些层分组，形成可以并行评估的对来利用层之间的分离。

通过这种计算图的修改——通过更好的并行性——在不重新训练或微调的情况下，我们获得了大约1.20倍的每秒生成token数的平均改进，同时保留了原始准确度的95%-99%。实证评估表明，这种方法在保持模型性能的同时显著提高了服务效率，为大规模语言模型的部署提供了实用的改进。 

---
# Intent Representation Learning with Large Language Model for Recommendation 

**Title (ZH)**: 使用大型语言模型进行意图表示学习的推荐方法 

**Authors**: Yu Wang, Lei Sang, Yi Zhang, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.03307)  

**Abstract**: Intent-based recommender systems have garnered significant attention for uncovering latent fine-grained preferences. Intents, as underlying factors of interactions, are crucial for improving recommendation interpretability. Most methods define intents as learnable parameters updated alongside interactions. However, existing frameworks often overlook textual information (e.g., user reviews, item descriptions), which is crucial for alleviating the sparsity of interaction intents. Exploring these multimodal intents, especially the inherent differences in representation spaces, poses two key challenges: i) How to align multimodal intents and effectively mitigate noise issues; ii) How to extract and match latent key intents across modalities. To tackle these challenges, we propose a model-agnostic framework, Intent Representation Learning with Large Language Model (IRLLRec), which leverages large language models (LLMs) to construct multimodal intents and enhance recommendations. Specifically, IRLLRec employs a dual-tower architecture to learn multimodal intent representations. Next, we propose pairwise and translation alignment to eliminate inter-modal differences and enhance robustness against noisy input features. Finally, to better match textual and interaction-based intents, we employ momentum distillation to perform teacher-student learning on fused intent representations. Empirical evaluations on three datasets show that our IRLLRec framework outperforms baselines. The implementation is available at this https URL. 

**Abstract (ZH)**: 基于意图的推荐系统因其在揭示隐含的精细偏好方面取得显著成效而受到广泛关注。意图作为交互的基础因素，对于提高推荐的可解释性至关重要。大多数方法将意图定义为在交互过程中可学习的参数。然而，现有框架往往忽视了文本信息（例如用户评论和商品描述），这些信息对于缓解交互意图稀疏性至关重要。探索这些多模态意图，特别是它们在表示空间中的固有差异，面临两个关键挑战：i) 如何对齐多模态意图并有效减轻噪声问题；ii) 如何提取和匹配各模态中的潜在关键意图。为了解决这些挑战，我们提出了一种模型无关的框架，即基于大型语言模型的意图表示学习（IRLLRec），该框架利用大型语言模型（LLMs）构建多模态意图并增强推荐效果。具体而言，IRLLRec 采用双重塔结构学习多模态意图表示。接下来，我们提出了成对和翻译对齐，以消除不同模态之间的差异并增强对噪声输入特征的鲁棒性。最后，为了更好地匹配基于文本和交互的意图，我们利用动量蒸馏技术，在融合意图表示上进行教师-学生学习。在三个数据集上的实证评估表明，我们的IRLLRec框架优于基准模型。开源代码可在以下链接获取：this https URL。 

---
